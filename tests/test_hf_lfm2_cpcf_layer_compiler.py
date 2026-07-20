from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
import neural_forge.hf_lfm2_structural_residual_attention_fold as structural_fold  # noqa: E402
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
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (  # noqa: E402
    Lfm2CompiledCpcfShortConvV4Fold,
    Lfm2CpcfShortConvV4FoldCase,
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
        direction.square().mean(dim=-1, keepdim=True) + float(module.variance_epsilon)
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
        "residual.post_layer": (post_operator + down)[:, -sampled:, :].clone(),
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
    additive = (
        torch.where(
            allowed,
            torch.zeros((), dtype=residual.dtype),
            torch.full((), torch.finfo(residual.dtype).min, dtype=residual.dtype),
        )
        .reshape(1, 1, sampled, sequence)
        .expand(batch, -1, -1, -1)
    )
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
            "residual.post_operator": full_post_operator[:, -sampled:, :].clone(),
            "attention.q_linear": q_linear_heads[:, -sampled:, :, :]
            .transpose(1, 2)
            .contiguous(),
            "attention.q": query[:, :, -sampled:, :].clone(),
            "attention.k_linear": k_linear_heads[:, -sampled:, :, :]
            .transpose(1, 2)
            .contiguous(),
            "attention.k": key[:, :, -sampled:, :].clone(),
            "attention.v": value[:, :, -sampled:, :].clone(),
            "attention.context": context.clone(),
            "attention.o": output.clone(),
            "attention.q_normalized_pre_rope": q_norm[:, :, -sampled:, :].clone(),
            "attention.k_normalized_pre_rope": k_norm[:, :, -sampled:, :].clone(),
            "attention.rope_cos": cosine[:, -sampled:, :].clone(),
            "attention.rope_sin": sine[:, -sampled:, :].clone(),
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
            # V4 consumes the complete cached parent operator-input history
            # for its affine child-to-parent lift.  The residual endpoint
            # remains the intentionally sampled suffix.
            "operator.input": operator_input.clone(),
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
        sampled = 6 if kind == "full_attention" else 4
        tensors = (
            _attention_evidence(source, residual, sampled)
            if kind == "full_attention"
            else _conv_evidence(source, residual, sampled)
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


def test_decode_short_conv_suffixes_reconstruct_complete_causal_histories() -> None:
    histories = {}
    prefill = SimpleNamespace(
        identity=(4, "prefill"),
        batch_index=4,
        phase="prefill",
        batch_size=1,
        full_prefix_tokens=5,
        sampled_suffix_tokens=5,
    )
    prefill_fields = {
        name: torch.arange(10, dtype=torch.float32).reshape(1, 5, 2) + field_index * 100
        for field_index, name in enumerate(layer_compiler._SHORT_CONV_HISTORY_FIELDS)
    }
    complete_prefill = layer_compiler._extend_short_conv_parent_histories(
        prefill,
        prefill_fields,
        layer_index=0,
        histories=histories,
    )
    assert all(tuple(value.shape) == (1, 5, 2) for value in complete_prefill.values())

    decode_zero = SimpleNamespace(
        identity=(4, "decode.000"),
        batch_index=4,
        phase="decode.000",
        batch_size=1,
        full_prefix_tokens=6,
        sampled_suffix_tokens=1,
    )
    decode_fields = {
        name: torch.full((1, 1, 2), 1000.0 + field_index)
        for field_index, name in enumerate(layer_compiler._SHORT_CONV_HISTORY_FIELDS)
    }
    decode_fields["short_conv.pre_bx_state"] = (
        prefill_fields["short_conv.b_history"] * prefill_fields["short_conv.x_history"]
    ).transpose(1, 2)[:, :, -3:]
    reconstructed = layer_compiler._extend_short_conv_parent_histories(
        decode_zero,
        decode_fields,
        layer_index=0,
        histories=histories,
    )
    for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS:
        assert tuple(reconstructed[name].shape) == (1, 6, 2)
        assert torch.equal(reconstructed[name][:, :5, :], prefill_fields[name])
        assert torch.equal(reconstructed[name][:, 5:, :], decode_fields[name])

    decode_one = SimpleNamespace(
        identity=(4, "decode.001"),
        batch_index=4,
        phase="decode.001",
        batch_size=1,
        full_prefix_tokens=7,
        sampled_suffix_tokens=1,
    )
    next_fields = {
        name: torch.full((1, 1, 2), 2000.0 + field_index)
        for field_index, name in enumerate(layer_compiler._SHORT_CONV_HISTORY_FIELDS)
    }
    next_fields["short_conv.pre_bx_state"] = (
        reconstructed["short_conv.b_history"] * reconstructed["short_conv.x_history"]
    ).transpose(1, 2)[:, :, -3:]
    reconstructed_next = layer_compiler._extend_short_conv_parent_histories(
        decode_one,
        next_fields,
        layer_index=0,
        histories=histories,
    )
    for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS:
        assert torch.equal(reconstructed_next[name][:, :6, :], reconstructed[name])
        assert torch.equal(reconstructed_next[name][:, 6:, :], next_fields[name])


def test_decode_short_conv_history_rejects_a_missing_causal_step() -> None:
    histories = {}
    prefill = SimpleNamespace(
        identity=(9, "prefill"),
        batch_index=9,
        phase="prefill",
        batch_size=1,
        full_prefix_tokens=3,
        sampled_suffix_tokens=3,
    )
    fields = {
        name: torch.zeros((1, 3, 2), dtype=torch.float32)
        for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS
    }
    layer_compiler._extend_short_conv_parent_histories(
        prefill,
        fields,
        layer_index=0,
        histories=histories,
    )
    skipped_decode = SimpleNamespace(
        identity=(9, "decode.001"),
        batch_index=9,
        phase="decode.001",
        batch_size=1,
        full_prefix_tokens=5,
        sampled_suffix_tokens=1,
    )
    suffix = {
        name: torch.zeros((1, 1, 2), dtype=torch.float32)
        for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS
    }
    suffix["short_conv.pre_bx_state"] = torch.zeros(
        (1, 2, 3),
        dtype=torch.float32,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="causal continuation",
    ):
        layer_compiler._extend_short_conv_parent_histories(
            skipped_decode,
            suffix,
            layer_index=0,
            histories=histories,
        )


def test_short_conv_phases_coalesce_into_one_causal_trajectory_per_record() -> None:
    phases = []
    states = {}
    replays = {}
    for batch_index, split in enumerate(("selection", "fit", "validation")):
        generator = torch.Generator().manual_seed(8800 + batch_index)
        parent_prefill = {
            name: torch.randn(
                (1, 3, 3),
                generator=generator,
                dtype=torch.float32,
            )
            for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS
        }
        parent_prefill["operator.input"] = torch.randn(
            (1, 3, 3),
            generator=generator,
            dtype=torch.float32,
        )
        parent_decode = {
            name: torch.randn(
                (1, 1, 3),
                generator=generator,
                dtype=torch.float32,
            )
            for name in layer_compiler._SHORT_CONV_HISTORY_FIELDS
        }
        parent_decode["operator.input"] = torch.randn(
            (1, 1, 3),
            generator=generator,
            dtype=torch.float32,
        )
        parent_decode["short_conv.pre_bx_state"] = (
            parent_prefill["short_conv.b_history"]
            * parent_prefill["short_conv.x_history"]
        ).transpose(1, 2)[:, :, -2:]
        child_input = torch.randn(
            (1, 4, 2),
            generator=generator,
            dtype=torch.float32,
        )
        child_state = torch.randn(
            (1, 4, 2),
            generator=generator,
            dtype=torch.float32,
        )
        phase_specs = (
            (
                "prefill",
                3,
                2,
                parent_prefill,
                child_input[:, :3, :],
                child_state[:, :3, :],
            ),
            (
                "decode.000",
                4,
                1,
                parent_decode,
                child_input,
                child_state,
            ),
        )
        for phase_name, width, sampled, parent_fields, replay, state in phase_specs:
            tensors = {
                **parent_fields,
                "residual.post_operator": torch.randn(
                    (1, sampled, 2),
                    generator=generator,
                    dtype=torch.float32,
                ),
            }
            evidence = SimpleNamespace(
                layer_index=0,
                operator_kind="short_conv",
                tensors=tensors,
            )
            phase = SimpleNamespace(
                identity=(batch_index, phase_name),
                batch_index=batch_index,
                record_id=f"record-{batch_index}",
                split=split,
                phase=phase_name,
                batch_size=1,
                full_prefix_tokens=width,
                sampled_suffix_tokens=sampled,
                input_ids=torch.arange(width).reshape(1, width),
                attention_mask=torch.ones((1, width), dtype=torch.long),
                position_ids=torch.arange(width).reshape(1, width),
                parent_evidence=SimpleNamespace(layers=(evidence,)),
            )
            phases.append(phase)
            states[phase.identity] = state
            replays[phase.identity] = SimpleNamespace(operator_input=replay)

    class InputBank:
        def __init__(self) -> None:
            self.phases = tuple(phases)

        def state(self, phase: SimpleNamespace) -> object:
            return states[phase.identity].clone()

    cases = layer_compiler._short_conv_cases(
        layer_index=0,
        input_bank=InputBank(),
        preinstall_replay=SimpleNamespace(replays=replays),
        chart=SimpleNamespace(
            post_operator=SimpleNamespace(
                encode_shaped=lambda value: value.clone(),
            )
        ),
    )

    assert len(cases) == 3
    assert tuple(case.split for case in cases) == (
        "selection",
        "fit",
        "validation",
    )
    for case in cases:
        assert ":trajectory:layer.0" in case.case_id
        assert case.suffix_start == 1
        assert tuple(case.actual_child_operator_input_history.shape) == (
            1,
            4,
            2,
        )
        assert tuple(case.parent_b_history.shape) == (1, 4, 3)
        assert tuple(case.parent_post_operator_residual_target_space_suffix.shape) == (
            1,
            3,
            2,
        )
        assert tuple(case.actual_child_pre_operator_residual_suffix.shape) == (
            1,
            3,
            2,
        )
        assert case.verify_authentication() is True

    v4_cases = layer_compiler._short_conv_cases(
        layer_index=0,
        input_bank=InputBank(),
        preinstall_replay=SimpleNamespace(replays=replays),
        chart=SimpleNamespace(
            post_operator=SimpleNamespace(
                encode_shaped=lambda value: value.clone(),
            )
        ),
        use_v4=True,
    )
    assert all(isinstance(case, Lfm2CpcfShortConvV4FoldCase) for case in v4_cases)
    for case in v4_cases:
        assert tuple(case.parent_operator_input_history.shape) == (1, 4, 3)
        assert case.verify_authentication() is True


def test_attention_phases_coalesce_exact_k_linear_chronology(
    monkeypatch,
) -> None:
    source, target = _models("full_attention")
    generator = torch.Generator().manual_seed(2026071704)
    parent_residual = torch.randn(
        (1, 4, 8),
        generator=generator,
        dtype=torch.float32,
    )
    parent_tensors = {
        "prefill": _attention_evidence(
            source,
            parent_residual[:, :3, :],
            3,
        ),
        "decode.000": _attention_evidence(
            source,
            parent_residual,
            1,
        ),
    }
    child_input = torch.randn(
        (1, 4, 4),
        generator=generator,
        dtype=torch.float32,
    )
    child_state = torch.randn(
        (1, 4, 4),
        generator=generator,
        dtype=torch.float32,
    )
    phases = []
    states = {}
    replays = {}
    for phase_name, width, sampled in (
        ("prefill", 3, 3),
        ("decode.000", 4, 1),
    ):
        evidence = SimpleNamespace(
            layer_index=0,
            operator_kind="full_attention",
            tensors=parent_tensors[phase_name],
        )
        phase = SimpleNamespace(
            identity=(0, phase_name),
            batch_index=0,
            record_id="fit-record",
            split="fit",
            phase=phase_name,
            batch_size=1,
            full_prefix_tokens=width,
            sampled_suffix_tokens=sampled,
            input_ids=torch.arange(width).reshape(1, width),
            attention_mask=torch.ones((1, width), dtype=torch.long),
            position_ids=torch.arange(width).reshape(1, width),
            parent_evidence=SimpleNamespace(layers=(evidence,)),
        )
        phases.append(phase)
        states[phase.identity] = child_state[:, :width, :].clone()
        replays[phase.identity] = SimpleNamespace(
            operator_input=child_input[:, :width, :].clone(),
        )

    class InputBank:
        def __init__(self) -> None:
            self.phases = tuple(phases)

        def state(self, phase: SimpleNamespace) -> object:
            return states[phase.identity].clone()

    def forbidden_recovery(**_kwargs):
        raise AssertionError("attention chronology used inverse-RoPE recovery")

    monkeypatch.setattr(
        layer_compiler,
        "recover_lfm2_full_key_linear_history",
        forbidden_recovery,
    )
    charts = _chart("full_attention")
    cases, runtime_receipt = layer_compiler._attention_cases(
        source_model=source,
        target_model=target,
        source_layer=source.model.layers[0],
        layer_index=0,
        input_bank=InputBank(),
        preinstall_replay=SimpleNamespace(replays=replays),
        chart=charts,
    )

    assert len(cases) == 1
    assert runtime_receipt["phase_count"] == 2
    assert runtime_receipt["prepared_composition_count"] == 1
    assert runtime_receipt["avoided_composition_count"] == 2
    assert runtime_receipt["report_sha256"] == _sha(
        {key: value for key, value in runtime_receipt.items() if key != "report_sha256"}
    )
    case = cases[0]
    expected_key_linear = torch.cat(
        (
            parent_tensors["prefill"]["attention.k_linear"],
            parent_tensors["decode.000"]["attention.k_linear"],
        ),
        dim=2,
    )
    expected_query = torch.cat(
        (
            parent_tensors["prefill"]["attention.q"],
            parent_tensors["decode.000"]["attention.q"],
        ),
        dim=2,
    )
    assert torch.equal(case.parent_key_linear, expected_key_linear)
    assert torch.equal(case.parent_query, expected_query)
    assert torch.equal(case.target_query_input, child_input)
    assert torch.equal(case.target_key_value_input, child_input)
    assert torch.equal(case.target_pre_attention_residual, child_state)
    assert torch.equal(
        case.key_mask,
        torch.ones((1, 4, 4), dtype=torch.bool).tril(),
    )
    assert tuple(case.parent_query.shape[2:]) == (4, 2)
    assert tuple(case.parent_key.shape[2:]) == (4, 2)
    assert tuple(case.parent_post_attention_residual_target_space.shape) == (1, 4, 4)
    parent_context = torch.cat(
        (
            parent_tensors["prefill"]["attention.context"],
            parent_tensors["decode.000"]["attention.context"],
        ),
        dim=2,
    ).to(torch.float64)
    output_weight = (
        source.model.layers[0].self_attn.out_proj.weight.detach().to(torch.float64)
    )
    output_blocks = output_weight.reshape(8, 4, 2)
    target_blocks = torch.einsum(
        "ohd,ot->hdt",
        output_blocks,
        charts.post_operator.mapping.encoder,
    )
    expected_contribution = torch.einsum(
        "bhqd,hdt->bhqt",
        parent_context,
        target_blocks,
    )
    assert torch.allclose(
        case.parent_contribution,
        expected_contribution,
        rtol=1e-12,
        atol=1e-12,
    )
    cached_output = torch.cat(
        (
            parent_tensors["prefill"]["attention.o"],
            parent_tensors["decode.000"]["attention.o"],
        ),
        dim=1,
    )
    assert torch.allclose(
        case.parent_contribution.sum(dim=1),
        charts.post_operator.encode_shaped(cached_output),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.allclose(
        case.query_row_sqrt_weights.square(),
        torch.tensor(
            [[2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0]],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.equal(
        case.query_row_sqrt_weights,
        case.key_value_row_sqrt_weights,
    )
    assert float(case.query_row_sqrt_weights.square().sum().item()) == pytest.approx(
        4.0
    )


def test_attention_o_contribution_accepts_bf16_rounding_but_rejects_mismatch() -> None:
    generator = torch.Generator().manual_seed(20260719)
    batch = 2
    query_heads = 3
    query_rows = 5
    head_dim = 2
    source_width = 7
    target_width = 4
    context = (
        torch.rand(
            (batch, query_heads, query_rows, head_dim),
            generator=generator,
            dtype=torch.float32,
        )
        + 0.5
    ).to(torch.bfloat16)
    output_weight = (
        torch.rand(
            (source_width, query_heads * head_dim),
            generator=generator,
            dtype=torch.float32,
        )
        + 0.5
    ).to(torch.bfloat16)
    encoder = torch.randn(
        (source_width, target_width),
        generator=generator,
        dtype=torch.float64,
    )
    output_blocks = torch.einsum(
        "ohd,ot->hdt",
        output_weight.to(torch.float64).reshape(
            source_width,
            query_heads,
            head_dim,
        ),
        encoder,
    )
    source_output = torch.einsum(
        "bhqd,ohd->bqo",
        context.to(torch.float32),
        output_weight.to(torch.float32).reshape(
            source_width,
            query_heads,
            head_dim,
        ),
    ).to(torch.bfloat16)
    target_output = source_output.to(torch.float64) @ encoder

    contribution = layer_compiler._encoded_parent_attention_contribution(
        parent_context=context,
        cached_source_output=source_output,
        cached_target_output=target_output,
        source_output_weight=output_weight,
        target_encoder=encoder,
        target_output_blocks=output_blocks,
        name="bf16-authentic",
    )
    assert tuple(contribution.shape) == (
        batch,
        query_heads,
        query_rows,
        target_width,
    )

    mismatched_source_output = (source_output.to(torch.float32) * 1.25).to(
        torch.bfloat16
    )
    mismatched_target_output = mismatched_source_output.to(torch.float64) @ encoder
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="accumulation-aware rounding bound",
    ):
        layer_compiler._encoded_parent_attention_contribution(
            parent_context=context,
            cached_source_output=mismatched_source_output,
            cached_target_output=mismatched_target_output,
            source_output_weight=output_weight,
            target_encoder=encoder,
            target_output_blocks=output_blocks,
            name="bf16-mismatch",
        )


def test_prepared_attention_o_transport_is_exact_sealed_and_reused() -> None:
    generator = torch.Generator().manual_seed(20260720)
    batch = 2
    query_heads = 3
    query_rows = 4
    head_dim = 2
    source_width = 7
    target_width = 5
    context = torch.randn(
        (batch, query_heads, query_rows, head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    output_weight = torch.randn(
        (source_width, query_heads * head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    encoder = torch.randn(
        (source_width, target_width),
        generator=generator,
        dtype=torch.float64,
    )
    source_output = torch.einsum(
        "bhqd,ohd->bqo",
        context,
        output_weight.reshape(source_width, query_heads, head_dim),
    )
    target_output = source_output @ encoder
    prepared = layer_compiler._prepare_attention_output_transport_from_tensors(
        source_output_weight=output_weight,
        target_encoder=encoder,
        query_heads=query_heads,
        head_dim=head_dim,
        name="sealed-test",
    )
    expected = torch.einsum(
        "bhqd,hdt->bhqt",
        context,
        prepared.target_output_blocks,
    )

    first = layer_compiler._encoded_parent_attention_contribution(
        parent_context=context,
        cached_source_output=source_output,
        cached_target_output=target_output,
        name="sealed-first",
        prepared_transport=prepared,
    )
    raw = layer_compiler._encoded_parent_attention_contribution(
        parent_context=context,
        cached_source_output=source_output,
        cached_target_output=target_output,
        source_output_weight=output_weight,
        target_encoder=encoder,
        target_output_blocks=prepared.target_output_blocks,
        name="raw-equivalent",
    )
    mismatched_blocks = prepared.target_output_blocks.clone()
    mismatched_blocks[0, 0, 0] += 1.0
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="do not match the source O/chart composition",
    ):
        layer_compiler._encoded_parent_attention_contribution(
            parent_context=context,
            cached_source_output=source_output,
            cached_target_output=target_output,
            source_output_weight=output_weight,
            target_encoder=encoder,
            target_output_blocks=mismatched_blocks,
            name="mismatched-static-composition",
        )
    # The prepared object owns copies.  Mutating the source tensors after the
    # one-time seal cannot alter any later phase contribution.
    output_weight.add_(1000.0)
    encoder.mul_(-3.0)
    second = layer_compiler._encoded_parent_attention_contribution(
        parent_context=context,
        cached_source_output=source_output,
        cached_target_output=target_output,
        name="sealed-second",
        prepared_transport=prepared,
    )

    torch.testing.assert_close(first, expected, atol=0.0, rtol=0.0)
    assert torch.equal(raw, first)
    torch.testing.assert_close(second, first, atol=0.0, rtol=0.0)
    assert len(prepared.seal_sha256) == 64
    receipt = layer_compiler._attention_output_transport_complexity_receipt(
        transport=prepared,
        phase_count=1_386,
    )
    assert receipt["legacy_composition_count"] == 1_387
    assert receipt["prepared_composition_count"] == 1
    assert receipt["avoided_composition_flops"] == 582_120
    assert receipt["avoided_phase_fp64_weight_materialization_bytes"] == 465_696
    assert receipt["transport_seal_sha256"] == prepared.seal_sha256
    assert receipt["report_sha256"] == _sha(
        {key: value for key, value in receipt.items() if key != "report_sha256"}
    )

    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="must be supplied alone",
    ):
        layer_compiler._encoded_parent_attention_contribution(
            parent_context=context,
            cached_source_output=source_output,
            cached_target_output=target_output,
            source_output_weight=output_weight,
            name="ambiguous-transport",
            prepared_transport=prepared,
        )

    prepared.target_output_blocks[0, 0, 0] += 1.0
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="differs from its prepared O transport seal",
    ):
        layer_compiler._encoded_parent_attention_contribution(
            parent_context=context,
            cached_source_output=source_output,
            cached_target_output=target_output,
            name="mutated-prepared-transport",
            prepared_transport=prepared,
        )


def test_attention_o_contribution_rejects_nonfinite_roundoff_bound() -> None:
    context = torch.ones((1, 1, 1, 2), dtype=torch.float64)
    output_weight = torch.tensor(
        [[1.0e308, -1.0e308]],
        dtype=torch.float64,
    )
    encoder = torch.tensor([[1.0e-308]], dtype=torch.float64)
    source_output = torch.zeros((1, 1, 1), dtype=torch.float64)
    target_output = torch.zeros((1, 1, 1), dtype=torch.float64)
    prepared = layer_compiler._prepare_attention_output_transport_from_tensors(
        source_output_weight=output_weight,
        target_encoder=encoder,
        query_heads=1,
        head_dim=2,
        name="nonfinite-bound",
    )

    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="non-finite roundoff bound",
    ):
        layer_compiler._encoded_parent_attention_contribution(
            parent_context=context,
            cached_source_output=source_output,
            cached_target_output=target_output,
            name="nonfinite-bound",
            prepared_transport=prepared,
        )


def test_endpoint_metrics_are_separated_by_exact_causal_phase() -> None:
    class Phase(SimpleNamespace):
        @property
        def identity(self):
            return self.batch_index, self.phase

        def suffix(self, value):
            return value[:, -self.sampled_suffix_tokens :, :]

    phases = []
    operator_replays = {}
    final_replays = {}
    for split_index, split in enumerate(("fit", "validation")):
        for phase_index, (phase_name, prefix, suffix) in enumerate(
            (("prefill", 3, 3), ("decode.000", 4, 1))
        ):
            base = float(10 * split_index + phase_index)
            operator_target = torch.full((1, suffix, 2), base)
            layer_target = torch.full((1, suffix, 2), base + 0.5)
            phase = Phase(
                batch_index=split_index,
                split=split,
                phase=phase_name,
                batch_size=1,
                full_prefix_tokens=prefix,
                sampled_suffix_tokens=suffix,
                phase_sha256=_sha(f"{split}:{phase_name}"),
                parent_evidence=SimpleNamespace(
                    layers=(
                        SimpleNamespace(
                            layer_index=0,
                            operator_kind="full_attention",
                            tensors={
                                "residual.post_operator": operator_target,
                                "residual.post_layer": layer_target,
                            },
                        ),
                    )
                ),
            )
            phases.append(phase)
            operator_prediction = torch.zeros((1, prefix, 2))
            layer_prediction = torch.zeros((1, prefix, 2))
            operator_prediction[:, -suffix:, :] = operator_target + 1.0
            layer_prediction[:, -suffix:, :] = layer_target + 2.0
            operator_replays[phase.identity] = SimpleNamespace(
                post_operator_residual=operator_prediction
            )
            final_replays[phase.identity] = SimpleNamespace(
                post_layer_residual=layer_prediction
            )

    metrics = layer_compiler._split_endpoint_metrics(
        layer_index=0,
        input_bank=SimpleNamespace(phases=tuple(phases)),
        operator_replay=SimpleNamespace(replays=operator_replays),
        final_replay=SimpleNamespace(replays=final_replays),
        chart=SimpleNamespace(
            operator_kind="full_attention",
            post_operator=SimpleNamespace(
                encode_shaped=lambda value: value.clone(),
            ),
            post_layer=SimpleNamespace(
                encode_shaped=lambda value: value.clone(),
            ),
        ),
    )

    for split in ("fit", "validation"):
        phase_metrics = metrics[split]["phase_metrics"]
        assert set(phase_metrics) == {"prefill", "decode.000"}
        assert phase_metrics["prefill"]["post_operator"]["rows"] == 3
        assert phase_metrics["decode.000"]["post_operator"]["rows"] == 1
        assert phase_metrics["prefill"]["post_operator"]["rmse"] == pytest.approx(1.0)
        assert phase_metrics["decode.000"]["post_layer"]["rmse"] == pytest.approx(2.0)
        assert phase_metrics["prefill"]["phase_count"] == 1
        assert len(phase_metrics["decode.000"]["report_sha256"]) == 64


def test_attention_trajectory_rejects_sampled_prefill_tail() -> None:
    source, target = _models("full_attention")
    generator = torch.Generator().manual_seed(2026071705)
    tensors = _attention_evidence(
        source,
        torch.randn(
            (1, 3, 8),
            generator=generator,
            dtype=torch.float32,
        ),
        2,
    )
    phase = SimpleNamespace(
        identity=(0, "prefill"),
        batch_index=0,
        record_id="sampled-prefill",
        split="fit",
        phase="prefill",
        batch_size=1,
        full_prefix_tokens=3,
        sampled_suffix_tokens=2,
        input_ids=torch.arange(3).reshape(1, 3),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        position_ids=torch.arange(3).reshape(1, 3),
        parent_evidence=SimpleNamespace(
            layers=(
                SimpleNamespace(
                    layer_index=0,
                    operator_kind="full_attention",
                    tensors=tensors,
                ),
            )
        ),
    )
    child = torch.randn(
        (1, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )

    class InputBank:
        phases = (phase,)

        @staticmethod
        def state(_phase: SimpleNamespace) -> object:
            return child.clone()

    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="complete prefill Q/K/V trajectory",
    ):
        layer_compiler._attention_cases(
            source_model=source,
            target_model=target,
            source_layer=source.model.layers[0],
            layer_index=0,
            input_bank=InputBank(),
            preinstall_replay=SimpleNamespace(
                replays={
                    phase.identity: SimpleNamespace(
                        operator_input=child.clone(),
                    )
                }
            ),
            chart=_chart("full_attention"),
        )


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
            observed["operator_splits"] = tuple(case.split for case in kwargs["cases"])
            return original_operator(**kwargs)

        def unexpected_v4(**_kwargs):
            pytest.fail("The v3-compatible default dispatched short-conv v4.")

        monkeypatch.setattr(
            layer_compiler,
            "compile_lfm2_cpcf_short_conv_fold",
            conv_spy,
        )
        monkeypatch.setattr(
            layer_compiler,
            "compile_lfm2_cpcf_short_conv_v4_fold",
            unexpected_v4,
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
    for split in ("fit", "validation"):
        assert set(result.split_metrics[split]["phase_metrics"]) == {"prefill"}
    assert observed["mlp_case_count"] == 1
    if kind == "full_attention":
        assert observed["operator_case_count"] == 1
    else:
        assert observed["operator_splits"] == (
            "selection",
            "fit",
            "validation",
        )
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
    if kind == "full_attention":
        assert policy["operator_case_count"] == 1
        assert policy["attention_fit_case_granularity"] == (
            "one_complete_causal_trajectory_per_fit_record"
        )
        assert policy["attention_key_linear_history"] == (
            "exact_prefill_plus_ordered_decode_suffixes"
        )
        assert policy["attention_inverse_rope_preimage_used"] is False
        assert policy["attention_repeated_prefix_fit_rows"] is False
        assert policy["attention_fit_phase_weighting"] == (
            "equal_total_mass_per_causal_phase_with_global_mean_square_one"
        )
        assert policy["attention_parent_residual_contribution"] == (
            "head_local_O_write_composed_through_post_operator_chart"
        )
        assert policy["attention_residual_contribution_direction_preserved"] is True
        runtime_receipt = policy["attention_runtime_complexity_receipt"]
        assert runtime_receipt["report_sha256"] == _sha(
            {
                key: value
                for key, value in runtime_receipt.items()
                if key != "report_sha256"
            }
        )
        output_receipt = runtime_receipt["output_transport"]
        assert output_receipt["phase_count"] == 1
        assert output_receipt["source_width"] == 8
        assert output_receipt["query_heads"] == 4
        assert output_receipt["head_dim"] == 2
        assert output_receipt["target_width"] == 4
        assert len(output_receipt["transport_seal_sha256"]) == 64
        offset_receipt = runtime_receipt["equal_offset_identity_short_circuit"]
        assert offset_receipt["identity_short_circuit_executed"] is True
        assert offset_receipt["fit_case_count"] == 1
        assert offset_receipt["fit_observation_rows"] == 12
        assert offset_receipt["avoided_offset_feature_bytes"] == 3_072
        assert offset_receipt["avoided_minimum_live_bytes_at_legacy_cat"] == 9_216
        assert result.result_sha256 == _sha(
            {
                key: value
                for key, value in result.report.items()
                if key != "result_sha256"
            }
        )
    else:
        assert policy["attention_runtime_complexity_receipt"] is None
    assert policy["endpoint_metrics_by_causal_phase"] is True
    state_policy = result.report["actual_child_state_policy"]
    assert state_policy["next_layer_input"] == (
        "complete_prefill_plus_ordered_decode_suffix_layer_replay"
    )
    assert state_policy["recomputed_historical_prefix_propagated"] is False
    assert _state_sha(source.model.layers[0]) == source_before
    assert {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    } == target_ids
    assert result.report["execution"]["training_steps"] == 0
    assert result.report["execution"]["target_model_forward_calls"] == 0


def test_attention_phase_weighting_is_explicitly_split_scoped() -> None:
    phases = (
        SimpleNamespace(
            identity=(0, "prefill"),
            split="fit",
            batch_size=2,
            sampled_suffix_tokens=3,
        ),
        SimpleNamespace(
            identity=(1, "prefill"),
            split="selection",
            batch_size=1,
            sampled_suffix_tokens=2,
        ),
        SimpleNamespace(
            identity=(2, "prefill"),
            split="selection",
            batch_size=2,
            sampled_suffix_tokens=1,
        ),
    )

    weights = layer_compiler._equal_causal_phase_sqrt_weights(
        phases,
        required_split="selection",
    )

    assert set(weights) == {(1, "prefill"), (2, "prefill")}
    assert (0, "prefill") not in weights
    observed_mass = sum(
        float(value.square().sum().item()) for value in weights.values()
    )
    assert observed_mass == pytest.approx(4.0)
    assert float(weights[(1, "prefill")][0, 0].square().item()) == pytest.approx(1.0)
    assert float(weights[(2, "prefill")][0, 0].square().item()) == pytest.approx(1.0)


def test_residual_attention_candidate_is_split_isolated_and_macrocell_selected(
    monkeypatch,
) -> None:
    import gc
    import weakref

    source, target, bank, charts = _fixture("full_attention")
    replay_events = []
    attention_case_calls = []
    case_identity = {}
    score_case_identity = {}
    mlp_preparation_count = 0
    mlp_compile_count = 0
    mlp_prepared_basis_identities = []
    mlp_prepared_basis_sha256 = []
    mlp_candidate_child_case_sha256 = []
    mlp_artifacts = []
    frozen_candidates = []
    candidate_replay_refs = []

    original_replay = layer_compiler.replay_lfm2_cpcf_phase_bank
    original_attention_cases = layer_compiler._attention_cases
    original_producer = layer_compiler.compile_lfm2_structural_attention_producer_fold
    original_score = (
        layer_compiler.evaluate_lfm2_compiled_attention_producer_score_preservation
    )
    original_prepare_mlp = (
        layer_compiler.prepare_lfm2_structural_residual_mlp_fold_basis
    )
    original_mlp = layer_compiler.compile_lfm2_structural_residual_mlp_fold
    original_freeze = layer_compiler._freeze_attention_candidate_winner
    original_seal = layer_compiler._seal_attention_candidate_evaluation

    def replay_spy(*args, **kwargs):
        splits = tuple(sorted({phase.split for phase in kwargs["input_bank"].phases}))
        replay_events.append(("replay", splits))
        return original_replay(*args, **kwargs)

    def seal_spy(**kwargs):
        candidate_replay_refs.append(
            (
                weakref.ref(
                    next(
                        iter(
                            kwargs[
                                "fit_operator_replay"
                            ].post_layer_bank.states.values()
                        )
                    )
                ),
                weakref.ref(
                    next(
                        iter(
                            kwargs[
                                "selection_replay"
                            ].post_layer_bank.states.values()
                        )
                    )
                ),
            )
        )
        return original_seal(**kwargs)

    def attention_cases_spy(**kwargs):
        required_split = kwargs["required_split"]
        observed_splits = {phase.split for phase in kwargs["input_bank"].phases}
        assert observed_splits == {required_split}
        result = original_attention_cases(**kwargs)
        attention_case_calls.append((required_split, observed_splits))
        case_identity[required_split] = result[0]
        return result

    def producer_spy(**kwargs):
        assert kwargs["cases"] is case_identity["fit"]
        return original_producer(**kwargs)

    def score_spy(producer, cases, **kwargs):
        assert cases is case_identity["selection"]
        score_case_identity["cases"] = cases
        return original_score(producer, cases, **kwargs)

    def prepare_mlp_spy(**kwargs):
        nonlocal mlp_preparation_count
        mlp_preparation_count += 1
        return original_prepare_mlp(**kwargs)

    def mlp_spy(**kwargs):
        nonlocal mlp_compile_count
        mlp_compile_count += 1
        prepared_basis = kwargs["prepared_basis"]
        assert prepared_basis is not None
        mlp_prepared_basis_identities.append(id(prepared_basis))
        mlp_prepared_basis_sha256.append(prepared_basis.preparation_sha256)
        mlp_candidate_child_case_sha256.append(
            tuple(
                (
                    structural_fold._tensor_sha256(case.target_input),
                    structural_fold._tensor_sha256(
                        case.target_pre_mlp_residual,
                    ),
                )
                for case in kwargs["cases"]
            )
        )
        artifact = original_mlp(**kwargs)
        mlp_artifacts.append(artifact)
        return artifact

    def freeze_spy(candidates):
        assert not any(
            "validation" in splits
            for event, splits in replay_events
            if event == "replay"
        )
        gc.collect()
        assert len(candidate_replay_refs) == 2
        assert all(
            replay_ref() is None
            for replay_refs in candidate_replay_refs
            for replay_ref in replay_refs
        )
        materialized = tuple(candidates)
        frozen_candidates.extend(materialized)
        assert {
            *layer_compiler._Lfm2CpcfAttentionCandidateEvaluation.__slots__
        }.isdisjoint({"fit_operator_replay", "selection_replay"})
        assert all(
            not hasattr(candidate, "fit_operator_replay")
            and not hasattr(candidate, "selection_replay")
            and not any(
                isinstance(
                    getattr(candidate, name),
                    layer_compiler.Lfm2CpcfLayerReplayBank,
                )
                for name in candidate.__slots__
            )
            for candidate in materialized
        )
        result = original_freeze(materialized)
        replay_events.append(("freeze", (result[0].candidate_id,)))
        return result

    monkeypatch.setattr(
        layer_compiler,
        "replay_lfm2_cpcf_phase_bank",
        replay_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "_attention_cases",
        attention_cases_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_structural_attention_producer_fold",
        producer_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "evaluate_lfm2_compiled_attention_producer_score_preservation",
        score_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "prepare_lfm2_structural_residual_mlp_fold_basis",
        prepare_mlp_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_structural_residual_mlp_fold",
        mlp_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "_freeze_attention_candidate_winner",
        freeze_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "_seal_attention_candidate_evaluation",
        seal_spy,
    )

    result = layer_compiler.compile_and_install_lfm2_cpcf_layer(
        source,
        target,
        layer_index=0,
        input_bank=bank,
        boundary_charts=charts,
    )

    assert attention_case_calls == [
        ("fit", {"fit"}),
        ("selection", {"selection"}),
    ]
    assert score_case_identity["cases"] is case_identity["selection"]
    assert mlp_preparation_count == 1
    assert mlp_compile_count == 2
    assert len(set(mlp_prepared_basis_identities)) == 1
    assert len(set(mlp_prepared_basis_sha256)) == 1
    assert len(set(mlp_candidate_child_case_sha256)) == 2
    assert all(artifact.prepared_basis_reused is True for artifact in mlp_artifacts)
    assert all(
        artifact.prepared_basis_sha256 == mlp_prepared_basis_sha256[0]
        for artifact in mlp_artifacts
    )
    freeze_index = next(
        index for index, event in enumerate(replay_events) if event[0] == "freeze"
    )
    validation_replay_indices = [
        index
        for index, (event, splits) in enumerate(replay_events)
        if event == "replay" and "validation" in splits
    ]
    assert validation_replay_indices
    assert all(index > freeze_index for index in validation_replay_indices)

    policy = result.report["solver_evidence_policy"]
    selection = policy["attention_candidate_selection"]
    evaluations = policy["attention_candidate_evaluations"]
    assert selection["winner_frozen_before_validation"] is True
    assert selection["validation_phase_count_consumed_before_freeze"] == 0
    assert selection["report_sha256"] == _sha(
        {name: value for name, value in selection.items() if name != "report_sha256"}
    )
    assert set(evaluations) == {
        "direct_monolithic",
        "residual_attention_split",
    }
    expected_winner = min(
        evaluations,
        key=lambda name: tuple(evaluations[name]["selection_key"]),
    )
    assert selection["selected_candidate_id"] == expected_winner
    selected_candidate = next(
        candidate
        for candidate in frozen_candidates
        if candidate.candidate_id == expected_winner
    )
    assert selected_candidate.operator_artifact is result.operator_artifact
    assert selected_candidate.mlp_artifact is result.mlp_artifact
    for candidate in evaluations.values():
        assert candidate["operator_compiled_from_splits"] == ["fit"]
        assert candidate["mlp_compiled_from_splits"] == ["fit"]
        assert candidate["candidate_evaluated_on_splits"] == ["selection"]
        assert candidate["validation_phase_count_consumed"] == 0
        assert candidate["rollback_proof"]["exact_layer_rollback_proven"] is True
        assert set(candidate["selection_metrics"]) == {"selection"}
        assert candidate["live_operator_state_authenticated_before_fit_replay"] is True
        assert candidate["live_mlp_state_authenticated_before_selection_replay"] is True
        assert candidate["selection_context_ranked_before_post_layer_endpoint"] is True
        assert candidate["selection_context_metrics"]["selection_split_only"] is True
        assert candidate["query_key_score_report"]["case_count"] == len(
            case_identity["selection"]
        )
        assert candidate["query_key_score_producer_view_exact_state_proven"] is True
        assert candidate["report_sha256"] == _sha(
            {
                name: value
                for name, value in candidate.items()
                if name != "report_sha256"
            }
        )

    split = evaluations["residual_attention_split"]["operator_artifact"]
    assert split["schema"] == (layer_compiler.LFM2_CPCF_RESIDUAL_ATTENTION_SPLIT_SCHEMA)
    assert split["fit_only_producer_compile"] is True
    assert split["fit_only_value_output_compile"] is True
    assert split["selection_only_query_key_score_evaluation"] is True
    assert split["validation_used_for_compilation_or_selection"] is False
    assert split["actual_context_source"] == (
        "temporary_installed_qkv_fit_bank_operator_output_input"
    )
    assert split["live_value_source"] == "compiled_producer_v"
    assert split["value_prior_source"] == "compiled_producer_v"
    assert split["output_prior_source"] == "original_blank_target_o"
    assert split["temporary_rollback_proof"]["exact_layer_rollback_proven"] is True
    assert split["score_selection_report"]["case_count"] == len(
        case_identity["selection"]
    )
    assert split["report_sha256"] == _sha(
        {name: value for name, value in split.items() if name != "report_sha256"}
    )
    assert isinstance(
        result.operator_artifact,
        (
            layer_compiler.Lfm2CompiledStructuralAttentionFold,
            layer_compiler.Lfm2CompiledCpcfResidualAttentionSplitFold,
        ),
    )


@pytest.mark.parametrize(
    "forced_candidate",
    ("direct_monolithic", "residual_attention_split"),
)
def test_attention_forced_winner_is_final_live_state_and_split_rejects_tamper(
    forced_candidate,
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("full_attention")
    original_key = layer_compiler._attention_candidate_selection_key

    def forced_key(candidate_id, selection_metrics, context_metrics, score_report):
        observed = original_key(
            candidate_id,
            selection_metrics,
            context_metrics,
            score_report,
        )
        return (
            0.0 if candidate_id == forced_candidate else 1e6,
            *observed[1:],
        )

    monkeypatch.setattr(
        layer_compiler,
        "_attention_candidate_selection_key",
        forced_key,
    )
    result = layer_compiler.compile_and_install_lfm2_cpcf_layer(
        source,
        target,
        layer_index=0,
        input_bank=bank,
        boundary_charts=charts,
    )

    selection = result.report["solver_evidence_policy"]["attention_candidate_selection"]
    assert selection["selected_candidate_id"] == forced_candidate
    attention = target.model.layers[0].self_attn
    live_operator_sha = layer_compiler._tensor_bundle_sha256(
        {
            "q_proj.weight": attention.q_proj.weight,
            "k_proj.weight": attention.k_proj.weight,
            "v_proj.weight": attention.v_proj.weight,
            "out_proj.weight": attention.out_proj.weight,
        }
    )
    assert live_operator_sha == layer_compiler._tensor_bundle_sha256(
        result.operator_artifact.state_dict()
    )
    mlp = target.model.layers[0].feed_forward
    live_mlp_sha = layer_compiler._tensor_bundle_sha256(
        {
            "w1.weight": mlp.w1.weight,
            "w3.weight": mlp.w3.weight,
            "w2.weight": mlp.w2.weight,
        }
    )
    assert live_mlp_sha == layer_compiler._tensor_bundle_sha256(
        result.mlp_artifact.state_dict()
    )

    if forced_candidate == "direct_monolithic":
        assert isinstance(
            result.operator_artifact,
            layer_compiler.Lfm2CompiledStructuralAttentionFold,
        )
        return

    artifact = result.operator_artifact
    assert isinstance(
        artifact,
        layer_compiler.Lfm2CompiledCpcfResidualAttentionSplitFold,
    )
    tampered_report = dict(artifact.report)
    tampered_producer = dict(tampered_report["producer_artifact"])
    tampered_producer["consumer_compiled"] = True
    tampered_report["producer_artifact"] = tampered_producer
    tampered_report["producer_artifact_sha256"] = _sha(tampered_producer)
    tampered_report["report_sha256"] = _sha(
        {
            name: value
            for name, value in tampered_report.items()
            if name != "report_sha256"
        }
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="artifact seal is invalid",
    ):
        replace(
            artifact,
            report=tampered_report,
        )


def test_short_conv_v4_diverse_macrocell_winner_can_beat_endpoint_top4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    source, target, bank, charts = _fixture("short_conv")
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    original = layer_compiler.compile_lfm2_cpcf_short_conv_v4_fold
    original_cluster_builder = structural_fold.build_gated_mlp_activation_cluster_plan
    original_replay = layer_compiler.replay_lfm2_cpcf_phase_bank
    observed = {}
    cluster_builder_calls = 0
    candidate_replay_refs = []

    def cluster_builder_spy(*args, **kwargs):
        nonlocal cluster_builder_calls
        cluster_builder_calls += 1
        return original_cluster_builder(*args, **kwargs)

    def replay_spy(*args, **kwargs):
        selector = observed.get("macrocell_candidate_selector")
        input_bank = kwargs["input_bank"]
        if selector is not None and input_bank is selector.fit_bank:
            if candidate_replay_refs:
                gc.collect()
                assert all(
                    replay_ref() is None
                    for replay_ref in candidate_replay_refs[-1]
                )
            replay = original_replay(*args, **kwargs)
            candidate_replay_refs.append(
                [
                    weakref.ref(
                        next(iter(replay.post_layer_bank.states.values()))
                    )
                ]
            )
            return replay
        replay = original_replay(*args, **kwargs)
        if selector is not None and input_bank is selector.selection_bank:
            candidate_replay_refs[-1].append(
                weakref.ref(next(iter(replay.post_layer_bank.states.values())))
            )
        return replay

    class EndpointControl:
        def __init__(self):
            self.requests = []

        def select(self, request):
            self.requests.append(copy.deepcopy(request))
            raise RuntimeError("endpoint selector found no feasible candidate")

    endpoint_control = EndpointControl()

    def v4_spy(**kwargs):
        observed["source_in_proj_weight"] = kwargs["source_in_proj_weight"]
        observed["source_fir_weight"] = kwargs["source_fir_weight"]
        observed["source_out_proj_weight"] = kwargs["source_out_proj_weight"]
        observed["output_encoder"] = kwargs["output_encoder"]
        observed["cases"] = tuple(kwargs["cases"])
        observed["solver_policy"] = kwargs["solver_policy"]
        observed["candidate_selector"] = kwargs["candidate_selector"]
        observed["macrocell_candidate_selector"] = kwargs[
            "macrocell_candidate_selector"
        ]
        observed["macrocell_fit_bank_sha256"] = kwargs["macrocell_fit_bank_sha256"]
        observed["macrocell_selection_bank_sha256"] = kwargs[
            "macrocell_selection_bank_sha256"
        ]
        return original(**kwargs)

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_v4_fold",
        v4_spy,
    )
    monkeypatch.setattr(
        structural_fold,
        "build_gated_mlp_activation_cluster_plan",
        cluster_builder_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "replay_lfm2_cpcf_phase_bank",
        replay_spy,
    )
    result = layer_compiler.compile_and_install_lfm2_cpcf_layer(
        source,
        target,
        layer_index=0,
        input_bank=bank,
        boundary_charts=charts,
        short_conv_v4_solver_policy=solver_policy,
        candidate_selector=endpoint_control,
    )

    source_conv = source.model.layers[0].conv
    assert observed["source_in_proj_weight"] is source_conv.in_proj.weight
    assert observed["source_fir_weight"] is source_conv.conv.weight
    assert observed["source_out_proj_weight"] is source_conv.out_proj.weight
    assert observed["output_encoder"] is charts.post_operator.mapping.encoder
    assert observed["solver_policy"] is solver_policy
    assert observed["candidate_selector"] is None
    assert len(endpoint_control.requests) == 1
    assert isinstance(
        observed["macrocell_candidate_selector"],
        layer_compiler._Lfm2CpcfShortConvMacrocellSelector,
    )
    macrocell_selector = observed["macrocell_candidate_selector"]
    assert {
        phase.split for phase in macrocell_selector.audit_bank.phases
    } == {"fit", "selection"}
    assert not hasattr(
        macrocell_selector,
        "input_bank",
    )
    assert len(observed["macrocell_fit_bank_sha256"]) == 64
    assert len(observed["macrocell_selection_bank_sha256"]) == 64
    assert (
        observed["macrocell_fit_bank_sha256"]
        != observed["macrocell_selection_bank_sha256"]
    )
    assert tuple(case.split for case in observed["cases"]) == (
        "selection",
        "fit",
        "validation",
    )
    assert all(
        isinstance(case, Lfm2CpcfShortConvV4FoldCase)
        and tuple(case.parent_operator_input_history.shape) == (2, 6, 8)
        and case.verify_authentication()
        for case in observed["cases"]
    )
    assert isinstance(result.operator_artifact, Lfm2CompiledCpcfShortConvV4Fold)
    assert result.operator_artifact.verify_authentication() is True
    assert (
        result.operator_artifact.report["solver_policy_sha256"]
        == solver_policy.authentication_sha256
    )
    artifact_report = result.operator_artifact.report
    assert artifact_report["macrocell_candidate_selector_active"] is True
    assert artifact_report["validation_used_for_macrocell_selection"] is False
    assert artifact_report["validation_frozen_until_after_macrocell_winner"] is True
    decision = result.report["solver_evidence_policy"]["short_conv_macrocell_selection"]
    assert decision["candidate_budget"] == 8
    assert len(decision["evaluated_candidate_ids"]) == 8
    assert decision["candidate_specific_fit_mlp_folds"] == 8
    assert decision["candidate_specific_selection_replays"] == 8
    assert decision["exact_candidate_rollbacks"] == 8
    assert decision["validation_accessible"] is False
    assert decision["validation_replays"] == 0
    gc.collect()
    assert len(candidate_replay_refs) == 8
    assert all(len(replay_refs) == 2 for replay_refs in candidate_replay_refs)
    assert all(
        replay_ref() is None
        for replay_refs in candidate_replay_refs
        for replay_ref in replay_refs
    )
    assert dict(macrocell_selector.decision) == decision
    assert {
        *layer_compiler._Lfm2CpcfShortConvMacrocellEvaluation.__slots__
    } == {
        "candidate",
        "mlp_artifact",
        "selection_key",
        "report",
    }
    assert not any(
        isinstance(value, layer_compiler.Lfm2CpcfLayerReplayBank)
        for value in vars(macrocell_selector).values()
    )
    for candidate_id, evaluation in macrocell_selector.evaluations.items():
        assert not hasattr(evaluation, "fit_operator_replay")
        assert not hasattr(evaluation, "selection_replay")
        assert not any(
            isinstance(getattr(evaluation, name), layer_compiler.Lfm2CpcfLayerReplayBank)
            for name in evaluation.__slots__
        )
        assert dict(evaluation.report) == decision["candidate_evaluations"][
            candidate_id
        ]
    assert cluster_builder_calls == 1
    evidence_policy = result.report["solver_evidence_policy"]
    assert evidence_policy["short_conv_mlp_preparation_count"] == 1
    assert evidence_policy["short_conv_mlp_candidate_compiles_from_prepared_basis"] == 8
    assert len(evidence_policy["short_conv_mlp_prepared_basis_sha256"]) == 64
    assert (
        evidence_policy["short_conv_mlp_prepared_basis_validation_evidence_consumed"]
        == 0
    )
    assert (
        len(
            {
                value["fit_mlp_compiled_state_sha256"]
                for value in decision["candidate_evaluations"].values()
            }
        )
        > 1
    )
    assert (
        decision["selected_candidate_id"]
        != artifact_report["default_v4_selected_candidate_id"]
    )
    endpoint_top4 = {
        value["candidate_id"]
        for value in artifact_report["selection_ranking"]["ranking"][:4]
    }
    assert decision["selected_candidate_id"] not in endpoint_top4
    assert decision["selected_candidate_id"] in decision["evaluated_candidate_ids"]
    assert "--anchor-" in decision["selected_candidate_id"]
    selected_evaluation = macrocell_selector.evaluations[
        decision["selected_candidate_id"]
    ]
    assert macrocell_selector.selected is selected_evaluation
    assert selected_evaluation.mlp_artifact is result.mlp_artifact
    assert (
        selected_evaluation.mlp_artifact.compiled_state_sha256
        == selected_evaluation.report["fit_mlp_compiled_state_sha256"]
    )
    for evaluation in decision["candidate_evaluations"].values():
        assert evaluation["fit_only_mlp_fold"] is True
        assert evaluation["selection_only_candidate_evaluation"] is True
        assert evaluation["validation_accessible"] is False
        assert evaluation["rollback_proof"]["exact_layer_snapshot_restored"] is True
        assert (
            len(evaluation["operator_install_report"]["installed_state_sha256"]) == 64
        )
    binding = result.report["solver_evidence_policy"][
        "short_conv_final_macrocell_binding"
    ]
    assert binding["selected_candidate_id"] == decision["selected_candidate_id"]
    assert binding["output_artifact_equals_selected_candidate"] is True
    assert binding["exact_selected_fit_only_mlp_reused"] is True
    assert binding["whole_macrocell_overturned_endpoint_control"] is True
    assert binding["endpoint_selector_report_sha256"] is None
    assert (
        binding["endpoint_control_candidate_id"]
        == (artifact_report["default_v4_selected_candidate_id"])
    )
    assert binding["endpoint_selector_diagnostic"]["status"] == (
        "raised_fallback_to_default"
    )
    live_mlp = target.model.layers[0].feed_forward
    assert result.mlp_artifact.prepared_basis_reused is True
    assert (
        result.mlp_artifact.prepared_basis_sha256
        == evidence_policy["short_conv_mlp_prepared_basis_sha256"]
    )
    assert binding["final_live_mlp_bundle_sha256"] == (
        layer_compiler._tensor_bundle_sha256(
            {
                "w1.weight": live_mlp.w1.weight,
                "w3.weight": live_mlp.w3.weight,
                "w2.weight": live_mlp.w2.weight,
            }
        )
    )
    assert result.operator_install_report["scope"] == (
        "authenticated_residual_endpoint_cpcf_short_conv_v4_install"
    )
    assert result.report["execution"]["training_steps"] == 0
    assert result.report["execution"]["parameters_pruned"] == 0
    assert result.report["execution"]["parameters_quantized"] == 0


def test_short_conv_macrocell_candidate_failure_restores_exact_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    before = copy.deepcopy(layer.state_dict())
    identities = {name: id(value) for name, value in layer.named_parameters()}
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    original_compile = layer_compiler._compile_cpcf_mlp_from_fit_replay
    original_prepare = layer_compiler._prepare_cpcf_mlp_basis_from_fit_replay
    prepared_bases = []

    def prepare_spy(**kwargs):
        basis = original_prepare(**kwargs)
        prepared_bases.append(basis)
        return basis

    def fail_candidate_mlp(**kwargs):
        basis = kwargs["prepared_basis"]
        assert basis is prepared_bases[-1]
        assert basis.verify_authentication() is True
        raise RuntimeError("forced candidate-owned MLP failure")

    monkeypatch.setattr(
        layer_compiler,
        "_prepare_cpcf_mlp_basis_from_fit_replay",
        prepare_spy,
    )
    monkeypatch.setattr(
        layer_compiler,
        "_compile_cpcf_mlp_from_fit_replay",
        fail_candidate_mlp,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilationFailure,
        match="forced candidate-owned MLP failure",
    ) as caught:
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
            short_conv_v4_solver_policy=solver_policy,
        )

    assert caught.value.rollback_proof["exact_layer_rollback_proven"] is True
    assert {name: id(value) for name, value in layer.named_parameters()} == identities
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert len(prepared_bases) == 1
    assert prepared_bases[0].verify_authentication() is True

    monkeypatch.setattr(
        layer_compiler,
        "_compile_cpcf_mlp_from_fit_replay",
        original_compile,
    )
    recovered = layer_compiler.compile_and_install_lfm2_cpcf_layer(
        source,
        target,
        layer_index=0,
        input_bank=bank,
        boundary_charts=charts,
        short_conv_v4_solver_policy=solver_policy,
    )
    assert len(prepared_bases) == 2
    assert prepared_bases[1] is not prepared_bases[0]
    assert prepared_bases[1].verify_authentication() is True
    assert recovered.mlp_artifact.prepared_basis_reused is True
    assert (
        recovered.mlp_artifact.prepared_basis_sha256
        == prepared_bases[1].preparation_sha256
    )


def test_short_conv_macrocell_rejects_resealed_mlp_binding_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    before = copy.deepcopy(layer.state_dict())
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    original = layer_compiler._seal_macrocell_evidence

    def tamper(kind, candidate_id, actual_report, **bindings):
        report = original(
            kind,
            candidate_id,
            actual_report,
            **bindings,
        )
        if kind != "actual_fit_mlp_artifact":
            return report
        forged = copy.deepcopy(report)
        actual = forged["actual_report"]
        actual["compiled_state_sha256"] = "f" * 64
        actual.pop("report_sha256")
        actual["report_sha256"] = layer_compiler._sha(actual)
        forged["actual_report_sha256"] = actual["report_sha256"]
        forged.pop("report_sha256")
        forged["report_sha256"] = layer_compiler._sha(forged)
        return forged

    monkeypatch.setattr(
        layer_compiler,
        "_seal_macrocell_evidence",
        tamper,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilationFailure,
        match="state-bound|candidate evidence",
    ) as caught:
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
            short_conv_v4_solver_policy=solver_policy,
        )

    assert caught.value.rollback_proof["exact_layer_rollback_proven"] is True
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_short_conv_v4_gate_failure_is_strictly_nested_and_rolls_back() -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    target_before = copy.deepcopy(layer.state_dict())
    target_ids = {name: id(value) for name, value in layer.named_parameters()}
    solver_policy = Lfm2CpcfShortConvV4SolverPolicy(
        ridge_alphas=(1e-6,),
        variable_projection_passes=2,
        target_dtype="float32",
    )
    gate_policy = layer_compiler.Lfm2CpcfShortConvGatePolicy(
        maximum_fit_endpoint_relative_rmse=0.0,
        maximum_validation_endpoint_relative_rmse=0.0,
        minimum_fit_endpoint_cosine_similarity=1.0,
        minimum_validation_endpoint_cosine_similarity=1.0,
        enforce=True,
    )

    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilationFailure,
        match="failed enforced behavior gates",
    ) as caught:
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
            short_conv_gate_policy=gate_policy,
            short_conv_v4_solver_policy=solver_policy,
        )

    failure = caught.value
    nested = failure.short_conv_validation_report
    assert nested is not None
    assert nested["schema"] == (
        layer_compiler.LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA
    )
    assert nested["compiled_short_conv_fold_issued"] is False
    assert nested["promotion_claimed"] is False
    assert nested["promotable_candidate"] is False
    assert nested["failure_evidence_only"] is True
    assert layer_compiler._valid_short_conv_validation_report(nested) is True
    forged = copy.deepcopy(nested)
    selected = forged["selected_candidate_id"]
    candidate = forged["candidate_selection_reports"][selected]
    candidate.pop("report_sha256")
    candidate["promotion_claimed"] = True
    candidate["report_sha256"] = layer_compiler._sha(candidate)
    forged.pop("report_sha256")
    forged["report_sha256"] = layer_compiler._sha(forged)
    assert layer_compiler._recursive_report_seals_valid(forged) is True
    assert layer_compiler._valid_short_conv_validation_report(forged) is False

    def reseal(value):
        value.pop("report_sha256", None)
        value["report_sha256"] = layer_compiler._sha(value)

    forged_provenance = copy.deepcopy(nested)
    provenance_id = next(iter(forged_provenance["candidate_fit_reports"]))
    provenance = forged_provenance["candidate_fit_reports"][provenance_id]
    provenance["source_alias_count"] += 1
    reseal(provenance)
    reseal(forged_provenance)
    assert layer_compiler._recursive_report_seals_valid(forged_provenance) is True
    assert (
        layer_compiler._valid_short_conv_validation_report(forged_provenance) is False
    )

    forged_pool = copy.deepcopy(nested)
    pool = forged_pool["candidate_pool_report"]
    pool["raw_candidate_count"] = "bad"
    reseal(pool)
    reseal(forged_pool)
    assert layer_compiler._recursive_report_seals_valid(forged_pool) is True
    assert layer_compiler._valid_short_conv_validation_report(forged_pool) is False

    forged_frontier = copy.deepcopy(nested)
    omitted_id = next(
        candidate_id
        for candidate_id, provenance in forged_frontier["candidate_fit_reports"].items()
        if candidate_id != forged_frontier["selected_candidate_id"]
        and any(
            source["candidate_kind"] == "accepted_fit_path_anchor"
            for source in provenance["sources"]
        )
    )
    omitted_provenance = forged_frontier["candidate_fit_reports"].pop(omitted_id)
    forged_frontier["candidate_selection_reports"].pop(omitted_id)
    frontier_pool = forged_frontier["candidate_pool_report"]
    frontier_pool["candidate_ids"] = sorted(forged_frontier["candidate_fit_reports"])
    frontier_pool["candidate_state_sha256"].pop(omitted_id)
    frontier_pool["raw_candidate_count"] -= omitted_provenance["source_alias_count"]
    frontier_pool["unique_target_dtype_candidate_count"] -= 1
    frontier_pool["target_dtype_duplicate_count"] = (
        frontier_pool["raw_candidate_count"]
        - frontier_pool["unique_target_dtype_candidate_count"]
    )
    reseal(frontier_pool)
    forged_frontier["selection_ranking"] = dict(
        layer_compiler.rank_short_conv_v4_selection_candidates(
            forged_frontier["candidate_selection_reports"],
            policy_sha256=forged_frontier["solver_policy_sha256"],
            feasibility_policy=frontier_pool["selection_feasibility_policy"],
        )
    )
    assert (
        forged_frontier["selection_ranking"]["selected_candidate_id"]
        in forged_frontier["candidate_fit_reports"]
    )
    assert frontier_pool["accepted_phase_anchors_included"] is True
    reseal(forged_frontier)
    assert layer_compiler._recursive_report_seals_valid(forged_frontier) is True
    assert layer_compiler._valid_short_conv_validation_report(forged_frontier) is False

    forged_ranking = copy.deepcopy(nested)
    ranking = forged_ranking["selection_ranking"]
    ranking["ranking"][1], ranking["ranking"][2] = (
        ranking["ranking"][2],
        ranking["ranking"][1],
    )
    reseal(ranking)
    reseal(forged_ranking)
    assert layer_compiler._recursive_report_seals_valid(forged_ranking) is True
    assert layer_compiler._valid_short_conv_validation_report(forged_ranking) is False

    assert nested["schema"].endswith("macrocell-selector-failure-v1")
    assert failure.stage == "operator_compile"
    assert failure.rollback_proof["exact_layer_rollback_proven"] is True
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert {name: id(value) for name, value in layer.named_parameters()} == target_ids


def test_layer_transaction_rolls_back_operator_when_mlp_compile_fails(
    monkeypatch,
):
    source, target, bank, charts = _fixture("full_attention")
    layer = target.model.layers[0]
    layer.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )
    target_before = copy.deepcopy(layer.state_dict())
    target_ids = {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    }

    def fail_mlp(**_kwargs):
        with torch.no_grad():
            layer.rotary_inv_freq.fill_(11.0)
        raise RuntimeError("forced post-operator failure")

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_structural_residual_mlp_fold",
        fail_mlp,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilationFailure,
        match="forced post-operator failure",
    ) as caught:
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    failure = caught.value
    assert failure.stage == "mlp_compile"
    assert failure.layer_index == 0
    assert failure.operator_kind == "full_attention"
    assert failure.input_bank_sha256 == bank.state_bank_sha256
    assert failure.boundary_charts_sha256 == charts.layer_charts_sha256
    assert failure.rollback_proof["exact_layer_rollback_proven"] is True
    assert (
        failure.rollback_proof["target_all_named_parameters_and_buffers_restored"]
        is True
    )
    assert failure.report["layer_result_issued"] is False
    assert failure.report["promotion_claimed"] is False
    for name, value in target.model.layers[0].state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    } == target_ids
    assert torch.equal(layer.rotary_inv_freq, torch.tensor([2.0]))


def test_short_conv_gate_failure_preserves_nested_report_and_exact_rollback(
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    layer.register_buffer(
        "rotary_inv_freq",
        torch.tensor([2.0]),
        persistent=False,
    )
    parameter = layer.conv.in_proj.weight
    original_gradient = torch.full_like(parameter, 7.0)
    parameter.grad = original_gradient
    gradient_storage_pointer = int(original_gradient.untyped_storage().data_ptr())
    target_before = copy.deepcopy(layer.state_dict())
    target_ids = {name: id(value) for name, value in layer.named_parameters()}
    nested_body = {
        "schema": layer_compiler.LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
        "source_hidden_size": 8,
        "target_hidden_size": 4,
        "hidden_width_compression_ratio": 2.0,
        "source_fir_length": 3,
        "target_fir_length": 3,
        "target_logical_parameters": 1,
        "fit_case_count": 1,
        "validation_case_count": 1,
        "fit_metrics": {},
        "validation_metrics": {},
        "fit_case_reports": [],
        "validation_case_reports": [],
        "output_writer_scope": "authenticated_residual_suffix_only",
        "producer_and_fir_scope": "complete_causal_prefix",
        "fir_state_policy": (
            "zero_at_sequence_origin_then_warm_over_every_prefix_token"
        ),
        "writer_fit_rows": 1,
        "fit_prefix_rows": 1,
        "fit_fir_warmup_rows": 0,
        "provenance": {},
        "behavior_gate": {
            "policy": {
                "maximum_fit_endpoint_relative_rmse": 1.0,
                "maximum_validation_endpoint_relative_rmse": 1.0,
                "minimum_fit_endpoint_cosine_similarity": 0.0,
                "minimum_validation_endpoint_cosine_similarity": 0.0,
                "enforce": True,
            },
            "checks": {
                "fit_endpoint_relative_rmse": True,
                "validation_endpoint_relative_rmse": False,
                "fit_endpoint_cosine_similarity": True,
                "validation_endpoint_cosine_similarity": True,
            },
            "passed": False,
            "heldout_validation_used_for_fitting": False,
        },
        "audit": {
            "source_inputs_sha256_before": "1" * 64,
            "source_inputs_sha256_after": "1" * 64,
            "source_inputs_unchanged": True,
            "target_templates_sha256_before": "2" * 64,
            "target_templates_sha256_after": "2" * 64,
            "target_templates_unchanged": True,
            "evidence_sha256_before": "3" * 64,
            "evidence_sha256_after": "3" * 64,
            "evidence_unchanged": True,
            "case_authentication_verified": True,
            "blank_target_values_used_by_compiler": False,
            "target_templates_used_for_shape_and_dtype_only": True,
            "returned_tensors_detached": True,
        },
        "execution": {
            "closed_form_linear_solve": True,
            "deterministic_barycentric_channel_clustering": True,
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "compiled_state_sha256": "a" * 64,
    }
    nested_report = {
        **nested_body,
        "report_sha256": layer_compiler._sha(nested_body),
    }
    nested_error = layer_compiler.Lfm2CpcfShortConvValidationError(
        "validation_endpoint_relative_rmse",
        nested_report,
    )
    nested_bytes = layer_compiler._canonical_json(nested_report)

    def fail_short_conv(**_kwargs):
        with torch.no_grad():
            parameter.fill_(91.0)
        original_gradient.data = original_gradient.clone()
        layer._non_persistent_buffers_set.clear()
        parameter.requires_grad_(False)
        parameter.grad = None
        layer.train()
        raise nested_error

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_fold",
        fail_short_conv,
    )

    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilationFailure,
        match="validation_endpoint_relative_rmse",
    ) as caught:
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    failure = caught.value
    assert failure.__cause__ is nested_error
    assert failure.stage == "operator_compile"
    assert failure.operator_kind == "short_conv"
    assert failure.input_bank_sha256 == bank.state_bank_sha256
    assert failure.boundary_charts_sha256 == charts.layer_charts_sha256
    assert failure.short_conv_validation_report == nested_error.report
    assert failure.short_conv_validation_report is not nested_error.report
    assert failure.short_conv_validation_report_bytes == nested_bytes
    assert failure.report["nested_short_conv_validation_report"] == nested_report
    assert (
        failure.report["nested_short_conv_validation_report_sha256"]
        == nested_report["report_sha256"]
    )
    assert (
        failure.report["nested_short_conv_validation_report_preserved_byte_for_byte"]
        is True
    )
    assert failure.rollback_proof["exact_layer_rollback_proven"] is True
    mutable_nested = failure.short_conv_validation_report
    assert mutable_nested is not None
    mutable_nested["behavior_gate"]["passed"] = True
    assert failure.short_conv_validation_report["behavior_gate"]["passed"] is False
    mutable_report = failure.report
    mutable_report["nested_short_conv_validation_report"]["behavior_gate"]["passed"] = (
        True
    )
    assert (
        failure.report["nested_short_conv_validation_report"]["behavior_gate"]["passed"]
        is False
    )
    forged_report = failure.report
    forged_report["layer_result_issued"] = True
    forged_body = dict(forged_report)
    forged_body.pop("report_sha256")
    forged_report["report_sha256"] = layer_compiler._sha(forged_body)
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerCompilerError,
        match="claims|context",
    ):
        layer_compiler.Lfm2CpcfLayerCompilationFailure(
            "forged success",
            stage=failure.stage,
            layer_index=failure.layer_index,
            operator_kind=failure.operator_kind,
            input_bank_sha256=failure.input_bank_sha256,
            input_tensor_sha256=failure.input_tensor_sha256,
            boundary_charts_sha256=(failure.boundary_charts_sha256),
            boundary_charts_payload_sha256=(failure.boundary_charts_payload_sha256),
            rollback_proof=failure.rollback_proof,
            report=forged_report,
            short_conv_validation_report=(failure.short_conv_validation_report),
            short_conv_validation_report_bytes=(
                failure.short_conv_validation_report_bytes
            ),
        )
    forged_nested_body = {
        **nested_body,
        "promotion_claimed": True,
    }
    forged_nested_report = {
        **forged_nested_body,
        "report_sha256": layer_compiler._sha(forged_nested_body),
    }
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerInstallError,
        match="success claim",
    ):
        layer_compiler._verified_short_conv_validation_failure(
            layer_compiler.Lfm2CpcfShortConvValidationError(
                "forged nested success",
                forged_nested_report,
            )
        )
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert {name: id(value) for name, value in layer.named_parameters()} == target_ids
    assert parameter.requires_grad is True
    assert parameter.grad is original_gradient
    assert (
        int(original_gradient.untyped_storage().data_ptr()) == gradient_storage_pointer
    )
    assert torch.equal(parameter.grad, torch.full_like(parameter, 7.0))
    assert "rotary_inv_freq" in layer._non_persistent_buffers_set
    assert layer.training is False


def test_layer_failure_detects_live_evidence_mapping_replacement(
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    before = copy.deepcopy(layer.state_dict())
    evidence = bank.phases[0].parent_evidence.layers[0]

    def replace_evidence(**_kwargs):
        replacement = dict(evidence.tensors)
        replacement["operator.input"] = replacement["operator.input"].clone()
        object.__setattr__(
            evidence,
            "tensors",
            MappingProxyType(replacement),
        )
        with torch.no_grad():
            layer.conv.in_proj.weight.fill_(9.0)
        raise RuntimeError("evidence mapping replaced")

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_fold",
        replace_evidence,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerInstallError,
        match="rollback could not be proven",
    ):
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_layer_failure_detects_boundary_chart_report_mutation(
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    before = copy.deepcopy(layer.state_dict())

    def mutate_chart(**_kwargs):
        object.__setattr__(
            charts.post_operator,
            "report",
            MappingProxyType(
                {
                    "report_sha256": charts.post_operator.report["report_sha256"],
                    "promotion_claimed": True,
                }
            ),
        )
        with torch.no_grad():
            layer.conv.in_proj.weight.fill_(9.0)
        raise RuntimeError("chart report changed")

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_fold",
        mutate_chart,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerInstallError,
        match="rollback could not be proven",
    ):
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_layer_failure_detects_phase_token_tensor_mutation(
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    phase = bank.phases[0]

    def mutate_tokens(**_kwargs):
        phase.input_ids.add_(1)
        raise RuntimeError("phase tokens changed")

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_fold",
        mutate_tokens,
    )
    with pytest.raises(
        layer_compiler.Lfm2CpcfLayerInstallError,
        match="rollback could not be proven",
    ):
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )


def test_layer_keyboard_interrupt_restores_before_propagating(
    monkeypatch,
) -> None:
    source, target, bank, charts = _fixture("short_conv")
    layer = target.model.layers[0]
    parameter = layer.conv.in_proj.weight
    before = parameter.detach().clone()

    def interrupt(**_kwargs):
        with torch.no_grad():
            parameter.fill_(9.0)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_cpcf_short_conv_fold",
        interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    torch.testing.assert_close(parameter, before, rtol=0, atol=0)


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
    rotated = normalized * cosine.unsqueeze(1) + _rotate_half(
        normalized
    ) * sine.unsqueeze(1)

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
    reconstructed = reconstructed * cosine.unsqueeze(1) + _rotate_half(
        reconstructed
    ) * sine.unsqueeze(1)
    torch.testing.assert_close(reconstructed, rotated, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(
        recovered[:, :, -2:, :],
        linear[:, :, -2:, :],
        rtol=0,
        atol=0,
    )
