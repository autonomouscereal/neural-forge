from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_sequence_effective_attention_fold as sequence_fold  # noqa: E402
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_attention_value_output_fold,
    compile_lfm2_structural_residual_attention_fold,
)


DTYPE = torch.float64
SOURCE_HIDDEN = 32
TARGET_HIDDEN = 8
SOURCE_QUERY_HEADS = 8
SOURCE_KV_HEADS = 4
TARGET_QUERY_HEADS = 2
TARGET_KV_HEADS = 1
HEAD_DIM = 4
PREFILL_LENGTH = 6
DECODE_STEPS = 6


def _rrmse(prediction, target) -> float:
    error = torch.mean((prediction - target).square()).sqrt()
    scale = torch.mean(target.square()).sqrt().clamp_min(1e-15)
    return float((error / scale).item())


def _make_parent(seed: int):
    generator = torch.Generator().manual_seed(seed)
    decoder = torch.linalg.qr(
        torch.randn(
            SOURCE_HIDDEN,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        mode="reduced",
    ).Q.contiguous()
    encoder = decoder.T.contiguous()
    base_query = torch.randn(
        TARGET_QUERY_HEADS,
        HEAD_DIM,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    ) / math.sqrt(TARGET_HIDDEN)
    base_key = torch.randn(
        HEAD_DIM,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    ) / math.sqrt(TARGET_HIDDEN)
    base_value = torch.randn(
        HEAD_DIM,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    ) / math.sqrt(TARGET_HIDDEN)
    base_output = torch.randn(
        TARGET_QUERY_HEADS,
        TARGET_HIDDEN,
        HEAD_DIM,
        generator=generator,
        dtype=DTYPE,
    ) / math.sqrt(HEAD_DIM)
    queries = []
    keys = []
    values = []
    outputs = []
    for _group in range(SOURCE_KV_HEADS):
        keys.append(
            (
                base_key
                + 0.18
                * torch.randn(
                    base_key.shape,
                    generator=generator,
                    dtype=DTYPE,
                )
            )
            @ encoder
        )
        values.append(
            (
                base_value
                + 0.32
                * torch.randn(
                    base_value.shape,
                    generator=generator,
                    dtype=DTYPE,
                )
            )
            @ encoder
        )
        for offset in range(2):
            queries.append(
                (
                    base_query[offset]
                    + 0.16
                    * torch.randn(
                        base_query[offset].shape,
                        generator=generator,
                        dtype=DTYPE,
                    )
                )
                @ encoder
            )
            outputs.append(
                decoder
                @ (
                    base_output[offset]
                    + 0.24
                    * torch.randn(
                        base_output[offset].shape,
                        generator=generator,
                        dtype=DTYPE,
                    )
                )
                / SOURCE_KV_HEADS
            )
    transition = torch.randn(
        TARGET_HIDDEN,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    transition = (
        0.75
        * transition
        / torch.linalg.matrix_norm(transition, ord=2).clamp_min(1e-12)
    )
    return SimpleNamespace(
        decoder=decoder,
        encoder=encoder,
        query=torch.cat(queries, dim=0).contiguous(),
        key=torch.cat(keys, dim=0).contiguous(),
        value=torch.cat(values, dim=0).contiguous(),
        output=torch.cat(outputs, dim=1).contiguous(),
        transition=transition.contiguous(),
    )


def _parent_case(parent, history, *, query_start: int):
    batch, key_length, _hidden = history.shape
    query_input = history[:, query_start:]
    query_length = int(query_input.shape[1])
    parent_hidden = history @ parent.decoder.T
    query = (
        (parent_hidden[:, query_start:] @ parent.query.T)
        .reshape(batch, query_length, SOURCE_QUERY_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    key = (
        (parent_hidden @ parent.key.T)
        .reshape(batch, key_length, SOURCE_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    value = (
        (parent_hidden @ parent.value.T)
        .reshape(batch, key_length, SOURCE_KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    query_positions = torch.arange(query_start, key_length, dtype=torch.long)
    key_positions = torch.arange(key_length, dtype=torch.long)
    key_mask = (
        (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .contiguous()
    )
    repeated_key = key.repeat_interleave(
        SOURCE_QUERY_HEADS // SOURCE_KV_HEADS,
        dim=1,
    )
    scores = query @ repeated_key.transpose(-2, -1) / math.sqrt(HEAD_DIM)
    probability = torch.softmax(
        scores.masked_fill(~key_mask.unsqueeze(1), -1e30),
        dim=-1,
        dtype=DTYPE,
    )
    context = probability @ value.repeat_interleave(
        SOURCE_QUERY_HEADS // SOURCE_KV_HEADS,
        dim=1,
    )
    flattened = context.permute(0, 2, 1, 3).reshape(
        batch,
        query_length,
        SOURCE_QUERY_HEADS * HEAD_DIM,
    )
    target_delta = (flattened @ parent.output.T) @ parent.encoder.T
    endpoint = query_input + target_delta
    contributions = []
    for head in range(SOURCE_QUERY_HEADS):
        block = parent.output[:, head * HEAD_DIM : (head + 1) * HEAD_DIM]
        contributions.append(context[:, head] @ block.T @ parent.encoder.T)
    case = Lfm2CachedAttentionFoldCase(
        target_query_input=query_input,
        target_key_value_input=history,
        parent_query_linear=query,
        parent_key_linear=key,
        parent_value=value,
        parent_query=query,
        parent_key=key,
        parent_context=context,
        parent_post_attention_residual_target_space=endpoint,
        target_pre_attention_residual=query_input,
        key_mask=key_mask,
        parent_contribution=torch.stack(contributions, dim=1),
    )
    return case, endpoint


def _next_latent(last_input, last_endpoint, forcing, transition):
    return torch.tanh(
        0.58 * last_endpoint
        + 0.27 * (last_input @ transition.T)
        + 0.15 * forcing
    )


def _equal_phase_weights(cases):
    query_rows = [
        int(case.target_query_input.shape[0] * case.target_query_input.shape[1])
        for case in cases
    ]
    key_rows = [
        int(
            case.target_key_value_input.shape[0]
            * case.target_key_value_input.shape[1]
        )
        for case in cases
    ]
    query_total = sum(query_rows)
    key_total = sum(key_rows)
    phase_count = len(cases)
    return tuple(
        replace(
            case,
            query_row_sqrt_weights=torch.full(
                case.target_query_input.shape[:2],
                math.sqrt(query_total / (phase_count * query_count)),
                dtype=DTYPE,
            ),
            key_value_row_sqrt_weights=torch.full(
                case.target_key_value_input.shape[:2],
                math.sqrt(key_total / (phase_count * key_count)),
                dtype=DTYPE,
            ),
        )
        for case, query_count, key_count in zip(
            cases,
            query_rows,
            key_rows,
            strict=True,
        )
    )


def _make_dataset(parent, *, seed: int, batch_size: int):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.empty(
        batch_size,
        PREFILL_LENGTH,
        TARGET_HIDDEN,
        dtype=DTYPE,
    )
    initial[:, 0] = torch.randn(
        batch_size,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    noise = torch.randn(
        batch_size,
        PREFILL_LENGTH - 1,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    for index in range(1, PREFILL_LENGTH):
        initial[:, index] = torch.tanh(
            initial[:, index - 1] @ parent.transition.T
            + 0.35 * noise[:, index - 1]
        )
    forcing = torch.randn(
        batch_size,
        DECODE_STEPS,
        TARGET_HIDDEN,
        generator=generator,
        dtype=DTYPE,
    )
    prefill_case, endpoint = _parent_case(parent, initial, query_start=0)
    cases = [prefill_case]
    history = initial.clone()
    last_endpoint = endpoint[:, -1]
    decode_inputs = []
    decode_endpoints = []
    for step in range(DECODE_STEPS):
        next_input = _next_latent(
            history[:, -1],
            last_endpoint,
            forcing[:, step],
            parent.transition,
        )
        history = torch.cat((history, next_input.unsqueeze(1)), dim=1)
        case, endpoint = _parent_case(
            parent,
            history,
            query_start=int(history.shape[1]) - 1,
        )
        cases.append(case)
        decode_inputs.append(next_input)
        decode_endpoints.append(endpoint[:, 0])
        last_endpoint = endpoint[:, 0]
    return SimpleNamespace(
        cases=_equal_phase_weights(tuple(cases)),
        initial_history=initial,
        forcing=forcing,
        decode_inputs=tuple(decode_inputs),
        decode_endpoints=tuple(decode_endpoints),
    )


def _child_forward(state, case, *, query_heads: int, key_value_heads: int):
    batch, query_length, _hidden = case.target_query_input.shape
    key_length = int(case.target_key_value_input.shape[1])
    query = (
        (case.target_query_input @ state["query"].T)
        .reshape(batch, query_length, query_heads, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    key = (
        (case.target_key_value_input @ state["key"].T)
        .reshape(batch, key_length, key_value_heads, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    value = (
        (case.target_key_value_input @ state["value"].T)
        .reshape(batch, key_length, key_value_heads, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )
    repeats = query_heads // key_value_heads
    scores = (
        query @ key.repeat_interleave(repeats, dim=1).transpose(-2, -1)
    ) / math.sqrt(HEAD_DIM)
    if case.key_mask is not None:
        scores = scores.masked_fill(~case.key_mask.unsqueeze(1), -1e30)
    probability = torch.softmax(scores, dim=-1, dtype=DTYPE)
    context_heads = probability @ value.repeat_interleave(repeats, dim=1)
    context = context_heads.permute(0, 2, 1, 3).reshape(
        batch,
        query_length,
        query_heads * HEAD_DIM,
    )
    endpoint = case.target_pre_attention_residual + context @ state["output"].T
    return SimpleNamespace(context=context, endpoint=endpoint)


def _state(artifact):
    return {
        "query": artifact.query_weight.to(DTYPE),
        "key": artifact.key_weight.to(DTYPE),
        "value": artifact.value_weight.to(DTYPE),
        "output": artifact.output_weight.to(DTYPE),
    }


def _teacher_forced_endpoint(state, dataset) -> float:
    predictions = []
    targets = []
    for case in dataset.cases:
        predictions.append(
            _child_forward(
                state,
                case,
                query_heads=TARGET_QUERY_HEADS,
                key_value_heads=TARGET_KV_HEADS,
            ).endpoint.reshape(-1, TARGET_HIDDEN)
        )
        targets.append(
            case.parent_post_attention_residual_target_space.reshape(
                -1,
                TARGET_HIDDEN,
            )
        )
    return _rrmse(torch.cat(predictions), torch.cat(targets))


def _closed_loop_endpoint(state, dataset, parent) -> float:
    history = dataset.initial_history.clone()
    prefill_case = replace(
        dataset.cases[0],
        target_query_input=history,
        target_key_value_input=history,
        target_pre_attention_residual=history,
    )
    last_endpoint = _child_forward(
        state,
        prefill_case,
        query_heads=TARGET_QUERY_HEADS,
        key_value_heads=TARGET_KV_HEADS,
    ).endpoint[:, -1]
    errors = []
    for step in range(DECODE_STEPS):
        next_input = _next_latent(
            history[:, -1],
            last_endpoint,
            dataset.forcing[:, step],
            parent.transition,
        )
        history = torch.cat((history, next_input.unsqueeze(1)), dim=1)
        case = replace(
            dataset.cases[step + 1],
            target_query_input=next_input.unsqueeze(1),
            target_key_value_input=history,
            target_pre_attention_residual=next_input.unsqueeze(1),
            key_mask=torch.ones(
                int(history.shape[0]),
                1,
                int(history.shape[1]),
                dtype=torch.bool,
            ),
        )
        last_endpoint = _child_forward(
            state,
            case,
            query_heads=TARGET_QUERY_HEADS,
            key_value_heads=TARGET_KV_HEADS,
        ).endpoint[:, 0]
        errors.append(_rrmse(last_endpoint, dataset.decode_endpoints[step]))
    return sum(errors) / len(errors)


def _templates(seed: int, *, query_heads: int, key_value_heads: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        "query": torch.randn(
            query_heads * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "key": torch.randn(
            key_value_heads * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "value": torch.randn(
            key_value_heads * HEAD_DIM,
            TARGET_HIDDEN,
            generator=generator,
            dtype=DTYPE,
        ),
        "output": torch.randn(
            TARGET_HIDDEN,
            query_heads * HEAD_DIM,
            generator=generator,
            dtype=DTYPE,
        ),
    }


def _compile_producer(
    parent,
    fit,
    templates,
    *,
    query_heads: int,
    key_value_heads: int,
):
    common = {
        "source_query_weight": parent.query,
        "source_key_weight": parent.key,
        "source_value_weight": parent.value,
        "source_output_weight": parent.output,
        "target_query_weight_template": templates["query"],
        "target_key_weight_template": templates["key"],
        "target_value_weight_template": templates["value"],
        "target_output_weight_template": templates["output"],
        "cases": fit.cases,
        "head_dim": HEAD_DIM,
        "residual_assignment_sha256": "a" * 64,
        "ridge": 1e-8,
        "cluster_ridge": 1e-8,
        "max_prior_delta_ratio": None,
    }
    producer = compile_lfm2_structural_attention_producer_fold(**common)
    assert producer.topology.target_query_heads == query_heads
    assert producer.topology.target_key_value_heads == key_value_heads
    return producer, common


@pytest.fixture(scope="module")
def sequence_frontier():
    seed = 1101
    parent = _make_parent(seed)
    fit = _make_dataset(parent, seed=seed + 10_000, batch_size=28)
    heldout = _make_dataset(parent, seed=seed + 30_000, batch_size=28)
    templates = _templates(
        seed + 40_000,
        query_heads=TARGET_QUERY_HEADS,
        key_value_heads=TARGET_KV_HEADS,
    )
    producer, common = _compile_producer(
        parent,
        fit,
        templates,
        query_heads=TARGET_QUERY_HEADS,
        key_value_heads=TARGET_KV_HEADS,
    )
    direct = compile_lfm2_structural_residual_attention_fold(**common)
    producer_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": producer.value_weight.to(DTYPE),
        "output": templates["output"],
    }
    actual_contexts = tuple(
        _child_forward(
            producer_state,
            case,
            query_heads=TARGET_QUERY_HEADS,
            key_value_heads=TARGET_KV_HEADS,
        ).context
        for case in fit.cases
    )
    split = compile_lfm2_structural_attention_value_output_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        actual_child_contexts=actual_contexts,
        cases=fit.cases,
        live_value_weight=producer.value_weight,
        value_weight_prior=producer.value_weight,
        output_weight_prior=templates["output"],
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=None,
        iterations=2,
    )
    compiled = sequence_fold.compile_lfm2_fixed_qk_sequence_effective_attention_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        fit_cases=fit.cases,
    )
    repeat = sequence_fold.compile_lfm2_fixed_qk_sequence_effective_attention_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        fit_cases=fit.cases,
    )
    return SimpleNamespace(
        parent=parent,
        fit=fit,
        heldout=heldout,
        templates=templates,
        producer=producer,
        direct=direct,
        split=split,
        compiled=compiled,
        repeat=repeat,
    )


def test_fixed_qk_sequence_effective_fold_wins_heldout_without_training(
    sequence_frontier,
) -> None:
    values = sequence_frontier
    direct_state = _state(values.direct)
    split_state = {
        "query": values.producer.query_weight.to(DTYPE),
        "key": values.producer.key_weight.to(DTYPE),
        "value": values.split.value_weight.to(DTYPE),
        "output": values.split.output_weight.to(DTYPE),
    }
    compiled_state = {
        "query": values.compiled.producer_artifact.query_weight.to(DTYPE),
        "key": values.compiled.producer_artifact.key_weight.to(DTYPE),
        "value": values.compiled.value_weight.to(DTYPE),
        "output": values.compiled.output_weight.to(DTYPE),
    }
    direct_endpoint = _teacher_forced_endpoint(direct_state, values.heldout)
    split_endpoint = _teacher_forced_endpoint(split_state, values.heldout)
    compiled_endpoint = _teacher_forced_endpoint(compiled_state, values.heldout)
    direct_drift = _closed_loop_endpoint(
        direct_state,
        values.heldout,
        values.parent,
    )
    split_drift = _closed_loop_endpoint(
        split_state,
        values.heldout,
        values.parent,
    )
    compiled_drift = _closed_loop_endpoint(
        compiled_state,
        values.heldout,
        values.parent,
    )

    assert compiled_endpoint < min(direct_endpoint, split_endpoint)
    assert compiled_drift < min(direct_drift, split_drift)
    assert values.compiled.compiled_state_sha256 == (
        values.repeat.compiled_state_sha256
    )
    assert values.compiled.report["fit_only_compilation"] is True
    assert values.compiled.report["compiled_from_splits"] == ["fit"]
    assert values.compiled.report["selection_case_count_consumed"] == 0
    assert values.compiled.report["validation_case_count_consumed"] == 0
    assert values.compiled.report["fixed_query_key_sha256"] == (
        values.producer.score_state_sha256
    )
    assert values.compiled.report["source_logical_parameters"] == 3072
    assert values.compiled.report["target_logical_parameters"] == 192
    assert values.compiled.report["parameter_reduction_fraction"] == pytest.approx(
        0.9375
    )
    assert values.compiled.verify_authentication() is True
    assert all(
        values.compiled.report["execution"][name] == 0
        for name in (
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
        )
    )


def test_sequence_effective_authentication_topology_and_install_rollback(
    sequence_frontier,
    monkeypatch,
) -> None:
    compiled = sequence_frontier.compiled
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="singular and fixed",
    ):
        replace(
            compiled.policy,
            ridge_scale=2e-4,
            authentication_sha256=compiled.policy.authentication_sha256,
        )
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="state authentication failed",
    ):
        replace(
            compiled,
            value_weight=torch.zeros_like(compiled.value_weight),
        )
    tampered_body = dict(compiled.report)
    tampered_body.pop("report_sha256")
    tampered_body["fit_only_compilation"] = False
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="artifact seal is invalid",
    ):
        replace(
            compiled,
            report=sequence_fold._seal(tampered_body),
        )
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="templates differ",
    ):
        sequence_fold.compile_lfm2_fixed_qk_sequence_effective_attention_fold(
            producer=sequence_frontier.producer,
            target_value_weight_template=torch.zeros(
                HEAD_DIM + 1,
                TARGET_HIDDEN,
                dtype=DTYPE,
            ),
            target_output_weight_template=sequence_frontier.templates["output"],
            fit_cases=sequence_frontier.fit.cases,
        )

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(
                TARGET_HIDDEN,
                TARGET_QUERY_HEADS * HEAD_DIM,
                bias=False,
                dtype=DTYPE,
            )
            self.k_proj = torch.nn.Linear(
                TARGET_HIDDEN,
                TARGET_KV_HEADS * HEAD_DIM,
                bias=False,
                dtype=DTYPE,
            )
            self.v_proj = torch.nn.Linear(
                TARGET_HIDDEN,
                TARGET_KV_HEADS * HEAD_DIM,
                bias=False,
                dtype=DTYPE,
            )
            self.out_proj = torch.nn.Linear(
                TARGET_QUERY_HEADS * HEAD_DIM,
                TARGET_HIDDEN,
                bias=False,
                dtype=DTYPE,
            )

    target = Attention()
    weights = {
        name: getattr(getattr(target, module), "weight")
        for name, module in (
            ("q_proj.weight", "q_proj"),
            ("k_proj.weight", "k_proj"),
            ("v_proj.weight", "v_proj"),
            ("out_proj.weight", "out_proj"),
        )
    }
    identities = {name: id(value) for name, value in weights.items()}
    before = {name: value.detach().clone() for name, value in weights.items()}
    original_copy = sequence_fold._copy_parameter
    copy_count = 0

    def fail_third_copy(parameter, value):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 3:
            raise RuntimeError("injected V/O installation failure")
        return original_copy(parameter, value)

    monkeypatch.setattr(sequence_fold, "_copy_parameter", fail_third_copy)
    with pytest.raises(RuntimeError, match="injected V/O"):
        sequence_fold.install_lfm2_fixed_qk_sequence_effective_attention_fold(
            target,
            compiled,
        )
    assert all(id(weights[name]) == identities[name] for name in weights)
    assert all(torch.equal(weights[name], before[name]) for name in weights)

    monkeypatch.setattr(sequence_fold, "_copy_parameter", original_copy)
    report = sequence_fold.install_lfm2_fixed_qk_sequence_effective_attention_fold(
        target,
        compiled,
    )
    assert report.accepted is True
    assert report.report["fixed_query_key_installed_exactly"] is True
    assert all(id(weights[name]) == identities[name] for name in weights)


def test_sequence_effective_fold_factors_each_gqa_group() -> None:
    seed = 2202
    parent = _make_parent(seed)
    fit = _make_dataset(parent, seed=seed + 10_000, batch_size=8)
    templates = _templates(
        seed + 40_000,
        query_heads=4,
        key_value_heads=2,
    )
    producer, _common = _compile_producer(
        parent,
        fit,
        templates,
        query_heads=4,
        key_value_heads=2,
    )
    compiled = sequence_fold.compile_lfm2_fixed_qk_sequence_effective_attention_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        fit_cases=fit.cases,
    )

    assert tuple(compiled.value_weight.shape) == (8, TARGET_HIDDEN)
    assert tuple(compiled.output_weight.shape) == (TARGET_HIDDEN, 16)
    assert [value["group"] for value in compiled.report["shared_value_svd"]] == [
        0,
        1,
    ]
    assert len(compiled.report["value_subspace_anchor"]) == 2
    assert compiled.report["weighted_fit_relative_rmse_target_dtype"] == pytest.approx(
        compiled.report["weighted_fit_relative_rmse_fp64_after_anchor"],
        abs=1e-12,
    )
    assert compiled.verify_authentication() is True


@pytest.mark.parametrize("seed", (1101, 2202, 3303))
def test_streamed_statistics_pcg_path_beats_both_controls_on_heldout_sequences(
    seed,
    monkeypatch,
) -> None:
    # Force the exact production scaling branch on the already-proven small
    # fixture so its quality is tested independently of the dense small path.
    monkeypatch.setattr(
        sequence_fold,
        "_MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS",
        64,
    )
    parent = _make_parent(seed)
    fit = _make_dataset(parent, seed=seed + 10_000, batch_size=28)
    heldout = _make_dataset(parent, seed=seed + 30_000, batch_size=28)
    templates = _templates(
        seed + 40_000,
        query_heads=TARGET_QUERY_HEADS,
        key_value_heads=TARGET_KV_HEADS,
    )
    producer, common = _compile_producer(
        parent,
        fit,
        templates,
        query_heads=TARGET_QUERY_HEADS,
        key_value_heads=TARGET_KV_HEADS,
    )
    direct = compile_lfm2_structural_residual_attention_fold(**common)
    producer_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": producer.value_weight.to(DTYPE),
        "output": templates["output"],
    }
    actual_contexts = tuple(
        _child_forward(
            producer_state,
            case,
            query_heads=TARGET_QUERY_HEADS,
            key_value_heads=TARGET_KV_HEADS,
        ).context
        for case in fit.cases
    )
    split = compile_lfm2_structural_attention_value_output_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        actual_child_contexts=actual_contexts,
        cases=fit.cases,
        live_value_weight=producer.value_weight,
        value_weight_prior=producer.value_weight,
        output_weight_prior=templates["output"],
        ridge=1e-8,
        context_ridge=1e-6,
        max_prior_delta_ratio=None,
        iterations=2,
    )
    compiled = sequence_fold.compile_lfm2_fixed_qk_sequence_effective_attention_fold(
        producer=producer,
        target_value_weight_template=templates["value"],
        target_output_weight_template=templates["output"],
        fit_cases=fit.cases,
    )
    direct_state = _state(direct)
    split_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": split.value_weight.to(DTYPE),
        "output": split.output_weight.to(DTYPE),
    }
    compiled_state = {
        "query": producer.query_weight.to(DTYPE),
        "key": producer.key_weight.to(DTYPE),
        "value": compiled.value_weight.to(DTYPE),
        "output": compiled.output_weight.to(DTYPE),
    }

    compiled_endpoint = _teacher_forced_endpoint(compiled_state, heldout)
    compiled_drift = _closed_loop_endpoint(compiled_state, heldout, parent)
    assert compiled_endpoint < min(
        _teacher_forced_endpoint(direct_state, heldout),
        _teacher_forced_endpoint(split_state, heldout),
    )
    assert compiled_drift < min(
        _closed_loop_endpoint(direct_state, heldout, parent),
        _closed_loop_endpoint(split_state, heldout, parent),
    )
    assert compiled.report["solver_strategy"] == (
        "streamed_cross_head_statistics_pcg"
    )
    assert compiled.report["accepted_als_iterations"] == 2
    assert compiled.report["complete_operator_dense_factorization_avoided"] is True
    assert (
        compiled.report["forbidden_value_coefficient_tensor_materialized"]
        is False
    )
    assert compiled.report["forbidden_hidden_head_dim_gram_materialized"] is False


def test_streamed_statistics_and_pcg_match_explicit_small_oracle(
    sequence_frontier,
    monkeypatch,
) -> None:
    values = sequence_frontier
    statistics, _report, _explicit = (
        sequence_fold._accumulate_streamed_sequence_statistics(
            producer=values.producer,
            cases=values.fit.cases,
            query_norm_weight=None,
            key_norm_weight=None,
            norm_epsilon=1e-6,
        )
    )
    attended = []
    targets = []
    for case_index, case in enumerate(values.fit.cases):
        normalized = sequence_fold._normalize_sequence_case(
            case,
            case_index=case_index,
            producer=values.producer,
            query_norm_weight=None,
            key_norm_weight=None,
            norm_epsilon=1e-6,
        )
        batch = int(normalized["batch"])
        query_length = int(normalized["query_length"])
        weight = case.query_row_sqrt_weights.reshape(-1, 1).to(DTYPE)
        attended.append(
            normalized["weighted_inputs"]
            .permute(0, 2, 1, 3)
            .reshape(
                batch * query_length,
                TARGET_QUERY_HEADS,
                TARGET_HIDDEN,
            )
            * weight.unsqueeze(1)
        )
        targets.append(normalized["endpoint"] * weight)
    attended = torch.cat(attended)
    target = torch.cat(targets)
    flat = attended.reshape(
        int(attended.shape[0]),
        TARGET_QUERY_HEADS * TARGET_HIDDEN,
    )
    explicit_gram = flat.T @ flat
    explicit_right = flat.T @ target
    streamed_gram = (
        statistics["cross_head_gram"]
        .permute(0, 2, 1, 3)
        .reshape(explicit_gram.shape)
    )
    streamed_right = statistics["endpoint_right"].reshape(explicit_right.shape)
    torch.testing.assert_close(streamed_gram, explicit_gram, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        streamed_right,
        explicit_right,
        rtol=1e-12,
        atol=1e-12,
    )

    monkeypatch.setattr(
        sequence_fold,
        "_MAXIMUM_DENSE_RIDGE_SYSTEM_ELEMENTS",
        64,
    )
    pcg_solution, pcg_report = (
        sequence_fold._fit_streamed_global_effective_operator(
            statistics=statistics,
            topology=values.producer.topology,
            policy=sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy(),
        )
    )
    policy = sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy()
    scale = float(torch.trace(explicit_gram)) / int(explicit_gram.shape[0])
    ridge = max(policy.absolute_ridge_floor, policy.ridge_scale * scale)
    dense_solution = torch.linalg.solve(
        explicit_gram
        + ridge * torch.eye(int(explicit_gram.shape[0]), dtype=DTYPE),
        explicit_right,
    ).reshape(pcg_solution.shape)
    torch.testing.assert_close(
        pcg_solution,
        dense_solution,
        rtol=1e-8,
        atol=1e-9,
    )
    assert pcg_report["relative_normal_residual"] <= 1e-8

    value_factor, output_blocks, _svd = (
        sequence_fold._initial_shared_value_factors(
            pcg_solution,
            target_key_value_heads=TARGET_KV_HEADS,
            heads_per_group=TARGET_QUERY_HEADS // TARGET_KV_HEADS,
            head_dim=HEAD_DIM,
        )
    )
    explicit_output, _features, _explicit_output_report = (
        sequence_fold._fit_output(
            attended,
            target,
            value_factor,
            heads_per_group=TARGET_QUERY_HEADS // TARGET_KV_HEADS,
            policy=policy,
            role="explicit_output_oracle",
        )
    )
    streamed_output, _streamed_output_report = (
        sequence_fold._fit_output_from_sequence_statistics(
            statistics=statistics,
            topology=values.producer.topology,
            value_factor=value_factor,
            policy=policy,
            role="streamed_output_oracle",
        )
    )
    torch.testing.assert_close(
        streamed_output,
        explicit_output,
        rtol=1e-10,
        atol=1e-10,
    )
    explicit_value, _explicit_value_reports = sequence_fold._block_value_sweep(
        attended,
        target,
        value_factor,
        explicit_output,
        heads_per_group=TARGET_QUERY_HEADS // TARGET_KV_HEADS,
        policy=policy,
        iteration=1,
    )
    streamed_value, _streamed_value_reports = (
        sequence_fold._structured_value_pcg_sweep(
            statistics=statistics,
            topology=values.producer.topology,
            value_factor=value_factor,
            output_blocks=explicit_output,
            policy=policy,
            iteration=1,
        )
    )
    torch.testing.assert_close(
        streamed_value,
        explicit_value,
        rtol=1e-6,
        atol=1e-7,
    )
    value_ridge = _explicit_value_reports[0]["ridge"]
    explicit_objective = float(
        (
            sequence_fold._prediction(
                attended,
                explicit_value,
                explicit_output,
                heads_per_group=TARGET_QUERY_HEADS // TARGET_KV_HEADS,
            )
            - target
        )
        .square()
        .sum()
        .item()
        + value_ridge * explicit_value.square().sum().item()
    )
    streamed_objective = float(
        (
            sequence_fold._prediction(
                attended,
                streamed_value,
                explicit_output,
                heads_per_group=TARGET_QUERY_HEADS // TARGET_KV_HEADS,
            )
            - target
        )
        .square()
        .sum()
        .item()
        + value_ridge * streamed_value.square().sum().item()
    )
    assert abs(streamed_objective - explicit_objective) / explicit_objective < 1e-8


def test_h776_q10_production_plan_bounds_all_dense_systems_and_tb_tensor() -> None:
    plan = sequence_fold._sequence_effective_solver_plan(
        fit_rows=40_000,
        target_query_heads=10,
        target_hidden_size=776,
        target_output_size=776,
        head_dim=64,
    )

    assert plan["strategy"] == "streamed_cross_head_statistics_pcg"
    assert plan["complete_operator_feature_count"] == 7_760
    assert plan["complete_operator_dense_system_elements"] == 7_760**2
    assert plan["complete_operator_dense_factorization_avoided"] is True
    assert plan["largest_executed_dense_system_dimension"] == 776
    assert plan["largest_executed_dense_system_elements"] == 776**2
    assert plan["estimated_streamed_statistics_bytes"] < 600 * 1024**2
    assert plan["forbidden_value_coefficient_elements_per_kv_group"] > 1.5e12
    assert plan["forbidden_value_coefficient_bytes_per_kv_group"] > 12e12
    assert plan["forbidden_value_coefficient_materialized"] is False
    assert plan["forbidden_value_dense_gram_bytes"] > 18e9
    assert plan["forbidden_value_dense_gram_materialized"] is False
    short_trace_plan = sequence_fold._sequence_effective_solver_plan(
        fit_rows=64,
        target_query_heads=10,
        target_hidden_size=776,
        target_output_size=776,
        head_dim=64,
    )
    assert short_trace_plan["strategy"] == (
        "streamed_cross_head_statistics_pcg"
    )


def test_pcg_fails_closed_when_authenticated_residual_cap_is_not_met(
    monkeypatch,
) -> None:
    matrix = torch.tensor(
        [[7.0, 2.0], [2.0, 3.0]],
        dtype=DTYPE,
    )
    right = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]],
        dtype=DTYPE,
    )
    monkeypatch.setattr(sequence_fold, "_PCG_MAXIMUM_ITERATIONS", 1)
    monkeypatch.setattr(
        sequence_fold,
        "_PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE",
        1e-16,
    )

    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="failed its authenticated residual tolerance",
    ):
        sequence_fold._pcg_multi_rhs(
            right=right,
            matvec=lambda value: matrix @ value,
            diagonal=torch.diagonal(matrix),
            role="ill_conditioned_test_oracle",
        )


def _legacy_pcg_single_tensor(*, right, matvec, diagonal, initial, role):
    """Exact pre-restart implementation used to prove fast-path stability."""

    solution = initial.clone()
    residual = right - matvec(solution)
    right_norm = torch.linalg.vector_norm(right).clamp_min(
        torch.finfo(torch.float64).eps
    )
    initial_relative = float(
        (torch.linalg.vector_norm(residual) / right_norm).item()
    )
    if (
        initial_relative
        <= sequence_fold._PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
    ):
        return solution.contiguous(), sequence_fold._seal(
            {
                "role": role,
                "solver": "deterministic_jacobi_pcg_single_tensor",
                "maximum_iterations": sequence_fold._PCG_MAXIMUM_ITERATIONS,
                "iterations_executed": 0,
                "required_relative_normal_residual": (
                    sequence_fold._PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
                ),
                "initial_relative_normal_residual": initial_relative,
                "final_relative_normal_residual": initial_relative,
                "relative_normal_residual_history": [],
                "solution_sha256": sequence_fold._tensor_sha256(solution),
            }
        )
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum()
    history = []
    iterations = 0
    for iteration in range(1, sequence_fold._PCG_MAXIMUM_ITERATIONS + 1):
        applied = matvec(direction)
        denominator = (direction * applied).sum()
        alpha = rz / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        relative = float(
            (torch.linalg.vector_norm(residual) / right_norm).item()
        )
        history.append(relative)
        iterations = iteration
        if (
            relative
            <= sequence_fold._PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        ):
            break
        next_preconditioned = residual / diagonal
        next_rz = (residual * next_preconditioned).sum()
        direction = next_preconditioned + (next_rz / rz) * direction
        preconditioned = next_preconditioned
        rz = next_rz
    body = {
        "role": role,
        "solver": "deterministic_jacobi_pcg_single_tensor",
        "maximum_iterations": sequence_fold._PCG_MAXIMUM_ITERATIONS,
        "iterations_executed": iterations,
        "required_relative_normal_residual": (
            sequence_fold._PCG_RELATIVE_NORMAL_RESIDUAL_TOLERANCE
        ),
        "initial_relative_normal_residual": initial_relative,
        "final_relative_normal_residual": history[-1],
        "relative_normal_residual_history": history,
        "solution_sha256": sequence_fold._tensor_sha256(solution),
    }
    return solution.contiguous(), sequence_fold._seal(body)


def _tridiagonal_spd_problem(size: int):
    generator = torch.Generator().manual_seed(7)
    right = torch.randn(size, generator=generator, dtype=DTYPE)
    diagonal = torch.full((size,), 2.000001, dtype=DTYPE)

    def matvec(value):
        result = 2.000001 * value.clone()
        result[1:] -= value[:-1]
        result[:-1] -= value[1:]
        return result

    return right, diagonal, matvec


def test_single_tensor_pcg_original_success_path_is_byte_identical() -> None:
    matrix = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [1.0, 4.0, 0.5],
            [0.0, 0.5, 3.0],
        ],
        dtype=DTYPE,
    )
    right = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
    diagonal = torch.diagonal(matrix)
    initial = torch.zeros_like(right)
    role = "original_lane_byte_identity"

    expected_solution, expected_report = _legacy_pcg_single_tensor(
        right=right,
        matvec=lambda value: matrix @ value,
        diagonal=diagonal,
        initial=initial,
        role=role,
    )
    observed_solution, observed_report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=lambda value: matrix @ value,
        diagonal=diagonal,
        initial=initial,
        role=role,
        restart_row_block_preconditioner=object(),
    )

    assert torch.equal(observed_solution, expected_solution)
    assert sequence_fold._canonical_json(observed_report) == (
        sequence_fold._canonical_json(expected_report)
    )
    assert observed_report["iterations_executed"] <= 64
    assert "restart_schedule" not in observed_report


def test_structured_row_blocks_match_explicit_operator_diagonal_blocks() -> None:
    generator = torch.Generator().manual_seed(6701)
    rows = 5
    block_size = 3
    heads = 2
    output_size = 4
    samples = 11
    features = torch.randn(
        samples,
        heads,
        rows,
        generator=generator,
        dtype=DTYPE,
    )
    cross = torch.einsum("nhi,nkj->hkij", features, features).contiguous()
    output_blocks = torch.randn(
        heads,
        block_size,
        output_size,
        generator=generator,
        dtype=DTYPE,
    )
    ridge = 0.125
    diagonal = torch.zeros(rows, block_size, dtype=DTYPE)
    for head in range(heads):
        for other in range(heads):
            coupling = output_blocks[other] @ output_blocks[head].T
            diagonal.add_(
                torch.diagonal(cross[head, other]).unsqueeze(1)
                * torch.diagonal(coupling).unsqueeze(0)
            )
    diagonal = diagonal + ridge
    role = "explicit_row_block_oracle"
    preconditioner = (
        sequence_fold._build_structured_value_row_block_jacobi_preconditioner(
            cross=cross,
            output_blocks=output_blocks,
            group_heads=tuple(range(heads)),
            diagonal=diagonal,
            ridge=ridge,
            role=role,
        )
    )

    def matvec(value):
        result = torch.zeros_like(value)
        for head in range(heads):
            for other in range(heads):
                coupling = output_blocks[other] @ output_blocks[head].T
                result.add_(cross[head, other] @ value @ coupling)
        return result + ridge * value

    features_count = rows * block_size
    explicit = torch.empty(features_count, features_count, dtype=DTYPE)
    for coordinate in range(features_count):
        basis = torch.zeros(features_count, dtype=DTYPE)
        basis[coordinate] = 1.0
        explicit[:, coordinate] = matvec(
            basis.reshape(rows, block_size)
        ).reshape(-1)
    for row in range(rows):
        row_slice = slice(
            row * block_size,
            (row + 1) * block_size,
        )
        torch.testing.assert_close(
            preconditioner.row_blocks[row],
            explicit[row_slice, row_slice],
            rtol=1e-12,
            atol=1e-12,
        )
    construction = dict(preconditioner.construction_report)
    assert construction["row_block_count"] == rows
    assert construction["row_block_size"] == block_size
    assert construction["scalar_diagonal_maximum_absolute_error"] <= (
        construction["scalar_diagonal_match_tolerance"]
    )
    assert construction["row_blocks_maximum_asymmetry"] <= (
        construction["row_blocks_symmetry_tolerance"]
    )
    assert construction["full_operator_materialized"] is False
    assert construction["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in construction.items()
            if key != "report_sha256"
        }
    )


def _real_like_64_coordinate_row_block_problem():
    generator = torch.Generator().manual_seed(1907)
    rows = 72
    block_size = 64
    row_operator = torch.diag(
        torch.full((rows,), 2.000001, dtype=DTYPE)
    )
    row_operator += torch.diag(
        torch.full((rows - 1,), -1.0, dtype=DTYPE),
        diagonal=1,
    )
    row_operator += torch.diag(
        torch.full((rows - 1,), -1.0, dtype=DTYPE),
        diagonal=-1,
    )
    coordinates = torch.linalg.qr(
        torch.randn(
            block_size,
            block_size,
            generator=generator,
            dtype=DTYPE,
        )
    ).Q
    spectrum = torch.logspace(-3, 3, block_size, dtype=DTYPE)
    coordinate_operator = (
        coordinates * spectrum.unsqueeze(0)
    ) @ coordinates.T
    output_blocks = torch.linalg.cholesky(
        coordinate_operator
    ).reshape(1, block_size, block_size)
    cross = row_operator.reshape(1, 1, rows, rows)
    ridge = 1e-8
    diagonal = (
        torch.diagonal(row_operator).unsqueeze(1)
        * torch.diagonal(coordinate_operator).unsqueeze(0)
        + ridge
    )
    right = torch.randn(
        rows,
        block_size,
        generator=generator,
        dtype=DTYPE,
    )

    def matvec(value):
        return (
            row_operator @ value @ coordinate_operator
            + ridge * value
        )

    return cross, output_blocks, diagonal, right, matvec, ridge


def test_continuous_64_coordinate_row_block_lane_solves_real_like_system() -> None:
    (
        cross,
        output_blocks,
        diagonal,
        right,
        matvec,
        ridge,
    ) = _real_like_64_coordinate_row_block_problem()
    role = "real_like_h776_row_block_oracle"
    preconditioner = (
        sequence_fold._build_structured_value_row_block_jacobi_preconditioner(
            cross=cross,
            output_blocks=output_blocks,
            group_heads=(0,),
            diagonal=diagonal,
            ridge=ridge,
            role=role,
        )
    )

    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as scalar_raised:
        sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=torch.zeros_like(right),
            role="real_like_scalar_control",
        )
    scalar_failure = dict(scalar_raised.value.failure_report)
    assert scalar_failure["failure_stage"] == "hard_iteration_cap"
    assert scalar_failure["iterations_executed"] == 512
    assert scalar_failure["final_relative_normal_residual"] > 1e-8

    solution, report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=matvec,
        diagonal=diagonal,
        initial=torch.zeros_like(right),
        role=role,
        restart_row_block_preconditioner=preconditioner,
    )
    exact_relative = float(
        (
            torch.linalg.vector_norm(right - matvec(solution))
            / torch.linalg.vector_norm(right)
        ).item()
    )

    assert report["solver"] == (
        "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
    )
    assert report["original_lane_iterations_executed"] == 64
    assert report["maximum_iterations"] == 4096
    assert report["periodic_restart_count"] == 0
    assert report[
        "single_restart_after_original_lane_exact_authentication_miss"
    ] is True
    assert report[
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction"
    ] is False
    assert 0 < report[
        "continuous_row_block_lane_iterations_executed"
    ] < 4096 - 64
    assert report["final_relative_normal_residual"] == exact_relative
    assert exact_relative <= 1e-8
    lane = report["continuous_row_block_lane"]
    assert lane["start_iteration"] == 64
    assert lane["planned_stop_iteration"] == 4096
    assert lane["iterations_executed"] == (
        report["iterations_executed"] - 64
    )
    assert lane["termination"] == "exact_tolerance_reached"
    assert lane["exact_diagnostic_matvec_calls"] >= 1
    restart = report["restart_preconditioner"]
    assert restart["authentication_attempted"] is True
    assert restart["authenticated"] is True
    assert restart["applied"] is True
    assert restart["application_attempt_count"] == restart[
        "application_count"
    ]
    assert restart["application_count"] > 0
    receipt = restart["receipt"]
    assert receipt["row_block_count"] == 72
    assert receipt["row_block_size"] == 64
    assert receipt["all_row_blocks_spd"] is True
    assert receipt["full_operator_materialized"] is False
    assert receipt["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "report_sha256"
        }
    )
    assert report["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in report.items()
            if key != "report_sha256"
        }
    )

    tampered_blocks = preconditioner.row_blocks.clone()
    tampered_blocks[0, 0, 0] += 1.0
    tampered = sequence_fold._PcgRestartRowBlockPreconditioner(
        row_blocks=tampered_blocks,
        construction_report=preconditioner.construction_report,
    )
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as tampered_raised:
        sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=torch.zeros_like(right),
            role=role,
            restart_row_block_preconditioner=tampered,
        )
    tampered_failure = dict(tampered_raised.value.failure_report)
    assert tampered_failure["failure_stage"] == (
        "row_block_preconditioner_authentication"
    )
    assert tampered_failure["failure_reason"] == (
        "row_block_preconditioner_tensor_hash_mismatch"
    )
    assert tampered_failure["original_lane_iterations_executed"] == 64
    assert tampered_failure["iterations_executed"] == 64
    assert tampered_failure["restart_preconditioner"][
        "authentication_attempted"
    ] is True
    assert tampered_failure["restart_preconditioner"][
        "authenticated"
    ] is False


def test_single_tensor_pcg_restarts_after_64_and_authenticates_exact_residual() -> None:
    right, diagonal, raw_matvec = _tridiagonal_spd_problem(72)
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        return raw_matvec(value)

    solution, report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=matvec,
        diagonal=diagonal,
        initial=torch.zeros_like(right),
        role="ill_conditioned_restarted_oracle",
    )
    exact_relative = float(
        (
            torch.linalg.vector_norm(right - raw_matvec(solution))
            / torch.linalg.vector_norm(right)
        ).item()
    )

    assert report["solver"] == (
        "deterministic_restarted_jacobi_pcg_single_tensor"
    )
    assert report["original_lane_maximum_iterations"] == 64
    assert report["original_lane_iterations_executed"] == 64
    assert report["restart_block_iterations"] == 64
    assert report["maximum_iterations"] == 512
    assert report["original_lane_final_recurrence_relative_normal_residual"] > 1e-8
    assert 64 < report["iterations_executed"] < 512
    assert report["final_relative_normal_residual"] == exact_relative
    assert exact_relative <= 1e-8
    assert report["exact_final_residual_authenticated"] is True
    assert report["total_matvec_calls"] == calls
    assert report["exact_relative_normal_residual_checkpoints"][0][
        "iteration"
    ] == 64
    assert report["exact_relative_normal_residual_checkpoints"][-1] == {
        "iteration": report["iterations_executed"],
        "relative_normal_residual": exact_relative,
        "finite": True,
    }
    assert (
        report["extended_after_original_lane_exact_authentication_miss"]
        is True
    )
    assert report["recurrence_history_sha256"] == sequence_fold._sha256(
        report["relative_normal_residual_history"]
    )
    assert report["exact_residual_checkpoints_sha256"] == sequence_fold._sha256(
        report["exact_relative_normal_residual_checkpoints"]
    )
    assert report["restart_schedule_sha256"] == sequence_fold._sha256(
        report["restart_schedule"]
    )
    assert report["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in report.items()
            if key != "report_sha256"
        }
    )


def test_single_tensor_pcg_hard_cap_fails_with_sealed_exact_residual() -> None:
    right, diagonal, raw_matvec = _tridiagonal_spd_problem(96)
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        return raw_matvec(value)

    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError,
        match="authenticated exact residual tolerance",
    ) as raised:
        sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=torch.zeros_like(right),
            role="ill_conditioned_hard_cap_oracle",
        )

    failure = dict(raised.value.failure_report)
    assert failure["status"] == "failed_closed"
    assert failure["failure_reason"] == "hard_iteration_cap_reached"
    assert failure["iterations_executed"] == 512
    assert failure["maximum_iterations"] == 512
    assert failure["final_relative_normal_residual"] > 1e-8
    assert failure["exact_final_residual_authenticated"] is True
    assert failure["exact_relative_normal_residual_checkpoints"][-1][
        "iteration"
    ] == 512
    assert failure["total_matvec_calls"] == calls
    assert failure["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in failure.items()
            if key != "report_sha256"
        }
    )


def test_sequence_policy_digest_remains_v2_compatible() -> None:
    assert (
        sequence_fold.FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_POLICY_SHA256
        == "fc9846f62e13a45e63e58b1c0414c3818c4ba8bb08d0cc72f1d7656c253fcabb"
    )
    body = sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy().as_dict()
    assert "pcg_restart_block_iterations" not in body
    assert "pcg_hard_maximum_iterations" not in body
    assert "pcg_exact_residual_recomputation_after_original_lane_miss" not in body


def test_single_tensor_pcg_rejects_false_fast_success_by_exact_residual() -> None:
    right = torch.tensor([1.0], dtype=DTYPE)
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.5 * value
        if calls == 2:
            return value
        return 0.5 * value

    solution, report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=matvec,
        diagonal=torch.ones_like(right),
        initial=torch.zeros_like(right),
        role="false_fast_success_exact_rejection",
    )

    assert torch.equal(solution, torch.tensor([2.0], dtype=DTYPE))
    assert report["solver"] == (
        "deterministic_restarted_jacobi_pcg_single_tensor"
    )
    assert report["original_lane_iterations_executed"] == 1
    assert (
        report["original_lane_final_recurrence_relative_normal_residual"]
        == 0.0
    )
    assert report["exact_relative_normal_residual_checkpoints"][0] == {
        "iteration": 1,
        "relative_normal_residual": 0.5,
        "finite": True,
    }
    assert (
        report["extended_after_original_lane_exact_authentication_miss"]
        is True
    )
    assert report["final_relative_normal_residual"] == 0.0
    assert report["exact_final_residual_authenticated"] is True
    assert report["total_matvec_calls"] == calls


@pytest.mark.parametrize("diagonal_value", (0.0, -1.0))
def test_single_tensor_pcg_rejects_non_positive_diagonal(
    diagonal_value,
) -> None:
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as raised:
        sequence_fold._pcg_single_tensor(
            right=torch.tensor([1.0], dtype=DTYPE),
            matvec=lambda value: value,
            diagonal=torch.tensor([diagonal_value], dtype=DTYPE),
            initial=torch.tensor([0.0], dtype=DTYPE),
            role="invalid_diagonal_oracle",
        )

    failure = dict(raised.value.failure_report)
    assert failure["failure_stage"] == "input_validation"
    assert failure["failure_details"]["diagonal_strictly_positive"] is False
    assert failure["total_matvec_calls"] == 0


def test_single_tensor_pcg_nonfinite_failure_report_is_json_safe() -> None:
    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as raised:
        sequence_fold._pcg_single_tensor(
            right=torch.tensor([1.0], dtype=DTYPE),
            matvec=lambda value: torch.full_like(value, float("nan")),
            diagonal=torch.tensor([1.0], dtype=DTYPE),
            initial=torch.tensor([0.0], dtype=DTYPE),
            role="nonfinite_operator_oracle",
        )

    failure = dict(raised.value.failure_report)
    assert failure["status"] == "failed_closed"
    assert failure["failure_reason"] == (
        "non_finite_initial_operator_application"
    )
    assert failure["exact_final_residual_authenticated"] is False
    assert failure["final_relative_normal_residual"] is None
    assert failure["total_matvec_calls"] == 1
    assert "nan" not in sequence_fold._canonical_json(failure).decode("ascii")
    assert failure["report_sha256"] == sequence_fold._sha256(
        {
            key: value
            for key, value in failure.items()
            if key != "report_sha256"
        }
    )


def test_single_tensor_pcg_indefinite_curvature_fails_closed() -> None:
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        return -value

    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as raised:
        sequence_fold._pcg_single_tensor(
            right=torch.tensor([1.0], dtype=DTYPE),
            matvec=matvec,
            diagonal=torch.tensor([1.0], dtype=DTYPE),
            initial=torch.tensor([0.0], dtype=DTYPE),
            role="indefinite_curvature_oracle",
        )

    failure = dict(raised.value.failure_report)
    assert failure["failure_reason"] == (
        "non_positive_or_non_finite_curvature"
    )
    assert failure["failure_stage"] == "original_lane_curvature"
    assert failure["failure_details"]["curvature"] == -1.0
    assert failure["failure_details"][
        "curvature_finite_strictly_positive"
    ] is False
    assert failure["iterations_executed"] == 0
    assert failure["total_matvec_calls"] == calls == 2
    assert failure["exact_final_residual_authenticated"] is True


def test_single_tensor_pcg_restart_curvature_failure_counts_attempt() -> None:
    right = torch.tensor([1.0], dtype=DTYPE)
    calls = 0

    def matvec(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.5 * value
        if calls == 2:
            return value
        if calls == 3:
            return 0.5 * value
        if calls == 4:
            return torch.zeros_like(value)
        raise AssertionError("solver continued after restart curvature failure")

    with pytest.raises(
        sequence_fold.Lfm2SequenceEffectiveAttentionFoldError
    ) as raised:
        sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=torch.ones_like(right),
            initial=torch.zeros_like(right),
            role="restart_singular_curvature_oracle",
        )

    failure = dict(raised.value.failure_report)
    assert failure["failure_stage"] == "restart_curvature"
    assert failure["failure_reason"] == (
        "non_positive_or_non_finite_curvature"
    )
    assert failure["iterations_executed"] == 1
    assert failure["total_matvec_calls"] == calls == 4
    assert failure["failure_details"]["curvature_matvec_call"] == 4
    assert failure["restart_schedule"][-1][
        "attempted_matvec_calls"
    ] == 1
    assert failure["restart_schedule"][-1]["iterations_executed"] == 0
    assert failure["restart_schedule"][-1]["termination"] == (
        "curvature_failure"
    )
    assert failure["exact_final_residual_authenticated"] is True
    assert failure["final_relative_normal_residual"] == 0.5
