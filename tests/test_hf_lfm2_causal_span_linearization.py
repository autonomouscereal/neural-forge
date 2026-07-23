from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_causal_span_linearization import (  # noqa: E402
    Lfm2CausalSpanAuthenticationError,
    Lfm2CausalSpanEndpointOperator,
    Lfm2SourceSpanBatchChain,
    Lfm2SourceSpanPrimalEvidence,
)
from neural_forge.hf_lfm2_global_linearization import (  # noqa: E402
    Lfm2AttentionParameters,
    Lfm2MlpParameters,
    Lfm2StackPrimalBatch,
    Lfm2WholeStackPrimalSnapshot,
    capture_lfm2_macrocell_linearization,
    capture_lfm2_mlp_primal,
)
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan  # noqa: E402


DTYPE = torch.float64


def _digest(value: int) -> str:
    return f"{value:064x}"


def _randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, dtype=DTYPE) * scale


def _attention_parameters(
    *,
    sequence: int,
    hidden: int,
) -> Lfm2AttentionParameters:
    angles = _randn(1, sequence, 1, scale=0.25)
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    return Lfm2AttentionParameters(
        operator_norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        operator_norm_epsilon=1e-3,
        query_weight=_randn(4, hidden, scale=0.13),
        key_weight=_randn(2, hidden, scale=0.13),
        value_weight=_randn(2, hidden, scale=0.13),
        output_weight=_randn(hidden, 4, scale=0.13),
        query_norm_gain=torch.rand(2, dtype=DTYPE) + 0.5,
        key_norm_gain=torch.rand(2, dtype=DTYPE) + 0.5,
        query_key_norm_epsilon=2e-3,
        rope_cosine=cosine,
        rope_sine=sine,
        causal_mask=torch.tril(
            torch.ones((sequence, sequence), dtype=torch.bool)
        ).reshape(1, 1, sequence, sequence),
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )


def _mlp_parameters(*, hidden: int, intermediate: int) -> Lfm2MlpParameters:
    return Lfm2MlpParameters(
        norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        norm_epsilon=1e-3,
        gate_weight=_randn(intermediate, hidden, scale=0.14),
        up_weight=_randn(intermediate, hidden, scale=0.14),
        down_weight=_randn(hidden, intermediate, scale=0.14),
    )


def _fixture(*, downstream_scale: float = 1.0) -> dict[str, object]:
    torch.manual_seed(901)
    sequence, hidden, intermediate = 5, 4, 6
    residual = _randn(1, sequence, hidden, scale=0.2)
    operator_0 = _attention_parameters(sequence=sequence, hidden=hidden)
    mlp_0 = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    operator_1 = _attention_parameters(sequence=sequence, hidden=hidden)
    mlp_1 = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    if downstream_scale != 1.0:
        mlp_1 = replace(
            mlp_1,
            down_weight=mlp_1.down_weight * downstream_scale,
        )
    macrocell_0 = capture_lfm2_macrocell_linearization(
        residual,
        operator_0,
        mlp_0,
    )
    macrocell_1 = capture_lfm2_macrocell_linearization(
        macrocell_0.mlp.output,
        operator_1,
        mlp_1,
    )
    positions = torch.arange(sequence, dtype=torch.int64)
    batch = Lfm2StackPrimalBatch(
        macrocells=(macrocell_0, macrocell_1),
        selected_target_positions=positions,
        canonical_boundary_row_indices=positions,
    )
    snapshot = Lfm2WholeStackPrimalSnapshot((batch,))
    chain = Lfm2SourceSpanBatchChain(
        batch_index=0,
        semantic_alignment_sha256=_digest(11),
        cache_payload_sha256=_digest(12),
        layer_pre_residuals=(residual, macrocell_0.mlp.output),
        layer_post_residuals=(
            macrocell_0.mlp.output,
            macrocell_1.mlp.output,
        ),
    )
    evidence = Lfm2SourceSpanPrimalEvidence(
        span=Lfm2ModelFoldSpan((7, 8), target_layer_index=3),
        snapshot=snapshot,
        batch_chains=(chain,),
        source_plan_sha256=_digest(13),
        cache_key=_digest(14),
        cache_manifest_sha256=_digest(15),
        cache_session_sha256=_digest(16),
        direct_parent_session_sha256=_digest(17),
    )
    return {
        "chain": chain,
        "evidence": evidence,
        "macrocell_0": macrocell_0,
        "macrocell_1": macrocell_1,
        "operator": Lfm2CausalSpanEndpointOperator(evidence),
        "snapshot": snapshot,
    }


def _seed(shape: tuple[int, ...]) -> torch.Tensor:
    count = 1
    for value in shape:
        count *= value
    return torch.linspace(-0.07, 0.09, count, dtype=DTYPE).reshape(shape)


def _endpoint_cotangent(shape: tuple[int, ...]) -> torch.Tensor:
    count = 1
    for value in shape:
        count *= value
    return torch.cos(torch.arange(count, dtype=DTYPE) * 0.31).reshape(shape)


def test_two_layer_full_history_post_attention_jvp_matches_finite_difference() -> None:
    fixture = _fixture()
    operator = fixture["operator"]
    macrocell_0 = fixture["macrocell_0"]
    macrocell_1 = fixture["macrocell_1"]
    assert isinstance(operator, Lfm2CausalSpanEndpointOperator)
    tangent = _seed(tuple(macrocell_0.operator.output.shape))
    seed = operator.make_seed(
        source_layer_index=7,
        boundary_role="post_operator",
        batch_tangents=(tangent,),
    )
    observed = operator.jvp(seed)
    step = 1e-6

    def replay(signed_step: float) -> torch.Tensor:
        perturbed_post_operator = macrocell_0.operator.output + signed_step * tangent
        perturbed_post_layer = capture_lfm2_mlp_primal(
            perturbed_post_operator,
            macrocell_0.mlp.parameters,
        ).output
        return capture_lfm2_macrocell_linearization(
            perturbed_post_layer,
            macrocell_1.operator.parameters,
            macrocell_1.mlp.parameters,
        ).mlp.output

    finite_difference = (replay(step) - replay(-step)) / (2.0 * step)
    torch.testing.assert_close(
        observed.batch_tangents[0],
        finite_difference,
        rtol=4e-6,
        atol=4e-8,
    )
    assert observed.verify_authentication()
    assert observed.seed_sha256 == seed.authentication_sha256


def test_two_layer_full_history_endpoint_map_has_exact_adjoint() -> None:
    fixture = _fixture()
    operator = fixture["operator"]
    macrocell_0 = fixture["macrocell_0"]
    macrocell_1 = fixture["macrocell_1"]
    assert isinstance(operator, Lfm2CausalSpanEndpointOperator)
    source = _seed(tuple(macrocell_0.operator.output.shape))
    endpoint = _endpoint_cotangent(tuple(macrocell_1.mlp.output.shape))
    seed = operator.make_seed(
        source_layer_index=7,
        boundary_role="post_operator",
        batch_tangents=(source,),
    )
    endpoint_bundle = operator.make_endpoint_cotangent((endpoint,))
    forward = operator.jvp(seed)
    transpose = operator.vjp(
        endpoint_bundle,
        source_layer_index=7,
        boundary_role="post_operator",
    )
    torch.testing.assert_close(
        (forward.batch_tangents[0] * endpoint).sum(),
        (source * transpose.batch_cotangents[0]).sum(),
        rtol=3e-11,
        atol=3e-11,
    )
    assert endpoint_bundle.verify_authentication()
    assert transpose.verify_authentication()


def test_identical_local_seed_changes_when_a_later_source_layer_changes() -> None:
    baseline = _fixture()
    changed = _fixture(downstream_scale=2.75)
    baseline_operator = baseline["operator"]
    changed_operator = changed["operator"]
    baseline_macrocell = baseline["macrocell_0"]
    changed_macrocell = changed["macrocell_0"]
    assert isinstance(baseline_operator, Lfm2CausalSpanEndpointOperator)
    assert isinstance(changed_operator, Lfm2CausalSpanEndpointOperator)
    source = _seed(tuple(baseline_macrocell.operator.output.shape))
    changed_source = _seed(tuple(changed_macrocell.operator.output.shape))
    assert torch.equal(source, changed_source)
    baseline_effect = baseline_operator.jvp(
        baseline_operator.make_seed(
            source_layer_index=7,
            boundary_role="post_operator",
            batch_tangents=(source,),
        )
    ).batch_tangents[0]
    changed_effect = changed_operator.jvp(
        changed_operator.make_seed(
            source_layer_index=7,
            boundary_role="post_operator",
            batch_tangents=(changed_source,),
        )
    ).batch_tangents[0]
    assert not torch.equal(baseline_effect, changed_effect)
    assert float(torch.linalg.vector_norm(baseline_effect - changed_effect)) > 1e-8


def test_source_chain_rejects_nonadjacent_full_histories() -> None:
    fixture = _fixture()
    chain = fixture["chain"]
    assert isinstance(chain, Lfm2SourceSpanBatchChain)
    broken_second_pre = chain.layer_pre_residuals[1].clone()
    broken_second_pre[0, 0, 0] += 0.125
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="exact adjacent residual chain",
    ):
        Lfm2SourceSpanBatchChain(
            batch_index=chain.batch_index,
            semantic_alignment_sha256=chain.semantic_alignment_sha256,
            cache_payload_sha256=chain.cache_payload_sha256,
            layer_pre_residuals=(
                chain.layer_pre_residuals[0],
                broken_second_pre,
            ),
            layer_post_residuals=chain.layer_post_residuals,
        )


@pytest.mark.parametrize(
    "tamper",
    ("plan", "cache_receipt", "history", "weight"),
)
def test_sealed_span_rejects_plan_cache_history_and_weight_tampering(
    tamper: str,
) -> None:
    fixture = _fixture()
    evidence = fixture["evidence"]
    operator = fixture["operator"]
    macrocell_0 = fixture["macrocell_0"]
    assert isinstance(evidence, Lfm2SourceSpanPrimalEvidence)
    assert isinstance(operator, Lfm2CausalSpanEndpointOperator)
    seed = operator.make_seed(
        source_layer_index=7,
        boundary_role="post_operator",
        batch_tangents=(_seed(tuple(macrocell_0.operator.output.shape)),),
    )
    if tamper == "plan":
        object.__setattr__(
            evidence,
            "span",
            Lfm2ModelFoldSpan((8, 9), target_layer_index=3),
        )
    elif tamper == "cache_receipt":
        object.__setattr__(evidence, "cache_session_sha256", _digest(999))
    elif tamper == "history":
        evidence.batch_chains[0].layer_pre_residuals[0][0, 0, 0] += 0.25
    else:
        (
            evidence.snapshot.batches[0].macrocells[1].mlp.parameters.down_weight[0, 0]
        ) += 0.25
    assert not evidence.verify_authentication()
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="changed after operator construction",
    ):
        operator.jvp(seed)
