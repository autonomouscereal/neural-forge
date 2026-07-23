from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_causal_span_linearization import (  # noqa: E402
    Lfm2CausalSpanAuthenticationError,
    Lfm2CausalSpanEndpointOperator,
    Lfm2SourceSpanBatchChain,
    Lfm2SourceSpanCacheReceipt,
    Lfm2SourceSpanDirectLayerRecord,
    Lfm2SourceSpanDirectParentReceipt,
    Lfm2SourceSpanPlanReceipt,
    Lfm2SourceSpanPrimalEvidence,
    lfm2_source_span_provenance_root_sha256,
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


def _scalar_tensor(value: int | float, *, integer: bool = False) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int64 if integer else DTYPE)


def _macrocell_parameter_state(macrocell: object) -> dict[str, torch.Tensor]:
    operator = macrocell.operator
    mlp = macrocell.mlp
    p = operator.parameters
    m = mlp.parameters
    return {
        "operator.kind": _scalar_tensor(0, integer=True),
        "operator.norm_gain": p.operator_norm_gain,
        "operator.norm_epsilon": _scalar_tensor(p.operator_norm_epsilon),
        "attention.query_weight": p.query_weight,
        "attention.key_weight": p.key_weight,
        "attention.value_weight": p.value_weight,
        "attention.output_weight": p.output_weight,
        "attention.query_norm_gain": p.query_norm_gain,
        "attention.key_norm_gain": p.key_norm_gain,
        "attention.query_key_norm_epsilon": _scalar_tensor(p.query_key_norm_epsilon),
        "attention.query_heads": _scalar_tensor(p.query_heads, integer=True),
        "attention.key_value_heads": _scalar_tensor(
            p.key_value_heads,
            integer=True,
        ),
        "attention.head_dim": _scalar_tensor(p.head_dim, integer=True),
        "attention.scaling": _scalar_tensor(float(p.scaling)),
        "mlp.norm_gain": m.norm_gain,
        "mlp.norm_epsilon": _scalar_tensor(m.norm_epsilon),
        "mlp.gate_weight": m.gate_weight,
        "mlp.up_weight": m.up_weight,
        "mlp.down_weight": m.down_weight,
    }


def _copy_chain(
    chain: Lfm2SourceSpanBatchChain,
    *,
    source_layer_indices: tuple[int, ...] | None = None,
    layer_post_operator_residuals: tuple[torch.Tensor, ...] | None = None,
    layer_post_residuals: tuple[torch.Tensor, ...] | None = None,
) -> Lfm2SourceSpanBatchChain:
    source = (
        chain.source_layer_indices
        if source_layer_indices is None
        else source_layer_indices
    )
    return Lfm2SourceSpanBatchChain(
        batch_index=chain.batch_index,
        semantic_alignment_sha256=chain.semantic_alignment_sha256,
        source_layer_indices=source,
        operator_kinds=chain.operator_kinds,
        layer_pre_residuals=chain.layer_pre_residuals,
        layer_post_operator_residuals=(
            chain.layer_post_operator_residuals
            if layer_post_operator_residuals is None
            else layer_post_operator_residuals
        ),
        layer_post_residuals=(
            chain.layer_post_residuals
            if layer_post_residuals is None
            else layer_post_residuals
        ),
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
    authoritative_chain = Lfm2SourceSpanBatchChain(
        batch_index=0,
        semantic_alignment_sha256=_digest(11),
        source_layer_indices=(7, 8),
        operator_kinds=("full_attention", "full_attention"),
        layer_pre_residuals=(residual, macrocell_0.mlp.output),
        layer_post_operator_residuals=(
            macrocell_0.operator.output,
            macrocell_1.operator.output,
        ),
        layer_post_residuals=(
            macrocell_0.mlp.output,
            macrocell_1.mlp.output,
        ),
    )
    direct_layers = tuple(
        Lfm2SourceSpanDirectLayerRecord(
            source_layer_index=index,
            operator_kind="full_attention",
            parameter_state=(
                _macrocell_parameter_state(macrocell_0)
                if index == 7
                else _macrocell_parameter_state(macrocell_1)
                if index == 8
                else {"dummy": _scalar_tensor(index)}
            ),
        )
        for index in range(9)
    )
    direct_receipt = Lfm2SourceSpanDirectParentReceipt(
        parent_graph_fingerprint=_digest(12),
        checkpoint_configuration={
            "model_type": "lfm2",
            "num_hidden_layers": 9,
        },
        layers=direct_layers,
    )
    ordered_spans = ((0, 1), (2, 3), (4, 5, 6), (7, 8))
    plan_receipt = Lfm2SourceSpanPlanReceipt(
        parent_model_identity_sha256=(direct_receipt.parent_model_identity_sha256),
        parent_graph_fingerprint=direct_receipt.parent_graph_fingerprint,
        ordered_source_spans=ordered_spans,
        target_configuration={
            "model_type": "lfm2",
            "num_hidden_layers": len(ordered_spans),
        },
    )
    cache_receipt = Lfm2SourceSpanCacheReceipt(
        parent_model_identity_sha256=(direct_receipt.parent_model_identity_sha256),
        layer_topology=("full_attention",) * 9,
        batch_chains=(authoritative_chain,),
    )
    provenance_root = lfm2_source_span_provenance_root_sha256(
        plan_receipt,
        cache_receipt,
        direct_receipt,
    )
    chain = _copy_chain(authoritative_chain)
    evidence = Lfm2SourceSpanPrimalEvidence(
        span=Lfm2ModelFoldSpan((7, 8), target_layer_index=3),
        snapshot=snapshot,
        batch_chains=(chain,),
        plan_receipt=plan_receipt,
        cache_receipt=cache_receipt,
        direct_parent_receipt=direct_receipt,
        expected_provenance_root_sha256=provenance_root,
    )
    return {
        "chain": chain,
        "cache_receipt": cache_receipt,
        "direct_receipt": direct_receipt,
        "evidence": evidence,
        "macrocell_0": macrocell_0,
        "macrocell_1": macrocell_1,
        "operator": Lfm2CausalSpanEndpointOperator(evidence),
        "plan_receipt": plan_receipt,
        "provenance_root": provenance_root,
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
            source_layer_indices=chain.source_layer_indices,
            operator_kinds=chain.operator_kinds,
            layer_pre_residuals=(
                chain.layer_pre_residuals[0],
                broken_second_pre,
            ),
            layer_post_operator_residuals=(chain.layer_post_operator_residuals),
            layer_post_residuals=chain.layer_post_residuals,
        )


def test_fully_resealed_absolute_layer_misjoin_is_rejected() -> None:
    fixture = _fixture()
    chain = fixture["chain"]
    cache_receipt = fixture["cache_receipt"]
    direct_receipt = fixture["direct_receipt"]
    snapshot = fixture["snapshot"]
    assert isinstance(chain, Lfm2SourceSpanBatchChain)
    assert isinstance(cache_receipt, Lfm2SourceSpanCacheReceipt)
    assert isinstance(direct_receipt, Lfm2SourceSpanDirectParentReceipt)
    assert isinstance(snapshot, Lfm2WholeStackPrimalSnapshot)
    relabelled_chain = _copy_chain(
        chain,
        source_layer_indices=(100, 101),
    )
    relabelled_plan = Lfm2SourceSpanPlanReceipt(
        parent_model_identity_sha256=(direct_receipt.parent_model_identity_sha256),
        parent_graph_fingerprint=direct_receipt.parent_graph_fingerprint,
        ordered_source_spans=(
            tuple(range(25)),
            tuple(range(25, 50)),
            tuple(range(50, 100)),
            (100, 101),
        ),
        target_configuration={"num_hidden_layers": 4},
    )
    relabelled_root = lfm2_source_span_provenance_root_sha256(
        relabelled_plan,
        cache_receipt,
        direct_receipt,
    )
    assert relabelled_chain.verify_authentication()
    assert relabelled_plan.verify_authentication()
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="do not describe one parent|does not join",
    ):
        Lfm2SourceSpanPrimalEvidence(
            span=Lfm2ModelFoldSpan((100, 101), target_layer_index=3),
            snapshot=snapshot,
            batch_chains=(relabelled_chain,),
            plan_receipt=relabelled_plan,
            cache_receipt=cache_receipt,
            direct_parent_receipt=direct_receipt,
            expected_provenance_root_sha256=relabelled_root,
        )


def test_valid_looking_fake_identity_digest_cannot_replace_cache_authority() -> None:
    fixture = _fixture()
    chain = fixture["chain"]
    cache_receipt = fixture["cache_receipt"]
    direct_receipt = fixture["direct_receipt"]
    plan_receipt = fixture["plan_receipt"]
    snapshot = fixture["snapshot"]
    assert isinstance(chain, Lfm2SourceSpanBatchChain)
    assert isinstance(cache_receipt, Lfm2SourceSpanCacheReceipt)
    assert isinstance(direct_receipt, Lfm2SourceSpanDirectParentReceipt)
    assert isinstance(plan_receipt, Lfm2SourceSpanPlanReceipt)
    assert isinstance(snapshot, Lfm2WholeStackPrimalSnapshot)
    fake_cache = Lfm2SourceSpanCacheReceipt(
        parent_model_identity_sha256=_digest(999),
        layer_topology=cache_receipt.layer_topology,
        batch_chains=cache_receipt.batch_chains,
    )
    fake_root = lfm2_source_span_provenance_root_sha256(
        plan_receipt,
        fake_cache,
        direct_receipt,
    )
    assert fake_cache.verify_authentication()
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="do not describe one parent",
    ):
        Lfm2SourceSpanPrimalEvidence(
            span=Lfm2ModelFoldSpan((7, 8), target_layer_index=3),
            snapshot=snapshot,
            batch_chains=(chain,),
            plan_receipt=plan_receipt,
            cache_receipt=fake_cache,
            direct_parent_receipt=direct_receipt,
            expected_provenance_root_sha256=fake_root,
        )


def test_fully_resealed_direct_parameter_misjoin_is_rejected() -> None:
    fixture = _fixture()
    chain = fixture["chain"]
    cache_receipt = fixture["cache_receipt"]
    direct_receipt = fixture["direct_receipt"]
    plan_receipt = fixture["plan_receipt"]
    snapshot = fixture["snapshot"]
    assert isinstance(chain, Lfm2SourceSpanBatchChain)
    assert isinstance(cache_receipt, Lfm2SourceSpanCacheReceipt)
    assert isinstance(direct_receipt, Lfm2SourceSpanDirectParentReceipt)
    assert isinstance(plan_receipt, Lfm2SourceSpanPlanReceipt)
    assert isinstance(snapshot, Lfm2WholeStackPrimalSnapshot)
    layers = list(direct_receipt.layers)
    original = layers[7]
    modified_state = dict(original.parameter_state)
    modified_state["mlp.down_weight"] = modified_state["mlp.down_weight"].clone()
    modified_state["mlp.down_weight"][0, 0] += 0.25
    layers[7] = Lfm2SourceSpanDirectLayerRecord(
        source_layer_index=7,
        operator_kind=original.operator_kind,
        parameter_state=modified_state,
    )
    fake_direct = Lfm2SourceSpanDirectParentReceipt(
        parent_graph_fingerprint=direct_receipt.parent_graph_fingerprint,
        checkpoint_configuration=direct_receipt.checkpoint_configuration,
        layers=tuple(layers),
    )
    fake_plan = Lfm2SourceSpanPlanReceipt(
        parent_model_identity_sha256=fake_direct.parent_model_identity_sha256,
        parent_graph_fingerprint=fake_direct.parent_graph_fingerprint,
        ordered_source_spans=plan_receipt.ordered_source_spans,
        target_configuration=plan_receipt.target_configuration,
    )
    fake_cache = Lfm2SourceSpanCacheReceipt(
        parent_model_identity_sha256=fake_direct.parent_model_identity_sha256,
        layer_topology=cache_receipt.layer_topology,
        batch_chains=cache_receipt.batch_chains,
    )
    fake_root = lfm2_source_span_provenance_root_sha256(
        fake_plan,
        fake_cache,
        fake_direct,
    )
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="Direct parameters or operator topology",
    ):
        Lfm2SourceSpanPrimalEvidence(
            span=Lfm2ModelFoldSpan((7, 8), target_layer_index=3),
            snapshot=snapshot,
            batch_chains=(chain,),
            plan_receipt=fake_plan,
            cache_receipt=fake_cache,
            direct_parent_receipt=fake_direct,
            expected_provenance_root_sha256=fake_root,
        )


@pytest.mark.parametrize("boundary", ("post_operator", "terminal_post_layer"))
def test_fully_resealed_nonadjacent_boundary_history_is_rejected(
    boundary: str,
) -> None:
    fixture = _fixture()
    chain = fixture["chain"]
    evidence = fixture["evidence"]
    assert isinstance(chain, Lfm2SourceSpanBatchChain)
    assert isinstance(evidence, Lfm2SourceSpanPrimalEvidence)
    if boundary == "post_operator":
        post_operator = list(chain.layer_post_operator_residuals)
        post_operator[-1] = post_operator[-1].clone()
        post_operator[-1][0, 0, 0] += 0.25
        resealed = _copy_chain(
            chain,
            layer_post_operator_residuals=tuple(post_operator),
        )
    else:
        post = list(chain.layer_post_residuals)
        post[-1] = post[-1].clone()
        post[-1][0, 0, 0] += 0.25
        resealed = _copy_chain(
            chain,
            layer_post_residuals=tuple(post),
        )
    assert resealed.verify_authentication()
    assert resealed.authentication_sha256 != chain.authentication_sha256
    with pytest.raises(
        Lfm2CausalSpanAuthenticationError,
        match="not the authoritative cached span",
    ):
        Lfm2SourceSpanPrimalEvidence(
            span=evidence.span,
            snapshot=evidence.snapshot,
            batch_chains=(resealed,),
            plan_receipt=evidence.plan_receipt,
            cache_receipt=evidence.cache_receipt,
            direct_parent_receipt=evidence.direct_parent_receipt,
            expected_provenance_root_sha256=(evidence.expected_provenance_root_sha256),
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
        object.__setattr__(evidence.cache_receipt, "session_sha256", _digest(999))
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
