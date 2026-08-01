"""Stage-1 proofs for the rectangular residual boundary charts.

Three things are proven here, all on cached parent evidence with no optimizer,
no gradient, and no parent forward beyond the single sealed capture:

* the chart artifacts are what they claim — orthonormal encoders, a
  cycle-anchored decoder, measured defects, and a seal that detects tampering;
* the load-bearing sequential target really is
  ``Y = h_{b+1} E_{b+1} - hhat_actual``, including the upstream-error
  correction term whose absence is the measured A0025 failure; and
* every fail-closed condition the design names actually closes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_cross_width_charts import (  # noqa: E402
    ATTENTION_INPUT_BOUNDARY,
    MLP_INPUT_BOUNDARY,
    TERMINAL_BOUNDARY_ID,
    CrossWidthResidualChartAuthenticationError,
    CrossWidthResidualChartError,
    CrossWidthResidualChartRankError,
    LiveChildBoundaryRecapture,
    ResidualBoundaryChart,
    ResidualBoundaryChartPolicy,
    ResidualBoundaryChartSet,
    build_sequential_branch_target,
    child_residual_topology_sha256,
    compile_residual_boundary_chart_set,
    identity_residual_boundary_chart_set,
    interface_locked_boundary_ids,
    plan_reverse_reclosure_sweep,
    residual_boundary_id,
    residual_boundary_ids,
)


TOKENIZER = canonical_sha256({"tokenizer": "cross-width-charts"})

PARENT_WIDTH = 64
CHILD_WIDTH = 48
INTERMEDIATE = 96
LAYERS = 2
VOCABULARY = 48


def _llama(seed: int, hidden: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": VOCABULARY,
        "hidden_size": hidden,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**configuration))
    model.eval()
    return model


def confine_parent_to_subspace(model, rank: int, seed: int = 913):
    """Force every residual state of one parent into an exact ``rank`` subspace.

    This is the design's stage-1 exact-subspace control: consumers read only
    the subspace, producers write only into it, and the norms are constant so
    RMSNorm cannot rotate a row out of it.  A chart that finds this subspace
    loses nothing, so any residual endpoint error is the machinery's, not the
    contraction's.
    """

    generator = torch.Generator().manual_seed(seed)
    width = int(model.config.hidden_size)
    basis, _ = torch.linalg.qr(torch.randn(width, rank, generator=generator))
    with torch.no_grad():

        def consume(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_((weight @ basis) @ basis.T)

        def produce(path: str) -> None:
            weight = model.get_parameter(path)
            weight.copy_(basis @ (basis.T @ weight))

        consume("model.embed_tokens.weight")
        consume("lm_head.weight")
        model.get_parameter("model.norm.weight").fill_(1.0)
        for index in range(LAYERS):
            prefix = f"model.layers.{index}."
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            ):
                consume(prefix + name + ".weight")
            for name in ("self_attn.o_proj", "mlp.down_proj"):
                produce(prefix + name + ".weight")
            model.get_parameter(prefix + "input_layernorm.weight").fill_(1.0)
            model.get_parameter(prefix + "post_attention_layernorm.weight").fill_(1.0)
    return basis


def input_ids(sequences: int = 24, length: int = 24):
    return torch.randint(
        0,
        VOCABULARY,
        (sequences, length),
        generator=torch.Generator().manual_seed(7),
    )


@pytest.fixture(scope="module")
def subspace_parent():
    parent = _llama(20260801, PARENT_WIDTH)
    basis = confine_parent_to_subspace(parent, CHILD_WIDTH)
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": input_ids()},
        parent_graph_fingerprint=graph.fingerprint,
        metadata={"test": "cross-width-exact-subspace-control"},
    )
    return parent, graph, evidence, basis


@pytest.fixture(scope="module")
def child_model():
    return _llama(4711, CHILD_WIDTH)


@pytest.fixture(scope="module")
def chart_set(subspace_parent, child_model) -> ResidualBoundaryChartSet:
    parent, _graph, evidence, _basis = subspace_parent
    return compile_residual_boundary_chart_set(evidence, parent, child_model)


def test_boundary_ids_cover_every_residual_write_and_lock_the_interfaces() -> None:
    ordered = residual_boundary_ids(LAYERS)

    assert ordered == (
        "layer000.attention_in",
        "layer000.mlp_in",
        "layer001.attention_in",
        "layer001.mlp_in",
        TERMINAL_BOUNDARY_ID,
    )
    assert len(ordered) == 2 * LAYERS + 1
    assert interface_locked_boundary_ids(LAYERS) == (ordered[0], TERMINAL_BOUNDARY_ID)
    with pytest.raises(CrossWidthResidualChartError):
        residual_boundary_id(0, "somewhere_else")


def test_charts_are_orthonormal_with_measured_defects_inside_policy(
    chart_set: ResidualBoundaryChartSet,
) -> None:
    assert chart_set.source_width == PARENT_WIDTH
    assert chart_set.target_width == CHILD_WIDTH
    assert chart_set.boundary_ids == residual_boundary_ids(LAYERS)
    identity = torch.eye(CHILD_WIDTH, dtype=torch.float64)

    for chart in chart_set.charts:
        assert tuple(chart.encoder.shape) == (PARENT_WIDTH, CHILD_WIDTH)
        assert tuple(chart.decoder.shape) == (CHILD_WIDTH, PARENT_WIDTH)
        # E_b^T E_b = I_Hc: the orthonormality that makes the norm-commutation
        # rescale a single well-behaved scalar.
        assert torch.allclose(chart.encoder.T @ chart.encoder, identity, atol=1e-10)
        diagnostics = chart.report["diagnostics"]
        assert diagnostics["encoder_orthonormality_max_abs_error"] < 1e-9
        # On the exact-subspace control the chart discards nothing, so every
        # measured defect collapses to numerical noise.
        assert diagnostics["cycle_defect"] < 1e-5
        assert diagnostics["norm_commutation_row_fluctuation"] < 1e-9
        assert abs(diagnostics["norm_commutation_mean_ratio"] - 1.0) < 1e-6
        assert diagnostics["norm_commutation_measured_rows"] > 0
        absorption = diagnostics["absorption_demand"]
        if chart.boundary_id == TERMINAL_BOUNDARY_ID:
            assert absorption is None
        else:
            assert absorption["unabsorbed_energy_fraction"] < 1e-6
            assert absorption["writer_representable_rank"] > 0

    locked = {chart.boundary_id for chart in chart_set.charts if chart.interface_locked}
    assert locked == set(interface_locked_boundary_ids(LAYERS))


def test_chart_roundtrip_recovers_the_parent_rows_it_charted(
    subspace_parent,
    chart_set: ResidualBoundaryChartSet,
) -> None:
    _parent, _graph, evidence, basis = subspace_parent
    fit = evidence.fit_token_mask.reshape(-1)

    for chart in chart_set.charts:
        # The chart's span is the parent's own residual subspace, so the
        # encode/decode cycle is exact up to the anchored ridge.
        projector = chart.encoder @ chart.encoder.T
        assert torch.allclose(
            projector @ basis.double(),
            basis.double(),
            atol=1e-6,
        )

    chart = chart_set.chart(residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY))
    rows = evidence.layers[0].block_input.reshape(-1, PARENT_WIDTH).double()[fit]
    recovered = chart.decode(chart.encode(rows))
    error = float(
        (recovered - rows).square().sum().sqrt() / rows.square().sum().sqrt()
    )
    assert error == pytest.approx(chart.report["diagnostics"]["cycle_defect"], rel=1e-9)
    assert error < 1e-5


def test_chart_set_seal_detects_a_tampered_encoder(
    chart_set: ResidualBoundaryChartSet,
) -> None:
    chart_set.verify()
    original = chart_set.charts[1]
    tampered_encoder = original.encoder.clone()
    tampered_encoder[0, 0] += 1e-3
    tampered = ResidualBoundaryChart(
        boundary_id=original.boundary_id,
        encoder=tampered_encoder,
        decoder=original.decoder,
        interface_locked=original.interface_locked,
        report=dict(original.report),
    )
    forged = ResidualBoundaryChartSet(
        parent_evidence_sha256=chart_set.parent_evidence_sha256,
        child_topology_sha256=chart_set.child_topology_sha256,
        charts=(chart_set.charts[0], tampered, *chart_set.charts[2:]),
        report=dict(chart_set.report),
    )

    with pytest.raises(CrossWidthResidualChartAuthenticationError):
        forged.verify()
    with pytest.raises(CrossWidthResidualChartAuthenticationError):
        tampered.verify()


def test_chart_set_authentication_binds_evidence_and_child_topology(
    chart_set: ResidualBoundaryChartSet,
    child_model,
) -> None:
    chart_set.authenticate(
        parent_evidence_sha256=chart_set.parent_evidence_sha256,
        child_topology_sha256=child_residual_topology_sha256(child_model),
        role="test",
    )
    with pytest.raises(
        CrossWidthResidualChartAuthenticationError,
        match="different parent evidence",
    ):
        chart_set.authenticate(
            parent_evidence_sha256=canonical_sha256({"other": "evidence"}),
            child_topology_sha256=chart_set.child_topology_sha256,
            role="test",
        )
    with pytest.raises(
        CrossWidthResidualChartAuthenticationError,
        match="different child topology",
    ):
        chart_set.authenticate(
            parent_evidence_sha256=chart_set.parent_evidence_sha256,
            child_topology_sha256=canonical_sha256({"other": "topology"}),
            role="test",
        )
    with pytest.raises(CrossWidthResidualChartAuthenticationError):
        chart_set.chart("layer099.attention_in")


def test_child_topology_seal_moves_with_the_child_geometry(child_model) -> None:
    same = _llama(4711, CHILD_WIDTH)
    narrower_mlp = _llama(4711, CHILD_WIDTH, intermediate_size=64)

    assert child_residual_topology_sha256(child_model) == (
        child_residual_topology_sha256(same)
    )
    assert child_residual_topology_sha256(child_model) != (
        child_residual_topology_sha256(narrower_mlp)
    )


def test_width_preserving_identity_charts_have_exactly_zero_defects(
    subspace_parent,
) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    charts = identity_residual_boundary_chart_set(evidence, parent)
    charts.verify()
    identity = torch.eye(PARENT_WIDTH, dtype=torch.float64)

    assert charts.source_width == charts.target_width == PARENT_WIDTH
    for chart in charts.charts:
        assert torch.equal(chart.encoder, identity)
        assert torch.equal(chart.decoder, identity)
        diagnostics = chart.report["diagnostics"]
        assert diagnostics["cycle_defect"] == 0.0
        assert diagnostics["norm_commutation_row_fluctuation"] == 0.0
        assert diagnostics["norm_commutation_mean_ratio"] == 1.0
        absorption = diagnostics["absorption_demand"]
        if absorption is not None:
            assert absorption["frobenius"] == 0.0
            assert absorption["unabsorbed_energy_fraction"] == 0.0


def test_identity_charts_refuse_a_child_of_a_different_width(
    subspace_parent,
    child_model,
) -> None:
    _parent, _graph, evidence, _basis = subspace_parent
    with pytest.raises(CrossWidthResidualChartError, match="equal parent and child"):
        identity_residual_boundary_chart_set(evidence, child_model)


def test_sequential_target_carries_the_upstream_error_correction_term() -> None:
    """The A0025 guard, hand-computed on a two-boundary case.

    Two 3->2 charts differ by one column swap, so every term of the expansion
    is nonzero and separable by hand.  If the builder dropped the ``-e_b``
    term — the parent-coordinate-local target A0025 measured as the failure —
    ``Y`` would come out as ``transported + transition`` and this test would
    show the exact difference.
    """

    encoder_in = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    encoder_out = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    decoder = encoder_in.T.clone()

    def chart(boundary_id: str, encoder: torch.Tensor) -> ResidualBoundaryChart:
        from neural_forge.hf_causal_cross_width_charts import _seal_chart

        return _seal_chart(
            boundary_id=boundary_id,
            encoder=encoder,
            decoder=decoder,
            interface_locked=False,
            diagnostics={"hand_computed_fixture": True},
            lineage={"hand_computed_fixture": True},
        )

    parent_input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float64)
    parent_delta = torch.tensor(
        [[[0.5, -1.0, 2.0], [1.5, 0.0, -2.0]]],
        dtype=torch.float64,
    )
    parent_output = parent_input + parent_delta
    child_state = torch.tensor([[[1.25, 1.75], [3.5, 5.5]]], dtype=torch.float64)
    fit_mask = torch.tensor([[True, True]])

    target = build_sequential_branch_target(
        input_chart=chart("layer000.attention_in", encoder_in),
        output_chart=chart("layer000.mlp_in", encoder_out),
        parent_input_state=parent_input,
        parent_output_state=parent_output,
        parent_branch_delta=parent_delta,
        parent_normalized_input=parent_input,
        child_input_state=child_state,
        fit_token_mask=fit_mask,
    )

    # Y = h_next @ E_next - hhat, computed by hand.
    #   h_next        = [[1.5, 1.0, 5.0], [5.5, 5.0, 4.0]]
    #   h_next E_next = [[5.0, 1.0], [4.0, 5.0]]
    #   Y             = [[3.75, -0.75], [0.5, -0.5]]
    expected = torch.tensor([[[3.75, -0.75], [0.5, -0.5]]], dtype=torch.float64)
    assert torch.allclose(target.endpoint_target, expected, atol=1e-12)

    transported = parent_delta @ encoder_out
    transition = parent_input @ (encoder_out - encoder_in)
    upstream = child_state - parent_input @ encoder_in
    assert torch.allclose(
        target.endpoint_target,
        transported + transition - upstream,
        atol=1e-12,
    )
    # Every responsibility is separately nonzero on this fixture, so the
    # equality above is not satisfied by a degenerate term.
    assert float(transported.abs().max()) > 0.0
    assert float(transition.abs().max()) > 0.0
    assert float(upstream.abs().max()) > 0.0
    # And the parent-coordinate-local target really is a different number.
    assert not torch.allclose(
        target.endpoint_target,
        transported + transition,
        atol=1e-9,
    )

    report = target.report
    assert report["upstream_error_correction_included"] is True
    assert report["sequential_child_replay"] is True
    assert report["parent_residual_coordinate_endpoints"] is False
    assert report["sequential_expansion_max_abs_error"] < 1e-12
    assert report["fit_upstream_error_energy"] == pytest.approx(
        float(upstream.square().sum()),
        rel=1e-12,
    )
    assert report["fit_chart_transition_energy"] == pytest.approx(
        float(transition.square().sum()),
        rel=1e-12,
    )
    assert report["fit_transported_parent_delta_energy"] == pytest.approx(
        float(transported.square().sum()),
        rel=1e-12,
    )
    assert report["input_boundary_id"] == "layer000.attention_in"
    assert report["output_boundary_id"] == "layer000.mlp_in"


def test_sequential_target_rejects_a_child_state_of_the_wrong_width(
    chart_set: ResidualBoundaryChartSet,
    subspace_parent,
) -> None:
    _parent, _graph, evidence, _basis = subspace_parent
    layer = evidence.layers[0]
    input_chart = chart_set.chart(residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY))
    output_chart = chart_set.chart(residual_boundary_id(0, MLP_INPUT_BOUNDARY))

    with pytest.raises(CrossWidthResidualChartError, match="recaptured child state"):
        build_sequential_branch_target(
            input_chart=input_chart,
            output_chart=output_chart,
            parent_input_state=layer.block_input,
            parent_output_state=layer.post_attention_residual,
            parent_branch_delta=layer.attention_delta,
            parent_normalized_input=layer.attention_norm_output,
            child_input_state=layer.block_input,
            fit_token_mask=evidence.fit_token_mask,
        )


def test_chart_selection_refuses_width_expansion(subspace_parent) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    wider = _llama(31337, PARENT_WIDTH * 2)

    with pytest.raises(CrossWidthResidualChartRankError, match="expansion"):
        compile_residual_boundary_chart_set(evidence, parent, wider)


def test_chart_selection_refuses_a_supplied_chart_at_an_ordinary_boundary(
    subspace_parent,
    child_model,
) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    encoder = torch.eye(PARENT_WIDTH, dtype=torch.float64)[:, :CHILD_WIDTH]

    with pytest.raises(CrossWidthResidualChartError, match="interface-locked"):
        compile_residual_boundary_chart_set(
            evidence,
            parent,
            child_model,
            interface_locked_encoders={
                residual_boundary_id(0, MLP_INPUT_BOUNDARY): encoder
            },
        )


def test_chart_selection_refuses_a_non_orthogonal_interface_chart(
    subspace_parent,
    child_model,
) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    encoder = torch.eye(PARENT_WIDTH, dtype=torch.float64)[:, :CHILD_WIDTH] * 2.0

    with pytest.raises(CrossWidthResidualChartError, match="orthogonal rotation"):
        compile_residual_boundary_chart_set(
            evidence,
            parent,
            child_model,
            interface_locked_encoders={
                residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY): encoder
            },
        )


def test_narrow_attention_fails_the_absorption_demand_rank_bound(
    subspace_parent,
) -> None:
    """A narrow-attention child cannot absorb its own chart transition.

    The absorption demand ``A_b = D_b (E_{b+1} - E_b)`` has to be realized by
    the branch's ordinary writer, whose rank is at most
    ``query_heads * head_dim`` for attention.  With one four-wide query head
    against a 48-wide residual, a third of the demand lands outside that rank,
    so the *default* policy refuses the crossing at chart-compile time — before
    any child tensor is touched.  This is the constraint the design says
    belongs at plan time rather than as a runtime surprise.
    """

    parent, _graph, evidence, _basis = subspace_parent
    narrow = _llama(
        4711,
        CHILD_WIDTH,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
    )

    with pytest.raises(CrossWidthResidualChartRankError) as failure:
        compile_residual_boundary_chart_set(evidence, parent, narrow)
    message = str(failure.value)
    assert "outside the writer's representable rank" in message
    assert "layer001.attention_in" in message

    # The same child passes once the policy admits the measured demand, which
    # proves the bound — not some unrelated failure — is what closed it.
    admitted = compile_residual_boundary_chart_set(
        evidence,
        parent,
        narrow,
        policy=ResidualBoundaryChartPolicy(
            max_unabsorbed_transition_energy_fraction=0.5,
        ),
    )
    demand = admitted.chart(
        residual_boundary_id(1, ATTENTION_INPUT_BOUNDARY)
    ).report["diagnostics"]["absorption_demand"]
    assert demand["writer_representable_rank"] == 4
    assert 0.25 < demand["unabsorbed_energy_fraction"] < 0.5


def test_policy_bounds_are_sealed_and_move_the_chart_lineage(
    subspace_parent,
    child_model,
    chart_set: ResidualBoundaryChartSet,
) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    default = ResidualBoundaryChartPolicy()
    stricter = replace(default, cycle_consistency_weight=5e-2)

    assert default.policy_sha256 != stricter.policy_sha256
    assert chart_set.report["policy_sha256"] == default.policy_sha256

    other = compile_residual_boundary_chart_set(
        evidence,
        parent,
        child_model,
        policy=stricter,
    )
    assert other.report["policy_sha256"] == stricter.policy_sha256
    assert other.set_sha256 != chart_set.set_sha256


def test_a_resumed_chart_compile_is_byte_identical(
    subspace_parent,
    child_model,
    chart_set: ResidualBoundaryChartSet,
) -> None:
    parent, _graph, evidence, _basis = subspace_parent
    again = compile_residual_boundary_chart_set(evidence, parent, child_model)

    assert again.set_sha256 == chart_set.set_sha256
    for left, right in zip(again.charts, chart_set.charts, strict=True):
        assert left.chart_sha256 == right.chart_sha256
        assert torch.equal(left.encoder, right.encoder)
        assert torch.equal(left.decoder, right.decoder)


def test_reverse_reclosure_sweep_is_sealed_and_reports_that_it_has_not_run(
    chart_set: ResidualBoundaryChartSet,
) -> None:
    sweep = plan_reverse_reclosure_sweep(chart_set)

    assert sweep.ordered_boundary_ids == tuple(
        reversed(residual_boundary_ids(LAYERS)[:-1])
    )
    assert sweep.terminal_boundary_id == TERMINAL_BOUNDARY_ID
    assert sweep.charts_frozen is True
    assert sweep.executed is False
    assert sweep.report["writers_only"] is True
    assert sweep.report["joint_global_optimization"] is False
    assert sweep.report["chart_set_sha256"] == chart_set.set_sha256
    assert len(sweep.sweep_sha256) == 64


def test_live_child_recapture_returns_every_requested_boundary(
    child_model,
) -> None:
    ids = input_ids(sequences=2, length=8)
    recapture = LiveChildBoundaryRecapture({"input_ids": ids})
    wanted = residual_boundary_ids(LAYERS)

    states = recapture(child_model, wanted)

    assert set(states) == set(wanted)
    assert recapture.recapture_calls == 1
    assert len(recapture.inputs_sha256) == 64
    for boundary_id, value in states.items():
        assert tuple(value.shape) == (2, 8, CHILD_WIDTH)
        assert bool(torch.isfinite(value).all())
    # The recapture is the live child, so it must move when the child does.
    with torch.no_grad():
        child_model.get_parameter("model.layers.0.mlp.down_proj.weight").add_(0.05)
    moved = recapture(child_model, wanted)
    with torch.no_grad():
        child_model.get_parameter("model.layers.0.mlp.down_proj.weight").sub_(0.05)
    assert not torch.allclose(
        moved[residual_boundary_id(1, ATTENTION_INPUT_BOUNDARY)],
        states[residual_boundary_id(1, ATTENTION_INPUT_BOUNDARY)],
    )
    assert torch.allclose(
        moved[residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY)],
        states[residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY)],
    )


def test_recapture_refuses_inputs_it_cannot_run(child_model) -> None:
    with pytest.raises(CrossWidthResidualChartError, match="input_ids"):
        LiveChildBoundaryRecapture({"attention_mask": torch.ones(2, 4)})
    recapture = LiveChildBoundaryRecapture({"input_ids": input_ids(2, 8)})
    with pytest.raises(CrossWidthResidualChartError, match="at least one boundary"):
        recapture(child_model, ())
