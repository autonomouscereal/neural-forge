from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_closed_form_causal_circuit_fold import (  # noqa: E402
    CausalCircuitFoldIntegrationContract,
    CausalFeatureBlock,
    ClosedFormCausalCircuitCapacityError,
    ClosedFormCausalCircuitFoldError,
    DenseParameterSite,
    FoldingOnlyOperationAudit,
    GatedActivation,
    audit_diagonal_operator_closure,
    fit_attention_value_output,
    fit_bilinear_attention_scores,
    fit_functional_short_conv,
    fit_functional_swiglu,
    fit_predictive_boundary_map,
    fit_rmsnorm_diagonal,
    seal_causal_circuit_fold_plan,
)


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_predictive_boundary_keeps_low_variance_future_causal_direction():
    generator = torch.Generator().manual_seed(7101)
    rows = 160
    noise = torch.randn(rows, generator=generator) * 30.0
    signal = torch.randn(rows, generator=generator) * 0.05
    states = torch.stack((noise, signal), dim=1)
    future = (signal * 200.0).unsqueeze(1)
    result = fit_predictive_boundary_map(
        states[:120].clone(),
        (CausalFeatureBlock("margin", "action_margin", future[:120]),),
        target_width=1,
        validation_states=states[120:].clone(),
        validation_features=(
            CausalFeatureBlock("margin", "action_margin", future[120:]),
        ),
        state_reconstruction_weight=1e-8,
        ridge=1e-10,
        minimum_predictive_energy_fraction=0.90,
        minimum_validation_block_retention=0.90,
    )
    prediction = result.predict_feature("margin", states[120:])
    target = future[120:].to(dtype=torch.float64)
    relative = torch.linalg.vector_norm(prediction - target) / (
        torch.linalg.vector_norm(target)
    )
    assert float(relative.item()) < 1e-4
    assert abs(float(result.encoder[1, 0])) > abs(
        float(result.encoder[0, 0])
    )
    assert result.report["numeric_weight_matching_objective"] is False


def test_predictive_rank_gate_rejects_too_small_target_width():
    generator = torch.Generator().manual_seed(7102)
    states = torch.randn(120, 3, generator=generator)
    features = states[:, :2].clone()
    with pytest.raises(
        ClosedFormCausalCircuitCapacityError,
        match="minimum sealed-evidence width",
    ):
        fit_predictive_boundary_map(
            states[:80].clone(),
            (
                CausalFeatureBlock(
                    "future",
                    "future_residual",
                    features[:80],
                ),
            ),
            target_width=1,
            validation_states=states[80:].clone(),
            validation_features=(
                CausalFeatureBlock(
                    "future",
                    "future_residual",
                    features[80:],
                ),
            ),
            state_reconstruction_weight=0.0,
            ridge=1e-8,
            minimum_predictive_energy_fraction=0.90,
        )


def test_diagonal_closure_audit_exposes_dense_basis_cross_terms():
    dense_basis = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [0.5, -0.5],
        ]
    )
    operators = torch.tensor([[1.0, 5.0, 2.0, 9.0]])
    dense = audit_diagonal_operator_closure(dense_basis, operators)
    assert dense.closed is False
    assert dense.maximum_off_diagonal_energy_fraction > 0.0

    selection_basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    selection = audit_diagonal_operator_closure(
        selection_basis,
        operators,
    )
    assert selection.closed is True
    assert selection.maximum_off_diagonal_energy_fraction == 0.0


def test_rmsnorm_is_fitted_in_child_coordinates():
    generator = torch.Generator().manual_seed(7103)
    rows = torch.randn(80, 3, generator=generator)
    expected_weight = torch.tensor([0.5, 1.5, -0.75])
    normalized = rows * (
        rows.square().mean(dim=1, keepdim=True) + 1e-5
    ).rsqrt()
    targets = normalized * expected_weight
    fitted = fit_rmsnorm_diagonal(
        rows,
        targets,
        epsilon=1e-5,
        ridge=1e-12,
    )
    assert torch.allclose(
        fitted.weight,
        expected_weight.to(dtype=torch.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    assert fitted.report["source_gain_projection_used"] is False


def test_bilinear_qk_fit_recovers_score_function_not_parent_weights():
    generator = torch.Generator().manual_seed(7104)
    query = torch.randn(90, 3, generator=generator)
    key = torch.randn(90, 3, generator=generator)
    effective = torch.tensor(
        [
            [1.2, -0.4, 0.2],
            [0.3, 0.7, -0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    scores = torch.einsum("ni,ij,nj->n", query, effective, key)
    fitted = fit_bilinear_attention_scores(
        query,
        key,
        scores,
        head_dim=2,
        ridge=1e-10,
    )
    prediction = fitted.predict(query, key)
    assert torch.allclose(
        prediction,
        scores.to(dtype=torch.float64),
        atol=2e-6,
        rtol=2e-6,
    )
    assert fitted.report["raw_qk_weight_matching_objective"] is False


def test_value_output_fit_recovers_low_rank_residual_writer():
    generator = torch.Generator().manual_seed(7105)
    contexts = torch.randn(120, 4, generator=generator)
    left = torch.randn(4, 2, generator=generator)
    right = torch.randn(2, 3, generator=generator)
    targets = contexts @ left @ right
    fitted = fit_attention_value_output(
        contexts,
        targets,
        value_width=2,
        ridge=1e-10,
    )
    assert torch.allclose(
        fitted.predict(contexts),
        targets.to(dtype=torch.float64),
        atol=2e-6,
        rtol=2e-6,
    )
    assert fitted.report["raw_value_weight_matching_objective"] is False


def test_functional_swiglu_merges_duplicate_parent_features():
    generator = torch.Generator().manual_seed(7106)
    inputs = torch.randn(180, 3, generator=generator)
    gate_weight = torch.randn(2, 3, generator=generator)
    up_weight = torch.randn(2, 3, generator=generator)
    gates_two = inputs @ gate_weight.T
    ups_two = inputs @ up_weight.T
    gates = torch.cat((gates_two, gates_two), dim=1)
    ups = torch.cat((ups_two, ups_two), dim=1)
    features = torch.nn.functional.silu(gates) * ups
    source_down = torch.tensor(
        [
            [0.4, -0.8, 0.6, 0.3],
            [1.0, 0.2, -0.5, 0.7],
        ]
    )
    targets = features @ source_down.T
    fitted = fit_functional_swiglu(
        inputs,
        gates,
        ups,
        targets,
        target_neurons=2,
        ridge=1e-10,
    )
    prediction = fitted.predict(inputs)
    relative = torch.linalg.vector_norm(
        prediction - targets.to(dtype=torch.float64)
    ) / torch.linalg.vector_norm(targets)
    assert float(relative.item()) < 1e-5
    assert fitted.report["all_source_gate_up_rows_consumed"] is True
    assert fitted.report["complete_parent_residual_target_consumed"] is True


def _situ(gates, ups, gate_beta: float, up_beta: float):
    """chimera-omni ``FeedForward._activate`` under SiTU (``layers.py:350-355``)."""

    return (
        gate_beta * torch.tanh(gates / gate_beta)
        * torch.sigmoid(gates)
        * (up_beta * torch.tanh(ups / up_beta))
    )


def _swiglu_evidence(seed: int):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(180, 3, generator=generator)
    gates = torch.randn(180, 4, generator=generator) * 6.0
    ups = torch.randn(180, 4, generator=generator) * 40.0
    return inputs, gates, ups


def test_the_silu_swiglu_report_body_is_byte_pinned():
    """The SiLU serialization is a pin, not merely a stable default.

    Every sealed SwiGLU receipt in the tree carries this body.  The gated
    activation became a contract constant so a dense SiTU channel mixer could
    fold through the same helper; that must not move one byte of what a SiLU
    fold already emits, and this is what says so.
    """

    inputs, gates, ups = _swiglu_evidence(7106)
    targets = (torch.nn.functional.silu(gates) * ups) @ torch.randn(
        2, 4, generator=torch.Generator().manual_seed(11)
    ).T
    fitted = fit_functional_swiglu(
        inputs, gates, ups, targets, target_neurons=3, ridge=1e-4
    )
    explicit = fit_functional_swiglu(
        inputs,
        gates,
        ups,
        targets,
        target_neurons=3,
        ridge=1e-4,
        activation=GatedActivation(),
    )

    # Recomputed from the pre-widening implementation at 8a4ed56f and pinned
    # here, so this is the historical digest rather than a snapshot of the
    # current code agreeing with itself.
    assert fitted.report["report_sha256"] == (
        "24148a208085a44f2d4b1d1496a17f044b38991b1efeb7e7729ef28e8b0bb751"
    )
    assert "gated_activation" not in fitted.report
    # Passing the default explicitly is the same fold, byte for byte.
    assert dict(explicit.report) == dict(fitted.report)
    assert fitted.activation == GatedActivation(kind="silu")


def test_the_gated_dictionary_compiles_situ_rather_than_approximating_it():
    """SiTU is a different function of the same two rows, and it is fitted as one.

    The parent's own capped-gate arithmetic generates the target here, and the
    evidence is scaled so both caps genuinely bind -- otherwise SiTU degenerates
    to SwiGLU and the test would pass for the wrong reason.  The SiLU
    dictionary is fitted against the *same* rows and misses; the SiTU one lands
    on it.
    """

    inputs, gates, ups = _swiglu_evidence(7108)
    activation = GatedActivation(kind="situ", gate_beta=4.0, up_beta=25.0)
    assert float(
        (gates.abs() > activation.gate_beta).double().mean()
    ) > 0.3, "the gate cap must actually bind"
    assert float((ups.abs() > activation.up_beta).double().mean()) > 0.3
    source_down = torch.randn(2, 4, generator=torch.Generator().manual_seed(13))
    targets = _situ(gates, ups, 4.0, 25.0) @ source_down.T

    situ = fit_functional_swiglu(
        inputs, gates, ups, targets, target_neurons=4, ridge=1e-8,
        activation=activation,
    )
    silu = fit_functional_swiglu(
        inputs, gates, ups, targets, target_neurons=4, ridge=1e-8
    )

    def relative(fold) -> float:
        prediction = fold.predict(inputs)
        difference = prediction - targets.to(dtype=torch.float64)
        return float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(targets)
            ).item()
        )

    assert relative(situ) < relative(silu)
    assert situ.report["gated_activation"] == {
        "kind": "situ",
        "gate_beta": 4.0,
        "up_beta": 25.0,
    }
    assert situ.report["report_sha256"] != silu.report["report_sha256"]
    # ``predict`` replays the sealed constant, not the module default.
    manual = activation(
        inputs.to(dtype=torch.float64) @ situ.gate_weight.T,
        inputs.to(dtype=torch.float64) @ situ.up_weight.T,
    ) @ situ.down_weight.T
    assert torch.equal(manual.contiguous(), situ.predict(inputs))


def test_an_unnamed_gated_activation_fails_closed():
    with pytest.raises(ClosedFormCausalCircuitFoldError, match="gated activation"):
        GatedActivation(kind="gelu")
    for beta in (0.0, -1.0, float("nan")):
        with pytest.raises(ClosedFormCausalCircuitFoldError, match="gate_beta"):
            GatedActivation(kind="situ", gate_beta=beta)
    inputs, gates, ups = _swiglu_evidence(7109)
    with pytest.raises(ClosedFormCausalCircuitFoldError, match="GatedActivation"):
        fit_functional_swiglu(
            inputs,
            gates,
            ups,
            gates,
            target_neurons=2,
            activation="situ",
        )


def test_functional_short_conv_fits_true_multiplicative_fir_channels():
    generator = torch.Generator().manual_seed(7107)
    batch, sequence, width, channels, fir = 5, 8, 3, 2, 3
    inputs = torch.randn(batch, sequence, width, generator=generator)
    b_weight = torch.randn(channels, width, generator=generator)
    c_weight = torch.randn(channels, width, generator=generator)
    x_weight = torch.randn(channels, width, generator=generator)
    b_values = inputs @ b_weight.T
    c_values = inputs @ c_weight.T
    x_values = inputs @ x_weight.T
    kernels = torch.tensor([[0.2, -0.3, 0.7], [0.1, 0.4, -0.2]])

    def context(b_value, c_value, x_value):
        gated = b_value * x_value
        filtered = torch.zeros_like(gated)
        for lag in range(fir):
            tap = kernels[:, fir - 1 - lag]
            if lag == 0:
                filtered.add_(gated * tap)
            else:
                filtered[:, lag:, :].add_(gated[:, :-lag, :] * tap)
        return c_value * filtered

    source_context = context(b_values, c_values, x_values)
    output_weight = torch.randn(4, channels, generator=generator)
    targets = source_context @ output_weight.T
    fitted = fit_functional_short_conv(
        inputs,
        b_values,
        c_values,
        x_values,
        kernels,
        targets,
        target_channels=2,
        ridge=1e-10,
    )
    prediction = fitted.predict(inputs)
    relative = torch.linalg.vector_norm(
        prediction - targets.to(dtype=torch.float64)
    ) / torch.linalg.vector_norm(targets)
    assert float(relative.item()) < 2e-5
    assert (
        fitted.report["dense_basis_kernel_diagonalization_used"] is False
    )


def test_exact_parameter_contract_counts_tied_tensor_once_and_seals_plan():
    sites = (
        DenseParameterSite(
            "embed.weight",
            (7, 3),
            "embedding",
            alias_group="tied_vocab",
        ),
        DenseParameterSite(
            "head.weight",
            (7, 3),
            "lm_head",
            alias_group="tied_vocab",
        ),
        DenseParameterSite("block.weight", (3, 3), "residual_writer"),
    )
    contract = CausalCircuitFoldIntegrationContract(
        parent_cache_sha256=_sha("parent"),
        fit_split_sha256=_sha("fit"),
        validation_split_sha256=_sha("validation"),
        target_topology_sha256=_sha("topology"),
        target_parameter_count=30,
        target_sites=sites,
    )
    tied = torch.randn(7, 3)
    block = torch.randn(3, 3)
    component_body = {"schema": "test", "result": "accepted"}
    import hashlib
    import json

    component_body["report_sha256"] = hashlib.sha256(
        json.dumps(
            component_body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    plan = seal_causal_circuit_fold_plan(
        contract,
        {
            "embed.weight": tied,
            "head.weight": tied,
            "block.weight": block,
        },
        component_reports={"block": component_body},
    )
    assert plan.parameter_audit.target_parameter_count == 30
    assert plan.operation_audit.all_zero is True
    assert plan.as_dict()["contract"]["target_parameter_count"] == 30

    with pytest.raises(
        ClosedFormCausalCircuitFoldError,
        match="not one tensor object",
    ):
        seal_causal_circuit_fold_plan(
            contract,
            {
                "embed.weight": tied,
                "head.weight": tied.clone(),
                "block.weight": block,
            },
            component_reports={"block": component_body},
        )


def test_plan_rejects_any_nonfold_accounting_and_source_has_no_gradient_path():
    sites = (DenseParameterSite("weight", (2, 2), "test"),)
    contract = CausalCircuitFoldIntegrationContract(
        parent_cache_sha256=_sha("parent"),
        fit_split_sha256=_sha("fit"),
        validation_split_sha256=_sha("validation"),
        target_topology_sha256=_sha("topology"),
        target_parameter_count=4,
        target_sites=sites,
    )
    component = {"report_sha256": _sha("component")}
    with pytest.raises(
        ClosedFormCausalCircuitFoldError,
        match="forbidden runtime",
    ):
        seal_causal_circuit_fold_plan(
            contract,
            {"weight": torch.ones(2, 2)},
            component_reports={"component": component},
            operation_audit=FoldingOnlyOperationAudit(optimizer_steps=1),
        )

    import neural_forge.hf_closed_form_causal_circuit_fold as module

    source = inspect.getsource(module)
    forbidden = ".back" + "ward("
    forbidden_optim = "torch." + "optim"
    assert forbidden not in source
    assert forbidden_optim not in source
