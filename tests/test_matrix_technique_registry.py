"""Contract tests for the matrix campaign's technique dispatch table.

This is the other half of the frozen interface (design §4.3): the harness lane
and the technique lane meet here and nowhere else.  What is pinned:

1. A component resolves to the *parent's real tensor names and geometry*, and a
   cell that cannot name its own parent tensor fails closed instead of quietly
   folding nothing.
2. Level widths are a function of the component's own width, and ``X4`` is a
   lookup that refuses to be guessed -- inventing a 4x layout the campaign never
   adopted would silently change what every X4 cell means.
3. Most-specific-wins resolution, so the token-interface T5 route and the
   body-T5 generalization coexist under one technique id without either lane
   editing the other's file.
4. ``validate_result`` rejects every way a technique could turn a one-component
   cell into something else: a stray tensor, a changed shape, non-finite
   values, or missing conditioning telemetry.
5. The T1 reference kernel actually works.  On a component whose parent write is
   rank-r by construction, contracting to r recovers it to solver precision; on
   a full-rank component the same code loses a measurable amount.  That pairing
   is the same calibration the layer-0 study uses: without it a low measured
   cosine could not be read as a result rather than a harness artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    CellTechniqueResult,
    component_spec,
    describe_registry,
    register,
    registered_pairs,
    resolve,
    resolve_level_width,
    validate_result,
)

HIDDEN = 32
INTERMEDIATE = 80
CONTEXT = 48


class _Evidence:
    """Minimal `MatrixCellEvidence`: row-aligned endpoints, fit/heldout split."""

    def __init__(self, fit, heldout):
        self._fit = dict(fit)
        self._heldout = dict(heldout)

    @property
    def endpoints(self):
        return tuple(sorted(self._fit))

    @property
    def selection_sha256(self):
        return "0" * 64

    @property
    def fit_row_count(self):
        return int(next(iter(self._fit.values())).shape[0])

    @property
    def heldout_row_count(self):
        return int(next(iter(self._heldout.values())).shape[0])

    def fit_rows(self, endpoint):
        return self._fit[endpoint]

    def heldout_rows(self, endpoint):
        return self._heldout[endpoint]


def _parent_state(generator):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {
        "model.embed_tokens.weight": randn(512, HIDDEN),
        "model.embedding_norm.weight": torch.ones(HIDDEN, dtype=torch.float64),
        "model.layers.0.conv.in_proj.weight": randn(3 * HIDDEN, HIDDEN),
        "model.layers.0.conv.conv.weight": randn(HIDDEN, 1, 3),
        "model.layers.0.conv.out_proj.weight": randn(HIDDEN, HIDDEN),
        "model.layers.0.operator_norm.weight": torch.ones(HIDDEN, dtype=torch.float64),
        "model.layers.0.ffn_norm.weight": torch.ones(HIDDEN, dtype=torch.float64),
        "model.layers.0.feed_forward.w1.weight": randn(INTERMEDIATE, HIDDEN),
        "model.layers.0.feed_forward.w2.weight": randn(HIDDEN, INTERMEDIATE),
        "model.layers.0.feed_forward.w3.weight": randn(INTERMEDIATE, HIDDEN),
        "model.layers.2.self_attn.q_proj.weight": randn(HIDDEN, HIDDEN),
        "model.layers.2.self_attn.k_proj.weight": randn(HIDDEN // 2, HIDDEN),
        "model.layers.2.self_attn.v_proj.weight": randn(HIDDEN // 2, HIDDEN),
        "model.layers.2.self_attn.out_proj.weight": randn(HIDDEN, HIDDEN),
        "model.layers.5.feed_forward.w2.weight": randn(HIDDEN, INTERMEDIATE),
    }


def _request(cell_text, state, evidence, *, level_width, knobs=None):
    cell = CellId.parse(cell_text)
    spec = component_spec(cell, state)
    return CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=level_width,
        seed=0,
        parent_weights=state,
        evidence=evidence,
        knobs=dict(knobs or {}),
        workspace=Path("."),
        log=lambda message: None,
    )


# ------------------------------------------------------------ component spec


def test_component_spec_uses_the_parents_real_tensor_names():
    state = _parent_state(torch.Generator().manual_seed(1))
    spec = component_spec(CellId.parse("C13.T1.M16.L05"), state)
    assert spec.weight_keys == ("model.layers.5.feed_forward.w2.weight",)
    assert spec.contraction_axis == "input"
    assert spec.parent_output_width == HIDDEN
    assert spec.parent_input_width == INTERMEDIATE
    assert spec.own_width == INTERMEDIATE
    assert spec.input_endpoint == "mlp.post_swiglu"
    assert spec.write_endpoint == "mlp.down"

    gate = component_spec(CellId.parse("C11.T1.M16.L00"), state)
    assert gate.contraction_axis == "output"
    assert gate.own_width == INTERMEDIATE
    assert gate.write_endpoint == "mlp.gate"

    final_norm = component_spec(CellId.parse("C14.T1.NA.FINAL"), state)
    assert final_norm.weight_keys == ("model.embedding_norm.weight",)


def test_component_spec_fails_closed_on_a_missing_parent_tensor():
    state = _parent_state(torch.Generator().manual_seed(1))
    with pytest.raises(CellTechniqueError, match="missing"):
        component_spec(CellId.parse("C09.T1.M16.L08"), state)


# --------------------------------------------------------------- level width


def test_level_widths_contract_the_components_own_width():
    state = _parent_state(torch.Generator().manual_seed(1))
    spec = component_spec(CellId.parse("C13.T1.M16.L05"), state)
    assert resolve_level_width(spec, "M16") == INTERMEDIATE * 15 // 16
    assert resolve_level_width(spec, "M8") == INTERMEDIATE * 7 // 8
    assert resolve_level_width(spec, "M4") == INTERMEDIATE * 3 // 4


def test_x4_refuses_to_be_guessed_but_the_token_route_is_settled():
    state = _parent_state(torch.Generator().manual_seed(1))
    body = component_spec(CellId.parse("C13.T1.X4.L05"), state)
    with pytest.raises(CellTechniqueError, match="option-bank"):
        resolve_level_width(body, "X4")
    assert resolve_level_width(body, "X4", x4_width=21) == 21
    token = component_spec(CellId.parse("C01.T5.X4"), state)
    assert resolve_level_width(token, "X4") == 4096


def test_norms_have_no_level_axis():
    state = _parent_state(torch.Generator().manual_seed(1))
    norms = component_spec(CellId.parse("C14.T1.NA.L00"), state)
    assert resolve_level_width(norms, "NA") == HIDDEN
    with pytest.raises(CellTechniqueError):
        resolve_level_width(norms, "M16")


# ---------------------------------------------------------------- dispatch


def test_most_specific_registration_wins():
    def general(request):  # pragma: no cover - identity only
        raise AssertionError("general kernel should not be reached")

    def pinned(request):  # pragma: no cover - identity only
        raise AssertionError("pinned kernel should not be reached")

    register("T3", general, replace=True)
    register("T3", pinned, component="C05", replace=True)
    assert resolve("T3", "C05") is pinned
    assert resolve("T3", "C13") is general
    assert ("T3", "C05") in registered_pairs()
    assert any(e["technique"] == "T3" for e in describe_registry()["entries"])


def test_unregistered_technique_names_the_registration_point():
    with pytest.raises(CellTechniqueError, match="register"):
        resolve("T9", "C13")


def test_registering_an_unknown_axis_is_refused():
    with pytest.raises(CellTechniqueError):
        register("T42", lambda request: None)
    with pytest.raises(CellTechniqueError):
        register("T2", lambda request: None, component="C99")


# ------------------------------------------------------------- result guard


def _valid_solve():
    return {
        "solver_path": "x",
        "ridge_relative": 0.0,
        "condition_number": 1.0,
        "condition_cap_saturated": False,
        "prior_norm": 1.0,
        "correction_norm": 0.0,
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 1.0,
        "clamp_engaged": False,
    }


def test_result_guard_rejects_a_two_component_fold():
    state = _parent_state(torch.Generator().manual_seed(1))
    evidence = _Evidence({}, {})
    request = _request("C13.T1.M16.L05", state, evidence, level_width=60)
    result = CellTechniqueResult(
        installed_weights={
            "model.layers.5.feed_forward.w2.weight": state[
                "model.layers.5.feed_forward.w2.weight"
            ],
            "model.layers.0.feed_forward.w1.weight": state[
                "model.layers.0.feed_forward.w1.weight"
            ],
        },
        stored_parameter_count=10,
        solve=_valid_solve(),
    )
    with pytest.raises(CellTechniqueError, match="never folds two components"):
        validate_result(request, result)


def test_result_guard_rejects_a_reshaped_component():
    state = _parent_state(torch.Generator().manual_seed(1))
    request = _request("C13.T1.M16.L05", state, _Evidence({}, {}), level_width=60)
    result = CellTechniqueResult(
        installed_weights={
            "model.layers.5.feed_forward.w2.weight": torch.zeros(
                (HIDDEN, 60), dtype=torch.float64
            )
        },
        stored_parameter_count=10,
        solve=_valid_solve(),
    )
    with pytest.raises(CellTechniqueError, match="copy_exact against it"):
        validate_result(request, result)


def test_result_guard_rejects_missing_conditioning_telemetry():
    state = _parent_state(torch.Generator().manual_seed(1))
    request = _request("C13.T1.M16.L05", state, _Evidence({}, {}), level_width=60)
    solve = _valid_solve()
    del solve["condition_number"]
    result = CellTechniqueResult(
        installed_weights={
            "model.layers.5.feed_forward.w2.weight": state[
                "model.layers.5.feed_forward.w2.weight"
            ]
        },
        stored_parameter_count=10,
        solve=solve,
    )
    with pytest.raises(CellTechniqueError, match="condition_number"):
        validate_result(request, result)


def test_result_guard_rejects_non_finite_weights():
    state = _parent_state(torch.Generator().manual_seed(1))
    request = _request("C13.T1.M16.L05", state, _Evidence({}, {}), level_width=60)
    broken = state["model.layers.5.feed_forward.w2.weight"].clone()
    broken[0, 0] = float("nan")
    result = CellTechniqueResult(
        installed_weights={"model.layers.5.feed_forward.w2.weight": broken},
        stored_parameter_count=10,
        solve=_valid_solve(),
    )
    with pytest.raises(CellTechniqueError, match="non-finite"):
        validate_result(request, result)


def test_route_techniques_may_publish_a_child_instead_of_installing_a_tensor():
    state = _parent_state(torch.Generator().manual_seed(1))
    request = _request("C01.T5.X4", state, _Evidence({}, {}), level_width=4096)
    validate_result(
        request,
        CellTechniqueResult(
            installed_weights={},
            stored_parameter_count=1,
            solve=_valid_solve(),
            child_directory=Path("."),
        ),
    )
    with pytest.raises(CellTechniqueError, match="published no child"):
        validate_result(
            request,
            CellTechniqueResult(
                installed_weights={},
                stored_parameter_count=1,
                solve=_valid_solve(),
            ),
        )


# ------------------------------------------------------- copy_exact control


def test_copy_exact_returns_the_parents_bytes():
    state = _parent_state(torch.Generator().manual_seed(3))
    request = _request("C13.T1.M16.L05", state, _Evidence({}, {}), level_width=60)
    result = resolve("copy_exact", "C13")(request)
    key = "model.layers.5.feed_forward.w2.weight"
    assert torch.equal(result.installed_weights[key], state[key])
    assert result.stored_parameter_count == HIDDEN * INTERMEDIATE
    validate_result(request, result)


# -------------------------------------------------------------- T1 kernel


def _linear_evidence(weight, rows, generator, *, subspace=None):
    """Rows through one linear map, optionally confined to a fixed subspace.

    ``subspace`` is passed in rather than drawn per call: fit and held-out rows
    have to live in the *same* subspace or the "rank-limited by construction"
    control is not a control at all.
    """

    width = int(weight.shape[1])
    inputs = torch.randn((rows, width), dtype=torch.float64, generator=generator)
    if subspace is not None:
        inputs = inputs @ subspace @ subspace.T
    return inputs, inputs @ weight.T


def test_t1_output_axis_recovers_a_rank_limited_component_exactly():
    generator = torch.Generator().manual_seed(11)
    state = _parent_state(generator)
    key = "model.layers.0.feed_forward.w1.weight"
    # Plant a genuinely rank-r parent write: only r output directions carry
    # anything, so contracting the output width to r is lossless by
    # construction.  If this ever regresses, a low measured cosine elsewhere
    # cannot be read as a representability result.
    rank = 12
    left = torch.linalg.qr(
        torch.randn((INTERMEDIATE, rank), dtype=torch.float64, generator=generator)
    )[0]
    right = torch.randn((rank, HIDDEN), dtype=torch.float64, generator=generator)
    state[key] = (left @ right).contiguous()
    fit_in, fit_out = _linear_evidence(state[key], 512, generator)
    held_in, held_out = _linear_evidence(state[key], 128, generator)
    evidence = _Evidence(
        {"mlp.input": fit_in, "mlp.gate": fit_out},
        {"mlp.input": held_in, "mlp.gate": held_out},
    )
    request = _request(
        "C11.T1.M16.L00",
        state,
        evidence,
        level_width=rank,
        knobs={"ridge": 1e-10, "max_prior_delta_ratio": None},
    )
    result = resolve("T1", "C11")(request)
    validate_result(request, result)
    predicted = result.predict(held_in)
    error = torch.linalg.vector_norm(predicted - held_out) / torch.linalg.vector_norm(
        held_out
    )
    assert float(error) < 1e-8
    assert result.stored_parameter_count == INTERMEDIATE * rank + rank * HIDDEN
    assert result.solve["solver_path"]
    assert result.solve["condition_number"] > 0.0
    assert result.solve["clamp_engaged"] is False


def test_t1_output_axis_loses_something_real_on_a_full_rank_component():
    generator = torch.Generator().manual_seed(12)
    state = _parent_state(generator)
    fit_in, fit_out = _linear_evidence(
        state["model.layers.0.feed_forward.w1.weight"], 512, generator
    )
    held_in, held_out = _linear_evidence(
        state["model.layers.0.feed_forward.w1.weight"], 128, generator
    )
    evidence = _Evidence(
        {"mlp.input": fit_in, "mlp.gate": fit_out},
        {"mlp.input": held_in, "mlp.gate": held_out},
    )
    request = _request(
        "C11.T1.M4.L00",
        state,
        evidence,
        level_width=HIDDEN // 2,
        knobs={"ridge": 1e-10, "max_prior_delta_ratio": None},
    )
    result = resolve("T1", "C11")(request)
    predicted = result.predict(held_in)
    error = torch.linalg.vector_norm(predicted - held_out) / torch.linalg.vector_norm(
        held_out
    )
    assert 1e-3 < float(error) < 1.0


def test_t1_input_axis_recovers_a_rank_limited_writer():
    generator = torch.Generator().manual_seed(13)
    state = _parent_state(generator)
    key = "model.layers.5.feed_forward.w2.weight"
    rank = 10
    subspace = torch.linalg.qr(
        torch.randn((INTERMEDIATE, rank), dtype=torch.float64, generator=generator)
    )[0]
    fit_in, fit_out = _linear_evidence(state[key], 512, generator, subspace=subspace)
    held_in, held_out = _linear_evidence(state[key], 128, generator, subspace=subspace)
    evidence = _Evidence(
        {"mlp.post_swiglu": fit_in, "mlp.down": fit_out},
        {"mlp.post_swiglu": held_in, "mlp.down": held_out},
    )
    request = _request(
        "C13.T1.M16.L05",
        state,
        evidence,
        level_width=rank,
        knobs={"ridge": 1e-10, "max_prior_delta_ratio": None},
    )
    result = resolve("T1", "C13")(request)
    validate_result(request, result)
    error = torch.linalg.vector_norm(
        result.predict(held_in) - held_out
    ) / torch.linalg.vector_norm(held_out)
    assert float(error) < 1e-8


def test_t1_correction_is_zero_when_the_write_is_exactly_the_parent_map():
    # Not a curiosity, and worth pinning so nobody later reads the zero as a
    # dead fit: when the sealed write really is ``input @ P.T``, the projected
    # prior is already the reduced-rank optimum on either axis, so the
    # production solver has nothing left to correct and the cell seals
    # ``update_scale 1.0`` with a zero correction norm.  The whole contraction
    # lives in the basis.
    generator = torch.Generator().manual_seed(17)
    state = _parent_state(generator)
    key = "model.layers.0.feed_forward.w1.weight"
    fit_in, fit_out = _linear_evidence(state[key], 256, generator)
    evidence = _Evidence(
        {"mlp.input": fit_in, "mlp.gate": fit_out},
        {"mlp.input": fit_in, "mlp.gate": fit_out},
    )
    result = resolve("T1", "C11")(
        _request(
            "C11.T1.M4.L00",
            state,
            evidence,
            level_width=HIDDEN // 2,
            knobs={"ridge": 1e-8, "max_prior_delta_ratio": 1e-4},
        )
    )
    assert result.solve["correction_norm"] < 1e-9
    assert result.solve["clamp_engaged"] is False
    assert result.solve["update_scale"] == 1.0


def test_t1_reports_the_clamp_when_the_budget_binds():
    generator = torch.Generator().manual_seed(14)
    state = _parent_state(generator)
    key = "model.layers.5.feed_forward.w2.weight"
    fit_in, fit_out = _linear_evidence(state[key], 256, generator)
    # The sealed rows are captured from a bf16 runtime and several components'
    # write endpoints are not literally ``input @ W.T`` (RoPE and the q/k
    # layernorms sit between the projection and its captured write), so a
    # nonzero correction is the normal case rather than the exception.  Perturb
    # the target to put the solve in that regime and check the clamp is
    # observable there -- the receipt's second chain link is unreadable
    # otherwise.
    fit_out = fit_out + 0.25 * torch.randn(
        fit_out.shape, dtype=torch.float64, generator=generator
    )
    evidence = _Evidence(
        {"mlp.post_swiglu": fit_in, "mlp.down": fit_out},
        {"mlp.post_swiglu": fit_in, "mlp.down": fit_out},
    )
    request = _request(
        "C13.T1.M4.L05",
        state,
        evidence,
        level_width=INTERMEDIATE * 3 // 4,
        knobs={"ridge": 1e-8, "max_prior_delta_ratio": 1e-4},
    )
    result = resolve("T1", "C13")(request)
    assert result.solve["correction_norm"] > 0.0
    assert result.solve["clamp_engaged"] is True
    assert result.solve["requested_clamp_ratio"] > result.solve["allowed_clamp_ratio"]
    assert result.solve["update_scale"] < 1.0
    # Unclamped, the same fit walks the whole way and reports it.
    free = resolve("T1", "C13")(
        _request(
            "C13.T1.M4.L05",
            state,
            evidence,
            level_width=INTERMEDIATE * 3 // 4,
            knobs={"ridge": 1e-8, "max_prior_delta_ratio": None},
        )
    )
    assert free.solve["clamp_engaged"] is False
    assert free.solve["condition_number"] > 1.0


def test_t1_refuses_components_it_does_not_model():
    state = _parent_state(torch.Generator().manual_seed(15))
    request = _request(
        "C04.T1.M16.L00", state, _Evidence({}, {}), level_width=HIDDEN - 2
    )
    with pytest.raises(CellTechniqueError, match="needs its own kernel"):
        resolve("T1", "C04")(request)


def test_c14_standalone_norm_refit_recovers_a_planted_gain():
    generator = torch.Generator().manual_seed(16)
    state = _parent_state(generator)
    gain = torch.rand((HIDDEN,), dtype=torch.float64, generator=generator) + 0.5
    rows = torch.randn((768, HIDDEN), dtype=torch.float64, generator=generator)
    held = torch.randn((256, HIDDEN), dtype=torch.float64, generator=generator)

    def normalized(values):
        return values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)

    evidence = _Evidence(
        {"residual.pre": rows, "operator.input": normalized(rows) * gain},
        {"residual.pre": held, "operator.input": normalized(held) * gain},
    )
    request = _request(
        "C14.T1.NA.L00",
        state,
        evidence,
        level_width=HIDDEN,
        knobs={"norm_ridge": 0.0},
    )
    result = resolve("T1", "C14")(request)
    validate_result(request, result)
    fitted = result.installed_weights["model.layers.0.operator_norm.weight"]
    assert torch.allclose(fitted, gain, rtol=1e-8, atol=1e-9)
    assert result.solve["norm_fit_relative_rmse"] < 1e-9
    assert result.solve["norm_validation_relative_rmse"] < 1e-9
