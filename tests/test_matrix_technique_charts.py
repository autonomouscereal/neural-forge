"""Contract tests for the T2/T3 matrix-cell adapters.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §4.1 names both techniques and wave
1 ran neither, so these are the first tests that exist for them.  What is
pinned here is the *adapter*, not the estate kernels underneath -- those have
their own suites -- and specifically the four ways an adapter can be wrong in a
manner a receipt would not show:

1. The installed tensor is parent-shaped and finite, so every copy_exact
   neighbour still composes against it (`validate_result`).
2. ``stored_parameter_count`` is the honest ``r * (in + out)`` of the stored
   form, not the size of the parent-shaped product that gets installed.  This
   is the trap `NOTES-MX-E` §4.1 measured: a level names a rank, and a rank is
   not a size.
3. The technique is *not* a relabelled copy of another one.  T2 shares T1's
   basis by construction, so what has to differ is the second factor; T3's
   whole claim is that a balanced basis is not the POD basis.  Both are
   asserted against real numbers.
4. A refusal is a refusal.  T3's reader arm has no observability lane it owns,
   and it must say so rather than substitute an identity and silently
   duplicate T1.
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
from neural_forge.matrix_technique_charts import (  # noqa: E402
    describe_chart_techniques,
    t2_cross_width_chart_technique,
    t3_balanced_basis_technique,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    resolve,
    validate_result,
)

HIDDEN = 32
INTERMEDIATE = 80


class _Evidence:
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
        "model.layers.0.conv.out_proj.weight": randn(HIDDEN, HIDDEN),
        "model.layers.5.feed_forward.w2.weight": randn(HIDDEN, INTERMEDIATE),
        "model.layers.2.self_attn.q_proj.weight": randn(HIDDEN, HIDDEN),
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


def _writer_evidence(generator, *, rows=384, heldout=96, width=INTERMEDIATE):
    """Rows with a decaying spectrum, so a truncation has something to lose."""

    scale = torch.logspace(0.0, -2.0, width, dtype=torch.float64)
    fit = torch.randn((rows, width), dtype=torch.float64, generator=generator) * scale
    held = torch.randn((heldout, width), dtype=torch.float64, generator=generator) * scale
    return fit, held


# ------------------------------------------------------------------ wiring


def test_both_techniques_are_wired_into_the_dispatch_table():
    assert resolve("T2", "C05") is t2_cross_width_chart_technique
    assert resolve("T2", "C13") is t2_cross_width_chart_technique
    assert resolve("T3", "C05") is t3_balanced_basis_technique
    inventory = describe_chart_techniques()
    assert inventory["techniques"]["T2"]["axes"] == ("input", "output")
    # T3 declares one axis, and the declaration is the same statement its
    # reader arm raises with.
    assert inventory["techniques"]["T3"]["axes"] == ("input",)


# ---------------------------------------------------------------------- T2


def test_t2_installs_a_parent_shaped_chart_transport_and_counts_what_it_stores():
    generator = torch.Generator().manual_seed(7)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator)
    parent = state["model.layers.5.feed_forward.w2.weight"]
    evidence = _Evidence(
        {"mlp.post_swiglu": fit, "mlp.down": fit @ parent.T},
        {"mlp.post_swiglu": held, "mlp.down": held @ parent.T},
    )
    rank = 60
    request = _request("C13.T2.M16.L05", state, evidence, level_width=rank)
    result = t2_cross_width_chart_technique(request)
    validate_result(request, result)

    weight = result.installed_weights["model.layers.5.feed_forward.w2.weight"]
    assert tuple(weight.shape) == (HIDDEN, INTERMEDIATE)
    # The stored form is the encoder plus the child factor -- r*(in+out) -- and
    # NOT the parent-shaped product that lands in the child.
    assert result.stored_parameter_count == rank * (INTERMEDIATE + HIDDEN)
    assert result.stored_parameter_count != weight.numel()
    # The decoder is spent at fit time and reported separately; counting it as
    # storage would be as wrong as pretending it was free.
    assert result.solve["chart_fitting_only_parameters"] == rank * INTERMEDIATE

    assert result.solve["chart_source_width"] == INTERMEDIATE
    assert result.solve["chart_target_width"] == rank
    assert len(result.solve["chart_sha256"]) == 64
    assert result.solve["chart_encoder_orthonormality_max_abs_error"] < 1e-9
    # A held-out cycle defect is a real generalization reading, not the fit one.
    assert 0.0 <= result.solve["chart_cycle_defect_heldout"] < 1.0


def test_t2_is_not_t1_wearing_a_chart():
    """Same basis, same rank, same rows: the second factor is the whole delta.

    T2 shares T1's POD encoder by construction (module docstring, deviation 1),
    so the two differ only in how the second factor is produced -- T1 solves it
    against the component's own target, T2 transports the parent through the
    chart's decoder and never looks at the target.

    On a *noiseless* linear component the two answers converge: the ridge
    refit's optimum on a truncated input is the projection, and the cycle
    anchored ridge decoder of an orthonormal encoder is its transpose.  Measured
    here at rank 60 of 80 the installed weights differ by ~5e-9 relative.  So
    this test pins that they are distinguishable at all, and the receipt field
    below is what makes the coincidence readable in a real cell instead of
    inferred from two rows landing on the same number.
    """

    generator = torch.Generator().manual_seed(11)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator)
    parent = state["model.layers.5.feed_forward.w2.weight"]
    evidence = _Evidence(
        {"mlp.post_swiglu": fit, "mlp.down": fit @ parent.T},
        {"mlp.post_swiglu": held, "mlp.down": held @ parent.T},
    )
    rank = 60
    t2 = t2_cross_width_chart_technique(
        _request("C13.T2.M16.L05", state, evidence, level_width=rank)
    )
    t1 = resolve("T1", "C13")(
        _request("C13.T1.M16.L05", state, evidence, level_width=rank)
    )
    key = "model.layers.5.feed_forward.w2.weight"
    delta = float(
        torch.linalg.matrix_norm(t2.installed_weights[key] - t1.installed_weights[key])
    )
    assert delta > 1e-12
    # ... and they store the same amount, which is what makes the comparison
    # a comparison of criteria rather than of budgets.
    assert t2.stored_parameter_count == t1.stored_parameter_count
    # The receipt says how close the chart came to being a plain projection.
    assert (
        t2.solve["chart_decoder_vs_encoder_transpose_relative_deviation"] >= 0.0
    )


def test_t2_reader_axis_transports_through_the_write_endpoint():
    generator = torch.Generator().manual_seed(13)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator, width=HIDDEN)
    parent = state["model.layers.2.self_attn.q_proj.weight"]
    evidence = _Evidence(
        {"operator.input": fit, "attention.q_linear": fit @ parent.T},
        {"operator.input": held, "attention.q_linear": held @ parent.T},
    )
    rank = 30
    request = _request("C06.T2.M16.L02", state, evidence, level_width=rank)
    result = t2_cross_width_chart_technique(request)
    validate_result(request, result)
    assert result.solve["charted_endpoint"] == "attention.q_linear"
    assert result.stored_parameter_count == rank * (HIDDEN + HIDDEN)


def test_t2_refuses_a_component_that_is_not_a_linear_channel_map():
    generator = torch.Generator().manual_seed(17)
    state = _parent_state(generator)
    state["model.layers.0.conv.conv.weight"] = torch.randn(
        (HIDDEN, 1, 3), dtype=torch.float64, generator=generator
    )
    evidence = _Evidence({"operator.input": torch.zeros((4, HIDDEN), dtype=torch.float64)}, {})
    with pytest.raises(CellTechniqueError, match="linear channel axis"):
        t2_cross_width_chart_technique(
            _request("C04.T2.M16.L00", state, evidence, level_width=30)
        )


# ---------------------------------------------------------------------- T3


def test_t3_balanced_basis_truncates_and_reports_both_energies():
    generator = torch.Generator().manual_seed(19)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator)
    parent = state["model.layers.5.feed_forward.w2.weight"]
    evidence = _Evidence(
        {"mlp.post_swiglu": fit, "mlp.down": fit @ parent.T},
        {"mlp.post_swiglu": held, "mlp.down": held @ parent.T},
    )
    rank = 60
    request = _request("C13.T3.M16.L05", state, evidence, level_width=rank)
    result = t3_balanced_basis_technique(request)
    validate_result(request, result)

    assert result.stored_parameter_count == rank * (INTERMEDIATE + HIDDEN)
    assert result.solve["balanced_svd_fallback_allowed"] is False
    for field in (
        "fit_retained_energy_fraction",
        "heldout_retained_energy_fraction",
        "observability_retained_energy_fraction",
    ):
        assert 0.0 < result.solve[field] <= 1.0
    # Balanced truncation trades input energy for output visibility, so the
    # observability retention is what it is optimizing and must not be the
    # accidental by-product a plain POD would leave.
    assert result.solve["observability_retained_energy_fraction"] > 0.5


def test_t3_basis_is_not_the_pod_basis():
    """The whole claim of T3 is that a different criterion picks a different
    subspace.  If the balanced basis coincided with the POD basis the cell
    would be T1 with extra telemetry, so the two folds have to differ.
    """

    generator = torch.Generator().manual_seed(23)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator)
    parent = state["model.layers.5.feed_forward.w2.weight"]
    evidence = _Evidence(
        {"mlp.post_swiglu": fit, "mlp.down": fit @ parent.T},
        {"mlp.post_swiglu": held, "mlp.down": held @ parent.T},
    )
    rank = 40
    t3 = t3_balanced_basis_technique(
        _request("C13.T3.M16.L05", state, evidence, level_width=rank)
    )
    t1 = resolve("T1", "C13")(
        _request("C13.T1.M16.L05", state, evidence, level_width=rank)
    )
    key = "model.layers.5.feed_forward.w2.weight"
    delta = float(
        torch.linalg.matrix_norm(t3.installed_weights[key] - t1.installed_weights[key])
    )
    assert delta > 1e-9


def test_t3_refuses_the_reader_axis_rather_than_faking_an_observability_lane():
    generator = torch.Generator().manual_seed(29)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator, width=HIDDEN)
    parent = state["model.layers.2.self_attn.q_proj.weight"]
    evidence = _Evidence(
        {"operator.input": fit, "attention.q_linear": fit @ parent.T},
        {"operator.input": held, "attention.q_linear": held @ parent.T},
    )
    with pytest.raises(CellTechniqueError, match="downstream consumer"):
        t3_balanced_basis_technique(
            _request("C06.T3.M16.L02", state, evidence, level_width=30)
        )


def test_t3_is_deterministic_across_runs_of_the_same_cell():
    generator = torch.Generator().manual_seed(31)
    state = _parent_state(generator)
    fit, held = _writer_evidence(generator)
    parent = state["model.layers.5.feed_forward.w2.weight"]
    evidence = _Evidence(
        {"mlp.post_swiglu": fit, "mlp.down": fit @ parent.T},
        {"mlp.post_swiglu": held, "mlp.down": held @ parent.T},
    )
    key = "model.layers.5.feed_forward.w2.weight"
    first = t3_balanced_basis_technique(
        _request("C13.T3.M16.L05", state, evidence, level_width=50)
    )
    second = t3_balanced_basis_technique(
        _request("C13.T3.M16.L05", state, evidence, level_width=50)
    )
    assert torch.equal(first.installed_weights[key], second.installed_weights[key])
