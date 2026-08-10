#!/usr/bin/env python3
"""T2 and T3 matrix-cell adapters over the estate's existing chart/basis kernels.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §4.1 names two techniques whose
mechanism was already built and sealed elsewhere in this estate before the
campaign started, and which wave 1 never ran:

    T2  cross-width residual charts      ``ResidualBoundaryChart`` POD (S1)
    T3  balanced controllability /       the balanced-basis phase machinery
        observability basis truncation   (S2 inputs)

This module is **adapter-only**.  It computes no new linear algebra: the
decoder, the chart defect policy, the chart seal, the balanced POD and the
canonical-subspace extraction are all imported from the modules that already
own them (`hf_causal_cross_width_charts`, `balanced_krylov`,
`dynamic_balanced_circuit_basis`), and the deterministic sign-pinned POD basis
is `matrix_technique_registry._pod_basis` -- the same one T1 contracts with.
What this file adds is the translation between the matrix-cell contract
(`CellTechniqueRequest` -> `CellTechniqueResult`) and those kernels' own
signatures, plus the parent-shaped materialization the registry requires.

Two deviations from the production paths, recorded rather than absorbed:

1. **The T2 encoder is the campaign's POD, not the estate's balanced POD.**
   `compile_residual_boundary_chart_set` selects its encoder with
   `build_balanced_circuit_basis`, which needs the parent's residual-facing
   operator inventory, a child ``DecoderLayout``, and one snapshot block per
   evidence split -- objects that exist at *whole-crossing* scope and not at
   one-component-one-site scope.  Rebuilding them inside a cell would be a
   second, unsealed implementation of the crossing compiler.  Using the
   campaign's own `_pod_basis` instead makes the cell's basis *identical to
   T1's at the same component, level and site*, which turns T2-vs-T1 into a
   controlled comparison of exactly one thing: how the second factor is
   produced.  T1 solves it against the component's own target with a
   prior-anchored ridge; T2 transports the parent weight through the chart's
   cycle-anchored ridge decoder and never looks at the target.  Everything
   downstream of the encoder -- `_fit_decoder`, `_boundary_defects`,
   `_seal_chart`, `ResidualBoundaryChart.verify` -- is the estate's, unmodified,
   and the chart's own defect bounds still fail the cell closed.
2. **T3 covers the input axis only.**  Balanced truncation needs a
   controllability lane and an observability lane over one shared state.  For a
   writer (``contraction_axis == "input"``) both exist inside the cell: the
   state is the component's own input, the sealed rows are its controllability
   evidence, and the parent weight *is* the output sensitivity of the map with
   respect to that state, one row per output channel.  For a reader
   (``contraction_axis == "output"``) the state is the component's write and the
   observability operator is the *downstream consumer* -- the attention score
   kernel for C06/C07, the value path for C08 -- which a single-component cell
   does not own and must not guess.  The reader arm therefore refuses with the
   reason instead of substituting an identity observability, which would
   silently reduce balanced truncation to plain POD and produce a confident
   duplicate of T1.

Storage, per `NOTES-MX-E` §4.1's trap.  Both techniques store a ``w x r`` basis
plus an ``r x w'`` second factor, so both store ``r * (in + out)`` against a
dense ``in * out`` -- the same form T1 stores, and on a square component that
is a contraction only below half width.  `stored_parameter_count` says so, and
the T2 result additionally reports the chart's own decoder size under
``solve.chart_fitting_only_parameters`` because the decoder is spent at fit
time and never installed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "T2_TECHNIQUE_ID",
    "T3_TECHNIQUE_ID",
    "t2_cross_width_chart_technique",
    "t3_balanced_basis_technique",
]

T2_TECHNIQUE_ID = "T2"
T3_TECHNIQUE_ID = "T3"

#: The body components these adapters are wired for.  The token-interface
#: components (C01/C02/C15) publish whole children through A9/A11's own sealed
#: route lane and are never installed as a tensor, and C04/C10/C14 are not
#: linear channel maps at all.
_COMPONENTS: tuple[str, ...] = (
    "C03",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C11",
    "C12",
    "C13",
)


def _torch() -> Any:
    import torch

    return torch


def _charted_axis_rows(request: Any, spec: Any) -> tuple[Any, Any, str]:
    """Return ``(fit rows, heldout rows or None, endpoint)`` on the charted axis.

    The level contracts the component's *own* width, which is the input width
    for a writer and the output width for a reader (``ComponentSpec.own_width``
    is the same rule).  The charted state is therefore the component's input
    rows on the input axis and its write rows on the output axis, and both are
    sealed endpoints the harness already loaded -- no endpoint is derived and
    none is invented.
    """

    torch = _torch()
    endpoint = (
        spec.input_endpoint if spec.contraction_axis == "input" else spec.write_endpoint
    )
    fit = request.evidence.fit_rows(endpoint).to(dtype=torch.float64)
    heldout = None
    if int(getattr(request.evidence, "heldout_row_count", 0)) > 0:
        heldout = request.evidence.heldout_rows(endpoint).to(dtype=torch.float64)
    return fit, heldout, str(endpoint)


def _guard_linear_channel_axis(request: Any, spec: Any, technique: str) -> None:
    from .matrix_technique_registry import CellTechniqueError

    if spec.contraction_axis not in ("input", "output"):
        raise CellTechniqueError(
            f"{technique} contracts a linear channel axis; {spec.component_id} "
            f"has axis {spec.contraction_axis!r} and needs its own kernel."
        )
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input/write endpoints for "
            f"{technique}."
        )


def _relative_l2(candidate: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(candidate - reference).item())
    denominator = float(torch.linalg.matrix_norm(reference).item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


# --------------------------------------------------------------------------
# T2 -- cross-width residual charts
# --------------------------------------------------------------------------


def t2_cross_width_chart_technique(request: Any) -> Any:
    """T2: transport one component through a sealed narrow residual chart.

    The chart is a real `ResidualBoundaryChart`: orthonormal encoder, the
    estate's cycle-anchored ridge decoder, the estate's defect measurements
    (orthonormality, cycle defect, norm commutation) checked against
    `ResidualBoundaryChartPolicy`'s own bounds, and the estate's content-address
    seal, re-verified before a single weight is folded.  A chart whose cycle
    defect exceeds the policy bound raises out of `_boundary_defects` and the
    cell seals ``infra`` with that reason -- the technique refuses rather than
    installing a fold nobody could reconstruct.

    The fold itself is the composition the matrix cell's premise forces.  Every
    neighbour is copy_exact and therefore parent-width, so a contracted
    component has to enter child coordinates and come back:

        axis "input"   y = ((x @ E) @ D) @ Wᵀ      W_child = W @ Dᵀ   (out x r)
        axis "output"  y = ((x @ Wᵀ) @ E) @ D      W_child = Eᵀ @ W   (r x in)

    In both arms the installed tensor is the parent-shaped product and the
    *stored* form is the encoder plus the child factor.  The decoder is spent at
    fit time: it produces the child factor and is not installed, so it is
    reported separately rather than counted as storage.
    """

    from .hf_causal_cross_width_charts import (
        CrossWidthResidualChartError,
        ResidualBoundaryChartPolicy,
        _boundary_defects,
        _fit_decoder,
        _seal_chart,
    )
    from .matrix_technique_registry import (
        CellTechniqueError,
        CellTechniqueResult,
        _pod_basis,
    )

    torch = _torch()
    spec = request.component
    _guard_linear_channel_axis(request, spec, "T2 cross-width charts")
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    rank = int(request.level_width)
    fit_rows, heldout_rows, endpoint = _charted_axis_rows(request, spec)
    if int(fit_rows.shape[1]) != int(spec.own_width):
        raise CellTechniqueError(
            f"{spec.component_id} charts endpoint {endpoint!r} at width "
            f"{int(fit_rows.shape[1])}, but its level contracts "
            f"{int(spec.own_width)}."
        )

    policy = ResidualBoundaryChartPolicy(
        chart_ridge_ratio=float(request.knob("chart_ridge_ratio", 1e-8)),
        cycle_consistency_weight=float(
            request.knob("chart_cycle_consistency_weight", 1e-2)
        ),
        max_cycle_defect=float(request.knob("chart_max_cycle_defect", 0.75)),
        max_norm_commutation_fluctuation=float(
            request.knob("chart_max_norm_commutation_fluctuation", 0.25)
        ),
    )
    encoder = _pod_basis(fit_rows, rank)  # (own_width x r), orthonormal
    boundary_id = f"matrix-cell:{request.cell.text}:{endpoint}"
    request.log(
        f"T2 {request.cell.text}: axis={spec.contraction_axis} rank={rank} "
        f"endpoint={endpoint} rows={int(fit_rows.shape[0])} "
        f"width={int(fit_rows.shape[1])}"
    )
    try:
        decoder, decoder_report = _fit_decoder(encoder, fit_rows, policy)
        diagnostics = _boundary_defects(
            boundary_id=boundary_id,
            encoder=encoder,
            decoder=decoder,
            fit_rows=fit_rows,
            next_encoder=None,
            previous_decoder=None,
            capacity=rank,
            policy=policy,
        )
        chart = _seal_chart(
            boundary_id=boundary_id,
            encoder=encoder,
            decoder=decoder,
            interface_locked=False,
            diagnostics={**diagnostics, "decoder": decoder_report},
            lineage={
                "matrix_cell_id": request.cell.text,
                "component": spec.component_id,
                "site": spec.site,
                "level": request.cell.level,
                "charted_endpoint": endpoint,
                "evidence_selection_sha256": str(
                    request.evidence.selection_sha256
                ),
                "encoder_source": (
                    "matrix_technique_registry._pod_basis of the charted "
                    "endpoint's FIT rows (see module docstring, deviation 1)"
                ),
                "policy_sha256": policy.policy_sha256,
            },
        ).verify()
    except CrossWidthResidualChartError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    if spec.contraction_axis == "input":
        # W_child = W @ Dᵀ  (out x r); installed = W_child @ Eᵀ = W @ Dᵀ @ Eᵀ.
        child_factor = (parent @ chart.decoder.transpose(0, 1)).contiguous()
        weight = (child_factor @ chart.encoder.transpose(0, 1)).contiguous()
    else:
        # W_child = Eᵀ @ W  (r x in); installed = Dᵀ @ W_child = Dᵀ @ Eᵀ @ W.
        child_factor = (chart.encoder.transpose(0, 1) @ parent).contiguous()
        weight = (chart.decoder.transpose(0, 1) @ child_factor).contiguous()
    if tuple(weight.shape) != tuple(parent.shape):
        raise CellTechniqueError(
            f"T2 materialized {tuple(weight.shape)} for a parent of "
            f"{tuple(parent.shape)}."
        )

    stored = int(chart.encoder.numel() + child_factor.numel())
    cycle_heldout = (
        None
        if heldout_rows is None
        else _relative_l2((heldout_rows @ chart.encoder) @ chart.decoder, heldout_rows)
    )
    solve = {
        "solver_path": "cross_width_chart_cycle_anchored_ridge_decoder",
        # The chart's ridge is a *ratio* of the design's spectral scale, which
        # is exactly what this field means everywhere else in the campaign.
        "ridge_relative": float(policy.chart_ridge_ratio),
        "condition_number": float(
            torch.linalg.cond(chart.encoder.T @ chart.encoder).item()
        ),
        "condition_cap_saturated": False,
        "prior_norm": float(torch.linalg.matrix_norm(parent).item()),
        "correction_norm": float(torch.linalg.matrix_norm(weight - parent).item()),
        # No prior-delta clamp lane exists in the chart path; the schema's null
        # says so rather than an invented budget.
        "requested_clamp_ratio": None,
        "allowed_clamp_ratio": None,
        "clamp_engaged": False,
        "omp": None,
        "chart_sha256": chart.chart_sha256,
        "chart_boundary_id": boundary_id,
        "chart_source_width": int(chart.source_width),
        "chart_target_width": int(chart.target_width),
        "chart_policy_sha256": policy.policy_sha256,
        "chart_cycle_defect_fit": float(diagnostics["cycle_defect"]),
        "chart_cycle_defect_heldout": cycle_heldout,
        "chart_encoder_orthonormality_max_abs_error": float(
            diagnostics["encoder_orthonormality_max_abs_error"]
        ),
        "chart_norm_commutation_row_fluctuation": float(
            diagnostics["norm_commutation_row_fluctuation"]
        ),
        # How far the fitted decoder is from the encoder transpose, relative.
        # For an orthonormal encoder the regularized reciprocal *is* Eᵀ, and the
        # cycle-anchored ridge only moves it when the fit rows say it should.
        # When this is ~0 the chart is a plain orthogonal projection and T2's
        # fold coincides with a POD truncation -- which is a finding about this
        # component, not a defect, but it has to be visible in the receipt
        # rather than inferred from two cells landing on the same number.
        "chart_decoder_vs_encoder_transpose_relative_deviation": float(
            (
                torch.linalg.matrix_norm(
                    chart.decoder - chart.encoder.transpose(0, 1)
                )
                / torch.linalg.matrix_norm(chart.encoder)
            ).item()
        ),
        "charted_endpoint": endpoint,
        "contraction_basis_rank": rank,
        # Spent at fit time, never installed: the decoder produces the child
        # factor and then stops existing.  Reported so the storage number below
        # cannot be read as if the decoder were free *and* absent.
        "chart_fitting_only_parameters": int(chart.decoder.numel()),
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ weight.T

    return CellTechniqueResult(
        installed_weights={key: weight},
        stored_parameter_count=stored,
        solve=solve,
        predict=predict,
        artifact_identities={"t2_chart_sha256": chart.chart_sha256},
        notes=(
            f"T2 cross-width residual chart on {endpoint} at rank {rank} of "
            f"{spec.own_width}; encoder POD + estate cycle-anchored ridge "
            f"decoder, cycle defect {float(diagnostics['cycle_defect']):.6f}."
        ),
    )


# --------------------------------------------------------------------------
# T3 -- balanced controllability/observability basis truncation
# --------------------------------------------------------------------------


def t3_balanced_basis_technique(request: Any) -> Any:
    """T3: truncate to the balanced controllability/observability subspace.

    Balanced truncation keeps the directions that are simultaneously *reachable*
    by the evidence and *visible* at the output, rather than the directions that
    merely carry the most input energy -- which is what a plain POD keeps and
    what T1 and T2 both contract with.  That is the whole content of the
    comparison: same storage form, same rank, same site, a basis chosen by a
    different criterion.

    The two lanes, both owned by the cell and neither invented:

    * controllability -- the sealed FIT rows at the component's input endpoint;
    * observability   -- the parent weight, whose row ``j`` is exactly
      ``d y_j / d x`` for this linear map, so the observability Gram is
      ``Wᵀ W`` and no downstream model has to be consulted.

    `build_balanced_krylov_basis` computes the balanced POD from the compact
    Gramian factors with the fallback **disabled** (``allow_svd_fallback=False``)
    -- a rank-deficient or ill-conditioned Hankel factor has to refuse the cell,
    not silently degrade into the feature-space SVD that would make this
    technique a duplicate of T1.  `_canonical_subspace_basis` then pins the
    frame so two runs of the same cell are the same cell.
    """

    from .balanced_krylov import BalancedKrylovError
    from .dynamic_balanced_circuit_basis import (
        BalancedCircuitBasisError,
        _canonical_subspace_basis,
        _retained_energy,
    )
    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    torch = _torch()
    from .balanced_krylov import build_balanced_krylov_basis

    spec = request.component
    _guard_linear_channel_axis(request, spec, "T3 balanced truncation")
    if spec.contraction_axis != "input":
        raise CellTechniqueError(
            f"T3 balanced truncation needs a controllability lane and an "
            f"observability lane over one shared state.  {spec.component_id} "
            "contracts its output axis, where the observability operator is the "
            "downstream consumer this cell does not own; substituting an "
            "identity observability would reduce balanced truncation to plain "
            "POD and duplicate T1 under a different name."
        )
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    rank = int(request.level_width)
    fit_rows, heldout_rows, endpoint = _charted_axis_rows(request, spec)
    rank_tolerance = float(request.knob("balanced_rank_tolerance", 1e-12))
    regularization = float(request.knob("balanced_gram_regularization", 1e-10))
    max_condition = float(request.knob("balanced_max_condition_number", 1e12))
    request.log(
        f"T3 {request.cell.text}: rank={rank} endpoint={endpoint} "
        f"rows={int(fit_rows.shape[0])} width={int(fit_rows.shape[1])} "
        f"observability={tuple(int(v) for v in parent.shape)}"
    )

    width = int(fit_rows.shape[1])
    ridge = torch.eye(width, dtype=torch.float64) * (regularization / width)
    controllability_sqrt = _psd_sqrt(fit_rows.T @ fit_rows + ridge)
    observability_sqrt = _psd_sqrt(parent.T @ parent + ridge)
    try:
        balanced = build_balanced_krylov_basis(
            controllability_sqrt.numpy(),
            rank,
            output_sensitivities=observability_sqrt.numpy(),
            center=False,
            svd_atol=0.0,
            svd_rtol=rank_tolerance,
            max_condition_number=max_condition,
            allow_svd_fallback=False,
        )
        basis = _canonical_subspace_basis(
            torch.from_numpy(balanced.decoder.copy()).to(dtype=torch.float64),
            rank,
        )
    except (BalancedKrylovError, BalancedCircuitBasisError, ValueError) as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    child_factor = (parent @ basis).contiguous()  # (out x r)
    weight = (child_factor @ basis.transpose(0, 1)).contiguous()
    stored = int(basis.numel() + child_factor.numel())
    controllability_gram = (fit_rows.T @ fit_rows).contiguous()
    observability_gram = (parent.T @ parent).contiguous()
    heldout_retained = None
    if heldout_rows is not None:
        heldout_retained = _retained_energy(
            basis, (heldout_rows.T @ heldout_rows).contiguous()
        )
    solve = {
        "solver_path": "balanced_krylov_controllability_observability_truncation",
        "ridge_relative": regularization,
        "condition_number": float(
            torch.linalg.cond(basis.T @ controllability_gram @ basis).item()
        ),
        "condition_cap_saturated": False,
        "prior_norm": float(torch.linalg.matrix_norm(parent).item()),
        "correction_norm": float(torch.linalg.matrix_norm(weight - parent).item()),
        "requested_clamp_ratio": None,
        "allowed_clamp_ratio": None,
        "clamp_engaged": False,
        "omp": None,
        "contraction_basis_rank": rank,
        "balanced_state_endpoint": endpoint,
        "balanced_rank_tolerance": rank_tolerance,
        "balanced_gram_regularization": regularization,
        "balanced_svd_fallback_allowed": False,
        "balanced_hankel_rank": int(balanced.rank),
        "fit_retained_energy_fraction": _retained_energy(basis, controllability_gram),
        "heldout_retained_energy_fraction": heldout_retained,
        "observability_retained_energy_fraction": _retained_energy(
            basis, observability_gram
        ),
        "observability_lane": "parent weight rows as d(output)/d(state)",
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ weight.T

    return CellTechniqueResult(
        installed_weights={key: weight},
        stored_parameter_count=stored,
        solve=solve,
        predict=predict,
        notes=(
            f"T3 balanced controllability/observability truncation on "
            f"{endpoint} at rank {rank} of {spec.own_width}; retained "
            f"{solve['fit_retained_energy_fraction']:.6f} of the fit "
            f"controllability energy and "
            f"{solve['observability_retained_energy_fraction']:.6f} of the "
            f"observability energy."
        ),
    )


def _psd_sqrt(gram: Any) -> Any:
    """Symmetric PSD square root, through the balanced-basis module's own kernel."""

    from .dynamic_balanced_circuit_basis import _psd_sqrt_and_rank

    square_root, _rank, _spectrum = _psd_sqrt_and_rank(
        gram,
        name="matrix cell Gram",
        rank_tolerance=1e-12,
    )
    return square_root


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def _register_chart_techniques() -> None:
    """Wire T2/T3 into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    for component in _COMPONENTS:
        register(
            T2_TECHNIQUE_ID,
            t2_cross_width_chart_technique,
            component=component,
            replace=True,
        )
        register(
            T3_TECHNIQUE_ID,
            t3_balanced_basis_technique,
            component=component,
            replace=True,
        )


_register_chart_techniques()


def describe_chart_techniques() -> Mapping[str, Any]:
    """Serializable inventory of what this adapter wires, for a cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-charts-v1",
        "techniques": {
            T2_TECHNIQUE_ID: {
                "name": "cross_width_residual_charts",
                "kernel": "hf_causal_cross_width_charts.ResidualBoundaryChart",
                "encoder": "matrix_technique_registry._pod_basis",
                "decoder": "hf_causal_cross_width_charts._fit_decoder",
                "axes": ("input", "output"),
            },
            T3_TECHNIQUE_ID: {
                "name": "balanced_basis_truncation",
                "kernel": "balanced_krylov.build_balanced_krylov_basis",
                "canonicalization": (
                    "dynamic_balanced_circuit_basis._canonical_subspace_basis"
                ),
                "axes": ("input",),
            },
        },
        "components": _COMPONENTS,
    }
