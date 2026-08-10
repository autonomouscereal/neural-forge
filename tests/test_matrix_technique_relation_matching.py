"""Stage-1 proofs for T7 attention-relation and T8 Gram-relation matching.

Proven here, on synthetic sealed-evidence-shaped tensors with no optimizer, no
gradient and no model execution:

* the Stein primitive is the design's normal equations and recovers an exact
  solution when one exists;
* T7 at no contraction reproduces the parent's *scores* — not merely its
  factors, which the score kernel is invariant to — through the published child
  weights, softmax scale included;
* T7's Procrustes gauge and factor balancing leave the score kernel invariant,
  and GQA groups get one shared child K;
* T8 at no contraction returns the parent weight exactly, and its row
  weighting really is the scale-free objective (a rescaled evidence row must
  not move the fit the way a plain ridge moves);
* every refusal both techniques name actually closes.
"""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from neural_forge.matrix_technique_body_routes import (  # noqa: E402
    select_kept_channels as body_route_select_kept_channels,
)
from neural_forge.matrix_technique_relation_matching import (  # noqa: E402
    MATRIX_TECHNIQUE_ENTRY_SCHEMA,
    T7_DEFAULT_ITERATION_CAP,
    T7_RELATION_RECORD_SCHEMA,
    T8_RELATION_RECORD_SCHEMA,
    RelationMatchConditioningError,
    RelationMatchDisciplineError,
    RelationMatchError,
    RelationMatchEvidenceError,
    RelationMatchGeometryError,
    fit_attention_relation_match,
    fit_gram_relation_match,
    matrix_technique_registry_entries,
    matrix_technique_registry_entry,
    select_kept_channels,
    solve_two_sided_stein_ridge,
    solve_two_sided_stein_ridge_gram,
    verify_sealed_relation_record,
)


def _normal(seed: int, rows: int, columns: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=torch.float64)


def _captured_split(
    seed: int,
    fit_rows: int,
    heldout_rows: int,
    channels: int,
    latent: int,
    *,
    noise: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    total = fit_rows + heldout_rows
    factors = torch.randn(total, latent, generator=generator, dtype=torch.float64)
    mixing = torch.randn(latent, channels, generator=generator, dtype=torch.float64)
    perturbation = torch.randn(
        total, channels, generator=generator, dtype=torch.float64
    )
    rows = factors @ mixing + noise * perturbation
    return rows[:fit_rows].contiguous(), rows[fit_rows:].contiguous()


def _parent_scores(
    rows: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head: int,
    head_dim: int,
    group: int,
) -> torch.Tensor:
    left = query[head * head_dim : (head + 1) * head_dim, :]
    kv_head = head // group
    right = key[kv_head * head_dim : (kv_head + 1) * head_dim, :]
    return (rows @ left.transpose(0, 1)) @ (
        rows @ right.transpose(0, 1)
    ).transpose(0, 1) / math.sqrt(head_dim)


# ---------------------------------------------------------------------------
# The Stein primitive
# ---------------------------------------------------------------------------


def test_stein_solve_recovers_an_exactly_representable_solution():
    left = _normal(101, 60, 6)
    right = _normal(102, 5, 9)
    solution = _normal(103, 6, 5)
    solved, report = solve_two_sided_stein_ridge(
        left, right, left @ solution @ right, ridge=0.0
    )
    relative = float(
        torch.linalg.matrix_norm(solved - solution, ord="fro").item()
    ) / float(torch.linalg.matrix_norm(solution, ord="fro").item())
    assert relative < 1e-9
    assert report["suppressed_direction_count"] == 0
    assert report["left_rank"] == 6


def test_stein_solve_satisfies_the_designs_normal_equations():
    left = _normal(104, 40, 5)
    right = _normal(105, 4, 7)
    target = _normal(106, 40, 7)
    ridge = 0.25
    solved, _report = solve_two_sided_stein_ridge(
        left, right, target, ridge=ridge
    )
    left_gram = left.transpose(0, 1) @ left
    right_gram = right @ right.transpose(0, 1)
    residual = (
        left_gram @ solved @ right_gram
        + ridge * solved
        - left.transpose(0, 1) @ target @ right.transpose(0, 1)
    )
    assert float(torch.linalg.matrix_norm(residual, ord="fro").item()) < 1e-8


def test_stein_solve_reuses_a_supplied_left_eigendecomposition():
    left = _normal(107, 30, 5)
    right = _normal(108, 4, 6)
    target = _normal(109, 30, 6)
    left_gram = left.transpose(0, 1) @ left
    right_gram = right @ right.transpose(0, 1)
    cross = left.transpose(0, 1) @ target @ right.transpose(0, 1)
    plain, _first = solve_two_sided_stein_ridge_gram(
        left_gram, right_gram, cross, ridge=0.1
    )
    values, vectors = torch.linalg.eigh(left_gram)
    reused, _second = solve_two_sided_stein_ridge_gram(
        left_gram,
        right_gram,
        cross,
        ridge=0.1,
        left_eigen=(values.clamp_min(0.0), vectors),
    )
    assert torch.allclose(plain, reused, atol=1e-12)


def test_stein_solve_rejects_mismatched_shapes():
    with pytest.raises(RelationMatchGeometryError, match="row counts differ"):
        solve_two_sided_stein_ridge(
            _normal(110, 10, 4), _normal(111, 3, 5), _normal(112, 9, 5)
        )
    with pytest.raises(RelationMatchGeometryError, match="column counts differ"):
        solve_two_sided_stein_ridge(
            _normal(113, 10, 4), _normal(114, 3, 5), _normal(115, 10, 6)
        )


# ---------------------------------------------------------------------------
# T7 exact-identity and invariance gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2), (6, 2)])
def test_no_contraction_reproduces_parent_scores(heads, kv_heads):
    hidden, head_dim = 24, 6
    query = _normal(120 + heads, heads * head_dim, hidden)
    key = _normal(140 + kv_heads, kv_heads * head_dim, hidden)
    rows, held = _captured_split(160, 240, 40, hidden, hidden)
    fit = fit_attention_relation_match(
        query_weight=query,
        key_weight=key,
        evidence_rows=rows,
        head_dim=head_dim,
        query_heads=heads,
        key_value_heads=kv_heads,
        level="M1",
        site="L02",
    )
    assert fit.child_head_dim == head_dim
    assert fit.calibration["score_kernel_relative_error_max"] < 1e-6
    assert fit.softmax_scale_compensation == 1.0
    group = heads // kv_heads
    for head in range(heads):
        reference = _parent_scores(
            held, query, key, head=head, head_dim=head_dim, group=group
        )
        child = fit.score_kernel(held, head=head)
        relative = float(
            torch.linalg.matrix_norm(child - reference, ord="fro").item()
        ) / float(torch.linalg.matrix_norm(reference, ord="fro").item())
        assert relative < 1e-6


def test_child_key_is_shared_across_a_grouped_query_head_group():
    hidden, head_dim, heads, kv_heads = 32, 8, 6, 2
    fit = fit_attention_relation_match(
        query_weight=_normal(170, heads * head_dim, hidden),
        key_weight=_normal(171, kv_heads * head_dim, hidden),
        evidence_rows=_captured_split(172, 300, 0, hidden, hidden)[0],
        head_dim=head_dim,
        query_heads=heads,
        key_value_heads=kv_heads,
        level="M4",
    )
    assert fit.key_weight.shape == (kv_heads * fit.child_head_dim, hidden)
    assert fit.query_weight.shape == (heads * fit.child_head_dim, hidden)
    assert [record["key_value_head"] for record in fit.per_head] == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_softmax_scale_compensation_is_folded_into_the_child_query():
    hidden, head_dim, heads = 32, 8, 2
    common = {
        "query_weight": _normal(180, heads * head_dim, hidden),
        "key_weight": _normal(181, heads * head_dim, hidden),
        "evidence_rows": _captured_split(182, 300, 0, hidden, hidden)[0],
        "head_dim": head_dim,
        "query_heads": heads,
        "key_value_heads": heads,
        "level": "M4",
    }
    compensated = fit_attention_relation_match(**common)
    raw = fit_attention_relation_match(**common, compensate_softmax_scale=False)
    expected = math.sqrt(compensated.child_head_dim / head_dim)
    assert compensated.softmax_scale_compensation == pytest.approx(expected)
    assert raw.softmax_scale_compensation == 1.0
    assert torch.allclose(
        compensated.query_weight, raw.query_weight * expected, atol=1e-10
    )
    assert torch.allclose(compensated.key_weight, raw.key_weight, atol=1e-10)


def test_gauge_and_balancing_leave_the_score_kernel_invariant():
    hidden, head_dim, heads, kv_heads = 32, 8, 4, 2
    common = {
        "query_weight": _normal(190, heads * head_dim, hidden),
        "key_weight": _normal(191, kv_heads * head_dim, hidden),
        "evidence_rows": _captured_split(192, 320, 0, hidden, hidden)[0],
        "head_dim": head_dim,
        "query_heads": heads,
        "key_value_heads": kv_heads,
        "level": "M4",
    }
    gauged = fit_attention_relation_match(**common)
    plain = fit_attention_relation_match(
        **common, procrustes_gauge=False, balance_factors=False
    )
    assert gauged.calibration["score_kernel_relative_error_max"] == pytest.approx(
        plain.calibration["score_kernel_relative_error_max"], rel=1e-9, abs=1e-12
    )
    # The gauge is not a no-op on the factors, only on the relation.
    assert not torch.allclose(gauged.query_weight, plain.query_weight, atol=1e-6)
    for record in gauged.solve["procrustes"]:
        assert record["orthogonality_deviation"] < 1e-8


def test_contraction_degrades_monotonically_and_is_reported_per_head():
    hidden, head_dim, heads = 40, 10, 3
    common = {
        "query_weight": _normal(200, heads * head_dim, hidden),
        "key_weight": _normal(201, heads * head_dim, hidden),
        "evidence_rows": _captured_split(202, 400, 0, hidden, hidden)[0],
        "head_dim": head_dim,
        "query_heads": heads,
        "key_value_heads": heads,
    }
    errors = []
    for level in ("M1", "M16", "M8", "M4"):
        fit = fit_attention_relation_match(**common, level=level)
        assert len(fit.per_head) == heads
        errors.append(fit.calibration["score_kernel_relative_error_max"])
    assert errors == sorted(errors)
    # The no-contraction gate is 1e-6; the residual here is set by the
    # conditioning of ``M^{-1/2}`` on this evidence, not by the fit.
    assert errors[0] < 1e-6


def test_closed_form_initializer_is_recorded_as_the_optimum_at_zero_ridge():
    hidden, head_dim, heads = 32, 8, 2
    fit = fit_attention_relation_match(
        query_weight=_normal(210, heads * head_dim, hidden),
        key_weight=_normal(211, heads * head_dim, hidden),
        evidence_rows=_captured_split(212, 300, 0, hidden, hidden)[0],
        head_dim=head_dim,
        query_heads=heads,
        key_value_heads=heads,
        level="M4",
    )
    assert fit.solve["iteration_cap"] == T7_DEFAULT_ITERATION_CAP
    assert fit.solve["alternating_improved_on_closed_form"] is False
    assert fit.solve["iterations"], "the alternating lane must actually run"


def test_attention_relation_refusals_close():
    hidden, head_dim, heads = 24, 6, 4
    query = _normal(220, heads * head_dim, hidden)
    key = _normal(221, heads * head_dim, hidden)
    rows = _captured_split(222, 200, 0, hidden, hidden)[0]
    base = {
        "query_weight": query,
        "key_weight": key,
        "evidence_rows": rows,
        "head_dim": head_dim,
        "query_heads": heads,
        "key_value_heads": heads,
        "level": "M4",
    }
    with pytest.raises(RelationMatchGeometryError, match="not divisible"):
        fit_attention_relation_match(**{**base, "key_value_heads": 3})
    with pytest.raises(RelationMatchGeometryError, match="query_weight must be"):
        fit_attention_relation_match(**{**base, "query_weight": _normal(223, 5, 5)})
    with pytest.raises(RelationMatchDisciplineError, match="fold-only"):
        fit_attention_relation_match(
            **{**base, "evidence_rows": rows.clone().requires_grad_(True)}
        )
    with pytest.raises(RelationMatchConditioningError, match="not identifiable"):
        fit_attention_relation_match(
            **{
                **base,
                "evidence_rows": _captured_split(
                    224, 200, 0, hidden, 3, noise=0.0
                )[0],
            }
        )


# ---------------------------------------------------------------------------
# T8 exact-identity, invariance, and refusals
# ---------------------------------------------------------------------------


def test_no_contraction_returns_the_parent_weight():
    weight = _normal(230, 40, 24)
    rows = _captured_split(231, 300, 0, 24, 24)[0]
    fit = fit_gram_relation_match(
        parent_output_rows=rows @ weight.transpose(0, 1),
        contracted_design_rows=rows,
        component="conv_out_proj",
        component_id="C05",
        site="L00",
    )
    delta = float((fit.child_weight - weight).abs().max().item())
    assert delta < 1e-6
    assert fit.calibration["gram_relative_error"] < 1e-6
    assert fit.calibration["direction_cosine_mean"] == pytest.approx(1.0, abs=1e-9)
    assert fit.solve["gram_truncation_active"] is False
    assert fit.solve["row_weighting"] == "inverse-parent-row-norm-squared"


def test_scale_free_weighting_ignores_a_rescaled_evidence_row():
    # The invariance claim is about the *solve*, so the keep set is pinned:
    # channel selection ranks raw column energy (the rule T5 shares) and is
    # deliberately not scale-free.
    weight = _normal(240, 12, 32)
    rows = _captured_split(241, 200, 0, 32, 14)[0]
    output = rows @ weight.transpose(0, 1)
    scaled_rows = rows.clone()
    scaled_output = output.clone()
    scaled_rows[0] = scaled_rows[0] * 50.0
    scaled_output[0] = scaled_output[0] * 50.0
    common = {
        "component": "mlp_down",
        "component_id": "C13",
        "kept_channel_ids": list(range(24)),
    }
    base = fit_gram_relation_match(
        parent_output_rows=output, parent_input_rows=rows, **common
    )
    rescaled = fit_gram_relation_match(
        parent_output_rows=scaled_output, parent_input_rows=scaled_rows, **common
    )
    unweighted_base = fit_gram_relation_match(
        parent_output_rows=output,
        parent_input_rows=rows,
        normalization="none",
        **common,
    )
    unweighted_rescaled = fit_gram_relation_match(
        parent_output_rows=scaled_output,
        parent_input_rows=scaled_rows,
        normalization="none",
        **common,
    )
    scale_free_shift = float(
        torch.linalg.matrix_norm(
            rescaled.child_weight - base.child_weight, ord="fro"
        ).item()
    )
    ridge_shift = float(
        torch.linalg.matrix_norm(
            unweighted_rescaled.child_weight - unweighted_base.child_weight,
            ord="fro",
        ).item()
    )
    assert scale_free_shift < 1e-8
    assert ridge_shift > 1e-3
    assert unweighted_base.solve["row_weighting"] == "uniform"
    assert unweighted_base.solve["scale_free_objective"] is False


def test_rank_truncation_activates_only_when_the_child_cannot_carry_the_rank():
    weight = _normal(250, 24, 32)
    rows = _captured_split(251, 300, 0, 32, 20)[0]
    output = rows @ weight.transpose(0, 1)
    wide = fit_gram_relation_match(
        parent_output_rows=output, parent_input_rows=rows, level="M16"
    )
    assert wide.solve["gram_truncation_active"] is False
    narrow = fit_gram_relation_match(
        parent_output_rows=output,
        parent_input_rows=rows,
        kept_width=8,
        level="M1",
    )
    assert narrow.gram_rank == 8
    assert narrow.solve["gram_truncation_active"] is True
    assert narrow.calibration["gram_relative_error"] > wide.calibration[
        "gram_relative_error"
    ]


def test_explicit_gram_rank_is_bounded_by_the_achievable_rank():
    weight = _normal(260, 16, 32)
    rows = _captured_split(261, 200, 0, 32, 12)[0]
    output = rows @ weight.transpose(0, 1)
    fit = fit_gram_relation_match(
        parent_output_rows=output,
        contracted_design_rows=rows,
        gram_rank=6,
    )
    assert fit.gram_rank == 6
    with pytest.raises(RelationMatchGeometryError, match="exceeds the achievable"):
        fit_gram_relation_match(
            parent_output_rows=output,
            contracted_design_rows=rows,
            gram_rank=999,
        )


def test_gram_relation_reports_the_direction_magnitude_split():
    weight = _normal(270, 20, 48)
    rows, held = _captured_split(271, 400, 96, 48, 22)
    output = rows @ weight.transpose(0, 1)
    fit = fit_gram_relation_match(
        parent_output_rows=output,
        parent_input_rows=rows,
        level="M4",
        component="conv_out_proj",
        component_id="C05",
        heldout_design_rows=held,
        heldout_output_rows=held @ weight.transpose(0, 1),
    )
    assert 0.0 <= fit.calibration["direction_cosine_mean"] <= 1.0
    assert fit.calibration["magnitude_ratio_mean"] > 0.0
    assert fit.calibration["heldout"]["rows"] == 96
    assert fit.calibration["heldout"]["direction_cosine_mean"] > 0.5


def test_gram_relation_refusals_close():
    weight = _normal(280, 16, 32)
    rows = _captured_split(281, 200, 0, 32, 16)[0]
    output = rows @ weight.transpose(0, 1)
    with pytest.raises(RelationMatchGeometryError, match="not both"):
        fit_gram_relation_match(
            parent_output_rows=output,
            contracted_design_rows=rows,
            parent_input_rows=rows,
        )
    with pytest.raises(RelationMatchGeometryError, match="needs a level"):
        fit_gram_relation_match(
            parent_output_rows=output, parent_input_rows=rows
        )
    with pytest.raises(RelationMatchEvidenceError, match="row count"):
        fit_gram_relation_match(
            parent_output_rows=output[:100], contracted_design_rows=rows
        )
    with pytest.raises(RelationMatchDisciplineError, match="fold-only"):
        fit_gram_relation_match(
            parent_output_rows=output.clone().requires_grad_(True),
            contracted_design_rows=rows,
        )
    with pytest.raises(RelationMatchGeometryError, match="row_unit"):
        fit_gram_relation_match(
            parent_output_rows=output,
            contracted_design_rows=rows,
            normalization="whiten",
        )
    with pytest.raises(RelationMatchEvidenceError, match="zero norm"):
        fit_gram_relation_match(
            parent_output_rows=torch.zeros_like(output),
            contracted_design_rows=rows,
        )


def test_apply_accepts_contracted_or_full_width_rows():
    weight = _normal(290, 12, 32)
    rows, held = _captured_split(291, 200, 40, 32, 14)
    fit = fit_gram_relation_match(
        parent_output_rows=rows @ weight.transpose(0, 1),
        parent_input_rows=rows,
        level="M4",
    )
    full = fit.apply(held)
    narrow = fit.apply(held.index_select(1, fit.kept_channel_ids))
    assert torch.allclose(full, narrow, atol=1e-12)
    with pytest.raises(RelationMatchGeometryError, match="contracted or full"):
        fit.apply(held[:, :5])


# ---------------------------------------------------------------------------
# Shared keep rule and the registry contract
# ---------------------------------------------------------------------------


def test_keep_rule_agrees_with_the_body_route_module():
    rows = _captured_split(300, 300, 0, 64, 30)[0]
    for width in (16, 32, 60):
        assert torch.equal(
            select_kept_channels(rows, kept_width=width),
            body_route_select_kept_channels(
                rows, kept_width=width, policy="evidence-energy"
            ),
        )


def test_records_seal_and_detect_tampering():
    weight = _normal(310, 16, 32)
    rows = _captured_split(311, 200, 0, 32, 16)[0]
    fit = fit_gram_relation_match(
        parent_output_rows=rows @ weight.transpose(0, 1),
        parent_input_rows=rows,
        level="M8",
        component="mlp_down",
        component_id="C13",
    )
    verified = verify_sealed_relation_record(
        fit.record, schema=T8_RELATION_RECORD_SCHEMA
    )
    assert verified["record_sha256"] == fit.record_sha256
    assert verified["operation_audit"]["backward_calls"] == 0
    tampered = dict(fit.record)
    tampered["kept_width"] = int(tampered["kept_width"]) + 1
    with pytest.raises(Exception, match="seal is invalid"):
        verify_sealed_relation_record(tampered, schema=T8_RELATION_RECORD_SCHEMA)


def test_registry_entries_match_the_dispatch_contract():
    seven, eight = matrix_technique_registry_entries()
    assert seven.schema == MATRIX_TECHNIQUE_ENTRY_SCHEMA
    assert (seven.technique_id, eight.technique_id) == ("T7", "T8")
    assert seven.describe()["record_schema"] == T7_RELATION_RECORD_SCHEMA
    assert eight.describe()["record_schema"] == T8_RELATION_RECORD_SCHEMA
    assert "C05" in eight.components and "C13" in eight.components
    assert eight.describe()["covers_short_conv_outputs"] is True
    assert matrix_technique_registry_entry("T8").technique_id == "T8"
    with pytest.raises(RelationMatchError, match="not a relation-matching"):
        matrix_technique_registry_entry("T5")


def test_registry_fit_round_trips_both_techniques():
    hidden, head_dim, heads = 24, 6, 2
    seven, eight = matrix_technique_registry_entries()
    attention = seven.fit(
        {
            "query_weight": _normal(320, heads * head_dim, hidden),
            "key_weight": _normal(321, heads * head_dim, hidden),
            "evidence_rows": _captured_split(322, 200, 0, hidden, hidden)[0],
            "head_dim": head_dim,
            "query_heads": heads,
            "key_value_heads": heads,
            "level": "M4",
        }
    )
    assert attention["technique_id"] == "T7"
    assert set(attention["child_parameters"]) == {"query_weight", "key_weight"}
    assert callable(attention["apply"])

    weight = _normal(323, 12, 32)
    rows = _captured_split(324, 200, 0, 32, 14)[0]
    gram = eight.fit(
        {
            "parent_output_rows": rows @ weight.transpose(0, 1),
            "parent_input_rows": rows,
            "level": "M8",
            "component": "mlp_down",
            "component_id": "C13",
        }
    )
    assert gram["technique_id"] == "T8"
    assert set(gram["child_parameters"]) == {"weight"}
    with pytest.raises(RelationMatchError, match="Unsupported T8 request keys"):
        eight.fit({"parent_output_rows": rows, "epochs": 3})
    with pytest.raises(RelationMatchError, match="Missing T7 request keys"):
        seven.fit({"head_dim": head_dim})
