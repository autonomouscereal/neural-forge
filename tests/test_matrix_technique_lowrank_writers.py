"""Stage-1 proofs for T4 factored (low-rank) writers.

Proven here, on synthetic sealed-evidence-shaped tensors with no optimizer, no
gradient and no model execution:

* the exact-identity gate — at full rank the factor pair round-trips to the
  parent writer under both weightings;
* the evidence-weighted lane really is a different (and better) fit than the
  plain weight SVD on correlated evidence, which is the whole reason the fold
  lane exists;
* the factor tensors drop into the LFM2 blank materializer's factorized
  ``out_proj`` state keys unchanged, and this test fails loudly if that
  module's key layout ever moves;
* the parameter-budget and conditioning refusals close.
"""

from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import TECHNIQUES  # noqa: E402
from neural_forge.matrix_technique_lowrank_writers import (  # noqa: E402
    MATRIX_TECHNIQUE_ENTRY_SCHEMA,
    SHORT_CONV_EVIDENCE_DECODER_SUFFIX,
    SHORT_CONV_EVIDENCE_ENCODER_SUFFIX,
    SHORT_CONV_FACTOR_DECODER_SUFFIX,
    SHORT_CONV_FACTOR_ENCODER_SUFFIX,
    T4_EVIDENCE_ALGORITHM,
    T4_LOWRANK_WRITER_RECORD_SCHEMA,
    T4_WEIGHT_ALGORITHM,
    LowRankWriterBudgetError,
    LowRankWriterConditioningError,
    LowRankWriterDisciplineError,
    LowRankWriterError,
    LowRankWriterEvidenceError,
    LowRankWriterGeometryError,
    factor_writer,
    lfm2_short_conv_writer_state_entries,
    matrix_technique_registry_entry,
    maximum_shrinking_rank,
    resolve_writer_rank,
    verify_sealed_lowrank_writer_record,
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


# ---------------------------------------------------------------------------
# Rank algebra
# ---------------------------------------------------------------------------


def test_levels_resolve_against_the_writers_full_rank():
    assert resolve_writer_rank(level="M16", full_rank=1024) == 960
    assert resolve_writer_rank(level="M8", full_rank=1024) == 896
    assert resolve_writer_rank(level="M4", full_rank=1024) == 768
    assert resolve_writer_rank(level="M1", full_rank=1024, rank=31) == 31
    with pytest.raises(LowRankWriterGeometryError, match="witness is rank 31"):
        resolve_writer_rank(level="X4", full_rank=1024)
    with pytest.raises(LowRankWriterGeometryError, match="outside 1"):
        resolve_writer_rank(level="M1", full_rank=64, rank=65)


# ---------------------------------------------------------------------------
# Exact-identity gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["none", "evidence"])
def test_full_rank_round_trips_the_parent_writer(weighting):
    weight = _normal(401, 48, 64)
    rows = _captured_split(402, 400, 0, 64, 64)[0]
    factorization = factor_writer(
        parent_weight=weight,
        rank=48,
        weighting=weighting,
        evidence_rows=rows if weighting == "evidence" else None,
        allow_parameter_increase=True,
        component="conv_out_proj",
        site="L13",
    )
    assert factorization.encoder.shape == (48, 64)
    assert factorization.decoder.shape == (64 - 16, 48)
    delta = float((factorization.dense_equivalent - weight).abs().max().item())
    assert delta < 1e-6
    assert factorization.calibration["weight_relative_l2"] < 1e-6
    assert factorization.calibration["at_full_rank"] is True
    probe = _normal(403, 20, 64)
    assert torch.allclose(
        factorization.apply(probe), probe @ weight.transpose(0, 1), atol=1e-8
    )


def test_algorithms_are_named_per_weighting():
    weight = _normal(404, 24, 48)
    rows = _captured_split(405, 300, 0, 48, 30)[0]
    plain = factor_writer(parent_weight=weight, rank=8, weighting="none")
    weighted = factor_writer(
        parent_weight=weight, rank=8, weighting="evidence", evidence_rows=rows
    )
    assert plain.solve["algorithm"] == T4_WEIGHT_ALGORITHM
    assert weighted.solve["algorithm"] == T4_EVIDENCE_ALGORITHM
    assert plain.solve["evidence_rank"] == 48
    assert weighted.solve["evidence_rank"] == 48


# ---------------------------------------------------------------------------
# The evidence-weighted lane is the fold lane
# ---------------------------------------------------------------------------


def test_evidence_weighting_beats_the_plain_svd_on_correlated_evidence():
    weight = _normal(410, 32, 64)
    rows, held = _captured_split(411, 600, 120, 64, 12)
    common = {
        "parent_weight": weight,
        "rank": 8,
        "evidence_rows": rows,
        "heldout_rows": held,
        "component": "conv_out_proj",
    }
    plain = factor_writer(**common, weighting="none")
    weighted = factor_writer(**common, weighting="evidence")
    assert (
        weighted.calibration["evidence"]["write_relative_l2"]
        < plain.calibration["evidence"]["write_relative_l2"]
    )
    assert (
        weighted.calibration["heldout"]["write_relative_l2"]
        < plain.calibration["heldout"]["write_relative_l2"]
    )
    # The trade the evidence lane makes is visible, not silent: it spends no
    # rank on weight directions the evidence never excites.
    assert weighted.solve["weight_energy_outside_evidence_range"] >= 0.0
    assert plain.calibration["weight_relative_l2"] < weighted.calibration[
        "weight_relative_l2"
    ]


def test_higher_rank_never_writes_worse():
    weight = _normal(420, 32, 64)
    rows = _captured_split(421, 500, 0, 64, 20)[0]
    errors = []
    for rank in (4, 8, 16, 20):
        factorization = factor_writer(
            parent_weight=weight,
            rank=rank,
            weighting="evidence",
            evidence_rows=rows,
        )
        errors.append(factorization.calibration["evidence"]["write_relative_l2"])
    assert errors == sorted(errors, reverse=True)


def test_spectral_tail_reports_the_feasibility_witness_signal():
    weight = _normal(430, 32, 64)
    rows = _captured_split(431, 500, 0, 64, 10, noise=0.0)[0]
    factorization = factor_writer(
        parent_weight=weight, rank=10, weighting="evidence", evidence_rows=rows
    )
    assert 0.0 <= factorization.calibration["spectral_tail_fraction"] < 1e-6
    assert int(factorization.singular_values.numel()) == 32


# ---------------------------------------------------------------------------
# Wiring into the blank materializer's factorized writer
# ---------------------------------------------------------------------------


def test_state_entries_match_the_materializer_sequential_layout():
    weight = _normal(440, 128, 128)
    rows = _captured_split(441, 400, 0, 128, 60)[0]
    factorization = factor_writer(
        parent_weight=weight,
        rank=31,
        weighting="evidence",
        evidence_rows=rows,
        component="conv_out_proj",
        site="L13",
    )
    entries = lfm2_short_conv_writer_state_entries(
        factorization,
        layer_index=13,
        internal_channel_width=128,
        hidden_size=128,
    )
    encoder_key = f"model.layers.13.{SHORT_CONV_FACTOR_ENCODER_SUFFIX}"
    decoder_key = f"model.layers.13.{SHORT_CONV_FACTOR_DECODER_SUFFIX}"
    assert sorted(entries) == sorted((encoder_key, decoder_key))
    # torch Linear stores (out_features, in_features): Linear(128, 31) then
    # Linear(31, 128).
    assert tuple(entries[encoder_key].shape) == (31, 128)
    assert tuple(entries[decoder_key].shape) == (128, 31)
    generic = factorization.state_entries("model.layers.13.conv.out_proj")
    assert set(generic) == set(entries)
    with pytest.raises(LowRankWriterGeometryError, match="internal_channel_width"):
        lfm2_short_conv_writer_state_entries(
            factorization, layer_index=13, internal_channel_width=64
        )


def test_state_key_layout_still_matches_the_owning_materializer():
    materializer = pytest.importorskip(
        "neural_forge.hf_lfm2_dynamic_blank_materializer"
    )
    source = inspect.getsource(
        materializer.lfm2_factorized_short_conv_evidence_state_key_mapping
    )
    for suffix in (
        SHORT_CONV_FACTOR_ENCODER_SUFFIX,
        SHORT_CONV_FACTOR_DECODER_SUFFIX,
        SHORT_CONV_EVIDENCE_ENCODER_SUFFIX,
        SHORT_CONV_EVIDENCE_DECODER_SUFFIX,
    ):
        stem = suffix.removeprefix("conv.")
        assert stem in source, (
            f"{suffix!r} no longer appears in the materializer's factorized "
            "short-conv key bridge; T4 would emit uninstallable state."
        )


def test_parameter_accounting_matches_the_frontier_formula():
    weight = _normal(450, 128, 96)
    factorization = factor_writer(parent_weight=weight, rank=20, weighting="none")
    record = factorization.record
    assert record["parent_parameters"] == 128 * 96
    assert record["child_parameters"] == 20 * (128 + 96)
    assert record["writer_topology"] == "factorized"
    assert record["writer_factor_count"] == 2
    assert record["materializer_backed"] is False
    assert record["parameter_ratio"] < 1.0


# ---------------------------------------------------------------------------
# The refusal ladder
# ---------------------------------------------------------------------------


def test_marginal_levels_cannot_shrink_a_square_writer():
    # The campaign's marginal levels keep 15/16, 7/8 and 3/4 of the full rank;
    # a square writer only shrinks below half.  This is a real limit on the
    # wave-1 T4 cells at C05/C09, not a solver defect, so it refuses loudly.
    assert maximum_shrinking_rank(1024, 1024) == 511
    assert maximum_shrinking_rank(2560, 1024) == 731
    for level in ("M16", "M8", "M4"):
        with pytest.raises(LowRankWriterBudgetError, match="only shrinks at rank"):
            factor_writer(
                parent_weight=_normal(455, 128, 128),
                level=level,
                weighting="none",
                component="conv_out_proj",
            )
    witness = factor_writer(
        parent_weight=_normal(456, 128, 128),
        rank=31,
        weighting="none",
        component="conv_out_proj",
        site="L13",
    )
    assert witness.record["shrinks_the_writer"] is True
    assert witness.record["maximum_shrinking_rank"] == 63


def test_parameter_increase_is_refused_by_default():
    with pytest.raises(LowRankWriterBudgetError, match="does not shrink"):
        factor_writer(parent_weight=_normal(460, 16, 16), rank=16, weighting="none")
    allowed = factor_writer(
        parent_weight=_normal(461, 16, 16),
        rank=16,
        weighting="none",
        allow_parameter_increase=True,
    )
    assert allowed.record["parameter_ratio"] > 1.0


def test_evidence_weighting_requires_evidence():
    with pytest.raises(LowRankWriterEvidenceError, match="requires evidence_rows"):
        factor_writer(parent_weight=_normal(470, 24, 48), rank=8)


def test_rank_beyond_the_evidence_rank_is_refused():
    rows = _captured_split(480, 200, 0, 48, 6, noise=0.0)[0]
    with pytest.raises(LowRankWriterConditioningError, match="not identifiable"):
        factor_writer(
            parent_weight=_normal(481, 24, 48),
            rank=12,
            weighting="evidence",
            evidence_rows=rows,
        )


def test_autograd_and_shape_refusals_close():
    weight = _normal(490, 24, 48)
    rows = _captured_split(491, 200, 0, 48, 24)[0]
    with pytest.raises(LowRankWriterDisciplineError, match="fold-only"):
        factor_writer(
            parent_weight=weight.clone().requires_grad_(True),
            rank=8,
            weighting="none",
        )
    with pytest.raises(LowRankWriterEvidenceError, match="input features"):
        factor_writer(
            parent_weight=weight,
            rank=8,
            weighting="evidence",
            evidence_rows=rows[:, :20],
        )
    with pytest.raises(LowRankWriterGeometryError, match="weighting must be"):
        factor_writer(parent_weight=weight, rank=8, weighting="whitened")
    with pytest.raises(LowRankWriterGeometryError, match="not a campaign writer"):
        factor_writer(parent_weight=weight, rank=8, weighting="none", component="C11")


# ---------------------------------------------------------------------------
# Sealing and the registry contract
# ---------------------------------------------------------------------------


def test_record_seals_and_detects_tampering():
    factorization = factor_writer(
        parent_weight=_normal(500, 64, 96),
        rank=20,
        weighting="evidence",
        evidence_rows=_captured_split(501, 300, 0, 96, 40)[0],
        component="C05",
        site="L13",
    )
    verified = verify_sealed_lowrank_writer_record(factorization.record)
    assert verified["schema"] == T4_LOWRANK_WRITER_RECORD_SCHEMA
    assert verified["record_sha256"] == factorization.record_sha256
    assert verified["materializer_backed"] is True
    assert verified["operation_audit"]["backward_calls"] == 0
    tampered = dict(factorization.record)
    tampered["writer_rank"] = 19
    with pytest.raises(Exception, match="seal is invalid"):
        verify_sealed_lowrank_writer_record(tampered)


def test_registry_entry_matches_the_dispatch_contract():
    entry = matrix_technique_registry_entry()
    assert entry.schema == MATRIX_TECHNIQUE_ENTRY_SCHEMA
    assert entry.technique_id == "T4"
    assert entry.technique == TECHNIQUES["T4"] == "low_rank_factor_writers"
    assert entry.components == ("C05", "C09", "C13")
    assert entry.describe()["materializer_backed_components"] == ("C05",)
    result = entry.fit(
        {
            "parent_weight": _normal(510, 64, 96),
            "rank": 20,
            "weighting": "none",
            "component": "C05",
        }
    )
    assert result["technique_id"] == "T4"
    assert set(result["child_parameters"]) == {"encoder", "decoder"}
    assert result["route"]["writer_rank"] == 20
    assert result["route"]["writer_factor_count"] == 2
    assert callable(result["apply"])
    with pytest.raises(LowRankWriterError, match="Unsupported T4 request keys"):
        entry.fit({"parent_weight": _normal(511, 8, 16), "epochs": 2})
    with pytest.raises(LowRankWriterError, match="Missing T4 request keys"):
        entry.fit({"level": "M4"})
