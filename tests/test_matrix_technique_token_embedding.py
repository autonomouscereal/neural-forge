"""Contract tests for C01, the tied-embedding fold wave 9 adds to the matrix.

What is pinned here, and why each one exists:

1. **The identity gate.**  At full rank the factorization reconstructs the table
   exactly.  Wave 3's kernels earned this test the hard way: a contraction that
   cannot reproduce its own parent at the un-contracted rank is not a
   contraction, it is a bug wearing one, and every measured error below full
   rank would be uninterpretable.
2. **The storage algebra is the registry's, not a second copy.**  C01 registers
   under T4 precisely so ``storage_model("T4","C01")`` already returns
   ``r*(vocab+hidden)``; if the kernel and the planner ever disagreed, the
   planner would resolve a width whose realized count the runner then refuses,
   after the solve.
3. **Uniform weighting is the plain truncated SVD**, checked against
   ``torch.linalg.svd`` directly rather than against itself.
4. **The weighted arm is actually a weighted least squares**: it beats the
   uniform arm on the weighted objective it optimizes, and it collapses back to
   the uniform arm when every weight is equal.  Without the second half, a
   weighting that silently did nothing would look like a win.
5. **The harness contract.**  ``validate_result`` accepts what the kernel
   returns: parent-shaped, one owned key, finite, with every mandatory solve
   field present and sealable.
6. **The refusals.**  A rank past full rank, a rank that does not contract, a
   frequency file whose digest has moved, and a frequency file whose length is
   not the vocabulary all fail closed.
7. **The T5 route.**  Its head rows are stored *exactly* -- checked with
   ``torch.equal``, not ``allclose`` -- its storage counts the token ids it
   routes by, and the width the registry resolves for it is the tail rank and
   not the head.
"""

from __future__ import annotations

import hashlib
import json
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
    component_spec,
    resolve,
    storage_model,
    storage_width_for_ratio,
    validate_result,
)
from neural_forge.matrix_technique_token_embedding import (  # noqa: E402
    TOKEN_FREQUENCY_SCHEMA,
    TokenEmbeddingFoldError,
    c01_sparse_route_sparsity_for_ratio,
    c01_sparse_route_storage,
    c01_token_route_storage,
    factor_token_embedding,
    load_token_frequencies,
    maximum_shrinking_rank,
    route_token_embedding,
    sparse_route_token_embedding,
    t4_token_embedding_technique,
    t5_token_embedding_route_technique,
)

VOCAB = 96
HIDDEN = 16

#: The parent's real geometry, used wherever a test is about the actual campaign
#: numbers rather than about the algebra.
PARENT_VOCAB = 65536
PARENT_HIDDEN = 1024
PARENT_EMBEDDING_PARAMETERS = PARENT_VOCAB * PARENT_HIDDEN  # 67,108,864


class _NoEvidence:
    """C01 reads no rows; anything that asks for one is a bug, loudly."""

    @property
    def endpoints(self):
        return ()

    @property
    def selection_sha256(self):
        return "0" * 64

    @property
    def fit_row_count(self):
        return 0

    @property
    def heldout_row_count(self):
        return 0

    def fit_rows(self, endpoint):  # pragma: no cover - the point is it is unused
        raise AssertionError(f"C01 asked for evidence rows at {endpoint!r}.")

    def heldout_rows(self, endpoint):  # pragma: no cover - same
        raise AssertionError(f"C01 asked for held-out rows at {endpoint!r}.")


def _table(seed: int = 7, *, vocab: int = VOCAB, hidden: int = HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((vocab, hidden), dtype=torch.float64, generator=generator)


def _state(table=None):
    table = _table() if table is None else table
    return {"model.embed_tokens.weight": table}


def _request(state, *, level="S4", level_width, knobs=None):
    cell = CellId.parse(f"C01.T4.{level}")
    spec = component_spec(cell, state)
    return CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=level_width,
        seed=0,
        parent_weights=state,
        evidence=_NoEvidence(),
        knobs=dict(knobs or {}),
        workspace=Path("."),
        log=lambda message: None,
    )


def _frequency_file(directory: Path, counts, **extra) -> tuple[Path, str]:
    body = {
        "schema": TOKEN_FREQUENCY_SCHEMA,
        "counts": [int(value) for value in counts],
        "corpus_sha256": "a" * 64,
        "corpus_record_count": 256,
        **extra,
    }
    path = directory / "token-frequency.json"
    raw = json.dumps(body, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------- the identity gate


def test_full_rank_reconstructs_the_table_exactly():
    table = _table()
    folded = factor_token_embedding(embedding=table, rank=HIDDEN)
    reconstruction = folded.dense_equivalent
    assert reconstruction.shape == table.shape
    assert torch.allclose(reconstruction, table, atol=1e-10, rtol=0.0)
    assert folded.solve["reconstruction_relative_frobenius"] < 1e-12
    assert folded.solve["retained_energy_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert folded.solve["row_cosine_min"] > 1.0 - 1e-9


def test_full_rank_reconstructs_exactly_under_the_weighted_arm_too():
    table = _table()
    generator = torch.Generator().manual_seed(11)
    counts = torch.randint(0, 50, (VOCAB,), generator=generator).to(torch.float64)
    counts[0] = 1.0  # never all-zero
    folded = factor_token_embedding(
        embedding=table,
        rank=HIDDEN,
        weights=counts,
        weighting="token-frequency",
    )
    assert torch.allclose(folded.dense_equivalent, table, atol=1e-10, rtol=0.0)


# ------------------------------------------------------- the truncated-SVD claim


def test_uniform_weighting_is_the_plain_truncated_svd():
    table = _table()
    rank = 5
    folded = factor_token_embedding(embedding=table, rank=rank)
    left, values, right = torch.linalg.svd(table, full_matrices=False)
    reference = (
        left[:, :rank] * values[:rank].unsqueeze(0)
    ) @ right[:rank, :]
    assert torch.allclose(folded.dense_equivalent, reference, atol=1e-9, rtol=0.0)
    # Eckart-Young: the residual is the tail of the spectrum, exactly.
    tail = float(values[rank:].square().sum().sqrt().item())
    assert folded.solve["correction_norm"] == pytest.approx(tail, rel=1e-9)


def test_the_factors_are_what_is_stored_and_they_rebuild_the_table():
    table = _table()
    rank = 6
    folded = factor_token_embedding(embedding=table, rank=rank)
    assert tuple(folded.row_factor.shape) == (VOCAB, rank)
    assert tuple(folded.basis.shape) == (HIDDEN, rank)
    assert folded.stored_parameter_count == rank * (VOCAB + HIDDEN)
    rebuilt = folded.row_factor @ folded.basis.T
    assert torch.allclose(rebuilt, folded.dense_equivalent, atol=0.0, rtol=0.0)
    # The basis is orthonormal, so the tied readout's projection is an honest
    # projection and not a rescaling hiding in the factorization.
    identity = folded.basis.T @ folded.basis
    assert torch.allclose(
        identity, torch.eye(rank, dtype=torch.float64), atol=1e-10, rtol=0.0
    )


def test_the_basis_sign_is_pinned_so_two_runs_are_the_same_run():
    table = _table()
    first = factor_token_embedding(embedding=table, rank=4)
    second = factor_token_embedding(embedding=table.clone(), rank=4)
    assert torch.equal(first.basis, second.basis)
    assert first.record_sha256 == second.record_sha256


# --------------------------------------------------------- the weighted-fit claim


def _weighted_error(table, approximation, weights) -> float:
    residual = (approximation - table).square().sum(dim=1)
    return float((weights * residual).sum().item())


def test_the_weighted_arm_beats_the_uniform_arm_on_the_weighted_objective():
    table = _table(seed=3)
    generator = torch.Generator().manual_seed(5)
    # A frequency profile with real mass concentration: a few hundred-count
    # tokens, a long tail of zeros.  That is the shape a real corpus has and it
    # is the case where the two arms should differ.
    counts = torch.zeros(VOCAB, dtype=torch.float64)
    counts[:12] = torch.randint(50, 500, (12,), generator=generator).to(torch.float64)
    rank = 5
    uniform = factor_token_embedding(embedding=table, rank=rank)
    weighted = factor_token_embedding(
        embedding=table, rank=rank, weights=counts, weighting="token-frequency"
    )
    weighted_error = _weighted_error(table, weighted.dense_equivalent, counts)
    uniform_error = _weighted_error(table, uniform.dense_equivalent, counts)
    assert weighted_error < uniform_error
    # ...and it pays for that on the objective it is not optimizing, which is
    # the trade the A/B exists to measure rather than assume away.
    assert (
        weighted.solve["reconstruction_relative_frobenius"]
        > uniform.solve["reconstruction_relative_frobenius"]
    )


def test_equal_weights_reproduce_the_uniform_arm():
    table = _table(seed=9)
    counts = torch.full((VOCAB,), 17.0, dtype=torch.float64)
    uniform = factor_token_embedding(embedding=table, rank=7)
    weighted = factor_token_embedding(
        embedding=table, rank=7, weights=counts, weighting="token-frequency"
    )
    assert torch.allclose(
        weighted.dense_equivalent, uniform.dense_equivalent, atol=1e-9, rtol=0.0
    )


# ------------------------------------------------------------- the storage algebra


def test_the_storage_model_is_the_registrys_and_the_kernel_agrees():
    state = _state()
    spec = component_spec(CellId.parse("C01.T4.S4"), state)
    model = storage_model("T4", "C01")
    for rank in (1, 3, 9):
        folded = factor_token_embedding(embedding=state["model.embed_tokens.weight"], rank=rank)
        assert model(spec, rank) == folded.stored_parameter_count


def test_the_parent_geometry_resolves_a_true_quarter_and_a_true_eighth():
    """The two widths wave 9 actually runs, derived not asserted."""

    state = {
        "model.embed_tokens.weight": torch.zeros(
            (PARENT_VOCAB, PARENT_HIDDEN), dtype=torch.float64
        )
    }
    spec = component_spec(CellId.parse("C01.T4.S4"), state)
    assert spec.parent_parameter_count == PARENT_EMBEDDING_PARAMETERS
    quarter = storage_width_for_ratio(spec, "T4", 0.25)
    eighth = storage_width_for_ratio(spec, "T4", 0.125)
    assert quarter == 252
    assert eighth == 126
    stored_quarter = quarter * (PARENT_VOCAB + PARENT_HIDDEN)
    stored_eighth = eighth * (PARENT_VOCAB + PARENT_HIDDEN)
    assert stored_quarter <= 0.25 * PARENT_EMBEDDING_PARAMETERS
    assert stored_eighth <= 0.125 * PARENT_EMBEDDING_PARAMETERS
    # One rank deeper would break each bound, which is what makes these the
    # widths and not merely widths.
    assert (quarter + 1) * (PARENT_VOCAB + PARENT_HIDDEN) > 0.25 * PARENT_EMBEDDING_PARAMETERS
    assert (eighth + 1) * (PARENT_VOCAB + PARENT_HIDDEN) > 0.125 * PARENT_EMBEDDING_PARAMETERS


def test_the_break_even_rank_is_the_lopsided_one_the_table_deserves():
    assert maximum_shrinking_rank(PARENT_VOCAB, PARENT_HIDDEN) == 1008
    # A square writer's ceiling is just under half its full rank; the embedding
    # table's is 98% of it.  That asymmetry is why a deep rank is affordable
    # here and not at C05/C09/C13.
    assert maximum_shrinking_rank(1024, 1024) == 511


# ------------------------------------------------------------- the harness contract


def test_the_kernel_is_registered_at_c01_and_wins_over_the_generic_t4():
    from neural_forge import matrix_technique_lowrank_writers  # noqa: F401

    assert resolve("T4", "C01") is t4_token_embedding_technique
    assert resolve("T4", "C05") is not t4_token_embedding_technique


def test_validate_result_accepts_what_the_kernel_returns():
    state = _state()
    request = _request(state, level_width=5)
    result = t4_token_embedding_technique(request)
    validate_result(request, result)
    assert set(result.installed_weights) == {"model.embed_tokens.weight"}
    installed = result.installed_weights["model.embed_tokens.weight"]
    assert tuple(installed.shape) == (VOCAB, HIDDEN)
    assert result.stored_parameter_count == 5 * (VOCAB + HIDDEN)
    assert result.solve["clamp_engaged"] is False
    assert result.solve["allowed_clamp_ratio"] is None
    assert result.solve["token_frequency"] is None
    assert result.predict is None


def test_the_weighted_arm_seals_the_digest_of_the_file_it_read(tmp_path):
    state = _state()
    counts = [3] * VOCAB
    path, digest = _frequency_file(tmp_path, counts)
    request = _request(
        state,
        level_width=5,
        knobs={
            "embedding_weighting": "token-frequency",
            "token_frequency_path": str(path),
            "token_frequency_sha256": digest,
        },
    )
    result = t4_token_embedding_technique(request)
    validate_result(request, result)
    provenance = result.solve["token_frequency"]
    assert provenance["token_frequency_sha256"] == digest
    assert provenance["observed_token_types"] == VOCAB
    assert provenance["corpus_record_count"] == 256
    assert result.solve["weighting"] == "token-frequency"


def test_a_moved_frequency_file_is_refused(tmp_path):
    state = _state()
    path, digest = _frequency_file(tmp_path, [1] * VOCAB)
    request = _request(
        state,
        level_width=5,
        knobs={
            "embedding_weighting": "token-frequency",
            "token_frequency_path": str(path),
            "token_frequency_sha256": "f" * 64,
        },
    )
    with pytest.raises(CellTechniqueError, match="not the expected"):
        t4_token_embedding_technique(request)
    assert digest != "f" * 64


def test_a_frequency_table_of_the_wrong_length_is_refused(tmp_path):
    path, digest = _frequency_file(tmp_path, [1] * (VOCAB - 1))
    with pytest.raises(TokenEmbeddingFoldError, match="counts"):
        load_token_frequencies(path, vocab_size=VOCAB, expected_sha256=digest)


def test_an_all_zero_frequency_table_is_refused(tmp_path):
    path, digest = _frequency_file(tmp_path, [0] * VOCAB)
    with pytest.raises(TokenEmbeddingFoldError, match="carries no evidence"):
        load_token_frequencies(path, vocab_size=VOCAB, expected_sha256=digest)


# ------------------------------------------------------------------- the refusals


def test_a_rank_past_full_rank_is_refused():
    with pytest.raises(TokenEmbeddingFoldError, match="exceeds the table"):
        factor_token_embedding(embedding=_table(), rank=HIDDEN + 1)


def test_a_rank_that_does_not_contract_is_refused_by_the_kernel():
    # VOCAB=96, HIDDEN=16 -> break-even rank 13, so 15 stores more than the
    # table it replaced and the cell refuses rather than sealing a child that
    # grew.
    state = _state()
    request = _request(state, level_width=15)
    with pytest.raises(CellTechniqueError, match="highest rank that contracts"):
        t4_token_embedding_technique(request)


def test_the_deliberate_override_admits_a_non_contracting_rank():
    state = _state()
    request = _request(
        state, level_width=15, knobs={"c01_allow_parameter_increase": True}
    )
    result = t4_token_embedding_technique(request)
    assert result.solve["shrinks_the_table"] is False


def test_a_weight_vector_on_the_uniform_arm_is_refused():
    with pytest.raises(TokenEmbeddingFoldError, match="takes no weight vector"):
        factor_token_embedding(
            embedding=_table(), rank=4, weights=torch.ones(VOCAB, dtype=torch.float64)
        )


def test_a_negative_weight_is_refused():
    counts = torch.ones(VOCAB, dtype=torch.float64)
    counts[3] = -1.0
    with pytest.raises(TokenEmbeddingFoldError, match="nonnegative"):
        factor_token_embedding(
            embedding=_table(),
            rank=4,
            weights=counts,
            weighting="token-frequency",
        )


def test_an_autograd_tensor_is_refused():
    table = _table().requires_grad_(True)
    with pytest.raises(TokenEmbeddingFoldError, match="autograd"):
        factor_token_embedding(embedding=table, rank=4)


# ------------------------------------------------------- T5 at C01: the route


def test_the_route_stores_the_head_rows_exactly():
    table = _table(seed=21)
    counts = torch.zeros(VOCAB, dtype=torch.float64)
    counts[:8] = 100.0
    route = route_token_embedding(
        embedding=table, head_size=8, tail_rank=4, weights=counts
    )
    assert route.head_ids.tolist() == list(range(8))
    rebuilt = route.dense_equivalent
    # Exact, not approximately exact: the head is a copy.
    assert torch.equal(rebuilt[:8], table[:8])
    assert not torch.equal(rebuilt[8:], table[8:])
    assert route.solve["head_row_cosine_mean"] == pytest.approx(1.0, abs=1e-12)
    assert route.solve["route"]["observed_token_types_in_head"] == 8
    assert route.solve["route"]["observed_token_types_outside_head"] == 0
    assert route.solve["route"]["corpus_mass_in_head"] == pytest.approx(1.0)


def test_the_route_storage_counts_the_ids_it_routes_by():
    """A route that stores rows and not their ids is a table nobody can index."""

    stored = c01_token_route_storage(
        vocab_size=65536, hidden_size=1024, head_size=4096, tail_rank=201
    )
    values = 4096 * 1024
    ids = 4096
    tail = 201 * ((65536 - 4096) + 1024)
    assert stored == values + ids + tail
    assert stored == 16_753_664
    assert stored <= 0.25 * PARENT_EMBEDDING_PARAMETERS


def test_the_registry_resolves_the_route_width_from_its_own_storage_model():
    state = {
        "model.embed_tokens.weight": torch.zeros(
            (PARENT_VOCAB, PARENT_HIDDEN), dtype=torch.float64
        )
    }
    spec = component_spec(CellId.parse("C01.T5.S4"), state)
    model = storage_model("T5", "C01")
    quarter = storage_width_for_ratio(spec, "T5", 0.25)
    eighth = storage_width_for_ratio(spec, "T5", 0.125)
    assert quarter == 201
    assert eighth == 67
    assert model(spec, quarter) <= 0.25 * PARENT_EMBEDDING_PARAMETERS
    assert model(spec, quarter + 1) > 0.25 * PARENT_EMBEDDING_PARAMETERS
    assert model(spec, eighth) <= 0.125 * PARENT_EMBEDDING_PARAMETERS
    assert model(spec, eighth + 1) > 0.125 * PARENT_EMBEDDING_PARAMETERS
    # The route's model is C01's alone; the body keeps T5's own route algebra.
    assert storage_model("T5", "C13") is not model


def test_without_a_frequency_table_the_head_is_the_lowest_token_ids():
    """BPE merge order, which measured 62.95% prompt coverage against row norm's
    2.34% on the frozen review surface's own tokens."""

    table = _table(seed=31)
    table[40] *= 50.0  # a huge row a norm-ordered head would have taken first
    table[7] *= 40.0
    route = route_token_embedding(embedding=table, head_size=3, tail_rank=3)
    assert route.head_ids.tolist() == [0, 1, 2]
    assert route.solve["route"]["head_selection_rule"].startswith("token-id-ascending")


def test_counts_beat_ids_and_ids_break_every_tie():
    table = _table(seed=32)
    counts = torch.zeros(VOCAB, dtype=torch.float64)
    counts[90] = 5.0
    counts[50] = 9.0
    route = route_token_embedding(
        embedding=table, head_size=4, tail_rank=3, weights=counts
    )
    # the two counted ids, then the two lowest ids among the uncounted rest
    assert route.head_ids.tolist() == [0, 1, 50, 90]


def test_the_route_kernel_is_registered_at_c01_under_t5(tmp_path):
    from neural_forge import matrix_technique_body_routes  # noqa: F401

    assert resolve("T5", "C01") is t5_token_embedding_route_technique
    assert resolve("T5", "C13") is not t5_token_embedding_route_technique

    state = _state(_table(seed=41))
    counts = [0] * VOCAB
    for index in range(10):
        counts[index] = 100 - index
    path, digest = _frequency_file(tmp_path, counts)
    cell = CellId.parse("C01.T5.S4")
    spec = component_spec(cell, state)
    request = CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=3,
        seed=0,
        parent_weights=state,
        evidence=_NoEvidence(),
        knobs={
            "token_route_head_size": 8,
            "token_frequency_path": str(path),
            "token_frequency_sha256": digest,
        },
        workspace=Path("."),
        log=lambda message: None,
    )
    result = t5_token_embedding_route_technique(request)
    validate_result(request, result)
    assert result.stored_parameter_count == 8 * (HIDDEN + 1) + 3 * (
        (VOCAB - 8) + HIDDEN
    )
    installed = result.installed_weights["model.embed_tokens.weight"]
    assert torch.equal(installed[:8], state["model.embed_tokens.weight"][:8])
    assert result.solve["token_frequency"]["token_frequency_sha256"] == digest
    assert result.solve["route"]["head_size"] == 8


def test_a_head_that_covers_the_table_is_refused():
    with pytest.raises(TokenEmbeddingFoldError, match="covers the whole"):
        route_token_embedding(embedding=_table(), head_size=VOCAB, tail_rank=2)


def test_a_tail_rank_past_the_tails_full_rank_is_refused():
    with pytest.raises(TokenEmbeddingFoldError, match="exceeds the tail"):
        route_token_embedding(embedding=_table(), head_size=8, tail_rank=HIDDEN + 1)


# ---------------------------------------------------------------------------
# Wave 10: the A9/A11 sparse-OMP tail
#
# 8. **The two tails are handed the same head.**  The head-to-head this wave
#    exists to run is only a comparison of tails if the head-selection rule is
#    literally the same function, so that is pinned with ``torch.equal`` on the
#    ids rather than argued in a comment.
# 9. **The storage counts the indices.**  A9's own accounting counted the
#    coefficients and not the atom ids they index; that understatement is the
#    reason its 1.361447x is not a stored ratio.  The kernel reports both, and
#    the honest one is the one the harness sees.
# 10. **Exactness where exactness is claimed.**  Head rows are verbatim; a zero
#    row costs no support at all; and at ``k`` equal to the atom count on an
#    atom-spanned table the tail reconstructs to solver precision.
# ---------------------------------------------------------------------------


def test_both_tail_modes_select_the_identical_head():
    """The comparison is between tails, or it is not a comparison."""

    table = _table(seed=19)
    counts = [0] * VOCAB
    for index in (5, 40, 77, 3):
        counts[index] = 50 + index
    weights = torch.tensor(counts, dtype=torch.float64)

    low_rank = route_token_embedding(
        embedding=table, head_size=12, tail_rank=4, weights=weights
    )
    sparse = sparse_route_token_embedding(
        embedding=table, head_size=12, sparsity=4, weights=weights
    )
    assert torch.equal(low_rank.head_ids, sparse.head_ids)
    assert low_rank.record["head_selection_rule"] == sparse.record["head_selection_rule"]
    assert low_rank.record["head_ids_sha256"] == sparse.record["head_ids_sha256"]


def test_the_sparse_route_stores_the_head_rows_exactly():
    table = _table(seed=23)
    route = sparse_route_token_embedding(embedding=table, head_size=10, sparsity=3)
    assert torch.equal(route.table[route.head_ids], table[route.head_ids])
    assert torch.equal(route.head_rows, table[route.head_ids])


def test_the_sparse_route_storage_counts_the_indices_and_the_offsets():
    """The exact arithmetic the A9-era counted-x left out."""

    table = _table(seed=29)
    route = sparse_route_token_embedding(embedding=table, head_size=10, sparsity=3)
    tail = VOCAB - 10
    expected = 10 * (HIDDEN + 1) + 2 * route.nonzero_count + (tail + 1)
    assert route.stored_parameter_count == expected
    assert (
        c01_sparse_route_storage(
            hidden_size=HIDDEN,
            head_size=10,
            tail_size=tail,
            nonzero_count=route.nonzero_count,
        )
        == expected
    )
    # The legacy number is the one A9 published: continuous values only.
    assert (
        route.record["legacy_continuous_only_parameters"]
        == 10 * HIDDEN + route.nonzero_count
    )
    assert route.record["legacy_continuous_only_parameters"] < expected


def test_a_zero_tail_row_selects_no_support_and_is_reconstructed_exactly():
    """An unused vocabulary slot is free, not a special case."""

    table = _table(seed=31).clone()
    table[VOCAB - 1] = 0.0
    route = sparse_route_token_embedding(embedding=table, head_size=8, sparsity=4)
    assert bool(torch.all(route.table[VOCAB - 1] == 0.0))
    # Its support is empty, so it contributes nothing to the stored count.
    assert route.nonzero_count < (VOCAB - 8) * 4


def test_a_zero_head_row_is_refused_rather_than_substituted():
    table = _table(seed=37).clone()
    table[3] = 0.0
    with pytest.raises(TokenEmbeddingFoldError, match="exactly zero"):
        sparse_route_token_embedding(embedding=table, head_size=8, sparsity=4)


def test_a_full_support_over_a_spanning_head_reconstructs_the_table():
    """The identity gate for the sparse arm, and the price of the ridge.

    Unridged, the solve is exact to float64: every tail row here really is a
    combination of the head's atoms, OMP recovers all 16, and the least squares
    is the identity.  The kernel's *default* ridge is the body lane's 1e-10,
    which on this deliberately nasty Gram (condition 2.5e4) biases the
    reconstruction to 2e-9 relative -- measured, not assumed, and pinned here so
    that a future ridge change shows up as a failing gate rather than as a
    quietly worse fold.
    """

    generator = torch.Generator().manual_seed(11)
    atoms = torch.randn((HIDDEN, HIDDEN), dtype=torch.float64, generator=generator)
    mixture = torch.randn(
        (VOCAB - HIDDEN, HIDDEN), dtype=torch.float64, generator=generator
    )
    table = torch.cat((atoms, mixture @ atoms), dim=0)

    exact = sparse_route_token_embedding(
        embedding=table, head_size=HIDDEN, sparsity=HIDDEN, ridge_relative=0.0
    )
    assert torch.allclose(exact.table, table, atol=1e-11, rtol=1e-11)
    assert exact.solve["reconstruction_relative_frobenius"] < 1e-12
    assert exact.nonzero_count == (VOCAB - HIDDEN) * HIDDEN

    ridged = sparse_route_token_embedding(
        embedding=table, head_size=HIDDEN, sparsity=HIDDEN
    )
    assert ridged.solve["reconstruction_relative_frobenius"] < 1e-8
    assert (
        ridged.solve["reconstruction_relative_frobenius"]
        > exact.solve["reconstruction_relative_frobenius"]
    )


def test_the_sparsity_planner_never_plans_over_the_budget():
    """Planned on the worst case, so a realized route can only come in under."""

    ratio = 0.125
    sparsity = c01_sparse_route_sparsity_for_ratio(
        vocab_size=PARENT_VOCAB,
        hidden_size=PARENT_HIDDEN,
        head_size=4096,
        ratio=ratio,
    )
    assert sparsity is not None
    tail = PARENT_VOCAB - 4096
    worst_case = c01_sparse_route_storage(
        hidden_size=PARENT_HIDDEN,
        head_size=4096,
        tail_size=tail,
        nonzero_count=tail * sparsity,
    )
    assert worst_case <= ratio * PARENT_EMBEDDING_PARAMETERS
    one_deeper = c01_sparse_route_storage(
        hidden_size=PARENT_HIDDEN,
        head_size=4096,
        tail_size=tail,
        nonzero_count=tail * (sparsity + 1),
    )
    assert one_deeper > ratio * PARENT_EMBEDDING_PARAMETERS


def test_a_head_too_big_for_the_budget_cannot_express_the_level():
    """ADDENDUM 6's cannot-express, rather than a shallower fold."""

    assert (
        c01_sparse_route_sparsity_for_ratio(
            vocab_size=PARENT_VOCAB,
            hidden_size=PARENT_HIDDEN,
            head_size=32768,
            ratio=0.125,
        )
        is None
    )


def test_the_sparse_arm_runs_through_the_registry_and_validates(tmp_path):
    from neural_forge import matrix_technique_body_routes  # noqa: F401

    state = _state(_table(seed=43))
    counts = [0] * VOCAB
    for index in range(6):
        counts[index] = 100 - index
    path, digest = _frequency_file(tmp_path, counts)
    cell = CellId.parse("C01.T5.S4")
    spec = component_spec(cell, state)
    request = CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=5,
        seed=0,
        parent_weights=state,
        evidence=_NoEvidence(),
        knobs={
            "token_route_tail_mode": "sparse-omp",
            "token_route_head_size": 12,
            "token_frequency_path": str(path),
            "token_frequency_sha256": digest,
        },
        workspace=Path("."),
        log=lambda message: None,
    )
    result = t5_token_embedding_route_technique(request)
    validate_result(request, result)
    assert result.solve["route"]["tail_mode"] == "sparse-omp"
    assert result.solve["route"]["sparsity"] == 5
    assert result.solve["solver_path"] == "c01_token_route_exact_head_sparse_omp_tail"
    assert result.solve["omp"]["requested_active_k"] == 5
    assert result.solve["token_frequency"]["token_frequency_sha256"] == digest
    installed = result.installed_weights["model.embed_tokens.weight"]
    assert torch.equal(installed[:6], state["model.embed_tokens.weight"][:6])
    tail = VOCAB - 12
    nonzero = result.solve["route"]["nonzero_count"]
    assert result.stored_parameter_count == 12 * (HIDDEN + 1) + 2 * nonzero + (tail + 1)


def test_the_default_tail_mode_is_still_wave_nines():
    """A wave-9 recipe must fold the way wave 9 folded, byte for byte."""

    state = _state(_table(seed=47))
    cell = CellId.parse("C01.T5.S4")
    spec = component_spec(cell, state)

    def build(knobs):
        return CellTechniqueRequest(
            cell=cell,
            component=spec,
            level=cell.level,
            level_width=4,
            seed=0,
            parent_weights=state,
            evidence=_NoEvidence(),
            knobs=dict(knobs),
            workspace=Path("."),
            log=lambda message: None,
        )

    default = t5_token_embedding_route_technique(build({"token_route_head_size": 9}))
    explicit = t5_token_embedding_route_technique(
        build({"token_route_head_size": 9, "token_route_tail_mode": "low-rank"})
    )
    assert torch.equal(
        default.installed_weights["model.embed_tokens.weight"],
        explicit.installed_weights["model.embed_tokens.weight"],
    )
    assert default.solve["solver_path"] == "c01_token_route_exact_head_low_rank_tail"
    assert default.stored_parameter_count == c01_token_route_storage(
        vocab_size=VOCAB, hidden_size=HIDDEN, head_size=9, tail_rank=4
    )


def test_an_unknown_tail_mode_fails_closed():
    state = _state(_table(seed=53))
    cell = CellId.parse("C01.T5.S4")
    request = CellTechniqueRequest(
        cell=cell,
        component=component_spec(cell, state),
        level=cell.level,
        level_width=4,
        seed=0,
        parent_weights=state,
        evidence=_NoEvidence(),
        knobs={"token_route_tail_mode": "svd-ish"},
        workspace=Path("."),
        log=lambda message: None,
    )
    with pytest.raises(CellTechniqueError, match="token_route_tail_mode"):
        t5_token_embedding_route_technique(request)


def test_a_sparsity_past_the_dictionary_is_refused():
    with pytest.raises(TokenEmbeddingFoldError, match="exceeds the"):
        sparse_route_token_embedding(embedding=_table(), head_size=8, sparsity=9)


def test_the_sparse_route_refuses_autograd_state():
    table = _table(seed=59).clone().requires_grad_(True)
    with pytest.raises(TokenEmbeddingFoldError, match="autograd"):
        sparse_route_token_embedding(embedding=table, head_size=8, sparsity=3)


def test_an_explicit_solve_device_is_recorded_and_the_result_comes_home():
    """The harness hands C01 a CPU table; a cuda solve must still return CPU."""

    table = _table(seed=61)
    home = sparse_route_token_embedding(
        embedding=table, head_size=10, sparsity=4, solve_device="cpu"
    )
    assert home.solve["route"]["solve_device"] == "cpu"
    assert home.table.device == table.device
    assert home.supports.device == table.device

    if torch.cuda.is_available():
        moved = sparse_route_token_embedding(
            embedding=table, head_size=10, sparsity=4, solve_device="cuda:0"
        )
        assert moved.solve["route"]["solve_device"].startswith("cuda")
        # Comes home regardless of where it was solved.
        assert moved.table.device == table.device
        assert moved.stored_parameter_count == home.stored_parameter_count
