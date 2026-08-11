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
    factor_token_embedding,
    load_token_frequencies,
    maximum_shrinking_rank,
    t4_token_embedding_technique,
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
