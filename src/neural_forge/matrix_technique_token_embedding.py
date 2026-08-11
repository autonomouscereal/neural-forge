"""C01 — the tied token embedding, factored, as a matrix-campaign cell.

Why this module exists at all is arithmetic rather than curiosity.  The parent
carries 229,693,184 unique parameters and **67,108,864 of them (29.2%) are one
tensor**: ``model.embed_tokens.weight``, a 65,536 x 1,024 table that LFM2 ties
to the readout.  ADDENDUM 6's storage levels bound what a *component* stores,
and every component the campaign has folded through wave 8 lives in the other
70.8%.  Folding the whole body at a true quarter therefore lands at 0.469x of
the parent and can go no lower: **no body-only recipe reaches 0.25x.**  The
embedding has to be folded, at 4x or deeper, or the campaign's terminal target
is unreachable by construction.

What is stored
--------------
A rank-``r`` factorization of the table::

    E  ~  A @ V^T        A = E @ V   (vocab x r)      V   (hidden x r)

so the child stores ``r * (vocab + hidden)`` against the parent's
``vocab * hidden``.  That is exactly `matrix_technique_registry
._factor_pair_storage`, which is why this kernel registers under **T4**
(``low_rank_factor_writer``) rather than inventing a technique id whose storage
algebra the planner would not already know: ``storage_width_for_ratio`` can
resolve a width for S2/S4/S8 at C01 the moment this kernel exists, with no new
storage model and no receipt-schema move.

At 65,536 x 1,024 the break-even rank is ``vocab*hidden/(vocab+hidden)`` =
**1,008**, so anything at or below the table's full rank of 1,024 is within a
few ranks of contracting and a true quarter is r = 252.  The embedding is the
one component in the whole matrix where a *deep* rank is cheap, because the two
axes are so lopsided.

The tie is the point, not a complication
----------------------------------------
``lm_head.weight`` is the same storage as ``model.embed_tokens.weight``, so a
single install folds both the lookup and the readout::

    lookup(t)  =  A[t] @ V^T
    logits(h)  =  h @ E^T  =  h @ (A V^T)^T  =  (h @ V) @ A^T

The readout therefore projects the hidden state onto ``span(V)`` before it
scores tokens, which is the entire behavioural risk of this fold and the reason
wave 9 measures C01 alone before it is composed with anything.  The child's
`state_dict` still holds one parent-shaped table -- the campaign's cells always
materialize at parent geometry and declare the contraction in
``stored_parameter_count`` (`matrix_technique_registry`, "why parent-shaped
output") -- so no neighbour, no tokenizer and no generation path changes.

Two weightings, one closed form, no gradients
---------------------------------------------
The fit minimizes a **weighted** row error::

    minimize  sum_v  w_v * || E_v - Ehat_v ||^2      over rank(Ehat) <= r

With ``W = diag(w)`` and ``M = W^(1/2) E``, Eckart-Young gives
``Ehat = W^(-1/2) trunc_r(M)``, and since
``W^(-1/2) U_r S_r = W^(-1/2) M V_r = E V_r`` the solution collapses to a
projection::

    Ehat = E V_r V_r^T ,   V_r = top-r eigenvectors of  E^T W E

which is numerically safe for ``w_v = 0`` (an unseen token still gets its best
projection onto the retained subspace; it simply does not vote on which
subspace that is), needs one 1,024 x 1,024 eigendecomposition, and is exact:

* ``weighting="uniform"`` (``w = 1``) is **plain truncated SVD** of the table --
  ``E^T E``'s top eigenvectors are E's right singular vectors -- and at
  ``r = hidden`` it reconstructs E identically, which is the tiny-rank identity
  gate `tests/test_matrix_technique_token_embedding.py` pins.
* ``weighting="token-frequency"`` weights each row by how often that token
  actually occurs in the campaign's sealed evidence corpus.  It is a closed-form
  weighted least squares, not a training loop: **no autograd, no optimizer, no
  parent forward.**

The frequency table is supplied as a file plus its digest, and the digest is
sealed into the receipt, because a weighting the receipt cannot name is a fit
nobody can reproduce.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import canonical_sha256

TOKEN_EMBEDDING_RECORD_SCHEMA = "neural-forge-matrix-c01-token-embedding-record-v1"
TOKEN_FREQUENCY_SCHEMA = "neural-forge-matrix-token-frequency-v1"

#: The technique this kernel registers under.  T4 already publishes the
#: ``r * (in + out)`` storage algebra this factorization stores at, so C01 needs
#: no new storage model and no new technique id.
TECHNIQUE_ID = "T4"
COMPONENT_ID = "C01"

WEIGHTINGS = ("uniform", "token-frequency")


class TokenEmbeddingFoldError(ValueError):
    """The embedding factorization was asked for something it cannot honestly do."""


def _torch() -> Any:
    import torch

    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TokenEmbeddingFoldError(f"{name} must be a positive integer.")
    return int(value)


def maximum_shrinking_rank(vocab_size: int, hidden_size: int) -> int:
    """Highest rank at which ``r*(V+H)`` still costs less than ``V*H``.

    On the parent's 65,536 x 1,024 table that is 1,008 against a full rank of
    1,024, so the table contracts at all but the last sixteen ranks -- the
    opposite of a square writer, where the same algebra kills every marginal
    level (`matrix_technique_lowrank_writers.maximum_shrinking_rank`).
    """

    vocab = _positive_int(vocab_size, "vocab_size")
    hidden = _positive_int(hidden_size, "hidden_size")
    return max((vocab * hidden - 1) // (vocab + hidden), 0)


def load_token_frequencies(
    path: Path | str,
    *,
    vocab_size: int,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Read a sealed token-frequency table and return ``(weights, provenance)``.

    The file is a plain JSON object -- there is no database, no ORM and no
    serialization framework anywhere in this campaign -- carrying one count per
    vocabulary id and the digest of the corpus it was counted over.  The file's
    own sha256 is checked against ``expected_sha256`` when the caller supplies
    one, so a receipt that names a weighting also names the exact bytes that
    produced it.
    """

    torch = _torch()
    location = Path(path).expanduser()
    raw = location.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise TokenEmbeddingFoldError(
            f"{location} digests to {digest[:16]}..., not the expected "
            f"{str(expected_sha256)[:16]}...; the weighting is not the sealed one."
        )
    body = json.loads(raw.decode("utf-8"))
    if str(body.get("schema")) != TOKEN_FREQUENCY_SCHEMA:
        raise TokenEmbeddingFoldError(
            f"{location} does not declare {TOKEN_FREQUENCY_SCHEMA!r}."
        )
    counts = body.get("counts")
    if not isinstance(counts, list) or len(counts) != int(vocab_size):
        raise TokenEmbeddingFoldError(
            f"{location} carries {0 if not isinstance(counts, list) else len(counts)} "
            f"counts; the parent's vocabulary is {int(vocab_size)}."
        )
    weights = torch.tensor(
        [float(value) for value in counts], dtype=torch.float64
    )
    if not bool(torch.isfinite(weights).all().item()) or bool(
        (weights < 0).any().item()
    ):
        raise TokenEmbeddingFoldError("Token counts must be finite and nonnegative.")
    total = float(weights.sum().item())
    if total <= 0.0:
        raise TokenEmbeddingFoldError(
            "Every token count is zero; the weighting carries no evidence."
        )
    observed = int((weights > 0).sum().item())
    provenance = {
        "token_frequency_path": str(location),
        "token_frequency_sha256": digest,
        "token_frequency_schema": TOKEN_FREQUENCY_SCHEMA,
        "corpus_sha256": body.get("corpus_sha256"),
        "corpus_record_count": body.get("corpus_record_count"),
        "corpus_token_count": total,
        "vocabulary_size": int(vocab_size),
        "observed_token_types": observed,
        "observed_token_type_fraction": observed / float(vocab_size),
        "unobserved_token_types": int(vocab_size) - observed,
    }
    return weights, provenance


@dataclass(frozen=True)
class TokenEmbeddingFactorization:
    """One tied embedding table factored into ``A @ V^T`` at a fixed rank."""

    rank: int
    vocab_size: int
    hidden_size: int
    weighting: str
    row_factor: Any  # A = E V   (vocab x r)
    basis: Any  # V         (hidden x r)
    eigenvalues: Any  # the retained spectrum of E^T W E, descending
    record: Mapping[str, Any]
    solve: Mapping[str, Any]

    @property
    def stored_parameter_count(self) -> int:
        return int(self.rank) * (int(self.vocab_size) + int(self.hidden_size))

    @property
    def dense_equivalent(self) -> Any:
        """``A @ V^T`` -- the parent-shaped table the child installs."""

        return self.row_factor @ self.basis.transpose(0, 1)

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])


def factor_token_embedding(
    *,
    embedding: Any,
    rank: int,
    weights: Any = None,
    weighting: str = "uniform",
) -> TokenEmbeddingFactorization:
    """Closed-form weighted rank-``r`` factorization of a token embedding table.

    ``embedding`` is the parent's ``(vocab, hidden)`` table, read-only.
    ``weights`` is an optional per-row nonnegative weight vector; when it is
    ``None`` the fit is the plain truncated SVD.  Nothing here differentiates,
    steps an optimizer, quantizes or prunes: it is one symmetric
    eigendecomposition of a ``hidden x hidden`` Gram and two matrix products.
    """

    torch = _torch()
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise TokenEmbeddingFoldError("The embedding must be a 2-D tensor.")
    if embedding.requires_grad or embedding.grad_fn is not None:
        raise TokenEmbeddingFoldError(
            "The embedding carries autograd state; this is a fold, not a training step."
        )
    name = str(weighting)
    if name not in WEIGHTINGS:
        raise TokenEmbeddingFoldError(
            f"weighting={name!r} is not one of {list(WEIGHTINGS)}."
        )
    table = embedding.detach().to(dtype=torch.float64)
    vocab_size = int(table.shape[0])
    hidden_size = int(table.shape[1])
    full_rank = min(vocab_size, hidden_size)
    resolved = _positive_int(rank, "rank")
    if resolved > full_rank:
        raise TokenEmbeddingFoldError(
            f"rank {resolved} exceeds the table's full rank {full_rank}."
        )

    if name == "uniform":
        if weights is not None:
            raise TokenEmbeddingFoldError(
                "weighting='uniform' takes no weight vector; pass "
                "weighting='token-frequency' to use one."
            )
        row_weights = None
        gram = table.transpose(0, 1) @ table
    else:
        if weights is None:
            raise TokenEmbeddingFoldError(
                "weighting='token-frequency' needs a per-token weight vector."
            )
        row_weights = weights.detach().to(dtype=torch.float64).reshape(-1)
        if int(row_weights.shape[0]) != vocab_size:
            raise TokenEmbeddingFoldError(
                f"The weight vector carries {int(row_weights.shape[0])} entries "
                f"for a {vocab_size}-token table."
            )
        if bool((row_weights < 0).any().item()):
            raise TokenEmbeddingFoldError("Token weights must be nonnegative.")
        # Normalizing to mean 1 keeps the Gram's scale comparable with the
        # uniform arm's, so the two arms' condition numbers and retained-energy
        # fractions are read on the same axis.  It cannot change the subspace:
        # scaling W scales every eigenvalue equally.
        total = float(row_weights.sum().item())
        row_weights = row_weights * (float(vocab_size) / total)
        gram = table.transpose(0, 1) @ (row_weights.unsqueeze(1) * table)

    gram = ((gram + gram.transpose(0, 1)) * 0.5).contiguous()
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    basis = vectors[:, :resolved].contiguous()
    # Sign is pinned by the largest-magnitude entry, exactly as `_pod_basis`
    # does, so two runs of the same cell are the same cell rather than the same
    # subspace wearing different signs.
    pivots = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivots, torch.arange(basis.shape[1])])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = (basis * signs).contiguous()

    row_factor = (table @ basis).contiguous()
    approximation = row_factor @ basis.transpose(0, 1)
    residual = approximation - table

    spectrum = values.clamp_min(0.0)
    total_energy = float(spectrum.sum().item())
    retained_energy = float(spectrum[:resolved].sum().item())
    smallest_retained = float(spectrum[resolved - 1].item())
    largest = float(spectrum[0].item())
    condition = (
        largest / smallest_retained if smallest_retained > 0.0 else float("inf")
    )

    parent_norm = float(torch.linalg.matrix_norm(table).item())
    residual_norm = float(torch.linalg.matrix_norm(residual).item())
    row_error = torch.linalg.vector_norm(residual, dim=1)
    row_norm = torch.linalg.vector_norm(table, dim=1).clamp_min(1e-30)
    relative_rows = row_error / row_norm
    cosines = torch.nn.functional.cosine_similarity(
        approximation, table, dim=1, eps=1e-30
    )

    if row_weights is None:
        weighted_relative = float((residual_norm / max(parent_norm, 1e-30)))
        weighted_row_cosine = float(cosines.mean().item())
    else:
        weighted_square = float(
            (row_weights * row_error.square()).sum().item()
        )
        weighted_parent = float(
            (row_weights * torch.linalg.vector_norm(table, dim=1).square())
            .sum()
            .item()
        )
        weighted_relative = math.sqrt(
            weighted_square / max(weighted_parent, 1e-30)
        )
        weighted_row_cosine = float(
            ((row_weights * cosines).sum() / row_weights.sum()).item()
        )

    stored = resolved * (vocab_size + hidden_size)
    parent_parameters = vocab_size * hidden_size
    ceiling = maximum_shrinking_rank(vocab_size, hidden_size)
    record: dict[str, Any] = {
        "schema": TOKEN_EMBEDDING_RECORD_SCHEMA,
        "component_id": COMPONENT_ID,
        "technique_id": TECHNIQUE_ID,
        "weighting": name,
        "rank": resolved,
        "full_rank": full_rank,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "parent_parameters": parent_parameters,
        "child_parameters": stored,
        "parameter_ratio": stored / float(parent_parameters),
        "maximum_shrinking_rank": ceiling,
        "shrinks_the_table": bool(stored < parent_parameters),
        "row_factor_sha256": _tensor_sha256(row_factor),
        "basis_sha256": _tensor_sha256(basis),
        "dense_equivalent_sha256": _tensor_sha256(approximation),
    }
    record["record_sha256"] = canonical_sha256(record)

    solve: dict[str, Any] = {
        "solver_path": f"c01_token_embedding_weighted_svd:{name}",
        "ridge_relative": 0.0,
        # The Gram's retained spectrum, which is the system the projection is
        # read off.  Finite by construction unless the table is rank deficient
        # inside the retained block, and that case is reported as saturated
        # rather than sealed as a number that does not exist.
        "condition_number": condition if math.isfinite(condition) else None,
        "condition_cap_saturated": not math.isfinite(condition),
        "condition_number_unbounded_reason": (
            None if math.isfinite(condition) else "singular-retained-spectrum"
        ),
        "prior_norm": parent_norm,
        "correction_norm": residual_norm,
        "requested_clamp_ratio": residual_norm / max(parent_norm, 1e-30),
        # No prior-delta clamp lane exists for a projection: there is no budget
        # to exceed, so the schema's null says so rather than an invented number.
        "allowed_clamp_ratio": None,
        "clamp_engaged": False,
        "omp": None,
        "weighting": name,
        "embedding_rank": resolved,
        "embedding_full_rank": full_rank,
        "embedding_vocab_size": vocab_size,
        "embedding_hidden_size": hidden_size,
        "maximum_shrinking_rank": ceiling,
        "shrinks_the_table": bool(stored < parent_parameters),
        "gram_total_energy": total_energy,
        "gram_retained_energy": retained_energy,
        "retained_energy_fraction": (
            retained_energy / total_energy if total_energy > 0.0 else None
        ),
        "smallest_retained_eigenvalue": smallest_retained,
        "largest_eigenvalue": largest,
        "reconstruction_relative_frobenius": residual_norm
        / max(parent_norm, 1e-30),
        "weighted_reconstruction_relative_frobenius": weighted_relative,
        "row_relative_l2_mean": float(relative_rows.mean().item()),
        "row_relative_l2_max": float(relative_rows.max().item()),
        "row_cosine_mean": float(cosines.mean().item()),
        "row_cosine_min": float(cosines.min().item()),
        "weighted_row_cosine_mean": weighted_row_cosine,
        "technique_record_sha256": str(record["record_sha256"]),
    }
    return TokenEmbeddingFactorization(
        rank=resolved,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        weighting=name,
        row_factor=row_factor,
        basis=basis,
        eigenvalues=spectrum[:resolved].contiguous(),
        record=record,
        solve=solve,
    )


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    contiguous = value.detach().to(dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def t4_token_embedding_technique(request: Any) -> Any:
    """T4's kernel at C01: fold the tied embedding table, install it dense.

    Reads no evidence rows.  C01's layout declares no input, write or macrocell
    endpoint -- there is no activation boundary at which a table lookup is a
    nameable linear write -- so the fit is a function of the parent tensor and,
    on the weighted arm, of a sealed token-frequency file whose digest rides in
    the receipt.  The harness's per-component calibration and boundary links are
    declared inapplicable for the same reason, and the cell rests its verdict on
    the composed forward, the free-run probe and the frozen review, exactly as
    C03/C04/C10/C14 already do.
    """

    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    spec = request.component
    if spec.component_id != COMPONENT_ID:
        raise CellTechniqueError(
            f"This kernel folds {COMPONENT_ID}; it was handed {spec.component_id}."
        )
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach()
    vocab_size = int(parent.shape[0])
    hidden_size = int(parent.shape[1])

    rank = int(request.knob("embedding_rank", int(request.level_width)))
    weighting = str(request.knob("embedding_weighting", "uniform"))
    weights = None
    provenance: dict[str, Any] = {"weighting": weighting}
    if weighting == "token-frequency":
        path = request.knob("token_frequency_path")
        if not path:
            raise CellTechniqueError(
                "weighting='token-frequency' needs knob token_frequency_path."
            )
        try:
            weights, provenance = load_token_frequencies(
                path,
                vocab_size=vocab_size,
                expected_sha256=request.knob("token_frequency_sha256"),
            )
        except TokenEmbeddingFoldError as exc:
            raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc
        provenance["weighting"] = weighting

    request.log(
        f"C01 {request.cell.text}: rank={rank} of {min(vocab_size, hidden_size)}, "
        f"weighting={weighting}, table {vocab_size}x{hidden_size}"
    )
    try:
        factorization = factor_token_embedding(
            embedding=parent,
            rank=rank,
            weights=weights,
            weighting=weighting,
        )
    except TokenEmbeddingFoldError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    if not factorization.record["shrinks_the_table"] and not bool(
        request.knob("c01_allow_parameter_increase", False)
    ):
        raise CellTechniqueError(
            f"{request.cell.text}: rank {factorization.rank} stores "
            f"{factorization.stored_parameter_count} against a table of "
            f"{vocab_size * hidden_size}; the highest rank that contracts is "
            f"{factorization.record['maximum_shrinking_rank']}."
        )

    installed = factorization.dense_equivalent.contiguous()
    solve = dict(factorization.solve)
    # Nested, not flattened: the provenance of a weighting is a statement about
    # a file, and burying its digest among the solve's own measurements would
    # make the one field a reproduction actually needs the hardest to find.
    solve["token_frequency"] = (
        None if weighting == "uniform" else {
            name: value for name, value in provenance.items() if name != "weighting"
        }
    )
    request.log(
        f"C01 {request.cell.text}: stored {factorization.stored_parameter_count} "
        f"({factorization.record['parameter_ratio']:.6f}x), retained energy "
        f"{solve['retained_energy_fraction']}, row cosine mean "
        f"{solve['row_cosine_mean']:.6f}"
    )
    return CellTechniqueResult(
        installed_weights={key: installed},
        stored_parameter_count=int(factorization.stored_parameter_count),
        solve=solve,
        # No sealed write endpoint, so no held-out write predictor exists; the
        # harness declares the boundary link inapplicable rather than scoring a
        # prediction against an endpoint that carries a different width.
        predict=None,
        artifact_identities={
            "c01_token_embedding_record_sha256": factorization.record_sha256
        },
        notes=(
            f"C01 tied token embedding factored at rank {factorization.rank} "
            f"({weighting} weighting), storing "
            f"{factorization.stored_parameter_count} against "
            f"{vocab_size * hidden_size} -- the lookup and the tied readout are "
            "one tensor and this install folds both."
        ),
    )


def _register_token_embedding_technique() -> None:
    """Wire C01 into the harness dispatch table at import time.

    Pinned to the component, so `resolve` prefers it over T4's generic writer
    kernel by the most-specific-wins rule and neither registration edits the
    other's file.
    """

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    register(
        TECHNIQUE_ID,
        t4_token_embedding_technique,
        component=COMPONENT_ID,
        replace=True,
    )


_register_token_embedding_technique()


__all__ = [
    "COMPONENT_ID",
    "TECHNIQUE_ID",
    "TOKEN_EMBEDDING_RECORD_SCHEMA",
    "TOKEN_FREQUENCY_SCHEMA",
    "WEIGHTINGS",
    "TokenEmbeddingFactorization",
    "TokenEmbeddingFoldError",
    "factor_token_embedding",
    "load_token_frequencies",
    "maximum_shrinking_rank",
    "t4_token_embedding_technique",
]
