"""The lane's self-tests, on a TINY random-init qwen3_5.

Why a random-init model and not the 4B: every claim these tests make is a claim
about MECHANICS -- indices, shapes, config re-derivation, conventions, the
identity gates -- and a mechanic that is wrong is wrong at hidden 128 too.
Running them here costs a minute on a CPU and buys the right to run the 4B once
instead of five times.

Why random-init and not a checkpoint: this suite must be runnable before any
parent has finished copying to local disk, which is exactly the situation the
lane is in.

Deliberate design choices in the fixture (see ``qf_arch.tiny_text_config``):
every norm gets a NONZERO weight, because ``Qwen3_5RMSNorm`` initialises to zero
and zero is the one value at which the ``1 + w`` convention and the ``w``
convention agree -- a suite run on a default-init model would pass with the gain
handling completely wrong.

Run:  python qf_selftest.py [--only NAME] [--keep-artifacts]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qf_arch
import qf_attn
import qf_corpus
import qf_excise
import qf_export
import qf_gdn
import qf_mlp
import qf_solve
import qf_termgain
import qf_vocab
from qf_harness import Harness, harness_parity

TESTS: list = []
WORK: Path = Path(".")
SEQUENCE = 48
BATCHES = 9
HELD_EVERY = 3

#: The fp64 recompute of an attention layer against the fp32 module's own
#: probabilities.  A proposed decision line for Cyril, at the same order as the
#: harness's measured hook-placement parity (~4e-7), never a gate this file
#: invents.
PROBE_TOLERANCE = 1e-5

#: How far the child's attention logits may be from the parent's in overall
#: SCALE, measured by :func:`qf_attn.logit_scale_ratio` on the causal support.
#: A C6 child that forgot the softmax-temperature correction lands at
#: ``sqrt(child_head_dim/parent_head_dim)`` -- 0.661 at the fixture's 32 -> 14 --
#: so this line separates "the truncation's own error" from "the wrong
#: temperature" by a factor of two.  Proposed decision line, not a gate.
TEMPERATURE_LINE = 0.15


def test(fn):
    TESTS.append(fn)
    return fn


def quiet(*_args, **_kwargs) -> None:
    return None


# ------------------------------------------------------------------- fixtures
def tiny_parent(tag: str = "parent") -> tuple[Path, dict[str, Any]]:
    """Save a tiny random-init model + a real BPE tokenizer + sealed evidence."""

    directory = WORK / tag
    if directory.exists():
        shutil.rmtree(directory)
    config = qf_arch.tiny_text_config()
    directory.mkdir(parents=True)
    qf_vocab.synthetic_tokenizer(int(config.vocab_size), directory / "tokenizer.json")

    # The hard ids are taken from the tokenizer's SPECIALS, which sit at the TOP
    # of the table, and `eos_token_id` is a LIST -- both on purpose.  A fixture
    # with scalar bos=0/eos=1 puts every hard id inside the byte alphabet at the
    # bottom of the keep set, where the C1 remap is the IDENTITY and a remap that
    # does nothing at all still passes.  Real Qwen configs carry
    # `eos_token_id = [<|im_end|>, <|endoftext|>]`, so the fixture does too.
    body = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    specials = {row["content"]: int(row["id"]) for row in body["added_tokens"]}
    #: The highest MERGE-COMPOSED row: not a byte token, not an added token, and
    #: numerically above the keep size the C1 test uses, so it is mandatory only
    #: because a config names it and a fold that forgets it leaves an id the
    #: child cannot address.
    composed = max(
        index for index in body["model"]["vocab"].values()
        if index >= 256 and index not in set(specials.values())
    )
    config.bos_token_id = specials["<|im_start|>"]
    config.eos_token_id = [specials["<|im_end|>"], specials["<|endoftext|>"]]
    config.pad_token_id = specials["<|endoftext|>"]

    model = qf_arch.seeded_tiny_model(config)
    # `generation_config` is a SEPARATE object, it is the one `generate` reads,
    # and `save_pretrained` writes it back verbatim.  It is given an id the
    # model config does NOT carry, so a fold that remaps only the config cannot
    # pass by accident: `composed` is above the C1 keep size, so leaving it
    # alone leaves a generation config naming a row the child does not have.
    model.generation_config.bos_token_id = specials["<|im_start|>"]
    model.generation_config.eos_token_id = [
        specials["<|im_end|>"], specials["<|endoftext|>"]
    ]
    model.generation_config.pad_token_id = composed
    model.save_pretrained(str(directory), safe_serialization=True)
    manifest = qf_corpus.synthetic_manifest(
        vocab_size=int(config.vocab_size),
        sequence_length=SEQUENCE,
        batches=BATCHES,
        held_every=HELD_EVERY,
        # The fixture's evidence names the fixture's OWN tokenizer, so the
        # consumption-time tokenizer gate in qf_build is exercised by every build
        # this suite runs instead of being escaped by a synthetic label.
        tokenizer_sha256=qf_corpus.file_sha256(directory / "tokenizer.json"),
        # The stream is drawn from the byte alphabet only, so it is spellable by
        # every coreset C1 can produce and the vocabulary test measures the fold
        # rather than a missing re-tokenization.
        max_id=256,
        out=WORK / f"{tag}-evidence.json",
    )
    return directory, manifest


def fresh_harness(parent: Path, manifest: dict[str, Any]) -> Harness:
    return Harness(parent, "cpu", torch.float32, manifest)


def runs(harness: Harness) -> torch.Tensor:
    return harness.logits(harness.batches[0][1])


# --------------------------------------------------------------------- tests
@test
def test_architecture_map(state):
    """§0.1 of the port spec, asserted against the live modules."""

    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    assert type(harness.model).__name__ == "Qwen3_5ForCausalLM"
    assert len(harness.layers()) == int(tc.num_hidden_layers)

    kinds = [qf_arch.layer_kind(layer) for layer in harness.layers()]
    assert kinds == list(tc.layer_types), (kinds, list(tc.layer_types))
    assert qf_arch.FULL in kinds and qf_arch.LINEAR in kinds, kinds
    qf_arch.assert_layer_types_agree(harness.model)

    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)

    # The two gain conventions, checked by running the modules rather than by
    # reading their source: a plain RMSNorm at weight w must scale by 1+w.
    norm = harness.final_norm()
    x = torch.randn(1, 1, int(tc.hidden_size))
    produced = norm(x)
    manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    manual = manual * qf_arch.effective_gain(norm)
    assert torch.allclose(produced, manual, atol=1e-6), (produced[0, 0, :4], manual[0, 0, :4])
    assert not torch.allclose(
        qf_arch.effective_gain(norm), norm.weight.data
    ), "the fixture's norms are at zero; the 1+w convention is untestable"

    gdn = next(l.linear_attn for l in harness.layers() if qf_arch.layer_kind(l) == qf_arch.LINEAR)
    assert qf_arch.is_gated_norm(gdn.norm)
    assert torch.equal(qf_arch.effective_gain(gdn.norm), gdn.norm.weight.data)

    layout = qf_arch.plane_layout(int(tc.head_dim), qf_arch.rotary_dim_of(harness.model.config))
    assert len(layout["lo"]) == qf_arch.rotary_dim_of(harness.model.config) // 2
    assert len(layout["nope"]) == int(tc.head_dim) - qf_arch.rotary_dim_of(harness.model.config)
    return {
        "class": type(harness.model).__name__,
        "layer_types": kinds,
        "parameters_live": counts["live"],
        "parameters_closed_form": counts["total"],
        "plane_layout": {
            "rotary_planes": len(layout["lo"]),
            "nope_coordinates": len(layout["nope"]),
        },
    }


@test
def test_evidence_seal(state):
    """A manifest that names its own digest is re-sealed, and a tampered one refuses."""

    manifest = state["manifest"]
    path = WORK / "parent-evidence.json"
    reread = qf_corpus.read_manifest(path)
    assert reread["manifest_sha256"] == manifest["manifest_sha256"]

    tampered = WORK / "tampered-evidence.json"
    body = json.loads(path.read_text())
    body["rows"][0]["ids"][0] = (body["rows"][0]["ids"][0] + 1) % 400
    tampered.write_text(json.dumps(body))
    caught = False
    try:
        qf_corpus.read_manifest(tampered)
    except SystemExit as error:
        caught = "seal is broken" in str(error)
    assert caught, "a tampered evidence manifest was accepted"

    mask = qf_corpus.fit_check_rows(manifest)
    fit_batches = sum(1 for row in manifest["rows"] if row["split"] == "fit")
    assert mask.numel() == fit_batches * SEQUENCE
    assert 0 < int(mask.sum()) < mask.numel()
    return {
        "batches": len(manifest["rows"]),
        "fit_batches": fit_batches,
        "internal_check_rows": int(mask.sum()),
        "tamper_refused": caught,
    }


@test
def test_harness_parity(state):
    harness = fresh_harness(state["parent"], state["manifest"])
    report = harness_parity(harness, sites=(0, 1, 3))
    assert report["deterministic"]
    assert report["logits_bit_identical_under_hooks"]
    assert report["hook_placement_worst"] < 1e-5, report["hook_placement_relL2"]
    return report


@test
def test_causal_mask_assertion(state):
    """Positive, repaired and NEGATIVE control for the mask-dispatch trap."""

    parent, manifest = state["parent"], state["manifest"]

    clean = fresh_harness(parent, manifest)
    on_parent = qf_excise.assert_causal_masks(clean)
    assert all(
        row.get("above_diagonal_mass", 0.0) == 0.0
        for row in on_parent.values()
        if row["kind"] == qf_arch.FULL
    )

    repaired = fresh_harness(parent, manifest)
    kinds = [qf_arch.layer_kind(l) for l in repaired.layers()]
    full_position = kinds.index(qf_arch.FULL)
    linear_position = kinds.index(qf_arch.LINEAR)
    qf_excise.excise_live(repaired.model, [linear_position])
    after = qf_excise.assert_causal_masks(repaired)
    assert all(
        row.get("above_diagonal_mass", 0.0) == 0.0
        for row in after.values()
        if row["kind"] == qf_arch.FULL
    )

    # Negative control: delete a layer BEFORE a full-attention layer and leave
    # the config alone, so an attention module lands in a slot the config calls
    # linear and is handed `linear_attn_mask` (None) instead of the causal mask.
    broken = fresh_harness(parent, manifest)
    victim = next(
        i for i in range(full_position)
        if qf_arch.layer_kind(broken.layers()[i]) == qf_arch.LINEAR
    )
    qf_excise.unrepaired_excision(broken.model, victim)
    caught = ""
    try:
        qf_excise.assert_causal_masks(broken)
    except SystemExit as error:
        caught = str(error)
    assert "attends to the future" in caught or "was handed a mask" in caught, (
        f"the negative control did not fire: {caught!r}"
    )
    leak = qf_excise.above_diagonal_mass(broken)
    assert max(leak.values()) > 0.0, leak
    return {
        "parent_above_diagonal": 0.0,
        "repaired_excision_above_diagonal": 0.0,
        "unrepaired_excision_leak": max(leak.values()),
        "refusal": caught.splitlines()[-1].strip(),
    }


@test
def test_solve_laws(state):
    """The algebra's own identities, before any of it is trusted on a model."""

    torch.manual_seed(7)
    A = torch.randn(400, 24, dtype=torch.float64)
    W = torch.randn(9, 24, dtype=torch.float64)
    Y = A @ W.T + 0.01 * torch.randn(400, 9, dtype=torch.float64)
    prior = W + 0.05 * torch.randn_like(W)

    rows, guard = qf_solve.law_abiding_solve(A, Y, prior)
    gram, guard_gram = qf_solve.law_abiding_solve_gram(A.T @ A, A.T @ Y, prior)
    assert torch.allclose(rows, gram, atol=1e-9), float((rows - gram).abs().max())
    assert guard["update_scale"] <= 1.0

    # THE LAW refuses an impossible move rather than installing it.
    refused = False
    try:
        qf_solve.law_abiding_solve(A, Y * 1e9, torch.full_like(prior, 1e-12))
    except SystemExit:
        refused = True
    assert refused, "the lambda ladder admitted an unbounded update"

    # The Gram walk must give the ROW form's answer at the lambda it selected;
    # `law_abiding_solve` and `law_abiding_solve_split` now form `A^T A` once
    # and walk the ladder in the Gram, and this is what pins that refactor to
    # the `prior_anchored` it replaced instead of to itself.
    assert torch.allclose(
        rows, qf_solve.prior_anchored(A, Y, prior, rel=guard["lambda_relative"]), atol=1e-9
    ), float((rows - qf_solve.prior_anchored(A, Y, prior, rel=guard["lambda_relative"])).abs().max())
    check_mask = torch.zeros(A.shape[0], dtype=torch.bool)
    check_mask[::4] = True
    split, split_guard = qf_solve.law_abiding_solve_split(A, Y, check_mask, prior)
    split_rowform = qf_solve.prior_anchored(A, Y, prior, rel=split_guard["lambda_relative"])
    assert torch.allclose(split, split_rowform, atol=1e-9), float(
        (split - split_rowform).abs().max()
    )

    F = torch.randn(300, 32, dtype=torch.float64)
    assert torch.equal(qf_solve.pivoted_columns(F, 8), qf_solve.pivoted_columns_gram(F.T @ F, 8))
    assert torch.equal(qf_solve.pivoted_columns(F, 64), torch.arange(32))

    # ---- the coreset's SCALE degeneracy.  The greedy pivot is a property of
    # the Gram's directions and must not depend on its magnitude.  The absolute
    # `best + 1e-15` tie epsilon this routine used to carry made a small-scale
    # site (diag ~3e-20 here) fail every comparison after the first pivot and
    # return 0..k-1 IN INDEX ORDER -- a selection that is not a selection, with
    # no error and nothing in the report to reveal it.
    picks = qf_solve.pivoted_columns(F, 8)
    tiny_report: dict = {}
    tiny_picks = qf_solve.pivoted_columns(F * 1e-10, 8, report=tiny_report)
    assert torch.equal(picks, tiny_picks), (picks.tolist(), tiny_picks.tolist())
    assert not torch.equal(picks, torch.arange(8)), (
        "index order IS the fixture's answer, so the degeneracy control proves nothing"
    )
    assert tiny_report["stopped"] == "completed", tiny_report
    assert 0.0 < tiny_report["residual_diagonal_fraction"] < 1.0, tiny_report

    # whitened_lra at full rank keeps everything.
    stack = torch.randn(11, 16, dtype=torch.float64)
    G = torch.randn(200, 16, dtype=torch.float64)
    P, lra = qf_solve.whitened_lra(stack, (G.T @ G) / 200, 16)
    assert tuple(P.shape) == (16, 16), (
        "a rank-deficient writer must NOT silently shrink the projection"
    )
    assert abs(lra["retained_energy"] - 1.0) < 1e-9
    assert torch.allclose(qf_solve.project_writer(stack, P) @ P, stack, atol=1e-8)
    narrow, narrow_report = qf_solve.whitened_lra(stack, (G.T @ G) / 200, 4)
    assert tuple(narrow.shape) == (4, 16)
    assert narrow_report["retained_energy"] < 1.0

    # ---- whitening DIRECTION, on a deliberately ANISOTROPIC gram.
    # `G = randn(200, 16)` above has a near-isotropic gram, where G^{1/2} and
    # G^{-1/2} are both ~cI and the two directions pick the SAME subspace -- so
    # that fixture cannot see the difference and did not.  Here the writer reads
    # both coordinates and the evidence populates only the first.  The correct
    # objective is the truncated SVD of `stack G^{1/2}`, which keeps coordinate
    # 0; feeding `stack G^{-1/2}` keeps coordinate 1, the one the evidence never
    # populates, and the value-space error jumps from Var(v_1) = eps to Var(v_0).
    eps = 1e-6
    thin_gram = torch.diag(torch.tensor([1.0, eps], dtype=torch.float64))
    thin_stack = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    thin_P, _ = qf_solve.whitened_lra(thin_stack, thin_gram, 1)
    assert tuple(thin_P.shape) == (1, 2)
    assert int(thin_P.abs().argmax()) == 0, (
        f"whitened_lra kept the coordinate the evidence never populates: {thin_P}"
    )

    def value_sse(P: torch.Tensor) -> float:
        """``E_v || stack v - (stack P^+) P v ||^2`` under ``thin_gram``."""

        residual = thin_stack - qf_solve.project_writer(thin_stack, P) @ P
        return float(torch.trace(residual @ thin_gram @ residual.T))

    root, inverse_root = qf_solve.matrix_sqrt(thin_gram)

    def projection(whitened: torch.Tensor) -> torch.Tensor:
        Vh = torch.linalg.svd(whitened, full_matrices=True)[2]
        Q, _ = torch.linalg.qr((Vh[:1] @ inverse_root).T)
        return Q[:, :1].T.contiguous()

    transposed_P = projection(thin_stack @ inverse_root)
    installed_sse, transposed_sse = value_sse(thin_P), value_sse(transposed_P)
    assert installed_sse <= eps * 1.001, installed_sse
    assert transposed_sse > 0.5, transposed_sse
    assert installed_sse < transposed_sse

    # A "merge" of ONE key head is the identity: relation error must be zero.
    layout = qf_arch.plane_layout(16, 8)
    q = [torch.randn(16, 20, dtype=torch.float64) for _ in range(3)]
    k = [torch.randn(16, 20, dtype=torch.float64)]
    k_hat, q_hats, relation = qf_solve.joint_relation_solve(q, k, [0, 0, 0], [1.0, 1.0, 1.0], layout)
    assert relation["relation_error_max"] < 1e-9, relation["relation_error_per_slot"]
    # TWO parents: the case where the query re-fit coefficient is a non-trivial
    # COMPLEX number on the rotary planes, and therefore the only case in which
    # `gamma` and `conj(gamma)` differ at all.  A bound like "0 < error < 2" says
    # nothing about which of the two was installed, so the installed choice is
    # compared against the conjugate and against the do-nothing gamma = 1 here.
    k_two = [k[0], torch.randn(16, 20, dtype=torch.float64)]
    group = [0, 0, 1]
    k_hat_two, _, two_report = qf_solve.joint_relation_solve(
        q, k_two, group, [1.0, 1.0, 1.0], layout
    )
    two = (None, None, two_report)
    assert 0.0 < two_report["relation_error_mean"] < 2.0
    k_hat_planes = qf_solve.as_planes(k_hat_two, layout)
    installed_errors = two_report["relation_error_per_slot"]
    conjugated_errors: list[float] = []
    unit_errors: list[float] = []
    for slot, slot_rows in enumerate(q):
        phi = qf_solve.as_planes(slot_rows, layout)
        kappa = qf_solve.as_planes(k_two[group[slot]], layout)
        conjugated = torch.zeros_like(phi)
        for plane in range(phi.shape[0]):
            gamma = (kappa[plane].conj() @ k_hat_planes[plane]) / max(
                float(k_hat_planes[plane].norm().pow(2)), 1e-30
            )
            conjugated[plane] = phi[plane] * gamma.conj()
        conjugated_errors.append(
            qf_solve.rank_one_relation_error(phi, kappa, conjugated, k_hat_planes)
        )
        unit_errors.append(
            qf_solve.rank_one_relation_error(phi, kappa, phi, k_hat_planes)
        )
    for slot in range(len(q)):
        assert installed_errors[slot] <= conjugated_errors[slot] + 1e-12, (
            f"slot {slot}: the CONJUGATE of the installed coefficient fits better "
            f"({conjugated_errors[slot]:.6f} < {installed_errors[slot]:.6f}), which "
            f"means the conjugate is what should have been installed"
        )
        assert installed_errors[slot] <= unit_errors[slot] + 1e-12, slot
    assert max(conjugated_errors) > max(installed_errors) * 1.02, (
        "the fixture has no rotary phase for the conjugate to get wrong; this "
        "comparison is vacuous as written"
    )

    round_trip = qf_solve.from_planes(qf_solve.as_planes(q[0], layout), layout)
    assert torch.allclose(round_trip, q[0], atol=1e-12)

    # The gated value CG re-fit: closed-form operator, no gradients.  Solved
    # against a target its own forward generated, so the objective must fall.
    X = torch.randn(150, 12, dtype=torch.float64)
    readers = [torch.randn(6, 12, dtype=torch.float64) for _ in range(2)]
    writers = [torch.randn(9, 5, dtype=torch.float64) for _ in range(2)]
    gates = [torch.sigmoid(torch.randn(150, 5, dtype=torch.float64)) for _ in range(2)]
    truth = torch.randn(5, 6, dtype=torch.float64) * 0.3
    target = sum(
        (g * ((X @ A.T) @ truth.T)) @ W.T for W, g, A in zip(writers, gates, readers)
    )
    prior = truth + 0.1 * torch.randn_like(truth)
    solved, cg = qf_solve.gated_value_cg_refit(
        X, writers, gates, readers, target, prior
    )
    assert cg["cg_refit"] == "installed", cg
    assert cg["objective_exit"] < cg["objective_entry"], cg
    # The claim is about the OBJECTIVE, not about recovering `truth`: the
    # operator is rank-deficient in directions the evidence does not populate,
    # so a solve that fits better can sit further from the generating weight and
    # that is correct behaviour, not a failure.
    assert cg["update_scale"] <= 1.0, cg
    return {
        "gram_equals_rows_max_abs": float((rows - gram).abs().max()),
        "law_refuses_unbounded_update": refused,
        "one_parent_relation_error": relation["relation_error_max"],
        "two_parent_relation_error": two[2]["relation_error_mean"],
        "two_parent_installed_vs_conjugate_vs_unit": {
            "installed": installed_errors,
            "conjugated_coefficient": conjugated_errors,
            "gamma_equals_one": unit_errors,
        },
        "whitened_lra_direction": {
            "installed_value_sse": installed_sse,
            "transposed_whitening_value_sse": transposed_sse,
            "anisotropic_gram": [1.0, eps],
        },
        "gram_ladder_equals_row_form": True,
        "coreset_scale_invariant_selection": [int(i) for i in picks],
        "coreset_residual_diagonal_fraction_at_1e-10": tiny_report[
            "residual_diagonal_fraction"
        ],
        "plane_round_trip_exact": True,
        "gated_cg_objective": [cg["objective_entry"], cg["objective_exit"]],
    }


@test
def test_gram_refusal_and_eigen_fallback(state):
    """What a degenerate Gram does now, instead of one LAPACK sentence.

    The K2KV build died at full-attention layer 27 in
    ``prior_anchored_gram -> torch.linalg.cholesky`` with "the leading minor of
    order 1 is not positive-definite" and nothing else -- no channel, no reason,
    and no way to tell a dead channel from a poisoned one without re-running a
    hundred minutes of fold to look.  This unit pins the four cases apart.

    The first assertion is the load-bearing one and it is a NEGATIVE result: a
    design matrix with an identically DEAD channel still factorises, because
    ``A^T A`` is positive semi-definite by construction and ``lam = rel *
    mean(diag)`` is strictly positive whenever any other channel carries energy.
    So a zero column CANNOT produce this error on its own -- which leaves only
    an all-zero stream (``lam`` is then zero too) or a NON-FINITE one, and the
    non-finite case is a defect in the captured ROWS, not in the algebra.
    """

    torch.manual_seed(20260903)
    columns, rows_n = 12, 240
    A = torch.randn(rows_n, columns, dtype=torch.float64)
    W = torch.randn(4, columns, dtype=torch.float64)
    Y = A @ W.T
    prior = W + 0.05 * torch.randn_like(W)
    check = torch.zeros(rows_n, dtype=torch.bool)
    check[::4] = True

    # ---- 0.  the happy path still says so, in the receipt.
    healthy, healthy_guard = qf_solve.law_abiding_solve_split(A, Y, check, prior)
    assert healthy_guard["solver"] == "cholesky", healthy_guard["solver"]
    assert healthy_guard["solver_fallback"] == [], healthy_guard["solver_fallback"]
    assert torch.isfinite(healthy).all()

    # ---- 1.  a DEAD leading channel is not this failure.  Column 0 identically
    # zero, everything else untouched: the Gram is singular, the ridge is not,
    # and the solve goes through -- so "an all-zero channel" alone was never a
    # sufficient explanation for the layer-27 crash.
    dead = A.clone()
    dead[:, 0] = 0.0
    dead_solved, dead_guard = qf_solve.law_abiding_solve_split(dead, Y, check, prior)
    assert dead_guard["solver"] == "cholesky", dead_guard["solver"]
    assert torch.isfinite(dead_solved).all()
    dead_gram_00 = float((dead.T @ dead)[0, 0])
    assert dead_gram_00 == 0.0, dead_gram_00

    # ---- 2.  a NaN in the leading channel.  One bad value in one row poisons
    # every entry of A^T A, so the refusal must say ROW, name the channels, and
    # be a GramNotPositiveDefinite -- not the bare _LinAlgError the build died
    # on, and not a number.
    poisoned = A.clone()
    poisoned[3, 0] = float("nan")
    nan_refusal = None
    try:
        qf_solve.law_abiding_solve_split(poisoned, Y, check, prior)
    except qf_solve.GramNotPositiveDefinite as refusal:
        nan_refusal = refusal
    assert nan_refusal is not None, "a NaN design matrix was solved rather than refused"
    text = str(nan_refusal)
    assert "non-finite Gram entries" in text, text
    assert "fix the capture" in text, text
    assert "channels_with_any_nonfinite_entry" not in text  # a message, not a dump
    # One NaN in channel 0 puts a NaN in row 0 and column 0 of the Gram -- that
    # is 2*columns - 1 entries -- and then the DIAGONAL MEAN is NaN, so the ridge
    # is NaN and every entry of G + lam I is NaN.  That last step is why the
    # LAPACK complaint was about "the leading minor of order 1": channel 0 is
    # simply where a matrix of NaNs is looked at first.
    assert nan_refusal.report["gram_nonfinite_entries"] == 2 * columns - 1, (
        nan_refusal.report["gram_nonfinite_entries"]
    )
    assert nan_refusal.report["channels_with_nonfinite_diagonal"] == 1
    assert nan_refusal.report["channels_with_nonfinite_diagonal_first"] == [0]
    assert nan_refusal.report["channels_with_any_nonfinite_entry"] == columns
    assert nan_refusal.report["channels_with_any_nonfinite_entry_first"][0] == 0
    assert nan_refusal.report["ridge_is_finite"] is False, nan_refusal.report
    assert "the ridge itself is not finite" in text, text
    # The refusal that reaches the caller must be the CHOLESKY one -- the ridge
    # it saw, the sentence LAPACK printed -- with the fallback's reason appended,
    # not the fallback's refusal in its place.  Replacing it loses the NaN ridge,
    # which is the single fact that explains "leading minor of order 1".
    assert "the eigen fallback refused as well" in text, text
    assert "eigen_anchored_gram" in nan_refusal.report["eigen_fallback_refusal"]
    assert nan_refusal.report["gram"] == "inner-split gram", nan_refusal.report["gram"]

    # The eigen fallback must REFUSE this one too rather than invent a number:
    # eigh of a NaN is a NaN, and a receipt carrying it would be a fabrication.
    eigen_refused = False
    try:
        qf_solve.eigen_anchored_gram(poisoned.T @ poisoned, poisoned.T @ Y, prior)
    except qf_solve.GramNotPositiveDefinite:
        eigen_refused = True
    assert eigen_refused, "the eigen fallback answered a non-finite Gram"

    # ---- 3.  an all-zero stream: lam is zero as well, so the ridged Gram is
    # the zero matrix and channel 0 IS the leading minor the message names.
    zero_refusal = None
    try:
        qf_solve.law_abiding_solve_split(
            torch.zeros_like(A), torch.zeros_like(Y), check, prior
        )
    except qf_solve.GramNotPositiveDefinite as refusal:
        zero_refusal = refusal
    assert zero_refusal is not None, "an identically zero stream was solved"
    assert zero_refusal.report["channels_with_nonpositive_diagonal"] == columns
    assert zero_refusal.report["channels_with_nonpositive_diagonal_first"][0] == 0
    assert "non-positive diagonal" in str(zero_refusal), str(zero_refusal)

    # ---- 4.  the RECORDED fallback.  `law_abiding_solve_gram` is handed a Gram
    # it did not form (the 27B path accumulates them), so it is the entry point
    # where a numerically indefinite-but-finite matrix can actually arrive.  The
    # eigen solve answers it, and the answer arrives with the floor it used.
    #
    # Built in its own eigenbasis so every number in the fixture is chosen, not
    # hoped for: one eigenvalue is negative and the spectrum SUMS TO ZERO, so
    # ``lam = rel * mean(diag)`` is zero on every rung of the ladder and no rung
    # can lift the matrix into a Cholesky.  The right-hand side is given no
    # component along the offending eigenvector, which is what makes the floored
    # inverse a sane answer rather than a 1e12 blow-up -- and the assertions
    # below check that the answer was in fact taken from the fallback.
    basis = torch.linalg.qr(torch.randn(columns, columns, dtype=torch.float64))[0]
    spectrum = torch.ones(columns, dtype=torch.float64)
    spectrum[0] = -float(columns - 1)                    # trace, hence lam, is 0
    indefinite = basis @ torch.diag(spectrum) @ basis.T
    indefinite = 0.5 * (indefinite + indefinite.T)
    assert abs(float(torch.diagonal(indefinite).mean())) < 1e-12, float(
        torch.diagonal(indefinite).mean()
    )
    positive_part = basis[:, 1:] @ basis[:, 1:].T                 # spectrum without q0
    B = positive_part @ W.T
    cholesky_refused = False
    try:
        torch.linalg.cholesky(indefinite)
    except Exception:  # noqa: BLE001
        cholesky_refused = True
    assert cholesky_refused, "the indefinite fixture is factorisable; it proves nothing"
    fallback_solved, fallback_guard = qf_solve.law_abiding_solve_gram(
        indefinite, B, prior
    )
    assert torch.isfinite(fallback_solved).all()
    assert fallback_guard["solver"] == "cholesky + eigen fallback", fallback_guard["solver"]
    recorded = fallback_guard["solver_fallback"]
    assert recorded, "the fallback fired but recorded nothing"
    first = recorded[0]
    assert first["solver_fallback"].startswith("eigen-decomposition"), first
    assert first["eigen_floor_relative"] == qf_solve.EIGEN_FLOOR_REL, first
    assert first["eigen_floor_absolute"] > 0.0, first
    assert first["eigenvalues_lifted_to_the_floor"] >= 1, first
    assert first["eigenvalue_min_before_floor"] < 0.0, first
    assert "refusal" in first and "will not factorise" in first["refusal"], first

    # A Gram with no positive eigenvalue is refused by the fallback as well: a
    # stream that says nothing in any direction has no solve to fall back to.
    dead_everywhere = False
    try:
        qf_solve.eigen_anchored_gram(
            torch.zeros(columns, columns, dtype=torch.float64),
            torch.zeros(columns, prior.shape[0], dtype=torch.float64),
            prior,
        )
    except qf_solve.GramNotPositiveDefinite as refusal:
        dead_everywhere = "no positive eigenvalue" in str(refusal)
    assert dead_everywhere, "the fallback answered a Gram with no energy at all"

    return {
        "healthy_solver": healthy_guard["solver"],
        "healthy_solver_fallback": healthy_guard["solver_fallback"],
        "dead_leading_channel_still_factorises": {
            "gram_00": dead_gram_00,
            "solver": dead_guard["solver"],
            "means": "a zero column alone cannot raise the layer-27 error",
        },
        "nan_leading_channel": {
            "exception": type(nan_refusal).__name__,
            "gram_nonfinite_entries": nan_refusal.report["gram_nonfinite_entries"],
            "channels_poisoned": nan_refusal.report["channels_with_any_nonfinite_entry"],
            "eigen_fallback_also_refused": eigen_refused,
            "message": str(nan_refusal)[:400],
        },
        "zero_stream": {
            "channels_with_nonpositive_diagonal": zero_refusal.report[
                "channels_with_nonpositive_diagonal"
            ],
            "message": str(zero_refusal)[:300],
        },
        "eigen_fallback": {
            "solver": fallback_guard["solver"],
            "eigen_floor_relative": first["eigen_floor_relative"],
            "eigen_floor_absolute": first["eigen_floor_absolute"],
            "eigenvalues_lifted": first["eigenvalues_lifted_to_the_floor"],
            "eigenvalue_min_before_floor": first["eigenvalue_min_before_floor"],
            "update_scale": fallback_guard["update_scale"],
        },
    }


@test
def test_export_and_verify(state):
    """Export an UNFOLDED child, reload it through the stock class, carry MTP."""

    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    out = WORK / "child-identity"
    if out.exists():
        shutil.rmtree(out)
    export = qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, reloaded = qf_export.verify_child(out)
    assert verify["child_tensors_sha256"] == export["child_tensors_sha256"]
    assert verify["trust_remote_code"] is False
    assert verify["parameters_counted_live"] == verify["parameters_closed_form"]

    before = harness.logits(harness.batches[0][1])
    after = Harness.wrap(reloaded, "cpu", torch.float32, manifest).logits(harness.batches[0][1])
    assert torch.equal(before, after), float((before - after).abs().max())

    # Carry-through: a tensor transformers refuses to own must survive export.
    from safetensors.torch import save_file

    fake_parent = WORK / "parent-with-mtp"
    if fake_parent.exists():
        shutil.rmtree(fake_parent)
    shutil.copytree(parent, fake_parent)
    save_file(
        {"mtp.fc.weight": torch.randn(4, 4), "mtp.norm.weight": torch.randn(4)},
        str(fake_parent / "model-mtp.safetensors"),
        metadata={"format": "pt"},
    )
    carried = qf_export.carry_through(fake_parent, out)
    assert carried["carried"] == ["mtp.fc.weight", "mtp.norm.weight"], carried
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.fc.weight"] == "model-carried.safetensors"
    verify_again, _ = qf_export.verify_child(out)
    assert verify_again["parameters_counted_live"] == verify["parameters_counted_live"], (
        "the carried MTP tensors were loaded as model parameters; they must be "
        "ignored by the loader and counted separately"
    )
    return {
        "digest": export["child_tensors_sha256"][:16],
        "tensors_digested": len(export["tensor_sha256"]),
        "reload_bit_identical": True,
        "mtp_carried": carried["carried"],
    }


@test
def test_c1_vocab(state):
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    parent_vocab = int(harness.text_config.vocab_size)
    keep = 384
    report = qf_vocab.vocab_axis(
        harness, keep=keep, tokenizer_path=parent / "tokenizer.json", log=quiet
    )
    assert int(harness.text_config.vocab_size) == keep
    assert len(report["kept_ids"]) == keep
    assert report["parameters_removed"] == (parent_vocab - keep) * 128 * 2

    # ---- the id space, which is the half of C1 that does not show up in a
    # shape.  Every hard id must have MOVED (the fixture puts them all above the
    # keep size on purpose), the LIST-valued eos must be renumbered elementwise,
    # and the generation config -- a separate object, the one `generate` reads --
    # must be covered by both the protection and the remap.
    hard = report["hard_ids"]
    assert "generation_config.pad_token_id" in hard, sorted(hard)
    parent_eos = hard["text_config.eos_token_id"]["parent"]
    child_eos = hard["text_config.eos_token_id"]["child"]
    assert isinstance(parent_eos, list) and isinstance(child_eos, list)
    assert len(child_eos) == len(parent_eos) == 2
    stationary = [w for w, row in hard.items() if row["parent"] == row["child"]]
    assert not stationary, f"hard ids that were never renumbered: {stationary}"
    assert report["mandatory_breakdown"]["hard_id_rows"] == 1, (
        "the generation config's merge-composed pad id is the only hard id that "
        "is neither an added token nor a byte token; if it is not counted here "
        "the generation config never reached the mandatory set"
    )
    for value in report["generation_config_ids"].values():
        for index in (value if isinstance(value, list) else [value]):
            assert 0 <= index < keep, (value, keep)

    vocab = qf_vocab.Vocabulary(parent / "tokenizer.json")
    qf_vocab.merge_closure_intact(vocab, set(report["kept_ids"]))
    assert len(vocab.byte_alphabet()) == 256

    # NEGATIVE CONTROL for the byte alphabet.  The fixture is built so that
    # single-character == byte token, which is exactly the coincidence that let
    # "every len(token) == 1 row" pass for "the byte alphabet".  A real parent's
    # table has thousands of single-character rows that are not byte tokens, so
    # one is added here and must NOT be reserved; and a table missing one byte
    # token must be refused.
    body = json.loads((parent / "tokenizer.json").read_text(encoding="utf-8"))
    free_id = max(body["model"]["vocab"].values()) + 1
    body["model"]["vocab"]["中"] = free_id
    probe_path = WORK / "tokenizer-with-cjk.json"
    probe_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    probe = qf_vocab.Vocabulary(probe_path)
    assert len(probe.byte_alphabet()) == 256
    assert free_id not in probe.byte_alphabet(), "a CJK single-char row was called a byte"
    assert free_id not in qf_vocab.mandatory_ids(probe, set())

    broken = json.loads(json.dumps(body))
    del broken["model"]["vocab"][qf_vocab.bytes_to_unicode()[ord("A")]]
    broken_path = WORK / "tokenizer-missing-byte.json"
    broken_path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    missing_byte_refused = False
    try:
        qf_vocab.Vocabulary(broken_path).byte_alphabet()
    except SystemExit as error:
        missing_byte_refused = "ByteLevel byte tokens" in str(error)
    assert missing_byte_refused, "a table that cannot spell byte 0x41 was accepted"
    for entry in report["tokenizer_body"]["model"]["merges"]:
        left, right = qf_vocab.Vocabulary._pair(entry)
        table = report["tokenizer_body"]["model"]["vocab"]
        assert left in table and right in table and (left + right) in table

    # The kept rows must be the parent's rows, byte for byte.  The reference is
    # re-seeded from the PARENT'S OWN config on disk and not from
    # `tiny_text_config()`: the fixture sets `pad_token_id`, that becomes the
    # embedding's `padding_idx`, and `_init_weights` ZEROES that row -- so a
    # reference built from the default config differs from the parent in exactly
    # one row and the comparison fails for a reason that has nothing to do with
    # the fold.
    reference = qf_arch.seeded_tiny_model(qf_arch.load_config(parent))
    kept = torch.tensor(report["kept_ids"])
    assert torch.equal(
        qf_arch.embed(harness.model).weight.data,
        reference.model.embed_tokens.weight.data.index_select(0, kept),
    )

    child_manifest = qf_vocab.remap_manifest(
        manifest, report["kept_ids"], WORK / "child-c1-evidence.json"
    )
    child_harness = Harness.wrap(harness.model, "cpu", torch.float32, child_manifest)
    logits = child_harness.logits(child_harness.batches[0][1])
    assert logits.shape[-1] == keep
    refused = False
    try:
        qf_vocab.remap_manifest(
            manifest, report["kept_ids"][:8], WORK / "should-not-exist.json"
        )
    except SystemExit as error:
        refused = "re-tokenize the source text" in str(error)
    assert refused, "remap_manifest renamed a stream the child cannot spell"
    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)

    out = WORK / "child-c1"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    written = qf_vocab.write_tokenizer(report, out, parent)
    verify, _ = qf_export.verify_child(out)
    child_vocab = qf_vocab.Vocabulary(out / "tokenizer.json")
    assert len(child_vocab.vocab) == keep
    assert max(child_vocab.vocab.values()) == keep - 1

    # The generation config ON DISK, which is what a downstream `generate` reads
    # and what `qf_export.SIDECARS` would otherwise have copied from the parent.
    assert "generation_config.json" in written
    child_generation = json.loads(
        (out / "generation_config.json").read_text(encoding="utf-8")
    )
    parent_generation = json.loads(
        (parent / "generation_config.json").read_text(encoding="utf-8")
    )
    assert child_generation["eos_token_id"] != parent_generation["eos_token_id"], (
        "the child's generation_config.json still names the parent's eos ids"
    )
    assert qf_vocab.assert_ids_in_mapping(child_generation, keep, where="child") >= 3

    # NEGATIVE CONTROL for the post-condition: put ONE parent-space id back and
    # the check must fire.  Without this the whole id-space block above could be
    # a walk that visits nothing.
    tampered = json.loads(json.dumps(child_generation))
    tampered["eos_token_id"] = parent_generation["eos_token_id"]
    leak_refused = False
    try:
        qf_vocab.assert_ids_in_mapping(tampered, keep, where="negative control")
    except SystemExit as error:
        leak_refused = "the id-space remap did not reach them" in str(error)
    assert leak_refused, "a parent-space eos id survived the post-condition"

    return {
        "parent_vocab": parent_vocab,
        "child_vocab": keep,
        "fold": parent_vocab / keep,
        "hard_ids": hard,
        "generation_config_on_disk": {
            "parent": {k: v for k, v in parent_generation.items()
                       if k.endswith("_token_id")},
            "child": {k: v for k, v in child_generation.items()
                      if k.endswith("_token_id")},
        },
        "parent_space_id_refused": leak_refused,
        "ids_addressable": report["ids_addressable"],
        "mandatory_rows": report["mandatory_rows"],
        "mandatory_breakdown": report["mandatory_breakdown"],
        "byte_alphabet_negative_control": {
            "cjk_single_char_not_reserved": True,
            "missing_byte_token_refused": missing_byte_refused,
        },
        "merges": [report["merges_parent"], report["merges_child"]],
        "sidecars": written,
        "stream_remap_refuses_unspellable": refused,
        "parameters": counts["live"],
        "reload_class": verify["loaded_class"],
    }


@test
def test_c2_mlp(state):
    """The MLP coreset at 4x, plus its identity gate at no reduction."""

    parent, manifest = state["parent"], state["manifest"]

    gate_harness = fresh_harness(parent, manifest)
    width = int(gate_harness.text_config.intermediate_size)
    identity = qf_mlp.narrow_mlp_site(gate_harness, 0, width, lambda_split=False)
    # The model runs in fp32 and the recompute is fp64, so "exact" here is the
    # fp32 module's own rounding, not zero.
    assert identity["scores"]["fit"]["relL2"] < 1e-5, identity["scores"]
    assert identity["selected_channels"] == list(range(width))
    assert identity["solve"]["update_scale"] < 1e-4, identity["solve"]

    harness = fresh_harness(parent, manifest)
    keep = width // 4
    calls_before = harness.forward_calls
    report = qf_mlp.mlp_axis(harness, keep, group_size=3, log=quiet)
    assert report["config_standard"]
    assert int(harness.text_config.intermediate_size) == keep
    assert abs(report["component_fold"] - 4.0) < 1e-9

    # The group schedule is a schedule, not a docstring: ONE capture per group,
    # shared by its sites.  8 sites in groups of 3 is 3 corpus passes, and a
    # corpus pass is len(batches) forward calls -- so the count is checkable
    # rather than asserted.
    schedule = report["schedule"]
    assert schedule["group_size"] == 3
    assert schedule["groups"] == [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert schedule["corpus_passes"] == 3
    assert all(site["shared_group_capture"] for site in report["per_site"])
    captures = (harness.forward_calls - calls_before) / len(harness.batches)
    assert captures == schedule["corpus_passes"], (captures, schedule)
    worst = max(site["scores"]["fit"]["relL2"] for site in report["per_site"])
    repaired_better = sum(
        1 for site in report["per_site"]
        if site["scores"]["fit"]["relL2"] < site["no_repair_scores"]["relL2"]
    )
    assert repaired_better == len(report["per_site"]), (
        f"the ridge refit failed to beat the no-repair drop at "
        f"{len(report['per_site']) - repaired_better} sites"
    )
    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    runs(harness)

    out = WORK / "child-c2"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, _ = qf_export.verify_child(out)
    return {
        "identity_gate_relL2": identity["scores"]["fit"]["relL2"],
        "parent_intermediate": width,
        "child_intermediate": keep,
        "component_fold": report["component_fold"],
        "schedule": {
            "group_size": schedule["group_size"],
            "groups": schedule["groups"],
            "corpus_passes": schedule["corpus_passes"],
            "measured_corpus_passes": captures,
        },
        "worst_site_fit_relL2": worst,
        "held_relL2": [
            site["scores"].get("held", {}).get("relL2") for site in report["per_site"][:3]
        ],
        "sites_where_repair_beat_drop": f"{repaired_better}/{len(report['per_site'])}",
        "parameters": counts["live"],
        "reload": verify["loaded_class"],
    }


@test
def test_c3_c4_excision(state):
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    parent_config = copy.deepcopy(harness.model.config)
    census = qf_excise.excision_census(harness)

    kinds = [qf_arch.layer_kind(l) for l in harness.layers()]
    victims = sorted({kinds.index(qf_arch.FULL), kinds.index(qf_arch.LINEAR)})
    report = qf_excise.excise_live(harness.model, victims)
    report["parameters_removed"] = qf_excise.parameters_removed(
        qf_arch.text_config(parent_config), report["excised_kinds"]
    )
    qf_excise.assert_causal_masks(harness)

    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    parent_counts = qf_arch.parameter_count(parent_config)
    assert parent_counts["total"] - counts["total"] == report["parameters_removed"], (
        parent_counts["total"] - counts["total"], report["parameters_removed"]
    )
    for position, layer in enumerate(harness.layers()):
        assert qf_arch.mixer(layer).layer_idx == position
    runs(harness)

    out = WORK / "child-c34"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, reloaded = qf_export.verify_child(out)
    assert int(qf_arch.text_config(reloaded.config).num_hidden_layers) == report["child_layers"]
    return {
        "cheapest_first": census["cheapest_first"],
        "excised": report["excised"],
        "kinds": report["excised_kinds"],
        "layers": [report["parent_layers"], report["child_layers"]],
        "parameters_removed": report["parameters_removed"],
        "closed_form_agrees": True,
        "child_layer_types": report["child_layer_types"],
    }


@test
def test_c5_gdn_head_count(state):
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    before = (int(tc.linear_num_value_heads), int(tc.linear_num_key_heads))
    census = qf_gdn.value_head_census(harness, [i for i, l in enumerate(harness.layers())
                                                if qf_arch.layer_kind(l) == qf_arch.LINEAR][0])
    report = qf_gdn.gdn_axis(
        harness, value_heads=before[0] // 2, key_heads=before[1] // 2, log=quiet
    )
    assert int(tc.linear_num_value_heads) == before[0] // 2
    assert int(tc.linear_num_key_heads) == before[1] // 2
    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    runs(harness)

    out = WORK / "child-c5count"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, _ = qf_export.verify_child(out)
    decay = [
        entry["decay_refit"]
        for layer in report["per_layer"] for entry in layer["per_child_head"]
    ]
    # The regression check for the `inverse_softplus` operator-precedence bug
    # that made every decay refit NaN.  It must count INSTALLED refits and
    # require STRICT improvement: on a refusal `_refit_decay` returns
    # `softplus_relL2 == anchor_only_relL2` (qf_gdn.py:381-391), so a `<=` over
    # every row scores a refusal as a win and an all-NaN solve reports a clean
    # sweep.  A silent all-refusal now fails this test instead of printing 12/12.
    installed = [row for row in decay if row["refit"] == "installed"]
    assert installed, (
        "every decay refit was refused; a NaN or degenerate solve looks exactly "
        "like this and the count-based check cannot tell them apart"
    )
    for row in installed:
        assert row["softplus_relL2"] < row["anchor_only_relL2"], row
        assert row["softplus_relL2"] == row["softplus_relL2"], row
    improved = len(installed)

    # ---- BETA.  The delta rule consumes `sigmoid(b.x)` (modeling 466, 516), so
    # a merged head whose row is the energy-weighted mean of its parents' LOGITS
    # is Jensen-biased, not fitted.  Same guard shape as the decay row: only
    # INSTALLED rows are counted and they must beat the weighted-logit row
    # strictly, so an all-refusal or an all-NaN solve cannot report a sweep.
    beta = [
        entry["beta_refit"]
        for layer in report["per_layer"] for entry in layer["per_child_head"]
    ]
    beta_installed = [row for row in beta if row["refit"] == "installed"]
    for row in beta_installed:
        assert row["beta_relL2"] < row["weighted_logit_relL2"], row
        assert row["beta_relL2"] == row["beta_relL2"], row

    # The NEGATIVE CONTROL for the bias itself, on a fixture built so the two
    # parents sit at logits -6 and +2 (betas 0.0025 and 0.881, mean 0.44): the
    # weighted-logit row lands at logit -2, beta 0.119 -- a 3.7x understatement
    # of the state write at every token.  This is the failure the fixture model
    # cannot show, because a random GDN's value-side error sits at ~1.0 either
    # way.
    torch.manual_seed(11)
    mu = torch.full((16,), 0.25, dtype=torch.float64)
    Xb = mu.unsqueeze(0) + 0.02 * torch.randn(512, 16, dtype=torch.float64)
    rows_b = torch.stack([-6.0 * mu, 2.0 * mu])
    equal = torch.tensor([0.5, 0.5], dtype=torch.float64)
    fitted_row, fitted = qf_gdn._refit_beta(
        qf_gdn._refit_grams(Xb), rows_b, [0, 1], equal
    )
    target_mean = float(
        0.5 * torch.sigmoid(Xb @ rows_b[0]).mean() + 0.5 * torch.sigmoid(Xb @ rows_b[1]).mean()
    )
    weighted_logit_mean = float(torch.sigmoid(Xb @ (0.5 * rows_b[0] + 0.5 * rows_b[1])).mean())
    fitted_mean = float(torch.sigmoid(Xb @ fitted_row).mean())
    assert weighted_logit_mean < 0.5 * target_mean, (
        "the fixture does not exhibit the Jensen bias, so it cannot prove the fix "
        f"(target {target_mean:.4f}, weighted logit {weighted_logit_mean:.4f})"
    )
    assert fitted["refit"] == "installed", fitted
    assert abs(fitted_mean - target_mean) < 0.1 * abs(weighted_logit_mean - target_mean), (
        fitted_mean, weighted_logit_mean, target_mean
    )

    return {
        "parent": before,
        "child": (int(tc.linear_num_value_heads), int(tc.linear_num_key_heads)),
        "component_ratio": report["component_ratio"],
        "fit_relL2": [layer["scores"]["fit"]["relL2"] for layer in report["per_layer"]],
        "decay_refit_installed_and_strictly_better": f"{improved}/{len(decay)}",
        "decay_refit_refused": len(decay) - improved,
        "beta_refit_installed_and_strictly_better": f"{len(beta_installed)}/{len(beta)}",
        "beta_relL2_worst": max(row["beta_relL2"] for row in beta),
        "beta_jensen_negative_control": {
            "parent_beta_means": [
                float(torch.sigmoid(Xb @ rows_b[0]).mean()),
                float(torch.sigmoid(Xb @ rows_b[1]).mean()),
            ],
            "target_mean_beta": target_mean,
            "weighted_logit_row_mean_beta": weighted_logit_mean,
            "understatement_factor": target_mean / max(weighted_logit_mean, 1e-30),
            "refit_row_mean_beta": fitted_mean,
            "refit_relL2": fitted["beta_relL2"],
            "weighted_logit_relL2": fitted["weighted_logit_relL2"],
        },
        "census_pairs": len(census["key_head_pairs"]),
        "parameters": counts["live"],
        "reload": verify["loaded_class"],
    }


@test
def test_c5_refit_gram_hoist(state):
    """Round-2 blocker: the value-head refits re-formed ``X^T X`` per head.

    Two claims, and the hoist is a refactor only if both hold.

    1. On a fixture where the parents disagree hard enough that both refits
       INSTALL, the shared-Gram form returns the row, the bias and every
       reported number BIT-IDENTICALLY equal to the row form it replaced.  The
       pre-fix bodies are transcribed verbatim below (qf_gdn.py at
       ``93331a05367628f3cd36e95b2cc64e60282d6c049d4776f938fa7adc3d1b5227``), so
       this compares against the shipped code and not against a paraphrase.
    2. A real fold of every GDN layer forms the Grams once per LAYER and calls
       the Gram solver twice per CHILD HEAD.  That is the only call graph in
       which the per-head re-formation is actually gone -- an equality on the
       numbers alone would still pass if the hoist had been wired up and then
       shadowed by a second formation inside the loop.
    """

    # ------------------------------------------------ the pre-fix bodies
    def old_refit_decay(X, a, dt, A_log, parent_heads, weights, anchor):
        target = torch.zeros(X.shape[0], dtype=torch.float64)
        for w, j in zip(weights, parent_heads):
            target += float(w) * float(A_log[j].exp()) * F.softplus(X @ a[j] + dt[j])
        target = target / float(A_log[anchor].exp())
        design = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)
        prior = torch.cat([a[anchor], dt[anchor].reshape(1)]).unsqueeze(0)
        solved = qf_solve.prior_anchored(
            design, qf_solve.inverse_softplus(target).unsqueeze(1), prior
        )
        produced = F.softplus(design @ solved.squeeze(0))
        scale = target.norm().clamp_min(1e-30)
        error = float((produced - target).norm() / scale)
        anchor_only = F.softplus(X @ a[anchor] + dt[anchor])
        anchor_error = float((anchor_only - target).norm() / scale)
        if not (error == error) or error > anchor_error:
            return a[anchor].clone(), dt[anchor].clone(), anchor_error, "refused"
        return (
            solved.squeeze(0)[:-1].contiguous(), solved.squeeze(0)[-1], error, "installed"
        )

    def old_refit_beta(X, b, parent_heads, weights):
        anchor_row = sum(float(w) * b[j] for w, j in zip(weights, parent_heads))
        target = torch.zeros(X.shape[0], dtype=torch.float64)
        for w, j in zip(weights, parent_heads):
            target += float(w) * torch.sigmoid(X @ b[j])
        scale = target.norm().clamp_min(1e-30)
        anchor_error = float((torch.sigmoid(X @ anchor_row) - target).norm() / scale)
        clamped = target.clamp(1e-12, 1.0 - 1e-12)
        logit_target = torch.log(clamped) - torch.log1p(-clamped)
        solved = qf_solve.prior_anchored(
            X, logit_target.unsqueeze(1), anchor_row.unsqueeze(0)
        )
        row = solved.squeeze(0)
        error = float((torch.sigmoid(X @ row) - target).norm() / scale)
        if not (error == error) or error >= anchor_error:
            return anchor_row.contiguous(), anchor_error, "refused"
        return row.contiguous(), error, "installed"

    # ------------------------------------------------ claim 1: same answer
    torch.manual_seed(4358)
    hidden, rows = 96, 512
    mu = torch.full((hidden,), 0.25, dtype=torch.float64)
    X = mu.unsqueeze(0) + 0.35 * torch.randn(rows, hidden, dtype=torch.float64)
    # Two parents at logits -6.mu and +2.mu: the Jensen gap the beta refit
    # exists for, so the fit is admitted rather than refused.
    b = torch.stack([-6.0 * mu, 2.0 * mu])
    a = torch.stack([0.8 * mu, -0.3 * mu + 0.05 * torch.randn(hidden, dtype=torch.float64)])
    dt = torch.tensor([0.4, -1.1], dtype=torch.float64)
    A_log = torch.tensor([0.7, -0.9], dtype=torch.float64)
    weights = torch.tensor([0.4, 0.6], dtype=torch.float64)
    anchor = 1
    grams = qf_gdn._refit_grams(X)

    new_beta_row, new_beta = qf_gdn._refit_beta(grams, b, [0, 1], weights)
    old_beta_row, old_beta_error, old_beta_state = old_refit_beta(X, b, [0, 1], weights)
    beta_row_difference = float((new_beta_row - old_beta_row).abs().max())
    assert beta_row_difference == 0.0, beta_row_difference
    assert new_beta["beta_relL2"] == old_beta_error, (new_beta["beta_relL2"], old_beta_error)
    assert new_beta["refit"].startswith(old_beta_state), (new_beta["refit"], old_beta_state)

    new_decay_row, new_decay_bias, new_decay = qf_gdn._refit_decay(
        grams, a, dt, A_log, [0, 1], weights, anchor
    )
    old_decay_row, old_decay_bias, old_decay_error, old_decay_state = old_refit_decay(
        X, a, dt, A_log, [0, 1], weights, anchor
    )
    decay_row_difference = float((new_decay_row - old_decay_row).abs().max())
    decay_bias_difference = float((new_decay_bias - old_decay_bias).abs().max())
    assert decay_row_difference == 0.0, decay_row_difference
    assert decay_bias_difference == 0.0, decay_bias_difference
    assert new_decay["softplus_relL2"] == old_decay_error, (
        new_decay["softplus_relL2"], old_decay_error
    )
    assert new_decay["refit"].startswith(old_decay_state), (new_decay["refit"], old_decay_state)
    # A refusal returns the anchor's own row on both sides, and two anchors are
    # equal whatever the solver did -- so the fixture has to install both or it
    # proves nothing about the solve path.
    assert old_beta_state == "installed" and old_decay_state == "installed", (
        old_beta_state, old_decay_state
    )

    # ------------------------------------------- claim 2: the call graph
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    before = (int(tc.linear_num_value_heads), int(tc.linear_num_key_heads))
    calls = {"formations": 0, "gram_solves": 0}
    real_grams, real_solve = qf_gdn._refit_grams, qf_gdn.prior_anchored_gram

    def counted_grams(evidence_rows):
        calls["formations"] += 1
        return real_grams(evidence_rows)

    def counted_solve(*args, **kwargs):
        calls["gram_solves"] += 1
        return real_solve(*args, **kwargs)

    qf_gdn._refit_grams = counted_grams
    qf_gdn.prior_anchored_gram = counted_solve
    try:
        report = qf_gdn.gdn_axis(
            harness, value_heads=before[0] // 2, key_heads=before[1] // 2, log=quiet
        )
    finally:
        qf_gdn._refit_grams = real_grams
        qf_gdn.prior_anchored_gram = real_solve

    layers = len(report["per_layer"])
    child_heads = sum(len(layer["per_child_head"]) for layer in report["per_layer"])
    assert calls["formations"] == layers, (calls, layers)
    assert calls["gram_solves"] == 2 * child_heads, (calls, child_heads)
    assert child_heads > layers, (
        "one child head per layer cannot tell a per-layer hoist from a per-head one"
    )

    return {
        "beta_row_max_abs_difference_vs_row_form": beta_row_difference,
        "beta_relL2_bit_identical": new_beta["beta_relL2"] == old_beta_error,
        "beta_refit": new_beta["refit"],
        "decay_row_max_abs_difference_vs_row_form": decay_row_difference,
        "decay_bias_max_abs_difference_vs_row_form": decay_bias_difference,
        "decay_relL2_bit_identical": new_decay["softplus_relL2"] == old_decay_error,
        "decay_refit": new_decay["refit"],
        "layers_folded": layers,
        "child_value_heads": child_heads,
        "gram_formations": calls["formations"],
        "gram_formations_before_the_fix": 2 * child_heads,
        "gram_solves": calls["gram_solves"],
    }


@test
def test_c5_gdn_head_dim(state):
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    before = (int(tc.linear_key_head_dim), int(tc.linear_value_head_dim))
    report = qf_gdn.gdn_axis(
        harness, key_head_dim=before[0] // 4, value_head_dim=before[1] // 4, log=quiet
    )
    assert int(tc.linear_key_head_dim) == before[0] // 4
    assert int(tc.linear_value_head_dim) == before[1] // 4
    gdn = next(l.linear_attn for l in harness.layers()
               if qf_arch.layer_kind(l) == qf_arch.LINEAR)
    assert int(gdn.norm.weight.shape[0]) == before[1] // 4
    assert gdn.conv1d.weight.shape[0] == gdn.conv_dim
    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    runs(harness)
    out = WORK / "child-c5dim"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, _ = qf_export.verify_child(out)
    return {
        "parent_head_dims": before,
        "child_head_dims": (int(tc.linear_key_head_dim), int(tc.linear_value_head_dim)),
        "component_ratio": report["component_ratio"],
        "fit_relL2": [layer["scores"]["fit"]["relL2"] for layer in report["per_layer"]],
        "query_scale_note": report["per_layer"][0]["query_scale_correction"],
        "parameters": counts["live"],
        "reload": verify["loaded_class"],
    }


@test
def test_c6_nope_contraction(state):
    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    rotary = qf_arch.rotary_dim_of(harness.model.config)
    parent_head_dim = int(tc.head_dim)
    keep = (parent_head_dim - rotary) // 4   # 4x on the NoPE coordinates

    # The FIRST full-attention layer, because every layer before it is a GDN
    # layer that C6 does not touch -- so the child's input to this site is the
    # parent's, byte for byte, and the parent/child probability comparison below
    # is a comparison of this fold and nothing else.
    site = next(i for i, l in enumerate(harness.layers())
                if qf_arch.layer_kind(l) == qf_arch.FULL)
    assert all(qf_arch.layer_kind(l) == qf_arch.LINEAR for l in harness.layers()[:site])
    # The instrument BEFORE the fold: if the recompute disagrees with the
    # module's own probabilities, every number derived from it is void.
    before = qf_attn.attention_gate_probe(harness, site, keep_tensors=True)
    assert before["recompute_relL2"] < PROBE_TOLERANCE, before["recompute_relL2"]

    report = qf_attn.nope_dim_contraction(harness, keep_nope=keep, log=quiet)
    child_head_dim = rotary + keep

    # ---- the SHARED query frame.  `Pk` is shared across a kv group because the
    # child has one key row block per group; `Pq` is free per head and is built
    # from the group-averaged metric.  For a fixed `Pk` the per-head conditional
    # optimum `(Pk Ck Pk^T)^-1 Pk Ck` is head-independent, so the installed
    # operator should already BE that optimum for every head.  The axis solves
    # it separately and this is the check that the two agree; without it the
    # docstring's claim is a claim.
    frames = [
        grp["query_metric_aggregation"]
        for layer in report["per_layer"] for grp in layer["kv_groups"]
    ]
    assert frames and all(len(grp["per_head"]) > 1 for grp in frames), (
        "a kv group with one query head cannot exhibit a sharing cost; the check "
        "would be vacuous"
    )
    query_frame_gap = max(
        grp["installed_vs_per_head_conditional_optimum_max_gap"] for grp in frames
    )
    assert query_frame_gap < 1e-6, (
        f"the installed query frame is {query_frame_gap:.3e} away from the per-head "
        f"conditional optimum; either the averaging is costing something the "
        f"algebra says it cannot, or the construction is not what the docstring says"
    )
    head_errors = [
        row["own_metric_relerror_installed"] for grp in frames for row in grp["per_head"]
    ]
    assert int(tc.head_dim) == child_head_dim
    assert abs(float(tc.rope_parameters["partial_rotary_factor"]) - rotary / child_head_dim) < 1e-12
    assert qf_arch.rotary_dim_of(harness.model.config) == rotary, (
        "the rotary width moved; mrope_section would no longer be valid"
    )
    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    runs(harness)
    qf_excise.assert_causal_masks(harness)

    # The instrument AFTER the fold.  head_dim moved, so the module's scaling
    # moved with it; the construction preserves q^T k, so the only thing that
    # can make the child's logits a different SIZE than the parent's is the
    # softmax temperature.  logit_scale_ratio measures exactly that and is blind
    # to the truncation's own (orthogonal) error.
    after = qf_attn.attention_gate_probe(harness, site, keep_tensors=True)
    assert after["recompute_relL2"] < PROBE_TOLERANCE, after["recompute_relL2"]
    alpha = qf_attn.logit_scale_ratio(after["logits"], before["logits"])
    assert abs(alpha - 1.0) < TEMPERATURE_LINE, (
        f"the child's attention logits are {1.0 / alpha:.4f}x the parent's: at "
        f"head_dim {parent_head_dim} -> {child_head_dim} the module's scaling is "
        f"{after['scaling']:.6f} where the parent's was {before['scaling']:.6f}, and "
        f"the softmax-temperature correction is missing or wrong"
    )

    out = WORK / "child-c6"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, reloaded = qf_export.verify_child(out)
    # The temperature correction lives in a WEIGHT (q_norm) and the temperature
    # itself is recomputed from the child's CONFIG on re-load.  This is the one
    # check that says the two agree after a round trip through disk.
    ids = harness.batches[0][1]
    round_trip = float(
        (reloaded(input_ids=ids, use_cache=False).logits - harness.logits(ids)).norm()
        / harness.logits(ids).norm()
    )
    assert round_trip < 1e-6, round_trip
    assert float(qf_arch.layers(reloaded)[site].self_attn.scaling) == after["scaling"]
    return {
        "export_round_trip_logit_relL2": round_trip,
        "parent_head_dim": parent_head_dim,
        "child_head_dim": child_head_dim,
        "rotary_dim_unchanged": rotary,
        "kv_cache_ratio": report["kv_cache_ratio"],
        "fit_relL2": [layer["scores"]["fit"]["relL2"] for layer in report["per_layer"]],
        "rms_correction": report["per_layer"][0]["rms_correction"]["query"],
        "query_frame": {
            "installed_vs_per_head_conditional_optimum_max_gap": query_frame_gap,
            "own_metric_relerror_spread": [min(head_errors), max(head_errors)],
            "heads_priced": len(head_errors),
        },
        "softmax_temperature_correction": report["softmax_temperature_correction"],
        "probe": {
            "site": site,
            "parent_recompute_relL2": before["recompute_relL2"],
            "child_recompute_relL2": after["recompute_relL2"],
            "scaling": [before["scaling"], after["scaling"]],
            "logit_scale_ratio_child_vs_parent": alpha,
            "uncorrected_would_be": (child_head_dim / parent_head_dim) ** 0.5,
            "mean_max_probability": [
                before["mean_max_probability"], after["mean_max_probability"]
            ],
        },
        "parameters": counts["live"],
        "reload": verify["loaded_class"],
    }


@test
def test_c7_kv_merge(state):
    parent, manifest = state["parent"], state["manifest"]

    harness = fresh_harness(parent, manifest)
    tc = harness.text_config
    parent_kv = int(tc.num_key_value_heads)
    parent_heads = int(tc.num_attention_heads)
    reference = fresh_harness(parent, manifest)

    site = next(i for i, l in enumerate(harness.layers())
                if qf_arch.layer_kind(l) == qf_arch.FULL)
    assert all(qf_arch.layer_kind(l) == qf_arch.LINEAR for l in harness.layers()[:site])
    before = qf_attn.attention_gate_probe(harness, site, keep_tensors=True)
    assert before["recompute_relL2"] < PROBE_TOLERANCE, before["recompute_relL2"]

    report = qf_attn.kv_head_merge(harness, kv_heads=parent_kv // 2, arm="kv-only", log=quiet)
    assert int(tc.num_key_value_heads) == parent_kv // 2
    assert int(tc.num_attention_heads) == parent_heads

    # The kv-only arm's exactness claim, checked rather than asserted: every
    # gate row must be the parent's, byte for byte.
    position = next(i for i, l in enumerate(harness.layers())
                    if qf_arch.layer_kind(l) == qf_arch.FULL)
    child_attn = harness.layers()[position].self_attn
    parent_attn = reference.layers()[position].self_attn
    slices = qf_attn.attention_slices(parent_attn, reference.model.config)
    for head in range(parent_heads):
        assert torch.equal(
            child_attn.q_proj.weight.data[slices["gate_rows"](head)],
            parent_attn.q_proj.weight.data[slices["gate_rows"](head)],
        ), f"gate rows of head {head} moved in the kv-only arm"

    counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    runs(harness)
    qf_excise.assert_causal_masks(harness)

    # head_dim does not move in C7, so the temperature is the parent's by
    # construction; the probe is here as the arithmetic check the arm otherwise
    # has no numerical instrument for, and the scale ratio is RECORDED (a key
    # merge is allowed to move the score, unlike a temperature change).
    after = qf_attn.attention_gate_probe(harness, site, keep_tensors=True)
    assert after["recompute_relL2"] < PROBE_TOLERANCE, after["recompute_relL2"]
    assert after["scaling"] == before["scaling"]
    alpha = qf_attn.logit_scale_ratio(after["logits"], before["logits"])

    grouped = fresh_harness(parent, manifest)
    full = qf_attn.kv_head_merge(
        grouped, kv_heads=parent_kv // 2, arm="full-group", log=quiet
    )
    assert int(grouped.text_config.num_attention_heads) == parent_heads // 2
    qf_arch.assert_parameter_count(grouped.model, grouped.model.config)
    runs(grouped)

    out = WORK / "child-c7"
    if out.exists():
        shutil.rmtree(out)
    qf_export.export_child(harness.model, out, parent_dir=parent)
    verify, _ = qf_export.verify_child(out)
    return {
        "kv_only": {
            "kv_heads": [parent_kv, parent_kv // 2],
            "attention_heads_unchanged": parent_heads,
            "gate_rows_bit_identical": True,
            "kv_cache_ratio": report["kv_cache_ratio"],
            "fit_relL2": [l["scores"]["fit"]["relL2"] for l in report["per_layer"]],
            "relation_error_mean": [
                head["relation"]["relation_error_mean"]
                for head in report["per_layer"][0]["per_kv_head"]
            ],
            "key_rms_ratio": report["per_layer"][0]["key_rms_ratio"],
            "probe": {
                "site": site,
                "parent_recompute_relL2": before["recompute_relL2"],
                "child_recompute_relL2": after["recompute_relL2"],
                "scaling_unchanged": before["scaling"],
                "logit_scale_ratio_child_vs_parent": alpha,
                "mean_max_probability": [
                    before["mean_max_probability"], after["mean_max_probability"]
                ],
            },
        },
        "full_group": {
            "attention_heads": [parent_heads, int(grouped.text_config.num_attention_heads)],
            "fit_relL2": [l["scores"]["fit"]["relL2"] for l in full["per_layer"]],
        },
        "parameters": counts["live"],
        "reload": verify["loaded_class"],
    }


@test
def test_c7_zero_gain_coordinate(state):
    """What actually killed the K2KV build at full-attention layer 27.

    Qwen3.5-4B layer 27's ``q_norm`` carries ONE coordinate -- index 1 -- whose
    bf16 weight is exactly -1.0, so its effective gain ``1 + weight`` is exactly
    0.  No other full-attention layer of that parent has one, which is why the
    build folded layers 3, 7, 11, 15, 19 and 23 and then died.  C7 solves the
    relation on gain-scaled rows and divides the gain back out, and coordinate 1
    is the REAL half of rotary plane 1: the scaled row is zero, the plane is not,
    and the complex coefficient the relation solve applies to the plane puts a
    non-zero real part back into that row.  ``non-zero / 0`` is +-Inf, ``q_norm``
    turns that into NaN across the whole head, and the writer's Gram, its
    diagonal mean and therefore its ridge are all NaN -- which ``cholesky``
    reports as "the leading minor of order 1 is not positive-definite".

    The fixture plants exactly that gain at exactly that geometry.  It asserts
    the failure APPEARS on the naked division and DISAPPEARS through
    :func:`qf_attn.unfold_gain`, with the same inputs, and then that a whole
    fold over the planted parent installs finite weights and runs.
    """

    parent, manifest = state["parent"], state["manifest"]
    harness = fresh_harness(parent, manifest)
    reference = fresh_harness(parent, manifest)
    site = next(i for i, l in enumerate(harness.layers())
                if qf_arch.layer_kind(l) == qf_arch.FULL)
    head_dim = int(harness.text_config.head_dim)
    rotary_dim = qf_arch.rotary_dim_of(harness.model.config)
    layout = qf_arch.plane_layout(head_dim, rotary_dim)
    dead = 1                                  # lo[1] -- the real half of plane 1
    assert dead in layout["lo"].tolist(), (dead, layout["lo"].tolist())

    for target in (harness, reference):
        with torch.no_grad():
            target.layers()[site].self_attn.q_norm.weight[dead] = -1.0
    parent_attn = reference.layers()[site].self_attn
    gq = qf_arch.effective_gain(parent_attn.q_norm).detach().to("cpu", torch.float64)
    assert float(gq[dead]) == 0.0, float(gq[dead])
    assert float(gq.abs().min()) == 0.0

    # ---- the failure, in isolation: a plane rotation leaks into the dead row
    # and the division installs an infinity.
    slices = qf_attn.attention_slices(parent_attn, reference.model.config)
    raw = parent_attn.q_proj.weight.detach().to("cpu", torch.float64)[slices["q_rows"](0)]
    scaled = raw * gq.unsqueeze(1)
    assert float(scaled[dead].abs().max()) == 0.0, "the fixture's dead row is not dead"
    rotated = qf_solve.as_planes(scaled, layout) * complex(0.6, 0.8)
    leaked = qf_solve.from_planes(rotated, layout)
    assert float(leaked[dead].abs().max()) > 0.0, (
        "the rotary phase did not leak into the zero-gain row; this fixture "
        "cannot see the bug it is about"
    )
    naked = leaked / gq.unsqueeze(1)
    assert not bool(torch.isfinite(naked).all()), "the naked division stayed finite"
    naked_nonfinite = int((~torch.isfinite(naked)).sum())

    # ---- and the same inputs through the guard.
    record: list = []
    fixed = qf_attn.unfold_gain(leaked, gq, raw, record=record, where="fixture")
    assert bool(torch.isfinite(fixed).all()), "unfold_gain still installed a non-finite"
    assert torch.equal(fixed[dead], raw[dead]), (
        "the dead coordinate must carry the PARENT's raw row, so the per-token "
        "RMS q_norm divides the head by is unchanged"
    )
    live = torch.ones(head_dim, dtype=torch.bool)
    live[dead] = False
    assert torch.allclose(fixed[live], naked[live], atol=1e-12), (
        "unfold_gain changed a coordinate whose gain was not zero"
    )
    assert record and record[0]["coordinates"] == [dead], record

    # ---- the whole fold over the planted parent.
    parent_kv = int(harness.text_config.num_key_value_heads)
    report = qf_attn.kv_head_merge(
        harness, kv_heads=parent_kv // 2, arm="kv-only", log=quiet
    )
    child_attn = harness.layers()[site].self_attn
    assert bool(torch.isfinite(child_attn.q_proj.weight).all()), (
        "the fold installed a non-finite q_proj"
    )
    for key in ("k_proj", "v_proj", "o_proj"):
        assert bool(torch.isfinite(getattr(child_attn, key).weight).all()), key

    entry = next(row for row in report["per_layer"] if row["layer"] == site)
    assert entry["zero_gain_coordinates"], entry["zero_gain_coordinates"]
    assert all(row["coordinates"] == [dead] for row in entry["zero_gain_coordinates"]), (
        entry["zero_gain_coordinates"]
    )
    for other in report["per_layer"]:
        if other["layer"] != site:
            assert other["zero_gain_coordinates"] == [], other["layer"]

    # Every query slot's dead row is the parent's own row, byte for byte.
    for head in range(int(reference.text_config.num_attention_heads)):
        rows = slices["q_rows"](head)
        assert torch.equal(
            child_attn.q_proj.weight.data[rows][dead],
            parent_attn.q_proj.weight.data[rows][dead],
        ), head

    logits = runs(harness)
    assert bool(torch.isfinite(logits).all()), "the folded child produces non-finite logits"

    # The other arm divides by the same gain and is guarded by the same helper.
    grouped = fresh_harness(parent, manifest)
    with torch.no_grad():
        grouped.layers()[site].self_attn.q_norm.weight[dead] = -1.0
    full = qf_attn.kv_head_merge(
        grouped, kv_heads=parent_kv // 2, arm="full-group", log=quiet
    )
    grouped_attn = grouped.layers()[site].self_attn
    assert bool(torch.isfinite(grouped_attn.q_proj.weight).all()), (
        "the full-group arm installed a non-finite q_proj"
    )
    assert bool(torch.isfinite(runs(grouped)).all())

    return {
        "planted": {
            "layer": site,
            "coordinate": dead,
            "geometry": "lo[1]: the real half of rotary plane 1",
            "effective_gain": float(gq[dead]),
            "convention": "Qwen3_5RMSNorm multiplies by (1.0 + weight)",
        },
        "naked_division_nonfinite_entries": naked_nonfinite,
        "unfold_gain_nonfinite_entries": 0,
        "dead_row_equals_parent_row": True,
        "kv_only": {
            "child_q_proj_finite": True,
            "zero_gain_coordinates": entry["zero_gain_coordinates"],
            "fit_relL2": [row["scores"]["fit"]["relL2"] for row in report["per_layer"]],
            "logits_finite": True,
        },
        "full_group": {
            "child_q_proj_finite": True,
            "fit_relL2": [row["scores"]["fit"]["relL2"] for row in full["per_layer"]],
        },
    }


@test
def test_c9_terminal_gain(state):
    """The identity gate first, then a real repair on a folded child."""

    parent, manifest = state["parent"], state["manifest"]

    a, b = fresh_harness(parent, manifest), fresh_harness(parent, manifest)
    prior = qf_arch.effective_gain(a.final_norm()).clone().to(torch.float64)
    identity = qf_termgain.terminal_gain(a, b, lambda_split=False, log=quiet)
    solved = qf_arch.effective_gain(a.final_norm()).to(torch.float64)
    drift = float((solved - prior).abs().max())
    assert drift < 1e-8, f"identity gate: gain moved by {drift:.3e} at Z == N"

    child = fresh_harness(parent, manifest)
    qf_mlp.mlp_axis(child, int(child.text_config.intermediate_size) // 4, log=quiet)
    reference = fresh_harness(parent, manifest)
    # lambda_split=False so the assertion is on the unregularised least squares:
    # with a rung chosen on an internal split the fit SSE is allowed to be worse
    # than the prior's, which is the point of the split and not a failure.
    report = qf_termgain.terminal_gain(child, reference, lambda_split=False, log=quiet)
    assert report["reader_relative_sse_after"]["fit"] <= report["reader_relative_sse_before"]["fit"]
    split_child = fresh_harness(parent, state["manifest"])
    qf_mlp.mlp_axis(split_child, int(split_child.text_config.intermediate_size) // 4, log=quiet)
    split_report = qf_termgain.terminal_gain(
        split_child, fresh_harness(parent, state["manifest"]), lambda_split=True, log=quiet
    )
    runs(child)
    return {
        "identity_gate_max_gain_drift": drift,
        "identity_relative_sse": identity["reader_relative_sse_after"]["fit"],
        "folded_child_reader_relSSE": [
            report["reader_relative_sse_before"]["fit"],
            report["reader_relative_sse_after"]["fit"],
        ],
        "held": [
            report["reader_relative_sse_before"].get("hld"),
            report["reader_relative_sse_after"].get("hld"),
        ],
        "gain_update_scale": report["gain_update_scale"],
        "lambda_split_path_lambda": split_report["solve"]["lambda_relative"],
    }


@test
def test_c1_chain_via_remapped_evidence(state):
    """The vocabulary child is exclusive, and the CHAIN that replaces combining it.

    C1 renames every embedding row, so a second axis in the same call would be
    solved on a stream in the parent's id space.  The build refuses that; this
    test proves the refusal is not a dead end -- the vocabulary child writes its
    own remapped, re-sealed manifest, and the next build runs against THAT.
    """

    import qf_build

    parent, manifest = state["parent"], state["manifest"]
    first = WORK / "chain-c1"
    second = WORK / "chain-c1-c2"
    for path in (first, second):
        if path.exists():
            shutil.rmtree(path)

    receipt_one = qf_build.build(qf_build.parse_args([
        "--parent", str(parent), "--out", str(first),
        "--evidence", str(WORK / "parent-evidence.json"),
        "--tag", "CHAINV", "--device", "cpu", "--dtype", "float32",
        "--vocab-keep", "384",
    ]))
    descendants = receipt_one["preconditions"]["evidence_for_descendants"]
    remapped_path = Path(descendants["path"])
    assert remapped_path.name == "evidence-remapped.json" and remapped_path.exists()
    remapped = qf_corpus.read_manifest(remapped_path)      # re-seals, or refuses
    assert remapped["manifest_sha256"] == descendants["manifest_sha256"]
    assert remapped["manifest_sha256"] != manifest["manifest_sha256"]
    highest = max(int(i) for row in remapped["rows"] for i in row["ids"])
    assert highest < 384, highest
    # The remap is a RENAME, not a re-tokenization: same shape, same splits,
    # different ids.
    assert len(remapped["rows"]) == len(manifest["rows"])
    assert [row["split"] for row in remapped["rows"]] == \
        [row["split"] for row in manifest["rows"]]
    # The remapped stream is sealed under the CHILD's tokenizer, which is what
    # lets the chained build's tokenizer gate pass on a measurement rather than
    # on an exception -- and it is a different digest from the parent's.
    child_tokenizer_sha = qf_corpus.file_sha256(first / "tokenizer.json")
    assert remapped["tokenizer_sha256"] == child_tokenizer_sha
    assert remapped["tokenizer_sha256"] != manifest["tokenizer_sha256"]
    assert "qf_vocab.remap_manifest" in remapped["source"]
    # (On this fixture the rename happens to be the identity: the stream is drawn
    # from the 256 byte tokens, which are mandatory and are the lowest ids, so
    # they keep their numbers.  The SEAL still changes, which is why the child's
    # manifest is a different file and not the parent's under another name.)

    receipt_two = qf_build.build(qf_build.parse_args([
        "--parent", str(first), "--out", str(second),
        "--evidence", str(remapped_path),
        "--tag", "CHAINV2", "--device", "cpu", "--dtype", "float32",
        "--lineage", receipt_one["export"]["child_tensors_sha256"],
        "--mlp-keep", "64",
    ]))
    assert receipt_two["lineage"] == [receipt_one["export"]["child_tensors_sha256"]]
    assert receipt_two["verification"]["parameters_counted_live"] < \
        receipt_one["verification"]["parameters_counted_live"]
    return {
        "vocab_child": {
            "tag": receipt_one["tag"],
            "vocab": [512, 384],
            "remapped_evidence": remapped_path.name,
            "remapped_manifest_sha256": remapped["manifest_sha256"][:16],
            "highest_id_in_child_space": highest,
            "parameters": receipt_one["verification"]["parameters_counted_live"],
        },
        "chained_child": {
            "tag": receipt_two["tag"],
            "axes": [axis["axis"] for axis in receipt_two["axes"]],
            "parameters": receipt_two["verification"]["parameters_counted_live"],
            "net_stored_ratio": receipt_two["storage"]["net_stored_ratio"],
            "lineage": [d[:16] for d in receipt_two["lineage"]],
        },
    }


@test
def test_build_cli_end_to_end(state):
    """The whole CLI: fold, export, verify, seal a receipt, parse it back."""

    import qf_build

    parent, _ = state["parent"], state["manifest"]
    out = WORK / "child-cli"
    if out.exists():
        shutil.rmtree(out)
    args = qf_build.parse_args(
        [
            "--parent", str(parent),
            "--out", str(out),
            "--evidence", str(WORK / "parent-evidence.json"),
            "--tag", "SELFTESTC2",
            "--parent-code", "TINY-128",
            "--device", "cpu",
            "--dtype", "float32",
            "--mlp-keep", "64",
            "--excise", "0",
            "--notes", "qf_selftest end-to-end",
        ]
    )
    # The combination refusals, BEFORE the build that is expected to work: a
    # vocabulary fold renames every row and leaves the sealed stream in the
    # parent's id space, so it cannot share a call with an axis that captures
    # afterwards; and mtp.* cannot be carried past a shape it reads.
    def refused(argv: list[str]) -> str:
        try:
            qf_build.refuse_illegal_combinations(qf_build.parse_args(argv))
        except SystemExit as error:
            return str(error)
        return ""

    base = ["--parent", str(parent), "--out", str(WORK / "never"),
            "--evidence", str(WORK / "parent-evidence.json"), "--tag", "X"]
    vocab_refusal = refused(base + ["--vocab-keep", "384", "--mlp-keep", "64"])
    assert "cannot be combined with --mlp-keep" in vocab_refusal, vocab_refusal
    assert not refused(base + ["--vocab-keep", "384"])
    assert not refused(base + ["--mlp-keep", "64", "--excise", "0"])
    mtp_refusal = ""
    try:
        qf_build.assert_mtp_carry_legal(
            qf_arch.tiny_text_config(), qf_arch.tiny_text_config(head_dim=16)
        )
    except SystemExit as error:
        mtp_refusal = str(error)
    assert "head_dim 32 -> 16" in mtp_refusal, mtp_refusal

    receipt = qf_build.build(args)
    assert receipt["schema"] == "neural-forge-omni-qwen3_5-standard-shape-child-v1"
    assert receipt["quantization"] == "none"
    assert receipt["standard_shape"] is True
    assert receipt["storage"]["net_stored_ratio"] < 1.0
    assert receipt["verification"]["parameters_counted_live"] == \
        receipt["storage"]["child_parameter_count"]
    assert receipt["preconditions"]["causal_masks"] is not None
    assert not receipt["gguf"]["ran"] and receipt["gguf"]["command"][1].endswith(
        "convert_hf_to_gguf.py"
    )
    assert not receipt["review_hook"]["ran"]
    axes = {axis["axis"] for axis in receipt["axes"]}
    assert axes == {"layer_excision", "intermediate_size"}, axes
    reread = json.loads((out / "qf-build-receipt.json").read_text(encoding="ascii"))
    assert reread["receipt_sha256"] == receipt["receipt_sha256"]
    return {
        "tag": receipt["tag"],
        "axes": sorted(axes),
        "net_stored_ratio": receipt["storage"]["net_stored_ratio"],
        "net_fold": receipt["storage"]["net_fold"],
        "per_component": {
            row["component"]: row["ratio"] for row in receipt["per_component_ratios"]
        },
        "child_tensors_sha256": receipt["export"]["child_tensors_sha256"][:16],
        "receipt_sha256": receipt["receipt_sha256"][:16],
        "gguf_command": " ".join(receipt["gguf"]["command"][:3]) + " ...",
        "wall_time_seconds": receipt["wall_time_seconds"],
    }



@test
def test_evidence_builder_row_atomicity(state):
    """The evidence builder places rows WHOLE, and the old packer did not.

    The delivered manifest was packed by concatenating rows into one flat stream
    and slicing it at fixed offsets, which cut a document at every batch boundary:
    every held batch opened with the tail of a document the solve had just been
    fitted on.  This test packs the same pools both ways and asserts the
    difference, because "the split is a document split" is the claim the whole
    fit/held number rests on.
    """

    import qf_evidence_build as qfev

    routes = ("text", "code", "math")
    # Row lengths in the same proportion to the batch as the real registry's are
    # to 4,096 (mean ~490, longest a few thousand): a fixture whose rows are
    # nearly as long as a batch would measure the fixture, not the packer.
    lengths = {"text": (140, 210, 96), "code": (320, 64, 180), "math": (260, 120, 90)}
    separator = 999
    pools = {}
    for route in routes:
        rows = []
        for index in range(60):
            size = lengths[route][index % 3] + (index % 7)
            ids = [(index * 31 + position) % 500 + 1 for position in range(size - 1)]
            rows.append(
                qfev.Sampled(
                    route=route,
                    ids=ids + [separator],
                    record_id=f"{route}-{index}",
                    dataset_id=f"ds-{route}-{index % 3}",
                    shard_path=f"/fake/{route}.jsonl",
                    row_sha256=f"{index:064x}",
                    characters=size * 4,
                )
            )
        pools[route] = rows

    length, batches = 1024, 8
    ids, owners, report = qfev.pack_batches(
        pools, routes=routes, sequence_length=length, batches=batches,
        seed=20260902, separator_id=separator,
    )
    assert len(ids) == batches and all(len(batch) == length for batch in ids)
    leakage = report["split_leakage"]
    assert leakage["rows_spanning_batches"] == 0, leakage
    assert leakage["batches_starting_mid_row"] == 0, leakage
    # Independently of the builder's own report: a batch that was cut through a
    # document does not end on the separator.
    assert all(batch[-1] == separator for batch in ids)
    # Padding is eos, it only ever sits at the END of a batch, and the owner
    # stream labels it as padding rather than as a route.
    for batch, owner in zip(ids, owners):
        pad = owner.count("pad")
        assert owner[length - pad:] == ["pad"] * pad
        assert all(token == separator for token in batch[length - pad:])
    # The invariant, not a magic number: row-atomic packing can only ever waste
    # what no remaining row fits into, so a batch's padding is bounded by the
    # longest row it could have taken, and on rows this shape it is a fraction of
    # a percent.  (The bound is the honest one; the fraction is the measurement.)
    longest = max(len(row.ids) for pool in pools.values() for row in pool)
    assert all(pad < longest for pad in report["padding_tokens_by_batch"]), report
    assert report["padding_fraction"] < 0.02, report["padding_fraction"]
    assert set(report["rows_consumed_by_route"]) == set(routes)

    # NEGATIVE CONTROL: the flat-stream packer this replaced, on the same pools.
    flat: list[int] = []
    starts: list[int] = []
    for index in range(200):
        route = routes[index % len(routes)]
        row = pools[route][index // len(routes)]
        starts.append(len(flat))
        flat.extend(row.ids)
        if len(flat) >= length * batches:
            break
    boundaries = [position * length for position in range(1, batches)]
    row_starts = set(starts)
    cut = [position for position in boundaries if position not in row_starts]
    assert cut, "the flat packer was expected to cut rows; the fixture is wrong"
    return {
        "batches": batches,
        "sequence_length": length,
        "rows_placed": leakage["rows_placed"],
        "rows_spanning_batches": leakage["rows_spanning_batches"],
        "batches_starting_mid_row": leakage["batches_starting_mid_row"],
        "padding_tokens": report["padding_tokens"],
        "padding_fraction": report["padding_fraction"],
        "padding_bound_longest_row": longest,
        "rows_taken_from_look_ahead": report["rows_taken_from_look_ahead"],
        "flat_packer_boundaries_cutting_a_row": f"{len(cut)} of {len(boundaries)}",
    }


@test
def test_evidence_builder_screen(state):
    """The contamination screen catches real prompts, per pack, and not filler."""

    import qf_evidence_build as qfev

    directory = WORK / "screen-packs"
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / "pack_one.json"
    second = directory / "pack_two.json"
    first.write_text(json.dumps({"cases": [
        {"prompt": "Explain why a rotary embedding is unchanged by a NoPE contraction, "
                   "in three sentences, without using the word dimension."},
        {"prompt": "Summarise the failure mode where a causal mask is dispatched "
                   "positionally and the above-diagonal mass stops being zero."},
    ]}), encoding="utf-8")
    second.write_text(json.dumps({"triplets": [
        {"source": "Describe the difference between a coreset chosen by pivoted QR "
                   "and one chosen by leverage score, for an audience of engineers."},
        {"source": "Write a paragraph about a ridge refit that re-forms its Gram "
                   "matrix once per layer instead of once per head."},
    ]}), encoding="utf-8")

    screen = qfev.build_review_screen(
        [first, second], needle_characters=48, allow_unsealed=True
    )
    liveness = qfev.screen_liveness(screen, probes=8)
    assert liveness["screen_is_live"], liveness
    assert liveness["spliced_probes_caught"] == liveness["probes"] > 0
    assert liveness["rewrapped_probes_caught"] == liveness["rewrapped_probes"]
    assert liveness["short_quote_controls_falsely_caught"] == 0
    assert liveness["clean_controls_falsely_caught"] == 0
    assert not liveness["sources_without_full_coverage"]
    assert all(row["attributed_to_this_source"] == row["probe_texts"]
               for row in liveness["by_source"]), liveness["by_source"]

    # The containment rule itself: a long quote is caught, a quote shorter than
    # the needle is not, and both readings are the point of the 48-char line.
    prompt = json.loads(first.read_text(encoding="utf-8"))["cases"][0]["prompt"]
    assert screen.drop_reasons("preamble " + prompt[:120] + " postamble")
    assert not screen.drop_reasons("preamble " + prompt[:47] + " postamble")

    # A pack that contributes nothing is a screen built over the wrong file.
    empty = directory / "pack_empty.json"
    empty.write_text(json.dumps({"cases": []}), encoding="utf-8")
    refused = ""
    try:
        qfev.build_review_screen([empty], needle_characters=48, allow_unsealed=True)
    except SystemExit as error:
        refused = str(error)
    assert "no strings" in refused, refused

    # A pinned pack whose bytes moved is refused unless the exception is recorded.
    pinned = directory / "lfm2_paraphrase_triplets_v1.json"
    pinned.write_text(second.read_text(encoding="utf-8"), encoding="utf-8")
    pin_refusal = ""
    try:
        qfev.build_review_screen([pinned], needle_characters=48, allow_unsealed=False)
    except SystemExit as error:
        pin_refusal = str(error)
    assert "the campaign pinned" in pin_refusal, pin_refusal
    return {
        "sources": liveness["sources"],
        "probes": liveness["probes"],
        "caught": liveness["spliced_probes_caught"],
        "rewrapped_caught": liveness["rewrapped_probes_caught"],
        "short_quote_controls_falsely_caught": liveness["short_quote_controls_falsely_caught"],
        "clean_controls_falsely_caught": liveness["clean_controls_falsely_caught"],
        "windows": len(screen.windows),
        "unsealed_pack_refused": pin_refusal[:80],
        "empty_pack_refused": refused[:60],
    }


@test
def test_evidence_manifest_is_not_destroyed(state):
    """A sealed manifest is not overwritten without --force, nor half-written."""

    import qf_evidence_build as qfev

    directory = WORK / "evidence-guard"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "omni-evidence-v1.json"
    manifest = qf_corpus.synthetic_manifest(
        vocab_size=512, sequence_length=16, batches=6, held_every=3, out=target
    )
    before = qf_corpus.file_sha256(target)
    receipt = target.with_name(target.stem + "-build-receipt.json")
    receipt.write_text(json.dumps({"schema": "stand-in"}), encoding="utf-8")

    refusal = ""
    try:
        qfev.guard_output(qfev.parse_args(["--out", str(target)]))
    except SystemExit as error:
        refusal = str(error)
    assert "already exists" in refusal and manifest["manifest_sha256"] in refusal, refusal
    assert qf_corpus.file_sha256(target) == before

    forced = qfev.guard_output(qfev.parse_args(["--out", str(target), "--force"]))
    backup = Path(forced["backup_directory"])
    assert backup.is_dir() and backup.name.startswith(".superseded-")
    assert forced["superseded_manifest_sha256"] == manifest["manifest_sha256"]
    assert qf_corpus.file_sha256(backup / target.name) == before
    assert (backup / receipt.name).is_file()
    assert qf_corpus.file_sha256(target) == before, "the guard must not write"

    # And the write itself is atomic: a failure mid-serialisation leaves the
    # previous seal on disk, not a fragment of the next one.
    class _Boom:
        @staticmethod
        def dumps(*_args, **_kwargs):
            raise RuntimeError("simulated: no space left on device")

    original = qf_corpus.json
    qf_corpus.json = _Boom
    try:
        qf_corpus.write_manifest(
            [{"batch_index": 0, "split": "fit", "ids": [1, 2, 3]}],
            tokenizer_sha256="x", source="atomicity probe", out=target,
        )
        raise AssertionError("the simulated failure did not propagate")
    except RuntimeError as error:
        assert "no space left" in str(error)
    finally:
        qf_corpus.json = original
    assert qf_corpus.file_sha256(target) == before, "a killed write truncated the seal"
    assert not target.with_name(target.name + ".tmp").exists()
    return {
        "refused_without_force": refusal[:80],
        "backup_directory": backup.name,
        "backed_up": [row["name"] for row in forced["backed_up"]],
        "manifest_sha256_unchanged": qf_corpus.file_sha256(target) == before,
    }


@test
def test_evidence_tokenizer_gate(state):
    """A build refuses evidence tokenized by a different tokenizer."""

    import qf_build

    parent, manifest = state["parent"], state["manifest"]
    passing = qf_build.assert_evidence_tokenizer(
        manifest, parent, allow_mismatch=False
    )
    assert passing["matches"], passing
    assert passing["parent_tokenizer_sha256"] == manifest["tokenizer_sha256"]

    # A second parent whose table is WIDE ENOUGH for every id in the stream but
    # whose tokenizer is a different file: nothing else downstream would notice.
    other = WORK / "other-parent"
    if other.exists():
        shutil.rmtree(other)
    other.mkdir(parents=True)
    qf_vocab.synthetic_tokenizer(512, other / "tokenizer.json", merges=180)
    assert qf_corpus.file_sha256(other / "tokenizer.json") != manifest["tokenizer_sha256"]
    refusal = ""
    try:
        qf_build.assert_evidence_tokenizer(manifest, other, allow_mismatch=False)
    except SystemExit as error:
        refusal = str(error)
    assert "was tokenized by" in refusal, refusal
    recorded = qf_build.assert_evidence_tokenizer(manifest, other, allow_mismatch=True)
    assert not recorded["matches"] and recorded["allow_mismatch"]

    # A parent with no tokenizer.json at all is a refusal too, not a shrug.
    bare = WORK / "bare-parent"
    bare.mkdir(parents=True, exist_ok=True)
    bare_refusal = ""
    try:
        qf_build.assert_evidence_tokenizer(manifest, bare, allow_mismatch=False)
    except SystemExit as error:
        bare_refusal = str(error)
    assert "(missing)" in bare_refusal, bare_refusal
    return {
        "parent_matches": passing["matches"],
        "mismatch_refused": refusal[:96],
        "mismatch_recordable": recorded["statement"],
        "missing_tokenizer_refused": bool(bare_refusal),
    }


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--work", default="")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    torch.manual_seed(20260901)
    torch.use_deterministic_algorithms(False)
    disabled = qf_arch.force_torch_kernels()

    WORK = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="qf-selftest-"))
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"[qf-selftest] transformers "
          f"{__import__('transformers').__version__}, torch {torch.__version__}")
    print(f"[qf-selftest] work dir {WORK}")
    print(f"[qf-selftest] fused kernels disabled: "
          f"{sorted(k for k, v in disabled.items() if v) or 'none were available'}")

    parent, manifest = tiny_parent()
    state = {"parent": parent, "manifest": manifest}
    print(f"[qf-selftest] tiny parent at {parent}, "
          f"{qf_arch.live_parameter_count(qf_arch.seeded_tiny_model(qf_arch.tiny_text_config())):,} parameters")

    results: dict[str, Any] = {}
    failures = 0
    for fn in TESTS:
        if args.only and args.only not in fn.__name__:
            continue
        started = __import__("time").time()
        try:
            payload = fn(state)
            elapsed = __import__("time").time() - started
            results[fn.__name__] = {"status": "PASS", "seconds": round(elapsed, 2),
                                    "evidence": payload}
            print(f"\nPASS  {fn.__name__}  ({elapsed:.1f}s)")
            print("      " + json.dumps(payload, default=str)[:1800])
        except Exception:
            failures += 1
            elapsed = __import__("time").time() - started
            trace = traceback.format_exc()
            results[fn.__name__] = {"status": "FAIL", "seconds": round(elapsed, 2),
                                    "traceback": trace}
            print(f"\nFAIL  {fn.__name__}  ({elapsed:.1f}s)")
            print(trace)

    print(f"\n[qf-selftest] {len(results) - failures}/{len(results)} passed")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"[qf-selftest] results written to {args.json_out}")
    if not args.keep_artifacts and not args.work:
        shutil.rmtree(WORK, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
