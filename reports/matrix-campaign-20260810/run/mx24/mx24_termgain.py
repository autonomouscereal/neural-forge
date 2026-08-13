#!/usr/bin/env python3
"""Wave 24: the terminal read, re-solved in the child's OWN token distribution.

### What this is, and what it is not

The slate's S4 (PAIR-KKT) asks for "one convex quadratic over BOTH folds' free
tensors."  Run against wave 23's mild pair -- ``ARCH.V16X13``, block 13 excised
plus the 16,384-row vocabulary coreset, 5/15 at 0.728353x -- that objective has a
problem the slate did not anticipate and which is reported here as a finding
rather than worked around:

**both folds are zero-solve.**  ``mx24_vocab.py``'s base arm gathers rows and
solves nothing; and ``MX16-X13A``'s build receipt carries ``keep_ff: null``, so
the excision took the ``continue`` branch of the per-site loop and re-solved
nothing either -- its own axis entry says "absorbed by every downstream
conditioned solve", and wave 15 §5 already priced the local re-solve of an
excision as an identical no-op (update_scale 0.029).  There is no pair of solved
tensors to couple.  A joint solve therefore has to CREATE its free set, and the
honest question is which free set is (a) already counted, (b) exportable in a
stock config, and (c) actually coupled to both folds.

There is exactly one: **the final norm's diagonal**, ``model.embedding_norm``.
It is 1,024 parameters the child already carries, it is the only post-norm
degree of freedom the architecture provides (nothing can be absorbed into a block
writer, because RMSNorm sits between them and is not linear), and it is coupled
to both folds at once -- the excision decides the residual that arrives, the
vocabulary coreset decides the table it is read through.  This is S1's arm B,
which the slate wrote and wave 23 did not build.

### The objective, exactly

Let ``t`` range over the campaign's 24 captured batches, decoded to text.  Let
``sigma_c(t)`` be the CHILD's segmentation and ``iota`` the child->parent
injection on kept rows (`mx24_retok.py`; both are asserted, not assumed).  Run
the parent on ``iota(sigma_c(t))`` and the child on ``sigma_c(t)``: the two
streams are position-aligned over identical text.  Write

* ``N`` = the parent's RMS-normalised terminal residual, rows = positions,
* ``Z`` = the child's RMS-normalised terminal residual, same rows,
* ``E`` = the child's tied token table (K x H), which IS the parent's rows at the
  kept ids,
* ``g`` = the parent's final-norm gain, which the child carries verbatim.

Then solve

    min_d  || Z diag(d) E^T  -  N diag(g) E^T ||_F^2  +  lambda ||d - g||^2

-- the child's own logits over its own vocabulary against the PARENT's logits
over the same rows on the same text.  It is convex, it is one system, and its
normal equations never materialise a K-column target:

    [ (E^T E) o (Z^T Z) ] d  =  [ (E^T E) o (Z^T N) ] g

with ``o`` the Hadamard product -- one H x H SPD solve.  ``lambda -> infinity``
returns ``g``, which is the sealed child in bytes; and at ``Z = N`` the solution
is ``d = g`` EXACTLY, not approximately, which is asserted synthetically every
run (`--no-identity-gate` to skip is not offered).

This is the terminal read, not a trajectory: the target is the model's OUTPUT on
its own input, which is the only quantity behaviour is a function of.  Wave 15's
harmful arm targeted the parent's stream at INTERMEDIATE sites; nothing here does.

### The 29-diagonal arm

``--block-gains`` adds the 28 block RMSNorm diagonals (``operator_norm`` and
``ffn_norm`` per surviving block) under the identical algebra with ``E^T E``
replaced by ``W_i^T W_i``, the reader's own column Gram: q/k/v concatenated for
an attention block's operator norm, ``in_proj`` for a conv block's, ``w1``/``w3``
concatenated for every FFN norm.  That is S6's 29-DIAGONAL, run on this wave's
evidence, and it is conditioned in depth order -- each install changes the
child's stream, so the child is re-captured before the next endpoint.  The
parent is captured ONCE: its stream does not move.

Storage delta is ZERO on both arms.  1,024 + 30,464 values are overwritten in
place; no tensor is added, no shape changes, ``config.json`` is untouched.

Usage::

    mx24_termgain.py --source <sealed child dir> --child <out dir> --tag V16X13G
    mx24_termgain.py ... --lambda-relative 1.0     # one rung of the atlas
    mx24_termgain.py ... --block-gains             # the 29-diagonal arm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

import mx24_retok

PARENT = mx24_retok.PARENT
PARENT_PARAMS = 229_693_184
SCHEMA = "neural-forge-matrix-wave24-standard-shape-child-v1"
LAMBDA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
#: Wave 18's law: lambda is chosen on a split of the FIT rows and never on the
#: campaign's held-out block.  Every third fit batch, counted from the third.
INTERNAL_CHECK_STRIDE = 3
INTERNAL_CHECK_OFFSET = 2
#: The exact identity the solve must reproduce at ``Z = N``.  fp64 through one
#: Cholesky of a 1024x1024 SPD system; the measured value is in the receipt.
IDENTITY_GATE_TOLERANCE = 1e-8


def log(message: str) -> None:
    print(f"[mx24gain] {message}", flush=True)


# --------------------------------------------------------------------- capture
class Streams:
    """The two position-aligned streams, captured once and reused by every solve."""

    def __init__(self, source: Path, device: str) -> None:
        from transformers import Lfm2ForCausalLM

        self.device = device
        self.rows, self.meta = mx24_retok.retokenize(source)
        self.parent = (
            Lfm2ForCausalLM.from_pretrained(str(PARENT), dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.child = (
            Lfm2ForCausalLM.from_pretrained(str(source), dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.eps = float(self.child.config.norm_eps)
        self.forward_calls = 0
        fit = [r["batch_index"] for r in self.rows if r["split"] == "fit"]
        check = set(fit[INTERNAL_CHECK_OFFSET :: INTERNAL_CHECK_STRIDE])
        self.internal_check = check
        log(
            f"streams: {len(self.rows)} batches, "
            f"{self.meta['child_stream_positions']} child positions "
            f"({self.meta['sequence_length_ratio']:.4f}x the parent's), internal "
            f"check batches {sorted(check)}"
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()

    @torch.no_grad()
    def capture(self, model: Any, modules: dict[str, Any], key: str) -> dict[str, Any]:
        """RMS-normalised inputs of each named module, plus the row bookkeeping.

        ``key`` selects which id stream this model is fed: ``parent_ids`` for the
        parent, ``child_ids`` for the child.  Feeding the wrong one is the single
        error this whole file would not survive, so it is an argument rather than
        an assumption about which model was passed.
        """

        state: dict[str, torch.Tensor] = {}
        handles = []
        for name, module in modules.items():
            handles.append(
                module.register_forward_pre_hook(
                    (
                        lambda n: lambda module, args, kwargs: state.__setitem__(
                            n, (args[0] if args else kwargs["hidden_states"]).detach()
                        )
                    )(name),
                    with_kwargs=True,
                )
            )
        blocks: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
        splits: list[str] = []
        lengths: list[int] = []
        indices: list[int] = []
        try:
            for row in self.rows:
                if len(row["parent_ids"]) != len(row["child_ids"]):
                    raise SystemExit(
                        f"batch {row['batch_index']}: the two id streams are "
                        "different lengths, so they are not position-aligned"
                    )
                ids = torch.tensor([row[key]], dtype=torch.long, device=self.device)
                model(input_ids=ids, use_cache=False)
                self.forward_calls += 1
                for name in modules:
                    hidden = state[name].reshape(-1, state[name].shape[-1])
                    blocks[name].append(self._norm(hidden.to("cpu", torch.float64)))
                splits.append(row["split"])
                lengths.append(len(row[key]))
                indices.append(row["batch_index"])
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return {
            "rows": {name: torch.cat(v, dim=0) for name, v in blocks.items()},
            "splits": splits,
            "lengths": lengths,
            "indices": indices,
        }

    def masks(self, captured: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Row masks: fit, the internal check split of fit, and the held-out block."""

        fit, check, held = [], [], []
        for index, split, length in zip(
            captured["indices"], captured["splits"], captured["lengths"]
        ):
            is_fit = split == "fit"
            fit += [is_fit] * length
            check += [is_fit and index in self.internal_check] * length
            held += [not is_fit] * length
        return {
            "fit": torch.tensor(fit),
            "check": torch.tensor(check),
            "held": torch.tensor(held),
        }


# ----------------------------------------------------------------------- solve
def gain_system(
    C: torch.Tensor, Z: torch.Tensor, N: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``G d = b`` for ``min_d ||(Z diag(d) - N diag(prior)) C^(1/2)||^2``.

    ``C`` is the reader's column Gram (``E^T E`` at the head, ``W^T W`` at a
    block).  The K-column -- or 2560-column -- target is never formed.
    """

    G = C * (Z.T @ Z)
    b = (C * (Z.T @ N)) @ prior
    scaled = N * prior[None, :]
    energy = float(((scaled @ C) * scaled).sum())
    return G, b, energy


def relative_sse(
    C: torch.Tensor, Z: torch.Tensor, N: torch.Tensor, d: torch.Tensor,
    prior: torch.Tensor, energy: float,
) -> float:
    """``||Z diag(d) - N diag(prior)||^2_C`` over the target's own energy."""

    diff = Z * d[None, :] - N * prior[None, :]
    return float(((diff @ C) * diff).sum() / energy)


def solve_gain(
    C: torch.Tensor,
    Z: torch.Tensor,
    N: torch.Tensor,
    prior: torch.Tensor,
    masks: dict[str, torch.Tensor],
    forced_lambda: float | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One diagonal, prior-anchored, lambda on the internal fit split.

    THE LAW is a VETO on update scale, never a vote: a rung above 1.0 is not
    admitted, and among the admitted rungs the one with the lowest check-split
    sum of squares wins.  `--lambda-relative` overrides the choice and is how the
    atlas walks the portfolio; the veto still applies to what it names.
    """

    width = int(C.shape[0])
    inner = masks["fit"] & ~masks["check"]
    reference = float(prior.norm())
    G_all, b_all, energy_all = gain_system(C, Z[masks["fit"]], N[masks["fit"]], prior)
    G_in, b_in, _ = gain_system(C, Z[inner], N[inner], prior)
    scale = float(torch.diagonal(G_all).mean())
    eye = torch.eye(width, dtype=torch.float64)

    # The identity gate, synthetic and exact: at Z = N the answer is the prior.
    G_id, b_id, _ = gain_system(C, N[masks["fit"]], N[masks["fit"]], prior)
    identity = torch.linalg.solve(
        G_id + 1e-6 * scale * eye, b_id + 1e-6 * scale * prior
    )
    identity_error = float((identity - prior).abs().max())
    if identity_error > IDENTITY_GATE_TOLERANCE:
        raise SystemExit(
            f"the identity gate fails: at Z = N the solve returns a diagonal "
            f"{identity_error:.3e} from the prior, and it must return the prior."
        )

    Z_ck, N_ck = Z[masks["check"]], N[masks["check"]]
    Z_hd, N_hd = Z[masks["held"]], N[masks["held"]]
    _, _, energy_ck = gain_system(C, Z_ck, N_ck, prior)
    _, _, energy_hd = gain_system(C, Z_hd, N_hd, prior)
    ladder: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        lam = rel * scale
        inner_solution = torch.linalg.solve(G_in + lam * eye, b_in + lam * prior)
        full = torch.linalg.solve(G_all + lam * eye, b_all + lam * prior)
        ladder.append(
            {
                "lambda_relative": rel,
                "update_scale": float((full - prior).norm() / reference),
                "check_relative_sse": relative_sse(
                    C, Z_ck, N_ck, inner_solution, prior, energy_ck
                ),
                "heldout_relative_sse": relative_sse(
                    C, Z_hd, N_hd, full, prior, energy_hd
                ),
                "solved": full,
            }
        )
    admitted = [row for row in ladder if row["update_scale"] <= 1.0]
    if not admitted:
        raise SystemExit(
            "no lambda on the ladder brings the diagonal's update_scale to 1.0; "
            "THE LAW refuses the install"
        )
    if forced_lambda is None:
        chosen = min(admitted, key=lambda row: row["check_relative_sse"])
        rule = "internal fit split (lowest check relative SSE among admitted rungs)"
    else:
        named = [row for row in admitted if row["lambda_relative"] == forced_lambda]
        if not named:
            raise SystemExit(
                f"--lambda-relative {forced_lambda} is not an admitted rung "
                f"(admitted: {[row['lambda_relative'] for row in admitted]})"
            )
        chosen = named[0]
        rule = f"FORCED by --lambda-relative {forced_lambda} (atlas rung)"
    report = {
        "lambda_relative": chosen["lambda_relative"],
        "lambda_rule": rule,
        "update_scale": chosen["update_scale"],
        "identity_gate_max_abs_error": identity_error,
        "prior_relative_sse_fit": relative_sse(
            C, Z[masks["fit"]], N[masks["fit"]], prior, prior, energy_all
        ),
        "solved_relative_sse_fit": relative_sse(
            C, Z[masks["fit"]], N[masks["fit"]], chosen["solved"], prior, energy_all
        ),
        "prior_relative_sse_heldout": relative_sse(
            C, Z_hd, N_hd, prior, prior, energy_hd
        ),
        "solved_relative_sse_heldout": chosen["heldout_relative_sse"],
        "rows_fit": int(masks["fit"].sum()),
        "rows_check": int(masks["check"].sum()),
        "rows_heldout": int(masks["held"].sum()),
        "ladder": [{k: v for k, v in row.items() if k != "solved"} for row in ladder],
    }
    return chosen["solved"], report


# ------------------------------------------------------------------ the readers
def reader_gram(layer: Any, which: str) -> torch.Tensor:
    """``W^T W`` for the module a block norm feeds.

    ``operator_norm`` feeds the attention projections (q, k and v share the
    normalised residual) at an attention block and ``conv.in_proj`` at a conv
    block; ``ffn_norm`` feeds ``w1`` and ``w3``.  Concatenating the readers that
    share one norm is not a choice -- they share the diagonal, so the objective
    is over their stacked rows.
    """

    if which == "ffn_norm":
        rows = [layer.feed_forward.w1.weight, layer.feed_forward.w3.weight]
    elif hasattr(layer, "self_attn") and getattr(layer, "is_attention_layer", False):
        attn = layer.self_attn
        rows = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
    else:
        rows = [layer.conv.in_proj.weight]
    W = torch.cat([r.data.to(torch.float64).cpu() for r in rows], dim=0)
    return W.T @ W


# ------------------------------------------------------------------- the build
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lambda-relative", type=float)
    ap.add_argument("--block-gains", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    started = time.monotonic()

    source = args.source
    source_receipts = sorted(source.glob("mx*-build-receipt.json"))
    if not source_receipts:
        raise SystemExit(f"{source} carries no mx*-build-receipt.json")
    source_build = json.loads(source_receipts[-1].read_text(encoding="utf-8"))
    excised = [int(x) for x in source_build.get("excised", [])]

    streams = Streams(source, args.device)
    child_model = streams.child
    parent_model = streams.parent
    embed = child_model.model.embed_tokens.weight.data.to(torch.float64).cpu()
    prior_final = child_model.model.embedding_norm.weight.data.to(torch.float64).cpu()
    parent_final = parent_model.model.embedding_norm.weight.data.to(torch.float64).cpu()
    if not torch.equal(prior_final, parent_final):
        raise SystemExit(
            "the source child's final gain is not the parent's; this solve's prior "
            "assumes the sealed child carries the parent's diagonal verbatim."
        )

    block_report: list[dict[str, Any]] = []
    if args.block_gains:
        # Conditioned in depth order: each install moves the child's stream, so
        # the child is re-captured per endpoint.  The parent is captured once
        # per endpoint set -- its stream does not move -- and both captures are
        # taken through the SAME position-aligned pair of id streams.
        surviving = [
            l for l in range(len(parent_model.model.layers)) if l not in excised
        ]
        parent_modules = {}
        for position, parent_layer in enumerate(surviving):
            layer = parent_model.model.layers[parent_layer]
            parent_modules[f"{position}.operator_norm"] = layer.operator_norm
            parent_modules[f"{position}.ffn_norm"] = layer.ffn_norm
        parent_capture = streams.capture(parent_model, parent_modules, "parent_ids")
        masks = streams.masks(parent_capture)
        for position, parent_layer in enumerate(surviving):
            child_layer = child_model.model.layers[position]
            for which in ("operator_norm", "ffn_norm"):
                name = f"{position}.{which}"
                module = getattr(child_layer, which)
                capture = streams.capture(child_model, {name: module}, "child_ids")
                Z = capture["rows"][name]
                N = parent_capture["rows"][name]
                C = reader_gram(child_layer, which)
                prior = module.weight.data.to(torch.float64).cpu()
                solved, report = solve_gain(
                    C, Z, N, prior, masks, args.lambda_relative
                )
                module.weight.data.copy_(solved.to(module.weight.dtype))
                report.update(
                    {"endpoint": name, "parent_layer": parent_layer, "reader": which}
                )
                block_report.append(report)
                log(
                    f"  {name} (parent L{parent_layer:02d}) lam "
                    f"{report['lambda_relative']:g} update "
                    f"{report['update_scale']:.4f} relSSE fit "
                    f"{report['prior_relative_sse_fit']:.4f} -> "
                    f"{report['solved_relative_sse_fit']:.4f} held "
                    f"{report['prior_relative_sse_heldout']:.4f} -> "
                    f"{report['solved_relative_sse_heldout']:.4f}"
                )

    # The head last, always: it reads the residual every block above it wrote,
    # so a head solved before the blocks is solved against a stream that no
    # longer exists by the time it runs.
    head_capture_parent = streams.capture(
        parent_model, {"final": parent_model.model.embedding_norm}, "parent_ids"
    )
    head_capture_child = streams.capture(
        child_model, {"final": child_model.model.embedding_norm}, "child_ids"
    )
    masks = streams.masks(head_capture_child)
    C_head = embed.T @ embed
    solved_final, head_report = solve_gain(
        C_head,
        head_capture_child["rows"]["final"],
        head_capture_parent["rows"]["final"],
        prior_final,
        masks,
        args.lambda_relative,
    )
    child_model.model.embedding_norm.weight.data.copy_(
        solved_final.to(child_model.model.embedding_norm.weight.dtype)
    )
    log(
        f"  final gain: lam {head_report['lambda_relative']:g} update "
        f"{head_report['update_scale']:.4f} logit relSSE fit "
        f"{head_report['prior_relative_sse_fit']:.4f} -> "
        f"{head_report['solved_relative_sse_fit']:.4f}, held-out "
        f"{head_report['prior_relative_sse_heldout']:.4f} -> "
        f"{head_report['solved_relative_sse_heldout']:.4f}"
    )

    # ------------------------------------------------------------------ write
    outdir = args.child
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, outdir / name)
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(source / "model.safetensors"))
    tensors["model.embedding_norm.weight"] = (
        child_model.model.embedding_norm.weight.data.to("cpu", tensors[
            "model.embedding_norm.weight"
        ].dtype).clone()
    )
    if args.block_gains:
        for position in range(len(child_model.model.layers)):
            layer = child_model.model.layers[position]
            for which in ("operator_norm", "ffn_norm"):
                key = f"model.layers.{position}.{which}.weight"
                tensors[key] = (
                    getattr(layer, which)
                    .weight.data.to("cpu", tensors[key].dtype)
                    .clone()
                )
    save_file(tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})
    live = int(sum(int(t.numel()) for t in tensors.values()))
    if live != int(source_build["child_parameters"]):
        raise SystemExit(
            f"the gain solve changed the parameter count: {live} against the "
            f"source's {source_build['child_parameters']}.  It overwrites values."
        )

    touched = 1024 + (30_464 if args.block_gains else 0)
    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx24_termgain.py",
        "tag": str(args.tag),
        "arm": (
            "terminal-read re-solve on the child's own token distribution"
            + (" + 29-diagonal block gains" if args.block_gains else "")
        ),
        "source_model": str(source),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "source_tensors_sha256": source_build.get("child_tensors_sha256"),
        "child_model": str(outdir),
        "child_tensors_sha256": hashlib.sha256(
            (outdir / "model.safetensors").read_bytes()
        ).hexdigest(),
        "child_parameters": live,
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": live / PARENT_PARAMS,
        "excised": excised,
        "residual_basis": source_build.get("residual_basis"),
        "axes": list(source_build.get("axes", []))
        + [
            {
                "axis": "rms_norm_gain",
                "kind": "terminal-read"
                + ("+per-reader" if args.block_gains else ""),
                "sites": ["model.embedding_norm"]
                + (
                    [f"model.layers.{r['endpoint']}" for r in block_report]
                    if args.block_gains
                    else []
                ),
                "config_fields": [],
                "selection": (
                    "no selection: the diagonals that already exist are "
                    "overwritten in place, so the parameter count, every tensor "
                    "shape and config.json are unchanged"
                ),
                "solve": (
                    "prior-anchored ridge on the diagonal, closed form.  min_d "
                    "||Z diag(d) E^T - N diag(g) E^T||_F^2 + lambda ||d - g||^2 "
                    "with Z the CHILD's RMS-normalised terminal residual on its "
                    "OWN segmentation, N the PARENT's on the position-aligned "
                    "injection of that same segmentation, E the child's tied "
                    "table and g the parent's gain (also the prior).  Normal "
                    "equations [(E^T E) o (Z^T Z)] d = [(E^T E) o (Z^T N)] g, one "
                    "1024x1024 SPD solve, no K-column target ever formed.  "
                    "lambda on the internal fit split (wave-18 law), THE LAW a "
                    "veto on update scale.  Identity gate: at Z = N the solve "
                    "returns the prior exactly, asserted synthetically every run"
                ),
                "evidence": (
                    "the campaign's 24 captured batches, RE-TOKENIZED into the "
                    "child's own vocabulary; the parent is run on the injection "
                    "of that same segmentation, so the two streams are position-"
                    "aligned over identical text.  No new corpus, no new prompt"
                ),
                "parameters_removed": 0,
                "parameters_overwritten": touched,
                "heldout": {
                    "final_gain_relative_logit_sse": {
                        "prior": head_report["prior_relative_sse_heldout"],
                        "solved": head_report["solved_relative_sse_heldout"],
                    },
                    "block_gains": [
                        {
                            "endpoint": r["endpoint"],
                            "prior": r["prior_relative_sse_heldout"],
                            "solved": r["solved_relative_sse_heldout"],
                        }
                        for r in block_report
                    ],
                },
            }
        ],
        "pair_kkt_finding": (
            "S4 asks for one convex quadratic over BOTH folds' free tensors.  On "
            "this pair there are none: mx24_vocab.py's base arm is a gather and "
            "MX16-X13A's build receipt carries keep_ff null, so the excision took "
            "the per-site loop's `continue` branch and re-solved nothing either.  "
            "The free set had to be CREATED, and the only set that is already "
            "counted, exportable in a stock config and coupled to both folds is "
            "the RMSNorm diagonals -- the excision decides the residual that "
            "arrives at the head, the coreset decides the table it is read "
            "through, and one system carries both."
        ),
        "final_gain": head_report,
        "block_gains": block_report,
        "retokenized_evidence": streams.meta,
        "evidence": {
            "trace_cache_key": str(mx24_retok.TRACE.name),
            "fit_rows": head_report["rows_fit"],
            "heldout_rows": head_report["rows_heldout"],
            "internal_check_rows": head_report["rows_check"],
            "parent_model_forward_calls": streams.forward_calls,
            "split_authority": "matrix-campaign-capture-manifest",
            "conditioned": True,
        },
        "solve": {
            "gradients": 0,
            "optimizer_steps": 0,
            "closed_form": "Hadamard normal equations, one SPD Cholesky per diagonal",
        },
        "identity_gate": {
            "kind": "algebraic",
            "assertion": "at Z = N the prior-anchored solve returns the prior",
            "max_abs_error": head_report["identity_gate_max_abs_error"],
            "tolerance": IDENTITY_GATE_TOLERANCE,
            "passed": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (outdir / "mx24-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=1))
    log(
        f"{args.tag}: {live:,} parameters, {live / PARENT_PARAMS:.6f}x, "
        f"{touched:,} values overwritten, 0 added"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
