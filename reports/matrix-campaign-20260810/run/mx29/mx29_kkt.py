#!/usr/bin/env python3
"""GLOBAL-KKT: one convex terminal re-solve over EVERY installed writer, jointly.

`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #4, ADDENDUM 16.2, A1 scope confirmed
by Cyril.  This is the campaign's only attack on the tax constant itself, and its
NULL is as load-bearing as its success: a clean zero on both pilots across a full
lambda ladder seals *no closed-form objective, local or global, repairs
composition through the terminal read*, which is the ceiling evidence a
defensible phase-1 exit needs.  `PLAN_GAPS_2026-08-14.md` GAP 1 correctly narrows
that wording -- the residual-corrected sequential family is a different branch and
stays open by name.

**What is actually solved, stated exactly, because the honest version is smaller
than the slogan.**  "Solve jointly over all folded writer tensors" as literally
written is a dense system over every entry of fourteen 896x1024 matrices --
1.3e7 unknowns, a 1.6e14-entry normal matrix.  That is not a solve, it is a
sentence.  The tractable exact form of the same idea, and the one the S4 spec's
"per-writer prior anchors and trust regions" already implies, is a **per-writer
output gain solved jointly in the reader metric**:

    h(e) = h_child + sum_w  e_w (*) y_w

where ``y_w`` is writer w's own contribution to the residual stream, ``e_w`` is
its deviation from identity, and the sum is over every installed writer at once.
Two arms:

* ``--arm scalar``    one number per writer.  A W x W system.
* ``--arm diagonal``  1,024 numbers per writer.  A (1024 W) x (1024 W) system
  with genuine cross-writer blocks ``sum_rows diag(u_a) C diag(u_b)`` -- the
  downstream cross term M1's autopsy named as the object no per-component
  objective can see.  This is the whole point: the off-diagonal blocks are the
  interference.

Zero parameters are added.  ``e_w`` is absorbed into writer w's own rows on the
way out, so the child is a standard checkpoint at byte-identical storage, and the
``lambda -> infinity`` rung returns ``e = 0`` and therefore a **byte-identical**
copy of the unrepaired child.  That rung is built and checked, not asserted.

**The frozen-gate approximation is MEASURED, never assumed.**  Scaling a writer
changes what every later block reads, so the linear response is exact only under
frozen gates, attention patterns and norm statistics.  So this file runs the
rung-0 discriminator the plan asks for: it installs the solved gains, takes a
real forward pass, and compares the ACTUAL final residual against the one the
linearisation predicted.  If the linearisation cannot predict its own repair, the
mechanism dies for one capture and the number that killed it is in the receipt.

No gradient of any loss with respect to any parameter is taken anywhere in this
file.  Every solve is ``torch.linalg`` on a Gram assembled from cached forward
activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx29_termgain as tg  # noqa: E402

PARENT_PARAMS = tg.PARENT_PARAMS
SCHEMA = "neural-forge-matrix-wave29-global-kkt-build-receipt-v1"
LAMBDA_LADDER = (1e9, 10.0, 1.0, 0.3, 0.1)


def log(message: str) -> None:
    print(message, flush=True)


def writer_modules(model: Any, excised: list[int]) -> dict[str, Any]:
    """Every writer that is exactly linear under frozen gates.

    ``feed_forward.w2`` and ``self_attn.out_proj`` write into the residual stream
    through no nonlinearity of their own: the gate that multiplies them is
    computed upstream, so with that gate frozen their contribution is linear in
    their own rows.  ``w1``/``w3``/``q``/``k``/``v`` are NOT in this set and are
    not solved here -- the restriction is the A1 ruling's, not a convenience.
    """

    out: dict[str, Any] = {}
    for position, layer in enumerate(model.model.layers):
        out[f"L{position:02d}.w2"] = layer.feed_forward.w2
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(layer, "is_attention_layer", False):
            out[f"L{position:02d}.o"] = attn.out_proj
    return out


@torch.no_grad()
def capture_writers(streams: tg.Streams, model: Any, modules: dict[str, Any],
                    final_module: Any, key: str) -> dict[str, Any]:
    """Writer OUTPUTS plus the final-norm input, on one id stream.

    ``Streams.capture`` records module INPUTS, which is what a diagonal on a norm
    needs.  A writer gain needs the module's OUTPUT, so this takes its own hooks
    -- and takes them in the same pass as the final residual, because the two
    have to come from one forward or they are not the same forward.
    """

    state: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in modules.items():
        handles.append(module.register_forward_hook(
            (lambda n: lambda m, args, output: state.__setitem__(
                n, (output[0] if isinstance(output, tuple) else output).detach()
            ))(name)
        ))
    handles.append(final_module.register_forward_pre_hook(
        (lambda m, args, kwargs: state.__setitem__(
            "final", (args[0] if args else kwargs["hidden_states"]).detach()
        )), with_kwargs=True
    ))
    blocks: dict[str, list[torch.Tensor]] = {n: [] for n in list(modules) + ["final"]}
    splits: list[str] = []
    lengths: list[int] = []
    indices: list[int] = []
    try:
        for row in streams.rows:
            ids = torch.tensor([row[key]], dtype=torch.long, device=streams.device)
            model(input_ids=ids, use_cache=False)
            streams.forward_calls += 1
            for name in blocks:
                blocks[name].append(state[name][0].detach().to(torch.float64).cpu())
            # ONE row per BATCH, expanded to positions by `Streams.masks`.  The
            # first version of this file appended one split per position and no
            # indices or lengths at all, and `masks` refused on the missing key --
            # after both captures had already been paid for.  The bookkeeping is
            # the same bookkeeping `Streams.capture` keeps, so it is kept the
            # same way rather than a second way.
            splits.append(row["split"])
            lengths.append(int(state["final"].shape[1]))
            indices.append(row["batch_index"])
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {
        "rows": {n: torch.cat(v, dim=0) for n, v in blocks.items()},
        "splits": splits,
        "lengths": lengths,
        "indices": indices,
    }


def build_system(C: torch.Tensor, U: dict[str, torch.Tensor], R: torch.Tensor,
                 names: list[str], scalar: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The joint normal equations, cross-writer blocks included.

    For the diagonal arm, block (a, b) is ``sum_t u_a[t] u_b[t]^T (*) C`` -- the
    same Hadamard identity `mx29_termgain.gain_system` uses for one writer,
    carried to a pair.  For the scalar arm it contracts to a number.
    """

    n = len(names)
    if scalar:
        G = torch.zeros((n, n), dtype=torch.float64)
        b = torch.zeros(n, dtype=torch.float64)
        CU = {a: U[a] @ C for a in names}
        for i, a in enumerate(names):
            b[i] = -float((CU[a] * R).sum())
            for j, bn in enumerate(names[i:], start=i):
                G[i, j] = G[j, i] = float((CU[a] * U[bn]).sum())
        return G, b
    width = int(C.shape[0])
    G = torch.zeros((n * width, n * width), dtype=torch.float64)
    b = torch.zeros(n * width, dtype=torch.float64)
    for i, a in enumerate(names):
        b[i * width:(i + 1) * width] = -((U[a] * (R @ C)).sum(dim=0))
        for j, bn in enumerate(names[i:], start=i):
            block = (U[a].T @ U[bn]) * C
            G[i * width:(i + 1) * width, j * width:(j + 1) * width] = block
            if j != i:
                G[j * width:(j + 1) * width, i * width:(i + 1) * width] = block.T
    return G, b


def objective(C: torch.Tensor, U: dict[str, torch.Tensor], R: torch.Tensor,
              names: list[str], e: torch.Tensor, width: int, scalar: bool) -> float:
    resid = R.clone()
    for i, name in enumerate(names):
        if scalar:
            resid = resid + float(e[i]) * U[name]
        else:
            resid = resid + U[name] * e[i * width:(i + 1) * width]
    return float(((resid @ C) * resid).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-cell-receipt", type=Path)
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", choices=("scalar", "diagonal"), default="diagonal")
    ap.add_argument("--lambda-relative", type=float)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--inventory-only", action="store_true",
                    help="step 0a: report the free set and build nothing")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    started = time.monotonic()

    source_receipts = sorted(args.source.glob("mx*-build-receipt.json"))
    if not source_receipts:
        raise SystemExit(f"{args.source} carries no mx*-build-receipt.json")
    source_build = json.loads(source_receipts[-1].read_text(encoding="utf-8"))
    excised = [int(x) for x in source_build.get("excised", [])]

    streams = tg.Streams(args.source, args.device)
    child, parent = streams.child, streams.parent
    modules = writer_modules(child, excised)
    names = sorted(modules)
    log(f"installed writers in the linear-under-frozen-gates set: {len(names)}")
    log("  " + ", ".join(names))

    # STEP 0a -- the free-set inventory, on paper.  The mx28 finding is that a
    # vocab-times-excision stack is ZERO-SOLVE: its free set is empty, and a
    # joint solve over an empty set is not a null result, it is a category
    # error.  So the inventory is reported before any capture is paid for.
    axes = source_build.get("axes", [])
    solved_axes = [a for a in axes if a.get("solve")]
    inventory = {
        "writers_linear_under_frozen_gates": len(names),
        "writer_names": names,
        "source_axes": [a.get("axis") for a in axes],
        "source_axes_with_a_solve": [a.get("axis") for a in solved_axes],
        "free_set_is_empty": len(names) == 0,
    }
    if args.inventory_only:
        out = {"schema": "neural-forge-matrix-wave29-kkt-inventory-v1",
               "source": str(args.source), "tag": args.tag, "inventory": inventory}
        print(json.dumps(out, indent=1))
        if args.report:
            args.report.write_text(json.dumps(out, indent=1))
        return 0
    if not names:
        raise SystemExit("REFUSED: the free set is empty; there is nothing to solve jointly")

    embed = child.model.embed_tokens.weight.data.to(torch.float64).cpu()
    C = embed.T @ embed
    final_gain = child.model.embedding_norm.weight.data.to(torch.float64).cpu()
    # The reader metric seen through the final diagonal: the head reads
    # gain (*) normalise(h), so the metric on h is diag(g) C diag(g).
    C = C * torch.outer(final_gain, final_gain)

    log("capturing the child's writer outputs and final residual")
    child_capture = capture_writers(streams, child, modules,
                                    child.model.embedding_norm, "child_ids")
    log("capturing the parent's final residual on the aligned stream")
    parent_capture = capture_writers(streams, parent, {},
                                     parent.model.embedding_norm, "parent_ids")
    # The capture dict goes to `masks` WHOLE.  Handing it a hand-assembled
    # subset is what dropped the batch indices and lengths, and the wave-18
    # internal check split has to be the same split the single-diagonal solves
    # use or a KKT rung is not comparable to the rung it is meant to beat.
    masks = streams.masks(child_capture)

    H = child_capture["rows"]["final"]
    N = parent_capture["rows"]["final"]
    if H.shape != N.shape:
        raise SystemExit(f"streams are not position-aligned: {tuple(H.shape)} vs {tuple(N.shape)}")
    eps = float(child.config.norm_eps)
    sigma = H.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()   # FROZEN
    Z = H * sigma
    Nn = N * N.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
    R_all = Z - Nn
    U_all = {n: child_capture["rows"][n] * sigma for n in names}
    width = int(C.shape[0])
    scalar = args.arm == "scalar"

    fit, check, held = masks["fit"], masks["check"], masks["held"]
    inner = fit & ~check
    def sub(mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {n: U_all[n][mask] for n in names}, R_all[mask]

    U_in, R_in = sub(inner)
    U_fit, R_fit = sub(fit)
    U_ck, R_ck = sub(check)
    U_hd, R_hd = sub(held)

    log(f"assembling the joint system ({args.arm}, "
        f"{len(names) * (1 if scalar else width)} unknowns)")
    G_in, b_in = build_system(C, U_in, R_in, names, scalar)
    G_fit, b_fit = build_system(C, U_fit, R_fit, names, scalar)
    scale = float(torch.diagonal(G_fit).mean())
    eye = torch.eye(G_fit.shape[0], dtype=torch.float64)

    prior_ck = objective(C, U_ck, R_ck, names, torch.zeros(G_fit.shape[0], dtype=torch.float64), width, scalar)
    prior_hd = objective(C, U_hd, R_hd, names, torch.zeros(G_fit.shape[0], dtype=torch.float64), width, scalar)

    rungs: list[dict[str, Any]] = []
    ladder = (args.lambda_relative,) if args.lambda_relative else LAMBDA_LADDER
    for rel in ladder:
        lam = rel * scale
        e_in = torch.linalg.solve(G_in + lam * eye, b_in)
        e_full = torch.linalg.solve(G_fit + lam * eye, b_fit)
        rungs.append({
            "lambda_relative": rel,
            "update_norm": round(float(e_full.norm()), 8),
            "check_relative_sse": round(objective(C, U_ck, R_ck, names, e_in, width, scalar) / max(prior_ck, 1e-30), 6),
            "heldout_relative_sse": round(objective(C, U_hd, R_hd, names, e_full, width, scalar) / max(prior_hd, 1e-30), 6),
            "solved": e_full,
        })
        log(f"  lam {rel:g}: |e| {rungs[-1]['update_norm']:.6f} "
            f"check {rungs[-1]['check_relative_sse']:.6f} "
            f"held {rungs[-1]['heldout_relative_sse']:.6f}")

    chosen = min(rungs, key=lambda r: r["check_relative_sse"]) if not args.lambda_relative else rungs[0]
    e = chosen["solved"]
    log(f"chosen lambda_relative {chosen['lambda_relative']:g}")

    # Install: absorbed into each writer's own OUTPUT rows.  Zero parameters
    # added, every shape unchanged, config.json untouched.
    for i, name in enumerate(names):
        module = modules[name]
        gain = (1.0 + (float(e[i]) if scalar else e[i * width:(i + 1) * width]))
        g = torch.as_tensor(gain, dtype=torch.float64)
        w = module.weight.data.to(torch.float64).cpu()
        module.weight.data.copy_((w * g.unsqueeze(1) if g.dim() else w * g).to(module.weight.dtype))

    # RUNG 0 DISCRIMINATOR: does the linearisation predict its own repair?
    log("rung-0 discriminator: real forward pass against the linear prediction")
    actual = capture_writers(streams, child, {}, child.model.embedding_norm, "child_ids")
    H_actual = actual["rows"]["final"]
    predicted = H.clone()
    for i, name in enumerate(names):
        contribution = child_capture["rows"][name]
        predicted = predicted + (contribution * (float(e[i]) if scalar else e[i * width:(i + 1) * width]))
    lin_error = float((H_actual - predicted).norm() / max(float((predicted - H).norm()), 1e-30))
    lin_rel = float((H_actual - predicted).norm() / max(float(H.norm()), 1e-30))
    log(f"  linearisation error / repair size = {lin_error:.4f}; "
        f"error / |h| = {lin_rel:.6f}")

    outdir = args.child
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja", "vocab.json", "merges.txt"):
        if (args.source / name).is_file():
            shutil.copyfile(args.source / name, outdir / name)
    from safetensors.torch import load_file, save_file
    tensors = load_file(str(args.source / "model.safetensors"))
    for name in names:
        key = {
            f"L{p:02d}.w2": f"model.layers.{p}.feed_forward.w2.weight"
            for p in range(len(child.model.layers))
        }.get(name) or f"model.layers.{int(name[1:3])}.self_attn.out_proj.weight"
        tensors[key] = modules[name].weight.data.to("cpu", tensors[key].dtype).clone()
    save_file(tensors, str(outdir / "model.safetensors"), metadata={"format": "pt"})
    live = int(sum(int(t.numel()) for t in tensors.values()))
    if live != int(source_build["child_parameters"]):
        raise SystemExit(
            f"the joint solve changed the parameter count: {live} against "
            f"{source_build['child_parameters']}.  It overwrites values."
        )
    digest = hashlib.sha256((outdir / "model.safetensors").read_bytes()).hexdigest()
    identical = digest == source_build.get("child_tensors_sha256")

    receipt = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx29_kkt.py",
        "tag": str(args.tag),
        "arm": (
            f"GLOBAL-KKT joint terminal re-solve, {args.arm} gain per installed "
            f"writer, {len(names)} writers solved AT ONCE in the reader metric"
        ),
        "addendum": "16.2 (A1 scope: all-writer form ruled legal)",
        "inventory": inventory,
        "source_model": str(args.source),
        "source_cell_receipt": str(args.source_cell_receipt or ""),
        "source_tensors_sha256": source_build.get("child_tensors_sha256"),
        "child_model": str(outdir),
        "child_tensors_sha256": digest,
        "byte_identical_to_source": identical,
        "child_parameters": live,
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": live / PARENT_PARAMS,
        "parameters_added": 0,
        "excised": excised,
        "lambda_ladder": [
            {k: v for k, v in r.items() if k != "solved"} for r in rungs
        ],
        "lambda_chosen": chosen["lambda_relative"],
        "rung0_discriminator": {
            "linearisation_error_over_repair": round(lin_error, 6),
            "linearisation_error_over_state": round(lin_rel, 8),
            "reading": (
                "the linearisation predicts its own repair (error is a small "
                "fraction of the repair it made)" if lin_error < 0.5 else
                "THE LINEARISATION CANNOT PREDICT ITS OWN REPAIR: the frozen-gate "
                "response is not the response, and a null on this pilot is a null "
                "about the approximation as much as about the objective"
            ),
        },
        "forward_calls": streams.forward_calls,
        "retokenized_evidence": streams.meta,
        "axes": list(source_build.get("axes", [])) + [{
            "axis": "writer_output_gain",
            "kind": f"global-kkt-{args.arm}",
            "sites": names,
            "config_fields": [],
            "selection": (
                "no selection: the gains are absorbed into writer rows that already "
                "exist, so the parameter count, every tensor shape and config.json "
                "are unchanged"
            ),
            "solve": (
                "one joint convex quadratic over every installed writer at once, in "
                "the reader metric diag(g) embed^T embed diag(g), prior-anchored at "
                "identity, lambda chosen on a wave-18 internal split.  The "
                "cross-writer blocks ARE the downstream cross term that no "
                "per-component objective can see."
            ),
        }],
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    (outdir / "mx29-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    if args.report:
        args.report.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("axes", "retokenized_evidence", "inventory")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
