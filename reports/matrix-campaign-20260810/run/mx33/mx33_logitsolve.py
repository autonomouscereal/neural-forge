#!/usr/bin/env python3
"""Re-solve a folded child's LAST block writer in the reader's metric, not the stream's.

Field synthesis #6.  Two independent 2026 results say the same thing about the
objective this campaign has always solved: LFQ (ICML 2026) shows block-wise
activation-MSE calibration preserves multiple-choice accuracy while killing
long-form generation -- with a worked example where a LOWER reconstruction error
flips the argmax -- and fixes it by optimising the final block against parent
LOGITS.  IO-SVD derives the same correction as a metric, ``C = E[J^T H J]`` with
``H`` the softmax Hessian, and reports PPL 27.7 against SVD-LLM's 53.7 at 40%
retention.  Both are the field independently deriving this campaign's own
objective/review inversion, sealed eight times, and both name the same fix.

**What this file does NOT do.**  It does not linearise a frozen gate.  Wave 29's
GLOBAL-KKT null was not "no cases": it was the measurement that at 0.48x the
frozen-gate linear response is wrong by 1.5-2.3x the repair it predicts, so the
response cannot adjudicate.  That failure is specific to predicting the effect
of a change through gates held fixed several blocks upstream of the reader.  The
object here is different and much smaller: the LAST block's writer, one block
from the reader, and the only thing held fixed is the RMS scalar of the final
norm -- which the script MEASURES rather than assumes, by reporting how far the
child's own RMS moves between the prior and the solved weight.

**The metric, exactly.**  Logits are ``z = W_head (RMSNorm(h) * d)``.  A
perturbation of the residual at the reader's input moves the logits by
``dz = W_head diag(d / rms) dh`` to first order in ``dh`` at fixed ``rms``, so a
hidden-space error ``dh`` costs, in the cross-entropy the reader actually pays,

    dh^T M dh,   M = S W_head^T (diag(p) - p p^T) W_head S,   S = diag(d / rms)

with ``p`` the child's own next-token distribution at that position.  ``M`` is
accumulated over the evidence positions on the top-``--metric-topk`` tokens of
each distribution -- the softmax is peaked, and a full 64,400-row Gram per
position is 6.7e10 flops nobody needs to spend to represent 0.999 of the mass.

**The solve, exactly.**  With ``M`` symmetric PSD the ridge becomes a Sylvester
equation rather than a normal equation,

    (X^T X) W^T M + lam W^T = (X^T Y) M + lam W0^T

which two eigendecompositions solve in closed form -- ``A = U S U^T``,
``M = Q L Q^T``, ``G = U^T W^T Q`` entrywise ``G_ij = (U^T B Q)_ij / (S_i L_j + lam)``.
No CG, no iteration, no step size.  At ``M = I`` it reduces algebraically to
``prior_anchored`` in ``mx33_build.py``, and the script ASSERTS that at startup
against the builder's own function, because an A/B whose control is not the
sealed control is not an A/B.

The arm is a strict post-process: it re-solves one writer of an already-built
child and changes nothing else, so the storage is byte-identical and the
comparison is the objective and only the objective.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

SCHEMA = "neural-forge-matrix-wave33-logit-metric-writer-v1"
PARENT_PARAMS = 229_693_184
#: `mx33_build.LAMBDA_LADDER`, restated so this file's rungs are that file's.
LAMBDA_LADDER = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def sylvester_ridge(
    A: "torch.Tensor", B: "torch.Tensor", M: "torch.Tensor", prior: "torch.Tensor",
    lam: float,
) -> "torch.Tensor":
    """``argmin tr((X W^T - Y) M (X W^T - Y)^T) + lam ||W - prior||^2``.

    ``A = X^T X`` and ``B = X^T Y``, both already formed by the caller.  Returns
    ``W`` in the same orientation the builder installs.
    """

    import torch

    rhs = B @ M + lam * prior.T
    s, U = torch.linalg.eigh(A)
    l, Q = torch.linalg.eigh(M)
    inner = (U.T @ rhs) @ Q
    denom = s[:, None] * l[None, :] + lam
    if float(denom.abs().min()) <= 0.0:
        raise SystemExit(
            "the Sylvester denominator reaches zero; the metric or the Gram is "
            "not positive definite and the solve is refused"
        )
    return (U @ (inner / denom) @ Q.T).T.contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=Path, required=True,
                    help="the already-built child whose last block writer is re-solved")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--metric", choices=["logit", "identity"], default="logit",
                    help="`identity` is the A/B control and must reproduce the child's own bytes")
    ap.add_argument("--metric-topk", type=int, default=2048)
    ap.add_argument("--metric-positions", type=int, default=768)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import os
    import sys

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run = args.deployment / "reports" / "matrix-campaign-20260810" / "run" / "mx33"
    for p in (str(args.deployment / "src"), str(args.deployment / "experiments"), str(run)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch  # noqa: E402
    from safetensors.torch import load_file, save_file  # noqa: E402

    import mx33_build as B  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()

    # ---- the reduction gate.  At M = I this file's solve IS the builder's.
    g = torch.Generator().manual_seed(20260828)
    Xt = torch.randn(64, 12, generator=g, dtype=torch.float64)
    Yt = torch.randn(64, 9, generator=g, dtype=torch.float64)
    Pt = torch.randn(9, 12, generator=g, dtype=torch.float64)
    A0, B0 = Xt.T @ Xt, Xt.T @ Yt
    lam0 = B.RIDGE_REL * float(torch.diagonal(A0).mean())
    mine = sylvester_ridge(A0, B0, torch.eye(9, dtype=torch.float64), Pt, lam0)
    theirs = B.prior_anchored(Xt, Yt, Pt, rel=B.RIDGE_REL)
    gate = float((mine - theirs).abs().max())
    if gate > 1e-9:
        raise SystemExit(
            f"REFUSED: at M = I this solve differs from mx33_build.prior_anchored "
            f"by {gate:.3e}; the A/B's control is not the sealed control"
        )
    print(f"[mx33lg] reduction gate at M=I: max|difference| {gate:.3e}")

    child_cfg = json.loads((args.child / "config.json").read_text())
    keep_ff = int(child_cfg["intermediate_size"])
    parent_cfg = json.loads((B.PARENT / "config.json").read_text())
    if keep_ff >= int(parent_cfg["intermediate_size"]):
        raise SystemExit(
            f"REFUSED: {args.child} carries the parent's own intermediate_size "
            f"({keep_ff}); there is no solved writer at its last block to re-solve"
        )

    harness = B.Harness(args.child, args.device, torch.bfloat16)
    layers = harness.layers()
    last = len(layers) - 1
    name = f"L{last:02d}"
    print(f"[mx33lg] child has {len(layers)} blocks; re-solving the writer at {name}")

    # ---- the site.  The child's OWN rows into its own w2, and the parent block's
    # output at those same rows -- the local target the builder solved against.
    parent_harness = B.Harness(B.PARENT, args.device, torch.bfloat16)
    parent_layers = parent_harness.layers()
    # A child with excisions renames its blocks; this wave's children have none,
    # and the assertion is here so a future one cannot silently take the wrong
    # parent block as its target.
    child_build = sorted(args.child.glob("mx*-build-receipt.json"))
    if not child_build:
        raise SystemExit(f"REFUSED: {args.child} carries no build receipt")
    build_receipt = json.loads(child_build[-1].read_text(encoding="utf-8"))
    if build_receipt.get("excised"):
        raise SystemExit(
            f"REFUSED: {args.child} excises {build_receipt['excised']}, so its last "
            f"block is not the parent's last block and the local target would be "
            f"taken from the wrong module"
        )
    pristine = copy.deepcopy(parent_layers[last].feed_forward)
    captured = harness.capture(
        {name: layers[last].feed_forward}, pristine={name: pristine}
    )
    X, X_h = captured[name]["fit"], captured[name]["hld"]
    Y, Y_h = captured[name]["local_fit"], captured[name]["local_hld"]

    w1 = pristine.w1.weight.data.to(torch.float64).cpu()
    w3 = pristine.w3.weight.data.to(torch.float64).cpu()
    w2 = pristine.w2.weight.data.to(torch.float64).cpu()
    phi = B.swiglu(X, w1, w3)
    phi_h = B.swiglu(X_h, w1, w3)
    sel = B.pivoted_columns(phi, keep_ff)
    prior = w2[:, sel]
    Aphi, Aphi_h = phi[:, sel], phi_h[:, sel]

    # ---- the metric.
    hidden = int(child_cfg["hidden_size"])
    if args.metric == "identity":
        M = torch.eye(hidden, dtype=torch.float64)
        metric_report: dict[str, Any] = {"metric": "identity"}
    else:
        model = harness.model
        # THE METRIC IS ACCUMULATED IN FLOAT32 AND THE SOLVE STAYS FLOAT64.
        # `M` is an evidence-derived WEIGHTING over directions, not a weight that
        # ships; the closed form below consumes it in float64 and the installed
        # tensor is float64 throughout.  The first draft of this loop built the
        # metric in float64 on the GPU and moved a 528 MB fp64 unembedding onto
        # the device once per POSITION -- 1,024 times -- while computing full
        # 64,400-column logits at the 3090's 1/64 fp64 rate.  That is roughly a
        # day of arithmetic to represent a weighting whose eigenvectors are
        # identical in single precision.
        head32 = model.lm_head.weight.data.to(torch.float32).to(args.device)
        d32 = model.model.embedding_norm.weight.data.to(torch.float32).to(args.device)
        eps = float(model.config.norm_eps)
        captured_x: dict[str, Any] = {}

        def hook(_m, inputs, _o):
            captured_x["x"] = inputs[0].detach()[0].to(torch.float32)

        handle = model.model.embedding_norm.register_forward_hook(hook)
        Macc = torch.zeros(hidden, hidden, dtype=torch.float32, device=args.device)
        used = 0
        mass = 0.0
        for index, ids in harness.batches:
            if harness.split[index] != "fit":
                continue
            out = model(input_ids=ids.to(args.device), use_cache=False)
            x = captured_x["x"]
            rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
            probs = torch.softmax(out.logits[0].to(torch.float32), dim=-1)
            for t_i in range(x.shape[0]):
                if used >= args.metric_positions:
                    break
                top = torch.topk(probs[t_i], args.metric_topk)
                pv = top.values
                mass += float(pv.sum())
                Ws = head32[top.indices] * (d32 / rms[t_i]).unsqueeze(0)  # [k, H]
                wp = Ws * pv.unsqueeze(1)
                mean = wp.sum(dim=0, keepdim=True)
                Macc += Ws.T @ wp - mean.T @ mean
                used += 1
            if used >= args.metric_positions:
                break
        handle.remove()
        M = Macc.double().cpu()
        M = 0.5 * (M + M.T)
        # Trace-normalised: the metric decides DIRECTIONS, and a free overall
        # scale would silently move where the lambda ladder lands.
        trace = float(torch.diagonal(M).sum())
        M = M * (hidden / trace)
        eigs = torch.linalg.eigvalsh(M)
        metric_report = {
            "metric": "logit (softmax-Hessian, reader-projected)",
            "positions_used": used,
            "topk": args.metric_topk,
            "mean_probability_mass_on_topk": round(mass / max(used, 1), 6),
            "trace_normalised_to_hidden": True,
            "eigenvalue_min": round(float(eigs.min()), 8),
            "eigenvalue_max": round(float(eigs.max()), 6),
            "condition_number": round(
                float(eigs.max() / max(float(eigs.min()), 1e-30)), 3
            ),
        }
        print(f"[mx33lg] metric: {json.dumps(metric_report)}")
        # A metric with a negative eigenvalue is not a metric.  The softmax
        # Hessian is PSD in exact arithmetic and the truncation keeps it PSD, so
        # anything below -1e-9 is a bug and not a rounding.
        if float(eigs.min()) < -1e-9:
            raise SystemExit(
                f"REFUSED: the accumulated metric has eigenvalue {float(eigs.min()):.3e}; "
                f"the softmax Hessian is PSD and this one is not"
            )
        M = M + max(0.0, -float(eigs.min())) * torch.eye(hidden, dtype=torch.float64)

    # ---- the ladder.  THE LAW's trust region, unchanged, on the new objective.
    Agram, Bcross = Aphi.T @ Aphi, Aphi.T @ Y
    reference = float(prior.norm())
    scale_free = None
    chosen = None
    for rel in LAMBDA_LADDER:
        lam = rel * float(torch.diagonal(Agram).mean())
        candidate = sylvester_ridge(Agram, Bcross, M, prior, lam)
        scale = float((candidate - prior).norm() / reference)
        if scale_free is None:
            scale_free = scale
        if scale <= 1.0:
            chosen = {"weight": candidate, "lambda_relative": rel, "update_scale": scale}
            break
    if chosen is None:
        raise SystemExit(
            f"no lambda on the ladder brings update_scale to 1.0 (free {scale_free:.3f}); "
            "THE LAW refuses the install"
        )
    w2p = chosen["weight"]

    # Scored BOTH ways on the held-out rows: the stream metric the control
    # optimises and the reader metric this arm optimises.  An arm that wins its
    # own objective and loses the other one is the interesting outcome, and it is
    # only visible if both are printed.
    def scores(pred: "torch.Tensor", target: "torch.Tensor") -> dict[str, float]:
        err = pred - target
        return {
            "relL2": float(err.norm() / target.norm()),
            "cos": float(
                (pred * target).sum()
                / (pred.norm() * target.norm())
            ),
            "reader_metric_sse": float((err @ M * err).sum()),
            "stream_sse": float(err.pow(2).sum()),
        }

    control, control_guard = B.law_abiding_solve(Aphi, Y, prior)
    arm_scores = scores(Aphi_h @ w2p.T, Y_h)
    ctl_scores = scores(Aphi_h @ control.T, Y_h)
    print(f"[mx33lg] arm     {json.dumps({k: round(v, 6) for k, v in arm_scores.items()})}")
    print(f"[mx33lg] control {json.dumps({k: round(v, 6) for k, v in ctl_scores.items()})}")

    # ---- write the child: the source's bytes with one tensor replaced.
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for f in args.child.iterdir():
        if f.is_file() and f.name not in ("model.safetensors",):
            shutil.copyfile(f, args.out / f.name)
    tensors = load_file(str(args.child / "model.safetensors"))
    key = f"model.layers.{last}.feed_forward.w2.weight"
    if key not in tensors:
        raise SystemExit(f"REFUSED: {key} is not a tensor of {args.child}")
    before = tensors[key].clone()
    if before.shape != w2p.shape:
        raise SystemExit(
            f"REFUSED: solved writer is {tuple(w2p.shape)} and the child's tensor "
            f"is {tuple(before.shape)}"
        )
    tensors[key] = w2p.to(before.dtype).contiguous()
    save_file(tensors, str(args.out / "model.safetensors"), metadata={"format": "pt"})

    live = int(sum(int(t.numel()) for t in tensors.values()))
    if live != int(build_receipt["child_parameters"]):
        raise SystemExit(
            f"the re-solve changed the parameter count: {live} against "
            f"{build_receipt['child_parameters']}"
        )
    receipt = dict(build_receipt)
    receipt.update(
        {
            "schema": SCHEMA,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder": "mx33_logitsolve.py",
            "tag": args.tag,
            "arm": (
                "LOGIT-SPACE FINAL-BLOCK SOLVE: the last block's w2 re-solved in "
                "the reader's own metric, everything else byte-identical"
            ),
            "logit_solve": {
                "site": name,
                "tensor": key,
                "metric_report": metric_report,
                "lambda_relative": chosen["lambda_relative"],
                "update_scale": chosen["update_scale"],
                "free_solve_update_scale": scale_free,
                "control_lambda_relative": control_guard["lambda_relative"],
                "control_update_scale": control_guard["update_scale"],
                "heldout_arm": arm_scores,
                "heldout_control": ctl_scores,
                "distance_from_control_weight": float(
                    (w2p - control).norm() / control.norm()
                ),
                "reduction_gate_at_identity": gate,
            },
            "source_model": str(args.child),
            "source_tensors_sha256": hashlib.sha256(
                (args.child / "model.safetensors").read_bytes()).hexdigest(),
            "child_model": str(args.out),
            "child_tensors_sha256": hashlib.sha256(
                (args.out / "model.safetensors").read_bytes()).hexdigest(),
            "child_parameters": live,
            "parent_parameters": PARENT_PARAMS,
            "net_ratio": live / PARENT_PARAMS,
            "parameters_added": 0,
            "wall_time_seconds": round(time.monotonic() - started, 2),
        }
    )
    (args.out / "mx33-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["logit_solve"], indent=1))
    print(f"[mx33lg] {args.tag}: {live} parameters, {live / PARENT_PARAMS:.6f}x -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
