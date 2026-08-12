#!/usr/bin/env python3
"""Wave 14: build a genuinely NARROWER LFM2 and read it on the frozen surface.

Why this script exists at all.  `mxq_materialize_child.py:120` refuses any folded
tensor whose shape differs from the parent's, so every child the campaign has
sealed is **parent-shaped**: the fold lives in a storage format and the model on
disk is still 229,693,184 parameters.  ADDENDUM 11's R3 asks for a
config-consistent smaller model that stock `transformers` loads.  This builds
one.

**The fold.**  VC-FOLD as registered: the SwiGLU intermediate axis is block-private,
so keeping ``r`` of the 2,560 channels is a *narrower SwiGLU and nothing else*.
``w1`` and ``w3`` keep the selected rows **verbatim** -- the survivors' gate and
up pre-activations are bit-faithful to the parent -- and all of the error is
pushed into one closed-form prior-anchored ridge on ``w2``.  Storage per site is
``3 * r * 1024`` against ``3 * 2560 * 1024``, exact, with **zero auxiliary
tensors**: the selection permutation vanishes into the layout because unit
identity is not needed at inference.

**Why one rung is all fourteen sites.**  `intermediate_size` is a single config
field shared by every block, so a config-consistent child folds every MLP to the
same width or it is not config-consistent.  That is the big bite ADDENDUM 11 §11.2
D2 asks for, and it is forced by the export requirement rather than chosen.

**The export path**, stated as R3 requires: `Lfm2MLP` reads
``config.intermediate_size`` directly (`modeling_lfm2.py:138-149`, with
``block_auto_adjust_ff_dim`` false in this checkpoint), so the child is a stock
`Lfm2ForCausalLM` with `intermediate_size` and `block_ff_dim` set to ``r``.  No
`trust_remote_code`, no custom module, no hook.

The review half re-uses the sealed runner's own `_generate` and its own prompt
authentication, so the surface is the frozen one by construction and not by
imitation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8")
for extra in (DEPLOY / "src", DEPLOY / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
sys.path.insert(0, "/home/cereal/nf-mx14")

from mx14_measure import MLP_LAYERS, have, rows, weight  # noqa: E402
from mx14_measure2 import pivoted_columns, prior_anchored, write_scores  # noqa: E402

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
PROTOCOL_SEAL = DEPLOY / "reports/r12-certified-20260804/PROTOCOL_SEAL.json"
PROMPTS = DEPLOY / "research/lfm2_semantic_whole_response_prompts_v1.json"
SIDECARS = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

PARENT_PARAMS = 229_693_184
SCHEMA = "neural-forge-matrix-wave14-narrow-child-v1"


def log(message: str) -> None:
    print(f"[mx14] {message}", flush=True)


# --------------------------------------------------------------------------- build
def build(keep: int, outdir: Path, arm: str = "C_pivoted") -> dict:
    from safetensors.torch import load_file, save_file

    torch.set_grad_enabled(False)
    state = load_file(str(PARENT / "model.safetensors"))
    config = json.loads((PARENT / "config.json").read_text())
    receipts = []

    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        if not have(name):
            raise SystemExit(f"no sealed evidence for {name}; refusing to guess a selection")
        phi = rows(name, "mlp.post_swiglu", "fit")
        phi_h = rows(name, "mlp.post_swiglu", "hld")
        Y = rows(name, "mlp.down", "fit")
        Y_h = rows(name, "mlp.down", "hld")
        w1 = weight(f"model.layers.{layer}.feed_forward.w1.weight")
        w3 = weight(f"model.layers.{layer}.feed_forward.w3.weight")
        w2 = weight(f"model.layers.{layer}.feed_forward.w2.weight")

        # Precondition, checked rather than assumed: post_swiglu @ w2^T IS mlp.down.
        # If this fails the endpoint is not the gated intermediate and every
        # selection below would be selecting the wrong axis.
        identity = float((phi @ w2.T - Y).norm() / Y.norm())
        if identity > 5e-3:
            raise SystemExit(
                f"L{layer:02d}: post_swiglu @ w2^T is {identity:.3e} from mlp.down; "
                "the intermediate-axis identity does not hold and the fold is refused"
            )

        if arm == "C_pivoted":
            sel = pivoted_columns(phi, keep)
        elif arm == "A_mass":
            score = phi.abs().mean(dim=0) * w2.norm(dim=0)
            sel = torch.tensor(
                sorted(sorted(range(score.numel()), key=lambda i: (-float(score[i]), i))[:keep]),
                dtype=torch.long,
            )
        else:
            raise SystemExit(f"unknown arm {arm}")

        # THE LAW as a kernel guard (slate §4.4): the install is refused unless the
        # re-solve stays within its own prior's norm.  The registered remedy is the
        # trust region, not an exception -- lambda is raised on a pinned ascending
        # ladder until the solve is compliant, and both the free solve and the
        # compliant one are sealed so the guard's cost is visible rather than hidden.
        free = prior_anchored(phi[:, sel], Y, w2[:, sel], rel=1e-6)
        free_scale = float((free - w2[:, sel]).norm() / w2[:, sel].norm())
        free_scores = write_scores(phi_h[:, sel] @ free.T, Y_h)
        w2p, update_scale, used_lambda = None, None, None
        for rel in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            candidate = prior_anchored(phi[:, sel], Y, w2[:, sel], rel=rel)
            scale = float((candidate - w2[:, sel]).norm() / w2[:, sel].norm())
            if scale <= 1.0:
                w2p, update_scale, used_lambda = candidate, scale, rel
                break
        if w2p is None:
            raise SystemExit(
                f"L{layer:02d}: no lambda on the ladder brings update_scale to 1.0 "
                f"(free solve was {free_scale:.3f}); THE LAW refuses the install"
            )
        scores = write_scores(phi_h[:, sel] @ w2p.T, Y_h)

        state[f"model.layers.{layer}.feed_forward.w1.weight"] = w1[sel].to(torch.bfloat16)
        state[f"model.layers.{layer}.feed_forward.w3.weight"] = w3[sel].to(torch.bfloat16)
        state[f"model.layers.{layer}.feed_forward.w2.weight"] = w2p.to(torch.bfloat16)
        receipts.append(
            {
                "layer": layer,
                "arm": arm,
                "keep": keep,
                "identity_check": identity,
                "update_scale": update_scale,
                "lambda_relative": used_lambda,
                "free_solve_update_scale": free_scale,
                "free_solve_relL2": free_scores["relL2"],
                "free_solve_cos": free_scores["cos"],
                "selection_first8": [int(x) for x in sel[:8]],
                **scores,
            }
        )
        log(
            f"L{layer:02d} keep={keep} relL2 {scores['relL2']:.4f} cos {scores['cos']:.4f} "
            f"update_scale {update_scale:.3f} (lam {used_lambda:g}, free {free_scale:.3f} "
            f"cos {free_scores['cos']:.4f})"
        )

    config["intermediate_size"] = keep
    config["block_ff_dim"] = keep
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(outdir / "model.safetensors"), metadata={"format": "pt"})
    for side in SIDECARS:
        if (PARENT / side).is_file():
            shutil.copy2(PARENT / side, outdir / side)

    total = sum(int(v.numel()) for v in state.values())
    tied = int(state["model.embed_tokens.weight"].numel()) if config.get("tie_word_embeddings") else 0
    receipt = {
        "schema": SCHEMA,
        "keep": keep,
        "arm": arm,
        "parent_parameters": PARENT_PARAMS,
        "child_parameters": total,
        "net_ratio": total / PARENT_PARAMS,
        "removed": PARENT_PARAMS - total,
        "tied_embedding_counted_once": bool(tied),
        "per_site": receipts,
        "export_path": (
            "stock Lfm2ForCausalLM with intermediate_size=block_ff_dim="
            f"{keep}; no trust_remote_code, no custom module, no hook"
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / "mx14-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    log(f"child parameters {total:,} = {total / PARENT_PARAMS:.6f}x  -> {outdir}")
    return receipt


def verify(outdir: Path) -> dict:
    """Load the child with stock transformers and count what actually loaded."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(outdir), dtype=torch.float32)
    n = sum(p.numel() for p in model.parameters())
    cfg = model.config
    out = {
        "loaded_with": "transformers.AutoModelForCausalLM.from_pretrained, no trust_remote_code",
        "class": type(model).__name__,
        "parameters_counted_live": int(n),
        "net_ratio_live": n / PARENT_PARAMS,
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
    }
    log(f"stock load OK: {type(model).__name__}, {n:,} params = {n / PARENT_PARAMS:.6f}x")
    del model
    return out


# -------------------------------------------------------------------------- review
def review(child: Path, out: Path, device: str, cap: int) -> int:
    import run_lfm2_composition_frozen_review as frozen
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seal = frozen._authenticated_protocol(PROTOCOL_SEAL)
    rows_, surface = frozen._authenticated_prompts(
        PROMPTS, expected_file_sha256=str(seal["prompt_pack_sha256"])
    )
    tokenizer = AutoTokenizer.from_pretrained(str(PARENT))
    eos = tokenizer.eos_token_id

    log("loading child")
    child_model = AutoModelForCausalLM.from_pretrained(str(child), dtype=torch.float32).to(device).eval()
    child_records = frozen._generate(
        child_model, tokenizer, rows_, cap=cap, device=device, eos_id=eos, label="child"
    )
    del child_model
    torch.cuda.empty_cache()

    log("loading parent (same process, ADDENDUM 6)")
    parent_model = AutoModelForCausalLM.from_pretrained(str(PARENT), dtype=torch.float32).to(device).eval()
    parent_records = frozen._generate(
        parent_model, tokenizer, rows_, cap=cap, device=device, eos_id=eos, label="parent"
    )
    del parent_model
    torch.cuda.empty_cache()

    build_receipt = json.loads((child / "mx14-build-receipt.json").read_text())
    dossier = {
        "schema": "neural-forge-matrix-wave14-narrow-frozen-review-v1",
        "protocol_sha256": str(seal["protocol_sha256"]),
        "prompt_pack_sha256": str(seal["prompt_pack_sha256"]),
        "prompt_surface_sha256": surface,
        "cases": len(rows_),
        "maximum_new_tokens": cap,
        "device": device,
        "dtype": "float32",
        "child_directory": str(child),
        "child_build_receipt": build_receipt,
        "child_responses": child_records,
        "parent_responses": parent_records,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dossier, indent=1))
    caps = sum(1 for r in child_records if r["termination"] == "censored-at-cap")
    pcaps = sum(1 for r in parent_records if r["termination"] == "censored-at-cap")
    log(f"dossier -> {out}  (child cap-hits {caps}/{len(rows_)}, parent {pcaps}/{len(rows_)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["build", "verify", "review"])
    ap.add_argument("--keep", type=int, default=640)
    ap.add_argument("--arm", default="C_pivoted")
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    args = ap.parse_args()
    if args.mode == "build":
        build(args.keep, args.child, args.arm)
        print(json.dumps(verify(args.child), indent=1))
    elif args.mode == "verify":
        print(json.dumps(verify(args.child), indent=1))
    else:
        return review(args.child, args.out or (args.child / "frozen-review-dossier.json"),
                      args.device, args.maximum_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
