#!/usr/bin/env python3
"""m3_c11_qvl_rows.py -- C11 patch-embed + merger interface rows for the QVL32 ViT unit.

Lane M3 (Claude, 2026-10-10). One native ocr row (type 6) = one raw 2x16x16x3 tubelet
(1,536 values, the parent's own patch_embed input). The rows written into the trunk bridge
(`COMPOSITE_S6_INPUT_FORMATS.md` section 1, `--c11 DIR`) are a LINEAR map raw tubelet ->
trunk input-embedding space built from the parent's own patch embedding and merger:

    p      = P x + b                     P = patch_embed.proj.weight [1152, 3*2*16*16] (exact)
    m(g)  ~= M (sum_{j in g} p_j) + c    the merger (LayerNorm -> 2x2 concat -> fc1 -> GELU -> fc2)
                                         FITTED as one shared per-patch linear map: the 4 rows of a
                                         2x2 group sum to the parent's merged LM token
    arm A  row = A (M p + c/4) + a/4     A: Qwen3-VL-32B LM input space -> trunk input space,
                                         ridge on the shared token strings of the two vocabularies
    arm B  row = M_B p + c_B/4           Qwen3.8-27B-U's OWN visual tower: its merger already lands
                                         in the trunk parent's input-embedding space (C11 "second arm")

so feature_proj[:, band] = A M P (arm A) and type_embed_delta = A (M b + c/4) + a/4. The
merger target m is the FULL vision tower's output (27 blocks + merger) for the same image:
the rows are the best linear map from raw pixels to where the parent's whole pipeline puts the
token (held-out relL2 / cos recorded as DIAGNOSTICS, never gates). OCR is input-only: the
emission band is one zero row (c16 needs b < e) with a zero type_affine shift.

Fit split: by IMAGE (every 5th image held out), lambda chosen on an inner by-image split of the
fit images, then refit on all fit images. fp64 normal equations. CPU only.

    OMP_NUM_THREADS=8 nice -n 19 venv/bin/python m3_c11_qvl_rows.py --captures DIR/qvl32 \
      --parent <qwen3vl file> --prefix visual. --arm qvl32 --align-vocab <qwen2 vocab.json> \
      --trunk-tokenizer <27B-U tokenizer.json> --trunk-native <S3 native dir> --out <c11 dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TOOL_NAME = "m3_c11_qvl_rows"
TOOL_VERSION = "1.0.0"
INTERFACE_SCHEMA = "chimera_omni.composite.interface_set.v1"
LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table(path: Path) -> dict[str, str]:
    from chimera_omni.importing.safetensors_io import read_safetensors

    return {name: str(view.sha256()) for name, view in sorted(read_safetensors(path, verify=False).items())}


def ridge_stats(X, Y, chunk: int = 8192):
    """Centered fp64 normal-equation statistics, chunked (X / Y may be fp32 and large)."""

    import torch

    n = X.shape[0]
    mx = torch.zeros(X.shape[1], dtype=torch.float64)
    my = torch.zeros(Y.shape[1], dtype=torch.float64)
    for i in range(0, n, chunk):
        mx += X[i:i + chunk].double().sum(0)
        my += Y[i:i + chunk].double().sum(0)
    mx, my = mx / n, my / n
    G = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64)
    XY = torch.zeros(X.shape[1], Y.shape[1], dtype=torch.float64)
    for i in range(0, n, chunk):
        xc = X[i:i + chunk].double() - mx
        G += xc.T @ xc
        XY += xc.T @ (Y[i:i + chunk].double() - my)
    return mx, my, G, XY


def ridge_solve(stats, lam_rel: float):
    """min ||[X 1] W - Y||^2 + lam * tr(Xc^T Xc)/d * ||W||^2 -> (W [dx, dy], bias [dy])."""

    import torch

    mx, my, G, XY = stats
    lam = lam_rel * float(torch.trace(G)) / G.shape[0]
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), XY)
    return W, my - mx @ W


def ridge_fit(X, Y, lam_rel: float):
    return ridge_solve(ridge_stats(X, Y), lam_rel)


def scores(pred, Y) -> dict[str, float]:
    import torch

    pred, Y = pred.double(), Y.double()

    rel = float((pred - Y).norm() / Y.norm())
    cos = torch.nn.functional.cosine_similarity(pred, Y, dim=1)
    base = float((Y - Y.mean(0)).norm() / Y.norm())
    return {"rel_l2": rel, "cos_mean": float(cos.mean()), "cos_p05": float(cos.quantile(0.05)),
            "mean_predictor_rel_l2": base, "rows": int(Y.shape[0])}


def select_lambda(X, Y, groups, fit_mask):
    """Inner by-image split of the fit images: every 4th fit image validates."""

    import torch

    fit_ids = sorted(set(groups[fit_mask].tolist()))
    inner_val = set(fit_ids[3::4])
    val = torch.tensor([int(g) in inner_val for g in groups.tolist()]) & fit_mask
    tr = fit_mask & ~val
    ladder = {}
    stats = ridge_stats(X[tr], Y[tr])
    for lam in LAMBDAS:
        W, c = ridge_solve(stats, lam)
        ladder[str(lam)] = scores(X[val].double() @ W + c, Y[val])["rel_l2"]
    best = min(LAMBDAS, key=lambda v: ladder[str(v)])
    return best, ladder


def load_captures(directory: Path):
    import torch
    from safetensors.torch import load_file

    receipt = json.loads((directory / "capture_receipt.json").read_text())
    pix, pe, s, m, gid = [], [], [], [], []
    for record in receipt["images"]:
        body = load_file(str(directory / record["file"]))
        pixels, patch, merged = body["pixels"].double(), body["patch_embed"].double(), body["merged"].double()
        if pixels.shape[0] != 4 * merged.shape[0]:
            raise SystemExit(f"{record['file']}: {pixels.shape[0]} rows vs {merged.shape[0]} merged tokens")
        pix.append(pixels)
        pe.append(patch)
        s.append(patch.view(-1, 4, patch.shape[1]).sum(1))
        m.append(merged)
        gid.append(torch.full((merged.shape[0],), int(record["index"]), dtype=torch.long))
    return receipt, torch.cat(pix), torch.cat(pe), torch.cat(s), torch.cat(m), torch.cat(gid)


def parent_patch_embed(parent: Path, prefix: str):
    from safetensors import safe_open

    files = [parent]
    if parent.is_dir():
        index = json.loads((parent / "model.safetensors.index.json").read_text())["weight_map"]
        files = [parent / index[prefix + "patch_embed.proj.weight"]]
    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
        w = handle.get_tensor(prefix + "patch_embed.proj.weight")
        b = handle.get_tensor(prefix + "patch_embed.proj.bias")
    return w.double().reshape(w.shape[0], -1), b.double(), str(files[0])


def token_alignment(vocab_json: Path, trunk_tokenizer: Path, qvl_parent: Path, trunk_native: Path, held_every: int = 10):
    """A: Qwen3-VL-32B input embeddings -> trunk input embeddings on identical byte-level token strings."""

    import torch
    from safetensors import safe_open

    qvocab = json.loads(vocab_json.read_text(encoding="utf-8"))
    tvocab = json.loads(trunk_tokenizer.read_text(encoding="utf-8"))["model"]["vocab"]
    pairs = sorted((qid, tvocab[tok]) for tok, qid in qvocab.items() if tok in tvocab)
    qids = torch.tensor([p[0] for p in pairs])
    tids = torch.tensor([p[1] for p in pairs])
    with safe_open(str(qvl_parent), framework="pt", device="cpu") as handle:
        qemb = handle.get_tensor("model.embed_tokens.weight")[qids].float()
    index = json.loads((trunk_native / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(trunk_native / index["embed.weight"]), framework="pt", device="cpu") as handle:
        temb = handle.get_tensor("embed.weight")[tids].float()
    held = torch.arange(len(pairs)) % held_every == held_every - 1
    best, ladder = None, {}
    inner = (torch.arange(len(pairs)) % held_every == held_every - 2)
    stats = ridge_stats(qemb[~held & ~inner], temb[~held & ~inner])
    for lam in LAMBDAS:
        W, c = ridge_solve(stats, lam)
        ladder[str(lam)] = scores(qemb[inner].double() @ W + c, temb[inner])["rel_l2"]
    best = min(LAMBDAS, key=lambda v: ladder[str(v)])
    W, c = ridge_fit(qemb[~held], temb[~held], best)
    report = {"matched_token_strings": len(pairs), "qvl_vocab": len(qvocab), "trunk_vocab": len(tvocab),
              "held_rows": int(held.sum()), "lambda_rel": best, "lambda_ladder_inner_rel_l2": ladder,
              "held": scores(qemb[held].double() @ W + c, temb[held]),
              "source_files": {"qvl_vocab": str(vocab_json), "qvl_vocab_sha256": sha256_file(vocab_json),
                               "trunk_tokenizer": str(trunk_tokenizer), "trunk_tokenizer_sha256": sha256_file(trunk_tokenizer),
                               "trunk_embed_shard": str(trunk_native / index["embed.weight"])}}
    return W.T.contiguous(), c, report  # A [5120_trunk, 5120_qvl]


def bridge_scale_diagnostic(trunk_native: Path, feature_rows, delta, sample_pixels, type_id: int):
    """How large the written rows are against the S3 bridge's random S0 type embedding (pre-RMSNorm)."""

    import torch
    from safetensors import safe_open

    index = json.loads((trunk_native / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(trunk_native / index["native_media_bridge.type_embed.weight"]), framework="pt", device="cpu") as handle:
        type_row = handle.get_tensor("native_media_bridge.type_embed.weight")[type_id].double()
    feature = sample_pixels @ feature_rows.T
    written = feature + delta
    return {"row_norm_median_feature_term": float(feature.norm(dim=1).median()),
            "type_embed_delta_norm": float(delta.norm()),
            "s0_type_embed_row_norm": float(type_row.norm()),
            "ratio_written_to_s0_type_embed_median": float(written.norm(dim=1).median() / type_row.norm()),
            "note": "c16 ADDS the delta to the random S0 type_embed row; before the bridge RMSNorm the written row "
                    "is this ratio of that random vector (diagnostic; the bridge's position/noise terms are not counted)"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--prefix", required=True, choices=("visual.", "model.visual."))
    parser.add_argument("--arm", required=True, choices=("qvl32", "q38u"))
    parser.add_argument("--align-vocab", help="arm qvl32: the Qwen2/Qwen3-VL vocab.json (byte-level token -> id)")
    parser.add_argument("--trunk-tokenizer", help="arm qvl32: the trunk parent tokenizer.json")
    parser.add_argument("--trunk-native", required=True, help="S3 native bundle dir (embed.weight, bridge tensors)")
    parser.add_argument("--type-id", type=int, default=6)
    parser.add_argument("--feature-band", type=int, nargs=2, default=(640, 2176))
    parser.add_argument("--emission-band", type=int, nargs=2, default=(384, 385))
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--native-feature-dim", type=int, default=6144)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import save_file

    started = time.time()
    torch.set_grad_enabled(False)
    captures = Path(args.captures)
    receipt, pixels, patch, s, m, gid = load_captures(captures)
    P, b, patch_file = parent_patch_embed(Path(args.parent), args.prefix)
    if P.shape[1] != args.feature_band[1] - args.feature_band[0]:
        raise SystemExit(f"feature band width {args.feature_band} != tubelet width {P.shape[1]}")
    exact = float(((pixels @ P.T + b) - patch).norm() / patch.norm())
    held_images = {int(r["index"]) for r in receipt["images"] if int(r["index"]) % 5 == 4}
    held = torch.tensor([int(g) in held_images for g in gid.tolist()])  # per merged token
    held_rows = held.repeat_interleave(4)                                # per patch row (4 per token)
    fit = ~held
    lam, ladder = select_lambda(s, m, gid, fit)
    W, c = ridge_fit(s[fit], m[fit], lam)  # m ~= s @ W + c ; M = W^T [out, 1152]
    M = W.T.contiguous()
    merger_fit = {"lambda_rel": lam, "lambda_ladder_inner_rel_l2": ladder,
                  "fit": scores(s[fit] @ W + c, m[fit]), "held": scores(s[held] @ W + c, m[held]),
                  "split": f"by image: {len(held_images)} held images (index % 5 == 4), inner lambda split by image"}
    # control: an unconstrained per-position map on the 2x2 concat (4 x 1152 -> out)
    concat = patch.view(-1, 4 * patch.shape[1])
    Wc, cc = ridge_fit(concat[fit], m[fit], lam)
    merger_fit["control_unshared_concat_held"] = scores(concat[held] @ Wc + cc, m[held])

    feature = M @ P                       # [out, 1536]
    delta = M @ b + c / 4.0               # [out]
    alignment = None
    if args.arm == "qvl32":
        if not (args.align_vocab and args.trunk_tokenizer):
            raise SystemExit("arm qvl32 needs --align-vocab and --trunk-tokenizer (its merger lands in Qwen3-VL's LM space)")
        A, a, alignment = token_alignment(Path(args.align_vocab), Path(args.trunk_tokenizer), Path(args.parent), Path(args.trunk_native))
        feature = A @ feature
        delta = A @ delta + a / 4.0
        held_tokens = (s[held] @ W + c) @ A.T + a
        alignment["merged_tokens_in_trunk_space_note"] = "held merged-token predictions mapped through A (no trunk-space target exists)"
        alignment["held_merged_norm_median_trunk_space"] = float(held_tokens.norm(dim=1).median())
    if feature.shape != (args.hidden, P.shape[1]):
        raise SystemExit(f"feature rows {tuple(feature.shape)} != [{args.hidden}, {P.shape[1]}]")
    target = (s[held] @ W + c) if alignment is None else held_tokens
    group_check = float(((pixels[held_rows] @ feature.T + delta).view(-1, 4, args.hidden).sum(1) - target).norm() / target.norm())
    eb, ee = args.emission_band
    state = {
        "bridge.feature_proj.weight": feature.to(torch.float32).contiguous(),
        "bridge.type_embed_delta": delta.to(torch.float32).contiguous(),
        "head.out.weight": torch.zeros(ee - eb, args.hidden, dtype=torch.float32),
        "head.type_affine_shift": torch.zeros(ee - eb, dtype=torch.float32),
    }
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise SystemExit("non-finite interface rows")
    name = "qvl32_patch_merger" if args.arm == "qvl32" else "q38u_own_visual_patch_merger"
    out = Path(args.out)
    rows_rel = Path(name) / "linear" / "composed_interface.safetensors"
    (out / rows_rel).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / rows_rel))
    diag = bridge_scale_diagnostic(Path(args.trunk_native), feature, delta, pixels[held_rows][:4096], args.type_id)
    entry = {
        "name": name, "source": "QVL32" if args.arm == "qvl32" else "Q38-27BU-visual",
        "type_id": int(args.type_id), "feature_band": list(args.feature_band), "emission_band": [eb, ee],
        "rows": rows_rel.as_posix(), "rung": "linear", "companion_decoder": False,
        "psnr_absorbed_db": None, "psnr_parent_roundtrip_db": None,
        "tensor_sha256": table(out / rows_rel),
    }
    companions = {"schema": INTERFACE_SCHEMA, "hidden": int(args.hidden), "native_feature_dim": int(args.native_feature_dim),
                  "entries": [entry]}
    (out / "companions.json").write_text(json.dumps(companions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(Path(__file__).resolve()),
        "argv": sys.argv, "arm": args.arm, "entry": entry,
        "construction": "feature_proj = %sM P; type_embed_delta = %s(M b + c/4)%s" % (
            "A " if args.arm == "qvl32" else "", "A " if args.arm == "qvl32" else "", " + a/4" if args.arm == "qvl32" else ""),
        "captures": {"dir": str(captures), "receipt_sha256": sha256_file(captures / "capture_receipt.json"),
                     "images": len(receipt["images"]), "rows": int(pixels.shape[0]), "merged_rows": int(m.shape[0]),
                     "parent": receipt["parent"], "config": receipt["config"]},
        "patch_embed": {"file": patch_file, "shape": list(P.shape), "exactness_rel_l2_vs_capture": exact},
        "merger_fit": merger_fit,
        "alignment": alignment,
        "group_sum_consistency_rel_l2": group_check,
        "bridge_scale_diagnostic": diag,
        "bands": {"source": "R0 proposal of the media-chart-pairing design (ocr/6 f[640,2176) e[384,385)); owner decision",
                  "emission": "OCR is input-only: one zero emission row and a zero type_affine shift (c16 needs b < e)"},
        "not_carried": ["the learned interpolated pos_embed (position-dependent; trunk positions instead)",
                        "the 2-D vision RoPE and all 27 ViT blocks (folded into the fitted M as a linear stand-in)",
                        "DeepStack mergers (Qwen3-VL-32B blocks 8/16/24 -> LM layers)"],
        "diagnostics_policy": "every fit number here is a diagnostic (owner policy 2026-10-06); structural checks: shapes, finiteness, band fit",
        "wall_seconds": time.time() - started,
    }
    (out / "c11_rows_receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "entry": name, "merger_held": merger_fit["held"],
                      "control_held": merger_fit["control_unshared_concat_held"],
                      "alignment_held": None if alignment is None else alignment["held"],
                      "patch_embed_exactness": exact, "group_sum_consistency": group_check, "bridge": diag}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
