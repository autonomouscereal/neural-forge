#!/usr/bin/env python3
"""c18_music_band.py -- Lane M2: the C18 [65,536 x H] music_codes band table.

Builds the ``--c18 FILE`` input of ``c16_install_media.py`` (format of record:
``composite/COMPOSITE_S6_INPUT_FORMATS.md`` section 2): one safetensors file with
``embed.weight [V_band, H]`` and ``lm_head.weight [V_band, H]`` (fp32) plus the sidecar
``FILE.json`` carrying ``tensor_sha256`` (the installer verifies it) and this receipt.

What the rows are
-----------------
The ACE-Step 1.5 LM (Qwen3, hidden 2,048, TIED embeddings, 217,204 ids) speaks audio codes
as added tokens ``[code_begin, code_begin + code_count)``. The trunk (Qwen3.8-27B-U Chimera,
hidden 5,120, UNTIED lm_head) declares the ``music_codes`` band ``[252,416, 317,952)`` for
them. A code token has no byte spelling, so ``bridge.token_alignment`` cannot map it; the
code rows are carried across by a closed-form linear chart fitted on the tokens the two
vocabularies DO share:

1. ``token_alignment(trunk, ace, corpus, full_vocab=True)`` walks the trunk's full
   248,320-row base vocabulary (the S6B gate's "aligned / unmapped counts over 248,320").
2. Anchors = trunk ids whose bytes re-encode to exactly ONE ordinary ACE text id (no added /
   special ids). Pairs ``(ace_embed[a], trunk_embed[t])`` and ``(ace_embed[a], trunk_lm_head[t])``.
3. Two ridge maps 2,048 -> 5,120 (unregularised intercept; lambda chosen on an inner split
   from ``mediafold.linalg.LAMBDA_LADDER`` x mean Gram diagonal, then refitted on all fit
   anchors): ``M_e`` to the trunk's input embedding, ``M_h`` to its output head (the ACE
   head IS its embedding, tied). With ``--deviation-calibration median`` (default) the
   predicted deviation from the trunk's mean row is rescaled so its median norm on held
   anchors equals the true rows' (ridge shrinks a low-R^2 prediction toward the mean row;
   the band would otherwise be ~64,000 near-copies of the average text row). Both the raw
   ridge and the calibrated scores are recorded.
4. Band rows ``[0, code_count)`` = ``M(ace_embed[code_begin + k])``; rows
   ``[code_count, V_band)`` stay ZERO (Codex's ACE codec bridge declaration: 64,000
   generator-eligible codes -> native ``[252,416, 316,416)``; the 1,535 named-but-excluded
   codes and the non-existent index 65,535 stay unused).

Diagnostics only (owner policy 2026-10-06): held-anchor relL2 / cosine / retrieval@k for both
maps, code-row norms against text-row norms, how much of the code rows' energy lies in the
anchor subspace the maps were fitted on (extrapolation), and the installer's preconditions
(band width, zero target rows, untied head, sidecar digest) checked without writing a child.

Structural refusals (exit 2): a band whose width differs from the table, nonzero target rows in
the trunk, a code interval outside the ACE table, a component whose weights file does not match
its own pin, a non-finite table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_VERSION = "1.0.0"
RECEIPT_SCHEMA = "chimera_omni.composite.c18_music_band.v1"
HERE = Path(__file__).resolve().parent
LAMBDA_LADDER = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)  # = mediafold.linalg.LAMBDA_LADDER

DEFAULT_CORPUS = (
    "<|im_start|>user\nWrite a slow piano ballad in C minor, 72 bpm.<|im_end|>",
    "def fold(x):\n    return [v * 2 for v in x if v > 0]  # comment",
    "Die Katze sitzt auf der Matte. Le chat est sur le tapis. El gato está en la alfombra.",
    "日本語の文章とEnglish mixed text, 数字 12345 and emoji-free punctuation: ;:!?",
    "[verse]\nneon rain on the boulevard\n[chorus]\nhold the line, hold the line\n",
    "tags: lo-fi, hip hop, vinyl crackle, mellow, 85 bpm, instrumental",
)


class Refusal(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[c18 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch():
    import torch

    return torch


# ---------------------------------------------------------------------------
# closed-form ridge with an unregularised intercept
# ---------------------------------------------------------------------------


def ridge_fit(x: Any, y: Any, *, inner_fraction: float, seed: int) -> dict[str, Any]:
    """Pick lambda (relative to mean Gram diagonal) on an inner split, refit on all rows."""

    torch = _torch()
    n = x.shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=gen)
    inner = max(1, int(round(inner_fraction * n)))
    val_idx, fit_idx = order[:inner], order[inner:]

    def solve(rows_x, rows_y, lambdas):
        xm = rows_x.mean(0, keepdim=True)
        ym = rows_y.mean(0, keepdim=True)
        xc = rows_x - xm
        yc = rows_y - ym
        gram = xc.transpose(0, 1) @ xc
        cross = xc.transpose(0, 1) @ yc
        evals, evecs = torch.linalg.eigh(gram)
        proj = evecs.transpose(0, 1) @ cross
        scale = float(torch.diagonal(gram).mean())
        out = []
        for rel in lambdas:
            lam = float(rel) * scale
            w = evecs @ (proj / (evals.clamp_min(0) + lam).unsqueeze(1))  # [in, out]
            b = ym - xm @ w
            out.append((rel, lam, w, b))
        return out

    trail = []
    best = None
    for rel, lam, w, b in solve(x[fit_idx], y[fit_idx], LAMBDA_LADDER):
        pred = x[val_idx] @ w + b
        err = float((pred - y[val_idx]).norm() / y[val_idx].norm())
        trail.append({"ridge_rel": rel, "inner_val_rel_l2": err})
        if best is None or err < best[1]:
            best = (rel, err)
    rel = best[0]
    (_, lam, w, b), = solve(x, y, (rel,))
    return {"weight": w, "bias": b.squeeze(0), "ridge_rel": rel, "ridge_lambda": lam, "ladder": trail, "fit_rows": int(n), "inner_val_rows": int(inner)}


def scores(pred: Any, target: Any) -> dict[str, float]:
    torch = _torch()
    rel = float((pred - target).norm() / target.norm())
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    centred = target - target.mean(0, keepdim=True)
    mean_rel = float(centred.norm() / target.norm())
    return {"rel_l2": rel, "cos_median": float(cos.median()), "cos_p05": float(cos.quantile(0.05)), "mean_baseline_rel_l2": mean_rel, "zero_baseline_rel_l2": 1.0}


def retrieval(pred: Any, candidates: Any, truth: Any, ks=(1, 10)) -> dict[str, float]:
    torch = _torch()
    p = torch.nn.functional.normalize(pred.to(torch.float32), dim=1)
    c = torch.nn.functional.normalize(candidates.to(torch.float32), dim=1)
    kmax = min(max(ks), int(c.shape[0]))
    top = torch.cat([(p[i : i + 256] @ c.transpose(0, 1)).topk(kmax, dim=1).indices for i in range(0, p.shape[0], 256)])
    out = {}
    for k in ks:
        out[f"at_{k}"] = float((top[:, :k] == truth.unsqueeze(1)).any(dim=1).float().mean())
    out["candidates"] = int(candidates.shape[0])
    out["queries"] = int(pred.shape[0])
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def load_trunk_rows(native: Path, names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    torch = _torch()
    index = json.loads((native / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((native / "config.json").read_text(encoding="utf-8"))
    recorded = index.get("metadata", {}).get("tensor_sha256", {})
    out = {}
    for name in names:
        shard = index["weight_map"].get(name)
        if shard is None:
            raise Refusal(f"{native}: no {name}")
        with safe_open(str(native / shard), framework="pt") as handle:
            out[name] = handle.get_tensor(name)
    return out, config, {n: recorded.get(n) for n in names}


def run(args: argparse.Namespace) -> int:
    torch = _torch()
    from chimera_omni.importing.bridge import token_alignment
    from chimera_omni.importing.tokenizer import ChimeraTokenizer
    from chimera_omni.importing.safetensors_io import read_safetensors
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.time()
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "tool": {"name": "c18_music_band.py", "version": TOOL_VERSION, "sha256": sha256_file(Path(__file__))},
        "argv": sys.argv[1:],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owner_policy": "finite numeric values are diagnostics, never gates (Cyril 2026-10-06)",
    }
    native = Path(args.trunk_native)
    component = Path(args.ace_component)

    # -- the band the trunk declares -------------------------------------------
    trunk_tensors, trunk_config, trunk_shas = load_trunk_rows(native, ("embed.weight", "lm_head.weight"))
    ledger = trunk_config.get("ledger", {})
    band = next((r for r in ledger.get("ranges", []) if r.get("name") == args.band_name), None)
    if band is None:
        raise Refusal(f"trunk ledger declares no band {args.band_name!r}")
    begin, end, width = int(band["begin"]), int(band["end"]), int(band["width"])
    cfg = trunk_config["config"]
    hidden = int(cfg["hidden_size"])
    if bool(cfg.get("tie_word_embeddings")):
        raise Refusal("the trunk ties lm_head to embed; this tool writes untied tables (rebuild with equal maps)")
    embed, head = trunk_tensors["embed.weight"], trunk_tensors["lm_head.weight"]
    nonzero = {"embed.weight": int(embed[begin:end].ne(0).sum()), "lm_head.weight": int(head[begin:end].ne(0).sum())}
    if any(nonzero.values()):
        raise Refusal(f"trunk band rows are not zero: {nonzero}")
    receipt["trunk"] = {"native_dir": str(native), "chimera_commit": trunk_config.get("chimera_commit"), "hidden": hidden, "band": band, "band_rows_nonzero_before": nonzero, "tensor_sha256_recorded": trunk_shas, "base_vocab": int(cfg.get("ledger_base_vocab_size", 0))}

    # -- the ACE LM component ------------------------------------------------------
    pin_path = component / "native_component.json"
    weights = component / "model.safetensors"
    code_begin, code_count = args.code_begin, args.code_count
    comp_meta: dict[str, Any] = {"dir": str(component)}
    if pin_path.is_file():
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        codec = pin.get("codec_bridge", {})
        comp_meta["codec_bridge"] = codec
        comp_meta["native_component_json_sha256"] = sha256_file(pin_path)
        if code_begin is None:
            code_begin = int(codec["source_generator_interval"][0])
        if code_count is None:
            code_count = int(codec["source_generator_interval"][1]) - int(codec["source_generator_interval"][0])
        eligible = codec.get("native_eligible_interval")
        if eligible and (int(eligible[0]) != begin or int(eligible[1]) - int(eligible[0]) != code_count):
            raise Refusal(f"codec bridge native_eligible_interval {eligible} disagrees with band begin {begin} / code count {code_count}")
        weight_pin = pin.get("weight_pin", {})
        if weight_pin.get("sha256"):
            actual = sha256_file(weights)
            if actual != weight_pin["sha256"]:
                raise Refusal(f"{weights} sha256 {actual} != the component's own weight_pin {weight_pin['sha256']}")
            comp_meta["weights_sha256"] = actual
            comp_meta["weight_pin_verified"] = True
    else:
        comp_meta["weights_sha256"] = sha256_file(weights)
        comp_meta["weight_pin_verified"] = None
    if code_begin is None or code_count is None:
        raise Refusal("code interval unknown: pass --code-begin/--code-count or a component with native_component.json")
    if code_count > width:
        raise Refusal(f"{code_count} codes do not fit the {width}-row band")
    with safe_open(str(weights), framework="pt") as handle:
        ace_embed = handle.get_tensor(args.ace_embed_key).to(torch.float32)
    ace_vocab, ace_hidden = ace_embed.shape
    if not 0 <= code_begin < code_begin + code_count <= ace_vocab:
        raise Refusal(f"code interval [{code_begin}, {code_begin + code_count}) outside the ACE table [0, {ace_vocab})")
    comp_meta.update({"embed_key": args.ace_embed_key, "embed_shape": [int(ace_vocab), int(ace_hidden)], "code_interval": [int(code_begin), int(code_begin + code_count)], "tied_head": "the ACE LM ties its head to embed_tokens (config tie_word_embeddings true)"})
    receipt["ace_component"] = comp_meta

    # -- full-vocab alignment over the trunk's base vocabulary ------------------------------
    trunk_tok = ChimeraTokenizer.from_qwen(args.trunk_tokenizer_dir, base_vocab_size=int(args.trunk_base_vocab))
    ace_tok = ChimeraTokenizer.from_qwen(Path(args.ace_tokenizer_dir or component / "tokenizer"), base_vocab_size=int(ace_vocab))
    corpus = list(DEFAULT_CORPUS)
    log(f"token_alignment(full_vocab=True) over {trunk_tok.base_vocab_size} trunk ids")
    pairs, alignment = token_alignment(trunk_tok, ace_tok, corpus, full_vocab=True)
    ace_special = set(int(i) for i in ace_tok.base.special_ids)
    anchors = sorted((t, p[0]) for t, p in pairs.items() if len(p) == 1 and p[0] not in ace_special and p[0] < code_begin)
    if len(anchors) < float(args.min_anchor_factor) * ace_hidden:
        raise Refusal(f"only {len(anchors)} anchors for a {ace_hidden}-wide map (need {args.min_anchor_factor} x width)")
    receipt["alignment"] = {**alignment, "anchors_single_piece_text": len(anchors), "anchor_rule": "trunk id -> exactly one ACE id that is not added/special and lies below the code interval", "trunk_tokenizer_dir": str(args.trunk_tokenizer_dir), "trunk_tokenizer_sha256": sha256_file(Path(args.trunk_tokenizer_dir) / "tokenizer.json"), "ace_tokenizer_sha256": sha256_file(Path(args.ace_tokenizer_dir or component / "tokenizer") / "tokenizer.json")}
    log(f"aligned {alignment['aligned_tokens']} / unmapped {alignment['unmapped_tokens']} of {alignment['candidate_tokens']}; anchors {len(anchors)}")

    t_ids = torch.tensor([t for t, _ in anchors], dtype=torch.long)
    a_ids = torch.tensor([a for _, a in anchors], dtype=torch.long)
    gen = torch.Generator().manual_seed(int(args.seed))
    perm = torch.randperm(len(anchors), generator=gen)
    held_n = max(1, int(round(float(args.holdout_fraction) * len(anchors))))
    held, fit = perm[:held_n], perm[held_n:]
    x_all = ace_embed[a_ids].to(torch.float64)
    maps: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    codes = ace_embed[code_begin : code_begin + code_count].to(torch.float64)
    for key, target_table in (("embed.weight", embed), ("lm_head.weight", head)):
        y_all = target_table[t_ids].to(torch.float64)
        log(f"{key}: ridge {ace_hidden} -> {hidden} on {int(fit.numel())} anchors")
        result = ridge_fit(x_all[fit], y_all[fit], inner_fraction=float(args.inner_fraction), seed=int(args.seed))
        w, b = result["weight"], result["bias"]
        pred_held = x_all[held] @ w + b
        q = min(int(args.retrieval_queries), int(held.numel()))
        d: dict[str, Any] = {
            "fit": {k: v for k, v in result.items() if k not in ("weight", "bias")},
            "ridge": {
                "held": {**scores(pred_held, y_all[held]), "rows": int(held.numel())},
                "fit_scores": scores(x_all[fit] @ w + b, y_all[fit]),
                "held_retrieval_among_all_anchor_rows": retrieval(pred_held[:q], y_all, held[:q]),
            },
        }
        # deviation calibration: ridge shrinks a weak (low-R^2) prediction toward the trunk's mean
        # row, which would make every code row ~ the average text row at a fraction of a text
        # row's spread. "median" rescales the predicted DEVIATION from the trunk mean so its
        # median norm on held anchors equals the true rows' (direction unchanged; a heuristic).
        x_mean = x_all[fit].mean(0)
        y_mean = x_mean @ w + b
        dev_pred = (x_all[held] - x_mean) @ w
        dev_true = y_all[held] - y_all[fit].mean(0)
        gamma = float(dev_true.norm(dim=1).median() / dev_pred.norm(dim=1).median()) if args.deviation_calibration == "median" else 1.0
        w_used = w * gamma
        b_used = y_mean - x_mean @ w_used
        pred_cal = x_all[held] @ w_used + b_used
        d["calibrated"] = {
            "mode": args.deviation_calibration,
            "gamma": gamma,
            "held": {**scores(pred_cal, y_all[held]), "rows": int(held.numel())},
            "held_retrieval_among_all_anchor_rows": retrieval(pred_cal[:q], y_all, held[:q]),
        }
        d["held"] = d["calibrated"]["held"] if args.deviation_calibration == "median" else d["ridge"]["held"]
        d["held_retrieval_among_all_anchor_rows"] = d["calibrated"]["held_retrieval_among_all_anchor_rows"] if args.deviation_calibration == "median" else d["ridge"]["held_retrieval_among_all_anchor_rows"]
        mapped = codes @ w_used + b_used
        if not bool(torch.isfinite(mapped).all()):
            raise Refusal(f"{key}: mapped code rows are not finite")
        text_norms = target_table[: int(args.trunk_base_vocab)].to(torch.float32).norm(dim=1)
        d["norms"] = {
            "trunk_text_row_norm_median": float(text_norms[text_norms > 0].median()),
            "mapped_code_row_norm_median": float(mapped.norm(dim=1).median()),
            "mapped_code_row_norm_median_uncalibrated": float((codes @ w + b).norm(dim=1).median()),
            "mapped_anchor_row_norm_median_held": float(pred_cal.norm(dim=1).median()),
            "true_anchor_row_norm_median_held": float(y_all[held].norm(dim=1).median()),
            "trunk_mean_row_norm": float(y_all[fit].mean(0).norm()),
        }
        maps[key] = {"weight": w_used, "bias": b_used}
        table = torch.zeros(width, hidden, dtype=torch.float32)
        table[:code_count] = mapped.to(torch.float32)
        tables[key] = table
        diag[key] = d
        log(f"{key}: ridge held rel_l2 {d['ridge']['held']['rel_l2']:.4f} cos_med {d['ridge']['held']['cos_median']:.4f} r@1 {d['ridge']['held_retrieval_among_all_anchor_rows']['at_1']:.4f}; gamma {gamma:.3f} -> calibrated rel_l2 {d['held']['rel_l2']:.4f} cos_med {d['held']['cos_median']:.4f} r@1 {d['held_retrieval_among_all_anchor_rows']['at_1']:.4f}")
    # extrapolation: how much of the code rows lies in the anchor rows' principal subspace
    xc = x_all[fit] - x_all[fit].mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    energy = s.pow(2) / s.pow(2).sum()
    k = int((energy.cumsum(0) < 0.99).sum()) + 1
    basis = vh[:k]
    cc = codes - x_all[fit].mean(0, keepdim=True)
    inside = (cc @ basis.transpose(0, 1)).pow(2).sum(1) / cc.pow(2).sum(1).clamp_min(1e-30)
    text_rows = ace_embed[:code_begin].to(torch.float32).norm(dim=1)
    diag["extrapolation"] = {
        "anchor_pca_rank_99pct": k,
        "code_rows_energy_in_anchor_99pct_subspace_median": float(inside.median()),
        "code_row_offset_from_anchor_mean_rel": float(cc.norm(dim=1).median() / xc.norm(dim=1).median()),
        "ace_code_row_norm_median": float(codes.norm(dim=1).median()),
        "ace_text_row_norm_median": float(text_rows[text_rows > 0].median()),
        "cos_between_mapped_code_mean_and_trunk_text_mean_embed": float(torch.nn.functional.cosine_similarity(tables["embed.weight"][:code_count].mean(0), embed[: int(args.trunk_base_vocab)].to(torch.float32).mean(0), dim=0)),
        "meaning": "the maps are fitted on text anchors; code rows are a different region of the ACE table, so the band rows are an extrapolation of a text-fitted chart (diagnostic only)",
    }
    receipt["diagnostics"] = diag
    receipt["band_rows"] = {"written": int(code_count), "zero": int(width - code_count), "zero_rule": "rows >= code_count stay zero: Codex codec bridge native_unused_interval (generator-excluded named codes + the nonexistent index)"}

    # -- write table + sidecar ----------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tables.items()}, str(out))
    views = read_safetensors(out, verify=False)
    tensor_sha = {name: str(view.sha256()) for name, view in sorted(views.items())}
    maps_path = out.with_name(out.stem + "_maps.safetensors")
    save_file({f"{k}.map.weight": v["weight"].to(torch.float32).contiguous() for k, v in maps.items()} | {f"{k}.map.bias": v["bias"].to(torch.float32).contiguous() for k, v in maps.items()}, str(maps_path))
    receipt["table"] = {"path": str(out), "file_sha256": sha256_file(out), "dtype": "float32", "shape": [width, hidden]}
    receipt["maps_file"] = {"path": str(maps_path), "sha256": sha256_file(maps_path)}
    receipt["tensor_sha256"] = tensor_sha

    # -- the installer's own checks (no child written) --------------------------------------
    sys.path.insert(0, str(HERE))
    import c16_install_media as c16

    check = c16.verify_declared_sha256(c16.read_single(out), tensor_sha, where=str(out))
    receipt["installer_check"] = {
        "c16_tool_sha256": sha256_file(Path(c16.__file__)),
        "tensor_sha256_check": check,
        "band_width_matches_table": tables["embed.weight"].shape[0] == width,
        "target_rows_zero": not any(nonzero.values()),
        "untied_head_table_present": "lm_head.weight" in tables,
        "verdict": "c16 install_band preconditions hold (width, zero rows, untied head, sidecar digest)",
    }
    receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt["seconds"] = round(time.time() - started, 1)
    sidecar = Path(str(out) + ".json")
    sidecar.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    log(f"table {out} sha256 {receipt['table']['file_sha256']}; sidecar {sidecar} sha256 {sha256_file(sidecar)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    oc = os.environ.get("OC", "/home/cereal/omni-campaign")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trunk-native", default="/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native")
    p.add_argument("--trunk-tokenizer-dir", default="/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07")
    p.add_argument("--trunk-base-vocab", type=int, default=248320)
    p.add_argument("--ace-component", default=f"{oc}/runtime/ace-oct06-native-492961ec-82c3-4bed-98ce-cb6c01af973e/output/ace-lm-full28-development/component")
    p.add_argument("--ace-tokenizer-dir", default=None)
    p.add_argument("--ace-embed-key", default="embed_tokens.weight")
    p.add_argument("--code-begin", type=int, default=None)
    p.add_argument("--code-count", type=int, default=None)
    p.add_argument("--band-name", default="music_codes")
    p.add_argument("--holdout-fraction", type=float, default=0.1)
    p.add_argument("--inner-fraction", type=float, default=0.1)
    p.add_argument("--retrieval-queries", type=int, default=2000)
    p.add_argument("--deviation-calibration", default="median", choices=("median", "none"), help="rescale predicted deviations from the trunk mean to the held anchors' median spread")
    p.add_argument("--min-anchor-factor", type=float, default=4.0, help="refuse with fewer than this many anchors per ACE hidden dim")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--out", default=f"{oc}/composite/s7/c18/music_codes_band.safetensors")
    p.add_argument("--chimera-src", default=os.environ.get("CHIMERA_SRC", f"{oc}/import/rq27b/tree_ef1ffb9/src"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chimera_src and args.chimera_src not in sys.path:
        sys.path.insert(0, str(args.chimera_src))
    try:
        return run(args)
    except Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
