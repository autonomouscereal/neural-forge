#!/usr/bin/env python3
"""s7_interface_latent.py -- R1 of the media-chart pairing: LATENT-RUNG C15/C11 interface rows at hidden 5120.

Lane M1, Claude, 2026-10-10. Writes an interface-row set in the S6 format
(`COMPOSITE_S6_INPUT_FORMATS.md` section 1; consumed by `c16_install_media.py --c15 DIR`):

    <out>/companions.json
    <out>/<entry>/composed_interface.safetensors   bridge.feature_proj.weight [H, e-b], bridge.type_embed_delta [H],
                                                   head.out.weight [e'-b', H], head.type_affine_shift [e'-b']
    <out>/b15/<unit>.safetensors                   the B15 chart pair used (B_in [H, W], B_out [W, H]) + its PCA inputs

One native row = ONE parent token; its feature is the parent's own patch-embed INPUT (latent rung), so

    feature_proj[:, band] = B_in @ P_src          (E = identity; P_src = the parent's patch projection)
    type_embed[type]     += B_in @ p_src_bias
    head.out[band', :]    = P_out @ B_out          (D = identity)
    head shift            = p_out_bias             (c16 solves the hidden-space shift by least squares)

B_in = g * U_trunk^T V_parent: the top-r principal directions of the parent's patch-embedded tokens
(P_src f + b over the unit's FIT pair rows, `s7_pairs_parent.py`) mapped onto the top-r principal
directions of the trunk's layer-0 input rows (C17 taps L00 `prenorm`), r = min(W, H). Nothing pairs at
layer 0, so this chart is a DESIGN CHOICE (an isometry up to g), not a fit. g is chosen so the median
content norm |B_in (P_src f + b)| equals the norm of the S3 bridge's own type_embed row for that type
(the random S0 row every media row of that type carries), i.e. after the bridge RMSNorm the latent
content is not drowned by the type row. B_out = V^T U / g (the pseudo-inverse); the emission rows are a
stand-in until the optional post-assembly C15b refit.

Bands (R0(a), disjoint, feature side then emission side; F = 6144):
    music  4  ACE    f[0,384)    e[0,128)
    video  2  H3     f[384,480)  e[128,224)
    audio  3  H3     f[480,512)  e[224,256)
    image  1  QIMG   f[512,576)  e[256,320)
    volume 7  HY3D   f[576,640)  e[320,384)
    ocr    6  QVL32  f[640,2176) e[384,385)   (C11 encoder; one ZERO emission row because c16 needs b < e)

Every numeric value in the receipt is a diagnostic. CPU only; honours OMP_NUM_THREADS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_interface_latent"
TOOL_VERSION = "1.0.0"
INTERFACE_SCHEMA = "chimera_omni.composite.interface_set.v1"
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
HIDDEN = 5120
FEATURE_DIM = 6144
S3_NATIVE = Path("/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native")
TRUNK_TAPS_L00 = OC / "composite/c17/trunk_taps/s3native-c2k4352-held-all/L00/ffn_input_rows.safetensors"

ACE_W = "/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors"
H3_W = "/home/cereal/comfyui/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
QIMG_W = "/home/cereal/comfyui/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
HY3D_W = "/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors"
QVL_W = "/home/cereal/comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"

# unit -> (parent width, entries); each entry: name, source, type_id, feature band, emission band, feature width,
# emission width, P_src keys, P_out keys
UNITS: dict[str, dict[str, Any]] = {
    "ace_dit_L12": {"width": 2048, "weights": ACE_W, "entries": [
        {"name": "music_ace", "source": "ACE", "type_id": 4, "feature_band": [0, 384], "emission_band": [0, 128],
         "p_src": ("decoder.proj_in.1.weight", "decoder.proj_in.1.bias", "conv1d_k2"),
         "p_out": ("decoder.proj_out.1.weight", "decoder.proj_out.1.bias", "convt1d_k2")}]},
    "h3_dit_L25_curve6": {"width": 5376, "weights": H3_W, "entries": [
        {"name": "video_h3", "source": "H3", "type_id": 2, "feature_band": [384, 480], "emission_band": [128, 224],
         "p_src": ("video_patch_proj.weight", "video_patch_proj.bias", "linear"),
         "p_out": ("final_layer.video_out.weight", "final_layer.video_out.bias", "linear")},
        {"name": "audio_h3", "source": "H3", "type_id": 3, "feature_band": [480, 512], "emission_band": [224, 256],
         "p_src": ("audio_patch_proj.weight", "audio_patch_proj.bias", "linear"),
         "p_out": ("final_layer.audio_out.weight", "final_layer.audio_out.bias", "linear")}]},
    "qimg_mmdit_L30_img": {"width": 3072, "weights": QIMG_W, "entries": [
        {"name": "image_qimg", "source": "QIMG", "type_id": 1, "feature_band": [512, 576], "emission_band": [256, 320],
         "p_src": ("img_in.weight", "img_in.bias", "linear"), "p_out": ("proj_out.weight", "proj_out.bias", "linear")}]},
    "hy3d_dit_L10": {"width": 2048, "weights": HY3D_W, "entries": [
        {"name": "volume_hy3d", "source": "HY3D", "type_id": 7, "feature_band": [576, 640], "emission_band": [320, 384],
         "p_src": ("model.x_embedder.weight", "model.x_embedder.bias", "linear"),
         "p_out": ("model.final_layer.linear.weight", "model.final_layer.linear.bias", "linear")}]},
    "qvl32_vit_L13": {"width": 1152, "weights": QVL_W, "set": "c11", "entries": [
        {"name": "ocr_qvl32", "source": "QVL32", "type_id": 6, "feature_band": [640, 2176], "emission_band": [384, 385],
         "p_src": ("visual.patch_embed.proj.weight", "visual.patch_embed.proj.bias", "conv3d_flat"), "p_out": None}]},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    return torch


def bind_chimera(tree: str) -> None:
    src = Path(tree) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def bind_mediafold() -> None:
    media = OC / "media"
    if str(media) not in sys.path:
        sys.path.insert(0, str(media))


def read_tensor(path: str, key: str) -> Any:
    """One tensor widened to fp64 (fp8 / bf16 / fp16 widen exactly); a sibling scale key is refused."""

    torch = _torch()
    from safetensors import safe_open

    with safe_open(path, "pt") as h:
        keys = set(h.keys())
        for scale in (key + "_scale", key.rsplit(".", 1)[0] + ".scale_weight"):
            if scale in keys:
                raise SystemExit(f"{path}: {key} carries a scale tensor {scale}; widening alone would be wrong")
        return h.get_tensor(key).to(torch.float64)


def source_projection(weights: str, spec: tuple[str, str, str]) -> tuple[Any, Any]:
    """P_src [W, feature width] and its bias [W] in the token-feature order s7_pairs_parent writes."""

    w = read_tensor(weights, spec[0])
    b = read_tensor(weights, spec[1])
    kind = spec[2]
    if kind == "linear":
        return w, b
    if kind == "conv1d_k2":  # [out, in, k] -> feature index in*2 + k
        return w.reshape(w.shape[0], -1), b
    if kind == "conv3d_flat":  # [out, C, T, P, P] -> flattened (C, T, P, P)
        return w.reshape(w.shape[0], -1), b
    raise SystemExit(f"unknown projection kind {kind}")


def output_projection(weights: str, spec: tuple[str, str, str] | None, width: int) -> tuple[Any, Any]:
    torch = _torch()
    if spec is None:
        return None, None
    w = read_tensor(weights, spec[0])
    b = read_tensor(weights, spec[1])
    if spec[2] == "linear":
        return w, b
    if spec[2] == "convt1d_k2":  # ConvTranspose1d weight [in=W, out=c, k] -> P_out[c*2+k, i]
        p = w.permute(1, 2, 0).reshape(w.shape[1] * w.shape[2], w.shape[0])
        return p, b.repeat_interleave(w.shape[2])
    raise SystemExit(f"unknown output projection kind {spec[2]}")


def pca_directions(rows: Any, rank: int) -> tuple[Any, dict[str, Any]]:
    """Top-`rank` principal directions [rank, width] of `rows` (centred), via the fp64 Gram eigendecomposition."""

    torch = _torch()
    x = rows.to(torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    gram = xc.T @ xc
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    keep = min(int(rank), vectors.shape[1])
    total = float(values.clamp_min(0).sum())
    directions = vectors[:, :keep].T.contiguous()
    # deterministic sign: largest-magnitude coordinate positive
    signs = torch.sign(directions.gather(1, directions.abs().argmax(dim=1, keepdim=True))).clamp(min=0) * 2 - 1
    directions = directions * signs
    return directions, {"rank": keep, "rows": int(x.shape[0]), "width": int(x.shape[1]),
                        "retained_energy": float(values[:keep].clamp_min(0).sum()) / max(total, 1e-300),
                        "mean_norm": float(mean.norm())}


def trunk_type_rows(tree: str) -> Any:
    torch = _torch()
    bind_chimera(tree)
    from chimera_omni.importing.safetensors_io import read_index, read_safetensors

    name = "native_media_bridge.type_embed.weight"
    views = read_safetensors(S3_NATIVE, [name], verify=False)
    return views[name].torch(dtype=torch.float64)


def tensor_sha256_table(path: Path, tree: str) -> dict[str, str]:
    bind_chimera(tree)
    from chimera_omni.importing.safetensors_io import read_safetensors

    return {name: str(view.sha256()) for name, view in sorted(read_safetensors(path, verify=False).items())}


def build(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(TRUNK_TAPS_L00), "pt") as h:
        trunk_rows = h.get_tensor(args.trunk_key).to(torch.float64)
    trunk_dirs, trunk_notes = pca_directions(trunk_rows, HIDDEN)
    trunk_notes["file"] = str(TRUNK_TAPS_L00)
    trunk_notes["file_sha256"] = sha256_file(TRUNK_TAPS_L00)
    trunk_notes["key"] = args.trunk_key
    log(f"trunk PCA: {trunk_notes['rows']} rows, rank {trunk_notes['rank']}")
    type_rows = trunk_type_rows(args.chimera_src)
    units = [u for u in args.units.split(",") if u]
    sets: dict[str, list[dict[str, Any]]] = {"c15": [], "c11": []}
    reports: dict[str, Any] = {}
    for unit in units:
        spec = UNITS[unit]
        width = int(spec["width"])
        pair_path = Path(args.pairs) / unit / "parent_capture.safetensors"
        with safe_open(str(pair_path), "pt") as h:
            feats = h.get_tensor("features").to(torch.float64)
            fw = h.get_tensor("feature_width")
            types = h.get_tensor("type_ids")
            held = h.get_tensor("held")
        embedded = []
        projections = {}
        for entry in spec["entries"]:
            p_src, p_bias = source_projection(spec["weights"], entry["p_src"])
            if p_src.shape[0] != width:
                raise SystemExit(f"{unit}/{entry['name']}: P_src rows {p_src.shape[0]} != parent width {width}")
            fwidth = entry["feature_band"][1] - entry["feature_band"][0]
            if p_src.shape[1] != fwidth:
                raise SystemExit(f"{unit}/{entry['name']}: P_src columns {p_src.shape[1]} != band width {fwidth}")
            mask = (types == entry["type_id"]) & (~held)
            if bool((fw[types == entry["type_id"]] != fwidth).any()):
                raise SystemExit(f"{unit}/{entry['name']}: pair rows carry feature widths != {fwidth}")
            rows = feats[mask][:, :fwidth] @ p_src.T + p_bias
            embedded.append(rows)
            projections[entry["name"]] = (p_src, p_bias)
        parent_embedded = torch.cat(embedded, dim=0)
        rank = min(width, HIDDEN)
        parent_dirs, parent_notes = pca_directions(parent_embedded, rank)
        U = trunk_dirs[:rank]  # [r, H]
        Vd = parent_dirs  # [r, W]
        iso = U.T @ Vd  # [H, W]
        content = (parent_embedded @ iso.T).norm(dim=1)
        type_id0 = spec["entries"][0]["type_id"]
        type_norm = float(type_rows[type_id0].norm())
        gain = type_norm / float(content.median()) if args.gain == "auto" else float(args.gain)
        B_in = gain * iso
        B_out = (1.0 / gain) * iso.T
        b15_path = out / "b15" / f"{unit}.safetensors"
        b15_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"B_in": B_in.to(torch.float32).contiguous(), "B_out": B_out.to(torch.float32).contiguous(),
                   "U_trunk": U.to(torch.float32).contiguous(), "V_parent": Vd.to(torch.float32).contiguous()}, str(b15_path))
        unit_report = {
            "parent_width": width, "rank": rank, "gain": gain, "gain_rule": "median |B (P_src f + b)| == |type_embed[type]| of the S3 bridge" if args.gain == "auto" else "explicit",
            "type_embed_norm": type_norm, "content_norm_median_before_gain": float(content.median()),
            "parent_pca": parent_notes, "b15_file": str(b15_path), "b15_sha256": sha256_file(b15_path),
            "pairs_file": str(pair_path), "pairs_sha256": sha256_file(pair_path), "entries": [],
        }
        for entry in spec["entries"]:
            p_src, p_bias = projections[entry["name"]]
            fwidth = p_src.shape[1]
            e_b, e_e = entry["emission_band"]
            ewidth = e_e - e_b
            if entry["p_out"] is None:
                head_out = torch.zeros(ewidth, HIDDEN, dtype=torch.float64)
                out_bias = torch.zeros(ewidth, dtype=torch.float64)
                emission_note = "encoder-only (C11): one ZERO emission row; nothing is emitted for this type"
            else:
                p_out, out_bias = output_projection(spec["weights"], entry["p_out"], width)
                if p_out.shape != (ewidth, width):
                    raise SystemExit(f"{unit}/{entry['name']}: P_out {tuple(p_out.shape)} != ({ewidth}, {width})")
                head_out = p_out @ B_out
                emission_note = "P_out @ B_out (stand-in pseudo-inverse chart; refit after assembly = C15b)"
            # mediafold.c15_fit.compose_interface with E = D = identity, written out because the
            # feature and emission widths differ (ACE reads 384 = context|noisy x 2 frames, emits 128)
            state = {
                "bridge.feature_proj.weight": (B_in @ p_src).to(torch.float32),
                "bridge.type_embed_delta": (B_in @ p_bias).to(torch.float32),
                "head.out.weight": head_out.to(torch.float32),
                "head.type_affine_shift": out_bias.to(torch.float32),
            }
            composed_notes = {"formula": "feature_proj = B_in P_src; type_embed_delta = B_in p_src_bias; head.out = P_out B_out; shift = p_out_bias",
                              "shapes": {k: list(v.shape) for k, v in state.items()},
                              "parameters": int(sum(v.numel() for v in state.values()))}
            entry_dir = out / entry["name"]
            entry_dir.mkdir(parents=True, exist_ok=True)
            rows_path = entry_dir / "composed_interface.safetensors"
            save_file({k: v.contiguous() for k, v in state.items()}, str(rows_path))
            digests = tensor_sha256_table(rows_path, args.chimera_src)
            set_name = spec.get("set", "c15")
            sets[set_name].append({
                "name": entry["name"], "source": entry["source"], "type_id": entry["type_id"],
                "feature_band": entry["feature_band"], "emission_band": entry["emission_band"],
                "rows": f"{entry['name']}/composed_interface.safetensors", "rung": "latent (one native row = one parent token)",
                "companion_decoder": entry["p_out"] is not None, "psnr_absorbed_db": None, "psnr_parent_roundtrip_db": None,
                "tensor_sha256": digests, "unit": unit,
            })
            unit_report["entries"].append({
                "name": entry["name"], "type_id": entry["type_id"], "rows_file": str(rows_path), "rows_sha256": sha256_file(rows_path),
                "p_src": list(entry["p_src"][:2]), "p_out": list(entry["p_out"][:2]) if entry["p_out"] else None,
                "notes": composed_notes, "emission": emission_note,
                "norms": {k: float(v.norm()) for k, v in state.items()},
            })
            log(f"{unit}/{entry['name']}: band f{entry['feature_band']} e{entry['emission_band']} gain {gain:.4g} rank {rank}")
        reports[unit] = unit_report
    written = {}
    for set_name, entries in sets.items():
        if not entries:
            continue
        set_dir = out if set_name == "c15" else out / "c11"
        if set_name == "c11":
            # c16 reads --c11 as its own directory: give it a self-contained copy of the rows
            for e in entries:
                src = out / e["rows"]
                dst = set_dir / e["rows"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        body = {"schema": INTERFACE_SCHEMA, "hidden": HIDDEN, "native_feature_dim": FEATURE_DIM, "entries": entries,
                "producer": {"tool": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)}}
        path = set_dir / "companions.json"
        path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
        written[set_name] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {
        "schema": "omni-campaign.s7_interface_latent.v1",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)},
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hidden": HIDDEN, "native_feature_dim": FEATURE_DIM, "trunk_pca": trunk_notes,
        "s3_type_embed": {"checkpoint": str(S3_NATIVE), "norms": [float(r.norm()) for r in type_rows]},
        "sets": written, "units": reports,
        "decision_record": "R0(a) latent-rung bands; B15 = PCA-to-PCA isometry x gain (design choice, nothing pairs at layer 0); "
                           "QVL32 as C11 type 6 with one zero emission row",
        "capability_claim": "none; interface rows a chart pairing needs, not a fitted absorption",
    }
    rpath = out / "interface_receipt.json"
    rpath.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (out / "interface_receipt.json.sha256").write_text(sha256_file(rpath) + "\n", encoding="utf-8")
    log(f"wrote {rpath}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--units", default=",".join(UNITS))
    p.add_argument("--pairs", default=str(OC / "composite/s7/pairs"))
    p.add_argument("--out", default=str(OC / "composite/s7/interface_latent"))
    p.add_argument("--trunk-key", default="prenorm")
    p.add_argument("--gain", default="auto")
    p.add_argument("--chimera-src", default=str(OC / "import/rq27b/tree_ef1ffb9"))
    args = p.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
