#!/usr/bin/env python3
"""s7_host_rows.py -- R3 of the media-chart pairing: TRUNK side of the 5120-wide chart pairs.

Lane M1, Claude, 2026-10-10. For one S7 media unit, runs the S3 native trunk prefix (blocks
0..insert_after, default 0..31 -- every unit pairs after trunk layer 31 by depth fraction) over

    [ trunk tokens of the forward's prompt | one native media row per parent token ]

with the media rows injected through the S3 bridge exactly as the runtime does
(`ChimeraOmniModel._add_native_media`: placeholder ids from the ledger `media` band, the bridge row
scatter-ADDED at native_media_indices, per-row noise levels), after patching the unit's latent-rung
interface rows (`s7_interface_latent.py`, S6 format) into the in-memory bridge the way
`c16_install_media.install_interface` writes them. The block output x_{insert_after} at each media row
is the chart's HOST row; the parent block-L input for the same token (`s7_pairs_parent.py`) is the
chart's PARENT row. Output (the c16 `--tap-rows` format plus provenance arrays):

    <out>/<unit>/paired_rows.safetensors   host_rows fp32 [N, 5120], parent_rows fp32 [N, W],
                                           held bool [N], group / forward / token / type_ids int64 [N], sigma fp32 [N],
                                           bridge_rows_sample fp32 [S, 5120] + bridge_rows_index int64 [S]
    <out>/<unit>/paired_rows.json          (+ .sha256) provenance and diagnostics

The model is built with the reviewed streamed engine (`run_omni_frozen_review.StreamedChimera`,
imported, not edited): head tensors assigned once (the unused lm_head is dropped after load), blocks
0..insert_after built once each with the engine's own `_build_block` and kept resident. The trunk is
causal, so nothing after the last media row is needed and blocks after insert_after are never built.

Diagnostics only (owner policy 2026-10-06): the prompt-row causality check (prompt rows of x_L are
identical with and without the media rows) and the content-energy share are recorded, never gates.
Structural checks (shapes, finiteness, interface preconditions, row counts) refuse.

1.2.0 (PREP-B, Claude, 2026-10-10): TIME ROWS. Pair sets from s7_pairs_parent >= 1.1.0 carry `time_row`;
a time row is a native media row like any other (placeholder id, bridge row from zero features + type
embed + position + noise level) so the trunk prefix runs over [prompt | time row | latent rows] and its
x_L is the time row's HOST row. The layout contract (runtime_requirements.hy3d_time_token_row: one time
row, FIRST in its forward, owned type, zero features, TIME_ROW_POSITION, the segment sigma) is checked
structurally, and a unit whose charted_block.json declares that requirement is REFUSED pairs without
time rows. `time_row` is written into paired_rows.safetensors.

1.3.0 (S7-on-S5 media lane, Claude, 2026-10-13): `--stream-blocks`. The S5 union child's prefix 0..31
(FFN width 16,640 after C10) is ~24 GiB of BF16 block weights on top of the 3.3 GiB embedding, which
does not fit a 24 GiB RTX 3090 when every block is resident. With `--stream-blocks` the forward is run
LAYER-MAJOR: every batch's input rows are built first, then block i is built once, applied to every
batch (and to the prompt-only causal-check row), and freed before block i+1 is built. The per-batch
arithmetic is the same sequence of block calls on the same tensors as the resident (batch-major) path,
so the host rows are the same numbers; only the residency changes. Without the flag the 1.2.0 path runs
unchanged. The receipt records `trunk.block_residency`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_host_rows"
TOOL_VERSION = "1.3.0"
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
HERE = Path(__file__).resolve().parent
S3_NATIVE = "/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native"
TIME_ROW_REQUIREMENT = "hy3d_time_token_row"


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_hr", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_hr"] = module
    spec.loader.exec_module(module)
    return module


def unit_time_row_requirement(unit: str, units_root: str | Path) -> dict[str, Any] | None:
    """The unit's declared time-row runtime requirement (read from its charted_block.json, never assumed)."""

    unit_dirs = _load_sibling("s7_chart_fit").UNIT_DIRS  # the one unit -> directory map (read from source)
    if unit not in unit_dirs:
        return None
    body_path = Path(units_root) / unit_dirs[unit] / "charted_block.json"
    if not body_path.is_file():
        return None
    body = json.loads(body_path.read_text())
    req = (body.get("runtime_requirements") or {}).get(TIME_ROW_REQUIREMENT)
    return None if req is None else {"unit_json": str(body_path), "unit_json_sha256": sha256_file(body_path), "requirement": req,
                                     "type_ids": [int(t) for t in body["spec"]["type_ids"]]}


def check_time_rows(time_row: Any, forward: Any, token: Any, type_ids: Any, features: Any, widths: Any, positions: Any,
                    sigma: Any, *, time_position: tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0),
                    owned_types: list[int] | None = None, required: bool = False) -> dict[str, Any]:
    """Structural check of the time-row layout contract over a whole pair set; refuses (SystemExit) on breach."""

    import torch

    time_row = time_row.to(torch.bool)
    n_forwards = int(forward.max()) + 1 if forward.numel() else 0
    per_forward = torch.zeros(n_forwards, dtype=torch.long).index_add_(0, forward[time_row], torch.ones(int(time_row.sum()), dtype=torch.long)) if n_forwards else torch.zeros(0, dtype=torch.long)
    total = int(time_row.sum())
    if required and (total == 0 or bool((per_forward != 1).any())):
        raise SystemExit(f"the unit declares {TIME_ROW_REQUIREMENT}: every forward needs exactly one time row "
                         f"(pairs carry {total} over {n_forwards} forwards; rebuild them with s7_pairs_parent >= 1.1.0)")
    if total == 0:
        return {"time_rows": 0, "forwards": n_forwards, "required": bool(required)}
    if bool((per_forward > 1).any()):
        raise SystemExit("a forward carries more than one time row")
    if bool((token[time_row] != 0).any()):
        raise SystemExit("a time row is not the first row of its forward")
    if owned_types is not None and bool((~torch.isin(type_ids[time_row], torch.tensor(owned_types))).any()):
        raise SystemExit(f"a time row's type is not one the unit owns ({owned_types})")
    if bool(features[time_row].ne(0).any()) or bool((widths[time_row] != 0).any()):
        raise SystemExit("a time row carries feature content (the contract: zero features, width 0)")
    expected = torch.tensor(time_position, dtype=positions.dtype).expand(total, -1)
    if not torch.equal(positions[time_row], expected):
        raise SystemExit(f"a time row's position is not {list(time_position)}")
    for f in torch.nonzero(per_forward == 1).flatten().tolist():
        m = forward == f
        if bool((sigma[m] != sigma[m][0]).any()):
            raise SystemExit(f"forward {f}: the time row's sigma differs from its segment's")
    return {"time_rows": total, "forwards": n_forwards, "forwards_with_time_row": int((per_forward == 1).sum()),
            "required": bool(required), "position": list(time_position), "first_in_forward": True, "zero_features": True,
            "sigma": "equal to the segment sigma in every forward"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_review_module():
    spec = importlib.util.spec_from_file_location("run_omni_frozen_review_m1", HERE / "run_omni_frozen_review.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_omni_frozen_review_m1"] = module
    spec.loader.exec_module(module)
    return module


def interface_entries(set_dirs: list[Path], unit: str, type_ids: set[int]) -> list[dict[str, Any]]:
    """Entries of the unit from S6-format interface sets: by their `unit` key (s7_interface_latent), else by
    type id (sets written by other tools, e.g. M2's c15_interface_rows.py, carry no unit key)."""

    found = []
    for set_dir in set_dirs:
        path = set_dir / "companions.json"
        if not path.is_file():
            continue
        body = json.loads(path.read_text())
        for entry in body["entries"]:
            mine = entry.get("unit") == unit if "unit" in entry else int(entry["type_id"]) in type_ids
            if mine:
                found.append({**entry, "_dir": str(set_dir), "_companions": str(path), "_companions_sha256": sha256_file(path),
                              "_selected_by": "unit" if "unit" in entry else "type_id"})
    got = {int(e["type_id"]) for e in found}
    if not found or got != set(type_ids):
        raise SystemExit(f"{[str(d) for d in set_dirs]}: interface entries for {unit} cover types {sorted(got)}, the pair rows need {sorted(type_ids)}")
    return found


def patch_bridge(model: Any, cfg: Any, entries: list[dict[str, Any]], tree_src: str) -> list[dict[str, Any]]:
    """c16_install_media.install_interface's bridge writes, in memory (fp32 arithmetic, stored in the bridge dtype)."""

    import torch
    from chimera_omni.importing.safetensors_io import read_safetensors

    bridge = model.native_media_bridge
    hidden = int(cfg.hidden_size)
    reports = []
    with torch.no_grad():
        for entry in entries:
            path = Path(entry["_dir"]) / entry["rows"]
            views = read_safetensors(path, verify=False)
            for name, digest in (entry.get("tensor_sha256") or {}).items():
                if str(views[name].sha256()) != str(digest):
                    raise SystemExit(f"{path}: {name} sha256 does not match companions.json")
            E = views["bridge.feature_proj.weight"].torch(dtype=torch.float32)
            delta = views["bridge.type_embed_delta"].torch(dtype=torch.float32)
            b, e = (int(v) for v in entry["feature_band"])
            t = int(entry["type_id"])
            pre = {
                "type_feature_scale[type, band] nonzero": int(bridge.type_feature_scale.weight[t, b:e].ne(0).sum()),
                "type_affine[type, scale half] nonzero": int(bridge.type_affine.weight[t, :hidden].ne(0).sum()),
            }
            if any(pre.values()):
                raise SystemExit(f"{entry['name']}: c16 interface preconditions violated {pre}")
            w = bridge.feature_proj.weight
            if tuple(E.shape) != (hidden, e - b):
                raise SystemExit(f"{entry['name']}: rows {tuple(E.shape)} != ({hidden}, {e - b})")
            previous = float(w[:, b:e].float().norm())
            w[:, b:e] = E.to(device=w.device, dtype=w.dtype)
            te = bridge.type_embed.weight
            te[t] = (te[t].float() + delta.to(te.device)).to(te.dtype)
            reports.append({"name": entry["name"], "type_id": t, "feature_band": [b, e], "rows_file": str(path),
                            "rows_sha256": sha256_file(path), "previous_band_norm": previous,
                            "written_band_norm": float(E.norm()), "type_embed_delta_norm": float(delta.norm()), "preconditions": pre})
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", required=True, help="unit name, e.g. ace_dit_L12")
    p.add_argument("--model", default=S3_NATIVE)
    p.add_argument("--tokenizer", default=str(Path(S3_NATIVE).parent / "tokenizer" / "tokenizer.json"))
    p.add_argument("--interface", default=str(OC / "composite/s7/interface_latent"))
    p.add_argument("--interface-c11", default=None, help="C11 set dir (default <interface>/c11)")
    p.add_argument("--pairs", default=str(OC / "composite/s7/pairs"))
    p.add_argument("--out", default=str(OC / "composite/s7/tap_rows_s3"))
    p.add_argument("--insert-after", type=int, default=31)
    p.add_argument("--chimera-src", default=str(OC / "import/rq27b/tree_ef1ffb9"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    p.add_argument("--max-batch-tokens", type=int, default=12288)
    p.add_argument("--bridge-sample", type=int, default=4096)
    p.add_argument("--max-forwards", type=int, default=0, help="smoke: stop after this many forwards")
    p.add_argument("--units-root", default=str(OC / "composite/s7/units"), help="where the unit's charted_block.json (time-row requirement) is read")
    p.add_argument("--stream-blocks", action="store_true",
                   help="layer-major forward with ONE trunk block on the device at a time (1.3.0; for prefixes that do not fit resident)")
    args = p.parse_args(argv)

    review = load_review_module()
    tree = review.bind_chimera_tree(args.chimera_src, None)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    dtype = getattr(torch, args.dtype)
    started = time.time()
    engine = review.StreamedChimera(Path(args.model), device=args.device, dtype=dtype, ledger_band_logit_mask=False)
    model, cfg = engine.model, engine.cfg
    model.lm_head = torch.nn.Identity()  # never used on the prefix; frees ~3.3 GB
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    L = int(args.insert_after)
    if not 0 <= L < int(cfg.num_layers):
        raise SystemExit(f"--insert-after {L} outside the trunk")
    pair_path = Path(args.pairs) / args.unit / "parent_capture.safetensors"
    with safe_open(str(pair_path), "pt") as h:
        unit_types = {int(t) for t in torch.unique(h.get_tensor("type_ids")).tolist()}
    c11_dir = Path(args.interface_c11) if args.interface_c11 else Path(args.interface) / "c11"
    entries = interface_entries([Path(args.interface), c11_dir], args.unit, unit_types)
    patch_reports = patch_bridge(model, cfg, entries, tree["src"])
    blocks = []
    if not args.stream_blocks:
        for index in range(L + 1):
            blocks.append(engine._build_block(index))
    log(f"trunk prefix 0..{L} {'streamed (layer-major)' if args.stream_blocks else 'resident'} on {args.device} ({args.dtype}); "
        f"load {time.time() - started:.0f}s; "
        f"cuda alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if args.device.startswith("cuda") else "cpu")
    placeholder = int(cfg.ledger().range("media").begin)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    pair_json = json.loads((pair_path.parent / "parent_capture.json").read_text())
    with safe_open(str(pair_path), "pt") as h:
        pairs = {k: h.get_tensor(k) for k in h.keys()}
    n_rows = int(pairs["features"].shape[0])
    forward_of = pairs["forward"]
    time_row = pairs["time_row"].to(torch.bool) if "time_row" in pairs else torch.zeros(n_rows, dtype=torch.bool)
    requirement = unit_time_row_requirement(args.unit, args.units_root)
    time_check = check_time_rows(time_row, forward_of, pairs["token"], pairs["type_ids"], pairs["features"], pairs["feature_width"],
                                 pairs["positions"], pairs["sigma"], time_position=tuple(_load_sibling("s7_pairs_parent").TIME_ROW_POSITION),
                                 owned_types=(requirement or {}).get("type_ids"), required=requirement is not None)
    time_check["unit_requirement"] = None if requirement is None else {k: requirement[k] for k in ("unit_json", "unit_json_sha256")}
    log(f"time rows: {time_check}")
    bands = {int(e["type_id"]): [int(v) for v in e["feature_band"]] for e in entries}
    forwards = pair_json["forwards"]
    if args.max_forwards:
        forwards = forwards[: int(args.max_forwards)]
    prompt_cache: dict[str, list[int]] = {}
    plan = []
    starts = torch.searchsorted(forward_of, torch.arange(len(pair_json["forwards"]) + 1))
    for f in forwards:
        i = int(f["index"])
        a, b = int(starts[i]), int(starts[i + 1])
        if b - a != int(f["rows"]):
            raise SystemExit(f"forward {i}: {b - a} pair rows vs declared {f['rows']}")
        ids = prompt_cache.setdefault(f["prompt"], tokenizer.encode(f["prompt"], add_special_tokens=False).ids)
        plan.append({"forward": i, "rows": (a, b), "prompt_ids": ids})
    # batch forwards of identical (prompt length, media rows) under a token budget
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in plan:
        buckets.setdefault((len(item["prompt_ids"]), item["rows"][1] - item["rows"][0]), []).append(item)
    batches = []
    for (plen, nm), items in sorted(buckets.items()):
        per = max(1, int(args.max_batch_tokens) // (plen + nm))
        for k in range(0, len(items), per):
            batches.append(items[k : k + per])
    feature_dim = int(cfg.native_feature_dim)
    host = torch.zeros(n_rows, int(cfg.hidden_size), dtype=torch.float32)
    filled = torch.zeros(n_rows, dtype=torch.bool)
    generator = torch.Generator().manual_seed(0)
    sample_set = set(torch.randperm(n_rows, generator=generator)[: int(args.bridge_sample)].tolist())
    bridge_rows: dict[int, Any] = {}
    content_share = []
    causal_check = None
    def prepare(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Embedding + native media rows of one batch (the input of block 0) and its diagnostics."""

        plen = len(batch[0]["prompt_ids"])
        nm = batch[0]["rows"][1] - batch[0]["rows"][0]
        B = len(batch)
        ids = torch.tensor([item["prompt_ids"] + [placeholder] * nm for item in batch], dtype=torch.long, device=args.device)
        feats = torch.zeros(B, nm, feature_dim, dtype=torch.float32)
        types = torch.zeros(B, nm, dtype=torch.long)
        pos = torch.zeros(B, nm, 4, dtype=torch.float32)
        sig = torch.zeros(B, nm, dtype=torch.float32)
        for r, item in enumerate(batch):
            a, b = item["rows"]
            t_rows = pairs["type_ids"][a:b]
            content = ~time_row[a:b]
            for t in torch.unique(t_rows).tolist():
                lo, hi = bands[int(t)]
                m = (t_rows == t) & content
                widths = pairs["feature_width"][a:b][m]
                if bool((widths != hi - lo).any()):
                    raise SystemExit(f"forward {item['forward']}: feature width != band width {hi - lo} for type {t}")
                feats[r, m, lo:hi] = pairs["features"][a:b][m][:, : hi - lo]
            types[r] = t_rows
            pos[r] = pairs["positions"][a:b]
            sig[r] = pairs["sigma"][a:b]
        if bool((sig < 0).any()) and not bool((sig < 0).all()):
            raise SystemExit("mixed sigma / no-sigma rows in one forward")
        noise = None if bool((sig < 0).all()) else sig.to(args.device)
        indices = torch.arange(plen, plen + nm, device=args.device).view(1, -1).expand(B, -1)
        valid_media = torch.ones(B, nm, dtype=torch.bool, device=args.device)
        token_valid = torch.ones(B, plen + nm, dtype=torch.bool, device=args.device)
        x = model.embed(ids)
        media_x = model.native_media_bridge.embed(feats.to(args.device), types.to(args.device), pos.to(args.device), valid_media, noise)
        x, count = model._add_native_media(x, feats.to(args.device), types.to(args.device), pos.to(args.device), indices, valid_media, noise)
        if count != nm:
            raise SystemExit("media count mismatch")
        bridge = model.native_media_bridge
        content = torch.nn.functional.linear(feats.to(args.device, bridge.feature_proj.weight.dtype), bridge.feature_proj.weight).float()
        pre = content + bridge.type_embed(types.to(args.device)).float()
        share = float((content.norm(dim=-1) / pre.norm(dim=-1).clamp_min(1e-30)).median())
        x = torch.where(token_valid.unsqueeze(-1), x, torch.zeros_like(x))
        return {"batch": batch, "plen": plen, "nm": nm, "B": B, "ids": ids, "x": x, "token_valid": token_valid,
                "media_x": media_x.float().cpu(), "share": share}

    def collect(bi: int, item_set: dict[str, Any]) -> None:
        x = item_set["x"]
        if not bool(torch.isfinite(x).all()):
            raise SystemExit(f"non-finite trunk rows in batch {bi}")
        plen = item_set["plen"]
        out_rows = x[:, plen:, :].float().cpu()
        mx = item_set["media_x"]
        for r, item in enumerate(item_set["batch"]):
            a, b = item["rows"]
            host[a:b] = out_rows[r]
            filled[a:b] = True
            for j in range(a, b):
                if j in sample_set:
                    bridge_rows[j] = mx[r, j - a]

    residency = "resident (batch-major: every block 0..L on the device; each batch runs the whole prefix)"
    with torch.no_grad():
        if args.stream_blocks:
            residency = ("streamed (layer-major: one block on the device at a time; block i is applied to every batch "
                         "and to the prompt-only causal row, then freed, before block i+1 is built)")
            prepared = []
            for bi, batch in enumerate(batches):
                prepared.append(prepare(batch))
                content_share.append(prepared[-1]["share"])
            first = prepared[0]
            xp = model.embed(first["ids"][:1, : first["plen"]])
            for index in range(L + 1):
                block = engine._build_block(index)
                for item_set in prepared:
                    y = block(item_set["x"], token_valid_mask=item_set["token_valid"])
                    item_set["x"] = torch.where(item_set["token_valid"].unsqueeze(-1), y, torch.zeros_like(y))
                xp = block(xp, token_valid_mask=first["token_valid"][:1, : first["plen"]])
                del block, y
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if index % 4 == 0 or index == L:
                    log(f"block {index}/{L} applied to {len(prepared)} batches ({time.time() - started:.0f}s; "
                        + (f"cuda alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB)" if args.device.startswith("cuda") else "cpu)"))
            diff = (xp.float() - first["x"][:1, : first["plen"]].float()).abs().max()
            causal_check = {"prompt_tokens": first["plen"], "max_abs_diff_prompt_rows_with_vs_without_media": float(diff),
                            "prompt_row_norm_median": float(xp.float().norm(dim=-1).median())}
            for bi, item_set in enumerate(prepared):
                collect(bi, item_set)
                item_set["x"] = None
            log(f"streamed prefix done: rows {int(filled.sum())}/{n_rows} ({time.time() - started:.0f}s)")
        for bi, batch in enumerate([] if args.stream_blocks else batches):
            plen = len(batch[0]["prompt_ids"])
            nm = batch[0]["rows"][1] - batch[0]["rows"][0]
            B = len(batch)
            ids = torch.tensor([item["prompt_ids"] + [placeholder] * nm for item in batch], dtype=torch.long, device=args.device)
            feats = torch.zeros(B, nm, feature_dim, dtype=torch.float32)
            types = torch.zeros(B, nm, dtype=torch.long)
            pos = torch.zeros(B, nm, 4, dtype=torch.float32)
            sig = torch.zeros(B, nm, dtype=torch.float32)
            for r, item in enumerate(batch):
                a, b = item["rows"]
                t_rows = pairs["type_ids"][a:b]
                content = ~time_row[a:b]  # a time row carries no features (checked above); it still gets its band's zeros
                for t in torch.unique(t_rows).tolist():
                    lo, hi = bands[int(t)]
                    m = (t_rows == t) & content
                    widths = pairs["feature_width"][a:b][m]
                    if bool((widths != hi - lo).any()):
                        raise SystemExit(f"forward {item['forward']}: feature width != band width {hi - lo} for type {t}")
                    feats[r, m, lo:hi] = pairs["features"][a:b][m][:, : hi - lo]
                types[r] = t_rows
                pos[r] = pairs["positions"][a:b]
                sig[r] = pairs["sigma"][a:b]
            if bool((sig < 0).any()) and not bool((sig < 0).all()):
                raise SystemExit("mixed sigma / no-sigma rows in one forward")
            noise = None if bool((sig < 0).all()) else sig.to(args.device)
            indices = torch.arange(plen, plen + nm, device=args.device).view(1, -1).expand(B, -1)
            valid_media = torch.ones(B, nm, dtype=torch.bool, device=args.device)
            token_valid = torch.ones(B, plen + nm, dtype=torch.bool, device=args.device)
            x = model.embed(ids)
            media_x = model.native_media_bridge.embed(feats.to(args.device), types.to(args.device), pos.to(args.device), valid_media, noise)
            x, count = model._add_native_media(x, feats.to(args.device), types.to(args.device), pos.to(args.device), indices, valid_media, noise)
            if count != nm:
                raise SystemExit("media count mismatch")
            # energy share of the band content inside the bridge row before RMSNorm (diagnostic)
            bridge = model.native_media_bridge
            content = torch.nn.functional.linear(feats.to(args.device, bridge.feature_proj.weight.dtype), bridge.feature_proj.weight).float()
            pre = content + bridge.type_embed(types.to(args.device)).float()
            content_share.append(float((content.norm(dim=-1) / pre.norm(dim=-1).clamp_min(1e-30)).median()))
            x = torch.where(token_valid.unsqueeze(-1), x, torch.zeros_like(x))
            for block in blocks:
                x = block(x, token_valid_mask=token_valid)
                x = torch.where(token_valid.unsqueeze(-1), x, torch.zeros_like(x))
            if not bool(torch.isfinite(x).all()):
                raise SystemExit(f"non-finite trunk rows in batch {bi}")
            if causal_check is None:
                xp = model.embed(ids[:1, :plen])
                for block in blocks:
                    xp = block(xp, token_valid_mask=token_valid[:1, :plen])
                diff = (xp.float() - x[:1, :plen].float()).abs().max()
                causal_check = {"prompt_tokens": plen, "max_abs_diff_prompt_rows_with_vs_without_media": float(diff),
                                "prompt_row_norm_median": float(xp.float().norm(dim=-1).median())}
            out_rows = x[:, plen:, :].float().cpu()
            mx = media_x.float().cpu()
            for r, item in enumerate(batch):
                a, b = item["rows"]
                host[a:b] = out_rows[r]
                filled[a:b] = True
                for j in range(a, b):
                    if j in sample_set:
                        bridge_rows[j] = mx[r, j - a]
            if bi % 10 == 0 or bi == len(batches) - 1:
                done = int(filled.sum())
                log(f"batch {bi + 1}/{len(batches)} B={B} seq={plen + nm}: rows {done}/{n_rows} ({time.time() - started:.0f}s)")
    rows_done = int(filled.sum())
    if not args.max_forwards and rows_done != n_rows:
        raise SystemExit(f"only {rows_done}/{n_rows} rows filled")
    keep = filled
    out_dir = Path(args.out) / args.unit
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_index = torch.tensor(sorted(bridge_rows), dtype=torch.long)
    state = {
        "host_rows": host[keep].contiguous(),
        "parent_rows": pairs["parent_rows"][keep].to(torch.float32).contiguous(),
        "held": pairs["held"][keep].contiguous(),
        "group": pairs["group"][keep].contiguous(),
        "forward": pairs["forward"][keep].contiguous(),
        "token": pairs["token"][keep].contiguous(),
        "type_ids": pairs["type_ids"][keep].contiguous(),
        "sigma": pairs["sigma"][keep].contiguous(),
        "time_row": time_row[keep].contiguous(),
        "bridge_rows_sample": torch.stack([bridge_rows[i] for i in sorted(bridge_rows)]) if bridge_rows else torch.zeros(0, int(cfg.hidden_size)),
        "bridge_rows_index": sample_index,
    }
    path = out_dir / "paired_rows.safetensors"
    tmp = path.with_suffix(".tmp")
    save_file(state, str(tmp))
    os.replace(tmp, path)
    model_dir = Path(args.model)
    receipt = {
        "schema": "omni-campaign.s7_host_rows.v1",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__),
                 "engine": {"module": str(HERE / "run_omni_frozen_review.py"), "sha256": sha256_file(HERE / "run_omni_frozen_review.py"), "class": "StreamedChimera"}},
        "unit": args.unit,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": vars(args),
        "chimera_tree": tree,
        "trunk": {"directory": str(model_dir), "config_sha256": sha256_file(model_dir / "config.json"),
                  "index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
                  "s3_native_receipt_sha256": sha256_file(model_dir.parent / "s3_native_receipt.json") if (model_dir.parent / "s3_native_receipt.json").is_file() else None,
                  "blocks_run": [0, L], "block_kinds": [cfg.block_kind(i) for i in range(L + 1)], "dtype": args.dtype, "device": args.device,
                  "host_site": f"block {L} OUTPUT (the residual the charted block inserted after layer {L} reads)",
                  "block_residency": residency, "block_loads": int(engine.block_loads)},
        "interface": {"entries": patch_reports, "companions": sorted({e["_companions"] for e in entries}),
                      "companions_sha256": sorted({e["_companions_sha256"] for e in entries}),
                      "selected_by": sorted({e["_selected_by"] for e in entries})},
        "pairs": {"file": str(pair_path), "sha256": sha256_file(pair_path), "json_sha256": sha256_file(pair_path.parent / "parent_capture.json")},
        "sequence_layout": {"order": "[prompt tokens | media rows]", "placeholder_token_id": placeholder, "placeholder_band": "media",
                            "noise_levels": "per-row sigma" if bool((pairs["sigma"] >= 0).all()) else "None (encoder rows)",
                            "tokenizer": args.tokenizer, "tokenizer_sha256": sha256_file(args.tokenizer), "add_special_tokens": False},
        "file": str(path), "file_sha256": sha256_file(path),
        "rows": int(keep.sum()), "rows_held": int(state["held"].sum()),
        "rows_time": int(state["time_row"].sum()),
        "time_rows": time_check,
        "forwards": len(plan), "batches": len(batches),
        "diagnostics": {
            "time_row_host_norm": ([float(v) for v in state["host_rows"][state["time_row"]].norm(dim=1)] if bool(state["time_row"].any()) else None),
            "time_row_parent_norm": ([float(v) for v in state["parent_rows"][state["time_row"]].norm(dim=1)] if bool(state["time_row"].any()) else None),
            "causal_prompt_check": causal_check,
            "content_share_pre_norm_median_per_batch": {"min": min(content_share), "median": sorted(content_share)[len(content_share) // 2], "max": max(content_share)},
            "host_row_norm_median": float(state["host_rows"].norm(dim=1).median()),
            "parent_row_norm_median": float(state["parent_rows"].norm(dim=1).median()),
            "bridge_row_norm_median": float(state["bridge_rows_sample"].norm(dim=1).median()) if bridge_rows else None,
        },
        "seconds": time.time() - started,
        "cuda_peak_gib": (torch.cuda.max_memory_allocated() / 2**30) if args.device.startswith("cuda") else None,
        "capability_claim": "none; host rows for a chart fit",
    }
    jpath = out_dir / "paired_rows.json"
    jpath.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (out_dir / "paired_rows.json.sha256").write_text(sha256_file(jpath) + "\n", encoding="utf-8")
    log(f"wrote {path} rows {receipt['rows']} in {receipt['seconds']:.0f}s; causal check {causal_check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
