#!/usr/bin/env python3
"""s7_omni1_check.py -- load / identity / forward check of the WRITTEN S7 composite (OMNI1).

OMNI1 assembly lane, Claude, 2026-10-13.  The composite is ~55 GB of BF16, more than one GPU holds, and the
tree's low-memory loader refuses charted blocks, so the check streams it layer-major (one block on --device at a
time) and proves the load in pieces that together cover every tensor exactly once:

  1. structural load  -- ``ChimeraOmniModel(cfg)`` on the meta device; its state names / shapes must equal the
     checkpoint headers exactly (missing / unexpected / shape-mismatched all empty).  Every module is then built
     for real and loaded with ``load_state_dict(strict=True)`` from SHA-VERIFIED tensors
     (``LazySafetensorsMapping(verify=True)`` checks each tensor against the index's sha256 pin): the top-level
     modules once, each block as the pass reaches it.  The union of the loaded names must equal the header set.
     H3's FP32 coefficients are loaded into the FP32 islands ``ChartedSourceBlock._apply`` keeps.
  2. accounting       -- ``oc_seal.safetensors_parameter_count`` of the composite against the accounting parent
     (--parent-parameter-count) and against the S5 child + the charted state.
  3. identity vs S5   -- every trunk block of OMNI1 is bit-compared (torch.equal per tensor) with the S5 child's
     block it was copied from (the manifest's trunk_layer_map); every non-block tensor is compared and its
     changed rows / columns are listed against the manifest's declared writes (C18 band rows, C15/C11 interface
     columns / rows / type rows).
  4. text forward     -- the 8 R-Q prompts through OMNI1, the S5 child and the 27B parent S0 import in the same
     layer-major pass; top-1 agreement over the base vocab [0, band_begin) of OMNI1 vs S5 and vs the parent, the
     S5 vs parent agreement (a cross-check of this harness against the S5 receipts), max |logit diff| OMNI1 vs S5,
     and how many OMNI1 positions would pick a band token if the band were not masked.  Charted blocks must be
     the identity on text (exact zeros written).  All DIAGNOSTICS except finiteness.
  5. media forward    -- for every installed charted block, ONE held forward of its own S5 pair set, laid out as
     ``s7_host_rows`` laid it out ([prompt | media rows], placeholder ids, the OMNI1 bridge -- with the C15/C11
     rows c16 wrote, nothing patched in memory), run through the WHOLE composite (all trunk blocks and all five
     charted blocks, ``_charted_layout`` routing, H3 source positions zeros as in the plan check).  Per block and
     segment: rows owned, write finite, write non-zero on its own segment, exact zeros off its rows and on the
     other units' segments.  Diagnostics: x_31 at the media rows vs the S5 host rows the charts were fitted on,
     the write / input norm ratio, final logits and ``native_media_head`` finiteness.  A block that writes
     nothing on its own segment or writes non-finite values is a refusal (a dead block is not installed).

Owner policy 2026-10-06: finite fit numbers are diagnostics; structure / identity / non-finite refuse (exit 3).

1.1.0 (2026-10-14, lane d1seal) -- a stage-D child composed from a REGISTERED DELTA (d1_compose_eval.py output):
  --delta-plan PLAN.json   the sealed `oc_checkpoint.py delta-compose` plan the composed --omni1 dir was built from
                           (only `replace` ops).  Step 3 then expects each targeted trunk tensor to differ from S5
                           ONLY inside the plan's declared regions, and those regions to hold exactly the delta's bytes
                           (read from the delta files the plan pins, sha256-checked); everything else stays
                           bit-identical.  Anything outside a region, or a region that is not the delta, refuses.
                           Region elements that differ from S5 are counted (diagnostic).
  --omni1-equivalence-receipt R   OMNI1's own s7 receipt.  When it shows OMNI1 == S5 on the R-Q text (top-1 on every
                           position, max |logit diff| 0.0) over the SAME prompts, the S5 column of this run is also the
                           OMNI1 column and is reported as rq_top1_child_vs_omni1 (with the receipt sha), so "R-Q vs
                           OMNI1" needs no fourth 55 GB stream.  Without it no OMNI1 column is claimed.
Without --delta-plan the check is 1.0.0's, unchanged (any trunk difference refuses).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_omni1_check"
TOOL_VERSION = "1.1.0"   # 1.0.0 (2026-10-13) S7 lane; 1.1.0 (2026-10-14, d1seal): --delta-plan, --omni1-equivalence-receipt
INDEX_FILENAME = "model.safetensors.index.json"
MANIFEST_FILENAME = "composite_manifest.json"


class Refusal(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_gib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


# ----------------------------------------------------------------------------- checkpoint access

class Checkpoint:
    """config + sha-verified lazy views of one Chimera checkpoint directory."""

    def __init__(self, directory: Path, label: str, *, verify: bool = True) -> None:
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.importing.safetensors_io import LazySafetensorsMapping, read_checkpoint_config

        self.directory = directory
        self.label = label
        self.document = read_checkpoint_config(directory)
        self.cfg = ChimeraConfig.from_manifest(self.document["config"])
        self.views = LazySafetensorsMapping(directory, verify=verify)
        self.tied = dict(self.views.tied_weights)
        self.pins = dict((self.views.index.get("metadata") or {}).get("tensor_sha256") or {})
        self.loaded: set[str] = set()

    def tensor(self, name: str, *, device: str = "cpu", dtype: Any = None) -> Any:
        owner = self.tied.get(name, name)
        self.loaded.add(name)
        return self.views[owner].torch(device=device, dtype=dtype)

    def meta(self, name: str) -> dict[str, Any]:
        return self.views.tensor_metadata(self.tied.get(name, name))


def torch_dtype(name: str) -> Any:
    from chimera_omni.importing.safetensors_io import _torch_dtype
    return _torch_dtype(name)


def load_module_strict(module: Any, ckpt: Checkpoint, prefix: str, *, device: str) -> Any:
    """``module.load_state_dict(strict=True)`` from the checkpoint's verified tensors under ``prefix``, each tensor
    loaded in the dtype its reference parameter has (so FP32 islands stay FP32)."""

    import torch

    state = {}
    for name, reference in module.state_dict().items():
        dtype = reference.dtype if reference.is_floating_point() else None
        state[name] = ckpt.tensor(prefix + name, device=device, dtype=dtype)
    module.load_state_dict(state, strict=True)
    return module


def build_block(ckpt: Checkpoint, index: int, *, device: str, dtype: Any) -> Any:
    """One ChimeraBlock of ``ckpt`` on ``device``: fp32 RoPE tables kept (as the tree's loader does), H3 FP32
    coefficients kept by ``ChartedSourceBlock._apply``; loaded strict from verified tensors."""

    import torch
    from chimera_omni.modeling.layers import ChimeraBlock

    block = ChimeraBlock(ckpt.cfg, index)
    rope = getattr(getattr(block, "mixer", None), "inv_freq", None)
    block = block.to(device=device, dtype=dtype)
    if rope is not None:
        block.mixer.inv_freq = rope.to(device=device, dtype=torch.float32)
    load_module_strict(block, ckpt, f"blocks.{index}.", device=device)
    return block.eval()


def build_top(ckpt: Checkpoint, *, device: str) -> dict[str, Any]:
    """Every top-level non-block module of ``ckpt`` built for real (constructors initialise their buffers) in the
    stored dtype and loaded strict.  Unknown parameter-carrying children refuse."""

    import torch
    from torch import nn
    from chimera_omni.modeling import ChimeraOmniModel
    from chimera_omni.modeling.chimera import NativeContinuousMediaBridge, UnifiedNativeMediaEmissionHead

    cfg = ckpt.cfg
    with torch.device("meta"):
        meta = ChimeraOmniModel(cfg)
    built: dict[str, Any] = {}
    h, v = int(cfg.hidden_size), int(cfg.vocab_size)
    for name, child in meta.named_children():
        if name == "blocks" or child is None:
            continue
        params = list(child.state_dict())
        if not params:
            continue
        stored = {ckpt.meta(f"{name}.{p}")["dtype"] for p in params}
        if name == "embed":
            module = nn.Embedding(v, h, dtype=torch_dtype(stored.pop()) if len(stored) == 1 else torch.float32)
        elif name == "lm_head":
            if cfg.tie_word_embeddings:
                continue
            module = nn.Linear(h, v, bias=False, dtype=torch_dtype(stored.pop()) if len(stored) == 1 else torch.float32)
        elif name == "norm":
            module = type(meta.norm)(h, cfg.rms_norm_eps)
        elif name == "native_media_bridge":
            module = NativeContinuousMediaBridge(cfg)
        elif name == "native_media_head":
            module = UnifiedNativeMediaEmissionHead(cfg)
        elif name in ("flow_head", "grounding_head", "sync_head"):
            module = nn.Linear(h, int(child.weight.shape[0]), bias=False)
        elif name == "mtp_heads":
            module = nn.ModuleList([nn.Linear(h, v, bias=False) for _ in range(len(child))])
        else:
            raise Refusal(f"{ckpt.label}: unknown top-level module {name!r} with parameters {params[:3]}")
        stored_all = {ckpt.meta(f"{name}.{p}")["dtype"] for p in params}
        if len(stored_all) == 1:
            module = module.to(dtype=torch_dtype(next(iter(stored_all))))
        module = module.to(device=device)
        load_module_strict(module, ckpt, f"{name}.", device=device)
        built[name] = module.eval()
    if cfg.tie_word_embeddings:
        built["lm_head_weight"] = built["embed"].weight
        ckpt.loaded.add("lm_head.weight")
    else:
        built["lm_head_weight"] = built["lm_head"].weight
    built["meta"] = meta
    return built


def structural(ckpt: Checkpoint) -> dict[str, Any]:
    """Meta-model state vs the checkpoint headers (what load_state_dict(strict=True) checks), plus dtypes."""

    import torch
    from chimera_omni.modeling import ChimeraOmniModel

    with torch.device("meta"):
        model = ChimeraOmniModel(ckpt.cfg)
    expected = {n: tuple(t.shape) for n, t in model.state_dict().items()}
    present = {n: tuple(ckpt.views.tensor_metadata(n)["shape"]) for n in ckpt.views}
    for alias, owner in ckpt.tied.items():
        present[alias] = present[owner]
    missing = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    mismatched = sorted(n for n in set(expected) & set(present) if expected[n] != present[n])
    dtypes: dict[str, int] = {}
    for n in ckpt.views:
        d = ckpt.views.tensor_metadata(n)["dtype"]
        dtypes[d] = dtypes.get(d, 0) + 1
    f32 = sorted(n for n in ckpt.views if ckpt.views.tensor_metadata(n)["dtype"] == "F32")
    pinned = sum(1 for n in ckpt.views if n in ckpt.pins)
    return {"tensors": len(present), "missing": missing, "unexpected": unexpected, "shape_mismatched": mismatched,
            "strict_equivalent": not (missing or unexpected or mismatched), "dtypes": dtypes, "f32_tensors": f32,
            "sha256_pinned_tensors": pinned, "tied": ckpt.tied,
            "block_kinds": [ckpt.cfg.block_kind(i) for i in range(int(ckpt.cfg.num_layers))]}


# ----------------------------------------------------------------------------- prompts and media segments

def rq_prompts(chimera_src: Path, tokenizer_dir: str | None, prompt_ids: str | None, max_tokens: int) -> tuple[list[list[int]], dict]:
    if prompt_ids:
        packs = [[int(t) for t in p][:max_tokens] for p in json.loads(Path(prompt_ids).read_text(encoding="utf-8"))]
        return packs, {"source": prompt_ids, "sha256": sha256_file(prompt_ids)}
    if not tokenizer_dir:
        raise Refusal("--tokenizer-dir or --prompt-ids is required")
    script = chimera_src / "scripts" / "import_receipt_qwen35.py"
    spec = importlib.util.spec_from_file_location("import_receipt_qwen35_omni1", script)
    rq = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rq)
    texts = tuple(rq.PROMPT_PACK)
    packs = rq.tokenize(Path(tokenizer_dir), texts, max_tokens)
    return packs, {"source": f"{script} PROMPT_PACK ({len(texts)} R-Q prompts)", "script_sha256": sha256_file(script),
                   "tokenizer_dir": tokenizer_dir, "prompts": [{"index": i, "tokens": len(p)} for i, p in enumerate(packs)]}


def media_segment(unit: str, tap_root: Path, bands: dict[int, tuple[int, int]], cfg: Any, *,
                  media_prompt_ids: list[int] | None = None) -> dict[str, Any]:
    """ONE held forward of the unit's pair set laid out as s7_host_rows does: [prompt | media rows]."""

    import torch
    from safetensors import safe_open

    tap_json_path = tap_root / unit / "paired_rows.json"
    tap_path = tap_root / unit / "paired_rows.safetensors"
    if not tap_json_path.is_file() or not tap_path.is_file():
        raise Refusal(f"{unit}: no host rows under {tap_root / unit}")
    tap_json = json.loads(tap_json_path.read_text(encoding="utf-8"))
    if tap_json.get("file_sha256") and sha256_file(tap_path) != tap_json["file_sha256"]:
        raise Refusal(f"{unit}: {tap_path} does not hash like its receipt")
    pair_path = Path(tap_json["pairs"]["file"])
    if sha256_file(pair_path) != tap_json["pairs"]["sha256"]:
        raise Refusal(f"{unit}: pair set {pair_path} does not hash like the host-row receipt records")
    pair_json = json.loads((pair_path.parent / "parent_capture.json").read_text(encoding="utf-8"))
    with safe_open(str(pair_path), "pt") as handle:
        pairs = {k: handle.get_tensor(k) for k in handle.keys()}
    with safe_open(str(tap_path), "pt") as handle:
        tap = {k: handle.get_tensor(k) for k in ("host_rows", "forward", "held")}
    held_forwards = tap["forward"][tap["held"].to(torch.bool)]
    if held_forwards.numel() == 0:
        raise Refusal(f"{unit}: no held rows")
    f0 = int(held_forwards[0])
    forward_of = pairs["forward"]
    rows = torch.nonzero(forward_of == f0).flatten()
    a, b = int(rows[0]), int(rows[-1]) + 1
    if b - a != int(rows.numel()):
        raise Refusal(f"{unit}: forward {f0} rows are not contiguous")
    entry = next((f for f in pair_json["forwards"] if int(f["index"]) == f0), None)
    if entry is None or int(entry["rows"]) != b - a:
        raise Refusal(f"{unit}: forward {f0} not declared with {b - a} rows")
    if media_prompt_ids is not None:
        prompt_ids = list(media_prompt_ids)
        tokenizer_record = "explicit --media-prompt-ids"
    else:
        from tokenizers import Tokenizer
        tok_path = tap_json["sequence_layout"]["tokenizer"]
        if sha256_file(tok_path) != tap_json["sequence_layout"]["tokenizer_sha256"]:
            raise Refusal(f"{unit}: tokenizer {tok_path} changed since the host-row capture")
        prompt_ids = Tokenizer.from_file(tok_path).encode(entry["prompt"], add_special_tokens=False).ids
        tokenizer_record = tok_path
    nm = b - a
    feature_dim = int(cfg.native_feature_dim)
    t_rows = pairs["type_ids"][a:b].to(torch.long)
    time_row = pairs["time_row"][a:b].to(torch.bool) if "time_row" in pairs else torch.zeros(nm, dtype=torch.bool)
    feats = torch.zeros(1, nm, feature_dim, dtype=torch.float32)
    for t in torch.unique(t_rows).tolist():
        if int(t) not in bands:
            raise Refusal(f"{unit}: type {t} has no installed interface band")
        lo, hi = bands[int(t)]
        m = (t_rows == t) & ~time_row
        widths = pairs["feature_width"][a:b][m]
        if bool((widths != hi - lo).any()):
            raise Refusal(f"{unit}: feature width != band width {hi - lo} for type {t}")
        feats[0, m, lo:hi] = pairs["features"][a:b][m][:, : hi - lo].to(torch.float32)
    sig = pairs["sigma"][a:b].to(torch.float32)
    if bool((sig < 0).any()) and not bool((sig < 0).all()):
        raise Refusal(f"{unit}: mixed sigma / no-sigma rows")
    placeholder = int(cfg.ledger().range("media").begin)
    plen = len(prompt_ids)
    host = tap["host_rows"][(tap["forward"] == f0)].to(torch.float32)
    if int(host.shape[0]) != nm:
        raise Refusal(f"{unit}: {host.shape[0]} host rows for forward {f0}, {nm} pair rows")
    return {
        "unit": unit, "forward": f0, "prompt": entry.get("prompt"), "prompt_tokens": plen, "media_rows": nm,
        "ids": torch.tensor([prompt_ids + [placeholder] * nm], dtype=torch.long),
        "features": feats, "types": t_rows.view(1, nm), "positions": pairs["positions"][a:b].to(torch.float32).view(1, nm, -1),
        "sigma": None if bool((sig < 0).all()) else sig.view(1, nm), "time_rows": int(time_row.sum()),
        "indices": torch.arange(plen, plen + nm, dtype=torch.long).view(1, nm), "host_rows_s5": host,
        "record": {"tap_json": str(tap_json_path), "tap_json_sha256": sha256_file(tap_json_path), "tap_sha256": sha256_file(tap_path),
                   "pairs": str(pair_path), "pairs_sha256": tap_json["pairs"]["sha256"], "tokenizer": tokenizer_record,
                   "type_ids": sorted({int(t) for t in t_rows.tolist()})},
    }


# ----------------------------------------------------------------------------- comparison helpers

def changed_ranges(mask_1d: Any) -> list[list[int]]:
    idx = mask_1d.nonzero().flatten().tolist()
    out: list[list[int]] = []
    for i in idx:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def merged(spans: list[list[int]]) -> list[list[int]]:
    """Sorted spans with adjacent / overlapping ones coalesced (declared bands are written side by side)."""
    out: list[list[int]] = []
    for a, b in sorted([int(x), int(y)] for x, y in spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def within(ranges: list[list[int]], allowed: list[list[int]]) -> bool:
    spans = merged(allowed)
    return all(any(lo >= a and hi <= b for a, b in spans) for lo, hi in ranges)


def declared_writes(manifest: dict[str, Any]) -> dict[str, dict[str, list[list[int]]]]:
    """name -> {'rows': [[lo, hi]...], 'cols': [...]} from the manifest's interface and band records."""

    out: dict[str, dict[str, list[list[int]]]] = {}

    def add(name: str, axis: str, span: list[int]) -> None:
        out.setdefault(name, {"rows": [], "cols": []})[axis].append([int(span[0]), int(span[1])])

    for entry in (manifest.get("interface") or {}).get("entries", []) or []:
        for tname, w in entry["writes"].items():
            if "columns" in w:
                add(tname, "cols", w["columns"])
            if "rows" in w:
                add(tname, "rows", w["rows"])
            if "row" in w:
                add(tname, "rows", [w["row"], w["row"] + 1])
    band = manifest.get("band") or {}
    for tname, w in (band.get("writes") or {}).items():
        add(tname, "rows", w["rows"])
    return out


# ----------------------------------------------------------------------------- 1.1.0: delta plan + OMNI1 equivalence

def load_delta_plan(path: str, ledger_root: str) -> dict[str, Any]:
    """{target: [(spans, delta file, delta tensor)]} from a sealed delta-compose plan (replace ops only)."""

    sys.path.insert(0, str(Path(ledger_root)))
    import oc_seal  # type: ignore
    from safetensors import safe_open

    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    oc_seal.verify_seal(plan)
    if plan.get("schema") != "neural-forge-omni-delta-plan-v1":
        raise Refusal(f"{path} is not a delta plan")
    targets: dict[str, list[tuple[Any, Path, str]]] = {}
    for link in plan["deltas"]:
        directory = Path(link["directory"])
        holder: dict[str, Path] = {}
        for f in link["files"]:
            p = directory / f["path"]
            if sha256_file(p) != f["sha256"]:
                raise Refusal(f"{p} does not hash to the plan's {f['sha256'][:16]}")
            with safe_open(str(p), "pt") as handle:
                for name in handle.keys():
                    holder[name] = p
        for entry in link["entries"]:
            if entry.get("op") != "replace":
                raise Refusal(f"delta op {entry.get('op')!r}: only replace regions are checked here")
            spans = {int(d): (int(a), int(b)) for d, a, b in (entry.get("slice") or [])}
            targets.setdefault(entry["target"], []).append((spans, holder[entry["tensor"]], entry["tensor"]))
    return {"targets": targets, "record": {"path": str(Path(path).resolve()), "file_sha256": sha256_file(path),
            "receipt_sha": plan["receipt_sha"], "checkpoint_id": plan["checkpoint_id"],
            "root": {k: plan["root"][k] for k in ("checkpoint_id", "directory_sha256")},
            "deltas": [d["checkpoint_id"] for d in plan["deltas"]], "replace_targets": len(targets)}}


def delta_identity(j: int, so: dict[str, Any], sr: dict[str, Any], regions: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Block j of the composed child vs its S5 block: equal outside the plan's regions, the delta's bytes inside."""

    import torch
    from safetensors import safe_open

    problems: list[str] = []
    changed, differing = 0, 0
    if set(so) != set(sr):
        return False, {"problems": [f"tensor sets differ: {sorted(set(so) ^ set(sr))[:4]}"]}
    for n in sorted(so):
        a, b = so[n], sr[n]
        spec = regions["targets"].get(f"blocks.{j}.{n}")
        if spec is None:
            if not torch.equal(a, b):
                problems.append(f"{n}: changed with no delta region")
            continue
        changed += 1
        mask = torch.zeros(a.shape, dtype=torch.bool, device=a.device)
        for spans, file, name in spec:
            index = tuple(slice(*spans.get(d, (0, a.shape[d]))) for d in range(a.ndim))
            mask[index] = True
            with safe_open(str(file), "pt") as handle:
                value = handle.get_tensor(name)
            if tuple(value.shape) != tuple(a[index].shape) or not torch.equal(a[index].cpu(), value.to(a.dtype)):
                problems.append(f"{n}: region {[spans.get(d) for d in range(a.ndim)]} is not the delta tensor {name}")
            differing += int((a[index] != b[index]).sum())
        if not torch.equal(a[~mask], b[~mask]):
            problems.append(f"{n}: changed outside its delta regions")
    return not problems, {"delta_tensors": changed, "region_elements_differing_from_s5": differing,
                          "bit_identical_outside_delta_regions": not any("outside" in p or "no delta" in p for p in problems),
                          "problems": problems}


def omni1_equivalence(path: str, prompts: dict[str, Any], positions: int) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    t = doc.get("text_forward") or {}
    theirs = doc.get("prompts") or {}
    same_prompts = all(theirs.get(k) == prompts.get(k) for k in ("source", "script_sha256", "tokenizer_dir", "prompts", "sha256"))
    holds = bool(same_prompts and t.get("positions") == positions and t.get("rq_top1_omni1_vs_s5") == positions
                 and t.get("max_abs_logit_diff_omni1_vs_s5_base_vocab") == 0.0 and not doc.get("refusals"))
    return {"receipt": str(Path(path).resolve()), "receipt_sha256": sha256_file(path), "same_prompts": same_prompts,
            "omni1_vs_s5_top1": t.get("rq_top1_omni1_vs_s5"), "omni1_vs_s5_max_abs": t.get("max_abs_logit_diff_omni1_vs_s5_base_vocab"),
            "holds": holds, "note": "OMNI1 == S5 on these R-Q positions exactly, so the S5 column is the OMNI1 column" if holds
            else "not shown equal: no OMNI1 column is claimed"}


# ----------------------------------------------------------------------------- the pass

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omni1", required=True)
    ap.add_argument("--s5", required=True, help="the lineage parent (S5 child) the trunk was copied from")
    ap.add_argument("--parent", default=None, help="the 27B parent S0 import (Chimera format); omitted -> no parent column")
    ap.add_argument("--tap-rows", required=True, help="S5 host rows root (<unit>/paired_rows.{json,safetensors})")
    ap.add_argument("--tokenizer-dir", default=None, help="HF dir for the R-Q prompts")
    ap.add_argument("--prompt-ids", default=None, help="JSON list of token-id lists (instead of the R-Q prompts)")
    ap.add_argument("--media-prompt-ids", default=None, help="JSON token-id list used as every media segment's prompt (tests)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC", ""))
    ap.add_argument("--ledger-root", default=os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))
    ap.add_argument("--parent-parameter-count", type=int, required=True)
    ap.add_argument("--insert-after", type=int, default=31, help="host-row site the charts were fitted at")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-plan", default=None, help="1.1.0: the sealed delta-compose plan --omni1 was composed from")
    ap.add_argument("--omni1-equivalence-receipt", default=None, help="1.1.0: OMNI1's s7 receipt (OMNI1 == S5 on R-Q text)")
    args = ap.parse_args(argv)
    started = time.time()
    import torch
    import torch.nn.functional as F
    import chimera_omni

    receipt: dict[str, Any] = {"schema": "omni-campaign.s7_omni1_check.v1", "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                               "sha256": sha256_file(Path(__file__).resolve())}, "argv": list(sys.argv[1:] if argv is None else argv),
                               "torch": torch.__version__, "device": args.device, "dtype": args.dtype,
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    refusals: list[str] = []
    src = Path(args.chimera_src).resolve() if args.chimera_src else None
    imported = Path(chimera_omni.__file__).resolve()
    receipt["chimera_omni"] = {"imported_from": str(imported), "chimera_src": str(src) if src else None,
                               "bound": bool(src and str(imported).startswith(str(src)))}
    if src and not receipt["chimera_omni"]["bound"]:
        raise Refusal(f"chimera_omni imported from {imported}, not from --chimera-src {src}")
    device, dtype = args.device, getattr(torch, args.dtype)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    omni = Checkpoint(Path(args.omni1), "omni1")
    s5 = Checkpoint(Path(args.s5), "s5")
    parent = Checkpoint(Path(args.parent), "parent") if args.parent else None
    manifest_path = Path(args.omni1) / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt["omni1"] = {"directory": args.omni1, "manifest_sha256": sha256_file(manifest_path),
                        "index_sha256": sha256_file(Path(args.omni1) / INDEX_FILENAME),
                        "config_sha256": sha256_file(Path(args.omni1) / "config.json"),
                        "chimera_commit": omni.document.get("chimera_commit")}
    receipt["s5"] = {"directory": args.s5, "index_sha256": sha256_file(Path(args.s5) / INDEX_FILENAME)}
    delta_regions = load_delta_plan(args.delta_plan, args.ledger_root) if args.delta_plan else None
    if delta_regions is not None:
        receipt["delta_plan"] = delta_regions["record"]
    if parent is not None:
        receipt["parent"] = {"directory": args.parent, "index_sha256": sha256_file(Path(args.parent) / INDEX_FILENAME),
                             "layers": int(parent.cfg.num_layers), "vocab": int(parent.cfg.vocab_size)}

    # 1. structure ---------------------------------------------------------------------------------------
    st = structural(omni)
    receipt["structural"] = st
    if not st["strict_equivalent"]:
        refusals.append(f"structural: missing {st['missing'][:3]} unexpected {st['unexpected'][:3]} shape {st['shape_mismatched'][:3]}")
    for ck in (omni, s5) + ((parent,) if parent is not None else ()):
        if str(ck.cfg.residual_mode) != "none":
            raise Refusal(f"{ck.label}: residual_mode {ck.cfg.residual_mode!r}; this layer-major pass is the residual_mode 'none' "
                          "forward (ChimeraOmniModel._forward_trunk's plain loop)")
    kinds = st["block_kinds"]
    charted_idx = [i for i, k in enumerate(kinds) if k == "charted"]
    trunk_idx = [i for i, k in enumerate(kinds) if k != "charted"]
    n_trunk = len(trunk_idx)
    if n_trunk != int(s5.cfg.num_layers):
        raise Refusal(f"OMNI1 has {n_trunk} trunk blocks, the S5 child {s5.cfg.num_layers}")
    layer_map = {int(k): int(v) for k, v in (manifest.get("blocks") or {}).get("trunk_layer_map", {}).items()}
    if layer_map and [layer_map[k] for k in range(n_trunk)] != trunk_idx:
        refusals.append("manifest trunk_layer_map disagrees with the config's block kinds")
    placements = {int(p["layer_index"]): p for p in (manifest.get("blocks") or {}).get("placements", [])}
    if sorted(placements) != charted_idx:
        refusals.append(f"charted blocks {charted_idx} != manifest placements {sorted(placements)}")
    receipt["layout"] = {"layers": len(kinds), "trunk_blocks": n_trunk, "charted_blocks": charted_idx,
                         "charted": {str(i): {"container": placements[i]["container"], "source": placements[i]["source"],
                                              "after_trunk_layer": trunk_idx.index(i - 1) if (i - 1) in trunk_idx else None,
                                              "type_ids": placements[i]["type_ids"]} for i in charted_idx if i in placements}}

    # 2. accounting ----------------------------------------------------------------------------------------
    sys.path.insert(0, str(Path(args.ledger_root)))
    import oc_seal  # type: ignore
    o_count, _ = oc_seal.safetensors_parameter_count(Path(args.omni1))
    s_count, _ = oc_seal.safetensors_parameter_count(Path(args.s5))
    added = int((manifest.get("accounting") or {}).get("charted_or_blend_added", -1))
    receipt["accounting"] = {"omni1_header_parameters": int(o_count), "s5_header_parameters": int(s_count),
                             "charted_added_manifest": added, "omni1_equals_s5_plus_charted": int(o_count) == int(s_count) + added,
                             "accounting_parent_parameters": int(args.parent_parameter_count),
                             "headroom": int(args.parent_parameter_count) - int(o_count),
                             "below_parent": int(o_count) < int(args.parent_parameter_count),
                             "source": "oc_seal.safetensors_parameter_count (the ledger's header reader)"}
    if not receipt["accounting"]["below_parent"]:
        refusals.append("accounting: OMNI1 is not below the accounting parent")
    if not receipt["accounting"]["omni1_equals_s5_plus_charted"]:
        refusals.append("accounting: OMNI1 != S5 + charted state")

    # top-level modules (strict) and non-block identity vs S5 ------------------------------------------------
    log("building OMNI1 / S5 top-level modules (strict, verified)")
    top_o = build_top(omni, device="cpu")
    top_s = build_top(s5, device="cpu")
    declared = declared_writes(manifest)
    nonblock: dict[str, Any] = {}
    sdo, sds = {}, {}
    for name, module in top_o.items():
        if name in ("meta", "lm_head_weight"):
            continue
        for p, t in module.state_dict().items():
            sdo[f"{name}.{p}"] = t
    for name, module in top_s.items():
        if name in ("meta", "lm_head_weight"):
            continue
        for p, t in module.state_dict().items():
            sds[f"{name}.{p}"] = t
    if set(sdo) != set(sds):
        refusals.append(f"non-block tensor sets differ: {sorted(set(sdo) ^ set(sds))[:5]}")
    for name in sorted(set(sdo) & set(sds)):
        a, b = sdo[name], sds[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            refusals.append(f"{name}: shape/dtype changed {tuple(b.shape)}/{b.dtype} -> {tuple(a.shape)}/{a.dtype}")
            continue
        if torch.equal(a, b):
            nonblock[name] = {"identical": True}
            continue
        diff = (a != b)
        rec: dict[str, Any] = {"identical": False, "finite": bool(torch.isfinite(a.float()).all())}
        if diff.ndim == 2:
            rec["rows_changed"] = changed_ranges(diff.any(dim=1))
            rec["cols_changed"] = changed_ranges(diff.any(dim=0))
        else:
            rec["elements_changed"] = changed_ranges(diff.flatten())
        allowed = declared.get(name)
        ok = False
        if allowed is not None and diff.ndim == 2:
            ok = ((allowed["rows"] and within(rec["rows_changed"], allowed["rows"])) or
                  (allowed["cols"] and within(rec["cols_changed"], allowed["cols"])))
        rec["declared"] = allowed
        rec["within_declared_writes"] = bool(ok)
        if not ok:
            refusals.append(f"{name}: changed outside the manifest's declared writes")
        if not rec["finite"]:
            refusals.append(f"{name}: non-finite after the install")
        nonblock[name] = rec
    receipt["identity_nonblock_vs_s5"] = {"tensors": len(nonblock), "identical": sum(1 for r in nonblock.values() if r["identical"]),
                                          "changed": {k: v for k, v in nonblock.items() if not v["identical"]}}
    del sds

    # prompts, media segments --------------------------------------------------------------------------------
    packs, prompt_record = rq_prompts(src or Path("."), args.tokenizer_dir, args.prompt_ids, args.max_tokens)
    receipt["prompts"] = prompt_record
    bands = {int(e["type_id"]): (int(e["feature_band"][0]), int(e["feature_band"][1]))
             for e in (manifest.get("interface") or {}).get("entries", []) or []}
    media_prompt = json.loads(Path(args.media_prompt_ids).read_text()) if args.media_prompt_ids else None
    segments = [media_segment(placements[i]["container"], Path(args.tap_rows), bands, omni.cfg, media_prompt_ids=media_prompt)
                for i in charted_idx]
    meta = top_o["meta"]
    bridge = top_o["native_media_bridge"].to(device=device, dtype=dtype)
    for name in ("native_media_bridge",):
        setattr(meta, name, bridge)

    with torch.no_grad():
        embed_o, embed_s = top_o["embed"], top_s["embed"]
        xs = {"omni1": [], "s5": []}
        valids = []
        for ids in packs:
            t = torch.tensor([ids], dtype=torch.long)
            xs["omni1"].append(embed_o(t).to(device=device, dtype=dtype))
            xs["s5"].append(embed_s(t).to(device=device, dtype=dtype))
            valids.append(torch.ones(t.shape, dtype=torch.bool, device=device))
        if parent is not None:
            p_embed = parent.tensor("embed.weight", dtype=dtype)
            xs["parent"] = [F.embedding(torch.tensor([ids], dtype=torch.long), p_embed).to(device) for ids in packs]
            del p_embed
        seg_state = []
        for seg in segments:
            ids = seg["ids"]
            x = embed_o(ids).to(device=device, dtype=dtype)
            types = seg["types"].to(device)
            valid_media = torch.ones(types.shape, dtype=torch.bool, device=device)
            noise = None if seg["sigma"] is None else seg["sigma"].to(device)
            x, count = type(meta)._add_native_media(meta, x, seg["features"].to(device), types, seg["positions"].to(device),
                                                    seg["indices"].to(device), valid_media, noise)
            if count != seg["media_rows"]:
                raise Refusal(f"{seg['unit']}: media count {count} != {seg['media_rows']}")
            row_types, levels = type(meta)._charted_layout(meta, ids.to(device), count, seg["indices"].to(device), types,
                                                            valid_media, noise)
            valid = torch.ones(ids.shape, dtype=torch.bool, device=device)
            positions = {spec.source: torch.zeros(1, ids.shape[1], 3, device=device)
                         for spec in omni.cfg.charted_blocks if spec.modulation == "h3_curve6"}
            seg_state.append({"x": torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)), "valid": valid, "row_types": row_types,
                              "levels": levels, "positions": positions or None, "media_x_finite": bool(torch.isfinite(x).all())})
        del embed_s
        top_s.pop("embed", None)
        log(f"inputs ready: {len(packs)} prompts, {len(segments)} media segments ({time.time() - started:.0f}s)")

        def run_trunk(block: Any, x: Any, valid: Any) -> Any:
            y = block(x, native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False, token_valid_mask=valid)
            return torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))

        per_layer: list[dict[str, Any]] = []
        trunk_identity: list[dict[str, Any]] = []
        charted_reports: dict[str, Any] = {}
        x31: list[Any] = []
        for k, j in enumerate(trunk_idx):
            blk = build_block(omni, j, device=device, dtype=dtype)
            ref = build_block(s5, k, device=device, dtype=dtype)
            so, sr = blk.state_dict(), ref.state_dict()
            same = set(so) == set(sr) and all(torch.equal(so[n], sr[n]) for n in so)
            row_id: dict[str, Any] = {"s5_block": k, "omni1_block": j, "bit_identical": bool(same)}
            if not same and delta_regions is not None:
                ok, detail = delta_identity(j, so, sr, delta_regions)
                row_id.update(detail)
                if not ok:
                    refusals.append(f"OMNI1 block {j} differs from S5 block {k} outside the delta plan's regions, or a "
                                    f"region is not the delta's bytes: {detail.get('problems', [])[:3]}")
            elif not same:
                refusals.append(f"OMNI1 block {j} is not bit-identical to S5 block {k}")
            trunk_identity.append(row_id)
            xs["omni1"] = [run_trunk(blk, x, v) for x, v in zip(xs["omni1"], valids)]
            for s in seg_state:
                s["x"] = run_trunk(blk, s["x"], s["valid"])
            del blk, so
            xs["s5"] = [run_trunk(ref, x, v) for x, v in zip(xs["s5"], valids)]
            del ref, sr
            if parent is not None:
                pb = build_block(parent, k, device=device, dtype=dtype)
                xs["parent"] = [run_trunk(pb, x, v) for x, v in zip(xs["parent"], valids)]
                del pb
            row = {"layer": k, "omni1_block": j,
                   "prompt_residual_finite": all(bool(torch.isfinite(x).all()) for x in xs["omni1"]),
                   "segment_residual_finite": all(bool(torch.isfinite(s["x"]).all()) for s in seg_state),
                   "omni1_vs_s5_prompt_max_abs": max(float((a.float() - b.float()).abs().max()) for a, b in zip(xs["omni1"], xs["s5"]))}
            per_layer.append(row)
            if k == args.insert_after:
                x31 = [s["x"].float().clone() for s in seg_state]
            # the charted blocks that follow this trunk block
            while (j + 1) in charted_idx:
                j += 1
                cb = build_block(omni, j, device=device, dtype=dtype)
                p = placements[j]
                rep: dict[str, Any] = {"container": p["container"], "source": p["source"], "type_ids": p["type_ids"],
                                       "after_trunk_layer": k, "dtype_path": str(dtype).replace("torch.", ""), "text": {}, "segments": {}}
                text_zero = True
                for i, (x, v) in enumerate(zip(xs["omni1"], valids)):
                    y = run_trunk(cb, x, v)
                    text_zero = text_zero and bool(torch.equal(y, x))
                    xs["omni1"][i] = y
                rep["text"] = {"identity_on_text": text_zero}
                if not text_zero:
                    refusals.append(f"charted block {j} ({p['container']}) changes text rows")
                for si, (seg, s) in enumerate(zip(segments, seg_state)):
                    def write_of(block: Any, x: Any) -> Any:
                        return block.charted_output(x, s["row_types"], s["levels"], s["valid"], s["positions"])
                    try:
                        w = write_of(cb, s["x"])
                        path = rep["dtype_path"]
                    except (RuntimeError, ValueError) as error:  # dtype contract of the bf16 runtime, recorded
                        rep.setdefault("bf16_errors", {})[seg["unit"]] = f"{type(error).__name__}: {error}"
                        cb32 = build_block(omni, j, device=device, dtype=torch.float32)
                        w = write_of(cb32, s["x"].float()).to(s["x"].dtype)
                        path = "float32 (bf16 path raised)"
                        del cb32
                    owned = cb.charted.owned_rows(s["row_types"], s["valid"])
                    n_owned = 0 if owned is None else int(owned.sum())
                    wf = w.float()
                    off = wf if owned is None else wf.masked_fill(owned.unsqueeze(-1), 0.0)
                    own_norm = 0.0 if owned is None else float(wf[owned].norm())
                    xin = s["x"].float()
                    rec = {"rows_owned": n_owned, "write_finite": bool(torch.isfinite(wf).all()),
                           "write_norm_owned": own_norm, "zeros_off_owned": bool((off == 0).all()), "path": path}
                    if owned is not None and n_owned:
                        ratio = wf[owned].norm(dim=-1) / xin[owned].norm(dim=-1).clamp_min(1e-30)
                        rec["write_over_input_norm_median"] = float(ratio.median())
                        rec["rows_written_nonzero"] = int((wf[owned].abs().sum(dim=-1) > 0).sum())
                    own_segment = seg["unit"] == p["container"]
                    rec["own_segment"] = own_segment
                    if not rec["write_finite"]:
                        refusals.append(f"charted block {j} ({p['container']}): non-finite write on segment {seg['unit']}")
                    if not rec["zeros_off_owned"]:
                        refusals.append(f"charted block {j} ({p['container']}): writes off its own rows on segment {seg['unit']}")
                    if own_segment and (n_owned == 0 or own_norm == 0.0):
                        refusals.append(f"charted block {j} ({p['container']}): owns / writes nothing on its own segment")
                    if not own_segment and own_norm != 0.0:
                        refusals.append(f"charted block {j} ({p['container']}): writes on the {seg['unit']} segment")
                    rep["segments"][seg["unit"]] = rec
                    s["x"] = torch.where(s["valid"].unsqueeze(-1), s["x"] + w, torch.zeros_like(s["x"]))
                charted_reports[str(j)] = rep
                del cb
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if k % 8 == 0 or k == n_trunk - 1:
                log(f"trunk layer {k}/{n_trunk - 1} (OMNI1 block {j}) done ({time.time() - started:.0f}s, rss {rss_gib():.1f} GiB"
                    + (f", cuda peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)" if device.startswith("cuda") else ")"))
        receipt["identity_trunk_vs_s5"] = {"blocks": len(trunk_identity), "bit_identical": sum(r["bit_identical"] for r in trunk_identity),
                                           "per_block": trunk_identity}
        receipt["per_layer"] = per_layer
        receipt["charted"] = charted_reports

        # heads ---------------------------------------------------------------------------------------------
        band = int(omni.cfg.ledger_band_begin())
        if parent is not None:
            band = min(band, int(parent.cfg.ledger_band_begin()))
        norm_o = top_o["norm"].to(device=device, dtype=dtype)
        head_o = top_o["lm_head_weight"].to(device=device, dtype=dtype)
        logits = {"omni1": [F.linear(norm_o(x), head_o).float().cpu()[0] for x in xs["omni1"]]}
        seg_logits_finite, seg_media_pred = [], []
        head_media = top_o["native_media_head"].to(device=device, dtype=dtype)
        for seg, s in zip(segments, seg_state):
            hidden = norm_o(s["x"])
            lg = F.linear(hidden, head_o)
            seg_logits_finite.append(bool(torch.isfinite(lg).all()))
            idx = seg["indices"].to(device)
            mh = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            pred = head_media(mh, seg["types"].to(device)).float()
            seg_media_pred.append({"finite": bool(torch.isfinite(pred).all()), "norm": float(pred.norm()),
                                   "nonzero": bool((pred != 0).any())})
            del lg
        del head_o
        top_o.pop("lm_head", None)
        top_o.pop("lm_head_weight", None)
        norm_s = top_s["norm"].to(device=device, dtype=dtype)
        head_s = top_s["lm_head_weight"].to(device=device, dtype=dtype)
        logits["s5"] = [F.linear(norm_s(x), head_s).float().cpu()[0] for x in xs["s5"]]
        del head_s
        if parent is not None:
            from chimera_omni.modeling import ChimeraOmniModel
            with torch.device("meta"):
                pmeta = ChimeraOmniModel(parent.cfg)
            pnorm = type(pmeta.norm)(int(parent.cfg.hidden_size), parent.cfg.rms_norm_eps).to(device=device, dtype=dtype)
            pnorm.weight.copy_(parent.tensor("norm.weight", device=device, dtype=dtype))
            phead = parent.tensor("lm_head.weight" if "lm_head.weight" in parent.views or "lm_head.weight" in parent.tied else "embed.weight",
                                  device=device, dtype=dtype)
            logits["parent"] = [F.linear(pnorm(x), phead).float().cpu()[0] for x in xs["parent"]]
            del phead

    # metrics ---------------------------------------------------------------------------------------------
    def agree(a: list, b: list) -> tuple[int, int]:
        n = m = 0
        for la, lb in zip(a, b):
            ta, tb = la[:, :band].argmax(-1), lb[:, :band].argmax(-1)
            n += int((ta == tb).sum())
            m += int(ta.shape[0])
        return n, m

    finite = {k: all(bool(torch.isfinite(l).all()) for l in v) for k, v in logits.items()}
    o_s, total = agree(logits["omni1"], logits["s5"])
    text = {"positions": total, "band_begin": band, "logits_finite": finite,
            "rq_top1_omni1_vs_s5": o_s, "rq_top1_omni1_vs_s5_fraction": o_s / max(total, 1),
            "max_abs_logit_diff_omni1_vs_s5_base_vocab": max(float((a[:, :band] - b[:, :band]).abs().max()) for a, b in zip(logits["omni1"], logits["s5"])),
            "omni1_full_vocab_argmax_in_band_positions": sum(int((l.argmax(-1) >= band).sum()) for l in logits["omni1"]),
            "omni1_max_band_logit": max(float(l[:, band:].max()) for l in logits["omni1"]) if logits["omni1"][0].shape[1] > band else None}
    if parent is not None:
        o_p, _ = agree(logits["omni1"], logits["parent"])
        s_p, _ = agree(logits["s5"], logits["parent"])
        text.update({"rq_top1_omni1_vs_parent": o_p, "rq_top1_omni1_vs_parent_fraction": o_p / max(total, 1),
                     "rq_top1_s5_vs_parent": s_p, "rq_top1_s5_vs_parent_fraction": s_p / max(total, 1)})
    text["note"] = "diagnostics only (owner policy 2026-10-06); top-1 over the base vocab [0, band_begin), every prompt position"
    if args.omni1_equivalence_receipt:
        eq = omni1_equivalence(args.omni1_equivalence_receipt, receipt.get("prompts") or {}, total)
        text["omni1_equivalence"] = eq
        if eq["holds"]:
            text["rq_top1_child_vs_omni1"] = o_s
            text["rq_top1_child_vs_omni1_fraction"] = o_s / max(total, 1)
    receipt["text_forward"] = text
    for name, ok in finite.items():
        if not ok:
            refusals.append(f"non-finite {name} logits")

    media = {}
    for si, (seg, s) in enumerate(zip(segments, seg_state)):
        rec = {"forward": seg["forward"], "prompt_tokens": seg["prompt_tokens"], "media_rows": seg["media_rows"],
               "time_rows": seg["time_rows"], "media_input_finite": s["media_x_finite"], "final_logits_finite": seg_logits_finite[si],
               "native_media_head": seg_media_pred[si], "inputs": seg["record"]}
        if x31:
            xm = x31[si][0, seg["prompt_tokens"]:].cpu()
            host = seg["host_rows_s5"]
            d = xm - host
            rec["x31_vs_s5_host_rows"] = {"rel_l2": float(d.norm() / host.norm().clamp_min(1e-30)), "max_abs": float(d.abs().max()),
                                          "row_cos_median": float(F.cosine_similarity(xm, host, dim=-1).median()),
                                          "note": "diagnostic: same layout and interface rows as the capture, single forward here vs batched there"}
        media[seg["unit"]] = rec
        if not seg_logits_finite[si] or not seg_media_pred[si]["finite"] or not s["media_x_finite"]:
            refusals.append(f"{seg['unit']}: non-finite media forward")
    receipt["media_forward"] = media
    loaded = omni.loaded
    expected_names = set(omni.views) | set(omni.tied)
    receipt["strict_load_coverage"] = {"tensors_loaded": len(loaded & expected_names), "tensors_expected": len(expected_names),
                                       "not_loaded": sorted(expected_names - loaded)[:10], "complete": loaded >= expected_names,
                                       "sha256_verified_on_read": omni.views.verify, "read_stats": dict(omni.views.read_stats)}
    if not receipt["strict_load_coverage"]["complete"]:
        refusals.append(f"strict load did not cover {len(expected_names - loaded)} tensor(s)")
    receipt["refusals"] = refusals
    if delta_regions is not None and not refusals:
        receipt["identity_trunk_vs_s5"]["delta_plan_regions"] = "every difference from S5 lies inside the plan's replace regions, which hold the delta's bytes"
    receipt["verdict"] = ("OMNI1 loads strict; identity vs S5 and accounting hold; text and every charted block's media forward finite; "
                          "every charted block writes on its own segment and only there" if not refusals else "REFUSED")
    receipt["seconds"] = round(time.time() - started, 1)
    receipt["peak_rss_gib"] = rss_gib()
    if device.startswith("cuda"):
        receipt["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out = Path(args.out)
    out.write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    Path(str(out) + ".sha256").write_text(sha256_file(out) + "\n", encoding="utf-8")
    print(receipt["verdict"])
    for r in refusals:
        print("REFUSAL", r)
    print("accounting", json.dumps(receipt["accounting"]))
    print("structural", json.dumps({k: st[k] for k in ("tensors", "strict_equivalent", "dtypes", "sha256_pinned_tensors")}))
    print("identity", receipt["identity_trunk_vs_s5"]["bit_identical"], "/", receipt["identity_trunk_vs_s5"]["blocks"], "trunk blocks bit-identical;",
          "non-block changed:", {k: (v.get("rows_changed"), v.get("cols_changed"), v["within_declared_writes"]) for k, v in receipt["identity_nonblock_vs_s5"]["changed"].items()})
    print("text", json.dumps({k: v for k, v in text.items() if k not in ("note", "omni1_equivalence")}))
    if delta_regions is not None:
        print("delta", json.dumps({"plan": receipt["delta_plan"]["receipt_sha"][:16], "targets": receipt["delta_plan"]["replace_targets"],
                                   "blocks_changed": sum(1 for r in trunk_identity if not r["bit_identical"]),
                                   "region_elements_differing_from_s5": sum(r.get("region_elements_differing_from_s5", 0) for r in trunk_identity)}))
    for j, rep in charted_reports.items():
        own = rep["segments"].get(rep["container"], {})
        others = {u: r["write_norm_owned"] for u, r in rep["segments"].items() if u != rep["container"]}
        print("charted", j, rep["container"], "text_identity", rep["text"]["identity_on_text"], "own", json.dumps(own), "others", others,
              "bf16_errors", rep.get("bf16_errors"))
    for u, rec in media.items():
        print("media", u, json.dumps({k: rec[k] for k in ("forward", "media_rows", "time_rows", "final_logits_finite", "native_media_head")}),
              "x31", json.dumps(rec.get("x31_vs_s5_host_rows")))
    print("coverage", json.dumps({k: receipt["strict_load_coverage"][k] for k in ("tensors_loaded", "tensors_expected", "complete")}))
    print("out", out, sha256_file(out))
    return 0 if not refusals else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refusal as error:
        print(f"REFUSED: {error}", flush=True)
        sys.exit(3)
