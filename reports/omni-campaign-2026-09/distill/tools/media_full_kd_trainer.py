#!/usr/bin/env python3
"""Full-parameter MEDIA distillation of Chimera Omni, stage M1: every weight of the media modules.

WHY THIS FILE EXISTS.  Cyril, 2026-10-17: full distillation, no LoRA, from the original parents "[Cyril quote withheld]"; 2026-10-18: do the media distillation in this session.  `d3_hybrid_module.py` (the D3 engine of record)
holds the hybrid model (OMNI1's qwen3_5 text layers + the Chimera charted units + the native media bridge + the flow head),
the objectives and the native-batch binders, but no run driver, and its trainable sets are LoRA r16 + slice deltas.
This driver trains the media modules FULL-parameter on the teachers' native flow batches.

WHAT TRAINS (stage M1).  Every parameter of every charted unit (ACE DiT L12, QIMG MMDiT L30, HY3D DiT L10, H3 DiT L25,
QVL32 ViT L13; the source-frozen H3 curve table and RoPE frequencies excepted, as in D3) and every parameter of the native
media emission head, in fp32 masters, with the full trainer's streamed Adafactor (factored second moments, RMS clip).
The text trunk is FROZEN here at OMNI1D1 (the clone + the D1 plan, exactly what full-KD run 1 starts from): run 1 owns
the trunk on GPU4 right now, and two diverging copies of one tensor are never trained.  The media tensors are disjoint
from run 1's 851 text tensors, so the M1 result joins run 1's exported student exactly (stage M2 then trains EVERY
weight, trunk included, on a text + media mixture from that merged student).

THE STEP (D3b, guided-velocity KD, 2026-10-14 prereg lane loss, x0 head).  A group of K native flow batches (one sample
each, any routes) runs LAYER-MAJOR: each trunk layer is loaded once and applied to every sample of the group, so the
27B trunk streams once per group, not once per sample:
  embedding + bridge (no grad) -> layers 0..d-1 (no grad) -> x_d ->
  units (no grad; recomputed with grad at the end) -> layers d..L-1 (no grad, host boundaries per sample) ->
  final norm + flow head (graph) -> loss per sample (bind_native_flow_batch: the sealed receipt's targets, bands, signs)
  -> layers L-1..d walked back (weights reloaded, activations' gradients only) -> units recomputed with grad, backward.
The group loss is the mean of the sample losses.  `selftest` proves the gradients equal d3_hybrid_module's own per-sample
`d3b_forward_backward` on a tiny Chimera child built by the real importer.

DATA.  Codex's sealed `chimera_recovery.native_flow_cache.v1` batches (compact `native_band.v1` storage), read through
`inflate_native_batch` + `bind_native_flow_batch` (every Codex ReadyObjectiveAdapter guard) + `native_batch_segment`:
  music_generate  native-ace-full-v5    ACE-Step 1.5 guided velocity, authentic code conditioning (2,764 batches)
  image_generate  native-qimg-full-v3   Qwen-Image guided velocity (1,126 batches)
  joint_av        native-h3-v1          H3 joint audio+video (32 parity batches; the production cache is still owed)
The tensor file of every batch is checked against its receipt's `cache_sha256` before use.  A per-route held set (the
last `--held-per-route` items of a seeded order) is never trained on and gives the eval loss.

Commands
--------
  selftest  tiny Chimera child (CPU fp32): layer-major group gradients == per-sample d3b_forward_backward; M1 loss falls
  probe     real OMNI1D1 weights, lr 0: N groups, timing, peak VRAM, host RSS, every trainable gets a finite gradient
  train     the job (resumable; rolling checkpoints; export of the media modules in Chimera names)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

TOOL_NAME = "media_full_kd_trainer"
TOOL_VERSION = "1.3.3"
RECEIPT_SCHEMA = "neural-forge-omni-media-full-kd-v1"
CHECKPOINT_SCHEMA = "neural-forge-omni-media-full-kd-checkpoint-v1"
EXPORT_SCHEMA = "neural-forge-omni-media-modules-v1"
HERE = Path(__file__).resolve().parent
FLOW_SCHEMA = "chimera_recovery.native_flow_cache.v1"
#: source-frozen unit coefficients (d3_hybrid_module.unit_parameters keeps them frozen too)
FROZEN_UNIT_PARAMS = ("h3_time.curve_table", "attn.rope_inv_freq")


def _load(name: str, path: Path) -> Any:
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d3 = _load("d3_hybrid_module", HERE / "d3_hybrid_module.py")
fk = _load("streamed_full_kd_trainer", HERE / "streamed_full_kd_trainer.py")


class MediaError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tool_identity() -> dict[str, Any]:
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()), "d3": d3.peer_record(d3), "full_kd": d3.peer_record(fk)}


def write_json_atomic(path: Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return file_sha256(path)


# --------------------------------------------------------------------------
# The trainable media modules of a HybridD3.
# --------------------------------------------------------------------------


def media_trainables(hybrid: Any) -> dict[str, torch.Tensor]:
    """Chimera name -> live fp32 parameter: every unit parameter (bar the source-frozen coefficients) and every
    parameter of the flow head (cast to fp32 in place; the head then runs on fp32 rows, see `emission`)."""

    out: dict[str, torch.Tensor] = {}
    for placement in hybrid.layout:
        unit = hybrid.units[placement.container]
        for name, p in unit.named_parameters():
            if name in FROZEN_UNIT_PARAMS:
                p.requires_grad_(False)
                continue
            if p.dtype != torch.float32:
                raise MediaError(f"unit {placement.container}.{name} is {p.dtype}; M1 keeps fp32 unit masters")
            p.requires_grad_(True)
            out[f"blocks.{placement.block}.charted.{name}"] = p
    hybrid.media_head.float()
    for name, p in hybrid.media_head.named_parameters():
        p.requires_grad_(True)
        out[f"native_media_head.{name}"] = p
    if hybrid.head_deltas or hybrid.bridge_deltas:
        raise MediaError("M1 trains the head itself, never slice deltas beside it")
    #: d3 losses that call `model.emission` (speech_out_flow_loss) get the same fp32 head path
    hybrid.emission = lambda normed, context: emission(hybrid, normed, context)
    return out


class MusicBand:
    """D3c, full-parameter: the C18 music band's WRITTEN code rows of the head (dense, every step) and of the embedding
    (sparse: the code rows a group's inputs used), as fp32 masters.  The head rows live on the device; the embedding
    rows are the whole band table on the host (HybridD3.embed reads it through `band_embed_override`).  Ledger rows
    below the band and the reserved rows past the written codes are digested at construction and re-checked after every
    update (d3's BandTrainer rule, Codex's D3c finding)."""

    def __init__(self, hybrid: Any, *, codes: int) -> None:
        self.hybrid = hybrid
        self.codes = int(codes)
        self.offset = hybrid.music_offset
        self.contract = hybrid.music_contract(self.codes)
        self.embed = hybrid.band_embed.float().clone()
        hybrid.band_embed_override = self.embed
        self.head = hybrid.band_head[self.offset:self.offset + self.codes].float().to(hybrid.device).clone().requires_grad_(True)
        self.row_grads: dict[int, torch.Tensor] = {}
        self.frozen = self._digests()

    def _digests(self) -> dict[str, str]:
        o, c, r = self.offset, self.codes, self.hybrid.music_rows
        return {"ledger": d3.tensor_sha256(self.embed[:o]), "reserved": d3.tensor_sha256(self.embed[o + c:o + r])}

    def check_frozen(self) -> None:
        if self._digests() != self.frozen:
            raise MediaError("D3c moved a ledger or reserved band row of the embedding")

    def head_loss(self, positions: torch.Tensor, topk_codes: torch.Tensor, topk_lp_band: torch.Tensor) -> Callable:
        def loss_fn(normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
            rows = normed[0, positions.to(normed.device)]
            return d3.band_kl(rows, self.head, topk_codes.to(normed.device), topk_lp_band.to(normed.device))
        return loss_fn

    def input_grad(self, ids: torch.Tensor) -> Callable:
        hybrid = self.hybrid

        def collect(grad: torch.Tensor) -> None:
            band = ids[0] >= hybrid.base_vocab
            if not bool(band.any()):
                return
            rows = (ids[0][band] - hybrid.base_vocab).cpu()
            g = grad[0][band.to(grad.device)].float()
            uniq, inverse = torch.unique(rows, return_inverse=True)
            summed = torch.zeros(uniq.numel(), g.shape[-1], dtype=torch.float32, device=g.device)
            summed.index_add_(0, inverse.to(g.device), g)
            for r, v in zip(uniq.tolist(), summed):
                held = self.row_grads.get(r)
                self.row_grads[r] = v if held is None else held + v
        return collect

    def apply(self, optimizer: Any, lr: float, step: int, stats: dict[str, Any]) -> None:
        if self.head.grad is not None:
            g = self.head.grad.float()
            norm = float(torch.linalg.vector_norm(g))
            if math.isfinite(norm):
                stats["grad_sq"] += norm * norm
                if lr > 0:
                    optimizer.update("music.head", self.head.data, g, lr, step=step)
                stats["updated"] += 1
            else:
                stats["nonfinite"].append("music.head")
            self.head.grad = None
        if self.row_grads:
            rows = torch.tensor(sorted(self.row_grads), dtype=torch.long)
            lo, hi = self.offset, self.offset + self.codes
            if bool(((rows < lo) | (rows >= hi)).any()):
                raise MediaError("D3c: an input band row outside the written codes got a gradient")
            g = torch.stack([self.row_grads[int(r)] for r in rows])
            norm = float(torch.linalg.vector_norm(g))
            if math.isfinite(norm):
                stats["grad_sq"] += norm * norm
                if lr > 0:
                    w = self.embed.index_select(0, rows).to(g.device)
                    optimizer.update_rows("music.embed", w, g, rows - lo, self.codes, lr, step=step)
                    self.embed.index_copy_(0, rows, w.to("cpu"))
                stats["updated"] += 1
            else:
                stats["nonfinite"].append("music.embed")
            self.row_grads = {}
        self.check_frozen()

    def tensors(self) -> dict[str, torch.Tensor]:
        o, c = self.offset, self.codes
        return {"music.head": self.head.detach(), "music.embed": self.embed[o:o + c]}

    def load(self, values: dict[str, torch.Tensor]) -> None:
        o, c = self.offset, self.codes
        with torch.no_grad():
            self.head.copy_(values["music.head"].to(self.head.device))
            self.embed[o:o + c] = values["music.embed"].float()
        self.check_frozen()


def emission(hybrid: Any, normed: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
    """The fp32 flow head on the media rows (the trunk may run in bf16; the head and the loss stay fp32)."""

    rows = hybrid.media_hidden(normed, context).float()
    return hybrid._head_call(rows, context["types"])


# --------------------------------------------------------------------------
# The layer-major group step.
# --------------------------------------------------------------------------


def scope_of(sample: dict[str, Any]) -> Any:
    import contextlib

    factory = sample.get("scope")
    return factory() if factory is not None else contextlib.nullcontext()


def group_forward_backward(hybrid: Any, samples: list[dict[str, Any]], *, depth: int, offload: str = "cpu",
                           loss_scale: float = 1.0) -> dict[str, Any]:
    """One group: every sample `{"segment", "weight", "loss_fn" | "head_loss", "scope"?}`; gradients accumulate into the
    media trainables' `.grad`.  `loss_fn(prediction)` is a flow loss on the fp32 head output; `head_loss(normed, context)`
    any loss on the final normed hidden (speech-in CE on the text head, speech-out flow); `scope()` a context manager
    around every unit application of that sample (D3e's H3 prompt-only read scope).  Returns per-sample losses.  Units after layers other than depth-1 are refused (OMNI1: all five sit after
    text layer 31); the trunk weights never receive a gradient."""

    L = int(hybrid.num_layers)
    placed = sorted({p.after_text_layer for p in hybrid.layout})
    if placed != [depth - 1]:
        raise MediaError(f"units sit after text layers {placed}; the group step supports units after layer {depth - 1} only")
    engine = hybrid.engine
    #: full depth when a sample needs d loss / d (its input rows): D3c's music code rows are embedding rows
    full = any(s.get("input_grad") is not None for s in samples)
    contexts, xs, ectx = [], [], []
    for s in samples:
        c = hybrid.prepare(s["segment"])
        contexts.append(c)
        xs.append(c["x"])
        ectx.append(hybrid.contexts(c["x"]))
    low: list[list[torch.Tensor]] = [[] for _ in samples]
    with torch.no_grad():
        for i in range(0, depth):
            layer = engine.layer(i, i + 1)
            for j in range(len(xs)):
                if full:
                    low[j].append(xs[j].detach().to(offload, copy=True))
                xs[j] = engine._run(layer, i, xs[j], ectx[j])
        x_d = [x.detach() for x in xs]
        xs = []
        for j, x in enumerate(x_d):
            with scope_of(samples[j]):
                xs.append(hybrid.apply_units(x, depth - 1, contexts[j]))
        bounds: list[list[torch.Tensor]] = [[] for _ in samples]
        for i in range(depth, L):
            layer = engine.layer(i, i + 1 if i + 1 < L else None)
            for j in range(len(xs)):
                bounds[j].append(xs[j].detach().to(offload, copy=True))
                xs[j] = engine._run(layer, i, xs[j], ectx[j])
    grads, losses = [], []
    for j, s in enumerate(samples):
        last = xs[j].detach().requires_grad_(True)
        with torch.enable_grad():
            normed = hybrid.final_norm(last)
            if s.get("head_loss") is not None:
                loss = s["head_loss"](normed, contexts[j])
            else:
                loss = s["loss_fn"](emission(hybrid, normed, contexts[j]))
        if not bool(torch.isfinite(loss)):
            raise MediaError(f"sample {s.get('name')}: loss is not finite ({float(loss)})")
        (loss * float(s["weight"]) * float(loss_scale)).backward()
        grads.append(last.grad)
        losses.append(float(loss.detach()))
        del last, normed, loss
    xs = None
    for i in reversed(range(depth, L)):
        layer = engine.layer(i, i - 1 if i - 1 >= depth else None)
        for j in range(len(samples)):
            x = bounds[j][i - depth].to(hybrid.device, copy=True).requires_grad_(True)
            with torch.enable_grad():
                y = engine._run(layer, i, x, ectx[j])
            y.backward(grads[j])
            grads[j] = x.grad
            bounds[j][i - depth] = None
            del x, y
    for j in range(len(samples)):
        y = x_d[j].detach().requires_grad_(full)
        feature = samples[j].get("feature")
        captured: list[torch.Tensor] = []
        capture = (d3.capture_block_output(hybrid.units[feature.container], captured) if feature is not None
                   else scope_of({}))
        with torch.enable_grad(), scope_of(samples[j]), capture:
            z = hybrid.apply_units(y, depth - 1, contexts[j])
        if feature is not None:
            # D3d's QVL32 term: weight 0.1 x MSE(unit block output, ViT block-13 output), joined to this recompute
            f_loss = feature.loss(captured) * float(samples[j]["weight"]) * float(loss_scale)
            torch.autograd.backward((z, f_loss), (grads[j], torch.ones_like(f_loss)))
            losses[j] += float(feature.value)
        elif z.requires_grad:
            z.backward(grads[j])
        if full:
            grads[j] = y.grad if y.grad is not None else grads[j]
        del z, y
    if full:
        for i in reversed(range(0, depth)):
            layer = engine.layer(i, i - 1 if i > 0 else None)
            for j in range(len(samples)):
                x = low[j][i].to(hybrid.device, copy=True).requires_grad_(True)
                with torch.enable_grad():
                    y = engine._run(layer, i, x, ectx[j])
                y.backward(grads[j])
                grads[j] = x.grad
                low[j][i] = None
                del x, y
        for j, s in enumerate(samples):
            if s.get("input_grad") is not None:
                s["input_grad"](grads[j])
    return {"losses": losses}


def group_eval(hybrid: Any, samples: list[dict[str, Any]], *, depth: int) -> list[float]:
    """Held-item losses, forward only (layer-major like the training step; no parameter receives a gradient).  The
    head losses that differentiate internally (chunked text objectives) run on a detached input."""

    L = int(hybrid.num_layers)
    engine = hybrid.engine
    contexts, xs, ectx = [], [], []
    for sm in samples:
        c = hybrid.prepare(sm["segment"])
        contexts.append(c)
        xs.append(c["x"])
        ectx.append(hybrid.contexts(c["x"]))
    with torch.no_grad():
        for i in range(L):
            layer = engine.layer(i, i + 1 if i + 1 < L else None)
            for j in range(len(xs)):
                xs[j] = engine._run(layer, i, xs[j], ectx[j])
            if i == depth - 1:
                for j in range(len(xs)):
                    with scope_of(samples[j]):
                        xs[j] = hybrid.apply_units(xs[j], i, contexts[j])
    out = []
    for j, sm in enumerate(samples):
        normed = hybrid.final_norm(xs[j].detach())
        with torch.enable_grad():
            if sm.get("head_loss") is not None:
                loss = sm["head_loss"](normed.detach(), contexts[j])
            else:
                with torch.no_grad():
                    loss = sm["loss_fn"](emission(hybrid, normed, contexts[j]))
        out.append(float(loss.detach()))
    for p in list(media_trainables_cache.get(id(hybrid), [])):
        p.grad = None
    return out


media_trainables_cache: dict[int, list[torch.Tensor]] = {}


def evaluate_held(hybrid: Any, sampler: "GroupSampler", *, depth: int, verify: bool, per_route: int) -> dict[str, Any]:
    result = {}
    for route, items in sorted(sampler.held.items()):
        losses = []
        chunk: list[dict[str, Any]] = []
        for it in items[:per_route]:
            try:
                chunk.append(bind_item(hybrid, it, verify=verify))
            except (MediaError, d3.D3Error) as exc:  # a held item that cannot be bound is reported, never fatal
                sampler.skipped.setdefault(f"held:{route}", []).append({"name": it.get("name"), "reason": str(exc)[:300]})
                log(f"eval: skip held {route} item {it.get('name')}: {exc}")
                continue
            if len(chunk) == 8:
                losses += group_eval(hybrid, chunk, depth=depth)
                chunk = []
        if chunk:
            losses += group_eval(hybrid, chunk, depth=depth)
        if losses:
            result[route] = {"mean": sum(losses) / len(losses), "items": len(losses)}
    return result


def reference_per_sample(hybrid: Any, samples: list[dict[str, Any]], *, depth: int) -> list[float]:
    """d3_hybrid_module's own D3b step, one sample at a time (the selftest reference)."""

    losses = []
    for s in samples:
        context = hybrid.prepare(s["segment"])
        with torch.no_grad():
            x_d = hybrid.run_layers(context["x"], 0, depth, context, units_at_stop=False)
        losses.append(d3.d3b_forward_backward(hybrid, context, x_d, depth, s["loss_fn"], loss_scale=float(s["weight"])))
    return losses


# --------------------------------------------------------------------------
# Data: sealed native flow batches.
# --------------------------------------------------------------------------


#: data-config entry kinds -> the receipt schema each one reads
#: answers that are only registry metadata ("image artifact: x.png; mime=...; bytes=...") are never trained on
import re as _re
METADATA_ANSWER = _re.compile(r"^(image|video) artifact:.*(mime=|bytes=)")


class VisionIndex:
    """D3d data: vision_in rows with ONE image, a non-trivial answer, a sealed QVL32 ViT L13 shard for that image
    (joined by image sha256) and the row's Q38 parent answer distribution (qtext_vl cache, same row index)."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self.corpus = Path(entry["corpus"])
        self.qtext = Path(entry["qtext_vl"])
        self.qvl = Path(entry["qvl32"])
        m = json.loads((self.qtext / "manifest.json").read_text(encoding="utf-8"))
        if not (m.get("all_finite") and m.get("all_structural_ok")) or m["corpus"].get("jsonl_sha256") != file_sha256(self.corpus):
            raise MediaError(f"{self.qtext}: not a sealed qtext_vl cache over {self.corpus}")
        self.rows: dict[int, tuple[str, int, int, int]] = {}
        for e in m["shards"]:
            rec = json.loads((self.qtext / e["shard"] / "shard.json").read_text(encoding="utf-8"))
            for t in rec["row_table"]:
                self.rows[int(t["row"])] = (e["shard"], int(t["offset"]), int(t["count"]), int(t["length"]))
        self.qvl_by_sha: dict[str, Path] = {}
        for side in sorted(self.qvl.glob("shard-*.json")):
            j = json.loads(side.read_text(encoding="utf-8"))
            item = j.get("item") or {}
            if item.get("held"):
                continue
            if item.get("sha256"):
                self.qvl_by_sha[str(item["sha256"])] = side
        self.max_pixels = int((json.loads((self.qvl / "manifest.json").read_text(encoding="utf-8")).get("run_config") or {})
                              .get("knobs", {}).get("max_pixels", 262144))
        self.items: list[dict[str, Any]] = []
        skipped = {"not_one_image": 0, "metadata_or_empty_answer": 0, "no_qvl_shard": 0, "no_cache_row": 0}
        for r, line in enumerate(self.corpus.read_text(encoding="utf-8").splitlines()):
            rec = json.loads(line)
            media = rec.get("media") or []
            if len(media) != 1 or media[0].get("media") != "image":
                skipped["not_one_image"] += 1
                continue
            answer = (rec.get("reference_response") or "").strip()
            if not answer or METADATA_ANSWER.match(answer):
                skipped["metadata_or_empty_answer"] += 1
                continue
            sha = str(media[0].get("sha256"))
            if sha not in self.qvl_by_sha:
                skipped["no_qvl_shard"] += 1
                continue
            if r not in self.rows:
                skipped["no_cache_row"] += 1
                continue
            self.items.append({"name": f"vision-{r:05d}", "row": r, "image": media[0].get("stored_path") or media[0]["path"],
                               "sha256": sha, "qvl_sidecar": str(self.qvl_by_sha[sha])})
        self.skipped = skipped
        self._shard_cache: dict[str, Any] = {}

    def q38_row(self, r: int) -> dict[str, Any]:
        import qtext_teacher_cache as qtc  # the cache format's own reader

        shard, offset, count, length = self.rows[r]
        if shard not in self._shard_cache:
            if len(self._shard_cache) >= 2:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[shard] = qtc.read_shard(self.qtext / shard)
        a = self._shard_cache[shard]
        sl = slice(offset, offset + count)
        return {"ids": a["ids"][sl], "pos": a["pos"][sl], "topk_ids": a["topk_ids"][sl].to(torch.int64),
                "topk_lp": a["topk_logprobs"][sl].float(), "length": length}


def bind_vision(hybrid: Any, index: VisionIndex, item: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """One D3d sample: the Q38 row with its image-pad run replaced by the QVL32 patch rows (type 6, C11 band), the
    teacher's answer distribution at the shifted answer positions, and the QVL32 block-13 feature term."""

    from safetensors.torch import load_file

    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vstart = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vend = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    row = index.q38_row(int(item["row"]))
    L = int(row["length"])
    ids_q = torch.full((L,), int(pad), dtype=torch.int64)
    ids_q[row["pos"]] = row["ids"].to(torch.int64)
    stored = torch.zeros(L, dtype=torch.bool)
    stored[row["pos"]] = True
    gaps = (~stored).nonzero().flatten()
    if gaps.numel() == 0:
        raise MediaError(f"{item['name']}: the Q38 row has no image tokens")
    a, b = int(gaps[0]), int(gaps[-1]) + 1
    if b - a != int(gaps.numel()) or int(ids_q[a - 1]) != vstart or int(ids_q[b]) != vend:
        raise MediaError(f"{item['name']}: the Q38 row's image tokens are not one run inside vision_start / vision_end")
    side = json.loads(Path(item["qvl_sidecar"]).read_text(encoding="utf-8"))
    teacher = load_file(str(Path(item["qvl_sidecar"]).with_suffix(".safetensors")))
    patches, _hw = d3.qvl_patches_from_image(item["image"], item["sha256"], max_pixels=index.max_pixels)
    n = int(teacher["block_out"].shape[0])
    if int(patches.shape[0]) != n:
        raise MediaError(f"{item['name']}: {patches.shape[0]} patch rows vs teacher {n}")
    ph = int(hybrid.media_placeholder)
    ids_s = torch.cat([ids_q[:a], torch.full((n,), ph, dtype=torch.int64), ids_q[b:]]).view(1, -1)
    delta = n - (b - a)
    lo, hi = hybrid.bands[6]["feature"]
    width = int(hybrid.cfg.native_feature_dim)
    features = torch.zeros(1, n, width)
    features[0, :, lo:hi] = patches
    segment = {"ids": ids_s, "features": features, "types": torch.full((1, n), 6, dtype=torch.int64),
               "positions": teacher["positions"].float().view(1, n, 4), "indices": torch.arange(a, a + n).view(1, n),
               "sigma": None, "media_rows": n, "prompt_tokens": a}
    d3.check_vision_alignment(segment, teacher, hybrid.bands, qvl_type=6)
    # the answer: after the LAST "<|im_start|>assistant" header (and its empty think block), through its <|im_end|>
    starts = (ids_q == im_start).nonzero().flatten().tolist()
    head = starts[-1]
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    j = head + 1
    while j < L and int(ids_q[j]) != think_end and j < head + 4:
        j += 1
    body = (j + 2) if j < L and int(ids_q[j]) == think_end else (head + 3)  # skip </think> and its blank-line token
    ends = [k for k in range(body, L) if int(ids_q[k]) == im_end]
    if not ends:
        raise MediaError(f"{item['name']}: no <|im_end|> after the assistant answer")
    stop = ends[0]
    # q38 positions p predicting answer tokens body..stop (inclusive): p in [body - 1, stop - 1]
    want = torch.arange(body - 1, stop)
    lookup = {int(pp): k for k, pp in enumerate(row["pos"].tolist())}
    if any(int(pp) not in lookup for pp in want):
        raise MediaError(f"{item['name']}: an answer position is not stored in the Q38 cache")
    sel = torch.tensor([lookup[int(pp)] for pp in want])
    positions_s = want + delta
    next_ids = ids_s[0, positions_s + 1]
    targets = d3.answer_kd_targets(int(ids_s.shape[1]), positions_s, next_ids, row["topk_ids"][sel], row["topk_lp"][sel])
    mask = torch.zeros(ids_s.shape, dtype=torch.bool)
    mask[0, positions_s + 1] = True
    count = d3.check_answer_targets(targets, ids_s, mask, int(hybrid.base_vocab))
    container = next(p.container for p in hybrid.layout if p.source == "qvl32")
    return {"segment": segment, "head_loss": d3.d3d_head_loss(hybrid, targets, denominator=float(count)),
            "feature": d3.UnitFeatureKD(container=container, teacher=teacher["block_out"].float()),
            "binding": {"kind": "vision_answer", "row": int(item["row"]), "patch_rows": n, "answer_positions": count,
                        "q38_image_tokens": b - a, "qvl_sha256": side.get("file_sha256")}}


KIND_SCHEMAS = {"native_flow": FLOW_SCHEMA, "speech_flow": "chimera_recovery.speech_flow_cache.v1",
                "speech_ce": "chimera_recovery.audio_cache.v1", "ace_codes": "omni-campaign.teacher_cache_shard.v2",
                "vision_answer": "vision_in.jsonl + qtext_vl + qvl32_vit_l13"}
SPEECH_ROUTES = {"speech_flow": "speech_out", "speech_ce": "speech_in", "ace_codes": "music_codes", "vision_answer": "vision_in"}
#: ACE-Step 1.5 LM (qwen_1.7b_ace15): code k = token - 151669 for tokens in [151669, 215669); 64,000 written codes.
#: The cache (media_b_teacher_cache ace_lm_codes, band_topk + the transcribed LM loop) stores CODE INDICES k for both the
#: generated sequence and the top-k ids, with the EOS step dropped (one record per emitted code).  1.3.3 reads them as
#: such; 1.2.0-1.3.2 compared them with the absolute band and refused every real shard.
ACE_LM_BAND = (151_669, 215_669)


def ace_code_indices(values: torch.Tensor, what: str, codes: int) -> torch.Tensor:
    """Code indices [0, codes) as stored; an absolute LM-token copy [151669, 215669) is shifted; anything else refused
    (the two ranges cannot overlap, so the layout is never ambiguous)."""

    lo, hi = ACE_LM_BAND
    if values.numel() == 0 or bool(((values >= 0) & (values < codes)).all()):
        return values
    if bool(((values >= lo) & (values < hi)).all()):
        return values - lo
    raise MediaError(f"{what}: values are neither code indices [0, {codes}) nor ACE LM tokens [{lo}, {hi})")
_TOKENIZER: dict[str, Any] = {}


def student_tokenizer(path: str) -> Any:
    if path not in _TOKENIZER:
        from transformers import AutoTokenizer

        _TOKENIZER[path] = AutoTokenizer.from_pretrained(path)
    return _TOKENIZER[path]


def ace_prompt_prefix(tokenizer: Any, caption: str, lyrics: str, node_defaults: dict[str, Any]) -> list[int]:
    """The music prompt exactly as Codex's ACE native batches render it (ace_cache.py): the user turn
    "Music caption:\n{caption}\nLyrics:\n{lyrics}\nTeacher settings:\n{canonical node defaults}", then the assistant
    turn opened with an empty think block; the codes follow."""

    knobs = json.dumps(node_defaults, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    prompt = "Music caption:\n" + caption + "\nLyrics:\n" + (lyrics or "") + "\nTeacher settings:\n" + knobs
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def bind_ace_codes(hybrid: Any, music: "MusicBand", item: dict[str, Any], sidecar: dict[str, Any], stored: dict[str, torch.Tensor],
                   tokenizer: Any, node_defaults: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    codes = ace_code_indices(stored["tokens"].to(torch.int64).flatten(), f"{item['name']} tokens", music.codes)
    n = int(codes.numel())
    if n < 1:
        raise MediaError(f"{item['name']}: no ACE codes stored")
    tk = ace_code_indices(stored["topk_ids"].to(torch.int64), f"{item['name']} topk_ids", music.codes)
    for key in ("topk_ids", "topk_logprobs", "lse_full", "lse_band"):
        if int(stored[key].shape[0]) != n:
            raise MediaError(f"{item['name']}: {key} has {int(stored[key].shape[0])} rows for {n} codes")
    rec = sidecar["item"]
    prefix = ace_prompt_prefix(tokenizer, rec.get("caption") or "", rec.get("lyrics") or "", node_defaults)
    p = len(prefix)
    ids = torch.tensor([prefix + (codes + hybrid.music_band_begin).tolist()], dtype=torch.int64)
    lp_band = d3.band_teacher_logprobs(stored["topk_logprobs"], stored["lse_full"], stored["lse_band"])
    # bf16 storage noise: on rows where the conditional branch puts ~all mass on EOS (log p_full ~ -32, bf16 step 0.25,
    # ~1 % of rows, the last of a min_tokens sequence) the renormalised top-k can sum past 1; such a row is scaled back
    # to mass 1 (tail 0), never left over-weighted (the bucket tail clamps at 0, so the KL would count it > 1)
    excess = torch.logsumexp(lp_band, dim=-1).clamp(min=0.0)
    lp_band = lp_band - excess.unsqueeze(-1)
    # each row's top-k codes (descending), padded with unused distinct codes at a negligible log-prob if short
    out_ids = torch.empty(n, top_k, dtype=torch.int64)
    out_lp = torch.empty(n, top_k, dtype=torch.float32)
    for r in range(n):
        ids_r, lp_r = tk[r], lp_band[r]
        order = torch.argsort(lp_r, descending=True, stable=True)
        ids_r, lp_r = ids_r[order][:top_k], lp_r[order][:top_k]
        if ids_r.numel() < top_k:
            used = set(ids_r.tolist())
            pad = [c for c in range(music.codes) if c not in used][: top_k - ids_r.numel()]
            ids_r = torch.cat([ids_r, torch.tensor(pad, dtype=torch.int64)])
            lp_r = torch.cat([lp_r, torch.full((len(pad),), -100.0)])
        out_ids[r], out_lp[r] = ids_r, lp_r
    positions = torch.arange(p - 1, p - 1 + n)
    return {"segment": {"ids": ids, "prompt_tokens": p}, "head_loss": music.head_loss(positions, out_ids, out_lp),
            "input_grad": music.input_grad(ids), "binding": {"kind": "ace_codes", "prompt_tokens": p, "codes": n, "eos_hit": bool(sidecar.get("eos_hit")),
                        "rows_renormalised": int((excess > 0).sum())}}


def tensors_beside(receipt_path: Path) -> Path:
    """`<stem>.json` -> `<stem>.safetensors`; Codex's `<stem>.receipt.json` -> `<stem>.safetensors`."""

    name = receipt_path.name
    stem = name[: -len(".receipt.json")] if name.endswith(".receipt.json") else name[: -len(".json")]
    return receipt_path.with_name(stem + ".safetensors")


class FlowSet:
    """Every sealed train-split receipt under the configured directories, by route.

    Entry kinds: `native_flow` (Codex native_flow_cache.v1: ACE / QIMG / H3 guided velocity), `speech_flow` (Codex
    speech_flow_cache.v1: D3e_out, the H3 audio channel speaking), `speech_ce` (Codex audio_cache.v1: D3e_in, answer CE
    on audio observations).  The speech recordings and texts are Cyril's own data (Cyril, 2026-10-18)."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.routes: dict[str, list[dict[str, Any]]] = {}
        self.fractions: dict[str, float] = {}
        self.kinds: dict[str, str] = {}
        #: per-route group token cap (full-depth routes keep all 64 layers of boundaries, so they run smaller groups)
        self.group_tokens: dict[str, int] = {}
        for entry in spec["routes"]:
            kind = entry.get("kind", "native_flow")
            if kind not in KIND_SCHEMAS:
                raise MediaError(f"entry kind {kind!r}: one of {sorted(KIND_SCHEMAS)}")
            route = d3.flow_route(entry["route"]) if kind == "native_flow" else SPEECH_ROUTES[kind]
            if route in self.routes:
                raise MediaError(f"route {route} appears twice")
            items, skipped = [], 0
            if kind == "vision_answer":
                vindex = VisionIndex(entry)
                EXTRAS["vision_index"] = vindex
                items = [{**it, "route": route, "kind": kind, "receipt_path": it["qvl_sidecar"], "tensors": ""} for it in vindex.items]
                log(f"route {route}: {len(items)} vision rows; skipped {vindex.skipped}")
            for directory in ([] if kind == "vision_answer" else entry["dirs"]):
                for path in sorted(Path(directory).glob(entry.get("glob", "*.json"))):
                    if path.name == "manifest.json":
                        continue
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                    if receipt.get("schema") != KIND_SCHEMAS[kind]:
                        continue
                    held = (receipt.get("item") or {}).get("held", False) if kind == "ace_codes" else receipt.get("held", False)
                    if receipt.get("split", "train") != "train" or held:
                        skipped += 1
                        continue
                    if kind == "native_flow" and d3.flow_route(receipt.get("route")) != route:
                        raise MediaError(f"{path}: route {receipt.get('route')} under the {route} entry")
                    tensors = tensors_beside(path)
                    if not tensors.is_file():
                        raise MediaError(f"{path}: no tensor file {tensors.name} beside the receipt")
                    items.append({"name": path.name, "receipt_path": str(path), "tensors": str(tensors), "route": route,
                                  "kind": kind})
            if not items:
                raise MediaError(f"route {route}: no train {KIND_SCHEMAS[kind]} receipts under {entry['dirs']}")
            self.routes[route] = items
            self.kinds[route] = kind
            self.fractions[route] = float(entry.get("fraction", 1.0))
            if entry.get("group_tokens") is not None:
                self.group_tokens[route] = int(entry["group_tokens"])
            if skipped:
                log(f"route {route}: {skipped} held / non-train receipts left out")
        if not self.routes:
            raise MediaError("the media data config names no route")

    def identity(self) -> dict[str, Any]:
        return {route: {"items": len(items), "kind": self.kinds[route], "fraction": self.fractions[route],
                        "names_sha256": canonical_sha256([i["name"] for i in items])} for route, items in self.routes.items()}


def receipt_tensor_sha(receipt: dict[str, Any]) -> str | None:
    return receipt.get("cache_sha256") or receipt.get("tensors_sha256")


def speech_segment(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    """A Codex speech batch (one unpadded sample) as a HybridD3 segment, H3 charted source positions included."""

    if batch["input_ids"].shape[0] != 1 or not bool(batch["native_media_valid_mask"].all()):
        raise MediaError("HybridD3 runs one unpadded sample per call")
    return {"ids": batch["input_ids"], "features": batch["native_media_features"], "types": batch["native_media_type_ids"],
            "positions": batch["native_media_positions"], "indices": batch["native_media_indices"],
            "sigma": batch["native_media_noise_levels"], "media_rows": int(batch["native_media_type_ids"].shape[1]),
            "prompt_tokens": int(batch["native_media_indices"].min()),
            "source_positions": {"h3": batch["charted_source_positions"]}}


def speech_sample(hybrid: Any, kind: str, segment: dict[str, Any], prompt_length: int, *, target: torch.Tensor | None = None,
                  weights: torch.Tensor | None = None, target_mask: torch.Tensor | None = None,
                  completion_eos_id: int | None = None) -> dict[str, Any]:
    """D3e as a group sample: d3_hybrid_module's own input checks and losses, the H3 unit's prompt-only read scope
    (speech-in: the prompt; speech-out: the whole sequence), exactly as `speech_lane_step` uses them."""

    unit = d3.h3_speech_unit(hybrid)
    if kind == "speech_flow":
        d3.check_speech_out(hybrid, segment, target, weights, prompt_length)
        head_loss = d3.speech_out_flow_loss(hybrid, segment, target, weights)
        scope_len = int(segment["ids"].shape[1])
    elif kind == "speech_ce":
        d3.check_speech_in(hybrid, segment, prompt_length)
        targets, count = d3.speech_in_targets(hybrid, segment, target_mask, prompt_length, int(completion_eos_id))
        head_loss = d3.speech_in_head_loss(hybrid, targets, count)
        scope_len = prompt_length
    else:
        raise MediaError(f"speech kind {kind!r}")
    return {"segment": segment, "head_loss": head_loss, "binding": {"kind": kind, "prompt_length": prompt_length},
            "scope": (lambda c=unit.container, n=scope_len: d3.prompt_only_unit(hybrid, c, n))}


EXTRAS: dict[str, Any] = {}


def bind_item(hybrid: Any, item: dict[str, Any], *, verify: bool = True) -> dict[str, Any]:
    from safetensors.torch import load_file

    if item.get("kind") == "vision_answer":
        out = {"name": item["name"], "route": item["route"], "weight": 1.0,
               **bind_vision(hybrid, EXTRAS["vision_index"], item, student_tokenizer(EXTRAS.get("tokenizer") or EXTRAS["parent_config"]))}
        out["tokens"] = int(out["segment"]["ids"].shape[1])
        return out
    receipt = json.loads(Path(item["receipt_path"]).read_text(encoding="utf-8"))
    if item.get("kind") == "ace_codes":
        if verify and file_sha256(Path(item["tensors"])) != receipt.get("file_sha256"):
            raise MediaError(f"{item['name']}: ACE codes shard does not hash like its sidecar")
        music = EXTRAS.get("music")
        if music is None:
            raise MediaError("the music_codes route needs the music band trainables (MusicBand)")
        stored = load_file(item["tensors"])
        out = {"name": item["name"], "route": item["route"], "weight": 1.0,
               **bind_ace_codes(hybrid, music, item, receipt, stored, student_tokenizer(EXTRAS["tokenizer"]),
                                EXTRAS["ace_node_defaults"], top_k=int(stored["topk_ids"].shape[1]))}
        out["tokens"] = int(out["segment"]["ids"].shape[1])
        return out
    if receipt.get("sealed") is not True:
        raise MediaError(f"{item['name']}: receipt is not sealed")
    if verify:
        got = file_sha256(Path(item["tensors"]))
        if got != receipt_tensor_sha(receipt):
            raise MediaError(f"{item['name']}: tensor file sha {got[:16]} != receipt {str(receipt_tensor_sha(receipt))[:16]}")
    stored = load_file(item["tensors"])
    storage = receipt.get("storage") or {}
    width = int(hybrid.cfg.native_feature_dim)
    batch = d3.inflate_native_batch(stored, storage, width=width) if storage.get("schema") == "chimera_recovery.native_band.v1" else stored
    kind = item.get("kind", "native_flow")
    out = {"name": item["name"], "route": item["route"], "weight": 1.0}
    if kind == "native_flow":
        loss_fn, info = d3.bind_native_flow_batch(receipt, batch, hybrid.bands, route=item["route"], flow_param="x0",
                                                  signs=d3.flow_signs())
        segment = d3.native_batch_segment(batch, media_placeholder=hybrid.media_placeholder)
        out.update({"segment": segment, "loss_fn": loss_fn, "binding": info})
    elif kind == "speech_flow":
        if receipt.get("target_kind") != "x0" or receipt.get("prediction_kind") != "x0":
            raise MediaError(f"{item['name']}: speech flow must be x0 head / x0 target")
        segment = speech_segment(batch)
        target = batch["native_media_targets"]
        weights = (batch["native_media_target_mask"].float() if "native_media_target_mask" in batch
                   else torch.ones(segment["types"].shape))
        out.update(speech_sample(hybrid, kind, segment, int(receipt["prompt_length"]), target=target, weights=weights))
    else:
        segment = speech_segment(batch)
        out.update(speech_sample(hybrid, kind, segment, int(receipt["prompt_length"]), target_mask=batch["assistant_target_mask"],
                                 completion_eos_id=int(receipt["completion_eos_id"])))
    out["tokens"] = int(out["segment"]["ids"].shape[1])
    return out


class GroupSampler:
    """Deterministic groups: route drawn by fraction (seeded per group), items in a seeded permutation per route and
    epoch, a group closes at `group_tokens` (a longer item runs alone).  State = (groups drawn, per-route position)."""

    def __init__(self, flows: FlowSet, *, seed: int, held_per_route: int, group_tokens: int, max_group: int) -> None:
        self.seed = int(seed)
        self.group_tokens = int(group_tokens)
        self.max_group = int(max_group)
        self.train: dict[str, list[dict[str, Any]]] = {}
        self.held: dict[str, list[dict[str, Any]]] = {}
        for route, items in flows.routes.items():
            order = list(items)
            random.Random(f"{seed}:split:{route}").shuffle(order)
            k = min(int(held_per_route), max(0, len(order) - 1))
            self.held[route] = order[len(order) - k:] if k else []
            self.train[route] = order[: len(order) - k]
        self.fractions = {r: flows.fractions[r] for r in self.train}
        self.route_tokens = dict(flows.group_tokens)
        self.skipped: dict[str, list[dict[str, Any]]] = {}
        self.groups = 0
        self.position = {r: 0 for r in self.train}

    def _item(self, route: str) -> dict[str, Any]:
        items = self.train[route]
        pos = self.position[route]
        epoch, index = divmod(pos, len(items))
        order = list(range(len(items)))
        random.Random(f"{self.seed}:epoch:{route}:{epoch}").shuffle(order)
        self.position[route] = pos + 1
        return items[order[index]]

    def next_route(self) -> str:
        rng = random.Random(f"{self.seed}:group:{self.groups}")
        total = sum(self.fractions.values())
        u = rng.random() * total
        acc = 0.0
        for route in sorted(self.fractions):
            acc += self.fractions[route]
            if u <= acc:
                return route
        return sorted(self.fractions)[-1]

    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "groups": self.groups, "position": dict(self.position)}

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise MediaError("sampler seed differs from the checkpoint's")
        self.groups = int(state["groups"])
        self.position.update({k: int(v) for k, v in state["position"].items()})


def draw_group(sampler: GroupSampler, hybrid: Any, *, verify: bool) -> list[dict[str, Any]]:
    route = sampler.next_route()
    cap = int(sampler.route_tokens.get(route, sampler.group_tokens))
    group, tokens = [], 0
    misses = 0
    while len(group) < sampler.max_group:
        saved = dict(sampler.position)
        item = sampler._item(route)
        try:
            sample = bind_item(hybrid, item, verify=verify)
        except (MediaError, d3.D3Error) as exc:
            misses += 1
            sampler.skipped.setdefault(route, []).append({"name": item.get("name"), "reason": str(exc)[:300]})
            log(f"skip {route} item {item.get('name')}: {exc}")
            if misses > 50:
                raise MediaError(f"route {route}: more than 50 items in a row cannot be bound (last: {exc})")
            continue
        if group and tokens + sample["tokens"] > cap:
            sampler.position = saved  # this item opens the next group of the route
            break
        group.append(sample)
        tokens += sample["tokens"]
        if tokens >= cap:
            break
    sampler.groups += 1
    for s in group:
        s["weight"] = 1.0 / len(group)
    return group


# --------------------------------------------------------------------------
# Optimizer step, checkpoints, export.
# --------------------------------------------------------------------------


def apply_updates(trainables: dict[str, torch.Tensor], optimizer: Any, lr: float, step: int) -> dict[str, Any]:
    stats = {"updated": 0, "no_grad": [], "nonfinite": [], "clipped": 0, "grad_sq": 0.0}
    for name, p in trainables.items():
        g = p.grad
        if g is None:
            stats["no_grad"].append(name)
            continue
        norm = float(torch.linalg.vector_norm(g.float()))
        if not math.isfinite(norm):
            stats["nonfinite"].append(name)
            p.grad = None
            continue
        stats["grad_sq"] += norm * norm
        if lr > 0:
            info = optimizer.update(name, p.data, g.float(), lr, step=step)
            stats["clipped"] += int(info["scale"] < 1.0)
        stats["updated"] += 1
        p.grad = None
    stats["grad_norm"] = math.sqrt(stats["grad_sq"])
    return stats


def lr_at(step: int, *, base: float, warmup: int, total: int, min_ratio: float) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(directory: Path, step: int, trainables: dict[str, torch.Tensor], optimizer: Any, state: dict[str, Any],
                    keep: int) -> Path:
    from safetensors.torch import save_file

    out = directory / f"step-{step:07d}"
    tmp = directory / f".step-{step:07d}.tmp-{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in trainables.items()}, str(tmp / "media.safetensors"))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    write_json_atomic(tmp / "trainer_state.json", {"schema": CHECKPOINT_SCHEMA, **state,
                                                   "media_sha256": file_sha256(tmp / "media.safetensors")})
    os.replace(tmp, out)
    kept = sorted(p for p in directory.glob("step-*") if p.is_dir())
    for old in kept[:-int(keep)]:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    return out


def latest_checkpoint(directory: Path) -> Path | None:
    found = sorted(p for p in Path(directory).glob("step-*") if (p / "trainer_state.json").is_file() and (p / "media.safetensors").is_file())
    return found[-1] if found else None


def load_checkpoint(path: Path, trainables: dict[str, torch.Tensor], optimizer: Any) -> dict[str, Any]:
    from safetensors.torch import load_file

    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise MediaError(f"{path}: checkpoint schema {state.get('schema')}")
    if file_sha256(path / "media.safetensors") != state["media_sha256"]:
        raise MediaError(f"{path}: media.safetensors does not hash like its state")
    values = load_file(str(path / "media.safetensors"))
    unknown = sorted(set(values) - set(trainables))
    if unknown:
        raise MediaError(f"{path}: checkpoint tensors this run does not train: {unknown[:4]}")
    fresh = sorted(set(trainables) - set(values))
    if fresh:
        log(f"resume: {len(fresh)} trainables are new since the checkpoint and keep their base values (first {fresh[:3]})")
    with torch.no_grad():
        for name, p in trainables.items():
            if name in values:
                p.copy_(values[name].to(p.device, p.dtype))
    state["fresh_since_checkpoint"] = fresh
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    return state


def export_media(out: Path, trainables: dict[str, torch.Tensor], metadata: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    path = out / "media_modules.safetensors"
    save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in trainables.items()}, str(path),
              metadata={"schema": EXPORT_SCHEMA, "tool": f"{TOOL_NAME} {TOOL_VERSION}"})
    record = {"schema": EXPORT_SCHEMA, "file": str(path), "sha256": file_sha256(path), "tensors": len(trainables),
              "parameters": int(sum(v.numel() for v in trainables.values())), "names": "Chimera names (blocks.<b>.charted.*, "
              "native_media_head.*): they replace the same tensors of the base child at reassembly", **metadata}
    write_json_atomic(out / "media_modules.json", record)
    return record


# --------------------------------------------------------------------------
# Hybrid construction.
# --------------------------------------------------------------------------


def build_hybrid(args: argparse.Namespace) -> Any:
    dtype = getattr(torch, args.dtype)
    options = d3.HybridOptions(dtype=dtype, device=args.device, head_device=args.head_device or "cpu", kernels=args.kernels, lora=None,
                               plan=args.plan, pin_boundaries=False, prefetch=True)
    return d3.HybridD3(Path(args.omni1), Path(args.parent_config), options)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def tiny_speech(hybrid: Any, cfg: Any, lane: str, seed: int) -> dict[str, Any]:
    """A tiny D3e sample that passes d3's own check_speech_out / check_speech_in."""

    g = torch.Generator().manual_seed(seed)
    width = int(cfg.native_feature_dim)
    ph = int(hybrid.media_placeholder)
    vocab = int(hybrid.base_vocab)
    flo, fhi = hybrid.noisy_feature_bands()[3]
    lo, hi = hybrid.bands[3]["feature"]
    elo, ehi = hybrid.bands[3]["emission"]
    if lane == "speech_flow":
        p, m = 5, 6
        ids = torch.cat([torch.randint(0, vocab, (1, p), generator=g), torch.full((1, m), ph)], dim=1)
        features = torch.zeros(1, m, width)
        features[..., flo:fhi] = torch.randn(1, m, fhi - flo, generator=g)
        target = torch.zeros(1, m, width)
        target[..., elo:ehi] = torch.randn(1, m, ehi - elo, generator=g)
        seg = {"ids": ids, "features": features, "types": torch.full((1, m), 3), "positions": torch.rand(1, m, 4, generator=g),
               "indices": torch.arange(p, p + m).view(1, m), "sigma": torch.full((1, m), 0.5), "media_rows": m, "prompt_tokens": p,
               "source_positions": {"h3": torch.zeros(1, p + m, 3)}}
        weights = torch.ones(1, m)
        sample = speech_sample(hybrid, lane, seg, p, target=target, weights=weights)
        sample["reference"] = {"prompt_length": p, "target": target, "weights": weights}
    else:
        a, m, b = 2, 3, 2  # [text a | audio m | text b] = the prompt, then the answer + EOS
        eos = vocab - 1
        p = a + m + b
        answer = torch.randint(0, vocab - 1, (1, 3), generator=g)
        ids = torch.cat([torch.randint(0, vocab - 1, (1, a), generator=g), torch.full((1, m), ph),
                         torch.randint(0, vocab - 1, (1, b), generator=g), answer, torch.tensor([[eos]])], dim=1)
        features = torch.zeros(1, m, width)
        features[..., lo:hi] = torch.randn(1, m, hi - lo, generator=g)
        seg = {"ids": ids, "features": features, "types": torch.full((1, m), 3), "positions": torch.rand(1, m, 4, generator=g),
               "indices": torch.arange(a, a + m).view(1, m), "sigma": torch.zeros(1, m), "media_rows": m, "prompt_tokens": a,
               "source_positions": {"h3": torch.zeros(1, ids.shape[1], 3)}}
        mask = torch.zeros(ids.shape, dtype=torch.bool)
        mask[0, p:] = True
        sample = speech_sample(hybrid, lane, seg, p, target_mask=mask, completion_eos_id=eos)
        sample["reference"] = {"prompt_length": p, "target_mask": mask, "completion_eos_id": eos}
    sample["name"] = f"tiny-{lane}-{seed}"
    return sample


def speech_parity(hybrid: Any, cfg: Any, trainables: dict[str, torch.Tensor], depth: int) -> dict[str, Any]:
    out = {}
    for lane, d3_lane in (("speech_flow", "speech_out_flow"), ("speech_ce", "speech_in_ce")):
        samples = [tiny_speech(hybrid, cfg, lane, seed) for seed in (41, 42)]
        for s in samples:
            s["weight"] = 1.0 / len(samples)
        for p in trainables.values():
            p.grad = None
        group = group_forward_backward(hybrid, samples, depth=depth)
        got = {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in trainables.items()}
        for p in trainables.values():
            p.grad = None
        ref_losses = []
        for s in samples:
            r = s["reference"]
            kwargs = ({"target": r["target"], "weights": r["weights"]} if lane == "speech_flow"
                      else {"target_mask": r["target_mask"], "completion_eos_id": r["completion_eos_id"]})
            ref_losses.append(d3.speech_lane_step(hybrid, d3_lane, s["segment"], prompt_length=r["prompt_length"],
                                                  loss_scale=float(s["weight"]), **kwargs)["loss"])
        want = {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in trainables.items()}
        for p in trainables.values():
            p.grad = None
        worst, compared, mismatch = 0.0, 0, []
        for name in trainables:
            a, b = got[name], want[name]
            if (a is None) != (b is None):
                mismatch.append(name)
                continue
            if a is None or float(b.norm()) == 0.0:
                continue
            compared += 1
            worst = max(worst, float((a - b).norm() / b.norm()))
        out[lane] = {"worst_rel": worst, "compared": compared, "presence_mismatch": mismatch,
                     "loss_gap": max(abs(x - y) for x, y in zip(group["losses"], ref_losses)), "losses": group["losses"]}
    return out


def vision_parity(tmp: Path) -> dict[str, Any]:
    """D3d through the group step (answer KD on the text head + the unit feature term) vs d3's own
    `full_depth_forward_backward(..., feature=UnitFeatureKD)` on a tiny child with a QVL-like unit."""

    built = d3.build_tiny_child(tmp / "child-qvl", with_qvl=True)
    h = d3.HybridD3(built["dir"], built["parent_config"],
                    d3.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=None, pin_boundaries=False))
    tr = media_trainables(h)
    depth = h.layout[0].depth
    qvl = next(p for p in h.layout if p.source == "qvl32")
    g = torch.Generator().manual_seed(77)

    def sample(seed: int) -> dict[str, Any]:
        seg = d3.tiny_segment(h.cfg, "ocr", rows=5, seed=seed, suffix=6)
        n, L = int(seg["media_rows"]), int(seg["ids"].shape[1])
        answer = torch.arange(int(seg["prompt_tokens"]) + n, L - 1)
        q_ids = torch.randint(0, int(h.base_vocab), (answer.numel(), 8), generator=g)
        q_lp = torch.log_softmax(torch.randn(answer.numel(), 8, generator=g), -1) - 1.0
        targets = d3.answer_kd_targets(L, answer, seg["ids"][0, answer + 1], q_ids, q_lp)
        sink: list[torch.Tensor] = []
        with torch.no_grad(), d3.capture_block_output(h.units[qvl.container], sink):
            h.forward(seg)
        teacher = torch.randn(tuple(sink[0].shape), generator=g) * 0.1  # the unit's own block width
        return {"segment": seg, "targets": targets, "count": int(answer.numel()), "teacher": teacher}

    raw = [sample(91), sample(92)]
    mine = [{"name": f"tiny-vis-{k}", "segment": r["segment"], "weight": 0.5,
             "head_loss": d3.d3d_head_loss(h, r["targets"], denominator=float(r["count"])),
             "feature": d3.UnitFeatureKD(container=qvl.container, teacher=r["teacher"])} for k, r in enumerate(raw)]
    for p in tr.values():
        p.grad = None
    out = group_forward_backward(h, mine, depth=depth)
    got = {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in tr.items()}
    for p in tr.values():
        p.grad = None
    ref_losses = []
    for r in raw:
        feature = d3.UnitFeatureKD(container=qvl.container, teacher=r["teacher"])
        loss, _, _ = d3.full_depth_forward_backward(h, r["segment"], d3.d3d_head_loss(h, r["targets"], denominator=float(r["count"])),
                                                     loss_scale=0.5, feature=feature)
        ref_losses.append(loss)
    worst, compared = 0.0, 0
    for name, p in tr.items():
        a, b = got[name], p.grad
        if b is None or float(b.norm()) == 0.0:
            continue
        compared += 1
        worst = max(worst, float((a - b).norm() / b.norm()) if a is not None else float("inf"))
    for p in tr.values():
        p.grad = None
    return {"worst_rel": worst, "compared": compared, "loss_gap": max(abs(x - y) for x, y in zip(out["losses"], ref_losses))}


def command_vision_check(args: argparse.Namespace) -> int:
    """The D3d data path on real files, with a light stand-in for the hybrid (bands, placeholder, layout from the
    child's own manifest): every bound row's student sequence decodes to the corpus answer, the patch rows match the
    teacher shard, and the answer KD targets pass d3's checks."""

    from types import SimpleNamespace
    from chimera_omni.config import ChimeraConfig

    spec = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    entry = next(e for e in spec["routes"] if e.get("kind") == "vision_answer")
    index = VisionIndex(entry)
    manifest = d3.read_manifest(Path(args.omni1))
    cfg = ChimeraConfig.from_manifest(d3.p1_module()._chimera_config(Path(args.omni1))["config"])  # as HybridD3 reads it
    bands = d3.interface_bands(manifest)
    placement = next(p for p in (manifest.get("blocks") or {}).get("placements", []) if "qvl" in str(p.get("container")))
    stand = SimpleNamespace(media_placeholder=int(cfg.ledger().range("media").begin), bands=bands, cfg=cfg, base_vocab=248_320,
                            layout=[SimpleNamespace(source="qvl32", container=str(placement["container"]))], engine=None, T=d3.trainer_module())
    tok = student_tokenizer(str(args.parent_config))
    corpus = Path(entry["corpus"]).read_text(encoding="utf-8").splitlines()
    report = {"items": len(index.items), "skipped": index.skipped, "checked": []}
    for item in index.items[: int(args.rows)]:
        out = bind_vision(stand, index, item, tok)
        seg = out["segment"]
        rec = json.loads(corpus[int(item["row"])])
        ids = seg["ids"][0]
        tail = tok.decode([int(t) for t in ids[-(out["binding"]["answer_positions"] + 1):] if int(t) < 248_320])
        ok = rec["reference_response"].strip()[:40] in tail
        report["checked"].append({"row": item["row"], "patch_rows": out["binding"]["patch_rows"], "q38_image_tokens": out["binding"]["q38_image_tokens"],
                                  "answer_positions": out["binding"]["answer_positions"], "answer_decodes": ok})
    report["pass"] = all(c["answer_decodes"] for c in report["checked"]) and bool(report["checked"])
    print(json.dumps(report, indent=1), flush=True)
    return 0 if report["pass"] else 3


def music_parity(hybrid: Any, depth: int) -> dict[str, Any]:
    """D3c through the group step (full depth) vs d3's own `full_depth_forward_backward` with input grads, on the tiny
    child's music band: the head-row gradient and every input code row's gradient."""

    codes = 12
    music = MusicBand(hybrid, codes=codes)
    g = torch.Generator().manual_seed(33)
    samples, refs = [], []
    for k in range(2):
        prompt = torch.randint(0, d3.TINY.VOCAB, (1, 4), generator=g)
        seq = torch.randint(0, codes, (6,), generator=g)
        ids = torch.cat([prompt, (seq + hybrid.music_band_begin).view(1, -1)], dim=1)
        lp_full = torch.log_softmax(torch.randn(seq.numel(), codes, generator=g) * 2.0, -1)
        top_lp, top_ids = lp_full.topk(4, dim=-1)
        lse = torch.zeros(seq.numel())
        lp_band = d3.band_teacher_logprobs(top_lp, lse, lse)
        positions = torch.arange(3, 3 + seq.numel())
        samples.append({"name": f"tiny-codes-{k}", "segment": {"ids": ids}, "weight": 0.5,
                        "head_loss": music.head_loss(positions, top_ids, lp_band), "input_grad": music.input_grad(ids)})
        refs.append((ids, positions, top_ids, lp_band))
    music.head.grad = None
    music.row_grads = {}
    group_forward_backward(hybrid, samples, depth=depth)
    got_head = music.head.grad.detach().clone()
    got_rows = {r: v.detach().clone() for r, v in music.row_grads.items()}
    music.head.grad = None
    music.row_grads = {}
    head_ref = music.head.detach().clone().requires_grad_(True)
    want_rows: dict[int, torch.Tensor] = {}
    for ids, positions, top_ids, lp_band in refs:
        def head_loss(normed: torch.Tensor, context: dict[str, Any], positions=positions, top_ids=top_ids, lp_band=lp_band) -> torch.Tensor:
            return d3.band_kl(normed[0, positions], head_ref, top_ids, lp_band)
        _, grad_x0, _ = d3.full_depth_forward_backward(hybrid, {"ids": ids}, head_loss, loss_scale=0.5, input_grad=True)
        band = ids[0] >= hybrid.base_vocab
        for r, v in zip((ids[0][band] - hybrid.base_vocab).tolist(), grad_x0[0][band]):
            want_rows[r] = v.float() if r not in want_rows else want_rows[r] + v.float()
    for p in media_trainables(hybrid).values():
        p.grad = None
    head_rel = float((got_head - head_ref.grad).norm() / head_ref.grad.norm())
    embed_rel = max(float((got_rows[r] - v).norm() / v.norm().clamp_min(1e-30)) for r, v in want_rows.items()) if set(got_rows) == set(want_rows) else float("inf")
    hybrid.band_embed_override = None
    EXTRAS.pop("music", None)
    return {"head_rel": head_rel, "embed_rel": embed_rel, "rows": len(want_rows), "row_sets_equal": set(got_rows) == set(want_rows)}


def command_selftest(args: argparse.Namespace) -> int:
    import tempfile

    torch.manual_seed(0)
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "selftest", "tool": tool_identity(), "torch": torch.__version__}
    refusals: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        built = d3.build_tiny_child(Path(tmp) / "child")
        cfg = built["cfg"]
        hybrid = d3.HybridD3(built["dir"], built["parent_config"],
                             d3.HybridOptions(dtype=torch.float32, device="cpu", kernels="torch", lora=None, pin_boundaries=False))
        depth = hybrid.layout[0].depth
        trainables = media_trainables(hybrid)
        with torch.no_grad():  # a non-trivial head so every head gradient is exercised
            g = torch.Generator().manual_seed(4)
            for name, p in trainables.items():
                if name.startswith("native_media_head."):
                    p.add_(torch.randn(p.shape, generator=g) * 0.01)

        def samples_for(seeds: list[int]) -> list[dict[str, Any]]:
            out = []
            for k, seed in enumerate(seeds):
                kind = ("ace", "h3", "hy3d")[k % 3]
                seg = d3.tiny_flow_batch(cfg, kind, rows=6 + k, seed=seed)
                context = hybrid.prepare(seg)
                mask = torch.ones(1, seg["media_rows"])
                loss_fn = d3.make_flow_loss("guided_velocity", context, seg, seg["raw"], mask, hybrid.bands, target_kind="parent_raw",
                                            noisy_feature_bands=hybrid.noisy_feature_bands(), signs=seg["signs"])
                out.append({"name": f"tiny-{kind}-{seed}", "segment": seg, "loss_fn": loss_fn, "weight": 1.0 / len(seeds)})
            return out

        samples = samples_for([21, 22, 23, 24])
        for p in trainables.values():
            p.grad = None
        group = group_forward_backward(hybrid, samples, depth=depth)
        got = {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in trainables.items()}
        for p in trainables.values():
            p.grad = None
        ref_losses = reference_per_sample(hybrid, samples, depth=depth)
        want = {k: (None if p.grad is None else p.grad.detach().clone()) for k, p in trainables.items()}
        for p in trainables.values():
            p.grad = None
        worst, compared, zero, mismatch = 0.0, 0, 0, []
        for name in trainables:
            a, b = got[name], want[name]
            if (a is None) != (b is None):
                mismatch.append(name)
                continue
            if a is None:
                continue
            compared += 1
            if float(b.norm()) == 0.0:
                zero += 1
                continue
            worst = max(worst, float((a - b).norm() / b.norm()))
        loss_gap = max(abs(x - y) for x, y in zip(group["losses"], ref_losses))
        eval_losses = group_eval(hybrid, samples, depth=depth)
        eval_gap = max(abs(x - y) for x, y in zip(eval_losses, group["losses"]))
        receipt["eval_parity"] = {"gap": eval_gap}
        log(f"eval parity: forward-only held losses vs training losses, max gap {eval_gap:.2e}")
        if eval_gap > 1e-6:
            refusals.append(f"eval parity: gap {eval_gap:.2e}")
        for p in trainables.values():
            p.grad = None
        receipt["grad_parity"] = {"tensors": len(trainables), "compared": compared, "zero_reference": zero, "worst_rel": worst,
                                  "presence_mismatch": mismatch, "loss_gap": loss_gap, "threshold": 1e-5,
                                  "units": [p.container for p in hybrid.layout]}
        log(f"grad parity: worst rel {worst:.3e} over {compared} tensors ({zero} zero-reference), loss gap {loss_gap:.2e}, "
            f"presence mismatch {len(mismatch)}")
        if worst > 1e-5 or mismatch or loss_gap > 1e-6 or compared == 0:
            refusals.append(f"grad parity failed: worst {worst:.3e}, mismatch {mismatch[:4]}, loss gap {loss_gap:.2e}")
        # speech lanes (D3e): the group step with each sample's H3 prompt-only scope == d3's own speech_lane_step
        receipt["speech_parity"] = speech_parity(hybrid, cfg, trainables, depth)
        log("speech parity: " + ", ".join(f"{k} worst rel {v['worst_rel']:.3e} loss gap {v['loss_gap']:.2e}"
                                          for k, v in receipt["speech_parity"].items()))
        for lane, rep_ in receipt["speech_parity"].items():
            if rep_["worst_rel"] > 1e-5 or rep_["loss_gap"] > 1e-6 or rep_["compared"] == 0 or rep_["presence_mismatch"]:
                refusals.append(f"speech parity {lane} failed: {rep_}")
        receipt["vision_parity"] = vision_parity(Path(tmp))
        vp = receipt["vision_parity"]
        log(f"vision (D3d) parity: unit worst rel {vp['worst_rel']:.3e} over {vp['compared']} tensors, loss gap {vp['loss_gap']:.2e}")
        if vp["worst_rel"] > 1e-5 or vp["compared"] == 0 or vp["loss_gap"] > 1e-6:
            refusals.append(f"vision parity failed: {vp}")
        receipt["music_parity"] = music_parity(hybrid, depth)
        log(f"music (D3c) parity: head worst rel {receipt['music_parity']['head_rel']:.3e}, embed rows worst rel "
            f"{receipt['music_parity']['embed_rel']:.3e} over {receipt['music_parity']['rows']} rows")
        if receipt["music_parity"]["head_rel"] > 1e-5 or receipt["music_parity"]["embed_rel"] > 1e-5 or not receipt["music_parity"]["rows"]:
            refusals.append(f"music parity failed: {receipt['music_parity']}")
        # training: the group loss on a fixed set falls under the streamed Adafactor
        optimizer = fk.StreamedAdafactor(torch.device("cpu"), seed=5)
        fixed = samples_for([31, 32, 33])
        curve = []
        for step in range(1, int(args.steps) + 1):
            out = group_forward_backward(hybrid, fixed, depth=depth)
            curve.append(sum(out["losses"]) / len(out["losses"]))
            apply_updates(trainables, optimizer, float(args.lr), step)
        receipt["training"] = {"first": curve[0], "last": curve[-1], "steps": len(curve), "lr": float(args.lr)}
        log(f"training: group loss {curve[0]:.4g} -> {curve[-1]:.4g} in {len(curve)} steps")
        if not curve[-1] < curve[0] * float(args.decrease):
            refusals.append(f"training: loss {curve[0]:.4g} -> {curve[-1]:.4g} (needs < x{args.decrease})")
    receipt.update({"refusals": refusals, "verdict": "PASS" if not refusals else "FAIL"})
    sha = write_json_atomic(Path(args.receipt), receipt)
    log(f"MEDIA SELFTEST {receipt['verdict']}: {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


# --------------------------------------------------------------------------
# probe / train
# --------------------------------------------------------------------------


def run_groups(args: argparse.Namespace, *, train: bool) -> int:
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    spec = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    flows = FlowSet(spec)
    hybrid = build_hybrid(args)
    depth = sorted({p.after_text_layer for p in hybrid.layout})[0] + 1
    trainables = media_trainables(hybrid)
    optimizer = fk.StreamedAdafactor(device, seed=int(args.seed), clip=float(args.clip))
    EXTRAS["parent_config"] = str(args.parent_config)
    music = None
    if "music_codes" in flows.routes:
        music = MusicBand(hybrid, codes=d3.ACE_CODES)
        EXTRAS.update({"music": music, "tokenizer": str(args.parent_config),
                       "ace_node_defaults": json.loads(Path(spec["ace_node_defaults"]).read_text(encoding="utf-8"))
                       if isinstance(spec.get("ace_node_defaults"), str) else spec["ace_node_defaults"]})
        log(f"D3c music band: {music.codes:,} written code rows of the head and the embedding train (fp32)")

    def saved_tensors() -> dict[str, torch.Tensor]:
        return {**trainables, **(music.tensors() if music is not None else {})}

    sampler = GroupSampler(flows, seed=int(args.seed), held_per_route=int(args.held_per_route),
                           group_tokens=int(args.group_tokens), max_group=int(args.max_group))
    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "kind": "train" if train else "probe", "tool": tool_identity(),
                               "argv": sys.argv[1:], "torch": torch.__version__, "device": str(device), "dtype": args.dtype,
                               "hybrid": hybrid.identity(), "data": flows.identity(), "depth": depth,
                               "trainable_tensors": len(trainables),
                               "trainable_parameters": int(sum(p.numel() for p in saved_tensors().values())),
                               "trunk": "frozen at the base (stage M1: run 1 owns the trunk)",
                               "held": {r: [i["name"] for i in v] for r, v in sampler.held.items()},
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if device.type == "cuda":
        receipt["card"] = torch.cuda.get_device_name(device)
    log(f"media M1: {receipt['trainable_parameters']:,} trainable parameters in {len(trainables)} tensors; depth {depth}; "
        f"routes {flows.identity()}")
    step = 0
    history: list[dict[str, Any]] = []
    total = int(args.steps)
    resumed_state = None
    if train and args.resume:
        found = latest_checkpoint(ckpt_dir)
        if found is not None:
            holders = saved_tensors()
            state = load_checkpoint(found, holders, optimizer)
            if music is not None:
                music.check_frozen()
            sampler.load_state(state["sampler"])
            resumed_state = state
            step = int(state["step"])
            history = list(state.get("history_tail", []))
            log(f"resumed from {found} at step {step}")
    last_save = time.time()
    refusals: list[str] = []
    media_trainables_cache[id(hybrid)] = list(trainables.values()) + ([music.head] if music is not None else [])
    evals: list[dict[str, Any]] = list((resumed_state or {}).get("evals", []))

    def run_eval(tag: str) -> None:
        if not train or not int(args.eval_every):
            return
        t_eval = time.time()
        res = evaluate_held(hybrid, sampler, depth=depth, verify=not args.no_verify, per_route=int(args.eval_items))
        record = {"step": step, "tag": tag, "routes": res, "seconds": round(time.time() - t_eval, 1)}
        evals.append(record)
        print("EVAL " + json.dumps(record, sort_keys=True), flush=True)

    if train and (step == 0 or not evals):
        run_eval("start")
    while step < total:
        if train and (out / "STOP").exists():
            state = {"step": step, "sampler": sampler.state(), "history_tail": history[-50:], "argv": sys.argv[1:],
                     "tool": tool_identity(), "data": flows.identity(), "evals": evals[-40:]}
            path = save_checkpoint(ckpt_dir, step, saved_tensors(), optimizer, state, keep=int(args.keep))
            log(f"STOP file present: checkpoint {path}, exit")
            break
        t0 = time.time()
        group = draw_group(sampler, hybrid, verify=not args.no_verify)
        bind_s = time.time() - t0
        load_before = hybrid.engine.load_seconds
        result = group_forward_backward(hybrid, group, depth=depth, offload=args.offload)
        lr = lr_at(step + 1, base=float(args.lr), warmup=int(args.warmup), total=total, min_ratio=float(args.min_lr_ratio)) if train else 0.0
        before = None
        if not train:
            name0 = next(iter(trainables))
            before = trainables[name0].detach().clone()
        stats = apply_updates(trainables, optimizer, lr, step + 1)
        if music is not None:
            music.apply(optimizer, lr, step + 1, stats)
            stats["grad_norm"] = math.sqrt(stats["grad_sq"])
        if before is not None and not torch.equal(before, trainables[next(iter(trainables))].detach()):
            refusals.append("an lr = 0 probe step changed a parameter")
        step += 1
        seconds = time.time() - t0
        tokens = sum(s["tokens"] for s in group)
        record = {"step": step, "route": group[0]["route"], "samples": len(group), "tokens": tokens,
                  "loss": sum(result["losses"]) / len(result["losses"]), "losses": result["losses"], "lr": lr,
                  "grad_norm": stats["grad_norm"], "updated": stats["updated"], "no_grad": len(stats["no_grad"]),
                  "nonfinite": stats["nonfinite"], "clipped": stats["clipped"], "seconds": round(seconds, 2),
                  "bind_seconds": round(bind_s, 2), "layer_load_seconds": round(hybrid.engine.load_seconds - load_before, 2),
                  "tok_per_s": round(tokens / max(seconds, 1e-9), 2)}
        if device.type == "cuda":
            record["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        record["host_rss_gb"] = hybrid.T.host_rss_gb()
        print("STEP " + json.dumps(record, sort_keys=True), flush=True)
        history.append(record)
        if stats["nonfinite"]:
            refusals.append(f"step {step}: non-finite gradient in {stats['nonfinite'][:3]}")
        if not train:
            receipt.setdefault("no_grad_by_route", {})[record["route"]] = stats["no_grad"]
        if train and int(args.eval_every) and step % int(args.eval_every) == 0:
            run_eval(f"step-{step}")
        if train and (time.time() - last_save >= 60 * float(args.save_minutes) or step >= total):
            state = {"step": step, "sampler": sampler.state(), "history_tail": history[-50:], "argv": sys.argv[1:],
                     "tool": tool_identity(), "data": flows.identity(), "evals": evals[-40:]}
            path = save_checkpoint(ckpt_dir, step, saved_tensors(), optimizer, state, keep=int(args.keep))
            last_save = time.time()
            log(f"checkpoint {path}")
    receipt["history"] = history[-200:]
    receipt["evals"] = evals
    receipt["skipped_items"] = sampler.skipped
    receipt["steps_done"] = step
    if train and step >= total:
        receipt["export"] = export_media(out / "export", saved_tensors(), {"base": hybrid.identity()["source"], "steps": step,
                                                                     "data": flows.identity(), "tool": tool_identity()})
        log(f"exported {receipt['export']['file']}")
    receipt.update({"refusals": refusals, "verdict": "PASS" if not refusals else "FAIL", "seconds": round(time.time() - started, 1)})
    sha = write_json_atomic(Path(args.receipt), receipt)
    log(f"MEDIA {'TRAIN' if train else 'PROBE'} {receipt['verdict']}: {args.receipt} sha256 {sha}")
    return 0 if not refusals else 3


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--receipt", required=True)
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--decrease", type=float, default=0.7)
    v = sub.add_parser("vision-check", help="build the D3d index and bind N real rows' sequences (no model, CPU)")
    v.add_argument("--omni1", required=True)
    v.add_argument("--parent-config", required=True)
    v.add_argument("--data-config", required=True)
    v.add_argument("--rows", type=int, default=12)
    for name in ("probe", "train"):
        p = sub.add_parser(name)
        p.add_argument("--omni1", required=True, help="the Chimera child directory (composite_manifest.json)")
        p.add_argument("--parent-config", required=True)
        p.add_argument("--plan", default=None, help="the P1 delta plan (OMNI1D1 = clone-pre-D1 + p1_plan.json)")
        p.add_argument("--data-config", required=True, help='{"routes": [{"route", "dirs": [...], "glob"?, "fraction"?}]}')
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
        p.add_argument("--kernels", default="auto", choices=("auto", "torch"))
        p.add_argument("--group-tokens", type=int, default=30_000)
        p.add_argument("--max-group", type=int, default=64)
        p.add_argument("--offload", default="cuda", choices=("cuda", "cpu"),
                       help="where the layer boundaries of a group live between forward and backward (cuda: no host RAM)")
        p.add_argument("--head-device", default=None,
                       help="device of the TEXT head (speech-in CE needs it; default cpu, use cuda:0 when speech_in is a route)")
        p.add_argument("--held-per-route", type=int, default=16)
        p.add_argument("--seed", type=int, default=20261018)
        p.add_argument("--lr", type=float, default=1e-5)
        p.add_argument("--warmup", type=int, default=20)
        p.add_argument("--min-lr-ratio", type=float, default=0.1)
        p.add_argument("--clip", type=float, default=1.0)
        p.add_argument("--steps", type=int, default=2 if name == "probe" else 2000)
        p.add_argument("--out", required=True)
        p.add_argument("--receipt", required=True)
        p.add_argument("--save-minutes", type=float, default=30.0)
        p.add_argument("--keep", type=int, default=2)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--no-verify", action="store_true", help="skip the per-batch tensor sha256 check")
        p.add_argument("--eval-every", type=int, default=25, help="held-item eval every N groups (0 = never)")
        p.add_argument("--eval-items", type=int, default=8, help="held items per route in each eval")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    log(f"{TOOL_NAME} {TOOL_VERSION} {args.command}")
    try:
        if args.command == "selftest":
            return command_selftest(args)
        if args.command == "vision-check":
            return command_vision_check(args)
        if args.command == "probe":
            return run_groups(args, train=False)
        if args.command == "train":
            return run_groups(args, train=True)
    except (MediaError, d3.D3Error) as exc:
        log(f"REFUSED: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
