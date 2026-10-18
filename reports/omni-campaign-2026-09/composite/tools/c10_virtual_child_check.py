#!/usr/bin/env python3
"""c10_virtual_child_check.py -- load / identity / forward check of a C10 union child that is
NOT written to disk: the trunk's tensors plus ``<out>.partial/L{k:02d}.safetensors`` from
``c10_union_install.py --stop-after-layer <last>``.

Why: on the AI server the written child (~54 GB) plus the partial it is built from (~33 GB)
cannot sit on root above the 60 GB floor while K3 is still writing, so the rehearsal stops
after the last layer (exit 3) and this tool checks the child the partial DEFINES:

  1. per-layer identity  -- each partial file hashes like its record (file + tensor bytes), the
     FFN dtype / shapes are the uniform width, the trunk's Qwen gate/up rows are carried
     verbatim (bit-exact), zero-pad rows are exactly zero where the pool was short;
  2. accounting          -- header parameter count of the virtual child = trunk headers minus
     the trunk FFN tensors plus the partial FFN tensors (oc_seal.safetensors_tensor_counts on the
     trunk), against c10's arithmetic, the parent record and the S7 media budget;
  3. structural load     -- ``ChimeraOmniModel(child_cfg)`` on the meta device names exactly the
     virtual state's tensors with exactly its shapes (what ``load_state_dict(strict=True)`` checks);
  4. streamed forward    -- layer-major, one block on --device at a time, every block loaded
     ``strict=True`` from trunk views (+ the partial FFN for the child), fp32 RoPE tables kept,
     text-only, all tokens valid; final norm, untied/tied head, ledger band masked.  The S3 trunk
     and the virtual child run on the same prompts in the same pass; the receipt records
     finiteness, top-1 agreement child vs trunk, KL(trunk || child), max |logit diff| and the
     per-layer residual relL2 (all DIAGNOSTICS; only non-finite values and identity/structure
     failures are refusals -- owner policy 2026-10-06).

The test (test_c10_virtual_child_check.py) proves on a synthetic Chimera that this virtual child's
logits equal the child c10 writes on --resume, loaded with ``load_chimera_checkpoint``, and that
the accounting equals oc_seal on the written child.

Prompts: ``--prompt-ids`` JSON (lists of ids), else the 8 R-Q prompts
(``<tree>/scripts/import_receipt_qwen35.PROMPT_PACK``) tokenized with ``--tokenizer-dir``.

1.2.0 (2026-10-11, the C10 design sweep):
  * a partial file may carry ``down`` only, its record naming the sweep's SHARED up file and that
    file's sha (``c10_union_install.py --sweep-targets``); the shared up is verified once per file
    (file sha, tensor sha against every record, finite, Qwen rows verbatim, zero pad) and reused.
  * ``--variant NAME=DIR`` (repeatable): several virtual children (sweep variants) checked for
    identity / accounting / structure each, then evaluated in ONE layer-major pass beside the trunk,
    partial tensors read lazily per layer (``LazyTensor``; the shared up is read once per layer).
  * ``--reference NAME=DIR``: a full Chimera checkpoint (the S0 import of the 27B parent) run in the
    same pass from lazily read views; children are scored against it as well as against the trunk.
  * ``--evidence MANIFEST``: the manifest's HELD rows are added as teacher-forced sequences; per
    model the mean next-token NLL over the base vocab, top-1 agreement with the trunk and with the
    reference, KL(trunk || child).  Per layer and per child: residual relL2 vs the trunk and the
    residual norm ratio ||x_child|| / ||x_trunk|| (held rows and prompts reported separately).
  All of it DIAGNOSTICS; refusals stay identity / structure / non-finite only.
  The single ``--partial`` path without these flags is the 1.1.1 computation, unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c10_union_install as c10  # noqa: E402  (binds CHIMERA_SRC / lanes exactly as c10 does)
import oc_seal  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.importing.safetensors_io import (  # noqa: E402
    _torch_dtype,
    read_checkpoint_config,
    read_index,
    read_safetensors,
)
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402
from chimera_omni.modeling.layers import ChimeraBlock  # noqa: E402

TOOL_NAME = "c10_virtual_child_check"
TOOL_VERSION = "1.2.0"   # 1.1.0: --pad-missing-with-trunk (labelled hybrid); 1.1.1: dangling symlinks are not layers; 1.2.0: sweep variants, --reference, --evidence held NLL
FFN_NAMES = ("ffn.up.weight", "ffn.down.weight")


class Refusal(SystemExit):
    pass


class LazyTensor:
    """A partial tensor read on demand.  ``shape`` / ``dtype`` / ``numel()`` serve the accounting and
    the structural check without bytes; ``load()`` reads it.  A SHARED up (one file for every sweep
    variant of a layer) is kept in a one-entry class cache, so a layer's variants read it once."""

    _shared: tuple[tuple[str, str], torch.Tensor] | None = None

    def __init__(self, path: str, key: str, shape: tuple[int, ...], dtype: torch.dtype, shared: bool = False) -> None:
        self.path, self.key, self.shape, self.dtype, self.shared = path, key, tuple(shape), dtype, shared

    def numel(self) -> int:
        return int(math.prod(self.shape))

    def load(self) -> torch.Tensor:
        from safetensors import safe_open

        ident = (self.path, self.key)
        if self.shared and LazyTensor._shared is not None and LazyTensor._shared[0] == ident:
            return LazyTensor._shared[1]
        with safe_open(self.path, "pt") as handle:
            tensor = handle.get_tensor(self.key)
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.dtype:
            raise Refusal(f"{self.path}[{self.key}] changed since the identity check")
        if self.shared:
            LazyTensor._shared = (ident, tensor)
        return tensor


def materialize(t: Any) -> torch.Tensor:
    return t.load() if isinstance(t, LazyTensor) else t


#: verified shared-up files: (path, recorded file sha) -> facts (checked once, reused by every variant)
_SHARED_UP_FACTS: dict[tuple[str, str, int, int], dict[str, Any]] = {}   # + (size, mtime_ns): an edited file re-verifies


def log(message: str) -> None:
    print(f"[{TOOL_NAME}] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# 1. per-layer identity
# --------------------------------------------------------------------------
def tensor_bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and c10.sha256_tensor_bytes(a) == c10.sha256_tensor_bytes(b)


def trunk_padded_layer(trunk_views: dict[str, Any], k: int, q: int, h: int, width: int,
                       ffn_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The trunk's own FFN at the union width: gate/up rows + zero rows, down + zero columns.
    Function-preserving exactly (silu(0) * 0 = 0 and the zero columns read nothing)."""
    up_t = trunk_views[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=ffn_dtype)
    down_t = trunk_views[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=ffn_dtype)
    pad = width - q
    zeros_rows = torch.zeros(pad, h, dtype=ffn_dtype)
    up = torch.cat([up_t[:q], zeros_rows, up_t[q:], zeros_rows]).contiguous()
    down = torch.cat([down_t, torch.zeros(h, pad, dtype=ffn_dtype)], dim=1).contiguous()
    return {"ffn.up.weight": up, "ffn.down.weight": down}


def _up_facts(up: torch.Tensor, trunk_views: dict[str, Any], k: int, q: int, w: int, k3_rows: int, pad: int,
              ffn_dtype: torch.dtype) -> dict[str, Any]:
    trunk_up = trunk_views[f"blocks.{k}.ffn.up.weight"].torch(device="cpu", dtype=ffn_dtype)
    facts = {"dtype": up.dtype, "shape": tuple(up.shape), "tensor_sha256": c10.sha256_tensor_bytes(up),
             "finite": bool(torch.isfinite(up.float()).all()),
             "gate_verbatim": tensor_bytes_equal(up[:q].contiguous(), trunk_up[:q].contiguous()),
             "up_verbatim": tensor_bytes_equal(up[w:w + q].contiguous(), trunk_up[q:].contiguous()),
             "pad_zero": True}
    if pad:
        lo = q + k3_rows
        facts["pad_zero"] = bool((up[lo:w] == 0).all()) and bool((up[w + lo:] == 0).all())
    return facts


def check_partial(partial_dir: Path, trunk_views: dict[str, Any], n_layers: int, q: int, h: int,
                  ffn_dtype: torch.dtype, pad_missing: bool = False, lazy: bool = False
                  ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], int]:
    from safetensors.torch import load_file

    layers: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    width = None
    prefix_end = None
    for k in range(n_layers):
        tensors_path = partial_dir / f"L{k:02d}.safetensors"
        record_path = partial_dir / f"L{k:02d}.json"
        if not tensors_path.is_file() or not record_path.is_file():
            if not pad_missing:
                raise Refusal(f"{partial_dir}: layer {k} is not persisted ({tensors_path.name} / {record_path.name}); "
                              "the virtual child needs every layer (or --pad-missing-with-trunk for a labelled hybrid)")
            if k == 0:
                raise Refusal(f"{partial_dir}: no persisted layer; nothing to check")
            prefix_end = k
            break
        record = json.loads(record_path.read_text(encoding="utf-8"))
        file_sha = c10.sha256_file(str(tensors_path))
        if file_sha != (record.get("partial") or {}).get("tensors_sha256"):
            raise Refusal(f"{tensors_path}: sha256 {file_sha} != its record's {record.get('partial')}")
        got = load_file(str(tensors_path))
        down = got["down"]
        inst = record["installed"]
        w = int(inst["intermediate_size"])
        width = w if width is None else width
        if w != width:
            raise Refusal(f"layer {k}: intermediate_size {w} != {width} (the child must be uniform)")
        k3_rows = int(inst["k3_rows"])
        pad = int(inst["zero_pad_rows"])
        if q + k3_rows + pad != w:
            raise Refusal(f"layer {k}: qwen {q} + k3 {k3_rows} + pad {pad} != width {w}")
        up_source = str(tensors_path)
        shared_up = False
        if "up" in got:
            up = got["up"]
            facts = _up_facts(up, trunk_views, k, q, w, k3_rows, pad, ffn_dtype)
        else:   # 1.2.0: a sweep variant -- down only, the record names the shared up file and its sha
            part = record.get("partial") or {}
            up_path, up_file_sha = part.get("up_tensors"), part.get("up_tensors_sha256")
            if not up_path or not up_file_sha:
                raise Refusal(f"{tensors_path}: no 'up' tensor and the record names no shared up_tensors / sha")
            stat = os.stat(str(up_path))
            key = (str(up_path), str(up_file_sha), int(stat.st_size), int(stat.st_mtime_ns))
            facts = _SHARED_UP_FACTS.get(key)
            up = None
            if facts is None:
                actual = c10.sha256_file(str(up_path))
                if actual != up_file_sha:
                    raise Refusal(f"{up_path}: sha256 {actual} != {up_file_sha} named by {record_path}")
                up = load_file(str(up_path))["up"]
                facts = _up_facts(up, trunk_views, k, q, w, k3_rows, pad, ffn_dtype)
                _SHARED_UP_FACTS[key] = facts
            elif not lazy:
                up = load_file(str(up_path))["up"]
            up_source, shared_up = str(up_path), True
        if facts["dtype"] != ffn_dtype or down.dtype != ffn_dtype:
            raise Refusal(f"layer {k}: partial dtypes {facts['dtype']}/{down.dtype} != the trunk FFN dtype {ffn_dtype}")
        if facts["shape"] != (2 * w, h) or tuple(down.shape) != (h, w):
            raise Refusal(f"layer {k}: shapes {facts['shape']}/{tuple(down.shape)} != [{2 * w},{h}]/[{h},{w}]")
        if facts["tensor_sha256"] != inst["up_sha256"] or c10.sha256_tensor_bytes(down) != inst["down_sha256"]:
            raise Refusal(f"layer {k}: tensor bytes do not hash like the record")
        if not facts["finite"]:
            raise Refusal(f"layer {k}: partial up carries non-finite values")
        if not bool(torch.isfinite(down.float()).all()):
            raise Refusal(f"layer {k}: partial down carries non-finite values")
        gate_verbatim, up_verbatim = facts["gate_verbatim"], facts["up_verbatim"]
        if not (gate_verbatim and up_verbatim):
            raise Refusal(f"layer {k}: the trunk's Qwen rows are not carried verbatim (gate {gate_verbatim}, "
                          f"up {up_verbatim})")
        trunk_down = trunk_views[f"blocks.{k}.ffn.down.weight"].torch(device="cpu", dtype=ffn_dtype)
        pad_zero = True
        if pad:
            lo = q + k3_rows
            pad_zero = facts["pad_zero"] and bool((down[:, lo:] == 0).all())
            if not pad_zero:
                raise Refusal(f"layer {k}: {pad} zero-pad rows are not exactly zero")
        qwen_down_rel = c10.relative_l2(down[:, :q].float(), trunk_down.float())
        if lazy:
            layers[k] = {"ffn.up.weight": LazyTensor(up_source, "up", facts["shape"], facts["dtype"], shared=shared_up),
                         "ffn.down.weight": LazyTensor(str(tensors_path), "down", tuple(down.shape), down.dtype)}
            del down, up
        else:
            layers[k] = {"ffn.up.weight": up, "ffn.down.weight": down}
        rows.append({
            "layer": k, "k3_layers": record.get("k3_layers"), "k3_rows": k3_rows, "zero_pad_rows": pad,
            "file_sha256": file_sha, "qwen_gate_rows_verbatim": gate_verbatim, "qwen_up_rows_verbatim": up_verbatim,
            "zero_pad_exact": pad_zero, "shared_up": shared_up, "qwen_down_columns_relL2_vs_trunk_down": qwen_down_rel,
            "heldout_relL2": record.get("heldout_relL2"),
            "k3_term_energy_fraction_held": (record.get("k3_term_energy_fraction") or {}).get("held"),
            "veto_binding": record.get("veto_binding"), "veto_reason": record.get("veto_reason"),
            "update_scale": record.get("update_scale"),
            "union_target": (record.get("union_target") or {}).get("spec"),
            "union_target_w": (record.get("union_target") or {}).get("w"),
            "k3_columns_write_fraction_held": record.get("k3_columns_write_fraction_held"),
            "trunk_only_held_relL2": (((record.get("scores") or {}).get("held") or {}).get("trunk_only") or {}).get("relL2"),
            "seconds": record.get("seconds"), "peak_rss_gigabytes": record.get("peak_rss_gigabytes"),
        })
        del trunk_down
    assert width is not None
    if prefix_end is not None:
        # a layer counts as persisted only when its file resolves (a dangling overflow symlink is not a layer)
        later = [f"L{j:02d}" for j in range(prefix_end, n_layers)
                 if (partial_dir / f"L{j:02d}.safetensors").is_file() or (partial_dir / f"L{j:02d}.json").is_file()]
        if later:
            raise Refusal(f"{partial_dir}: persisted layers after the gap at {prefix_end}: {sorted(later)}")
        for k in range(prefix_end, n_layers):
            layers[k] = trunk_padded_layer(trunk_views, k, q, h, width, ffn_dtype)
            rows.append({"layer": k, "source": "TRUNK FFN zero-padded to the union width -- not a c10 layer",
                         "k3_rows": 0, "zero_pad_rows": width - q, "veto_binding": None})
    return layers, rows, width


# --------------------------------------------------------------------------
# 2. accounting  /  3. structural load
# --------------------------------------------------------------------------
def account(trunk_dir: Path, layers: dict[int, dict[str, torch.Tensor]], n_layers: int, h: int, k3_channels: int,
            parent_count: int | None, media_budget: int | None) -> dict[str, Any]:
    counts = oc_seal.safetensors_tensor_counts(trunk_dir)
    trunk_total = int(sum(counts.values()))
    trunk_ffn = sum(counts[f"blocks.{k}.{n}"] for k in range(n_layers) for n in FFN_NAMES)
    child_ffn = sum(int(t.numel()) for k in layers for t in layers[k].values())
    child_total = trunk_total - trunk_ffn + child_ffn
    projected = trunk_total + n_layers * 3 * h * k3_channels
    doc: dict[str, Any] = {
        "trunk_header_parameters": trunk_total, "trunk_ffn_parameters": trunk_ffn,
        "child_ffn_parameters": child_ffn, "child_header_parameters": child_total,
        "c10_arithmetic_projection": projected, "equals_c10_arithmetic": child_total == projected,
        "method": "oc_seal.safetensors_tensor_counts(trunk) - trunk blocks.*.ffn.{up,down} + partial up/down numel",
    }
    if parent_count is not None:
        doc["parent_parameter_count"] = parent_count
        doc["headroom_vs_parent"] = parent_count - child_total
        doc["below_parent"] = child_total < parent_count
        if media_budget is not None:
            doc["s7_media_units_parameters"] = media_budget
            doc["headroom_after_media_units"] = parent_count - child_total - media_budget
            doc["fits_with_media_units"] = child_total + media_budget < parent_count
            doc["max_k3_channels_with_media_units"] = (parent_count - 1 - media_budget - trunk_total) // (n_layers * 3 * h)
    if child_total != projected:
        raise Refusal(f"virtual child counts {child_total:,} header parameters, c10's arithmetic says {projected:,}")
    return doc


def child_config(trunk_cfg: ChimeraConfig, width: int, k3_channels: int, q: int) -> ChimeraConfig:
    """Exactly what c10 main() writes (dataclasses.replace + validate)."""
    cfg = dataclasses.replace(
        trunk_cfg, intermediate_size=width, intermediate_sizes=tuple([width] * trunk_cfg.num_layers),
        metadata={**trunk_cfg.metadata, "c10_union_install": f"{c10.TOOL_NAME} {c10.TOOL_VERSION}",
                  "c10_k3_channels": str(k3_channels), "c10_trunk_intermediate_size": str(q)})
    cfg.validate()
    return cfg


def structural_load(child_cfg: ChimeraConfig, trunk_views: dict[str, Any], layers: dict[int, dict[str, torch.Tensor]],
                    tied: dict[str, str]) -> dict[str, Any]:
    with torch.device("meta"):
        model = ChimeraOmniModel(child_cfg)
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    virtual = {name: tuple(v.shape) for name, v in trunk_views.items()}
    for k, ffn in layers.items():
        for local, t in ffn.items():
            virtual[f"blocks.{k}.{local}"] = tuple(t.shape)
    for alias, target in tied.items():          # a tied head is stored once
        if alias not in virtual and target in virtual:
            virtual[alias] = virtual[target]
    missing = sorted(set(expected) - set(virtual))
    unexpected = sorted(set(virtual) - set(expected))
    mismatched = sorted(n for n in set(expected) & set(virtual) if expected[n] != virtual[n])
    del model
    ok = not (missing or unexpected or mismatched)
    doc = {"state_tensors": len(expected), "missing": missing[:20], "unexpected": unexpected[:20],
           "shape_mismatched": mismatched[:20], "strict_load_equivalent": ok,
           "rule": "ChimeraOmniModel(child_cfg) on meta: names and shapes == trunk headers with the partial FFNs swapped in"}
    if not ok:
        raise Refusal(f"structural load fails: missing {missing[:5]}, unexpected {unexpected[:5]}, shapes {mismatched[:5]}")
    return doc


# --------------------------------------------------------------------------
# 4. streamed forward (layer-major)
# --------------------------------------------------------------------------
def build_block(cfg: ChimeraConfig, k: int, views: dict[str, Any], device: torch.device, dtype: torch.dtype,
                ffn: dict[str, torch.Tensor] | None) -> ChimeraBlock:
    block = ChimeraBlock(cfg, k)
    rope = getattr(getattr(block, "mixer", None), "inv_freq", None)
    block = block.to(device=device, dtype=dtype)
    if rope is not None:
        block.mixer.inv_freq = rope.to(device=device, dtype=torch.float32)
    state: dict[str, torch.Tensor] = {}
    for name, reference in block.state_dict().items():
        if ffn is not None and name in FFN_NAMES:
            state[name] = materialize(ffn[name]).to(device=device, dtype=dtype)
            continue
        state[name] = views[f"blocks.{k}.{name}"].torch(device=str(device),
                                                       dtype=dtype if reference.is_floating_point() else reference.dtype)
    block.load_state_dict(state, strict=True)
    return block.eval()


def run_block(block: ChimeraBlock, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    x = block(x, native_media_slot_mask=None, valid_media_mask=None, enable_native_media_refinement=False,
              token_valid_mask=valid)
    return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


def streamed_logits(trunk_cfg: ChimeraConfig, child_cfg: ChimeraConfig, views: dict[str, Any],
                    layers: dict[int, dict[str, torch.Tensor]], tied: dict[str, str], packs: list[list[int]],
                    device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    from torch import nn

    started = time.time()
    h = int(trunk_cfg.hidden_size)
    embed = views["embed.weight"].torch(device=str(device), dtype=dtype)
    with torch.device("meta"):
        meta = ChimeraOmniModel(trunk_cfg)
    norm = type(meta.norm)(h, trunk_cfg.rms_norm_eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(views["norm.weight"].torch(device=str(device), dtype=dtype))
    if "lm_head.weight" in views:
        head_w = views["lm_head.weight"].torch(device=str(device), dtype=dtype)
        head_source = "lm_head.weight"
    elif trunk_cfg.tie_word_embeddings and tied.get("lm_head.weight") == "embed.weight":
        head_w = embed
        head_source = "embed.weight (tied)"
    else:
        raise Refusal("no lm_head.weight and no recorded tie")
    del meta
    xs_t, xs_c, valids = [], [], []
    with torch.no_grad():
        for ids in packs:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            valid = torch.ones(t.shape, dtype=torch.bool, device=device)
            x = F.embedding(t, embed)
            x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
            xs_t.append(x)
            xs_c.append(x.clone())
            valids.append(valid)
        per_layer = []
        for k in range(int(trunk_cfg.num_layers)):
            blk = build_block(trunk_cfg, k, views, device, dtype, None)
            xs_t = [run_block(blk, x, v) for x, v in zip(xs_t, valids)]
            del blk
            blk = build_block(child_cfg, k, views, device, dtype, layers[k])
            xs_c = [run_block(blk, x, v) for x, v in zip(xs_c, valids)]
            del blk
            num = sum(float((c.float() - t.float()).pow(2).sum()) for c, t in zip(xs_c, xs_t))
            den = sum(float(t.float().pow(2).sum()) for t in xs_t)
            finite = all(bool(torch.isfinite(c).all()) for c in xs_c)
            per_layer.append({"layer": k, "residual_relL2_child_vs_trunk": math.sqrt(num / den) if den > 0 else None,
                              "child_residual_finite": finite})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        logits_t, logits_c = [], []
        for xt, xc, valid in zip(xs_t, xs_c, valids):
            for x, sink in ((xt, logits_t), (xc, logits_c)):
                hidden = norm(x)
                hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
                sink.append(F.linear(hidden, head_w).float().cpu()[0])     # == nn.Linear lm_head
    facts = {"engine": "layer-major streamed: one ChimeraBlock on --device, load_state_dict(strict=True) per block, "
                       "fp32 RoPE tables kept; trunk and child in the same pass",
             "head": head_source, "seconds": round(time.time() - started, 2), "per_layer": per_layer}
    return logits_t, logits_c, facts


def compare(logits_t: list[torch.Tensor], logits_c: list[torch.Tensor], band_begin: int) -> dict[str, Any]:
    prompts = []
    agree_total, pos_total = 0, 0
    kl_sum = 0.0
    finite_all = True
    for i, (lt, lc) in enumerate(zip(logits_t, logits_c)):
        bt, bc = lt[:, :band_begin], lc[:, :band_begin]      # the ledger band is masked (-inf) in the model
        finite_c = bool(torch.isfinite(bc).all())
        finite_t = bool(torch.isfinite(bt).all())
        finite_all = finite_all and finite_c and finite_t
        top_t, top_c = bt.argmax(-1), bc.argmax(-1)
        agree = int((top_t == top_c).sum())
        lp_t, lp_c = F.log_softmax(bt.double(), -1), F.log_softmax(bc.double(), -1)
        kl = (lp_t.exp() * (lp_t - lp_c)).sum(-1)
        n = int(bt.shape[0])
        agree_total += agree
        pos_total += n
        kl_sum += float(kl.sum())
        prompts.append({"index": i, "positions": n, "child_logits_finite": finite_c, "trunk_logits_finite": finite_t,
                        "top1_agree": agree, "top1_agreement": agree / n,
                        "last_position_top1": {"trunk": int(top_t[-1]), "child": int(top_c[-1])},
                        "kl_trunk_child_mean": float(kl.mean()), "kl_trunk_child_max": float(kl.max()),
                        "max_abs_logit_diff": float((bc - bt).abs().max())})
    return {"prompts": prompts, "positions": pos_total, "top1_agree": agree_total,
            "top1_agreement": agree_total / max(pos_total, 1), "kl_trunk_child_mean": kl_sum / max(pos_total, 1),
            "all_logits_finite": finite_all, "band_begin": band_begin,
            "note": "diagnostics only (owner policy 2026-10-06); top-1 over the base vocab [0, band_begin)"}


def _final(cfg: ChimeraConfig, views: dict[str, Any], tied: dict[str, str], device: torch.device,
           dtype: torch.dtype) -> tuple[torch.Tensor, Any, torch.Tensor, str]:
    """(embed, final norm, head weight, head source) of one checkpoint, as ``streamed_logits`` builds them."""
    h = int(cfg.hidden_size)
    embed = views["embed.weight"].torch(device=str(device), dtype=dtype)
    with torch.device("meta"):
        meta = ChimeraOmniModel(cfg)
    norm = type(meta.norm)(h, cfg.rms_norm_eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(views["norm.weight"].torch(device=str(device), dtype=dtype))
    del meta
    if "lm_head.weight" in views:
        return embed, norm, views["lm_head.weight"].torch(device=str(device), dtype=dtype), "lm_head.weight"
    if cfg.tie_word_embeddings and tied.get("lm_head.weight") == "embed.weight":
        return embed, norm, embed, "embed.weight (tied)"
    raise Refusal("no lm_head.weight and no recorded tie")


def streamed_multi_eval(trunk_cfg: ChimeraConfig, views: dict[str, Any], tied: dict[str, str],
                        children: dict[str, tuple[ChimeraConfig, dict[int, dict[str, Any]]]],
                        reference: tuple[str, ChimeraConfig, dict[str, Any], dict[str, str]] | None,
                        prompts: list[list[int]], held: list[list[int]], device: torch.device, dtype: torch.dtype,
                        band_begin: int, position_chunk: int = 512) -> dict[str, Any]:
    """Layer-major pass of the trunk, the reference (optional) and every child over the R-Q prompts
    and the held evidence rows; the same block / norm / head ops as ``streamed_logits``, one model's
    block on --device at a time.  Returns per-model metrics (DIAGNOSTICS)."""
    started = time.time()
    seqs = [("prompts", ids) for ids in prompts] + [("held", ids) for ids in held]
    kinds = ("prompts", "held")
    t_embed, t_norm, t_head, t_head_src = _final(trunk_cfg, views, tied, device, dtype)
    models: dict[str, list[torch.Tensor]] = {}
    valids = []
    with torch.no_grad():
        base = []
        for _kind, ids in seqs:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            valid = torch.ones(t.shape, dtype=torch.bool, device=device)
            x = F.embedding(t, t_embed)
            base.append(torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)))
            valids.append(valid)
        models["trunk"] = base
        for name in children:
            models[name] = [x.clone() for x in base]
        r_final = None
        if reference is not None:
            r_name, r_cfg, r_views, r_tied = reference
            r_final = _final(r_cfg, r_views, r_tied, device, dtype)
            ref_x = []
            for (_kind, ids), valid in zip(seqs, valids):
                x = F.embedding(torch.tensor([ids], dtype=torch.long, device=device), r_final[0])
                ref_x.append(torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)))
            models["__reference__"] = ref_x
        per_layer: dict[str, list[dict[str, Any]]] = {n: [] for n in models if n != "trunk"}
        for k in range(int(trunk_cfg.num_layers)):
            blk = build_block(trunk_cfg, k, views, device, dtype, None)
            models["trunk"] = [run_block(blk, x, v) for x, v in zip(models["trunk"], valids)]
            del blk
            if reference is not None:
                blk = build_block(reference[1], k, reference[2], device, dtype, None)
                models["__reference__"] = [run_block(blk, x, v) for x, v in zip(models["__reference__"], valids)]
                del blk
            for name, (ccfg, layers) in children.items():
                blk = build_block(ccfg, k, views, device, dtype, layers[k])
                models[name] = [run_block(blk, x, v) for x, v in zip(models[name], valids)]
                del blk
            for name in per_layer:
                row: dict[str, Any] = {"layer": k}
                finite = True
                for kind in kinds:
                    num = den = cn = 0.0
                    for (s_kind, _ids), xc, xt in zip(seqs, models[name], models["trunk"]):
                        if s_kind != kind:
                            continue
                        diff = xc.float() - xt.float()
                        num += float(diff.pow(2).sum())
                        den += float(xt.float().pow(2).sum())
                        cn += float(xc.float().pow(2).sum())
                        finite = finite and bool(torch.isfinite(xc).all())
                    row[f"{kind}_residual_relL2_vs_trunk"] = math.sqrt(num / den) if den > 0 else None
                    row[f"{kind}_residual_norm_ratio_vs_trunk"] = math.sqrt(cn / den) if den > 0 else None
                row["residual_finite"] = finite
                per_layer[name].append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log(f"  eval layer {k:02d} done ({round(time.time() - started)} s)")

        # -- heads and metrics -------------------------------------------------------------
        def logits_of(name: str, i: int, lo: int, hi: int) -> torch.Tensor:
            x = models[name][i][:, lo:hi]
            if name == "__reference__":
                assert r_final is not None
                hidden = r_final[1](x)
                return F.linear(hidden, r_final[2]).float()[0][:, :band_begin]
            hidden = t_norm(x)
            return F.linear(hidden, t_head).float()[0][:, :band_begin]

        names = [n for n in models if n != "trunk" and n != "__reference__"]
        has_ref = reference is not None
        acc: dict[str, dict[str, dict[str, float]]] = {}
        for n in ["trunk"] + (["__reference__"] if has_ref else []) + names:
            acc[n] = {kind: {"positions": 0, "nll_sum": 0.0, "nll_positions": 0, "agree_trunk": 0, "agree_ref": 0,
                             "kl_trunk_sum": 0.0, "kl_ref_sum": 0.0, "finite": True} for kind in kinds}
        for i, (kind, ids) in enumerate(seqs):
            length = len(ids)
            targets = torch.tensor(ids, dtype=torch.long, device=device)
            for lo in range(0, length, position_chunk):
                hi = min(lo + position_chunk, length)
                lp_t = F.log_softmax(logits_of("trunk", i, lo, hi), -1)
                top_t = lp_t.argmax(-1)
                lp_r = F.log_softmax(logits_of("__reference__", i, lo, hi), -1) if has_ref else None
                top_r = lp_r.argmax(-1) if lp_r is not None else None
                n_next = max(0, min(hi, length - 1) - lo)       # positions with a next token
                nxt = targets[lo + 1:lo + 1 + n_next]
                for n in acc:
                    if n == "trunk":
                        lp = lp_t
                    elif n == "__reference__":
                        lp = lp_r
                    else:
                        lp = F.log_softmax(logits_of(n, i, lo, hi), -1)
                    a = acc[n][kind]
                    a["finite"] = a["finite"] and bool(torch.isfinite(lp).all())
                    top = lp.argmax(-1)
                    a["positions"] += int(hi - lo)
                    a["agree_trunk"] += int((top == top_t).sum())
                    a["kl_trunk_sum"] += float((lp_t.exp() * (lp_t - lp)).sum(-1).double().sum())
                    if has_ref:
                        a["agree_ref"] += int((top == top_r).sum())
                        a["kl_ref_sum"] += float((lp_r.exp() * (lp_r - lp)).sum(-1).double().sum())
                    if n_next:
                        valid_t = nxt < band_begin
                        picked = lp[:n_next].gather(1, nxt.clamp_max(band_begin - 1).unsqueeze(1)).squeeze(1)
                        a["nll_sum"] += float((-picked[valid_t]).double().sum())
                        a["nll_positions"] += int(valid_t.sum())
                    if lp is not lp_t and lp is not lp_r:
                        del lp
                del lp_t, lp_r
        result: dict[str, Any] = {}
        for n, per_kind in acc.items():
            label = reference[0] if (has_ref and n == "__reference__") else n
            out = {}
            for kind, a in per_kind.items():
                pos = max(a["positions"], 1)
                out[kind] = {"positions": a["positions"], "logits_finite": a["finite"],
                             "top1_agreement_vs_trunk": a["agree_trunk"] / pos, "top1_agree_vs_trunk": a["agree_trunk"],
                             "kl_trunk_child_mean": a["kl_trunk_sum"] / pos,
                             "nll_mean": (a["nll_sum"] / a["nll_positions"]) if a["nll_positions"] else None,
                             "nll_positions": a["nll_positions"]}
                if has_ref:
                    out[kind].update({"top1_agreement_vs_reference": a["agree_ref"] / pos,
                                      "top1_agree_vs_reference": a["agree_ref"],
                                      "kl_reference_model_mean": a["kl_ref_sum"] / pos})
            if n in per_layer:
                out["per_layer"] = per_layer[n]
            result[label] = out
    return {"models": result, "reference": reference[0] if reference is not None else None,
            "band_begin": band_begin, "head": t_head_src,
            "sequences": {"prompts": len(prompts), "held": len(held),
                          "held_positions": int(sum(len(x) for x in held)),
                          "prompt_positions": int(sum(len(x) for x in prompts))},
            "definitions": {"nll_mean": "mean -log p(next token) over the base vocab [0, band_begin), teacher forced",
                            "top1_agreement": "argmax over [0, band_begin) equal, every position",
                            "kl": "KL(trunk || model) resp. KL(reference || model) per position, fp32 log-softmax",
                            "residual_norm_ratio_vs_trunk": "||x_model|| / ||x_trunk|| after block k (all positions)"},
            "seconds": round(time.time() - started, 2),
            "note": "diagnostics only (owner policy 2026-10-06)"}


def resolve_prompts(args: argparse.Namespace) -> tuple[list[list[int]], dict[str, Any]]:
    if args.prompt_ids:
        packs = json.loads(Path(args.prompt_ids).read_text(encoding="utf-8"))
        packs = [[int(t) for t in p][: args.max_tokens] for p in packs]
        return packs, {"source": args.prompt_ids, "sha256": c10.sha256_file(args.prompt_ids)}
    if not args.tokenizer_dir:
        raise Refusal("--tokenizer-dir (the HF S3 child dir) or --prompt-ids is required")
    tree = Path(c10.CHIMERA_SRC).parent
    script = tree / "scripts" / "import_receipt_qwen35.py"
    spec = importlib.util.spec_from_file_location("import_receipt_qwen35", script)
    rq = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rq)
    texts = tuple(rq.PROMPT_PACK)
    packs = rq.tokenize(Path(args.tokenizer_dir), texts, args.max_tokens)
    return packs, {"source": f"{script} PROMPT_PACK (8 R-Q prompts)", "script_sha256": c10.sha256_file(str(script)),
                   "tokenizer_dir": args.tokenizer_dir,
                   "prompts": [{"index": i, "text": t, "tokens": len(p)} for i, (t, p) in enumerate(zip(texts, packs))]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--partial", default=None, help="<c10 --out>.partial (required unless --variant)")
    ap.add_argument("--variant", action="append", default=[],
                    help="1.2.0: NAME=DIR, a sweep variant directory (c10 --sweep-targets <out>/variants/<name>); repeatable; "
                         "all variants run in one layer-major pass beside the trunk")
    ap.add_argument("--reference", default=None,
                    help="1.2.0: NAME=DIR, a full Chimera checkpoint (the S0 import of the parent) evaluated in the same pass "
                         "(views read lazily)")
    ap.add_argument("--evidence", default=None, help="1.2.0: evidence manifest; its HELD rows are scored (NLL, top-1, KL)")
    ap.add_argument("--held-rows", type=int, default=None, help="1.2.0: score only the first N held rows")
    ap.add_argument("--position-chunk", type=int, default=512, help="1.2.0: positions per head chunk")
    ap.add_argument("--k3-channels", type=int, required=True)
    ap.add_argument("--parent-parameter-count", type=int, default=None)
    ap.add_argument("--media-units-parameters", type=int, default=None, help="S7 budget (E4: 591,437,129)")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--prompt-ids", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="default: the trunk's FFN dtype")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--pad-missing-with-trunk", action="store_true",
                    help="layers after the persisted prefix get the trunk FFN zero-padded to the union width (a HYBRID "
                         "child, labelled as such in the receipt; for prefix rehearsals only)")
    ap.add_argument("--out", required=True, help="receipt JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    trunk_dir = Path(args.trunk).absolute()
    if not args.partial and not args.variant:
        raise Refusal("--partial or --variant is required")
    if args.partial and args.variant:
        raise Refusal("--partial and --variant are exclusive")
    device = c10.canonical_device(args.device)
    doc = read_checkpoint_config(trunk_dir)
    trunk_cfg = ChimeraConfig.from_manifest(doc["config"])
    widths = {trunk_cfg.resolved_intermediate_size(k) for k in range(trunk_cfg.num_layers)}
    if len(widths) != 1:
        raise Refusal(f"trunk is not uniform: {sorted(widths)}")
    q = widths.pop()
    n_layers, h = int(trunk_cfg.num_layers), int(trunk_cfg.hidden_size)
    views = read_safetensors(trunk_dir, verify=True)
    ffn_dtype = _torch_dtype(views["blocks.0.ffn.up.weight"].dtype)
    dtype = getattr(torch, args.dtype) if args.dtype else ffn_dtype
    if args.variant:
        return main_variants(args, argv, started, trunk_dir, doc, trunk_cfg, q, n_layers, h, views, ffn_dtype, dtype,
                             device)
    partial_dir = Path(args.partial).absolute()
    inputs = json.loads((partial_dir / c10.PARTIAL_INPUTS).read_text(encoding="utf-8"))
    identity = inputs.get("identity", {})
    if int(identity.get("k3_channels", -1)) != args.k3_channels:
        raise Refusal(f"--k3-channels {args.k3_channels} != the partial run's {identity.get('k3_channels')}")
    if identity.get("trunk") != c10.directory_file_digests(trunk_dir):
        raise Refusal("the partial run was made on a different trunk (file digests differ)")
    log(f"checking {n_layers} partial layers under {partial_dir}")
    layers, rows, width = check_partial(partial_dir, views, n_layers, q, h, ffn_dtype, args.pad_missing_with_trunk)
    if width != q + args.k3_channels:
        raise Refusal(f"partial width {width} != trunk {q} + k3 {args.k3_channels}")
    receipt: dict[str, Any] = {
        "schema": "neural-forge-c10-virtual-child-check-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": c10.sha256_file(os.path.abspath(__file__)), "c10_tool_sha256": c10.sha256_file(c10.__file__),
        "argv": list(argv) if argv is not None else sys.argv[1:], "torch": torch.__version__,
        "chimera_src": c10.CHIMERA_SRC, "device": str(device), "dtype": str(dtype).replace("torch.", ""),
        "trunk": {"path": str(trunk_dir), "chimera_commit": doc.get("chimera_commit"), "intermediate_size": q},
        "partial": {"path": str(partial_dir), "inputs_sha256": c10.sha256_file(str(partial_dir / c10.PARTIAL_INPUTS)),
                    "c10_tool_version": inputs.get("tool_version")},
        "child_intermediate_size": width, "k3_channels": args.k3_channels,
        "identity_rows": rows,
    }
    union_layers = [r["layer"] for r in rows if "source" not in r]
    padded_layers = [r["layer"] for r in rows if "source" in r]
    receipt["child_kind"] = ("virtual union child (every layer from the partial)" if not padded_layers else
                             f"HYBRID: c10 union layers {union_layers[0]}..{union_layers[-1]} from the partial, trunk FFN "
                             f"zero-padded to the union width on layers {padded_layers[0]}..{padded_layers[-1]}")
    receipt["union_layers"] = union_layers
    receipt["trunk_padded_layers"] = padded_layers
    receipt["accounting"] = account(trunk_dir, layers, n_layers, h, args.k3_channels, args.parent_parameter_count,
                                    args.media_units_parameters)
    child_cfg = child_config(trunk_cfg, width, args.k3_channels, q)
    tied = dict((read_index(trunk_dir) or {}).get("metadata", {}).get("tied_weights", {}))
    receipt["structural_load"] = structural_load(child_cfg, views, layers, tied)
    receipt["veto_binding_layers"] = int(sum(bool(r["veto_binding"]) for r in rows if "source" not in r))
    if not args.skip_forward:
        packs, prompt_record = resolve_prompts(args)
        receipt["prompts"] = prompt_record
        logits_t, logits_c, facts = streamed_logits(trunk_cfg, child_cfg, views, layers, tied, packs, device, dtype)
        receipt["forward"] = facts
        receipt["comparison"] = compare(logits_t, logits_c, int(trunk_cfg.ledger_band_begin()))
        if not receipt["comparison"]["all_logits_finite"]:
            receipt["verdict"] = "REFUSED: non-finite logits"
    receipt.setdefault("verdict", ("HYBRID prefix child " if padded_layers else "") + "loads; identity and accounting hold" +
                       ("; text forward finite" if not args.skip_forward else "; forward skipped"))
    receipt["seconds"] = round(time.time() - started, 2)
    receipt["peak_rss_gigabytes"] = c10.rss_gigabytes()
    if device.type == "cuda":
        receipt["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    receipt["receipt_sha256"] = c10.write_receipt(args.out, receipt)
    log(f"{receipt['verdict']}; receipt -> {args.out}")
    return 0 if not receipt["verdict"].startswith("REFUSED") else 2


def held_sequences(path: str, limit: int | None) -> tuple[list[list[int]], dict[str, Any]]:
    """The evidence manifest's HELD rows (qf_corpus split), one token list per row."""
    evidence = c10.load_evidence(path)
    rows: list[list[int]] = []
    for batch_index, ids in evidence.batches:
        if evidence.split[batch_index] != "held":
            continue
        table = ids if ids.ndim == 2 else ids.unsqueeze(0)
        rows.extend([[int(t) for t in row.tolist()] for row in table])
    if limit is not None:
        rows = rows[:limit]
    return rows, {"path": path, "file_sha256": evidence.sha256, "manifest_sha256": evidence.manifest_sha256,
                  "held_batches": evidence.held_batches, "rows": len(rows), "sequence_length": evidence.sequence_length}


def main_variants(args: argparse.Namespace, argv: list[str] | None, started: float, trunk_dir: Path, doc: dict[str, Any],
                  trunk_cfg: ChimeraConfig, q: int, n_layers: int, h: int, views: dict[str, Any],
                  ffn_dtype: torch.dtype, dtype: torch.dtype, device: torch.device) -> int:
    """1.2.0: several sweep variants (+ an optional reference) in one pass (module docstring)."""
    LazyTensor._shared = None
    digests = c10.directory_file_digests(trunk_dir)
    tied = dict((read_index(trunk_dir) or {}).get("metadata", {}).get("tied_weights", {}))
    receipt: dict[str, Any] = {
        "schema": "neural-forge-c10-virtual-child-check-variants-v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": c10.sha256_file(os.path.abspath(__file__)), "c10_tool_sha256": c10.sha256_file(c10.__file__),
        "argv": list(argv) if argv is not None else sys.argv[1:], "torch": torch.__version__,
        "chimera_src": c10.CHIMERA_SRC, "device": str(device), "dtype": str(dtype).replace("torch.", ""),
        "trunk": {"path": str(trunk_dir), "chimera_commit": doc.get("chimera_commit"), "intermediate_size": q,
                  "files": digests},
        "k3_channels": args.k3_channels, "variants": {},
    }
    children: dict[str, tuple[ChimeraConfig, dict[int, dict[str, Any]]]] = {}
    for spec in args.variant:
        if "=" not in spec:
            raise Refusal(f"--variant expects NAME=DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        if name in children or name in ("trunk", "__reference__"):
            raise Refusal(f"--variant name {name!r} is repeated or reserved")
        vdir = Path(path).absolute()
        inputs = json.loads((vdir / c10.PARTIAL_INPUTS).read_text(encoding="utf-8"))
        identity = inputs.get("identity", {})
        if int(identity.get("k3_channels", -1)) != args.k3_channels:
            raise Refusal(f"{vdir}: --k3-channels {args.k3_channels} != the sweep's {identity.get('k3_channels')}")
        if identity.get("trunk") != digests:
            raise Refusal(f"{vdir}: the sweep was made on a different trunk (file digests differ)")
        log(f"checking variant {name}: {n_layers} layers under {vdir}")
        layers, rows, width = check_partial(vdir, views, n_layers, q, h, ffn_dtype, False, lazy=True)
        if width != q + args.k3_channels:
            raise Refusal(f"{vdir}: width {width} != trunk {q} + k3 {args.k3_channels}")
        child_cfg = child_config(trunk_cfg, width, args.k3_channels, q)
        receipt["variants"][name] = {
            "dir": str(vdir), "inputs_sha256": c10.sha256_file(str(vdir / c10.PARTIAL_INPUTS)),
            "union_target": identity.get("union_target"), "open_loop": identity.get("open_loop"),
            "c10_tool_version": inputs.get("tool_version"), "identity_rows": rows,
            "accounting": account(trunk_dir, layers, n_layers, h, args.k3_channels, args.parent_parameter_count,
                                  args.media_units_parameters),
            "structural_load": structural_load(child_cfg, views, layers, tied),
            "veto_binding_layers": int(sum(bool(r["veto_binding"]) for r in rows)),
        }
        children[name] = (child_cfg, layers)
    reference = None
    band = int(trunk_cfg.ledger_band_begin())
    if args.reference:
        if "=" not in args.reference:
            raise Refusal(f"--reference expects NAME=DIR, got {args.reference!r}")
        r_name, r_path = args.reference.split("=", 1)
        r_dir = Path(r_path).absolute()
        r_doc = read_checkpoint_config(r_dir)
        r_cfg = ChimeraConfig.from_manifest(r_doc["config"])
        if int(r_cfg.hidden_size) != h or int(r_cfg.num_layers) != n_layers:
            raise Refusal(f"--reference {r_dir}: hidden {r_cfg.hidden_size} / layers {r_cfg.num_layers} != the trunk's")
        r_views = c10.lazy_views(r_dir)
        r_tied = dict((read_index(r_dir) or {}).get("metadata", {}).get("tied_weights", {}))
        reference = (r_name, r_cfg, r_views, r_tied)
        band = min(band, int(r_cfg.ledger_band_begin()))
        receipt["reference"] = {"name": r_name, "path": str(r_dir), "chimera_commit": r_doc.get("chimera_commit"),
                                "intermediate_size": int(r_cfg.intermediate_size), "tensors": len(r_views),
                                "views": "lazy (headers; tensor bytes read per block; files NOT re-hashed here)"}
    if not args.skip_forward:
        packs, prompt_record = resolve_prompts(args)
        held, held_record = held_sequences(args.evidence, args.held_rows) if args.evidence else ([], None)
        receipt["prompts"] = prompt_record
        receipt["evidence"] = held_record
        ev = streamed_multi_eval(trunk_cfg, views, tied, children, reference, packs, held, device, dtype, band,
                                 args.position_chunk)
        receipt["eval"] = ev
        finite = all(m[kind]["logits_finite"] for m in ev["models"].values() for kind in ("prompts", "held")
                     if m[kind]["positions"])
        finite = finite and all(row["residual_finite"] for m in ev["models"].values() for row in m.get("per_layer", []))
        if not finite:
            receipt["verdict"] = "REFUSED: non-finite logits or residuals"
    receipt.setdefault("verdict", f"{len(children)} variant child(ren) load; identity and accounting hold"
                       + ("; forward finite" if not args.skip_forward else "; forward skipped"))
    receipt["seconds"] = round(time.time() - started, 2)
    receipt["peak_rss_gigabytes"] = c10.rss_gigabytes()
    if device.type == "cuda":
        receipt["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    receipt["receipt_sha256"] = c10.write_receipt(args.out, receipt)
    log(f"{receipt['verdict']}; receipt -> {args.out}")
    return 0 if not receipt["verdict"].startswith("REFUSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
