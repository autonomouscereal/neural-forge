#!/usr/bin/env python3
"""T(ii): the streamed forward+backward LoRA KD trainer for the OMNI1 27B student.

WHY THIS FILE EXISTS.  DISTILL_DESIGN_2026-10-12 section 4 prices stage D2 on two
engines.  The planning engine is the measured split cell (RTX 8000 + 3090, fp16
LoRA, 151.24 tok/s).  The upside engine is a streamed bf16 fwd+bwd on the 3090s,
and the design says plainly that it "does not exist yet": the streamed teacher in
`distill/tps_probe.py` loads each layer into two reused buffers with
`assign=False`, so the weights are gone before any backward.  This file is that
missing engine, plus the D2 objective, the D2 LoRA site plan and the run plumbing
(host-staged data parallel, rolling safety checkpoints, exact resume).  The split
cell is the same algorithm with every layer resident across two cards, behind
the same CLI (`--engine split`).

THE ALGORITHM, IN ONE PARAGRAPH.  Forward runs under `no_grad`, one decoder layer
resident: the input of every layer (the "boundary") is copied to pinned host RAM
and only the current hidden state stays on the card.  The final norm and the
chunked head compute the loss and its gradient with respect to the last hidden
state.  Backward then walks the layers in reverse: layer i's weights are loaded
AGAIN, its boundary comes back from the host with `requires_grad`, the layer is
recomputed with autograd on, and `y.backward(dh)` produces the adapter gradients
of layer i and dh for layer i-1.  That is per-layer gradient checkpointing with
the weights themselves checkpointed to the page cache.  The split engine is the
same loop with resident layers on two devices and host-staged hops between them
(no P2P anywhere: `x.to("cpu").to(dev)`).

LoRA SITES (design section 4 D2, verbatim intent).  Trainable: trunk FFN
gate/up/down r64 on the 4,352 trunk channels only; GDN (`linear_attn`) in/out
projections r64; gated-GQA v/o r64; the K3 down columns get their own r32 LoRA
and a per-channel fp32 gain initialised at exactly 1.0.  Frozen: GQA q/k, RoPE,
KDA decay (`A_log`, `dt_bias`), every norm, embed, head and all media tensors.
Adapters are attached with forward HOOKS on the stock `nn.Linear` modules rather
than by wrapping them, so the state-dict names of a layer never change and the
streamed engine can keep loading layer weights by their checkpoint names.

THE GAIN, AND WHERE IT FOLDS.  The K3 channels of the uniform 16,640-wide FFN are
columns [W_t, I) of `down_proj` (rows [W_t, I) of gate/up), W_t = 4,352 (c10's
`[gate_q | gate_k3 ; up_q | s.up_k3]` fused layout, sliced by the P1 reader).
The gain multiplies the K3 rows of `up_proj`'s output, i.e. the K3 channel
activation, so it is exactly a per-column scale of the K3 down columns.  The K3
down LoRA sees the gained activation, so the D4 fold is
`down[:, W_t:] <- (down[:, W_t:] + s_k3 * B_k3 @ A_k3) * gain` (equivalently
`up[W_t:, :] *= gain` with the LoRA merged unscaled), in fp32, once.

THE OBJECTIVE (design section 4 D2; every term over vocab rows [0, 248,320)).
Per next-token position p, summed and divided by the step's global position
count (so data parallel is an exact split of one big batch):
  w_q38(p) * KL_bucket(teacher Q38 top-64 + tail || student)
+ w_k3(p)  * KL_bucket(K3 top-32 mapped to Qwen ids by identical bytes, unmatched
             and out-of-top-k mass in the tail || student)
+ w_ce     * CE(ground truth)                         (the LM-CE floor, 0.1)
+ w_eos    * CE(ground truth) at positions whose target is an EOS id
+ w_ul     * sum_c -log(1 - q(c)) over tokens c that would repeat a 3-gram
On K3 rows w_q38 = lambda_k3 * anchor and w_k3 = lambda_k3 * (1 - anchor)
(anchor 0.5, lambda_k3 0.3: the 0.5/0.5 Q38 anchor of the design, with lambda_k3
the weight of the K3 row family against text rows).  On-policy rows may use the
generalised JSD (beta 0.5) over the same buckets instead of the forward KL.  The
student's log-softmax is taken over the text rows only -- the head weight is
sliced to [0, text_vocab_rows) before the matmul -- so the ledger and C18 music
bands can never be pushed down by a text loss.

THE WEIGHT SOURCE IS AN INTERFACE.  The student is OMNI1 read through the S0 P1
zero-copy reader `p1_qwen35_reader.py` (Chimera -> qwen3_5 names and slices,
uniform FFN 16,640, embed/head rows [0, 248,320)), with registered deltas (D1)
composed per tensor by that reader from a sealed `oc_checkpoint.py` plan
(`--p1-chimera --p1-parent-config [--p1-plan]`).  The engine only needs a
duck-typed surface (`get`, `has`, `names_under`, `config_dict`, `identity`,
`text_prefix`): `P1Source` adapts the P1 reader to it, `SafetensorsSource` is the
plain fallback (how the Qwen3.5 parity models are read), `--source-plugin` takes
any other reader.

THE TEACHER TARGETS ARE OTHER TOOLS' FORMATS, READ THROUGH THOSE TOOLS.  Q38
targets are sealed `qtext_teacher_cache.py` caches; K3 targets are
`k3_teacher_cache.py xtok` files.  Both are loaded by path from this directory
(`load_peer_module`), so their schema constants are theirs, not copies.

WHAT IS PROVEN WHERE.  `grad-parity` compares every adapter gradient of the
streamed engine (and optionally the resident/split engine with a host hop)
against a stock full-load autograd of the same model on the same batch, per
site, in fp32.  `test_streamed_kd_trainer.py` (beside this file) proves the
objective against an independent per-position reference, the hooks against
explicit algebra (including the D4 fold rule), both engines against stock
autograd, exact resume, DP == single-process, and the Q-text / xtok readers
against caches written by the real peer tools.

RUN SAFETY (1.3.0).  A run that keeps skipping steps (fp16 overflow, non-finite
gradients) HALTS: after --max-consecutive-skips skips in a row, or more than
--max-skip-fraction of the last --skip-window steps, it writes a safety
checkpoint and a receipt with `verdict: halted_nonfinite` and exits 3 without
exporting adapters (design section 4 risk row 2: "pauses for a ruling if they
persist").  A segment that applied no step exits 3 the same way.  A resume
compares the checkpoint's base-weight identity and data identity (Q-text
manifests, corpora, xtok files) with the current ones and refuses on any
difference unless --allow-identity-change, which is receipted per segment.  An
xtok K3 target is joined to a Q-text cache only when its Qwen corpus is proven
to be the cache's corpus: the `qwen_corpus_sha256` metadata, or the
xtok_receipt.json -> boundary map -> `ids_files_sha256` chain k3_teacher_cache.py
already writes; otherwise the data config must say `xtok_unverified_ok`.

WHAT IS NOT HERE.  No rate is claimed (S0 P2 measures it on a 3090).  The
export of the trained adapters into Chimera-named delta entries for
`oc_checkpoint.py --delta-base` is not built (the map is P1's `specs`; see the
fold rule in adapters_manifest.json).

Examples
--------
  PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python   # P40 (sm61)
  T=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 $PY $T grad-parity \\
      --model /home/cereal/models/Qwen3.5-0.8B --dtype float32 --kernels torch \\
      --trunk-ffn-width 1024 --receipt /tmp/parity.json
  $PY $T --self-test                     # CPU only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TOOL_NAME = "streamed_kd_trainer"
TOOL_VERSION = "1.3.2"
# 1.3.2 (trainer-prep lane, review fix, 2026-10-13): a train run that can move a weight (--lr != 0, or --gain-lr != 0
# with the gain trained) is refused before anything loads unless it names --preregistration (verified against the
# pin) or waives it with --no-preregistration REASON (recorded in the receipt).  A lr = 0 / gain-lr = 0 rate cell
# moves nothing and runs unpinned, receipted as such.  1.3.1 accepted a training run with no pin at all.
# 1.3.1 (trainer-prep lane, E5/E6, 2026-10-13): a Q-text corpus that carries the distill corpus's in-band
# `real_lengths` (and no `lengths`) is cut to it -- qtext_teacher_cache reads only `lengths`, so on the K3-row
# Qwen files (1,991,482 real of 1,998,848 positions) it stores the separator padding as real text and 1.3.0
# trained CE / EOS / KL on it; an xtok target is refused when an id is outside the cache's vocab rows, a
# position repeats a valid id (the bucket KL assumes unique ids) or a K3 logprob is not finite;
# --preregistration is checked against the digest pinned in stage2_training_receipt.py (refused if it moved)
RECEIPT_SCHEMA = "neural-forge-omni-streamed-kd-trainer-v1"
ADAPTER_SCHEMA = "neural-forge-omni-d2-adapters-v1"
CHECKPOINT_SCHEMA = "neural-forge-omni-safety-checkpoint-v1"

#: exit code of a run that halted on persistent skips or applied no step (2 is a refusal)
EXIT_HALTED = 3

#: Text rows of the OMNI1 vocabulary.  [248,320, 252,416) are the ledger bands and
#: [252,416, 317,952) the C18 music band (design section 4).  No text loss and no
#: student log-softmax ever sees a row at or above this.
TEXT_VOCAB_ROWS = 248_320

#: Qwen3.5 tokenizer: 248,044 regular rows; random parity ids stay below it.
DEFINED_VOCABULARY_ROWS = 248_044

#: EOS ids of the Qwen3.8-27B-U generation config (`eos_token_id: [248046, 248044]`).
DEFAULT_EOS_IDS = (248046, 248044)

DEFAULT_TEXT_PREFIX = "model.language_model"


class TrainerError(RuntimeError):
    """A refusal that is this tool's own, never a bare torch traceback."""


# --------------------------------------------------------------------------
# Small shared utilities (same one-line log format as tps_probe / the c-tools).
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json_atomic(path: Path, value: Any) -> str:
    """Write pretty JSON via a temp file + rename; returns the file sha256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return file_sha256(path)


def stable_seed(*parts: Any) -> int:
    """A 63-bit seed from arbitrary parts, identical in every process and rank."""

    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def resolve_dtype(name: str) -> torch.dtype:
    allowed = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
    if name not in allowed:
        raise TrainerError(f"{name!r} is not one of {sorted(allowed)}")
    return allowed[name]


def synchronise(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
        else:
            dev = torch.device(device)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)


def device_memory_report() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    out = {}
    for index in range(torch.cuda.device_count()):
        out[f"cuda:{index}"] = {
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(index) / 1e9, 3),
            "name": torch.cuda.get_device_name(index),
        }
    return out


def host_rss_gb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) * 1024 / 1e9, 3)
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------
# Linear-attention kernel selection.
#
# `Qwen3_5GatedDeltaNet.__init__` binds its kernels from MODULE GLOBALS at
# construction time (fla's chunk kernel and FusedRMSNormGated when fla imports).
# Triton does not run on Pascal (sm61) and FusedRMSNormGated asks for
# `torch.cuda.current_device()` in its constructor, which a CPU-only process
# does not have.  "torch" therefore patches those globals BEFORE any layer is
# built, so the stock reference and the streamed buffers get the SAME torch
# implementation -- the parity arm compares the engine, not two kernel stacks.
# --------------------------------------------------------------------------


def make_fp32_core(rule: Any) -> Any:
    """The transformers torch GDN rule with its inputs upcast to fp32 first.

    `torch_chunk_gated_delta_rule` applies the qk l2norm BEFORE its own fp32
    cast, i.e. in the activation dtype.  In fp16 the backward of that rsqrt
    overflows for any head vector of small norm (measured: 184 of 312 adapter
    gradients of the 0.8B non-finite at every loss scale, from layer 14 down,
    with forward |h| <= 12.8), which fla's kernel does not do: it normalises in
    fp32 inside the kernel.  For fp32 inputs this wrapper is the identity, so the
    fp32 parity arm is unaffected; for half inputs the output is cast back.
    """

    def fp32_core(query, key, value, g, beta, **kwargs):
        dtype = query.dtype
        if dtype in (torch.float16, torch.bfloat16):
            out, state = rule(query.float(), key.float(), value.float(), g.float(), beta.float(), **kwargs)
            return out.to(dtype), state
        return rule(query, key, value, g, beta, **kwargs)

    fp32_core.__qualname__ = "fp32_core(torch_chunk_gated_delta_rule)"
    fp32_core._kd_wrapped = rule
    return fp32_core


def select_linear_attention_kernels(mode: str) -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    if mode not in ("auto", "torch"):
        raise TrainerError(f"--kernels must be auto or torch, not {mode!r}")
    if mode == "torch":
        mq.FusedRMSNormGated = None
        mq.chunk_gated_delta_rule = None
        mq.fused_recurrent_gated_delta_rule = None
        mq.causal_conv1d_fn = None
        mq.causal_conv1d_update = None
        mq.is_fast_path_available = False
        if not hasattr(mq.torch_chunk_gated_delta_rule, "_kd_wrapped"):
            mq.torch_chunk_gated_delta_rule = make_fp32_core(mq.torch_chunk_gated_delta_rule)
    return {
        "mode": mode,
        "fused_rmsnorm_gated": mq.FusedRMSNormGated is not None,
        "chunk_gated_delta_rule": getattr(mq.chunk_gated_delta_rule, "__module__", "torch_fallback")
        if mq.chunk_gated_delta_rule is not None
        else getattr(mq.torch_chunk_gated_delta_rule, "__qualname__", "torch_chunk_gated_delta_rule"),
        "causal_conv1d": mq.causal_conv1d_fn is not None,
    }


def linear_attention_kernel_facts(layer: nn.Module) -> dict[str, Any] | None:
    attn = getattr(layer, "linear_attn", None)
    if attn is None:
        return None
    return {
        "norm": type(attn.norm).__name__,
        "chunk_rule": getattr(attn.chunk_gated_delta_rule, "__qualname__", str(attn.chunk_gated_delta_rule)),
        "causal_conv1d_fn": attn.causal_conv1d_fn is not None,
    }


# --------------------------------------------------------------------------
# Weight sources.  THE INTERFACE (duck-typed; P1 implements the same five
# members plus `text_prefix`):
#
#   get(name) -> torch.Tensor     CPU tensor under its qwen3_5 name (a view is fine)
#   has(name) -> bool
#   names_under(prefix) -> list[str]
#   config_dict() -> dict         qwen3_5 config.json content (full or text-only)
#   identity() -> dict            JSON-able; pinned into every receipt and delta
#   text_prefix: str              "model.language_model" (VL layout) or "model"
# --------------------------------------------------------------------------


class SafetensorsSource:
    """A sharded (or single-file) safetensors checkpoint, read on demand to CPU.

    Handles stay open across passes so the second pass is served by the page
    cache.  `bytes_read` counts every byte handed out, for the receipt's MB/s.
    """

    def __init__(self, model_dir: Path) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = dict(index["weight_map"])
            self.index_sha256 = file_sha256(index_path)
        else:
            single = self.model_dir / "model.safetensors"
            if not single.is_file():
                raise TrainerError(f"no model.safetensors(.index.json) under {self.model_dir}")
            with safe_open(str(single), framework="pt", device="cpu") as handle:
                self.weight_map = {name: single.name for name in handle.keys()}
            self.index_sha256 = None
        self._handles: dict[str, Any] = {}
        self.bytes_read = 0
        if f"{DEFAULT_TEXT_PREFIX}.embed_tokens.weight" in self.weight_map:
            self.text_prefix = DEFAULT_TEXT_PREFIX
        elif "model.embed_tokens.weight" in self.weight_map:
            self.text_prefix = "model"
        else:
            raise TrainerError(f"{self.model_dir}: no text embed_tokens under a known prefix")

    def _handle(self, filename: str) -> Any:
        handle = self._handles.get(filename)
        if handle is None:
            handle = self._safe_open(str(self.model_dir / filename), framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise TrainerError(f"{name!r} is not in {self.model_dir}")
        tensor = self._handle(self.weight_map[name]).get_tensor(name)
        self.bytes_read += tensor.numel() * tensor.element_size()
        return tensor

    def names_under(self, prefix: str) -> list[str]:
        return sorted(name for name in self.weight_map if name.startswith(prefix))

    def config_dict(self) -> dict[str, Any]:
        return json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))

    def identity(self) -> dict[str, Any]:
        files = []
        for filename in sorted(set(self.weight_map.values())):
            stat = (self.model_dir / filename).stat()
            files.append({"file": filename, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return {
            "kind": "safetensors",
            "dir": str(self.model_dir),
            "index_sha256": self.index_sha256,
            "config_sha256": file_sha256(self.model_dir / "config.json"),
            "tensors": len(self.weight_map),
            "files": files,
        }


def load_peer_module(name: str) -> Any:
    """A sibling tool of this lane (same distill/tools dir), imported by path.

    The cache formats and the P1 map belong to their own tools; this file reads
    them THROUGH those tools' code and constants rather than carrying a second
    copy of the contract (a copied contract goes stale silently).
    """

    path = Path(os.environ.get(f"{name.upper()}_PATH", str(Path(__file__).resolve().parent / f"{name}.py")))
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(path):
        return sys.modules[name]
    if not path.is_file():
        raise TrainerError(f"peer tool {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def peer_identity(name: str) -> dict[str, Any]:
    module = load_peer_module(name)
    return {"name": name, "path": module.__file__, "sha256": file_sha256(Path(module.__file__)),
            "version": getattr(module, "TOOL_VERSION", None)}


class P1Source:
    """OMNI1 (or any registered Chimera child) served in the qwen3_5 layout by the
    S0-P1 reader p1_qwen35_reader.py, zero-copy, with registered deltas composed
    per tensor from a sealed oc_checkpoint.py delta-compose plan (--p1-plan).

    The reader serves embed / lm_head as rows [0, ledger_base_vocab) only (the
    band rows are its extras), so the text-row rule holds by construction; the
    qwen3_5 config is the parent's with the child's uniform FFN width.
    """

    text_prefix = DEFAULT_TEXT_PREFIX

    def __init__(self, chimera_dir: Path, parent_config_dir: Path, plan: Path | None = None) -> None:
        self.module = load_peer_module("p1_qwen35_reader")
        self.reader = self.module.open_reader(str(chimera_dir), plan_path=str(plan) if plan else None)
        self.chimera_dir = Path(chimera_dir)
        self.parent_config_path = Path(parent_config_dir) / "config.json"
        self.parent_config = json.loads(self.parent_config_path.read_text(encoding="utf-8"))
        self.plan = Path(plan) if plan else None

    @property
    def bytes_read(self) -> int:
        return int(self.reader.bytes_read)

    def get(self, name: str) -> torch.Tensor:
        return self.reader.get(name)

    def has(self, name: str) -> bool:
        return self.reader.has(name)

    def names_under(self, prefix: str) -> list[str]:
        return self.reader.names_under(prefix)

    def config_dict(self) -> dict[str, Any]:
        return self.reader.qwen35_config(self.parent_config)

    def identity(self) -> dict[str, Any]:
        summary = self.reader.summary()
        summary.pop("coverage", None)
        return {
            "kind": "p1",
            "p1_tool": peer_identity("p1_qwen35_reader"),
            "chimera_dir": str(self.chimera_dir),
            "chimera_config_sha256": file_sha256(self.chimera_dir / "config.json"),
            "chimera_shards": {name: info["bytes"] for name, info in sorted(self.reader.shards.files.items())},
            "parent_config_sha256": file_sha256(self.parent_config_path),
            "plan": None if self.plan is None else {"path": str(self.plan), "sha256": file_sha256(self.plan)},
            "summary": summary,
        }


def load_plugin_source(spec: str, args_json: str | None) -> Any:
    """file.py:factory -> factory(**json.loads(args_json)): any other reader with
    the same surface (get / has / names_under / config_dict / identity)."""

    path_text, _, factory_name = spec.rpartition(":")
    if not path_text or not factory_name:
        raise TrainerError(f"--source-plugin wants file.py:factory, got {spec!r}")
    module_spec = importlib.util.spec_from_file_location("trainer_source_plugin", path_text)
    if module_spec is None or module_spec.loader is None:
        raise TrainerError(f"cannot import {path_text}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    source = getattr(module, factory_name)(**(json.loads(args_json) if args_json else {}))
    for member in ("get", "has", "names_under", "config_dict", "identity"):
        if not callable(getattr(source, member, None)):
            raise TrainerError(f"source from {spec} lacks {member}()")
    if not hasattr(source, "text_prefix"):
        source.text_prefix = DEFAULT_TEXT_PREFIX
    return source


def build_source(args: argparse.Namespace) -> Any:
    chosen = [bool(getattr(args, "p1_chimera", None)), bool(getattr(args, "source_plugin", None)), bool(getattr(args, "model", None))]
    if sum(chosen) != 1:
        raise TrainerError("give exactly one of --p1-chimera (OMNI1 via P1), --source-plugin, --model (safetensors)")
    if getattr(args, "p1_chimera", None):
        if not getattr(args, "p1_parent_config", None):
            raise TrainerError("--p1-chimera needs --p1-parent-config (the parent HF dir whose config.json P1 widens)")
        return P1Source(Path(args.p1_chimera), Path(args.p1_parent_config), Path(args.p1_plan) if args.p1_plan else None)
    if getattr(args, "source_plugin", None):
        return load_plugin_source(args.source_plugin, getattr(args, "source_plugin_args", None))
    return SafetensorsSource(Path(args.model))


def text_config_from_source(source: Any, dtype: torch.dtype, attention_implementation: str) -> Any:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    raw = dict(source.config_dict())
    model_type = raw.get("model_type")
    if model_type == "qwen3_5":
        config = Qwen3_5Config.from_dict(raw).text_config
    elif model_type == "qwen3_5_text":
        config = Qwen3_5TextConfig.from_dict(raw)
    else:
        raise TrainerError(f"config model_type {model_type!r} is not qwen3_5 / qwen3_5_text")
    config.dtype = dtype
    config._attn_implementation = attention_implementation
    return config


def head_weight_name(source: Any) -> str:
    prefix = getattr(source, "text_prefix", DEFAULT_TEXT_PREFIX)
    return "lm_head.weight" if source.has("lm_head.weight") else f"{prefix}.embed_tokens.weight"


# --------------------------------------------------------------------------
# The D2 site plan and the adapter bank.
# --------------------------------------------------------------------------


@dataclass
class SiteSpec:
    site: str  # submodule path inside a decoder layer, e.g. "mlp.gate_proj"
    kind: str  # "full" | "ffn_in" | "ffn_out"
    in_features: int
    out_features: int
    rank: int
    alpha: float
    trunk: int = 0  # FFN trunk width W_t for the ffn_* kinds
    k3_rank: int = 0
    k3_alpha: float = 0.0
    gain: bool = False

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)

    @property
    def k3_scaling(self) -> float:
        return float(self.k3_alpha) / float(self.k3_rank) if self.k3_rank else 0.0

    def part_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.kind == "full":
            return {"a": (self.rank, self.in_features), "b": (self.out_features, self.rank)}
        if self.kind == "ffn_in":
            shapes = {"a": (self.rank, self.in_features), "b": (self.trunk, self.rank)}
            if self.gain and self.out_features > self.trunk:
                shapes["gain"] = (self.out_features - self.trunk,)
            return shapes
        if self.kind == "ffn_out":
            shapes = {"a": (self.rank, self.trunk), "b": (self.out_features, self.rank)}
            if self.k3_rank and self.in_features > self.trunk:
                shapes["k3_a"] = (self.k3_rank, self.in_features - self.trunk)
                shapes["k3_b"] = (self.out_features, self.k3_rank)
            return shapes
        raise TrainerError(f"unknown site kind {self.kind!r}")


@dataclass
class SitePlanConfig:
    rank: int = 64
    alpha: float = 64.0
    k3_rank: int = 32
    k3_alpha: float = 32.0
    trunk_ffn_width: int = 4352
    gain: bool = True
    gdn_sites: tuple[str, ...] = ("in_proj_qkv", "out_proj")
    gqa_sites: tuple[str, ...] = ("v_proj", "o_proj")
    ffn: bool = True


def build_site_plan(text_config: Any, plan: SitePlanConfig) -> dict[str, list[SiteSpec]]:
    """Site specs per layer type, from the geometry in the config (no weights read)."""

    hidden = int(text_config.hidden_size)
    inter = int(text_config.intermediate_size)
    trunk = int(plan.trunk_ffn_width) if plan.trunk_ffn_width else inter
    if not 0 < trunk <= inter:
        raise TrainerError(f"--trunk-ffn-width {trunk} is outside (0, intermediate_size={inter}]")
    head_dim = int(getattr(text_config, "head_dim", hidden // text_config.num_attention_heads))
    q_heads = int(text_config.num_attention_heads)
    kv_heads = int(text_config.num_key_value_heads)
    k_dim = int(text_config.linear_key_head_dim) * int(text_config.linear_num_key_heads)
    v_dim = int(text_config.linear_value_head_dim) * int(text_config.linear_num_value_heads)
    gdn_geometry = {
        "in_proj_qkv": (hidden, 2 * k_dim + v_dim),
        "in_proj_z": (hidden, v_dim),
        "out_proj": (v_dim, hidden),
    }
    gqa_geometry = {"v_proj": (hidden, kv_heads * head_dim), "o_proj": (q_heads * head_dim, hidden)}
    for name in plan.gqa_sites:
        if name in ("q_proj", "k_proj"):
            raise TrainerError("GQA q/k are frozen by design (section 4 D2, section 6 item 1)")
        if name not in gqa_geometry:
            raise TrainerError(f"unknown gated-GQA site {name!r}")
    for name in plan.gdn_sites:
        if name not in gdn_geometry:
            raise TrainerError(f"unknown GDN site {name!r} (in_proj_a/in_proj_b feed the KDA decay and stay frozen)")
    ffn: list[SiteSpec] = []
    if plan.ffn:
        k3_rank = plan.k3_rank if inter > trunk else 0
        ffn = [
            SiteSpec("mlp.gate_proj", "ffn_in", hidden, inter, plan.rank, plan.alpha, trunk=trunk),
            SiteSpec("mlp.up_proj", "ffn_in", hidden, inter, plan.rank, plan.alpha, trunk=trunk, gain=plan.gain and inter > trunk),
            SiteSpec("mlp.down_proj", "ffn_out", inter, hidden, plan.rank, plan.alpha, trunk=trunk, k3_rank=k3_rank, k3_alpha=plan.k3_alpha),
        ]
    linear = [SiteSpec(f"linear_attn.{n}", "full", *gdn_geometry[n], plan.rank, plan.alpha) for n in plan.gdn_sites]
    full = [SiteSpec(f"self_attn.{n}", "full", *gqa_geometry[n], plan.rank, plan.alpha) for n in plan.gqa_sites]
    return {"linear_attention": linear + ffn, "full_attention": full + ffn}


def _lora(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    return F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * scaling


def site_forward_hook(module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
    """Adds the bound adapter to a frozen `nn.Linear`'s output.

    The adapters are fp32 and are their own master copy; they are cast to the
    activation dtype on the way into the matmul (tps_probe's LoRALinear rule:
    explicit casts, no autocast, so the no-grad forward and the recompute are
    the same function).  The gain is applied in fp32: at bf16 resolution near
    1.0 (2^-7) a learned gain would otherwise be quantised to 0.8% steps.
    """

    bound = getattr(module, "_kd_bound", None)
    if bound is None:
        return output
    spec: SiteSpec = module._kd_spec
    x = args[0]
    dtype = output.dtype
    if spec.kind == "full":
        return output + _lora(x, bound["a"], bound["b"], spec.scaling).to(dtype)
    if spec.kind == "ffn_in":
        trunk = spec.trunk
        head = output[..., :trunk] + _lora(x, bound["a"], bound["b"], spec.scaling).to(dtype)
        if trunk == output.shape[-1]:
            return head
        tail = output[..., trunk:]
        gain = bound.get("gain")
        if gain is not None:
            tail = (tail.float() * gain).to(dtype)
        return torch.cat([head, tail], dim=-1)
    if spec.kind == "ffn_out":
        trunk = spec.trunk
        out = output + _lora(x[..., :trunk], bound["a"], bound["b"], spec.scaling).to(dtype)
        if "k3_a" in bound:
            out = out + _lora(x[..., trunk:], bound["k3_a"], bound["k3_b"], spec.k3_scaling).to(dtype)
        return out
    raise TrainerError(f"unknown site kind {spec.kind!r}")


def install_site_hooks(layer: nn.Module, specs: list[SiteSpec]) -> None:
    for spec in specs:
        module = layer.get_submodule(spec.site)
        if not isinstance(module, nn.Linear):
            raise TrainerError(f"site {spec.site} is a {type(module).__name__}, not nn.Linear")
        if (module.in_features, module.out_features) != (spec.in_features, spec.out_features):
            raise TrainerError(
                f"site {spec.site}: module is {module.in_features}->{module.out_features}, "
                f"plan says {spec.in_features}->{spec.out_features}"
            )
        if getattr(module, "_kd_hooked", False):
            continue
        module._kd_spec = spec
        module._kd_bound = None
        module.register_forward_hook(site_forward_hook)
        module._kd_hooked = True


class AdapterBank:
    """Every trainable tensor of the run, keyed `layers.{i}.{site}.{part}`.

    The tensors live OUTSIDE the decoder layers: the streamed engine reuses one
    layer buffer per layer type, so a layer index is "bound" to the buffer's
    hooked modules just before the layer runs.  Initialisation is deterministic
    per tensor name (so DP ranks and a resumed process agree without a
    broadcast): A ~ kaiming-uniform(a=sqrt 5) = U(-1/sqrt(fan_in), +), B = 0
    (or N(0, b_init_std) for the parity arm, where B = 0 would make every A
    gradient zero and the comparison vacuous), gain = 1.0 exactly.
    """

    def __init__(
        self,
        plan: dict[str, list[SiteSpec]],
        layer_types: list[str],
        device: torch.device | str,
        seed: int,
        *,
        b_init_std: float = 0.0,
        gain_jitter: float = 0.0,
        train_gain: bool = True,
    ) -> None:
        self.plan = plan
        self.layer_types = list(layer_types)
        self.device = torch.device(device)
        self.tensors: dict[str, torch.Tensor] = {}
        self.layer_sites: list[dict[str, dict[str, torch.Tensor]]] = []
        self.train_gain = bool(train_gain)
        for index, layer_type in enumerate(self.layer_types):
            sites: dict[str, dict[str, torch.Tensor]] = {}
            for spec in plan[layer_type]:
                parts: dict[str, torch.Tensor] = {}
                for part, shape in spec.part_shapes().items():
                    name = f"layers.{index}.{spec.site}.{part}"
                    generator = torch.Generator(device="cpu").manual_seed(stable_seed(seed, name))
                    if part.endswith("a"):
                        bound = 1.0 / math.sqrt(shape[1])
                        value = (torch.rand(shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0) * bound
                    elif part.endswith("b"):
                        value = torch.zeros(shape, dtype=torch.float64)
                        if b_init_std > 0:
                            value = torch.randn(shape, generator=generator, dtype=torch.float64) * b_init_std
                    elif part == "gain":
                        value = torch.ones(shape, dtype=torch.float64)
                        if gain_jitter > 0:
                            value = value + (torch.rand(shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0) * gain_jitter
                    else:
                        raise TrainerError(f"unknown adapter part {part}")
                    tensor = value.to(dtype=torch.float32, device=self.device)
                    trainable = part != "gain" or self.train_gain
                    tensor.requires_grad_(trainable)
                    parts[part] = tensor
                    self.tensors[name] = tensor
                sites[spec.site] = parts
            self.layer_sites.append(sites)
        self.names = sorted(self.tensors)

    def bind(self, layer: nn.Module, index: int) -> None:
        for spec in self.plan[self.layer_types[index]]:
            layer.get_submodule(spec.site)._kd_bound = self.layer_sites[index][spec.site]

    def trainable(self) -> list[torch.Tensor]:
        return [self.tensors[n] for n in self.names if self.tensors[n].requires_grad]

    def trainable_names(self) -> list[str]:
        return [n for n in self.names if self.tensors[n].requires_grad]

    def parameter_groups(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        lora = [self.tensors[n] for n in self.names if self.tensors[n].requires_grad and not n.endswith(".gain")]
        gains = [self.tensors[n] for n in self.names if self.tensors[n].requires_grad and n.endswith(".gain")]
        return lora, gains

    def census(self) -> dict[str, Any]:
        by_part: dict[str, int] = {}
        for name, tensor in self.tensors.items():
            key = ".".join(name.split(".")[2:])  # drop "layers.{i}."
            by_part[key] = by_part.get(key, 0) + tensor.numel()
        total = sum(t.numel() for t in self.tensors.values())
        trainable = sum(t.numel() for t in self.tensors.values() if t.requires_grad)
        return {"tensors": len(self.tensors), "parameters_total": total, "parameters_trainable": trainable, "by_site_part": by_part}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: self.tensors[name].detach().to("cpu").contiguous() for name in self.names}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.names) - set(state))
        extra = sorted(set(state) - set(self.names))
        if missing or extra:
            raise TrainerError(f"adapter state mismatch: missing {missing[:4]} extra {extra[:4]}")
        with torch.no_grad():
            for name in self.names:
                if tuple(state[name].shape) != tuple(self.tensors[name].shape):
                    raise TrainerError(f"adapter {name}: shape {tuple(state[name].shape)} != {tuple(self.tensors[name].shape)}")
                self.tensors[name].copy_(state[name].to(self.tensors[name].device, torch.float32))

    def zero_grad(self) -> None:
        for tensor in self.tensors.values():
            tensor.grad = None


# --------------------------------------------------------------------------
# The objective.  See the module docstring for the formula.
# --------------------------------------------------------------------------


@dataclass
class ObjectiveConfig:
    w_q38: float = 1.0
    lambda_k3: float = 0.3
    k3_q38_anchor: float = 0.5
    w_ce: float = 0.1
    w_eos: float = 0.1
    w_ul: float = 0.1
    ul_ngram: int = 3
    ul_max_candidates: int = 8
    onpolicy_divergence: str = "fkl"  # or "jsd"
    jsd_beta: float = 0.5
    eos_ids: tuple[int, ...] = DEFAULT_EOS_IDS
    head_chunk_positions: int = 1024
    text_vocab_rows: int = TEXT_VOCAB_ROWS


def _first_valid(ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per row, the first valid id (0 if none) -- a harmless filler for masked slots."""

    position = torch.argmax(valid.to(torch.int8), dim=-1, keepdim=True)
    first = ids.gather(-1, position)
    return torch.where(valid.any(dim=-1, keepdim=True), first, torch.zeros_like(first))


def bucket_terms(logq: torch.Tensor, ids: torch.Tensor, teacher_logprobs: torch.Tensor):
    """Student/teacher bucket probabilities over (top-k ids that exist) + tail.

    Returns (p [n,K], lq [n,K], p_tail [n], log_q_tail [n], valid [n,K]).  An id
    of -1 is "no student counterpart" (a K3 token with no identical-byte Qwen
    token): its teacher mass goes to the tail.  The student tail is computed
    EXACTLY as the logsumexp of the student over every text row that is not a
    valid bucket id, not as 1 - sum(q), which loses all precision when the
    student is peaked on the teacher's top-k.
    """

    valid = ids >= 0
    safe = torch.where(valid, ids, _first_valid(ids, valid).expand_as(ids))
    lq = logq.gather(1, safe)
    p = torch.where(valid, teacher_logprobs.to(lq.dtype).exp(), torch.zeros_like(lq))
    p_tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0)
    #: a row whose valid ids cover every text row has no tail at all (only in
    #: toy vocabularies); its teacher "tail" is rounding residue and its student
    #: tail is an empty logsumexp, whose gradient would be NaN.  Valid ids are
    #: unique per row (top-k ids; the K3 byte map is injective).
    has_tail = valid.sum(dim=-1) < logq.shape[-1]
    masked = logq.scatter(1, safe, float("-inf"))
    if not bool(has_tail.all()):
        masked = torch.where(has_tail[:, None], masked, torch.zeros_like(masked))
        p_tail = torch.where(has_tail, p_tail, torch.zeros_like(p_tail))
    log_q_tail = torch.logsumexp(masked, dim=-1)
    log_q_tail = torch.where(has_tail, log_q_tail, torch.full_like(log_q_tail, float("-inf")))
    return p, lq, p_tail, log_q_tail, valid


def _plogp_ratio(p: torch.Tensor, log_p: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """p * (log p - log r) with 0 log 0 = 0 and no NaN from masked slots."""

    positive = p > 0
    safe_log_p = torch.where(positive, log_p, torch.zeros_like(log_p))
    safe_log_r = torch.where(positive, log_r, torch.zeros_like(log_r))
    return torch.where(positive, p * (safe_log_p - safe_log_r), torch.zeros_like(p))


def bucket_forward_kl(logq: torch.Tensor, ids: torch.Tensor, teacher_logprobs: torch.Tensor) -> torch.Tensor:
    p, lq, p_tail, log_q_tail, valid = bucket_terms(logq, ids, teacher_logprobs)
    head = _plogp_ratio(p, teacher_logprobs.to(lq.dtype), lq).sum(dim=-1)
    tail = _plogp_ratio(p_tail, torch.log(p_tail.clamp(min=1e-30)), log_q_tail)
    return head + tail


def bucket_jsd(logq: torch.Tensor, ids: torch.Tensor, teacher_logprobs: torch.Tensor, beta: float) -> torch.Tensor:
    """Generalised JSD_beta(p, q) over the same K+1 buckets (GKD's on-policy loss)."""

    p, lq, p_tail, log_q_tail, valid = bucket_terms(logq, ids, teacher_logprobs)
    q = torch.where(valid, lq.exp(), torch.zeros_like(lq))
    p_all = torch.cat([p, p_tail[:, None]], dim=1)
    q_all = torch.cat([q, log_q_tail.exp()[:, None]], dim=1)
    log_q_all = torch.cat([torch.where(valid, lq, torch.zeros_like(lq)), log_q_tail[:, None]], dim=1)
    log_p_all = torch.log(p_all.clamp(min=1e-30))
    m = beta * p_all + (1.0 - beta) * q_all
    log_m = torch.log(m.clamp(min=1e-30))
    return beta * _plogp_ratio(p_all, log_p_all, log_m).sum(-1) + (1.0 - beta) * _plogp_ratio(q_all, log_q_all, log_m).sum(-1)


def chunked_objective(
    normed: torch.Tensor,
    head_weight: torch.Tensor,
    targets: dict[str, torch.Tensor],
    config: ObjectiveConfig,
    *,
    denominator: float,
    loss_scale: float = 1.0,
    backward: bool = True,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """The D2 loss over position chunks; returns (term sums / denominator, d loss / d normed).

    `normed` is [B, L, H] (post final norm) WITH or without a graph; it is
    detached here, the head runs on chunks of the flattened positions, each
    chunk's loss is backwarded into the detached copy and its [n, V] logits are
    freed, so the peak is one chunk.  `head_weight` must already be the text
    rows only ([text_vocab_rows, H]).  `targets` is flattened to [B*L, ...].
    """

    if head_weight.shape[0] > config.text_vocab_rows:
        raise TrainerError(f"head weight has {head_weight.shape[0]} rows; slice it to {config.text_vocab_rows} first")
    detached = normed.detach().requires_grad_(backward)
    flat = detached.reshape(-1, detached.shape[-1])
    total_positions = flat.shape[0]
    device = flat.device
    sums = {k: torch.zeros((), dtype=torch.float64, device=device) for k in ("q38", "k3", "ce", "eos", "ul", "total")}
    counts = {"q38_positions": 0, "k3_positions": 0, "eos_positions": 0, "ul_candidates": 0}
    tdev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
    for start in range(0, total_positions, max(1, int(config.head_chunk_positions))):
        stop = min(start + int(config.head_chunk_positions), total_positions)
        rows = slice(start, stop)
        logits = F.linear(flat[rows], head_weight)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        logq = torch.log_softmax(logits, dim=-1)
        del logits
        loss = torch.zeros((), dtype=torch.float32, device=device)
        next_ids = tdev["next_ids"][rows]
        next_valid = next_ids >= 0
        ce = -logq.gather(1, next_ids.clamp(min=0)[:, None]).squeeze(1)
        ce = torch.where(next_valid, ce, torch.zeros_like(ce))
        ce_w = tdev["ce_w"][rows]
        eos_w = tdev["eos_w"][rows]
        term_ce = (ce_w * ce).sum()
        term_eos = (eos_w * ce).sum()
        loss = loss + term_ce + term_eos
        counts["eos_positions"] += int((eos_w > 0).sum())
        q_w = tdev.get("q_w")
        if q_w is not None and bool((q_w[rows] > 0).any()):
            ids = tdev["q_ids"][rows]
            lp = tdev["q_lp"][rows]
            kl = bucket_forward_kl(logq, ids, lp)
            jsd_rows = tdev.get("jsd")
            if jsd_rows is not None and bool(jsd_rows[rows].any()):
                js = bucket_jsd(logq, ids, lp, config.jsd_beta)
                kl = torch.where(jsd_rows[rows], js, kl)
            term_q = (q_w[rows] * kl).sum()
            loss = loss + term_q
            sums["q38"] += term_q.detach().double()
            counts["q38_positions"] += int((q_w[rows] > 0).sum())
        k_w = tdev.get("k_w")
        if k_w is not None and bool((k_w[rows] > 0).any()):
            kl3 = bucket_forward_kl(logq, tdev["k_ids"][rows], tdev["k_lp"][rows])
            term_k = (k_w[rows] * kl3).sum()
            loss = loss + term_k
            sums["k3"] += term_k.detach().double()
            counts["k3_positions"] += int((k_w[rows] > 0).sum())
        ul_ids = tdev.get("ul_ids")
        if ul_ids is not None and config.w_ul > 0 and bool((ul_ids[rows] >= 0).any()):
            cand = ul_ids[rows]
            cand_valid = cand >= 0
            safe = torch.where(cand_valid, cand, torch.zeros_like(cand))
            lc = logq.gather(1, safe)
            one_minus = (-torch.expm1(lc)).clamp(min=1e-6)
            ul = torch.where(cand_valid, -torch.log(one_minus), torch.zeros_like(lc)).sum(-1)
            term_ul = config.w_ul * ul.sum()
            loss = loss + term_ul
            sums["ul"] += term_ul.detach().double()
            counts["ul_candidates"] += int(cand_valid.sum())
        sums["ce"] += term_ce.detach().double()
        sums["eos"] += term_eos.detach().double()
        sums["total"] += loss.detach().double()
        if backward and loss.requires_grad:
            (loss * (float(loss_scale) / float(denominator))).backward()
        del logq, loss
    out = {k: float(v.item()) / float(denominator) for k, v in sums.items()}
    out.update(counts)
    return out, (detached.grad if backward else None)


def unlikelihood_candidates(ids: torch.Tensor, ngram: int, max_candidates: int) -> torch.Tensor:
    """For each position t (predicting t+1): tokens c != ids[t+1] such that the
    n-gram (ids[t-n+2..t], c) already occurred earlier in the row.

    ids [B, L] (CPU, long) -> [B, L, max_candidates] long, -1 padded; the most
    recent candidates are kept.  The gold continuation is never a candidate:
    teacher-forced text that genuinely repeats is not punished for it.
    """

    batch, length = ids.shape
    out = torch.full((batch, length, max_candidates), -1, dtype=torch.long)
    context = ngram - 1
    rows = ids.tolist()
    for b, row in enumerate(rows):
        seen: dict[tuple[int, ...], list[int]] = {}
        for t in range(length):
            # register the n-gram ending at t (prefix row[t-context..t-1] -> row[t])
            if t >= context:
                prefix = tuple(row[t - context : t])
                followers = seen.setdefault(prefix, [])
                if row[t] in followers:
                    followers.remove(row[t])
                followers.append(row[t])
            if t + 1 >= length or t + 1 < context:
                continue
            key = tuple(row[t + 1 - context : t + 1])
            followers = seen.get(key)
            if not followers:
                continue
            gold = row[t + 1]
            picks = [c for c in reversed(followers) if c != gold][:max_candidates]
            for j, c in enumerate(picks):
                out[b, t, j] = c
    return out


# --------------------------------------------------------------------------
# The layerwise engine (streamed or resident).
# --------------------------------------------------------------------------


def normalize_device(device: str | torch.device) -> torch.device:
    """`cuda` -> `cuda:<current>` so a device name always equals `str(tensor.device)`."""

    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def place_bank_on_layers(bank: AdapterBank, layer_devices: list[str | torch.device]) -> None:
    """Move each layer's adapters to that layer's device (the resident/split engine)."""

    for index, sites in enumerate(bank.layer_sites):
        target = normalize_device(layer_devices[index])
        for site, parts in sites.items():
            for part, tensor in list(parts.items()):
                if tensor.device == target:
                    continue
                moved = tensor.detach().to(target).requires_grad_(tensor.requires_grad)
                parts[part] = moved
                bank.tensors[f"layers.{index}.{site}.{part}"] = moved


def hop(tensor: torch.Tensor, device: torch.device, force: bool = False) -> torch.Tensor:
    """Move between devices THROUGH THE HOST.  Never P2P (P40s corrupt silently;
    GPU0<->GPU4 is untested).  Differentiable: autograd routes the gradient back
    the same way.  `force` makes the host round trip even on one device (the
    parity arm's proof that the hop itself is bit-exact)."""

    if tensor.device == device and not force:
        return tensor
    if tensor.device.type == "cuda" and device.type == "cuda":
        return tensor.to("cpu").to(device)
    return tensor.to(device)


class LayerwiseEngine:
    """Forward under no_grad with host boundaries; backward layer by layer.

    mode "streamed": one buffer per layer type on `device`, weights loaded from
    the source for every use (twice per micro-batch: forward and backward).
    mode "resident": every layer built once on its `layer_devices[i]` (the split
    cell: e.g. 40 layers on the RTX 8000, 24 on a 3090), hops host-staged.
    """

    def __init__(
        self,
        source: Any,
        text_config: Any,
        *,
        mode: str,
        dtype: torch.dtype,
        layer_devices: list[str],
        head_device: str,
        bank: AdapterBank | None,
        text_vocab_rows: int,
        pin_boundaries: bool = True,
        prefetch: bool = True,
        force_host_hops: tuple[int, ...] = (),
    ) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        if mode not in ("streamed", "resident"):
            raise TrainerError(f"engine mode {mode!r}")
        self.source = source
        self.config = text_config
        self.mode = mode
        self.dtype = dtype
        self.prefix = f"{getattr(source, 'text_prefix', DEFAULT_TEXT_PREFIX)}.layers"
        self.num_layers = int(text_config.num_hidden_layers)
        self.layer_types = [str(t) for t in text_config.layer_types[: self.num_layers]]
        if len(layer_devices) == 1:
            layer_devices = layer_devices * self.num_layers
        if len(layer_devices) != self.num_layers:
            raise TrainerError(f"{len(layer_devices)} layer devices for {self.num_layers} layers")
        self.layer_devices = [normalize_device(d) for d in layer_devices]
        if mode == "streamed" and len(set(self.layer_devices)) != 1:
            raise TrainerError("the streamed engine runs on one device; use --engine split for two")
        self.head_device = normalize_device(head_device)
        self.bank = bank
        self.pin = bool(pin_boundaries) and torch.cuda.is_available()
        self.load_seconds = 0.0
        self.layer_loads = 0
        self.force_host_hops = frozenset(int(i) for i in force_host_hops)
        self.bytes_start = getattr(source, "bytes_read", 0)

        embed_name = f"{getattr(source, 'text_prefix', DEFAULT_TEXT_PREFIX)}.embed_tokens.weight"
        #: The embedding stays on the HOST (3.26 GB at OMNI1 width): one lookup
        #: per micro-batch, then only [B, L, H] crosses PCIe.
        self.embed_weight = source.get(embed_name)
        head = source.get(head_weight_name(source))
        rows = min(int(text_vocab_rows), int(head.shape[0]))
        self.text_vocab_rows = rows
        self.head_weight = head[:rows].to(device=self.head_device, dtype=dtype).contiguous()
        del head
        self.norm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps).to(self.head_device, dtype)
        with torch.no_grad():
            self.norm.weight.copy_(source.get(f"{getattr(source, 'text_prefix', DEFAULT_TEXT_PREFIX)}.norm.weight"))
        self.norm.requires_grad_(False)
        #: Rotary is NOT cast: inv_freq stays fp32 exactly as the stock model's
        #: (tps_probe's stream-verify was bit-identical with this choice).
        self.rotary = {}
        for dev in sorted({str(d) for d in self.layer_devices}):
            self.rotary[dev] = Qwen3_5TextRotaryEmbedding(config=text_config).to(dev)

        self.expected_names: dict[str, list[str]] = {}
        self.buffers: dict[str, nn.Module] = {}
        self.layers: list[nn.Module] = []
        if mode == "streamed":
            first_index: dict[str, int] = {}
            for index, layer_type in enumerate(self.layer_types):
                first_index.setdefault(layer_type, index)
            for layer_type, index in first_index.items():
                with torch.device(self.layer_devices[0]):
                    layer = Qwen3_5DecoderLayer(text_config, index)
                layer = layer.to(device=self.layer_devices[0], dtype=dtype).eval().requires_grad_(False)
                if bank is not None:
                    install_site_hooks(layer, bank.plan[layer_type])
                self.buffers[layer_type] = layer
                self.expected_names[layer_type] = sorted(layer.state_dict().keys())
        else:
            for index, layer_type in enumerate(self.layer_types):
                with torch.device("meta"):
                    layer = Qwen3_5DecoderLayer(text_config, index)
                layer = layer.to_empty(device=self.layer_devices[index])
                layer = layer.to(dtype=dtype).eval().requires_grad_(False)
                self.expected_names.setdefault(layer_type, sorted(layer.state_dict().keys()))
                self._load_into(layer, index, self._fetch_state(index))
                if bank is not None:
                    install_site_hooks(layer, bank.plan[layer_type])
                    bank.bind(layer, index)
                self.layers.append(layer)
        for index, layer_type in enumerate(self.layer_types):
            found = sorted(n[len(f"{self.prefix}.{index}.") :] for n in source.names_under(f"{self.prefix}.{index}."))
            if found != self.expected_names[layer_type]:
                missing = sorted(set(self.expected_names[layer_type]) - set(found))
                extra = sorted(set(found) - set(self.expected_names[layer_type]))
                raise TrainerError(f"layer {index} ({layer_type}): source names differ; missing {missing[:4]}, extra {extra[:4]}")
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if (prefetch and mode == "streamed") else None
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._boundaries: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._contexts: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

    # -- weights ---------------------------------------------------------

    def _fetch_state(self, index: int) -> dict[str, torch.Tensor]:
        base = f"{self.prefix}.{index}."
        return {name: self.source.get(base + name) for name in self.expected_names[self.layer_types[index]]}

    def _load_into(self, layer: nn.Module, index: int, state: dict[str, torch.Tensor]) -> None:
        targets = dict(layer.named_parameters())
        targets.update(dict(layer.named_buffers()))
        with torch.no_grad():
            for name, value in state.items():
                target = targets[name]
                if tuple(value.shape) != tuple(target.shape):
                    raise TrainerError(f"layer {index} {name}: source {tuple(value.shape)} != module {tuple(target.shape)}")
                target.copy_(value)

    def _state_for(self, index: int) -> dict[str, torch.Tensor]:
        future = self._pending.pop(index, None)
        return future.result() if future is not None else self._fetch_state(index)

    def _prefetch(self, index: int | None) -> None:
        if self._executor is None or index is None or not (0 <= index < self.num_layers) or index in self._pending:
            return
        self._pending[index] = self._executor.submit(self._fetch_state, index)

    def layer(self, index: int, next_index: int | None = None) -> nn.Module:
        if self.mode == "resident":
            return self.layers[index]
        start = time.perf_counter()
        state = self._state_for(index)
        self._prefetch(next_index)
        layer = self.buffers[self.layer_types[index]]
        self._load_into(layer, index, state)
        del state
        if self.bank is not None:
            self.bank.bind(layer, index)
        self.load_seconds += time.perf_counter() - start
        self.layer_loads += 1
        return layer

    # -- per-shape context ------------------------------------------------

    def _context(self, batch: int, length: int) -> dict[str, dict[str, Any]]:
        key = (batch, length)
        if key in self._contexts:
            return self._contexts[key]
        from transformers.masking_utils import create_causal_mask

        contexts = {}
        for dev_name, rotary in self.rotary.items():
            dev = torch.device(dev_name)
            position_ids = torch.arange(length, device=dev).view(1, 1, -1).expand(4, batch, -1)
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
            probe = torch.zeros(batch, length, self.config.hidden_size, dtype=self.dtype, device=dev)
            cos_sin = rotary(probe, rope_position_ids)
            mask = create_causal_mask(
                config=self.config,
                inputs_embeds=probe,
                attention_mask=None,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            contexts[dev_name] = {"pe": cos_sin, "mask": mask, "text_pos": text_position_ids}
            del probe
        self._contexts = {key: contexts}  # one shape cached at a time
        return contexts

    def _boundary_buffers(self, batch: int, length: int) -> list[torch.Tensor]:
        key = (batch, length)
        if key not in self._boundaries:
            self._boundaries.clear()
            self._boundaries[key] = [
                torch.empty(batch, length, self.config.hidden_size, dtype=self.dtype, device="cpu", pin_memory=self.pin)
                for _ in range(self.num_layers)
            ]
        return self._boundaries[key]

    def _run(self, layer: nn.Module, index: int, hidden: torch.Tensor, contexts: dict[str, dict[str, Any]]) -> torch.Tensor:
        ctx = contexts[str(hidden.device)]
        mask = None if self.layer_types[index] == "linear_attention" else ctx["mask"]
        return layer(
            hidden,
            position_embeddings=ctx["pe"],
            attention_mask=mask,
            position_ids=ctx["text_pos"],
            past_key_values=None,
            use_cache=False,
        )

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.to("cpu"), self.embed_weight).to(device=self.layer_devices[0], dtype=self.dtype)

    # -- the step --------------------------------------------------------

    def forward_hidden(self, ids: torch.Tensor, *, store_boundaries: bool) -> torch.Tensor:
        batch, length = ids.shape
        contexts = self._context(batch, length)
        bounds = self._boundary_buffers(batch, length) if store_boundaries else None
        hidden = self.embed(ids)
        self._prefetch(0)
        with torch.no_grad():
            for index in range(self.num_layers):
                hidden = hop(hidden, self.layer_devices[index], force=index in self.force_host_hops)
                if bounds is not None:
                    bounds[index].copy_(hidden, non_blocking=self.pin)
                layer = self.layer(index, index + 1)
                hidden = self._run(layer, index, hidden, contexts)
        return hop(hidden, self.head_device)

    def forward_backward(
        self,
        ids: torch.Tensor,
        targets: dict[str, torch.Tensor],
        objective: ObjectiveConfig,
        *,
        denominator: float,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        batch, length = ids.shape
        contexts = self._context(batch, length)
        hidden = self.forward_hidden(ids, store_boundaries=True)
        bounds = self._boundary_buffers(batch, length)
        last = hidden.detach().requires_grad_(True)
        with torch.enable_grad():
            normed = self.norm(last)
        parts, grad_normed = chunked_objective(
            normed, self.head_weight, targets, objective, denominator=denominator, loss_scale=loss_scale, backward=True
        )
        normed.backward(grad_normed)
        grad = last.grad
        del normed, last, hidden, grad_normed
        self._prefetch(self.num_layers - 1)
        for index in reversed(range(self.num_layers)):
            device = self.layer_devices[index]
            layer = self.layer(index, index - 1)
            #: copy=True even on the host: a CPU-resident layer would otherwise
            #: get the pinned boundary buffer ITSELF as its leaf, and its .grad
            #: would leak into the next micro-batch.
            x = bounds[index].to(device, non_blocking=False, copy=True).requires_grad_(True)
            grad = hop(grad, device, force=index + 1 in self.force_host_hops)
            with torch.enable_grad():
                y = self._run(layer, index, x, contexts)
            y.backward(grad)
            grad = x.grad
            del x, y
        del grad
        return parts

    def streamed_bytes(self) -> int:
        return int(getattr(self.source, "bytes_read", 0) - self.bytes_start)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# The stock full-load reference (parity arm only).
# --------------------------------------------------------------------------


def load_reference(model_dir: Path, dtype: torch.dtype, device: str, attention_implementation: str):
    """Stock transformers load of the same checkpoint: (text model, head rows, facts)."""

    import transformers

    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    declared = str(config["architectures"][0])
    loader = getattr(transformers, declared, None)
    if loader is None:
        raise TrainerError(f"stock transformers has no {declared!r}")
    model = loader.from_pretrained(str(model_dir), dtype=dtype, attn_implementation=attention_implementation)
    inner = getattr(model, "model", None)
    text = getattr(inner, "language_model", None) or inner
    visual = getattr(inner, "visual", None)
    if visual is not None:
        inner.visual = None
    text = text.to(device).eval().requires_grad_(False)
    head = model.lm_head.weight.detach().to(device)
    return model, text, head


def reference_forward_backward(
    text_model: nn.Module,
    head_weight: torch.Tensor,
    bank: AdapterBank,
    ids: torch.Tensor,
    targets: dict[str, torch.Tensor],
    objective: ObjectiveConfig,
    *,
    denominator: float,
) -> dict[str, float]:
    for index, layer in enumerate(text_model.layers):
        install_site_hooks(layer, bank.plan[bank.layer_types[index]])
        bank.bind(layer, index)
    device = head_weight.device
    with torch.enable_grad():
        normed = text_model(input_ids=ids.to(device), use_cache=False).last_hidden_state
    rows = min(objective.text_vocab_rows, head_weight.shape[0])
    parts, grad = chunked_objective(normed, head_weight[:rows], targets, objective, denominator=denominator)
    normed.backward(grad)
    return parts


# --------------------------------------------------------------------------
# Teacher targets.  Both formats are their own tools' contracts and are read
# through those tools:
#   * the Q38 caches (Q-text, Q-long, self-gen and on-policy rescoring) are
#     qtext_teacher_cache.py caches: <root>/<teacher>/<corpus_sha>/ with a
#     sealed manifest.json and shard-NNNNN/ dirs of stored positions
#     (ids.u32, pos.u32, topk_ids.i32, topk_logprobs.bf16, lse.f32, shard.json
#     with a row table); position t carries the teacher's distribution for t+1;
#   * the K3 targets are k3_teacher_cache.py xtok output: xtok_target.safetensors
#     with one entry per aligned pair (q_file, q_row, q_pos, qwen_ids[., 32] with
#     -1 = no identical-byte Qwen token in [0, 248,320), k3_logprobs[., 32]).
# --------------------------------------------------------------------------


class QTextCache:
    """Row access to a SEALED qtext_teacher_cache cache, memory-mapped.

    The input ids come from the cache's own corpus file (path and sha256 in the
    manifest; a moved or edited corpus is a refusal), so positions the cache did
    not store (right padding, VL placeholders) still have their ids.  The ids the
    cache stored are checked against the corpus on every row read.
    """

    def __init__(self, cache_dir: Path, *, corpus_path: Path | None = None, verify_corpus: bool = True) -> None:
        q = load_peer_module("qtext_teacher_cache")
        self.dir = Path(cache_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise TrainerError(f"{self.dir} has no sealed manifest.json (the cache is not finished)")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != q.MANIFEST_SCHEMA:
            raise TrainerError(f"{manifest_path}: schema {self.manifest.get('schema')!r} != {q.MANIFEST_SCHEMA}")
        if not (self.manifest.get("all_finite") and self.manifest.get("all_structural_ok")):
            raise TrainerError(f"{manifest_path}: the cache is not all finite / structurally ok")
        if not self.manifest["teacher"].get("equal_before_and_after_every_session"):
            raise TrainerError(f"{manifest_path}: the teacher digest moved during the cache build")
        self.manifest_sha256 = file_sha256(manifest_path)
        config = self.manifest["cache_config"]
        self.top_k = int(config["top_k"])
        self.vocab_rows = int(config["vocab_rows"])
        corpus = self.manifest["corpus"]
        if corpus.get("kind") != "text":
            raise TrainerError(f"{self.dir}: corpus kind {corpus.get('kind')!r}; D2 trains on text caches (VL rows are D3d's)")
        path = Path(corpus_path) if corpus_path else Path(corpus["path"])
        if verify_corpus and file_sha256(path) != corpus["sha256"]:
            raise TrainerError(f"corpus {path} does not digest to the manifest's {corpus['sha256'][:16]}")
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            self.corpus_ids = handle.get_tensor(corpus.get("ids_key", "input_ids")).to(torch.int64)
            keys = set(handle.keys())
            #: the distill corpus (distill_corpus_build, k3_corpus_build layout) states its padding in-band as
            #: `real_lengths`; qtext_teacher_cache reads only `lengths`, so a cache over such a file treats the
            #: separator padding as real text.  The corpus is digest-checked above, so its own statement wins.
            real_lengths = (handle.get_tensor("real_lengths").to(torch.int64).reshape(-1)
                            if "real_lengths" in keys and "lengths" not in keys else None)
        self.seq_len = int(self.corpus_ids.shape[1])
        if real_lengths is not None and real_lengths.numel() != self.corpus_ids.shape[0]:
            raise TrainerError(f"corpus {path}: real_lengths has {real_lengths.numel()} rows, input_ids {self.corpus_ids.shape[0]}")
        self.index: dict[int, tuple[str, int, int, int]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.rows_cut_to_real_lengths = 0
        for entry in self.manifest["shards"]:
            shard_dir = self.dir / entry["shard"]
            if file_sha256(shard_dir / "shard.json") != entry["shard_json_sha256"]:
                raise TrainerError(f"{shard_dir}/shard.json does not match the manifest")
            record = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
            self.records[entry["shard"]] = record
            for row in record["row_table"]:
                r, length = int(row["row"]), int(row["length"])
                if real_lengths is not None and int(real_lengths[r]) < length:
                    if int(real_lengths[r]) < 1:
                        raise TrainerError(f"corpus {path}: row {r} has real_lengths {int(real_lengths[r])}")
                    length = int(real_lengths[r])
                    self.rows_cut_to_real_lengths += 1
                self.index[r] = (entry["shard"], int(row["offset"]), int(row["count"]), length)
        self.rows = len(self.index)
        self.length_source = "corpus real_lengths" if real_lengths is not None else "cache row_table"
        if self.rows_cut_to_real_lengths:
            log(f"{self.dir}: {self.rows_cut_to_real_lengths} rows cut to the corpus's real_lengths (the cache stored padding)")
        self._maps: dict[str, dict[str, np.memmap]] = {}

    def _arrays(self, shard: str) -> dict[str, np.memmap]:
        if shard not in self._maps:
            record = self.records[shard]
            p, k = int(record["positions"]), int(record["top_k"])
            base = self.dir / shard
            self._maps[shard] = {
                "ids": np.memmap(base / "ids.u32", dtype="<u4", mode="r", shape=(p,)),
                "pos": np.memmap(base / "pos.u32", dtype="<u4", mode="r", shape=(p,)),
                "topk_ids": np.memmap(base / "topk_ids.i32", dtype="<i4", mode="r", shape=(p, k)),
                "topk_logprobs": np.memmap(base / "topk_logprobs.bf16", dtype="<i2", mode="r", shape=(p, k)),
            }
        return self._maps[shard]

    def row(self, row: int) -> dict[str, Any]:
        if row not in self.index:
            raise TrainerError(f"row {row} is not in cache {self.dir}")
        shard, offset, count, length = self.index[row]
        arrays = self._arrays(shard)
        pos = torch.from_numpy(np.array(arrays["pos"][offset : offset + count]).astype(np.int64))
        cached_ids = torch.from_numpy(np.array(arrays["ids"][offset : offset + count]).astype(np.int64))
        ids = self.corpus_ids[row]
        if count and not torch.equal(ids[pos], cached_ids):
            raise TrainerError(f"cache {self.dir} row {row}: stored ids differ from the corpus ids")
        topk_ids = torch.full((self.seq_len, self.top_k), -1, dtype=torch.long)
        topk_lp = torch.full((self.seq_len, self.top_k), float("-inf"), dtype=torch.float32)
        topk_ids[pos] = torch.from_numpy(np.array(arrays["topk_ids"][offset : offset + count]).astype(np.int64))
        raw = np.array(arrays["topk_logprobs"][offset : offset + count])
        topk_lp[pos] = torch.from_numpy(raw).view(torch.bfloat16).float()
        stored = torch.zeros(self.seq_len, dtype=torch.bool)
        stored[pos] = True
        stored[length:] = False  # a row cut to the corpus's real_lengths keeps no target on its padding
        return {"ids": ids, "length": length, "stored": stored, "topk_ids": topk_ids, "topk_logprobs": topk_lp}


class XtokTargets:
    """k3_teacher_cache.py xtok output, indexed by Qwen row for one Qwen corpus file."""

    REQUIRED = ("q_file", "q_row", "q_pos", "qwen_ids", "k3_logprobs")

    def __init__(self, path: Path, qwen_file: str) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        with safe_open(str(self.path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            missing = [k for k in self.REQUIRED if k not in handle.keys()]
            if missing:
                raise TrainerError(f"{self.path}: xtok target lacks {missing}")
            tensors = {k: handle.get_tensor(k) for k in self.REQUIRED}
        files = json.loads(metadata.get("qwen_files", "[]"))
        if qwen_file not in files:
            raise TrainerError(f"{self.path}: qwen file {qwen_file!r} not among {files}")
        keep = tensors["q_file"].long() == files.index(qwen_file)
        self.metadata = dict(metadata)
        self.qwen_file = qwen_file
        self.sha256 = file_sha256(self.path)
        self.top_k = int(tensors["qwen_ids"].shape[1])
        rows = tensors["q_row"].long()[keep]
        pos = tensors["q_pos"].long()[keep]
        ids = tensors["qwen_ids"][keep].long()
        lp = tensors["k3_logprobs"][keep].float()
        self.by_row: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for r in torch.unique(rows).tolist():
            sel = rows == r
            self.by_row[int(r)] = (pos[sel], ids[sel], lp[sel])
        self.pairs = int(keep.sum())

    def verify_against(self, cache: QTextCache, *, unverified_ok: bool = False, boundary_map: Path | None = None) -> dict[str, Any]:
        """Prove that (q_row, q_pos) index THIS cache's corpus before any row is joined.

        The join is by file stem only, so a rebuilt corpus under the same name
        would put K3 distributions on unrelated positions with every value still
        finite.  Provenance, first that applies:
          1. metadata `qwen_corpus_sha256` ({qwen_file: sha} or one sha) equals
             the cache manifest's corpus sha256;
          2. the chain k3_teacher_cache.py xtok already writes: xtok_receipt.json
             beside this file records this file's sha and the boundary map's
             (path, sha); the boundary map's `ids_files_sha256[qwen_file]` is the
             Qwen ids file sha, which must equal the cache's corpus sha;
          3. otherwise refused, unless the data config says `xtok_unverified_ok`.
        Always, structurally: every pair of a cached row sits at a position that
        predicts a real token of that row (0 <= q_pos, q_pos + 1 < length), which
        is what the producer's aligned_pairs() guarantees on the right corpus.
        """

        expected = str(cache.manifest["corpus"]["sha256"])
        bad, in_cache = [], 0
        for r, (pos, _, _) in self.by_row.items():
            if r not in cache.index:
                continue
            in_cache += 1
            length = cache.index[r][3]
            if pos.numel() and (int(pos.min()) < 0 or int(pos.max()) + 1 >= length):
                bad.append((r, int(pos.min()), int(pos.max()), length))
        if bad:
            raise TrainerError(f"{self.path}: {len(bad)} rows have K3 pairs outside the row's predicting positions "
                               f"in {cache.dir} (first: row {bad[0][0]} q_pos [{bad[0][1]}, {bad[0][2]}] length {bad[0][3]}); "
                               "the xtok was not built on this corpus")
        structure = self.structural_check(cache.vocab_rows)
        record: dict[str, Any] = {"qwen_file": self.qwen_file, "expected_corpus_sha256": expected,
                                  "rows_in_cache": in_cache, "rows_not_in_cache": len(self.by_row) - in_cache,
                                  "structure": structure}
        declared = self.metadata.get("qwen_corpus_sha256")
        if declared:
            try:
                table = json.loads(declared)
            except json.JSONDecodeError:
                table = declared
            got = table.get(self.qwen_file) if isinstance(table, dict) else table
            if got != expected:
                raise TrainerError(f"{self.path}: built on Qwen corpus {str(got)[:16]} but the Q-text cache {cache.dir} "
                                   f"is over {expected[:16]}")
            record.update(method="metadata", corpus_sha256=got)
            return record
        receipt_path = self.path.parent / "xtok_receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("target") or {}).get("sha256") != self.sha256:
                raise TrainerError(f"{receipt_path} is not the receipt of {self.path} (target sha differs)")
            recorded = receipt.get("boundary_map") or {}
            bm_path = Path(boundary_map) if boundary_map else Path(str(recorded.get("path", "")))
            if not bm_path.is_file():
                raise TrainerError(f"{receipt_path}: boundary map {bm_path} is missing (give xtok_boundary_map if it moved)")
            bm_sha = file_sha256(bm_path)
            if bm_sha != recorded.get("sha256"):
                raise TrainerError(f"boundary map {bm_path} digests to {bm_sha[:16]}, not the {str(recorded.get('sha256'))[:16]} "
                                   f"the xtok was built from")
            ids_sha = (json.loads(bm_path.read_text(encoding="utf-8")).get("ids_files_sha256") or {}).get(self.qwen_file)
            if ids_sha is None:
                raise TrainerError(f"boundary map {bm_path} records no ids file sha for {self.qwen_file!r}")
            if ids_sha != expected:
                raise TrainerError(f"{self.path}: its boundary map pairs Qwen file {self.qwen_file!r} sha {ids_sha[:16]}, "
                                   f"but the Q-text cache {cache.dir} is over corpus {expected[:16]}")
            record.update(method="receipt_chain", corpus_sha256=ids_sha, xtok_receipt_sha256=file_sha256(receipt_path),
                          boundary_map={"path": str(bm_path), "sha256": bm_sha})
            return record
        if not unverified_ok:
            raise TrainerError(f"{self.path}: no proof that it was built on the Q-text corpus {expected[:16]} (no "
                               "qwen_corpus_sha256 metadata, no xtok_receipt.json beside it); set xtok_unverified_ok "
                               "in the data config to accept it anyway (receipted)")
        log(f"WARNING {self.path}: Qwen corpus provenance unverified (xtok_unverified_ok); structural check only")
        record.update(method="unverified (data-config xtok_unverified_ok)", corpus_sha256=None)
        return record

    def structural_check(self, vocab_rows: int) -> dict[str, Any]:
        """What bucket_terms assumes of a target, checked over every pair (functional refusals):
        valid ids are Qwen TEXT rows [0, vocab_rows) -- k3_teacher_cache maps into [0, 248,320) only, and an id
        at or past the student's sliced head would index past log q; valid ids are unique per position (a
        repeated id would count the student's probability twice); the K3 logprobs at valid ids are finite."""

        ids = torch.cat([v[1] for v in self.by_row.values()]) if self.by_row else torch.zeros(0, self.top_k, dtype=torch.long)
        lp = torch.cat([v[2] for v in self.by_row.values()]) if self.by_row else torch.zeros(0, self.top_k)
        valid = ids >= 0
        if bool((ids < -1).any()):
            raise TrainerError(f"{self.path}: ids below -1 (the only negative id is -1, 'no identical-byte Qwen token')")
        top = int(ids.max()) if ids.numel() else -1
        if top >= int(vocab_rows):
            raise TrainerError(f"{self.path}: K3 target id {top} is outside the cache's text rows [0, {vocab_rows})")
        ordered = torch.sort(torch.where(valid, ids, torch.full_like(ids, -1)), dim=1).values
        repeats = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any(dim=1)
        if bool(repeats.any()):
            raise TrainerError(f"{self.path}: {int(repeats.sum())} positions repeat a valid Qwen id (the K3 byte map "
                               "is not injective there); the bucket KL would count the student's mass twice")
        if bool((~torch.isfinite(lp[valid])).any()):
            raise TrainerError(f"{self.path}: non-finite K3 logprob at a mapped id")
        mass = torch.where(valid, lp.exp(), torch.zeros_like(lp)).sum(dim=1)
        return {"pairs": int(ids.shape[0]), "max_id": top, "vocab_rows": int(vocab_rows),
                "positions_without_a_mapped_id": int((~valid.any(dim=1)).sum()),
                "matched_mass_mean": float(mass.mean()) if mass.numel() else None,
                "matched_mass_max": float(mass.max()) if mass.numel() else None}

    def row(self, row: int, seq_len: int) -> dict[str, torch.Tensor]:
        pos, ids, lp = self.by_row[row]
        out_ids = torch.full((seq_len, self.top_k), -1, dtype=torch.long)
        out_lp = torch.full((seq_len, self.top_k), float("-inf"), dtype=torch.float32)
        valid = torch.zeros(seq_len, dtype=torch.bool)
        out_ids[pos] = ids
        out_lp[pos] = lp
        valid[pos] = (ids >= 0).any(dim=1)
        return {"k3_qids": out_ids, "k3_logprobs": out_lp, "k3_valid": valid}


# --------------------------------------------------------------------------
# Data: sources, the deterministic mixture sampler and the batch builder.
# --------------------------------------------------------------------------


@dataclass
class DataSource:
    name: str
    kind: str  # "text" | "k3" | "onpolicy"
    cache: QTextCache
    xtok: XtokTargets | None = None
    row_ids: list[int] = field(default_factory=list)
    token_budget: float | None = None
    weight: float = 1.0
    fraction: float | None = None  # fixed share of rows once active (on-policy)
    start_after_tokens: float = 0.0

    @property
    def rows(self) -> int:
        return len(self.row_ids)


def load_data_config(path: Path) -> tuple[list[DataSource], dict[str, Any]]:
    """{"sources": [{"name", "kind": text|k3|onpolicy, "qtext_cache": DIR,
    "xtok": FILE (k3), "xtok_qwen_file": NAME (k3; default the corpus file stem),
    "xtok_boundary_map": FILE (k3; only if the map moved after the xtok build),
    "xtok_unverified_ok": bool (k3; accept an xtok without corpus provenance),
    "tokens"?, "weight"?, "fraction"?, "start_after_tokens"?, "corpus"?}, ...]}

    A K3 source is the rows of its Q-text cache that have aligned K3 pairs: those
    rows carry BOTH the Q38 anchor and the K3 term.  The xtok must be proven to
    index the cache's own corpus (XtokTargets.verify_against)."""

    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = []
    provenance: dict[str, dict[str, Any]] = {}
    for item in spec["sources"]:
        kind = item.get("kind", "text")
        if kind not in ("text", "k3", "onpolicy"):
            raise TrainerError(f"source {item.get('name')}: kind {kind!r}")
        cache = QTextCache(Path(item["qtext_cache"]), corpus_path=Path(item["corpus"]) if item.get("corpus") else None)
        xtok = None
        row_ids = sorted(cache.index)
        if kind == "k3":
            if not item.get("xtok"):
                raise TrainerError(f"source {item['name']}: kind k3 needs xtok")
            qwen_file = item.get("xtok_qwen_file") or Path(cache.manifest["corpus"]["path"]).stem
            xtok = XtokTargets(Path(item["xtok"]), qwen_file)
            provenance[item["name"]] = xtok.verify_against(
                cache, unverified_ok=bool(item.get("xtok_unverified_ok", False)),
                boundary_map=Path(item["xtok_boundary_map"]) if item.get("xtok_boundary_map") else None)
            row_ids = sorted(r for r in xtok.by_row if r in cache.index)
            if not row_ids:
                raise TrainerError(f"source {item['name']}: no xtok row is in the Q-text cache")
        sources.append(DataSource(
            name=item["name"], kind=kind, cache=cache, xtok=xtok, row_ids=row_ids,
            token_budget=float(item["tokens"]) if item.get("tokens") is not None else None,
            weight=float(item.get("weight", 1.0)),
            fraction=float(item["fraction"]) if item.get("fraction") is not None else None,
            start_after_tokens=float(item.get("start_after_tokens", 0.0)),
        ))
    lengths = {s.cache.seq_len for s in sources}
    if len(lengths) != 1:
        raise TrainerError(f"every source of one run must share seq_len; got {sorted(lengths)} (run the long rung separately)")
    tops = {s.cache.top_k for s in sources}
    if len(tops) != 1:
        raise TrainerError(f"every Q38 source must share top_k; got {sorted(tops)}")
    identity = {
        "config_sha256": file_sha256(Path(path)),
        "qtext_tool": peer_identity("qtext_teacher_cache"),
        "sources": [{"name": s.name, "kind": s.kind, "qtext_manifest_sha256": s.cache.manifest_sha256,
                     "corpus_sha256": s.cache.manifest["corpus"]["sha256"], "rows": s.rows,
                     "length_source": s.cache.length_source, "rows_cut_to_real_lengths": s.cache.rows_cut_to_real_lengths,
                     "xtok_sha256": None if s.xtok is None else s.xtok.sha256,
                     "xtok_pairs": None if s.xtok is None else s.xtok.pairs,
                     "xtok_provenance": provenance.get(s.name)} for s in sources],
    }
    return sources, identity


def data_content_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """The part of a data identity that decides WHICH targets a step trains on:
    the data config's bytes and every source's manifest / corpus / xtok shas.
    The Q-text tool's own file sha is left out (a reader fix is not new data)."""

    return {k: v for k, v in identity.items() if k != "qtext_tool"}


class MixtureSampler:
    """Deterministic row order over several sources; its state is two counters.

    Row g of the run comes from the source picked by u_g = U(seed, g) against
    the weights in force when g is drawn: a fixed-fraction source (on-policy)
    takes its fraction once tokens_seen >= start_after_tokens; the others share
    the rest in proportion to their REMAINING token budgets (or weight when no
    budget is set).  Inside a source rows follow a seeded permutation per epoch.
    The state is (global_rows, per-source consumed), so a resumed run draws
    exactly the rows the uninterrupted one would have.
    """

    def __init__(self, sources: list[DataSource], seed: int) -> None:
        self.sources = sources
        self.seed = int(seed)
        self.global_rows = 0
        self.consumed = {s.name: 0 for s in sources}
        self._perm_cache: dict[tuple[str, int], torch.Tensor] = {}

    def _perm(self, source: DataSource, epoch: int) -> torch.Tensor:
        key = (source.name, epoch)
        if key not in self._perm_cache:
            generator = torch.Generator().manual_seed(stable_seed(self.seed, source.name, epoch))
            self._perm_cache = {k: v for k, v in self._perm_cache.items() if k[0] != source.name}
            self._perm_cache[key] = torch.randperm(source.rows, generator=generator)
        return self._perm_cache[key]

    def _weights(self, tokens_seen: float) -> list[float]:
        seq = self.sources[0].cache.seq_len
        fixed = [(i, s.fraction) for i, s in enumerate(self.sources) if s.fraction is not None and tokens_seen >= s.start_after_tokens]
        fixed_total = sum(f for _, f in fixed)
        rest = []
        for i, s in enumerate(self.sources):
            if s.fraction is not None:
                continue
            if tokens_seen < s.start_after_tokens:
                rest.append((i, 0.0))
            elif s.token_budget is not None:
                remaining = max(s.token_budget - self.consumed[s.name] * seq, 0.0)
                rest.append((i, remaining if remaining > 0 else 1e-9 * s.weight))
            else:
                rest.append((i, s.weight))
        rest_total = sum(w for _, w in rest)
        weights = [0.0] * len(self.sources)
        for i, f in fixed:
            weights[i] = f
        for i, w in rest:
            weights[i] = (1.0 - fixed_total) * (w / rest_total) if rest_total > 0 else 0.0
        return weights

    def next_rows(self, count: int, tokens_seen: float) -> list[tuple[int, int]]:
        """(source index, cache row id) for the next count rows."""

        picks = []
        for _ in range(count):
            weights = self._weights(tokens_seen)
            u = random.Random(stable_seed(self.seed, "row", self.global_rows)).random() * sum(weights)
            acc, choice = 0.0, max(i for i, w in enumerate(weights) if w > 0)
            for i, w in enumerate(weights):
                acc += w
                if u < acc and w > 0:
                    choice = i
                    break
            source = self.sources[choice]
            n = self.consumed[source.name]
            epoch, offset = divmod(n, source.rows)
            picks.append((choice, source.row_ids[int(self._perm(source, epoch)[offset])]))
            self.consumed[source.name] = n + 1
            self.global_rows += 1
        return picks

    def state(self) -> dict[str, Any]:
        return {"global_rows": self.global_rows, "consumed": dict(self.consumed), "seed": self.seed}

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise TrainerError("sampler seed differs from the checkpoint")
        self.global_rows = int(state["global_rows"])
        self.consumed = {k: int(v) for k, v in state["consumed"].items()}


def build_batch(sources: list[DataSource], picks: list[tuple[int, int]], objective: ObjectiveConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rows -> (ids [B, L], flattened targets with the per-position term weights).

    KD weights sit on STORED positions (the teacher's distribution exists there,
    including the last real position); CE / EOS / unlikelihood only where the
    next token is a real token of the row (t + 1 < length).
    """

    rows = [sources[s].cache.row(r) for s, r in picks]
    ids = torch.stack([r["ids"] for r in rows])
    batch, length = ids.shape
    lengths = torch.tensor([r["length"] for r in rows])
    positions = torch.arange(length).unsqueeze(0)
    next_valid = (positions + 1) < lengths.unsqueeze(1)
    next_ids = torch.full_like(ids, -1)
    next_ids[:, :-1] = ids[:, 1:]
    next_ids = torch.where(next_valid, next_ids, torch.full_like(next_ids, -1))
    stored = torch.stack([r["stored"] for r in rows])
    q_w = torch.zeros(batch, length)
    k_w = torch.zeros(batch, length)
    jsd = torch.zeros(batch, length, dtype=torch.bool)
    k3_rows = [b for b, (s, _) in enumerate(picks) if sources[s].kind == "k3"]
    k_top = max((sources[picks[b][0]].xtok.top_k for b in k3_rows), default=1)
    k_ids = torch.full((batch, length, k_top), -1, dtype=torch.long)
    k_lp = torch.full((batch, length, k_top), float("-inf"))
    for b, (s, r) in enumerate(picks):
        source = sources[s]
        if source.kind == "k3":
            aligned = source.xtok.row(r, length)
            q_w[b] = torch.where(stored[b], objective.lambda_k3 * objective.k3_q38_anchor, 0.0)
            k_w[b] = torch.where(aligned["k3_valid"], objective.lambda_k3 * (1.0 - objective.k3_q38_anchor), 0.0)
            k_ids[b, :, : aligned["k3_qids"].shape[-1]] = aligned["k3_qids"]
            k_lp[b, :, : aligned["k3_logprobs"].shape[-1]] = aligned["k3_logprobs"]
        else:
            q_w[b] = torch.where(stored[b], objective.w_q38, 0.0)
            if source.kind == "onpolicy" and objective.onpolicy_divergence == "jsd":
                jsd[b] = stored[b]
    eos = torch.zeros_like(next_valid)
    for e in objective.eos_ids:
        eos |= next_ids == int(e)
    targets = {
        "next_ids": next_ids.reshape(-1),
        "ce_w": torch.where(next_valid, objective.w_ce, 0.0).reshape(-1).float(),
        "eos_w": torch.where(eos & next_valid, objective.w_eos, 0.0).reshape(-1).float(),
        "q_ids": torch.stack([r["topk_ids"] for r in rows]).reshape(batch * length, -1),
        "q_lp": torch.stack([r["topk_logprobs"] for r in rows]).reshape(batch * length, -1),
        "q_w": q_w.reshape(-1),
    }
    if bool(jsd.any()):
        targets["jsd"] = jsd.reshape(-1)
    if k3_rows:
        targets["k_ids"] = k_ids.reshape(batch * length, -1)
        targets["k_lp"] = k_lp.reshape(batch * length, -1)
        targets["k_w"] = k_w.reshape(-1)
    if objective.w_ul > 0:
        cand = unlikelihood_candidates(ids, objective.ul_ngram, objective.ul_max_candidates)
        cand = torch.where(next_valid.unsqueeze(-1), cand, torch.full_like(cand, -1))
        targets["ul_ids"] = cand.reshape(batch * length, -1)
    return ids, targets


# --------------------------------------------------------------------------
# Optimisation plumbing: loss scaler (fp16 split cell), LR schedule, DP reduce.
# --------------------------------------------------------------------------


class DynamicLossScaler:
    """tps_probe's hand-rolled scaler, init 2^10 for the split cell (design G5 / P7).

    Halve and skip on a non-finite gradient, double after `growth_interval` clean
    steps; the counters are explicit because the skip rate is a ruling input.
    """

    def __init__(self, initial_scale: float = 2.0**10, growth_interval: int = 200) -> None:
        self.scale = float(initial_scale)
        self.growth_interval = int(growth_interval)
        self.clean_streak = 0
        self.skipped_steps = 0
        self.growths = 0
        self.backoffs = 0

    def unscale_and_check(self, parameters: list[torch.Tensor]) -> bool:
        inverse = 1.0 / self.scale
        found: dict[torch.device, torch.Tensor] = {}
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                parameter.grad.mul_(inverse)
                device = parameter.grad.device
                if device not in found:
                    found[device] = torch.zeros((), dtype=torch.float32, device=device)
                found[device] += (~torch.isfinite(parameter.grad).all()).float()
        #: one read per device, not one synchronisation per adapter tensor
        finite = all(float(value.item()) == 0.0 for value in found.values())
        if finite:
            self.clean_streak += 1
            if self.clean_streak >= self.growth_interval:
                self.scale *= 2.0
                self.clean_streak = 0
                self.growths += 1
        else:
            self.skipped_steps += 1
            self.clean_streak = 0
            self.scale = max(self.scale * 0.5, 1.0)
            self.backoffs += 1
        return finite

    def state(self) -> dict[str, Any]:
        return {"scale": self.scale, "growth_interval": self.growth_interval, "clean_streak": self.clean_streak,
                "skipped_steps": self.skipped_steps, "growths": self.growths, "backoffs": self.backoffs}

    def load_state(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup then cosine to `min_ratio`; `step` is the 0-based step about to run."""

    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class HostAllReduce:
    """Sum gradients across DP ranks THROUGH HOST RAM (gloo on CPU tensors).

    One flat pinned buffer per rank; device grads are copied in, all-reduced by
    gloo, and copied back.  No NCCL, no P2P: the design forbids P2P everywhere.
    """

    def __init__(self, parameters: list[torch.Tensor]) -> None:
        import torch.distributed as dist

        self.dist = dist
        self.parameters = parameters
        self.numel = sum(p.numel() for p in parameters)
        pin = torch.cuda.is_available() and parameters and parameters[0].is_cuda
        self.buffer = torch.zeros(self.numel, dtype=torch.float32, pin_memory=bool(pin))
        self.seconds: list[float] = []

    def __call__(self) -> None:
        start = time.perf_counter()
        offset = 0
        with torch.no_grad():
            for p in self.parameters:
                n = p.numel()
                if p.grad is None:
                    self.buffer[offset : offset + n].zero_()
                else:
                    self.buffer[offset : offset + n].copy_(p.grad.reshape(-1))
                offset += n
            self.dist.all_reduce(self.buffer, op=self.dist.ReduceOp.SUM)
            offset = 0
            for p in self.parameters:
                n = p.numel()
                chunk = self.buffer[offset : offset + n].view_as(p).to(p.device)
                if p.grad is None:
                    p.grad = chunk.clone()
                else:
                    p.grad.copy_(chunk)
                offset += n
        self.seconds.append(time.perf_counter() - start)


# --------------------------------------------------------------------------
# Checkpoints: rolling safety (adapters + optimizer + state, keep N), rungs and
# the final adapter delta.
# --------------------------------------------------------------------------


def save_adapters(path: Path, bank: AdapterBank, metadata: dict[str, Any]) -> str:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(bank.state_dict(), str(tmp), metadata={"schema": ADAPTER_SCHEMA, "json": json.dumps(metadata, sort_keys=True, default=str)})
    os.replace(tmp, path)
    return file_sha256(path)


def load_adapters(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def save_safety_checkpoint(directory: Path, step: int, *, bank: AdapterBank, optimizer: torch.optim.Optimizer, state: dict[str, Any], keep: int) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"ckpt-{step:08d}"
    tmp = directory / f".tmp-ckpt-{step:08d}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    sha = {"adapters.safetensors": save_adapters(tmp / "adapters.safetensors", bank, {"step": step})}
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    sha["optimizer.pt"] = file_sha256(tmp / "optimizer.pt")
    sha["trainer_state.json"] = write_json_atomic(tmp / "trainer_state.json", state)
    write_json_atomic(tmp / "COMPLETE.json", {"schema": CHECKPOINT_SCHEMA, "step": step, "sha256": sha, "tool_version": TOOL_VERSION})
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    complete = sorted(p for p in directory.glob("ckpt-*") if (p / "COMPLETE.json").is_file())
    for old in complete[:-keep] if keep > 0 else []:
        shutil.rmtree(old)
    return final


def safety_checkpoint_problem(candidate: Path) -> str | None:
    """None when `candidate` is a complete safety checkpoint whose files match
    their COMPLETE.json digests; otherwise what is wrong with it."""

    candidate = Path(candidate)
    marker = candidate / "COMPLETE.json"
    if not marker.is_file():
        return "no COMPLETE.json (unfinished or not a safety checkpoint)"
    record = json.loads(marker.read_text(encoding="utf-8"))
    for name in ("adapters.safetensors", "optimizer.pt", "trainer_state.json"):
        if name not in record.get("sha256", {}):
            return f"COMPLETE.json records no digest for {name}"
    for name, digest in record["sha256"].items():
        if not (candidate / name).is_file():
            return f"{name} is missing"
        if file_sha256(candidate / name) != digest:
            return f"{name} does not match its COMPLETE.json digest"
    return None


def latest_safety_checkpoint(directory: Path) -> Path | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob("ckpt-*"), reverse=True):
        if not (candidate / "COMPLETE.json").is_file():
            continue
        problem = safety_checkpoint_problem(candidate)
        if problem is None:
            return candidate
        log(f"safety checkpoint {candidate} skipped: {problem}")
    return None


# --------------------------------------------------------------------------
# Commands.
# --------------------------------------------------------------------------


def objective_from_args(args: argparse.Namespace, text_vocab_rows: int) -> ObjectiveConfig:
    return ObjectiveConfig(
        w_q38=args.w_q38,
        lambda_k3=args.lambda_k3,
        k3_q38_anchor=args.k3_q38_anchor,
        w_ce=args.w_ce,
        w_eos=args.w_eos,
        w_ul=args.w_ul,
        ul_ngram=args.ul_ngram,
        ul_max_candidates=args.ul_max_candidates,
        onpolicy_divergence=args.onpolicy_divergence,
        jsd_beta=args.jsd_beta,
        eos_ids=tuple(int(x) for x in str(args.eos_ids).split(",") if x.strip()),
        head_chunk_positions=args.head_chunk_positions,
        text_vocab_rows=text_vocab_rows,
    )


def site_plan_from_args(args: argparse.Namespace) -> SitePlanConfig:
    return SitePlanConfig(
        rank=args.lora_rank,
        alpha=float(args.lora_alpha if args.lora_alpha else args.lora_rank),
        k3_rank=args.k3_lora_rank,
        k3_alpha=float(args.k3_lora_alpha if args.k3_lora_alpha else args.k3_lora_rank),
        trunk_ffn_width=args.trunk_ffn_width,
        gain=not args.no_k3_gain,
        gdn_sites=tuple(s for s in args.gdn_sites.split(",") if s),
        gqa_sites=tuple(s for s in args.gqa_sites.split(",") if s),
    )


def synthetic_parity_batch(batch: int, length: int, vocab_rows: int, k3_top: int, q_top: int, seed: int, eos_ids: tuple[int, ...], objective: ObjectiveConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ids with repeated motifs (so unlikelihood has candidates), EOS inserted,
    random teacher top-k (normalised logprobs), K3 targets on half the rows with
    some unmatched ids.  Every loss term is non-zero, so every gradient path of
    the objective is exercised by the parity arm."""

    g = torch.Generator().manual_seed(seed)
    upper = min(vocab_rows, DEFINED_VOCABULARY_ROWS)
    motif = torch.randint(0, upper, (batch, 16), generator=g)
    ids = motif.repeat(1, (length + 15) // 16)[:, :length].clone()
    noise = torch.rand(batch, length, generator=g) < 0.3
    ids = torch.where(noise, torch.randint(0, upper, (batch, length), generator=g), ids)
    for b in range(batch):
        for e in eos_ids:
            if e < vocab_rows:
                ids[b, int(torch.randint(1, length - 1, (1,), generator=g))] = int(e)
    n = batch * length
    q_ids = torch.stack([torch.randperm(upper, generator=g)[:q_top] for _ in range(n)])
    q_logits = torch.randn(n, q_top, generator=g) * 2.0
    q_lp = torch.log_softmax(q_logits, -1) + math.log(0.9)  # 10% teacher mass in the tail
    k_ids = torch.stack([torch.randperm(upper, generator=g)[:k3_top] for _ in range(n)])
    k_ids = torch.where(torch.rand(n, k3_top, generator=g) < 0.2, torch.full_like(k_ids, -1), k_ids)
    k_lp = torch.log_softmax(torch.randn(n, k3_top, generator=g), -1) + math.log(0.8)
    next_ids = torch.full_like(ids, -1)
    next_ids[:, :-1] = ids[:, 1:]
    next_valid = next_ids >= 0
    k3_rows = torch.zeros(batch, length, dtype=torch.bool)
    k3_rows[batch // 2 :] = True
    eos = torch.zeros_like(next_valid)
    for e in eos_ids:
        eos |= next_ids == int(e)
    q_w = torch.where(k3_rows, objective.lambda_k3 * objective.k3_q38_anchor, objective.w_q38) * next_valid
    k_w = torch.where(k3_rows & next_valid, objective.lambda_k3 * (1.0 - objective.k3_q38_anchor), 0.0)
    targets = {
        "next_ids": next_ids.reshape(-1),
        "ce_w": (next_valid.float() * objective.w_ce).reshape(-1),
        "eos_w": ((eos & next_valid).float() * objective.w_eos).reshape(-1),
        "q_ids": q_ids,
        "q_lp": q_lp,
        "q_w": q_w.reshape(-1).float(),
        "k_ids": k_ids,
        "k_lp": k_lp,
        "k_w": k_w.reshape(-1).float(),
        "ul_ids": unlikelihood_candidates(ids, objective.ul_ngram, objective.ul_max_candidates).reshape(n, -1),
    }
    return ids, targets


def collect_grads(bank: AdapterBank) -> dict[str, torch.Tensor]:
    out = {}
    for name in bank.trainable_names():
        grad = bank.tensors[name].grad
        out[name] = torch.zeros_like(bank.tensors[name], device="cpu", dtype=torch.float64) if grad is None else grad.detach().to("cpu", torch.float64)
    return out


def compare_grads(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows = []
    per_site: dict[str, dict[str, float]] = {}
    for name in sorted(reference):
        r, c = reference[name], candidate[name]
        diff = (c - r)
        ref_norm = float(r.norm())
        rel_l2 = float(diff.norm()) / ref_norm if ref_norm > 0 else (0.0 if float(diff.norm()) == 0 else float("inf"))
        ref_max = float(r.abs().max())
        max_rel = float(diff.abs().max()) / ref_max if ref_max > 0 else (0.0 if float(diff.abs().max()) == 0 else float("inf"))
        rows.append({"tensor": name, "ref_norm": ref_norm, "rel_l2": rel_l2, "max_abs_rel": max_rel})
        site = ".".join(name.split(".")[2:])
        entry = per_site.setdefault(site, {"tensors": 0, "max_rel_l2": 0.0, "max_abs_rel": 0.0, "min_ref_norm": float("inf")})
        entry["tensors"] += 1
        entry["max_rel_l2"] = max(entry["max_rel_l2"], rel_l2)
        entry["max_abs_rel"] = max(entry["max_abs_rel"], max_rel)
        entry["min_ref_norm"] = min(entry["min_ref_norm"], ref_norm)
    return rows, per_site


def command_grad_parity(args: argparse.Namespace) -> int:
    """Streamed (and optionally resident/split) LoRA grads vs a stock full-load autograd."""

    kernels = select_linear_attention_kernels(args.kernels)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    source = SafetensorsSource(Path(args.model))
    config = text_config_from_source(source, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = objective_from_args(args, text_rows)
    plan = build_site_plan(config, site_plan_from_args(args))
    layer_types = [str(t) for t in config.layer_types[: config.num_hidden_layers]]
    ids, targets = synthetic_parity_batch(args.batch, args.seq_len, text_rows, args.k3_top_k, args.q38_top_k, args.seed, objective.eos_ids, objective)
    denominator = float(args.batch * (args.seq_len - 1))
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "grad-parity",
        "argv": sys.argv, "model": str(args.model), "source_identity": source.identity(), "dtype": args.dtype,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "kernels": kernels, "batch": args.batch, "seq_len": args.seq_len,
        "objective": asdict(objective), "site_plan": {k: [asdict(s) for s in v] for k, v in plan.items()},
    }

    def make_bank(where: torch.device) -> AdapterBank:
        return AdapterBank(plan, layer_types, where, args.seed, b_init_std=args.lora_b_init_std, gain_jitter=args.gain_jitter)

    log("reference: stock full load + autograd")
    bank_ref = make_bank(device)
    receipt["adapter_census"] = bank_ref.census()
    start = time.perf_counter()
    model, text, head = load_reference(Path(args.model), dtype, str(device), args.attn_implementation)
    receipt["reference_kernel_facts"] = linear_attention_kernel_facts(next(l for l in text.layers if hasattr(l, "linear_attn")))
    ref_parts = reference_forward_backward(text, head, bank_ref, ids, targets, objective, denominator=denominator)
    synchronise(device)
    receipt["reference_seconds"] = round(time.perf_counter() - start, 3)
    ref_grads = collect_grads(bank_ref)
    del model, text, head, bank_ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    #: GATING arms run every layer on the parity device, so a difference can only
    #: come from the engine: "streamed" (the D2 upside engine) and "resident-hop"
    #: (the split-cell engine, all layers resident, with a forced host round trip
    #: at the middle boundary in both directions).  The optional "cross-device"
    #: arm puts the second half on another device (the real split shape); its
    #: difference also contains that device's own arithmetic, so it is a
    #: DIAGNOSTIC with its own bound, printed but never folded into the gate.
    half = config.num_hidden_layers // 2
    arms = {"streamed": ([str(device)], (), True)}
    if args.also_resident:
        arms["resident-hop"] = ([str(device)] * config.num_hidden_layers, (half,), True)
    if args.cross_device:
        arms["cross-device"] = ([str(device)] * half + [args.cross_device] * (config.num_hidden_layers - half), (), False)
    receipt["arms"] = {}
    worst = 0.0
    diagnostic_worst = 0.0
    for arm, (devices, forced, gating) in arms.items():
        log(f"arm {arm}: devices {sorted(set(devices))}")
        bank = make_bank(device)
        if arm != "streamed":
            place_bank_on_layers(bank, devices)
        start = time.perf_counter()
        engine = LayerwiseEngine(
            source, config, mode="streamed" if arm == "streamed" else "resident", dtype=dtype,
            layer_devices=devices, head_device=devices[-1], bank=bank, text_vocab_rows=text_rows,
            force_host_hops=forced,
        )
        receipt["arms"].setdefault(arm, {})["kernel_facts"] = linear_attention_kernel_facts(
            engine.buffers.get("linear_attention") if arm == "streamed" else next(l for l in engine.layers if hasattr(l, "linear_attn"))
        )
        parts = engine.forward_backward(ids, targets, objective, denominator=denominator)
        synchronise(device)
        seconds = time.perf_counter() - start
        grads = collect_grads(bank)
        rows, per_site = compare_grads(ref_grads, grads)
        loss_rel = abs(parts["total"] - ref_parts["total"]) / max(abs(ref_parts["total"]), 1e-30)
        arm_worst = max(max(v["max_abs_rel"] for v in per_site.values()), max(v["max_rel_l2"] for v in per_site.values()))
        if gating:
            worst = max(worst, arm_worst, loss_rel)
        else:
            diagnostic_worst = max(diagnostic_worst, arm_worst, loss_rel)
        print(f"ARM {arm} ({'gating' if gating else 'diagnostic'}; devices {sorted(set(devices))}; forced host hops {list(forced)}): "
              f"loss ref {ref_parts['total']:.9f} arm {parts['total']:.9f} rel {loss_rel:.3e}", flush=True)
        for term in ("q38", "k3", "ce", "eos", "ul"):
            print(f"  term {term:4s} ref {ref_parts[term]:.9f} arm {parts[term]:.9f}", flush=True)
        for site, entry in sorted(per_site.items()):
            print(f"  SITE {site:34s} tensors {entry['tensors']:3d} max_rel_l2 {entry['max_rel_l2']:.3e} max_abs_rel {entry['max_abs_rel']:.3e} min_ref_norm {entry['min_ref_norm']:.3e}", flush=True)
        receipt["arms"][arm].update({
            "devices": sorted(set(devices)), "forced_host_hops": list(forced), "gating": gating,
            "seconds": round(seconds, 3), "loss_ref": ref_parts, "loss_arm": parts,
            "loss_rel_diff": loss_rel, "per_site": per_site, "per_tensor": rows, "worst": arm_worst,
            "layer_loads": engine.layer_loads, "streamed_gb": round(engine.streamed_bytes() / 1e9, 3),
        })
        engine.close()
        del engine, bank
    receipt["threshold"] = args.threshold
    receipt["worst_relative_difference"] = worst
    receipt["pass"] = bool(worst <= args.threshold)
    if args.cross_device:
        receipt["cross_device_threshold"] = args.cross_device_threshold
        receipt["cross_device_worst"] = diagnostic_worst
        receipt["cross_device_within_bound"] = bool(diagnostic_worst <= args.cross_device_threshold)
        print(f"CROSS-DEVICE (diagnostic): worst {diagnostic_worst:.3e} vs bound {args.cross_device_threshold:.1e} "
              f"-> {'within' if receipt['cross_device_within_bound'] else 'OUTSIDE'}", flush=True)
    receipt["memory"] = device_memory_report()
    if args.receipt:
        sha = write_json_atomic(Path(args.receipt), receipt)
        log(f"receipt {args.receipt} sha256 {sha}")
    print(f"GRAD-PARITY {'PASS' if receipt['pass'] else 'FAIL'}: worst relative difference {worst:.3e} (threshold {args.threshold:.1e})", flush=True)
    return 0 if receipt["pass"] else 1


def command_source_check(args: argparse.Namespace) -> int:
    """Structural check of a REAL student source (OMNI1 / S5 via P1) on the host.

    Reads the source exactly as `train` would, builds the streamed engine with the
    D2 site plan (its constructor refuses any layer whose source names differ from
    the stock qwen3_5 decoder layer), prints the exact trainable census at that
    geometry, then runs a forward + layerwise backward of the first GDN and the
    first gated-GQA layer on a few random tokens and checks every adapter gradient
    is finite.  No GPU, no write anywhere but the receipt.
    """

    kernels = select_linear_attention_kernels(args.kernels)
    dtype = resolve_dtype(args.dtype)
    source = build_source(args)
    config = text_config_from_source(source, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = objective_from_args(args, text_rows)
    plan = build_site_plan(config, site_plan_from_args(args))
    layer_types = [str(t) for t in config.layer_types[: config.num_hidden_layers]]
    probe_layers = sorted({layer_types.index("linear_attention"), layer_types.index("full_attention")})
    bank = AdapterBank(plan, layer_types, args.device, args.seed, b_init_std=args.lora_b_init_std)
    census = bank.census()
    log(f"source {type(source).__name__}: {config.num_hidden_layers} layers, hidden {config.hidden_size}, "
        f"FFN {config.intermediate_size} (trunk {args.trunk_ffn_width}), vocab {config.vocab_size}, text rows {text_rows}")
    log(f"D2 trainable parameters at this geometry: {census['parameters_trainable']:,} in {census['tensors']} tensors")
    started = time.perf_counter()
    engine = LayerwiseEngine(source, config, mode="streamed", dtype=dtype, layer_devices=[args.device], head_device=args.device,
                             bank=bank, text_vocab_rows=text_rows, prefetch=False)
    #: run only the probe layers: the engine's own loop, restricted
    engine.num_layers_full = engine.num_layers
    sub_types = [layer_types[i] for i in probe_layers]
    engine.layer_types = sub_types
    engine.num_layers = len(probe_layers)
    original_fetch = engine._fetch_state
    engine._fetch_state = lambda i: original_fetch_mapped(i)  # noqa: E731

    def original_fetch_mapped(i: int) -> dict[str, torch.Tensor]:
        real = probe_layers[i]
        base = f"{engine.prefix}.{real}."
        return {name: source.get(base + name) for name in engine.expected_names[layer_types[real]]}

    bank_view_types = bank.layer_types
    bank.layer_types = sub_types
    bank.layer_sites = [bank.layer_sites[i] for i in probe_layers]
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, min(text_rows, DEFINED_VOCABULARY_ROWS), (1, args.seq_len), generator=g)
    _, targets = synthetic_parity_batch(1, args.seq_len, text_rows, 32, 64, args.seed, objective.eos_ids, objective)
    parts = engine.forward_backward(ids, targets, objective, denominator=float(args.seq_len - 1))
    bank.layer_types = bank_view_types
    grads = {n: bank.tensors[n].grad for n in bank.trainable_names() if bank.tensors[n].grad is not None}
    finite = all(bool(torch.isfinite(g_).all()) for g_ in grads.values())
    nonzero = sum(1 for g_ in grads.values() if float(g_.abs().max()) > 0)
    receipt = {
        "schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "source-check", "argv": sys.argv,
        "source_identity": source.identity(), "kernels": kernels, "dtype": args.dtype, "device": args.device,
        "geometry": {"layers": int(config.num_hidden_layers), "hidden": int(config.hidden_size), "ffn": int(config.intermediate_size),
                     "trunk_ffn_width": args.trunk_ffn_width, "vocab": int(config.vocab_size), "text_rows": text_rows,
                     "layer_types_count": {t: layer_types.count(t) for t in sorted(set(layer_types))}},
        "names_checked_layers": int(engine.num_layers_full), "probe_layers": probe_layers, "adapter_census": census,
        "probe_loss": parts, "probe_grad_tensors": len(grads), "probe_grad_nonzero": nonzero, "probe_grads_finite": finite,
        "seconds": round(time.perf_counter() - started, 2), "host_rss_gb": host_rss_gb(),
        "bytes_read_gb": round(getattr(source, "bytes_read", 0) / 1e9, 3),
    }
    ok = finite and len(grads) > 0
    receipt["pass"] = ok
    if args.receipt:
        log(f"receipt {args.receipt} sha256 {write_json_atomic(Path(args.receipt), receipt)}")
    print(f"SOURCE-CHECK {'PASS' if ok else 'FAIL'}: all {engine.num_layers_full} layers name-checked; probe layers {probe_layers} "
          f"fwd+bwd loss {parts['total']:.6f}; {len(grads)} adapter grads ({nonzero} non-zero) finite={finite}; "
          f"trainable {census['parameters_trainable']:,}", flush=True)
    return 0 if ok else 1


def run_config_digest(args: argparse.Namespace) -> str:
    """sha256 of the training-relevant arguments, with path arguments counted by
    CONTENT as well as by name (an edited data.json or a resealed plan at the
    same path is a different run).  Operational switches -- where to write, how
    often to checkpoint, the skip guard, the override flags -- are left out."""

    skip = {"resume", "out", "safety_dir", "max_steps_this_run", "receipt", "log_every", "safety_minutes", "safety_every_steps", "safety_keep",
            "dump_step_grads", "allow_config_change", "allow_identity_change", "max_consecutive_skips", "max_skip_fraction", "skip_window",
            "no_preregistration"}
    config: dict[str, Any] = {k: v for k, v in vars(args).items() if k not in skip and k != "func"}
    files = {}
    for key in ("data_config", "p1_plan"):
        value = getattr(args, key, None)
        if value:
            files[key] = file_sha256(Path(value))
    plugin = getattr(args, "source_plugin", None)
    if plugin:
        files["source_plugin"] = file_sha256(Path(plugin.rpartition(":")[0]))
    config["file_sha256"] = files
    return canonical_sha256(config)


def run_moves_weights(args: argparse.Namespace) -> bool:
    """True when an applied step can change an adapter: a LoRA lr, or a gain lr on a trained gain."""

    return float(args.lr) != 0.0 or (not args.freeze_k3_gain and float(args.gain_lr) != 0.0)


def preregistration_record(path: str | None, waiver: str | None = None, *, moves_weights: bool = True) -> dict[str, Any]:
    """The run's pre-registration, checked against the digest PINNED in the campaign's receipt module.

    DISTILL_DESIGN section 4: PREREGISTRATION_OMNI_STAGE_D_v1.md's sha is pinned in every training receipt
    (`stage2_training_receipt.py`, CAMPAIGN section 1.8; prereg section 2.8).  A document whose frozen body no
    longer hashes to the pin is refused before any step: a moved pre-registration is not the one the run claims
    to follow.  A run that can move a weight and names no document is refused too, unless the waiver
    (--no-preregistration REASON) says why; a run that moves nothing (lr = 0 rate cells) needs neither.
    """

    if path and waiver is not None:
        raise TrainerError("--preregistration and --no-preregistration are exclusive")
    if waiver is not None:
        if not waiver.strip():
            raise TrainerError("--no-preregistration needs a non-empty REASON (it is recorded in the receipt)")
        return {"pinned": False, "waived": True, "reason": waiver}
    if not path:
        if moves_weights:
            raise TrainerError("a run that moves weights must name --preregistration (prereg section 2.8: every "
                               "training receipt records the pin) or waive it with --no-preregistration REASON")
        return {"pinned": False, "waived": False, "reason": "lr = 0 and gain-lr = 0 (or gain frozen): no weight moves"}
    receipts = load_peer_module("stage2_training_receipt")
    try:
        observed = receipts.assert_preregistration_unedited(Path(path))
    except receipts.Stage2TrainingReceiptError as error:
        raise TrainerError(f"--preregistration {path}: {error}") from error
    return {"pinned": True, "path": str(path), "sha256": file_sha256(Path(path)), **observed,
            "pinned_by": peer_identity("stage2_training_receipt")}


def skip_guard_reason(consecutive: int, window: list[int], *, max_consecutive: int, max_fraction: float, window_size: int) -> str | None:
    """Why a run must halt on skipped steps, or None.  `window` holds 1 per
    skipped and 0 per applied step, newest last (only its last `window_size`
    entries count, and the fraction rule waits until the window is full)."""

    if max_consecutive > 0 and consecutive >= max_consecutive:
        return f"{consecutive} consecutive skipped steps (limit --max-consecutive-skips {max_consecutive})"
    if window_size > 0 and max_fraction < 1.0 and len(window) >= window_size:
        skipped = sum(window[-window_size:])
        if skipped > max_fraction * window_size:
            return (f"{skipped} of the last {window_size} steps skipped ({skipped / window_size:.3f} > "
                    f"--max-skip-fraction {max_fraction})")
    return None


def command_train(args: argparse.Namespace) -> int:
    if getattr(args, "registered_native_input_plan", None):
        raise TrainerError("registered native packets require a native-aware backend/objective; run native-input-preflight; no model or optimizer was loaded")
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    #: before anything loads: an unpinned run that can move weights is refused at zero cost
    preregistration = preregistration_record(args.preregistration, args.no_preregistration,
                                             moves_weights=run_moves_weights(args))
    if world > 1:
        dist.init_process_group(backend="gloo")
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    kernels = select_linear_attention_kernels(args.kernels)
    dtype = resolve_dtype(args.dtype)
    source = build_source(args)
    config = text_config_from_source(source, dtype, args.attn_implementation)
    text_rows = min(args.text_vocab_rows, int(config.vocab_size))
    objective = objective_from_args(args, text_rows)
    plan = build_site_plan(config, site_plan_from_args(args))
    layer_types = [str(t) for t in config.layer_types[: config.num_hidden_layers]]
    if args.engine == "streamed":
        device = devices[local_rank % len(devices)] if world > 1 else devices[0]
        layer_devices, head_device, mode = [device], device, "streamed"
    elif args.engine == "split":
        if world > 1:
            raise TrainerError("the split cell is one replica over two cards; it does not run data parallel")
        if len(devices) != 2:
            raise TrainerError("--engine split needs --devices first,second")
        first = int(args.split_layers) if args.split_layers else config.num_hidden_layers // 2
        layer_devices = [devices[0]] * first + [devices[1]] * (config.num_hidden_layers - first)
        head_device, mode, device = devices[1], "resident", devices[0]
    else:
        raise TrainerError(f"--engine {args.engine!r}")
    if torch.device(device).type == "cuda":
        #: FusedRMSNormGated is built on torch.cuda.current_device(); make that
        #: this rank's card before any layer exists
        torch.cuda.set_device(normalize_device(device))
    if args.engine == "split" and dtype == torch.float16 and args.initial_loss_scale <= 0:
        raise TrainerError("fp16 split needs a positive --initial-loss-scale (2^10 by design)")
    bank = AdapterBank(plan, layer_types, device if mode == "streamed" else head_device, args.seed, train_gain=not args.freeze_k3_gain)
    if mode == "resident":
        place_bank_on_layers(bank, layer_devices)
    if args.init_adapters:
        bank.load_state_dict(load_adapters(Path(args.init_adapters)))
    engine = LayerwiseEngine(source, config, mode=mode, dtype=dtype, layer_devices=layer_devices, head_device=head_device,
                             bank=bank, text_vocab_rows=text_rows, pin_boundaries=not args.no_pin, prefetch=not args.no_prefetch)
    sources, data_identity = load_data_config(Path(args.data_config))
    if sources[0].cache.seq_len != args.seq_len:
        raise TrainerError(f"--seq-len {args.seq_len} != cache seq_len {sources[0].cache.seq_len}")
    sampler = MixtureSampler(sources, args.seed)
    lora_params, gain_params = bank.parameter_groups()
    groups = [{"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay, "base_lr": args.lr}]
    if gain_params:
        groups.append({"params": gain_params, "lr": args.gain_lr, "weight_decay": 0.0, "base_lr": args.gain_lr})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    scaler = DynamicLossScaler(args.initial_loss_scale, args.loss_scale_growth_interval) if dtype == torch.float16 else None
    reducer = HostAllReduce(bank.trainable()) if world > 1 else None
    tokens_per_step = args.micro_batch * args.seq_len * args.accumulation * world
    total_steps = int(args.total_steps) if args.total_steps else max(1, int(math.ceil(float(args.token_budget) / tokens_per_step)))
    denominator = float(args.micro_batch * (args.seq_len - 1) * args.accumulation * world)
    config_digest = run_config_digest(args)
    source_identity = source.identity()
    source_identity_sha = canonical_sha256(source_identity)
    data_identity_sha = canonical_sha256(data_content_identity(data_identity))
    out = Path(args.out)
    safety_dir = Path(args.safety_dir)
    step, tokens_seen, history = 0, 0, []
    consecutive_skips, skip_window = 0, []
    segments: list[dict[str, Any]] = []
    resumed_from = None
    identity_override = None
    if args.resume:
        if args.resume == "auto":
            checkpoint = latest_safety_checkpoint(safety_dir)
        else:
            checkpoint = Path(args.resume)
            problem = safety_checkpoint_problem(checkpoint)
            if problem is not None:
                raise TrainerError(f"--resume {checkpoint}: {problem}")
        if checkpoint is not None:
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            if state["config_digest"] != config_digest and not args.allow_config_change:
                raise TrainerError(f"{checkpoint} was written by a different run config ({state['config_digest'][:16]} != {config_digest[:16]})")
            #: the base weights and the teacher data are part of what a resumed
            #: run continues; a rebuilt cache or a resealed plan at the same path
            #: must not be picked up silently
            old_data = state.get("data_identity_sha256")
            if old_data is None and state.get("data_identity") is not None:
                old_data = canonical_sha256(data_content_identity(state["data_identity"]))
            changed = {}
            if state.get("source_identity_sha256") != source_identity_sha:
                old_source = state.get("source_identity") or {}
                changed["source"] = {"checkpoint": state.get("source_identity_sha256"), "now": source_identity_sha,
                                     "differing_keys": sorted(k for k in set(old_source) | set(source_identity)
                                                              if old_source.get(k) != source_identity.get(k)) if old_source else None}
            if old_data != data_identity_sha:
                changed["data"] = {"checkpoint": old_data, "now": data_identity_sha}
            if changed:
                if not args.allow_identity_change:
                    raise TrainerError(f"{checkpoint} was trained on a different {' and '.join(sorted(changed))} identity "
                                       f"{json.dumps(changed, sort_keys=True)}; refusing to resume (--allow-identity-change "
                                       "continues and records the change in every later receipt)")
                identity_override = changed
                if rank == 0:
                    log(f"WARNING identity changed on resume, continuing under --allow-identity-change: {json.dumps(changed, sort_keys=True)}")
            bank.load_state_dict(load_adapters(checkpoint / "adapters.safetensors"))
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
            sampler.load_state(state["sampler"])
            if scaler is not None and state.get("scaler"):
                scaler.load_state(state["scaler"])
            step, tokens_seen, history = int(state["step"]), int(state["tokens_seen"]), list(state.get("history_tail", []))
            guard = state.get("skip_guard") or {}
            consecutive_skips, skip_window = int(guard.get("consecutive_skips", 0)), [int(v) for v in guard.get("window", [])]
            segments = list(state.get("segments", []))
            resumed_from = str(checkpoint)
            if rank == 0:
                log(f"resumed from {checkpoint} at step {step}, tokens {tokens_seen:,}, consecutive skips {consecutive_skips}")
    segments.append({"segment": len(segments), "start_step": step, "tool_version": TOOL_VERSION, "resumed_from": resumed_from,
                     "config_digest": config_digest, "source_identity_sha256": source_identity_sha,
                     "data_identity_sha256": data_identity_sha, "identity_override": identity_override})
    rung_tokens = sorted(int(float(x)) for x in args.rungs.split(",") if x.strip()) if args.rungs else []
    rungs_done = {r for r in rung_tokens if r <= tokens_seen}
    last_safety = time.monotonic()
    receipt_steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    stop_at = total_steps if not args.max_steps_this_run else min(total_steps, step + int(args.max_steps_this_run))
    if rank == 0:
        census = bank.census()
        log(f"engine {args.engine} ({mode}) world {world} dtype {args.dtype} steps {step}->{stop_at} of {total_steps}; "
            f"trainable {census['parameters_trainable']:,}; tokens/step {tokens_per_step:,}")

    def safety_state() -> dict[str, Any]:
        return {"schema": CHECKPOINT_SCHEMA, "tool_version": TOOL_VERSION, "step": step, "tokens_seen": tokens_seen,
                "sampler": sampler.state(), "scaler": None if scaler is None else scaler.state(), "config_digest": config_digest,
                "history_tail": history[-50:], "source_identity_sha256": source_identity_sha, "source_identity": source_identity,
                "data_identity": data_identity, "data_identity_sha256": data_identity_sha,
                "skip_guard": {"consecutive_skips": consecutive_skips, "window": skip_window[-args.skip_window:] if args.skip_window > 0 else []},
                "segments": segments, "halt_reason": halt_reason}

    halt_reason: str | None = None
    while step < stop_at:
        step_start = time.perf_counter()
        bank.zero_grad()
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_factor(step, args.warmup_steps, total_steps, args.min_lr_ratio)
        scale = scaler.scale if scaler is not None else 1.0
        parts_sum: dict[str, float] = {}
        for _ in range(args.accumulation):
            picks = sampler.next_rows(args.micro_batch * world, tokens_seen)
            mine = picks[rank * args.micro_batch : (rank + 1) * args.micro_batch]
            ids, targets = build_batch(sources, mine, objective)
            parts = engine.forward_backward(ids, targets, objective, denominator=denominator, loss_scale=scale)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v)
        if reducer is not None:
            reducer()
            totals = torch.tensor([parts_sum.get(k, 0.0) for k in ("total", "q38", "k3", "ce", "eos", "ul")], dtype=torch.float64)
            dist.all_reduce(totals)
            parts_sum.update(dict(zip(("total", "q38", "k3", "ce", "eos", "ul"), totals.tolist())))
        finite = True
        if scaler is not None:
            finite = scaler.unscale_and_check(bank.trainable())
        if args.dump_step_grads and not receipt_steps and rank == 0:
            #: the reduced (and unscaled) gradient of this run's first step, for
            #: a gradient-level DP == single-process comparison; Adam's per-element
            #: normalisation amplifies fp32 summation-order noise in the weights
            from safetensors.torch import save_file

            save_file({n: bank.tensors[n].grad.detach().float().cpu().contiguous() for n in bank.trainable_names()
                       if bank.tensors[n].grad is not None}, str(args.dump_step_grads))
        grad_norm = None
        if finite:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(bank.trainable(), args.grad_clip if args.grad_clip > 0 else float("inf")))
            if not math.isfinite(grad_norm):
                finite = False
        if finite:
            optimizer.step()
        step += 1
        tokens_seen += tokens_per_step
        consecutive_skips = 0 if finite else consecutive_skips + 1
        if args.skip_window > 0:
            skip_window.append(0 if finite else 1)
            del skip_window[: -args.skip_window]
        halt_reason = skip_guard_reason(consecutive_skips, skip_window, max_consecutive=args.max_consecutive_skips,
                                        max_fraction=args.max_skip_fraction, window_size=args.skip_window)
        if world > 1:
            #: every rank sees the same reduced gradient, so the verdict agrees;
            #: it is still made collective so no rank is left in a barrier alone
            flag = torch.tensor([1 if halt_reason else 0], dtype=torch.int64)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if bool(flag.item()) and halt_reason is None:
                halt_reason = "another rank's skip guard tripped"
        seconds = time.perf_counter() - step_start
        record = {"step": step, "tokens": tokens_seen, "loss": parts_sum.get("total"), "q38": parts_sum.get("q38"), "k3": parts_sum.get("k3"),
                  "ce": parts_sum.get("ce"), "eos": parts_sum.get("eos"), "ul": parts_sum.get("ul"), "grad_norm": grad_norm,
                  "applied": finite, "consecutive_skips": consecutive_skips, "lr": optimizer.param_groups[0]["lr"], "seconds": round(seconds, 3),
                  "tok_per_s": round(tokens_per_step / seconds, 2), "scale": scale}
        history.append(record)
        receipt_steps.append(record)
        if rank == 0 and (step % args.log_every == 0 or step == stop_at or halt_reason):
            print("STEP " + json.dumps(record, sort_keys=True), flush=True)
        if halt_reason:
            #: design section 4 risk row 2: persistent skips pause for a ruling.
            #: Save where the run stands (resumable once ruled on) and stop;
            #: no rung and no adapter export from a halted run
            if rank == 0:
                path = save_safety_checkpoint(safety_dir, step, bank=bank, optimizer=optimizer, state=safety_state(), keep=args.safety_keep)
                log(f"HALT at step {step}: {halt_reason}; safety checkpoint {path}")
            if world > 1:
                dist.barrier()
            break
        for rung in rung_tokens:
            if rung not in rungs_done and tokens_seen >= rung:
                rungs_done.add(rung)
                if rank == 0:
                    sha = save_adapters(out / "rungs" / f"rung-{rung}" / "adapters.safetensors", bank, {"rung_tokens": rung, "step": step, "tokens_seen": tokens_seen})
                    log(f"rung {rung:,} tokens -> adapters sha256 {sha[:16]}")
        due = bool(args.safety_every_steps and step % args.safety_every_steps == 0) or (time.monotonic() - last_safety >= args.safety_minutes * 60)
        if world > 1:
            #: the decision is collective: rank 0's clock alone must not send it
            #: into a barrier the others never reach
            flag = torch.tensor([1 if due else 0], dtype=torch.int64)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            due = bool(flag.item())
        if due or step == stop_at:
            if rank == 0:
                path = save_safety_checkpoint(safety_dir, step, bank=bank, optimizer=optimizer, state=safety_state(), keep=args.safety_keep)
                log(f"safety checkpoint {path}")
            if world > 1:
                dist.barrier()
            last_safety = time.monotonic()
    wall = time.perf_counter() - started
    applied = [r for r in receipt_steps if r["applied"]]
    if halt_reason:
        verdict = "halted_nonfinite"
    elif receipt_steps and not applied:
        verdict = "no_step_applied"
    else:
        verdict = "completed" if step >= total_steps else "segment_done"
    failed = verdict in ("halted_nonfinite", "no_step_applied")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        adapters_sha = None
        if not failed:
            adapters_sha = save_adapters(out / "adapters.safetensors", bank, {"step": step, "tokens_seen": tokens_seen})
            delta = {
                "schema": ADAPTER_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "adapters_sha256": adapters_sha,
                "base_identity_sha256": source_identity_sha, "base_identity": source_identity,
                "data_identity_sha256": data_identity_sha, "segments": segments,
                "site_plan": {k: [asdict(s) for s in v] for k, v in plan.items()},
                "fold_rule": ("full: W += s*B@A; ffn_in: W[:Wt] += s*B@A; ffn_out: W[:, :Wt] += s*B@A, W[:, Wt:] += s3*B3@A3; "
                              "then gain (up_proj rows [Wt, I)) folds as W_down[:, Wt:] *= gain (after the K3 LoRA merge), fp32, once"),
                "step": step, "tokens_seen": tokens_seen,
            }
            write_json_atomic(out / "adapters_manifest.json", delta)
        first = [r["loss"] for r in receipt_steps[:5]]
        last = [r["loss"] for r in receipt_steps[-5:]]
        receipt = {
            "schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "command": "train", "argv": sys.argv,
            "verdict": verdict, "halt_reason": halt_reason, "exit_code": EXIT_HALTED if failed else 0,
            "skip_guard": {"max_consecutive_skips": args.max_consecutive_skips, "max_skip_fraction": args.max_skip_fraction,
                           "skip_window": args.skip_window, "consecutive_skips": consecutive_skips,
                           "window_skipped": sum(skip_window), "window_len": len(skip_window)},
            "engine": args.engine, "mode": mode, "world_size": world, "dtype": args.dtype, "kernels": kernels,
            "torch": torch.__version__, "config_digest": config_digest, "resumed_from": resumed_from,
            "identity_override": identity_override, "segments": segments,
            "preregistration": preregistration,
            "source_identity": source_identity, "source_identity_sha256": source_identity_sha,
            "data_identity": data_identity, "data_identity_sha256": data_identity_sha, "objective": asdict(objective),
            "adapter_census": bank.census(), "total_steps": total_steps, "steps_this_run": len(receipt_steps), "step": step,
            "tokens_seen": tokens_seen, "tokens_per_step": tokens_per_step, "wall_seconds": round(wall, 3),
            "tokens_per_second_this_run": round(len(receipt_steps) * tokens_per_step / wall, 3) if wall > 0 else None,
            "applied_steps": len(applied), "skipped_steps": len(receipt_steps) - len(applied),
            "scaler": None if scaler is None else scaler.state(), "loss_first5_mean": sum(first) / len(first) if first else None,
            "loss_last5_mean": sum(last) / len(last) if last else None, "steps": receipt_steps,
            "streamed_gb": round(engine.streamed_bytes() / 1e9, 3), "layer_loads": engine.layer_loads,
            "layer_load_seconds": round(engine.load_seconds, 3), "allreduce_seconds_mean": None if reducer is None or not reducer.seconds else sum(reducer.seconds) / len(reducer.seconds),
            "memory": device_memory_report(), "host_rss_gb": host_rss_gb(), "adapters_sha256": adapters_sha,
        }
        receipt_path = Path(args.receipt) if args.receipt else out / "training-receipt.json"
        sha = write_json_atomic(receipt_path, receipt)
        if failed:
            print(f"TRAIN HALTED verdict {verdict} at step {step}: {halt_reason or 'no step of this segment was applied'}; "
                  f"steps {len(receipt_steps)} (applied {len(applied)}); no adapters exported; receipt {receipt_path} "
                  f"sha256 {sha[:16]}; resume from {safety_dir} after a ruling", flush=True)
        else:
            print(f"TRAIN DONE steps {len(receipt_steps)} (applied {len(applied)}) loss first5 {receipt['loss_first5_mean']} last5 {receipt['loss_last5_mean']} "
                  f"adapters {adapters_sha[:16]} receipt {receipt_path} sha256 {sha[:16]}", flush=True)
    engine.close()
    if world > 1:
        dist.destroy_process_group()
    return EXIT_HALTED if failed else 0


# --------------------------------------------------------------------------
# Self-test (CPU only; the thorough suite is tests/test_streamed_kd_trainer.py).
# --------------------------------------------------------------------------


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"{'PASS' if condition else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}", flush=True)
        if not condition:
            failures.append(name)

    torch.manual_seed(20261012)
    # 1. bucket KL with K = V equals the full KL.
    vocab = 11
    logits_s = torch.randn(5, vocab, dtype=torch.float64)
    logits_t = torch.randn(5, vocab, dtype=torch.float64)
    logq = torch.log_softmax(logits_s, -1)
    logp = torch.log_softmax(logits_t, -1)
    ids = torch.arange(vocab).expand(5, vocab).clone()
    full = (logp.exp() * (logp - logq)).sum(-1)
    check("bucket KL with K=V equals full KL", torch.allclose(bucket_forward_kl(logq, ids, logp), full, atol=1e-12))
    # 2. top-k + tail equals the KL of the merged-tail distributions.
    k = 4
    top = torch.topk(logp, k, dim=-1)
    rest_mask = torch.ones_like(logp, dtype=torch.bool).scatter(1, top.indices, False)
    p_tail = (logp.exp() * rest_mask).sum(-1)
    q_tail = (logq.exp() * rest_mask).sum(-1)
    head = (top.values.exp() * (top.values - logq.gather(1, top.indices))).sum(-1)
    expected = head + p_tail * (p_tail.log() - q_tail.log())
    check("bucket KL top-k + tail", torch.allclose(bucket_forward_kl(logq, top.indices, top.values), expected, atol=1e-12))
    # 3. unlikelihood candidates on a hand example: "a b c a b d a b ?" at the second-to-last position.
    row = torch.tensor([[1, 2, 3, 1, 2, 4, 1, 2, 5]])
    cand = unlikelihood_candidates(row, 3, 4)
    check("UL candidates exclude gold, keep earlier followers", sorted(int(c) for c in cand[0, 7] if c >= 0) == [3, 4], str(cand[0, 7].tolist()))
    # 4. lr schedule shape.
    check("lr warmup/cosine", abs(lr_factor(0, 10, 100, 0.1) - 0.1) < 1e-12 and abs(lr_factor(100, 10, 100, 0.1) - 0.1) < 1e-12)
    # 5. the skip guard: consecutive limit, fraction only once the window is full.
    guard = dict(max_consecutive=3, max_fraction=0.2, window_size=10)
    check("skip guard trips on persistent skips only",
          skip_guard_reason(2, [1, 1], **guard) is None and skip_guard_reason(3, [1, 1, 1], **guard) is not None
          and skip_guard_reason(0, [1, 0] * 4, **guard) is None and skip_guard_reason(0, [1, 1, 0] * 3 + [0], **guard) is not None
          and skip_guard_reason(0, [1, 0, 0, 0, 0] * 2, **guard) is None)
    print(f"SELF-TEST {'PASS' if not failures else 'FAIL'} ({5 - len(failures)}/5)", flush=True)
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kernels", default="auto", choices=("auto", "torch"))
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--text-vocab-rows", type=int, default=TEXT_VOCAB_ROWS)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=0.0, help="0 = rank (scaling 1.0)")
    p.add_argument("--k3-lora-rank", type=int, default=32)
    p.add_argument("--k3-lora-alpha", type=float, default=0.0)
    p.add_argument("--trunk-ffn-width", type=int, default=4352)
    p.add_argument("--no-k3-gain", action="store_true")
    p.add_argument("--gdn-sites", default="in_proj_qkv,out_proj")
    p.add_argument("--gqa-sites", default="v_proj,o_proj")
    p.add_argument("--w-q38", type=float, default=1.0)
    p.add_argument("--lambda-k3", type=float, default=0.3)
    p.add_argument("--k3-q38-anchor", type=float, default=0.5)
    p.add_argument("--w-ce", type=float, default=0.1)
    p.add_argument("--w-eos", type=float, default=0.1)
    p.add_argument("--w-ul", type=float, default=0.1)
    p.add_argument("--ul-ngram", type=int, default=3)
    p.add_argument("--ul-max-candidates", type=int, default=8)
    p.add_argument("--onpolicy-divergence", default="fkl", choices=("fkl", "jsd"))
    p.add_argument("--jsd-beta", type=float, default=0.5)
    p.add_argument("--eos-ids", default=",".join(str(e) for e in DEFAULT_EOS_IDS))
    p.add_argument("--head-chunk-positions", type=int, default=1024)


def command_registered_native_preflight(args):
    import importlib.util
    adapter = Path(args.native_adapter)
    raw = adapter.read_bytes()
    if not adapter.is_absolute() or not 0 < len(raw) <= 128 * 1024 or hashlib.sha256(raw).hexdigest() != args.native_adapter_sha256:
        raise TrainerError("exact bounded native input adapter required")
    spec = importlib.util.spec_from_file_location("reviewed_registered_native_input_adapter_v29", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.command_preflight(args)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    top.add_argument("--self-test", action="store_true")
    top.add_argument("--version", action="store_true")
    sub = top.add_subparsers(dest="command")

    ni = sub.add_parser("native-input-preflight", help="seven registered native inputs; CPU preparation only, no model/optimizer")
    ni.add_argument("--native-plan", required=True)
    ni.add_argument("--native-plan-sha256", required=True)
    ni.add_argument("--native-plan-bytes", required=True, type=int)
    ni.add_argument("--native-adapter", required=True)
    ni.add_argument("--native-adapter-sha256", required=True)
    ni.add_argument("--device", required=True)
    ni.add_argument("--dtype", required=True)
    ni.add_argument("--receipt", required=True)
    ni.set_defaults(func=command_registered_native_preflight)

    gp = sub.add_parser("grad-parity", help="streamed LoRA grads vs a stock full-load autograd")
    add_common(gp)
    gp.add_argument("--model", required=True)
    gp.add_argument("--device", default="cuda:0")
    gp.add_argument("--batch", type=int, default=2)
    gp.add_argument("--seq-len", type=int, default=256)
    gp.add_argument("--q38-top-k", type=int, default=64)
    gp.add_argument("--k3-top-k", type=int, default=32)
    gp.add_argument("--lora-b-init-std", type=float, default=0.02)
    gp.add_argument("--gain-jitter", type=float, default=0.05)
    gp.add_argument("--also-resident", action="store_true", help="also gate the resident (split-cell) engine, forced host hop mid-stack")
    gp.add_argument("--cross-device", default="", help="diagnostic arm: second half of the layers on this device (e.g. cpu)")
    gp.add_argument("--cross-device-threshold", type=float, default=1e-3)
    gp.add_argument("--threshold", type=float, default=1e-4)
    gp.add_argument("--receipt")
    gp.set_defaults(func=command_grad_parity)

    sc = sub.add_parser("source-check", help="structural check of a real student source (P1 / safetensors) on the host")
    add_common(sc)
    sc.add_argument("--model", help="safetensors fallback source")
    sc.add_argument("--p1-chimera")
    sc.add_argument("--p1-parent-config")
    sc.add_argument("--p1-plan")
    sc.add_argument("--source-plugin")
    sc.add_argument("--source-plugin-args")
    sc.add_argument("--device", default="cpu")
    sc.add_argument("--seq-len", type=int, default=16)
    sc.add_argument("--lora-b-init-std", type=float, default=0.02)
    sc.add_argument("--receipt")
    sc.set_defaults(func=command_source_check)

    tr = sub.add_parser("train", help="the D2 job (streamed or split engine)")
    add_common(tr)
    tr.add_argument("--engine", default="streamed", choices=("streamed", "split"))
    tr.add_argument("--model", help="safetensors fallback source (qwen3_5 layout)")
    tr.add_argument("--p1-chimera", help="Chimera child dir served in the qwen3_5 layout by p1_qwen35_reader.py (OMNI1)")
    tr.add_argument("--p1-parent-config", help="parent HF dir whose config.json P1 widens to the child's FFN width")
    tr.add_argument("--p1-plan", help="sealed oc_checkpoint delta-compose plan (base + registered deltas, e.g. D1)")
    tr.add_argument("--source-plugin", help="file.py:factory for any other reader with the same surface")
    tr.add_argument("--source-plugin-args", help="JSON kwargs for --source-plugin")
    tr.add_argument("--devices", default="cuda:0", help="streamed: one per DP rank; split: first,second")
    tr.add_argument("--split-layers", type=int, default=0, help="split: layers on the first device (default half)")
    tr.add_argument("--data-config", required=True)
    tr.add_argument("--seq-len", type=int, required=True)
    tr.add_argument("--micro-batch", type=int, default=1)
    tr.add_argument("--accumulation", type=int, default=1)
    tr.add_argument("--token-budget", type=float, default=0)
    tr.add_argument("--total-steps", type=int, default=0)
    tr.add_argument("--max-steps-this-run", type=int, default=0)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--gain-lr", type=float, default=1e-3)
    tr.add_argument("--freeze-k3-gain", action="store_true", help="G3 fallback: gain frozen at 1.0")
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--warmup-steps", type=int, default=20)
    tr.add_argument("--min-lr-ratio", type=float, default=0.1)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--initial-loss-scale", type=float, default=2.0**10)
    tr.add_argument("--loss-scale-growth-interval", type=int, default=200)
    tr.add_argument("--init-adapters", help="start from an adapters.safetensors (e.g. the long rung after the short run)")
    tr.add_argument("--rungs", default="")
    tr.add_argument("--out", required=True)
    tr.add_argument("--safety-dir", required=True)
    tr.add_argument("--safety-minutes", type=float, default=30.0)
    tr.add_argument("--safety-every-steps", type=int, default=0)
    tr.add_argument("--safety-keep", type=int, default=2)
    tr.add_argument("--resume", default="", help="'auto' = newest verified checkpoint in --safety-dir, or a checkpoint dir")
    tr.add_argument("--allow-config-change", action="store_true")
    tr.add_argument("--allow-identity-change", action="store_true",
                    help="resume although the base weights or the teacher data differ from the checkpoint's (receipted per segment)")
    tr.add_argument("--max-consecutive-skips", type=int, default=8,
                    help="halt (exit 3) after this many skipped steps in a row; 0 = off.  8 covers GradScaler warm-up "
                         "from 2^16 (6 halvings measured in TPS) with margin")
    tr.add_argument("--max-skip-fraction", type=float, default=0.2,
                    help="halt (exit 3) when more than this fraction of the last --skip-window steps were skipped; >= 1 = off")
    tr.add_argument("--skip-window", type=int, default=100, help="steps in the skip-fraction window; 0 = off")
    tr.add_argument("--preregistration", help="PREREGISTRATION_OMNI_STAGE_D_v1.md; verified against the pin in "
                    "stage2_training_receipt.py.  Required when --lr or a trained --gain-lr is non-zero")
    tr.add_argument("--no-preregistration", metavar="REASON",
                    help="run a weight-moving job without a pin; REASON is recorded in the receipt")
    tr.add_argument("--receipt")
    tr.add_argument("--log-every", type=int, default=1)
    tr.add_argument("--no-pin", action="store_true")
    tr.add_argument("--no-prefetch", action="store_true")
    tr.add_argument("--deterministic", action="store_true")
    tr.add_argument("--dump-step-grads", help="save the reduced gradient of this run's first step (rank 0) to this .safetensors")
    tr.add_argument("--registered-native-input-plan", help="recognized but refused by the legacy D2 backend; use native-input-preflight")
    tr.set_defaults(func=command_train)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.version:
        print(f"{TOOL_NAME} {TOOL_VERSION}")
        return 0
    if args.self_test:
        return self_test()
    if not getattr(args, "func", None):
        parser().print_help()
        return 2
    try:
        return int(args.func(args))
    except TrainerError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
