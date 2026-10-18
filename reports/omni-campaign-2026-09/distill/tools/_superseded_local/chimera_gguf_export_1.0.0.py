#!/usr/bin/env python3
"""E3: per-tensor GGUF converter.  A Chimera checkpoint served by the S0-P1 reader
-> a llama.cpp `qwen35` GGUF, text rows only, with no merged HF copy on any disk.

Why it exists (DISTILL_DESIGN_2026-10-12 S0-P9 / section 8 / STAGE_D_WAVE1_STATUS E3)
------------------------------------------------------------------------------------
The on-policy sampler (D2 option i), the D4 H checks and the final Q4 all need
OMNI1 (or a delta child of it) as a llama.cpp GGUF.  The stock path is
`export_hf_to_gguf.py` -> `convert_hf_to_gguf.py <hf dir>`, which needs a 55 GB
qwen3_5 safetensors copy that no disk here can hold next to the clones.  This
tool runs the SAME stock converter class (llama.cpp b10752, `Qwen3_5TextModel`,
in-process) but swaps its tensor source: instead of indexing `*.safetensors` in
a directory it indexes `p1_qwen35_reader.Qwen35LayoutReader`, one lazy tensor
per qwen3_5 name.  Each tensor is produced by the reader (a zero-copy mmap view,
or one assembled tensor) only when the GGUF writer reaches it, so the peak
working set is one tensor, and nothing but the GGUF is written.

What the GGUF holds, and what it does not
-----------------------------------------
* every qwen3_5 text tensor the reader maps (851 on OMNI1: 64 text layers, 48
  linear + 16 full attention), through the stock converter's own rewrites
  (norm + 1, A_log -> -exp, V-head reorder, conv squeeze);
* `token_embd` / `output` = rows [0, ledger_base_vocab_size) = [0, 248,320): the
  ledger bands and the C18 music band are P1 extras and never reach the GGUF;
* the `1 + w` norms as the CHILD stores them: the importer wrote bf16(1 + w), P1
  serves c - 1 exactly and the converter adds the 1 back in fp32, so the GGUF
  holds c.  The stock converter on the parent's HF weights writes fp32(w) + 1
  instead; the two differ by exactly that bf16 rounding (the test pins it);
* NO charted media block, bridge, flow/media/aux head, band row or MTP block
  (the Chimera checkpoint has no MTP; `no_mtp` is set, so no nextn key);
* the vocabulary, chat template and special tokens from the PARENT's tokenizer
  files (the child's text vocabulary is the parent's [0, 248,320) by
  construction), copied with the qwen3_5 config into a small `<tag>.hfmeta/`
  directory beside the GGUF -- config + tokenizer only, never a weight.

Subcommands (server; CPU; read-only on the checkpoint and the parent)::

    PY=/home/cereal/model-surgery-lab/venv/bin/python
    $PY chimera_gguf_export.py map     --chimera DIR --parent-dir PDIR --receipt OUT.json
    $PY chimera_gguf_export.py convert --chimera DIR --parent-dir PDIR --out-dir D --tag T \\
        [--layers N] [--quants Q4_K_M] [--plan PLAN.json] [--min-free-gib 60]
    $PY chimera_gguf_export.py verify  --gguf F --chimera DIR --parent-dir PDIR [--layers N]
    $PY chimera_gguf_export.py logits  --gguf F --chimera DIR --parent-dir PDIR --layers N \\
        --tps-probe-dir $OC/distill
    $PY chimera_gguf_export.py compare --gguf A --other B

`map` is S0-P9: the converter runs on the reader with the writer in dry-run mode
(headers and shapes only), every GGUF tensor is traced to its qwen3_5 name and
its Chimera parts, the layout is checked against the parent's own dry-run map
(same names, shapes and types except the FFN width), and nothing is written.
`verify` reads a written GGUF back and requires every tensor's bytes to equal
the converter applied eagerly to the reader, and -- independently of both the
converter and the reader's assembly code -- every untransformed tensor's bytes
to equal the Chimera shard slice, and every norm to equal the stored `1 + w`.
`logits` is the end-to-end check: llama-server (CPU) loads a truncated GGUF and
its teacher-forced top-k log-probs are compared with a torch forward of the same
truncated model through tps_probe's streamed builder on the same reader.
`--layers N` truncates for that check only; a truncated GGUF is never an export.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

TOOL_VERSION = "chimera_gguf_export 1.0.0"
MAP_SCHEMA = "neural-forge-omni-e3-gguf-map-v1"
CONVERT_SCHEMA = "neural-forge-omni-e3-gguf-export-v1"
VERIFY_SCHEMA = "neural-forge-omni-e3-gguf-verify-v1"
LOGITS_SCHEMA = "neural-forge-omni-e3-gguf-logits-v1"

OC = Path("/home/cereal/omni-campaign")
DEFAULT_LLAMACPP_ROOT = OC / "export" / "llamacpp"
DEFAULT_EXPORT_DIR = OC / "export"
DEFAULT_PARENT_MAP = OC / "distill" / "selfgen" / "receipts" / "tensor_map_parent_dryrun.json"
HERE = Path(__file__).resolve().parent

#: Copied from the parent into `<tag>.hfmeta/` when present: the converter's vocab
#: and chat-template inputs.  Never a weight file.
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")

#: GGUF tensors the stock qwen35 converter writes WITHOUT any rewrite: their bytes
#: must equal the Chimera slice the reader maps them to.  (gguf suffix -> qwen3_5
#: suffix.)  Linear-attention tensors are absent on purpose: the converter
#: reorders their V heads.
IDENTITY_SUFFIXES = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
}
IDENTITY_GLOBAL = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
}
#: GGUF norms written as `w + 1` from the reader's `c - 1`: their F32 values must
#: equal the Chimera stored `1 + w` exactly.
NORM_SUFFIXES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
NORM_GLOBAL = {"output_norm.weight": "model.language_model.norm.weight"}


class E3Error(RuntimeError):
    """The converter refused: the GGUF would not be the reader's text tensors exactly."""


# --------------------------------------------------------------------------
# peers: the P1 reader, the export lane's helpers, llama.cpp's converter
# --------------------------------------------------------------------------


def _torch() -> Any:
    import torch

    return torch


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_p1(tools_dir: Path = HERE) -> Any:
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import p1_qwen35_reader

    return p1_qwen35_reader


def load_export_lane(export_dir: Path) -> Any:
    """`export_hf_to_gguf.py` (and its `llama_probe`) from the export lane, unchanged."""

    if str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))
    import export_hf_to_gguf

    return export_hf_to_gguf


class Converter:
    """llama.cpp's `conversion` package and the gguf-py of the SAME source tree."""

    def __init__(self, llamacpp_root: Path) -> None:
        self.root = Path(llamacpp_root)
        tree = self.root / "llama.cpp"
        gguf_py = tree / "gguf-py"
        for path in (str(gguf_py), str(tree)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import gguf
        from conversion import base, qwen

        if Path(gguf.__file__).resolve().parents[1] != gguf_py.resolve():
            raise E3Error(f"gguf imported from {gguf.__file__}, not the tree's {gguf_py}")
        self.gguf = gguf
        self.base = base
        self.qwen = qwen
        self.tree = tree
        build = self.root / "build-receipt.json"
        built = json.loads(build.read_text(encoding="utf-8")) if build.is_file() else {}
        self.identity = {
            "llamacpp_root": str(self.root),
            "llama_cpp_commit": built.get("llama_cpp_commit"),
            "llama_cpp_describe": built.get("llama_cpp_describe"),
            "build_receipt_sha256": file_sha256(build) if build.is_file() else None,
            "gguf_py": str(Path(gguf.__file__).resolve().parent),
            "converter_files": {
                rel: file_sha256(tree / rel)
                for rel in ("convert_hf_to_gguf.py", "conversion/base.py", "conversion/qwen.py")
            },
        }

    def ftype(self, outtype: str) -> Any:
        table = {
            "f32": self.gguf.LlamaFileType.ALL_F32,
            "f16": self.gguf.LlamaFileType.MOSTLY_F16,
            "bf16": self.gguf.LlamaFileType.MOSTLY_BF16,
            "q8_0": self.gguf.LlamaFileType.MOSTLY_Q8_0,
        }
        if outtype not in table:
            raise E3Error(f"outtype {outtype!r} is not one of {sorted(table)}")
        return table[outtype]


# --------------------------------------------------------------------------
# the converter's input directory: qwen3_5 config + the parent's tokenizer
# --------------------------------------------------------------------------


def served_names(reader: Any, layers: int) -> list[str]:
    """The reader's qwen3_5 names, the first `layers` text layers only when truncating."""

    if not layers:
        return reader.names()
    if not 0 < layers <= reader.num_layers:
        raise E3Error(f"--layers {layers} is outside 1..{reader.num_layers}")
    prefix = "model.language_model.layers."
    keep = []
    for name in reader.names():
        if name.startswith(prefix) and int(name[len(prefix):].split(".", 1)[0]) >= layers:
            continue
        keep.append(name)
    return keep


def write_hfmeta(reader: Any, parent_dir: Path, out: Path, layers: int) -> dict[str, Any]:
    """`out/` = the child's qwen3_5 config.json + the parent's tokenizer files.

    Refuses a directory that already holds anything else (or any weight file),
    because the stock converter would read whatever weights it finds there.
    """

    parent_config = json.loads((parent_dir / "config.json").read_text(encoding="utf-8"))
    config = reader.qwen35_config(parent_config)
    text = config.get("text_config", config)
    if layers:
        text["num_hidden_layers"] = int(layers)
        if text.get("layer_types"):
            text["layer_types"] = list(text["layer_types"])[: int(layers)]
    out.mkdir(parents=True, exist_ok=True)
    allowed = {"config.json", *TOKENIZER_FILES}
    stray = sorted(p.name for p in out.iterdir() if p.name not in allowed)
    if stray:
        raise E3Error(f"{out} holds {stray[:3]}; the converter input directory holds config + tokenizer only")
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    copied = {}
    for name in TOKENIZER_FILES:
        source = parent_dir / name
        if source.is_file():
            shutil.copyfile(source, out / name)
            copied[name] = file_sha256(out / name)
    if "tokenizer.json" not in copied and "vocab.json" not in copied:
        raise E3Error(f"{parent_dir} has no tokenizer.json / vocab.json for the GGUF vocabulary")
    weights = [p.name for p in out.iterdir() if p.name.endswith(WEIGHT_SUFFIXES)]
    if weights:
        raise E3Error(f"{out} holds weight files {weights[:3]}")
    return {
        "directory": str(out),
        "config_sha256": file_sha256(out / "config.json"),
        "tokenizer_files": copied,
        "parent_config_sha256": file_sha256(parent_dir / "config.json"),
        "layers": int(layers) or reader.num_layers,
        "truncated": bool(layers) and int(layers) != reader.num_layers,
    }


# --------------------------------------------------------------------------
# the stock qwen35 converter on the reader
# --------------------------------------------------------------------------


def make_model_class(conv: Converter, reader: Any, names: list[str], trace: dict[str, Any]) -> Any:
    """`Qwen3_5TextModel` with its tensor source swapped for the reader.

    Everything else -- hparams, vocab, the name map, every rewrite, the type
    rules and the writer -- is the stock class.  `trace` collects the
    qwen3_5 -> converter-name -> GGUF-name chain for the receipt.
    """

    torch = _torch()
    stock = conv.qwen.Qwen3_5TextModel
    lazy = conv.base.LazyTorchTensor
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    layer_count = 1 + max(
        [int(n.split(".")[3]) for n in names if n.startswith("model.language_model.layers.")] or [-1]
    )
    trace.setdefault("source_of", {})
    trace.setdefault("rows", [])

    def serve(name: str, widen: bool, want: Any) -> Any:
        tensor = reader.get(name)
        if widen:
            # a 1 + w norm comes out as c - 1 in bf16 or fp32 (P1's exactness rule);
            # the converter upcasts to fp32 before adding the 1 back, so do it here
            tensor = tensor.to(torch.float32)
        if tensor.dtype != want:
            raise E3Error(f"{name}: the reader served {tensor.dtype}, the index declared {want}")
        return tensor

    class ChimeraQwen35Model(stock):  # type: ignore[misc, valid-type]
        model_arch = stock.model_arch
        no_mtp = True       # a Chimera checkpoint has no MTP block: no nextn layer, no nextn key
        mtp_only = False

        def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Any]:
            type(self)._original_block_count = layer_count
            type(self).opt_num_mtp_layers = 0
            tensors: dict[str, Any] = {}
            for name in names:
                spec = reader.specs[name]
                widen = spec["kind"] == "one_plus"
                want = torch.float32 if widen else dtypes[spec["dtype"]]
                meta = lazy.meta_with_dtype_and_shape(want, tuple(spec["shape"]))

                def gen(name: str = name, widen: bool = widen, want: Any = want, meta: Any = meta) -> Any:
                    return lazy(meta=meta, args=(name,), func=lambda q: serve(q, widen, want))

                item = self.filter_tensors((name, gen))
                if item is None:
                    raise E3Error(f"the stock converter's filter drops {name}; it would be missing from the GGUF")
                converted, generator = item
                if converted in tensors:
                    raise E3Error(f"two reader names map to converter name {converted}")
                trace["source_of"][converted] = name
                tensors[converted] = generator
            return tensors

        def modify_tensors(self, data_torch: Any, name: str, bid: int | None) -> Iterable[tuple[str, Any]]:
            for new_name, out in super().modify_tensors(data_torch, name, bid):
                trace["rows"].append({"converter_name": name, "gguf": new_name})
                yield new_name, out

    return ChimeraQwen35Model


@contextlib.contextmanager
def _quiet_logs() -> Iterator[None]:
    import logging

    logger = logging.getLogger("hf-to-gguf")
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def build_model(conv: Converter, reader: Any, hfmeta: Path, outfile: Path, *, names: list[str],
                outtype: str, dry_run: bool, tag: str, eager: bool = False) -> tuple[Any, dict[str, Any]]:
    trace: dict[str, Any] = {}
    cls = make_model_class(conv, reader, names, trace)
    model = cls(hfmeta, conv.ftype(outtype), outfile, dry_run=dry_run, eager=eager, model_name=tag)
    return model, trace


def _gguf_shape(shape: Iterable[int]) -> list[int]:
    return [int(n) for n in reversed(list(shape))]


def tensor_table(conv: Converter, model: Any, reader: Any, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per GGUF tensor: ggml shape and type, qwen3_5 source, Chimera parts."""

    gguf_to_converter = {row["gguf"]: row["converter_name"] for row in trace["rows"]}
    table = []
    for shard in model.gguf_writer.tensors:
        for name, info in shard.items():
            converter_name = gguf_to_converter.get(name)
            qname = trace["source_of"].get(converter_name) if converter_name else None
            spec = reader.specs.get(qname) if qname else None
            table.append({
                "gguf": name,
                "shape": _gguf_shape(info.shape),
                "type": conv.gguf.GGMLQuantizationType(info.dtype).name,
                "nbytes": int(info.nbytes),
                "qwen35": qname,
                "p1_kind": spec["kind"] if spec else None,
                "chimera_parts": [[c, list(r) if r else None] for c, r in spec["parts"]] if spec else None,
            })
    return table


def kv_table(conv: Converter, model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kv in model.gguf_writer.kv_data:
        for key, value in kv.items():
            raw = value.value
            if isinstance(raw, (list, tuple)) and len(raw) > 16:
                out[key] = {"len": len(raw), "head": [str(v) for v in raw[:3]]}
            elif isinstance(raw, (bytes, bytearray)):
                out[key] = {"bytes": len(raw)}
            elif isinstance(raw, str) and len(raw) > 200:
                out[key] = {"chars": len(raw), "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
            else:
                out[key] = raw if isinstance(raw, (int, float, str, bool, type(None))) else (
                    [v if isinstance(v, (int, float, str, bool)) else str(v) for v in raw]
                    if isinstance(raw, (list, tuple)) else str(raw))
    return out


def structural_checks(reader: Any, names: list[str], table: list[dict[str, Any]],
                      kv: Mapping[str, Any], trace: Mapping[str, Any], layers: int,
                      parent_map: Mapping[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
    """The P9 gates: nothing dropped, nothing extra, text rows only, parent layout."""

    problems: list[str] = []
    n_layers = layers or reader.num_layers
    consumed = sorted(trace["source_of"].values())
    if consumed != sorted(names):
        missing = sorted(set(names) - set(consumed))
        problems.append(f"{len(missing)} served tensor(s) never indexed: {missing[:3]}")
    produced_from = {row["converter_name"] for row in trace["rows"]}
    silent = sorted(set(trace["source_of"]) - produced_from)
    if silent:
        problems.append(f"{len(silent)} indexed tensor(s) produced no GGUF tensor: {silent[:3]}")
    per_block: dict[str, int] = {}
    non_block = []
    for row in table:
        parts = row["gguf"].split(".")
        if parts[0] == "blk":
            per_block[parts[1]] = per_block.get(parts[1], 0) + 1
        else:
            non_block.append(row["gguf"])
    if sorted(non_block) != ["output.weight", "output_norm.weight", "token_embd.weight"]:
        problems.append(f"non-block GGUF tensors are {sorted(non_block)}")
    if sorted(int(b) for b in per_block) != list(range(n_layers)):
        problems.append(f"GGUF blocks are {sorted(int(b) for b in per_block)}, not 0..{n_layers - 1}")
    kinds = list(reader.block_kinds)[:n_layers]
    want_counts = {str(i): (14 if kind == "kda" else 11) for i, kind in enumerate(kinds)}
    if per_block != want_counts:
        bad = {b: (per_block.get(b), want_counts.get(b)) for b in set(per_block) | set(want_counts)
               if per_block.get(b) != want_counts.get(b)}
        problems.append(f"per-block tensor counts differ from 14 (linear) / 11 (full): {dict(list(bad.items())[:4])}")
    by_name = {row["gguf"]: row for row in table}
    for name in ("token_embd.weight", "output.weight"):
        row = by_name.get(name)
        if row and row["shape"] != [reader.hidden, reader.base_vocab]:
            problems.append(f"{name} is {row['shape']}, not [hidden, text rows] = [{reader.hidden}, {reader.base_vocab}]")
    arch = kv.get("general.architecture")
    if arch != "qwen35":
        problems.append(f"general.architecture is {arch!r}")
    if kv.get(f"{arch}.block_count") != n_layers:
        problems.append(f"{arch}.block_count is {kv.get(f'{arch}.block_count')}, not {n_layers}")
    if reader.ffn_width is not None and kv.get(f"{arch}.feed_forward_length") != reader.ffn_width:
        problems.append(f"{arch}.feed_forward_length is {kv.get(f'{arch}.feed_forward_length')}, not {reader.ffn_width}")
    if any("nextn" in key for key in kv) or any("nextn" in row["gguf"] for row in table):
        problems.append("a nextn (MTP) key or tensor is present; the checkpoint has no MTP block")
    tokens = kv.get("tokenizer.ggml.tokens")
    if isinstance(tokens, dict) and tokens.get("len") != reader.base_vocab:
        problems.append(f"the GGUF vocabulary has {tokens.get('len')} tokens, not {reader.base_vocab}")
    for row in table:
        if row["gguf"].startswith("blk.") and row["gguf"].endswith(("ffn_gate.weight", "ffn_up.weight")):
            if row["shape"][1] != reader.ffn_width:
                problems.append(f"{row['gguf']} is {row['shape']}, not width {reader.ffn_width}")
    parent: dict[str, Any] = {"compared": False}
    if parent_map is not None:
        parent = compare_with_parent_map(table, parent_map, reader, n_layers)
        problems.extend(parent["problems"])
    facts = {"per_block_counts": dict(sorted(per_block.items(), key=lambda kv_: int(kv_[0]))),
             "non_block": sorted(non_block), "parent_map": parent}
    return problems, facts


def compare_with_parent_map(table: list[dict[str, Any]], parent_map: Mapping[str, Any],
                            reader: Any, n_layers: int) -> dict[str, Any]:
    """The parent's own stock dry run: same names, shapes and types, except the
    FFN width (17,408 -> the child's) and the parent's MTP block and later layers."""

    want: dict[str, dict[str, Any]] = {}
    parent_width = None
    for row in parent_map.get("mapped") or []:
        name = row["gguf"]
        parts = name.split(".")
        if "nextn" in name:
            continue
        if parts[0] == "blk" and int(parts[1]) >= n_layers:
            continue
        shape = list(row["shape"])
        if parts[0] == "blk" and parts[2] in ("ffn_gate", "ffn_up"):
            parent_width = shape[1]
            shape[1] = reader.ffn_width
        if parts[0] == "blk" and parts[2] == "ffn_down":
            shape[0] = reader.ffn_width
        want[name] = {"shape": shape, "type": row["to"]}
    got = {row["gguf"]: {"shape": row["shape"], "type": row["type"]} for row in table}
    problems = []
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing:
        problems.append(f"parent map has {len(missing)} tensor(s) the child GGUF lacks: {missing[:3]}")
    if extra:
        problems.append(f"child GGUF has {len(extra)} tensor(s) the parent map lacks: {extra[:3]}")
    differ = sorted(n for n in set(want) & set(got) if want[n] != got[n])
    if differ:
        problems.append(f"{len(differ)} tensor(s) differ from the parent map in shape/type: "
                        f"{[(n, got[n], want[n]) for n in differ[:2]]}")
    return {"compared": True, "parent_tensors_compared": len(set(want) & set(got)),
            "parent_ffn_width": parent_width, "child_ffn_width": reader.ffn_width,
            "missing": missing[:20], "extra": extra[:20], "differ": differ[:20],
            "problems": problems}


# --------------------------------------------------------------------------
# receipts (the export lane's canonical writer, unchanged)
# --------------------------------------------------------------------------


def _peak_rss() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, AttributeError):
        return None


def _write(lane: Any, path: str | None, body: dict[str, Any]) -> str | None:
    """`llama_probe.write_receipt` as the export lane imports it: canonical JSON,
    atomic, plus the immutable `<stem>.<sha8>` sidecar."""

    if not path:
        return None
    digest = lane.write_receipt(path, body)
    print(f"receipt {path} sha256={digest}")
    return digest


def _source_identity(p1: Any, reader: Any, args: argparse.Namespace) -> dict[str, Any]:
    chimera = Path(args.chimera)
    summary = reader.summary()
    return {
        "chimera_dir": str(chimera),
        "chimera_config_sha256": file_sha256(chimera / "config.json"),
        "chimera_index_sha256": file_sha256(chimera / "model.safetensors.index.json")
        if (chimera / "model.safetensors.index.json").is_file() else None,
        "chimera_shards": {name: info["bytes"] for name, info in sorted(reader.shards.files.items())},
        "p1_tool": {"version": p1.TOOL_VERSION, "sha256": file_sha256(Path(p1.__file__).resolve())},
        "plan": summary["delta_plan"],
        "root_check": summary["root_check"],
        "reader_summary": {k: summary[k] for k in (
            "num_layers", "num_blocks", "text_blocks", "charted_blocks", "layer_types", "ffn_width",
            "channel_partition", "base_vocab", "vocab", "tied", "kinds")},
        "coverage": summary["coverage"],
    }


def _tool_identity() -> dict[str, Any]:
    return {"version": TOOL_VERSION, "sha256": file_sha256(Path(__file__).resolve()),
            "argv": sys.argv, "host": socket.gethostname(), "python": sys.version.split()[0]}


def _open(args: argparse.Namespace) -> tuple[Any, Any, Converter, Any]:
    p1 = load_p1()
    lane = load_export_lane(Path(args.export_dir))
    conv = Converter(Path(args.llamacpp_root))
    reader = p1.open_reader(args.chimera, plan_path=args.plan, verify_root_sha=args.verify_root_sha)
    return p1, lane, conv, reader


# --------------------------------------------------------------------------
# map (S0-P9): the converter on the reader, headers only
# --------------------------------------------------------------------------


def run_map(args: argparse.Namespace) -> int:
    started = time.time()
    p1, lane, conv, reader = _open(args)
    names = served_names(reader, args.layers)
    work = Path(tempfile.mkdtemp(prefix="e3-map-", dir=args.work_dir))
    try:
        hfmeta = write_hfmeta(reader, Path(args.parent_dir), work / "hfmeta", args.layers)
        model, trace = build_model(conv, reader, Path(hfmeta["directory"]), work / "dryrun.gguf",
                                   names=names, outtype=args.outtype, dry_run=True, tag=args.tag)
        with _quiet_logs():
            model.prepare_tensors()
            model.prepare_metadata(vocab_only=False)
        table = tensor_table(conv, model, reader, trace)
        kv = kv_table(conv, model)
        parent_map = json.loads(Path(args.parent_map).read_text(encoding="utf-8")) if args.parent_map else None
        problems, facts = structural_checks(reader, names, table, kv, trace, args.layers, parent_map)
        written = sorted(p.name for p in work.iterdir() if p.name != "hfmeta")
        if written:
            problems.append(f"the dry run wrote {written}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    types: dict[str, int] = {}
    for row in table:
        types[row["type"]] = types.get(row["type"], 0) + 1
    total = sum(row["nbytes"] for row in table)
    passed = not problems
    body = {
        "schema": MAP_SCHEMA, "kind": "e3_gguf_tensor_map_dry_run", "created_at": lane.utc_now(),
        "tool": _tool_identity(), "converter": conv.identity, "source": _source_identity(p1, reader, args),
        "hfmeta": {k: v for k, v in hfmeta.items() if k != "directory"},
        "outtype": args.outtype, "layers": args.layers or reader.num_layers,
        "gguf_tensor_count": len(table), "gguf_out_types": types, "gguf_tensor_bytes": total,
        "served_tensors": len(names), "kv": kv, "facts": facts,
        "parent_map_path": str(args.parent_map) if args.parent_map else None,
        "parent_map_sha256": file_sha256(Path(args.parent_map)) if args.parent_map else None,
        "problems": problems, "pass": passed, "wrote_files": False,
        "seconds": round(time.time() - started, 2), "peak_rss_bytes": _peak_rss(),
        "tensors": table,
    }
    reader.close()
    _write(lane, args.receipt, body)
    pm = facts["parent_map"]
    print(
        f"P9 MAP {'PASS' if passed else 'FAIL'}: {len(names)} reader tensors -> {len(table)} GGUF tensors "
        f"({types}), {total / 1e9:.2f} GB {args.outtype}, blocks {len(facts['per_block_counts'])}, "
        f"text rows {reader.base_vocab}, ffn {reader.ffn_width}, "
        f"parent map {'compared ' + str(pm.get('parent_tensors_compared')) + ' tensors' if pm.get('compared') else 'not given'}, "
        f"{len(problems)} problem(s), {body['seconds']} s"
    )
    for line in problems[:10]:
        print(f"  PROBLEM {line}")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# convert: write the GGUF (bf16), optionally quantize with llama-quantize
# --------------------------------------------------------------------------


def run_convert(args: argparse.Namespace) -> int:
    started = time.time()
    p1, lane, conv, reader = _open(args)
    tag = lane.safe_tag(args.tag)
    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / lane.gguf_name(tag, args.outtype)
    partial = outfile.with_name(outfile.name + ".partial")
    if outfile.exists() and not args.force:
        print(f"REFUSED: {outfile} exists (pass --force to rebuild)", file=sys.stderr)
        return 3
    names = served_names(reader, args.layers)
    hfmeta = write_hfmeta(reader, Path(args.parent_dir), out_dir / f"{tag}.hfmeta", args.layers)
    model, trace = build_model(conv, reader, Path(hfmeta["directory"]), partial,
                               names=names, outtype=args.outtype, dry_run=False, tag=tag)
    stages: list[dict[str, Any]] = []
    with _quiet_logs():
        model.prepare_tensors()
    estimate = sum(info.nbytes for shard in model.gguf_writer.tensors for info in shard.values())
    free = shutil.disk_usage(out_dir).free
    floor = int(args.min_free_gib * (1 << 30))
    if free - estimate < floor:
        print(f"REFUSED: {out_dir} has {free / (1 << 30):.1f} GiB free; writing {estimate / (1 << 30):.1f} GiB "
              f"would cross the {args.min_free_gib} GiB floor", file=sys.stderr)
        reader.close()
        return 3
    # model.write() minus the prepare_tensors() already done above
    with _quiet_logs():
        model.prepare_metadata(vocab_only=False)
    begin = time.time()
    model.gguf_writer.write_header_to_file(path=partial)
    model.gguf_writer.write_kv_data_to_file()
    model.gguf_writer.write_tensors_to_file(progress=False)
    model.gguf_writer.close()
    partial.replace(outfile)
    stages.append({"stage": "convert", "rc": 0, "wall_s": round(time.time() - begin, 2),
                   "page_drops": reader.page_drops, "bytes_read_through_reader": reader.bytes_read})
    table = tensor_table(conv, model, reader, trace)
    kv = kv_table(conv, model)
    parent_map = json.loads(Path(args.parent_map).read_text(encoding="utf-8")) if args.parent_map else None
    problems, facts = structural_checks(reader, names, table, kv, trace, args.layers, parent_map)
    artifacts: dict[str, Any] = {}
    begin = time.time()
    digest = file_sha256(outfile)
    artifacts[args.outtype] = {"path": str(outfile), "bytes": outfile.stat().st_size, "sha256": digest,
                               "hash_wall_s": round(time.time() - begin, 2)}
    print(f"convert wall={stages[0]['wall_s']}s -> {outfile} {outfile.stat().st_size / 1e9:.2f} GB sha256={digest[:16]}")
    for quant in lane.parse_quants(args.quants) if args.quants else []:
        target = out_dir / lane.gguf_name(tag, quant)
        if target.exists() and not args.force:
            print(f"REFUSED: {target} exists", file=sys.stderr)
            problems.append(f"{target} exists; not re-quantized")
            continue
        quantizer = Path(args.llamacpp_root) / "bin" / "llama-quantize"
        env = dict(os.environ, LD_LIBRARY_PATH=f"{Path(args.llamacpp_root) / 'bin'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                   CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        cmd = [str(quantizer), str(outfile), str(target), quant, str(args.threads)]
        begin = time.time()
        rc, out, err = lane.run(cmd, timeout=args.quantize_timeout, env=env)
        stages.append({"stage": f"quantize:{quant}", "cmd": cmd, "rc": rc, "wall_s": round(time.time() - begin, 2),
                       "stdout_tail": out.strip().splitlines()[-6:], "stderr_tail": err.strip().splitlines()[-6:]})
        print(f"quantize {quant} rc={rc} wall={stages[-1]['wall_s']}s -> {target}")
        if rc != 0:
            problems.append(f"llama-quantize {quant} rc={rc}")
            continue
        artifacts[quant] = {"path": str(target), "bytes": target.stat().st_size, "sha256": file_sha256(target)}
    passed = not problems
    body = {
        "schema": CONVERT_SCHEMA, "kind": "e3_gguf_export", "created_at": lane.utc_now(),
        "tool": _tool_identity(), "converter": conv.identity, "source": _source_identity(p1, reader, args),
        "hfmeta": hfmeta, "tag": tag, "outtype": args.outtype, "layers": args.layers or reader.num_layers,
        "truncated_test_artifact": hfmeta["truncated"],
        "quantization_stage": args.quantization_stage, "threads": args.threads,
        "gguf_tensor_count": len(table), "kv": kv, "facts": facts, "stages": stages,
        "artifacts": artifacts, "free_bytes_before": free, "estimated_bytes": estimate,
        "min_free_gib": args.min_free_gib, "problems": problems, "pass": passed,
        "seconds": round(time.time() - started, 2), "peak_rss_bytes": _peak_rss(),
    }
    reader.close()
    receipt = args.receipt or str(out_dir / f"e3-export-receipt-{tag}.json")
    _write(lane, receipt, body)
    print(f"E3 CONVERT {'PASS' if passed else 'FAIL'}: {len(table)} tensors, {len(problems)} problem(s), "
          f"peak RSS {(_peak_rss() or 0) / 1e9:.2f} GB, {body['seconds']} s")
    for line in problems[:10]:
        print(f"  PROBLEM {line}")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# verify: read the GGUF back, tensor-exact
# --------------------------------------------------------------------------


def _gguf_tensors(conv: Converter, path: Path) -> tuple[Any, dict[str, Any]]:
    gguf_reader = conv.gguf.GGUFReader(str(path))
    return gguf_reader, {t.name: t for t in gguf_reader.tensors}


def _raw(tensor: Any) -> bytes:
    import numpy as np

    return np.ascontiguousarray(tensor.data).tobytes()


def run_verify(args: argparse.Namespace) -> int:
    """Every GGUF tensor's bytes == the converter applied eagerly to the reader, and
    the untransformed ones == the Chimera slice, the norms == the stored 1 + w."""

    torch = _torch()
    import numpy as np
    from safetensors import safe_open

    started = time.time()
    p1, lane, conv, reader = _open(args)
    gguf_reader, tensors = _gguf_tensors(conv, Path(args.gguf))
    names = served_names(reader, args.layers)
    work = Path(tempfile.mkdtemp(prefix="e3-verify-", dir=args.work_dir))
    failures: list[str] = []
    independent = {"identity": 0, "norm": 0, "ssm_norm": 0}
    checked = 0
    try:
        hfmeta = write_hfmeta(reader, Path(args.parent_dir), work / "hfmeta", args.layers)
        model, trace = build_model(conv, reader, Path(hfmeta["directory"]), work / "eager.gguf",
                                   names=names, outtype=args.outtype, dry_run=True, tag="verify", eager=True)
        seen: set[str] = set()
        for converter_name, data in model.get_tensors():
            data = conv.base.LazyTorchTensor.to_eager(data)
            if data.dtype not in (torch.float16, torch.float32):
                data = data.to(torch.float32)
            bid = next((int(p) for p in converter_name.split(".") if p.isdecimal()), None)
            for gguf_name, out in model.modify_tensors(data, converter_name, bid):
                seen.add(gguf_name)
                stored = tensors.get(gguf_name)
                if stored is None:
                    failures.append(f"{gguf_name}: absent from the GGUF")
                    continue
                qtype = conv.gguf.GGMLQuantizationType(int(stored.tensor_type))
                expect = conv.gguf.quants.quantize(np.ascontiguousarray(out.numpy()), qtype)
                if _gguf_shape(out.shape) != [int(n) for n in stored.shape]:
                    failures.append(f"{gguf_name}: shape {list(stored.shape)} vs {_gguf_shape(out.shape)}")
                elif expect.tobytes() != _raw(stored):
                    failures.append(f"{gguf_name}: bytes differ from the converter on the reader ({qtype.name})")
                checked += 1
            reader.drop_pages()
        extra = sorted(set(tensors) - seen)
        if extra:
            failures.append(f"{len(extra)} GGUF tensor(s) the reader does not produce: {extra[:3]}")

        # independent of the converter AND of the reader's assembly: shard bytes
        def chimera(cname: str, rows: Any) -> Any:
            spec = reader.shards.spec(cname)
            with safe_open(str(reader.directory / spec["file"]), framework="pt", device="cpu") as handle:
                piece = handle.get_slice(cname)
                tensor = piece[rows[0]:rows[1]] if rows else piece[:]
            return tensor

        for gguf_name, stored in tensors.items():
            parts = gguf_name.split(".")
            if parts[0] == "blk":
                suffix = ".".join(parts[2:])
                q_prefix = f"model.language_model.layers.{parts[1]}."
                identity = q_prefix + IDENTITY_SUFFIXES[suffix] if suffix in IDENTITY_SUFFIXES else None
                norm = q_prefix + NORM_SUFFIXES[suffix] if suffix in NORM_SUFFIXES else None
                ssm_norm = q_prefix + "linear_attn.norm.weight" if suffix == "ssm_norm.weight" else None
            else:
                identity = IDENTITY_GLOBAL.get(gguf_name)
                norm = NORM_GLOBAL.get(gguf_name)
                ssm_norm = None
            qname = identity or norm or ssm_norm
            if qname is None or qname not in reader.specs:
                continue
            (cname, rows), = reader.specs[qname]["parts"]
            source = chimera(cname, rows)
            if identity:
                want = source.contiguous().view(torch.int16).numpy().tobytes() if source.dtype == torch.bfloat16 \
                    else source.contiguous().numpy().tobytes()
                key = "identity"
            elif norm:
                want = source.float().contiguous().numpy().tobytes()
                key = "norm"
            else:
                want = source[0].float().contiguous().numpy().tobytes()
                key = "ssm_norm"
            if want != _raw(stored):
                failures.append(f"{gguf_name}: bytes differ from the Chimera slice {cname}{rows or ''} ({key})")
            independent[key] += 1
            del source
    finally:
        shutil.rmtree(work, ignore_errors=True)
    passed = not failures and checked == len(tensors)
    body = {
        "schema": VERIFY_SCHEMA, "kind": "e3_gguf_readback", "created_at": lane.utc_now(),
        "tool": _tool_identity(), "converter": conv.identity, "source": _source_identity(p1, reader, args),
        "gguf": str(args.gguf), "gguf_sha256": file_sha256(Path(args.gguf)) if args.hash_gguf else None,
        "gguf_tensors": len(tensors), "checked_against_converter_on_reader": checked,
        "checked_independently_against_chimera_bytes": independent,
        "failures": failures, "pass": passed,
        "seconds": round(time.time() - started, 2), "peak_rss_bytes": _peak_rss(),
    }
    reader.close()
    _write(lane, args.receipt, body)
    print(f"E3 VERIFY {'PASS' if passed else 'FAIL'}: {checked}/{len(tensors)} GGUF tensors byte-equal to the "
          f"stock converter on the P1 reader; independently byte-equal to Chimera shards: {independent}; "
          f"{len(failures)} failure(s), {body['seconds']} s")
    for line in failures[:10]:
        print(f"  FAIL {line}")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# compare two GGUFs (tests: the reader path vs the stock HF path)
# --------------------------------------------------------------------------


def compare_ggufs(conv: Converter, left: Path, right: Path, ignore_kv_prefixes: tuple[str, ...] = ("general.",)) -> dict[str, Any]:
    _, a = _gguf_tensors(conv, left)
    _, b = _gguf_tensors(conv, right)
    differ = []
    for name in sorted(set(a) & set(b)):
        ta, tb = a[name], b[name]
        if int(ta.tensor_type) != int(tb.tensor_type) or list(ta.shape) != list(tb.shape) or _raw(ta) != _raw(tb):
            differ.append(name)

    def kvs(path: Path) -> dict[str, Any]:
        reader = conv.gguf.GGUFReader(str(path))
        out = {}
        for field in reader.fields.values():
            if field.name.startswith(ignore_kv_prefixes) or field.name.startswith("GGUF."):
                continue
            out[field.name] = [bytes(field.parts[i]).hex() for i in field.data]
        return out

    ka, kb = kvs(left), kvs(right)
    kv_differ = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
    return {"only_left": sorted(set(a) - set(b)), "only_right": sorted(set(b) - set(a)),
            "differ": differ, "common": len(set(a) & set(b)), "kv_differ": kv_differ,
            "equal": not differ and set(a) == set(b) and not kv_differ}


def run_compare(args: argparse.Namespace) -> int:
    conv = Converter(Path(args.llamacpp_root))
    result = compare_ggufs(conv, Path(args.gguf), Path(args.other))
    print(json.dumps(result, indent=1))
    return 0 if result["equal"] else 1


# --------------------------------------------------------------------------
# logits: llama-server (CPU) on a truncated GGUF vs a torch forward of the same
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(url: str, payload: Mapping[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def torch_reference(p1: Any, reader: Any, args: argparse.Namespace, ids: list[int], work: Path) -> Any:
    """Teacher-forced fp32 logits [T, V] of the first `layers` text layers, built by
    tps_probe's own streamed builder on the same reader (the P1 seam)."""

    torch = _torch()
    sys.path.insert(0, str(Path(args.tps_probe_dir).resolve()))
    import tps_probe  # noqa: WPS433 (the engine, from the path given)

    config_dir = work / "torch-config"
    config_dir.mkdir()
    parent = json.loads((Path(args.parent_dir) / "config.json").read_text(encoding="utf-8"))
    config = reader.qwen35_config(parent)
    text = config.get("text_config", config)
    text["num_hidden_layers"] = int(args.layers)
    text["layer_types"] = list(text.get("layer_types") or [])[: int(args.layers)]
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with p1.tps_probe_seam(reader, tps_probe):
        model, head, _info = tps_probe.build_streamed_text_model(config_dir, "cpu", torch.float32, "sdpa")
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
        logits = head(hidden[0]).float()
    return logits


def run_logits(args: argparse.Namespace) -> int:
    torch = _torch()
    started = time.time()
    p1, lane, conv, reader = _open(args)
    if not args.layers:
        raise E3Error("logits needs --layers N (a truncated GGUF of the same N layers)")
    _, tensors = _gguf_tensors(conv, Path(args.gguf))
    blocks = sorted({int(n.split(".")[1]) for n in tensors if n.startswith("blk.")})
    if blocks != list(range(int(args.layers))):
        raise E3Error(f"{args.gguf} holds blocks {blocks[:3]}..{blocks[-1:]}, not the {args.layers} asked for")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.parent_dir))
    ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"][: int(args.tokens)]
    work = Path(tempfile.mkdtemp(prefix="e3-logits-", dir=args.work_dir))
    server = None
    log_path = Path(args.server_log) if args.server_log else work / "llama-server.log"
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        begin = time.time()
        reference = torch_reference(p1, reader, args, ids, work)
        torch_seconds = round(time.time() - begin, 2)
        logprobs = torch.log_softmax(reference, dim=-1)
        torch_finite = bool(torch.isfinite(reference).all())
        port = _free_port()
        bin_dir = Path(args.llamacpp_root) / "bin"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                   LD_LIBRARY_PATH=f"{bin_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        cmd = [str(bin_dir / "llama-server"), "-m", str(args.gguf), "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "0", "-c", str(max(256, 2 * len(ids))), "-np", "1", "-t", str(args.threads),
               "--no-warmup"]
        log = open(log_path, "w", encoding="utf-8")
        server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + args.load_timeout
        healthy = False
        while time.time() < deadline and server.poll() is None:
            with contextlib.suppress(Exception):
                if _get(f"{url}/health").get("status") == "ok":
                    healthy = True
                    break
            time.sleep(2)
        if not healthy:
            raise E3Error(f"llama-server did not become healthy (rc={server.poll()}); log {log_path}")
        load_seconds = round(time.time() - begin - torch_seconds, 2)
        for position in range(len(ids)):
            body = _post(f"{url}/completion", {
                "prompt": ids[: position + 1], "n_predict": 1, "n_probs": int(args.top_k),
                "temperature": 0.0, "cache_prompt": False, "post_sampling_probs": False,
            })
            probs = (body.get("completion_probabilities") or [{}])[0]
            top = probs.get("top_logprobs") or []
            llama_ids = [int(item["id"]) for item in top]
            llama_lp = [float(item["logprob"]) for item in top]
            row_lp = logprobs[position]
            torch_top = torch.topk(row_lp, int(args.top_k))
            diffs = [abs(lp - float(row_lp[i])) for i, lp in zip(llama_ids, llama_lp)]
            rows.append({
                "position": position, "next_true": ids[position + 1] if position + 1 < len(ids) else None,
                "llama_top1": llama_ids[0] if llama_ids else None, "torch_top1": int(torch_top.indices[0]),
                "llama_topk": llama_ids, "torch_topk": [int(i) for i in torch_top.indices],
                "llama_finite": all(lp == lp and abs(lp) != float("inf") for lp in llama_lp) and bool(llama_lp),
                "max_abs_logprob_diff_on_llama_topk": max(diffs) if diffs else None,
                "topk_overlap": len(set(llama_ids) & {int(i) for i in torch_top.indices}),
                "tokens_evaluated": body.get("tokens_evaluated"),
            })
        props = _get(f"{url}/props")
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                server.wait(timeout=30)
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGKILL)
        server_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:] if log_path.is_file() else []
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    agree = sum(int(r["llama_top1"] == r["torch_top1"]) for r in rows)
    finite = torch_finite and all(r["llama_finite"] for r in rows)
    worst = max((r["max_abs_logprob_diff_on_llama_topk"] or 0.0) for r in rows) if rows else None
    evaluated = [r["tokens_evaluated"] for r in rows]
    if not finite:
        failures.append("a non-finite logit or logprob")
    if len(rows) != len(ids):
        failures.append(f"{len(rows)} of {len(ids)} positions answered")
    if evaluated and any(e is not None and e != i + 1 for i, e in enumerate(evaluated)):
        failures.append(f"llama-server evaluated {evaluated[:5]}... tokens, not the prefix lengths (a BOS was added?)")
    agreement = agree / len(rows) if rows else 0.0
    passed = not failures
    body = {
        "schema": LOGITS_SCHEMA, "kind": "e3_gguf_llamacpp_vs_torch", "created_at": lane.utc_now(),
        "tool": _tool_identity(), "converter": conv.identity, "source": _source_identity(p1, reader, args),
        "gguf": str(args.gguf), "gguf_sha256": file_sha256(Path(args.gguf)), "layers": int(args.layers),
        "prompt": args.prompt, "ids": ids, "top_k": int(args.top_k),
        "torch": {"engine": "tps_probe.build_streamed_text_model via p1 seam", "dtype": "float32",
                  "device": "cpu", "seconds": torch_seconds, "finite": torch_finite},
        "llama_server": {"cmd": cmd, "device": "cpu (-ngl 0, CUDA_VISIBLE_DEVICES='')",
                         "load_seconds": load_seconds, "log_tail": server_tail,
                         "props_subset": {k: props.get(k) for k in ("model_path", "n_ctx", "build_info")}
                         if isinstance(props, dict) else None},
        "top1_agreement": {"agree": agree, "positions": len(rows), "fraction": round(agreement, 6)},
        "max_abs_logprob_diff_on_llama_topk": worst, "finite": finite,
        "note": "finite + loads = functional; top-1 agreement and logprob diffs are diagnostics "
        "(llama.cpp CPU vs torch fp32 on the same bf16 weights)",
        "failures": failures, "pass": passed, "rows": rows,
        "seconds": round(time.time() - started, 2), "peak_rss_bytes": _peak_rss(),
    }
    reader.close()
    _write(lane, args.receipt, body)
    print(f"E3 LOGITS {'PASS' if passed else 'FAIL'}: llama.cpp (CPU) vs torch fp32 on {args.layers} OMNI1 text "
          f"layer(s), {len(rows)} positions: top-1 agreement {agree}/{len(rows)} = {agreement:.4f}, "
          f"top-{args.top_k} overlap mean {sum(r['topk_overlap'] for r in rows) / max(1, len(rows)):.2f}, "
          f"max |dlogprob| on llama top-{args.top_k} {worst}, finite={finite}, {body['seconds']} s")
    for line in failures[:10]:
        print(f"  FAIL {line}")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser, *, chimera: bool = True) -> None:
    if chimera:
        p.add_argument("--chimera", required=True, help="Chimera checkpoint directory (read-only)")
        p.add_argument("--parent-dir", required=True,
                       help="the parent HF directory: its config.json and tokenizer files only")
        p.add_argument("--plan", default=None, help="sealed oc_checkpoint.py delta-compose plan")
        p.add_argument("--verify-root-sha", dest="verify_root_sha",
                       action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--layers", type=int, default=0,
                       help="first N text layers only (end-to-end checks; never an export)")
        p.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32", "q8_0"])
        p.add_argument("--work-dir", default=None, help="scratch for the config+tokenizer staging dir")
        p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--llamacpp-root", default=str(DEFAULT_LLAMACPP_ROOT))
    p.add_argument("--receipt", default=None)
    p.add_argument("--threads", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=TOOL_VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("map", help="S0-P9: converter dry run on the reader (headers only)")
    _common(p)
    p.add_argument("--tag", default="E3-map")
    p.add_argument("--parent-map", default=str(DEFAULT_PARENT_MAP) if DEFAULT_PARENT_MAP.is_file() else None,
                   help="the parent's stock dry-run map receipt; '' to skip")
    p.set_defaults(func=run_map)
    p = sub.add_parser("convert", help="write the GGUF through the reader, optionally quantize")
    _common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--quants", default="", help="e.g. Q4_K_M (Q4 happens once, at the end of stage T)")
    p.add_argument("--quantization-stage", default="unspecified")
    p.add_argument("--quantize-timeout", type=int, default=10800)
    p.add_argument("--min-free-gib", type=float, default=60.0, help="refuse a write that crosses this floor")
    p.add_argument("--parent-map", default=str(DEFAULT_PARENT_MAP) if DEFAULT_PARENT_MAP.is_file() else None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=run_convert)
    p = sub.add_parser("verify", help="read a GGUF back, tensor-exact")
    _common(p)
    p.add_argument("--gguf", required=True)
    p.add_argument("--hash-gguf", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=run_verify)
    p = sub.add_parser("logits", help="llama-server CPU vs torch fp32 on a truncated GGUF")
    _common(p)
    p.add_argument("--gguf", required=True)
    p.add_argument("--tps-probe-dir", required=True)
    p.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is Berlin. "
                   "The capital of Italy is Rome. The capital of Spain is")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--load-timeout", type=float, default=600.0)
    p.add_argument("--server-log", default=None)
    p.add_argument("--keep-work", action="store_true")
    p.set_defaults(func=run_logits)
    p = sub.add_parser("compare", help="two GGUFs: tensors byte-equal, non-general KV equal")
    _common(p, chimera=False)
    p.add_argument("--gguf", required=True)
    p.add_argument("--other", required=True)
    p.set_defaults(func=run_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "parent_map", None) == "":
        args.parent_map = None
    try:
        return int(args.func(args))
    except E3Error as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # the reader's own refusals keep their message and exit 3
        if type(exc).__name__ == "P1Error":
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
