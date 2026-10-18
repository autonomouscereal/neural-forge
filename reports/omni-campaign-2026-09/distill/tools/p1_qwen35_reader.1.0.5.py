#!/usr/bin/env python3
"""S0-P1: a zero-copy reader that serves an OMNI1/S5 Chimera checkpoint in the
qwen3_5 (HF transformers 5.12.1) text layout, tensor-exact, with an optional
delta overlay composed per tensor from a sealed `oc_checkpoint.py` plan.

Why it exists (DISTILL_DESIGN_2026-10-12 G0 / S0-P1 / section 4.1)
-------------------------------------------------------------------
Stage D trains in the qwen3_5 layout, but no disk here holds a 55 GB projected
copy of OMNI1 next to its clones.  So the projection is never written: every
qwen3_5 tensor is a NAME/SLICE onto the Chimera shards, opened with `mmap` and
handed out as a `torch.frombuffer` view -- no byte is copied for any tensor the
map can express as a slice, and the few it cannot (below) are assembled one
tensor at a time, never as a checkpoint.

The map is the exact inverse of `chimera_omni/importing/qwen35.py` at ef1ffb9 (the
tree S3/S5 were written with), and every rewrite there has its inverse here:

=======================  ===================================  ===============
qwen3_5 name (per layer)  Chimera source                       how
=======================  ===================================  ===============
mlp.gate_proj            ffn.up rows [0, I)                    view
mlp.up_proj              ffn.up rows [I, 2I)                   view
mlp.down_proj            ffn.down                              view
input/post_attn norm     norm1/norm2, stored as 1 + w          c - 1, checked
linear_attn.in_proj_qkv  cat(q_proj, k_proj, v_proj)           assembled
linear_attn.conv1d       cat(q_conv, k_conv, v_conv)           assembled
linear_attn.norm         output_norm row 0 (rows all equal)    view, checked
linear_attn.in_proj_z/b/a, dt_bias, A_log, out_proj            view
self_attn.q_proj         per head [q_h ; gate_h]               assembled
self_attn.k/v/o_proj     k_proj / v_proj / out                 view
self_attn.q_norm/k_norm  stored as 1 + w                       c - 1, checked
embed_tokens, lm_head    rows [0, ledger_base_vocab_size)      view
=======================  ===================================  ===============

I is the uniform FFN width read from the headers (16,640 on S5 = 4,352 trunk +
12,288 K3 channels, the order c10_union_install writes them in: trunk first).
There is NO gain tensor: the 0.03 is inside the jointly refit down columns, and
any FFN tensor other than up/down is a refusal rather than a tensor this map
would silently drop.  Band rows (`embed`/`lm_head` rows >= the base vocab), the
media bridge/head, the aux heads and any charted media block are EXTRAS: served
by Chimera name through `extra()`, never mixed into the qwen3_5 names.  Charted
blocks sit BETWEEN text blocks (OMNI1: blocks 32-36 of 69), so qwen3_5 layer t is
the t-th non-charted Chimera block (`text_blocks[t]`), not `blocks.t`.

"Tensor-exact" is checked, not assumed.  A `1 + w` norm is served as `c - 1`
in the source dtype when `bf16(c - 1) + 1 == c` holds for every element, else in
fp32 when `fp32(c - 1) + 1 == c` holds, else refused; the dtype each norm came
out in is recorded (`norm_dtypes`), because a bf16 student that loads an fp32
norm rounds it.  An output-norm whose rows are not all equal is refused.

The streamed engine (`distill/tps_probe.py`) reads its checkpoint through a
`ShardReader` with `weight_map`, `has`, `get`, `names_under`, `bytes_read` and
`total_size`; this reader has the same surface and the same names, so a
`StreamedLayerList` drives it with `load_state_dict(strict=True)` unchanged.

Usage (server; CPU; read-only)::

    PY=/home/cereal/model-surgery-lab/venv/bin/python
    $PY p1_qwen35_reader.py map    --chimera DIR [--parent-config PDIR] [--plan PLAN.json]
    $PY p1_qwen35_reader.py parity --chimera DIR --parent-dir PDIR --receipt OUT.json \\
        [--plan PLAN.json [--no-verify-root-sha]]

With `--plan`, the opened directory must digest to the plan's sealed root
(`--verify-root-sha`, the default) or, under the recorded opt-out, BE the plan's
root directory; see `DeltaOverlay`.

`parity` reads every mapped tensor through the reader AND directly through
`safetensors.safe_open`, asserts equality, pushes the reader's tensor back through
the importer's forward rewrite and asserts it equals the Chimera bytes, checks the
name/shape set against the parent's text tensors, accounts for every Chimera
element, and writes a receipt.  Forward parity (R-Q) is step 1's, not this tool's.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: 1.0.1: a page budget -- mapped pages are dropped (MADV_DONTNEED) once this
#: process has read `page_budget_bytes` through the maps, so an unchanged
#: streamed engine's RSS stays bounded over a full 54 GB pass.  1.0.0's parity
#: receipt read 102.6 GB peak RSS, all of it file-backed mapped pages.
#: 1.0.2: `write_qwen35_config` + `tps_probe_seam`, so tps_probe's own
#: `build_streamed_text_model` runs on this reader with no edit to tps_probe.
#: 1.0.3: `seam-smoke` -- tps_probe's stock builder on a real child, every layer
#: strict-loaded through the seam, plus an optional finiteness-only forward.
#: 1.0.4: the delta root check FAILS CLOSED.  1.0.3 matched a plan's root by
#: shard names and sizes only, so any same-architecture sibling (another c10
#: child, a clone rewritten in place) took the plan and composed the delta onto
#: bytes it was never trained on.  Now the opened directory must digest, the
#: ledger's way, to the plan's sealed `root.directory_sha256` (default, on every
#: CLI), or -- only under the recorded opt-out `--no-verify-root-sha` -- BE the
#: plan's root directory.  And `parity --plan` checks against an independently
#: composed reference instead of the raw base, so a delta child has a parity arm.
#: 1.0.5: charted media blocks.  OMNI1 is 69 blocks: c16_install_media INSERTED
#: five `charted` blocks at blocks 32-36, so text layer t is block t for t < 32
#: and block t + 5 after them.  1.0.4 read `blocks.t` as text layer t and refused
#: OMNI1 at block 32 ("FFN carries []").  Now text layer t is the t-th block whose
#: `block_pattern` kind is not `charted`; a charted block must carry `charted.*`
#: tensors only and must be listed, at the same index, in the config's
#: `charted_blocks` (else refused), and its tensors are extras like every other
#: media tensor.  `num_layers` stays the TEXT layer count every consumer means;
#: `num_blocks` and `text_blocks` record the Chimera indices.
TOOL_VERSION = "p1_qwen35_reader 1.0.5"
SMOKE_SCHEMA = "neural-forge-omni-p1-seam-smoke-v1"
RECEIPT_SCHEMA = "neural-forge-omni-p1-reader-parity-v1"
DELTA_PLAN_SCHEMA = "neural-forge-omni-delta-plan-v1"
CHIMERA_CHECKPOINT_SCHEMA = "chimera_omni.checkpoint.v1"

QWEN_TEXT_PREFIX = "model.language_model."
QWEN_LAYER_PREFIX = "model.language_model.layers."
QWEN_LM_HEAD = "lm_head.weight"

#: Chimera block kind -> qwen3_5 `layer_types` entry.
LAYER_TYPE = {"kda": "linear_attention", "gated_gqa": "full_attention"}

#: The only tensors a Chimera FFN may carry.  A third one (a gain, a bias) would
#: be a tensor the qwen3_5 layout has no slot for, so it is a refusal.
FFN_TENSORS = ("ffn.up.weight", "ffn.down.weight")

_DTYPE_NAMES = {
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64",
    "BOOL": "bool",
}


class P1Error(RuntimeError):
    """The reader refused: the map would not be tensor-exact, or the input moved."""


def _torch() -> Any:
    import torch

    return torch


# --------------------------------------------------------------------------
# canonical seal (byte-identical to ledger/oc_seal.py, kept here so the reader
# has no import path into the ledger tree)
# --------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def verify_sealed(body: Mapping[str, Any], what: str) -> None:
    claimed = body.get("receipt_sha")
    probe = json.loads(json.dumps(dict(body), sort_keys=True))
    probe["receipt_sha"] = "self"
    if hashlib.sha256(_canonical_bytes(probe)).hexdigest() != claimed:
        raise P1Error(f"{what} does not verify against its own receipt_sha")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["receipt_sha"] = "self"
    sealed["receipt_sha"] = hashlib.sha256(_canonical_bytes(sealed)).hexdigest()
    return sealed


#: ledger/oc_seal.py's digest walk, restated for the same reason as the seal
#: above.  A restated contract decays silently, so the tests pin both constants
#: and the digest itself equal to `oc_seal.directory_digest` on real directories.
DIGEST_EXCLUDE_SUFFIXES = (".partial", ".pyc", ".lock")
DIGEST_EXCLUDE_DIRS = ("__pycache__", ".git", ".ipynb_checkpoints")


def directory_digest(root: str | Path) -> tuple[str, dict[str, str]]:
    """`oc_seal.directory_digest`: sha256 over sorted [[path, sha256], ...].

    Every ordinary file under `root` except the excluded suffixes and
    directories; a symlink anywhere is a refusal (a digest that skips bytes can
    lie).  Returns the digest and `{relative path: file sha256}`.
    """

    base = Path(root).expanduser().absolute()
    if not base.is_dir():
        raise P1Error(f"{base} is not a directory")
    files: dict[str, str] = {}
    for here, dirs, names in os.walk(base):
        for name in sorted(dirs):
            if name not in DIGEST_EXCLUDE_DIRS and (Path(here) / name).is_symlink():
                raise P1Error(f"{Path(here) / name} is a symlinked directory; not digested")
        dirs[:] = sorted(d for d in dirs if d not in DIGEST_EXCLUDE_DIRS)
        for name in sorted(names):
            if name.endswith(DIGEST_EXCLUDE_SUFFIXES):
                continue
            path = Path(here) / name
            if path.is_symlink():
                raise P1Error(f"{path} is a symlink; not digested")
            files[path.relative_to(base).as_posix()] = file_sha256(path)
    if not files:
        raise P1Error(f"{base} contains no files to digest")
    pairs = sorted([path, sha] for path, sha in files.items())
    return hashlib.sha256(_canonical_bytes(pairs)).hexdigest(), files


# --------------------------------------------------------------------------
# zero-copy shard access
# --------------------------------------------------------------------------


class ChimeraShards:
    """Every tensor of a safetensors directory as a read-only mmap view.

    `view(name)` returns a tensor that SHARES the mapped file pages: nothing is
    read until it is touched, nothing is copied, and the page cache is the only
    cache.  The mapping is read-only, so an in-place write through a view is a
    fault, not a silent edit of a registered checkpoint.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise P1Error(f"{self.directory} is not a directory")
        shards = sorted(self.directory.glob("*.safetensors"))
        if not shards:
            raise P1Error(f"{self.directory} holds no *.safetensors")
        self.specs: dict[str, dict[str, Any]] = {}
        self.files: dict[str, dict[str, Any]] = {}
        for shard in shards:
            if shard.is_symlink():
                raise P1Error(f"{shard} is a symlink; read the target directly")
            size = shard.stat().st_size
            with open(shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            data_start = 8 + length
            self.files[shard.name] = {"bytes": size, "data_start": data_start}
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                if name in self.specs:
                    raise P1Error(f"tensor {name!r} appears in two shards")
                start, end = (int(v) for v in spec["data_offsets"])
                if not 0 <= start <= end <= size - data_start:
                    raise P1Error(f"{shard.name}: {name} spans outside the file")
                self.specs[name] = {
                    "file": shard.name,
                    "dtype": str(spec["dtype"]),
                    "shape": [int(d) for d in spec["shape"]],
                    "offset": data_start + start,
                    "nbytes": end - start,
                }
        self._maps: dict[str, Any] = {}
        self._handles: dict[str, Any] = {}

    def names(self) -> list[str]:
        return sorted(self.specs)

    def has(self, name: str) -> bool:
        return name in self.specs

    def spec(self, name: str) -> dict[str, Any]:
        if name not in self.specs:
            raise P1Error(f"{name!r} is not a tensor of {self.directory}")
        return self.specs[name]

    def _map(self, filename: str) -> Any:
        mapped = self._maps.get(filename)
        if mapped is None:
            handle = open(self.directory / filename, "rb")
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._handles[filename] = handle
            self._maps[filename] = mapped
        return mapped

    def view(self, name: str) -> Any:
        torch = _torch()
        spec = self.spec(name)
        dtype = getattr(torch, _DTYPE_NAMES[spec["dtype"]])
        elements = 1
        for dim in spec["shape"]:
            elements *= dim
        if elements == 0:
            return torch.empty(spec["shape"], dtype=dtype)
        with warnings.catch_warnings():
            # the mapping is read-only by design; torch warns that it is
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            flat = torch.frombuffer(
                self._map(spec["file"]), dtype=dtype, count=elements, offset=spec["offset"]
            )
        return flat.view(spec["shape"])

    def drop_pages(self) -> int:
        """Drop this process's mapped pages; they stay in the page cache.

        Safe with views still alive: the mapping is shared, file-backed and
        read-only, so a later touch re-faults the same bytes from the cache.
        Returns how many maps were advised.
        """

        advised = 0
        flag = getattr(mmap, "MADV_DONTNEED", None)
        for mapped in self._maps.values():
            if flag is not None and hasattr(mapped, "madvise"):
                mapped.madvise(flag)
                advised += 1
        return advised

    def element_count(self, name: str) -> int:
        count = 1
        for dim in self.spec(name)["shape"]:
            count *= dim
        return count

    def close(self) -> None:
        for mapped in self._maps.values():
            mapped.close()
        for handle in self._handles.values():
            handle.close()
        self._maps.clear()
        self._handles.clear()


# --------------------------------------------------------------------------
# delta overlay (a sealed oc_checkpoint.py delta-compose plan)
# --------------------------------------------------------------------------


def _region_index(slices: Sequence[Sequence[int]] | None, ndim: int) -> tuple[Any, ...]:
    index: list[Any] = [slice(None)] * ndim
    for dim, start, stop in slices or []:
        index[int(dim)] = slice(int(start), int(stop))
    return tuple(index)


class DeltaOverlay:
    """Compose base + every delta in a plan, one target tensor at a time.

    The ledger (`oc_checkpoint.py delta-compose`) has already checked every pin
    and every op against the headers; this re-checks, on the bytes it actually
    opens, the plan's seal, every delta file's sha256, and that the opened root
    IS the plan's root.  That last check fails closed:

    * `verify_root_sha=True` (the default): the opened directory is re-digested
      the ledger's way and must equal the plan's sealed `root.directory_sha256`
      (and each shard its per-file sha256 when the plan carries them).  The
      plan's `deep` flag does not matter: the re-hash happens here.  A
      byte-identical copy at another path passes, and both paths are recorded.
    * `verify_root_sha=False` (the explicit opt-out, recorded in `root_check`):
      no re-hash, so the opened directory must resolve to the plan's root
      directory.  Names and sizes cannot tell siblings apart -- they depend only
      on names, shapes and dtypes -- so they are never the identity check.

    Apply rule (the plan states it too): base -> fp32 -> each link in chain
    order, each entry in recipe order -> one cast back to the base dtype.
    """

    def __init__(
        self, plan: Mapping[str, Any], root: ChimeraShards, *, verify_root_sha: bool = True
    ) -> None:
        if plan.get("schema") != DELTA_PLAN_SCHEMA:
            raise P1Error(f"plan schema is {plan.get('schema')!r}, not {DELTA_PLAN_SCHEMA}")
        verify_sealed(plan, "the delta plan")
        self.plan = plan
        plan_root = plan["root"]
        on_disk = {
            name: info["bytes"] for name, info in root.files.items()
        }
        declared = {
            Path(item["path"]).name: int(item["bytes"]) for item in plan_root["files"]
        }
        if on_disk != declared:
            raise P1Error(
                f"the plan's root {plan_root['checkpoint_id']} declares shards "
                f"{sorted(declared.items())[:3]}..., {root.directory} holds "
                f"{sorted(on_disk.items())[:3]}...; this is not the base the plan is about"
            )
        opened = root.directory.expanduser().resolve()
        planned = Path(str(plan_root["directory"])).expanduser().resolve()
        self.root_check: dict[str, Any] = {
            "mode": "directory_sha256" if verify_root_sha else "path_identity_opt_out",
            "plan_root_checkpoint_id": plan_root["checkpoint_id"],
            "plan_root_directory": str(planned),
            "opened_directory": str(opened),
            "same_path": opened == planned,
            "sealed_directory_sha256": plan_root["directory_sha256"],
            "live_directory_sha256": None,
        }
        if verify_root_sha:
            live, per_file = directory_digest(opened)
            self.root_check["live_directory_sha256"] = live
            for item in plan_root["files"]:
                want = item.get("sha256")
                if want is not None and per_file.get(str(item["path"])) != want:
                    raise P1Error(
                        f"BASE DIGEST MISMATCH: root shard {item['path']} in {opened} digests "
                        f"{str(per_file.get(str(item['path'])))[:16]}, the plan says {want[:16]}; "
                        "the delta would compose onto bytes it was not trained on"
                    )
            if live != plan_root["directory_sha256"]:
                raise P1Error(
                    f"BASE DIGEST MISMATCH: {opened} digests {live[:16]}, the plan's root "
                    f"{plan_root['checkpoint_id']} is sealed at "
                    f"{str(plan_root['directory_sha256'])[:16]}; the delta would compose "
                    "onto bytes it was not trained on"
                )
        elif opened != planned:
            raise P1Error(
                f"{opened} is not the plan's root directory {planned}, and with "
                "--no-verify-root-sha nothing else identifies it (names and sizes match "
                "any same-architecture sibling); open the root itself or verify its digest"
            )
        self.links: list[tuple[dict[str, Any], ChimeraShards]] = []
        self.ops: dict[str, list[tuple[dict[str, Any], ChimeraShards]]] = {}
        for link in plan["deltas"]:
            directory = Path(link["directory"])
            for item in link["files"]:
                live = file_sha256(directory / item["path"])
                if live != item["sha256"]:
                    raise P1Error(
                        f"delta {link['checkpoint_id']} file {item['path']} digests "
                        f"{live[:16]}, the plan says {item['sha256'][:16]}"
                    )
            shards = ChimeraShards(directory)
            self.links.append((link, shards))
            for entry in link["entries"]:
                if entry["target"] not in root.specs:
                    raise P1Error(f"plan entry targets {entry['target']!r}, absent in the root")
                self.ops.setdefault(str(entry["target"]), []).append((entry, shards))

    def touches(self, name: str) -> bool:
        return name in self.ops

    def compose(self, name: str, base: Any) -> Any:
        torch = _torch()
        out = base.to(dtype=torch.float32, copy=True)
        for entry, shards in self.ops[name]:
            index = _region_index(entry.get("slice"), out.dim())
            op = entry["op"]
            if op == "replace":
                out[index] = shards.view(entry["tensor"]).to(torch.float32)
            elif op == "lora":
                a = shards.view(entry["a"]).to(torch.float32)
                b = shards.view(entry["b"]).to(torch.float32)
                out[index] += (float(entry["alpha"]) / a.shape[0]) * (b @ a)
            elif op == "scale":
                gain = shards.view(entry["tensor"]).to(torch.float32)
                shape = [1] * out[index].dim()
                shape[int(entry["dim"])] = -1
                out[index] *= gain.view(shape)
            else:
                raise P1Error(f"plan op {op!r} is not replace/lora/scale")
        if not bool(torch.isfinite(out).all()):
            raise P1Error(f"composing {name} produced a non-finite value")
        return out.to(dtype=base.dtype)

    def close(self) -> None:
        for _link, shards in self.links:
            shards.close()


# --------------------------------------------------------------------------
# the qwen3_5 <- Chimera map
# --------------------------------------------------------------------------


def _chimera_config(directory: Path) -> dict[str, Any]:
    path = directory / "config.json"
    if not path.is_file():
        raise P1Error(f"{directory} has no config.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CHIMERA_CHECKPOINT_SCHEMA or "config" not in document:
        raise P1Error(f"{path} is not a {CHIMERA_CHECKPOINT_SCHEMA} manifest")
    return document


class Qwen35LayoutReader:
    """A Chimera checkpoint served under qwen3_5 names, zero-copy where possible.

    `kinds[name]` says how each tensor is produced: `view` (a slice of one
    Chimera tensor, no copy), `cat` / `interleave` (assembled per access from
    zero-copy parts, see `parts`), `one_plus` (a 1 + w norm, served as c - 1
    after the exactness check) or `row0` (a broadcast row, after the
    rows-all-equal check).  With a delta plan, every Chimera tensor the plan
    touches is composed first, and the map is applied to the composed tensor.
    """

    def __init__(
        self,
        chimera_dir: str | Path,
        *,
        plan: Mapping[str, Any] | None = None,
        device: str = "cpu",
        require_uniform_ffn: bool = True,
        verify_root_sha: bool = True,
        page_budget_bytes: int = 8 << 30,
    ) -> None:
        self.directory = Path(chimera_dir)
        self.manifest = _chimera_config(self.directory)
        self.config = self.manifest["config"]
        self.shards = ChimeraShards(self.directory)
        self.overlay = (
            DeltaOverlay(plan, self.shards, verify_root_sha=verify_root_sha) if plan else None
        )
        self.device = device
        self.bytes_read = 0
        self.page_budget_bytes = int(page_budget_bytes)
        self.page_drops = 0
        self._since_drop = 0
        self.norm_dtypes: dict[str, str] = {}
        self.base_vocab = int(self.config["ledger_base_vocab_size"])
        self.vocab = int(self.config["vocab_size"])
        # Chimera blocks, charted media blocks included; `num_layers` (the text
        # layer count) is set by `_check_ffn_and_kinds`
        self.num_blocks = int(self.config["num_layers"])
        self.num_layers = 0
        self.hidden = int(self.config["hidden_size"])
        self.tied = bool(self.config.get("tie_word_embeddings", False))
        self._check_ffn_and_kinds(require_uniform_ffn)
        self.specs: dict[str, dict[str, Any]] = {}
        self._build_map()
        self.weight_map = {name: spec["file"] for name, spec in self.specs.items()}
        self.total_size = sum(spec["nbytes"] for spec in self.specs.values())
        self._extras = self._build_extras()

    # -- structure ----------------------------------------------------------

    def _check_ffn_and_kinds(self, require_uniform: bool) -> None:
        names = set(self.shards.names())
        pattern = list(self.config["block_pattern"])
        self.block_kinds: list[str] = []
        self.text_blocks: list[int] = []
        self.charted_blocks: list[int] = []
        widths: list[int] = []
        for index in range(self.num_blocks):
            base = f"blocks.{index}."
            declared = pattern[index % len(pattern)]
            if self.config.get("force_final_gated_mla") and index == self.num_blocks - 1:
                declared = "gated_mla"
            if declared == "charted":
                # a media unit: no text tensor may hide in it, or the map would drop it
                own = sorted(n[len(base):] for n in names if n.startswith(base))
                stray = [n for n in own if not n.startswith("charted.")]
                if not own or stray:
                    raise P1Error(
                        f"block {index} is `charted` in block_pattern but carries "
                        f"{stray[:4] or 'no tensors'}; a charted block holds charted.* only"
                    )
                self.charted_blocks.append(index)
                continue
            ffn = sorted(n[len(base):] for n in names if n.startswith(base + "ffn."))
            if ffn != sorted(FFN_TENSORS):
                raise P1Error(
                    f"layer {index} FFN carries {ffn}, not exactly {list(FFN_TENSORS)}: a "
                    "gain, bias or extra tensor has no qwen3_5 slot and would be dropped"
                )
            up = self.shards.spec(base + "ffn.up.weight")["shape"]
            down = self.shards.spec(base + "ffn.down.weight")["shape"]
            if up[0] % 2 or up[0] // 2 != down[1] or up[1] != self.hidden or down[0] != self.hidden:
                raise P1Error(
                    f"layer {index}: fused up {up} and down {down} are not [2I, h] / [h, I]"
                )
            widths.append(up[0] // 2)
            if f"{base}mixer.q_conv.conv.weight" in names:
                kind = "kda"
            elif f"{base}mixer.gate_proj.weight" in names:
                kind = "gated_gqa"
            else:
                raise P1Error(f"layer {index} is neither a KDA nor a gated-GQA block")
            if kind != declared:
                raise P1Error(
                    f"layer {index} tensors say {kind}, the config's block_pattern says "
                    f"{declared}"
                )
            self.block_kinds.append(kind)
            self.text_blocks.append(index)
        listed = sorted(int(item["layer_index"]) for item in self.config.get("charted_blocks") or [])
        if listed != self.charted_blocks:
            raise P1Error(
                f"block_pattern puts charted blocks at {self.charted_blocks}, the config's "
                f"charted_blocks list says {listed}; which blocks are text is ambiguous"
            )
        if not self.text_blocks:
            raise P1Error("no text block: every block is charted")
        self.num_layers = len(self.text_blocks)
        if require_uniform and len(set(widths)) != 1:
            raise P1Error(f"FFN width is not uniform across layers: {sorted(set(widths))}")
        self.ffn_widths = widths
        self.ffn_width = widths[0] if len(set(widths)) == 1 else None
        metadata = self.config.get("metadata") or {}
        trunk = metadata.get("c10_trunk_intermediate_size")
        k3 = metadata.get("c10_k3_channels")
        self.channel_partition: dict[str, list[int]] | None = None
        if trunk is not None or k3 is not None:
            trunk_n, k3_n = int(trunk or 0), int(k3 or 0)
            if self.ffn_width is None or trunk_n + k3_n != self.ffn_width:
                raise P1Error(
                    f"config metadata says {trunk_n} trunk + {k3_n} K3 channels, the FFN "
                    f"width is {self.ffn_width}"
                )
            # c10_union_install writes [trunk | K3] in both halves of the fused up
            # and in the down columns (c10 1.5.1, `torch.cat([gate_q, gate_p[chosen]])`).
            self.channel_partition = {"trunk": [0, trunk_n], "k3": [trunk_n, trunk_n + k3_n]}

    def _add(self, name: str, kind: str, parts: list[tuple[str, Any]], shape: list[int],
             **extra: Any) -> None:
        first = self.shards.spec(parts[0][0])
        nbytes = 1
        for dim in shape:
            nbytes *= dim
        self.specs[name] = {
            "kind": kind,
            "parts": parts,
            "shape": shape,
            "dtype": first["dtype"],
            "file": first["file"],
            "nbytes": nbytes * (2 if first["dtype"] in ("BF16", "F16") else 4),
            **extra,
        }

    def _view(self, qname: str, cname: str, rows: tuple[int, int] | None = None) -> None:
        shape = list(self.shards.spec(cname)["shape"])
        if rows is not None:
            shape[0] = rows[1] - rows[0]
        self._add(qname, "view", [(cname, rows)], shape)

    def _build_map(self) -> None:
        cfg = self.config
        text = QWEN_TEXT_PREFIX
        embed = self.shards.spec("embed.weight")["shape"]
        if embed != [self.vocab, self.hidden]:
            raise P1Error(f"embed.weight is {embed}, the config says [{self.vocab}, {self.hidden}]")
        self._view(f"{text}embed_tokens.weight", "embed.weight", (0, self.base_vocab))
        self._add(f"{text}norm.weight", "one_plus", [("norm.weight", None)], [self.hidden])
        if not self.tied:
            self._view(QWEN_LM_HEAD, "lm_head.weight", (0, self.base_vocab))
        heads = int(cfg["num_heads"])
        head_dim = int(cfg["attention_head_dim"])
        for index, kind in enumerate(self.block_kinds):
            q = f"{QWEN_LAYER_PREFIX}{index}."
            c = f"blocks.{self.text_blocks[index]}."
            width = self.ffn_widths[index]
            self._add(f"{q}input_layernorm.weight", "one_plus", [(f"{c}norm1.weight", None)], [self.hidden])
            self._add(f"{q}post_attention_layernorm.weight", "one_plus",
                      [(f"{c}norm2.weight", None)], [self.hidden])
            self._view(f"{q}mlp.gate_proj.weight", f"{c}ffn.up.weight", (0, width))
            self._view(f"{q}mlp.up_proj.weight", f"{c}ffn.up.weight", (width, 2 * width))
            self._view(f"{q}mlp.down_proj.weight", f"{c}ffn.down.weight")
            if kind == "kda":
                parts = [(f"{c}mixer.{p}_proj.weight", None) for p in ("q", "k", "v")]
                rows = sum(self.shards.spec(p)["shape"][0] for p, _ in parts)
                self._add(f"{q}linear_attn.in_proj_qkv.weight", "cat", parts, [rows, self.hidden])
                conv = [(f"{c}mixer.{p}_conv.conv.weight", None) for p in ("q", "k", "v")]
                kernel = self.shards.spec(conv[0][0])["shape"]
                rows = sum(self.shards.spec(p)["shape"][0] for p, _ in conv)
                self._add(f"{q}linear_attn.conv1d.weight", "cat", conv, [rows, *kernel[1:]])
                for qn, cn in (
                    ("in_proj_z.weight", "output_gate.weight"),
                    ("in_proj_b.weight", "write_strength.weight"),
                    ("in_proj_a.weight", "decay_down.weight"),
                    ("dt_bias", "decay_bias"),
                    ("A_log", "decay_log_scale"),
                    ("out_proj.weight", "out.weight"),
                ):
                    self._view(f"{q}linear_attn.{qn}", f"{c}mixer.{cn}")
                norm = self.shards.spec(f"{c}mixer.output_norm.weight")["shape"]
                self._add(f"{q}linear_attn.norm.weight", "row0",
                          [(f"{c}mixer.output_norm.weight", None)], [norm[1]])
            else:
                parts = [(f"{c}mixer.q_proj.weight", None), (f"{c}mixer.gate_proj.weight", None)]
                rows = sum(self.shards.spec(p)["shape"][0] for p, _ in parts)
                if rows != heads * 2 * head_dim:
                    raise P1Error(f"layer {index}: q+gate rows {rows} != {heads} x 2 x {head_dim}")
                self._add(f"{q}self_attn.q_proj.weight", "interleave", parts, [rows, self.hidden],
                          heads=heads, head_dim=head_dim)
                for qn, cn in (("k_proj", "k_proj"), ("v_proj", "v_proj"), ("o_proj", "out")):
                    self._view(f"{q}self_attn.{qn}.weight", f"{c}mixer.{cn}.weight")
                for qn in ("q_norm", "k_norm"):
                    self._add(f"{q}self_attn.{qn}.weight", "one_plus",
                              [(f"{c}mixer.{qn}.weight", None)], [head_dim])

    def _build_extras(self) -> dict[str, dict[str, Any]]:
        """Every Chimera element the qwen3_5 map does not serve, by Chimera name."""

        consumed = {cname for spec in self.specs.values() for cname, _ in spec["parts"]}
        extras: dict[str, dict[str, Any]] = {}
        band = self.vocab - self.base_vocab
        if band:
            extras["embed.weight[base:]"] = {"source": "embed.weight", "rows": [self.base_vocab, self.vocab]}
            if not self.tied:
                extras["lm_head.weight[base:]"] = {"source": "lm_head.weight",
                                                   "rows": [self.base_vocab, self.vocab]}
        for name in self.shards.names():
            if name not in consumed:
                extras[name] = {"source": name, "rows": None}
        return extras

    # -- the ShardReader surface tps_probe.py drives --------------------------

    def has(self, name: str) -> bool:
        return name in self.specs

    def names_under(self, prefix: str) -> list[str]:
        return sorted(name for name in self.specs if name.startswith(prefix))

    def names(self) -> list[str]:
        return sorted(self.specs)

    def kind(self, name: str) -> str:
        return str(self.specs[name]["kind"])

    def _raw(self, cname: str) -> Any:
        base = self.shards.view(cname)
        if self.overlay is not None and self.overlay.touches(cname):
            return self.overlay.compose(cname, base)
        return base

    def parts(self, name: str) -> list[Any]:
        """The zero-copy pieces a tensor is made of, before any assembly."""

        if name not in self.specs:
            raise P1Error(f"{name!r} is not a qwen3_5 tensor of this map")
        out = []
        for cname, rows in self.specs[name]["parts"]:
            tensor = self._raw(cname)
            out.append(tensor[rows[0]:rows[1]] if rows else tensor)
        return out

    def drop_pages(self) -> None:
        self.shards.drop_pages()
        if self.overlay is not None:
            for _link, shards in self.overlay.links:
                shards.drop_pages()
        self.page_drops += 1
        self._since_drop = 0

    def get(self, name: str) -> Any:
        torch = _torch()
        if name not in self.specs:
            raise P1Error(f"{name!r} is not a qwen3_5 tensor of this map")
        if self.page_budget_bytes > 0 and self._since_drop >= self.page_budget_bytes:
            # before this read, never after it: the tensor handed out last is
            # still valid either way, but this keeps the newest pages resident
            self.drop_pages()
        spec = self.specs[name]
        pieces = self.parts(name)
        kind = spec["kind"]
        if kind == "view":
            tensor = pieces[0]
        elif kind == "cat":
            tensor = torch.cat(pieces, dim=0)
        elif kind == "interleave":
            heads, head_dim = int(spec["heads"]), int(spec["head_dim"])
            query, gate = pieces
            # the inverse of qwen35.py: per_head = fused.view(H, 2*hd, h);
            # q = per_head[:, :hd], gate = per_head[:, hd:]
            tensor = torch.stack(
                (query.reshape(heads, head_dim, -1), gate.reshape(heads, head_dim, -1)), dim=1
            ).reshape(heads * 2 * head_dim, -1)
        elif kind == "row0":
            rows = pieces[0]
            if not bool((rows == rows[0:1]).all()):
                raise P1Error(
                    f"{name}: the Chimera per-head output norm rows are not all equal, so "
                    "no single qwen3_5 row reproduces it (not tensor-exact)"
                )
            tensor = rows[0]
        elif kind == "one_plus":
            stored = pieces[0]
            exact = stored.float()
            native = (exact - 1.0).to(stored.dtype)
            if bool(((native.float() + 1.0) == exact).all()):
                tensor = native
            else:
                wide = exact - 1.0
                if not bool(((wide + 1.0) == exact).all()):
                    raise P1Error(f"{name}: c - 1 is not exact even in fp32")
                tensor = wide
            self.norm_dtypes[name] = str(tensor.dtype).replace("torch.", "")
        else:
            raise P1Error(f"{name}: unknown kind {kind!r}")
        if list(tensor.shape) != spec["shape"]:
            raise P1Error(f"{name}: produced {list(tensor.shape)}, mapped {spec['shape']}")
        self.bytes_read += tensor.numel() * tensor.element_size()
        self._since_drop += tensor.numel() * tensor.element_size()
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

    # -- extras ---------------------------------------------------------------

    def extras(self) -> list[str]:
        return sorted(self._extras)

    def extra(self, name: str) -> Any:
        if name not in self._extras:
            raise P1Error(f"{name!r} is not an extra of this checkpoint")
        item = self._extras[name]
        tensor = self._raw(item["source"])
        if item["rows"]:
            tensor = tensor[item["rows"][0]:item["rows"][1]]
        return tensor

    # -- accounting -----------------------------------------------------------

    def coverage(self) -> dict[str, Any]:
        """Every Chimera element is served exactly once, by the map or as an extra.

        A row0 source counts as covered in full: its other rows are asserted
        equal to row 0 on every read, so they carry no information the map drops.
        """

        covered: dict[str, int] = {}
        for spec in self.specs.values():
            for cname, rows in spec["parts"]:
                full = self.shards.element_count(cname)
                if rows is None:
                    covered[cname] = covered.get(cname, 0) + full
                else:
                    per_row = full // self.shards.spec(cname)["shape"][0]
                    covered[cname] = covered.get(cname, 0) + (rows[1] - rows[0]) * per_row
        for item in self._extras.values():
            cname = item["source"]
            full = self.shards.element_count(cname)
            if item["rows"] is None:
                covered[cname] = covered.get(cname, 0) + full
            else:
                per_row = full // self.shards.spec(cname)["shape"][0]
                covered[cname] = covered.get(cname, 0) + (item["rows"][1] - item["rows"][0]) * per_row
        total = sum(self.shards.element_count(n) for n in self.shards.names())
        wrong = {
            name: [covered.get(name, 0), self.shards.element_count(name)]
            for name in self.shards.names()
            if covered.get(name, 0) != self.shards.element_count(name)
        }
        mapped = sum(
            self.shards.element_count(c) if r is None
            else (r[1] - r[0]) * (self.shards.element_count(c) // self.shards.spec(c)["shape"][0])
            for spec in self.specs.values() for c, r in spec["parts"]
        )
        return {
            "chimera_tensors": len(self.shards.names()),
            "chimera_elements": total,
            "mapped_elements": mapped,
            "extra_elements": total - mapped,
            "mapped_tensors": len(self.specs),
            "extra_entries": len(self._extras),
            "miscovered": wrong,
        }

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for spec in self.specs.values():
            kinds[spec["kind"]] = kinds.get(spec["kind"], 0) + 1
        return {
            "tool_version": TOOL_VERSION,
            "chimera_dir": str(self.directory),
            "chimera_commit": self.manifest.get("chimera_commit"),
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "text_blocks": self.text_blocks,
            "charted_blocks": [
                {"block": int(item["layer_index"]), "component": item.get("component"),
                 "source": item.get("source")}
                for item in self.config.get("charted_blocks") or []
            ],
            "layer_types": [LAYER_TYPE.get(k, k) for k in self.block_kinds],
            "ffn_width": self.ffn_width,
            "channel_partition": self.channel_partition,
            "base_vocab": self.base_vocab,
            "vocab": self.vocab,
            "tied": self.tied,
            "kinds": kinds,
            "zero_copy_tensors": kinds.get("view", 0),
            "delta_plan": self.overlay.plan["receipt_sha"] if self.overlay else None,
            "root_check": self.overlay.root_check if self.overlay else None,
            "page_budget_bytes": self.page_budget_bytes,
            "coverage": self.coverage(),
        }

    def qwen35_config(self, parent_config: Mapping[str, Any]) -> dict[str, Any]:
        """The parent's HF config with the child's FFN width; everything else checked equal."""

        config = json.loads(json.dumps(dict(parent_config)))
        text = config.get("text_config", config)
        if int(text["hidden_size"]) != self.hidden or int(text["num_hidden_layers"]) != self.num_layers:
            raise P1Error("the parent config's hidden size / layer count is not this child's")
        if int(text["vocab_size"]) != self.base_vocab:
            raise P1Error(
                f"the parent's vocab_size {text['vocab_size']} != the child's base vocab "
                f"{self.base_vocab}"
            )
        declared = list(text.get("layer_types") or [])
        mine = [LAYER_TYPE[k] for k in self.block_kinds]
        if declared and declared[: self.num_layers] != mine:
            raise P1Error("the parent's layer_types do not match the child's block kinds")
        if self.ffn_width is None:
            raise P1Error("a qwen3_5 config needs one FFN width; this child's is not uniform")
        text["intermediate_size"] = self.ffn_width
        return config

    def close(self) -> None:
        self.shards.close()
        if self.overlay is not None:
            self.overlay.close()


def write_qwen35_config(
    reader: Qwen35LayoutReader, parent_config_dir: str | Path, out_dir: str | Path
) -> Path:
    """A directory holding ONLY the child's qwen3_5 `config.json` (no weights).

    `tps_probe.build_streamed_text_model(model_dir, ...)` reads its HF config
    with `Qwen3_5Config.from_pretrained(model_dir)`; this is that directory.
    The weights stay where they are and come through `tps_probe_seam`.
    """

    parent = json.loads((Path(parent_config_dir) / "config.json").read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stray = [p.name for p in out.iterdir() if p.name != "config.json"]
    if stray:
        raise P1Error(f"{out} holds {stray[:3]}; the config directory holds config.json only")
    target = out / "config.json"
    target.write_text(json.dumps(reader.qwen35_config(parent), indent=1), encoding="utf-8")
    return target


@contextlib.contextmanager
def tps_probe_seam(reader: Qwen35LayoutReader, tps_probe_module: Any) -> Iterable[None]:
    """Inside this block, tps_probe's `ShardReader(model_dir, device)` IS `reader`.

    The one place the streamed engine opens a checkpoint is that constructor;
    everything after it (`get`, `has`, `names_under`, `bytes_read`,
    `total_size`, `weight_map`) is the surface this reader already has.  So the
    engine runs on OMNI1 with no edit to tps_probe.py: open the reader, write
    the config directory, and call `build_streamed_text_model(config_dir, ...)`
    in here.  The device must match the reader's.
    """

    original = tps_probe_module.ShardReader

    def construct(model_dir: Any, device: str) -> Qwen35LayoutReader:
        if str(device) != str(reader.device):
            raise P1Error(f"the engine asked for device {device}, the reader serves {reader.device}")
        return reader

    tps_probe_module.ShardReader = construct
    try:
        yield
    finally:
        tps_probe_module.ShardReader = original


def open_reader(chimera_dir: str | Path, *, plan_path: str | Path | None = None,
                **kwargs: Any) -> Qwen35LayoutReader:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8")) if plan_path else None
    return Qwen35LayoutReader(chimera_dir, plan=plan, **kwargs)


# --------------------------------------------------------------------------
# parity: every mapped tensor vs a direct read, and back through the importer
# --------------------------------------------------------------------------


def importer_forward(name: str, kind: str, tensor: Any, spec: Mapping[str, Any]) -> list[Any]:
    """The importer's own rewrite (qwen35.py @ ef1ffb9) applied to a reader tensor.

    Returns what `import_qwen35_state` would write into each Chimera part, so a
    reader that is the exact inverse gives back the Chimera bytes.  Kept to the
    importer's literal operations: `torch.split` on qkv/conv, `view(H, 2hd, h)`
    for q/gate, `weight.float() + 1.0` cast to the stored dtype for norms, a
    broadcast for the gated norm.
    """

    torch = _torch()
    if kind == "view":
        return [tensor]
    if kind == "cat":
        sizes = [int(s) for s in spec["part_rows"]]
        return list(torch.split(tensor, sizes, dim=0))
    if kind == "interleave":
        heads, head_dim = int(spec["heads"]), int(spec["head_dim"])
        per_head = tensor.view(heads, 2 * head_dim, -1)
        return [
            per_head[:, :head_dim].reshape(heads * head_dim, -1),
            per_head[:, head_dim:].reshape(heads * head_dim, -1),
        ]
    if kind == "one_plus":
        return [tensor.float() + 1.0]
    if kind == "row0":
        rows = int(spec["broadcast_rows"])
        return [tensor.view(1, -1).expand(rows, -1).contiguous()]
    raise P1Error(f"{name}: unknown kind {kind!r}")


def plan_targets(plan: Mapping[str, Any] | None) -> set[str]:
    """Every Chimera tensor a plan touches, read off the plan itself."""

    if not plan:
        return set()
    return {str(entry["target"]) for link in plan["deltas"] for entry in link["entries"]}


def reference_compose(plan: Mapping[str, Any], cname: str, base: Any) -> Any:
    """The plan's composition of `cname`, written apart from `DeltaOverlay`.

    The parity reference for a delta child: delta tensors come through
    `safetensors.safe_open` (not this module's mmap views), regions through
    chained `narrow` (not slice tuples), and the plan is walked link by link,
    entry by entry, filtering on the target (not a prebuilt op table).  The
    arithmetic is the ledger's spec (oc_checkpoint.DELTA_OPS), in its order:
    fp32, `region += (alpha / rank) * (B @ A)`, one cast at the end.
    """

    torch = _torch()
    from safetensors import safe_open

    def load(link: Mapping[str, Any], tensor_name: str) -> Any:
        directory = Path(link["directory"])
        for item in link["files"]:
            with safe_open(str(directory / item["path"]), framework="pt", device="cpu") as handle:
                if tensor_name in handle.keys():
                    return handle.get_tensor(tensor_name).to(torch.float32)
        raise P1Error(f"delta {link['checkpoint_id']} has no tensor {tensor_name!r}")

    out = base.to(torch.float32).clone()
    for link in plan["deltas"]:
        for entry in link["entries"]:
            if str(entry["target"]) != cname:
                continue
            region = out
            for dim, start, stop in entry.get("slice") or []:
                region = region.narrow(int(dim), int(start), int(stop) - int(start))
            if entry["op"] == "replace":
                region.copy_(load(link, entry["tensor"]))
            elif entry["op"] == "lora":
                a, b = load(link, entry["a"]), load(link, entry["b"])
                region += (float(entry["alpha"]) / a.shape[0]) * (b @ a)
            elif entry["op"] == "scale":
                gain = load(link, entry["tensor"])
                axis = int(entry["dim"])
                region *= gain.reshape([-1 if d == axis else 1 for d in range(region.dim())])
            else:
                raise P1Error(f"plan op {entry['op']!r} is not replace/lora/scale")
    return out.to(base.dtype)


def _parent_text_index(parent_dir: Path) -> dict[str, dict[str, Any]]:
    shards = ChimeraShards(parent_dir)
    out = {
        name: {"dtype": spec["dtype"], "shape": spec["shape"]}
        for name, spec in shards.specs.items()
        if name.startswith(QWEN_TEXT_PREFIX) or name == QWEN_LM_HEAD
    }
    shards.close()
    return out


def run_parity(args: argparse.Namespace) -> int:
    torch = _torch()
    from safetensors import safe_open

    torch.set_num_threads(int(args.threads))
    started = time.time()
    reader = open_reader(args.chimera, plan_path=args.plan, verify_root_sha=args.verify_root_sha)
    summary = reader.summary()
    # the reference composes from the plan file on its own, never via reader.overlay
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    targets = plan_targets(plan)
    parent_config = json.loads((Path(args.parent_dir) / "config.json").read_text(encoding="utf-8"))
    hf_config = reader.qwen35_config(parent_config)
    parent = _parent_text_index(Path(args.parent_dir))
    names = reader.names()
    if args.limit:
        names = names[: int(args.limit)]

    # name / shape / dtype set against the parent's text tensors
    width = reader.ffn_width
    parent_width = int((parent_config.get("text_config") or parent_config)["intermediate_size"])
    shape_problems: list[str] = []
    missing = sorted(set(parent) - set(reader.names()))
    unknown = sorted(set(reader.names()) - set(parent))
    for name in reader.names():
        if name not in parent:
            continue
        want = list(parent[name]["shape"])
        if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            want[0] = width if want[0] == parent_width else -1
        if name.endswith("mlp.down_proj.weight"):
            want[1] = width if want[1] == parent_width else -1
        if reader.specs[name]["shape"] != want:
            shape_problems.append(f"{name}: {reader.specs[name]['shape']} vs expected {want}")

    def direct(cname: str) -> Any:
        # a fresh handle per read: safetensors maps the file too, and a cached
        # handle would put the whole 54 GB of mapped pages into this RSS
        filename = reader.shards.spec(cname)["file"]
        with safe_open(str(reader.directory / filename), framework="pt", device="cpu") as handle:
            raw = handle.get_tensor(cname).clone()
        # with a plan, the reference is the plan's composition of the raw base
        return reference_compose(plan, cname, raw) if cname in targets else raw

    per_tensor: list[dict[str, Any]] = []
    failures: list[str] = []
    zero_copy = 0
    delta_composed = 0
    elements = 0
    for count, name in enumerate(names):
        spec = dict(reader.specs[name])
        kind = spec["kind"]
        try:
            got = reader.get(name)
        except P1Error as exc:
            failures.append(f"{name}: {exc}")
            per_tensor.append({"name": name, "kind": kind, "ok": False, "error": str(exc)})
            continue
        sources = [direct(cname) for cname, _rows in spec["parts"]]
        # (a) equal to a direct read through the same mapping, written independently
        if kind == "view":
            rows = spec["parts"][0][1]
            want = sources[0][rows[0]:rows[1]] if rows else sources[0]
        elif kind == "cat":
            want = torch.cat(sources, dim=0)
        elif kind == "interleave":
            heads, head_dim = int(spec["heads"]), int(spec["head_dim"])
            q_src, g_src = sources
            want = torch.empty_like(got)
            for head in range(heads):
                base = head * 2 * head_dim
                want[base:base + head_dim] = q_src[head * head_dim:(head + 1) * head_dim]
                want[base + head_dim:base + 2 * head_dim] = g_src[head * head_dim:(head + 1) * head_dim]
        elif kind == "row0":
            want = sources[0][0]
        else:  # one_plus
            want = (sources[0].float() - 1.0).to(got.dtype)
        equal_direct = bool(got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want))
        # (b) back through the importer's forward rewrite -> the Chimera bytes
        spec["part_rows"] = [s.shape[0] for s in sources]
        spec["broadcast_rows"] = sources[0].shape[0]
        forward = importer_forward(name, kind, got, spec)
        round_trip = []
        for (cname, rows), produced, source in zip(spec["parts"], forward, sources):
            target = source[rows[0]:rows[1]] if rows else source
            produced = produced.to(target.dtype)
            round_trip.append(bool(produced.shape == target.shape and torch.equal(produced, target)))
        equal_round_trip = all(round_trip)
        storage_shared = False
        composed = any(cname in targets for cname, _rows in spec["parts"])
        if kind == "view":
            base_view = reader.shards.view(spec["parts"][0][0])
            storage_shared = got.untyped_storage().data_ptr() == base_view.untyped_storage().data_ptr()
            zero_copy += int(storage_shared)
        # a view the plan does not touch must still be a zero-copy view of the base
        ok = equal_direct and equal_round_trip and (kind != "view" or storage_shared or composed)
        if not ok:
            failures.append(
                f"{name}: direct={equal_direct} round_trip={round_trip} "
                f"zero_copy={storage_shared} kind={kind}"
            )
        elements += got.numel()
        per_tensor.append({
            "name": name, "kind": kind, "shape": list(got.shape),
            "dtype": str(got.dtype).replace("torch.", ""),
            "equal_direct": equal_direct, "equal_importer_round_trip": equal_round_trip,
            "zero_copy": storage_shared, "delta_composed": composed, "ok": ok,
        })
        delta_composed += int(composed)
        del got, sources, forward, want
        reader.drop_pages()
        if args.progress and (count + 1) % 50 == 0:
            print(f"  {count + 1}/{len(names)} tensors, {len(failures)} failure(s), "
                  f"{time.time() - started:.0f} s", flush=True)

    coverage = summary["coverage"]
    norm_dtypes: dict[str, int] = {}
    for dtype in reader.norm_dtypes.values():
        norm_dtypes[dtype] = norm_dtypes.get(dtype, 0) + 1
    structural_ok = (
        not missing and not unknown and not shape_problems and not coverage["miscovered"]
    )
    passed = structural_ok and not failures and len(names) == len(reader.names())
    peak_rss = None
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (ImportError, AttributeError):
        pass
    receipt = seal({
        "schema": RECEIPT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "tool_sha256": file_sha256(Path(__file__).resolve()),
        "argv": sys.argv,
        "chimera_dir": str(reader.directory),
        "chimera_commit": reader.manifest.get("chimera_commit"),
        "chimera_index_sha256": file_sha256(reader.directory / "model.safetensors.index.json")
        if (reader.directory / "model.safetensors.index.json").is_file() else None,
        "chimera_config_sha256": file_sha256(reader.directory / "config.json"),
        "parent_dir": str(args.parent_dir),
        "plan": summary["delta_plan"],
        "summary": summary,
        "qwen35_config_intermediate_size": (hf_config.get("text_config") or hf_config)["intermediate_size"],
        "names": {"reader": len(reader.names()), "parent_text": len(parent),
                  "missing_vs_parent": missing, "unknown_vs_parent": unknown},
        "shape_problems": shape_problems,
        "tensors_checked": len(names),
        "elements_checked": elements,
        "zero_copy_views": zero_copy,
        "delta_composed_tensors": delta_composed,
        "reference_note": "with a plan, every touched tensor's reference is "
        "reference_compose() over safe_open reads, independent of DeltaOverlay",
        "norm_dtypes": norm_dtypes,
        "norms_served_fp32": sorted(n for n, d in reader.norm_dtypes.items() if d != "bfloat16"),
        "failures": failures,
        "pass": passed,
        "device": "cpu",
        "threads": int(args.threads),
        "seconds": round(time.time() - started, 2),
        "peak_rss_bytes": peak_rss,
        "peak_rss_note": "VmHWM: anonymous memory plus every mapped page touched; the "
        "reader drops its mapped pages after each tensor in this run",
        "page_drops": reader.page_drops,
        "per_tensor": per_tensor,
    })
    reader.close()
    if args.receipt:
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".partial")
        tmp.write_bytes(_canonical_bytes(receipt))
        tmp.replace(out)
        print(f"receipt {out} {receipt['receipt_sha']}")
    print(
        f"P1 PARITY {'PASS' if passed else 'FAIL'}: {len(names)}/{len(reader.names())} tensors, "
        f"{elements:,} elements, {len(failures)} failure(s), zero-copy views {zero_copy}, "
        f"delta-composed {delta_composed}, root check "
        f"{(summary['root_check'] or {}).get('mode', 'no plan')}, missing {len(missing)} unknown {len(unknown)} shape problems {len(shape_problems)}, "
        f"miscovered {len(coverage['miscovered'])}, norms {norm_dtypes}, "
        f"{receipt['seconds']} s"
    )
    for line in failures[:10]:
        print(f"  FAIL {line}")
    return 0 if passed else 1


def run_seam_smoke(args: argparse.Namespace) -> int:
    """tps_probe's own streamed builder on a real child, through the seam.

    Every layer is materialised by the engine's `StreamedLayerList`
    (`load_state_dict(strict=True)` into a stock HF `Qwen3_5DecoderLayer`) and
    each loaded parameter is compared with what the reader served, cast to the
    engine dtype.  `--forward-tokens N` then runs one `use_cache=False` forward
    (as tps_probe does) and checks the last position's logits are finite.  This
    is a plumbing smoke, NOT forward parity: R-Q parity is step 1's.
    """

    torch = _torch()
    torch.set_num_threads(int(args.threads))
    sys.path.insert(0, str(Path(args.tps_probe_dir).resolve()))
    import tps_probe  # noqa: WPS433 (the engine under test, from the path given)

    started = time.time()
    dtype = getattr(torch, args.dtype)
    reader = open_reader(args.chimera, plan_path=args.plan, verify_root_sha=args.verify_root_sha)
    config_path = write_qwen35_config(reader, args.parent_dir, args.config_dir)
    with tps_probe_seam(reader, tps_probe):
        model, head, info = tps_probe.build_streamed_text_model(
            Path(args.config_dir), "cpu", dtype, "sdpa")
    failures: list[str] = []
    checked = 0
    rounded = 0
    resident = {
        "embed_tokens.weight": (model.embed_tokens.weight, f"{QWEN_TEXT_PREFIX}embed_tokens.weight"),
        "norm.weight": (model.norm.weight, f"{QWEN_TEXT_PREFIX}norm.weight"),
        "lm_head.weight": (head.weight, QWEN_LM_HEAD),
    }
    for label, (param, qname) in resident.items():
        want = reader.get(qname)
        checked += 1
        if not torch.equal(param.detach(), want.to(param.dtype)):
            failures.append(label)
        rounded += int(want.dtype != param.dtype)
    for index in range(reader.num_layers):
        layer = model.layers[index]
        params = dict(layer.named_parameters())
        prefix = f"{QWEN_LAYER_PREFIX}{index}."
        names = reader.names_under(prefix)
        if sorted(n[len(prefix):] for n in names) != sorted(params):
            failures.append(f"layer {index}: name sets differ")
        for name in names:
            want = reader.get(name)
            got = params[name[len(prefix):]].detach()
            checked += 1
            rounded += int(want.dtype != got.dtype)
            if not torch.equal(got, want.to(got.dtype)):
                failures.append(name)
        reader.drop_pages()
    forward: dict[str, Any] | None = None
    if int(args.forward_tokens) > 0:
        ids = torch.randint(0, reader.base_vocab, (1, int(args.forward_tokens)),
                            generator=torch.Generator().manual_seed(0))
        begin = time.time()
        with torch.no_grad():
            hidden = model(input_ids=ids, use_cache=False).last_hidden_state
            logits = head(hidden[:, -1:, :]).float()
        forward = {
            "tokens": int(args.forward_tokens),
            "finite": bool(torch.isfinite(logits).all()),
            "hidden_finite": bool(torch.isfinite(hidden.float()).all()),
            "last_logit_absmax": float(logits.abs().max()),
            "seconds": round(time.time() - begin, 2),
            "layer_loads": int(model.layers.layer_loads),
        }
        if not (forward["finite"] and forward["hidden_finite"]):
            failures.append("forward produced a non-finite value")
    peak_rss = None
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (ImportError, AttributeError):
        pass
    passed = not failures and checked == len(reader.names())
    receipt = seal({
        "schema": SMOKE_SCHEMA,
        "tool_version": TOOL_VERSION,
        "tool_sha256": file_sha256(Path(__file__).resolve()),
        "tps_probe_sha256": file_sha256(Path(tps_probe.__file__).resolve()),
        "tps_probe_path": str(Path(tps_probe.__file__).resolve()),
        "argv": sys.argv,
        "chimera_dir": str(reader.directory),
        "qwen35_config_path": str(config_path),
        "qwen35_config_sha256": file_sha256(config_path),
        "plan": reader.summary()["delta_plan"],
        "root_check": reader.summary()["root_check"],
        "engine_dtype": args.dtype,
        "tensors_loaded_and_compared": checked,
        "tensors_mapped": len(reader.names()),
        "served_wider_than_engine_dtype": rounded,
        "layer_loads": int(model.layers.layer_loads),
        "layer_buffer_types": info["facts"]["layer_buffer_types"],
        "forward": forward,
        "forward_note": "finiteness smoke, use_cache=False as tps_probe calls it; not parity",
        "failures": failures,
        "pass": passed,
        "seconds": round(time.time() - started, 2),
        "peak_rss_bytes": peak_rss,
        "device": "cpu",
    })
    reader.close()
    if args.receipt:
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".partial")
        tmp.write_bytes(_canonical_bytes(receipt))
        tmp.replace(out)
        print(f"receipt {out} {receipt['receipt_sha']}")
    print(
        f"P1 SEAM SMOKE {'PASS' if passed else 'FAIL'}: {checked}/{len(reader.names())} tensors "
        f"strict-loaded by tps_probe and equal to the reader ({args.dtype}), "
        f"{rounded} served wider than the engine dtype, layer loads "
        f"{receipt['layer_loads']}, forward {forward}, {receipt['seconds']} s"
    )
    for line in failures[:10]:
        print(f"  FAIL {line}")
    return 0 if passed else 1


def run_map(args: argparse.Namespace) -> int:
    reader = open_reader(args.chimera, plan_path=args.plan, verify_root_sha=args.verify_root_sha)
    summary = reader.summary()
    if args.parent_config:
        parent = json.loads((Path(args.parent_config) / "config.json").read_text(encoding="utf-8"))
        summary["qwen35_config"] = reader.qwen35_config(parent)
    reader.close()
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(text if not args.quiet else json.dumps(summary["coverage"]))
    return 0


def _root_check_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verify-root-sha", dest="verify_root_sha", action=argparse.BooleanOptionalAction,
        default=True,
        help="with --plan: re-digest the opened root and require the plan's sealed "
        "root.directory_sha256 (default; minutes on a 55 GB root).  "
        "--no-verify-root-sha skips the re-hash and instead requires the opened "
        "directory to BE the plan's root; the choice is recorded as root_check.mode",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=TOOL_VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("map", help="headers only: the map, its kinds and its coverage")
    p.add_argument("--chimera", required=True)
    p.add_argument("--plan", default=None)
    _root_check_flag(p)
    p.add_argument("--parent-config", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=run_map)
    p = sub.add_parser("parity", help="every mapped tensor vs a direct read (CPU)")
    p.add_argument("--chimera", required=True)
    p.add_argument("--parent-dir", required=True)
    p.add_argument("--plan", default=None)
    _root_check_flag(p)
    p.add_argument("--receipt", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=run_parity)
    p = sub.add_parser("seam-smoke", help="tps_probe's stock builder on the reader (CPU)")
    p.add_argument("--chimera", required=True)
    p.add_argument("--parent-dir", required=True)
    p.add_argument("--config-dir", required=True,
                   help="scratch directory that will hold ONLY the child's qwen3_5 config.json")
    p.add_argument("--tps-probe-dir", required=True)
    p.add_argument("--plan", default=None)
    _root_check_flag(p)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--forward-tokens", type=int, default=0)
    p.add_argument("--receipt", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=run_seam_smoke)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except P1Error as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
