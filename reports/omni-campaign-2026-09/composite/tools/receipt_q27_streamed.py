#!/usr/bin/env python3
"""receipt_q27_streamed.py -- the R-Q exactness receipt for a 27B-class Chimera import, STREAMED.

`scripts/import_receipt_qwen35.py` (R-Q) holds the imported `ChimeraOmniModel`
AND the fp32 HF reference resident at once; at Qwen3.8-27B geometry that is the
230.76 GiB whole-reference run the September 4 review measured, which cannot be
co-scheduled with anything.  This tool measures the SAME numbers -- max |logit
diff| over the source vocabulary, argmax agreement per position, the per-layer
hidden-state max difference for the first N decoder layers, on the same fixed
prompt pack -- with neither model resident:

* the Chimera side streams `blocks.{i}.*` ONE BLOCK AT A TIME out of the
  checkpoint's safetensors shards (`safetensors_io.read_safetensors` on the
  block's names: header-driven, offset-sorted, plain file reads, no mmap), into
  one preallocated `ChimeraBlock` buffer per (kind, ffn width); the trunk is a
  real `ChimeraOmniModel` built on the meta device whose `blocks` is replaced by
  the streaming list, so `ChimeraOmniModel.forward(ledger_band_logit_mask=True)`
  runs unmodified;
* the HF side is `distill/tps_probe.build_streamed_text_model`'s engine: the
  stock `Qwen3_5TextModel` prologue / decoder layer / epilogue with `layers`
  replaced by `tps_probe.make_streamed_layer_list`'s generator (two shared
  per-type buffers).  The builder here is that function's body generalised over
  the two loader classes (`model.language_model.` vs `model.` tensor prefix)
  and single-file checkpoints, so the same code path serves the tiny CPU
  fixture and the real sharded checkpoints; `test_receipt_q27_streamed.py`
  pins it bit-for-bit to `tps_probe.build_streamed_text_model` on the 0.8B.

The two engines never coexist: the HF pass runs over every prompt first, its
logits and first-layer hiddens are parked on the host in fp32, the engine is
freed, then the Chimera pass runs.  The device peak is therefore ONE engine
(embedding + head + one layer buffer per kind + activations), not two.

Columns.  `--device cpu --dtype float32` forces the transformers torch
fallbacks with the exact per-position recurrence (`import_receipt_qwen35.
disable_fast_paths(recurrent_reference=True)`, applied BEFORE the layer
buffers are constructed because transformers 5.12.1 binds
`chunk_gated_delta_rule or torch_chunk_gated_delta_rule` at
`Qwen3_5GatedDeltaNet.__init__`): that is the R-Q exactness column (0.8B:
210/210, 3.19e-05).  A cuda column restores the fast paths so the reference
runs the fla chunked kernel against Chimera's per-position recurrence: the R-Q
bf16 tolerance column (0.8B on GPU6: 209/210, max 0.5, hidden 0.015625).

Refusals (exit 3, nothing written): a checkpoint whose manifest hidden size,
layer count, base vocabulary or block schedule is not the HF config's; a
manifest with `residual_mode != "none"`; a prompt pack under `--min-positions`
argmax positions; a cuda device already carrying more than
`--max-preexisting-used-mib` (2000) before the run.  All of these fire before
any weight is read.  Budget assertions (exit 4, receipt WRITTEN with
`assertions.passed = false`): peak host RSS above `--max-host-rss-gib` (70 GB
= 65.19 GiB by default, stated in the receipt) and torch reserved memory on
the device above `--max-gpu-reserved-gib` (18 GB = 16.76 GiB).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY receipt_q27_streamed.py \\
        --chimera-dir $OC/composite/s0/q38-27bu-chimera-bf16-sep04 \\
        --hf-dir /home/cereal/models/Qwen3.8-27B-Uncensored \\
        --device cuda:0 --dtype bfloat16 --allow-unverified-output-gate \\
        --out $OC/composite/s0/receipt_s0_streamed_exactness.json

Nothing here is a gate; the numbers are what they are and the receipt names
where each residual comes from.  `[proposed]` line for the 27B bf16 run (run
book S0): argmax >= 99%, reported not gated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "receipt_q27_streamed"
TOOL_VERSION = "1.0.0"
RECEIPT_SCHEMA = "chimera_omni.receipt.import.qwen35.v1"
RECEIPT_MODE = "streamed-teacher-forced"
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"
CHECKPOINT_SCHEMA = "chimera_omni.checkpoint.v1"
INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"
EXIT_REFUSED = 3
EXIT_BUDGET = 4
GIB = 1024.0 ** 3
GB = 1e9
#: Run-book lines: host < 70 GB, GPU4 reserved <= 18 GB.  Decimal GB as written
#: there, converted once here and stated in GiB in the receipt.
DEFAULT_MAX_HOST_RSS_GIB = 70.0 * GB / GIB
DEFAULT_MAX_GPU_RESERVED_GIB = 18.0 * GB / GIB
DEFAULT_MAX_PREEXISTING_USED_MIB = 2000
DEFAULT_MIN_POSITIONS = 200
DTYPES = ("bfloat16", "float32", "float16")
#: Qwen `layer_types` value -> Chimera block kind (the import's own mapping).
LAYER_TYPE_TO_KIND = {"linear_attention": "kda", "full_attention": "gated_gqa"}

# ----------------------------------------------------------- lane imports
#: `composite/tools` sits two levels under the campaign root on the server
#: (`$OC`) and in the workstation mirror; `distill/tps_probe.py` is the engine.
HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parents[1]
DISTILL_DIR = Path(os.environ.get("QF_DISTILL_DIR", CAMPAIGN_ROOT / "distill"))


class Refusal(RuntimeError):
    """A precondition the tool will not run past; exit code 3."""


class BudgetExceeded(RuntimeError):
    """A resource assertion failed after the numbers were measured; exit code 4."""


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# tree binding + module loading
# --------------------------------------------------------------------------


def bind_chimera_tree(tree: str | Path) -> dict[str, Any]:
    """Put `<tree>/src` first on sys.path and import `chimera_omni` from it."""

    root = Path(tree).resolve()
    src = root / "src"
    if not (src / "chimera_omni" / "__init__.py").is_file():
        raise Refusal(f"{root} has no src/chimera_omni package")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import chimera_omni  # noqa: WPS433 -- deliberately late: the path was just bound

    loaded = Path(chimera_omni.__file__).resolve().parents[1]
    if loaded != src.resolve():
        raise Refusal(
            f"chimera_omni was imported from {loaded}, not from the bound tree {src}; "
            "clear PYTHONPATH / installed copies before running"
        )
    commit_file = root / "COMMIT"
    commit = None
    if commit_file.is_file():
        for line in commit_file.read_text(encoding="utf-8").splitlines():
            token = line.strip().split()[0] if line.strip() else ""
            if token and not token.startswith("#"):
                commit = token.lower()
                break
    return {"tree": str(root), "src": str(src), "commit_declared": commit}


def load_rq_script(tree: str | Path) -> Any:
    """`scripts/import_receipt_qwen35.py` from the bound tree, as a module.

    Reused from it: `PROMPT_PACK` (the 8 bank-v1 prompts), `tokenize`,
    `disable_fast_paths` / `restore_fast_paths`, `output_gate_waiver`.  The
    script binds its own `REPO_ROOT/src` on import, which is the tree already
    bound here.
    """

    path = Path(tree).resolve() / "scripts" / "import_receipt_qwen35.py"
    if not path.is_file():
        raise Refusal(f"{path} is missing; the R-Q script is the prompt pack and the fast-path switch")
    spec = importlib.util.spec_from_file_location("import_receipt_qwen35", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_tps_probe() -> Any:
    if str(DISTILL_DIR) not in sys.path:
        sys.path.insert(0, str(DISTILL_DIR))
    import tps_probe  # noqa: WPS433

    return tps_probe


# --------------------------------------------------------------------------
# small helpers (same conventions as s0_import_27b.py)
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=1 << 22) as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_receipt(path: Path, document: dict[str, Any]) -> str:
    text = canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def gib(num_bytes: float | None) -> float | None:
    return None if num_bytes is None else round(float(num_bytes) / GIB, 4)


def ids_sha256(packs: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(packs, separators=(",", ":")).encode("ascii")).hexdigest()


# --------------------------------------------------------------------------
# preflight: configs, geometry, budgets -- before any weight is read
# --------------------------------------------------------------------------


def read_chimera_manifest(chimera_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(checkpoint config document, index)` -- the two small files only."""

    config_path = chimera_dir / CONFIG_FILENAME
    index_path = chimera_dir / INDEX_FILENAME
    if not config_path.is_file():
        raise Refusal(f"{chimera_dir} has no {CONFIG_FILENAME}")
    if not index_path.is_file():
        raise Refusal(f"{chimera_dir} has no {INDEX_FILENAME}; the streamed reader needs the weight_map")
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != CHECKPOINT_SCHEMA:
        raise Refusal(f"{config_path} is not a {CHECKPOINT_SCHEMA} checkpoint config")
    manifest = document.get("config")
    if not isinstance(manifest, dict):
        raise Refusal(f"{config_path} carries no config manifest")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise Refusal(f"{index_path} carries no weight_map")
    return document, index


def hf_text_config(hf_dir: Path) -> dict[str, Any]:
    path = hf_dir / CONFIG_FILENAME
    if not path.is_file():
        raise Refusal(f"{hf_dir} has no {CONFIG_FILENAME}")
    body = json.loads(path.read_text(encoding="utf-8"))
    nested = body.get("text_config")
    return dict(nested) if isinstance(nested, dict) else dict(body)


def hf_layer_types(text: dict[str, Any]) -> list[str]:
    """`layer_types` as transformers resolves them (explicit list, else the interval)."""

    count = int(text["num_hidden_layers"])
    declared = text.get("layer_types")
    if isinstance(declared, list) and declared:
        if len(declared) != count:
            raise Refusal(f"HF layer_types carries {len(declared)} entries for {count} layers")
        return [str(item) for item in declared]
    interval = int(text.get("full_attention_interval", 4))
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]


def check_geometry(manifest: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
    """Refuse a Chimera manifest that is not the HF config's trunk. Returns the facts."""

    mismatches: list[str] = []
    pairs = (
        ("hidden_size", "hidden_size"),
        ("num_layers", "num_hidden_layers"),
        ("ledger_base_vocab_size", "vocab_size"),
        ("intermediate_size", "intermediate_size"),
    )
    for ours, theirs in pairs:
        left = manifest.get(ours)
        right = text.get(theirs)
        if left != right:
            mismatches.append(f"manifest {ours}={left!r} vs HF {theirs}={right!r}")
    kinds_theirs = [LAYER_TYPE_TO_KIND.get(t, t) for t in hf_layer_types(text)]
    pattern = [str(k) for k in manifest.get("block_pattern", [])]
    count = int(manifest.get("num_layers", 0))
    kinds_ours = [pattern[i % len(pattern)] for i in range(count)] if pattern else []
    if manifest.get("force_final_gated_mla") and kinds_ours:
        kinds_ours[-1] = "gated_mla"
    if kinds_ours != kinds_theirs:
        first = next((i for i, (a, b) in enumerate(zip(kinds_ours, kinds_theirs)) if a != b), None)
        mismatches.append(f"block schedule differs from HF layer_types (first at layer {first})")
    if str(manifest.get("residual_mode", "none")) != "none":
        mismatches.append(
            f"residual_mode={manifest.get('residual_mode')!r}; the streamed trunk runs the plain "
            "pre-norm residual loop only (a Qwen import is residual_mode='none')"
        )
    if manifest.get("intermediate_sizes"):
        sizes = [int(v) for v in manifest["intermediate_sizes"]]
        if len(sizes) != count:
            mismatches.append("intermediate_sizes does not carry one entry per layer")
    if mismatches:
        raise Refusal("checkpoint does not match the HF config: " + "; ".join(mismatches))
    return {
        "hidden_size": int(manifest["hidden_size"]),
        "num_layers": count,
        "base_vocab": int(manifest["ledger_base_vocab_size"]),
        "vocab_size": int(manifest["vocab_size"]),
        "block_kinds": kinds_ours,
        "tie_word_embeddings": bool(manifest.get("tie_word_embeddings", False)),
    }


def cuda_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def preflight_device(device: str, max_preexisting_used_mib: int) -> dict[str, Any]:
    """Refuse to start on a card that is already carrying someone else's job."""

    import torch

    if not str(device).startswith("cuda"):
        return {"device": device, "cuda": False}
    if not torch.cuda.is_available():
        raise Refusal(f"--device {device} but torch sees no CUDA device")
    index = cuda_index(device)
    if index >= torch.cuda.device_count():
        raise Refusal(f"--device {device} but only {torch.cuda.device_count()} CUDA device(s) are visible")
    free, total = torch.cuda.mem_get_info(index)
    used_mib = (int(total) - int(free)) / (1024.0 ** 2)
    props = torch.cuda.get_device_properties(index)
    facts = {
        "device": device,
        "cuda": True,
        "name": props.name,
        "uuid": str(getattr(props, "uuid", "")),
        "compute_architecture": f"sm{props.major}{props.minor}",
        "total_gib": gib(total),
        "used_before_start_mib": round(used_mib, 1),
        "max_preexisting_used_mib": int(max_preexisting_used_mib),
    }
    if used_mib > float(max_preexisting_used_mib):
        raise Refusal(
            f"{device} ({props.name}) already has {used_mib:.0f} MiB in use, above "
            f"--max-preexisting-used-mib {max_preexisting_used_mib}; never start on a busy card"
        )
    return facts


# --------------------------------------------------------------------------
# the HF streamed engine (tps_probe.build_streamed_text_model, generalised)
# --------------------------------------------------------------------------


def hf_tensor_prefix(config: Any) -> str:
    """`model.language_model` for `Qwen3_5ForConditionalGeneration`, `model` for the text class."""

    return "model.language_model" if getattr(config, "model_type", "") == "qwen3_5" else "model"


def make_single_or_sharded_reader(tps_probe: Any) -> Any:
    class SingleOrShardedReader(tps_probe.ShardReader):
        """`tps_probe.ShardReader` that also accepts an un-sharded checkpoint.

        The same extension `qf_stream27.SingleOrShardedReader` makes: a fixture
        saved by `save_pretrained` is one `model.safetensors` with no index.
        """

        def __init__(self, model_dir: Path, device: str) -> None:
            from safetensors import safe_open

            model_dir = Path(model_dir)
            if (model_dir / INDEX_FILENAME).is_file():
                super().__init__(model_dir, device)
                return
            single = model_dir / "model.safetensors"
            if not single.is_file():
                raise Refusal(f"{model_dir}: neither {INDEX_FILENAME} nor model.safetensors")
            self._safe_open = safe_open
            self.model_dir = model_dir
            self.device = device
            with safe_open(str(single), framework="pt", device="cpu") as handle:
                self.weight_map = {name: single.name for name in handle.keys()}
            self.total_size = single.stat().st_size
            self._handles = {}
            self.bytes_read = 0

        def close(self) -> None:
            """Drop the open handles so the mmap'd shards leave this process's RSS."""

            self._handles = {}

    return SingleOrShardedReader


def make_tracked_layer_list(tps_probe: Any, decoder_class: Any) -> Any:
    """`tps_probe.make_streamed_layer_list`, plus `current_index` for the hidden tap.

    Identity-keyed hooks collide on the two shared buffers (C04), so the tap
    below reads the index of the layer materialised last -- the
    `qf_stream27.IndexAwareTap` pattern.
    """

    base_class = tps_probe.make_streamed_layer_list(decoder_class)

    class TrackedStreamedLayerList(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.current_index: int | None = None

        def _materialise(self, index: int) -> Any:
            layer = super()._materialise(index)
            self.current_index = int(index)
            return layer

    return TrackedStreamedLayerList


class HiddenTap:
    """Forward hooks on shared layer buffers, keyed by the list's `current_index`."""

    def __init__(self, layer_list: Any, buffers: list[Any], layers_reported: int) -> None:
        self.layer_list = layer_list
        self.layers_reported = int(layers_reported)
        self.states: dict[int, Any] = {}
        self.handles = [module.register_forward_hook(self._hook) for module in buffers]

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        index = self.layer_list.current_index
        if index is None or index >= self.layers_reported:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        self.states[int(index)] = tensor.detach().float().to("cpu")

    def take(self) -> dict[int, Any]:
        out = self.states
        self.states = {}
        return out

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class StreamedHFEngine:
    """`tps_probe.build_streamed_text_model`'s trunk + head, one layer resident at a time.

    Construction is line-for-line the probe's (meta-device `Qwen3_5TextModel`,
    resident embedding / final norm / rotary, `lm_head` tied when the checkpoint
    ships none, `layers` replaced by the streaming list) with two generalisations:
    the tensor prefix follows the loader class and the reader accepts a single
    `model.safetensors`.  `attention_implementation` is `eager`, the R-Q's.
    """

    def __init__(self, tps_probe: Any, model_dir: Path, device: str, dtype: Any, *, layers_reported: int) -> None:
        import torch
        from torch import nn
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextModel,
            Qwen3_5TextRotaryEmbedding,
        )

        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        config = AutoConfig.from_pretrained(str(self.model_dir))
        text_config = getattr(config, "text_config", None) or config
        text_config.dtype = dtype
        text_config._attn_implementation = "eager"
        self.text_config = text_config
        self.prefix = hf_tensor_prefix(config)

        with torch.device("meta"):
            model = Qwen3_5TextModel(text_config)
        reader_class = make_single_or_sharded_reader(tps_probe)
        self.reader = reader_class(self.model_dir, device=device)
        embed = nn.Embedding(text_config.vocab_size, text_config.hidden_size, dtype=dtype, device=device)
        with torch.no_grad():
            embed.weight.copy_(self.reader.get(f"{self.prefix}.embed_tokens.weight"))
        norm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        norm = norm.to(device=device, dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(self.reader.get(f"{self.prefix}.norm.weight"))
        rotary = Qwen3_5TextRotaryEmbedding(config=text_config).to(device)
        model.embed_tokens = embed
        model.norm = norm
        model.rotary_emb = rotary
        del model.layers
        streamed_class = make_tracked_layer_list(tps_probe, Qwen3_5DecoderLayer)
        model.layers = streamed_class(text_config, self.reader, f"{self.prefix}.layers", device, dtype)
        model.eval()
        self.trunk = model

        head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False, dtype=dtype, device=device)
        self.head_tied = not self.reader.has("lm_head.weight")
        with torch.no_grad():
            if self.head_tied:
                head.weight = embed.weight
            else:
                head.weight.copy_(self.reader.get("lm_head.weight"))
        self.head = head.eval()
        self.tap = HiddenTap(model.layers, list(model.layers.buffers_by_type.values()), layers_reported)
        resident = sum(p.numel() * p.element_size() for p in list(model.parameters()) + list(head.parameters()))
        self.facts = {
            "engine": "tps_probe.build_streamed_text_model (generalised prefix + single-file reader)",
            "streaming": True,
            "layers_streamed": int(text_config.num_hidden_layers),
            "layer_buffer_types": sorted(model.layers.buffers_by_type.keys()),
            "tensor_prefix": self.prefix,
            "lm_head_tied_to_embedding": self.head_tied,
            "resident_parameter_bytes_gib": gib(resident),
            "checkpoint_total_size_gib": gib(self.reader.total_size),
            "attention_implementation": "eager",
        }

    def forward(self, ids: Any) -> tuple[Any, dict[int, Any]]:
        """`(logits fp32 on the host, {layer index: hidden fp32 on the host})` for one prompt."""

        import torch

        with torch.no_grad():
            hidden = self.trunk(input_ids=ids.to(self.device), use_cache=False).last_hidden_state
            logits = self.head(hidden).float().to("cpu")
        return logits, self.tap.take()

    def close(self) -> dict[str, Any]:
        facts = {
            "layer_loads": int(self.trunk.layers.layer_loads),
            "layer_load_seconds": round(float(self.trunk.layers.layer_load_seconds), 3),
            "bytes_read": int(self.reader.bytes_read),
        }
        self.tap.remove()
        self.reader.close()
        del self.trunk, self.head, self.reader
        return facts


# --------------------------------------------------------------------------
# the Chimera streamed engine
# --------------------------------------------------------------------------


def make_streamed_chimera_blocks() -> Any:
    """The streaming stand-in for `ChimeraOmniModel.blocks` (an `nn.Module`, built lazily so torch loads late)."""

    from torch import nn

    class StreamedChimeraBlocks(nn.Module):
        """A stand-in for `ChimeraOmniModel.blocks` that owns one buffer per (kind, ffn width).

        `_forward_trunk` iterates `self.blocks`; iterating this object materialises
        block i's tensors into the buffer of its kind (`safetensors_io.
        read_safetensors` on exactly the names `blocks.{i}.*`, per-tensor sha256
        verified against the index when `verify`) and yields it.  `load_state_dict`
        is `strict=True`, so a block whose checkpoint tensors do not cover the
        module refuses rather than running half-initialised.
        """

        def __init__(
            self,
            cfg: Any,
            chimera_dir: Path,
            index: dict[str, Any],
            device: str,
            dtype: Any,
            *,
            verify: bool,
            layers_reported: int,
        ) -> None:
            super().__init__()
            self._build(cfg, chimera_dir, index, device, dtype, verify=verify, layers_reported=layers_reported)

        def _build(
            self,
            cfg: Any,
            chimera_dir: Path,
            index: dict[str, Any],
            device: str,
            dtype: Any,
            *,
            verify: bool,
            layers_reported: int,
        ) -> None:
            import torch
            from torch import nn

            from chimera_omni.modeling.layers import ChimeraBlock

            self.cfg = cfg
            self.root = Path(chimera_dir)
            self.device = device
            self.dtype = dtype
            self.verify = bool(verify)
            self.count = int(cfg.num_layers)
            self.layers_reported = int(layers_reported)
            weight_map: dict[str, str] = index["weight_map"]
            self.names_by_index: dict[int, list[str]] = {}
            for name in weight_map:
                if name.startswith("blocks."):
                    head, _, _rest = name[len("blocks."):].partition(".")
                    self.names_by_index.setdefault(int(head), []).append(name)
            missing = [i for i in range(self.count) if i not in self.names_by_index]
            if missing:
                raise Refusal(f"checkpoint index names no tensors for blocks {missing[:5]}")
            self.block_buffers = nn.ModuleDict()
            self.buffer_key: dict[int, str] = {}
            for i in range(self.count):
                key = f"{cfg.block_kind(i)}:{cfg.resolved_intermediate_size(i)}"
                self.buffer_key[i] = key
                if key not in self.block_buffers:
                    #: Built on the target device in the default dtype and cast, as
                    #: tps_probe does: one transient fp32 block (~1.7 GB at 27B) beats a
                    #: global default-dtype switch a raised exception could leak.
                    with torch.device(device):
                        block = ChimeraBlock(cfg, i)
                    self.block_buffers[key] = block.to(device=device, dtype=dtype).eval()
            self.current_index: int | None = None
            self.layer_loads = 0
            self.layer_load_seconds = 0.0
            self.bytes_read = 0
            self.tensors_verified = 0
            self.states: dict[int, Any] = {}
            self._handles = [module.register_forward_hook(self._hook) for module in self.block_buffers.values()]

        def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
            index = self.current_index
            if index is None or index >= self.layers_reported:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            self.states[int(index)] = tensor.detach().float().to("cpu")

        def take(self) -> dict[int, Any]:
            out = self.states
            self.states = {}
            return out

        def _materialise(self, index: int) -> Any:
            import torch

            from chimera_omni.importing.safetensors_io import read_safetensors

            block = self.block_buffers[self.buffer_key[index]]
            started = time.perf_counter()
            names = self.names_by_index[index]
            views = read_safetensors(self.root, names, verify=self.verify)
            target = block.state_dict()
            prefix = f"blocks.{index}."
            state: dict[str, Any] = {}
            for name, view in views.items():
                local = name[len(prefix):]
                if local not in target:
                    raise Refusal(f"checkpoint tensor {name!r} has no place in the block buffer")
                reference = target[local]
                state[local] = view.torch(
                    device=self.device,
                    dtype=self.dtype if reference.is_floating_point() else reference.dtype,
                )
                self.bytes_read += len(view.buffer)
            if self.verify:
                self.tensors_verified += len(views)
            with torch.no_grad():
                block.load_state_dict(state, strict=True)
            del state, views
            self.layer_load_seconds += time.perf_counter() - started
            self.layer_loads += 1
            self.current_index = int(index)
            return block

        def __len__(self) -> int:
            return self.count

        def __iter__(self):
            for index in range(self.count):
                yield self._materialise(index)

        def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
            if isinstance(key, slice):
                return (self._materialise(i) for i in range(*key.indices(self.count)))
            return self._materialise(int(key))

        def remove(self) -> None:
            for handle in self._handles:
                handle.remove()
            self._handles = []

    return StreamedChimeraBlocks


class StreamedChimeraEngine:
    """`ChimeraOmniModel` on the meta device with embed / norm / head resident and blocks streamed."""

    def __init__(
        self,
        chimera_dir: Path,
        document: dict[str, Any],
        index: dict[str, Any],
        device: str,
        dtype: Any,
        *,
        verify: bool,
        layers_reported: int,
    ) -> None:
        import torch
        from torch import nn

        from chimera_omni.config import ChimeraConfig
        from chimera_omni.importing.safetensors_io import read_safetensors
        from chimera_omni.modeling import ChimeraOmniModel

        self.root = Path(chimera_dir)
        self.device = device
        self.cfg = ChimeraConfig.from_manifest(document["config"])
        tied: dict[str, str] = dict(index.get("metadata", {}).get("tied_weights", {}))
        resident_names = ["embed.weight", "norm.weight"]
        untied_head = not self.cfg.tie_word_embeddings
        if untied_head:
            resident_names.append("lm_head.weight")
        views = read_safetensors(self.root, resident_names, verify=verify)
        if dtype is None:
            # The dtype the shards ARE (header token), never a widened default.
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}.get(views["embed.weight"].dtype)
            if dtype is None:
                raise Refusal(f"embed.weight is {views['embed.weight'].dtype}; name --dtype explicitly")
        self.dtype = dtype

        with torch.device("meta"):
            model = ChimeraOmniModel(self.cfg)
        embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            embed.weight.copy_(views["embed.weight"].torch(device=device, dtype=dtype))
        norm = type(model.norm)(self.cfg.hidden_size, self.cfg.rms_norm_eps).to(device=device, dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(views["norm.weight"].torch(device=device, dtype=dtype))
        head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False, device=device, dtype=dtype)
        if untied_head:
            with torch.no_grad():
                head.weight.copy_(views["lm_head.weight"].torch(device=device, dtype=dtype))
        else:
            if tied.get("lm_head.weight") != "embed.weight":
                raise Refusal("tied config but the index records no lm_head.weight -> embed.weight tie")
            head.weight = embed.weight
        self.resident_bytes = sum(len(v.buffer) for v in views.values())
        del views
        model.embed = embed
        model.norm = norm
        model.lm_head = head
        self.blocks = make_streamed_chimera_blocks()(
            self.cfg, self.root, index, device, dtype, verify=verify, layers_reported=layers_reported
        )
        # As tps_probe does with `layers`: drop the meta ModuleList, install the stream.
        del model.blocks
        model.blocks = self.blocks
        model.eval()
        self.model = model
        self.facts = {
            "engine": "ChimeraOmniModel (meta) + embed/norm/lm_head resident + blocks streamed one at a time",
            "streaming": True,
            "blocks_streamed": int(self.cfg.num_layers),
            "block_buffers": sorted(self.blocks.block_buffers.keys()),
            "lm_head_tied_to_embedding": not untied_head,
            "resident_table_bytes_gib": gib(self.resident_bytes),
            "tensor_sha256_verified_on_read": bool(verify),
            "load_dtype": str(dtype),
            "band_begin": int(self.cfg.ledger_band_begin()),
            "vocab_size": int(self.cfg.vocab_size),
        }

    def forward(self, ids: Any) -> tuple[Any, dict[int, Any]]:
        """`(logits over the FULL vocab with the band rows masked, fp32 host, {layer: hidden})`."""

        import torch

        with torch.no_grad():
            out = self.model(ids.to(self.device), ledger_band_logit_mask=True)
            logits = out["logits"].float().to("cpu")
        return logits, self.blocks.take()

    def close(self) -> dict[str, Any]:
        facts = {
            "layer_loads": int(self.blocks.layer_loads),
            "layer_load_seconds": round(float(self.blocks.layer_load_seconds), 3),
            "bytes_read": int(self.blocks.bytes_read),
            "tensors_sha256_verified": int(self.blocks.tensors_verified),
        }
        self.blocks.remove()
        del self.model, self.blocks
        return facts


# --------------------------------------------------------------------------
# the column
# --------------------------------------------------------------------------


def free_device(device: str) -> None:
    import torch

    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_index(device))
        torch.cuda.empty_cache()


def run_hf_pass(
    tps_probe: Any, hf_dir: Path, device: str, dtype: Any, packs: list[list[int]], *, layers_reported: int
) -> tuple[list[Any], list[dict[int, Any]], dict[str, Any]]:
    """Every prompt through the streamed HF engine; logits + first hiddens parked on the host."""

    import torch

    started = time.perf_counter()
    engine = StreamedHFEngine(tps_probe, hf_dir, device, dtype, layers_reported=layers_reported)
    build_seconds = time.perf_counter() - started
    logits_all: list[Any] = []
    hidden_all: list[dict[int, Any]] = []
    started = time.perf_counter()
    for prompt_index, ids in enumerate(packs):
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits, hiddens = engine.forward(input_ids)
        logits_all.append(logits)
        hidden_all.append(hiddens)
        log(f"[hf {device}/{str(dtype).replace('torch.', '')}] prompt {prompt_index}: {len(ids)} tokens ({time.perf_counter() - started:.1f}s)")
    facts = dict(engine.facts)
    facts.update(engine.close())
    facts["build_seconds"] = round(build_seconds, 3)
    facts["forward_seconds"] = round(time.perf_counter() - started, 3)
    del engine
    free_device(device)
    return logits_all, hidden_all, facts


def run_chimera_pass(
    chimera_dir: Path,
    document: dict[str, Any],
    index: dict[str, Any],
    device: str,
    dtype: Any,
    packs: list[list[int]],
    *,
    verify: bool,
    layers_reported: int,
) -> tuple[list[Any], list[dict[int, Any]], dict[str, Any]]:
    import torch

    started = time.perf_counter()
    engine = StreamedChimeraEngine(
        chimera_dir, document, index, device, dtype, verify=verify, layers_reported=layers_reported
    )
    build_seconds = time.perf_counter() - started
    logits_all: list[Any] = []
    hidden_all: list[dict[int, Any]] = []
    started = time.perf_counter()
    for prompt_index, ids in enumerate(packs):
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits, hiddens = engine.forward(input_ids)
        logits_all.append(logits)
        hidden_all.append(hiddens)
        log(f"[chimera {device}/{str(engine.dtype).replace('torch.', '')}] prompt {prompt_index}: {len(ids)} tokens ({time.perf_counter() - started:.1f}s)")
    facts = dict(engine.facts)
    facts.update(engine.close())
    facts["build_seconds"] = round(build_seconds, 3)
    facts["forward_seconds"] = round(time.perf_counter() - started, 3)
    del engine
    free_device(device)
    return logits_all, hidden_all, facts


def compare_prompts(
    packs: list[list[int]],
    reference_logits: list[Any],
    reference_hidden: list[dict[int, Any]],
    produced_logits: list[Any],
    produced_hidden: list[dict[int, Any]],
    *,
    base_vocab: int,
    block_kinds: list[str],
    layers_reported: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The R-Q column loop over parked tensors: same fields, same arithmetic."""

    import torch

    prompts: list[dict[str, Any]] = []
    worst_logit = 0.0
    worst_hidden = 0.0
    total_positions = 0
    matching_positions = 0
    band_finite_total = 0
    for prompt_index, ids in enumerate(packs):
        reference = reference_logits[prompt_index]
        produced_full = produced_logits[prompt_index]
        produced = produced_full[..., :base_vocab]
        band = produced_full[..., base_vocab:]
        band_finite = int(torch.isfinite(band).sum().item()) if band.numel() else 0
        band_finite_total += band_finite
        difference = (produced - reference[..., :base_vocab]).abs()
        max_logit = float(difference.max().item())
        reference_argmax = reference[..., :base_vocab].argmax(dim=-1)
        produced_argmax = produced.argmax(dim=-1)
        agree = int((reference_argmax == produced_argmax).sum().item())
        positions = int(reference_argmax.numel())
        per_layer: list[dict[str, Any]] = []
        for layer in range(layers_reported):
            left = reference_hidden[prompt_index].get(layer)
            right = produced_hidden[prompt_index].get(layer)
            if left is None or right is None:
                continue
            gap = float((right.float() - left.float()).abs().max().item())
            per_layer.append(
                {
                    "layer": layer,
                    "kind": block_kinds[layer] if layer < len(block_kinds) else None,
                    "max_abs_diff": gap,
                    "reference_rms": float(left.float().pow(2).mean().sqrt().item()),
                }
            )
            worst_hidden = max(worst_hidden, gap)
        prompts.append(
            {
                "index": prompt_index,
                "tokens": len(ids),
                "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest(),
                "max_abs_logit_diff": max_logit,
                "reference_logit_absmax": float(reference[..., :base_vocab].abs().max().item()),
                "argmax_agreement": agree / positions if positions else 1.0,
                "argmax_positions": positions,
                "argmax_disagreements": positions - agree,
                "band_logits_finite": band_finite,
                "per_layer_hidden_max_abs_diff": per_layer,
            }
        )
        worst_logit = max(worst_logit, max_logit)
        total_positions += positions
        matching_positions += agree
        log(f"prompt {prompt_index}: {len(ids)} tokens, max|dlogit|={max_logit:.3e}, argmax {agree}/{positions}")
    summary = {
        "max_abs_logit_diff": worst_logit,
        "max_abs_hidden_diff_first_layers": worst_hidden,
        "argmax_agreement": matching_positions / total_positions if total_positions else 1.0,
        "argmax_positions": total_positions,
        "argmax_matching_positions": matching_positions,
        "band_logits_finite_total": band_finite_total,
    }
    return prompts, summary


def check_budgets(
    *,
    peak_rss_bytes_value: int | None,
    sampled_rss_bytes: int | None,
    reserved_bytes: int | None,
    max_host_rss_gib: float,
    max_gpu_reserved_gib: float,
    cuda: bool,
) -> dict[str, Any]:
    """The two run-book assertions, as facts; `passed` decides the exit code."""

    host_peak = max(int(peak_rss_bytes_value or 0), int(sampled_rss_bytes or 0))
    host = {
        "limit_gib": round(float(max_host_rss_gib), 4),
        "peak_rss_gib": gib(host_peak),
        "peak_rss_bytes": host_peak,
        "source": "max(resource.getrusage ru_maxrss, sampled VmRSS)",
        "passed": host_peak <= float(max_host_rss_gib) * GIB,
    }
    gpu = {
        "limit_gib": round(float(max_gpu_reserved_gib), 4),
        "peak_reserved_gib": gib(reserved_bytes) if cuda else None,
        "peak_reserved_bytes": int(reserved_bytes) if (cuda and reserved_bytes is not None) else None,
        "source": "torch.cuda.max_memory_reserved(device)",
        "passed": (not cuda) or (reserved_bytes is not None and reserved_bytes <= float(max_gpu_reserved_gib) * GIB),
        "applies": bool(cuda),
    }
    return {"host_rss": host, "gpu_reserved": gpu, "passed": bool(host["passed"] and gpu["passed"])}


def checkpoint_identity(chimera_dir: Path, index: dict[str, Any], hf_dir: Path, *, hash_hf_shards: bool) -> dict[str, Any]:
    """Small-file hashes always; the 54 GB HF shards only on request (sizes always)."""

    shards = sorted(set(str(v) for v in index["weight_map"].values()))
    chimera = {
        "directory": str(chimera_dir),
        "index_sha256": file_sha256(chimera_dir / INDEX_FILENAME),
        "config_sha256": file_sha256(chimera_dir / CONFIG_FILENAME),
        "tensors": len(index["weight_map"]),
        "tensor_sha256_recorded": len(index.get("metadata", {}).get("tensor_sha256", {})),
        "tied_weights": dict(index.get("metadata", {}).get("tied_weights", {})),
        "total_size": index.get("metadata", {}).get("total_size"),
        "shards": {name: (chimera_dir / name).stat().st_size for name in shards},
    }
    hf_files: dict[str, Any] = {}
    for name in sorted(os.listdir(hf_dir)):
        path = hf_dir / name
        if not path.is_file():
            continue
        if name.endswith(".safetensors"):
            hf_files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path) if hash_hf_shards else None}
        elif name in {CONFIG_FILENAME, "tokenizer.json", INDEX_FILENAME}:
            hf_files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {
        "chimera": chimera,
        "hf": {
            "directory": str(hf_dir),
            "files": hf_files,
            "shards_hashed": bool(hash_hf_shards),
            "note": "HF shard bytes are hashed only with --hash-hf-shards; sizes and the config/index/tokenizer hashes are always recorded",
        },
    }


def resolve_packs(args: argparse.Namespace, rq: Any, hf_dir: Path) -> tuple[list[list[int]], list[dict[str, Any]], str]:
    """Prompt ids: `--prompt-ids` JSON, else `--prompts` JSON strings, else the R-Q pack, tokenized."""

    if args.prompt_ids:
        packs = json.loads(Path(args.prompt_ids).read_text(encoding="utf-8"))
        if not isinstance(packs, list) or not all(isinstance(p, list) and p for p in packs):
            raise Refusal(f"--prompt-ids {args.prompt_ids} must be a JSON list of non-empty id lists")
        packs = [[int(t) for t in p][: args.max_tokens] for p in packs]
        pack = [{"index": i, "text": None, "tokens": len(ids)} for i, ids in enumerate(packs)]
        return packs, pack, f"--prompt-ids {args.prompt_ids}"
    if args.prompts:
        texts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
        if not isinstance(texts, list) or not all(isinstance(t, str) and t for t in texts):
            raise Refusal(f"--prompts {args.prompts} must be a JSON list of non-empty strings")
        source = f"--prompts {args.prompts}"
    else:
        texts = list(rq.PROMPT_PACK)
        source = "scripts/import_receipt_qwen35.PROMPT_PACK (8 bank-v1 prompts)"
    packs = rq.tokenize(hf_dir, tuple(texts), args.max_tokens)
    pack = [{"index": i, "text": text, "tokens": len(ids)} for i, (text, ids) in enumerate(zip(texts, packs))]
    return packs, pack, source


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    started_wall = time.time()
    tool_sha256 = file_sha256(Path(__file__))
    tree = bind_chimera_tree(args.chimera_src)
    rq = load_rq_script(tree["tree"])
    tps_probe = load_tps_probe()
    tree["rq_script_sha256"] = file_sha256(Path(tree["tree"]) / "scripts" / "import_receipt_qwen35.py")
    tree["tps_probe_sha256"] = file_sha256(Path(tps_probe.__file__))

    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from chimera_omni.importing.profiles import qwen35_chimera_config

    chimera_dir = Path(args.chimera_dir).resolve()
    hf_dir = Path(args.hf_dir).resolve()
    if args.dtype not in DTYPES + ("checkpoint",):
        raise Refusal(f"--dtype must be one of {DTYPES} or 'checkpoint', not {args.dtype!r}")
    device = str(args.device)
    document, index = read_chimera_manifest(chimera_dir)
    text = hf_text_config(hf_dir)
    geometry = check_geometry(document["config"], text)
    log(
        f"geometry agrees: hidden {geometry['hidden_size']}, layers {geometry['num_layers']}, "
        f"base vocab {geometry['base_vocab']}, vocab {geometry['vocab_size']}"
    )
    device_facts = preflight_device(device, args.max_preexisting_used_mib)
    cuda = bool(device_facts.get("cuda"))

    # The output-gate waiver: the same reading the import ran under, recorded the same way.
    hf_config = json.loads((hf_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    try:
        cfg_for_waiver = qwen35_chimera_config(hf_config, allow_unverified_output_gate=args.allow_unverified_output_gate)
    except ValueError as error:
        raise Refusal(f"config profile refused {hf_dir}: {error}") from error
    waiver = rq.output_gate_waiver(cfg_for_waiver)

    packs, pack, pack_source = resolve_packs(args, rq, hf_dir)
    positions = sum(len(ids) for ids in packs)
    if positions < int(args.min_positions):
        raise Refusal(f"prompt pack yields {positions} argmax positions, below --min-positions {args.min_positions}")
    out_path = Path(args.out).resolve()
    if out_path.exists() and not args.overwrite:
        raise Refusal(f"{out_path} exists; pass --overwrite to replace the receipt")

    dtype = None if args.dtype == "checkpoint" else getattr(torch, args.dtype)
    if dtype is None:
        # The Chimera checkpoint's own dtype (its embed anchor), read from the header.
        from chimera_omni.importing.safetensors_io import read_header, shard_paths

        anchor = None
        for shard in shard_paths(chimera_dir):
            header, _ = read_header(shard)
            if "embed.weight" in header:
                anchor = str(header["embed.weight"]["dtype"])
                break
        if anchor is None:
            raise Refusal("no embed.weight in the checkpoint headers; cannot infer --dtype checkpoint")
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}.get(anchor)
        if dtype is None:
            raise Refusal(f"embed.weight is {anchor}; name --dtype explicitly")
    dtype_name = str(dtype).replace("torch.", "")

    force_torch_path = not cuda
    if force_torch_path:
        fast_paths = rq.disable_fast_paths(modeling_qwen3_5, recurrent_reference=not args.chunked_cpu_reference)
        fast_path_available = False
    else:
        rq.restore_fast_paths(modeling_qwen3_5)
        fast_paths = {}
        fast_path_available = bool(getattr(modeling_qwen3_5, "is_fast_path_available", False))
    column: dict[str, Any] = {
        "device": device,
        "dtype": dtype_name,
        "mode": RECEIPT_MODE,
        "engine_order": "hf-streamed pass first, freed, then chimera-streamed pass",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation": "eager",
        "fast_paths_disabled": fast_paths,
        "fast_path_available": fast_path_available,
        "reference_multi_token_path": (
            fast_paths.get("multi_token_path") if force_torch_path else "fla chunk_gated_delta_rule (cuda fast path)"
        ),
    }
    if cuda:
        column["gpu_name"] = device_facts["name"]
        column["gpu_capability"] = device_facts["compute_architecture"]

    sampler = tps_probe.ResourceSampler(interval_seconds=0.5)
    if cuda:
        torch.cuda.reset_peak_memory_stats(cuda_index(device))
    sampler.start()
    try:
        log(f"[{device}/{dtype_name}] HF streamed pass over {len(packs)} prompts from {hf_dir}")
        hf_logits, hf_hidden, hf_facts = run_hf_pass(tps_probe, hf_dir, device, dtype, packs, layers_reported=args.layers)
        log(f"[{device}/{dtype_name}] Chimera streamed pass from {chimera_dir}")
        ch_logits, ch_hidden, ch_facts = run_chimera_pass(
            chimera_dir, document, index, device, dtype, packs, verify=not args.no_verify_tensor_sha, layers_reported=args.layers
        )
    finally:
        sampler.stop()
    prompts, summary = compare_prompts(
        packs, hf_logits, hf_hidden, ch_logits, ch_hidden,
        base_vocab=geometry["base_vocab"], block_kinds=geometry["block_kinds"], layers_reported=args.layers,
    )
    column["prompts"] = prompts
    column["summary"] = summary
    column["engines"] = {"hf": hf_facts, "chimera": ch_facts}

    reserved = torch.cuda.max_memory_reserved(cuda_index(device)) if cuda else None
    allocated = torch.cuda.max_memory_allocated(cuda_index(device)) if cuda else None
    sampled = sampler.report()
    rss = peak_rss_bytes()
    budgets = check_budgets(
        peak_rss_bytes_value=rss,
        sampled_rss_bytes=sampler.peak_host_rss_kib * 1024,
        reserved_bytes=reserved,
        max_host_rss_gib=args.max_host_rss_gib,
        max_gpu_reserved_gib=args.max_gpu_reserved_gib,
        cuda=cuda,
    )
    column["peak_rss_bytes"] = rss
    column["resources"] = {
        "peak_rss_bytes": rss,
        "peak_rss_gib": gib(rss),
        "peak_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss * 1024",
        "sampled": sampled,
        "torch_peak_allocated_gib": gib(allocated) if cuda else None,
        "torch_peak_reserved_gib": gib(reserved) if cuda else None,
        "device_wide_min_free_gib": sampled["device_minimum_free_gb"].get(str(cuda_index(device))) if cuda else None,
        "rss_note": "the HF engine's safetensors handles mmap the shards, so touched shard pages count in RSS until the engine is freed",
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "mode": RECEIPT_MODE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "argv": list(args.argv),
            "sha256": tool_sha256,
            "command": tps_probe.exact_command(),
        },
        "chimera": tree,
        "source_prefix": rq.QWEN_TEXT_PREFIX if hasattr(rq, "QWEN_TEXT_PREFIX") else None,
        "inputs": {
            "chimera_dir": str(chimera_dir),
            "hf_dir": str(hf_dir),
            "identity": checkpoint_identity(chimera_dir, index, hf_dir, hash_hf_shards=args.hash_hf_shards),
            "geometry": geometry,
            "prompt_source": pack_source,
            "prompt_pack": pack,
            "prompt_pack_sha256": ids_sha256(packs),
            "max_tokens": int(args.max_tokens),
            "layers_reported": int(args.layers),
        },
        "device": device_facts,
        "devices_visible": tps_probe.device_records() if cuda else [],
        "nvidia_smi": tps_probe.nvidia_smi_inventory() if cuda else None,
        "columns": [column],
        "round_trip_column": None,
        "output_gate_waiver": waiver,
        "assertions": budgets,
        "timing": {"wall_seconds": round(time.time() - started_wall, 2)},
        "notes": [
            "streamed-teacher-forced: neither the HF reference nor the Chimera model is ever resident; "
            "the HF trunk streams one decoder layer at a time (tps_probe engine) and the Chimera trunk one block at a time",
            "the two passes run one after the other on the same device, so the device peak is one engine, not two",
            "cpu columns force the transformers torch fallbacks with the exact per-position recurrence (R-Q exactness column); "
            "cuda columns run the fla chunked kernel against Chimera's per-position recurrence (R-Q bf16 tolerance column)",
            "norm weights are imported as 1 + w; exact in an fp32 checkpoint, rounded once at import in a bf16 one",
            "band logits are masked to -inf by ledger_band_logit_mask=True and excluded from the comparison; "
            "band_logits_finite counts any that were not",
            "nothing here is a gate; the [proposed] 27B bf16 line is argmax >= 99%, reported not gated",
        ],
    }
    receipt_sha = write_receipt(out_path, receipt)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "receipt_sha256": receipt_sha,
                "mode": RECEIPT_MODE,
                "device": device,
                "dtype": dtype_name,
                "summary": summary,
                "peak_rss_gib": gib(rss),
                "torch_peak_reserved_gib": gib(reserved) if cuda else None,
                "assertions_passed": budgets["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt, bool(budgets["passed"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="R-Q exactness receipt for a Chimera Qwen import, both models streamed (never resident)",
    )
    parser.add_argument("--chimera-dir", required=True, help="Chimera checkpoint directory (write_safetensors layout)")
    parser.add_argument("--hf-dir", required=True, help="HF Qwen3.5/3.8 checkpoint the import came from (27B-U or the C2 child)")
    parser.add_argument("--out", required=True, help="receipt JSON path (a .sha256 sidecar is written beside it)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", help=f"{'|'.join(DTYPES)}|checkpoint (the Chimera shards' own dtype)")
    parser.add_argument("--prompts", default=None, help="JSON list of prompt strings (default: the 8 bank-v1 R-Q prompts)")
    parser.add_argument("--prompt-ids", default=None, help="JSON list of token-id lists (bypasses the tokenizer; fixtures)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4, help="per-layer hidden max diff for the first N layers")
    parser.add_argument("--min-positions", type=int, default=DEFAULT_MIN_POSITIONS)
    parser.add_argument("--max-host-rss-gib", type=float, default=DEFAULT_MAX_HOST_RSS_GIB, help="host RAM assertion (default 70 GB)")
    parser.add_argument("--max-gpu-reserved-gib", type=float, default=DEFAULT_MAX_GPU_RESERVED_GIB, help="torch reserved assertion (default 18 GB)")
    parser.add_argument("--max-preexisting-used-mib", type=int, default=DEFAULT_MAX_PREEXISTING_USED_MIB)
    parser.add_argument("--chunked-cpu-reference", action="store_true", help="cpu reference uses the chunked torch fallback instead of the exact recurrence")
    parser.add_argument("--allow-unverified-output-gate", action="store_true", help="accept a declared non-sigmoid output gate (Qwen3.8-27B: swish); recorded as the R-Q waiver")
    parser.add_argument("--no-verify-tensor-sha", action="store_true", help="skip the per-tensor sha256 check against the Chimera index while streaming")
    parser.add_argument("--hash-hf-shards", action="store_true", help="also sha256 every HF shard into the receipt (54 GB at 27B)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--chimera-src",
        default=os.environ.get("CHIMERA_OMNI_SRC", DEFAULT_CHIMERA_SRC),
        help="chimera-omni tree root (contains src/chimera_omni and scripts/; $CHIMERA_OMNI_SRC)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    try:
        _receipt, passed = run(args)
    except Refusal as error:
        print(f"[{TOOL_NAME}] REFUSED: {error}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    if not passed:
        print(f"[{TOOL_NAME}] BUDGET ASSERTION FAILED: see assertions in {args.out}", file=sys.stderr, flush=True)
        return EXIT_BUDGET
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
