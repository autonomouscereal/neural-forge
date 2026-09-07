#!/usr/bin/env python3
"""``qf_build.py --streamed``: the C2 (MLP channel coreset) fold on a STREAMED trunk.

Why this file exists.  ``qwen-fold/qf_build.py`` folds a model it holds resident
on one GPU, and a 27B in bf16 is 54 GB -- more than any card on this box.
``distill/tps_probe.build_streamed_text_model`` runs the same stock
``Qwen3_5TextModel`` prologue, decoder layer and epilogue with ONE layer's
weights resident at a time (two shared per-type buffers), bit-identical to the
resident forward on the 0.8B.  This module puts the two together without
editing either lane:

* the ENGINE is ``tps_probe``'s streamed layer list (subclassed here so a
  folded layer is materialised with its NARROW MLP from a host cache instead of
  the parent's wide one from disk);
* the TAP is index-aware.  ``qf_harness.capture_grams`` keys its hooks by the
  module the hook is on, and on the streamed engine every linear layer is the
  SAME module object (and every full layer the other one), so an identity-keyed
  hook records one site per buffer type and silently overwrites the rest.  The
  tap here hooks the two shared ``mlp`` modules once and keys every capture by
  the layer index the layer list materialised last;
* the RESIDUAL STREAM is cached on the host between layers, in the model's own
  dtype (bf16 on the GPUs), so the next group's capture starts at the boundary
  of the already-folded prefix instead of at the embedding.  The residual
  between two decoder layers IS a bf16 tensor in the stock forward, so caching
  it changes no arithmetic: group g's sites see exactly the activations the
  resident harness's ``group_size`` schedule shows them, at ~2 layer-forwards
  per layer per build instead of ~22;
* the SOLVE is ``qf_solve``'s Gram form: ``X^T X``, ``X^T Y`` and ``||Y||^2``
  are accumulated per split (fit-inner, fit-check, held) on the CPU in fp64,
  one corpus batch at a time; the pivoted coreset is ``pivoted_columns_gram``;
  the lambda ladder is walked with ``_ladder_rung`` and the internal-split
  choice is scored from the check-split Gram.  The rows are never held in
  fp64.  ``X`` and ``Y`` are stashed on the host in bf16 (which is lossless:
  that is their dtype) so the receipt's per-row scores can be computed after
  the writer is known;
* EXPORT is ``qf_export.export_child`` on a CPU copy of the parent with
  ``qf_mlp.install_narrow_mlp`` applied per site, so the shards, the digest
  method and the receipt shape are the resident build's, unchanged.

Every number the receipt prints is a proposed decision line for Cyril.  The
schedule (``--mlp-group``), the rung (``--mlp-keep``) and the laws
(``qf_solve``) are the resident lane's; only the engine is new.

    qf_stream27.py --streamed --parent DIR --out DIR --evidence manifest.json \\
        --tag C2K4352 --parent-code Q38-27BU --device cuda:0 --dtype bfloat16 \\
        --mlp-keep 4352 --mlp-group 3 --drop-mtp [--src-dir STAGED] \\
        [--solve-form rows|gram] [--resident-check] [--allow-share-read]

Two solve forms.  ``rows`` (default) runs ``qf_mlp.narrow_mlp_site`` itself on
the stashed rows -- the resident's own GEMMs in the resident's own order, which
is what makes the streamed child BYTE-IDENTICAL to a resident build on the
same device and dtype.  ``gram`` solves from the per-batch fp64 accumulation
and never forms the design matrix; it equals the row form to fp64 summation
order and is the form for a corpus whose rows do not fit.  Every parity
receipt records the accumulation-vs-rows distance at the parity sites.

``--src-dir`` names a LOCAL staging copy the shards are read from while
``--parent`` stays the recorded identity; a read source under ``/mnt/ai_data``
is refused unless ``--allow-share-read`` records the choice.

Without ``--streamed`` the call is handed to ``qf_build.build`` verbatim.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

# ----------------------------------------------------------- lane imports
#: ``composite/tools`` sits two levels under the campaign root on the server
#: (``$OC``) and in the workstation mirror (``reports/omni-campaign-2026-09``).
#: ``QF_LANE_DIR`` / ``QF_DISTILL_DIR`` override for a test that relocates them.
HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parents[1]
QWEN_FOLD_DIR = Path(os.environ.get("QF_LANE_DIR", CAMPAIGN_ROOT / "qwen-fold"))
DISTILL_DIR = Path(os.environ.get("QF_DISTILL_DIR", CAMPAIGN_ROOT / "distill"))
for lane in (QWEN_FOLD_DIR, DISTILL_DIR):
    if not (lane / "__init__.py").exists() and str(lane) not in sys.path:
        sys.path.insert(0, str(lane))

import qf_arch  # noqa: E402
import qf_build  # noqa: E402
import qf_corpus  # noqa: E402
import qf_export  # noqa: E402
import qf_mlp  # noqa: E402
import qf_solve  # noqa: E402
import tps_probe  # noqa: E402

LOG_PREFIX = "[qf-stream]"
RECEIPT_NAME = qf_build.RECEIPT_NAME
DTYPES = qf_build.DTYPES

#: Flags this entry adds on top of ``qf_build.parse_args``; stripped before the
#: lane's parser sees the argv, so the lane's CLI is used verbatim.
EXTRA_FLAGS = {
    "--streamed": ("streamed", False),
    "--resident-check": ("resident_check", False),
    "--allow-share-read": ("allow_share_read", False),
    "--full-sequence-preflight": ("full_sequence_preflight", False),
}
#: Valued flags this entry adds (``--flag VALUE``), likewise stripped.
EXTRA_VALUED = {
    "--solve-form": ("solve_form", "rows"),
    "--src-dir": ("src_dir", ""),
}
SHARE_ROOT = "/mnt/ai_data"


def log(message: str) -> None:
    print(message if message.startswith("[") else f"{LOG_PREFIX} {message}", flush=True)


# ----------------------------------------------------------------- readers
def checkpoint_weight_manifest(directory: str | Path) -> dict[str, Any]:
    """Hash the exact index and weight files the streamed reader will consume.

    A staging directory is a byte copy, not merely a compatible architecture.
    Hash in bounded chunks; never load a 27B checkpoint to check its identity.
    """
    root = Path(directory).resolve(strict=True)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        body = json.loads(index.read_text(encoding="utf-8"))
        weight_map = body.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SystemExit(f"{root}: missing or empty checkpoint weight_map")
        shards = list(weight_map.values())
        if any(not isinstance(name, str) or not name for name in shards):
            raise SystemExit(f"{root}: invalid checkpoint shard name")
        # HF from_pretrained prefers a single file; our streamed reader prefers
        # the index. Refuse layouts that would give the two engines different
        # source weights despite an identical indexed-shard manifest.
        if (root / "model.safetensors").exists() and set(shards) != {"model.safetensors"}:
            raise SystemExit(f"{root}: ambiguous single-file and indexed checkpoint layout")
        names = sorted(set(shards)) + [index.name]
    else:
        names = ["model.safetensors"]
    files: dict[str, Any] = {}
    for name in sorted(names):
        relative = Path(name)
        if relative.anchor or ".." in relative.parts or "\\" in name:
            raise SystemExit(f"{root}: unsafe checkpoint file path {name!r}")
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise SystemExit(f"{root}: missing or escaping checkpoint file {name!r}")
        before = path.stat()
        digest = hashlib.sha256()
        length = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                length += len(chunk)
        after = path.stat()
        stamp = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stamp(before) != stamp(after) or length != before.st_size:
            raise SystemExit(f"{root}: checkpoint file changed while hashing {name!r}")
        files[name] = {"bytes": length, "sha256": digest.hexdigest()}
    return {"files": files, "sha256": qf_arch.canonical_sha256(files)}


def bind_checkpoint_source(model_dir: str | Path, read_dir: str | Path) -> dict[str, Any]:
    """Refuse substituted staged weights BEFORE model construction or capture."""
    parent = Path(model_dir).resolve(strict=True)
    source = Path(read_dir).resolve(strict=True)
    declared = qf_arch.load_config(parent).to_dict()
    staged = qf_arch.load_config(source).to_dict()
    for body in (declared, staged):
        body.pop("_name_or_path", None)
        for sub in body.values():
            if isinstance(sub, dict):
                sub.pop("_name_or_path", None)
    if declared != staged:
        raise SystemExit(f"--src-dir {source} carries a different config.json than --parent {parent}")
    parent_manifest = checkpoint_weight_manifest(parent)
    source_manifest = parent_manifest if source == parent else checkpoint_weight_manifest(source)
    if parent_manifest != source_manifest:
        raise SystemExit(f"--src-dir {source} carries different checkpoint weights/index than --parent {parent}")
    return {
        "schema": "qf.checkpoint-source-identity.v1",
        "parent_dir": str(parent), "read_dir": str(source),
        "mode": "same-resolved-path" if source == parent else "byte-identical-staged-checkpoint",
        "config_sha256": qf_arch.canonical_sha256(declared),
        "weights": parent_manifest,
    }


def recheck_checkpoint_source(identity: dict[str, Any]) -> None:
    """Do not assemble or seal a child after its source changed during capture."""
    current = bind_checkpoint_source(identity["parent_dir"], identity["read_dir"])
    if current != identity:
        raise SystemExit("checkpoint source changed after the initial provenance check")


class SingleOrShardedReader(tps_probe.ShardReader):
    """``tps_probe.ShardReader`` that also accepts an un-sharded checkpoint.

    A tiny fixture saved by ``save_pretrained`` is one ``model.safetensors``
    with no index; the streamed engine is otherwise the same code path.
    """

    def __init__(self, model_dir: Path, device: str) -> None:
        from safetensors import safe_open

        model_dir = Path(model_dir)
        if (model_dir / "model.safetensors.index.json").is_file():
            super().__init__(model_dir, device)
            return
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise SystemExit(f"{model_dir}: neither model.safetensors.index.json nor model.safetensors")
        self._safe_open = safe_open
        self.model_dir = model_dir
        self.device = device
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            self.weight_map = {name: single.name for name in handle.keys()}
        self.total_size = single.stat().st_size
        self._handles = {}
        self.bytes_read = 0


# ------------------------------------------------------------ the engine
def make_foldable_layer_list(decoder_class: Any) -> Any:
    """``tps_probe``'s streamed layer list, plus a per-index narrow-MLP override.

    ``folded[index] = {"gate", "up", "down"}`` (host tensors in the model's
    dtype, narrow) makes ``_materialise(index)`` load the layer's non-MLP
    tensors from disk as usual and its MLP from the cache into a NARROW buffer
    -- one per buffer type, created on first use -- with ``strict=True`` load
    semantics kept, because the narrow tensors are put INTO the state dict the
    stock ``load_state_dict`` receives rather than loaded beside it.

    ``current_index`` is the index of the layer materialised last; the tap
    reads it from inside a hook on the shared ``mlp`` module.
    """

    base_class = tps_probe.make_streamed_layer_list(decoder_class)

    class FoldableStreamedLayerList(base_class):
        def __init__(self, config: Any, reader: Any, prefix: str, device: str, dtype: Any) -> None:
            super().__init__(config, reader, prefix, device, dtype)
            self.device = device
            self.dtype = dtype
            self.folded: dict[int, dict[str, torch.Tensor]] = {}
            self.wide_mlp = nn.ModuleDict(
                {kind: buffer.mlp for kind, buffer in self.buffers_by_type.items()}
            )
            self.narrow_mlp = nn.ModuleDict()
            self.current_index: int | None = None
            self.mlp_listeners: list[Any] = []

        def mlp_modules(self) -> list[nn.Module]:
            return list(self.wide_mlp.values()) + list(self.narrow_mlp.values())

        def _narrow_buffer(self, layer_type: str, keep: int) -> nn.Module:
            if layer_type in self.narrow_mlp:
                existing = self.narrow_mlp[layer_type]
                if int(existing.intermediate_size) == keep:
                    return existing
            wide = self.wide_mlp[layer_type]
            # The stock MLP class at the narrow width -- not a deepcopy of the
            # wide buffer, which would carry the tap's hooks with it and fire
            # every capture twice.
            with torch.device(self.device):
                narrow = type(wide)(self.text_config, keep)
            narrow = narrow.to(device=self.device, dtype=self.dtype).eval()
            self.narrow_mlp[layer_type] = narrow
            for listener in self.mlp_listeners:
                listener(narrow)
            return narrow

        def _materialise(self, index: int) -> Any:
            layer_type = str(self.text_config.layer_types[index])
            layer = self.buffers_by_type[layer_type]
            start = time.perf_counter()
            base = f"{self.prefix}.{index}."
            names = self.reader.names_under(base)
            folded = self.folded.get(index)
            if folded is None:
                layer.mlp = self.wide_mlp[layer_type]
                state = {name[len(base):]: self.reader.get(name) for name in names}
            else:
                keep = int(folded["gate"].shape[0])
                layer.mlp = self._narrow_buffer(layer_type, keep)
                state = {
                    name[len(base):]: self.reader.get(name)
                    for name in names
                    if not name[len(base):].startswith("mlp.")
                }
                state["mlp.gate_proj.weight"] = folded["gate"]
                state["mlp.up_proj.weight"] = folded["up"]
                state["mlp.down_proj.weight"] = folded["down"]
            layer.load_state_dict(state, strict=True, assign=False)
            del state
            self.layer_load_seconds += time.perf_counter() - start
            self.layer_loads += 1
            self.current_index = int(index)
            return layer

    return FoldableStreamedLayerList


class StreamedTrunk:
    """A streamed ``Qwen3_5TextModel`` + head, with the prologue exposed.

    The stock forward computes the causal mask, the linear-attention mask and
    the rotary tables ONCE per call and hands every layer the same objects;
    :meth:`prologue` computes exactly those (same functions, same arguments) so
    :meth:`run_layers` can start at any layer boundary from a cached residual.
    """

    def __init__(self, model_dir: str | Path, device: str, dtype: torch.dtype) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextModel,
            Qwen3_5TextRotaryEmbedding,
        )

        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        # Same rule as qf_harness.Harness: the fused kernels are CUDA-only and
        # the torch fallbacks are what every CPU capture in this campaign uses.
        self.kernels_disabled = (
            qf_arch.force_torch_kernels() if not str(device).startswith("cuda") else {}
        )
        self.config = qf_arch.load_config(self.model_dir)
        self.text_config = qf_arch.text_config(self.config)
        self.text_config.dtype = dtype
        self.text_config._attn_implementation = "eager"
        self.prefix = qf_arch.tensor_prefix(self.config).rstrip(".")

        with torch.device("meta"):
            trunk = Qwen3_5TextModel(self.text_config)
        self.reader = SingleOrShardedReader(self.model_dir, device=device)
        tc = self.text_config
        embed = nn.Embedding(tc.vocab_size, tc.hidden_size, dtype=dtype, device=device)
        norm = Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(device=device, dtype=dtype)
        with torch.no_grad():
            embed.weight.copy_(self.reader.get(f"{self.prefix}.embed_tokens.weight"))
            norm.weight.copy_(self.reader.get(f"{self.prefix}.norm.weight"))
        rotary = Qwen3_5TextRotaryEmbedding(config=tc).to(device)
        trunk.embed_tokens = embed
        trunk.norm = norm
        trunk.rotary_emb = rotary
        del trunk.layers
        trunk.layers = make_foldable_layer_list(Qwen3_5DecoderLayer)(
            tc, self.reader, f"{self.prefix}.layers", device, dtype
        )
        trunk.eval()
        self.trunk = trunk

        head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False, dtype=dtype, device=device)
        self.head_tied = not self.reader.has("lm_head.weight")
        with torch.no_grad():
            if self.head_tied:
                head.weight = embed.weight
            else:
                head.weight.copy_(self.reader.get("lm_head.weight"))
        self.head = head.eval()
        self.forward_calls = 0
        self.layer_forwards = 0

    # ------------------------------------------------------------ pieces
    @property
    def layers(self):
        return self.trunk.layers

    @property
    def count(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @torch.no_grad()
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.trunk.embed_tokens(ids.to(self.device))

    @torch.no_grad()
    def prologue(self, inputs_embeds: torch.Tensor) -> dict[str, Any]:
        """The stock ``Qwen3_5TextModel.forward`` prologue for a cache-less call."""

        from transformers.masking_utils import create_causal_mask

        trunk = self.trunk
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
        causal_mask = create_causal_mask(
            config=trunk.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = trunk._update_linear_attn_mask(None, None)
        position_embeddings = trunk.rotary_emb(inputs_embeds, position_ids)
        return {
            "text_position_ids": text_position_ids,
            "causal_mask": causal_mask,
            "linear_attn_mask": linear_attn_mask,
            "position_embeddings": position_embeddings,
        }

    @torch.no_grad()
    def run_layers(self, hidden: torch.Tensor, context: dict[str, Any], start: int, stop: int) -> torch.Tensor:
        """Layers ``start .. stop-1`` on ``hidden``, exactly as the stock loop calls them."""

        layer_types = self.text_config.layer_types
        for index in range(start, stop):
            layer = self.trunk.layers[index]
            mask = (
                context["linear_attn_mask"]
                if layer_types[index] == "linear_attention"
                else context["causal_mask"]
            )
            hidden = layer(
                hidden,
                position_embeddings=context["position_embeddings"],
                attention_mask=mask,
                position_ids=context["text_position_ids"],
                past_key_values=None,
                use_cache=False,
            )
            self.layer_forwards += 1
        return hidden

    @torch.no_grad()
    def final_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """The whole trunk, through the stock forward (no cache), normed."""

        self.forward_calls += 1
        return self.trunk(input_ids=ids.to(self.device), use_cache=False).last_hidden_state

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.final_hidden(ids))


# ------------------------------------------------------------------ the tap
class IndexAwareTap:
    """Hooks on the SHARED per-type ``mlp`` modules, keyed by the materialised index.

    ``state[index] = (args, kwargs, output)`` for every index in ``wanted``.
    The hooks are installed once on each buffer's MLP (wide and narrow, the
    latter as they are created) and read ``layers.current_index`` when they
    fire; nothing about the module object identifies the site, and that is the
    whole point.
    """

    def __init__(self, layers: Any) -> None:
        self.layers = layers
        self.wanted: set[int] = set()
        self.state: dict[int, tuple[tuple, dict, Any]] = {}
        self.enabled = False
        self.handles: list[Any] = []
        self.fired: list[int] = []
        for module in layers.mlp_modules():
            self._install(module)
        layers.mlp_listeners.append(self._install)

    def _install(self, module: nn.Module) -> None:
        self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
        self.handles.append(module.register_forward_hook(self._post, with_kwargs=True))

    def _pre(self, module, args, kwargs):
        if not self.enabled:
            return None
        index = self.layers.current_index
        if index in self.wanted:
            self.state[index] = (args, kwargs, None)
        return None

    def _post(self, module, args, kwargs, output):
        if not self.enabled:
            return None
        index = self.layers.current_index
        if index in self.wanted:
            pre_args, pre_kwargs, _ = self.state[index]
            self.state[index] = (pre_args, pre_kwargs, output)
            self.fired.append(int(index))
        return None

    def arm(self, wanted: list[int]) -> None:
        self.wanted = {int(i) for i in wanted}
        self.state = {}
        self.fired = []
        self.enabled = True

    def disarm(self) -> None:
        self.enabled = False
        self.wanted = set()
        self.state = {}

    def take(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mlp input, mlp output)`` for ``index``; refuses if the tap did not fire."""

        if index not in self.state or self.state[index][2] is None:
            raise SystemExit(
                f"the index-aware tap did not fire for layer {index}; fired {sorted(set(self.fired))}"
            )
        args, kwargs, output = self.state.pop(index)
        hidden = args[0] if args else kwargs["hidden_states"]
        if isinstance(output, tuple):
            output = output[0]
        return hidden, output

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ---------------------------------------------------------- the residual
class ResidualCache:
    """The residual stream at one layer boundary, per batch, on the host.

    ``boundary`` is the number of layers already applied.  ``advance(to)``
    runs layers ``boundary .. to-1`` (which by then are folded) on every
    cached batch and moves the boundary; the tensors are kept in the model's
    dtype, which is the dtype the stock forward passes between layers.
    """

    def __init__(self, trunk: StreamedTrunk, batches: list[tuple[int, torch.Tensor]]) -> None:
        self.trunk = trunk
        self.batches = batches
        self.boundary = 0
        self.hidden: dict[int, torch.Tensor] = {}
        self.bytes = 0

    @torch.no_grad()
    def advance(self, to: int) -> None:
        if to < self.boundary:
            raise SystemExit(f"residual cache cannot move backwards ({self.boundary} -> {to})")
        if to == self.boundary:
            return
        for index, ids in self.batches:
            embeds = self.trunk.embed(ids)
            context = self.trunk.prologue(embeds)
            hidden = self.hidden[index].to(self.trunk.device) if index in self.hidden else embeds
            hidden = self.trunk.run_layers(hidden, context, self.boundary, to)
            self.hidden[index] = hidden.to("cpu")
        self.boundary = to
        self.bytes = sum(t.numel() * t.element_size() for t in self.hidden.values())

    @torch.no_grad()
    def run_from_boundary(self, index: int, ids: torch.Tensor, stop: int) -> torch.Tensor:
        """Layers ``boundary .. stop-1`` on this batch's cached residual (not stored)."""

        embeds = self.trunk.embed(ids)
        context = self.trunk.prologue(embeds)
        hidden = self.hidden[index].to(self.trunk.device) if index in self.hidden else embeds
        return self.trunk.run_layers(hidden, context, self.boundary, stop)


# ------------------------------------------------------------- gram sink
def rows64(tensor: torch.Tensor) -> torch.Tensor:
    """``qf_harness.Harness._rows``: flat rows, fp64, on the CPU."""

    return tensor.detach().reshape(-1, tensor.shape[-1]).to("cpu", torch.float64)


def check_flag(fit_position: int) -> bool:
    """Is the ``fit_position``-th fit batch in the internal check split?

    The same rule as :func:`qf_corpus.fit_check_rows`, per batch instead of per
    row; ``test_check_split_matches_fit_check_rows`` holds the two together.
    """

    return (
        fit_position >= qf_corpus.INTERNAL_CHECK_OFFSET
        and (fit_position - qf_corpus.INTERNAL_CHECK_OFFSET) % qf_corpus.INTERNAL_CHECK_STRIDE == 0
    )


class SiteGrams:
    """Gram accumulation for one MLP site: no fp64 rows are ever kept.

    Three splits.  ``inner`` and ``check`` partition the fit rows exactly as
    :func:`qf_corpus.fit_check_rows` does; ``held`` carries only ``||Y||^2``
    and a row count because the held score is computed from the bf16 stash
    after the writer is known.  ``X``/``Y`` (the MLP's input and output, in the
    model's own dtype, so the stash is lossless) are kept on the host per batch
    for the scoring pass.
    """

    SPLITS = ("inner", "check", "held")

    def __init__(
        self, index: int, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
        *, accumulate: bool = True,
    ) -> None:
        self.index = index
        self.name = f"L{index:03d}"
        self.gate64 = gate.detach().to("cpu", torch.float64)
        self.up64 = up.detach().to("cpu", torch.float64)
        self.down64 = down.detach().to("cpu", torch.float64)
        #: Per-batch fp64 Gram accumulation (``X^T X`` summed one corpus batch
        #: at a time).  It is the receipted DIAGNOSTIC and the ``--solve-form
        #: gram`` path; the byte-identical ``rows`` solve reads the stash only,
        #: so a fold does not pay a second ``phi^T phi`` GEMM per site.
        self.accumulate = bool(accumulate)
        self.G: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        self.YY: dict[str, float] = {s: 0.0 for s in self.SPLITS}
        self.rows: dict[str, int] = {s: 0 for s in self.SPLITS}
        self.stash: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self.batches_seen: list[int] = []

    def add(self, batch_index: int, split: str, hidden: torch.Tensor, output: torch.Tensor) -> None:
        self.stash.append((split, hidden.detach().reshape(-1, hidden.shape[-1]).to("cpu"),
                           output.detach().reshape(-1, output.shape[-1]).to("cpu")))
        self.batches_seen.append(int(batch_index))
        self.rows[split] += int(hidden.reshape(-1, hidden.shape[-1]).shape[0])
        if not self.accumulate:
            return
        X = rows64(hidden)
        Y = rows64(output)
        self.YY[split] += float(Y.pow(2).sum())
        if split == "held":
            return
        phi = qf_solve.swiglu(X, self.gate64, self.up64)
        G = phi.T @ phi
        B = phi.T @ Y
        self.G[split] = G if split not in self.G else self.G[split] + G
        self.B[split] = B if split not in self.B else self.B[split] + B

    # ------------------------------------------------------------- rows
    def captured_rows(self) -> dict[str, torch.Tensor]:
        """The resident ``Harness.capture(pristine=...)`` blocks, from the stash.

        ``fit``/``hld`` are the MLP inputs and ``local_fit``/``local_hld`` the
        parent MLP's outputs on them, fp64, concatenated in corpus order --
        the same bytes the resident's ``_rows`` + ``torch.cat`` produce, because
        the stash holds the tensors in their own dtype and the up-cast is exact.
        """

        blocks: dict[str, list[torch.Tensor]] = {"fit": [], "hld": [], "local_fit": [], "local_hld": []}
        for split, X, Y in self.stash:
            bucket = "hld" if split == "held" else "fit"
            blocks[bucket].append(X.to(torch.float64))
            blocks[f"local_{bucket}"].append(Y.to(torch.float64))
        return {k: torch.cat(v, dim=0).contiguous() for k, v in blocks.items() if v}

    def row_gram(self) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        """``(G, B, ||Y||^2, rows)`` over the fit rows in the resident's ONE-GEMM form."""

        rows = self.captured_rows()
        phi = qf_solve.swiglu(rows["fit"], self.gate64, self.up64)
        Y = rows["local_fit"]
        return phi.T @ phi, phi.T @ Y, float(Y.pow(2).sum()), int(Y.shape[0])

    def accumulation_vs_rows(self) -> dict[str, Any]:
        """How far the per-batch accumulation sits from the one-GEMM Gram (fp64 order)."""

        if not self.accumulate:
            raise SystemExit(f"{self.name}: Grams were not accumulated; nothing to compare")
        G_acc, B_acc, YY_acc, rows_acc = self.fit_grams()
        G_row, B_row, YY_row, rows_row = self.row_gram()
        return {
            "rows": rows_acc,
            "rows_agree": rows_acc == rows_row,
            "G_max_abs_diff_rel": float((G_acc - G_row).abs().max() / G_row.abs().max()),
            "B_max_abs_diff_rel": float((B_acc - B_row).abs().max() / B_row.abs().max()),
            "YY_rel_diff": abs(YY_acc - YY_row) / max(abs(YY_row), 1e-300),
            "note": "per-batch fp64 accumulation vs one fp64 GEMM over the concatenated rows; differs only by summation order",
        }

    # ---------------------------------------------------------- derived
    def fit_grams(self) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        """``(G, B, ||Y||^2, rows)`` over ALL fit rows (inner + check)."""

        if not self.accumulate:
            raise SystemExit(f"{self.name}: Grams were not accumulated (rows-only capture)")
        if "inner" not in self.G:
            raise SystemExit(f"{self.name}: no fit rows were captured")
        G = self.G["inner"].clone()
        B = self.B["inner"].clone()
        if "check" in self.G:
            G += self.G["check"]
            B += self.B["check"]
        return G, B, self.YY["inner"] + self.YY["check"], self.rows["inner"] + self.rows["check"]

    def has_check(self) -> bool:
        return "check" in self.G

    def signature(self) -> dict[str, Any]:
        """Bytes of the fit Gram, for the determinism receipt."""

        G, B, YY, rows = self.fit_grams()
        return {
            "G_sha256": qf_arch.canonical_sha256(_tensor_digest_body(G)),
            "B_sha256": qf_arch.canonical_sha256(_tensor_digest_body(B)),
            "YY": YY,
            "rows": rows,
            "stash_sha256": [qf_arch.canonical_sha256([_tensor_digest_body(x), _tensor_digest_body(y)])
                             for _, x, y in self.stash],
        }


def _tensor_digest_body(tensor: torch.Tensor) -> str:
    import hashlib

    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    return hashlib.sha256(bytes(memoryview(flat.numpy()))).hexdigest()


def sse_from_gram(W: torch.Tensor, G: torch.Tensor, B: torch.Tensor, YY: float) -> float:
    """``||A W^T - Y||_F^2`` from ``G = A^T A``, ``B = A^T Y`` and ``||Y||^2``."""

    return float(((W @ G) * W).sum() - 2.0 * (W * B.T).sum() + YY)


def law_abiding_solve_split_gram(
    G_in: torch.Tensor, B_in: torch.Tensor,
    G_ck: torch.Tensor, B_ck: torch.Tensor, YY_ck: float,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """:func:`qf_solve.law_abiding_solve_split` from Grams alone.

    Same ladder, same veto, same selection rule (the admitted rung with the
    lowest check sum-of-squares, re-solved on all rows); the only difference is
    that the check score is ``tr(W G_ck W^T) - 2 tr(W B_ck) + ||Y_ck||^2``
    instead of a residual over rows the 27B path never materialises.
    """

    G_all = G_in + G_ck
    B_all = B_in + B_ck
    reference = float(prior.norm())
    rungs: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    for rel in qf_solve.LAMBDA_LADDER:
        candidate_in = qf_solve._ladder_rung(G_in, B_in, prior, rel, fallbacks, "inner-split gram")
        candidate = qf_solve._ladder_rung(G_all, B_all, prior, rel, fallbacks, "all-rows gram")
        rungs.append(
            {
                "lambda_relative": rel,
                "check_sse": sse_from_gram(candidate_in, G_ck, B_ck, YY_ck),
                "update_scale": float((candidate - prior).norm() / reference),
                "solved": candidate,
            }
        )
    free_scale = rungs[0]["update_scale"]
    admitted = [row for row in rungs if row["update_scale"] <= qf_solve.UPDATE_SCALE_CAP]
    if not admitted:
        raise SystemExit(
            f"no lambda on the ladder brings update_scale to {qf_solve.UPDATE_SCALE_CAP} "
            f"(free {free_scale:.3f}); THE LAW refuses the install"
        )
    chosen = min(admitted, key=lambda row: row["check_sse"])
    return chosen["solved"], {
        "update_scale": chosen["update_scale"],
        "lambda_relative": chosen["lambda_relative"],
        "free_solve_update_scale": free_scale,
        "lambda_selected_on": "internal fit split",
        "lambda_law_would_have_picked": admitted[0]["lambda_relative"],
        "check_sse_chosen": chosen["check_sse"],
        "check_sse_law_pick": admitted[0]["check_sse"],
        "ladder": [{k: v for k, v in row.items() if k != "solved"} for row in rungs],
        "solver": "cholesky" if not fallbacks else "cholesky + eigen fallback",
        "solver_fallback": fallbacks,
        "form": "gram (streamed accumulation; check SSE from the check-split Gram)",
    }


class ScoreAccumulator:
    """``qf_solve.write_scores`` over rows that arrive one batch at a time."""

    def __init__(self) -> None:
        self.diff2 = 0.0
        self.target2 = 0.0
        self.cos: list[torch.Tensor] = []

    def add(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.diff2 += float((pred - target).pow(2).sum())
        self.target2 += float(target.pow(2).sum())
        self.cos.append(torch.nn.functional.cosine_similarity(pred, target, dim=1))

    def result(self) -> dict[str, float]:
        cos = torch.cat(self.cos)
        return {
            "relL2": float(self.diff2 ** 0.5 / self.target2 ** 0.5),
            "cos": float(cos.mean()),
            "cos_p05": float(torch.quantile(cos, torch.tensor(0.05, dtype=cos.dtype))),
        }


# --------------------------------------------------------------- capture
class StreamedHarness:
    """The streamed engine, the sealed corpus, the tap and the residual cache."""

    def __init__(
        self, model_dir: str | Path, device: str, dtype: torch.dtype, manifest: dict[str, Any],
        *, read_dir: str | Path | None = None,
    ) -> None:
        read_dir = Path(read_dir or model_dir)
        self.source_identity = bind_checkpoint_source(model_dir, read_dir)
        self.engine = StreamedTrunk(read_dir, device, dtype)
        self.model_dir = read_dir
        self.parent_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.manifest = manifest
        self.batches = qf_corpus.corpus(manifest)
        self.split = qf_corpus.split_by_batch(manifest)
        self.tap = IndexAwareTap(self.engine.layers)
        self.residual = ResidualCache(self.engine, self.batches)
        self.gram_passes = 0
        fit_position = 0
        self.batch_split: dict[int, str] = {}
        for index, _ in self.batches:
            if self.split[index] == "fit":
                self.batch_split[index] = "check" if check_flag(fit_position) else "inner"
                fit_position += 1
            else:
                self.batch_split[index] = "held"

    @property
    def text_config(self):
        return self.engine.text_config

    def parent_mlp_weights(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = f"{self.engine.prefix}.layers.{index}.mlp."
        reader = self.engine.reader
        return (
            reader.get(base + "gate_proj.weight"),
            reader.get(base + "up_proj.weight"),
            reader.get(base + "down_proj.weight"),
        )

    @torch.no_grad()
    def capture_group_grams(self, sites: list[int], *, accumulate: bool = True) -> dict[int, SiteGrams]:
        """One corpus pass over the group's layers from the cached boundary.

        ``accumulate=False`` keeps the rows only (the ``rows`` solve form); the
        per-batch Gram accumulation is then skipped, which halves the CPU GEMM
        work of a fold at the 27B width.
        """

        sites = sorted(int(s) for s in sites)
        if sites[0] < self.residual.boundary:
            raise SystemExit(
                f"group {sites} starts below the residual boundary {self.residual.boundary}"
            )
        for index in sites:
            if index in self.engine.layers.folded:
                raise SystemExit(f"layer {index} is already folded; it cannot be captured pristine")
        grams = {
            index: SiteGrams(index, *self.parent_mlp_weights(index), accumulate=accumulate)
            for index in sites
        }
        stop = sites[-1] + 1
        self.tap.arm(sites)
        try:
            for batch_index, ids in self.batches:
                self.residual.run_from_boundary(batch_index, ids, stop)
                split = self.batch_split[batch_index]
                for index in sites:
                    hidden, output = self.tap.take(index)
                    grams[index].add(batch_index, split, hidden, output)
        finally:
            self.tap.disarm()
        self.gram_passes += 1
        return grams

    def install(self, index: int, sel: torch.Tensor, writer: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> dict[str, torch.Tensor]:
        """The narrow MLP for ``index``: verbatim rows, the solved writer, in the model dtype.

        The writer's dtype conversion is done by the SAME call the resident
        ``install_narrow_mlp`` makes (``.to(device=, dtype=)`` on the model's
        device) and the result is kept, so export re-rounds nothing.
        """

        index_t = sel.to(gate.device)
        folded = {
            "gate": gate[index_t].detach().to("cpu").contiguous(),
            "up": up[index_t].detach().to("cpu").contiguous(),
            "down": writer.to(device=self.device, dtype=self.dtype).detach().to("cpu").contiguous(),
        }
        self.engine.layers.folded[int(index)] = folded
        return folded

    def live_mlp_copy(self, index: int) -> nn.Module:
        """A stock MLP on the build device holding layer ``index``'s PARENT weights.

        It is the object ``qf_mlp.install_narrow_mlp`` mutates in the ``rows``
        solve form, so the verbatim-row gather and the writer's dtype cast run on
        exactly the device and dtype the resident lane runs them on.
        """

        layer_type = str(self.text_config.layer_types[index])
        wide = self.engine.layers.wide_mlp[layer_type]
        gate, up, down = self.parent_mlp_weights(index)
        with torch.device(self.device):
            module = type(wide)(self.text_config, int(gate.shape[0]))
        module = module.to(device=self.device, dtype=self.dtype).eval()
        with torch.no_grad():
            module.gate_proj.weight.copy_(gate)
            module.up_proj.weight.copy_(up)
            module.down_proj.weight.copy_(down)
        return module


class _SiteShim:
    """The three things :func:`qf_mlp.narrow_mlp_site` reads from a harness.

    ``dtype``, ``manifest`` and ``layers()[index].mlp``: the last is a
    throwaway stock MLP on the build device, so the lane's own install code
    runs unmodified and the narrow tensors are read back off it.
    """

    def __init__(self, harness: StreamedHarness, index: int, mlp: nn.Module) -> None:
        self.dtype = harness.dtype
        self.manifest = harness.manifest
        self._layers = {int(index): _Layer(mlp)}

    def layers(self) -> dict[int, "_Layer"]:
        return self._layers


class _Layer:
    def __init__(self, mlp: nn.Module) -> None:
        self.mlp = mlp


# ------------------------------------------------------------- the fold
def narrow_mlp_site_rows(
    harness: StreamedHarness,
    grams: SiteGrams,
    keep: int,
    *,
    lambda_split: bool = True,
) -> dict[str, Any]:
    """:func:`qf_mlp.narrow_mlp_site` ITSELF on the stashed rows -- the byte-identical form.

    The resident lane solves a site on the concatenated fp64 rows: one
    ``swiglu`` over all fit rows, ``pivoted_columns`` (which forms ``F^T F`` in
    one GEMM), ``law_abiding_solve_split`` on ``phi[:, sel]``.  Summation order
    inside those GEMMs is what fixes the last ulp of the writer, and the writer
    is rounded to bf16 on install -- so reproducing the resident child byte for
    byte means running the resident's own function on the resident's own row
    blocks, which the stash provides losslessly.  Nothing here is re-derived:
    the site report is the lane's, plus the engine tags.
    """

    index = grams.index
    live = harness.live_mlp_copy(index)
    shim = _SiteShim(harness, index, live)
    prepared = {"rows": grams.captured_rows(), "pristine": live}
    report = qf_mlp.narrow_mlp_site(
        shim, index, keep, lambda_split=lambda_split, dtype=harness.dtype, prepared=prepared,
    )
    del prepared
    mlp = shim.layers()[index].mlp
    if int(mlp.gate_proj.weight.shape[0]) != keep:
        raise SystemExit(f"{grams.name}: install left the MLP at {int(mlp.gate_proj.weight.shape[0])}, not {keep}")
    harness.engine.layers.folded[int(index)] = {
        "gate": mlp.gate_proj.weight.detach().to("cpu").contiguous(),
        "up": mlp.up_proj.weight.detach().to("cpu").contiguous(),
        "down": mlp.down_proj.weight.detach().to("cpu").contiguous(),
    }
    del live, shim, mlp
    report["engine"] = "streamed"
    report["form"] = "rows (qf_mlp.narrow_mlp_site on the bf16-stashed rows up-cast to fp64; byte-identical to the resident lane)"
    report["fit_rows"] = grams.rows["inner"] + grams.rows["check"]
    return report


SOLVE_FORMS = ("rows", "gram")


def narrow_mlp_site_gram(
    harness: StreamedHarness,
    grams: SiteGrams,
    keep: int,
    *,
    lambda_split: bool = True,
) -> dict[str, Any]:
    """:func:`qf_mlp.narrow_mlp_site`, from this site's Grams and stash.

    Same steps in the same order: identity precondition, pivoted selection,
    prior-anchored ladder (internal split or plain), scores, install.  The
    identity check and the selection read the fit Gram; the ladder reads the
    inner/check Grams; the scores read the bf16 stash.
    """

    index = grams.index
    name = grams.name
    gate, up, down = grams.gate64, grams.up64, grams.down64
    G_fit, B_fit, YY_fit, rows_fit = grams.fit_grams()

    identity = (max(sse_from_gram(down, G_fit, B_fit, YY_fit), 0.0) ** 0.5) / (YY_fit ** 0.5)
    if identity > qf_mlp.IDENTITY_TOLERANCE:
        raise SystemExit(
            f"{name}: the parent's own writer reproduces its own output to only "
            f"relL2 {identity:.3e} on the captured design matrix (line "
            f"{qf_mlp.IDENTITY_TOLERANCE}).  The capture, not the fold, is wrong."
        )

    selection_report: dict[str, Any] = {}
    sel = qf_solve.pivoted_columns_gram(G_fit, keep, report=selection_report)
    if int(sel.numel()) != keep:
        raise SystemExit(
            f"{name}: pivoted selection returned {int(sel.numel())} of {keep} channels "
            f"-- the design matrix is rank deficient at this width"
        )

    prior = down[:, sel]
    if lambda_split:
        if not grams.has_check():
            raise SystemExit(f"{name}: the internal check split is empty; nothing to select lambda on")
        writer, guard = law_abiding_solve_split_gram(
            grams.G["inner"][sel][:, sel], grams.B["inner"][sel],
            grams.G["check"][sel][:, sel], grams.B["check"][sel], grams.YY["check"],
            prior,
        )
    else:
        writer, guard = qf_solve.law_abiding_solve_gram(G_fit[sel][:, sel], B_fit[sel], prior)

    # Scores from the stash: the selected SwiGLU columns are recomputed in fp64
    # from the bf16 rows, which is what the resident's `phi_fit[:, sel]` is.
    gate_sel, up_sel = gate[sel], up[sel]
    fit_scores, drop_scores, held_scores = ScoreAccumulator(), ScoreAccumulator(), ScoreAccumulator()
    for split, X, Y in grams.stash:
        A = qf_solve.swiglu(rows64(X), gate_sel, up_sel)
        Y64 = rows64(Y)
        if split == "held":
            held_scores.add(A @ writer.T, Y64)
        else:
            fit_scores.add(A @ writer.T, Y64)
            drop_scores.add(A @ prior.T, Y64)
    scores = {"fit": fit_scores.result()}
    if held_scores.cos:
        scores["held"] = held_scores.result()

    gate_live, up_live, _ = harness.parent_mlp_weights(index)
    harness.install(index, sel, writer, gate_live, up_live)

    hidden = int(gate.shape[1])
    return {
        "site": name,
        "layer": index,
        "axis": "intermediate_size",
        "parent_intermediate": int(gate.shape[0]),
        "child_intermediate": keep,
        "component_ratio": keep / float(gate.shape[0]),
        "selection": "evidence",
        "selection_greedy": selection_report,
        "shared_group_capture": True,
        "engine": "streamed",
        "form": "gram (X^T X, X^T Y, ||Y||^2 per split; rows stashed bf16 for scores only)",
        "selected_channels": [int(i) for i in sel],
        "identity_precondition_relL2": identity,
        "solve": guard,
        "scores": scores,
        "no_repair_scores": drop_scores.result(),
        "fit_rows": rows_fit,
        "parameters_removed": 3 * hidden * (int(gate.shape[0]) - keep),
    }


def mlp_axis_streamed(
    harness: StreamedHarness,
    keep: int,
    *,
    sites: list[int] | None = None,
    group_size: int = 3,
    lambda_split: bool = True,
    solve_form: str = "rows",
    log=print,
) -> dict[str, Any]:
    """:func:`qf_mlp.mlp_axis` on the streamed engine: same groups, same law.

    ``solve_form="rows"`` (default) is the resident solve on the stashed rows
    and is the form that reproduces the resident child byte for byte;
    ``"gram"`` solves from the per-batch fp64 Gram accumulation and never
    materialises the design matrix (equal to fp64 summation order).
    """

    if solve_form not in SOLVE_FORMS:
        raise SystemExit(f"solve form {solve_form!r} is not one of {SOLVE_FORMS}")
    solve_site = narrow_mlp_site_rows if solve_form == "rows" else narrow_mlp_site_gram
    count = harness.engine.count
    sites = list(range(count)) if sites is None else sorted(int(s) for s in sites)
    parent_width = int(harness.text_config.intermediate_size)
    if keep >= parent_width:
        raise SystemExit(f"keep {keep} is not a fold of {parent_width}")
    group_size = max(1, int(group_size))
    groups = [sites[start:start + group_size] for start in range(0, len(sites), group_size)]
    reports: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        started = time.time()
        # Layers below the group are final (folded or never asked for): apply
        # them to the cached residual once, then the group's capture starts at
        # this boundary.
        harness.residual.advance(group[0])
        grams = harness.capture_group_grams(group, accumulate=(solve_form == "gram"))
        for layer_index in group:
            report = solve_site(harness, grams.pop(layer_index), keep, lambda_split=lambda_split)
            report["group_index"] = group_index
            report["captured_with"] = list(group)
            reports.append(report)
            log(
                f"[qf] C2 {report['site']}: {report['parent_intermediate']} -> {keep} "
                f"({1.0 / report['component_ratio']:.2f}x)  fit relL2 "
                f"{report['scores']['fit']['relL2']:.4f}  lambda "
                f"{report['solve']['lambda_relative']:g}  "
                f"[group {group_index} {time.time() - started:.0f}s]"
            )
        del grams
        gc.collect()

    uniform = len({r["child_intermediate"] for r in reports}) == 1
    covered = sorted(r["layer"] for r in reports)
    complete = covered == list(range(count))
    if uniform and complete:
        qf_arch.config_field(harness.text_config, "intermediate_size", keep)
    return {
        "axis": "intermediate_size",
        "kind": "channel coreset (verbatim rows) + prior-anchored writer refit",
        "engine": "streamed (tps_probe.build_streamed_text_model layer list, index-aware tap, host residual cache)",
        "solve_form": solve_form,
        "sites": len(reports),
        "keep": keep,
        "parent_intermediate": parent_width,
        "component_ratio": keep / float(parent_width),
        "component_fold": parent_width / float(keep),
        "schedule": {
            "group_size": group_size,
            "groups": [list(g) for g in groups],
            "corpus_passes": len(groups),
            "law": (
                "one capture per group, shared by every site in it; the next group is "
                "captured against a model that already contains the previous one"
            ),
            "engine_note": (
                "a corpus pass here runs only the group's layers from the cached "
                "residual boundary; the boundary is advanced through the folded "
                "layers before the next group's pass"
            ),
        },
        "config_standard": bool(uniform and complete),
        "config_note": (
            "intermediate_size is one global field; a child that folds only some "
            "sites is a parent-shaped diagnostic and is NOT exportable as standard"
        ),
        "parameters_removed": sum(r["parameters_removed"] for r in reports),
        "per_site": reports,
    }


# --------------------------------------------------------------- parity
@torch.no_grad()
def full_sequence_preflight(harness: StreamedHarness) -> dict[str, Any]:
    """Exercise one complete evidence row before expensive Gram/parity allocation.

    This is a finite-output and measured-memory check, NOT behavioural acceptance
    or a parent comparison. It keeps the exact corpus, kernels and layer weights.
    The last-token head avoids materialising a sequence-by-vocabulary tensor.
    """
    import psutil

    if harness.residual.boundary != 0 or harness.engine.layers.folded:
        raise SystemExit("full-sequence preflight requires the pristine engine")
    batch_index, ids = harness.batches[0]
    cuda = str(harness.device).startswith("cuda")
    before_rss = psutil.Process().memory_info().rss
    started = time.time()
    if cuda:
        torch.cuda.synchronize(harness.device)
        torch.cuda.reset_peak_memory_stats(harness.device)
    hidden = harness.engine.final_hidden(ids)
    expected_shape = (*ids.shape, int(harness.engine.text_config.hidden_size))
    if tuple(hidden.shape) != expected_shape or not bool(torch.isfinite(hidden).all()):
        raise SystemExit("full-sequence preflight produced invalid final hidden states")
    last_logits = harness.engine.head(hidden[:, -1:])
    expected_logits = (ids.shape[0], 1, int(harness.engine.text_config.vocab_size))
    if tuple(last_logits.shape) != expected_logits or not bool(torch.isfinite(last_logits).all()):
        raise SystemExit("full-sequence preflight produced invalid last-token logits")
    if cuda:
        torch.cuda.synchronize(harness.device)
    report = {
        "schema": "qf.full-sequence-preflight.v1", "batch_index": int(batch_index),
        "input_shape": list(ids.shape), "final_hidden_shape": list(hidden.shape),
        "input_ids_sha256": hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "layers": harness.engine.count, "finite_hidden_and_last_logits": True,
        "last_token_argmax": last_logits.argmax(-1).cpu().tolist(),
        "seconds": round(time.time() - started, 3),
        "host_rss_before_bytes": before_rss,
        "host_rss_after_bytes": psutil.Process().memory_info().rss,
        "host_available_after_bytes": psutil.virtual_memory().available,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(harness.device) if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(harness.device) if cuda else None,
        "scope": "one complete corpus row; no fold quality, parity, or all-modality acceptance claim",
    }
    del hidden, last_logits
    return report


@torch.no_grad()
def streamed_parity(harness: StreamedHarness, sites: tuple[int, ...] = (0, 1, 2), *, resident_check: bool = False) -> dict[str, Any]:
    """:func:`qf_harness.harness_parity` for the streamed engine.

    1. determinism: two Gram passes over the sites are bit-identical (Grams AND
       the bf16 stash);
    2. hook placement: ``swiglu(X) @ down^T`` reproduces the tapped output at
       every site (relL2 from the fit Gram);
    3. exactness under the tap: the whole-trunk final hidden state with the tap
       armed is bit-identical to the same call with it disarmed;
    4. (``resident_check``) the stock resident model on the same device gives
       bit-identical logits on batch 0 and bit-identical MLP inputs at the
       sites -- the streaming analogue of tps_probe's stream-verify.
    """

    if harness.residual.boundary != 0 or harness.engine.layers.folded:
        raise SystemExit("parity must run on the pristine engine before any fold")
    count = harness.engine.count
    sites = tuple(s for s in sites if s < count)
    report: dict[str, Any] = {"sites": list(sites), "engine": "streamed"}

    first = harness.capture_group_grams(list(sites))
    second = harness.capture_group_grams(list(sites))
    sig_first = {i: first[i].signature() for i in sites}
    sig_second = {i: second[i].signature() for i in sites}
    report["deterministic"] = sig_first == sig_second
    if not report["deterministic"]:
        raise SystemExit("streamed capture is not deterministic; nothing downstream is measurable")
    report["fit_gram_sha256"] = {f"L{i:03d}": sig_first[i]["G_sha256"][:16] for i in sites}

    placement: dict[str, float] = {}
    accumulation: dict[str, Any] = {}
    for i in sites:
        G, B, YY, _ = first[i].fit_grams()
        placement[f"L{i:03d}"] = (max(sse_from_gram(first[i].down64, G, B, YY), 0.0) ** 0.5) / (YY ** 0.5)
        # Run-book test (2) as a receipt line: the per-batch accumulation vs
        # the resident's one-GEMM Gram on the same rows, fp64 order only.
        accumulation[f"L{i:03d}"] = first[i].accumulation_vs_rows()
    report["hook_placement_relL2"] = placement
    report["hook_placement_worst"] = max(placement.values()) if placement else 0.0
    report["gram_accumulation_vs_rows"] = accumulation
    report["gram_accumulation_worst_rel"] = max(
        (v["G_max_abs_diff_rel"] for v in accumulation.values()), default=0.0
    )
    del first, second

    ids = harness.batches[0][1]
    plain = harness.engine.final_hidden(ids)
    harness.tap.arm(list(sites))
    try:
        hooked = harness.engine.final_hidden(ids)
        fired = sorted(set(harness.tap.fired))
    finally:
        harness.tap.disarm()
    report["final_hidden_bit_identical_under_tap"] = bool(torch.equal(plain, hooked))
    report["tap_fired_for"] = fired
    if not report["final_hidden_bit_identical_under_tap"]:
        raise SystemExit("arming the tap changed the trunk's output")
    if fired != sorted(sites):
        raise SystemExit(f"the tap fired for {fired}, wanted {sorted(sites)}")

    if resident_check:
        report["resident"] = resident_comparison(harness, sites, ids)
    report["forward_calls"] = harness.engine.forward_calls
    report["layer_forwards"] = harness.engine.layer_forwards
    return report


@torch.no_grad()
def resident_comparison(harness: StreamedHarness, sites: tuple[int, ...], ids: torch.Tensor) -> dict[str, Any]:
    """The stock resident model vs the streamed engine on ONE batch, same device."""

    import qf_harness

    resident = qf_harness.Harness(harness.model_dir, harness.device, harness.dtype, harness.manifest)
    layers = qf_arch.layers(resident.model)
    grabbed: dict[int, torch.Tensor] = {}
    handles = []

    def spy(index: int):
        def hook(module, args, kwargs):
            grabbed[index] = (args[0] if args else kwargs["hidden_states"]).detach()
            return None
        return hook

    for i in sites:
        handles.append(layers[i].mlp.register_forward_pre_hook(spy(i), with_kwargs=True))
    try:
        resident_logits = resident.logits(ids).float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    resident_hidden = {i: grabbed[i].cpu() for i in sites}
    del layers, resident
    gc.collect()
    if str(harness.device).startswith("cuda"):
        torch.cuda.empty_cache()

    streamed_logits = harness.engine.logits(ids).float().cpu()
    harness.tap.arm(list(sites))
    try:
        harness.engine.final_hidden(ids)
        streamed_hidden = {i: harness.tap.take(i)[0].cpu() for i in sites}
    finally:
        harness.tap.disarm()
    per_site = {
        f"L{i:03d}": {
            "mlp_input_bit_identical": bool(torch.equal(resident_hidden[i], streamed_hidden[i])),
            "mlp_input_max_abs_diff": float((resident_hidden[i].float() - streamed_hidden[i].float()).abs().max()),
        }
        for i in sites
    }
    diff = (resident_logits - streamed_logits).abs()
    return {
        "logits_bit_identical": bool(torch.equal(resident_logits, streamed_logits)),
        "logits_max_abs_diff": float(diff.max()),
        "logits_argmax_agreement": float((resident_logits.argmax(-1) == streamed_logits.argmax(-1)).float().mean()),
        "per_site": per_site,
        "all_mlp_inputs_bit_identical": all(v["mlp_input_bit_identical"] for v in per_site.values()),
        "note": "stock class, same device and dtype, one batch; the streamed engine's only claim is that it changes no arithmetic",
    }


# --------------------------------------------------------------- export
@torch.no_grad()
def assemble_child(parent_dir: str | Path, folded: dict[int, dict[str, torch.Tensor]], keep: int, dtype: torch.dtype) -> tuple[Any, dict[str, int], dict[str, int]]:
    """A CPU copy of the parent with every folded site's narrow MLP installed."""

    qf_arch.force_torch_kernels()
    config = qf_arch.load_config(parent_dir)
    loader = qf_arch.loader_class(config)
    model = loader.from_pretrained(str(parent_dir), dtype=dtype, attn_implementation="eager").to("cpu").eval()
    parent_counts = qf_arch.assert_parameter_count(model, model.config)
    layers = qf_arch.layers(model)
    for index in sorted(folded):
        entry = folded[index]
        mlp = layers[index].mlp
        hidden = int(mlp.gate_proj.weight.shape[1])
        new_gate = nn.Linear(hidden, keep, bias=False, dtype=dtype)
        new_up = nn.Linear(hidden, keep, bias=False, dtype=dtype)
        new_down = nn.Linear(keep, hidden, bias=False, dtype=dtype)
        new_gate.weight.data.copy_(entry["gate"])
        new_up.weight.data.copy_(entry["up"])
        new_down.weight.data.copy_(entry["down"])
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = new_gate, new_up, new_down
        mlp.intermediate_size = keep
    if sorted(folded) == list(range(len(layers))):
        qf_arch.config_field(qf_arch.text_config(model.config), "intermediate_size", keep)
    qf_arch.assert_layer_types_agree(model)
    child_counts = qf_arch.assert_parameter_count(model, model.config)
    return model, parent_counts, child_counts


# ---------------------------------------------------------------- build
def refuse_unsupported(args: argparse.Namespace) -> None:
    """The streamed entry folds C2 and nothing else; say so before loading anything."""

    qf_build.refuse_illegal_combinations(args)
    asked = {flag for flag, on in qf_build.structural_axis_flags(args).items() if on and flag != "--mlp-keep"}
    if args.vocab_keep is not None:
        asked.add("--vocab-keep")
    if asked:
        raise SystemExit(
            f"--streamed implements the C2 MLP coreset only; {', '.join(sorted(asked))} "
            f"need the resident qf_build.py"
        )
    if args.mlp_keep is None:
        raise SystemExit("--streamed implements --mlp-keep (C2) only; no MLP rung was asked for")
    if args.mlp_select != "evidence":
        raise SystemExit("--streamed selects on the evidence Gram only (--mlp-select evidence)")
    if args.carry_mtp:
        # A C2 child moves intermediate_size, which the MTP block reads.
        raise SystemExit("--carry-mtp is never legal for a C2 child (intermediate_size moves); use --drop-mtp")
    if args.solve_form not in SOLVE_FORMS:
        raise SystemExit(f"--solve-form must be one of {SOLVE_FORMS}, not {args.solve_form!r}")
    read_dir = str(Path(read_dir_of(args)).resolve())
    if read_dir.startswith(SHARE_ROOT) and not args.allow_share_read:
        raise SystemExit(
            f"the weights would be read from {read_dir}, which is on the share ({SHARE_ROOT}); "
            f"the streamed engine reads every layer of every batch and would price the whole "
            f"build at share speed.  Stage the parent on local disk and point --src-dir at it, "
            f"or pass --allow-share-read to record the choice."
        )


def read_dir_of(args: argparse.Namespace) -> str:
    """Where the shards are READ from: ``--src-dir`` if given, else ``--parent``.

    ``--parent`` stays the recorded identity (config, tokenizer, receipt); a
    local staging copy of the same checkpoint can be named as the read source
    so nothing prices at share speed.  The copy must carry the parent's config
    and index; ``build_streamed`` refuses a mismatched config.
    """

    return args.src_dir or args.parent


def mtp_inventory(directory: str | Path) -> list[dict[str, Any]]:
    """Count indexed MTP tensors from headers, without loading their weights."""
    root = Path(directory).resolve(strict=True)
    reader = SingleOrShardedReader(root, device="cpu")
    rows = []
    for name in sorted(reader.weight_map):
        if not name.startswith(qf_arch.carried_prefixes()):
            continue
        shard = reader.weight_map[name]
        relative = Path(shard)
        if relative.anchor or ".." in relative.parts or "\\" in shard:
            raise SystemExit(f"unsafe MTP shard path {shard!r}")
        if not (root / relative).resolve(strict=True).is_relative_to(root):
            raise SystemExit(f"escaping MTP shard path {shard!r}")
        shape = list(reader._handle(shard).get_slice(name).get_shape())
        rows.append({"name": name, "shape": shape, "parameters": math.prod(shape)})
    return rows


def drop_exported_mtp(parent_dir: str | Path, child_dir: str | Path) -> dict[str, Any]:
    """Make the canonical text config agree with absent MTP weights.

    MTP was never loaded by the stock HF model. Its exact omitted parameter
    count is reported separately, never credited to the C2 fold ratio.
    """
    path = Path(child_dir) / "config.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("model_type") == "qwen3_5":
        text = body.get("text_config")
        if not isinstance(text, dict):
            raise SystemExit("VL child has no canonical text_config for MTP metadata")
        field = "text_config.mtp_num_hidden_layers"
    else:
        text = body
        field = "mtp_num_hidden_layers"
    dropped = mtp_inventory(parent_dir)
    if mtp_inventory(child_dir):
        raise SystemExit("cannot declare MTP dropped: exported child still contains MTP tensors")
    previous = text.get("mtp_num_hidden_layers")
    text["mtp_num_hidden_layers"] = 0
    # Keep an existing legacy alias consistent, but do not add one to VL configs.
    if text is not body and "mtp_num_hidden_layers" in body:
        body["mtp_num_hidden_layers"] = 0
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {
        "mtp": "dropped" if dropped else "not present in the parent checkpoint",
        "config_field": field,
        "previous_config_value": previous,
        "exported_config_value": 0,
        "dropped_tensors": dropped,
        "dropped_parameters": sum(row["parameters"] for row in dropped),
        "count_method": "indexed safetensors header shapes; no tensor materialization",
        "credited_to_fold_ratio": False,
        "note": "MTP is absent from the child; omitted MTP parameters are outside the stock-model C2 ratio",
    }


def build_streamed(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    refuse_unsupported(args)
    dtype = DTYPES[args.dtype]
    manifest = qf_corpus.read_manifest(args.evidence)
    evidence_tokenizer = qf_build.assert_evidence_tokenizer(
        manifest, args.parent, allow_mismatch=args.allow_tokenizer_mismatch
    )
    parent_config_snapshot = copy.deepcopy(qf_arch.load_config(args.parent))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; a build never overwrites a child -- move it or pick another tag")

    read_dir = read_dir_of(args)
    log(f"parent {args.parent} (streamed engine, {args.device}, {args.dtype}, solve form {args.solve_form})")
    if read_dir != args.parent:
        log(f"weights read from --src-dir {read_dir}")
    harness = StreamedHarness(args.parent, args.device, dtype, manifest, read_dir=read_dir)
    closed = qf_arch.parameter_count(parent_config_snapshot)
    parent_total = closed["total_with_vision"] if closed["vision"] else closed["total"]
    log(f"parent parameters {parent_total:,} (closed form; the live count is checked at export)")

    receipts: dict[str, Any] = {
        "parity": None,
        "full_sequence_preflight": None,
        "causal_masks": None,
        "evidence_tokenizer": evidence_tokenizer,
        "source_identity": harness.source_identity,
        "engine": {
            "streamed": True,
            "solve_form": args.solve_form,
            "weights_read_from": str(read_dir),
            "layer_buffer_types": sorted(harness.engine.layers.buffers_by_type.keys()),
            "kernels_disabled": harness.engine.kernels_disabled,
            "lm_head_tied_to_embedding": harness.engine.head_tied,
            "checkpoint_total_size_gb": tps_probe.gigabytes(harness.engine.reader.total_size),
        },
    }
    log(
        f"evidence tokenizer {evidence_tokenizer['manifest_tokenizer_sha256'][:16]} "
        f"vs parent {str(evidence_tokenizer['parent_tokenizer_sha256'])[:16]}: "
        f"{'match' if evidence_tokenizer['matches'] else 'MISMATCH (allowed)'}"
    )
    if args.full_sequence_preflight:
        receipts["full_sequence_preflight"] = full_sequence_preflight(harness)
        log("full-sequence preflight: " + json.dumps(receipts["full_sequence_preflight"], sort_keys=True))
    if not args.no_parity:
        receipts["parity"] = streamed_parity(harness, resident_check=args.resident_check)
        log(
            f"parity: deterministic, hook placement worst relL2 "
            f"{receipts['parity']['hook_placement_worst']:.3e}, final hidden bit-identical under the tap"
        )
        if args.resident_check:
            resident = receipts["parity"]["resident"]
            log(
                f"resident check: logits bit-identical {resident['logits_bit_identical']} "
                f"(max abs {resident['logits_max_abs_diff']:.3e}), mlp inputs bit-identical "
                f"{resident['all_mlp_inputs_bit_identical']}"
            )

    axis = mlp_axis_streamed(
        harness, args.mlp_keep, group_size=args.mlp_group,
        lambda_split=not args.no_lambda_split, solve_form=args.solve_form, log=log,
    )
    axes = [axis]
    engine_facts = {
        "layer_loads": harness.engine.layers.layer_loads,
        "layer_load_seconds": round(harness.engine.layers.layer_load_seconds, 3),
        "layer_forwards": harness.engine.layer_forwards,
        "gram_passes": harness.gram_passes,
        "bytes_read": harness.engine.reader.bytes_read,
        "residual_cache_bytes": harness.residual.bytes,
        "residual_cache_dtype": str(dtype).replace("torch.", ""),
        "residual_boundary_at_end": harness.residual.boundary,
    }
    folded = dict(harness.engine.layers.folded)
    keep = int(args.mlp_keep)
    harness.tap.remove()
    del harness
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    log("assembling the child on the CPU from the parent + the folded sites")
    recheck_checkpoint_source(receipts["source_identity"])
    model, parent_counts, child_counts = assemble_child(read_dir, folded, keep, dtype)
    log(f"child parameters {child_counts['live']:,} (closed form agrees; parent live {parent_counts['live']:,})")
    export = qf_export.export_child(model, args.out, parent_dir=args.parent, sidecars=True)
    child_config = copy.deepcopy(model.config)
    del model
    gc.collect()
    recheck_checkpoint_source(receipts["source_identity"])

    mtp = {"mtp": "not present in the parent index"}
    if args.drop_mtp:
        mtp = drop_exported_mtp(read_dir, out)

    verify, reloaded = qf_export.verify_child(args.out, device="cpu", dtype=torch.float32)
    log(
        f"verified: {verify['loaded_class']} re-loaded from disk, "
        f"{verify['parameters_counted_live']:,} parameters, digest {verify['child_tensors_sha256'][:16]}"
    )
    if verify["child_tensors_sha256"] != export["child_tensors_sha256"]:
        raise SystemExit("the child changed between export and verification")
    reloaded_config = copy.deepcopy(reloaded.config)
    del reloaded
    gc.collect()

    gguf = qf_export.convert_to_gguf(
        args.out,
        args.gguf_out or (out.parent / f"{out.name}.gguf"),
        llama_cpp=args.gguf or "/home/cereal/llama.cpp",
        dry_run=not args.gguf,
    )
    review = qf_export.frozen_review(
        review_script=args.review_script or "run_omni_frozen_review.py",
        receipt=out / RECEIPT_NAME,
        child_dir=args.out,
        parent_dir=args.parent,
        out=args.review_out or (out / "frozen-review-dossier.json"),
        device=args.device,
        dtype="bfloat16" if args.dtype == "bfloat16" else args.dtype,
        dry_run=not args.review_script,
    )
    receipt = qf_export.shape_receipt(
        parent_config=parent_config_snapshot,
        child_config=reloaded_config,
        parent_dir=args.parent,
        export=export,
        verify=verify,
        axes=axes,
        evidence=qf_corpus.evidence_receipt(manifest, args.evidence),
        stage=args.stage,
        parent_code=args.parent_code,
        lineage=[v for v in args.lineage.split(",") if v.strip()],
        notes=args.notes,
    )
    receipt["tag"] = args.tag
    receipt["passed_through"] = mtp
    receipt["preconditions"] = receipts
    receipt["engine"] = engine_facts
    receipt["builder"] = "qf_stream27.py --streamed (composite/tools)"
    receipt["gguf"] = gguf
    receipt["review_hook"] = review
    receipt["wall_time_seconds"] = round(time.time() - started, 3)
    receipt["forward_calls"] = engine_facts["layer_forwards"]
    receipt["receipt_sha256"] = qf_arch.canonical_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    written = qf_export.write_receipt(receipt, out / RECEIPT_NAME)
    log(
        f"child {args.out}  net stored ratio {receipt['storage']['net_stored_ratio']:.4f} "
        f"({receipt['storage']['net_fold']:.2f}x)  receipt {written['receipt_sha256'][:16]}"
    )
    return written


# ------------------------------------------------------------------ CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``qf_build.parse_args`` plus this entry's flags, stripped before the lane's parser."""

    argv = list(sys.argv[1:] if argv is None else argv)
    extra = {dest: default for dest, default in EXTRA_FLAGS.values()}
    extra.update({dest: default for dest, default in EXTRA_VALUED.values()})
    rest: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token in EXTRA_FLAGS:
            extra[EXTRA_FLAGS[token][0]] = True
        elif token in EXTRA_VALUED:
            try:
                extra[EXTRA_VALUED[token][0]] = next(tokens)
            except StopIteration:
                raise SystemExit(f"{token} needs a value") from None
        elif any(token.startswith(flag + "=") for flag in EXTRA_VALUED):
            flag, value = token.split("=", 1)
            extra[EXTRA_VALUED[flag][0]] = value
        else:
            rest.append(token)
    args = qf_build.parse_args(rest)
    for dest, value in extra.items():
        setattr(args, dest, value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.streamed:
        log("no --streamed: handing the call to qf_build.build verbatim")
        qf_build.build(args)
        return 0
    build_streamed(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
