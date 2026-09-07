#!/usr/bin/env python3
"""run_omni_frozen_review.py -- the family-neutral, Chimera-native review runner.

The HF runner (`calibration/oc_run_frozen_bank.py`) reviews a stock-transformers
checkpoint.  The composite child of COMPOSITE step 1 is a Chimera Omni checkpoint
(`chimera_omni.checkpoint.v1`, written by `safetensors_io.write_safetensors`) and no
stock class loads it, so the same review surface has to be produced by a runner
that speaks Chimera.  This file is that runner.  It writes the EXACT
`oc_run_frozen_bank` dossier schema (`neural-forge-omni-parent-calibration-dossier-v1`)
so `oc_assess.py`, `oc_package.py` and `wavea_review.sh` consume it unchanged, and it
pays the same protocol lines the HF runner pays:

  * the same digest-pinned bank v1 (`0ee6bf14...`), the same chat template with
    `add_generation_prompt=True`, greedy argmax with a KV/recurrent cache, the
    1,024-token cap, the three termination states plus `out-of-memory`, and
    "grades nothing";
  * the decode-path preflight (here: cached step decode vs. full-prefix recompute,
    the Chimera analogue of manual-loop-vs-generate), the emit-zero floor, the
    three-prompt smoke, the directory-digest identity, the lane digest and its
    validity tuple.

WHAT IS CHIMERA-SPECIFIC, ALL RECORDED IN THE DOSSIER:

  `--split A,B`      free-run decode with blocks 0..A-1 on the first visible CUDA
                     device and blocks A..A+B-1 on the second; `embed`, `norm`,
                     `lm_head` (and the tied table, kept tied), the native-media
                     bridge/head and the aux heads live on the FIRST device.  The
                     residual moves device once per boundary.  The split trunk is
                     `ChimeraOmniModel._forward_trunk_with_cache` for
                     `residual_mode == "none"` with `.to(device)` at the boundary;
                     `test_split_decoder_matches_reference_cached_path` pins the two
                     paths bit-for-bit on one device, `block_attnres` is refused.
  `--stream --teacher-forced`
                     proxy scoring only: blocks are streamed one at a time from the
                     shards (one block resident, `ChimeraBlock` built on `meta` and
                     assign-loaded), every bank row is scored TEACHER-FORCED on a
                     reference response (`--reference-dossier`), and the record's
                     `response_verbatim` is the per-position argmax under teacher
                     forcing.  Generation is REFUSED in this mode: `--maximum-new-
                     tokens` must be 0 (free-run at ~0.7 s x 64 block loads per token
                     is not a review).  `mode` = `teacher-forced-proxy`.
  `--tap-residuals`  per-layer FFN-input rows (`blocks.{i}.norm2` input, pre-norm,
                     plus the normed rows on request) written to
                     `<tap-out>/L{i:02d}/ffn_input_rows.safetensors` for C17.
  `--vision-cards`   bank-v2 vision rows: the PNG is packetised into native media
                     tokens (16x16x3 patches zero-padded to `native_feature_dim`,
                     positions = patch centres) added at `media`-band placeholder
                     ids in front of the chat text, then decoded greedily.  On an
                     imported checkpoint the bridge is random-init and the dossier
                     says so; the mechanism is what is exercised.
  `--media-out`      native media TOKEN STREAMS for a declared ledger band (e.g.
                     `music_codes`): greedy decode restricted to the band rows,
                     written per case with non-empty / distinct / loop diagnostics.
                     Continuous samplers (image/video/volume x0 diagnostics) need
                     the S6 companions and are refused by name here.

USAGE (server; free-run on GPU6 + GPU4)

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,4 $PY run_omni_frozen_review.py \
      --model composite/s0/q38-27bu-chimera-bf16-sep04 --tokenizer-dir /home/cereal/models/Qwen3.8-27B-Uncensored \
      --label PROBE-S0 --stage parent --split 46,24 --dtype bfloat16 --thinking off \
      --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json \
      --probe-prompts calibration/banks/matrix_campaign_probe_prompts_v1.json \
      --smoke-prompts calibration/banks/matrix_campaign_smoke_prompts_v1.json \
      --maximum-new-tokens 1024 --ledger-band-logit-mask --out out/oc-dossier-PROBE-S0-NOTHINK.json

  proxy (GPU4):  ... --stream --teacher-forced --maximum-new-tokens 0 \
                     --reference-dossier calibration/out/oc-dossier-Q38-27BU-NOTHINK.json --device cuda:0

  S4 taps (GPU4): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY run_omni_frozen_review.py \
      --model composite/s3/c2k4352-chimera-bf16 --tokenizer-dir /home/cereal/models/Qwen3.8-27B-Uncensored \
      --label S4-TAPS --stream --teacher-forced --maximum-new-tokens 0 --dtype bfloat16 --device cuda:0 \
      --input qwen-fold/evidence/omni-evidence-27bu-v1.json --held-only --micro-batch 1 \
      --tap-residuals --tap-out /mnt/lexar/omni-campaign/composite/taps/c2k4352/ \
      --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json \
      --probe-prompts calibration/banks/matrix_campaign_probe_prompts_v1.json --skip-smoke \
      --out composite/s4/oc-dossier-S4-TAPS.json

  `--micro-batch` is the streamed engine's rows-per-pass: it defaults to 1 for --input
  evidence rows (a 4,096-token row through a 27B gated_gqa block materialises fp32
  scores [B, 24, 4096, 4096] = 12.9 GB per copy, two copies live under masked_fill +
  softmax; 8 rows OOM a 24 GB card) and to 8 for bank rows; it is recorded in the
  dossier and in taps_receipt.json (C17 replays the tap selection with it).  An OOM
  in a streamed pass is a REFUSAL with `<out>.refused.json` written, never a crash.
  The run refuses to start on a CUDA card above 2,000 MiB used (standing rule) and
  records CUDA_VISIBLE_DEVICES, every card's name and free memory in `placement`.
  RoPE inverse frequencies stay fp32 in both loaders whatever --dtype is.

Exit codes: 0 written, 3 refusal (nothing sealed), 1 unexpected error.
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

HERE = Path(__file__).resolve().parent
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/import/rq27b/tree_c5e6052"
DEFAULT_CALIBRATION_DIR = "/home/cereal/omni-campaign/calibration"

TOOL_NAME = "run_omni_frozen_review"
TOOL_VERSION = "1.2.0"
RUNTIME = "chimera-omni-native"
KERNEL_REPORT_SCHEMA = "chimera-omni-kernel-path-v1"
MODE_FREE_RUN = "free-run"
MODE_PROXY = "teacher-forced-proxy"
TAP_SCHEMA = "chimera-omni-ffn-input-taps-v1"
MEDIA_STREAM_SCHEMA = "chimera-omni-band-token-stream-v1"
IMAGE_PATCH = 16
IMAGE_TYPE_ID = 1
EXIT_REFUSED = 3
CHARTED_KIND = "charted"
GPU_BUSY_LIMIT_MIB = 2000
DEFAULT_MICRO_BATCH_BANK = 8
DEFAULT_MICRO_BATCH_EVIDENCE = 1
REFUSAL_SCHEMA = "chimera-omni-review-refusal-v1"
ZERO_GAIN_KEEP = 64


class Refusal(RuntimeError):
    """The review surface could not be established.  Nothing is sealed.

    `receipt` (optional) is written beside `--out` as `<out>.refused.json` by `main`,
    so a refusal raised hours into a streamed pass (an OOM in a gated_gqa block after
    the KDA recurrence ran) leaves a record instead of a bare crash.
    """

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None):
        super().__init__(message)
        self.receipt = receipt


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Binding the two code trees this runner is made of.
# --------------------------------------------------------------------------


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refusal(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bind_chimera_tree(tree: str | Path, declared_commit: str | None) -> dict[str, Any]:
    """Put `<tree>/src` first on sys.path and record which commit the code is.

    Reuses `s0_import_27b.resolve_tree_commit` / `bind_chimera_tree` so a declared
    `--chimera-commit` is checked the way the import tool checks it.
    """

    s0 = _load_module("s0_import_27b_for_review", HERE / "s0_import_27b.py")
    root = Path(tree).resolve()
    if declared_commit:
        try:
            return s0.bind_chimera_tree(root, declared_commit)
        except s0.Refusal as exc:
            raise Refusal(str(exc)) from exc
    src = root / "src"
    if not (src / "chimera_omni" / "__init__.py").is_file():
        raise Refusal(f"{root} has no src/chimera_omni package")
    commit, how = s0.resolve_tree_commit(root)
    sys.path.insert(0, str(src))
    import chimera_omni  # noqa: WPS433

    loaded = Path(chimera_omni.__file__).resolve().parents[1]
    if loaded != src.resolve():
        raise Refusal(f"chimera_omni was imported from {loaded}, not from {src}")
    return {"tree": str(root), "src": str(src), "commit": commit, "commit_source": how}


def bind_calibration(directory: str | Path) -> tuple[Any, Any]:
    """`oc_calib` and `oc_run_frozen_bank` from the calibration lane, imported from source.

    The dossier schema, the bank loader, the form/termination bookkeeping and the
    render/thinking helpers are READ from those files rather than copied here, so
    a schema change there is a schema change here.
    """

    root = Path(directory).resolve()
    for name in ("oc_calib.py", "oc_run_frozen_bank.py", "oc_bank_v2_draft.py"):
        if not (root / name).is_file():
            raise Refusal(f"calibration lane {root} has no {name}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import oc_calib  # noqa: WPS433
    import oc_run_frozen_bank  # noqa: WPS433

    return oc_calib, oc_run_frozen_bank


# --------------------------------------------------------------------------
# Checkpoint loading: meta construction + assign, placed per block.
# --------------------------------------------------------------------------


def parse_split(text: str | None) -> list[int] | None:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise Refusal("--split is empty")
    counts = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise Refusal(f"--split entries must be integers, got {part!r}") from exc
        if value <= 0:
            raise Refusal(f"--split entries must be positive, got {value}")
        counts.append(value)
    return counts


def block_index_of(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    return int(name.split(".", 2)[1])


def refuse_charted(cfg: Any) -> None:
    """The streamed proxy still lacks charted media layouts; refuse that lane.

    SplitDecoder supports charted text identity and explicitly laid-out media
    prefill. This guard is not used by split placement/loading.
    """

    kinds = [str(cfg.block_kind(i)) for i in range(int(cfg.num_layers))]
    charted = [i for i, kind in enumerate(kinds) if kind == CHARTED_KIND]
    if charted:
        raise Refusal(
            f"block_pattern names {len(charted)} charted block(s) at {charted[:8]}: "
            "the streamed proxy has no charted media layout; use the pinned split runtime"
        )


def fp32_source_names(cfg: Any) -> set[str]:
    """Exact stored H3 coefficient islands, not every FP32 meta placeholder."""
    specs = [spec for spec in getattr(cfg, "charted_blocks", ()) if spec.modulation == "h3_curve6"]
    if not specs:
        return set()
    from chimera_omni.charted import H3_FP32_PARAMETERS
    return {f"blocks.{spec.layer_index}.charted.{name}" for spec in specs for name in H3_FP32_PARAMETERS}


def materialize_charted_buffers(block: Any, device: str) -> None:
    """Restore the source-owned, non-persistent type IDs after meta loading."""
    if getattr(block, "charted", None) is not None:
        import torch
        block.charted.routed_type_ids = torch.tensor(
            sorted(block.charted.spec.type_ids), device=device, dtype=torch.long
        )


def rope_inv_freq(reference_gqa: Any, device: str) -> Any:
    """The source's fp32 RoPE inverse frequencies, on `device`, NEVER cast to the run dtype.

    `GatedGQAAttention` registers `inv_freq` fp32 and computes the angles in fp32
    (`positions.float() * inv_freq`); the HF teacher keeps inv_freq fp32 too.  A bf16
    copy carries a relative error up to 2^-9 per frequency, i.e. the relative-position
    angle scaled by (1 + eps): ~2 rad at distance 1,024 on the high-frequency dims, so
    the 1,024-token free-run lanes and the 4,096-token taps would silently leave the
    teacher while every fp32 test still passed.
    """

    import torch

    inv_freq = reference_gqa.inv_freq
    if inv_freq.dtype != torch.float32:
        raise Refusal(f"the source's GatedGQAAttention.inv_freq is {inv_freq.dtype}, not fp32; the RoPE contract changed")
    return inv_freq.detach().to(device=device, dtype=torch.float32)


def placement_for(cfg: Any, split: list[int] | None, devices: list[str]) -> dict[str, Any]:
    """Which device each block (and everything else) lives on."""

    num_layers = int(cfg.num_layers)
    if split is None:
        split = [num_layers]
    if len(split) > len(devices):
        raise Refusal(f"--split names {len(split)} segments but only {len(devices)} device(s) are given")
    if sum(split) != num_layers:
        raise Refusal(f"--split {split} sums to {sum(split)}, the checkpoint has {num_layers} blocks")
    block_device: list[str] = []
    for segment, device in zip(split, devices):
        block_device.extend([device] * segment)
    return {
        "split": list(split),
        "devices": list(devices[: len(split)]),
        "block_device": block_device,
        "head_device": devices[0],
        "note": (
            "blocks are placed contiguously in split order; embed, norm, lm_head (tied "
            "table kept tied), the native-media bridge/head and the aux heads are on the "
            "first device; the residual crosses a device boundary once per segment."
        ),
    }


def tensor_device(name: str, placement: dict[str, Any]) -> str:
    index = block_index_of(name)
    if index is None:
        return str(placement["head_device"])
    return str(placement["block_device"][index])


def load_chimera_split(
    root: Path,
    *,
    placement: dict[str, Any],
    dtype: Any,
    verify_tensors: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """`ChimeraOmniModel` from a checkpoint directory, built on `meta`, assign-loaded.

    Why not `safetensors_io.load_chimera_checkpoint`: it constructs the model on the
    CPU in fp32 (108 GB for the 27B) and holds every raw shard buffer beside it.
    Here one shard's views are alive at a time and every tensor goes straight to
    its device.  The tie is re-established by handing both names ONE Parameter.
    `inv_freq` (the only non-persistent buffer) is copied from a reference
    `GatedGQAAttention` built by the source's own constructor.
    """

    import torch
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import (
        read_checkpoint_config,
        read_index,
        read_safetensors_file,
        shard_paths,
    )
    from chimera_omni.modeling import ChimeraOmniModel
    from chimera_omni.modeling.layers import GatedGQAAttention

    document = read_checkpoint_config(root)
    cfg = ChimeraConfig.from_manifest(document["config"])
    if cfg.residual_mode != "none":
        raise Refusal(
            f"residual_mode={cfg.residual_mode!r}: the split decoder mirrors the plain "
            "pre-norm residual trunk only; block_attnres is not placed across devices here"
        )
    index = read_index(root) or {}
    tied: dict[str, str] = dict(index.get("metadata", {}).get("tied_weights", {}))
    expected_sha: dict[str, str] = dict(index.get("metadata", {}).get("tensor_sha256", {}))

    with torch.device("meta"):
        model = ChimeraOmniModel(cfg)
    target = model.state_dict()
    fp32_names = fp32_source_names(cfg)
    param_names = {key for key, _ in model.named_parameters()}
    state: dict[str, Any] = {}
    read_bytes = 0
    for shard in shard_paths(root):
        views = read_safetensors_file(shard)
        for name, view in views.items():
            if name not in target:
                raise Refusal(f"checkpoint tensor {name!r} has no place in this model")
            if verify_tensors and name in expected_sha and view.sha256() != expected_sha[name]:
                raise Refusal(f"tensor {name!r} does not match the sha256 recorded in the index")
            reference = target[name]
            device = tensor_device(name, placement)
            if name in fp32_names and view.dtype != "F32":
                raise Refusal(f"H3 source coefficient {name!r} must be stored in FP32, got {view.dtype}")
            tensor = view.torch(
                device=device, dtype=(torch.float32 if name in fp32_names else
                                      dtype if reference.is_floating_point() else reference.dtype)
            )
            read_bytes += len(view.buffer)
            state[name] = torch.nn.Parameter(tensor, requires_grad=False) if name in param_names else tensor
        del views
    for alias, owner in tied.items():
        if owner not in state:
            raise Refusal(f"tied tensor {alias!r} points at missing tensor {owner!r}")
        state[alias] = state[owner]
    missing = sorted(set(target) - set(state))
    if missing:
        raise Refusal(f"checkpoint is missing tensors: {missing[:5]}")
    model.load_state_dict(state, strict=True, assign=True)
    del state

    reference_gqa = GatedGQAAttention(cfg) if any(
        isinstance(block.mixer, GatedGQAAttention) for block in model.blocks
    ) else None
    for block_index, block in enumerate(model.blocks):
        materialize_charted_buffers(block, placement["block_device"][block_index])
        if isinstance(block.mixer, GatedGQAAttention) and block.mixer.inv_freq is not None:
            block.mixer.inv_freq = rope_inv_freq(reference_gqa, placement["block_device"][block_index])
    if cfg.tie_word_embeddings and model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr():
        raise Refusal("tied checkpoint lost its shared embedding table on load")
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise Refusal(f"tensors were never materialised: {leftover[:5]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = {
        "directory": str(root),
        "schema": document.get("schema"),
        "chimera_commit": document.get("chimera_commit"),
        "source": document.get("source"),
        "tensors": len(target),
        "tied_weights": dict(sorted(tied.items())),
        "total_size": index.get("metadata", {}).get("total_size"),
        "bytes_read": read_bytes,
        "tensor_sha256_verified": bool(verify_tensors),
        "load_dtype": str(dtype),
        "rope_inv_freq_dtype": "torch.float32",
        "fp32_source_coefficients": sorted(fp32_names),
        "charted_runtime": "text identity and explicit native-media prefill; continuous sampling is separate",
    }
    return model, cfg, receipt


# --------------------------------------------------------------------------
# The split decoder: the reference cached path, one `.to(device)` per boundary.
# --------------------------------------------------------------------------


class SplitDecoder:
    """Cached causal decode over a `ChimeraOmniModel` whose blocks sit on >= 1 device.

    Mirrors `ChimeraOmniModel._forward_trunk_with_cache` for residual_mode "none"
    line for line (attention with cache, residual add, feed-forward, validity
    where) and adds nothing but the device moves.  The test file asserts the
    single-device case is bit-identical to `forward_causal_with_cache`.
    """

    def __init__(self, model: Any, cfg: Any, placement: dict[str, Any], *, ledger_band_logit_mask: bool):
        import torch

        self.torch = torch
        self.model = model
        self.cfg = cfg
        self.placement = placement
        self.head_device = str(placement["head_device"])
        self.mask_band = bool(ledger_band_logit_mask)
        self.taps: Any = None
        from chimera_omni.modeling import ChimeraCausalCache

        self._cache_class = ChimeraCausalCache

    def _trunk(self, x: Any, valid: Any, cache: Any, *, row_types: Any = None,
               noise_levels: Any = None, source_positions: Any = None) -> tuple[Any, Any]:
        torch = self.torch
        model = self.model
        blocks = model.blocks
        if cache is None:
            block_caches = (None,) * len(blocks)
        else:
            if len(cache.block_caches) != len(blocks):
                raise ValueError("causal cache does not match the model block schedule")
            block_caches = cache.block_caches
        next_caches = []
        for index, (block, block_cache) in enumerate(zip(blocks, block_caches)):
            device = self.placement["block_device"][index]
            if str(x.device) != device:
                x = x.to(device)
            v = valid.to(device=device, dtype=torch.bool)
            if index == 0:
                x = torch.where(v.unsqueeze(-1), x, torch.zeros_like(x))
            if getattr(block, "charted", None) is not None:
                from chimera_omni.modeling.layers import MixerHistory
                types = None if row_types is None else row_types.to(device)
                levels = None if noise_levels is None else noise_levels.to(device)
                positions = (None if source_positions is None else
                             {name: value.to(device) for name, value in source_positions.items()})
                if types is not None:
                    x = x + block.charted_output(x, types, levels, v, positions)
                x = torch.where(v.unsqueeze(-1), x, torch.zeros_like(x))
                next_caches.append(MixerHistory(inputs=x[:, :0], valid_mask=v[:, :0]))
                continue
            attention, next_cache = block.attention_output_with_cache(x, block_cache, v)
            x = x + attention
            x = x + block.feed_forward_output(x, v)
            x = torch.where(v.unsqueeze(-1), x, torch.zeros_like(x))
            next_caches.append(next_cache)
        return x, self._cache_class(tuple(next_caches))

    def forward(self, input_ids: Any, cache: Any = None, *, media: dict[str, Any] | None = None) -> dict[str, Any]:
        """Logits for `input_ids` given `cache`; returns the next cache too."""

        torch = self.torch
        model = self.model
        ids = input_ids.to(self.head_device)
        valid = model._token_validity(ids, None)
        model._validate_token_ids(ids)
        x = model.embed(ids)
        row_types = noise_levels = source_positions = None
        if media is not None:
            if cache is not None:
                raise Refusal("native media is supported only in the initial prefill, not an existing causal cache")
            raw_noise = media.get("noise_levels")
            if media.get("bidirectional_refinement", False):
                raise Refusal("cached split prefill does not implement bidirectional trunk refinement")
            model._validate_native_contract(
                features=media["features"], type_ids=media["type_ids"], positions=media["positions"],
                indices=media["indices"], valid_mask=media["valid_mask"],
                update_mask=media.get("update_mask"), noise_levels=raw_noise,
                targets=None, loss_mask=None, target_kind=media.get("target_kind", "reconstruction"),
            )
            x, _count = model._add_native_media(
                x,
                media["features"].to(self.head_device),
                media["type_ids"].to(self.head_device),
                media["positions"].to(self.head_device),
                media["indices"].to(self.head_device),
                media["valid_mask"].to(self.head_device),
                None if raw_noise is None else raw_noise.to(self.head_device),
            )
            row_types, noise_levels = model._charted_layout(
                ids, _count, media["indices"], media["type_ids"], media["valid_mask"], raw_noise
            )
            source_positions = media.get("charted_source_positions")
            if row_types is not None and raw_noise is None and any(
                spec.modulation == "h3_curve6" and any(bool(row_types.eq(t).any()) for t in spec.type_ids)
                for spec in self.cfg.charted_blocks
            ):
                raise Refusal("H3 media prefill requires explicit actual per-type sigma values")
        x, next_cache = self._trunk(x, valid, cache, row_types=row_types,
                                  noise_levels=noise_levels, source_positions=source_positions)
        if str(x.device) != self.head_device:
            x = x.to(self.head_device)
        v = valid.to(self.head_device)
        hidden = model.norm(x)
        hidden = torch.where(v.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        logits = model.lm_head(hidden)
        if self.mask_band:
            logits = model._mask_ledger_band_logits(logits)
        return {"logits": logits, "hidden": hidden, "cache": next_cache}

    def band_logits(self, logits: Any, band: tuple[int, int]) -> Any:
        return logits[..., band[0] : band[1]]


# --------------------------------------------------------------------------
# Greedy generation on the split decoder: `oc_calib.generate_greedy`, ported.
# --------------------------------------------------------------------------


def generate_greedy_chimera(
    decoder: SplitDecoder,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    oc_calib: Any,
    cap: int,
    stops: set[int],
    label: str,
    thinking: str,
    media_for_row: Any = None,
    keep_ids: bool = True,
) -> list[dict[str, Any]]:
    """One record per row with the HF runner's fields (+ `generated_token_ids`)."""

    torch = decoder.torch
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            started = time.monotonic()
            messages = oc_calib.messages_for(row)
            prompt = build_prompt_ids(tokenizer, messages, thinking=thinking, oc_calib=oc_calib)
            media = None
            media_note = None
            if media_for_row is not None:
                media, media_note = media_for_row(row, prompt["ids"])
                if media is not None:
                    prompt = {**prompt, "ids": media["input_ids"]}
            ids = torch.tensor([prompt["ids"]], dtype=torch.long)
            prompt_tokens = int(ids.shape[-1])
            generated: list[int] = []
            termination = "censored-at-cap"
            error_text = None
            result = None
            cache = None
            try:
                result = decoder.forward(ids, None, media=media)
                cache = result["cache"]
                for _step in range(cap):
                    next_id = int(result["logits"][0, -1].argmax().item())
                    if next_id in stops:
                        termination = "complete"
                        break
                    generated.append(next_id)
                    current = torch.tensor([[next_id]], dtype=torch.long)
                    result = decoder.forward(current, cache)
                    cache = result["cache"]
            except torch.OutOfMemoryError as exc:
                termination = "out-of-memory"
                error_text = f"{type(exc).__name__}: {exc}"
                log(f"{label} {row['id']}: OUT OF MEMORY at {prompt_tokens} prompt tokens -- recorded, continuing")
            except RuntimeError as exc:
                termination = "runtime-error"
                error_text = f"{type(exc).__name__}: {exc}"
                log(f"{label} {row['id']}: {error_text}")
            result = None
            cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            record = {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "prompt_messages": messages,
                "prompt_tokens": prompt_tokens,
                "response_verbatim": decoded,
                "response_sha256": oc_calib.canonical_sha256({"text": decoded}),
                "logits_to_keep_used": False,
                "generated_tokens": len(generated),
                "generated_token_ids_sha256": oc_calib.canonical_sha256({"ids": [int(x) for x in generated]}),
                "loop_detected": oc_calib.detect_loop(generated),
                "termination": termination,
                "runtime_error": error_text,
                "wall_time_seconds": round(time.monotonic() - started, 3),
            }
            if keep_ids:
                record["generated_token_ids"] = [int(x) for x in generated]
            if media_note is not None:
                record["native_media"] = media_note
            records.append(record)
            log(
                f"{label} {index}/{len(rows)} {row['id']}: prompt {prompt_tokens} tok -> "
                f"{len(generated)} tok ({termination}) in {record['wall_time_seconds']:.1f}s"
            )
    return records


def build_prompt_ids(tokenizer: Any, messages: list[dict[str, Any]], *, thinking: str, oc_calib: Any) -> dict[str, Any]:
    """Chat template + `add_generation_prompt=True`, exactly as the HF runner renders it.

    Vision rows carry list content; the text parts are rendered and the image
    parts are dropped from the TEXT (the pixels enter as native media tokens).
    """

    text_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if part.get("type") == "text")
        text_messages.append({**message, "content": content})
    text = oc_calib.render_prompt(tokenizer, text_messages, thinking=thinking)
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    return {"text": text, "ids": [int(x) for x in ids]}


def decode_path_preflight_chimera(
    decoder: SplitDecoder, tokenizer: Any, row: dict[str, Any], *, oc_calib: Any, stops: set[int], thinking: str, steps: int = 8
) -> dict[str, Any]:
    """Eight tokens: cached step decode vs. full-prefix recompute.  Published, not acted on."""

    torch = decoder.torch
    prompt = build_prompt_ids(tokenizer, oc_calib.messages_for(row), thinking=thinking, oc_calib=oc_calib)
    ids = torch.tensor([prompt["ids"]], dtype=torch.long)
    cached: list[int] = []
    recomputed: list[int] = []
    with torch.no_grad():
        result = decoder.forward(ids, None)
        cache = result["cache"]
        for _ in range(steps):
            next_id = int(result["logits"][0, -1].argmax().item())
            cached.append(next_id)
            if next_id in stops:
                break
            result = decoder.forward(torch.tensor([[next_id]], dtype=torch.long), cache)
            cache = result["cache"]
        result = None
        cache = None
        running = list(prompt["ids"])
        for _ in range(steps):
            out = decoder.forward(torch.tensor([running], dtype=torch.long), None)
            next_id = int(out["logits"][0, -1].argmax().item())
            recomputed.append(next_id)
            if next_id in stops:
                break
            running.append(next_id)
        out = None
    common = min(len(cached), len(recomputed))
    return {
        "case_id": str(row["id"]),
        "steps": steps,
        "manual_loop_ids": cached,
        "greedy_generate_ids": recomputed,
        "agree": bool(cached[:common] == recomputed[:common] and len(cached) == len(recomputed)),
        "logits_to_keep_available": False,
        "decode_path_used": "cached step decode over ChimeraCausalCache (the protocol's argmax loop)",
        "why_this_is_published": (
            "Chimera has no `generate`; the cached step path (KDA recurrent state + "
            "MixerHistory) is compared with a full-prefix recompute at every step.  "
            "Disagreement does NOT switch paths and does NOT fail the run -- it is "
            "recorded so no reader grades text produced by an unchecked decode path."
        ),
    }


def emit_zero_floor_chimera(decoder: SplitDecoder, tokenizer: Any, probe_prompts: list[str], *, oc_calib: Any) -> dict[str, Any]:
    """`oc_calib.emit_zero_floor` on the Chimera path: same schema, same definition."""

    torch = decoder.torch
    rows: list[dict[str, Any]] = []
    terminals = []
    with torch.no_grad():
        for index, prompt in enumerate(probe_prompts, start=1):
            ids = torch.tensor([tokenizer(prompt, return_tensors=None)["input_ids"]], dtype=torch.long)
            out = decoder.forward(ids, None)
            terminal = out["hidden"][0, -1].float()
            logits = out["logits"][0, -1].float()
            finite_logits = logits[torch.isfinite(logits)]
            terminals.append(terminal)
            rows.append(
                {
                    "probe_index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "terminal_l2": float(terminal.norm().item()),
                    "terminal_finite": bool(torch.isfinite(terminal).all().item()),
                    "logit_std": float(finite_logits.std().item()) if finite_logits.numel() > 1 else 0.0,
                    "logit_finite": bool(torch.isfinite(finite_logits).all().item()) and finite_logits.numel() > 0,
                    "argmax_token_id": int(logits.argmax().item()),
                }
            )
            out = None
    distinct = len({row["argmax_token_id"] for row in rows})
    passed = (
        all(row["terminal_finite"] and row["logit_finite"] for row in rows)
        and min(row["terminal_l2"] for row in rows) > 0.0
        and min(row["logit_std"] for row in rows) > 0.0
    )
    return {
        "schema": oc_calib.EMIT_ZERO_FLOOR_SCHEMA,
        "probes": rows,
        "probe_count": len(rows),
        "distinct_argmax_tokens": distinct,
        "min_terminal_l2": min(row["terminal_l2"] for row in rows),
        "min_logit_std": min(row["logit_std"] for row in rows),
        "reference_terminal_relative_l2": None,
        "terminal_states_sha256": oc_calib.canonical_sha256({"rows": [round(row["terminal_l2"], 6) for row in rows]}),
        "passed": bool(passed),
        "definition": (
            "finite, non-zero terminal hidden state (post-norm, last position) and "
            "non-constant finite logits on every probe; the ledger-band rows masked to "
            "-inf are excluded from the logit statistics.  A parent has no parent, so the "
            "relL2 form is null and is not faked."
        ),
    }


def smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        if record["termination"] == "complete":
            record["terminated"] = "eos"
        elif record["termination"] == "runtime-error":
            record["terminated"] = "error"
        elif record["loop_detected"]:
            record["terminated"] = "loop"
        else:
            record["terminated"] = "cap"
    return records


# --------------------------------------------------------------------------
# FFN-input taps (C17).
# --------------------------------------------------------------------------


def zero_gain_coordinates(norm: Any, *, keep: int = ZERO_GAIN_KEEP) -> dict[str, Any]:
    """The §3b trap, recorded: coordinates where an RMSNorm gain is EXACTLY zero.

    The 27B's `layers.7.post_attention_layernorm[3994]` is `blocks.7.norm2`, the very
    norm this tool taps; its `normed` dump at L07 carries a structurally dead channel
    and C17 must never read channel 3994 as a fit failure.  (`prenorm`, the default,
    is the norm's INPUT and is unaffected.)
    """

    import torch

    weight = norm.weight.detach()
    zeros = torch.nonzero(weight == 0).flatten().tolist()
    return {
        "count": len(zeros),
        "coordinates": [int(c) for c in zeros[:keep]],
        "coordinates_truncated": len(zeros) > keep,
        "hidden": int(weight.numel()),
    }


class ResidualTaps:
    """Forward hooks on every `blocks.{i}.norm2`: the FFN input, pre-norm (and normed).

    `resident=False` attaches nothing at construction (the streamed engine's blocks sit
    on `meta` until built); `attach` is then called per streamed block by
    `stream_with_taps`.  Every attach also censuses the norm2 gain (`zero_gain`).
    """

    def __init__(self, model: Any, *, normed: bool, resident: bool = True):
        self.model = model
        self.normed = bool(normed)
        self.handles: list[Any] = []
        self.rows: dict[int, list[Any]] = {}
        self.normed_rows: dict[int, list[Any]] = {}
        self.zero_gain: dict[int, dict[str, Any]] = {}
        self.select: Any = None  # [batch, seq] bool mask on the current forward, or None (all)
        if resident:
            for index, block in enumerate(model.blocks):
                self.attach(index, block)

    def attach(self, index: int, block: Any) -> None:
        self.handles.append(block.norm2.register_forward_hook(self._hook(index)))
        self.zero_gain[index] = zero_gain_coordinates(block.norm2)

    def _hook(self, index: int):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            x = inputs[0]
            if self.select is None:
                picked = x.reshape(-1, x.shape[-1])
                picked_out = output.reshape(-1, output.shape[-1])
            else:
                mask = self.select.to(device=x.device)
                picked = x[mask]
                picked_out = output[mask]
            self.rows.setdefault(index, []).append(picked.detach().to("cpu"))
            if self.normed:
                self.normed_rows.setdefault(index, []).append(picked_out.detach().to("cpu"))

        return hook

    def clear(self) -> None:
        self.rows.clear()
        self.normed_rows.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def write_taps(taps: ResidualTaps, out_dir: Path, *, meta: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    layers: list[dict[str, Any]] = []
    for index in sorted(taps.rows):
        rows = torch.cat(taps.rows[index], dim=0).contiguous()
        all_zero = int((rows == 0).all(dim=1).sum().item()) if rows.numel() else 0
        if all_zero:
            raise Refusal(
                f"tap layer {index}: {all_zero} all-zero row(s) of {int(rows.shape[0])} -- right-padding leaked "
                "into the dump (the tap selection must be the batch validity mask); nothing written"
            )
        payload = {"prenorm": rows}
        if taps.normed:
            payload["normed"] = torch.cat(taps.normed_rows[index], dim=0).contiguous()
        layer_dir = out_dir / f"L{index:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / "ffn_input_rows.safetensors"
        save_file(payload, str(path), metadata={"schema": TAP_SCHEMA, "layer": str(index)})
        layers.append({"layer": index, "rows": int(rows.shape[0]), "hidden": int(rows.shape[1]), "dtype": str(rows.dtype), "path": str(path), "sha256": file_sha256(path), "all_zero_rows": all_zero, "zero_gain_coordinates": taps.zero_gain.get(index)})
    receipt = {
        "schema": TAP_SCHEMA,
        "convention": "prenorm = input of blocks.{i}.norm2 (post-attention residual); normed = its output",
        "zero_gain_note": (
            "zero_gain_coordinates = channels whose blocks.{i}.norm2 gain is exactly 0 (run-book §3b: the 27B's "
            "layers.7.post_attention_layernorm[3994]); the `normed` dump is structurally dead there and C17 must "
            "not read such a channel as a fit failure; `prenorm` is the norm's input and is unaffected"
        ),
        **meta,
        "layers": layers,
    }
    (out_dir / "taps_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def stream_with_taps(engine: Any, taps: ResidualTaps) -> None:
    """Route `engine._build_block` so every streamed block is hooked (and censused) as it is built."""

    build = engine._build_block

    def build_with_taps(index: int, _build: Any = build, _taps: ResidualTaps = taps) -> Any:
        block = _build(index)
        _taps.attach(index, block)
        return block

    engine._build_block = build_with_taps  # type: ignore[method-assign]


def streamed_pass(engine: Any, what: str, fn: Any, *, micro_batch: int, context: dict[str, Any] | None = None) -> Any:
    """Run one streamed pass; a CUDA OOM becomes a Refusal with a receipt, never a bare crash.

    The resident path records `out-of-memory` per row; the streamed engine has no per-row
    recovery (one OOM in a gated_gqa block after the KDA recurrence has run for hours
    used to exit 1 with nothing written).
    """

    torch = engine.torch
    try:
        return fn()
    except torch.OutOfMemoryError as exc:
        peak = None
        if torch.cuda.is_available():
            try:
                peak = int(torch.cuda.max_memory_allocated())
            except RuntimeError:
                peak = None
            torch.cuda.empty_cache()
        receipt = {
            "schema": REFUSAL_SCHEMA,
            "reason": "out-of-memory",
            "pass": what,
            "micro_batch": int(micro_batch),
            "device": str(getattr(engine, "device", None)),
            "dtype": str(getattr(engine, "dtype", None)),
            "peak_cuda_memory_allocated_bytes": peak,
            "error": f"{type(exc).__name__}: {exc}",
            **(context or {}),
        }
        raise Refusal(
            f"OUT OF MEMORY in the streamed {what} at --micro-batch {micro_batch} (peak allocated "
            f"{peak if peak is None else peak >> 20} MiB): {exc}; nothing sealed -- rerun with --micro-batch 1 "
            "(and --held-only for 4,096-token evidence rows)",
            receipt=receipt,
        ) from exc


def resolve_micro_batch(args: argparse.Namespace) -> int:
    """`--micro-batch`, else 1 for --input evidence rows (4,096 tokens), 8 for bank rows."""

    if getattr(args, "micro_batch", None) is not None:
        value = int(args.micro_batch)
        if value < 1:
            raise Refusal(f"--micro-batch must be >= 1, got {value}")
        return value
    return DEFAULT_MICRO_BATCH_EVIDENCE if getattr(args, "input", None) else DEFAULT_MICRO_BATCH_BANK


def cuda_inventory(devices: list[str]) -> list[dict[str, Any]]:
    """Name, capability and free/total memory of every CUDA device the run will touch."""

    import torch

    rows: list[dict[str, Any]] = []
    for name in devices:
        if not str(name).startswith("cuda"):
            rows.append({"device": str(name), "name": None, "used_mib": None})
            continue
        index = torch.device(name).index or 0
        free, total = torch.cuda.mem_get_info(index)
        major, minor = torch.cuda.get_device_capability(index)
        rows.append(
            {
                "device": str(name),
                "name": torch.cuda.get_device_name(index),
                "capability": f"sm{major}{minor}",
                "total_bytes": int(total),
                "free_bytes": int(free),
                "used_mib": int((int(total) - int(free)) // (1024 * 1024)),
            }
        )
    return rows


def check_gpu_free(inventory: list[dict[str, Any]], *, limit_mib: int = GPU_BUSY_LIMIT_MIB) -> None:
    """Standing rule: never start on a card above 2,000 MiB used."""

    busy = [row for row in inventory if row.get("used_mib") is not None and int(row["used_mib"]) > int(limit_mib)]
    if busy:
        named = ", ".join(f"{row['device']} ({row.get('name')}) {row['used_mib']} MiB used" for row in busy)
        raise Refusal(f"standing rule: never start on a card above {limit_mib} MiB used -- {named}")


# --------------------------------------------------------------------------
# The streamed teacher-forced engine (proxy mode).
# --------------------------------------------------------------------------


class StreamedChimera:
    """Blocks streamed one at a time from the shards; teacher-forced forward only.

    Same construction trick as the split loader (meta + assign) per block.  Runs a
    LIST of sequences right-padded into one batch with a validity mask, which the
    Chimera blocks honour exactly (invalid rows are zeroed, KDA updates are gated by
    validity, RoPE positions are ranks over valid tokens).
    """

    def __init__(self, root: Path, *, device: str, dtype: Any, ledger_band_logit_mask: bool, verify_tensors: bool = False):
        import torch
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.importing.safetensors_io import read_checkpoint_config, read_index, read_safetensors
        from chimera_omni.modeling import ChimeraOmniModel
        from chimera_omni.modeling.layers import GatedGQAAttention

        self.torch = torch
        self.root = Path(root)
        self.device = device
        self.dtype = dtype
        self.mask_band = bool(ledger_band_logit_mask)
        self.verify_tensors = bool(verify_tensors)
        document = read_checkpoint_config(self.root)
        self.cfg = ChimeraConfig.from_manifest(document["config"])
        refuse_charted(self.cfg)
        if self.cfg.residual_mode != "none":
            raise Refusal("the streamed engine mirrors the plain residual trunk only (residual_mode must be 'none')")
        self.document = document
        self.index = read_index(self.root) or {}
        self.tied: dict[str, str] = dict(self.index.get("metadata", {}).get("tied_weights", {}))
        self.weight_map: dict[str, str] = dict(self.index.get("weight_map", {}))
        self._read_safetensors = read_safetensors
        self._gqa_class = GatedGQAAttention
        self._reference_gqa = None
        with torch.device("meta"):
            self.model = ChimeraOmniModel(self.cfg)
        self.block_names: dict[int, list[str]] = {}
        head_names: list[str] = []
        for name in self.model.state_dict():
            index = block_index_of(name)
            if index is None:
                head_names.append(name)
            else:
                self.block_names.setdefault(index, []).append(name)
        # The head tensors are assigned into the WHOLE model, whose block tensors stay on
        # `meta` until each block is streamed in; `strict=True` would report every block
        # tensor missing, so the head assign names exactly the block set as allowed gaps.
        self._assign(
            self.model,
            head_names,
            prefix="",
            allow_missing={name for names in self.block_names.values() for name in names},
        )
        if self.cfg.tie_word_embeddings and self.model.lm_head.weight.data_ptr() != self.model.embed.weight.data_ptr():
            raise Refusal("tied checkpoint lost its shared embedding table on load")
        self.block_loads = 0

    def _owner(self, name: str) -> str:
        return self.tied.get(name, name)

    def _assign(self, module: Any, names: list[str], *, prefix: str, allow_missing: set[str] | None = None) -> None:
        """Assign `names` (checkpoint names) into `module` (whose keys drop `prefix`).

        `allow_missing` is the EXACT set of module keys that may stay unmaterialised
        (the streamed blocks, when the head is assigned into the whole model); any
        other gap, and any unexpected key, is a refusal -- the strict load's
        guarantees are kept by hand rather than dropped with `strict=False`.
        """

        torch = self.torch
        owners = sorted({self._owner(n) for n in names})
        views = self._read_safetensors(self.root, owners, verify=self.verify_tensors)
        target = module.state_dict()
        params = {key for key, _ in module.named_parameters()}
        state: dict[str, Any] = {}
        loaded: dict[str, Any] = {}
        for name in names:
            local = name[len(prefix):]
            reference = target[local]
            owner = self._owner(name)
            if owner not in loaded:
                view = views[owner]
                tensor = view.torch(device=self.device, dtype=self.dtype if reference.is_floating_point() else reference.dtype)
                loaded[owner] = torch.nn.Parameter(tensor, requires_grad=False) if local in params else tensor
            state[local] = loaded[owner]
        del views
        if allow_missing is None:
            module.load_state_dict(state, strict=True, assign=True)
        else:
            result = module.load_state_dict(state, strict=False, assign=True)
            if result.unexpected_keys:
                raise Refusal(f"unexpected tensors in the head load: {sorted(result.unexpected_keys)[:5]}")
            gaps = set(result.missing_keys) ^ set(allow_missing)
            if gaps:
                raise Refusal(f"the head load's gaps are not exactly the streamed block set: {sorted(gaps)[:5]}")
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _build_block(self, index: int) -> Any:
        torch = self.torch
        from chimera_omni.modeling.layers import ChimeraBlock

        with torch.device("meta"):
            block = ChimeraBlock(self.cfg, index)
        self._assign(block, self.block_names[index], prefix=f"blocks.{index}.")
        if isinstance(block.mixer, self._gqa_class) and block.mixer.inv_freq is not None:
            if self._reference_gqa is None:
                self._reference_gqa = self._gqa_class(self.cfg)
            block.mixer.inv_freq = rope_inv_freq(self._reference_gqa, self.device)
        leftover = [n for n, t in list(block.named_parameters()) + list(block.named_buffers()) if t.is_meta]
        if leftover:
            raise Refusal(f"block {index}: tensors never materialised: {leftover[:3]}")
        block.eval()
        self.block_loads += 1
        return block

    def run(self, sequences: list[list[int]], *, taps: Any = None, tap_select: Any = None, micro_batch: int = DEFAULT_MICRO_BATCH_BANK, media: list[dict[str, Any] | None] | None = None) -> dict[str, Any]:
        """Teacher-forced forward over every sequence; returns per-sequence logits (fp32, cpu) and terminal hidden.

        `micro_batch` rows share one padded pass; with `tap_select` None the taps take
        the batch VALIDITY mask (never a right-padded position).
        """

        torch = self.torch
        model = self.model
        if int(micro_batch) < 1:
            raise Refusal(f"micro_batch must be >= 1, got {micro_batch}")
        micro_batch = int(micro_batch)
        if not sequences:
            raise Refusal("nothing to score")
        lengths = [len(s) for s in sequences]
        if min(lengths) <= 0:
            raise Refusal("an empty sequence cannot be scored")
        order = sorted(range(len(sequences)), key=lambda i: -lengths[i])
        batches: list[list[int]] = [order[i : i + micro_batch] for i in range(0, len(order), micro_batch)]
        residuals: list[Any] = []
        valids: list[Any] = []
        with torch.no_grad():
            for batch in batches:
                width = max(lengths[i] for i in batch)
                ids = torch.zeros((len(batch), width), dtype=torch.long)
                valid = torch.zeros((len(batch), width), dtype=torch.bool)
                for row, i in enumerate(batch):
                    ids[row, : lengths[i]] = torch.tensor(sequences[i], dtype=torch.long)
                    valid[row, : lengths[i]] = True
                ids = ids.to(self.device)
                valid = valid.to(self.device)
                model._validate_token_ids(ids)
                x = model.embed(ids)
                if media is not None:
                    for row, i in enumerate(batch):
                        m = media[i]
                        if m is None:
                            continue
                        x_row, _ = model._add_native_media(
                            x[row : row + 1], m["features"].to(self.device), m["type_ids"].to(self.device),
                            m["positions"].to(self.device), m["indices"].to(self.device), m["valid_mask"].to(self.device), None,
                        )
                        x = torch.cat((x[:row], x_row, x[row + 1 :]), dim=0)
                x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
                residuals.append(x)
                valids.append(valid)
            tap_masks = [valids[b] if tap_select is None else tap_select(batch, valids[b]) for b, batch in enumerate(batches)]
            for index in range(int(self.cfg.num_layers)):
                block = self._build_block(index)
                for b, x in enumerate(residuals):
                    if taps is not None:
                        taps.select = tap_masks[b]
                    x = block(x, token_valid_mask=valids[b])
                    residuals[b] = torch.where(valids[b].unsqueeze(-1), x, torch.zeros_like(x))
                del block
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logits_out: list[Any] = [None] * len(sequences)
            hidden_out: list[Any] = [None] * len(sequences)
            for b, x in enumerate(residuals):
                hidden = model.norm(x)
                hidden = torch.where(valids[b].unsqueeze(-1), hidden, torch.zeros_like(hidden))
                for row, i in enumerate(batches[b]):
                    h = hidden[row : row + 1, : lengths[i]]
                    logits = model.lm_head(h)
                    if self.mask_band:
                        logits = model._mask_ledger_band_logits(logits)
                    logits_out[i] = logits[0].float().cpu()
                    hidden_out[i] = h[0, -1].float().cpu()
        return {"logits": logits_out, "terminal_hidden": hidden_out, "block_loads": self.block_loads, "micro_batch": micro_batch, "batches": len(batches)}


def reference_response_ids(reference_record: dict[str, Any], tokenizer: Any) -> tuple[list[int], str]:
    ids = reference_record.get("generated_token_ids")
    if isinstance(ids, list) and ids:
        return [int(x) for x in ids], "reference dossier token ids (exact)"
    text = str(reference_record.get("response_verbatim", ""))
    return [int(x) for x in tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]], "re-tokenised reference response_verbatim (BPE re-merge may differ from the original ids)"


def teacher_forced_proxy(
    engine: StreamedChimera,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    oc_calib: Any,
    reference: dict[str, Any],
    thinking: str,
    stops: set[int],
    label: str,
    taps: Any = None,
    micro_batch: int = DEFAULT_MICRO_BATCH_BANK,
) -> list[dict[str, Any]]:
    torch = engine.torch
    by_id = {str(r["id"]): r for r in reference.get("child_responses", [])}
    sequences: list[list[int]] = []
    prompts: list[dict[str, Any]] = []
    refs: list[tuple[list[int], str]] = []
    for row in rows:
        if str(row["id"]) not in by_id:
            raise Refusal(f"--reference-dossier has no response for case {row['id']!r}")
        messages = oc_calib.messages_for(row)
        recorded = by_id[str(row["id"])].get("prompt_messages")
        if recorded is not None and recorded != messages:
            raise Refusal(
                f"--reference-dossier case {row['id']!r} was generated on different prompt_messages than this "
                "bank row renders; teacher forcing against a different prompt measures nothing"
            )
        prompt = build_prompt_ids(tokenizer, messages, thinking=thinking, oc_calib=oc_calib)
        ref_ids, how = reference_response_ids(by_id[str(row["id"])], tokenizer)
        prompts.append(prompt)
        refs.append((ref_ids, how))
        sequences.append(prompt["ids"] + ref_ids)
    started = time.monotonic()
    out = streamed_pass(
        engine,
        "teacher-forced bank pass",
        lambda: engine.run(sequences, taps=taps, micro_batch=micro_batch),
        micro_batch=micro_batch,
        context={"rows": len(sequences), "longest_sequence": max(len(s) for s in sequences)},
    )
    wall = time.monotonic() - started
    records: list[dict[str, Any]] = []
    for row, prompt, (ref_ids, how), logits in zip(rows, prompts, refs, out["logits"]):
        p = len(prompt["ids"])
        n = len(ref_ids)
        # position p-1 predicts ref[0], ..., p+n-2 predicts ref[n-1]; p+n-1 predicts the stop
        predict = logits[p - 1 : p - 1 + n] if n else logits[p - 1 : p - 1]
        argmax = [int(x) for x in predict.argmax(dim=-1).tolist()] if n else []
        agree = sum(1 for a, b in zip(argmax, ref_ids) if a == b)
        nll = None
        if n:
            logprob = torch.log_softmax(predict, dim=-1)
            picked = logprob[torch.arange(n), torch.tensor(ref_ids)]
            nll = float((-picked).mean().item())
        after = int(logits[p - 1 + n].argmax().item())
        decoded = tokenizer.decode(argmax, skip_special_tokens=False)
        ref_record = by_id[str(row["id"])]
        records.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "prompt_messages": oc_calib.messages_for(row),
                "prompt_tokens": p,
                "response_verbatim": decoded,
                "response_sha256": oc_calib.canonical_sha256({"text": decoded}),
                "logits_to_keep_used": False,
                "generated_tokens": n,
                "generated_token_ids_sha256": oc_calib.canonical_sha256({"ids": argmax}),
                "loop_detected": oc_calib.detect_loop(argmax),
                "termination": MODE_PROXY,
                "runtime_error": None,
                "wall_time_seconds": round(wall / max(1, len(rows)), 3),
                "wall_time_seconds_is": "mean of the streamed batch wall over the rows scored together (all rows share one pass)",
                "generated_token_ids": argmax,
                "teacher_forced": {
                    "batch_wall_time_seconds": round(wall, 3),
                    "batch_rows": len(rows),
                    "micro_batch": int(micro_batch),
                    "reference_response_sha256": ref_record.get("response_sha256"),
                    "reference_termination": ref_record.get("termination"),
                    "reference_ids_source": how,
                    "reference_tokens": n,
                    "argmax_agreement": agree,
                    "argmax_agreement_rate": (agree / n) if n else None,
                    "mean_nll": nll,
                    "predicts_stop_after_reference": after in stops,
                    "note": (
                        "response_verbatim is the per-position ARGMAX under teacher forcing on the "
                        "reference response, not a free-run decode; it is a proxy and the dossier's "
                        "mode says so."
                    ),
                },
            }
        )
        log(f"{label} {row['id']}: teacher-forced {n} tok, argmax agree {agree}/{n}, nll {nll}")
    return records


# --------------------------------------------------------------------------
# Vision cards: PNG -> native media tokens.
# --------------------------------------------------------------------------


def packetise_image(path: Path, *, feature_dim: int, patch: int = IMAGE_PATCH, side: int = 256) -> dict[str, Any]:
    import torch

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - server has Pillow
        raise Refusal("Pillow is required for --vision-cards") from exc
    image = Image.open(path).convert("RGB").resize((side, side), Image.BILINEAR)
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).view(side, side, 3) / 255.0
    pixels = pixels * 2.0 - 1.0
    grid = side // patch
    patches = pixels.view(grid, patch, grid, patch, 3).permute(0, 2, 1, 3, 4).reshape(grid * grid, patch * patch * 3)
    if patches.shape[1] > feature_dim:
        raise Refusal(f"a {patch}x{patch}x3 patch ({patches.shape[1]}) exceeds native_feature_dim {feature_dim}")
    features = torch.zeros((1, grid * grid, feature_dim), dtype=torch.float32)
    features[0, :, : patches.shape[1]] = patches
    centres = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    ys, xs = torch.meshgrid(centres, centres, indexing="ij")
    positions = torch.zeros((1, grid * grid, 4), dtype=torch.float32)
    positions[0, :, 0] = xs.reshape(-1)
    positions[0, :, 1] = ys.reshape(-1)
    return {
        "features": features,
        "positions": positions,
        "type_ids": torch.full((1, grid * grid), IMAGE_TYPE_ID, dtype=torch.long),
        "valid_mask": torch.ones((1, grid * grid), dtype=torch.bool),
        "recipe": {
            "resize": [side, side],
            "patch": patch,
            "tokens": grid * grid,
            "feature_layout": f"{patch}x{patch}x3 RGB in [-1,1], zero-padded to {feature_dim}",
            "positions": "patch centre (x, y) in [0,1], dims 2-3 zero",
            "type_id": IMAGE_TYPE_ID,
            "source_sha256": file_sha256(path),
        },
    }


def media_row_builder(cfg: Any, assets_dir: Path, ledger: Any):
    import torch

    band = ledger.range("media")
    placeholder = int(band.begin)

    def build(row: dict[str, Any], prompt_ids: list[int]):
        asset_ids = [
            str(part.get("asset_id"))
            for message in row.get("messages", []) or []
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image"
        ]
        if not asset_ids:
            return None, None
        if len(asset_ids) != 1:
            raise Refusal(f"case {row['id']} carries {len(asset_ids)} images; one per case is supported")
        path = assets_dir / f"{asset_ids[0]}.png"
        if not path.is_file():
            raise Refusal(f"case {row['id']}: asset {path} is missing")
        packet = packetise_image(path, feature_dim=int(cfg.native_feature_dim))
        count = int(packet["features"].shape[1])
        ids = [placeholder] * count + list(prompt_ids)
        media = {
            **packet,
            "indices": torch.arange(count, dtype=torch.long).view(1, -1),
            "input_ids": ids,
        }
        note = {
            "asset_id": asset_ids[0],
            "asset_path": str(path),
            "media_tokens": count,
            "placeholder_token_id": placeholder,
            "placeholder_band": "media",
            "placement": "media tokens precede the chat text; chat positions shift by the token count",
            **packet["recipe"],
        }
        return media, note

    return build


# --------------------------------------------------------------------------
# Media-out: band token streams.
# --------------------------------------------------------------------------


def band_token_stream(
    decoder: SplitDecoder, tokenizer: Any, rows: list[dict[str, Any]], *, oc_calib: Any, band: Any, cap: int, thinking: str, out_dir: Path, label: str
) -> list[dict[str, Any]]:
    torch = decoder.torch
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for row in rows:
            started = time.monotonic()
            prompt = build_prompt_ids(tokenizer, oc_calib.messages_for(row), thinking=thinking, oc_calib=oc_calib)
            ids = torch.tensor([prompt["ids"]], dtype=torch.long)
            generated: list[int] = []
            result = decoder.forward(ids, None)
            cache = result["cache"]
            for _ in range(cap):
                # the band's rows only: the argmax is taken over [begin, end) regardless of the text mask
                raw = decoder.model.lm_head(result["hidden"][:, -1:])[0, -1]
                local = int(raw[band.begin : band.end].argmax().item())
                token = int(band.begin) + local
                generated.append(token)
                result = decoder.forward(torch.tensor([[token]], dtype=torch.long), cache)
                cache = result["cache"]
            result = None
            cache = None
            local_ids = [t - int(band.begin) for t in generated]
            distinct = len(set(local_ids))
            counts: dict[int, int] = {}
            for t in local_ids:
                counts[t] = counts.get(t, 0) + 1
            entropy = -sum((c / len(local_ids)) * math.log(c / len(local_ids)) for c in counts.values()) if local_ids else 0.0
            path = out_dir / f"{row['id']}.{band.name}.json"
            payload = {
                "schema": MEDIA_STREAM_SCHEMA,
                "case_id": str(row["id"]),
                "band": band.name,
                "band_begin": int(band.begin),
                "band_end": int(band.end),
                "prompt_tokens": len(prompt["ids"]),
                "token_ids": generated,
                "band_local_ids": local_ids,
            }
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            record = {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "band": band.name,
                "tokens": len(generated),
                "non_empty": len(generated) > 0,
                "distinct_codes": distinct,
                "code_entropy_nats": round(entropy, 6),
                "loop_detected": oc_calib.detect_loop(generated),
                "stream_path": str(path),
                "stream_sha256": file_sha256(path),
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "note": "a token stream in an untrained band is a mechanism check, not a music result; Cyril grades decoded media, and nothing decodes here",
            }
            records.append(record)
            log(f"{label} media-out {row['id']}: {len(generated)} {band.name} tokens, {distinct} distinct")
    return records


# --------------------------------------------------------------------------
# Evidence manifest rows (for --tap-residuals in stream mode).
# --------------------------------------------------------------------------


def evidence_sequences(manifest_path: Path, *, held_only: bool) -> tuple[list[list[int]], dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Refusal(f"{manifest_path} has no rows")
    chosen = [r for r in sorted(rows, key=lambda r: int(r["batch_index"])) if (not held_only or str(r["split"]) == "held")]
    if not chosen:
        raise Refusal("no rows selected from the evidence manifest")
    return [[int(x) for x in r["ids"]] for r in chosen], {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(Path(manifest_path)),
        "tokenizer_sha256": payload.get("tokenizer_sha256"),
        "rows": [int(r["batch_index"]) for r in chosen],
        "splits": sorted({str(r["split"]) for r in chosen}),
        "held_only": bool(held_only),
    }


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")


def tokenizer_files_digest(directory: Path, oc_calib: Any) -> str:
    """Digest over the tokenizer files only -- the HF parent dir also holds 55 GB of weights."""

    rows = [
        {"path": name, "sha256": file_sha256(directory / name), "bytes": (directory / name).stat().st_size}
        for name in TOKENIZER_FILES
        if (directory / name).is_file()
    ]
    return oc_calib.canonical_sha256({"schema": "chimera-omni-tokenizer-files-digest-v1", "files": rows})


def kernel_report(cfg: Any) -> dict[str, Any]:
    kinds = sorted({cfg.block_kind(i) for i in range(int(cfg.num_layers))})
    return {
        "kernel_report_schema": KERNEL_REPORT_SCHEMA,
        "kernel_path": "chimera-reference (per-position fp32 KDA recurrence + torch SDPA-free attention)",
        "block_kinds": kinds,
        "note": "the Chimera reference modules bind no fla/causal_conv1d kernel; recorded so the validity tuple names the path that ran",
    }


def reference_dossier_checks(reference: dict[str, Any], *, path: Path, bank_digest: str, surface: str, subset: bool, thinking: str) -> dict[str, Any]:
    """Refuse a reference taken on another bank, another prompt surface or another thinking setting.

    Bank digest alone is not enough: a reference rendered template-default and a run with
    --thinking off teacher-force different prompts, and the agreement numbers mean nothing.
    The surface digest must match when the run scores the whole selection; a --lanes /
    --max-cases subset renders a different surface by construction, so there the check is
    per case (`teacher_forced_proxy` compares each record's prompt_messages) and the
    subset is recorded.  An HF dossier written before the runner had --thinking carries no
    thinking block; only template-default can be teacher-forced against it, and the
    assumption is written down.
    """

    bank = reference.get("bank", {}) if isinstance(reference.get("bank"), dict) else {}
    generation = reference.get("generation", {}) if isinstance(reference.get("generation"), dict) else {}
    block = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "dossier_sha256": reference.get("dossier_sha256"),
        "label": reference.get("label"),
        "mode": reference.get("mode", MODE_FREE_RUN),
        "bank_file_sha256": bank.get("file_sha256"),
        "dossier_prompt_surface_sha256": bank.get("dossier_prompt_surface_sha256"),
        "this_run_prompt_surface_sha256": surface,
        "surface_check": None,
        "thinking_requested": None,
        "thinking_check": None,
        "dtype": generation.get("dtype"),
        "compute_architecture": generation.get("compute_architecture"),
        "runtime": generation.get("runtime"),
    }
    if block["bank_file_sha256"] != bank_digest:
        raise Refusal(f"--reference-dossier was taken on bank {block['bank_file_sha256']}, this run reads {bank_digest}")
    if block["dossier_prompt_surface_sha256"] == surface:
        block["surface_check"] = "bank.dossier_prompt_surface_sha256 == this run's surface"
    elif subset:
        block["surface_check"] = "SUBSET (--lanes/--max-cases): surface digests differ by construction; every case's prompt_messages is compared to the reference record"
    else:
        raise Refusal(
            f"--reference-dossier renders prompt surface {str(block['dossier_prompt_surface_sha256'])[:16]}, this run renders "
            f"{surface[:16]}; teacher forcing against a different prompt surface measures nothing"
        )
    recorded = generation.get("thinking")
    requested = recorded.get("requested") if isinstance(recorded, dict) else None
    if requested is None:
        if thinking != "template-default":
            raise Refusal(
                f"--reference-dossier records no chat-template thinking setting (an HF dossier before --thinking, i.e. the "
                f"template default); this run asks --thinking {thinking}: pass --thinking template-default or re-take the reference"
            )
        block["thinking_requested"] = None
        block["thinking_check"] = "reference records no thinking setting; template-default assumed (the only setting the HF runner had) and this run is template-default"
    elif str(requested) != thinking:
        raise Refusal(f"--reference-dossier was rendered with thinking {requested!r}, this run asks --thinking {thinking!r}; the prompts differ")
    else:
        block["thinking_requested"] = str(requested)
        block["thinking_check"] = "generation.thinking.requested == this run's --thinking"
    return block


def refuse(message: str) -> int:
    print(f"REFUSED: {message}", file=sys.stderr, flush=True)
    return EXIT_REFUSED


def visible_devices(device: str) -> list[str]:
    import torch

    if not str(device).startswith("cuda"):
        return [str(device)]
    count = torch.cuda.device_count()
    if count == 0:
        raise Refusal("CUDA requested but no device is visible")
    return [f"cuda:{i}" for i in range(count)]


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # ---- mode refusals first, before anything is loaded -------------------
    stream = bool(args.stream)
    if stream and not args.teacher_forced:
        raise Refusal("--stream is teacher-forced proxy scoring only; pass --teacher-forced (free-run generation is not offered on the streamed engine)")
    if args.teacher_forced and not stream:
        raise Refusal("--teacher-forced is the stream mode's scoring; pass --stream")
    if stream and int(args.maximum_new_tokens) > 0:
        raise Refusal(
            f"--stream refuses generation: --maximum-new-tokens is {args.maximum_new_tokens}, it must be 0 "
            "in stream mode (free-run on the streamed engine costs ~0.69 s x 64 block loads per token)"
        )
    if stream and args.split:
        raise Refusal("--split is the free-run placement; the streamed engine runs on --device alone")
    if not stream and int(args.maximum_new_tokens) <= 0:
        raise Refusal("free-run needs --maximum-new-tokens > 0")
    if stream and not (args.reference_dossier or args.input):
        raise Refusal("stream mode needs --reference-dossier (bank rows scored teacher-forced) and/or --input <evidence manifest> (--tap-residuals rows)")
    if args.media_out:
        kinds = [k.strip() for k in str(args.media_out).split(",") if k.strip()]
        unsupported = [k for k in kinds if k not in ("music",)]
        if unsupported:
            raise Refusal(f"--media-out {unsupported}: continuous samplers (image/video/volume x0 diagnostics) need the S6 companions and are not built in this runner; 'music' (the music_codes band token stream) is")
        if stream:
            raise Refusal("--media-out decodes and is refused in stream mode")
    if args.vision_cards and stream:
        raise Refusal("--vision-cards decodes and is refused in stream mode")
    if args.tap_residuals and not args.tap_out:
        raise Refusal("--tap-residuals needs --tap-out <dir>")
    if args.input and not (args.tap_residuals and args.tap_out):
        raise Refusal("--input <evidence manifest> is the --tap-residuals row source; pass --tap-residuals --tap-out <dir> with it")
    micro_batch = resolve_micro_batch(args)

    tree = bind_chimera_tree(args.chimera_src, args.chimera_commit)
    oc_calib, ocr = bind_calibration(args.calibration_dir)
    import torch

    torch.set_grad_enabled(False)
    started = time.monotonic()
    device = str(args.device)
    architecture = oc_calib.compute_architecture(device)
    devices = visible_devices(device)
    split = parse_split(args.split)
    log(f"device {device} -> {architecture}; visible {devices}; split {split}; micro-batch {micro_batch}")
    used_devices = [device] if stream else list(devices[: len(split) if split else 1])
    cuda_placement = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "devices_used": used_devices,
        "inventory_at_start": cuda_inventory(used_devices),
        "busy_limit_mib": GPU_BUSY_LIMIT_MIB,
    }
    check_gpu_free(cuda_placement["inventory_at_start"])
    for row in cuda_placement["inventory_at_start"]:
        if row.get("name"):
            log(f"{row['device']}: {row['name']} ({row['capability']}), {row['used_mib']} MiB used of {row['total_bytes'] >> 20}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dtype = oc_calib.resolve_dtype(str(args.dtype))
    if stream and not device.startswith("cuda") and dtype != torch.float32:
        raise Refusal("bf16 on the CPU is ~400x slower than fp32; --stream on cpu needs --dtype float32")

    # ---- bank ---------------------------------------------------------------
    bank_path = Path(args.bank)
    expected = args.bank_sha256
    if expected is None and bank_path.name.startswith("lfm2_semantic_whole_response"):
        expected = ocr.FROZEN_V1_PACK_SHA256
    rows, bank_payload, bank_digest = oc_calib.load_bank(bank_path, expected_file_sha256=expected)
    media_rows: list[dict[str, Any]] = []
    if args.vision_cards:
        rows = [r for r in rows if str(r.get("modality_in", "text")) == "text+image"]
        if not rows:
            raise Refusal("--vision-cards: the bank has no text+image cases")
        rows, selection = ocr.select_cases(rows, lanes=args.lanes, text_only=False, max_cases=args.max_cases)
        for row in rows:
            if row.get("runnable", True) is False:
                raise Refusal(f"vision case {row['id']} is runnable:false")
    else:
        rows, selection = ocr.select_cases(rows, lanes=args.lanes, text_only=bool(args.text_only), max_cases=args.max_cases)
        if args.media_out:
            # `materialise_rows` refuses a runnable:false case by design (a text decoder
            # must not file prose as a clip); the band token stream IS this runner's
            # answer for those cases, so they are set aside BEFORE the text rows are
            # materialised and never reach the text decode.
            media_rows = [r for r in rows if r.get("runnable", True) is False or r.get("media_out")]
            rows = [r for r in rows if r not in media_rows]
            if not media_rows:
                raise Refusal("--media-out: the selection has no media-out (runnable:false) cases; use --bank banks/omni_review_bank_v2_draft.json")
        rows = ocr.materialise_rows(rows) if rows else []
    surface = oc_calib.prompt_surface_sha256(rows + media_rows)
    log(f"bank {bank_path.name}: {len(rows)} text case(s) + {len(media_rows)} media-out case(s), file {bank_digest[:16]}")

    # ---- tokenizer (the HF one, for the chat template) -----------------------
    model_dir = Path(args.model).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve() if args.tokenizer_dir else model_dir
    if not (tokenizer_dir / "tokenizer_config.json").is_file():
        raise Refusal(f"{tokenizer_dir} carries no tokenizer_config.json; a Chimera checkpoint needs --tokenizer-dir <HF parent dir> for the chat template")
    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=False)
    tokenizer_digest = tokenizer_files_digest(tokenizer_dir, oc_calib)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    extra_ids, extra_names = ocr.stop_token_ids(tokenizer, [t for t in str(args.extra_eos_tokens).split(",") if t])
    stops = set(extra_ids) | ({int(eos_id)} if eos_id is not None else set())

    # ---- identity + load -------------------------------------------------------
    digest, files = oc_calib.directory_digest(model_dir)
    provenance = {"model_directory": str(model_dir), "directory_sha256": digest, "files": files, "file_count": len(files)}
    receipt_path = model_dir / "import_receipt.json"
    if receipt_path.is_file():
        try:
            import_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            provenance["import_receipt"] = {"path": str(receipt_path), "sha256": file_sha256(receipt_path), "state_sha256": import_receipt.get("state_sha256") or import_receipt.get("checkpoint", {}).get("state_sha256")}
        except (OSError, ValueError):
            provenance["import_receipt"] = {"path": str(receipt_path), "unreadable": True}
    log(f"directory digest {digest[:16]} over {len(files)} file(s)")

    load_started = time.monotonic()
    taps = None
    if stream:
        engine = StreamedChimera(model_dir, device=device, dtype=dtype, ledger_band_logit_mask=bool(args.ledger_band_logit_mask), verify_tensors=bool(args.verify_tensors))
        cfg = engine.cfg
        model = engine.model
        placement = {"split": None, "devices": [device], "block_device": None, "head_device": device, "note": "streamed: one block resident at a time", "micro_batch": micro_batch}
        load_receipt = {"directory": str(model_dir), "schema": engine.document.get("schema"), "chimera_commit": engine.document.get("chimera_commit"), "source": engine.document.get("source"), "streamed": True}
    else:
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.importing.safetensors_io import read_checkpoint_config

        cfg = ChimeraConfig.from_manifest(read_checkpoint_config(model_dir)["config"])
        placement = placement_for(cfg, split, devices)
        model, cfg, load_receipt = load_chimera_split(model_dir, placement=placement, dtype=dtype, verify_tensors=bool(args.verify_tensors))
        if args.tap_residuals:
            taps = ResidualTaps(model, normed=bool(args.tap_normed))
    placement["cuda"] = cuda_placement
    load_wall = round(time.monotonic() - load_started, 3)
    #: `numel` is defined on meta tensors too, so the streamed model (blocks never
    #: all resident) reports the same live count as the resident one.
    live = int(sum(p.numel() for p in model.parameters()))
    base_vocab = int(cfg.ledger_band_begin())
    if int(len(tokenizer)) > base_vocab:
        raise Refusal(f"the tokenizer has {len(tokenizer)} ids but the checkpoint's text vocabulary is {base_vocab}")
    for stop in stops:
        if stop >= base_vocab:
            raise Refusal(f"stop token id {stop} sits inside the ledger bands")
    ledger = cfg.ledger()
    export = {
        "loader": "chimera_omni.importing.safetensors_io (meta construction + assign, per-block placement)",
        "loaded_class": "ChimeraOmniModel",
        "declared_class": str(load_receipt.get("schema")),
        "trust_remote_code": False,
        "parameters_counted_live": live,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "linear_attention_kernels": kernel_report(cfg),
        "load_wall_time_seconds": load_wall,
        "peak_cuda_memory_allocated_bytes": None,
        "chimera_tree": tree,
        "checkpoint": load_receipt,
        "config": {
            "vocab_size": int(cfg.vocab_size), "ledger_base_vocab_size": base_vocab, "ledger_bands": [[n, w] for n, w in cfg.ledger_bands],
            "hidden_size": int(cfg.hidden_size), "num_layers": int(cfg.num_layers), "block_pattern": list(cfg.block_pattern),
            "residual_mode": cfg.residual_mode, "tie_word_embeddings": bool(cfg.tie_word_embeddings), "native_feature_dim": int(cfg.native_feature_dim),
            "metadata": dict(cfg.metadata),
        },
        "placement": placement,
        "interface_note": (
            "on an imported checkpoint the native-media bridge/head and the ledger-band rows are "
            "random-init or zero; anything measured through them is a mechanism check"
        ),
    }
    log(f"loaded: {live:,} parameters in {load_wall:.1f}s ({'streamed' if stream else 'resident, split ' + str(placement['split'])})")

    first_row = (rows or media_rows)[0]
    thinking = oc_calib.thinking_report(tokenizer, oc_calib.messages_for(first_row) if not args.vision_cards else [{"role": "user", "content": "probe"}], str(args.thinking))
    log(f"chat template thinking: requested={thinking['requested']} took_effect={thinking['took_effect']}")

    probe_payload = json.loads(Path(args.probe_prompts).read_text(encoding="utf-8"))
    probe_digest = file_sha256(Path(args.probe_prompts))
    smoke_payload = None
    smoke_digest = None
    if not args.skip_smoke:
        smoke_payload = json.loads(Path(args.smoke_prompts).read_text(encoding="utf-8"))
        smoke_digest = file_sha256(Path(args.smoke_prompts))

    mode = MODE_PROXY if stream else MODE_FREE_RUN
    responses: list[dict[str, Any]] = []
    smoke: list[dict[str, Any]] = []
    media_records: list[dict[str, Any]] = []
    tap_receipt = None
    preflight: dict[str, Any]
    floor: dict[str, Any]
    reference_block = None
    tap_meta: dict[str, Any] = {"model_directory": str(model_dir), "directory_sha256": digest, "mode": mode, "hidden_size": int(cfg.hidden_size), "micro_batch": micro_batch if stream else None, "dtype": str(args.dtype)}

    if stream:
        # ---- proxy: floor probes + bank rows (+ evidence rows) in ONE streamed pass each
        preflight = {"case_id": str(rows[0]["id"]), "agree": None, "not_run": "teacher-forced proxy mode decodes nothing", "why_not_run": "there is no decode path to check in stream mode; the free-run dossier carries the preflight"}
        probe_seqs = [[int(x) for x in tokenizer(p, return_tensors=None)["input_ids"]] for p in probe_payload["prompts"]]
        probe_out = streamed_pass(engine, "emit-zero floor", lambda: engine.run(probe_seqs, micro_batch=micro_batch), micro_batch=micro_batch)
        floor_rows = []
        for index, (prompt, logits, hidden) in enumerate(zip(probe_payload["prompts"], probe_out["logits"], probe_out["terminal_hidden"]), start=1):
            last = logits[-1]
            finite = last[torch.isfinite(last)]
            floor_rows.append({"probe_index": index, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "terminal_l2": float(hidden.norm().item()), "terminal_finite": bool(torch.isfinite(hidden).all().item()), "logit_std": float(finite.std().item()) if finite.numel() > 1 else 0.0, "logit_finite": bool(torch.isfinite(finite).all().item()) and finite.numel() > 0, "argmax_token_id": int(last.argmax().item())})
        floor = {"schema": oc_calib.EMIT_ZERO_FLOOR_SCHEMA, "probes": floor_rows, "probe_count": len(floor_rows), "distinct_argmax_tokens": len({r["argmax_token_id"] for r in floor_rows}), "min_terminal_l2": min(r["terminal_l2"] for r in floor_rows), "min_logit_std": min(r["logit_std"] for r in floor_rows), "reference_terminal_relative_l2": None, "terminal_states_sha256": oc_calib.canonical_sha256({"rows": [round(r["terminal_l2"], 6) for r in floor_rows]}), "passed": bool(all(r["terminal_finite"] and r["logit_finite"] for r in floor_rows) and min(r["terminal_l2"] for r in floor_rows) > 0.0 and min(r["logit_std"] for r in floor_rows) > 0.0), "definition": "streamed teacher-forced pass; same definition as the resident floor"}
        log(f"emit-zero floor: passed={floor['passed']} min|h|={floor['min_terminal_l2']:.4f} min logit sd={floor['min_logit_std']:.4f}")
        if args.reference_dossier:
            reference = json.loads(Path(args.reference_dossier).read_text(encoding="utf-8"))
            reference_block = reference_dossier_checks(reference, path=Path(args.reference_dossier), bank_digest=bank_digest, surface=surface, subset=bool(args.lanes or args.max_cases), thinking=str(args.thinking))
            if args.tap_residuals and not args.input:
                taps = ResidualTaps(model, normed=bool(args.tap_normed), resident=False)
                stream_with_taps(engine, taps)  # the hooks live on the streamed blocks, attached per block build
            responses = teacher_forced_proxy(engine, tokenizer, rows, oc_calib=oc_calib, reference=reference, thinking=str(args.thinking), stops=stops, label=str(args.label), taps=taps, micro_batch=micro_batch)
            if taps is not None and not args.input:
                tap_receipt = write_taps(taps, Path(args.tap_out), meta={**tap_meta, "rows_from": "bank rows under teacher forcing (prompt + reference response), all positions"})
        if args.input:
            sequences, evidence_meta = evidence_sequences(Path(args.input), held_only=bool(args.held_only))
            taps = ResidualTaps(model, normed=bool(args.tap_normed), resident=False)
            stream_with_taps(engine, taps)
            max_rows = int(args.tap_max_rows)
            total = sum(len(s) for s in sequences)
            stride = max(1, total // max_rows) if max_rows > 0 else 1
            counter = {"n": 0}

            def select(batch: list[int], valid: Any):
                mask = torch.zeros_like(valid)
                flat = torch.nonzero(valid.reshape(-1)).flatten()
                for k in flat.tolist():
                    if counter["n"] % stride == 0:
                        mask.reshape(-1)[k] = True
                    counter["n"] += 1
                return mask

            counter["n"] = 0
            evidence_out = streamed_pass(
                engine,
                "evidence tap pass",
                lambda: engine.run(sequences, taps=taps, tap_select=select if max_rows > 0 else None, micro_batch=micro_batch),
                micro_batch=micro_batch,
                context={"rows": len(sequences), "longest_sequence": max(len(s) for s in sequences), "evidence": evidence_meta},
            )
            tap_receipt = write_taps(taps, Path(args.tap_out), meta={**tap_meta, "rows_from": "evidence manifest rows", "evidence": evidence_meta, "stride": stride, "tap_max_rows": max_rows, "tap_selection": "every stride-th VALID position (padding never tapped)" if max_rows > 0 else "every valid position (padding never tapped)"})
            del evidence_out
        if not args.skip_smoke:
            smoke = []  # nothing decodes in stream mode
    else:
        decoder = SplitDecoder(model, cfg, placement, ledger_band_logit_mask=bool(args.ledger_band_logit_mask))
        media_builder = None
        if args.vision_cards:
            media_builder = media_row_builder(cfg, Path(args.assets_dir) if args.assets_dir else bank_path.parent / "assets", ledger)
        if taps is not None:
            taps.select = None
        try:
            if args.vision_cards:
                preflight = {"case_id": str(first_row["id"]), "agree": None, "not_run": "vision-card run", "why_not_run": "the preflight is taken on the text bank; the free-run text dossier of the same checkpoint carries it"}
            else:
                preflight = decode_path_preflight_chimera(decoder, tokenizer, first_row, oc_calib=oc_calib, stops=stops, thinking=str(args.thinking))
        except RuntimeError as exc:
            preflight = {"case_id": str(first_row["id"]), "agree": None, "not_run": f"{type(exc).__name__}: {exc}", "why_not_run": "the preflight recomputes the full prefix eight times; it could not complete here and never gated anything"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        log(f"decode-path preflight: agree={preflight['agree']}")
        if taps is not None:
            taps.clear()
        floor = emit_zero_floor_chimera(decoder, tokenizer, probe_payload["prompts"], oc_calib=oc_calib)
        log(f"emit-zero floor: passed={floor['passed']} min|h|={floor['min_terminal_l2']:.4f} min logit sd={floor['min_logit_std']:.4f}")
        if taps is not None:
            taps.clear()
        if args.media_out:
            if "music_codes" not in {name for name, _ in cfg.ledger_bands}:
                raise Refusal("--media-out music needs a checkpoint that declares the music_codes band")
            media_records = band_token_stream(decoder, tokenizer, media_rows, oc_calib=oc_calib, band=ledger.range("music_codes"), cap=int(args.media_tokens), thinking=str(args.thinking), out_dir=Path(args.media_out_dir or Path(args.out).parent / "media_out"), label=str(args.label))
        if rows:
            responses = generate_greedy_chimera(decoder, tokenizer, rows, oc_calib=oc_calib, cap=int(args.maximum_new_tokens), stops=stops, label=str(args.label), thinking=str(args.thinking), media_for_row=media_builder)
        if taps is not None:
            tap_receipt = write_taps(taps, Path(args.tap_out), meta={**tap_meta, "rows_from": "bank prompts (prefill) and every generated step, in decode order"})
            taps.remove()
        if not args.skip_smoke and not args.vision_cards:
            smoke_rows = [{"id": f"smoke-{i}", "category": "decode-smoke", "prompt": p} for i, p in enumerate(smoke_payload["prompts"], start=1)]
            smoke = smoke_records(generate_greedy_chimera(decoder, tokenizer, smoke_rows, oc_calib=oc_calib, cap=int(args.smoke_maximum_new_tokens), stops=stops, label="smoke", thinking=str(args.thinking), keep_ids=False))

    body: dict[str, Any] = {
        "schema": ocr.DOSSIER_SCHEMA,
        "purpose": (
            "The verbatim review surface for one omni model under a named review bank, "
            "produced by the Chimera-native runner.  GRADING HAPPENS ELSEWHERE; this file "
            "grades nothing.  Only measured intelligence read by a human grades, and every "
            "machine column downstream of this file is a diagnostic."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": str(args.label),
        "parent_code": str(args.parent_code or args.label),
        "stage": str(args.stage),
        "campaign": "omni-2026-09",
        "is_random_init": False,
        "mode": mode,
        "runner": {"tool": TOOL_NAME, "version": TOOL_VERSION, "tool_sha256": file_sha256(Path(__file__)), "calibration_dir": str(Path(args.calibration_dir).resolve()), "oc_run_frozen_bank_sha256": file_sha256(Path(args.calibration_dir) / "oc_run_frozen_bank.py"), "oc_calib_sha256": file_sha256(Path(args.calibration_dir) / "oc_calib.py")},
        "provenance": provenance,
        "export": export,
        "bank": {
            "path": str(bank_path),
            "file_sha256": bank_digest,
            "declared_schema": bank_payload.get("schema"),
            "declared_status": bank_payload.get("status"),
            "digest_was_pinned": expected is not None,
            "cases_run": len(responses),
            "cases_in_bank": len(bank_payload.get("prompts", [])),
            "cases_skipped_unsupported_modality": selection["skipped_unsupported_modality"],
            "cases_skipped_not_runnable": selection["skipped_not_runnable"],
            "dossier_prompt_surface_sha256": surface,
            "dossier_prompt_surface_note": "this runner's own digest over the validated rows (oc_calib.prompt_surface_sha256); NOT the V5 protocol's published normalized surface 5615b9e4...",
            "frozen_v1_protocol_sha256": ocr.FROZEN_V1_PROTOCOL_SHA256,
            "frozen_v1_pack_sha256": ocr.FROZEN_V1_PACK_SHA256,
        },
        "generation": {
            "mode": mode,
            "maximum_new_tokens": int(args.maximum_new_tokens),
            "decoding": "greedy, argmax, no sampling" if not stream else "none (teacher-forced argmax under the reference response)",
            "decode_path": "manual argmax loop over ChimeraCausalCache (split trunk)" if not stream else "streamed teacher-forced forward, one block resident",
            "device": device,
            "devices": devices,
            "split": placement["split"],
            "micro_batch": micro_batch if stream else None,
            "compute_architecture": architecture,
            "dtype": str(args.dtype),
            "eos_token_id": None if eos_id is None else int(eos_id),
            "extra_stop_token_ids": extra_ids,
            "extra_stop_tokens": extra_names,
            "tokenizer_directory": str(tokenizer_dir),
            "tokenizer_directory_sha256": tokenizer_digest,
            "chat_template_applied": True,
            "add_generation_prompt": True,
            "thinking": thinking,
            "ledger_band_logit_mask": bool(args.ledger_band_logit_mask),
            "ledger_band_begin": base_vocab,
            "smoke_thinking": "the coherence smoke decodes on the SAME thinking setting as the bank",
            "cases": len(responses),
            "runtime": RUNTIME,
        },
        "decode_path_preflight": preflight,
        "emit_zero_floor": floor,
        "decode_smoke": {
            "prompts_file": str(args.smoke_prompts) if not args.skip_smoke else None,
            "prompts_file_sha256": smoke_digest,
            "prompts_file_sha256_expected": ocr.SMOKE_PROMPTS_SHA256,
            "digest_matches_sealed": smoke_digest == ocr.SMOKE_PROMPTS_SHA256,
            "maximum_new_tokens": int(args.smoke_maximum_new_tokens),
            "records": smoke,
            "verdict": "pending -- a human reads these three and says coherent or incoherent" if smoke else "not decoded in this mode",
        },
        "probe_prompts": {
            "prompts_file": str(args.probe_prompts),
            "prompts_file_sha256": probe_digest,
            "prompts_file_sha256_expected": ocr.PROBE_PROMPTS_SHA256,
            "digest_matches_sealed": probe_digest == ocr.PROBE_PROMPTS_SHA256,
            "count": len(probe_payload["prompts"]),
        },
        "child_responses": responses,
        "parent_responses": [],
        "parent_control_note": (
            "This runner writes one model's surface.  A parent measured through this SAME "
            "runner (the PROBE.S0 27B-U import) is the side-by-side; the HF runner's dossier "
            "of the same weights is the cross-runtime check, not the control."
        ),
        "teacher_forced_reference": reference_block,
        "residual_taps": tap_receipt,
        "media_out": {"records": media_records, "band": "music_codes" if media_records else None, "note": "band token streams; no decoder ran"} if args.media_out else None,
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    if torch.cuda.is_available():
        try:
            export["peak_cuda_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        except RuntimeError:
            pass
    body["lane_digest"] = oc_calib.canonical_sha256([[record["id"], record["response_verbatim"]] for record in responses])
    body["lane_digest_validity"] = {
        "note": "valid ONLY for this exact tuple; any element moving moves the digest and a moved lane grades nothing.",
        "bank_file_sha256": bank_digest,
        "prompt_surface_sha256": surface,
        "dtype": str(args.dtype),
        "compute_architecture": architecture,
        "transformers_version": transformers.__version__,
        "maximum_new_tokens": int(args.maximum_new_tokens),
        "runtime": RUNTIME,
        "mode": mode,
        "decode_path": body["generation"]["decode_path"],
        "linear_attention_kernel_path": export["linear_attention_kernels"]["kernel_path"],
        "linear_attention_kernel_report_schema": KERNEL_REPORT_SCHEMA,
        "chat_template_thinking": thinking["requested"],
        "chat_template_thinking_took_effect": thinking["took_effect"],
        "ledger_band_logit_mask": bool(args.ledger_band_logit_mask),
        "chimera_commit": tree["commit"],
        "split": placement["split"],
    }
    body["dossier_sha256"] = oc_calib.canonical_sha256(body)
    out_digest = oc_calib.write_canonical(Path(args.out), body)
    log(f"dossier {body['dossier_sha256'][:16]} -> {args.out}")
    print(json.dumps({"out": str(args.out), "file_sha256": out_digest, "dossier_sha256": body["dossier_sha256"], "lane_digest": body["lane_digest"], "cases": len(responses), "mode": mode, "emit_zero_floor_passed": floor["passed"], "decode_path_agree": preflight["agree"], "media_out_records": len(media_records), "taps": None if tap_receipt is None else len(tap_receipt["layers"])}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="Chimera checkpoint directory (chimera_omni.checkpoint.v1)")
    ap.add_argument("--tokenizer-dir", type=Path, help="HF directory with tokenizer + chat template (defaults to --model)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--parent-code")
    ap.add_argument("--stage", default="parent", choices=["parent", "fold", "distill", "finetune"])
    ap.add_argument("--chimera-src", default=os.environ.get("CHIMERA_OMNI_SRC", DEFAULT_CHIMERA_SRC), help="chimera-omni tree root ($CHIMERA_OMNI_SRC)")
    ap.add_argument("--chimera-commit", help="refuse unless the tree is at this commit (s0_import_27b's check)")
    ap.add_argument("--calibration-dir", default=os.environ.get("OC_CALIBRATION_DIR", DEFAULT_CALIBRATION_DIR), help="calibration lane with oc_calib.py / oc_run_frozen_bank.py ($OC_CALIBRATION_DIR)")
    ap.add_argument("--split", help="free-run placement, e.g. 46,24 over the visible CUDA devices in order")
    ap.add_argument("--stream", action="store_true", help="streamed engine, one block resident; teacher-forced proxy only")
    ap.add_argument("--teacher-forced", action="store_true")
    ap.add_argument("--reference-dossier", type=Path, help="stream mode: dossier whose child_responses are teacher-forced")
    ap.add_argument("--input", type=Path, help="stream mode: qwen-fold evidence manifest for --tap-residuals")
    ap.add_argument("--held-only", action="store_true")
    ap.add_argument("--micro-batch", type=int, help="stream mode rows per padded pass (default 1 with --input evidence rows, 8 for bank rows); recorded in the dossier and taps_receipt.json")
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--bank-sha256")
    ap.add_argument("--lanes")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--probe-prompts", required=True, type=Path)
    ap.add_argument("--smoke-prompts", type=Path)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--thinking", default="template-default", choices=["template-default", "off"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument("--smoke-maximum-new-tokens", type=int, default=128)
    ap.add_argument("--extra-eos-tokens", default="<|im_end|>,<|endoftext|>")
    ap.add_argument("--ledger-band-logit-mask", action="store_true")
    ap.add_argument("--verify-tensors", action="store_true", help="also check every tensor against the index's tensor_sha256 while loading")
    ap.add_argument("--tap-residuals", action="store_true")
    ap.add_argument("--tap-out", type=Path)
    ap.add_argument("--tap-normed", action="store_true", help="also dump the normed FFN input")
    ap.add_argument("--tap-max-rows", type=int, default=8192, help="stream --input: keep every k-th position so the dump holds about this many rows per layer (0 = all)")
    ap.add_argument("--vision-cards", action="store_true", help="run the bank's text+image cases through the native media bridge")
    ap.add_argument("--assets-dir", type=Path, help="PNG assets for --vision-cards (default: <bank dir>/assets)")
    ap.add_argument("--media-out", help="comma list; only 'music' (music_codes band token stream) is built")
    ap.add_argument("--media-out-dir", type=Path)
    ap.add_argument("--media-tokens", type=int, default=256)
    ap.add_argument("--stimuli", type=Path, help="reserved for the S6 continuous samplers (refused with --media-out image/video/volume)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--x0-diagnostic", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.smoke_prompts and not args.skip_smoke:
        args.smoke_prompts = args.probe_prompts.parent / "matrix_campaign_smoke_prompts_v1.json"
    try:
        return run(args)
    except RuntimeError as exc:
        if type(exc).__name__ in ("Refusal", "CalibrationError"):
            receipt = getattr(exc, "receipt", None)
            if receipt is not None and getattr(args, "out", None):
                out = Path(args.out)
                path = out.with_name(out.name + ".refused.json")
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({**receipt, "message": str(exc), "label": str(args.label), "model": str(args.model), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2, sort_keys=True), encoding="utf-8")
                    print(f"REFUSAL RECEIPT: {path}", file=sys.stderr, flush=True)
                except OSError as write_error:
                    print(f"refusal receipt could not be written: {write_error}", file=sys.stderr, flush=True)
            return refuse(str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
