#!/usr/bin/env python3
"""media_b_teacher_cache.py -- stage-C media teacher caches, lane media-b (Claude, 2026-10-12).

Design of record: distill/DISTILL_DESIGN_2026-10-12.md section 3 (rows "HY3D trajectories", "ACE DiT + LM",
"QVL32 ViT features") and S0 P4 / P8. Builds on composite/tools/s7_pairs_parent.py 1.1.0 (loaded by path and
version-pinned: its sha256_file, Qwen-VL patchify, image positions and native type ids are reused verbatim).

Every teacher output is written ONCE under

    <cache-root>/<teacher>/<corpus_sha256>/<config_sha256>/cache_config.json      (the run config, first start)
    <cache-root>/<teacher>/<corpus_sha256>/<config_sha256>/shard-NNNNN.safetensors (+ shard-NNNNN.json, last)
    <cache-root>/<teacher>/<corpus_sha256>/<config_sha256>/manifest.json          (+ .sha256, by `seal`)

one shard per corpus item. config_sha256 is the sha of the canonical RUN CONFIG: every argument that changes a
stored target (HY3D steps/guidance/octree/chunks, ACE cfg/clip seconds/sigmas per clip/VAE device, LM top-k and
durations, QVL block/max pixels) plus the compat epoch (COMPAT). A relaunch with another --cfg or --block therefore
writes a NEW cache instead of resuming into the old one. Resume skips a shard only when its file sha matches, its
config sha is this cache's, and its tool version is in COMPAT[teacher]; `seal` refuses any shard that is not, and the
manifest records the run config. The manifest also records the teacher integrity check before and after (a full
sha256 of every teacher file), the corpus sha, the shard read order and a finite check per shard (DISTILL_DESIGN
section 3). `address -- <worker args>` prints the cache dir a worker would use.

Checks are named for what they prove. "recording" (HY3D Euler chain residual) and "determinism" (a standalone
replay of one step / block) re-run this tool's own path, so they cannot catch a transcription error against the
parent. "reference" arms compare against an independent implementation: HY3D = the upstream
Hunyuan3DDiTFlowMatchingPipeline.__call__ from the same source tree on the same image and seed (final latents vs
x0_final, first item of each run); ACE LM = ComfyUI generate_audio_codes on the same prompt and seed; QVL = the block-13
rows s7_pairs_parent captured on 2026-10-10 for the same image at the same square side. The ACE DiT check (ComfyUI
denoised vs x - sigma*v) proves the hook captured the tensor ComfyUI uses; it says nothing about x0, codes or
conditioning matching the deployed path.

Caches (teacher -> shard tensors, one shard per item):

    hy3d_trajectories   Hunyuan3D 2.1, native hy3dshape source, fp32 on a P40. One image -> one guided sampling
                        trajectory of the upstream Hunyuan3DDiTFlowMatchingPipeline loop (CFG, guidance 5.0,
                        sigmas linspace(0, 1, steps)): x_t [S,4096,64], v_guided [S,4096,64], t_native [S] (the
                        model's timestep input; HY3D convention t=0 is PURE NOISE, t=1 data), noise_level = 1-t,
                        sigma_next [S], x0_final, the DINO context, the processed image, and the ShapeVAE
                        reference mesh decoded from x0_final (vertices, faces; also written as a .ply sidecar).
    ace_dit_velocity    ACE-Step 1.5 turbo DiT through ComfyUI's own model code (in-process, fp32 on a P40). One
                        clip -> Oobleck VAE latents x0 of a real curated clip window, its own semantic codes
                        (the parent's tokenizer), the parent's text/lyric conditioning, and at 4 sigmas from the
                        deployed sampler schedule (shift 3, simple, 8 steps): x_t = sigma*noise + (1-sigma)*x0 and
                        the raw DiT output v (the velocity; ComfyUI denoised = x_t - sigma*v), guided at --cfg.
    ace_lm_codes        ACE-Step 1.5 LM qwen_1.7b_ace15 (ComfyUI Qwen3_2B_ACE15_lm, fp32 on a P40). One prompt ->
                        a self-generated audio-code sequence with ComfyUI's own sampling loop (CFG 2.0, T 0.85,
                        top-p 0.9: sample_manual_loop_no_classes, transcribed with a recording hook and checked
                        token-for-token against the original on the first prompt) and at every position the
                        CONDITIONAL branch's top-64 over the 64,000 audio codes [151669, 215669): code ids int32,
                        log-probs bf16 (log-softmax over the full 217,204 vocab), lse_full, lse_band, eos logprob.
    qvl32_vit_l13       Qwen3-VL-32B vision tower (visual.* of qwen3vl_32b_minimax_h3_bf16), fp32. One image ->
                        block-13 INPUT and OUTPUT rows [N,1152] + grid/positions; the block is replayed standalone on
                        the stored input and must reproduce the stored output (determinism only). The parent
                        reference is the s7 arm, carried for a production corpus by a side cache of the same run
                        config over the s7 capture images (`check`, 1.0.9).

Subcommands
    host side (any python, no GPU):  corpus-hy3d, corpus-ace, corpus-qvl, corpus-import, p8-header, seal, verify, launch,
                                     address
    workers (inside a runtime):      hy3d-run, hy3d-probe, ace-encode (CPU only), ace-dit-run, ace-lm-run, ace-probe,
                                     qvl-run, selfcheck

Runtimes (recorded in every receipt):
    HY3D + ACE   docker image local/hy3d-base:735020f27cb59da7 (python 3.10, torch 2.6.0+cu124 with sm_60 kernels,
                 ComfyUI b1693ec at /opt/ComfyUI, hy3dshape deps). `launch --runtime docker` runs one container per
                 card with --gpus device=N (the process sees ONE card: no P2P is possible).
    QVL          the Pascal venv model-surgery-lab/venv-pascal-cu118-torch260 (torch 2.6.0+cu118, transformers
                 5.12.1, the version s7_pairs_parent's qvl capture used). `launch --runtime pascal` sets
                 CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=N.

Standing rules: no training, nothing here changes a weight, parents are opened read-only and hashed before and
after. Finite fit numbers are diagnostics. STRUCTURAL failures refuse (exit 3): missing/changed shard, non-finite
tensor, teacher changed, a worker seeing more than one card, an empty decoded mesh in a mini-cache gate, a forbidden
card. GPU 0/4/6 belong to the step-1 workflow and GPU 5 to another project: `launch` refuses them, and refuses any
card already above 2,000 MiB.

1.0.1 (2026-10-12): the ACE worker re-asserts the DiT on the card (comfy load_models_gpu, force_full_load) before the
tokenizer and every apply_model, because loading the VAE or the text encoder lets ComfyUI's memory manager evict it;
the tool's own sha256 (and s7_pairs_parent's) is taken at IMPORT, so a receipt names the code that actually ran even
if the file is replaced while a worker is running. 1.0.0 produced only the P4 HY3D probe receipt and the corpora.

1.0.2 (2026-10-12): the 1.0.0/1.0.1 ACE probe crashed writing its receipt on a NaN (the denoised-vs-v check) after
ComfyUI fell back to TILED VAE encoding on an out-of-memory. Now: the Oobleck encode runs with every other model
unloaded and a tiled fallback is detected and refused (--allow-tiled-vae records it instead); x0, the text/lyric
conditioning and v are finite-checked by name, and a non-finite v triggers a forward-hook sweep that records the first
DiT submodule emitting a non-finite value; ace-dit-run encodes all its clips before the DiT is loaded; receipts write
non-finite floats as strings instead of being lost. HY3D/QVL paths are unchanged.

1.0.3 (2026-10-12): the 1.0.2 probe named the NaN: the Oobleck encode of a 30 s window in fp32 does not fit a P40
even alone (peak 23,877 MiB allocated), ComfyUI falls back to TILED encoding, and the tiled encode returns NON-FINITE
latents. The VAE now runs the regular (untiled) encode on the CPU in fp32 by default (--vae-device cpu; ComfyUI's own
VAE class with device=cpu), so x0 is the parent's exact regular encode; the DiT and LM stay on the card. The LM shard
meta records the LM parameter dtypes (the file is bf16; ComfyUI keeps some weights stored bf16 and computes in fp32).
The text encoder is ALWAYS the dual load qwen_0.6b_ace15 + qwen_1.7b_ace15 (the deployed DualCLIPLoader pair): the
0.6B file alone is detected as a generic Qwen3-0.6B (vocab 151,936 vs the file's 151,669) and fails to load; and the
TE patcher is loaded with force_full_load (a plain load_model left part of the LM on the CPU, which 1.0.2 refused).
`seal` compares the teacher check only against before-records over the SAME file set (mode + paths); records over
another file set (a worker of an older tool version) are listed as superseded, never silently counted as equal.

1.0.4 (2026-10-12, probe only): the ACE P4 LM phase unloads the DiT first (ace-lm-run never loads it; the 1.0.3 probe
ran the LM beside the resident fp32 DiT and a second lane's job on the same card and hit an OOM casting the head), and
records the card's total used memory at each phase; containers get PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
Every ComfyUI call (VAE encode, text encode, the LM loop and the reference generate_audio_codes it is checked against)
runs under torch.inference_mode(), as ComfyUI's executor runs every node: 1.0.3 called them bare, autograd tracked the
encode, and the shard writer refused x0 for carrying requires_grad (values unaffected; memory and time were).

1.0.5 (2026-10-13, review fixes): (1) the cache address carries the run config (see above): 1.0.4 keyed a cache on the
corpus only and resumed on file sha + tool version alone, so a relaunch with --cfg 2 would have skipped the cfg-1
shards and sealed a mixed cache. Shards and manifests of 1.0.4 have no config and are not resumable or sealable by
1.0.5 (the COMPAT epoch starts at 1.0.5). (2) ace-dit-run is resumable through the CPU encode: each clip's x0 is
written to a sha-checked stage file as soon as it is encoded (<corpus_sha>/x0stage-<x0 config sha>/, keyed on the
x0-affecting arguments only, so a --cfg change reuses them), the DiT is loaded only AFTER the encode and text phases,
and `ace-encode` runs the encode alone in a CPU-only container (`launch --cpu-only`, no card claimed). (3) checks are
relabelled recording / determinism / reference, and two reference arms are added: HY3D upstream pipeline final
latents and QVL vs the s7_pairs_parent block-13 capture.

1.0.6 (2026-10-13): the CPU-only ace-encode container failed at ComfyUI import (comfy_kitchen -> triton raised
RuntimeError "0 active drivers" with no GPU, past its ImportError guard); the CPU-only path now blocks the triton
import so it surfaces as ImportError. hy3d-run imports the upstream pipeline module at run start instead of at the
first reference arm (1.0.5 imported ComfyUI's model_management after the first trajectory, so items of one run could
execute in two process states). The QVL reference arm gates on whole-tensor relative error (rel_fro <= 2e-2) instead
of max_abs <= 1% of absmax: 1.0.5 refused the 4th batch1 image at max_abs 4.92 because the s7 capture itself carries
TF32 drift (see QVL_REF_NOTE; our P40 rows match an fp32 CPU run to rel_fro <= 1.7e-5). No stored target changes:
COMPAT lists 1.0.5 and 1.0.6 together.

1.0.7 (2026-10-13): ACE x0 is a posterior SAMPLE of the Oobleck bottleneck (randn from the global RNG), so 1.0.4-1.0.6
encodes were not reproducible (the 1.0.4 and 1.0.5 minis of the same clip differ by max 2.6e-3; an in-process repeat
by 2.7e-3); a resumed or re-encoded item got another x0. The sample is now seeded per item
(item_seed(sha256, 'ace-vae-posterior'), inside fork_rng). This changes stored ACE-DiT targets: a new compat epoch
for ace_dit_velocity (and its x0 stage address); HY3D, ACE-LM and QVL are unchanged and keep 1.0.5/1.0.6 compatible.

1.0.8 (2026-10-13, wave-2 production on the P40s): (1) `corpus-import` turns a distill_corpus_build media set
(<set>/items.jsonl + manifest.json: the screened, eval-split-free HY3D / ACE / ViT sets) into a teacher corpus; the
items file sha must equal the set manifest's, every referenced file is re-hashed against its recorded sha, a row
whose file is byte-identical to an earlier row's is dropped and listed (the ACE set has 60 such pairs: two registry
rows cutting the same window of one source), and the upstream screen is recorded as applied. (2) Caches may live on /mnt/lexar (exFAT): exFAT has no hard links (os.link
raises EPERM), so claim_config falls back to an O_EXCL create and a settled re-read; a docker worker whose
--cache-root is under TEACHER_CACHE_WRITE_ROOTS outside $OC gets that root mounted read-write (the rest of
/mnt/lexar/omni-campaign stays read-only); any other root outside $OC is refused at launch. (3) `check` = verify + a structural read
of a sealed cache with a receipt: HY3D reference meshes have faces > 0 and a .ply, ACE DiT x0 latents are finite and
non-zero with no tiled VAE, LM shards hold codes, QVL shards hold rows (exit 3 on any problem). No stored target
changes: 1.0.8 joins every COMPAT epoch.

1.0.9 (2026-10-13, review fix): a QVL cache over a production corpus can never fire its s7 reference arm (the arm
needs an image of the 32-image s7 capture at a square side; smart_resize keeps aspect, and the corpus is disjoint),
so 1.0.8 `check` passed a QVL cache whose only per-shard check was the determinism replay -- the same block re-run
on its own captured input, which cannot catch a mistransposed or misnormalised block input. `check` of a block-13
QVL cache now requires a PARENT REFERENCE: either reference arms inside the cache, or a sealed side cache with the
SAME run config sha (so the same knobs; corpus = the s7 capture images, which are 512x512, so the unchanged qvl-run
fires its arm at side 512 on every item) found under QVL_PARENT_REF_ROOT or named by --parent-reference. Every arm's
rel_fro is re-gated from the shard metas; no reference, a failing arm or a mismatched config is a structural FAIL.
qvl-run, the stored tensors and the cache address are unchanged: 1.0.9 joins every COMPAT epoch.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

TOOL_NAME = "media_b_teacher_cache"
TOOL_VERSION = "1.0.9"
CACHE_SCHEMA = "omni-campaign.teacher_cache.v2"
SHARD_SCHEMA = "omni-campaign.teacher_cache_shard.v2"
CONFIG_SCHEMA = "omni-campaign.teacher_cache_config.v1"
STAGE_SCHEMA = "omni-campaign.ace_x0_stage.v1"
CORPUS_SCHEMA = "omni-campaign.teacher_corpus.v1"
PROBE_SCHEMA = "omni-campaign.p4_media_probe.v1"
# Tool versions whose shards are interchangeable within one run config. The FIRST entry is the compat epoch and is
# part of the cache address. A release that changes what a teacher stores starts a new epoch (a new first entry);
# a release that does not appends its version. A version not listed is never resumed into and never sealed.
COMPAT = {"hy3d_trajectories": ("1.0.5", "1.0.6", "1.0.7", "1.0.8", "1.0.9"), "ace_dit_velocity": ("1.0.7", "1.0.8", "1.0.9"),
          "ace_lm_codes": ("1.0.5", "1.0.6", "1.0.7", "1.0.8", "1.0.9"), "qvl32_vit_l13": ("1.0.5", "1.0.6", "1.0.7", "1.0.8", "1.0.9")}
ACE_VAE_POSTERIOR_SALT = "ace-vae-posterior"
WORKER_TEACHER = {"hy3d-run": "hy3d_trajectories", "ace-encode": "ace_dit_velocity", "ace-dit-run": "ace_dit_velocity",
                  "ace-lm-run": "ace_lm_codes", "qvl-run": "qvl32_vit_l13"}
CPU_WORKERS = ("ace-encode",)
HERE = Path(__file__).resolve().parent
TOOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # at import: the code that runs
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
S7_REQUIRED_VERSION = "1.1.0"

MODELS = Path("/home/cereal/comfyui/models")
HY3D_CKPT = MODELS / "checkpoints/hunyuan_3d_v2.1.safetensors"
HY3D_SOURCE = Path("/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1")
HY3D_PACKAGE = "hy3dshape/hy3dshape"
HY3D_CONFIG = "configs/dit_config_2_1.yaml"
ACE_DIT = MODELS / "diffusion_models/acestep_v1.5_turbo.safetensors"
ACE_VAE = MODELS / "vae/ace_1.5_vae.safetensors"
ACE_TE = MODELS / "text_encoders/qwen_0.6b_ace15.safetensors"
ACE_LM = MODELS / "text_encoders/qwen_1.7b_ace15.safetensors"
QVL_WEIGHTS = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
RUNTIME_IMAGE = "local/hy3d-base:735020f27cb59da7"
CONTAINER_PYTHON = "/opt/environments/python/comfyui/bin/python"
PASCAL_PYTHON = "/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python"
REGISTRY_POINTER = Path("/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json")
# Cache roots a worker may write outside $OC (wave 2: teacher caches on Lexar). A docker worker gets exactly its
# --cache-root mounted read-write when it lies under one of these; the rest of /mnt/lexar/omni-campaign stays :ro.
TEACHER_CACHE_WRITE_ROOTS = (Path("/mnt/lexar/omni-campaign/teacher_cache"),)
DISTILL_SET_SCHEMA = "neural-forge-omni-distill-corpus-v1/media/"
S7_QVL_CAPTURE = OC / "composite/s7/pairs/qvl32_vit_L13"
# The s7 capture ran on GPU0, an RTX 3090 (allocator m1-media-charts-gpu0-claude, 2026-10-10 08:31:16-08:32:16Z), where
# PyTorch runs cuDNN convolutions (the Conv3d patch embed) in TF32 by default. Measured 2026-10-13 on 4 batch1 images
# at side 512: our P40 rows vs an fp32 CPU run of the same image rel_fro 6.7e-6..1.7e-5 (row cosine 1.000000); the
# s7 rows vs that fp32 CPU run rel_fro 2.9e-3..6.8e-3 (row cosine >= 0.9927), max_abs up to 4.9 on rows with
# |x| ~300. A transcription error (normalisation, patch order, wrong block) moves rel_fro by O(0.1..1), so the
# cross-device arm gates on rel_fro, not on max_abs.
QVL_REF_REL_FRO = 2e-2
QVL_REF_NOTE = ("s7 reference captured on an RTX 3090 (TF32 cuDNN conv default); exact-fp32 drift vs it is rel_fro ~3e-3..7e-3 "
                "(measured 2026-10-13); gate rel_fro <= 2e-2 catches transcription errors, not TF32 drift")
# Side caches that carry the QVL parent reference for production caches of the same run config (1.0.9): qvl-run over a
# corpus of the s7 capture images (512x512 -> smart_resize keeps 512x512 -> the arm fires) with --cache-root here.
QVL_PARENT_REF_ROOT = TEACHER_CACHE_WRITE_ROOTS[0] / "qvl32_vit_ref"

# ACE-Step 1.5 LM vocabulary (comfy/text_encoders/ace15.py sample_manual_loop_no_classes defaults @ b1693ec)
ACE_AUDIO_START = 151669
ACE_AUDIO_END = 215669
ACE_EOS = 151645
ACE_CODES = ACE_AUDIO_END - ACE_AUDIO_START
ACE_LM_DEFAULTS = {"cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}
# TextEncodeAceStepAudio1.5 node defaults (comfy_extras/nodes_ace.py @ b1693ec)
ACE_NODE_DEFAULTS = {"bpm": 120, "timesignature": 4, "language": "en", "keyscale": "C major"}
# deployed ACE 1.5 turbo workflow (text-to-music skill): ModelSamplingAuraFlow shift 3, euler, simple, cfg 1.0
ACE_SHIFT = 3.0
ACE_STEPS = 8
ACE_DEPLOYED_CFG = 1.0
ACE_SECONDS_PER_LATENT = 1764.0 / 44100.0
HY3D_GUIDANCE = 5.0          # Hunyuan3DDiTFlowMatchingPipeline.__call__ default
HY3D_STEPS = 30              # DISTILL_DESIGN section 3: 256 shapes x 30 steps
HY3D_BOX_V = 1.01
HY3D_MC_LEVEL = 0.0

ALLOWED_CARDS = (1, 2, 3)
FORBIDDEN_CARDS = {0: "step-1 workflow (RTX 3090)", 4: "step-1 workflow (RTX 3090)",
                   5: "reserved for another project (P40)", 6: "step-1 workflow (RTX 8000)"}
MAX_START_MIB = 2000
MAX_THREADS = 16
LYRIC_TAG = re.compile(r"\[(verse|chorus|bridge|intro|outro|pre-chorus|hook|refrain)[^\]]*\]", re.IGNORECASE)
AUDIO_EXT = (".ogg", ".mp3", ".flac", ".wav", ".m4a", ".opus")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".ppm", ".bmp")


class Refusal(Exception):
    """A structural failure: the run is refused (exit 3). Never raised for a finite fit number."""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require(condition: Any, message: str) -> None:
    if not condition:
        raise Refusal(message)


# --------------------------------------------------------------------------
# s7_pairs_parent 1.1.0 (the tool this lane builds on), loaded by path
# --------------------------------------------------------------------------

_S7 = None


def s7_candidates() -> list[Path]:
    paths = []
    if os.environ.get("S7_PAIRS_PARENT"):
        paths.append(Path(os.environ["S7_PAIRS_PARENT"]))
    paths += [OC / "composite/tools/s7_pairs_parent.py", HERE.parent.parent / "composite/tools/s7_pairs_parent.py"]
    return paths


def load_s7(path: Path | None = None):
    """Import s7_pairs_parent by file path and refuse any version other than 1.1.0."""

    global _S7
    if _S7 is not None and path is None:
        return _S7
    found = [p for p in ([path] if path else s7_candidates()) if p and p.is_file()]
    require(found, f"s7_pairs_parent.py not found in {[str(p) for p in s7_candidates()]}")
    spec = importlib.util.spec_from_file_location("s7_pairs_parent_mediab", found[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(getattr(module, "TOOL_VERSION", None) == S7_REQUIRED_VERSION,
            f"s7_pairs_parent {getattr(module, 'TOOL_VERSION', None)} at {found[0]}; this tool is built on {S7_REQUIRED_VERSION}")
    module.MEDIA_B_PATH = str(found[0])
    module.MEDIA_B_SHA256 = hashlib.sha256(found[0].read_bytes()).hexdigest()
    if path is None:
        _S7 = module
    return module


def sha256_file(path: str | Path) -> str:
    return load_s7().sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def json_safe(value: Any) -> Any:
    """Non-finite floats become the strings 'nan' / 'inf' / '-inf' so a receipt is never lost to a NaN."""

    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if value != value else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json_atomic(path: Path, body: Any, *, sidecar_sha: bool = False) -> str:
    body = json_safe(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    digest = sha256_file(path)
    if sidecar_sha:
        (path.parent / (path.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    return digest


def tool_identity() -> dict[str, Any]:
    s7 = load_s7()
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": TOOL_SHA256, "path": str(Path(__file__).resolve()),
            "builds_on": {"name": s7.TOOL_NAME, "version": s7.TOOL_VERSION, "sha256": s7.MEDIA_B_SHA256, "path": s7.MEDIA_B_PATH}}


# --------------------------------------------------------------------------
# corpus manifests (content-addressed inputs)
# --------------------------------------------------------------------------


def corpus_sha(kind: str, items: list[dict[str, Any]]) -> str:
    """The sha of the item list only: provenance notes can change without changing the cache address."""

    return sha256_bytes(canonical_json({"schema": CORPUS_SCHEMA, "kind": kind, "items": items}))


def item_seed(sha: str, salt: str) -> int:
    return int(sha256_bytes(f"{salt}:{sha}".encode())[:8], 16)


def held_split(sha: str, held_mod: int) -> bool:
    """Deterministic diagnostic holdout by content (never by position)."""

    return held_mod > 0 and int(sha[-8:], 16) % int(held_mod) == 0


def load_exclusions(path: str | None) -> tuple[set[str], dict[str, Any]]:
    """The contamination carve-out list: sha256 (file or payload) / record ids, one per line or a json list."""

    if not path:
        return set(), {"applied": False, "list": None, "why": "no carve-out list given (recorded deviation)"}
    text = Path(path).read_text(encoding="utf-8")
    try:
        body = json.loads(text)
        values = body if isinstance(body, list) else body.get("exclude", [])
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    values = {str(v).strip().lower() for v in values}
    return values, {"applied": True, "list": str(path), "list_sha256": sha256_file(path), "entries": len(values)}


def write_corpus(out: Path, kind: str, items: list[dict[str, Any]], *, source: dict[str, Any], screen: dict[str, Any],
                 deviations: list[str]) -> dict[str, Any]:
    require(items, f"{kind}: empty corpus")
    ids = [item["id"] for item in items]
    require(len(ids) == len(set(ids)), f"{kind}: duplicate item ids")
    shas = [item["sha256"] for item in items]
    require(len(shas) == len(set(shas)), f"{kind}: duplicate item content (sha256)")
    body = {"schema": CORPUS_SCHEMA, "kind": kind, "corpus_sha256": corpus_sha(kind, items), "items": items,
            "n_items": len(items), "n_held": sum(1 for i in items if i.get("held")), "source": source,
            "contamination_screen": screen, "deviations": deviations, "tool": tool_identity(), "created_at_utc": utc_now()}
    write_json_atomic(out, body, sidecar_sha=True)
    log(f"corpus {kind}: {len(items)} items, sha {body['corpus_sha256'][:16]} -> {out}")
    return body


def read_corpus(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    require(body.get("schema") == CORPUS_SCHEMA, f"{path}: not a {CORPUS_SCHEMA} file")
    require(corpus_sha(body["kind"], body["items"]) == body["corpus_sha256"], f"{path}: corpus sha does not match its items")
    body["path"] = str(path)
    body["file_sha256"] = sha256_file(path)
    return body


def image_captions(directory: Path) -> dict[str, str]:
    """QIMG calibration receipts carry the generating prompt per file (parent_images_receipt.json)."""

    receipt = directory / "parent_images_receipt.json"
    if not receipt.is_file():
        return {}
    body = json.loads(receipt.read_text(encoding="utf-8"))
    return {str(e.get("file")): str(e.get("prompt", "")) for e in body.get("images", []) if e.get("file")}


def image_dir_items(dirs: list[str], *, names: list[str] | None, salt: str, held_mod: int,
                    exclude: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items, counts = [], {"files": 0, "excluded": 0, "name_filtered": 0}
    for d in dirs:
        root = Path(d)
        captions = image_captions(root)
        for path in sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXT):
            counts["files"] += 1
            if names and path.stem not in names:
                counts["name_filtered"] += 1
                continue
            sha = sha256_file(path)
            if sha in exclude or path.stem.lower() in exclude:
                counts["excluded"] += 1
                continue
            items.append({"id": f"{root.name}/{path.stem}", "path": str(path), "sha256": sha, "bytes": path.stat().st_size,
                          "caption": captions.get(path.name, ""), "seed": item_seed(sha, salt), "held": held_split(sha, held_mod)})
    return items, counts


def order_and_limit(items: list[dict[str, Any]], *, salt: str, limit: int | None) -> list[dict[str, Any]]:
    """Deterministic content-keyed order (independent of directory listing order), then the first `limit`."""

    ordered = sorted(items, key=lambda i: (item_seed(i["sha256"], salt + ":order"), i["sha256"]))
    return ordered[: int(limit)] if limit else ordered


def registry_rows(routes: set[str], *, max_rows: int | None = None):
    """(entry, record, row) for curated registry rows on `routes` via chimera_omni.data.curated_shards.

    Entries whose shard pointer cannot be read are skipped and counted (two such entries on 2026-10-12)."""

    src = os.environ.get("CHIMERA_OMNI_SRC")
    if src:
        sys.path.insert(0, str(Path(src) / "src") if (Path(src) / "src").is_dir() else src)
    from chimera_omni.data import curated_shards as cs  # noqa: E402

    stats = {"entries": 0, "entries_unreadable": 0, "rows": 0}
    emitted = 0
    for entry in cs.iter_master_entries(os.environ.get("CURATED_REGISTRY", str(REGISTRY_POINTER))):
        mc = {cs.canonical_route(k) for k in (entry.get("modality_counts") or {})}
        if not (mc & routes):
            continue
        stats["entries"] += 1
        try:
            specs = list(cs.iter_shard_specs(entry))
        except (OSError, ValueError):
            stats["entries_unreadable"] += 1
            continue
        for shard in specs:
            for row, line in cs.iter_rows_for_shard(shard):
                rec = cs.normalize_row(row, source_dataset=str(entry.get("dataset_id", "")), training_status=str(entry.get("training_status", "")),
                                       shard_path=str(shard.get("path", "")), line_number=line)
                if rec is None or rec.route not in routes:
                    continue
                stats["rows"] += 1
                yield entry, rec, row, stats
                emitted += 1
                if max_rows and emitted >= max_rows:
                    return


def split_caption_lyrics(text: str) -> tuple[str, str]:
    """Curated music rows carry either a tag/description line or a lyric sheet in the instruction.

    Rule (recorded in the corpus): strip a leading 'user:'; if a lyric section tag ([verse], [chorus], ...) is
    present, everything from the first tag on is the lyric sheet and the text before it the caption (or 'A song.'
    if empty); otherwise the whole text is the caption and the lyrics are empty (instrumental)."""

    text = re.sub(r"^\s*user:\s*", "", str(text or "")).strip()
    match = LYRIC_TAG.search(text)
    if not match:
        return text[:1024], ""
    caption = text[: match.start()].strip() or "A song."
    return caption[:1024], text[match.start():].strip()[:4096]


# --------------------------------------------------------------------------
# shard store (one shard per corpus item) + teacher integrity
# --------------------------------------------------------------------------


def _is_float(t: Any) -> bool:
    return bool(getattr(t, "is_floating_point", lambda: False)())


def finite_check(tensors: dict[str, Any]) -> None:
    import torch

    for name, value in tensors.items():
        if _is_float(value):
            for part in value.reshape(-1).split(1 << 22):
                require(bool(torch.isfinite(part.float()).all()), f"non-finite teacher tensor {name}")


def require_finite(name: str, tensor: Any) -> None:
    import torch

    require(bool(torch.isfinite(tensor.float()).all()), f"non-finite {name} (shape {list(tensor.shape)})")


def device_type(device: str) -> str:
    """'cuda:0' -> 'cuda', 'cpu' -> 'cpu' (the numerics depend on the device kind, not the index)."""

    return str(device).split(":")[0].strip().lower()


def ace_x0_knobs(args: argparse.Namespace) -> dict[str, Any]:
    """Every argument that changes an ACE x0 (the Oobleck latents of one clip window)."""

    return {"clip_seconds": float(args.clip_seconds), "vae_device": device_type(args.vae_device),
            "allow_tiled_vae": bool(args.allow_tiled_vae), "window": "offset item_seed(str(seed), 'ace-window')",
            "posterior_sample": f"VAEBottleneck randn seeded per item: item_seed(sha256, '{ACE_VAE_POSTERIOR_SALT}')"}


def run_config(teacher: str, args: argparse.Namespace) -> dict[str, Any]:
    """The canonical run config of a cache: every argument that changes a stored target, plus the compat epoch.
    Arguments that only change checks or logging (replay cadence, reference arms, parity, --require-mesh) are out."""

    if teacher == "hy3d_trajectories":
        knobs = {"steps": int(args.steps), "guidance": float(args.guidance), "octree_resolution": int(args.octree_resolution),
                 "num_chunks": int(args.num_chunks), "box_v": HY3D_BOX_V, "mc_level": HY3D_MC_LEVEL,
                 "sigmas": "linspace(0, 1, steps)", "dtype": "fp32"}
    elif teacher == "ace_dit_velocity":
        knobs = {**ace_x0_knobs(args), "cfg": float(args.cfg), "sigmas_per_clip": int(args.sigmas_per_clip), "shift": ACE_SHIFT,
                 "steps": ACE_STEPS, "node_defaults": ACE_NODE_DEFAULTS, "uncond": "zeroed text embedding, same lyrics + codes", "dtype": "fp32"}
    elif teacher == "ace_lm_codes":
        knobs = {"topk": int(args.topk), "min_seconds": float(args.min_seconds), "max_seconds": float(args.max_seconds),
                 "sampling": ACE_LM_DEFAULTS, "node_defaults": ACE_NODE_DEFAULTS}
    elif teacher == "qvl32_vit_l13":
        knobs = {"block": int(args.block), "max_pixels": int(args.max_pixels), "store_patches": bool(args.store_patches),
                 "resize_factor": 32, "dtype": "fp32"}
    else:
        raise Refusal(f"unknown teacher {teacher}")
    return {"schema": CONFIG_SCHEMA, "teacher": teacher, "compat_epoch": COMPAT[teacher][0], "knobs": knobs}


def claim_config(directory: Path, name: str, config: dict[str, Any]) -> str:
    """Write <directory>/<name> on first start (atomic create, never overwritten); refuse a different config."""

    digest = sha256_bytes(canonical_json(config))
    require(directory.name.endswith(digest), f"{directory} is not addressed by its config sha {digest[:16]}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    text = json.dumps(config, indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.link(tmp, path)  # fails if another worker created it first
    except FileExistsError:
        pass
    except OSError as exc:
        # exFAT (/mnt/lexar) has no hard links: EPERM. An O_EXCL create is the atomic first-writer-wins there.
        require(exc.errno in NO_HARDLINK_ERRNOS, f"{path}: claim failed: {exc}")
        create_exclusive(path, text.encode("utf-8"))
    finally:
        tmp.unlink(missing_ok=True)
    on_disk = read_json_settled(path)
    require(sha256_bytes(canonical_json(on_disk)) == digest, f"{path} holds another run config: refusing to mix targets")
    return digest


NO_HARDLINK_ERRNOS = {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EMLINK, errno.EXDEV}


def create_exclusive(path: Path, data: bytes) -> bool:
    """Create `path` holding `data` only if it does not exist (O_EXCL); False when another writer was first."""

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def read_json_settled(path: Path, *, tries: int = 40, pause: float = 0.25) -> Any:
    """Read a json file another worker may still be writing (the O_EXCL path has no rename): retry only on a
    partial (undecodable) body, never on a complete one."""

    for attempt in range(int(tries)):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt == int(tries) - 1:
                raise Refusal(f"{path}: still not a complete json document after {tries} reads")
            time.sleep(pause)
    raise Refusal(f"{path}: unreadable")


class ShardStore:
    """teacher_cache/<teacher>/<corpus_sha>/<config_sha>/shard-NNNNN.{safetensors,json}; the json is written last."""

    def __init__(self, root: str | Path, teacher: str, corpus: dict[str, Any], config: dict[str, Any]):
        require(config.get("teacher") == teacher, f"run config is for {config.get('teacher')}, not {teacher}")
        self.teacher = teacher
        self.corpus = corpus
        self.config = config
        self.config_sha = sha256_bytes(canonical_json(config))
        self.dir = Path(root) / teacher / corpus["corpus_sha256"] / self.config_sha

    @classmethod
    def from_dir(cls, cache_dir: Path, teacher: str, corpus: dict[str, Any]) -> "ShardStore":
        """The store a sealed or sealing cache dir belongs to, from its own cache_config.json."""

        path = Path(cache_dir) / "cache_config.json"
        require(path.is_file(), f"{path} missing: not a {TOOL_VERSION}-layout cache (1.0.4 caches have no run config)")
        store = cls(Path(cache_dir).parent.parent.parent, teacher, corpus, json.loads(path.read_text(encoding="utf-8")))
        require(store.dir.resolve() == Path(cache_dir).resolve(), f"{cache_dir} is not {teacher}/{corpus['corpus_sha256']}/{store.config_sha}")
        return store

    def claim(self) -> None:
        claim_config(self.dir, "cache_config.json", self.config)

    def name(self, index: int) -> str:
        return f"shard-{int(index):05d}"

    def paths(self, index: int) -> tuple[Path, Path]:
        return self.dir / f"{self.name(index)}.safetensors", self.dir / f"{self.name(index)}.json"

    def compatible(self, body: dict[str, Any]) -> str | None:
        """None when a shard json belongs in this cache; otherwise why not."""

        if body.get("config_sha256") != self.config_sha:
            return f"config {str(body.get('config_sha256'))[:16]} != cache config {self.config_sha[:16]}"
        if body.get("tool_version") not in COMPAT[self.teacher]:
            return f"tool {body.get('tool_version')} not in COMPAT[{self.teacher}] {COMPAT[self.teacher]}"
        if body.get("teacher") != self.teacher or body.get("corpus_sha256") != self.corpus["corpus_sha256"]:
            return "written for another teacher or corpus"
        return None

    def done(self, index: int) -> bool:
        data, meta = self.paths(index)
        if not (data.is_file() and meta.is_file()):
            return False
        body = json.loads(meta.read_text(encoding="utf-8"))
        return self.compatible(body) is None and body.get("file_sha256") == sha256_file(data)

    def write(self, index: int, tensors: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        from safetensors.torch import save_file

        item = self.corpus["items"][index]
        for name, value in tensors.items():
            require(value.device.type == "cpu" and not value.requires_grad, f"{name}: shard tensors must be detached CPU tensors")
        finite_check(tensors)
        self.dir.mkdir(parents=True, exist_ok=True)
        data, jpath = self.paths(index)
        tmp = data.with_name(data.name + f".tmp{os.getpid()}")
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
        os.replace(tmp, data)
        body = {"schema": SHARD_SCHEMA, "teacher": self.teacher, "index": int(index), "item_id": item["id"],
                "item_sha256": item["sha256"], "corpus_sha256": self.corpus["corpus_sha256"], "file": data.name,
                "file_sha256": sha256_file(data), "bytes": data.stat().st_size, "finite": True,
                "tensors": {k: {"dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape)} for k, v in sorted(tensors.items())},
                **meta, "config_sha256": self.config_sha, "tool_version": TOOL_VERSION, "written_at_utc": utc_now()}
        write_json_atomic(jpath, body)
        return body


class X0Stage:
    """ACE x0 stage files: <root>/ace_dit_velocity/<corpus_sha>/x0stage-<x0 config sha>/x0-NNNNN.{safetensors,json}.

    Keyed on the x0-affecting arguments only (clip seconds, VAE device, tiled fallback), so ace-dit-run with another
    --cfg reuses the encodes. The json is written last and names the file sha, the item sha and the config sha."""

    def __init__(self, root: str | Path, corpus: dict[str, Any], knobs: dict[str, Any]):
        self.corpus = corpus
        self.config = {"schema": STAGE_SCHEMA, "teacher": "ace_dit_velocity", "compat_epoch": COMPAT["ace_dit_velocity"][0], "knobs": knobs}
        self.config_sha = sha256_bytes(canonical_json(self.config))
        self.dir = Path(root) / "ace_dit_velocity" / corpus["corpus_sha256"] / f"x0stage-{self.config_sha}"

    def claim(self) -> None:
        claim_config(self.dir, "stage_config.json", self.config)

    def paths(self, index: int) -> tuple[Path, Path]:
        return self.dir / f"x0-{int(index):05d}.safetensors", self.dir / f"x0-{int(index):05d}.json"

    def load(self, index: int) -> tuple[Any, dict[str, Any]] | None:
        """(x0, audio meta) when a valid stage file exists for this item, else None (it will be encoded)."""

        from safetensors.torch import load_file

        data, jpath = self.paths(index)
        if not (data.is_file() and jpath.is_file()):
            return None
        body = json.loads(jpath.read_text(encoding="utf-8"))
        item = self.corpus["items"][index]
        if (body.get("config_sha256") != self.config_sha or body.get("item_sha256") != item["sha256"]
                or body.get("tool_version") not in COMPAT["ace_dit_velocity"] or body.get("file_sha256") != sha256_file(data)):
            return None
        x0 = load_file(str(data))["x0"]
        finite_check({"x0": x0})
        return x0, body["audio"]

    def save(self, index: int, x0: Any, audio: dict[str, Any]) -> None:
        from safetensors.torch import save_file

        require(x0.device.type == "cpu" and not x0.requires_grad, "x0 stage tensor must be a detached CPU tensor")
        require_finite("x0 (Oobleck latents)", x0)
        self.dir.mkdir(parents=True, exist_ok=True)
        data, jpath = self.paths(index)
        tmp = data.with_name(data.name + f".tmp{os.getpid()}")
        save_file({"x0": x0.contiguous()}, str(tmp))
        os.replace(tmp, data)
        item = self.corpus["items"][index]
        write_json_atomic(jpath, {"schema": STAGE_SCHEMA, "index": int(index), "item_id": item["id"], "item_sha256": item["sha256"],
                                  "file_sha256": sha256_file(data), "config_sha256": self.config_sha, "audio": audio,
                                  "tool_version": TOOL_VERSION, "written_at_utc": utc_now()})


def teacher_integrity(files: list[Path], mode: str = "full") -> dict[str, Any]:
    """Full sha256 of every teacher file (DISTILL_DESIGN section 3); 'stamp' = size + mtime only (tests, probes)."""

    require(mode in ("full", "stamp"), f"integrity mode {mode}")
    rows = []
    for path in files:
        st = Path(path).stat()
        row = {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        if mode == "full":
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return {"mode": mode, "files": rows, "digest": sha256_bytes(canonical_json(rows)), "at_utc": utc_now()}


def teacher_files(teacher: str) -> list[Path]:
    if teacher == "hy3d_trajectories":
        return [HY3D_CKPT] + hy3d_source_files()
    if teacher == "ace_dit_velocity":
        return [ACE_DIT, ACE_VAE, ACE_TE, ACE_LM]
    if teacher == "ace_lm_codes":
        return [ACE_LM, ACE_TE]
    if teacher == "qvl32_vit_l13":
        return [QVL_WEIGHTS]
    raise Refusal(f"unknown teacher {teacher}")


def hy3d_source_files() -> list[Path]:
    root = HY3D_SOURCE
    files = [root / HY3D_CONFIG]
    if (root / HY3D_PACKAGE).is_dir():
        files += sorted(p for p in (root / HY3D_PACKAGE).rglob("*.py") if "__pycache__" not in p.parts)
    return files


def record_integrity(store: ShardStore, phase: str, rank: int, mode: str) -> dict[str, Any]:
    body = teacher_integrity(teacher_files(store.teacher), mode)
    write_json_atomic(store.dir / f"integrity.{phase}.rank{rank}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", body)
    return body


def check_summary(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Manifest roll-up of the per-shard reference and determinism arms (from each shard's meta_summary)."""

    refs = [s["meta"] for s in shards if "ref_max_abs_diff" in s["meta"]]
    reps = [s["meta"] for s in shards if "replay_max_abs_diff" in s["meta"]]
    return {"reference_arms": len(refs), "reference_bitwise_equal": sum(1 for m in refs if m.get("ref_bitwise")),
            "reference_max_abs_diff": max((m["ref_max_abs_diff"] for m in refs), default=None),
            "determinism_replays": len(reps), "determinism_bitwise_equal": sum(1 for m in reps if m.get("replay_bitwise")),
            "determinism_max_abs_diff": max((m["replay_max_abs_diff"] for m in reps), default=None)}


def seal(cache_dir: Path, corpus: dict[str, Any], teacher: str, *, mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify every shard (presence, sha, finite, run config, tool compat) and the teacher before/after equality,
    then write manifest.json with the run config."""

    from safetensors.torch import load_file

    store = ShardStore.from_dir(cache_dir, teacher, corpus)
    shards, missing, versions = [], [], {}
    for index, item in enumerate(corpus["items"]):
        data, jpath = store.paths(index)
        if not (data.is_file() and jpath.is_file()):
            missing.append(item["id"])
            continue
        body = json.loads(jpath.read_text(encoding="utf-8"))
        why = store.compatible(body)
        require(why is None, f"{data.name} does not belong in this cache: {why}")
        digest = sha256_file(data)
        require(digest == body["file_sha256"], f"{data.name}: sha256 changed since it was written")
        require(body["item_sha256"] == item["sha256"], f"{data.name}: written for another item")
        finite_check(load_file(str(data)))
        versions[body["tool_version"]] = versions.get(body["tool_version"], 0) + 1
        shards.append({"name": data.name, "index": index, "item_id": item["id"], "sha256": digest, "bytes": data.stat().st_size,
                       "finite": True, "config_sha256": body["config_sha256"], "tool_version": body["tool_version"],
                       "meta": body.get("meta_summary", {})})
    require(not missing, f"{len(missing)} shards missing: {missing[:5]}")
    befores = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(store.dir.glob("integrity.before.*.json"))]
    require(befores, "no teacher integrity record from any worker")
    after = teacher_integrity(teacher_files(teacher), mode)
    paths = [f["path"] for f in after["files"]]
    comparable = [b for b in befores if b["mode"] == mode and [f["path"] for f in b["files"]] == paths]
    superseded = [{"at_utc": b.get("at_utc"), "mode": b["mode"], "paths": [f["path"] for f in b["files"]]} for b in befores if b not in comparable]
    require(comparable, f"no before-record in mode {mode} over the teacher file set {paths}")
    equal = all(b["digest"] == after["digest"] for b in comparable)
    require(equal, "teacher files changed between the worker start and the seal")
    manifest = {
        "schema": CACHE_SCHEMA, "teacher": teacher, "tool": tool_identity(), "sealed_at_utc": utc_now(),
        "run_config": store.config, "config_sha256": store.config_sha, "shard_tool_versions": versions,
        "compat": list(COMPAT[teacher]), "checks": check_summary(shards),
        "corpus": {"sha256": corpus["corpus_sha256"], "file": corpus["path"], "file_sha256": corpus["file_sha256"],
                   "kind": corpus["kind"], "n_items": len(corpus["items"]), "contamination_screen": corpus.get("contamination_screen")},
        "shard_read_order": [s["name"] for s in shards], "shards": shards,
        "totals": {"shards": len(shards), "bytes": sum(s["bytes"] for s in shards)},
        "teacher_integrity": {"mode": mode, "before": comparable, "after": after, "equal": equal, "superseded_before_records": superseded},
        "finite": all(s["finite"] for s in shards),
        "capability_claim": "none; teacher outputs for distillation (diagnostic until a student is graded)",
        **(extra or {}),
    }
    digest = write_json_atomic(store.dir / "manifest.json", manifest, sidecar_sha=True)
    log(f"sealed {teacher}: {len(shards)} shards, {manifest['totals']['bytes']} B, manifest sha {digest[:16]}")
    return manifest


def verify(cache_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    mpath = cache_dir / "manifest.json"
    require(mpath.is_file(), f"{mpath} missing (not sealed)")
    side = (cache_dir / "manifest.json.sha256").read_text().strip()
    require(side == sha256_file(mpath), "manifest.json sha256 sidecar mismatch")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    config = manifest.get("run_config")
    require(config is not None, f"{mpath}: no run_config (a pre-1.0.5 cache: uniform targets cannot be shown)")
    digest = sha256_bytes(canonical_json(config))
    require(digest == manifest.get("config_sha256") == cache_dir.name, f"{cache_dir}: run config sha {digest[:16]} != manifest / dir name")
    on_disk = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    require(sha256_bytes(canonical_json(on_disk)) == digest, "cache_config.json differs from the manifest run_config")
    for shard in manifest["shards"]:
        data = cache_dir / shard["name"]
        require(sha256_file(data) == shard["sha256"], f"{shard['name']}: sha256 changed after the seal")
        body = json.loads(data.with_suffix(".json").read_text(encoding="utf-8"))
        require(body.get("config_sha256") == digest and body.get("tool_version") in manifest["compat"],
                f"{shard['name']}: config {str(body.get('config_sha256'))[:16]} / tool {body.get('tool_version')} not this cache's")
        finite_check(load_file(str(data)))
    log(f"verified {cache_dir}: {len(manifest['shards'])} shards")
    return {"verified": True, "shards": len(manifest["shards"]), "manifest_sha256": side}


# --------------------------------------------------------------------------
# pure math pieces (tested)
# --------------------------------------------------------------------------


def cfg_combine(cond: Any, uncond: Any, scale: float) -> Any:
    """uncond + scale * (cond - uncond); the upstream HY3D pipeline form (scale 1 returns cond exactly)."""

    if float(scale) == 1.0:
        return cond
    return uncond + float(scale) * (cond - uncond)


def hy3d_noise_level(t_native: float) -> float:
    """HY3D 2.1 flow time: sigmas run linspace(0, 1) from pure noise (t=0) to data (t=1)."""

    return 1.0 - float(t_native)


def euler_chain_residual(x_t: Any, v: Any, sigmas: Any) -> float:
    """max |x[i+1] - (x[i] + (s[i+1]-s[i]) v[i])| / max |x| over the stored trajectory (upstream scheduler.step)."""

    import torch

    if int(x_t.shape[0]) < 2:
        return 0.0
    s = sigmas.to(torch.float32)
    pred = x_t[:-1].float() + (s[1: x_t.shape[0]] - s[: x_t.shape[0] - 1]).view(-1, *([1] * (x_t.dim() - 1))) * v[:-1].float()
    scale = float(x_t.float().abs().max().clamp_min(1e-30))
    return float((x_t[1:].float() - pred).abs().max()) / scale


def flow_shift(t: float, shift: float) -> float:
    """comfy.model_sampling.time_snr_shift for flow models: shift*t / (1 + (shift-1)*t)."""

    return shift * t / (1.0 + (shift - 1.0) * t)


def ace_schedule(steps: int = ACE_STEPS, shift: float = ACE_SHIFT, timesteps: int = 1000) -> list[float]:
    """The deployed sampler's sigmas: ComfyUI 'simple' scheduler over ModelSamplingDiscreteFlow(shift, multiplier 1).

    model_sampling.sigmas[k] = shift-mapped (k+1)/timesteps; simple picks sigmas[-(1 + int(x * len/steps))]."""

    table = [flow_shift((k + 1) / timesteps, shift) for k in range(timesteps)]
    stride = len(table) / steps
    return [table[-(1 + int(x * stride))] for x in range(steps)]


def ace_clip_sigmas(index: int, schedule: list[float], per_clip: int = 4) -> list[float]:
    """Clip i takes schedule points (i + 2k) mod steps, k < per_clip: two consecutive clips cover all 8 points."""

    n = len(schedule)
    stride = max(1, n // per_clip)
    return [schedule[(index + stride * k) % n] for k in range(per_clip)]


def band_topk(logits: Any, *, start: int = ACE_AUDIO_START, end: int = ACE_AUDIO_END, k: int = 64,
              eos: int = ACE_EOS) -> dict[str, Any]:
    """Top-k over the audio-code band of one logit row (fp32 math). ids are CODE indices (token - start);
    logprobs are log-softmax over the FULL vocabulary; lse_band - lse_full = log mass inside the band."""

    import torch

    row = logits.reshape(-1).float()
    lse_full = torch.logsumexp(row, dim=0)
    band = row[start:end]
    lse_band = torch.logsumexp(band, dim=0)
    vals, ids = torch.topk(band, int(k))
    return {"ids": ids.to(torch.int32), "logprobs": (vals - lse_full), "lse_full": lse_full, "lse_band": lse_band,
            "eos_logprob": row[eos] - lse_full}


def smart_resize(height: int, width: int, *, factor: int = 32, min_pixels: int = 64 * 32 * 32,
                 max_pixels: int = 512 * 512) -> tuple[int, int]:
    """Qwen2/3-VL processor smart_resize: multiples of `factor`, area clamped to [min_pixels, max_pixels]."""

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


# --------------------------------------------------------------------------
# P8: safetensors header read (no torch)
# --------------------------------------------------------------------------


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        require(0 < length < (1 << 30), f"{path}: implausible header length {length}")
        return json.loads(handle.read(length))


def header_summary(header: dict[str, Any]) -> dict[str, Any]:
    """Is there an lm_head, how many decoder layers, how many vision blocks, parameter counts by group."""

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    def numel(shape):
        n = 1
        for s in shape:
            n *= int(s)
        return n
    groups: dict[str, int] = {}
    layers: set[int] = set()
    vblocks: set[int] = set()
    dtypes: dict[str, int] = {}
    for name, spec in tensors.items():
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
        top = name.split(".")[0] if not name.startswith("model.") else ".".join(name.split(".")[:2])
        groups[top] = groups.get(top, 0) + numel(spec["shape"])
        m = re.search(r"(?:^|\.)(?:language_model\.)?layers\.(\d+)\.", name)
        if m and not name.startswith("visual.") and ".visual." not in name:
            layers.add(int(m.group(1)))
        b = re.search(r"visual\.blocks\.(\d+)\.", name)
        if b:
            vblocks.add(int(b.group(1)))
    lm_head = sorted(n for n in tensors if "lm_head" in n)
    embed = sorted(n for n in tensors if n.endswith("embed_tokens.weight"))
    return {
        "tensors": len(tensors), "dtypes": dtypes, "parameters": sum(numel(v["shape"]) for v in tensors.values()),
        "parameters_by_group": dict(sorted(groups.items())), "has_lm_head": bool(lm_head), "lm_head_tensors": lm_head,
        "lm_head_shapes": {n: tensors[n]["shape"] for n in lm_head}, "embed_tokens": {n: tensors[n]["shape"] for n in embed},
        "decoder_layers": len(layers), "decoder_layer_max_index": max(layers) if layers else None,
        "vision_blocks": len(vblocks), "metadata": header.get("__metadata__"),
    }


def run_p8(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.weights)
    st = path.stat()
    header = read_safetensors_header(path)
    summary = header_summary(header)
    verdict = ("full VLM with an lm_head: usable as a second answer teacher (design weight 0.3)" if summary["has_lm_head"]
               else "no lm_head: a text-encoder copy (vision tower + decoder stack without head) -> ViT feature source only")
    body = {"schema": "omni-campaign.s0_p8_header.v1", "probe": "P8", "tool": tool_identity(), "file": str(path), "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns, "header_bytes_sha256": sha256_bytes(canonical_json(header)), "summary": summary,
            "verdict": verdict, "at_utc": utc_now()}
    write_json_atomic(Path(args.out), body, sidecar_sha=True)
    log(f"P8: lm_head={summary['has_lm_head']} decoder_layers={summary['decoder_layers']} vision_blocks={summary['vision_blocks']} "
        f"params={summary['parameters']:,} -> {args.out}")
    return body


# --------------------------------------------------------------------------
# GPU guard + launch (host side)
# --------------------------------------------------------------------------


def parse_nvidia_smi(text: str) -> dict[int, dict[str, Any]]:
    rows = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows[int(parts[0])] = {"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "used_mib": int(float(parts[3])),
                               "total_mib": int(float(parts[4]))}
    return rows


def query_gpus() -> dict[int, dict[str, Any]]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True, timeout=60).stdout
    return parse_nvidia_smi(out)


def check_card(card: int, gpus: dict[int, dict[str, Any]]) -> dict[str, Any]:
    require(card not in FORBIDDEN_CARDS, f"GPU {card} refused: {FORBIDDEN_CARDS.get(card)}")
    require(card in ALLOWED_CARDS, f"GPU {card} is not one of this lane's P40s {ALLOWED_CARDS}")
    require(card in gpus, f"GPU {card} not reported by nvidia-smi")
    row = gpus[card]
    require(row["used_mib"] <= MAX_START_MIB, f"GPU {card} already uses {row['used_mib']} MiB (> {MAX_START_MIB}): not starting")
    return row


def docker_argv(*, card: int | None, name: str, tool_args: list[str], uid: int, gid: int, threads: int = 8,
                image: str = RUNTIME_IMAGE, extra_mounts: list[str] | None = None, tool_path: str | None = None) -> list[str]:
    """One container per card; it sees exactly one GPU (no P2P path exists), nice 15, <= 16 threads.
    card=None: a CPU-only container (no --gpus, NVIDIA_VISIBLE_DEVICES=void, CUDA_VISIBLE_DEVICES empty)."""

    if card is None:
        require(tool_args and tool_args[0] in CPU_WORKERS, f"a CPU-only container runs only {CPU_WORKERS}")
    else:
        require(card in ALLOWED_CARDS and card not in FORBIDDEN_CARDS, f"GPU {card} refused")
    threads = int(threads)
    require(1 <= threads <= MAX_THREADS, f"threads {threads} outside 1..{MAX_THREADS}")
    mounts = [f"{OC}:{OC}:rw", f"{MODELS}:{MODELS}:ro", f"{HY3D_SOURCE}:{HY3D_SOURCE}:ro",
              "/mnt/lexar/omni-campaign:/mnt/lexar/omni-campaign:ro",
              "/home/cereal/omnicoder_2026_work/weights/media_artifacts_2026:/home/cereal/omnicoder_2026_work/weights/media_artifacts_2026:ro",
              "/mnt/ai_data/omnicoder_2026/media_artifacts:/mnt/ai_data/omnicoder_2026/media_artifacts:ro"]
    mounts += list(extra_mounts or [])
    env = {"HOME": "/tmp", "XDG_CACHE_HOME": "/tmp/.cache", "TORCH_HOME": "/tmp/.torch", "MPLCONFIGDIR": "/tmp/.mpl",
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "MEDIA_B_PHYSICAL_GPU": str(card if card is not None else "none"),
           "OC": str(OC), "COMFY_ROOT": str(COMFY_ROOT)}
    if card is None:
        env.update({"MEDIA_B_CPU_ONLY": "1", "NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ""})
    if os.environ.get("CHIMERA_OMNI_SRC"):
        env["CHIMERA_OMNI_SRC"] = os.environ["CHIMERA_OMNI_SRC"]
    argv = ["docker", "run", "--rm", "--name", name] + (["--gpus", f"device={card}"] if card is not None else [])
    argv += ["--cpus", str(threads), "--shm-size", "8g", "--user", f"{uid}:{gid}"]
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    for mount in mounts:
        argv += ["-v", mount]
    argv += ["--workdir", "/tmp", "--entrypoint", "/usr/bin/nice", image, "-n", "15", CONTAINER_PYTHON, "-u",
             tool_path or str(Path(__file__).resolve()), *tool_args]
    return argv


def pascal_argv(*, card: int, tool_args: list[str], threads: int = 8, tool_path: str | None = None) -> tuple[list[str], dict[str, str]]:
    require(card in ALLOWED_CARDS and card not in FORBIDDEN_CARDS, f"GPU {card} refused")
    require(1 <= int(threads) <= MAX_THREADS, f"threads {threads} outside 1..{MAX_THREADS}")
    env = dict(os.environ)
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card), "MEDIA_B_PHYSICAL_GPU": str(card),
                "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return ["nice", "-n", "15", PASCAL_PYTHON, "-u", tool_path or str(Path(__file__).resolve()), *tool_args], env


def strip_dashdash(values: list[str]) -> list[str]:
    values = list(values)
    return values[1:] if values and values[0] == "--" else values


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def worker_write_mounts(tool_args: list[str]) -> list[str]:
    """The extra read-write docker mount a worker's --cache-root needs: none under $OC (mounted rw already); the
    root itself when it lies under TEACHER_CACHE_WRITE_ROOTS (created first, so docker does not create it); any
    other root is refused, since the container could not write it and a silent fallback would lose the cache."""

    tool_args = strip_dashdash(tool_args)
    ns = build_parser().parse_args(tool_args)
    root = getattr(ns, "cache_root", None)
    if not root:
        return []
    root = Path(os.path.abspath(root))
    if _under(root, Path(os.path.abspath(OC))):
        return []
    require(any(_under(root, base) for base in TEACHER_CACHE_WRITE_ROOTS),
            f"--cache-root {root} is outside $OC and outside {[str(b) for b in TEACHER_CACHE_WRITE_ROOTS]}")
    root.mkdir(parents=True, exist_ok=True)
    return [f"{root}:{root}:rw"]


def run_launch(args: argparse.Namespace) -> dict[str, Any]:
    tool_args = strip_dashdash(args.worker_args)
    require(tool_args and tool_args[0] in WORKERS, f"launch needs a worker subcommand {sorted(WORKERS)}")
    if args.cpu_only:
        require(args.card is None, "--cpu-only takes no --card")
        require(args.runtime == "docker", "--cpu-only runs in the docker runtime")
        card, row = None, {"name": "cpu only (no card)", "uuid": None, "used_mib": None}
    else:
        require(args.card is not None, "launch needs --card (or --cpu-only for a CPU worker)")
        require(tool_args[0] not in CPU_WORKERS, f"{tool_args[0]} is CPU-only: launch it with --cpu-only, not on a card")
        card = int(args.card)
        row = check_card(card, query_gpus())
    mounts = worker_write_mounts(tool_args)
    if args.runtime == "docker":
        argv = docker_argv(card=card, name=args.name, tool_args=tool_args, uid=os.getuid(), gid=os.getgid(), threads=args.threads,
                           extra_mounts=mounts)
        env = dict(os.environ)
    else:
        argv, env = pascal_argv(card=card, tool_args=tool_args, threads=args.threads)
    alloc = OC / "composite/receipts/allocator" / f"{args.name}.json"
    receipt = {"lane": f"media-b teacher caches ({tool_args[0]})", "gpu": card, "card": row["name"], "uuid": row["uuid"],
               "used_mib_at_start": row["used_mib"], "cpu_only": bool(args.cpu_only), "by": "claude media-b lane", "tool": TOOL_NAME,
               "tool_version": TOOL_VERSION, "runtime": args.runtime, "started_utc": utc_now(), "log": str(args.log), "argv": argv}
    write_json_atomic(alloc, receipt)
    log(f"launch {args.name} on {'CPU only' if card is None else f'GPU {card}'} ({row['name']}, {row['used_mib']} MiB used) -> {args.log}")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "ab") as handle:
        code = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
    receipt.update(ended_utc=utc_now(), exit_code=code)
    write_json_atomic(alloc, receipt)
    log(f"launch {args.name}: exit {code}")
    return {"exit_code": code, "allocator": str(alloc), "rows": 0}


# --------------------------------------------------------------------------
# worker plumbing (inside a runtime)
# --------------------------------------------------------------------------


def _torch():
    import torch

    torch.set_num_threads(max(1, min(MAX_THREADS, int(os.environ.get("OMP_NUM_THREADS", "8")))))
    return torch


def device_receipt() -> dict[str, Any]:
    """The worker must see exactly ONE card (one process per card, no P2P)."""

    torch = _torch()
    require(torch.cuda.is_available(), "no CUDA device visible")
    require(torch.cuda.device_count() == 1, f"worker sees {torch.cuda.device_count()} cards; exactly one is required")
    props = torch.cuda.get_device_properties(0)
    return {"physical_gpu": os.environ.get("MEDIA_B_PHYSICAL_GPU"), "name": props.name, "uuid": str(getattr(props, "uuid", "")),
            "capability": [props.major, props.minor], "total_mib": props.total_memory // (1 << 20),
            "torch": torch.__version__, "cuda": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(),
            "python": sys.version.split()[0], "executable": sys.executable}


def mib(n: int) -> float:
    return round(n / (1 << 20), 1)


def mem_snapshot() -> dict[str, float]:
    torch = _torch()
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_mib": mib(torch.cuda.max_memory_allocated()), "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
            "device_used_mib_now": mib(total - free)}


def host_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return -1.0


def timed(fn, *, repeat: int = 3, warmup: int = 1) -> tuple[Any, list[float]]:
    torch = _torch()
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return out, times


def rank_items(corpus: dict[str, Any], rank: int, world: int, limit: int | None) -> list[int]:
    indices = [i for i in range(len(corpus["items"])) if i % int(world) == int(rank)]
    return indices[: int(limit)] if limit else indices


# --------------------------------------------------------------------------
# HY3D (native hy3dshape source, fp32)
# --------------------------------------------------------------------------


def hy3d_native(source_root: Path):
    """Load the genuine hy3dshape source under a private namespace (the package __init__ imports ComfyUI UI code;
    namespace containers give paths only, every computational module executes its own file). The same pattern as
    Codex's c16_capture_hy3d.native_modules."""

    root = Path(source_root) / HY3D_PACKAGE
    require(root.is_dir(), f"{root} missing")
    prefix = "_mediab_hy3d_native"
    for suffix in ("", ".models", ".models.autoencoders", ".models.denoisers"):
        name = prefix + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*suffix.strip(".").split("."))) if suffix else str(root)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module
    names = {"dit": "models.denoisers.hunyuandit", "conditioner": "models.conditioner", "image": "preprocessors",
             "vae": "models.autoencoders.model", "schedulers": "schedulers"}
    return types.SimpleNamespace(**{k: importlib.import_module(f"{prefix}.{v}") for k, v in names.items()})


def hy3d_pipeline_class(native: Any) -> Any:
    """The upstream Hunyuan3DDiTFlowMatchingPipeline from the SAME source tree (pipelines.py), for the reference arm.

    pipelines.py imports comfy.utils / comfy.model_management (ComfyUI must be importable) and does
    `from .models.autoencoders import ShapeVAE, SurfaceExtractors`; the private namespace package does not execute
    autoencoders/__init__.py, so those two names are bound from their own genuine modules first."""

    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    prefix = "_mediab_hy3d_native"
    autoencoders = sys.modules[prefix + ".models.autoencoders"]
    autoencoders.ShapeVAE = native.vae.ShapeVAE
    autoencoders.SurfaceExtractors = importlib.import_module(prefix + ".models.autoencoders.surface_extractors").SurfaceExtractors
    return importlib.import_module(prefix + ".pipelines").Hunyuan3DDiTFlowMatchingPipeline


def strict_fp32(module: Any, checkpoint: Path, prefix: str, device: str) -> Any:
    """Load exactly this component's tensors of the combined F16 checkpoint, widened to fp32 (exact)."""

    torch = _torch()
    from safetensors import safe_open

    expected = module.state_dict()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = {k[len(prefix):] for k in handle.keys() if k.startswith(prefix)}
        require(keys == set(expected), f"strict {prefix} key mismatch: missing {sorted(set(expected) - keys)[:4]} extra {sorted(keys - set(expected))[:4]}")
        state = {}
        for name, ref in expected.items():
            value = handle.get_tensor(prefix + name)
            require(list(value.shape) == list(ref.shape), f"strict shape mismatch {prefix}{name}")
            state[name] = value.to(torch.float32)
    module.load_state_dict(state, strict=True)
    del state
    module.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    return module


class Hy3dTeacher:
    def __init__(self, *, device: str = "cuda:0", source_root: Path = HY3D_SOURCE, checkpoint: Path = HY3D_CKPT):
        torch = _torch()
        import yaml

        started = time.time()
        self.device = device
        self.config = yaml.safe_load((source_root / HY3D_CONFIG).read_text(encoding="utf-8"))
        self.native = hy3d_native(source_root)
        with torch.device(device):
            dit = self.native.dit.HunYuanDiTPlain(**self.config["model"]["params"])
        self.model = strict_fp32(dit, checkpoint, "model.", device)
        with torch.device(device):
            cond = self.native.conditioner.SingleImageEncoder(**self.config["conditioner"]["params"])
        self.conditioner = strict_fp32(cond, checkpoint, "conditioner.", device)
        with torch.device(device):
            vae = self.native.vae.ShapeVAE(**self.config["vae"]["params"])
        self.vae = strict_fp32(vae, checkpoint, "vae.", device)
        self.processor = self.native.image.ImageProcessorV2(**self.config["image_processor"]["params"])
        self.load_seconds = time.time() - started
        self.params = {k: sum(int(p.numel()) for p in m.parameters()) for k, m in
                       (("dit", self.model), ("conditioner", self.conditioner), ("vae", self.vae))}

    def scheduler(self, steps: int):
        import numpy as np

        sched = self.native.schedulers.FlowMatchEulerDiscreteScheduler(**self.config["scheduler"]["params"])
        # Hunyuan3DDiTFlowMatchingPipeline: sigmas = np.linspace(0, 1, steps); retrieve_timesteps(..., sigmas=sigmas)
        sched.set_timesteps(sigmas=np.linspace(0, 1, int(steps)), device=self.device)
        return sched

    def condition(self, image_path: str) -> dict[str, Any]:
        torch = _torch()
        inputs = self.processor(str(image_path))
        image = inputs["image"].to(self.device, torch.float32)
        mask = inputs["mask"].to(self.device)
        with torch.inference_mode():
            cond = self.conditioner(image=image, mask=mask)
            uncond = self.conditioner.unconditional_embedding(image.shape[0])
        require(set(cond) == {"main"}, "native single-image conditioning required")
        uncond = uncond if isinstance(uncond, dict) else {"main": uncond}
        return {"image": image, "mask": mask, "cond": cond["main"], "uncond": uncond["main"],
                "contexts": {"main": torch.cat([cond["main"], uncond["main"]], dim=0).to(torch.float32)}}

    def velocity(self, latents: Any, t: Any, contexts: dict[str, Any], guidance: float, num_train: int) -> Any:
        torch = _torch()
        inp = torch.cat([latents] * 2)
        timestep = t.expand(inp.shape[0]).to(latents.dtype) / num_train
        with torch.inference_mode():
            pred = self.model(inp, timestep, contexts, guidance=None)
        cond, uncond = pred.chunk(2)
        return cfg_combine(cond, uncond, guidance), timestep[0]

    def trajectory(self, image_path: str, *, seed: int, steps: int, guidance: float) -> dict[str, Any]:
        torch = _torch()
        from diffusers.utils.torch_utils import randn_tensor

        c = self.condition(image_path)
        sched = self.scheduler(steps)
        num_train = int(sched.config.num_train_timesteps)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = randn_tensor((1, *self.vae.latent_shape), generator=generator, device=torch.device(self.device), dtype=torch.float32)
        latents = latents * getattr(sched, "init_noise_sigma", 1.0)
        xs, vs, ts, times = [], [], [], []
        for t in sched.timesteps:
            start = time.perf_counter()
            v, t_native = self.velocity(latents, t, c["contexts"], guidance, num_train)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            xs.append(latents[0].detach().cpu().clone())
            vs.append(v[0].detach().float().cpu().clone())
            ts.append(float(t_native))
            latents = sched.step(v, t, latents).prev_sample
        sigmas = sched.sigmas.detach().float().cpu()
        return {"x_t": torch.stack(xs), "v": torch.stack(vs), "t_native": torch.tensor(ts, dtype=torch.float32),
                "sigmas": sigmas, "x0": latents[0].detach().cpu().clone(), "forward_seconds": times, "cond": c}

    def upstream_latents(self, image_path: str, *, seed: int, steps: int, guidance: float) -> tuple[Any, float]:
        """Reference arm: the upstream pipeline's own __call__ (it returns the final latents) built on the SAME loaded
        modules, a fresh scheduler from the same config, and the same seed. Independent of this tool's loop."""

        torch = _torch()
        cls = hy3d_pipeline_class(self.native)
        sched = self.native.schedulers.FlowMatchEulerDiscreteScheduler(**self.config["scheduler"]["params"])
        pipe = cls(vae=self.vae, model=self.model, scheduler=sched, conditioner=self.conditioner,
                   image_processor=self.processor, device=self.device, dtype=torch.float32)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        start = time.perf_counter()
        latents = pipe(image=str(image_path), num_inference_steps=int(steps), guidance_scale=float(guidance),
                       generator=generator, enable_pbar=False)
        torch.cuda.synchronize()
        return latents[0].detach().float().cpu(), time.perf_counter() - start

    def mesh(self, x0: Any, *, octree: int, chunks: int) -> dict[str, Any]:
        torch = _torch()
        start = time.perf_counter()
        with torch.inference_mode():
            decoded = self.vae((1.0 / self.vae.scale_factor) * x0.unsqueeze(0).to(self.device))
            outputs = self.vae.latents2mesh(decoded, bounds=HY3D_BOX_V, mc_level=HY3D_MC_LEVEL, num_chunks=int(chunks),
                                            octree_resolution=int(octree), mc_algo="mc", enable_pbar=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        out = outputs[0] if outputs else None
        if out is None:
            return {"vertices": torch.zeros(0, 3), "faces": torch.zeros(0, 3, dtype=torch.int32), "seconds": seconds, "ok": False}
        import numpy as np

        vertices = torch.from_numpy(np.ascontiguousarray(out.mesh_v, dtype=np.float32))
        faces = torch.from_numpy(np.ascontiguousarray(out.mesh_f[:, ::-1], dtype=np.int32))  # export_to_trimesh flips winding
        return {"vertices": vertices, "faces": faces, "seconds": seconds, "ok": True}


def write_ply(path: Path, vertices: Any, faces: Any) -> str:
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    mesh.export(str(tmp), file_type="ply")
    os.replace(tmp, path)
    return sha256_file(path)


def run_hy3d(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    require(corpus["kind"] == "hy3d_images", f"{args.corpus}: kind {corpus['kind']} is not hy3d_images")
    store = ShardStore(args.cache_root, "hy3d_trajectories", corpus, run_config("hy3d_trajectories", args))
    store.claim()
    log(f"hy3d cache {store.dir}")
    record_integrity(store, "before", args.rank, args.integrity)
    teacher = Hy3dTeacher(device="cuda:0")
    # import the upstream pipeline (and with it ComfyUI's model_management) BEFORE the first trajectory, always, so
    # every trajectory of every run executes in the same process state whether or not a reference arm runs
    hy3d_pipeline_class(teacher.native)
    log(f"hy3d teacher loaded in {teacher.load_seconds:.1f}s params {teacher.params} on {dev['name']} gpu {dev['physical_gpu']}")
    meshes_ok, rows = [], 0
    reference_left = int(args.reference_first)
    for index in rank_items(corpus, args.rank, args.world, args.limit):
        item = corpus["items"][index]
        if store.done(index):
            log(f"hy3d {store.name(index)} exists, skipped")
            continue
        require(sha256_file(item["path"]) == item["sha256"], f"{item['path']}: image changed since the corpus was built")
        torch.cuda.reset_peak_memory_stats()
        traj = teacher.trajectory(item["path"], seed=item["seed"], steps=args.steps, guidance=args.guidance)
        steps = int(traj["x_t"].shape[0])
        residual = euler_chain_residual(traj["x_t"], traj["v"], traj["sigmas"])
        replay = None
        if args.replay_every and index % int(args.replay_every) == 0:
            k = steps // 2
            sched = teacher.scheduler(args.steps)
            v2, _ = teacher.velocity(traj["x_t"][k:k + 1].to(teacher.device), sched.timesteps[k], traj["cond"]["contexts"],
                                     args.guidance, int(sched.config.num_train_timesteps))
            v2 = v2[0].float().cpu()
            replay = {"kind": "determinism (this tool's velocity() again on the stored x_t[k]; not a parent comparison)", "step": k,
                      "bitwise_equal": bool(torch.equal(v2, traj["v"][k])), "max_abs_diff": float((v2 - traj["v"][k]).abs().max())}
        reference = None
        if reference_left > 0:
            reference_left -= 1
            ref_x0, ref_seconds = teacher.upstream_latents(item["path"], seed=item["seed"], steps=args.steps, guidance=args.guidance)
            x0 = traj["x0"].float()
            ref_diff = float((ref_x0 - x0).abs().max())
            reference = {"kind": "reference: upstream Hunyuan3DDiTFlowMatchingPipeline.__call__ final latents vs x0_final (same image, seed, steps, guidance)",
                         "pipeline_file": str(HY3D_SOURCE / HY3D_PACKAGE / "pipelines.py"),
                         "pipeline_sha256": sha256_file(HY3D_SOURCE / HY3D_PACKAGE / "pipelines.py"),
                         "bitwise_equal": bool(torch.equal(ref_x0, x0)), "max_abs_diff": ref_diff, "x0_absmax": float(x0.abs().max()),
                         "seconds": ref_seconds}
            log(f"hy3d reference arm {item['id']}: upstream pipeline bitwise {reference['bitwise_equal']} max_abs_diff {ref_diff:.3e}")
            require(ref_diff <= 1e-4 * max(1.0, float(x0.abs().max())),
                    f"HY3D trajectory diverges from the upstream pipeline: max_abs_diff {ref_diff}")
        mesh = teacher.mesh(traj["x0"], octree=args.octree_resolution, chunks=args.num_chunks)
        mem = mem_snapshot()
        faces = int(mesh["faces"].shape[0])
        meshes_ok.append(faces > 0)
        tensors = {
            "x_t": traj["x_t"].float(), "v_guided": traj["v"].float(), "t_native": traj["t_native"],
            "noise_level": 1.0 - traj["t_native"], "sigma_next": traj["sigmas"][1: steps + 1].clone(),
            "x0_final": traj["x0"].float(), "context_cond": traj["cond"]["cond"][0].float().cpu(),
            "image_input": traj["cond"]["image"][0].float().cpu(), "image_mask": traj["cond"]["mask"][0].float().cpu(),
            "mesh_vertices": mesh["vertices"], "mesh_faces": mesh["faces"],
        }
        ply_sha = None
        if faces > 0:
            ply_sha = write_ply(store.dir / f"{store.name(index)}.mesh.ply", mesh["vertices"], mesh["faces"])
        meta = {"item": {k: item.get(k) for k in ("id", "path", "sha256", "caption", "seed", "held")},
                "steps": steps, "guidance": args.guidance, "time_convention": "t_native: model timestep input, 0 = pure noise, 1 = data (HY3D 2.1 flow); noise_level = 1 - t_native",
                "forward_seconds_cfg_batch2": traj["forward_seconds"],
                "checks": {"recording_euler_chain_residual_rel": residual,
                           "recording_note": "re-derives x[i+1] from the stored x[i], v[i], sigmas: proves the stored tensors are the ones stepped, not that the loop matches the parent",
                           "determinism_replay": replay, "reference_upstream_pipeline": reference},
                "mesh": {"faces": faces, "vertices": int(mesh["vertices"].shape[0]), "seconds": mesh["seconds"], "octree_resolution": args.octree_resolution,
                         "num_chunks": args.num_chunks, "ply": f"{store.name(index)}.mesh.ply" if ply_sha else None, "ply_sha256": ply_sha},
                "memory": mem, "host_rss_hwm_mib": host_rss_mib(), "device": dev,
                "meta_summary": {"faces": faces, "held": bool(item.get("held")), "s_per_cfg_forward": sum(traj["forward_seconds"]) / steps,
                                 **({"replay_max_abs_diff": replay["max_abs_diff"], "replay_bitwise": replay["bitwise_equal"]} if replay else {}),
                                 **({"ref_max_abs_diff": reference["max_abs_diff"], "ref_bitwise": reference["bitwise_equal"]} if reference else {})}}
        store.write(index, tensors, meta)
        rows += steps
        log(f"hy3d {store.name(index)} {item['id']}: {steps} pairs, {sum(traj['forward_seconds']):.1f}s DiT, mesh faces {faces} "
            f"({mesh['seconds']:.1f}s), peak {mem['peak_allocated_mib']} MiB, recording residual {residual:.2e}")
    record_integrity(store, "after", args.rank, args.integrity)
    if args.require_mesh:
        require(meshes_ok and all(meshes_ok), f"a decoded reference mesh has 0 faces ({meshes_ok})")
    return {"rows": rows, "cache_dir": str(store.dir)}


def run_hy3d_probe(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    item = corpus["items"][0]
    torch.cuda.reset_peak_memory_stats()
    teacher = Hy3dTeacher(device="cuda:0")
    load_mem = mem_snapshot()
    torch.cuda.reset_peak_memory_stats()
    c, t_cond = timed(lambda: teacher.condition(item["path"]), repeat=args.repeat)
    cond_mem = mem_snapshot()
    sched = teacher.scheduler(HY3D_STEPS)
    gen = torch.Generator(device="cuda:0").manual_seed(int(item["seed"]))
    x = torch.randn((1, *teacher.vae.latent_shape), generator=gen, device="cuda:0", dtype=torch.float32)
    t_mid = sched.timesteps[HY3D_STEPS // 2]
    num_train = int(sched.config.num_train_timesteps)
    torch.cuda.reset_peak_memory_stats()
    _, t_cfg = timed(lambda: teacher.velocity(x, t_mid, c["contexts"], HY3D_GUIDANCE, num_train), repeat=args.repeat)
    cfg_mem = mem_snapshot()
    torch.cuda.reset_peak_memory_stats()
    single = {"main": c["cond"].float()}
    def one():
        with torch.inference_mode():
            return teacher.model(x, (t_mid / num_train).expand(1).float(), single, guidance=None)
    _, t_one = timed(one, repeat=args.repeat)
    one_mem = mem_snapshot()
    torch.cuda.reset_peak_memory_stats()
    mesh = teacher.mesh(x[0], octree=args.octree_resolution, chunks=args.num_chunks)
    mesh_mem = mem_snapshot()
    body = {"schema": PROBE_SCHEMA, "probe": "P4", "teacher": "hunyuan3d_2.1", "tool": tool_identity(), "device": dev,
            "dtype": "fp32 (F16 checkpoint widened exactly)", "params": teacher.params, "load_seconds": teacher.load_seconds,
            "image": {k: item[k] for k in ("id", "sha256")},
            "s_per_forward": {"dit_cfg_batch2": sum(t_cfg) / len(t_cfg), "dit_single_batch1": sum(t_one) / len(t_one),
                              "conditioner_with_preprocess": sum(t_cond) / len(t_cond), "vae_decode_plus_mesh": mesh["seconds"]},
            "timings_seconds": {"dit_cfg_batch2": t_cfg, "dit_single_batch1": t_one, "conditioner": t_cond},
            "peak_vram": {"after_load": load_mem, "conditioner": cond_mem, "dit_cfg_batch2": cfg_mem, "dit_single": one_mem, "mesh_decode": mesh_mem},
            "mesh_from_random_latent": {"faces": int(mesh["faces"].shape[0]), "note": "decode-cost probe on a NOISE latent; not a shape"},
            "latent_tokens": list(teacher.vae.latent_shape), "context_tokens": list(c["cond"].shape),
            "host_rss_hwm_mib": host_rss_mib(),
            "derived": {"cache_wall_one_card_s": (sum(t_cfg) / len(t_cfg)) * args.cache_pairs + mesh["seconds"] * (args.cache_pairs / HY3D_STEPS),
                        "cache_pairs": args.cache_pairs, "how": "s/cfg-forward x pairs + mesh decode per shape; divide by cards for the 3-P40 wall"},
            "at_utc": utc_now()}
    body["derived"]["cache_wall_three_p40_h"] = body["derived"]["cache_wall_one_card_s"] / 3 / 3600
    write_json_atomic(Path(args.out), body, sidecar_sha=True)
    log(f"P4 hy3d: cfg forward {body['s_per_forward']['dit_cfg_batch2']:.3f}s, single {body['s_per_forward']['dit_single_batch1']:.3f}s, "
        f"peak {cfg_mem['peak_allocated_mib']} MiB, mesh {mesh['seconds']:.1f}s -> {args.out}")
    return {"rows": 0}


# --------------------------------------------------------------------------
# ACE-Step 1.5 (ComfyUI's own model code, in-process)
# --------------------------------------------------------------------------


def comfy_modules():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    if os.environ.get("MEDIA_B_CPU_ONLY") == "1":
        # comfy_kitchen's triton backend guards `import triton` with `except ImportError`, but with no GPU driver
        # triton raises RuntimeError("0 active drivers") at import; make the import fail as ImportError instead so
        # the backend reports itself unavailable (its kernels are GPU quantized ops; the CPU fp32 encode never uses them).
        if "triton" not in sys.modules:
            sys.modules["triton"] = None
        import comfy.cli_args  # model_management reads args.cpu at import

        comfy.cli_args.args.cpu = True
    import comfy.model_management as mm
    import comfy.sd
    import comfy.text_encoders.ace15 as te15
    import comfy.utils

    return types.SimpleNamespace(sd=comfy.sd, utils=comfy.utils, mm=mm, te15=te15)


def comfy_identity() -> dict[str, Any]:
    files = ["comfy/ldm/ace/ace_step15.py", "comfy/text_encoders/ace15.py", "comfy/model_base.py", "comfy/model_sampling.py",
             "comfy_extras/nodes_audio.py", "comfy_extras/nodes_ace.py"]
    return {"root": str(COMFY_ROOT), "sha256": {f: sha256_file(COMFY_ROOT / f) for f in files if (COMFY_ROOT / f).is_file()}}


class AceTeacher:
    """dit: load the fp32 DiT now (else lazily via load_dit()); vae: build the Oobleck VAE (default: with the DiT);
    text: load the dual text encoder (TE + LM)."""

    def __init__(self, *, dit: bool, lm: bool, device: str = "cuda:0", vae_device: str = "cpu", vae: bool | None = None,
                 text: bool = True):
        torch = _torch()
        self.c = comfy_modules()
        self.device = torch.device(device)
        started = time.time()
        self.patcher = self.model = self.dm = None
        self.dit_load_seconds = None
        if dit:
            self.load_dit()
        # dual load always (DualCLIPLoader semantics); `lm` only says whether the LM will be sampled
        paths = [str(ACE_TE), str(ACE_LM)]
        self.wants_lm = bool(lm)
        self.clip = (self.c.sd.load_clip(ckpt_paths=paths, clip_type=self.c.sd.CLIPType.ACE,
                                         model_options={"load_device": self.device, "offload_device": torch.device("cpu"), "dtype": torch.float32})
                     if text else None)
        self.vae_device = torch.device(vae_device)
        self.vae = (self.c.sd.VAE(sd=self.c.utils.load_torch_file(str(ACE_VAE)), device=self.vae_device, dtype=torch.float32)
                    if (dit if vae is None else vae) else None)
        self.load_seconds = time.time() - started

    def load_dit(self) -> None:
        """Load the fp32 DiT (first call) and put it fully on the card; ace-dit-run calls this only after the CPU
        encode and the text phase, so the card is not pinned while clips encode."""

        torch = _torch()
        if self.patcher is None:
            started = time.time()
            self.patcher = self.c.sd.load_diffusion_model(str(ACE_DIT), model_options={"dtype": torch.float32})
            self.model = self.patcher.model
            self.dm = self.model.diffusion_model
            ms = self.model.model_sampling
            require(abs(float(ms.timestep(torch.tensor(0.5))) - 0.5) < 1e-7, "ACE model_sampling.timestep(sigma) != sigma (multiplier)")
            require(abs(float(ms.shift) - ACE_SHIFT) < 1e-9, f"ACE sampling shift {ms.shift} != {ACE_SHIFT}")
            self.c.mm.load_models_gpu([self.patcher], force_full_load=True)
            self.dit_load_seconds = time.time() - started
        else:
            self.c.mm.load_models_gpu([self.patcher], force_full_load=True)
        dtypes = {str(p.dtype) for p in self.dm.parameters()}
        devices = {p.device.type for p in self.dm.parameters()}
        require(dtypes == {"torch.float32"} and devices == {"cuda"}, f"ACE DiT not fp32 on cuda: {dtypes} {devices}")

    def ensure_dit_loaded(self) -> None:
        """Re-assert the DiT fully on the card (a no-op when resident; the VAE/TE loads may have evicted it)."""

        require(self.patcher is not None, "ACE DiT used before load_dit()")
        self.load_dit()

    def ensure_clip_loaded(self) -> Any:
        """Load the TE patcher on the card and set the execution device the LM loop reads (encode paths do this)."""

        self.c.mm.load_models_gpu([self.clip.patcher], force_full_load=True)
        self.clip.cond_stage_model.set_clip_options({"execution_device": self.device})
        lm_name = getattr(self.clip.cond_stage_model, "lm_model", None)
        return getattr(self.clip.cond_stage_model, lm_name) if lm_name else None

    # ---- conditioning (TextEncodeAceStepAudio1.5 semantics)

    def tokens(self, caption: str, lyrics: str, *, seconds: float, seed: int, generate_codes: bool) -> dict[str, Any]:
        d = ACE_NODE_DEFAULTS
        return self.clip.tokenize(caption, lyrics=lyrics, bpm=d["bpm"], duration=float(seconds), timesignature=int(d["timesignature"]),
                                  language=d["language"], keyscale=d["keyscale"], seed=int(seed), generate_audio_codes=bool(generate_codes),
                                  **ACE_LM_DEFAULTS)

    def encode_text(self, caption: str, lyrics: str, *, seconds: float, seed: int) -> dict[str, Any]:
        tokens = self.tokens(caption, lyrics, seconds=seconds, seed=seed, generate_codes=False)
        with _torch().inference_mode():
            out = self.clip.encode_from_tokens(tokens, return_dict=True)
        require("cond" in out and "conditioning_lyrics" in out, f"ACE TE output keys {sorted(out)}")
        require_finite("ACE text conditioning", out["cond"])
        require_finite("ACE lyric conditioning", out["conditioning_lyrics"])
        return out

    # ---- audio -> Oobleck latents (VAEEncodeAudio semantics)

    def load_window(self, path: str, *, seconds: float, seed: int) -> tuple[Any, dict[str, Any]]:
        torch = _torch()
        import torchaudio
        from comfy_extras.nodes_audio import load as comfy_audio_load

        wav, sr = comfy_audio_load(str(path))
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        require(wav.shape[0] == 2, f"{path}: {wav.shape[0]} channels")
        vae_sr = int(getattr(self.vae, "audio_sample_rate", 44100))
        if sr != vae_sr:
            wav = torchaudio.functional.resample(wav, sr, vae_sr)
        total = wav.shape[1]
        want = int(round(seconds * vae_sr))
        if total > want:
            offset = int(item_seed(str(seed), "ace-window") % (total - want + 1))
            wav = wav[:, offset: offset + want]
        else:
            offset = 0
        return wav, {"source_sample_rate": sr, "vae_sample_rate": vae_sr, "source_seconds": total / vae_sr,
                     "window_offset_s": offset / vae_sr, "window_seconds": wav.shape[1] / vae_sr}

    def encode_audio(self, wav: Any, *, seed: int, allow_tiled: bool = False) -> tuple[Any, dict[str, Any]]:
        """VAEEncodeAudio: vae.encode(waveform.movedim(1, -1)), with every other model unloaded first so the
        regular (untiled) encode has the card; ComfyUI's silent tiled fallback is detected from its log record.

        The Oobleck bottleneck returns a POSTERIOR SAMPLE (comfy/ldm/audio/autoencoder.py vae_sample:
        randn_like(mean) * stdev + mean) from the global RNG, so an unseeded encode is not reproducible (measured
        2026-10-13: the same waveform twice in one process, max |dx0| 2.7e-3, rel_fro ~5e-4). The sample is drawn
        from the VAE device's generator seeded with `seed`, inside fork_rng so no other RNG state changes."""

        import logging

        if self.vae_device.type == "cuda":
            self.c.mm.unload_all_models()
            self.c.mm.soft_empty_cache()
        seen: list[str] = []

        class Catch(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        handler = Catch(level=logging.WARNING)
        logging.getLogger().addHandler(handler)
        torch = _torch()
        devices = [self.vae_device.index or 0] if self.vae_device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                if devices:
                    torch.cuda.manual_seed(int(seed))
                else:
                    torch.random.default_generator.manual_seed(int(seed))
                x0 = self.vae.encode(wav.unsqueeze(0).movedim(1, -1))
        finally:
            logging.getLogger().removeHandler(handler)
        tiled = any("tiled" in m.lower() for m in seen)
        require(allow_tiled or not tiled, "ComfyUI fell back to TILED VAE encoding (out of memory); refused (--allow-tiled-vae records it)")
        require_finite("x0 (Oobleck latents)", x0)
        return x0, {"tiled_fallback": tiled, "vae_warnings": seen[:5], "vae_dtype": str(getattr(self.vae, "vae_dtype", "?")),
                    "posterior_seed": int(seed),
                    "vae_device": str(self.vae_device)}

    def first_nonfinite_module(self, x_t: Any, sigma: float, cond: dict[str, Any]) -> dict[str, Any]:
        """Diagnostic: rerun once with forward hooks on every DiT submodule; the first (in call order) whose output
        holds a non-finite value is named, with the input's finiteness."""

        torch = _torch()
        order: list[dict[str, Any]] = []
        handles = []
        for name, module in self.dm.named_modules():
            def hook(m, a, o, name=name):
                outs = o if isinstance(o, (tuple, list)) else (o,)
                bad = any(torch.is_tensor(t) and t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in outs)
                ins = [t for t in a if torch.is_tensor(t) and t.is_floating_point()]
                if bad:
                    order.append({"module": name or "<root>", "type": type(m).__name__,
                                  "inputs_finite": all(bool(torch.isfinite(t).all()) for t in ins)})
            handles.append(module.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                self.model.apply_model(x_t, torch.tensor([float(sigma)], device=self.device, dtype=torch.float32), **cond)
        finally:
            for h in handles:
                h.remove()
        first = next((r for r in order if r["inputs_finite"]), order[0] if order else None)
        return {"nonfinite_modules": len(order), "first_with_finite_inputs": first, "first_in_completion_order": order[:3]}

    def codes(self, x0: Any) -> Any:
        """The parent's own 5 Hz semantic codes of x0 (AceStepAudioTokenizer.tokenize -> ResidualFSQ indices)."""

        torch = _torch()
        self.ensure_dit_loaded()
        with torch.inference_mode():
            _, indices = self.dm.tokenizer.tokenize(x0.to(self.device, torch.float32).movedim(-1, -2))
        require(indices.shape[-1] == 1, f"ACE tokenizer quantizers {indices.shape}")
        return indices[0, :, 0].to(torch.int64)

    def conds(self, x_t: Any, cross_attn: Any, lyrics: Any, codes: list[int]) -> dict[str, Any]:
        out = self.model.extra_conds(device=self.device, noise=x_t, cross_attn=cross_attn, conditioning_lyrics=lyrics, audio_codes=[codes])
        return {k: v.cond for k, v in out.items()}

    def velocity(self, x_t: Any, sigma: float, cond: dict[str, Any], uncond: dict[str, Any] | None, cfg: float) -> tuple[Any, dict[str, Any]]:
        """ComfyUI apply_model with a forward hook on the diffusion model: v = the raw DiT output (CONST: denoised = x - sigma v)."""

        torch = _torch()
        self.ensure_dit_loaded()
        captured = []
        handle = self.dm.register_forward_hook(lambda m, a, o: captured.append(o.detach()))
        try:
            s = torch.tensor([float(sigma)], device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                den_c = self.model.apply_model(x_t, s, **cond)
                v_c = captured.pop()
                parity = float((den_c - (x_t - v_c * float(sigma))).abs().max())
                if uncond is not None and float(cfg) != 1.0:
                    self.model.apply_model(x_t, s, **uncond)
                    v_u = captured.pop()
                    v = cfg_combine(v_c, v_u, cfg)
                else:
                    v = v_c
        finally:
            handle.remove()
        if not bool(torch.isfinite(v).all()):
            diag = self.first_nonfinite_module(x_t, sigma, cond)
            raise Refusal(f"non-finite ACE DiT velocity at sigma {sigma}: {json.dumps(json_safe(diag), default=str)[:1500]}")
        return v, {"denoised_vs_x_minus_sigma_v_max_abs": parity}


def stage_x0(stage: X0Stage, teacher: AceTeacher, index: int, item: dict[str, Any], args: argparse.Namespace) -> tuple[Any, dict[str, Any], bool]:
    """(x0 [CPU fp32], audio meta, loaded_from_stage). Encodes and writes the stage file when none is valid."""

    got = stage.load(index)
    if got is not None:
        return got[0], got[1], True
    require(sha256_file(item["path"]) == item["sha256"], f"{item['path']}: clip changed since the corpus was built")
    wav, audio_meta = teacher.load_window(item["path"], seconds=args.clip_seconds, seed=item["seed"])
    start = time.perf_counter()
    x0, vae_meta = teacher.encode_audio(wav, seed=item_seed(item["sha256"], ACE_VAE_POSTERIOR_SALT), allow_tiled=args.allow_tiled_vae)
    audio = {**audio_meta, **vae_meta, "encode_seconds": time.perf_counter() - start}
    x0 = x0.float().cpu()
    stage.save(index, x0, audio)
    return x0, audio, False


def run_ace_encode(args: argparse.Namespace) -> dict[str, Any]:
    """Phase A of ace-dit-run alone, CPU only (`launch --cpu-only`): the regular fp32 Oobleck encode of every clip
    window of this rank into sha-checked x0 stage files. No card is claimed; ace-dit-run then only loads them."""

    torch = _torch()
    require(device_type(args.vae_device) == "cpu", "ace-encode is the CPU encode (--vae-device cpu)")
    if os.environ.get("MEDIA_B_CPU_ONLY") == "1":
        require(not torch.cuda.is_available(), "a CPU-only ace-encode container can see a CUDA device")
    corpus = read_corpus(args.corpus)
    require(corpus["kind"] == "ace_clips", f"{args.corpus}: kind {corpus['kind']} is not ace_clips")
    stage = X0Stage(args.cache_root, corpus, ace_x0_knobs(args))
    stage.claim()
    teacher = AceTeacher(dit=False, lm=False, device="cpu", vae_device="cpu", vae=True, text=False)
    log(f"ace-encode: stage {stage.dir} (VAE on CPU, {torch.get_num_threads()} threads, cuda visible {torch.cuda.is_available()})")
    encoded = loaded = 0
    for index in rank_items(corpus, args.rank, args.world, args.limit):
        x0, audio, was = stage_x0(stage, teacher, index, corpus["items"][index], args)
        loaded += int(was)
        encoded += int(not was)
        what = "stage valid, kept" if was else "encoded in %.1fs" % audio["encode_seconds"]
        log(f"ace-encode x0-{index:05d}: {what}, T {x0.shape[-1]}")
    return {"rows": encoded + loaded, "encoded": encoded, "already_staged": loaded, "stage_dir": str(stage.dir)}


def run_ace_dit(args: argparse.Namespace) -> dict[str, Any]:
    """Three phases so no model evicts another mid-item: (A) Oobleck encode of every clip window (CPU by default),
    each written to an x0 stage file as it finishes (or loaded from one: ace-encode, or an earlier crashed run),
    (B) text/lyric conditioning, (C) the DiT, loaded only now: codes + guided velocities."""

    torch = _torch()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    require(corpus["kind"] == "ace_clips", f"{args.corpus}: kind {corpus['kind']} is not ace_clips")
    store = ShardStore(args.cache_root, "ace_dit_velocity", corpus, run_config("ace_dit_velocity", args))
    store.claim()
    stage = X0Stage(args.cache_root, corpus, ace_x0_knobs(args))
    stage.claim()
    log(f"ace-dit cache {store.dir}; x0 stage {stage.dir.name}")
    record_integrity(store, "before", args.rank, args.integrity)
    teacher = AceTeacher(dit=False, lm=False, vae_device=args.vae_device, vae=True)
    schedule = ace_schedule()
    log(f"ace VAE + TE built in {teacher.load_seconds:.1f}s on {dev['name']} (DiT not loaded yet); schedule {['%.4f' % s for s in schedule]}")
    todo = [i for i in rank_items(corpus, args.rank, args.world, args.limit) if not store.done(i)]
    staged: dict[int, dict[str, Any]] = {}
    from_stage = 0
    for index in todo:  # phase A
        x0, audio, was = stage_x0(stage, teacher, index, corpus["items"][index], args)
        staged[index] = {"x0": x0, "audio": audio, "x0_from_stage": was}
        from_stage += int(was)
    log(f"ace-dit phase A: {len(todo)} clips, {from_stage} x0 loaded from stage files, {len(todo) - from_stage} encoded")
    for index in todo:  # phase B
        item = corpus["items"][index]
        text = teacher.encode_text(item.get("caption", ""), item.get("lyrics", ""), seconds=staged[index]["audio"]["window_seconds"], seed=item["seed"])
        staged[index]["cond"] = text["cond"].float().cpu()
        staged[index]["lyr"] = text["conditioning_lyrics"].float().cpu()
    if todo:
        teacher.load_dit()
        log(f"ace DiT loaded in {teacher.dit_load_seconds:.1f}s")
    rows = 0
    for index in todo:  # phase C
        item = corpus["items"][index]
        st = staged.pop(index)
        torch.cuda.reset_peak_memory_stats()
        x0 = st["x0"].to(teacher.device)
        codes = teacher.codes(x0)
        cond_tokens = st["cond"].to(teacher.device)
        lyr = st["lyr"].to(teacher.device)
        sigmas = ace_clip_sigmas(index, schedule, per_clip=args.sigmas_per_clip)
        xs, vs, seeds, parity, secs = [], [], [], [], []
        for k, sigma in enumerate(sigmas):
            seed = item_seed(item["sha256"], f"ace-noise-{k}")
            gen = torch.Generator(device=teacher.device).manual_seed(seed)
            noise = torch.randn(x0.shape, generator=gen, device=teacher.device, dtype=torch.float32)
            x_t = teacher.model.model_sampling.noise_scaling(torch.tensor([sigma], device=teacher.device), noise, x0)
            cond = teacher.conds(x_t, cond_tokens, lyr, codes.tolist())
            uncond = teacher.conds(x_t, torch.zeros_like(cond_tokens), lyr, codes.tolist()) if args.cfg != 1.0 else None
            start = time.perf_counter()
            v, check = teacher.velocity(x_t, sigma, cond, uncond, args.cfg)
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - start)
            xs.append(x_t[0].float().cpu())
            vs.append(v[0].float().cpu())
            seeds.append(seed)
            parity.append(check["denoised_vs_x_minus_sigma_v_max_abs"])
        require(max(parity) <= 1e-4 * max(1.0, float(x0.abs().max())), f"ComfyUI denoised != x - sigma*v (max {max(parity)})")
        mem = mem_snapshot()
        tensors = {"x0": x0[0].float().cpu(), "codes": codes.to(torch.int32).cpu(), "sigma": torch.tensor(sigmas, dtype=torch.float32),
                   "x_t": torch.stack(xs), "v_guided": torch.stack(vs), "noise_seed": torch.tensor(seeds, dtype=torch.int64),
                   "text_cond": cond_tokens[0].float().cpu(), "lyric_embed": lyr[0].float().cpu()}
        meta = {"item": {k: item.get(k) for k in ("id", "path", "sha256", "caption", "lyrics", "seed", "held", "dataset")},
                "audio": st["audio"], "cfg": args.cfg, "sigmas": sigmas, "latent_frames": int(x0.shape[-1]), "code_frames": int(codes.shape[0]),
                "convention": "x_t = sigma*noise + (1-sigma)*x0 (comfy CONST.noise_scaling); v = raw DiT output; denoised = x_t - sigma*v",
                "x0_stage": {"dir": stage.dir.name, "loaded_from_stage": st["x0_from_stage"]},
                "checks": {"hook_capture_denoised_vs_x_minus_sigma_v_max_abs": parity,
                           "hook_capture_note": "proves the forward hook captured the tensor ComfyUI's apply_model uses; not that x0, codes or conditioning match the deployed path"},
                "forward_seconds": secs, "memory": mem, "device": dev,
                "meta_summary": {"held": bool(item.get("held")), "latent_frames": int(x0.shape[-1]), "s_per_guided_forward": sum(secs) / len(secs),
                                 "tiled_vae": st["audio"]["tiled_fallback"]}}
        store.write(index, tensors, meta)
        rows += len(sigmas)
        log(f"ace-dit {store.name(index)}: T {x0.shape[-1]} codes {codes.shape[0]} sigmas {['%.3f' % s for s in sigmas]} {sum(secs):.2f}s")
    record_integrity(store, "after", args.rank, args.integrity)
    return {"rows": rows, "cache_dir": str(store.dir)}


def lm_sample_recording(model: Any, ids: list[list[int]], **kwargs: Any) -> dict[str, Any]:
    """Run the recording loop under torch.inference_mode() (ComfyUI's executor context)."""

    with _torch().inference_mode():
        return _lm_sample_recording(model, ids, **kwargs)


def _lm_sample_recording(model: Any, ids: list[list[int]], *, record_k: int, seed: int, min_tokens: int, max_new_tokens: int,
                         cfg_scale: float, temperature: float, top_p: float, top_k: int, min_p: float) -> dict[str, Any]:
    """comfy/text_encoders/ace15.py sample_manual_loop_no_classes @ b1693ec, line for line, plus a recording hook
    that keeps the CONDITIONAL branch's band top-k at every position (before CFG/filtering touch the logits)."""

    torch = _torch()
    import comfy.model_management

    audio_start_id, audio_end_id, eos_token_id = ACE_AUDIO_START, ACE_AUDIO_END, ACE_EOS
    device = model.execution_device
    execution_dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(ids, device)
    embeds_batch = embeds.shape[0]
    output_audio_codes, records = [], {"ids": [], "logprobs": [], "lse_full": [], "lse_band": [], "eos_logprob": []}
    past_key_values = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model_config = model.transformer.model.config
    past_kv_shape = [embeds_batch, model_config.num_key_value_heads, embeds.shape[1] + min_tokens, model_config.head_dim]
    for _ in range(model_config.num_hidden_layers):
        past_key_values.append((torch.empty(past_kv_shape, device=device, dtype=execution_dtype),
                                torch.empty(past_kv_shape, device=device, dtype=execution_dtype), 0))
    eos_hit = False
    for step in range(max_new_tokens):
        outputs = model.transformer(None, attention_mask, embeds=embeds.to(execution_dtype), num_tokens=num_tokens, intermediate_output=None,
                                    dtype=execution_dtype, embeds_info=embeds_info, past_key_values=past_key_values)
        next_token_logits = model.transformer.logits(outputs[0])[:, -1]
        past_key_values = outputs[2]
        rec = band_topk(next_token_logits[0], k=record_k)
        for key in records:
            records[key].append(rec[key].detach().float().cpu() if key != "ids" else rec[key].detach().cpu())
        if cfg_scale != 1.0:
            cond_logits = next_token_logits[0:1]
            uncond_logits = next_token_logits[1:2]
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        else:
            cfg_logits = next_token_logits[0:1]
        use_eos_score = eos_token_id is not None and eos_token_id < audio_start_id and min_tokens < step
        if use_eos_score:
            eos_score = cfg_logits[:, eos_token_id].clone()
        remove_logit_value = torch.finfo(cfg_logits.dtype).min
        cfg_logits[:, :audio_start_id] = remove_logit_value
        cfg_logits[:, audio_end_id:] = remove_logit_value
        if use_eos_score:
            cfg_logits[:, eos_token_id] = eos_score
        if top_k is not None and top_k > 0:
            top_k_vals, _ = torch.topk(cfg_logits, top_k)
            min_val = top_k_vals[..., -1, None]
            cfg_logits[cfg_logits < min_val] = remove_logit_value
        if min_p is not None and min_p > 0:
            probs = torch.softmax(cfg_logits, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            indices_to_remove = probs < (min_p * p_max)
            cfg_logits[indices_to_remove] = remove_logit_value
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            cfg_logits[indices_to_remove] = remove_logit_value
        if temperature > 0:
            cfg_logits = cfg_logits / temperature
            next_token = torch.multinomial(torch.softmax(cfg_logits, dim=-1), num_samples=1, generator=generator).squeeze(1)
        else:
            next_token = torch.argmax(cfg_logits, dim=-1)
        token = next_token.item()
        if token == eos_token_id:
            eos_hit = True
            break
        embed, _, _, _ = model.process_tokens([[token]], device)
        embeds = embed.repeat(embeds_batch, 1, 1)
        attention_mask = torch.cat([attention_mask, torch.ones((embeds_batch, 1), device=device, dtype=attention_mask.dtype)], dim=1)
        output_audio_codes.append(token - audio_start_id)
    n = len(output_audio_codes)
    # the record at the EOS step has no code token; keep exactly one record per emitted code
    stacked = {k: torch.stack(v[:n]) if n else torch.zeros(0) for k, v in records.items()}
    return {"codes": output_audio_codes, "records": stacked, "eos_hit": eos_hit, "execution_dtype": str(execution_dtype)}


def lm_ids(teacher: AceTeacher, tokens: dict[str, Any], cfg_scale: float) -> list[list[int]]:
    """generate_audio_codes' positive/negative id construction (pad-left to equal length)."""

    lm = getattr(teacher.clip.cond_stage_model, teacher.clip.cond_stage_model.lm_model)
    positive = [[t for t, _ in inner] for inner in tokens["lm_prompt"]][0]
    if cfg_scale == 1.0:
        return [positive]
    negative = [[t for t, _ in inner] for inner in tokens["lm_prompt_negative"]][0]
    if len(negative) < len(positive):
        negative = [lm.special_tokens["pad"]] * (len(positive) - len(negative)) + negative
    if len(negative) > len(positive):
        positive = [lm.special_tokens["pad"]] * (len(negative) - len(positive)) + positive
    return [positive, negative]


def run_ace_lm(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    require(corpus["kind"] in ("ace_clips", "ace_prompts"), f"{args.corpus}: kind {corpus['kind']}")
    store = ShardStore(args.cache_root, "ace_lm_codes", corpus, run_config("ace_lm_codes", args))
    store.claim()
    log(f"ace-lm cache {store.dir}")
    record_integrity(store, "before", args.rank, args.integrity)
    teacher = AceTeacher(dit=False, lm=True)
    lm = teacher.ensure_clip_loaded()
    require(lm is not None, "ACE TE loaded without its LM (qwen_1.7b_ace15)")
    lm_dtypes = {str(p.dtype) for p in lm.parameters()}
    lm_devices = {p.device.type for p in lm.parameters()}
    log(f"ace lm loaded in {teacher.load_seconds:.1f}s: {teacher.clip.cond_stage_model.lm_model} dtypes {lm_dtypes} devices {lm_devices}")
    require(lm_devices == {"cuda"}, f"ACE LM not on the card: {lm_devices}")
    s = ACE_LM_DEFAULTS
    parity = None
    rows = 0
    for index in rank_items(corpus, args.rank, args.world, args.limit):
        item = corpus["items"][index]
        if store.done(index):
            continue
        seconds = float(min(args.max_seconds, max(args.min_seconds, float(item.get("seconds") or args.max_seconds))))
        tokens = teacher.tokens(item.get("caption", ""), item.get("lyrics", ""), seconds=seconds, seed=item["seed"], generate_codes=True)
        meta_lm = tokens["lm_metadata"]
        ids = lm_ids(teacher, tokens, s["cfg_scale"])
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = lm_sample_recording(lm, ids, record_k=args.topk, seed=meta_lm["seed"], min_tokens=meta_lm["min_tokens"],
                                  max_new_tokens=meta_lm["min_tokens"], cfg_scale=s["cfg_scale"], temperature=s["temperature"],
                                  top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"])
        torch.cuda.synchronize()
        seconds_used = time.perf_counter() - start
        reference = None
        if parity is None and args.parity_first:
            with torch.inference_mode():
                ref = teacher.c.te15.generate_audio_codes(lm, tokens["lm_prompt"], tokens["lm_prompt_negative"], min_tokens=meta_lm["min_tokens"],
                                                           max_tokens=meta_lm["min_tokens"], seed=meta_lm["seed"], cfg_scale=s["cfg_scale"],
                                                           temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], min_p=s["min_p"])
            parity = {"kind": "reference: ComfyUI comfy.text_encoders.ace15.generate_audio_codes, same prompt and seed",
                      "item": item["id"], "codes_equal": list(ref) == list(out["codes"]), "n_ours": len(out["codes"]), "n_comfy": len(ref)}
            reference = parity
            require(parity["codes_equal"], f"transcribed LM loop diverged from comfy generate_audio_codes: {parity}")
        n = len(out["codes"])
        require(n > 0, f"{item['id']}: the LM emitted no code tokens")
        r = out["records"]
        tensors = {"tokens": torch.tensor(out["codes"], dtype=torch.int32), "topk_ids": r["ids"].to(torch.int32),
                   "topk_logprobs": r["logprobs"].to(torch.bfloat16), "lse_full": r["lse_full"].float(), "lse_band": r["lse_band"].float(),
                   "eos_logprob": r["eos_logprob"].float(), "prompt_ids": torch.tensor(ids[0], dtype=torch.int32)}
        require(int(tensors["tokens"].min()) >= 0 and int(tensors["tokens"].max()) < ACE_CODES, "sampled token outside the 64,000 codes")
        mem = mem_snapshot()
        band_mass = float((r["lse_band"] - r["lse_full"]).exp().mean())
        meta = {"item": {k: item.get(k) for k in ("id", "sha256", "caption", "lyrics", "seed", "held")}, "seconds": seconds,
                "lm_param_dtypes": sorted(lm_dtypes), "lm_metadata": meta_lm, "sampling": s, "eos_hit": out["eos_hit"], "execution_dtype": out["execution_dtype"],
                "band": {"start": ACE_AUDIO_START, "end": ACE_AUDIO_END, "student_rows": "C18 band rows [252416, 316416) = codes [0, 64000)"},
                "recorded": "conditional branch logits (row 0) at every emitted position: top-k over the band, log-softmax over the full vocab",
                "tok_per_s": n / seconds_used, "wall_seconds": seconds_used, "mean_band_mass": band_mass,
                "checks": {"reference_comfy_generate_audio_codes": reference},
                "memory": mem, "device": dev, "meta_summary": {"tokens": n, "tok_per_s": n / seconds_used, "held": bool(item.get("held")),
                                                               **({"ref_max_abs_diff": 0.0 if reference["codes_equal"] else float("inf"),
                                                                   "ref_bitwise": reference["codes_equal"]} if reference else {})}}
        store.write(index, tensors, meta)
        rows += n
        log(f"ace-lm {store.name(index)}: {n} codes in {seconds_used:.1f}s ({n / seconds_used:.1f} tok/s), band mass {band_mass:.4f}, peak {mem['peak_allocated_mib']} MiB")
    record_integrity(store, "after", args.rank, args.integrity)
    return {"rows": rows, "cache_dir": str(store.dir), "parity": parity}


def run_ace_probe(args: argparse.Namespace) -> dict[str, Any]:
    """P4 for ACE: VAE encode (alone on the card), TE, DiT single + CFG 2, LM with KV at CFG 2. Each phase records its
    own error instead of taking the receipt down with it; any failed phase makes the probe exit 3 after writing."""

    import traceback

    torch = _torch()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    item = corpus["items"][0]
    teacher = AceTeacher(dit=True, lm=True, vae_device=args.vae_device)
    load_mem = mem_snapshot()
    results: dict[str, Any] = {}
    state: dict[str, Any] = {}

    def phase(name, fn):
        torch.cuda.reset_peak_memory_stats()
        try:
            results[name] = fn()
            results[name]["memory"] = mem_snapshot()
        except Exception as exc:  # recorded; the probe still writes its receipt
            results[name] = {"error": f"{type(exc).__name__}: {exc}"[:3000], "traceback": traceback.format_exc()[-3000:], "memory": mem_snapshot()}
            log(f"P4 ace phase {name} FAILED: {type(exc).__name__}: {str(exc)[:300]}")

    def vae():
        wav, audio_meta = teacher.load_window(item["path"], seconds=args.clip_seconds, seed=item["seed"])
        start = time.perf_counter()
        x0, meta = teacher.encode_audio(wav, seed=item_seed(item["sha256"], ACE_VAE_POSTERIOR_SALT), allow_tiled=True)
        secs = time.perf_counter() - start
        state["x0"], state["audio"] = x0.float().cpu(), audio_meta
        return {"seconds": secs, "latent_shape": list(x0.shape), **meta, "audio": audio_meta, "x0_absmax": float(x0.abs().max())}

    def te():
        text, t = timed(lambda: teacher.encode_text(item.get("caption", ""), item.get("lyrics", ""), seconds=state["audio"]["window_seconds"], seed=1),
                        repeat=args.repeat)
        state["cond"], state["lyr"] = text["cond"].float().cpu(), text["conditioning_lyrics"].float().cpu()
        return {"seconds": t, "cond_shape": list(text["cond"].shape), "lyric_shape": list(text["conditioning_lyrics"].shape)}

    def dit(cfg):
        def run():
            x0 = state["x0"].to(teacher.device)
            codes = teacher.codes(x0)
            ct, lyr = state["cond"].to(teacher.device), state["lyr"].to(teacher.device)
            sigma = ace_schedule()[3]
            noise = torch.randn(x0.shape, generator=torch.Generator(device=teacher.device).manual_seed(0), device=teacher.device)
            x_t = teacher.model.model_sampling.noise_scaling(torch.tensor([sigma], device=teacher.device), noise, x0)
            cond = teacher.conds(x_t, ct, lyr, codes.tolist())
            uncond = teacher.conds(x_t, torch.zeros_like(ct), lyr, codes.tolist())
            (v, check), t = timed(lambda: teacher.velocity(x_t, sigma, cond, uncond, cfg), repeat=args.repeat)
            return {"cfg": cfg, "seconds": t, "s_per_forward": sum(t) / len(t), "parity": check, "sigma": sigma,
                    "latent_frames": int(x0.shape[-1]), "code_frames": int(codes.shape[0]), "v_absmax": float(v.abs().max())}
        return run

    def lm():
        teacher.c.mm.unload_all_models()  # as in ace-lm-run: the DiT is not on the card while the LM samples
        teacher.c.mm.soft_empty_cache()
        lm_model = teacher.ensure_clip_loaded()
        require(lm_model is not None, "ACE TE loaded without its LM (qwen_1.7b_ace15)")
        tokens = teacher.tokens(item.get("caption", ""), item.get("lyrics", ""), seconds=args.lm_seconds, seed=1, generate_codes=True)
        ids = lm_ids(teacher, tokens, ACE_LM_DEFAULTS["cfg_scale"])
        start = time.perf_counter()
        out = lm_sample_recording(lm_model, ids, record_k=64, seed=1, min_tokens=tokens["lm_metadata"]["min_tokens"],
                                  max_new_tokens=tokens["lm_metadata"]["min_tokens"], **ACE_LM_DEFAULTS)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        n = len(out["codes"])
        r = out["records"]
        return {"tokens": n, "seconds": wall, "tok_per_s": n / wall if wall and n else None, "prompt_tokens": len(ids[0]),
                "execution_dtype": out["execution_dtype"], "eos_hit": out["eos_hit"],
                "records_finite": bool(n and torch.isfinite(r["logprobs"]).all() and torch.isfinite(r["lse_full"]).all()),
                "mean_band_mass": float((r["lse_band"] - r["lse_full"]).exp().mean()) if n else None,
                "lm_dtypes": sorted({str(p.dtype) for p in lm_model.parameters()}), "lm_devices": sorted({p.device.type for p in lm_model.parameters()})}

    phase("vae_encode", vae)
    if "x0" in state:
        phase("text_encoder", te)
    if "cond" in state:
        phase("dit_single", dit(1.0))
        phase("dit_cfg2", dit(2.0))
    phase("lm_cfg2_kv", lm)

    def rate(key):
        r = results.get(key, {})
        return r.get("s_per_forward") if "error" not in r else None

    lm_rate = (results.get("lm_cfg2_kv") or {}).get("tok_per_s")
    derived = {"dit_cache_1024x4_cfg1_h_one_card": 1024 * 4 * rate("dit_single") / 3600 if rate("dit_single") else None,
               "dit_cache_1024x4_cfg2_h_one_card": 1024 * 4 * rate("dit_cfg2") / 3600 if rate("dit_cfg2") else None,
               "lm_2M_tokens_h_one_card": 2.0e6 / lm_rate / 3600 if lm_rate else None,
               "lm_2M_tokens_h_three_p40": 2.0e6 / lm_rate / 3600 / 3 if lm_rate else None}
    failed = sorted(k for k, v in results.items() if "error" in v)
    body = {"schema": PROBE_SCHEMA, "probe": "P4", "teacher": "ace_step_1.5_turbo", "tool": tool_identity(), "device": dev, "comfy": comfy_identity(),
            "dtype": "fp32 (comfy should_use_fp16/bf16 False on sm61)", "load_seconds": teacher.load_seconds, "after_load": load_mem,
            "clip": {k: item.get(k) for k in ("id", "sha256")}, "results": results, "derived": derived, "failed_phases": failed,
            "host_rss_hwm_mib": host_rss_mib(), "at_utc": utc_now()}
    write_json_atomic(Path(args.out), body, sidecar_sha=True)
    log(f"P4 ace -> {args.out}; failed phases {failed}; dit single {rate('dit_single')} cfg2 {rate('dit_cfg2')} lm {lm_rate} tok/s")
    require(not failed, f"P4 ace phases failed: {failed}")
    return {"rows": 0}


# --------------------------------------------------------------------------
# QVL32 ViT block-13 in/out pairs (Pascal venv, transformers 5.12.1)
# --------------------------------------------------------------------------


def ref_row_stats(ours: Any, ref: Any) -> dict[str, Any]:
    """Row-tensor agreement in float64: bitwise, max_abs, |ref| max, whole-tensor relative error (rel_fro) and the
    minimum per-row cosine. rel_fro is the gate for a cross-device reference; max_abs is a diagnostic."""

    import torch

    a, b = ours.double(), ref.double()
    delta = a - b
    return {"bitwise_equal": bool(torch.equal(ours, ref)), "max_abs_diff": float(delta.abs().max()), "ref_absmax": float(b.abs().max()),
            "rel_fro": float(delta.norm() / b.norm().clamp_min(1e-30)),
            "row_cosine_min": float(torch.nn.functional.cosine_similarity(a, b, dim=1).min())}


def s7_forward_slice(forwards: list[dict[str, Any]], sha_by_path: dict[str, str], item_sha: str, h: int, w: int) -> tuple[int, int, int] | None:
    """(forward index, first row, rows) of the s7 capture forward of the image with content sha `item_sha` at a square
    side equal to h == w; None when the capture has no such forward. Rows of a forward are contiguous, in order."""

    if h != w:
        return None
    start = 0
    for fwd in forwards:
        if sha_by_path.get(str(fwd.get("image"))) == item_sha and int(fwd.get("side", -1)) == int(h):
            return int(fwd["index"]), start, int(fwd["rows"])
        start += int(fwd["rows"])
    return None


class S7QvlReference:
    """The block-13 input rows s7_pairs_parent captured (composite/s7/pairs/qvl32_vit_L13, 2026-10-10): a separate
    run of the parent's ViT with its own square-side preprocess. Its json and safetensors shas are checked on open."""

    def __init__(self, capture_dir: Path):
        jpath = Path(capture_dir) / "parent_capture.json"
        require(jpath.is_file(), f"{jpath} missing")
        side = (Path(capture_dir) / "parent_capture.json.sha256").read_text().strip()
        require(side == sha256_file(jpath), f"{jpath}: sha256 sidecar mismatch")
        self.body = json.loads(jpath.read_text(encoding="utf-8"))
        self.file = Path(self.body["file"])
        require(sha256_file(self.file) == self.body["file_sha256"], f"{self.file}: sha256 differs from its receipt")
        self.sha_by_path = {str(i["file"]): str(i["sha256"]) for i in self.body.get("images", [])}
        self.identity = {"dir": str(capture_dir), "json_sha256": side, "file_sha256": self.body["file_sha256"],
                         "tool": self.body.get("tool"), "created_at_utc": self.body.get("created_at_utc")}

    def rows(self, item_sha: str, h: int, w: int) -> tuple[Any, dict[str, Any]] | None:
        from safetensors import safe_open

        hit = s7_forward_slice(self.body["forwards"], self.sha_by_path, item_sha, h, w)
        if hit is None:
            return None
        index, start, n = hit
        with safe_open(str(self.file), framework="pt") as handle:
            forward = handle.get_slice("forward")[start: start + n]
            rows = handle.get_slice("parent_rows")[start: start + n]
        require(bool((forward == index).all()), f"s7 capture rows {start}:{start + n} are not forward {index}")
        return rows.float(), {"forward": index, "rows": [start, start + n], "side": int(h)}


def run_qvl(args: argparse.Namespace) -> dict[str, Any]:
    torch = _torch()
    from PIL import Image
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    s7 = load_s7()
    dev = device_receipt()
    corpus = read_corpus(args.corpus)
    require(corpus["kind"] == "qvl_images", f"{args.corpus}: kind {corpus['kind']}")
    store = ShardStore(args.cache_root, "qvl32_vit_l13", corpus, run_config("qvl32_vit_l13", args))
    store.claim()
    log(f"qvl cache {store.dir}")
    record_integrity(store, "before", args.rank, args.integrity)
    s7_ref = None
    if args.reference_s7 and int(args.block) == 13:  # the s7 capture holds block-13 INPUT rows only
        s7_ref = S7QvlReference(Path(args.reference_s7))
        log(f"qvl reference arm: s7 capture {s7_ref.identity['dir']} ({s7_ref.identity['tool']})")
    config = Qwen3VLVisionConfig(out_hidden_size=5120)  # s7_pairs_parent.run_qvl construction
    config._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(config)
    state = {}
    with safe_open(str(QVL_WEIGHTS), "pt") as h:
        for key in h.keys():
            if key.startswith("visual."):
                state[key[len("visual."):]] = h.get_tensor(key)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    require(not unexpected and not [m for m in missing if not m.endswith("inv_freq")], f"qvl load: missing {missing[:5]} unexpected {unexpected[:5]}")
    model = model.to(device="cuda:0", dtype=torch.float32).eval()
    block = model.blocks[int(args.block)]
    captured: dict[str, Any] = {}

    def pre(module, a, kw):
        captured["x"] = a[0].detach()
        captured["cu_seqlens"] = kw.get("cu_seqlens", a[1] if len(a) > 1 else None)
        captured["position_embeddings"] = kw.get("position_embeddings")

    def post(module, a, kw, out):
        captured["y"] = (out[0] if isinstance(out, tuple) else out).detach()

    hooks = [block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post, with_kwargs=True)]
    rows = 0
    for index in rank_items(corpus, args.rank, args.world, args.limit):
        item = corpus["items"][index]
        if store.done(index):
            continue
        require(sha256_file(item["path"]) == item["sha256"], f"{item['path']}: image changed since the corpus was built")
        rgb = Image.open(item["path"]).convert("RGB")
        h, w = smart_resize(rgb.height, rgb.width, max_pixels=args.max_pixels)
        img = rgb.resize((w, h), Image.BICUBIC)
        pixels = torch.tensor(list(img.getdata()), dtype=torch.float32).view(h, w, 3).permute(2, 0, 1) / 255.0
        pixels = (pixels - 0.5) / 0.5
        flat, grid, yx = s7.qwen_vl_patchify(pixels)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            model(flat.to("cuda:0"), grid.to("cuda:0"))
        torch.cuda.synchronize()
        secs = time.perf_counter() - start
        x_in, y_out = captured["x"], captured["y"]
        require(x_in.shape[0] == flat.shape[0], f"block rows {x_in.shape[0]} vs {flat.shape[0]} patches")
        with torch.inference_mode():
            replay = block(x_in, cu_seqlens=captured["cu_seqlens"], position_embeddings=captured["position_embeddings"])
        replay = replay[0] if isinstance(replay, tuple) else replay
        replay_equal = bool(torch.equal(replay, y_out))
        replay_diff = float((replay - y_out).abs().max())
        require(replay_diff <= 1e-5 * max(1.0, float(y_out.abs().max())), f"block-{args.block} standalone replay differs by {replay_diff}")
        reference = None
        got = s7_ref.rows(item["sha256"], h, w) if s7_ref is not None else None
        if got is not None:
            ref_rows, where = got
            ours = x_in.float().cpu()
            require(tuple(ref_rows.shape) == tuple(ours.shape), f"s7 reference rows {tuple(ref_rows.shape)} vs ours {tuple(ours.shape)}")
            st = ref_row_stats(ours, ref_rows)
            reference = {"kind": "reference: s7_pairs_parent block-13 input rows of the same image at the same square side (separate run)",
                         **s7_ref.identity, **where, **st, "gate": f"rel_fro <= {QVL_REF_REL_FRO}", "numerics_note": QVL_REF_NOTE}
            log(f"qvl reference arm {item['id']}: vs s7 capture rel_fro {st['rel_fro']:.3e} cos_min {st['row_cosine_min']:.6f} "
                f"max_abs {st['max_abs_diff']:.3e} (absmax {st['ref_absmax']:.3e})")
            require(st["rel_fro"] <= QVL_REF_REL_FRO, f"QVL block-13 input diverges from the s7 capture: rel_fro {st['rel_fro']:.3e} > {QVL_REF_REL_FRO}")
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        tensors = {"block_in": x_in.float().cpu(), "block_out": y_out.float().cpu(), "grid_thw": grid.to(torch.int64),
                   "yx": yx.to(torch.int64), "positions": s7.image_positions(yx, gh, gw), "cu_seqlens": captured["cu_seqlens"].to(torch.int32).cpu()}
        if args.store_patches:
            tensors["patch_flat"] = flat.float()
        mem = mem_snapshot()
        meta = {"item": {k: item.get(k) for k in ("id", "path", "sha256", "held", "route")}, "resized_hw": [h, w], "grid_thw": grid[0].tolist(),
                "block": int(args.block),
                "checks": {"determinism_replay": {"kind": "determinism (the same block re-run on its own captured input; not a parent comparison)",
                                                  "bitwise_equal": replay_equal, "max_abs_diff": replay_diff},
                           "reference_s7_capture": reference},
                "preprocess": "smart_resize(factor 32, max_pixels) + PIL bicubic, (x/255-0.5)/0.5, frame duplicated to temporal 2 (s7_pairs_parent.qwen_vl_patchify)",
                "seconds": secs, "memory": mem, "device": dev,
                "meta_summary": {"rows": int(x_in.shape[0]), "held": bool(item.get("held")), "replay_max_abs_diff": replay_diff,
                                 "replay_bitwise": replay_equal,
                                 **({"ref_max_abs_diff": reference["max_abs_diff"], "ref_bitwise": reference["bitwise_equal"]} if reference else {})}}
        store.write(index, tensors, meta)
        rows += int(x_in.shape[0])
        log(f"qvl {store.name(index)} {item['id']}: {x_in.shape[0]} rows {h}x{w} {secs:.2f}s replay equal {replay_equal}")
    for hook in hooks:
        hook.remove()
    record_integrity(store, "after", args.rank, args.integrity)
    return {"rows": rows, "cache_dir": str(store.dir)}


# --------------------------------------------------------------------------
# corpus builders (host side)
# --------------------------------------------------------------------------


def run_corpus_hy3d(args: argparse.Namespace) -> dict[str, Any]:
    exclude, screen = load_exclusions(args.exclude)
    names = [n for n in (args.names or "").split(",") if n]
    items, counts = image_dir_items(args.image_dirs, names=names or None, salt="hy3d", held_mod=args.held_mod, exclude=exclude)
    items = order_and_limit(items, salt="hy3d", limit=args.limit)
    deviations = ["no background removal: rembg weights are not cached offline; images without alpha are full-frame foreground in ImageProcessorV2"]
    if not screen["applied"]:
        deviations.append("contamination carve-out list not applied")
    body = write_corpus(Path(args.out), "hy3d_images", items, source={"image_dirs": args.image_dirs, "names": names, "counts": counts,
                        "order": "item_seed(sha, 'hy3d:order')", "seed": "item_seed(sha, 'hy3d')"}, screen=screen, deviations=deviations)
    return {"rows": len(body["items"])}


def prompt_items(clips: list[dict[str, Any]], *, seeds_per_prompt: int, seconds: float, held_mod: int) -> list[dict[str, Any]]:
    """LM self-gen items: (caption, lyrics, seed, seconds); the content key is the sha of exactly those inputs."""

    out, seen = [], set()
    for clip in clips:
        key = (clip["caption"], clip["lyrics"])
        if key in seen:
            continue
        seen.add(key)
        for k in range(int(seeds_per_prompt)):
            spec = {"caption": clip["caption"], "lyrics": clip["lyrics"], "seed": item_seed(clip["sha256"], f"ace-lm-{k}"), "seconds": float(seconds)}
            sha = sha256_bytes(canonical_json(spec))
            out.append({"id": f"{clip['id']}#lm{k}", "sha256": sha, **spec, "source_clip_sha256": clip["sha256"],
                        "held": held_split(clip["sha256"], held_mod)})
    return out


def run_corpus_ace(args: argparse.Namespace) -> dict[str, Any]:
    exclude, screen = load_exclusions(args.exclude)
    items, stats_out, skipped = [], {}, {"no_audio": 0, "missing": 0, "excluded": 0, "duplicate": 0}
    seen = set()
    for entry, rec, row, stats in registry_rows({"music"}, max_rows=args.max_rows):
        stats_out = dict(stats)
        paths = [p for p in rec.sidecar_paths if p.lower().endswith(AUDIO_EXT)]
        if not paths:
            skipped["no_audio"] += 1
            continue
        path = Path(paths[0])
        if not path.is_file():
            skipped["missing"] += 1
            continue
        if rec.record_id.lower() in exclude or rec.payload_sha256.lower() in exclude:
            skipped["excluded"] += 1
            continue
        sha = sha256_file(path)
        if sha in exclude:
            skipped["excluded"] += 1
            continue
        if sha in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(sha)
        caption, lyrics = split_caption_lyrics(rec.instruction)
        items.append({"id": rec.record_id, "path": str(path), "sha256": sha, "bytes": path.stat().st_size, "caption": caption, "lyrics": lyrics,
                      "dataset": rec.source_dataset, "payload_sha256": rec.payload_sha256, "seed": item_seed(sha, "ace"),
                      "held": held_split(sha, args.held_mod)})
    kind = "ace_clips"
    if args.kind == "prompts":
        kind = "ace_prompts"
        items = prompt_items(items, seeds_per_prompt=args.seeds_per_prompt, seconds=args.seconds, held_mod=args.held_mod)
    items = order_and_limit(items, salt="ace", limit=args.limit)
    deviations = ["captions/lyrics split from the row instruction by split_caption_lyrics (rule in the tool); bpm/key/time signature are the node defaults"]
    if not screen["applied"]:
        deviations.append("contamination carve-out list not applied")
    body = write_corpus(Path(args.out), kind, items, source={"registry": os.environ.get("CURATED_REGISTRY", str(REGISTRY_POINTER)),
                        "route": "music", "registry_stats": stats_out, "skipped": skipped}, screen=screen, deviations=deviations)
    return {"rows": len(body["items"])}


def run_corpus_qvl(args: argparse.Namespace) -> dict[str, Any]:
    exclude, screen = load_exclusions(args.exclude)
    items, skipped, seen, stats_out = [], {"missing": 0, "excluded": 0, "duplicate": 0, "not_image": 0}, set(), {}
    if args.image_dirs:
        found, counts = image_dir_items(args.image_dirs, names=None, salt="qvl", held_mod=args.held_mod, exclude=exclude)
        for it in found:
            it["route"] = "dir"
        items += found
        stats_out["dirs"] = counts
    if args.registry:
        for entry, rec, row, stats in registry_rows({"image", "ocr"}, max_rows=args.max_rows):
            stats_out["registry"] = dict(stats)
            for p in rec.sidecar_paths:
                if not p.lower().endswith(IMAGE_EXT):
                    skipped["not_image"] += 1
                    continue
                path = Path(p)
                if not path.is_file():
                    skipped["missing"] += 1
                    continue
                sha = sha256_file(path)
                if sha in exclude or rec.record_id.lower() in exclude:
                    skipped["excluded"] += 1
                    continue
                if sha in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(sha)
                items.append({"id": f"{rec.record_id}:{path.name}", "path": str(path), "sha256": sha, "bytes": path.stat().st_size,
                              "route": rec.route, "dataset": rec.source_dataset, "seed": item_seed(sha, "qvl"), "held": held_split(sha, args.held_mod)})
    items = order_and_limit(items, salt="qvl", limit=args.limit)
    deviations = [] if screen["applied"] else ["contamination carve-out list not applied"]
    body = write_corpus(Path(args.out), "qvl_images", items, source={"image_dirs": args.image_dirs, "registry": bool(args.registry),
                        "stats": stats_out, "skipped": skipped}, screen=screen, deviations=deviations)
    return {"rows": len(body["items"])}


IMPORT_KINDS = {"hy3d_images": ("hy3d", "hy3d"), "ace_clips": ("ace", "ace"), "ace_prompts": ("ace", "ace"), "qvl_images": ("vit", "qvl")}


def read_distill_set(set_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """(set manifest, rows, provenance) of a distill_corpus_build media set; the items file must be the one the
    manifest names (sha256 + row count)."""

    mpath, ipath = set_dir / "manifest.json", set_dir / "items.jsonl"
    require(mpath.is_file() and ipath.is_file(), f"{set_dir}: not a distill_corpus_build media set (manifest.json + items.jsonl)")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    require(str(manifest.get("schema", "")).startswith(DISTILL_SET_SCHEMA), f"{mpath}: schema {manifest.get('schema')}")
    declared = manifest.get("items_file") or {}
    items_sha = sha256_file(ipath)
    require(items_sha == declared.get("sha256"), f"{ipath}: sha256 {items_sha[:16]} != the set manifest's {str(declared.get('sha256'))[:16]}")
    rows = [json.loads(line) for line in ipath.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == int(declared.get("items", -1)), f"{ipath}: {len(rows)} rows, the set manifest says {declared.get('items')}")
    provenance = {"set_dir": str(set_dir), "set_schema": manifest["schema"], "set_manifest": str(mpath), "set_manifest_file_sha256": sha256_file(mpath),
                  "set_manifest_sha256_field": manifest.get("manifest_sha256"), "items_file": str(ipath), "items_sha256": items_sha,
                  "rows": len(rows), "built_by": manifest.get("identity"), "excluded": manifest.get("excluded")}
    return manifest, rows, provenance


def _local_file(record: dict[str, Any], *, prefer: tuple[str, ...]) -> tuple[Path, str]:
    """The first existing path among `prefer` whose sha256 is the recorded one (a changed file is refused)."""

    sha = str(record.get("sha256", "")).lower()
    require(len(sha) == 64, f"record without a sha256: {json.dumps(record)[:200]}")
    for key in prefer:
        if record.get(key) and Path(record[key]).is_file():
            path = Path(record[key])
            got = sha256_file(path)
            require(got == sha, f"{path}: sha256 {got[:16]} != recorded {sha[:16]}")
            return path, sha
    raise Refusal(f"none of {[record.get(k) for k in prefer]} exists")


def import_items(kind: str, rows: list[dict[str, Any]], *, held_mod: int, dropped: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Corpus items (this tool's fields) from distill set rows; ace_prompts start from the ace_clips items.

    A row whose file is byte-identical to an earlier row's (two registry rows cutting the same window of the same
    source) is dropped and listed in `dropped`: the item seed, the ACE posterior seed and the noise seeds all derive
    from the content sha, so a second copy would only repeat the first one's targets."""

    _, salt = IMPORT_KINDS[kind]
    items, seen = [], {}
    for row in rows:
        record = (row.get("image") if kind in ("hy3d_images", "qvl_images") else row.get("clip")) or {}
        first = seen.get(str(record.get("sha256", "")).lower())
        if first is not None:
            if dropped is not None:
                dropped.append({"rid": row.get("rid"), "sha256": record.get("sha256"), "kept_rid": first})
            continue
        seen[str(record.get("sha256", "")).lower()] = row.get("rid")
        if kind in ("hy3d_images", "qvl_images"):
            image = row.get("image") or {}
            path, sha = _local_file(image, prefer=("stored_path", "path"))
            item = {"id": f"{row.get('rid')}#{sha[:16]}", "path": str(path), "sha256": sha, "bytes": path.stat().st_size,
                    "dataset": row.get("dataset_id", ""), "rid": row.get("rid"), "seed": item_seed(sha, salt), "held": held_split(sha, held_mod)}
            if kind == "hy3d_images":
                item.update(caption=str(row.get("caption", ""))[:1024], tier=row.get("tier"))
            else:
                item["route"] = row.get("route")
        else:
            clip = row.get("clip") or {}
            path, sha = _local_file(clip, prefer=("path",))
            caption, lyrics = split_caption_lyrics(row.get("caption", ""))
            item = {"id": f"{row.get('rid')}#{sha[:16]}", "path": str(path), "sha256": sha, "bytes": path.stat().st_size,
                    "caption": caption, "lyrics": lyrics, "dataset": row.get("dataset_id", ""), "rid": row.get("rid"),
                    "clip_seconds": row.get("seconds"), "tier": row.get("tier"), "seed": item_seed(sha, salt), "held": held_split(sha, held_mod)}
        items.append(item)
    return items


def run_corpus_import(args: argparse.Namespace) -> dict[str, Any]:
    set_dir = Path(args.set_dir)
    manifest, rows, provenance = read_distill_set(set_dir)
    want_set, salt = IMPORT_KINDS[args.kind]
    require(manifest["schema"] == DISTILL_SET_SCHEMA + want_set, f"{set_dir}: a {manifest['schema']} set cannot feed {args.kind}")
    dropped: list[dict[str, Any]] = []
    items = import_items(args.kind, rows, held_mod=args.held_mod, dropped=dropped)
    if dropped:
        log(f"corpus-import {args.kind}: {len(dropped)} rows dropped as byte-identical to an earlier row ({len(items)} distinct files)")
    if args.kind == "ace_prompts":
        items = prompt_items(items, seeds_per_prompt=args.seeds_per_prompt, seconds=args.seconds, held_mod=args.held_mod)
    items = order_and_limit(items, salt=salt, limit=args.limit)
    screen = {"applied": True, "where": "upstream, by distill_corpus_build (contamination screen + eval-split carve-out; screened rows never in the pool)",
              "set_manifest": provenance["set_manifest"], "set_manifest_file_sha256": provenance["set_manifest_file_sha256"],
              "excluded": manifest.get("excluded")}
    deviations = []
    if dropped:
        deviations.append(f"{len(dropped)} of {len(rows)} set rows dropped: file byte-identical to an earlier row's (see source.duplicate_content_rows)")
    if args.kind == "hy3d_images":
        deviations.append("no background removal: rembg weights are not cached offline; images without alpha are full-frame foreground in ImageProcessorV2")
    if args.kind in ("ace_clips", "ace_prompts"):
        deviations.append("captions/lyrics split from the row instruction by split_caption_lyrics (rule in the tool); bpm/key/time signature are the node defaults")
    if args.kind == "ace_clips":
        secs = sorted({i.get("clip_seconds") for i in items})
        deviations.append(f"clips are the corpus lane's pre-cut windows ({secs} s): run ace-encode / ace-dit-run with --clip-seconds equal to them")
    body = write_corpus(Path(args.out), args.kind, items, source={"distill_set": provenance, "rows_in_set": len(rows), "duplicate_content_rows": dropped,
                        "order": f"item_seed(sha, '{salt}:order')", "seed": f"item_seed(sha, '{salt}')", "limit": args.limit},
                        screen=screen, deviations=deviations)
    return {"rows": len(body["items"])}


CHECK_SCHEMA = "omni-campaign.teacher_cache_check.v1"


def _mean(values: list[float]) -> float | None:
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def qvl_reference_arms(metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The s7 parent-reference results (checks.reference_s7_capture) of a QVL cache's shard metas, one per arm."""

    return [m["checks"]["reference_s7_capture"] for m in metas if (m.get("checks") or {}).get("reference_s7_capture")]


def qvl_arm_summary(source: str, cache_dir: Path, manifest: dict[str, Any], arms: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Roll-up of a cache's parent-reference arms, re-gated here (rel_fro <= QVL_REF_REL_FRO, finite) rather than
    trusted from the run: an arm without rel_fro (1.0.5 recorded max_abs only) does not count."""

    problems, rel, cos = [], [], []
    for arm in arms:
        value = arm.get("rel_fro")
        if value is None or not math.isfinite(float(value)):
            problems.append(f"{cache_dir.name[:16]}: QVL reference arm without a finite rel_fro (forward {arm.get('forward')})")
            continue
        rel.append(float(value))
        cos.append(float(arm.get("row_cosine_min", float("nan"))))
        if float(value) > QVL_REF_REL_FRO:
            problems.append(f"{cache_dir.name[:16]}: QVL reference arm rel_fro {float(value):.3e} > {QVL_REF_REL_FRO} (forward {arm.get('forward')})")
    capture = sorted({json.dumps({k: a.get(k) for k in ("dir", "json_sha256", "file_sha256")}, sort_keys=True) for a in arms})
    body = {"source": source, "cache_dir": str(cache_dir), "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "config_sha256": manifest.get("config_sha256"), "corpus_sha256": (manifest.get("corpus") or {}).get("sha256"),
            "shard_tool_versions": manifest.get("shard_tool_versions"), "arms": len(arms), "arms_gated": len(rel),
            "rel_fro_max": max(rel, default=None), "rel_fro_min": min(rel, default=None), "row_cosine_min": min(cos, default=None),
            "sides": sorted({int(a["side"]) for a in arms if a.get("side") is not None}), "gate": f"rel_fro <= {QVL_REF_REL_FRO}",
            "s7_capture": [json.loads(c) for c in capture], "numerics_note": QVL_REF_NOTE}
    if not arms:
        problems.append(f"{cache_dir.name[:16]}: no QVL reference arm in the cache")
    return body, problems


def qvl_parent_reference(cache_dir: Path, manifest: dict[str, Any], metas: list[dict[str, Any]],
                         parent_ref: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    """The parent reference of a QVL cache: its own s7 arms when it has any, else a sealed side cache of the SAME run
    config (--parent-reference, or found under QVL_PARENT_REF_ROOT/<teacher>/*/<config sha>/) whose arms pass. The
    determinism replay is not a parent comparison, so a cache with neither is a structural FAIL."""

    teacher, config_sha = manifest["teacher"], manifest["config_sha256"]
    block = int(((manifest.get("run_config") or {}).get("knobs") or {}).get("block", -1))
    if block != 13:
        return None, [f"QVL block {block}: the s7 capture holds block-13 input rows only, so no parent reference exists for this cache"]
    own = qvl_reference_arms(metas)
    if own and parent_ref is None:
        body, problems = qvl_arm_summary("in-cache s7 reference arms", cache_dir, manifest, own)
        return body, problems
    if parent_ref is not None:
        candidates = [Path(parent_ref)]
    else:
        candidates = sorted(p.parent for p in (QVL_PARENT_REF_ROOT / teacher).glob(f"*/{config_sha}/manifest.json"))
    tried: list[str] = []
    for side in candidates:
        if side.resolve() == Path(cache_dir).resolve():
            continue
        try:
            verify(side)
            side_manifest = json.loads((side / "manifest.json").read_text(encoding="utf-8"))
            require(side_manifest["teacher"] == teacher, f"{side}: teacher {side_manifest['teacher']}, not {teacher}")
            require(side_manifest["config_sha256"] == config_sha,
                    f"{side}: run config {str(side_manifest['config_sha256'])[:16]} is not this cache's {config_sha[:16]}")
            side_metas = [json.loads((side / (Path(s["name"]).stem + ".json")).read_text(encoding="utf-8")) for s in side_manifest["shards"]]
        except (Refusal, OSError, KeyError, ValueError) as exc:
            tried.append(f"parent reference {side} refused: {exc}")
            continue
        body, problems = qvl_arm_summary("side cache of the same run config (s7 capture images)", side, side_manifest, qvl_reference_arms(side_metas))
        if not problems:
            body["candidates"] = len(candidates)
            return body, []
        tried += problems
    if not tried:
        tried.append(f"no QVL parent reference: 0 in-cache s7 arms and no sealed side cache {QVL_PARENT_REF_ROOT / teacher}/*/{config_sha[:16]}... "
                     "(the determinism replay re-runs the block on its own input; it is not a parent comparison)")
    return None, tried


def check_cache(cache_dir: Path, parent_ref: Path | None = None) -> dict[str, Any]:
    """Structural read of a VERIFIED cache (run `verify` first): HY3D reference meshes have faces and their .ply;
    ACE DiT x0 latents are finite and non-zero and no tiled VAE was used; every LM shard holds codes; every QVL
    shard holds rows and the cache has a passing parent reference (qvl_parent_reference). Problems are structural
    (the receipt says FAIL); the rates are diagnostics."""

    import torch
    from safetensors import safe_open

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    teacher, shards = manifest["teacher"], manifest["shards"]
    metas = [json.loads((cache_dir / (Path(s["name"]).stem + ".json")).read_text(encoding="utf-8")) for s in shards]
    shapes = [m.get("tensors", {}) for m in metas]
    problems: list[str] = []
    stats: dict[str, Any] = {"shards": len(shards), "bytes": manifest["totals"]["bytes"], "held": sum(1 for s in shards if s["meta"].get("held"))}
    if teacher == "hy3d_trajectories":
        faces = [int(s["meta"].get("faces", 0)) for s in shards]
        zero = [s["item_id"] for s, f in zip(shards, faces) if f <= 0]
        no_ply = [s["name"] for s, m in zip(shards, metas) if int(m["mesh"]["faces"]) > 0 and not (cache_dir / str(m["mesh"]["ply"])).is_file()]
        problems += [f"reference mesh with 0 faces: {i}" for i in zero] + [f"reference mesh .ply missing: {n}" for n in no_ply]
        stats.update(pairs=sum(int(t["v_guided"]["shape"][0]) for t in shapes), faces_min=min(faces, default=0),
                     faces_median=sorted(faces)[len(faces) // 2] if faces else 0, zero_face_items=zero,
                     s_per_cfg_forward_mean=_mean([s["meta"].get("s_per_cfg_forward") for s in shards]),
                     mesh_seconds_mean=_mean([m["mesh"]["seconds"] for m in metas]))
    elif teacher == "ace_dit_velocity":
        absmax, tiled = [], [s["item_id"] for s in shards if s["meta"].get("tiled_vae")]
        for s in shards:
            with safe_open(str(cache_dir / s["name"]), framework="pt") as handle:
                x0 = handle.get_tensor("x0").float()
            amax = float(x0.abs().max()) if x0.numel() else 0.0
            if not bool(torch.isfinite(x0).all()):
                problems.append(f"non-finite ACE x0: {s['item_id']}")
            elif amax <= 0.0:
                problems.append(f"all-zero ACE x0: {s['item_id']}")
            absmax.append(amax)
        problems += [f"tiled VAE encode: {i}" for i in tiled]
        stats.update(velocity_targets=sum(int(t["v_guided"]["shape"][0]) for t in shapes),
                     latent_frames=sorted({int(t["x0"]["shape"][-1]) for t in shapes}), x0_absmax_min=min(absmax, default=None),
                     x0_absmax_max=max(absmax, default=None), s_per_guided_forward_mean=_mean([s["meta"].get("s_per_guided_forward") for s in shards]))
    elif teacher == "ace_lm_codes":
        tokens = [int(t["tokens"]["shape"][0]) for t in shapes]
        problems += [f"LM shard without codes: {s['item_id']}" for s, n in zip(shards, tokens) if n <= 0]
        stats.update(code_tokens=sum(tokens), tokens_min=min(tokens, default=0), tok_per_s_mean=_mean([s["meta"].get("tok_per_s") for s in shards]),
                     eos_hit=sum(1 for m in metas if m.get("eos_hit")), mean_band_mass=_mean([m.get("mean_band_mass") for m in metas]))
    elif teacher == "qvl32_vit_l13":
        rows = [int(t["block_in"]["shape"][0]) for t in shapes]
        problems += [f"QVL shard without rows: {s['item_id']}" for s, n in zip(shards, rows) if n <= 0]
        parent, why = qvl_parent_reference(cache_dir, manifest, metas, parent_ref)
        problems += why
        stats.update(rows=sum(rows), determinism_replay_bitwise=sum(1 for s in shards if s["meta"].get("replay_bitwise")),
                     reference_arms=manifest["checks"]["reference_arms"], parent_reference=parent,
                     seconds_mean=_mean([m.get("seconds") for m in metas]))
    else:
        raise Refusal(f"unknown teacher {teacher}")
    return {"schema": CHECK_SCHEMA, "tool": tool_identity(), "cache_dir": str(cache_dir), "teacher": teacher,
            "manifest_sha256": sha256_file(cache_dir / "manifest.json"), "finite": manifest.get("finite"), "stats": stats,
            "problems": problems, "verdict": "FAIL" if problems else "PASS", "at_utc": utc_now(),
            "note": "structural check only; finite fit numbers and rates are diagnostics"}


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    verify(cache_dir)
    body = check_cache(cache_dir, Path(args.parent_reference) if args.parent_reference else None)
    write_json_atomic(Path(args.out), body, sidecar_sha=True)
    s = {k: v for k, v in body["stats"].items() if not isinstance(v, list) or len(v) <= 4}
    parent = s.pop("parent_reference", None)
    if parent is not None:
        s["parent_reference"] = {k: parent.get(k) for k in ("source", "arms", "rel_fro_max", "row_cosine_min", "sides", "cache_dir")}
    log(f"CHECK {body['teacher']} {body['verdict']}: {json.dumps(json_safe(s), default=str)} problems {len(body['problems'])} -> {args.out}")
    require(not body["problems"], f"{len(body['problems'])} structural problems: {body['problems'][:5]}")
    return {"rows": body["stats"]["shards"]}


def run_seal(args: argparse.Namespace) -> dict[str, Any]:
    corpus = read_corpus(args.corpus)
    manifest = seal(Path(args.cache_dir), corpus, args.teacher, mode=args.integrity)
    return {"rows": manifest["totals"]["shards"]}


def run_verify(args: argparse.Namespace) -> dict[str, Any]:
    out = verify(Path(args.cache_dir))
    return {"rows": out["shards"]}


def worker_cache_dir(tool_args: list[str]) -> Path:
    """The cache (or x0 stage) dir a worker invocation writes, computed by the same parser and run_config."""

    tool_args = strip_dashdash(tool_args)
    require(tool_args and tool_args[0] in WORKER_TEACHER, f"address needs a cache worker {sorted(WORKER_TEACHER)}")
    ns = build_parser().parse_args(tool_args)
    corpus = read_corpus(ns.corpus)
    if ns.cmd == "ace-encode":
        return X0Stage(ns.cache_root, corpus, ace_x0_knobs(ns)).dir
    teacher = WORKER_TEACHER[ns.cmd]
    return ShardStore(ns.cache_root, teacher, corpus, run_config(teacher, ns)).dir


def run_address(args: argparse.Namespace) -> dict[str, Any]:
    path = worker_cache_dir(args.worker_args)
    print(f"CACHE_DIR {path}", flush=True)
    return {"rows": 0, "cache_dir": str(path)}


def run_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    """Imports the runtime's stacks without loading weights: proves the container/venv can run the workers."""

    body: dict[str, Any] = {"tool": tool_identity(), "device": device_receipt(), "uid": os.getuid()}
    if args.stack in ("hy3d", "all"):
        native = hy3d_native(HY3D_SOURCE)
        body["hy3d_modules"] = sorted(vars(native))
        body["hy3d_source_files"] = len(hy3d_source_files())
        body["hy3d_upstream_pipeline"] = hy3d_pipeline_class(native).__qualname__
    if args.stack in ("ace", "all"):
        comfy_modules()
        body["comfy"] = comfy_identity()
    if args.stack == "qvl":
        import transformers

        body["transformers"] = transformers.__version__
    print(json.dumps(body, indent=1, default=str))
    return {"rows": 0}


# --------------------------------------------------------------------------


WORKERS = {"hy3d-run": run_hy3d, "hy3d-probe": run_hy3d_probe, "ace-encode": run_ace_encode, "ace-dit-run": run_ace_dit,
           "ace-lm-run": run_ace_lm, "ace-probe": run_ace_probe, "qvl-run": run_qvl, "selfcheck": run_selfcheck}
HOST = {"corpus-hy3d": run_corpus_hy3d, "corpus-ace": run_corpus_ace, "corpus-qvl": run_corpus_qvl, "corpus-import": run_corpus_import,
        "p8-header": run_p8, "check": run_check,
        "seal": run_seal, "verify": run_verify, "launch": run_launch, "address": run_address}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)
    default_root = str(OC / "distill/teacher_cache")

    def worker(name: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name)
        s.add_argument("--corpus", required=True)
        s.add_argument("--cache-root", default=default_root)
        s.add_argument("--rank", type=int, default=0)
        s.add_argument("--world", type=int, default=1)
        s.add_argument("--limit", type=int, default=None)
        s.add_argument("--integrity", choices=("full", "stamp"), default="full")
        return s

    s = worker("hy3d-run")
    s.add_argument("--steps", type=int, default=HY3D_STEPS)
    s.add_argument("--guidance", type=float, default=HY3D_GUIDANCE)
    s.add_argument("--octree-resolution", type=int, default=256)
    s.add_argument("--num-chunks", type=int, default=8000)
    s.add_argument("--replay-every", type=int, default=16)
    s.add_argument("--reference-first", type=int, default=1,
                   help="reference arm: run the upstream pipeline __call__ on the first N items this run writes (0 = off)")
    s.add_argument("--require-mesh", action="store_true", help="refuse when any decoded reference mesh has 0 faces (mini-cache gate)")
    s = sub.add_parser("hy3d-probe")
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--repeat", type=int, default=3)
    s.add_argument("--octree-resolution", type=int, default=256)
    s.add_argument("--num-chunks", type=int, default=8000)
    s.add_argument("--cache-pairs", type=int, default=7680)
    s = worker("ace-encode")
    s.add_argument("--clip-seconds", type=float, default=30.0)
    s.add_argument("--allow-tiled-vae", action="store_true", help="record (not refuse) a ComfyUI tiled-VAE fallback")
    s.add_argument("--vae-device", default="cpu", choices=("cpu",))
    s = worker("ace-dit-run")
    s.add_argument("--clip-seconds", type=float, default=30.0)
    s.add_argument("--sigmas-per-clip", type=int, default=4)
    s.add_argument("--cfg", type=float, default=ACE_DEPLOYED_CFG)
    s.add_argument("--allow-tiled-vae", action="store_true", help="record (not refuse) a ComfyUI tiled-VAE fallback")
    s.add_argument("--vae-device", default="cpu", help="cpu (default: the regular fp32 encode does not fit a P40) or cuda:0")
    s = worker("ace-lm-run")
    s.add_argument("--topk", type=int, default=64)
    s.add_argument("--min-seconds", type=float, default=10.0)
    s.add_argument("--max-seconds", type=float, default=120.0)
    s.add_argument("--parity-first", action="store_true", default=True)
    s.add_argument("--no-parity-first", dest="parity_first", action="store_false")
    s = sub.add_parser("ace-probe")
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--repeat", type=int, default=3)
    s.add_argument("--clip-seconds", type=float, default=30.0)
    s.add_argument("--lm-seconds", type=float, default=10.0)
    s.add_argument("--vae-device", default="cpu")
    s = worker("qvl-run")
    s.add_argument("--block", type=int, default=13)
    s.add_argument("--max-pixels", type=int, default=512 * 512)
    s.add_argument("--store-patches", action="store_true")
    s.add_argument("--reference-s7", default=str(S7_QVL_CAPTURE),
                   help="s7_pairs_parent block-13 capture dir for the reference arm ('' = off; used only at --block 13)")
    s = sub.add_parser("selfcheck")
    s.add_argument("--stack", choices=("hy3d", "ace", "qvl", "all"), default="all")

    s = sub.add_parser("corpus-hy3d")
    s.add_argument("--image-dirs", nargs="+", required=True)
    s.add_argument("--names", default="")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--held-mod", type=int, default=8)
    s.add_argument("--exclude", default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("corpus-ace")
    s.add_argument("--kind", choices=("clips", "prompts"), default="clips")
    s.add_argument("--seeds-per-prompt", type=int, default=3)
    s.add_argument("--seconds", type=float, default=60.0, help="prompts: requested duration per LM item (5 code tokens per second)")
    s.add_argument("--limit", type=int, default=1024)
    s.add_argument("--max-rows", type=int, default=None)
    s.add_argument("--held-mod", type=int, default=8)
    s.add_argument("--exclude", default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("corpus-qvl")
    s.add_argument("--image-dirs", nargs="*", default=[])
    s.add_argument("--registry", action="store_true")
    s.add_argument("--limit", type=int, default=512)
    s.add_argument("--max-rows", type=int, default=None)
    s.add_argument("--held-mod", type=int, default=8)
    s.add_argument("--exclude", default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("corpus-import", help="a distill_corpus_build media set (<set>/items.jsonl + manifest.json) as a teacher corpus")
    s.add_argument("--set-dir", required=True, help="e.g. $OC/distill/corpus/media/hy3d")
    s.add_argument("--kind", required=True, choices=sorted(IMPORT_KINDS))
    s.add_argument("--seeds-per-prompt", type=int, default=3, help="ace_prompts: LM items per distinct (caption, lyrics)")
    s.add_argument("--seconds", type=float, default=60.0, help="ace_prompts: requested duration per LM item (5 code tokens per second)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--held-mod", type=int, default=8)
    s.add_argument("--out", required=True)
    s = sub.add_parser("p8-header")
    s.add_argument("--weights", default=str(QVL_WEIGHTS))
    s.add_argument("--out", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--teacher", required=True, choices=("hy3d_trajectories", "ace_dit_velocity", "ace_lm_codes", "qvl32_vit_l13"))
    s.add_argument("--integrity", choices=("full", "stamp"), default="full")
    s = sub.add_parser("verify")
    s.add_argument("--cache-dir", required=True)
    s = sub.add_parser("check", help="verify, then the structural read (mesh faces, finite ACE x0, LM codes, QVL rows + parent "
                                     "reference) -> receipt")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--parent-reference", default=None,
                   help="QVL: a sealed side cache of the same run config whose s7 arms are the parent reference "
                        f"(default: its own arms, else found under {QVL_PARENT_REF_ROOT})")
    s = sub.add_parser("launch")
    s.add_argument("--card", type=int, default=None)
    s.add_argument("--cpu-only", action="store_true", help="CPU worker (ace-encode): a container with no GPU, no card claimed")
    s.add_argument("--name", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--runtime", choices=("docker", "pascal"), default="docker")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("worker_args", nargs=argparse.REMAINDER)
    s = sub.add_parser("address", help="print CACHE_DIR <dir> for a worker invocation: address -- <worker> <args>")
    s.add_argument("worker_args", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = {**WORKERS, **HOST}[args.cmd]
    started = time.time()
    try:
        out = runner(args)
    except Refusal as exc:
        log(f"REFUSED ({args.cmd}): {exc}")
        return 3
    log(f"{args.cmd}: done in {time.time() - started:.1f}s rows {out.get('rows')}")
    return int(out.get("exit_code", 0) or 0)


if __name__ == "__main__":
    sys.exit(main())
