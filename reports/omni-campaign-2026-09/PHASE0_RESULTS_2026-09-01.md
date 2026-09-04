# Chimera Omni campaign — Phase 0 results (2026-09-01)

Everything below was measured on the AI server today. No fold child has been built yet; these
are the ground-truth numbers the fold ladder starts from.

## 1. The end-goal envelope exists on both card classes (text trunk, stock model, no fold)

Model: Qwen3.6-27B Q4_K_M GGUF (same `qwen35` hybrid architecture class as Qwen3.8-27B),
llama.cpp `server-cuda` image build 10068, `-c 262144 -fa on -ctk q4_0 -ctv q4_0 -np 1 -ngl 99`.

| Card | VRAM at 262,144 context | Load time (local disk) |
|---|---|---|
| RTX 3090 (GPU0) | 21,546 / 24,576 MiB | 7 s |
| Tesla P40 (GPU1) | 21,381 / 24,576 MiB | 56 s |

Needle-in-a-haystack (a 6-digit code planted at 25% and 75% of the context, exact-match answer):

| Card | Prompt tokens | Hit (25% / 75%) | Prefill tok/s | Prefill time | Decode tok/s |
|---|---|---|---|---|---|
| 3090 | 8,756 | yes / yes | 1,251 / 1,247 | 7 s | 39.0 / 40.7 |
| 3090 | 34,940 | yes / yes | 1,117 / 1,100 | 31 s | 33.4 / 33.7 |
| 3090 | 139,676 | yes / yes | 717 / 722 | 195 s | 21.4 / 21.3 |
| 3090 | **250,940** | **yes / yes** | 524 / 522 | 479-481 s | 15.1 / 15.2 |
| P40 | 8,756 | yes / yes | 163 (cold) / 233 | 54 s / 38 s | 3.9 (cold) / 12.9 |
| P40 | 34,940 | yes / yes | 182 / 181 | 192 s | 9.5 / 9.5 |
| P40 | 139,676 | yes / yes | 98 / 98 | 1,424 s (24 min) | 4.6 / 4.6 |
| P40 | **250,940** | **yes / yes** | 66 / 66 | 3,788 s (63 min) | 2.95 / 2.95 |

Warm short-context throughput on the P40: 246 tok/s prompt processing, 14.1 tok/s decode,
content-correct answers. The cold first request on the P40 measured 2.8 tok/s prompt processing
because the weights were being paged in from the network share through mmap — a share effect,
not a kernel effect.

What this settles: a 27B-class hybrid at Q4 with 4-bit KV fits 256k on a 24 GB card, on Ampere
and on Pascal, and retrieves at depth — 16 of 16 needle retrievals across both cards, through
250,940 tokens. The P40 is roughly 8x slower than the 3090 at depth (63-minute prefill, 3 tok/s). Folding is therefore about headroom (8-bit KV, media
token bands, P40 throughput) and about compressing intelligence — not about existence.

## 2. Kimi K3 is foldable on this box: the streamed-layer mechanism works

Layer 1 of K3 (5,404 tensors, 16.99 GB in shard 2 of 96), CPU only:

| Step | Result |
|---|---|
| Read from the CIFS share | 3,207 s (5.3 MB/s while other copies ran; 25.5 MB/s clean) |
| Dequantize 896 MXFP4 experts to bf16 | 245 s; expert-0 `w1` [3072 x 3584], std 0.0247, |max| 0.125, 11.5% exact zeros (bf16 shared-expert gate std 0.0188 for reference) |
| Expert mass per layer | 59.19 GB bf16 |
| MoE forward, 256 random tokens | 2.48 s; 192 of 896 experts touched; max 58 tokens on one expert |

No full-model inference is needed for K3: one layer at a time in RAM, calibration activations
cached per layer, then the expert-union coreset folds each layer's experts into one dense FFN.
A full 93-layer pass costs about 16 hours at the share's clean read rate.

## 3. Sources staged and verified

| Source | State |
|---|---|
| Qwen3.8-27B-Uncensored (the Qwen source) | 13/13 shards, 21 files size-exact vs the Hugging Face manifest, 55.58 GB; copied to local disk `/home/cereal/models/Qwen3.8-27B-Uncensored`; architecture identical to the base 27B |
| Qwen3.8-27B (base, control only) | 18/18 shards verified, 55.59 GB, on the share |
| Qwen3.5-0.8B / 4B (iteration ladder) | local disk `/home/cereal/models/` |
| Qwen3.5-9B | on the share |
| Kimi K3 | 96/96 shards, 1.42 TiB MXFP4, modeling code present |
| MiniMax H3, Qwen-Image, Hunyuan3D 2.1, ACE-Step (+ 1.7B / 0.6B LMs), Qwen3-VL-32B | on local disk under `/home/cereal/comfyui/models` |
| The curated high-value dataset | `/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json` — 631 entries, 94,318 rows, 640 shards (2026-08-29) |

All seven GPUs are allocated to the campaign (your order; Codex's idle sprite container was
stopped and Codex notified).

## 4. What is being built right now

- Phase-0 build-out workflow (six lanes, adversarially reviewed): the `qwen3_5` fold harness for
  Wave A (vocab coreset, MLP channel coreset, layer excision, GDN value-head fold, NoPE-dim
  contraction, kv-head merge, terminal gain), the campaign ledger + checkpoint/clone/delete
  tooling, the K3 streamed harness + expert-union coreset solver, the media harness (VAE
  absorption + DiT transplant on ACE-Step first), the export/envelope lane, and the parent
  calibration runner with bank v2.
- Import-lane workflow on chimera-omni branch `claude/omni-import`: the config switches that let
  Qwen3.5/3.8 load exactly into the Chimera architecture and K3 layers load natively, the
  fixed-band ledger, tokenizer, native 3D type, safetensors IO, and a GGUF writer — with
  exact-import receipts (Qwen3.5-0.8B logits vs HF; K3 layer 0 vs its reference code).

Next results package: Wave A single-class children on Qwen3.5-4B, dual-column reviewed, with
your review queue of anything at least partially capable.
