# CHIMERA OMNI FOLDING PLAN — Draft 3 for Cyril's review

Date: 2026-09-01. Author: Claude (interactive session a763fdb3). Status: **PLAN; the first
experiments are running** (section 11, items marked RUNNING). No fold child has been built yet.

**Your two directives today, verbatim, which govern this draft:**

> [Cyril quote withheld]

> [Cyril quote withheld]

And your standing directive from 2026-08-28 (session 7d20c05f, verbatim): *"[Cyril quote withheld]"* — corrected the same day to: don't benchmark
them, build H3's architecture (joint audio-video generation in one flow stream, flow-matching
for continuous media, multi-format conditioning) into Chimera itself. Codex did the architecture
part (dense H3-inspired native media path, commit 3e2623e; joint-AV x0-flow proof passed at
3.19M parameters, 13/13 gates, cc67c49). This plan does the weight part.

Your third directive, later the same day: *"[Cyril quote withheld]"* — the final model is in the Chimera Omni architecture
(Chimera's K3-shaped block set is the skeleton; Qwen folds into it), and the K3 ladder is mandatory.
And your fourth: full permissions to build end to end — fold to the ceiling, checkpoint and clone,
distill from the parents for all modalities, checkpoint and clone, finetune on the curated dataset;
uncensored Qwen3.8 as the Qwen source; delete bad checkpoints; package partially-capable ones for
your review.

So the rule for this document: **every model you named is a fold source. There are no "cannot"
verdicts.** Where a class looks impossible, the plan states the mechanism that would make it
possible and the cheapest experiment that tests it, and the ladder runs until intelligence
stops surviving — measured, never predicted. Draft 2 and the 12-agent workflow synthesis
(archived in `docs/chimera-omni-plan-inputs-2026-09-01/`) said "only Qwen folds; K3 and the
media models are teachers." You rejected that reading; it is withdrawn.

Standing rules carried in unchanged: only measured intelligence gates anything; every fold
class is >= 4x on its component; quantization is forbidden until folding is proven and applied
LAST (K3's experts arrive pre-quantized by Moonshot — we dequantize to bf16 before any fold and
never re-quantize until the end); storage counts parameters; end results export to a standard
runnable checkpoint; dense->MoE is cheating; Chimera Omni is DENSE; artifact-tolerant grading
(loops, echo, cap-rides, missing EOS never count; only content-misses do); every class closes
only by a BUILT child on THIS parent; phase-1 exit is your call alone. Every "pass criterion"
below is a proposed decision line for you to ratify.

---

## 0. The short version

1. **What the target is.** One dense Chimera Omni trunk, Q4 at the very end, fitting a 24 GB
   card (3090 and P40) at 256k context, all modalities in and out. The 24 GB / 256k arithmetic
   (section 2) says the envelope is decided by the full-attention layers' KV cache, not by
   parameter count; a 27B fits with 4-bit KV today, a ~20B child with 4 full layers excised
   fits with 8-bit KV and ~4 GiB for media. That is the floor. The program keeps folding as
   deep as intelligence survives on top of it.
2. **Every named model folds in, by these mechanisms (section 3b):**
   - **Qwen3.8-27B** — the skeleton image and the largest coherent dense body (verified
     27.7B, hybrid 3:1 Gated-DeltaNet : gated attention, 256k native). Classes: MLP channel
     coreset, layer excision, value-head folds, vocab coreset, width folds, exact embed into
     Chimera.
   - **Kimi K3** — 2.8T MoE (verified), whose KDA / gated-MLA / Block-AttnRes / SiTU body is
     *exactly* Chimera's parametrization (Codex built the alignment; the tensor names on disk
     match Chimera's KDA modules one-to-one). Its intelligence enters by a **per-layer streamed
     calibration forward** (one layer's experts in RAM at a time — no full-model inference is
     ever needed) followed by an **expert-union channel coreset** that collapses each layer's
     896 experts into one dense FFN, then depth and width folds and a trunk merge with Qwen.
   - **MiniMax H3** — the video/audio fold source (pruned 20B bf16 + turbo few-step LoRA +
     audio/video VAEs + its Qwen3-VL-32B conditioner, all on disk). Its joint-AV flow DiT folds
     into Chimera's H3-inspired media-refinement path by **VAE absorption into the boundary
     projection / emission head** and **DiT block transplant through residual charts**; its
     conditioner folds into the text trunk as a same-family donor. Target: H3's quality at
     LTX's speed — speed from fewer parameters, few-step sampling (H3's own step-600 turbo
     LoRA proves it), and the KDA trunk. **LTX is out**: not a source, not a teacher.
   - **Qwen-Image, Hunyuan3D 2.1, ACE-Step (DiT + 1.7B LM), Qwen3-VL-32B** — same two
     mechanisms (VAE absorption + block transplant), plus the proven token-interface bridge
     for the ACE LM (audio-as-vocabulary).
3. **The iteration ladder is small first, exactly as you said.** Qwen3.5-0.8B / 2B / 4B / 9B
   share Qwen3.8-27B's architecture; the K3 mechanics are developed on one streamed K3 layer
   and on Chimera's own K3-shaped tiny models before touching all 93 layers; the H3 mechanics
   are developed on the 3.19M joint-AV proof trunk and ACE-Step's 2.2B DiT before the 20B.
4. **Qwen does not embed exactly into Chimera as written; K3 does.** Verified in code
   (section 4): Qwen needs a handful of config switches (grouped KDA keys, GDN decay kind, SiLU
   output gate, per-layer FFN widths, a fixed-band ledger, tokenizer adoption, a checkpoint
   writer); K3's KDA and gated-MLA are Chimera's exact modules (K3's `f_a_proj`/`f_b_proj` =
   Chimera's `decay_down`/`decay_up`, `g_proj` = `output_gate`, `o_norm` = headwise RMSNorm,
   `*_res_proj` = the AttnRes pseudo-queries). Block-AttnRes with a zero-init query is uniform
   mixing, so a Qwen import uses `residual_mode="none"`; a K3-derived block brings its own
   trained AttnRes queries with it.
5. **Standard export is solved for text and open for media.** llama.cpp master runs `qwen35`
   (text + vision projector), `kimi-linear` / `kimi-k3` (KDA + gated-MLA + AttnRes + SiTU — the
   K3 graph *includes* AttnRes and SiTU) and `deepseek2` (MLA). So both skeleton candidates
   export; what has no runtime anywhere is media output from an encoder-free trunk at Q4 —
   proposal: trunk GGUF + companion media GGUF + a driver over llama.cpp's APIs (D5b).
6. **The two hardware facts that shape the schedule:** the CIFS share (`/mnt/ai_data`) is
   reading at under 3 MB/s while the Qwen download writes to it — streaming K3's 1.42 TiB from
   it is a multi-day pass until that clears, so K3's non-expert body (~100 GB) and every working
   parent get copied to the server's local disk (615 GB free, 391 MB/s measured); and the P40's
   flash-attention with quantized KV at head_dim 256 is unverified — the envelope test is
   running now on the on-box Qwen3.6-27B GGUF (section 11 item 1).

Decisions for you are in section 10 (fourteen); the running and next experiments are in
sections 11-12.

---

## 1. Where the LFM2 campaign left us (what transfers)

45 waves, 1,056 sealed ledger rows on LFM2.5-230M. Machine-verified intelligence floor
**0.4116x** (column B); echo-persistence floor **0.3860x** (column A); best parent-band child
14/15 at 0.795x (vocab coreset); retained intelligence to 0.4116x, parent-band coherence never
below ~0.62x. Laws in `reports/matrix-campaign-20260810/DEEP_RECORD.md` section 8; the
record's own caveat governs: *"no wave has measured what any of this transfers to."* Every law
is a one-model result used to ORDER attempts, never to close a class.

| LFM2 law | How it orders the program |
|---|---|
| mlp x vocab mildest composition (1.17 cases) | first pair to build |
| first structural step survivable, second usually fatal (five axes) | single-axis children, then pairs, never three; every pair built on the small ladder before it is scheduled at scale |
| block choice selects surviving capabilities; excision ladder 1->13/15, 2->7, 3->3, 4->0 | excision census per layer, scored per capability |
| verbatim channel selection beats low-rank on SwiGLU (40/42) | MLP and expert-union folds are channel coresets + ridge refit, never SVD |
| install 3 sites at a time; all-sites 4x died (0/15) | coverage grows site by site, priced on the real base |
| conditioned capture worth ~3 cases | every capture conditioned on the review distribution (+ vision / AV splits) |
| 16->12 head merge 8/15 with a joint solve; 16->8 0/15 | head merges built last, mild ratios first |
| the 15-case instrument lacks one-case resolution | bank v2 authored with you before any 27B child |
| sub-0.5x retaining ANY intelligence = success | artifact-tolerant dual-column grading from wave 1 |

Not transferable: LFM2's short-conv transactions, its parent calibration and frozen digest —
every new parent is re-calibrated (section 5.1).

---

## 2. The target, quantified (verified config + arithmetic)

**Qwen3.8-27B** (`Qwen/Qwen3.8-27B`, apache-2.0, 55.6 GB bf16, 18 shards; **landed and
verified complete on the share** — every one of its 31 files matches the Hugging Face manifest
size, index 1,199 tensors; the uncensored variant is downloading next, then the Qwen3.5 0.8B /
4B / 9B ladder):
`Qwen3_5ForConditionalGeneration`, dense; hidden 5120, 64 layers = 16 x (3 GDN + 1 gated GQA);
GDN 16 key / 48 value heads x 128, conv-4; attention 24 q / 4 kv heads x 256, partial RoPE 64,
sigmoid output gate; FFN 17,408; vocab 248,320 untied; 262,144 native; 1 MTP layer; SigLIP-class
ViT ~0.45B.

| Component | Total | Share |
|---|---|---|
| SwiGLU MLP (64 x 0.267B) | **17.1B** | 62% |
| Gated DeltaNet layers (48 x 0.116B) | 5.6B | 20% |
| Gated attention layers (16 x 0.105B) | 1.7B | 6% |
| Embedding + LM head (untied) | 2.5B | 9% |
| MTP + vision | ~0.8B | 3% |
| **Total** | **~27.7B** | |

**Kimi K3** (`moonshotai/Kimi-K3`, verified from config + safetensors headers): 93 layers (69
KDA + 24 gated MLA), hidden 7168, vocab 163,840, 1M context; per KDA layer the Stable
LatentMoE = `routed_expert_down_proj` 7168->3584, **896 experts** of SwiGLU intermediate 3072 in
that 3584 latent (`w1`/`w3` [3072 x 3584], `w2` [3584 x 3072], stored MXFP4 group-32),
`routed_expert_norm`, `routed_expert_up_proj` 3584->7168, plus 2 shared experts (bf16 SwiGLU
6144 in the full 7168 space), router [896 x 7168] with `e_score_correction_bias`; KDA body
q/k/v/g [12288 x 7168] (96 heads x 128), convs, rank-128 decay (`f_a_proj` [128 x 7168],
`f_b_proj` [12288 x 128]), `A_log` [128], `dt_bias`, `b_proj` [96 x 7168], `o_norm`, `o_proj`;
Block-AttnRes pseudo-queries `self_attention_res_proj` / `mlp_res_proj` [1 x 7168] with norms.
Expert mass ~29.6B per layer (2.72T total); non-expert body ~0.6B per layer + 2.35B embeddings
(~50B). 96 shards, 1.42 TiB, complete on the share.

**Memory at 256k on a 24 GiB card** (23.5 usable, ~1.5 GiB buffers, `-np 1`):

| Configuration | Weights | KV @256k | Total | 24 GiB? |
|---|---|---|---|---|
| Stock 27.7B Q4_K_M, f16 KV | 15.4 | 16.0 | 33.1 | no |
| Stock 27.7B Q4_K_M, q8_0 KV | 15.4 | 8.5 | 25.6 | no |
| Stock 27.7B Q4_K_M, **q4_0 KV** | 15.4 | 4.5 | **21.6** | fits, ~1.9 GiB spare |
| 27B with 4 full layers excised, q8_0 | 14.6 | 6.4 | 22.5 | fits, ~1 GiB |
| **20B child**, q8_0 | 11.2 | 8.5 | 21.4 | fits, ~2 GiB |
| 20B child, 4 full layers excised, q8_0 | 11.2 | 6.4 | **19.3** | fits, ~4 GiB for media |
| K3-shaped trunk (KDA:MLA-576 latent), any size <= 27B, q8_0 | <= 15.4 | 2.4 | <= 19.5 | fits, ~4 GiB — the K3 geometry's KV is 4x smaller |

Parameter ceilings from the envelope: q8 KV with 16 GQA full layers <= 23.4B; with 4 excised
<= 27.1B; q4 KV <= 30.4B; **K3-shaped (MLA) full layers <= 34B**. Your 25-30B is inside the
envelope with 4-bit KV, with >= 4 full layers removed, or with the K3 attention geometry — which
is one more reason the skeleton choice (D1) is an experiment, not an assumption. The GDN/KDA
recurrent state is 0.15 GiB and context-independent; it must stay fp32.

3090 vs P40: same memory, very different speed (P40 346 GB/s, no bf16, fp16 at 1/64 rate;
llama.cpp's GDN kernel is sequential per token today, so a 256k prefill is minutes on the 3090
and tens of minutes on the P40; needs a CUDA 12.8 build — no nvcc on the host).

---

## 3. Source model inventory and roles — every one is a fold source

| Model | Verified facts (on disk) | Fold role |
|---|---|---|
| **Qwen3.8-27B** | 27.7B dense VL hybrid; downloading (share) | Skeleton image; the big coherent body; the text/VL intelligence source; teacher for its own children. |
| Qwen3.5-0.8B / 2B / 4B / 9B | identical architecture; queued | Fast ladder for every Qwen-side class. 4B's tied vocab is ~16% of it vs 9% at 27B — vocab results at 4B do not transfer in size. |
| Qwen3.6-27B Q4_K_M GGUF | on the share, `qwen35` arch | Envelope test asset only (quantized, never a fold source). |
| **Kimi K3** | 1.42 TiB MXFP4 experts + bf16 body, 96/96 shards; modeling code in the repo (`modeling_kimi_k3.py`, `modeling_kimi_linear.py`) | **Fold source** via streamed calibration + expert-union coreset (section 3b, K3 ladder). Also the exact architecture donor (already in Chimera). |
| **MiniMax H3** | `minimax_h3_fl2va_pruned_bf16` 38 GB (~20B), fl2va / ref2va int8 20 GB, `minimax_h3_turbo_v4_step600_ema` LoRA, audio VAE fp32 0.6 GB, video VAE fp16 4.9 GB, frozen `qwen3vl_32b` conditioner 48 GB bf16 / 15 GB nvfp4; ComfyUI runtime image on the box; your internal-use ruling recorded (catalog gate D12) | **The video / audio fold source** (H3 ladder). Its conditioner is a same-family text/VL donor. |
| Qwen-Image (fp8) | 20 GB MMDiT + Qwen2.5-VL-7B text encoder + VAE | Image fold source (VAE absorption + block transplant). |
| **Hunyuan3D 2.1** | 6.9 GB checkpoint + 0.6 GB VAE + paint/skintoken nodes | The 3D fold source (D10 confirms it is the one you meant); needs a native 3D type in Chimera. |
| ACE-Step 1.5 turbo | 4.5 GB DiT (~2.2B) + **1.7B / 0.6B Qwen3-type LMs** (vocab 217,204 incl. 64k audio codes) + VAE; MIT | Music fold sources: the LM by token-interface bridge (proven class), the DiT by VAE absorption + transplant. The smallest DiT we own — the H3 mechanics are developed on it first. |
| Qwen3-VL-32B | 48 GB bf16 (H3's conditioner) | Same-family VL donor into the text trunk (within-family residual charts retain 73-98% in the field). |
| Qwen3-ASR / Qwen3-TTS 0.6B | not on disk | Speech sources — download (D8), fold whole. |
| LTX 2.3, HunyuanVideo 1.5 | on disk | **Out.** Not a source, not a teacher (your ruling: the LTX architecture isn't as good). |
| The 3.19M joint-AV proof trunk | v4, cc67c49 | Mechanics donor: the bridge, head, x0 flow, refinement mask, and the H3-mechanics test bed. |

Locations: media sources on the server's local disk `/home/cereal/comfyui/models`; LLMs and K3
on the CIFS share.

### 3b. How each named model folds in — the mechanisms and the experiment ladders

Each ladder starts at the cheapest rung that can falsify the mechanism and climbs until
intelligence stops surviving. Rungs are single transactions; pairs only after singles.

**K3 ladder (the 2.8T MoE into a dense trunk).**
- *K3-1 Streamed calibration forward (harness).* Load the K3 modeling code; for layer i, read
  its shard(s) from the share (or local copy), dequantize the 896 experts' MXFP4 (group-32 fp4
  with uint8 shared exponents — a 30-line kernel) to bf16 in RAM (~30 GB per layer at bf16;
  fits 251 GiB), run the full layer (KDA or MLA + LatentMoE + shared experts + AttnRes) on a
  calibration set of 64k-256k tokens on a 3090, cache the residual stream, the router decisions
  and the MoE (X, Y) pairs, free the layer, next. No full-model inference is ever needed — this
  is exactly how quantizers process trillion-parameter MoEs. Deliverable: wall time per layer,
  which prices everything after it. **This is the first K3 experiment and it needs no ruling.**
- *K3-2 Expert-union channel coreset (new class C14).* Treat each layer's routed channels
  (896 x 3072 in the 3584 latent) plus the shared experts as one gated FFN; select channels by
  the router-mass-weighted stream statistics (pivoted selection, two-stage: per-expert coreset
  then union coreset so the Gram fits in RAM) and ridge-refit the down projection on the cached
  (X, Y); rungs at dense intermediate 65,536 / 32,768 / 17,408 / 8,704 (42x / 84x / 158x /
  316x on the expert mass — all far past the 4x minimum). Measured by the whole model's
  behavior with ONE layer replaced (streamed review, the estate's "install 3 sites at a time"
  law), then growing coverage.
- *K3-3 Dense-K3 review.* A K3 with every layer's experts collapsed at 17,408 is ~93B dense —
  186 GB bf16, reviewable layer-streamed from RAM on this box (slow, fine for the review bank).
  This is the first "K3 intelligence, dense" measurement; you score it.
- *K3-4 Depth and width.* Excision census over the 93 layers (C3/C4), value-head folds on the
  96-head KDA (C5), width fold 7168 -> 5120 by residual charts (C17), vocab 163,840 through a
  cross-tokenizer interface bridge to Qwen's vocabulary (C18).
- *K3-5 Skeleton merge.* K3-derived blocks and Qwen-derived blocks in ONE trunk: layer
  interleaving through residual charts (C12), sibling weight-blends in an aligned basis (the
  LFM2 shallow-depth blend class), or a K3-shaped skeleton with Qwen folded *into it* — three
  experiments, one measured winner.
- *K3-0 (in parallel, cheap): mechanics on Chimera's own K3-shaped tiny models* (the
  `k3_aligned` config and the 3.19M proof trunk with `feed_forward_kind` temporarily MoE on a
  throwaway copy) so C14 is debugged before a 30 GB layer is in RAM.

**H3 ladder (as good as H3, as fast as LTX).**
- *H3-1 VAE absorption (new class C15).* Fit closed-form maps between raw pixel/waveform patches
  (Chimera's native tokens) and H3's video/audio VAE latents on cached pairs (ridge / kernel
  ridge, per patch geometry), so the boundary projection and the emission head speak H3's
  latent language without an encoder module in the trunk. Measured by reconstruction first, then
  by the trunk's media intelligence.
- *H3-2 DiT block transplant (new class C16).* Run H3 on captured latents, cache each DiT
  block's I/O; run Chimera's media-refinement blocks on the same media tokens; fit B_in / B_out
  residual charts (Procrustes / ridge); transplant blocks (replace / insert / control arms, the
  T1 protocol), folding each transplanted block 4x by channel coreset before insertion. Rungs:
  1 block, 4 blocks, 16 blocks, at 4x / 8x.
- *H3-3 Conditioning fold.* H3's AdaLN / time conditioning into Chimera's sigma-conditioned
  bridge; H3's reference-image / reference-video / audio conditioning into the typed native
  segments (multi-format conditioning is what you asked Chimera to inherit).
- *H3-4 Conditioner trunk.* H3's Qwen3-VL-32B is a same-family dense model: fold it (C2/C3/C5)
  and chart it into the text trunk (C12) so the VL understanding H3 relies on lives in the one
  trunk.
- *H3-5 Speed.* The LTX-speed target is measured as wall-clock per clip against the recorded LTX
  2.3 baseline (331-361 s per 608x352 6-step clip on the RTX 8000) and H3's own turbo few-step
  path; speed comes from fewer parameters (fold), fewer steps (x0 flow, the turbo LoRA absorbed
  as a fold of the step schedule), and the KDA trunk's linear-time prefix. Quality is your
  blind 1-5 score against H3 clips.
- *H3-0 (first): the same two mechanisms on ACE-Step's 2.2B DiT and on the 3.19M proof trunk*,
  because they are small enough to iterate in minutes.

**Image (Qwen-Image), 3D (Hunyuan3D), music (ACE DiT):** the H3 mechanisms (C15 + C16) with
each model's VAE; Hunyuan3D additionally needs the native 3D type (D10). **ACE-Step LM and the
speech models:** the token-interface bridge (proven at 10-12/16 on LFM2) into an extended media
band of the ledger, plus a residual chart up to the trunk width. **Qwen3-VL-32B:** same-family
charts into the text trunk.

Honest priors, stated as priors not verdicts: the record's cross-kind and cross-width results
were weak (32-41% branch energy; width folds died on LFM2); the field's one cross-model
linear-bridge result gave zero functional transfer at cosine 0.97. These order the rungs (cheap
ones first) and set expectations; they do not close anything. What survives is what you score.

---

## 4. The architecture bridge (verified in code)

Read side by side: `chimera-omni/src/chimera_omni/modeling/layers.py` + `chimera.py` (HEAD
c1ca0b2), transformers `qwen3_5/modeling_qwen3_5.py`, and K3's on-disk tensor layout.

| Block | Qwen3.5 | Chimera Omni | Exact? |
|---|---|---|---|
| Channel mixer | SwiGLU silu(gate)·up | fused up(2I) -> silu(a)·b, `activation_kind="silu"` | **Yes** (SiTU is K3's, `hidden_act: situ`) |
| Linear-attention heads | 16 key heads serve 48 value heads | one `num_heads` | needs `kda_num_key_heads` |
| Linear-attention decay | per value head, softplus, unbounded | per channel, rank-`kda_decay_rank`, sigmoid, floor e^-5 (**= K3's `f_a`/`f_b` + `A_log` exactly**) | needs `kda_decay_kind="gdn"` for Qwen; exact for K3 |
| Linear-attention output | RMSNorm(o)·SiLU(z) | HeadwiseRMSNorm(o)·sigmoid(g) (**= K3's `o_norm` + `g_proj`**) | needs `kda_output_gate="silu"` for Qwen; exact for K3 |
| Full attention | gated GQA 24/4 x 256, partial RoPE 64 | gated NoPE MLA, latent + extra key dims (**= K3's MLA**) | Qwen needs a gated-GQA kind at import; K3 exact |
| Residual | plain | `"none"` exact; `"block_attnres"` with zero-init query = 0.5·x + 0.5·prefix (reviewer read `chimera.py` 499-527) | Qwen imports with `"none"`; **K3 brings trained pseudo-queries (`*_res_proj`) that install directly** |
| Ledger / tokenizer | Qwen 248,320 untied / K3 163,840 tiktoken | proportional ledger; **no tokenizer in the repo** | fixed-band ledger + tokenizer adoption |
| Serialization | HF safetensors | `torch.save` only | safetensors + GGUF writer needed |
| Vision | Qwen ViT / K3 vision tower | encoder-free | absorbed (C11 / C15) |

**D4 (Codex lane):** `kda_num_key_heads`, `kda_decay_kind`, `kda_output_gate`, per-layer
`intermediate_sizes`, SiTU β = ∞ option, a gated-GQA full-attention kind with partial RoPE,
`attnres_init="identity"`, fixed-band ledger, tokenizer adoption (Qwen's; K3's tiktoken via
the cross-tokenizer bridge), native 3D type, `native_feature_dim` 6144, safetensors + GGUF
writers. Acceptance receipts: Qwen3.5-0.8B exact logits; one K3 layer exact against the K3
reference code on the same input (K3-1 gives us that for free). All defaults stay at current
K3 behaviour.

---

## 5. The folding program (the matrix, re-based)

### 5.1 Stage 0 — calibration and instruments
Frozen 15-case review + **bank v2 authored with you** (~30 harder cases incl. long-context at
8k/64k/200k, code, reasoning, multi-turn, a vision split, an AV split) on Qwen3.5-4B / 9B /
3.8-27B and on dense-K3 rungs; sealed capture distribution with >= 32k-token sequences (none
exists — built here); registry/shard checksum walk. **Your scoring of the parents is the first
thing I need after this review.**

### 5.2 Component classes — >= 4x each, OPEN until a built child closes it on this parent

| # | Class | Applies to | Prior (orders only) |
|---|---|---|---|
| C1 | Vocab coreset | Qwen (248k), K3 (164k) | LFM2 champion; +15.8% sequence tax at 256k; verbatim at 27B by default (D11) |
| C2 | MLP channel coreset | Qwen MLP (62%), dense-K3 FFNs, transplanted DiT blocks | largest lever; 3 sites at a time |
| C3 | Full-attention layer excision | Qwen 16 layers, K3 24 MLA layers | fold-stage KV lever (−6.25% KV per Qwen layer) |
| C4 | Linear-attention layer excision | Qwen 48 GDN, K3 69 KDA | census; zero record coverage |
| C5 | Value-head fold | GDN 48 heads, KDA 96 heads | 16->12 held, 16->8 died on LFM2; mild rungs first |
| C6 | NoPE-dim contraction | Qwen full layers (192 NoPE dims) | new cell; needs a P40 FA smoke |
| C7 | kv-head merge | Qwen 4 kv heads | built last |
| C8 | Residual-width fold | 5120 or 7168 | late; one build |
| C9 | Terminal gain | any | last, +1-3 cases |
| C10 | MTP + vision handling | Qwen, K3 | copy exact through Phase 0 |
| C11 | Encoder absorption | Qwen ViT, K3 vision tower | patch-embed + merger into the bridge, then the ViT blocks as C16 transplants |
| C12 | Cross-model residual bridging | every donor pair | the multi-donor mechanism; T1 tests it |
| C13 | K3-optimization folds | GDN->KDA decay, residual->AttnRes, SiLU->SiTU, GQA->latent | each measured |
| **C14** | **Expert-union channel coreset** (new) | K3's 896 experts per layer | 42x-316x on the expert mass; the K3 ladder's core |
| **C15** | **VAE absorption** (new) | H3, Qwen-Image, Hunyuan3D, ACE VAEs | closed-form maps raw patches <-> latents |
| **C16** | **DiT block transplant** (new) | H3, Qwen-Image, Hunyuan3D, ACE DiTs | charts + 4x coreset per block |
| **C17** | **Cross-width chart** (new) | K3 7168 -> 5120, ACE 2048 -> 5120 | interface fold |
| **C18** | **Cross-tokenizer interface bridge** (new) | K3 tiktoken -> Qwen BPE; ACE codes -> media band | the LFM2 token-bridge class, cross-vocab |
| **C19** | **Trunk merge / blend** (new) | Qwen-derived x K3-derived blocks | shallow blends held on LFM2 |

### 5.3 Composition order, all lanes in parallel on the GPUs we hold
1. **Wave A (Qwen3.5-4B): single-class children** C1-C7, C9 — both 3090s.
2. **K3-1 / K3-0 / H3-0** (harnesses and mechanics on the small test beds) — RTX 8000 + P40s.
3. **Wave B (4B): pairs** (C2 x C3 the budget pair; C1 x C2; C1 x C3). **K3-2** first layers.
   **H3-1** VAE absorption fits.
4. **Wave C (4B): the envelope stack** at the deepest coherent ratio, plus 0.5x / 0.25x variants
   for the spectrum. **K3-3** dense-K3 review. **H3-2** first block transplants.
5. **Wave D (9B)**; **K3-4** depth/width; **H3-3/H3-4**.
6. **Wave E (Qwen3.8-27B)** with the validated recipe (excision census first, then MLP rungs,
   then one pair). **K3-5 skeleton merge** — the first ONE-trunk child with Qwen and K3
   intelligence in it.
7. **Wave F: Chimera import (D4) + C13 optimizations**, one at a time.
8. **Wave G: modality slices** (section 6), each shipping a decoded artifact you can look at.

### 5.4 Scale mechanics (reviewer-corrected)
Captures layer-streamed on a 3090 (Qwen's GDN needs `fla` Triton kernels + bf16; the RTX 8000
has no native bf16 — it takes the fp16 review split after an overflow check); children stored
as (config + index lists + refit tensors), best ~5 per axis materialized; working parents and
K3's non-expert body copied to local disk (the share is the bottleneck); review of 27B-class and
93B-class dense children by streamed or split decode.

---

## 6. Modalities — input and output of everything, from the sources you named

| Slice | In | Out | Fold source(s) | Evidence you look at |
|---|---|---|---|---|
| M1 vision | image, video, OCR | text | Qwen ViT (C11), K3 vision tower (C11), Qwen3-VL-32B (C12) | VL split vs parent |
| M2 speech-in | speech | text | Qwen3-ASR 0.6B (whole-model fold, C18 bridge) | WER, transcripts |
| M3 music / speech-out | text | wav | ACE LM (C18) + ACE DiT (C15/C16); Qwen3-TTS | decoded WAVs, blind score |
| M4 image-out | text, image | image | Qwen-Image (C15/C16) | PNGs |
| **M5 video + audio out** | text, image, video, audio refs | video + synced audio | **H3** (C15/C16/H3-3) | MP4 + WAV, blind score vs H3, wall-clock vs the LTX baseline |
| M6 3D | points / voxels / views | 3D | Hunyuan3D 2.1 (C15/C16) + native 3D type | rendered views |

Order (D6): M5's mechanics are developed first on ACE and the proof trunk because you named H3
as the priority; then M1, M4, M3, M6, M2 as its sources land. Each slice ends with a decoded
artifact and one batched taste sitting.

---

## 7. Compute plan

| Resource | Verified state | Use |
|---|---|---|
| GPU0 RTX 3090 | free; reserved to `neural-forge` | Qwen-side folds and reviews; GDN captures |
| GPU4 RTX 3090 | 17.7 GB held by Codex's sprite worker (`comfyui_h3_gpu4`, idle for hours); reserved to `neural-forge` | second Qwen queue once released via relay (D14) |
| GPU6 RTX 8000 46 GB | idle; reserved to `codex-h3-breck` in `comfy-gpu-owners.json` | K3 streamed layers (bf16 layer weights in RAM, fp16 compute after overflow check) and H3 runs; claim via relay (D14) |
| P40 x4 | idle; one running the envelope test now | llama.cpp benches; fp32 iteration on the small test beds |
| Host | 88 CPUs, 251 GiB RAM, 615 GB local free at 391 MB/s; CIFS share reading < 3 MB/s while the download writes | all closed-form solves; K3 expert dequant + Gram solves in RAM; local staging of K3's body (~100 GB), H3 (38 + 48 GB), parents (~80 GB) |

---

## 8. Export and the "standard checkpoint" rule

- Qwen-shaped children: `qwen35` GGUF today (text + deepstack vision projector).
- K3-shaped / Chimera-native children: `kimi-k3` GGUF **includes KDA, gated MLA, Block-AttnRes
  and SiTU** (merged 2026-08-15; CPU + CUDA); whether its loader accepts every layer dense
  (`first_k_dense_replace = n_layers`) is unverified — test T6-B. If it does, a dense K3-shaped
  child is standard today; if not, a small loader patch.
- Media output: no runtime anywhere samples media from an encoder-free trunk at Q4. Proposal:
  trunk GGUF + companion `-media.gguf` (the `mmproj` convention) + a driver over llama.cpp's
  APIs (embedding input on hybrid archs, per-token final hidden state, a non-causal media pass
  over a cached prefix, `seq_cp` per flow step). Plausible, unverified — week-1 verification
  list; ruling D5b.
- P40: CUDA 12.8 build; matched KV quant pairs only; FA with quantized KV at head_dim 256 on
  cc 6.1 unverified (the running test asserts the kernel ran); K3-geometry MLA (DKQ 576) FA on
  Pascal likewise unverified.
- Quantization once, at the end, re-graded after quantization.

---

## 9. The hardest experiments (not "risks that kill" — the rungs most likely to fail first)

| # | Experiment | Why it is hard | Cheapest rung |
|---|---|---|---|
| H1 | C14 at 158x on a K3 layer keeps that layer's function | no published method collapses 896 experts closed-form; the LFM2 FFN cliff was 2.86x per site | K3-2 on one layer at 42x, whole-model streamed review |
| H2 | C16 transplants carry a DiT block's computation into the trunk | the field's cross-model bridge gave zero transfer at cos 0.97 | H3-0 on ACE's 2.2B DiT, replace-vs-control arms |
| H3 | C15 VAE absorption without an encoder module keeps H3's latent semantics | a VAE is nonlinear; a ridge map is an approximation | reconstruction PSNR then media intelligence on the proof trunk |
| H4 | the pair MLP x excision holds on a hybrid (the envelope pair; LFM2's death pattern) | second structural step | Wave B on the 4B |
| H5 | the P40 envelope (FA + quantized KV at D=256 on sm61; sequential GDN prefill) | unmeasured | running now |
| H6 | K3 streaming throughput on this share | reads < 3 MB/s right now | K3-1 timing; local staging |

---

## 10. Decisions I need from you

| # | Decision | My recommendation |
|---|---|---|
| D1 | ~~Skeleton experiment~~ **RESOLVED by you (2026-09-01): the final model is in the Chimera Omni architecture** (KDA, gated-MLA, Block-AttnRes, SiTU, encoder-free native media, dense). Qwen3.8-27B-Uncensored folds INTO it through the import switches; K3's layers import natively; the qwen3_5 ladder is only the iteration mechanism for the fold classes; export = `kimi-k3`-shaped GGUF + companion media GGUF | Done. |
| D2 | K3 ladder order as in 3b (streamed forward -> expert-union coreset -> dense-K3 review -> depth/width -> merge) | Approve; K3-1 needs no ruling and starts now. |
| D3 | **H3 not LTX** (confirmed) — LTX and HunyuanVideo removed entirely, including as teachers | Confirm. |
| D4 | Chimera import-compatibility switches + ledger + tokenizer + writers (Codex lane) | Approve. |
| D5 | Export policy for children that outgrow `qwen35` / `kimi-k3` | Standard-layout first; arch patch only when a measured child needs it. |
| D5b | "trunk GGUF + companion media GGUF + driver" as standard for media output | Yes, with the driver's four API assumptions verified in week 1. |
| D6 | Modality order: M5 mechanics first (ACE + proof trunk), then M1, M4, M3, M6, M2 | Approve or reorder. |
| D7 | Audio path: raw waveform (Chimera native) with the ACE-LM bridge as an experiment, vs ACE codebook as the path | Raw waveform native; ACE bridge measured. |
| D8 | Download Qwen3-ASR / Qwen3-TTS 0.6B | Yes. |
| D9 | Vision: carry the Qwen ViT exactly through Phase 0, then absorb (C11) | Yes; `mmproj` export as fallback until C11 reaches your bar. |
| D10 | Hunyuan3D 2.1 confirmed; native 3D representation (voxel-SDF patches vs multi-view) | Voxel-SDF. |
| D11 | Vocab coreset at 27B (sequence-length tax) or verbatim | Verbatim at 27B; 4B data point. |
| D12 | Flip the H3 catalog fail-closed gate per your ruling (relay 8324ea96) — needed to run H3 for calibration captures | Yes. |
| D13 | The never-run stage-2 pricing measurement (fold-init vs random-init on LFM2 children, ~20-27 GPU-h) — a training experiment; **not a phase-1 exit**; it prices whether a fold is worth anything as an initializer | Your call; if yes, which arm (column A 0.3860x or column B 0.4116x). |
| D14 | Claim GPU4 (after Codex's sprite worker) and GPU6 (Codex-owned) via relay; K3 stays on disk | Claim both. |

---

### 10b. Process rulings I took under your full-permission order (you review results, not process; say so if any is wrong)

| Ruling | Choice |
|---|---|
| Intelligence column | Column B (copy-backs refused) is the default; column A always published beside it |
| Parent band tolerance | 2 cases, carried from LFM2 |
| Degraded band | ceil(parent band / 2) per parent |
| "At least partially capable" (goes to your review queue) | >= 1 canonical signal lane on any surface — the 15-case bank, bank v2, or a media output with non-zero content |
| Bank v2 | sealed now as a DRAFT (15 text + 4 vision + 2 AV-in scored; media-out scored only by your blind 1-5); amend by supersede |
| Review dtype | bf16 on GPU0 (3090) primary; fp16 on GPU6 (RTX 8000) as a labelled second surface after an overflow check |
| Chimera-native children before the writers land | sealed as PROBE.<TAG>; ARCH-sealed under a schema v4 addendum (pinned loader + sealed GGUF) once the D4 writers exist |
| Checkpoint naming | OC<NN>-<F/D/T>-<PARENT>-<TAG>; clone receipts before every distill/finetune step; deletes only through the retention manifest with the rules in `CHECKPOINT_POLICY.md` |
| Parent codes | Q35-0.8B / 2B / 4B / 9B (ladder), Q38-27BU (source), Q38-27B (control only), K3-L<i> (streamed layer), DK3-<rung> (dense-K3), H3, ACE, QIMG, HY3D, QVL32 |

## 11. Phase 0 — running now and next (no ruling needed for the running items)

**Results so far (2026-09-01, ~22:30Z), all measured on the box:**
- **Envelope proven on both cards.** Stock Qwen3.6-27B Q4_K_M (same `qwen35` architecture class
  as the 3.8) at `-c 262144 -fa on -ctk q4_0 -ctv q4_0 -np 1`: RTX 3090 = 21,546 / 24,576 MiB,
  correct arithmetic, 40.8 tok/s decode; Tesla P40 = 21,381 / 24,576 MiB, loaded in 56 s from
  local disk; warm P40 short-context throughput 246 tok/s prompt / 14.1 tok/s decode, content-
  correct. **Needle-at-depth with 4-bit KV, 3090:** exact-code retrieval at 8,756 / 34,940 /
  139,676 tokens (both positions) and at **250,940 tokens** (prefill 524 tok/s = 8 min, decode
  15.1 tok/s); prefill fell 1,250 -> 1,117 -> 717 -> 524 tok/s and decode 41 -> 33 -> 21 -> 15
  tok/s with depth. **P40:** retrieval at 8,756 / 34,940 / 139,676 / **250,940** tokens, both
  positions (prefill 233 -> 182 -> 98 -> 66 tok/s; decode 12.9 -> 9.5 -> 4.6 -> 2.95 tok/s; the 251k
  prefill took 63 minutes). **16 of 16 retrievals across both cards.** The 256k-on-24-GB end-goal
  envelope is therefore demonstrated end to end for the text trunk, on both card classes, before
  any fold; the P40 is usable but roughly 8x slower than the 3090 at depth.
- **K3 streamed-layer mechanism works.** Layer 1 (5,404 tensors, 17 GB) read off the share,
  all 896 experts dequantized from MXFP4 in 245 s on CPU (weights statistically sane against the
  bf16 shared experts), 256-token MoE forward in 2.5 s on CPU touching 192 experts. The share is
  the only bottleneck: 5.3 MB/s during copies, 25.5 MB/s clean → one full 93-layer pass ≈ 16 h.
- **Sources staged locally:** Qwen3.8-27B-Uncensored (verified 13/13 shards, 55.58 GB),
  Qwen3.5-0.8B, Qwen3.5-4B, the 27B GGUF; Qwen3.5-9B on the share. All seven GPUs are the
  campaign's (your order; Codex's idle sprite container stopped and Codex notified).
- **The curated high-value dataset is located:** `/mnt/ai_data/omnicoder_2026/datasets/
  MASTER_SHARD_INDEX_POINTER.json` — 631 entries, 94,318 rows, 640 shards as of 2026-08-29
  (the plan inputs' "83,095 rows / 505 shards" was a stale June snapshot); routes text, code,
  math, tool, reasoning, long_context, image, video, audio, music, tts, speech, sound, ocr;
  loader `chimera_omni.data.curated_shards`.

1. **DONE — P40 / 3090 256k envelope test** (results above; both servers retired). Still open
   from the protocol: the q8_0/q8_0 KV variant (expected not to fit at 27B), the explicit
   flash-attention no-fallback assertion (the export lane wrote a negative control for it), and
   the 15-case bank through `llama-server`.
2. **RUNNING — downloads**: Qwen3.8-27B (13+/18 shards), then the Qwen3.5 ladder (0.8B / 4B /
   9B), all to the share. Then: hash-verify, copy the working parents local.
3. **NEXT — K3-1 harness**: K3's modeling code is in the repo (`modeling_kimi_k3.py`,
   `modeling_kimi_linear.py`, MXFP4 via `compressed-tensors` — installing it or a hand-rolled
   dequant into `model-surgery-lab/venv`); load layer 1 from shard 2, dequantize, run it against
   the reference on a fixed input (exactness receipt), time it. Then stage K3's non-expert body
   to local disk.
4. **NEXT — H3-0 / K3-0 mechanics** on ACE-Step's DiT and Chimera's K3-shaped tiny models.
5. Campaign root `reports/omni-campaign-2026-09/`; parent calibration (4B / 9B) + bank v2 with
   you; the >= 32k evidence corpus; checksum walk.
6. Port the LFM2 surgery units to `qwen3_5` (new `Harness`; keep the MLP / vocab / head / width
   units; add excision with `layer_types` re-derivation and the above-diagonal assertion; add
   the GDN value-head fold); GGUF round-trip of one unfolded 4B.
7. **Wave A** on the 4B, both 3090s.
8. **DONE (2026-09-02) — the Chimera import lane (D4)**, now MERGED: chimera-omni `main` fast-forwarded
   c1ca0b2 -> 5a87018 (local, not pushed; 392 tests; the MLA layer receipt against the real module at
   8.9e-08 and the reviewer's three merge fences included; Codex given the evidence): config switches, gated-GQA
   block, per-layer FFN widths, fixed-band ledger, tokenizer + cross-tokenizer bridge, native 3D
   (volume) type, safetensors IO, exact-import loaders, GGUF writer. Receipts: Qwen3.5-0.8B ->
   Chimera reproduces the HF logits to 3.2e-05 (fp32, argmax 210/210); K3 layer 0 -> Chimera matches
   its reference to 3.5e-08; the imported 0.8B exports to a `qwen35` GGUF that llama.cpp runs with
   identical text on 3/3 probe prompts. The 27B's `output_gate_type: swish` flag is inert in every
   transformers version (sigmoid applied), so the 27B profile builds with that recorded. Remaining:
   the K3 MLA layer's receipt against the real module (shard being staged), a streaming GGUF writer
   for 4B/27B, a kimi-k3 export of a real child, the companion media GGUF. Relay: GPU6/GPU5 claims
   granted, recovery requested on the rest, all seven GPUs allocated by your order; every brief to
   Codex posted (pivot, D4, all-models-fold-in, H3-not-LTX, architecture, GPU takeover, review).

---


### 12b. Execution state — 2026-09-02 16:20Z

- **Closeout complete**: all six Phase-0 blocker items landed and adversarially reviewed (qwen-fold 17/17,
  ledger 101/101, k3 30/30, media clip-aligned C16 holdout rel_l2 0.5442 / cos 0.8415, evidence manifest
  `omni-evidence-v1.json` sealed, registry walk 631/631 entries and 640/640 shards sha-exact).
- **Parents registered** in the campaign ledger with Hugging-Face provenance receipts: `PARENT-Q35-4B`
  (directory sha 1473b188…, 4,659,865,088 header parameters) and `PARENT-Q35-0.8B` (94f2c2b5…, 873,438,784).
- **Wave A running on GPU0**: first child `C2K2304` (MLP channel coreset 9,216 → 2,304, 4x) — parity on the real 4B
  deterministic and logits bit-identical; per-layer fit relL2 0.33–0.57 so far. Fourteen more single-class
  children queue behind it; the review loop (bank v1 thinking-off + upper-bound columns + ledger review record +
  package) is being scripted for GPU4.
- **P40 8-bit KV boundary**: the stock 27B Q4_K_M loads at 233,472 context and fails at 237,568 with q8_0 KV, so
  256k on 24 GB needs 4-bit KV or ~10% KV headroom from C7/C3 folds.
- **R1 delivered**: the 4B parent scoring sheet is with Cyril; his grades become `PARENT_REVIEW_BAND[Q35-4B]`.

### 12c. Resequenced 2026-09-02/03 — step 1 is the composite fold

- Cyril's ruling: step 1 = ONE registered dense Chimera Omni child containing ALL seven sources; the 4B ladder is
  side measurement only. His K3 ruling: fold ALL experts of ALL 93 layers, each layer fitted on the folded-prefix
  activations, then merge across the Chimera trunk; single-layer receipts are diagnostics, never gates.
- **Composite artifact 1 (2026-09-03 01:51Z): Qwen3.8-27B-Uncensored loads into the Chimera architecture exactly.**
  fp32 CPU, 210 of 210 argmax positions, max abs logit diff 3.0e-04 against a reference absmax of 23.5; per-layer
  hidden-state error grows with the residual RMS to a worst ratio of 1.4e-3 at layer 63 (Codex's 09-04 correction); 26.90B parameters imported, vision and MTP carried as side records; the
  output-gate waiver and the Q1 observation sealed in the receipt. `PARENT-Q38-27BU` is registered with Hugging Face
  provenance. The 27B evidence manifest is sealed (its own tokenizer, 24 x 4,096).
- Infrastructure: after a power cut the media server (dataset share) stayed down until powered; a 3.7 TB Lexar SSD
  now sits on the AI server at `/mnt/lexar` and the whole Kimi K3 checkpoint is copying to it (~20 h), after which
  the K3 streamed pass reads locally instead of at 5-25 MB/s over CIFS.
- **First composite measurements (2026-09-07)**: all nine composite tools built and adversarially reviewed. The
  bf16 Chimera checkpoint of the 27B trunk agrees with the Hugging Face model on 210 of 210 argmax positions under
  streamed teacher forcing (max logit difference 0.37, the bf16-checkpoint column). On the real Kimi K3 read from the
  SSD, layer 0 is sealed verbatim; layer 1's expert-union fold was refused by the lane's prior-anchored veto at rung
  17,408 (the update never reaches the prior's scale), which is a measurement, not a crash. A diagnosis pass is
  testing prior scale, target scale, and fit mechanics on the real layer before any acceptance rule changes. On
  media, MiniMax H3 DiT block 25 was captured and folded at 4x on its MLP with held-out relative error 0.30 and
  cosine 0.95 against a control of 1.42, the first real H3 fold receipt.
- **Codex review and handoff, 2026-09-04**: course confirmed (step 1 not done). Codex hardened three composite tools
  (S0 can no longer overwrite its parent; the streamed 27B fold hashes its source shards; the K3 residual apply is
  token-chunked), 108 server tests, and RAN stage S0: the 27B-Uncensored is now a Chimera bf16 checkpoint with the
  declared music band, 27.70B header parameters, every tensor hash verified. Not registered and not behaviour-verified
  yet; the streamed parity receipt comes next. The Lexar K3 copy is size-verified only; digest verification is running.
- **The step-1 run book exists** (`reports/omni-campaign-2026-09/COMPOSITE_STEP1_RUNBOOK.md`, 55-agent design pass,
  16 claims refuted and corrected): child `OC02-F-Q38-27BU-OMNI1` at ~27.3B parameters; 13 stages; nine new tools.
  K3 enters as all experts of all 92 MoE layers folded with the folded prefix feeding each fit, then charted and
  installed across the 64 trunk layers; media enter as counted interface rows, a declared music band, and charted
  4x blocks. S2P (tool build + review) is running; S0/S1 launch at its end.

## 12. The additional folding tests you offered to help with

| Order | Test | Decides | Cost | From you |
|---|---|---|---|---|
| 1 | **T-P40** envelope smoke (running) | whether the end-goal envelope exists on this hardware | ~5 GPU-h | nothing |
| 2 | **K3-1** streamed layer harness + exactness receipt + timing | whether K3 calibration is hours or days here | CPU + one GPU, 1-2 days of code | nothing |
| 3 | **H3-0** VAE absorption + one-block transplant on ACE's DiT and the proof trunk | whether the two media mechanisms carry anything | 1-2 GPU-days | your blind score on decoded clips |
| 4 | **T1** cross-model residual bridge (LFM2 host, Qwen3-0.6B / 4B donors; replace / insert / control / identity / random arms) | the multi-donor mechanism at text scale | 4-6 GPU-h | scoring ~15 children |
| 5 | **K3-2** expert-union coreset on one K3 layer at 42x / 84x / 158x, whole-model streamed review | whether a K3 layer survives densification | RAM solves + streamed review | scoring |
| 6 | **T6-B** export: `convert_hf_to_gguf` -> Q4 -> P40 parity on a Qwen3.5 hybrid; all-dense `kimi_k3` loader probe | standard export for both skeletons | ~4 GPU-h | nothing |
| 7 | **T5** 27B feasibility: layer-streamed capture, one MLP coreset at 4x on three sites, one excision, one vocab rung | 27B matrix cost; whether the laws keep their sign on a hybrid | 10-15 GPU-h | 27B landed; scoring |
| 8 | **T3** modality-token interface probe (raw patches -> text trunk) | where the media interface holds and where it needs C16 | 3-4 GPU-h | D9 |
| 9 | **T2** GDN -> KDA decay refit, SiLU -> SiTU, on the 4B | whether the K3 optimizations survive as folds | small | scoring |
| — | **T4** stage-2 pricing | D13 only | 20-27 GPU-h | D13 |

What you supply: parent scoring, decisions D1-D14, scoring of survivors, and — if you want to
run children locally — every accepted child ships as an HF checkpoint plus a GGUF with exact
llama.cpp commands.
