# Chimera Omni — Staged Roadmap (planner section)

Written 2026-09-01 against the five estate digests (READ 1–5). Every number is tagged: **[E]** estate fact (repo, server inventory, fold ledger, Cyril's rulings), **[D]** derived arithmetic from [E] facts, **[L]** literature per READ 5, **[A]** assumption to verify. Cyril's standing rules (closed-form folds, behavior-first artifact-tolerant grading, no proxies as gates, build-don't-predict, dense only, quantize last, standard export) are treated as hard constraints throughout.

---

## 0. What is actually foldable — the honest map of Cyril's source list

| Source (as Cyril named it) | What exists on the box [E] | Role it can honestly play | Why |
|---|---|---|---|
| **Qwen3.8-27B** | NOT on the box: only a 170 MB `.part` of a third-party Q4_K_M + its vision mmproj. Qwen3.6-27B exists as Q4_K_M/Q4_K_XL GGUF only (15.4/16.7 GiB, qwen35 arch, 64 blocks, dense). No 27B safetensors anywhere. | **The trunk.** Download bf16 safetensors (~54 GB [D], `Qwen/Qwen3.8-27B` exists per READ 5 config fetch). It is already dense, already 3:1 GDN:gated-attention, already 256k native, already ships Q4 GGUFs that run on llama.cpp. | READ 5: hidden 5120, 64 layers = 48 GDN + 16 gated full attention (4 KV × 256, partial-RoPE 0.25), FFN 17,408, vocab 248,320, 1-layer MTP. [D] ≈26.8B params: FFN 17.1B (64%), GDN 5.5B, full-attn 1.7B, embed+head 2.5B (9.3%, assuming untied [A]). |
| **Kimi K3** | 1.42 TiB MXFP4 MoE, 96 shards, complete. 2.8T total / 104B active, 896 experts top-16. | **Architecture donor only** (already absorbed: KDA, gated MLA, AttnRes, SiTU are in the repo). Not a weight source, not a runnable teacher here. | 896→1 expert collapse is 100× beyond anything published (best closed-form MoE merges are 50% cuts with router calibration; the one full MoE→dense result needs ~4B distillation tokens) [L]. Cannot run here as a teacher: llama.cpp K3 Q2_K ≈ 700 GB vs 184 GB VRAM + 251 GiB RAM [E]. Its dense pieces (hidden 7168, 96 heads) don't shape-match a 5120 trunk; the record's own cross-width/cross-model stitches gave 32–41% branch energy or zero functional transfer [E/L]. |
| **MiniMax H3** | ComfyUI repacks only (fl2va bf16 37.5 GiB, 20.1B elements, dense single-stream AV flow DiT); no HF config; catalog fail-closed (`training_output_authorized=false`); Cyril ruled internal use OK 2026-08-28, relay task 8324ea96 unverified. | **Teacher (AV pairs)** once the catalog flag is flipped per Cyril's ruling; **recipe donor** (joint AV x0-flow — already absorbed in v4). | A DiT with AdaLN/time conditioning and a separate 25.75B Qwen3-VL conditioner has no closed-form map into a decoder-only trunk; the record's cross-kind kernels (conv→attention 32–41%, attention→conv non-viable) and the cos-0.97-zero-transfer negative result [L] are the evidence. |
| **Image models** (Qwen-Image 20B fp8, SDXL, controlnets) | present | Teachers (t2i pairs, edit pairs). Qwen-Image is catalog-qualified. | same as H3 |
| **Video** (LTX 2.3 22B distilled, HunyuanVideo 1.5) | present | LTX rejected as teacher by the 2026-08-28 bench (0.5–2.0/5); HunyuanVideo 1.5 untested → Stage 0 bench. | same |
| **3D** (Hunyuan3D 2.1 + VAE, SkinToken) | present | Teacher (image→mesh) for a **new native 3D type** (Chimera has no 3D type id today [E]). | same |
| **Audio/music** (ACE-Step 1.5 turbo, 5 Hz LM 1.7B) | present; accepted as music specialist; no TTS/ASR weights anywhere | ACE-Step = music teacher; speech from LibriSpeech/LJSpeech subsets + a TTS/ASR teacher to download [A]. | same |
| **The 3.19M AV-proof trunk** | v4 passed 13/13 gates 2026-08-28 | **Code and harness donor, not a weight donor.** | H=128 procedural-data weights have nothing to fold into a 1024–5120-wide trunk; what transfers is `native_media.py`, the bridge/head, the x0-flow + exact refinement mask, and the proof/report/gate discipline. |

**Consequence (the plan's central honest statement):** "folding all the models into one 27B" is, in Cyril's own weight-accounting ("[Cyril quote withheld]"), *not* a fold of the donors — including them would add weights. The donors reduce the final size only by being **replaced** by a trunk graft trained on their outputs, i.e. distillation, which by his stage-order rule comes after the fold ceiling. The closed-form fold stage therefore applies to exactly one object: **the Qwen trunk**, and its purpose at 27B is **headroom** (KV quality, media tokens), not existence — the stock 27B already fits (§1).

---

## 1. The end-goal arithmetic (what "fits on 24 GB with 256k" means)

READ 5 numbers [E/D], 24 GiB card ≈ 23.5 GiB usable, ~1.5–2 GiB compute/logit buffers [A, unmeasured]:

| Configuration | Weights | KV @ 262,144 tok | Total | Fits? |
|---|---|---|---|---|
| Stock Qwen3.8-27B UD-Q4_K_M + **q4_0 KV** | 15.4 GiB | 4.5 GiB (+0.15 GDN state f32) | ≈21.6–22.1 | **Yes** (q4_0 KV on a GDN hybrid measured BLEU 1.000 vs f16 on Qwen3.5-9B [L]) |
| Stock UD-IQ4_XS + q8_0 KV | 13.3 | 8.5 | ≈23.3–23.8 | borderline / no |
| **0.80x-folded** trunk at Q4_K_M + q8_0 KV | ≈12.3 | 8.5 | ≈22.3–22.8 | Yes — *this is what a fold buys* |
| MLA-folded full layers (16 × 576 latent) + Q4_K_M | 15.4 | 2.4 GiB q8_0 | ≈19.3–19.8 | Yes, but non-standard graph (§2) |
| All-layer GQA (what a naïve dense rebuild would be) | — | 34 GiB q8_0 | — | never |

So the deployment target is achievable **today** for text with the stock model at q4_0 KV; the program's real deliverables are (a) all-modality native I/O inside that budget, (b) measured headroom folds, (c) proof on the P40 kernel path. Stage 0 runs this fit test before any fold — it is the cheapest kill-test in the program.

---

## 2. Architecture decision Cyril must make first

**Option A — "Chimera-pure" trunk** (repo `ChimeraConfig`: KDA + gated MLA + NoPE + Block AttnRes + SiTU, dense). No 27B weights exist for it. The synthesis KILL LIST [E] and READ 5 [L] agree post-hoc linear-attention conversion needs finetuning, and every 3:1 hybrid in the field is a fresh ~30T-token pretrain. On this hardware a 27B pretrain is impossible (bf16 weights alone are 54 GB; full Adam ≈ 430 GB; 1B tokens on one RTX 8000 ≈ 37 days [D] and 1B tokens is ~0.04 tokens/param).

**Option B — "Qwen-hosted Chimera" (recommended):** Qwen3.8-27B is the trunk *as-is* (it already carries GDN ≈ KDA-cousin, sigmoid output-gated attention, partial RoPE, MTP, 256k). Graft Chimera's encoder-free native bridge + emission head + x0-flow + exact refinement mask (<100M params [D]: 2 × 3072×5120 ≈ 31M plus embeds/affines). Train the graft fully and the trunk by LoRA. Adopt each Chimera-specific block only as a **single-axis, closed-form-initialized, behaviorally gated fold test** (T1–T4 in §5), never as a bundle — the record's most-replicated law is "one structural axis survivable, two fatal" [E].

Standard-export tension to rule on now: llama.cpp already has the ops a Chimera-pure graph needs — GATED_DELTA_NET with per-dimension (KDA) gate, fp32-only, no cc guards (runs on Pascal) [L]; MLA-576 latent cache (DeepSeek/Kimi arches); tanh/softmax for SiTU/AttnRes — but **no architecture entry composes them**. So SiTU/AttnRes/MLA adoption at 27B means a new llama.cpp arch (a C++ integration task like the GLM-5.3-Flash PRs, not a kernel task) or a sidecar. The media bridge/head are never llama.cpp-standard: proposal = **trunk GGUF (standard qwen35 graph) + a small fp16 sidecar** (bridge/head/flow sampler, tens of MB) driven through llama.cpp's input-embedding / hidden-state API [A: verify `llama_batch` embd + pooling `none` output on the pinned version]. Ruling needed: does "standard GGUF trunk + fp16 I/O sidecar" satisfy R3?

Also rule: the encoder-free rule stands, so the trunk's own ViT (mmproj) is **dropped**; native pixel patches must re-learn image understanding through the bridge. That is measurable at Stage 1 and is a real risk (§6, R2).

---

## 3. Stages

### Stage 0 — Ground truth, tooling, and the end-goal fit test (no folding, no training)

**Deliverables**
1. Weights: `Qwen/Qwen3.8-27B` bf16 (~54 GB [D]; 615 GB free on `/` [E]); a **small GDN-hybrid sibling** for Stage 1 — [A] Qwen3.5/3.6-class 0.8B–4B hybrids exist in bf16 on HF (READ 5 cites Qwen3.5-9B as a 3:1 hybrid); fallback already on the box: Qwen3-0.6B / Qwen3-4B-Base (plain GQA+RoPE, HF format) [E] — the fallback loses the hybrid-specific lessons, so verify the sibling first.
2. Tooling: llama.cpp checkout + `gguf-py` (none exists anywhere [E]); build CUDA **12.8** for sm_61+75+86 inside an `nvidia/cuda:12.8-devel` container (host has no nvcc; driver 580 is fine [E]; CUDA 13 dropped sm_61 [L]); `pip install peft` or hand-rolled LoRA in raw torch (peft absent in the surgery venv [E]); a safetensors writer + HF-config adapter for grafted checkpoints (none in repo [E]).
3. **Fit test (the end-goal test, run first):** stock Qwen3.6-27B Q4_K_M (already on the box, same arch class) then Qwen3.8-27B UD-Q4_K_M: `-c 262144`, `-ctk q4_0 -ctv q4_0`, on RTX 3090 idx0 and **P40 idx1**. Record VRAM, 256k prefill wall (READ 5 extrapolates 15–30 min on P40 from the sequential GDN kernel), decode t/s, needle retrieval at 64k/128k/200k/250k. Also run a DeepSeek-V2-Lite-class MLA GGUF on the P40 to confirm the DKQ=576 flash-attention path on sm_61 (OPEN in READ 5) — decides whether the MLA fold road is even P40-viable.
4. **Frozen review bank v2** authored with Cyril before any child exists: the 15-case bank + ~30 harder cases (long-context retrieval at 8k/64k/200k, multi-turn, code, reasoning, instruction) so the instrument does not saturate at 27B (record law 9: the 15-case bank lacks one-case resolution and the parent reads 12/15 even on LFM2 [E]). Cyril scores; artifact-tolerant lens is the canonical lens.
5. Parent baselines: bf16 Qwen3.8-27B sharded over RTX 8000 + one 3090 (72 GB) on the bank + the cached benchmark suite (arc_e, hella, piqa, wino, lambada, wikitext [E]) as *partial evaluators, never gates*.
6. Data audit: checksum-walk the 83,095-row/505-shard registry (integrity after the 08-19/08-22 cleanups is OPEN [E]); flip the H3 catalog flag per Cyril's ruling (relay 8324ea96); bench HunyuanVideo 1.5 as a video teacher (LTX was rejected); locate/download a TTS/ASR teacher [A].

**Compute/time:** ~5 GPU-days; 1–2 weeks wall (downloads and builds dominate). P40 fit test is the earliest cheap test for risk R3.

---

### Stage 1 — Mini-omni (0.6B–2B trunk) — fold laws on a second model + first real-data native I/O

**Trunk:** the small hybrid sibling [A] (else Qwen3-0.6B). Note the interface share here is LFM2-like (Qwen3-0.6B: 151,936×1024 tied ≈ 26% of 0.6B [D]) — the vocab lever will look strong at this stage and is worth ≤9% at 27B; do not mistake it for a transferable prize (READ 2 law 17).

**Fold recipe (closed-form, frozen protocol, the three axes that ever survived on LFM2 [E]):**
1. Seal parent (sha256, bank, benchmarks).
2. **Excision census**: every block removed singly, `layer_types`/full-attention index re-derived on each deletion (harness law: the un-rederived mask put 28.6–48% attention mass on future tokens [E]); rank behaviorally, never by activation statistics.
3. **FFN intermediate coreset**: stream-selected pivoted-QR channel selection with conditioned capture (+3 cases on LFM2 at byte-identical storage), 3 sites at a time, priced on the real base and re-priced after acceptance; ladder 0.9/0.8/0.7/0.6 of I. All-sites-at-once 4x died on LFM2 (ff640 0/15) — do not repeat it.
4. **Vocab coreset** by prompt-token *occurrence coverage* over the target corpus (≥~81% crossing on LFM2; re-derive), K ladder; record the sequence-length tax (+15.8% at K=19,456 on LFM2) — at 256k this tax is context you lose.
5. Compose **vocab × one structural axis only**; build the second-axis child anyway as a data point (the record predicts it dies; a hybrid may differ — build-don't-predict).
6. Cyril scores every child that holds anything; best child by his score is the fold-init.
7. **Run the never-run Stage-2 measurement** (designed 2026-08-08, 4 pre-registered arms, trainer + tests exist, 0 runs [E]): fold-init + minimal recovery vs distill-from-scratch at matched compute, at 0.6B: 4 arms × ~50M tokens ≈ 10 h on one 3090 [D]. This is the single most important unrun experiment in the program; it decides whether folding is a better initializer than nothing.

**Architecture-conversion folds T1–T4 (§5)** each as a separate single-axis child, reviewed.

**Graft (Chimera code, re-trained, not transplanted):** bridge + emission head + x0-flow + exact refinement mask at H=trunk width; per-type `native_feature_dim` (video tubelet-2 at 32×32 needs 6144 and today gets flagged lossy → decoders refuse [E]; config change). Real data, resolution ladder 32→64→128 px, audio 5 s at 32 kHz (segment 1536 = 41.7 tok/s/channel → ~420 tokens per 5 s stereo pair [D]). Add the **native 3D type** (type id 7): proposal = voxel-SDF patches (16³ = 4096 features, exactly invertible by marching cubes) or multi-view RGB+depth as native images with the view angle in the 4-D position; Hunyuan3D 2.1 makes the pairs. Ruling needed.

**Retraining/recovery budget:** 0.6B full fine-tune fits a 3090 (≈9.6 GB Adam states + activations [D]); 2B full FT needs the RTX 8000 (fp16 mixed precision — Turing has no bf16 tensor cores [A/L]; watch fp16 overflow on Qwen massive activations). Throughput assumption ≈20 TFLOPs sustained per card [A] → 0.6B ≈ 5.5k tok/s → 200M recovery tokens ≈ 10 h; media ≈ 100–200M tokens ≈ 6–12 h. Use the trunk's own fused/chunked GDN path (transformers 5.12 [E]) — Chimera's reference KDA is a Python per-token loop that trained v4 at ~870 tok/s [D from 2.3M tokens/44 min] and is a proof tool, not a trainer.

**Behavioral gates (Cyril's standard):**
- Text: bank v2, artifact-tolerant; fold-only child within −3 of parent to be accepted as init (Cyril sets the number); post-recovery within −1; benchmarks reported, never gating.
- Per-modality input+output proofs in the v4 template (before/after, held-out, seed, GPU UUID, hashed artifacts, 13-gate style): image/video/audio/music/tts/ocr/3D. Input side = held-out understanding accuracy; output side = structural pass (decodable PNG/MP4/WAV/GLB) + x0-loss improvement + **Cyril's blind 1–5 score** (release gate already requires human_review or blind_score [E]).
- Multitask interference harness (specialist vs joint deltas; the piano-music gate 0.018 failed at 2.6M — expect the trunk to fix it; measure).

**Export / quantize / fit:** HF safetensors → `convert_hf_to_gguf` (trunk) → Q4_K_M → llama.cpp on 3090 and **P40**; sidecar driver for media I/O built and demonstrated here (2 days); Q4 vs bf16 measured on the bank (quant tax never assumed).

**Data:** registry rows for image/audio/music/tts/ocr/video; ACE-Step (80.6 s per 10 s track on RTX 8000 [E] → 1,000 tracks ≈ 22 h); Qwen-Image/SDXL t2i pairs; LibriSpeech/LJSpeech subsets; Hunyuan3D pairs; text recovery corpus = parent self-generation 100–200M tokens (ruling: a Q8 teacher for *data* is not a fold — allowed? default yes, bf16 for any logit-KD subset).

**Compute/time:** CPU solves 1–2 weeks (88 cores, 251 GiB RAM); ~30 GPU-days over 3 GPUs; **4–6 weeks wall**.

---

### Stage 2 — Mid (4B–9B hybrid) — does anything transfer, and what does headroom cost?

**Purpose:** the record's biggest hole is "no second model ever folded under the frozen protocol" [E]; Stage 1 fixes that at 0.6B, Stage 2 checks the *direction* of every law with size (interface share falling, attention share rising, MLP two-thirds) on a 9B hybrid [A available] before anything touches the 27B.

**Fold recipe:** only the axes that survived Stage 1; target the 27B-relevant question directly — *does a 0.85x / 0.80x FFN coreset or a 6–10-block excision cost ≤2 bank cases on a hybrid, and does LoRA recovery restore it?* Also the **MLA fold** (T3) on the 16 full layers with LoRA recovery — the only Chimera block with a deployment payoff (KV 8.5→2.4 GiB q8_0 at 256k [D]); if it survives here, it is a candidate at 27B behind a llama.cpp arch entry.

**Recovery budget:** 4B full FT does not fit the RTX 8000 with Adam (≈64 GB [D]); LoRA/DoRA on the trunk + full on the graft; 9B LoRA ≈ 4×9e9 FLOP/token → ≈550 tok/s → 100M tokens ≈ 50 h [D/A]. 3 usable training GPUs (2×3090 + RTX 8000, no NVLink, PCIe DDP/FSDP); **P40s are inference/eval only** (fp32 12 TFLOPs, fp16 1/64, no bf16 [E]).

**Gates / export / fit:** as Stage 1, plus 256k needle tests on the Q4 child on 3090 and P40, and the sidecar media round-trip on the Q4 trunk.

**Compute/time:** ~3 weeks GPU; **6–8 weeks wall**.

---

### Stage 3 — 27B Chimera Omni

**Fold recipe (headroom folds, CPU-side, bf16 trunk resident in 251 GiB RAM):**
1. 64-block excision census: config edit + bank review per block on bf16 sharded over RTX 8000 + 3090 (~10 min/child [A]) → ~11 h. Cheap; do it first.
2. FFN coreset rungs 0.90/0.85/0.80 of I (17,408 → 15,667/14,797/13,926), 3–4 sites at a time, conditioned capture, re-priced; ridge solves at 5120×17,408 are CPU-feasible (days).
3. Optional vocab coreset K≈124k (2x) if Cyril accepts the language-coverage and token-length cost (worth ≤4–5% of the model [D]).
4. Compose vocab × one structural axis; Cyril scores; the accepted child is the init.
Target: **0.80–0.85x** (Q4_K_M ≈ 12.3–13.1 GiB) so that q8_0 KV at 256k fits (§1). Anything deeper is not a headroom need and the record says fold-only coherence below ~0.6x has never been shown (0/15 at every 0.5x set on LFM2 [E]).

**Recovery/graft budget (the binding constraint):** 54 GB bf16 exceeds any single card, so LoRA with the frozen trunk split across RTX 8000 + one 3090 (72 GB), gradient checkpointing, fp16 on the Turing half. Throughput assumption **≈180 tok/s** [A — measure on day 1 of Stage 3; if <100 tok/s the trunk update collapses to graft + last-N-layer LoRA or rented compute]. Budget: text recovery 100M tokens ≈ 6.5 days; media graft 200M tokens ≈ 13 days; 2–3 iterations ≈ 6–8 weeks GPU. QLoRA would help memory but quantizes the base during training — rule conflict, needs a ruling (default: no).

**Gates:** bank v2 vs the bf16 parent (Cyril's scoring), per-modality proofs at the real geometry (image 256–512 px = 64–256 tokens; 1 s video at 256 px tubelet-2 = 128 tokens; 10 s audio ≈ 830 tokens stereo [D]), interference harness, 256k needle + a 200k-token real-document QA.

**Export / quantize / 24 GB fit (the end-goal test):** `convert_hf_to_gguf` (qwen35) → `llama-quantize` Q4_K_M and IQ4_XS → on **RTX 3090 and P40** at 262,144 ctx with q8_0 KV (folded) and q4_0 KV (control), stock Qwen3.8-27B Q4 as parent control; Q4 vs bf16 on the bank; media I/O via sidecar on the Q4 trunk on both cards. If MLA/AttnRes/SiTU were adopted: llama.cpp arch entry composing existing ops (GATED_DELTA_NET KDA-mode, MLA-576, tanh, softmax) — C++ days-to-weeks, upstreamable; otherwise those blocks stay out of the 27B.

**Honest ceiling:** within ~200M media tokens on a LoRA-limited 27B, all ten modalities can be **proven** (structural + intelligible + understanding) but video/music fidelity will not approach H3/LTX/ACE. That is a compute statement, not a design one.

**Compute/time:** **10–16 weeks wall.**

---

## 4. Timeline summary (assumes 3 training GPUs 24/7, P40s for inference, no rented compute, no kills)

| Stage | Wall | GPU-days | Kill-test inside it |
|---|---|---|---|
| 0 Ground truth | 1–2 wk | ~5 | 256k on P40; MLA FA on sm_61 |
| 1 Mini-omni | 4–6 wk | ~30 | raw-pixel/raw-waveform fidelity; fold laws on model #2; stage-2 measurement |
| 2 Mid | 6–8 wk | ~25 | headroom-fold cost on a hybrid; MLA fold survival |
| 3 27B | 10–16 wk | ~60 | LoRA tok/s on split trunk; Q4+256k fit with q8_0 KV |
| **Total** | **≈5–8 months** | | |

---

## 5. Folding tests for Cyril to score (each a single-axis child, closed-form init, frozen bank)

- **T1 SiLU-GLU → SiTU-GLU swap.** SiTU = β_g·tanh(g/β_g)·σ(g)·β_u·tanh(u/β_u), β 4/25 [E]; inside |g|<4, |u|<25 it is ≈ SiLU-GLU. Census the fraction of Qwen pre-activations outside the caps (cheap), then **build** the swapped child regardless and review.
- **T2 GDN → KDA.** Exact k-head expansion (16→48 by copy); decay functional forms differ (softplus-based vs −5·σ bounded [E/L]) → least-squares fit of the decay MLP on captured activations, not exact; review.
- **T3 GQA → gated MLA.** Exact embedding when latent ≥ GQA width (TransMLA-style [L]), keep the 64 RoPE dims per head as the decoupled part, then SVD the NoPE K/V (4×192×2 = 1,536 → 512 latent, 3x [D]); the gate matches Qwen's sigmoid gate. Recovery LoRA at Stage 2.
- **T4 Block AttnRes identity graft.** Zero-init pseudo-query gives *uniform* softmax over depth sources = mean, not the residual [E]; add a per-source logit bias initialized to select the prefix sum so the graft is bit-exact at insertion (small code change), then let training wake it.
- **T5 FFN coreset on a hybrid** (Stage 1) and **one 0.85x rung on the 27B** (Stage 3, CPU).
- **T6 64-block excision census on the 27B** (~11 h).
- **T7 The stage-2 measurement** (fold-init vs scratch at matched compute) at 0.6B.
- **T8 Cross-family donor stitch (only if Cyril insists donors' weights be folded in):** Procrustes/ridge map from an SDXL or H3 mid-block token stream into the 0.6B trunk residual; check whether *any* media signal survives (the literature predicts zero [L]; 1 GPU-day settles it).

---

## 6. Risks that can kill the program — and the earliest cheap test for each

| # | Risk | Evidence | Earliest cheap test |
|---|---|---|---|
| R1 | Donor weights have no closed-form path into a decoder trunk; "fold all models" is distillation by another name | cross-kind kernels non-viable [E]; cos-0.97 → zero transfer [L]; MoE→dense needs ~4B tokens [L] | T8 (1 GPU-day); otherwise accept donors-as-teachers now |
| R2 | Encoder-free raw-pixel / raw-waveform emission and understanding do not reach usable fidelity at real resolution/sample rate | proven only at 32×32 / 0.384 s procedural [E]; music gate failed at 2.6M [E]; trunk's ViT dropped | Stage 1: 1k real images at 128 px + 1k real 5 s clips, Cyril's blind score (2 GPU-days) |
| R3 | 256k on 24 GB fails on the P40 kernel path (sequential GDN prefill, tile FA, DKQ=576 unconfirmed on sm_61) | READ 5 OPEN items | Stage 0 fit test with the on-box Qwen3.6-27B Q4 (1 day incl. build) |
| R4 | LFM2 fold laws do not transfer to a hybrid / 27B (every law is one-model) | STANDING_STATE §7.4 [E] | Stage 1 three-axis matrix on the sibling (CPU weeks) + T6 |
| R5 | 27B recovery throughput too low on split GPUs | ≈180 tok/s [A] | measure LoRA tok/s on day 1 of Stage 3 (½ day) |
| R6 | Data/teacher blockers: registry integrity, H3 flag, no TTS/ASR teacher, LTX rejected | READ 3/4 OPEN | Stage 0 checksum walk + catalog flip + HunyuanVideo bench |
| R7 | Export: media I/O not llama.cpp-standard; Chimera-pure blocks need an arch entry | OMNICODER_EXPORT_MAP "GGUF not imported" [E] | Stage 1 sidecar demo (2 days); Cyril's R3 ruling |
| R8 | Instrument saturation at 27B / lack of resolution | record law 9 [E] | bank v2 authored in Stage 0 |
| R9 | Kimi K3 expectations (weights or teacher) | 700 GB Q2_K vs 435 GB total memory [E/L] | none — ruling: architecture donor only, or provide API/another host |
| R10 | Qwen3.8-27B bf16 unavailable / gated | only a .part on the box [E] | Stage 0 download; fallback Qwen3.6-27B from HF [A] |
| R11 | fp16 overflow on the Turing card during recovery | Qwen massive activations; no bf16 tensor cores on sm_75 [A] | first 100 steps of Stage 1 on the RTX 8000 with loss-scaling diagnostics |

---

## 7. Rulings needed from Cyril before Stage 1

1. Option B (Qwen-hosted Chimera) as the trunk strategy; Chimera-pure blocks only as gated single-axis tests.
2. "Standard GGUF trunk + fp16 media sidecar" satisfies the standard-export rule — or a llama.cpp arch entry is required.
3. Kimi K3 = architecture donor only (no weight or teacher role on this hardware).
4. Donor media models = teachers; T8 only if he wants the falsification on record.
5. Encoder-free stands → drop the trunk's ViT.
6. Native 3D representation (voxel-SDF patches vs multi-view images).
7. Quantized teacher allowed for *data* generation (not weights); QLoRA disallowed by default.
8. Vocab coreset at 27B: accept the language/token-length cost for ≤5% headroom, or skip.
9. Bank v2 contents and the per-stage accept bands (fold-only −3, post-recovery −1 proposed).
10. GPU takeover per his 2026-08-28 ruling (GPU4 currently held by `comfyui_h3_gpu4`, 17.7 GB).

Process carry-overs: seal every child (sha256 + ledger row), keep only the best ~5 per axis, brief Codex via the relay on every requirement change, and search PostgreSQL memory before each stage.