# Target architecture + the 24 GB / 256k budget

Citations: [R1] prompt ledger, [R2] LFM2 fold-only record, [R3] chimera-omni repo audit, [R4] AI-server inventory, [R5] field research/llama.cpp status. Every number below that is not tagged as an estate fact was computed here from the stated formula and is reproducible from `scratchpad/co_budget.py`.

## 0. Bottom line

1. **The ONE dense Chimera Omni should be Qwen3.8-27B-shaped** (H=5120, 64 layers, 48 linear : 16 full, FFN 17,408, Qwen tokenizer), re-expressed in the repo's module classes with every Chimera mechanism set to its *exact-embed* value at fold-init. Day-0 child = Qwen bit-exact (the estate's identity-gate discipline [R2 law 62]). Everything else (K3, H3, image/3D/audio models) is *absorbed* into that skeleton, never used as a skeleton.
2. **Net parameter fold ≈ 1.0x.** The record's best parent-band fold is 0.795x on a 230M model and the intelligence floor is 0.4116x at 1/15 [R2 §4]; the sources sum to ~170B (Qwen 26.9B + K3 non-routed ≈51B + H3 20.1B + Qwen3-VL-32B 25.8B + Qwen-Image 20.4B + LTX 23.1B + Hunyuan3D ≈3.3B + ACE ≈2.2B [R4]). 170B→27B is a 6.3x fold; nothing within 2.5x of that has ever held a single review case. The fold stage for Chimera Omni is therefore **re-parametrization + cache folds + absorption**, not shrinking.
3. **The 24 GB / 256k constraint is decided by the 16 full-attention layers, not by parameter count.** Qwen-native gated GQA (4 KV × 256) needs 8.5 GiB at q8_0 and does NOT fit next to Q4_K_M weights; a closed-form fold of those 16 layers into the repo's gated MLA (latent 512 + rope 64, the K3/DeepSeek geometry llama.cpp already runs with flash-attention) cuts KV to 2.39 GiB q8_0 and fits with ~4 GiB spare — on the 3090 and on the P40 (same 24,576 MiB).
4. **Standard export is satisfiable for the text trunk today** (`qwen35` arch for the exact-embed child, proven on P40 [R5]); the MLA-folded trunk needs either the merged `kimi_k3` graph run dense or a ~few-hundred-line llama.cpp arch patch (both ggml building blocks exist). **Media generation is a runtime, not a weight**: ship a companion `-media.gguf` (mmproj convention, exactly how Qwen3.6-27B already ships on the box [R4]) and drive the x0-flow loop through stock llama.cpp APIs; fallback = torch reference runtime (dev only).
5. **Infeasible on this hardware/record and stated plainly in §7:** running Kimi K3 at all (1.42 TiB MXFP4 vs 184 GB VRAM + 251 GB RAM [R4]); any closed-form fold of the latent-space media DiTs into an encoder-free raw-patch trunk; full fine-tuning of a 27B on 94 GiB of usable training VRAM; a Q4 GGUF as a fold *source* (Qwen3.8-27B bf16 safetensors are not on the box [R4]; a quantized source violates the quantization-last ruling [R1 P32]).

## 1. Design principle: skeleton + exact-embed, then fold one axis at a time

**Why Qwen is the skeleton.** It is the only source that is (a) dense [R1 P35 no-MoE], (b) already ~27B, (c) natively 262k context, (d) a 3:1 GDN:full hybrid — the 2026 cross-lab consensus recipe and the repo's own recipe [R3, R5], (e) runnable in llama.cpp on Pascal today [R5]. The record's most-replicated law — one structural fold axis survivable, two fatal [R2 law 27] — means every fold applied to a 27B skeleton must be priced as a single axis on the real base [R2 law 32]. Starting from anything that needs depth *and* width *and* head alignment (K3: 93→64 layers, 7168→5120 hidden, 96→48 heads) violates that law three times before the first review.

**Exact-embed table.** Each repo mechanism [R3] has a config value at which the folded Qwen function is reproduced exactly; the mechanism is then "switched on" later by a fold test or by stage-3 training, never silently.

| Repo mechanism [R3] | Qwen3.8-27B counterpart [R5] | Exact-embed setting at fold-init | What switching it on costs / needs |
|---|---|---|---|
| `KimiDeltaAttention` (KDA: conv-4, L2-norm q/k, delta rule, per-channel bounded decay `exp(-5·σ(·))`, sigmoid out-gate, headwise RMSNorm) | Gated DeltaNet: conv-4, L2-norm q/k, delta rule, **per-head** decay `exp(-A·softplus(·))`, **SiLU** out-gate, 16 k-heads / 48 v-heads × 128 | Add three config fields: `kda_num_key_heads=16` (GQA-style KDA, same grouping llama.cpp's GATED_DELTA_NET op already executes for Qwen3-Next 16/32 [R5]), `kda_decay_kind="gdn"`, `kda_gate_activation="silu"` → weights copy 1:1 | Moving to per-channel bounded KDA decay = fold test T2 (§8); clip at e⁻⁵ vs GDN's unbounded decay is the only non-representable part |
| `GatedMultiHeadLatentAttention` (NoPE, latent kv, shared extra channels, sigmoid gate) | Gated full attention: 24 Q / 4 KV heads × 256, partial RoPE 64 of 256, sigmoid output gate | Option **F1**: keep gated GQA as a fifth full-layer kind (exact, KV 2,048/token/layer). Option **F2**: gated MLA with `mla_latent_size=512`, `mla_extra_head_dim=64` carrying RoPE (needs a `mla_rope_extra=True` flag — K3/DeepSeek style, not in repo), `mla_key_head_dim=192`, `mla_value_head_dim=256`, `num_heads=24` | F2 is a closed-form fold of [W_k;W_v] (5120→2,048) to rank 512 plus a 4→1 merge of the group rope keys = fold test T3. Exact intermediate exists (latent 1,792 + rope 256, 147.7M params/layer) but its 768-dim key breaks llama.cpp's FA path (only DKQ=576/V=512 is a supported MLA FA shape [R5]) |
| SiTU-GLU `β_g·tanh(g/β_g)·σ(g)·β_u·tanh(u/β_u)` (β 4 / 25) | SwiGLU `g·σ(g)·u` | `situ_gate_beta=situ_up_beta=∞` (or ≥1e4): tanh(x/β)·β → x, so SiTU → SwiGLU exactly | Finite β (K3's 4/25) clips |g|≤4, |u|≤25 = fold test T4 |
| `DepthAttentionResidual` (softmax over completed depth blocks + prefix sum; zero-init query → uniform weights) | Plain residual stream | `attnres_block_size=64` (= L): the only source is the prefix sum, softmax over one source = weight 1 → exact plain residual | Any block size < L with zero-init query gives *uniform* mixing, not identity → AttnRes can only be activated by stage-3 training, not by a fold |
| Typed ledger (control/media/grounding/sync/tool ranges) | Qwen vocab 248,320 (untied, ASSUMPTION — Qwen3-32B-class models are untied; verify from config.json) | Keep Qwen's tokenizer verbatim; **append** 4,096 rows (control 512, media 2,048, grounding 1,024, sync 128, tool 384) zero-init in embed and head → exact on all text | Replaces the repo's proportional policy (which would add 54,320 rows = 556M params at V=248k) — the repo audit flags that policy as "not a chosen allocation" [R3 OPEN] |
| Native bridge / emission head / aux heads | none | Zero-init `type_affine` already makes the bridge a pure scatter-add of projected features; head is new | Initialized from source models only where a raw-pixel linear map exists (T7); otherwise stage 3 |
| MTP heads | 1-layer MTP | `mtp_heads=1`, copied | Excluded from the 24 GB deployment budget by default (unsloth's MTP GGUF is +1.27 GiB [R4]) |

Vocab/tokenizer note: Cyril ruled vocab reduction counts as folding but is "~1% at scale" [R1 P40]; at H=5120, V=252,416 untied the interface is 2.58B = 9.6% of the model (formula V/(V+L·c·d) [R2 law 17] with untied doubling). A vocab coreset is the estate's champion axis on LFM2 (14/15 at 0.795x [R2 law 18]) but it is corpus/tokenizer-specific and costs +15.8% sequence length; it is the one shrink axis I would try *if* Cyril wants headroom (§6), never by default, because it re-tokenizes every source and teacher.

## 2. The target: CO-27

Layer schedule = repo `block_pattern=("kda","kda","kda","gated_mla")` with `force_final_gated_mla` → full layers at i ≡ 3 (mod 4) = layers 3,7,…,63, which is exactly Qwen's `full_attention_interval=4` positions [R5]. Layer kinds: 48 KDA-class + 16 full. `residual_mode=block_attnres` with `attnres_block_size=64` at init. `activation_kind=situ`, β=∞ at init. `feed_forward_kind=dense` (the only value `validate()` accepts [R3]). `native_feature_dim=6144`, `native_type_count=8`, position dim 4, 8 Fourier frequencies (repo defaults except feature_dim).

Parameter accounting (exact formulas; "[Cyril quote withheld]" [R1 P34]):

| Block | Per-unit params | Count | CO-27 F1 (Qwen-native full) | CO-27 F2 (gated MLA 512+64) |
|---|---|---|---|---|
| KDA-class layer (q,k: 16×128; v,gate: 48×128; out; β/decay; conv-4; norm) | 115.88M | 48 | 5.562B | 5.562B |
| Full layer F1: q/gate 24×256, k/v 4×256, o, q/k norms | 104.86M | 16 | 1.678B | — |
| Full layer F2: q 24×(192+64), gate 24×256, kv_down 5120→576, kv_up 512→24×448, o | 102.83M | 16 | — | 1.645B |
| FFN 3·H·I (5120×17,408) | 267.39M | 64 | 17.113B | 17.113B |
| RMSNorms (2L+1)·H | — | — | 0.7M | 0.7M |
| Embedding + lm_head, V=252,416 untied | 1.292B each | 2 | 2.585B | 2.585B |
| Native bridge (feature_proj 6144→5120, type/pos/Fourier/noise-MLP incl. one H×H) | — | 1 | 58.3M | 58.3M |
| Emission head (RMSNorm, type affine, 5120→6144) | — | 1 | 31.6M | 31.6M |
| Aux heads flow/grounding/sync | — | 1 | 46k | 46k |
| **Total** | | | **27.028B** | **26.995B** |

Cross-check: the same formulas give Qwen3.8-27B (no MTP, V=248,320) = 26.896B, consistent with its "27B" label; the +4,096 ledger rows are 41.9M; bridge+head+aux are 0.33% of the model. The KDA per-channel decay MLP (rank 32) adds 0.23M/layer if T2 passes — negligible.

Required repo changes to reach this config (all small, all config-driven): `kda_num_key_heads`, `kda_decay_kind`, `kda_gate_activation`, RoPE on MLA extra channels, per-layer/∞ SiTU β, native type 7 ("volume") for 3D, `feature_dim=6144` default, and a converter that writes GGUF (none exists: `OMNICODER_EXPORT_MAP.md` lists GGUF as Not Imported and there is no safetensors writer either [R3]).

## 3. Unified multimodal token I/O (encoder-free, per the proofs)

Token space = one sequence: text ids from the Qwen tokenizer + ledger ids for control/media/grounding/sync/tool + **native continuous tokens** (raw patches/segments scattered onto placeholder-token positions by the bridge [R3]). No encoder anywhere; the repo's forbidden-dependency test blocks CLIP/SigLIP/QFormer/Whisper/ImageBind/VisionEncoder strings [R3].

| Modality (type id) | Native token = raw features | Features/token (F=6144, zero-pad if fewer; >F is lossy and decoders refuse [R3]) | Position code (t,h,w,r) | Tokens per unit | Output |
|---|---|---|---|---|---|
| Text | Qwen token id | — | causal index (NoPE trunk; RoPE only inside F1/F2 rope-64 channels) | 1 | autoregressive |
| Image (1) | 32×32×3 pixel patch | 3,072 | (0,y,x,0) | 1024² → 1,024; 512² → 256 | x0-flow |
| Video (2) | tubelet 2 frames × 32×32×3 | 6,144 | (center-time s, y, x, 1/fps) | 512² @24 fps → 3,072 tok/s | joint AV x0-flow, video shift 12.0 (v4 proof [R3]) |
| Audio/speech (3), music (4), TTS (5) | 6,144 samples per channel window (stride ≤ window) | 6,144 | (center s, 0, window s, channel) | 48 kHz stereo → 15.6 tok/s; v4 used 32 kHz/1,536-sample windows (20.8 tok/s stereo) [R3] | x0-flow, audio shift 3.0 (v4 proof) |
| OCR (6) | as image, OCR type id | 3,072 | as image | as image | text (autoregressive) |
| **3D (7, new)** | 16³ SDF/occupancy cube of a normalized grid | 4,096 | (z, y, x, voxel size) — re-using the 4-vector | 64³ grid → 64 tokens; 128³ → 512 | x0-flow → marching cubes outside the model (deterministic, not learned; consistent with encoder-free). Input 3D additionally as multi-view image tokens (zero new code). ASSUMPTION: SDF-grid as the native 3D space is my choice; Hunyuan3D 2.1 (6.86 GiB on the box [R4]) generates the teacher grids |

Output mechanics (from the passed v4 proof [R3]): emit a media control token + N media placeholder ids with types/positions; start from noise at σ=1; 16 Euler steps of the x0-target rectified flow with per-type shift `σ' = s·σ/(1+(s−1)σ)` (video 12.0, audio 3.0; **image shift ASSUMPTION 3.0**, to be swept); each step = bridge(x_σ, σ, thwr, type) → trunk with the text prefix cached and the media suffix attending bidirectionally in the 16 full layers only (`native_bidirectional_refinement`, exact overhead 1.011–1.015× measured [R3]) → emission head → x̂₀ → v=(x−x̂₀)/σ. The velocity target did not train at small scale; x0 did [R3] — keep x0. Text and media generation are separate passes (mask contract [R3]).

What 256k holds (encoder-free tax, computed): 256 images at 1024², or 85 s of 512p/24 fps video, or 4.7 h of 48 kHz stereo audio, or 512 grids of 128³ SDF, in any mix with text. Video token rate is the pressure point; the design knobs are patch 32→64 (4× fewer tokens, F=24,576 needed — exceeds F) or fps subsampling; ASSUMPTION that 32×32/tubelet-2 is right for the target — the proofs ran 16×16/tubelet-2 at 32×32 px [R3].

## 4. The arithmetic: 24 GiB at 256k, 3090 and P40

Assumptions (all stated as such): usable VRAM 23.5 GiB of 24,576 MiB (headless server; both cards report 24,576 MiB [R4]); bytes/param derived from the actual unsloth Qwen3.8-27B files divided by the 26.896B analytic count → Q4_K_M 0.6135 B/param (4.91 bpw), IQ4_XS 0.5317 (4.25 bpw) [R5 sizes]; KV cache bytes/value f16 2.0, q8_0 1.0625, q4_0 0.5625 [R5]; recurrent state f32 always (FP16 SSM cache → +37–40% verbosity, Nemotron report [R5]); compute buffers 1.5 GiB (Runpod's "Q4_K_M at full 262k ≈ 34 GB" implies ≈0.3 GiB overhead above weights+f16 KV [R5], so 1.5 is conservative; the 252k-vocab logits buffer alone is 1 MB per output token). Context = 262,144 tokens.

Per-token cache: F1 = 16 layers × 2 × 4 heads × 256 = 32,768 values (64 KB f16); F2 = 16 × 576 = 9,216 values (18 KB f16). KDA state = 48 layers × 48 heads × 128² = 37.7M values = 0.141 GiB f32, sequence-independent (matches "~148 MiB" [R5]).

| Config | Weights | KV @256k | State | Buffers | Total | 24 GiB card |
|---|---|---|---|---|---|---|
| CO-27 **F1**, Q4_K_M, KV f16 | 15.44 | 16.00 | 0.14 | 1.5 | 33.1 | NO |
| CO-27 F1, Q4_K_M, KV q8_0 | 15.44 | 8.50 | 0.14 | 1.5 | 25.6 | NO |
| CO-27 F1, IQ4_XS, KV q8_0 | 13.38 | 8.50 | 0.14 | 1.5 | 23.5 | borderline NO (zero margin) |
| CO-27 F1, Q4_K_M, KV **q4_0** | 15.44 | 4.50 | 0.14 | 1.5 | **21.6** | FITS (1.9 spare) — q4_0 KV measured BLEU 1.000 on a Qwen3.5-9B hybrid, one data point [R5] |
| CO-27 **F2**, Q4_K_M, KV f16 | 15.42 | 4.50 | 0.14 | 1.5 | **21.6** | FITS |
| CO-27 F2, Q4_K_M, KV q8_0 | 15.42 | 2.39 | 0.14 | 1.5 | **19.5** | FITS (4.0 spare) ← **recommended deployment** |
| CO-27 F2, IQ4_XS, KV q8_0 | 13.37 | 2.39 | 0.14 | 1.5 | 17.4 | FITS (6.1 spare) |
| CO-27 F2 at **1M** ctx, q8_0 / q4_0 | 15.42 | 9.56 / 5.06 | 0.14 | 1.5 | 26.6 / 22.1 | NO / FITS (q4_0 only) |
| CO-30 (L=72, 30.04B) F2, Q4_K_M, q8_0 | 17.16 | 2.69 | 0.16 | 1.5 | 21.5 | FITS — but see §6 on why not |
| CO-27 F2b (exact rope-256 latent 768, 27.39B), Q4_K_M, q8_0 | 15.65 | 3.19 | 0.14 | 1.5 | 20.5 | FITS memory-wise, but no FA kernel for DKQ=768 → non-FA KQ scratch n_kv×n_ubatch×24 heads×4 B = 12.9 GB at ubatch 512 → effectively NO |

Parameter ceiling implied (Q4_K_M, 1.5 GiB buffers): with F2 + q8_0 KV, weights ≤ 19.5 GiB → **≤ 34B**; with F1 + q4_0 KV → ≤ 30B; with F1 + q8_0 KV → ≤ 23B. So "25–30B" [R1 P47] is inside the envelope only with MLA-folded full layers or q4_0 KV.

**3090 vs P40 — memory identical, everything else not:**

| | RTX 3090 (sm_86, 936 GB/s) | Tesla P40 (sm_61, 346 GB/s, no bf16, fp16 at 1/64 rate [R5]) |
|---|---|---|
| Memory table above | applies | applies (same 24,576 MiB) |
| Decode bandwidth floor, F2 q8_0 @256k (reads 15.42+2.39+0.14 GiB/token) | 20.6 ms → ≤48 t/s; realistic ~25–30 t/s (ASSUMPTION 50–60% of floor) | 55.7 ms → ≤18 t/s; realistic ~7–10 t/s (consistent with [R5]'s 8–9 t/s extrapolation) |
| Decode, F1 q4_0 @256k | ≤43 t/s | ≤16 t/s |
| Short context, F2 | ≤56 t/s | ≤21 t/s |
| 256k prefill | GDN CUDA kernel is sequential per token ("TODO: Add chunked kernel") [R5]; ESTIMATE 3–6 min | same kernel, fp32-only, no cc guards → runs; generic tile FA, no tensor cores; [R5] extrapolation 15–30 min |
| Build | any CUDA | must build with CUDA 12.8/12.9 (CUDA 13 dropped sm_61) [R5]; no nvcc on the host today [R4] |
| KV-quant kernels | FA q8_0/q8_0 and q4_0/q4_0 pairs only by default; mixed pairs silently fall back to CPU (issue #24485) [R5] — never mix K/V types | same, plus unverified whether the DKQ=576 MLA FA path compiles for cc 6.1 [R5 OPEN] |
| Torch side (fold calibration, media dev runtime) | bf16 native; 27B bf16 = 50.1 GiB → needs 3090+3090 or 3090+RTX 8000 device_map | no bf16, fp16 crippled → fp32 only: 27B fp32 = 100 GiB → P40s unusable for any 27B torch pass; usable for ≤4B fp32 iteration models |

Media generation cost (16 steps × N media tokens through the 24.3B trunk; ESTIMATE at 40 TFLOP/s effective on 3090, 9 TFLOP/s fp32 on P40): 256 tokens (512² image) 5 s / 22 s; 1,024 tokens (1024² image) 20 s / 89 s; 4,096 tokens (~1.3 s of 512p video) 80 s / 354 s. On P40 a 1024² image is ~1.5 min — usable, not interactive.

Activation memory: not a deployment concern in llama.cpp (ubatch-bounded, inside the 1.5 GiB). In the torch reference runtime the repo's full layers materialize [B,n,T,T] fp32 scores and `MixerHistory` stores every past input [R3] → 6.87e10 scores/head at 256k; the torch runtime is therefore dev-only at ≤ a few thousand tokens until the planned paged latent cache exists (`paged_state_implemented False` [R3]).

## 5. GGUF / llama.cpp export path

What stock llama.cpp has (as of 2026-08-31 [R5]): `qwen35` arch (GDN + gated full attention, partial RoPE, MTP) — shipped, runs on P40 at 128k [R5]; GATED_DELTA_NET op with KDA mode (per-channel gate) and Qwen3-Next mode, fp32 CUDA, no arch guards; `kimi_k3` arch merged 2026-08-15 (KDA + gated MLA + Block AttnRes + SiTU + MoE); MLA stored as 576-dim latent with an FA path for DKQ=576/V=512. On the box: only the `ghcr.io/ggml-org/llama.cpp:server-cuda` image (build 10068) and LM Studio's bundled backends; no source checkout, no `gguf-py`, no `convert_hf_to_gguf.py`, no `llama-quantize` [R4].

| Child state | Exportable as | Standard? | Work |
|---|---|---|---|
| Stage A: exact-embed child (F1, GDN decay, SiLU gate, β=∞, AttnRes block=L, +4,096 ledger rows) | `qwen35` GGUF (append rows to `token_embd`/`output`, vocab 252,416) | Yes, stock llama.cpp, P40-proven | write the converter (from Chimera state_dict/safetensors → GGUF via gguf-py); verify ledger rows load |
| Stage B: F2 (MLA-folded full layers) + GDN linear layers | no stock arch has GDN-decay linear layers + MLA full layers | Needs a graph: either export as `kimi_k3` with `n_expert=0` on every layer (requires the loader to accept all-dense, and the decay nonlinearity to be KDA's, i.e. T2 must pass first) or a `chimera` arch patch = qwen35's linear block + deepseek2/kimi_k3's MLA block (both exist in ggml; ESTIMATE a few hundred lines) | patch + upstream PR; until merged, "standard" = a pinned fork build |
| Stage C: AttnRes active / finite-β SiTU / per-channel KDA decay | `kimi_k3` dense | Yes if the dense-all-layers path loads | verify with the K3 PR's loader |
| Media bridge + emission head + aux | companion `co27-media.gguf` (tensors: feature_proj, type/pos/Fourier/noise MLP, head norm/affine/linear) | Same convention as the `mmproj-*.gguf` files already beside Qwen3.6-27B on the box [R4]; llama.cpp rejects unknown tensors inside the main file, so a second file is required | converter only |
| Media generation loop | external driver (numpy/torch on CPU or GPU) using llama.cpp APIs: `llama_batch.embd` input for media positions, per-token final-hidden output (pooling none), `llama_set_causal_attn(false)` for the media-suffix pass so the 16 full layers attend bidirectionally while KDA layers stay causal — which is exactly the repo's refinement mask [R3] | Weights standard; the loop is a runtime | verify three things in week 1: embd input on hybrid archs, per-token embeddings from a generative model, causal flag + cached prefix semantics. Fallback: torch reference runtime on RTX 8000 + 3090 (fp16, 70 GiB), dev only |

What Cyril's standard-export ruling implies [R1 P35]: "[Cyril quote withheld]". The text trunk meets it literally at Stage A and after a small arch patch at Stage B. Media I/O meets the *weight* half (two standard GGUF files, no custom tensor formats, no codebooks) but not the *runtime* half — stock `llama-server` will never sample an image. **Ruling needed:** is "stock llama.cpp runs the text model from the same GGUF; a companion GGUF + Chimera driver does media" standard enough? If not, the only alternative is a llama.cpp fork with a media sampler, which is a larger runtime project than the model.

Toolchain prerequisites on the box (none present [R4]): llama.cpp checkout + CUDA 12.8 toolkit (sm_61 + sm_86 + sm_75 in one build), `gguf-py` in `/home/cereal/model-surgery-lab/venv`, and — required before any fold — **Qwen3.8-27B bf16 safetensors** (~54 GB; only Q4 GGUFs and a dead 170 MB `.part` exist [R4]; a Q4 source is forbidden by P32).

## 6. Recommendation: one target, two iteration configs

**Target — CO-27 (F2)**: H=5120, L=64 (48 KDA-class + 16 gated MLA at i≡3 mod 4), I=17,408, KDA 16 k-heads / 48 v-heads × 128, MLA 24 heads, latent 512 + RoPE-64 shared key, nope 192, v 256, sigmoid gates, SiTU (β=∞ at init), AttnRes block 64 at init, V=252,416 untied, F=6144, 8 native types incl. volume, MTP 1 (optional). **26.995B params; Q4_K_M 15.42 GiB; 256k KV 2.39 GiB q8_0; fits both 24 GiB cards with 4.0 GiB spare; 1M fits only with q4_0 KV.** Reached from Qwen3.8-27B by: exact embed (T1) → +ledger rows (T6) → MLA cache fold of 16 layers (T3) → optional T2/T4 → stage 2/3. If T3 fails the behavior grade, the fallback deployment is CO-27 F1 with q4_0 KV (21.6 GiB), which still fits.

**Why not 30B.** CO-30 fits (21.5 GiB), but the extra 3.0B would be layers no dense source provides — random or K3-aligned blocks that the record says cannot be repaired without gradients [R2 law 33]. Parameters that are not folded from a coherent source are training budget, not folding budget; add them in stage 3 only if a measured deficit demands it.

**If Cyril wants a real shrink.** One axis only [R2 law 27], priced on the real base [R2 law 32]: MLP `intermediate_size` coreset 17,408 → 15,360 (0.88x on the MLP, −2.0B → 25.0B, Q4_K_M 14.3 GiB). This is the estate's safest body axis (stream-selected channel folds, conditioned capture worth +3 cases [R2 laws 34, 47–48]) but on LFM2 even 0.856x MLP cost 5 of 15 cases; expect a stage-2 recovery bill, not a free fold. Excision of interior-late blocks (13/15 at 0.947x on LFM2 [R2 law 53]) is the other single-axis option (−0.38B per layer here) and must not be combined with the MLP fold.

**Iteration configs (same recipe, same tokenizer, same ledger, same F=6144, same 3:1 schedule and head dim 128):**

| | CO-1 | CO-7 |
|---|---|---|
| Dims | H=1536, L=24 (18+6), I=4,608, KDA 4 k / 12 v heads, MLA 12 heads latent 256 + rope 64, tied embeddings | H=4096, L=32 (24+8), I=12,288, KDA 16 k / 32 v heads, MLA 32 heads latent 512 + rope 64, untied |
| Params | 1.133B total (trunk 0.72B; the 252k tied table is 0.39B = 34%) | 9.108B total (trunk 6.97B; embeddings 2.07B = 23%) |
| Q4_K_M / KV q8_0 @256k | 0.65 GiB / 0.50 GiB | 5.20 GiB / 1.20 GiB |
| Fits a P40 in **fp32 torch** for training-side iteration | yes (4.5 GiB weights fp32) | no (36 GiB) → 3090/RTX 8000 bf16/fp16 |
| Fold source | ASSUMPTION: nearest Qwen3.5/3.6 small GDN hybrid (the 0.8B–2B class) with bf16 safetensors downloaded and dims read from its `config.json`; the shapes above are placeholders to be replaced by that model's, so the fold pipeline is identical at 1B and 27B | ASSUMPTION: Qwen3.5-9B class (the model llama.cpp's q4_0-KV measurement used [R5]); dims likewise to be replaced by its config |
| What it can and cannot prove | Trunk-side folds (T2–T5) and the media bridge/head/3D type end to end; NOT interface folds (interface share 34% here vs 9.6% at 27B — the interface law says the number does not transfer [R2 law 17]) | Everything at ¼ scale incl. the GGUF converter, the MLA FA path, and P40 inference; media flow at real geometry |

On the box today for immediate mechanism tests without downloads: Qwen3-4B-Base and Qwen3-0.6B in HF format (dense GQA, no GDN) [R4] — they can exercise T3 (GQA→MLA), T4 (SwiGLU→SiTU), T6 (ledger rows) and the converter, not T2.

## 7. What the hardware and the record make infeasible (honest)

1. **Kimi K3 cannot run here.** 1,453.74 GiB MXFP4 MoE on a CIFS share; llama.cpp support exists but its Q2_K is ≈700 GB [R5] against 184 GB VRAM + 251 GB RAM [R4]. No forward pass ⇒ no calibration activations ⇒ MergeMoE/ExFold/HARC-class closed-form expert folds (all need 16–40 calibration samples [R5]) are unavailable, and no published method collapses 896 experts to one dense FFN without ~4B distillation tokens [R5]. Its non-routed bf16 tensors (ESTIMATE from config: 69 KDA layers 30.5B + 24 MLA 5.6B + shared experts 12.2B + layer-0 FFN 0.73B + embeddings 2.35B ≈ 51B) are readable shard-wise but are larger than the whole target and live in a 7168-wide, 93-deep, 163,840-vocab space. What K3 realistically contributes: the architecture (already adopted, 3:1 KDA:gated-MLA, AttnRes, SiTU, gated MLA geometry [R3]) and one cheap, expected-to-fail weight-space test (T8). Distillation from K3 needs a host that can run it.
2. **The media models have no closed-form fold into an encoder-free trunk.** MiniMax H3 (20.1B pruned, ComfyUI repack only, no diffusers config), LTX 2.3 (23.1B), Qwen-Image (20.4B fp8), Hunyuan3D 2.1, ACE-Step 1.5 all operate on VAE latents [R4]; Chimera's tokens are raw pixels/samples and its tests forbid encoder modules [R3]. Their transformer blocks' input spaces do not exist in Chimera. Their role is teachers/data (ACE accepted as music specialist; LTX 2.3 rejected at 0.5–2.0/5; H3 quarantined pending the catalog applying Cyril's 2026-08-28 internal-use ruling [R3, R4]) — i.e., stage 3, which is exactly where the record and Cyril's own staged plan put the 90% recovery [R1 P25]. The single raw-pixel linear map on the box that can seed the bridge is Qwen3-VL-32B's patch embedding (T7).
3. **Fold ratio.** Every number in the record is one 230M model; "no wave has measured what any of this transfers to" [R2 §4]. The safe assumption for planning is net ≈1.0x with cache folds; anything below 0.9x on a 27B is a bet the record does not underwrite.
4. **Training-side VRAM.** Usable bf16/fp16 training cards = 2× 3090 + RTX 8000 = 94 GiB; the P40s are fp32-only. Full-parameter recovery of a 27B (bf16 weights 50 GiB + grads 50 + Adam 200) is impossible on this box; stage 2/3 must be parameter-efficient or offloaded — the training section owns this, but the architecture must not assume full fine-tuning.
5. **Instrument.** The 15-case review lacks one-case resolution and is 89.6% one-dimensional [R2 laws 8–9]; at 27B the grade must be the benchmark-lab suite [R4] plus Cyril's own scoring [R1 P26], behavior-first and artifact-tolerant [R1 P44].
6. **256k prefill is slow everywhere** (sequential GDN kernel): minutes on the 3090, tens of minutes on the P40 [R5] — a property of today's llama.cpp, not of the design; a chunked kernel is upstream's TODO.

## 8. Fold tests this section needs (closed-form, no gradients, each graded behavior-first, each one axis on the real base)

| # | Test | Pass criterion | Cost |
|---|---|---|---|
| T1 | Exact embed of Qwen3.8-27B into Chimera modules (F1, GDN decay, SiLU gate, β=∞, AttnRes block 64) | bit-exact logits on a fixed prompt pack; hash-verified (verify artifacts, not exit codes [R2 law 62]) | CPU, hours (needs the bf16 download) |
| T2 | GDN per-head softplus decay → KDA per-channel bounded decay, least-squares refit of the decay MLP on calibration logits, clip at e⁻⁵ | retained intelligence at parent band on CO-1 and CO-7 before 27B | CPU + one 3090 |
| T3 | 16 full layers: [W_k;W_v] rank-512 fold (SVD then conditioned capture on stream activations [R2 law 34]) + 4→1 rope-key merge; intermediate exact latent-1,792 check | ≥ parent band − 1 case on the 15-prompt pack and no benchmark-lab collapse; this is a **cache** fold (102.8M vs 104.9M params/layer), so grade only on behavior | CPU + 3090 pair for activations |
| T4 | SiTU finite β (4/25) vs ∞ | zero-loss or explain which layers clip | CPU |
| T5 | (negative control) AttnRes block 12 with zero-init query | expected to damage (uniform mixing) — documents that AttnRes is a stage-3 activation, not a fold | CPU |
| T6 | +4,096 ledger rows, zero-init | bit-exact on text; media placeholder tokens route through the bridge in the proofs | CPU |
| T7 | Qwen3-VL-32B `patch_embed` (Conv3d on raw 2×14×14×3 pixels) → `feature_proj` via least squares onto 32×32 patches | measured media MSE at step 0 vs random init on the v4 proof geometry | one 3090 |
| T8 | K3 → CO-27 attention prior via Procrustes on shared-token embeddings (no K3 forward) | expected fail per the record (cos 0.97 → zero transfer [R5]); run once, cheap, then drop K3 from the fold stage if it fails | CPU, CIFS shard reads |

Decision points for Cyril, in order: (1) approve the Qwen skeleton + absorption reading of "fold my models into one"; (2) rule on the companion-GGUF + driver as "standard export" for media; (3) approve the Qwen3.8-27B bf16 download and a llama.cpp/CUDA-12.8 build on the AI server; (4) choose F2 (MLA cache fold, T3) as the deployment default vs F1 + q4_0 KV.