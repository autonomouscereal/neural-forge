# Adversarial review — Chimera Omni plan (Planners 1–4 vs the estate)

Verified in this pass: parameter/KV/budget arithmetic recomputed (scratchpad run), and `chimera.py` lines 499–527 read to settle the AttnRes identity claim.

## A. Required fixes, by category

### A1. Rule violations

1. **Planner 1 recommends a non-standard-export child as the deployment default.** CO-27 "F2" (gated-MLA-folded full layers) has no llama.cpp architecture; Planner 1 itself says it needs "a pinned fork build" or an upstream patch. Under Cyril's R3 ruling ("[Cyril quote withheld]") that is not a deliverable until the arch is merged upstream. **Fix:** deployment default = F1 (Qwen-native gated GQA, `qwen35` arch) with q4_0 KV: 15.44 + 4.50 + 0.14 + 1.5 = **21.6 GiB**, fits both 24 GiB cards. F2 becomes a stage-3 option contingent on (a) an upstream llama.cpp arch PR and (b) a fold test that survives Cyril's scoring.

2. **F2's fold is contradicted by the record, not just unexported.** Reaching llama.cpp's only MLA FA shape (DKQ 576 = 512 latent + 64 rope) from Qwen's 4 KV heads × 256 requires (i) a 4→1 merge of the per-head RoPE keys and (ii) a ~3.5x low-rank fold of NoPE K/V at all 16 attention sites simultaneously. Record: 16→8 head merge 0/15 with any solve; three stacked attention folds collapse the decoder; all-sites-at-once folds die where 3-at-a-time survive (laws 43, 46, 52). Planner 2 rejects the MLA route outright; Planner 3's T3 keeps rope-64 per head (DKQ 768), which Planner 1's own F2b row shows has no FA kernel (12.9 GB KQ scratch). The three planners are mutually inconsistent here. **Fix:** strike F2/T3-MLA from the fold stage; record it as "trained-delta only, needs arch entry"; the KV lever in the fold stage is full-layer excision (each removed full layer = −6.25% KV, −372M params) plus KV-cache quantization at inference (quantization-last compliant).

3. **Invented gates.** Planner 1 T3 "≥ parent band − 1 case"; Planner 4 T1 "≥3 cases over excision control", T2 "≥3 cases over M1", T3 "arm E ≥30% top-1" (an LM-scored proxy, not behavior), T6-A "needle ≥90%, ≥3 t/s"; Planner 3 "fold-only −3, post-recovery −1". Cyril's standing rule: never invent gates, he scores. **Fix:** relabel every threshold as "proposed decision line for Cyril's ratification"; T3's top-1 probe is a reported measurement, never a go/no-go; the envelope numbers (VRAM, t/s, needle) are engineering measurements, and the plan must say Cyril sets the acceptable floor.

4. **Planner 3 misplaces the never-run stage-2 measurement.** It schedules "the stage-2 measurement at 0.6B" inside Stage 1, but the sealed stage-2 design, trainer ae4b8fb5, 26 corpus shards and the four arms are LFM2.5-230M artifacts; no Qwen-0.6B fold children exist. **Fix:** run it as Planner 4's T4 on the LFM2 children first (and correct its cost, see A3.6); a Qwen-sibling repeat only after Stage 1 folds exist.

5. **"Floor of what we built" misstatement.** Planner 3 §3: "fold-only coherence below ~0.6x has never been shown (0/15 at every 0.5x set)". Record: V12CL896G03 4/15 at 0.451x, V9CL896G10 4–5 at 0.437x, 14 of 17 sub-0.5x children hold intelligence under the canonical lens; what was never achieved is the **parent band** at ≤0.5x. Planner 1's "nothing within 2.5x has held a single case" is also off: the floor child at 2.43x holds one. **Fix:** state it as "retained intelligence to 0.4116x; parent-band coherence never below ~0.62x".

6. **Scope creep outside Cyril's source list.** Planner 4 T2 downloads OLMoE-1B-7B and DeepSeek-V2-Lite (~46 GB, 10–15 GPU-h) to license a MoE→dense mechanism that cannot be applied to K3 on this hardware anyway (1.45 TiB on CIFS vs 435 GB RAM+VRAM). **Fix:** defer T2 until a K3-capable host exists; K3 = architecture donor only, which all four planners already conclude.

### A2. Hardware unreality

7. **Torch-side captures on the RTX 8000 for GDN layers.** Planner 4 T5 and Planner 2 route 27B evidence capture through GPU6 (Turing, no bf16). Qwen3.5+ GDN needs `flash-linear-attention` Triton kernels; Turing lacks native bf16 and Qwen's massive activations are an fp16 overflow risk (Planner 3 R11). **Fix:** layer-stream captures on a 3090 (one layer 0.55 GB + 1M-token activations 9.5 GiB fits 24 GB); use GPU6 for the bf16→fp16 review split only after an overflow check; verify `fla` installs against torch 2.11.0+cu128/triton 3.6.0 (unverified assumption).

8. **Storage for a 27B fold matrix.** Planner 4 T5 prices "a 100-candidate 27B matrix"; each materialized child is 54 GB bf16 → 5.4 TB against 615 GB free on `/` and 1.6 TB on the CIFS share. **Fix:** children are stored as (config + channel-index lists + refit tensors), priced on the real base in RAM, and only the best 5 (Cyril's P23 rule) are materialized, to `/mnt/ai_data`.

9. **H3 residual-chart teacher line (Planner 2 §2.3).** Driving H3 from the child's states via a Procrustes chart into Qwen3-VL-32B's residual requires the 48 GB bf16 conditioner in torch (RTX 8000 + 3090) plus H3 (19.5–37.5 GB) plus VAEs, i.e. all three bf16-capable GPUs for teachers and none for the student, plus re-plumbing ComfyUI's H3 conditioning. **Fix:** H3 = offline teacher clips only (the existing GPU4 container already proves that path); drop the chart line from phase 1.

10. **GPU ownership.** `comfy-gpu-owners.json` reserves GPU6 to `codex-h3-breck`; every planner assumes GPU6 as the main capture/training card. **Fix:** claim GPU6 and GPU4 through the relay before Stage 0 (Cyril's 2026-08-28 "take all GPU power" ruling applies, but Codex must be told).

11. **Months, stated plainly.** Planner 3's 5–8 months and Planner 2's step 6 both depend on ≈180 tok/s LoRA on a 27B split across RTX 8000 + 3090 (unmeasured). 100M tokens = 6.4 days at that rate; if it lands at <100 tok/s the 27B graft/recovery loop is ≥2 weeks per 100M tokens. **Fix:** measure it on day 1 of the 27B stage; make the small-ladder (≤4B, full-parameter on RTX 8000) the place where fold+recovery is proven before any 27B training is scheduled.

### A3. Arithmetic corrections

12. **Fold ratio needed for q8_0 KV at 256k.** Planner 3 targets "0.80–0.85x". At 0.85x: 13.12 + 8.50 + 0.14 + 1.5 = **23.26 GiB** (0.24 GiB margin, effectively no fit). At 0.80x: **22.49 GiB**. **Fix:** ≤0.80x for q8_0 KV with all 16 full layers, or excise 4 full layers (12 × 2 × 4 × 256 = 24,576 values/token → **6.38 GiB** q8_0) and keep the body near 1.0x.

13. **Parameter envelope for Cyril's "25–30B".** Q4_K_M = 0.571 GiB per B params (16.5 GB / 26.90B), usable 23.5 GiB, buffers 1.5 GiB, state 0.14 GiB: F1 + q8_0 KV → **≤23.4B**; F1 with 4 full layers excised + q8_0 → **≤27.1B**; F1 + q4_0 KV → **≤30.4B**; F2 + q8_0 → ≤34.1B (non-standard, see A1.1–2). So 25–30B is reachable only with q4_0 KV or ≥4 full-layer excisions.

14. **Audio token rates (2x errors in both directions).** 32 kHz, 1,536-sample windows, stride = segment (as in v4): **20.8 tok/s per channel, 41.7 tok/s stereo**. Planner 1 wrote "20.8 tok/s stereo" (half); Planner 3 wrote "~420 tokens per 5 s stereo" and "10 s ≈ 830 tokens" (double): correct values **208** and **416** tokens. Planner 1's 48 kHz / 6,144-sample row (15.6 tok/s stereo) is right.

15. **Planner 2's GDN-layer count.** "≈4.1B excluding FFN" contradicts its own earlier 116M × 48; correct is **5.56B** (per layer 5120×16,480 + 10,240×4 + 6,144×5,120 = 115.9M). Totals: FFN 17.11B, GDN 5.56B, full 1.68B, embed+head 2.54B (untied — still an assumption until `config.json` is read), **26.90B**.

16. **Planner 4 T4 cost.** 2.3 h/arm assumes 25 TFLOP/s effective; the stage-2 design's own rate (4 arms × 1.12B tokens in 145–150 GPU-h) is ≈8,500 tok/s → 200M tokens = **6.6 h/arm**, 3 arms ≈ 20 GPU-h, 4 arms ≈ 27 GPU-h (plus evals). Still cheap; still the highest-value unrun test.

17. **Patch size inconsistency.** Planner 1 uses 32×32 patches (1024² image = 1,024 tokens); Planner 2 uses patch 16 (1024² = 4,096 tokens). **Fix:** one geometry (32×32, feature 3,072 = repo default; video tubelet-2 = 6,144 → `native_feature_dim` must rise to 6,144, which Planner 1 states and Planner 3 flags as a config change).

18. **Ledger size inconsistency.** Planner 1 appends 4,096 rows (41.9M params); Planner 2 needs ≥65,536 media rows (0.67B) for ACE codes. **Fix:** tie to decision D7 (audio-as-vocabulary vs raw waveform); the budget table must carry whichever is chosen.

### A4. Claims contradicted by the record or the code

19. **AttnRes "exact at block size = L" (Planner 1) is false by the code.** `chimera.py` 505–527: at `layer_index 0`, `0 % attnres_block_size == 0` pushes the embedding as a completed block and sets `prefix_sum = None`; the attention output then starts a new prefix, and every later `feed_forward_residual`/`attention_residual` is a zero-init (uniform) softmax over `[x, prefix]` = 0.5·x + 0.5·prefix, never x + prefix. The only exact embed is `residual_mode="none"` (Planner 2) or a new `attnres_init="identity"` flag with a large fixed logit on the prefix source (Planner 2/3 T4). **Fix:** T1 exact-embed uses `residual_mode="none"`; AttnRes stays a stage-3 activation.

20. **Planner 2 Plan A (MLP coreset avg 0.6 + 2–4 full-layer excisions) is the record's death pattern, presented as the main plan.** LFM2: all-sites MLP at 0.6 ≈ CL1536 = 4/15 of 12; excision ladder 1→13, 2→7, 3→3, 4→0; MLP+excision at depth 1/15; second structural step fatal on five axes. Planner 2 says the pair "LFM2 never held" but still targets ≈0.70x. **Fix:** present Plan A as a build-don't-predict measurement on the 4B sibling with the recorded prior (≤2/15 before recovery), not as the 27B recipe; the 27B fold ladder is exact-embed (1.0x) → single-axis excision census → single-axis MLP rungs 0.90/0.85 → one pair only if the 4B pair held.

21. **"Standard export proven on P40" (Planner 1 §0.4).** The P40 evidence is Qwen3.6-35B-A3B at 128k on WSL2, not a dense 27B at 256k. **Fix:** "shown for a smaller `qwen35` hybrid at 128k; 27B @256k on sm_61 is unmeasured (T6-A)".

22. **NoPE-dim contraction cell (Planner 2 item 3).** head_dim 192/160 changes FA kernel eligibility: llama.cpp's vec/tile kernels on Pascal are built for D = 64/128/256; D=192 exists for the deepseek path but quantized-KV support at 192 on cc 6.1 is unverified. **Fix:** this cell must clear a P40 FA + q4_0/q8_0 KV smoke before it is built at 27B.

23. **Vocab coreset at 27B.** Planner 2 keeps it as option D4. Beyond the +15.8% sequence-length tax at 256k, a 248k BPE deletion requires pruning merges consistently and re-emitting a GGUF tokenizer; the record notes transformers 5.12.1 refuses many-to-one tokenizers nondeterministically. Cyril: "[Cyril quote withheld]". **Fix:** off by default at 27B (Planner 3 item 8); 4B-only data point.

### A5. Missing pieces

24. **FA + quantized KV + head_dim 256 on Pascal.** Qwen3.8-27B full layers are 4 KV × 256. llama.cpp's quantized-KV FA kernels have historically covered D=128 fully and D=256 only for F16/F16; issue #21385's BLEU 1.000 result does not state the backend. If q4_0/q8_0 KV at D=256 is unsupported on cc 6.1, llama.cpp falls back silently to CPU attention (issue #24485 pattern). **Fix:** T6-A step 2 must assert the FA kernel actually ran (no fallback log line) at both KV types on the P40; if it fails, the P40 envelope is f16 KV = 16 GiB → 27B does not fit on P40 at 256k with any standard child.

25. **llama-server allocates KV for `n_ctx × n_parallel`.** All budget tables assume one sequence. **Fix:** `-np 1`, and state it.

26. **Media-driver plumbing on llama.cpp is unproven.** The x0 loop needs the prefix KV and GDN recurrent state restored 16 times per generation; recurrent memory does not support partial `seq_rm`, so the driver must use `seq_cp` onto a scratch seq_id per step. Feasible, unverified, and not "stock llama.cpp runs the model". **Fix:** week-1 verification list (embd input on hybrid archs, per-token final hidden with pooling none, non-causal batch over a cached prefix, seq_cp/seq_rm on hybrid memory) and Cyril's R3 ruling on "trunk GGUF + companion GGUF + driver".

27. **Image understanding after dropping the ViT.** Qwen3.8-27B's image understanding lives in its SigLIP-class tower (0.87 GiB mmproj). Encoder-free per P9 means it is dropped and re-learned through the native bridge — a retraining line with no fold path (SmolVLM2 native fold reached 50.4% top-1, not parity). Planner 1 omits this cost entirely. **Fix:** add to the "needs retraining" table and the Stage-1 risk list.

28. **Data gaps not on any planner's critical path:** no ≥32k-token evidence corpus exists (needed to price full-layer excision honestly for 256k); registry integrity after the 08-19/08-22 cleanups unverified; TTS/ASR teachers absent; H3 catalog still fail-closed pending relay 8324ea96. **Fix:** Stage 0 deliverables, blocking.

29. **Qwen3.8-27B is in flight, not absent.** READ 4 saw pid 2227173 ("qwen dl"); Planner 2 records the 18-shard bf16 download launched 2026-09-01. Planners 3/4 say "not on the box". **Fix:** hash-verify on landing; nothing at 27B schedules before that.

30. **Codex is owed a requirement-change brief** (P6/P9/P13/P19/P39): the pivot to Chimera Omni, GPU claims, and the H3 gate flip. No planner lists it as a step.

## B. Top 5 program-killing risks

1. **"Fold my models into one" has a closed-form path for exactly one donor (Qwen).** K3 cannot run here (1.45 TiB vs 435 GB memory; 896→1 expert collapse has no method); H3/Qwen-Image/LTX/Hunyuan/ACE-DiT live on VAE latents with no paired residual space; the field's cross-model result at cos 0.97 was zero functional transfer. Every donor except Qwen enters as a teacher, i.e. distillation — which Cyril's own stage order puts after the fold ceiling. If he does not ratify that reading (Planner 2 D1/D6, Planner 3 rulings 3–4), the program has no executable stage 1.

2. **The fold stage may buy nothing the stock model does not already have.** Stock Qwen3.8-27B Q4_K_M + q4_0 KV already fits 24 GiB at 256k (21.6–22.1 GiB). Fold headroom only matters for q8_0 KV or media rows, and the record predicts the required two-axis child (MLP + excision) dies before recovery. No LFM2 law has been re-measured on a second model. Earliest test: Stage-1 matrix on the 4B sibling + T4.

3. **27B recovery/graft training is unproven on this box.** Full-parameter is impossible (≈430 GB Adam); LoRA on a split RTX 8000 + 3090 trunk at an assumed 180 tok/s, fp16 on Turing with Qwen massive activations, no FSDP/DeepSpeed/bnb/peft stack, and nothing larger than 230M ever trained here. All-modality at 27B is months per iteration; ten modalities at real geometry are ~200M media tokens minimum. Earliest test: day-1 tok/s measurement.

4. **No inference engine exists for media output at Q4 on a 24 GB card.** llama.cpp does text; mtmd does encoder-based input; nothing runs the native bridge/head or x0 sampler; torch on P40 is fp32-only (108 GB for 27B). The custom driver over llama.cpp APIs is plausible but unverified (item 26). The end goal "all modalities in and out on a 24 GB card" is an engine project independent of folding.

5. **The P40/llama.cpp envelope is unmeasured and may fail.** CUDA 12.8 build needed (no nvcc on host; docker image arch set unknown); FA with quantized KV at head_dim 256 on cc 6.1 unverified (fallback = 16 GiB f16 KV = no fit); sequential GDN prefill 15–30 min per 256k on P40; DKQ=576 MLA FA on Pascal unverified. If T6-A fails, the only escape (F2/MLA) is both non-standard export and the record's deadliest fold class.

Secondary but real: encoder-free image understanding at Qwen quality has no fold path (item 27); no long-context evidence corpus exists (item 28); GPU6 is Codex-owned until claimed (item 10).