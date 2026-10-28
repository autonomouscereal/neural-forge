# Candidate review: TiDAR and NVIDIA diffusion "thinking" for Chimera Omni v2

Revision 2 of 2026-10-28. Revision 1 (2026-10-22) is at `backups/cand_tidar.20261022.md` (sha256 a6e32836...a5e4). This is a read-only review: nothing was trained, launched or changed. Cyril's test applies: "[Cyril quote withheld]". All web sources were fetched on 2026-10-22 or 2026-10-28 and are listed at the end.

Tags:
- **[P]** primary source: paper, slides, model card or project page, read in full or at the section level.
- **[S]** secondary source, or a primary source read only at abstract level.
- **[R]** our own records.
- **[I]** my inference. Nothing in this file was measured by us.

## What changed since revision 1

1. **New and directly on point: dQwen3.5** (UT Austin, arXiv 2609.20751, 17 Sep 2026) [P]. It adapts **Qwen3.5 0.8B/2B/4B/9B into masked-diffusion LMs** while keeping the Gated DeltaNet (GDN) layers *causal* and making only the attention layers (6 of 24 at 2B) bidirectional. It works, it reaches a given loss in about half the tokens of a full-attention control, and Apache-2.0 weights are on HF. Revision 1 said "draft quality on a 3:1 GDN hybrid is unpublished". That is now **partly answered: the diffusion side works on our exact trunk shape.** Self-speculation (diffusion draft + AR verify) on a GDN hybrid is still unpublished.
2. **New phone-NPU evidence: llada.cpp** (Beihang/Tsinghua, arXiv 2606.13740, Jun 2026) [P]. It is the first published diffusion-LM runtime on Hexagon NPUs (Snapdragon 8 Gen 3 / 8 Elite / 8 Elite Gen 5). Its per-forward cost table is the first real data on how a k-token forward scales on Hexagon. Read naively, it puts a drafting scheme **at or near our kill line** (§5.2). That lowers my expectation for A-DIFF on the phone. The P-SPEC microbenchmark still decides it.
3. **A free, zero-training acceptance test now exists.** dQwen3.5-2B-Base can draft and stock Qwen3.5-2B can verify. That measures acceptance on our own prompts before we spend any P40 time (§7, step 1).
4. **Correction.** Revision 1 said NLD's SFT "doubles only the response segment (L_text + L instead of 2L)". That was wrong. The quoted L_text + L is NLD's **VLM** asymmetric dual stream, which drops never-masked vision tokens from the noisy half (§5.3). NLD's SFT computes loss only on the answer and does not mask the prompt (§5.2) [P]. Applying the same drop to our prompt and media tokens is my inference, not a published result [I].
5. **Confirmed.** NLD's quadratic self-speculation cites TiDAR (its ref [20]). Its joint-training attention pattern follows Set Block Decoding (ref [14], Meta, 2509.04185). Its conclusion is quoted verbatim: "Linear self-speculation achieves the best efficiency in the current infrastructure" (§8) [P].
6. **Still true.** TiDAR has released no code and no weights. The project page still says "SGLang inference code will be released soon".

## Verdict

**What he means: identified with high confidence.** Cyril's paper is **TiDAR: Think in Diffusion, Talk in Autoregression** [P].
- arXiv 2511.08923 v1, 12 Nov 2025, "NVIDIA-Tech Report".
- Authors: Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang and Pavlo Molchanov.
- **MLSys 2026 oral** (19 May 2026), in the proceedings.
- His description matches every point: NVIDIA, diffusion "thinking", attention-based AR "talking", and one forward pass.

**One correction to his memory.** TiDAR's "thinking" is **not chain-of-thought**. It is the parallel drafting of the next block of tokens. Work that diffuses the reasoning itself does exist (LaDiR, DCoLT, Uno-style diffusion-augmented LMs), but none of it is NVIDIA's (§1).

**The NVIDIA follow-up matters more than TiDAR.** It is **Nemotron-Labs-Diffusion (NLD)**, arXiv 2607.05722 (July 2026; EMNLP 2026 main), from the same group, with TiDAR's first author as a co-author [P].
- It ships weights: 3B, 8B and 14B, in base, instruct and VLM variants.
- It runs TiDAR as its "quadratic self-speculation" mode, and concludes the two-pass **linear** mode is the most efficient on real devices.

**Recommendation:**
1. **Run 0: change nothing.** A `<|mask|>` row is not a run-0 requirement. Teacher KL covers only the first 248,320 rows, so appending a row later is as cheap as appending it now. If the special-token block is being written anyway, one unused row is harmless housekeeping. This matches ARCH_CANDIDATES §4 [R].
2. **Reject TiDAR's one-pass quadratic layout for the phone.**
   - At draft 16 a step is q_len = 16 × 17 = 272 query tokens [P formula].
   - On our hybrid it also needs one GDN state fork per possible acceptance point: 17 × 18 MiB f32, about 306 MiB of state traffic per step [R state size, I].
   - Its premise is "free token slots" on an H100 at batch 1. llada.cpp shows a k-token forward on Hexagon is sublinear but **not free** (§5.2).
3. **Keep one gated later arm, A-DIFF: NLD-style joint AR + diffusion text training with linear self-speculation.**
   - It adds zero parameters, so it is compatible with the Qwen plant.
   - It runs after run 0 and not inside the fold pass, where it would confound the fold result.
   - It is the only route in this literature with batch-1, bandwidth-bound wins over MTP and EAGLE-3: DGX Spark 3.14× over AR at FP8 and 1.8× over EAGLE-3 [P].
   - It ships only if the phone microbenchmark shows an end-to-end gain and behaviour holds.
   - The order is now: (a) **P-SPEC**, a phone k-token cost curve, no training; (b) **ACC-0**, dQwen3.5-2B drafting for Qwen3.5-2B on the server, no training; (c) a pilot only if both pass (§7).
4. **For media, test one idea, as scheduling only.** Fuse chunk *n*'s causal commit pass with chunk *n+1*'s first flow-refinement step in a single forward. That is TiDAR's "verify old, draft new" applied to our x0-flow chunks.
   - It needs no training change, because our noisy span already attends a clean prefix.
   - For audio at 4 flow steps it saves one forward in five, about 20 % [I].
   - It belongs in the A13 / P1 phone experiments.
5. **TiDAR does not otherwise make images or video faster.** Our media path already *is* "diffusion in chunks with refinement passes", which is what Cyril described. Media cost is (flow steps) × (span tokens per forward), and TiDAR changes neither. Media speed comes from step distillation (S6, 16 → 4 [R]) and the A8 / global-share decisions.

## 1. Identification

| Search / fetch (2026-10-22 and 2026-10-28) | Result |
|---|---|
| WebSearch (extended): "NVIDIA diffusion chain of thought reasoning language model 2025 hybrid autoregressive diffusion" | TiDAR (2511.08923), Nemotron-Labs-Diffusion (NVIDIA Research, May 2026), dQwen3.5 (2609.20751), LaDiR (2510.04573), REPR-ALIGN (2605.06885) |
| WebSearch: "TiDAR MLSys 2026", "TiDAR … accepted OR github" | MLSys 2026 oral 3751 (19 May 2026), proceedings entry, slides PDF, project page tidarlm.github.io |
| WebFetch: arXiv abs + HTML v1, MLSys proceedings, project page | Method and results (§2). No code or weights as of 2026-10-28 |
| WebFetch: NLD research page, HF card, arXiv HTML (body + references) | §3, including refs [14] and [20] and verbatim conclusions |
| WebSearch (extended): NVIDIA latent / diffusion reasoning 2026 | LaDiR, Uni-LaDiR (2609.19878), latent-token DLMs (2602.03769); NVIDIA LPR lists DiLaDiff and Esoteric LMs, neither a CoT-diffusion hybrid in snippets [S] |
| WebSearch (extended): diffusion self-speculation on phone NPU | llada.cpp (2606.13740), mobile speculative decoding (2510.15312) |
| WebSearch (extended): TiDAR + Gated DeltaNet hybrid | Nothing combining them; Gated DeltaNet-2 (2605.22791) unrelated |

**Candidates, and why each is or is not "it":**

| Candidate | NVIDIA? | Diffusion "thinks", AR "talks"? | Verdict |
|---|---|---|---|
| **TiDAR** (2511.08923, MLSys 2026) | Yes | Yes. It is the title, and it happens in one forward pass | **This is it** |
| **Nemotron-Labs-Diffusion** (2607.05722) | Yes, same group | Diffusion drafts, AR verifies (self-speculation), three modes in one model | The product follow-up (§3) |
| Fast-dLLM v2 (2509.26328, ICLR 2026; HKU/NVIDIA/MIT) | Partly | Block diffusion converted from Qwen2.5 in about 1B tokens, 2.54× over Qwen2.5-7B-Instruct; no AR verifier | Related, not it [S] |
| Efficient-DLM (2512.14067; Georgia Tech/MIT/NVIDIA) | Partly | AR-to-DLM conversion study at 1.5B/4B/8B | Related, not it [S] |
| LaDiR (2510.04573, ICLR 2026) | No; affiliations not on arXiv page; Apple ML Research hosts a page | Reasoning diffused as VAE-latent "thought blocks", then AR answer | Closest to "diffusion chain of thought". Needs a VAE encoder, which is a tower. Not NVIDIA [S] |
| DCoLT (2505.10446) | No | Whole model is diffusion; reverse steps are "lateral thoughts" trained with RL | Not a hybrid [S] |
| Uno / Ψ-Spec (2609.04010, Sep 2026) | Not stated | AR weights plus lightweight diffusion weights, *lossless* parallel sampling, up to 3× over base AR | Newest relative; adds parameters; abstract only [S] |

## 2. TiDAR: what it is [P unless tagged]

### 2.1 Mechanism
- **Base and parameters.** The base is a standard AR transformer. Every base is pure attention: Qwen2.5-1.5B and Qwen3-8B. Qwen3-4B appears only for a Block Diffusion baseline. **No new parameters.** Only the attention mask and a mask token change.
- **Training layout.**
  - The sequence is doubled: a clean copy plus an appended copy made entirely of `[mask]` tokens. This is the "full mask" strategy.
  - The clean part uses causal attention and gets the next-token loss, with labels shifted by one.
  - The mask part attends the clean prefix, is bidirectional within its block, and gets the diffusion loss, with labels aligned.
- **Loss.** L = 1/(1+α) · (α·L_AR + L_Diff), with α = 1 in most runs.
  - Full masking beat random masking: average 48.42 vs 45.45 at 4 drafts and 48.94 vs 45.68 at 8 drafts (Table 5).
- **Inference (the "one pass").** Each forward does three things.
  1. It verifies the previous step's draft causally, by rejection sampling against the AR distribution.
  2. It pre-drafts the next block by one-step diffusion, *for every possible acceptance outcome*.
  3. It keeps exact KV for accepted tokens, evicts rejected ones, and recomputes nothing.
- **The mask is pre-built.** It has size q_len × (q_len + max_seq_len), with q_len = block × (1 + block), and is sliced per step under Flex Attention. The per-step query count therefore grows quadratically with draft length.
- **Draft length.** The deployed draft length is 16; the paper tests 4, 8 and 16.
- **Mixing knob.** β mixes the AR and diffusion logits at sampling time ("trust AR" vs "trust diffusion").
- **Tuning.** There are no decoding hyperparameters to tune.

### 2.2 Training recipe (Sec. 4.1)
- **Continued pretraining from an AR checkpoint.**
  - 1.5B: 50B tokens, global batch 2M tokens.
  - 8B: 150B tokens.
- **Settings.** Sequence length 4,096. Distributed Adam. Cosine schedule with LR 1e-5 → 3e-6 and 1 % warm-up. BF16. Gradient checkpointing at 8B.
- **Stack.** H100s, Megatron-LM with TorchTitan.
- **Data source.** Not specified.

### 2.3 Results
**Generative (Table 2):**

| Model | HumanEval | MBPP | GSM8K | Minerva Math | 6-task average | Tokens per forward (T/NFE) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 35.98 | 43.60 | 54.74 | 26.40 | 41.64 | 1 |
| **TiDAR 1.5B** | 43.29 | 41.40 | 53.90 | 25.48 | **44.03** | 7.45 |
| Qwen3-8B | 64.63 | 69.40 | 81.80 | 52.94 | 68.09 | 1 |
| TiDAR 8B (trust diffusion) | 57.93 | 65.40 | 80.44 | 51.64 | 65.31 | 8.25 |
| Dream 7B | 54.88 | 56.80 | 77.18 | 39.60 | 58.74 | 1 (as run) |
| LLaDA 8B | 32.32 | 40.80 | 70.96 | 27.30 | 41.78 | 1 (as run) |

**Likelihood average (Table 3), computed in one causal pass:**
- TiDAR 1.5B: 64.43 vs Qwen2.5-1.5B 65.16.
- TiDAR 8B: 75.40 vs Qwen3-8B 74.25.

**Draft-length trade-off (Table 4, 1.5B):**
- 4 drafts: 3.47 T/NFE.
- 8 drafts: 5.49 T/NFE.
- 16 drafts: 6.97 T/NFE.

**Throughput:**
- **4.71×** over Qwen2.5-1.5B and **5.91×** over Qwen3-8B.
- Measured on **one H100, batch size 1, native PyTorch + Flex Attention**.
- It beats EAGLE-3 (public Qwen3-8B-Instruct EAGLE-3 weights), with numbers in Figure 4 only.
- The project page adds an SGLang figure: up to 5.8× AR and over 2.5× EAGLE-3, also on one H100.
- The premise is "free token slots": on an H100 at batch 1, a forward over K tokens costs about the same as over 1 when K is small. It was profiled on Qwen3-32B with FlashAttention-2.

### 2.4 Caveats
- **AR baselines are not equal-token.** The baselines did not get the extra 50B or 150B tokens, so part of the +2.4 at 1.5B may be extra training [I].
- **"Closes the gap" is not lossless.**
  - 1.5B likelihood is −0.73 below its base.
  - 8B generative is −2.78 below Qwen3-8B.
  - Output is not guaranteed identical to the base model.
- **Batch 1 only.** Larger batches are claimed zero-shot by shortening the draft, but no numbers are given.
- **Long context is deferred.** Training doubles sequence length, and the slides say advantages "may diminish" at long context.
- **No code, no weights.** As of 2026-10-28 the project page still says code is "coming soon".
- **Edge, mobile and multimodal: not discussed.**

## 3. Nemotron-Labs-Diffusion: the evidence that informs us [P]

### 3.1 Recipe
- **Tri-mode.** One model has three decoding modes: AR, block diffusion, and self-speculation (diffusion draft, AR verify, shared KV).
- **Training stream.** Training uses a dual stream (noisy + clean). The clean-to-clean mask is strictly causal, so both losses train in one pass without label leakage (§2.2).
- **Loss.** L = L_AR + 0.3·L_diff, averaged globally over tokens.
- **Initialisation.** Initialised from **Ministral3** (pure attention).
  - Stage 1: 1T tokens of AR.
  - Stage 2: 300B tokens of joint training.
- **Settings.** Sequence length 4,096, 256 H100s, WSD schedule with LR 1e-5 → 3e-6. SFT: 45B tokens at 16k, loss on answers only.
- **Release.** Weights on HF under the NVIDIA open model licence. HF usage shows `ar_generate`, `generate` (diffusion, `block_length=32`) and `linear_spec_generate`, plus an optional LoRA drafter.

### 3.2 The diffusion loss barely moves AR quality
The ablation is 25B tokens from Ministral3-8B, with and without the diffusion loss (Table 3):

| Model | Average without diffusion loss | Average with diffusion loss |
|---|---|---|
| Base | 72.50 | 72.64 |
| Instruct | 63.18 | 63.61 |

IFEval drops 3.01 points: "strict instruction-following compliance is slightly affected by the diffusion objective". HumanEval also drops slightly.

### 3.3 Our size class (Table 7, 3B instruct)

| Model | Mode | Average | Tokens per forward |
|---|---|---|---|
| Qwen3-4B | AR | 53.23 | 1.00 |
| NLD-3B | AR | 55.50 | 1.00 |
| NLD-3B | Diffusion | 52.90 | 1.91 |
| NLD-3B | Linear self-speculation | 55.00 | 4.36 |
| NLD-3B | Quadratic self-speculation (TiDAR) | 55.80 | 5.42 |

At 8B instruct the tokens per forward are 2.57 (diffusion), 5.99 (linear) and 6.38 (quadratic). Larger block lengths degrade accuracy (§4.2).

### 3.4 Against MTP and EAGLE-3 (Table 10, §6.5)

**Average acceptance length:**

| Method | Acceptance length |
|---|---|
| NLD, native | 5.46 |
| NLD, with the LoRA drafter | 6.82 |
| MTP (labelled Qwen3-9B-MTP) | 4.24 |
| EAGLE-3 | 2.75 |

**Batch-1 linear self-speculation on real hardware:**

| Device | Linear SS vs EAGLE-3 | Linear SS throughput (gain over AR) |
|---|---|---|
| GB200 | 2.4× | up to 3.3×; 1015 tok/s (3.97×) with an optimised kernel |
| RTX Pro 6000 | 2.3× | 277 tok/s FP8 (3.46×), 525 tok/s INT4 (2.35×) |
| DGX Spark | 1.8× | 77.5 tok/s FP8 (3.14×), 112.5 tok/s INT4 (2.69×) |

The DGX Spark is an LPDDR, bandwidth-bound box, the closest thing to a phone in the paper. The **gain shrinks at INT4** (2.69× vs 3.14×). That is relevant because our phone runs Q1/Q2 [I].

Self-speculation helps most at low concurrency, and AR is preferred for high-concurrency serving.

### 3.5 Linear vs quadratic
- Quadratic (TiDAR) has the higher tokens per forward, but "the real-device efficiency of quadratic self-speculation falls behind the linear one" (§6.1).
- "Linear self-speculation achieves the best efficiency in the current infrastructure", with quadratic "more promising at batch size 1 with improved infrastructure support" (§8).

### 3.6 Vision
The NLD VLM bolts on a vision encoder and an MLP projector taken from Ministral3-8B-Instruct. That is a tower, so it is excluded by Cyril's rule. One trick does transfer: never-masked tokens are removed from the noisy stream, giving length L_text + L instead of 2L (§5.3).

### 3.7 Community llama.cpp port [S]
The `buun-llama-cpp` fork adds `--diffusion-self-spec`. It reports 42-50 tok/s for the 14B Q8_0 on an RTX 3090 at k = 4, with no AR baseline. My rough AR ceiling for that setup is about 60 tok/s [I]. There is still **no evidence that the gain survives a llama.cpp-class runtime**, and Sprite runs on llama.cpp.

## 4. Related 2026 work that bears on us

### 4.1 dQwen3.5: diffusion on our exact trunk shape [P]
- **Scope.** Qwen3.5 0.8B/2B/4B/9B adapted to masked diffusion with a time-reweighted masked loss.
- **GDN handling.** GDN stays causal; only the attention layers drop their causal mask (6/24 at 0.8B and 2B, 8/32 at 4B and 9B).
- **Training.**
  - 50B or 100B tokens per size, 512 × 4,096-token sequences per step.
  - Data mix: 50 % code, 35 % text, 15 % math, from Nemotron-derived subsets.
  - 2B LR 2e-5, run on the TACC Vista cluster.
- **Efficiency.** It reaches the same training loss in about half the tokens of a full-attention Qwen3-1.7B control (median 2.21×).
- **2B against its AR parent (Table 2).**
  - Behind on MMLU, GSM8K and MATH500.
  - Equal on HumanEval (37.20).
  - Ahead on MBPP (34.20 vs 23.40) and MBPP+ (40.48 vs 31.75).
- **Decoding.** Parallel decoding "often remains ahead on code" (Fig. 6), but tokens-per-step for 2B are not in the text. The decode order stays largely left-to-right: global AR-ness about 0.88-0.97.
- **Throughput.** On one GH200 with FLA kernels the 2B is slower than the control at short sequences and faster at long ones (C.8, figure only).
- **Limits.** Base models only, no post-training, a single architecture pair, and "longer adaptation (100B) does not uniformly help".
- **Release.** `UT-IFML/dQwen3.5-2B-Base`, Apache-2.0, with a 50B checkpoint at revision `step25000-swa`. Default decoding is confidence-threshold τ = 0.9, or `block_length=32`.
- **What it means for us [I].**
  - A diffusion draft mode trained into a Qwen3.5-2B-shaped trunk is feasible with the 18 GDN layers left causal, and that is how our A-DIFF arm would be built.
  - The prior worry about GDN layers giving masked blocks no in-block context did not kill it, at least on code.
  - It gives an off-the-shelf drafter of identical shape for a zero-training acceptance test (§7).
  - It is *not* evidence for self-speculation speed.

### 4.2 llada.cpp: diffusion LM on a phone NPU [P]
- **Setup.**
  - Phones: OnePlus 12 (SM8650, 45 TOPS), Ace5 Pro (SM8750, 65 TOPS) and 15 (SM8850, 89 TOPS), all on Hexagon HMX/HVX.
  - Models: LLaDA-8B and Dream-7B at Q4_0, against Llama-3-8B.
- **k-token forward cost on the SM8750 NPU (Table 2, one LLaDA-8B denoising pass with prefix KV):**

  | Active tokens | Pass time | Per token |
  |---|---|---|
  | 32 | 478 ms | about 14.9 ms |
  | 64 | 862 ms | about 13.5 ms |
  | 128 | 1,532 ms | about 12.0 ms |

  The curve is sublinear. Figure 3 shows stage-like steps from tile padding.
- **End to end.** About 37.7× over CPU with prefix KV (Table 6). It is faster than Llama on 8 Elite and 8 Elite Gen 5, but **slower than AR Llama on 8 Gen 3** at 128 tokens. Its multi-block speculation cost accuracy (GSM8K 41.5 → 37.5) until dual-path revision restored it.
- **What it means for us [I, weak cross-system comparison].**
  - Our Bonsai-8B decodes 15-21 tok/s on the phone NPU, about 48-67 ms per token [R].
  - LLaDA's marginal cost is about 12-15 ms per extra token. Extrapolating, cost(k = 8) is roughly 48-67 + 7 × 12-15 ms, about 2.3-3.2 × cost(1).
  - That straddles the P-SPEC kill rule (2.5×).
  - If tile padding makes small-k forwards cost as much as k = 32, it is worse.
  - Different model, runtime and quant, so this only **lowers the prior**. It does not replace P-SPEC on our build.

### 4.3 Others
- **REPR-ALIGN** (2605.06885) [S]. Aligns DLM hidden states to a frozen AR copy per layer, reporting "up to 4× training acceleration", especially in low-data regimes. That matters if A-DIFF ever has to fit a P40 budget. Code is in pengzhangzhi/Open-dLLM. Token counts are not in the abstract.
- **Uno / Ψ-Spec** (2609.04010) [S]. "Diffusion-augmented LLMs" keep the AR distribution exactly ("lossless") and add lightweight diffusion weights trained by diffusion distillation. They report up to 3× over the base AR at every batch size tested. Lossless is attractive under our behaviour-retention rule. The extra weights are a drafting head, not a tower. Abstract only; worth a full read if A-DIFF gets that far.
- **LaDiR** (ICLR 2026) [S]. This is the real "diffusion chain of thought": VAE-latent thought blocks denoised, then an AR answer. It needs a VAE encoder, which conflicts with the no-tower rule. Revision 1 recorded a gain of 43.5 vs 42.0 averaged pass@1 on LLaMA 3.1 8B. Not useful to us now.

## 5. What it would change for Chimera Omni v2

### 5.1 Against each constraint

| Our constraint | Effect |
|---|---|
| **Qwen3.5-2B plant: 24 layers = 18 GDN + 6 gated GQA [R ARCH §3]** | Zero new parameters, so fully plant-compatible. dQwen3.5 now shows the diffusion side trains on exactly this shape with GDN causal [P]. Self-speculation on a GDN hybrid is unpublished. Untrained self-drafting on Qwen3.5's linear subgraph accepted only α = 0.038 (2605.01106) [S], so only a *trained* draft mode is plausible. |
| **4 P40s, fp32, about 33-46M tokens/day [R survey_hw]** | Dual-stream training roughly halves useful throughput, to about 17-23M/day [I]. Published budgets: TiDAR 1.5B used 50B tokens, NLD's ablation 25B, dQwen3.5 50B. At our rate even 25B is over a year. Any A-DIFF run would be 1-3B tokens, a large unproven extrapolation, softened only by REPR-ALIGN's low-data claim [S] and dQwen3.5's 2× hybrid efficiency [P]. |
| **Run 0: plant, then distill** | No change. Putting a second objective into run 0 would confound its only question: whether the plant plus distillation retains intelligence. |
| **Next pass: fold from parents** | Keep A-DIFF out of the fold pass for the same reason. It goes after, on a cloned checkpoint. |
| **50+ tok/s phone goal; spec and MTP measured poorly on our phone [R ARCH L247]** | A-DIFF is the only lever here with batch-1, bandwidth-bound wins (DGX Spark 2.7-3.1×). Against it: the INT4 gain shrinks, Q1/Q2 likely cuts acceptance (Bonsai fragility [R]), GDN rollback needs a state checkpoint (18 MiB, about 1 ms at 40 GB/s [I]; kernel K3 is on our list [R]), and llada.cpp's cost curve puts it near the kill line. If 1.5× survived, 2B Q2 short-context decode would go from 37-48 to about 55-72 tok/s [R projection × I]. **Unmeasured either way.** |
| **256k capable, 50k phone minimum** | The phone's exact decode at long context is KV-bandwidth floored (2B: 19-30 tok/s at 32k, 4-9 at 256k [R projection]). A k-query verify reads the KV once for k tokens, so drafting could amortise that floor, *if* our split-KV kernel takes multiple queries [I]. This is the opposite of TiDAR's H100 caveat and is worth including in P-SPEC. Training doubles only the response span if never-masked tokens are dropped from the noisy stream (the NLD VLM trick) [P, applied by I]. |
| **Media: x0-flow chunks; Cyril's "chunks with refinement passes"; joint refinement of all units** | Our media already is chunked diffusion with refinement. TiDAR adds only the fused commit-plus-next-refine forward (Verdict 4). A-DIFF's masked text block is the discrete form of our refinement mask: bidirectional inside the span, causal to the prefix, then committed. If it ever ships, text and media share one generation mechanism through one trunk. That fits the no-tower intent, but it is not a speed argument [I]. |
| **Output-decision and modality tokens** | No conflict. `<|mask|>` can be appended any time. |
| **Top-n matching, never top-1** | No conflict. Verification is against the AR distribution, and the top-k KL applies to the AR stream unchanged. Acceptance should be measured with sampled verification, not just greedy [I]. |
| **No towers, no sparse attention** | No conflict for the text recipe. The NLD VLM encoder and the LaDiR VAE are excluded. |

### 5.2 Cost
- **Zero-training gates.** P-SPEC is about 1 day of Claude-owned phone work under the phone lock. ACC-0 needs a few hours on CPU or one free P40, claimed via lanes.py, never GPU0/4/6.
- **Pilot, if both gates pass.** The trainer's doubled-stream mask is an extension of `native_bidirectional_refinement` [R]. The runtime needs draft and verify plus GDN checkpointing in our llama.cpp/Hexagon build. That is Claude-owned perf work [R], plus P40 time against the run-0 and fold budgets.

### 5.3 Risk
- **Behaviour drift.** NLD's IFEval fell 3 points, and our tool calls are fragile at low bits [R Bonsai].
- **Phone gain may be zero or negative,** as it was for MTP and speculative decoding.
- **Budget extrapolation.** Our tokens are roughly 1/20 of the smallest published budget.
- **Long-context interaction.** The doubled training sequence competes with 256k context-extension work.

## 6. Evidence quality

| Source | Quality | Notes |
|---|---|---|
| TiDAR (2511.08923) | **Moderate.** Peer-reviewed (MLSys oral), vendor-authored | One H100, batch 1. AR baselines not equal-token. EAGLE-3 numbers in a figure only. No code or weights. No edge data. |
| NLD (2607.05722) | **Moderate to good.** Vendor-authored; EMNLP 2026 | Weights and pipeline released. Several devices, including DGX Spark. Equal-token (25B) ablation. All pure attention. |
| dQwen3.5 (2609.20751) | **Moderate.** Academic preprint | Exact trunk match, weights released. Base only, one architecture pair, speed in figures only, no AR verification. |
| llada.cpp (2606.13740) | **Moderate.** Academic preprint | Real Hexagon measurements, but a different model, runtime and quant. 200-sample quality subsets. No tok/s table against AR. |
| Community llama.cpp port | **Weak [S]** | No AR baseline. |
| Fast-dLLM v2, Efficient-DLM, REPR-ALIGN, Uno, LaDiR, DCoLT, 2605.01106 | **[S], abstract level** | Orientation only. |
| Our phone numbers | **[R], with gaps** | Bonsai NPU prefill and decode are measured. Every co2-2b phone number is a projection (ARCH §7 says so). Why MTP and spec lost is not isolated. |

## 7. Proposed gated arm (not for run 0, not inside the fold pass)

**Step 1. P-SPEC (phone, no training; extends A13).**
- **Set up:** stock Qwen3.5-2B at Q2 and Q1 in our llama.cpp/Hexagon build, one HTP session.
- **Measure:**
  - the forward latency at k = 1, 2, 4, 8 and 16;
  - GDN state save and restore;
  - k-query attention at 4k and 32k.
- **Kill rule:** if cost(k = 8) ≥ about 2.5 × cost(k = 1) at short context *and* no long-context amortisation appears, drop A-DIFF and the MTP question for good. This also tells us why MTP lost.

**Step 2. ACC-0 (server, no training).**
- **Set up:** dQwen3.5-2B-Base drafts blocks of k = 4/8/16 (one step, and τ = 0.9). Stock Qwen3.5-2B verifies with greedy and sampled rejection.
- **Prompts:** our scorecard and tool-call replay prompts, with the raw capture spool excluded.
- **Measure:** acceptance length.
- **Note:** this uses two models, so it is only a proxy for a self-speculating co2-2b, not the ship configuration. A model trained jointly as in NLD might accept more or fewer tokens.
- **Gate:** expected speedup = acceptance-implied tokens per cycle × cost(1)/cost(k), taken from P-SPEC, ≥ 1.3× [I].

**Step 3. Pilot, only if 1 and 2 pass.**
- Run on a clone of the post-run-0 checkpoint, 0.8B first, then 2B.
- Stage 1 is our AR distillation; stage 2 is joint at α = 0.3 with global token averaging.
- Never-masked tokens are dropped from the noisy stream.
- Consider REPR-ALIGN to stretch the token budget.
- Log it in omni_lab.

**Ship gate.**
- End-to-end phone gain ≥ 1.3× at 4k *and* 32k.
- Tool-call replay sweep and scorecard behaviour held.
- Judged on behaviour, never on a loss number.

**Independent media test (cheap).** Fused commit plus first-refine forward for audio and image chunks in P1/P2. A/B by listening and viewing, plus time to first audio.

## Sources

- TiDAR: arXiv abstract https://arxiv.org/abs/2511.08923; full HTML v1 https://arxiv.org/html/2511.08923v1; MLSys 2026 oral https://mlsys.org/virtual/2026/oral/3751; slides https://mlsys.org/media/mlsys-2026/Slides/3751_1jPpJd3.pdf; proceedings https://proceedings.mlsys.org/paper_files/paper/2026/hash/1367d856028f65a9555b0274db09e608-Abstract-Conference.html; project page https://tidarlm.github.io/; HF paper page https://huggingface.co/papers/2511.08923
- Nemotron-Labs-Diffusion: arXiv https://arxiv.org/abs/2607.05722 (HTML https://arxiv.org/html/2607.05722v1, body and references read); NVIDIA Research https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive; HF 8B card https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B; HF 3B card https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B; VLM card https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B; ACL Anthology https://aclanthology.org/2026.emnlp-main.2416/; NVIDIA HF blog https://huggingface.co/blog/nvidia/nemotron-labs-diffusion
- dQwen3.5: https://arxiv.org/abs/2609.20751, HTML https://arxiv.org/html/2609.20751v1, weights https://huggingface.co/UT-IFML/dQwen3.5-2B-Base
- llada.cpp: https://arxiv.org/abs/2606.13740, HTML https://arxiv.org/html/2606.13740v1
- Set Block Decoding (NLD ref [14]): https://arxiv.org/abs/2509.04185
- Fast-dLLM v2: https://arxiv.org/abs/2509.26328, https://nvlabs.github.io/Fast-dLLM/v2
- Efficient-DLM: https://arxiv.org/abs/2512.14067
- REPR-ALIGN: https://arxiv.org/abs/2605.06885
- Uno / Ψ-Spec: https://arxiv.org/abs/2609.04010
- LaDiR: https://arxiv.org/abs/2510.04573, https://machinelearning.apple.com/research/ladir
- DCoLT: https://arxiv.org/html/2505.10446v1
- Component-Aware Self-Speculative Decoding in Hybrid LMs: https://arxiv.org/abs/2605.01106
- Mobile speculative decoding with NPU coordination: https://arxiv.org/abs/2510.15312
- Community llama.cpp port: https://huggingface.co/spiritbuun/Nemotron-Labs-Diffusion-14B-Q8_0-GGUF
- Our records:
  - `ARCHITECTURE_V2_2026-10-21.md`: §3; L158 `native_bidirectional_refinement`; L247 spec-decode net loss; L294/L321 GDN state 18 MiB; §7 L329-348 decode projections.
  - `ARCH_CANDIDATES_2026-10-22.md` §4: synthesis, mask-row ruling, P-SPEC kill rule.
  - `run0-20261022/survey_hw.md`: P40 throughput.
  - Memory `sprite-claude-pickup-2026-10-19.md`: Bonsai 8B 285-500 / 15-21 tok/s prefill/decode.
  - Memory `omni-v2-direction-2026-10-21.md`: Cyril's rulings.
