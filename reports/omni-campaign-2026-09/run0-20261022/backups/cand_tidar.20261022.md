# Candidate review: TiDAR and NVIDIA diffusion "thinking" for Chimera Omni v2

Date: 2026-10-22. This is a read-only review. Cyril asked us to "[Cyril quote withheld]". All sources were fetched on 2026-10-22 and are listed at the end.

Tags:
- **[P]** primary source: paper, slides, model card or project page.
- **[S]** secondary source, or a primary source I read only at abstract level.
- **[R]** our own records.
- **[I]** my inference. Nothing in this file was measured by us.

## Verdict

**What he means: identified, with high confidence.** Cyril's paper is **TiDAR: Think in Diffusion, Talk in Autoregression**.
- arXiv 2511.08923 v1, 12 Nov 2025.
- Authors: Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang and Pavlo Molchanov (NVIDIA tech report).
- **MLSys 2026 oral** (19 May 2026), in proceedings volume 8 [P].
- Every part of his description matches: NVIDIA, "diffusion thinking", "regular attention-based output", and one forward pass.

**One correction to how he remembers it.** In TiDAR, "thinking" is **not chain-of-thought**. It is parallel drafting of the next block of tokens. The paper calls it parallel token computation ("thinking") via diffusion [P]. The work that really does diffuse the reasoning itself is listed in §1. None of it is NVIDIA's.

**The NVIDIA successor matters more than TiDAR itself.** It is **Nemotron-Labs-Diffusion** (NLD): arXiv 2607.05722, July 2026, from the same group, with TiDAR's first author as a co-author [P].
- **It ships weights:** 3B, 8B and 14B, as base, instruct and VLM variants.
- **It cites TiDAR as its "quadratic self-speculation" mode.** It then reports that the simpler **linear self-speculation** (two passes) "is currently the most efficient mode" [P].
- **Anything we adopt should follow NLD, not TiDAR's one-pass quadratic layout.**

**Recommendation:**
1. **Run 0: change nothing.** There is one cheap exception. While the modality and output-decision tokens are being added, reserve a `<|mask|>` special-token row in the vocab plan. That leaves the door open at zero cost.
2. **Do not adopt TiDAR's one-pass quadratic draft+verify for the phone.**
   - At its published draft length of 16, each step sends about 16 × 17 = 272 query tokens through the model.
   - On our hybrid it would also need 17 forked Gated DeltaNet (GDN) recurrent states per step.
   - Both are wrong for a memory- and overhead-bound phone NPU running 18 GDN layers out of 24 (§4) [I].
3. **Park one gated later arm, A-DIFF**, for the fold pass or later. It is NLD-style joint AR + diffusion text training (α = 0.3, two-stage: AR distillation first, then joint) with *linear* self-speculation decoding. It adds **zero parameters**, so it is compatible with the Qwen plant. It ships only if both of these hold:
   - a phone microbenchmark shows a real end-to-end tok/s gain;
   - behaviour holds on the tool-call replay sweep and the scorecard.

   It is the one route in this literature that beats MTP/EAGLE-3 on memory-bound hardware (DGX Spark 3.14× over AR at batch 1) [P]. That is why it is worth one gated arm despite Cyril's "spec and MTP don't work on the phone".
4. **For media, one small idea is worth testing as inference scheduling only:** fuse chunk *n*'s causal commit pass with chunk *n+1*'s first flow-refinement step in a single forward. This is TiDAR's "verify the old, draft the new in one pass", applied to our x0-flow chunks.
   - It needs no training change: the noisy span already attends a clean prefix.
   - It saves one forward per chunk. For audio at 4 flow steps per chunk, that is 5 forwards down to 4, about 20 % [I].
   - Add it to the A13 / P1 phone experiments.
5. **TiDAR does not make images or video faster in any other way.** Our media path already *is* "diffusion in chunks with refinement passes". Media cost is (flow steps) × (span tokens per forward). TiDAR changes neither [I].

## 1. Identification

| Search / fetch | Result |
|---|---|
| WebSearch (extended): "NVIDIA diffusion chain of thought reasoning language model 2025 2026 hybrid autoregressive diffusion" | TiDAR (arXiv 2511.08923) and Nemotron-Labs-Diffusion (NVIDIA Research, May 2026). The non-NVIDIA hits were DoT (2024, out of window) and a planner-executor study (arXiv 2510.15244) |
| WebSearch (extended): "TiDAR MLSys 2026" | MLSys 2026 oral 3751, proceedings volume 8, slides PDF |
| WebFetch: arXiv abs + HTML v1, MLSys slides, tidarlm.github.io | Full method and results (§2) |
| WebSearch (extended) on latent / diffusion CoT | LaDiR (arXiv 2510.04573, ICLR 2026) and DCoLT (arXiv 2505.10446) |
| WebSearch (extended) on hybrid linear-attention diffusion / self-speculation | DeltaFlow (2608.01240), Component-Aware Self-Speculative Decoding (2605.01106), Retrofitting Linear Attention into DLMs (2608.06628) |

**Candidates and why each is or is not "it":**

| Candidate | NVIDIA? | Diffusion "thinks", AR "talks"? | Verdict |
|---|---|---|---|
| **TiDAR** (2511.08923, MLSys 2026) | Yes | Yes, literally the title; one forward pass | **This is it** |
| **Nemotron-Labs-Diffusion** (2607.05722, Jul 2026) | Yes, same group | Diffusion drafts, AR verifies (self-speculation) | The productised follow-up; reviewed in §3 |
| LaDiR (Apple + UCSD, 2510.04573, ICLR 2026) | No | Reasoning is diffused as VAE-latent "thought blocks", then the answer is written AR | Closest to "diffusion chain of thought", but not NVIDIA. Needs a VAE encoder. The reported gain is small (43.5 vs 42.0 pass@1 averaged over seven math benchmarks on LLaMA 3.1 8B) [S] |
| DCoLT (2505.10446) | No | The whole model is diffusion; each reverse step is a "lateral thought" trained by outcome RL | Not a hybrid AR output; not it |
| Planner-executor (2510.15244, EPFL / MBZUAI / ETH / MSRA) | No | A diffusion planner plus an AR executor, as two models | Two models; not it |

## 2. TiDAR: what it is [P unless tagged]

### 2.1 Mechanism
- **Base:** a standard AR transformer. All bases are pure attention (Qwen2.5-1.5B, Qwen3-4B, Qwen3-8B). **No new parameters.** Only the attention mask and a mask token change.
- **Training layout:**
  - The sequence is **doubled**: a clean copy plus an appended copy made entirely of `[mask]` tokens ("full mask").
  - The clean part uses causal attention and gets the AR next-token loss.
  - The mask part attends the clean prefix and is **bidirectional within its block**. It gets the diffusion loss.
- **Loss:** L = 1/(1+α) · (α·L_AR + L_Diff), with α = 1 "for most cases".
  - Full masking beat random masking: average 48.42 vs 45.45 at 4 drafts, and 48.94 vs 45.68 at 8 drafts (Table 5).
  - The slides also claim easy loss balancing, because both terms have the same number of loss tokens.
- **Inference (the "one pass"):** each forward does three things at once.
  1. It **verifies** the previous step's draft causally, by rejection sampling against the AR distribution.
  2. It **pre-drafts** the next block by one-step diffusion, *for every possible acceptance outcome*.
  3. It writes exact KV for the accepted tokens and evicts the rest.
- **The mask is pre-built:** q_len = block × (1 + block), sliced per step with Flex Attention. So the per-step query count grows quadratically with draft length. NLD calls this "quadratic".
- **Draft length:** the 1.5B and 8B models use 16 in practice (project page).
- **Mixing knob:** final tokens use argmax of β·logits_AR + (1−β)·logits_diff, where β picks whether to trust AR or diffusion.

### 2.2 Training recipe
- **Continued pretraining from an AR checkpoint:**
  - 1.5B: 50B tokens, block sizes 4, 8 and 16, global batch 2M tokens.
  - 8B: 150B tokens, block 16.
- **Sequence length 4,096.** Distributed Adam, cosine schedule with LR from 1e-5 to 3e-6, 1 % warm-up, BF16.
- **Hardware:** H100s, with Megatron-LM and TorchTitan.
- **No TiDAR-4B result is reported.** Qwen3-4B appears only as a baseline.

### 2.3 Results
**Generative (Table 2):**

| Model | HumanEval | MBPP | GSM8K | Minerva Math | 6-task average |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 35.98 | 43.60 | 54.74 | 26.40 | 41.64 |
| **TiDAR 1.5B** | 43.29 | 41.40 | 53.90 | 25.48 | **44.03** |
| Qwen3-8B | 64.63 | 69.40 | 81.80 | 52.94 | 68.09 |
| TiDAR 8B (trust diffusion) | 57.93 | 65.40 | 80.44 | 51.64 | 65.31 |
| Dream 7B | 54.88 | 56.80 | 77.18 | 39.60 | 58.74 |
| LLaDA 8B | 32.32 | 40.80 | 70.96 | 27.30 | 41.78 |

**Likelihood average (Table 3):**
- TiDAR 1.5B 64.43 vs Qwen2.5-1.5B 65.16.
- TiDAR 8B 75.40 vs Qwen3-8B 74.25.

**Tokens per forward (NFE):** 7.45 average at 1.5B; 8.23-8.25 at 8B. Table 4 shows the draft-length trade-off:
- 4 drafts: 3.47 T/NFE, GSM8K 55.87.
- 8 drafts: 5.49 T/NFE, GSM8K 54.74.
- 16 drafts: 6.97 T/NFE, GSM8K 53.90.

**Throughput:**
- **4.71× tokens/s** (1.5B vs Qwen2.5-1.5B) and **5.91×** (8B vs Qwen3-8B).
- Measured on **one H100, batch size 1, native PyTorch + Flex Attention, exact KV for all methods.**
- The paper says it beats EAGLE-3 (Qwen3-8B, public weights) on throughput, but the numbers are only in Figure 4.
- The project page adds an SGLang figure: up to 5.8× AR and over 2.5× EAGLE-3 on one H100.

**What it rests on: "free token slots".** On an H100 at batch 1, decoding K tokens costs about the same as decoding 1 when K is small. This was shown on Qwen3-32B with FlashAttention-2. The paper gives no numeric K threshold.

### 2.4 Caveats the paper itself states, or its tables show
- **The comparisons are not equal-token.** The AR baselines did not receive the same 50B or 150B continued-pretraining tokens. The 1.5B generative gain (+2.4) may partly be extra training [I].
- **"Lossless" is not strictly true.** 1.5B likelihood is −0.73 below its base. 8B generative is −2.78 below Qwen3-8B.
- **Unlike speculative decoding, TiDAR does not guarantee identical output to the base model.**
- **Batch 1 only.** Larger batches are claimed to work "zero-shot" by shortening the draft, but no results are given.
- **Long context:** training doubles the sequence length, and long-context extension is left as future work. The slides add that at long context the model "may be less memory bound and certain advantages of TiDAR may diminish".
- **Code and weights: none.** The project page says "SGLang inference code will be released soon".
- **Edge or mobile: not discussed.**

## 3. Nemotron-Labs-Diffusion: the evidence that actually informs us [P]

### 3.1 Recipe
- **Tri-mode:** one model with three decoding modes.
  - AR.
  - Block diffusion, with an optional small trained sampler.
  - Self-speculation, where diffusion drafts and AR verifies with a shared KV cache.
- **Training mask:** a dual stream (noisy + clean). Block diffusion runs inside the noisy stream, and the clean stream is *strictly causal*. That lets the AR and diffusion losses train in one forward/backward without label leakage.
- **Loss and averaging:** α = 0.3, and losses are averaged globally over tokens to cut the variance from masking.
- **Two stages:**
  1. AR only.
  2. Joint AR + diffusion.
- **Base models:** initialised from **Ministral3** (pure attention; **not** a hybrid with linear layers).
  - 1T tokens in stage 1, 300B tokens in stage  2.
  - Sequence 4,096, 256 H100s.
  - SFT: 45B tokens at 16k.
- **Released:** training and inference via Megatron Bridge, plus HF weights under the NVIDIA open model licence.

### 3.2 The diffusion loss does not cost AR quality
Table 3 compares 25B tokens of training from Ministral3-8B, with and without the diffusion loss:

| Model | AR only | + diffusion loss |
|---|---|---|
| Base, average | 72.50 | 72.64 |
| Instruct, average | 63.18 | 63.61 |

IFEval fell from 71.66 to 68.65. The authors note instruction-following compliance is "slightly affected".

### 3.3 Results at our size class
Table 7, 3B instruct:

| Model | Mode | Average accuracy | TPF (tokens per forward) |
|---|---|---|---|
| Qwen3-4B | AR | 53.23 | 1.00 |
| NLD-3B | AR | 55.50 | 1.00 |
| NLD-3B | Diffusion | 52.90 | 1.91 |
| NLD-3B | Linear SS | 55.00 | 4.36 |
| NLD-3B | Quadratic SS (TiDAR-style) | 55.80 | 5.42 |

Adding a LoRA drafter raised TPF by 14.4 % at 3B and 32.5 % at 8B. Larger models get higher TPF.

### 3.4 Self-speculation vs MTP
Table 10 reports average acceptance length:
- NLD: 5.46 native, 6.82 with the LoRA drafter.
- EAGLE-3: 2.75. MTP: 4.24.

At batch 1, linear self-speculation (SS) measured:
- 2.4× over EAGLE-3 on GB200, 2.3× on RTX Pro 6000 and 1.8× on DGX Spark.
- On RTX Pro 6000: 277 tok/s at FP8 and 525 tok/s at INT4 (3.46× and 2.35× over AR).
- On DGX Spark: 77.5 tok/s at FP8 and 112.5 tok/s at INT4 (3.14× and 2.69× over AR).

DGX Spark is the closest analogue to a phone in that paper: a memory-bandwidth-bound LPDDR device.

### 3.5 Linear vs quadratic self-speculation
Quadratic SS (TiDAR) has the higher TPF. The authors still conclude that linear SS "achieves the best efficiency in the current infrastructure". Quadratic SS is "more promising at batch size 1 with improved infrastructure support".

### 3.6 Vision
The NLD VLM adds a **vision encoder + projector** taken from Ministral3-8B-Instruct. That is a tower, so it is not usable under Cyril's no-tower rule. Only the text recipe is relevant to us.

### 3.7 Independent llama.cpp port
A community port exists: the `buun-llama-cpp` fork with `--diffusion-self-spec`. It reports 42-50 tok/s at k = 4 for the 14B Q8_0 on an RTX 3090 [S], but gives no AR baseline. My rough AR ceiling for that setup is 936 GB/s ÷ about 15 GB ≈ 60 tok/s [I]. So there is **no evidence yet that the gain survives a llama.cpp-class runtime.** That matters, because Sprite runs on llama.cpp.

## 4. What it would change for Chimera Omni v2

| Our constraint | Effect of TiDAR / NLD |
|---|---|
| **Qwen3.5-2B plant, 24 layers = 18 GDN + 6 gated GQA [R ARCH §3]** | **Zero new parameters, so the plant is fully compatible.** But every published TiDAR/NLD result is on **pure-attention** models. In our hybrid, a draft block's in-block bidirectional mixing would exist only in the 6 attention layers. The 18 GDN layers are causal recurrences, so a run of identical `[mask]` inputs gets little in-block context there [I]. Bidirectional GDN is possible (DeltaFlow 2608.01240 alternates or parallelises forward and backward scans [S]), and it is the same mechanism as our A8 reverse span scan. **Draft quality on a 3:1 GDN hybrid is unpublished.** |
| **Cheap internal drafting on Qwen3.5** | Measured as nearly useless: using Qwen3.5's own linear-attention subgraph as the drafter gave acceptance α = 0.038, against 0.68 on Falcon-H1 (2605.01106) [S]. Only a *trained* draft pathway, like NLD's diffusion mode, is plausible on our trunk. |
| **TiDAR quadratic one-pass on the phone** | **Reject.** At draft 16 that is about 272 query tokens per step. In the GDN layers it also needs 17 state forks (one per possible acceptance point, 18 MiB f32 each at 2B [R ARCH §6]), which is about 306 MiB of state traffic per step [I]. Its "free token slots" premise is an H100 fact, and it is not established on Hexagon. |
| **Linear self-speculation on the phone** | **Plausible, unmeasured.** On the NPU, prefill throughput is about 19-24× decode (Bonsai 8B: 285-500 vs 15-21 tok/s [R sprite-claude-pickup]). Our decode model also carries a fixed 24 × c ≈ 4.3-7.9 ms per forward [R ARCH §7]. Both suggest a k-token forward is much cheaper than k single-token forwards [I]. Against that, our record says speculative decoding and MTP "measured as a net loss on 1-bit kernels" [R ARCH L247], and Cyril says they "[Cyril quote withheld]" [R]. The likely causes are low acceptance under 1-2-bit quantisation (Bonsai fragility), GDN rollback cost, and a second model in memory [I]. NLD removes the second model and roughly doubles acceptance over MTP [P], but GDN rollback remains. Each step needs a state checkpoint: about 18 MiB at 2B, roughly 1 ms to save and restore at 40 GB/s [I]. Kernel K3 (GDN state save/restore) is already on our list [R]. |
| **50+ tok/s phone goal** | A-DIFF is the only lever in this literature with batch-1, bandwidth-bound evidence (DGX Spark 2.7-3.1×). If even 1.5× survived on Hexagon, the 2B Q2 projection would gain directly. **This is the one reason not to drop the idea.** |
| **256k context, 50k phone minimum** | Training doubles sequence length (clean + masked). NLD's SFT trick doubles only the response segment ("L_text + L instead of 2L"), so long prompts are not doubled [P]. Drafting cost does not grow with context. As context grows, the attention term dominates decode and the relative gain shrinks; TiDAR's slides say the same [P]. |
| **Media x0-flow chunks; Cyril's "refinement passes"** | Already diffusion-style. TiDAR adds only the fused commit-plus-first-refine pass described in Verdict item 4 [I]. Media speed still comes from fewer flow steps (S6 16→4 step distillation [R]) and the A8 / global-share decisions. |
| **Output-decision token, modality tokens** | No conflict. Reserving a `<|mask|>` token alongside them costs nothing. |
| **"Top-n matching, never top-1" grading** | No conflict. TiDAR and NLD verify against the AR distribution. Joint AR and diffusion losses are both cross-entropy, so the top-k KL can be applied to the AR stream unchanged [I]. |
| **No towers, no sparse attention** | No conflict for the text recipe. The NLD VLM's encoder is not adopted. |

### 4.1 Cost on 4 P40s
- **Throughput budget:** survey_hw measures about 380-535 tok/s, or 33-46M tokens/day [R].
- **Dual-stream training roughly halves useful tokens per second:** about 17-23M per day [I].
- **Published adaptation budgets are far beyond us:**
  - TiDAR 1.5B used 50B tokens: about 1,100 days at our full rate even without the doubling.
  - NLD's *ablation* used 25B tokens: about 540-760 days.
- **No paper shows how few tokens are enough.** A-DIFF at our budget (say 1-2B tokens in a month) is an unproven extrapolation [I].
- **Engineering cost:**
  - The doubled-sequence mask in our trainer: the media refinement mask already does the same clean-prefix-plus-bidirectional-span job, so it extends naturally [R ARCH: `native_bidirectional_refinement`].
  - Draft-then-verify plus GDN checkpointing in our llama.cpp/Hexagon runtime: Claude-owned perf work [R].

### 4.2 Risk
- **Behavioural drift.** NLD saw IFEval fall by 3 points. Our tool-call behaviour is fragile at low bit widths [R Bonsai].
- **Phone gain may be zero or negative**, as it was with our MTP and speculative decoding.
- **Hybrid-GDN draft quality is unknown.**
- **Doubling sequence length** competes with run 0's distillation budget.
- **None of this belongs in run 0.**

## 5. Evidence quality

| Source | Quality | Notes |
|---|---|---|
| TiDAR (2511.08923) | **Moderate.** Peer-reviewed (MLSys 2026 oral), NVIDIA | One H100, batch 1, native PyTorch. AR baselines not equal-token. EAGLE-3 numbers only in a figure. No code or weights, so it cannot be reproduced. No edge data. |
| Nemotron-Labs-Diffusion (2607.05722) | **Moderate to good.** Tech report, not yet peer-reviewed; vendor-authored | Weights and training pipeline released. Several GPUs, including DGX Spark, plus an ablation with and without the diffusion loss at equal tokens (25B). Everything is pure attention. |
| Community llama.cpp port | **Weak [S]** | Speed numbers have no AR baseline. |
| LaDiR, DCoLT, DeltaFlow, 2605.01106 | **[S] abstract level only** | Cited for orientation; not read in full. |
| Our phone numbers | **[R], with gaps** | Bonsai NPU prefill/decode is measured. Every 2B co2 phone number is a projection; ARCH §7 states that none was measured. |

## 6. Proposed gated arm (not for run 0)

**A-DIFF: NLD-style joint AR + diffusion text, linear self-speculation.**

**Phone gate first (extends A13):**
- On the 2B Q2 trunk, measure k-token forward latency on one HTP session at k = 1, 2, 4, 8 and 16.
- Measure the cost of a GDN state save and restore.
- If cost(k = 8) ≥ about 2.5 × cost(k = 1), stop. The arm cannot pay [I].

**Server pilot:**
- Run 0.8B → 2B, two-stage (after the AR distillation), α = 0.3, global loss averaging, response-only doubling, and a `<|mask|>` token.
- Measure acceptance length with greedy and with sampled verification.

**Ship gate:**
- End-to-end phone tok/s gain ≥ 1.3× at 4k and 32k.
- Tool-call replay sweep and scorecard behaviour held.
- Judged on behaviour, never on a loss number.

**Media scheduling test (independent, cheap):** fused commit plus first-refine forward for audio and image chunks in P1/P2. Check A/B by listening and viewing, and measure time to first audio.

## Sources

- TiDAR, arXiv abstract: https://arxiv.org/abs/2511.08923
- TiDAR, full HTML v1: https://arxiv.org/html/2511.08923v1
- TiDAR, MLSys 2026 oral page: https://mlsys.org/virtual/2026/oral/3751
- TiDAR, MLSys slides (PDF): https://mlsys.org/media/mlsys-2026/Slides/3751_1jPpJd3.pdf
- TiDAR, MLSys proceedings: https://proceedings.mlsys.org/paper_files/paper/2026/hash/1367d856028f65a9555b0274db09e608-Abstract-Conference.html
- TiDAR, project page: https://tidarlm.github.io/
- Nemotron-Labs-Diffusion, arXiv PDF (the report was read in full from this file): https://arxiv.org/pdf/2607.05722
- Nemotron-Labs-Diffusion, NVIDIA Research page: https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive
- Nemotron-Labs-Diffusion-3B model card: https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B
- Nemotron-Labs-Diffusion, NVIDIA HF blog: https://huggingface.co/blog/nvidia/nemotron-labs-diffusion
- Community llama.cpp port (14B Q8_0 GGUF card): https://huggingface.co/spiritbuun/Nemotron-Labs-Diffusion-14B-Q8_0-GGUF
- LaDiR: https://huggingface.co/papers/2510.04573 and https://machinelearning.apple.com/research/ladir
- DCoLT: https://arxiv.org/html/2505.10446v1
- Planner-executor study: https://arxiv.org/pdf/2510.15244v2
- DeltaFlow (bidirectional GDN): https://arxiv.org/abs/2608.01240
- Component-Aware Self-Speculative Decoding in Hybrid LMs: https://arxiv.org/abs/2605.01106
- Retrofitting Linear Attention into Diffusion LMs: https://arxiv.org/html/2608.06628
- Our records:
  - `ARCHITECTURE_V2_2026-10-21.md` §§3, 5, 6, 7, 11 (L166, L247, L294, L329-348, L367).
  - `run0-20261022/survey_hw.md` (P40 throughput).
  - Memory `sprite-claude-pickup-2026-10-19.md` (NPU prefill/decode).
  - Memory `omni-v2-direction-2026-10-21.md` (Cyril's rulings 2, 5, 11, 12).
