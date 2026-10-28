# Architecture candidates: verdicts for run 0 and the fold pass

**Revision 2, 2026-10-28.** Revision 1 (2026-10-22) is kept at `run0-20261022/backups/ARCH_CANDIDATES_2026-10-22.20261022.md` (sha256 e9a999d0...58fb1, identical to the file it replaced). This revision re-judges the four candidates against the re-verified reviews of 2026-10-27/28. **No verdict for run 0 or the fold pass changed.** What changed: MemT's cost numbers got worse, DeepInfer's KV ideas are grouped into one later arm, and the TiDAR / phone-speed analysis now uses the first real Hexagon k-token cost data (llada.cpp) and a zero-training acceptance test (dQwen3.5). Nothing was trained, launched or changed to produce this file.

**Cyril's test:** "[Cyril quote withheld]".

**Bar for each verdict:**
- **INCLUDE IN RUN 0:** fits the Qwen3.5-2B plant, is low-risk on the 4 P40s, and clearly helps.
- **INCLUDE IN THE FOLDING PASS:** helps the fold itself without confounding the run-0-vs-fold comparison.
- **WORTH A SMALL EXPERIMENT LATER:** plausible gain toward a stated goal, with a cheap gate that can kill it.
- **NO:** not meaningful, conflicts with a rule, or unverifiable.

**Inputs:** `run0-20261022/cand_memt.md` (rev 10-28), `cand_sage.md` (rev 10-27), `cand_deepinfer.md` (rev 10-27), `cand_tidar.md` (rev 2, 10-28); `survey_hw.md`, `survey_lit.md`, `RUN0_BUILD_PLAN.md`; `ARCHITECTURE_V2_2026-10-21.md` (§5 generation, §7 phone decode projections); memories `omni-v2-direction-2026-10-21`, `trust-the-model-memory-is-tools`, `omni-full-distill-k3-primary`, `sprite-claude-pickup-2026-10-19`.

**Tags:** **[P]** primary source cited by a review; **[R]** our records; **[I]** my inference. **No phone number in this file was measured on co2-2b.**

## Bottom line

| Candidate | Run 0 | Fold pass | Later | Verdict |
|---|---|---|---|---|
| **MemT-1.0** (MemTensor recurrent memory branch on a frozen LLM) | No | No | No | **NO** |
| **Sage-A1.5** (InternScience agent post-train of Qwen3.6-35B-A3B) | No | No | No: not as architecture, teacher or data | **NO** |
| **DeepInfer** (DeepSeek inference engine + V4.1-Flash ideas) | No | No | **KV diet arm:** FP4 KV trained in (inside S6 QAT) + cross-layer KV sharing between the 6 attention layers | **NO for the engine. KV diet: WORTH A SMALL EXPERIMENT LATER.** Encoder/decoder split: NO for now |
| **TiDAR** (NVIDIA, diffusion draft + AR verify in one pass) | No | No | **A-DIFF arm** (NLD-style joint AR+diffusion, *linear* self-speculation) behind two zero-training gates; **fused media commit+refine** schedule | **WORTH A SMALL EXPERIMENT LATER.** TiDAR's one-pass quadratic layout on the phone: **NO**. Diffusion chain-of-thought: **NO** |

**Run 0 does not change.** No candidate meets the run-0 bar. Run 0 stays as Cyril ordered: plant Qwen3.5-2B, ablate, distill across all modalities on our architecture, all 4 P40s.

**The fold pass does not change.** "[Cyril quote withheld]" makes the fold pass the comparison against run 0. A new loss, module or KV layout there would confound the one question it exists to answer.

**Housekeeping only (not inclusions):**
- Keep Sage-A1.5 outputs and anything derived from them out of every corpus and teacher cache.
- A `<|mask|>` vocab row is not needed for run 0. Teacher KL covers only the first 248,320 rows, so appending one later costs the same as now. If the special-token block is being written anyway, one unused row is harmless.

## 1. MemT-1.0: NO

**What it is [P]:** a fixed-size delta-rule memory branch added beside the full-attention layers of a *frozen* LLM (output = attention + memory read, one state update per token). Released only on MoE backbones (Nano: Qwen3.6-35B-A3B, 451M trainable; Flash: DeepSeek-V4-Flash, 3.61B trainable). 34B training tokens, with a top-64 distillation stage that is worth 10.69 points of their own memory average. No media evidence.

**Why NO:**
1. **It conflicts with Cyril's memory doctrine [R].** Memory is tools: record everything, never auto-load, the model retrieves when it decides. MemT is an always-on, implicit, lossy memory in recurrent state that nobody can inspect, scope or correct. Sprite already has inspectable tool-based memory.
2. **The evidence at our scale is bad.** Metis-4B, the same lab's predecessor on a Qwen3.5-4B backbone (our plant's family), barely moves with empty memory, but once unrelated history is stored IFEval falls 76.71 → 54.53 and MMLU-Pro 46.0 → 40.9 [P]. That is a behaviour-retention failure, which is exactly how Cyril judges pass/fail.
3. **It works against the speed and context goals.**
   - About 0.10-0.14B extra weights read per decoded token on co2-2b, so about 5-7 % slower phone decode, and no decode speedup inside the exact window [I, from cand_memt].
   - Its only upside is beyond the window, where it is lossy: memory average 46.97 / 50.60 against 66.27 for the same backbone with full context; LongMemEval-S 37.2, BEAM-1M 37.04 [P]. Cyril's target is 50k real context minimum and 256k capable. MemT does not substitute for real context.
4. **It cannot fit our compute or plan.** Their 34B tokens at our measured 33-46M tokens/day is about 740-1,030 days; even Metis-scale (about 406M tokens) takes all four P40s for 9-12 days, the whole window [R survey_hw, I]. It assumes a finished, frozen trunk; run 0 and the fold both retrain every parameter, so a branch trained now would have to be retrained after folding.
5. **Weak, encumbered evidence.** Single-lab preprint about a week old, unreplicated, inconsistent model sizes; code PolyForm Noncommercial and paper CC BY-NC-SA.
6. **Its useful details are already planned.** Top-k KL distillation (survey_lit, top-32 to top-64) and windowed state recompute.

**On the reviewer's "persistent session memory after run 0 and folding" suggestion:** I keep revision 1's override because of point 1. Revisit only if Cyril explicitly asks for implicit cross-session memory in the weights. If he does, the reviewer's gate is the right one: train the branch on a settled, frozen trunk starting from the planted attention weights; judge it with the memory already filled with unrelated history (where Metis-4B failed); ship only if it beats spending the same phone memory on more exact KV. Skip Mixture-of-Storage (neither released model uses it).

## 2. Sage-A1.5: NO

**What it is [P]:** an agentic post-train of the standard Qwen3.5/3.6 MoE (40 layers, width 2048, 256 experts / 8 active, about 3B active of 35B, 3:1 GDN:attention, MTP, 262k context, separate 27-layer vision tower). The paper's additions are behaviour only: learned subagent spawning, opt-in self-verification, multi-domain SFT then outcome RL plus on-policy multi-teacher distillation. "Capability composition" means agent behaviour, not weight merging or folding.

**Why NO:**
1. **No architecture for us.** Its non-MoE shape is already in co2-2b via the Qwen3.5-2B plant. MoE is what Cyril is sceptical of, MTP measured poorly on our phone [R], and the vision tower violates the no-towers rule for the end model.
2. **Excluded as a teacher or data source by three standing rules.**
   - About 3B active makes it a small model ("never small models for data gen").
   - The paper says demonstrations were collected "under ... Claude Code, Codex, and internal harnesses", with real-world trajectories from unnamed strong teachers "with Claude Code as the harness" [P]. Claude/Codex-derived outputs cannot be ruled out ("never train on Claude or Codex outputs").
   - It is not an original parent (K3 stays primary [R]).
3. **Not a plant or fold source.** Head counts, layer count, MoE vs dense FFN and the embedding tie (`tie_word_embeddings: false` vs `true` on Qwen3.5-2B) all differ, so no clean transplant. Collapsing its experts into a dense model is the MoE→dense conversion Cyril counts as cheating in folding.
4. **Weak claims.** Vendor-run benchmarks; its "matches or beats all twelve baselines" is contradicted by its own table (GAIA 95.1 vs 96.0; DeepResearch-Bench II 43.7 vs 48.6). The only speed figure is one community run on a datacenter GPU.
5. **The behaviour ideas are curriculum, not architecture.** Subagent orchestration, self-verification and on-policy distillation are possible tool-use training targets after v1, once the 2B makes single tool calls reliably (Bonsai-class prompt fragility [R]), and must be generated from allowed teachers. On-policy distillation needs live teacher queries, which is costly on P40s. Not tracked as an architecture item.

## 3. DeepInfer: NO for the engine; one KV arm later

**What it is [P]:** DeepSeek's in-house engine (Apache-2.0, 2026-10-20), a vLLM fork on FlashMLA / DeepGEMM / DeepEP / TileLang. Serves one model, DeepSeek-V4.1-Flash, on Blackwell SM100 or Ascend 950 only, without its KV-cache system, PD disaggregation or load balancing; smallest efficient deployment about 64 prefill + 64 decode GPUs. Throughput and cost claims are DeepSeek's own and unreplicated. Some base-model scores in the 10-22 review could not be re-found and are marked unverified.

**Engine: NO.** It does not build for the 3090 (SM86) or RTX 8000 (SM75), does not serve our models (V4.1-Flash needs at least 614 GB VRAM), and its gains are cluster-scale. llama.cpp already covers what Strata needs. Batch-invariant deterministic output is worth keeping as a principle for reproducible behaviour replays; nothing to build.

**Technique by technique:**

| Idea | Verdict | Reason |
|---|---|---|
| DSpark speculative decoding (about 14B drafter) | **NO** | Loses 1.5 % at 16 sequences per GPU, gains only at 60-96; no batch-1 or device results; a 14B-class drafter for a 2B target is absurd on a phone; our phone spec and MTP measured as net losses [R]. |
| CSA2 compressed sparse attention | **NO** | Sparse attention is excluded by our rules. |
| Engram n-gram hash tables (196B) | **NO** | Tokenizer-bound lookup, and the END model must be free of tokenizer-specific parts; also a huge table on the phone. |
| Sliding-window replay | **NO** | The planted shape has no SWA layers; GDN state is already fixed-size. |
| **FP4 KV cache trained in (fake-quant)** | **WORTH A SMALL EXPERIMENT LATER, inside S6 QAT** | S6 already quantizes the body. KV fake-quant in the same pass is cheap and plant-compatible. |
| **Cross-layer KV sharing between full-attention layers** | **WORTH A SMALL EXPERIMENT LATER, same arm, as an ablation** | Share K/V between attention layers, then distill. Only the 6 GQA layers hold growing KV on co2-2b, so sharing halves the dominant long-context memory cost. |
| Encoder/decoder split (top half reads one KV from the middle layer) | **NO for now.** Revisit only if a phone measurement shows media prefill or TTFT is the bottleneck | It halves prefill but does not speed short-context decode, so it does not move the 50 tok/s goal. On our hybrid, most top-half layers are GDN and would need converting to cross-attention and relearning. DeepSeek published no ablation and no conversion recipe. It would confound the fold if put there. |

**Why the KV diet arm earns a slot [I]:**
- The 32k gap to 50 tok/s is a KV-bandwidth problem. Projected 2B decode is 19-30 tok/s at 32k and 4-9 at 256k [R ARCH §7]. co2-2b KV is 12,288 B/token in bf16, about 0.4 GB at 32k and 3.2 GB at 256k [R cand_memt], against about 0.72 GB of Q2 weights per step.
- If ARCH §7 assumed bf16 KV: FP4 KV alone cuts bytes per step at 32k from about 1.12 GB to 0.82 GB (about 1.35×); with pairwise sharing, about 0.77 GB (about 1.45×). At 256k: about 2.6× and 3.5×. **First check which KV dtype ARCH §7 assumed;** if it already assumed Q8 KV, the gains roughly halve.
- It does nothing for short context, where the default Q2 build (37-48 tok/s projected) also misses 50. That gap needs Q1 or drafting (§4).
- Runtime KV quantization in llama.cpp is the baseline the trained version must beat.
- **Gate:** long-context behaviour (needle and multi-hop recall at 32k, tool-call replay at 32k, 256k spot checks), never a loss number. Run on a clone after run 0; not in the fold pass.

## 4. TiDAR, diffusion "thinking", and the 50 tok/s question

### 4.1 What it is
- **TiDAR [P]** (arXiv 2511.08923, NVIDIA, MLSys 2026 oral). Zero new parameters. Training doubles the sequence: a causal clean copy with next-token loss and an all-mask copy, bidirectional inside its block, with diffusion loss (α = 1). Each forward verifies the previous draft against the AR distribution and drafts the next block for every acceptance point, so a step carries q_len = block × (1 + block) queries (272 at draft 16).
  - 7.45 / 8.25 tokens per forward; 4.71× / 5.91× over AR on one H100 at batch 1 (1.5B / 8B).
  - Not lossless (8B 65.31 vs 68.09), AR baselines not equal-token, no code or weights, no edge, multimodal or long-context results.
- **"Thinking" in TiDAR is parallel drafting, not chain-of-thought.** Work that diffuses reasoning itself (LaDiR, DCoLT, Uno) is not NVIDIA's.
- **The follow-up that matters is Nemotron-Labs-Diffusion (NLD) [P]** (2607.05722, EMNLP 2026, weights at 3B/8B/14B). It runs TiDAR as "quadratic" self-speculation and concludes: "Linear self-speculation achieves the best efficiency in the current infrastructure."
  - 3B instruct: AR 55.5 vs linear SS 55.0 at 4.36 tokens per forward; quadratic 55.8 at 5.42.
  - Acceptance 5.46 vs MTP 4.24 vs EAGLE-3 2.75.
  - DGX Spark (bandwidth-bound LPDDR, the closest device to a phone): 3.14× over AR at FP8, but only **2.69× at INT4**. The gain shrinks as weights get smaller, and our phone runs Q1/Q2.
  - Equal-token ablation: the diffusion loss leaves the average flat but costs about 3 IFEval points.
- **New since revision 1:**
  - **dQwen3.5** (2609.20751) [P] converts Qwen3.5 0.8B-9B to masked diffusion with the GDN layers kept causal and only the attention layers (6/24 at 2B) bidirectional. The diffusion side therefore trains on our exact trunk shape. Self-speculation on a GDN hybrid is still unpublished.
  - **llada.cpp** (2606.13740) [P] is the first diffusion-LM runtime measured on Hexagon NPUs. On SM8750 a LLaDA-8B pass over 32/64/128 active tokens took 478/862/1,532 ms: sublinear, but far from free.

### 4.2 Can diffusion drafting + AR verify get the phone past 50 tok/s where MTP and spec did not?

**Short answer.**
- **TiDAR's one-pass layout: no.** Rejected on arithmetic, below.
- **NLD-style linear self-speculation: possibly, and it is the only lever in this literature with batch-1, bandwidth-bound wins.** But the first Hexagon cost data puts it on the kill line, so it gets two zero-training gates before any P40 time.

**Where co2-2b stands with no drafting [R ARCH §7, projections]:**

| 2B config | short | 4k | 32k |
|---|---|---|---|
| Q2 body + Q4 head (default) | 37-48 | 33-45 | 19-30 |
| Q1 body + Q2 head | 55-77 | 47-69 | 23-39 |

The default build needs about 1.1-1.4× at short context and about 1.7-2.6× at 32k to reach 50. The 32k gap is better attacked by the KV diet (§3). Drafting is the candidate for the short-context gap, or Q1.

**The k-token cost curve is the whole question [I, weak cross-system comparison from cand_tidar].**
- Bonsai-8B decodes at 15-21 tok/s on our phone NPU, about 48-67 ms per token [R]. llada.cpp's marginal cost is about 12-15 ms per extra token. Extrapolated, cost(k = 8) ≈ 2.3-3.2 × cost(1). That straddles the P-SPEC kill rule of 2.5×, and tile padding could make small-k forwards worse.
- **TiDAR one-pass, rejected.** Even at its smallest block of 4 (q_len 20, 3.47 tokens per forward at 1.5B), the same extrapolation gives cost(20) ≈ 4.4-6.9 × cost(1), so about 0.5-0.8×: a net loss. On our hybrid it also needs one GDN state fork per acceptance point (5 × 18 MiB at block 4, about 306 MiB at block 16). Its "free token slots" premise is an H100 fact that llada.cpp shows does not hold on Hexagon.
- **Linear self-speculation, open.** The answer depends on how many tokens one cycle commits relative to its cost.
  - Optimistic: if NLD's 4.36 tokens per forward survived Q2 and a k ≈ 8 forward costs 2.3-3.2×, gain ≈ 1.4-1.9×, putting the short-context default at roughly 52-91 tok/s.
  - Pessimistic: acceptance about 3 (plausible at 1-2-bit weights, given Bonsai fragility and NLD's INT4 shrink) with two passes per cycle gives about 0.5-0.65×, a net loss like MTP.
  - The range runs from a loss to nearly 2×. **Nothing published decides it; our own measurements do.**
- **Why MTP and spec lost, and whether self-speculation escapes it [R, I].** The record says only "net loss on 1-bit kernels". The plausible causes hit every drafting scheme: k-token forwards that are not cheap on HTP kernels tuned for GEMV, GDN state rollback (about 18 MiB, about 1 ms per save/restore), and low acceptance under 1-2-bit quantization. Self-speculation removes the second model and raises acceptance over MTP (about 1.3×) and EAGLE-3 (about 2×). **It does not remove the first two causes.** Untrained self-drafting on Qwen3.5's linear subgraph accepted only 0.038 [S 2605.01106], so only a trained draft mode is plausible.
- **One long-context upside to measure, not assume [I].** A k-query verify reads the KV once for k tokens. If our split-KV kernel accepts multiple queries, drafting amortizes the 32k KV floor, the opposite of TiDAR's H100 long-context caveat.

**Order of work (none touches run 0's P40s or GPU0/4/6):**
1. **P-SPEC, phone, no training, Claude-owned perf work under the phone lock.** Stock Qwen3.5-2B at Q2 and Q1 in our llama.cpp/Hexagon build, one HTP session: forward latency at k = 1, 2, 4, 8, 16; GDN state save/restore; k-query attention at 4k and 32k.
   - **Kill rule:** cost(k = 8) ≥ about 2.5 × cost(k = 1) at short context *and* no long-context amortization → drop A-DIFF and the MTP question for good, and put the effort into Q1 kernels and the KV diet.
   - It also finally explains why MTP lost.
2. **ACC-0, server, no training.** dQwen3.5-2B-Base drafts blocks of k = 4/8/16; stock Qwen3.5-2B verifies with greedy and with sampled rejection, on our scorecard and tool-call replay prompts (raw capture spool excluded). Run it on CPU (two 2B models, small RAM footprint, leaving Strata's headroom), or on a P40 only after run 0 frees one via lanes.py.
   - It is a two-model proxy for a self-speculating co2-2b, not the ship configuration.
   - **Gate:** acceptance-implied tokens per cycle × cost(1)/cost(k) from P-SPEC ≥ 1.3×.
3. **A-DIFF pilot, only if both gates pass.** On a clone of the post-run-0 checkpoint, after run 0 and outside the fold pass; 0.8B first, then 2B. Stage 1 is our AR distillation; stage 2 joint at α = 0.3 with global token averaging; never-masked tokens dropped from the noisy stream (my extension of NLD's VLM trick, not published for text); consider REPR-ALIGN to stretch the budget. Logged in omni_lab.
   - **Budget reality:** dual-stream training roughly halves useful P40 throughput to about 17-23M tokens/day; the smallest published adaptation budget is 25B tokens. A 1-3B-token pilot is a large extrapolation.
4. **Ship gate:** end-to-end phone gain ≥ 1.3× at 4k *and* 32k; tool-call replay sweep and scorecard behaviour held (NLD's IFEval −3 is the risk to watch); judged on behaviour, never a loss number. Top-n KL on the AR stream is unchanged by this arm.

**Diffusion chain-of-thought (LaDiR-style): NO.** Its gain is small (43.5 vs 42.0 averaged at 8B), it needs a VAE latent encoder (a tower), and it does not make output faster. **Unverified block-diffusion text output** (NLD's pure diffusion mode, 1.91 tokens per forward but 55.5 → 52.9 at 3B) is chunk output for text without a verifier and loses quality; if text is ever drafted in chunks, it should be verified (self-speculation), never trusted raw.

### 4.3 How it relates to our x0-flow joint refinement of media
- **Our media path already is Cyril's "[Cyril quote withheld]" [R ARCH §5].** A noised media span attends the prefix and the whole span, is refined N times (16 steps, distilled to 4 in S6), then a causal commit pass writes it into KV and GDN state. Speech runs in 4-8-unit chunks at 4 flow steps.
- **Joint refinement of all units of an image or clip is where media speed already comes from.** Every unit of a span is refined in the same forward, so the number of sequential forwards equals the flow steps, not the unit count. That is the decisive advantage over AR media and it stays at real image sizes (Cyril's ruling against 2x2 toys); larger images cost more tokens per forward, not more sequential forwards.
- **dQwen3.5 is indirect support for our span mask [I].** It shows a Qwen3.5-2B-shaped trunk learns in-block bidirectional generation with the 18 GDN layers left causal and only the 6 attention layers opened. Our media refinement mask has the same structure (completed for GDN by the A8 reverse scan or a larger global share). This is text-only, base-model evidence, so it lowers risk without proving anything for media.
- **A-DIFF's text draft is the discrete form of the same mechanism:** a span bidirectional inside, causal to the prefix, refined, then committed. If A-DIFF ever ships, text and media share one generation mechanism through one trunk, which fits the no-tower, unified-trunk intent. That is an architectural argument for keeping the arm alive, not evidence that it is faster.
- **The one TiDAR idea that transfers directly to media: fused commit + next refine. WORTH A SMALL EXPERIMENT LATER (scheduling only).** Run chunk n's causal commit and chunk n+1's first refinement step in the same forward ("verify old, draft new"). No training change is needed because the noisy span already attends a clean prefix. For audio at 4 steps it saves about 1 forward in 5 (about 20 %) [I]; for a jointly refined image it removes the separate commit pass when generation continues. Test in the P1/P2 phone experiments: A/B by listening and viewing, plus time to first audio.
- **Other media speed levers are already ours, not TiDAR's:** fewer flow steps (S6 16 → 4); same-weights coarse draft then refine at full resolution (the H3 SOL structure, which lifted video-out 0.400 → 0.625 in the round-4 diagnostic [R ARCH §14]). Cyril already expects images and video to be slow.

## 5. For the run 0 operator
- **Change nothing** in the run 0 config, data mix or schedule.
- Keep Sage-A1.5 and Sage-derived data out of every corpus and teacher cache.
- **Zero-training work that can run during run 0 without touching its P40s:** P-SPEC on the phone, and ACC-0 on CPU. Together they decide whether any drafting scheme can help the phone reach 50 tok/s.
- **After run 0, before or beside the fold pass but never inside it:** the KV diet arm (FP4 KV + KV sharing) folded into S6 QAT, and the fused media commit+refine schedule. A-DIFF only if P-SPEC and ACC-0 pass.

## Evidence quality

| Candidate | Identification | Claims |
|---|---|---|
| MemT-1.0 | Solid (arXiv 2610.26825, repo, HF) | Weak to moderate: week-old single-lab preprint, unreplicated, inconsistent sizes, unnamed GPUs; Metis-4B is the only evidence at our scale and it is negative under populated memory. |
| Sage-A1.5 | Solid (official config) | Medium to weak: vendor-run benchmarks contradicted by its own table; one community speed anecdote. |
| DeepInfer | Solid (README, Q&A, independent press) | Medium for V4.1 ideas (DeepSeek's own, few ablations); weak for anything we would run. FP4 KV "marginal degradation" has no published deltas. |
| TiDAR | Solid (MLSys 2026) | Moderate: peer-reviewed, not reproducible (no code or weights). NLD moderate to good (weights, equal-token ablation, DGX Spark data). dQwen3.5 and llada.cpp moderate preprints. All self-speculation results are on pure-attention models. |

Every co2-2b phone number here is a projection or an inference. The llada.cpp-to-our-phone comparison crosses model, runtime and quant, so it only sets a prior; P-SPEC replaces it.
