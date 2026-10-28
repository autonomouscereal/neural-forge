# Architecture candidates: verdicts for run 0 and the fold pass

**Date:** 2026-10-22. **Scope:** a judgment on the four candidates Cyril asked us to review (MemT-1.0, Sage-A1.5, DeepInfer, TiDAR), applying his test: "[Cyril quote withheld]". Nothing was trained, launched or changed to produce this.

**Inputs:** the four reviews in `run0-20261022/` (`cand_memt.md`, `cand_sage.md`, `cand_deepinfer.md`, `cand_tidar.md`), plus `survey_hw.md`, `survey_lit.md`, `ARCHITECTURE_V2_2026-10-21.md` (§5 generation, §7 phone decode projections), `TOKENIZER_FREE_STUDY_2026-10-22.md`, and Cyril's rulings in memory `omni-v2-direction-2026-10-21`.

**Tags:**
- **[P]** a primary source cited by a candidate review.
- **[R]** our own records.
- **[I]** my inference. None of the phone numbers in this file was measured.

## Bottom line

| Candidate | Run 0 | Fold pass | Later | Verdict |
|---|---|---|---|---|
| **MemT-1.0** (MemTensor, weight-state memory branch) | No | No | No | **NO** |
| **Sage-A1.5** (InternScience agent MoE post-train) | No | No | No (as architecture, teacher or data) | **NO** |
| **DeepInfer** (DeepSeek's inference engine) | No | No | FP4 KV trained in: a small arm inside S6 QAT. Half-depth prefill: only if a measurement calls for it | **NO for the engine.** One idea is **WORTH A SMALL EXPERIMENT LATER** |
| **TiDAR** (NVIDIA, diffusion drafts + AR verify) | No | No | Phone microbenchmark first; then the A-DIFF arm (NLD-style, *linear* self-speculation); the fused media commit+refine schedule | **WORTH A SMALL EXPERIMENT LATER.** TiDAR's own one-pass quadratic layout: **NO** |

**Run 0 does not change.** No candidate meets the bar of fitting the Qwen plant, being low-risk on the P40s, and clearly helping. Run 0 stays as ordered: plant Qwen3.5-2B, ablate, then distill across all modalities.

**The fold pass does not change either.** Cyril framed it as "[Cyril quote withheld]", which makes it the comparison against run 0. Adding a new loss or module there would confound the one thing it exists to test.

## 1. MemT-1.0: NO

**What it is [P]:** a delta-rule memory branch beside attention on the global layers. Only the branch trains; the backbone stays frozen. It was released only on 3B-active and 17B-active MoE models, with no media.

**Why not:**
1. **It conflicts with the memory doctrine.** Cyril's ruling is that memory is tools: record everything, never auto-load, and let the model search when it decides [R `trust-the-model-memory-is-tools`]. MemT is the opposite. It is an always-on, implicit and lossy memory written into weight-state that nobody can inspect, correct or scope. Sprite already has inspectable phone memory through tools [R].
2. **It works against the goals.**
   - It slightly slows decode, against the 50 tok/s target.
   - It only breaks even with real context at 32K, and it falls far behind on long benchmarks (LongMemEval-S 37.2 vs 83.2; BEAM-1M 37.0 vs 72.6) [P]. Cyril's requirement is 50k real context minimum and 256k capable, and MemT does not substitute for that.
   - There is no evidence on media, where a fixed state would be overwritten quickly.
3. **It conflicts with run 0.** It assumes a finished, frozen trunk, and both run 0 and folding retrain everything. Copying their 34B-token recipe would take about 650 days on four P40s [R survey_hw: 33-46M tok/day].
4. **The evidence is weak.**
   - It is a two-day-old, single-lab preprint with internal contradictions on model size and on the memory parameter count.
   - The licences conflict: the code is noncommercial (PolyForm), while the model card says Apache-2.0.
5. **The two useful details are already in our plan.**
   - Top-k KL distillation is already the plan (survey_lit §2.4, top-32 to top-64 plus a sampled tail).
   - Windowed recompute of the memory state is the same idea as the activation checkpointing we already use.

So there is nothing to add. I am overriding the review's suggested "later persistent-memory item" because of point 1. Revisit it only if Cyril asks for implicit cross-session memory in the weights.

## 2. Sage-A1.5: NO

**What it is [P]:**
- A standard Qwen3.5/3.6-family MoE: 35B total, about 3B active, with a separate image encoder.
- An agent post-train covering parallel subagents and a self-verification mode.
- No architectural novelty.

**Why not:**
1. **Nothing for the architecture.** Everything that is not MoE is already in co2-2b through the Qwen3.5-2B shape. MoE is what Cyril is sceptical of, and the image encoder is a tower he wants removed.
2. **Excluded as a teacher or data source on two rules.**
   - About 3B active parameters makes it a small model under "never small models for data gen".
   - Its trajectories partly came from teachers running inside the Claude Code and Codex harnesses, which risks Claude/Codex outputs ("never train on Claude or Codex outputs").
   - Unless Cyril decides otherwise, keep it out of every corpus and teacher cache.
3. **Not a fold parent.** It is not an original parent (K3 stays primary), and collapsing MoE into dense is a fold Cyril counts as cheating.
4. **The behaviour ideas are data and curriculum, not architecture.** These are subagent splitting, self-checking and on-policy reverse-KL. They belong to post-v1 tool-use training, and only after the 2B reliably makes single tool calls (Bonsai-class prompt fragility [R]). They are not tracked here.
5. **The evidence is vendor-run only.** The paper's own table contradicts its "matches or beats everywhere" claim.

## 3. DeepInfer: NO for the engine; one idea worth a small later experiment

**What it is [P]:** DeepSeek's vLLM-derived engine. It serves only V4.1-Flash on Blackwell or Ascend 950, its smallest efficient deployment is 64+64 GPUs, and it supports offline inference only.

**Engine verdict, NO:**
- It does not build for the 3090 or RTX 8000, so it can't serve Strata.
- It can't run the phone runtime.
- Its speed claims are cluster-scale and unreplicated.

**Technique by technique:**

| Idea | Verdict | Reason |
|---|---|---|
| DSpark speculative decoding | **NO** | No single-user or on-device numbers. DeepSeek's own notes show a 1.5% loss at low concurrency. Its stated weakness, wasted draft work at low acceptance, is how speculative decoding failed on our phone [R ARCH §5]. |
| **FP4 KV cache trained in during post-training** | **WORTH A SMALL EXPERIMENT LATER, inside S6 QAT** | S6 already quantizes the body to Q2. Adding a KV fake-quant to the same pass costs little, and it directly cuts phone KV memory at 32k-256k. It is judged on long-context behaviour (recall, tool replay at 32k), never on loss. Plain runtime KV quantization in llama.cpp is the baseline it must beat. |
| Half-depth prefill (top layers reuse the middle layer's KV) | **NO for now. Revisit only if a measurement shows media prefill or TTFT is the phone bottleneck** | It is a real structural change. With the 3:1 GDN:attention pattern, about 9 of the top 12 layers are GDN with no KV, so they would need converting or re-projecting. DeepSeek published no comparison against an ordinary model. It does not belong in the fold pass, where it would confound the fold result. |
| Sliding-window replay | **NO** | Our planted shape has no sliding-window layers. |
| Batch-invariant deterministic output | No action | Worth keeping as a principle for reproducible behaviour replays; there is nothing to build. |

## 4. TiDAR and diffusion "thinking": the main question

### 4.1 What TiDAR does, and what Cyril remembers
- **TiDAR [P]** (NVIDIA, MLSys 2026) adds no parameters. Training doubles the sequence (a clean causal copy plus an all-mask copy that is bidirectional within its block).
  - One forward verifies the previous draft against the AR distribution and pre-drafts the next block for every acceptance point.
  - Speedups: 4.71× at 1.5B and 5.91× at 8B, on one H100 at batch 1.
  - It is not strictly lossless: 8B generative 65.31 vs 68.09, and the AR baselines did not get equal tokens.
  - No code or weights are released.
- **"Thinking" in TiDAR means parallel drafting, not reasoning.** The papers that diffuse the reasoning itself (LaDiR, DCoLT) are not NVIDIA's.
  - LaDiR's gain is small: 43.5 vs 42.0 averaged over seven math benchmarks at 8B.
  - It needs a VAE latent encoder, which is a tower-like build part.
  - **Verdict on diffusion chain-of-thought: NO.** It does not deliver Cyril's goal of building output faster, and it adds a component.
- **The follow-up that informs us is Nemotron-Labs-Diffusion (NLD) [P].**
  - It ships 3B/8B/14B weights and calls TiDAR "quadratic self-speculation".
  - It finds the two-pass **linear self-speculation** more efficient on current infrastructure.
  - At 3B: AR 55.5 / linear SS 55.0 accuracy at 4.36 tokens per forward.
  - Acceptance length 5.46 vs MTP 4.24 and EAGLE-3 2.75.
  - On DGX Spark at batch 1, a bandwidth-bound LPDDR device: 2.7-3.1× over AR.
  - Adding the diffusion loss cost nothing on average at equal tokens, but IFEval fell 3 points.

### 4.2 Could it reach 50+ tok/s on the phone where MTP and speculative decoding did not?

**Short answer:**
- **TiDAR's one-pass layout: no.**
- **NLD-style linear self-speculation: possibly.** It is the only lever in this literature with batch-1, bandwidth-bound evidence.
- **The test that decides it costs no training and should run first.** MTP failed for reasons we never isolated, and it shares the same failure points with every drafting scheme.

**Where the 2B stands without any drafting [R ARCH §7, projections, not measurements]:**

| 2B config | short | 4k | 32k |
|---|---|---|---|
| Q2 body + Q4 head (default) | 37-48 | 33-45 | 19-30 |
| Q1 body + Q2 head | 55-77 | 47-69 | 23-39 |

So the default build misses 50 even at short context, and every config misses it at 32k. Something has to give more than about 1.1-1.4× at short context and about 1.7-2.6× at 32k, or the model has to go to Q1.

**Why drafting could help on a bandwidth-bound phone [I]:**
- At Q2 a decode step reads about 0.72 GB of weights, which is about 17-19 ms of the step at 37-43 GB/s.
- A k-token forward reads those weights once. If the NPU has spare compute per weight read, k tokens cost little more than 1.
- The NPU measurements are encouraging: Bonsai-8B prefill runs at 285-500 tok/s against 15-21 tok/s decode, about 19-24× [R sprite-claude-pickup].
- Then the gain is roughly (accepted tokens per step) ÷ (cost of the step relative to one decode). With linear SS's two passes per step, an acceptance of 3 and a k = 8 forward costing 1.4× a single token, a step costs about 2.8 decode-equivalents for 3 tokens. That is only about 1.1×.
- It reaches about 2× only if the draft pass is folded into the verify pass (as TiDAR does) or acceptance approaches NLD's 4-5.

**This is why the idea is fragile.** The arithmetic only pays if k-token forwards are nearly free and acceptance stays high at 1-2-bit weights.

**Why MTP and speculative decoding failed, and whether self-speculation escapes it [R, I]:**
- The record says only "measured as a net loss on 1-bit kernels", with no breakdown.
- The plausible causes all hit any drafting scheme:
  - k-token verify forwards that are not cheap on our HTP kernels (they are tuned for GEMV);
  - GDN recurrent-state rollback (about 18 MiB of state at 2B, roughly 1 ms per save/restore);
  - low acceptance under 1-2-bit quantization (Bonsai fragility).
- Self-speculation removes the second model and roughly doubles acceptance over EAGLE-3 and about 1.3× over MTP [P]. **It does not remove the first two causes.**
- Untrained drafting on Qwen3.5's own linear layers accepted only 0.038 [P 2605.01106]. Only a *trained* draft mode is plausible on our 18-GDN / 6-attention trunk, and that is unpublished on any GDN hybrid.

**Why TiDAR's one-pass quadratic layout is rejected [I, from cand_tidar]:**
- At draft length 16 that is about 272 query tokens per step.
- On our hybrid it needs about 17 forked GDN states per step, about 306 MiB of state traffic.
- Its "free token slots" premise is an H100 fact, and it is not established on Hexagon.

**Order of work (Claude-owned performance work, phone via lock + Relay):**
1. **P-SPEC microbenchmark, no training needed.** On the phone, with the *existing* Qwen3.5-2B at Q2 and Q1 in our llama.cpp/Hexagon build, in one HTP session:
   - forward latency at k = 1, 2, 4, 8 and 16;
   - GDN state save/restore cost (kernel K3 is already on the list);
   - at context 4k and 32k.

   **Kill rule:** if cost(k = 8) ≥ about 2.5 × cost(k = 1), no drafting scheme can pay. Drop A-DIFF and the MTP question for good, and put the effort into Q1 kernels.

   This also tells us *why* MTP lost, which we don't know today.
2. **Only if P-SPEC passes: the A-DIFF server pilot** (after run 0, not in the fold pass).
   - NLD recipe: α = 0.3, the AR-distilled checkpoint first, then joint AR + diffusion. Double only the response, not the prompt. Linear self-speculation.
   - Cost: the dual stream halves useful tokens to about 17-23M per day on the P40s. NLD's own ablation used 25B tokens, so a 1-2B-token pilot is an extrapolation.
   - Measure acceptance with greedy and with sampled verification.
3. **Ship gate:**
   - an end-to-end phone gain of at least 1.3× at 4k *and* 32k;
   - tool-call replay sweep and scorecard behaviour held (NLD's IFEval −3 is the risk to watch);
   - judged on behaviour, never on a loss number.

A `<|mask|>` token is **not** a run 0 requirement. Appending one row later is as cheap as now, because teacher KL covers only the first 248,320 rows. If the special-token block is being written anyway, one unused row is harmless, but it is housekeeping, not an inclusion.

### 4.3 How it relates to our x0-flow media refinement
- **Our media path already is Cyril's "chunk output, diffusion in chunks with refinement passes" [R ARCH §5].**
  - A noised media span attends the prefix and the whole span and is refined N times (16 steps, distilled to 4 in S6).
  - Then a causal commit pass writes it into the KV cache and GDN state.
  - Joint refinement of *all* units of an image or clip is what the bidirectional-in-span mask does; for the 18 GDN layers it is completed by the A8 reverse scan or a larger global share.
  - Speech runs in 4-8-unit chunks with 4 flow steps each.
- **A-DIFF's text draft is the discrete form of the same mask.** Its masked block is a span that is bidirectional inside and causal to the prefix, then committed, which is our refinement mask (survey_lit line 166). If A-DIFF ever ships, text and media share **one** generation mechanism through the same trunk: draft a span bidirectionally, refine it, commit it causally. That fits Cyril's unified, no-tower intent. It is the best architectural argument for keeping the arm alive, but it is not evidence that it speeds anything up.
- **TiDAR adds exactly one thing for media: fused commit plus next refinement.**
  - Chunk n's causal commit and chunk n+1's first refinement step run in a single forward. That is "verify the old, draft the new in one pass" applied to x0-flow chunks.
  - It needs no training change, because the noisy span already attends a clean prefix.
  - For audio at 4 steps per chunk it saves about 1 of 5 forwards (about 20%) [I]. For an image refined jointly, it saves the separate commit pass when generation continues.
  - **Verdict: WORTH A SMALL EXPERIMENT LATER**, as inference scheduling in the P1/P2 phone experiments. Check it A/B by listening and viewing, and measure time to first audio.
- **Media speed otherwise comes from levers we already have,** not from TiDAR. Media cost is (flow steps) × (span tokens per forward), and TiDAR changes neither.
  - Fewer flow steps (the S6 16→4 distillation).
  - The same-weights draft-then-refine at lower resolution (the H3 SOL structure). In the round-4 diagnostic it lifted video-out from 0.400 to 0.625 [R ARCH §14].
  - Training at real image sizes (Cyril's ruling against 2x2 toys).
  - Cyril already expects video and images to be slow.

## 5. What this means for the run 0 operator
- **Change nothing in the run 0 config, data mix or schedule.**
- Keep Sage-A1.5 outputs and any Sage-derived data out of every corpus and teacher cache.
- The P-SPEC phone microbenchmark can run any time, on the existing Qwen3.5-2B GGUF and independently of run 0. It is the cheapest way to settle whether *any* drafting can help the phone reach 50 tok/s.

## Evidence quality

| Candidate | Identification | Claims |
|---|---|---|
| MemT-1.0 | Solid | Weak to moderate: a two-day-old preprint with internal contradictions. |
| Sage-A1.5 | Solid | Weak to medium: vendor-run benchmarks, contradicted by the paper's own table. |
| DeepInfer | Solid | Weak for our use: vendor numbers, cluster-scale. |
| TiDAR | Solid | Moderate: peer-reviewed but not reproducible. NLD is moderate to good (weights released, includes an equal-token ablation). |

All results are on pure-attention models. Every phone number in this file is a projection or an inference [I]. None was measured.
