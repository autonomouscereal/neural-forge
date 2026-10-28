# Candidate review: MemT-1.0 ("memt 1.0") for Chimera Omni v2

Read-only review, 2026-10-22, re-verified against the primary sources on 2026-10-28. The 10-28 pass made three corrections: the 2B branch size and its decode cost (§3), the P40 token budget (§3, §4), and the Metis-4B numbers, which were missing before (§2.6). The previous text is kept in `backups/cand_memt.20261022.md`. Cyril asked: "[Cyril quote withheld]".
Nothing was trained, launched or changed. Sources are listed at the end; `[P]` = primary (paper, repo, model card), `[2nd]` = secondary press, `[E]` = my estimate, not measured.

## 0. Verdict

- **Identified: FOUND.** "memt 1.0" is **MemT-1.0: Towards Memory-Native Evolution**, MemTensor, arXiv 2610.26825.
  - v1 was submitted 2026-10-19 and v2 on 2026-10-20, which is two to three days before Cyril's ask. It has 74 authors (Zhiyu Li et al.), 66 pages, 15 figures and 16 tables. [P1]
  - The Chinese press uses his exact spelling: "记忆张量MemT 1.0发布并开源". [2nd1]
- **It is not an architecture for our trunk.** It bolts a fixed-size recurrent memory branch onto a frozen LLM. That gives the model **cross-turn and cross-session memory without keeping the KV cache**.
  - It does not make decode faster.
  - It does nothing for media.
  - It is lossy: about 19 points below full context on its own memory suite.
- **Recommendation:**
  - **Do not add it to run 0.** It would be an untested new block, it trains against a frozen backbone, and Cyril said to skip testing on this pass.
  - **Keep one idea as a post-run-0 candidate** (A-list): a persistent memory branch on the 6 (2B) / 8 (4B) exact GQA layers, for Sprite's phone memory beyond the 32k exact window. It should be judged on behaviour.
  - Two training details corroborate things we already planned. Neither is new:
    - top-K (K = 64) KL distillation on response tokens (survey_lit §2.4 already recommends this);
    - windowed recompute of the recurrent state for long sequences.

## 1. Identification evidence and rejected candidates

| Candidate | Why it might be it | Why it is or is not | Verdict |
|---|---|---|---|
| **MemT-1.0 (MemTensor, arXiv 2610.26825, 2026-10-19)** | The exact name "MemT-1.0"; repo `MemTensor/MemT-1.0`; HF `MemTensor/MemT-1.0-Nano`; released the week of the ask; Chinese launch coverage | Name, version and timing all match | **It** [P1-P4, 2nd1-2] |
| Metis: Memory Foundation Model (MemTensor, arXiv 2607.26760, 2026-07-29) | Same lab; MemT-1.0's direct predecessor; built on Qwen3.5 4B/9B/27B | Different name; MemT-1.0 explicitly supersedes it | Predecessor, reviewed in §2.6 [P5] |
| MeMo: Memory as a Model (arXiv 2605.15156, May 2026) | Similar name | A separate memory *model* beside a frozen LLM; no "1.0" | Not it [P6] |
| Mem-T-4B (Qwen3-4B fine-tune, Jan 2026) | Similar name | A fine-tune, not an architecture paper. It came from the task's earlier quick search and was not re-verified here | Not it |

## 2. What it is

### 2.1 Mechanism [P2 §2]

- **A parallel memory branch on selected decoder layers.**
  - The backbone's attention path is untouched. The output is `A = A_ori + A_mem`.
  - Nano puts the branch on the 10 full-attention layers 3, 7, …, 39 of Qwen3.6-35B-A3B. That is the every-4th global layer of the hybrid GDN layout, the same family as our plant. [P2 App. B]
- **Local memory.**
  - The state is per layer, `M ∈ R^{H_kv × d_k × d_v}`. It starts at zero for each trajectory and has a fixed size, independent of history length.
- **Hyper memory.**
  - The trainable parts are the memory projections, the erase/write/forget gates and an output gate.
  - They are frozen at inference. Only the state evolves, through forward passes alone (no test-time gradients).
- **Update.** A token-level "GDN-2" delta rule:
  - `M_i = [I − UᵀU]·D_i·M_{i−1} + Uᵀ(ω ⊙ V)`, with `U = √β ⊙ K` and L2-normalized keys.
  - The readout uses the query, then RMSNorm, then a SoftCapSiLU gate (τ = 2).
- **Mixture-of-Storage (MoS).** E parallel memory experts with **dense** routing (not sparse MoE). Separate storage and utilization routers share an expert-address matrix.
  - Both released models ship with **E = 1**, so MoS is not in the released checkpoints. [P2 App. B]
- **Memory lifetime.** Training and evaluation reset it per trajectory. The serving API exposes `save_state` / restore for persistent sessions. [P2 §7.2, P3, P4]

### 2.2 Scale [P2 §1, App. B; P3; P4]

| | MemT-1.0-Nano | MemT-1.0-Flash |
|---|---|---|
| Backbone | Qwen3.6-35B-A3B (as released) | DeepSeek-V4-Flash, post-trained first |
| Total / active | 36B / 3B | 288B / 17B (§3.1 also says 284B / 13B; not reconciled) |
| Memory layers | 10 | 20 |
| H_kv, d_k = d_v | 16, 256 | 64, 512 |
| Trainable memory params | 450.9M | 3.61B |
| State per trajectory | 40 MiB f32 (= 10.5M values, the "10M dynamic memory") | 1,280 MiB f32 ("336M") |

- **Inconsistency:** an appendix also describes 10 blocks with 512×512 states and 104,962,580 trainable params (about 0.3 % of the backbone). It is unclear which configuration that is; possibly the agentic-experiment variant. [P2]
- No dense model under 4B, and no multimodal variant, exists or is mentioned.

### 2.3 Training recipe [P2 §3]

- **Backbone frozen; only the memory parameters train.** The frozen backbone with full context is the teacher.
- **34B tokens in total:**
  - Stage 1 is continued pre-training on 20B tokens. It uses CE over 2K-token segments, with the state carried across segments and reset per sequence.
  - Stage 2 is post-training on 14B tokens. It uses **top-K (K = 64) KL distillation from the full-context teacher**, with loss on response tokens only and gradients flowing through memory reads and writes. MoS adds a router load-balancing loss.
- **Data:** context-query-answer format; part of it comes from MemOS's anonymized internal data, so it cannot be reproduced.
- **Optimizer and schedule:** AdamW (0.9/0.999, wd 0.01, clip 1.0) with WSD, peak LR 1e-4 for Nano and 8e-5 for Flash.
  - Batch: 64 packs of up to 64K tokens (Nano).
- **Precision:** the backbone runs in bf16. Memory states, norms, softmax and optimizer state are f32.
- **Initialization (App. B):**
  - Nano's memory Q/K/V are **copied from the backbone attention projections**. Each of the 2 backbone KV heads is copied into 8 memory heads.
  - Gate inputs start at N(0, 0.02²) and the forget-gate expansion at zero. Forget half-lives are sampled log-uniformly, and states start at zero.
  - On our side this maps directly to initializing from the planted Qwen GQA weights.
- **Stage ablation (Table 4, Nano memory average 46.97):**
  - Without CPT it scores 44.85 (−2.12).
  - Without post-training it scores 36.28 (−10.69). Almost all of the gain comes from the top-64 KD stage.
- **Framework:** Megatron-Core, Nano at EP 8 / DP 64. **The GPU type, count and hours are not stated.**
- **Windowed execution (App. F / Table 1):** 32K pack on one GPU, step 2,048.
  - A 2,048-token window cuts peak memory from 47.83 to 6.03 GiB (−87.4 %) for +6.6 % time.
  - A 4,096-token window cuts it to 9.49 GiB for +1.6 % time.

### 2.4 Results [P2 Tables 2-3, §5, §7.4]

- **Memory suite (11 benchmarks, average):**

  | System | Average |
  |---|---|
  | Flash | 50.60 |
  | Nano | 46.97 |
  | Best other parametric method (Temp-LoRA) | 38.08 |
  | Full context, Qwen | 66.27 |
  | Full context, DeepSeek | 64.98 |
  | MemOS, textual memory | 68.55 |
  | Mem0 | 65.53 |
  | DenseRAG | 57.28 |

  - MemT beats the other parametric methods but **trails full context by about 19 points** (Nano vs Qwen full context).
  - On the long recall benchmarks it collapses: LongMemEval-S 37.20 vs 83.20 (MemOS), and BEAM-1M 37.04 vs 72.59.
- **General ability with memory on:**
  - Flash: average change 0.0 vs its backbone (MMLU-Pro −1.8).
  - Nano: −0.9 average, **LiveCodeBench v6 −5.8**, IFEval +3.3.
- **Ablations (Nano):**
  - The GDN-2 "stable" update adds very little: 46.97 vs 46.72 (plain GDN) vs 46.43.
  - **The token-level update is what matters.** Turn-level updates score 40.98.
  - Bidirectional encoding is worse, at 45.43.
- **Scaling:**
  - Two epochs raise the average from 36.65 to 46.80.
  - 4 MoS experts beat 2 by 2.0-2.7 points. This is an ablation only; the release uses E = 1.
- **Agentic results** (Qwen3.6 backbone, with past experience internalized):
  - SWE-bench Verified: success 66.07 → 69.33 %, pass^3 54.0 → 61.6 %.
  - SWE-bench Multilingual: pass^3 45.33 → 53.67 %.

### 2.5 Speed and storage claims [P2 §6.5-6.6]

- **Setup:** HF Transformers on **two GPUs of an unnamed model**, batch 1, 128 generated tokens, averaged over 35 BEAM histories.
- **End-to-end speedup vs re-prefilling the full context:**

  | History length | Speedup |
  |---|---|
  | 32K | about 1.0× |
  | 64K | 1.20× |
  | 128K | 1.86× |
  | 256K | 3.99× |

  - Query TTFT stays at 104-112 ms. Full context reaches 257 ms at 128K and 445 ms at 256K, which makes MemT 2.44× and 4.19× faster to first token at those lengths.
  - The full-context baseline is plain HF inference, not an optimized prefix-cached server, so these ratios are upper bounds for a deployment that keeps its KV cache.
- **Storage:** the per-trajectory state is constant, about 66× smaller than the KV cache at 128K.
- **What the speed numbers are not:** the gain comes from *not re-prefilling history*. **No decode tok/s figure is given anywhere.** No phone, NPU or quantized result exists.

### 2.6 Metis, the predecessor [P5]

- Built on Qwen3.5 4B, 9B and 27B with the backbone frozen.
- Trained on about 406M primary tokens plus about 609k auxiliary samples, 14k steps for 4B, on **8×H100** in bf16.
- Its token *selection* (top-ρ plus straight-through) was replaced in MemT-1.0 by the dense token-level update.
- **Metis-4B is the only evidence at our scale. It is a Qwen3.5-4B backbone, the same family as our plant** (v2 text via alphaXiv):
  - Memory scores: MemOps (gold) 17.84, LoCoMo (gold) 16.31, NextMem 41.69, and 56.72 on its own test set. The 27B scores 24.76 / 26.74 / 50.82 / 73.77.
  - General ability with an **empty** memory is roughly flat: MMLU-Pro 45.20 vs 46.00, IFEval 79.85 vs 79.30, GSM8K 82.03 vs 83.09.
  - **After irrelevant memory is stored, the "active" stage, ability drops sharply:**
    - IFEval falls from 76.71 to 54.53 (−22.2).
    - MMLU-Pro falls from 46.00 to 40.90.
    - GSM8K falls from 84.53 to 78.92.
    - MMMLU falls from 59.90 to 56.60.
  - The authors also report that accuracy falls when a single write exceeds a few hundred words or as sequential updates accumulate, plus "information confusion" between stored items.
  - **Read this as: at 4B, a populated memory state can damage instruction following badly.** MemT-1.0 has no ≤4B result showing that this was fixed.

### 2.7 Evidence quality

- **Weak to moderate.**
  - A single-lab preprint, two days old and not peer reviewed.
  - Part of the data and the strongest baseline (MemOS) are the authors' own product.
  - No training hardware is disclosed, the efficiency GPUs are unnamed, and no decode throughput is reported.
- **Internal inconsistencies:** Flash size (288B/17B vs 284B/13B) and memory parameter count (450.9M vs 104.96M).
- **Licensing is contradictory:**
  - The HF card says Apache-2.0. [P4]
  - The repo code is PolyForm **Noncommercial** and the paper is CC BY-NC-SA. [P3]
  - Anything we take should be re-implemented from the equations, not copied from their code.

## 3. What it would change for Chimera Omni v2

| Our requirement | MemT-1.0 effect |
|---|---|
| Dense 2B/4B, Qwen3.5-shaped plant (18 GDN + 6 GQA / 24 + 8) | **Structurally compatible.** It targets exactly the global-attention layers of the same hybrid family: layers 3, 7, …, 23 on co2-2b.<br>**Size if we follow Nano's recipe** (memory heads = q heads, copied from the KV heads): 8 heads × 256 × 256 × 6 layers = 3.1M state values, which is **12 MiB f32**. The parameters scale from Nano's 45M per layer at the same width 2048 with twice the heads, which gives **roughly 0.1-0.14B, about 5-7 % of the 1.92B trunk**. [E]<br>A minimal H_kv = 2 variant would be about 3 MiB and a few tens of M params, but it is untested. (The 10-22 text's "a few M params" was too low.) |
| Raw media in/out, shared weights, no towers | Neutral. The branch operates on hidden states, so it does not violate the rule. **But there is zero multimodal evidence**, and a fixed state fed 1,400 tok/s of video will be overwritten fast. Untested. |
| No sparse attention | Compliant. It is a dense recurrent state, not sparse attention. |
| 50+ tok/s on the phone | **No help inside the exact window; a measurable cost.** Each memory layer adds Q/K/V/gate projections plus a delta-rule update and a read per decoded token. At Nano's recipe that means reading about 0.1-0.14B extra weights per token. Phone decode is bandwidth-bound, so that is **about 5-7 % slower decode** [E]. The state math itself is small (8 × 256 × 256 per layer).<br>Its one speed upside is beyond the exact window. Past turns move into a 12 MiB state instead of KV: the 6 GQA layers cost 12,288 B/token in bf16, which is about 400 MB at 32k and 3.2 GB at 256k [E, from the co2-2b shape]. **No decode tok/s is claimed anywhere in the paper.** |
| 50k minimum, 256k capable | **It does not replace exact context.** It is lossy (about 19 points under full context, collapsing at 1M). Speed parity is at 32K, our phone's exact window. Its legitimate use is *beyond* that window, or across sessions, where our 6 global layers would otherwise drop history. Our 18 GDN layers already carry a fixed state over the whole context. |
| Trained on 4 P40s in f32 | Feasible only as a **small add-on phase** after run 0: train the branch only, at Metis scale (well under 1B tokens), with windowed recompute at 2K steps. Not at the 34B-token MemT scale. survey_hw measures 380-535 tok/s across 4 P40s, which is 33-46M tokens/day, so 34B tokens would take **about 740-1,030 days**. Even Metis-scale (about 406M tokens) is **about 9-12 days, all four P40s for the whole 10-day window**. That holds even though a frozen trunk skips weight gradients, because activation backward through the trunk is still needed. [E] (The 10-22 text's "600 tok/s, 650 days" overstated throughput.) |
| Run 0 = planted co2-2b, ablated, then full distillation | **Conflicts if added now.** MemT assumes a frozen, finished backbone and trains only the branch. Run 0 is a full-parameter relearn. Adding the branch mid-relearn changes the architecture without a test, against Cyril's "[Cyril quote withheld]". |
| Next pass = folding from parents | Orthogonal. A branch trained on run 0's backbone would need retraining after folding changes the trunk. That argues again for adding it last, on a settled backbone. |
| Pass/fail on behaviour; top-n matching is valid, top-1 is not | Its stage-2 loss (top-64 KL on response tokens) is the kind of target Cyril allows. It **corroborates** survey_lit §2.4, which already plans top-32/64 plus sampled tail. Nothing new to add. |

## 4. Cost and risk

- **Cost to add later (branch only, post-run-0):**
  - About 1 engineer-day to implement from the equations. We already have the chunked GDN scan (`chunked_gated_delta_rule`, ARCH §1), so the update is a reuse. [E]
  - Plus a P40 phase on context-query-answer data built from our own corpus. MemOS data is not available, and Claude/Codex outputs are excluded.
  - A llama.cpp/HTP export needs one extra GATED_DELTA_NET call per global layer, which our runtime already has.
- **Risks:**
  - Lossy recall can produce confident wrong "memories".
  - Nano's −5.8 on LiveCodeBench shows the branch can bleed into general ability. It would need a gate (zero-init output) and behaviour review.
  - **The worst at-scale evidence is Metis-4B:** IFEval fell 22 points once irrelevant memory was stored (§2.6). The behaviour gate must test a *populated* memory full of unrelated history, not an empty one.
  - Training it would consume a full 4-P40, 10-day window. That window is better spent on run 0 and folding.
  - The value for media is unknown.
  - The paper's numbers are unreplicated, and the licence confusion means we must not copy their code.

## 5. Recommendation

1. **Run 0: no change.** Do not add MemT blocks.
2. **Already covered, no action:** top-K KL distillation (survey_lit §2.4). If long sequences hit P40 memory limits, windowed recompute of recurrent state (MemT App. F: −80 to −87 % peak memory for +1.6 to +6.6 % time) supports the checkpointing already in use (survey_hw §135).
3. **Candidate A-item after run 0 and folding, on a settled trunk: "persistent session memory".**
   - Design: a zero-initialised MemT-style token-level delta-rule branch on the exact GQA layers, backbone frozen, trained against the same model with full context.
   - Behaviour gate:
     - The phone recalls facts and preferences from earlier sessions without them in context.
     - No regression in general behaviour.
     - Media spans checked separately.
   - Ship only if it beats simply keeping more exact KV at the same memory.
4. **Skip MoS.** Neither release uses it (E = 1), and the gain is only an ablation result.

## Sources

- [P1] arXiv abs, MemT-1.0: Towards Memory-Native Evolution, 2610.26825 (v1 2026-10-19, v2 2026-10-20): https://arxiv.org/abs/2610.26825
- [P2] Full text (HTML v2), sections 2-8 and appendices B and F: https://arxiv.org/html/2610.26825
- [P3] Official repo (SGLang inference, licences): https://github.com/MemTensor/MemT-1.0/
- [P4] HF model card, MemT-1.0-Nano: https://huggingface.co/MemTensor/MemT-1.0-Nano ; HF paper page: https://huggingface.co/papers/2610.26825
- [P5] Metis: Memory Foundation Model, arXiv 2607.26760 (2026-07-29): https://arxiv.org/abs/2607.26760 , https://arxiv.org/html/2607.26760v2
- [P5b] Metis v2 full text via alphaXiv (Metis-4B/9B/27B tables, general ability before and after memory): https://www.alphaxiv.org/abs/2607.26760
- [P6] MeMo: Memory as a Model, arXiv 2605.15156 (2026-05-14): https://arxiv.org/abs/2605.15156
- [2nd1] The Paper (澎湃), MemT 1.0 launch: https://www.thepaper.cn/newsDetail_forward_34288299
- [2nd2] AI Weekly, "MemT-1.0 Memory-Native LLMs Cut KV Storage 66x at 128K Tokens": https://aiweekly.co/alerts/memt-10-memory-native-llms-cut-kv-storage-66x-at-128k-tokens
- Internal: run0-20261022/survey_lit.md §2.4 (top-k KD), run0-20261022/survey_hw.md (P40 throughput), ARCHITECTURE_V2_2026-10-21.md §1, §6, §7 (layer counts, phone timing, 32k exact window).
