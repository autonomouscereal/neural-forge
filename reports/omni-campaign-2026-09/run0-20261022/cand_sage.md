# Candidate review: Sage-A1.5 (MoE) for Chimera Omni v2

Date: 2026-10-22, re-verified 2026-10-27 (see section 8). Read-only review requested by Cyril ("[Cyril quote withheld]"). Sources were fetched on 2026-10-22 and again on 2026-10-27, and are listed at the end. Tags: **[P]** primary source (paper, model card or config), **[S]** secondary source, **[I]** our inference. The 2026-10-22 version is kept at `backups/cand_sage.20261022.md` (sha256 fd7b0c5f…7012).

## Verdict

**Identified, with high confidence.** Sage-A1.5 is InternScience / Shanghai AI Laboratory's agentic post-train of a Qwen3.6-35B-A3B-class MoE. The paper is arXiv 2610.28578 (v1, 20 Oct 2026), the weights are at `InternScience/Sage-A1.5`, and the licence is Apache-2.0.

**Recommendation: do not adopt anything from it into the Omni v2 architecture.** It contains no new architecture. Its config is the stock `qwen3_5_moe` (Qwen3.5/3.6-MoE) layout. The MoE part is exactly what Cyril is sceptical of, and it does not fit a phone. Its contributions are post-training methods for agent behaviour: learned subagent orchestration, a learned verification mode, and multi-teacher on-policy distillation with reverse KL.

Two narrow items may be worth keeping. Neither belongs in run 0.
1. **The orchestration and verification behaviours, as later SFT/tool-calling targets.** This is a data and behaviour idea, not an architecture change. It would only matter once the 2B can already do reliable single-agent tool calls.
2. **The on-policy reverse-KL multi-teacher recipe, as a later-stage distillation option.** It fits Cyril's "top-n matching is a valid target". It needs online teacher queries, which our P40 throughput makes expensive.

**Do not use Sage-A1.5 as a teacher or data generator.** It has about 3B active parameters, which falls under the "never small models for data gen" rule. Its provenance is also unclear: some of its SFT trajectories came from unnamed "strong teacher models" running in the Claude Code harness. That means it may carry Claude-derived outputs, which conflicts with "never train on Claude or Codex outputs" (section 6).

## 1. Identification

| Search | Result |
|---|---|
| "Sage-A1.5 MoE model", "\"Sage A1.5\" huggingface" (WebSearch extended) | Direct hits: arXiv 2610.28578, HF paper page, `InternScience/Sage-A1.5` plus FP8/GGUF variants, project page alpha-innovator.github.io/Sage-A1.5, and an aiweekly.co write-up. |

The earlier "found nothing" quick search most likely ran before the release reached the index. The paper is dated 20 Oct 2026 and the HF page showed "published 3 days ago" [P]. There is a single unambiguous match. No other "Sage A1.5" model exists in the results, so no fallback guess was needed.

## 2. What it is

### 2.1 Architecture: inherited, nothing new [P, HF config.json]
`Qwen3_5MoeForConditionalGeneration`, model_type `qwen3_5_moe`:

| Field | Sage-A1.5 | Qwen3.5-2B (our plant source) |
|---|---|---|
| Hidden size | 2048 | 2048 |
| Layers | 40 | 24 |
| Layer pattern | 3 linear-attn (Gated DeltaNet) : 1 full-attn, `full_attention_interval` 4 | same 3:1, interval 4 |
| Full-attn heads (Q / KV / head_dim) | 16 / 2 / 256 | 8 / 2 / 256 |
| GDN key heads × dim | 16 × 128 | 16 × 128 |
| GDN value heads × dim | 32 × 128 | 16 × 128 |
| FFN | MoE: 256 experts, top-8, expert FFN 512, shared expert 512, aux loss 0.001 | dense FFN 6144 |
| Vocab | 248,320 | 248,320 |
| Context | 262,144 native; YaRN ×2 gives 512K per the card | 262,144 |
| RoPE | theta 1e7, partial 0.25, interleaved mRoPE [11,11,10] | identical |
| MTP | 1 layer | 1 layer |
| Vision | separate ViT tower: 27 layers, 1152 wide, patch 16 | separate ViT tower: 24 layers, 1024 wide |
| Params | 35B total (36B in safetensors metadata), about 3B active | about 2B dense |

The paper itself says only "35B MoE … 35B total and 3B active". It gives no architecture section, because the architecture is the base model's [P].

### 2.2 Mechanism: "capability composition" is behaviour, not weights [P]
"Capability composition" does **not** mean weight merging, model composition or folding. The paper defines it as how the model "organizes, coordinates and adapts" problem-solving during execution. It has three parts:
- **Agentic orchestration.** The model decides to call a `spawn_subagent` tool, which starts parallel subagents with isolated contexts. Defaults are 5 tasks per spawn and 20 turns per subagent. It uses exploratory patterns (A∨B∨C) and constructive patterns (A∧B∧C).
- **Agentic verification.** A learned self-check of progress and evidence inside the reasoning. It is switched on by a system-prompt flag (`chat_template_kwargs: {"verification": true}`).
- **Data composition.** Domains (search, SWE, MLE, science, real-world workflows), reasoning modes and harnesses are trained into one model.

### 2.3 Training recipe [P]
- **Stage I, multi-domain SFT.** Demonstrations were collected "under multiple compatible harnesses, including Claude Code, Codex, and internal harnesses". Loss is token-normalised NLL with tool outputs masked.
- **Domain teachers.** Each domain has its own teacher model.
  - Science: SFT of Qwen3.5-35B-A3B on filtered trajectories from an unnamed "strong model".
  - Long-context: 3,512 hard QA pairs, with three trajectories each sampled from DeepSeek-V4-Pro. 1,949 LongBlocks trajectories were kept.
  - Real-world: more than 1,000 workflows. Trajectories came from unnamed "strong teacher models, with Claude Code as the harness".
  - MLE: two-stage SFT, then Windowed Interaction RL (PPO-style clipping, asynchronous updates).
  - SWE: SWE-rebench-V2 plus in-house tasks, run under mini-swe-agent and Claude Code.
- **Stage II, unified post-training.** Outcome RL combined with multi-teacher **on-policy distillation using reverse KL**. The process loss is KL(student ‖ domain teacher), averaged over the student's own generated tokens, with the teacher chosen by domain tag. Each sample gets outcome-only, process-only or mixed supervision. Teachers run "as independent prefill services".
- **Scale and compute.** "Several hundred million supervised tokens" for the real-world set. GPU count and total compute are **not stated**.

### 2.4 Results [P, model card / paper Table 1; vendor-run]
Selected rows: Sage-A1.5 vs its predecessor Agents-A1 vs base Qwen3.6-35B-A3B.

| Benchmark | Sage-A1.5 | Agents-A1 | Qwen3.6-35B-A3B |
|---|---|---|---|
| BrowseComp | 81.7 (82.8 with orchestration) | 75.5 | 67.9 |
| SWE-bench Pro | 61.2 | 43.6 | 49.5 |
| SWE-bench Multilingual | 72.7 | 58.7 | 67.2 |
| MLE-Bench-Lite | 65.2 | 43.9 | 37.9 |
| IFBench | 81.6 | 80.6 | 64.4 |
| AA-LCR | 81.0 | 74.2 | 71.7 |
| HLE with tools | 55.1 | 47.6 | 36.2 |
| FrontierScience-Research | 48.3 | 40.0 | 2.9 |
| GAIA | 95.1 | **96.0** | 78.6 |
| MatTools | 42.6 | **47.1** | 15.9 |
| OmniDocBench | 91.5 | 89.5 | 89.9 |

Other reported effects:
- Verification mode lifted BrowseComp from 61.0 to 85.5, against 81.0 in normal mode. This comes from an ablation in the paper.
- Orchestration lifted DeepResearch-Bench II from 43.7 to 47.5.

It still trails the T-scale models (Kimi-K3, GLM-5.3, GPT-5.6-sol) on SWE-Multilingual, HLE, CritPt, ALE-CLI and VitaBench.

The paper's own limitations [P]:
- The unified model "does not yet fully retain the peak performance of every domain teacher".
- Ultra-long trajectories accumulate context overhead.
- Orchestration and verification are trained mainly on search and research tasks.

### 2.5 Inference speed claims [P]
**No tokens/s or latency numbers anywhere** (paper, card or project page).
- **Hardware support** is claimed for Ascend 910B/C, A100/H100/H200/B200, RTX cards, MI300X and Apple Silicon, with no measurements.
- **Serving examples** use TP=1 on one GPU. The card suggests MTP / EAGLE speculative decoding.
- **pVisor memory offload** cut resident memory by about 80%, from one run per condition in preliminary tests.
- **Quantised size:** Q4_K_M is about 21.7 GB and Q8_0 about 37.8 GB [P, HF GGUF repos].

## 3. What it would change for Chimera Omni v2

| Omni v2 constraint | Sage-A1.5 | Effect |
|---|---|---|
| On-device dense 2B/4B | 35B-A3B MoE, Q4 file about 22 GB | Not deployable on the phone. The MoE gives no lesson we need. Cyril's rule is dense by default, with fixed MoE only for lookup tables. |
| Raw media through shared weights, no towers | Separate ViT tower plus mRoPE, the same pattern as Qwen3.5 | Nothing to learn here. It is the tower design we are removing. The paper makes no multimodal claim beyond OmniDocBench. |
| No sparse attention | GDN 3:1 hybrid, same as our plant | Already in co2-2b through the Qwen3.5-2B shape. Nothing new. |
| 50+ tok/s on the phone | No speed data. Suggests MTP/EAGLE | MTP and speculative decoding already measured poorly on our phone. Nothing to adopt. |
| 256k context | 262,144 native, YaRN ×2 to 512K | Same RoPE settings as the Qwen3.5-2B plant. Nothing new. YaRN ×2 is a generic, known option. |
| Run 0: plant Qwen3.5-2B, ablate, distill | — | **No change.** Nothing in Sage alters the run 0 plan. |
| Output-decision token, modality tokens | Its "verification" switch is a chat-template flag plus learned behaviour | A weak analogy only. Our output-decision token must come from understanding the request, which we already plan. Sage adds no evidence for it. |
| Tool calling (Cyril's binding ask) | `spawn_subagent` orchestration and a verification protocol, learned end to end | **The one relevant idea.** A later SFT target once the base can make single tool calls. On a 2B it would need to be re-proved on behaviour. |
| Distillation objective | On-policy reverse KL, per-domain teacher, mixed outcome and process supervision | A candidate for a **later** stage after offline top-k KD (survey_lit.md §2.4). It needs live teacher prefill on student samples. With K3 remote and the P40s at about 380-535 tok/s (survey_hw.md §6), it is expensive. Not for run 0. |
| Folding next pass | Same width (2048), same vocab, same layer pattern, same KV/GDN key geometry as co2-2b | Could in principle be a shape-compatible parent. However, MoE→dense expert collapse is folding-like, and Cyril treats dense↔MoE conversion as cheating. Its knowledge is mostly an agentic post-train of a Qwen base we can take from the original parents directly. Not recommended. |

## 4. Compatibility with the Qwen plant [P configs; I]
- **Same tokenizer and vocab (248,320) and the same RoPE/mRoPE settings** as Qwen3.5-2B. Exact logit KD from Sage into co2-2b would need no vocab mapping. This is the only compatibility fact that matters, and it cannot be used because of the teacher rules in section 6.
- **Weight transplant is not direct.**
  - Q heads differ (16 vs 8).
  - GDN value heads differ (32 vs 16).
  - The FFN is MoE (256 × 512 plus 512 shared) against dense 6144.
  - It has 40 layers against 24.
  - Untied input/output embeddings (`tie_word_embeddings: false`) against tied in Qwen3.5-2B (added 2026-10-27).
- Planting from Sage instead of Qwen3.5-2B would mean slicing, merging or collapsing experts. That is folding work, Cyril put it in the next pass, and it uses a non-original parent.

## 5. Cost and risk
- **Run 0 cost if adopted: none, because nothing is adopted.** Not adopting it carries no risk to run 0.
- **Download, if ever used as a reference model:** bf16 is about 70 GB, Q8_0 about 38 GB and Q4_K_M about 22 GB. This is disk and bandwidth on a NAS that has been near full. It would serve no purpose for run 0.
- **Risk if used as a teacher:**
  - Rule violations (section 6).
  - The capacity-gap and teacher-quality literature in survey_lit.md suggests a 3B-active agent model adds little over the original parents (K3 is primary).
- **Risk if the orchestration/verification SFT is copied too early:** a 2B could learn the surface protocol (spawn and verify tokens) without the underlying competence. That is a behaviour regression we would only catch with behaviour review.

## 6. Rule conflicts
- **"Never use small models (incl. Qwen3.8-27B) to generate training data."** Sage-A1.5 has about 3B active parameters and is a post-train of a 35B-A3B. It is small by that rule.
- **"Never train on Claude or Codex outputs."** Stage I demonstrations were "collect[ed] … under multiple compatible harnesses, including Claude Code, Codex". Real-world trajectories came from unnamed "strong teacher models, with Claude Code as the harness". The teacher models are not named, so Claude-generated content cannot be ruled out. Distilling from Sage could pass Claude/Codex-derived behaviour into our model. **Treat it as excluded unless Cyril decides otherwise.**
- **"Full distill from ORIGINAL parents (K3 99%)."** Sage is not an original parent. It is a derivative of the Qwen3.6 MoE.

## 7. Evidence quality
- **Identification: strong.** The paper, the official HF org, the config and the project page all agree.
- **Architecture: strong.** It comes from config.json. The paper adds no architectural claims.
- **Benchmarks: medium to weak.**
  - All numbers are vendor-run, often with custom harnesses (mini-swe-agent, temperature 1.0, 200K context with compaction).
  - Several comparison numbers are taken from other models' reports.
  - The paper says it "matches or surpasses" Agents-A1 and Ornith-1.5 on all twelve benchmarks, but its own Table 1 contradicts this: GAIA and MatTools trail Agents-A1, and DRB-II trails Ornith-1.5.
  - ALE-CLI is reported as 19.0 in the text and 19.1 in the table.
  - We found no independent replication. The aiweekly write-up only repeats the vendor numbers.
- **Speed and memory: none or anecdotal.** There are no tok/s figures, and the memory saving comes from one run per condition.

## 8. Re-verification 2026-10-27

Repeated the searches ("Sage-A1.5 MoE model", "\"Sage A1.5\" huggingface", WebSearch extended) and re-fetched the paper, model card, config, project page and the Qwen3.5-2B config. **Verdict and recommendation unchanged: identified, adopt nothing for run 0, do not use as a teacher.**

Confirmed again [P]:
- `InternScience/Sage-A1.5` config: `qwen3_5_moe`, 40 layers, hidden 2048, 256 experts top-8 (expert FFN 512, shared 512), full-attention every 4th layer (30 GDN + 10 full), 16 Q / 2 KV heads × 256, GDN 16 key / 32 value heads × 128, vocab 248,320, 262,144 positions, rope theta 1e7, partial 0.25, mRoPE [11,11,10], 1 MTP layer, separate 27-layer ViT (1152 wide, patch 16).
- Qwen3.5-2B text config matches the section 2.1 column: 24 layers, hidden 2048, dense FFN 6144, 8 Q / 2 KV × 256, GDN 16/16 × 128, same vocab/RoPE/MTP; ViT 24 × 1024.
- Paper section 2.6: "We collect demonstrations under multiple compatible harnesses, including Claude Code, Codex, and internal harnesses." Section 2.5.2: trajectories "from strong teacher models, with Claude Code as the harness." Section 2.4: three trajectories per QA task from DeepSeek-V4-Pro. Rule conflict in section 6 stands.
- Training compute (GPU count, hours) is still not reported anywhere. No official tokens/s figure exists.

New or corrected items:
- **Embedding tie differs.** Sage-A1.5 has `tie_word_embeddings: false`; Qwen3.5-2B has `true` [P]. Another reason a direct weight transplant is not clean (section 4).
- **Only throughput number found is community, single run [S].** prithivMLmods' MXFP4_MOE GGUF card reports 270.4 tok/s generation and 94.1 tok/s prompt processing on one RTX PRO 6000 Blackwell (96 GB), 8,192 context, MTP draft (max 2 draft tokens), 19.72 GiB peak VRAM. That is a datacenter GPU, a short context and one run. It says nothing about a phone or a 2B dense model, and it does not change section 3.
- **Name collision, not the target.** `amy-sage/sage-a1.5` on HF is a third-party "abliterated Q4_K_M build of Qwen3.5-MoE", unrelated to the InternScience release. `IsValorum/Sage-A1.5-APEX-I-*` are community re-quants of the official model. Neither is the model Cyril means.
- **Paper self-inconsistency confirmed again.** The introduction says it "matches or surpasses" Agents-A1 and Ornith-1.5 on all twelve benchmarks; Table 1 has Agents-A1 ahead on GAIA (96.0 vs 95.1) and Ornith-1.5 ahead on DeepResearch-Bench II (48.6 vs 43.7).

## Sources
- [P] Sage-A1.5 tech report, arXiv 2610.28578: https://arxiv.org/html/2610.28578 (HF paper page https://huggingface.co/papers/2610.28578)
- [P] Model card: https://huggingface.co/InternScience/Sage-A1.5
- [P] Config: https://huggingface.co/InternScience/Sage-A1.5/blob/main/config.json
- [P] GGUF/FP8 variants: https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF, https://huggingface.co/InternScience/Sage-A1.5-Q8_0-GGUF, https://huggingface.co/InternScience/Sage-A1.5-FP8
- [P] Project page: https://alpha-innovator.github.io/Sage-A1.5/
- [P] Qwen3.5-2B config (plant comparison): https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
- [S] aiweekly.co summary: https://aiweekly.co/alerts/sage-a15-moe-agent-hits-817-browsecomp-612-swe-bench-pro
- [S] Community quantisations: https://huggingface.co/prithivMLmods/Sage-A1.5-MXFP4_MOE-MTP (single-run speed numbers, section 8), https://huggingface.co/Duckets/Sage-A1.5-MLX-4bit, https://huggingface.co/IsValorum/Sage-A1.5-APEX-I-NanoPlus-V2.1-GGUF
- [S] Name collision (not the target): https://huggingface.co/amy-sage/sage-a1.5
- [S] Social post repeating vendor stats: https://x.com/TeksEdge/status/2112974699835859236
- [P] Official GGUF sizes: https://huggingface.co/InternScience/Sage-A1.5-F16-GGUF, https://huggingface.co/InternScience/Sage-A1.5-Q6_K-GGUF
- Local context: run0-20261022/survey_lit.md (§2.4 KD recipe), run0-20261022/survey_hw.md (§6 P40 throughput)
