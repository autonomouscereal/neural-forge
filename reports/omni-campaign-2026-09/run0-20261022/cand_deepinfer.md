# Candidate review: DeepSeek DeepInfer (re-verified 2026-10-27; first written 2026-10-22)

This is a read-only review. Cyril asked: "[Cyril quote withheld]"

**2026-10-27 re-check.** Every primary source was fetched again. The verdict is unchanged. The re-check added three things:

1. vLLM serving facts for V4.1-Flash: MTP was dropped, the drafter is about 14B parameters, and the model needs at least 614 GB of VRAM.
2. A note that CSA2 is sparse attention, so it is excluded by our rules.
3. The cross-layer KV reuse idea, which is the dense-compatible part of CSA2.

## 1. Identification: FOUND (high confidence)

"DeepInfer" is **deepseek-ai/DeepInfer**. DeepSeek released it on **2026-10-20** under the Apache 2.0 licence, and it is DeepSeek's in-house inference framework. It is not DeepInfra, a different company that hosts DeepSeek models. The earlier quick search confused the two.

Primary sources:
- https://github.com/deepseek-ai/DeepInfer, the README. Fetched 2026-10-27: about 23.5k stars, 2.8k forks, 190 PRs. On 2026-10-22 it had 19.7k stars.
- https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md, the deployment Q&A.
- DeepSeek's announcement on X: https://x.com/deepseek_ai/status/2112476496283775135. It reads "Introducing DeepInfer, our in-house inference framework, now fully open source!"

Independent news that agrees:
- SiliconANGLE, 2026-10-20: https://siliconangle.com/2026/10/20/deepseek-open-sources-deepinfer-the-inference-framework-behind-its-production-services/
- The Next Web: https://thenextweb.com/news/deepseek-deepinfer-open-source-inference-engine
- Quartz: https://qz.com/deepseek-open-source-deepinfer-inference-software-ai-102026
- The New Stack: https://thenewstack.io/deepseek-deepinfer-huawei-nvidia/

The techniques inside DeepInfer come from two DeepSeek papers:
- **DSpark**, arXiv 2607.05147, 2026-07-06, by Xin Cheng and 32 others including Wenfeng Liang: https://arxiv.org/abs/2607.05147
- **DeepSeek-V4.1-Flash**, arXiv 2609.19969, 2026-09-17: https://arxiv.org/abs/2609.19969 and https://arxiv.org/html/2609.19969

Third-party serving recipe: vLLM for V4.1-Flash, https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash

Name collisions to ignore:
- DeepInfra, the hosting company.
- Medium and Activepieces explainers that mix the two names up.
- deepinfer.net, a fan site.

## 2. What it is

DeepInfer is a Python engine. Its README says it was "originally built primarily on vLLM", with ideas taken from SGLang and TensorRT-LLM. It runs on DeepSeek's own kernel libraries: FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect, plus custom TileLang kernels.

| Item | Fact | Source |
|---|---|---|
| Models | **One model only: DeepSeek-V4.1-Flash.** The default checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017`. | README |
| Hardware | **NVIDIA SM100 (Blackwell) with CUDA 13, or Huawei Ascend 950 with CANN 9.2.** Hopper and older NVIDIA cards are not listed. Batch invariance and deterministic sampling work on NVIDIA only. | README; SiliconANGLE |
| Build | Linux, GCC 13 or newer, Python 3.12, `./build.sh` | README |
| Scope | Offline inference plus the core features for serving at scale. These are **not included**: the KV-cache system, VE/prefill/decode disaggregation (the README says the engine side is "mostly implemented"), and load balancing across DP replicas. The deployment Q&A says the KV cache and load balancing are "tightly coupled to our internal infrastructure". | README; deployment Q&A |
| Parallelism | Attention runs data-parallel with balanced CP. MoE runs expert-parallel with EPLB. Decode uses wide EP of 64-128. TP and CP are not used in production. The smallest efficient deployment is **about 64 prefill GPUs plus 64 decode GPUs**. | README; deployment Q&A |
| Features | DSpark speculative decoding with confidence-scheduled verification (`DEEPINFER_ENABLE_MTP_VARLEN=1`, up to 5 drafts). Decoder SWA bounded replay (`YOCO_PREFILL_TRUNCATION_MODE=compact`, keep 140 tokens). Async "zero-overhead" scheduling that overlaps CPU work for step N+1 with GPU work for step N. Batch-invariant inference and deterministic sampling. Structured output via regex, JSON schema or EBNF. | README |
| Quantisation | Main KV uses e4m3 with per-16 scaling, which is NVFP4-like but has no global scale. Indexer KV and routed experts use MXFP4 (per-32, ue8m0). SWA KV uses MXFP8 at 528 B per token. | deployment Q&A |
| KV store | Paged block KV is kept on a 3FS SSD cluster (10+2 erasure coding, about 100 PiB). SWA KV is kept in host RAM. TTL is about 3 days for agent traffic and 7 days for chat. | deployment Q&A |
| Benchmarks | Results appear only as Pareto plots, with no numbers in the text: prefill on 4 SM100 GPUs, decode on 64. Production claims are "about 155 trillion tokens/day" per cluster and $38 per GPU-hour at valley pricing. The open FlashMLA decode kernel is 5-10% slower than the internal one, and end to end the gap is 5% or less. **There are no third-party reproductions yet.** | README; deployment Q&A |

### The model it serves: DeepSeek-V4.1-Flash (arXiv 2609.19969, vLLM recipe)

- **Size and activation.** The backbone has 552B parameters, plus 196B Engram n-gram hash tables in layers 1 and 14. Each token routes to 6 of 384 experts plus 1 shared expert. **About 8B parameters are active per prompt token and 16B per output token.** The model has 40 layers (hidden size 5120) behind a 32-layer vision encoder. Context reaches 1M tokens with YaRN at factor 16. Pretraining used 45T multimodal tokens.
- **Causal Encoder-Decoder (CED), described as YOCO-inspired.**
  - The model is split into a 20-layer encoder and a 20-layer decoder.
  - The decoder's global KV is projected from the encoder's final hidden state.
  - Prefill therefore needs only the encoder half for global KV, plus a bounded SWA replay in the decoder. That "reduces nearly half of the prefill computation while maintaining performance comparable to the baseline."
  - **No ablation numbers, baseline identity or ablation scale are given in the text.** The paper says the sparse model was "trained from scratch at 64K". **No recipe for converting an existing decoder-only model is published.**
- **Attention.** Every layer has 128-token SWA, and layers 0-1 use SWA only. The other layers add **CSA2, a compressed sparse attention** in which an indexer keeps the top 512 compressed entries. **CSA2 cross-layer KV reuse:** each layer is Full, Reindex or Reuse, and only layers 2, 8, 14 and 20 compress their own KV while the rest read it.
- **SWA bounded replay.** SWA KV is not persisted. It is rebuilt approximately by replaying the last 128 tokens. The paper calls the degradation "negligible" with no numbers, and warns of possible "capability degradation in untested boundary cases".
- **KV size.** **Global KV is 890 B per token**, about 1/4 of V4-Flash. It uses FP4 main KV with QAT in post-training, which the paper reports as "only marginal performance degradation" with no deltas. Persistent KV is about 1/8 of V4-Flash.
- **MTP.** The vLLM recipe says "V4.1 dropped the MTP module that V3 and V4 trained alongside the backbone." DSpark replaces it, and the drafter adds about 14B parameters.
- **Results against V4-Flash** (Table 3 of the paper, max effort; these are agentic and reasoning scores, not base-model scores):

| Benchmark | V4-Flash | V4.1-Flash |
|---|---|---|
| GPQA-Diamond | 89.9 | 90.9 |
| Terminal-Bench 2.1 | 82.7 | 90.6 |
| DeepSWE v1.1 | 54.4 | 74.2 |
| MathArena Apex | 58.6 | 65.6 |

  The base-model numbers quoted on 2026-10-22 (MMLU-Pro 74.1 vs 68.3, HumanEval 79.4 vs 69.5, GSM8K 93.0 vs 90.8) were not found again in the sections fetched on 2026-10-27. Treat them as unverified.
- **Minimum hardware for the model.** It needs at least 614 GB of VRAM; the checkpoint is about 511 GB. vLLM 0.31 or newer serves it on H200, B200, GB200/GB300, MI300X and MI325X/MI355X. Neither vLLM nor DeepInfer covers consumer, Ampere or Turing GPUs (vLLM recipe).

### DSpark (arXiv 2607.05147)

- **Drafter.** It is semi-autoregressive. A parallel backbone produces 5 draft positions in one pass, and a light Markov head (low rank) adds the dependencies between draft tokens. The production drafter is 3 blocks with 128-token SWA.
- **Confidence head.** It predicts the acceptance probability of each position. The scheduler then verifies only the prefix that maximises expected throughput, and per-request lengths are computed on the GPU by a TileLang kernel.
- **Training.** The drafter is trained after pretraining with the backbone frozen, using responses regenerated by the target model. It keeps training alongside post-training, with no gradient flowing into the backbone.
- **Results.**
  - Against MTP-1 in the V4 serving system, the paper reports **"60 to 85 percent" higher per-user speed at matched throughput** (abstract).
  - In production, confidence scheduling cuts verified tokens per step from 5.00 to 3.2-4.4 while accepted length stays about the same (3.29 vs 3.20). The throughput change is **−1.5% at 16 sequences/rank, +5.4% at 60, and +11.0% at 96** (deployment Q&A).
  - **No batch-1, edge or phone results are reported.**

## 3. Use for Strata (llama.cpp on 3090 SM86 + RTX 8000 SM75 + RAM offload)

**None of it is directly usable.**

- **The engine.** DeepInfer builds only for SM100/CUDA 13 or Ascend 950. Our cards are two and three generations older, and FlashMLA, DeepGEMM and DeepEP target Hopper and Blackwell.
- **The model.** DeepInfer serves only V4.1-Flash. That is not our Strata model, and it would not fit anyway: it needs 614 GB of VRAM against our 24 + 48 GB plus RAM.
- **The scaling tricks.** The gains come from wide EP across 64-128 GPUs, an RDMA/3FS KV store, and EPLB weight shuffling over NVLink. None of that maps onto one node with 2 GPUs and RAM offload.
- **Features we already have.** llama.cpp already provides the ones we could use: grammar-constrained output, an async server loop and a prompt cache.
- **One principle worth keeping: batch-invariant decoding.** If output is the same regardless of batch composition, behaviour evals run against Strata become reproducible. It is a note for the eval harness only. No action is needed now.

## 4. Use for Chimera Omni v2 (dense 2B/4B on-device, raw media in/out, no towers, no sparse attention; run 0 = Qwen plant + ablate + distill)

| Technique | Is it meaningful for us? | Verdict |
|---|---|---|
| **The DeepInfer engine itself** | No. It is a server engine for one model, on Blackwell or Ascend only. | Reject |
| **CSA2 compressed sparse attention / indexer** | It is sparse attention, which Cyril's v2 direction rules out. | Reject |
| **Engram n-gram hash memory** | Its 196B parameters are lookup tables keyed on token n-grams. That is tokenizer-coupled, and it would add a large table to phone memory. It conflicts with the goal of an end model free of tokenizer-specific parts. | Reject |
| **DSpark drafter + confidence scheduler** | Unlikely. Our phone measurements of speculative decoding and MTP were poor. DSpark's evidence comes from large-batch serving against an MTP-1 baseline, and its own scheduler **loses 1.5% at low concurrency** (16 sequences/rank). There is no batch-1 or device data. Its production drafter is about 14B parameters for a 16B-active target, which is heavy relative to a 2B target. Training it needs responses regenerated by the target, which is acceptable only if our own model is the target. | Reject for run 0 and run 1. If speculative decoding is ever revisited on the phone, the only cheap probe is the confidence-truncation idea on its own, and it must be measured on the phone at real sampling. |
| **CED / YOCO split.** The upper-half KV is projected from the mid-depth hidden state, so prefill needs only the lower half plus a bounded replay. | **Potentially yes, after run 0.** Raw media input makes prefill dominate, because real-size images and video become very long prompts, so halving prefill directly serves TTFT. One shared global KV for the whole upper half also cuts KV memory, and at 256k it cuts the KV bandwidth each decode step reads. It does not speed up short-context decode: every output token still runs both halves, so it does not move the 50 tok/s goal at short context. The "encoder" is the lower half of the shared trunk, not a modality tower, so it is compatible with the no-towers rule. | **Not for run 0.** It needs a top-half retrofit (see compatibility below). DeepSeek publishes no decoder-only ablation and no conversion recipe, so the quality cost is unproven. It is a candidate variant for the folding pass, gated on behaviour. |
| **Cross-layer KV reuse** (the dense part of CSA2: only a few layers compute KV, the rest read it) | **Plausible, and smaller than CED.** Our planted shape is a Qwen3.5-style hybrid in which only the full-attention layers hold a growing KV. Sharing KV between neighbouring full-attention layers would shrink 256k KV memory on the phone. | Later, as an ablation-friendly retrofit (share K/V, then distill). Judge it on long-context behaviour. |
| **SWA bounded replay** | It applies only if we add SWA layers. The planted shape uses linear-attention (Gated-DeltaNet) layers plus full attention, and DeltaNet state is already constant-size, so there is nothing to replay. | Not applicable |
| **FP4 KV cache with QAT** (E2M1 values, one E4M3 scale per 16 channels, no global scale) | **Plausibly yes, for 256k on the phone.** The KV of the full-attention layers is what limits long context on device. DeepSeek reports "marginal" degradation when QAT is applied in post-training. | Later, as fake-quant in a distill stage after run 0. Judge it on long-context recall behaviour, not on the reported number. |
| **Batch invariance / determinism** | It makes behaviour judging reproducible. | Principle only |
| **Structured outputs, async CPU/GPU overlap** | llama.cpp and the Sprite runtime already have these. Async overlap is worth a quick check in the Sprite kernel loop if it does not already overlap host sampling with NPU work. | Nothing new for the model. A possible small runtime check for Sprite. |

### Compatibility with the Qwen plant

- **Nothing in DeepInfer requires a change to run 0.** The plant, ablation and full distillation on the 4 P40s proceed as planned in RUN0_BUILD_PLAN.md.
- **CED is the only candidate that changes the architecture, and it conflicts with a straight Qwen plant.** Applying it would mean three changes:
  1. Replace the upper-half attention layers' own K/V projections with projections from the layer-L/2 state.
  2. Convert the upper-half Gated-DeltaNet layers, which still need the whole prompt, into cross-attention + SWA. Otherwise there is no prefill saving.
  3. Relearn the changed layers.
- **Order:** run 0 first as planned; CED and KV reuse after that, as separately distilled variants gated on behaviour, preferably as targets of the folding pass.
- **FP4 KV QAT** is compatible with the plant and needs only trainer fake-quant.

## 5. Cost and risk

- **Adopting the engine.** It cannot run on our hardware, so there is nothing to spend.
- **DSpark on the phone.**
  - A drafter training run, plus target-regenerated data that must come from our own model, plus phone measurement.
  - The expected gain at batch 1 is unproven, and the scheduler's own evidence at low concurrency is negative.
- **CED retrofit.**
  - An architecture change in co2, plus partial re-initialisation.
  - Probably a GDN-to-attention conversion in the upper half.
  - A separate distillation budget on the P40s.
  - Added complexity for chunked x0-flow image/video refinement: refined chunks still pass through both halves, and the decoder reads encoder-derived KV.
  - **The quality risk is moderate and undocumented.** No public ablation exists, and there is no small-model evidence.
- **Cross-layer KV reuse.** Low to moderate cost: share K/V and distill. The risk is lost long-context retrieval, which behaviour checks must look for.
- **FP4 KV QAT.** Low to moderate cost. The risk is long-context recall loss, which behaviour tests must check.

## 6. Evidence quality

- **Strong.** The repo's existence, date, licence, hardware, model scope, feature list and omissions come from the primary README and deployment Q&A, and independent news agrees with them.
- **Medium.** Every performance and quality number is a DeepSeek self-report:
  - 155T tokens/day, $38 per GPU-hour, DSpark +60-85%, scheduler −1.5% to +11%.
  - CED "comparable to baseline", bounded replay "negligible", FP4 KV "marginal".
  - Most of these have no named hardware, baseline or ablation numbers.
  - There are **zero third-party reproductions** as of 2026-10-27.
  - The vLLM recipe is an independent source, but only for serving facts.
- **Weak / ignore.**
  - Secondary explainers: dev.to, Medium, apxml (it lists "552B active parameters", which is wrong), daily.dev, nexforce.
  - Star counts, which differ between sources: 10.5k, 17.9k, 19.7k and 23.5k at different times.

## 7. Bottom line

DeepInfer is real. DeepSeek released it on 2026-10-20 as an engine for one 552B MoE model on Blackwell or Ascend. **It adds nothing to run 0 and nothing usable for Strata**, so per Cyril's "[Cyril quote withheld]" it is not included now.

Three ideas from the model paper behind it go on the list for after run 0. Each must be judged on behaviour:

1. **CED (YOCO-style) half-depth prefill.** It improves phone TTFT and 256k context for prompts with raw media. It needs an upper-half retrofit and has no public ablation.
2. **Cross-layer KV sharing** between full-attention layers. It saves memory at 256k and is a cheap retrofit.
3. **FP4 KV cache with QAT.** It saves phone memory at 256k.

**These are rejected:**
- DSpark speculative decoding. Its gains come from large-batch serving, its scheduler loses 1.5% at low concurrency, and our own speculative-decoding and MTP results on the phone were poor.
- CSA2 sparse attention, which the no-sparse rule excludes.
- Engram hash memory, which is tokenizer-coupled and memory-heavy.
