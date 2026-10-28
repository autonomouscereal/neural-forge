# Run 0 literature survey (2026-10-22)

Scope: Cyril's run-0 order. Plant the ~2B Qwen donor into the v2 trunk, ablate it so it can relearn, then distill hard on all modalities. Images and video are allowed to be slow. They should come out in chunks with refinement passes.

Three questions:
1. Plasticity: how to "ablate" a planted model so it can relearn.
2. Recipes for growing a ~2B text LM into an omni model without losing text.
3. Parallel chunk refinement of raw units inside an autoregressive trunk.

Sources are 2025-2026. Older methods are named only as the origin of a technique. Tags:
- **[S]**: stated in the cited source's abstract, proceedings page or paper text as read this session.
- **[2nd]**: reported only by a secondary summary.
- **[E]**: my engineering inference for our setup, not found in any source.

Pass/fail for run 0 stays behaviour-based intelligence retention (Cyril's rule). Nothing below changes that.

---

## 0. Run-0 settings in one table

Every row is **[E]** unless it says otherwise. The sections below give the evidence.

| Knob | Run-0 recommendation | Evidence |
|---|---|---|
| New parts (raw-unit in-projection, shared x0 head, modality/output-decision tokens) | Fresh init. Zero-init the last projection of the x0 head and of any gated/adaLN modulation, so step 0 equals the donor exactly | Show-o2 flow head uses adaLN-Zero [S] |
| "Ablate" the donor trunk | Mild, targeted shrink-and-perturb. Do NOT do a global aggressive reset (see §1.4) | §1 |
| Shrink factor λ (θ ← λθ + ε) | 0.9 on MLP/FFN weights of the middle and upper blocks; 1.0 (untouched) on token embeddings, the LM head, norms and the bottom ~25-30% of blocks | §1.2-1.4 |
| Perturb σ | ~0.05-0.1 × the layer's init std, i.e. ~1e-3 absolute for 0.02-std layers | §1.1 (σ = 1e-3 is the classic setting) |
| Attention heads | Reinit only *collapsed* heads (attention entropy near 0 or near uniform on a probe batch): fresh Q/K/V, zeroed O-proj | §1.3 [2nd] |
| Weight decay | 0.1 floor, consider 0.1-0.2 on trunk matrices; none on norms/embeddings | §1.2 [S] |
| LR | Re-warm (1-2k steps) to a peak well below the donor's pretrain peak (trunk ~1e-4 to 2e-4), then WSD/cosine re-decay. New modules at ~5-10× the trunk LR | §2.1 [S for re-warm/re-decay]; Show-o2 new modules at constant 1e-4 [S] |
| Text share of tokens | ≥ 40% text+tool every step (the ARCH v2 floor stands). Use curated/instructional or KD text, not raw web | §2.2: Show-o2 raw-web co-training collapsed GSM8K on 1.5B [S] |
| Text loss | Sparse teacher-logit KD on top-k ∪ sampled tail (k ≥ 32, 128-256 sampled entries per position), plus CE on gold | §2.4 [S] |
| Loss weights | Per-type-balanced. Start with text KD weight 1.0 and x0-flow 1.0. Ramp flow from 0.2→1.0 over the first ~10% if text drifts; never let media tokens set the gradient scale | §2.3 |
| Packing | Document-masked packing. Block mask: causal across pieces, bidirectional inside each noised media span. Concept bundles (same concept in text + image + audio) in one context | §2.5, §3 |
| Media sampler | Train standard x0-flow (logit-normal t, resolution-shifted). Sample 16-32 joint refinement passes on the server. Distill to 4-8 passes for the phone later (MeanFlow / DMD-style) | §3 |

---

## 1. "[Cyril quote withheld]": warm-start plasticity

### 1.1 Shrink-and-perturb is still the baseline

- **Verwimp, Hacohen, Tuytelaars, "Same accuracy, twice as fast: continuous training surpasses retraining from scratch" (arXiv 2502.21147, Feb 2025).**
  - Abstract [S]: initialisation, regularisation, data selection and hyperparameters each cut warm-start cost. Combined, they reach up to 2.7× compute reduction against scratch, at equal or better accuracy.
  - [2nd]: a search summary reports they use the classic shrink-and-perturb factors λ = 0.4 and σ = 0.001.
  - Caveat: the experiments are vision models (ResNet-18 / CIFAR-100), not LMs.
- **Mallik, Janowski, Hog, Rakotoarison, Grabocka, Hutter, Klein, "When is Warmstarting Effective for Scaling Language Models?" (arXiv 2605.13405, May 2026).**
  - Abstract [S]: keeping the small model's function exact after growth is *not* needed for good final results. Simple growth beats complex warm-start operators, and 2× growth is the most reliable.
  - [S]: gains are largest near ~20 tokens/param and shrink with budget. Past a growth factor bound, scratch training wins.
  - [2nd]: shrink values around 0.2-0.4 worked best *in the growth setting*. That setting is about speed to a scratch-equivalent loss, not about retaining a donor's behaviour.

Takeaway [E]: λ = 0.2-0.4 is a recipe for "use the donor as a better init". It is not a recipe for "keep the donor's intelligence". Our pass criterion is retention, so our shrink must be far milder (λ ≈ 0.9) and targeted.

### 1.2 Weight decay is a plasticity lever

- **"Weight Decay Improves Language Model Plasticity" (arXiv 2602.11137, ICML 2026). Kempner blog by Han, Bordt, Zhang, Kakade, Jul 2026.**
  - [S]: LMs pretrained with higher weight decay adapt better in later fine-tuning and reach higher final performance. The common default of 0.1 is not optimal for adaptability.
  - [S]: lower pretraining validation loss does not predict better post-fine-tune performance.
  - [S]: mechanisms are more linearly separable representations, lower-rank attention, and a smaller train/val gap.
  - Caveat [E]: the study varies WD during *pretraining*. We cannot change the Qwen donor's history. Applying WD ≥ 0.1 during our distill is the transferable part.

### 1.3 Partial / surgical re-initialisation in transformers

- **Hernandez-Garcia, Figliolia, Millidge, "Can Scale Save Us From Plasticity Loss in Large Language Models?" (arXiv 2606.24752, Jun 2026).**
  - [S]: plasticity loss appears in GPT-style transformers from 5M to 314M non-embedding params. Its onset follows a scaling law that grows sublinearly with size. Scale delays it but probably does not prevent it. It appears even in stationary multilingual training.
  - [2nd]: in attention it shows up as more heads that are either overly diffuse or overly concentrated ("lazy/collapsed heads"). Selective Weight Reinitialization (Hernandez-Garcia et al., 2025) is the named remedy.
- **Han et al., "FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff" (arXiv 2602.08040, ICLR 2026 oral).**
  - [S]: picks the reinit point by minimising the squared Frobenius distance to the current weights, subject to the weights being an isometry (Deviation-from-Isometry = 0). Solved with Newton-Schulz.
  - [S]: beats no-intervention and standard resets across domains, including continual LM pretraining with GPT-0.1B on OpenWebText. The abstract gives no LM numbers.
  - [E]: this is the most principled "ablate" for us. Push each linear toward its nearest orthogonal matrix by a fraction, instead of adding random noise. It costs one Newton-Schulz pass per matrix, the same iteration Muon uses. Candidate for the folding run, not required for run 0.
- **Surgical head reinit (aggregator page hyper.ai 2603.09616) [2nd, unverified].**
  - Reinitialises Q/K/V of collapsed heads and zeroes their O-proj. Reported recovery: 242→379 of 384 working heads in BLOOM-1b7.
  - Stated rationale: gradient-only fine-tuning does not revive collapsed heads.
- **Zheng, Cai, Qiu, Ma, "Spurious Forgetting in Continual Learning of Language Models" (arXiv 2501.13453, ICLR 2025).**
  - [S]: much apparent forgetting is lost *task alignment*, not lost knowledge. It is caused by early large orthogonal updates that start in the bottom layers and propagate up.
  - [S]: freezing the bottom layers (including embeddings) gives large gains: ~11%→~44% retained Task-0 accuracy in their slides.
  - [E] for us: do not perturb the bottom ~25-30% of donor blocks or the embedding table, and keep their LR low during the first warmup. The new raw-unit in-projection has to learn to *feed* those layers, not to rewrite them.

### 1.4 What to do in run 0, and what not to do [E]

1. **Global aggressive shrink-and-perturb (λ ≤ 0.5): no.** Every 2025-26 source that recommends it optimises time-to-scratch-loss, not retention. On a 2B donor whose value is its text intelligence, it throws away the thing we are grading.
2. **Mild targeted shrink-and-perturb: yes.** Apply λ = 0.9 and σ ≈ 0.05-0.1 × init std to FFN matrices of the middle and upper blocks only. Leave attention, norms, embeddings, the LM head and the bottom blocks untouched.
   - Rationale: the FFNs hold most of the parameters and are where new modality features must be written. The upper blocks are where output-mode decisions and the x0 head's inputs live.
3. **Reinit collapsed heads only.** Probe ~512 mixed text sequences. Reinit a head when its mean attention entropy is below ~0.1 nats or above 0.95 × log(context).
4. **Do not reset norm gains.** No source supports it for retention, and RMSNorm gains carry the donor's per-channel scaling.
5. **Re-warm the LR** (§2.1) and run WD 0.1 or higher. The 2026 evidence suggests WD does more for plasticity than one-shot noise.
6. **Measure, don't guess.** Log the scorecard drift at step 0 (after ablation) and at the first checkpoint. If step-0 text behaviour is already broken, the ablation was too strong. Undo it from the pre-ablation clone (checkpoint and clone first, per standing rules).

---

## 2. Growing a ~2B text LM into an omni model

### 2.1 LR schedule: re-warm then re-decay

- **"Learning Dynamics in Continual Pre-Training for Large Language Models" (arXiv 2505.07796, ICML 2025).**
  - [S]: the continued-pretraining loss curve splits into two effects: distribution shift, and LR annealing.
  - [S]: they fit a scaling law over peak LR, steps and replay ratio. It predicts loss at any step and schedule, and it is meant for tuning the general-vs-domain balance.
- The standard recipe these papers build on [S, as cited in the same search]: re-warm the LR, re-decay it, and replay old data. Together these match full retraining.
- [E]: use a short re-warm (1-2k steps) to a trunk peak of ~1e-4 to 2e-4 for a 2B. That is lower than Qwen's pretrain peak, which keeps the shift small. Use WSD so we can branch checkpoints for the folding run without restarting decay.

### 2.2 Data mixture and replay: what protects text

- **Show-o2 (Xie et al., arXiv 2506.15564, NeurIPS 2025), Qwen2.5-1.5B and 7B base.** The most relevant single datapoint at our size.
  - [S, Table 13]: one-stage co-training with raw RefinedWeb text **collapsed** language ability. GSM8K fell to ~4.6 on 1.5B.
  - [S]: the two-stage recipe stayed close to the original Qwen2.5-Instruct scores. Stage 1 trains only the new modules. Stage 2 tunes everything on curated understanding/instruction data, which includes text-only instructional data.
  - Lesson [E]: *replay of generic web text is not retention.* The replay has to be the kind of text the model's behaviour is graded on (instructions, reasoning, tools), and ideally teacher-KD'd.
- **Qwen3-Omni Technical Report (arXiv 2509.17765, Sep 2025).**
  - [S]: matched-size text-only, vision-only and omni models show no text degradation. The key ingredient is mixing unimodal and cross-modal data *early* in pretraining.
  - Caveat [S]: no sweep across sizes.
  - [E]: early mixing is from-scratch pretraining at very large scale. It does not transfer directly to a planted 2B. It does support Cyril's "every batch carries every modality".
- **Codec speech LLM on Qwen1.5-0.5B (arXiv 2502.16897, 2025)** [S]: continued pretraining with ~10% text-only replay as a stabiliser, ~15% per speech task. No sweep.
- **VLM practice, per "Deep Pre-Alignment for VLMs" (arXiv 2605.15300, 2026)** [S]: text forgetting is a primary bottleneck. The prevailing fix is the text-only vs multimodal mixing ratio. DeepSeek-VL keeps text above a floor in every stage. The paper argues this is only a trade-off.
- **Forgetting in unified models: "Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models" (MoDE, arXiv 2512.03125, NeurIPS 2025)** [S]: inter-modal forgetting comes from cross-modal *gradient conflict*. Their fix (modality-decoupled experts + KD) is a tower, so it is not allowed for us. The KD half is allowed.
- **Tower-based retention (LMFusion, arXiv 2412.15188, NeurIPS 2025)** [S]: a frozen text path plus separate image modules keeps Llama-3's text ability. It is the cost-of-sharing reference the ARCH v2 A7 ablation measures against. Never quietly adopted.
- **2026 survey of continual learning for speech/audio (arXiv 2605.24863)** [2nd]: keeping text requires frozen backbones, text replay, *and* cross-modal distillation, often together.

Range found: 10-15% text (speech LLMs) up to "a floor in every stage" (VLMs). [E]: our ≥40% text+tool floor (ARCH v2 §S1-S4) is more conservative than the literature. That is correct, because (a) we share every tensor (no towers), so gradient conflict hits the text path directly, and (b) the pass criterion is retention. Keep it for run 0. Lower it only if the scorecard shows headroom.

### 2.3 Loss weighting between text KD and media flow

- **Show-o2** [S]: L = α·L_NTP + L_FM, with α = 0.2 in stage 1 (new modules only) and α = 1.0 in stage 2 (full model).
- Transfusion (2024, origin) up-weighted the diffusion term (λ = 5). Background only.
- [E]: in a fully shared trunk with retention as the gate, do not up-weight flow on the trunk.
  - Use per-type-balanced reduction (ARCH v2 `native_loss_reduction="type_balanced"`).
  - Start at text 1.0 / flow 1.0. If the step-500 scorecard drifts, ramp flow from 0.2 instead.
  - The x0 head's own parameters can take a larger effective LR; that is the cheap way to get Show-o2's "train the new head harder" without pushing media gradients into text weights.

### 2.4 Top-k logit KD

- **Anshumann et al., "Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs" (arXiv 2503.16870, ACL 2025).**
  - [S]: caching only the teacher's **top-K** probabilities is a *biased* estimator. The student learns a scaled-up, mis-calibrated teacher and never sees the tail.
  - [S]: **random importance-sampled** logits are unbiased, preserve the gradient in expectation, and need fewer stored entries. They are competitive with full KD from 300M to 3B, at <10% overhead over CE.
- **Gemma 3 Technical Report (arXiv 2503.19786, 2025)** [S]: samples **256 logits per token, weighted by teacher probability**. The student is trained with cross-entropy on the renormalised sampled distribution; non-sampled entries are zero.
- **Busbridge et al., "Distillation Scaling Laws" (arXiv 2502.08606, 2025).**
  - [S]: student loss follows a broken power law in teacher loss. Past a capacity gap, a stronger teacher stops helping or hurts.
  - [2nd, 2026 on-policy-distillation survey arXiv 2604.00626]: forward KL seems to worsen the gap and reverse KL to soften it.
- [E] for run 0:
  1. Store top-32 to top-64 exact teacher logprobs plus ~128-192 probability-weighted samples from the remainder, with the residual mass as one bucket. This is compatible with Cyril's "top-n (10+) matching is a valid target; top-1 is not".
  2. Use forward KL on the sparse support with temperature 1, plus a small gold-CE term.
  3. Grade with top-10 overlap / behaviour, never top-1.
  4. K3 (the main teacher) is much larger than a 2B, which is the capacity-gap regime. Mixing in the same-vocab mid-size teachers already in the ARCH v2 plan (Q38-27BU / Strata logits) is the literature-consistent hedge.
  5. Exact logit KD needs a shared vocabulary. Planting the Qwen table gives that with Qwen-family teachers. Other teachers need a cross-tokenizer method. Not surveyed here.

### 2.5 Sequence packing and joint concepts

- **Speech-text interleaving, 2025** [2nd, via arXiv 2510.20860-area results in search]: a *deterministic* alternation of speech and text chunks beat stochastic modality sampling. A gradual reduction of the text share eases modality adaptation. Single-source, unverified.
- **BAGEL, "Emerging Properties in Unified Multimodal Pretraining" (arXiv 2505.14683, May 2025)** [S]: trillions of interleaved text/image/video/web tokens. Free-form manipulation, future-frame prediction and 3D/navigation abilities *emerge as interleaved data scales*. It supports Cyril's "train concepts jointly in the same context". BAGEL uses MoT experts, which are towers for us; the data finding is what transfers.
- [E]:
  - Pack with document masking (no attention across packed documents).
  - Inside a document use the ARCH v2 block mask: causal across pieces, bidirectional inside each noised media span.
  - Put the output-decision token at the start of every assistant turn and train it with CE like any other token. Its supervision comes from the data, which is the "understanding, never keyword routing" requirement.

---

## 3. Parallel chunk refinement for raw media inside an AR trunk

This is Cyril's "almost diffusion-based output in chunks, with refinement passes". The 2025-26 literature has converged on exactly this shape.

### 3.1 Whole-image / whole-chunk joint refinement in the trunk

- **Emu3.5 (BAAI, arXiv 2510.26583, Oct 2025): Discrete Diffusion Adaptation (DiDA).**
  - [S]: converts token-by-token image decoding into **bidirectional parallel prediction**. The full image sequence is initialised at once and refined over a few denoising steps.
  - [S]: noisy image tokens attend *causally to prior clean context* and *bidirectionally within the image*. Per-image inference is ~**20× faster** with no quality loss.
  - [S]: trained by self-distillation from the AR model.
  - This is our refinement mask, in discrete form.
- **MAGI-1 (Sand AI, arXiv 2505.13211, May 2025)** [S]: video in **24-frame chunks**. Block-causal attention: full inside a chunk, causal across chunks. Noise increases over time across chunks, so a new chunk starts once the previous one reaches a set noise level, and up to 4 chunks are in flight. Peak inference cost per chunk is constant regardless of length.
  - [E]: this is the clean answer to "video will be slow". Chunk-wise refinement with pipelining gives streaming output.
- **Self Forcing (Huang et al., arXiv 2506.08009, NeurIPS 2025).**
  - [S]: trains the AR video model on its *own* rolled-out context with KV cache, with the loss on the whole video. This removes exposure bias.
  - [S]: uses a few-step diffusion model, stochastic gradient truncation and a rolling KV cache. ~17 FPS real-time streaming on one GPU, quality comparable to non-causal models.
  - [S, arXiv 2602.02214 "Causal Forcing", 2026]: distilling from a *bidirectional* teacher leaves an architecture gap. An AR teacher for ODE init is better.
- **LLaDA2.0-Uni (arXiv 2604.20796, Apr 2026)** [2nd] and **Muddit (arXiv 2505.23606, ICLR 2026)** [S]: discrete-diffusion unified models with block-wise attention and parallel generation across modalities. A further step from our AR trunk. Listed for completeness.

### 3.2 Raw units: predict x0, use big patches

- **Li & He, "Back to Basics: Let Denoising Generative Models Denoise" (JiT, arXiv 2511.13720, CVPR 2026).**
  - [S]: a plain ViT on **raw pixel patches**: no tokenizer, no VAE, no pretraining, no extra loss. Patch **16 and 32** work at 256 and 512 px.
  - [S]: predicting clean data (**x-prediction**) is required in high-dimensional patches. Noise/velocity prediction fails badly there.
  - [S]: a 16×16×3 = 768-dim patch equals the hidden size, so the network has no spare width per patch and x-pred still works.
  - [E] for us:
    - Our raw unit is 32×32×3 = 3,072 values against a ~2,048-wide 2B trunk. That is *more* compressed per unit than JiT-B/16. JiT is the strongest published evidence that x0-prediction on such units is the right target. Keep x0 (ARCH v2 §5).
    - JiT is also direct evidence against "2×2-patch toys": real resolutions with 16-32 px patches are the published regime.
- **Representation Forcing (arXiv 2605.31604, 2026)** [2nd]: a VAE-free unified model. The trunk predicts visual *representation* tokens as intermediates, then does pixel-space diffusion in the same transformer. It matches VAE-based unified models on T2I and beats them on understanding.
  - Caveat: it uses MoT-routed FFNs (towers).
  - [E]: the representation-first idea is close to our REPA term. Without towers, it maps to "draft then refine" (ARCH v2 §5).
- **Per-patch heads (the alternative we rejected, for calibration):**
  - **NextStep-1** (StepFun, arXiv 2508.10711, ICLR 2026) [S]: 14B AR trunk + 157M MLP flow head, one patch at a time.
  - **JanusFlow** (DeepSeek, arXiv 2411.07975, CVPR 2025) [S]: 1.3B LLM with rectified flow inside the LLM framework, but with a 2×2-patchify generation encoder and an SDXL-VAE.
  - Both show per-unit flow heads work at 1-14B. Neither refines the whole image jointly in the trunk.

### 3.3 Audio: patch-level refinement fixes the mean-collapse

- **DiTAR (ByteDance Seed, arXiv 2502.03930, ICML 2025)** [S]: adjacent continuous speech frames are highly similar. So they group P frames into a **patch** and generate each patch with a **bidirectional** local diffusion transformer (LocDiT), conditioned on the causal LM.
  - [S]: temperature is defined as the point on the reverse ODE where noise is injected.
  - [S]: state of the art in zero-shot TTS robustness and similarity, and it scales well.
- **VibeVoice (Microsoft, arXiv 2508.19205, 2025)** [S]: next-token diffusion on continuous σ-VAE speech latents at 7.5 Hz. A Qwen2.5 LLM with a lightweight diffusion head. 90 min of speech in 64K context. The 1.5B model is open.
- [E]: our round-5/6 finding was raw-audio x0 that is level-correct but content-free ("piano" for everything). That is the conditional-mean regression a deterministic per-unit x0 loss produces.
  - Both papers keep **stochastic sampling per patch** and **bidirectional refinement within a patch group**.
  - That argues for (a) refining several consecutive raw audio units jointly, like DiTAR's patch, inside our bidirectional media span, and (b) keeping the STFT/mel terms already required by ARCH v2.
  - It does not argue for a separate audio head.

### 3.4 Step counts and few-step training

- **MeanFlow (Geng, Deng, Bai, Kolter, He, arXiv 2505.13447, NeurIPS 2025 oral)** [S]: models the *average* velocity, linked to the instantaneous one by an identity. No pretraining, distillation or curriculum. 1-NFE FID 3.43 on ImageNet-256 from scratch.
  - [E]: this is the most direct route to a 1-4 pass phone sampler from our x0 head, as a fine-tune after run 0.
- **Show-o2 at 1.5B** [S]: ablations use 25-100 sampling steps with CFG 2.5-10. Gains above CFG ~5 are small. 7.5 CFG at 50 steps appears in the ablation. This is latent space, so raw pixels may need more.
- **Self Forcing** [S]: a few-step model per chunk. **Emu3.5 DiDA** [S]: a few parallel refinement steps over the whole image at ~20× speed-up.
- [E] for run 0:
  - Train ordinary x0-flow with a logit-normal t, shifted toward high noise at larger resolutions.
  - Sample with **16-32 joint passes** on the server for grading. Check whether behaviour, not a number, is acceptable at 8.
  - Plan a MeanFlow-style or DMD/Self-Forcing-style distillation to **4 passes** before any phone target.
  - On a phone each pass is a *prefill-shaped* trunk forward over all units of the chunk. It is compute-bound and parallel, which suits the NPU better than token-by-token decode. "Slow but working" is therefore K × prefill(N units), and K is the lever.

---

## 4. Gaps and conflicts to flag

- **No 2025-26 paper tests shrink-and-perturb magnitudes for *retaining* a planted LM's behaviour.** All the magnitudes above are extrapolated **[E]**. Run 0's step-0 scorecard is the first real data point.
- **Towers vs. sharing.** The strongest text-retention results (LMFusion, MoT in BAGEL and Representation Forcing, MoDE) all isolate parameters by modality. Our no-tower design has to buy retention with data share, KD and LR control. A7's cost-of-sharing measurement stays the honest check; escalate to Cyril if it fails, never quietly add towers.
- **Raw web text replay is actively harmful** at 1.5B (Show-o2 Table 13). Run 0's text share must be curated/instructional plus teacher-KD text from the original parents. Never Claude/Codex outputs, never small-model generations.
- **Unverified items** (aggregator-only): surgical head reinit numbers (hyper.ai 2603.09616), LLaDA2.0-Uni results, deterministic speech-text alternation, and the 2nd-hand shrink values.

## Sources

- Verwimp, Hacohen, Tuytelaars 2025: https://arxiv.org/abs/2502.21147
- Mallik et al. 2026, warmstarting LMs: https://arxiv.org/abs/2605.13405
- Weight Decay Improves LM Plasticity (ICML 2026): https://arxiv.org/abs/2602.11137 ; blog https://kempnerinstitute.harvard.edu/research/deeper-learning/training-language-models-that-can-continue-to-learn/
- Hernandez-Garcia, Figliolia, Millidge 2026: https://arxiv.org/abs/2606.24752
- FIRE (ICLR 2026): https://arxiv.org/abs/2602.08040
- Surgical head reinit (aggregator): https://hyper.ai/en/papers/2603.09616
- Spurious Forgetting (ICLR 2025): https://arxiv.org/abs/2501.13453
- Learning Dynamics in Continual Pre-Training (ICML 2025): https://arxiv.org/abs/2505.07796
- Show-o2 (NeurIPS 2025): https://arxiv.org/abs/2506.15564
- Qwen3-Omni Technical Report: https://arxiv.org/abs/2509.17765
- Codec speech LLM continual pretraining: https://arxiv.org/abs/2502.16897
- Deep Pre-Alignment for VLMs (2026): https://arxiv.org/abs/2605.15300
- MoDE, intra/inter-modal forgetting (NeurIPS 2025): https://arxiv.org/abs/2512.03125
- LMFusion (NeurIPS 2025): https://arxiv.org/abs/2412.15188
- Continual learning for speech/audio survey (2026): https://arxiv.org/abs/2605.24863
- Sparse Logit Sampling (ACL 2025): https://arxiv.org/abs/2503.16870
- Gemma 3 Technical Report: https://arxiv.org/abs/2503.19786
- Distillation Scaling Laws: https://arxiv.org/abs/2502.08606
- On-policy distillation survey (2026): https://arxiv.org/abs/2604.00626
- BAGEL: https://arxiv.org/abs/2505.14683
- Emu3.5: https://arxiv.org/abs/2510.26583
- MAGI-1: https://arxiv.org/abs/2505.13211
- Self Forcing (NeurIPS 2025): https://arxiv.org/abs/2506.08009
- Causal Forcing (2026): https://arxiv.org/abs/2602.02214
- LLaDA2.0-Uni (2026): https://www.alphaxiv.org/abs/2604.20796
- Muddit (ICLR 2026): https://arxiv.org/abs/2505.23606
- JiT, Back to Basics (CVPR 2026): https://arxiv.org/abs/2511.13720
- Representation Forcing (2026, aggregator): https://hyper.ai/en/papers/2605.31604
- NextStep-1 (ICLR 2026): https://arxiv.org/abs/2508.10711
- JanusFlow (CVPR 2025): https://arxiv.org/abs/2411.07975
- DiTAR (ICML 2025): https://arxiv.org/abs/2502.03930
- VibeVoice: https://arxiv.org/abs/2508.19205
- MeanFlow (NeurIPS 2025): https://arxiv.org/abs/2505.13447
