# Tokenizer-free Chimera Omni v2: study of arXiv 2609.12303 and the 2025-2026 field (2026-10-22)

Written for Cyril, who said "[Cyril quote withheld]" and asked us to read arXiv 2609.12303.

Everything here was read-only. Nothing was trained, launched, downloaded into the training tree or changed on a server. The only server action during the study was an SFTP read of the Qwen3.5-4B `tokenizer.json`, used to measure bytes per token.

Speed figures are projections from the ARCH §7 budget model, not phone measurements.

**How to read the citations**
- **Paper:** arXiv 2609.12303. "L" line numbers refer to the local text dump `C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\0c189f21-3de3-4c40-b42c-db54c9d84462\scratchpad\paper2609\paper.txt`, which is a temp folder and may be cleaned. Online copies are at https://arxiv.org/abs/2609.12303 and https://arxiv.org/html/2609.12303. Section and table numbers are the stable reference: the HTML has no line numbers, and the text dump misaligns some table rows (see Verification, C1).
- **ARCH:** `ARCHITECTURE_V2_2026-10-21.md` in this folder.
- **SD:** `distill/STAGE_D_RULINGS_2026-10-12.md`.
- **DRIFT:** `ARCH_DRIFT_FINDINGS_2026-10-22.md`.
- **dNNN:** a verbatim directive row in the `omni_lab.directive` table. That table stops at id 693, so the 10-21 evening rulings are cited from SD L113-120.
- **[derived]:** my arithmetic from cited numbers, not a figure printed in the source.

---

## 0. Short answer

1. **What the paper shows.** A plain decoder that reads and writes raw bytes, with no tokenizer, no patcher and no local encoder or decoder, can be distilled from a token teacher in a single teacher pass. At the paper's own scale it ends up level with an equally distilled token model.
   - Measured, at each method's largest run: 44.6 for the byte model (880B units) vs 44.4 for the token model (220B tokens) on a six-benchmark average. Getting there cost about 4.6x the student training compute [derived] (HTML Appendix C, Table 18; Table 9). "Level" is our reading of a 0.2-point gap with no error bars; the paper does not call it a tie.
   - The headline "+4 points" and "one-sixth the data" come from scaling-law fits extrapolated past the runs they trained (Tables 5-7).
   - The paper is **text-only** (abs).
2. **Rule fit.** For text it is the cleanest no-towers option in the literature. Every published way of making byte models fast adds a patcher, a local encoder/decoder or a lookup table, and those break your rules (§2, §3a).
3. **Phone cost.** About 3.2-4.5 decode steps and KV positions per token's worth of text.
   - The projected 2B byte model gives about 11-16 text tok/s at short context, against 37-48 for today's token plan.
   - The 50k minimum becomes about 160k-230k positions.
   - Sources: §3b; ARCH §7 budget model.
4. **Folding.** The 256 byte rows may exist inside K3's and Qwen's byte-level BPE vocabularies; this is unverified for K3 on disk. What does not fold is "byte sequence → token meaning". Our own July bridge measured exactly that loss: 0/72 multi-piece winners survived (§3c, §4).
5. **Distillation.** Byte space suits your top-n rule. The full teacher distribution fits in 260-261 entries, so it can be matched with no top-n truncation. Teacher top-10 *tokens* can still be scored exactly through End-Of-Token (§3d).
6. **Media.** The paper changes nothing for media. v2 already feeds media as raw 3,072-value units with no tokenizer (ARCH L192-215). The raw-media literature does have concrete leads for the three outputs that never pass (image edit, video out, speech out) (§3e).
7. **Status.** No decision is made here. Section 5 lists the options; section 6 lists the rulings only you can make.

---

## 1. The paper in one page

**What it is**
- Title: "Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models".
- Authors: Marathe, Pagnoni, Limisiewicz, Li, Lewis, Zettlemoyer and Iyer (UW + Meta FAIR).
- Submitted 2026-09-11 under CC BY 4.0 (abs page).
- No code or weights have been released. The only code is App. F.1's `byte_probabilities`, about 21 non-blank lines of NumPy labelled as pseudocode (abs page; HTML App. F.1).

### 1.1 What they built

**Three models**
- All three are Llama-3-style decoder-only transformers with SwiGLU and RMSNorm. They differ only in vocabulary size (§3.1 L310-314; A.1 L1213-1216; §9 L1036-1042).
- None has patching, a latent stage or a local encoder/decoder.

| | Token-1B | Bytes-1B | End-Of-Token-1B |
|---|---|---|---|
| Vocabulary | 128,256 (Llama 3 BPE) | 260 (256 bytes + 4 specials) | 261 (+ `<eot>`) |
| Body (d, FFN, layers, heads) | 2048, 5632, 25, 16 | same | same |
| Layer parameters | 1,284,608,000 | same | same |
| Total parameters | 1,809,946,624 | 1,285,675,008 | 1,285,679,104 |
| Training FLOPs per position | 8,182,345,728 | 8,811,540,480 | 9,440,698,368 |
| Context | 2,048 tokens | 9,216 bytes | 11,264 positions |

Source: Table 9; A.4 L1308-1317.

**What the context lengths mean.** The byte context is the byte string of the same 2,048 tokens, truncated or padded. So the byte models see about the same amount of text, not more (A.4 L1308-1317).

**Training details**
- **Stability.** The first byte runs had loss spikes. Adding qk-norm and changing the sequence lengths fixed them (A.4 L1310-1313).
- **Hardware and batch.** 64 H200s per run, with a global batch of about 0.5M positions (A.4).
- **Teacher and data.** The teacher is Llama 3-8B, with validation BPB 0.85511 (Fig. 3; A.1). The data is the Llama-2 pretraining mix (§3.1; A.2).
- **Loss.** L = α·KL(teacher‖student) + (1−α)·CE, using forward KL (Eqs. 3-4).
  - The α value used is not stated as a number.
  - No temperature is mentioned anywhere.
- **Optimiser.** AdamW with betas 0.9/0.95, cosine schedule, 10% warmup, peak learning rate swept over {1e-3, 4e-3, 8e-3}, weight decay 1e-4, gradient clip 1.0 (A.3).
- **Teacher logit storage.**
  - The token student gets the teacher's top-600 logits.
  - The byte students get the full 260-261 distribution.
  - Both cost about the same storage per unit of text: top-k at 8k+4 bytes per token, against 260 × 4.5 float32 values + 9 bytes. Setting those equal gives k = 585.6, rounded up to 600 (A.6 L1336-1359; Table 10).

### 1.2 The exact byte-logit conversion

The problem: the teacher outputs a distribution over BPE tokens, and the student needs a distribution over the next byte.

**Marginalize-It (approximate; §2.1 L180-201; App. F.2)**
- **First byte of a token.** Sum the teacher's probability over every token that starts with each byte. This step is exact.
- **Later bytes.** Keep only tokens whose spelling matches the true bytes so far, then renormalise.
- **Where it loses probability.** Mass on shorter tokens that end at the current prefix is dropped. In the paper's example, the continuations of "is" are "silently dropped", and isu/isk are renormalised from 0.5/0.125 to 0.8/0.2.
- **Where it is exact.** For first bytes (Case I), and when every alternative token has the same length (Case II).
- **Where it is not.** When alternatives are shorter or longer than the true token (Cases III-IV, Figs 29-30).

**End-Of-Token, or EOT (exact; §2.2 L203-222)**
- **The idea.** Each teacher token becomes its bytes followed by an `<eot>` symbol, so the vocabulary grows by one.
- **Training data.** It is rewritten the same way, with `<eot>` after every BPE token. Example: `[<bos>,<eot>,T,<eot>,i,r,a,m,<eot>,i,s,u,<eot>,<eos>,<eot>]`.
- **The conversion.** At each byte, sum over all tokens matching the prefix. Probability on tokens that stop there goes to `<eot>`. In the example: isu 0.667, isk 0.167, is 0.167, renormalised from 0.75.
- **Why it is exact.** It reproduces the teacher's next-token distribution exactly, given the teacher's true token boundaries, with "no additional inferences over teacher" (L218-219).
- **Cost.** `<eot>` appears about every 4.5 bytes, which adds about 30.94% compute per unit of data (L219-221).
- **Cost against older methods.** Both methods need one teacher forward pass. Earlier exact methods, Hayase 2025 (arXiv 2506.14123) and Phan 2024 (arXiv 2410.09303, background), need several (§1.1 L106-113; §9 L1018-1027).
- **Speed of the conversion itself.** Not reported. The App. F.1 loop runs a bincount over the whole vocabulary at each byte of each token.
- **Open point:** the paper never says how `<eot>` is handled at generation time. It is unclear whether the symbol is stripped from output, and whether prompts must have `<eot>` inserted, which would need a BPE tokenizer at runtime.

### 1.3 Results

**Fitted asymptotes.** These are the scaling-law limits with unlimited compute, LR 4e-3 (Table 2 for BPB, Table 5 for accuracy):

| | Best possible BPB (c) | Best possible accuracy |
|---|---|---|
| Token Supervised | 0.9568 | 46.0 |
| Token Distilled | 0.9407 | 48.4 |
| Bytes Supervised | 0.8967 | 51.2 |
| Marginalize-It Distilled | 0.9016 | 50.5 |
| Bytes w/ `<eot>` Supervised | 0.8891 | 51.6 |
| **End-Of-Token Distilled** | 0.8983 | **52.4** |

What these fits say:
- **Learning-rate sensitivity.** At LR 1e-3 / 8e-3, EOT Distilled reaches 49.7 / 49.8 and Token Distilled 46.6 / 45.7 (Table 15).
- **Distillation adds little to bytes.** In the fits, plain supervised bytes (51.2) already beat distilled tokens (48.4). Distillation adds only about 0.8 points over `<eot>` supervision (Table 5).
- **Compute to cross over.** The training compute an EOT model needs to reach Token Distilled's ceiling is 3.91e22 / **6.33e22** / 2.13e22 FLOPs at the three learning rates. Marginalize-It needs 7.05e22 / 2.04e23 / 9.27e22 (Tables 6 / 16).
- **The second validation set changes this** (Tables 19-23):
  - At LR 1e-3 there is no crossover at all.
  - At LR 4e-3 the crossover moves to 1.69e23 (EOT) and 1.60e25 (Marginalize-It).

**What they measured.** These are the largest runs actually trained, six-benchmark average, LR 4e-3 (Table 18; App. C). "Largest" means each method's own maximum data scale, so the rows are not at equal data. Cite the HTML table (https://arxiv.org/html/2609.12303, App. C, Table 18), not the text dump: paper.txt L1861-1883 puts the averages on the wrong rows.

| Model | Data | Average |
|---|---|---|
| Token Supervised | 500B tokens | 43.0 |
| Token Distilled | 220B tokens | 44.4 |
| Bytes Supervised | 880B bytes | 41.8 |
| Marginalize-It Distilled | 880B | 43.2 |
| Bytes w/ `<eot>` Supervised | 880B | 42.6 |
| **End-Of-Token Distilled** | 880B | **44.6** |
| Llama-3.2-1B / Gemma-3-1B-pt / Gemma 2B | 9T / 2T / 3T | 45.9 / 44.3 / 50.3 |

- **The measured result is level (our reading).** 44.6 vs 44.4 is a 0.2-point gap with no error bars; the paper makes its "better" claim only for the extrapolated asymptotes (§7-8). It cost about 8.3e21 FLOPs for EOT against about 1.8e21 for tokens, roughly 4.6x [derived: 880B × 9.44e9 vs 220B × 8.18e9; 880B read as byte units/positions per App. A.2 and Table 7]. The paper prints no "4.6x".
  - This is student compute only. Table 9's formula covers the student; whether Llama-3-8B teacher forwards are counted is not stated in the parts read.
  - It does not reproduce any compute the paper reports. The paper's own figure for EOT Distilled to match Token Distilled's *asymptote* is 3.91e22 / 6.33e22 / 2.13e22 FLOPs (Table 16).
- **The headline gains compare unlike things.** The "+6.5 / +8.1 / +2.1%" over the open models sets the *predicted* 52.4 against *measured* scores.
- **The crossover is extrapolated.** The paper trains "up to 1 trillion bytes"; at 9.44e9 FLOPs per unit that is about 9.4e21 FLOPs, or about 1.1e22 if EOT units somewhat exceed bytes [derived; the paper prints neither figure]. The 6.33e22 crossover is therefore about 5.7-6.7x beyond the largest EOT run [derived]. Section 7.3 calls these "crossover points"; on validation set 2, LR 8e-3 still crosses for EOT at 3.01e22 (Table 22).

**The "one-sixth the data" and "one-fifth the storage" claims (Table 7 / 17)**
- Data savings are 4.85x (Marginalize-It) and 6.35x (EOT). Storage savings are 4.85x and 5.19x.
- These numbers are the same in every reported cell: all three learning rates on validation set 1 (Table 17), and LR 4e-3 and 8e-3 on validation set 2 (Table 23; LR 1e-3 there has no crossover, shown as dashes). They follow from per-position costs, not from measurements, because the fitted training FLOPs cancel out of the ratio: 9.44 × 5.5 / 8.18 = 6.35, 4.5 × 9.44 / 8.18 = 5.19, and 8.81 × 4.5 / 8.18 = 4.85 for Marginalize-It [derived]. The paper never states that the ratios are invariant. The only measured part, the FLOPs needed to reach Token Distilled's asymptote, decides whether parity is reached at all, not the size of the ratio.
- At equal compute, a byte model reads less text because each byte costs more. That is not a measured gain in learning per example.

**Bits per byte (BPB) misleads.** At about 1.8e21 FLOPs, the four byte models (the Bytes-1B and End-Of-Token-1B variants) have lower validation BPB than Token Supervised, yet do worse downstream "across benchmarks" and are "much worse in most cases" (§4.1; §5.2; §5.4; §6.6). Token Distilled is the one model the paper names as better. App. G.1 / Table 24 illustrates the point with a **hypothetical toy example, not trained models**: imagined models A and B both give each correct token p=0.4 on a 7-token, 20-byte phrase, so both have BPB 0.462 (the text also says 0.463). Greedy decoding gives 7/7 when the correct token ranks first and 0/7 when a 0.5 distractor outranks it. This supports ranking-aware metrics such as your top-n matching and your behaviour-first rule.

### 1.4 Limits

**Stated by the authors**
- "End-Of-Token-1B models remain significantly expensive at inference time." The subject is the EOT-1B models, not byte models in general. A matched-inference-cost comparison (larger Token-1B vs smaller EOT-1B) is "beyond the scope of this study" (§10 "Discussion", "Inference Costs" paragraph; the paper has no section titled "Limitations"). No speed or latency numbers are reported anywhere.
- Text only, dense models only, one size (about 1.28B layer parameters for all three; Table 9), Llama 3-8B teacher (App. A.1), up to about 1T bytes of data, eight benchmarks (§10, §10.1).
- Marginalize-It is "on par or suboptimally" compared with its supervised counterpart (§6.5).
- The fits are sensitive to dropping single points (up to 0.0213 BPB), to fit weighting (up to 0.0376) and to the choice of validation set. All of these effects are larger for bytes than for tokens (App. D.1-D.3, Figs 21-23).

**Not stated by the authors**
- Token Distilled was only run up to 220B tokens, so its fitted plateau rests on short runs (Table 18).
- At LR 8e-3, Token Distilled's ceiling (45.7) sits below Token Supervised's (45.8) (Table 15).
- §6.1 item 2 contradicts the figure captions, probably a typo (readings: paper §6).

---

## 2. The 2025-2026 landscape

### 2.1 Byte language models

**Flat byte transformers (no patcher, no local modules).** These are the only ones that match your no-towers rule.

**2609.12303 (above).**

**2610.05978, "Byte LMs: Scaling, Emergent Abstractions" (2026-10-05).** Source: https://arxiv.org/html/2610.05978
- **Architecture.**
  - A flat transformer from 200M to 3B dense parameters, plus MoE up to about 16.4B total (Tables 5-6).
  - Uses hashed byte n-gram embeddings: 4 tables of 32,768 entries over n = 5-8 (App. A.3).
  - Uses token-superposition training for the first 30% of updates (§2.1).
- **Quality.**
  - Bytes beat subwords at every size on CUTE, OCRBench and TextVQA. At 3B: 99.93 / 42.40 / 55.39 vs 80.83 / 34.70 / 50.76 (Table 1).
  - Subwords are better at equal training FLOPs (Fig. 1).
  - Up to 25% of middle layers can be restricted to local information with no loss (abs).
- **Speed.**
  - Self-speculation was tested in an oracle simulation only: 9.53 accepted units per pass vs 2.83 for subwords.
  - Accepted *bytes* are about equal, 9.527 vs 9.523 (§4.2, Table 3), so there is no text-throughput gain.
  - No wall-clock time is reported.

**EvaByte (Jan 2025, background).** Sources: https://hkunlp.github.io/blog/2025/evabyte/ and https://huggingface.co/evabyte/EvaByte-Phase1
- **Architecture.** 6.5B parameters trained on 1.5T bytes, with 8 multibyte prediction heads and a 32K context.
- **Attention.** EVA attention is chunked and linearised, so it needs checking against d689.
- **Speed.** "5-10x faster" than vanilla byte decoding and "up to 2x faster" than token LMs. Measured on an H800 at batch 1 with HF generate, not split by component.
- **Images.** Fed as JPEG bytes, which is a codec: about 10K bytes per 384×384 image.

**Byte models with patching or local modules.** These all add towers.

| Work | What it adds | Scale | Quality | Speed evidence |
|---|---|---|---|---|
| BLT 2412.09871 (Dec 2024, background) | Entropy patcher (a separate small LM), local encoder/decoder, hash n-grams (§2.3, §3.2-3.3, §4.8) | ≤8B, 4T bytes | Matches token models at scale (abs) | Fewer FLOPs from longer patches |
| Fast BLT 2605.08044 | BLT local decoder (160M) | 1B, 3B | 3B MMLU: BLT 41.15, BLT-D-4 39.07, BLT-D-16 37.08 | Estimates only (>50% less memory bandwidth); no wall clock |
| Scratchpad Patching 2605.09630 | Patcher | — | At 16 bytes/patch, matches or approaches the byte baseline | 16x smaller KV cache, 3-4x less inference compute |
| H-Net 2507.07955 | Mamba-2 encoder/decoder, routing module, dechunker (§2.1-2.2) | ≤1.6B, 100B tokens | XL average 58.2 vs 55.5 for BPE (Table 2) | 0.69 GFLOPs/byte; no wall clock |
| AU-Net 2506.14761 | Byte U-Net with regex word pooling (§2.2) | 1B-8B | 8B HellaSwag 79.1 vs 77.2; trails on GSM8K and MMLU at 1e22 FLOPs (Fig. 3) | Training throughput only |
| MBLM 2502.14553 | Hierarchical decoder stack | — | — | 5M-byte context on one GPU |
| ByteFlow 2603.03583 | Learned compression-based chunking | — | "Beats BPE" (no numbers on the abstract page) | — |

**Two further findings**
- **Diffusion makes bytes worse.** The efficiency gap in byte modelling is worse for masked diffusion than for autoregressive generation ("context fragility") (2605.12928 abs).
- **Proxy Compression** (2602.04289, ICML 2026) trains on compressor outputs plus bytes and runs on flat bytes at inference. On code it matches or beats tokenizer models at larger scale. The abstract page gives no numbers.

### 2.2 Byteification: turning a trained token model into a byte model

**Bolmo (2512.15586).** Source: https://arxiv.org/html/2512.15586v2
- **Architecture.**
  - Adds an mLSTM local encoder, a 4-layer mLSTM decoder and a boundary predictor with 1 byte of lookahead to Olmo 3 (§3.1).
  - Keeps a residual lookup of subword embeddings (§3.1).
- **Cost.** +4.5% parameters at 7B, trained on 9.8B + 39.3B tokens (§3.2).
- **Results at 7B** (Table 1):
  - CUTE 78.6 vs 56.9.
  - GSM8K 68.0 vs 73.1 and MATH 29.8 vs 37.5. The Math category average is 48.9 vs 55.3.
  - HumanEval@1 40.6 vs 49.0.
- **Speed.** About 125 vs 150 bytes/s on GPU. It breaks even at about 6.6 bytes per patch (§6.3).
- **Distillation.** Stage 1 distils against teacher patch likelihoods (τ=5).

**Bao et al. (2602.01007).** Source: https://arxiv.org/html/2602.01007v1
- **Architecture.** H-Net-style Mamba2 encoder/decoder plus a router. The teacher layers and LM head initialise the backbone (§4.1).
- **Teachers.** Llama-3.2-3B and Qwen3-4B, 125B bytes (§5.1).
- **Results.** The average drop against the teacher is 2.80 (Llama) and 2.19 (Qwen). For comparison: BLT 7.34, ALM 4.50, H-Net 9.85.
- **Speed.** Not reported.

**Converting token models to bytes without retraining**
- **Phan 2410.09303 (background).** Zero-shot conversion with an exact byte-token lemma.
- **Hayase 2506.14123.** Byte-at-a-time sampling from any BPE model, using logits only.

**Cross-tokenizer distillation**

| Work | What it does | Result |
|---|---|---|
| ALM 2503.20083 | Aligned-chunk likelihood matching | — |
| BLD 2604.07466 | Byte targets plus a byte head; tested at 1-8B | Gains "not consistent" |
| BPM 2607.22334 | Byte-prefix marginalisation, exact at >99% of positions; teachers include Qwen3-32B and MiniMax-M2.7 | +3.7 to +6.6 avg@8 |
| CSD 2610.28895 | — | Up to +4.2 at Llama-3.1-8B |
| X-Token 2605.21699 | Top-8,192 KL | Snippet only, not opened |

- **Haltiuk 2608.03599** is a position paper with no results. It argues that byte models share one output space, so knowledge can transfer between them exactly.

**The gap.** Every published byteification keeps the teacher body on a patch clock and adds local mLSTM or Mamba modules. **No work byteifies a token model into a flat per-byte transformer** (byte-survey reading §4.5).

### 2.3 Tokenizer-free multimodal and raw-space generation

**No complete example exists.** No 2025-2026 model does text, image, audio and video in and out of one shared trunk with no tokenizer, encoder or codec. The pieces exist separately (raw-media survey reading).

**Raw bytes as the media format**
- EvaByte needed JPEG for images (above). MBLM answers questions about image bytes but reports no generation quality (2502.14553).
- I found no raw-pixel or raw-waveform *byte* LLM at 1B scale or larger in 2025-2026.

**Continuous raw patches with flow matching.** This is what works for output.

- **Prediction target.** The network predicts the clean signal (x-prediction) and the loss is taken on velocity.
  - **JiT** (2511.13720, Table 2a): at 768 values per patch with a v-loss, x-prediction scores FID **8.62**, ε-prediction 372.38 and v-prediction 96.53.
  - With 48-value patches, all nine target/loss combinations work (FID 3.46-6.20, Table 2b).
  - **pMF** (2601.22158, Table 2): at 768 values, x-prediction 9.56 vs velocity-prediction 164.89.
  - Speech agrees. **WavTTS:** x-prediction WER 1.65 / speaker similarity 0.65, vs v-prediction 1.67 / 0.61 (2606.03455, Table 3). **BareWave** predicts the clean waveform with a velocity loss (2606.09048, §3.2).
  - **MFlow** (2610.23854, Table 3) learns the x/velocity mix per direction. It moves FID only 1.82 → 1.81.
- **Patch width can exceed model width.**
  - JiT keeps 16×16 tokens per image at every resolution. Patches are 64px (12,288 values) at 1024px, and FID stays 4.37 / 4.64 / 4.82 (§5).
  - A linear bottleneck down to 16 dimensions does not collapse; bottlenecks of 32-512 *improve* FID (§4.2, Fig. 4).
  - JiT-G (2B) reaches FID 1.82 at 256px (Tables 7-8).
- **Steps.** About 50 is the default everywhere: JiT, WavTTS, BareWave, PixelUMM, HiDream-O1.
  - One-step generation (pMF-H: FID 2.22 at 256px) needs a perceptual loss. Without it, FID rises from 3.53 to 9.56 (pMF Table 2b, Tables 6-7).
- **Training-only auxiliary losses**
  - **WavTTS mel loss.** Speaker similarity is 0.56 at weight 0, 0.65 at 0.05 and 0.55 at 0.5 (Table 3).
  - **BareWave.** A frozen WavLM teacher is used for representation alignment during training only. Similarity rises from 0.471 to 0.610 (Table 3).
  - **LPIPS** helps every pixel baseline (2610.23854, Table 2).
- **Fine detail**
  - JiT-family images miss −6.69 dB in the 0.5-0.7 cycles/pixel band.
  - A residual conv refiner, trained with LPIPS and a GAN loss on a frozen backbone, cuts this to −1.29 dB at the same FID (2610.23854, §4.1, Table 1).
  - Linear output heads leave grid artifacts at high guidance (PixelUMM ablation F3).

**By output mode**

- **Image edit, pixel space.**
  - **Tuna-2** (7B, no vision encoder or VAE) scores ImgEdit 4.09, vs 4.27 for Qwen-Image (2604.24763, Table 4).
  - **HiDream-O1** (8B) scores GEdit 7.60, but uses SigLIP-2 to read condition images (2605.11061, §3.2, Tables 6-7).
  - **No pixel-space edit result exists below about 7B.**
- **Pixel generation inside a unified model.**
  - It collapses without help: GenEval 0.25 vs 0.52 for the VAE version.
  - Predicting DINOv3 tokens first raises it to 0.76, but that needs an understanding encoder. A REPA loss alone reaches only 0.43 (2605.31604, Table 3a).
- **Video out.**
  - **PixelUMM** (8B, no VAE or encoder) generates 96 frames at 512px, VBench 83.24. It routes understanding and generation to separate experts (mixture-of-transformers) and uses 4×16×16 tubelets (2609.38597, Table 9, F2).
  - **UCF-101 at 128px:** pixel JiT-L scores FVD 111 and MFlow-L 97 (2610.23854, Tables 4-5).
- **Speech out with no codec.**
  - **WavTTS** (673M): WER 1.50 / similarity 0.65 / UTMOS 3.92, vs F5-TTS 1.65 / 0.66 / 3.73 (Table 1). Similarity falls to about 0.3 with about 585 hours of data.
  - **BareWave** (984M): 1.75 / 0.602 / 3.72 (Table 2).
  - **No unified LLM outputs raw-waveform speech.** Qwen3.5-Omni still uses Code2Wav (2604.15804).
- **Folding a latent parent into pixel space.**
  - Iris-3B converted FLUX.2 Klein from latent to pixel space (2610.09450).
  - The result trailed both the latent original and a 3B model trained from scratch in pixels.
  - This matters because Qwen-Image 2.1 and LTX are latent parents.
- **Shared weights grow encoder-like layers anyway.**
  - Encoder-free Gemma-4 12B and Inkling-Small grow early-layer "implicit encoders" (2610.24239, abs).
  - 2610.05978 finds up to 25% of middle layers act locally.
  - Shared weights still self-organise this way, so behaviour-first grading should check for it.

**On phones.** I found no 2025-2026 measurement of a byte LLM running on a phone. The closest is Google's ByteQRNN (2022, background).

---

## 3. What it means for Chimera Omni v2

### (a) What "[Cyril quote withheld]" would look like in our architecture

**Starting point**
- Media is already tokenizer-free in v2. One raw unit is 3,072 values in [-1, 1] for image, video, audio and 3D (ARCH L192-215). The music-code, media, sync and tool bands are gone, and ACE codes are not used (readings: our-record §1).
- The only tokenizer left is text:
  - a tied table of 248,320 Qwen rows plus 512 control rows, kept "so the Q38/Strata logit caches apply as-is" (ARCH L160, L245);
  - 509,607,936 parameters in co2-2b (26.5% of 1,923,133,041) and 637,009,920 in co2-4b (14.9% of 4,264,073,793) (ARCH L159-160, L173-174).
- The v2 *proxy* proof already encodes text as raw UTF-8 bytes: `chimera-omni` @d6027f3, `src/chimera_omni/proof/v2_proxy_mixed.py` L16 and L180-185, with `text_vocab=1024` in both `co2-proxy` (`config.py` L837-849) and the round-4 `co2-proxy-media` config (L871-882). Only ids 0-255 of that band carry text. Text scored 1.000 in every arm (ARCH §14.3 L569, §14.5 L678).

**Variants, checked against your directives.** Directive texts are verbatim from omni_lab and SD; the conformance calls come from the critic pass.

| Variant | No tokenizer (d5) | No towers, encoders, adapters, routing (d28, d33, d131) | Every layer, shared (d685) | No text-only tables (SD L118) | Dense, full attention (d93, d689) | On device (d684, SD L116) | Fold first, then distil (SD L119, d116) |
|---|---|---|---|---|---|---|---|
| **V1a.** Bytes as ordinary symbols in the shared trunk; Marginalize-It targets | conforms | conforms | conforms | tension | conforms | tension | tension |
| **V1b.** V1a with End-Of-Token targets | **tension** | conforms | conforms | tension | conforms | tension, worse (5.5 vs 4.5 positions per token) | tension |
| **V2.** Bytes through the shared raw boundary (`NativeContinuousMediaBridge`) | conforms | conforms, unless the boundary is ruled a "bridge" (d29; DRIFT §6 Q4, open) | conforms | probably avoids it (see below) | conforms | tension | tension |
| **V3.** BLT | conforms (narrowly) | **violates**: local encoder/decoder, a separate patcher model (d101) | **violates** | **violates** (hash n-gram table) | conforms | tension | **violates** |
| **V4.** H-Net | conforms | **violates as published**: an inside-trunk chunker is a tension | violates as published | conforms | conforms | better on positions | **violates** as published |
| **V5.** AU-Net | **violates** (the regex word splitter is a pre-tokenizer) | **violates** | **violates** | conforms | conforms | better on positions | **violates** |
| **V6.** Multi-byte heads | n/a (an add-on) | tension (extra text-only output heads) | tension | tension | EvaByte's EVA attention is a tension | the speed claim is not shown here | tension |

**Why V1b's EOT is a tokenizer tension**
- `<eot>` marks K3's or Qwen's BPE boundaries, so the student learns to reproduce a tokenizer's segmentation (§2.2).
- If prompts need `<eot>` at inference, a BPE tokenizer must run on the phone. The paper is silent on this.
- V1a uses tokenizers only inside the *teachers*, which are parents and are allowed tokenizers. Its targets are approximate after the first byte (§2.1; F.2).

**V2 sub-options.** These are my analysis of the bridge (ARCH §4), and none has been tested.
- **V2a: one byte as one scalar.** Reject. RMSNorm mostly removes a single value's magnitude, and it imposes a false ordering ('a' next to 'b').
- **V2b: one-hot in 256 of the 3,072 slots.**
  - Mathematically this is a 256-row embedding, but its rows are columns of the *shared* `feature_proj` that media also trains. That means no text-only table.
  - The cost is interference: those slots are also the first 256 values of every image patch and audio segment.
- **V2c: 12 bytes per unit (12 × 256 one-hots).** This is fixed-stride patching with no patcher, and it cuts positions 12x.
  - **But:** the shared output head is an x0 flow regressor (ARCH L246, L251), not a softmax, so it cannot carry a KL against K3.
  - Twelve bytes from one trunk state would also have no order among them. That job is exactly what MEGABYTE and BLT give a local decoder.
- **The output side is the catch for every V2 option.** A separate 261-row output head brings a text-only table back. Reading logits from a slice of the shared head output is untested.

**Does a 261-row byte table count as a "text-only table"?**
- **Literally, yes**, if only text positions read it, and in V1 they do.
- **What the ruling targets is unclear.** The only item ever flagged under SD L118 is the A9(a) text-keyed PLE (ARCH L461; DRIFT L228-229). Neither 10-22 report applied it to the 248,832-row token table.
- If the token table is allowed, a byte table about 950x smaller (261 vs 248,832 rows; about 320x if the 773-row figure with the 512 shared control rows is used) is allowed too [derived]. If it is not, the byte table is also text-only.
- **This is your call** (§6, Q1).

### (b) Phone speed and context in bytes

**Bytes per token on our own text** (not the paper's 4.5). The critic ran a local `tokenizer.json` (Qwen3.5 family: vocab 248,070, `<|endoftext|>` at 248044; fetched as the Qwen3.5-4B tokenizer, though the file alone cannot confirm the 4B) with `C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\0c189f21-3de3-4c40-b42c-db54c9d84462\scratchpad\bpt_scripts\bpt.py` (UTF-8 bytes; `encode(add_special_tokens=False)`).

| Text | Bytes per Qwen token |
|---|---|
| `ARCHITECTURE_V2` md | 2.78 |
| 91 memory .md files (595 KB) | 3.15 |
| The paper's PDF text | 3.83 |
| `chimera.py` source | 4.50 |

- **The paper's figure.** Its 4.5 is Llama-3 on its pretraining mix (§2.2). 2610.05978 App. A.1 gives 4.56.
- **K3.** It is roughly Qwen's ratio × 1.04. On the 24-row evidence split (`omni-evidence-27b-v1`), K3 makes 92,929 tokens against Qwen's 96,475, with 91,049 aligned boundary pairs: 94.4% of Qwen tokens and 98.0% of K3 tokens (different denominators; 94.3% / 97.1% across all 72 rows) (`composite\corpus\k3_corpus_build_receipt.json` L419-427, `boundary_aggregate.evidence`). The Qwen counts there come from the Qwen3.8-27B tokenizer (qwen3_5 family), not the 4B. This is inferred; K3's tiktoken was not run here.
- **Not yet measured:** Sprite's real chat traffic and non-English text.

**Projected decode speed.** 2B, Q2 body + Q4 head, from the ARCH §7 budget model (L329-348) with the 248,832-row table swapped for a 773-row byte table. Scripts: `C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\0c189f21-3de3-4c40-b42c-db54c9d84462\scratchpad\v2_budget_bytes.py` and `C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\0c189f21-3de3-4c40-b42c-db54c9d84462\scratchpad\bpt_scripts\v2_budget_bytes_ctx.py`; both are labelled "Projection only."
- The byte-model step rates exist only in these scripts; ARCH §7 (L335-341) tabulates the token plan only.
- The model assumes one full model step per byte with the token body unchanged; no patching or multi-byte prediction.
- "50k" in the scripts is 51,200 tokens (230,400 positions at 4.5 B/tok). The token-plan weights print as 0.67 GB against ARCH's 0.72 GB file size.
- Verified on re-run: the short-context row, the 50k row at 4.5 B/tok, and the 256k position count. The 32k row and the 3.15 B/tok cells at 32k and 50k were not re-checked.

| Context (text tokens) | Token plan, tok/s | Byte model, steps/s → text tok/s at 4.5 B/tok | At 3.15 B/tok |
|---|---|---|---|
| Short | 37-48 | 51-70 → **11-16** | 16-22 |
| 32k (current exact phone limit, ARCH L348) | 19-30 | 7-14 at 147k positions → **1.6-3.1** | 3.2-6.0 |
| 50k (SD L116 minimum) | 15-25 | 5-10 at 230k positions → **1.1-2.2** | 2.2-4.1 |
| 256k | 4-8 | 1-2 at 1.18M positions → **0.2-0.4** | — |

**Reading the table**
- **Per step, the byte model is faster,** because the 2B head shrinks from 509.6M to about 1.58M parameters (773 × 2048 [derived]; 773 is not in ARCH). It still loses badly per token of text.
- **Even the best case misses.** A 2B Q1 byte variant gives about 16-23 text tok/s at short context.
- **Reaching about 50 tok/s** would need at least 2.2-3.2 accepted bytes per pass before verify overhead [derived]. Our record says speculative decode "measured as a net loss on 1-bit kernels" (ARCH L247).
- **"50 tok/s" is not a verbatim directive.** It is attributed at ARCH L343. The verbatim rules are d684 "[Cyril quote withheld]" and d19 "[Cyril quote withheld]".
- **d19 does not clearly favour bytes at our budget.** At measured scale the paper's byte model is not smarter, only level (§1.3).

**KV cache and context**
- Our hybrid already caps part of the cost: only the 6 global-attention layers (8 on the 4B) hold KV, and the 18 GDN layers keep a fixed 18 MiB (48 MiB on the 4B) state (ARCH L310-322). The decode-step multiplier is unaffected.
- q8 KV at 50k tokens:
  - at 4.5 B/tok: 1.40 GiB on the 2B and 3.74 GiB on the 4B;
  - at 3.15 B/tok: 0.98 GiB / 2.61 GiB [derived].
- The 4B at 50k bytes exceeds the roughly 3.5 GB single HTP session (ARCH L322-326).
- **256k tokens is about 1.18M positions,** 4.5x past the native 262,144. That would need context-extension retraining, which the token plan does not (ARCH L312).
- 262,144 positions hold only about 58k-83k tokens of text [derived].
- **Prefill.** At 256k positions it already takes 3-11 minutes and hits the 95 °C trip (ARCH L347). A 50k-token byte prompt lands in that regime.
- **Separate issue:** ARCH L348's 32k exact-decode cap already conflicts with the 50k ruling. Bytes make it worse.

**Training compute on P40/3090** [derived, lower bounds that leave out attention-score FLOPs]
- **Per unit of text:**
  - Plain bytes cost 3.3x on the 2B (2.3x at 3.15 B/tok) and 3.8x on the 4B.
  - EOT costs 4.0x and 4.7x.
  - The paper's own figures are 4.85x and 6.35x (Table 7).
- **The training lane shrinks.** The unverified 86M tok/day lane (ARCH L398) becomes about 26M text tokens/day (21M with EOT).
- **Sequence lengths outgrow the model.** 32k-64k-token training sequences (ARCH L407) become 144k-295k positions; 64k tokens at 4.5 B/tok overshoots 262,144.
- **One gain:** the 65 GB logit tensor (ARCH L394) becomes about 68 MB.
- **The paper's crossover is out of reach.** 6.33e22 FLOPs against our about 1e18 FLOPs/day is about 60,000+ days. This is order-of-magnitude only, since the crossover was fitted for a different model and teacher. Below the crossover, the token model leads (abs).

### (c) Folding from token parents

**What folds**
- **The body.** Trunk tensors fold as they do today, but onto a byte clock nobody has tested (below).
- **The 256 byte rows, probably.**
  - K3's tokenizer is tiktoken with 163,584 ranks + 256 reserved (`COMPOSITE_STEP1_RUNBOOK.md` L93), and Qwen's is byte-level BPE with a ByteLevel post-processor (memory `neural-forge-matrix-campaign.md` L1572-1573). Byte-level BPE normally contains all 256 single bytes.
  - **Unverified for K3 on disk.** If they are present, the readings' line "no parent has byte rows" is too strong (critic correction 1).
- **The vocab axis is the cheapest fold axis we have measured.** ARCH.V19 scored 14/15 at 0.7946x (`REALIGNMENT_REPORT_2026-10-22.md` L204-208). Ruling A2 counts vocab/tokenizer reduction as folding (`folding-behavior-first-grading.md` L164).
- **The cliff.** In review, quality fell off a cliff at 0.81 emission coverage (memory `neural-forge-matrix-campaign.md` ~L660-664). Bytes are the far end of that axis: about 773 rows instead of 248,832 [derived: 256 bytes + 512 control + 5 further rows; ARCH does not define this figure].

**What cannot fold, on current evidence**
- **Byte sequence → token meaning.** Nothing we have folds that composition.
  - Our July bridge failed exactly here (§4).
  - The lower layers would have to *learn* the composition, which strains SD L119: "[Cyril quote withheld]".
  - The raw boundary and the x0 head are in the same position today (DRIFT L316-324).
- **The position clock.** The parents' attention learned token-spaced positions, and bytes stretch those distances about 3-4.5x. Untested. The July A0 design deliberately kept the body on the donor token clock (bridge note L34-38).
- **No precedent.** Bolmo and Bao keep the teacher body on a patch clock with added local modules (§2.2). No one has folded or byteified into a flat byte trunk.
- **Already gone anyway.** The T matrix the fold plan was going to build the table from (`V2_FOLD_INIT_PLAN_2026-10-21.md` L147) was discarded on 10-22. Untied→tied rows had median cosine 0.027 (DRIFT L316). A byte table removes that problem rather than creating one.

### (d) Distillation and top-n matching in byte space

**Teacher passes**
- One teacher forward pass per sequence, the same as token KD (L218-219).
- K3 forwards are the real cost either way. The only K3 pricing on record is 17-82 h for 98k tokens, limited by the NAS (`COMPOSITE_STEP1_RUNBOOK.md` S2 row).

**Exact conversion needs the full teacher distribution**
- Our Q38/Strata caches are top-64 + logsumexp, 384 B/token (`CAMPAIGN.md` L487). The tail mass is known but not how it splits across bytes, so they convert only approximately.
- Exact byte targets need a re-dump (as derived views, which d125 and memory `omni-corpus-tokenizer-agnostic.md` L21 allow) or live conversion during the teacher pass.
- A full fp32 byte cell is 4,689 B/token, about 12x our cache: roughly 940 GB vs 77 GB per 200M tokens, half that at fp16 [derived]. The NAS was 98% full on 10-18 (memory `omni-state-2026-10-18.md`).
- Conversion cost: about bytes-per-token × V bincount operations per token, around 0.74M for K3 and 1.1M for Qwen [derived]. Probably CPU-feasible, but unmeasured.

**Two teachers**
- K3 and Qwen disagree at about 2-6% of token boundaries (§3b). In EOT form, each teacher's `<eot>` stream differs.
- Mixing them needs either a per-teacher boundary channel or Marginalize-It, which is approximate after the first byte and "on par or suboptimal" (§6.5).
- What becomes simpler: the cross-tokenizer K3 KD (ARCH L250, L415) turns into same-space byte KD.

**Top-n under your rules**
- **Do not use top-10 over the alphabet.** Over 261 symbols it is nearly trivial and should not be the metric.
- **Score the teacher's top-10 *tokens* instead.** Under EOT, a teacher token's probability equals the product of the student's byte probabilities × P(`<eot>`). So for every teacher top-10 token, rescore its byte path through the student and check its rank. That keeps your "top-10+ token matching" as both target and metric, exactly.
- `<eot>` also gives a unit for "in a row" counts, so `folding_retention.py` (5/8/16/32) can be adapted.
- The KL target itself covers the whole byte distribution, more than top-n.
- **Grading stays behaviour-first.** Our record shows top-k coexisting with dead behaviour (REALIGNMENT L246-250), and the paper's BPB-vs-accuracy split says the same thing (App. G.1).

### (e) What the raw-media literature says about our failing image edit, video out and speech out

**Where we stand.** After rounds 4-7, every modality goes in through the untyped trunk. Raw out passed for music and 3D on both seeds of one round, and for image on one seed. **Image mirror edit (best 0.650), video out (best 0.625) and speech out (best 0.375, typed; untyped ≤0.125) never passed in any arm, typed control included** (ARCH §14.9 L782-800).

**Diagnosis on record**
- Edit outputs ignore the source.
- Video frames are per-row blobs with no coherent content to move.
- Speech requests decode as music.
- Audio x0 MSE sits above the signal variance without the skip (0.31-0.35 vs 0.25), and with the skip the level is right but the class still does not follow the request.
- Sources: ARCH L698, L730, L752-756, L780, L794-800.

**What the literature adds.** Mapping these leads onto our runs is my analysis; none of it has been tested here.

1. **Loss form.**
   - We already predict x0, which matches the literature (ARCH L251-252: "rectified flow predicting x0 and deriving velocity at sampling").
   - We train on **x0-MSE** (ARCH L591, L730). JiT, WavTTS, BareWave, PixelUMM, Tuna-2 and WavFlow all use x-prediction with a **velocity-space loss** (§2.3).
   - That is the same output but a different weighting across noise levels; see JiT 2511.13720 §3 for the exact form, which I did not re-derive here.
   - No source on record has tested our x0-MSE against v-loss. ARCH A7 lists "x0 vs velocity" as a test still to run (L459).
2. **Width is not the bottleneck.**
   - Round 7 moved to JiT's 4:1 ratio (L760) and generation got *worse* (L780).
   - JiT says a patch can be far wider than the model with a bottleneck (§4.2).
   - Both point away from width and toward schedule, data or loss. Our proxy was 9.8M / 72M parameters at 6,000 steps with about 24k examples per behaviour (L792). The smallest working literature models are 673M (WavTTS) to 2B (JiT-G).
3. **Speech out.**
   - WavTTS gets its similarity from a small mel loss (0.56 → 0.65 at weight 0.05; too much hurts) (Table 3). BareWave adds a training-only WavLM alignment.
   - The plan already lists training-only STFT/mel and chunk-boundary losses for audio (ARCH L255), and ARCH L591 says the STFT/mel and boundary losses are "now required before any speech or music generation claim".
   - **Rule check:** a frozen WavLM teacher used only in training is a training loss, not a tower in the shipped model. It is still your call (§6).
4. **Per-patch blobs in image and video.**
   - JiT generates at 16×16 = 256 tokens per image. Our proxy images are 2×2 patches of 32px (L594), so there is little cross-patch structure to learn.
   - PixelUMM found 16px patches beat 32px, and less video compression beats more (F1, F2).
   - Inside unified models, pixel generation collapses without guidance (GenEval 0.25 vs 0.52). The one strong fix uses DINOv3 tokens, which needs an encoder (2605.31604).
   - LPIPS and a residual refiner fix fine detail (2610.23854), but the refiner is a separate stage that may count as a decoder.
5. **Edit.**
   - No pixel-space edit result exists below about 7B. Tuna-2 credits pretrained visual priors (§2.3).
   - Our edit never uses its source, while the route is pinned in tests (L796). That looks like budget or scale, not wiring. Unverified.
6. **Steps on the phone.**
   - About 50 steps is standard. One step needs a perceptual loss (pMF).
   - Our A7 test plans 16 → 4 steps (L459).
7. **Rule conflicts in the best systems.** None of the strongest pixel models meets your rules:
   - PixelUMM and SenseNova-U1 route to separate experts;
   - HiDream-O1 uses SigLIP-2;
   - Representation Forcing uses DINOv3.
   A 2B shared-everything raw design is unproven in the literature.
8. **Bytes do not touch any of this.** Raw *bytes* for media would be worse: EvaByte needed JPEG, which is a codec, and bytes do worse with diffusion (2605.12928).

---

## 4. What our own record already shows

**Byte text already works at proxy scale**
- The v2 proxy proof is the only all-modalities in-and-out proof on record, and it uses raw UTF-8 bytes for text (`v2_proxy_mixed.py` @d6027f3 L16, L180-185). Text scored 1.000 in every arm (ARCH §14.3, §14.5).
- The task is upper-casing held-out strings (ARCH L569), so this proves the plumbing, not language ability.

**The August AV proof had no tokenizer either.** Its text was 4 synthetic answer ids (`h3_hybrid_av_flow.py` @c5a5c18 L63).

**The July token-bridge experiments (LFM2).** Exact spelling transfers; meaning does not.

- **Exact byte mapping works.**
  - All 64,402 donor ids map, with 0 failures and a longest code of 15 rows (`neural-forge\docs\CURRENT_FOLDING_GOAL.md` L170-178).
  - Single-row Unicode decode is lossy for 665 fragment rows, so output must be decoded as a whole sequence (foldrec `reports-july.json` F10).
  - SentencePiece without byte fallback fails closed (`NOTES-2D.md` L377-405).
  - SmolLM2 lacks 21 of the 256 byte atoms, so no exact bridge exists for it (`NOTES-W7A.md` L305-312). That is a reason to verify K3's 256 rows before relying on them.
- **Meaning does not survive.** Source: `neural-forge\research\2026-07-30-lfm2-dual-channel-token-bridge-gap.md`.
  - Composing a token's embedding from its spelling pieces: direct winners survive 293/293, multi-piece winners **0/72**, composition cosine about 0.1209 (L48-56).
  - Sparse codes at k=8/16 give multi-piece top-1 of 1/113 (L100-103).
  - **The input side is the bigger break.** In the four-lane A4 test (residual-OMP, k=16), exact/exact matched 14/16, exact/folded 7, folded/exact 3 and folded/folded 2/16. Folded input lost 11 decisions and folded readout lost 7; mean terminal prompt-hidden cosine under folded input was 0.5528 (L170-188).
  - Assigning exact carrier rows to the review prompts gave 16/16 coherent and 5 useful. That is an upper bound; the same idea from disjoint corpora was rejected (L234-279).
  - At k=32, advisory cached-hidden top-1 was 551/648 and top-10 605/648, but manual review found only 3/16 useful (7 partial, 6 observed misses) (`reports/sparse-semantic-a9-k32-residual-omp-20260730/manual-review.md` L30-33, L71-73). Top-n alone does not prove retention, which is your rule.
- **Lesson for bytes.** Under bytes, nearly every vocabulary entry becomes a multi-piece spelling. The step that failed in July is the step a byte child has to learn.

**Vocab folds**
- ARCH.V19 scored 14/15 at 0.7946x (REALIGNMENT L204-208), with a cliff at 0.81 emission coverage (matrix memory ~L660-664).
- The 27B plan priced a smaller vocabulary as "+15.8% seq-length tax at 256k" (matrix memory L1328). Bytes are the same tax at about 3-4.5x.

**K3 cross-tokenizer KD already aligns at byte boundaries**
- `COMPOSITE_STEP1_RUNBOOK.md` L93 and L120, including a check that K3 detokenizes back to the raw text bytes.
- 22,769 boundary pairs in the held set (`REVIEW_AND_HANDOFF_2026-09-04.md` L78-80).
- The byte-level plumbing for teacher alignment exists.

**Directives that support going byte-level**
- d5: "[Cyril quote withheld]".
- d4: "[Cyril quote withheld]".
- d125: the corpus must not depend on one tokenizer.
- d131: every tensor carries all modalities.
- Ruling A2: a vocab change counts as folding.

**Directives in tension with it**
- d684 and d693: on-device TPS.
- SD L116: 50k minimum on the phone.
- SD L119: fold-only initialisation.
- d19 only clearly favours bytes if they are actually smarter at our budget, and the paper shows a tie at its measured scale.
- The new wish itself is recorded as "a preference, not yet a ruling" (memory `omni-v2-direction-2026-10-21.md`).

---

## 5. Options for the plan (no decision)

Costs are relative to today's plan: a 248,832-row token table and a 2B Q2 projection of 37-48 tok/s at short context.

**Tags**
- **[TEST FIRST]:** a small test is needed before choosing.
- **[RULING]:** your call before anyone builds it.

### O1. Keep the token table (today's plan)
- **What it costs:** nothing new.
- **What it does not do:** meet "[Cyril quote withheld]". It also leaves open whether the 248,832-row table is itself a "text-only table" (SD L118).
- **Risk:** the open question becomes a late surprise.
- **Needs:** [RULING] on SD L118 vs the token table.

### O2. Flat bytes, Marginalize-It targets (V1a)
- **What it gives:**
  - Text tokenizer gone at inference.
  - Text table 248,832 → about 773 rows (about 1.58M parameters on the 2B, 1.98M on the 4B) [derived].
  - Fits d5, d125, d131 and d685.
- **What it costs:**
  - About 3.2-4.5x decode steps per token (short context 11-22 text tok/s on the 2B).
  - The 50k minimum becomes 160k-230k positions.
  - 2.3-3.8x training compute per text token.
  - Teacher caches must be re-dumped or converted live.
- **Risks:**
  - Byte→meaning has no fold source (July 0/72).
  - Approximate targets after the first byte; Marginalize-It underperformed in the paper (§6.5).
  - Parent attention on a byte clock is untested.
- **Unknowns:** bytes per token on Sprite traffic; K3's 256 rows on disk.
- **Needs:** [TEST FIRST] and [RULING] on the 261-row table.

### O3. Flat bytes, End-Of-Token targets (V1b)
- **What it gives over O2:** exact teacher targets, plus exact top-10-token scoring through byte-path rescoring.
- **What it costs:** everything in O2, plus about 31% more positions (5.5 per token) and per-teacher `<eot>` streams that differ by 2-6% between K3 and Qwen.
- **Risk:** inference may need a BPE tokenizer to insert `<eot>` into prompts. That would not be "no tokenizer".
- **Needs:** [TEST FIRST], including whether a model trained with `<eot>` tolerates prompts without it, and [RULING] on whether `<eot>` counts as a tokenizer.

### O4. Bytes through the shared raw boundary (V2b / V2c)
- **What it gives:** byte rows become shared `feature_proj` columns, so there is probably no text-only table. V2c cuts positions 12x with no patcher.
- **What it costs:**
  - A categorical text readout is still needed for KL against K3, and a separate head brings a text table back.
  - V2c has no order among the 12 bytes in a unit.
- **Risks:** interference with media's leading values; text-out through the flow head has no evidence at all.
- **Needs:** [TEST FIRST] (a proxy arm) and [RULING] on whether the bridge is the "bridge" d29 rejects (DRIFT §6 Q4).

### O5. Smaller token vocabulary as a stepping stone (vocab-axis fold)
- **What it gives:** a cheaper table with a measured fold record (ARCH.V19 14/15 at 0.7946x).
- **What it costs:** it is still a tokenizer, so the wish is unmet. Sequence-length tax grows as the vocabulary shrinks (+15.8% at 256k in the 27B plan).
- **Risk:** low; the cliff at 0.81 coverage is known.
- **Needs:** nothing new to test; existing evidence covers it.

### O6. Speed add-ons for O2-O4: multi-byte prediction heads
- **What it gives:** the only route to speed I found that adds no tower. EvaByte claims "5-10x" over plain byte decoding.
- **What it costs and risks:**
  - The heads are extra text-only output heads.
  - Speculative decode measured as a net loss on our 1-bit kernels (ARCH L247).
  - 2610.05978 shows no byte-throughput gain over subword speculation.
- **Middle ground:** as a training-only auxiliary dropped at export it conforms, but then gives no speed.
- **Needs:** [TEST FIRST] (tree-verify on Hexagon) and [RULING] on the extra heads.

### O7. Patching or latent byte models (BLT, H-Net, AU-Net, Bolmo-style)
- **What it gives:** fixes speed and KV (16x smaller KV in Scratchpad Patching).
- **What it costs:** violates d28, d33, d101, d131 and SD L118 as published (§3a). Bolmo also keeps a subword lookup table.
- **Middle ground:** an inside-trunk H-Net-style chunker that reads its boundary score from the shared hidden state is a tension, not a clear violation. Its router and dechunker still have no fold source.
- **Needs:** [RULING] before anyone explores it.

### Separate from bytes: media generation
The literature's levers for edit, video out and speech out are independent of the text decision (§3e):
- v-loss vs x0-MSE, already listed as A7;
- mel/STFT loss for speech, already required by ARCH L591;
- more patches per image at proxy scale;
- a longer schedule;
- LPIPS.

**Small tests that would inform the choice.** All are read or proxy scale; none needs the 2B.
1. **v2 proxy text arm:** plain bytes (already there) vs EOT vs a token table, at the same steps, graded behaviour-first. Covers O2, O3, O5.
2. **Bytes per token** for Qwen and K3 (real tiktoken) on the canonical corpus and on Sprite logs. Covers O2 and O3.
3. **Check K3 on disk** for all 256 single-byte rows (the SmolLM2 trap). Covers O2 and O3.
4. **A13 phone microbenchmark** at 147k and 230k positions (ARCH L451, not yet run). Covers O2, O3, O6.
5. **EOT robustness:** does an EOT-trained proxy decode correctly from prompts without `<eot>`? Covers O3.
6. **V2b interference arm:** byte one-hots inside the shared boundary, measuring media-in/out scores against the current proxy. Covers O4.
7. **Exact conversion speed:** App. F.1's loop over K3's 163,840-row vocabulary for a sample batch on CPU. Covers O3.

---

## 6. Questions for Cyril

1. **Text-only tables.** Does a 261-row byte table (plus the shared 512 control rows) break "no text only tables" (SD L118)? Does today's 248,832-row token table already break it?
2. **`<eot>`.** Does a teacher-BPE boundary symbol inside the student count as "a tokenizer"? This applies especially if prompts would need `<eot>` inserted at inference.
3. **The bridge.** Is the shared `NativeContinuousMediaBridge` the "bridge" you rejected in d29? This decides whether bytes can enter through it (O4). It is DRIFT §6 Q4.
4. **Speed vs smarts.** Taking d19 together with d684, what is the floor? At its largest measured runs the paper's byte model is level with the token model (44.6 vs 44.4, our reading) at about 4.6x student compute, with gains only in extrapolation. A byte 2B projects to about 11-22 text tok/s at short context and 1-4 at 50k. Is there a tok/s number below which a byte model is off the table, whatever its quality?
5. **50k context.** With bytes, the 50k minimum becomes about 160k-230k positions, and 256k tokens exceeds the native 262,144 positions about 4.5x. Should the minimum be stated in text tokens or in positions?
6. **Fold-first.** Byte→meaning composition has no fold source; July measured 0/72. Is it acceptable for that part to come from training while the rest is folded, as the raw boundary and x0 head already do?
7. **Extra heads.** Are multi-byte prediction heads allowed (O6)? Or are they extra text-only output heads, and so out?
8. **Training-only teachers for media.** Are frozen encoders used only as training losses acceptable? Examples are BareWave's WavLM alignment and DINOv3 representation targets. They are gone at inference, but they are encoders.
9. **Tests.** Should any of the small tests in §5 run before the text-representation decision, with the proxy text arm and bytes-per-token on Sprite logs first? Or should the 2B plan stay on tokens for now and the byte option be revisited later?

---

## 7. Verification (2026-10-22)

Twelve load-bearing claims were re-checked against the arXiv HTML (https://arxiv.org/html/2609.12303) and the local sources, read-only. Two scripts were re-run as CPU arithmetic only.

| ID | Claim | Verdict | What changed in this document |
|---|---|---|---|
| C1 | Table 18 measured averages (43.0 / 44.4 / 41.8 / 43.2 / 42.6 / 44.6) | Confirmed | Rows are each method's own maximum data scale, not equal data. "Tie" marked as our reading (0.2 points, no error bars). Cite the HTML table: paper.txt L1861-1883 misaligns the rows. |
| C2 | About 4.6x compute for the measured tie | Confirmed | Marked as student compute only (teacher forwards not stated) and as not matching the paper's own Table 16 compute (3.91e22 / 6.33e22 / 2.13e22). The paper prints no "4.6x". |
| C3 | Headline gains are extrapolated; crossover 6.33e22 beyond the largest run | Confirmed | Largest run restated from "up to 1 trillion bytes": about 9.4e21-1.1e22 FLOPs, so about 5.7-6.7x [derived]. Added validation-set-2 LR 8e-3 crossover (3.01e22). |
| C4 | EOT mechanics, vocab 261, exactness, 30.94%, inference handling unstated | Confirmed | None. |
| C5 | Data/storage savings are arithmetic identities | Confirmed | Scope narrowed: validation set 2 has no LR 1e-3 cells. |
| C6 | BPB vs benchmarks; 7/7 vs 0/7 | Partly | "Every benchmark" and "every byte model" replaced with the paper's "across benchmarks" / "much worse in most cases" for the four byte models. The 7/7 vs 0/7 case is now labelled a hypothetical toy example (App. G.1 / Table 24). |
| C7 | Text-only, one size, no inference numbers, no code | Partly | The "significantly expensive" quote is now attributed to End-Of-Token-1B models, not byte models in general. §10 is "Discussion" ("Inference Costs"), not "Limitations"; the paper.txt line numbers there could not be checked against the HTML. F.1 restated as about 21 non-blank lines of NumPy pseudocode. |
| C8 | Bytes per token 2.78 / 3.15 / 3.83 / 4.50; K3 vs Qwen counts | Confirmed | The K3 receipt's Qwen counts come from the Qwen3.8-27B tokenizer, not the 4B. 94.4% and 98.0% have different denominators, on the 24-row evidence split. The local tokenizer is Qwen3.5-family; the file alone does not prove it is the 4B. |
| C9 | Byte-model speed projections | Confirmed | Added script caveats: byte step rates come from scratchpad scripts only, one full step per byte, "50k" = 51,200 tokens, token weights 0.67 vs 0.72 GB. The 32k row and the 3.15 B/tok cells at 32k/50k were not re-checked. |
| C10 | Text table size and share; byte-table size | Partly | Fixed: 509,607,936 is the 2B table (26.5%); the 4B table is 637,009,920 (14.9%). 773 rows / 1.58M marked as derived and not in ARCH; 4B equivalent 1.98M added. The "960x" ratio corrected to about 950x (261 rows) or about 320x (773 rows) [derived]. |
| C11 | v2 proxy text is raw UTF-8 bytes, text 1.000 in every arm | Confirmed | Round 4 config citation corrected to `co2-proxy-media` (`config.py` L871-882). Only ids 0-255 of the 1024 band are used. |
| C12 | July bridge numbers (64,402 ids; 293/293, 0/72, 0.1209; four-lane 14/16 vs 2/16; A9 k=32) | Confirmed | Line cites tightened (L48-56, L170-188). Four-lane lanes and the 0.5528 cosine added. A9 top-1 551/648 added; 3/16 useful re-read at `manual-review.md` L30. |

**Not covered by this verification.** The §2 landscape figures (2610.05978, EvaByte, BLT, Fast BLT, Bolmo, Bao, BPM, JiT, pMF, WavTTS, BareWave, MFlow, Tuna-2, HiDream-O1, PixelUMM, Iris-3B and the others) and the §3e media diagnosis were not re-checked here. They stand as first-pass readings with their original citations.
