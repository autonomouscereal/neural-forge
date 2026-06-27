# 2026-06-27 25x Blank-Target Compression Validation

Donor: `Qwen/Qwen3-4B-Base`

Blank target shape/tokenizer: `HuggingFaceTB/SmolLM2-135M`

Folded child size: `162,826,560` parameters, roughly 25x smaller than the donor.

Remote run root: `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627`

## Best Current Checkpoint

`qwen4b_mean_cached_cka_mixed8192_1200s_best`

Pipeline:

1. Training-free donor fold into blank SmolLM2-135M shape.
2. Cross-tokenizer broad recovery with corrected projected-vocab masking and span CKA hidden transfer.
3. High-signal EOS recovery.
4. Mixed 8,192-row recovery using high-signal rows, arithmetic/choice rows, and clean curated 2026 v9 shard rows.

## Comparison

| Checkpoint | PPL mean | Built-in choice | Generation probes | ARC-Easy limit-20 | PIQA limit-20 | Manual notes |
|---|---:|---:|---:|---:|---:|---|
| Random blank SmolLM2-135M | 85894.18 | 0/4 | gibberish | not rerun | not rerun | no useful text |
| Raw 25x fold | 19840.45 | 1/4 | 0/3 | not rerun | not rerun | incoherent |
| Old broad xrecover | 5454.16 | 1/4 | 0/3 | not rerun | not rerun | repetitive |
| Old readable high-signal | 4239.15 | 1/4 | 3/3 | 0.30 | 0.50 | readable pruning/code/merge answers |
| P-KL + high-signal | 1677.10 | 1/4 | 1/3 | 0.40 | 0.50 | lower PPL but manual text degraded |
| CKA + high-signal | 3212.60 | 1/4 | 3/3 | 0.45 | 0.55 | readable, better than old high-signal |
| CKA + mixed8192 | 3104.34 | 4/4 | 3/3 | 0.50 | 0.65 | best automated score; still weak manual generalization |
| CKA + response-auto targeted | 2702.51 | 4/4 | 3/3 | 0.45 | 0.65 | best PPL; manual failures remain, ARC regresses |
| X-tokenizer response+behavior recovery | 2189.62 | 3/4 | 3/3 | 0.45 | 0.60 | better PPL, but behavior only 1/6 and manual text still collapses |
| Cross-tokenizer head calibration | 22103.76 | 2/4 | 2/3 | not run | not run | projected-logit fit improved, but generation degraded badly |
| Omnicoder curated recovery | 1754.09 | 4/4 | 3/3 | 0.40 | 0.60 | best PPL/choice after 25,874-row curated recovery; behavior still 1/6 |
| Cross-tokenizer reconstruction, MLP-only alpha 0.05 | 1648.84 | 4/4 | 3/3 | not run | not run | lower PPL, but behavior still 1/6 and manual output remains repetitive |
| Cross-tokenizer reconstruction, both alpha 0.02 | 1685.81 | 4/4 | 3/3 | not run | not run | safe but no behavior gain |
| Cross-tokenizer MLP-output alpha 0.02 | 1876.39 | 4/4 | 1/3 | not run | not run | output projection reconstruction degraded generation |
| Curated + targeted behavior | 2808.69 | 3/4 | 3/3 | 0.40 | 0.50 | behavior improved to 3/6; arithmetic/code exactness still fail |
| Curated + targeted + EOS | 2865.05 | 2/4 | 2/3 | 0.45 | 0.50 | EOS stops runaway outputs; wrong numeric answers remain |
| Task alpha 0.60, curated->targeted EOS | 2170.67 | 3/4 | 3/3 | 0.40 | 0.45 | blends behavior to 3/6 with less PPL hit, but benchmark regresses |
| Donor Qwen3-4B slice | about 270 PPL in lightweight eval | 4/4 | useful | 0.80 | 0.80 | prior reference run |

All lm-eval values above use `--limit 20`; they are regression signals, not final benchmark claims.

## Manual Inspection

The best checkpoint answers these usefully:

- `Describe neuron merging in one sentence.`
- `Write a tiny Python function that adds two numbers.`
- The built-in pruning and merge-risk generation probes.

It still fails or degrades on:

- `Write a Python function multiply(a, b).`
- `What is 12 + 19? Answer only the number.`
- `A driver sees heavy rain. What should they do?`
- free-form tokenizer mismatch answers become verbose/repetitive after mixed recovery.

The response-only recovery option (`--response-loss-mode auto`) improved PPL from `3104.34` to `2702.51` and kept built-in choice/generation probes green, but it did not fix the manual failures above. Because ARC-Easy limit-20 dropped from `0.50` to `0.45`, the response-auto checkpoint is not the current best overall.

Later recovery ablations showed the same split:

- Curated Omnicoder recovery improved PPL to `1754.09` and preserved `4/4` built-in choice, but did not improve behavior-suite generalization.
- Targeted behavior recovery improved behavior-suite pass rate to `3/6`, but reduced ARC/PIQA and still failed held-out arithmetic (`12 * 13`, `19 + 26`) and exact named-code generation.
- `--append-eos` fixed answer/code runaway continuations, producing concise stops, but it did not fix numeric correctness.
- Same-shape task arithmetic from curated to targeted EOS found a compromise at alpha `0.60`, but it still regressed external benchmarks.
- Focused cross-tokenizer token-span recovery, answer contrastive recovery, and teacher-ranked candidate recovery were added after the response-mask fix. They are useful diagnostics, but they did not solve the 25x blank-target proof:
  - Hard-negative answer contrastive math recovery: PPL `1851.62`, choice `4/4`, behavior `1/6`; held-out `12 * 13 -> 30`, `19 + 26 -> 40`.
  - Mixed broad replay + teacher-ranked candidate recovery: PPL `2120.21`, choice `4/4`, generation probes `3/3`, behavior `0/6`; held-out `12 * 13 -> 4`, `19 + 26 -> 42`.
  - Held-out-clean answer-only arithmetic-format recovery: PPL `2374.74`, choice `4/4`, generation probes `3/3`, behavior `0/6`; held-out `12 * 13 -> 400`, `19 + 26 -> 70`.
  - Task-vector sweep from curated recovery toward the answer-format branch did not expose a useful direction: alpha `0.05` through `0.30` kept choice `4/4` but stayed at behavior `1/6`; alpha `0.45+` dropped behavior to `0/6`.
- Donor candidate scoring showed that `Qwen/Qwen3-4B-Base` ranks the held-out answer strings correctly for the behavior prompts (`156` best for `12 * 13`, `45` best for `19 + 26`). The folded child can rank a whole candidate string correctly while still greedily decoding the wrong first digit, so average sequence-level candidate losses are not sufficient. The next recovery objective needs stepwise/on-policy correction or blockwise activation reconstruction rather than more whole-string ranking.
- Cross-tokenizer block reconstruction added group-aware donor layer selection and unshifted activation span pairing. It made local reconstruction metrics and PPL better in safe settings, but it did not improve behavior:
  - all-layer MLP-only alpha `0.05`, donor layer mode `fold`: PPL `1648.84`, choice `4/4`, generation `3/3`, behavior `1/6`.
  - all-layer MLP-only alpha `0.10`, donor layer mode `fold`: PPL `1580.63`, choice `3/4`, generation `2/3`, behavior `1/6`.
  - all-layer attention+MLP alpha `0.02`, donor layer mode `fold`: PPL `1685.81`, choice `4/4`, generation `3/3`, behavior `1/6`.
  - MLP-output/down-projection reconstruction was not safe at these settings: alpha `0.02` produced PPL `1876.39`, choice `4/4`, generation `1/3`, behavior `1/6`; alpha `0.05` produced PPL `2967.94`, behavior `0/6`.

## Same-Tokenizer Blank Qwen Target Stress Test

To isolate cross-tokenizer effects, a local blank Qwen3-compatible target config was created with the Qwen3 tokenizer retained.

Run root: `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_qwen_tiny_blank_20260627`

Tied-head target:

- Config: hidden size `512`, FFN/intermediate `1536`, `16` layers, `8` attention heads, `2` KV heads, `head_dim=64`, tied embeddings/head.
- Parameter count after tokenizer resize: `125,907,968`, about `31.94x` smaller than the 4B donor.
- Best pre-recovery init among tested variants was still dead. Examples:
  - transport + activation similarity + target RMS scale: PPL `15,574,931`, choice `0/4`, behavior `0/6`.
  - mean fold, no scale: PPL `224,591.59`, choice `0/4`, behavior `0/6`.
  - RMS fold, no scale: PPL `342,369.29`, choice `1/4`, behavior `0/6`.

Untied-head target:

- Config: hidden size `384`, FFN/intermediate `1024`, `16` layers, `6` attention heads, `2` KV heads, `head_dim=64`, untied output head.
- Parameter count after tokenizer resize: `141,662,336`, about `28.39x` smaller than the 4B donor.
- Pre-recovery variants were also dead:
  - mean fold, no scale: PPL `226,830.66`, choice `1/4`, behavior `0/6`.
  - RMS fold, no scale: PPL `179,744.23`, choice `1/4`, behavior `0/6`.
  - mean fold + target RMS scale: PPL `164,514.45`, choice `1/4`, behavior `0/6`.
- Same-tokenizer KD recovery moved the model but did not make it useful:
  - 600 steps on 1,200 calibration rows: PPL `3150.39`, choice `1/4`, behavior `0/6`; manual outputs were short wrong tokens such as `42`, `4.`, `22`.
  - an additional 2,400 full-sequence KD steps on 3,000 rows: PPL `2199.87`, choice `1/4`, behavior `0/6`; manual output remained short/repetitive.
  - cross-tokenizer/same-tokenizer head calibration after that worsened PPL to `4106.65`, choice `2/4`, behavior `0/6`.

Head-dim-preserving untied Qwen target:

- Config: hidden size `384`, FFN/intermediate `1024`, `16` layers, `3` attention heads, `1` KV head, `head_dim=128`, untied output head.
- Parameter count after tokenizer resize: `141,664,384`, about `28.39x` smaller than the 4B donor.
- This isolates head-dimension/RoPE folding by preserving Qwen3's donor head dimension while still compressing head count, width, and depth.
- Tested init variants:
  - representative donor layer, consumer hidden reduction `mean`, no projection scale: PPL `151,838.32`, choice `1/4`, behavior `0/6`.
  - folded donor layer group, consumer hidden reduction `mean`, no projection scale: PPL `122,525.84`, choice `1/4`, behavior `0/6`.
  - representative donor layer, consumer hidden reduction `sqrt-sum`, no projection scale: PPL `386,070.11`, choice `2/4`, behavior `0/6`.
  - representative donor layer, consumer hidden reduction `mean`, target RMS scale: PPL `144,624.83`, choice `0/4`, behavior `0/6`.
- Four-layer `hf-xfamily-reconstruct` on the best head-dim-preserving init (`fold_mean_none`) with `mlp-all`, alpha `0.2`, 1,024 tokens/layer worsened PPL to `178,375.10` and kept behavior at `0/6`.

### 2026 HLD-Inspired Recovery Follow-Up

New implementation added after the first stress test:

- same-tokenizer hidden KD in `hf-recover`, with `projected`, `cka`, and `both` metrics;
- `--hidden-layer all` for proportional all-layer child-to-donor hidden matching;
- `--candidate-train-dtype` so the 141M child can train in `float32` while the 4B teacher stays `float16`;
- non-finite gradient/parameter guards before saving recovery checkpoints.

The dtype guard found a real failure: training the child in `float16` made parameters non-finite after the first AdamW step (`embed_tokens`, attention projections, etc.), which explains earlier degenerate `!` checkpoints. Candidate-side `float32` training fixed that numerical corruption.

Best new same-tokenizer blank-target result:

- Checkpoint: `qwen4b_to_qwen3_tiny_head128_fold_mean_none_hld800_cka_childfp32_noeos`.
- Method: 800 steps, child `float32`, donor `float16`, middle-layer CKA hidden KD plus light token KD/LM, no forced EOS.
- PPL improved from `122,525.84` to `3,259.02`.
- Built-in choice stayed weak at `2/4`; behavior stayed `0/6`.
- Manual decoding still failed: answer-only prompts collapsed to `22`; open generations became empty or repeated periods.
- lm-eval limit-20 slice: ARC-Easy `0.25`, PIQA `0.55`. The raw folded baseline on the same slice was ARC-Easy `0.30`, PIQA `0.50`, so HLD improved PPL but not external-task capability.

Other HLD ablations:

- first-token KD/LM in `float16` corrupted the checkpoint before the finite-parameter guard was added;
- 600-step middle-layer CKA with forced EOS reached PPL `5,904.62`, choice `3/4`, behavior `0/6`, but decoded immediate EOS or fixed short numbers;
- 500-step all-layer proportional CKA reached PPL `5,551.85`, choice `2/4`, behavior `0/6`;
- head-only first-token fine-tuning from the HLD checkpoint regressed PPL to `7,576.06`, choice `1/4`, behavior `0/6`.

### Sparse Top-K KD Follow-Up

New implementation:

- same-tokenizer sparse top-k KD in `hf-recover`, using the teacher's per-token top-k logits;
- `--top-k-kd-weight`, `--top-k-kd-k`, and `--first-token-top-k-kd-weight`;
- unit tests for top-k teacher-subset KL and sparse-only recovery loss.

Validation started from the best finite HLD checkpoint (`hld800_cka_childfp32_noeos`) rather than the raw fold:

- `topk600_last4`: 600 steps, top-k `16`, last-4 layers/head, light LM, first-token top-k. Result: PPL `10,848.12`, choice `3/4`, behavior `0/6`. Manual output shifted from fixed `22`/periods to wrong `44`, `C`, repeated backticks, and repeated `be`; not useful.
- `topk400_last4_unlikelihood`: lower top-k weight plus unlikelihood. Result: PPL `7,465.76`, choice `3/4`, behavior `0/6`. Manual output still failed, with wrong `44`, `C`, repeated periods, and repeated `def`.
- lm-eval limit-20 for `topk400_last4_unlikelihood`: ARC-Easy `0.30`, PIQA `0.50`. Compared with HLD best (`0.25`, `0.55`), this is not a net capability gain.

Sparse KD can sharpen the built-in multiple-choice probe, but it still does not fix the greedy generation/behavior collapse. The current failure is not just "teacher preferred token is missing"; the blank 28x child lacks a stable generative manifold after fold+light recovery.

### Activation-PCA Basis Initializer Follow-Up

New implementation:

- `hf-donor-project-init --hidden-basis-mode activation-pca`;
- `--hidden-basis-scope global` for one donor activation basis shared across hidden axes;
- `--hidden-basis-scope layer` for target-layer input/output boundary bases;
- row-capped donor hidden activation capture and reports for retained PCA energy.

Validation used the same Qwen3-4B donor and same blank 141,664,384-param head-dim-preserving Qwen target.

- Global PCA init (`basispca1024`): retained energy `0.999162`, raw PPL `151,989.41`, choice `1/4`, behavior `0/6`. Manual output repeated punctuation/characters.
- Global PCA + same HLD recovery (`basispca1024_hld800_cka_childfp32_noeos`): PPL `2,730.39`, choice `3/4`, behavior `0/6`. Manual output still failed: punctuation, empty output, or short repeated tokens. lm-eval limit-20: ARC-Easy `0.20`, PIQA `0.50`, worse than the old HLD baseline on this slice.
- Per-layer PCA init (`layerpca512`): raw PPL `474,375.40`, choice `2/4`, behavior `0/6`. Manual output was more diverse than global PCA but incoherent multilingual/token salad.
- Per-layer PCA + same HLD recovery (`layerpca512_hld800_cka_childfp32_noeos`): PPL `3,401.72`, choice `4/4`, behavior `1/6` by answering the addition probe (`19 + 26 -> 45`). Manual open text still failed: repeated phrases, empty code output, and numeric loops. lm-eval limit-20: ARC-Easy `0.25`, PIQA `0.55`, matching the old HLD baseline but not improving it.

This is a useful structural signal but still not a 25x proof. Per-layer residual bases can improve local multiple-choice and one behavior item, but PCA basis projection alone is not enough to create stable generative behavior in a blank 28x child. The next initializer needs activation-aware least-squares/SVD reconstruction of each module output, not only hidden-axis basis projection.

Cross-family/same-tokenizer reconstruction from the strongest per-layer PCA HLD
checkpoint was tested as a post-hoc repair:

- Checkpoint: `qwen4b_to_qwen3_tiny_head128_layerpca512_hld800_recon_all_a005`.
- Method: all-layer ridge reconstruction, alpha `0.05`, 2,048 anchors, 512 tokens
  per layer.
- Result: 16 repairs ran, but PPL regressed to `182,857.94`, choice fell to
  `2/4`, and behavior stayed `0/6`.
- Manual output was not useful; all three generation probes looped on repeated
  phrases such as `important secret`.

This rejects the current post-hoc reconstruction recipe for the blank Qwen
stress test. Local LS repair after a collapsed fold can make the model worse;
the next change must improve the fold map itself before recovery.

### Dense Transport Fold Follow-Up

New implementation:

- added `--transport-backend sinkhorn` as an opt-in balanced Sinkhorn transport
  plan for `fold_mode=transport` when the source-target channel product fits in
  memory;
- kept the existing sparse centroid routing as the default transport backend;
- preserved sparse top-k transport summaries for reports and fallback;
- added unit coverage that checks row mass conservation and balanced target
  load.

This is closer to DOTResize-style neuron merging because donor channels can now
redistribute mass across child channels under a global load constraint instead
of being routed mostly by local clusters.

Validation on Qwen3-4B -> blank 141M Qwen-compatible target did not improve:

- Raw Sinkhorn transport + per-layer PCA init:
  PPL `376,549.34`, choice `1/4`, behavior `0/6`, manual output incoherent.
  The run used 16 dense Sinkhorn MLP transport plans.
- Same HLD recovery as the prior best:
  PPL `3,862.52`, choice `4/4`, behavior `1/6`. Manual output was worse than
  the prior per-layer PCA HLD checkpoint: one prompt looped on `to/of`, and two
  open prompts produced empty continuations.

Sinkhorn transport is therefore useful infrastructure, but it is not the next
promoted 25x initializer. It needs either better cost features or module-level
LS/SVD reconstruction before recovery.

### Uncentered Activation-Basis Follow-Up

New implementation:

- added `--hidden-basis-preprocess centered-rowrms|uncentered-rowrms|uncentered`
  for `hf-donor-project-init`;
- kept the previous centered row-RMS PCA as default;
- added unit coverage showing uncentered row-RMS PCA preserves a shared/common
  activation direction that centered PCA removes.

Validation on the same Qwen3-4B -> blank 141M Qwen-compatible target:

- Raw uncentered row-RMS per-layer PCA init:
  PPL `111,054.17`, choice `1/4`, behavior `0/6`. This is much better than
  centered per-layer PCA raw PPL (`474,375.40`) and slightly better than the
  old mean-fold raw PPL (`122,525.84`), but manual generation still collapsed
  into punctuation and repeated fragments.
- Same HLD recovery as prior best:
  PPL `3,289.83`, choice `2/4`, behavior `0/6`. Manual output was not useful:
  one prompt produced `A. A`, and the code/merge prompts emitted empty
  continuations.

Uncentered residual bases improve raw numeric fit, but they do not convert into
coherent behavior under the current HLD recipe.

This same-tokenizer stress test is important: removing tokenizer mismatch did not make arbitrary 25x blank-target folding work. The current blank Qwen target initializer is not preserving enough residual/backbone geometry for light recovery to restore behavior.

The strongest current evidence is narrower: the SmolLM2-135M blank-target path can transfer partial linguistic/domain behavior and beat a random blank target, but same-tokenizer blank Qwen stress tests show that the general "fold any parent into any blank target at 25x" claim is not yet proven.

### Layerwise Gated Reconstruction Follow-Up

New implementation:

- `hf-xfamily-reconstruct` now supports `--fit-gate validation`, which fits
  activation maps on train rows and mutates weights only when held-out
  activation rows improve without cosine regression.
- It also supports `--layer-gate-metric ppl`, which snapshots each candidate
  layer and reverts that layer if a small held-out PPL probe regresses.
- `hf-tie-head` ties a causal LM output head back to input embeddings and saves
  the smaller checkpoint. This makes post-recovery head compaction reproducible.

Qwen3-4B -> blank 141M Qwen-compatible target:

- Gated `all` reconstruction, alpha `0.02`: only 1 of 16 layer repairs survived
  the PPL gate. PPL improved to `3,063.38`, choice `3/4`, behavior `1/6`, but
  manual output still looped or emitted empty/numeric continuations.
- Gated `attention-output`, alpha `0.005`: 9 of 16 repairs survived. PPL improved
  to `2,396.22`, choice `4/4`, behavior `1/6`, and ARC-Easy limit-20 improved
  to `0.30`, but manual output still failed. PIQA normalized fell to `0.40`.
- A 400-step hidden-KD recovery from that gated checkpoint reduced PPL to
  `2,261.67` but regressed choice to `2/4` and behavior to `0/6`.

Conclusion: layerwise gates prevent catastrophic reconstruction accumulation and
find a better numeric checkpoint, but they still do not create a coherent 28x
blank Qwen child. Local activation fit and PPL gates are necessary but not
sufficient.

### 29.9x SmolLM Blank-Target Evidence

Exact counts:

- Donor `Qwen/Qwen3-4B-Base`: `4,022,468,096` parameters.
- Random blank SmolLM target: `134,515,008` parameters, `29.90x` smaller.
- Best recovered untied SmolLM child:
  `qwen4b_mean_cached_cka_curated_xfamily_token_auto_800s_best`,
  `162,826,560` parameters, `24.70x` smaller.
- Tied-head compaction of that child:
  `qwen4b_mean_cached_cka_curated_xfamily_token_auto_800s_tiedhead`,
  `134,515,008` parameters, `29.90x` smaller.

Random blank target rerun:

- PPL `85,894.18`, choice `0/4`, behavior `0/6`.
- Manual outputs were diverse token salad, not useful text.

Useful untied child (`24.70x`, below the strict 25x gate):

- PPL `2,525.91`, choice `3/4`, behavior `4/6`.
- Manual examples include a valid small add function and a plausible merge-risk
  answer, but arithmetic remains broken and text still repeats.

Tied-head child (`29.90x`, strict gate met):

- PPL `3,845.90`, choice `3/4`, behavior `3/6`.
- Manual output is partially useful: tokenizer mismatch, rain safety, and neuron
  merging prompts produce recognizable answers. The code prompt is malformed,
  and arithmetic remains broken.
- lm-eval limit-20 comparison:
  - random blank: ARC-Easy `0.35`, ARC-Easy norm `0.20`, PIQA `0.60`, PIQA norm `0.50`;
  - tied 29.9x child: ARC-Easy `0.45`, ARC-Easy norm `0.25`, PIQA `0.55`, PIQA norm `0.40`;
  - donor Qwen3-4B: ARC-Easy `0.80`, ARC-Easy norm `0.65`, PIQA `0.80`, PIQA norm `0.80`.

This is the first strict 25x+ blank-target result that is not merely tensor
compatibility. It transfers some useful behavior and beats the random target on
ARC-Easy and behavior probes, but it is still not donor-like and does not meet
the broader arbitrary-target claim.

## Current Conclusion

The 29.9x SmolLM blank-target fold is no longer just fitting tensor shapes. It
transfers enough behavior to beat the random blank by a wide margin, generate
some useful short text, and pass parts of the lightweight behavior suite.

It is not yet a successful replacement for donor capability, and the same-tokenizer Qwen blank-target results show that the utility cannot yet claim arbitrary blank-target 25x compression. The remaining failure mode is structural recovery, not just benchmark tuning: benchmark slices can improve while manual arithmetic, simple code variation, and common-sense instruction following remain unstable or absent.

Next useful work should prioritize representation-preserving fold repair over more one-off choice tuning: DOT-style transport plans, activation-aware least-squares/SVD module reconstruction against donor activations, depth-interface magnitude compensation, and only then numeric/token-aware cross-tokenizer objectives.
