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

### Consumer-Input, Dual-Space, and ALM Follow-Up

New implementation:

- `hf-xfamily-reconstruct --reconstruction-target consumer-input`, which fits a
  cross-tokenizer donor/child hidden map at each layer boundary and absorbs the
  map into the next child layer's attention and MLP input projections.
- `hf-donor-project-init --activation-fit-target`, so activation-fitted linear
  replacements can be restricted to safer module families such as
  `attention-output`, `mlp-output`, or `input-projections`.
- `hf-xfamily-recover --dual-space-kd-weight`, a DSKD-style fixed-projector loss
  that compares student hidden states through the donor head and donor hidden
  states through the child head on aligned span positions.
- `hf-xfamily-recover --alm-kd-weight`, an ALM-style cross-tokenizer chunk
  likelihood loss that matches donor and child negative log likelihood over the
  same character chunks instead of requiring matching token boundaries.

Validation used the strict 29.9x tied SmolLM child as the starting checkpoint.

| Checkpoint | Params | PPL mean | Choice | Behavior | ARC-Easy limit-20 | PIQA limit-20 | Manual notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline tied 29.9x | 134,515,008 | 3845.90 | 3/4 | 3/6 | 0.45 / norm 0.25 | 0.55 / norm 0.40 | partially useful, code/math broken |
| Consumer-input alpha 0.02, gated | 134,515,008 | 3871.39 | 3/4 | 3/6 | not run | not run | stable but slightly worse PPL; loops remain |
| Consumer-input alpha 0.005, strict PPL gate | 134,515,008 | 3845.68 | 3/4 | 3/6 | not run | not run | 13 layer repairs accepted, 17 reverted; neutral |
| Dual-space KD 240 steps | 134,515,008 | 2754.10 | 3/4 | 3/6 | not run | not run | better PPL; manual output still token/numeric loops |
| Dual-space + ALM 240 steps | 134,515,008 | 2721.31 | 3/4 | 4/6 | 0.40 / norm 0.25 | 0.55 / norm 0.40 | behavior regex improves, but manual code/math still fail |

Manual inspection of the ALM+dual checkpoint:

- `tokenizer_mismatch`, `rain_common_sense`, and `neuron_merge_definition`
  remain recognizable but repetitive.
- The `python_add_function` behavior probe passes the regex gate, but the code
  is not syntactically valid.
- Arithmetic remains failed (`12 * 13` and `19 + 26` both devolve into numeric
  loops).
- External lm-eval does not improve over the tied baseline: ARC-Easy raw
  regresses from `0.45` to `0.40`, ARC-Easy normalized is unchanged at `0.25`,
  and PIQA is unchanged at `0.55` / normalized `0.40`.

Conclusion: DSKD-style dual-space recovery and ALM-style chunk likelihood are
useful additions because they improve PPL and one built-in behavior item at
strict 29.9x without increasing parameter count. They still do not meet the
manual quality bar. The next recovery target should be on-policy/stepwise
decoding correction and stronger blockwise reconstruction, not more regex-tuned
short-answer recovery.

### Head-Dim Transport, Weighted Token Init, and On-Policy Follow-Up

New implementation:

- `hf-donor-project-init --token-init-mode donor-tokenized-weighted`, which
  fills target-token rows from donor-tokenized pieces with decoded-length
  weighting instead of flat averaging.
- `hf-donor-project-init --attention-head-dim-transport qk-vo`, which collects
  donor activation signatures for per-head dimensions and shares one transport
  map for Q/K and one for V/O during attention folding.
- `hf-xfamily-recover --on-policy-anchor-kd-weight` and
  `--on-policy-unlikelihood-weight`, which generate short student rollouts
  during recovery and train on those rollouts with shared-anchor KD and
  repeat-token unlikelihood.

Validation again used strict `134,515,008` parameter SmolLM-shaped targets
against the `4,022,468,096` parameter Qwen3-4B donor (`29.90x`).

| Checkpoint | Start | PPL mean | Choice | Behavior | Manual notes |
|---|---|---:|---:|---:|---|
| Weighted token + QK/VO head-dim raw init | blank SmolLM | 72,497,907.23 | 1/4 | 0/6 | repeated `problem`; catastrophic |
| Weighted token only raw init | blank SmolLM | 227,757.94 | 1/4 | 0/6 | repeated `and`; not recoverable enough |
| QK/VO head-dim only raw init | blank SmolLM | 72,468,656.38 | 1/4 | 0/6 | repeated `problem`; head-dim transport is harmful as implemented |
| On-policy recovery, moderate weights | tied 29.9x baseline | 3,561.50 | 3/4 | 0/6 | more diverse than baseline but numeric/code garbage |
| On-policy recovery, low weights | dual+ALM checkpoint | 3,234.21 | 2/4 | 1/6 | one tokenizer item passes; worse than dual+ALM's 4/6 behavior |

Manual inspection:

- Head-dimension transport damaged the raw fold badly. It should stay
  experimental/off by default; the current activation cost does not preserve
  Q/K dot-product geometry well enough.
- Weighted donor-tokenized vocab rows are less destructive than head-dim
  transport, but still do not create a usable raw 29.9x child.
- On-policy rollout unlikelihood reduces single-token loops and raises lexical
  diversity, but it shifts failures into numeric/code-like noise. It does not
  improve the manual bar.

Conclusion update: the next useful path is not more initialization knobs or
rollout unlikelihood tuning. The missing piece is true sequential blockwise
nonlinear reconstruction: train one folded target block at a time against
projected donor block outputs and gate on held-out generation/benchmark signals.

### Blockwise Nonlinear Recovery and OMP Token Init Follow-Up

New implementation:

- `hf-xfamily-block-recover`, which freezes the child, unfreezes one decoder
  block at a time, and trains that block against projected donor block hidden
  states on cross-tokenizer span-aligned positions.
- Optional `--lm-weight` inside blockwise recovery, which adds a child-token LM
  stabilizer during each per-block update.
- Optional per-layer PPL gating and revert for blockwise recovery.
- `hf-donor-project-init --token-init-mode donor-omp`, an experimental sparse
  decoded-token OMP initializer that reconstructs target-token rows from small
  donor surface-feature candidate sets.

Validation again used the strict `134,515,008` parameter tied SmolLM-shaped
child against the `4,022,468,096` parameter Qwen3-4B donor (`29.90x`).

| Checkpoint | Start | PPL mean | Choice | Behavior | Manual notes |
|---|---|---:|---:|---:|---|
| Blockwise 2 layers, no LM stabilizer | tied 29.9x baseline | 4,817.50 | 2/4 | 1/6 | local hidden loss improved, but global text regressed into numeric/code noise |
| Blockwise 2 layers, LM 0.05 | tied 29.9x baseline | 4,546.22 | 3/4 | 2/6 | more topical; code/math still corrupted |
| Blockwise 2 layers, LM 0.15 | tied 29.9x baseline | 4,372.72 | 3/4 | 1/6 | code tokens appear but syntax and numbers remain broken |
| OMP token init, raw tied 29.9x | blank SmolLM | 75,020.37 | 1/4 | 0/6 | initializes 48,951 rows but produces severe token garbage |
| Blockwise all layers, LM 0.05, strict PPL gate 1.02 | tied 29.9x baseline | 3,973.17 | 2/4 | 3/6 | 27/30 block updates accepted, 3 reverted; topical but worse than baseline choice/PPL |
| Blockwise all layers + dual/ALM recovery 120 steps | blockwise all-layer run | 3,356.29 | 2/4 | 2/6 | PPL improves, but behavior and manual code/math regress |

Manual inspection of the best blockwise all-layer run:

- `python_add_function` output contains `pythondef add_numbers(a, b): return a + b`,
  but it appends numeric garbage and is not clean executable code.
- Merge-risk and pruning prompts are topical but repetitive and often miss the
  exact requested concept.
- Arithmetic remains failed and still devolves into numeric loops.

Conclusion update: sequential blockwise recovery is useful instrumentation and
does reduce projected donor/child hidden losses layer by layer, but it does not
yet improve the strict tied baseline behavior. The OMP surface-feature token
initializer is harmful as implemented and should remain experimental. The next
method needs stronger activation transport/basis initialization before
blockwise training, not broader use of this OMP variant.

### Strict 25x Untied-Head FFN Fold Follow-Up

New implementation:

- `hf-donor-project-init --hidden-basis-mode shared-token-ridge`, an
  experimental donor-hidden to child-hidden map fit from shared decoded-token
  embedding rows.
- The mode is intentionally reported with anchor fit quality so weak target
  coordinate maps are visible in the init report.

Validation findings:

- Shared-token ridge is not useful for a blank SmolLM target as implemented.
  Qwen3-4B to blank SmolLM2-135M with tied head produced anchor fit
  `mean_cosine=0.2814`, relative error `0.9591`, PPL `73,517.64` to
  `82,810.18`, choice `1/4` to `2/4`, and manual output collapsed into control
  or blank-like characters. Keep this mode experimental.
- The best useful path is still the recovered untied-head child at
  `162,826,560` params, which is only `24.70x` against the
  `4,022,468,096` parameter Qwen3-4B donor.
- Folding the recovered untied child FFN intermediate size from `1536` to
  `1496` with combined-cosine neuron assignment yields `160,752,960` params,
  or `25.02x`, while preserving some donor behavior.

Strict 25x local eval:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior suite | Manual notes |
|---|---:|---:|---:|---:|---:|---|
| Source recovered untied child | 162,826,560 | 24.70x | 1,648.84 | 4/4 | n/a | useful but repetitive pruning/risk text, malformed code skeleton |
| FFN1496 folded strict child | 160,752,960 | 25.02x | 1,449.60 | 3/4 | 0/6 | topical pruning/risk text and recognizable `def add_numbers` skeleton, still repetitive |
| FFN1496 ridge-repaired child | 160,752,960 | 25.02x | 1,643.74 | 4/4 | 1/6 | choice improves, manual text becomes more repetitive |
| FFN1496 + cross-tokenizer MLP reconstruction alpha 0.05 | 160,752,960 | 25.02x | 1,384.22 | 3/4 | 1/6 | best strict PPL; pruning/risk/code are recognizable but not clean |

External lm-eval slice (`arc_easy`, `piqa`, limit 20):

| Checkpoint | ARC-Easy acc / norm | PIQA acc / norm |
|---|---:|---:|
| Source recovered untied child, 24.70x | 0.35 / 0.20 | 0.65 / 0.50 |
| FFN1496 folded strict child, 25.02x | 0.45 / 0.25 | 0.60 / 0.50 |
| FFN1496 ridge-repaired strict child, 25.02x | 0.30 / 0.25 | 0.60 / 0.60 |
| FFN1496 + MLP reconstruction strict child, 25.02x | 0.45 / 0.25 | 0.55 / 0.50 |
| Qwen3-4B donor | 0.80 / 0.65 | 0.80 / 0.80 |

Manual inspection of the best strict child:

- Pruning prompt is on-topic but repetitive: it discusses channels, weights,
  folding, and training.
- Code prompt emits a recognizable Python fenced block and
  `def add_numbers(a,a): return a + b`, but the rest is malformed repetition.
- Merge-risk prompt mentions risk, merge, redundant rows/channels, and model
  text, but still loops.

Conclusion update: this is the first verified `>=25x` blank-target compression
candidate with nonzero useful behavior after parent folding plus light
reconstruction. It does not meet the final goal. The donor remains far stronger,
arithmetic and clean code generation are not stable, and behavior-suite pass
rate is only `1/6`. The next method should target head-wise/MLP activation
subspace folding and validation-gated reconstruction, not shared-token random
target coordinates.

### Validation-Gated Reconstruction Follow-Up

Code change:

- Fixed `hf-xfamily-reconstruct --layer-gate-metric ppl` rollback for targets
  that mutate downstream consumer layers. `consumer-input` and `all` now
  snapshot and restore `candidate_layer + 1` as well as the current layer when a
  PPL gate rejects a repair. This matters because consumer-input reconstruction
  absorbs maps into the next layer's attention/MLP input projections.

Validation used the same strict `160,752,960` parameter FFN1496 child
(`25.02x` vs Qwen3-4B), starting from
`alpha05_ffn1496_folded` unless noted.

| Checkpoint | PPL mean | Choice | Behavior | Manual notes |
|---|---:|---:|---:|---|
| Gated `mlp-all`, alpha 0.05 | 1,306.59 | 2/4 | 0/6 | best PPL in this sweep, but output collapses into `model`/`knowledge` repetition |
| Gated `attention-all`, alpha 0.02 | 1,398.30 | 3/4 | 1/6 | merge-risk text is more topical but still loops; not better than current best |
| Gated `all`, alpha 0.02 | 1,471.70 | 3/4 | 0/6 | broad repair accepted all layer gates but worsened manual repetition |
| Safer gated `mlp`, alpha 0.02 | 1,407.08 | 3/4 | 1/6 | one layer rejected; useful but weaker than prior alpha 0.05 MLP reconstruction |
| Safer gated `attention`, alpha 0.005 | 1,455.80 | 3/4 | 0/6 | fit gate accepted only 1/30 maps; no meaningful improvement |
| Layer-limited `consumer-input`, alpha 0.005 | 1,449.60 | 3/4 | 0/6 | unchanged; report confirms snapshot examples include `[0,1]`, `[1,2]`, ... |
| Dual/ALM recovery 120 steps from best strict MLP recon | 1,424.08 | 3/4 | 0/6 | recovery shifted output toward `knowledge`/`model` loops |

Negative result:

- Validation and PPL gates prevent some bad repairs, but they do not guarantee
  better greedy generation. `mlp-all` improved PPL to `1,306.59` while behavior
  fell to `0/6`, which reinforces that PPL is only one signal.
- The current best strict checkpoint remains `FFN1496 + cross-tokenizer MLP
  reconstruction alpha 0.05`: PPL `1,384.22`, choice `3/4`, behavior `1/6`,
  with recognizable but malformed pruning/code/merge-risk output.

### Activation-Subspace 29x Blank-Target Follow-Up

Code change:

- Added `hf-donor-project-init --activation-feature-mode subspace` with
  `--activation-subspace-rank`. The activation profile now appends a
  canonicalized per-channel SVD/PCA signature to the prior scalar stats
  (`mean`, `std`, `abs_mean`, positive fraction). This gives transport and
  similarity assignment a richer activation-subspace cost while preserving the
  existing stats mode.
- Fixed the subspace signature tensor to stay on the activation device. The
  first GPU run caught this because CUDA activations were being concatenated
  with a CPU zero signature.

Unit/remote validation:

- Local: `82 passed, 65 skipped`.
- Remote AI lab: `147 passed`.

Method rationale from 2025-2026 research:

- DOT-style resizing and model-folding work points toward activation/weight
  transport before pruning (`arXiv:2507.04517`).
- FLAT-LLM-style compression uses activation low-rank/subspace structure rather
  than raw weight shape alone.
- GRAIL/Free-Lunch-style repair suggests block/module reconstruction after
  folding, not only matrix averaging.
- OMP tokenizer transplantation (`arXiv:2506.06607`) motivates sparse
  embedding/head alignment when tokenizers diverge.

Blank-target experiments run on the AI server:

| Checkpoint | Target | Params | Ratio | PPL mean | Choice | Behavior | Manual notes |
|---|---|---:|---:|---:|---:|---:|---|
| `runs/qwen4b_to_smollm135m_subspace_raw25_20260627_200317/model` | blank SmolLM2-135M config, tied | 134,515,008 | 29.90x | 485,165,195.41 | 2/4 | 0/6 | collapsed into repeated backticks/`scientifically`; not useful |
| `runs/qwen4b_to_smollm135m_subspace_qkvo_pca_omp_raw25_20260627_200753/model` | blank SmolLM2-135M config, tied | 134,515,008 | 29.90x | 77,259.58 | 0/4 | 0/6 | much better PPL from qk/vo head-dim transport + activation PCA + OMP, but text is token salad |
| `.../reconstruct_mlp_a05_anchor4096` | same as above + gated MLP reconstruction | 134,515,008 | 29.90x | 78,310.95 | 0/4 | 0/6 | accepted local fits but worsened PPL/manual output |
| `.../block_recover_s24_lm005` | same as above + block recovery | 134,515,008 | 29.90x | 67,981.89 | 0/4 | 0/6 | block losses and PPL improved, but manual output remained junk |
| `.../block_s24_xrecover120_last4` | same as above + mixed last4 recovery | 134,515,008 | 29.90x | 66,184.77 | 0/4 | 0/6 | recovery loss moved, PPL improved slightly, behavior stayed dead |
| `runs/qwen4b_to_blank_qwen_tiny_subspace_raw25_20260627_202759/model` | blank Qwen3-compatible tied config | 143,777,280 | 27.98x | 198,678.67 | 2/4 | 0/6 | same tokenizer did not fix collapse; repeated tabs/punctuation |
| `runs/qwen4b_to_blank_qwen_tiny_untied_subspace_raw25_20260627_203153/model` | blank Qwen3-compatible untied config | 137,136,768 | 29.33x | 155,361.38 | 1/4 | 0/6 | untied head reduced neither looping nor behavior failure |

Interpretation:

- Subspace activation signatures are useful infrastructure, and the qk/vo
  attention head-dim transport + activation PCA + OMP variant was a real PPL
  improvement over subspace-only (`4.85e8` to `7.73e4`), but it still did not
  produce useful text.
- Same-tokenizer blank Qwen targets also failed, so the current blocker is not
  only tokenizer mismatch or tied heads. The folded internal dynamics are still
  not preserving autoregressive behavior at ~28-30x.
- Gated matrix reconstruction and 24-step-per-layer block recovery improved
  local metrics but not greedy outputs. This confirms the next repair needs a
  stronger behavior-coupled/on-policy objective or deeper blockwise fitting,
  not another PPL-only reconstruction pass.

### Stepwise Donor-Correction Follow-Up

Code change:

- Added `hf-xfamily-recover --teacher-generate-lm-weight`, which asks the donor
  to generate plain-text continuations from calibration prompts and trains the
  folded child on those strings using the child tokenizer. This is a
  tokenizer-agnostic bridge: it does not assume shared vocab rows.
- Added `hf-xfamily-recover --stepwise-donor-kd-weight`, which generates greedy
  child prefixes, decodes each prefix to text, scores that prefix with the
  donor, projects donor next-token logits into the child vocabulary, and trains
  the child at the exact next-token position.
- Added finite gradient/parameter guards to cross-tokenizer recovery and
  blockwise recovery. Same-tokenizer recovery already had these; blank-target
  cross-tokenizer runs should fail fast rather than silently corrupt weights.

Rationale from 2025-2026 research and agent review:

- On-policy/cross-tokenizer distillation work points at per-prefix correction
  rather than offline sequence-averaged KD.
- Byte/chunk-level and ALM-style methods matter because tokenizer mismatch is
  most visible in numerals/code, where prior runs could rank a full candidate
  answer correctly while still decoding the wrong first token.
- Chain-based distillation and decode-aware calibration suggest the single-step
  `4B -> 135M` jump is probably too abrupt without intermediate anchors, but
  stepwise donor correction is still useful infrastructure for testing that
  hypothesis.

Strict 25x validation on the current best blank-target child:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior | lm-eval ARC-Easy | lm-eval PIQA | Manual notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Prior best `alpha05_ffn1496_folded_xreconstruct_mlp_a05` | 160,752,960 | 25.02x | 1,384.22 | 3/4 | 1/6 | 0.45 / 0.25 | 0.55 / 0.50 | topical but malformed; recognizable add-function skeleton |
| Stepwise + teacher-LM, 24 steps | 160,752,960 | 25.02x | 1,648.56 | 2/4 | 0/6 | not promoted | not promoted | projected stepwise loss decreased, but output regressed into repetition/noise |
| Teacher-generated LM only, 40 steps | 160,752,960 | 25.02x | 1,502.17 | 2/4 | 2/6 | 0.40 / 0.20 | 0.60 / 0.50 | slightly better regex behavior and PIQA raw, but code still wrong and repetition remains heavy |

Interpretation:

- The new losses are mechanically active: stepwise donor counts were nonzero
  and losses decreased during recovery.
- The projected-vocab stepwise loss is too blurry or too aggressively weighted
  in the tested schedule; it worsened PPL, choice, behavior, and manual output.
- Donor-generated plain-text LM recovery is less destructive and may be useful
  as a staged warmup, but the 40-step ablation still does not beat the prior
  strict best overall. It improves behavior regex to `2/6` and PIQA raw to
  `0.60`, while losing ARC-Easy, choice, and PPL.
- No result in this sweep meets the user's 25x useful-behavior bar. The best
  strict checkpoint remains the prior FFN1496 + MLP reconstruction run.

### Staged Decode-Recovery and Behavior-First Ranking

Code change:

- Added `hf-rank-evals`, which ranks existing `hf-eval` reports by
  behavior-suite pass rate first, then choice accuracy, generation health,
  smaller parameter count, and finally PPL. This is meant to prevent PPL-only
  improvements from being promoted during high-ratio compression sweeps.

Strict FFN1496 staged recovery from the prior best
`alpha05_ffn1496_folded_xreconstruct_mlp_a05`:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior | Manual notes |
|---|---:|---:|---:|---:|---:|---|
| Prior best | 160,752,960 | 25.02x | 1,384.22 | 3/4 | 1/6 | topical but malformed; still current strict quality baseline |
| Stage 1 head-norm donor-text warmup | 160,752,960 | 25.02x | 1,370.65 | 2/4 | 0/6 | best PPL, but behavior collapses; rejected |
| Stage 2 low-LR all-layer donor-text recovery | 160,752,960 | 25.02x | 1,435.54 | 3/4 | 1/6 | does not beat prior best; code still malformed |
| Stage 3 last-4 ALM/anchor polish | 160,752,960 | 25.02x | 1,427.29 | 2/4 | 1/6 | no useful manual improvement |

Strict rank file:
`runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250/rank_strict_staged_decode.json`.

Behavior-first ranking placed `alpha05_ffn1496_teacher_lm_only_s40` first
because it has `2/6` behavior, but manual inspection still rejects it: it has
wrong code arguments and heavy repetition. This is a useful reminder that the
ranker is a promotion filter, not a replacement for manual inspection.

29.9x dual-space+ALM decode polish:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior | Manual notes |
|---|---:|---:|---:|---:|---:|---|
| `tied29x_dualalm240_last4` baseline | 134,515,008 | 29.90x | 2,721.31 | 3/4 | 4/6 | best strict behavior branch, but code/math malformed |
| Head-norm donor-text polish | 134,515,008 | 29.90x | 2,713.93 | 2/4 | 4/6 | slightly lower PPL, worse choice; manual still malformed |
| Last-4 donor-text/ALM polish | 134,515,008 | 29.90x | 2,729.68 | 2/4 | 4/6 | no improvement over baseline |

29.9x rank file:
`runs/qwen4b_to_smollm135m_dualspace_alm_20260627/rank_dualalm_decode_polish.json`.

The ranker correctly kept the original dual+ALM checkpoint first because the
polish variants preserved regex behavior but lost choice accuracy. Manual output
also did not improve.

### Blockwise Stepwise Donor-KD Follow-Up

Code change:

- Added optional stepwise donor next-token KD to
  `hf-xfamily-block-recover`. Unlike `hf-xfamily-recover`, this keeps only the
  current child block trainable, so the donor prefix-correction signal is used
  during block/interface repair instead of full-model recovery.
- New flags:
  `--stepwise-donor-kd-weight`, `--stepwise-donor-max-new-tokens`,
  `--stepwise-donor-interval`, and `--projected-vocab-max-source-tokens`.
- The blockwise skip condition now permits stepwise-only block updates even
  when span alignment finds no activation pairs.

Strict FFN1496 validation:

- Starting checkpoint:
  `alpha05_ffn1496_folded_xreconstruct_mlp_a05`, `160,752,960` params,
  `25.02x`, PPL `1,384.22`, choice `3/4`, behavior `1/6`.
- Run:
  `runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250/block_stepwise_first4_s4`.
- Settings: first 4 child blocks, 4 steps/layer, `lm_weight=0.03`,
  `hidden_metric=both`, `stepwise_donor_kd_weight=0.05`,
  `stepwise_donor_max_new_tokens=4`, PPL layer gate `1.05`.
- Report: 4/4 layer updates accepted, 0 rejected; 12 logged updates had
  nonzero stepwise donor counts, max count `4`.
- Eval: PPL `1,383.58`, choice `2/4`, behavior `1/6`, ratio `25.02x`.
- Rank artifact:
  `runs/qwen4b_to_smollm135m_strict25_ffnfold_20260627_191250/rank_block_stepwise_first4_s4.json`.

Interpretation:

- The block-local stepwise signal is mechanically active and did not blow up.
- It is not a useful improvement at this setting. PPL improved by less than
  `1`, choice accuracy regressed, behavior did not improve, and manual outputs
  remained repetitive/malformed.
- This supports the current working conclusion: decode-aware losses are useful
  infrastructure, but the next meaningful jump probably needs stronger
  block/interface initialization before recovery, not another small loss
  coefficient on the current weak folded manifold.

### RSD Hidden Recovery and Behavior-First Checkpointing

Code change:

- Added opt-in RSD-style hidden recovery metrics:
  `--hidden-kd-metric rsd` and `--hidden-kd-metric projected-rsd`.
- `rsd` combines dimension-free CKA invariance with a student hidden-feature
  decorrelation penalty, so it can run across different hidden widths without a
  temporary projection.
- `projected-rsd` uses the existing donor-hidden to child-hidden projection for
  invariance and adds the same decorrelation penalty.
- Added behavior-first checkpoint selection for `hf-recover`,
  `hf-lm-recover`, and `hf-xfamily-recover`:
  `--save-best-behavior`, `--behavior-eval-interval`,
  `--behavior-max-new-tokens`, and `--behavior-max-probes`.
  This selects checkpoints by held-out behavior probes first, using loss only as
  a tie-breaker.

Strict 29.9x validation:

- Starting checkpoint:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_mean_cached_cka_curated_xfamily_token_auto_800s_tiedhead`.
- Parameters: `134,515,008`, ratio `29.90x` against
  `Qwen/Qwen3-4B-Base`.
- RSD behavior-best run:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_tied29_rsd_behaviorbest_60s`.
- Settings: `60` steps, `train_scope=last4`, `hidden_kd_metric=rsd`,
  `hidden_kd_weight=0.25`, `span_hidden_kd_weight=0.15`,
  `anchor_kd_weight=0.08`, `lm_weight=0.55`, behavior eval every `20`
  steps over all `6` behavior probes.
- Behavior checkpoint history: steps `1`, `20`, `40`, and `60` all scored
  `3/6`; the selector chose step `1` because behavior tied and loss was lowest.
- Eval comparison:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior | Gate | Manual notes |
|---|---:|---:|---:|---:|---:|---|---|
| tied 29.9x baseline | 134,515,008 | 29.90x | 3,845.90 | 3/4 | 3/6 | fail | partially useful but code/math malformed |
| RSD behavior-best 60s | 134,515,008 | 29.90x | 3,825.12 | 3/4 | 3/6 | fail | slightly lower PPL, no manual generation improvement |

Gate failure reason:

- `perplexity_ratio`, `choice_accuracy_floor`, `choice_accuracy_drop`,
  `behavior_suite`, and `behavior_suite_drop` passed.
- `generation_health` failed because the code-generation probe still lacks a
  valid function definition and continuations remain malformed/repetitive.

Interpretation:

- The RSD path is mechanically valid and does not require shared hidden width.
- Behavior-first checkpointing works and prevents loss-only selection from
  silently choosing a lower-loss but worse-behavior checkpoint.
- This RSD setting is not a promoted model result: it preserved behavior and
  slightly improved PPL, but manual outputs did not improve and the gate still
  failed.

### Executable Code Behavior Gate Follow-Up

Code change:

- Replaced the regex-only `python_add_function` behavior probe with a
  constrained AST scorer.
- The scorer requires a named `add` function or `add = lambda ...`, validates a
  small expression whitelist, and evaluates fixed add-function tests without
  executing arbitrary generated code.
- This closes a false-positive path where malformed text containing `def`,
  `return`, and `+` could pass the behavior suite.

Strict 29.9x re-eval with the executable code scorer:

| Checkpoint | Params | Ratio | PPL mean | Choice | Behavior | Code probe | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| tied 29.9x baseline, strict-code eval | 134,515,008 | 29.90x | 3,844.96 | 3/4 | 3/6 | fail: syntax error | fail |
| RSD behavior-best 60s, strict-code eval | 134,515,008 | 29.90x | 3,825.12 | 3/4 | 3/6 | fail: syntax error | fail |

Artifacts:

- `runs/qwen4b_to_smollm135m_transport_random_20260627/eval_best_token_auto_800_tiedhead_strictcode.json`
- `runs/qwen4b_to_smollm135m_transport_random_20260627/eval_qwen4b_tied29_rsd_behaviorbest_60s_strictcode.json`
- `runs/qwen4b_to_smollm135m_transport_random_20260627/gate_qwen4b_tied29_rsd_behaviorbest_60s_strictcode.json`
- `runs/qwen4b_to_smollm135m_transport_random_20260627/rank_qwen4b_tied29_rsd_behaviorbest_60s_strictcode.json`

Interpretation:

- The stricter code probe agrees with manual inspection: both checkpoints emit
  malformed Python-like text, not usable code.
- The headline behavior count stayed at `3/6` for these two checkpoints because
  their previous passing items were tokenizer mismatch, rain common-sense, and
  neuron merging; the code item was already failing for them.
- This change matters for future branches such as dual+ALM where regex-based
  code checks could overstate behavior.

### Behavior-First Gate Audit And Strict Replay Attempts

Code changes:

- Added `ppl_gate_mode` to `gate_text_report`: `hard` preserves old behavior,
  while `advisory` records PPL failure without letting PPL alone reject a
  behavior-good candidate.
- `hf-gate` exposes this as `--ppl-gate-mode hard|advisory`; CLI default is
  `advisory`.
- Automated compression candidate sorting now ranks passed candidates by
  behavior pass rate, choice accuracy, generation health, shrink, and only then
  PPL.
- Cross-family usable candidate sorting now uses the same behavior-first policy.
- Blockwise cross-tokenizer recovery now supports `--layer-gate-metric behavior`
  with held-out behavior snapshots and per-block revert. PPL-only block gates
  remain available for telemetry/safety experiments.

Strict 29.9x behavior replay validation:

| Checkpoint | Start | Params | Ratio | PPL mean | Choice | Behavior | Generation health | Rank outcome | Manual notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| RSD behavior-best 60s | tied 29.9x | 134,515,008 | 29.90x | 3,825.12 | 3/4 | 3/6 | 2/3 | best retained | code/math malformed but best overall behavior/choice/generation mix |
| Strict replay `last2` 120s | RSD behavior-best | 134,515,008 | 29.90x | 3,859.08 | 2/4 | 3/6 | 0/3 | rejected | output quality degraded; code still malformed |
| Strict replay `head_norm` 160s | RSD behavior-best | 134,515,008 | 29.90x | 4,239.07 | 3/4 | 3/6 | 0/3 | rejected | math prefixes moved closer but collapsed into digit loops; code still malformed |
| Strict replay anti-repetition `head_norm` 120s | RSD behavior-best | 134,515,008 | 29.90x | 4,111.44 | 3/4 | 3/6 | 1/3 | rejected | unlikelihood improved one generation-health item but math/code still failed |

Artifacts:

- Calibration:
  `experiments/strict_behavior_recovery_20260628.jsonl`
- Rejected evals:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/eval_qwen4b_tied29_strict_behavior_replay_last2_120s_strictcode.json`
- `runs/qwen4b_to_smollm135m_transport_random_20260627/eval_qwen4b_tied29_strict_behavior_replay_headnorm_160s_strictcode.json`
- `runs/qwen4b_to_smollm135m_transport_random_20260627/eval_qwen4b_tied29_strict_behavior_antirep_headnorm_120s_strictcode.json`
- Rank file:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/rank_qwen4b_tied29_behavior_replay_all2_strictcode.json`

Interpretation:

- The earlier concern was valid: some paths could still select or reject by PPL
  before behavior/manual quality. The tooling now makes PPL hardness explicit
  and ranks by behavior first.
- Short response replay did not fix the actual model. It reduced training loss
  and sometimes moved arithmetic prefixes closer, but greedy decoding still
  overgenerated digits and malformed Python.
- These runs reinforce that the next useful work is not more response-only
  replay. The target needs stronger parent-first structural repair: OT/PCA
  residual maps, full-block reconstruction, numeric/tokenizer-specific output
  control, and on-policy correction.

### Generation-Health-Aware Behavior Checkpoints

Code changes:

- `--save-best-behavior` now evaluates and snapshots the untouched input
  checkpoint as step `0` before any recovery update.
- Held-out behavior summaries now include the normal generation prompts used by
  `hf-eval`, reported as `generation_health_correct/total`.
- Behavior checkpoint selection now sorts by behavior correctness, behavior
  pass rate, generation-health correctness, generation-health rate, then loss.
- Behavior layer gates also reject a block update if it loses previously passing
  generation prompts, not only if it loses behavior-suite probe IDs.

Remote validation run:

- Start:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_tied29_rsd_behaviorbest_60s`
- Output:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_tied29_genhealth_guard_last2_40s`
- Settings: `40` recovery steps, `train_scope=last2`, behavior eval every `10`
  steps, same strict behavior replay calibration.
- Checkpoint history:
  - step `0`: behavior `3/6`, generation health `2/3`
  - step `1`: behavior `3/6`, generation health `2/3`, selected by loss tie
  - step `10`: behavior `3/6`, generation health `1/3`, rejected
  - step `20`: behavior `3/6`, generation health `1/3`, rejected
  - step `30`: behavior `3/6`, generation health `0/3`, rejected
  - step `40`: behavior `3/6`, generation health `0/3`, rejected

Eval comparison:

| Checkpoint | PPL mean | Choice | Behavior | Generation health | Gate | Manual notes |
|---|---:|---:|---:|---:|---|---|
| RSD behavior-best 60s | 3,825.12 | 3/4 | 3/6 | 2/3 | fail | code/math malformed |
| Generation-health guard last2 40s | 3,820.76 | 3/4 | 3/6 | 2/3 | fail | preserved generation health; code/math still malformed |

Interpretation:

- The new selector correctly prevented later low-signal recovery checkpoints
  with identical behavior count but worse generation health from being saved.
- This is not a promoted model result. It is a safety/control improvement for
  future structural recovery work.

### Activation-Energy Donor Depth Grouping

Research trigger:

- 2025-2026 work on OT/activation-aware merging points away from blind depth
  bins and toward activation correspondences: DOTResize uses discrete optimal
  transport for neuron merging, Transport and Merge aligns heterogeneous model
  activations with OT, and OPRD/cross-tokenizer on-policy distillation argue for
  recovery on child rollouts instead of static PPL-only calibration.

Code changes:

- `hf-donor-project-init` and `hf-xfamily-block-recover` now accept
  `--donor-layer-group-strategy depth|activation-energy`.
- `depth` preserves the old proportional donor depth bins.
- `activation-energy` partitions donor layers by calibrated residual update
  energy. It uses a dynamic-programming contiguous partition with a depth-ratio
  group-size cap, so activation energy can move fold boundaries without creating
  a large final catch-all group.
- Blockwise recovery can compute the same donor residual-update scores from the
  teacher hidden states on paired calibration batches.

Remote validation:

- Remote focused tests after the final DP partitioner: `147 passed`.
- DP sanity check for the observed Qwen3-4B to SmolLM2-135M spike case:
  `[[0, 1], [2, 3], [4, 5], [6], [7, 8], [9, 10], [11, 12], ...]`,
  max group size `2`, full donor-depth partition preserved.

Initializer smoke results:

| Checkpoint | Params | Grouping | PPL mean | Choice | Behavior | Manual notes |
|---|---:|---|---:|---:|---:|---|
| Activation-energy uncapped smoke | 162,826,560 | early group `[0..5]`, then layer `6` spike | 485,165,195.41 | 2/4 | 0/6 | emitted empty continuations or prompt echo |
| Activation-energy ratio-cap smoke | 162,826,560 | max group size `2`, layer `6` isolated | 485,165,195.41 | 1/4 | 0/6 | repeated `sett`, backticks, and malformed fragments |

Interpretation:

- This branch is not promoted. The activation-energy depth grouping is now safe
  enough to experiment with, but the tested initializer is much worse than the
  existing depth/transport baselines.
- The useful next step is not more static depth grouping. The evidence points to
  sequential block-output reconstruction or on-policy representation recovery
  with behavior gates.

### On-Policy Hidden Representation Recovery

Code changes:

- `hf-xfamily-recover` now accepts `--on-policy-hidden-kd-weight`.
- When enabled, recovery generates short greedy child rollouts, feeds those
  rollout texts through donor and child, and aligns hidden states on
  cross-tokenizer decoded-token spans when offsets match.
- If exact span matches are sparse, it falls back to pooled hidden-state
  alignment on the same child-generated text.
- This targets the observed failure where static calibration losses improve
  while greedy behavior still collapses.

Remote smoke:

- Start:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_tied29_rsd_behaviorbest_60s`
- Output:
  `runs/qwen4b_to_smollm135m_transport_random_20260627/qwen4b_tied29_onpolicy_hidden_smoke`
- Settings: Qwen3-4B donor, SmolLM2-135M-family child, `2` steps,
  `train_scope=last1`, `on_policy_hidden_kd_weight=0.2`,
  `hidden_kd_metric=cka`, `on_policy_interval=1`, `on_policy_max_new_tokens=8`.
- Result: finite real-checkpoint losses with rollout hidden alignments observed:
  step `1` count `21`, hidden loss `0.2566`; step `2` count `23`, hidden loss
  `0.3495`.
- Remote focused tests after implementation: `127 passed`.

Interpretation:

- This is a tool-path validation, not a promoted model result.
- It confirms the new objective runs on the real Qwen4B-to-SmolLM folded child
  and produces an actual child-rollout representation signal. Longer
  behavior-gated runs are still required before calling it useful for the 25x
  blank-target compression goal.

## Current Conclusion

The SmolLM blank-target fold is no longer just fitting tensor shapes. The best
strict child now reaches `25.02x` Qwen3-4B-to-SmolLM compression and transfers
enough behavior to generate some useful short text and beat several prior random
or tied baselines.

It is not yet a successful replacement for donor capability, and the same-tokenizer Qwen blank-target results show that the utility cannot yet claim arbitrary blank-target 25x compression. The remaining failure mode is structural recovery, not just benchmark tuning: benchmark slices can improve while manual arithmetic, simple code variation, and common-sense instruction following remain unstable or absent.

Next useful work should prioritize representation-preserving fold repair over more one-off choice tuning: DOT-style transport plans, activation-aware least-squares/SVD module reconstruction against donor activations, depth-interface magnitude compensation, and only then numeric/token-aware cross-tokenizer objectives.
