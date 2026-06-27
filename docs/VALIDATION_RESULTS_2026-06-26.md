# Validation Results - 2026-06-26

Hardware: AI server `192.168.50.222`, RTX 3090 target device, PyTorch
2.11.0+cu128, Transformers 5.12.1.

Run directory on the AI server:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_real_validation_20260626
```

## Text Merge And Surgery

Text parents:

- `Qwen/Qwen3-0.6B-Base`
- `Qwen/Qwen3-0.6B`

The merged checkpoint is a same-shape 50/50 weight average. The expanded
checkpoint takes that merged model and expands every gated FFN intermediate
dimension by 5%.

| Model | Params | Mean PPL | Choice Acc | Result |
|---|---:|---:|---:|---|
| Qwen3-0.6B-Base | 596,049,920 | 280.375 | 1.00 | Parent baseline |
| Qwen3-0.6B | 596,049,920 | 535.942 | 1.00 | Parent baseline |
| 50/50 merged | 596,049,920 | 383.395 | 1.00 | Passed behavior smoke test |
| merged + FFN fold 0.900 | 569,643,008 | 1,615,475.234 | 0.00 | Failed; collapsed |
| merged + FFN fold 0.990 | 593,383,424 | 954.659 | 1.00 | Loaded, but degraded/repetitive |
| merged + FFN fold 0.995 | 594,759,680 | 611.624 | 1.00 | Loaded, but degraded |
| merged + FFN expand 1.050 | 609,296,384 | 383.855 | 1.00 | Passed; near-preserved behavior |

Behavior samples:

- 50/50 merged model answered simple choice probes 4/4 and generated coherent
  text for pruning, Python addition, and merge-risk prompts.
- 10% blind FFN folding produced blank-ish continuations and failed all choice
  probes. This is a real negative result: making the target shape valid is not
  enough.
- 5% FFN expansion stayed on the merged model's behavior curve. This matches the
  expected Net2Wider-style property: duplicated channels split their outgoing
  path and preserve the function much better than blind folding.

## Task-Arithmetic Merge Sweep

A second merge run used `hf-sweep-task` to create
`base + alpha * (tuned - base)` candidates from the two Qwen3 parents, then
evaluated and gated each candidate against the base-model report.

Run directory on the AI server:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_task_sweep_20260626
```

Sweep:

```text
base: Qwen/Qwen3-0.6B-Base
tuned: Qwen/Qwen3-0.6B
baseline report: text_base.json
alphas: 0.25, 0.5, 0.75, 1.0, 1.25
dtype: float16
max_ppl_ratio: 2.25
min_choice_accuracy: 0.75
max_accuracy_drop: 0.0
```

| Alpha | Params | Mean PPL | PPL Ratio | Choice Acc | Gate |
|---:|---:|---:|---:|---:|---|
| 0.25 | 596,049,920 | 321.469 | 1.147 | 1.00 | Passed |
| 0.50 | 596,049,920 | 383.395 | 1.367 | 1.00 | Passed |
| 0.75 | 596,049,920 | 449.954 | 1.605 | 1.00 | Failed: generation health |
| 1.00 | 596,049,920 | 535.942 | 1.912 | 1.00 | Passed |
| 1.25 | 596,049,920 | 665.984 | 2.375 | 1.00 | Failed: PPL ratio |

Useful generation samples from passed candidates:

- Alpha `0.25` explained pruning coherently as removing unnecessary connections
  and neurons to improve efficiency.
- Alpha `0.50` produced a plausible short explanation of pruning and sketched an
  `add_numbers(a, b)` function task in natural language.
- Alpha `1.00`, the tuned parent end of the task vector, still answered choice
  probes 4/4, but some generations were more instruction-repetitive than alpha
  `0.25` or `0.50`.

Interpretation:

- Task arithmetic gives a controllable same-shape cross-model merge path for
  related checkpoints that share a base.
- Alpha sweeps are necessary. Passing scalar choice probes alone is not enough;
  alpha `0.75` hit acceptable PPL and choice accuracy but failed the generation
  health gate.
- This is a merge-plus-eval workflow, not a blind tensor operation.

Post-audit smoke:

- A patched `hf-task-merge` alpha `0.25` run was repeated after adding finite
  alpha validation, duplicate-tag rejection, and tokenizer compatibility
  reporting.
- The Qwen base/tuned tokenizers had matching vocabulary length `151,669` and
  matching vocab entries, with different special-token maps recorded in the
  merge report.
- The patched candidate reproduced the same metrics: mean PPL `321.469`, choice
  accuracy `1.00`, and gate passed.

## Automated Gate Sweep

A follow-up sweep used `hf-sweep-ffn` with:

```text
source: text_merged_alpha_050
baseline report: text_merged_alpha_050.json
ratios: 0.995, 1.05
max_ppl_ratio: 1.25
min_choice_accuracy: 0.75
max_accuracy_drop: 0.0
```

Run directory on the AI server:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_safe_sweep_20260626
```

| Candidate | Params | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---|
| FFN 0.995 | 594,759,680 | 611.624 | 1.00 | Failed: PPL ratio 1.595 > 1.25 |
| FFN 1.050 | 609,296,384 | 383.855 | 1.00 | Passed |

This is now the expected workflow: model surgery creates candidates, behavior
probes measure them, and `hf-gate` / `hf-sweep-ffn` decide whether the candidate
is safe enough to promote. Dimension-valid checkpoints are not automatically
accepted.

## Calibration Repair Attempt

A follow-up run used `hf-repair-ffn` on the failed `FFN 0.995` shrink. The
repair refits each resized gated FFN `down_proj` with ridge least squares using
parent-model FFN inputs/outputs collected from calibration text.

Run directory on the AI server:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_repair_0995_20260626
```

| Candidate | Params | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---|
| FFN 0.995 unrepaired | 594,759,680 | 611.624 | 1.00 | Failed |
| FFN 0.995 repaired | 594,759,680 | 432.449 | 1.00 | Failed: generation repetition |

Interpretation:

- Calibration repair substantially improved perplexity: PPL ratio moved from
  `1.595` down to `1.128` against the merged baseline.
- The repaired candidate still failed the generation-health gate because outputs
  became repetitive on two of three generation probes.
- This confirms the repair direction is useful but incomplete. The next shrink
  milestone should add sequential layer repair and/or short recovery training,
  then re-run the same gate.

## Cross-Family Vocabulary Transplant

The corrected cross-family path keeps a host runtime architecture and folds
donor signal through shared decoded-token anchors into host embedding and
lm-head rows. It does not average mismatched transformer blocks.

SmolLM2 host with Qwen3 donor:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_smollm2_qwen3_20260626
```

| Candidate | Params | Shared Anchors | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| SmolLM2-135M host | 134,515,008 | - | 347.405 | 0.50 | Baseline |
| Qwen3-0.6B donor | 596,049,920 | - | 280.375 | 1.00 | Donor reference |
| host <- donor alpha 0.01 | 134,515,008 | 31,977 | 344.662 | 0.50 | Passed |
| host <- donor alpha 0.05 | 134,515,008 | 31,977 | 336.179 | 0.50 | Passed |
| host <- donor alpha 0.10 | 134,515,008 | 31,977 | 334.584 | 0.50 | Passed |

Pythia/GPT-NeoX host with SmolLM2 donor:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_pythia_smollm2_small_alpha_20260626
```

| Candidate | Params | Shared Anchors | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| Pythia-160M host | 162,322,944 | - | 361.452 | 0.25 | Baseline |
| host <- donor alpha 0.001 | 162,322,944 | 29,291 | 355.177 | 0.25 | Passed |
| host <- donor alpha 0.005 | 162,322,944 | 29,291 | 367.867 | 0.25 | Passed |
| host <- donor alpha 0.010 | 162,322,944 | 29,291 | 376.080 | 0.25 | Failed: generation health |
| host <- donor alpha 0.020 | 162,322,944 | 29,291 | 413.958 | 0.25 | Passed |
| host <- donor alpha 0.050 | 162,322,944 | 29,291 | 698.400 | 0.25 | Failed |
| host <- donor alpha 0.100 | 162,322,944 | 29,291 | 4,178.032 | 0.25 | Failed |

Interpretation:

- Cross-family/tokenizer mismatch is now handled by an explicit anchor map and
  ridge projection, not by pretending incompatible state dicts line up.
- Alpha search is required. Pythia tolerated very small donor injection but
  failed at larger alphas.
- This is a constrained cross-family transplant, not yet a full donor-block
  architecture merge.

## Cross-Family MLP-Channel Transplant

The next cross-family path keeps a host runtime architecture and merges donor
MLP channel signal into the host MLP weights. Shared decoded-token anchors solve
a donor-hidden to host-hidden projection. Donor MLP channels are projected,
assigned to similar host MLP channels, then blended into the host weights. This
does not use an adapter at inference time; the output is a normal host
checkpoint.

Supported layer families in this pass:

- Qwen/Llama/SmolLM-style gated MLPs: `gate_proj`, `up_proj`, `down_proj`;
- Liquid LFM2.5 gated MLPs: `feed_forward.w1`, `feed_forward.w3`,
  `feed_forward.w2`;
- GPT-NeoX/Pythia dense MLPs: `dense_h_to_4h`, `dense_4h_to_h`.

Qwen3 host with Pythia donor:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_mlp_qwen3_pythia_20260626
```

| Candidate | Host | Donor | Params | Mean PPL | Choice Acc | Gate |
|---|---|---|---:|---:|---:|---|
| baseline | Qwen3-0.6B-Base | - | 596,049,920 | 280.375 | 1.00 | Baseline |
| alpha 0.0005 | Qwen3-0.6B-Base | Pythia-160M | 596,049,920 | 279.701 | 1.00 | Passed |
| alpha 0.0010 | Qwen3-0.6B-Base | Pythia-160M | 596,049,920 | 279.810 | 1.00 | Passed |

Qwen3 host with LFM2.5 donor:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_mlp_qwen3_lfm25_20260626
```

| Candidate | Host | Donor | Params | Mean PPL | Choice Acc | Gate |
|---|---|---|---:|---:|---:|---|
| baseline | Qwen3-0.6B-Base | - | 596,049,920 | 280.375 | 1.00 | Baseline |
| alpha 0.0005 | Qwen3-0.6B-Base | LFM2.5-230M | 596,049,920 | 280.324 | 1.00 | Passed |
| alpha 0.0010 | Qwen3-0.6B-Base | LFM2.5-230M | 596,049,920 | 279.878 | 1.00 | Passed |

LFM2.5 host with Qwen3 donor:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_mlp_lfm25_qwen3_20260626
```

| Candidate | Host | Donor | Params | Mean PPL | Choice Acc | Gate |
|---|---|---|---:|---:|---:|---|
| baseline | LFM2.5-230M | - | 229,693,184 | 404.602 | 1.00 | Baseline |
| alpha 0.0005 | LFM2.5-230M | Qwen3-0.6B-Base | 229,693,184 | 403.082 | 1.00 | Failed: generation health |
| alpha 0.0010 | LFM2.5-230M | Qwen3-0.6B-Base | 229,693,184 | 401.442 | 1.00 | Failed: generation health |

Interpretation:

- This proves a real cross-family, cross-architecture MLP transplant path:
  gated Qwen host from dense Pythia donor passed, and gated Qwen host from
  gated LFM2.5 donor passed.
- The LFM2.5-host run proves the tool can emit a valid 230M LFM-shaped
  checkpoint after merging Qwen donor channels. It preserved PPL and choice
  accuracy, but the strict text gate rejected it because the LFM baseline and
  candidates all stopped immediately on one generic merge-risk generation
  prompt.
- These runs still use very small alphas and only the top 128 MLP channels per
  layer. They are safe cross-family signal transplants, not proof that large
  donor capability transfer works without recovery training.

Cross-family MLP transplant with integrated recovery:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_mlp_qwen3_lfm25_recover3_20260626
```

Run settings:

```text
host: Qwen/Qwen3-0.6B-Base
donor: LiquidAI/LFM2.5-230M
alpha: 0.005
shared-token anchors: 30,000
host channels considered per layer: 256
donor channels considered per layer: 256
recovery: enabled
recovery steps: 8
recovery train scope: last1
recovery dtype: float32
```

| Stage | Params | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---|
| Qwen3 host baseline | 596,049,920 | 280.375 | 1.00 | Baseline |
| Qwen3 <- LFM2.5 MLP alpha 0.005 | 596,049,920 | 284.984 | 1.00 | Passed |
| recovered alpha 0.005 | 596,049,920 | 227.893 | 1.00 | Passed |

Recovery details:

- `hf-sweep-xfamily-mlp --recover` saved the recovered checkpoint under
  `recovered_models/xfamily_mlp_alpha_0p0050_recovered`;
- recovery used 8 calibration texts that do not overlap the eval prompts;
- train scope `last1` updated 171,313,408 trainable parameters;
- logged recovery loss moved from `3.743` at step 1 to `2.959` at step 8,
  with a transient spike at step 5.

Sample behavior:

- baseline and merged model both explained neural pruning as removing
  unnecessary or redundant neurons/connections;
- recovered model kept the pruning explanation readable and concise;
- baseline, merged, and recovered models all produced a usable answer about
  merge risk; the recovered answer warned that merged models may generalize
  poorly when parent learning patterns differ.

## Cross-Family Merge-Compress Pipeline

The repo now includes `hf-xfamily-compress`, which chains:

1. cross-family MLP-channel merge into a host-shaped checkpoint;
2. merge-first width/depth compression search;
3. optional lightweight recovery;
4. gates against both the cross-family intermediate and the original host
   baseline.

The donor in both runs below was `LiquidAI/LFM2.5-230M`.

Successful recovered compression run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_compress_qwen3_lfm25_20260626_182026
```

| Stage | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| Qwen3 host baseline | 596,049,920 | - | 280.375 | 1.00 | Baseline |
| Qwen3 <- LFM2.5 MLP alpha 0.005 | 596,049,920 | 0.00% | 284.984 | 1.00 | Passed |
| width 0.995 + depth average 13+14 | 579,074,816 | 2.85% | 310.255 | 0.75 | Passed |
| recovered compressed candidate | 579,074,816 | 2.85% | 266.520 | 0.75 | Passed |

Recovery loss moved from `4.2172` at step 1 to `3.2870` at step 8.

Safer 5%+ merge-compress run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_compress_qwen3_lfm25_5pct_safe_20260626_182857
```

| Stage | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| Qwen3 host baseline | 596,049,920 | - | 280.375 | 1.00 | Baseline |
| Qwen3 <- LFM2.5 MLP alpha 0.005 | 596,049,920 | 0.00% | 284.984 | 1.00 | Passed |
| depth fold 13+14 and 15+16 | 564,588,032 | 5.28% | 294.424 | 0.75 | Passed |

The same 5.28% candidate was also tested with a short recovery pass in:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_compress_qwen3_lfm25_5pct_20260626_182346
```

That recovery improved mean PPL to `243.582` but dropped choice accuracy to
`0.50`, so the pipeline was updated to record `usable_candidates` and
`best_usable_candidate`, and to support `--recover-only-failed`. The safe run
above used that option and did not promote the harmful recovered checkpoint.

Sample generated behavior from the 5.28% best candidate:

- pruning prompt: "Neural pruning is a process where the brain eliminates
  unnecessary connections..." It remained on-topic but more repetitive than the
  host.
- Python-add prompt: described a two-argument sum function and remained
  coherent, again with more repetition than the host.

## Merge-First Width And Depth Folding

After the pruning-first baseline failed, the Qwen3 run was repeated with
merge/fold-first operations:

- width: combined SwiGLU channel similarity over `gate_proj`, `up_proj`, and
  `down_proj`, then average incoming gate/up paths and sum outgoing down paths;
- depth: residual decoder layer groups are merged into a survivor layer, then
  duplicate layer slots are removed.

Width-only run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_width_combined_20260626
```

| Candidate | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B-Base | 596,049,920 | - | 280.375 | 1.00 | Baseline |
| FFN 0.9975 combined-cosine | 595,361,792 | 0.115% | 295.301 | 1.00 | Passed |
| FFN 0.9950 combined-cosine | 594,759,680 | 0.216% | 294.778 | 1.00 | Passed |

Depth-only run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_layer_merge_20260626
```

| Candidate | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| average layers 13+14 | 580,318,976 | 2.64% | 282.380 | 0.75 | Passed |
| fold layers 13+14 | 580,318,976 | 2.64% | 295.854 | 0.75 | Passed |
| average layers 19+20 | 580,318,976 | 2.64% | 343.944 | 0.50 | Failed |
| fold layers 19+20 | 580,318,976 | 2.64% | 317.273 | 0.50 | Failed |
| average layers 8+9 and 19+20 | 564,588,032 | 5.28% | 319.927 | 0.25 | Failed |
| fold layers 8+9 and 19+20 | 564,588,032 | 5.28% | 315.499 | 0.25 | Failed |

Width plus depth chain:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_width_depth_combo_20260626
```

| Candidate | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---:|---|
| width 0.995 combined-cosine | 594,759,680 | 0.216% | 294.778 | 1.00 | Passed |
| width 0.995 + average layers 13+14 | 579,074,816 | 2.848% | 305.656 | 0.75 | Passed |
| width 0.995 + fold layers 13+14 | 579,074,816 | 2.848% | 320.186 | 1.00 | Passed |

Automated merge-first search:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_merge_first_search_20260626
```

Search settings:

```text
source: Qwen/Qwen3-0.6B-Base
width_ratios: 0.995
layer_groups: 13+14
depth_methods: average,fold
target_shrink: 0.028
assignment: combined-cosine
max_ppl_ratio: 1.5
min_choice_accuracy: 0.75
max_accuracy_drop: 0.25
```

| Candidate | Kind | Params | Shrink | Mean PPL | Choice Acc | Gate |
|---|---|---:|---:|---:|---:|---|
| width_0p9950 | width | 594,759,680 | 0.216% | 294.778 | 1.00 | Passed |
| depth_average_13-14 | depth | 580,318,976 | 2.639% | 282.380 | 0.75 | Passed |
| depth_fold_13-14 | depth | 580,318,976 | 2.639% | 295.854 | 0.75 | Passed |
| width_0p9950_depth_average_13-14 | width_depth | 579,074,816 | 2.848% | 305.656 | 0.75 | Passed |
| width_0p9950_depth_fold_13-14 | width_depth | 579,074,816 | 2.848% | 320.186 | 1.00 | Passed |

The search summary selected `width_0p9950_depth_average_13-14` as the best
target-passing candidate. Both width+depth candidates met `target_shrink=0.028`,
so the automated path reproduced the manual merge-first result.

Useful generation samples from the combined pass:

- pruning prompt: explained pruning as eliminating unnecessary connections and
  neurons for efficient information processing;
- Python prompt: generated a valid `add_numbers(a, b): return a + b` function;
- merge-risk prompt: warned about loss of diversity/nuance when merging models.

Interpretation:

- The merge-first correction matters. The old down-only FFN fold failed even at
  small ratios; the combined SwiGLU similarity fold passed at 0.995.
- Depth folding is layer-location sensitive. Layer group `13+14` survived; later
  and multi-group folds failed behavior gates.
- The current proven checkpoint is modest compression, not arbitrary 70B->7B.
  The safe workflow is search merge candidates, gate them, then iterate.

## Vision Parent Probe

Vision model:

- `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- Params: 256,484,928

Synthetic probes:

- red square plus blue circle
- black `42` in a bordered image

Outputs:

- Shape probe: "A red square and a blue circle are placed side by side."
- OCR probe: "The text or number visible in the image is \"42\"."

## Composite Frankenstack Probe

Composite:

- vision parent: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- text backend: `text_merged_ffn_105`, the expanded Qwen3 50/50 merge

This is not a native multimodal graft. It is a transitional stitched system:
image observation from the VLM is handed to the merged/expanded text model.

Results:

- Shape chain answered with the correct red-square/blue-circle description.
- OCR chain answered `42`, then became repetitive.

Interpretation:

- The merged text model can participate in a useful vision-language chain.
- The chain is verbose and repetitive; native vision expansion needs training and
  decoding/eval gates.
- The next safe technical milestone is calibration-aware FFN folding plus a
  trainable raw-patch/token vision graft, not blind large compression.

## Native Early-Fusion Sandbox

The repo now includes `native-vlm-demo`, a tiny native multimodal transformer
that trains over raw image patches and text tokens in one causal sequence. This
is not a frozen vision encoder plus bridge. The image path is patch projection
into the same transformer stream used for text tokens.

Run directory on the AI server:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/native_vlm_ocr_full_20260626
```

Task:

- 16x16 synthetic RGB images;
- colored square and colored circle;
- seven-segment OCR-lite digit `zero` through `three`;
- questions: `what color is square ?`, `what color is circle ?`, and
  `what number ?`;
- answer generated as a text token.

Run:

```text
steps: 1600
batch_size: 96
eval_batches: 32
params: 350,227
device: cuda:0
```

Result:

| Metric | Value |
|---|---:|
| Final logged loss | 0.000534 |
| Eval samples | 1,024 |
| Exact answer accuracy | 1.000 |

Sample predictions:

- expected `green`, predicted `green`;
- expected `three`, predicted `three`;
- expected `red`, predicted `red`;
- expected `zero`, predicted `zero`.

Interpretation:

- This proves the repo has a working native early-fusion multimodal training
  path, not only a VLM-to-text chain.
- The scale is deliberately tiny and synthetic, but the mechanics are the same
  family needed for raw-patch/domain expansion experiments: image patches and
  text tokens share the transformer stack and the model learns answers
  end-to-end.

## Merged-Parent Native Vision Graft

The repo now also includes `hf-vision-graft`, which feeds raw image patch
embeddings into an existing Hugging Face causal LM parent. For this run, the
parent was the already merged Qwen3 alpha `0.25` task-arithmetic checkpoint:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_task_guard_smoke_20260626
```

The successful all-task graft run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen3_alpha025_native_graft_all_patch_20260626
```

Run:

```text
parent params: 596,049,920
graft params: 66,560
trainable parent params: 0
steps: 900
batch_size: 16
eval_batches: 8
device: cuda:0
dtype: float16
tasks: square_color, circle_color, digit
```

Result:

| Metric | Before Graft Training | After Graft Training |
|---|---:|---:|
| Eval samples | 64 | 64 |
| Exact answer accuracy | 0.234 | 1.000 |
| Mean gold loss | 5.045 | 0.0015 |
| Non-finite losses | no | no |

Sample generated answers after graft training:

- square-color question: expected `red`, predicted `red`, generated `red`;
- circle-color question: expected `green`, predicted `green`, generated
  `green`;
- digit question: expected `one`, predicted `one`, generated `one`;
- digit question: expected `zero`, predicted `zero`, generated `zero`.

Negative runs:

- A short frozen-parent all-task run learned answer-format priors but not stable
  pixel grounding.
- Unfreezing the last decoder block in `float16` produced NaN losses. The tool
  now rejects `--train-last-layers > 0` with `--dtype float16` and aborts before
  saving if any non-finite loss appears.

Interpretation:

- This is the current strongest proof that the repo can combine existing text
  parents, preserve the merged parent's language behavior, and add a small
  native vision input path that produces useful text answers from pixels.
- The parent LM was frozen in the successful run, so text behavior cannot drift
  from parent weights in this graft. The trainable addition is a raw patch
  projection into the same causal token stream, not an external frozen vision
  encoder.
- The task is still synthetic and small. It proves the mechanics and validation
  harness, not general open-world visual reasoning.

## Cross-Family Compressed Parent Vision Graft

The raw-patch graft was then run on the actual cross-family compressed parent:

```text
parent:
/home/cereal/model-surgery-lab/neural-forge-src/runs/xfamily_compress_qwen3_lfm25_5pct_safe_20260626_182857/compression/models/depth_fold_13-14_15-16

graft run:
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen_lfm25_5pct_native_graft_clean_20260626_184018
```

Parent provenance:

- host: `Qwen/Qwen3-0.6B-Base`;
- donor: `LiquidAI/LFM2.5-230M`;
- cross-family operation: MLP-channel transplant, `alpha=0.005`;
- compression: depth-folded layer groups `13+14` and `15+16`;
- parent params after compression: `564,588,032`;
- shrink vs Qwen3 host baseline: `5.28%`;
- text gate before graft: passed, with mean PPL `294.424` and choice accuracy
  `0.75`.

Vision graft run:

```text
graft params: 66,560
trainable parent params: 0
steps: 900
batch_size: 16
eval_batches: 8
device: cuda:0
dtype: float16
tasks: square_color, circle_color, digit
```

Result:

| Metric | Before Graft Training | After Graft Training |
|---|---:|---:|
| Eval samples | 64 | 64 |
| Exact answer accuracy | 0.219 | 1.000 |
| Mean gold loss | 7.4789 | 0.0004 |
| Logged training loss | 6.9178 | 0.0004 |

Sample generated answers after graft training:

- square-color question: expected `yellow`, predicted `yellow`, generated
  `yellow`;
- circle-color question: expected `green`, predicted `green`, generated
  `green`;
- square-color question: expected `green`, predicted `green`, generated
  `green`;
- circle-color question: expected `blue`, predicted `blue`, generated `blue`.

Interpretation:

- This is the strongest current end-to-end proof in the repo: a Qwen host was
  cross-family-merged with LFM2.5, compressed by merge-first layer folding, kept
  passing text behavior gates, then learned a small raw-patch vision task with
  the parent frozen.
- Because `trainable parent params` was `0`, the text parent weights did not
  drift during vision graft training. The added capability came from the small
  patch projection inserted into the same causal embedding stream.
- This still does not prove arbitrary open-world multimodal ability or arbitrary
  target-size compression. It proves the toolchain can stitch a real
  cross-family compressed parent and add a validated native pixel input path.

## First 4B Donor Compression Tests

The first 4B-class donor tests used `Qwen/Qwen3-4B-Base` as donor. The target
models deliberately excluded Liquid LFM2.5.

### 4B Qwen Into Pythia-160M

Run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_pythia160m_20260626_190053
```

Compression ratio:

- donor params: `4,022,468,096`;
- host/target params: `162,322,944`;
- donor-to-target ratio: `24.78x`.

Neural Forge behavior probes:

| Model | Params | Mean PPL | Choice Acc | Notes |
|---|---:|---:|---:|---|
| Pythia-160M host | 162,322,944 | 361.452 | 0.25 | repetitive generations |
| Qwen3-4B donor | 4,022,468,096 | 269.326 | 1.00 | clean code/pruning answers |
| folded alpha 0.001 | 162,322,944 | 361.431 | 0.25 | essentially unchanged |
| folded alpha 0.005 | 162,322,944 | 367.032 | 0.25 | worse |
| folded alpha 0.010 | 162,322,944 | 358.861 | 0.25 | PPL slightly better, gate failed |

Curated recovery:

- calibration source:
  `/mnt/ai_data/omnicoder_2026/datasets/shards/v9_base_20260605T142420Z/jsonl_shards/v9_base_validated_28025-000002.jsonl`;
- 512 texts loaded through `--recover-calibration-file`;
- 30 steps, `last2`, `kd_weight=0.2`, `lm_weight=0.8`.

Result: recovery made held-out PPL worse (`~593-612`) while choice accuracy
stayed `0.25`. This is a useful negative result: current cross-tokenizer MLP
channel transplant does not transfer measurable Qwen4B capability into
Pythia-160M.

### 4B Qwen Into SmolLM2-135M

Run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_smollm135m_20260626_190625
```

Compression ratio:

- donor params: `4,022,468,096`;
- host/target params: `134,515,008`;
- donor-to-target ratio: `29.90x`.

Neural Forge behavior probes:

| Model | Params | Mean PPL | Choice Acc |
|---|---:|---:|---:|
| SmolLM2-135M host | 134,515,008 | 347.405 | 0.50 |
| folded alpha 0.001, vocab alpha 0.001 | 134,515,008 | 348.699 | 0.50 |
| folded alpha 0.005, vocab alpha 0.001 | 134,515,008 | 352.821 | 0.50 |
| folded alpha 0.010, vocab alpha 0.001 | 134,515,008 | 358.797 | 0.50 |
| folded alpha 0.001 + LM-only recovery | 134,515,008 | 447.909 | 0.50 |
| host + same LM-only recovery control | 134,515,008 | 448.448 | 0.50 |

External benchmark-suite slice:

- tool: `/home/cereal/ai-benchmark-lab/envs/lm-eval`;
- tasks: `arc_easy`, `hellaswag`;
- limit: `20` examples/task;
- output root:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_smollm135m_20260626_190625/lm_eval_slice`.

| Model | ARC-Easy Acc | ARC-Easy Acc Norm | HellaSwag Acc | HellaSwag Acc Norm |
|---|---:|---:|---:|---:|
| SmolLM2-135M host | 0.55 | 0.40 | 0.35 | 0.30 |
| Qwen3-4B donor | 0.80 | 0.65 | 0.50 | 0.65 |
| folded alpha 0.001 | 0.55 | 0.40 | 0.35 | 0.30 |
| folded alpha 0.001 + LM-only recovery | 0.60 | 0.45 | 0.35 | 0.30 |
| host + same LM-only recovery control | 0.60 | 0.45 | 0.35 | 0.30 |

Interpretation:

- The recovered folded model improved ARC-Easy on the slice, but the host-only
  recovery control improved by the same amount. That means the measured gain
  came from curated LM recovery, not from Qwen4B donor folding.
- Current MLP/vocab transplant is shape-valid at nearly `30x` donor-to-target
  compression, but it is not yet a successful cross-tokenizer capability
  transfer method.

### 4B Qwen Into Qwen3-0.6B Same-Tokenizer Control

Run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_qwen06b_20260626_191954
```

Compression ratio:

- donor params: `4,022,468,096`;
- host/target params: `596,049,920`;
- donor-to-target ratio: `6.75x`.

Neural Forge behavior probes:

| Model | Params | Mean PPL | Choice Acc | Gate |
|---|---:|---:|---:|---|
| Qwen3-0.6B host | 596,049,920 | 280.375 | 1.00 | baseline |
| folded alpha 0.001 | 596,049,920 | 280.118 | 1.00 | passed |
| folded alpha 0.005 | 596,049,920 | 280.786 | 1.00 | passed |
| folded alpha 0.010 | 596,049,920 | 281.637 | 1.00 | passed |
| alpha 0.001 + Qwen4B-teacher recovery | 596,049,920 | 283.723 | 1.00 | worse PPL |

The first fp16 Qwen4B-teacher recovery attempt failed with non-finite loss at
step 2. Retrying in `bfloat16` with a lower LR was stable and reduced recovery
loss from `2.0156` to `1.2344`, but worsened held-out PPL. The raw alpha
`0.001` fold remains the best artifact from this run.

External benchmark-suite slice:

| Model | ARC-Easy Acc | ARC-Easy Acc Norm | HellaSwag Acc | HellaSwag Acc Norm |
|---|---:|---:|---:|---:|
| Qwen3-0.6B host | 0.60 | 0.45 | 0.35 | 0.35 |
| Qwen3-4B donor | 0.80 | 0.65 | 0.50 | 0.65 |
| folded alpha 0.001 | 0.60 | 0.45 | 0.35 | 0.35 |
| alpha 0.001 + Qwen4B-teacher recovery | 0.60 | 0.45 | 0.35 | 0.35 |

Interpretation:

- Same-tokenizer Qwen folding produced a small PPL win at `6.75x` donor-to-target
  compression, but did not move the external benchmark slice.
- Donor-teacher recovery did not help yet. The next technical step is not
  longer blind recovery; it is better alignment before/while folding:
  activation-calibrated channel assignment, per-layer alpha search, and
  cross-tokenizer teacher targets that do not collapse to host-only SFT.

## Activation-Hybrid 4B Donor Smoke

After adding activation-calibrated channel assignment, a constrained smoke run
retested `Qwen/Qwen3-4B-Base` into `HuggingFaceTB/SmolLM2-135M` with matched
channel limits.

Runs:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_smollm135m_activation_hybrid_smoke_20260627_014148
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_smollm135m_weight_matched_smoke_20260627_014257
```

Settings:

- host baseline report:
  `runs/qwen4b_into_smollm135m_20260626_190625/reports/host_smollm2_135m.json`;
- donor: `Qwen/Qwen3-4B-Base`;
- host: `HuggingFaceTB/SmolLM2-135M`;
- alpha: `0.001`;
- vocab alpha: `0.001`;
- max host channels: `128`;
- max donor channels: `256`;
- activation calibration: 8 curated Omnicoder texts, max length `64`;
- max new tokens during smoke eval: `24`.

| Model / assignment | Params | Mean PPL | Choice Acc | Gate | First-layer assignment similarity |
|---|---:|---:|---:|---|---:|
| SmolLM2-135M host baseline | 134,515,008 | 347.405 | 0.50 | baseline | n/a |
| weight-cosine matched smoke | 134,515,008 | 348.631 | 0.50 | passed | 0.077 |
| activation-hybrid smoke | 134,515,008 | 347.377 | 0.50 | passed | 0.512 |

Interpretation:

- The new activation-hybrid path successfully ran end-to-end with a 4B donor,
  generated a host-shaped checkpoint, and recorded activation-backed layer
  reports (`activation_features_used=true`).
- Under the same channel budget, activation-hybrid was less damaging than
  weight-cosine and slightly better than the host baseline on the lightweight
  PPL probe.
- This is still only a smoke result. It does not prove donor intelligence
  transfer on ARC/HellaSwag or broad behavior; it just validates the next
  alignment mechanism needed before larger recovery/benchmark runs.

## 4B Donor Benchmark Slice After Activation/Layer-Alpha Changes

After the PPL-only smoke, the same Qwen3-4B -> SmolLM2-135M family of artifacts
was benchmarked with `lm-evaluation-harness`, because PPL alone is not enough
evidence for intelligence or coherence.

Run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_smollm135m_layer_alpha_bench_20260627_015220
```

Benchmark settings:

- tasks: `arc_easy`, `hellaswag`, `piqa`, `winogrande`;
- limit: `50` examples/task, so this is a comparison slice, not a final score;
- models: SmolLM2 host, Qwen3-4B donor, matched weight-cosine fold,
  activation-hybrid constant-alpha fold, and activation-hybrid similarity-scaled
  layer-alpha fold.

Per-layer fold settings:

- alpha: `0.001`;
- channel assignment: `hybrid`;
- activation calibration: 8 curated Omnicoder texts, max length `64`;
- max host/donor channels: `128` / `256`;
- layer alpha mode: `similarity`;
- scale range: `0.25` to `1.25`;
- similarity floor: `0.30`;
- effective alpha range: `0.000512` to `0.000572`, mean `0.000556`.

Neural Forge probe/generation gate:

| Model | Mean PPL | Choice Acc | Generation Gate |
|---|---:|---:|---|
| host | 347.405 | 0.50 | baseline |
| weight-cosine matched | 348.631 | 0.50 | passed |
| activation-hybrid constant alpha | 347.377 | 0.50 | passed |
| activation-hybrid layer alpha | 346.531 | 0.50 | failed |

The layer-alpha artifact lowered PPL, but failed generation health: the
`Explain neural pruning...` sample repeated enough to produce a unique-token
ratio of `0.16` against the `0.20` floor. This is a concrete example where PPL
looked better while coherence got worse.

Benchmark slice:

| Model | ARC-Easy Acc / Norm | HellaSwag Acc / Norm | PIQA Acc / Norm | Winogrande Acc |
|---|---:|---:|---:|---:|
| SmolLM2 host | 0.52 / 0.48 | 0.44 / 0.48 | 0.62 / 0.64 | 0.48 |
| Qwen3-4B donor | 0.70 / 0.76 | 0.54 / 0.74 | 0.80 / 0.74 | 0.78 |
| weight-cosine fold | 0.52 / 0.48 | 0.44 / 0.48 | 0.62 / 0.64 | 0.48 |
| activation-hybrid constant alpha | 0.52 / 0.48 | 0.44 / 0.48 | 0.62 / 0.64 | 0.48 |
| activation-hybrid layer alpha | 0.52 / 0.48 | 0.44 / 0.48 | 0.62 / 0.64 | 0.48 |

Interpretation:

- The donor is clearly stronger on the benchmark slice.
- None of the SmolLM2-shaped folded artifacts inherited measurable donor
  capability on these tasks. They are benchmark-identical to the host at this
  slice size.
- Activation-hybrid remains less damaging than weight-cosine by the small PPL
  probe, but benchmark and coherence results show that this is not sufficient.
- Similarity-scaled layer alpha made PPL slightly better but generation
  coherence worse. It should not be promoted as a recovery solution without a
  stronger benchmark/coherence gain.

## Blank Random Target Compression Test

This run tested the user correction that the target architecture may be
untrained/random, not an already trained host. The target was a random
SmolLM2-135M-shaped checkpoint initialized from config and tokenizer only, then
folded with Qwen3-4B donor signal.

Run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_into_random_smollm135m_20260627_020620
```

Setup:

- random target config/tokenizer: `HuggingFaceTB/SmolLM2-135M`;
- random seed: `20260627`;
- target params: `134,515,008`;
- donor: `Qwen/Qwen3-4B-Base`;
- donor-to-target ratio: roughly `29.90x`;
- fold: cross-family MLP weight-cosine assignment;
- alpha / vocab alpha: `0.001` / `0.001`;
- max host/donor channels considered per layer: `128` / `256`;
- benchmark tasks: `arc_easy`, `hellaswag`, `piqa`, `winogrande`;
- benchmark limit: `50` examples/task.

Neural Forge probes:

| Model | Mean PPL | Choice Acc | Old Length/Diversity Gate | Strict Relevance Gate |
|---|---:|---:|---|---|
| random SmolLM2 target | 88,650.193 | 0.00 | baseline | n/a |
| Qwen fold into random target | 88,539.163 | 0.00 | passed | failed |

The old generation gate was too weak: it only checked continuation length and
token diversity. The folded random model produced long, diverse continuations,
but the content was unrelated to all three prompts. The gate now also requires
prompt-specific relevance for the built-in generation probes:

- pruning explanation must mention pruning/removal/compression and model
  structure such as neurons, weights, channels, or parameters;
- Python addition prompt must look like a function/lambda and include addition;
- merge-risk prompt must mention merging/combining, a risk/failure mode, and
  model/tokenizer/capability context.

Strict gate result:

```text
runs/qwen4b_into_random_smollm135m_20260627_020620/fold/reports/xfamily_mlp_alpha_0p0010_strict_gate.json
passed: false
```

Benchmark slice:

| Model | ARC-Easy Acc / Norm | HellaSwag Acc / Norm | PIQA Acc / Norm | Winogrande Acc |
|---|---:|---:|---:|---:|
| trained SmolLM2 host | 0.52 / 0.48 | 0.44 / 0.48 | 0.62 / 0.64 | 0.48 |
| random SmolLM2 target | 0.20 / 0.22 | 0.22 / 0.30 | 0.56 / 0.58 | 0.46 |
| Qwen3-4B donor | 0.70 / 0.76 | 0.54 / 0.74 | 0.80 / 0.74 | 0.78 |
| folded random target | 0.20 / 0.22 | 0.22 / 0.30 | 0.56 / 0.58 | 0.46 |

Interpretation:

- The fold produced a loadable checkpoint with the requested target shape, but
  did not transfer measurable donor intelligence into a blank target.
- The folded random target was benchmark-identical to the random target on all
  four task slices.
- PIQA and Winogrande are binary-choice tasks and the slice is only 50 examples,
  so near-0.5 values there are not evidence of intelligence. The ARC/HellaSwag
  results and nonsensical generations confirm the blank target is not useful.
- This rules out tiny-alpha MLP/vocab transplant as a blank-model compression
  solution. Blank-target compression needs a stronger stage: function-preserving
  initialization, cross-depth/width activation alignment, and recovery that is
  benchmark-gated rather than PPL-gated.

## Donor-Projected Blank Target Initialization

After the tiny-alpha random-target fold failed, `hf-donor-project-init` was
added to initialize the blank target directly from donor tensors. It creates the
target architecture from config, maps target layers to donor layers
proportionally, and folds donor tensor axes to the target tensor shape. This is
still merge/fold-first: donor dimensions are averaged into target dimensions,
not discarded one-by-one.

Mean-fold run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_projected_smollm135m_20260627_023125
```

RMS-fold run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_projected_rms_smollm135m_20260627_023639
```

Activation-weighted mean-fold run:

```text
/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_projected_activation_smollm135m_20260627_024603
```

Shared setup:

- target config/tokenizer: `HuggingFaceTB/SmolLM2-135M`;
- donor: `Qwen/Qwen3-4B-Base`;
- target params: `134,515,008`;
- donor-to-target ratio: roughly `29.90x`;
- target layers / donor layers: `30` / `36`;
- initialized layer tensors: `270`;
- shared decoded token anchors available: `31,977`;
- token rows initialized: `30,000` embedding rows and `30,000` LM-head rows;
- benchmark tasks: `arc_easy`, `hellaswag`, `piqa`, `winogrande`;
- benchmark limit: `50` examples/task.
- activation-weighted run calibration: 8 curated Omnicoder JSONL texts, max
  length `96`, donor activations collected on `cuda:0`.

Neural Forge probes:

| Model | Mean PPL | Choice Acc | Strict Generation Gate | Manual output |
|---|---:|---:|---|---|
| random SmolLM2 target | 88,650.193 | 0.00 | baseline | nonsensical |
| projected mean-fold | 63,726.945 | 0.25 | failed | repeated `fibromyalgia`, `apple`, `infringe` |
| projected RMS-fold | 68,513.184 | 0.00 | failed | repeated `height`, `fleas/dilemmas`, `Flu/contour` |
| projected activation-weighted mean-fold | 80,711.923 | 0.25 | failed | repeated `entropy`, `remission`, `polymerase` |

Benchmark slice:

| Model | ARC-Easy Acc / Norm | HellaSwag Acc / Norm | PIQA Acc / Norm | Winogrande Acc |
|---|---:|---:|---:|---:|
| random SmolLM2 target | 0.20 / 0.22 | 0.22 / 0.30 | 0.56 / 0.58 | 0.46 |
| projected mean-fold | 0.24 / 0.30 | 0.24 / 0.32 | 0.54 / 0.58 | 0.64 |
| projected RMS-fold | 0.14 / 0.28 | 0.26 / 0.32 | 0.40 / 0.58 | 0.46 |
| projected activation-weighted mean-fold | 0.24 / 0.28 | 0.24 / 0.26 | 0.60 / 0.62 | 0.52 |

Interpretation:

- Mean-fold projection is the first blank-target method here that moved some
  benchmark numbers above the random baseline, especially Winogrande
  `0.46 -> 0.64`.
- The movement is not enough to call the model useful or coherent. Manual
  generations collapsed into repeated tokens, and the strict generation gate
  correctly failed.
- RMS variance preservation did not improve this run. It is available as an
  option, but mean-fold remains the default because it had the stronger probe
  and benchmark result.
- Activation-weighted folding changed the benchmark profile and improved PIQA
  over random (`0.56 / 0.58 -> 0.60 / 0.62`), but worsened PPL relative to plain
  mean-fold and still collapsed generation. Activation ranking alone is
  insufficient.
- The result is a partial signal: direct donor tensor folding can implant some
  task-statistical structure into a blank target, but still needs activation or
  logit recovery before it is a usable compression path.
- The next research-backed step is not another global ranking heuristic. It
  should fit post-fold reconstruction maps from calibration activations,
  preferably layer-by-layer after recomputing downstream activations, and/or
  use optimal-transport matching instead of hard round-robin activation bins.
