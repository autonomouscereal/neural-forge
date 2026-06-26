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
