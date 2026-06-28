# Neural Forge

Neural Forge is a small, working toolkit for neural checkpoint shape surgery:
shrinking dense models with neuron merging, expanding models with
function-preserving width/depth growth, and merging multiple models into a
requested target shape.

The dense feed-forward path is the clean-room core because hidden neurons have an
explicit axis: a row in one layer and the matching column in the next. The repo
also includes experimental Hugging Face causal-LM checkpoint surgery and a tiny
native early-fusion multimodal training sandbox for real-model validation.

## What works now

- `shrink`: reduce hidden widths by clustering similar neurons, averaging incoming
  weights/biases, and summing outgoing paths.
- `expand`: widen hidden layers with a Net2Wider-style copy-and-split operation
  that preserves outputs exactly when `--noise-std 0`; optionally insert ReLU
  identity layers to increase depth.
- `merge`: resize multiple same-depth dense models into a common target width,
  align hidden neurons by greedy cosine matching, then average or SLERP the weights.
- `toy`: create tiny checkpoints for testing and experiments.
- `inspect`: print architecture and parameter count.
- `hf-merge` / `hf-task-merge`: same-shape Hugging Face causal-LM averaging and
  task-arithmetic merge.
- `hf-resize-ffn` / `hf-repair-ffn`: gated-MLP FFN width merge/fold,
  expansion, and calibration repair for Qwen/Gemma/Llama-like decoder blocks.
- `hf-merge-layers`: merge/fold residual decoder layer groups before reducing
  depth.
- `hf-search-compress`: automated merge-first width/depth candidate search with
  behavior gates.
- `hf-xfamily-vocab-merge` / `hf-sweep-xfamily`: constrained cross-family
  embedding/lm-head transplant through shared decoded-token anchors.
- `hf-xfamily-mlp-merge` / `hf-sweep-xfamily-mlp`: constrained cross-family
  MLP-channel transplant through shared-token hidden projections, with optional
  activation-calibrated channel assignment.
- `hf-xfamily-compress`: cross-family MLP merge followed by merge-first
  width/depth compression, optional recovery, and host-baseline gates.
- `hf-recover`: lightweight same-tokenizer KD/LM recovery after surgery.
- `hf-eval`, `hf-gate`, and sweep commands: behavior checks so shape-valid
  checkpoints are not promoted blindly.
- `native-vlm-demo`: tiny raw-patch plus text-token transformer for native
  early-fusion multimodal experiments.
- `hf-vision-graft`: train/evaluate a raw-patch image prefix directly against a
  Hugging Face causal LM parent; no external vision encoder is used.

## Roadmap

This repo starts with clean-room dense surgery. The next practical layers are:

- safetensors/Hugging Face adapters for transformer MLP width, attention heads,
  layer insertion, config updates, and dry-run shape reports;
- activation-calibrated merge/fold scoring and GRAIL-style ridge compensation;
- MergeKit-compatible recipe import/export for same-shape checkpoint merges;
- optional export hooks for vLLM/llm-compressor, NVIDIA ModelOpt, Intel Neural
  Compressor, NNCF, and ONNX pipelines;
- native multimodal and tokenizer-free growth sandboxes for embedding/vocab,
  patch-token, byte-token, and modality-tag expansion.

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Quick start

```powershell
neural-forge toy --output runs/a.npz --widths 8,32,4 --seed 1
neural-forge toy --output runs/b.npz --widths 8,48,4 --seed 2

neural-forge inspect runs/a.npz
neural-forge shrink runs/a.npz runs/a-small.npz --target-params 180
neural-forge expand runs/a-small.npz runs/a-wide.npz --target-params 420 --add-layers 1
neural-forge merge runs/merged.npz --models runs/a.npz runs/b.npz --target-params 260
```

The parameter target is best-effort. Some targets are mathematically impossible
for a fixed input/output shape; in that case the planner returns the closest
reachable hidden widths.

## Real Hugging Face Checkpoints

Install the optional HF stack before running these:

```powershell
py -3.11 -m pip install -e ".[hf]"
```

Same-shape text-model merge:

```powershell
neural-forge hf-merge runs/qwen-merged `
  --models Qwen/Qwen3-0.6B-Base Qwen/Qwen3-0.6B `
  --dtype float16
neural-forge hf-task-merge runs/qwen-task-alpha-075 `
  --base-model Qwen/Qwen3-0.6B-Base `
  --tuned-model Qwen/Qwen3-0.6B `
  --alpha 0.75
neural-forge hf-sweep-task runs/qwen-task-sweep `
  --base-model Qwen/Qwen3-0.6B-Base `
  --tuned-model Qwen/Qwen3-0.6B `
  --alphas 0.25,0.5,0.75,1.0
```

Create a blank/untrained target architecture for transplant tests:

```powershell
neural-forge hf-random-init runs/smollm2-135m-random `
  --config-model HuggingFaceTB/SmolLM2-135M `
  --tokenizer-model HuggingFaceTB/SmolLM2-135M `
  --seed 123 `
  --dtype float16
```

Fold a larger parent/donor directly into a blank child/target architecture:

```powershell
neural-forge hf-donor-project-init runs/qwen4b-into-smollm2 `
  --target-config-model HuggingFaceTB/SmolLM2-135M `
  --target-tokenizer-model HuggingFaceTB/SmolLM2-135M `
  --donor-model Qwen/Qwen3-4B-Base `
  --fold-mode mean `
  --activation-fold-assignment similarity `
  --donor-layer-fold-mode fold
```

`--donor-layer-fold-mode representative` preserves the old proportional-layer
sampling behavior. `average` and `fold` fold parent layer ranges into each child
layer before width folding; `fold` sums residual output projections and averages
other parameters.

Run cross-tokenizer recovery after a parent-to-child fold without adding a
saved adapter or bridge:

```powershell
neural-forge hf-xfamily-recover Qwen/Qwen3-4B-Base runs/qwen4b-into-smollm2 runs/qwen4b-into-smollm2-recovered `
  --anchor-kd-weight 0.3 `
  --span-kd-weight 0.2 `
  --span-max-pairs 256 `
  --lm-weight 0.5 `
  --first-token-lm-weight 0.0 `
  --response-end-lm-weight 0.0 `
  --focused-lm-weight 0.0 `
  --focused-projected-vocab-kd-weight 0.0 `
  --hidden-kd-weight 0.2 `
  --on-policy-hidden-kd-weight 0.1 `
  --hidden-layer last `
  --hidden-max-anchors 512 `
  --candidate-dtype float32
```

Hidden KD uses a temporary donor-hidden to child-hidden projection fitted from
shared decoded-token embedding rows. The exported checkpoint remains the child
architecture only.
Span KD uses tokenizer offset mappings to align source and child prediction
positions for exact decoded-token spans, then applies KL over the shared anchor
vocabulary at those positions. It also leaves no runtime adapter.
`--first-token-lm-weight` applies response-only CE to the first answer token for
cross-tokenizer recovery. It is useful as a narrow experiment, but validation has
shown it does not by itself repair arithmetic/code behavior after an aggressive
blank-target fold.
`--response-end-lm-weight` applies response-only CE to the final response token;
with `--append-eos`, that final token is the stop token. It is default-off
because validation showed strong EOS pressure can damage generation health.
`--focused-lm-weight` applies hard-label CE only to focused response spans
selected by `--token-focus` (`numbers`, `code`, `auto`, etc.). It is meant for
behavior recovery on answer/code spans after the donor has already been folded
into the child shape. `--focused-projected-vocab-kd-weight` applies projected
vocab KD to those same spans when source and child token counts align exactly.
`--on-policy-hidden-kd-weight` generates short greedy child rollouts during
recovery, runs both donor and child on those rollouts, and aligns hidden states
on matched decoded-token spans when possible. It is meant for cases where static
calibration PPL improves while greedy behavior remains unstable. The same flag
is available on `hf-xfamily-block-recover`, where it applies the rollout hidden
alignment to the currently trainable child block.
For block recovery, `--block-target delta` trains the residual update produced
by the child block against the folded donor-layer update instead of matching
absolute hidden states.
`hf-xfamily-block-recover --module-output-kd-weight` adds differentiable
attention/MLP output reconstruction for the currently trainable child block.
Use `--module-output-targets attention|mlp|both` and
`--module-output-transport paired|sinkhorn|both` to choose which submodule
residuals are matched and whether row matching is strict span-paired, soft
transported, or mixed.
`--projected-vocab-kd-weight` adds static span-aligned donor next-token KD
projected into the child vocabulary during block recovery. `--byte-kd-weight`
adds a tokenizer-neutral first-byte/other-bucket KD signal for the same aligned
prediction positions, which keeps digits, punctuation, spaces, and newlines in
the loss even when tokenizers disagree. `--byte-hidden-kd-weight` applies the
same donor byte targets directly to the active block hidden states through a
fixed byte probe built from the child output embeddings, so byte supervision can
reach the folded block without training or relying on the LM head.
`--train-output-head-with-kd` optionally
trains the output head with each block under the same layer behavior gate; this
is default-off because tied input/output heads also mutate token embeddings.
When a tied head is detected, the recovery report records
`output_head_training_warning`.
`--teacher-generate-lm-weight` lets the donor generate short completions from
the calibration prompts, re-tokenizes those completions with the child
tokenizer, and applies child-token LM loss while the current block and optional
output head are trainable. `--teacher-generate-hidden-kd-weight` applies the
same generated texts to span-aligned block hidden recovery. These are
default-off because donor-generated text can overfit small calibration prompts;
use behavior-gated runs to check whether they improve coherence rather than
only local loss.
`--save-best-layer-behavior` evaluates held-out behavior during each block and
keeps the best behavior checkpoint instead of the lowest local reconstruction
loss. This is slower, but it prevents PPL/local-loss selection from silently
discarding better intermediate generations.

`hf-xfamily-calibrate-head --head-calibration-mode mean-shift` is a safer
alternative to full head refitting: it shifts output rows along the observed
child hidden mean to reduce average projected-logit error. It is default-off;
validation showed that even small mean shifts can preserve PPL while damaging
generation health, so use it as an experiment rather than a default repair.

Task arithmetic defaults to `--dtype auto` and saves the base tokenizer after
verifying that the tuned tokenizer has the same vocabulary. Special-token map
differences are recorded in the merge report. Use an explicit dtype such as
`--dtype float16` when you want smaller/faster validation artifacts.

Merge-first FFN width fold or expansion for Qwen/Gemma/Llama-like decoder
blocks:

```powershell
neural-forge hf-resize-ffn Qwen/Qwen3-0.6B-Base runs/qwen-width-0995 `
  --ratio 0.995 `
  --assignment combined-cosine
neural-forge hf-repair-ffn runs/qwen-merged runs/qwen-merged-shrunk runs/qwen-merged-shrunk-repaired
```

Merge-first depth fold:

```powershell
neural-forge hf-merge-layers runs/qwen-width-0995 runs/qwen-width-depth `
  --groups 13+14 `
  --method average
```

Automated merge-first width/depth search:

```powershell
neural-forge hf-search-compress Qwen/Qwen3-0.6B-Base runs/qwen-search `
  --width-ratios 0.995 `
  --layer-groups 13+14 `
  --depth-methods average,fold `
  --target-shrink 0.028
```

Cross-family host/donor vocabulary transplant:

```powershell
neural-forge hf-sweep-xfamily runs/pythia-smollm2-xfamily `
  --host-model EleutherAI/pythia-160m `
  --donor-model HuggingFaceTB/SmolLM2-135M `
  --alphas 0.001,0.005,0.01,0.02
```

Cross-family MLP-channel transplant:

```powershell
neural-forge hf-sweep-xfamily-mlp runs/qwen-lfm25-mlp `
  --host-model Qwen/Qwen3-0.6B-Base `
  --donor-model LiquidAI/LFM2.5-230M `
  --alphas 0.0005,0.001 `
  --vocab-alpha 0 `
  --max-host-channels 128 `
  --max-donor-channels 128
```

Activation-calibrated channel assignment uses the same calibration prompts on
the host and donor, then matches MLP channels by observed response summaries
instead of weight cosine alone:

```powershell
neural-forge hf-sweep-xfamily-mlp runs/qwen4b-smollm-activation `
  --host-model HuggingFaceTB/SmolLM2-135M `
  --donor-model Qwen/Qwen3-4B-Base `
  --alphas 0.001,0.005 `
  --channel-assignment hybrid `
  --activation-calibration-file runs/calibration.jsonl `
  --activation-max-calibration-texts 64 `
  --activation-max-length 96 `
  --layer-alpha-mode similarity `
  --layer-alpha-min-scale 0.25 `
  --layer-alpha-max-scale 1.25 `
  --layer-alpha-similarity-floor 0.30 `
  --max-host-channels 512 `
  --max-donor-channels 1024
```

For real-model claims, run benchmark-suite task scores as well as `hf-eval`.
PPL alone is only a damage signal, not proof of intelligence retention or
transfer:

```powershell
python experiments/run_lm_eval_comparison.py `
  --output-dir runs/qwen4b-smollm-bench `
  --tasks arc_easy,hellaswag,piqa,winogrande `
  --limit 50 `
  --model host=HuggingFaceTB/SmolLM2-135M `
  --model donor=Qwen/Qwen3-4B-Base `
  --model folded=runs/qwen4b-smollm-activation/models/xfamily_mlp_alpha_0p0010
```

Add lightweight recovery after each cross-family MLP candidate:

```powershell
neural-forge hf-sweep-xfamily-mlp runs/qwen-lfm25-mlp-recovered `
  --host-model Qwen/Qwen3-0.6B-Base `
  --donor-model LiquidAI/LFM2.5-230M `
  --alphas 0.005 `
  --recover `
  --recover-steps 8 `
  --recover-train-scope last1 `
  --recover-calibration-file runs/calibration.jsonl `
  --recover-max-calibration-texts 512
```

Cross-family merge plus merge-first compression in one pipeline:

```powershell
neural-forge hf-xfamily-compress runs/qwen-lfm25-compressed `
  --host-model Qwen/Qwen3-0.6B-Base `
  --donor-model LiquidAI/LFM2.5-230M `
  --alpha 0.005 `
  --vocab-alpha 0 `
  --max-host-channels 256 `
  --max-donor-channels 256 `
  --max-anchors 30000 `
  --width-ratios 0.995 `
  --layer-groups 13+14,15+16 `
  --target-shrink 0.05 `
  --recover `
  --recover-only-failed `
  --recover-calibration-file runs/calibration.jsonl `
  --recover-max-calibration-texts 512
```

Behavior probes:

```powershell
neural-forge hf-eval Qwen/Qwen3-0.6B runs/qwen3-report.json
neural-forge hf-eval runs/qwen-merged-shrunk runs/qwen-merged-shrunk-report.json
neural-forge hf-gate runs/qwen-merged-report.json runs/qwen-merged-shrunk-report.json
neural-forge hf-sweep-ffn runs/qwen-merged runs/qwen-sweep `
  --baseline-report runs/qwen-merged-report.json `
  --ratios 0.995,1.05 `
  --repair-shrinks
neural-forge vision-probe HuggingFaceTB/SmolVLM2-256M-Video-Instruct runs/vision-report.json
neural-forge vision-chain runs/vision-chain-report.json `
  --vision-model HuggingFaceTB/SmolVLM2-256M-Video-Instruct `
  --text-model runs/qwen-merged
```

Broad behavior gates:

```powershell
neural-forge hf-build-eval-suite runs/broad-suite-1000.json `
  --behavior-count 1000 `
  --generation-count 200 `
  --choice-count 250 `
  --text-count 64 `
  --seed 20260628
neural-forge hf-eval runs/qwen-merged-shrunk runs/qwen-merged-shrunk-broad.json `
  --eval-suite runs/broad-suite-1000.json `
  --max-new-tokens 96
neural-forge hf-rank-evals runs/rank-broad.json runs/*-broad.json
```

The broad suite keeps the original smoke checks available while adding a
repeatable larger gate for compression experiments. `hf-eval --eval-suite`
records the suite counts in the report, and `hf-rank-evals` ranks by behavior
pass rate, behavior correct count, generation health, choice accuracy, size,
then PPL. This keeps a candidate with better greedy behavior above one that
only lowered perplexity.

Blank target control:

```powershell
neural-forge hf-random-init runs/random-smollm2-135m `
  --config-model HuggingFaceTB/SmolLM2-135M `
  --tokenizer-model HuggingFaceTB/SmolLM2-135M `
  --seed 20260627
neural-forge hf-sweep-xfamily-mlp runs/qwen4b-into-random-smollm2 `
  --host-model runs/random-smollm2-135m `
  --donor-model Qwen/Qwen3-4B-Base `
  --alphas 0.001 `
  --vocab-alpha 0.001 `
  --max-host-channels 128 `
  --max-donor-channels 256
```

Donor-projected blank target initialization folds donor layer tensors directly
into a target config shape instead of leaving the target mostly random:

```powershell
neural-forge hf-donor-project-init runs/qwen4b-projected-smollm2 `
  --target-config-model HuggingFaceTB/SmolLM2-135M `
  --target-tokenizer-model HuggingFaceTB/SmolLM2-135M `
  --donor-model Qwen/Qwen3-4B-Base `
  --max-anchors 30000 `
  --fold-mode mean `
  --donor-layer-fold-mode fold `
  --donor-layer-group-strategy activation-energy `
  --activation-calibration-file runs/calibration.jsonl `
  --activation-max-calibration-texts 8 `
  --activation-max-length 96
```

When activation calibration is provided, donor residual dimensions and MLP
intermediate channels are folded with activation-weighted bins instead of only
contiguous index groups. `--donor-layer-group-strategy activation-energy` also
partitions donor depth by calibrated residual-update energy before layer
folding, so a small child layer is not forced to represent a uniform parent
depth slice when the donor's changes are concentrated elsewhere. This is an
experimental initializer strategy, not a promoted default; it must be followed
by `hf-eval`, `hf-gate`, manual generation inspection, and benchmark slices
before claiming capability transfer.

`hf-gate` checks PPL ratio, choice-probe accuracy, behavior probes, generation
length, generation diversity, and prompt-specific generation relevance for the
built-in generation probes. PPL can be a hard gate or advisory telemetry via
`--ppl-gate-mode hard|advisory`; automated surgery selection uses behavior,
choice, and generation health before compression ratio, and uses PPL only after
those output-facing metrics tie. A random or broken model can produce long
diverse text, so length and unique-token ratio alone are not treated as
coherence evidence.

Recovery commands with `--save-best-behavior` snapshot the untouched input as
step `0`, then rank checkpoints by behavior probes, generation health, and only
then loss. This prevents a lower-loss recovery step from replacing a checkpoint
that still generates healthier text.

Native early-fusion sandbox:

```powershell
neural-forge native-vlm-demo runs/native-vlm-demo --device cuda:0 --steps 600
neural-forge hf-vision-graft runs/qwen-task-alpha-025 runs/qwen-native-graft `
  --steps 900 `
  --batch-size 16 `
  --eval-batches 8 `
  --device cuda:0 `
  --dtype float16 `
  --task-keys square_color,circle_color,digit
```

The graft command also accepts a compressed cross-family checkpoint as the
parent, for example the best output from `hf-xfamily-compress`:

```powershell
neural-forge hf-vision-graft runs/qwen-lfm25-compressed/compression/models/depth_fold_13-14_15-16 `
  runs/qwen-lfm25-compressed-native-graft `
  --steps 900 `
  --batch-size 16 `
  --eval-batches 8
```

This trains a tiny single-transformer model over raw image patches plus text
tokens on synthetic shape/color VQA. `hf-vision-graft` takes the next step: it
feeds raw image patch embeddings into an existing merged text checkpoint and
trains a tiny patch projection while leaving the parent LM frozen by default.
It is an experimental native multimodal path, separate from the transitional
`vision-chain`.

## Checkpoint format

The current format is a compressed NumPy `.npz` with:

- `__meta__`: JSON metadata.
- `W0`, `b0`, `W1`, `b1`, ...: dense layer weights and biases.

Layer weights use shape `(out_features, in_features)`.

## Current research map

See [research/2026-06-26-research.md](research/2026-06-26-research.md) for the
current 2025-2026 research summary and the next implementation targets.
