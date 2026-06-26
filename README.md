# Neural Forge

Neural Forge is a small, working toolkit for neural checkpoint shape surgery:
shrinking dense models with neuron merging, expanding models with function-preserving
width/depth growth, and merging multiple models into a requested target shape.

This first release is intentionally scoped to dense feed-forward checkpoints.
That is the right place to prove the mechanics because hidden neurons have an
explicit axis: a row in one layer and the matching column in the next. Transformer
MLP blocks and attention heads can be adapted onto the same primitives next.

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
```

Gated-MLP FFN shrink or expansion for Qwen/Gemma/Llama-like decoder blocks:

```powershell
neural-forge hf-resize-ffn runs/qwen-merged runs/qwen-merged-shrunk --ratio 0.95
```

Behavior probes:

```powershell
neural-forge hf-eval Qwen/Qwen3-0.6B runs/qwen3-report.json
neural-forge hf-eval runs/qwen-merged-shrunk runs/qwen-merged-shrunk-report.json
neural-forge vision-probe HuggingFaceTB/SmolVLM2-256M-Video-Instruct runs/vision-report.json
neural-forge vision-chain runs/vision-chain-report.json `
  --vision-model HuggingFaceTB/SmolVLM2-256M-Video-Instruct `
  --text-model runs/qwen-merged
```

## Checkpoint format

The current format is a compressed NumPy `.npz` with:

- `__meta__`: JSON metadata.
- `W0`, `b0`, `W1`, `b1`, ...: dense layer weights and biases.

Layer weights use shape `(out_features, in_features)`.

## Current research map

See [research/2026-06-26-research.md](research/2026-06-26-research.md) for the
current 2025-2026 research summary and the next implementation targets.
