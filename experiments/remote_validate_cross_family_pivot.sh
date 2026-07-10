#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/neural-forge-pivot
PY=/home/cereal/model-surgery-lab/venv/bin/python
export PYTHONPATH="$ROOT/src"

"$PY" -m py_compile \
  "$ROOT/experiments/launch_cross_family_blank_run.py" \
  "$ROOT/src/neural_forge/hf_blank_bootstrap.py" \
  "$ROOT/src/neural_forge/hf_blank_checkpoint.py" \
  "$ROOT/src/neural_forge/hf_targeted_trace.py"

"$PY" - <<'PY'
import torch
import transformers

from neural_forge.hf_arch_layout import build_decoder_layout
from neural_forge.hf_blank_checkpoint import bootstrap_blank_decoder_checkpoint
from neural_forge.hf_targeted_trace import collect_hf_targeted_trace

config = transformers.LlamaConfig(
    vocab_size=151936,
    hidden_size=384,
    intermediate_size=768,
    num_hidden_layers=24,
    num_attention_heads=6,
    num_key_value_heads=1,
    head_dim=64,
    max_position_embeddings=40960,
    tie_word_embeddings=False,
    attention_bias=False,
    mlp_bias=False,
)
model = transformers.LlamaForCausalLM(config)
layout = build_decoder_layout(model)
count = sum(parameter.numel() for parameter in model.parameters())
assert count == 146196864, count
assert layout.layer_count == 24
assert callable(bootstrap_blank_decoder_checkpoint)
assert callable(collect_hf_targeted_trace)
print(
    {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "target_params": count,
        "target_family": layout.family,
    }
)
PY
