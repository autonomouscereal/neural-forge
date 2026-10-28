set -euo pipefail

PY=/home/cereal/model-surgery-lab/venv/bin/python

echo "REMOTE_HITS"
rg -n --max-filesize 10M -g '*.py' -g '*.json' -g '*.jsonl' -g '*.log' -g '*.txt' \
  '_HeterogeneousLfm2ForCausalLM|Heterogeneous rotary buffer model\.rotary_emb\.inv_freq' \
  /home/cereal/model-surgery-lab 2>/dev/null || true

"$PY" - <<'PY'
import inspect
import torch
import transformers
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

print("VERSIONS", torch.__version__, transformers.__version__)
print(inspect.getsource(Lfm2RotaryEmbedding))
PY
