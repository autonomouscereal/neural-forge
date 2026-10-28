set -euo pipefail

PY=/home/cereal/model-surgery-lab/venv/bin/python
OVERLAY=/home/cereal/model-surgery-lab/python_overlays_transformers_4_57_3

PYTHONPATH="$OVERLAY" "$PY" - <<'PY'
import inspect
import torch
import transformers
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

print("VERSIONS", torch.__version__, transformers.__version__, transformers.__file__)
print(inspect.getsource(Lfm2RotaryEmbedding))

config = transformers.Lfm2Config(
    vocab_size=4096,
    hidden_size=784,
    intermediate_size=567,
    num_hidden_layers=14,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=64,
    max_position_embeddings=128000,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    layer_types=[
        "conv", "conv", "full_attention", "conv", "full_attention", "conv",
        "full_attention", "conv", "full_attention", "conv", "full_attention",
        "conv", "full_attention", "conv",
    ],
)
rotary = Lfm2RotaryEmbedding(config)
for name, buffer in rotary.named_buffers(remove_duplicate=False):
    print(
        "BUFFER",
        name,
        str(buffer.dtype),
        str(buffer.device),
        tuple(buffer.shape),
        buffer.is_contiguous(),
        int(buffer.storage_offset()),
        int(buffer.untyped_storage().nbytes()),
        int(buffer.numel() * buffer.element_size()),
        int(buffer.untyped_storage().data_ptr()),
    )
PY
