set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/runtimes/neural-forge-07fafdb0
PY=/home/cereal/model-surgery-lab/venv/bin/python
RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_heterogeneous_20260729/neural-forge-07fafdb0-d04ae8f2/a0004-aggressive-h784-exact57420144/run.log
export PYTHONPATH="$ROOT/src:$ROOT"

echo "ARTIFACT_FILES"
find "$(dirname "$RUN")" -maxdepth 2 -type f -printf '%P\n' | sort

"$PY" - "$RUN" <<'PY'
import copy
import json
import sys
from pathlib import Path

import torch
import transformers
from neural_forge.hf_lfm2_dynamic_blank_materializer import (
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (
    restore_lfm2_heterogeneous_physical_layout,
)

run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[-1])
configuration = run["planned_target"]["configuration"]
descriptor = restore_lfm2_heterogeneous_physical_layout(
    json.loads(configuration["_neural_forge_lfm2_heterogeneous_layout_descriptor_json"])
)
first_intermediate = descriptor.layers[0].mlp_intermediate_size
first_attention = next(
    layer for layer in descriptor.layers if layer.layer_type == "full_attention"
)
config = transformers.Lfm2Config(
    architectures=["Lfm2ForCausalLM"],
    dtype="float32",
    vocab_size=descriptor.vocab_size,
    hidden_size=descriptor.hidden_size,
    intermediate_size=first_intermediate,
    num_hidden_layers=descriptor.layer_count,
    num_attention_heads=first_attention.query_heads,
    num_key_value_heads=first_attention.key_value_heads,
    head_dim=first_attention.head_dim,
    max_position_embeddings=descriptor.max_position_embeddings,
    initializer_range=0.02,
    norm_eps=descriptor.norm_eps,
    use_cache=descriptor.use_cache,
    pad_token_id=descriptor.pad_token_id,
    bos_token_id=descriptor.bos_token_id,
    eos_token_id=descriptor.eos_token_id,
    tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": descriptor.rope_theta},
    conv_bias=False,
    conv_L_cache=descriptor.conv_kernel_size,
    block_multiple_of=256,
    block_ffn_dim_multiplier=1.0,
    block_auto_adjust_ff_dim=False,
    layer_types=[layer.layer_type for layer in descriptor.layers],
)
config._name_or_path = ""
config._attn_implementation = descriptor.attention_implementation
config.head_dim = first_attention.head_dim
print("VERSIONS", torch.__version__, transformers.__version__)
print(
    "CONFIG",
    config.head_dim,
    config.rope_parameters,
    config.dtype,
    config._attn_implementation,
)
model = transformers.Lfm2ForCausalLM(copy.deepcopy(config))
config = model.config
print("MODEL", type(model), type(model.model.rotary_emb), model.model.rotary_emb.config is config)

parameter_storage_ids = {
    (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
    for parameter in model.parameters()
}
seen_objects = set()
seen_storages = set()
for name, buffer in model.named_buffers(remove_duplicate=False):
    expected = 1.0 / (
        descriptor.rope_theta
        ** (
            torch.arange(
                0,
                first_attention.head_dim,
                2,
                dtype=torch.int64,
                device=buffer.device,
            ).to(dtype=torch.float32)
            / first_attention.head_dim
        )
    )
    storage = buffer.untyped_storage()
    storage_id = (str(buffer.device), int(storage.data_ptr()))
    print(
        "BUFFER",
        name,
        {
            "exact_type": type(buffer) is torch.Tensor,
            "dtype": str(buffer.dtype),
            "device": str(buffer.device),
            "layout": str(buffer.layout),
            "quantized": bool(getattr(buffer, "is_quantized", False)),
            "shape": tuple(buffer.shape),
            "contiguous": buffer.is_contiguous(),
            "storage_offset": int(buffer.storage_offset()),
            "requires_grad": buffer.requires_grad,
            "equal": torch.equal(buffer.detach(), expected),
            "max_abs_diff": float((buffer.detach() - expected).abs().max()),
            "storage_nbytes": int(storage.nbytes()),
            "expected_nbytes": int(buffer.numel() * buffer.element_size()),
            "duplicate_object": id(buffer) in seen_objects,
            "duplicate_storage": storage_id in seen_storages,
            "parameter_storage_collision": storage_id in parameter_storage_ids,
            "data_ptr": storage_id[1],
        },
    )
    seen_objects.add(id(buffer))
    seen_storages.add(storage_id)

try:
    receipt = materialize_lfm2_dynamic_blank_child(
        model,
        config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    print("MATERIALIZED", receipt["receipt_sha256"])
except Exception as exc:
    print("MATERIALIZATION_ERROR", type(exc).__name__, str(exc))
    raise
PY
