from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoTokenizer


ROOT = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_complete_causal_evidence_20260720_a0003/parent/"
    "dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b"
)
TOKENIZER = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER,
    local_files_only=True,
    trust_remote_code=False,
)
for index in range(16):
    path = ROOT / "batches" / f"batch-{index:06d}" / "trace.safetensors"
    tensors = load_file(path, device="cpu")
    prompt_ids = tensors["input.input_ids"].reshape(-1).tolist()
    forced_ids = tensors["input.forced_decode_ids"].reshape(-1).tolist()
    print(
        json.dumps(
            {
                "batch_index": index,
                "prompt_ids": prompt_ids,
                "forced_ids": forced_ids,
                "prompt_text": tokenizer.decode(prompt_ids),
                "forced_text": tokenizer.decode(forced_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
