import json
import sys
from pathlib import Path

import transformers


sys.path.insert(0, "experiments")
import run_qwen_compensated_neuron_fold as runner


model = Path(
    "/home/cereal/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Base/snapshots/"
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
selection = runner._selected_suite(
    Path("runs/qwen3_compensated_selection_20260710.json"),
    max_text_evals=32,
    max_choice_probes=128,
    max_generation_prompts=256,
    max_behavior_probes=256,
)
certification = runner._selected_suite(
    Path(
        "/home/cereal/model-surgery-lab/neural-forge-src/runs/"
        "qwen4b_to_smollm135m_transport_random_20260627/"
        "behavior_suite_exactgen_20260702.json"
    ),
    max_text_evals=1000,
    max_choice_probes=250,
    max_generation_prompts=1000,
    max_behavior_probes=1000,
)
print(
    json.dumps(
        runner.verify_selection_certification_disjoint(
            tokenizer,
            selection,
            certification,
            max_length=160,
        ),
        sort_keys=True,
    )
)
