from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments import run_lfm2_exact4x_causal_head_frontier as runner
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_state,
    fingerprint_model_state_entries,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact


source = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/"
    "d6f5e1c8/structural-first-effectful-a0005"
)
donor_path = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/"
    "eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
)
target_path = Path(
    "/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/"
    "v4096_modelcap_v2"
)
donor = load_lfm2_direct_local_tokenizer(donor_path)
target = load_lfm2_direct_local_tokenizer(target_path)
bridge = load_strict_tokenizer_bridge_artifact(
    target_path,
    donor_tokenizer=donor,
    target_tokenizer=target,
    expected_artifact_sha256=(
        "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
    ),
)
checkpoint_io = Lfm2DirectLocalCheckpointIO(
    tokenizer_bridge=bridge,
    device=torch.device("cuda:0"),
    dtype=torch.bfloat16,
)
model = checkpoint_io.load(source / "structural-body-checkpoint")
model.requires_grad_(False)
model.eval()
structural = json.loads(
    (source / "structural-interface-install.json").read_text(encoding="utf-8")
)
deferred = json.loads(
    (source / "deferred-final-w2-install.json").read_text(encoding="utf-8")
)
topology = runner._target_topology(model)
observed_topology = {
    key: topology[key] for key in runner.EXPECTED_CHILD_TOPOLOGY
}
input_weight = model.get_input_embeddings().weight
output_weight = model.get_output_embeddings().weight
body_state = fingerprint_model_state(model)
body_identity = fingerprint_lfm2_parent(model).identity_sha256
checks = {
    "unique_parameter_count": runner._unique_parameter_count(model),
    "expected_unique_parameter_count": runner.EXPECTED_CHILD_UNIQUE_PARAMETERS,
    "observed_topology": observed_topology,
    "expected_topology": runner.EXPECTED_CHILD_TOPOLOGY,
    "input_output_same_object": input_weight is output_weight,
    "input_output_values_equal": bool(
        torch.equal(
            input_weight.detach().to(device="cpu"),
            output_weight.detach().to(device="cpu"),
        )
    ),
    "structural_exact_target_unique_parameters": structural.get(
        "exact_target_unique_parameters"
    ),
    "installed_input_table_sha256": structural.get(
        "installed_input_table_sha256"
    ),
    "installed_output_baseline_sha256": structural.get(
        "installed_output_baseline_sha256"
    ),
    "body_state": body_state,
    "deferred_model_state_after": deferred.get("model_state_after_sha256"),
    "body_identity": body_identity,
    "training_steps": deferred.get("training_steps"),
    "optimizer_steps": deferred.get("optimizer_steps"),
    "backward_calls": deferred.get("backward_calls"),
}
print(json.dumps(checks, allow_nan=False, indent=2, sort_keys=True))

first_entries = dict(fingerprint_model_state_entries(model))
parameter_names = {name for name, _value in model.named_parameters()}
buffer_names = {name for name, _value in model.named_buffers()}
del model
torch.manual_seed(123456789)
torch.cuda.manual_seed_all(123456789)
second_model = checkpoint_io.load(source / "structural-body-checkpoint")
second_entries = dict(fingerprint_model_state_entries(second_model))
differences = []
for name in sorted(set(first_entries).union(second_entries)):
    if first_entries.get(name) == second_entries.get(name):
        continue
    differences.append(
        {
            "name": name,
            "kind": (
                "parameter"
                if name in parameter_names
                else "buffer"
                if name in buffer_names
                else "unknown"
            ),
            "first_sha256": first_entries.get(name),
            "second_sha256": second_entries.get(name),
        }
    )
print(
    json.dumps(
        {
            "first_entry_count": len(first_entries),
            "second_entry_count": len(second_entries),
            "different_entries": differences,
            "second_body_state": fingerprint_model_state(second_model),
        },
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
)
