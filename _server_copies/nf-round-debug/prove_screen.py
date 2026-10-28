import json, sys
from pathlib import Path
import torch
LOCAL = Path("/home/cereal/nf-round-debug/localtest")
RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, str(LOCAL/"src"))
from experiments import run_lfm2_residual_attention_fold as residual
import neural_forge.hf_lfm2_circuit_trace_cache as tc
import neural_forge.hf_lfm2_on_policy_evidence_round as rnd
import importlib.util
LAB = Path("/home/cereal/model-surgery-lab")
BASE = LAB/"artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
spec = importlib.util.spec_from_file_location("r12run", RUNTIME/"experiments/run_r12_on_policy_round.py")
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)
bc = tc.open_lfm2_circuit_trace_cache(BASE, verify=False)
bm = dict(bc.manifest); pfp = bc.parent_fingerprint; del bc
cfg = r12._capture_config_from_sealed_manifest(bm["capture"])
dev = torch.device("cuda:0")
parent, dt = r12._load_parent_matching_fingerprint(LAB/"models/LFM2.5-230M", device=dev, expected_fingerprint=pfp, requested_dtype="float32")
r12._bind_sealed_execution_identity(parent, device=dev, sealed_execution=bm["execution"])
B = json.load(open("/home/cereal/nf-round-debug/repro-round/capture-batches.json"))
models = [{"input_ids": torch.tensor([b["prompt_token_ids"]], dtype=torch.long),
           "attention_mask": torch.ones((1, len(b["prompt_token_ids"])), dtype=torch.long),
           "forced_decode_ids": torch.tensor([b["forced_decode_token_ids"]], dtype=torch.long)} for b in B]
reports = rnd.screen_lfm2_cached_decode_admissibility(parent, models, config=cfg, input_device=dev)
bad = [r for r in reports if not r["admissible"]]
print("SCREEN over", len(reports), "real r12 batches: admissible", len(reports)-len(bad), "excluded", len(bad))
print("excluded batch indices:", [r["index"] for r in bad])
for r in bad:
    f = r["first_inadmissible_phase"]
    print("  batch", r["index"], "forced=", B[r["index"]]["forced_decode_token_ids"],
          "\n      phase=", f["phase"], "max_abs_error=", f["max_abs_error"],
          "violations=", f["elementwise_tolerance_violations"], "/", f["compared_elements"],
          "dtype=", f["logit_dtype"], "truncated=", B[r["index"]]["truncated"])
print("total screen parent forwards:", sum(r["parent_forward_calls"] for r in reports))
