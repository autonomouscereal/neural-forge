"""End-to-end: r12 round with the cached-decode admissibility screen enabled.

Uses the LOCAL modified round module (localtest/src) over a byte-identical
collector.  Subsets the candidate batches so the run is cheap while still
containing the two trajectories that failed the gate.
"""
import json, sys, shutil, traceback
from pathlib import Path
import torch

LOCAL = Path("/home/cereal/nf-round-debug/localtest")
RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME)); sys.path.insert(0, str(LOCAL/"src"))

from experiments import run_lfm2_residual_attention_fold as residual
import neural_forge.hf_lfm2_circuit_trace_cache as tc
import neural_forge.hf_lfm2_on_policy_evidence_round as rnd
from neural_forge.hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact

print("round module from:", rnd.__file__)
print("collector from   :", tc.__file__)
print("collector digest :", tc.implementation_identity()["collector_source_sha256"])

LAB = Path("/home/cereal/model-surgery-lab")
PACK = LAB/"artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5"
BASE = PACK/"parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
DONOR = LAB/"artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
BRIDGE = LAB/"artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"
PARENT = LAB/"models/LFM2.5-230M"
CHILD = LAB/"artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12/published-child/child"
SEALED_ROUND = LAB/"artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12"
FIT = RUNTIME/"experiments/lfm2_on_policy_fit_prompts_v1.json"
MANUAL = RUNTIME/"experiments/lfm2_manual_review_reserved_digests_v1.json"
BRIDGE_ART = "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"
OUT = Path("/home/cereal/nf-round-debug/screened-round")
if OUT.exists(): shutil.rmtree(OUT)
(OUT/"round-000").mkdir(parents=True)
shutil.copy2(SEALED_ROUND/"round-000/generation-cache.json", OUT/"round-000/generation-cache.json")

SUBSET = 6
orig_batches = rnd._capture_batches_from_corpus
def subset(corpus, **kw):
    cands, rejected = orig_batches(corpus, **kw)
    print("full candidate count:", len(cands), "-> using first", SUBSET, flush=True)
    return cands[:SUBSET], rejected
rnd._capture_batches_from_corpus = subset

import importlib.util
spec = importlib.util.spec_from_file_location("r12run", RUNTIME/"experiments/run_r12_on_policy_round.py")
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)

base_cache = tc.open_lfm2_circuit_trace_cache(BASE, verify=False)
base_manifest = dict(base_cache.manifest); parent_fp = base_cache.parent_fingerprint
del base_cache
capture_config = r12._capture_config_from_sealed_manifest(base_manifest["capture"])
print("selection:", capture_config.as_dict()["selection_sha256"], flush=True)

device = torch.device("cuda:0")
parent, dt = r12._load_parent_matching_fingerprint(PARENT, device=device, expected_fingerprint=parent_fp, requested_dtype="float32")
print("parent dtype:", dt, flush=True)
r12._bind_sealed_execution_identity(parent, device=device, sealed_execution=base_manifest["execution"])
ptok = residual.load_local_tokenizer(DONOR)
bridge = load_strict_tokenizer_bridge_artifact(BRIDGE, donor_tokenizer=ptok,
    target_tokenizer=residual.load_local_tokenizer(BRIDGE), expected_artifact_sha256=BRIDGE_ART)

rollout = rnd.Lfm2OnPolicyRolloutConfig(
    max_prompts=128, max_new_tokens=8, max_context_tokens=512, seed=0,
    max_capture_batches=256, truncate_forced_decode_to_capture_budget=True,
    screen_cached_decode_admissibility=True)
print("rollout screened:", rollout.screen_cached_decode_admissibility, flush=True)

try:
    ex = rnd.run_lfm2_on_policy_evidence_round(
        child_artifact_directory=CHILD, parent_model=parent, parent_tokenizer=ptok,
        parent_fingerprint=parent_fp, fit_prompts=r12._load_fit_prompts(FIT),
        manual_prompt_manifest_path=MANUAL, base_cache_directory=BASE, output_root=OUT,
        capture_config=capture_config, round_index=0, max_rounds=2, rollout=rollout,
        previous_round_manifest_paths=(), child_device="cuda:0", child_dtype="float32",
        input_device=device, parent_tokenizer_name="parent", verify_base_cache=False,
        tokenizer_bridge=bridge, expected_tokenizer_bridge_artifact_sha256=BRIDGE_ART)
    print("ROUND SEALED. batches:", ex.batch_count, flush=True)
    adm = ex.manifest["capture"]["cached_decode_admissibility"]
    print("screened:", adm["screened"], "candidates:", adm["candidate_batch_count"],
          "admissible:", adm["admissible_batch_count"], "excluded:", adm["excluded_batch_count"])
    for e in adm["excluded"]:
        print("  EXCLUDED forced=", e["forced_decode_token_ids"],
              "phase=", e["first_inadmissible_phase"]["phase"],
              "max_abs_error=", e["first_inadmissible_phase"]["max_abs_error"],
              "violations=", e["first_inadmissible_phase"]["elementwise_tolerance_violations"])
    print("parent block:", json.dumps({k:v for k,v in ex.manifest["parent"].items() if "execut" in k or "screen" in k}, indent=1))
except Exception as exc:
    print("ROUND FAILED:", type(exc).__name__, exc, flush=True)
    traceback.print_exc()
