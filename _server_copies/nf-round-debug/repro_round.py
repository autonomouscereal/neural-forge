"""Instrumented replay of the r12 on-policy round parent capture.

Read-only against every sealed artifact: the sealed round generation cache is
copied into scratch and the replay writes only under /home/cereal/nf-round-debug.
"""
import json, sys, shutil, argparse, traceback
from pathlib import Path
import torch

RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME / "src")); sys.path.insert(0, str(RUNTIME))

from experiments import run_lfm2_residual_attention_fold as residual
import neural_forge.hf_lfm2_circuit_trace_cache as tc
import neural_forge.hf_lfm2_on_policy_evidence_round as rnd
from neural_forge.hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact

LAB = Path("/home/cereal/model-surgery-lab")
PACK = LAB / "artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5"
BASE = PACK / "parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
DONOR = LAB / "artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
BRIDGE = LAB / "artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2"
PARENT = LAB / "models/LFM2.5-230M"
R12 = LAB / "artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12"
CHILD = R12 / "published-child/child"
SEALED_ROUND = LAB / "artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12"
FIT = RUNTIME / "experiments/lfm2_on_policy_fit_prompts_v1.json"
MANUAL = RUNTIME / "experiments/lfm2_manual_review_reserved_digests_v1.json"
SCRATCH = Path("/home/cereal/nf-round-debug")
BRIDGE_ART = "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(SCRATCH / "repro-round"))
ap.add_argument("--truncate", action="store_true", default=True)
ap.add_argument("--no-truncate", dest="truncate", action="store_false")
ap.add_argument("--max-new-tokens", type=int, default=8)
args = ap.parse_args()

OUT = Path(args.out)
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "round-000").mkdir(parents=True)
src_cache = SEALED_ROUND / "round-000/generation-cache.json"
if src_cache.exists():
    shutil.copy2(src_cache, OUT / "round-000/generation-cache.json")
    print("reused sealed generation cache", flush=True)

records = []
captured_batches = {"candidates": None}

orig_compare = tc._compare_logits
def traced_compare(torch_mod, cached, full, *, config, name):
    try:
        result = orig_compare(torch_mod, cached, full, config=config, name=name)
    except Exception as exc:
        c = cached.detach().to("cpu").to(torch_mod.float64)
        f = full.detach().to("cpu").to(torch_mod.float64)
        d = (c - f).abs()
        flat = d.reshape(-1)
        k = int(flat.argmax())
        tol = float(config.logit_atol) + float(config.logit_rtol) * float(f.reshape(-1)[k].abs())
        viol = (d > (float(config.logit_atol) + float(config.logit_rtol) * f.abs())).sum().item()
        records.append({"name": name, "status": "FAIL", "max_abs_error": float(flat.max()),
                        "argmax_flat": k, "cached_at_argmax": float(c.reshape(-1)[k]),
                        "full_at_argmax": float(f.reshape(-1)[k]), "tol_at_argmax": tol,
                        "violating_elements": int(viol), "shape": list(c.shape),
                        "cached_dtype": str(cached.dtype), "error": str(exc)})
        Path(args.out + "/compare-trace.json").write_text(json.dumps(records, indent=1))
        raise
    c = cached.detach().to("cpu").to(torch_mod.float64)
    f = full.detach().to("cpu").to(torch_mod.float64)
    records.append({"name": name, "status": "ok", "max_abs_error": float((c-f).abs().max()),
                    "cached_dtype": str(cached.dtype), "shape": list(c.shape)})
    return result
tc._compare_logits = traced_compare
rnd._compare_logits = traced_compare if hasattr(rnd, "_compare_logits") else None

orig_batches = rnd._capture_batches_from_corpus
def traced_batches(corpus, **kw):
    cands, rejected = orig_batches(corpus, **kw)
    captured_batches["candidates"] = cands
    dump = [{"index": i, "prompt_token_ids": list(b.prompt_token_ids),
             "forced_decode_token_ids": list(b.forced_decode_token_ids),
             "target_prompt_token_ids": list(b.target_prompt_token_ids),
             "target_forced_decode_token_ids": list(b.target_forced_decode_token_ids),
             "truncated": bool(b.truncated), "trajectory_sha256": b.trajectory_sha256,
             "origin_prompt_indices": list(b.origin_prompt_indices)}
            for i, b in enumerate(cands)]
    Path(args.out + "/capture-batches.json").write_text(json.dumps(dump, indent=1))
    Path(args.out + "/rejections.json").write_text(json.dumps(rejected, indent=1))
    print(f"capture batches: {len(cands)} candidates, {len(rejected)} rejections", flush=True)
    return cands, rejected
rnd._capture_batches_from_corpus = traced_batches

base_cache = tc.open_lfm2_circuit_trace_cache(BASE, verify=False)
base_manifest = dict(base_cache.manifest)
parent_fp = base_cache.parent_fingerprint
del base_cache
capture_block = base_manifest["capture"]

sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("r12run", RUNTIME / "experiments/run_r12_on_policy_round.py")
r12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)

capture_config = r12._capture_config_from_sealed_manifest(capture_block)
print("capture selection:", capture_config.as_dict()["selection_sha256"], flush=True)
print("logit_rtol", capture_config.logit_rtol, "logit_atol", capture_config.logit_atol,
      "max_decode_steps", capture_config.max_decode_steps, flush=True)

prompts = r12._load_fit_prompts(FIT)
rollout = rnd.Lfm2OnPolicyRolloutConfig(
    max_prompts=128, max_new_tokens=args.max_new_tokens, max_context_tokens=512, seed=0,
    max_capture_batches=256, truncate_forced_decode_to_capture_budget=args.truncate)

device = torch.device("cuda:0")
parent, parent_dtype = r12._load_parent_matching_fingerprint(
    PARENT, device=device, expected_fingerprint=parent_fp, requested_dtype="float32")
print("parent_load_dtype:", parent_dtype, "param dtype:", next(parent.parameters()).dtype, flush=True)
execution = r12._bind_sealed_execution_identity(parent, device=device, sealed_execution=base_manifest["execution"])
parent_tok = residual.load_local_tokenizer(DONOR)
bridge = load_strict_tokenizer_bridge_artifact(
    BRIDGE, donor_tokenizer=parent_tok, target_tokenizer=residual.load_local_tokenizer(BRIDGE),
    expected_artifact_sha256=BRIDGE_ART)

try:
    executed = rnd.run_lfm2_on_policy_evidence_round(
        child_artifact_directory=CHILD, parent_model=parent, parent_tokenizer=parent_tok,
        parent_fingerprint=parent_fp, fit_prompts=prompts, manual_prompt_manifest_path=MANUAL,
        base_cache_directory=BASE, output_root=OUT, capture_config=capture_config,
        round_index=0, max_rounds=2, rollout=rollout, previous_round_manifest_paths=(),
        child_device="cuda:0", child_dtype="float32", input_device=device,
        parent_tokenizer_name="parent", verify_base_cache=False,
        tokenizer_bridge=bridge, expected_tokenizer_bridge_artifact_sha256=BRIDGE_ART)
    print("ROUND SEALED", executed.extension_cache_key, flush=True)
except Exception as exc:
    print("ROUND FAILED:", type(exc).__name__, exc, flush=True)
    traceback.print_exc()
finally:
    Path(args.out + "/compare-trace.json").write_text(json.dumps(records, indent=1))
    print("comparisons recorded:", len(records), flush=True)
