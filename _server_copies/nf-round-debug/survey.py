import json, sys
from pathlib import Path
import torch
RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME/"src")); sys.path.insert(0, str(RUNTIME))
from experiments import run_lfm2_residual_attention_fold as residual
LAB = Path("/home/cereal/model-surgery-lab")
PARENT = LAB/"models/LFM2.5-230M"
dev = torch.device("cuda:0")
batches = json.load(open("/home/cereal/nf-round-debug/repro-round/capture-batches.json"))
m = residual.load_local_lfm2_model(PARENT, device=dev, dtype=torch.bfloat16)
m.config._attn_implementation = "eager"; m.eval()
ATOL, RTOL = 1.0, 0.2
rows = []
with torch.inference_mode():
    for b in batches:
        ids = torch.tensor([b["prompt_token_ids"]], dtype=torch.long, device=dev)
        mask = torch.ones_like(ids)
        fp = m(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        cp = m(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        past = cp.past_key_values
        n = int(ids.shape[1])
        s0 = max(0, n-256)
        fl = fp.logits[:, s0:, :].to(torch.float64).cpu(); cl = cp.logits[:, s0:, :].to(torch.float64).cpu()
        d = (cl-fl).abs()
        steps = [{"phase": "cached_prefill", "max": float(d.max()), "viol": int((d > (ATOL+RTOL*fl.abs())).sum())}]
        pref, pm = ids, mask
        for s, t in enumerate(b["forced_decode_token_ids"]):
            tt = torch.tensor([[t]], dtype=torch.long, device=dev)
            pref = torch.cat((pref, tt), dim=1)
            pm = torch.cat((pm, torch.ones((1,1), dtype=pm.dtype, device=dev)), dim=1)
            cs = m(input_ids=tt, attention_mask=pm, use_cache=True, past_key_values=past, return_dict=True)
            past = cs.past_key_values
            c = cs.logits[:, -1:, :].to(torch.float64).cpu()
            f = m(input_ids=pref, attention_mask=pm, use_cache=False, return_dict=True).logits[:, -1:, :].to(torch.float64).cpu()
            dd = (c-f).abs()
            steps.append({"phase": f"decode.{s:03d}", "max": float(dd.max()),
                          "viol": int((dd > (ATOL+RTOL*f.abs())).sum())})
        fail = [x["phase"] for x in steps if x["viol"] > 0]
        forced = b["forced_decode_token_ids"]
        rows.append({"index": b["index"], "truncated": b["truncated"],
                     "prompt_len": n, "forced": forced,
                     "eot_count": sum(1 for t in forced if t == 2),
                     "max_over_steps": max(x["max"] for x in steps),
                     "failing_phases": fail, "gate_fail": bool(fail), "steps": steps})
        r = rows[-1]
        print("batch", r["index"], "trunc=", r["truncated"], "eot=", r["eot_count"], "max=", round(r["max_over_steps"],6), "fail=", fail, flush=True)
Path("/home/cereal/nf-round-debug/survey.json").write_text(json.dumps(rows, indent=1))
nf = [r for r in rows if r["gate_fail"]]
print("=== SUMMARY ===")
print("total batches:", len(rows), "gate failures:", len(nf))
print("failing indices:", [r["index"] for r in nf])
print("truncated batches:", [r["index"] for r in rows if r["truncated"]])
print("truncated AND failing:", [r["index"] for r in rows if r["truncated"] and r["gate_fail"]])
import statistics
print("max_abs_error over all batches: max=", max(r["max_over_steps"] for r in rows))
byeot = {}
for r in rows: byeot.setdefault(r["eot_count"], [0,0])
for r in rows:
    byeot[r["eot_count"]][0]+=1
    byeot[r["eot_count"]][1]+= (1 if r["gate_fail"] else 0)
print("eot_count -> (batches, failures):", dict(sorted(byeot.items())))
