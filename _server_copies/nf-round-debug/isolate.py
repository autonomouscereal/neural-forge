import json, sys
from pathlib import Path
import torch
RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME/"src")); sys.path.insert(0, str(RUNTIME))
from experiments import run_lfm2_residual_attention_fold as residual
LAB = Path("/home/cereal/model-surgery-lab")
PARENT = LAB/"models/LFM2.5-230M"
DONOR = LAB/"artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46"
b = json.load(open("/home/cereal/nf-round-debug/repro-round/capture-batches.json"))[0]
prompt = b["prompt_token_ids"]; forced = b["forced_decode_token_ids"]
tok = residual.load_local_tokenizer(DONOR)
print("PROMPT TEXT:", repr(tok.decode(prompt)))
print("FORCED tokens:", [(i, repr(tok.convert_ids_to_tokens([i])[0])) for i in forced])
print("id2:", repr(tok.convert_ids_to_tokens([2])[0]), "| special ids:", tok.all_special_ids if hasattr(tok,"all_special_ids") else None)
dev = torch.device("cuda:0")

def run(prompt_ids, forced_ids, dtype, label):
    m = residual.load_local_lfm2_model(PARENT, device=dev, dtype=dtype)
    m.config._attn_implementation = "eager"
    m.eval()
    out = []
    with torch.inference_mode():
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
        mask = torch.ones_like(ids)
        r = m(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        past = r.past_key_values
        pref = ids; pm = mask
        for s, t in enumerate(forced_ids):
            tt = torch.tensor([[t]], dtype=torch.long, device=dev)
            pref = torch.cat((pref, tt), dim=1)
            pm = torch.cat((pm, torch.ones((1,1), dtype=pm.dtype, device=dev)), dim=1)
            cs = m(input_ids=tt, attention_mask=pm, use_cache=True, past_key_values=past, return_dict=True)
            past = cs.past_key_values
            cl = cs.logits[:, -1:, :].to(torch.float64).cpu()
            fs = m(input_ids=pref, attention_mask=pm, use_cache=False, return_dict=True)
            fl = fs.logits[:, -1:, :].to(torch.float64).cpu()
            d = (cl-fl).abs()
            viol = int((d > (1.0 + 0.2*fl.abs())).sum())
            out.append((s, float(d.max()), viol))
    del m; torch.cuda.empty_cache()
    print(f"--- {label} ({dtype}) ---")
    for s, mx, v in out:
        flag = "  <== GATE FAIL" if v > 0 else ""
        print(f"  decode.{s:03d} max_abs_err={mx:.6g} violators={v}{flag}")
    return out

run(prompt, forced, torch.bfloat16, "batch0 exact (degenerate)")
run(prompt, forced, torch.float32, "batch0 exact (degenerate)")
nondeg = [816, 1000, 1361, 3771, 810, 1112, 1014, 1842]
run(prompt, nondeg, torch.bfloat16, "batch0 prompt + NON-degenerate forced")
run(prompt, [2]*8, torch.bfloat16, "batch0 prompt + 8x token2")
