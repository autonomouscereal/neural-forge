import json, sys
from pathlib import Path
import torch
RUNTIME = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a")
sys.path.insert(0, str(RUNTIME/"src")); sys.path.insert(0, str(RUNTIME))
from experiments import run_lfm2_residual_attention_fold as residual
LAB = Path("/home/cereal/model-surgery-lab")
dev = torch.device("cuda:0")
m = residual.load_local_lfm2_model(LAB/"models/LFM2.5-230M", device=dev, dtype=torch.bfloat16)
m.config._attn_implementation = "eager"; m.eval()
cfg = m.config
print("layer_types:", getattr(cfg, "layer_types", None))
print("full_attn_idxs:", getattr(cfg, "full_attn_idxs", None))
layers = m.model.layers
print("n_layers:", len(layers), "layer0 children:", [n for n,_ in layers[0].named_children()])

B = json.load(open("/home/cereal/nf-round-debug/repro-round/capture-batches.json"))
grabs = {}
def mk(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grabs[i] = h.detach()[:, -1:, :].to(torch.float64).cpu()
    return hook

def probe(bi, step_index):
    b = B[bi]
    prompt = b["prompt_token_ids"]; forced = b["forced_decode_token_ids"]
    handles = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    with torch.inference_mode():
        ids = torch.tensor([prompt], dtype=torch.long, device=dev)
        mask = torch.ones_like(ids)
        past = m(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True).past_key_values
        pref, pm = ids, mask
        for s in range(step_index+1):
            tt = torch.tensor([[forced[s]]], dtype=torch.long, device=dev)
            pref = torch.cat((pref, tt), dim=1)
            pm = torch.cat((pm, torch.ones((1,1), dtype=pm.dtype, device=dev)), dim=1)
            grabs.clear()
            cs = m(input_ids=tt, attention_mask=pm, use_cache=True, past_key_values=past, return_dict=True)
            past = cs.past_key_values
            cached_layers = dict(grabs)
            grabs.clear()
            m(input_ids=pref, attention_mask=pm, use_cache=False, return_dict=True)
            full_layers = dict(grabs)
        for h in handles: h.remove()
    print("--- batch", bi, "decode." + str(step_index).zfill(3), "token", forced[step_index], "---")
    print("  layer | max|cached-full| | max|full| | rel")
    for i in range(len(layers)):
        c = cached_layers[i]; f = full_layers[i]
        d = float((c-f).abs().max()); mag = float(f.abs().max())
        print("   {:2d}   | {:14.6g} | {:9.6g} | {:.3g}".format(i, d, mag, d/max(mag,1e-9)))
    return cached_layers, full_layers

probe(0, 0)
probe(0, 1)
probe(109, 1)
