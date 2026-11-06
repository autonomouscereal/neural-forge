"""CPU-only teacher-forced agreement probe: LFM2.5-230M parent vs existing shared-vocab matrix children.

Read-only on every model directory.  No training, no GPU (CUDA_VISIBLE_DEVICES must be empty).
Text = parent responses: set A = the 16 frozen-review prompts with the parent's sealed verbatim
responses from a frozen-review dossier (first MAXA response tokens); set B = the 30 extension-bank-v2
prompts with the parent's own greedy continuation generated here on CPU (MAXB tokens).
Per scored position (response tokens only), child q vs parent p over the full 65,536 vocab:
top-1 agreement; top-k set overlap |topk(q) & topk(p)|/k for k in KS; parent-argmax-in-child-top-k
containment; KL(p||q); child NLL of the text token; child mass on parent's top-10 set.
Writes one JSON line per child to OUT (resumable: children already in OUT are skipped).
"""
import json, os, sys, time, math, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THREADS = int(os.environ.get("ORA_THREADS", "16"))
torch.set_num_threads(THREADS)
torch.set_grad_enabled(False)
assert not torch.cuda.is_available(), "refusing: a GPU is visible"

PARENT = "/home/cereal/model-surgery-lab/models/LFM2.5-230M"
MX = "/mnt/ai_data/neural-forge-matrix"
DOSSIER = f"{MX}/wave43/cells/ARCH.HD48/frozen-review-dossier.json"
BANK2 = sys.argv[1]
OUT = sys.argv[2]
CHILDREN = sys.argv[3:]
MAXA, MAXB = 384, 256
KS = (1, 5, 10, 20, 64)

tok = AutoTokenizer.from_pretrained(PARENT)


def load(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    m.eval()
    return m


def chat_ids(messages):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


parent = load(PARENT)
t0 = time.time()
SEQS = os.environ.get("ORA_SEQS", "seqs.json")
seqs = []  # (set, id, prompt_len, ids)
if os.path.exists(SEQS):
    seqs = [tuple(s) for s in json.load(open(SEQS))]
    print(f"[ora] loaded {len(seqs)} cached sequences from {SEQS}", flush=True)
dos = json.load(open(DOSSIER))
if not seqs:
    for r in dos["parent_responses"]:
        p = chat_ids(r["prompt_messages"])
        resp = tok(r["response_verbatim"], add_special_tokens=False)["input_ids"][:MAXA]
        seqs.append(("A", r["id"], len(p), p + resp))
    bank = json.load(open(BANK2))
    for r in bank["prompts"]:
        p = chat_ids(r["messages"])
        x = torch.tensor([p])
        g = parent.generate(x, max_new_tokens=MAXB, do_sample=False, attention_mask=torch.ones_like(x))
        seqs.append(("B", r["id"], len(p), g[0].tolist()))
    json.dump(seqs, open(SEQS, "w"))
print(f"[ora] {len(seqs)} sequences built in {time.time()-t0:.1f}s; scored positions "
      f"{sum(len(s[3]) - s[2] for s in seqs)}", flush=True)


def logprobs(model, ids, plen):
    x = torch.tensor([ids])
    lo = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float()
    # position t predicts token t+1; score predictions of response tokens: t in [plen-1, len-2]
    return torch.log_softmax(lo[plen - 1:len(ids) - 1], dim=-1)


t0 = time.time()
P = [logprobs(parent, s[3], s[2]) for s in seqs]
PTOP = [lp.topk(max(KS), dim=-1).indices for lp in P]
print(f"[ora] parent logprobs cached in {time.time()-t0:.1f}s", flush=True)
del parent

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["child"])
        except Exception:
            pass


def score(lq, lp, ptop, nxt):
    q = lq.exp()
    p = lp.exp()
    ctop = lq.topk(max(KS), dim=-1).indices
    out = {}
    for k in KS:
        a, b = ctop[:, :k], ptop[:, :k]
        inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float() / k
        out[f"top{k}_overlap"] = inter
    for k in (5, 10, 20):
        out[f"parent_top1_in_child_top{k}"] = (ctop[:, :k] == ptop[:, :1]).any(-1).float()
    out["kl_p_q"] = (p * (lp - lq)).sum(-1)
    out["nll_text"] = -lq.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
    out["parent_nll_text"] = -lp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
    out["child_mass_on_parent_top10"] = q.gather(-1, ptop[:, :10]).sum(-1)
    out["parent_mass_on_parent_top10"] = p.gather(-1, ptop[:, :10]).sum(-1)
    return out


for path in CHILDREN:
    name = os.path.basename(path.rstrip("/"))
    if name in done:
        continue
    t0 = time.time()
    try:
        cfg = json.load(open(os.path.join(path, "config.json")))
        if cfg.get("vocab_size") != 65536:
            raise RuntimeError(f"vocab {cfg.get('vocab_size')} != parent 65536")
        m = load(path)
        acc = {"A": {}, "B": {}}
        npos = {"A": 0, "B": 0}
        for s, lp, ptop in zip(seqs, P, PTOP):
            lq = logprobs(m, s[3], s[2])
            nxt = torch.tensor(s[3][s[2]:])
            sc = score(lq, lp, ptop, nxt)
            for k, v in sc.items():
                acc[s[0]][k] = acc[s[0]].get(k, 0.0) + float(v.sum())
            npos[s[0]] += lq.shape[0]
        del m
        res = {"child": name, "path": path, "n_pos": npos, "seconds": round(time.time() - t0, 1)}
        for st in ("A", "B"):
            res[st] = {k: v / npos[st] for k, v in acc[st].items()}
        res["AB"] = {k: (acc["A"][k] + acc["B"][k]) / (npos["A"] + npos["B"]) for k in acc["A"]}
    except Exception as e:  # recorded, not hidden
        res = {"child": name, "path": path, "error": repr(e)[:500], "seconds": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"[ora] {name} {res.get('seconds')}s {'ERR ' + res['error'][:120] if 'error' in res else 'top1 %.3f top10 %.3f kl %.3f' % (res['AB']['top1_overlap'], res['AB']['top10_overlap'], res['AB']['kl_p_q'])}", flush=True)
print("[ora] DONE", flush=True)
