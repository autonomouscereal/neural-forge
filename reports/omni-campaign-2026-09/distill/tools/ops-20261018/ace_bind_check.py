"""Real-shard check of media_full_kd_trainer 1.3.3 bind_ace_codes with stand-in hybrid/music objects (the binding's
index logic only; the loss itself is checked by the trainer's own selftest music parity)."""
import importlib.util, json, os, sys, torch
from safetensors.torch import load_file
tool, D, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
names = sys.argv[4:]
spec = importlib.util.spec_from_file_location("mfk", tool); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class H: music_band_begin = 252_416
class Mu:
    codes = 64_000
    def head_loss(self, positions, ids, lp): return ("head", positions, ids, lp)
    def input_grad(self, ids): return ("grad", ids)
tok = m.student_tokenizer(tok_path)
cfg = json.load(open("/mnt/lexar/omni-campaign/distill/media_m1/media_data_v4.json"))
report = []
for n in names:
    side = json.load(open(os.path.join(D, n)))
    stored = load_file(os.path.join(D, n[:-5] + ".safetensors"))
    out = m.bind_ace_codes(H(), Mu(), {"name": n}, side, stored, tok, cfg["ace_node_defaults"], top_k=int(stored["topk_ids"].shape[1]))
    ids = out["segment"]["ids"][0]; p = out["segment"]["prompt_tokens"]
    _, pos, oid, olp = out["head_loss"]
    code_rows = ids[p:] - H.music_band_begin
    mass = torch.logsumexp(olp, dim=-1)
    report.append({"name": n, "prompt_tokens": p, "codes": out["binding"]["codes"],
                   "seq_equals_stored": bool(torch.equal(code_rows, stored["tokens"].long())),
                   "prefix_decodes_caption": (side["item"]["caption"][:30] in tok.decode(ids[:p].tolist())),
                   "positions_ok": int(pos[0]) == p - 1 and int(pos[-1]) == p - 2 + code_rows.numel(),
                   "ids_in_range": bool(((oid >= 0) & (oid < 64000)).all()),
                   "lp_finite": bool(torch.isfinite(olp).all()),
                   "band_mass_top64_min_max": [round(float(mass.min().exp()), 4), round(float(mass.max().exp()), 4)],
                   "rows_renormalised": out["binding"]["rows_renormalised"], "sampled_in_top64": round(float((oid == code_rows[:, None]).any(1).float().mean()), 3)})
ok = all(r["seq_equals_stored"] and r["prefix_decodes_caption"] and r["positions_ok"] and r["ids_in_range"] and r["lp_finite"]
         and r["band_mass_top64_min_max"][1] <= 1.000001 for r in report)
print(json.dumps({"pass": ok, "checked": report}, indent=1))
sys.exit(0 if ok else 3)
