"""Sensitivity: ADDENDUM 6 verdict words (alive-parent-band/alive-degraded/impaired vs dead) as the outcome."""
import json, re, collections, sys
sys.path.insert(0, ".")
from analyze_stats import auc, spearman
d = json.load(open("dump.json", encoding="utf-8"))
R = json.load(open("results.json"))
arm = {a["experiment_id"]: a for a in d["arm"]}
def cid(a):
    k = re.sub(r"^MX\.", "", a["arm_key"] or ""); return re.split(r"[@#]", k)[0].upper()
ver = collections.defaultdict(list)
for b in d["beh"]:
    t = (b["text"] or "") + " " + (b["score"] or "")
    m = re.search(r"verdict[:\s]+(alive-parent-band|alive-degraded|impaired|dead)", t)
    if m and b["scorer"] in ("claude", "codex", "mixed", "cyril"):
        a = arm[b["experiment_id"]]; ver[(a["program"], cid(a))].append(m.group(1))
lab = {k: (0 if collections.Counter(v).most_common(1)[0][0] == "dead" else 1) for k, v in ver.items()}
rows = {(r["program"], r["child"]): r for r in R["rows"]}
DIR = {"free_run_top1_match": 1, "mean_lcp_h64": 1, "drift_vs_control": -1, "ppl_lambada": -1, "ppl_wikitext": -1,
       "terminal_cos_vs_parent": 1, "terminal_relL2_vs_parent": -1, "heldout_local_relL2": -1, "write_heldout_cos": 1}
print("children with a verdict:", len(lab), collections.Counter(lab.values()))
dead_counts = [round(rows[k]["beh"] * 15) for k, l in lab.items() if l == 0 and k in rows]
print("x/15 counts of verdict-dead children:", collections.Counter(dead_counts))
out = {}
for f, s in DIR.items():
    pos = [rows[k]["met"][f] * s for k, l in lab.items() if l == 1 and k in rows and f in rows[k]["met"]]
    neg = [rows[k]["met"][f] * s for k, l in lab.items() if l == 0 and k in rows and f in rows[k]["met"]]
    if len(pos) >= 3 and len(neg) >= 3:
        out[f] = {"auc": auc(pos, neg), "n_alive": len(pos), "n_dead": len(neg)}
        print(f"  {f:26s} AUC={out[f]['auc']:.3f} alive={len(pos)} dead={len(neg)}")
# agreement between verdict and x/15 count
both = [(lab[k], rows[k]["beh"]) for k in lab if k in rows]
print("AUC of the x/15 count itself vs verdict:", auc([b for l, b in both if l == 1], [b for l, b in both if l == 0]))
json.dump({"n": len(lab), "dead_count_hist": dict(collections.Counter(dead_counts)), "auc": out}, open("verdict_sens.json", "w"), indent=1)
