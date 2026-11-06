"""Second pass over results.json rows: alive-band subsets, paired rho differences, record-only tables."""
import json, random, re, statistics, collections, math, sys
sys.path.insert(0, ".")
from analyze_stats import spearman, auc, boot_paired_diff, boot_ci

R = json.load(open("results.json"))
rows = R["rows"]
DIR = {"free_run_top1_match": 1, "mean_lcp_h64": 1, "drift_vs_control": -1, "ppl_lambada": -1, "ppl_wikitext": -1,
       "terminal_cos_vs_parent": 1, "terminal_relL2_vs_parent": -1, "heldout_local_relL2": -1,
       "write_heldout_cos": 1, "worst_site_cos": 1, "bench_acc": 1, "retained_energy": 1}
out = {}


def table(sub, label):
    res = []
    for f, d in DIR.items():
        pts = [(r["met"][f] * d, r["beh"]) for r in sub if f in r["met"]]
        if len(pts) < 8:
            continue
        x, y = zip(*pts)
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        res.append({"family": f, "n": len(x), "rho": spearman(list(x), list(y)), "ci": boot_ci(list(x), list(y), spearman)})
    out[label] = res
    print("==", label)
    for t in sorted(res, key=lambda t: -t["rho"]):
        print(f"   {t['family']:26s} n={t['n']:4d} rho={t['rho']:.3f} ci=[{t['ci'][0]:.3f},{t['ci'][1]:.3f}]")


mx = [r for r in rows if r["program"] == "matrix-campaign"]
table([r for r in mx if r["beh"] >= 3 / 15 - 1e-9], "matrix alive band (>=3/15)")
table([r for r in mx if r["beh"] >= 5 / 15 - 1e-9], "matrix alive (>=5/15)")
table([r for r in mx if r["child"].startswith("ARCH.")], "matrix ARCH children only")
table([r for r in mx if re.match(r"^(C\d\d\.|COMP\.)", r["child"])], "matrix cells+COMP (parent-shaped)")

# paired: free-run vs lcp vs drift on the same children
print("== paired differences (same children)")
pairs = [("mean_lcp_h64", "free_run_top1_match"), ("ppl_lambada", "free_run_top1_match"),
         ("drift_vs_control", "free_run_top1_match"), ("terminal_cos_vs_parent", "free_run_top1_match"),
         ("write_heldout_cos", "free_run_top1_match")]
pr = []
for a, b in pairs:
    for label, sub in (("all matrix", mx), ("alive band", [r for r in mx if r["beh"] >= 3 / 15 - 1e-9])):
        pts = [(r["met"][a] * DIR[a], r["met"][b] * DIR[b], r["beh"]) for r in sub if a in r["met"] and b in r["met"]]
        if len(pts) < 8:
            continue
        xa, xb, y = map(list, zip(*pts))
        ra, rb = spearman(xa, y), spearman(xb, y)
        lo, hi = boot_paired_diff(xa, xb, y)
        pr.append({"a": a, "b": b, "subset": label, "n": len(y), "rho_a": ra, "rho_b": rb, "diff": ra - rb, "ci": [lo, hi]})
        print(f"   {label:11s} {a} vs {b}: n={len(y)} {ra:.3f} vs {rb:.3f} diff={ra-rb:+.3f} ci=[{lo:+.3f},{hi:+.3f}]")
out["paired"] = pr

# matrix per-site technique table (equal ratio, single component installed)
cells = collections.defaultdict(dict)
for r in rows:
    m = re.match(r"^(C\d\d)\.(T\d+)\.(S\d+|M\d+)\.(L\d\d)$", r["child"])
    if m:
        c, t, lvl, s = m.groups()
        cells[(c, lvl, s)][t] = r["beh"] * 15
dev = collections.defaultdict(list)
wins = collections.Counter()
for k, v in cells.items():
    if len(v) < 3 or k[1] != "S4":
        continue
    mu = statistics.mean(v.values())
    best = max(v.values())
    for t, s in v.items():
        dev[t].append(s - mu)
        if s == best:
            wins[t] += 1
out["cells"] = {"/".join(k): v for k, v in cells.items() if len(v) >= 2}
out["cell_dev"] = {t: {"n_sites": len(d), "mean_dev": statistics.mean(d), "wins_or_ties": wins[t]} for t, d in dev.items()}
print("== S4 per-site technique deviation from site mean (cases of 15)")
for t, d in sorted(out["cell_dev"].items(), key=lambda kv: -kv[1]["mean_dev"]):
    print(f"   {t}: sites={d['n_sites']} mean_dev={d['mean_dev']:+.2f} best-or-tied={d['wins_or_ties']}")

# sealed vs robust (wave 33)
sealed = [12, 11, 11, 10, 8, 8, 7, 5]
robust = [6, 2, 2, 3, 2, 0, 1, 2]
out["sealed_vs_robust_rho"] = spearman(sealed, robust)
print("== sealed vs robust rho (n=8):", round(out["sealed_vs_robust_rho"], 3))
json.dump(out, open("results2.json", "w"), indent=1)
