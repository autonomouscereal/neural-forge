"""Join the CPU top-k probe (probe.jsonl) with human frozen-15 reads (results.json rows) and compare
top-1 against top-k / KL / NLL / matched mass as predictors of behaviour on the SAME children."""
import json, sys, collections, statistics, math
sys.path.insert(0, ".")
from analyze_stats import spearman, auc, boot_ci, boot_paired_diff, boot_paired_auc_diff

PROBE = sys.argv[1] if len(sys.argv) > 1 else "probe.jsonl"
R = json.load(open("results.json"))
beh = {r["child"]: r for r in R["rows"] if r["program"] == "matrix-campaign"}

probe = [json.loads(l) for l in open(PROBE) if l.strip()]
errs = [p for p in probe if "error" in p]
ok = [p for p in probe if "error" not in p]
tags = collections.defaultdict(list)
for p in ok:
    tags[p["child"].split("-", 1)[1].upper()].append(p)
dups = {t for t, v in tags.items() if len(v) > 1}

METRICS = [  # (name, direction)
    ("top1_overlap", 1), ("top5_overlap", 1), ("top10_overlap", 1), ("top20_overlap", 1), ("top64_overlap", 1),
    ("parent_top1_in_child_top5", 1), ("parent_top1_in_child_top10", 1), ("parent_top1_in_child_top20", 1),
    ("kl_p_q", -1), ("nll_text", -1), ("child_mass_on_parent_top10", 1),
]
DBM = [("free_run_top1_match", 1), ("mean_lcp_h64", 1), ("drift_vs_control", -1), ("ppl_lambada", -1),
       ("terminal_cos_vs_parent", 1)]

joined = []
for t, ps in tags.items():
    if t in dups:
        continue
    b = beh.get("ARCH." + t)
    if b is None:
        continue
    joined.append({"tag": t, "beh": b["beh"], "n_reads": b["n_reads"], "probe": ps[0], "db": b["met"],
                   "ratio": b["ratio_net"]})
print(f"probe children ok {len(ok)}, errors {len(errs)}, duplicate tags excluded {sorted(dups)}, joined {len(joined)}")
for e in errs:
    print("  ERR", e["child"], e["error"][:150])

out = {"n_joined": len(joined), "errors": errs, "dups": sorted(dups), "sets": {}}
for st in ("AB", "A", "B"):
    print(f"\n== text set {st} (n={len(joined)}); alive >=5/15, dead <=2/15")
    y = [j["beh"] for j in joined]
    lab = [1 if v >= 5 / 15 - 1e-9 else 0 if v <= 2 / 15 + 1e-9 else None for v in y]
    res = {}
    for m, d in METRICS:
        x = [j["probe"][st][m] * d for j in joined]
        rho = spearman(x, y)
        ci = boot_ci(x, y, spearman)
        pos = [xx for xx, l in zip(x, lab) if l == 1]
        neg = [xx for xx, l in zip(x, lab) if l == 0]
        a = auc(pos, neg)
        # within the alive band only (>=3/15)
        xa = [xx for xx, yy in zip(x, y) if yy >= 3 / 15 - 1e-9]
        ya = [yy for yy in y if yy >= 3 / 15 - 1e-9]
        rho_alive = spearman(xa, ya) if len(set(ya)) > 1 else float("nan")
        res[m] = {"rho": rho, "ci": ci, "auc": a, "n_alive": len(pos), "n_dead": len(neg), "rho_alive_band": rho_alive,
                  "n_alive_band": len(ya), "mean": statistics.mean(j["probe"][st][m] for j in joined)}
        print(f"   {m:28s} mean={res[m]['mean']:.3f} rho={rho:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] AUC={a:.3f} "
              f"(A{len(pos)}/D{len(neg)}) rho_alive_band={rho_alive:+.3f} (n={len(ya)})")
    # paired differences against top-1
    x1 = [j["probe"][st]["top1_overlap"] for j in joined]
    lab01 = [l for l in lab]
    diffs = {}
    for m, d in METRICS[1:]:
        xm = [j["probe"][st][m] * d for j in joined]
        lo, hi = boot_paired_diff(xm, x1, y)
        keep = [i for i, l in enumerate(lab01) if l is not None]
        alo, ahi = boot_paired_auc_diff([xm[i] for i in keep], [x1[i] for i in keep], [lab01[i] for i in keep])
        diffs[m] = {"rho_diff": res[m]["rho"] - res["top1_overlap"]["rho"], "rho_diff_ci": [lo, hi],
                    "auc_diff": res[m]["auc"] - res["top1_overlap"]["auc"], "auc_diff_ci": [alo, ahi]}
        print(f"   {m:28s} vs top1: drho={diffs[m]['rho_diff']:+.3f} [{lo:+.3f},{hi:+.3f}]  dAUC={diffs[m]['auc_diff']:+.3f} [{alo:+.3f},{ahi:+.3f}]")
    # paired differences inside the alive band (>=3/15) only
    band = [i for i, v in enumerate(y) if v >= 3 / 15 - 1e-9]
    yb = [y[i] for i in band]
    x1b = [x1[i] for i in band]
    for m, d in METRICS[1:]:
        xmb = [j["probe"][st][m] * d for j in (joined[i] for i in band)]
        lo, hi = boot_paired_diff(xmb, x1b, yb)
        diffs[m]["rho_diff_alive_band"] = spearman(xmb, yb) - spearman(x1b, yb)
        diffs[m]["rho_diff_alive_band_ci"] = [lo, hi]
        print(f"   alive band {m:28s} vs top1: drho={diffs[m]['rho_diff_alive_band']:+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(band)})")
    # correlation between top1 and top10 across children
    x10 = [j["probe"][st]["top10_overlap"] for j in joined]
    res["_corr_top1_top10"] = spearman(x1, x10)
    print(f"   spearman(top1, top10) across children = {res['_corr_top1_top10']:.3f}")
    out["sets"][st] = {"metrics": res, "diffs": diffs}

# DB metrics on the same joined children for reference
print("\n== DB diagnostics on the same children (where recorded)")
out["db_same_children"] = {}
for m, d in DBM:
    pts = [(j["db"][m] * d, j["beh"]) for j in joined if m in j["db"]]
    if len(pts) < 8:
        continue
    x, y = map(list, zip(*pts))
    out["db_same_children"][m] = {"n": len(x), "rho": spearman(x, y), "ci": boot_ci(x, y, spearman)}
    print(f"   {m:24s} n={len(x)} rho={out['db_same_children'][m]['rho']:+.3f}")
out["joined"] = [{"tag": j["tag"], "beh15": round(j["beh"] * 15, 2), "n_reads": j["n_reads"],
                  "AB": {k: round(v, 4) for k, v in j["probe"]["AB"].items()}} for j in joined]
json.dump(out, open("probe_results.json", "w"), indent=1)
