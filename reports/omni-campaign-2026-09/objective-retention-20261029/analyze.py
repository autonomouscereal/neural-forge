"""Objective-vs-retention analysis over the omni_lab dump (dump.json, read as omni_lab_ro).

Outcome  = human-read behaviour (scorer claude/codex/mixed/cyril) with a numeric score and denominator.
           Machine-scored rows are excluded from the outcome.  Per child: median of score/denominator.
Metrics  = fold_diagnostic rows mapped to families by name + note text; per child: median value_num.
Stats    = Spearman rho (average ranks), bootstrap 95% CI, permutation p; AUC alive vs dead; LOO
           single-threshold balanced accuracy.  Pure Python (no scipy on this runtime).
"""
import json, re, random, math, statistics, collections, sys

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "dump.json", encoding="utf-8"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "results.json"
arm = {a["experiment_id"]: a for a in D["arm"]}
HUMAN = {"claude", "codex", "mixed", "cyril"}


def child_id(a):
    k = a["arm_key"] or ""
    k = re.sub(r"^MX\.", "", k)
    k = re.split(r"[@#]", k)[0]
    return k.upper()


# ---------------- outcome ----------------
beh_rows = collections.defaultdict(list)
for b in D["beh"]:
    if b["scorer"] not in HUMAN:
        continue
    v, den = b["score_value"], b["score_denominator"]
    frac = None
    if v is not None and den:
        frac = float(v) / float(den)
    elif b["total"] and b["useful"] is not None:
        frac = (b["useful"] + (b["partial"] or 0)) / b["total"]
    if frac is None or not (0 <= frac <= 1.0001):
        continue
    beh_rows[b["experiment_id"]].append((frac, float(den) if den else float(b["total"]), b))

# ---------------- metric families ----------------
def num(r):
    v = r["value_num"]
    if v is not None:
        return float(v)
    t = r["value_text"] or ""
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", t)
    if m and int(m.group(2)) > 0:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r"\s*([0-9.]+)\s*%", t)
    if m:
        return float(m.group(1)) / 100
    m = re.match(r"\s*(-?[0-9]*\.[0-9]+|-?[0-9]+)\b", t)
    if m:
        return float(m.group(1))
    return None


def family(r):
    n, note, vt = r["name"], (r["note"] or ""), (r["value_text"] or "")
    t = (note + " " + vt).lower()
    if n == "free_run_match":
        return "free_run_top1_match"
    if n == "lcp":
        return "mean_lcp_h64"
    if n in ("tf_top1", "top1", "top1_agreement_vs_parent", "top1_agreement_vs_trunk"):
        return "tf_top1"
    if n in ("topk_overlap", "top5", "top10"):
        return "topk"
    if n == "kl":
        return "kl"
    if n == "nll":
        return "nll"
    if n == "mse":
        return "mse"
    if n == "ppl":
        if "lambada" in t or "lam_ppl" in t:
            return "ppl_lambada"
        if "wiki" in t:
            return "ppl_wikitext"
        return "ppl_other"
    if n == "relL2":
        if "terminal_rel" in t or "terminal rel" in t:
            if "control" in t:
                return "drift_vs_control"
            return "terminal_relL2_vs_parent"
        if "copy_exact" in t or "drift" in t or "control" in t:
            return "drift_vs_control"
        if "heldout" in t or "held" in (r["split"] or "") or "holdout" in t:
            return "heldout_local_relL2"
        return "relL2_other"
    if n == "cos":
        if "terminal_cos" in t or "terminal cos" in t:
            return "terminal_cos_vs_parent"
        if "worst_site" in t:
            return "worst_site_cos"
        if "write" in t:
            return "write_heldout_cos"
        return "cos_other"
    if n == "retained_energy":
        return "retained_energy"
    if n == "other":
        if re.search(r"(hellaswag|arc_easy|arc-easy|piqa|winogrande|\bwino\b|lambada_openai acc|lambada acc)", t) and "acc" in t:
            return "bench_acc"
        if "drift_vs_control" in t or re.search(r"global drift", t):
            return "drift_vs_control"
        if re.search(r"matched[ _-]mass|probability mass", t):
            return "matched_mass"
    return None


DIRECTION = {  # +1 = higher should mean more retained
    "free_run_top1_match": 1, "mean_lcp_h64": 1, "tf_top1": 1, "topk": 1, "kl": -1, "nll": -1, "mse": -1,
    "ppl_lambada": -1, "ppl_wikitext": -1, "ppl_other": -1, "terminal_relL2_vs_parent": -1,
    "drift_vs_control": -1, "heldout_local_relL2": -1, "relL2_other": -1, "terminal_cos_vs_parent": 1,
    "worst_site_cos": 1, "write_heldout_cos": 1, "cos_other": 1, "retained_energy": 1, "bench_acc": 1,
    "matched_mass": 1,
}

met_rows = collections.defaultdict(lambda: collections.defaultdict(list))
for r in D["diag"]:
    f = family(r)
    if f is None:
        continue
    v = num(r)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        continue
    if f in ("free_run_top1_match", "tf_top1", "topk", "terminal_cos_vs_parent", "worst_site_cos",
             "write_heldout_cos", "retained_energy", "bench_acc") and not (-1.0001 <= v <= 1.0001):
        if f in ("tf_top1", "topk", "bench_acc") and 1 < v <= 100:
            v = v / 100
        else:
            continue
    met_rows[r["experiment_id"]][f].append(v)

# ---------------- per-child aggregation ----------------
children = collections.defaultdict(lambda: {"beh": [], "met": collections.defaultdict(list), "arms": set()})
for e, rows in beh_rows.items():
    a = arm[e]
    c = children[(a["program"], child_id(a))]
    c["beh"].extend(fr for fr, _, _ in rows)
    c["arms"].add(e)
for e, fams in met_rows.items():
    a = arm[e]
    key = (a["program"], child_id(a))
    if key not in children:
        continue  # metric on a child with no human read
    for f, vs in fams.items():
        children[key]["met"][f].extend(vs)
        children[key]["arms"].add(e)

rows = []
for (prog, cid), c in children.items():
    if not c["beh"]:
        continue
    ratios = [float(arm[e]["ratio_net"]) for e in c["arms"] if arm[e]["ratio_net"] is not None]
    rows.append({
        "program": prog, "child": cid, "beh": statistics.median(c["beh"]), "beh_max": max(c["beh"]),
        "beh_min": min(c["beh"]), "n_reads": len(c["beh"]),
        "ratio_net": statistics.median(ratios) if ratios else None,
        "met": {f: statistics.median(v) for f, v in c["met"].items()},
    })


# ---------------- statistics ----------------
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def boot_ci(a, b, fn, n=2000, seed=7):
    rnd = random.Random(seed)
    idx = list(range(len(a)))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        x, y = [a[i] for i in s], [b[i] for i in s]
        v = fn(x, y)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def perm_p(a, b, fn, n=2000, seed=11):
    rnd = random.Random(seed)
    obs = abs(fn(a, b))
    bb = list(b)
    c = 0
    for _ in range(n):
        rnd.shuffle(bb)
        if abs(fn(a, bb)) >= obs - 1e-12:
            c += 1
    return (c + 1) / (n + 1)


def loo_balacc(x, y):
    """y in {0,1}; choose threshold+direction on n-1 points, score held-out point."""
    n = len(x)
    hits = {0: [0, 0], 1: [0, 0]}
    for i in range(n):
        xs = [x[j] for j in range(n) if j != i]
        ys = [y[j] for j in range(n) if j != i]
        best, bt, bd = -1, None, 1
        cand = sorted(set(xs))
        ths = [(cand[k] + cand[k + 1]) / 2 for k in range(len(cand) - 1)] or cand
        for t in ths:
            for dsgn in (1, -1):
                tp = sum(1 for xx, yy in zip(xs, ys) if yy == 1 and dsgn * (xx - t) > 0)
                tn = sum(1 for xx, yy in zip(xs, ys) if yy == 0 and dsgn * (xx - t) <= 0)
                p1 = sum(ys); p0 = len(ys) - p1
                if p1 == 0 or p0 == 0:
                    continue
                ba = 0.5 * (tp / p1 + tn / p0)
                if ba > best:
                    best, bt, bd = ba, t, dsgn
        if bt is None:
            continue
        pred = 1 if bd * (x[i] - bt) > 0 else 0
        hits[y[i]][0] += 1 if pred == y[i] else 0
        hits[y[i]][1] += 1
    if hits[0][1] == 0 or hits[1][1] == 0:
        return float("nan")
    return 0.5 * (hits[0][0] / hits[0][1] + hits[1][0] / hits[1][1])


DEAD, ALIVE = 2 / 15 + 1e-9, 5 / 15 - 1e-9


def analyse(sub, label):
    out = []
    fams = sorted({f for r in sub for f in r["met"]})
    for f in fams:
        pts = [(r["met"][f], r["beh"]) for r in sub if f in r["met"]]
        if len(pts) < 5:
            out.append({"family": f, "subset": label, "n": len(pts)})
            continue
        x = [p[0] * DIRECTION.get(f, 1) for p in pts]
        y = [p[1] for p in pts]
        if len(set(x)) < 2 or len(set(y)) < 2:
            out.append({"family": f, "subset": label, "n": len(pts), "note": "constant"})
            continue
        rho = spearman(x, y)
        lo, hi = boot_ci(x, y, spearman)
        p = perm_p(x, y, spearman)
        alive = [xx for xx, yy in zip(x, y) if yy >= ALIVE]
        dead = [xx for xx, yy in zip(x, y) if yy <= DEAD]
        a = auc(alive, dead)
        bx = [xx for xx, yy in zip(x, y) if yy >= ALIVE or yy <= DEAD]
        by = [1 if yy >= ALIVE else 0 for yy in y if yy >= ALIVE or yy <= DEAD]
        ba = loo_balacc(bx, by) if len(alive) >= 3 and len(dead) >= 3 else float("nan")
        out.append({"family": f, "subset": label, "n": len(pts), "rho": rho, "ci": [lo, hi], "perm_p": p,
                    "n_alive": len(alive), "n_dead": len(dead), "auc": a, "loo_balacc": ba})
    return out


results = {"n_children": len(rows), "by_program": dict(collections.Counter(r["program"] for r in rows)), "tables": []}
results["tables"] += analyse(rows, "all")
results["tables"] += analyse([r for r in rows if r["program"] == "matrix-campaign"], "matrix-campaign")
results["tables"] += analyse([r for r in rows if r["program"] != "matrix-campaign"], "non-matrix")
real = [r for r in rows if r["program"] == "matrix-campaign" and r["ratio_net"] is not None and r["ratio_net"] < 0.95]
shaped = [r for r in rows if r["program"] == "matrix-campaign" and r["ratio_net"] is not None and r["ratio_net"] >= 0.95]
results["tables"] += analyse(real, "matrix ratio_net<0.95")
results["tables"] += analyse(shaped, "matrix ratio_net>=0.95 (parent-shaped)")
results["rows"] = rows
json.dump(results, open(OUT, "w"), indent=1, default=str)

print("children with human read:", len(rows), results["by_program"])
print(f"{'subset':38s} {'family':26s} {'n':>4s} {'rho':>6s} {'ci95':>15s} {'p':>6s} {'A':>3s} {'D':>3s} {'AUC':>5s} {'LOObal':>6s}")
for t in results["tables"]:
    if "rho" not in t:
        continue
    print(f"{t['subset']:38s} {t['family']:26s} {t['n']:4d} {t['rho']:6.3f} [{t['ci'][0]:6.3f},{t['ci'][1]:6.3f}] {t['perm_p']:6.4f} {t['n_alive']:3d} {t['n_dead']:3d} {t['auc']:5.3f} {t['loo_balacc']:6.3f}")
