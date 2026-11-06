"""Pure-Python rank statistics (no scipy on C:/Python314)."""
import math, random


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
    s = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def boot_ci(a, b, fn, n=2000, seed=7):
    rnd = random.Random(seed)
    idx = list(range(len(a)))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        v = fn([a[i] for i in s], [b[i] for i in s])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return [float("nan"), float("nan")]
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def boot_paired_diff(xa, xb, y, n=2000, seed=13):
    """Bootstrap CI of spearman(xa,y) - spearman(xb,y) resampling children jointly."""
    rnd = random.Random(seed)
    idx = list(range(len(y)))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        ya = [y[i] for i in s]
        d = spearman([xa[i] for i in s], ya) - spearman([xb[i] for i in s], ya)
        if not math.isnan(d):
            vals.append(d)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def boot_paired_auc_diff(xa, xb, lab, n=2000, seed=17):
    rnd = random.Random(seed)
    idx = list(range(len(lab)))
    vals = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        pa = [xa[i] for i in s if lab[i] == 1]; na = [xa[i] for i in s if lab[i] == 0]
        pb = [xb[i] for i in s if lab[i] == 1]; nb = [xb[i] for i in s if lab[i] == 0]
        if not pa or not na:
            continue
        vals.append(auc(pa, na) - auc(pb, nb))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
