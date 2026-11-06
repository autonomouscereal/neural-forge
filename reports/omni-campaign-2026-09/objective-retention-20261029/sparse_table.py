"""Record-only paired top-1 vs top-k table: LFM2 sparse-semantic token-interface children (same 648 cached rows)."""
import sys; sys.path.insert(0, ".")
from analyze_stats import auc, spearman
# (child, top1, top5, top10 of 648; behaviour; alive?)  sources: reports/sparse-semantic-*/cached-hidden-diagnostic.json; omni_lab fold_behaviour
rows = [
 ("A2 (k16; top-5/10 from a2-k16-rank-retro, same child)", 533, 579, 584, "manual review rejected (empty, repetition)", 0),
 ("A3 (k16 norm gamma0.5 cap3)", 513, 601, 611, "rejected; worse overall", 0),
 ("A4 (k16 residual OMP)", 544, 582, 588, "0/16 completed task; 3 empty, 6 looped", 0),
 ("A9 (k32 residual OMP)", 551, 589, 605, "3u/7p/6m of 16; V5 8/15", 1),
 ("A11 (k32 split C_out)", 585, 632, 639, "3u/9p/4m of 16; V5 9/15", 1),
]
for i, name in ((1, "top1"), (2, "top5"), (3, "top10")):
    pos = [r[i] for r in rows if r[5] == 1]; neg = [r[i] for r in rows if r[5] == 0]
    print(name, "AUC", auc(pos, neg), "values alive", pos, "dead", neg)
