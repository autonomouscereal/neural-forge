# NOTES-MX-AG — wave 31: the axis the plan sent it down is closed, and the axis it was told was solved moved five frontier rungs

Wave 31 was ordered to deepen the MLP interior — the 1,664-channel hole wave 30
opened — and to answer three leftovers beside it. The MLP interior is closed:
it falls at its first new rung and the vocabulary composition that was free at
`intermediate_size` 1792 costs two cases at every rung below it. The wave's
result is somewhere else, and it came from a probe that took twenty-nine seconds
of CPU each.

**The one-line result.** On the sealed `ARCH.CL1792` body, the composed
vocabulary axis is **FLAT from K = 26,624 down to K = 8,192** — eight rungs,
17,408 rows, 0.082 of stored ratio, reading 7, 7, 7, 7, 8, 7, 6, 5 — while the
same eight rungs read 14, 13, 9, 11, 9, 4 *alone*. The composed child reads the
**body's** number and not the vocabulary's. Five frontier rungs move:

| threshold | previous deepest | wave 31 | moved by |
|---|---|---|---:|
| ≥8/15 | `COMP.MA.MXO3` 0.729103x | **`V16CL1792` 0.637073x** | 0.092030 |
| ≥7/15 | `V19CL1792` 0.650769x | **`V12CL1792` 0.618813x** | 0.031956 |
| ≥6/15 | `V19CL1792` 0.650769x | **`V9CL1792` 0.605117x** | 0.045652 |
| ≥5/15 | `V19CL1792` 0.650769x | **`V8CL1792` 0.600552x** | 0.050217 |
| ≥4/15 | `V19CL1792` 0.650769x | **`V8CL1792` 0.600552x** | 0.050217 |

---

## 1. The MLP interior ladder is a closed axis

Three new rungs, each a capture plus fourteen site solves through wave 30's
unchanged solver, and each composed with `ARCH.V19`:

| `intermediate_size` | ratio | solo | V19-composed ratio | composed | vocabulary's price |
|---:|---:|---:|---:|---:|---:|
| 2,560 (parent) | 1.000000 | 12/15 | 0.794570 | 14/15 | **+2** |
| 1,792 (w30) | 0.856199 | 7/15 | 0.650769 | 7/15 | **0** |
| **1,664** | 0.832232 | **5/15** | 0.626802 | **3/15** | **−2** |
| **1,536** | 0.808265 | **4/15** | 0.602835 | **2/15** | **−2** |
| **1,408** | 0.784298 | **3/15** | 0.578868 | **1/15** | **−2** |
| 1,280 (w30) | 0.760331 | 4/15 | 0.554901 | 1/15 | −3 |
| 896 (w15) | 0.688431 | 4/15 | — | — | — |

Two things fall out, and neither is what the build list expected.

**The zero-cost composition is a property of the 1792 rung, not of the axis.**
`ARCH.V19CL1792` is still the only child in this campaign where a vocabulary
fold costs nothing. One hundred and twenty-eight channels below it the tax is
back at −2 and it stays there.

**The axis is not monotone.** `ARCH.CL1408` reads **3/15 at 0.784298x**, below
`ARCH.CL1280`'s 4/15 at 0.760331x and below `ARCH.CL896`'s 4/15 at 0.688431x.
Both readings are one case apart and both are the same reviewer, so this is
reported as a *shape* and not a step: the axis has a trough near 1,408 that it
climbs out of by 1,280. The power-law-then-plateau story wave 30 read off two
points does not survive five.

The refitted allocator now brackets the mlp transition to **0.856199x (CL1792,
7/15) → 0.832232x (CL1664, 5/15), width 0.023967** — 128 channels, down from
wave 30's 512-channel bracket of 0.095868.

## 2. Where the frontier actually went: the composed vocabulary ladder

`ARCH.CL1792` is retained from wave 30, so every rung below is one CPU row-gather
off sealed bytes — twenty-nine seconds each, no GPU, no capture. The wave built
eight of them because the first two were flat and the flatness was the finding.

| cell | K | token coverage | ratio | **composed** | solo (sealed) |
|---|---:|---:|---:|---:|---:|
| `V26CL1792` | 26,624 | 0.92592 | 0.682724 | **7/15** | — |
| `V24CL1792` | 24,576 | 0.91915 | 0.673594 | **7/15** | — |
| `V19CL1792` | 19,456 | 0.90563 | 0.650769 | 7/15 (w30) | 14/15 |
| `V18CL1792` | 18,432 | 0.89517 | 0.646204 | **7/15** | 13/15 |
| `V16CL1792` | 16,384 | 0.88 | 0.637073 | **8/15** | 9/15 |
| `V12CL1792` | 12,288 | 0.85183 | 0.618813 | **7/15** | 11/15 |
| `V9CL1792` | 9,216 | **0.81402** | 0.605117 | **6/15** | 9/15 |
| `V8CL1792` | 8,192 | **0.80049** | 0.600552 | **5/15** | 4/15 |

Read the last two columns against each other. The solo ladder has a summit at
K = 19,456 (14/15), a shoulder at 18,432 (13/15), a dip to 9/15 at 16,384, a
bump back to 11/15 at 12,288, and the campaign's sharpest transition between
9,216 and 8,192 — **five cases over 1,024 rows**. Composed onto a
pre-transition MLP body, *none of that structure survives*: seven rungs read
6-to-8 and the 1,024-row cliff costs **one** case, not five.

**Three consequences.**

* **The summit K is the wrong place to stand in the deep band.** `ARCH.V19` is
  the vocabulary axis's best solo rung by two cases and it is not the best
  composed rung by anything. Wave 27's finding — that a 1,024-row step in either
  direction from 19,456 costs a case — is a fact about the *solo* ladder and has
  been carried into every composed child since as if it were general. It is not.
* **The 0.81 coverage law is a law about the SOLO axis.** Wave 30 located the
  review cliff exactly at the 0.81 emission-coverage crossing and proposed it as
  a GPU-free gate on K. It is a gate on the solo reading. Composed, `V9CL1792`
  sits *on* the crossing and reads 6/15, `V8CL1792` sits *below* it and reads
  5/15, and the whole crossing is worth one case. The law's own falsifier —
  written into `preregistration-wave31.json` before any of these were built —
  was "if both new rungs read at or above 7, more rows bought cases"; both read
  exactly 7, which is the other half of the same answer.
* **More rows are pure cost.** `V26CL1792` buys 7,168 rows over `V19CL1792`, nine
  points of coverage over the crossing and **zero cases**, for 0.032 of ratio.

The `≥8/15` row deserves its caveat stated where the claim is. `V16CL1792`'s
eighth case is `creative-continuity`, read Partial on wave 30's `ARCH.CL1792`
precedent for a structured meta-description that carries the keeper, the foghorn
and the question motif; `ARCH.V19CL1792`'s own two-sentence circular version of
the same lane was sealed a **miss**. That precedent is split, so the ≥8 rung is a
one-case margin. **The claim that does not ride on it is `7/15 at 0.637073x`**,
which still beats the previous ≥7 by 0.014, and `V12CL1792`'s 7/15 at 0.618813x,
which beats it by 0.032 and rides on no one-case margin at all.

## 3. The M-inversion: both mechanisms tested, both fail

Wave 30 left `ARCH.CL1792M` reading 8/15 solo and `ARCH.V19CL1792M` 6/15 composed
— one *below* the stream-selected twin — an inversion it could describe and not
account for. Wave 31 built the ordered arm and one more.

**The ordered arm (`--evidence`), and the confound in it.** The plan reads the
RETOK-BODY item as "re-solve M against the vocabulary child's evidence stream".
Wave 30 already proved the V19 tokenizer segments the corpus **identically to the
parent at every one of 4,181 positions**, so there is no different segmentation
to solve against. What `--evidence` actually swaps is the *capture*: 3,145 fit
rows and 1,036 held-out on the canonical re-encoding, instead of 4,751 and 1,584
on the sealed ids.

> `ARCH.CL1792ME` reads **2/15**, against `ARCH.CL1792M`'s 8/15 at byte-identical
> storage. **Thirty-four per cent fewer fit rows costs six cases.**

That is the largest single-variable effect this campaign has measured on this
body, and it is an evidence-budget effect rather than a tokenizer effect. Its two
components — fewer rows and a different encoding of the same text — are *not*
separated by this build, and that is the honest boundary of the finding. Composed,
it reads 3/15: up one, inside the reader's own noise.

**The mechanism arm (`--select-metric-rows`), and its falsification.**
`reader_metric` builds `M` from the parent's 65,536-row head. A V19 composition
cannot emit 46,080 of those rows *at all* — the fold deletes them from a tied
table on the output side too — so wave 30's reader-metric coreset was ranked
against a model that does not survive composition. That is a mechanism, it
predicts the direction of the inversion, and it is one masked softmax away from
being testable. Wave 31 added the flag: probabilities masked to the 19,454
emittable rows and renormalised before the top-k, everything downstream
unchanged, and a **refusal** rather than a no-op when the mask covers every row.

Measured, before any review: the parent places **0.941431** of its own
probability mass on the emittable rows, the metric's condition number falls from
400,125 to 268,880 and its top eigenvalue from 56.29 to 48.50. The metric really
did change.

| arm | metric | solo | V19-composed |
|---|---|---:|---:|
| `CL1792` | stream | 7/15 | 7/15 |
| `CL1792M` (w30) | reader, full 65,536 rows | 8/15 | 6/15 |
| `CL1792MV` | reader, **19,454 emittable rows** | 7/15 | **5/15** |

Ranking against the reader that will actually exist costs **one more case**. The
mechanism is falsified in exactly the direction it predicted, which is the
cleanest form the answer could take. And the extension surface refuses the whole
family: `CL1792M` 7/30 vs `CL1792MV` 8/30 (+1 for the restricted metric, where
the sealed fifteen says −1), `V19CL1792M` 9/30 vs `V19CL1792MV` 8/30. Three arms
in, at three metrics, neither surface can separate them. **The reader metric does
not pay on this axis and the campaign should stop paying for it.**

## 4. The triple, and what it did to the interference table

`ARCH.V19CL1792X13` — vocabulary 19,456 × `intermediate_size` 1792 × excision of
block 13, the campaign's freest depth site at 13/15 alone — reads **2/15 at
0.608519x** against a gate of five. The lane does not open.

The allocator predicted **3.0** and was one case out, its *best* prediction of the
wave, by charging all three pair terms at once: 4.4444 + 1.75 + 0.8056 = 7.0
cases of interference against an additive 10.0. So the interference model is
right about triples, and triples are not worth building. Both halves are this
child's result.

The two-axis base is the interesting row. `ARCH.CL1792X13` reads **4/15 at
0.813949x** against the body's own 7/15 — the allocator predicted 7.19 and was
3.19 optimistic, its **worst** error of the wave. Refitting on the post-wave
878-row ledger moves the pair terms accordingly:

| pair | wave 30 fit (25 children) | pre-wave-31 refit (27) | **post-wave-31 (39)** |
|---|---:|---:|---:|
| depth × vocabulary | 4.44 | 4.3835 | **4.3835** |
| depth × mlp | 1.39 | 0.8056 | **2.9034** |
| mlp × vocabulary | 1.17 | 1.75 | **0.7613** |

MAE 2.273 cases over 39 composed children; arithmetic gate **73 cells checked, 0
mismatched**. Depth-with-mlp more than tripled and mlp-with-vocabulary more than
halved, both for the same reason: this wave measured eight nearly-free
mlp×vocabulary compositions and two expensive depth×mlp ones. The recommendation
the table now makes is the one wave 31 arrived at by accident — **spend the whole
budget on vocabulary and mlp, and never on depth.**

## 5. Pre-registration, and how the allocator did

Every count was predicted to the parameter before the build, in
`prereads/preregistration-wave31.json`, committed before the first review was
opened. Ten named splits; the allocator returned nine, because
`intermediate_size` 1536 at 0.808265x and 1792-with-block-13 at 0.813949x both
round to `"0.81"` in its two-decimal recommendation key — 0.0057 apart, *inside*
its own default tolerance of 0.006. The key is six decimals now.

| cell | predicted | observed | error |
|---|---:|---:|---:|
| `CL1664` | 6.25 | 5 | −1.25 |
| `CL1536` | 5.50 | 4 | −1.50 |
| `CL1408` | 4.75 | 3 | −1.75 |
| `V19CL1664` | 6.50 | 3 | −3.50 |
| `V19CL1536` | 5.75 | 2 | −3.75 |
| `V19CL1408` | 5.00 | 1 | −4.00 |
| `CL1792X13` | 7.19 | 4 | −3.19 |
| `V19CL1792X13` | 3.00 | 2 | **−1.00** |
| `V24CL1792` | 5.85 † | 7 | **+1.15** |
| `V26CL1792` | 5.45 † | 7 | **+1.55** |

† The allocator **refused** both L4 splits — its vocabulary grid contains only K
the campaign has sealed — so 5.85 and 5.45 are its own interpolation rule applied
by hand, and they are the only two predictions in the table that are *pessimistic*.
It is systematically optimistic everywhere it can price a split itself and
systematically pessimistic on the two rungs it declined to price, which is worth
more than the average error: the model's bias has a sign that depends on which
axis is moving.

## 6. Two refusals, both correct

* **The sealer refused six reviews before a dossier was generated.**
  `BUILD_RECEIPT_SCHEMAS` is append-only and no wave had told it that
  `neural-forge-matrix-wave31-standard-shape-child-v1` exists. Six empty cell
  directories, no shards, no partial state — the refusal is at the top of the
  sealer and not in the middle of it. The three strings were appended in a
  commit and shipped as a verified deployment rather than edited into a deployed
  file.
* **The vocabulary builder refused K = 6,144, 5,120 and 4,096.** Its
  sealed-corpus coverage gate fires: at K = 6,144 the keep-set loses one
  occurrence of the donor-mapped stream and 782 of the canonical one. So the
  vocabulary axis has a hard floor between **6,144 and 8,192 rows on this
  corpus** — a property of the evidence, not of the model, and the first
  vocabulary refusal in the campaign.

## 7. Benchmarks

`limit=200 seed=1234 batch=4 max_length=2048 float32`, sm86. The parent row was
re-measured this wave and reproduces the sealed reference exactly (0.5450 /
0.5050 / 0.6800 / 0.5500 / 0.4600, ppl 24.39, wiki 81.80) — the fourth
calibration.

| model | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 12/15 |
| `CL1792MV` | 0.4150 | 0.4600 | 0.5500 | 0.5650 | 0.2350 | 267.61 | 194.94 | 7/15 |
| `CL1664` | 0.3800 | 0.4250 | 0.5750 | 0.5450 | 0.1950 | 222.97 | 201.96 | 5/15 |
| `CL1536` | 0.3800 | 0.3950 | 0.5650 | 0.5250 | 0.1750 | 381.47 | 226.68 | 4/15 |
| `CL1408` | 0.3900 | 0.3850 | 0.5850 | 0.5150 | 0.1700 | 377.38 | 233.36 | 3/15 |
| `CL1792ME` | 0.4200 | 0.3950 | 0.5550 | 0.4800 | 0.1550 | 719.76 | 274.95 | 2/15 |
| `V26CL1792` | 0.4050 | 0.4100 | 0.5500 | 0.5950 | 0.2150 | 368.46 | 334.29 | 7/15 |
| `V24CL1792` | 0.3950 | 0.4300 | 0.5700 | 0.5900 | 0.2300 | 386.90 | 356.60 | 7/15 |
| `V18CL1792` | 0.3900 | 0.4000 | 0.5600 | 0.5400 | 0.2300 | 437.36 | 464.15 | 7/15 |
| `V16CL1792` | 0.4000 | 0.4050 | 0.5450 | 0.5200 | 0.2150 | 446.95 | 523.97 | **8/15** |
| `V12CL1792` | 0.3700 | 0.3850 | 0.5300 | 0.5350 | 0.1900 | 522.32 | 770.56 | **7/15** |
| `V9CL1792` | 0.2800 | 0.3650 | 0.4850 | 0.5550 | 0.1550 | 831.68 | 1321.05 | 6/15 |
| `V19CL1792MV` | 0.3950 | 0.3800 | 0.5500 | 0.5450 | 0.1950 | 554.68 | 450.16 | 5/15 |
| `CL1792X13` | 0.3750 | 0.3950 | 0.5750 | 0.5400 | 0.1400 | 2415.01 | 622.93 | 4/15 |
| `V19CL1792X13` | 0.3650 | 0.3450 | 0.5850 | 0.5200 | 0.1550 | 3596.88 | 1701.54 | 2/15 |

The inversion is on this table for the tenth time and harder than ever:
`V12CL1792` reads 7/15 with wiki word-perplexity **770**, and `CL1792ME` reads
2/15 at **275**. Word-perplexity orders these children almost exactly backwards
inside the deep band.

## 8. Both surfaces, and what the second one says

Bank `c9d37004c901d757…`, unchanged since wave 29; parent run once.
**Spearman 0.7411 over nineteen children — `VALIDATED: the sealed rank order
reproduces on the extension surface`**, against wave 30's 0.6544 `PARTIAL`.

| child | sealed 15 | ext 30 | charged (30) |
|---|---:|---:|---:|
| `V16CL1792` | 8 | 9 | 5 |
| `V24CL1792` | 7 | 10 | 3 |
| `V12CL1792` | 7 | 9 | 3 |
| `V18CL1792` | 7 | 9 | 3 |
| `CL1792MV` | 7 | 8 | 4 |
| `V26CL1792` | 7 | 8 | 6 |
| `V9CL1792` | 6 | 6 | 6 |
| `V8CL1792` | 5 | 8 | 5 |
| `V19CL1792MV` | 5 | 8 | 3 |
| `CL1664` | 5 | 5 | 7 |
| `CL1536` | 4 | 6 | 8 |
| `CL1792X13` | 4 | 3 | 9 |
| `V19CL1792ME` | 3 | 8 | 5 |
| `V19CL1664` | 3 | 7 | 9 |
| `CL1408` | 3 | 4 | 8 |
| `CL1792ME` | 2 | 7 | 6 |
| `V19CL1792X13` | 2 | 5 | 13 |
| `V19CL1536` | 2 | 2 | 8 |
| `V19CL1408` | 1 | 5 | 11 |

**The second surface confirms the flat ladder and does not confirm the ≥8 rung.**
The whole `V*CL1792` family reads 7-8 sealed and 8-10 on the extension, with
`V24CL1792` — not `V16CL1792` — at the top of the second column. Neither surface
separates them. The frontier claim that survives both is *the ladder is flat*;
the individual rung ordering inside it does not survive either.

The bank's known failure mode fires again: `CL1792ME` scores 7/30 against a
sealed 2/15 with six charged lanes, and `V19CL1792ME` 8/30 against 3/15. The
rubric is fooled by fluent strings exactly as its docstring says it can be.

## 9. What was ordered and not run

* **L5 in full.** The paraphrase triplets are now authored and unrun for **three**
  waves. B4's controlled evidence-refresh pair is unrun for **three** waves — and
  §3's `--evidence` result is the closest thing to it anyone has measured: 34%
  fewer fit rows costs six cases, which is the number a refresh discipline would
  have to beat. LACE shrinkage on the CL ladder did not run.
* **The reproduction check on the builder change.** `CL1664`, `CL1536`, `CL1408`
  and `CL1792X13` were built under deployment `dbf2294e`; `CL1792ME` and
  `CL1792MV` under `830842f1`, whose `mx31_build.py` differs. The diff is
  confined to `emittable_parent_ids`, the interior of `reader_metric` and one
  argparse block, and **every hunk inside `build()` is under
  `if select_metric == "logit"`** — so the stream-metric path that wrote the four
  bodies is byte-identical between the two. That is an argument from the diff,
  not a measurement: rebuilding one body under the final deployment and checking
  the digest would have cost nine minutes on an idle card and was not done.
* **`V4CL1792`, `V5CL1792`, `V6CL1792`** — refused by the builder, §6.

## 10. Estate

Eight commits, five deployments, every deployment verified file by file against
the commit's own blob digests: **5,714 files, 0 content mismatches, 0 missing, 0
undeclared, 0 hardlinks**, every time, plus a post-wave re-verification of the
survivor at the same five numbers. Two of the five deployments exist because
something refused, and both fixes shipped as a commit and a verified deployment
rather than as an edit to a deployed file — the rule wave 30 broke once and this
wave did not. Four bodies were built under `dbf2294e` and two under `830842f1`;
everything was sealed under `898f227d` or later, and `nf-mx31/env.sh` is the one
place each new deployment was named.

**The tree cut.** `run/mx31/cut_wave31.py` applies `NOTES-MX-AF` §9's named fix:
five substitutions, all identifiers, paths and schema strings, **no prose form of
the wave number**, so no docstring that records what wave 30 measured was
rewritten. Three files — the seal verifier, the retention script and the builder
— were copied with **no substitution at all** and hand-edited once each.
`--audit` reads every file back and refuses a surviving wave-30 name. **26
declared seal roots became 27 with `wave30/cells` still declared**, and
`BASIS_SCHEMAS_ACCEPTED` **gained** wave 30's schema instead of losing it.
`run/mx31/wave31-tree-cut-manifest.json` is the explicit diff, per file, per
substitution.

The audit earned itself twice. It passed on the cut, then **refused the tree
again** the first time it ran after the retention policy was written by hand:
`mx31_retain.py` had gained seven `MX30-CL1792` references, because every dropped
vocabulary composite cites the retained wave-30 body it is rebuilt from and that
is the whole justification for deleting it. Both surviving wave-30 names are now
declared exceptions **at pinned counts** — a waiver with a count breaks again if
the file changes, a waiver without one is a hole — and the nineteen KEEP/DROP
keys were checked separately for an `MX30` tag. There is none.

Ledger: **840 → 878 rows**, 38 appended, 0 superseded. Backup
`backup/ledger-pre-wave31.jsonl` at the pre-merge digest `e1de3fd9853a6bb2…`,
`cmp` identical; merge applied once behind `.MERGED_ONCE`; the authority's old
bytes are a **byte prefix** of the new (1,433,106 of 1,500,006); new digest
`85d5f12674169787…`. Seal verification: **27 roots audited against disk, 0
undeclared; 874 of 878 rows resolve and all 874 verify, 0 failures**, with the
same four unresolved rows waves 25–30 carry (`CAL.C01.T5.X4` ×3, `COMP.MC.MXO0`).

`sealq.sh` ran **once**, behind `.SEALQ_RAN`; the eight-argument guard fired on
none of the nineteen. Parent lane calibration unmoved; the parent's benchmark row
reproduces the sealed reference exactly, the fourth calibration.

Both sm86 3090s ran the wave. The two H3 containers were queue-empty and holding
256 MiB each, so both were restarted before any allocation — and came back
holding 256 MiB each, which establishes that 288/279 MiB *is* the free-standing
floor on these cards rather than a leak. `/mnt/ai_data` at 1.7 TB free, 64% used,
unchanged across the wave; both cards free-standing at 288 and 279 MiB with
nothing running when it closed.

Retention ran **after** the package, `--plan` read before `--apply`: **eight
children kept (2.68 GiB), eleven deleted (2.97 GiB)**. The rule applied is *keep
what a GPU built, drop what one CPU pass rebuilds from a retained source*; the
deviation is stated in the manifest, which keeps `V16CL1792` and `V12CL1792`
anyway because a frontier row that must be regenerated before it can be
re-examined is a frontier row nobody re-examines. Deployments pruned from seven
to two (current+1), survivor re-verified after the wave.

## 11. What the next wave should do with this

The ladder in §2 was mapped in one direction only — **downward in K, on one
body, at one `intermediate_size`**. Three things follow and none of them needs a
new mechanism:

1. **Re-run the ladder on a second body.** Everything in §2 could be a property
   of `ARCH.CL1792` rather than of pre-transition bodies generally. The cheapest
   control is the same eight K on `ARCH.V19`'s own body — parent-shaped, no mlp
   fold — which would say whether flatness is about the *body being
   pre-transition* or about the *mlp axis specifically*.
2. **Push K below 8,192 by fixing the corpus, not the model.** The refusal in §6
   is an evidence-coverage gate, so a larger or better-covered evidence corpus
   moves the floor. At 0.600552x with 5/15 still held, the axis has not visibly
   run out; the *builder* has.
3. **Stop building triples and stop paying for reader metrics.** §3 and §4 each
   closed a lane with a measurement rather than a null, and the post-wave
   interference table now says in its own numbers where the budget goes.
