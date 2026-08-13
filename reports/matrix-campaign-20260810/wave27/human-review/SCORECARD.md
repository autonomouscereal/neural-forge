# Wave 27 — the shelf is a peak, the excision cost belongs to the block, and prompt-echo is not a width signature

**Deployment** `neural-forge-matrix-49770e9f` (4,981 files, 0 discrepancies before
the wave). **Verified again AFTER the wave**: 0 content mismatches, 0 missing,
0 hardlinks, **135 undeclared**, all 135 `__pycache__` bytecode written by running
the tree — the same result and the same correct refusal wave 26 recorded
(`post-wave-deployment-verification-49770e9f.json`).
**Ledger** 724 → **770 rows**, sha
`e4a6a863f9a752f7016137f5230fb9207dc9188c5bcede0c59e444ed74ab7dcd`.
Pre-wave backup taken with sha + `cmp`, byte-prefix of the result verified against
it, 46 rows appended (23 architecture + 23 verdict), no row rewritten.
**Seals** 766 of 770 rows resolve to a receipt on disk and all 766 verify; 0 fail.
**23 cell roots audited against disk, 0 undeclared** — `wave26/cells` still
declared, `wave27/cells` added as ONE line.
**Parent lane** `865f006e3813712ec202cb23` on all 23 dossiers, protocol
`e5e377688597` / pack `0ee6bf14cef5`.
**Parent benchmark row** 0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl
**24.39**, wikitext wppl **81.80**, bits/byte **1.1882** — the sealed reference,
**fifteenth consecutive match**.

Twenty-three children built, reviewed, benchmarked and sealed, on two GPUs in
parallel.

## 1. Every child, both readings, sorted by net stored ratio

| child | net stored | params | review | U / P / M | caps | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | bits/byte |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PARENT** | 1.000000 | 229,693,184 | **12/15** | 5 / 7 / 3 | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 1.1882 |
| `ARCH.V19` | **0.794570** | 182,507,264 | **14/15** | 6 / 8 / 1 | 0 | 0.4150 | 0.4300 | 0.6200 | 0.5250 | 0.3650 | 60.71 | 191.20 | **1.4173** |
| `ARCH.V17` | 0.785440 | 180,410,112 | **11/15** | 4 / 7 / 4 | 0 | 0.4050 | 0.4250 | 0.6100 | 0.5300 | 0.3700 | 62.59 | 214.64 | 1.4485 |
| `ARCH.V20X07` | 0.746614 | 171,492,096 | **9/15** | 3 / 6 / 6 | 0 | 0.3700 | 0.4250 | 0.5700 | 0.5500 | 0.1550 | 344.68 | 310.81 | 1.5484 |
| `ARCH.V20X09` | 0.746614 | 171,492,096 | **9/15** | 3 / 6 / 6 | 3 | 0.4050 | 0.4050 | 0.5800 | 0.5950 | 0.1900 | 607.30 | 425.87 | 1.6334 |
| `ARCH.V20X09G03` | 0.746614 | 171,492,096 | **8/15** | 3 / 5 / 7 | 3 | 0.4050 | 0.4050 | 0.5650 | 0.6000 | 0.1650 | 700.26 | 424.20 | 1.6323 |
| `ARCH.V18X13` | 0.737484 | 169,394,944 | **10/15** | 3 / 7 / 5 | 3 | 0.4100 | 0.3850 | 0.6250 | 0.5050 | 0.2950 | 314.94 | 902.71 | 1.8360 |
| `ARCH.V18X13G03` | 0.737484 | 169,394,944 | **10/15** | 2 / 8 / 5 | 3 | 0.4000 | 0.3800 | 0.6100 | 0.5050 | 0.3000 | 242.52 | 756.72 | 1.7884 |
| `ARCH.V18X09` | 0.737484 | 169,394,944 | **10/15** | 3 / 7 / 5 | 1 | 0.4350 | 0.3900 | 0.5950 | 0.5700 | 0.1650 | 767.94 | 480.81 | 1.6661 |
| `ARCH.V18X09G03` | 0.737484 | 169,394,944 | **9/15** | 3 / 6 / 6 | 2 | 0.4300 | 0.3950 | 0.5800 | 0.5800 | 0.1600 | 879.16 | 478.26 | 1.6647 |
| `ARCH.V18X07` | 0.737484 | 169,394,944 | **9/15** | 3 / 6 / 6 | 0 | 0.3850 | 0.4000 | 0.5800 | 0.5550 | 0.1550 | 377.83 | 345.99 | 1.5773 |
| `ARCH.V18X07G03` | 0.737484 | 169,394,944 | **8/15** | 3 / 5 / 7 | 0 | 0.3900 | 0.4000 | 0.5800 | 0.5600 | 0.1500 | 412.52 | 344.78 | 1.5764 |
| `ARCH.V22X13KV6JQ` | 0.735201 | 168,870,656 | **2/15** | 1 / 1 / 13 | 2 | 0.3750 | 0.3950 | 0.5550 | 0.5300 | 0.1450 | 1268.19 | 2414.49 | 2.1015 |
| `ARCH.V20X13KV6JQ` | 0.726071 | 166,773,504 | **2/15** | 1 / 1 / 13 | 3 | 0.3500 | 0.3850 | 0.5900 | 0.5200 | 0.1450 | 1267.91 | 2659.75 | 2.1276 |
| `ARCH.V20X13KV6JQG03` | 0.726071 | 166,773,504 | **2/15** | 1 / 1 / 13 | 5 | 0.3400 | 0.3900 | 0.6000 | 0.4950 | 0.1350 | 887.69 | 1860.62 | 2.0312 |
| `ARCH.V18X13KV6JQ` | 0.716941 | 164,676,352 | **2/15** | 1 / 1 / 13 | 1 | 0.3600 | 0.3650 | 0.5850 | 0.5050 | 0.1350 | 1423.87 | 3025.62 | 2.1623 |
| `ARCH.V9X13` | 0.696398 | 159,957,760 | **6/15** | 2 / 4 / 9 | 1 | 0.3300 | 0.3850 | 0.5650 | 0.4750 | 0.2750 | 794.64 | 3202.09 | 2.1776 |
| `ARCH.V20X13X01` | 0.694093 | 159,428,352 | **3/15** | 0 / 3 / 12 | 1 | 0.3500 | 0.3750 | 0.5400 | 0.5150 | 0.1600 | 4494.91 | 2706.12 | 2.1322 |
| `ARCH.V20X13X01G03` | 0.694093 | 159,428,352 | **1/15** | 1 / 0 / 14 | 5 | 0.3300 | 0.3700 | 0.5450 | 0.4900 | 0.1450 | 2711.43 | 1596.28 | 1.9898 |
| `ARCH.V18X13X01` | 0.684962 | 157,331,200 | **3/15** | 1 / 2 / 12 | 1 | 0.3350 | 0.3700 | 0.5400 | 0.5150 | 0.1700 | 4697.83 | 3093.56 | 2.1683 |
| `ARCH.V18W896` | 0.675277 | 155,106,432 | **2/15** | 1 / 1 / 13 | 1 | 0.3000 | 0.4050 | 0.5400 | 0.5150 | 0.1400 | 1115.10 | 1112.31 | 1.8924 |
| `ARCH.V18X13X01KV6JQ` | 0.664419 | 152,612,608 | **0/15** | 0 / 0 / 15 | 3 | 0.3450 | 0.3200 | 0.5250 | 0.5000 | 0.0950 | 21786.49 | 3046.82 | 2.1642 |
| `ARCH.V32CL896` | 0.542347 | 124,573,440 | **2/15** | 0 / 2 / 13 | 4 | 0.3700 | 0.4050 | 0.4900 | 0.4950 | 0.1000 | 3398.82 | 585.02 | 1.7190 |
| `ARCH.V20CL896` | **0.487566** | 111,990,528 | **2/15** | 0 / 2 / 13 | 4 | 0.3250 | 0.3450 | 0.5150 | 0.4800 | 0.1050 | 4249.61 | 834.58 | 1.8149 |

`bits/byte` is the tokenizer-independent column and every child here segments
differently from the parent, so it is the only perplexity number on this page
that compares across rows.

## 2. What the wave found

**a. THE SHELF IS A PEAK, AND THE PEAK BEATS THE PARENT BY TWO.** `ARCH.V19` —
vocab_size 19,456 on the parent, zero solve — reads **14/15 at 0.794570x with
zero cap hits**. That is the highest review in twenty-seven waves, two cases above
the parent's own twelve, and it is a rung nobody had built. Its neighbour one
1,024-row step down reads eleven. The vocabulary curve is now:

| K | net stored | review | lam ppl | bits/byte |
|---:|---:|---:|---:|---:|
| 32,768 | 0.853916 | 11/15 | 156.31 | 1.6713 |
| 22,528 | 0.808265 | 13/15 | 55.08 | 1.3787 |
| 20,480 | 0.799135 | 13/15 | 56.29 | 1.4038 |
| **19,456** | **0.794570** | **14/15** | **60.71** | **1.4173** |
| 18,432 | 0.790005 | 13/15 | 66.26 | 1.4306 |
| **17,408** | **0.785440** | **11/15** | **62.59** | **1.4485** |
| 16,384 | 0.780874 | 9/15 | 268.29 | 1.8289 |
| 14,336 | 0.771744 | 9/15 | 83.71 | 1.5101 |
| 12,288 | 0.762614 | 11/15 | 442.57 | 2.0002 |
| 10,240 | 0.753484 | 10/15 | 116.31 | 1.6508 |
| 9,216 | 0.748919 | 9/15 | 148.17 | 1.7191 |
| 8,192 | 0.744354 | 4/15 | 1027.86 | 2.2412 |

Wave 26 called 18,432–22,528 a flat 4,096-row shelf. With both interior rungs now
built it is a **peak at 19,456 with 1,024-row shoulders**: 11 / 13 / **14** / 13 /
13. The perplexity column does not mark the peak — V22 has the best lambada ppl
and the best bits/byte on the whole axis and reads one case lower — which is the
fourteenth time this campaign has found the benchmark column and the review column
disagreeing about which child is best.

**b. THE EXCISION-COMPOSITION COST BELONGS TO THE BLOCK, NOT TO K.** This was the
question the wave was sent to answer, and the identity matrix answers it:

| base | K = 18,432 | K = 20,480 | cost |
|---|---:|---:|---|
| solo (no excision) | 13/15 | 13/15 | — |
| **block 13** | **10/15** | **10/15** (w26) | **−3, −3** |
| **block 7** | **9/15** | **9/15** | **−4, −4** |
| **block 9** | **10/15** | **9/15** | **−3, −4** |

Same parameter count, same ratio, same base, same zero solve on both axes — and
block 7 costs one case more than block 13 at *both* K. Block 9 is the only cell
that moves with K and it moves by one. So the cost is a property of the **block**
to within one case, and K moves it by at most one. It also survives leaving the
shelf: `ARCH.V9X13` reads **6/15** against V9's nine, block 13 costing three again
four and a half points of ratio lower.

The wave-26 puzzle is resolved in passing. The excision cost 1 case on V32 and 3
on V20 not because the block changed but because **the base did**: V32 reads 11 and
lands at 10, V18/V20 read 13 and land at 10. On every shelf-or-better base, block
13 lands the child at ten.

**c. `ARCH.V18X13` AND `ARCH.V18X09` ARE THE DEEPEST TEN IN THE CAMPAIGN**, at
0.737484x — nine thousandths below wave 26's `V20X13`, tied, from two different
blocks.

**d. PROMPT-ECHO IS NOT A WIDTH SIGNATURE. Wave 26's claim is withdrawn.** Wave 26
concluded from three sightings — `V8CL896`, `V32W896`, `V20W896`, all width
children — that returning the prompt verbatim is a signature of the width axis.
This wave produced **four more sightings on children with no width fold at all**:
`V18X13X01` and `V20X13X01` (vocabulary + double block excision, hidden size 1024
throughout) each echo the prompt on four of sixteen lanes, and both `CL896` probes
echo it on four more. Prompt-echo is a symptom of severe damage from any axis, and
the width reading was an artifact of which children had been built deep enough.

**e. THE HEAD-MERGE AXIS IS FATAL ON A VOCABULARY BASE — first time the pair has
been built.** All three pairings read **2/15**: 18,432 / 20,480 / 22,528 on
`X13KV6JQ` at 0.716941 / 0.726071 / 0.735201x. Three second folds are now priced on
one base:

| second fold on a shelf vocabulary base | cost |
|---|---:|
| block excision (13) | **−3** |
| head merge (16→12 joint) | **−11** |
| width (1024→896) | **−11** |

And once the merge has taken the child, **K stops mattering entirely**: seven of
sixteen lanes are *byte-identical* between `V20X13KV6JQ` and `V22X13KV6JQ`. The
same inertness shows on the healthy side — six of sixteen lanes are byte-identical
between `V18X07` and `V20X07`.

**f. λ = 0.3 IS NOT "THE RUNG THAT NEVER HURTS". It hurt on three bases out of
four.** Wave 26's four ladders made 0.3 the peak or joint peak everywhere. This
wave's four arms on shelf-excision bases read, against their own zero-solve
controls at identical parameter counts:

| base | control | λ = 0.3 | delta |
|---|---:|---:|---:|
| `V18X13` | 10 | 10 | **0** |
| `V18X09` | 10 | 9 | **−1** |
| `V20X09` | 9 | 8 | **−1** |
| `V18X07` | 9 | 8 | **−1** |
| `V20X13X01` | 3 | 1 | **−2** |
| `V20X13KV6JQ` | 2 | 2 | 0 |

Six more ladders, no gain anywhere, and four losses. The two sharpest rows are
`V20X13X01G03`, which improved its own held-out terminal-logit objective by **23%**
(relSSE 0.5166 → 0.3985) and lost **two** review cases, and `V20X13KV6JQG03`, which
improved it by 26% and took the cap count from three lanes to five. This is the
cleanest evidence the campaign has for grading on the text.

The wave-26 **LIST→SET regression reproduced**: `V18X13` accumulates into a list
with append and holds `coding-deduplicate`; `V18X13G03` accumulates into a set and
loses it. Third base, same defect — a property of the terminal-gain solve.

**g. `ARCH.V20CL896` IS THE DEEPEST CHILD IN THE CAMPAIGN TO ANSWER ANYTHING.**
2/15 at **0.487566x** — under half the parent, fifteen points of ratio below the
previous holder (`V20X13W896`, 2/15 at 0.639307x). It holds summarisation and the
uncertainty lane in part.

**h. `ARCH.V18X13X01KV6JQ` IS THE FIRST ZERO OF FIFTEEN IN 27 WAVES.** Four axes at
0.664419x and not one lane answered even in part. Every other collapsed child in
this wave — the 2/15s at 0.487x, 0.675x, 0.726x — still answers **summarisation**,
usually in full. This child does not. **Summarisation is the last lane to go**, and
this is the child that took it.

**i. The coherence spectrum after wave 27** (deepest child at each floor):

| floor | deepest | ratio | wave |
|---|---|---:|---|
| ≥ 14/15 | `ARCH.V19` | **0.794570** | 27 |
| ≥ 13/15 | `ARCH.V18` | 0.790005 | 26 |
| ≥ 11/15 | `ARCH.V20X13G03` | 0.746614 | 26 |
| ≥ 10/15 | `ARCH.V18X13` / `ARCH.V18X09` | **0.737484** | 27 |
| ≥ 7/15 | `ARCH.V8X13GL01` | 0.691832 | 24 |
| ≥ 4/15 | `ARCH.CL896` | 0.688431 | 15 |
| ≥ 2/15 | `ARCH.V20CL896` | **0.487566** | 27 |
| ≥ 1/15 | `ARCH.V16CL896` | 0.469305 | 23 |

Wave 27 moves three of the eight floors. **Deepest ≥ 7/15 did not move** — no child
this wave read between 7 and 9 below 0.6918x.

## 3. What went wrong, recorded rather than smoothed

* **A SEALED BODY DID NOT REPRODUCE.** `bodyq.sh` rebuilt wave 17's
  `ARCH.F1301KV6Q` (blocks 1+13 excised plus the 16→12 joint head merge) and the
  guard **refused**: identical parameter count (200,847,104) and a different tensor
  digest (rebuilt `ad406f480a181901…`, sealed `90a9e5187feef678…`). Rebuilding a
  second time produced a **byte-identical** result to the first, so this is **code
  drift between wave 17's builder and wave 27's on the joint-solve path, not
  nondeterminism**. Wave 26 established that the *pure excision* path rebuilds
  bit-exactly across nine waves; that result does **not** extend to a path with a
  solve in it, and no wave had checked. `ARCH.V18X13X01KV6JQ` therefore cites its
  own rebuilt body's receipt and **not** wave 17's sealed cell.
* **Two λ arms could not be built at all, and the refusal is correct.**
  `V18W896G03` and `V20X13W896G03` were refused by `mx27_termgain.py`'s own identity
  gate: *the child's token table is NOT the parent's rows at the injected ids*. The
  λ objective is defined against the parent's own table rows, and a width fold
  projects the embedding to 896 dimensions, destroying that precondition. **The
  terminal-gain arm is not applicable to width children.** Left unfixed on purpose —
  the gate is protecting a real precondition, and disabling it would have produced
  two numbers that meant nothing. The failed queue left two empty `cells/ARCH.*`
  directories (a `mkdir -p` that ran before the sealer); both were removed after
  confirming they contained no receipt and no child.
* **Wave 26's scorecard mis-described the unresolved ledger rows.** It says the 4
  rows that do not resolve "are the same `CAL.C01.T5.X4` rows every wave since 15
  has reported". Three are; the fourth is **`COMP.MC.MXO0`**. Verified by re-running
  the verifier against this wave's pre-merge backup, where the same four appear, so
  wave 27 introduced nothing — but the count of `CAL` rows is three, not four.

## 4. Two-GPU parallelism

Builds were pipelined on CPU (three concurrent vocabulary batches) while GPU 0 and
GPU 4 ran independent seal+review queues, and the benchmark round was split the
same way. Measured review cost was **113–188 s per child**; 23 children plus 24
benchmark rows completed in **≈75 minutes of wall clock** against **≈150 minutes**
the same work would have taken on one GPU — the parallelism saved about **1 h 15 m**,
close to the ideal 2× because nothing in the review path contends. Sealing and the
ledger merge were deliberately kept single-writer and serial at the end.

## 5. Where the dossiers are

`READ-<TAG>.txt` is the child verbatim, `READ-PARENT-<TAG>.txt` the parent's own
sixteen answers from the same process, `per-case-<TAG>.json` the lane-by-lane
reading, `verdict-<TAG>.json` the sealed ADDENDUM 6 receipt, `dossier-<TAG>.json`
the frozen surface it was read off, `benchmark-table.json` every number in §1's
right-hand block. Nine of twenty-five checkpoints are retained (2.9 GiB on the
share); the sixteen that are not are named with reasons in
`children-retention-manifest-wave27.json`. **The package was built BEFORE
retention**, so all 25 build receipts survive in it — wave 26's ordering mistake was
not repeated.
