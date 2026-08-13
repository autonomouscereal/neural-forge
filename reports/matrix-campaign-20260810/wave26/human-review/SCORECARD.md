# Wave 26 — the thirteen is a shelf, the cliff is 1,024 rows wide, and the base carries a second fold

**Deployment** `neural-forge-matrix-b6f79b56` (4,870 files, 0 discrepancies before
the wave). **Verified again AFTER the wave** — a thing no wave has done before —
and the answer is worth having: **0 content mismatches, 0 missing, 0 hardlinks,
135 undeclared**, and all 135 are `__pycache__` bytecode written by running the
tree. The declared bytes are still the commit's, so the sealing claim holds; the
verifier refuses anyway, correctly, by its own rule. Recorded in
`post-wave-deployment-verification-b6f79b56.json` rather than fixed, because a
verifier that learns to ignore a directory is a verifier with an exception in it.
**Ledger** 706 → **724 rows**, sha
`de12c860276e686eb519dad3161cf115c9d3b07ba149b349599513b2028db49b`.
Byte-prefix of the pre-wave backup verified, 37,901 bytes appended, no row rewritten.
**Seals** 720 of 724 rows resolve to a receipt on disk and all 720 verify; 0 fail;
the 4 that do not resolve are the same `CAL.C01.T5.X4` rows every wave since 15
has reported. **22 cell roots audited against disk, 0 undeclared** — `wave25/cells`
still declared, `wave26/cells` added as ONE line.
**Parent lane** `865f006e3813712ec202cb23` on all nine dossiers, protocol
`e5e377688597` / pack `0ee6bf14cef5`.
**Parent benchmark row** 0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl
**24.39**, wikitext wppl **81.80**, bits/byte **1.1882** — the sealed reference,
**fourteenth consecutive match**.

Nine children built, nine reviewed, nine benchmarked, nine sealed.

## 1. Every child, both readings

| child | net stored | params | review | U / P / M | per-lane | caps | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | bits/byte |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V20X13W896` | 0.639307 | 146,844,416 | **2/15** | 1 / 1 / 13 | impaired (14/16) | 5 | 0.3100 | 0.4150 | 0.5850 | 0.5250 | 0.1650 | 1902.95 | 2402.80 | 2.1002 |
| `ARCH.V20W896` | 0.683266 | 156,941,440 | **2/15** | 0 / 2 / 13 | impaired (13/16) | 1 | 0.2950 | 0.4150 | 0.5400 | 0.5050 | 0.1300 | 1067.87 | 1016.05 | 1.8679 |
| `ARCH.V20X13X07` | 0.694093 | 159,428,352 | **5/15** | 0 / 5 / 10 | impaired (10/16) | 4 | 0.3650 | 0.3950 | 0.6200 | 0.4750 | 0.1200 | 3068.64 | 1075.19 | 1.8832 |
| `ARCH.V20X13` | 0.746614 | 171,492,096 | **10/15** | 3 / 7 / 5 | **alive-parent-band** (6/16) | 1 | 0.3850 | 0.4000 | 0.6050 | 0.5150 | 0.3050 | 269.13 | 797.65 | 1.8027 |
| `ARCH.V20X13G03` | 0.746614 | 171,492,096 | **11/15** | 2 / 9 / 4 | **alive-parent-band** (4/16) | 2 | 0.3850 | 0.4150 | 0.5950 | 0.5250 | 0.3100 | 201.79 | 664.80 | 1.7535 |
| `ARCH.V9` | 0.748919 | 172,021,504 | **9/15** | 2 / 7 / 6 | alive-degraded (8/16) | 0 | 0.3200 | 0.4000 | 0.5600 | 0.5050 | 0.2800 | 148.17 | 585.31 | 1.7191 |
| `ARCH.V18` | 0.790005 | 181,458,688 | **13/15** | 5 / 8 / 2 | **alive-parent-band** (5/16) | 0 | 0.4200 | 0.4250 | 0.6250 | 0.5300 | 0.3650 | 66.26 | 200.84 | 1.4306 |
| `ARCH.V32X13G03` | 0.801395 | 184,075,008 | **10/15** | 4 / 6 / 5 | **alive-parent-band** (6/16) | 3 | 0.4600 | 0.4350 | 0.6050 | 0.5300 | 0.3550 | 117.58 | 399.12 | 1.6158 |
| `ARCH.V22` | 0.808265 | 185,652,992 | **13/15** | 4 / 9 / 2 | **alive-parent-band** (5/16) | 0 | 0.4550 | 0.4750 | 0.6250 | 0.5300 | 0.3800 | 55.08 | 165.68 | 1.3787 |
| **PARENT** | 1.000000 | 229,693,184 | **12/15** | 5 / 7 / 3 | — | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 1.1882 |

`bits/byte` is the tokenizer-independent column and every child on this page
segments differently from the parent, so it is the only perplexity number here
that compares across rows. It is monotone in the review on this page with one
exception (`V20X13X07`), which is the first time in two waves it has not been.

## 2. What the wave found

**a. Wave 25's thirteen is a SHELF, not a spike.** K = 18,432 and K = 22,528 were
built for one reason — to find out whether 20,480 was a lucky table — and both
read **13/15**, zero cap hits, zero solve:

| K | net stored | review | lambada ppl | bits/byte |
|---:|---:|---:|---:|---:|
| 32,768 | 0.853916 | 11/15 | 156.31 | 1.6713 |
| **22,528** | **0.808265** | **13/15** | 55.08 | 1.3787 |
| **20,480** | **0.799135** | **13/15** | 56.29 | 1.4038 |
| **18,432** | **0.790005** | **13/15** | 66.26 | 1.4306 |
| 16,384 | 0.780874 | 9/15 | 268.29 | 1.8289 |
| 14,336 | 0.771744 | 9/15 | 83.71 | 1.5101 |
| 12,288 | 0.762614 | 11/15 | 442.57 | 2.0002 |
| 10,240 | 0.753484 | 10/15 | 116.31 | 1.6508 |
| **9,216** | **0.748919** | **9/15** | 148.17 | 1.7191 |
| 8,192 | 0.744354 | 4/15 | 1027.86 | 2.2412 |

The campaign's highest review is a **4,096-row-wide plateau** spanning 0.790 to
0.808, and the deepest point of it is `ARCH.V18` at 0.790005x — nine thousandths
deeper than the row wave 25 sealed. This does not make the thirteen a claim about
intelligence; it retires exactly one of the four reasons wave 25 gave for
doubting it.

**b. The vocabulary axis dies inside 1,024 rows.** `ARCH.V9` reads **9/15**.
Wave 25 could only say the collapse was somewhere in the 2,048 rows between 8,192
(4/15) and 10,240 (10/15). It is in the lower half: 9,216 is still on the living
side, and 1,024 rows below it the axis reads four. Nothing about the ratio marks
that boundary — 0.744354 to 0.748919 is four thousandths.

**c. The 13/15 base carries a second fold, which is the answer wave 25 could not
give about its own number.** `ARCH.V20X13` — the 20,480-row coreset on the
block-13 excision, zero solve on either axis — reads **10/15 at 0.746614x**, the
**deepest ten in 26 waves by 5.5 points of ratio**. A base that is a measurement
artefact does not survive a second fold; this one does. What the excision costs is
**three** cases, not the one it cost the 32,768 rung, and it takes them out of
*completeness*: V20 answered five lanes in full and this child answers three.

**d. `ARCH.V20X13G03` is the deepest ELEVEN in the campaign.** One 1,024-entry
diagonal re-solved at relative lambda 0.3, zero parameters added, zero shapes
moved, an identical 171,492,096 parameters — and 11/15 against its source's ten.
The previous deepest eleven was `ARCH.V12` at 0.762614x, so this is 1.6 points
deeper at the same count. The solve buys **breadth, not depth**: it converts four
missed lanes to Partial and gives up one Useful, and it gives up
`coding-deduplicate` on the container — the source accumulates into a LIST, this
rung into a SET, which is the exact defect wave 25 wrote the criterion around.

**e. The lambda atlas has no fixed peak and no fixed SHAPE.** Four ladders now
exist and no two agree:

| base | net | no solve | λ 1.0 | **λ 0.3** | λ 0.1 | λ 0.01 |
|---|---:|---:|---:|---:|---:|---:|
| `V16X13` | 0.728353 | 5 | 6 | **7** | 6 | 5 |
| `V8X13` | 0.691832 | 6 | 6 | **7** | 7 | 4 |
| `V32X13` | 0.801395 | 10 | 10 | **10** | 9 | 10 |
| `V20X13` | 0.746614 | 10 | — | **11** | — | — |

Unimodal on two bases, **flat on `V32X13`**, and rising on `V20X13`. Wave 25
proposed λ = 0.3 as the peak; it is the peak or the joint peak on all four, and on
`V32X13` it buys nothing at all. The held SET moves even where the count does not:
`V32X13G03` reads the same ten as its source while holding **four** lanes in full
against the source's one.

**f. The third axis is fatal, again, and this time on the best base in the
campaign.** `ARCH.V20X13X07` — 20,480 rows on the block-13 **and** block-7 double
excision — reads **5/15, all Partial, nothing in full**. V20 alone reads 13, the
double excision alone read 6 in wave 17, and composed they read 5. The second
block costs five cases on top of the three the first one cost.

**g. Prompt-echo is a WIDTH-axis signature.** `ARCH.V20W896` returns the prompt
itself on the arithmetic and ordering lanes. That is the third sighting, after
wave 24's `ARCH.V8CL896` and wave 25's `ARCH.V32W896`, and **all three are width
children**. On the same base in the same wave, the block excision costs the 20,480
rung three cases and the width fold costs it **eleven**.

**h. `ARCH.V20X13W896` is the deepest child ever to answer a lane in FULL.** Three
axes at 0.639307x, 2/15 — and its summarisation lane names the decision, the
one-week expansion risk and the weekly schedule evidence, complete and correct, at
64% of the parent. It strictly dominates wave 23's `ARCH.V16W896` (2/15 at
0.667288x) on both axes at once.

**i. An excision rebuilt bit-identically across nine waves.**
`mx26_build.py --excise 13 --excise 7` reproduces wave 17's sealed `ARCH.P1307`
exactly: `child_tensors_sha256 329167100d024d33883235090c6f5e548d70a157c1da64ca9b01c9705a30fba5`,
205,565,696 parameters, through a rewritten excision path and a different
deployment. So `V20X13X07` carries wave 17's own sealed cell receipt rather than a
driver's re-description of it.

## 3. What went wrong, recorded rather than smoothed

* **Wave 25 never committed its merged ledger.** Its scorecard cites 706 rows at
  sha `cd22df33` and git blob `94ef44d3` — all correct — but the repository stood
  at 682. Fixed in commit `c817308b` before any wave-26 work, with the byte-prefix
  property checked rather than assumed.
* **Retention ran before the package on this wave**, which deleted three children
  and with them their build receipts. All three were rebuilt and every one
  reproduced its **model tensors byte for byte** against the sealed cell receipt;
  the receipt FILE digest differs, and `wave26-drop-reproduction.json` records
  which fields are non-deterministic (`built_at`, `elapsed_seconds`,
  `child_model`), measured by rebuilding one child twice and diffing.
* **`mx*_compare.py` has raised `KeyError: net_ratio` on every vocabulary-carried
  width child since wave 25**, so the width comparison table was silently absent
  from wave 25's package. Wave 25 built one such child and shipped without
  noticing. Fixed; both of this wave's width children are in
  `wave26-child-comparison.json`.
* **The bare integer `--campaign-wave 25`** in `seal_and_review.sh` is invisible to
  the token rename that cuts each wave's tree. Caught by name before the first
  seal, and asserted on the first sealed receipt (`campaign_wave = 26`).

## 4. Where the dossiers are

`READ-<TAG>.txt` is the child verbatim, `READ-PARENT-<TAG>.txt` the parent's own
sixteen answers from the same process, `per-case-<TAG>.json` the lane-by-lane
reading, `verdict-<TAG>.json` the sealed ADDENDUM 6 receipt, `dossier-<TAG>.json`
the frozen surface it was read off, `benchmark-table.json` every number in §1's
right-hand block. Seven of nine checkpoints are retained on the share; the three
that are not are named, with reasons and with their reproduction check, in
`children-retention-manifest-wave26.json` and `wave26-drop-reproduction.json`.
