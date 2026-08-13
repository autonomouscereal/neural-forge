# Wave 25 — the vocabulary axis is not a cost axis, and the ladder was never being walked

**Deployment** `neural-forge-matrix-e47600a7` (4,740 files, 0 discrepancies).
**Ledger** 682 → **706 rows**, sha
`cd22df335ab2c503ebcf634da253cef66e8e6c22a52ce23e1085d30672408018`,
git blob `94ef44d32dac3f6b4ad9694596cc1ee979e79416`.
**Parent lane** `865f006e3813712ec202cb23` on all twelve dossiers.
**Parent benchmark row** 0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl
**24.39**, wikitext wppl **81.80**, bits/byte **1.1882** — the sealed reference,
thirteenth consecutive match.

Twelve children built, twelve reviewed, twelve benchmarked, twelve sealed.

## 1. Every child, both readings

| child | net stored | params | review | U / P / M | per-lane | caps | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | bits/byte |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V8X13GL03` | 0.691832 | 158,909,184 | **7/15** | 3 / 4 / 8 | impaired (7/16) | 2 | 0.3100 | 0.3850 | 0.5450 | 0.4600 | 0.2550 | 1027.86 | 4053.19 | 2.2412 |
| `ARCH.V12X13` | 0.710093 | 163,103,488 | **7/15** | 1 / 6 / 8 | impaired (7/16) | 0 | 0.3850 | 0.3900 | 0.5900 | 0.4950 | 0.3050 | 442.57 | 1659.06 | 2.0002 |
| `ARCH.V16X13GL03` | 0.728353 | 167,297,792 | **7/15** | 3 / 4 / 8 | impaired (7/16) | 3 | 0.3850 | 0.3950 | 0.6150 | 0.4950 | 0.3000 | 268.29 | 878.99 | 1.8289 |
| `ARCH.V32W896` | 0.731199 | 167,951,488 | **3/15** | 1 / 2 / 12 | impaired (12/16) | 2 | 0.3700 | 0.4150 | 0.5000 | 0.5250 | 0.1500 | 770.57 | 706.58 | 1.7699 |
| `ARCH.V10` | 0.753484 | 173,070,080 | **10/15** | 3 / 7 / 5 | **alive-parent-band** (5/16) | 0 | 0.3600 | 0.3900 | 0.5750 | 0.5100 | 0.3000 | 116.31 | 454.32 | 1.6508 |
| `ARCH.V12` | 0.762614 | 175,167,232 | **11/15** | 3 / 8 / 4 | **alive-parent-band** (5/16) | 0 | 0.3950 | 0.4500 | 0.5800 | 0.5250 | 0.3400 | 94.08 | 334.06 | 1.5678 |
| `ARCH.V14` | 0.771744 | 177,264,384 | **9/15** | 4 / 5 / 6 | alive-degraded (4/16) | 0 | 0.4000 | 0.4250 | 0.5800 | 0.5400 | 0.3450 | 83.71 | 269.74 | 1.5101 |
| `ARCH.V20` | 0.799135 | 183,555,840 | **13/15** | 5 / 8 / 2 | **alive-parent-band** (2/16) | 0 | 0.4300 | 0.4650 | 0.6250 | 0.5300 | 0.3800 | 56.29 | 181.89 | 1.4038 |
| `ARCH.V32X13` | 0.801395 | 184,075,008 | **10/15** | 1 / 9 / 5 | **alive-parent-band** (6/16) | 2 | 0.4650 | 0.4000 | 0.6150 | 0.5100 | 0.3450 | 156.31 | 490.19 | 1.6713 |
| `ARCH.V32X13G` | 0.801395 | 184,075,008 | **10/15** | 1 / 9 / 5 | **alive-parent-band** (4/16) | 3 | 0.4650 | 0.4300 | 0.6150 | 0.5300 | 0.3400 | 128.30 | 379.80 | 1.6025 |
| `ARCH.V32X13GL1` | 0.801395 | 184,075,008 | **10/15** | 2 / 8 / 5 | **alive-parent-band** (4/16) | 3 | 0.4500 | 0.4200 | 0.6150 | 0.5300 | 0.3600 | 115.26 | 422.65 | 1.6313 |
| `ARCH.V32X13GL01` | 0.801395 | 184,075,008 | **9/15** | 2 / 7 / 6 | alive-degraded (5/16) | 3 | 0.4650 | 0.4350 | 0.6000 | 0.5300 | 0.3450 | 122.63 | 386.49 | 1.6072 |
| **PARENT** | 1.000000 | 229,693,184 | **12/15** | 5 / 7 / 3 | — | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 | 1.1882 |

`bits/byte` is the tokenizer-independent column and every child on this page
segments differently from the parent, so it is the only perplexity number on the
page that compares. It is monotone in K and it is the one measurement in the wave
that behaves the way the axis was expected to.

## 2. What the wave found

**a. The vocabulary axis has a cliff, not a slope, and the review does not track
the segmentation shift.** Wave 24's law is falsified twice over by rungs built off
the identical census. K = 12,288 agrees with the parent's segmentation on **9.92%**
of review positions — *less* than K = 16,384's 14.60% — and reads 11 against that
rung's 9. And K = 10,240 agrees on **9.92%**, identical to four places, and reads
10. Seven rungs now exist:

| K | net stored | position agreement | tokens/byte | review |
|---:|---:|---:|---:|---:|
| 32,768 | 0.853916 | 0.3264 | 0.2279 | 11/15 |
| 20,480 | 0.799135 | 0.1736 | 0.2388 | **13/15** |
| 16,384 | 0.780874 | 0.1460 | 0.2473 | 9/15 |
| 14,336 | 0.771744 | 0.1322 | 0.2505 | 9/15 |
| 12,288 | 0.762614 | 0.0992 | 0.2573 | **11/15** |
| 10,240 | 0.753484 | 0.0992 | 0.2655 | 10/15 |
| 8,192 | 0.744354 | 0.0565 | 0.2773 | 4/15 |

The review is non-monotone in the ratio and non-monotone in the agreement, and
the collapse is entirely inside the 2,048 rows between K = 8,192 and K = 10,240.

**b. `ARCH.V20` reads 13/15, one more than the parent.** Zero solve, zero cap
hits, stock `Lfm2ForCausalLM` with `vocab_size` 20,480. Five lanes in full,
including the only uncertainty lane in 25 waves that refuses to conclude and
names all four missing quantities, and the planning lane with all five work items
across six dependency-ordered steps. It loses only the ordering proof and the
four-snack correction. **This is not evidence that folding improves the model** —
see the qualification on the campaign scorecard — it is evidence that at 0.799x
this axis costs nothing this surface can measure.

**c. The lambda atlas was being sampled, not walked, and the fix found two
children.** `mx25_termgain.py` now accepts any positive relative lambda instead
of the nine fixed rungs; the selection ladder is untouched, so wave 24's
split-selected children remain reproducible. The new rung at 0.3 sits in the
ladder's widest gap and is the **peak on both bases that have one**:

| base | no solve | λ 1.0 | **λ 0.3** | λ 0.1 | λ 0.01 |
|---|---:|---:|---:|---:|---:|
| `V16X13` (0.728353x) | 5 | 6 | **7** | 6 | 5 |
| `V8X13` (0.691832x) | 6 | 6 | **7** | 7 | 4 |
| `V32X13` (0.801395x) | 10 | 10 | — | 9 | 10 |

`ARCH.V8X13GL03` matches the child Cyril is scoring — 7/15 at 0.691832x — and
carries **three** Useful lanes to its two. `ARCH.V16X13GL03` is the V16 recovery
the wave was sent to find: 5 → 7 at an identical parameter count.

**d. Wave 24's wider claim about the solver criterion does not replicate.** On
the third base the split-selected rung holds its count (10) and the λ = 0.1 rung
— the one that produced the bar child — is the only rung of four that *loses* a
case. What survives is the narrow claim, and it is the one that matters: **the
solver's out-of-sample error does not order the review.** On `V32X13` the
held-out terminal-logit SSE falls 0.3174 → 0.2510 → 0.2310 → 0.2300 across the
four rungs while the review reads 10, 10, 9, 10.

**e. Prompt-echo is seen a second time.** `ARCH.V32W896` returns the prompt
itself, verbatim, on three of sixteen lanes — the signature wave 24 first saw on
`ARCH.V8CL896` and called new in 662 rows. Unlike that child, this one *also*
hits the cap twice, so prompt-echo is not an alternative to looping and not a
property of one body.

**f. Two lanes no child had ever held are now held.** `coding-deduplicate`
(`ARCH.V32X13`, `ARCH.V20`) and `creative-continuity` (`ARCH.V14`, `ARCH.V20`).
The reading criterion for each is written into `run/mx25/sealq.sh` and was
calibrated by re-grading wave 24's `ARCH.V32` dossier blind: it came out 6 / 5 /
4 = 11/15, lane for lane what wave 24 sealed.

## 3. SEGBASIS (S2): the pre-read ran, and it is not encouraging

`run/mx25/mx25_segcensus.py`, pre-read **P2**, over all thirteen cuts at three
widths, from one capture pass (36 s, 24 forward calls). Best cut at every width
is c = 2, and at every width the arithmetic reads the same way:

| width | worst reader, global | worst reader, segmented | gain | seam identity cost |
|---:|---:|---:|---:|---:|
| 896 | 0.9580 (`B01.ffn`) | 0.9602 | **+0.0022** | 0.0493 |
| 768 | 0.9053 | 0.9085 | **+0.0032** | 0.1091 |
| 736 | 0.8908 | 0.8936 | **+0.0028** | 0.1257 |

The segmented basis buys two to three thousandths of the worst reader's retained
read energy and the identity seam it must pay for costs **twenty to forty-five
times that**. The minimum principal-angle cosine between the two segment bases is
under 0.007 at every cut and every width — they are nearly orthogonal in at least
one direction, which is exactly why passing the accumulator through unchanged is
expensive. Also worth recording against the spec: at width 896 **no reader is
under 0.90 at all** (min 0.9580, mean 0.9729), so S2's stated wall does not exist
at the control shape wave 25 was told to build at.

The census **orders and does not gate**, so the build remains permitted; it was
not spent this wave and the reason is the table above. The T\* transition twin
that S2 makes mandatory is a separate, unscoped blocker: it is a forward hook, and
`seal_matrix_architecture_child.py` performs a stock `from_pretrained` load, so
the twin cannot go through the ARCH review path without harness work nobody has
costed.

## 4. Where the dossiers are

`READ-<TAG>.txt` is the child verbatim, `READ-PARENT-<TAG>.txt` the parent's own
sixteen answers from the same process, `per-case-<TAG>.json` the lane-by-lane
reading, `verdict-<TAG>.json` the sealed ADDENDUM 6 receipt, `dossier-<TAG>.json`
the frozen surface it was all read off. `P2-segbasis-cuts*.json` are the census
above. `benchmark-table.json` is every number in §1's right-hand block.
