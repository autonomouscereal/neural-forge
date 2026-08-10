# Stage-1 matrix campaign -- ledger

Rendered 2026-08-10T19:47:36+00:00 from `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`.

**59 cells sealed** (126 ledger rows including re-runs); newest seal 2026-08-10T19:47:15.145+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 11 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `fail` | 46 | crossed a chain threshold; see first_divergent_metric |
| `infra` | 2 | calibration refused the cell -- void, not evidence |

## 1. The matrix

Rows are components, columns are techniques.  A position shows the deepest level whose cells all passed, or the shallowest level attempted and the chain link it broke at.  `.` is untested.

| component | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|
| C01 token_interface_in | . | . | . | . | X4:free | . | . | . | . |
| C03 conv_in_proj | . | . | . | . | M16:comp | . | . | . | . |
| C04 conv_kernel | M16:infra | . | . | . | . | . | . | . | . |
| C05 conv_out_proj | M16:comp | M16:comp | M16:comp | M16:solve | M16:comp | . | . | M16:comp | . |
| C06 attn_q | PASS@M8 | . | . | . | . | . | PASS@M8 | . | . |
| C07 attn_k | PASS@M16 | . | . | . | . | . | . | . | . |
| C08 attn_v | M16:free | . | . | . | M16:comp | . | . | . | . |
| C09 attn_o | M16:free | M16:free | . | M16:solve | M16:comp | . | . | M16:comp | . |
| C10 attn_per_head | M16:infra | . | . | . | . | . | . | . | . |
| C11 mlp_gate | . | . | . | . | M16:comp | . | . | . | . |
| C12 mlp_up | . | . | . | . | M16:comp | . | . | . | . |
| C13 mlp_down | M16:comp | M16:comp | M16:comp | . | M16:comp | . | . | M16:comp | . |
| C14 norms | PASS@NA | . | . | . | . | . | . | . | . |

### Calibration row

| cell | status | free-run vs control | notes |
|---|---|---:|---|
| `CTRL.copy_exact` | PASS | 1.0000 | decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verba... |
| `CAL.C01.T5.X4` | FAIL | 0.1250 | decode-smoke review by claude-opus-5 (frozen V5 protocol e5e37768) of receipt 4018355521b93a30: Frozen-V5 q... |

## 2. Level ladders

§5 rule 2: a technique+component works *at level L* only if it passes at L and every shallower level, at >= 2 of its 3 sites.  A pass deeper than a fail is flagged `non-monotone` and does not count.

| component | technique | M16 | M8 | M4 | M2 | X4 | verdict |
|---|---|---|---|---|---|---|---|
| C01 | T5 | . | . | . | . | 0/1 | fails at free-run |
| C03 | T5 | 0/1 | . | . | . | . | fails at composition |
| C04 | T1 | 0/1 | . | . | . | . | void (infra) -- no evidence |
| C05 | T1 | 0/3 | 0/1 | . | 0/1 | . | fails at composition |
| C05 | T2 | 0/1 | . | . | . | . | fails at composition |
| C05 | T3 | 0/1 | . | . | . | . | fails at composition |
| C05 | T4 | 0/1 | . | . | . | . | fails at local-solve |
| C05 | T5 | 0/3 | 0/1 | . | . | . | fails at composition |
| C05 | T8 | 0/1 | . | . | . | . | fails at composition |
| C06 | T1 | 3/3 | 1/1 | 0/1 | 0/1 | . | works to M8 |
| C06 | T7 | 1/3 | 1/1 | 0/1 | . | . | non-monotone (not working) |
| C07 | T1 | 3/3 | 0/1 | 0/1 | 0/1 | . | works to M16 |
| C08 | T1 | 0/1 | 0/1 | 0/1 | . | . | fails at composition |
| C08 | T5 | 0/1 | . | . | . | . | fails at composition |
| C09 | T1 | 0/2 | 0/1 | 0/1 | . | . | fails at composition |
| C09 | T2 | 0/1 | . | . | . | . | fails at free-run |
| C09 | T4 | 0/1 | . | . | . | . | fails at local-solve |
| C09 | T5 | 0/1 | . | . | . | . | fails at composition |
| C09 | T8 | 0/1 | . | . | . | . | fails at composition |
| C10 | T1 | 0/1 | . | . | . | . | void (infra) -- no evidence |
| C11 | T5 | 0/1 | . | . | . | . | fails at composition |
| C12 | T5 | 0/1 | . | . | . | . | fails at composition |
| C13 | T1 | 0/3 | . | . | . | . | fails at composition |
| C13 | T2 | 0/1 | . | . | . | . | fails at composition |
| C13 | T3 | 0/1 | . | . | . | . | fails at composition |
| C13 | T5 | 0/3 | . | . | . | . | fails at composition |
| C13 | T8 | 0/1 | . | . | . | . | fails at composition |
| C14 | T1 | . | . | . | . | . | awaiting review |

## 3. Failure classes

The chain is walked mechanically (§5 rule 5); this histogram is the campaign's failure analysis in full.  `free-run` is the F3 signature -- a cell that cleared every numeric link and still lost behavior -- and is the most information-dense class in the matrix.

| first divergent metric | cells | share |
|---|---:|---:|
| `infra` | 2 | 4% |
| `local-solve` | 3 | 6% |
| `composition` | 35 | 73% |
| `free-run` | 8 | 17% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 11 | 43 |
| out of band | 0 | 0 |

Agreement 11/54 (20%).

## 4b. Storage honesty

A level names a *rank*, and a rank is not a size.  A technique whose stored form is a `w x r` basis plus an `r x w` solved factor stores `r * (in + out)` against a dense `in * out`, so on a square component it does not contract at all until `r < w/2`.  Cells above 1.0 below cost more storage than the tensor they replaced.

| cell | technique | level | stored | dense parent | stored/dense |
|---|---|---|---:|---:|---:|
| `C03.T5.M16.L00` | T5 | M16 | 2949632 | 3145728 | 0.9377 |
| `C05.T1.M8.L05` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T1.M2.L05` | T1 | M2 | 1048576 | 1048576 | 1.0000 |
| `C05.T2.M16.L05` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L05` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T5.M8.L05` | T5 | M8 | 917504 | 1048576 | 0.8750 |
| `C06.T1.M16.L08` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C06.T1.M16.L12` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C06.T1.M8.L02` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C06.T1.M4.L02` | T1 | M4 | 1572864 | 1048576 | 1.5000 |
| `C06.T1.M2.L02` | T1 | M2 | 1048576 | 1048576 | 1.0000 |
| `C06.T7.M16.L12` | T7 | M16 | 1044480 | 1048576 | 0.9961 |
| `C06.T7.M8.L02` | T7 | M8 | 974848 | 1048576 | 0.9297 |
| `C06.T7.M4.L02` | T7 | M4 | 835584 | 1048576 | 0.7969 |
| `C07.T1.M16.L08` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M16.L12` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C07.T1.M2.L02` | T1 | M2 | 393216 | 524288 | 0.7500 |
| `C08.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C08.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C09.T1.M16.L08` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T1.M8.L02` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C09.T1.M4.L02` | T1 | M4 | 1572864 | 1048576 | 1.5000 |
| `C09.T2.M16.L02` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C13.T1.M16.L13` | T1 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T2.M16.L05` | T2 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T3.M16.L05` | T3 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T5.M16.L05` | T5 | M16 | 2457600 | 2621440 | 0.9375 |
| `C14.T1.NA.L00` | T1 | NA | 1024 | 2048 | 0.5000 |

8 of 30 sealed cells with a storage reading actually contract.

## 5. Per-cell drill-down

### `CAL.C01.T5.X4` -- FAIL

token_interface_in / sparse_omp_routes / level X4 / site GLOBAL / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5b8e6ee3` | -- | fail | free-run | -- | -- | -- | 0.125 | 0.327 | -- | 1.0 | 76.9 | `28564b7156f8` |
| `5b8e6ee3` | -- | fail | free-run | -- | -- | -- | 0.125 | 0.327 | -- | 1.0 | 76.9 | `4018355521b9` |
| `0305653c` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 1321.3 | `276fd2e49dd4` |
| `c9fcf7bd` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 26.7 | `c079146235cd` |
| `d0e0fdaf` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 5.3 | `a9eef18a3639` |
| `3871e23b` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 5.2 | `9b0cc7b841a5` |

> decode-smoke review by claude-opus-5 (frozen V5 protocol e5e37768) of receipt 4018355521b93a30: Frozen-V5 quality-band review, reports/matrix-campaign-20260810/cells/CAL.C01.T5.X4-mxc/QUALITY_BAND_REVIEW.md. Whole-response band 8/15 useful-or-partial (3 useful / 5 partial / 7 observed miss), identical case-for-case to the sealed A9 child re-graded at the same 256 cap (8/15, 3/5/7) and to the frontier re-score's published A9 band (8/15) -- band delta 0, so the calibration criterion of quality-band agreement PASSES at any tolerance. The three receipt decode-smoke rows are arithmetic-multi-step, coding-deduplicate and comparison-ram-rom, all three Observed miss on both children, hence verdict incoherent; the receipt's status is fail on chain link 6 (free-run 0.125) with or without this verdict.

### `C03.T5.M16.L00` -- FAIL

conv_in_proj / sparse_omp_routes / level M16 / site L00 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fbc908bf` | sm86 | fail | composition | -- | -- | 0.0276 | 0.787 | 0.684 | 0.938 | 15.8 | 29.2 | `afb675cc1809` |
| `fbc908bf` | sm86 | fail | composition | -- | -- | 0.0276 | 0.787 | 0.684 | 0.938 | 15.8 | 29.2 | `cbd74797fe81` |
| `5623aca1` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 35.6 | `2907d224b20f` |

> decode-smoke review by ClaudeCode-MXF of receipt cbd74797fe81e3f2: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C04.T1.M16.L00` -- INFR

conv_kernel / dense_prior_anchored_ridge / level M16 / site L00 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2dadae7a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.3 | `a30591546f77` |
| `f5c7bf74` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 34.9 | `6001d2de2582` |

> CellTechniqueError: T1's reference kernel contracts linear maps; C04 has axis 'kernel' and needs its own kernel.

### `C05.T1.M16.L00` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5e4ad7d6` | -- | fail | composition | 0.9984 | 0.0395 | 0.0437 | 0.686 | 0.666 | -- | 33734.7 | 50.0 | `cd3fbfaaed26` |
| `5e4ad7d6` | -- | fail | composition | 0.9984 | 0.0395 | 0.0437 | 0.686 | 0.666 | -- | 33734.7 | 50.0 | `9414632eb139` |

> decode-smoke review by claude-opus-5 of receipt 9414632eb13903ef: 3/3 on task, 2 of 3 on EOS. (1) circuit breaker: correct, with one circular clause ('shutting it off to prevent a potential overload'). (2) one-pan dinner: a real single-dish plan with an ingredient list, cap-censored inside 'Preheat oven to 400F' -- censored, not failed, though preheating an oven misses the one-pan constraint. (3) neighbour note: the control's text verbatim, carrying its 'for your convenience' slip. The dense-ridge baseline at the same site as wave-1 cell 1 reads no worse than the T5 route did; the cell still fails at chain link 5 (composed drift 0.0437).

### `C05.T1.M16.L05` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level M16 / site L05 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `86525729` | -- | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | -- | 6684.9 | 49.9 | `ec491321ddda` |
| `86525729` | -- | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | -- | 6684.9 | 49.9 | `b6e60173b846` |

> decode-smoke review by claude-opus-5 of receipt b6e60173b8462c8f: 3/3 on task, all EOS. (1) circuit breaker: correct mechanism, though 'sending a signal to the power supply to shut off the circuit' is a mechanism the parent does not claim. (2) one-pan dinner: a genuine single-dish plan reaching 'Prep the pan' before the cap. (3) neighbour note: adds 'Please let me know if there's any specific reason for this', which is backwards -- the writer is the one making the request. Fails at chain link 5, composed drift 0.0387.

### `C05.T1.M16.L13` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level M16 / site L13 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `eb0f466f` | -- | fail | composition | 0.9964 | 0.0155 | 0.0339 | 0.734 | 0.709 | -- | 18964.5 | 49.8 | `45939feaef97` |
| `eb0f466f` | -- | fail | composition | 0.9964 | 0.0155 | 0.0339 | 0.734 | 0.709 | -- | 18964.5 | 49.8 | `85876702140c` |

> decode-smoke review by claude-opus-5 of receipt 85876702140c6201: 3/3 on task, all EOS. (1) circuit breaker: correct and the cleanest of the three T1 conv-writer sites. (2) one-pan dinner: a plan whose ingredient list includes cooked brown rice and two broths and then preheats an oven -- on task, cap-censored, one-pan constraint missed. (3) neighbour note: the control's text plus 'Thank you for your time and consideration'. Fails at chain link 5, composed drift 0.0339.

### `C05.T1.M8.L05` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level M8 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `93613891` | sm86 | fail | composition | 0.9924 | 0.0376 | 0.0539 | 0.746 | 0.746 | 1.750 | 4650.8 | 29.4 | `9bf16cff4463` |
| `93613891` | sm86 | fail | composition | 0.9924 | 0.0376 | 0.0539 | 0.746 | 0.746 | 1.750 | 4650.8 | 29.4 | `f491602b64d1` |

> decode-smoke review by ClaudeCode-MXF of receipt f491602b64d1ec91: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T1.M2.L05` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level M2 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0f8c2006` | sm86 | fail | local-solve | 0.9482 | 0.0991 | 0.1185 | 0.246 | 0.197 | 1.000 | 1104.4 | 27.3 | `65da2b825d24` |
| `0f8c2006` | sm86 | fail | local-solve | 0.9482 | 0.0991 | 0.1185 | 0.246 | 0.197 | 1.000 | 1104.4 | 27.3 | `2a26719259f3` |

> decode-smoke review by ClaudeCode-MXF of receipt 2a26719259f30feb: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T2.M16.L05` -- FAIL

conv_out_proj / cross_width_residual_charts / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `99979d513ff3` |
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `6e59fd005767` |

> decode-smoke review by ClaudeCode-MXF of receipt 6e59fd005767d38a: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M16.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dca9532d` | sm86 | fail | composition | 0.9975 | 0.0220 | 0.0268 | 0.914 | 0.900 | 1.875 | 10573.8 | 41.9 | `6331ec8de0c5` |
| `dca9532d` | sm86 | fail | composition | 0.9975 | 0.0220 | 0.0268 | 0.914 | 0.900 | 1.875 | 10573.8 | 41.9 | `2ae667eaaeec` |

> decode-smoke review by ClaudeCode-MXF of receipt 2ae667eaaeecaa37: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T4.M16.L13` -- FAIL

conv_out_proj / low_rank_factor_writers / level M16 / site L13 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bc7eb877` | -- | fail | local-solve | 0.8027 | 0.1121 | 0.1979 | 0.225 | 0.170 | -- | 7068411.0 | 25.9 | `d2872dfefd2a` |
| `bc7eb877` | -- | fail | local-solve | 0.8027 | 0.1121 | 0.1979 | 0.225 | 0.170 | -- | 7068411.0 | 25.9 | `5ecf8394acb4` |

> decode-smoke review by claude-opus-5 of receipt 5ecf8394acb44c30: 3/3 on task and visibly degraded, which is what writer_rank 31 on a 1024x1024 writer buys. (1) circuit breaker: 'a safety valve that stops a circuit from running when it's too hot or too low' -- 'too low' is wrong and the thermal framing replaces the current one, but the trip-to-protect core survives. (2) one-pan dinner: a real plan, cap-censored, with a hedged ingredient list ('fresh or frozen' five times). (3) neighbour note: reshaped into a subject-lined email that asks to 'help with parking' rather than to move the trailer -- the request is softened past the instruction. The cell already fails at chain link 3 (write cosine 0.803 < 0.95); the text degrades without collapsing, unlike C09.T4 at the same rank.

### `C05.T5.M16.L00` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5304bb14` | -- | fail | composition | 0.9991 | 0.0367 | 0.0436 | 0.662 | 0.627 | -- | 834.0 | 52.0 | `5d6653040be3` |
| `5304bb14` | -- | fail | composition | 0.9991 | 0.0367 | 0.0436 | 0.662 | 0.627 | -- | 834.0 | 52.0 | `853ac00d38a0` |

> decode-smoke review by claude-opus-5 of receipt 853ac00d38a0bfb7: 3/3 on task, all EOS. (1) circuit breaker: correct mechanism (overload, trips, stops the flow), mildly redundant. (2) dinner plan: a real 40-minute claim but the answer is a ten-dish menu with no cooking order and no single pan -- both stated constraints missed, task understood. (3) neighbour note: polite, asks for the move, not four lines and the offer to assist is filler. Coherent with instruction misses; the cell still fails at chain link 5 (composition 0.0436 > 0.02).

### `C05.T5.M16.L05` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L05 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20e3c653` | -- | fail | composition | 0.9969 | 0.0226 | 0.0405 | 0.791 | 0.771 | -- | 656.1 | 52.7 | `1c4a09cd549a` |
| `20e3c653` | -- | fail | composition | 0.9969 | 0.0226 | 0.0405 | 0.791 | 0.771 | -- | 656.1 | 52.7 | `5b250adb83f0` |

> decode-smoke review by claude-opus-5 of receipt 5b250adb83f0a814: 3/3 on task. (1) circuit breaker: correct mechanism plus fault detection and the overheating/fire consequence. (2) one-pan dinner: a genuine single-dish plan with ingredient list, censored at the 128-token cap inside 'Instructions: 1. Prepare' -- censored, not failed. (3) neighbour note: polite and correctly shaped; 'for your convenience' is the same wording slip the copy_exact control made. The cell still fails at chain link 5 (composition 0.0405 > 0.02).

### `C05.T5.M16.L13` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L13 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dd23c9d2` | -- | fail | composition | 0.9958 | 0.0166 | 0.0295 | 0.688 | 0.686 | -- | 3359.7 | 53.2 | `be533d1bd1bd` |
| `dd23c9d2` | -- | fail | composition | 0.9958 | 0.0166 | 0.0295 | 0.688 | 0.686 | -- | 3359.7 | 53.2 | `12c4561c0f9b` |

> decode-smoke review by claude-opus-5 of receipt 12c4561c0f9b26b8: 3/3 on task. (1) circuit breaker: correct mechanism, longest and cleanest of the three cells. (2) one-pan dinner: structured plan with a prep step, censored at the cap; 'Cook the pan' is malformed and 180F is not medium-low. (3) neighbour note: polite, asks for the move, carries the same 'for your convenience' slip as L05. The cell still fails at chain link 5 (composition 0.0295 > 0.02).

### `C05.T5.M8.L05` -- FAIL

conv_out_proj / sparse_omp_routes / level M8 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `37d9d350` | sm86 | fail | composition | 0.9917 | 0.0382 | 0.0547 | 0.664 | 0.646 | 0.875 | 732.4 | 27.0 | `dee173f6f001` |
| `37d9d350` | sm86 | fail | composition | 0.9917 | 0.0382 | 0.0547 | 0.664 | 0.646 | 0.875 | 732.4 | 27.0 | `9d54f229342a` |

> decode-smoke review by ClaudeCode-MXF of receipt 9d54f229342a6c55: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T8.M16.L00` -- FAIL

conv_out_proj / gram_relation_matching / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f8038b65` | -- | fail | composition | 0.9987 | 0.0411 | 0.0543 | 0.523 | 0.342 | -- | 48638.1 | 34.2 | `f1dcd461137d` |
| `f8038b65` | -- | fail | composition | 0.9987 | 0.0411 | 0.0543 | 0.523 | 0.342 | -- | 48638.1 | 34.2 | `b95d24ebc6f1` |

> decode-smoke review by claude-opus-5 of receipt b95d24ebc6f16c99: 3/3 on task, all EOS, and the weakest smoke of the three C05 techniques at L00. (1) circuit breaker: correct, but 'typically installed in high-voltage circuits, such as those used in power grids' is a false narrowing -- breakers are ubiquitous at household voltage. (2) dinner: a five-item menu that opens with 'Start with a hearty breakfast: omelette or pancakes' for a dinner prompt, then asserts '40 minutes and one pan' without a cooking order -- on task, both stated constraints missed, and the breakfast opener is the first content error of its kind in the campaign. (3) neighbour note: three sentences rather than four, and it offers to assist rather than asking. Fails at chain link 5, composed drift 0.0543; free-run 0.523 is the worst of the three techniques at this site.

### `C06.T1.M16.L02` -- PASS

attn_q / dense_prior_anchored_ridge / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fe25157c` | -- | pass | -- | 1.0000 | 0.0066 | 0.0012 | 0.977 | 0.971 | -- | 240109510.2 | 51.2 | `77dad33570ca` |
| `fe25157c` | -- | pending-review | -- | 1.0000 | 0.0066 | 0.0012 | 0.977 | 0.971 | -- | 240109510.2 | 51.2 | `f16d917dfa88` |

> decode-smoke review by claude-opus-5 of receipt f16d917dfa88c92e: 3/3 on task. (1) circuit breaker: correct mechanism (overload detected, trips, cuts power), censored at the 128-token cap mid-sentence -- censored, not failed; more redundant than the control's but not wrong anywhere. (2) one-pan dinner: BYTE-IDENTICAL to the copy_exact control's 417-character output, cap-censored at the same point. (3) neighbour note: BYTE-IDENTICAL to the control, including the control's own 'for your convenience' wording slip. Two of three outputs are the parent's, character for character, on the strongest cell in wave 1 (free-run 0.977, composed drift 0.0012 against the 0.02 gate). PASS under section 5 rule 1: free-run 0.977 >= 0.90, smoke 3/3 coherent, emit-zero floor passed.

### `C06.T1.M16.L08` -- PASS

attn_q / dense_prior_anchored_ridge / level M16 / site L08 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `94c8ed46` | sm86 | pass | -- | 1.0000 | 0.0067 | 0.0011 | 0.977 | 0.971 | 1.875 | 31993.0 | 28.8 | `94114e90fbd7` |
| `94c8ed46` | sm86 | pending-review | -- | 1.0000 | 0.0067 | 0.0011 | 0.977 | 0.971 | 1.875 | 31993.0 | 28.8 | `2a871eae880c` |

> decode-smoke review by ClaudeCode-MXF of receipt 2a871eae880ccf91: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.M16.L12` -- PASS

attn_q / dense_prior_anchored_ridge / level M16 / site L12 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f283f4c5` | sm86 | pass | -- | 1.0000 | 0.0026 | 0.0015 | 1.000 | 1.000 | 1.875 | 27233.8 | 53.6 | `b67d70c59771` |
| `f283f4c5` | sm86 | pending-review | -- | 1.0000 | 0.0026 | 0.0015 | 1.000 | 1.000 | 1.875 | 27233.8 | 53.6 | `c462408870ac` |

> decode-smoke review by ClaudeCode-MXF of receipt c462408870ac2219: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.M8.L02` -- PASS

attn_q / dense_prior_anchored_ridge / level M8 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `986a3b25` | sm86 | pass | -- | 1.0000 | 0.0139 | 0.0029 | 0.971 | 0.957 | 1.750 | 240109510.2 | 27.7 | `fcd949f5570b` |
| `986a3b25` | sm86 | pending-review | -- | 1.0000 | 0.0139 | 0.0029 | 0.971 | 0.957 | 1.750 | 240109510.2 | 27.7 | `b5fb7c5edfb7` |

> decode-smoke review by ClaudeCode-MXF of receipt b5fb7c5edfb7a8d3: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.M4.L02` -- FAIL

attn_q / dense_prior_anchored_ridge / level M4 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `7635bfec` | sm86 | fail | free-run | 0.9998 | 0.0399 | 0.0082 | 0.871 | 0.840 | 1.500 | 240109510.2 | 28.9 | `2282ff98ae9c` |
| `7635bfec` | sm86 | fail | free-run | 0.9998 | 0.0399 | 0.0082 | 0.871 | 0.840 | 1.500 | 240109510.2 | 28.9 | `d03b75f69b62` |

> decode-smoke review by ClaudeCode-MXF of receipt d03b75f69b6244e9: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.M2.L02` -- FAIL

attn_q / dense_prior_anchored_ridge / level M2 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `af059555` | sm86 | fail | composition | 0.9981 | 0.1369 | 0.0257 | 0.609 | 0.600 | 1.000 | 240109510.2 | 28.2 | `ebe064ab0e2b` |
| `af059555` | sm86 | fail | composition | 0.9981 | 0.1369 | 0.0257 | 0.609 | 0.600 | 1.000 | 240109510.2 | 28.2 | `1e1579c339be` |

> decode-smoke review by ClaudeCode-MXF of receipt 1e1579c339be7a89: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T7.M16.L02` -- PASS

attn_q / attention_relation_matching / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b2df6340` | -- | pass | -- | 0.9988 | 0.1157 | 0.0112 | 0.920 | 0.914 | -- | 1024.0 | 28.4 | `a031bc54932a` |
| `b2df6340` | -- | pending-review | -- | 0.9988 | 0.1157 | 0.0112 | 0.920 | 0.914 | -- | 1024.0 | 28.4 | `e65bc747a820` |

> decode-smoke review by claude-opus-5 of receipt e65bc747a820d224: 3/3 on task. (1) circuit breaker: correct mechanism plus a correct closing about domestic/commercial/industrial use, terminated on EOS. (2) one-pan dinner: same dish as the control with a slightly shorter ingredient list, and it reaches the 'Instructions:' header the control never got to before the cap -- censored, not failed. (3) neighbour note: byte-identical to the control. The score-kernel contraction to head_dim 60 of 64 is not visible in the text. PASS under section 5 rule 1: free-run 0.920 >= 0.90, smoke 3/3 coherent, emit-zero floor passed.

### `C06.T7.M16.L08` -- FAIL

attn_q / attention_relation_matching / level M16 / site L08 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4bbc4947` | -- | fail | free-run | 0.9985 | 0.1517 | 0.0094 | 0.846 | 0.838 | -- | 1024.0 | 51.6 | `736fbb488e18` |
| `4bbc4947` | -- | fail | free-run | 0.9985 | 0.1517 | 0.0094 | 0.846 | 0.838 | -- | 1024.0 | 51.6 | `b49cf1897196` |

> decode-smoke review by claude-opus-5 of receipt b49cf18971964e85: 3/3 on task and materially indistinguishable from the L02 T7 cell's text: (1) circuit breaker cap-censored mid-clause, correct throughout; (2) one-pan dinner reaching 'Instructions: 1. Prepare the pan' before the cap; (3) neighbour note the control's, with '[Neighbour's Name]'. The cell still fails at chain link 6 -- free-run 0.846 against the 0.90 bar -- which is the point: the same technique at the same level one site deeper keeps composition (drift 0.0094) and loses 15% of the control's tokens, and the smoke cannot see the difference.

### `C06.T7.M16.L12` -- FAIL

attn_q / attention_relation_matching / level M16 / site L12 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1c571833` | sm86 | fail | free-run | 0.9991 | 0.0474 | 0.0169 | 0.680 | 0.672 | 0.996 | 1024.0 | 32.1 | `3fbc8b93ab9c` |
| `1c571833` | sm86 | fail | free-run | 0.9991 | 0.0474 | 0.0169 | 0.680 | 0.672 | 0.996 | 1024.0 | 32.1 | `6943f8cda827` |

> decode-smoke review by ClaudeCode-MXF of receipt 6943f8cda8276649: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T7.M8.L02` -- PASS

attn_q / attention_relation_matching / level M8 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6387facc` | sm86 | pass | -- | 0.9968 | 0.1899 | 0.0190 | 0.916 | 0.902 | 0.930 | 1024.0 | 29.7 | `328ef35c7743` |
| `6387facc` | sm86 | pending-review | -- | 0.9968 | 0.1899 | 0.0190 | 0.916 | 0.902 | 0.930 | 1024.0 | 29.7 | `fc3b33582e1d` |

> decode-smoke review by ClaudeCode-MXF of receipt fc3b33582e1de716: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T7.M4.L02` -- FAIL

attn_q / attention_relation_matching / level M4 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4f5706af` | sm86 | fail | composition | 0.9903 | 0.3281 | 0.0427 | 0.547 | 0.537 | 0.797 | 1024.0 | 31.6 | `c13a6733805c` |
| `4f5706af` | sm86 | fail | composition | 0.9903 | 0.3281 | 0.0427 | 0.547 | 0.537 | 0.797 | 1024.0 | 31.6 | `08ad05a1dd2c` |

> decode-smoke review by ClaudeCode-MXF of receipt 08ad05a1dd2c088c: coherent and formatted; the trailer note drifts to 'move a parked trailer to a nearby park'.  Semantic drift, not degeneracy.

### `C07.T1.M16.L02` -- PASS

attn_k / dense_prior_anchored_ridge / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6bd399a1` | -- | pass | -- | 0.9999 | 0.0000 | 0.0074 | 0.941 | 0.932 | -- | 240109510.2 | 50.3 | `cb74365aab67` |
| `6bd399a1` | -- | pending-review | -- | 0.9999 | 0.0000 | 0.0074 | 0.941 | 0.932 | -- | 240109510.2 | 50.3 | `1b7e75618682` |

> decode-smoke review by claude-opus-5 of receipt 1b7e75618682c476: 3/3 on task. (1) circuit breaker: the most precise of the four lanes read today -- 'automatically shutting off power to circuits that are experiencing too much current' names current rather than the vaguer 'overload', terminated on EOS. (2) one-pan dinner: byte-identical to the copy_exact control, cap-censored at the same point. (3) neighbour note: the control's text with '[Neighbour's Name]' for '[Neighbour]', carrying the control's own 'for your convenience' slip. PASS under section 5 rule 1: free-run 0.941 >= 0.90, smoke 3/3 coherent, emit-zero floor passed.

### `C07.T1.M16.L08` -- PASS

attn_k / dense_prior_anchored_ridge / level M16 / site L08 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5c224cb3` | sm86 | pass | -- | 1.0000 | 0.0000 | 0.0095 | 0.936 | 0.936 | 1.406 | 31993.0 | 54.1 | `9913ebebc7c6` |
| `5c224cb3` | sm86 | pending-review | -- | 1.0000 | 0.0000 | 0.0095 | 0.936 | 0.936 | 1.406 | 31993.0 | 54.1 | `16b7771ad0ba` |

> decode-smoke review by ClaudeCode-MXF of receipt 16b7771ad0bac334: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.M16.L12` -- PASS

attn_k / dense_prior_anchored_ridge / level M16 / site L12 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1510e0a9` | sm86 | pass | -- | 1.0000 | 0.0000 | 0.0054 | 0.994 | 0.986 | 1.406 | 27233.8 | 51.8 | `46b6134c335f` |
| `1510e0a9` | sm86 | pending-review | -- | 1.0000 | 0.0000 | 0.0054 | 0.994 | 0.986 | 1.406 | 27233.8 | 51.8 | `603b731a8d97` |

> decode-smoke review by ClaudeCode-MXF of receipt 603b731a8d97c049: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.M8.L02` -- FAIL

attn_k / dense_prior_anchored_ridge / level M8 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fc7a1cb8` | sm86 | fail | free-run | 0.9997 | 0.0000 | 0.0127 | 0.793 | 0.770 | 1.312 | 240109510.2 | 32.2 | `747f77a26b42` |
| `fc7a1cb8` | sm86 | fail | free-run | 0.9997 | 0.0000 | 0.0127 | 0.793 | 0.770 | 1.312 | 240109510.2 | 32.2 | `6e98df1706de` |

> decode-smoke review by ClaudeCode-MXF of receipt 6e98df1706dee86a: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.M4.L02` -- FAIL

attn_k / dense_prior_anchored_ridge / level M4 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `913ac423` | sm86 | fail | composition | 0.9989 | 0.0000 | 0.0236 | 0.631 | 0.588 | 1.125 | 240109510.2 | 28.1 | `5c6d5877f7ad` |
| `913ac423` | sm86 | fail | composition | 0.9989 | 0.0000 | 0.0236 | 0.631 | 0.588 | 1.125 | 240109510.2 | 28.1 | `b4f7d8598475` |

> decode-smoke review by ClaudeCode-MXF of receipt b4f7d859847556ad: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.M2.L02` -- FAIL

attn_k / dense_prior_anchored_ridge / level M2 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `48d4bb3a` | sm86 | fail | composition | 0.9944 | 0.0000 | 0.0485 | 0.529 | 0.457 | 0.750 | 240109510.2 | 43.9 | `4870a75b38d7` |
| `48d4bb3a` | sm86 | fail | composition | 0.9944 | 0.0000 | 0.0485 | 0.529 | 0.457 | 0.750 | 240109510.2 | 43.9 | `ae0716981fcd` |

> decode-smoke review by ClaudeCode-MXF of receipt ae0716981fcd4c72: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C08.T1.M16.L02` -- FAIL

attn_v / dense_prior_anchored_ridge / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ff717e09` | -- | fail | free-run | 0.9992 | 0.0000 | 0.0154 | 0.738 | 0.736 | -- | 240109510.2 | 25.9 | `c9c322a49d54` |
| `ff717e09` | -- | fail | free-run | 0.9992 | 0.0000 | 0.0154 | 0.738 | 0.736 | -- | 240109510.2 | 25.9 | `a99365bed294` |

> decode-smoke review by claude-opus-5 of receipt a99365bed294fd8c: 3/3 on task. (1) circuit breaker: correct mechanism and a correct closing about where breakers are used, EOS. (2) one-pan dinner: the control's dish, cap-censored in the ingredient list. (3) neighbour note: the control's text. Fails at chain link 6, free-run 0.738 -- composition held at 0.0154 and the behaviour did not.

### `C08.T1.M8.L02` -- FAIL

attn_v / dense_prior_anchored_ridge / level M8 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1fc527cf` | sm86 | fail | composition | 0.9979 | 0.0000 | 0.0244 | 0.650 | 0.637 | 1.312 | 240109510.2 | 26.5 | `e3916c6bd217` |
| `1fc527cf` | sm86 | fail | composition | 0.9979 | 0.0000 | 0.0244 | 0.650 | 0.637 | 1.312 | 240109510.2 | 26.5 | `95093816400c` |

> decode-smoke review by ClaudeCode-MXF of receipt 95093816400cf876: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C08.T1.M4.L02` -- FAIL

attn_v / dense_prior_anchored_ridge / level M4 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baca1a66` | sm86 | fail | composition | 0.9933 | 0.0000 | 0.0388 | 0.553 | 0.516 | 1.125 | 240109510.2 | 27.0 | `3d14ab2abbfe` |
| `baca1a66` | sm86 | fail | composition | 0.9933 | 0.0000 | 0.0388 | 0.553 | 0.516 | 1.125 | 240109510.2 | 27.0 | `d610481b74f0` |

> decode-smoke review by ClaudeCode-MXF of receipt d610481b74f0cc64: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C08.T5.M16.L02` -- FAIL

attn_v / sparse_omp_routes / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ceb1bff9` | -- | fail | composition | 0.9908 | 0.0000 | 0.0381 | 0.578 | 0.541 | -- | 17.3 | 50.6 | `b14d1caf3aeb` |
| `ceb1bff9` | -- | fail | composition | 0.9908 | 0.0000 | 0.0381 | 0.578 | 0.541 | -- | 17.3 | 50.6 | `0bcbe2436c46` |

> decode-smoke review by claude-opus-5 of receipt 0bcbe2436c46e514: 3/3 on task. (1) circuit breaker: correct, names short circuit and wiring fault explicitly, EOS. (2) one-pan dinner: on task, cap-censored, but it preheats an oven for a one-pan 40-minute plan -- the same constraint miss the control makes. (3) neighbour note: byte-identical to the copy_exact control. Fails at chain link 5, composed drift 0.0381.

### `C09.T1.M16.L02` -- FAIL

attn_o / dense_prior_anchored_ridge / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e5835171` | -- | fail | free-run | 0.9999 | 0.0072 | 0.0132 | 0.820 | 0.797 | -- | 963110.6 | 28.1 | `c61491e7d275` |
| `e5835171` | -- | fail | free-run | 0.9999 | 0.0072 | 0.0132 | 0.820 | 0.797 | -- | 963110.6 | 28.1 | `7866bddd609e` |

> decode-smoke review by claude-opus-5 of receipt 7866bddd609ed0cf: 3/3 on task. (1) circuit breaker: correct, EOS, with one clumsy tautology ('damage to the equipment, such as equipment, wiring, or other components'). (2) one-pan dinner: the control's dish, cap-censored inside 'Preheat oven to'. (3) neighbour note: polite and correctly shaped, offers to assist. The cell cleared composition (0.0132 against the 0.02 gate) and fails at chain link 6, free-run 0.820 -- the F3 signature, every numeric link passed and behaviour lost anyway.

### `C09.T1.M16.L08` -- FAIL

attn_o / dense_prior_anchored_ridge / level M16 / site L08 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f4be4c96` | sm86 | fail | composition | 0.9975 | 0.0305 | 0.0494 | 0.584 | 0.576 | 1.875 | 41374.2 | 54.0 | `2eaf8e77fb81` |
| `f4be4c96` | sm86 | fail | composition | 0.9975 | 0.0305 | 0.0494 | 0.584 | 0.576 | 1.875 | 41374.2 | 54.0 | `19bcea5bf644` |

> decode-smoke review by ClaudeCode-MXF of receipt 19bcea5bf6449a14: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T1.M8.L02` -- FAIL

attn_o / dense_prior_anchored_ridge / level M8 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6d299ae5` | sm86 | fail | composition | 0.9996 | 0.0155 | 0.0252 | 0.736 | 0.619 | 1.750 | 121606.9 | 27.2 | `2a5c9d574580` |
| `6d299ae5` | sm86 | fail | composition | 0.9996 | 0.0155 | 0.0252 | 0.736 | 0.619 | 1.750 | 121606.9 | 27.2 | `424cc07698d6` |

> decode-smoke review by ClaudeCode-MXF of receipt 424cc07698d6b752: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T1.M4.L02` -- FAIL

attn_o / dense_prior_anchored_ridge / level M4 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `478df310` | sm86 | fail | composition | 0.9977 | 0.0374 | 0.0573 | 0.445 | 0.336 | 1.500 | 17528.4 | 32.9 | `3e0e33a1d0a9` |
| `478df310` | sm86 | fail | composition | 0.9977 | 0.0374 | 0.0573 | 0.445 | 0.336 | 1.500 | 17528.4 | 32.9 | `502c8ae387eb` |

> decode-smoke review by ClaudeCode-MXF of receipt 502c8ae387ebf6de: coherent prose in the right four-line-note format, but the trailer request inverts: the letter says the writer moved a trailer and wants to relocate it.  Semantic drift, not degeneracy -- no repetition loop, terminated eos.

### `C09.T2.M16.L02` -- FAIL

attn_o / cross_width_residual_charts / level M16 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `11bbc563` | sm86 | fail | free-run | 0.9999 | 0.0072 | 0.0131 | 0.820 | 0.797 | 1.875 | 1.0 | 29.8 | `746ae111c0c9` |
| `11bbc563` | sm86 | fail | free-run | 0.9999 | 0.0072 | 0.0131 | 0.820 | 0.797 | 1.875 | 1.0 | 29.8 | `b3b3d5cc831c` |

> decode-smoke review by ClaudeCode-MXF of receipt b3b3d5cc831c746c: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T4.M16.L02` -- FAIL

attn_o / low_rank_factor_writers / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `047f32a6` | -- | fail | local-solve | 0.9000 | 0.2395 | 0.3180 | 0.049 | 0.029 | -- | 2119973.2 | 25.2 | `278d582e79b7` |
| `047f32a6` | -- | fail | local-solve | 0.9000 | 0.2395 | 0.3180 | 0.049 | 0.029 | -- | 2119973.2 | 25.2 | `55971e9be790` |

> decode-smoke review by claude-opus-5 of receipt 55971e9be790d02e: 1/3 on task; the only incoherent body cell in wave 1 and the only genuine behavioural collapse the campaign has produced. (1) circuit breaker: fluent but answers about the wrong object -- 'a safety net that stops a system or process from being used until it can handle a sudden surge in demand' is the software circuit-breaker/rate-limiter pattern, not an electrical breaker, and the paragraph then says the same thing three times. (2) one-pan dinner: DEGENERATE -- 'a balanced mix of protein, carbs, and carbs, and carbs,' repeated roughly forty times until the cap; the runner's termination classifier correctly labelled it 'loop' rather than 'cap', so this is a failure and not a censoring. (3) neighbour note: off task -- it narrates what the note would be and then writes one that reverses the roles ('I'm moving a parked trailer... I'm happy to help with any repairs'). The cell already fails at chain link 3 (write cosine 0.900 < 0.95) at writer_rank 31; the smoke shows what a refused local solve sounds like downstream.

### `C09.T5.M16.L02` -- FAIL

attn_o / sparse_omp_routes / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a19a2978` | -- | fail | composition | 0.9995 | 0.0164 | 0.0200 | 0.709 | 0.670 | -- | 142.6 | 54.6 | `dbc77de782b2` |
| `a19a2978` | -- | fail | composition | 0.9995 | 0.0164 | 0.0200 | 0.709 | 0.670 | -- | 142.6 | 54.6 | `cc3b204c9498` |

> decode-smoke review by claude-opus-5 of receipt cc3b204c949877a2: 3/3 on task. (1) circuit breaker: correct mechanism with a clean summary sentence, EOS. (2) one-pan dinner: the control's dish, cap-censored inside 'Instructions: 1. Pre'. (3) neighbour note: byte-identical to the copy_exact control. The cell fails at chain link 5, composed drift 0.0200 -- exactly on the 0.02 gate to four decimals and over it in full precision.

### `C09.T8.M16.L02` -- FAIL

attn_o / gram_relation_matching / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b2b652a3` | -- | fail | composition | 0.9995 | 0.0155 | 0.0210 | 0.678 | 0.656 | -- | 6277438.1 | 28.4 | `b62c6bebcc3b` |
| `b2b652a3` | -- | fail | composition | 0.9995 | 0.0155 | 0.0210 | 0.678 | 0.656 | -- | 6277438.1 | 28.4 | `448022d7bb90` |

> decode-smoke review by claude-opus-5 of receipt 448022d7bb90aef6: 3/3 on task. (1) circuit breaker: correct, adds a true statement about overload surges damaging motors and lights, cap-censored inside 'connected to a switch or relay, which is'. (2) one-pan dinner: the control's dish, cap-censored. (3) neighbour note: polite, correctly shaped, EOS. Fails at chain link 5 (composed drift 0.0210) with a boundary indistinguishable from T5's at the same site.

### `C10.T1.M16.L02` -- INFR

attn_per_head / dense_prior_anchored_ridge / level M16 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d3aed252` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.4 | `58cf73bb6cc1` |
| `22f3a3d5` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 34.8 | `fb6e3f2a8652` |

> CellTechniqueError: T1's reference kernel contracts linear maps; C10 has axis 'heads' and needs its own kernel.

### `C11.T5.M16.L00` -- FAIL

mlp_gate / sparse_omp_routes / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fbb942cb` | -- | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | -- | 61.3 | 56.6 | `ae83571d4f2e` |
| `fbb942cb` | -- | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | -- | 61.3 | 56.6 | `dd5be1b352fe` |

> decode-smoke review by claude-opus-5 of receipt dd5be1b352fe7cf9: 3/3 on task with the worst single factual error in wave 1. (1) circuit breaker: opens 'A circuit breaker is a safety device in a circuit breaker' -- self-referential -- and then states the mechanism INVERTED: 'monitoring the voltage in the circuit and shutting off power if the voltage drops below a certain threshold'. A breaker trips on overcurrent, not undervoltage. Fluent, on topic, mechanically wrong. (2) one-pan dinner: a real plan, cap-censored, oven-preheat again. (3) neighbour note: correctly shaped, asks for a preferred time. Fails at chain link 5, composed drift 0.0984 -- five times the gate, the second worst composed drift of any cell that reached the behaviour stage.

### `C12.T5.M16.L00` -- FAIL

mlp_up / sparse_omp_routes / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `325b8ed3` | -- | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | -- | 6.2 | 53.7 | `6800a1b9aaad` |
| `325b8ed3` | -- | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | -- | 6.2 | 53.7 | `7847e13e65d1` |

> decode-smoke review by claude-opus-5 of receipt 7847e13e65d103f9: 3/3 on task. (1) circuit breaker: short and correct, though it conflates the short-circuit trip with the overload trip. (2) one-pan dinner: 'Grilled Chicken with Vegetables & Breaded Eggs' with a time header, cap-censored -- on task and the only cell to state the 40-minute budget in the plan itself. (3) neighbour note: correct shape, then appends an assistant-voice trailer, 'Would you like me to also make a more formal or casual version of this note?', which the prompt did not ask for and no other cell produced. Fails at chain link 5 with composed drift 0.1945 -- the largest of any cell that reached the behaviour stage, ten times the gate.

### `C13.T1.M16.L00` -- FAIL

mlp_down / dense_prior_anchored_ridge / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `ad105cf52756` |
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `22b27a841a92` |

> decode-smoke review by claude-opus-5 of receipt 22b27a841a92620f: 3/3 on task, all EOS. (1) circuit breaker: correct, names short circuit and wiring fault, and correctly observes that other circuits keep operating. (2) one-pan dinner: a stir-fry with soy and honey, cap-censored at 'Instructions: 1'. (3) neighbour note: the control's text minus the closing thanks. Fails at chain link 5, composed drift 0.0668.

### `C13.T1.M16.L05` -- FAIL

mlp_down / dense_prior_anchored_ridge / level M16 / site L05 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cc2739a6` | -- | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.383 | 0.357 | -- | 2417440.9 | 30.8 | `6858f56371cb` |
| `cc2739a6` | -- | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.383 | 0.357 | -- | 2417440.9 | 30.8 | `9c6849ae858d` |

> decode-smoke review by claude-opus-5 of receipt 9c6849ae858d0c82: 3/3 on task, both long outputs cap-censored. (1) circuit breaker: correct, censored mid-sentence at 'help maintain the'. (2) one-pan dinner: an ingredient list carrying brown rice, quinoa AND broccoli plus '1 cup olive oil', censored before any instruction -- on task, quantity implausible. (3) neighbour note: correctly shaped. Fails at chain link 5, composed drift 0.0629, free-run 0.383.

### `C13.T1.M16.L13` -- FAIL

mlp_down / dense_prior_anchored_ridge / level M16 / site L13 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `527e062d` | sm86 | fail | composition | 0.9975 | 0.0266 | 0.0472 | 0.428 | 0.393 | 3.281 | 154692.4 | 30.3 | `5968579b7b34` |
| `527e062d` | sm86 | fail | composition | 0.9975 | 0.0266 | 0.0472 | 0.428 | 0.393 | 3.281 | 154692.4 | 30.3 | `9f3436610571` |

> decode-smoke review by ClaudeCode-MXF of receipt 9f343661057148e5: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T2.M16.L05` -- FAIL

mlp_down / cross_width_residual_charts / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `00a79e47` | sm86 | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.324 | 0.299 | 3.281 | 1.0 | 46.4 | `79b36586a4e7` |
| `00a79e47` | sm86 | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.324 | 0.299 | 3.281 | 1.0 | 46.4 | `55e5f7b0d099` |

> decode-smoke review by ClaudeCode-MXF of receipt 55e5f7b0d099c245: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T3.M16.L05` -- FAIL

mlp_down / balanced_basis_truncation / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6da9e0bc` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 213.4 | `e698cff22908` |
| `6da9e0bc` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 213.4 | `8ce72a8675cd` |

> decode-smoke review by ClaudeCode-MXF of receipt 8ce72a8675cd97c1: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T5.M16.L00` -- FAIL

mlp_down / sparse_omp_routes / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `7ad863c3` | -- | fail | composition | 0.9918 | 0.1097 | 0.0894 | 0.332 | 0.270 | -- | 30.0 | 58.3 | `0847c4ee885e` |
| `7ad863c3` | -- | fail | composition | 0.9918 | 0.1097 | 0.0894 | 0.332 | 0.270 | -- | 30.0 | 58.3 | `e24a66a70924` |

> decode-smoke review by claude-opus-5 of receipt e24a66a70924490d: 3/3 on task, 2 of 3 on EOS. (1) circuit breaker: correct and compact. (2) one-pan dinner: a quinoa-salad plan, cap-censored in the ingredient list. (3) neighbour note: a subject-lined note that correctly asks the neighbour to move the trailer and asks for a time -- better on the instruction than the control, which asks 'for your convenience'. Fails at chain link 5, composed drift 0.0894; free-run 0.332 is among the lowest in wave 1 and the smoke does not show it.

### `C13.T5.M16.L05` -- FAIL

mlp_down / sparse_omp_routes / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f6ec506d` | sm86 | fail | composition | 0.9935 | 0.0403 | 0.0575 | 0.430 | 0.387 | 0.938 | 149662.3 | 31.0 | `87cf45b06fc6` |
| `f6ec506d` | sm86 | fail | composition | 0.9935 | 0.0403 | 0.0575 | 0.430 | 0.387 | 0.938 | 149662.3 | 31.0 | `2884eda0a3a8` |
| `93af26a9` | -- | fail | composition | 0.9934 | 0.0407 | 0.0559 | 0.502 | 0.414 | -- | 149662.3 | 74.8 | `d535677abfb0` |
| `93af26a9` | -- | fail | composition | 0.9934 | 0.0407 | 0.0559 | 0.502 | 0.414 | -- | 149662.3 | 74.8 | `5796802efaf9` |

> decode-smoke review by ClaudeCode-MXF of receipt 2884eda0a3a84e08: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T5.M16.L13` -- FAIL

mlp_down / sparse_omp_routes / level M16 / site L13 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `081d2e34` | -- | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | -- | 402.0 | 53.5 | `045eedae39b4` |
| `081d2e34` | -- | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | -- | 402.0 | 53.5 | `386c905b5c0d` |

> decode-smoke review by claude-opus-5 of receipt 386c905b5c0d98e6: 3/3 on task. (1) circuit breaker: correct, though 'when it's too much' is an ungrammatical elision. (2) one-pan dinner: a real plan with '1 cup olive oil' again implausible, cap-censored at 'Instructions: 1.'. (3) neighbour note: correctly asks the neighbour to move the trailer and calls it a small change -- on instruction, no 'for your convenience' inversion. Fails at chain link 5, composed drift 0.0482; the best free-run (0.561) of the three C13 T5 sites.

### `C13.T8.M16.L00` -- FAIL

mlp_down / gram_relation_matching / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b16d2bfb` | -- | fail | composition | 0.9813 | 0.1266 | 0.1791 | 0.199 | 0.123 | -- | 302639.9 | 30.7 | `e7396d2c02c9` |
| `b16d2bfb` | -- | fail | composition | 0.9813 | 0.1266 | 0.1791 | 0.199 | 0.123 | -- | 302639.9 | 30.7 | `06dac323e512` |

> decode-smoke review by claude-opus-5 of receipt 06dac323e512bb99: 3/3 on task. (1) circuit breaker: correct, if repetitive. (2) one-pan dinner: opens 'Grilled Salmon with Lemon-Herb Butter' with '1 cup fresh lemon juice' -- an implausible quantity -- then a second plan with an oven preheat, cap-censored. (3) neighbour note: correct shape, asks for a preferred location and time. Fails at chain link 5 with composed drift 0.1791, and free-run 0.199 is the LOWEST of any cell in wave 1 that reached the behaviour stage -- the smoke is fluent and the token agreement with the control is almost gone, which is exactly the divergence section 5 rule 1 makes the review necessary but not sufficient for.

### `C14.T1.NA.L00` -- PASS

norms / dense_prior_anchored_ridge / level NA / site L00 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `662ecd09` | sm86 | pass | -- | 1.0000 | 0.0342 | 0.0001 | 1.000 | 1.000 | 0.500 | 1.0 | 26.3 | `b274434ceb89` |
| `662ecd09` | sm86 | pending-review | -- | 1.0000 | 0.0342 | 0.0001 | 1.000 | 1.000 | 0.500 | 1.0 | 26.3 | `00ce3f5c047c` |
| `7f7d50a8` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 35.1 | `66334299014d` |

> decode-smoke review by ClaudeCode-MXF of receipt 00ce3f5c047c70f8: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | -- | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | -- | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

