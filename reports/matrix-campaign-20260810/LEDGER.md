# Stage-1 matrix campaign -- ledger

Rendered 2026-08-10T17:46:16+00:00 from `C:\Users\cereal\Documents\ai training\neural-forge\reports\matrix-campaign-20260810\ledger.jsonl`.

**5 cells sealed** (14 ledger rows including re-runs); newest seal 2026-08-10T17:42:35.001+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 1 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `fail` | 4 | crossed a chain threshold; see first_divergent_metric |

## 1. The matrix

Rows are components, columns are techniques.  A position shows the deepest level whose cells all passed, or the shallowest level attempted and the chain link it broke at.  `.` is untested.

| component | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|
| C01 token_interface_in | . | . | . | . | X4:free | . | . | . | . |
| C05 conv_out_proj | . | . | . | . | M16:comp | . | . | . | . |

### Calibration row

| cell | status | free-run vs control | notes |
|---|---|---:|---|
| `CTRL.copy_exact` | PASS | 1.0000 | decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verba... |
| `CAL.C01.T5.X4` | FAIL | 0.1250 | decode-smoke review by claude-opus-5 (frozen V5 protocol e5e37768) of receipt 4018355521b93a30: Frozen-V5 q... |

## 2. Level ladders

§5 rule 2: a technique+component works *at level L* only if it passes at L and every shallower level, at >= 2 of its 3 sites.  A pass deeper than a fail is flagged `non-monotone` and does not count.

| component | technique | M16 | M8 | M4 | X4 | verdict |
|---|---|---|---|---|---|---|
| C01 | T5 | . | . | . | 0/1 | fails at free-run |
| C05 | T5 | 0/3 | . | . | . | fails at composition |

## 3. Failure classes

The chain is walked mechanically (§5 rule 5); this histogram is the campaign's failure analysis in full.  `free-run` is the F3 signature -- a cell that cleared every numeric link and still lost behavior -- and is the most information-dense class in the matrix.

| first divergent metric | cells | share |
|---|---:|---:|
| `composition` | 3 | 75% |
| `free-run` | 1 | 25% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 0 | 3 |
| out of band | 0 | 0 |

Agreement 0/3 (0%).

## 5. Per-cell drill-down

### `CAL.C01.T5.X4` -- FAIL

token_interface_in / sparse_omp_routes / level X4 / site GLOBAL / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `5b8e6ee3` | fail | free-run | -- | -- | -- | 0.125 | 0.327 | 1.0 | 76.9 | `28564b7156f8` |
| `5b8e6ee3` | fail | free-run | -- | -- | -- | 0.125 | 0.327 | 1.0 | 76.9 | `4018355521b9` |
| `0305653c` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 1321.3 | `276fd2e49dd4` |
| `c9fcf7bd` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 26.7 | `c079146235cd` |
| `d0e0fdaf` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 5.3 | `a9eef18a3639` |
| `3871e23b` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 5.2 | `9b0cc7b841a5` |

> decode-smoke review by claude-opus-5 (frozen V5 protocol e5e37768) of receipt 4018355521b93a30: Frozen-V5 quality-band review, reports/matrix-campaign-20260810/cells/CAL.C01.T5.X4-mxc/QUALITY_BAND_REVIEW.md. Whole-response band 8/15 useful-or-partial (3 useful / 5 partial / 7 observed miss), identical case-for-case to the sealed A9 child re-graded at the same 256 cap (8/15, 3/5/7) and to the frontier re-score's published A9 band (8/15) -- band delta 0, so the calibration criterion of quality-band agreement PASSES at any tolerance. The three receipt decode-smoke rows are arithmetic-multi-step, coding-deduplicate and comparison-ram-rom, all three Observed miss on both children, hence verdict incoherent; the receipt's status is fail on chain link 6 (free-run 0.125) with or without this verdict.

### `C05.T5.M16.L00` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L00 / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `5304bb14` | fail | composition | 0.9991 | 0.0367 | 0.0436 | 0.662 | 0.627 | 834.0 | 52.0 | `5d6653040be3` |
| `5304bb14` | fail | composition | 0.9991 | 0.0367 | 0.0436 | 0.662 | 0.627 | 834.0 | 52.0 | `853ac00d38a0` |

> decode-smoke review by claude-opus-5 of receipt 853ac00d38a0bfb7: 3/3 on task, all EOS. (1) circuit breaker: correct mechanism (overload, trips, stops the flow), mildly redundant. (2) dinner plan: a real 40-minute claim but the answer is a ten-dish menu with no cooking order and no single pan -- both stated constraints missed, task understood. (3) neighbour note: polite, asks for the move, not four lines and the offer to assist is filler. Coherent with instruction misses; the cell still fails at chain link 5 (composition 0.0436 > 0.02).

### `C05.T5.M16.L05` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L05 / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `20e3c653` | fail | composition | 0.9969 | 0.0226 | 0.0405 | 0.791 | 0.771 | 656.1 | 52.7 | `1c4a09cd549a` |
| `20e3c653` | fail | composition | 0.9969 | 0.0226 | 0.0405 | 0.791 | 0.771 | 656.1 | 52.7 | `5b250adb83f0` |

> decode-smoke review by claude-opus-5 of receipt 5b250adb83f0a814: 3/3 on task. (1) circuit breaker: correct mechanism plus fault detection and the overheating/fire consequence. (2) one-pan dinner: a genuine single-dish plan with ingredient list, censored at the 128-token cap inside 'Instructions: 1. Prepare' -- censored, not failed. (3) neighbour note: polite and correctly shaped; 'for your convenience' is the same wording slip the copy_exact control made. The cell still fails at chain link 5 (composition 0.0405 > 0.02).

### `C05.T5.M16.L13` -- FAIL

conv_out_proj / sparse_omp_routes / level M16 / site L13 / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dd23c9d2` | fail | composition | 0.9958 | 0.0166 | 0.0295 | 0.688 | 0.686 | 3359.7 | 53.2 | `be533d1bd1bd` |
| `dd23c9d2` | fail | composition | 0.9958 | 0.0166 | 0.0295 | 0.688 | 0.686 | 3359.7 | 53.2 | `12c4561c0f9b` |

> decode-smoke review by claude-opus-5 of receipt 12c4561c0f9b26b8: 3/3 on task. (1) circuit breaker: correct mechanism, longest and cleanest of the three cells. (2) one-pan dinner: structured plan with a prep step, censored at the cap; 'Cook the pan' is malformed and 180F is not medium-low. (3) neighbour note: polite, asks for the move, carries the same 'for your convenience' slip as L05. The cell still fails at chain link 5 (composition 0.0295 > 0.02).

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

