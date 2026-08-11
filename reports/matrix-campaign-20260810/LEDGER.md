# Stage-1 matrix campaign -- ledger

Rendered 2026-08-11T14:23:58+00:00 from `C:\Users\cereal\Documents\ai training\neural-forge\reports\matrix-campaign-20260810\ledger.jsonl`.

**110 cells sealed** (235 ledger rows including re-runs); newest seal 2026-08-11T14:20:43.864+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 18 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `fail` | 87 | crossed a chain threshold; see first_divergent_metric |
| `infra` | 5 | calibration refused the cell -- void, not evidence |

## 0. ADDENDUM 6 -- behaviour verdicts (the primary grade)

ADDENDUM 6 §6.1: the grade of a cell is the frozen 15-prompt review of the **whole model** with that cell's component folded and every other component copy_exact, scored against the parent's own 12/15 on the same surface in the same process.  `token_match_vs_control` is a diagnostic column here and gates nothing -- wave 4 sealed a child at 0.551 token match and 12/15, the parent's own band.

| verdict | cells | meaning |
|---|---:|---|
| `alive-parent-band` | 7 | within 2 cases of the parent's own 12/15 on the frozen surface, coherent, emit-zero floor passed |
| `alive-degraded` | 1 | >= the A9 band of 8/15, outside the parent's +/-2 |
| `dead` | 1 | incoherent, emit-zero floor failed, or degenerate output |

| cell | site | level | stored/parent | bound | verdict | review | parent | smoke | token-match (diag) |
|---|---|---|---:|---|---|---:|---:|---|---:|
| `C11.T1.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2500 |
| `C11.T5.M16.L00` | L00 | M16 | 0.9385 | -- | **alive-parent-band** | 12/15 | 12 | coherent | 0.4004 |
| `C11.T5.S4.L00` | L00 | S4 | 0.2494 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.2109 |
| `C12.T5.M16.L00` | L00 | M16 | 0.9385 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.3066 |
| `C13.T1.S4.L05` | L05 | S4 | 0.2488 | 0.250 ok | **dead** | 8/15 | 12 | coherent | 0.0410 |
| `C13.T3.M16.L05` | L05 | M16 | 3.2812 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.5859 |
| `C13.T3.M16.L13` | L13 | M16 | 3.2812 | -- | **alive-parent-band** | 12/15 | 12 | coherent | 0.5820 |
| `C13.T5.M16.L13` | L13 | M16 | 0.9384 | -- | **alive-parent-band** | 13/15 | 12 | coherent | 0.5605 |
| `C13.T5.S4.L05` | L05 | S4 | 0.2499 | 0.250 ok | **alive-parent-band** | 11/15 | 12 | coherent | 0.1152 |

**Survivors** (alive at a satisfied storage bound): 3 -- these are the assembly wave's input.

- `C11.T1.S4.L00` at L00: alive-parent-band, 10/15, stores 0.2488x
- `C11.T5.S4.L00` at L00: alive-degraded, 8/15, stores 0.2494x
- `C13.T5.S4.L05` at L05: alive-parent-band, 11/15, stores 0.2499x

## 1. The matrix

Rows are components, columns are techniques.  A position shows the deepest level with a passing site **and the sites it passed at** (ADDENDUM 5 rule 2-S), or the shallowest level attempted and the chain link it broke at.  `.` is untested.

| component | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|
| C01 token_interface_in | . | . | . | . | X4:free | . | . | . | . |
| C03 conv_in_proj | . | . | . | . | M16:comp | . | . | . | . |
| C04 conv_kernel | M16:comp | . | . | . | . | . | . | . | . |
| C05 conv_out_proj | M16:comp | M16:comp | M16:comp | M16:solve | M16:comp | . | . | M16:comp | M8:comp |
| C06 attn_q | PASS@X4@L12 | . | . | . | PASS@M16@L12 | . | PASS@M8@L02 | . | . |
| C07 attn_k | PASS@X4@L12 | . | . | . | . | . | . | . | . |
| C08 attn_v | M16:free | . | M16:infra | . | M16:comp | . | . | . | . |
| C09 attn_o | M16:free | M16:free | M16:free | M16:solve | M16:comp | . | . | M16:comp | . |
| C10 attn_per_head | M16:comp | . | . | . | . | . | . | . | . |
| C11 mlp_gate | S4:solve | S4:solve | S4:infra | S4:infra | M16:comp | . | . | S4:solve | . |
| C12 mlp_up | S4:solve | S4:solve | S4:infra | S4:infra | M16:comp | . | . | S4:solve | . |
| C13 mlp_down | M16:comp | M16:comp | M16:comp | S4:solve | M16:comp | . | . | M16:comp | . |
| C14 norms | PASS@NA@L00 | . | . | . | . | . | . | . | . |

### Calibration row

| cell | status | free-run vs control | notes |
|---|---|---:|---|
| `CTRL.copy_exact` | PASS | 1.0000 | decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verba... |
| `CAL.C01.T5.X4` | FAIL | 0.1250 | decode-smoke review by claude-opus-5 (frozen V5 protocol e5e37768) of receipt 4018355521b93a30: Frozen-V5 q... |

## 2. Level ladders

ADDENDUM 5 rule 2-S: a technique+component works *at level L at sites S*, where S is exactly the set of sites with a sealed passing receipt at L **and at every shallower level attempted at that site**.  The claim is never printed without its site set; the `good/attempted` count survives as a breadth annotation, never as the licence for an unqualified claim.  (This supersedes §5 rule 2, whose site-blind reading `NOTES-MX-G` §4.6 showed to be ambiguous.)

| component | technique | M16 | M8 | M4 | M2 | X4 | S2 | S4 | S8 | verdict (site-scoped) |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | T5 | . | . | . | . | 0/1 | . | . | . | fails at free-run |
| C03 | T5 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C04 | T1 | 0/1 | . | . | 0/1 | . | . | . | . | fails at composition |
| C05 | T1 | 0/3 | 0/1 | . | 0/1 | 0/1 | . | . | . | fails at composition |
| C05 | T2 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C05 | T3 | 0/3 | 0/1 | 0/1 | 0/1 | 0/1 | . | . | . | fails at composition |
| C05 | T4 | 0/1 | . | . | . | . | . | . | . | fails at local-solve |
| C05 | T5 | 0/3 | 0/1 | . | . | . | . | . | . | fails at composition |
| C05 | T8 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C05 | T9 | . | 0/1 | 0/1 | . | . | . | . | . | fails at composition |
| C06 | T1 | 3/3 L02,L08,L12 | 3/3 L02,L08,L12 | 0/1 | 0/1 | 1/3 L12 | . | . | . | works to X4 at L12 (1/3 sites); shallower elsewhere: L02:M8; L08:M8 |
| C06 | T5 | 1/2 L12 | . | . | . | . | . | . | . | works to M16 at L12 (1/2 sites) |
| C06 | T7 | 1/3 L02 | 1/1 L02 | 0/1 | . | . | . | . | . | works to M8 at L02 (1/3 sites) |
| C07 | T1 | 3/3 L02,L08,L12 | 2/3 L08,L12 | 0/1 | 0/1 | 1/2 L12 | . | . | . | works to X4 at L12 (1/3 sites); shallower elsewhere: L02:M16; L08:M8 |
| C08 | T1 | 0/1 | 0/1 | 0/1 | . | . | . | . | . | fails at composition |
| C08 | T3 | 0/1 | . | . | . | . | . | . | . | void (infra) -- no evidence |
| C08 | T5 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C09 | T1 | 0/2 | 0/1 | 0/1 | . | . | . | . | . | fails at composition |
| C09 | T2 | 0/1 | . | . | . | . | . | . | . | fails at free-run |
| C09 | T3 | 0/1 | 0/1 | . | . | 0/1 | . | . | . | fails at composition |
| C09 | T4 | 0/1 | . | . | . | . | . | . | . | fails at local-solve |
| C09 | T5 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C09 | T8 | 0/1 | . | . | . | . | . | . | . | fails at composition |
| C10 | T1 | 0/1 | 0/1 | . | . | . | . | . | . | fails at composition |
| C11 | T1 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C11 | T2 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C11 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C11 | T4 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C11 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C11 | T8 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C12 | T1 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C12 | T2 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C12 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C12 | T4 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C12 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C12 | T8 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C13 | T1 | 0/3 | . | . | . | . | . | 0/1 | . | fails at composition |
| C13 | T2 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C13 | T3 | 0/2 | 0/1 | 0/1 | . | . | . | 0/1 | . | fails at composition |
| C13 | T4 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C13 | T5 | 0/3 | . | . | . | . | . | 0/1 | . | fails at composition |
| C13 | T8 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C14 | T1 | . | . | . | . | . | . | . | . | works at NA (no level axis) at L00 (1/1 sites) |
| COMP | composition | . | . | . | . | . | . | . | . | fails at composition |

## 3. Failure classes

The chain is walked mechanically (§5 rule 5); this histogram is the campaign's failure analysis in full.  `free-run` is the F3 signature -- a cell that cleared every numeric link and still lost behavior -- and is the most information-dense class in the matrix.

| first divergent metric | cells | share |
|---|---:|---:|
| `infra` | 5 | 5% |
| `local-solve` | 17 | 18% |
| `macrocell-band` | 1 | 1% |
| `composition` | 57 | 62% |
| `free-run` | 12 | 13% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 18 | 83 |
| out of band | 0 | 1 |

Agreement 19/102 (19%).

## 4b. Storage honesty

A level names a *rank*, and a rank is not a size.  A technique whose stored form is a `w x r` basis plus an `r x w` solved factor stores `r * (in + out)` against a dense `in * out`, so on a square component it does not contract at all until `r < w/2`.  Cells above 1.0 below cost more storage than the tensor they replaced.

| cell | technique | level | stored | dense parent | stored/dense |
|---|---|---|---:|---:|---:|
| `C03.T5.M16.L00` | T5 | M16 | 2949632 | 3145728 | 0.9377 |
| `C04.T1.M16.L00` | T1 | M16 | 2048 | 3072 | 0.6667 |
| `C04.T1.M2.L00` | T1 | M2 | 1024 | 3072 | 0.3333 |
| `C05.T1.M8.L05` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T1.M2.L05` | T1 | M2 | 1048576 | 1048576 | 1.0000 |
| `C05.T1.X4.L05` | T1 | X4 | 1441792 | 1048576 | 1.3750 |
| `C05.T2.M16.L05` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L00` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L05` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L13` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M8.L05` | T3 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T3.M4.L05` | T3 | M4 | 1572864 | 1048576 | 1.5000 |
| `C05.T3.M2.L05` | T3 | M2 | 1048576 | 1048576 | 1.0000 |
| `C05.T3.X4.L05` | T3 | X4 | 1441792 | 1048576 | 1.3750 |
| `C05.T5.M8.L05` | T5 | M8 | 917504 | 1048576 | 0.8750 |
| `C05.T9.M8.L05` | T9 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T9.M4.L05` | T9 | M4 | 1572864 | 1048576 | 1.5000 |
| `C06.T1.M16.L08` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C06.T1.M16.L12` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C06.T1.M8.L02` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C06.T1.M8.L08` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C06.T1.M8.L12` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C06.T1.M4.L02` | T1 | M4 | 1572864 | 1048576 | 1.5000 |
| `C06.T1.M2.L02` | T1 | M2 | 1048576 | 1048576 | 1.0000 |
| `C06.T1.X4.L02` | T1 | X4 | 1572864 | 1048576 | 1.5000 |
| `C06.T1.X4.L08` | T1 | X4 | 1572864 | 1048576 | 1.5000 |
| `C06.T1.X4.L12` | T1 | X4 | 1572864 | 1048576 | 1.5000 |
| `C06.T5.M16.L08` | T5 | M16 | 983552 | 1048576 | 0.9380 |
| `C06.T5.M16.L12` | T5 | M16 | 983552 | 1048576 | 0.9380 |
| `C06.T7.M16.L12` | T7 | M16 | 1044480 | 1048576 | 0.9961 |
| `C06.T7.M8.L02` | T7 | M8 | 974848 | 1048576 | 0.9297 |
| `C06.T7.M4.L02` | T7 | M4 | 835584 | 1048576 | 0.7969 |
| `C07.T1.M16.L08` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M16.L12` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M8.L08` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M8.L12` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C07.T1.M2.L02` | T1 | M2 | 393216 | 524288 | 0.7500 |
| `C07.T1.X4.L08` | T1 | X4 | 589824 | 524288 | 1.1250 |
| `C07.T1.X4.L12` | T1 | X4 | 589824 | 524288 | 1.1250 |
| `C08.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C08.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C09.T1.M16.L08` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T1.M8.L02` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C09.T1.M4.L02` | T1 | M4 | 1572864 | 1048576 | 1.5000 |
| `C09.T2.M16.L02` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T3.M16.L02` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T3.M8.L02` | T3 | M8 | 1835008 | 1048576 | 1.7500 |
| `C09.T3.X4.L02` | T3 | X4 | 1572864 | 1048576 | 1.5000 |
| `C10.T1.M16.L02` | T1 | M16 | 3014656 | 3145728 | 0.9583 |
| `C10.T1.M8.L02` | T1 | M8 | 2883584 | 3145728 | 0.9167 |
| `C11.T1.S4.L00` | T1 | S4 | 652288 | 2621440 | 0.2488 |
| `C11.T2.S4.L00` | T2 | S4 | 652288 | 2621440 | 0.2488 |
| `C11.T5.M16.L00` | T5 | M16 | 2460321 | 2621440 | 0.9385 |
| `C11.T5.S4.L00` | T5 | S4 | 653763 | 2621440 | 0.2494 |
| `C11.T8.S4.L00` | T8 | S4 | 652288 | 2621440 | 0.2488 |
| `C12.T1.S4.L00` | T1 | S4 | 652288 | 2621440 | 0.2488 |
| `C12.T2.S4.L00` | T2 | S4 | 652288 | 2621440 | 0.2488 |
| `C12.T5.M16.L00` | T5 | M16 | 2460321 | 2621440 | 0.9385 |
| `C12.T5.S4.L00` | T5 | S4 | 653763 | 2621440 | 0.2494 |
| `C12.T8.S4.L00` | T8 | S4 | 652288 | 2621440 | 0.2488 |
| `C13.T1.M16.L00` | T1 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T1.M16.L13` | T1 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T1.S4.L05` | T1 | S4 | 652288 | 2621440 | 0.2488 |
| `C13.T2.M16.L05` | T2 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T2.S4.L05` | T2 | S4 | 652288 | 2621440 | 0.2488 |
| `C13.T3.M16.L05` | T3 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T3.M16.L13` | T3 | M16 | 8601600 | 2621440 | 3.2812 |
| `C13.T3.M8.L05` | T3 | M8 | 8028160 | 2621440 | 3.0625 |
| `C13.T3.M4.L05` | T3 | M4 | 6881280 | 2621440 | 2.6250 |
| `C13.T3.S4.L05` | T3 | S4 | 652288 | 2621440 | 0.2488 |
| `C13.T4.S4.L05` | T4 | S4 | 652288 | 2621440 | 0.2488 |
| `C13.T5.M16.L05` | T5 | M16 | 2457600 | 2621440 | 0.9375 |
| `C13.T5.M16.L13` | T5 | M16 | 2460000 | 2621440 | 0.9384 |
| `C13.T5.S4.L05` | T5 | S4 | 654975 | 2621440 | 0.2499 |
| `C13.T8.S4.L05` | T8 | S4 | 654975 | 2621440 | 0.2499 |
| `C14.T1.NA.L00` | T1 | NA | 1024 | 2048 | 0.5000 |
| `COMP.MA.RBEH` | composition | NA | 231693056 | 229693184 | 1.0087 |
| `COMP.MA.RSTO` | composition | NA | 229235456 | 229693184 | 0.9980 |
| `COMP.MB.RSTO` | composition | NA | 229235456 | 229693184 | 0.9980 |
| `COMP.MB.RBEH` | composition | NA | 231693056 | 229693184 | 1.0087 |

33 of 82 sealed cells with a storage reading actually contract.

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

### `C04.T1.M16.L00` -- FAIL

conv_kernel / dense_prior_anchored_ridge / level M16 / site L00 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `44a438a7` | sm86 | fail | composition | 0.9969 | 0.0752 | 0.0798 | 0.398 | 0.291 | 0.667 | 95.1 | 29.2 | `015b5bf7c9b6` |
| `44a438a7` | sm86 | fail | composition | 0.9969 | 0.0752 | 0.0798 | 0.398 | 0.291 | 0.667 | 95.1 | 29.2 | `5c50fafd1148` |
| `2dadae7a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.3 | `a30591546f77` |
| `f5c7bf74` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 34.9 | `6001d2de2582` |

> decode-smoke review by ClaudeCode-MXG of receipt 5c50fafd1148c061: three whole responses reviewed verbatim, all on task and terminating cleanly.  Recorded semantic drift, not degeneracy: the neighbour note relocates the trailer 'to a nearby park', the same drift wave 2 recorded on C06.T7.M4.L02.  Two FIR taps of three keeps the format and the register.

### `C04.T1.M2.L00` -- FAIL

conv_kernel / dense_prior_anchored_ridge / level M2 / site L00 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1e78bb36` | sm86 | fail | macrocell-band | 0.9367 | 0.4432 | 0.4829 | 0.113 | 0.082 | 0.333 | 1.0 | 27.3 | `4821af5b606d` |
| `1e78bb36` | sm86 | fail | macrocell-band | 0.9367 | 0.4432 | 0.4829 | 0.113 | 0.082 | 0.333 | 1.0 | 27.3 | `9f3f6a18b4d3` |

> decode-smoke review by ClaudeCode-MXG of receipt 9f3f6a18b4d3e79f: the campaign's second incoherent verdict and its first degenerate-by-emptiness one: the third response is EMPTY -- the child emitted EOS with no content at all for the neighbour-note prompt.  The censoring rule does not apply (termination is eos, not cap), so this is a real failure and not a truncation.  The first response also opens with a stray 'narrative:' header and the second echoes the prompt and then answers a different question (breakfast/lunch/dinner instead of the one-pan order).  One FIR tap of three is past the component's edge, and the cell is the campaign's first macrocell-band failure independently.

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

### `C05.T1.X4.L05` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level X4 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b3ca12b0` | sm86 | fail | composition | 0.9740 | 0.0700 | 0.0872 | 0.314 | 0.266 | 1.375 | 2123.9 | 26.7 | `de6c28c9619d` |
| `b3ca12b0` | sm86 | fail | composition | 0.9740 | 0.0700 | 0.0872 | 0.314 | 0.266 | 1.375 | 2123.9 | 26.7 | `db9acd921fc5` |

> decode-smoke review by ClaudeCode-MXG of receipt db9acd921fc588e1: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T2.M16.L05` -- FAIL

conv_out_proj / cross_width_residual_charts / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `99979d513ff3` |
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `6e59fd005767` |

> decode-smoke review by ClaudeCode-MXF of receipt 6e59fd005767d38a: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M16.L00` -- FAIL

conv_out_proj / balanced_basis_truncation / level M16 / site L00 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0a15d264` | sm86 | fail | composition | 0.9990 | 0.0332 | 0.0269 | 0.699 | 0.676 | 1.875 | 45814.4 | 42.5 | `70e3f3d30b77` |
| `0a15d264` | sm86 | fail | composition | 0.9990 | 0.0332 | 0.0269 | 0.699 | 0.676 | 1.875 | 45814.4 | 42.5 | `38f89cee5917` |

> decode-smoke review by ClaudeCode-MXG of receipt 38f89cee591712ff: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M16.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dca9532d` | sm86 | fail | composition | 0.9975 | 0.0220 | 0.0268 | 0.914 | 0.900 | 1.875 | 10573.8 | 41.9 | `6331ec8de0c5` |
| `dca9532d` | sm86 | fail | composition | 0.9975 | 0.0220 | 0.0268 | 0.914 | 0.900 | 1.875 | 10573.8 | 41.9 | `2ae667eaaeec` |

> decode-smoke review by ClaudeCode-MXF of receipt 2ae667eaaeecaa37: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M16.L13` -- FAIL

conv_out_proj / balanced_basis_truncation / level M16 / site L13 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9cd795bc` | sm86 | fail | composition | 0.9977 | 0.0125 | 0.0246 | 0.646 | 0.609 | 1.875 | 24777.3 | 44.5 | `e3570749a5df` |
| `9cd795bc` | sm86 | fail | composition | 0.9977 | 0.0125 | 0.0246 | 0.646 | 0.609 | 1.875 | 24777.3 | 44.5 | `0e8104429ff4` |

> decode-smoke review by ClaudeCode-MXG of receipt 0e8104429ff42d19: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M8.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level M8 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0838e7bb` | sm86 | fail | composition | 0.9947 | 0.0321 | 0.0396 | 0.777 | 0.752 | 1.750 | 7120.4 | 38.0 | `d1ed3bac6d0e` |
| `0838e7bb` | sm86 | fail | composition | 0.9947 | 0.0321 | 0.0396 | 0.777 | 0.752 | 1.750 | 7120.4 | 38.0 | `a44ca75d72a4` |

> decode-smoke review by ClaudeCode-MXG of receipt a44ca75d72a4faf4: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M4.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level M4 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `89e571a5` | sm86 | fail | composition | 0.9876 | 0.0492 | 0.0557 | 0.506 | 0.482 | 1.500 | 4402.1 | 37.7 | `1f9511cdd072` |
| `89e571a5` | sm86 | fail | composition | 0.9876 | 0.0492 | 0.0557 | 0.506 | 0.482 | 1.500 | 4402.1 | 37.7 | `20ad5dd19128` |

> decode-smoke review by ClaudeCode-MXG of receipt 20ad5dd191283416: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.M2.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level M2 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `35f77090` | sm86 | fail | composition | 0.9630 | 0.0852 | 0.0970 | 0.277 | 0.229 | 1.000 | 1748.8 | 34.4 | `1b1c0fd048c5` |
| `35f77090` | sm86 | fail | composition | 0.9630 | 0.0852 | 0.0970 | 0.277 | 0.229 | 1.000 | 1748.8 | 34.4 | `8e594f2420f8` |

> decode-smoke review by ClaudeCode-MXG of receipt 8e594f2420f8531b: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T3.X4.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level X4 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `763453f3` | sm86 | fail | composition | 0.9831 | 0.0576 | 0.0656 | 0.330 | 0.307 | 1.375 | 3448.4 | 35.4 | `b1c895fef767` |
| `763453f3` | sm86 | fail | composition | 0.9831 | 0.0576 | 0.0656 | 0.330 | 0.307 | 1.375 | 3448.4 | 35.4 | `be619c5c27a0` |

> decode-smoke review by ClaudeCode-MXG of receipt be619c5c27a09b72: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C05.T9.M8.L05` -- FAIL

conv_out_proj / progressive_multi_pass / level M8 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0a8445c2` | sm86 | fail | composition | 0.9924 | 0.0376 | 0.0539 | 0.746 | 0.746 | 1.750 | 4650.8 | 33.8 | `08543a204a90` |
| `0a8445c2` | sm86 | fail | composition | 0.9924 | 0.0376 | 0.0539 | 0.746 | 0.746 | 1.750 | 4650.8 | 33.8 | `dc1170c65119` |

> decode-smoke review by ClaudeCode-MXG of receipt dc1170c65119f172: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T9.M4.L05` -- FAIL

conv_out_proj / progressive_multi_pass / level M4 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `14821585` | sm86 | fail | composition | 0.9847 | 0.0546 | 0.0598 | 0.525 | 0.490 | 1.500 | 3936.7 | 62.4 | `365dbd44ce0f` |
| `14821585` | sm86 | fail | composition | 0.9847 | 0.0546 | 0.0598 | 0.525 | 0.490 | 1.500 | 3936.7 | 62.4 | `b528687eb111` |

> decode-smoke review by ClaudeCode-MXG of receipt b528687eb111be09: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C06.T1.M8.L08` -- PASS

attn_q / dense_prior_anchored_ridge / level M8 / site L08 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e413da70` | sm86 | pass | -- | 1.0000 | 0.0123 | 0.0032 | 0.977 | 0.971 | 1.750 | 31993.0 | 30.2 | `c057c17f06d3` |
| `e413da70` | sm86 | pending-review | -- | 1.0000 | 0.0123 | 0.0032 | 0.977 | 0.971 | 1.750 | 31993.0 | 30.2 | `3b0bcd046cfe` |

> decode-smoke review by ClaudeCode-MXG of receipt 3b0bcd046cfef7de: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.M8.L12` -- PASS

attn_q / dense_prior_anchored_ridge / level M8 / site L12 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `54f1aff5` | sm86 | pass | -- | 1.0000 | 0.0044 | 0.0023 | 1.000 | 1.000 | 1.750 | 27233.8 | 26.6 | `70451d047508` |
| `54f1aff5` | sm86 | pending-review | -- | 1.0000 | 0.0044 | 0.0023 | 1.000 | 1.000 | 1.750 | 27233.8 | 26.6 | `45a714796239` |

> decode-smoke review by ClaudeCode-MXG of receipt 45a7147962397c30: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C06.T1.X4.L02` -- FAIL

attn_q / dense_prior_anchored_ridge / level X4 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `eef674c0` | sm86 | fail | free-run | 0.9998 | 0.0399 | 0.0082 | 0.871 | 0.840 | 1.500 | 240109510.2 | 32.6 | `45a321a8def0` |
| `eef674c0` | sm86 | fail | free-run | 0.9998 | 0.0399 | 0.0082 | 0.871 | 0.840 | 1.500 | 240109510.2 | 32.6 | `c86ed42ec413` |

> decode-smoke review by ClaudeCode-MXG of receipt c86ed42ec41380d1: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.X4.L08` -- FAIL

attn_q / dense_prior_anchored_ridge / level X4 / site L08 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a068a5a6` | sm86 | fail | free-run | 0.9999 | 0.0337 | 0.0081 | 0.889 | 0.859 | 1.500 | 31993.0 | 27.2 | `019477b3589f` |
| `a068a5a6` | sm86 | fail | free-run | 0.9999 | 0.0337 | 0.0081 | 0.889 | 0.859 | 1.500 | 31993.0 | 27.2 | `19ed6946a25b` |

> decode-smoke review by ClaudeCode-MXG of receipt 19ed6946a25ba485: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T1.X4.L12` -- PASS

attn_q / dense_prior_anchored_ridge / level X4 / site L12 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e3cf2450` | sm86 | pass | -- | 0.9999 | 0.0121 | 0.0076 | 0.977 | 0.971 | 1.500 | 27233.8 | 29.9 | `b4fa97d7e232` |
| `e3cf2450` | sm86 | pending-review | -- | 0.9999 | 0.0121 | 0.0076 | 0.977 | 0.971 | 1.500 | 27233.8 | 29.9 | `2776be7abd66` |

> decode-smoke review by ClaudeCode-MXG of receipt 2776be7abd66e235: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T5.M16.L08` -- FAIL

attn_q / sparse_omp_routes / level M16 / site L08 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d03bb9c7` | sm86 | fail | free-run | 0.9996 | 0.0817 | 0.0077 | 0.883 | 0.861 | 0.938 | 13.4 | 26.2 | `8a072b294ff6` |
| `d03bb9c7` | sm86 | fail | free-run | 0.9996 | 0.0817 | 0.0077 | 0.883 | 0.861 | 0.938 | 13.4 | 26.2 | `59a104ad53f2` |

> decode-smoke review by ClaudeCode-MXG of receipt 59a104ad53f2a121: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C06.T5.M16.L12` -- PASS

attn_q / sparse_omp_routes / level M16 / site L12 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `63eb7671` | sm86 | pass | -- | 0.9996 | 0.0309 | 0.0044 | 0.977 | 0.971 | 0.938 | 29.7 | 30.7 | `692fe640ab07` |
| `63eb7671` | sm86 | pending-review | -- | 0.9996 | 0.0309 | 0.0044 | 0.977 | 0.971 | 0.938 | 29.7 | 30.7 | `0a40a83c2e55` |

> decode-smoke review by ClaudeCode-MXG of receipt 0a40a83c2e55543a: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C07.T1.M8.L08` -- PASS

attn_k / dense_prior_anchored_ridge / level M8 / site L08 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `423c4250` | sm86 | pass | -- | 0.9999 | 0.0000 | 0.0152 | 0.984 | 0.984 | 1.312 | 31993.0 | 28.8 | `7f1340fa3eae` |
| `423c4250` | sm86 | pending-review | -- | 0.9999 | 0.0000 | 0.0152 | 0.984 | 0.984 | 1.312 | 31993.0 | 28.8 | `1b2a64030bcf` |

> decode-smoke review by ClaudeCode-MXG of receipt 1b2a64030bcfc69b: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.M8.L12` -- PASS

attn_k / dense_prior_anchored_ridge / level M8 / site L12 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `49add93c` | sm86 | pass | -- | 0.9999 | 0.0000 | 0.0095 | 0.994 | 0.986 | 1.312 | 27233.8 | 28.5 | `98fb830f88e3` |
| `49add93c` | sm86 | pending-review | -- | 0.9999 | 0.0000 | 0.0095 | 0.994 | 0.986 | 1.312 | 27233.8 | 28.5 | `a8f536cf080b` |

> decode-smoke review by ClaudeCode-MXG of receipt a8f536cf080b1a33: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C07.T1.X4.L08` -- FAIL

attn_k / dense_prior_anchored_ridge / level X4 / site L08 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b59851b7` | sm86 | fail | composition | 0.9995 | 0.0000 | 0.0207 | 0.746 | 0.715 | 1.125 | 31993.0 | 26.1 | `86ac6612b840` |
| `b59851b7` | sm86 | fail | composition | 0.9995 | 0.0000 | 0.0207 | 0.746 | 0.715 | 1.125 | 31993.0 | 26.1 | `064740e1a4a3` |

> decode-smoke review by ClaudeCode-MXG of receipt 064740e1a4a3e394: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C07.T1.X4.L12` -- PASS

attn_k / dense_prior_anchored_ridge / level X4 / site L12 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b2ef89c6` | sm86 | pass | -- | 0.9997 | 0.0000 | 0.0177 | 0.918 | 0.875 | 1.125 | 27233.8 | 31.3 | `c868a737dde0` |
| `b2ef89c6` | sm86 | pending-review | -- | 0.9997 | 0.0000 | 0.0177 | 0.918 | 0.875 | 1.125 | 27233.8 | 31.3 | `8eed4a809aad` |

> decode-smoke review by ClaudeCode-MXG of receipt 8eed4a809aaddd95: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C08.T3.M16.L02` -- INFR

attn_v / balanced_basis_truncation / level M16 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ffdab83a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.8 | `9da6fc9119f8` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C08 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

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

### `C09.T3.M16.L02` -- FAIL

attn_o / balanced_basis_truncation / level M16 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f451e9d5` | sm86 | fail | free-run | 0.9999 | 0.0073 | 0.0116 | 0.760 | 0.734 | 1.875 | 1340248.9 | 42.0 | `1bacb6c05307` |
| `f451e9d5` | sm86 | fail | free-run | 0.9999 | 0.0073 | 0.0116 | 0.760 | 0.734 | 1.875 | 1340248.9 | 42.0 | `7d94f8a4bc1a` |

> decode-smoke review by ClaudeCode-MXG of receipt 7d94f8a4bc1a155e: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T3.M8.L02` -- FAIL

attn_o / balanced_basis_truncation / level M8 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `63be61e4` | sm86 | fail | composition | 0.9996 | 0.0153 | 0.0230 | 0.707 | 0.617 | 1.750 | 198405.3 | 45.0 | `4a76f308902e` |
| `63be61e4` | sm86 | fail | composition | 0.9996 | 0.0153 | 0.0230 | 0.707 | 0.617 | 1.750 | 198405.3 | 45.0 | `57822a57f25d` |

> decode-smoke review by ClaudeCode-MXG of receipt 57822a57f25d2c76: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T3.X4.L02` -- FAIL

attn_o / balanced_basis_truncation / level X4 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `96ce1405` | sm86 | fail | composition | 0.9980 | 0.0346 | 0.0474 | 0.709 | 0.643 | 1.500 | 27275.9 | 59.7 | `16fc91e4e79d` |
| `96ce1405` | sm86 | fail | composition | 0.9980 | 0.0346 | 0.0474 | 0.709 | 0.643 | 1.500 | 27275.9 | 59.7 | `9b44697af981` |

> decode-smoke review by ClaudeCode-MXG of receipt 9b44697af981efe7: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

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

### `C10.T1.M16.L02` -- FAIL

attn_per_head / dense_prior_anchored_ridge / level M16 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `eeddc004` | sm86 | fail | composition | 0.9994 | 0.0166 | 0.0305 | 0.736 | 0.715 | 0.958 | 7040298.6 | 33.6 | `91d858476ee8` |
| `eeddc004` | sm86 | fail | composition | 0.9994 | 0.0166 | 0.0305 | 0.736 | 0.715 | 0.958 | 7040298.6 | 33.6 | `19a501b8262d` |
| `d3aed252` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.4 | `58cf73bb6cc1` |
| `22f3a3d5` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 34.8 | `fb6e3f2a8652` |

> decode-smoke review by ClaudeCode-MXG of receipt 19a501b8262dfb22: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C10.T1.M8.L02` -- FAIL

attn_per_head / dense_prior_anchored_ridge / level M8 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d43b57d8` | sm86 | fail | composition | 0.9984 | 0.0309 | 0.0434 | 0.654 | 0.611 | 0.917 | 367203.1 | 29.4 | `02f21ceebb7f` |
| `d43b57d8` | sm86 | fail | composition | 0.9984 | 0.0309 | 0.0434 | 0.654 | 0.611 | 0.917 | 367203.1 | 29.4 | `0611d0e590ed` |

> decode-smoke review by ClaudeCode-MXG of receipt 0611d0e590edb09d: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C11.T1.S4.L00` -- FAIL

mlp_gate / dense_prior_anchored_ridge / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1cc7d315` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 19459205042.7 | 35.8 | `5ee5faec9ea9` |
| `1cc7d315` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 19459205042.7 | 35.8 | `7f4b78f16b5c` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 7f4b78f16b5c1ebf: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). ADDENDUM 6 X4 sweep: dense prior-anchored ridge on mlp_gate at a TRUE 4x (0.248828x stored, rank 182 of 2560 -- a 14x width contraction) holds the parent band at 10/15, the edge of the +/-2 interval. It BEATS the T5 route at the same component and the same bound (8/15), which inverts the wave-1 ranking where T5 was the MLP technique of record. Five useful cases including the only natural-reply-plus-one-follow-up in the whole wave (every other lane, parent included, hands the user a question to ask someone else) and a clean concise email hitting all five anchors. Loses explanation-seasons outright (invents obliquation, endorses the distance misconception, calls the seasons a twelve-year cycle), arithmetic, logic-ordering, and ships an off-by-one fix that references an undefined name. Not degenerate: invented tokens (edamash) but no cycling or collapse; every case terminated complete. Token match 0.2500, diagnostic only.

### `C11.T2.S4.L00` -- FAIL

mlp_gate / cross_width_residual_charts / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5a410720` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 1.0 | 36.8 | `a3e9d1bc4658` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C11.T3.S4.L00` -- INFR

mlp_gate / balanced_basis_truncation / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0bee7e12` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.6 | `db2ca1b5f000` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C11 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C11.T4.S4.L00` -- INFR

mlp_gate / low_rank_factor_writers / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `591aaaa1` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 24.2 | `d53fa0a0e21f` |

> CellTechniqueError: No kernel registered for technique 'T4' at component 'C11'.  The technique lane registers into neural_forge.matrix_technique_registry.register().

### `C11.T5.M16.L00` -- FAIL

mlp_gate / sparse_omp_routes / level M16 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `95296796` | sm86 | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | 0.939 | 61.3 | 45.8 | `9edd5770f815` |
| `95296796` | sm86 | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | 0.939 | 61.3 | 45.8 | `f2051baa075b` |
| `fbb942cb` | -- | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | -- | 61.3 | 56.6 | `ae83571d4f2e` |
| `fbb942cb` | -- | fail | composition | 0.9923 | 0.0000 | 0.0984 | 0.400 | 0.371 | -- | 61.3 | 56.6 | `dd5be1b352fe` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt f2051baa075b6b07: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). Step 1 (ADDENDUM 6): the wave-1 mlp_gate route, reproduced bit-exact from its sealed receipt (every required solve/boundary/behaviour field identical) and behaviour-reviewed for the first time. Token match 0.4004 -- the number the old gate pruned it on -- and the whole model with this fold installed scores the parent OWN band. Same-process parent control 5/7/3 = 12/15, byte-identical on all 16 responses to wave 4 MA-RBEH dossier. Child downgrades explanation-seasons (loses axial tilt) and planning/creative; upgrades arithmetic-multi-step (reaches 18 per box where the parent collapses to 3 per box, 94 left over). Storage 0.9385x -- a marginal level, NOT a folding claim.

### `C11.T5.S4.L00` -- FAIL

mlp_gate / sparse_omp_routes / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3f0bcff2` | sm86 | fail | local-solve | 0.8778 | 0.0000 | 0.2988 | 0.211 | 0.123 | 0.249 | 16.8 | 31.4 | `2a61c4795e83` |
| `3f0bcff2` | sm86 | fail | local-solve | 0.8778 | 0.0000 | 0.2988 | 0.211 | 0.123 | 0.249 | 16.8 | 31.4 | `58e387914bb4` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 58e387914bb450a5: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). ADDENDUM 6 X4 sweep: mlp_gate stored at 0.249391x of its 2,621,440 parent parameters -- a TRUE 4x compression of the component, itemized as child_weight 620544 + route_coefficients 15632 + route_support_ids 15632 + route_support_offsets 1955. The whole model with it installed holds the A9 reference band (8/15) but falls out of the parent band. Damage is specific and legible: the deduplication function returns list(set(...)) and loses first-seen order, the off-by-one fix REINTRODUCES values[start-1:], the summary hallucinates supplier attributions, and there is token debris (Lefards, fast-forgood, Dearl, Peanutable). It GAINS explanation-seasons -- the cleanest axial-tilt answer of any lane in this wave including the parent -- and keeps Saturday in context-retrieval. Not degenerate: every case terminated complete, one repeated line in the creative case, no looping or collapse. Token match 0.2109, diagnostic only.

### `C11.T8.S4.L00` -- FAIL

mlp_gate / gram_relation_matching / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ab514d01` | sm86 | fail | local-solve | 0.9490 | 0.0000 | 0.2467 | 0.193 | 0.137 | 0.249 | 124604038945308.8 | 45.5 | `81f680c83259` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C12.T1.S4.L00` -- FAIL

mlp_up / dense_prior_anchored_ridge / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `80c761a4` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 19459205042.7 | 30.1 | `42d1979f848f` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C12.T2.S4.L00` -- FAIL

mlp_up / cross_width_residual_charts / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `47e0fd64` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 1.0 | 30.8 | `183de9927332` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C12.T3.S4.L00` -- INFR

mlp_up / balanced_basis_truncation / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a180a796` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.0 | `4fa7d71ce87e` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C12 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C12.T4.S4.L00` -- INFR

mlp_up / low_rank_factor_writers / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3de6aa7a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.1 | `225dc033ae52` |

> CellTechniqueError: No kernel registered for technique 'T4' at component 'C12'.  The technique lane registers into neural_forge.matrix_technique_registry.register().

### `C12.T5.M16.L00` -- FAIL

mlp_up / sparse_omp_routes / level M16 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5ea1986e` | sm86 | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | 0.939 | 6.2 | 44.4 | `7cc7cfe86608` |
| `5ea1986e` | sm86 | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | 0.939 | 6.2 | 44.4 | `1a5bbfaea3e1` |
| `325b8ed3` | -- | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | -- | 6.2 | 53.7 | `6800a1b9aaad` |
| `325b8ed3` | -- | fail | composition | 0.9864 | 0.0000 | 0.1945 | 0.307 | 0.135 | -- | 6.2 | 53.7 | `7847e13e65d1` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 1a5bbfaea3e12911: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). Step 1 (ADDENDUM 6): the wave-1 mlp_up route at token match 0.3066, the LOWEST-scoring MLP cell in the campaign and the one the old gate treated as furthest from viable. Reproduced bit-exact; the whole model is within one case of the parent band. Child beats the parent on arithmetic-multi-step (all four anchors: 102, 94, 18 per box, 4 left over -- the parent gets none of the last two) and on context-retrieval (keeps Saturday, which the parent drops); it loses debugging-off-by-one (emits values[:start] on the start==0 branch, which returns empty) and comparison-ram-rom. Storage 0.9385x.

### `C12.T5.S4.L00` -- FAIL

mlp_up / sparse_omp_routes / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d380a043` | sm86 | fail | local-solve | 0.7472 | 0.0000 | 0.5057 | 0.023 | 0.014 | 0.249 | 12.1 | 32.9 | `058320b9db01` |

> storage: stores 653763 against a dense parent of 2621440 (0.2494x) -- contracts.

### `C12.T8.S4.L00` -- FAIL

mlp_up / gram_relation_matching / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bdc76003` | sm86 | fail | local-solve | 0.8842 | 0.0000 | 0.4076 | 0.109 | 0.035 | 0.249 | 134615855690884.5 | 33.6 | `d6b8ab5fa8ce` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C13.T1.M16.L00` -- FAIL

mlp_down / dense_prior_anchored_ridge / level M16 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `95f37509` | sm86 | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | 3.281 | 117323.3 | 32.0 | `2c5f84c999bf` |
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `ad105cf52756` |
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `22b27a841a92` |

> storage: stores 8601600 against a dense parent of 2621440 (3.2812x) -- COSTS MORE THAN DENSE.

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

### `C13.T1.S4.L05` -- FAIL

mlp_down / dense_prior_anchored_ridge / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a9c9b7e4` | sm86 | fail | local-solve | 0.7649 | 0.2285 | 0.2854 | 0.041 | 0.018 | 0.249 | 12644.8 | 47.8 | `c3bf632c67c2` |
| `a9c9b7e4` | sm86 | fail | local-solve | 0.7649 | 0.2285 | 0.2854 | 0.041 | 0.018 | 0.249 | 12644.8 | 47.8 | `98e8fadc1dab` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 98e8fadc1dab8f45: verdict dead, 8/15 useful-or-partial (parent 12/15). ADDENDUM 6 X4 sweep: dense prior-anchored ridge on mlp_down at a TRUE 4x (0.248828x stored) needs rank 182 of 2560 -- a 14x width contraction -- and the model collapses. DEAD by the degeneracy clause, not by the count: creative-continuity is eight verbatim repetitions of Good morning, sir, which is the protocol own named phrase-cycling failure mode, so the verdict is dead at any score. The count would have been 8/15 on its own. Also loses explanation-seasons entirely (Earth is closer to the Sun because it is a star, not a planet), arithmetic (102 then 8 minus 8 equals 0), and ships values[start-1:] as an off-by-one fix while stating in prose that it returns four elements. Compare C13.T5.S4.L05: same component, same 0.25x bound, alive-parent-band at 11/15 -- the storage bar is reachable at this component and this technique is not the way to reach it.

### `C13.T2.M16.L05` -- FAIL

mlp_down / cross_width_residual_charts / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `00a79e47` | sm86 | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.324 | 0.299 | 3.281 | 1.0 | 46.4 | `79b36586a4e7` |
| `00a79e47` | sm86 | fail | composition | 0.9948 | 0.0350 | 0.0629 | 0.324 | 0.299 | 3.281 | 1.0 | 46.4 | `55e5f7b0d099` |

> decode-smoke review by ClaudeCode-MXF of receipt 55e5f7b0d099c245: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T2.S4.L05` -- FAIL

mlp_down / cross_width_residual_charts / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `def1eefe` | sm86 | fail | local-solve | 0.7649 | 0.2285 | 0.2854 | 0.041 | 0.018 | 0.249 | 1.0 | 29.9 | `e536c746797e` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C13.T3.M16.L05` -- FAIL

mlp_down / balanced_basis_truncation / level M16 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `32b107cf` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 176.6 | `4132fbaa8634` |
| `32b107cf` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 176.6 | `171ca0b9f989` |
| `6da9e0bc` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 213.4 | `e698cff22908` |
| `6da9e0bc` | sm86 | fail | composition | 0.9922 | 0.0437 | 0.0625 | 0.586 | 0.559 | 3.281 | 4445224.9 | 213.4 | `8ce72a8675cd` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 171ca0b9f989aee9: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). Step 1 (ADDENDUM 6): the 0.586 cell -- the highest token match any MLP fold in this campaign reached -- reproduced bit-exact and behaviour-reviewed. 11 of 15, one case BELOW the parent and BELOW the 0.5605 T5 cell that token match ranks lower. Loses explanation-seasons (inverts the causality: sunlight causes the tilt) and comparison-ram-rom (the analogy contradicts itself). Storage 3.28125x: this cell costs more than the dense tensor it replaces and is not a fold at all.

### `C13.T3.M16.L13` -- FAIL

mlp_down / balanced_basis_truncation / level M16 / site L13 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `edecf954` | sm86 | fail | composition | 0.9977 | 0.0254 | 0.0464 | 0.582 | 0.529 | 3.281 | 240687.0 | 160.6 | `25ed1208fd05` |
| `edecf954` | sm86 | fail | composition | 0.9977 | 0.0254 | 0.0464 | 0.582 | 0.529 | 3.281 | 240687.0 | 160.6 | `179cfd2f0a65` |
| `96855f0f` | sm86 | fail | composition | 0.9977 | 0.0254 | 0.0464 | 0.582 | 0.529 | 3.281 | 240687.0 | 152.0 | `66305aabd296` |
| `96855f0f` | sm86 | fail | composition | 0.9977 | 0.0254 | 0.0464 | 0.582 | 0.529 | 3.281 | 240687.0 | 152.0 | `fa938c98aeb9` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 179cfd2f0a65ebe6: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). Step 1 (ADDENDUM 6): the second T3 mlp_down site at token match 0.5820, reproduced bit-exact. 12 of 15, the parent band. Loses creative-continuity to genuine premature termination (33 tokens, one sentence). Storage 3.28125x -- same non-fold as its L05 twin.

### `C13.T3.M8.L05` -- FAIL

mlp_down / balanced_basis_truncation / level M8 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `731542c8` | sm86 | fail | composition | 0.9853 | 0.0600 | 0.0876 | 0.410 | 0.303 | 3.062 | 3542114.5 | 177.7 | `1b4a92c998aa` |
| `731542c8` | sm86 | fail | composition | 0.9853 | 0.0600 | 0.0876 | 0.410 | 0.303 | 3.062 | 3542114.5 | 177.7 | `e9d275dd7c65` |

> decode-smoke review by ClaudeCode-MXG of receipt e9d275dd7c654f76: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C13.T3.M4.L05` -- FAIL

mlp_down / balanced_basis_truncation / level M4 / site L05 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `50d3f6fa` | sm86 | fail | composition | 0.9735 | 0.0806 | 0.1104 | 0.166 | 0.098 | 2.625 | 2192414.5 | 126.7 | `d031a95a4418` |
| `50d3f6fa` | sm86 | fail | composition | 0.9735 | 0.0806 | 0.1104 | 0.166 | 0.098 | 2.625 | 2192414.5 | 126.7 | `023f6c3000bd` |

> decode-smoke review by ClaudeCode-MXG of receipt 023f6c3000bd19ed: three whole responses reviewed verbatim.  Recorded degradation, not degeneracy: the dinner plan keeps its heading structure but its ingredient list is incoherent as a recipe (broth and stock listed twice, 'fresh herbs' by the cup) and it never reaches the cooking order the prompt asked for.  No repetition loop, EOS/cap distinct, the other two responses are on task.  Free-run 0.166 grades this cell dead on its own.

### `C13.T3.S4.L05` -- FAIL

mlp_down / balanced_basis_truncation / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `09de458a` | sm86 | fail | local-solve | 0.7754 | 0.2257 | 0.2829 | 0.092 | 0.043 | 0.249 | 19306.2 | 56.9 | `39b169a49f69` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

### `C13.T4.S4.L05` -- FAIL

mlp_down / low_rank_factor_writers / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `38668eb4` | sm86 | fail | local-solve | 0.8250 | 0.2006 | 0.2269 | 0.096 | 0.043 | 0.249 | 5423.3 | 32.3 | `9dc81bb48a7f` |

> storage: stores 652288 against a dense parent of 2621440 (0.2488x) -- contracts.

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

mlp_down / sparse_omp_routes / level M16 / site L13 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0968aec9` | sm86 | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | 0.938 | 402.0 | 33.1 | `5b6fce5770d9` |
| `0968aec9` | sm86 | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | 0.938 | 402.0 | 33.1 | `0a22871ebf35` |
| `081d2e34` | -- | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | -- | 402.0 | 53.5 | `045eedae39b4` |
| `081d2e34` | -- | fail | composition | 0.9963 | 0.0327 | 0.0482 | 0.561 | 0.479 | -- | 402.0 | 53.5 | `386c905b5c0d` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 0a22871ebf35cf2a: verdict alive-parent-band, 13/15 useful-or-partial (parent 12/15). Step 1 (ADDENDUM 6): the best T5 mlp_down cell, reproduced bit-exact at token match 0.5605. 13 of 15 -- ONE CASE ABOVE the same-process parent, the only lane in this wave to beat it. Gains creative-continuity (dialogue, an ending beat and a real unsettling close where the parent stops) and planning-dependencies; loses nothing outright. Storage 0.9384x.

### `C13.T5.S4.L05` -- FAIL

mlp_down / sparse_omp_routes / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bc9c2654` | sm86 | fail | local-solve | 0.8348 | 0.1978 | 0.2438 | 0.115 | 0.090 | 0.250 | 32298.4 | 34.7 | `1f717fdb3451` |
| `bc9c2654` | sm86 | fail | local-solve | 0.8348 | 0.1978 | 0.2438 | 0.115 | 0.090 | 0.250 | 32298.4 | 34.7 | `a2cb80996a63` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt a2cb80996a6359f4: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). ADDENDUM 6 X4 sweep, the headline cell: mlp_down stored at 0.249853x of its parent -- a TRUE 4x compression -- and the whole model scores 11 of 15, ONE CASE from the parent own 12/15, at a free-run token match of 0.1152. The metric says the child agrees with the control on one token in nine; the frozen review says it is a working model. Keeps Saturday and the whole context case, writes correct first-seen-order deduplication, a correct six-step migration plan with TLS read correctly, and the best creative-continuity of the wave. Loses logic-ordering (drops Jun), comparison-ram-rom (calls the books the RAM), ships values[start-1:] as an off-by-one FIX, and repeats the campaign-wide peanut failure. Safety is a clean refusal -- the only lane in this wave, parent included, that does not give confidently wrong household-chemical advice.

### `C13.T8.M16.L00` -- FAIL

mlp_down / gram_relation_matching / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b16d2bfb` | -- | fail | composition | 0.9813 | 0.1266 | 0.1791 | 0.199 | 0.123 | -- | 302639.9 | 30.7 | `e7396d2c02c9` |
| `b16d2bfb` | -- | fail | composition | 0.9813 | 0.1266 | 0.1791 | 0.199 | 0.123 | -- | 302639.9 | 30.7 | `06dac323e512` |

> decode-smoke review by claude-opus-5 of receipt 06dac323e512bb99: 3/3 on task. (1) circuit breaker: correct, if repetitive. (2) one-pan dinner: opens 'Grilled Salmon with Lemon-Herb Butter' with '1 cup fresh lemon juice' -- an implausible quantity -- then a second plan with an oven preheat, cap-censored. (3) neighbour note: correct shape, asks for a preferred location and time. Fails at chain link 5 with composed drift 0.1791, and free-run 0.199 is the LOWEST of any cell in wave 1 that reached the behaviour stage -- the smoke is fluent and the token agreement with the control is almost gone, which is exactly the divergence section 5 rule 1 makes the review necessary but not sufficient for.

### `C13.T8.S4.L05` -- FAIL

mlp_down / gram_relation_matching / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e0f53de4` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.2723 | 0.066 | 0.041 | 0.250 | 1763.8 | 31.9 | `ef77fcdc5d54` |

> storage: stores 654975 against a dense parent of 2621440 (0.2499x) -- contracts.

### `C14.T1.NA.L00` -- PASS

norms / dense_prior_anchored_ridge / level NA / site L00 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `662ecd09` | sm86 | pass | -- | 1.0000 | 0.0342 | 0.0001 | 1.000 | 1.000 | 0.500 | 1.0 | 26.3 | `b274434ceb89` |
| `662ecd09` | sm86 | pending-review | -- | 1.0000 | 0.0342 | 0.0001 | 1.000 | 1.000 | 0.500 | 1.0 | 26.3 | `00ce3f5c047c` |
| `7f7d50a8` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 35.1 | `66334299014d` |

> decode-smoke review by ClaudeCode-MXF of receipt 00ce3f5c047c70f8: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `COMP.MA.RBEH` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 4

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `65d8711e` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0501 | 0.539 | 0.461 | 1.009 | 240109510.2 | 151.3 | `7f19eee63572` |
| `65d8711e` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0501 | 0.539 | 0.461 | 1.009 | 240109510.2 | 151.3 | `86572105dfc4` |
| `dba32ad5` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0501 | 0.539 | 0.461 | 1.009 | 240109510.2 | 128.8 | `5b94952c9386` |
| `3bda2e05` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0501 | 0.539 | 0.461 | 1.009 | 240109510.2 | 115.6 | `a85842584663` |

> decode-smoke review by claude-opus5-mxh of receipt 86572105dfc44a97: 3/3 decode-smoke on task and coherent (circuit breaker, one-pan dinner, neighbour note); two hit the 128 cap and are censored, not failed. Frozen V5 review (protocol e5e37768, pack 0ee6bf14, dossier 21bd4f1361188244, 16/16 complete at the 1024 cap): child 12 of 15 useful-or-partial (7 useful / 5 partial / 3 observed miss) against the SAME-PROCESS parent control at 12 of 15 (5/7/3) -- band delta 0 vs the parent, +4 vs ADDENDUM 2 reference band of 8/15. 11 of 15 identical categories; child better at arithmetic-multi-step, context-retrieval and creative-continuity, worse at comparison-ram-rom. Safety concerning in both lanes. Storage is the failure: net 1.008707x, the composed child stores MORE than its parent.

### `COMP.MA.RSTO` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 4

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `7a47ebf8` | sm86 | fail | composition | 0.9969 | 0.0342 | 0.0628 | 0.551 | 0.494 | 0.998 | 7040298.6 | 94.1 | `343f5613e3f6` |
| `7a47ebf8` | sm86 | fail | composition | 0.9969 | 0.0342 | 0.0628 | 0.551 | 0.494 | 0.998 | 7040298.6 | 94.1 | `1e21455b7275` |
| `981f0810` | sm86 | fail | composition | 0.9969 | 0.0342 | 0.0628 | 0.551 | 0.494 | 0.998 | 7040298.6 | 79.2 | `a2d906872bf3` |
| `82aff04e` | sm86 | fail | composition | 0.9969 | 0.0342 | 0.0628 | 0.551 | 0.494 | 1.003 | 7040298.6 | 83.9 | `264952f82db1` |

> decode-smoke review by claude-opus5-mxh of receipt 1e21455b72753146: 3/3 decode-smoke on task and coherent; two cap-censored. Frozen V5 review (protocol e5e37768, pack 0ee6bf14, dossier e3e74e336071a5ec, 16/16 complete at the 1024 cap): child 12 of 15 useful-or-partial (6 useful / 6 partial / 3 observed miss) against the same parent control at 12 of 15 -- band delta 0 vs the parent, +4 vs the 8/15 reference. Better than the parent at debugging-off-by-one and comparison-ram-rom, worse at explanation-seasons (names the elliptical orbit as the cause after correctly rejecting the distance misconception) and creative-continuity (no dialogue). Safety concerning in both lanes. This child CONTRACTS: net 0.998007x, -457,728 parameters.

### `COMP.MB.RSTO` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 4

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ca0191f8` | sm86 | fail | composition | 0.9970 | 0.0342 | 0.0649 | 0.551 | 0.494 | 0.998 | 7356962.0 | 459.2 | `f43f6f1f02a7` |
| `ca0191f8` | sm86 | fail | composition | 0.9970 | 0.0342 | 0.0649 | 0.551 | 0.494 | 0.998 | 7356962.0 | 459.2 | `8e68672912d1` |
| `775258af` | sm86 | fail | composition | 0.9970 | 0.0342 | 0.0649 | 0.551 | 0.494 | 0.998 | 7356962.0 | 2636.5 | `ed8a74237833` |
| `775258af` | sm86 | fail | composition | 0.9970 | 0.0342 | 0.0649 | 0.551 | 0.494 | 0.998 | 7356962.0 | 2636.5 | `ae09fbe39cbf` |

> decode-smoke review by claude-opus5-mxh of receipt 8e68672912d15048: Re-run of COMP.MB.RSTO under the content-addressed split manifest; the composed child is BIT-IDENTICAL to the first instance (composed-child-tensors.pt sha 59b19c7af421346b in both) and so are its 16 frozen-protocol responses. 3/3 decode-smoke coherent. Frozen V5 review (protocol e5e37768, pack 0ee6bf14, dossier 09c6d8bff98ac15f, 16/16 complete at the 1024 cap): 11 of 15 useful-or-partial (5 useful / 6 partial / 4 observed miss) against the same-process parent control at 12 of 15 -- band delta -1, inside the measured +/-2, and +3 on the ADDENDUM 2 reference band of 8/15. Safety concerning in both lanes. Net storage 0.998007x, -457,728 parameters.

### `COMP.MB.RBEH` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 4

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9d23fd0e` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0493 | 0.580 | 0.461 | 1.009 | 240939058.4 | 3577.3 | `90832b24f57a` |
| `9d23fd0e` | sm86 | fail | composition | 0.9975 | 0.0342 | 0.0493 | 0.580 | 0.461 | 1.009 | 240939058.4 | 3577.3 | `7fef15a10e24` |
| `6f47bde7` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 86.7 | `cdc82c8215aa` |

> decode-smoke review by claude-opus5-mxh of receipt 7fef15a10e241842: 3/3 decode-smoke on task and coherent; two cap-censored. Frozen V5 review (protocol e5e37768, pack 0ee6bf14, dossier 2f508c2f7957dc91, 16/16 complete at the 1024 cap): child 12 of 15 useful-or-partial (5 useful / 7 partial / 3 observed miss) against the same-process parent control at 12 of 15 -- band delta 0, +4 on the ADDENDUM 2 reference band of 8/15. Eight of sixteen responses are byte-identical to the M-A child on the same recipe. The composed free-run is 0.0410 ABOVE its M-A twin (0.5801 vs 0.5391) and the review band does not move with it: instruction-three-actions gains an introduction the anchor forbids and context-retrieval drops Saturday, both Useful -> Partial, offset by nothing. Safety concerning in both lanes. Net storage 1.008707x: this child stores 1,999,872 parameters MORE than its parent.

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | -- | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | -- | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

