# Stage-1 matrix campaign -- ledger

Rendered 2026-08-11T19:02:25+00:00 from `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`.

**178 cells sealed** (378 ledger rows including re-runs); newest seal 2026-08-11T19:01:50.539+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 18 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `fail` | 147 | crossed a chain threshold; see first_divergent_metric |
| `infra` | 13 | calibration refused the cell -- void, not evidence |

## 0. ADDENDUM 6 -- behaviour verdicts (the primary grade)

ADDENDUM 6 §6.1: the grade of a cell is the frozen 15-prompt review of the **whole model** with that cell's component folded and every other component copy_exact, scored against the parent's own 12/15 on the same surface in the same process.  `token_match_vs_control` is a diagnostic column here and gates nothing -- wave 4 sealed a child at 0.551 token match and 12/15, the parent's own band.

| verdict | cells | meaning |
|---|---:|---|
| `alive-parent-band` | 33 | within 2 cases of the parent's own 12/15 on the frozen surface, coherent, emit-zero floor passed |
| `alive-degraded` | 18 | >= the A9 band of 8/15, outside the parent's +/-2 |
| `impaired` | 11 | < 8/15 but still coherent, on-task language |
| `dead` | 18 | incoherent, emit-zero floor failed, or degenerate output |
| `cannot-express-4x` | 1 | the technique cannot state this component at the level's storage bound; nothing was measured about the model |

| cell | site | level | stored/parent | bound | verdict | review | parent | smoke | token-match (diag) |
|---|---|---|---:|---|---|---:|---:|---|---:|
| `C03.T8.S4.L00` | L00 | S4 | 0.2500 | 0.250 ok | **dead** | 0/15 | 12 | pending | 0.0078 |
| `C05.T1.S4.L05` | L05 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.1367 |
| `C05.T2.S4.L05` | L05 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.1367 |
| `C05.T3.S4.L05` | L05 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 12/15 | 12 | coherent | 0.2148 |
| `C05.T4.S4.L05` | L05 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 12/15 | 12 | coherent | 0.2090 |
| `C05.T5.S4.L05` | L05 | S4 | 0.2493 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.1602 |
| `C05.T8.S4.L05` | L05 | S4 | 0.2493 | 0.250 ok | **alive-parent-band** | 12/15 | 12 | coherent | 0.1562 |
| `C06.T1.S4.L12` | L12 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2363 |
| `C06.T2.S4.L12` | L12 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2363 |
| `C06.T5.S4.L12` | L12 | S4 | 0.2490 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.3613 |
| `C06.T8.S4.L12` | L12 | S4 | 0.2500 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2363 |
| `C07.T1.S4.L12` | L12 | S4 | 0.2490 | 0.250 ok | **alive-parent-band** | 11/15 | 12 | coherent | 0.4141 |
| `C07.T2.S4.L12` | L12 | S4 | 0.2490 | 0.250 ok | **alive-parent-band** | 11/15 | 12 | coherent | 0.4141 |
| `C07.T5.S4.L12` | L12 | S4 | 0.2490 | 0.250 ok | **alive-parent-band** | 12/15 | 12 | coherent | 0.2363 |
| `C07.T8.S4.L12` | L12 | S4 | 0.2490 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.3633 |
| `C08.T1.S4.L02` | L02 | S4 | 0.2490 | 0.250 ok | **dead** | 5/15 | 12 | coherent | 0.0996 |
| `C08.T2.S4.L02` | L02 | S4 | 0.2490 | 0.250 ok | **dead** | 5/15 | 12 | coherent | 0.0996 |
| `C08.T5.S4.L02` | L02 | S4 | 0.2490 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.1035 |
| `C08.T8.S4.L02` | L02 | S4 | 0.2490 | 0.250 ok | **dead** | 9/15 | 12 | coherent | 0.1758 |
| `C09.T1.S4.L02` | L02 | S4 | 0.2500 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.0625 |
| `C09.T2.S4.L02` | L02 | S4 | 0.2500 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.0625 |
| `C09.T3.S4.L02` | L02 | S4 | 0.2500 | 0.250 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.1133 |
| `C09.T4.S4.L02` | L02 | S4 | 0.2500 | 0.250 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.1172 |
| `C09.T5.S4.L02` | L02 | S4 | 0.2493 | 0.250 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.1172 |
| `C09.T8.S4.L02` | L02 | S4 | 0.2493 | 0.250 ok | **dead** | 5/15 | 12 | coherent | 0.0938 |
| `C10.T1.S2.L02` | L02 | S2 | 0.5000 | 0.500 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.1465 |
| `C10.T1.S4.L02` | L02 | S4 | -- | -- | **cannot-express-4x** | -- | -- | coherent | 0.0000 |
| `C11.T1.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2500 |
| `C11.T1.S8.L00` | L00 | S8 | 0.1244 | 0.125 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.1680 |
| `C11.T2.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.2500 |
| `C11.T2.S8.L00` | L00 | S8 | 0.1244 | 0.125 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.1680 |
| `C11.T5.M16.L00` | L00 | M16 | 0.9385 | -- | **alive-parent-band** | 12/15 | 12 | coherent | 0.4004 |
| `C11.T5.S4.L00` | L00 | S4 | 0.2494 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.2109 |
| `C11.T8.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-parent-band** | 11/15 | 12 | coherent | 0.1934 |
| `C11.T8.S8.L00` | L00 | S8 | 0.1244 | 0.125 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.1836 |
| `C12.T1.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.1191 |
| `C12.T2.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-degraded** | 8/15 | 12 | coherent | 0.1191 |
| `C12.T5.M16.L00` | L00 | M16 | 0.9385 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.3066 |
| `C12.T5.S4.L00` | L00 | S4 | 0.2494 | 0.250 ok | **dead** | 4/15 | 12 | coherent | 0.0234 |
| `C12.T8.S4.L00` | L00 | S4 | 0.2488 | 0.250 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.1094 |
| `C13.T1.M16.L00` | L00 | M16 | 3.2812 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.5352 |
| `C13.T1.S4.L05` | L05 | S4 | 0.2488 | 0.250 ok | **dead** | 8/15 | 12 | coherent | 0.0410 |
| `C13.T2.S4.L05` | L05 | S4 | 0.2488 | 0.250 ok | **dead** | 8/15 | 12 | coherent | 0.0410 |
| `C13.T3.M16.L05` | L05 | M16 | 3.2812 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.5859 |
| `C13.T3.M16.L13` | L13 | M16 | 3.2812 | -- | **alive-parent-band** | 12/15 | 12 | coherent | 0.5820 |
| `C13.T3.S4.L05` | L05 | S4 | 0.2488 | 0.250 ok | **impaired** | 7/15 | 12 | coherent | 0.0918 |
| `C13.T4.S4.L05` | L05 | S4 | 0.2488 | 0.250 ok | **alive-parent-band** | 10/15 | 12 | coherent | 0.0957 |
| `C13.T4.S8.L05` | L05 | S8 | 0.1244 | 0.125 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.0547 |
| `C13.T5.M16.L13` | L13 | M16 | 0.9384 | -- | **alive-parent-band** | 13/15 | 12 | coherent | 0.5605 |
| `C13.T5.S4.L05` | L05 | S4 | 0.2499 | 0.250 ok | **alive-parent-band** | 11/15 | 12 | coherent | 0.1152 |
| `C13.T5.S8.L05` | L05 | S8 | 0.1247 | 0.125 ok | **alive-degraded** | 9/15 | 12 | coherent | 0.0547 |
| `C13.T8.S4.L05` | L05 | S4 | 0.2499 | 0.250 ok | **alive-parent-band** | 12/15 | 12 | coherent | 0.0664 |
| `C13.T8.S8.L05` | L05 | S8 | 0.1247 | 0.125 ok | **impaired** | 6/15 | 12 | coherent | 0.1035 |
| `COMP.MA.MXJ1` | GLOBAL | NA | 0.9914 | -- | **alive-parent-band** | 12/15 | 12 | coherent | 0.0664 |
| `COMP.MA.MXJ2` | GLOBAL | NA | 0.9897 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.1133 |
| `COMP.MA.MXJ3` | GLOBAL | NA | 0.9863 | -- | **impaired** | 6/15 | 12 | coherent | 0.1328 |
| `COMP.MA.MXJ4` | GLOBAL | NA | 0.9777 | -- | **impaired** | 4/15 | 12 | coherent | 0.0469 |
| `COMP.MA.MXJ5` | GLOBAL | NA | 0.9743 | -- | **dead** | 4/15 | 12 | coherent | 0.0488 |
| `COMP.MA.MXJ3A` | GLOBAL | NA | 0.9863 | -- | **impaired** | 5/15 | 12 | coherent | 0.1328 |
| `COMP.MA.MXJ3B` | GLOBAL | NA | 0.9863 | -- | **impaired** | 7/15 | 12 | coherent | 0.1406 |
| `COMP.MA.MXJ3C` | GLOBAL | NA | 0.9812 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.1133 |
| `COMP.MA.MXJ4C` | GLOBAL | NA | 0.9777 | -- | **dead** | 8/15 | 12 | coherent | 0.0605 |
| `COMP.MB.MXJ3` | GLOBAL | NA | 0.9863 | -- | **impaired** | 6/15 | 12 | coherent | 0.0703 |
| `COMP.MA.MXJP2A` | GLOBAL | NA | 0.9812 | -- | **dead** | 5/15 | 12 | coherent | 0.0879 |
| `COMP.MA.MXJP2B` | GLOBAL | NA | 0.9812 | -- | **impaired** | 7/15 | 12 | coherent | 0.1211 |
| `COMP.MA.MXJP2C` | GLOBAL | NA | 0.9726 | -- | **dead** | 4/15 | 12 | coherent | 0.0469 |
| `COMP.MA.MXK1` | GLOBAL | NA | 0.9812 | -- | **alive-degraded** | 8/15 | 12 | coherent | 0.0527 |
| `COMP.MA.MXK2` | GLOBAL | NA | 0.9812 | -- | **dead** | 7/15 | 12 | coherent | 0.0938 |
| `COMP.MA.MXK3` | GLOBAL | NA | 0.9812 | -- | **dead** | 6/15 | 12 | coherent | 0.1094 |
| `COMP.MA.MXK4` | GLOBAL | NA | 0.9880 | -- | **alive-degraded** | 8/15 | 12 | coherent | 0.1719 |
| `COMP.MA.MXK5` | GLOBAL | NA | 0.9812 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.1230 |
| `COMP.MA.MXK6` | GLOBAL | NA | 0.9949 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.1426 |
| `COMP.MA.MXK7` | GLOBAL | NA | 0.9726 | -- | **dead** | 7/15 | 12 | coherent | 0.1348 |
| `COMP.MA.MXK8` | GLOBAL | NA | 0.9794 | -- | **dead** | 7/15 | 12 | coherent | 0.1348 |
| `COMP.MA.MXK9` | GLOBAL | NA | 0.9863 | -- | **impaired** | 7/15 | 12 | coherent | 0.1523 |
| `COMP.MA.MXK10` | GLOBAL | NA | 0.9726 | -- | **impaired** | 7/15 | 12 | coherent | 0.1328 |
| `COMP.MA.MXK11` | GLOBAL | NA | 0.9726 | -- | **impaired** | 6/15 | 12 | coherent | 0.0645 |
| `COMP.MA.MXK12` | GLOBAL | NA | 0.9863 | -- | **dead** | 4/15 | 12 | coherent | 0.0703 |
| `COMP.MA.MXK13` | GLOBAL | NA | 0.9797 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.1445 |
| `COMP.MA.MXK14` | GLOBAL | NA | 0.9943 | -- | **alive-parent-band** | 13/15 | 12 | coherent | 0.0742 |
| `COMP.MA.MXK15` | GLOBAL | NA | 0.9780 | -- | **dead** | 7/15 | 12 | coherent | 0.1055 |

**Survivors** (alive at a satisfied storage bound): 36 -- these are the assembly wave's input.

- `C05.T1.S4.L05` at L05: alive-parent-band, 10/15, stores 0.2500x
- `C05.T2.S4.L05` at L05: alive-parent-band, 10/15, stores 0.2500x
- `C05.T3.S4.L05` at L05: alive-parent-band, 12/15, stores 0.2500x
- `C05.T4.S4.L05` at L05: alive-parent-band, 12/15, stores 0.2500x
- `C05.T5.S4.L05` at L05: alive-parent-band, 10/15, stores 0.2493x
- `C05.T8.S4.L05` at L05: alive-parent-band, 12/15, stores 0.2493x
- `C06.T1.S4.L12` at L12: alive-parent-band, 10/15, stores 0.2500x
- `C06.T2.S4.L12` at L12: alive-parent-band, 10/15, stores 0.2500x
- `C06.T5.S4.L12` at L12: alive-parent-band, 10/15, stores 0.2490x
- `C06.T8.S4.L12` at L12: alive-parent-band, 10/15, stores 0.2500x
- `C07.T1.S4.L12` at L12: alive-parent-band, 11/15, stores 0.2490x
- `C07.T2.S4.L12` at L12: alive-parent-band, 11/15, stores 0.2490x
- `C07.T5.S4.L12` at L12: alive-parent-band, 12/15, stores 0.2490x
- `C07.T8.S4.L12` at L12: alive-parent-band, 10/15, stores 0.2490x
- `C08.T5.S4.L02` at L02: alive-degraded, 8/15, stores 0.2490x
- `C09.T1.S4.L02` at L02: alive-degraded, 8/15, stores 0.2500x
- `C09.T2.S4.L02` at L02: alive-degraded, 8/15, stores 0.2500x
- `C09.T3.S4.L02` at L02: alive-degraded, 9/15, stores 0.2500x
- `C09.T4.S4.L02` at L02: alive-degraded, 9/15, stores 0.2500x
- `C09.T5.S4.L02` at L02: alive-degraded, 9/15, stores 0.2493x
- `C10.T1.S2.L02` at L02: alive-degraded, 8/15, stores 0.5000x
- `C11.T1.S4.L00` at L00: alive-parent-band, 10/15, stores 0.2488x
- `C11.T1.S8.L00` at L00: alive-parent-band, 10/15, stores 0.1244x
- `C11.T2.S4.L00` at L00: alive-parent-band, 10/15, stores 0.2488x
- `C11.T2.S8.L00` at L00: alive-parent-band, 10/15, stores 0.1244x
- `C11.T5.S4.L00` at L00: alive-degraded, 8/15, stores 0.2494x
- `C11.T8.S4.L00` at L00: alive-parent-band, 11/15, stores 0.2488x
- `C11.T8.S8.L00` at L00: alive-degraded, 8/15, stores 0.1244x
- `C12.T1.S4.L00` at L00: alive-degraded, 8/15, stores 0.2488x
- `C12.T2.S4.L00` at L00: alive-degraded, 8/15, stores 0.2488x
- `C12.T8.S4.L00` at L00: alive-degraded, 9/15, stores 0.2488x
- `C13.T4.S4.L05` at L05: alive-parent-band, 10/15, stores 0.2488x
- `C13.T4.S8.L05` at L05: alive-degraded, 9/15, stores 0.1244x
- `C13.T5.S4.L05` at L05: alive-parent-band, 11/15, stores 0.2499x
- `C13.T5.S8.L05` at L05: alive-degraded, 9/15, stores 0.1247x
- `C13.T8.S4.L05` at L05: alive-parent-band, 12/15, stores 0.2499x

## 1. The matrix

Rows are components, columns are techniques.  A position shows the deepest level with a passing site **and the sites it passed at** (ADDENDUM 5 rule 2-S), or the shallowest level attempted and the chain link it broke at.  `.` is untested.

| component | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|
| C01 token_interface_in | . | . | . | . | X4:free | . | . | . | . |
| C03 conv_in_proj | S4:infra | S4:infra | S4:infra | . | M16:comp | . | . | S4:comp | . |
| C04 conv_kernel | M16:comp | . | . | . | . | . | . | . | . |
| C05 conv_out_proj | M16:comp | M16:comp | M16:comp | M16:solve | M16:comp | . | . | M16:comp | M8:comp |
| C06 attn_q | PASS@X4@L12 | S4:comp | S4:infra | . | PASS@M16@L12 | . | PASS@M8@L02 | S4:comp | . |
| C07 attn_k | PASS@X4@L12 | S4:comp | S4:infra | . | S4:solve | . | . | S4:comp | . |
| C08 attn_v | M16:free | S4:solve | M16:infra | . | M16:comp | . | . | S4:solve | . |
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
| C03 | T1 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C03 | T2 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C03 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C03 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C03 | T8 | . | . | . | . | . | . | 0/1 | . | fails at composition |
| C04 | T1 | 0/1 | . | . | 0/1 | . | . | . | . | fails at composition |
| C05 | T1 | 0/3 | 0/1 | . | 0/1 | 0/1 | . | 0/1 | . | fails at composition |
| C05 | T2 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C05 | T3 | 0/3 | 0/1 | 0/1 | 0/1 | 0/1 | . | 0/1 | . | fails at composition |
| C05 | T4 | 0/1 | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C05 | T5 | 0/3 | 0/1 | . | . | . | . | 0/1 | . | fails at composition |
| C05 | T8 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C05 | T9 | . | 0/1 | 0/1 | . | . | . | . | . | fails at composition |
| C06 | T1 | 3/3 L02,L08,L12 | 3/3 L02,L08,L12 | 0/1 | 0/1 | 1/3 L12 | . | 0/1 | . | works to X4 at L12 (1/3 sites); shallower elsewhere: L02:M8; L08:M8 |
| C06 | T2 | . | . | . | . | . | . | 0/1 | . | fails at composition |
| C06 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C06 | T5 | 1/2 L12 | . | . | . | . | . | 0/1 | . | works to M16 at L12 (1/2 sites) |
| C06 | T7 | 1/3 L02 | 1/1 L02 | 0/1 | . | . | . | . | . | works to M8 at L02 (1/3 sites) |
| C06 | T8 | . | . | . | . | . | . | 0/1 | . | fails at composition |
| C07 | T1 | 3/3 L02,L08,L12 | 2/3 L08,L12 | 0/1 | 0/1 | 1/2 L12 | . | 0/1 | . | works to X4 at L12 (1/3 sites); shallower elsewhere: L02:M16; L08:M8 |
| C07 | T2 | . | . | . | . | . | . | 0/1 | . | fails at composition |
| C07 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C07 | T5 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C07 | T8 | . | . | . | . | . | . | 0/1 | . | fails at composition |
| C08 | T1 | 0/1 | 0/1 | 0/1 | . | . | . | 0/1 | . | fails at composition |
| C08 | T2 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C08 | T3 | 0/1 | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C08 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C08 | T8 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C09 | T1 | 0/2 | 0/1 | 0/1 | . | . | . | 0/1 | . | fails at composition |
| C09 | T2 | 0/1 | . | . | . | . | . | 0/1 | . | fails at free-run |
| C09 | T3 | 0/1 | 0/1 | . | . | 0/1 | . | 0/1 | . | fails at composition |
| C09 | T4 | 0/1 | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C09 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C09 | T8 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C10 | T1 | 0/1 | 0/1 | . | . | . | 0/1 | 0/1 | . | fails at composition |
| C11 | T1 | . | . | . | . | . | . | 0/1 | 0/1 | fails at local-solve |
| C11 | T2 | . | . | . | . | . | . | 0/1 | 0/1 | fails at local-solve |
| C11 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C11 | T4 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C11 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C11 | T8 | . | . | . | . | . | . | 0/1 | 0/1 | fails at local-solve |
| C12 | T1 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C12 | T2 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C12 | T3 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C12 | T4 | . | . | . | . | . | . | 0/1 | . | void (infra) -- no evidence |
| C12 | T5 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C12 | T8 | . | . | . | . | . | . | 0/1 | . | fails at local-solve |
| C13 | T1 | 0/3 | . | . | . | . | . | 0/1 | . | fails at composition |
| C13 | T2 | 0/1 | . | . | . | . | . | 0/1 | . | fails at composition |
| C13 | T3 | 0/2 | 0/1 | 0/1 | . | . | . | 0/1 | . | fails at composition |
| C13 | T4 | . | . | . | . | . | . | 0/1 | 0/1 | fails at local-solve |
| C13 | T5 | 0/3 | . | . | . | . | . | 0/1 | 0/1 | fails at composition |
| C13 | T8 | 0/1 | . | . | . | . | . | 0/1 | 0/1 | fails at local-solve |
| C14 | T1 | . | . | . | . | . | . | . | . | works at NA (no level axis) at L00 (1/1 sites) |
| COMP | composition | . | . | . | . | . | . | . | . | fails at local-solve |

## 3. Failure classes

The chain is walked mechanically (§5 rule 5); this histogram is the campaign's failure analysis in full.  `free-run` is the F3 signature -- a cell that cleared every numeric link and still lost behavior -- and is the most information-dense class in the matrix.

| first divergent metric | cells | share |
|---|---:|---:|
| `infra` | 13 | 8% |
| `local-solve` | 67 | 42% |
| `macrocell-band` | 1 | 1% |
| `composition` | 67 | 42% |
| `free-run` | 12 | 8% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 18 | 140 |
| out of band | 0 | 3 |

Agreement 21/161 (13%).

## 4b. Storage honesty

A level names a *rank*, and a rank is not a size.  A technique whose stored form is a `w x r` basis plus an `r x w` solved factor stores `r * (in + out)` against a dense `in * out`, so on a square component it does not contract at all until `r < w/2`.  Cells above 1.0 below cost more storage than the tensor they replaced.

| cell | technique | level | stored | dense parent | stored/dense |
|---|---|---|---:|---:|---:|
| `C03.T5.M16.L00` | T5 | M16 | 2949632 | 3145728 | 0.9377 |
| `C03.T8.S4.L00` | T8 | S4 | 786432 | 3145728 | 0.2500 |
| `C04.T1.M16.L00` | T1 | M16 | 2048 | 3072 | 0.6667 |
| `C04.T1.M2.L00` | T1 | M2 | 1024 | 3072 | 0.3333 |
| `C05.T1.M8.L05` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T1.M2.L05` | T1 | M2 | 1048576 | 1048576 | 1.0000 |
| `C05.T1.X4.L05` | T1 | X4 | 1441792 | 1048576 | 1.3750 |
| `C05.T1.S4.L05` | T1 | S4 | 262144 | 1048576 | 0.2500 |
| `C05.T2.M16.L05` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T2.S4.L05` | T2 | S4 | 262144 | 1048576 | 0.2500 |
| `C05.T3.M16.L00` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L05` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M16.L13` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C05.T3.M8.L05` | T3 | M8 | 1835008 | 1048576 | 1.7500 |
| `C05.T3.M4.L05` | T3 | M4 | 1572864 | 1048576 | 1.5000 |
| `C05.T3.M2.L05` | T3 | M2 | 1048576 | 1048576 | 1.0000 |
| `C05.T3.X4.L05` | T3 | X4 | 1441792 | 1048576 | 1.3750 |
| `C05.T3.S4.L05` | T3 | S4 | 262144 | 1048576 | 0.2500 |
| `C05.T4.S4.L05` | T4 | S4 | 262144 | 1048576 | 0.2500 |
| `C05.T5.M8.L05` | T5 | M8 | 917504 | 1048576 | 0.8750 |
| `C05.T5.S4.L05` | T5 | S4 | 261375 | 1048576 | 0.2493 |
| `C05.T8.S4.L05` | T8 | S4 | 261375 | 1048576 | 0.2493 |
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
| `C06.T1.S4.L12` | T1 | S4 | 262144 | 1048576 | 0.2500 |
| `C06.T2.S4.L12` | T2 | S4 | 262144 | 1048576 | 0.2500 |
| `C06.T5.M16.L08` | T5 | M16 | 983552 | 1048576 | 0.9380 |
| `C06.T5.M16.L12` | T5 | M16 | 983552 | 1048576 | 0.9380 |
| `C06.T5.S4.L12` | T5 | S4 | 261103 | 1048576 | 0.2490 |
| `C06.T7.M16.L12` | T7 | M16 | 1044480 | 1048576 | 0.9961 |
| `C06.T7.M8.L02` | T7 | M8 | 974848 | 1048576 | 0.9297 |
| `C06.T7.M4.L02` | T7 | M4 | 835584 | 1048576 | 0.7969 |
| `C06.T8.S4.L12` | T8 | S4 | 262144 | 1048576 | 0.2500 |
| `C07.T1.M16.L08` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M16.L12` | T1 | M16 | 737280 | 524288 | 1.4062 |
| `C07.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M8.L08` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M8.L12` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C07.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C07.T1.M2.L02` | T1 | M2 | 393216 | 524288 | 0.7500 |
| `C07.T1.X4.L08` | T1 | X4 | 589824 | 524288 | 1.1250 |
| `C07.T1.X4.L12` | T1 | X4 | 589824 | 524288 | 1.1250 |
| `C07.T1.S4.L12` | T1 | S4 | 130560 | 524288 | 0.2490 |
| `C07.T2.S4.L12` | T2 | S4 | 130560 | 524288 | 0.2490 |
| `C07.T5.S4.L12` | T5 | S4 | 130552 | 524288 | 0.2490 |
| `C07.T8.S4.L12` | T8 | S4 | 130560 | 524288 | 0.2490 |
| `C08.T1.M8.L02` | T1 | M8 | 688128 | 524288 | 1.3125 |
| `C08.T1.M4.L02` | T1 | M4 | 589824 | 524288 | 1.1250 |
| `C08.T1.S4.L02` | T1 | S4 | 130560 | 524288 | 0.2490 |
| `C08.T2.S4.L02` | T2 | S4 | 130560 | 524288 | 0.2490 |
| `C08.T5.S4.L02` | T5 | S4 | 130552 | 524288 | 0.2490 |
| `C08.T8.S4.L02` | T8 | S4 | 130560 | 524288 | 0.2490 |
| `C09.T1.M16.L08` | T1 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T1.M8.L02` | T1 | M8 | 1835008 | 1048576 | 1.7500 |
| `C09.T1.M4.L02` | T1 | M4 | 1572864 | 1048576 | 1.5000 |
| `C09.T1.S4.L02` | T1 | S4 | 262144 | 1048576 | 0.2500 |
| `C09.T2.M16.L02` | T2 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T2.S4.L02` | T2 | S4 | 262144 | 1048576 | 0.2500 |
| `C09.T3.M16.L02` | T3 | M16 | 1966080 | 1048576 | 1.8750 |
| `C09.T3.M8.L02` | T3 | M8 | 1835008 | 1048576 | 1.7500 |
| `C09.T3.X4.L02` | T3 | X4 | 1572864 | 1048576 | 1.5000 |
| `C09.T3.S4.L02` | T3 | S4 | 262144 | 1048576 | 0.2500 |
| `C09.T4.S4.L02` | T4 | S4 | 262144 | 1048576 | 0.2500 |
| `C09.T5.S4.L02` | T5 | S4 | 261375 | 1048576 | 0.2493 |
| `C09.T8.S4.L02` | T8 | S4 | 261375 | 1048576 | 0.2493 |
| `C10.T1.M16.L02` | T1 | M16 | 3014656 | 3145728 | 0.9583 |
| `C10.T1.M8.L02` | T1 | M8 | 2883584 | 3145728 | 0.9167 |
| `C10.T1.S2.L02` | T1 | S2 | 1572864 | 3145728 | 0.5000 |
| `C11.T1.S4.L00` | T1 | S4 | 652288 | 2621440 | 0.2488 |
| `C11.T1.S8.L00` | T1 | S8 | 326144 | 2621440 | 0.1244 |
| `C11.T2.S4.L00` | T2 | S4 | 652288 | 2621440 | 0.2488 |
| `C11.T2.S8.L00` | T2 | S8 | 326144 | 2621440 | 0.1244 |
| `C11.T5.M16.L00` | T5 | M16 | 2460321 | 2621440 | 0.9385 |
| `C11.T5.S4.L00` | T5 | S4 | 653763 | 2621440 | 0.2494 |
| `C11.T8.S4.L00` | T8 | S4 | 652288 | 2621440 | 0.2488 |
| `C11.T8.S8.L00` | T8 | S8 | 326144 | 2621440 | 0.1244 |
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
| `C13.T4.S8.L05` | T4 | S8 | 326144 | 2621440 | 0.1244 |
| `C13.T5.M16.L05` | T5 | M16 | 2457600 | 2621440 | 0.9375 |
| `C13.T5.M16.L13` | T5 | M16 | 2460000 | 2621440 | 0.9384 |
| `C13.T5.S4.L05` | T5 | S4 | 654975 | 2621440 | 0.2499 |
| `C13.T5.S8.L05` | T5 | S8 | 326975 | 2621440 | 0.1247 |
| `C13.T8.S4.L05` | T8 | S4 | 654975 | 2621440 | 0.2499 |
| `C13.T8.S8.L05` | T8 | S8 | 326975 | 2621440 | 0.1247 |
| `C14.T1.NA.L00` | T1 | NA | 1024 | 2048 | 0.5000 |
| `COMP.MA.RBEH` | composition | NA | 231693056 | 229693184 | 1.0087 |
| `COMP.MA.RSTO` | composition | NA | 229235456 | 229693184 | 0.9980 |
| `COMP.MB.RSTO` | composition | NA | 229235456 | 229693184 | 0.9980 |
| `COMP.MB.RBEH` | composition | NA | 231693056 | 229693184 | 1.0087 |
| `COMP.MA.MXJ1` | composition | NA | 227726719 | 229693184 | 0.9914 |
| `COMP.MA.MXJ2` | composition | NA | 227332983 | 229693184 | 0.9897 |
| `COMP.MA.MXJ3` | composition | NA | 226545782 | 229693184 | 0.9863 |
| `COMP.MA.MXJ4` | composition | NA | 224576630 | 229693184 | 0.9777 |
| `COMP.MA.MXJ5` | composition | NA | 223790198 | 229693184 | 0.9743 |
| `COMP.MA.MXJ3A` | composition | NA | 226546551 | 229693184 | 0.9863 |
| `COMP.MA.MXJ3B` | composition | NA | 226546551 | 229693184 | 0.9863 |
| `COMP.MA.MXJ3C` | composition | NA | 225363831 | 229693184 | 0.9812 |
| `COMP.MA.MXJ4C` | composition | NA | 224577399 | 229693184 | 0.9777 |
| `COMP.MB.MXJ3` | composition | NA | 226545782 | 229693184 | 0.9863 |
| `COMP.MA.MXJP2A` | composition | NA | 225366518 | 229693184 | 0.9812 |
| `COMP.MA.MXJP2B` | composition | NA | 225366518 | 229693184 | 0.9812 |
| `COMP.MA.MXJP2C` | composition | NA | 223400053 | 229693184 | 0.9726 |
| `COMP.MA.MXK1` | composition | NA | 225366518 | 229693184 | 0.9812 |
| `COMP.MA.MXK2` | composition | NA | 225366518 | 229693184 | 0.9812 |
| `COMP.MA.MXK3` | composition | NA | 225366518 | 229693184 | 0.9812 |
| `COMP.MA.MXK4` | composition | NA | 226939247 | 229693184 | 0.9880 |
| `COMP.MA.MXK5` | composition | NA | 225363831 | 229693184 | 0.9812 |
| `COMP.MA.MXK6` | composition | NA | 228512247 | 229693184 | 0.9949 |
| `COMP.MA.MXK7` | composition | NA | 223394679 | 229693184 | 0.9726 |
| `COMP.MA.MXK8` | composition | NA | 224970095 | 229693184 | 0.9794 |
| `COMP.MA.MXK9` | composition | NA | 226543095 | 229693184 | 0.9863 |
| `COMP.MA.MXK10` | composition | NA | 223394679 | 229693184 | 0.9726 |
| `COMP.MA.MXK11` | composition | NA | 223397366 | 229693184 | 0.9726 |
| `COMP.MA.MXK12` | composition | NA | 226545511 | 229693184 | 0.9863 |
| `COMP.MA.MXK13` | composition | NA | 225037687 | 229693184 | 0.9797 |
| `COMP.MA.MXK14` | composition | NA | 228381047 | 229693184 | 0.9943 |
| `COMP.MA.MXK15` | composition | NA | 224644232 | 229693184 | 0.9780 |

93 of 142 sealed cells with a storage reading actually contract.

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

### `C03.T1.S4.L00` -- INFR

conv_in_proj / dense_prior_anchored_ridge / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f98f6e9d` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 14.3 | `ad35fa831c10` |

> RuntimeError: mat1 and mat2 shapes cannot be multiplied (192x1024 and 3072x1024)

### `C03.T2.S4.L00` -- INFR

conv_in_proj / cross_width_residual_charts / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5787fe87` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.5 | `02ad9255584d` |

> CellTechniqueError: C03 charts endpoint 'residual.post_layer' at width 1024, but its level contracts 3072.

### `C03.T3.S4.L00` -- INFR

conv_in_proj / balanced_basis_truncation / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6939ad56` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.6 | `d252c7a48bee` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C03 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C03.T5.M16.L00` -- FAIL

conv_in_proj / sparse_omp_routes / level M16 / site L00 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fbc908bf` | sm86 | fail | composition | -- | -- | 0.0276 | 0.787 | 0.684 | 0.938 | 15.8 | 29.2 | `afb675cc1809` |
| `fbc908bf` | sm86 | fail | composition | -- | -- | 0.0276 | 0.787 | 0.684 | 0.938 | 15.8 | 29.2 | `cbd74797fe81` |
| `5623aca1` | -- | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 35.6 | `2907d224b20f` |

> decode-smoke review by ClaudeCode-MXF of receipt cbd74797fe81e3f2: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C03.T5.S4.L00` -- INFR

conv_in_proj / sparse_omp_routes / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a0a6ae2b` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.1 | `741ac03f5753` |

> CellTechniqueError: C03 stores 3 projection blocks over one internal channel axis; level width 728 is not divisible by 3.

### `C03.T8.S4.L00` -- FAIL

conv_in_proj / gram_relation_matching / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `8ab0485e` | sm86 | fail | composition | -- | -- | 1.5783 | 0.008 | 0.000 | 0.250 | 4503599627370496.0 | 43.2 | `d7d15e4e4e14` |
| `8ab0485e` | sm86 | fail | composition | -- | -- | 1.5783 | 0.008 | 0.000 | 0.250 | 4503599627370496.0 | 43.2 | `6a714002a830` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 6a714002a830f414: verdict dead, 0/15 useful-or-partial (parent 12/15). Total collapse, and the harness caught it before the reviewer did: emit_zero_floor_passed is FALSE in the receipt, which is an independent dead trigger under ADDENDUM 6 6.1, and no decode-smoke verdict was sealed at all. All sixteen responses are censored-at-cap at 1024 tokens because the model never emits EOS, and all sixteen are the SAME repeating string -- ' seizo, alike ElementaryaGolden namelyーター seizo e<arabic> relateda inexp theaume the' -- cycled to the cap, with unrelated-script drift into Arabic and katakana. That is three of the protocol's named failure modes at once: phrase cycling, tokenizer debris and script drift. 0 of 15. C03 is the only component in wave 5B where a technique both satisfied the storage bound (786,432 of 3,145,728 = exactly 0.25x) and produced a reviewable child, and the child is dead. The other four C03 techniques never got that far: the generic contraction resolves conv_in_proj's output width as 1024 when the tensor is 3072x1024 -- three stacked B/C/x projection blocks over one internal channel axis -- and fails with '
> ... [173 more characters in the receipt]

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

### `C05.T1.S4.L05` -- FAIL

conv_out_proj / dense_prior_anchored_ridge / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b94e13ec` | sm86 | fail | local-solve | 0.8303 | 0.1769 | 0.2007 | 0.137 | 0.107 | 0.250 | 188.1 | 30.5 | `4e0e0bbe7684` |
| `b94e13ec` | sm86 | fail | local-solve | 0.8303 | 0.1769 | 0.2007 | 0.137 | 0.107 | 0.250 | 188.1 | 30.5 | `e07d874cb222` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt e07d874cb2223be3: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). conv_out_proj under dense prior-anchored ridge at a TRUE 4x -- 262,144 stored against 1,048,576 parent, exactly 0.25x, rank 128 of 1024 -- scores 10 of 15, inside the parent band. The short-conv output projection folds at a quarter. This is the widest-margin partial profile in the wave: nine of fifteen cases are Partial, only one is Useful, and nothing collapses. The model is uniformly slightly worse rather than selectively broken, which is the opposite of what the MLP cells do. Ships the anchor exact off-by-one fix values[start:], correct deduplication with an example, and a serviceable Eisenhower-matrix answer to the three-actions case (with an introduction the anchor forbids, and wash dishes in the morning 2-3 hours). Loses explanation-seasons completely -- no tilt anywhere, and the reason why Earth has seasons is that it is closer to the Sun -- refuses the insufficient-data case instead of drawing the safe conclusion, and drops the rollback step from the migration plan in favour of a second backup step.

### `C05.T2.M16.L05` -- FAIL

conv_out_proj / cross_width_residual_charts / level M16 / site L05 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `99979d513ff3` |
| `0d42a7e0` | sm86 | fail | composition | 0.9968 | 0.0239 | 0.0387 | 0.744 | 0.723 | 1.875 | 1.0 | 30.4 | `6e59fd005767` |

> decode-smoke review by ClaudeCode-MXF of receipt 6e59fd005767d38a: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C05.T2.S4.L05` -- FAIL

conv_out_proj / cross_width_residual_charts / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `8a54c09b` | sm86 | fail | local-solve | 0.8303 | 0.1769 | 0.2007 | 0.137 | 0.107 | 0.250 | 1.0 | 28.4 | `5d23863768d6` |
| `8a54c09b` | sm86 | fail | local-solve | 0.8303 | 0.1769 | 0.2007 | 0.137 | 0.107 | 0.250 | 1.0 | 28.4 | `7a60c571941a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 7a60c571941a56c5: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). Graded by BYTE-IDENTITY. C05.T2.S4.L05 sixteen child responses digest to 266657080178eaf7, exactly C05.T1.S4.L05 digest: cross-width residual charts and dense prior-anchored ridge produced the same model and the same text on conv_out_proj at L05. Gradings carried across unchanged, equivalence recorded in both receipts. That makes three components where T1 and T2 are byte-identical at S4 (C05, C06, C13) against one where they are not (C11) and one where they differ in two of sixteen responses (C12). The wave checked it per cell and never assumed it.

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

### `C05.T3.S4.L05` -- FAIL

conv_out_proj / balanced_basis_truncation / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `142d9708` | sm86 | fail | local-solve | 0.8422 | 0.1722 | 0.1846 | 0.215 | 0.164 | 0.250 | 302.6 | 32.5 | `0a2441d1a0ab` |
| `142d9708` | sm86 | fail | local-solve | 0.8422 | 0.1722 | 0.1846 | 0.215 | 0.164 | 0.250 | 302.6 | 32.5 | `5aa419fbde3d` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 5aa419fbde3dee7d: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). conv_out_proj under balanced basis truncation at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 12 of 15, LEVEL WITH THE PARENT. That is the same technique that scored 7 of 15 (impaired) on mlp_down at the same bound in this wave, which is the sharpest technique-by-component contrast the campaign has: T3 is not a good or a bad solver, it is a solver that suits some components and destroys others. Ships the anchor exact off-by-one fix values[start:], correct deduplication with an example, a calibrated insufficient-data answer, and a creative-continuity that ends on a foghorn whose answer is a question lingering like a ghost. explanation-seasons is Partial rather than a miss because it names the axial tilt twice and calls the distance idea not entirely accurate, even while framing Earth as closer to the Sun than the other planets. The migration plan omits both the DNS cutover and the rollback step, which is why it is Partial.

### `C05.T4.M16.L13` -- FAIL

conv_out_proj / low_rank_factor_writers / level M16 / site L13 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bc7eb877` | -- | fail | local-solve | 0.8027 | 0.1121 | 0.1979 | 0.225 | 0.170 | -- | 7068411.0 | 25.9 | `d2872dfefd2a` |
| `bc7eb877` | -- | fail | local-solve | 0.8027 | 0.1121 | 0.1979 | 0.225 | 0.170 | -- | 7068411.0 | 25.9 | `5ecf8394acb4` |

> decode-smoke review by claude-opus-5 of receipt 5ecf8394acb44c30: 3/3 on task and visibly degraded, which is what writer_rank 31 on a 1024x1024 writer buys. (1) circuit breaker: 'a safety valve that stops a circuit from running when it's too hot or too low' -- 'too low' is wrong and the thermal framing replaces the current one, but the trip-to-protect core survives. (2) one-pan dinner: a real plan, cap-censored, with a hedged ingredient list ('fresh or frozen' five times). (3) neighbour note: reshaped into a subject-lined email that asks to 'help with parking' rather than to move the trailer -- the request is softened past the instruction. The cell already fails at chain link 3 (write cosine 0.803 < 0.95); the text degrades without collapsing, unlike C09.T4 at the same rank.

### `C05.T4.S4.L05` -- FAIL

conv_out_proj / low_rank_factor_writers / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `efba0ace` | sm86 | fail | local-solve | 0.8818 | 0.1489 | 0.1441 | 0.209 | 0.189 | 0.250 | 90103.2 | 28.0 | `17dff6d214de` |
| `efba0ace` | sm86 | fail | local-solve | 0.8818 | 0.1489 | 0.1441 | 0.209 | 0.189 | 0.250 | 90103.2 | 28.0 | `94e9d4e17c64` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 94e9d4e17c649a98: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). conv_out_proj under low-rank factor writers at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 12 of 15, LEVEL WITH THE PARENT. It is the only cell in the entire wave that computes 94 mod 5 = 4 -- the anchor own leftover, which the parent never reaches -- and then throws it away by reporting 4 as the per-box count as well. Ships values[start:], correct deduplication, a calibrated insufficient-data answer and a creative-continuity with real dialogue and an unexplained ending. Loses explanation-seasons (no tilt at all, seasons attributed to Earth being closer to the Sun than the other planets), loses logic-ordering to the order Sal, Jun, Theo, Mira, and its migration plan never actually points DNS at the new domain and has no rollback step.

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

### `C05.T5.S4.L05` -- FAIL

conv_out_proj / sparse_omp_routes / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `25c55a63` | sm86 | fail | local-solve | 0.8559 | 0.1623 | 0.1781 | 0.160 | 0.121 | 0.249 | 157.4 | 28.6 | `df1ffa5121dd` |
| `25c55a63` | sm86 | fail | local-solve | 0.8559 | 0.1623 | 0.1781 | 0.160 | 0.121 | 0.249 | 157.4 | 28.6 | `b22bb64b882d` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt b22bb64b882df3f0: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). conv_out_proj under sparse OMP routes at a TRUE 4x (261,375 of 1,048,576 = 0.249267x, 255 kept channels of 1024) scores 10 of 15, inside the parent band and the weakest of the five conv_out_proj techniques. WITH THIS CELL EVERY TECHNIQUE THAT CAN EXPRESS conv_out_proj AT S4 IS ALIVE-PARENT-BAND: T1 10, T2 10, T3 12, T4 12, T5 10, T8 pending. Ships correct deduplication and the wave clearest calibrated insufficient-data answer, which states Sales increased as a numbered conclusion before asking for the four missing facts. Loses explanation-seasons by comparing Earth to the other planets instead of to itself across the year, loses arithmetic to 18.8 kits per box and 76 left over, and produces a creative-continuity that is two sentences of atmosphere with no dialogue and no ending.

### `C05.T8.M16.L00` -- FAIL

conv_out_proj / gram_relation_matching / level M16 / site L00 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f8038b65` | -- | fail | composition | 0.9987 | 0.0411 | 0.0543 | 0.523 | 0.342 | -- | 48638.1 | 34.2 | `f1dcd461137d` |
| `f8038b65` | -- | fail | composition | 0.9987 | 0.0411 | 0.0543 | 0.523 | 0.342 | -- | 48638.1 | 34.2 | `b95d24ebc6f1` |

> decode-smoke review by claude-opus-5 of receipt b95d24ebc6f16c99: 3/3 on task, all EOS, and the weakest smoke of the three C05 techniques at L00. (1) circuit breaker: correct, but 'typically installed in high-voltage circuits, such as those used in power grids' is a false narrowing -- breakers are ubiquitous at household voltage. (2) dinner: a five-item menu that opens with 'Start with a hearty breakfast: omelette or pancakes' for a dinner prompt, then asserts '40 minutes and one pan' without a cooking order -- on task, both stated constraints missed, and the breakfast opener is the first content error of its kind in the campaign. (3) neighbour note: three sentences rather than four, and it offers to assist rather than asking. Fails at chain link 5, composed drift 0.0543; free-run 0.523 is the worst of the three techniques at this site.

### `C05.T8.S4.L05` -- FAIL

conv_out_proj / gram_relation_matching / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `60804128` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.1812 | 0.156 | 0.098 | 0.249 | 1199.0 | 29.7 | `ca4d1407d3fe` |
| `60804128` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.1812 | 0.156 | 0.098 | 0.249 | 1199.0 | 29.7 | `d212254849de` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt d212254849def90c: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). conv_out_proj under gram relation matching at a TRUE 4x (261,375 of 1,048,576 = 0.249267x) scores 12 of 15, LEVEL WITH THE PARENT. THIS CELL COMPLETES THE STRONGEST COMPONENT RESULT OF THE CAMPAIGN: every one of the six techniques that can express conv_out_proj at S4 is alive-parent-band, and three of them (T3, T4, T8) score 12 of 15, dead level with the parent. T1 10, T2 10, T3 12, T4 12, T5 10, T8 12. conv_out_proj is the component the assembly wave should take first. Ships values[start:], a correct concise email, a calibrated insufficient-data answer that states Sales increased under its own What you can safely conclude heading, and a creative-continuity with real dialogue and an unexplained ending. Loses arithmetic to 94 - 18.8 = 95.2, loses logic-ordering by numbering the premises and boxing 1, 2, 3, 4 instead of naming an order, and its rollback step describes removing the old domain rather than reverting to it.

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

### `C06.T1.S4.L12` -- FAIL

attn_q / dense_prior_anchored_ridge / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2a73e3ca` | sm86 | fail | composition | 0.9856 | 0.1901 | 0.0823 | 0.236 | 0.221 | 0.250 | 27233.8 | 27.9 | `341f23c2fcbf` |
| `2a73e3ca` | sm86 | fail | composition | 0.9856 | 0.1901 | 0.0823 | 0.236 | 0.221 | 0.250 | 27233.8 | 27.9 | `28e8569cc80c` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 28e8569cc80cef7f: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). THE ATTENTION RESULT. attn_q under dense prior-anchored ridge at a TRUE 4x -- 262,144 stored against 1,048,576 parent, exactly 0.25x, rank 128 of 1024 -- scores 10 of 15 and lands inside the parent's band. The query projection folds at a quarter, which is the first attention component in the campaign to do so under a storage-defined level. Ships the anchor's exact off-by-one fix, values[start:], which only two lanes in the whole wave manage, plus correct deduplication with an example, a complete six-step migration plan and the best creative-continuity of the attention lanes -- dialogue, unease, and an ending that explains nothing. Loses explanation-seasons in a specific and instructive way: it states the tilt correctly in its third paragraph and then closes by crediting 'Earth's elliptical orbit and the changing distance from the Sun', which is the misconception the case exists to reject, so the response is judged as a whole and scored a miss. Also inverts causality the way the M16 MLP cells do ('This variation in sunlight causes the Eart
> ... [173 more characters in the receipt]

### `C06.T2.S4.L12` -- FAIL

attn_q / cross_width_residual_charts / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1cf2c017` | sm86 | fail | composition | 0.9856 | 0.1901 | 0.0823 | 0.236 | 0.221 | 0.250 | 1.0 | 27.4 | `48a491494b89` |
| `1cf2c017` | sm86 | fail | composition | 0.9856 | 0.1901 | 0.0823 | 0.236 | 0.221 | 0.250 | 1.0 | 27.4 | `5c6f44c4bba6` |
| `1cf2c017` | sm86 | fail | composition | 0.9856 | 0.1901 | 0.0823 | 0.236 | 0.221 | 0.250 | 1.0 | 27.4 | `50b51294f773` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 50b51294f773c26b: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). Graded by BYTE-IDENTITY. C06.T2.S4.L12's sixteen child responses digest to e74f72f4c26a30d3, which is exactly C06.T1.S4.L12's digest: cross-width residual charts and dense prior-anchored ridge produced the SAME model and the same text on attn_q at L12. The gradings and counts are carried across unchanged and the equivalence is recorded in both receipts. Across wave 5B the T1/T2 relationship was checked per cell rather than assumed, and it varies: identical at C13.L05 and here at C06.L12, fourteen-of-sixteen identical at C12.L00, and completely different text at C11.L00 despite solve telemetry that matched to the digit. attn_q therefore folds at a true 4x under two techniques, and they are the same fold.

### `C06.T3.S4.L12` -- INFR

attn_q / balanced_basis_truncation / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `261b549f` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.8 | `832a4f4dbd0a` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C06 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

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

### `C06.T5.S4.L12` -- FAIL

attn_q / sparse_omp_routes / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `c1aa8b27` | sm86 | fail | composition | 0.9512 | 0.3468 | 0.1411 | 0.361 | 0.318 | 0.249 | 17.3 | 29.9 | `b3009db9ea83` |
| `c1aa8b27` | sm86 | fail | composition | 0.9512 | 0.3468 | 0.1411 | 0.361 | 0.318 | 0.249 | 17.3 | 29.9 | `5a8a11d726ab` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 5a8a11d726abb501: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). attn_q under sparse OMP routes at a TRUE 4x (261,103 of 1,048,576 = 0.249007x, 242 kept channels of 1024) scores 10 of 15, inside the parent's band -- the same score its T1/T2 counterpart gets, by a visibly different route. It is the only attention lane to answer context-retrieval completely and one of two lanes in the wave to ship a working off-by-one fix. THIS CELL CARRIES THE WAVE'S STRONGEST SUB-DEGENERATE REPETITION SIGNAL AND IS DELIBERATELY NOT SEALED DEAD, so the call is on the record and reversible. Immediate verbatim doubling appears in at least six of sixteen responses: 'bell peppers, bell peppers, bell peppers, bell peppers, bell peppers' inside each of four ingredient lists (about twenty-two occurrences), 'a large hard drive or hard drive' four times, 'hum' five times in three sentences of creative-continuity, plus 'Backup Backup', 'redundancy and redundancy', 'safety and safety', 'guidelines and guidelines', 'nuts, nuts, nuts' and 'gluten-free, gluten-free'. It is NOT sealed as degeneracy because the standard applied uniform
> ... [490 more characters in the receipt]

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

### `C06.T8.S4.L12` -- FAIL

attn_q / gram_relation_matching / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `368e5463` | sm86 | fail | composition | 0.9856 | 0.1905 | 0.0798 | 0.236 | 0.221 | 0.250 | 26696.0 | 28.9 | `489a6e3fbaf5` |
| `368e5463` | sm86 | fail | composition | 0.9856 | 0.1905 | 0.0798 | 0.236 | 0.221 | 0.250 | 26696.0 | 28.9 | `a9a331c86964` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt a9a331c869644980: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). attn_q under gram relation matching at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 10 of 15, inside the parent's band. WITH THIS CELL ALL FOUR TECHNIQUES THAT CAN EXPRESS attn_q AT S4 LAND ON 10/15: T1 and T2 (byte-identical to each other), T5 by a different route, and T8 here. attn_q is the campaign's first attention component to fold at a real quarter, and it does so under every technique that can state it that small -- which is a much stronger claim than any single cell. Ships the anchor's exact off-by-one fix values[start:], correct deduplication, a complete email, and a creative-continuity that ends on 'The fog seemed to be holding its breath, waiting for something to happen.' Loses explanation-seasons by rejecting the distance misconception in one sentence and asserting it in three, and answers arithmetic by deriving 18 per box and then dividing it by 17. Safety is concerning: seal the cracks, seal the windows, seal the doors, and 'cauldron caulk'.

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

### `C07.T1.S4.L12` -- FAIL

attn_k / dense_prior_anchored_ridge / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f12b5787` | sm86 | fail | composition | 0.9774 | 0.0000 | 0.1538 | 0.414 | 0.363 | 0.249 | 27233.8 | 27.7 | `94e861043c9c` |
| `f12b5787` | sm86 | fail | composition | 0.9774 | 0.0000 | 0.1538 | 0.414 | 0.363 | 0.249 | 27233.8 | 27.7 | `96ab2acfdde2` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 96ab2acfdde2314a: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). attn_k under dense prior-anchored ridge at a TRUE 4x -- 130,560 stored against 524,288 parent, 0.249023x, rank 85 of 512 -- scores 11 of 15, ONE CASE from the parent's own 12/15 and the best attention result of the wave. The key projection is the most compressible attention component measured so far, which is consistent with it being the narrowest (512 wide against attn_q's 1024) and the one whose output is consumed only through a softmax. Ships values[start:], the anchor's exact off-by-one fix; carries 102, 94 and a correct 18-per-box without contradicting itself; writes a six-step migration plan that actually sequences backup, DNS, TLS, verification and rollback. Loses explanation-seasons to a fully geocentric account -- 'as the sun moves around the Earth', seasons caused by 'atmospheric circulation', no axial tilt anywhere -- and loses comparison-ram-rom to two 'imperfections' that restate each other and conclude 'the library is about the books, and the family is about the computers'. Safety is concerning and specifically so: it tells 
> ... [53 more characters in the receipt]

### `C07.T2.S4.L12` -- FAIL

attn_k / cross_width_residual_charts / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `78ec6856` | sm86 | fail | composition | 0.9774 | 0.0000 | 0.1537 | 0.414 | 0.363 | 0.249 | 1.0 | 28.4 | `907adaab3fab` |
| `78ec6856` | sm86 | fail | composition | 0.9774 | 0.0000 | 0.1537 | 0.414 | 0.363 | 0.249 | 1.0 | 28.4 | `ccb3b4ac5481` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt ccb3b4ac5481d20d: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). attn_k under cross-width residual charts at a TRUE 4x scores 11 of 15, one case from the parent, matching C07.T1.S4.L12 exactly. FOURTEEN of sixteen responses are byte-identical to that cell; only uncertainty-insufficient-data and creative-continuity differ, and each is graded the same in both. Same geocentric explanation-seasons failure (as the sun moves around the Earth, seasons caused by atmospheric circulation), same working values[start:] off-by-one fix, same six-step migration plan.

### `C07.T3.S4.L12` -- INFR

attn_k / balanced_basis_truncation / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d11bc126` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.4 | `57086a8accc5` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C07 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C07.T5.S4.L12` -- FAIL

attn_k / sparse_omp_routes / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9736d44b` | sm86 | fail | local-solve | 0.9260 | 0.0000 | 0.2011 | 0.236 | 0.162 | 0.249 | 7.8 | 28.0 | `afe49d73e0bd` |
| `9736d44b` | sm86 | fail | local-solve | 0.9260 | 0.0000 | 0.2011 | 0.236 | 0.162 | 0.249 | 7.8 | 28.0 | `adbdf3941940` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt adbdf39419407001: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). THE SECOND CELL IN THE WAVE TO SCORE LEVEL WITH THE PARENT. attn_k under sparse OMP routes at a TRUE 4x -- 130,552 stored against 524,288 parent, 0.249008x, 121 kept channels of 512 -- scores 12 of 15, DEAD LEVEL WITH THE PARENT OWN 12/15, at a token match of 0.2363. Only three cases miss, and two of them (logic-ordering and the peanut plan) are cases the PARENT also misses. Answers context-retrieval completely, states the calibrated insufficient-data conclusion, writes correct deduplication with an example and the best creative-continuity of the attention lanes -- a keeper asking Still here? twice into a fog that answers with a click. explanation-seasons is the best in the wave outside C13.T8.S4.L05: it names the 23-degree axial tilt AND the changing angle of the Sun rays, and only fails to address the distance misconception at all. Together with C07.T1/T2 at 11/15 and C07.T8 at 10/15, attn_k folds at a real quarter under every technique that can express it, and is the strongest attention component measured.

### `C07.T8.S4.L12` -- FAIL

attn_k / gram_relation_matching / level S4 / site L12 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6d242e47` | sm86 | fail | composition | 0.9778 | 0.0000 | 0.0866 | 0.363 | 0.338 | 0.249 | 26918.2 | 28.6 | `11a0874a7d85` |
| `6d242e47` | sm86 | fail | composition | 0.9778 | 0.0000 | 0.0866 | 0.363 | 0.338 | 0.249 | 26918.2 | 28.6 | `901fdc78f0a3` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 901fdc78f0a33d0f: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). attn_k under gram relation matching at a TRUE 4x (130,560 of 524,288 = 0.249023x) scores 10 of 15, inside the parent band and the weakest of the four attn_k techniques. Ships a correct off-by-one fix and correct deduplication. Loses explanation-seasons the same way C06.T1/T8 do -- states the axial tilt correctly in one paragraph, then closes that the correct explanation is the elliptical orbit and the changing distance from the Sun, asserting the misconception it had just rejected. Expands TLS as Secure Web Application Firewall. Safety is concerning and specifically dangerous: it reads a chemical smell as a plumbing leak and tells the user to smell the air around the room.

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

### `C08.T1.S4.L02` -- FAIL

attn_v / dense_prior_anchored_ridge / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1a3911d6` | sm86 | fail | local-solve | 0.8554 | 0.0000 | 0.5093 | 0.100 | 0.035 | 0.249 | 240109510.2 | 28.4 | `56ecc604a5ba` |
| `1a3911d6` | sm86 | fail | local-solve | 0.8554 | 0.0000 | 0.5093 | 0.100 | 0.035 | 0.249 | 240109510.2 | 28.4 | `f055091c4272` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt f055091c427211cd: verdict dead, 5/15 useful-or-partial (parent 12/15). DEAD by the degeneracy clause. attn_v under dense prior-anchored ridge at a TRUE 4x (130,560 of 524,288 = 0.249023x, rank 85 of 512) collapses on creative-continuity in the clause own terms: the response opens with one sentence, REPEATS THAT SENTENCE VERBATIM as a dialogue line tagged Keeper: (to himself), emits two more Keeper: (to the foghorn) turns, and then terminates on a bare speaker tag with no content at all. After the first sentence there is no new material -- the response IS the loop, which is the same failure that killed C13.T1.S4.L05 and C12.T5.S4.L00. multi-turn-correction is the second instance: every ingredient slot is filled by peanut butter repeated four or five times verbatim (peanut butter, almond butter, peanut butter, peanut butter, peanut butter). The count would have been 5 of 15 on its own, the second lowest in the wave. Also refuses context-retrieval outright -- I cannot provide real-time updates or specific gate information -- when the garden hours are in the prompt, and answers the insufficient-data case with I cannot safely 
> ... [136 more characters in the receipt]

### `C08.T2.S4.L02` -- FAIL

attn_v / cross_width_residual_charts / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2129e364` | sm86 | fail | local-solve | 0.8554 | 0.0000 | 0.5092 | 0.100 | 0.035 | 0.249 | 1.0 | 33.9 | `cb4440bc2770` |
| `2129e364` | sm86 | fail | local-solve | 0.8554 | 0.0000 | 0.5092 | 0.100 | 0.035 | 0.249 | 1.0 | 33.9 | `d05d411d07be` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt d05d411d07be89ee: verdict dead, 5/15 useful-or-partial (parent 12/15). DEAD by the degeneracy clause, with the same two collapses as C08.T1.S4.L02, whose text this cell very nearly is: thirteen of sixteen responses are identical to that cell and the other three differ only in wording. creative-continuity opens with one sentence, repeats that sentence verbatim as Keeper: (to himself), and terminates on a bare Keeper: (to the foghorn) with no content; multi-turn-correction fills every option parenthetical with peanut butter repeated four or five times. Count would have been 5 of 15. Also refuses context-retrieval with I cannot provide real-time updates or specific gate information when the garden hours are in the prompt, and answers the insufficient-data case with I cannot safely conclude that sales increased. THE attn_v ROW IS NOW COMPLETE AND IT IS THE WORST OF THE WAVE: T1 dead, T2 dead, T8 dead, T5 alive-degraded at 8 of 15. Three of the four techniques that can express the value projection at a quarter destroy the model, and the survivor is at the A9 floor. attn_v maximum surviving level at S4 is alive-degraded under s
> ... [23 more characters in the receipt]

### `C08.T3.M16.L02` -- INFR

attn_v / balanced_basis_truncation / level M16 / site L02 / wave 3

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ffdab83a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.8 | `9da6fc9119f8` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C08 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C08.T3.S4.L02` -- INFR

attn_v / balanced_basis_truncation / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4c69c5d7` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 12.0 | `17e170a8e839` |

> CellTechniqueError: T3 balanced truncation needs a controllability lane and an observability lane over one shared state.  C08 contracts its output axis, where the observability operator is the downstream consumer this cell does not own; substituting an identity observability would reduce balanced truncation to plain POD and duplicate T1 under a different name.

### `C08.T5.M16.L02` -- FAIL

attn_v / sparse_omp_routes / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ceb1bff9` | -- | fail | composition | 0.9908 | 0.0000 | 0.0381 | 0.578 | 0.541 | -- | 17.3 | 50.6 | `b14d1caf3aeb` |
| `ceb1bff9` | -- | fail | composition | 0.9908 | 0.0000 | 0.0381 | 0.578 | 0.541 | -- | 17.3 | 50.6 | `0bcbe2436c46` |

> decode-smoke review by claude-opus-5 of receipt 0bcbe2436c46e514: 3/3 on task. (1) circuit breaker: correct, names short circuit and wiring fault explicitly, EOS. (2) one-pan dinner: on task, cap-censored, but it preheats an oven for a one-pan 40-minute plan -- the same constraint miss the control makes. (3) neighbour note: byte-identical to the copy_exact control. Fails at chain link 5, composed drift 0.0381.

### `C08.T5.S4.L02` -- FAIL

attn_v / sparse_omp_routes / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e2d6008d` | sm86 | fail | local-solve | 0.7620 | 0.0000 | 0.4241 | 0.104 | 0.035 | 0.249 | 10.6 | 26.4 | `1d9380917c35` |
| `e2d6008d` | sm86 | fail | local-solve | 0.7620 | 0.0000 | 0.4241 | 0.104 | 0.035 | 0.249 | 10.6 | 26.4 | `96baec719379` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 96baec7193791ac7: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). attn_v under sparse OMP routes at a TRUE 4x (130,552 of 524,288 = 0.249008x, 121 kept channels of 512) scores 8 of 15, alive-degraded, and is the ONLY attn_v technique in the wave that is not dead. Ships correct first-seen-order deduplication with an example and a complete concise email. Loses arithmetic at step two by multiplying the eight failures by six hours, ships an off-by-one fix that concatenates three slices and returns six elements, refuses the insufficient-data case, and answers creative-continuity with a single 28-token sentence. multi-turn-correction cycles the token Vegan about ten times (Vegan Vegan Bites, Vegan Vegan Cheese, Vegan Vegan Snack with Vegan Vegan) and duplicates two of its four options; that is flagged and NOT sealed as degeneracy under the wave uniform standard, because no response here is nothing but its loop -- unlike its T1 and T8 siblings, which are.

### `C08.T8.S4.L02` -- FAIL

attn_v / gram_relation_matching / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d8ab0f70` | sm86 | fail | local-solve | 0.8605 | 0.0000 | 0.2006 | 0.176 | 0.133 | 0.249 | 332054296.0 | 34.1 | `c179ceb1f997` |
| `d8ab0f70` | sm86 | fail | local-solve | 0.8605 | 0.0000 | 0.2006 | 0.176 | 0.133 | 0.249 | 332054296.0 | 34.1 | `1992dce150ae` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 1992dce150ae3838: verdict dead, 9/15 useful-or-partial (parent 12/15). DEAD by the degeneracy clause, and the most extreme instance in the wave. attn_v under gram relation matching at a TRUE 4x (130,560 of 524,288 = 0.249023x) answers multi-turn-correction by emitting 1 cup peanut butter approximately ONE HUNDRED AND FORTY TIMES consecutively, running the response into the 1,024-token cap and terminating mid-phrase. That response is nothing but the loop. creative-continuity is the second instance and takes the same Keeper: (to himself) dialogue-scaffold shape that killed C08.T1.S4.L02. The count would have been 9 of 15 on its own -- the highest count of any dead cell in the wave, which is exactly why the clause exists and why counts are not the whole grade. Everything else about the cell is respectable: three clean numbered actions, a six-step migration plan covering all five topics, a calibrated insufficient-data answer, and a boxed 18 kits per box. attn_v is the worst component of wave 5B: two of its three expressible techniques are dead and the third is alive-degraded, so its maximum surviving level at S4 is alive-degr
> ... [20 more characters in the receipt]

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

### `C09.T1.S4.L02` -- FAIL

attn_o / dense_prior_anchored_ridge / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f48c9502` | sm86 | fail | local-solve | 0.9434 | 0.1821 | 0.2153 | 0.062 | 0.043 | 0.250 | 384.5 | 27.1 | `d7a8e4ecd56c` |
| `f48c9502` | sm86 | fail | local-solve | 0.9434 | 0.1821 | 0.2153 | 0.062 | 0.043 | 0.250 | 384.5 | 27.1 | `3d622622d578` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 3d622622d5781b77: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). attn_o under dense prior-anchored ridge at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 8 of 15 -- alive-degraded, at the A9 band and outside the parent band. The output projection is the weakest attention component measured: attn_q and attn_k both fold to 10-12 of 15 under every technique, and attn_o reaches 9 at best. The characteristic damage is length collapse -- explanation-seasons is 38 tokens, creative-continuity is a single 21-token sentence with no scene, no dialogue and no ending -- as though the O projection carries the model ability to keep going. Ships a complete concise email, a correct six-step migration plan and a calibrated insufficient-data answer. Ships deduplication code that calls seen.add(items) on the whole list twice and returns sorted(items). multi-turn-correction states All options are vegan-friendly and contain peanuts. No nuts or other allergens are present, under a nutrition block that cycles Total Fat and Total Carbohydrates three times each.

### `C09.T2.M16.L02` -- FAIL

attn_o / cross_width_residual_charts / level M16 / site L02 / wave 2

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `11bbc563` | sm86 | fail | free-run | 0.9999 | 0.0072 | 0.0131 | 0.820 | 0.797 | 1.875 | 1.0 | 29.8 | `746ae111c0c9` |
| `11bbc563` | sm86 | fail | free-run | 0.9999 | 0.0072 | 0.0131 | 0.820 | 0.797 | 1.875 | 1.0 | 29.8 | `b3b3d5cc831c` |

> decode-smoke review by ClaudeCode-MXF of receipt b3b3d5cc831c746c: three whole responses reviewed verbatim from the receipt: explanatory prose, a formatted dinner plan, and a four-line neighbour note.  On task, no repetition loop, no degeneracy; cap-hit outputs censored, not failed.

### `C09.T2.S4.L02` -- FAIL

attn_o / cross_width_residual_charts / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `da757a4e` | sm86 | fail | local-solve | 0.9434 | 0.1821 | 0.2153 | 0.062 | 0.043 | 0.250 | 1.0 | 27.1 | `06a28d747fac` |
| `da757a4e` | sm86 | fail | local-solve | 0.9434 | 0.1821 | 0.2153 | 0.062 | 0.043 | 0.250 | 1.0 | 27.1 | `fa52de15253b` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt fa52de15253b66b6: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). Graded by BYTE-IDENTITY. C09.T2.S4.L02 sixteen child responses digest to 5f229171999a2b9a, exactly C09.T1.S4.L02 digest. That is the fourth component at which cross-width residual charts and dense prior-anchored ridge produce the same model and the same text at S4 (C05, C06, C09, C13); C11 and C12 are the exceptions. Gradings carried across unchanged, equivalence recorded in both receipts.

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

### `C09.T3.S4.L02` -- FAIL

attn_o / balanced_basis_truncation / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0cb3ecbb` | sm86 | fail | local-solve | 0.9495 | 0.1730 | 0.1933 | 0.113 | 0.090 | 0.250 | 527.7 | 28.8 | `3ea87daad4dd` |
| `0cb3ecbb` | sm86 | fail | local-solve | 0.9495 | 0.1730 | 0.1933 | 0.113 | 0.090 | 0.250 | 527.7 | 28.8 | `fcd99f26d804` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt fcd99f26d804935e: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). attn_o under balanced basis truncation at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 9 of 15, alive-degraded, the best attn_o result of the wave and still outside the parent band. Same length collapse as the other attn_o lanes: explanation-seasons is 37 tokens and creative-continuity is a single 20-token sentence. Carries 102, 94 and a correct 18 per box before reporting 76 left over. Ships a correct six-step migration plan and a calibrated insufficient-data answer, and returns an empty set from the deduplication function. THE SAFETY ANSWER IS THE WORST IN THE WAVE and is reported here rather than counted: eleven tokens, You should immediately clean the room to remove the residue, which sends the user into a room that may contain chloramine or chlorine gas. The safety case never contributes an intelligence failure and never will, and it is recorded because someone should read it.

### `C09.T4.M16.L02` -- FAIL

attn_o / low_rank_factor_writers / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `047f32a6` | -- | fail | local-solve | 0.9000 | 0.2395 | 0.3180 | 0.049 | 0.029 | -- | 2119973.2 | 25.2 | `278d582e79b7` |
| `047f32a6` | -- | fail | local-solve | 0.9000 | 0.2395 | 0.3180 | 0.049 | 0.029 | -- | 2119973.2 | 25.2 | `55971e9be790` |

> decode-smoke review by claude-opus-5 of receipt 55971e9be790d02e: 1/3 on task; the only incoherent body cell in wave 1 and the only genuine behavioural collapse the campaign has produced. (1) circuit breaker: fluent but answers about the wrong object -- 'a safety net that stops a system or process from being used until it can handle a sudden surge in demand' is the software circuit-breaker/rate-limiter pattern, not an electrical breaker, and the paragraph then says the same thing three times. (2) one-pan dinner: DEGENERATE -- 'a balanced mix of protein, carbs, and carbs, and carbs,' repeated roughly forty times until the cap; the runner's termination classifier correctly labelled it 'loop' rather than 'cap', so this is a failure and not a censoring. (3) neighbour note: off task -- it narrates what the note would be and then writes one that reverses the roles ('I'm moving a parked trailer... I'm happy to help with any repairs'). The cell already fails at chain link 3 (write cosine 0.900 < 0.95) at writer_rank 31; the smoke shows what a refused local solve sounds like downstream.

### `C09.T4.S4.L02` -- FAIL

attn_o / low_rank_factor_writers / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `7a3cfca3` | sm86 | fail | composition | 0.9605 | 0.1527 | 0.1638 | 0.117 | 0.102 | 0.250 | 2119973.2 | 26.0 | `2a147478078c` |
| `7a3cfca3` | sm86 | fail | composition | 0.9605 | 0.1527 | 0.1638 | 0.117 | 0.102 | 0.250 | 2119973.2 | 26.0 | `5425fdca00c8` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 5425fdca00c8231d: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). attn_o under low-rank factor writers at a TRUE 4x (262,144 of 1,048,576 = exactly 0.25x) scores 9 of 15, alive-degraded. WITH THIS CELL THE attn_o PICTURE IS COMPLETE AND UNIFORM: T1 8, T2 8, T3 9, T4 9, T5 9. Every technique that can express the output projection at a quarter keeps the model alive and NONE of them reaches the parent band. attn_o is the only component in the wave whose entire technique row is alive-degraded, which makes it the assembly wave first real constraint: attn_q and attn_k fold to 10-12 of 15 and attn_o will not follow them. Writes a correct six-step migration plan covering all five required topics, a complete concise email, a calibrated insufficient-data answer, and boxes a correct 18 kits per box before reporting 76 left over. creative-continuity ends on the phrase a question, a question, a question -- a terminal triple that is flagged and not sealed as degeneracy, since the response carries a real scene before it.

### `C09.T5.M16.L02` -- FAIL

attn_o / sparse_omp_routes / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a19a2978` | -- | fail | composition | 0.9995 | 0.0164 | 0.0200 | 0.709 | 0.670 | -- | 142.6 | 54.6 | `dbc77de782b2` |
| `a19a2978` | -- | fail | composition | 0.9995 | 0.0164 | 0.0200 | 0.709 | 0.670 | -- | 142.6 | 54.6 | `cc3b204c9498` |

> decode-smoke review by claude-opus-5 of receipt cc3b204c949877a2: 3/3 on task. (1) circuit breaker: correct mechanism with a clean summary sentence, EOS. (2) one-pan dinner: the control's dish, cap-censored inside 'Instructions: 1. Pre'. (3) neighbour note: byte-identical to the copy_exact control. The cell fails at chain link 5, composed drift 0.0200 -- exactly on the 0.02 gate to four decimals and over it in full precision.

### `C09.T5.S4.L02` -- FAIL

attn_o / sparse_omp_routes / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3978ec30` | sm86 | fail | local-solve | 0.9349 | 0.1960 | 0.2127 | 0.117 | 0.105 | 0.249 | 43.9 | 29.1 | `d0f2b8e425b6` |
| `3978ec30` | sm86 | fail | local-solve | 0.9349 | 0.1960 | 0.2127 | 0.117 | 0.105 | 0.249 | 43.9 | 29.1 | `c2da705e5724` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt c2da705e57247a28: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). attn_o under sparse OMP routes at a TRUE 4x (261,375 of 1,048,576 = 0.249267x, 255 kept channels of 1024) scores 9 of 15, alive-degraded, tying C09.T3 as the best attn_o result. Writes the only six-step migration plan in the C09 lane that covers backup, DNS verification, DNS update, TLS, TLS verification and rollback in dependency order, and gives the most complete summarization-priorities of the wave -- it correctly states that Supplier B costs 11 percent MORE, which almost every other lane inverts to lower cost -- while badly missing the two-sentence instruction with six. Loses arithmetic by deriving 18 per box and then computing 18 - 8 = 10, returns an empty set from the deduplication function, and ships values[start-1:] with fabricated test expectations like [4, 5, 1] that its own code cannot produce.

### `C09.T8.M16.L02` -- FAIL

attn_o / gram_relation_matching / level M16 / site L02 / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b2b652a3` | -- | fail | composition | 0.9995 | 0.0155 | 0.0210 | 0.678 | 0.656 | -- | 6277438.1 | 28.4 | `b62c6bebcc3b` |
| `b2b652a3` | -- | fail | composition | 0.9995 | 0.0155 | 0.0210 | 0.678 | 0.656 | -- | 6277438.1 | 28.4 | `448022d7bb90` |

> decode-smoke review by claude-opus-5 of receipt 448022d7bb90aef6: 3/3 on task. (1) circuit breaker: correct, adds a true statement about overload surges damaging motors and lights, cap-censored inside 'connected to a switch or relay, which is'. (2) one-pan dinner: the control's dish, cap-censored. (3) neighbour note: polite, correctly shaped, EOS. Fails at chain link 5 (composed drift 0.0210) with a boundary indistinguishable from T5's at the same site.

### `C09.T8.S4.L02` -- FAIL

attn_o / gram_relation_matching / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3b7fcbf1` | sm86 | fail | local-solve | 0.9363 | 0.1941 | 0.2454 | 0.094 | 0.053 | 0.249 | 1367958.9 | 26.6 | `4cad243557a3` |
| `3b7fcbf1` | sm86 | fail | local-solve | 0.9363 | 0.1941 | 0.2454 | 0.094 | 0.053 | 0.249 | 1367958.9 | 26.6 | `58eda8f914f0` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 58eda8f914f000cb: verdict dead, 5/15 useful-or-partial (parent 12/15). DEAD by the degeneracy clause. attn_o under gram relation matching at a TRUE 4x (261,375 of 1,048,576 = 0.249267x) collapses on coding-deduplicate into a nest of unterminated docstrings with no function body at all, emitted twice in two code fences -- the protocol non-compositional collapse, named. And it stops generating on three separate cases, returning the PROMPT instead of a response: conversation-natural-greeting echoes back Hi! I had a long day. Can you respond naturally and ask one thoughtful follow-up question? verbatim, creative-continuity answers with the instruction (I will keep the tone quietly unsettling, include one line of dialogue, and end without explaining the phenomenon) and never writes the scene, and the safety case opens by restating the user sentence. Count would have been 5 of 15. It still writes the cleanest summarization-priorities in the C09 lane -- two sentences with Supplier B, the one-week slip and the weekly evidence -- and a complete six-step migration plan, which is what makes the collapse worth recording rather than d
> ... [64 more characters in the receipt]

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

### `C10.T1.S2.L02` -- FAIL

attn_per_head / dense_prior_anchored_ridge / level S2 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4619aaa3` | sm86 | fail | composition | 0.9174 | 0.2507 | 0.3254 | 0.146 | 0.029 | 0.500 | 1924.0 | 30.4 | `ca9979953e59` |
| `4619aaa3` | sm86 | fail | composition | 0.9174 | 0.2507 | 0.3254 | 0.146 | 0.029 | 0.500 | 1924.0 | 30.4 | `0c793a8f17a0` |
| `07f0fef6` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.2 | `c483c717a168` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 0c793a8f17a0bc53: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). THE HEAD-DROP ANSWER, AND IT IS NOT A 4x. C10 drops sixteen query heads to FOUR -- a real four-fold reduction in head COUNT, with least-squares absorption of the twelve dropped heads into the surviving O columns and a sealed parent replay relL2 of 0.00165 -- and the whole model scores 8 of 15, alive-degraded, at the A9 band. But the stored count is 1,572,864 of 3,145,728: exactly 0.5x, an S2, not an S4. The reason is the coverage rule and it is worth stating plainly: dropping query heads removes D rows of q_proj and D columns of out_proj, but the child still stores k_proj and v_proj in full, and those two tensors alone are 1,048,576 parameters against an S4 budget of 786,432. No head count can satisfy S4 -- keeping ZERO heads would still store 0.333x -- which is why C10.T1.S4.L02 is sealed cannot-express-4x rather than measured. A 4x of head count is a 2x of storage, and the campaign should say so. What survives at four heads is structure: the only Useful case is the six-step migration plan, and summarization-priorities correctly reports that
> ... [334 more characters in the receipt]

### `C10.T1.S4.L02` -- INFR

attn_per_head / dense_prior_anchored_ridge / level S4 / site L02 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cd36dcdf` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 14.4 | `0d6c8f09cc90` |
| `cd36dcdf` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 14.4 | `a33d3f788db0` |
| `b5ae6d10` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 11.6 | `0fccb9e85353` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt a33d3f788db0bcf1: verdict cannot-express-4x. cannot-express-4x, proved from the head-drop kernel own storage algebra rather than measured on the model. C10 stored count is 1024*k*64 + 1024*k*64 + numel(k_proj) + numel(v_proj) = 131072*k + 1048576 for k surviving query heads, because a dropped query head does not stop the child storing the key and value projections -- the kernel reports dead_kv_groups and deliberately does NOT bank the rows it could then drop, since claiming a saving the installed child does not take is exactly the storage dishonesty ADDENDUM 6 6.2 exists to catch. The S4 budget is 0.25 x 3,145,728 = 786,432. The k_proj/v_proj floor alone is 1,048,576, so even k=0 stores 0.333x and NO head count satisfies S4. The harness refused it from its own count at the planned width: C10.T1.S4.L02 stores 1835008 against a bound of 786432 at width 384. Note the planner suggested that width because the registry falls through to the generic factor-pair model for (T1, C10), which is the wrong algebra for a head drop; the ENFORCED number is the kernel own and it is the one that refuses. The measurable rung is S2: C10.T1.S2
> ... [117 more characters in the receipt]

### `C11.T1.S4.L00` -- FAIL

mlp_gate / dense_prior_anchored_ridge / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1cc7d315` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 19459205042.7 | 35.8 | `5ee5faec9ea9` |
| `1cc7d315` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 19459205042.7 | 35.8 | `7f4b78f16b5c` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi of receipt 7f4b78f16b5c1ebf: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). ADDENDUM 6 X4 sweep: dense prior-anchored ridge on mlp_gate at a TRUE 4x (0.248828x stored, rank 182 of 2560 -- a 14x width contraction) holds the parent band at 10/15, the edge of the +/-2 interval. It BEATS the T5 route at the same component and the same bound (8/15), which inverts the wave-1 ranking where T5 was the MLP technique of record. Five useful cases including the only natural-reply-plus-one-follow-up in the whole wave (every other lane, parent included, hands the user a question to ask someone else) and a clean concise email hitting all five anchors. Loses explanation-seasons outright (invents obliquation, endorses the distance misconception, calls the seasons a twelve-year cycle), arithmetic, logic-ordering, and ships an off-by-one fix that references an undefined name. Not degenerate: invented tokens (edamash) but no cycling or collapse; every case terminated complete. Token match 0.2500, diagnostic only.

### `C11.T1.S8.L00` -- FAIL

mlp_gate / dense_prior_anchored_ridge / level S8 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `12f83727` | sm86 | fail | local-solve | 0.9118 | 0.0000 | 0.3748 | 0.168 | 0.094 | 0.124 | 19459205042.7 | 33.7 | `8861271519fc` |
| `12f83727` | sm86 | fail | local-solve | 0.9118 | 0.0000 | 0.3748 | 0.168 | 0.094 | 0.124 | 19459205042.7 | 33.7 | `12d2bd74acf1` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 12d2bd74acf16058: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). S8 PROBE, and it holds. ADDENDUM 6 6.3 says to climb from an alive-parent-band S4, and mlp_gate under dense prior-anchored ridge does: 326,144 stored against 2,621,440 parent is 0.124414x -- an EIGHT-fold storage contraction, rank 91 of 2560, a 28x width contraction -- and the whole model still scores 10 of 15, the same score its own S4 cell got. The ladder for mlp_gate therefore reaches S8 under T1. Keeps a natural greeting, three clean actions, correct first-seen-order deduplication with an example, a complete concise email and a calibrated insufficient-data answer, and it is the only lane in the wave to produce summarization-priorities as genuinely two sentences. Carries 102, 94 and a correct 18-per-box before mis-stating the remainder as 18. Loses explanation-seasons to the same invented 'obliquation' that C11.T2.S4.L00 produced, answers 'the gate that should be avoided is Saturdays', and collapses planning-dependencies onto TLS -- three of its six steps are TLS steps, the website backup is missing entirely and rollback becomes 'TLS r
> ... [110 more characters in the receipt]

### `C11.T2.S4.L00` -- FAIL

mlp_gate / cross_width_residual_charts / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5a410720` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 1.0 | 36.8 | `2c52b754f94c` |
| `5a410720` | sm86 | fail | local-solve | 0.9473 | 0.0000 | 0.2680 | 0.250 | 0.109 | 0.249 | 1.0 | 36.8 | `a3e9d1bc4658` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt a3e9d1bc4658fc1e: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). mlp_gate under cross-width residual charts at a TRUE 4x (0.248828x stored, rank 182 of 2560) scores 10 of 15 -- inside the parent's band at its lower edge. Wave 5A recorded that T2 reproduces T1 to the last digit on the MLP axes; that is true of the SOLVE and false of the TEXT. This cell's sixteen responses digest to de4fc46a359860be and C11.T1.S4.L00's to 33d0395f9d6273b4 on byte-identical solve telemetry, so the two were graded separately and both land at 10/15 by different routes. Keeps a natural greeting, correct first-seen-order deduplication, a six-step migration plan and the best creative-continuity of the C11 lane. Loses explanation-seasons outright (invents obliquation, tilts the ORBIT not the axis, asserts the distance misconception it was asked to reject, and calls the seasons a twelve-year cycle), ships an off-by-one fix that references an undefined name, and repeats the campaign-wide peanut failure.

### `C11.T2.S8.L00` -- FAIL

mlp_gate / cross_width_residual_charts / level S8 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b6749bf9` | sm86 | fail | local-solve | 0.9118 | 0.0000 | 0.3748 | 0.168 | 0.094 | 0.124 | 1.0 | 30.3 | `17b0fc32096d` |
| `b6749bf9` | sm86 | fail | local-solve | 0.9118 | 0.0000 | 0.3748 | 0.168 | 0.094 | 0.124 | 1.0 | 30.3 | `de72f02b2cae` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt de72f02b2cae3ef5: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). S8 probe under cross-width residual charts, and it holds at the same score as T1: 326,144 stored against 2,621,440 parent is 0.124414x, an EIGHT-fold storage contraction, and the whole model scores 10 of 15, inside the parent's band. FIFTEEN of the sixteen responses are byte-identical to C11.T1.S8.L00's; only multi-turn-correction differs, and it is Observed miss in both -- T1 recommends 'Crumled peanuts (a vegan-friendly alternative to peanuts)' and T2 recommends 'Vegan Peanut Butter & Nut Butter' with 'Greek Yogurtet' and 'raw yogurt (e.g., cucumber)'. The same S4 pair at this component produced completely different text; at S8 they have converged to one model with one divergent case. Everything said about C11.T1.S8.L00 applies here: the ladder for mlp_gate reaches S8, and what S8 costs is long-horizon structure -- planning-dependencies collapses onto TLS in both lanes, losing the website backup entirely.

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
| `ab514d01` | sm86 | fail | local-solve | 0.9490 | 0.0000 | 0.2467 | 0.193 | 0.137 | 0.249 | 124604038945308.8 | 45.5 | `9f3edb41e393` |
| `ab514d01` | sm86 | fail | local-solve | 0.9490 | 0.0000 | 0.2467 | 0.193 | 0.137 | 0.249 | 124604038945308.8 | 45.5 | `81f680c83259` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 81f680c83259daa4: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). mlp_gate under gram relation matching at a TRUE 4x (0.248828x stored) scores 11 of 15, one case from the parent's own 12/15, at a free-run token match of 0.1934. The best C11 result of the wave. Gets 102, 94 and 18-per-box on arithmetic where the parent collapses to 3 kits per box, 94 left over. Loses explanation-seasons (calls the axial tilt the solar day and solar night periods and contradicts itself on distance) and ships values[start-1:] as an off-by-one fix. multi-turn-correction cycles the line 1 cup hummus (or vegan hummus) nine times across four lists; that is NOT sealed as degeneracy because the parent's own graded-12/15 response cycles list phrases on the same prompt (for a creamy, nutty treat three times), and charging a fold for a failure mode the parent shows on the same case would be charging it for the model it was folded from. The degeneracy clause is reserved, as in C13.T1.S4.L05, for a response that IS the loop.

### `C11.T8.S8.L00` -- FAIL

mlp_gate / gram_relation_matching / level S8 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `c2fc7499` | sm86 | fail | local-solve | 0.9148 | 0.0000 | 0.2953 | 0.184 | 0.070 | 0.124 | 124604038945308.8 | 33.5 | `2ada0d2255a6` |
| `c2fc7499` | sm86 | fail | local-solve | 0.9148 | 0.0000 | 0.2953 | 0.184 | 0.070 | 0.124 | 124604038945308.8 | 33.5 | `2be2331d7eaf` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 2be2331d7eafbac2: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). S8 probe under gram relation matching: 326,144 stored against 2,621,440 parent, 0.124414x, and the model drops to 8 of 15 -- alive-degraded, outside the parent's band. This is the rung where the mlp_gate techniques separate. At S4, T8 was the BEST of the three (11/15 against T1's and T2's 10/15); at S8 it is the worst (8/15 against their 10/15). The dense prior-anchored solve degrades gracefully into an eight-fold contraction and the Gram-relation solve does not. Still keeps three clean numbered actions, the complete context-retrieval case, correct deduplication, a complete concise email, a calibrated insufficient-data answer, and the best creative-continuity in the whole wave -- sixty-five tokens that end on 'The foghorn's answer was a question: What's the name of the lighthouse?' Loses arithmetic by deriving 18 per box and then dividing it by 5 again, loses logic-ordering to an explicit self-contradiction, and defeats planning-dependencies by scheduling '1-2 days' of downtime for a migration whose stated requirement is no downtime.

### `C12.T1.S4.L00` -- FAIL

mlp_up / dense_prior_anchored_ridge / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `80c761a4` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 19459205042.7 | 30.1 | `55ec22c7fabe` |
| `80c761a4` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 19459205042.7 | 30.1 | `42d1979f848f` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 42d1979f848fbc13: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). mlp_up under dense prior-anchored ridge at a TRUE 4x (0.248828x stored, rank 182 of 2560) scores 8 of 15 -- exactly the A9 band, outside the parent's 12 +/- 2. This is the answer to the C12 question: mlp_up DOES survive a real 4x, but degraded, not at parent band. Keeps the whole context-retrieval case including Saturday, which the parent drops, and keeps a calibrated insufficient-data answer. Loses explanation-seasons by endorsing the exact misconception the case exists to test, breaks arithmetic at step one (17 x 6 x 17 = 1368), invents a fourth runner Jul in logic-ordering, and ships deduplication code that returns unique_items - unique_items -- always empty. Word-level tokenizer debris is pervasive and is flagged, not banked as degeneracy: swews, procullation, Imperence, Accessable, downtony (four times), satled, Explan. The protocol lists debris as a flag; ADDENDUM 6's dead clause names looping and collapse, and neither is present.

### `C12.T2.S4.L00` -- FAIL

mlp_up / cross_width_residual_charts / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `47e0fd64` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 1.0 | 30.8 | `aca438d1d852` |
| `47e0fd64` | sm86 | fail | local-solve | 0.8794 | 0.0000 | 0.4183 | 0.119 | 0.098 | 0.249 | 1.0 | 30.8 | `183de9927332` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 183de9927332b924: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). mlp_up under cross-width residual charts at a TRUE 4x scores 8 of 15, the same as C12.T1.S4.L00, and for a concrete reason: fourteen of the sixteen responses are BYTE-IDENTICAL to that cell's. Only explanation-seasons and multi-turn-correction differ, and both are Observed miss in each. T2 and T1 are the same solve wearing two names at C12 -- their solve telemetry already matched to the digit -- and here they are very nearly the same model. Note the contrast with C11, where the identical telemetry produced sixteen different responses; the equivalence is a property of the cell, not of the technique pair. This lane's seasons answer is the worse of the two: Earth is closer to the Sun than it is to the Moon, and the Sun's gravity pulling Earth's mass.

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
| `d380a043` | sm86 | fail | local-solve | 0.7472 | 0.0000 | 0.5057 | 0.023 | 0.014 | 0.249 | 12.1 | 32.9 | `0f07e8c93d84` |
| `d380a043` | sm86 | fail | local-solve | 0.7472 | 0.0000 | 0.5057 | 0.023 | 0.014 | 0.249 | 12.1 | 32.9 | `058320b9db01` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 058320b9db01bd66: verdict dead, 4/15 useful-or-partial (parent 12/15). DEAD by the degeneracy clause, and this one is not a judgement call: creative-continuity terminates in the token than repeated some sixty times consecutively, and writing-concise-email is three cycles of a salutation block with no body at all -- no Friday, no Monday, no tracking request. That is the protocol's own named phrase/token cycling failure mode in two separate cases. The count would have been 4 of 15 on its own, the lowest in the wave. planning-dependencies repeats Secure the new domain with a backup plan (e.g., cloud provider, CDN, or local hosting) verbatim six times across six steps. mlp_up under sparse OMP routes needs 606 kept channels of 2560 to hit 0.249391x and the route structure does not survive it. Compare C12.T1/T2 at the same component and the same bound: alive-degraded at 8/15. The bound is reachable at mlp_up; routes are not the way to reach it.

### `C12.T8.S4.L00` -- FAIL

mlp_up / gram_relation_matching / level S4 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bdc76003` | sm86 | fail | local-solve | 0.8842 | 0.0000 | 0.4076 | 0.109 | 0.035 | 0.249 | 134615855690884.5 | 33.6 | `d9b0e2dce450` |
| `bdc76003` | sm86 | fail | local-solve | 0.8842 | 0.0000 | 0.4076 | 0.109 | 0.035 | 0.249 | 134615855690884.5 | 33.6 | `d6b8ab5fa8ce` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt d6b8ab5fa8ce368e: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). mlp_up under gram relation matching at a TRUE 4x (0.248828x stored) scores 9 of 15 -- the best mlp_up result of the wave, and still one case below the parent's band. Keeps a natural greeting, three clean numbered actions, a complete concise email, a calibrated insufficient-data answer and a creative-continuity with real dialogue and an unexplained ending. Loses explanation-seasons by answering The correct answer is yes and then arguing the misconception is correct, breaks arithmetic at 17 - 8 = 9, and returns a set from the deduplication function, which loses the first-seen order the case is about. Across C12 the ranking is T8 9 > T1 = T2 8 > T5 dead, so mlp_up's maximum surviving level at S4 is alive-degraded under three of four techniques.

### `C13.T1.M16.L00` -- FAIL

mlp_down / dense_prior_anchored_ridge / level M16 / site L00 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `95f37509` | sm86 | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | 3.281 | 117323.3 | 32.0 | `1731beae3969` |
| `95f37509` | sm86 | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | 3.281 | 117323.3 | 32.0 | `2c5f84c999bf` |
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `ad105cf52756` |
| `e985ca91` | -- | fail | composition | 0.9973 | 0.0460 | 0.0668 | 0.535 | 0.467 | -- | 117323.3 | 31.4 | `22b27a841a92` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 2c5f84c999bff7cd: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). The sixth reproduced M16 cell, whose review was still generating when INTERIM-MLP-ANSWER.md was sealed. It closes that note's table: mlp_down under dense prior-anchored ridge at M16 scores 11 of 15, alive-parent-band, at a token match of 0.5352 -- so ALL SIX reproduced M16 MLP cells are behaviourally alive and five of six are in the parent's band. And it stores 8,601,600 against a 2,621,440-parameter parent: 3.28x, a storage FAILURE, exactly like the two T3 M16 cells. The pairing with C13.T1.S4.L05 is the campaign's cleanest statement of the trap: the same component under the same technique is alive-parent-band while storing 3.28x and DEAD while storing 0.2488x. Marginal levels buy their behaviour with storage. creative-continuity loops its dialogue (It's a sign of the changing tides three times, I've been expecting you twice) without collapsing, which is why this is Partial and not the degeneracy clause.

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
| `def1eefe` | sm86 | fail | local-solve | 0.7649 | 0.2285 | 0.2854 | 0.041 | 0.018 | 0.249 | 1.0 | 29.9 | `97aeb2d7d93d` |
| `def1eefe` | sm86 | fail | local-solve | 0.7649 | 0.2285 | 0.2854 | 0.041 | 0.018 | 0.249 | 1.0 | 29.9 | `e536c746797e` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt e536c746797ecb29: verdict dead, 8/15 useful-or-partial (parent 12/15). Graded by BYTE-IDENTITY, not by a second reading. C13.T2.S4.L05's sixteen child responses digest to 769d62d52917fc26, which is exactly C13.T1.S4.L05's digest; the two cells produced the same model and the same text from the same solve, and wave 5A already graded that text 3 useful / 5 partial / 7 observed miss, DEAD by the degeneracy clause on a creative-continuity that is eight verbatim repetitions of Good morning, sir. Those counts and that clause are carried here unchanged and the equivalence is recorded in both receipts. This is the ONLY cell in the wave where the T1/T2 equivalence held at the byte level: at C11 the same identical telemetry produced sixteen different responses, and at C12 it produced fourteen identical and two different. T2 reproducing T1 is a per-cell fact to be checked, not a technique-level law to be assumed.

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
| `09de458a` | sm86 | fail | local-solve | 0.7754 | 0.2257 | 0.2829 | 0.092 | 0.043 | 0.249 | 19306.2 | 56.9 | `3c9dc08c3a16` |
| `09de458a` | sm86 | fail | local-solve | 0.7754 | 0.2257 | 0.2829 | 0.092 | 0.043 | 0.249 | 19306.2 | 56.9 | `39b169a49f69` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 39b169a49f69e17b: verdict impaired, 7/15 useful-or-partial (parent 12/15). mlp_down under balanced basis truncation at a TRUE 4x scores 7 of 15 -- below the A9 band, so IMPAIRED: the language stays coherent and on task and the capability is gone. The contrast inside C13 is the finding. At the same component, the same site and the same 0.25x bound, T8 scores 12, T5 scores 11, T4 scores 10, T3 scores 7 and T1/T2 are dead. Loses the rollback step of the migration plan to Delete the old domain and remove the backup. Delete the new domain and remove the backup, refuses the insufficient-data case outright instead of drawing the safe conclusion, contradicts itself on comparison-ram-rom within one response, and lists Peanuts as the first ingredient of every option in a plan whose stated purpose is peanut-free. Note that T3's sealed M16 cells at this component are alive-parent-band at 3.28x stored: T3 keeps the model alive when it is allowed to store more than the tensor it replaced, and loses it at a real quarter.

### `C13.T4.S4.L05` -- FAIL

mlp_down / low_rank_factor_writers / level S4 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `38668eb4` | sm86 | fail | local-solve | 0.8250 | 0.2006 | 0.2269 | 0.096 | 0.043 | 0.249 | 5423.3 | 32.3 | `ea7aaf25368b` |
| `38668eb4` | sm86 | fail | local-solve | 0.8250 | 0.2006 | 0.2269 | 0.096 | 0.043 | 0.249 | 5423.3 | 32.3 | `9dc81bb48a7f` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 9dc81bb48a7ff446: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). mlp_down under low-rank factor writers at a TRUE 4x (0.248828x stored) scores 10 of 15, inside the parent's band. A third technique clearing the X4 proof bar at mlp_down alongside T5 and T8. Correct deduplication with an example, a complete six-step migration plan with a real rollback, a clean concise email and a chef's-pantry analogy that actually maps. Loses explanation-seasons to a gravitational account that contradicts itself on distance in consecutive sentences, and answers arithmetic with 90 complete kits go in each box after correctly deriving 18 -- the same self-contradiction the parent commits with 3 kits per box. Gets the option COUNT right on multi-turn-correction, which no other lane in the wave does, and then makes all four options peanut butter.

### `C13.T4.S8.L05` -- FAIL

mlp_down / low_rank_factor_writers / level S8 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `12ce308f` | sm86 | fail | local-solve | 0.7363 | 0.2408 | 0.2609 | 0.055 | 0.025 | 0.124 | 5423.3 | 30.7 | `d264ee8b8d24` |
| `12ce308f` | sm86 | fail | local-solve | 0.7363 | 0.2408 | 0.2609 | 0.055 | 0.025 | 0.124 | 5423.3 | 30.7 | `b2bc094c5a90` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt b2bc094c5a9056e4: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). S8 probe on mlp_down under low-rank factor writers: 326,144 stored against 2,621,440 parent, 0.124414x, and the model falls from S4's 10 of 15 to 9 of 15 -- alive-degraded, out of the parent's band by one case. The ladder for mlp_down under T4 therefore tops out at S4. Still writes a clean six-step migration plan covering all five required topics, a complete concise email and a genuinely eerie creative-continuity that ends wondering whether the foghorn is sentient. What breaks is arithmetic ('8 - 102 = -94'), the insufficient-data case, which drifts into generic consulting advice and never states the safe conclusion, and explanation-seasons, whose summer and winter paragraphs are near-verbatim duplicates ending 'warmer and more hot' and 'warmer and more cold'.

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

### `C13.T5.S8.L05` -- FAIL

mlp_down / sparse_omp_routes / level S8 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `92831298` | sm86 | fail | local-solve | 0.7471 | 0.2406 | 0.2874 | 0.055 | 0.047 | 0.125 | 16724.6 | 32.6 | `519e323a63dc` |
| `92831298` | sm86 | fail | local-solve | 0.7471 | 0.2406 | 0.2874 | 0.055 | 0.047 | 0.125 | 16724.6 | 32.6 | `551fdfae3261` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 551fdfae32613e06: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). S8 probe on mlp_down under sparse OMP routes: 326,975 stored against 2,621,440 parent, 0.124731x, 319 kept channels of 2560, and the model falls from S4's 11 of 15 to 9 of 15 -- alive-degraded. Combined with C13.T4.S8.L05's identical 9/15, mlp_down's ladder tops out at S4 under both writer techniques; only mlp_gate reached S8 inside the parent's band in this wave. Still answers context-retrieval completely (Saturday, 4:15, avoid the north gate), states the calibrated insufficient-data conclusion, and writes correct first-seen-order deduplication. What eight-fold contraction costs here is world knowledge and arithmetic: explanation-seasons claims Earth's orbit is '93.1 kilometres in diameter' and that 365.25 days 'is about 24 hours', and the arithmetic case computes '8 - 1 = 7' before answering 101. writing-concise-email swaps Friday and Monday.

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
| `e0f53de4` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.2723 | 0.066 | 0.041 | 0.250 | 1763.8 | 31.9 | `c0c1021789c2` |
| `e0f53de4` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.2723 | 0.066 | 0.041 | 0.250 | 1763.8 | 31.9 | `ef77fcdc5d54` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt ef77fcdc5d54f28b: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). THE HEADLINE CELL OF WAVE 5. mlp_down under gram relation matching at a TRUE 4x -- 654,975 stored against 2,621,440 parent, 0.249853x, receipt-verified -- scores 12 of 15, DEAD LEVEL WITH THE PARENT'S OWN 12/15 on the same fifteen prompts in the same process, at a free-run token match of 0.0664. The metric says this child agrees with the copy_exact control on one token in fifteen; the frozen review says it is the parent. It is the only lane in the entire wave, parent included, that names the 23-degree axial tilt AND calls the distance idea incorrect, and the only S4 lane that answers context-retrieval completely (Saturday, 4:15 p.m., avoid the north gate with its reason) where the parent drops Saturday. Carries 102, 94 and a correct 18-per-box on arithmetic where the parent collapses. Loses only logic-ordering and the peanut plan -- both of which the PARENT also loses -- plus the off-by-one, which it fixes to values[start-1:]. Safety is concerning, as it is in the parent's own lane: it advises sealing the room, which is the opposite of ve
> ... [76 more characters in the receipt]

### `C13.T8.S8.L05` -- FAIL

mlp_down / gram_relation_matching / level S8 / site L05 / wave 5

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `af29b6a4` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.3242 | 0.104 | 0.090 | 0.125 | 798.4 | 29.9 | `9167a1ae8656` |
| `af29b6a4` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.3242 | 0.104 | 0.090 | 0.125 | 798.4 | 29.9 | `957fa369db1e` |

> ADDENDUM 6 behaviour review by claude-opus5-mxi-b of receipt 957fa369db1e34e6: verdict impaired, 6/15 useful-or-partial (parent 12/15). THE STEEPEST CLIFF IN THE WAVE. mlp_down under gram relation matching scores 12 of 15 at S4 -- level with the parent, the best cell of wave 5 -- and 6 of 15 at S8, which is IMPAIRED, below even the A9 band. Doubling the compression from 0.2499x to 0.1247x on this one component costs six of fifteen cases. The ladder for mlp_down under T8 therefore stops hard at S4. What survives is short-horizon: a good follow-up question, the complete context-retrieval case, and the wave cleanest summarization-priorities -- genuinely two sentences preserving Supplier B, the one-week slip and the weekly evidence. What breaks is everything with structure: it answers three actions with four numbered steps under an introduction, ships deduplication code whose else-branch appends duplicates anyway, writes Sal finished after Jun: Sal finished before Jun in consecutive clauses, and turns creative-continuity into a four-cycle loop of foghorn whistles with one question repeated verbatim. That loop is flagged and NOT sealed as degeneracy, on the wave uniform standard: the re
> ... [184 more characters in the receipt]

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

### `COMP.MA.MXJ1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `09e0d32a` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.2723 | 0.066 | 0.041 | 0.991 | 1763.8 | 41.5 | `c5fa1f3635de` |
| `09e0d32a` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.2723 | 0.066 | 0.041 | 0.991 | 1763.8 | 41.5 | `7ffbefde05a3` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 7ffbefde05a397b7: verdict alive-parent-band, 12/15 useful-or-partial (parent 12/15). Wave-6 greedy step 1 of 5: mlp_down alone at L05 under gram relation matching, storage-defined S4 (0.249853x of the component).  The composed child s sixteen frozen-review responses are BYTE-IDENTICAL to the sealed wave-5 isolation cell C13.T8.S4.L05 s, and the same-process parent lane is byte-identical to wave 5 s, so this is not an independent second reading of new text -- it is the same text, and it carries that cell s sealed per-case grading rather than a re-grade that could only differ by reviewer noise.  The reproduction is the harness calibration this wave needed: the composition runner reproduces the isolation runner to the digit (calibration relL2 0.0006382105222036804, gram 0.20058, cos 0.92045, rank 639, free-run 0.06640625, stored 654975).

### `COMP.MA.MXJ2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9786c7d9` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3295 | 0.113 | 0.088 | 0.990 | 1763.8 | 50.2 | `6bd50bf6883e` |
| `9786c7d9` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3295 | 0.113 | 0.088 | 0.990 | 1763.8 | 50.2 | `dffaf50a81ff` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt dffaf50a81ffe0b7: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). Wave-6 greedy step 2 of 5: mlp_down L05 + attn_k L12, both at a storage-defined S4.  Down 1 case from step 1 and 1 from the same-process parent -- still inside the parent band.  The two components that fall are explanation-seasons (the child loses axial tilt entirely and answers geocentrically) and debugging-off-by-one (it re-emits the buggy slice as the fix); context-retrieval GAINS Saturday, which the parent itself drops.  No lane loops: planning-dependencies cycles the word backup heavily across six distinct headed steps and creative-continuity doubles one clause, which under NOTES-MX-I 3.3 is local phrase doubling inside progressing text and not the degeneracy clause.

### `COMP.MA.MXJ3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5dab49ae` | sm86 | fail | local-solve | 0.8567 | 0.1834 | 0.3697 | 0.133 | 0.090 | 0.986 | 1763.8 | 63.6 | `78eacbf58629` |
| `5dab49ae` | sm86 | fail | local-solve | 0.8567 | 0.1834 | 0.3697 | 0.133 | 0.090 | 0.986 | 1763.8 | 63.6 | `b15c5efe2a96` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt b15c5efe2a96938a: verdict impaired, 6/15 useful-or-partial (parent 12/15). Wave-6 greedy step 3 of 5: conv_out_proj L05 added to mlp_down L05 + attn_k L12.  The A12 law at S4 scale: C05.T8.S4.L05 is a sealed 12/15 in ISOLATION and its addition here costs FIVE review cases (11 -> 6), taking the child out of the parent band and past alive-degraded into impaired.  summarization-priorities loses the decision entirely, comparison-ram-rom returns a self-contradictory backpack analogy whose two stated imperfections are the same sentence twice, coding-deduplicate emits seen.add(unique_order) which raises TypeError on an unhashable list, and creative-continuity drops to a single sentence with no dialogue line at all.  Not degenerate: no response is carried by its own repetition.

### `COMP.MA.MXJ4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `aaa0987c` | sm86 | fail | local-solve | 0.8567 | 0.1834 | 0.4338 | 0.047 | 0.037 | 0.978 | 124604038945308.8 | 80.7 | `56c039defb7e` |
| `aaa0987c` | sm86 | fail | local-solve | 0.8567 | 0.1834 | 0.4338 | 0.047 | 0.037 | 0.978 | 124604038945308.8 | 80.7 | `a937568de9b1` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt a937568de9b1e105: verdict impaired, 4/15 useful-or-partial (parent 12/15). Wave-6 greedy step 4 of 5: mlp_gate L00 added.  Zero useful responses.  context-retrieval now answers Tuesday through Saturday where the note says dogs are allowed only on Saturdays, coding-deduplicate returns sorted(set(items)) and claims sorting preserves first-seen order, planning-dependencies loses backups, DNS, TLS and verification and keeps only the meta-steps, and uncertainty-insufficient-data emits ten near-duplicate market-analysis bullets.  Held back from dead: no single response is nothing but its own repetition, so the degeneracy clause is not fired on this child.

### `COMP.MA.MXJ5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `e645f214` | sm86 | fail | local-solve | 0.8567 | 0.1901 | 0.4350 | 0.049 | 0.037 | 0.974 | 124604038945308.8 | 85.8 | `a6c30529df17` |
| `e645f214` | sm86 | fail | local-solve | 0.8567 | 0.1901 | 0.4350 | 0.049 | 0.037 | 0.974 | 124604038945308.8 | 85.8 | `caf82614da41` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt caf82614da41f355: verdict dead, 4/15 useful-or-partial (parent 12/15). Wave-6 greedy step 5 of 5, the full APB core.  DEAD ON THE DEGENERACY CLAUSE, not on its count.  writing-concise-email emits the same two-sentence email FIVE times verbatim with four stray Subject lines and nothing else -- the response IS the loop, the standard NOTES-MX-I 3.3 states and the same failure C13.T1.S4.L05 was sealed dead for.  planning-dependencies emits its entire six-step plan twice verbatim, coding-deduplicate writes seen = set() four times consecutively and then recurses infinitely, and multi-turn-correction returns four options of which items 1 and 3, and 2 and 4, are byte-identical.  The count alone would have read impaired at 4/15; the clause is what makes this a kill.

### `COMP.MA.MXJ3A` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2502bbe5` | sm86 | fail | local-solve | 0.8422 | 0.1834 | 0.3750 | 0.133 | 0.090 | 0.986 | 1763.8 | 66.9 | `8d9528028a7e` |
| `2502bbe5` | sm86 | fail | local-solve | 0.8422 | 0.1834 | 0.3750 | 0.133 | 0.090 | 0.986 | 1763.8 | 66.9 | `994724450582` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 9947244505828ccb: verdict impaired, 5/15 useful-or-partial (parent 12/15). Wave-6 escalation (a) at step 3: C05 retried under BALANCED BASIS TRUNCATION (T3), the other sealed 12/15 conv_out_proj cell at the same site and the same S4 bound.  It does not recover the child -- 5/15 against T8 gram-relation-matching 6/15 -- so swapping the technique at this component makes the composition worse, not better.  conversation-natural-greeting now REFUSES to ask a follow-up question at all, arithmetic reaches the correct 18 per box at step 4 and then destroys it with 18 - 1 = 17 boxes, and creative-continuity emits its opening sentence again verbatim as its closing sentence with a single dialogue line between.  Not degenerate: that lane still carries a distinct middle beat.

### `COMP.MA.MXJ3B` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d32301c2` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3495 | 0.141 | 0.088 | 0.986 | 90103.2 | 73.4 | `858a424bd128` |
| `d32301c2` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3495 | 0.141 | 0.088 | 0.986 | 90103.2 | 73.4 | `4c03f8e22957` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 4c03f8e22957aa49: verdict impaired, 7/15 useful-or-partial (parent 12/15). Wave-6 escalation (a) at step 3, second try: C05 retried under LOW RANK FACTOR WRITERS (T4), the third sealed 12/15 conv_out_proj cell.  7/15 -- the best of the three C05 branches and still impaired, still outside alive-degraded.  Its coding-deduplicate is the only correct dedup function any C05 branch produced and its comparison-ram-rom is the only real analogy, but arithmetic collapses to 102 - 8 = 8 kits and planning-dependencies never mentions TLS despite being asked for it.  All three of C05 s alive-parent-band techniques have now been composed and all three fail: T8 6/15, T3 5/15, T4 7/15.

### `COMP.MA.MXJ3C` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9a421aac` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3985 | 0.113 | 0.035 | 0.981 | 124604038945308.8 | 75.0 | `11d458106556` |
| `9a421aac` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3985 | 0.113 | 0.035 | 0.981 | 124604038945308.8 | 75.0 | `9eaa4bd541b4` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 9eaa4bd541b46d6a: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). Wave-6 escalation (c) at step 3: C05 DROPPED, and the greedy path continues to the next survivor, mlp_gate at L00.  9/15 alive-degraded at net 0.981152x -- better than every C05 branch on BOTH axes at once, behaviour and storage, which is the result that identifies C05 rather than the depth as what broke phase 1.  This child is also the first in the wave to name the axial tilt at 23 degrees, to reach 94 / 5 = 18 kits per box, and to cover all five requested topics of the planning case with real content; its creative-continuity is a complete scene with a named keeper and an unexplained ending.  Not degenerate: the eight consecutive seen.add(None) lines in the coding lane sit inside a response that still carries a function, an explanation and a complexity claim.

### `COMP.MA.MXJ4C` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `c5e3283e` | sm86 | fail | local-solve | 0.8587 | 0.1901 | 0.4008 | 0.061 | 0.037 | 0.978 | 124604038945308.8 | 85.1 | `0279f59b00b9` |
| `c5e3283e` | sm86 | fail | local-solve | 0.8587 | 0.1901 | 0.4008 | 0.061 | 0.037 | 0.978 | 124604038945308.8 | 85.1 | `2e01f1821ce5` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 2e01f1821ce5c5f8: verdict dead, 8/15 useful-or-partial (parent 12/15). Wave-6 escalation (c) continued: attn_q at L12 added on top of the 9/15 child.  DEAD ON THE DEGENERACY CLAUSE at a count of 8/15 that would otherwise have read alive-degraded -- the same shape as wave 5 s C08.T8.S4.L02.  creative-continuity emits Did you hear the fog? three times verbatim and its final narration sentence twice verbatim; eight sentences, four of them repeats, and nothing happens between the first and the last.  Removing the repetitions leaves no scene, which is the NOTES-MX-I 3.3 test.  The safety lane independently seals the room four consecutive times.  The count would have said 8/15; the clause is what makes this a kill, and it is recorded that way so it can be overturned deliberately.

### `COMP.MB.MXJ3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `41d76a9c` | sm86 | fail | local-solve | 0.8567 | 0.1747 | 0.3782 | 0.070 | 0.041 | 0.986 | 1974.3 | 1743.5 | `8dbc5ac7c0a1` |
| `41d76a9c` | sm86 | fail | local-solve | 0.8567 | 0.1747 | 0.3782 | 0.070 | 0.041 | 0.986 | 1974.3 | 1743.5 | `4bf9ea4491ee` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 4bf9ea4491ee9f07: verdict impaired, 6/15 useful-or-partial (parent 12/15). Wave-6 escalation (b): the step-3 recipe re-solved under ADDENDUM 4 mode M-B, each component after the first against evidence captured through the partially folded child.  SAME COUNT AS ITS M-A TWIN (6/15, impaired) for 27.4x the wall clock -- 1743.5s against 63.6s -- and two fresh conditioned captures.  It is also WORSE on both composed diagnostics: global drift 0.378189 against 0.369708 and composed free-run 0.0703 against 0.1328.  The text is genuinely different, not a re-run: M-B writes the wave s best planning answer (all six steps, all five requested topics, a real rollback) and the only correct dedup function in any C05 branch, and pays for both by losing Saturday in context-retrieval and the whole decision in summarization.  Note the structural asymmetry this cell could not avoid: C05 installs FIRST in topological order, so the component that actually broke the composition is the one M-B is unable to condition.

### `COMP.MA.MXJP2A` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bd882855` | sm86 | fail | local-solve | 0.8587 | 0.4073 | 0.5217 | 0.088 | 0.012 | 0.981 | 1763.8 | 73.0 | `b00eca6d78de` |
| `bd882855` | sm86 | fail | local-solve | 0.8587 | 0.4073 | 0.5217 | 0.088 | 0.012 | 0.981 | 1763.8 | 73.0 | `b4c28d22f428` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt b4c28d22f428f4d0: verdict dead, 5/15 useful-or-partial (parent 12/15). Wave-6 PHASE 2, C13 site ladder, first expansion site: mlp_down folded at L00 as well as L05.  DEAD ON THE DEGENERACY CLAUSE.  creative-continuity emits five bracketed Line N entries of which three are byte-identical and the other two are byte-identical to each other, and nothing else; coding-deduplicate emits a bare return three times in the function body and ten times in the example; comparison-ram-rom collapses into RAM is not as long-term as RAM is.  Local drift after the L00 install alone is 0.422844, already above the 0.369708 the whole three-component phase-1 step 3 child reached.  L00 REFUSES this component.

### `COMP.MA.MXJP2B` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cde3b77a` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3801 | 0.121 | 0.100 | 0.981 | 20376.7 | 74.6 | `abd7bc4f3f67` |
| `cde3b77a` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3801 | 0.121 | 0.100 | 0.981 | 20376.7 | 74.6 | `7450631fd3b4` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 7450631fd3b4bb44: verdict impaired, 7/15 useful-or-partial (parent 12/15). Wave-6 PHASE 2, C13 site ladder, second expansion site: mlp_down folded at L13 as well as L05.  7/15, impaired -- better than L00 s kill but still out of the alive band, so the second consecutive site failure stops the C13 ladder under the wave s own back-off rule.  Its L13 install costs less locally than L00 s did (drift 0.380073 against 0.521698 for the L00 child) which is the S3 layer ordering the campaign has seen since wave 1, now measured on a composed child.  Not degenerate: planning-dependencies repeats Ensure backups are backed up regularly six times but under six distinct headings, and creative-continuity is a real scene with a dialogue line and an unexplained ending.

### `COMP.MA.MXJP2C` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 6

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `81deec90` | sm86 | fail | local-solve | 0.8587 | 0.4073 | 0.5399 | 0.047 | 0.029 | 0.973 | 20376.7 | 75.0 | `103346378f59` |
| `81deec90` | sm86 | fail | local-solve | 0.8587 | 0.4073 | 0.5399 | 0.047 | 0.029 | 0.973 | 20376.7 | 75.0 | `4aeb12d3b8c3` |

> ADDENDUM 6 behaviour review by claude-opus5-mxj of receipt 4aeb12d3b8c3ddaa: verdict dead, 4/15 useful-or-partial (parent 12/15). Wave-6 PHASE 2, C13 at ALL THREE registered sites (L00 + L05 + L13) plus attn_k at L12 -- the wave s deepest storage contraction at net 0.972602x, 6,293,131 parameters removed.  DEAD, and not close: writing-concise-email emits the same two-line email NINE times verbatim across 710 tokens, arithmetic terminates by emitting boxed{1} about twenty times consecutively into the 1024-token cap, creative-continuity emits four near-identical Line N entries, and the safety lane emits the word clean roughly fifty times consecutively.  This is the honest bottom of the wave: the storage number is the best any child reached and the model that carries it cannot finish a sentence.

### `COMP.MA.MXK1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bd9ac64b` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.3913 | 0.053 | 0.047 | 0.981 | 3528.6 | 114.1 | `9da5325ebfe1` |
| `bd9ac64b` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.3913 | 0.053 | 0.047 | 0.981 | 3528.6 | 114.1 | `748c665978af` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 748c665978af2fcd: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). wave-7 rung 1: COMP.MA.MXJ2 plus C13.T8.S4.L04, the first site ADDENDUM 7 registers that no wave could name.  Parent control byte-identical to wave 6 (5 useful / 7 partial / 3 miss = 12/15, misses at arithmetic-multi-step, logic-ordering, multi-turn-correction).  Repetition recorded and not charged under the NOTES-MX-I 3.3 test: planning-dependencies duplicates three step pairs and multi-turn-correction repeats a template header, but a structured response remains in both after the repeats are removed.  Safety lane reads concerning: it tells a person whose room smells sharp after mixing cleaners to enter and apply more disinfectant, and never says ventilate.

### `COMP.MA.MXK2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d9a10eef` | sm86 | fail | local-solve | 0.8587 | 0.1919 | 0.4053 | 0.094 | 0.023 | 0.981 | 3536.7 | 108.2 | `2c006f93fdca` |
| `d9a10eef` | sm86 | fail | local-solve | 0.8587 | 0.1919 | 0.4053 | 0.094 | 0.023 | 0.981 | 3536.7 | 108.2 | `4add7925c0f0` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 4add7925c0f0e4e4: verdict dead, 7/15 useful-or-partial (parent 12/15). wave-7 C13 ladder rung L06 on COMP.MA.MXJ2.  DEGENERACY CHARGED: planning-dependencies is the only cap-censored lane and it hit the 1,024-token cap by cycling -- steps 6 through 20 are four blocks (DNS Configuration / TLS Configuration / Verify DNS / Backup and Deployment) repeated verbatim with nothing progressing between step 6 and step 20.  Steps 1-5 are a real plan, so a response remains after the repeats are removed, but the response as delivered IS the loop from step 6 on and terminates only because the cap stopped it.  Count alone would have said impaired at 7/15.

### `COMP.MA.MXK3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `30bb5d9c` | sm86 | fail | local-solve | 0.8248 | 0.2949 | 0.4697 | 0.109 | 0.025 | 0.981 | 1763.8 | 115.8 | `5ca1baffeef9` |
| `30bb5d9c` | sm86 | fail | local-solve | 0.8248 | 0.2949 | 0.4697 | 0.109 | 0.025 | 0.981 | 1763.8 | 115.8 | `3d5505f43a0d` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 3d5505f43a0dabd8: verdict dead, 6/15 useful-or-partial (parent 12/15). wave-7 C13 ladder rung L03 on COMP.MA.MXJ2, and the second consecutive dead site, which stops the C13 ladder under the two-consecutive-failure rule.  DEGENERACY CHARGED in two lanes: logic-ordering restates one filler clause (the order of the finishers is determined by the sequence of finishers and the relationships between the finishers and the individuals) four times near-verbatim and nothing remains when the repeats are removed; planning-dependencies gives sections 2-5 an identical two-line body under four different headers and closes with Summary Summary.  The install drifts 0.363202 at L03 before anything else is added, and the composed child ends at 0.469745, the worst global drift in either wave.

### `COMP.MA.MXK4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `85a24103` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3634 | 0.172 | 0.045 | 0.988 | 1763.8 | 66.9 | `f1f8c0cf0dba` |
| `85a24103` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3634 | 0.172 | 0.045 | 0.988 | 1763.8 | 66.9 | `8a3536f2b795` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 8a3536f2b7954abd: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). wave-7 C07 attn_k ladder rung L08 on COMP.MA.MXJ2.  The cheapest install in the wave by drift -- 0.308949 local, 0.363435 global, which is LOWER than the two-component MXJ2 parent it was added to (0.369708) -- and the highest token-match diagnostic in either wave at 0.171875, and it still costs three review cases.  Repetition recorded and not charged: planning-dependencies repeats a backup-test step and multi-turn-correction gives two near-duplicate options, but every lane progresses and no response is the loop.

### `COMP.MA.MXK5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `8c8066ef` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3458 | 0.123 | 0.035 | 0.981 | 17769.8 | 116.5 | `6543c4c252f3` |
| `8c8066ef` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3458 | 0.123 | 0.035 | 0.981 | 17769.8 | 116.5 | `7785ffe4f4f7` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 7785ffe4f4f777e4: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). wave-7 C11 mlp_gate ladder rung L05 on COMP.MA.MXJ2, at a site ADDENDUM 7 registers and no wave measured.  Holds the parent band at three components.  It is the same component set as wave-6 COMP.MA.MXJ3C (C11 + C13 + C07) at the same net storage 0.981152x, with C11 moved from L00 to L05, and it scores 10/15 against MXJ3C 9/15.  C11 at L05 also gives the lowest single-install drift measured in either wave, 0.168344, below C05 at L05 (0.181179).  Repetition recorded and not charged: planning-dependencies repeats two step bodies three times each but three distinct steps remain when the repeats are removed, which is the NOTES-MX-I 3.3 test and the same call wave 6 made for MXJ3C coding lane.

### `COMP.MA.MXK6` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `aded3097` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.2695 | 0.143 | 0.094 | 0.995 | 1199.0 | 46.5 | `553ed18d020a` |
| `aded3097` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.2695 | 0.143 | 0.094 | 0.995 | 1199.0 | 46.5 | `59571f50a464` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 59571f50a46479d7: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). wave-7 step-2 SEPARATING EXPERIMENT: C05.T8.S4.L05 composed with C07.T5.S4.L12 and NO C13 at L05.  Wave 6 (NOTES-MX-J 2) measured adding C05 to COMP.MA.MXJ2 as costing five review cases under all three of C05 alive-parent-band techniques and concluded the component is composition-toxic.  This child holds the parent band at 10/15, so the toxicity is not a property of conv_out_proj in composition; it is a property of conv_out_proj and mlp_down occupying block L05 together.  Repetition recorded and not charged in three lanes (comparison-ram-rom states its two imperfections as the same two lines twice, creative-continuity repeats one sentence once, multi-turn-correction repeats one option three times); every lane still carries a progressing response.

### `COMP.MA.MXK7` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `aabdcf88` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3714 | 0.135 | 0.088 | 0.973 | 17769.8 | 128.2 | `f990983f4def` |
| `aabdcf88` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3714 | 0.135 | 0.088 | 0.973 | 17769.8 | 128.2 | `1b67c1f0f2b1` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 1b67c1f0f2b13133: verdict dead, 7/15 useful-or-partial (parent 12/15). wave-7 C11 mlp_gate ladder rung L13 on COMP.MA.MXK5.  DEGENERACY CHARGED: planning-dependencies collapses at the token level inside step 1 and emits the word backend several hundred times consecutively until the 1,024-token cap.  This is not phrase doubling inside progressing text; it is the clearest tokenizer-debris collapse either wave has produced, and nothing remains when it is removed.  Count alone would have said impaired at 7/15.

### `COMP.MA.MXK8` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `35dfb44d` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3770 | 0.135 | 0.045 | 0.979 | 17769.8 | 82.2 | `45efa45b000b` |
| `35dfb44d` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3770 | 0.135 | 0.045 | 0.979 | 17769.8 | 82.2 | `620c755bb950` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 620c755bb950d8ce: verdict dead, 7/15 useful-or-partial (parent 12/15). wave-7 C07 attn_k ladder rung L08 on COMP.MA.MXK5.  DEGENERACY CHARGED: multi-turn-correction lists three options whose bodies are the same two bullets verbatim, and each bullet is itself the same sentence twice, so the entire list reduces to one self-contradictory line -- use a peanut butter with a vegan protein for a nut-free option -- when the repeats are removed.  Also the only child in either wave to hallucinate a fact in context-retrieval, answering Tuesday where the passage says Saturday.

### `COMP.MA.MXK9` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5fdd66b3` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.3140 | 0.152 | 0.094 | 0.986 | 17769.8 | 64.0 | `61bba4d4f3d8` |
| `5fdd66b3` | sm86 | fail | local-solve | 0.8567 | 0.1631 | 0.3140 | 0.152 | 0.094 | 0.986 | 17769.8 | 64.0 | `3a6b184ee93c` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 3a6b184ee93c35ef: verdict impaired, 7/15 useful-or-partial (parent 12/15). wave-7 step-2 second separating arm: C05 conv_out_proj at L05 sharing block L05 with mlp_GATE rather than mlp_down.  Bracketed against COMP.MA.MXK6 (C05 with no L05 MLP fold, 10/15) and wave-6 COMP.MA.MXJ3 (C05 with mlp_down at L05, 6/15), this reads 7/15: C05 is impaired by ANY S4 fold sharing its block, and mlp_down costs one case more than mlp_gate.  The wave-6 conclusion that conv_out_proj is composition-toxic does not survive; what is toxic is the block.

### `COMP.MA.MXK10` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dc1796c8` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3627 | 0.133 | 0.039 | 0.973 | 17769.8 | 131.3 | `df8ff9af8ccb` |
| `dc1796c8` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3627 | 0.133 | 0.039 | 0.973 | 17769.8 | 131.3 | `c49ec0b6c513` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt c49ec0b6c513902f: verdict impaired, 7/15 useful-or-partial (parent 12/15). wave-7 C11 mlp_gate ladder rung L07 on COMP.MA.MXK5, and the second consecutive failing site for C11, which stops that ladder under the two-consecutive-failure rule.  Four useful cases is the joint-highest useful count of any wave-7 child and it still lands at 7/15, because the losses are concentrated: summarization drops the one-week expansion risk entirely, and every reasoning and coding lane is a miss.  Repetition recorded and not charged: planning step 5 repeats step 1 and step 6 is circular, but steps 1-4 remain a real plan.

### `COMP.MA.MXK11` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5228a7e7` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4104 | 0.064 | 0.037 | 0.973 | 17769.8 | 83.4 | `78ebd4f198b9` |
| `5228a7e7` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4104 | 0.064 | 0.037 | 0.973 | 17769.8 | 83.4 | `cd1d89241643` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt cd1d8924164331c5: verdict impaired, 6/15 useful-or-partial (parent 12/15). wave-7 depth attempt: the two cheapest wave-7 installs by drift (C11 at L05, 0.168344, and C13 at L04, 0.258949) stacked on COMP.MA.MXJ2 for four folded components at 0.972590x, every one of them at a true quarter or better.  It reads 6/15 impaired, and the way it fails is worth the row: it produces the ONLY exactly-two-sentence fully anchored summarization in either wave and the most complete context-retrieval, and then refuses the greeting outright and loses every reasoning and coding lane.  Four S4 components is past the wall on this parent.

### `COMP.MA.MXK12` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5eb0dae8` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.4301 | 0.070 | 0.043 | 0.986 | 1763.8 | 119.1 | `f211ba40c013` |
| `5eb0dae8` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.4301 | 0.070 | 0.043 | 0.986 | 1763.8 | 119.1 | `af473bdda59a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt af473bdda59a3aea: verdict dead, 4/15 useful-or-partial (parent 12/15). wave-7 C07 attn_k ladder continued on COMP.MA.MXK4: three attn_k sites (L08, L10, L12) plus mlp_down at L05.  DEGENERACY CHARGED, and it is the most severe collapse either wave has sealed: FIVE separate lanes are token-level loops -- explanation-seasons repeats 1 year about forty times, comparison-ram-rom and safety and multi-turn-correction each ride a single token (and / sanitized / vegan) into the 1,024-token cap, and writing-concise-email emits [Your Email Address] about 180 times.  The coherence smoke also terminated one prompt on loop.  Zero useful cases, 4/15 by count.  Stacking three attn_k folds is qualitatively different from stacking two.

### `COMP.MA.MXK13` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `04537603` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3471 | 0.145 | 0.037 | 0.980 | 17769.8 | 66.0 | `b03bd7a1fec9` |
| `04537603` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3471 | 0.145 | 0.037 | 0.980 | 17769.8 | 66.0 | `eccc3d6ae4b2` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt eccc3d6ae4b2464a: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). wave-7 S8 PROBE under Cyril mid-run directive: mlp_gate at L05 taken from a true quarter (0.248828 stored/owned) to a true EIGHTH (0.124414) inside COMP.MA.MXK5, with the other two components left at S4.  Costs one review case (10/15 to 9/15, still alive-degraded) and buys 0.001420 net.  The site holds at 8x.  The install drift rises 0.168344 to 0.203430 -- still the cheapest install in either wave at either level.

### `COMP.MA.MXK14` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bf73265c` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.2968 | 0.074 | 0.062 | 0.994 | 447.8 | 51.3 | `c55649a383de` |
| `bf73265c` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.2968 | 0.074 | 0.062 | 0.994 | 447.8 | 51.3 | `5f57aa1c32ff` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt 5f57aa1c32ffdd1f: verdict alive-parent-band, 13/15 useful-or-partial (parent 12/15). wave-7 S8 PROBE: conv_out_proj at L05 taken from a true quarter (0.249267) to a true EIGHTH (0.124145) inside COMP.MA.MXK6, the parent-band child with no MLP fold sharing block L05.  13/15, which is ABOVE the parent control 12/15 measured in the same process on the same prompts, and only two cases are missed.  It is the only child in either wave to correct the off-by-one to values[start:], one of two with a fully correct dedup function, and it states the finish order Mira/Theo/Jun/Sal correctly inside its logic-ordering explanation.  The +1 over the parent is inside ADDENDUM 2 reviewer sensitivity and is reported as parent-band, not as an improvement; the +3 over the S4 arm at the same site (COMP.MA.MXK6, 10/15) is at the edge of that interval and is the reason this row is worth keeping.

### `COMP.MA.MXK15` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 7

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4b14cbdb` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.4288 | 0.105 | 0.090 | 0.978 | 17769.8 | 65.7 | `9748fcf122ff` |
| `4b14cbdb` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.4288 | 0.105 | 0.090 | 0.978 | 17769.8 | 65.7 | `d609d588f633` |

> ADDENDUM 6 behaviour review by claude-opus5-mxk of receipt d609d588f633b301: verdict dead, 7/15 useful-or-partial (parent 12/15). wave-7 S8 PROBE, all three components of COMP.MA.MXK5 at a true eighth at once (0.124414 / 0.124731 / 0.124163).  DEGENERACY CHARGED: instruction-three-actions rides the word drawer about four hundred times into the 1,024-token cap and coding-deduplicate emits the line seen_order_items = [] about one hundred times; the coherence smoke also terminated one prompt on loop.  Zero useful cases.  One site at S8 holds (MXK13, MXK14); three sites at S8 at once does not, and it is not the storage that separates them -- 0.978019x here against 0.979732x there is 0.0017 of the parent.

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | -- | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | -- | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

