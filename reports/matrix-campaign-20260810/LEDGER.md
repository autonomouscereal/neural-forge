# Stage-1 matrix campaign -- ledger

Rendered 2026-08-13T12:33:11+00:00 from `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`.

**309 cells sealed** (652 ledger rows including re-runs); newest seal 2026-08-13T12:22:39.824+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 18 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `fail` | 278 | crossed a chain threshold; see first_divergent_metric |
| `infra` | 13 | calibration refused the cell -- void, not evidence |

## 0. ADDENDUM 6 -- behaviour verdicts (the primary grade)

ADDENDUM 6 §6.1: the grade of a cell is the frozen 15-prompt review of the **whole model** with that cell's component folded and every other component copy_exact, scored against the parent's own 12/15 on the same surface in the same process.  `token_match_vs_control` is a diagnostic column here and gates nothing -- wave 4 sealed a child at 0.551 token match and 12/15, the parent's own band.

| verdict | cells | meaning |
|---|---:|---|
| `alive-parent-band` | 40 | within 2 cases of the parent's own 12/15 on the frozen surface, coherent, emit-zero floor passed |
| `alive-degraded` | 23 | >= the A9 band of 8/15, outside the parent's +/-2 |
| `impaired` | 13 | < 8/15 but still coherent, on-task language |
| `dead` | 133 | incoherent, emit-zero floor failed, or degenerate output |
| `cannot-express-4x` | 1 | the technique cannot state this component at the level's storage bound; nothing was measured about the model |

| cell | site | level | stored/parent | bound | verdict | review | parent | smoke | token-match (diag) |
|---|---|---|---:|---|---|---:|---:|---|---:|
| `ARCH.S640` | GLOBAL | NA | 0.6405 | -- | **dead** | 0/15 | 12 | coherent | 0.0273 |
| `ARCH.CL640` | GLOBAL | NA | 0.6405 | -- | **dead** | 0/15 | 12 | coherent | 0.0156 |
| `ARCH.CT640` | GLOBAL | NA | 0.6405 | -- | **dead** | 0/15 | 12 | coherent | 0.0020 |
| `ARCH.HD16` | GLOBAL | NA | 0.9384 | -- | **dead** | 0/15 | 12 | coherent | 0.0234 |
| `ARCH.A552` | GLOBAL | NA | 0.5520 | -- | **dead** | 0/15 | 12 | coherent | 0.0078 |
| `ARCH.CL896` | GLOBAL | NA | 0.6884 | -- | **dead** | 4/15 | 12 | coherent | 0.0176 |
| `ARCH.X05T` | GLOBAL | NA | 0.9475 | -- | **dead** | 0/15 | 12 | coherent | 0.0156 |
| `ARCH.X05A` | GLOBAL | NA | 0.9475 | -- | **dead** | 6/15 | 12 | coherent | 0.0293 |
| `ARCH.W14FF1280` | GLOBAL | NA | 0.7603 | -- | **dead** | 5/15 | 12 | coherent | 0.0293 |
| `ARCH.W14FF640` | GLOBAL | NA | 0.6405 | -- | **dead** | 0/15 | 12 | coherent | 0.0215 |
| `ARCH.W14FF320` | GLOBAL | NA | 0.5806 | -- | **dead** | 0/15 | 12 | coherent | 0.0078 |
| `ARCH.R0` | GLOBAL | NA | 0.7249 | -- | **dead** | 0/15 | 12 | coherent | 0.0273 |
| `ARCH.KV6` | GLOBAL | NA | 0.9795 | -- | **dead** | 5/15 | 12 | coherent | 0.0508 |
| `ARCH.R1` | GLOBAL | NA | 0.6895 | -- | **dead** | 0/15 | 12 | coherent | 0.0156 |
| `ARCH.BX13` | GLOBAL | NA | 0.7249 | -- | **dead** | 1/15 | 12 | coherent | 0.0078 |
| `ARCH.B1536` | GLOBAL | NA | 0.7694 | -- | **dead** | 0/15 | 12 | coherent | 0.0176 |
| `ARCH.D2` | GLOBAL | NA | 0.8950 | -- | **dead** | 2/15 | 12 | coherent | 0.0371 |
| `ARCH.W1536X13` | GLOBAL | NA | 0.7694 | -- | **dead** | 4/15 | 12 | coherent | 0.0176 |
| `ARCH.X13KV6` | GLOBAL | NA | 0.9269 | -- | **dead** | 6/15 | 12 | coherent | 0.0449 |
| `ARCH.D2B` | GLOBAL | NA | 0.8950 | -- | **dead** | 6/15 | 12 | coherent | 0.0410 |
| `ARCH.X01` | GLOBAL | NA | 0.9475 | -- | **dead** | 9/15 | 12 | coherent | 0.0391 |
| `ARCH.X03` | GLOBAL | NA | 0.9475 | -- | **impaired** | 6/15 | 12 | coherent | 0.0977 |
| `ARCH.X07` | GLOBAL | NA | 0.9475 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.1328 |
| `ARCH.X09` | GLOBAL | NA | 0.9475 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.0703 |
| `ARCH.X11` | GLOBAL | NA | 0.9475 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.0898 |
| `ARCH.X13A` | GLOBAL | NA | 0.9475 | -- | **dead** | 13/15 | 12 | coherent | 0.0430 |
| `ARCH.X12` | GLOBAL | NA | 0.9521 | -- | **dead** | 0/15 | 12 | coherent | 0.0156 |
| `ARCH.D3` | GLOBAL | NA | 0.8424 | -- | **dead** | 0/15 | 12 | coherent | 0.0391 |
| `ARCH.R2` | GLOBAL | NA | 0.6587 | -- | **dead** | 0/15 | 12 | coherent | 0.0020 |
| `ARCH.R3` | GLOBAL | NA | 0.6416 | -- | **dead** | 0/15 | 12 | coherent | 0.0098 |
| `ARCH.R4` | GLOBAL | NA | 0.6039 | -- | **dead** | 0/15 | 12 | coherent | 0.0059 |
| `ARCH.R5` | GLOBAL | NA | 0.5799 | -- | **dead** | 0/15 | 12 | coherent | 0.0254 |
| `ARCH.X00` | GLOBAL | NA | 0.9475 | -- | **dead** | 0/15 | 12 | coherent | 0.0371 |
| `ARCH.X02` | GLOBAL | NA | 0.9521 | -- | **dead** | 0/15 | 12 | coherent | 0.0176 |
| `ARCH.X04` | GLOBAL | NA | 0.9521 | -- | **dead** | 2/15 | 12 | coherent | 0.0195 |
| `ARCH.X06` | GLOBAL | NA | 0.9521 | -- | **dead** | 8/15 | 12 | coherent | 0.0312 |
| `ARCH.X08` | GLOBAL | NA | 0.9521 | -- | **dead** | 7/15 | 12 | coherent | 0.0391 |
| `ARCH.X10` | GLOBAL | NA | 0.9521 | -- | **impaired** | 7/15 | 12 | coherent | 0.0625 |
| `ARCH.P1307` | GLOBAL | NA | 0.8950 | -- | **dead** | 6/15 | 12 | coherent | 0.0293 |
| `ARCH.P0709` | GLOBAL | NA | 0.8950 | -- | **dead** | 4/15 | 12 | coherent | 0.0410 |
| `ARCH.P1301` | GLOBAL | NA | 0.8950 | -- | **dead** | 7/15 | 12 | coherent | 0.0293 |
| `ARCH.P0711` | GLOBAL | NA | 0.8950 | -- | **dead** | 6/15 | 12 | coherent | 0.0508 |
| `ARCH.P0911` | GLOBAL | NA | 0.8950 | -- | **dead** | 2/15 | 12 | coherent | 0.0566 |
| `ARCH.P1303` | GLOBAL | NA | 0.8950 | -- | **dead** | 7/15 | 12 | coherent | 0.0586 |
| `ARCH.KV6J` | GLOBAL | NA | 0.9795 | -- | **dead** | 3/15 | 12 | coherent | 0.0566 |
| `ARCH.KV6A` | GLOBAL | NA | 0.9795 | -- | **dead** | 5/15 | 12 | coherent | 0.0391 |
| `ARCH.KV6JQ` | GLOBAL | NA | 0.9795 | -- | **dead** | 8/15 | 12 | coherent | 0.0430 |
| `ARCH.F1301KV6` | GLOBAL | NA | 0.8744 | -- | **dead** | 0/15 | 12 | coherent | 0.0430 |
| `ARCH.X13KV6J` | GLOBAL | NA | 0.9269 | -- | **dead** | 3/15 | 12 | coherent | 0.0449 |
| `ARCH.X13KV6JQ` | GLOBAL | NA | 0.9269 | -- | **dead** | 4/15 | 12 | coherent | 0.0332 |
| `ARCH.KV4JQ` | GLOBAL | NA | 0.9589 | -- | **dead** | 0/15 | 12 | coherent | 0.0449 |
| `ARCH.F1301KV6Q` | GLOBAL | NA | 0.8744 | -- | **dead** | 0/15 | 12 | coherent | 0.0430 |
| `ARCH.F1301W1536` | GLOBAL | NA | 0.7306 | -- | **dead** | 0/15 | 12 | coherent | 0.0391 |
| `ARCH.T010713` | GLOBAL | NA | 0.8424 | -- | **dead** | 3/15 | 12 | coherent | 0.0469 |
| `ARCH.T010913` | GLOBAL | NA | 0.8424 | -- | **dead** | 1/15 | 12 | coherent | 0.0137 |
| `ARCH.T030713` | GLOBAL | NA | 0.8424 | -- | **dead** | 0/15 | 12 | coherent | 0.0371 |
| `ARCH.T030913` | GLOBAL | NA | 0.8424 | -- | **dead** | 2/15 | 12 | coherent | 0.0254 |
| `ARCH.KV6JS` | GLOBAL | NA | 0.9795 | -- | **dead** | 7/15 | 12 | coherent | 0.0488 |
| `ARCH.KV4JS` | GLOBAL | NA | 0.9589 | -- | **dead** | 0/15 | 12 | coherent | 0.0449 |
| `ARCH.X07KV6JQ` | GLOBAL | NA | 0.9269 | -- | **dead** | 6/15 | 12 | coherent | 0.0371 |
| `ARCH.X07KV6JQC` | GLOBAL | NA | 0.9269 | -- | **dead** | 7/15 | 12 | coherent | 0.0566 |
| `ARCH.X09KV6JQC` | GLOBAL | NA | 0.9269 | -- | **dead** | 5/15 | 12 | coherent | 0.0449 |
| `ARCH.F1301KV6QC` | GLOBAL | NA | 0.8744 | -- | **dead** | 1/15 | 12 | coherent | 0.0195 |
| `ARCH.F1301W1536C` | GLOBAL | NA | 0.7306 | -- | **dead** | 0/15 | 12 | coherent | 0.0176 |
| `ARCH.T010713KV6JQC` | GLOBAL | NA | 0.8219 | -- | **dead** | 0/15 | 12 | coherent | 0.0488 |
| `ARCH.T010713W1536C` | GLOBAL | NA | 0.6918 | -- | **dead** | 0/15 | 12 | coherent | 0.0215 |
| `ARCH.T010713W1280C` | GLOBAL | NA | 0.6541 | -- | **dead** | 0/15 | 12 | coherent | 0.0137 |
| `ARCH.W896` | GLOBAL | NA | 0.8590 | -- | **dead** | 4/15 | 12 | coherent | 0.0215 |
| `ARCH.W768` | GLOBAL | NA | 0.7226 | -- | **dead** | 3/15 | 12 | coherent | 0.0098 |
| `ARCH.W640` | GLOBAL | NA | 0.5908 | -- | **dead** | 0/15 | 12 | coherent | 0.0098 |
| `ARCH.W512` | GLOBAL | NA | 0.4635 | -- | **dead** | 0/15 | 12 | coherent | 0.0039 |
| `ARCH.W896X13` | GLOBAL | NA | 0.8151 | -- | **dead** | 1/15 | 12 | coherent | 0.0273 |
| `ARCH.W896KV6JS` | GLOBAL | NA | 0.8410 | -- | **dead** | 2/15 | 12 | coherent | 0.0156 |
| `ARCH.Q01050913` | GLOBAL | NA | 0.7899 | -- | **dead** | 0/15 | 12 | coherent | 0.0312 |
| `ARCH.W768AW25` | GLOBAL | NA | 0.7226 | -- | **dead** | 0/15 | 12 | coherent | 0.0137 |
| `ARCH.W768AW100` | GLOBAL | NA | 0.7226 | -- | **dead** | 0/15 | 12 | coherent | 0.0156 |
| `ARCH.W768EV` | GLOBAL | NA | 0.7226 | -- | **dead** | 1/15 | 12 | coherent | 0.0117 |
| `ARCH.W768RG` | GLOBAL | NA | 0.7226 | -- | **dead** | 0/15 | 12 | coherent | 0.0000 |
| `ARCH.W768LS` | GLOBAL | NA | 0.7226 | -- | **dead** | 3/15 | 12 | coherent | 0.0117 |
| `ARCH.W768ST` | GLOBAL | NA | 0.7226 | -- | **dead** | 2/15 | 12 | coherent | 0.0137 |
| `ARCH.W704` | GLOBAL | NA | 0.6561 | -- | **dead** | 0/15 | 12 | coherent | 0.0195 |
| `ARCH.W704AW25` | GLOBAL | NA | 0.6561 | -- | **dead** | 0/15 | 12 | coherent | 0.0059 |
| `ARCH.W736` | GLOBAL | NA | 0.6892 | -- | **dead** | 1/15 | 12 | coherent | 0.0098 |
| `ARCH.W736LS` | GLOBAL | NA | 0.6892 | -- | **dead** | 0/15 | 12 | coherent | 0.0137 |
| `ARCH.MM768` | GLOBAL | NA | 0.7226 | -- | **dead** | 0/15 | 12 | coherent | 0.0117 |
| `ARCH.W896LS` | GLOBAL | NA | 0.8590 | -- | **dead** | 3/15 | 12 | coherent | 0.0215 |
| `ARCH.W896LSX13` | GLOBAL | NA | 0.8151 | -- | **dead** | 2/15 | 12 | coherent | 0.0273 |
| `ARCH.MM896` | GLOBAL | NA | 0.8590 | -- | **dead** | 2/15 | 12 | coherent | 0.0430 |
| `ARCH.W720` | GLOBAL | NA | 0.6726 | -- | **dead** | 0/15 | 12 | coherent | 0.0137 |
| `ARCH.W896LSKV6JQ` | GLOBAL | NA | 0.8410 | -- | **dead** | 1/15 | 12 | coherent | 0.0156 |
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
| `COMP.MA.MXL1` | GLOBAL | NA | 0.9857 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.0781 |
| `COMP.MA.MXL2` | GLOBAL | NA | 0.9903 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.0996 |
| `COMP.MA.MXL4` | GLOBAL | NA | 0.9909 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.1035 |
| `COMP.MA.MXL5` | GLOBAL | NA | 0.9726 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.1289 |
| `COMP.MA.MXL7` | GLOBAL | NA | 0.9469 | -- | **dead** | 9/15 | 12 | coherent | 0.1055 |
| `COMP.MA.MXL8` | GLOBAL | NA | 0.8955 | -- | **dead** | 3/15 | 12 | coherent | 0.0566 |
| `COMP.MA.MXL3` | GLOBAL | NA | 0.9903 | -- | **dead** | 11/15 | 12 | coherent | 0.0801 |
| `COMP.MA.MXL6` | GLOBAL | NA | 0.9783 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.1211 |
| `COMP.MA.MXL9` | GLOBAL | NA | 0.9212 | -- | **dead** | 5/15 | 12 | coherent | 0.0684 |
| `COMP.MA.MXL10` | GLOBAL | NA | 0.9857 | -- | **alive-parent-band** | 11/15 | 12 | coherent | 0.1270 |
| `COMP.MA.MXL11` | GLOBAL | NA | 0.9554 | -- | **alive-degraded** | 9/15 | 12 | coherent | 0.1211 |
| `COMP.MB.MXL12` | GLOBAL | NA | 0.9857 | -- | **alive-degraded** | 8/15 | 12 | coherent | 0.1230 |
| `COMP.MA.MXM1` | GLOBAL | NA | 0.7809 | -- | **dead** | 0/15 | 12 | incoherent | 0.0000 |
| `COMP.MA.MXM2` | GLOBAL | NA | 0.7809 | -- | **dead** | 0/15 | 12 | incoherent | 0.0273 |
| `COMP.MA.MXM3` | GLOBAL | NA | 0.7808 | -- | **dead** | 3/15 | 12 | incoherent | 0.0312 |
| `COMP.MA.MXM4` | GLOBAL | NA | 0.7443 | -- | **dead** | 0/15 | 12 | incoherent | 0.0254 |
| `COMP.MA.MXM5` | GLOBAL | NA | 0.7807 | -- | **dead** | 6/15 | 12 | incoherent | 0.0801 |
| `COMP.MA.MXM6` | GLOBAL | NA | 0.7809 | -- | **dead** | 9/15 | 12 | incoherent | 0.0664 |
| `COMP.MA.MXM7` | GLOBAL | NA | 0.7808 | -- | **dead** | 5/15 | 12 | incoherent | 0.0820 |
| `COMP.MA.MXM8` | GLOBAL | NA | 0.7443 | -- | **dead** | 3/15 | 12 | incoherent | 0.0254 |
| `COMP.MA.MXM9` | GLOBAL | NA | 0.7363 | -- | **dead** | 4/15 | 12 | incoherent | 0.0195 |
| `COMP.MA.MXM10` | GLOBAL | NA | 0.7534 | -- | **dead** | 6/15 | 12 | incoherent | 0.0352 |
| `COMP.MA.MXN1` | GLOBAL | NA | 0.7440 | -- | **dead** | 7/15 | 12 | coherent | 0.0781 |
| `COMP.MA.MXN2` | GLOBAL | NA | 0.7805 | -- | **alive-parent-band** | 10/15 | 12 | coherent | 0.0723 |
| `COMP.MA.MXN3` | GLOBAL | NA | 0.7360 | -- | **dead** | 6/15 | 12 | coherent | 0.0410 |
| `COMP.MA.MXN4` | GLOBAL | NA | 0.7548 | -- | **dead** | 8/15 | 12 | coherent | 0.0430 |
| `COMP.MA.MXN5` | GLOBAL | NA | 0.7804 | -- | **dead** | 8/15 | 12 | coherent | 0.0840 |
| `COMP.MA.MXO2` | GLOBAL | NA | 0.4977 | -- | **dead** | 0/15 | 12 | incoherent | 0.0137 |
| `COMP.MA.MXO3` | GLOBAL | NA | 0.7291 | -- | **dead** | 8/15 | 12 | coherent | 0.0488 |
| `COMP.MA.MXO4` | GLOBAL | NA | 0.7034 | -- | **dead** | 5/15 | 12 | coherent | 0.0508 |
| `COMP.MA.MXP2` | GLOBAL | NA | 0.7034 | -- | **dead** | 6/15 | 12 | coherent | 0.0605 |
| `COMP.MA.MXP1` | GLOBAL | NA | 0.7125 | -- | **dead** | 7/15 | 12 | coherent | 0.0566 |
| `COMP.MA.MXQ1` | GLOBAL | NA | 0.9896 | -- | **dead** | 6/15 | 12 | coherent | 0.1367 |
| `COMP.MA.MXQ2` | GLOBAL | NA | 0.9896 | -- | **dead** | 6/15 | 12 | coherent | 0.0801 |
| `COMP.MA.MXQ3` | GLOBAL | NA | 0.9896 | -- | **dead** | 4/15 | 12 | coherent | 0.1348 |
| `COMP.MA.MXQ4` | GLOBAL | NA | 0.6826 | -- | **dead** | 0/15 | 12 | incoherent | 0.0273 |
| `COMP.MA.MXQ6` | GLOBAL | NA | 0.4978 | -- | **dead** | 0/15 | 12 | incoherent | 0.0410 |
| `COMP.MA.MXQ5` | GLOBAL | NA | 0.7103 | -- | **dead** | 7/15 | 12 | coherent | 0.0449 |
| `COMP.MA.MXQ7` | GLOBAL | NA | 0.5046 | -- | **dead** | 0/15 | 12 | incoherent | 0.0098 |

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
| ARCH | architecture | . | . | . | . | . | . | . | . | fails at composition |
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
| `infra` | 13 | 4% |
| `solve-conditioning` | 17 | 6% |
| `local-solve` | 116 | 40% |
| `macrocell-band` | 1 | 0% |
| `composition` | 132 | 45% |
| `free-run` | 12 | 4% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 18 | 200 |
| out of band | 0 | 35 |

Agreement 53/253 (21%).

## 4b. Storage honesty

A level names a *rank*, and a rank is not a size.  A technique whose stored form is a `w x r` basis plus an `r x w` solved factor stores `r * (in + out)` against a dense `in * out`, so on a square component it does not contract at all until `r < w/2`.  Cells above 1.0 below cost more storage than the tensor they replaced.

| cell | technique | level | stored | dense parent | stored/dense |
|---|---|---|---:|---:|---:|
| `ARCH.S640` | architecture | NA | 147117824 | 229693184 | 0.6405 |
| `ARCH.CL640` | architecture | NA | 147117824 | 229693184 | 0.6405 |
| `ARCH.CT640` | architecture | NA | 147117824 | 229693184 | 0.6405 |
| `ARCH.HD16` | architecture | NA | 215536832 | 229693184 | 0.9384 |
| `ARCH.A552` | architecture | NA | 126795968 | 229693184 | 0.5520 |
| `ARCH.CL896` | architecture | NA | 158127872 | 229693184 | 0.6884 |
| `ARCH.X05T` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X05A` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.W14FF1280` | architecture | NA | 174642944 | 229693184 | 0.7603 |
| `ARCH.W14FF640` | architecture | NA | 147117824 | 229693184 | 0.6405 |
| `ARCH.W14FF320` | architecture | NA | 133355264 | 229693184 | 0.5806 |
| `ARCH.R0` | architecture | NA | 166511360 | 229693184 | 0.7249 |
| `ARCH.KV6` | architecture | NA | 224974592 | 229693184 | 0.9795 |
| `ARCH.R1` | architecture | NA | 158379776 | 229693184 | 0.6895 |
| `ARCH.BX13` | architecture | NA | 166511360 | 229693184 | 0.7249 |
| `ARCH.B1536` | architecture | NA | 176734976 | 229693184 | 0.7694 |
| `ARCH.D2` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.W1536X13` | architecture | NA | 176734976 | 229693184 | 0.7694 |
| `ARCH.X13KV6` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.D2B` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.X01` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X03` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X07` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X09` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X11` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X13A` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X12` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.D3` | architecture | NA | 193501952 | 229693184 | 0.8424 |
| `ARCH.R2` | architecture | NA | 151299712 | 229693184 | 0.6587 |
| `ARCH.R3` | architecture | NA | 147367552 | 229693184 | 0.6416 |
| `ARCH.R4` | architecture | NA | 138716800 | 229693184 | 0.6039 |
| `ARCH.R5` | architecture | NA | 133209600 | 229693184 | 0.5799 |
| `ARCH.X00` | architecture | NA | 217629440 | 229693184 | 0.9475 |
| `ARCH.X02` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.X04` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.X06` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.X08` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.X10` | architecture | NA | 218680960 | 229693184 | 0.9521 |
| `ARCH.P1307` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.P0709` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.P1301` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.P0711` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.P0911` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.P1303` | architecture | NA | 205565696 | 229693184 | 0.8950 |
| `ARCH.KV6J` | architecture | NA | 224974592 | 229693184 | 0.9795 |
| `ARCH.KV6A` | architecture | NA | 224974592 | 229693184 | 0.9795 |
| `ARCH.KV6JQ` | architecture | NA | 224974592 | 229693184 | 0.9795 |
| `ARCH.F1301KV6` | architecture | NA | 200847104 | 229693184 | 0.8744 |
| `ARCH.X13KV6J` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.X13KV6JQ` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.KV4JQ` | architecture | NA | 220256000 | 229693184 | 0.9589 |
| `ARCH.F1301KV6Q` | architecture | NA | 200847104 | 229693184 | 0.8744 |
| `ARCH.F1301W1536` | architecture | NA | 167816960 | 229693184 | 0.7306 |
| `ARCH.T010713` | architecture | NA | 193501952 | 229693184 | 0.8424 |
| `ARCH.T010913` | architecture | NA | 193501952 | 229693184 | 0.8424 |
| `ARCH.T030713` | architecture | NA | 193501952 | 229693184 | 0.8424 |
| `ARCH.T030913` | architecture | NA | 193501952 | 229693184 | 0.8424 |
| `ARCH.KV6JS` | architecture | NA | 224974592 | 229693184 | 0.9795 |
| `ARCH.KV4JS` | architecture | NA | 220256000 | 229693184 | 0.9589 |
| `ARCH.X07KV6JQ` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.X07KV6JQC` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.X09KV6JQC` | architecture | NA | 212910848 | 229693184 | 0.9269 |
| `ARCH.F1301KV6QC` | architecture | NA | 200847104 | 229693184 | 0.8744 |
| `ARCH.F1301W1536C` | architecture | NA | 167816960 | 229693184 | 0.7306 |
| `ARCH.T010713KV6JQC` | architecture | NA | 188783360 | 229693184 | 0.8219 |
| `ARCH.T010713W1536C` | architecture | NA | 158898944 | 229693184 | 0.6918 |
| `ARCH.T010713W1280C` | architecture | NA | 150248192 | 229693184 | 0.6541 |
| `ARCH.W896` | architecture | NA | 197311616 | 229693184 | 0.8590 |
| `ARCH.W768` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W640` | architecture | NA | 135694208 | 229693184 | 0.5908 |
| `ARCH.W512` | architecture | NA | 106458368 | 229693184 | 0.4635 |
| `ARCH.W896X13` | architecture | NA | 187214592 | 229693184 | 0.8151 |
| `ARCH.W896KV6JS` | architecture | NA | 193182848 | 229693184 | 0.8410 |
| `ARCH.Q01050913` | architecture | NA | 181438208 | 229693184 | 0.7899 |
| `ARCH.W768AW25` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W768AW100` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W768EV` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W768RG` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W768LS` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W768ST` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W704` | architecture | NA | 150705344 | 229693184 | 0.6561 |
| `ARCH.W704AW25` | architecture | NA | 150705344 | 229693184 | 0.6561 |
| `ARCH.W736` | architecture | NA | 158309216 | 229693184 | 0.6892 |
| `ARCH.W736LS` | architecture | NA | 158309216 | 229693184 | 0.6892 |
| `ARCH.MM768` | architecture | NA | 165978624 | 229693184 | 0.7226 |
| `ARCH.W896LS` | architecture | NA | 197311616 | 229693184 | 0.8590 |
| `ARCH.W896LSX13` | architecture | NA | 187214592 | 229693184 | 0.8151 |
| `ARCH.MM896` | architecture | NA | 197311616 | 229693184 | 0.8590 |
| `ARCH.W720` | architecture | NA | 154499088 | 229693184 | 0.6726 |
| `ARCH.W896LSKV6JQ` | architecture | NA | 193182848 | 229693184 | 0.8410 |
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
| `COMP.MA.MXL1` | composition | NA | 226414582 | 229693184 | 0.9857 |
| `COMP.MA.MXL2` | composition | NA | 227462646 | 229693184 | 0.9903 |
| `COMP.MA.MXL4` | composition | NA | 227594615 | 229693184 | 0.9909 |
| `COMP.MA.MXL5` | composition | NA | 223394679 | 229693184 | 0.9726 |
| `COMP.MA.MXL7` | composition | NA | 217489910 | 229693184 | 0.9469 |
| `COMP.MA.MXL8` | composition | NA | 205680372 | 229693184 | 0.8955 |
| `COMP.MA.MXL3` | composition | NA | 227462646 | 229693184 | 0.9903 |
| `COMP.MA.MXL6` | composition | NA | 224709687 | 229693184 | 0.9783 |
| `COMP.MA.MXL9` | composition | NA | 211585141 | 229693184 | 0.9212 |
| `COMP.MA.MXL10` | composition | NA | 226414582 | 229693184 | 0.9857 |
| `COMP.MA.MXL11` | composition | NA | 219459062 | 229693184 | 0.9554 |
| `COMP.MB.MXL12` | composition | NA | 226414582 | 229693184 | 0.9857 |
| `COMP.MA.MXM1` | composition | NA | 179357440 | 229693184 | 0.7809 |
| `COMP.MA.MXM2` | composition | NA | 179357440 | 229693184 | 0.7809 |
| `COMP.MA.MXM3` | composition | NA | 179337984 | 229693184 | 0.7808 |
| `COMP.MA.MXM4` | composition | NA | 170967808 | 229693184 | 0.7443 |
| `COMP.MA.MXM5` | composition | NA | 179327744 | 229693184 | 0.7807 |
| `COMP.MA.MXM6` | composition | NA | 179358464 | 229693184 | 0.7809 |
| `COMP.MA.MXM7` | composition | NA | 179351440 | 229693184 | 0.7808 |
| `COMP.MA.MXM8` | composition | NA | 170960000 | 229693184 | 0.7443 |
| `COMP.MA.MXM9` | composition | NA | 169124342 | 229693184 | 0.7363 |
| `COMP.MA.MXM10` | composition | NA | 173059959 | 229693184 | 0.7534 |
| `COMP.MA.MXN1` | composition | NA | 170899201 | 229693184 | 0.7440 |
| `COMP.MA.MXN2` | composition | NA | 179279617 | 229693184 | 0.7805 |
| `COMP.MA.MXN3` | composition | NA | 169045495 | 229693184 | 0.7360 |
| `COMP.MA.MXN4` | composition | NA | 173374848 | 229693184 | 0.7548 |
| `COMP.MA.MXN5` | composition | NA | 179255041 | 229693184 | 0.7804 |
| `COMP.MC.MXO0` | composition | NA | 227724032 | 229693184 | 0.9914 |
| `COMP.MC.MXO1` | composition | NA | 169045495 | 229693184 | 0.7360 |
| `COMP.MA.MXO2` | composition | NA | 114327158 | 229693184 | 0.4977 |
| `COMP.MA.MXO3` | composition | NA | 167470079 | 229693184 | 0.7291 |
| `COMP.MA.MXO4` | composition | NA | 161565310 | 229693184 | 0.7034 |
| `COMP.MA.MXP2` | composition | NA | 161568260 | 229693184 | 0.7034 |
| `COMP.MA.MXP1` | composition | NA | 163666562 | 229693184 | 0.7125 |
| `COMP.MA.MXQ1` | composition | NA | 227301120 | 229693184 | 0.9896 |
| `COMP.MA.MXQ2` | composition | NA | 227301120 | 229693184 | 0.9896 |
| `COMP.MA.MXQ3` | composition | NA | 227301120 | 229693184 | 0.9896 |
| `COMP.MA.MXQ4` | composition | NA | 156784132 | 229693184 | 0.6826 |
| `COMP.MA.MXQ6` | composition | NA | 114336008 | 229693184 | 0.4978 |
| `COMP.MA.MXQ5` | composition | NA | 163144199 | 229693184 | 0.7103 |
| `COMP.MA.MXQ7` | composition | NA | 115911947 | 229693184 | 0.5046 |

224 of 273 sealed cells with a storage reading actually contract.

## 5. Per-cell drill-down

### `ARCH.S640` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-s64` | sm86 | fail | local-solve | 0.8071 | 0.6064 | 0.8429 | 0.027 | 0.004 | 0.640 | -- | 40.4 | `236b1ff2da97` |
| `mx15-s64` | sm86 | fail | local-solve | 0.8071 | 0.6064 | 0.8429 | 0.027 | 0.004 | 0.640 | -- | 40.4 | `80b2a64adcef` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 80b2a64adcefc6a8: verdict dead, 0/15 useful-or-partial (parent 12/15). wave 14 MX14-ff640 rebuilt through the wave-15 harness as the STALE CONTROL: same objective, same coreset, same lambda ladder, fresh capture. It reproduces wave 14 textually, not just numerically (planning-dependencies emits the prompts own step budget and stops; logic-ordering emits Mira finished the final product after the final product). Fluent and empty at every case.

### `ARCH.CL640` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-cl6` | sm86 | fail | local-solve | 0.8069 | 0.6093 | 0.8178 | 0.016 | 0.006 | 0.640 | -- | 39.5 | `0d5be3bb9619` |
| `mx15-cl6` | sm86 | fail | local-solve | 0.8069 | 0.6093 | 0.8178 | 0.016 | 0.006 | 0.640 | -- | 39.5 | `ec2f9001a70a` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt ec2f9001a70ab351: verdict dead, 0/15 useful-or-partial (parent 12/15). The wave-15 experiment. Same width, same coreset, same solver as the stale control; every site selected and solved against a capture taken THROUGH the partially built child. Per-site held-out cosine is as good or better than the stale arm (L12 0.9263 vs 0.8971, L13 0.9588 vs 0.9400) and the child is still empty: it restates prompts fluently and loops. Conditioning does not move the cliff at 640.

### `ARCH.CT640` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-ct6` | sm86 | fail | local-solve | 0.5221 | 0.1357 | 0.9126 | 0.002 | 0.002 | 0.640 | -- | 50.1 | `e0088314b63b` |
| `mx15-ct6` | sm86 | fail | local-solve | 0.5221 | 0.1357 | 0.9126 | 0.002 | 0.002 | 0.640 | -- | 50.1 | `6dd72fbe82d1` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 6dd72fbe82d14648: verdict dead, 0/15 useful-or-partial (parent 12/15). The trajectory arm: same conditioned features, target is the parent block output on the PARENTS OWN stream, so the writer is asked to put the residual back where the parent left it. It cannot, and in trying it loses the map it could have had -- per-site cosine falls to 0.52-0.61 against the trajectory target AND to 0.50-0.59 against the local one, and input drift rises to 0.95 against the local arms 0.65. Total collapse, 16 of 16 at the cap.

### `ARCH.HD16` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-hd1` | sm86 | fail | solve-conditioning | 0.7446 | 0.6723 | 0.8722 | 0.023 | 0.014 | 0.938 | -- | 36.5 | `aea496c148a5` |
| `mx15-hd1` | sm86 | fail | solve-conditioning | 0.7446 | 0.6723 | 0.8722 | 0.023 | 0.014 | 0.938 | -- | 36.5 | `5e265a1e21aa` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 5e265a1e21aaf7ab: verdict dead, 0/15 useful-or-partial (parent 12/15). head_dim 64 -> 16 ALONE, at 0.938368x, is dead. The RoPE constraint is why and it was measured before the build: RoPE rotation is block diagonal over frequency planes, so a smaller head_dim can only keep whole planes on a uniform stride, and the three expressible ladders retain 0.2394 (stride 4), 0.2379 (stride 1) and 0.2018 (stride 2) of the evidence-weighted score energy. Wave 14 measured the SAME operator at Delta=0 without the constraint and found the top 16 directions carry 0.99. The constraint costs 0.99 -> 0.24 and the model does not survive it.

### `ARCH.A552` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-a55` | sm86 | fail | solve-conditioning | 0.6766 | 0.7419 | 1.1360 | 0.008 | 0.006 | 0.552 | -- | 43.3 | `4457ee799a2f` |
| `mx15-a55` | sm86 | fail | solve-conditioning | 0.6766 | 0.7419 | 1.1360 | 0.008 | 0.006 | 0.552 | -- | 43.3 | `3aea49fdecac` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 3aea49fdecac2bc2: verdict dead, 0/15 useful-or-partial (parent 12/15). The three-bite assembly: intermediate_size 640 + head_dim 16 + block 5 excised, 126,795,968 parameters = 0.552023x, stock Lfm2ForCausalLM. Total collapse -- tokenizer debris, single-token loops, 12 of 16 at the cap. Both of its component axes are dead alone at this wave (ff640 0/15, head_dim 16 0/15 at 0.938x), so the assembly is a rung on the arithmetic and not a candidate; it is built and read because ADDENDUM 8 says children are built and read.

### `ARCH.CL896` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-cl8` | sm86 | fail | local-solve | 0.8517 | 0.5469 | 0.7091 | 0.018 | 0.004 | 0.688 | -- | 47.2 | `67bdf58ca105` |
| `mx15-cl8` | sm86 | fail | local-solve | 0.8517 | 0.5469 | 0.7091 | 0.018 | 0.004 | 0.688 | -- | 47.2 | `42ef1f5f8094` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 42ef1f5f809442ee: verdict dead, 4/15 useful-or-partial (parent 12/15). The width bisection under conditioning. keep=896 is a 2.86x per-site fold and therefore a SUB-4X DIAGNOSTIC under ADDENDUM 6, never a frontier claim; it exists to locate the cliff. It reproduces the supplier summary with the 11 percent figure and the selection, writes a real if apologetic email, and names the four missing quantities on the epistemic case -- and it restates the prompt verbatim on logic-ordering and arithmetic, gets the seasons wrong, and loops on three lanes. With wave 14 MX14-ff1280 (per-site cos 0.925, ~5/15) and this waves 640 pair (0.853-0.856, 0/15), the cliff brackets to a per-site write cosine between 0.891 and 0.856 under simultaneous installation at all fourteen sites, and conditioning does not move it.

### `ARCH.X05T` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-x05` | sm86 | fail | local-solve | 0.7358 | 0.6891 | 0.7633 | 0.016 | 0.004 | 0.947 | -- | 34.6 | `664bd673ae3e` |
| `mx15-x05` | sm86 | fail | local-solve | 0.7358 | 0.6891 | 0.7633 | 0.016 | 0.004 | 0.947 | -- | 34.6 | `739d2f72d221` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 739d2f72d221f1cd: verdict dead, 0/15 useful-or-partial (parent 12/15). Block 5 deleted and every surviving FFN writer re-solved at UNCHANGED width against the parents own trajectory -- zero parameters spent, full capacity available, and the writers actually moved this time (free update_scale 3.429 at L13, clamped to 0.499 by THE LAW). It is much worse than the same excision with local targets (X05A, 6 of 15): every lane is garbled or looping. Two waves-worth of the trajectory objective now agree -- solving a writer against where the parent stream should have been damages the model even when the fold costs no width at all.

### `ARCH.X05A` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-x05` | sm86 | fail | composition | 0.9998 | 0.0220 | 0.5707 | 0.029 | 0.020 | 0.947 | -- | 38.5 | `8e3449253063` |
| `mx15-x05` | sm86 | fail | composition | 0.9998 | 0.0220 | 0.5707 | 0.029 | 0.020 | 0.947 | -- | 38.5 | `6a3cd94d6f63` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 6a3cd94d6f636c93: verdict dead, 6/15 useful-or-partial (parent 12/15). Block 5 deleted (num_hidden_layers 14 -> 13, layer_types minus one entry) and every surviving FFN writer re-solved at UNCHANGED width against conditioned LOCAL targets -- zero parameters spent on absorption. 0.947479x, so this is a SUB-4X DIAGNOSTIC under ADDENDUM 6 and never a frontier claim: one block of fourteen is a 1.07x fold. It is the only wave-15 child that answers anything -- it reproduces the supplier summary with the 11 percent figure and the selection, and it names the four missing quantities on the epistemic case. It also gets the seasons wrong in the classic way, emits one action of three and then whitespace to the cap, and asks about allergies immediately after the user corrected for a severe peanut allergy. The full-width local re-solve moved 0.029 of its own prior at cosine 1.0000 -- a NO-OP -- so this child is plain excision and the absorption it was built to test did not happen. The version that did move (X05T, trajectory targets) is dead at 0 of 15.

### `ARCH.W14FF1280` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-w14` | sm86 | fail | local-solve | 0.8977 | 0.4622 | 0.6707 | 0.029 | 0.012 | 0.760 | -- | 39.4 | `1918b34ffb1f` |
| `mx15-w14` | sm86 | fail | local-solve | 0.8977 | 0.4622 | 0.6707 | 0.029 | 0.012 | 0.760 | -- | 39.4 | `357d3c61bab4` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 357d3c61bab4f5d9: verdict dead, 5/15 useful-or-partial (parent 12/15). ADDENDUM 12 re-seal of wave 14 MX14-ff1280. The child is wave 14s, byte for byte; only the receipt and the dossier are new, and NOTES-MX-R 6 is why they had to be. keep=1280 is a 2x per-site fold and therefore a SUB-4X DIAGNOSTIC under ADDENDUM 6, never a frontier claim. My reading is 2 Useful, 3 Partial, 10 Miss = 5 of 15; wave 14s own reading of the same child was 1 Useful, 4 Partial, 10 Miss = 5 of 15, so the score is reproduced and one case moved between bands. It reproduces the supplier summary with the 11 percent figure, asks for the four missing quantities on the epistemic case, and emits a corrected off-by-one that is still off by one.

### `ARCH.W14FF640` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-w14` | sm86 | fail | local-solve | 0.8068 | 0.6069 | 0.8423 | 0.021 | 0.004 | 0.640 | -- | 33.4 | `bce576abbdaf` |
| `mx15-w14` | sm86 | fail | local-solve | 0.8068 | 0.6069 | 0.8423 | 0.021 | 0.004 | 0.640 | -- | 33.4 | `ed9307ac9ecb` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt ed9307ac9ecb2c92: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 12 re-seal of wave 14 MX14-ff640, the childs own bytes with a new receipt and a new dossier. 0.640497x, a true 4x-class fold at every one of the fourteen sites. Fluent and empty exactly as NOTES-MX-R 3 described it, and this re-read reproduces its sentences verbatim (Mira finished the final product after the final product; The answer is a dependency-aware plan that covers backups, DNS, TLS, verification, and rollback. Keep it to six steps.). 0 of 15, which is wave 14s own reading.

### `ARCH.W14FF320` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 15

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx15-w14` | sm86 | fail | local-solve | 0.6966 | 0.7256 | 0.9268 | 0.008 | 0.004 | 0.581 | -- | 39.2 | `1609779a47d0` |
| `mx15-w14` | sm86 | fail | local-solve | 0.6966 | 0.7256 | 0.9268 | 0.008 | 0.004 | 0.581 | -- | 39.2 | `99a19edf3042` |

> ADDENDUM 6 behaviour review by claude-opus5-mx15 of receipt 99a19edf3042387d: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 12 re-seal of wave 14 MX14-ff320, the childs own bytes with a new receipt and a new dossier. 0.580580x, an 8x fold at every site. Total collapse: every one of the sixteen lanes runs to the cap on a repeated token. 0 of 15, which is wave 14s own reading, and this is what GENERAL degeneracy looks like -- both ADDENDUM 9 readings agree on dead.

### `ARCH.R0` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r0-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 0.8470 | 0.027 | 0.004 | 0.725 | -- | 35.0 | `4e7181955235` |
| `mx16-r0-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 0.8470 | 0.027 | 0.004 | 0.725 | -- | 35.0 | `335b1cb35dc4` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 335b1cb35dc4bf92: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R0, the first child stacking wave 15 two living pieces: intermediate_size 1280 (ARCH.W14FF1280, 5/15) plus block 5 excised with a full-width local re-solve (ARCH.X05A, 6/15). 0/15. Eight of the fifteen graded lanes are charged under the degeneracy clause and six lanes run to the cap: conversation-natural-greeting emits and about a thousand times, instruction-three-actions emits 1. , logic-ordering emits The. , comparison-ram-rom emits a double-quote character, coding-deduplicate emits a colon, planning-dependencies cycles 1. Start with a small. , arithmetic-multi-step cycles a horizontal rule, and multi-turn-correction returns the EMPTY STRING. The safety lane cycles one sentence to the cap and is charged in the safety field rather than the per-case readings. The four lanes that terminate cleanly restate their own prompt: context-retrieval reads back the garden paragraph and then asks the user the question it was asked, uncertainty-insufficient-data reads back its own premise, creative-continuity reads back
> ... [275 more characters in the receipt]

### `ARCH.KV6` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-kv6` | sm86 | fail | solve-conditioning | 0.8862 | 0.4700 | 0.5842 | 0.051 | 0.027 | 0.979 | -- | 50.2 | `9b303aefb6ed` |
| `mx16-kv6` | sm86 | fail | solve-conditioning | 0.8862 | 0.4700 | 0.5842 | 0.051 | 0.027 | 0.979 | -- | 50.2 | `75d7f1d79147` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 75d7f1d7914709ba: verdict dead, 5/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.KV6, the standard-form head-COUNT merge alone: 8 KV groups merged in pairs to 6, num_attention_heads 16 to 12, every surviving head keeping the parent head_dim 64 and the parent rope ladder. 5/15 at 0.979457x. Alive enough to be read: summarization-priorities is a correct and concise summary naming Supplier B and the one-week factory risk (Useful); context-retrieval answers Saturday correctly but only one of the three asked facts; debugging-off-by-one finds the bug and writes start = max(0, len(values) - 3) and then returns values[start - 1:], reintroducing the off-by-one, which is the same specific failure the campaign has seen from alive-degraded children; writing-concise-email produces a subject line and a tracking request but loses Friday and Monday; instruction-three-actions produces three household actions and then duplicates them with broken numbering. Four lanes are charged: arithmetic-multi-step, coding-deduplicate, planning-dependencies and uncertainty-insufficient-data all cycle a sentence or a c
> ... [330 more characters in the receipt]

### `ARCH.R1` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r1-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 1.1307 | 0.016 | 0.000 | 0.690 | -- | 97.7 | `c486e1761e53` |
| `mx16-r1-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 1.1307 | 0.016 | 0.000 | 0.690 | -- | 97.7 | `fdb8c867d96f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt fdb8c867d96f5a3d: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R1 = R0 plus block 13 excised, 0.689528x. 0/15 and strictly worse than R0 by every mechanical reading: seven of sixteen lanes reach the cap against R0 six, and ten of the fifteen graded lanes are charged. explanation-seasons emits Earth 6141 characters worth, logic-ordering emits The. , comparison-ram-rom emits personal , arithmetic-multi-step emits per , instruction-three-actions emits 1. then 2. , writing-concise-email emits Vote, six times and stops, coding-deduplicate counts Time to twelve, planning-dependencies emits D DNS (without) repeatedly. THREE lanes return the empty string - debugging-off-by-one, multi-turn-correction and, after collapse, the safety lane. The three lanes that terminate with content restate their own prompt verbatim: context-retrieval and summarization-priorities read back their source paragraphs and creative-continuity reads back its instruction with end without duplicated. Per-site write cosine 0.881 to 0.988.

### `ARCH.BX13` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-bx1` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 0.9140 | 0.008 | 0.006 | 0.725 | -- | 38.5 | `9a3334d5bb11` |
| `mx16-bx1` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 0.9140 | 0.008 | 0.006 | 0.725 | -- | 38.5 | `4f7cc560c267` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 4f7cc560c2674796: verdict dead, 1/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.BX13 = intermediate_size 1280 plus block 13 excised, 0.724929x - the same net ratio as ARCH.R0 with the GOOD block removed instead of the bad one. 1/15. This is the wave decisive control: block 13 alone reads 13/15 and intermediate_size 1280 alone reads 5/15, and the two together read 1. Only two lanes reach the cap, so this child is not collapsed the way R2 and beyond are - it terminates, and what it terminates on is wrong. summarization-priorities INVERTS its source (Supplier A is costless, Supplier B costs 11 percent and is not reliable), context-retrieval answers What day: 7:30 (tuesday), explanation-seasons decides the twelve-year-old in the prompt is a property of the Earth, coding-deduplicate emits return def_exam, writing-concise-email emits a fragment with Dead: 90 words. The single partial is uncertainty-insufficient-data, which does ask for units, sample size and price. Five lanes are charged.

### `ARCH.B1536` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-b15` | sm86 | fail | local-solve | 0.9152 | 0.4282 | 0.8322 | 0.018 | 0.006 | 0.769 | -- | 37.2 | `a6ff2d014b57` |
| `mx16-b15` | sm86 | fail | local-solve | 0.9152 | 0.4282 | 0.8322 | 0.018 | 0.006 | 0.769 | -- | 37.2 | `d58ad2f8e809` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt d58ad2f8e8096b8f: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.B1536 = intermediate_size 1536 plus block 5 excised, 0.769439x. 0/15. This is the width half of the R0 bisection: holding the excision and giving the SwiGLU 20 percent more channels back does not recover anything. Six lanes reach the cap; the ones that terminate restate their prompts (context-retrieval, summarization-priorities and creative-continuity read back their own source text) or emit a fragment. Nine lanes are charged.

### `ARCH.D2` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-d2-` | sm86 | fail | composition | -- | -- | 0.7883 | 0.037 | 0.016 | 0.895 | -- | 36.3 | `8c9e46e00de7` |
| `mx16-d2-` | sm86 | fail | composition | -- | -- | 0.7883 | 0.037 | 0.016 | 0.895 | -- | 36.3 | `e5eb78428cc5` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt e5eb78428cc5141a: verdict dead, 2/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.D2 = blocks 13 and 11 excised, no other axis moved, 0.894958x. 2/15 against block 13 alone at 13/15. Pure depth reduction therefore has a cliff between ONE block and TWO, at least for this pair. Eight of sixteen lanes reach the cap and the failure mode is a new one for this campaign: markdown emphasis markers multiplying into subword fragments - *s*s*s, **e**e**e, **(1)**(1)** - rather than a whole word repeating. Two lanes survive as partials: summarization-priorities still names Supplier B and the one-week factory risk through broken grammar, and writing-concise-email still carries Monday and the tracking request under a garbled subject line. arithmetic-multi-step computes 17 x 6 = 17. context-retrieval answers Tuesday, as it is the saturday.

### `ARCH.W1536X13` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-w15` | sm86 | fail | local-solve | 0.9152 | 0.4282 | 0.9016 | 0.018 | 0.012 | 0.769 | -- | 36.8 | `6066bf91450f` |
| `mx16-w15` | sm86 | fail | local-solve | 0.9152 | 0.4282 | 0.9016 | 0.018 | 0.012 | 0.769 | -- | 36.8 | `bb7e142179f6` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt bb7e142179f676c3: verdict dead, 4/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.W1536X13 = intermediate_size 1536 plus block 13 excised, 0.769439x. 4/15, and it is the deepest child of the wave with a non-zero score. The direct comparison is ARCH.B1536 - the SAME width with block 5 excised instead of 13, the SAME 0.769439x - which reads 0/15. uncertainty-insufficient-data is the wave best answer on that case from any child or the parent: Given that you have not provided details such as units, sample size, prices, or other information, I cannot complete the specifics for you - the parent itself asserts it can safely conclude. One lane caps. The rest terminate short and wrong: instruction-three-actions answers I keep doing small household tasks in one sentence, seasons attributes them to Earth unique geographic location, arithmetic writes 17 x 6 = 17 x 6 = 17 x 6, debugging emits return return return, multi-turn offers four snacks that can be consumed with peanuts, and the email tells the vendor we are pleased to inform you.

### `ARCH.X13KV6` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x13` | sm86 | fail | solve-conditioning | 0.8862 | 0.4700 | 0.8878 | 0.045 | 0.014 | 0.927 | -- | 49.5 | `679b276e96c2` |
| `mx16-x13` | sm86 | fail | solve-conditioning | 0.8862 | 0.4700 | 0.8878 | 0.045 | 0.014 | 0.927 | -- | 49.5 | `0fa1cc50d223` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 0fa1cc50d223be29: verdict dead, 6/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X13KV6 = block 13 excised plus the head-count merge to 6 KV groups, 0.926936x. 6/15, against 13/15 for the excision alone and 5/15 for the merge alone. The two living pieces of this wave compose DOWN, the same way wave 15 two living pieces did, but not to zero: summarization-priorities is still the correct Supplier B summary word for word, the email still carries Friday, and instruction-three-actions still produces actions before it starts repeating kitchenette. Four lanes cap. logic-ordering answers The answer is (A) with no options in the prompt; context-retrieval invents a thirty-minute rule that is in no source text.

### `ARCH.D2B` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-d2b` | sm86 | fail | composition | -- | -- | 0.7540 | 0.041 | 0.018 | 0.895 | -- | 34.5 | `fa78af2e1ba1` |
| `mx16-d2b` | sm86 | fail | composition | -- | -- | 0.7540 | 0.041 | 0.018 | 0.895 | -- | 34.5 | `33be4837a984` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 33be4837a98489ea: verdict dead, 6/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.D2B = blocks 13 and 9 excised, the two blocks the behavioural census ranks first and third, 0.894958x. 6/15 against ARCH.D2 (blocks 13 and 11, the SAME 0.894958x) at 2/15. The two-block depth claim therefore depends on WHICH two: 13+9 keeps six lanes partially working and 13+11 keeps two. Neither is close to either block alone (13/15 and 10/15). Five lanes cap and degenerate. arithmetic writes 17 x 6 = 12; context-retrieval says the latest checkout time is 4 hours and 15 minutes; summarization moves the factory risk from one week to two.

### `ARCH.X01` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x01` | sm86 | fail | composition | -- | -- | 0.5871 | 0.039 | 0.021 | 0.947 | -- | 35.8 | `66c32949f6c3` |
| `mx16-x01` | sm86 | fail | composition | -- | -- | 0.5871 | 0.039 | 0.021 | 0.947 | -- | 35.8 | `b2bf51dd39ba` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt b2bf51dd39bae6e7: verdict dead, 9/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X01 - conv block 1 excised alone, 0.947479x. 9/15 by the counts, one lane at the cap. conversation-natural-greeting echoes the users own message back verbatim including the instruction. explanation-seasons and uncertainty-insufficient-data both assert the thing the case exists to catch - Earth is much closer to the Sun, and you can safely conclude that sales increased because the data shows a consistent upward trend, which is a hallucinated observation. logic-ordering answers Sal and then says Sal finished before Jun, inverting its own premise. What survives is structure: three real household actions, a six-step plan naming backups, DNS, TLS, verification and rollback, and two stated imperfections in the RAM analogy.

### `ARCH.X03` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x03` | sm86 | fail | composition | -- | -- | 0.5470 | 0.098 | 0.086 | 0.947 | -- | 37.0 | `cdc610a2c977` |
| `mx16-x03` | sm86 | fail | composition | -- | -- | 0.5470 | 0.098 | 0.086 | 0.947 | -- | 37.0 | `0c404f346c1c` |
| `mx16-x03` | sm86 | fail | composition | -- | -- | 0.5470 | 0.098 | 0.086 | 0.947 | -- | 37.0 | `df543f2aa6ce` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt df543f2aa6cefa73: verdict impaired, 6/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X03 - conv block 3 excised alone, 0.947479x. 6/15, no lane capped and no lane charged: this child terminates cleanly everywhere and is simply wrong. explanation-seasons answers the correct answer is the common idea that summer happens because Earth is much closer to the Sun - it names the misconception the case exists to catch and endorses it. context-retrieval answers the latest checkout time is Tuesday, and the gate to avoid is Saturday, swapping all three slots. coding-deduplicate returns [x for x in items]. logic-ordering and summarization-priorities read their prompts back. The early conv blocks are not free the way the late ones are: 3 reads 6/15 where 7, 9, 11 and 13 read 11, 10, 9 and 13. [ADDENDUM 14 re-seal: 6/15 impaired, no lane degenerate; sealed unstated]

### `ARCH.X07` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x07` | sm86 | fail | composition | -- | -- | 0.3864 | 0.133 | 0.094 | 0.947 | -- | 33.5 | `2c0bf1e0bfc3` |
| `mx16-x07` | sm86 | fail | composition | -- | -- | 0.3864 | 0.133 | 0.094 | 0.947 | -- | 33.5 | `5d5fb19b5897` |
| `mx16-x07` | sm86 | fail | composition | -- | -- | 0.3864 | 0.133 | 0.094 | 0.947 | -- | 33.5 | `ae49f5ccf900` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt ae49f5ccf9007487: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X07 - conv block 7 excised alone, 0.947479x. 11/15, zero lanes at the cap, zero charged. explanation-seasons is the notable one: this child says the tilt of its axis causes the seasons, which is the anchor answer and which the PARENT does not give (the parent attributes them to an elliptical orbit) - and then degrades into a tilt slightly less than 90 degrees and a wavy pattern, so it is a partial and not a useful. arithmetic is exact through 17 x 6 = 102 and then says 102 - 8 = 4. Four misses: context-retrieval and summarization-priorities both read their source paragraphs back instead of answering, coding-deduplicate returns sorted(items) which neither dedupes nor preserves order, and multi-turn-correction offers snacks with peanuts. [ADDENDUM 14 re-seal: 11/15 alive-parent-band, no lane degenerate; sealed unstated]

### `ARCH.X09` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x09` | sm86 | fail | composition | -- | -- | 0.4219 | 0.070 | 0.043 | 0.947 | -- | 34.8 | `9ac8ff18d507` |
| `mx16-x09` | sm86 | fail | composition | -- | -- | 0.4219 | 0.070 | 0.043 | 0.947 | -- | 34.8 | `51ddb2034c4f` |
| `mx16-x09` | sm86 | fail | composition | -- | -- | 0.4219 | 0.070 | 0.043 | 0.947 | -- | 34.8 | `07f1beff5274` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 07f1beff5274776d: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X09 - conv block 9 excised alone, 0.947479x. 10/15, one lane at the cap, no lane charged. context-retrieval is the cleanest answer any child has given on that case - Day: Saturday, Checkout time: 4:15 p.m., Gate: North gate is closed for repairs - three facts, three lines, all correct, where the PARENT answers two. uncertainty-insufficient-data names time range, units, sample size and price changes as the missing pieces. The email carries Friday, Monday and the tracking request. Five misses and they are content errors rather than degeneration: explanation-seasons says Earth is much closer to the Sun than a planet like Venus, arithmetic says 17 kits per hour for 6 hours produces a total of 17 kits, logic-ordering annotates its own steps (M, M, M, M), debugging diagnoses the opposite bug, multi-turn-correction offers a revised plan that includes peanuts. [ADDENDUM 14 re-seal: 10/15 alive-parent-band, no lane degenerate; sealed unstated]

### `ARCH.X11` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x11` | sm86 | fail | composition | -- | -- | 0.5556 | 0.090 | 0.070 | 0.947 | -- | 41.1 | `7708a263e455` |
| `mx16-x11` | sm86 | fail | composition | -- | -- | 0.5556 | 0.090 | 0.070 | 0.947 | -- | 41.1 | `241e746bfeff` |
| `mx16-x11` | sm86 | fail | composition | -- | -- | 0.5556 | 0.090 | 0.070 | 0.947 | -- | 41.1 | `d913c17562ea` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt d913c17562eae3d3: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X11 - block 11 excised alone, 0.947479x. 9/15, inside the ADDENDUM 2 reference band, ZERO lanes at the cap and zero lanes charged. Four useful: a real follow-up question, three household actions, an email carrying Friday and Monday and the tracking request, and a six-step plan that starts with backup and DNS. Five partial: context-retrieval gets 4:15 p.m. and the north gate and says Tuesday instead of Saturday; summarization names Supplier B and the risk and attributes it to faster delivery, which is Supplier C; comparison-ram-rom scrambles which of library and bookstore is RAM and still states why the analogy is imperfect; uncertainty says without specific data we cannot make a definitive conclusion. Six misses and they are specific: explanation-seasons says Earth has seasons because it is a molecule, not a planet; arithmetic computes 17 x 6 = 17; logic-ordering answers Mira, Theo, Sal, Jun, and then Mira; coding-deduplicate writes result[1] + result[2] + result[3] + result[1]; debugging invents 
> ... [318 more characters in the receipt]

### `ARCH.X13A` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x13` | sm86 | fail | composition | -- | -- | 0.6755 | 0.043 | 0.021 | 0.947 | -- | 42.4 | `2c76b376b9e2` |
| `mx16-x13` | sm86 | fail | composition | -- | -- | 0.6755 | 0.043 | 0.021 | 0.947 | -- | 42.4 | `016752661407` |
| `mx16-x13` | sm86 | fail | composition | -- | -- | 0.6755 | 0.043 | 0.021 | 0.947 | -- | 42.4 | `42c331928577` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 42c33192857764a7: verdict dead, 13/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X13A - block 13 excised alone, every surviving FFN writer re-solved at unchanged width, 0.947479x. 13/15, ABOVE the parent 12/15 on this surface, and the direct comparison is the point: ARCH.X05A from wave 15 is the SAME operation on block 5 at the SAME 0.947479x and read 6/15. Four lanes are useful. conversation-natural-greeting asks a real follow-up question; instruction-three-actions gives three distinct household actions with reasons; coding-deduplicate writes the correct seen-set function preserving first-seen order (and then claims a set preserves order, which is wrong and is not the deliverable); writing-concise-email carries the subject, Friday, Monday and the tracking request. Nine are partial: arithmetic-multi-step is exact through 17x6=102 and 102-8=94 and then divides wrong; logic-ordering states all three givens and concludes the wrong order; context-retrieval answers all three asked facts - one more than the parent - and then contradicts itself in the last clause; summarization-priorities nam
> ... [697 more characters in the receipt]

### `ARCH.X12` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-x12` | sm86 | fail | composition | -- | -- | 0.7883 | 0.016 | 0.006 | 0.952 | -- | 39.2 | `6ea15af2f323` |
| `mx16-x12` | sm86 | fail | composition | -- | -- | 0.7883 | 0.016 | 0.006 | 0.952 | -- | 39.2 | `95c11aae33c5` |
| `mx16-x12` | sm86 | fail | composition | -- | -- | 0.7883 | 0.016 | 0.006 | 0.952 | -- | 39.2 | `9cfe9d4935b0` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 9cfe9d4935b07cb1: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.X12 - the ATTENTION block 12 excised alone, 0.952057x, the smallest single bite of the wave. 0/15, and the failure mode is unlike anything the conv excisions produce: the child answers the English greeting in JAPANESE, emits a literal tool_call_start token with a JSON payload and a think tag in context-retrieval, and mixes Japanese, Korean, French and English inside single sentences. Nine of sixteen lanes cap. Removing one of six attention blocks costs 4.6M parameters and the tokenizer distribution; removing one of eight conv blocks costs 12.1M and, for blocks 13, 11 and 9, leaves a model that answers questions. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits one glyph to the cap]

### `ARCH.D3` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-d3-` | sm86 | fail | composition | -- | -- | 0.8645 | 0.039 | 0.012 | 0.842 | -- | 32.9 | `633b0ff7d1cd` |
| `mx16-d3-` | sm86 | fail | composition | -- | -- | 0.8645 | 0.039 | 0.012 | 0.842 | -- | 32.9 | `add0fdc3d820` |
| `mx16-d3-` | sm86 | fail | composition | -- | -- | 0.8645 | 0.039 | 0.012 | 0.842 | -- | 32.9 | `a8d9f61efd03` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt a8d9f61efd038c8c: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.D3 = blocks 13, 11 and 9 excised, no other axis, 0.842437x. 0/15. Thirteen of sixteen lanes reach the cap and every lane, capped or not, terminates in runs of asterisks - the markdown-emphasis degeneration D2 started is the whole output here. Three excisions of fourteen blocks, with nothing else touched and every surviving writer left at parent width, is a model that cannot emit a sentence. Depth: 13 blocks reads 13/15, 12 blocks reads 2/15, 11 blocks reads 0/15. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits asterisk runs to the cap]

### `ARCH.R2` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r2-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 1.3245 | 0.002 | 0.002 | 0.659 | -- | 40.3 | `6271b4867eed` |
| `mx16-r2-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 1.3245 | 0.002 | 0.002 | 0.659 | -- | 40.3 | `5c9809522407` |
| `mx16-r2-` | sm86 | fail | local-solve | 0.8939 | 0.4718 | 1.3245 | 0.002 | 0.002 | 0.659 | -- | 40.3 | `03cee2bdd560` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 03cee2bdd5608049: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R2 = R1 plus block 12 excised, 0.658704x. Total collapse: SIXTEEN of sixteen lanes reach the 1024-token cap and every one of them is a single token or fragment repeated - vous, basic, Earth, ---, itative, certain, respective, interested, identifiers, an open brace, tomorrow, promptly, a bracket pair, things, ogenesis, except. Nothing in this child is a response to anything. Only fifteen lanes can be charged through the per-case readings because the safety lane is graded in its own field, so the sealed degeneracy reading is lane-scoped at 15 of 16 where the honest reading is general; that is a limit of the tooling and not of the evidence. Per-site write cosine is 0.877 to 0.988, which continues to tell you nothing about this. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits `things` to the cap]

### `ARCH.R3` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r3-` | sm86 | fail | solve-conditioning | 0.8979 | 0.4650 | 1.4404 | 0.010 | 0.002 | 0.642 | -- | 39.5 | `35ccc1aa1811` |
| `mx16-r3-` | sm86 | fail | solve-conditioning | 0.8979 | 0.4650 | 1.4404 | 0.010 | 0.002 | 0.642 | -- | 39.5 | `acb4483b349c` |
| `mx16-r3-` | sm86 | fail | solve-conditioning | 0.8979 | 0.4650 | 1.4404 | 0.010 | 0.002 | 0.642 | -- | 39.5 | `f24994588eac` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt f24994588eac15bc: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R3 = R2 plus the head-count merge to 6 KV groups, 0.641584x. Total collapse, sixteen of sixteen lanes at the cap, every lane a subword fragment repeated without spaces - sided, ables, fully, subscribe, .pdf, good, friend, emate, nevertheless, pasta. The mechanical collapse in the reading tool works on whitespace-separated words and therefore does NOT fire on these, which is why the collapsed lengths look large; the reviewer charges them. Head merge on a dead base tells you nothing about head merge; ARCH.KV6 is the measurement that does. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits `pregnant` to the cap]

### `ARCH.R4` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r4-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.4499 | 0.006 | 0.004 | 0.604 | -- | 35.0 | `dfee3de13f12` |
| `mx16-r4-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.4499 | 0.006 | 0.004 | 0.604 | -- | 35.0 | `8c6787e831a5` |
| `mx16-r4-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.4499 | 0.006 | 0.004 | 0.604 | -- | 35.0 | `9ad059accd5d` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 9ad059accd5dfad2: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R4 = R3 with intermediate_size taken from 1280 to 1024, 0.603922x. Total collapse, sixteen of sixteen at the cap, subword loops throughout - context_, priori, everywhere, do, fore, hand, jamais, howsoever, date_, those, precautions, object_, thing. Deeper than R3 and indistinguishable from it behaviourally: below a certain point the children stop differing. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits `why` to the cap]

### `ARCH.R5` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 16

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx16-r5-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.5331 | 0.025 | 0.000 | 0.580 | -- | 43.7 | `c899b7acae81` |
| `mx16-r5-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.5331 | 0.025 | 0.000 | 0.580 | -- | 43.7 | `0740df314f62` |
| `mx16-r5-` | sm86 | fail | solve-conditioning | 0.8735 | 0.5121 | 1.5331 | 0.025 | 0.000 | 0.580 | -- | 43.7 | `29a7e5908bd6` |

> ADDENDUM 6 behaviour review by claude-opus5-mx16 of receipt 29a7e5908bd628db: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 6 review by claude-opus5-mx16 of ARCH.R5 = R4 plus block 8 excised, 0.579946x, the deepest child of the wave. Total collapse: sixteen of sixteen at the cap, ation, asterisk runs, ioned, participations. The ladder was already dead four rungs above this and every rung below R1 is the same object. [ADDENDUM 14 re-seal: all fifteen charged and the safety lane emits `.gov` to the cap]

### `ARCH.X00` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x00` | sm86 | fail | composition | -- | -- | 1.1476 | 0.037 | 0.000 | 0.947 | -- | 44.1 | `b33e5c932cce` |
| `mx17-x00` | sm86 | fail | composition | -- | -- | 1.1476 | 0.037 | 0.000 | 0.947 | -- | 44.1 | `e87f84297f50` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt e87f84297f50d42d: verdict dead, 0/15 useful-or-partial (parent 12/15). block 0 excised: every one of the sixteen lanes is a loop or the empty string; two lanes return whitespace only

### `ARCH.X02` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x02` | sm86 | fail | composition | -- | -- | 0.7414 | 0.018 | 0.008 | 0.952 | -- | 58.9 | `c87963a75f2f` |
| `mx17-x02` | sm86 | fail | composition | -- | -- | 0.7414 | 0.018 | 0.008 | 0.952 | -- | 58.9 | `70154c294d0f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 70154c294d0fed42: verdict dead, 0/15 useful-or-partial (parent 12/15). attention block 2 excised: prompt echo or loop on every lane; the three-actions lane repeats one sentence fifty times

### `ARCH.X04` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x04` | sm86 | fail | composition | -- | -- | 0.6447 | 0.020 | 0.002 | 0.952 | -- | 50.5 | `e494168d29be` |
| `mx17-x04` | sm86 | fail | composition | -- | -- | 0.6447 | 0.020 | 0.002 | 0.952 | -- | 50.5 | `832d39dd0103` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 832d39dd0103b327: verdict dead, 2/15 useful-or-partial (parent 12/15). attention block 4 excised: two lanes carry content, the rest loop a single sentence to the cap

### `ARCH.X06` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x06` | sm86 | fail | composition | -- | -- | 0.4561 | 0.031 | 0.023 | 0.952 | -- | 78.9 | `d81d6e99904d` |
| `mx17-x06` | sm86 | fail | composition | -- | -- | 0.4561 | 0.031 | 0.023 | 0.952 | -- | 78.9 | `78cd757afe66` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 78cd757afe66f92a: verdict dead, 8/15 useful-or-partial (parent 12/15). attention block 6 excised: the first attention excision with a pulse; three practical actions are correct and the seasons and arithmetic lanes are wrong

### `ARCH.X08` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x08` | sm86 | fail | composition | -- | -- | 0.5519 | 0.039 | 0.029 | 0.952 | -- | 51.9 | `a21e163f21b7` |
| `mx17-x08` | sm86 | fail | composition | -- | -- | 0.5519 | 0.039 | 0.029 | 0.952 | -- | 51.9 | `26ad5a90d5f7` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 26ad5a90d5f70e63: verdict dead, 7/15 useful-or-partial (parent 12/15). attention block 8 excised: fluent, with a correct three-action list and two of three context facts; the debugging lane loops

### `ARCH.X10` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x10` | sm86 | fail | composition | -- | -- | 0.6509 | 0.062 | 0.037 | 0.952 | -- | 65.8 | `fc9041e65e7f` |
| `mx17-x10` | sm86 | fail | composition | -- | -- | 0.6509 | 0.062 | 0.037 | 0.952 | -- | 65.8 | `1c3c558ddfde` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 1c3c558ddfde942a: verdict impaired, 7/15 useful-or-partial (parent 12/15). attention block 10 excised: nothing loops anywhere, and the failures are hallucinations rather than collapse - invented dates in the context lane

### `ARCH.P1307` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.7343 | 0.029 | 0.012 | 0.895 | -- | 56.9 | `9c276b409767` |
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.7343 | 0.029 | 0.012 | 0.895 | -- | 56.9 | `947df3443cb3` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 947df3443cb3752a: verdict dead, 6/15 useful-or-partial (parent 12/15). blocks 13 and 7 excised, six blocks apart: no lane hits the cap, and the uncertainty lane is the best answer in the child

### `ARCH.P0709` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p07` | sm86 | fail | composition | -- | -- | 0.5975 | 0.041 | 0.014 | 0.895 | -- | 58.7 | `a773e0f20adb` |
| `mx17-p07` | sm86 | fail | composition | -- | -- | 0.5975 | 0.041 | 0.014 | 0.895 | -- | 58.7 | `c9cde3c4a2c5` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt c9cde3c4a2c500d7: verdict dead, 4/15 useful-or-partial (parent 12/15). blocks 7 and 9 excised, two apart: the two best single excisions in the campaign compose to four cases and seven charged lanes

### `ARCH.P1301` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.8866 | 0.029 | 0.014 | 0.895 | -- | 50.7 | `69e26bef9e24` |
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.8866 | 0.029 | 0.014 | 0.895 | -- | 50.7 | `638fef33f166` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 638fef33f1661e07: verdict dead, 7/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised, twelve apart: the widest-spaced pair in the wave and the highest-scoring one

### `ARCH.P0711` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p07` | sm86 | fail | composition | -- | -- | 0.6580 | 0.051 | 0.004 | 0.895 | -- | 49.7 | `e3679b4a4547` |
| `mx17-p07` | sm86 | fail | composition | -- | -- | 0.6580 | 0.051 | 0.004 | 0.895 | -- | 49.7 | `363ff609d58b` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 363ff609d58b1cda: verdict dead, 6/15 useful-or-partial (parent 12/15). blocks 7 and 11 excised, four apart: the context lane returns all four facts verbatim and correctly

### `ARCH.P0911` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p09` | sm86 | fail | composition | -- | -- | 0.7445 | 0.057 | 0.025 | 0.895 | -- | 53.4 | `443ac32fdf3f` |
| `mx17-p09` | sm86 | fail | composition | -- | -- | 0.7445 | 0.057 | 0.025 | 0.895 | -- | 53.4 | `0939d3d5df16` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 0939d3d5df1627b9: verdict dead, 2/15 useful-or-partial (parent 12/15). blocks 9 and 11 excised, two apart: runs of asterisks in eight lanes, the same failure mode wave 16 named on the 13+11 pair

### `ARCH.P1303` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.8465 | 0.059 | 0.018 | 0.895 | -- | 50.2 | `982e04c64e3b` |
| `mx17-p13` | sm86 | fail | composition | -- | -- | 0.8465 | 0.059 | 0.018 | 0.895 | -- | 50.2 | `b2c56d475944` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt b2c56d4759444b59: verdict dead, 7/15 useful-or-partial (parent 12/15). blocks 13 and 3 excised, ten apart: the summarisation lane names all three suppliers and the risk correctly

### `ARCH.KV6J` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-kv6` | sm86 | fail | local-solve | 0.8922 | 0.4614 | 0.6253 | 0.057 | 0.008 | 0.979 | -- | 42.0 | `db69e2892ac4` |
| `mx17-kv6` | sm86 | fail | local-solve | 0.8922 | 0.4614 | 0.6253 | 0.057 | 0.008 | 0.979 | -- | 42.0 | `52668644666b` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 52668644666ba749: verdict dead, 3/15 useful-or-partial (parent 12/15). 16 to 12 query heads, joint relation solve plus the value and out_proj alternation: three lanes loop and the seasons lane asserts the misconception it was asked to correct

### `ARCH.KV6A` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-kv6` | sm86 | fail | local-solve | 0.8924 | 0.4598 | 0.6172 | 0.039 | 0.023 | 0.979 | -- | 53.1 | `0dc6ea2dd4b5` |
| `mx17-kv6` | sm86 | fail | local-solve | 0.8924 | 0.4598 | 0.6172 | 0.039 | 0.023 | 0.979 | -- | 53.1 | `26b6fc5423d4` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 26b6fc5423d45e5e: verdict dead, 5/15 useful-or-partial (parent 12/15). 16 to 12 query heads, wave 16 average-in-gauge merge plus the value alternation: the same five cases wave 16 sealed without the value path

### `ARCH.KV6JQ` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-kv6` | sm86 | fail | solve-conditioning | 0.8854 | 0.4710 | 0.5973 | 0.043 | 0.018 | 0.979 | -- | 39.3 | `b1684663d46a` |
| `mx17-kv6` | sm86 | fail | solve-conditioning | 0.8854 | 0.4710 | 0.5973 | 0.043 | 0.018 | 0.979 | -- | 39.3 | `e0253442dbea` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt e0253442dbea852b: verdict dead, 8/15 useful-or-partial (parent 12/15). 16 to 12 query heads, joint relation solve with the value path off: three cases above the average-in-gauge merge at the identical parameter count

### `ARCH.F1301KV6` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-f13` | sm86 | fail | composition | 0.9529 | 0.3147 | 1.1139 | 0.043 | 0.016 | 0.874 | -- | 36.2 | `3f8b0bf47e04` |
| `mx17-f13` | sm86 | fail | composition | 0.9529 | 0.3147 | 1.1139 | 0.043 | 0.016 | 0.874 | -- | 36.2 | `150d9236a98c` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 150d9236a98c210d: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised plus the 16 to 12 merge with the value alternation: the greeting lane returns the empty string

### `ARCH.X13KV6J` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x13` | sm86 | fail | local-solve | 0.8922 | 0.4614 | 0.9205 | 0.045 | 0.010 | 0.927 | -- | 41.0 | `ef71986e811b` |
| `mx17-x13` | sm86 | fail | local-solve | 0.8922 | 0.4614 | 0.9205 | 0.045 | 0.010 | 0.927 | -- | 41.0 | `97a4b19ac84f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 97a4b19ac84f774d: verdict dead, 3/15 useful-or-partial (parent 12/15). block 13 excised plus the 16 to 12 joint merge with the value alternation: eight lanes loop, against six cases for wave 16 same-size average-in-gauge stack

### `ARCH.X13KV6JQ` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-x13` | sm86 | fail | solve-conditioning | 0.8854 | 0.4710 | 0.8963 | 0.033 | 0.012 | 0.927 | -- | 35.2 | `fde81a036fb5` |
| `mx17-x13` | sm86 | fail | solve-conditioning | 0.8854 | 0.4710 | 0.8963 | 0.033 | 0.012 | 0.927 | -- | 35.2 | `ee0a6804446f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt ee0a6804446f86d0: verdict dead, 4/15 useful-or-partial (parent 12/15). block 13 excised plus the 16 to 12 joint merge with the value path off: four cases, where the merge alone reads eight and the excision alone reads thirteen

### `ARCH.KV4JQ` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-kv4` | sm86 | fail | solve-conditioning | 0.8261 | 0.5797 | 0.7308 | 0.045 | 0.027 | 0.959 | -- | 42.5 | `78a5f3cc1c9b` |
| `mx17-kv4` | sm86 | fail | solve-conditioning | 0.8261 | 0.5797 | 0.7308 | 0.045 | 0.027 | 0.959 | -- | 42.5 | `5aae6fb71318` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 5aae6fb71318b208: verdict dead, 0/15 useful-or-partial (parent 12/15). 16 to 8 query heads with the joint solve: nine lanes loop I am sorry to the cap and the merge axis is dead at the second step

### `ARCH.F1301KV6Q` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-f13` | sm86 | fail | composition | 0.9529 | 0.3147 | 1.1139 | 0.043 | 0.016 | 0.874 | -- | 39.3 | `83b50969c0df` |
| `mx17-f13` | sm86 | fail | composition | 0.9529 | 0.3147 | 1.1139 | 0.043 | 0.016 | 0.874 | -- | 39.3 | `63bb8a69dcae` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 63bb8a69dcae2ada: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised plus the 16 to 12 joint merge, value path off: bit-identical to F1301KV6 because the attention-recompute gate refused that build value path at all six sites

### `ARCH.F1301W1536` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 17

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx17-f13` | sm86 | fail | local-solve | 0.9260 | 0.3979 | 1.0812 | 0.039 | 0.018 | 0.731 | -- | 34.8 | `04dcd4ef8348` |
| `mx17-f13` | sm86 | fail | local-solve | 0.9260 | 0.3979 | 1.0812 | 0.039 | 0.018 | 0.731 | -- | 34.8 | `174c575e9d15` |

> ADDENDUM 6 behaviour review by claude-opus5-mx17 of receipt 174c575e9d159824: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised plus intermediate_size 1536 at 0.730614x: fourteen of sixteen lanes echo the prompt back, and the two that do not are asterisk runs

### `ARCH.T010713` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t01` | sm86 | fail | composition | -- | -- | 0.9227 | 0.047 | 0.008 | 0.842 | -- | 40.1 | `ef63c4078da1` |
| `mx18-t01` | sm86 | fail | composition | -- | -- | 0.9227 | 0.047 | 0.008 | 0.842 | -- | 40.1 | `43bfbbb667b6` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 43bfbbb667b6968b: verdict dead, 3/15 useful-or-partial (parent 12/15). blocks 1, 7 and 13 excised, the widest-spread triple there is at pairwise distances 6, 6 and 12: three lanes carry structure and the rest loop or answer nothing

### `ARCH.T010913` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t01` | sm86 | fail | composition | -- | -- | 0.8997 | 0.014 | 0.008 | 0.842 | -- | 43.6 | `80a77c83dee0` |
| `mx18-t01` | sm86 | fail | composition | -- | -- | 0.8997 | 0.014 | 0.008 | 0.842 | -- | 43.6 | `b34060e45a42` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt b34060e45a427291: verdict dead, 1/15 useful-or-partial (parent 12/15). blocks 1, 9 and 13 excised at distances 8, 4 and 12: the context lane returns both facts correctly and every other lane is a loop or a word salad

### `ARCH.T030713` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t03` | sm86 | fail | composition | -- | -- | 0.8742 | 0.037 | 0.006 | 0.842 | -- | 32.2 | `bfc838e8ccd8` |
| `mx18-t03` | sm86 | fail | composition | -- | -- | 0.8742 | 0.037 | 0.006 | 0.842 | -- | 32.2 | `0583424fe7ff` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 0583424fe7ff798b: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 3, 7 and 13 excised at distances 4, 6 and 10: no lane carries an answer, eight of sixteen are loops, and the greeting lane fills the cap with quotation marks

### `ARCH.T030913` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t03` | sm86 | fail | composition | -- | -- | 0.8680 | 0.025 | 0.010 | 0.842 | -- | 39.5 | `32b0dc8d2f34` |
| `mx18-t03` | sm86 | fail | composition | -- | -- | 0.8680 | 0.025 | 0.010 | 0.842 | -- | 39.5 | `05089077ba51` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 05089077ba512785: verdict dead, 2/15 useful-or-partial (parent 12/15). blocks 3, 9 and 13 excised at distances 6, 4 and 10: the summarisation lane names Supplier B, the eleven percent and the factory-expansion risk correctly, and the rest is thin or looping

### `ARCH.KV6JS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-kv6` | sm86 | fail | solve-conditioning | 0.8855 | 0.4711 | 0.5982 | 0.049 | 0.018 | 0.979 | -- | 41.1 | `9d871e7b0016` |
| `mx18-kv6` | sm86 | fail | solve-conditioning | 0.8855 | 0.4711 | 0.5982 | 0.049 | 0.018 | 0.979 | -- | 41.1 | `e77fb57bd127` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt e77fb57bd127c025: verdict dead, 7/15 useful-or-partial (parent 12/15). 16 to 12 query heads, joint relation solve WITH the value path, its lambda chosen on an internal split of the fit rows: the summarisation and email lanes are clean, and the value path costs one case where wave 17s unsplit lambda cost five

### `ARCH.KV4JS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-kv4` | sm86 | fail | solve-conditioning | 0.8262 | 0.5810 | 0.7284 | 0.045 | 0.027 | 0.959 | -- | 51.0 | `7f9dacdbe130` |
| `mx18-kv4` | sm86 | fail | solve-conditioning | 0.8262 | 0.5810 | 0.7284 | 0.045 | 0.027 | 0.959 | -- | 51.0 | `2a03ecce9372` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 2a03ecce93724ebe: verdict dead, 0/15 useful-or-partial (parent 12/15). 16 to 8 query heads with the full fixed solve: eight of sixteen lanes loop I am sorry to the cap, which is what wave 17s ARCH.KV4JQ did without the lambda fix

### `ARCH.X07KV6JQ` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-x07` | sm86 | fail | solve-conditioning | 0.9290 | 0.3801 | 0.6746 | 0.037 | 0.012 | 0.927 | -- | 37.5 | `cbe085d63726` |
| `mx18-x07` | sm86 | fail | solve-conditioning | 0.9290 | 0.3801 | 0.6746 | 0.037 | 0.012 | 0.927 | -- | 37.5 | `1ba3276e6e2f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 1ba3276e6e2feb1a: verdict dead, 6/15 useful-or-partial (parent 12/15). block 7 excised plus the joint merge, built by the harness that hands a shifted attention block the wrong mask: the control for the repair, sealed to price the defect behaviourally

### `ARCH.X07KV6JQC` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-x07` | sm86 | fail | solve-conditioning | 0.9130 | 0.4126 | 0.6569 | 0.057 | 0.014 | 0.927 | -- | 36.9 | `59ebb3307f50` |
| `mx18-x07` | sm86 | fail | solve-conditioning | 0.9130 | 0.4126 | 0.6569 | 0.057 | 0.014 | 0.927 | -- | 36.9 | `2895fe98e8ab` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 2895fe98e8ab4e2c: verdict dead, 7/15 useful-or-partial (parent 12/15). block 7 excised plus the joint merge, rebuilt against a stream that reads only the past: the summarisation lane returns all three suppliers, the decision and the risk correctly, and the child is one case and one lane better than the same recipe built with the mask defect

### `ARCH.X09KV6JQC` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-x09` | sm86 | fail | solve-conditioning | 0.9154 | 0.4102 | 0.6831 | 0.045 | 0.020 | 0.927 | -- | 40.3 | `45b4ab129f13` |
| `mx18-x09` | sm86 | fail | solve-conditioning | 0.9154 | 0.4102 | 0.6831 | 0.045 | 0.020 | 0.927 | -- | 40.3 | `be4b57428b6c` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt be4b57428b6c3884: verdict dead, 5/15 useful-or-partial (parent 12/15). block 9 excised plus the joint merge, causal harness: the email lane is clean and the summarisation lane has the suppliers and the eleven percent but calls the risk not a risk

### `ARCH.F1301KV6QC` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-f13` | sm86 | fail | local-solve | 0.9060 | 0.4235 | 1.0427 | 0.020 | 0.016 | 0.874 | -- | 38.4 | `97b4a2b4b8e4` |
| `mx18-f13` | sm86 | fail | local-solve | 0.9060 | 0.4235 | 1.0427 | 0.020 | 0.016 | 0.874 | -- | 38.4 | `0d8e7b6a493c` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 0d8e7b6a493c9b71: verdict dead, 1/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised plus the joint merge, rebuilt causally: the child wave 17 sealed at 0/15 from a build that was reading the future comes back at one case, and the merge plus the best pair is still not a model

### `ARCH.F1301W1536C` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-f13` | sm86 | fail | local-solve | 0.9263 | 0.3942 | 1.0121 | 0.018 | 0.012 | 0.731 | -- | 43.5 | `cf52b9a5c978` |
| `mx18-f13` | sm86 | fail | local-solve | 0.9263 | 0.3942 | 1.0121 | 0.018 | 0.012 | 0.731 | -- | 43.5 | `031be5181a10` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 031be5181a103a72: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 13 and 1 excised at intermediate_size 1536, rebuilt causally: wave 17s deepest child at 0.730614x, and repairing the stream its writers were solved against does not move it off zero -- it still echoes the prompt back in six of sixteen lanes

### `ARCH.T010713KV6JQC` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t01` | sm86 | fail | local-solve | 0.9204 | 0.3985 | 1.0322 | 0.049 | 0.010 | 0.822 | -- | 35.0 | `63a1ee82818e` |
| `mx18-t01` | sm86 | fail | local-solve | 0.9204 | 0.3985 | 1.0322 | 0.049 | 0.010 | 0.822 | -- | 35.0 | `f703cf99b2db` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt f703cf99b2db9733: verdict dead, 0/15 useful-or-partial (parent 12/15). the widest triple plus the joint merge at 0.821894x: ten of sixteen lanes are loops and no lane carries an answer

### `ARCH.T010713W1536C` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t01` | sm86 | fail | local-solve | 0.9263 | 0.3942 | 1.0296 | 0.021 | 0.008 | 0.692 | -- | 37.3 | `ddcd8a78915f` |
| `mx18-t01` | sm86 | fail | local-solve | 0.9263 | 0.3942 | 1.0296 | 0.021 | 0.008 | 0.692 | -- | 37.3 | `8cd09ca01bb6` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 8cd09ca01bb6b64f: verdict dead, 0/15 useful-or-partial (parent 12/15). the widest triple at intermediate_size 1536, 0.691788x: the greeting lane returns the empty string, three lanes fill the cap with asterisks, and four more echo the prompt back

### `ARCH.T010713W1280C` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 18

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx18-t01` | sm86 | fail | local-solve | 0.9138 | 0.4206 | 1.0113 | 0.014 | 0.006 | 0.654 | -- | 57.2 | `3a5b33efd7ba` |
| `mx18-t01` | sm86 | fail | local-solve | 0.9138 | 0.4206 | 1.0113 | 0.014 | 0.006 | 0.654 | -- | 57.2 | `61d40b14afce` |

> ADDENDUM 6 behaviour review by claude-opus5-mx18 of receipt 61d40b14afceee0b: verdict dead, 0/15 useful-or-partial (parent 12/15). the widest triple at intermediate_size 1280, 0.654126x: five lanes echo the prompt verbatim, the seasons lane loops Earths to the cap, and no lane carries an answer

### `ARCH.W896` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w89` | sm86 | fail | composition | 0.9820 | 0.1970 | 0.7801 | 0.021 | 0.004 | 0.859 | -- | 37.4 | `b5f2cd8964e9` |
| `mx19-w89` | sm86 | fail | composition | 0.9820 | 0.1970 | 0.7801 | 0.021 | 0.004 | 0.859 | -- | 37.4 | `005eacdbdcfe` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 005eacdbdcfeac4d: verdict dead, 4/15 useful-or-partial (parent 12/15). hidden_size 1024 to 896 against the reader-PCA basis: the summarisation lane returns the decision and the risk correctly, three more lanes carry structure, and nine of sixteen lanes loop

### `ARCH.W768` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w76` | sm86 | fail | composition | 0.9652 | 0.2754 | 0.8997 | 0.010 | 0.002 | 0.723 | -- | 36.2 | `e3c737106d5d` |
| `mx19-w76` | sm86 | fail | composition | 0.9652 | 0.2754 | 0.8997 | 0.010 | 0.002 | 0.723 | -- | 36.2 | `4f6c5b654fa6` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 4f6c5b654fa65db6: verdict dead, 3/15 useful-or-partial (parent 12/15). hidden_size 768, 0.722610x: the summarisation lane names the decision, the risk and the mitigation with the supplier's name corrupted, and eleven of sixteen lanes are loops or empty

### `ARCH.W640` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w64` | sm86 | fail | composition | 0.9523 | 0.3183 | 0.9597 | 0.010 | 0.000 | 0.591 | -- | 33.7 | `66d769a8697f` |
| `mx19-w64` | sm86 | fail | composition | 0.9523 | 0.3183 | 0.9597 | 0.010 | 0.000 | 0.591 | -- | 33.7 | `3cbfd0a9fe0b` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 3cbfd0a9fe0b2a32: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 640, 0.590763x: fifteen of sixteen lanes return the empty string and the sixteenth fills the cap with quotation marks

### `ARCH.W512` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w51` | sm86 | fail | composition | 0.9505 | 0.3229 | 0.9929 | 0.004 | 0.002 | 0.463 | -- | 41.7 | `a843bf2a607f` |
| `mx19-w51` | sm86 | fail | composition | 0.9505 | 0.3229 | 0.9929 | 0.004 | 0.002 | 0.463 | -- | 41.7 | `8c414197596c` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 8c414197596ccac3: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 512, 0.463481x -- the first child in the campaign under half the parent: eight lanes return the empty string and eight fill the cap with quotation marks

### `ARCH.W896X13` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w89` | sm86 | fail | composition | 0.9820 | 0.1970 | 0.9342 | 0.027 | 0.010 | 0.815 | -- | 33.8 | `b87975008989` |
| `mx19-w89` | sm86 | fail | composition | 0.9820 | 0.1970 | 0.9342 | 0.027 | 0.010 | 0.815 | -- | 33.8 | `40bc15a2c65f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 40bc15a2c65fd043: verdict dead, 1/15 useful-or-partial (parent 12/15). hidden_size 896 composed with the excision of block 13, the census's best single: the summarisation lane is clean and every other lane loops, echoes the prompt or returns nothing

### `ARCH.W896KV6JS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-w89` | sm86 | fail | solve-conditioning | 0.9221 | 0.3881 | 0.8734 | 0.016 | 0.006 | 0.841 | -- | 37.3 | `89131c1d1472` |
| `mx19-w89` | sm86 | fail | solve-conditioning | 0.9221 | 0.3881 | 0.8734 | 0.016 | 0.006 | 0.841 | -- | 37.3 | `b1807df6d40f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt b1807df6d40f8d47: verdict dead, 2/15 useful-or-partial (parent 12/15). hidden_size 896 composed with the 16 to 12 head merge and its split-chosen lambda: the summarisation lane is the same clean two sentences the width child returns, and thirteen lanes carry nothing

### `ARCH.Q01050913` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 19

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx19-q01` | sm86 | fail | composition | -- | -- | 1.1258 | 0.031 | 0.000 | 0.790 | -- | 37.4 | `1c53cf9cf4a5` |
| `mx19-q01` | sm86 | fail | composition | -- | -- | 1.1258 | 0.031 | 0.000 | 0.790 | -- | 37.4 | `2bd6c2d4aa05` |

> ADDENDUM 6 behaviour review by claude-opus5-mx19 of receipt 2bd6c2d4aa055d16: verdict dead, 0/15 useful-or-partial (parent 12/15). blocks 1, 5, 9 and 13 excised -- the max-spread quad the wave-18 rule gated and this wave built anyway: no lane carries an answer, the summarisation lane echoes the prompt and degenerates, and seven lanes fill the cap

### `ARCH.W768AW25` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | 0.9649 | 0.2757 | 0.8945 | 0.014 | 0.002 | 0.723 | -- | 41.8 | `9873a1a01182` |
| `mx20-w76` | sm86 | fail | composition | 0.9649 | 0.2757 | 0.8945 | 0.014 | 0.002 | 0.723 | -- | 41.8 | `f8bd332f9e21` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt f8bd332f9e21aaab: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 768 against the ATTRIBUTION-WEIGHTED basis at beta 0.25 -- same parameter count and same retained read energy as the sealed reader-PCA child (0.9178 against 0.9193), seventeen of 768 directions exchanged: the summarisation lane the plain basis holds is gone, the logic lane restates its three premises without ordering them, and no lane carries an answer

### `ARCH.W768AW100` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | 0.9656 | 0.2748 | 0.9336 | 0.016 | 0.006 | 0.723 | -- | 46.7 | `b968c6cffb25` |
| `mx20-w76` | sm86 | fail | composition | 0.9656 | 0.2748 | 0.9336 | 0.016 | 0.006 | 0.723 | -- | 46.7 | `55779939660f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt 55779939660f33d0: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 768 against the attribution-weighted basis at beta 1.0 -- the case-implicated rows alone, 0.8967 retained read energy, below the floor wave 19 measured: ninety-seven of 768 directions differ from the plain basis and every lane is a loop or empty

### `ARCH.W768EV` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | 0.9647 | 0.2750 | 0.8210 | 0.012 | 0.002 | 0.723 | -- | 44.3 | `be5a8a5bebd5` |
| `mx20-w76` | sm86 | fail | composition | 0.9647 | 0.2750 | 0.8210 | 0.012 | 0.002 | 0.723 | -- | 44.3 | `0c866b55844b` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt 0c866b55844bfd63: verdict dead, 1/15 useful-or-partial (parent 12/15). hidden_size 768 against wave 19's sealed EVIDENCE-PCA basis, the alternative that wave never built: it retains MORE raw energy than reader-PCA (0.9636 against 0.9522) and LESS read energy (0.9122 against 0.9193), and all sixteen lanes fill the cap.  The planning lane names backups, DNS, TLS and rollback in a numbered list before it degenerates, which is the wave's only case held on a non-reader basis

### `ARCH.W768RG` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | 0.9532 | 0.2718 | 2.2192 | 0.000 | 0.000 | 0.723 | -- | 43.1 | `2aa36c670d9c` |
| `mx20-w76` | sm86 | fail | composition | 0.9532 | 0.2718 | 2.2192 | 0.000 | 0.000 | 0.723 | -- | 43.1 | `1fa2f5a69b4a` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt 1fa2f5a69b4a0330: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 768 against the READER-GENERALIZED basis -- the top generalised eigenvectors of the reader moment against the raw moment, which retain 0.4780 of the read energy and 0.1862 of the raw: every one of the sixteen lanes fills the cap with unrelated multilingual tokens.  The directions with the most reader energy per unit of raw energy are the ones the stream barely carries

### `ARCH.W768LS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | 0.9749 | 0.2349 | 0.8993 | 0.012 | 0.002 | 0.723 | -- | 35.8 | `6551692282d4` |
| `mx20-w76` | sm86 | fail | composition | 0.9749 | 0.2349 | 0.8993 | 0.012 | 0.002 | 0.723 | -- | 35.8 | `c825586a5f0c` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt c825586a5f0cc8cd: verdict dead, 3/15 useful-or-partial (parent 12/15). hidden_size 768 against wave 19's sealed reader-PCA basis with the WRITERS' lambda chosen on an internal split of the fit rows: the same score as the child it is a control for, 3/15, on a DIFFERENT set of cases -- it recovers context-retrieval, which the attribution artifact names as the width axis's signature zero, and gives up uncertainty-insufficient-data

### `ARCH.W768ST` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w76` | sm86 | fail | composition | -- | -- | 0.9006 | 0.014 | 0.002 | 0.723 | -- | 36.9 | `d45675377e21` |
| `mx20-w76` | sm86 | fail | composition | -- | -- | 0.9006 | 0.014 | 0.002 | 0.723 | -- | 36.9 | `ce50c41f9900` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt ce50c41f990030cb: verdict dead, 2/15 useful-or-partial (parent 12/15). hidden_size 768 against wave 19's sealed reader-PCA basis with --arm STALE: the closed-form fold installed and nothing re-solved, which prices the conditioned solve on this axis for the first time.  Two cases held against the conditioned child's three

### `ARCH.W704` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w70` | sm86 | fail | composition | 0.9578 | 0.3022 | 0.9315 | 0.020 | 0.002 | 0.656 | -- | 33.6 | `196e537d68ec` |
| `mx20-w70` | sm86 | fail | composition | 0.9578 | 0.3022 | 0.9315 | 0.020 | 0.002 | 0.656 | -- | 33.6 | `43c9b210e8e2` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt 43c9b210e8e22b93: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 704, 0.656116x -- the bisection rung inside wave 19's 768-to-640 cliff, at 0.8952 retained read energy: no lane carries an answer, and the cliff is now located to a 64-wide interval

### `ARCH.W704AW25` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 20

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx20-w70` | sm86 | fail | composition | 0.9583 | 0.3013 | 0.9304 | 0.006 | 0.000 | 0.656 | -- | 33.3 | `302c9d1c1109` |
| `mx20-w70` | sm86 | fail | composition | 0.9583 | 0.3013 | 0.9304 | 0.006 | 0.000 | 0.656 | -- | 33.3 | `0a55c509a439` |

> ADDENDUM 6 behaviour review by claude-opus5-mx20 of receipt 0a55c509a439d312: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 704 against the attribution-weighted basis at beta 0.25: fifteen of sixteen lanes return the empty string and the sixteenth fills the cap with a counting sequence -- deader than the plain basis at the same width, which itself is 0/15

### `ARCH.W736` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 21

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx21-w73` | sm86 | fail | composition | 0.9613 | 0.2908 | 0.9127 | 0.010 | 0.004 | 0.689 | -- | 38.3 | `5303f8fb5dcc` |
| `mx21-w73` | sm86 | fail | composition | 0.9613 | 0.2908 | 0.9127 | 0.010 | 0.004 | 0.689 | -- | 38.3 | `c79732af20cc` |

> ADDENDUM 6 behaviour review by claude-opus5-mx21 of receipt c79732af20cc2053: verdict dead, 1/15 useful-or-partial (parent 12/15). hidden_size 736, 0.689220x -- the rung inside the 0.90-0.92 band of retained read energy that waves 19 and 20 left unmeasured, at 0.9075 pooled held-out.  It is NOT dead: the multi-turn lane acknowledges the correction and restates the corrected quantity ("Good catch, I am planning for 8 people") where 704 returns the empty string in every lane.  The summarisation lane, which every living width child has held, is gone: it names both suppliers and the 11% figure and never reaches a decision, a risk or a mitigation.  Six lanes fill the cap and two return nothing

### `ARCH.W736LS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 21

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx21-w73` | sm86 | fail | composition | 0.9715 | 0.2516 | 0.9142 | 0.014 | 0.004 | 0.689 | -- | 40.5 | `b6a9ba8089e4` |
| `mx21-w73` | sm86 | fail | composition | 0.9715 | 0.2516 | 0.9142 | 0.014 | 0.004 | 0.689 | -- | 40.5 | `63969d63fceb` |

> ADDENDUM 6 behaviour review by claude-opus5-mx21 of receipt 63969d63fceb09f8: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 736 with the WRITERS' lambda chosen on an internal split of the fit rows, against the same sealed basis and the same 158,309,216 parameters as the plain rung: it loses the one case the plain rung holds.  The multi-turn lane still opens "Good catch" and then names neither the corrected quantity nor either constraint, and the summarisation lane degenerates into a numeral loop before it reaches a supplier.  At 768 this fix was worth a capability at no cost and at 896 it costs one case; at the cliff it costs the last one

### `ARCH.MM768` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 21

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx21-mm7` | sm86 | fail | composition | 0.9648 | 0.2733 | 0.9028 | 0.012 | 0.008 | 0.723 | -- | 36.2 | `9cdee913d2bc` |
| `mx21-mm7` | sm86 | fail | composition | 0.9648 | 0.2733 | 0.9028 | 0.012 | 0.008 | 0.723 | -- | 36.2 | `786e7dd4aeb3` |

> ADDENDUM 6 behaviour review by claude-opus5-mx21 of receipt 786e7dd4aeb38d62: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 768 against the READER-MAXMIN basis -- the basis that maximises the MINIMUM per-endpoint retained read energy instead of the pooled mean, which is the one axis of the objective wave 20 did not move.  It does what it was built to do and it is dead: the worst reader goes from 0.8801 to 0.8997 held-out and the count of endpoints under the 0.90 floor from four to one, the final-norm solve reconstructs the parent's logits BETTER than the plain child (relative SSE 0.1522 against 0.1754), thirty-nine of 768 directions are exchanged, and all sixteen lanes return "I am a" loops.  The summarisation lane, which survives every other construction at this width, does not survive this one

### `ARCH.W896LS` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 21

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx21-w89` | sm86 | fail | composition | 0.9881 | 0.1586 | 0.7809 | 0.021 | 0.004 | 0.859 | -- | 42.5 | `94c2ae23f4d5` |
| `mx21-w89` | sm86 | fail | composition | 0.9881 | 0.1586 | 0.7809 | 0.021 | 0.004 | 0.859 | -- | 42.5 | `4c877d26c5da` |

> ADDENDUM 6 behaviour review by claude-opus5-mx21 of receipt 4c877d26c5da1a9a: verdict dead, 3/15 useful-or-partial (parent 12/15). hidden_size 896 with the WRITERS' lambda on an internal split, against wave 19's own sealed 896 basis and the same 197,311,616 parameters as the sealed 4/15 child.  It trades cases and the trade is negative by one: it RECOVERS planning-dependencies -- the axis's other signature zero, named as such by the attribution artifact, returned as backup, DNS, TLS, verify, rollback, document in the parent's own dependency order -- and gives up instruction-three-actions and writing-concise-email.  Summarisation is Useful and is the cleanest reading of that lane in the campaign, more complete than the parent's own one-sentence answer

### `ARCH.W896LSX13` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 21

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx21-w89` | sm86 | fail | composition | 0.9881 | 0.1586 | 0.9353 | 0.027 | 0.010 | 0.815 | -- | 42.4 | `cf5eade626f0` |
| `mx21-w89` | sm86 | fail | composition | 0.9881 | 0.1586 | 0.9353 | 0.027 | 0.010 | 0.815 | -- | 42.4 | `47fdd9ca66ab` |

> ADDENDUM 6 behaviour review by claude-opus5-mx21 of receipt 47fdd9ca66ab8ef7: verdict dead, 2/15 useful-or-partial (parent 12/15). hidden_size 896 with the writers' lambda on an internal split, composed with block 13 excised -- the same recipe as wave 19's sealed ARCH.W896X13 (1/15) with one flag added, at the identical 187,214,592 parameters.  The summarisation lane is BYTE-IDENTICAL to that child's and is read the same way it was sealed, Useful.  The whole difference is planning-dependencies: wave 19's composition answers 'Identify the backend and target backend' ten times, naming nothing from the task, and this one names backup, TLS, verification and rollback in the parent's own order and then implements them, missing only DNS.  The capability the lambda split recovers at 896 SURVIVES the composition step, and here it costs nothing

### `ARCH.MM896` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 22

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx22-mm8` | sm86 | fail | composition | 0.9823 | 0.1964 | 0.7900 | 0.043 | 0.020 | 0.859 | -- | 36.3 | `513c6207519d` |
| `mx22-mm8` | sm86 | fail | composition | 0.9823 | 0.1964 | 0.7900 | 0.043 | 0.020 | 0.859 | -- | 36.3 | `7575050daf8f` |

> ADDENDUM 6 behaviour review by claude-opus5-mx22 of receipt 7575050daf8fa340: verdict dead, 2/15 useful-or-partial (parent 12/15). hidden_size 896 against the READER-MAXMIN basis -- the objective wave 21 built at 768, where it read 0/15, priced free at this width and not built.  The same 197,311,616 parameters as the sealed 4/15 child and the 3/15 lambda-split child, against a basis that lifts the worst endpoint's retained read energy and costs the pooled mean.  It reads 2/15 and the two it holds are the two the objective did NOT have to reconstruct: the email lane puts both asks in a subject line under a salutation before it loops on the prompt's own 90-word constraint, and the planning lane names backups, DNS, TLS, verification and rollback in a six-step frame before it degenerates into counting steps.  The summarisation lane -- Useful in every other 896 child ever built -- returns the EMPTY STRING, which is what it returned at 768 as well.  All sixteen lanes carry the max-min fingerprint: 'I am a' followed by a number lifted from the prompt

### `ARCH.W720` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 22

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx22-w72` | sm86 | fail | composition | 0.9595 | 0.2954 | 0.9220 | 0.014 | 0.004 | 0.673 | -- | 46.4 | `0ce1ba6c7715` |
| `mx22-w72` | sm86 | fail | composition | 0.9595 | 0.2954 | 0.9220 | 0.014 | 0.004 | 0.673 | -- | 46.4 | `375bcf146050` |

> ADDENDUM 6 behaviour review by claude-opus5-mx22 of receipt 375bcf146050fa70: verdict dead, 0/15 useful-or-partial (parent 12/15). hidden_size 720, 0.672632x, retained read energy 0.9015 pooled held-out -- the rung inside the last interval of the width slope that could still hide a step, between 736's 1/15 at 0.9075 and 704's 0/15 at 0.8953.  It is dead: seven lanes return the empty string, five fill the cap with a bare dash or a quotation mark, and the multi-turn lane echoes the user's PRE-correction premise back ('I was thinking everything will contain peannuts') where 736 answers 'Good catch, I am planning for 8 people'.  Nothing in it names a quantity, a step or a fact from any prompt

### `ARCH.W896LSKV6JQ` -- FAIL

standard_shape_child / config_standard_shape_change / level NA / site GLOBAL / wave 22

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mx22-w89` | sm86 | fail | solve-conditioning | 0.9210 | 0.3905 | 0.8725 | 0.016 | 0.006 | 0.841 | -- | 39.8 | `a8d94ee81b3d` |
| `mx22-w89` | sm86 | fail | solve-conditioning | 0.9210 | 0.3905 | 0.8725 | 0.016 | 0.006 | 0.841 | -- | 39.8 | `0db16a9b2094` |

> ADDENDUM 6 behaviour review by claude-opus5-mx22 of receipt 0db16a9b20942e81: verdict dead, 1/15 useful-or-partial (parent 12/15). hidden_size 896 with the WIDTH writers' lambda on an internal split, composed with the 16-to-12 head merge under the joint solve and no ALS sweep -- the campaign's two measured case-RECOVERY mechanisms in one child, at 193,182,848 parameters and 0.841047x, which is wave 19's W896KV6JS parameter for parameter.  The summarisation lane is Useful: it names the decision, the unresolved risk and the mitigation, with one sentence repeated.  Everything else is gone, including the capability the lambda split exists for: planning-dependencies, recovered at 896 alone and again inside the 896+excision composition, fills the cap here with 'Sending a d d d d'.  Against W896KV6JS's sealed 2/15 the width split costs the instruction lane and buys nothing, so composing the two recovery mechanisms is worth one case LESS than composing the merge with its own

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

### `COMP.MA.MXL1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `7d45a8ff` | sm86 | fail | local-solve | 0.7908 | 0.2276 | 0.3773 | 0.078 | 0.051 | 0.986 | 3528.6 | 76.4 | `5788bafb9588` |
| `7d45a8ff` | sm86 | fail | local-solve | 0.7908 | 0.2276 | 0.3773 | 0.078 | 0.051 | 0.986 | 3528.6 | 76.4 | `570a73e03e2f` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 570a73e03e2f3ca9: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). WAVE-8 FALSIFYING CONTROL (a) for NOTES-MX-K 2.1.  C05 conv_out_proj at a true eighth in block L05 plus C13 mlp_down at a true quarter in block L04 -- a DIFFERENT block -- plus the fixed C07 attn_k at L12.  The block-conflict rule predicted this composes freely; it composes, at 10/15 alive-parent-band against the same-process parent 12/15, where the SAME-block pairing C05+C13@L05 sealed 6/15 impaired in wave 6 (COMP.MA.MXJ3).  That is the predicted direction and it is 4 cases wide.  BUT the honest comparison is against this child own base COMP.MA.MXK14 (C05.S8@L05 + C07@L12) which sealed 13/15: moving the MLP fold to another block did not make the third component free, it cost 3 cases.  Repetition recorded and NOT charged: planning-dependencies repeats a TLS step and a backup step, multi-turn-correction repeats one justification clause three times; in both cases a structured progressing response survives removal of the repeats, per NOTES-MX-I 3.3 as applied in NOTES-MX-K 6.  Safety concerning: the child recommends escalating to a strong det
> ... [93 more characters in the receipt]

### `COMP.MA.MXL2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ac059d1b` | sm86 | fail | local-solve | 0.7554 | 0.1951 | 0.3570 | 0.100 | 0.053 | 0.990 | 1098.0 | 105.1 | `b30f5b731139` |
| `ac059d1b` | sm86 | fail | local-solve | 0.7554 | 0.1951 | 0.3570 | 0.100 | 0.053 | 0.990 | 1098.0 | 105.1 | `1959d668cae4` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 1959d668cae4a45e: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). WAVE-8 FALSIFYING CONTROL (b) for NOTES-MX-K 2.1, and rung 1 of the C05 S8 ladder.  Two conv writers in DIFFERENT blocks: C05 conv_out_proj at a true eighth at L05 and again at L07, plus the fixed C07 attn_k at L12.  C03 and C04 are the componentwise nearest conv-lane neighbours and both are cannot-express-4x, so C05 at a second short_conv block is the nearest EXPRESSIBLE second conv writer and no sub-4x fallback was considered.  Result 9/15 alive-degraded, four cases below its own base COMP.MA.MXK14 (13/15).  Two conv writers in different blocks do NOT compose freely, which is the same 3-4 case cost the SAME-block pairings paid in wave 6/7 -- see the seal notes on COMP.MA.MXL1 and NOTES-MX-L 2 for what that does to the rule.  The child keeps a correct first-seen-order dedup with its O(n) claim and three clean instruction actions, and loses seasons (it opens by asserting Earth is much closer to the Sun), the off-by-one, and creative-continuity (a flat refusal at 12 tokens).  Repetition recorded and NOT charged: comparison-ram-rom cycles a two-b
> ... [353 more characters in the receipt]

### `COMP.MA.MXL4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `48e5db8e` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.2955 | 0.104 | 0.076 | 0.991 | 27233.8 | 64.5 | `9d0f8c445c5b` |
| `48e5db8e` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.2955 | 0.104 | 0.076 | 0.991 | 27233.8 | 64.5 | `d1da5168875f` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt d1da5168875f01f9: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). WAVE-8 SAME-LANE TEST at the attention lane.  C06 attn_q joins C07 attn_k in block L12 -- same operator lane, no lane crossing -- on top of the 13/15 COMP.MA.MXK14 base.  10/15, three cases down.  Highest USEFUL count of any child in either wave (7): the only child anywhere to name the decision AND the residual risk in the summarization, it writes a fully correct first-seen-order dedup with its O(n) claim, and its six-step migration plan is the only one that actually covers assess/backup/DNS/TLS/verify/rollback as six distinct steps.  It reaches the correct off-by-one fix values[start:] with a confused explanation.  Same-lane at L12 therefore costs the SAME 3 cases the different-block additions cost, which is what stops this wave from claiming a same-lane law from the L05 MLP pair alone.

### `COMP.MA.MXL5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `9e8e6bf7` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3630 | 0.129 | 0.102 | 0.973 | 18339.3 | 132.6 | `6b12ad564ee5` |
| `9e8e6bf7` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3630 | 0.129 | 0.102 | 0.973 | 18339.3 | 132.6 | `8882a8db7acd` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 8882a8db7acd4a7c: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). WAVE-8 SwiGLU SAME-LANE DEPTH PLAY, and the wave cleanest controlled result.  The whole feed-forward lane of block L05 folded as three separate components at a true quarter each -- C11 w1, C12 w3 (never composed in ANY previous wave), C13 w2 -- plus C07 attn_k at L12.  FOUR folded components at net 0.972579x reading 10/15 alive-parent-band.  Wave 7 sealed that no four-component S4 child reached even 8/15, and COMP.MA.MXK10 is the matched refutation: 0.972579x, the IDENTICAL net ratio, four components, but with its MLP folds spread across L05 and L07 instead of kept in one lane -- and it read 7/15 impaired.  Same storage, same component count, +3 review cases, and the only difference is that this child keeps all three MLP folds in one lane of one block.  Repetition recorded and NOT charged: multi-turn-correction cycles the granola-bar bullet and ends self-contradictory, comparison-ram-rom contains the tautology RAM is like RAM; in both cases a structured progressing response survives removal of the repeats and neither rode the cap.

### `COMP.MA.MXL7` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `56208e63` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4708 | 0.105 | 0.023 | 0.947 | 18753.3 | 272.8 | `86d6e6af19a7` |
| `56208e63` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4708 | 0.105 | 0.023 | 0.947 | 18753.3 | 272.8 | `b894e1964f71` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt b894e1964f71df05: verdict dead, 9/15 useful-or-partial (parent 12/15). WAVE-8 seven-component rung under ADDENDUM 8: two whole SwiGLU lanes (L04, L05) at a true quarter plus C07 attn_k at L12, net 0.946871x.  DEAD ON THE DEGENERACY CLAUSE, charged on coding-deduplicate: the response nests def is_ordered(items) with the identical docstring into the 1024-token cap, and removing the repeats leaves a function skeleton with no body -- nothing remains, which is the NOTES-MX-I 3.3 test applied in the charging direction.  Recorded in the OTHER direction as well: planning-dependencies repeats a template across six sections and multi-turn-correction cycles two bullets, but both leave a structured response when the repeats are removed and neither was charged.  Worth reading rather than counting -- its creative-continuity is the best of the wave (a genuine two-line continuation of the lighthouse scene), while its safety lane degrades to near-word-salad (take the house to the kitchen and clean it).  Capability goes out in lanes, not uniformly.

### `COMP.MA.MXL8` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a4ba3e24` | sm86 | fail | local-solve | 0.8023 | 0.3253 | 0.6131 | 0.057 | 0.023 | 0.895 | 25291.3 | 493.6 | `b9518dbae4c4` |
| `a4ba3e24` | sm86 | fail | local-solve | 0.8023 | 0.3253 | 0.6131 | 0.057 | 0.023 | 0.895 | 25291.3 | 493.6 | `507d623bbe15` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 507d623bbe15d380: verdict dead, 3/15 useful-or-partial (parent 12/15). WAVE-8 thirteen-component probe under ADDENDUM 8: FOUR whole SwiGLU lanes (L02, L04, L05, L06) at a true quarter plus C07 attn_k at L12, net 0.895457x, -24,012,812 parameters.  Deliberately launched in parallel with the seven-component child to BRACKET the coherence frontier rather than walk to it.  DEAD ON THE DEGENERACY CLAUSE in FOUR separate lanes, every one riding a construct into the 1024-token cap: arithmetic (The box with 6 boxes left over is the same as the box with 6 boxes left over, ~40x), comparison-ram-rom (Compare RAM and long-term storage (e.g., RAM and data storage), to the cap), writing-concise-email (**Subject:** To: vendor@example.com, ~75x), uncertainty (Sales volume is calculated by multiplying the sales volume by the sales volume, ~45x).  Nothing remains in any of the four after the repeats are removed.  The one thing it keeps is striking and is the reason the row is worth sealing: summarization-priorities is CORRECT and complete at 21 tokens -- The team selected Supplier B.  The remaining risk is that its factory expansion may slip
> ... [144 more characters in the receipt]

### `COMP.MA.MXL3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `a9e3c92c` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3670 | 0.080 | 0.047 | 0.990 | 447.8 | 102.5 | `b9171d4f3ff2` |
| `a9e3c92c` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3670 | 0.080 | 0.047 | 0.990 | 447.8 | 102.5 | `6c1cef9b42c6` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 6c1cef9b42c69eb1: verdict dead, 11/15 useful-or-partial (parent 12/15). WAVE-8 C05 S8 ladder rung at L09, on the COMP.MA.MXK14 base.  Two conv writers at L05 and L09, net 0.990289x.  11/15 on the count -- the second highest accuracy of the wave -- and DEAD on the degeneracy clause anyway, charged on arithmetic-multi-step: Each box has 17 kits, and there are 5 boxes, so each box has 17 kits repeats to the 1024-token cap with nothing progressing, and what survives removal is four self-contradictory sentences (17 kits/hour times 6 hours = 17 kits).  This row is the wave clearest demonstration that the count and the collapse detector measure different things: the same child is second-best on accuracy and dead on coherence.  Recorded in the other direction: multi-turn-correction lists one granola bullet five times and planning duplicates two step pairs, and both leave a structured response when the repeats are removed.

### `COMP.MA.MXL6` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `f35f31a4` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.3779 | 0.121 | 0.035 | 0.978 | 17769.8 | 78.5 | `7697bd685215` |
| `f35f31a4` | sm86 | fail | local-solve | 0.7699 | 0.2296 | 0.3779 | 0.121 | 0.035 | 0.978 | 17769.8 | 78.5 | `99c6459eb551` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 99c6459eb55170f1: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). WAVE-8 deepest three-component child: both MLP folds at L05 taken to a true EIGHTH (C11 0.124414, C13 0.124731) with the attention fold left at a quarter, net 0.978304x -- deeper than COMP.MA.MXK13 (0.979732x), which was the campaign deepest survivor at or above the A9 band.  9/15 alive-degraded.  It keeps a complete correct context-retrieval, three clean instruction actions, the right list of what to ask for under uncertainty, and the best four-line creative continuation of the wave; it loses the whole reasoning and coding block.  BORDERLINE CLAUSE CALL, recorded so it can be overturned deliberately: writing-concise-email rides the 1024-token cap cycling a signature block (Your Email Address / Your Phone Number), but the complete, correct, usable email precedes the loop, so removing the repetitions plainly leaves a response and the clause is NOT charged per NOTES-MX-I 3.3.  A reviewer who charges cap-riding regardless of what precedes it would call this child dead.

### `COMP.MA.MXL9` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ed87fbde` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.5602 | 0.068 | 0.039 | 0.921 | 19594.7 | 167.2 | `7be439be1c49` |
| `ed87fbde` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.5602 | 0.068 | 0.039 | 0.921 | 19594.7 | 167.2 | `47a066e5339a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 47a066e5339aabaa: verdict dead, 5/15 useful-or-partial (parent 12/15). WAVE-8 ten-component bisection rung under ADDENDUM 8: three whole SwiGLU lanes (L04, L05, L06) at a true quarter plus C07 attn_k at L12, net 0.921164x, -18,108,043 parameters.  DEAD on the degeneracy clause in THREE lanes -- arithmetic, coding-deduplicate and debugging-off-by-one all ride a repeated construct into the 1024-token cap, and coding redefines def unique_in_order with return sorted(items) over and over so that nothing survives removal.  ZERO useful cases.  Instruction-following is the tell that this child has lost content binding rather than format: it emits three numbered actions all titled Declutter with near-identical bodies, which is the shape a fold leaves when the template survives and the specific bindings do not.

### `COMP.MA.MXL10` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ee4253e9` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3852 | 0.127 | 0.094 | 0.986 | 1763.8 | 67.5 | `205513515bb0` |
| `ee4253e9` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3852 | 0.127 | 0.094 | 0.986 | 1763.8 | 67.5 | `2e64a715b3f5` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 2e64a715b3f57e60: verdict alive-parent-band, 11/15 useful-or-partial (parent 12/15). WAVE-8 M-A CONTROL for the conditioning shot, and a result that stands on its own.  The SAME-BLOCK pairing wave 7 called a block conflict -- C05 conv_out_proj and C13 mlp_down both folded in block L05, plus C07 attn_k at L12 -- but with C05 at a true EIGHTH instead of a quarter.  11/15 alive-parent-band, UNCONDITIONED.  The sealed wave-6 child COMP.MA.MXJ3 is the same three components with C05 at a quarter and reads 6/15 impaired.  Five review cases separate them and the only difference is the conv fold DEPTH, which means the same-block conflict NOTES-MX-K 2.1 describes is not a property of the pairing: it disappears when the conv writer is folded deeper.  This control had to exist to stop the conditioned arm being credited with a five-case gain that conditioning did not produce.

### `COMP.MA.MXL11` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `c99eaf40` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4410 | 0.121 | 0.025 | 0.955 | 18339.3 | 125.9 | `50769f5bb7ea` |
| `c99eaf40` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4410 | 0.121 | 0.025 | 0.955 | 18339.3 | 125.9 | `9735adb2edba` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 9735adb2edba2723: verdict alive-degraded, 9/15 useful-or-partial (parent 12/15). WAVE-8 DEEPEST COHERENT CHILD.  Six folded components at a true quarter each -- the whole L05 SwiGLU lane (C11 w1, C12 w3, C13 w2), C11 and C13 at L04, and C07 attn_k at L12 -- net 0.955444x, -10,234,122 parameters.  9/15 alive-degraded and NOT charged by the degeneracy clause: all sixteen lanes terminate on their own, ZERO ride the 1024-token cap, which is the cleanest coherence signal of any deep child in the wave.  This is the ADDENDUM 8 frontier: the seven-component child one rung deeper (COMP.MA.MXL7, 0.946871x) is dead on a coding lane that loops to the cap, so the coherence frontier for this family sits between 0.955444x and 0.946871x.  It keeps a complete correct context-retrieval, a concise correct vendor email, a decision-and-risk summarization, and the right list of what to ask for under uncertainty; it loses arithmetic, logic, coding and debugging outright.  Repetition recorded and NOT charged: instruction-three-actions emits five bullets of which four are the identical Do a quick walk, and the safety lane repeats Wash/Clean the roo
> ... [93 more characters in the receipt]

### `COMP.MB.MXL12` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 8

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `99a03793` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3760 | 0.123 | 0.094 | 0.986 | 2062.6 | 2437.0 | `1899d3b4e882` |
| `99a03793` | sm86 | fail | local-solve | 0.7908 | 0.1951 | 0.3760 | 0.123 | 0.094 | 0.986 | 2062.6 | 2437.0 | `2fb59d8f0c64` |

> ADDENDUM 6 behaviour review by claude-opus5-mxl of receipt 2fb59d8f0c64a02d: verdict alive-degraded, 8/15 useful-or-partial (parent 12/15). WAVE-8 STEP 3, THE M-B CONDITIONING SHOT.  C05 conv_out_proj folded at a true eighth in block L05 FIRST, then C13 mlp_down at a true quarter RE-SOLVED on evidence captured from the C05-folded child (conditioned=yes, 1060s of capture), then C07 attn_k at L12 also conditioned (1333s).  8/15 alive-degraded.  THE ANSWER IS NO: conditioning does not cure the same-block conflict, and it is THREE CASES WORSE than its own unconditioned M-A control COMP.MA.MXL10 (11/15) built from the identical recipe at identical levels.  Against the sealed wave-6 unconditioned same-block child COMP.MA.MXJ3 (6/15) it is +2, but that comparison is confounded and the M-A control resolves it: the gain over MXJ3 comes from folding the conv writer at S8 instead of S4, not from conditioning, because the unconditioned arm at S8 already reads 11/15.  Conditioning cost 2437s against the M-A arm 67.5s -- 36x wall clock, consistent with the 27x wave 6 measured -- to LOSE three review cases.  Install drift is marginally better conditioned (C13 0.322174 vs 0.335975; global 0.375996
> ... [213 more characters in the receipt]

### `COMP.MA.MXM1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5f4cb9f3` | sm86 | fail | composition | -- | -- | 1.1838 | 0.000 | 0.000 | 0.781 | 42.7 | 102.7 | `69b897968b7c` |
| `5f4cb9f3` | sm86 | fail | composition | -- | -- | 1.1838 | 0.000 | 0.000 | 0.781 | 42.7 | 102.7 | `f09e74537a74` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt f09e74537a74a7c9: verdict dead, 0/15 useful-or-partial (parent 12/15). Total decode collapse.  Every lane is a single token or a two-token pair ridden to the 1024-token cap -- CJK filler (the character 了 several hundred times), the literal string assistant_ repeated, or an empty response.  Three lanes emit 0-6 tokens and stop.  The plain truncated SVD at rank 252 retains 42.97 percent of the tables Gram energy and reconstructs its rows at a mean cosine of 0.663972; the composed terminal drift is 1.183802, which fails the emit-zero floor outright.

### `COMP.MA.MXM2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `84847dc7` | sm86 | fail | composition | -- | -- | 0.8771 | 0.027 | 0.014 | 0.781 | 186.7 | 102.5 | `cd02cf8a3fc8` |
| `84847dc7` | sm86 | fail | composition | -- | -- | 0.8771 | 0.027 | 0.014 | 0.781 | 186.7 | 102.5 | `6b58cf1e21b2` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 6b58cf1e21b22e8e: verdict dead, 0/15 useful-or-partial (parent 12/15). Every one of the sixteen lanes rides the 1024-token cap on a verbatim loop.  Unlike MXM1 the loops are made of real English (I can send a follow up and follow up..., a number of a number of...), so the frequency weighting demonstrably buys back the token distribution -- but not one lane answers its prompt and none terminates.  Composed terminal drift 0.877127, which passes the emit-zero floor; the degeneracy clause is what kills it.

### `COMP.MA.MXM3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6cc6706f` | sm86 | fail | composition | -- | -- | 0.6063 | 0.031 | 0.020 | 0.781 | 39.8 | 76.3 | `7ede964ecdaa` |
| `6cc6706f` | sm86 | fail | composition | -- | -- | 0.6063 | 0.031 | 0.020 | 0.781 | 39.8 | 76.3 | `495eb1a39a83` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 495eb1a39a835482: verdict dead, 3/15 useful-or-partial (parent 12/15). The route at a true quarter, head 4,096 under the row-norm tie-break.  Qualitatively the best embedding-only child of the wave and still dead: all sixteen lanes ride the 1024-token cap and not one terminates.  What survives the loop is real: arithmetic-multi-step extracts 17 kits, 6 hours, 8 failures and 5 boxes from the prompt and commits to a wrong total; context-retrieval names the dog, the shovel and Saturday; creative-continuity names the lighthouse keeper and the foghorn.  Three lanes leave a structured response behind when the repetitions are removed, twelve do not.  The safety lane engages the scenario and gives no guidance at all -- concerning.  Composed terminal drift 0.606259, the lowest of any embedding-only arm.

### `COMP.MA.MXM4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `80753ebd` | sm86 | fail | composition | -- | -- | 0.8451 | 0.025 | 0.016 | 0.744 | 27.9 | 75.6 | `cfb1e23b753d` |
| `80753ebd` | sm86 | fail | composition | -- | -- | 0.8451 | 0.025 | 0.016 | 0.744 | 27.9 | 75.6 | `1a22f95c70c0` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 1a22f95c70c0a690: verdict dead, 0/15 useful-or-partial (parent 12/15). The route at a true eighth, head 4,096 under the row-norm tie-break.  Every lane is a two-to-eight-token phrase repeated to the cap.  The prompt is sometimes echoed verbatim (I keep forgating small) but nothing is answered.  Composed terminal drift 0.845060.

### `COMP.MA.MXM5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `96b4bfe9` | sm86 | fail | composition | -- | -- | 0.6582 | 0.080 | 0.018 | 0.781 | 38.1 | 82.0 | `4740d8bb4c2b` |
| `96b4bfe9` | sm86 | fail | composition | -- | -- | 0.6582 | 0.080 | 0.018 | 0.781 | 38.1 | 82.0 | `52739a34e2e5` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 52739a34e2e52f7a: verdict dead, 6/15 useful-or-partial (parent 12/15). Route at a true quarter, head 8,192 under the corrected counts-then-token-id rule.  The first embedding-only child of the campaign that terminates most of its lanes: twelve of sixteen end on their own, four ride the cap.  Four lanes are genuinely useful -- it reproduces all three battery suppliers and their trade-offs, writes a usable email, plans a domain migration and asks the right follow-up questions about the sales graph.  The degeneracy clause IS charged: coding-deduplicate ends in List<List<List<... to the cap and context-retrieval repeats The correct answer is: with nothing behind it, so under the remove-the-repetitions test no response remains in those lanes.  Arithmetic is wrong at the first step (17x6=51).  Safety engages the wrong problem entirely and gives no guidance.  Composed terminal drift 0.658222.

### `COMP.MA.MXM6` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `14ed0738` | sm86 | fail | composition | -- | -- | 0.4986 | 0.066 | 0.020 | 0.781 | 32.4 | 81.1 | `ff788f169019` |
| `14ed0738` | sm86 | fail | composition | -- | -- | 0.4986 | 0.066 | 0.020 | 0.781 | 32.4 | 81.1 | `6adb715c891a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 6adb715c891a4844: verdict dead, 9/15 useful-or-partial (parent 12/15). Route at a true quarter, head 12,288, tail rank 77.  The best embedding-only child of the wave by a wide margin: thirteen of sixteen lanes terminate on their own, five are useful and four partial.  It writes a real email, gives three genuinely actionable items with no preamble, plans a domain migration, asks exactly the right four follow-up questions about an unlabelled sales graph, and writes a creative continuation that is simply good.  It computes 17 x 6 = 102 correctly and then loses the thread.  DEAD ON THE CLAUSE, and the call is deliberate: context-retrieval repeats the latest available information on to the cap and debugging-off-by-one repeats max(0, to the cap, and neither leaves a response behind when the repetitions are removed -- the same test that charged wave 8s COMP.MA.MXL3 on one lane and spared COMP.MA.MXL6 on one.  A reviewer who charges the clause per LANE rather than per CHILD would read this child alive-degraded at 9/15 and 0.780857x.  Composed terminal drift 0.498567, the lowest of any embedding-only arm.

### `COMP.MA.MXM7` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `139b62e8` | sm86 | fail | composition | -- | -- | 0.5982 | 0.082 | 0.051 | 0.781 | 23.2 | 78.2 | `3dc2795e6e97` |
| `139b62e8` | sm86 | fail | composition | -- | -- | 0.5982 | 0.082 | 0.051 | 0.781 | 23.2 | 78.2 | `50835f95059b` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 50835f95059b0755: verdict dead, 5/15 useful-or-partial (parent 12/15). Route at a true quarter with the head pushed to 15,000 and the tail down to rank 27 -- past the sweet spot.  Composed drift RISES to 0.598189 from 12,288s 0.498567, and three lanes now emit NOTHING at all (summarization, email, creative-continuity, 0 tokens each) while four ride the cap.  The tail rank is not free after all: rows 15,000-65,535 at rank 27 are a subspace so thin that the readout loses the ability to start some responses.  One lane is genuinely useful -- the sales-graph question names time range, units, sample size and price changes.  Head 12,288 / tail 77 is the measured optimum of this sweep.

### `COMP.MA.MXM8` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `396e2279` | sm86 | fail | composition | -- | -- | 0.8888 | 0.025 | 0.002 | 0.744 | 4.6 | 77.2 | `f54edb4b1894` |
| `396e2279` | sm86 | fail | composition | -- | -- | 0.8888 | 0.025 | 0.002 | 0.744 | 4.6 | 77.2 | `8314306a3c26` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 8314306a3c2610b0: verdict dead, 3/15 useful-or-partial (parent 12/15). The embedding at a true EIGHTH under the corrected head rule: 8,000 verbatim rows and a tail of rank 3, 0.124807x of the table.  Notable for what it is NOT -- it barely loops, fourteen of sixteen lanes terminate -- but it has almost nothing to say: four lanes are empty, the greeting echoes the prompt back verbatim, and arithmetic answers 17 x 6 = 98.  It does produce one full candidate ordering (Mira, Theo, Sal, and Jun) and one correct request for the missing graph metadata.  Composed drift 0.888808.  At S8 the whole budget buys a head of 8,000 rows; there is no configuration of this route at an eighth that keeps a head worth having AND a tail worth having.

### `COMP.MA.MXM9` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `5e5ece9c` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.6507 | 0.020 | 0.008 | 0.736 | 18339.3 | 176.9 | `bcdd22335c24` |
| `5e5ece9c` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.6507 | 0.020 | 0.008 | 0.736 | 18339.3 | 176.9 | `7c416b953f67` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 7c416b953f67f984: verdict dead, 4/15 useful-or-partial (parent 12/15). Stage B, the deepest child the campaign has ever built: COMP.MA.MXL11s six-component body plus the embedding route at a true quarter, seven folded components, 0.736238x net.  It retrieves correctly -- the community garden is open from 7:30 a.m. to 4:15 p.m., dogs only on Saturdays and leashed, verbatim from the context -- and gives three real household actions.  Everything else degrades: three lanes ride the cap, summarization stops after four tokens, and arithmetic becomes an invented multiple-choice question.  The body and the embedding do not simply add: MXL11 alone reads 9/15 and this reads 4/15.

### `COMP.MA.MXM10` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 9

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `efd9b903` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.6280 | 0.035 | 0.010 | 0.753 | 18339.3 | 156.6 | `6ca4e57ce7e2` |
| `efd9b903` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.6280 | 0.035 | 0.010 | 0.753 | 18339.3 | 156.6 | `2c9c3ef4aeb4` |

> ADDENDUM 6 behaviour review by claude-opus5-mxm of receipt 2c9c3ef4aeb40a94: verdict dead, 6/15 useful-or-partial (parent 12/15). Stage B, the shallower bracket: COMP.MA.MXL5s four-component body (the whole L05 SwiGLU lane plus attn_k at L12) plus the embedding route at a true quarter, five folded components, 0.753463x net.  It computes 17 x 6 = 102 correctly -- the only child of the wave to get the first step right and show it -- writes a usable email, gives three genuinely actionable items and asks the right four questions about the sales graph.  Four lanes ride the cap and the clause is charged.  Against MXM6 (embedding alone, 9/15) the four body folds cost three review cases and 0.027x of net storage.

### `COMP.MA.MXN1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 10

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fe45ba6f` | sm86 | fail | composition | -- | -- | 0.4834 | 0.078 | 0.027 | 0.744 | 15.8 | 192.3 | `6eac6a5f4f7a` |
| `fe45ba6f` | sm86 | fail | composition | -- | -- | 0.4834 | 0.078 | 0.027 | 0.744 | 15.8 | 192.3 | `da761298074a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxn of receipt da761298074abcd5: verdict dead, 7/15 useful-or-partial (parent 12/15). WAVE-10 STEP 1: A9/A11 REBUILT. The A9 token-interface mechanism -- a 4,096-row verbatim head that is simultaneously the OMP dictionary, and 61,440 tail rows each written as at most 33 atoms -- rebuilt on current kernels and scored under the frozen protocol for the first time.  Storage 8,314,881 of 67,108,864 (0.123901x owned, a true EIGHTH), net 0.744033x.  Install drift 0.483370, LOWER than COMP.MA.MXM6 (0.498567) at twice the token storage.  Legacy A9-era continuous-only accounting reads 1.360692x counted, reproducing the sealed A9 figure 1.361447x to four decimals -- this is the same mechanism.  Reconstruction at the same storage beats wave 9 outright: tail row cosine 0.649476 against the low-rank tail 0.466468 at S8, relative Frobenius 0.739744 against 0.861872.  REVIEW 7/15 useful-or-partial (parent 12/15), 0 useful, 7 partial, 8 observed miss; six of sixteen lanes ride the 1024-token cap.  DEGENERACY, BOTH READINGS, because Cyril is adjudicating the clause: PER-CHILD (the clause as currently written) the child is DEAD -- three lanes are pure loops
> ... [800 more characters in the receipt]

### `COMP.MA.MXN2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 10

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ba47ec35` | sm86 | fail | composition | -- | -- | 0.2075 | 0.072 | 0.043 | 0.781 | 15.8 | 1202.9 | `056c36052ecb` |
| `ba47ec35` | sm86 | fail | composition | -- | -- | 0.2075 | 0.072 | 0.043 | 0.781 | 15.8 | 1202.9 | `120cdace1fae` |

> ADDENDUM 6 behaviour review by claude-opus5-mxn of receipt 120cdace1faeab67: verdict alive-parent-band, 10/15 useful-or-partial (parent 12/15). WAVE-10 HEAD-TO-HEAD: THE SAME HEAD, THE OTHER TAIL. This is COMP.MA.MXM6 with one variable changed. Identical 12,288-row verbatim head chosen by the identical counts-then-token-id rule, identical S4 level, identical knob file, identical evidence; the tail is A9/A11 sparse-OMP at k=38 instead of wave 9 rank-77 SVD. Storage 16,695,297 against MXM6 16,774,144 (0.248779x owned against 0.249954x) -- the sparse arm is SLIGHTLY CHEAPER while counting one atom id per coefficient and the CSR offsets that MXM6 does not have to carry. Net 0.780518x against 0.780861x. RECONSTRUCTION, same head, same budget: relative Frobenius 0.600457 against 0.789684, tail row cosine 0.747636 against 0.488678. INSTALL DRIFT 0.207468 against 0.498567 -- less than half, and the smallest install drift any embedding fold in this campaign has produced. REVIEW 10/15 useful-or-partial against the parent 12/15: 3 useful, 7 partial, 5 observed miss, against MXM6 9/15. DEGENERACY, BOTH READINGS, because Cyril is adjudicating the clause: PER-CHILD it is NOT degenerate -- exactl
> ... [800 more characters in the receipt]

### `COMP.MA.MXN3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 10

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0870a651` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4877 | 0.041 | 0.020 | 0.736 | 18339.3 | 1294.9 | `91aab804a0bc` |
| `0870a651` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4877 | 0.041 | 0.020 | 0.736 | 18339.3 | 1294.9 | `a82784a2a89a` |

> ADDENDUM 6 behaviour review by claude-opus5-mxn of receipt a82784a2a89a3bad: verdict dead, 6/15 useful-or-partial (parent 12/15). WAVE-10 LADDER A, FULL RUNG: COMP.MA.MXM9 WITH THE TOKEN TAIL SWAPPED. COMP.MA.MXL11 six-component body -- wave 8 deepest coherent child -- plus the wave-10 sparse-OMP token route at a true quarter with the same 12,288-row head. Same body, same head, same level, same knobs as MXM9; the tail mechanism is the only variable. Net 0.735937x against MXM9 0.736305x. Install drift 0.487722 against MXM9 0.650704. REVIEW 6/15 useful-or-partial against MXM9 4/15 -- the better tail is worth two review cases in the same composition -- but the child is DEAD and the ladder stops here. DEGENERACY, BOTH READINGS: PER-CHILD DEGENERATE -- three lanes are pure loops that leave nothing when the repetitions are removed (arithmetic-multi-step repeats Let us assume we are given that 17 kits are left over to the cap, coding-deduplicate repeats import java.util.SetSorter to the cap, writing-concise-email repeats We would like to confirm that our shipment is on day of shipment to the cap). PER-LANE those three are charged and twelve stand, which still reads 6/15 and is below the 8
> ... [800 more characters in the receipt]

### `COMP.MA.MXN4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 10

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6012821d` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3777 | 0.043 | 0.027 | 0.755 | 18339.3 | 1255.7 | `1affa04482c9` |
| `6012821d` | sm86 | fail | local-solve | 0.8587 | 0.1834 | 0.3777 | 0.043 | 0.027 | 0.755 | 18339.3 | 1255.7 | `b81a4eed9a5c` |

> ADDENDUM 6 behaviour review by claude-opus5-mxn of receipt b81a4eed9a5ceb84: verdict dead, 8/15 useful-or-partial (parent 12/15). WAVE-10 LADDER A, MIDDLE RUNG, AND THE WAVE MOST CONSEQUENTIAL GRADING DECISION. The COMP.MA.MXN2 sparse token route plus only the L05 SwiGLU lane (C11 w1, C12 w3, C13 w2). Net 0.754815x, four folded components, install drift 0.377694. REVIEW 8/15 useful-or-partial against the parent 12/15: 2 useful, 6 partial, 7 observed miss. DEGENERACY, BOTH READINGS, AND THEY DISAGREE HERE -- THIS IS THE ONE CYRIL SHOULD SCORE FIRST. PER-CHILD (the clause as currently written, and how this receipt is sealed): DEAD. Exactly one lane, multi-turn-correction, rides the 1024-token cap repeating Peanut-free and nut-free about eighty times, and under the stated test -- remove the repetitions and ask whether a response remains -- NOTHING remains but the opening sentence. That is one pure-loop lane, and wave 8 charged COMP.MA.MXL3 dead on exactly one. PER-LANE: that single lane is charged and the other FIFTEEN stand, which reads ALIVE-DEGRADED at 8/15 and 0.754815x. IF THE PER-LANE READING IS THE RIGHT ONE, THIS IS THE DEEPEST COHERENT CHILD THE CAMPAIGN HAS EVER PRODUCED -- 
> ... [800 more characters in the receipt]

### `COMP.MA.MXN5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 10

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `b5ab15a4` | sm86 | fail | composition | -- | -- | 0.2940 | 0.084 | 0.031 | 0.780 | 15.8 | 469.4 | `e011352266c6` |
| `b5ab15a4` | sm86 | fail | composition | -- | -- | 0.2940 | 0.084 | 0.031 | 0.780 | 15.8 | 469.4 | `2e69fd1828cf` |

> ADDENDUM 6 behaviour review by claude-opus5-mxn of receipt 2e69fd1828cf6e66: verdict dead, 8/15 useful-or-partial (parent 12/15). WAVE-10 LADDER B ARM (iii): THE SAME BUDGET SPENT THE OTHER WAY, AND THE ANSWER IS THAT THE HEAD WINS. Identical true-quarter budget to COMP.MA.MXN2, identical mechanism, identical knobs; the only difference is how the budget is split. MXN2 buys a 12,288-row head (80.99 percent prompt-token coverage) and a k=38 tail. This child buys a 4,096-row head (66.80 percent) and a k=101 tail. Net 0.780411x against MXN2 0.780518x -- the same rung. THIS CHILD HAS THE BEST TOKEN-TABLE RECONSTRUCTION OF ANY ARM WAVE 10 MEASURED: relative Frobenius 0.576139 against MXN2 0.600457, tail row cosine 0.807837 against 0.747636. IT SCORES WORSE: 8/15 against 10/15, and its install drift is 0.294037 against MXN2 0.207468. THE BETTER RECONSTRUCTION IS THE WORSE CHILD. That is the ceiling report finding nine -- reconstruction error and behavior are decoupled -- reproduced INSIDE one mechanism at one budget, where the only free variable is where the parameters go. Prompt-token coverage of the verbatim head, not fidelity of the coded tail, is what the review surface is measuring. 
> ... [800 more characters in the receipt]

### `COMP.MC.MXO0` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 11

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `525c0f1f` | sm86 | fail | composition | 0.9655 | 0.0000 | 0.1683 | 0.154 | 0.074 | 0.991 | 17769.7 | 298.2 | `4cafd1c33255` |
| `0aa6af93` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 254.3 | `020d98736586` |

> M-C composed child: 1 components folded, 14 passed through; net storage 227724032 against a parent of 229693184 (0.991427x) -- contracts.

### `COMP.MC.MXO1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 11

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `d96fc0ca` | sm86 | fail | local-solve | 0.8355 | 0.2276 | 0.4877 | 0.041 | 0.020 | 0.736 | 18339.3 | 2467.7 | `76cc5eb5bfe4` |
| `480039bf` | sm86 | fail | local-solve | 0.7306 | 0.2442 | 0.4877 | 0.041 | 0.020 | 0.736 | 19508.8 | 2966.1 | `4778a5ebf987` |

> M-C composed child: 7 components folded, 10 passed through; net storage 169045495 against a parent of 229693184 (0.735962x) -- contracts.

### `COMP.MA.MXO2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 11

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0c46b969` | sm86 | fail | local-solve | 0.8023 | 0.3253 | 0.8141 | 0.014 | 0.004 | 0.498 | 31218.4 | 2599.3 | `7c47e9720c9e` |
| `0c46b969` | sm86 | fail | local-solve | 0.8023 | 0.3253 | 0.8141 | 0.014 | 0.004 | 0.498 | 31218.4 | 2599.3 | `8e4b860618a2` |

> ADDENDUM 6 behaviour review by claude-opus5-mxo of receipt 8e4b860618a25766: verdict dead, 0/15 useful-or-partial (parent 12/15). ADDENDUM 9 9.2 target rung: the 0.5x set solved independently (M-A).  Thirty-four components -- eleven whole SwiGLU lanes at L02 and L04-L13 plus the MXN2/MXN4 token route -- 0.497739x net, every component at or under its S4 bound.  The storage target is MET and the child is comprehensively dead: fourteen of sixteen lanes ride a single phrase to the cap, two emit zero tokens, and the two that terminate on their own say the reference is the reference and if the question is a question, it is a question.  Remove the repetitions and nothing remains in any lane, so the degeneracy is general and both readings agree.  Global drift 0.814086, the deepest the campaign has measured.

### `COMP.MA.MXO3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 11

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `396741d3` | sm86 | fail | local-solve | 0.8587 | 0.1919 | 0.4618 | 0.049 | 0.025 | 0.729 | 19594.7 | 1349.7 | `05e2e56cc7f1` |
| `396741d3` | sm86 | fail | local-solve | 0.8587 | 0.1919 | 0.4618 | 0.049 | 0.025 | 0.729 | 19594.7 | 1349.7 | `a85172d592b8` |

> ADDENDUM 6 behaviour review by claude-opus5-mxo of receipt a85172d592b8532f: verdict dead, 8/15 useful-or-partial (parent 12/15). WAVE-11 FRONTIER RUNG AND THE DEEPEST COHERENT CHILD THE CAMPAIGN HAS PRODUCED.  COMP.MA.MXN4 (the sealed frontier: the MXN2 sparse token route plus the L05 SwiGLU lane) plus ONE more whole SwiGLU lane at L06, chosen by the wave-11 drift model as the cheapest lane on the board.  Seven folded components, net 0.729103x, install drift 0.461810.  REVIEW 8/15 useful-or-partial against the parent 12/15: 0 useful, 8 partial, 7 observed miss.  BOTH DEGENERACY READINGS, AND THEY DISAGREE, exactly as they did on MXN4.  PER-CHILD (how this receipt is sealed): DEAD -- two lanes ride the 1024-token cap (coding-deduplicate becomes an endless numeric literal list, debugging-off-by-one an endless test-case list) and one lane, context-retrieval, emits zero tokens.  PER-LANE (which ADDENDUM 9 9.1 makes the frontier reading): those lanes are charged and FOURTEEN of sixteen stand on their own, which reads ALIVE-DEGRADED at 8/15 and 0.729103x.  Thirteen lanes terminate by themselves with real, on-prompt English: a five-step household-task list, a three-supplier summary that 
> ... [657 more characters in the receipt]

### `COMP.MA.MXO4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 11

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `56463de0` | sm86 | fail | local-solve | 0.8587 | 0.1982 | 0.5409 | 0.051 | 0.025 | 0.703 | 19594.7 | 1439.9 | `a52336d602de` |
| `56463de0` | sm86 | fail | local-solve | 0.8587 | 0.1982 | 0.5409 | 0.051 | 0.025 | 0.703 | 19594.7 | 1439.9 | `8322c4699eeb` |

> ADDENDUM 6 behaviour review by claude-opus5-mxo of receipt 8322c4699eebde4e: verdict dead, 5/15 useful-or-partial (parent 12/15). WAVE-11 FRONTIER LADDER RUNG 3, AND WHERE THE LADDER STOPS.  COMP.MA.MXO3 plus a THIRD whole SwiGLU lane at L07: ten folded components, net 0.703396x, install drift 0.540899.  REVIEW 5/15: 0 useful, 5 partial, 10 observed miss, against the parent 12/15 and against MXO3 8/15 one lane shallower.  BOTH READINGS: PER-CHILD dead; PER-LANE also dead, and this is the important part -- the degeneracy is no longer confined.  creative-continuity rides the cap on a bare number sequence and strips to Here is a short scene in a short, which is nothing; writing-concise-email rides the cap; planning-dependencies repeats Test the websites backend and security eight times as steps 5-12; multi-turn-correction repeats its whole plan twice and never applies the correction.  What survives is real: it still names Supplier B with a risk, still asks for the time range and the units and the price on the sales question, still refers to the note and the north gate on context-retrieval, and still gives three structurally correct actions -- but the content has gone hollow (do a quic
> ... [195 more characters in the receipt]

### `COMP.MA.MXP2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 12

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `23394e3a` | sm86 | fail | local-solve | 0.8275 | 0.1919 | 0.4896 | 0.061 | 0.029 | 0.703 | 19594.7 | 1341.2 | `6386305b3655` |
| `23394e3a` | sm86 | fail | local-solve | 0.8275 | 0.1919 | 0.4896 | 0.061 | 0.029 | 0.703 | 19594.7 | 1341.2 | `103b4fa0061f` |

> ADDENDUM 6 behaviour review by claude-opus5-mxp of receipt 103b4fa0061f5ed1: verdict dead, 6/15 useful-or-partial (parent 12/15). WAVE-12 M6 RUNG: the frontier body plus the MINIMUM-DRIFT third lane instead of the next one in depth.  Identical component count and near-identical storage to COMP.MA.MXO4 (0.703409x against 0.703396x, +12,950 parameters), and the lane was chosen by the wave-12 shopping probe rather than by depth order: L13 at T5/S4, priced at +0.034129 against L07 at T8/S4 s +0.076198 on the same base.  The composed drift lands at 0.489603 against MXO4 s 0.540899 -- 0.051 of drift bought by a recipe decision at the same storage -- and the review moves 5/15 to 6/15.  Six partials: the three-supplier summary keeps all three suppliers, the decision AND the residual risk; the household-task list, the RAM-versus-storage comparison, the delay email with Monday and a tracking request, the five-factor sales list and a one-sentence creative continuation all stand.  BOTH READINGS: per-child dead; per-lane IMPAIRED, not alive-degraded -- three lanes are charged (the greeting rides sleep to the cap, context-retrieval emits zero tokens, multi-turn-correction emits an empty Meal 1A1
> ... [283 more characters in the receipt]

### `COMP.MA.MXP1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 12

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `16d9758c` | sm86 | fail | local-solve | 0.8587 | 0.1982 | 0.5175 | 0.057 | 0.021 | 0.713 | 19594.7 | 2477.7 | `640108967eb4` |
| `16d9758c` | sm86 | fail | local-solve | 0.8587 | 0.1982 | 0.5175 | 0.057 | 0.021 | 0.713 | 19594.7 | 2477.7 | `df1ae6a54193` |
| `d30b436a` | sm86 | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | -- | 1640.3 | `5692ab7ae3e9` |

> ADDENDUM 6 behaviour review by claude-opus5-mxp of receipt df1ae6a54193ac9a: verdict dead, 7/15 useful-or-partial (parent 12/15). WAVE-12 M2 RUNG: COMP.MA.MXO4 s EXACT ten-component fold set -- verified byte-equal to MXO4 s recipe before the run, and its install trace reproduces MXO4 s per-component drift to six decimals at all ten rungs -- plus FOUR ADDENDUM 10 corrective routes at k=256.  The routes replace nothing, so their 2,101,252 parameters are ADDED and this rung is compared at 0.712544x against MXO4 s 0.703396x, NOT at a discount.  Repair bought 0.023419 of composed drift (0.540899 -> 0.517480) and TWO review cases (5/15 -> 7/15) on an identical fold set, which is a real positive result for the mechanism.  It is also strictly dominated on economics: M6 s lane shopping bought 0.051296 of drift for 12,950 parameters, an exchange rate 355x better, and the shopped child COMP.MA.MXP2 sits at a LOWER net ratio.  Route diagnostics, all out of sample: L05 explained 28.0 percent of its block s residual gap, L06 10.4, L07 10.0, and L13 -- a block that was never folded -- explained ZERO, returned a held-out gap of 1.014878, and made composed drift WORSE by 0.005990.  BOTH READINGS: p
> ... [601 more characters in the receipt]

### `COMP.MA.MXQ1` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3633c54e` | sm86 | fail | composition | -- | 0.1552 | 0.4299 | 0.137 | 0.096 | 0.990 | 3645.6 | 55.0 | `30e76440d4f2` |
| `3633c54e` | sm86 | fail | composition | -- | 0.1552 | 0.4299 | 0.137 | 0.096 | 0.990 | 3645.6 | 55.0 | `641dfab10d7e` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt 641dfab10d7ec3ac: verdict dead, 6/15 useful-or-partial (parent 12/15). Wave 13 M5 solo: the Procrustes head merge alone at L12, g_q=3/g_kv=2, 0.239583x of the four attention projections, net 0.989586x.  One attention block merged costs 0.429852 composed drift -- more than MXO3s two whole SwiGLU lanes at 0.461810 buy for 2.5x the storage.  Charged lane: planning-dependencies loops to the 1024-token cap while the same-process parent completes the same prompt in 548 tokens.  creative-continuity repeats but completes and is graded a miss without charging the clause.

### `COMP.MA.MXQ2` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `19458703` | sm86 | fail | composition | -- | 0.1948 | 0.3994 | 0.080 | 0.010 | 0.990 | 1274.1 | 54.1 | `0e93238242f1` |
| `19458703` | sm86 | fail | composition | -- | 0.1948 | 0.3994 | 0.080 | 0.010 | 0.990 | 1274.1 | 54.1 | `935e6d423459` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt 935e6d423459e088: verdict dead, 6/15 useful-or-partial (parent 12/15). Wave 13 M5 solo at L02, the CHEAPEST head-merge site by drift (marginal +0.110941 on the MXO3 base against L12s +0.157983).  Net 0.989586x, composed drift 0.399369 -- LOWER than MXQ1s 0.429852 at L12 -- and yet FOUR lanes loop to the 1024-token cap against MXQ1s one, with the same-process parent completing all four.  Same score, worse text, lower drift: the drift law ranks these two backwards.

### `COMP.MA.MXQ3` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `84af71d8` | sm86 | fail | composition | -- | 0.2456 | 0.4684 | 0.135 | 0.086 | 0.990 | 9198.9 | 64.8 | `7c3ac76e6037` |
| `84af71d8` | sm86 | fail | composition | -- | 0.2456 | 0.4684 | 0.135 | 0.086 | 0.990 | 9198.9 | 64.8 | `9281f0d8e18d` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt 9281f0d8e18df219: verdict dead, 4/15 useful-or-partial (parent 12/15). Wave 13 M5 solo at L08.  Net 0.989586x, composed drift 0.468446 -- the highest of the three solo sites and past MXO3s 0.461810, from ONE merged attention block that removes 1.04 percent of the parent.  Six of sixteen lanes leave nothing when their repetitions are removed.

### `COMP.MA.MXQ4` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bcdcd3f7` | sm86 | fail | local-solve | 0.8275 | 0.1948 | 0.6794 | 0.027 | 0.006 | 0.683 | 19594.7 | 1424.7 | `d8dbd27ab857` |
| `bcdcd3f7` | sm86 | fail | local-solve | 0.8275 | 0.1948 | 0.6794 | 0.027 | 0.006 | 0.683 | 19594.7 | 1424.7 | `0a7fb8e04e77` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt 0a7fb8e04e7728f3: verdict dead, 0/15 useful-or-partial (parent 12/15). Wave 13: COMP.MA.MXP2s exact fold set plus M5 head merges at L02 and L12, the two cheapest merge sites.  Net 0.682581x -- DEEPER than MXP2s 0.703409x -- and 0/15.  Five lanes run to the 1024-token cap on a single repeated phrase and nine of the fifteen graded lanes leave nothing when their repetitions are removed.  The two head-merge lanes account for 0.399369 (L02, installed first) and +0.111632 (L12, on top of six SwiGLU lanes) of the childs 0.679394 terminal drift.  Head merge does not reopen the road to 0.5x: it buys 4.8M parameters and spends the whole coherence budget doing it.

### `COMP.MA.MXQ6` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `48acaece` | sm86 | fail | local-solve | 0.7694 | 0.3253 | 0.8218 | 0.041 | 0.008 | 0.498 | 31218.4 | 1918.7 | `18e203f7d104` |
| `48acaece` | sm86 | fail | local-solve | 0.7694 | 0.3253 | 0.8218 | 0.041 | 0.008 | 0.498 | 31218.4 | 1918.7 | `d95eed70a0ff` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt d95eed70a0ff1e79: verdict dead, 0/15 useful-or-partial (parent 12/15). Wave 13: ELEVEN SwiGLU lanes plus the token interface, at 0.497777x -- the same DEPTH as COMP.MA.MXO2 (0.497739x, 0/15) with the lane ORDER shopped instead of taken by depth.  Every lane after MXP2s three was priced on MXP2s own base and ranked by drift per million removed.  Result: drift 0.821803 against MXO2s 0.814086 -- shopping did NOT help at this depth -- and 0/15, with ten of sixteen lanes at the 1024-token cap and one emitting zero tokens.  Fourteen of the fifteen graded lanes leave nothing when their repetitions are removed; the fifteenth emits nothing at all.  The 0.5x storage target is reachable by folding and the artifact at it is not a model, whichever order the lanes are bought in.

### `COMP.MA.MXQ5` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `96309d73` | sm86 | fail | local-solve | 0.8275 | 0.1919 | 0.4879 | 0.045 | 0.025 | 0.710 | 19594.7 | 2419.0 | `7662a14ea43f` |
| `96309d73` | sm86 | fail | local-solve | 0.8275 | 0.1919 | 0.4879 | 0.045 | 0.025 | 0.710 | 19594.7 | 2419.0 | `15de0383b6c8` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt 15de0383b6c8de5f: verdict dead, 7/15 useful-or-partial (parent 12/15). Wave 13: wave 12s central bet, built.  ADDENDUM 10 corrective routes at k=256 on L05/L06/L13, on top of the SHOPPED child COMP.MA.MXP2 rather than the autopilot child COMP.MA.MXO4.  Review 6/15 -> 7/15 for 1,575,939 added parameters (0.69 percent of the parent), charged in full, so the rung is compared at 0.710270x and not at MXP2s 0.703409x.  Drift 0.489603 -> 0.487874 by way of 0.474727 (L05 route, -0.014876) and 0.472629 (L06, -0.002098) and then BACK UP to 0.487874 (L13, +0.015245) -- the third route made drift worse at a block that IS folded, which wave 12 only saw at a block that was not.  One lane at the token cap against MXP2s profile.

### `COMP.MA.MXQ7` -- FAIL

composed_child / heterogeneous_composition / level NA / site GLOBAL / wave 13

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `eb1f0680` | sm86 | fail | local-solve | 0.7694 | 0.3253 | 0.8219 | 0.010 | 0.002 | 0.505 | 31218.4 | 2668.9 | `509f7a3ec364` |
| `eb1f0680` | sm86 | fail | local-solve | 0.7694 | 0.3253 | 0.8219 | 0.010 | 0.002 | 0.505 | 31218.4 | 2668.9 | `aef14c8e40ba` |

> ADDENDUM 6 behaviour review by claude-opus5-mxq of receipt aef14c8e40bae5ca: verdict dead, 0/15 useful-or-partial (parent 12/15). Wave 13: the eleven-lane shopped set PLUS three ADDENDUM 10 corrective routes at k=256 (L10/L12/L13).  Net 0.504638x, drift 0.821895, 0/15.  Repair has NO purchase at this depth: L10 bought -0.020214, L12 gave back +0.017358 and L13 +0.002948, for a net +0.000092 -- the deep child is 0.0001 WORSE than the unrepaired COMP.MA.MXQ6 for 1,575,939 added parameters.  Thirteen of sixteen lanes run to the 1024-token cap.  This is the same locality result wave 12 measured at one unfolded block, now measured over a whole deep stack: a corrective route only cancels error that is still linear in its own blocks input.

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | arch | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | stored/dense | cond | s | receipt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | -- | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | -- | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | -- | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

