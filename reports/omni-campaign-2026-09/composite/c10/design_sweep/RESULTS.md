# C10 D4 design sweep: results

Eval receipts: `virtual_check_p1.json` `144c11323a43bfd3...`, `virtual_check_p2.json` `9f8845fc954b9b4b...`, `virtual_check_p3.json` `7b71660a0d54c1b6...`, `virtual_check_p4.json` `9c70f8dbfb5e07c2...`; DESIGN.md sha `5715f05b4b37494e...` (selection rule registered before the run).

References: S3 trunk held NLL 3.756, trunk vs parent top-1 held 0.630 / R-Q 0.514; parent held NLL 4.290.

| variant | spec | w (mean) | K3 target energy | K3 cols write | R-Q top1 vs trunk | R-Q top1 vs parent | held NLL | dNLL vs trunk | held top1 vs trunk | held top1 vs parent | KL trunk (held) | max norm ratio (layer) | veto layers | coherent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | parent/additive:1 | 1.000 | 0.762 | 0.731 | 0.000 (0/210) | 0.000 | 12.339 | 8.584 | 0.001 | 0.001 | 11.255 | 27.281 (L7) | 64/64 | False |
| a0 | trunk/additive:0 | 0.000 | 0.000 | 0.000 | 0.967 (203/210) | 0.510 | 3.751 | -0.004 | 0.977 | 0.632 | 0.023 | 1.001 (L63) | 64/64 | True |
| a003 | trunk/additive:0.03 | 0.030 | 0.124 | 0.118 | 0.857 (180/210) | 0.524 | 3.414 | -0.341 | 0.732 | 0.543 | 1.111 | 1.166 (L10) | 7/64 | True |
| a01 | trunk/additive:0.1 | 0.100 | 0.294 | 0.282 | 0.643 (135/210) | 0.490 | 2.371 | -1.384 | 0.585 | 0.473 | 2.685 | 1.816 (L9) | 8/64 | False |
| a03 | trunk/additive:0.3 | 0.300 | 0.530 | 0.507 | 0.067 (14/210) | 0.081 | 7.835 | 4.079 | 0.088 | 0.069 | 7.500 | 6.533 (L20) | 26/64 | False |
| b01 | trunk/convex:0.1 | 0.100 | 0.314 | 0.301 | 0.610 (128/210) | 0.467 | 2.435 | -1.321 | 0.578 | 0.467 | 2.749 | 1.763 (L9) | 8/64 | False |
| b025 | trunk/convex:0.25 | 0.250 | 0.555 | 0.530 | 0.062 (13/210) | 0.062 | 7.332 | 3.571 | 0.100 | 0.081 | 7.229 | 4.765 (L20) | 28/64 | False |
| b05 | trunk/convex:0.5 | 0.500 | 0.800 | 0.754 | 0.048 (10/210) | 0.038 | 10.427 | 6.667 | 0.015 | 0.011 | 9.446 | 12.052 (L7) | 59/64 | False |
| g01 | trunk/rms:0.1 | 0.081 | 0.099 | 0.093 | 0.890 (187/210) | 0.524 | 3.660 | -0.101 | 0.839 | 0.617 | 0.444 | 1.061 (L41) | 8/64 | True |
| g03 | trunk/rms:0.3 | 0.242 | 0.286 | 0.267 | 0.690 (145/210) | 0.490 | 3.156 | -0.605 | 0.673 | 0.545 | 1.586 | 1.218 (L41) | 17/64 | False |

Coherent under the registered rule: a0, a003, g01. Pick: **a003**.

dNLL is against the trunk run in the same eval part (same card): the trunk reads 3.756 on the RTX 8000 parts and 3.760 on the 3090 parts.
CAVEAT: the c17 charts were fitted on trunk taps of THESE six held rows (c17 trunk_taps s3native-c2k4352-held-all, 24,576 rows/layer), so the K3 term is in-sample on the held rows and the held-NLL gains of K3-carrying variants are not out-of-sample evidence; the R-Q prompts are the clean measure. The parent (S0 import, bf16) reads a HIGHER held NLL than the trunk (4.29 vs 3.76) while its R-Q NLL is lower (2.21 vs 3.55): unexplained.

All numbers are diagnostics. The sweep is open loop (each layer fitted on the S3 trunk's rows); the evaluation runs each variant as a real virtual child (closed forward).
