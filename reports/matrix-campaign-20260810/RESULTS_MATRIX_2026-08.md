# RESULTS MATRIX — Neural Forge fold-only campaign, through wave 42

> **WAVE 42 UPDATE, PREPENDED.  The body below is the wave-40 generation and is
> left as one; waves 41 and 42 added thirteen children and this file has not
> been regenerated from `campaign-frontier-table.json` since wave 40.  Where the
> two disagree, `ledger.jsonl` (1040 rows,
> `803439d1d300622dea437d64aaea07de88c16af04e3a808c5e6f2155ce5c58c4`) and
> `DEEP_RECORD` §0 are authoritative.**
>
> **The 1/15 and 0/15 rows of §1's ladder have both moved, and §1's table is
> corrected in place below.** Everything else in the ladder is unchanged.
>
> | at least | was (wave 40) | is (wave 42) |
> |---:|---|---|
> | 1/15 | `ARCH.V9CL896X13` 0.407083x | **`ARCH.V6CL768` 0.399687x — the first child below 0.40x** |
> | 0/15 | `ARCH.V10CL768X09` 0.389393x | **`ARCH.V6CL896X13` 0.393388x is deeper but the 0/15 row wants the DEEPEST child, which is still `ARCH.V10CL768X09` at 0.389393x** |
>
> **THE 1/15 ROW IS ONE READING OF TWO.** Under the canonical copy-back clause
> as written, `ARCH.V6CL768`'s one held case is voided and so is every held case
> down to 0.4116484014x; the 1/15 row on that reading is
> `ARCH.V10CL896X09` at 0.411648x. `DEEP_RECORD` §0 carries both columns for all
> 52 children at or below 0.47x. A ruling picks one; neither costs a re-grade.


**Generated** 2026-08-29 from `campaign-frontier-table.json`, which is itself a join of the sealed `ledger.jsonl`, the per-wave child indexes and the sealed `lm-eval` output. Nothing in this file is recomputed from prose.

**Parent** LFM2.5-230M, 229,693,184 parameters, **12/15** on the frozen protocol. **337 whole-model children reviewed**, 274 holding at least one case, 247 benchmark rows, 1014 ledger rows.

**How to read the two review columns.** `sealed 15` is the frozen 15-prompt protocol, read by hand, and it is the **only** column any frontier claim, verdict or ledger row fires on — before wave 29 and after it. `ext 30` is ADDENDUM 16.1's extension surface, authored blind in wave 29 and machine-scored against a rubric written before any child was decoded. It prices and orders; it never grades. A row with a blank `ext 30` was not run on the second surface, which is a statement about GPU budget and not about the child.

**Storage column.** `S` is standard-shape — a directory stock `transformers` loads, R3-exportable. `P` is parent-shaped — a sealed composed tensor set at a fully-counted stored ratio. ADDENDUM 16.3 permits a parent-shaped child to headline the parent-band column provided its format is named, which is what the first column does on every row.

## 1. The frontier ladder — deepest child holding at least N of 15

| at least | child | net ratio | read | wave | storage |
|---:|---|---:|---:|---:|:--:|
| 14/15 | `ARCH.V19` | 0.794570x | 14/15 | 27 | S |
| 13/15 | `ARCH.V18` | 0.790005x | 13/15 | 26 | S |
| 12/15 | `ARCH.V18` | 0.790005x | 13/15 | 26 | S |
| 11/15 | `ARCH.B75` | 0.742049x | 11/15 | 29 | S |
| 10/15 | `ARCH.V18X09` | 0.737484x | 10/15 | 27 | S |
| 9/15 | `ARCH.V18X07` | 0.737484x | 9/15 | 27 | S |
| 8/15 | `ARCH.GV16L01` | 0.637073x | 8/15 | 33 | S |
| 7/15 | `ARCH.GV12L03` | 0.618813x | 7/15 | 33 | S |
| 6/15 | `ARCH.V9CL1792` | 0.605117x | 6/15 | 31 | S |
| 5/15 | `ARCH.V8CL1792` | 0.600552x | 5/15 | 31 | S |
| 4/15 | `ARCH.V9CL896G10` | 0.437349x | 4/15 | 36 | S |
| 3/15 | `ARCH.V9CL896G03` | 0.437349x | 3/15 | 36 | S |
| 2/15 | `ARCH.V10CL896X09G10` | 0.411648x | 2/15 | 38 | S |
| 1/15 | **`ARCH.V6CL768`** | **0.399687x** | 1/15 | **42** | S |
| 0/15 | `ARCH.V10CL768X09` | 0.389393x | 0/15 | 40 | S |

## 2. Every reviewed whole-model child, deepest last

| | child | net ratio | sealed 15 | ext 30 | charged | verdict | mechanism | benchmarks |
|:--:|---|---:|---:|---:|---:|---|---|---|
| S | `ARCH.V10CL768X09` | 0.389393x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.V12CL896X09X13` | 0.390512x | **0**/15 | -- | 16 | dead | -- | arc 0.310 · hella 0.310 · piqa 0.460 · wino 0.470 |
| S | `ARCH.V12CL768X09` | 0.398524x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.T9CL896X09` | 0.407083x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.V9CL896X09` | 0.407083x | **0**/15 | -- | 16 | dead | -- | arc 0.275 · hella 0.285 · piqa 0.495 · wino 0.480 |
| S | `ARCH.V9CL896X13` | 0.407083x | **1**/15 | -- | 14 | dead | -- | -- |
| S | `ARCH.V10CL896X09` | 0.411648x | **1**/15 | -- | 16 | dead | -- | arc 0.265 · hella 0.305 · piqa 0.505 · wino 0.475 |
| S | `ARCH.V10CL896X09G10` | 0.411648x | **2**/15 | -- | 16 | dead | -- | arc 0.280 · hella 0.300 · piqa 0.510 · wino 0.470 |
| S | `ARCH.V10CL896X13` | 0.411648x | **1**/15 | -- | 16 | dead | -- | arc 0.290 · hella 0.330 · piqa 0.500 · wino 0.485 |
| S | `ARCH.V9CL768` | 0.413383x | **1**/15 | -- | 14 | dead | -- | -- |
| S | `ARCH.V10CL768` | 0.417948x | **1**/15 | -- | 14 | dead | -- | -- |
| S | `ARCH.V12CL896X09` | 0.420779x | **2**/15 | -- | 16 | dead | -- | arc 0.305 · hella 0.325 · piqa 0.490 · wino 0.460 |
| S | `ARCH.V12CL896X09G03` | 0.420779x | **1**/15 | -- | 16 | dead | -- | arc 0.320 · hella 0.325 · piqa 0.490 · wino 0.465 |
| S | `ARCH.V12CL896X09G10` | 0.420779x | **2**/15 | -- | 16 | dead | -- | arc 0.330 · hella 0.325 · piqa 0.500 · wino 0.450 |
| S | `ARCH.V12CL896X13` | 0.420779x | **1**/15 | -- | 16 | dead | -- | arc 0.355 · hella 0.325 · piqa 0.530 · wino 0.500 |
| S | `ARCH.V12CL896X13G03` | 0.420779x | **1**/15 | -- | 16 | dead | -- | arc 0.350 · hella 0.325 · piqa 0.515 · wino 0.470 |
| S | `ARCH.V12CL768` | 0.427078x | **1**/15 | -- | 14 | dead | -- | -- |
| S | `ARCH.V14CL896X09` | 0.429909x | **2**/15 | -- | 16 | dead | -- | arc 0.305 · hella 0.315 · piqa 0.495 · wino 0.470 |
| S | `ARCH.V14CL896X13` | 0.429909x | **1**/15 | -- | 16 | dead | -- | arc 0.325 · hella 0.315 · piqa 0.545 · wino 0.485 |
| S | `ARCH.V8CL896` | 0.432784x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V9CL896` | 0.437349x | **2**/15 | -- | 16 | dead | -- | arc 0.270 · hella 0.345 · piqa 0.510 · wino 0.475 |
| S | `ARCH.V9CL896G03` | 0.437349x | **3**/15 | -- | 16 | dead | -- | arc 0.270 · hella 0.345 · piqa 0.510 · wino 0.485 |
| S | `ARCH.V9CL896G10` | 0.437349x | **4**/15 | -- | 16 | dead | -- | arc 0.270 · hella 0.350 · piqa 0.510 · wino 0.475 |
| S | `ARCH.V10CL896` | 0.441915x | **2**/15 | -- | 13 | dead | -- | arc 0.295 · hella 0.350 · piqa 0.510 · wino 0.475 |
| S | `ARCH.V10CL896G01` | 0.441915x | **2**/15 | -- | 16 | dead | -- | arc 0.275 · hella 0.340 · piqa 0.500 · wino 0.470 |
| S | `ARCH.V10CL896G03` | 0.441915x | **3**/15 | -- | 16 | dead | -- | arc 0.285 · hella 0.350 · piqa 0.510 · wino 0.475 |
| S | `ARCH.V10CL896G10` | 0.441915x | **3**/15 | -- | 16 | dead | -- | arc 0.290 · hella 0.345 · piqa 0.510 · wino 0.465 |
| S | `ARCH.V12CL896` | 0.451045x | **2**/15 | -- | 11 | dead | -- | arc 0.330 · hella 0.345 · piqa 0.500 · wino 0.485 |
| S | `ARCH.V12CL896G01` | 0.451045x | **2**/15 | -- | 11 | dead | -- | arc 0.315 · hella 0.340 · piqa 0.490 · wino 0.495 |
| S | `ARCH.V12CL896G03` | 0.451045x | **4**/15 | -- | 9 | dead | -- | arc 0.330 · hella 0.335 · piqa 0.495 · wino 0.490 |
| S | `ARCH.V12CL896G10` | 0.451045x | **2**/15 | -- | 10 | dead | -- | arc 0.320 · hella 0.335 · piqa 0.495 · wino 0.490 |
| S | `ARCH.V19X13CL896` | 0.452734x | **0**/15 | -- | 16 | dead | -- | arc 0.310 · hella 0.285 · piqa 0.555 · wino 0.500 |
| S | `ARCH.V14CL896` | 0.460175x | **2**/15 | -- | 16 | dead | -- | arc 0.325 · hella 0.360 · piqa 0.485 · wino 0.475 |
| S | `ARCH.V14CL896G03` | 0.460175x | **3**/15 | -- | 16 | dead | -- | arc 0.330 · hella 0.345 · piqa 0.490 · wino 0.470 |
| S | `ARCH.V14CL896G10` | 0.460175x | **2**/15 | -- | 16 | dead | -- | arc 0.320 · hella 0.350 · piqa 0.490 · wino 0.475 |
| S | `ARCH.W512` | 0.463481x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.V16CL896` | 0.469305x | **1**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V16CL896G03` | 0.469305x | **3**/15 | -- | 8 | dead | -- | arc 0.335 · hella 0.330 · piqa 0.490 · wino 0.470 |
| S | `ARCH.V16CL896G10` | 0.469305x | **2**/15 | -- | 16 | dead | -- | arc 0.335 · hella 0.330 · piqa 0.495 · wino 0.470 |
| S | `ARCH.V18CL896` | 0.478436x | **2**/15 | -- | 14 | dead | -- | arc 0.330 · hella 0.320 · piqa 0.515 · wino 0.480 |
| S | `ARCH.KKTD` | 0.483001x | **0**/15 | -- | 4 | dead | -- | arc 0.305 · hella 0.325 · piqa 0.550 · wino 0.510 |
| S | `ARCH.KKTS` | 0.483001x | **0**/15 | -- | 10 | dead | -- | arc 0.325 · hella 0.315 · piqa 0.520 · wino 0.475 |
| S | `ARCH.V19CL896` | 0.483001x | **2**/15 | -- | 14 | dead | -- | arc 0.335 · hella 0.320 · piqa 0.515 · wino 0.485 |
| S | `ARCH.V19CL896G03` | 0.483001x | **3**/15 | -- | 16 | dead | -- | arc 0.345 · hella 0.315 · piqa 0.510 · wino 0.490 |
| S | `ARCH.V20CL896` | 0.487566x | **2**/15 | -- | 11 | dead | -- | arc 0.325 · hella 0.345 · piqa 0.515 · wino 0.480 |
| S | `ARCH.V20CL896G03` | 0.487566x | **3**/15 | -- | 14 | dead | -- | arc 0.330 · hella 0.340 · piqa 0.510 · wino 0.475 |
| P | `COMP.MA.MXO2` | 0.497739x | **0**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXQ6` | 0.497777x | **0**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXQ7` | 0.504638x | **0**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.V12W896CL1792` | 0.525484x | **0**/15 | -- | 12 | dead | -- | arc 0.285 · hella 0.340 · piqa 0.470 · wino 0.445 |
| S | `ARCH.V16W896CL1792` | 0.541462x | **0**/15 | -- | 12 | dead | -- | arc 0.315 · hella 0.360 · piqa 0.485 · wino 0.455 |
| S | `ARCH.V32CL896` | 0.542347x | **2**/15 | -- | 9 | dead | -- | arc 0.370 · hella 0.405 · piqa 0.490 · wino 0.495 |
| S | `ARCH.A552` | 0.552023x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V19CL1280` | 0.554901x | **1**/15 | -- | 6 | dead | -- | -- |
| S | `ARCH.V8CL1792X13` | 0.558303x | **1**/15 | -- | 10 | dead | -- | arc 0.315 · hella 0.300 · piqa 0.495 · wino 0.520 |
| S | `ARCH.V12CL1792X13` | 0.576563x | **2**/15 | -- | 10 | dead | -- | arc 0.330 · hella 0.370 · piqa 0.530 · wino 0.525 |
| S | `ARCH.V19CL1408` | 0.578868x | **1**/15 | -- | 6 | dead | -- | arc 0.345 · hella 0.385 · piqa 0.565 · wino 0.515 |
| S | `ARCH.R5` | 0.579946x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.W14FF320` | 0.580580x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V4CL1792` | 0.582292x | **1**/15 | -- | 11 | dead | -- | arc 0.325 · hella 0.340 · piqa 0.520 · wino 0.485 |
| S | `ARCH.W640` | 0.590763x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.V6CL1792` | 0.591422x | **3**/15 | -- | 5 | dead | -- | arc 0.305 · hella 0.315 · piqa 0.495 · wino 0.530 |
| S | `ARCH.V8CL1792` | 0.600552x | **5**/15 | -- | 2 | dead | -- | arc 0.305 · hella 0.350 · piqa 0.495 · wino 0.575 |
| S | `ARCH.V8CL1792M` | 0.600552x | **3**/15 | -- | 3 | dead | -- | arc 0.310 · hella 0.355 · piqa 0.515 · wino 0.505 |
| S | `ARCH.V19CL1536` | 0.602835x | **2**/15 | -- | 4 | dead | -- | arc 0.310 · hella 0.360 · piqa 0.545 · wino 0.565 |
| S | `ARCH.R4` | 0.603922x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V9CL1792` | 0.605117x | **6**/15 | -- | 3 | dead | -- | arc 0.280 · hella 0.365 · piqa 0.485 · wino 0.555 |
| S | `ARCH.V9CL1792M` | 0.605117x | **2**/15 | -- | 5 | dead | -- | arc 0.290 · hella 0.365 · piqa 0.505 · wino 0.545 |
| S | `ARCH.V19CL1792X13` | 0.608519x | **2**/15 | -- | 4 | dead | -- | arc 0.365 · hella 0.345 · piqa 0.585 · wino 0.520 |
| S | `ARCH.GV12L01` | 0.618813x | **6**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.GV12L03` | 0.618813x | **7**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.GV12L10` | 0.618813x | **7**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.V12CL1792` | 0.618813x | **7**/15 | -- | 3 | dead | -- | arc 0.370 · hella 0.385 · piqa 0.530 · wino 0.535 |
| S | `ARCH.V12CL1792M` | 0.618813x | **4**/15 | -- | 4 | dead | -- | arc 0.365 · hella 0.360 · piqa 0.505 · wino 0.575 |
| S | `ARCH.V16W896X13` | 0.623329x | **1**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V19CL1664` | 0.626802x | **3**/15 | -- | 4 | dead | -- | arc 0.345 · hella 0.345 · piqa 0.580 · wino 0.595 |
| S | `ARCH.CL896X09X13` | 0.627898x | **0**/15 | -- | 16 | dead | -- | arc 0.355 · hella 0.350 · piqa 0.520 · wino 0.500 |
| S | `ARCH.CL768X09` | 0.635910x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.GV16L01` | 0.637073x | **8**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.GV16L03` | 0.637073x | **7**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.GV16L10` | 0.637073x | **7**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.V16BMV3` | 0.637073x | **3**/15 | -- | 3 | dead | -- | arc 0.355 · hella 0.435 · piqa 0.505 · wino 0.565 |
| S | `ARCH.V16CL1792` | 0.637073x | **8**/15 | -- | 3 | dead | -- | arc 0.400 · hella 0.405 · piqa 0.545 · wino 0.520 |
| S | `ARCH.V16CL1792M` | 0.637073x | **3**/15 | -- | 4 | dead | -- | arc 0.360 · hella 0.390 · piqa 0.515 · wino 0.565 |
| S | `ARCH.V19CL1728` | 0.638785x | **4**/15 | -- | 9 | dead | -- | -- |
| S | `ARCH.V20X13W896` | 0.639307x | **2**/15 | -- | 14 | dead | -- | arc 0.310 · hella 0.415 · piqa 0.585 · wino 0.525 |
| S | `ARCH.CL640` | 0.640497x | **0**/15 | -- | 4 | dead | -- | -- |
| S | `ARCH.CT640` | 0.640497x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.S640` | 0.640497x | **0**/15 | -- | 4 | dead | -- | -- |
| S | `ARCH.W14FF640` | 0.640497x | **0**/15 | -- | 6 | dead | -- | -- |
| S | `ARCH.R3` | 0.641584x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V18CL1792` | 0.646204x | **7**/15 | -- | 3 | dead | -- | arc 0.390 · hella 0.400 · piqa 0.560 · wino 0.540 |
| S | `ARCH.V19CL1792` | 0.650769x | **7**/15 | -- | 3 | dead | -- | arc 0.390 · hella 0.400 · piqa 0.580 · wino 0.545 |
| S | `ARCH.V19CL1792M` | 0.650769x | **6**/15 | -- | 3 | dead | -- | arc 0.365 · hella 0.385 · piqa 0.540 · wino 0.550 |
| S | `ARCH.V19CL1792ME` | 0.650769x | **3**/15 | -- | 3 | dead | -- | arc 0.400 · hella 0.340 · piqa 0.590 · wino 0.560 |
| S | `ARCH.V19CL1792MV` | 0.650769x | **5**/15 | -- | 2 | dead | -- | arc 0.395 · hella 0.380 · piqa 0.550 · wino 0.545 |
| S | `ARCH.T010713W1280C` | 0.654126x | **0**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.W704` | 0.656116x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.W704AW25` | 0.656116x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.CL896X09` | 0.658165x | **1**/15 | -- | 16 | dead | -- | arc 0.365 · hella 0.365 · piqa 0.535 · wino 0.510 |
| S | `ARCH.CL896X13` | 0.658165x | **1**/15 | -- | 16 | dead | -- | arc 0.335 · hella 0.370 · piqa 0.555 · wino 0.480 |
| S | `ARCH.R2` | 0.658704x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V18X13X01KV6JQ` | 0.664419x | **0**/15 | -- | 7 | dead | -- | arc 0.345 · hella 0.320 · piqa 0.525 · wino 0.500 |
| S | `ARCH.CL768` | 0.664464x | **1**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.V16W896` | 0.667288x | **2**/15 | -- | 12 | dead | -- | arc 0.310 · hella 0.410 · piqa 0.505 · wino 0.525 |
| S | `ARCH.W720` | 0.672632x | **0**/15 | -- | 16 | dead | -- | arc 0.260 · hella 0.405 · piqa 0.505 · wino 0.485 |
| S | `ARCH.V24CL1792` | 0.673594x | **7**/15 | -- | 3 | dead | -- | arc 0.395 · hella 0.430 · piqa 0.570 · wino 0.590 |
| S | `ARCH.V18W896` | 0.675277x | **2**/15 | -- | 13 | dead | -- | arc 0.300 · hella 0.405 · piqa 0.540 · wino 0.515 |
| P | `COMP.MA.MXQ4` | 0.682581x | **0**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.V26CL1792` | 0.682724x | **7**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.410 · piqa 0.550 · wino 0.595 |
| S | `ARCH.V20W896` | 0.683266x | **2**/15 | -- | 13 | dead | -- | arc 0.295 · hella 0.415 · piqa 0.540 · wino 0.505 |
| S | `ARCH.V18X13X01` | 0.684962x | **3**/15 | -- | 9 | dead | -- | arc 0.335 · hella 0.370 · piqa 0.540 · wino 0.515 |
| S | `ARCH.V18X13X01G01` | 0.684962x | **3**/15 | -- | 11 | dead | -- | arc 0.320 · hella 0.350 · piqa 0.545 · wino 0.510 |
| S | `ARCH.V18X13X01G03` | 0.684962x | **4**/15 | -- | 10 | dead | -- | arc 0.325 · hella 0.360 · piqa 0.535 · wino 0.505 |
| S | `ARCH.V18X13X01G10` | 0.684962x | **4**/15 | -- | 7 | dead | -- | arc 0.335 · hella 0.365 · piqa 0.525 · wino 0.510 |
| S | `ARCH.CL896` | 0.688431x | **4**/15 | -- | 3 | dead | -- | -- |
| S | `ARCH.W736` | 0.689220x | **1**/15 | -- | 15 | dead | -- | arc 0.285 · hella 0.405 · piqa 0.515 · wino 0.480 |
| S | `ARCH.W736LS` | 0.689220x | **0**/15 | -- | 16 | dead | -- | arc 0.305 · hella 0.400 · piqa 0.505 · wino 0.465 |
| S | `ARCH.R1` | 0.689528x | **0**/15 | -- | 10 | dead | -- | -- |
| S | `ARCH.V19X13X01` | 0.689528x | **3**/15 | -- | 11 | dead | -- | arc 0.345 · hella 0.380 · piqa 0.540 · wino 0.535 |
| S | `ARCH.V19X13X01G03` | 0.689528x | **4**/15 | -- | 10 | dead | -- | arc 0.325 · hella 0.375 · piqa 0.530 · wino 0.505 |
| S | `ARCH.V19X13X09` | 0.689528x | **5**/15 | -- | 10 | dead | -- | arc 0.375 · hella 0.360 · piqa 0.600 · wino 0.515 |
| S | `ARCH.T010713W1536C` | 0.691788x | **0**/15 | -- | 5 | dead | -- | -- |
| S | `ARCH.V8X13` | 0.691832x | **6**/15 | -- | 6 | dead | -- | arc 0.340 · hella 0.370 · piqa 0.540 · wino 0.455 |
| S | `ARCH.V8X13G` | 0.691832x | **4**/15 | -- | 7 | dead | -- | arc 0.310 · hella 0.380 · piqa 0.550 · wino 0.460 |
| S | `ARCH.V8X13GL01` | 0.691832x | **7**/15 | -- | 6 | dead | -- | arc 0.310 · hella 0.375 · piqa 0.550 · wino 0.455 |
| S | `ARCH.V8X13GL03` | 0.691832x | **7**/15 | -- | 7 | dead | -- | arc 0.310 · hella 0.385 · piqa 0.545 · wino 0.460 |
| S | `ARCH.V8X13GL1` | 0.691832x | **6**/15 | -- | 7 | dead | -- | arc 0.320 · hella 0.390 · piqa 0.530 · wino 0.450 |
| S | `ARCH.V20X13X01` | 0.694093x | **3**/15 | -- | 8 | dead | -- | arc 0.350 · hella 0.375 · piqa 0.540 · wino 0.515 |
| S | `ARCH.V20X13X01G03` | 0.694093x | **1**/15 | -- | 10 | dead | -- | arc 0.330 · hella 0.370 · piqa 0.545 · wino 0.490 |
| S | `ARCH.V20X13X07` | 0.694093x | **5**/15 | -- | 10 | dead | -- | arc 0.365 · hella 0.395 · piqa 0.620 · wino 0.475 |
| S | `ARCH.V9X13` | 0.696398x | **6**/15 | -- | 5 | dead | -- | arc 0.330 · hella 0.385 · piqa 0.565 · wino 0.475 |
| P | `COMP.MA.MXO4` | 0.703396x | **5**/15 | -- | -- | dead | -- | arc 0.420 · hella 0.435 · piqa 0.560 · wino 0.525 |
| P | `COMP.MA.MXP2` | 0.703409x | **6**/15 | -- | -- | dead | -- | arc 0.425 · hella 0.425 · piqa 0.585 · wino 0.540 |
| S | `ARCH.V12X13` | 0.710093x | **7**/15 | -- | 7 | dead | -- | arc 0.385 · hella 0.390 · piqa 0.590 · wino 0.495 |
| P | `COMP.MA.MXQ5` | 0.710270x | **7**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXP1` | 0.712544x | **7**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.V18X13KV6JQ` | 0.716941x | **2**/15 | -- | 10 | dead | -- | arc 0.360 · hella 0.365 · piqa 0.585 · wino 0.505 |
| S | `ARCH.MM768` | 0.722610x | **0**/15 | -- | 16 | dead | -- | arc 0.290 · hella 0.385 · piqa 0.500 · wino 0.515 |
| S | `ARCH.W768` | 0.722610x | **3**/15 | -- | 9 | dead | -- | arc 0.310 · hella 0.430 · piqa 0.520 · wino 0.485 |
| S | `ARCH.W768AW100` | 0.722610x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.W768AW25` | 0.722610x | **0**/15 | -- | 12 | dead | -- | arc 0.320 · hella 0.435 · piqa 0.535 · wino 0.520 |
| S | `ARCH.W768EV` | 0.722610x | **1**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.W768LS` | 0.722610x | **3**/15 | -- | 10 | dead | -- | arc 0.305 · hella 0.430 · piqa 0.505 · wino 0.460 |
| S | `ARCH.W768RG` | 0.722610x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.W768ST` | 0.722610x | **2**/15 | -- | 10 | dead | -- | arc 0.290 · hella 0.435 · piqa 0.515 · wino 0.455 |
| S | `ARCH.BX13` | 0.724929x | **1**/15 | -- | 5 | dead | -- | -- |
| S | `ARCH.R0` | 0.724929x | **0**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.V20X13KV6JQ` | 0.726071x | **2**/15 | -- | 10 | dead | -- | arc 0.350 · hella 0.385 · piqa 0.590 · wino 0.520 |
| S | `ARCH.V20X13KV6JQG03` | 0.726071x | **2**/15 | -- | 11 | dead | -- | arc 0.340 · hella 0.390 · piqa 0.600 · wino 0.495 |
| S | `ARCH.V16X13` | 0.728353x | **5**/15 | -- | 6 | dead | -- | arc 0.405 · hella 0.395 · piqa 0.610 · wino 0.495 |
| S | `ARCH.V16X13G` | 0.728353x | **5**/15 | -- | 7 | dead | -- | arc 0.380 · hella 0.410 · piqa 0.615 · wino 0.495 |
| S | `ARCH.V16X13GL01` | 0.728353x | **6**/15 | -- | 6 | dead | -- | arc 0.380 · hella 0.405 · piqa 0.615 · wino 0.485 |
| S | `ARCH.V16X13GL03` | 0.728353x | **7**/15 | -- | 7 | dead | -- | arc 0.385 · hella 0.395 · piqa 0.615 · wino 0.495 |
| S | `ARCH.V16X13GL1` | 0.728353x | **6**/15 | -- | 6 | dead | -- | arc 0.385 · hella 0.400 · piqa 0.625 · wino 0.485 |
| P | `COMP.MA.MXO3` | 0.729103x | **8**/15 | -- | -- | dead | -- | arc 0.430 · hella 0.430 · piqa 0.590 · wino 0.545 |
| S | `ARCH.F1301W1536` | 0.730614x | **0**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.F1301W1536C` | 0.730614x | **0**/15 | -- | 1 | dead | -- | -- |
| S | `ARCH.V32W896` | 0.731199x | **3**/15 | -- | 12 | dead | -- | arc 0.370 · hella 0.415 · piqa 0.500 · wino 0.525 |
| S | `ARCH.W896CL1792` | 0.733197x | **0**/15 | -- | 12 | dead | -- | arc 0.315 · hella 0.390 · piqa 0.505 · wino 0.525 |
| S | `ARCH.V22X13KV6JQ` | 0.735201x | **2**/15 | -- | 10 | dead | -- | arc 0.375 · hella 0.395 · piqa 0.555 · wino 0.530 |
| P | `COMP.MA.MXN3` | 0.735962x | **6**/15 | -- | -- | dead | -- | arc 0.475 · hella 0.430 · piqa 0.580 · wino 0.565 |
| P | `COMP.MA.MXM9` | 0.736305x | **4**/15 | -- | -- | dead | -- | arc 0.370 · hella 0.315 · piqa 0.510 · wino 0.515 |
| S | `ARCH.V18X07` | 0.737484x | **9**/15 | -- | 4 | dead | -- | arc 0.385 · hella 0.400 · piqa 0.580 · wino 0.555 |
| S | `ARCH.V18X07G03` | 0.737484x | **8**/15 | -- | 4 | dead | -- | arc 0.390 · hella 0.400 · piqa 0.580 · wino 0.560 |
| S | `ARCH.V18X09` | 0.737484x | **10**/15 | -- | 5 | dead | -- | arc 0.435 · hella 0.390 · piqa 0.595 · wino 0.570 |
| S | `ARCH.V18X09G03` | 0.737484x | **9**/15 | -- | 4 | dead | -- | arc 0.430 · hella 0.395 · piqa 0.580 · wino 0.580 |
| S | `ARCH.V18X13` | 0.737484x | **10**/15 | -- | 6 | dead | -- | arc 0.410 · hella 0.385 · piqa 0.625 · wino 0.505 |
| S | `ARCH.V18X13G03` | 0.737484x | **10**/15 | -- | 5 | dead | -- | arc 0.400 · hella 0.380 · piqa 0.610 · wino 0.505 |
| S | `ARCH.B25` | 0.742049x | **10**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.395 · piqa 0.610 · wino 0.495 |
| S | `ARCH.B50` | 0.742049x | **10**/15 | -- | 2 | dead | -- | arc 0.420 · hella 0.400 · piqa 0.600 · wino 0.500 |
| S | `ARCH.B75` | 0.742049x | **11**/15 | -- | 2 | dead | -- | arc 0.405 · hella 0.400 · piqa 0.610 · wino 0.505 |
| S | `ARCH.BP64` | 0.742049x | **10**/15 | -- | 2 | dead | -- | arc 0.410 · hella 0.370 · piqa 0.610 · wino 0.490 |
| S | `ARCH.MASK03` | 0.742049x | **9**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.410 · piqa 0.605 · wino 0.510 |
| S | `ARCH.SR24` | 0.742049x | **10**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.390 · piqa 0.605 · wino 0.495 |
| S | `ARCH.SR64` | 0.742049x | **10**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.380 · piqa 0.620 · wino 0.510 |
| S | `ARCH.V19X09` | 0.742049x | **10**/15 | -- | 6 | dead | -- | arc 0.430 · hella 0.390 · piqa 0.595 · wino 0.560 |
| S | `ARCH.V19X13` | 0.742049x | **10**/15 | -- | 5 | dead | -- | arc 0.400 · hella 0.400 · piqa 0.620 · wino 0.500 |
| S | `ARCH.V19X13G01` | 0.742049x | **11**/15 | -- | 3 | dead | -- | arc 0.400 · hella 0.405 · piqa 0.595 · wino 0.485 |
| S | `ARCH.V19X13G03` | 0.742049x | **11**/15 | -- | 2 | dead | -- | arc 0.405 · hella 0.400 · piqa 0.600 · wino 0.500 |
| S | `ARCH.V19X13G10` | 0.742049x | **10**/15 | -- | 3 | dead | -- | arc 0.410 · hella 0.400 · piqa 0.610 · wino 0.500 |
| P | `COMP.MA.MXN1` | 0.744033x | **7**/15 | -- | -- | dead | -- | arc 0.360 · hella 0.390 · piqa 0.575 · wino 0.510 |
| P | `COMP.MA.MXM8` | 0.744297x | **3**/15 | -- | -- | dead | -- | arc 0.275 · hella 0.280 · piqa 0.515 · wino 0.475 |
| P | `COMP.MA.MXM4` | 0.744331x | **0**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.V8` | 0.744354x | **4**/15 | -- | 2 | dead | -- | arc 0.330 · hella 0.385 · piqa 0.540 · wino 0.510 |
| S | `ARCH.V20X07` | 0.746614x | **9**/15 | -- | 3 | dead | -- | arc 0.370 · hella 0.425 · piqa 0.570 · wino 0.550 |
| S | `ARCH.V20X09` | 0.746614x | **9**/15 | -- | 5 | dead | -- | arc 0.405 · hella 0.405 · piqa 0.580 · wino 0.595 |
| S | `ARCH.V20X09G03` | 0.746614x | **8**/15 | -- | 5 | dead | -- | arc 0.405 · hella 0.405 · piqa 0.565 · wino 0.600 |
| S | `ARCH.V20X13` | 0.746614x | **10**/15 | -- | 6 | dead | -- | arc 0.385 · hella 0.400 · piqa 0.605 · wino 0.515 |
| S | `ARCH.V20X13G03` | 0.746614x | **11**/15 | -- | 4 | dead | -- | arc 0.385 · hella 0.415 · piqa 0.595 · wino 0.525 |
| S | `ARCH.V9` | 0.748919x | **9**/15 | -- | 8 | dead | -- | arc 0.320 · hella 0.400 · piqa 0.560 · wino 0.505 |
| P | `COMP.MA.MXM10` | 0.753440x | **6**/15 | -- | -- | dead | -- | arc 0.395 · hella 0.320 · piqa 0.540 · wino 0.465 |
| S | `ARCH.V10` | 0.753484x | **10**/15 | -- | 5 | dead | -- | arc 0.360 · hella 0.390 · piqa 0.575 · wino 0.510 |
| P | `COMP.MA.MXN4` | 0.754811x | **8**/15 | -- | -- | dead | -- | arc 0.425 · hella 0.450 · piqa 0.600 · wino 0.525 |
| S | `ARCH.CL1280` | 0.760331x | **4**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.W14FF1280` | 0.760331x | **5**/15 | -- | 3 | dead | -- | -- |
| S | `ARCH.V12` | 0.762614x | **11**/15 | -- | 5 | dead | -- | arc 0.395 · hella 0.450 · piqa 0.580 · wino 0.525 |
| S | `ARCH.B1536` | 0.769439x | **0**/15 | -- | 9 | dead | -- | -- |
| S | `ARCH.W1536X13` | 0.769439x | **4**/15 | -- | 3 | dead | -- | -- |
| S | `ARCH.V14` | 0.771744x | **9**/15 | -- | 4 | dead | -- | arc 0.400 · hella 0.425 · piqa 0.580 · wino 0.540 |
| P | `COMP.MA.MXN5` | 0.780411x | **8**/15 | -- | -- | dead | -- | arc 0.435 · hella 0.470 · piqa 0.610 · wino 0.500 |
| P | `COMP.MA.MXN2` | 0.780518x | **10**/15 | -- | -- | alive-parent-band | -- | arc 0.455 · hella 0.445 · piqa 0.640 · wino 0.555 |
| P | `COMP.MA.MXM5` | 0.780727x | **6**/15 | -- | -- | dead | -- | arc 0.380 · hella 0.360 · piqa 0.540 · wino 0.550 |
| P | `COMP.MA.MXM3` | 0.780772x | **3**/15 | -- | -- | dead | -- | arc 0.385 · hella 0.440 · piqa 0.560 · wino 0.510 |
| P | `COMP.MA.MXM7` | 0.780830x | **5**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXM1` | 0.780857x | **0**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXM2` | 0.780857x | **0**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXM6` | 0.780861x | **9**/15 | -- | -- | dead | -- | arc 0.365 · hella 0.320 · piqa 0.520 · wino 0.535 |
| S | `ARCH.V16` | 0.780874x | **9**/15 | -- | 0 | alive-degraded | -- | arc 0.430 · hella 0.455 · piqa 0.605 · wino 0.520 |
| S | `ARCH.CL1408` | 0.784298x | **3**/15 | -- | 5 | dead | -- | arc 0.390 · hella 0.385 · piqa 0.585 · wino 0.515 |
| S | `ARCH.V17` | 0.785440x | **11**/15 | -- | 0 | alive-parent-band | -- | arc 0.405 · hella 0.425 · piqa 0.610 · wino 0.530 |
| S | `ARCH.Q01050913` | 0.789916x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.V18` | 0.790005x | **13**/15 | -- | 5 | dead | -- | arc 0.420 · hella 0.425 · piqa 0.625 · wino 0.530 |
| S | `ARCH.V19` | 0.794570x | **14**/15 | -- | 0 | alive-parent-band | -- | arc 0.415 · hella 0.430 · piqa 0.620 · wino 0.525 |
| S | `ARCH.V19MXO3` | 0.794570x | **3**/15 | -- | 2 | dead | -- | arc 0.380 · hella 0.395 · piqa 0.555 · wino 0.595 |
| S | `ARCH.V19MXP2` | 0.794570x | **2**/15 | -- | 3 | dead | -- | arc 0.335 · hella 0.420 · piqa 0.575 · wino 0.580 |
| S | `ARCH.V20` | 0.799135x | **13**/15 | -- | 2 | dead | -- | arc 0.430 · hella 0.465 · piqa 0.625 · wino 0.530 |
| S | `ARCH.V32X13` | 0.801395x | **10**/15 | -- | 6 | dead | -- | arc 0.465 · hella 0.400 · piqa 0.615 · wino 0.510 |
| S | `ARCH.V32X13G` | 0.801395x | **10**/15 | -- | 4 | dead | -- | arc 0.465 · hella 0.430 · piqa 0.615 · wino 0.530 |
| S | `ARCH.V32X13G03` | 0.801395x | **10**/15 | -- | 6 | dead | -- | arc 0.460 · hella 0.435 · piqa 0.605 · wino 0.530 |
| S | `ARCH.V32X13GL01` | 0.801395x | **9**/15 | -- | 5 | dead | -- | arc 0.465 · hella 0.435 · piqa 0.600 · wino 0.530 |
| S | `ARCH.V32X13GL1` | 0.801395x | **10**/15 | -- | 4 | dead | -- | arc 0.450 · hella 0.420 · piqa 0.615 · wino 0.530 |
| S | `ARCH.CL1536` | 0.808265x | **4**/15 | -- | 4 | dead | -- | arc 0.380 · hella 0.395 · piqa 0.565 · wino 0.525 |
| S | `ARCH.V22` | 0.808265x | **13**/15 | -- | 5 | dead | -- | arc 0.455 · hella 0.475 · piqa 0.625 · wino 0.530 |
| S | `ARCH.CL1792X13` | 0.813949x | **4**/15 | -- | 4 | dead | -- | arc 0.375 · hella 0.395 · piqa 0.575 · wino 0.540 |
| S | `ARCH.W896LSX13` | 0.815064x | **2**/15 | -- | 14 | dead | -- | arc 0.360 · hella 0.410 · piqa 0.520 · wino 0.485 |
| S | `ARCH.W896X13` | 0.815064x | **1**/15 | -- | 11 | dead | -- | arc 0.375 · hella 0.400 · piqa 0.520 · wino 0.495 |
| S | `ARCH.T010713KV6JQC` | 0.821894x | **0**/15 | -- | 10 | dead | -- | -- |
| S | `ARCH.CL1664` | 0.832232x | **5**/15 | -- | 4 | dead | -- | arc 0.380 · hella 0.425 · piqa 0.575 · wino 0.545 |
| S | `ARCH.W896KV6JS` | 0.841047x | **2**/15 | -- | 11 | dead | -- | arc 0.315 · hella 0.385 · piqa 0.520 · wino 0.515 |
| S | `ARCH.W896LSKV6JQ` | 0.841047x | **1**/15 | -- | 15 | dead | -- | arc 0.330 · hella 0.395 · piqa 0.500 · wino 0.555 |
| S | `ARCH.D3` | 0.842437x | **0**/15 | -- | 15 | dead | -- | -- |
| S | `ARCH.T010713` | 0.842437x | **3**/15 | -- | 5 | dead | -- | arc 0.390 · hella 0.415 · piqa 0.535 · wino 0.460 |
| S | `ARCH.T010913` | 0.842437x | **1**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.T030713` | 0.842437x | **0**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.T030913` | 0.842437x | **2**/15 | -- | 4 | dead | -- | -- |
| S | `ARCH.CL1728` | 0.844215x | **4**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.V32` | 0.853916x | **11**/15 | -- | 0 | alive-parent-band | -- | arc 0.485 · hella 0.485 · piqa 0.640 · wino 0.525 |
| S | `ARCH.BM25` | 0.856199x | **6**/15 | -- | 3 | dead | -- | arc 0.405 · hella 0.450 · piqa 0.590 · wino 0.540 |
| S | `ARCH.BM50` | 0.856199x | **4**/15 | -- | 4 | dead | -- | arc 0.405 · hella 0.445 · piqa 0.580 · wino 0.550 |
| S | `ARCH.BM50N` | 0.856199x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.BM75` | 0.856199x | **2**/15 | -- | 5 | dead | -- | arc 0.405 · hella 0.460 · piqa 0.570 · wino 0.590 |
| S | `ARCH.BMS75` | 0.856199x | **5**/15 | -- | 3 | dead | -- | arc 0.410 · hella 0.430 · piqa 0.540 · wino 0.575 |
| S | `ARCH.BMV3` | 0.856199x | **7**/15 | -- | 2 | dead | -- | arc 0.405 · hella 0.455 · piqa 0.565 · wino 0.555 |
| S | `ARCH.BV50` | 0.856199x | **4**/15 | -- | 4 | dead | -- | arc 0.400 · hella 0.460 · piqa 0.590 · wino 0.550 |
| S | `ARCH.CL1792` | 0.856199x | **7**/15 | -- | 3 | dead | -- | arc 0.385 · hella 0.420 · piqa 0.580 · wino 0.570 |
| S | `ARCH.CL1792LG` | 0.856199x | **7**/15 | -- | 3 | dead | -- | arc 0.385 · hella 0.420 · piqa 0.580 · wino 0.570 |
| S | `ARCH.CL1792M` | 0.856199x | **8**/15 | -- | 3 | dead | -- | arc 0.435 · hella 0.460 · piqa 0.535 · wino 0.560 |
| S | `ARCH.CL1792ME` | 0.856199x | **2**/15 | -- | 4 | dead | -- | arc 0.420 · hella 0.395 · piqa 0.555 · wino 0.480 |
| S | `ARCH.CL1792MV` | 0.856199x | **7**/15 | -- | 2 | dead | -- | arc 0.415 · hella 0.460 · piqa 0.550 · wino 0.565 |
| S | `ARCH.CL1792S` | 0.856199x | **4**/15 | -- | 4 | dead | -- | arc 0.375 · hella 0.430 · piqa 0.590 · wino 0.545 |
| S | `ARCH.MM896` | 0.859023x | **2**/15 | -- | 16 | dead | -- | arc 0.320 · hella 0.435 · piqa 0.555 · wino 0.505 |
| S | `ARCH.W896` | 0.859023x | **4**/15 | -- | 9 | dead | -- | arc 0.355 · hella 0.400 · piqa 0.565 · wino 0.490 |
| S | `ARCH.W896LS` | 0.859023x | **3**/15 | -- | 13 | dead | -- | arc 0.365 · hella 0.405 · piqa 0.565 · wino 0.510 |
| S | `ARCH.F1301KV6` | 0.874415x | **0**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.F1301KV6Q` | 0.874415x | **0**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.F1301KV6QC` | 0.874415x | **1**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.D2` | 0.894958x | **2**/15 | -- | 10 | dead | -- | -- |
| S | `ARCH.D2B` | 0.894958x | **6**/15 | -- | 5 | dead | -- | -- |
| S | `ARCH.P0709` | 0.894958x | **4**/15 | -- | 7 | dead | -- | -- |
| S | `ARCH.P0711` | 0.894958x | **6**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.P0911` | 0.894958x | **2**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.P1301` | 0.894958x | **7**/15 | -- | 2 | dead | -- | arc 0.410 · hella 0.375 · piqa 0.560 · wino 0.540 |
| S | `ARCH.P1303` | 0.894958x | **7**/15 | -- | 3 | dead | -- | -- |
| S | `ARCH.P1307` | 0.894958x | **6**/15 | -- | 1 | dead | -- | arc 0.455 · hella 0.405 · piqa 0.605 · wino 0.545 |
| P | `COMP.MA.MXL8` | 0.895457x | **3**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXL9` | 0.921164x | **5**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.X07KV6JQ` | 0.926936x | **6**/15 | -- | 4 | dead | -- | -- |
| S | `ARCH.X07KV6JQC` | 0.926936x | **7**/15 | -- | 3 | dead | -- | arc 0.420 · hella 0.420 · piqa 0.535 · wino 0.545 |
| S | `ARCH.X09KV6JQC` | 0.926936x | **5**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.X13KV6` | 0.926936x | **6**/15 | -- | 4 | dead | -- | -- |
| S | `ARCH.X13KV6J` | 0.926936x | **3**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.X13KV6JQ` | 0.926936x | **4**/15 | -- | 6 | dead | -- | -- |
| S | `ARCH.HD16` | 0.938368x | **0**/15 | -- | 4 | dead | -- | -- |
| P | `COMP.MA.MXL7` | 0.946871x | **9**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.X00` | 0.947479x | **0**/15 | -- | 16 | dead | -- | -- |
| S | `ARCH.X01` | 0.947479x | **9**/15 | -- | 1 | dead | -- | -- |
| S | `ARCH.X03` | 0.947479x | **6**/15 | -- | 0 | impaired | -- | -- |
| S | `ARCH.X05A` | 0.947479x | **6**/15 | -- | 1 | dead | -- | -- |
| S | `ARCH.X05T` | 0.947479x | **0**/15 | -- | 8 | dead | -- | -- |
| S | `ARCH.X07` | 0.947479x | **11**/15 | -- | 0 | alive-parent-band | -- | -- |
| S | `ARCH.X09` | 0.947479x | **10**/15 | -- | 0 | alive-parent-band | -- | -- |
| S | `ARCH.X11` | 0.947479x | **9**/15 | -- | 0 | alive-degraded | -- | -- |
| S | `ARCH.X13A` | 0.947479x | **13**/15 | -- | 2 | dead | -- | arc 0.460 · hella 0.405 · piqa 0.635 · wino 0.520 |
| S | `ARCH.X02` | 0.952057x | **0**/15 | -- | 6 | dead | -- | -- |
| S | `ARCH.X04` | 0.952057x | **2**/15 | -- | 9 | dead | -- | -- |
| S | `ARCH.X06` | 0.952057x | **8**/15 | -- | 1 | dead | -- | arc 0.530 · hella 0.470 · piqa 0.605 · wino 0.520 |
| S | `ARCH.X08` | 0.952057x | **7**/15 | -- | 1 | dead | -- | -- |
| S | `ARCH.X10` | 0.952057x | **7**/15 | -- | 0 | impaired | -- | arc 0.510 · hella 0.445 · piqa 0.620 · wino 0.520 |
| S | `ARCH.X12` | 0.952057x | **0**/15 | -- | 15 | dead | -- | -- |
| P | `COMP.MA.MXL11` | 0.955444x | **9**/15 | -- | -- | alive-degraded | -- | arc 0.525 · hella 0.460 · piqa 0.635 · wino 0.570 |
| S | `ARCH.KV4JQ` | 0.958914x | **0**/15 | -- | 10 | dead | -- | -- |
| S | `ARCH.KV4JS` | 0.958914x | **0**/15 | -- | 8 | dead | -- | -- |
| P | `COMP.MA.MXK10` | 0.972579x | **7**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXK7` | 0.972579x | **7**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXL5` | 0.972579x | **10**/15 | -- | -- | alive-parent-band | -- | arc 0.510 · hella 0.485 · piqa 0.620 · wino 0.550 |
| P | `COMP.MA.MXK11` | 0.972590x | **6**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXJP2C` | 0.972602x | **4**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXJ5` | 0.974301x | **4**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXJ4` | 0.977724x | **4**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXJ4C` | 0.977728x | **8**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXK15` | 0.978019x | **7**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXL6` | 0.978304x | **9**/15 | -- | -- | alive-degraded | -- | arc 0.515 · hella 0.460 · piqa 0.670 · wino 0.540 |
| P | `COMP.MA.MXK8` | 0.979437x | **7**/15 | -- | -- | dead | -- | -- |
| S | `ARCH.KV6` | 0.979457x | **5**/15 | -- | 4 | dead | -- | arc 0.465 · hella 0.435 · piqa 0.605 · wino 0.510 |
| S | `ARCH.KV6A` | 0.979457x | **5**/15 | -- | 2 | dead | -- | -- |
| S | `ARCH.KV6J` | 0.979457x | **3**/15 | -- | 3 | dead | -- | -- |
| S | `ARCH.KV6JQ` | 0.979457x | **8**/15 | -- | 2 | dead | -- | arc 0.465 · hella 0.440 · piqa 0.610 · wino 0.520 |
| S | `ARCH.KV6JS` | 0.979457x | **7**/15 | -- | 2 | dead | -- | arc 0.465 · hella 0.435 · piqa 0.610 · wino 0.535 |
| P | `COMP.MA.MXK13` | 0.979732x | **9**/15 | -- | -- | alive-degraded | -- | -- |
| P | `COMP.MA.MXJ3C` | 0.981152x | **9**/15 | -- | -- | alive-degraded | -- | -- |
| P | `COMP.MA.MXK5` | 0.981152x | **10**/15 | -- | -- | alive-parent-band | -- | -- |
| P | `COMP.MA.MXJP2A` | 0.981163x | **5**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXJP2B` | 0.981163x | **7**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXK1` | 0.981163x | **8**/15 | -- | -- | alive-degraded | -- | -- |
| P | `COMP.MA.MXK2` | 0.981163x | **7**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXK3` | 0.981163x | **6**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXL1` | 0.985726x | **10**/15 | -- | -- | alive-parent-band | -- | arc 0.505 · hella 0.455 · piqa 0.605 · wino 0.570 |
| P | `COMP.MA.MXL10` | 0.985726x | **11**/15 | -- | -- | alive-parent-band | -- | arc 0.530 · hella 0.445 · piqa 0.605 · wino 0.575 |
| P | `COMP.MB.MXL12` | 0.985726x | **8**/15 | -- | -- | alive-degraded | -- | -- |
| P | `COMP.MA.MXK9` | 0.986286x | **7**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXK12` | 0.986296x | **4**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXJ3` | 0.986297x | **6**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MB.MXJ3` | 0.986297x | **6**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXJ3A` | 0.986301x | **5**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXJ3B` | 0.986301x | **7**/15 | -- | -- | impaired | -- | -- |
| P | `COMP.MA.MXK4` | 0.988010x | **8**/15 | -- | -- | alive-degraded | -- | -- |
| P | `COMP.MA.MXQ1` | 0.989586x | **6**/15 | -- | -- | dead | -- | arc 0.500 · hella 0.475 · piqa 0.665 · wino 0.560 |
| P | `COMP.MA.MXQ2` | 0.989586x | **6**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXQ3` | 0.989586x | **4**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXJ2` | 0.989725x | **11**/15 | -- | -- | alive-parent-band | -- | -- |
| P | `COMP.MA.MXL2` | 0.990289x | **9**/15 | -- | -- | alive-degraded | -- | arc 0.585 · hella 0.440 · piqa 0.630 · wino 0.570 |
| P | `COMP.MA.MXL3` | 0.990289x | **11**/15 | -- | -- | dead | -- | -- |
| P | `COMP.MA.MXL4` | 0.990864x | **10**/15 | -- | -- | alive-parent-band | -- | arc 0.545 · hella 0.465 · piqa 0.635 · wino 0.580 |
| P | `COMP.MA.MXJ1` | 0.991439x | **12**/15 | -- | -- | alive-parent-band | -- | -- |
| P | `COMP.MA.MXK14` | 0.994287x | **13**/15 | -- | -- | alive-parent-band | -- | -- |
| P | `COMP.MA.MXK6` | 0.994859x | **10**/15 | -- | -- | alive-parent-band | -- | -- |

## 4. What this matrix does not say

It does not rank children by benchmark. The benchmark columns are ADDENDUM 8 diagnostics and have been anti-correlated with the review often enough that reading them as a score would invert the campaign's own findings. It does not gate anything on the extension column. And it does not claim a parent-shaped row is exportable: R3 is unchanged, and R4 — no quantization, no MoE — holds on every row of both columns.

