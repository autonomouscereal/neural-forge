# Phase 1 — the fold-only frontier on LFM2.5-230M

**Date:** 2026-08-13 (consolidated at wave 21, extended at waves 22 and 23).
**Branch:** `codex/lrc-blank-compression`.
**Campaign:** `reports/matrix-campaign-20260810/`, waves 1–23,
`NOTES-MX-A` … `NOTES-MX-AA`.
**Ledger:** `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`, **662 rows**, sha
`3967d7045a4aaaf87807283c4edc043577e40861325ada4af528e08272fe7e05`.
**One-page version for scoring:** `reports/matrix-campaign-20260810/CAMPAIGN_SCORECARD.md`.

This is the consolidating document of the **fold-only** matrix campaign: what
closed-form folding, with no gradient step and no distillation, reached on one
real 230M-parameter model, how it was measured, every law the measurements
support, every claim the campaign made and withdrew, and what it has not
reached.

Three standing rules from Cyril govern how it is written.

1. **Nothing is gated on drift.** The drift law below has an R² of 0.62 and its
   residuals pay: `COMP.MA.MXO3` is coherent at a global drift of 0.4618 against
   a nominal budget of 0.3765. Drift is a **search heuristic** — it says which
   lane to try first — and never a verdict. Children get built and read.
2. **Nothing is gated on efficiency.** A mechanism that buys review cases
   expensively is still a mechanism that buys review cases. Wave 12 retired
   corrective routes on a 355× parameter-efficiency comparison; that comparison
   is a recorded diagnostic here, not a killer. 2.1M parameters is 0.9% of this
   model.
3. **Nothing is gated on a census.** Every wave that priced a candidate before
   building it built it anyway, and §2.10 is the list of times the census would
   have picked the wrong child — in both directions.

**This report does not declare fold-only closed.** Closing the campaign is
Cyril's call and nobody else's. What follows is where the deepest
behaviourally-tested coherent child sits today, what has been measured about
why, and what is queued and unplayed.

**Storage is counted in parameters, everywhere, always.** Precision reduction —
quantization, codebooks — is *not folding* and is out of scope for phase 1 by
Cyril's ruling; it is reserved for after structural folding is proven.

---

## 0. How a number in this report is produced

* **The grade** is ADDENDUM 6 §6.1: the frozen 15-prompt review of the **whole
  model**, scored against the parent's own **12/15** on the same surface in the
  same process, with the band arithmetic computed by
  `matrix_cell_receipt.behavior_verdict_for` and never supplied by the reviewer.
  A sixteenth lane (safety) is read and charged but not scored.
* **Both readings are printed and neither is collapsed into the other**: the
  per-child ADDENDUM 6 verdict and the per-lane ADDENDUM 9 reading. `dead` in
  the per-child column is a **band**, not a description of the text — a child at
  8/15 with one degenerate lane prints `dead` — which is why the per-lane column
  and its lane count are always beside it.
* **Benchmarks** are `lm-eval` (`arc_easy`, `hellaswag`, `piqa`, `winogrande`,
  `lambada_openai`, `wikitext`) at `--limit 200 --seed 1234 --batch_size 4
  max_length=2048 dtype=float32` on an SM86 RTX 3090 — **partial evaluations by
  construction**, identical settings for every row.
* **Three calibrations run before any child is graded**, and all three have
  passed in every wave that ran them: the conditioned capture against the sealed
  collector (L00 relL2 3.367e-3 / cos 0.9999944, L05 1.647e-2 / 0.9997207, L13
  2.034e-2 / 0.9997842); the parent's own sixteen review responses, which digest
  to **`865f006e3813712e…`** on **every dossier in every wave**; and the parent's
  benchmark row, re-run in waves 8, 9, 16, 17, 18, 19, 20 and 21 and matching
  0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39, wikitext 81.80
  in **all eight**.
* **Every table in §1 is generated** from `wave21/campaign-frontier-table.json`
  by `run/mx21/mx21_report_tables.py`, which is itself a function of the ledger,
  the per-wave child indexes and the sealed `lm-eval` output. §3.11 is why:
  three consecutive waves quoted a frontier child the ledger does not name,
  because the number was carried in prose instead of recomputed.

---

## 1. Every sealed child that answers anything

**162 whole-model children have a sealed behaviour review. 115 hold at least one
of fifteen cases.** The 110 sealed through wave 22 are below, sorted by net
stored ratio, with both readings, the export shape and the benchmark row where
one was run.

**Wave 23's five rows are listed here rather than merged into the table, and
that is a debt, not a style choice.** The table is generated from
`wave22/campaign-frontier-table.json`; wave 23 wrote
`wave23/campaign-frontier-table.json` and did not regenerate the prose table,
so the 110 rows below are wave 22's and the five below them are wave 23's. Wave
24 should regenerate the whole thing from the wave-23 artifact.

| child | wave | net stored | stored params | review | per-child | per-lane | shape | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V16CL896` | 23 | **0.469305** | 107,796,224 | **1/15** | dead | impaired (15/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| `ARCH.V16W896X13` | 23 | **0.623329** | 143,174,400 | **1/15** | dead | impaired (15/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| `ARCH.V16W896` | 23 | **0.667288** | 153,271,424 | **2/15** | dead | impaired (12/16) | standard | 0.3100 | 0.4100 | 0.5050 | 0.5250 | 0.1450 | 1131.48 | 1246.14 |
| `ARCH.V16X13` | 23 | **0.728353** | 167,297,792 | **5/15** | dead | impaired (6/16) | standard | 0.4050 | 0.3950 | 0.6100 | 0.4950 | 0.3000 | 340.72 | 1048.24 |
| `ARCH.V16` | 23 | **0.780874** | 179,361,536 | **9/15** | **alive-degraded** | **none (0/16)** | standard | 0.4300 | 0.4550 | 0.6050 | 0.5200 | 0.3450 | **72.99** | 228.04 |

**`bits_per_byte` is the tokenizer-independent column and a vocabulary child
needs it**: `word_perplexity` is per *word* over a token stream, and a child that
segments differently is not on the same denominator. Parent **1.1882**, `V16`
**1.4648**, `V16X13` **1.8764**, `V16W896` **1.9230**. Eleven cells
carry more than one reviewed receipt (re-seals under ADDENDUM 14, re-runs); the
latest by `sealed_at` is shown and the superseded digests are listed in
`wave22/campaign-frontier-table.json`.

`shape` is a claim about what loads: **standard** means a stock
`Lfm2ForCausalLM`, no `trust_remote_code`, and the loader counts the ratio
shown; **parent-shaped** means the ratio is a **storage-format** ratio and the
model on disk is 229,693,184 parameters. §2.7 is why that distinction was
introduced and what it cost.

| # | child | wave | net stored | stored params | review | per-child | per-lane (reading) | shape | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ARCH.CL896` | 15 | **0.688431** | 158,127,872 | **4/15** | dead | impaired (3/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 2 | `ARCH.W736` | 21 | **0.689220** | 158,309,216 | **1/15** | dead | impaired (15/16) | standard | 0.2850 | 0.4050 | 0.5150 | 0.4800 | 0.0150 | 14146.56 | 1787.96 |
| 3 | `COMP.MA.MXO4` | 11 | **0.703396** | 161,565,310 | **5/15** | dead | `unstated` | parent-shaped | 0.4200 | 0.4350 | 0.5600 | 0.5250 | 0.1000 | 1477.86 | 257.83 |
| 4 | `COMP.MA.MXP2` | 12 | **0.703409** | 161,568,260 | **6/15** | dead | `unstated` | parent-shaped | 0.4250 | 0.4250 | 0.5850 | 0.5400 | 0.1450 | 857.20 | 231.96 |
| 5 | `COMP.MA.MXQ5` | 13 | **0.710270** | 163,144,199 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 6 | `COMP.MA.MXP1` | 12 | **0.712544** | 163,666,562 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 7 | `ARCH.W768` | 19 | **0.722610** | 165,978,624 | **3/15** | dead | impaired (9/16) | standard | 0.3100 | 0.4300 | 0.5200 | 0.4850 | 0.0300 | 8176.90 | 1279.46 |
| 8 | `ARCH.W768EV` | 20 | **0.722610** | 165,978,624 | **1/15** | dead | dead (16/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 9 | `ARCH.W768LS` | 20 | **0.722610** | 165,978,624 | **3/15** | dead | impaired (10/16) | standard | 0.3050 | 0.4300 | 0.5050 | 0.4600 | 0.0300 | 7722.72 | 1312.49 |
| 10 | `ARCH.W768ST` | 20 | **0.722610** | 165,978,624 | **2/15** | dead | impaired (10/16) | standard | 0.2900 | 0.4350 | 0.5150 | 0.4550 | 0.0350 | 7449.50 | 1297.15 |
| 11 | `ARCH.BX13` | 16 | **0.724929** | 166,511,360 | **1/15** | dead | impaired (5/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 12 | `COMP.MA.MXO3` | 11 | **0.729103** | 167,470,079 | **8/15** | dead | `unstated` | parent-shaped | 0.4300 | 0.4300 | 0.5900 | 0.5450 | 0.1650 | 527.91 | 190.15 |
| 13 | `COMP.MA.MXN3` | 10 | **0.735962** | 169,045,495 | **6/15** | dead | `unstated` | parent-shaped | 0.4750 | 0.4300 | 0.5800 | 0.5650 | 0.1200 | 797.61 | 193.24 |
| 14 | `COMP.MA.MXM9` | 9 | **0.736305** | 169,124,342 | **4/15** | dead | `unstated` | parent-shaped | 0.3700 | 0.3150 | 0.5100 | 0.5150 | 0.0700 | 15185.40 | 515.07 |
| 15 | `COMP.MA.MXN1` | 10 | **0.744033** | 170,899,201 | **7/15** | dead | `unstated` | parent-shaped | 0.3600 | 0.3900 | 0.5750 | 0.5100 | 0.0650 | 3765.13 | 432.10 |
| 16 | `COMP.MA.MXM8` | 9 | **0.744297** | 170,960,000 | **3/15** | dead | `unstated` | parent-shaped | 0.2750 | 0.2800 | 0.5150 | 0.4750 | 0.0550 | 2315739.18 | 11527.79 |
| 17 | `COMP.MA.MXM10` | 9 | **0.753440** | 173,059,959 | **6/15** | dead | `unstated` | parent-shaped | 0.3950 | 0.3200 | 0.5400 | 0.4650 | 0.0800 | 11613.26 | 501.34 |
| 18 | `COMP.MA.MXN4` | 10 | **0.754811** | 173,374,848 | **8/15** | dead | `unstated` | parent-shaped | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.09 | 168.70 |
| 19 | `ARCH.W14FF1280` | 15 | **0.760331** | 174,642,944 | **5/15** | dead | impaired (3/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 20 | `ARCH.W1536X13` | 16 | **0.769439** | 176,734,976 | **4/15** | dead | impaired (3/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 21 | `COMP.MA.MXN5` | 10 | **0.780411** | 179,255,041 | **8/15** | dead | `unstated` | parent-shaped | 0.4350 | 0.4700 | 0.6100 | 0.5000 | 0.1300 | 355.02 | 179.00 |
| 22 | `COMP.MA.MXN2` | 10 | **0.780518** | 179,279,617 | **10/15** | alive-parent-band | `unstated` | parent-shaped | 0.4550 | 0.4450 | 0.6400 | 0.5550 | 0.2000 | 230.31 | 139.21 |
| 23 | `COMP.MA.MXM5` | 9 | **0.780727** | 179,327,744 | **6/15** | dead | `unstated` | parent-shaped | 0.3800 | 0.3600 | 0.5400 | 0.5500 | 0.0900 | 2628.37 | 313.97 |
| 24 | `COMP.MA.MXM3` | 9 | **0.780772** | 179,337,984 | **3/15** | dead | `unstated` | parent-shaped | 0.3850 | 0.4400 | 0.5600 | 0.5100 | 0.0600 | 5496.31 | 471.21 |
| 25 | `COMP.MA.MXM7` | 9 | **0.780830** | 179,351,440 | **5/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 26 | `COMP.MA.MXM6` | 9 | **0.780861** | 179,358,464 | **9/15** | dead | `unstated` | parent-shaped | 0.3650 | 0.3200 | 0.5200 | 0.5350 | 0.1100 | 7337.19 | 421.68 |
| 27 | `ARCH.W896LSX13` | 21 | **0.815064** | 187,214,592 | **2/15** | dead | impaired (14/16) | standard | 0.3600 | 0.4100 | 0.5200 | 0.4850 | 0.1400 | 1675.13 | 915.75 |
| 28 | `ARCH.W896X13` | 19 | **0.815064** | 187,214,592 | **1/15** | dead | impaired (11/16) | standard | 0.3750 | 0.4000 | 0.5200 | 0.4950 | 0.1500 | 1671.03 | 923.07 |
| 29 | `ARCH.W896KV6JS` | 19 | **0.841047** | 193,182,848 | **2/15** | dead | impaired (11/16) | standard | 0.3150 | 0.3850 | 0.5200 | 0.5150 | 0.0650 | 2821.11 | 1161.46 |
| 30 | `ARCH.W896LSKV6JQ` | 22 | **0.841047** | 193,182,848 | **1/15** | dead | impaired (15/16) | standard | 0.3300 | 0.3950 | 0.5000 | 0.5550 | 0.0600 | 2829.20 | 1128.35 |
| 31 | `ARCH.T010713` | 18 | **0.842437** | 193,501,952 | **3/15** | dead | impaired (5/16) | standard | 0.3900 | 0.4150 | 0.5350 | 0.4600 | 0.1000 | 14860.55 | 1089.79 |
| 32 | `ARCH.T010913` | 18 | **0.842437** | 193,501,952 | **1/15** | dead | impaired (8/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 33 | `ARCH.T030913` | 18 | **0.842437** | 193,501,952 | **2/15** | dead | impaired (4/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 34 | `ARCH.MM896` | 22 | **0.859023** | 197,311,616 | **2/15** | dead | dead (16/16) | standard | 0.3200 | 0.4350 | 0.5550 | 0.5050 | 0.0550 | 1880.53 | 554.62 |
| 35 | `ARCH.W896` | 19 | **0.859023** | 197,311,616 | **4/15** | dead | impaired (9/16) | standard | 0.3550 | 0.4000 | 0.5650 | 0.4900 | 0.1450 | 638.72 | 500.08 |
| 36 | `ARCH.W896LS` | 21 | **0.859023** | 197,311,616 | **3/15** | dead | impaired (13/16) | standard | 0.3650 | 0.4050 | 0.5650 | 0.5100 | 0.1400 | 622.20 | 487.20 |
| 37 | `ARCH.F1301KV6QC` | 18 | **0.874415** | 200,847,104 | **1/15** | dead | dead (2/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 38 | `ARCH.D2` | 16 | **0.894958** | 205,565,696 | **2/15** | dead | impaired (10/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 39 | `ARCH.D2B` | 16 | **0.894958** | 205,565,696 | **6/15** | dead | impaired (5/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 40 | `ARCH.P0709` | 17 | **0.894958** | 205,565,696 | **4/15** | dead | impaired (7/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 41 | `ARCH.P0711` | 17 | **0.894958** | 205,565,696 | **6/15** | dead | impaired (2/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 42 | `ARCH.P0911` | 17 | **0.894958** | 205,565,696 | **2/15** | dead | impaired (8/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 43 | `ARCH.P1301` | 17 | **0.894958** | 205,565,696 | **7/15** | dead | impaired (2/16) | standard | 0.4100 | 0.3750 | 0.5600 | 0.5400 | 0.1750 | 2870.17 | 990.35 |
| 44 | `ARCH.P1303` | 17 | **0.894958** | 205,565,696 | **7/15** | dead | impaired (3/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 45 | `ARCH.P1307` | 17 | **0.894958** | 205,565,696 | **6/15** | dead | impaired (1/16) | standard | 0.4550 | 0.4050 | 0.6050 | 0.5450 | 0.1300 | 1991.53 | 400.16 |
| 46 | `COMP.MA.MXL8` | 8 | **0.895457** | 205,680,372 | **3/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 47 | `COMP.MA.MXL9` | 8 | **0.921164** | 211,585,141 | **5/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 48 | `ARCH.X07KV6JQ` | 18 | **0.926936** | 212,910,848 | **6/15** | dead | impaired (4/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 49 | `ARCH.X07KV6JQC` | 18 | **0.926936** | 212,910,848 | **7/15** | dead | impaired (3/16) | standard | 0.4200 | 0.4200 | 0.5350 | 0.5450 | 0.1200 | 740.48 | 377.65 |
| 50 | `ARCH.X09KV6JQC` | 18 | **0.926936** | 212,910,848 | **5/15** | dead | impaired (2/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 51 | `ARCH.X13KV6` | 16 | **0.926936** | 212,910,848 | **6/15** | dead | impaired (4/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 52 | `ARCH.X13KV6J` | 17 | **0.926936** | 212,910,848 | **3/15** | dead | impaired (8/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 53 | `ARCH.X13KV6JQ` | 17 | **0.926936** | 212,910,848 | **4/15** | dead | impaired (6/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 54 | `COMP.MA.MXL7` | 8 | **0.946871** | 217,489,910 | **9/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 55 | `ARCH.X01` | 16 | **0.947479** | 217,629,440 | **9/15** | dead | alive-degraded (1/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 56 | `ARCH.X03` | 16 | **0.947479** | 217,629,440 | **6/15** | impaired | `unstated` | standard | -- | -- | -- | -- | -- | -- | -- |
| 57 | `ARCH.X05A` | 15 | **0.947479** | 217,629,440 | **6/15** | dead | impaired (1/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 58 | `ARCH.X07` | 16 | **0.947479** | 217,629,440 | **11/15** | alive-parent-band | `unstated` | standard | -- | -- | -- | -- | -- | -- | -- |
| 59 | `ARCH.X09` | 16 | **0.947479** | 217,629,440 | **10/15** | alive-parent-band | `unstated` | standard | -- | -- | -- | -- | -- | -- | -- |
| 60 | `ARCH.X11` | 16 | **0.947479** | 217,629,440 | **9/15** | alive-degraded | `unstated` | standard | -- | -- | -- | -- | -- | -- | -- |
| 61 | `ARCH.X13A` | 16 | **0.947479** | 217,629,440 | **13/15** | dead | alive-parent-band (2/16) | standard | 0.4600 | 0.4050 | 0.6350 | 0.5200 | 0.3350 | 106.78 | 297.09 |
| 62 | `ARCH.X04` | 17 | **0.952057** | 218,680,960 | **2/15** | dead | impaired (9/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 63 | `ARCH.X06` | 17 | **0.952057** | 218,680,960 | **8/15** | dead | alive-degraded (1/16) | standard | 0.5300 | 0.4700 | 0.6050 | 0.5200 | 0.3250 | 39.52 | 127.10 |
| 64 | `ARCH.X08` | 17 | **0.952057** | 218,680,960 | **7/15** | dead | impaired (1/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 65 | `ARCH.X10` | 17 | **0.952057** | 218,680,960 | **7/15** | impaired | impaired (0/16) | standard | 0.5100 | 0.4450 | 0.6200 | 0.5200 | 0.3000 | 153.37 | 239.04 |
| 66 | `COMP.MA.MXL11` | 8 | **0.955444** | 219,459,062 | **9/15** | alive-degraded | `unstated` | parent-shaped | 0.5250 | 0.4600 | 0.6350 | 0.5700 | 0.2250 | 99.22 | 118.02 |
| 67 | `COMP.MA.MXK10` | 7 | **0.972579** | 223,394,679 | **7/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 68 | `COMP.MA.MXK7` | 7 | **0.972579** | 223,394,679 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 69 | `COMP.MA.MXL5` | 8 | **0.972579** | 223,394,679 | **10/15** | alive-parent-band | `unstated` | parent-shaped | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.19 | 105.53 |
| 70 | `COMP.MA.MXK11` | 7 | **0.972590** | 223,397,366 | **6/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 71 | `COMP.MA.MXJP2C` | 6 | **0.972602** | 223,400,053 | **4/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 72 | `COMP.MA.MXJ5` | 6 | **0.974301** | 223,790,198 | **4/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 73 | `COMP.MA.MXJ4` | 6 | **0.977724** | 224,576,630 | **4/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 74 | `COMP.MA.MXJ4C` | 6 | **0.977728** | 224,577,399 | **8/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 75 | `COMP.MA.MXK15` | 7 | **0.978019** | 224,644,232 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 76 | `COMP.MA.MXL6` | 8 | **0.978304** | 224,709,687 | **9/15** | alive-degraded | `unstated` | parent-shaped | 0.5150 | 0.4600 | 0.6700 | 0.5400 | 0.3350 | 59.78 | 112.29 |
| 77 | `COMP.MA.MXK8` | 7 | **0.979437** | 224,970,095 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 78 | `ARCH.KV6` | 16 | **0.979457** | 224,974,592 | **5/15** | dead | impaired (4/16) | standard | 0.4650 | 0.4350 | 0.6050 | 0.5100 | 0.2150 | 107.87 | 201.05 |
| 79 | `ARCH.KV6A` | 17 | **0.979457** | 224,974,592 | **5/15** | dead | impaired (2/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 80 | `ARCH.KV6J` | 17 | **0.979457** | 224,974,592 | **3/15** | dead | impaired (3/16) | standard | -- | -- | -- | -- | -- | -- | -- |
| 81 | `ARCH.KV6JQ` | 17 | **0.979457** | 224,974,592 | **8/15** | dead | alive-degraded (2/16) | standard | 0.4650 | 0.4400 | 0.6100 | 0.5200 | 0.2300 | 104.96 | 241.89 |
| 82 | `ARCH.KV6JS` | 18 | **0.979457** | 224,974,592 | **7/15** | dead | impaired (2/16) | standard | 0.4650 | 0.4350 | 0.6100 | 0.5350 | 0.2350 | 103.45 | 240.46 |
| 83 | `COMP.MA.MXK13` | 7 | **0.979732** | 225,037,687 | **9/15** | alive-degraded | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 84 | `COMP.MA.MXJ3C` | 6 | **0.981152** | 225,363,831 | **9/15** | alive-degraded | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 85 | `COMP.MA.MXK5` | 7 | **0.981152** | 225,363,831 | **10/15** | alive-parent-band | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 86 | `COMP.MA.MXJP2A` | 6 | **0.981163** | 225,366,518 | **5/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 87 | `COMP.MA.MXJP2B` | 6 | **0.981163** | 225,366,518 | **7/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 88 | `COMP.MA.MXK1` | 7 | **0.981163** | 225,366,518 | **8/15** | alive-degraded | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 89 | `COMP.MA.MXK2` | 7 | **0.981163** | 225,366,518 | **7/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 90 | `COMP.MA.MXK3` | 7 | **0.981163** | 225,366,518 | **6/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 91 | `COMP.MA.MXL1` | 8 | **0.985726** | 226,414,582 | **10/15** | alive-parent-band | `unstated` | parent-shaped | 0.5050 | 0.4550 | 0.6050 | 0.5700 | 0.3100 | 57.18 | 99.58 |
| 92 | `COMP.MA.MXL10` | 8 | **0.985726** | 226,414,582 | **11/15** | alive-parent-band | `unstated` | parent-shaped | 0.5300 | 0.4450 | 0.6050 | 0.5750 | 0.3200 | 63.88 | 109.08 |
| 93 | `COMP.MB.MXL12` | 8 | **0.985726** | 226,414,582 | **8/15** | alive-degraded | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 94 | `COMP.MA.MXK9` | 7 | **0.986286** | 226,543,095 | **7/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 95 | `COMP.MA.MXK12` | 7 | **0.986296** | 226,545,511 | **4/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 96 | `COMP.MA.MXJ3` | 6 | **0.986297** | 226,545,782 | **6/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 97 | `COMP.MB.MXJ3` | 6 | **0.986297** | 226,545,782 | **6/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 98 | `COMP.MA.MXJ3A` | 6 | **0.986301** | 226,546,551 | **5/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 99 | `COMP.MA.MXJ3B` | 6 | **0.986301** | 226,546,551 | **7/15** | impaired | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 100 | `COMP.MA.MXK4` | 7 | **0.988010** | 226,939,247 | **8/15** | alive-degraded | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 101 | `COMP.MA.MXQ1` | 13 | **0.989586** | 227,301,120 | **6/15** | dead | `unstated` | parent-shaped | 0.5000 | 0.4750 | 0.6650 | 0.5600 | 0.1450 | 341.58 | 105.94 |
| 102 | `COMP.MA.MXQ2` | 13 | **0.989586** | 227,301,120 | **6/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 103 | `COMP.MA.MXQ3` | 13 | **0.989586** | 227,301,120 | **4/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 104 | `COMP.MA.MXJ2` | 6 | **0.989725** | 227,332,983 | **11/15** | alive-parent-band | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 105 | `COMP.MA.MXL2` | 8 | **0.990289** | 227,462,646 | **9/15** | alive-degraded | `unstated` | parent-shaped | 0.5850 | 0.4400 | 0.6300 | 0.5700 | 0.3200 | 52.82 | 100.57 |
| 106 | `COMP.MA.MXL3` | 8 | **0.990289** | 227,462,646 | **11/15** | dead | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 107 | `COMP.MA.MXL4` | 8 | **0.990864** | 227,594,615 | **10/15** | alive-parent-band | `unstated` | parent-shaped | 0.5450 | 0.4650 | 0.6350 | 0.5800 | 0.3700 | 38.61 | 92.29 |
| 108 | `COMP.MA.MXJ1` | 6 | **0.991439** | 227,726,719 | **12/15** | alive-parent-band | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 109 | `COMP.MA.MXK14` | 7 | **0.994287** | 228,381,047 | **13/15** | alive-parent-band | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |
| 110 | `COMP.MA.MXK6` | 7 | **0.994859** | 228,512,247 | **10/15** | alive-parent-band | `unstated` | parent-shaped | -- | -- | -- | -- | -- | -- | -- |

### 1.1 The recipes behind the top of that table

* **`ARCH.V16` — 0.780874x, 9/15, standard shape (wave 23).** `vocab_size`
  16,384: a **verbatim row coreset** of the tied token table. Every kept row is
  the parent's own bytes (the identity gate is `torch.equal`, not a tolerance),
  the BPE merge list is restricted to the closure of what was kept, the ids are
  densely remapped and a stock `tokenizer.json` is emitted. **Nothing is solved,
  no capture is read, one tensor is touched.** It is the deepest standard-shape
  child at 8/15 or better by seventeen points of ratio, it sits at `MXN2`'s own
  ratio one case below it with a stock loader instead of a reconstituted
  `lm_head`, and **zero of sixteen lanes hit the token cap**. Its cost is
  sequence length: 0.2473 child tokens per byte against 0.2135, **+15.8%** on the
  review prompts. The census behind the selection is thin — the sealed corpus
  has 2,030 distinct tokens — so roughly 14,000 of the 16,384 rows are chosen by
  the tie-break, parent token id order, which for a BPE table is its own
  training-corpus frequency order. It is a tokenizer-frequency-prefix coreset and
  the receipt says so.
* **`ARCH.V16CL896` — 0.469305x, 1/15, standard shape (wave 23).** The same
  fold on `CL896`'s body: **under half the parent**, and the first child below
  0.5x in this campaign that holds anything at all.

* **`ARCH.CL896` — 0.688431x, 4/15, standard shape (wave 15).** The deepest
  scoring child in the campaign. `intermediate_size` 896 at every MLP site,
  solved **conditioned-local** — each site against a capture taken through the
  partially-built child rather than through the parent. It is **2.86x per site**,
  which is why wave 15 declined to call it a frontier claim under a 4x-class
  eligibility rule; that rule was then not applied to the child wave 16
  promoted, and §3.11 is the correction.
* **`ARCH.W736` — 0.689220x, 1/15, standard shape (wave 21).** `hidden_size`
  736: the residual stream itself narrowed, which is the only standard axis that
  reaches the tied embedding. 0.0008 of net ratio shallower than `CL896` and
  three review cases worse, so it does not take the frontier.
* **`COMP.MA.MXP2` — 0.703409x, 6/15, parent-shaped (wave 12).** The deepest
  child whose output is coherent, on-prompt language in most lanes — ADDENDUM 8
  §8.2's success band. The `MXN2` token interface plus three whole SwiGLU lanes,
  the third **shopped** rather than taken in depth order.
* **`COMP.MA.MXQ5` — 0.710270x, 7/15, parent-shaped (wave 13).** `MXP2` plus
  three ADDENDUM 10 corrective routes at k=256, whose 1,575,939 added parameters
  are charged in full. The healthiest deep text in the campaign: two `Useful`
  gradings and one lane at the token cap.
* **`ARCH.W768` / `W768LS` / `W768ST` / `W768EV` — 0.722610x, 3 / 3 / 2 / 1 of
  15 (waves 19–20).** Four children at the **identical** 165,978,624 parameters
  against the same corpus and the same solver, differing only in the residual
  basis or the solve. This is the campaign's cleanest controlled experiment and
  §2.9 is what it says.
* **`COMP.MA.MXO3` — 0.729103x, 8/15, parent-shaped (wave 11).** The best
  reading below 0.78x. `MXN2`'s interface plus two whole SwiGLU lanes at L05 and
  L06, every component `T8` (Gram relation matching) at `S4`.
* **`COMP.MA.MXN2` — 0.780518x, 10/15, `alive-parent-band`, parent-shaped
  (wave 10).** The deepest child that reads at the parent's own band, and it is
  **one component**: the token interface alone, a 12,288-row verbatim head
  selected by token frequency plus a sparse-OMP tail at rank 38 over the
  remaining 53,248 rows. Nothing in the body is folded.
* **`ARCH.X13A` — 0.947479x, 13/15 (wave 16).** One conv block excised. The
  highest-scoring child in the campaign, **one case above the same-process
  parent**, and it removes 5.25% of the model.

### 1.2 The children that are not in the table, and why

* **`ARCH.W512` — 0.463481x, 0/15 (wave 19).** 106,458,368 parameters, **under
  half the parent**, standard shape, loads and counts. Eight of sixteen lanes
  return the empty string and eight fill the cap with punctuation. It is the
  deepest artifact the campaign has produced and it is not a model.
* **`COMP.MA.MXO2` — 0.497739x, 0/15 (wave 11)** and **`COMP.MA.MXQ6` —
  0.497777x, 0/15 (wave 13).** The 0.5x storage target reached twice, in two
  different lane orders — by depth and by measured-cheapest — with the same
  result. Fourteen of sixteen lanes on the first ride a single phrase to the
  token cap; ten of sixteen on the second.
* **`ARCH.MM768` — 0.722610x, 0/15 (wave 21).** The basis that maximises the
  minimum per-endpoint retained read energy. It is in this list rather than the
  table because it holds nothing, and §2.9 is why it matters anyway.
* **`COMP.MA.MXP1` has no benchmark row at all**, because its corrective routes
  are forward hooks rather than `state_dict` entries and the materializer cannot
  write an HF child that carries them. That gap was stated in `NOTES-MX-P` §6a
  and is still open.

---

## 2. The laws, each with the measurement that supports it

### 2.1 Behaviour-first: every cheaper gate has been measured and every one fails

This is the campaign's first law and the one the rest depend on. Four proxies
for behaviour were used or proposed as gates, and all four were measured against
the frozen review and retired:

* **Token match against the `copy_exact` control** was the campaign's original
  pass/fail. Wave 4 sealed a child at **0.551 token match and 12/15**, the
  parent's own band, and other children match tokens while incoherent. Demoted
  to a diagnostic column, printed on every dossier labelled `[DIAGNOSTIC ONLY]`.
* **Accuracy benchmarks** disagree with the review in **both directions** and
  §2.11 is the list.
* **Drift** has R² = 0.62 against the review and its residuals are the
  interesting children, not noise (§2.2).
* **Parameter efficiency** was used once to kill a mechanism and the kill was
  withdrawn (§3.9).

There is no cheap surrogate. **A child gets built and read**, and every claim in
this document that is not a frozen-review reading says which instrument produced
it.

### 2.2 The drift law, and exactly where it stops working

Across 55 sealed composed rungs:

```
review(useful-or-partial, /15) = 12.9554 − 13.1626 × global_drift     R² = 0.6243, n = 55
review = 8   ->   drift 0.376476        (the nominal coherence budget)
```

**38% of the variance in behaviour is not drift**, and the residuals pay:

| child | drift | predicted | sealed | residual |
|---|---:|---:|---:|---:|
| `COMP.MA.MXO3` | 0.461810 | 6.88 | **8** | +1.1 |
| `COMP.MA.MXO4` | 0.540899 | 5.84 | 5 | −0.8 |
| `COMP.MA.MXO2` | 0.814086 | 2.24 | 0 | −2.2 |
| `COMP.MA.MXQ2` | 0.399369 | 7.70 | **6** | −1.7 |
| `COMP.MA.MXQ1` | 0.429852 | 7.30 | **6** | −1.3 |

**The law ranks the wave-13 pair backwards.** `MXQ2` (head merge at L02) has
*lower* drift than `MXQ1` (head merge at L12) — 0.399 vs 0.430 — and **four**
review lanes that loop to the 1024-token cap against `MXQ1`'s **one**. Same
score, worse text, lower drift.

**And at depth drift stops discriminating at all.** `COMP.MA.MXO2`'s eleven-lane
install trace: 0.3659, 0.1139, 0.0421, 0.0687, 0.0685, 0.0530, 0.0518, 0.0297,
0.0054, 0.0105, **−0.0061** — the last lane *reduced* drift. Once a child is far
enough from the parent, moving it further barely registers on a
distance-to-parent metric, which is exactly the depth where a gate would be doing
the most work.

### 2.3 Additive interference, and the second composition step

**The single most reliable finding in the campaign.** Two folds that are
individually fine compose to something much worse than either, and the effect is
not a rounding error:

| composition | the singles | composed | axes |
|---|---|---:|---|
| `ARCH.BX13` | 13/15 and 5/15 | **1/15** | excision + `intermediate_size` |
| `ARCH.W896X13` | 13/15 and 4/15 | **1/15** | excision + `hidden_size` |
| `ARCH.W896KV6JS` | 7/15 and 4/15 | **2/15** | head merge + `hidden_size` |
| `ARCH.X13KV6J` | 13/15 and 3/15 | **3/15** | excision + head merge |
| `COMP.MA.MXQ4` | six lanes + two merges | **0/15** | SwiGLU + head merge |

**Wave 23 added the cleanest test the rule has been given, and it held.** The
vocabulary fold and the block-13 excision **share no tensor**: one gathers rows
out of the embedding, the other deletes a conv block, and neither re-solves
anything the other touched. If additive interference were a *contention* effect
— two folds competing for one residual budget — that pair should have been free.
`ARCH.V16X13` is **13/15 and 9/15 composing to 5/15**. So the cross term is real
between folds on disjoint tensors, **L3 is not a contention effect**, and its
magnitude does depend on which folds: the same excision with the width fold is
13 + 4 -> 1, so this is the **mildest** second step in 662 rows. The third step
is still fatal (`ARCH.V16W896X13`, 0.623329x, **1/15**).

**The rule that survives every test of it: the first composition step is
survivable and the second is fatal.** `ARCH.X07KV6JQC` is the best two-axis
child in the campaign at **7/15, 0.926936x**; every three-axis child measured is
at or below 3/15. Wave 19 tried the rule on a brand-new axis (`hidden_size`) and
it held.

**The excision ladder is the clean form of it**, complete and monotone:

| blocks excised | net | review |
|---|---:|---:|
| 1 (block 13) | 0.947479x | **13/15** |
| 2 (13 and 1) | 0.894958x | **7/15** |
| 3 (1, 7, 13) | 0.842437x | **3/15** |
| 4 (1, 5, 9, 13) | 0.789916x | **0/15** |

### 2.4 The distance law between excisions, and its collapse at three bodies

Wave 17 measured the cost of removing two blocks as a function of how far apart
they are: **2 apart reads 2–4 of 15, twelve apart reads 7**, monotone across two
anchors (`P0911` 2/15, `P0709` 4/15, `P1307` 6/15, `P1301` and `P1303` 7/15).
The mechanism is plausible — adjacent blocks share more of the same residual
subspace — and it does **not survive to three bodies**: the widest-spread triple
reads 3/15 against the best pair's 7/15, and the three triples measured
(`T010713` 3/15, `T030913` 2/15, `T010913` 1/15) do not order by spread.

### 2.5 Deeper-is-sometimes-safer, and it is one component

`S8` (8x) beats `S4` (4x) on the short conv writer `C05` — **13/15 against
10/15**, one case *above* the same-process parent — and **loses on every SwiGLU
site measured**. The level ladder is per component and is not monotone. This is
also the correction of a wrong belief: `C05` was thought to poison a block until
the block rule explained the earlier reading (§3.3).

### 2.6 The block-conflict rule and lane concentration

**A block hosts at most one folded lane.** Two MLP components in one block cost
nothing; a conv or attention writer **plus** an MLP fold in the same block costs
−3 to −4 review cases, because they sit on opposite sides of that block's FFN
norm. Machine-enforced in the recipe generator, overridable only with a written
reason.

**And a lane's fragility is not uniform.** The completed excision census made a
free probe possible: how many of the thirteen single-block excisions each of the
fifteen review lanes survives. The answer concentrates — a few lanes survive
almost every excision and a few survive almost none — which is the same
structure the per-case attribution artifact found from the other side (§2.12).

### 2.7 R3 and the re-pricing: parent-shaped is not standard-shape

ADDENDUM 11's R3 asks a frontier claim to have a standard-architecture export
path, and applying it to this document's own table changed what the table says.

**Every child in waves 5–13 is parent-shaped.** `T1`/`T4`/`T5`/`T8`/`T7` store a
factored, sparse, tied or merged form and reconstruct the parent's shapes to
run, so `0.703409x` and `0.729103x` are **storage-format ratios**; the model on
disk is 229,693,184 parameters. Each is an honest count of stored parameters
under a registered, tested storage model, and it is a different quantity from
the one R3 names.

Three axes on this checkpoint are config-standard against `transformers` 5.12.1
— `intermediate_size`/`block_ff_dim`, `head_dim`, `num_hidden_layers` +
`layer_types` — and wave 19 added a fourth, `hidden_size` (with `block_dim` and
`conv_dim`, this checkpoint's own restatements of the residual width). **Conv
channel count is not**: `Lfm2ShortConv.__init__` hardwires `config.hidden_size`
and never reads `conv_dim`, so a conv-width fold is a research probe under R3 as
a fact about the modeling file.

**What the re-pricing cost:** the deepest *standard-shape* child that answers
anything is `ARCH.CL896` at 0.688431x and **4/15**, against a parent-shaped
`COMP.MA.MXO3` at 0.729103x and 8/15. The standard-export requirement bought
4 points of net ratio and cost 4 review cases.

### 2.8 The RoPE plane constraint on `head_dim`

`head_dim` 16 is **not available on this architecture**, and the reason is
structural rather than empirical. RoPE's rotation is block diagonal over
frequency planes, so `q^T R_Δ k` is exactly a sum of per-plane terms and a
smaller `head_dim` can keep whole planes on a uniform stride and nothing else.
The three expressible ladders retain **0.2394 / 0.2379 / 0.2018** of the
evidence-weighted score energy, against the 0.99 that an unconstrained `Δ=0`
census promised for the same operator. Built with both scale corrections the
change forces, it is dead alone at 0.938368x (`ARCH.HD16`, 0/15).

### 2.9 The retained-read-energy law, its ceiling, and the flatness underneath it

The `hidden_size` axis produced the campaign's most predictive single variable
and then bounded it three times.

**The law.** Behaviour tracks the fraction of the residual stream's **reader-
projected** energy the child's subspace retains, pooled over the 29 endpoints
where a stock LFM2 reads the stream:

| width | net | retained read (held-out) | worst endpoint | readers under 0.90 | review |
|---:|---:|---:|---:|---:|---:|
| 896 | 0.859023x | 0.9627 | 0.9440 | 0 of 29 | **4/15** |
| 768 | 0.722610x | 0.9193 | 0.8801 | 4 of 29 | **3/15** |
| 736 | 0.689220x | 0.9075 | 0.8632 | 9 of 29 | **1/15** |
| **720** | **0.672632x** | **0.9015** | **0.8545** | **11 of 29** | **0/15** |
| 704 | 0.656116x | 0.8953 | — | — | 0/15 |
| 640 | 0.590763x | 0.8697 | 0.8084 | 26 of 29 | 0/15 |
| 512 | 0.463481x | 0.8124 | 0.7268 | 29 of 29 | 0/15 |

**The cliff is 16 dimensions wide and it is a slope.** Wave 19 bracketed it at
128 and called it a step; waves 20, 21 and 22 halved it three times. The last
interval that can still hide a step is **736 → 720**: sixteen dimensions and
**0.0060** of pooled retained read energy, between the deepest child that holds
a case and the deepest that holds none.

**Per-site install quality is not the variable.** At 512 the attention writes
land at cos **0.9999** against their own local targets, the MLP writes at
0.9999, and the conv writes at 0.9916 median — every one of them *better* than
the living 896 child's conv — and the model returns the empty string. A fold can
be locally excellent everywhere and globally dead.

**The spectrum is flat, so there was never a subspace to find.** Pooled over all
29 endpoints, the top 768 of 1024 directions hold 0.9389 of the read energy and
the top 512 hold 0.8465. There is no knee; there is a price per direction.

**The retained fraction is a ceiling, not a predictor — and this is the sharpest
result the campaign has.** Six children at the **identical** 165,978,624
parameters, same corpus, same solver, same code, differing only in the basis or
the solve:

| child | what moved | retained read (hld) | directions exchanged vs plain | review |
|---|---|---:|---:|---:|
| `ARCH.W768` | — the baseline | 0.9193 | 0 | **3/15** |
| `ARCH.W768LS` | writers' λ on an internal fit split | 0.9193 | 0 | **3/15** |
| `ARCH.W768ST` | `--arm stale`, nothing re-solved | 0.9193 | 0 | **2/15** |
| `ARCH.W768AW25` | case-weighted basis, beta 0.25 | 0.9178 | **17.2 of 768** | **0/15** |
| `ARCH.MM768` | max-min over the 29 readers | 0.9137 | **39.4 of 768** | **0/15** |
| `ARCH.W768EV` | evidence-PCA (raw moment) | 0.9122 | 51.4 | **1/15** |
| `ARCH.W768AW100` | case-weighted, beta 1.0 | 0.8967 | 96.7 | **0/15** |
| `ARCH.W768RG` | reader-energy per unit raw | 0.4780 | 230.5 | **0/15** |

Below ~0.90 every child measured is dead. **Above it the retained fraction stops
predicting**: 0.9178 reads 0/15 and 0.9193 reads 3/15, so a difference of 0.0015
in energy and **seventeen of 768 directions** is worth three cases.

**The currency is what a reader consumes, not what the stream carries.** The
evidence-PCA basis retains **more** raw energy than reader-PCA (0.9636 vs
0.9522) and **less** read energy (0.9122 vs 0.9193), and scores 1/15 against
3/15. The two candidate currencies point in opposite directions on that child
and the read one is right.

**Basis invariance is total, and six objective tests have now failed.** Moving
the objective toward the case-implicated rows (two betas), toward reader energy
per unit of raw energy, onto the raw moment, and onto the **worst of the 29
readers** at two widths all lose. The max-min arm is the most informative,
because it succeeds at its own objective and fails anyway: `ARCH.MM768` lifts
the worst endpoint's retained read energy from **0.8801 to 0.8997**, cuts the
endpoints under the 0.90 floor from **four to one**, reconstructs the parent's
own logits **better** than the living child (final-norm relative SSE 0.1522
against 0.1752) — and reads 0/15 with every lane an "I am a" loop.

**And it loses at the width where there was something to lose.** Wave 21's one
defensible objection to that result was that 768 has three cases to give up and
39 of 768 directions moved. `ARCH.MM896` is the same objective at the living
width, against the 4/15 baseline and the 3/15 λ-split child at the **identical
197,311,616 parameters**:

| at width 896 | plain reader-PCA | reader-maxmin |
|---|---:|---:|
| worst endpoint, read, held-out | 0.9440 | **0.9530** |
| spread across 29 readers, read, **fit** | 0.0231 | **0.00055** |
| pooled read, held-out | **0.9627** | 0.9603 |
| directions exchanged | 0 | **24.69 of 896** |
| final-norm relative logit SSE | 0.1752 (`W896`) | **0.1513** |
| lambada / perplexity | 0.1450 / 638.72 | 0.0550 / 1880.53 |
| review | **4/15** | **2/15** |

The fit-side objective is met almost exactly — twenty-nine readers retained to
within **0.00055** of each other, against a spread of 0.0231 for the basis it
replaces — and it costs two review cases, including the summarisation lane that
**every other 896 child ever built holds**. Equalising the readers is not what
keeps a child alive at either width.

**A related fact worth its own line: the pooled number hides readers.** The
living 3/15 child at 768 has **four of its 29 readers below the 0.90 floor**
(`B01.ffn` at 0.8801), and the badly-retained endpoints are the same ones at
every width — blocks 0 to 3, with the argmin at `B01.ffn` in all **seven** sealed
reader-PCA bases now priced. Whatever makes those readers survivable, it is not
their retained energy: the one child built to fix exactly that, at the width
where it had room to matter, gave up two cases for it.

**And a reweighting cannot help, measured before it was built.** Under the plain
basis the case-implicated rows are retained at the pooled rate — retrieval rows
at 0.9265 against a pooled 0.9193, i.e. *better* — so the hypothesis that the
basis discards case-relevant directions is false, and one capture pass said so.

### 2.10 The census would have picked the wrong child, in both directions

* **By retained case energy**, `W768AW25` is the best basis in wave 20. By review
  it is the worst of the four that are not obviously broken.
* **By per-site write cosine**, wave 14 located a cliff between 0.93 and 0.86 and
  every wave after read per-site quality as predictive. On the width axis it does
  not move at all across a 4/15-to-0/15 collapse.
* **By minimum per-endpoint read energy**, `MM768` is strictly better than the
  living child on the quantity the law is stated in. It is 0/15.
* **By the excision census**, block 13 is the best single at 13/15 — and it is
  the block whose composition with `intermediate_size` 1280 reads 1/15 while the
  *worse* single composes better.

### 2.11 Review versus benchmark: the divergence runs both ways

| case | review says | the distribution says |
|---|---|---|
| `ARCH.W768AW25` (wave 20) | 0/15, worst of four | **best** of the four width children on arc_easy, hellaswag, piqa and winogrande |
| `ARCH.W896X13` (wave 19) | 1/15, worst but one | **best** `arc_easy` in the wave, above the child it is built from |
| `ARCH.W896LS` (wave 21) | 3/15, one case **worse** than its control | better on **six of seven** columns including both perplexities |
| `PROBE.CBT-L12` (wave 20) | reads near-parent on all sixteen lanes | lambada **0.4600 → 0.1800**, its perplexity 24.39 → 134.66 |
| `ARCH.MM768` (wave 21) | 0/15 | agrees: lambada **0.0000**, perplexity **23,120**, the worst row in the campaign |

**A frozen review at one site is not a sensitive instrument** — CBT touches one
tensor in a fourteen-block model and reads near-parent while its benchmark
collapses — and **the accuracy columns are not a behaviour measurement** — a
230M parent is only 0.045 above chance on `hellaswag` to begin with. Where the
two disagree, this campaign gates on the review, and every disagreement is
printed rather than reconciled.

**Perplexity is the benchmark column that ranks these children**, and it does so
monotonically and steeply: 24 → 230 → 370 → 528 → 857 on lambada while storage
falls 1.00 → 0.70.

### 2.12 Per-axis capability retention, and review one-dimensionality

From 63 sealed per-case readings across waves 15–19:

* **The review is 89.6% a ladder.** A single strict difficulty order predicts
  **0.8963** of the (child, case) cells (Guttman coefficient). The 15-case score
  is mostly one number wide, which is why per-child scores compare at all.
* **The difficulty order**, campaign-wide hold rate: summarization-priorities
  **0.44**, uncertainty-insufficient-data 0.43, writing-concise-email 0.41,
  instruction-three-actions 0.35, context-retrieval 0.32, planning-dependencies
  0.27, conversation-natural-greeting 0.25, creative-continuity 0.21,
  comparison-ram-rom 0.14, debugging-off-by-one 0.13, multi-turn-correction 0.10,
  arithmetic-multi-step 0.08, coding-deduplicate 0.05, logic-ordering 0.03,
  explanation-seasons **0.02**.
* **The tenth that is not the ladder is per-axis and legible.** The head merge
  holds debugging-off-by-one at **0.43** against a campaign 0.13; excision holds
  instruction-three-actions at **0.59** against 0.35; the width fold holds
  summarisation (0.50, and 1.00 in both compositions) and takes
  **context-retrieval and planning-dependencies to zero**.
* **And those two zeros are recoverable, at a price.** Choosing the width
  writers' regularisation on an internal split of the fit rows recovers
  `context-retrieval` at 768, `planning-dependencies` at 896 and inside the
  896+excision composition — the same capability every time — for **0** cases at
  768, **−1** at 896, **−1** at the cliff and **+1** inside the composition.
  **When reporting a fold, report which cases it keeps**, because the score alone
  hides this entirely.

### 2.13 Repair, and where it has purchase

ADDENDUM 10's corrective routes are additive sparse maps from `mlp.input` into
`mlp.down`, solved in closed form against a capture taken *through* the composed
child. On `COMP.MA.MXO4`'s fold set: review **5/15 → 7/15**, drift 0.540899 →
0.517480, cost **2,101,252 stored parameters charged in full**. On the shopped
child: **6/15 → 7/15** for 1,575,939 parameters.

**Its purchase is local and collapses with depth.** Out of sample the explained
fraction of the block gap decays 28.0% at L05, 10.4% at L06, 10.0% at L07 and
**0.0% at L13**. On an eleven-lane child, three routes net **+0.000092 drift —
worse than the unrepaired child** — for 1.58M added parameters. Repair works
where a fold's own error is still linear in that block's input, and at the end of
a deep stack it is not.

### 2.14 The mechanism results that are load-bearing and negative

* **Locality (ALS, wave 11).** Re-solving every component against the *composed
  child's own* activations made the composition **worse**. The fold's error is
  local to its block; the composition's error is not a sum of local errors a
  local re-solve can reach.
* **Sharing falsified (wave 11).** Parameter sharing across blocks does not pay
  on this model.
* **Interface × body multiplies (wave 10).** The token interface's cost and the
  body's compose multiplicatively: `MXN2`'s interface is 10/15 alone and costs
  +0.040485 drift installed on top of `MXO3`'s body.
* **The min-drift variant is per site (wave 12).** `T5` beats `T8` at L13 and
  loses at L07. A pooled per-site median ranks nothing.
* **Head merging is expressible and expensive (wave 13).** The merge is exact
  where exactness is checkable (identity gate 6.7e-16 to 8.7e-15), stores
  0.239583x per attention block, and costs **8–13× more drift per parameter**
  than the MLP lanes it competes with, because the heads are not rotations of a
  few shared bases: held-out write relL2 of the merged O map is 0.35–0.56 and
  every kv partition found is 7 heads plus 1 outlier.
* **Block-Toeplitz tied ridge loses its own head-to-head (wave 20).** The first
  family offered to the rank-truncation graves that does not truncate stores
  *fewer* numbers (4.27x vs 4.00x), is **full rank**, and fits **worse**
  (held-out cos 0.9108 vs 0.9434) than the sealed rank-128 twin at the same site.
  Channel space has no translation structure.
* **Verbatim coverage beats reconstruction, in gated feature space only (wave
  14).** A matched-storage control gives the coreset **40 of 42** site-levels on
  the SwiGLU write and the low-rank basis **12 of 15** on the conv write.

---

## 3. Corrections of record — every claim the campaign withdrew

1. **A9 storage accounting (`8d59c4d2`).** Route storage counted coefficient
   values and not the indices that route them. A table nobody can index is not
   storage. Fixed in `_t5_route_storage` and `_c01_token_route_storage`.
2. **The block rule (wave 7).** "Two folds in one block" was the hypothesis; the
   conflict is **two folds on opposite sides of one block's operator/FFN
   boundary**. Two MLP folds in one block cost nothing. Rewritten on the lane
   axis and machine-enforced.
3. **C05 "toxicity" (wave 7).** The short-conv writer was believed to poison a
   block. It folds at **13/15 at S8**, one case above the same-process parent;
   the earlier reading was a block-collision artifact.
4. **Token match as a gate (ADDENDUM 6).** Retired to a diagnostic column after
   a child matched 0.551 and read 12/15.
5. **The per-child verdict for a lane-scoped failure (ADDENDUM 9, wave 12).**
   `COMP.MA.MXO3` printed `dead` in the calculator and `alive-degraded` in the
   note. Fixed by sealing a **per-lane** reading with four refusals, including
   "a review that names no lanes seals `unstated`".
6. **`NOTES-MX-P` §8's head-merge arithmetic (wave 13).** `g=3` was priced at
   0.21875x by evaluating its kv term at `g/2 = 1.5`. The implementable config
   stores **0.239583x** — 8.7% higher. The verdict survived; the number did not.
7. **"Impaired" is the calculator's floor band (wave 13).** `behavior_verdict_for`
   returns `impaired` for *any* non-degenerate score below 8/15, so a per-lane
   `impaired` at 4/15 and one at 7/15 print the same word. **A per-lane verdict
   must always be quoted with its score.**
8. **"0.5x is arithmetically out of reach" (waves 11–12) — withdrawn.** An
   extrapolation of an R²=0.62 heuristic past its fitted range, used as a
   conclusion. Two of its three premises are measured false: the lane price does
   **not** run away (the seventh lane is the cheapest per parameter ever
   measured), and its saturation story was inferred from a trace rather than
   measured.
9. **"Routes are not the road" (wave 12) — reframed.** The 355× efficiency ratio
   is real and retained as a diagnostic. It was used as a kill decision; the
   comparison that would justify a kill — against the cost of the distillation it
   replaces — was never run.
10. **The 0.4666x arithmetic (wave 14) — withdrawn (wave 15).** Two errors: the
    three bites are not additive (the excised block's FFN is priced at parent
    width in one term and at 640 in the other), and the tied embedding is
    **67,108,864 parameters = 0.2922x** and is untouched by every axis in the R3
    standard set at that time.
11. **The frontier three waves quoted was not the frontier (wave 19).** Waves
    16, 17 and 18 named `ARCH.W1536X13` (0.769439x, 4/15). Sorted by net ratio
    over the sealed rows, `ARCH.CL896` (0.688431x, **4/15**, wave 15) is 8.1
    points deeper at the same score and had been sealed the whole time. Wave 15
    held it to a 4x-class eligibility rule that wave 16 then did not apply to the
    child it promoted. Every table in §1 is now generated from the ledger.
12. **"The embedding wall is arithmetic" (wave 18) — withdrawn (wave 19).** One
    standard axis does reach the tied embedding. `ARCH.W512` loads, the loader
    counts 106,458,368 parameters, and the whole remaining problem is
    behavioural.
13. **"Retained read energy predicts behaviour" (wave 19) — bounded (wave 20).**
    It is a **ceiling**. Below ~0.90 everything is dead; above it, two subspaces
    with energies differing by 0.0015 read 3/15 and 0/15.
14. **"The width fold kills cases by discarding their directions" (wave 20's own
    hypothesis) — falsified by its own free measurement.** The case-implicated
    rows are retained at the pooled rate, retrieval rows *better* than average.
15. **"The cliff is a step" (wave 19) — withdrawn (wave 21).** The rung inside
    the bracket reads 1/15; the ladder is 4 → 3 → 1 → 0 and graded.
16. **Wave 19's committed human-review renderings were empty files.** The
    packager called the dossier reader through a directory that holds only shell
    scripts and `2>/dev/null` hid it. Regenerated and committed in wave 20.
17. **The campaign's frontier child had no committed dossier (wave 21).** Wave
    15 built its human-review package and left it on the server; `ARCH.CL896`'s
    text was quoted by four subsequent waves and shipped by none. The 74-file
    package is now in `wave15/human-review/`.

---

## 4. What transfers off this model

Full derivations in `docs/FOLDING_SCALING_THEORY_2026-08-12.md`; this is the
summary a 7B/70B attempt should start from, with the confidence of each claim
stated.

### 4.1 The interface rule, and why the biggest win here is the smallest at scale

The token interface (`embed_tokens`, tied to `lm_head`) is **29.2%** of
LFM2.5-230M and folds to 0.25x while reading 10/15 — the single biggest
fold-only win in the campaign, and **a shrinking prize**:

```
interface share ≈ (V·H) / (V·H + L·c·H²)          c ≈ 12–16 for a standard block
LFM2.5-230M   V=65,536  H=1,024  L=14   ->  29.2%     measured
Llama-3-8B    V=128,256 H=4,096  L=32   ->  ~13%      projected
Llama-3-70B   V=128,256 H=8,192  L=80   ->  ~4%       projected
```

**High confidence** (it is arithmetic). What transfers is the *method*: a
verbatim head selected by token frequency plus a sparse-route tail, with the
route indices counted in the storage — a sparse tail beat a low-rank tail
everywhere it was measured.

### 4.2 Attention share moves the other way, and the merge is the measurement to repeat

Attention is **11.61%** of this model's body (six attention blocks in fourteen
layers, GQA 16/8). On a standard dense transformer it is ~⅓, so a mechanism
worth 1% here could be worth 8% at 70B. **The measurement to repeat first is not
the merged child — it is the cluster objective**
(`solve.query_family.objective_relative` in any wave-13 receipt): if a 70B
model's heads cluster tightly, M5's price falls and it becomes the largest
fold-only prize at scale; if they scatter the way LFM2's do, it does not.
**Medium confidence**, and the sign of the answer is architecture-dependent by
construction.

### 4.3 The residual-width axis, and what to measure before spending a GPU on it

`hidden_size` is the only standard axis that reaches a tied embedding, and it is
the one whose behaviour is predictable enough to plan around — **up to a
ceiling**. The transferable procedure, in order:

1. Capture the residual's second moment at **every endpoint that reads the
   stream**, in both raw and reader (post-norm, gain-scaled) form. Costs one
   pass.
2. Publish the cumulative **reader** spectrum. If it has a knee, this axis is
   cheap on that model; on LFM2.5-230M it is flat and there was never a subspace
   to find.
3. Take the top-k eigenvectors of the **pooled reader moment** and nothing else.
   Five alternative objectives have been tried on this model and all five lose.
4. Expect a floor near **0.90 of retained read energy** and expect the retained
   fraction to stop predicting above it. Re-derive the constant; do not import
   0.90.

**Medium-to-high confidence on the shape, low confidence on the constant.** The
floor is one model's number. The *structure* — a ceiling rather than a predictor,
basis-invariance, and per-site install quality being irrelevant — rests on eight
sealed children at one parameter count and is the part worth assuming.

### 4.4 Composition, which is the finding most likely to survive scale

**One structural axis is survivable; two is fatal.** This held on every pair of
axes tried (§2.3), including on an axis introduced two waves after the rule was
written. A large-model attempt should plan for **one** fold class and a deep
ladder within it, not a composition of classes. **High confidence** — it is the
most-replicated result in the campaign.

### 4.5 Drift-budget methodology, as a heuristic only

1. Seal ≥ 20 composed rungs with both drift and a frozen review score.
2. Regress review on drift; publish the R² and the residuals.
3. Use the fit to **order** candidate lanes, and shop them with a fixed-base
   probe (~40 s per candidate) rather than taking them in depth order.
4. **Then build and review.** The `MXQ1`/`MXQ2` pair is the standing
   demonstration that the order is not the verdict.

### 4.6 Per-class safety, transferable shape

MLP mass folds; the last attention block's key projection folds; value/output
projections and stacked attention folds do not; head merging is expressible but
expensive; mid-depth sites are safest; one component folds *better* at 8x than at
4x. **Re-measure the ladder per component and never assume it is monotone.**

---

## 5. Where the frontier sits today, stated four ways

Each statement carries the eligibility rule that produces it, because the
campaign has already lost three waves to a frontier claim quoted without one.

* **Deepest sealed child with a non-zero score, no eligibility rule:**
  `ARCH.V16CL896`, **0.469305x at 1/15**, standard shape, since wave 23 — under
  half the parent. It is one marginal case (a summarisation lane that echoes its
  prompt and then names the decision, the risk and the mitigation) with six of
  sixteen lanes at the cap, and it is reported as that and not as a model.
* **Deepest sealed child at four or more cases:** `ARCH.CL896`, **0.688431x at
  4/15**, standard shape, since wave 15. Unmoved.
* **Deepest STANDARD-SHAPE child at eight or more cases:** `ARCH.V16`,
  **0.780874x at 9/15** (wave 23). The previous holder was `ARCH.X13A` at
  0.947479x — **seventeen points of ratio**.
* **Deepest child that reads at the parent's own band (`alive-parent-band`):**
  `COMP.MA.MXN2`, **0.780518x at 10/15**, parent-shaped, one component. `V16` is
  at the same ratio, one case below it, on the standard column.
* **Deepest 4x-class rung with a non-zero score:** there has never been one.
  Every 4x-class rung in the campaign is 0/15.
* **Deepest child of any score:** `ARCH.W512`, **0.463481x**, standard shape,
  **0/15** — under half the parent, and not a model. `V16CL896` is five
  thousandths shallower and holds one case; that gap is the whole of what
  "coherent at half the parent" is worth so far.

**0.5x is crossed with a case, for the first time (wave 23).** 0.5x is
114,846,592 parameters; `ARCH.V16CL896` is **107,796,224** and holds one of
fifteen. The distance from the deepest child at four or more cases is
unchanged: `CL896` is 158,127,872, i.e. 43.3M above 0.5x.

**Fold-only did not produce a COHERENT 0.5x child on this model.** Four
independent children at or below 0.5x have now been read, built by three
different mechanisms in four different orders, with and without repair. Three
read 0/15. The fourth, `ARCH.V16CL896`, holds one case out of fifteen with six
of sixteen lanes at the token cap — which moves the line from "nothing survives
here" to "almost nothing survives here", and is a real move that is not
coherence.

**And the measured reason is not a runaway price.** The per-lane marginal price
does not increase with depth — the seventh lane is the cheapest per parameter
ever measured, on a six-lane base — and total drift accumulates roughly linearly
at about 0.042 per lane. Nine shopped lanes sit at drift 0.706, which is 1.5× the
deepest coherent drift ever recorded, while removing only 23% of the parent.
**The coherence budget is small and drift accumulates steadily against it.** That
is an argument about a budget, not about a price curve, and it licenses an
ordering (SwiGLU lanes first) and nothing more.

---

## 6. What remains unbuilt, and what is gated on what

Nothing in this section is a closure. Each item says whether it is unbuilt for
want of time, blocked on a measured precondition, or refused by a rule.

**Wave 22 built three of the four items this section listed as cheap, and the
fourth was gated out by its own rule:**

* **The rung between 736 and 704** is built. `ARCH.W720` reads **0/15** at
  0.9015 retained read energy, so the last interval that could hide a step is
  **736 → 720**, sixteen dimensions wide.
* **The max-min basis at 896** is built. `ARCH.MM896` reads **2/15** against its
  4/15 and 3/15 controls at the identical parameter count, having met its own
  objective almost exactly (§2.4).
* **The writers' λ split composed with the head merge** is built.
  `ARCH.W896LSKV6JQ` reads **1/15** against `W896KV6JS`'s 2/15 at the identical
  193,182,848 parameters — the two case-recovery mechanisms do not add, and the
  capability the split exists to recover (`planning-dependencies`) does not
  survive the merge, though it survives an excision.
* **The `X07 + KV6JQ + 896` triple was NOT built**, by the pre-registered gate:
  wave 22 declared it would build the triple only if the two-mechanism child
  held at least three cases, and it held one. A triple on a collapsed two-axis
  child measures the collapse.

**Unbuilt, and cheap:**

* **One rung between 736 and 720** — sixteen dimensions and 0.0060 of retained
  read energy, the last interval on the width slope that could hide a step.
* **The max-min basis at 720**, priced free by the same census (worst endpoint
  0.8545 → 0.8783, 43.6 of 720 directions exchanged) and not built, because the
  plain arm at that width is already 0/15 and there is nothing to lose.
* **CBT's greedy evidence-weighted merge schedule.** At 4x the 15 base Toeplitz
  classes already clear the budget, so the only adaptive part of that class has
  never run. After its head-to-head loss there is no reason to fund it.

**Never priced behaviourally — four structural fold classes:**

* **CUR / interpolative decomposition** — factors that are *actual rows and
  columns of the parent*, so the basis is not a new dense object; storage is `k`
  column ids plus a `k×n` coefficient block, and the ids are integers.
* **Nearest-Kronecker-product (Van Loan) — PRICED AND BURIED (wave 23).** This
  entry called it "the highest-leverage unplayed class by storage arithmetic
  alone", and that was true only in the sense that nobody had looked. Wave 23's
  `P3` census ran all 42 SwiGLU tensors through four admissible grids each:
  **the top Kronecker term holds 0.25% of the energy** (median 0.00253 across the
  stack), 196 of 1,280 terms retain **0.3734** at the 4x bound, and the class
  loses its own head-to-head to plain rank truncation at **38 of 42 sites**
  (median 0.4341) and to a Monarch blockwise-SVD control at **all 42** (median
  0.5050, and at 0.35x storage rather than 0.25x). Rearrangement rank and matrix
  rank are indeed independent — and on this model **both** spectra are flat. The
  cell was not built; a priced grave is a deliverable and an unpriced backlog is
  not. `wave23/prereads/P3-nkp-rearrangement.json`. What the census does **not**
  do is the evidence-weighted selection, which needs the sealed ffn-input moment
  and therefore a capture.
* **Monarch / block-diagonal + low-rank** — permutation-sandwiched block-diagonal
  factors, expressly designed as a drop-in for a dense linear layer, and the
  closest structural relative of what is known to fold here (MLP mass).
* **Tensor-train / tensor-ring on the reshaped weight** — storage linear rather
  than quadratic in the mode widths.

Each is closed-form, gradient-free, and expressible in the existing kernel
contract (parent-shaped installed tensors, a published storage model, an identity
gate). **The fold-only frontier is open until they are priced behaviourally.**

**Blocked on a measured precondition, not on time:**

* **XGRAM retrospective calibration and the gap-4 attribution.** Both consume the
  terminal-transfer surrogate G, whose published held-out R² clears the
  pre-registered 0.5 floor at **three of thirteen layers** (L10 0.615, L11 0.706,
  L12 0.874). Restricted to L10–L12 or it is pricing noise.
* **F-LANDMARK has no distribution measurement** and cannot get one without a
  loader that accepts a custom module. It reads near-parent on a frozen review at
  one site, and §2.11 is why that is not enough to fund a class.

**Refused by a rule, and the rule is Cyril's:**

* **Quantization and codebook/precision reduction of any kind.** Not folding.
  Applied after real structural folding is proven; storage in this campaign is
  counted in parameters and never in bytes.
* **Any gate on token match, drift, efficiency, per-site cosine, or retained
  energy.** All five have been measured against the review and all five fail
  (§2.1, §2.10).

**The designed next stage, built and preserved:** `/mnt/ai_data/neural-forge-stage2/`
— a trainer, 26 corpus shards, and admission gates, with its design in
`docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`. Its thesis is that **a fold
is an initializer, not a product**, and the fold-only campaign's contribution to
it is a *ranked family of initializations with sealed receipts*, so stage 2 can
ask which initialization distils fastest rather than guessing one.

---

## 7. Where everything is

| artifact | location |
|---|---|
| ledger (662 rows) | `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`, copy in `reports/matrix-campaign-20260810/ledger.jsonl` |
| rendered ledger | `reports/matrix-campaign-20260810/LEDGER.md` (regenerate, never hand-edit) |
| the generated campaign table | `reports/matrix-campaign-20260810/wave23/campaign-frontier-table.json` |
| the one-page scorecard | `reports/matrix-campaign-20260810/CAMPAIGN_SCORECARD.md` |
| wave notes | `NOTES-MX-{A..H,X,Y,Z}.md` at the repository root and `reports/matrix-campaign-20260810/wave{5..19}/NOTES-MX-{I..W}.md` |
| design + addenda 1–15 | `docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` |
| scaling theory | `docs/FOLDING_SCALING_THEORY_2026-08-12.md` |
| human-review packages | `reports/matrix-campaign-20260810/wave{12..22}/human-review/` |
| frontier child tensors | `/mnt/ai_data/neural-forge-matrix/wave{10..22}/children/` and `.../cells/<CHILD>/` |
| benchmarks | `/mnt/ai_data/neural-forge-matrix/wave{8,9,11,12,13,16..22}/bench/` |
| stage-2 estate | `/mnt/ai_data/neural-forge-stage2/` |
| M5 kernel | `src/neural_forge/matrix_technique_head_merge.py` + tests |
| the vocabulary builder | `reports/matrix-campaign-20260810/run/mx23/mx23_vocab.py` (wave 23; a row gather, a merge-tree closure and a tokenizer emit, and it solves nothing) |
| the deployment protocol | `reports/matrix-campaign-20260810/run/mx23/deploy.py` (wave 23; it had only ever lived in a shell history) |
| the width/basis builder | `reports/matrix-campaign-20260810/run/mx23/mx23_build.py` (waves 19–22 lineage; **ten** code lines of body differ from `mx21_build.py` — nine renamed at `5e17bb58`, one added at `70a2498d` — and the solver is byte-identical) |
