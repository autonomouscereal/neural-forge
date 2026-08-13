# reports/matrix-campaign-20260810

Local artifacts for the stage-1 matrix campaign
(`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`). Written by the harness lane;
narrative in `NOTES-MX-A.md` at the repo root.

| path | what it is |
|---|---|
| `ledger.jsonl` | the append-only campaign ledger, one canonical-JSON row per sealed receipt (`neural-forge-lfm2-matrix-cell-ledger-v1`) |
| `LEDGER.md` | rendered from `ledger.jsonl` by `experiments/render_matrix_ledger.py` — regenerate, never hand-edit |
| `cells/CTRL.copy_exact/cell-receipt.json` | cell 0a as sealed by the runner, `decode_smoke_review: pending` |
| `cells/CTRL.copy_exact/cell-receipt-reviewed.json` | the same cell after the frozen decode-smoke review, sealed as a new instance, status `pass` |
| `cells/CTRL.copy_exact/control.json` | the behavioural control every later cell is scored against: probe token ids and decode-smoke text |
| `cells/CTRL.copy_exact/cell-diagnostics.json` | non-receipt telemetry: registry inventory, technique-module import results, terminal metrics, timing |
| `cells/CAL.C01.T5.X4/cell-receipt.json` | cell 0b, status `infra` — calibration failed, campaign stopped |
| `cells/CAL.C01.T5.X4/route-numeric-dtype-contract.json` | the reproduced route's dtype contract, `0f52a2d4…`, against A9's sealed `243a4e48…` |
| `cells/CAL.C01.T5.X4/route-identity-excerpt.md` | the reproduced codebook artifact digest and the terminal trace-cache failure, lifted from the route progress log |
| `cells/<cell-id>/wave2/` | a wave-2 re-run of a cell id wave 1 already sealed (four void cells plus the C13.T5 rescue probe).  The ledger keeps both instances by design; the wave-1 files beside it are untouched. |
| `run/mxf-*`, `run/run-mxf-batch.sh` | wave-2 (MX-F) knob files, batch driver, review spec and the summarize/smoke/review-seal drivers |


Wave-2 receipts carry three fields wave-1 receipts do not:
`code_identity.compute_architecture`, `calibration.applicable` and
`boundary.local_solve_applicable`.  Wave-1 receipts predate them, are preserved
byte-for-byte, and re-verify their seals but no longer round-trip through
`validate_receipt`.  Narrative in `NOTES-MX-F.md` at the repo root.

The server side is under `/mnt/ai_data/neural-forge-matrix/` (receipts, ledger,
`backup/` inventories) with the route's own working directory on local disk at
`/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/`.

Regenerate the report with:

```
python experiments/render_matrix_ledger.py \
  --ledger reports/matrix-campaign-20260810/ledger.jsonl \
  --out    reports/matrix-campaign-20260810/LEDGER.md
```

## Start here — the two consolidated documents

| path | what it is |
|---|---|
| `CAMPAIGN_SCORECARD.md` | one page: the top fifteen children by intelligence-per-compression, each with its net ratio, its review, the one-line character of its failure and its `READ-` dossier path. This is the scoring page |
| `../../docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` | the consolidating report for waves 1–21: **all 108** whole-model children that hold at least one review case, sorted by net ratio with both readings and every benchmark column; every law with its evidence; seventeen corrections of record; the scale-transfer statements with their confidence; and what remains unbuilt and why. It does **not** declare the campaign closed |
| `wave21/campaign-frontier-table.json` | the artifact both of the above are generated from — a function of the ledger, the per-wave child indexes and the sealed `lm-eval` output, built by `run/mx21/mx21_campaign_table.py` and rendered by `run/mx21/mx21_report_tables.py` |

## wave21/ — the cliff has a slope, and the worst reader is not the variable

`NOTES-MX-Y.md` at the repo root is the narrative. The rest is raw:

| path | what it is |
|---|---|
| `wave21/basis/endpoint-retention.json` | the free measurement the wave opens with: the retained **read** energy of every sealed basis **per endpoint**, fit and held-out, which no census had ever printed — the census records the per-endpoint moment for the raw residual only — plus what the max-min iteration achieves at 896, 768 and 736 |
| `wave21/wave21-child-index.json` | every wave-21 child read off its sealed ARCH receipt and its sealed ADDENDUM 6 verdict |
| `wave21/wave21-child-comparison.json` | one row per child with the basis, the retained read energy, the per-site write cosine, the drift and the review |
| `wave21/campaign-frontier-table.json` | the campaign-wide table (see above); regenerated in this wave for the first time |
| `wave21/scorecard-characters.json` | the prose column of the scorecard, kept apart from the generated numbers and keyed by cell id, so a child with no line written prints `**UNWRITTEN**` |
| `wave21/human-review/` | 45 files: every dossier, every child and parent response rendered verbatim, every run receipt, verdict receipt, per-case reading and build receipt, both new basis censuses, the endpoint retention report, the benchmark table and every table above |
| `wave21/ledger-merge-report.json` | the exactly-once merge of 10 shard rows into the campaign ledger (636 → 646): 5 ARCH run rows and 5 ADDENDUM 6 verdict rows, no superseded rows |
| `wave21/children-retention-manifest-wave21.json` | all five children kept, each with the controlled comparison it is an arm of |
| `wave21/deployments-prune-manifest-wave21.json` | the deployment pruned to hold current+1 |
| `run/mx21/` | the wave-21 builder (adds `--basis reader-maxmin`, `--mode endpoints`, and the per-endpoint **reader** retention the census was missing), its standalone algebra self-test (24 checks), the campaign-table and report-table generators, the drivers, the seal verifier, the ledger merge and the dossier readers |
| `wave15/human-review/` | **new in wave 21, four waves late.** The campaign's own frontier child, `ARCH.CL896`, had no dossier rendering anywhere in the repository; wave 15 built its 74-file package and left it on the server |

Wave 21's one-line result: **the retained-energy law names a necessary condition
and not the thing to optimise.** A basis that maximises the *minimum*
per-endpoint retained read energy — lifting the worst reader from 0.8801 to
0.8997, cutting the readers under the 0.90 floor from four to one, and
reconstructing the parent's own logits better than the living child — reads
**0/15** against that child's 3/15; the 0.90–0.92 band is measured and the cliff
is a **slope** (4 → 3 → 1 → 0 at 896/768/736/704), leaving 32 dimensions that
could still hide a step; and the writers' λ split is a **case trade** whose sign
depends on what the child has left to lose (0 at 768, −1 at 896, −1 at the cliff,
**+1** inside a composition, buying the same capability every time).

## wave20/ — the basis objective moved, and it was the wrong variable

`NOTES-MX-X.md` at the repo root is the narrative. The rest is raw:

| path | what it is |
|---|---|
| `wave20/wave20-child-index.json` | every wave-20 child read off its sealed ARCH receipt and its sealed ADDENDUM 6 verdict |
| `wave20/wave20-child-comparison.json` | the wave's central table: one row per child with the basis, the beta, the retained read energy, the per-selector retention, the drift and the review score — five children at 768 are the same 165,978,624 parameters and differ only in the basis or the solve |
| `wave20/human-review/` | 68 files: every dossier, every child and parent response rendered verbatim (`READ-*.txt`, `READ-PARENT-*.txt`, `READ-PROBE-*.txt`), every run receipt, verdict receipt, per-case reading and build receipt, the five new basis censuses, the free retention report, the subspace comparison, both probe receipts and dossiers, and the benchmark table |
| `wave20/human-review/basis-selector-retention-768.json` | the free measurement the wave opens with, and the one that falsified its own hypothesis: what every sealed basis retains of the pooled stream, of the retrieval rows and of the procedural rows, on a beta grid, with no build at all |
| `wave20/human-review/basis-subspace-768.json` | principal angles between every pair of sealed bases — the measurement that makes "same energy, different subspace" a number (17.2 of 768 directions exchanged at beta 0.25, 230.5 for the generalised basis) |
| `wave20/human-review/probe-CBT-L12-receipt.json`, `probe-CBT-L02-receipt.json` | the block-Toeplitz tied ridge at `attn_o`, full rank at 4.266x storage, and its matched-site comparison against the sealed rank-128 twin |
| `wave20/human-review/probe-LANDMARK-L12-receipt.json` | the verbatim-state landmark memory: six of eight KV groups converted at L12, the selection split between QR pivots and attended mass, and the beta grid that converged on the parent's own attention scaling |
| `wave20/ledger-merge-report.json` | the exactly-once merge of 16 shard rows into the campaign ledger (620 → 636): 8 ARCH run rows and 8 ADDENDUM 6 verdict rows, no superseded rows |
| `wave20/children-retention-manifest-wave20.json` | which of the 8 built children were kept and which were deleted, each with its sealed tensor digest |
| `wave20/deployments-prune-manifest-wave20.json` | the deployment pruned to hold current+1 |
| `run/mx20/` | the wave-20 builder (adds `--basis attribution-weighted`, `--basis reader-generalized`, `--width-lambda-split`, `--mode selectors`), its standalone algebra self-test (17 checks), the two slate probes and their review driver, the subspace comparison, the comparison table, the drivers, the seal verifier, the ledger merge and the dossier readers |

Wave 20's one-line result: **the residual basis objective is a real degree of
freedom and every direction the wave moved it in lost.** At an identical
parameter count and retained read energy differing by 0.0015, exchanging
**seventeen of 768 directions** takes the width child from 3/15 to 0/15, so
retained energy is a **ceiling and not a predictor**; the energy that matters is
what a **reader** consumes, not what the stream carries (evidence-PCA retains
more raw and less read energy, and scores 1/15 against 3/15); and choosing the
width writers' λ on an internal fit split keeps the score at 3/15 while
recovering **context-retrieval**, the case the attribution artifact names as this
axis's signature zero.

## wave18/ — the distance law at three bodies, the λ overfit fixed, and the mask an excision took away

`wave18/NOTES-MX-V.md` is the narrative. The rest is raw:

| path | what it is |
|---|---|
| `wave18/wave18-child-index.json` | every wave-18 child read off its sealed ARCH receipt and its sealed ADDENDUM 6 verdict — net ratio, config diff, export class, axes, both ADDENDUM 9 readings, and whether the safety lane was chargeable |
| `wave18/human-review/` | 111 files: every dossier, every child and parent response rendered verbatim (`READ-*.txt`, `READ-PARENT-*.txt`), every run receipt, verdict receipt, per-case reading and build receipt, plus the gate-probe and mask-probe measurements |
| `wave18/probes/mask-probe.json` | the falsifying measurement for the wave-17 anomaly: excise block 13 and the module's attention mass on the future is 0.0000; excise 13 and 1 and it is 0.286–0.480; repair `config.layer_types` and it is 0.0000 again |
| `wave18/probes/GATE2-*-gate/` | the in-build gate probe per site — the module's own probabilities against the fp64 recomputation, the peakedness and top-two margin of both, the per-row centred logit regression, and the input-rounding leg that is a null and is labelled one |
| `wave18/probes/forensics-F1301.json` | the same comparison on the EXPORTED child, which agrees at the bf16 floor — the measurement that localised the defect to the live harness rather than to the checkpoint |
| `wave18/ledger-merge-report.json` | the exactly-once merge of 28 shard rows into the campaign ledger (578 → 606): 14 run rows and 14 verdict rows |
| `wave18/children-retention-manifest-wave18.json` | which of the 21 built children were kept and which were deleted, each with its sealed tensor digest |
| `wave18/deployments-prune-manifest-wave18.json` | the three deployments pruned to hold current+1 |
| `run/mx18/` | the wave-18 builder (adds `--kv-lambda-split`, `--kv-gate-probe` and the causal-mask repair), its standalone algebra self-test, the mask probe, the gate forensics, the drivers, the seal verifier, the ledger merge and the dossier readers |

Wave 18's one-line result: **the pair distance law does not survive to three
blocks** (widest triple 3/15 against the best pair's 7/15), **choosing the value
path's λ on an internal split of the fit rows is worth four review cases**
(3/15 → 7/15 at one parameter count), and the attention gate wave 17 could not
explain was refusing because an excision had shifted every later attention block
into a conv's slot in `config.layer_types`, putting **28–48% of the live model's
attention mass on tokens that had not happened yet**.

## wave17/ — the census closed, the pair distance law, and the merge's first real solve

`wave17/NOTES-MX-U.md` is the narrative. The rest is raw:

| path | what it is |
|---|---|
| `wave17/wave17-child-index.json` | every wave-17 child read off its sealed ARCH receipt and its sealed ADDENDUM 6 verdict — net ratio, config diff, export class, axes, both ADDENDUM 9 readings, and whether the safety lane was chargeable |
| `wave17/human-review/` | 148 files: every dossier, every child and parent response rendered verbatim (`READ-*.txt`, `READ-PARENT-*.txt`), every run receipt, verdict receipt, per-case reading and build receipt |
| `wave17/ledger-merge-report.json` | the exactly-once merge of 53 shard rows into the campaign ledger (525 → 578): 21 run rows, 21 verdict rows and 11 ADDENDUM 14 re-seals of wave-16 verdicts |
| `wave17/children-retention-manifest-wave17.json` | which of the 22 built children were kept and which were deleted, each with its sealed tensor digest |
| `wave17/deployments-prune-manifest-wave17.json` | the two deployments pruned to hold current+1 |
| `wave17/lane-fragility-wave17.json` | a free no-GPU probe the closed census made possible: how many of the thirteen single-block excisions each of the fifteen lanes survives |
| `wave17/probes/KV6J-v1-unfixed-gauge/` | the build receipt and log of the first joint merge, the one that landed on the reflection of its own prior — kept because the fix is only legible beside it |
| `run/mx17/` | the wave-17 builder (adds ADDENDUM 14 14.3's joint two-sided solve), its standalone algebra self-test, the drivers, the seal verifier, the ledger merge, the wave-16 re-sealer and the dossier readers |

Wave 17's one-line result: **the cost of removing two blocks is a distance law**
(2 apart reads 2–4 of 15, twelve apart reads 7, monotone across two anchors), the
completed census puts **conv block 0 at 0/15** where conv block 13 is 13/15, and
T7's joint relation solve takes the head merge from **5/15 to 8/15** while the
value-side alternation it ships with takes it to 3/15.

## wave16/ — the assembly wave, and the excision census that replaced it

`wave16/NOTES-MX-T.md` is the narrative. The rest is raw:

| path | what it is |
|---|---|
| `wave16/wave16-child-index.json` | every wave-16 child read off its sealed ARCH receipt and its sealed ADDENDUM 6 verdict — net ratio, config diff, export class, axes, both ADDENDUM 9 readings |
| `wave16/human-review/` | 147 files: every dossier, every child and parent response rendered verbatim (`READ-*.txt`, `READ-PARENT-*.txt`), every run receipt, verdict receipt, per-case reading and build receipt |
| `wave16/ledger-merge-report.json` | the exactly-once merge of 42 shard rows into the campaign ledger (483 → 525) |
| `wave16/children-retention-manifest-wave16.json` | which of the 21 built children were kept and which were deleted, each with its sealed tensor digest |
| `wave16/deployments-prune-manifest-wave16.json` | the deployment pruned to hold current+1 |
| `run/mx16/` | the wave-16 builder (adds the head-COUNT merge axis), the drivers, the seal verifier, the ledger merge, the dossier readers and the benchmark driver |

Wave 16's one-line result: the ordered succession ladder is dead at rung 0, and
the single-block excision census that replaced it found that **block 13 alone
reads 13/15 where block 5 alone reads 6/15 at the identical parameter count** —
the ranking wave 14's absorbed-fraction census had inverted.
