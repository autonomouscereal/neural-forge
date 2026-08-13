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
