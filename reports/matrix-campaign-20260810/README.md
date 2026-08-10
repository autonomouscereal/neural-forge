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

The server side is under `/mnt/ai_data/neural-forge-matrix/` (receipts, ledger,
`backup/` inventories) with the route's own working directory on local disk at
`/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/`.

Regenerate the report with:

```
python experiments/render_matrix_ledger.py \
  --ledger reports/matrix-campaign-20260810/ledger.jsonl \
  --out    reports/matrix-campaign-20260810/LEDGER.md
```
