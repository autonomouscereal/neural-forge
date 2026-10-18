# Chimera Omni campaign scorecard — `omni-2026-09`

Regenerated 2026-10-10T08:08:33.802Z from the sealed registry. **Nothing in this file is typed by hand; every number is copied from a sealed record.**

Ledger `/home/cereal/omni-campaign/ledger.jsonl` — 34 rows, sha256 `bdd6cc801fe05f7131151423bf8b7e6af03b67dca022fc885b5406a6bf457c81`.

| status | checkpoints |
|---|---:|
| candidate | 3 |
| partially_capable | 10 |
| **total** | **13** |

Every threshold below is a proposed decision line for Cyril. The only grade is measured intelligence read by a human.

## Stage: parent  (3)

| checkpoint | tag | parent | ratio | ratio from | params | sealed/15 | parent/15 | canon | A | B | bank2 | media | status | package |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| `PARENT-Q35-0.8B` | — | Q35-0.8B | 1.000000 | headers | 873,438,784 | — | — | — | — | — | — | 0/0 | candidate | — |
| `PARENT-Q35-4B` | — | Q35-4B | 1.000000 | headers | 4,659,865,088 | — | — | — | — | — | — | 0/0 | candidate | — |
| `PARENT-Q38-27BU` | — | Q38-27BU | 1.000000 | headers | 27,781,427,952 | — | — | — | — | — | — | 0/0 | candidate | — |

## Stage: fold  (10)

| checkpoint | tag | parent | ratio | ratio from | params | sealed/15 | parent/15 | canon | A | B | bank2 | media | status | package |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| `OC01-F-Q35-4B-X08` | X08 | Q35-4B | 0.949886 | headers | 4,426,341,696 | 15 | 15 | 15 | 15 | 12 | 5/5 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X08` |
| `OC01-F-Q35-4B-X16` | X16 | Q35-4B | 0.949886 | headers | 4,426,341,696 | 15 | 15 | 15 | 15 | 13 | 5/5 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X16` |
| `OC01-F-Q35-4B-X03` | X03 | Q35-4B | 0.951053 | headers | 4,431,780,864 | 15 | 15 | 15 | 15 | 13 | 5/5 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X03` |
| `OC01-F-Q35-4B-X15` | X15 | Q35-4B | 0.951053 | headers | 4,431,780,864 | 15 | 15 | 15 | 15 | 13 | 5/5 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X15` |
| `OC02-F-Q38-27BU-C2K4352` | C2K4352 | Q38-27BU | 0.522729 | headers | 14,522,158,320 | 8 | 15 | 8 | 8 | 1 | — | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc02/human-review/OC02-F-Q38-27BU-C2K4352` |
| `OC01-F-Q35-4B-X31` | X31 | Q35-4B | 0.951053 | headers | 4,431,780,864 | 12 | 15 | 12 | 12 | 10 | 4/4 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X31` |
| `OC01-F-Q35-4B-X00` | X00 | Q35-4B | 0.949886 | headers | 4,426,341,696 | 10 | 15 | 10 | 10 | 10 | 4/4 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-X00` |
| `OC01-F-Q35-4B-C2K2304` | C2K2304 | Q35-4B | 0.609583 | headers | 2,840,572,416 | 5 | 15 | 5 | 5 | 2 | 0/0 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-C2K2304` |
| `OC01-F-Q35-4B-G5D32` | G5D32 | Q35-4B | 0.811976 | headers | 3,783,698,688 | 5 | 15 | 5 | 5 | 5 | 0/0 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-G5D32` |
| `OC01-F-Q35-4B-C2K1152` | C2K1152 | Q35-4B | 0.548826 | headers | 2,557,456,896 | 1 | 15 | 1 | 1 | 0 | 0/0 | 0/0 | partially_capable | `/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-C2K1152` |

Legend: **ratio** = child parameters / parent parameters (storage counted in PARAMETERS); **ratio from** = `headers` when BOTH counts were read from the safetensors headers on disk, **asserted** when one of them was typed by the operator and therefore proves nothing. **sealed/15** = the sealed per-case useful-or-partial count on the frozen 15-case bank. **canon** = canonical three-box signal lanes (artifact misses count for nothing; only content misses count against). **A** = copy-backs count; **B** = copy-backs refused, the machine-verified intelligence floor. ★ = frontier/floor holder, never dropped. 🗑 = bytes deleted, evidence retained. Status **unmeasured** = zero signal but zero content misses too — every lane was an artifact, so nothing was measured and nothing may be deleted; re-run it.
