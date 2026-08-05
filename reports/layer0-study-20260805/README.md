# Layer-0 representability study — fetched results (2026-08-05)

Single-boundary refit study of the r12 exact-4x child's layer-0 short-conv
boundary, run against the same sealed evidence the r12 body fold consumed.
Verdict and analysis: `../../NOTES-W13B.md`.

## Layout

| path | contents |
|---|---|
| `remote/` | everything written under the server output root `/home/cereal/model-surgery-lab/artifacts/neural_forge_layer0_study_v1/`, minus the two 2.1 GB evidence-cache payloads |
| `remote/<run>/arm-*.json` | one sealed per-arm record (config + metrics + prior-correction reports) |
| `remote/<run>/evidence-inventory.json` | sealed input identities and row accounting for that process |
| `remote/<run>/run.log` | timestamped progress journal |
| `remote/evidence-cache-layer00{0,1}/binding.json` | the authenticated evidence-cache binding (payload digest, sealed chart digest, split) |
| `remote/clamp-forensics.txt` | Frobenius-norm decomposition of the r12 capsule/published layer-0 weights against the source-derived and chart priors |
| `remote/child-input-alignment.txt` | the r12 fold's own child replay vs `encode(parent residual.pre)` at layer 0 |
| `summary-table.txt`, `summary-rows.json` | the rendered arm table |
| `summarize_layer0_study.py` | renderer (`python summarize_layer0_study.py remote/*/`) |
| `nf-layer0-study.sh` | the server launcher |

`remote/shardE/arm-width_768_r12_solver.json` is misnamed: `nested_chart`
refuses widths below the sealed 816, so the arm ran at 816 in a second,
independent process. Its record carries `target_width: 816`. It reproduces
`r12_reproduction` to every printed digit and is kept as the determinism check,
not as a ladder rung.

## Runner

`experiments/run_lfm2_layer0_representability_study.py` (repo root).
Tests: `tests/test_run_lfm2_layer0_representability_study.py`.

## Sealed inputs

| input | identity |
|---|---|
| parent trace cache | `76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95` |
| selective session | `00271f4eea02eeb032843261197d4574837439b7a40bb0379aec1eedca4dd770` |
| balanced-basis map | `df7f6e88b491df7b42f9ef0ffdb328233b4a77e96f1231b337d68f3cac75cbe9` |
| balanced-basis completion | `99226017391a3239167d995af93795960f5c33b7b3c8bb57b3f97c7e15f5f3da` |
| sealed chart tensor | `58ce5f180dbe6deabf249078ca09cc2c1d81cfac58d051431f9a59f6e18d1ea3` |
| layer-0 evidence selection | `f629e903436c6424c64aa642af421e5fe44546fde607e966ea0955e8fbe5d3ed` |
| sparse residual assignment | `9f1160809fc3cd8d1a7a79747f39e7f8c3d7127365b671666f7a2dac12369e66` |

Runtime: `/home/cereal/model-surgery-lab/deployments/neural-forge-layer0-3953ede4`,
whose `src/neural_forge` is byte-identical to local HEAD `3953ede4` across all
383 files.
