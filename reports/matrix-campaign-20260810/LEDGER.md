# Stage-1 matrix campaign -- ledger

Rendered 2026-08-10T15:04:55+00:00 from `C:\Users\cereal\Documents\ai training\neural-forge\reports\matrix-campaign-20260810\ledger.jsonl`.

**2 cells sealed** (6 ledger rows including re-runs); newest seal 2026-08-10T14:37:23.263+00:00.

| status | cells | meaning |
|---|---:|---|
| `pass` | 1 | free-run >= 0.90 and decode smoke reviewed coherent (§5 rule 1) |
| `infra` | 1 | calibration refused the cell -- void, not evidence |

## 1. The matrix

Rows are components, columns are techniques.  A position shows the deepest level whose cells all passed, or the shallowest level attempted and the chain link it broke at.  `.` is untested.

| component | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|
| C01 token_interface_in | . | . | . | . | X4:infra | . | . | . | . |

### Calibration row

| cell | status | free-run vs control | notes |
|---|---|---:|---|
| `CTRL.copy_exact` | PASS | 1.0000 | decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verba... |
| `CAL.C01.T5.X4` | INFR | 0.0000 | CellTechniqueError: A9 route exited 1. Tail: db23d","stage":"chunk-publication-complete"},"schema":"neural-... |

## 2. Level ladders

§5 rule 2: a technique+component works *at level L* only if it passes at L and every shallower level, at >= 2 of its 3 sites.  A pass deeper than a fail is flagged `non-monotone` and does not count.

| component | technique | M16 | M8 | M4 | X4 | verdict |
|---|---|---|---|---|---|---|
| C01 | T5 | . | . | . | 0/1 | void (infra) -- no evidence |

## 3. Failure classes

The chain is walked mechanically (§5 rule 5); this histogram is the campaign's failure analysis in full.  `free-run` is the F3 signature -- a cell that cleared every numeric link and still lost behavior -- and is the most information-dense class in the matrix.

| first divergent metric | cells | share |
|---|---:|---:|
| `infra` | 1 | 100% |

## 4. Band-versus-behavior agreement

§5 rule 4: boundary metrics are diagnostic only, and whether the S8 `<= 0.4` macrocell band *predicts* the behavioral outcome is itself a campaign output.  Rows are the band's prediction, columns the behavior.

| band | behaved | did not behave |
|---|---:|---:|
| in band (<= 0.4) | 0 | 0 |
| out of band | 0 | 0 |

Agreement 0/0 (no scored cells yet).

## 5. Per-cell drill-down

### `CAL.C01.T5.X4` -- INFR

token_interface_in / sparse_omp_routes / level X4 / site GLOBAL / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0305653c` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 1321.3 | `276fd2e49dd4` |
| `c9fcf7bd` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 26.7 | `c079146235cd` |
| `d0e0fdaf` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 5.3 | `a9eef18a3639` |
| `3871e23b` | infra | infra | -- | -- | -- | 0.000 | 0.000 | -- | 5.2 | `9b0cc7b841a5` |

> CellTechniqueError: A9 route exited 1.  Tail:
> db23d","stage":"chunk-publication-complete"},"schema":"neural-forge-lfm2-sparse-semantic-compile-progress-v1"}
> {"details":{"details":{"compiler_progress":{"core_progress":{"chunk_source":"computed-local-ridge-active-k-refit","completed_donor_rows":64402,"continuous_fold_coefficient_count":4486,"donor_row_start":64256,"donor_row_stop":64402,"event":"row-chunk-complete","phase":"active-k-local-ridge-refit","phase_cache_key":"a18045f6a5465344b4d14ffe750118c46cb4d9f19506c9026feda820a32c7ef3","previous_progress_sha256":"e5be6324b849fa3288b936b52047bf845a35483c5533dafedfc3b614bc179f07","progress_index":252,"progress_sha256":"d600196f3ac36c5ef7286b3d0397107b22d292640b6f9ca87d9f69edd9db757c","schema":"neural-forge-lfm2-sparse-semantic-progress-v1","semantic_code_coefficients_sha256":"472d95e4864dc95cc83ff12fca9747edf0ca9b58b7d6d56f64efba8c930947f4","total_donor_rows":64402},"telemetry":{"cuda_device_index":null,"cuda_max_memory_allocated_bytes":0,"cuda_memory_allocated_bytes":0,"cuda_memory_reserved_bytes":0,"device":"cpu","elapsed_seconds":73.94247139804065,"requested_dtype":"float64","resolved_dtype":"float64","rss_bytes":3015815168}}},"phase
> ... [800 more characters in the receipt]

### `CTRL.copy_exact` -- PASS

copy_exact / copy_exact / level NA / site GLOBAL / wave 1

| instance | status | link | write cos | macrocell relL2 | vs control | free-run | lcp | cond | s | receipt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0317de93` | pass | -- | -- | -- | 0.0000 | 1.000 | 1.000 | 1.0 | 30.9 | `56854f8dc2e5` |
| `0317de93` | pending-review | -- | -- | -- | 0.0000 | 1.000 | 1.000 | 1.0 | 30.9 | `2a98dd999585` |

> decode-smoke review by claude-deep-implementer/MX-A of receipt 2a98dd9995859f43: 3/3 on task from the verbatim text. 1 circuit breaker: correct mechanism, mildly repetitive, EOS. 2 dinner plan: on task, hit the 128-token cap -> censored not failed; the one-pan constraint is only nominally honoured. 3 neighbour note: correct shape and register, EOS; for your convenience is a wording slip and the trailer is not localised. This child is the parent bit-for-bit, so this is the campaign behavioural baseline, not a fold result.

