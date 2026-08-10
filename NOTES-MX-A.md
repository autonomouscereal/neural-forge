# NOTES-MX-A — matrix campaign, harness lane

Wave MX-A. Builds the instrumentation half of
`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md`: the cell runner, the sealed receipt
schema, the technique dispatch table, the ledger, the renderer, and the two
calibration cells. The technique half (T2–T9 kernels) is a parallel lane and
registers into the dispatch table without either side editing the other's file.

Repo baseline HEAD `dee1c952`. Nothing under `src/` or `experiments/` that
existed before this wave was modified; every file below is new. No `git add`,
no commit.

**Calibration verdict up front:**

| cell | verdict | one line |
|---|---|---|
| 0a `CTRL.copy_exact` | **PASS** | the all-copy-exact child reproduces the parent bit-for-bit; free-run 1.000, lcp 1.000, terminal relL2 0.0, decode smoke 3/3 coherent |
| 0b `CAL.C01.T5.X4` | **FAIL — campaign STOPS** | the sealed A9 route cannot be reproduced on this estate: the runtime dtype resolver gained a `pre-ampere-bfloat16-body-fallback-to-float32` rule after A9 was sealed, which changes the numeric-dtype contract, which changes every downstream phase key. Detail in §5. |

---

## 1. What was built

| path | role |
|---|---|
| `src/neural_forge/matrix_cell_receipt.py` | the sealed receipt schema (§3 of the design), the seven-link failure chain, the acceptance clause, the ledger reader/writer |
| `src/neural_forge/matrix_technique_registry.py` | the dispatch table, the component/level/evidence contract, and the T1 + `copy_exact` reference kernels |
| `experiments/run_lfm2_matrix_cell.py` | the cell CLI: one process, one cell, seven stages, one sealed receipt |
| `experiments/render_matrix_ledger.py` | `ledger.jsonl` → a readable matrix report |
| `experiments/seal_matrix_cell_review.py` | seals the frozen decode-smoke review as a separate cell instance (deviation 5) |
| `experiments/matrix_campaign_probe_prompts_v1.json` | the sealed 8-prompt free-run probe, `cf2f20a4…` |
| `experiments/matrix_campaign_smoke_prompts_v1.json` | the sealed 3-prompt decode smoke, `2d7c7f7d…` |
| `tests/test_matrix_cell_receipt.py` | 24 tests: cell-id axes, fail-closed fields, chain order, acceptance clause, seal, ledger, renderer |
| `tests/test_matrix_technique_registry.py` | 21 tests: component resolution, level widths, dispatch precedence, result guard, T1 and C14 kernels |
| `tests/test_run_lfm2_matrix_cell.py` | 16 tests: prompt sealing, evidence alignment, behaviour metrics, A9 calibration comparison, an end-to-end control run |

61 tests, all CPU, all toy-scale.

## 2. The receipt and the chain

`schema: neural-forge-lfm2-matrix-cell-receipt-v1`, canonical-JSON sealed. Two
properties do the work.

**Fail closed on a missing field.** `validate_receipt` checks a fixed dotted-path
set with declared types and refuses to seal anything incomplete. That is the
mechanism by which N1/N2/N3 are "closed by construction": a cell whose solve did
not report `condition_number`, `prior_norm`, `correction_norm`,
`requested_clamp_ratio` and `allowed_clamp_ratio` cannot produce a receipt at
all. Every numeric slot in `blank_receipt` starts at `None`, never `0.0`, so a
forgotten measurement fails loudly instead of claiming a perfect zero.

**The chain is mechanical.** `first_divergent_metric` walks the seven links in
the design's registered order and returns the first threshold crossed:

```
1 calibration.passed is false                                   -> infra
2 condition_cap_saturated, or clamp_engaged with
  requested/allowed > 4.0                                       -> solve-conditioning
3 boundary.write_heldout_cos      < 0.95                        -> local-solve
4 boundary.post_operator_heldout_relL2 > 0.40                   -> macrocell-band
5 composed.vs_copy_exact_control_delta > 0.02                   -> composition
6 behavior.free_run.token_match_vs_control < 0.90               -> free-run
7 decode_smoke_review == incoherent                             -> behavior
```

A link is skipped only when the receipt *declares* it inapplicable
(`boundary.applicable`, `composed.applicable`); a missing field is an error,
never a skip. `behavior.free_run.token_match_vs_control` can never be skipped —
it is the only acceptance signal, so a null there raises rather than passing.

**The acceptance clause reads behaviour only.** `campaign_pass` touches
`behavior.*` and `composed.emit_zero_floor_passed` and nothing else. A test pins
this from both sides: a wrecked boundary with an intact free-run passes, and a
perfect boundary with a broken free-run does not.

## 3. The technique contract

The dispatch table is keyed `(technique_id, component_id)` with fallback to
`(technique_id, None)`; most specific wins. That is what lets the token-interface
T5 route (the A9 mechanism, pinned to C01/C02) and the T5 body-route
generalization live under one technique id without either lane touching the
other's registration.

A kernel receives a `CellTechniqueRequest`:

```
cell, component (ComponentSpec), level, level_width, seed,
parent_weights   read-only, float64 CPU, keyed by real parent state-dict names
evidence         MatrixCellEvidence: fit_rows(endpoint) / heldout_rows(endpoint),
                 row-aligned across endpoints, float64 CPU
knobs, workspace, log
```

and returns a `CellTechniqueResult`: **parent-shaped** materialized weights for
its own component's keys only, `stored_parameter_count` (where the contraction
is actually declared), the nine mandatory `solve` telemetry fields, and an
optional `predict` for stateful techniques.

`validate_result` runs before anything is installed and rejects a stray key
(a two-component fold), a changed shape, non-finite values, and missing
telemetry. A *route* technique that publishes a whole child instead of
installing a tensor is the one permitted exception and must set
`child_directory`.

Why parent-shaped: §1.1 says everything except the folded component is
`copy_exact` with bit-identical parent weights. That is only literally true if a
contracted component still materializes at the parent's geometry — the level is
the width of the **stored** form. A cell that changed a shared dimension would
silently be a two-component cell.

**T1 is wired as the reference implementation**, using the production
`_fit_linear` unchanged with `solver_diagnostics=True`. One property of it is
worth knowing before reading a T1 receipt and is pinned by a test: when the
sealed write really is `input @ P.T` (which it is for every pure linear
component), the projected prior is already the reduced-rank optimum, so the
solve's correction is identically zero and the cell seals `update_scale 1.0`
with a zero correction norm. The contraction lives entirely in the basis. The
solver only does real work when the write is not literally the parent map
applied to the input endpoint — which is the case wherever RoPE, the q/k
layernorms, or the FIR state sit between a projection and its captured write.

C14 (norms) has its own kernel: the standalone RMSNorm gain refit through
`fit_diagonal_norm`, which is the N10 measurement nobody has taken.

**The parallel lane is wired and working.** As of this run the cell runner
auto-imports every `neural_forge.matrix_technique_*` module and the three that
exist register cleanly:

```json
"technique_modules": {"matrix_technique_body_routes": "imported",
                      "matrix_technique_lowrank_writers": "imported",
                      "matrix_technique_relation_matching": "imported"}
```

They register `T4`, `T5` (per body component), `T7` and `T8` through
`register(..., replace=True)`. A technique module that fails to import is
recorded in the cell diagnostics and does not take the cell down — a broken T7
must not cost a T1 cell.

## 4. The cell CLI

`--cell-id C13.T5.M16.L05 --mode cell --out DIR --ledger PATH --parent-model DIR
--control-artifact DIR [--knobs JSON] [--repo-head SHA]` plus the sealed-evidence
group (`--parent-trace-cache`, `--evidence-cache`, `--basis-phase-*`).

Stages, one process: materialize the copy_exact child → calibration gate →
technique solve → install → boundary and macrocell score on the authoritative
held-out split → composed terminal → free-run probe → decode smoke → seal →
append ledger.

The **calibration gate** is the layer-0 study's `published_child` pattern
generalized: score the *parent's own* component under this harness's row
selection and metric; if reproducing the parent is not ≈0 the harness is
measuring something else and the cell raises before any solve is believed. That
path seals `infra` (void, not evidence), which is the chain's first link.

**Evidence** opens the sealed selective trace session once per site, flattens
every requested endpoint into row-aligned `(rows, width)` float64, inherits the
authoritative FIT/heldout split from the sealed balanced-basis phase, and
persists an authenticated cache keyed on the trace cache key, layer, endpoint
set and phase list. Attention endpoints arrive as `(batch, tokens, heads,
head_dim)` and are folded to `heads*head_dim` before anything else, because that
is what the projection actually wrote. Without the sealed balanced-basis phase
the runner **refuses** to invent a split.

**Budget.** Target 720 s, hard cap 1200 s, enforced by a watchdog thread that
seals a complete `timeout` receipt, appends it to the ledger, and exits — a cell
that dies silently costs the campaign a row *and* the reason.

## 5. Calibration

### 5.1 Cell 0a — `CTRL.copy_exact` — PASS

RTX 3090, float32, 30.9 s wall.

| field | value |
|---|---|
| `calibration.copy_exact_reproduction_relL2` | 0.0 (installed digest == parent digest) |
| `behavior.free_run.token_match_vs_control` | **1.0** (8 prompts × 64 greedy tokens) |
| `behavior.free_run.mean_lcp` | **1.0** |
| `composed.terminal_heldout_relL2` | 0.0 |
| `composed.vs_copy_exact_control_delta` | 0.0 |
| `composed.norm_ratio_per_layer_minmax` | [1.0, 1.0] |
| `composed.emit_zero_floor_passed` | true |
| decode smoke | 3/3 on task; 2 EOS, 1 cap-censored |
| receipt | `2a98dd9995859f43…` |
| reviewed receipt | `56854f8dc2e5…`, status `pass` |

The three decode-smoke outputs, reviewed from the verbatim receipt text:

1. *circuit breaker* — correct mechanism, mildly repetitive, EOS. Coherent.
2. *dinner plan* — on task, hit the 128-token cap, so **censored, not failed**
   per the GOAL censoring rule; the one-pan constraint is only nominally
   honoured in the observed prefix.
3. *neighbour note* — correct shape and register, EOS; "for your convenience"
   is a wording slip and the trailer is not localised.

This child *is* the parent bit-for-bit, so these numbers are the campaign's
behavioural baseline, not a fold result. Every later cell's
`token_match_vs_control` is measured against this run's probe token ids and its
terminal states.

### 5.2 Cell 0b — `CAL.C01.T5.X4` — FAIL, campaign stops

Pre-registered tolerance (the design does not name one for this cell, so it is
registered here): the reproduced route's compiled artifact identity must be
**bit-identical** to the sealed A9 artifact — tolerance 0 — and the advisory
cached-hidden counts must match exactly. A9's own receipt records the route as
deterministic (deterministic double modified Gram-Schmidt, deterministic
tie-breaking on target id, integer-only support prefixes), and the sealed
codebook phase is reusable, so anything short of bit-identity means the harness
is driving a different mechanism.

**Result: the route compiled a different codebook, then died before it could
report.** Tesla P40 (`CUDA_VISIBLE_DEVICES=4`, compute capability 6.1 — the
accelerator A9's OMP prefix ABI is pinned to), `venv-pascal-cu118-torch260`
(torch 2.6.0+cu118, A9's runtime), the sealed checkpoint, bridge, donor
tokenizer, parent envelope and prompt bytes, `k=32`, `--support-selection
residual-omp`, ridges 1e-5. 1321 s wall, receipt
`276fd2e49dd4ac3b…`, status `infra`, `first_divergent_metric: infra`.

| identity | sealed A9 | this run |
|---|---|---|
| `numeric-dtype-contract.contract_sha256` | `243a4e48…` | `0f52a2d4…` |
| forward dtype | **bfloat16** | **float32** |
| candidate-inventory phase key | `fa786e82…` | `79917b7b…` |
| codebook phase key | `5b5557b8…` | `8c814244…` |
| **codebook artifact** | **`4439a2b7…`** | **`82c497f4…`** |

Bit-identity fails at every level, and the cause is mechanically attributable
and is **not the harness**:

1. **The runtime dtype resolver changed after A9 was sealed.** A9 ran
   `--dtype auto` and the then-current resolver chose **bfloat16** body
   execution on the P40. The current resolver is
   `neural-forge-lfm2-runtime-dtype-resolution-v2` and returns
   `resolution_reason: "automatic-float32-pascal-or-older-cuda"`. Asking for
   bfloat16 *explicitly* does not restore it either — the receipt then reads
   `requested_dtype: bfloat16`, `resolved_dtype: float32`,
   `fallback_recorded: true`,
   `resolution_reason: "pre-ampere-bfloat16-body-fallback-to-float32"`.
   The fallback is properly recorded rather than silent, which is good
   engineering and also means A9's sealed configuration is no longer reachable
   on Pascal.
2. That changes the numeric-dtype contract, which is an input to every
   downstream phase key, so nothing hit the sealed phase cache and the whole
   4,096-row residual-OMP support search recomputed from scratch — under
   float32 instead of bfloat16 evidence.
3. The recomputed codebook is therefore a *different artifact*, `82c497f4…`.
   It is not "close to" `4439a2b7…`; it is a different compiled object.

The route then failed outright at the cached-hidden diagnostic:

```
Lfm2CircuitTraceCorruptionError: Cache implementation differs from the current
collector/runtime and is not an explicitly compatible sealed revision.
```

so no `run-receipt.json` and no cached-hidden counts were produced at all; the
codebook digest above is recovered from the sealed progress log
(`route-stdout.log`). The harness now extracts that digest into the failure
message so a later run does not lose it.

**Verdict: calibration FAILED. Per §4.1 nothing else runs, and this is
reported rather than worked around.**

### 5.3 The bigger problem the calibration surfaced

Chasing the trace-cache error found something that blocks the campaign
independently of A9:

* **The sealed parent trace cache the design names as the campaign's evidence
  base cannot be opened at HEAD.**
  `…/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946e…`
  carries `implementation_sha256 eb3bddeb…`, which *is* listed in
  `hf_lfm2_circuit_trace_cache_compatibility._EXPLICIT_IMPLEMENTATION_COMPATIBILITY`
  — but only paired with collector sources `92ee0676…` and `b27c0913…`. The
  current collector is `23e47f4c…` and the current canonical contract source is
  `789c6a6f…` against the cache's `5956f909…`. The triple is unlisted, so the
  reader refuses. This is **pre-existing**: deployments `bccb5b78`, `9b03caf7`
  and `da5b9da6` all carry the same collector digest and all refuse the same
  cache. It is not caused by this wave and not by the technique lane's
  uncommitted edits.

* **The implementation identity includes the torch runtime**, so readability is
  venv-dependent:

  | venv | torch | `implementation_sha256` | opens `dd9efbe2…`? |
  |---|---|---|---|
  | `venv` | 2.11.0+cu128 | `d8af5f9d…` | **yes** |
  | `venv-pascal-cu118-torch260` | 2.6.0+cu118 | `689fe5b0…` | **no** |

  That is why cell 0b died: it *had* to run under the Pascal venv to preserve
  A9's OMP accelerator ABI, and that venv cannot read the diagnostic cache.
  There is no venv in which both constraints hold.

* A scan of all 202 trace-cache manifests under `artifacts/` finds exactly one
  readable **parent** cache at HEAD:
  `lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e…`
  (16 batches, stored impl `00700dc2…`, listed). Everything else is BLOCKED.

**What this means for wave 1.** Body cells (C03–C14) need the sealed parent
trace cache *and* the sealed balanced-basis phase bound to it, for the
authoritative FIT/heldout split. Neither is reachable today. Three decisions
are Cyril's, not mine, and I have deliberately made none of them:

1. Audit and extend the compatibility policy so `eb3bddeb…` is admitted against
   the current collector. The module's own docstring says changing the collector
   "invalidates every listed pair until this policy is explicitly audited and
   updated" — that audit is the gate, and the collector is frozen.
2. Or re-point the campaign at `dd9efbe2…` as the evidence base, which is a
   different corpus (16 batches) with no balanced-basis phase bound to it, so
   the authoritative split would have to be re-established.
3. Or decide that A9 is calibrated by its *sealed* artifacts rather than by
   re-execution, and drop cell 0b to a documentation check. That is a weaker
   calibration and it would mean the campaign never demonstrates it can drive
   the one known behavioral success.

Cell 0a stands regardless: the behavioural control is sealed, reviewed, and
passing, so the moment the evidence question is settled, wave 1 can start
against it.

## 6. Ledger sample

`ledger.jsonl` is append-only, one canonical-JSON line per sealed receipt. A
re-run of a cell gets a new `cell_instance_id`, never an overwrite, and the
renderer treats the newest instance as the cell's state while keeping every
older one in the drill-down. The passing control row, abridged:

```json
{"schema":"neural-forge-lfm2-matrix-cell-ledger-v1",
 "cell_id":"CTRL.copy_exact","cell_instance_id":"0317de93…-review",
 "cell_kind":"control","component":"copy_exact","technique":"copy_exact",
 "component_id":"CTRL","technique_id":"copy_exact","level":"NA","site":"GLOBAL",
 "campaign_wave":1,"status":"pass","first_divergent_metric":null,
 "decode_smoke_review":"coherent","calibration_passed":true,
 "write_heldout_cos":null,"post_operator_heldout_relL2":null,
 "endpoint_band_le_0p4":null,
 "vs_copy_exact_control_delta":0.0,"emit_zero_floor_passed":true,
 "token_match_vs_control":1.0,"mean_lcp":1.0,
 "condition_number":1.0,"wall_time_seconds":30.925,
 "receipt_sha":"56854f8dc2e5…","notes":"decode-smoke review by …"}
```

The rendered report (`reports/matrix-campaign-20260810/LEDGER.md`) has five
surfaces: the component × technique matrix, the per-row level ladders with §5
rule 2's `works to L` / `non-monotone` verdict, the `first_divergent_metric`
histogram (which *is* the campaign's failure analysis, since rule 5 forbids a
narrative one), the §5 rule 4 band-versus-behaviour 2×2, and a per-cell
drill-down with every instance and receipt digest. Notes are whitespace-collapsed
and pipe-escaped for table cells and re-emitted verbatim as a blockquote, so a
traceback in a receipt cannot silently shear the table — there is a test for
exactly that.

## 7. Deviations from the design, with reasons

1. **`outcome.status` gains `pending-review`.** §5 rule 1 makes a human
   decode-smoke verdict necessary for a pass, and that verdict cannot exist at
   seal time. Without the extra value every unreviewed cell would have to be
   sealed either as an unearned `pass` or a false `fail`.
2. **`boundary.macrocell_endpoint` is new.** §3 names the field
   `post_operator_heldout_*` while its own parenthetical says the score is taken
   at "the composed layer output". For an MLP component those are different
   trace endpoints. The field names stay frozen; the endpoint actually scored is
   named explicitly, and for body components it is `residual.post_layer` —
   the composed layer output, per the S5 rule that an operator is never scored
   alone when an MLP shares the layer.
3. **`cell_instance_id`, `boundary.applicable`, `composed.applicable`,
   `outcome.decode_smoke_review` are new mandatory fields.** The first is
   required by §3's own ledger rule; the other three exist so a cell can declare
   a link inapplicable *explicitly* rather than having it skipped because a
   field happened to be absent.
4. **`composed.vs_copy_exact_control_delta` equals
   `composed.terminal_heldout_relL2` in this campaign.** The control child is
   the parent bit-for-bit, so the control's terminal state *is* the parent's and
   the two quantities coincide. Both are recorded rather than collapsing them,
   because a future control that is not the parent would separate them.
5. **`experiments/seal_matrix_cell_review.py` is a fifth file the task did not
   name.** Without it the campaign can never reach `pass`, because the runner
   has no path to write anything but `pending` into
   `outcome.decode_smoke_review` — deliberately, so it cannot certify its own
   behaviour. The reviewer seals a *new instance* of the same cell carrying the
   digest of the receipt it graded; the ledger stays append-only and the
   unreviewed instance stays visible.
6. **The composed-terminal and free-run measurements use the sealed 8-prompt
   probe set as the "held-out prompt set".** §2.1 stage 4 says "the held-out
   prompt set" without naming one. The probe set is sealed for the whole
   campaign and disjoint from the 16 whole-response review prompts, so no cell is
   ever scored on text its review will also judge.
7. **Cell 0b ran with `--hard-cap-seconds 7200`, not 1200.** It drives the
   entire sealed A9 route (a 4,096-row OMP support search plus 16 whole-response
   generations at 256 tokens) — a campaign-calibration workload, not a matrix
   cell. The 12/20-minute budget governs matrix cells; sealing a `timeout` for
   the cell that decides whether the campaign may start would have destroyed the
   information the cell exists to produce. `cell-diagnostics.json` still records
   `target_seconds` and `over_target`.
8. **X4 widths are not guessed.** §1.1 defines X4 as "the width the canonical
   exact-4x layout assigns this component", which is an option-bank lookup.
   Only the token interface's X4 is settled (4,096, the proven A9/A11 route
   budget). Every other component's X4 must be passed explicitly with
   `--x4-width`; `resolve_level_width` raises otherwise rather than inventing a
   layout the campaign never adopted.
9. **Attention q/k/v cells cannot compose a macrocell from their write.** Those
   components write into `heads*head_dim`, not the residual width, so
   `residual + (child_write - parent_write)` is not defined. The receipt records
   `macrocell_endpoint` suffixed `#not-composable-from-write` and neutral values
   for links 4, so the chain reaches the behavioural links instead of failing on
   a quantity that does not exist. Those cells are decided by free-run, which is
   the correct authority anyway.

## 8. Findings that are not about the harness

* **The working runtime's review prompt file has drifted from the sealed bytes.**
  `runtime/neural-forge-wip-local-action-v2/research/lfm2_semantic_whole_response_prompts_v1.json`
  hashes to `0ee6bf14…`; A9 consumed `ff6fa27e…`, which is what every other
  runtime copy and the repo's own copy under
  `reports/89cd45b0-same-body-v4096-control/` still hash to. Any rerun launched
  from that runtime path is scoring a different prompt set. Cell 0b was pointed
  at the repo-sealed copy.
* **The estate's runtime dtype resolver changed after A9 was sealed.** See §5.2.
* **A third, independently-named matrix harness exists in the working tree**
  (`src/neural_forge/hf_lfm2_matrix_cell.py`,
  `experiments/run_lfm2_component_method_matrix.py`,
  `experiments/render_lfm2_matrix_ledger.py`). It is not this lane's and it does
  not use the frozen names from §4.3 of the design. Somebody has to decide which
  one the campaign runs before wave 1; they will produce incompatible ledgers.

## 9. Verification

```
# local, CPU, .venv (python 3.11.9, torch 2.13.0+cpu)
python -m pytest tests/test_matrix_cell_receipt.py \
                 tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell.py -q
# -> 61 passed

python -m pytest tests/test_run_lfm2_layer0_representability_study.py -q
# -> 4 passed   (the pattern this harness generalizes is not regressed)

python -m pytest tests/ -q -k matrix
# -> 231 passed  (this lane plus the technique lane's own tests, together)
```

Server runs are in §5. The renderer was run against the real ledger:

```
python experiments/render_matrix_ledger.py \
  --ledger reports/matrix-campaign-20260810/ledger.jsonl \
  --out    reports/matrix-campaign-20260810/LEDGER.md
```

The deployed copies are byte-identical to the repo's (sha256, first 16):

```
1511325bce1662d8  experiments/run_lfm2_matrix_cell.py
13715e6883322cc3  experiments/render_matrix_ledger.py
08fe52edc834ee0b  experiments/seal_matrix_cell_review.py
f268959a8310e7c9  src/neural_forge/matrix_cell_receipt.py
5762791a5066016e  src/neural_forge/matrix_technique_registry.py
```

**Not verified:** no body cell (C03–C14) has been run against the real sealed
trace cache. The evidence loader, the calibration gate, the boundary scorer and
the T1 kernel are proven at toy scale on synthetic evidence and against the
real parent's tensor names and shapes, but the first real body cell will be the
first time the 85 GB sealed session path executes end to end. That was out of
scope for this wave — the task's runnable deliverables were the two calibration
cells — and it is the first thing wave 1 should do.

**Backups and the destructive-change check.** Full path/size/mtime inventories
were taken before anything shared was touched:
`/mnt/ai_data/neural-forge-matrix/backup/phase-cache-inventory-before.txt`
(47,325 entries) and `.../a9-sealed-inventory-before.txt` (4,572 entries). The
matching `-after` inventory was taken when cell 0b finished. Result:

| check | result |
|---|---|
| phase-cache paths before / after | 47,325 → 88,930 |
| net new paths | 41,605 (the recomputed OMP supports and coefficient chunks) |
| paths removed | **0** |
| pre-existing files rewritten | **0** — the only entries whose mtime moved are the four *parent directories* the new entries were created inside |
| `find … -newermt` over the sealed A9 run directory | empty |
| `find … -newermt` over `lfm2_tokenizer_bridge_evidence_20260717` (the sealed parent trace cache) | empty |

Also kept: `.../backup/stale-locks/`, holding the two phase-cache transaction
locks removed after an interrupted run. Both owning pids were confirmed dead
with `kill -0` before removal, and the lock files were copied aside first.

## 10. Paths

* deployment: `/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952`
* campaign output: `/mnt/ai_data/neural-forge-matrix/` (`ledger.jsonl`,
  `cells/CTRL.copy_exact/`, `cells/CAL.C01.T5.X4/`, `backup/`)
* route working directory (local disk, not the CIFS share — the OMP chunk
  publisher measured ~6x slower over `/mnt`):
  `/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/`
* local report: `reports/matrix-campaign-20260810/`
