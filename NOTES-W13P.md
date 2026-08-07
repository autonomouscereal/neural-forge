# NOTES-W13P — the A12 rule ported to the body, and the tripwire that could never have fired

Wave 13P, branch `codex/lrc-blank-compression`, base `5023bf7e` (the sealed
overlap audit). Two structural fixes from `docs/OVERLAP_AUDIT_2026-08-07.md`
§Realignment items 4 and 5. Local CPU only. **Nothing committed.**

Both changes land in one file, `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`,
plus tests. `hf_lfm2_dense_budget_model_fold.py`, `dynamic_closed_loop_fold.py`,
`hf_lfm2_moving_gauge_residual_reclosure.py`, `hf_lfm2_on_policy_evidence_round.py`,
`run_lfm2_exact4x_moving_gauge_v5*` and `experiments/run_lfm2_dynamic_catalog.py`
are **untouched**. ARM A is running against `b65c81f5` + cherry-picks on the
server; nothing here reaches it.

```
before  src/.../hf_lfm2_dynamic_closed_loop_adapter.py  5e99e2c190aa6732c01bf2740f4400a0f3eca20ba32adfca09ccfe60ec555c47
after   src/.../hf_lfm2_dynamic_closed_loop_adapter.py  4872711acc55543923441c7b987175c16e9b79ea2f7ef89bd04bc8b9b2299238
before  tests/test_hf_lfm2_r13_retention_lane.py        3b310f58753006fd991f921474525c763a4141028d01543a54344d2284b46817
after   tests/test_hf_lfm2_r13_retention_lane.py        36aeedb972fb4eb76082f2813dd1041e7b5da78ef6d88db9682b9f05fef8b8bf
diff    2 files changed, 568 insertions(+), 15 deletions(-)
line endings  LF on both files, 0 CRLF pairs (W13M §6.5's self-inflicted
              round-trip did not repeat; checked by byte count, not by eye)
backups …/scratchpad/backup-w13p/{*.orig, *.modified, HEAD.txt}
```

---

## 1. Fix 1 — terminal non-regression authority over the body reclosure

### 1.1 What was open

A12 (2026-07-30) sealed the operational rule **reject any joint rewrite set
whose closure regresses the preserved proven baseline**, and applied it at the
token interface. The body reclosure — the same operation class, a post-hoc
co-adapted rewrite of already-solved components fitted against cached captures —
was never covered by it. The terminal composition floor's non-regression bound
lives in `_terminal_composition_floor` and was reachable from exactly one call
site: `_validation_decision`, which only `validate_layer` and
`validate_final_normalization` reach. `validate_body_reclosure` returned
`accepted=True` **unconditionally** for the moving-gauge mode, with its own
terminal number recorded as `heldout_terminal_relative_rmse` and explicitly
labelled `performance_metrics_are_decision_authority: false`.

In the sealed r13 run that is the hole the audit named: 0.806258 → 0.971148,
**0.165 worse**, accepted, because the 0.25 tolerance guarded layer installs
only (NOTES-W13M §1.3, §2).

### 1.2 The change

Three sites, one clause.

* `hf_lfm2_dynamic_closed_loop_adapter.py:3151` — `_BodyReclosurePayload` gains
  `terminal_baseline: FitMetrics | None = None`, mirroring `_ProducerPayload`'s
  field of the same name and the same comment shape. Defaulted, so the disabled
  (identity) reclosure construction is unchanged.
* `:14548` — `compile_body_reclosure`, inside the `moving_gauge_v3` branch only,
  measures the **pre-reclosure** composed endpoint through
  `_heldout_terminal_metrics(capture.paired, batch_indices=split.validation_batch_indices)`
  and stores it on the payload (`:15036`). This is literally the same call the
  per-layer floor makes for its baseline, on the same held-out rows, one stage
  later — so the reclosure is measured against the stack the fourteen accepted
  layers actually left behind, not against a differently-constructed number.
* `:16067`, `:16103` — `validate_body_reclosure` measures the **post-reclosure**
  endpoint off the fresh full-child recapture (before that capture is released)
  and runs `_terminal_composition_floor(layer_index=None, baseline=…, candidate=…)`.

`layer_index=None` is deliberate and load-bearing: it engages the
**depth-monotone non-regression bound and only that bound**. The absolute
no-branch floor is a per-install clause keyed on layer depth, and the rule being
ported is A12's non-regression rule. A reclosure that merely fails to improve an
already-bad terminal is not what A12 convicted; a reclosure that makes it worse
is. This is stated in the code comment and pinned by
`body_reclosure_terminal_absolute_floor_engaged: false` in the test.

**No new flag.** The clause keys entirely off
`terminal_composition_floor_enabled` (`--dense-closed-loop-terminal-composition-floor`)
and `terminal_composition_regression_tolerance`
(`--dense-closed-loop-terminal-regression-tolerance`, 0.25 in the arm). With
the floor off, `_heldout_terminal_metrics` returns `None` before doing any work
and `_terminal_composition_floor` returns `(False, (), {})` — no cost, no keys.

### 1.3 Fail closed, not roll back — and why

**Decision: fail closed.** Four reasons, in order of weight:

1. **There is nothing to roll back to.** `_BodyReclosurePayload` retains
   `seed_weight_sha256` and `seed_physical_state_sha256` — *digests* of the
   pre-reclosure writers, not the tensors. A rollback would require snapshotting
   the 82 writer parameters plus the 54 input-producer parameters, i.e.
   effectively the whole dense body (53,552,368 params in the arm), and holding
   that snapshot across the reclosure — the single most memory-pressured stage
   of the run, the one that already calls `release_closed_loop_capture_memory()`
   three times.
2. **The install receipt is already sealed against the post-reclosure child.**
   `_run_body_reclosure` calls `self._install(...)` before validation, and that
   receipt carries `state_sha256_before`/`state_sha256_after`. Restoring the
   pre-state after the receipt exists would leave a sealed receipt describing a
   state the child is not in — an unauthenticated divergence in precisely the
   ledger the program's whole discipline rests on.
3. **Fail-closed already has a first-class evidence path here.**
   `dynamic_closed_loop_fold.py:1695-1725` publishes the complete decision —
   canonical JSON body, `validation_decision_sha256`, `validation_metrics_sha256`,
   `validation_metrics_json` — through `child_closed_loop_validation_decision_rejected`
   *before* it raises. That is the same authority path the layer floor uses, and
   it was built exactly so a refusal keeps its numbers. Porting the rule onto it
   keeps one mechanism instead of two.
4. **Rollback is invasive in a forbidden file.** A "rolled back but continue"
   outcome is not expressible in `ValidationDecision`; `_run_body_reclosure` and
   `_validate` would both need to learn it, and `dynamic_closed_loop_fold.py` is
   off limits for this wave.

The refusal names both values and the stage, per spec. Verbatim from the toy:

```
body_reclosure validation rejected the folded child: moving_gauge_v3 structural
authentication passed on a fresh heldout full-child recapture; numeric fit is
diagnostic only; body reclosure refused: the composed held-out terminal relative
RMSE moved from 0.14604444 before the reclosure to 0.232561736 after it, beyond
the terminal non-regression tolerance 0.05; terminal composed held-out relative
RMSE regressed by 0.0865172959 beyond tolerance 0.05
```

### 1.4 Sealed keys

Twenty new keys under `body_reclosure.validation.metrics`, all prefixed
`body_reclosure_terminal_*` so they can never be confused with the
moving-gauge chart metric `heldout_terminal_relative_rmse` sitting next to them.
The floor's own metric block is re-keyed under that prefix, plus three
reclosure-specific ones:

```
body_reclosure_terminal_baseline_relative_rmse            <- pre-reclosure
body_reclosure_terminal_relative_rmse                     <- post-reclosure
body_reclosure_terminal_relative_rmse_regression
body_reclosure_terminal_non_regression_engaged / _passed / _tolerance
body_reclosure_terminal_absolute_floor_engaged / _passed / _minimum_layer_index
body_reclosure_terminal_layer_index / _rows / _mean_cosine_similarity
body_reclosure_terminal_no_branch_relative_rmse_baseline
body_reclosure_terminal_composition_floor_enabled / _engaged / _failed
body_reclosure_terminal_floor_is_safety_bound_not_intelligence_verdict
body_reclosure_terminal_floor_retains_authority_under_diagnostic_only
body_reclosure_terminal_non_regression_has_decision_authority       (new)
body_reclosure_terminal_non_regression_is_the_ported_a12_rule       (new)
```

The clause keeps **hard** authority — it does not yield to
`intermediate_layer_numeric_checks_diagnostic_only` or
`all_layer_finite_numeric_checks_diagnostic_only` the way a per-layer floor
failure does. The rationale is that those carve-outs exist for an *incomplete*
body; at the reclosure the body is complete by definition. That is recorded as a
key rather than left implicit.

### 1.5 Toy numbers

The toy is the real end-to-end `moving_gauge_v3` reclosure — two layers, the
weight-complete basis, `ChildClosedLoopCompiler(..., fold_body_reclosure=True)`.
Nothing is mocked; the numbers below are what the toy reclosure actually does.

| | composed held-out terminal relL2 |
|---|---|
| after layer 1, before the body reclosure | **0.14604444009075057** |
| after the body reclosure | **0.23256173596399110** |
| regression | **+0.08651729587324053** |

**The toy reproduces r13's shape.** The reclosure makes the composed terminal
endpoint worse (r13: +0.165 on 0.806; toy: +0.0865 on 0.146), and the writers it
rewrites still satisfy their own boundary objective — the same globally
destructive, locally improving signature. Two arms:

| tolerance | outcome |
|---|---|
| **0.25** (the sealed arm's own value) | accepted; pre/post/regression recorded, `non_regression_passed: true` |
| **0.05** | **refused**; `ClosedLoopFoldError`, `non_regression_passed: false`, `composition_floor_failed: true`, one `child_closed_loop_validation_decision_rejected` event with `scope_kind: body_reclosure` carrying the full metrics JSON |

Note the toy's `heldout_terminal_relative_rmse` (the moving-gauge chart metric)
and `body_reclosure_terminal_relative_rmse` (the floor metric) agree to the last
digit — 0.2325617359639911 — because the toy's terminal chart is trivial and the
row weights uniform. **They will not generally agree**, and that matters for how
the r13 counterfactual can be read (below). The guard itself is unaffected: it
compares the floor construction against itself on both sides.

**On the r13 counterfactual — what I can and cannot say.** The pre-reclosure
side is exact: my clause measures the same construction on the same held-out
rows as layer 13's `terminal_relative_rmse`, with no child mutation in between
and one `TrajectorySplit` for the whole run, so it would have read **0.806258**.
The post-reclosure side is *not* 0.971148. That number is
`heldout_terminal_relative_rmse` — the moving-gauge chart-encoded, row-weighted
metric computed in `validate_body_reclosure`, a different construction that only
coincides with the floor's when the terminal chart is trivial (as in the toy).
The floor-construction post-reclosure value for r13 was never computed and
cannot be recovered from the sealed run. **So I cannot say whether a rerun of
r13's exact flags would have been accepted or refused**, and I am not going to
pretend otherwise. What the audit's 0.806 → 0.971 does establish is direction
and rough magnitude: the reclosure moved the composed terminal the wrong way by
order 0.1–0.2 against a 0.25 budget, i.e. close enough to the tolerance that the
choice of tolerance is now a real decision rather than a formality.

**Behavior change, intended.** From r14 on, the body reclosure is inside the
same non-regression authority as a layer install: it is measured before and
after in one construction, the pair is sealed in its own decision instead of
having to be reconstructed from two metrics in two report blocks, and a
regression past the configured tolerance stops the run instead of being
recorded and passed over.

---

## 2. Fix 2 — the free-run tripwire that never evaluated

### 2.1 Root cause, two layers

**Proximate cause: the probe attribute is never assigned in any production path.**
`Lfm2DynamicClosedLoopAdapter.__init__` sets
`self.terminal_free_run_match_probe = None` (`:3573` pre-change) and documents
that a caller supplies it. Nothing does. The CLI flag
`--dense-closed-loop-terminal-free-run-tripwire` threads only
`terminal_free_run_tripwire_enabled` into the adapter *config*
(`experiments/run_lfm2_dynamic_catalog.py:10228`); the probe attribute is written
in exactly three places in the repository and all three are test files
(`tests/test_hf_lfm2_r13_retention_lane.py:424/447/468`,
`tests/test_hf_lfm2_dynamic_closed_loop_adapter.py:4749/4766`). So
`_terminal_free_run_tripwire` did what it was written to do: emitted its four
configuration keys, found `probe is None`, wrote
`terminal_free_run_tripwire_evaluated: false`, and returned. At all 14 layers.
In every run that has ever set the flag. It was never going to fire.

Not the merged-evidence path, not a phase problem, not a gating condition that
happened to be false — the config was threaded and the observation channel was
not.

**Structural cause: under this arm's configuration there is no observation to
thread.** The only counterfactual child-only free rollout inside the fold is the
macro-cell attractor gate's (`hf_lfm2_macro_cell_attractor_gate.py:632`, feeding
`diagnostic_mean_parent_top1_exact_token_match_fraction` at `:699/:773`). Two
facts kill it for r13 and for ARM A:

1. The gate was **off** — `--dense-macro-cell-attractor-gate` is absent from
   `.codex-test-runs/nf-launch-r12-round-refold.sh` and from W13O's re-audited
   ARM A flag list.
2. Even if asked for, it would have been **refused**: cross-tokenizer runs raise
   at `run_lfm2_dynamic_catalog.py:2640` — *"Cross-tokenizer macro-attractor
   replay is not target-token aware"* — and r13/ARM A are bridged runs.

So the honest statement is: the tripwire is enabled in an arm that structurally
carries no free-run measurement at fold time. That is the real gap, and it is
bigger than one clause.

### 2.2 The change

* `:8197` — new `Lfm2DynamicClosedLoopAdapter._macro_cell_free_run_match(snapshot)`
  reads `diagnostic_mean_parent_top1_exact_token_match_fraction` out of a
  macro-cell rollout snapshot. That metric is the mean positional agreement
  between the child's own greedy rollout and the sealed parent-greedy rollout it
  was seeded from — the *same construction* as the run-level
  `free_run.token_match_fraction` (`run_lfm2_dynamic_catalog.py:6975`), and
  exactly the observation the tripwire was specified against.
* `:8227`-`:8288` — `_terminal_free_run_tripwire` gains
  `free_run_match_observation`. Resolution order is **assigned probe first**,
  supplied rollout second; an explicitly assigned probe still wins, so every
  existing test and every existing semantic is preserved.
* `:8356`, `:8557` — `_validation_decision` threads the observation to the
  tripwire.
* `:14251`-`:14305` — `validate_layer` hoists the macro-cell candidate capture
  to *before* the decision **only when the tripwire is enabled**, hands its
  free-run number to `_validation_decision`, and reuses that same snapshot for
  the gate comparison below. One rollout per layer, not two. Nothing between the
  hoisted capture and the comparison touches the child, and `gate.capture` is
  itself state-preserving by construction (it raises if the child state,
  gradients or training flags moved).

The adapter still **never generates for the tripwire**. It reads a rollout it
already performs. The clause remains advisory: it warns
(`Lfm2TerminalFreeRunTripwire`) and records; only the pre-existing, off-by-default
`terminal_free_run_tripwire_stops_acceptance` rung can refuse, and nothing here
touched it. The terminal floor stays the only terminal decision authority — now
including the reclosure, per Fix 1.

### 2.3 Making the silence self-diagnosing

The audit's actual complaint is that r13 recorded `evaluated: false` with **no
stated cause** and nobody could say why. Two new keys close that permanently:

```
terminal_free_run_tripwire_observation_source :
    "assigned_probe" | "macro_cell_attractor_free_rollout" | "none"
terminal_free_run_tripwire_unevaluated_reason :
    "the assigned free-run probe returned no usable observation"
  | "no free-run observation source: no probe is assigned and the macro-cell
     attractor free rollout is not running, so this run carries no
     counterfactual free-run measurement"
```

Under ARM A's exact flag set the tripwire will still read `evaluated: false` —
because there is genuinely nothing to observe (§2.1) — but it will now say so in
its own sealed decision instead of leaving the next reader to do W13M's grep
archaeology.

### 2.4 What I deliberately did not do

* **I did not add a runner-side refusal** of `--dense-closed-loop-terminal-free-run-tripwire`
  without an observation source. It is tempting (`run_lfm2_dynamic_catalog.py:2808`
  already has the precedent shape, refusing `--…-tripwire-stops` without
  `--…-tripwire`) and it would make an inert instrument impossible. It would also
  make ARM A's exact, re-audited flag list *illegal*, which is a bigger
  commitment than the audit asked for and not mine to impose mid-experiment.
  Recommend it as an r14 decision.
* **I did not give the tripwire acceptance authority.** Explicitly out of scope
  per the task and per the audit's metric policy.
* **I did not wire a new free-run generator** into the fold. Doing so would put
  per-layer child generation into every run carrying the flag and would violate
  the clause's own "the adapter never generates here" contract. Closing the gap
  for bridged arms needs a **target-token-aware** counterfactual rollout — the
  thing `run_lfm2_dynamic_catalog.py:2640` says does not exist yet. That is a
  real piece of work and it is the honest next step, not a one-liner.

### 2.5 Toy numbers

End-to-end toy fold, tripwire enabled, real `Lfm2MacroCellAttractorGate` over the
sealed toy trace cache (2 records, 2 steps, comparison thresholds set permissive
so the gate cannot refuse and the test stays about the tripwire):

```
layer 0 / layer 1, both identical:
  terminal_free_run_tripwire_enabled             true
  terminal_free_run_tripwire_evaluated           true      <- was false, always
  terminal_free_run_tripwire_observation_source  macro_cell_attractor_free_rollout
  terminal_free_run_tripwire_observed            0.0
  terminal_free_run_tripwire_tripped             true
  terminal_free_run_tripwire_is_advisory         true
  terminal_free_run_chance_match_fraction        0.0
  macro_cell_attractor_gate_accepted             true
  validation.accepted                            true      <- advisory, no authority
```

The toy child free-runs at chance, so the clause fires and warns — which is
exactly the signature it was built for, on the first run in which it has ever
been able to speak — and the fold proceeds, because it is advisory.

---

## 3. Byte-identity proofs

Harness: `…/scratchpad/byte_identity.py`. It runs the full toy
`moving_gauge_v3` closed loop (two layers + body reclosure + final
normalization) and dumps **every sealed validation decision, `decision_sha256`
included**, then digests the lot. Run once with the pre-change adapter
(`…orig`, sha `5e99e2c1…`) and once with the post-change adapter
(`…modified`, sha `4872711a…`), same seeds, same split.

**Proof 1 — both lanes off (floor off, tripwire off):**

```
before  DECISIONS_SHA256 2e376acfd2596f2d3ac90d5e28ccf949834389cfdd7e27b436a95603a6157839
after   DECISIONS_SHA256 2e376acfd2596f2d3ac90d5e28ccf949834389cfdd7e27b436a95603a6157839
diff -q decisions_before.json decisions_after.json  ->  identical
```

Byte-identical: layer decisions, body-reclosure decision, final-normalization
decision, all digests.

**Proof 2 — floor ON, tripwire off** (isolates Fix 1 to the stage it belongs to):

```
layers identical      : True
final norm identical  : True
reclosure identical   : False
new reclosure keys    : 20
```

Fix 1 perturbs nothing except the body-reclosure decision, which gains exactly
the 20 keys of §1.4. The per-layer floor is bit-for-bit what it was.

Pinned in the suite as well: `test_body_reclosure_terminal_regression_is_absent_with_the_floor_off`
(no `body_reclosure_terminal_*` key exists at all with the floor off) and
`test_disabled_free_run_tripwire_writes_no_metrics` (no `terminal_free_run*`
key exists with the tripwire off, probe assigned *and* observation supplied).

---

## 4. Verification

Everything below is `.venv/Scripts/python.exe` (torch 2.13.0+cpu,
transformers 5.13.0) from the repo root. `C:/Python314/python.exe` has no torch,
so it can only run `ruff` and `ast.parse` — stated because a green run under the
wrong interpreter would be a lie.

```
python -m ruff check src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py \
                     tests/test_hf_lfm2_r13_retention_lane.py
    All checks passed!

python -m pytest tests/test_hf_lfm2_r13_retention_lane.py -q -p no:randomly
    39 passed in 69.91s            (29 before; +10 new)

python -m pytest tests/test_hf_lfm2_dynamic_closed_loop_adapter.py \
                 tests/test_hf_lfm2_moving_gauge_residual_reclosure.py \
                 tests/test_hf_lfm2_moving_gauge_input_producer_reclosure.py \
                 tests/test_dynamic_closed_loop_fold.py \
                 tests/test_hf_lfm2_closed_loop_restart_capsule.py \
                 tests/test_hf_lfm2_macro_cell_attractor_gate.py -q -p no:randomly
    231 passed in 336.03s

python -m pytest tests/test_run_lfm2_dynamic_catalog.py \
                 tests/test_run_lfm2_dynamic_catalog_r13_retention.py \
                 tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
                 tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py \
                 tests/test_dynamic_fold_artifact.py -q -p no:randomly
    212 passed, 1 skipped in 67.22s

python -m pytest tests/test_dynamic_lfm2_component_dispatch.py \
                 tests/test_dynamic_lfm2_dense_budget_fold_stack.py \
                 tests/test_hf_lfm2_cached_functional_width.py \
                 tests/test_hf_lfm2_cached_global_topology_frontier.py \
                 tests/test_hf_lfm2_cached_heterogeneous_capacity.py \
                 tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
                 tests/test_hf_lfm2_global_topology_phase_cache_exact4x.py \
                 tests/test_hf_lfm2_retained_topology_evidence.py \
                 tests/test_run_lfm2_weight_complete_closed_loop.py \
                 tests/test_hf_lfm2_dense_budget_model_fold_heterogeneous.py -q -p no:randomly
    126 passed in 136.81s
```

That is every test file in the repository that imports the changed module, plus
the reclosure, compiler, capsule, gate, catalog and artifact suites.

**Known-red, unchanged:**

```
python -m pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly
    3 failed, 6 passed in 258.06s
      test_carrier_dense_core_alternates_fold_round_and_refold
      test_bridged_carrier_lane_alternates_fold_round_and_refold
      test_carrier_production_margin_pin_composes_through_a_round
    all three:  Lfm2DynamicClosedLoopAdapterError:
                'Selected parent component payload changed after authenticated load.'
```

Identical count, identical names, identical signature to NOTES-W13M §6.5 —
`d296b602`'s regression, not this change's.

### New tests (10)

Fix 1, in `tests/test_hf_lfm2_r13_retention_lane.py`:

* `test_body_reclosure_terminal_regression_is_absent_with_the_floor_off`
* `test_body_reclosure_records_a_terminal_regression_inside_tolerance`
* `test_body_reclosure_fails_closed_on_a_terminal_regression`

All three drive a **real** toy `moving_gauge_v3` reclosure end to end through
`ChildClosedLoopCompiler`; the regression they measure is one the reclosure
actually produces, not a scripted metric.

Fix 2, same file:

* `test_disabled_free_run_tripwire_writes_no_metrics`
* `test_free_run_tripwire_names_why_it_could_not_evaluate`
* `test_free_run_tripwire_names_an_assigned_probe_that_observed_nothing`
* `test_an_assigned_probe_outranks_the_supplied_rollout`
* `test_the_macro_cell_rollout_is_read_as_the_free_run_observation`
* `test_the_free_run_tripwire_evaluates_inside_a_real_toy_fold`
* `test_the_tripwire_rollout_is_the_gate_rollout_not_a_second_one`

The last one pins the cost claim: exactly one pre-cell and one post-cell capture
per layer with the tripwire on, i.e. the hoist reuses the snapshot and does not
double the rollout.

---

## 5. What I could not verify

* **Production scale.** Everything here is a 2-layer, 4-batch CPU toy. The
  reclosure guard's per-call cost at arm scale is two extra
  `_heldout_terminal_metrics` passes over 13,464 held-out rows × 816 in fp64 —
  the same allocation the per-layer floor already makes 14 times, so the marginal
  peak is ~350 MB transient at a stage that already frees captures three times.
  I did not measure it on the real corpus and cannot.
* **The r13 counterfactual.** §1.5: the post-reclosure value my clause measures
  is a different construction from the sealed 0.971148, was never computed, and
  cannot be recovered from the run directory. Whether r13's exact flag set would
  now pass or fail at the reclosure is **unknown**, and I did not re-run r13.
* **Cross-tokenizer free-run.** §2.4: I could not make the tripwire evaluate
  under ARM A's flag set, because no target-token-aware counterfactual rollout
  exists. The fix makes that fact explicit in the sealed evidence; it does not
  remove it.
* **The reclosure's own acceptance ledger** (NOTES-W13M §1.1) is still dropped
  on the floor at the lift boundary. Out of scope here, still the highest-value
  small change for r14.

## 6. What I deliberately left out

* No rollback path for the reclosure (§1.3) — fail closed instead, with reasons.
* No runner-side refusal of an unobservable tripwire (§2.4).
* No acceptance authority for the free-run signal (§2.4).
* No absolute no-branch floor at the reclosure — non-regression only (§1.2).
* `d296b602`, item 8's depth telemetry, and the boundary-acceptance ledger lift:
  all named in the audit, none of them this wave's items 4 and 5.
* **No `git add`, no commit, no push.**

## 7. Artifacts

| path | |
|---|---|
| `C:\Users\cereal\Documents\ai training\neural-forge\NOTES-W13P.md` | this file |
| `C:\Users\cereal\Documents\ai training\neural-forge\src\neural_forge\hf_lfm2_dynamic_closed_loop_adapter.py` | both fixes, **uncommitted** |
| `C:\Users\cereal\Documents\ai training\neural-forge\tests\test_hf_lfm2_r13_retention_lane.py` | 10 new tests, **uncommitted** |

Scratch (not part of the deliverable, kept for reproduction):
`…\scratchpad\backup-w13p\` (pre-edit `.orig`, post-edit `.modified`, `HEAD.txt`),
`…\scratchpad\byte_identity.py`, `…\scratchpad\probe_reclosure.py`,
`…\scratchpad\probe_tripwire.py`,
`…\scratchpad\{decisions,floor}_{before,after}.json`.
