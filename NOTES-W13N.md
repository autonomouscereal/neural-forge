# NOTES-W13N — the three W13M fixes: lift the ledger, fix the depth read, give the gate a tooth

Wave 13N, branch `codex/lrc-blank-compression`, on `7e25d781`. Three surgical
changes motivated entirely by `NOTES-W13M`:

1. **§1.1** — the reclosure's sealed per-boundary acceptance ledger now reaches
   the run's own report instead of dying at the adapter boundary.
2. **§3** — item-8 depth telemetry read a key on an object that has no such
   field. Both halves fixed; the test that "passed" against the broken code is
   rebuilt on the real emitter.
3. **§1.2** — the acceptance gate refused nothing while the reclosure rewrote 82
   weights, 81 upward, median 33x, 974x at layer 7. It now has an opt-in
   write-norm ratio ceiling that refuses on growth regardless of the boundary
   metric.

Nothing is committed. Backups of all six edited files, plus `HEAD.txt`, are at
`…/scratchpad/backup-20260807-w13n/`.

---

## 1. Fix 1 — the acceptance ledger reaches the run report

### 1.1 What landed

`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`

| site | |
|---|---|
| `:1130` `_lifted_boundary_acceptance_ledger(report)` | pulls `report["diagnostics"]["boundary_acceptance"]`, **re-verifies it against the `boundary_acceptance_sha256` the reclosure publication sealed for it**, and returns the schema, mode, tolerance, digest, compact summary and the **complete ledger** — every decision and every `per_boundary` row exactly as sealed. Returns `None` when there is no ledger. |
| `:1188` `_boundary_acceptance_metadata(report)` | flattens that into the compiled stage's scalar metadata: five counts/digests plus `boundary_acceptance_decisions_json`, one canonical-JSON string built by the module's existing `_durable_diagnostic_json` — the same shape the MLP solver diagnostics already use. Returns `{}` when absent. |
| `:14759` | the lift, computed right after the causal-coverage check |
| `:14894` | `if boundary_acceptance is not None: body.update(boundary_acceptance)` before the artifact is sealed |
| `:14991` | `**_boundary_acceptance_metadata(payload.report)` in `_scalar_metadata(...)` |

The metadata line is the one that actually mattered. `CompiledFoldStage.as_dict`
(`dynamic_closed_loop_fold.py:657-666`) publishes `scope`, `stage`,
`fit_record_keys`, `artifact_sha256` and `metadata` — **and nothing else**. The
payload, and with it the whole artifact body, is never serialized into the run.
That is the mechanical reason a recursive grep for `boundary_acceptance` over the
entire sealed r13 run directory returned zero files: the decisions were sealed,
the digest was carried, and the only surface that reaches the receipt took
scalars only.

Now a guarded run's `closed_loop.body_reclosure.compiled.metadata` carries:

```
writer_reclosure_acceptance                            require_boundary_improvement
boundary_regression_tolerance                          <float>
boundary_acceptance_sha256                             <the ledger's own report_sha256>
boundary_acceptance_evaluated_writer_overwrites        <int>
boundary_acceptance_accepted_writer_overwrites         <int>
boundary_acceptance_refused_writer_overwrites          <int>
boundary_acceptance_refused_transition_indices_json    "[…]"
boundary_acceptance_decisions_json                     {"decisions":[…],"per_boundary":[…]}
```

`refused_transition_indices` — `NOTES-W13I` §11's headline question — is now a
key in the receipt rather than a preimage nobody kept.

### 1.2 The default-mode decision, and the proof

**Chosen: an unconditional, presence-conditional lift. No separate flag. Default
runs lose nothing, because in default mode there is nothing to lose.**

The task asked for an honest call on whether the default unguarded mode also has
a ledger to lift. It does not, and this is structural, not incidental:

* `hf_lfm2_moving_gauge_residual_reclosure.py:4980` —
  `acceptance_decisions = [] if guarded_acceptance else None`
* `:5253` — the ledger is built and sealed **only** inside
  `if acceptance_decisions is not None:`
* `:5570` — `acceptance_schema` / `boundary_acceptance_sha256` /
  `boundary_acceptance_summary` are added to the publication body **only** inside
  `if acceptance is not None:`

So an unguarded reclosure report has no `diagnostics["boundary_acceptance"]` key
at all. `_lifted_boundary_acceptance_ledger` returns `None`, `body.update` is
skipped, `_boundary_acceptance_metadata` returns `{}`, and the artifact body,
its `report_sha256`, the compiled metadata and every digest downstream are
untouched. There is no "less good but byte-safe" trade to make here: the
byte-safe option and the complete option are the same option.

**Proof, three ways.**

*(a) Module-level digest pins, before and after the edit, same process, same
seeds* (`…/scratchpad/baseline_digests.py`, captured on the unmodified tree and
re-run on the final tree — all eleven MATCH):

```
policy_unguarded            391318199830753ee60399a0725b7d774be5027760d62741b0ed82347f14eb55
policy_guarded              a433e5ba74f70cf1399075228ec4cbaf6ec20e36658f8d00d3fb95f17df671af
policy_guarded_tol          c2aad8534acb733f5367d23136b1c13a63cd245aaa8a7a1827e826b12c9e01ff
policy_fail_closed          e9c9e4027b4cd1c93b7d59e4be89e276da79481d854f3428f410412b0aab7806
policy_recaptured           d08f0791dbe77a200efde5fcbf7b8863d8c49dd24a38c3f9574358424a1b6bc1
policy_recaptured_guarded   f773d128411dafa907982abf623ee877dbab311096aa5fa6444423b115ac653d
toy7_unguarded_report       2913fac74507b99334e128f119137a44d78b31cac300b58bba33ed8fd8707509
toy7_guarded_report         dadcd627668a4d43050cf05a6ac8e5790450ba70f41268a9b1ecef99aa25c07e
toy7_guarded_ledger         c5a68e9fbdb091822fd4d50e8d3c0b02f65f87533638a14e9f4d77743e434bdc
+ both replacement_weight_sha256 lists
```

*(b) Adapter-level, on the real `moving_gauge_v3` toy compile.* Measured on the
unmodified tree first: the unguarded body-reclosure compiled metadata has exactly
15 keys, `moving_gauge_replacement_state_sha256` is
`ba967c782a1c4bd89b323fc8957fe876427127b3ecea2d4cd05808e9eb05cc0c` and the
validation decision is `155a9c643c4043aaf77625ba5c069a22abf0f6fcbe24fc1f0997d7e84cffdd36`.
All three are unchanged after the edit.
`test_unguarded_body_reclosure_lifts_no_acceptance_keys` pins that 15-key set as
a tuple and additionally asserts `_lifted_boundary_acceptance_ledger(...) is
None` and `_boundary_acceptance_metadata(...) == {}` on the real unguarded
replacement state.

*(c) Negative result, recorded because it is the honest half.* The toy's
body-reclosure `artifact_sha256` is **not** reproducible across processes — the
same test on the **unmodified** tree produced `71b2fbb5…` then `ad005b5f…` on two
consecutive runs, so it was never a byte-identity pin and is not used as one
here. The stable digests in (b) are.

### 1.3 Tests

`tests/test_hf_lfm2_dynamic_closed_loop_adapter.py`

* `test_unguarded_body_reclosure_lifts_no_acceptance_keys` — the default-mode
  contract above.
* `test_guarded_body_reclosure_lifts_the_sealed_acceptance_ledger` — a complete
  guarded `moving_gauge_v3` body reclosure: the artifact carries the ledger bound
  to the digest the reclosure published for it, the four decisions each carry
  both write errors, the norm ratio and the per-parameter Frobenius ratios, and
  the compiled metadata's JSON replays to the same accept/ratio sequence.
* `_adapter(...)` gained three additive kwargs so the guarded lanes are
  expressible at toy scale.

---

## 2. Fix 2 — item-8 depth telemetry read a field that does not exist

### 2.1 What landed

`experiments/run_lfm2_dynamic_catalog.py`

* `:10967` — new `_closed_loop_transaction_report(run)`: walks `run.attempts`,
  takes the first attempt whose `transaction_report` carries a `closed_loop`
  mapping. This is the traversal the call site already uses forty lines earlier
  for `vocabulary_margin_candidate_gate`.
* `:10988` — `_closed_loop_depth_telemetry(config, run, evaluator)`; the
  parameter is the fold **run**, not the plan.
* `:11016` — reads `closed_loop["layer_reports"]`, the key
  `dynamic_closed_loop_fold.py:3290` actually emits. `layers` never existed.
* `:12853` — the call site passes `run`, which was already bound at `:12413` from
  `result.published.compilation.run`.

Both halves of W13M §3 are closed: `DynamicFoldPlan` is a `slots=True` frozen
dataclass with no `report` field, so `getattr(plan, "report", None)` returned
`None` and the whole block was dead; and even reached, `.get("layers")` would
have returned `None`.

### 2.2 The test, rebuilt on the real emitter

The old `_plan_with_closed_loop_report()` was a hand-written `SimpleNamespace`
matching neither the broken read nor the real shape — it made a dead function
look tested. Replaced in
`tests/test_run_lfm2_dynamic_catalog_r13_retention.py`:

* `_real_closed_loop_report()` runs the **real** `ChildClosedLoopCompiler` over
  `test_dynamic_closed_loop_fold._ToyAdapter` with `fold_body_reclosure=True`.
  The fixture is the emitter's own output, so a key rename breaks it.
* `_fold_run_carrying(report)` wraps it in a **real** `DynamicFoldRun` /
  `FoldAttempt` / `ExperimentSignature` / `FoldDecision`, i.e. the containers
  the catalog actually receives.
* `test_depth_telemetry_reads_the_shapes_the_program_really_emits` pins the
  four claims that were guessed before:
  `DynamicFoldCompilation.__dataclass_fields__["plan"].type` is
  `DynamicFoldPlan` and that class has no `report`; the `run` field is
  `DynamicFoldRun` and its `attempts` carry `transaction_report`; the real
  report has `layer_reports` and does **not** have `layers`; and each entry's
  `scope`/`validation` key sets are supersets of the real
  `FoldScope(...).as_dict()` and `ValidationDecision(...).as_dict()`.
* `test_depth_telemetry_reconstructs_the_depth_profile` now asserts against the
  real report rather than literals: same layer count, same layer-index order,
  same `accepted`/`reason`/`decision_sha256`/`metrics` per layer, and
  `body_reclosure_validation` equal to the compiler's own body-reclosure
  validation.
* `test_depth_telemetry_survives_a_missing_closed_loop_report` is parametrized
  over four real degenerate shapes (no run, no attempts, no `closed_loop` key,
  `closed_loop: None`).

---

## 3. Fix 3 — the norm-ratio ceiling

### 3.1 What landed

`src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py`

| site | |
|---|---|
| `:86` | `LFM2_MOVING_GAUGE_WRITE_NORM_CEILING_SCHEMA` |
| `:132-133` | two distinct refusal reasons: `boundary_write_error_regressed`, `write_norm_ratio_above_ceiling` |
| `:1149` | `Lfm2MovingGaugeReclosurePolicy.writer_reclosure_norm_ratio_ceiling: float \| None = None` |
| `:1201-1218` | validation: finite positive, and **only** with a guarded acceptance mode (an unguarded lane never scores a proposal, so a ceiling there would claim a clause that can never run) |
| `:1280-1298` | `authentication_body` is additive one level below the acceptance block — an unset ceiling returns the pre-existing guarded body untouched |
| `:4346-4362` | the clause itself: `ceiling_exceeded` on the `write_norm_ratio` the decision already recorded; `accepted = boundary_accepted and not ceiling_exceeded` |
| `:4409-4424` | six additive decision keys, present **only** when the ceiling is set |
| `:4426-4438` | fail-closed raises a distinct message naming the ratio and the ceiling |
| `:5352-5397` | additive ledger keys split by cause |
| `:5615-5647` | additive publication-body keys and summary rows |

The clause is deliberately narrow: it reads `write_norm_ratio`
(proposed / incumbent branch-write L2 norm), which the decisions already
recorded as evidence and never gated on, and refuses **regardless of the
boundary metric**. Attribution is one-way: a refusal is `boundary_...` if the
metric clause refused it, and `write_norm_ratio_above_ceiling` only if the
metric clause *accepted* it — so
`norm_ratio_ceiling_refused_writer_overwrites` counts exactly what the ceiling
added.

**The one judgement call, stated plainly.** `_receipt_ratio` returns `None` when
the incumbent write norm is exactly zero. A ratio that does not exist is not
above any ceiling, so those are accepted, not refused — refusing them would
permanently freeze a from-zero seed writer and make the ceiling unusable. They
are never silent: the ledger carries
`unmeasured_write_norm_ratio_writer_overwrites`, and the policy body asserts
`unmeasurable_write_norm_ratio_is_not_above_the_ceiling: true`. In production
this case does not arise — the forward sweep's incumbent is the real LFM2 seed
weight. In the depth toy it is all three forward decisions, because that toy
seeds its writers at zero.

Plumbing, matching the existing `reverse_sweep_mode` / acceptance pattern
exactly:

* `hf_lfm2_dynamic_closed_loop_adapter.py:1376`
  `moving_gauge_writer_reclosure_norm_ratio_ceiling`, validated at `:1769-1791`,
  mirrored at `:3553`, threaded into the policy at `:14625`.
* `experiments/run_lfm2_dynamic_catalog.py:1258`
  `dense_moving_gauge_writer_norm_ratio_ceiling`, validated at `:1898-1920`,
  **conditional** `identity()` entry at `:3336-3348`, adapter config at
  `:10165`, flag `--dense-moving-gauge-writer-norm-ratio-ceiling` at `:14098`,
  `config_from_args` at `:15012`.

### 3.2 The toy numbers

Depth toy `_toy_fold(7)`, three transitions, writers seeded at zero. All ratios
below are `write_norm_ratio` from the ledger itself.

**The unguarded publication, measured** (guarded at `boundary_regression_tolerance = 1e9`,
which is byte-identical to unguarded — asserted):

| sweep | transition | ratio | metric ratio | accepted |
|---|---|---|---|---|
| forward | 0 | *unmeasured* (seed writes 0) | 0.6795 | yes |
| forward | 1 | *unmeasured* | 0.4999 | yes |
| forward | 2 | *unmeasured* | 0.6358 | yes |
| reverse | 2 | 1.0000000198 | 1.0000 | yes |
| reverse | 1 | **2.1542** | 2.7902 | yes |
| reverse | 0 | 1.1163 | 1.1066 | yes |

**Same lane plus `writer_reclosure_norm_ratio_ceiling = 1.5`:**

```
refused_writer_overwrites                        2
refused_by_sweep                                 {"forward": 0, "reverse": 2}
norm_ratio_ceiling_refused_writer_overwrites     2
norm_ratio_ceiling_refused_transition_indices    [0, 1]
boundary_metric_refused_writer_overwrites        0     <- the metric refused nothing
unmeasured_write_norm_ratio_writer_overwrites    3
maximum_accepted_write_norm_ratio                1.0000000198
reverse 1  REFUSED  write_norm_ratio_above_ceiling  ratio 2.1542
reverse 0  REFUSED  write_norm_ratio_above_ceiling  ratio 3.3266
```

Transition 0's ratio moves 1.1163 → 3.3266 because refusing transition 1 changes
the state transition 0 then solves from; 3.3266 is exactly the value the strict
metric lane sees. The published weights differ from the unguarded lane and match
the strict metric lane's, and the two refused reverse steps realize
`realized_terminal_l2_norm == 0.0` — a refusal publishes nothing.

**Composition, both clauses on** (`require_boundary_improvement` at tolerance 0
plus ceiling 1.5): same two transitions refused, same published weights, but
attributed `boundary_metric_refused_writer_overwrites: 2`,
`norm_ratio_ceiling_refused_writer_overwrites: 0` — the metric clause got there
first and the ledger says so. The composed policy is its own `policy_sha256`.

**The sharper case**, because "the toy at unbounded tolerance" only shows the
metric *declining* to refuse, not being *unable* to. `_norm_exploding_proposal()`
constructs one writer against a 10x target: the incumbent writes 1x (relative
error 9), the proposal writes 8x (relative error 2). The boundary metric
**improves by 4.5x** — `relative_write_error_ratio == 2/9` — while
`write_norm_ratio == 8.0` exactly. Ceiling unset: accepted. Ceiling 1.5: refused,
`refusal_reason == "write_norm_ratio_above_ceiling"`,
`boundary_metric_accepted is True`. Ceiling 8.0: accepted. Under
`require_boundary_improvement_fail_closed` it raises "exceeds the ceiling".
That is r12's failure signature in four lines of tensors, and it is the case no
boundary metric can ever catch.

**Adapter end-to-end.** The `moving_gauge_v3` toy compile's four decisions have
ratios `[1.4843, 1.7029, 1.0001, 1.0000]`; at ceiling 1.5 the reclosure refuses
the 1.7029 one, the metric clause refuses nothing, and
`writer_reclosure_norm_ratio_ceiling` /
`boundary_acceptance_norm_ceiling_refusals` /
`boundary_acceptance_maximum_accepted_write_norm_ratio` all reach the run
report's compiled metadata through the Fix-1 lift.

### 3.3 Default-`None` byte identity

`test_unset_norm_ratio_ceiling_publishes_the_pinned_guarded_bytes` pins the
guarded toy publication at `dadcd627…` and its ledger at `c5a68e9f…` — the
values captured on the unmodified tree — for both an omitted ceiling and an
explicit `None`, and asserts that no ceiling key appears anywhere in the sealed
report, the ledger, or any decision.
`test_norm_ratio_ceiling_is_unset_by_default_and_additive_in_the_digest` pins the
guarded policy digest at `a433e5ba…`, the unguarded at
`391318199830…`, and proves five distinct digests across the ceiling/mode/reverse
-sweep combinations.

---

## 4. Verification

Runtime `.venv/Scripts/python.exe` (3.11.9, torch 2.13.0+cpu), CPU only.

```
ruff check src/ experiments/ tests/
   39 errors, all pre-existing and none in a file this wave touched.
   The only touched file that appears is experiments/run_lfm2_dynamic_catalog.py
   with its 3 known F401s -- byte-identical to the baseline captured before the
   first edit.

pytest tests/test_hf_lfm2_moving_gauge_residual_reclosure.py -q -p no:randomly
   82 passed                                     (77 before, 5 new)

pytest tests/test_hf_lfm2_dynamic_closed_loop_adapter.py -q -p no:randomly
   74 passed in 304.28s                          (70 before, 4 new)

pytest tests/test_run_lfm2_dynamic_catalog.py -q -p no:randomly
   119 passed, 1 skipped in 53.95s               (118 before, 1 new)

pytest tests/test_run_lfm2_dynamic_catalog_r13_retention.py -q -p no:randomly
   14 passed                                     (13 before; 3 rewritten,
                                                  1 new, 1 parametrized x4)

pytest tests/test_dynamic_closed_loop_fold.py \
       tests/test_hf_lfm2_r13_retention_lane.py \
       tests/test_hf_lfm2_moving_gauge_stack_integration.py \
       tests/test_hf_lfm2_moving_gauge_input_producer_reclosure.py \
       tests/test_hf_lfm2_moving_gauge_causal_targets.py \
       tests/test_run_lfm2_dynamic_catalog_portable_runtime.py \
       tests/test_run_lfm2_dynamic_catalog_tokenizer_bridge.py \
       tests/test_dynamic_fold_artifact.py -q -p no:randomly
   189 passed

pytest tests/test_dynamic_lfm2_component_dispatch.py \
       tests/test_dynamic_lfm2_dense_budget_fold_stack.py \
       tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
       tests/test_lfm2_exact4x_moving_gauge_v5_candidate_materializer.py \
       tests/test_run_lfm2_exact4x_moving_gauge_v5_live.py \
       tests/test_run_lfm2_weight_complete_closed_loop.py -q -p no:randomly
   80 passed

pytest tests/test_hf_lfm2_closed_loop_restart_capsule.py \
       tests/test_hf_lfm2_pre_causal_body_resume_capsule.py \
       tests/test_hf_lfm2_mapped_causal_target_capsule.py -q -p no:randomly
   71 passed
```

The last set is the every-file-that-imports the reclosure policy, the adapter
config or the depth telemetry sweep, plus the capsule suites, because the
body-reclosure artifact digest feeds restart capsules.

### 4.1 The three known-red carrier e2e tests

```
pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly
   3 failed, 6 passed in 282.24s

   test_carrier_dense_core_alternates_fold_round_and_refold
   test_bridged_carrier_lane_alternates_fold_round_and_refold
   test_carrier_production_margin_pin_composes_through_a_round
```

Same three names, same 3/6 split, and the same failure `NOTES-W13L` §7 and
`NOTES-W13M` §6.5 attribute to `d296b602`:

```
AssertionError: {'category': 'alternation_pass_failed_closed', 'pass_index': 1,
 'pass_failure': {… 'Lfm2DynamicClosedLoopAdapterError',
 'message': 'Selected parent component payload changed after authenticated load.'}]},
 'fail_closed': True}
assert 'failed_closed' == 'complete'
```

Not this wave's. `d296b602` remains unfixed and this file does not touch it.

**Line endings.** Every file in this repo is CRLF in the working copy with
`core.autocrlf=true` and `eol=lf` in `.gitattributes`. All six edited files were
100 % CRLF before and are 100 % CRLF after; `git diff --stat` is 1 423 insertions
/ 99 deletions, not tens of thousands. `NOTES-W13M` §6.5's LF/CRLF trap did not
recur.

---

## 5. What is not verified, and what was left out

* **The Fix-2 call site has no end-to-end coverage.** `_closed_loop_depth_telemetry`
  is unit-tested against the real containers and the real emitter, and the plan
  and run field annotations on `DynamicFoldCompilation` are asserted, but no test
  runs `run_production_catalog` with `--dense-closed-loop-depth-telemetry` and
  reads the resulting receipt. It had no such coverage before either, which is
  exactly how a dead read survived. This is the weakest link in the wave and
  should be stated to whoever launches r14.
* **Fix 1 and Fix 3 have never run at production scale.** Everything above is
  toy-scale CPU. The r13 body reclosure ran 15 boundaries and 28 physical
  writers; the toys run 2-3.
* **The ceiling has no recommended value.** W13M measured a median 33x and a
  974x. Nothing here argues for 1.5, or 2, or 10 — the toy uses 1.5 because that
  is the value that separates its own rewrites. Choosing a production ceiling is
  an r14 decision and it should be made against measured ratios, not by analogy
  to the toy.
* **A ceiling refusal is not proof of a better child.** Refusing keeps the
  incumbent writer, which the boundary metric may have wanted to replace. On the
  depth toy at unbounded tolerance the ceiling and the metric clause happen to
  publish identical weights; there is no general reason for that.
* **Unmeasurable ratios are accepted** (§3.1). Counted, asserted, and stated in
  the policy body — but accepted.
* **Not touched, as scoped:** `hf_lfm2_dense_budget_model_fold.py`,
  `dynamic_closed_loop_fold.py`, `run_lfm2_exact4x_moving_gauge_v5*`,
  `hf_lfm2_on_policy_evidence_round.py`, the frozen collector,
  `dynamic_fold_artifact.py`.
* **Nothing committed.** No `git add`, no commit, no push.

## 6. Files

| path | |
|---|---|
| `src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py` | Fix 3 policy + gate |
| `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` | Fix 1 lift, Fix 3 config |
| `experiments/run_lfm2_dynamic_catalog.py` | Fix 2 telemetry, Fix 3 flag |
| `tests/test_hf_lfm2_moving_gauge_residual_reclosure.py` | 5 new ceiling tests |
| `tests/test_hf_lfm2_dynamic_closed_loop_adapter.py` | 4 new lift/ceiling tests |
| `tests/test_run_lfm2_dynamic_catalog.py` | 1 new flag test |
| `tests/test_run_lfm2_dynamic_catalog_r13_retention.py` | depth-telemetry tests rebuilt |
| `…/scratchpad/backup-20260807-w13n/` | pre-edit copies of all six + `HEAD.txt` |
| `…/scratchpad/baseline_digests.{py,json}` | the pre-change digest pins in §1.2 |
