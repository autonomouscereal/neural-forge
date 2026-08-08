# NOTES-W13U — the carrier ABI becomes a two-entry allowlist

Wave 13U, branch `codex/lrc-blank-compression`, local HEAD `9326a564`.
Source change only. **No `git add`, no commit, no push, nothing deployed.**
The exact-2x arm is now *expressible*; it has still never been run.

This implements the change NOTES-W13T §3.3 prepared, as an **allowlist
extension, not a relaxation**: the three sites that pinned one 4x tuple now
admit exactly two, matched coherently, and refuse everything else with a
message that names both.

| profile id | ratio | counted total | dense body | hidden |
|---|---|---|---|---|
| `exact-4x-h816` | 4.0 | 57,420,144 | 53,552,368 | 816 |
| `exact-2x-h816` | 2.0 | 114,846,592 | 110,978,816 | 816 |

Both keep the untied 4096-row H816 child ABI, unchanged. Both reserve the same
fixed `2 x 1,933,888` route subtraction, unchanged.

---

## 1. What was implemented

### 1.1 The sealed allowlist — `src/neural_forge/lfm2_carrier_child_package.py`

The ABI's canonical home already held `CURRENT_LFM2_CARRIER_*`; the allowlist
lives beside them so the catalog runner has one thing to import.

| site | |
|---|---|
| `lfm2_carrier_child_package.py:65` | `Lfm2CarrierCountedAbiProfile` — frozen `(profile_id, compression_ratio, counted_target_total_parameter_elements, counted_dense_body_parameter_elements, hidden_size)` with `matches(...)` and `describe()` |
| `:109`, `:126` | `LFM2_CARRIER_EXACT_4X_COUNTED_ABI`, `LFM2_CARRIER_EXACT_2X_COUNTED_ABI` |
| `:135` | `ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES` — the complete admitted set, a tuple |
| `:153` | `_seal_admitted_carrier_counted_abi()` — runs at import; refuses to load an allowlist whose own arithmetic does not close |
| `:189` | `describe_admitted_lfm2_carrier_counted_abi()` — renders both tuples for a refusal |
| `:198` | `admitted_lfm2_carrier_counted_abi_profile(...)` — returns the one profile a request *entirely* matches, or `None` |

`matches()` is all-or-nothing across four fields. There is no per-field
comparison anywhere in the new code, which is what makes a cross-field mixture
inexpressible rather than merely discouraged.

### 1.2 The runtime re-pin — same file

`current_exact4x_counted_plan()` (`:758`) is now a two-line wrapper over
`admitted_counted_plan(profile)` (`:722`). The general method keeps both of the
old guard's jobs and adds one:

* the profile must be **an element of** `ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES`
  by identity — an equal-valued tuple a caller assembles is refused, so the
  allowlist cannot be widened from a call site;
* the plan the live routes produce must still reserve that profile's exact
  dense body, which is the original `:564-575` assertion, now parameterized.

### 1.3 The config guards — `experiments/run_lfm2_dynamic_catalog.py`

| before | after |
|---|---|
| `:2237-2252` pinned `math.isclose(compression_ratio, 4.0, rel_tol=0, abs_tol=0)` **and** the three topology booleans | `:2320-2329` keeps `compression_ratio is not None` and the three booleans; the *value* moved into the tuple check. Same message. |
| `:2272-2287` pinned `total == 57_420_144`, `body == 53_552_368`, plus two structural invariants | `:2357-2379` calls the allowlist on `(ratio, total, body, carrier_expected_hidden_size)` and keeps both structural invariants verbatim. New message names both tuples. |
| `:2288-2295` untied 4096-row H816 | **unchanged** (see the delta in §2) |

New message:

```
Carrier-routed production must reserve both independent route channels inside
the counted target and give the remainder to the physical body at one admitted
carrier ABI: ratio 4 / counted total 57,420,144 / dense body 53,552,368 /
hidden 816; ratio 2 / counted total 114,846,592 / dense body 110,978,816 /
hidden 816.
```

`ProductionLfm2CatalogConfig.carrier_counted_abi_profile` (`:3164`) resolves the
run's one admitted tuple. Configuration already refused everything else, so it
resolves rather than decides; it raises for a non-carrier config. Every later
re-pin reads the run's profile from there instead of a module constant:

* `_build_registered_carrier_counted_plan(profile=…)` — `:1030`, called at `:11966`
* `_authenticated_carrier_phase_dependencies(profile=…)` — `:10105`, called at `:12529`
  (this re-derives the counted plan on **every** balanced-basis phase-cache
  authentication, so it had to move too)
* `CarrierRouteSmokeBehaviorEvaluator(counted_abi_profile=…)` — `:8755`,
  constructed at `:12502`; the routed ABI smoke's `expected_counts` (`:8895`)
  now come from the profile

`LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS` / `_DENSE_BODY_…` / `_HIDDEN_SIZE`
are now derived from the 4x profile rather than re-declared, so the 4x numbers
have exactly one definition. They remain the argparse and dataclass defaults —
a run that passes no budget flags is still a 4x run.

---

## 2. Delta from NOTES-W13T's prepared change

W13T §3.3 specified four parts. Three of them differ from the allowlist design,
and the allowlist design won. One site W13T did not list turned out to matter.

**Part 1 (the ratio pin).** W13T offered "a consistency check against
`parent_unique / counted_target_total`, or drop the ratio clause and keep the
three topology booleans." Neither was taken. Dropping the clause loses the
ratio entirely; a derived consistency check cannot be satisfied by the 4x lane
(see §3). The ratio is instead one field of the coherent tuple, so
`--compression-ratio 2` with the 4x budget still refuses — W13T §3 probe 2's
result is preserved for a better reason.

**Part 2 (the counted-totals pin).** W13T proposed replacing the two constant
equalities with "the invariant that is actually the ABI:
`total - body == 2 * ROUTE_ELEMENTS_PER_ROLE` and
`dense_heterogeneous_target_parameters == counted_dense_body`." **That is a
relaxation, not an allowlist**: it admits any budget at all whose route
subtraction is right — including the exact quarter (57,423,296 / 53,555,520),
which W13T §2.2 itself flags as unproven end to end. The equalities were kept
and turned into a two-entry lookup. Both of W13T's invariants are still checked,
in the same `if`, as belt-and-braces over the profile.

**Part 3 (the runtime re-pin).** W13T said "parameterized on the requested
total." Parameterizing on the *total* would let any total through the re-pin and
rely on the config guard alone. It is parameterized on the profile **object**,
with an identity check against the sealed tuple, so the re-pin is an independent
fail-closed boundary rather than a mirror of the config's decision.

**Part 4 (H816).** W13T: "`:2288-2295` needs no change if the arm keeps H816."
Correct, and it is byte-for-byte unchanged. But `hidden_size` is a field of the
coherent tuple, so for an H1024 request the allowlist guard now fires **first**,
with the message naming both admitted tuples, and `:2288-2295` is reachable only
via a wrong `--carrier-expected-target-rows`. This is the design's "cross-field,
not per-field" requirement doing its job; the H816 requirement is enforced
earlier and identically, not removed.

**A fifth site W13T did not list.** `CarrierRouteSmokeBehaviorEvaluator._route_smoke`
compared the live child's observed `(body, C_in, C_out, effective total)` against
`LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS` / `…TARGET_TOTAL…` directly. W13T's
guard table (§3.1) has six rows and this is not one of them. It would have
refused a real 2x child at the first routed ABI smoke — exactly the class of
failure W13T §7 predicted ("if 1–3 are relaxed and 5 is missed, the run will
fail closed minutes in"), one site further out than the notes named. Extended.

**W13T guard-table row 5** (`run_lfm2_dynamic_catalog.py:11818-11845`) needed no
edit of its own: it already reads `config.counted_dense_body_parameters` and the
per-role route constant, both of which are correct for either profile. Only the
`profile=` argument threading through `_build_registered_carrier_counted_plan`
changed.

---

## 3. One thing the notes had wrong, found by the seal

`_seal_admitted_carrier_counted_abi()` originally required
`parent == round(ratio * total)`. It refused the **4x** profile at import:

```
229,693,184 / 57,420,144 = 4.0002195745
```

The declared `--compression-ratio 4` on the sealed 4x lane is a **rung label,
not a derived quantity** — W13T §2.2 records the 3,152-element gap between
57,420,144 and the exact quarter and flags it as unexplained, but the
consequence for the ratio field was not drawn. This is why W13T part 1's
"consistency check against `parent_unique / counted_target_total`" would have
broken the proven lane on the first run.

The seal now allows a 0.1% label tolerance
(`_ADMITTED_CARRIER_RATIO_LABEL_TOLERANCE`, `:150`), which admits 4.0002195745
labelled `4.0` and would still refuse a profile whose label misdescribes its own
budget. The exact-2x rung needs no tolerance: 229,693,184 is even and the half
is exact.

---

## 4. The refused-layer telemetry fix (NOTES-W13S §6.4)

The gap: R1/R2/R3 per-solve numbers reached disk only through an **accepted**
layer's completion receipt. A layer the terminal floor refused published no
restart capsule and no completion receipt, so the requested-versus-allowed
prior-delta ratios, the effective ridge and λ·σ_max were computed and then
discarded. The arm that most needed them was the one that could not see them.

Three links, all minimal:

1. **`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py:8447`** —
   `_refused_layer_prior_correction_diagnostics(scope)` reads the layer's own
   producer / consumer / downstream payloads, which are still resident at
   refusal time, and serializes every `prior_correction_reports` entry through
   the existing `_durable_diagnostic_json` encoder.
   **`:8737`** — `_validation_decision` attaches them to `metrics` when and only
   when the decision is **not** accepted. Accepted decisions are byte-identical,
   so no sealed capsule, completion receipt or decision digest moves.
   Every lookup in the helper is optional: this runs on the refusal path and a
   diagnostic that raised would replace a clean fail-closed rejection with an
   infrastructure crash.
2. **`experiments/run_lfm2_dynamic_catalog.py:645`** — `_ProgressJournal` observes
   `child_closed_loop_validation_decision_rejected` and decodes the reports out
   of `validation_metrics_json`. This follows the journal's existing precedent
   for replay-shard telemetry ("the journal is the one place every such event
   already passes through"). `refused_layer_solver_diagnostics()` at `:683`.
3. **`:13619`** — the fail-closed receipt's `failure` block gains
   `refused_layer_solver_diagnostics`, `None` when no layer was refused.

`dynamic_closed_loop_fold.py` is untouched, which is why the event name is
re-declared as `CHILD_CLOSED_LOOP_VALIDATION_REJECTED_STAGE` (`:589`) in the
runner that reads it rather than exported from the module that emits it.

Verified on a **real** floor-refused toy layer, not a stub — the R13 retention
lane's existing wall test now reads its own refusal:

```
metrics["refused_layer_index"] == 1
metrics["refused_layer_prior_correction_solve_count"] > 0
reports[stage][i]["prior_delta_clamp"]["requested_prior_delta_ratio"]  is not None
reports[stage][i]["prior_delta_clamp"]["allowed_prior_delta_ratio"]    is not None
```

---

## 5. Tests

Moved (the four NOTES-W13T §3.3 named) and added:

| file | what |
|---|---|
| `tests/test_run_lfm2_dynamic_catalog.py` | `test_carrier_mode_rejects_legacy_full_body_budget` — **moved** to the new admitted-tuple message |
| | `test_carrier_abi_allowlist_admits_the_4x_and_exact_2x_tuples` — **new**, both tuples construct end to end at config level (W13T §3 probe 6 plus probe 1) |
| | `test_carrier_abi_allowlist_refuses_cross_field_tuples` — **new**, 2x ratio + 4x totals, 4x ratio + 2x totals, 2x totals + H1024 |
| | `test_carrier_abi_allowlist_still_refuses_an_unlisted_budget` — **new**, the exact quarter |
| | `test_carrier_route_smoke_pins_the_run_s_own_admitted_counts` — **new** |
| | `test_refused_layer_solver_diagnostics_reach_the_failure_receipt` — **new**, journal harvest + receipt field |
| | `test_carrier_phase_dependencies_…`, `test_carrier_counted_plan_…`, `test_carrier_route_smoke_is_structural_…` — updated for the new signatures |
| `tests/test_lfm2_carrier_child_package_fresh_process.py` | `test_counted_plan_re_pin_admits_both_sealed_carrier_budgets` — **new** |
| | `test_counted_plan_re_pin_refuses_a_mismatched_package` — **new**, routes of the wrong size reserve neither body |
| | `test_counted_plan_re_pin_refuses_an_unlisted_profile` — **new**, an equal-valued forged tuple |
| | `test_admitted_carrier_abi_lookup_is_coherent_not_per_field` — **new**, 2 admitted / 6 refused |
| `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` | `test_authenticated_carrier_routes_stop_at_the_sealed_production_abi` — **moved** to the new message, plus an assertion that the refusal names both tuples |
| `tests/test_hf_lfm2_r13_retention_lane.py` | `test_a_run_headed_past_the_no_branch_baseline_fails_closed_at_the_wall` — extended with the requested/allowed ratio assertions |

Runner `.venv/Scripts/python.exe -m pytest … -q -p no:randomly` (Python 3.11.9,
torch 2.13.0+cpu; the `C:/Python314` interpreter has no torch).

| suite | result |
|---|---|
| `tests/test_run_lfm2_dynamic_catalog.py` | **124 passed, 1 skipped** (119/1 before; +5) |
| `tests/test_lfm2_carrier_child_package_fresh_process.py` | **8 passed** (4 before; +4) |
| `tests/test_hf_lfm2_r13_retention_lane.py` + `tests/test_run_lfm2_dynamic_catalog_r13_retention.py` | **78 passed** |
| `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` | **6 passed, 3 failed — known red** |
| `tests/test_hf_lfm2_dynamic_closed_loop_adapter.py` | **74 passed** |
| `tests/test_dynamic_closed_loop_fold.py` + `tests/test_dynamic_counted_child_state.py` | passed (104 with the adapter suite) |
| `test_lfm2_carrier_portable_runtime` + `…_catalog_portable_runtime` + `…_tokenizer_bridge` + `test_run_lfm2_carrier_reclosure` + `test_audit_lfm2_exact4x_first_divergence` + `test_run_lfm2_exact4x_moving_gauge_v5` | **87 passed** |

The three carrier-e2e reds are the `d296b602` set:
`test_carrier_dense_core_alternates_fold_round_and_refold`,
`test_bridged_carrier_lane_alternates_fold_round_and_refold`,
`test_carrier_production_margin_pin_composes_through_a_round`, all failing on
`Selected parent component payload changed after authenticated load.`
**Verified pre-existing, not assumed**: the working tree was stashed
(`git stash push -- <the four edited source/test files>`),
`test_carrier_production_margin_pin_composes_through_a_round` was re-run at clean
HEAD, failed identically, and the stash was popped.

Ruff: `ruff check` over all seven edited files reports **3 errors**, all
pre-existing `F401`s on unused on-policy imports in
`experiments/run_lfm2_dynamic_catalog.py:161,163,174`. Verified against the same
three source files extracted from HEAD (`git show HEAD:… > …`), which report the
same 3. `ruff check tests/` reports 31 both with the four edited test files and
with them stashed back to HEAD. **Zero new findings.**

---

## 6. Deliberately left out

* **`config.identity()` is unchanged.** No `profile_id` field was added. The
  identity feeds the global-topology and balanced-basis phase keys; a new field
  rotates every 4x phase key and drops the inherited `a0025` basis payload for
  no gain. `identity()["carrier_routes"]` already carries
  `counted_target_total_parameters`, `counted_dense_body_parameters`,
  `continuous_route_parameters` and `expected_hidden_size`, so the two profiles
  are already identity-distinct.
* **`CarrierRouteSmokeBehaviorEvaluator`'s gate manifest body is unchanged**, for
  the same reason: adding the profile would rotate `manifest_fingerprint` and
  the `BehaviorGateBinding` on every published 4x receipt. The evaluator holds
  the profile and enforces it; it does not restate it in the sealed manifest.
* **argparse and dataclass defaults stay 4x.** A run that passes no budget flags
  is a 4x run, exactly as before.
* **`experiments/run_lfm2_exact4x_moving_gauge_v5.py` and
  `src/neural_forge/lfm2_exact4x_moving_gauge_v5_candidate_materializer.py` are
  untouched.** They pin `CURRENT_LFM2_CARRIER_{TARGET,DENSE}_PARAMETER_ELEMENTS`
  in nine places. Those constants still hold the 4x numbers, so v5 is unaffected
  — and a 2x arm cannot be driven through the v5 runner without a further change
  that was out of scope here.
* The frozen collector, `hf_lfm2_dense_budget_model_fold.py`,
  `dynamic_closed_loop_fold.py` and the on-policy round are untouched.

## 7. What could not be verified

* **No 2x fold was run, at any scale.** The exact-2x tuple is proven to
  construct at configuration time, to resolve a profile, to produce the right
  counted plan against the real 1,933,888-element route channels, and to arm the
  routed ABI smoke. It has never produced a child. The carrier e2e lane's toy
  fixtures are built around the 4x numbers, so there is no toy 2x end-to-end
  path short of new fixtures.
* **`_route_smoke`'s 2x branch is covered at construction only** — profile
  stored, unlisted profile refused. No 2x child exists to smoke against, so the
  expected-count comparison itself is exercised only on the 4x path.
* **The failure-receipt link is covered in two pieces**, not one live floor
  refusal through the runner: a real refusal proves the adapter serializes the
  reports (retention lane), and a seeded journal proves
  `run_production_catalog`'s receipt carries them. Producing a real floor-refused
  layer *inside* `run_production_catalog` needs a full toy fold through the
  catalog, which no existing fixture provides.
* Nothing was deployed. `neural-forge-postpub-e893767a` is still at
  `SOURCE_COMMIT 5739c742…`; the W13T §3.3 bundle/cherry-pick/SOURCE_COMMIT-bump
  chain has **not** been performed.

## 8. Files changed

| path | +/- |
|---|---|
| `experiments/run_lfm2_dynamic_catalog.py` | +180 / -22 |
| `src/neural_forge/lfm2_carrier_child_package.py` | +194 / -4 |
| `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` | +68 / -0 |
| `tests/test_run_lfm2_dynamic_catalog.py` | +314 / -3 |
| `tests/test_lfm2_carrier_child_package_fresh_process.py` | +145 / -0 |
| `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py` | +19 / -4 |
| `tests/test_hf_lfm2_r13_retention_lane.py` | +21 / -0 |
| `NOTES-W13U.md` | this file |

No file was deleted, renamed, or backed up over. Nothing was staged or
committed.
