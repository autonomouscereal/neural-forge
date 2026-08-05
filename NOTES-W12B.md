# NOTES-W12B — reverse reclosure against recaptured states, and the two efficiency items

Wave 12B, branch `codex/lrc-blank-compression`, base `01bea537`.
Covers R13_PLAN item 6 and the efficiency section's install-sweep collapse,
fresh-reload dedup, and terminal restart capsule.

Line numbers below are against `01bea537` plus this wave's landed diff.

---

## 1. What landed (implemented and verified here)

| Item | Site | State |
|---|---|---|
| Item 6 — recaptured reverse sweep | `src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py` | landed, flag-gated, default byte-identical |
| Efficiency — fresh-reload dedup | `src/neural_forge/dynamic_fold_artifact.py` | landed, on by default, fail-closed |
| Efficiency — terminal restart capsule | `src/neural_forge/hf_lfm2_closed_loop_restart_capsule.py` + `src/neural_forge/dynamic_closed_loop_fold.py` + `src/neural_forge/hf_lfm2_dense_budget_model_fold.py` | publication landed (opt-in flag); CLI flag + resume-from-terminal specced in §4 |
| Efficiency — install-sweep collapse | `hf_lfm2_dynamic_closed_loop_adapter.py` | NOT implemented (file owned by the parallel agent) — spec in §2 |

The adapter (`hf_lfm2_dynamic_closed_loop_adapter.py`) and
`experiments/run_lfm2_dynamic_catalog.py` were not touched.

---

## 2. Install-sweep collapse — implementation spec

### 2.1 The exact regions

`src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py`:

- `install_body_reclosure` — `:11481`.
- Initial post-writer recapture — `:11704-11727`
  (`_body_internal_recapture(..., selected_layer_indices=(0,), body_internal_role="pre_operator")`,
  then `_authenticated_capture_sha256_and_release`, then `del` + `release_closed_loop_capture_memory()`).
- Per-layer sweep — `for layer_index, planned_step in enumerate(payload.causal_w2_step_receipts)` at `:11729`,
  containing exactly four recapture blocks per layer:
  - `post_operator_producer` — `:11823-11853`
  - `post_operator_consumer` — `:11862-11892`
  - `post_mlp_producer` — `:11900-11930`
  - `post_w2` — `:11938-11994` (this one also advances `current_capture_sha256`)
- Terminal publication authentication — `:11995-12040`
  (`_moving_gauge_writer_publication_matches`, producer fingerprint equality,
  `materially_changed`, `physical_after`, `state_after`).
- The preview that produced `payload.causal_w2_step_receipts` —
  `_compile_moving_gauge_causal_completion` at `:5275`, whose own recaptures are at
  `:5365`, `:5501`, `:5578`, `:5666`, `:5733`.

### 2.2 Why it is redundant today

Install performs **zero solving**. Every tensor it publishes came from the sealed
preview (`writer_replacements`, `producer_replacements`, `causal_operator_replacements`,
`causal_w2_replacements`), and each `planned_step` already pins
`replacement_weight_sha256`, `operator_producer_replacement_weight_sha256`,
`mlp_producer_replacement_weight_sha256` and the operator/mlp/w2 parameter ids
(checked at `:11757-11815`). The 1 + 4L recaptures exist only to re-derive a
capture-sha256 chain that the preview already sealed. At L=14 that is 57 sharded
replays; at r12's 7,777 s/replay-set this is the ~17.7 h line item.

What the sweep genuinely proves that fingerprints do not: that the *installed*
child, replayed end to end, reproduces the preview's activation captures — i.e.
that publication order and the physical parameter aliasing behave as previewed.
That property is worth keeping for certification runs, not for every run.

### 2.3 The flag

New adapter constructor / catalog flag:

```
--dense-moving-gauge-install-verification {fingerprint,full-recapture}
```

- default `fingerprint`
- `full-recapture` = today's behavior, byte-for-byte, including every
  `moving_gauge_install_recapture_started/completed` progress event.

Store as `self.install_verification_mode`, validated against a frozenset in
`__init__`, and add it to `_ADAPTER_IDENTITY_FIELDS`
(`hf_lfm2_closed_loop_restart_capsule.py:179`) so a restart capsule cannot resume
a `fingerprint` run into a `full-recapture` process or vice versa. That is a
capsule-schema-visible change: bump nothing, but note that adapter-identity
digests change for all new runs (old capsules keep authenticating because
`_open_capsule` recomputes the identity digest from the stored body — but
`identity_body`'s key set is checked exactly at `:1078`, so
`_ADAPTER_IDENTITY_FIELDS` must gain the key *and* `_open_capsule` must accept
manifests without it: use
`set(identity_body) in ({"schema", *_ADAPTER_IDENTITY_FIELDS}, {"schema", *_LEGACY_ADAPTER_IDENTITY_FIELDS})`).

### 2.4 `fingerprint` mode semantics

Per layer, in the same publication order, replacing each recapture block with a
pure-fingerprint authentication:

1. Publish the group (unchanged `_publish_dense_replacements` /
   `_publish_moving_gauge_writer_replacements` calls).
2. Immediately re-read each published physical parameter and require
   `fingerprint_lfm2_moving_gauge_tensor(parameter) == planned_step[<the matching>_weight_sha256]`.
   Today the equivalent check happens once at the end (`:12009-12025`); in
   `fingerprint` mode it must happen per step, so a mis-ordered publication is
   still caught at the step that caused it.
3. Record the step with `"capture_sha256": None` and
   `"verification": "replacement_weight_fingerprint"`, keeping the key set of
   `installation_steps` (`:11972-11991`) otherwise identical so downstream
   receipt consumers do not break. Add
   `"install_verification_mode"` to the install receipt body.
4. Emit `moving_gauge_install_fingerprint_verified` progress events in place of
   the recapture pair, with the same `layer_index` / `capture_boundary` fields,
   so the progress-chain consumers keep a per-boundary heartbeat.

After the loop, in **both** modes:

5. One terminal capture:
   `_body_internal_recapture(fit_records, audit, selected_layer_indices=(len(child_layers)-1,), body_internal_role="terminal_receipt")`
   and require its sha256 to equal the preview's
   `final_preview_capture_sha256` (`_compile_moving_gauge_causal_completion`,
   `:5848` / `:5905`). This is the single end-to-end proof that the installed
   child equals the previewed child, and it is what makes the collapse safe:
   1 replay instead of 1 + 4L.

Cost: 57 replays → 1. Expected saving ≈ 17.3 h of the 17.7 h.

### 2.5 What must stay for certification runs

- `--dense-moving-gauge-install-verification full-recapture` must remain able to
  reproduce the current capture chain exactly, including
  `pre_operator_capture_sha256` threading through `current_capture_sha256`
  (`:11756`, `:11992`).
- The final publication-equality block (`:11995-12040`) stays in both modes.
- The `payload.causal_w2_step_receipts` authentication (`:11732-11815`) stays in
  both modes — it is fingerprint work, not replay work.
- The install receipt must state the mode; a certification receipt claiming
  full-recapture with `installation_steps[*].capture_sha256 == None` must fail
  closed at receipt authentication.
- Any run whose receipt is used to argue *composition* (item 2's terminal floor)
  should use `full-recapture` at least once per topology change.

---

## 3. Item 6 — production wiring diff for the adapter

Item 6 landed in the reclosure module behind a distinct sealed policy. Nothing in
the adapter changes today: with `reverse_sweep_mode="forward_only_identity_transport"`
(the default) the compiled state, its diagnostics, and the policy digest are
byte-identical to `01bea537` (proof in §6).

To turn it on in the carrier lane, the adapter needs exactly this — no other
edit, and no new adapter method:

```diff
--- a/src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py
+++ b/src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py
@@ -11145,10 +11145,60 @@ class Lfm2DynamicClosedLoopAdapter:
-            replacement_state = compile_lfm2_moving_gauge_residual_reclosure(
-                tuple(boundaries),
-                tuple(writer_evidence),
-                policy=Lfm2MovingGaugeReclosurePolicy(),
-            )
+            if self.body_reverse_sweep_mode == "recaptured_terminal_anchored":
+                writer_input_widths = {
+                    site.parameter_id: int(site.seed_weight.shape[1])
+                    for site in writer_evidence
+                }
+
+                def _mid_sweep_recapture(request):
+                    """Publish the forward writers, replay, and measure."""
+
+                    snapshots = {
+                        name: parameter.detach()
+                        .to(device="cpu")
+                        .contiguous()
+                        .clone()
+                        for name, parameter in (
+                            self._moving_gauge_physical_writer_parameters(
+                                writer_sites
+                            ).items()
+                        )
+                    }
+                    try:
+                        self._publish_moving_gauge_writer_replacements(
+                            writer_sites,
+                            request["forward_replacement_weights"],
+                            tuple(request["request_body"]["parameter_ids"]),
+                            (),
+                        )
+                        recapture = self._body_internal_recapture(
+                            fit_records,
+                            audit,
+                            selected_layer_indices=tuple(
+                                range(len(self.child_layers))
+                            ),
+                            body_internal_role="pre_operator",
+                        )
+                        paired = self._capture_payload(recapture).paired
+                        rows = [
+                            self._moving_gauge_paired_rows(
+                                paired,
+                                manifest,
+                                layer_index=(0 if index == 0 else index - 1),
+                                suffix=(
+                                    "residual.pre"
+                                    if index == 0
+                                    else "residual.post_layer"
+                                ),
+                            )[1]
+                            for index in range(len(self.child_layers) + 1)
+                        ]
+                        writer_rows = {
+                            site.parameter_id: (
+                                self._moving_gauge_child_writer_rows(
+                                    paired,
+                                    manifest,
+                                    layer_index=site.transition_index,
+                                    suffix=_MOVING_GAUGE_WRITER_SUFFIX[
+                                        site.parameter_id
+                                    ],
+                                    input_width=writer_input_widths[
+                                        site.parameter_id
+                                    ],
+                                )
+                            )
+                            for site in writer_evidence
+                        }
+                        return Lfm2MovingGaugeRecapturedChildState(
+                            request_sha256=request["request_sha256"],
+                            boundary_child_rows=tuple(rows),
+                            writer_input_rows=writer_rows,
+                            row_alignment_sha256=row_alignment_sha256,
+                            recapture_provenance=canonical_sha256(
+                                {
+                                    "split_sha256": split_sha256,
+                                    "request_sha256": request["request_sha256"],
+                                    "role": "mid_sweep_reclosure_recapture",
+                                }
+                            ),
+                            child_state_sha256=(
+                                self.child_state_fingerprint()
+                            ),
+                            capture_sha256=recapture.capture_sha256,
+                            child_replay_capture_calls=1,
+                            child_model_forward_calls=len(fit_records),
+                        )
+                    finally:
+                        self._restore_physical_parameters(snapshots)
+
+                replacement_state = compile_lfm2_moving_gauge_residual_reclosure(
+                    tuple(boundaries),
+                    tuple(writer_evidence),
+                    policy=Lfm2MovingGaugeReclosurePolicy(
+                        reverse_sweep_mode="recaptured_terminal_anchored",
+                    ),
+                    mid_sweep_recapture_provider=_mid_sweep_recapture,
+                )
+            else:
+                replacement_state = compile_lfm2_moving_gauge_residual_reclosure(
+                    tuple(boundaries),
+                    tuple(writer_evidence),
+                    policy=Lfm2MovingGaugeReclosurePolicy(),
+                )
```

Prerequisites the diff assumes, all of which the adapter already has or needs
trivially:

- `self.body_reverse_sweep_mode` — new constructor argument, default
  `"forward_only_identity_transport"`, plumbed from
  `--dense-body-reverse-sweep-mode`. Add it to `_ADAPTER_IDENTITY_FIELDS`
  (same compatibility note as §2.3).
- `_MOVING_GAUGE_WRITER_SUFFIX` — the `parameter_id -> capture suffix` map that
  is currently built inline at `:11096-11113`
  (`attention.context` / `short_conv.context` / `mlp.post_swiglu`); hoist it into
  a dict in the same loop instead of duplicating the branch.
- `_restore_physical_parameters(snapshots)` — the rollback already exists in
  spirit in `_compile_moving_gauge_causal_completion` (`:5326-5336` snapshot,
  and its `finally` restore). Reuse that helper rather than writing a second one.
  **The rollback is mandatory**: the compile stage must leave the child exactly
  as it found it, because `install_body_reclosure` is the only stage allowed to
  mutate it. The reclosure module verifies nothing about the live child, so this
  is the adapter's invariant to hold.
- Cost: exactly one extra full-stack replay per body reclosure (~2.2 h at r12's
  rate), which is the number quoted in the plan.

If the parallel agent prefers not to inline a closure, the equivalent shape is a
small private method `_moving_gauge_mid_sweep_recapture(request, *, writer_sites,
writer_evidence, manifest, fit_records, audit, row_alignment_sha256,
split_sha256)`; the module only requires a callable taking the request mapping.

### 3.1 Contract the provider must satisfy (enforced, fail-closed)

`_authenticate_mid_sweep_recapture` rejects the reclosure unless:

- the returned object is an `Lfm2MovingGaugeRecapturedChildState` whose own
  report seal verifies against its tensors;
- `request_sha256` equals the request the compiler just sealed (the request
  digest covers the policy digest, the row alignment, the boundary ids and
  widths, the writer ids and input widths, and the sha256 of *every forward
  replacement tensor*) — a recapture measured on a stale publication cannot be
  substituted;
- boundary count, boundary widths, row count, row alignment, writer id set, and
  every writer input width match this reclosure;
- `child_replay_capture_calls >= 1` — a "recapture" that never replayed the
  child is refused.

---

## 4. Terminal restart capsule — landed half and the remaining half

**Landed, opt-in.** `Lfm2ClosedLoopRestartCapsuleWriter.record_terminal_phase(...)`
publishes a sealed capsule with `capsule_phase="terminal_post_output_interface"`
carrying a `terminal_phase` seal
(`lfm2_restart_capsule_terminal_phase`) that pins the body-reclosure,
final-normalization, and output-interface report digests. The orchestrator calls
it from `ChildClosedLoopCompiler._record_terminal_restart_capsule`
(`dynamic_closed_loop_fold.py`), after the output interface and before the
transaction commit and any evaluation.

The gate is `Lfm2ClosedLoopRestartCapsuleWriter(..., publish_terminal_capsule=False)`,
fed by `Lfm2DenseBudgetFoldConfig.restart_capsule_publish_terminal`
(`hf_lfm2_dense_budget_model_fold.py:1812-1816`, passed through at `:3726`), and
surfaced in the transaction receipt as
`restart_capsule.terminal_capsule_enabled`. Three ways the old behavior is kept
exactly:

- a writer that does not implement `record_terminal_phase` at all,
- a writer that implements it and returns `None` (declined — what the LFM2
  writer does when the flag is off),
- accepted-layer capsule manifests, which gain no `capsule_phase` key and keep
  their content key, chain digest, and manifest digest.

`--dense-restart-capsule-terminal` still has to be added to
`experiments/run_lfm2_dynamic_catalog.py` (one `add_argument` plus one field in
the `Lfm2DenseBudgetFoldConfig` construction) — that file is owned by Codex's
uncommitted diff, so the CLI flag is the one piece of this item left for whoever
lands it next. The r13 flag set in the plan calls for it explicitly.

**Not landed: resuming *from* a terminal capsule.** Loading one already restores
the exact post-output-interface child (`load_lfm2_closed_loop_restart_capsule`
returns `capsule_phase == "terminal_post_output_interface"` and
`resume_layer_index is None`), and today that resume fails closed in
`ChildClosedLoopCompiler.run` with "A restart-capsule resume left no layer to
execute." To convert that into a real skip-to-evaluation resume:

1. `hf_lfm2_dense_budget_model_fold.py:_build_restart_capsule_writer` /
   `_load_restart_capsule_resume` (`:3660`): when the discovered capsule's
   `capsule_phase` is terminal, do not construct a body-fold resume at all —
   return a `TerminalPhaseResume` carrying the restored child plus the sealed
   `terminal_phase`.
2. `dynamic_closed_loop_fold.py`: accept `terminal_phase_resume` on
   `ChildClosedLoopCompiler`; when present, skip `_run_layer` for every selected
   layer, skip `_run_body_reclosure`, `_run_final_normalization` and
   `_run_output_interface`, and synthesize their reports from the sealed digests
   in the capsule (a `mode: "terminal_restart_capsule_resume"` report whose
   `report_sha256` re-derives the pinned stage digests, exactly like
   `_resumed_layer_report` does per layer). Fail closed if any pinned digest is
   absent.
3. The adapter must certify the skip the way it certifies mid-body resume:
   an `assert_terminal_phase_resume_supported(...)` analogous to
   `assert_mid_body_resume_supported`, proving the live child equals the capsule
   child and that no stage is being silently skipped. **This is the piece that
   requires the adapter file**, which is why it is specced rather than landed.
4. `Lfm2ClosedLoopRestartCapsuleWriter.assert_resume_supported` must keep
   refusing a terminal capsule for *body* resume (it already does: `completed ==
   selected` leaves `remaining` empty and the orchestrator raises).

Value: a crash in the 11–14 h evaluator window currently costs the entire
transaction (body ~18 h + reclosure ~19 h + terminal solves ~12 h). With the
resume half landed it costs only the evaluators.

---

## 5. Fresh-reload dedup — what changed

`dynamic_fold_artifact.py`:

- `_portable_child_state_sha256` — publication-side twin of the capsule's
  `portable_child_state_sha256`: dense content plus storage-alias structure, no
  `id()` anywhere.
- `_fresh_reload_evaluation_fingerprint` — binds artifact tree, portable child
  state, process state fingerprint, serialized-state inventory, structural
  topology fingerprint, loader id + implementation + resource fingerprints,
  evaluator implementation + resource fingerprints, behavior-gate manifest
  fingerprint, and the inference execution contract.
- `publish_dynamic_fold_artifact(..., reuse_proven_identical_fresh_reload_evaluation=True)`:
  the staged decision (and its observed execution counts and fold-only guard) are
  reused for the final destination only when both fingerprints are equal.
  Anything else — including bytes that changed under the rename — re-evaluates in
  full and then still hits the existing tree/state/topology equality checks.
- Every structural check on both paths is unchanged: disjointness, dynamic-target
  revalidation, frozen parameters, inspector topology match, exact budget, state
  fingerprint, inventory, gradient absence, tree stability, provenance bundle.
- The certificate is unchanged in content and in schema; `--dedup` and
  `--no-dedup` publications certify identically (test
  `test_deduplicated_and_full_evaluation_certify_identically` compares the full
  certificate body minus `path`).

Saving: one of the two 8.5 h evaluator passes per publication.

---

## 6. Verification actually run

Runtime: `.venv/Scripts/python.exe` (torch 2.13.0+cpu, py3.11.9) — the repo's
own environment; `C:/Python314/python.exe` has no torch and was used for `ruff`.

- `pytest tests/test_hf_lfm2_moving_gauge_residual_reclosure.py tests/test_dynamic_fold_artifact.py tests/test_hf_lfm2_closed_loop_restart_capsule.py -q`
  → 131 passed (the three suites named in the task).
- 21 impacted suites (artifact, compiler, system, closed-loop, capsule, moving
  gauge, catalog, dense-budget stack, heterogeneous) in one `pytest -q` run →
  542 passed.
- `pytest tests/test_hf_lfm2_dynamic_closed_loop_adapter.py tests/test_dynamic_lfm2_dense_budget_fold_stack.py -q`
  → 71 passed, with **no edit to any adapter-owned test file**: the terminal
  capsule is opt-in precisely so
  `test_restart_capsule_resume_reproduces_the_uninterrupted_fold`'s
  `capsules_written == 2` stays true.
- `ruff check` clean on every file touched. (The 50 pre-existing ruff findings in
  the repo are all in files this wave did not touch, including the untracked
  `tests/test_hf_lfm2_r13_retention_lane.py`.)
- Default-path byte-identity vs `HEAD`: the working-tree module and the module
  as committed at `01bea537` were both imported into one process and compiled on
  the same fixed evidence, for `transition_mode` in
  {`moving_gauge`, `forced_identity_diagnostic`}. Equal in: policy
  `authentication_sha256`, policy body, state `report_sha256`,
  `diagnostics_sha256`, the whole diagnostics body, the whole report body, and
  the raw bytes of every published replacement tensor. Forward-only policy digest
  is pinned in the test suite as
  `391318199830753ee60399a0725b7d774be5027760d62741b0ed82347f14eb55`; the
  recaptured policy seals to a different digest.

### 6.1 Toy proof numbers (item 6 success criterion)

Toy: 3-layer residual child, width 3, per-layer write `tanh(state @ reader) @ W.T`
(so a writer change moves later writer inputs *and* reaches the terminal through
a nonlinearity), fixed nonlinear parent, 48 FIT rows, 32 held-out rows through
the same parent. Metric: terminal held-out rRMSE against the chart-encoded parent
terminal.

Seed 2026 (the seed pinned in the test):

| | held-out terminal rRMSE |
|---|---|
| unfolded child | 0.7296 |
| forward-only reverse sweep | 0.5132 |
| recaptured reverse sweep | 0.2852 (−44.4%) |

30-seed sweep (2020–2049), recaptured vs forward-only:
**28/30 better, median −25.4%, best −59.3%, worst +11.5%.**
The pinned test asserts only `recaptured < 0.8 * forward_only` on seed 2026, so
it fails loudly on a regression without being a numeric tripwire.

### 6.2 Two things the sweep caught that the item did not mention

1. **Do not invert the measured response.** The first implementation requested
   `residual @ pinv(J)` at each boundary. On seed 2034 the measured downstream
   gain was ~0.02, so the solver requested a local step ~40x the residual to buy
   a 1% terminal improvement, left the linearization entirely, and made held-out
   rRMSE 6x *worse* (0.458 → 2.85). The landed solver instead solves
   `min ||X·delta·J - R||_w^2 + ridge||delta||^2` through one paired symmetric
   eigendecomposition (`_terminal_anchored_composed_ridge`), so a direction the
   child barely propagates is simply not requested. At `J = I` it reduces exactly
   to the anchored ridge the forward-only sweep already uses (asserted in
   `test_identity_downstream_response_reproduces_the_anchored_writer_ridge`).
2. **The measured transport still needs a trust region.** Even without an
   inverse, the unregularized composed solve blew up on the same seed
   (+515% worst case). `Lfm2MovingGaugeReclosurePolicy.recaptured_step_ridge_ratio`
   (default `1e-2`, sealed in the recaptured policy body only) damps only the
   steps whose transport is a *measured linearization*; the terminal boundary,
   whose transport is the exact identity, keeps `writer_ridge_ratio`. Sweep over
   the ratio (30 seeds, worst-case delta): `1e-8` +515%, `1e-4` +9.5%,
   `1e-3` +9.9%, `1e-2` +11.5%, `1e-1` +15.0%; median gain is flat at −24…−25.6%
   across `1e-4…1e-1`. `1e-2` was chosen for the best median with no blow-up.

Read that as: the recaptured sweep is a damped Gauss-Newton step on measured
state, not an exact solve. The receipt records the per-boundary amplification
ratio (`terminal_amplification_ratio`), which is the quantity that predicts when
the step is worthless — a boundary with a tiny ratio is one where the writer
cannot reach the terminal at all.

---

## 7. What is deliberately not claimed

- The toy is a nonlinear residual stack, not LFM2. It reproduces the two
  mechanisms the item targets (moving writer inputs, nonlinear downstream
  response); it does not reproduce RMSNorm gain, GQA, short-conv, or 14-deep
  compounding. The r13 carrier run is what tests those.
- The reclosure module measures nothing about the live child. The recapture's
  provenance, the rollback of the preview publication, and the honesty of the
  declared replay/forward counts are the caller's obligations, sealed into the
  receipt (`mid_sweep_recapture`) but not independently verifiable from inside
  the module.
- `reconciled_terminal_relative_rmse` in the recaptured mode is still an analytic
  projection of the *measured* residual, not a second measurement. Only another
  recapture (or item 2's terminal floor) can confirm the realized terminal error.
