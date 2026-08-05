# NOTES-W13C — the moving-gauge body reclosure must prove it improved the boundary it overwrites

Wave 13C implements NOTES-W13B §8's named fix, recommendation 3:

> **Make the body reclosure prove it improved the boundary it rewrote.** Any
> writer replacement that lowers the write cosine or raises the composed
> post-operator relL2 for its own layer should be rejected and the pre-reclosure
> writer kept.

r12's reclosure left layer 0's `conv.in_proj` byte-identical, multiplied
`conv.out_proj`'s Frobenius norm by **4.73x** (26.88 → 127.23), took the write's
cosine from 0.911 to 0.171, and pushed the composed post-operator residual from
**0.494 to 0.836** — through the 1.0 emit-zero floor. Layer 1 repeats it worse
(writer norm ratio 0.48 → 6.37). Nothing in the reclosure measured that. It
optimizes a terminal objective and every writer it touches is published
unconditionally.

This wave adds the missing per-boundary guard. **Zero training, zero gradients,
zero new evidence paths, zero new solves.** The gate is a pure filter on
publication: it reuses the module's own residual-write target machinery and its
own weighted relative error, and it can only ever keep a writer the reclosure
already computed or the incumbent it was going to overwrite.

---

## 1. What landed

| file | change |
|---|---|
| `src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py` | the gate: policy fields + additive authentication body, the acceptance decision, the recaptured target frame, the sealed ledger |
| `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` | config plumbing only (2 fields, validation, mirror, policy construction) |
| `experiments/run_lfm2_dynamic_catalog.py` | two additive flags, validation, conditional `identity()` entry, adapter threading |
| `tests/test_hf_lfm2_moving_gauge_residual_reclosure.py` | +9 appended tests (68 → 77); **no existing test modified** |
| `tests/test_hf_lfm2_dynamic_closed_loop_adapter.py` | +1 appended test (56 → 57) |
| `tests/test_run_lfm2_dynamic_catalog.py` | +1 appended test |

Named anchors:

- Policy fields and validation —
  `hf_lfm2_moving_gauge_residual_reclosure.py:1125-1126`, `:1165-1181`.
- Additive authentication body (the `reverse_sweep_mode` pattern, extended) —
  `:1206-1245`.
- `_boundary_acceptance_decision` (the whole gate) — `:4235-4334`.
- `_recaptured_boundary_write_target` (composition with the recaptured sweep) —
  `:4336-4369`.
- Forward-sweep gate — `:4963-4990`.
- Forward-only reverse-sweep gate — `:5104-5129`.
- Recaptured reverse-sweep gate — `:4673-4711`.
- Sealed per-boundary ledger — `:5154-5272`; publication-body summary —
  `:5433-5462`.
- Adapter config — `hf_lfm2_dynamic_closed_loop_adapter.py:1113-1124`,
  `:1489-1519`, `:3108-3115`; policy construction at the single reclosure call
  site `:11329-11342`.
- Catalog flags — `run_lfm2_dynamic_catalog.py:13607-13646`; validation
  `:1776-1822`; adapter threading `:9842-9849`; `config_from_args` `:14491-14496`.

---

## 2. Gate semantics

### The mode

`Lfm2MovingGaugeReclosurePolicy.writer_reclosure_acceptance`:

| value | behavior |
|---|---|
| `unguarded` | **default.** Byte-identical to every reclosure sealed before this wave: no measurement, no ledger, no policy-digest change. |
| `require_boundary_improvement` | measure every proposed writer overwrite; on regression **keep the incumbent for that boundary**, record the refusal, continue. **r13 recommendation.** |
| `require_boundary_improvement_fail_closed` | same measurement; on regression **abort** the reclosure naming the boundary and both metrics. |

`boundary_regression_tolerance` (default `0.0`) is the relative slack:
a proposal is accepted iff
`proposed <= incumbent * (1 + tolerance)`. Supplying a nonzero tolerance with
`unguarded` fails closed — a tolerance without a gate would claim a guard that is
not running.

### What is measured

For **every** writer group the reclosure proposes to overwrite — both the
forward sweep and the reverse/terminal-anchored sweep — the incumbent and the
proposal are scored against the **same branch-write target frame the boundary's
own solve used**:

```
target = desired_total_write - fixed_other_write
```

exactly as the forward sweep builds it (`chart.residual_write_target` for the
moving gauge, `chart.encode(parent_delta)` for the forced-identity diagnostic),
with `fixed_other_write` the part of the realized branch write not attributable
to these writer sites. The metric is the module's existing
`_relative_rmse(prediction, target, row_weights)` under the boundary's own
`fit_weights` — a weighted relative L2 on the branch **write**, which is the
quantity NOTES-W13B measured at 0.6425 (capsule 13) versus 1.0986 (published).

**Rows.** The measurement is taken on the same rows the sweep it guards solves
on: the sealed aligned FIT writer-input rows for the forward sweep and the
forward-only reverse sweep, and the authenticated mid-sweep recapture for the
recaptured reverse sweep. Every decision names its frame
(`aligned_fit_writer_input_rows` / `recaptured_writer_input_rows`).

**Honest limitation, stated once.** `Lfm2MovingGaugeWriterEvidence` carries
exactly one design-row set per site and no held-out counterpart;
`Lfm2MovingGaugeBoundaryEvidence.heldout_*` rows are boundary states, not writer
inputs, so they cannot address a writer. Adding a held-out writer-row field would
change every existing writer-evidence digest, which the byte-identity
requirement forbids. The gate therefore measures on the writer's own aligned
rows and says so in the receipt rather than calling a fit-row number "held out".
The r12 failure is a factor of **1.7x on the write and 1.69x on the composed
residual at fit scale** — nowhere near a generalization-gap question.

### Monotonicity

Each stage's decision is non-regressing in its own frame, so the composition of
the two stages is non-regressing against the pre-reclosure seed to within
`(1 + tolerance)^2`. The ledger records that end-to-end claim per boundary
(`seed_relative_write_error`, `published_relative_write_error`,
`published_no_worse_than_seed`) in the FIT frame, always, so a receipt reader
can check it without replaying either sweep.

Caveat: with the recaptured reverse sweep on, the reverse decisions are taken in
the recaptured frame while the summary is reported in the FIT frame. The
per-stage guarantee holds in the frame each stage solves in; the FIT-frame
summary line is a diagnostic, not a second guarantee. It is nonetheless True on
every arm measured below.

### Receipt

Sealed ledger at `report["diagnostics"]["boundary_acceptance"]`
(`verify_lfm2_moving_gauge_report`-checkable, own `report_sha256`), carrying per
decision: `sweep`, `transition_index`, `boundary_id`, `parameter_ids`,
`row_frame`, `rows`, `incumbent_stage`, `accepted`/`refused`,
`incumbent`/`proposed`/`accepted_relative_write_error`,
`relative_write_error_ratio`, `acceptance_threshold`, `regression_tolerance`,
`incumbent`/`proposed`/`accepted_write_l2_norm`, **`write_norm_ratio`**, and
per-parameter `incumbent`/`proposed`/`accepted_weight_frobenius_norm` with
**`weight_frobenius_norm_ratio_by_parameter`** — the literal 4.73x signature.
Plus `refused_by_sweep`, `refused_transition_indices`, the per-boundary
end-to-end table, and `every_published_boundary_no_worse_than_seed`.

The publication body carries `acceptance_schema`,
`writer_reclosure_acceptance`, `boundary_regression_tolerance`,
`boundary_acceptance_sha256` and a compact `boundary_acceptance_summary`. In
`unguarded` mode none of these keys exist and the diagnostics carry no
`boundary_acceptance` — the sealed digest of every prior run is unchanged.

### Digests

```
unguarded / forward-only      391318199830753ee60399a0725b7d774be5027760d62741b0ed82347f14eb55  (pinned pre-existing)
guarded   / forward-only      a433e5ba74f70cf1399075228ec4cbaf6ec20e36658f8d00d3fb95f17df671af
guarded   / recaptured        f773d128411dafa907982abf623ee877dbab311096aa5fa6444423b115ac653d
guarded   / recaptured 0.05   a3f5dc9f70cf7d61deb6483b3891b25cc28338560af50bd5dfc058592f481067
```

The unguarded/forward-only digest is the constant the existing suite already
pins (`FORWARD_ONLY_POLICY_SHA256`) and it did not move. All six combinations of
{unguarded, guarded, fail-closed, guarded+tolerance} x {forward-only,
recaptured} produce six distinct digests.

---

## 3. Toy proof

### 3.1 Accept-all reproduces the unguarded result exactly

Two independent demonstrations.

**A converging fold, at tolerance 0.0.** A single-transition fold (64 rows, 4
writer features, 3-wide residual, nonlinear parent, nonzero seed writer). With
one transition the terminal residual *is* that boundary's own remaining write
error, so the reverse sweep re-solves the forward sweep's problem anchored at the
forward answer and can only improve it:

| decision | incumbent | proposed | accepted |
|---|---|---|---|
| forward t=0 | 0.238811 | 0.232780 | yes |
| reverse t=0 | 0.232780 | 0.232780 | yes |

`replacement_weight_sha256` **identical** to the unguarded compile. Same at
tolerance `1e-6`, and the same fold compiles clean under
`require_boundary_improvement_fail_closed`.

**Unbounded tolerance on the regressing fold.** Setting
`boundary_regression_tolerance=1e9` on the 3-layer nonlinear depth toy (seed 7)
evaluates all 6 overwrites, accepts all 6, and reproduces the unguarded
`replacement_weight_sha256` **byte-for-byte in both reverse-sweep modes**. This
is the load-bearing property: the gate adds no solve, no reordering and no
numerical perturbation, so any difference between a guarded and an unguarded
publication is a refusal and nothing else.

### 3.2 Engineered regression: the gate refuses, and the boundary is better

The existing depth toy (`_toy_fold`, `_ToyChild`: every layer writes
`phi(state @ reader) @ writer.T`, so a writer moves every later layer's input and
reaches the terminal through a nonlinearity) already exhibits the r12 pathology:
its reverse sweep's terminal objective conflicts with the first two boundaries'
own writes. Seed 7, `reverse_sweep_mode="forward_only_identity_transport"`:

**Per-boundary published write error (FIT frame, weighted relative L2)**

| boundary | unguarded | guarded (tol 0.0) | improvement |
|---|---|---|---|
| t=0 | 0.751958 | **0.679516** | 1.11x |
| t=1 | 1.394758 | **0.499882** | **2.79x** |
| t=2 | 0.635841 | 0.635841 | untouched |

**The refusals**

| refusal | incumbent | proposed | error ratio | write-norm ratio | weight Frobenius ratio |
|---|---|---|---|---|---|
| reverse t=1 | 0.499882 | 1.394758 | **2.79x** | 2.154 | 1.965 |
| reverse t=0 | 0.679516 | 2.846621 | **4.19x** | 3.327 | 1.086 |

That is NOTES-W13B §6's 0.494 → 0.836 in miniature, with the same signature: the
proposal is directionally wrong *and* over-scaled, and it is the terminal
objective that asked for it. Refused, the incumbent is kept
(`realized_terminal_l2_norm == 0.0` on those reverse steps),
`every_published_boundary_no_worse_than_seed` is True, and every forward
decision is accepted (`refused_by_sweep = {forward: 0, reverse: 2}`) — the
forward sweep solves the boundary objective directly and does not regress it.
Only the terminal-absorbing reverse sweep does.

**The terminal, both ways, honestly.** Refusing the terminal absorption raises
the reclosure's own FIT-row terminal metric (`reconciled_terminal_relative_rmse`
0.264983 → 0.503874) — that is the objective being refused, by construction.
On this toy the *held-out* terminal nonetheless improves, 0.396784 → **0.324152**
(1.22x). Under the recaptured sweep the held-out terminal is 0.271755 → 0.269390
(essentially flat) while both regressed boundaries recover. The gate's contract
is per-boundary monotonicity, not the terminal; R13 item 2's terminal floor is
the object that speaks to the terminal, and the two compose.

### 3.3 fail_closed

`require_boundary_improvement_fail_closed` on the same fold aborts with:

```
Lfm2MovingGaugeResidualReclosureError: The moving-gauge reverse sweep regressed
boundary toy:boundary:2: relative write error 1.3947577573425352 exceeds the
incumbent 0.4998824926195698 at tolerance 0.0.
```

Both modes are selectable; skipping is the r13 recommendation because the r12
evidence says most boundaries are fine and the reclosure is worth keeping for
them.

### 3.4 Tolerance is a real dial

Seed 2031, transition 1 regresses by ratio 1.30x. At tolerance 0.0 it is refused;
at `ratio - 1 + 1e-9` it is admitted and the refusal count drops. Tested.

---

## 4. Composition with `reverse_sweep_mode="recaptured_terminal_anchored"`

Requirement: with both on, the acceptance measurement must use the same
recaptured states the sweep solves against.

`_recaptured_boundary_write_target` rebuilds the forward sweep's target with
every child-side quantity replaced by the authenticated mid-sweep recapture: the
pre-state `measured[index]`, the realized branch write
`measured[index+1] - measured[index]`, and the incumbent writer's own share of it
measured on the recaptured writer-input rows. Parent endpoints are cached
evidence and do not move, so the desired total write is the same chart target.

Proven by `test_boundary_gate_measures_the_recaptured_states_the_sweep_solves_on`,
which recomputes one decision independently from the `Lfm2MovingGaugeRecapturedChildState`
the provider returned and the `forward_replacement_weights` the sealed request
pinned, and matches the ledger to `rel=1e-12`. It also asserts the forward
decisions are identical between the two reverse modes (only the reverse frame
moved) and that the recaptured reverse incumbent metric genuinely **differs**
from the forward-only one (seed 7, transition 2: 0.508600 recaptured vs 0.635841
FIT).

Seed 7 with both on, tolerance 0.0:

| boundary | unguarded (recaptured) | guarded (recaptured) | improvement |
|---|---|---|---|
| t=0 | 0.739860 | **0.679516** | 1.09x |
| t=1 | 0.919246 | **0.499882** | **1.84x** |
| t=2 | 0.652632 | 0.652632 | untouched |

Refusals in the recaptured frame: reverse t=1 (0.489399 → 0.881225, write-norm
ratio 1.527), reverse t=0 (0.679516 → 1.631940, write-norm ratio 1.643). One
provider call, `parent_model_forward_calls: 0`, `backward_calls: 0`,
`optimizer_steps: 0`, publication authenticates.

Both extensions land in one policy body: `policy_schema` (recaptured) and
`acceptance_schema` (acceptance) coexist and the digest is distinct from either
alone.

---

## 5. Recommended r13 flag set for the reclosure

```
--dense-body-reclosure-mode moving_gauge_v3
--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement
--dense-moving-gauge-boundary-regression-tolerance 0.0
```

- **Mode `require_boundary_improvement`, not fail-closed.** r12's damage was
  concentrated: the forward sweep never regressed a boundary on any arm measured
  here, and only the terminal-absorbing reverse steps did. Aborting the whole
  reclosure on the first bad boundary throws away the boundaries it helps.
  Reserve `require_boundary_improvement_fail_closed` for a certification arm
  that wants the reclosure to be all-or-nothing.
- **Tolerance 0.0.** The study's rule is strict monotonicity, and every
  regression measured here is 1.1x–4.2x — orders of magnitude above any
  numerical ambiguity. One caveat worth knowing: at exactly 0.0 the FP32
  publication cast can refuse a change that is numerically a no-op (observed
  once at ratio 1 + 4e-9 on a converging fold). Refusing a no-op costs nothing,
  but it inflates the refusal count. Use `1e-6` if a clean refusal count matters
  more than literal strictness.
- **`reverse_sweep_mode`:** the gate composes with
  `recaptured_terminal_anchored` and the recaptured lane is the better one on
  every arm above (it refuses less and lands closer). **But the recaptured
  sweep is still module-level only** — the adapter builds its policy without
  `reverse_sweep_mode` and passes no `mid_sweep_recapture_provider`
  (`hf_lfm2_dynamic_closed_loop_adapter.py:11329`), so there is no catalog flag
  for it yet. Wiring it is R13 item 6's remaining half and was explicitly out of
  scope for this wave.
- **Read `boundary_acceptance_summary.refused_transition_indices` first** in the
  r13 receipt. If it is empty, the reclosure earned every writer it published.
  If layer 0 is in it, the r12 failure was reproduced and refused.

---

## 6. Verification

```
.venv/Scripts/python.exe -m pytest \
  tests/test_hf_lfm2_moving_gauge_residual_reclosure.py \
  tests/test_hf_lfm2_dynamic_closed_loop_adapter.py \
  tests/test_run_lfm2_dynamic_catalog.py -q
-> 248 passed, 1 failed, 1 skipped in 452.91s
```

- `tests/test_hf_lfm2_moving_gauge_residual_reclosure.py` — **77 passed**
  (68 pre-existing, unmodified; +9 appended). The unmodified pass includes the
  pinned `FORWARD_ONLY_POLICY_SHA256`.

**Direct byte-identity against HEAD.** Stronger than the pinned constant: HEAD's
module (`git show f9821339:src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py`)
was loaded side-by-side with the current one under `importlib`, each given
evidence built from its own classes, and both compiled the same depth toy.

| lane | quantity | HEAD | current |
|---|---|---|---|
| forward-only | `report_sha256` | `2913fac74507b993…` | `2913fac74507b993…` |
| forward-only | `diagnostics_sha256` | `5565e44f988e7a72…` | `5565e44f988e7a72…` |
| recaptured | `report_sha256` | `143403eaa63eb4e6…` | `143403eaa63eb4e6…` |
| recaptured | `diagnostics_sha256` | `80cf55cea3cc78ab…` | `80cf55cea3cc78ab…` |

`replacement_weight_sha256` identical in both lanes; policy digests identical
(`3913181998307…` forward-only, `d08f0791dbe77a2…` recaptured). The default
publication is unchanged to the byte, receipts included.
- `tests/test_hf_lfm2_dynamic_closed_loop_adapter.py` — **57 passed** (56 + 1).
- `tests/test_run_lfm2_dynamic_catalog.py` — 115 passed, 1 skipped, **1 failed**.

**The one failure is not mine.** `test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`
asserts `Lfm2DenseBudgetFoldConfig` raises "requires this process to fold every
layer". A parallel agent's *uncommitted* diff to
`src/neural_forge/hf_lfm2_dense_budget_model_fold.py` deletes exactly that
`raise` and replaces it with a terminal-only restart-resume path
(`_closed_loop_compiler_config_for_execution`). That file is untouched by this
wave and the failing path never reaches any code changed here.

Ruff:

```
.venv/Scripts/python.exe -m ruff check <6 touched files>
-> 3 errors, all F401 unused imports in experiments/run_lfm2_dynamic_catalog.py
```

All three are present at HEAD `f9821339` (verified against
`git show f9821339:experiments/run_lfm2_dynamic_catalog.py`), pre-date this
wave, and sit in the import block, not in anything added here. Left alone rather
than deleting imports out from under in-flight work in that file.

---

## 7. What I did not do

- **No held-out writer rows.** Explained in §2; would require a new field on
  `Lfm2MovingGaugeWriterEvidence`, which changes every existing writer-evidence
  digest.
- **No cosine gate.** NOTES-W13B recommends "lowers the write cosine or raises
  the composed post-operator relL2". Relative L2 on the write already fails both
  r12 failure signatures (direction collapse *and* over-scaling), it is the
  module's existing metric, and it needs no new evidence path. The direction
  half is instead made visible through `write_norm_ratio` and
  `weight_frobenius_norm_ratio_by_parameter`: a refusal with error ratio >> 1 at
  norm ratio ~1 is a rotation, one at norm ratio >> 1 is the r12 blowup.
- **No adapter flag for `reverse_sweep_mode`.** Out of scope (§5).
- **No composed post-operator metric.** The gate scores the branch **write**,
  which is what the writer controls. Composed post-layer residual is R13 item
  2's terminal floor, already landed separately.
- **The structural-effectfulness check is unchanged.** If a guarded reclosure
  refuses *every* writer at every transition, the existing
  "structurally ineffective in the target dtype" error still fires. That is the
  correct fail-closed outcome for a reclosure that earned nothing, but it means
  a fully-refused run aborts rather than publishing the seed.
- **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
- **Not committed.** No `git add`, no commit, no push.
