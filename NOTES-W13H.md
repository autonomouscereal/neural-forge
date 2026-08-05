# W13H — the `assert 3 == 4` behavior-gate count failures

Branch `codex/lrc-blank-compression`, HEAD `84acded2`. Local CPU only, run with
`.venv/Scripts/python.exe` (Python 3.11.9, torch 2.13.0+cpu).

## 1. Symptom

Four suites, one test each, failing on a hard-coded behavior-gate evaluation
count:

```
FAILED tests/test_dynamic_neo_native_vision.py::test_neo_dynamic_fold_cache_stack_behavior_artifact_and_receipt
FAILED tests/test_native_audio_executable_lane.py::test_native_audio_fold_cache_stack_behavior_artifact_and_receipt
FAILED tests/test_native_video_executable_lane.py::test_native_video_fold_cache_stack_behavior_artifact_and_receipt
FAILED tests/test_native_points3d_executable_lane.py::test_native_points3d_fold_cache_stack_behavior_artifact_and_receipt

>       assert behavior_gate.evaluation_count == 4
E       assert 3 == 4
```

A **fifth** test in the same family, not named in the brief, was found by
running the shared machinery and has the same root cause and the same fix:

```
FAILED tests/test_dynamic_qwen_compensated_adapter.py::test_tiny_real_qwen_replay_runs_through_automatic_system

>       assert generator.calls == 3
E       assert 2 == 3
```

It is pre-existing: it fails identically in a clean worktree checked out at
`84acded2` with no local edits.

## 2. Root cause

`src/neural_forge/dynamic_fold_artifact.py:3106` — the fresh-reload evaluation
dedup inside `publish_dynamic_fold_artifact`.

Publication evaluates the child's behavior gate twice on disk: once on the
staged tree (`dynamic_fold_artifact.py:2921`) and once on the final
destination after the atomic rename (`dynamic_fold_artifact.py:3136`). Since
commit **`3f71c6ac` "Reclose writers against recaptured child reality"**
(Tue Aug 4 2026 20:43:14 -0600), the second evaluation is skipped when the
digest built by `_fresh_reload_evaluation_fingerprint`
(`dynamic_fold_artifact.py:762`) matches across the rename
(`dynamic_fold_artifact.py:2977` vs `:3082`), gated by the new
`reuse_proven_identical_fresh_reload_evaluation` keyword, default `True`
(`dynamic_fold_artifact.py:2565`).

That digest binds the artifact tree (relative paths, so it survives the
rename), the portable child state (`_portable_child_state_sha256`,
`dynamic_fold_artifact.py:736` — dense content plus storage-alias structure,
no `id()`), the child state fingerprint, the serialized inference-state
inventory, the structural topology fingerprint, the loader id and its
implementation/resource fingerprints, the evaluator's implementation/resource
fingerprints, the behavior-gate manifest fingerprint, and the inference
execution contract.

So publication went from four to three gate evaluations per lane run, and the
five tests still asserted the pre-cache count.

**It was not `7f5187d4`.** That commit adds only
`src/neural_forge/hf_lfm2_sealed_projection_cache.py` and its test — two new
files, nothing shared with this path. It cannot have changed any count.

### Bisect

`git bisect` over 73 commits, `18ccb1b4` (good) → `84acded2` (bad), running
`tests/test_dynamic_neo_native_vision.py::test_neo_dynamic_fold_cache_stack_behavior_artifact_and_receipt`:

```
# good: 8dff2ea0    # bad: 73d40512    # good: 0bd1dfce
# good: c546030f    # good: b9ae384b   # bad: 3f71c6ac    # good: 1a441515
3f71c6ac660033340d529b30b1652aa55688dfb2 is the first bad commit
```

**Trap worth recording.** The venv carries an editable install
(`.venv/Lib/site-packages/__editable__.neural_forge-0.1.0.pth`) whose single
line points at the *main* checkout's `src`. Running pytest inside a git
worktree therefore imports HEAD's `neural_forge` against that worktree's
historical tests, which silently produces garbage bisect verdicts (my first
pass "proved" the failure predated `3f71c6ac`, and even predated the test
files themselves). Every historical run must set
`PYTHONPATH=<worktree>/src`, which takes precedence over the `.pth` entry.

## 3. Which side was wrong: **the tests**

The skipped evaluation is a true duplicate.

**Direct measurement.** Instrumenting `NeoNativeVisionBehaviorGate.evaluate_model`
and running the vision lane with the dedup disabled records four evaluations:

| # | call site | portable child state | verdict | gate report digest |
|---|---|---|---|---|
| 1 | test, unmarked control | `658928b6…` | reject | `9633bbaf…` |
| 2 | `dynamic_fold_executor` in-fold admission | `32b327e9…` | accept | `e456304b…` |
| 3 | `dynamic_fold_artifact.py:2921` stage reload | `32b327e9…` | accept | `e456304b…` |
| 4 | `dynamic_fold_artifact.py:3136` final reload | `32b327e9…` | accept | `e456304b…` |

Evaluations 3 and 4 are byte-identical in subject *and* in the evidence the
verdict was computed from — same portable state, same decision, same full gate
report digest. Only the directory the identical bytes live in differs.

**The dedup does not skip any structural proof.** With the verdict reused, the
final destination is still independently loaded and then checked for storage
disjointness from the in-memory child, from the stage reload, and from every
live parent; dynamic-target runtime revalidation; frozen inference parameters;
full re-inspection; structural topology fingerprint match against the compiled
child *and* cross-check against the stage fingerprint; exact unique-parameter
budget; state fingerprint against the accepted in-memory child; state
inventory; absence of gradients; artifact tree equality; and provenance bundle
authentication.

**The repo already proved certificate equivalence.**
`tests/test_dynamic_fold_artifact.py::test_deduplicated_and_full_evaluation_certify_identically`
publishes the same child under both settings and asserts the certificate
bodies are equal apart from `path`.

**Toggle proof (failure appearing and disappearing on one factor).** With
`reuse_proven_identical_fresh_reload_evaluation=False` injected into
`dynamic_fold_compiler.publish_dynamic_fold_artifact`, all five tests pass at
their original counts; with the default `True` each is exactly one lower.

`3f71c6ac` also added a fresh-reload dedup path — this *is* the same commit and
the same change; there is no second competing cause in `dynamic_fold_artifact.py`.

## 4. What the strengthened assertions now prove

A bare integer cannot distinguish "the gate decided on the right distinct
children" from "the gate decided on one object N times", and it can be
repaired by editing a digit. Each of the five tests now records the *subjects*
the gate actually ran, via a small test-local recording subclass
(`_Recording*BehaviorGate`, and `_ForwardGenerator.gated_children` for the qwen
replay) that keeps the model, its `fingerprint_model_state`, the verdict, and
the set of parameter `data_ptr()` values. Holding the models keeps their
storages alive, so `data_ptr` values cannot be recycled into a false reading.

The four lane tests now assert:

- `behavior_gate.evaluation_count == 3` **and** `len(gated_children) == 3`
- the verdict sequence is exactly `[False, True, True]`
- the rejected subject *is* the `unmarked` control object, and its state
  fingerprint differs from the folded child's
- the two accepted subjects carry the **same** folded state fingerprint
- the two accepted subjects are **different objects** with **disjoint**
  parameter storages — i.e. one was gated in memory and one was gated as a
  genuinely reloaded child, not the same object handed back twice
- the gated reload's state fingerprint equals `fingerprint_model_state` of the
  child re-loaded from `artifact_path / "child"` — the gated bytes are this
  artifact's bytes

The qwen replay test asserts the same shape at `calls == 2` (it has no
hand-rolled negative control, so its subject list is the accepted pair only).

**Verified that this is strictly stronger than the count.** A simulated future
evaluation-skip regression — a "widened cache" that reuses the in-fold
admission verdict for the fresh reload while still running the reloaded child
forward so the execution-contract observer stays satisfied — publishes
successfully (production does *not* fail closed on it) and drops the count to
2. Editing the assertion to `== 2`, the way a stale-count fix would, still
fails:

```
        assert behavior_gate.evaluation_count == 2
        assert len(behavior_gate.gated_children) == 2
>       control, in_memory_child, reloaded_child = behavior_gate.gated_children
E       ValueError: not enough values to unpack (expected 3, got 2)
```

The count is also still exact in the other direction: with the dedup disabled
the suites fail `assert 4 == 3` / `assert 3 == 2`.

## 5. Open seam (observed, measured, **not** changed)

`fingerprint_model_runtime_state` (`dynamic_fold_executor.py:627`) hashes
`id(parameter)`, `id(buffer)` and `id(module)`, so it is meaningless across two
distinct model objects and is correctly absent from the dedup digest. The
consequence is that module `.training` flags and hook registries are the one
decision-relevant dimension the dedup does not bind before skipping an
evaluation. Frozen-ness *is* separately proven on both paths by
`_assert_frozen_inference_parameters`.

Measured on the points3d lane, a portable mode/hook signature (per-module
qualified type, `.training`, and hook-registry sizes) is identical for the
stage and final reloads — `4b550832…` both times, both eval-mode, zero hooks —
so the evidence set the reduced count guards is intact today.

Proposed, **not applied** (out of the brief's scope for a correct cache, and
`dynamic_fold_artifact.py` sits next to another agent's live work): add that
portable mode/hook signature as a field of `_fresh_reload_evaluation_fingerprint`.
It is internal-only (the digest is never persisted; the certificate body is
identical either way), it would not change any count, and it would close the
last dimension under which an evaluation may be skipped.

## 6. Ruled out

- **`7f5187d4` "Add sealed LFM2 projection cache"** — adds two brand-new files
  (`hf_lfm2_sealed_projection_cache.py`, its test), touches nothing shared.
- **A second competing dedup.** `dynamic_fold_artifact.py` has exactly one
  commit in its history at or after the bisect window (`3f71c6ac`); the
  fresh-reload dedup the brief attributed to a separate change is that same
  commit.
- **The gates themselves.** The `evaluation_count` increment sites in
  `dynamic_neo_fold.py`, `dynamic_audio_fold.py`, `dynamic_video_fold.py` and
  `dynamic_points3d_fold.py` are untouched since the lanes were introduced.
- **Environment / torch version / test ordering.** All runs used
  `-p no:randomly`; the same failures reproduce in a clean worktree at
  `84acded2` and the tests pass at `18ccb1b4` under the same interpreter.
- **The other agent's uncommitted work.** The only files this task modified are
  the five test files listed below; `dynamic_closed_loop_fold.py`,
  `hf_lfm2_dense_budget_model_fold.py`, `hf_lfm2_balanced_circuit_basis.py`,
  `hf_lfm2_sealed_evidence_set.py`, `hf_lfm2_circuit_trace_cache.py`,
  `hf_lfm2_balanced_basis_phase_cache.py`,
  `hf_lfm2_streaming_vocabulary_margin_adapter.py`,
  `hf_lfm2_on_policy_evidence_round.py` and the
  `run_lfm2_exact4x_moving_gauge_v5*` scripts were not touched.

## 7. Files changed

Tests only — no production change.

- `tests/test_dynamic_neo_native_vision.py`
- `tests/test_native_audio_executable_lane.py`
- `tests/test_native_video_executable_lane.py`
- `tests/test_native_points3d_executable_lane.py`
- `tests/test_dynamic_qwen_compensated_adapter.py`

## 8. Suites run

Green:

| Suite | Result |
|---|---|
| the four native lanes + `test_dynamic_qwen_compensated_adapter.py` (final run) | **52 passed** |
| `test_dynamic_fold_artifact.py`, `test_dynamic_fold_compiler.py`, `test_dynamic_system.py`, `test_dynamic_behavior_gate.py`, `test_dynamic_fold_executor.py`, `test_native_points3d_output_lane.py` | 146 passed |
| `test_dynamic_lfm2_budgeted_materialization.py`, `test_dynamic_lfm2_dense_budget_fold_stack.py`, `test_dynamic_lfm2_system.py`, `test_dynamic_parameter_frontier.py`, `test_dynamic_production_catalog.py`, `test_dynamic_qwen_global_codebook.py`, `test_dynamic_search.py`, `test_dynamic_semantic_fold.py` | 119 passed (the 1 failure in this batch was the qwen count above, now fixed) |
| `test_run_lfm2_dynamic_catalog.py`, `test_dynamic_lfm2_cpcf_production_cli.py`, `test_dynamic_lfm2_exact4x_production.py`, `test_hf_lfm2_exact4x_composite_interface.py` | 157 passed, 1 skipped, 1 failed — the failure is only `test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`, the known pre-existing one from Codex's uncommitted `hf_lfm2_dense_budget_model_fold.py` diff |

Grand total across the shared machinery: **474 passed, 1 skipped, 1 known
pre-existing failure that is not this task's.**

`ruff check` clean on all five touched files. (`ruff format --check` reports
those files as unformatted, but it reports the *committed* versions the same
way, so `ruff format` is not this repo's gate.)

Not run / not mine:
`test_mid_body_capsule_resume_is_refused_under_moving_gauge_reclosure`, a known
pre-existing failure from Codex's uncommitted diff.

## 9. Reproduction commands

```bash
cd "C:/Users/cereal/Documents/ai training/neural-forge"

# the five tests
.venv/Scripts/python.exe -m pytest \
  tests/test_dynamic_neo_native_vision.py \
  tests/test_native_audio_executable_lane.py \
  tests/test_native_video_executable_lane.py \
  tests/test_native_points3d_executable_lane.py \
  tests/test_dynamic_qwen_compensated_adapter.py \
  -q --no-header -p no:randomly

# the toggle proof: a pytest plugin doing
#   dynamic_fold_compiler.publish_dynamic_fold_artifact = functools.partial(
#       original, reuse_proven_identical_fresh_reload_evaluation=False)
# restores the old counts and fails the new assertions.
```
