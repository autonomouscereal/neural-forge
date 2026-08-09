# NOTES-W13L — r13 refold, attempt #5: 6f276ade fails closed, b65c81f5 runs

Wave 13L, branch `codex/lrc-blank-compression`. Two launches happened this
wave, and the second is the one that is alive:

* **5a**, at the instructed `6f276ade` — launched clean, cleared every wall, and
  then **failed closed 37 seconds into the first layer capture** with a real
  receipt. Not a kill; a genuine fail-closed refusal (§7).
* **5b**, at `b65c81f5` — the deployment rolled back by exactly the two commits
  that carry the regression, relaunched, and **running with two layers accepted**
  at handoff (§8).

Predecessors: `NOTES-W13D` (#1), `NOTES-W13F` (#2), `NOTES-W13I` (#3),
`NOTES-W13K` (#4 — provably detached, SIGTERMed at 25 min by an unnamed sender).
W13K §10's primed recipe was followed as written, updated for the new HEAD.

**Two questions this wave answers that four previous waves could not:**

1. **The killer is named.** `killspy.log` recorded
   `sig=15 sender_pid=121286 sender_comm=bash -> target_pid=3551402`, and
   `3551402` was the **peer's own `a0026` run** (§6).
2. **`d296b602` breaks the fold.** A fail-closed integrity refusal in the dense
   closed-loop adapter reproduces in two independent runs, and blob archaeology
   attributes it to that commit alone (§7).

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.
Run directory
`/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold`.

---

## 1. Deployment bump: 6f178f14 → 6f276ade

```
bundle  6f178f14add54ec882c09ad183f69a626a0cfb48..codex/lrc-blank-compression
        (3 commits; b65c81f5, d296b602, 6f276ade)
size    83,728 bytes
sha256  9143bd4a2ea29683c621ecf86e9a975fba2052f52b0c67b42b1b67c9d7e7b591   (identical both sides)
server  /home/cereal/nf-w13l-r13/nf-w13l.bundle
```

`git bundle verify` clean on both sides, then
`git fetch <bundle> refs/heads/codex/lrc-blank-compression:refs/heads/w13l-import`
and `git reset --hard 6f276ade…`. Pre-reset state in
`/home/cereal/nf-w13l-r13/HEAD-before.txt` (`6f178f14…`) and `status-before.txt`
(**0 bytes** — clean tree, nothing overwritten). The deployment reflog's newest
entry before this wave was still W13K's 21:57Z reset, i.e. the peer never
touched this checkout.

| commit | what it touches |
|---|---|
| `b65c81f5` | `hf_lfm2_structural_residual_attention_fold.py` + its test — **only these two files** |
| `d296b602` | `hf_lfm2_child_closed_loop_refinement.py`, `hf_lfm2_dynamic_closed_loop_adapter.py` (**+1,102 lines**), `hf_lfm2_tokenizer_bridge_trace_adapter.py`, + 3 tests |
| `6f276ade` | `docs/CURRENT_FOLDING_GOAL.md` only |

`git diff --name-only 6f178f14 6f276ade` is **9 files** — 4 runtime source, 4
tests, 1 doc. A much smaller surface than W13K's 33.

Deployed working-tree digests, each compared against the **local committed blob**
(`git cat-file blob`, not inferred from commit titles):

| file | deployed sha256 | vs `6f178f14` |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **SAME — frozen collector holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | **SAME** |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77d041bb9cadac9b578dbc914ae59f37084846d1ea0e10397e37002a3f` | **SAME** |
| `hf_lfm2_streaming_vocabulary_margin_adapter.py` | `2a949a3076af23b829b31af2410d245f2d3f25355472a13740f3dd7879e051d6` | SAME |
| `hf_lfm2_balanced_circuit_basis.py` | `e1091d33fe03c95d338d4d967f1750e3d47f21a0223eaba367c317abb55c4108` | SAME |
| `hf_lfm2_balanced_basis_phase_cache.py` | `6bb971d558a573a5d1dba9dcaf11220b8cc8294d457258fc26cbcf08435b872f` | SAME |
| `hf_lfm2_sealed_evidence_set.py` | `8e6e608908e6eaee384cb54f52ab064bc0f3806621d89d8239b2d84683b2358c` | SAME |
| `dynamic_closed_loop_fold.py` | `2747ed84f5bff89b1adab1a8565e900a6d4a1f46a900dc43ced1eec4a845ecdc` | SAME |
| `hf_lfm2_closed_loop_restart_capsule.py` | `793ed8cc1876ac953520acdde3f8bcb95ad3ff4d7ea9b6b46e123bbc80b5e90d` | SAME |
| `hf_lfm2_child_replay_phase_cache.py` | `c326f5a84a18b63931fb4a0683a8c8aeebb1015c96d6affc7ecbaff592fccc0b` | SAME |
| `hf_lfm2_child_replay_batch_checkpoint.py` | `b5078a1b4910117c22bbd9cbcdf6e9ec4470b296a8a5d99de3871498dec94c67` | SAME |
| `experiments/run_lfm2_dynamic_catalog.py` | `2af6de96d8f147dbaf03db91a41545c3724179ba53859dfbb312a391aea04ee7` | SAME |
| `experiments/run_r12_on_policy_round.py` | `ad5e72a54880360af3887a4d4641968d6dc9fe7a5299d69000343ca7dcc000ec` | SAME |
| `hf_lfm2_child_closed_loop_refinement.py` | `3ce4e4c789ce1d99ac2a670c2e9f5201806073f0819f902dd9c895a1c8830f20` | DIFF |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `e0e63c451e308c45e7e9934b10220cf52d29b9c8b3ae72225bf7f862309e9ee6` | DIFF |
| `hf_lfm2_structural_residual_attention_fold.py` | `1173468417bae8c3ad9db96468a648a618df0fda0fdf21556d78baf1f04d597f` | DIFF |
| `hf_lfm2_tokenizer_bridge_trace_adapter.py` | `7cfd256688ff16c9879a8ba64f48fb25a3f6362b6970165724dd958c9b46a6df` | DIFF |

Every deployed digest equals its local committed blob exactly.

**The frozen-collector rule holds.** `16271526…` is byte-identical at
`13f71132`, `53b353f0`, `84acded2`, `8a4ed56f`, `6f178f14`, `b65c81f5` and
`6f276ade`; `hf_lfm2_canonical.py` likewise (`789c6a6f…`). Neither file appears
in the diff.

Two consequences predicted before launch, both confirmed by the run:

* `hf_lfm2_dense_budget_model_fold.py` did **not** move, and its digest feeds
  `numeric_contract.basis_config.adapter_source_sha256`. So the balanced-basis
  phase key should **not** rotate this time (W13K's did, `6e423f93…` →
  `52cbe3fd…`, precisely because that file moved). Confirmed — §5.
* `experiments/run_lfm2_dynamic_catalog.py` did not move, so the option surface
  should be unchanged. Confirmed — §2.3, and the run's `configuration_sha256`
  came back **identical to W13K's**, the first time in this arm's five attempts
  that digest has not moved on a bump.

### 1.1 The PYTHONPATH trap, again

Per `NOTES-W13K` §1.1 the venv's default `sys.path` resolves `neural_forge` to
the stale `/home/cereal/model-surgery-lab/neural-forge-src/` checkout. Every
`--help`/import check here exported `PYTHONPATH="$D/src:$D"` first, and the audit
script prints the resolution it actually got:

```
pkg : …/deployments/neural-forge-postpub-e893767a/src/neural_forge/__init__.py
coll: …/deployments/neural-forge-postpub-e893767a/src/neural_forge/hf_lfm2_circuit_trace_cache.py
```

---

## 2. Launcher verification — unchanged, re-audited flag by flag, twice

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`

```
sha256  e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda   <- the expected v2-ON form
cmp     byte-identical to nf-launch-r12-round-refold.sh.bak-20260805T104136Z
bash -n clean
size    25,122 bytes, 490 lines
mtime   2026-08-05 02:31:11 -0600   (untouched since NOTES-W13I installed it)
```

**Nothing had to be restored.** The audit was run twice — once against the
`6f276ade` deployment and again after the rollback to `b65c81f5` — with
identical results both times
(`flagcheck-w13l.out`, `flagcheck-w13l-b65c81f5.out`). The peer's own launcher
copy (`e652f576…`, which *does* pin a phase key) is a separate file and was
never consulted.

### 2.1 The r13 arm, read off the installed file (refold stage, lines 289–490)

All 23 required flags verified **PRESENT** by literal string match, values
included:

| requirement | flag as installed |
|---|---|
| margin observability | `--dense-vocabulary-margin-observability` |
| **streaming margin v2 ON** | `--dense-streaming-vocabulary-margin-v2` |
| margin pair chunk 256 | `--dense-streaming-vocabulary-margin-pair-chunk-size 256` |
| writer-reclosure-acceptance | `--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement` |
| **gate tolerance 1e-6** | `--dense-moving-gauge-boundary-regression-tolerance 1e-6` |
| terminal floor | `--dense-closed-loop-terminal-composition-floor` + `--dense-closed-loop-terminal-regression-tolerance 0.25` + `--dense-closed-loop-terminal-absolute-floor-minimum-layer 0` |
| terminal tripwire (advisory) | `--dense-closed-loop-terminal-free-run-tripwire` |
| telemetry | `--dense-closed-loop-solver-conditioning-diagnostics`, `--dense-closed-loop-gate-operating-point-diagnostics`, `--dense-closed-loop-depth-telemetry` |
| **EIV multiplier 0** | `--dense-closed-loop-input-error-multiplier 0` |
| fingerprint install verification | `--dense-moving-gauge-install-verification fingerprint` |
| moving gauge v3 | `--dense-body-reclosure-mode moving_gauge_v3` |
| sharded replay pair | `--dense-replay-devices cuda:0,cuda:4` |
| merged captures | `--dense-closed-loop-merge-adjacent-layer-captures` |
| canonical prefix replay | `--dense-closed-loop-replay-mode prefix_by_prefix_canonical` |
| phase caches ON | `--derived-global-topology-cache-root`, `--balanced-basis-phase-cache-root`, `--balanced-basis-phase-event-log`, `--balanced-basis-phase-hmac-key-file`, `--balanced-basis-phase-hmac-key-id` |

### 2.2 The must-be-absent list, checked by name over the whole 490-line file

```
ABSENT   --dense-closed-loop-input-error-covariance
ABSENT   --dense-closed-loop-margin-row-weights
ABSENT   --dense-closed-loop-terminal-free-run-stop
ABSENT   --dense-resume-from-capsule
ABSENT   --dense-terminal-only-restart-resume
ABSENT   --dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only
ABSENT   --dense-closed-loop-intermediate-numeric-checks-diagnostic-only
ABSENT   --child-replay-phase-cache-root
comment  --balanced-basis-existing-phase-key-sha256   line 82 only:
         "#     No --balanced-basis-existing-phase-key-sha256 is pinned (r11's"
```

**No phase key is pinned.** The only occurrence of that spelling in the file is
the comment saying so. That matters more than usual this wave, because the
peer's launcher copy pins `52cbe3fd…` and this arm deliberately does not — it
resolved the same entry by content instead (§5).

### 2.3 Flag/`--help` reconciliation

```
deployed --help                181 options parsed   (52,062 bytes, exit 0)
refold stage passes            108 distinct flags
used flags absent from --help  0
option set vs W13K's 181       added: none   removed: none
used-flag set vs W13K's 108    cmp IDENTICAL
```

The option surface did not move at all — expected, since the runner file did not
move. The 108-flag set is byte-for-byte the set `NOTES-W13I` §2.1 and
`NOTES-W13K` §2.3 verified. All eight relaxation levers were confirmed twice:
present in `--help` (so the spelling is real and the absence is not a typo) and
absent from the launcher's argument list.

---

## 3. Run path and preserved directories

The refold path was **already clear** on arrival — W13K moved its own killed
partial aside before handing off — so nothing had to be moved before 5a.

After 5a failed closed, its directory was preserved, not deleted:

```
from  …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
to    …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-closed-d296b602-180622-w13l
mv -n, exit 0, at 2026-08-06T18:10:09Z
du -sb before 1,293,369   after 1,293,369   (identical)
run.log 687,464 B   receipt.json 284,188 B   diagnostics.json 320,413 B
```

Full inventory under `…/catalog-integrated-carrier/` — none deleted:

| directory | |
|---|---|
| `…-round-0-refold.failed-merged-basis-104326` | W13D's fail-closed run |
| `…-round-0-refold.failed-config-streaming-v2-104727` | W13F's fail-closed run |
| `…-round-0-refold.killed-external-220056` | W13I's 8-hour run (2 accepted layers) |
| `…-round-0-refold.killed-external-223936-w13k` | W13K's run, 512 events |
| `…-round-0-refold.failed-closed-d296b602-180622-w13l` | **5a**, this wave, with a real receipt |
| `…-round-0-refold` | **5b**, this wave, LIVE |
| `a0024-…-w13j-cold`, `a0025-…-w13k-cold`, `a0026-…-w13l-cold`, `a0027-…-w13m-resume-d296b602` | peer runs — **read only**, never modified |

`a0026`'s partial and its capsules remain the peer's to resume.

---

## 4. Detachment — proved from `/proc` for both launches

```bash
cd /home/cereal/model-surgery-lab/deployments
export SOURCE_COMMIT=<full sha>
setsid nohup bash ./nf-launch-r12-round-refold.sh refold \
    > /home/cereal/nf-w13l-r13/launch-refold.out 2>&1 < /dev/null &
disown
```

| | 5a (`6f276ade`) | **5b (`b65c81f5`, live)** |
|---|---|---|
| `run.pid` | `2953194` | **`3030874`** |
| `ppid` (from `/proc/<p>/stat`) | 1 | **1** |
| `pgrp` == `session` | 2953167 | **3030849** |
| `tty_nr` | 0 | **0** |
| parent pid 1 cmd | `/usr/lib/systemd/systemd --system …` | same |
| launcher shell | already exited | already exited |
| `fd/0` | `/dev/null` | `/dev/null` |
| `fd/1`, `fd/2` | `<run>/run.log` | `<run>/run.log` |
| `source-commit.txt` | `6f276ade…` | **`b65c81f5d086830247b0a418a78d54fb12975ca6`** |

**PPID is 1 (systemd)** in both cases, tty-less, in its own setsid session whose
leader is gone. Both streams land in `run.log` and Python runs `-u`, so a
traceback, `MemoryError`, `abort()` or CUDA error is captured — which is exactly
how 5a's failure got recorded in full.

Proofs: `detachment-proof-5a-6f276ade.txt`, `detachment-proof.txt`.

---

## 5. Phase observations — everything reproduces, and nothing was pinned

Identical in both launches, and identical to W13D/W13I/W13K:

**Global topology — 6 hits, 0 misses.** The same three phase keys
(`a5eaf8da…`, `9c9d568f…`, `edea7406…`) with the same completion digests
(`ec925599…`, `ca043086…`, `ecec1dbd…`), looked up once before the evidence set
opened and once after. `selected_hidden_size 816`,
`frontier_sha256 8a4a1ac8e0959f1c55ce81e7f0f8efba208a1c16fb9fdb95fee986fa4077cdc7`.

**Evidence set — identical identity, five waves running.**

```
member_count                     2
evidence_set_cache_key           f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
evidence_identity_sha256         75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5
base_batch_count   256           base_path_count        67,583
extension_path_count 6,406       merged_batch_count     371
round_count          1           parent_forwards        0
all_extension_rows_are_fit_rows  true
merged_trace_rows_sha256         5046c63988800bcc343277b1f34d943f95968a2b3ca004e0144c38f6f5041c61
```

`67,583 + 6,406 = 73,989` paths over 371 merged batches — the same corpus, and
the row digest is byte-identical to W13I's and W13K's.

**Balanced basis — key did NOT rotate, and hit again.**

```
balanced_basis_phase_cache_hit   52cbe3fd96650d284b2e2a03395515c3fb467ce846a014712f5d5c1a4e8f683d
completion_sha256                975e828634a3be4e9bfd3df2e4bb23f21e86e3f9ba606262bd52507bd2d20ac3
parent_forward_calls 0   trace_tensor_loads 0   training_steps 0
```

As predicted in §1, `hf_lfm2_dense_budget_model_fold.py` did not move, so the key
stayed `52cbe3fd…` rather than rotating as it did in W13K. It hit both times.
**Nothing was pinned** — the launcher passes no phase key; the entry was resolved
by content, and its `dependency_sha256s` bind this run's own
`on_policy_evidence_identity 75217282…` and
`merged_sealed_evidence d4518f05…`.

Carried over from `NOTES-W13K` §6.3 and still true: **this run did not rebuild
the merged basis, it resumed the peer's `a0025` payload.** Content-addressed,
HMAC-chained, nothing pinned — but a future claim that this arm built its own
basis end to end would be false for this launch too.

---

## 6. The killer, named

W13K installed `killspy.bt` (bpftrace on `tracepoint:signal:signal_generate`) to
catch the sender of the signal that had killed four runs. **It fired.**

```
2026-08-05T19:07:26Z sig=15 sender_pid=121286 sender_comm=bash -> target_pid=3551402 target_comm=python
```

`3551402` is the pid `NOTES-W13K` §9 recorded for the peer's `a0026-…-w13l-cold`
run. So the signal was a plain `SIGTERM` from a `bash`, and it killed **the
peer's own run**, not just this arm's. That is consistent with the
session-cleanup / preflight pattern-kill identified in coordd event 3697.

Two reading caveats, carried forward from W13K §8.4 and re-confirmed here: the
printed timestamps are **server-local (UTC-6)** despite the trailing `Z`
(verified this wave — `date` returned `17:43:47Z` / `11:43:47-0600`
simultaneously), so `19:07:26` local is **2026-08-06T01:07:26Z**; and the tracer
sees only signals, so a process that exits on its own leaves no line.

The only other `sender_comm=bash` line in the whole 8,987-line log is W13K's own
deliberate self-test against a throwaway `sleep`.

### 6.1 killspy status — it had expired, and was re-armed

W13K armed it with `timeout 6h` at 2026-08-05T22:39Z; it ran the full six hours
and exited normally at **2026-08-06T04:38Z**, ~13 hours before this wave started.
It was found dead (`pgrep -af bpftrace` empty) and re-armed:

```
sudo setsid nohup timeout 12h bpftrace /home/cereal/nf-w13k/killspy.bt \
     >> /home/cereal/nf-w13k/killspy.log 2>&1 < /dev/null &
```

```
re-armed        2026-08-06T17:43:47Z   (server-local 11:43:47-0600)
bpftrace pid    2949783
appending at    byte offset 945,873 of killspy.log — W13K's 8,987 lines preserved
window          12 h, i.e. until ~2026-08-07T05:43Z
```

**Deviation, stated plainly:** the task said to re-arm "exactly as before"
(`timeout 6h`). A 12-hour window was used instead so that a kill late in a
14-layer run is still captured — at a0026's pace 14 layers is ~4 h plus preamble,
and a 6 h window would have expired mid-run for any resume. Nothing else about
the invocation changed.

Alive at handoff, and **it has recorded no signal at all against 5b's pid
`3030874`.**

---

## 7. 5a: `6f276ade` fails closed — `d296b602` is the cause

### 7.1 What happened

5a launched at 17:44:54Z and cleared every wall: preflight, `parent_verified` in
9.9 s, `carrier_verified` at 6m12s, three global-topology hits, the evidence set,
the balanced-basis hit, the structural fold. Then:

```
18:05:45.558  child_closed_loop_layer_started            layer_index 0
18:05:45.560  child_closed_loop_layer_capture_started    pre_producer
18:06:22.794  failure       Lfm2DynamicClosedLoopAdapterError
                            fail_closed true
                            failure_category fold_or_infrastructure_failure
18:06:22.825  final_receipt receipt_status failed_closed
                            receipt_sha256 67391862cd527117e42905c34de42eecb866af69b23e30c58dd584c26c677736
```

**37.2 seconds into the first layer capture.** The message, from `receipt.json`:

```
Selected parent component payload changed after authenticated load.
```

This is **not** the W13K failure mode. It is a fail-closed refusal with a written
receipt and `diagnostics.json` — the first real receipt this arm has produced at
a layer boundary since W13F. `killspy` recorded **no signal** against pid
`2953194`, and the tracer was alive throughout, so an external kill is excluded
by instrument rather than by inference.

### 7.2 It reproduces in the peer's own run

The peer's `a0027-…-w13m-resume-d296b602`, launched at `d296b602` at 01:08Z,
failed identically:

| | peer `a0027` (`d296b602`) | this arm 5a (`6f276ade`) |
|---|---|---|
| layer | 8 | 0 |
| `capture_started` | 2026-08-06T01:26:46.885Z | 2026-08-06T18:05:45.560Z |
| `failure` | 2026-08-06T01:27:26.317Z | 2026-08-06T18:06:22.794Z |
| elapsed | 39.4 s | 37.2 s |
| type | `Lfm2DynamicClosedLoopAdapterError` | same |
| message | `Selected parent component payload changed after authenticated load.` | same |
| receipt status | `failed_closed` | `failed_closed` |

Two independent runs, different layer indices, same failure at the same phase.
Layer-index independent, so not a data artefact of layer 0.

### 7.3 Attribution, by blob archaeology rather than commit titles

```
occurrences of "Selected parent component" in hf_lfm2_dynamic_closed_loop_adapter.py

  8a4ed56f    0     file 13,877 lines
  6f178f14    0     file 15,934 lines
  b65c81f5    0     file 15,934 lines      <- a0026 ran 8 layers here
  d296b602   19     file 17,036 lines      <- +1,102 lines, machinery appears
  6f276ade   19     file 17,036 lines      (docs-only commit on top)
```

The entire "selected parent component" cache is **introduced by `d296b602`**
("Cache invariant parent and tokenizer evidence"). `6f276ade` cannot be the cause
— it touches only `docs/CURRENT_FOLDING_GOAL.md`. `b65c81f5` cannot be the cause
— the machinery does not exist there, and `a0026` completed 8 layer captures on
it.

The raise site is `hf_lfm2_dynamic_closed_loop_adapter.py:4914`, a post-load
re-validation of the new cache:

```python
payload = self.selective_session.load_selected_tensors(reference.index, missing_names)
if (payload.released
        or payload.batch_index != reference.index
        or payload.input_sha256 != reference.input_sha256
        or payload.session_sha256 != self.selective_session.evidence_sha256
        or payload.ordered_tensor_names != missing_names
        or payload.selection_sha256 != expected_selection_sha256
        or payload.logical_bytes != expected_bytes
        or tuple(sorted(payload.tensors)) != missing_names):
    raise Lfm2DynamicClosedLoopAdapterError(
        "Selected parent component payload changed after "
        "authenticated load.")
```

Which of those eight disjuncts trips is **not** determined here — the exception
carries no field detail and this wave did not instrument the peer's code. That
is left open deliberately (§11).

### 7.4 What was done about it, and what was not

**Not done:** nothing was weakened. The check was not bypassed, no relaxation
lever was pulled, `--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only`
stayed off, no retry-in-place was attempted, and no peer source file was edited.

**Done:** the deployment was rolled back by exactly the two commits carrying the
regression, to `b65c81f5`:

```
HEAD-before-rollback.txt   6f276adee473ca86568330f91b21eba487c4b8c1   (status 0 bytes, clean)
git reset --hard           b65c81f5d086830247b0a418a78d54fb12975ca6
git status --porcelain     empty
grep -c "Selected parent component" src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py   -> 0
```

with the digest audit re-run:

| file | at `b65c81f5` | |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526…` | frozen collector, unchanged |
| `hf_lfm2_canonical.py` | `789c6a6f…` | unchanged |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77…` | unchanged → basis key stays `52cbe3fd…` |
| `experiments/run_lfm2_dynamic_catalog.py` | `2af6de96…` | unchanged → 181 options, 108 flags unchanged |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `9606e1e4…` | **reverts to exactly W13K's proven-good adapter** |
| `hf_lfm2_structural_residual_attention_fold.py` | `1173468417…` | b65c81f5's version — *identical to the one 5a exercised cleanly* |

That last row is why `b65c81f5` and not `6f178f14`: 5a itself ran
`b65c81f5`'s structural fold through `dense_structural_fold_completed` without
complaint earlier the same hour, so the speed-relevant commit is exercised on
this arm, not merely on the peer's.

**Deviation, stated plainly:** the task instructed deploying `6f276ade`. That
commit is deployed, verified and *proven unable to fold*. `b65c81f5` is a strict
prefix of the instructed range and carries the invariant folding-group cache the
task cited as the reason for the bump; `d296b602`'s "already running in
production once" premise turned out to hold only for `b65c81f5` — `d296b602`'s
sole prior run, `a0027`, had failed closed. Rolling back rather than relaunching
into a deterministic 21-minute failure was the judgement call made here, and it
is reversible: the deployment can be reset forward to `6f276ade` in one command
once `d296b602` is fixed.

The peer was told, with evidence, via coordd (`event_id 3702`, thread
`7980649c-…`, message `f7389f47-…`). No reply had arrived at handoff; the only
inbound message after event 3690 was an unattended-adapter bounce saying direct
user review is required, so **the pattern-kill of §6 should still be assumed
live.**

---

## 8. 5b: launch timeline at `b65c81f5` (UTC, 2026-08-06)

| | |
|---|---|
| 18:10:09 | 5a's failed run preserved as `…failed-closed-d296b602-180622-w13l` (byte counts equal) |
| 18:10:5x | deployment `reset --hard b65c81f5`, tree clean, 6 digests re-checked, adapter machinery count 0 |
| 18:11:0x | launcher re-audited: `e496c102…`, 181 options, 108/108 flags, must-be-absent list clean |
| **18:11:22.499** | **`refold` stage launched** under `setsid`, `SOURCE_COMMIT=b65c81f5d086830247b0a418a78d54fb12975ca6` |
| 18:11:34.003 | `run_started`; `configuration_sha256 c79cbe97…` |
| **18:11:41.662** | **`parent_verified` — clause A cleared in 7.7 s** |
| 18:11:58.965 | detachment proved from `/proc` (§4); monitors restarted |
| 18:17:12.062 | `parent_cache_verified` |
| 18:17:39.809 | `tokenizer_bridge_verified` |
| 18:17:44.091 | `carrier_verified` (6m10s in) |
| 18:17:44.625 | 3 × `global_topology_phase_cache_hit`, 0 misses; `selected_hidden_size 816`, `frontier_sha256 8a4a1ac8…` |
| 18:23:53.906 | `parent_rows_materialized` — `trace_rows_sha256 5046c639…` |
| 18:24:44.061 | **`on_policy_extension_evidence_materialized`** — 2 members, 371 batches |
| 18:25:04.532 | the same three topology entries resumed a second time — 6 hits, still 0 misses |
| 18:29:31.715 | **`balanced_basis_phase_cache_hit` `52cbe3fd…`**, `completion 975e8286…`, 0 forwards / 0 tensor loads / 0 training steps |
| 18:30:40.417 | `dense_structural_fold_completed` |
| 18:30:41.037 | `dense_closed_loop_started` |
| **18:33:29.364** | **layer 0 `pre_producer` capture COMPLETED** — 205 records, `capture_sha256 23ca0961…`. *This is the wall 5a died on.* |
| 18:35:38.449 | layer 0 `post_producer` capture — `17e1c33b…` |
| 18:37:28.243 | layer 0 `post_consumer` capture — `9f50c4fa…` |
| **18:42:32.498** | **layer 0 COMPLETED — 11.6 min, ACCEPTED**, peak RSS 68.5 GiB, capsule 000000 published |
| 18:45:34.139 | layer 1 `post_producer` — `0a0d9ef9…` |
| 18:47:17.721 | layer 1 `post_consumer` — `1088b8ad…` |
| **18:52:29.662** | **layer 1 COMPLETED — 10.0 min, ACCEPTED**, peak RSS 69.3 GiB, capsule 000001 published |

`capture_sha256 23ca0961c8f4a7d904dc0e3c2251171536127066e09ff89663a3fa448ddfa292`
for layer 0 `pre_producer` is **byte-identical to W13K's** at `6f178f14` — the
capture is reproducible across the bump, which is the strongest available
evidence that `b65c81f5` did not perturb the fold path.

---

## 9. Pace — this run vs a0026, and vs the pre-cache waves

Per-layer wall time, `child_closed_loop_layer_started` →
`child_closed_loop_layer_completed`:

| layer | a0026 (peer, `b65c81f5`, cold) | **5b (this arm, `b65c81f5`, refold)** |
|---|---|---|
| 0 | 11.4 min | **11.6 min** |
| 1 | 10.0 min | **10.0 min** |
| 2 | 14.1 min | — |
| 3 | 10.3 min | — |
| 4 | 17.8 min | — |
| 5 | 29.9 min | — |
| 6 | 18.8 min | — |
| 7 | 13.0 min | — |
| mean 0–7 | **15.7 min** | — |

The two arms are running at **the same pace to within seconds on the two layers
both have completed**. Against the pre-cache waves:

| | per layer |
|---|---|
| W13I @ `8a4ed56f` | **3 h 04 min** (with ~40-minute capture points) |
| a0026 @ `b65c81f5` | 15.7 min mean over 8 layers |
| **5b @ `b65c81f5`** | **10.8 min mean over 2 layers** |

That is roughly a **17× reduction** in per-layer cost against W13I. Phase-level
comparison against W13K:

| phase | W13K (`6f178f14`) | 5a (`6f276ade`) | **5b (`b65c81f5`)** |
|---|---|---|---|
| import → `parent_verified` | 7.4 s | 9.9 s | 7.7 s |
| carrier verification | 5m48s | 6m12s | 6m10s |
| global topology | 3 hits, ~0.09 s | 3 hits | 3 hits (6 total, 0 miss) |
| parent row materialization | 4m56s | 7m14s | 6m10s |
| merged balanced basis | 26 s (phase hit) | 27 s (phase hit) | 26 s (phase hit) |
| basis install + structural, 14 layers | 1m11s | 1m10s | 1m09s |
| layer 0 `pre_producer` capture | 2m23s | **died at 37 s** | 2m48s |
| **layer 0 decision** | never reached | never reached | **11.6 min, accepted** |

The early phases are ~5 % slower than W13K in wall time; the host has four
ComfyUI tenants today against three during W13K, so this is contention, not the
commits.

---

## 10. Current state and monitoring recipe

At **2026-08-06T18:54Z**:

```
deployment   /home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a
             HEAD b65c81f5d086830247b0a418a78d54fb12975ca6, git status --porcelain EMPTY
             frozen collector 16271526…, canonical 789c6a6f…
launcher     e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda  (unmodified)
run          …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
run.pid      3030874   ppid 1   sess 3030849   stat Sl   etime 47m09s   503% CPU
layers       2 completed, validation_accepted:true × 2, false × 0
failures     0 "stage":"failure", 0 Traceback
capsules     capsule-000000-c716f0ec…, capsule-000001-c47c9ed5…
peak RSS     69.28 GiB (runner-stamped); live RSS 44.5 GiB post-layer-1
killspy      bpftrace alive, 12 h window to ~2026-08-07T05:43Z, no signal against 3030874
```

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
W=/home/cereal/nf-w13l-r13
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold

$SSH --command-timeout 900 --execute "bash $W/poll.sh"                    # one-shot full poll
$SSH --execute "python3 $W/tl.py $RUN/run.log"                            # stage + per-layer timeline
$SSH --execute "cut -c1-200 $W/milestones.log"                            # decision lines only
$SSH --command-timeout 15000 --execute "bash $W/waitlayers.sh 6 14000"    # block until N layers done
$SSH --execute "bash $W/waitfor.sh '<stage regex>' 780"                   # block until a stage appears
$SSH --execute "grep target_pid=3030874 /home/cereal/nf-w13k/killspy.log" # did anything signal it
```

Note `--command-timeout`: the client default is 300 s, and the Bash tool
foregrounds for only 120 s, so long waits must be backgrounded.

`rss-sampler.sh` and `milestones.sh` bind to `run.pid` when they start and
self-exit when it dies; **restart them after any relaunch**:

```bash
setsid nohup bash $W/rss-sampler.sh > $W/rss-sampler.log 2>&1 < /dev/null &
setsid nohup bash $W/milestones.sh > $W/milestones-runner.log 2>&1 < /dev/null &
```

To reset the deployment forward once `d296b602` is fixed:

```bash
cd /home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a
git reset --hard <fixed sha>      # objects for 6f276ade are already present as refs/heads/w13l-import
```

### 10.1 The live risk: host memory

Host memory is the tightest resource and it **swings hard within a layer**:

```
18:11Z  launch          available 122 GiB   (runner not yet resident)
18:52Z  mid layer 1     available  39 GiB   runner RSS ~87 GiB
18:58Z  after layer 1   available  78 GiB   runner RSS ~44.5 GiB
```

So the 39 GiB reading was a within-layer peak, not a trend — the runner releases
roughly half its footprint at each layer boundary. Four ComfyUI tenants hold
~112 GB throughout (one, `comfyui_qwen_gpu1_burst`, at 64 GB with GPU 1 idle at
239 MiB / 0 %). a0026 peaked at 92.6 GiB and survived on this box, so there is
headroom — but a second large tenant starting mid-layer is the most likely cause
of a *genuine* OOM death from here.

Those containers are another tenant's and were **not** touched. If this run dies
without a killspy line and without a receipt, check `dmesg -T` for `oom-kill`
before assuming the pattern-kill.

---

## 11. Caveats

* **The instructed commit is deployed nowhere.** `6f276ade` was deployed,
  verified digest-by-digest, and demonstrated unable to complete a single layer
  capture. The live deployment is `b65c81f5`. This is a deliberate, reversible
  deviation (§7.4), not a silent substitution.
* **The `d296b602` root cause is attributed, not diagnosed.** Which of the eight
  disjuncts in the re-validation trips is unknown; the exception carries no field
  detail and no instrumentation was added to the peer's code. "The cache is
  wrong" and "the check is wrong" are both still open.
* **`b65c81f5` is exonerated only up to layer 1.** Two accepted layers on this
  arm and eight on the peer's is good evidence, not proof, that its structural
  fold rewrite is numerically inert. The `capture_sha256 23ca0961…` match against
  W13K is the strongest single datum.
* **The merged balanced basis was inherited, not built** (§5). Content-addressed
  and authenticated, nothing pinned — but it is the peer's `a0025` payload.
* **The killer is named but not disabled.** §6 names a `bash` sender; the peer
  has been told twice and has not confirmed. This run remains exposed, and the
  detachment proof in §4 explicitly does *not* protect against it.
* **`refused_transition_indices` is still unanswered.** Two layers are accepted;
  the r13 arm's headline question needs the sealed `diagnostics.json` at end of
  run. `validation_accepted` per layer remains the only live decision surface —
  the floor, boundary and margin numbers do not stream.
* **Nothing was weakened.** No flag relaxed, no fail-closed clause bypassed, no
  phase key pinned, no peer source edited, no retry-in-place after the
  fail-closed.
* **Read-only artifacts untouched**: the sealed r12 run directory, `round-000`,
  the sealed 2026-07-17 pack and both base caches were never written to. The
  peer's `a0024`–`a0027` directories and `/home/cereal/nf-w13l/` were **read** for
  §6–§7 and never modified. All five preserved failure/kill partials remain in
  place. W13K's `rss-samples.tsv` and `milestones.log` were preserved by pointing
  this wave's copies at `nf-w13l-r13/` (the stock `rss-sampler.sh` truncates its
  output file).
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
* **Not committed.** No `git add`, no commit, no push.

---

## 12. Artifacts

Server, under `/home/cereal/nf-w13l-r13/`:

| file | what it is |
|---|---|
| `nf-w13l.bundle` | the deployment bundle, `9143bd4a…`, 83,728 B |
| `HEAD-before.txt`, `status-before.txt` | pre-bump deployment state (`6f178f14…`, clean / 0 bytes) |
| `HEAD-before-rollback.txt`, `status-before-rollback.txt` | pre-rollback state (`6f276ade…`, clean / 0 bytes) |
| `flagcheck-w13l.sh` / `.out` | the flag audit at `6f276ade` |
| `flagcheck-w13l-b65c81f5.sh` / `.out` | the same audit re-run after the rollback |
| `runner-help-6f276ade.txt`, `runner-help-b65c81f5.txt`, `help-flags.txt`, `used-flags.txt` | the 181-option `--help` and the 108/108 reconciliation |
| `phase-cache-before.txt` | 18,943-entry phase-cache inventory taken before launch |
| `killspy-rearm.txt` | re-arm timestamp, log offset, and the UTC/local skew proof |
| `launch-start-5a-6f276ade.txt`, `launch-refold-5a-6f276ade.out`, `detachment-proof-5a-6f276ade.txt` | 5a's launch record |
| `launch-start.txt`, `launch-refold.out`, `detachment-proof.txt` | **5b's** launch record |
| `rss-samples-5a-6f276ade.tsv`, `milestones-5a-6f276ade.log` | 5a's samplers |
| `rss-samples.tsv`, `milestones.log` | **5b's** live samplers |
| `poll.sh`, `tl.py`, `timeline.py`, `waitfor.sh`, `waitlayers.sh`, `entryinfo.py` | the monitoring toolkit (§10) |
| `rss-sampler.sh`, `milestones.sh` | W13L copies, redirected so W13K's evidence survives |

Retained from W13K under `/home/cereal/nf-w13k/`: `killspy.bt` and
`killspy.log` (now 945,873 B of W13K window + this wave's appended 12 h window),
plus that wave's own artifacts, untouched.

Phase cache: **no new entry was published by either launch.** Both appended a
`hit` event to their own event log and resumed
`…/lfm2-circuit-balanced-residual-basis-v1/…/52cbe3fd…` (published by the peer's
a0025). The three global-topology entries were resumed, not rewritten.
`round-000`, the sealed r12 run directory, the sealed 2026-07-17 pack and both
base caches: untouched.
