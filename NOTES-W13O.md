# NOTES-W13O — r13 ceiling ARM A: the sealed run preserved, a cherry-picked deployment, and the norm ceiling launched

Wave 13O, branch `codex/lrc-blank-compression`. This wave answers `NOTES-W13M`'s
§1.2 finding — the reclosure gate accepted 82 writer rewrites, 81 upward, median
33x, 974x at layer 7 — by launching the `NOTES-W13N` §3 norm-ratio ceiling at
**2.0** on a production refold.

Three things happened, in this order:

1. **The sealed first composed r13 run was moved aside intact** (§1). Byte
   counts, sha256s, `du -sb` and file count are identical across the move.
2. **The deployment could not go to `d299f212` wholesale, and did not** (§2).
   The `d296b602` regression is live in that tree and was reproduced locally.
   Three commits were cherry-picked onto `b65c81f5` instead, and the resulting
   tree was **proved to fold** before it was deployed.
3. **ARM A is running**, detached, at ceiling 2.0 (§3–§5).

---

## 1. Preservation — the sealed run is intact under a new name

The refold path held the **sealed first composed r13 run** — receipt
`c9a8f746…`, published child, certificate, capsules. It was moved, never
deleted, with `mv -n`:

```
from  …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
to    …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-w13m-firstcomposed
mv -n -v, exit 0, at 2026-08-07T13:03:43Z   (rename within one filesystem)
```

| quantity | before | after |
|---|---|---|
| `receipt.json` bytes | **289,012** | **289,012** |
| `receipt.json` sha256 | `d76730a71d530b60102502d9089ba48b2b3dcff5499e263e0478b997e2d2c563` | **same** |
| `diagnostics.json` bytes | 323,507 | 323,507 |
| `diagnostics.json` sha256 | `0d053185dd09b872cd7ab30d0017648a72896c3548536c55b09453f140df57b5` | **same** |
| `run.log` bytes | 1,568,047 | 1,568,047 |
| `ledger.jsonl` bytes | 8,252 | 8,252 |
| `du -sb` | **101,019,877,001** | **101,019,877,001** |
| `find -type f \| wc -l` | **1,522** | **1,522** |

The receipt's own `receipt_sha256` field reads
`c9a8f7468934f55a97f37cd14e7a7fb91721657dfa2f54f4218d9a962d98e4c9` — the
`NOTES-W13M` receipt, confirmed in place after the move. `published-child/`
still carries `certificate.json` (7,923 B), `child/` and `provenance/`;
`restart-capsules/` still carries `capsule-000012-25386639…` and
`capsule-000013-19292673…`; `source-commit.txt` still reads `b65c81f5…`.

Nothing was deleted. All six previously preserved failure/kill partials and all
four peer directories (`a0024`–`a0027`) remain untouched. Proof:
`/home/cereal/nf-w13o/preservation.txt`.

---

## 2. Deployment — the check the task asked for, and what it forced

### 2.1 The pre-flight check said no

The task's own instruction was to verify before bundling. Run at local HEAD
`d299f212`:

```
.venv/Scripts/python.exe -m pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly

3 failed, 6 passed in 307.45s (0:05:07)

FAILED test_carrier_dense_core_alternates_fold_round_and_refold
FAILED test_bridged_carrier_lane_alternates_fold_round_and_refold
FAILED test_carrier_production_margin_pin_composes_through_a_round
```

The same three names `NOTES-W13M` §6.5 and `NOTES-W13N` §4.1 record. **The
`d296b602` regression is live in the `d299f212` tree** — `d296b602` is an
ancestor of `d299f212`, so its payload-revalidation code is in it. A wholesale
deployment to `d299f212` would have reproduced W13L's 5a: a deterministic
fail-closed ~37 s into the first layer capture. It was not done.

### 2.2 The cherry-pick, dry-run before it touched the deployment

Three commits were cherry-picked onto `b65c81f5`. The surfaces:

| commit | files | overlaps `d296b602`? |
|---|---|---|
| `7e25d781` smoke-subject fix | `run_lfm2_dynamic_catalog.py`, its test, `NOTES-W13M.md`, `responses.md` | **no** |
| `815e791e` ledger lift + telemetry + norm ceiling | adapter (+179), reclosure (+270), catalog (+100), 3 tests, `NOTES-W13N.md` | **yes — the adapter and its test** |
| `d299f212` retention test | `test_run_lfm2_dynamic_catalog_r13_retention.py` only | no |

`hf_lfm2_dynamic_closed_loop_adapter.py` is touched by `d296b602` (+1,200) and
by `815e791e` (+179), so a conflict was the expected failure mode. It was
dry-run first in an isolated `git clone --shared` at
`…/scratchpad/cp-dryrun`, **not** in the deployment and not in the local
working tree:

```
git checkout b65c81f5
git cherry-pick 7e25d781    -> clean
git cherry-pick 815e791e    -> Auto-merging src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py
                               Auto-merging tests/test_hf_lfm2_dynamic_closed_loop_adapter.py
                               clean, no conflict
git cherry-pick d299f212    -> clean
```

**No conflict.** Both files auto-merged: `815e791e`'s additions are in regions
`d296b602` does not touch, so removing `d296b602` from underneath them changes
only the line offsets.

### 2.3 The dry-run tree was proved to fold before it was deployed

This is the step that makes the deployment safe rather than merely clean:

```
PYTHONPATH=<dryrun>/src pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly

9 passed in 378.15s (0:06:18)
```

**3 failed / 6 passed at `d299f212` → 9 passed on the cherry-picked tree**, same
suite, same runner, same venv. The three `d296b602` failures are gone and
nothing else broke. Import resolution was pinned to the dry-run clone and
verified (`neural_forge.__file__` under `cp-dryrun/src`), because the venv
carries an editable `.pth` pointing at the main repo.

### 2.4 The deployment, and the tree-identity proof

```
bundle  b65c81f5..codex/lrc-blank-compression   (5 commits' objects)
size    179,521 bytes
sha256  23b88c293285846bda50b3180a226b60179c99a762e7f1153a429e9862577480   (identical both sides)
server  /home/cereal/nf-w13o/nf-w13o.bundle
verify  "is okay", contains d299f212…, requires b65c81f5…
```

Pre-state recorded before anything moved: `HEAD-before.txt` = `b65c81f5…`,
`status-before.txt` = **0 bytes** (clean tree, nothing overwritten). Then
`git fetch <bundle> …:refs/heads/w13o-import`, `checkout b65c81f5`, and the
same three cherry-picks. All three applied clean on the server too.

```
HEAD  = 2db9f0851694289d64a39f9ab3d0e40b714a219d      <- the arm's SOURCE_COMMIT
TREE  = 14dc0ebf266fb1bb1a95fa10326df98301a2c69c
git status --porcelain   EMPTY
```

**The deployed tree sha `14dc0ebf…` is byte-identical to the dry-run tree that
passed 9/9.** Commit shas differ (cherry-pick rewrites the committer and its
timestamp); the tree does not. That is the proof that what is deployed is what
was tested, and it is stronger than a per-file digest table.

Deployed working-tree digests:

| file | sha256 | |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **frozen collector — byte-identical, rule holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | unchanged |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77d041bb9cadac9b578dbc914ae59f37084846d1ea0e10397e37002a3f` | unchanged → basis key must not rotate |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `eea7ecfe49b5f5ec6b445867a3da00d327f7116735bd7bdf9977d17c1d82b41c` | new — W13N lift, no `d296b602` |
| `hf_lfm2_moving_gauge_residual_reclosure.py` | `c1f4462c17278a6d332273d7e6dcbf0ecde6d2e03e6c9aee660c5a531f14f1b3` | new — the ceiling |
| `experiments/run_lfm2_dynamic_catalog.py` | `9d072ccf6241d7338a26ee58c7517fe8d7a5276a9a3974336f7c8b4eac6fba6e` | new — smoke subjects + flag |

Semantic checks on the deployed tree:

```
grep -c "Selected parent component"  hf_lfm2_dynamic_closed_loop_adapter.py   -> 0   (d296b602 absent)
grep -c "_lifted_boundary_acceptance_ledger"                                  -> 2   (W13N Fix 1 present)
grep -c "writer_reclosure_norm_ratio_ceiling"  …reclosure.py                  -> 17  (W13N Fix 3 present)
grep -c "evaluate_published_child_reload"      run_lfm2_dynamic_catalog.py    -> 2   (W13M §6.4 present)
```

**`SOURCE_COMMIT = 2db9f0851694289d64a39f9ab3d0e40b714a219d`.**

---

## 3. Launcher — one line added

Timestamped backup taken first, `cp -p`:

```
/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh.bak-w13o-20260807T130654Z
sha256 e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda   <- the expected v2-ON form
```

The complete diff against that backup:

```diff
436a437
>     --dense-moving-gauge-writer-norm-ratio-ceiling 2.0 \
```

In context, inserted immediately after the writer-reclosure-acceptance lines:

```
    --dense-body-reclosure-mode moving_gauge_v3 \
    --dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement \
    --dense-moving-gauge-boundary-regression-tolerance 1e-6 \
    --dense-moving-gauge-writer-norm-ratio-ceiling 2.0 \          <- added
    --dense-moving-gauge-install-verification fingerprint \
```

```
bash -n     CLEAN
before      25,122 bytes, 490 lines, e496c102…
after       25,179 bytes, 491 lines, 6098d17904e95b6f677d4ae65681caee8277f92df4636fdca333618baf616873
```

Everything else is unchanged, and was re-audited rather than assumed:

* gate tolerance still `1e-6`, floor block intact
  (`--dense-closed-loop-terminal-composition-floor`,
  `--dense-closed-loop-terminal-regression-tolerance 0.25`,
  `--dense-closed-loop-terminal-absolute-floor-minimum-layer 0`,
  `--dense-closed-loop-terminal-free-run-tripwire`)
* telemetry on (`--dense-closed-loop-depth-telemetry`,
  `--dense-closed-loop-solver-conditioning-diagnostics`,
  `--dense-closed-loop-gate-operating-point-diagnostics`)
* `--dense-closed-loop-input-error-multiplier 0` (EIV 0)
* `--dense-moving-gauge-install-verification fingerprint`
* `--dense-streaming-vocabulary-margin-v2`
* **no pinned phase keys**, and the must-be-absent list is clean:

```
--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only : 0
--balanced-basis-phase-key                                          : 0
--global-topology-phase-key                                         : 0
--dense-closed-loop-terminal-composition-floor-disabled             : 0
```

Flag reconciliation against the deployed runner's `--help` (with
`PYTHONPATH=$D/src:$D`, so the `NOTES-W13K` §1.1 stale-checkout trap is avoided
— resolution printed and verified to the deployment):

```
options in --help          184        (181 before; +3 from 815e791e)
distinct flags used        109        (108 before; +1, the ceiling)
unknown flags              0          ALL USED FLAGS EXIST IN HELP
--dense-moving-gauge-writer-norm-ratio-ceiling   present at help lines 72 and 391
```

---

## 4. Launch and detachment

```
launch start   2026-08-07T13:14:52.411Z
command        SOURCE_COMMIT=2db9f085… setsid nohup bash ./nf-launch-r12-round-refold.sh refold
               > /home/cereal/nf-w13o/launch-refold.out 2>&1 < /dev/null &  ; disown
preflight      OK  (child bridge matches sealed bridge; both base capture selections verified;
                    round 0 carries a target-token preimage for 115 batches)
run.pid        2427715
run dir        …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
```

The ceiling is confirmed **on the live process command line**, not merely in the
launcher file:

```
… --dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement
  --dense-moving-gauge-boundary-regression-tolerance 1e-6
  --dense-moving-gauge-writer-norm-ratio-ceiling 2.0
  --dense-moving-gauge-install-verification fingerprint …
```

Detachment, from `/proc` (`detachment-proof.txt`):

| | |
|---|---|
| `run.pid` | **2427715** |
| `PPID` (from `/proc/2427715/stat`) | **1** |
| pid 1 cmdline | `/usr/lib/systemd/systemd --system --deserialize=158 splash` |
| `pgrp` == `session` | 2427689 == 2427689 |
| `tty_nr` | **0** (no controlling terminal) |
| `fd/0` | `/dev/null` |
| `fd/1`, `fd/2` | `<run>/run.log` |
| launcher shell | already exited |
| `source-commit.txt` | `2db9f0851694289d64a39f9ab3d0e40b714a219d` |

Python runs `-u` and both streams land in `run.log`, so a traceback,
`MemoryError`, `abort()` or CUDA error is captured.

**killspy re-armed** — its 12 h window from W13L had expired at ~05:43Z and
`pgrep -af bpftrace` was empty:

```
sudo setsid nohup timeout 12h bpftrace /home/cereal/nf-w13k/killspy.bt \
     >> /home/cereal/nf-w13k/killspy.log 2>&1 < /dev/null &

re-armed      2026-08-07T13:14:38Z  (server-local 07:14:38-0600)
bpftrace pid  2426750
appending at  byte offset 2,885,125 — every prior line preserved
window        12 h, i.e. until ~2026-08-08T01:14Z
```

Host at launch: all 7 GPUs idle (0–24 MiB used, 0 % util — ComfyUI tenants
stopped as reported), 231 GiB available of 251 GiB.

---

## 5. Early phases — everything reproduces, and the sealed basis entry hit

```
2026-08-07T13:15:02.212Z  run_started            configuration_sha256 38d1766b8ad0228c9511d1399331b7c9a5343addc68de02a40bdf17248ebb102
2026-08-07T13:15:11.821Z  parent_verified        <- clause A cleared in 9.6 s
2026-08-07T13:21:26.868Z  parent_cache_verified
2026-08-07T13:21:54.011Z  tokenizer_bridge_verified
2026-08-07T13:21:58.801Z  carrier_verified       <- 7m06s in
2026-08-07T13:22:00.412Z  global_frontier_preplanned
2026-08-07T13:27:07.722Z  parent_rows_materialized
2026-08-07T13:27:57.411Z  on_policy_extension_evidence_materialized
2026-08-07T13:28:28.823Z  wiring_planned
2026-08-07T13:28:29.938Z  system_fold_started
2026-08-07T13:32:19.624Z  balanced_basis_merged_evidence_bound
2026-08-07T13:32:43.748Z  balanced_basis_merged_evidence_bound
2026-08-07T13:32:46.561Z  balanced_basis_phase_cache_hit
```

**`configuration_sha256` rotated to `38d1766b…`**, and it had to: the ceiling is
a new configured value. This is the digest-level confirmation that the flag
entered the run's configuration rather than being silently ignored.

**Global topology — 6 hits, 0 misses.** The same three phase keys
(`a5eaf8da…`, `9c9d568f…`, `edea7406…`) with the same completion digests
(`ec925599…`, `ca043086…`, `ecec1dbd…`) as W13D/W13I/W13K/W13L/W13M.

**Evidence set — 2 members, identity unchanged for a sixth wave.**

```
member_count               2
evidence_set_cache_key     f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
```

Byte-identical to the sealed run's `parent.on_policy_evidence_set`
(`NOTES-W13M` §5).

**Balanced basis — the sealed run's entry HIT, content-addressed.**

```
balanced_basis_phase_cache_hit
phase_key_sha256      52cbe3fd96650d284b2e2a03395515c3fb467ce846a014712f5d5c1a4e8f683d
completion_sha256     975e828634a3be4e9bfd3df2e4bb23f21e86e3f9ba606262bd52507bd2d20ac3
parent_forward_calls  0
trace_tensor_loads    0
training_steps        0
```

Exactly the key and completion the sealed run used. As predicted in §2.4 from
`hf_lfm2_dense_budget_model_fold.py` being unchanged (`c2797d77…`), the key did
**not** rotate despite three new commits, and the entry was resolved by content
— the launcher passes no phase key. Carried forward unchanged from
`NOTES-W13L` §5 and `NOTES-W13M` §5: **this run did not build the merged basis,
it inherited the peer's `a0025` payload.** Content-addressed and HMAC-chained,
nothing pinned, but a claim that this arm built its own basis end to end would
be false for this launch too.

Pace against the sealed run (W13L 5b, which became W13M): `parent_verified` 9.6 s
vs 7.7 s, `carrier_verified` 7m06s vs 6m10s. Slightly slower on the preamble
despite an empty GPU host, well within wave-to-wave variation.

---

## 6. Layer progress at handoff

At **2026-08-07T14:09:13Z**, 54 minutes after launch:

```
2026-08-07T13:33:52.540Z  dense_structural_fold_completed
2026-08-07T13:33:53.123Z  dense_closed_loop_started
2026-08-07T13:34:07.867Z  child_closed_loop_layer_started              layer 0
2026-08-07T13:45:09.450Z  child_closed_loop_layer_validation_completed layer 0  validation_accepted TRUE
2026-08-07T13:45:23.954Z  child_closed_loop_restart_capsule_published  layer 0
2026-08-07T13:45:23.954Z  child_closed_loop_layer_completed            layer 0
2026-08-07T13:55:10.391Z  child_closed_loop_layer_validation_completed layer 1  validation_accepted TRUE
2026-08-07T13:55:26.891Z  child_closed_loop_layer_completed            layer 1
2026-08-07T14:08:49.734Z  child_closed_loop_layer_validation_completed layer 2  validation_accepted TRUE
2026-08-07T14:09:13.023Z  child_closed_loop_layer_completed            layer 2
2026-08-07T14:09:13.023Z  child_closed_loop_layer_started              layer 3   <- live
```

| layer | wall | accepted |
|---|---|---|
| 0 | 11 m 16 s | **true** |
| 1 | 10 m 03 s | **true** |
| 2 | 13 m 46 s | **true** |

```
layers accepted   3        validation_accepted:false   0
failure stages    0        Traceback lines             0
restart capsules  2        (keep=2, rotating as designed)
replay sharding   cuda:0,cuda:4 — 4 shard completions on layer 0, 3 on each of layers 1-2
runner peak RSS   90.97 GiB (runner-stamped)   live RSS 39.8 GiB post-layer-2
host available    191 GiB of 251 GiB
killspy           alive; 0 signals against pid 2427715
process           pid 2427715  ppid 1  sess 2427689  stat Sl  etime 56m
```

**~10–14 min per layer**, which is the expected pace. At this rate the
fourteenth layer install lands around 16:45–17:15Z, with the body reclosure —
the stage the ceiling actually gates — after that.

One thing to be clear about for whoever monitors: **the floor, boundary and
norm-ratio numbers do not stream.** The live validation event carries exactly

```json
{"stage":"child_closed_loop_layer_validation_completed","layer_index":0,
 "validation_accepted":true,
 "validation_decision_sha256":"c84628c14fa2223a797c5c8e7ef56b7f9b3db76bd09c8213976617428b22a966",
 "process_peak_rss_bytes":73667432448, …}
```

and no metrics block. `validation_accepted` is the only live decision surface,
exactly as `NOTES-W13L` §11 recorded. The §2-style terminal-relL2 table cannot
be built until `diagnostics.json` and `receipt.json` are sealed at end of run.

Likewise the ceiling's configured value is **not** echoed into `run.log`. Its
three live proofs are the process command line (§4), the rotated
`configuration_sha256 38d1766b…` (§5), and — at seal time — `receipt.json`.

---

## 7. Monitoring recipe

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
W=/home/cereal/nf-w13o
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold

$SSH --command-timeout 900 --execute "bash $W/poll.sh"                     # one-shot full poll
$SSH --execute "python3 $W/w13o_tl.py timeline"                            # stage timeline
$SSH --execute "python3 $W/w13o_tl.py basis"                               # balanced-basis hit detail
$SSH --execute "bash $W/ceiling.sh"                                        # what the CEILING did (see below)
$SSH --execute "cut -c1-200 $W/milestones.log"                             # decision lines only
$SSH --command-timeout 15000 --execute "bash $W/waitlayers.sh 6 14000"     # block until N layers done
$SSH --execute "grep target_pid=2427715 /home/cereal/nf-w13k/killspy.log"  # did anything signal it
```

`--command-timeout` matters: the client default is 300 s and the Bash tool
foregrounds for only ~120 s, so long waits must be backgrounded.

`rss-sampler.sh` and `milestones.sh` bind to `run.pid` at start and self-exit
when it dies; **restart them after any relaunch**:

```bash
setsid nohup bash $W/rss-sampler.sh  > $W/rss-sampler.log     2>&1 < /dev/null &
setsid nohup bash $W/milestones.sh   > $W/milestones-runner.log 2>&1 < /dev/null &
```

**The ceiling's own evidence does not exist until the body reclosure runs**,
which is after all fourteen layer installs. `ceiling.sh` reads the keys
`NOTES-W13N` §3.1 added, all of which arrive at once at that point:

```
norm_ratio_ceiling_refused_writer_overwrites
norm_ratio_ceiling_refused_transition_indices
boundary_metric_refused_writer_overwrites
refused_writer_overwrites  /  refused_by_sweep
maximum_accepted_write_norm_ratio
unmeasured_write_norm_ratio_writer_overwrites
boundary_acceptance_refused_transition_indices_json      <- W13I §11's headline question
boundary_acceptance_decisions_json                       <- the full ledger, W13N Fix 1
```

Because `815e791e` is deployed, those keys now reach
`closed_loop.body_reclosure.compiled.metadata` and therefore the receipt. The
sealed run could not answer `refused_transition_indices`; this one should.

### 7.1 killspy expires before this run is expected to finish

The window is 12 h to **~2026-08-08T01:14Z**. The sealed run took **16 h 34 m**.
At the same pace this run ends ~2026-08-08T05:49Z, so the last ~4.5 h are
uninstrumented. Re-arm before then:

```bash
sudo setsid nohup timeout 12h bpftrace /home/cereal/nf-w13k/killspy.bt \
     >> /home/cereal/nf-w13k/killspy.log 2>&1 < /dev/null &
```

12 h was used because the task specified it; the gap is stated rather than
silently widened.

### 7.2 Risks carried forward

* **The pattern-kill of `NOTES-W13L` §6 should still be assumed live.** It was
  named (`sig=15` from a `bash`), never disabled, and the peer never confirmed.
  The detachment proof in §4 explicitly does not protect against it.
* **Host memory.** The sealed run peaked at ~87 GiB RSS mid-layer. 231 GiB is
  available now and every ComfyUI tenant is stopped, so headroom is far better
  than W13L's. If the run dies with no killspy line and no receipt, check
  `dmesg -T` for `oom-kill` before assuming the pattern-kill.

---

## 8. Caveats

* **The instructed commit `d299f212` is not deployed, and deliberately so**
  (§2.1). Its tree cannot fold — 3 of 9 carrier e2e tests fail with
  `d296b602`'s `Selected parent component payload changed after authenticated
  load.` The deployment is `b65c81f5` + the three named cherry-picks =
  `2db9f085…`, whose tree is byte-identical to a tree proved at 9/9. This is a
  reversible deviation: `refs/heads/w13o-import` carries `d299f212`'s objects,
  so the deployment can be reset forward in one command once `d296b602` is
  fixed.
* **`d296b602` remains unfixed and undiagnosed.** This wave routed around it, as
  W13L did. Which of the eight disjuncts in the re-validation trips is still
  unknown. `NOTES-W13M` §6.5's 4-minute local reproducer is the place to start.
* **The ceiling value 2.0 is a decision, not a derivation.** `NOTES-W13N` §5
  says plainly that nothing argues for 1.5, 2 or 10, and that the choice should
  be made against measured ratios. Against `NOTES-W13M` §1.2's measured
  distribution — 81 of 82 rewrites upward, median 33.1x, 64 of 82 above 10x,
  minimum 2.45x at layer 0 — a ceiling of 2.0 sits **below every single
  upward rewrite the sealed run performed.** If those ratios recur, this arm
  refuses essentially all 81. That is the intended experiment (does refusing
  them help or hurt the composed terminal metric), but it should be read as a
  near-maximal intervention, not a light touch.
* **Fix 1 and Fix 3 have never run at production scale before this launch**
  (`NOTES-W13N` §5). The r13 body reclosure runs 15 boundaries and 28 physical
  writers; the toys ran 2–3. The first production exercise of both is this run.
* **The Fix-2 depth-telemetry call site still has no end-to-end coverage**
  (`NOTES-W13N` §5), and `_verify_carrier_route_smoke_phase_coverage` still has
  none either (`NOTES-W13M` §8). `--dense-closed-loop-depth-telemetry` is on, so
  this run is also the first end-to-end exercise of the Fix-2 read.
* **A ceiling refusal is not proof of a better child.** Refusing keeps the
  incumbent writer, which the boundary metric may have wanted to replace.
* **Unmeasurable ratios are accepted** (`NOTES-W13N` §3.1). Not expected to
  arise in production, where the incumbent is the real LFM2 seed weight.
* **The merged balanced basis was inherited, not built** (§5).
* **Nothing was weakened.** No flag relaxed, no fail-closed clause bypassed, no
  phase key pinned, no peer source edited, no cherry-pick conflict resolved
  ad hoc — there were none to resolve.
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
* **Not committed.** No `git add`, no commit, no push. The local working tree
  was not modified except for this file; the cherry-pick dry run happened in an
  isolated `--shared` clone under the scratchpad.

---

## 9. Artifacts

Local:

| path | |
|---|---|
| `NOTES-W13O.md` | this file |

Server, under `/home/cereal/nf-w13o/`:

| file | |
|---|---|
| `preservation.txt` | §1's before/after proof of the sealed-run move |
| `nf-w13o.bundle` | the deployment bundle, `23b88c29…`, 179,521 B |
| `HEAD-before.txt`, `status-before.txt` | pre-cherry-pick deployment state (`b65c81f5…`, clean / 0 bytes) |
| `deploy.txt` | the fetch + cherry-pick transcript, ending in HEAD/TREE |
| `deploy-verify.txt` | §2.4's deployed digest table and semantic greps |
| `flagcheck.txt` | §3's 184-option `--help`, 109/109 reconciliation, must-be-absent list |
| `runner-help.txt`, `help-flags.txt`, `used-flags.txt` | the reconciliation inputs |
| `nf-launch-r12-round-refold.sh.bak-w13o-20260807T130654Z` | *(in `…/deployments/`)* the pre-edit launcher, `e496c102…` |
| `killspy-rearm.txt` | re-arm timestamp, log offset, window |
| `launch-start.txt`, `launch-refold.out` | the launch record |
| `detachment-proof.txt` | §4's `/proc` proof |
| `timeline.txt` | §5's stage timeline |
| `w13o_tl.py` | timeline / balanced-basis extractor |
| `ceiling.sh` | the ceiling-evidence reader (§7) |
| `poll.sh`, `tl.py`, `timeline.py`, `waitfor.sh`, `waitlayers.sh`, `entryinfo.py` | W13L's toolkit, repointed to `nf-w13o` |
| `rss-sampler.sh`, `milestones.sh`, `rss-samples.tsv`, `milestones.log` | live samplers, redirected so W13L's evidence survives |

Preserved, never modified:
`…/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-w13m-firstcomposed`
— the sealed first composed r13 run, receipt `c9a8f746…`.
