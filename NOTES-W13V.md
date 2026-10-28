# NOTES-W13V — the ~2x ceiling arm ran, and layer 0 cleared the floor by 0.209

Wave 13V, branch `codex/lrc-blank-compression`, local HEAD `e72bf169`.
Deployment `neural-forge-postpub-e893767a` advanced from SOURCE_COMMIT
`5739c742…` to **`c17b28d07e78de9d6f6d58b22493939adf8d4af0`**, tree
`6bcf9376…`. The arm is **launched and running**, PID **2210929**.

**The number this wave existed to produce:**

```
layer 0, terminal composed held-out relative RMSE, no-branch baseline = 1.0

  arm A   3.469x body   0.96988110   accepted
  arm R   3.469x body   1.00576305   REFUSED  (>= 1.0)
  THIS    1.559x body   0.76064618   ACCEPTED

  delta vs arm A  -0.20923493
  delta vs arm R  -0.24511687
```

**Body contraction is the cliff variable, and the effect is large.** At a 1.559x
body the layer-0 composition sits 0.209 below the 3.469x arm that barely passed
and 0.245 below the one that failed. Four layers have now been accepted, none
rejected.

One number matters as much as the headline and was not anticipated: the
**pre-install** terminal baseline at layer 0. Arm A's was `1.0022686933` — the
4x child was already worse than emitting nothing *before* any body install. This
arm's is `0.7538707245`. The 4x arm was recovering toward a line it had already
crossed; this arm starts below it.

---

## 1. Deployment — the chain extended and proved before it was deployed

### 1.1 The missing set

```
git log --oneline cbfd9ab9..codex/lrc-blank-compression

9326a564  Seal the R-arm: better local solves compose worse than silence   (NOTES-W13S.md, 1 file, +685)
e72bf169  Admit the exact-2x carrier ABI and let refused layers keep their story   (8 files, +1266/-33)
```

Exactly two, exactly the set the task named.

### 1.2 The local dry run reproduced the deployment exactly, then extended it

In an isolated `git clone --shared` under the scratchpad — never the deployment,
never the local working tree:

```
git checkout b65c81f5
git cherry-pick 7e25d781 815e791e d299f212 5023bf7e 38cc3bc9 d5b78f5a 729f0a7a cbfd9ab9
TREE = 48386cb015d2d2f8ec2d6fe317e559c768e54f55   <- the deployment's live tree, exactly
```

That is the control: the clone is provably at the deployment's state before
anything new is applied. Then the two:

```
git cherry-pick 9326a564   -> clean
git cherry-pick e72bf169   -> Auto-merging src/…/hf_lfm2_dynamic_closed_loop_adapter.py   clean
TREE = 6bcf93761c81f69ab4fa48ce3ada0cb56b75a6e0
```

**No conflict on either side.** The adapter auto-merged: `e72bf169`'s +68 lines
landed on the deployment-lineage adapter, which lacks `d296b602`.

### 1.3 The dry-run tree was proved to fold before it was deployed

```
PYTHONPATH=<dryrun>/src;<dryrun>  pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py \
                                         tests/test_lfm2_carrier_child_package_fresh_process.py -q -p no:randomly

17 passed in 369.06s (0:06:09)
```

Import resolution pinned to the dry-run clone and printed
(`neural_forge -> …/cp-dryrun/src/neural_forge/__init__.py`), because the venv
carries an editable `.pth` pointing at the main repo.

**That is 9/9 on the carrier e2e lane** — the three `d296b602` reds that
NOTES-W13U records against the *local* tree do not appear here, because
`d296b602` is not an ancestor of anything in the deployment chain. W13U's known
red set is a property of the local branch, not of what was deployed. Plus 8/8 on
the new `test_lfm2_carrier_child_package_fresh_process.py`, which is where the
allowlist's identity check and the two-profile counted plan live.

### 1.4 The deployment, and the tree-identity proof

Pre-state recorded before anything moved: `HEAD-before.txt` = `5739c742…`,
`TREE-before.txt` = `48386cb0…`, `status-before.txt` **0 bytes**.

```
bundle  cbfd9ab9..codex/lrc-blank-compression
size    83,770 bytes
sha256  70481c81cd2a45cbfaff03a24bd4d662bc0e579980a71608a3b24575adf3c2d7   (identical both sides)
server  /home/cereal/nf-w13v/nf-w13v.bundle
verify  "is okay", contains e72bf169…, requires cbfd9ab9… (present as refs/heads/w13s-import)
fetch   refs/heads/w13v-import = e72bf1699853c9f9c051ab2f986682a4c8cb81c9
```

The deployment repo still has no `user.name`/`user.email`, so the chain was
applied with a per-invocation identity from the start, matching the convention
the existing commits carry:

```
git -c user.name=w13v -c user.email=w13v@local cherry-pick 9326a564   -> clean
git -c user.name=w13v -c user.email=w13v@local cherry-pick e72bf169   -> Auto-merging adapter, clean
```

No global or repo-local git config was written.

```
HEAD  = c17b28d07e78de9d6f6d58b22493939adf8d4af0      <- the arm's SOURCE_COMMIT
TREE  = 6bcf93761c81f69ab4fa48ce3ada0cb56b75a6e0
git status --porcelain   EMPTY
```

**The deployed tree sha is byte-identical to the dry-run tree that passed 17/17.**

Deployed working-tree digests:

| file | sha256 | |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **frozen collector — byte-identical, rule holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | unchanged |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77d041bb9cadac9b578dbc914ae59f37084846d1ea0e10397e37002a3f` | unchanged |
| `hf_lfm2_moving_gauge_residual_reclosure.py` | `c1f4462c17278a6d332273d7e6dcbf0ecde6d2e03e6c9aee660c5a531f14f1b3` | **unchanged from arm A — the ceiling is the same code** |
| `hf_lfm2_structural_residual_attention_fold.py` | `300b792601b4c7ba2291758dd2930f523ea872f03d7f6de89960056d4ff3d60c` | unchanged from W13S |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `f74cd5059748334e60e6dc301e57d509fc8aee7379217d26a080d87e696ed175` | new — refused-layer telemetry |
| `lfm2_carrier_child_package.py` | `3b9ea571f51aa5243b2e6d20d073e1a8f99759b37a4052a8aa2d509f55adc4d1` | new — the ABI allowlist |
| `experiments/run_lfm2_dynamic_catalog.py` | `9fd803ede211cf17bf93883698f81a53fd381bacbfb6d9704b2f3c0f96241fd3` | new — allowlist guards + receipt field |

Every one matches the dry-run tree. All eight W13S semantic checks reproduce
exactly on the deployed tree, plus the new ones:

```
Selected.parent.component        adapter  -> 0    (d296b602 still absent)
_lifted_boundary_acceptance_ledger        -> 2    (W13N Fix 1)
writer_reclosure_norm_ratio_ceiling recl. -> 17   (W13N Fix 3)
terminal_free_run_tripwire       adapter  -> 26   (38cc3bc9)
relative_ridge_lambda            attn.fold-> 25   (R1)
body_max_prior_delta_ratio       adapter  -> 12   (R2)
prior_delta_clamp                attn.fold-> 6    (R3)
evaluate_published_child_reload  catalog  -> 2    (W13M §6.4)
ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES pkg -> 6    <- NEW
admitted_lfm2_carrier_counted_abi_profile catalog -> 3 <- NEW
_refused_layer_prior_correction_diagnostics adapter -> 2 <- NEW
refused_layer_solver_diagnostics catalog  -> 9         <- NEW
```

**`SOURCE_COMMIT = c17b28d07e78de9d6f6d58b22493939adf8d4af0`.**

---

## 2. The probe — the ABI now admits the arm, 8/8

W13T's harness re-used **verbatim** (`/home/cereal/nf-w13t/refusal_probe.py`), so
the flag set under test is byte-for-byte W13T's and the comparison is apples to
apples. Only the deployed tree changed. Run through the deployment's own
`config_from_args(build_parser().parse_args(argv))`; no model load, no GPU, no
filesystem write (`refusal-probe-run` still does not exist afterward).

| # | probe | W13T | **now** |
|---|---|---|---|
| 1 | **THE ARM** — ratio 2, 2x budget, H816 | REFUSED | **ACCEPTED**, profile `exact-2x-h816` |
| 2 | CONTROL — arm A's numbers, ratio 4, 4x budget, H816 | ACCEPTED | **ACCEPTED**, profile `exact-4x-h816` |
| 3 | MIXTURE — ratio 2 + 4x budget + H816 | REFUSED | REFUSED |
| 4 | MIXTURE — ratio 4 + 2x budget + H816 | REFUSED | REFUSED |
| 5 | MIXTURE — ratio 2 + 2x budget + H1024 | — | REFUSED |
| 6 | MIXTURE — ratio 2 + 2x total + **4x body** | — | REFUSED |
| 7 | UNLISTED — the exact quarter 57,423,296 / 53,555,520 | — | REFUSED |
| 8 | COUPLING — the 2x arm with no carrier routes | REFUSED | REFUSED |

**8/8 as expected.** Probe 1 resolves the run's tuple through the deployed
source itself:

```
compression_ratio            2.0
counted_target_total         114846592
counted_dense_body           110978816
dense_heterogeneous_target   110978816
required_target_hidden_size  816
uses_carrier_routes          True
carrier ABI profile_id       exact-2x-h816
```

Probe 2 is the regression control that matters most: **the 4x lane did not
move.** Probes 3–7 all refuse with the one new message, which names both
admitted tuples:

```
Carrier-routed production must reserve both independent route channels inside the
counted target and give the remainder to the physical body at one admitted carrier
ABI: ratio 4 / counted total 57,420,144 / dense body 53,552,368 / hidden 816;
ratio 2 / counted total 114,846,592 / dense body 110,978,816 / hidden 816.
```

Probe 6 is the sharpest: a 2x total with a 4x body is refused by the *allowlist*,
not merely by the route-subtraction invariant — the match is cross-field, as
W13U claimed. Probe 7 confirms the exact quarter is still unlisted, so W13U's
choice of an allowlist over W13T §3.3's proposed relaxation held.

Transcript: `/home/cereal/nf-w13v/w13v-probe.txt`.

---

## 3. The launcher

`/home/cereal/model-surgery-lab/deployments/nf-launch-r13-ceiling-2x.sh`

```
sha256  0b93832021abecc39e11a28080769ee7ae0df6f1b149d8519fea07708b77f5f6
size    20,335 bytes, 383 lines
bash -n CLEAN   (locally and on the server)
```

**New file — nothing was overwritten, so no backup was needed.** Verified absent
before upload. `nf-launch-r12-round-refold.sh` (`088dbe6b…`) and its W13S backup
(`6098d179…`) are both re-hashed and **untouched**.

### 3.1 The invocation is r12's refold minus four flags, with five values changed

Mechanical diff of the runner invocation, r12 refold (114 lines) against this
one (110):

```
REMOVED
  --on-policy-round-manifest "$ROUND_MANIFEST"
  --on-policy-max-rounds 2
  --dense-closed-loop-relative-ridge-lambda 1e-5
  --dense-closed-loop-body-max-prior-delta-ratio 8.0

CHANGED
  --counted-target-total-parameters       57420144 -> 114846592
  --counted-dense-body-parameters         53552368 -> 110978816
  --dense-heterogeneous-target-parameters 53552368 -> 110978816
  --compression-ratio                            4 -> 2
  --child-name  lfm2-integrated-carrier-exact4x-h816-mgv3-r12-round-0-refold
             -> lfm2-ceiling-2x-h816-mgv3-a0028
```

Nothing else on the invocation moved. That is exactly W13T §4's composition.

Staging collapsed to a single stage that defaults to `run`, so both
`bash nf-launch-r13-ceiling-2x.sh` and `… run` work. Kept: the `SOURCE_COMMIT`
gate and the dirty-checkout refusal (verbatim), the run-directory reuse refusal,
the `nohup`/`setsid` pattern, the `$PROMPTS` digest check, the base-capture
selection probe, the two `$REBOUND` digest checks and the
parent_cache_key/sealed-rows probe. Dropped with the round: the `$R12`
publication checks and child-bridge preflight (this arm reads no published
child), the `ROUND_PREIMAGE_PREFLIGHT` block, and the `FIT_PROMPTS` /
`MANUAL_DIGESTS` digest checks — those two files are inputs to
`run_r12_on_policy_round.py` only and are not passed to this runner at all.

### 3.2 Flag reconciliation against the **deployed** `--help`

With `PYTHONPATH=$D/src:$D` and resolution printed and asserted to the
deployment (the NOTES-W13K §1.1 stale-checkout trap):

```
neural_forge -> …/neural-forge-postpub-e893767a/src/neural_forge/__init__.py
runner       -> …/neural-forge-postpub-e893767a/experiments/run_lfm2_dynamic_catalog.py

distinct options in --help                 187
distinct flags on the invocation           107   (111 on the r12 refold, -4)
unknown flags                              0     ALL USED FLAGS EXIST IN HELP
```

The 2x tuple, each exactly once on the invocation: `--compression-ratio 2`,
`--counted-target-total-parameters 114846592`,
`--counted-dense-body-parameters 110978816`,
`--dense-heterogeneous-target-parameters 110978816`,
`--required-target-hidden-size 816`, `--carrier-expected-hidden-size 816`,
`--carrier-expected-target-rows 4096`.

Must-be-absent, **all 0 on the invocation**: `--on-policy-round-manifest`,
`--on-policy-max-rounds`, `--dense-closed-loop-relative-ridge-lambda`,
`--dense-closed-loop-body-max-prior-delta-ratio`,
`--balanced-basis-existing-phase-key-sha256`, `--balanced-basis-phase-key`,
`--global-topology-phase-key`,
`--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only`,
`--dense-closed-loop-terminal-composition-floor-disabled`,
`--dense-resume-from-capsule`, `--dense-terminal-only-restart-resume`.

(The first check of this list counted 1 for several because the header comment
*names* the removed flags in explaining why they are gone; re-scoped to the
invocation region, all are 0.)

---

## 4. Launch and detachment

```
launch start   2026-08-08T10:51:59.705Z
command        SOURCE_COMMIT=c17b28d0… setsid nohup bash ./nf-launch-r13-ceiling-2x.sh
               > /home/cereal/nf-w13v/launch-ceiling2x.out 2>&1 < /dev/null &  ; disown
preflight      OK  (prompts digest; base capture selection f229e5f3… in BOTH the sealed cache
                    and the W12E recapture; execution 3d077b94… eager on cuda; implementations
                    eb3bddeb… -> 732ca9ce…; both rebound envelopes bind parent_cache_key
                    c29b6737… and carry the sealed pack rows)
run.pid        2210929
run dir        …/catalog-integrated-carrier/a0028-exact2x-h816-mgv3-ceiling
```

Detachment, from `/proc`:

| | |
|---|---|
| `run.pid` | **2210929** |
| `PPID` | **1** |
| `pgrp` == `session` | 2210912 == 2210912 |
| `tty_nr` | **0** (no controlling terminal) |
| `fd/0` | `/dev/null` |
| `fd/1`, `fd/2` | `<run>/run.log` |
| launcher shell | already exited (0 matching processes) |
| `source-commit.txt` | `c17b28d07e78de9d6f6d58b22493939adf8d4af0` |

Confirmed **on the live `/proc/<pid>/cmdline`**, not merely in the launcher file:
the full 2x tuple present; `--dense-closed-loop-relative-ridge-lambda` and
`--dense-closed-loop-body-max-prior-delta-ratio` count **0**; `--on-policy-*`
count **0**.

Host at launch: all 7 GPUs idle (5–24 MiB, 0 %), no `run_lfm2_dynamic_catalog`
process, deployment clean, run directory did not exist.

**killspy.** A fresh 24 h instance was armed at **2026-08-08T10:51:31Z** with its
own log (`/home/cereal/nf-w13v/killspy-w13v.log`), expiring ~2026-08-09T10:51Z.
The W13S instance (armed 08:24:46Z) had ~21.6 h left and was **not** expired; it
was left alone so the evidence streams stay separable. Both remain alive.
`grep target_pid=2210929` in the W13V log: **0 matches**.

Note for whoever reads that log: `killspy.bt`'s timestamps are server-local
(UTC-6) but labelled `Z`. The first line reads `2026-08-08T04:51:31Z` for a
10:51:31Z event.

`rss-sampler.sh` and `milestones.sh` were restarted against the new `run.pid`.

---

## 5. Phases — everything cold, exactly as predicted, but the time landed elsewhere

```
2026-08-08T10:52:04.589Z  run_started            configuration_sha256 66eab8dc34624236fb7370b07232aaf889f045b7105b4e06c5cd814cca15da6e
2026-08-08T10:52:16.479Z  parent_verified                    <- 11.9 s
2026-08-08T10:52:17.565Z  parent_cache_verified              <- 1.1 s later
2026-08-08T10:52:45.946Z  tokenizer_bridge_verified
2026-08-08T10:52:50.323Z  carrier_verified                   <- 46 s in
2026-08-08T11:01:38.656Z  parent_rows_materialized
2026-08-08T11:38:53.128Z  wiring_planned                     <- after a 36m54s COLD topology build
2026-08-08T11:38:54.024Z  system_fold_started
2026-08-08T11:41:18.161Z  balanced_basis_phase_cache_miss
2026-08-08T11:50:14.120Z  balanced_basis_phase_cache_published   <- 8m56s COLD basis build
2026-08-08T11:50:58.343Z  dense_structural_fold_completed
2026-08-08T11:50:59.599Z  dense_closed_loop_started          <- 58m55s in
2026-08-08T11:51:16.827Z  child_closed_loop_layer_started    layer_index=0
```

`configuration_sha256` rotated to `66eab8dc…` (arm A `38d1766b…`, R-arm
`637bfe3f…`), which is the digest-level confirmation that the new budget entered
the run's configuration.

**Phase caches: 0 hits, 6 misses. Completely cold.** Arm A and every 4x wave
since W13D had 6 topology lookups / 6 hits / 0 misses and inherited `a0025`'s
basis payload.

```
global topology   5 lookups, 5 misses, 3 published
    04c3efdb3ce4c876af8ef2679cdd67e3ce1848a27dd485c669e9eee493a36d56 -> e041d3d1…
    ada08b8641a1719984dd3056fa3f1e0ef328fdae630fc5821194c5d94a0a6cf2 -> f17b2dee…
    695382bd92a333cf98e6b9d847939e555eef80235baf5119f1b92d21d26d93e4 -> 5d46f2dd…
    (arm A's three were a5eaf8da… / 9c9d568f… / edea7406…)
balanced basis    1 lookup, 1 miss, 1 published
    a67e4640bef3b74a754f2254aeffd90462bdc9c3557a8fdc27bab5c583f2bc6f
    completion  c85af5750488a8419bf453c4e49359bb91fce78c08c8ccb64502a3c6226df674
    (arm A hit 52cbe3fd… / 975e8286…)
global_topology_frontier_selection_completed   selected_hidden_size = 816
```

**This run built its own balanced basis. It inherited nothing.**

**Base evidence only, confirmed structurally**: `balanced_basis_merged_evidence_bound`
count **0** (arm A had 2), any `on_policy` event count **0**. Arm A's two-member
evidence set `f62110d7…` is not in play.

### 5.1 Where W13T's timing prediction was wrong, and it matters

W13T §6 predicted the *balanced basis* would be the long cold phase ("W13S
measured the 4x cold build at ~35 min … budget for a longer, not shorter,
build"). The direction was right, the location was wrong:

* the **balanced basis** cold build took **8m56s**, four times *faster* than the
  4x figure W13T was extrapolating from;
* the **global topology** build took **36m54s** (11:01:49 first post-parent-rows
  miss → 11:38:43 final publish), including a 256-batch capacity scan
  (11:02 → 11:12:52) and a ~26 min solve. That is the phase every 4x wave since
  W13D skipped entirely on a cache hit, so no prior wave had a number for it.

Total time to `dense_closed_loop_started` was 58m55s versus arm A's 18m51s and
the R-arm's 19m35s. All of the extra 40 minutes is the cold topology.

---

## 6. The result — layer 0 cleared the floor, and so did 1, 2 and 3

```
2026-08-08T12:01:04.681Z  child_closed_loop_layer_validation_completed  layer_index=0 validation_accepted=True
2026-08-08T12:01:32.154Z  child_closed_loop_restart_capsule_published   layer_index=0
2026-08-08T12:10:22.098Z  layer_index=1 validation_accepted=True
2026-08-08T12:22:26.392Z  layer_index=2 validation_accepted=True
2026-08-08T12:31:12.971Z  layer_index=3 validation_accepted=True
```

Layer 0 took 10m15s from `child_closed_loop_layer_started` to
`child_closed_loop_layer_completed`, with four `child_replay_sharding_completed`
events — the same shape the R-arm had before it was refused at the same point.

### 6.1 The floor numbers, layer by layer, against arm A

Both columns read the **same field** (`validation_decision.metrics
.terminal_relative_rmse`) out of each run's own restart-capsule
`accepted_layer_completion_ledger`, by the same script. Verified apples to
apples: read this way, arm A's layer 0 comes out `0.9698811016092053`, which is
exactly the number NOTES-W13S §6.2 quotes from
`reports/stage1-attribution-20260808/per_layer_profile.json`.

| layer | **this arm** (1.559x body) | arm A (3.469x body) | delta |
|---|---|---|---|
| pre-install baseline | **0.7538707245** | 1.0022686933 | −0.2484 |
| 0 | **0.7606461756** | 0.9698811016 | **−0.2092** |
| 1 | **0.7729266870** | 0.9323935011 | −0.1595 |
| 2 | **0.7332837840** | 0.9130796440 | −0.1798 |
| 3 | **0.7469952127** | 0.8894681547 | −0.1425 |

Terminal mean cosine similarity, same layers: this arm `0.7340, 0.7288, 0.7427,
0.7435`; arm A `0.3850, 0.4572, 0.4849, 0.5191`.

Layer 0's full decision, verbatim:

```
accepted                                     True
terminal_relative_rmse                       0.760646175567523
terminal_baseline_relative_rmse              0.7538707244950512
terminal_relative_rmse_regression            0.006775451072471794
terminal_no_branch_relative_rmse_baseline    1.0
terminal_absolute_floor_engaged              True
terminal_absolute_floor_minimum_layer_index  0
terminal_absolute_floor_passed               True
terminal_composition_floor_engaged           True
terminal_composition_floor_failed            False
terminal_non_regression_engaged              True
terminal_non_regression_passed               True
terminal_non_regression_tolerance            0.25
terminal_mean_cosine_similarity              0.7339714472504058
terminal_layer_index                         13
terminal_rows                                13464
terminal_free_run_tripwire_enabled           True
terminal_free_run_tripwire_evaluated         False
baseline_relative_rmse                       0.3400043084023315
candidate_relative_rmse                      0.5033450119749837
relative_rmse_improvement                    -0.16334070357265223
counterfactual_performance_threshold_passed  False
performance_metrics_are_decision_authority   False
performance_metrics_diagnostic_only          True
```

### 6.2 Three things in that block worth reading carefully

**(a) The pre-install baseline is the deeper result.** Arm A's layer-0
`terminal_baseline_relative_rmse` is `1.0022686933` — *above* the no-branch line
of 1.0 before a single body install. The 4x child's 14-layer composition was
already worse than emitting zeros, and arm A's whole install sequence was
clawing back toward a line it had started on the wrong side of. This arm starts
at `0.7538707245`. The ~2x rung is not "the same experiment with more headroom";
it starts in a qualitatively different place.

**(b) The per-boundary and composed clauses disagree in sign here too — but the
other way round from the R-arm.** This arm *fails* the counterfactual
per-boundary clause at layer 0 (`relative_rmse_improvement = -0.1633`,
`counterfactual_performance_threshold_passed = False`) and clears the composed
floor by 0.24. The R-arm did the reverse: passed the boundary clause and failed
the floor. Arm A also fails the boundary clause at layer 0 (`-0.0928`). In all
three arms this clause is diagnostic-only
(`performance_metrics_are_decision_authority = False`), from
`--dense-closed-loop-performance-metrics-diagnostic-only`, which arm A also
carried — so it gates nothing. It is recorded because "local fit and composed
result disagree in sign" is the A12/W12B signature and it is present here at a
body ratio where the composition is nonetheless healthy.

**(c) The per-layer trajectory is flatter and noisier, not monotonically
better.** Arm A's `terminal_relative_rmse_regression` is negative at all four
layers (−0.0324, −0.0375, −0.0193, −0.0236): every install improved the
composition. This arm's is `+0.0068, +0.0123, −0.0396, +0.0137` — it oscillates
around a much lower plateau, and three of the first four installs made the
composition very slightly *worse* while staying far inside the 0.25
non-regression tolerance. A claim that this arm improves monotonically would be
false. What it does is sit ~0.15–0.21 lower throughout.

### 6.3 The body-solve telemetry arrives for free

This arm passes neither R flag, but the deployed tree carries R3's reporting, so
the clamp block is serialized anyway against the default allowed ratio of 1.0 —
the numbers the R-arm was built to produce and could not show. Layer 0:

```
layer role         requested  allowed  clamped  upd_scale  ridge_mode  cond
    0 mlp_gate         12.37        1     True    0.08087    absolute  2.201e+07
    0 mlp_up           8.044        1     True     0.1243    absolute  2.201e+07
    0 mlp_down         3.577        1     True     0.2795    absolute  4.099e+04
solves 3   clamp engaged 3   released 0
```

Against arm A's reconstructed layer-wide range of 1.446x–140.4x (median ~37x,
NOTES-W13S §6.5), layer 0 at 2x requests **12.37x / 8.04x / 3.58x**. A
less-contracted body asks for a much smaller correction. This is one layer of
one arm, not the 42-solve distribution arm A's table covers.

`ridge_mode` is `absolute` on all three, correctly — no
`--dense-closed-loop-relative-ridge-lambda` was passed.

---

## 7. Open operational risk — this run will run out of disk before it finishes

**This is the one thing that needs a decision, and I did not act on it.**

```
/dev/sda2   1.8T   1.7T used   59G available   97%
```

Arm A's completed run is **95G**, of which **94G is `cache/`**, in two trees:

```
cache/child-final    47G   written 2026-08-07T22:02 - 22:35   (~8h47m after run start)
cache/child-reload   47G   written 2026-08-07T23:33 - 23:39   (the W13M §6.4 published-child reload pass)
```

Neither exists during the closed loop. This run's `cache/` is currently **4.0K**;
its whole directory is 1.8G after four layers (990M snapshots, 848M capsules,
and `--dense-restart-capsule-keep 2` bounds the capsules).

Arm A started 2026-08-07T13:15:02Z and wrote `child-final` at 22:02 — 8h47m in.
This run started 10:52:04Z, so the equivalent point is roughly **19:40Z today**.
At that point it needs on the order of 94G against 59G free — and the 2x child
is *twice* the parameter count of the 4x child, so its decode-trace cache should
be **larger** than 94G, not smaller.

**Expected outcome if nothing changes: ENOSPC during the final child evaluation,
after ~9 hours of good work, with all fourteen layers already folded.**

Reclaimable, all read-only-surveyed and **not touched**:

| path | size |
|---|---|
| `…-refold.sealed-w13m-firstcomposed/cache` | 94G |
| `…-refold.sealed-armA-complete/cache` | 94G |
| `…/a0023-exact4x-h816-mgv3-r12/cache` | 86G |

Those are the only three large consumers in the whole
`catalog-integrated-carrier` tree; every other entry is ≤1.8G. There is also a
CIFS mount `//192.168.50.250/ai` on `/mnt/ai_data` with **2.0T free**, so the
caches could be moved rather than deleted.

I did not delete, move or archive anything: two of those three are **sealed**
directories, the task's standing instruction is that sealed dirs are preserved,
and deleting 94G of a sealed run's evidence is exactly the class of change that
needs an explicit decision rather than an agent's judgement. Flagged, quantified,
and left.

---

## 8. Monitoring recipe

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
W=/home/cereal/nf-w13v
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0028-exact2x-h816-mgv3-ceiling

$SSH --command-timeout 900   --execute "bash $W/poll.sh"                  # one-shot full poll
$SSH --execute "python3 $W/w13v_tl.py timeline"                           # stage timeline
$SSH --execute "python3 $W/phases.py"                                     # phase-cache keys, hits/misses   <- NEW
$SSH --execute "python3 $W/l0.py"                                         # every accepted layer's floor metrics + the ceiling-map comparison   <- NEW
$SSH --execute "python3 $W/floor.py"                                      # hunts the floor number in log / receipt / capsules   <- NEW
$SSH --execute "python3 $W/r12r3.py $RUN"                                 # per-solve prior/ridge table
$SSH --execute "bash $W/ceiling.sh"                                       # ceiling evidence (post-body-reclosure only)
$SSH --command-timeout 15000 --execute "bash $W/waitlayers.sh 8 14000"    # block until N layers accepted
$SSH --execute "grep target_pid=2210929 $W/killspy-w13v.log"              # did anything signal it
$SSH --execute "df -h /home; du -sh $RUN $RUN/cache"                      # THE DISK WATCH (section 7)
```

Two operational notes learned this wave:

* **`ssh_client.py` serializes on a local lease file.** A long `sleep` inside a
  `--script` holds it and every other invocation fails with `[SSH BUSY] Timed out
  waiting for local SSH lease`. Put the waiting on the *local* side (or in a
  detached server-side process), not inside a held SSH session.
* **Single quotes inside a single-quoted `--script` argument break silently** and
  can produce a shell error that looks like a result. Any grep pattern containing
  a space or a quote belongs in an uploaded script, not inline.

The floor/boundary numbers still do not stream; the first observable per-layer
numbers arrive with each accepted layer's restart capsule, ~10 min apart.

---

## 9. Caveats

* **The run is not finished.** As of 2026-08-08T12:33Z it is at layer 4 of 14,
  4 accepted / 0 rejected, PID 2210929 alive, 1h41m elapsed. There is no child,
  no publication, no routed manual dossier and **no manual-review verdict** —
  which is R13_PLAN item 4's actual deliverable. Everything in §6 is layer-level
  numeric evidence from an in-flight run.
* **§7 says this run is likely to fail on disk** before it produces that
  verdict. That is the most probable way this ends without intervention.
* **Nothing here is an intelligence verdict**, per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`. `terminal_relative_rmse`
  is a safety bound; the metrics block says so itself
  (`terminal_floor_is_safety_bound_not_intelligence_verdict = True`).
* **One arm at one rung.** The 0.209 gap is one budget change against one prior
  arm on layer 0. Body ratio (1.559x vs 3.469x) is the intended single variable
  and the flag sets are otherwise identical — but the evidence set also changed
  from arm A's 2-member `f62110d7…` to this arm's 1-member base-only set, because
  dropping the round is what makes a 2x child's evidence on-policy. **That is a
  second variable, and it is unavoidable, not an oversight.** A 4x arm on base
  evidence only would separate them; it has not been run.
* **The topology cold-build cost (36m54s) is a first measurement**, not a
  reproduction. No 4x wave since W13D has paid it.
* **`_route_smoke`'s 2x branch has still never compared a real 2x child**, per
  NOTES-W13U §7. It will, in this run, after the closed loop.
* **The refused-layer telemetry fix is deployed but unexercised.** No layer has
  been refused, so `refused_layer_solver_diagnostics` has not been produced by a
  live run. If a later layer refuses, harvest it from `receipt.json`'s `failure`
  block — `floor.py` prints it.
* **`d296b602` remains unfixed and undiagnosed.** This wave routed around it the
  same way W13O and W13S did; it is simply not in the chain.
  `refs/heads/w13v-import` carries the objects.
* **Nothing was weakened.** No flag relaxed, no fail-closed clause bypassed, no
  phase key pinned, no peer source edited, no cherry-pick conflict resolved ad
  hoc — there were none to resolve. `--dense-moving-gauge-writer-norm-ratio-ceiling`
  is still arm A's defanged 2.0 and has not been exercised: the body reclosure it
  gates runs only after all fourteen installs.
* **Preservation verified before and after.** `…sealed-armA-complete/receipt.json`
  `45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10` and
  `…failed-closed-terminal-floor-L0-w13s/receipt.json` `5d61ae20…` are unchanged.
  The directory went from 24 entries to 25 — the one new entry is this run's.
* **Not committed.** No `git add`, no commit, no push. The only local file
  written is this one.

---

## 10. Artifacts

Local:

| path | |
|---|---|
| `NOTES-W13V.md` | this file |

Server, under `/home/cereal/nf-w13v/`:

| file | |
|---|---|
| `nf-w13v.bundle` | `cbfd9ab9..codex/lrc-blank-compression`, 83,770 B, `70481c81…` |
| `HEAD-before.txt` / `TREE-before.txt` / `status-before.txt` | pre-deployment state, `5739c742…` / `48386cb0…` / 0 bytes |
| `w13v_probe.py`, `w13v-probe.txt` | §2's eight config-time probes |
| `flagcheck.sh`, `flagcheck.txt`, `deployed-help.txt`, `help-options.txt`, `launcher-flags.txt`, `unknown-flags.txt` | §3.2's reconciliation, `unknown-flags.txt` is 0 bytes |
| `detachment-proof.txt` | §4's `/proc` evidence |
| `launch-ceiling2x.out`, `launch-start.txt` | launcher stdout and launch timestamp |
| `killspy-w13v.log` | this wave's killspy instance |
| `phases.py`, `l0.py`, `floor.py` | new extractors (phase caches; per-layer floor metrics; floor-number hunt) |
| `poll.sh`, `w13v_tl.py`, `r12r3.py`, `waitlayers.sh`, `ceiling.sh`, `rss-sampler.sh`, `milestones.sh`, `entryinfo.py` | W13S toolkit, repointed to the a0028 run |
| `l0-floor.txt` | the layer-0 comparison table as printed |

Deployment: `/home/cereal/model-surgery-lab/deployments/nf-launch-r13-ceiling-2x.sh`
(`0b938320…`), new file, nothing overwritten.

Untouched and re-verified: `…sealed-armA-complete` (`45bc68c2…`),
`…sealed-w13m-firstcomposed`, `…failed-closed-terminal-floor-L0-w13s`
(`5d61ae20…`), every other peer run, `nf-launch-r12-round-refold.sh`
(`088dbe6b…`) and its W13S backup (`6098d179…`), and the W13T probe directory
`/home/cereal/nf-w13t/` (read only).
