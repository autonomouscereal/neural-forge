# NOTES-W13S — r13 ceiling ARM R: the remedies deployed, launched, and refused at layer 0

Wave 13S, branch `codex/lrc-blank-compression`. Launches `NOTES-W13R` §4's
recommended arm — the last never-tried fold-only lever — on top of arm A's exact
composition. One variable moved: the body MLP solves got a real ridge (R1) and
room to correct (R2), plus R3's telemetry.

**The arm failed closed at layer 0, 12 minutes into the closed loop, and the
failure is the result.** The per-boundary clause passed — the local solve did
improve — and the terminal composition floor refused the install because the
composed held-out relative RMSE came out at **1.00576305**, above the no-branch
baseline of 1.0, where arm A's same layer sat at **0.96988110**. A better-
conditioned, eight-times-larger step made the composition worse than emitting
nothing. That is the A12/W12B signature at production scale, and it is the one
thing `NOTES-W13R` §3 said its toy could not establish.

Five things happened, in this order:

1. **Arm A (SEALED COMPLETE, frozen-reviewed 0/15) was moved aside intact** (§1).
2. **The deployment cherry-pick chain was extended by five commits** (§2), the
   resulting tree proved to fold at 9/9 *before* it was deployed.
3. **The launcher gained exactly two lines** (§3).
4. **The arm launched detached and reproduced every upstream phase** (§4–§5).
5. **It failed closed on layer 0** (§6). Nothing was relaunched, no flag was
   relaxed, no gate was bypassed.

---

## 1. Preservation — arm A is intact under a new name

The refold path held **arm A, sealed and complete**: receipt file
`45bc68c2…` (the digest the arm-A frozen review and
`reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md` §6 both recorded),
inner `receipt_sha256` `1cdec0a6…`, published child, certificate, capsules. It
was moved, never deleted, with `mv -n`:

```
from  …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
to    …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-armA-complete
mv -n -v, exit 0, at 2026-08-08T08:14:29Z   (rename within one filesystem)
```

| quantity | before | after |
|---|---|---|
| `receipt.json` bytes | **348,544** | **348,544** |
| `receipt.json` sha256 | `45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10` | **same** |
| `receipt_sha256` field | `1cdec0a61cd94712bc3b26fffb26dce970291acd5b21817f970a030ad3f38168` | **same** |
| `diagnostics.json` bytes / sha256 | 324,305 / `cc8d1e64…` | **same** |
| `ledger.jsonl` bytes / sha256 | 8,252 / `b7d916ae…` | **same** |
| `run.log` bytes / sha256 | 1,674,944 / `042d6b16…` | **same** |
| `routed-manual-dossier.json` bytes / sha256 | 2,974,270 / `409de3ed…` | **same** |
| `du -sb` | **101,029,613,827** | **101,029,613,827** |
| `find -type f \| wc -l` | **1,542** | **1,542** |
| `published-child/certificate.json` | 7,927 B | 7,927 B |
| `source-commit.txt` | `2db9f085…` | `2db9f085…` |

`published-child/` still carries `certificate.json`, `child/` and
`provenance/`; `restart-capsules/` still carries `capsule-000012-7c1a86bc…`
and `capsule-000013-974918bf…`.

Nothing was deleted. All six previously preserved failure/kill partials, the
`sealed-w13m-firstcomposed` run, and all four peer directories (`a0024`–`a0027`)
remain untouched. **Re-verified after everything else in this wave finished**:
still 348,544 B / `45bc68c2…`, still 101,029,613,827 B / 1,542 files, and
`find -newermt` since the move returns 0 entries.

Proof: `/home/cereal/nf-w13s/preservation.txt`.

### 1.1 The failed R-arm is preserved too

The R-arm's own failed-closed directory was moved aside by the same rule, so a
relaunch cannot clobber it:

```
to    …/a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-closed-terminal-floor-L0-w13s
mv -n -v, exit 0, at 2026-08-08T08:58:08Z
receipt.json  284,653 B  sha256 5d61ae20913810f06324780e966d11522758352552d461312c549571ef40a1e7
              receipt_sha256 field 42bc21c308691b831ed6f294f1fa75f2a206629e09c5123d4238aca800164873
              status failed_closed
diagnostics.json 320,413 B  3fd70338…      run.log 716,841 B  36958c8e…
du -sb 1,323,211    files 10
```

Byte-identical across the move. This is a judgement call the task did not ask
for: it is protective, not destructive — the canonical run path is where the
launcher writes, and leaving a sealed fail-closed receipt there to be
overwritten would have destroyed the only record of this result.

---

## 2. Deployment — the chain extended, and proved before it was deployed

### 2.1 The exact missing set

```
git log --oneline d299f212..cbfd9ab9

cbfd9ab9  Give the body solves a real ridge and room to correct        <- R1+R2+R3
729f0a7a  Attribute the collapse: an inert ridge feeding a starving clamp
d5b78f5a  Arm A frozen review: 0/15 -- the reclosure was necessary, not sufficient
38cc3bc9  Port the A12 rule over the reclosure and wire the tripwire's eyes
5023bf7e  Seal the overlap audit: the reclosure re-derived A12, twice
```

Exactly five, exactly the set the task named. `7e25d781` is already in the
deployment (it is one of arm A's three cherry-picks), as verified before
bundling. **All five were taken — no doc-only commit was skipped**, because
every one applied clean and keeping the chain contiguous is what makes the tree
comparison in §2.3 possible.

### 2.2 The local dry run reproduced the deployment exactly, then extended it

In an isolated `git clone --shared` under the scratchpad — never the deployment,
never the local working tree:

```
git checkout b65c81f5
git cherry-pick 7e25d781 815e791e d299f212     -> clean
TREE = 14dc0ebf266fb1bb1a95fa10326df98301a2c69c   <- the deployment's live tree, exactly
```

That is the control: the local clone is provably at the deployment's state
before anything new is applied. Then the five:

```
git cherry-pick 5023bf7e   -> Auto-merging docs/CURRENT_FOLDING_GOAL.md            clean
git cherry-pick 38cc3bc9   -> Auto-merging src/…/hf_lfm2_dynamic_closed_loop_adapter.py  clean
git cherry-pick d5b78f5a   -> clean
git cherry-pick 729f0a7a   -> clean   (18 files, 150,200 insertions)
git cherry-pick cbfd9ab9   -> Auto-merging src/…/hf_lfm2_dynamic_closed_loop_adapter.py  clean
TREE = 48386cb015d2d2f8ec2d6fe317e559c768e54f55
```

**No conflict at any step**, on either side. The adapter is touched by both
`38cc3bc9` (+208) and `cbfd9ab9` (+156) and auto-merged both times.

### 2.3 The dry-run tree was proved to fold before it was deployed

```
PYTHONPATH=<dryrun>/src pytest tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py -q -p no:randomly

9 passed in 355.83s (0:05:55)
```

Import resolution was pinned to the dry-run clone and printed
(`neural_forge.__file__` under `cp-dryrun/src`), because the venv carries an
editable `.pth` pointing at the main repo.

**The three known-red tests did not appear**, which was the specific thing to
check: `d296b602` is not an ancestor of anything in this chain, so
`test_carrier_dense_core_alternates_fold_round_and_refold`,
`test_bridged_carrier_lane_alternates_fold_round_and_refold` and
`test_carrier_production_margin_pin_composes_through_a_round` all pass. The
W13O routing-around still holds and was not re-litigated.

### 2.4 The deployment, and the tree-identity proof

Pre-state recorded before anything moved: `HEAD-before.txt` = `2db9f085…`,
`TREE-before.txt` = `14dc0ebf…`, `status-before.txt` = **0 bytes**.

```
bundle  d299f212..codex/lrc-blank-compression
size    540,603 bytes
sha256  009aa71244dda3c443585fbe49c21b8e0a0239ce37f1cda3a53bd80a32008172   (identical both sides)
server  /home/cereal/nf-w13s/nf-w13s.bundle
verify  "is okay", contains cbfd9ab9…, requires d299f212… (present as refs/heads/w13o-import)
fetch   refs/heads/w13s-import = cbfd9ab9644cbcfa537a20fb75a309b21d0170d7
```

The first cherry-pick attempt **stopped on its own**, and correctly: the
deployment repo has no `user.name`/`user.email` in local or global config, so
`git cherry-pick` refused with `Committer identity unknown` after a clean
auto-merge. The staged state was backed out with `git cherry-pick --abort`
(HEAD and tree verified back at `2db9f085…` / `14dc0ebf…`, status empty) and
the chain redone with a per-invocation identity, matching the convention the
arm-A commits already carry (`C:w13o <w13o@local>`):

```
git -c user.name=w13s -c user.email=w13s@local cherry-pick …
```

No global or repo-local git config was written.

```
HEAD  = 5739c7425a4ca90eda6ca2af35e7533bfec58fac      <- the arm's SOURCE_COMMIT
TREE  = 48386cb015d2d2f8ec2d6fe317e559c768e54f55
git status --porcelain   EMPTY
```

**The deployed tree sha is byte-identical to the dry-run tree that passed 9/9.**
Commit shas differ (cherry-pick rewrites committer and timestamp); the tree does
not.

Deployed working-tree digests:

| file | sha256 | |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **frozen collector — byte-identical, rule holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | unchanged |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77d041bb9cadac9b578dbc914ae59f37084846d1ea0e10397e37002a3f` | unchanged → basis key must not rotate |
| `hf_lfm2_moving_gauge_residual_reclosure.py` | `c1f4462c17278a6d332273d7e6dcbf0ecde6d2e03e6c9aee660c5a531f14f1b3` | **unchanged from arm A** — the ceiling is the same code |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `e1e7879171d78628e25cd23efe0900e3cf4472de6979f6aa13faf0c93fce2d50` | new — A12 port + tripwire + R1/R2/R3 wiring |
| `hf_lfm2_structural_residual_attention_fold.py` | `300b792601b4c7ba2291758dd2930f523ea872f03d7f6de89960056d4ff3d60c` | new — R1/R2/R3 solver |
| `experiments/run_lfm2_dynamic_catalog.py` | `abad68f57c7bf62fd5306b8f97b14f7f1a1c1c675b0dbf24f6ba050fe4dc3cf1` | new — the two CLI flags |

Every one matches the dry-run tree. Semantic checks on the deployed tree:

```
grep -c "Selected parent component"        adapter  -> 0    (d296b602 absent)
grep -c "_lifted_boundary_acceptance_ledger"        -> 2    (W13N Fix 1 present)
grep -c "writer_reclosure_norm_ratio_ceiling" recl. -> 17   (W13N Fix 3 present)
grep -c "terminal_free_run_tripwire"       adapter  -> 26   (38cc3bc9 present)
grep -c "relative_ridge_lambda"            fold     -> 25   (R1 present)
grep -c "body_max_prior_delta_ratio"       adapter  -> 12   (R2 present)
grep -c "prior_delta_clamp"                fold     -> 6    (R3 present)
grep -c "evaluate_published_child_reload"  catalog  -> 2    (W13M §6.4 present)
```

**`SOURCE_COMMIT = 5739c7425a4ca90eda6ca2af35e7533bfec58fac`.**

---

## 3. Launcher — two lines added

Timestamped backup taken first, `cp -p`:

```
/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh.bak-w13s-20260808T082203Z
sha256 6098d17904e95b6f677d4ae65681caee8277f92df4636fdca333618baf616873   <- arm A's exact post-edit form
```

The complete diff against that backup:

```diff
446a447,448
>     --dense-closed-loop-relative-ridge-lambda 1e-5 \
>     --dense-closed-loop-body-max-prior-delta-ratio 8.0 \
```

In context, in the solver-configuration cluster:

```
    --dense-closed-loop-terminal-free-run-tripwire \
    --dense-closed-loop-input-error-multiplier 0 \
    --dense-closed-loop-relative-ridge-lambda 1e-5 \          <- added
    --dense-closed-loop-body-max-prior-delta-ratio 8.0 \      <- added
    --dense-closed-loop-solver-conditioning-diagnostics \
    --dense-closed-loop-gate-operating-point-diagnostics \
    --dense-closed-loop-depth-telemetry \
```

`NOTES-W13R` §4 recommends four flags; two of them
(`--dense-closed-loop-solver-conditioning-diagnostics`,
`--dense-closed-loop-depth-telemetry`) were already on arm A's line, so only two
were added. **One variable moved.**

```
bash -n     CLEAN
before      25,179 bytes, 491 lines, 6098d179…
after       25,289 bytes, 493 lines, 088dbe6b8521b0edc8a5d32319bcd7eb4b3a564c7948e7e0839e877b14f6a2c0
```

The first patch attempt **refused and left the file untouched** (anchor-
uniqueness assertion; the backslash was mangled in SSH transit — sha and diff
both confirmed unchanged). It was redone from an uploaded script.

Everything else re-audited rather than assumed — the ceiling stays:

```
--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement : 1
--dense-moving-gauge-boundary-regression-tolerance 1e-6                       : 1
--dense-moving-gauge-writer-norm-ratio-ceiling 2.0                            : 1   <- arm A's ceiling, unchanged
--dense-moving-gauge-install-verification fingerprint                         : 1
--dense-closed-loop-terminal-composition-floor                                : 1
--dense-closed-loop-terminal-regression-tolerance 0.25                        : 1
--dense-closed-loop-terminal-absolute-floor-minimum-layer 0                   : 1
--dense-closed-loop-terminal-free-run-tripwire                                : 1
--dense-closed-loop-input-error-multiplier 0                                  : 1
--dense-streaming-vocabulary-margin-v2                                        : 1
--dense-body-reclosure-mode moving_gauge_v3                                   : 1
--max-selection-rows 13728 / --max-fit-rows 40656 / --max-validation-rows 13199: 1 each
--dense-replay-devices cuda:0,cuda:4                                          : 1
```

Must-be-absent list, all **0**: `--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only`,
`--balanced-basis-phase-key`, `--global-topology-phase-key`,
`--dense-closed-loop-terminal-composition-floor-disabled`,
`--dense-resume-from-capsule`, `--dense-terminal-only-restart-resume`.

Flag reconciliation against the **deployed** runner's `--help`, with
`PYTHONPATH=$D/src:$D` and resolution printed and verified to the deployment
(the `NOTES-W13K` §1.1 stale-checkout trap):

```
distinct options in --help                 187
distinct flags on the refold invocation    111    (109 in arm A, +2)
unknown flags                              0      ALL USED FLAGS EXIST IN HELP
--dense-closed-loop-relative-ridge-lambda        help lines 87, 496
--dense-closed-loop-body-max-prior-delta-ratio   help lines 88, 506
```

Both new flags' help text is the W13R text, including the coupling refusal
("A value above 1.0 requires `--dense-closed-loop-relative-ridge-lambda`").

---

## 4. Launch and detachment

```
launch start   2026-08-08T08:23:42.764Z
command        SOURCE_COMMIT=5739c742… setsid nohup bash ./nf-launch-r12-round-refold.sh refold
               > /home/cereal/nf-w13s/launch-refold.out 2>&1 < /dev/null &  ; disown
preflight      OK  (child bridge matches sealed bridge; both base capture selections verified;
                    round 0 carries a target-token preimage for 115 batches; both rebound
                    envelopes bind parent_cache_key c29b6737… and carry the sealed pack rows)
run.pid        1806785
run dir        …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
```

Both new flags confirmed **on the live process command line**, not merely in the
launcher file:

```
--dense-moving-gauge-writer-norm-ratio-ceiling  2.0
--dense-closed-loop-input-error-multiplier      0
--dense-closed-loop-relative-ridge-lambda       1e-5
--dense-closed-loop-body-max-prior-delta-ratio  8.0
```

Detachment, from `/proc` (`detachment-proof.txt`):

| | |
|---|---|
| `run.pid` | **1806785** |
| `PPID` (from `/proc/1806785/stat`) | **1** |
| pid 1 cmdline | `/usr/lib/systemd/systemd --system --deserialize=158 splash` |
| `pgrp` == `session` | 1806723 == 1806723 |
| `tty_nr` | **0** (no controlling terminal) |
| `fd/0` | `/dev/null` |
| `fd/1`, `fd/2` | `<run>/run.log` |
| launcher shell | already exited |
| `source-commit.txt` | `5739c7425a4ca90eda6ca2af35e7533bfec58fac` |

Host at launch: all 7 GPUs idle (5–24 MiB used, 0 % util — ComfyUI tenants
stopped as reported), 231 GiB available of 251 GiB.

**killspy.** The W13L/W13O instance (pid 3234058, started 2026-08-07 09:47:11
server-local, 24 h window) was **not expired** — 16 h 37 m elapsed, ~7.4 h left,
expiring ~2026-08-08T15:47Z. A run at arm A's pace would have outlived it, so a
**fresh 24 h instance** was armed at 2026-08-08T08:24:53Z with its own log
(`/home/cereal/nf-w13s/killspy-w13s.log`) so the two evidence streams stay
separable and the pre-existing log is untouched. Both were alive throughout.

---

## 5. Early phases — everything upstream reproduces

```
2026-08-08T08:23:55.383Z  run_started            configuration_sha256 637bfe3f29c70250929d1310c5409390e72f923261fd3487c935a2d60e7b9f55
2026-08-08T08:24:03.317Z  parent_verified        <- clause A cleared in 8.2 s
2026-08-08T08:30:11.126Z  parent_cache_verified
2026-08-08T08:30:39.158Z  tokenizer_bridge_verified
2026-08-08T08:30:44.723Z  carrier_verified       <- 6m49s in
2026-08-08T08:30:45.241Z  global_frontier_preplanned
2026-08-08T08:36:20.620Z  parent_rows_materialized
2026-08-08T08:37:15.381Z  on_policy_extension_evidence_materialized
2026-08-08T08:37:51.074Z  wiring_planned
2026-08-08T08:37:52.294Z  system_fold_started
2026-08-08T08:41:47.581Z  balanced_basis_merged_evidence_bound
2026-08-08T08:42:12.586Z  balanced_basis_merged_evidence_bound
2026-08-08T08:42:15.793Z  balanced_basis_phase_cache_hit
2026-08-08T08:43:29.597Z  dense_structural_fold_completed
2026-08-08T08:43:30.220Z  dense_closed_loop_started
```

**`configuration_sha256` rotated to `637bfe3f…`** (arm A: `38d1766b…`), and it
had to: the two flags are new configured values and `NOTES-W13R` §2 makes the
catalog's `dense_closed_loop_body_solve` identity entry conditional on them
being set. This is the digest-level confirmation that they entered the run's
configuration rather than being silently ignored.

**Evidence set — 2 members, identity unchanged for a seventh wave.**

```
member_count               2
evidence_set_cache_key     f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
```

Byte-identical to arm A and to the sealed run.

**Global topology — 6 lookups, 6 hits, 0 misses.** Same three phase keys
(`a5eaf8da…`, `9c9d568f…`, `edea7406…`) as W13D/W13I/W13K/W13L/W13M/W13O.

**Balanced basis — arm A's entry HIT, content-addressed.**

```
balanced_basis_phase_cache_hit
phase_key_sha256      52cbe3fd96650d284b2e2a03395515c3fb467ce846a014712f5d5c1a4e8f683d
completion_sha256     975e828634a3be4e9bfd3df2e4bb23f21e86e3f9ba606262bd52507bd2d20ac3
parent_forward_calls  0
trace_tensor_loads    0
training_steps        0
```

Exactly the key and completion arm A used. As predicted from
`hf_lfm2_dense_budget_model_fold.py` being unchanged (`c2797d77…`), the key did
**not** rotate despite five new commits: the R1/R2 solve changes are strictly
downstream of the basis. Carried forward unchanged: **this run did not build the
merged basis, it inherited the peer's `a0025` payload.**

Pace against arm A: `parent_verified` 8.2 s vs 9.6 s, `carrier_verified` 6m49s
vs 7m06s, `dense_closed_loop_started` 19m35s vs 18m51s. Normal.

---

## 6. The result — refused at layer 0 by the terminal composition floor

```
2026-08-08T08:43:44.980Z  child_closed_loop_layer_started      layer_index=0
2026-08-08T08:45:59.697Z  child_replay_sharding_completed
2026-08-08T08:48:14.513Z  child_replay_sharding_completed
2026-08-08T08:50:04.795Z  child_replay_sharding_completed
2026-08-08T08:54:48.412Z  child_replay_sharding_completed
2026-08-08T08:55:46.978Z  failure          ClosedLoopFoldError, fail_closed=true
2026-08-08T08:55:47.011Z  final_receipt    receipt_status=failed_closed
```

The exact error, verbatim from `receipt.failure`:

```json
{
  "category": "fold_or_infrastructure_failure",
  "exception_chain": [
    {
      "message": "layer:000000 validation rejected the folded child: held-out post-layer residual improved/non-regressed; terminal composed held-out relative RMSE 1.00576305 is at or above the no-branch baseline 1",
      "type": "ClosedLoopFoldError"
    }
  ],
  "fail_closed": true
}
```

```
receipt_sha256        42bc21c308691b831ed6f294f1fa75f2a206629e09c5123d4238aca800164873
receipt.json sha256   5d61ae20913810f06324780e966d11522758352552d461312c549571ef40a1e7
status                failed_closed
configuration_sha256  637bfe3f29c70250929d1310c5409390e72f923261fd3487c935a2d60e7b9f55
tolerance_weakened    false
training_steps        0
folding_only          true
publication_present   false
```

### 6.1 Read the message carefully — both clauses are in it

The two strings are not both failures. Traced to source on the deployed tree:

* `"held-out post-layer residual improved/non-regressed"` is
  `hf_lfm2_dynamic_closed_loop_adapter.py:7498`, the **pass** verdict of the
  counterfactual threshold. **The per-boundary clause passed.** R1+R2 did what
  they were built to do locally.
* `"terminal composed held-out relative RMSE … is at or above the no-branch
  baseline"` is `:7234`, the **absolute** arm of `_terminal_composition_floor`,
  engaged at layer 0 because arm A's line carries
  `--dense-closed-loop-terminal-absolute-floor-minimum-layer 0`. Its bound is
  literally `terminal_rrmse >= 1.0`, and 1.0 is by construction the score of
  predicting exactly zero.

### 6.2 The comparison that makes this decisive

Arm A's own per-layer terminal numbers are in
`reports/stage1-attribution-20260808/per_layer_profile.json`:

| | arm A layer 0 | **arm R layer 0** |
|---|---|---|
| terminal composed held-out relL2 | **0.9698811016092053** | **1.00576305** |
| regression vs pre-install baseline | −0.03238759 (improved) | above the no-signal line |
| absolute floor | passed | **REFUSED** |
| per-boundary residual clause | passed | passed |

**+0.03588 absolute, and across the line.** Same parent, same corpus, same
evidence set (`f62110d7…`), same inherited basis (`52cbe3fd…`), same ceiling
2.0, same floor, same tolerances, same row budgets, same devices — the only
difference in the entire configuration is λ=1e-5 and ratio 8.0 on the three body
MLP fits.

So, stated as narrowly as the evidence allows: **at λ=1e-5 and ratio 8.0, a
better-conditioned and up-to-eight-times-larger body correction improved the
local boundary and made the layer-0 composition worse than emitting nothing.**
That is the A12/W12B signature — better fit, worse composition — reproduced at
production scale on a real 4x refold, which is precisely what `NOTES-W13R` §3
listed as the thing its 240-row toy could not establish.

It is *not* a measurement of λ=1e-4, of ratio 2.0, of the λ-only control arm
W13R §4 also recommended, or of any layer past 0.

### 6.3 It was not a kill, an OOM, or infrastructure

```
killspy, both instances: grep "target_pid=1806785"   -> 0 matches
dmesg -T | grep -i "oom-kill|Out of memory"          -> empty
process 1806785                                      -> exited on its own
writer-reclosure / norm-ratio-ceiling activity       -> 0 (it runs after all 14 layers; never reached)
```

`failure_category: fold_or_infrastructure_failure` is the runner's generic
bucket; the exception chain is a validation rejection. The gate did its job.

### 6.4 What could NOT be measured, and why it matters

**The R1/R2/R3 per-solve numbers do not exist for this run.** The requested vs
allowed prior-delta ratios, the effective ridge, and λ·σ_max — the numbers the
task asked for and the entire reason R3 was built — were computed at layer 0 and
then discarded when the layer was rejected.

Verified, not assumed: `grep -rl "mlp_prior_correction_reports_json"`,
`prior_delta_clamp` and `relative_ridge` over the whole sealed run directory all
return **nothing**. The fail-closed `receipt.json` has no `fold` section at all
(top-level keys are `failure`, `parent`, `planned_target`, the phase caches, the
policies — no fold record). `diagnostics.json` carries none of it either.

The reason is structural: those reports reach disk only through an **accepted**
layer's completion receipt — `accepted_layer_completion_ledger.receipts[N]
.stage_preimages.downstream.compiled.metadata.mlp_prior_correction_reports_json`
in a restart-capsule manifest, or the published child's
`provenance/run.json`. A layer refused by the terminal floor publishes no
capsule, so **R3's telemetry is observable only on layers that pass**. The
first arm that would most have benefited from it is exactly the arm that cannot
see it.

This is a real, newly-exposed instrumentation gap and it should be closed before
the next arm: the prior-correction reports need to reach the failure receipt (or
a rejected-layer diagnostic) the same way they reach an accepted one.

The extraction path itself is built and proved — `/home/cereal/nf-w13s/r12r3.py`
reads the reports out of the newest capsule manifest and was validated against
**arm A's** sealed capsule, which is where the baseline table below comes from.

### 6.5 Arm A's baseline, recovered — what the R-arm was aiming at

From arm A's sealed `capsule-000013` manifest, all 42 body MLP solves:

```
solves 42   ridge_mode absolute (42/42)   prior_delta_clamp block ABSENT (42/42)
report keys: correction_norm, input_width, objective, output_width, prior_norm,
             prior_provided, role, rows, solver_conditioning, underdetermined, update_scale
```

Arm A predates R2/R3, so it seals no `prior_delta_clamp` and no
`relative_ridge`; the requested ratio is recoverable only as `1/update_scale`,
exactly the reconstruction the Stage-A attribution had to perform:

| role | min | median | max |
|---|---|---|---|
| `mlp_gate` | 2.335x | 37.77x | **140.4x** |
| `mlp_up` | **1.446x** | 35.10x | 128.0x |
| `mlp_down` | 15.57x | 49.28x | 140.1x |
| **all 42** | **1.446x** | — | **140.4x** |

against an allowed 1.0x at every one — the attribution's "1.4x–140x", confirmed
independently here. `mlp_down` carries no `solver_conditioning` in arm A at any
layer, which is the R3(c) gap; `mlp_residual_endpoint_relative_rmse` is `None`
at all 14 layers, which is the R3(b) gap.

A ratio of 8.0 therefore released roughly the low tens of percent of a median
request and still bound on most solves. It was not an unbounded step. It was
still enough to cross the floor at layer 0.

---

## 7. Monitoring recipe (for the next arm)

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
W=/home/cereal/nf-w13s
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold

$SSH --command-timeout 900 --execute "bash $W/poll.sh"                 # one-shot full poll
$SSH --execute "python3 $W/w13s_tl.py timeline"                        # stage timeline
$SSH --execute "python3 $W/w13s_tl.py basis"                           # balanced-basis hit detail
$SSH --execute "python3 $W/r12r3.py $RUN"                              # R1/R2/R3 per-solve table  <- NEW
$SSH --execute "python3 $W/r12r3.py $RUN --raw 0"                      # layer 0's raw report block
$SSH --execute "bash $W/ceiling.sh"                                    # ceiling evidence (post-body-reclosure only)
$SSH --command-timeout 15000 --execute "bash $W/waitlayers.sh 6 14000" # block until N layers done
$SSH --execute "grep target_pid=<pid> $W/killspy-w13s.log"             # did anything signal it
```

`--command-timeout` matters: the client default is 300 s and the Bash tool
foregrounds for only ~120 s, so long waits must be backgrounded.

`rss-sampler.sh` and `milestones.sh` bind to `run.pid` at start and self-exit
when it dies (both did); **restart them after any relaunch**:

```bash
setsid nohup bash $W/rss-sampler.sh  > $W/rss-sampler.log       2>&1 < /dev/null &
setsid nohup bash $W/milestones.sh   > $W/milestones-runner.log 2>&1 < /dev/null &
```

**The floor, boundary and norm-ratio numbers still do not stream.** The live
validation event carries `validation_accepted` and a decision digest and no
metrics block, exactly as `NOTES-W13L` §11 and `NOTES-W13O` §6 recorded — and,
per §6.4, on a *rejected* layer even the sealed artifacts carry nothing. The
first observable R1/R2 numbers arrive with the first **accepted** layer's
restart capsule, ~11 minutes into the closed loop.

---

## 8. Caveats

* **This is one arm at one rung, refused at one layer.** It measures λ=1e-5 with
  ratio 8.0 at layer 0. `NOTES-W13R` §4 explicitly says the rung should be set
  from `requested_prior_delta_ratio` read off the first two layers of the arm —
  and §6.4 is why that read never happened. The λ-only control arm W13R §4 also
  recommended has still not been run, and it is now the cheapest next
  experiment: it separates "the ridge hurt composition" from "the released step
  hurt composition", and this run cannot.
* **Whether stage 1 is complete is not settled by this run.** The attribution's
  §4.3 condition is "if R1+R2+R3 does not move decode". This arm never produced
  a child, so it produced no decode and no manual-review verdict. What it did
  produce is a hard negative at one point in the (λ, ratio) plane, plus the
  observation that the boundary and the composition disagree in sign there.
* **The failure is a gate doing its job, not a defect.** No flag was relaxed to
  get past it, and `tolerance_weakened` is `false` in the sealed receipt.
* **R3's telemetry is invisible on rejected layers** (§6.4). Newly found, not
  previously recorded, and it blocked the specific deliverable this task asked
  for. Named as a defect, not smoothed over.
* **`d296b602` remains unfixed and undiagnosed.** This wave routed around it the
  same way W13O did — it is simply not in the chain. `refs/heads/w13o-import`
  and `refs/heads/w13s-import` both carry the objects, so the deployment can be
  reset forward in one command once it is fixed.
* **The merged balanced basis was inherited, not built** (§5). A claim that this
  arm built its own basis end to end would be false.
* **The ceiling 2.0, the floor, and every other arm-A flag are unchanged**, and
  none of them were exercised past layer 0 — the body reclosure the ceiling
  gates runs only after all fourteen installs and was never reached.
* **`terminal_free_run_tripwire_evaluated` is `false` at all 14 of arm A's
  layers** and the R-arm never got far enough to change that. `38cc3bc9` wired
  the tripwire's eyes; this run did not test them.
* **Nothing was weakened.** No flag relaxed, no fail-closed clause bypassed, no
  phase key pinned, no peer source edited, no cherry-pick conflict resolved ad
  hoc — there were none to resolve.
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
* **Not committed.** No `git add`, no commit, no push. The local working tree was
  not modified except for this file; the cherry-pick dry run happened in an
  isolated `--shared` clone under the scratchpad.

---

## 9. Artifacts

Local:

| path | |
|---|---|
| `NOTES-W13S.md` | this file |

Server, under `/home/cereal/nf-w13s/`:

| file | |
|---|---|
| `preservation.txt` | §1's before/after proof for **both** moves, plus arm A re-verified at the end |
| `nf-w13s.bundle` | the deployment bundle, `009aa712…`, 540,603 B |
| `HEAD-before.txt`, `TREE-before.txt`, `status-before.txt` | pre-cherry-pick deployment state (`2db9f085…` / `14dc0ebf…` / 0 bytes) |
| `deploy.txt` | the fetch + five cherry-picks, ending in HEAD/TREE |
| `deploy-verify.txt` | §2.4's deployed digest table and semantic greps |
| `flagcheck.txt` | §3's `--help` reconciliation, 111/111, must-be-absent list |
| `runner-help.txt`, `help-flags.txt`, `used-flags.txt`, `used-flags-refold.txt` | the reconciliation inputs |
| `patch_launcher.py` | the fail-closed launcher patcher (anchor-unique or refuse) |
| `nf-launch-r12-round-refold.sh.bak-w13s-20260808T082203Z` | *(in `…/deployments/`)* the pre-edit launcher, `6098d179…` |
| `killspy-rearm.txt`, `killspy-w13s.log` | §4's fresh 24 h window and its own log |
| `launch-start.txt`, `launch-refold.out` | the launch record |
| `detachment-proof.txt` | §4's `/proc` proof |
| `failclosed.txt` | §6's failure event, final receipt, and receipt digests |
| **`r12r3.py`** | **the R1/R2/R3 per-solve reader — validated against arm A's sealed capsule** |
| `poll.sh`, `w13s_tl.py`, `ceiling.sh`, `tl.py`, `timeline.py`, `waitfor.sh`, `waitlayers.sh`, `entryinfo.py` | the W13L/W13O toolkit, repointed to `nf-w13s` |
| `rss-sampler.sh`, `milestones.sh` | live samplers (self-exited with the run) |

Preserved, never modified:

| path | |
|---|---|
| `…/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-armA-complete` | **arm A, sealed complete, frozen-reviewed 0/15** — receipt file `45bc68c2…`, `receipt_sha256` `1cdec0a6…` |
| `…/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-w13m-firstcomposed` | the sealed first composed r13 run, receipt `c9a8f746…` |
| `…/a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-closed-terminal-floor-L0-w13s` | **this wave's R-arm**, receipt `42bc21c3…`, `failed_closed` |
