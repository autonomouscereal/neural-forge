# NOTES-W13F — r13 refold relaunch on the merged-evidence balanced-basis fix

Wave 13F, branch `codex/lrc-blank-compression`. This deploys `84acded2`
(*Let merged base+extension evidence through the balanced-basis lane*), drops
streaming vocabulary-margin v2 per the coordinator decision recorded against
`NOTES-W13E` §6, and relaunches the refold that `NOTES-W13D` failed closed.

**Status: FAILED CLOSED at 10:44:13.7 UTC, 10.1 seconds in, at configuration
construction — before the runner did any work.** (Almost all of that is the
`torch`/`transformers` import; the config raised on the first thing it checked.) The W13E fix is deployed and
is not implicated. The blocker is a **mutual exclusion between two fail-closed
clauses** that makes a carrier-routed refold consuming an on-policy round
impossible on this commit in *either* margin configuration (§4). Nothing was
weakened, no flags were retried, and no GPU work started.

The deployment bump (§1) and the launcher edit (§2) are both verified good and
are exactly what was asked for. What they revealed is that the coordinator's
recorded decision — "drop v2, keep plain observability" — is **not executable
on `84acded2`**, because the production config validator *requires* v2 whenever
carrier routes are on. That requirement was not visible from `NOTES-W13E` §6,
which examined the balanced-basis side only.

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.

---

## 1. Deployment bump: 53b353f0 → 84acded2

```
bundle  53b353f0ef5f2e105c35fc58c4ed28f3a6d4f2bf..codex/lrc-blank-compression
        (3 commits; 8e08844c, ecb16477, 84acded2)
size    136,275 bytes
sha256  8f52c9eedc1a60772da6d19a73f5ca1ed92b8f398d795c6ac4eaca2bed442116   (identical both sides)
server  /home/cereal/nf-w13f/nf-w13f.bundle
```

`git bundle verify` clean inside the deployment repo, then
`git fetch <bundle> refs/heads/codex/lrc-blank-compression:refs/heads/w13f-import`
and `git reset --hard 84acded2…`. Pre-reset state recorded in
`/home/cereal/nf-w13f/HEAD-before.txt` (`53b353f0…`) and `status-before.txt`
(**0 bytes** — the tree was clean, nothing was overwritten).

Deployment `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`:

| | |
|---|---|
| `git rev-parse HEAD` | `84acded28acabcac1603fb7a46b0e85194d8f192` |
| `git status --porcelain` | **empty** |

Deployed file digests, compared against the local committed blobs at both
`53b353f0` and `84acded2` (`git cat-file blob`, not assumed from commit titles):

| file | deployed sha256 | vs `53b353f0` |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **SAME — frozen collector holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | **SAME** |
| `hf_lfm2_balanced_basis_phase_cache.py` | `6bb971d558a573a5d1dba9dcaf11220b8cc8294d457258fc26cbcf08435b872f` | DIFF (W13E §2.1) |
| `hf_lfm2_sealed_evidence_set.py` | `8e6e608908e6eaee384cb54f52ab064bc0f3806621d89d8239b2d84683b2358c` | DIFF (W13E §2.2) |
| `hf_lfm2_balanced_circuit_basis.py` | `36d3c965a32cf8f9cbeb5321b81ca3822facfbf0b26d1a8be35012e9b041240b` | DIFF (W13E §2.3) |
| `hf_lfm2_dense_budget_model_fold.py` | `ec9cc29d1800737f1aadb8cdc363c7fb4534bcfa1cbac0db3622c4c3c7596117` | SAME |
| `dynamic_closed_loop_fold.py` | `ab357dba7d52d59187dc1f4c0481f6d57d72a670d4c5c366880295b9474f7b09` | SAME |
| `experiments/run_lfm2_dynamic_catalog.py` | `1f2069489e4016794854a1a15187cc2fa44b9c188867c500e2cd838647652b0e` | SAME |
| `experiments/run_r12_on_policy_round.py` | `ad5e72a54880360af3887a4d4641968d6dc9fe7a5299d69000343ca7dcc000ec` | SAME |

The frozen-collector rule holds: `16271526…` is byte-identical at `13f71132`,
`53b353f0` and `84acded2`. Exactly the three W13E files moved.

Codex's uncommitted local diffs to `hf_lfm2_dense_budget_model_fold.py` and
`dynamic_closed_loop_fold.py` are **not** deployed — the deployment carries the
committed blobs, which is why both show SAME above.

---

## 2. Launcher edit

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`

```
backup       nf-launch-r12-round-refold.sh.bak-20260805T104136Z   (also /home/cereal/nf-w13f/…preinstall)
pre-edit     e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda
launched     f49d4861827cd855be81b37045097363c4cfa2e62d8f3a27e4ecd2b03b59217b
installed    24c61cb32a5738eb13598ee658769dc64916df4445cacd49763212f4f5710633   (header amended after the failure, §2.2)
```

`bash -n` clean locally and on the server, before and after each install.

### 2.1 The flag change — the whole of it

```diff
     --dense-vocabulary-margin-observability \
-    --dense-streaming-vocabulary-margin-v2 \
-    --dense-streaming-vocabulary-margin-pair-chunk-size 256 \
     --sample-last-tokens 256 \
```

A `diff` of the entire `refold)` stage against the pre-edit launcher shows
**exactly these two deletions and nothing else**. Gate/floor/telemetry stay on,
EIV and margin row weights stay off, install verification stays `fingerprint`,
both phase caches stay on, no phase key is pinned.

All **106** flags the refold stage passes were checked against the deployed
runner's `--help` at `84acded2` (**179** options parsed): every one present.
That check passes and is not what failed — the refusal is semantic, inside
`__post_init__`, not an argparse rejection.

### 2.2 Header amended after the failure

After the run failed, the W13F header block I had added was rewritten, because
its original wording asserted that removing v2 lets the phase caches run as
composed — which the launch proved false. Leaving a false operational claim in
a launcher is a hazard for the next operator. **The flag set was not touched by
this amendment** (verified by the same stage-level `diff`); only the comment
block changed, and `bash -n` is clean. The installed header now states the
mutual exclusion, names both code sites, and says a source decision is required.

---

## 3. Relaunch timeline (UTC, 2026-08-05)

| | |
|---|---|
| 10:3x | bundle built (`8f52c9ee…`), uploaded, `git bundle verify` clean; deployment reset to `84acded2`, tree clean, digests checked |
| 10:41:36 | launcher backed up (`…bak-20260805T104136Z`, `e496c102…`) |
| 10:4x | edited launcher installed (`f49d4861…`), `bash -n` clean; 106/106 flags confirmed against the deployed `--help` |
| 10:43:26 | W13D's failed run moved aside → `…-round-0-refold.failed-merged-basis-104326` |
| **10:44:03** | `refold` stage launched, `SOURCE_COMMIT=84acded2…` |
| 10:44:0x | preflight OK — both caches, both rebound envelopes, round preimage; identical to W13D §6.1 |
| **10:44:13.7** | **failed closed** — `ProductionLfm2CatalogError` at config construction (`run.log` mtime; `run.pid` was written at 10:44:03.637) |
| 10:44:53 | first poll: process already dead, `run.log` 1,242 bytes |
| 10:47:27 | run dir moved aside → `…-round-0-refold.failed-config-streaming-v2-104727` |
| 10:48–10:5x | launcher header amended and reinstalled; final installed form `24c61cb3…` (flag set byte-identical to the launched form throughout) |

```
PID   1426127  (started 10:44:03.637, exited 10:44:13.7 — 10.1 s wall clock)
RUN   …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-config-streaming-v2-104727
log   $RUN/run.log   1,242 bytes   sha256 9c153bec1ccde1b6fc942cf714ec20aae213d694744f100300fa8560cc5b8b41
```

GPU state at launch and after: cuda:0 308 MiB, cuda:4 499 MiB (ComfyUI,
coexisting), both 0% — the same idle r12 pair, reused as instructed. No GPU
work started.

**There is no receipt.** The runner raised while constructing
`ProductionLfm2CatalogConfig`, before the run object existed, so no
`receipt.json`, `diagnostics.json`, `ledger.jsonl`, `balanced-basis-phase-events.jsonl`
or `published-child/` was written. `run.log` above is the entire evidence.

---

## 4. The failure, exactly — a proven mutual exclusion

```
ProductionLfm2CatalogError: Carrier-routed production must pin the best proven
folding lane: moving-gauge-v3 closed loop, canonical prefix replay,
diagnostic-only closed-loop metrics, coupled value/output attention with
geometry risk, and streaming-v2 vocabulary margin observability.

  run_lfm2_dynamic_catalog.py:14738  main
  run_lfm2_dynamic_catalog.py:14373  config_from_args
  run_lfm2_dynamic_catalog.py:2135   ProductionLfm2CatalogConfig.__post_init__
```

### 4.1 Clause A — carrier routes REQUIRE streaming v2

`run_lfm2_dynamic_catalog.py:2123-2141`. Guarded by `uses_carrier_routes`,
which is simply `carrier_route_publication is not None` (`:2888`) — true for
this arm. The conjunction requires all of:

```
dense_run_closed_loop
dense_body_reclosure_mode == "moving_gauge_v3"
resolved_dense_closed_loop_replay_capture_mode == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
dense_closed_loop_performance_metrics_diagnostic_only
dense_attention_consumer_mode == "coupled-value-output"
dense_attention_geometry_risk_enabled
dense_vocabulary_margin_observability
dense_streaming_vocabulary_margin_v2        <-- the only false term
```

The launcher passes every other term. `dense_streaming_vocabulary_margin_v2` is
the sole term the edit turned off, and `--dense-streaming-vocabulary-margin-pair-chunk-size`
does not appear in the conjunction. W13D's run, identical except that v2 was on,
passed this same check and reached the balanced-basis phase 53 minutes later —
so the failed term is established by the two runs, not only by reading.

### 4.2 Clause B — merged evidence REFUSES streaming v2

`hf_lfm2_balanced_circuit_basis.py:3179-3190`, new in `84acded2`:

> The streaming vocabulary-margin adapter binds one sealed cache manifest
> digest and cannot consume merged on-policy extension evidence.

and the session handed to that builder is **unconditionally** the merged one
whenever the evidence set carries extension evidence —
`_resolve_dense_evidence_handles` (`hf_lfm2_dense_budget_model_fold.py`) ends:

```python
if evidence_set is None or not evidence_set.has_extension_evidence:
    return resolved, selective_session
if selective_session is None:
    raise Lfm2DenseBudgetFoldError(...)
return resolved, evidence_set.merged_session()
```

There is also no v2-without-session escape: `:3173` refuses
`streaming_vocabulary_margin_v2` when `selective_session is None`.

### 4.3 The deadlock

| configuration | outcome | when |
|---|---|---|
| streaming v2 **ON** | refused at clause B, in the balanced-basis builder | ~50 min in (W13D reached the phase at 09:21; W13E measured the refusal on toy evidence) |
| streaming v2 **OFF** | refused at clause A, at config construction | 10.1 s, nearly all import (**this run, measured**) |

No third option exists on `84acded2`: the deployed `--help` (179 options)
contains **no** override / allow / force / unpin flag for the carrier
production pin — checked by grep over the parsed flag list.

Dropping `--carrier-route-publication` would clear clause A, but it deletes the
arm: this refold *is* the carrier-routed integrated child, and per `NOTES-W13E`
§7 non-carrier lanes select `weight_complete` and never build a balanced basis
at all, so the W13E fix would go unexercised.

### 4.4 Why W13E did not catch it

`NOTES-W13E` §6 asked exactly the right question — "confirm whether the r13
launcher passes `--dense-streaming-vocabulary-margin-v2`" — and correctly
identified the toy carrier lane
(`test_run_lfm2_dynamic_catalog_carrier_e2e.py:813`) as pinning both flags
whenever the residual basis is `circuit_balanced`. What it read as a *test
fixture convention* is in fact enforced by the **production config validator**
for every carrier-routed run. So §6's option "the relaunch must drop margin v2
for this arm" is not available; only its alternative — "a separate design
change" — is.

### 4.5 Side effects: none

- Nothing written under either phase-cache root since launch
  (`find … -newermt "2026-08-05 04:44"` empty).
- The three global-topology entries W13D published are intact and still keyed
  on this evidence identity, so the ~38 min saving survives for a future
  relaunch:
  `lfm2-global-capacity-union-v1/a5eaf8da…`,
  `lfm2-global-functional-width-v1/9c9d568f…`,
  `lfm2-global-frontier-selection-v1/edea7406…`.
- `round-000`, the sealed r12 run directory, the sealed 2026-07-17 pack, both
  base caches and the rebound envelopes: untouched
  (`find … -newermt "2026-08-05 04:00"` empty on the round and the r12 run dir).
- No child published, no GPU allocation, no process left running.

---

## 5. What was NOT observed

Stated plainly, because the task asked for these and this run produced none of
them:

- **Evidence set opened** — not reached. Expected 2 members,
  `evidence_set_cache_key f62110d7…`, 73,989 rows.
- **Global-topology phase reuse** — not reached, so the "REUSED, not
  recomputed" skip is still **unconfirmed**. The entries exist and are keyed on
  the evidence identity, but no run has yet hit them.
- **The merged balanced-basis phase publishing** — not reached. No
  `balanced_basis_merged_evidence_bound` event, no published phase key, no
  timing. **The W13E fix has still never executed on production-scale
  evidence.**
- **First body-layer solve, floor/tripwire and writer-reclosure-acceptance gate
  events** — not reached.
- **RSS peak: none observed.** The process was already dead at the first poll
  (50 s after launch), so no `ps -o rss=` sample was ever taken. The ~27.7 GB
  row-slicing cost of plain observability and the 138 GB r12 peak remain
  *unmeasured here* — the RSS watch the coordinator asked for could not begin.
  It stays an open risk for whatever configuration eventually runs.

---

## 6. The decision the relaunch needs — not taken here

The coordinator's recorded decision (drop v2, keep observability, accept the
row-slicing cost) is the right *intent* and cannot be executed by configuration.
Making it executable is a source change, and it is a design call:

1. **Stop pinning streaming v2 in clause A for extension-bearing runs.** The
   smallest change that invents no authentication claim: the carrier production
   pin currently requires a flag that is provably unavailable when the run
   consumes merged evidence, so for those runs it pins an impossibility. A
   one-clause amendment in `ProductionLfm2CatalogConfig.__post_init__`
   (require observability always; require v2 only when the run consumes no
   on-policy extension evidence) makes the composed r13 arm launchable, at the
   memory cost already accepted. Costs a new commit and another deployment bump,
   and it does change what "the best proven folding lane" means for
   extension-bearing carrier runs — which is precisely why it is not mine to
   take.
2. **Teach the streaming margin adapter merged evidence.** `NOTES-W13E` §6
   declined this deliberately: it would require handing a derived digest to an
   authenticator that documents itself as binding a real sealed manifest file.
   That is a design call about what an authentication claim means.
3. **Drop carrier routes**, which deletes the arm (§4.3). Listed for
   completeness.

Until one is taken, no configuration of this launcher can start the r13 refold.

---

## 7. Monitoring recipe (for whichever relaunch happens)

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold

# alive + RSS in one shot (RSS is the number to watch on the v2-off arm)
$SSH --execute "ps -o pid=,etime=,rss=,stat= -p \$(cat $RUN/run.pid)"

# stage stream (one JSON object per line, 'stage' is the event name)
$SSH --execute "grep -o '\"stage\":\"[a-z_0-9]*\"' $RUN/run.log | tail -20"

# the phase this wave has never seen run
$SSH --execute "grep -E 'phase_cache|balanced_basis|merged_evidence|evidence_materialized' $RUN/run.log | tail -12"

# the r13 arm's decision events
$SSH --execute "grep -E 'terminal_composition|terminal_absolute_floor|terminal_free_run|boundary_acceptance|refused|closed_loop_layer' $RUN/run.log | tail -40"
```

A ready-made poller is on the server at `/home/cereal/nf-w13f/poll.sh` — it
reports alive/RSS, log size, last 12 stages, phase-cache and merged-evidence
events, the balanced-basis event-log size, a failure count and the two GPUs.
Run it with `bash /home/cereal/nf-w13f/poll.sh`.

Read, in order: preflight → `on_policy_extension_evidence_materialized`
(2 members, `f62110d7…`, 73,989 rows) → global-topology phase **hits** (the skip
is unproven, confirm it) → `balanced_basis_merged_evidence_bound` and the
published merged phase key (the new, unmeasured cold build on 73,989 rows) →
`child_closed_loop_layer_validation_completed` for layer 0. Sample RSS at every
poll; the v2-off arm reads whole tensors per batch over a 27.7 GB retained
corpus, and nobody has measured that peak.

`NOTES-W13D` §7's phase timings and §8's terminal-floor caveat still stand
unchanged — in particular, if the run aborts at layer 0, read
`terminal_baseline_relative_rmse` **before** concluding anything.

Any relaunch must have a clear run directory: the launcher exits 40 if
`…-round-0-refold` exists. Both failed runs are already moved aside (§8), so the
path is currently clear.

---

## 8. Caveats

- **Zero r13 gate evidence, again.** Like W13D, this wave produced no
  measurement of the reclosure acceptance gate, the terminal floor, or the
  telemetry. It also produced no measurement of the W13E fix. What it produced
  is a proof that the arm cannot start on `84acded2`.
- **The W13E fix is deployed but unexercised at scale.** Its toy-scale proofs
  (W13E §5, §8) are all the evidence there is.
- **RSS unmeasured** (§5). The coordinator's cost estimate for dropping v2
  (~27.7 GB row slicing) is inherited from W13E, not confirmed on this box.
- **The launcher is left in the v2-OFF form**, which fails in ten seconds
  with a message naming clause A. That is deliberate: it is the coordinator's
  recorded decision, and it is a far cheaper failure than the v2-ON form's
  ~50-minute path to clause B. `nf-launch-r12-round-refold.sh.bak-20260805T104136Z`
  restores the v2-ON form if the deep failure is wanted.
- **Nothing was weakened and nothing was retried.** One launch, one failure,
  stop — as instructed.
- **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
- **Not committed.** No `git add`, no commit, no push.

---

## 9. Artifacts

Server, under `/home/cereal/nf-w13f/` (kept):

| file | what it is |
|---|---|
| `nf-w13f.bundle` | the deployment bundle, sha256 `8f52c9ee…` |
| `HEAD-before.txt`, `status-before.txt` | pre-reset deployment state (`53b353f0…`, clean) |
| `nf-launch-r12-round-refold.sh.preinstall` | the pre-edit launcher, `e496c102…` |
| `nf-launch-r12-round-refold.sh.new` | the installed launcher, `24c61cb3…` |
| `launch-refold.out` | the launcher stage's own stdout (preflight + `RUN=`/`PID=`) |
| `runner-help-84acded2.txt`, `help-flags.txt`, `used-flags.txt` | the 179-option `--help` and the 106/106 flag check |
| `poll.sh` | the monitoring poller (§7) |

Preserved failed runs, under
`…/neural_forge_exact4x_20260730/catalog-integrated-carrier/` — **neither
deleted**:

| directory | why it is evidence |
|---|---|
| `a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-merged-basis-104326` | W13D's run. `receipt.json` (283,801 B) status `failed_closed`, `Lfm2BalancedBasisPhaseCacheError: trace_cache must be Lfm2CircuitTraceCache.`; `run.log` 336 KB; `diagnostics.json` 320 KB. Its three global-topology phase entries are still live |
| `a0023-exact4x-h816-mgv3-r12-round-0-refold.failed-config-streaming-v2-104727` | this run. `run.log` 1,242 B `9c153bec…`, `source-commit.txt` `84acded2…`. No receipt — the failure precedes the run object |

Deployment: `nf-launch-r12-round-refold.sh.bak-20260805T104136Z` (`e496c102…`,
the W13D-installed launcher) alongside W13D's own
`…bak-20260805T083111Z` and the older `…bak-20260805T042559Z`.

`round-000`, the sealed r12 run directory, the sealed 2026-07-17 pack, both base
caches, the rebound envelopes and both phase-cache roots were never written to.
