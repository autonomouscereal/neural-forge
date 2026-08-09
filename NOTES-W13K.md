# NOTES-W13K — r13 refold, attempt #4: relaunched at 6f178f14, properly detached

Wave 13K, branch `codex/lrc-blank-compression`. This deploys `6f178f14`
(*Bulk-derive child replay checkpoint requests*, 12 commits past `8a4ed56f`),
re-verifies the launcher `NOTES-W13I` restored, moves that wave's externally
killed partial run aside, and relaunches the refold **under `setsid`** so no
session or process-group teardown can reach it again.

Predecessors: `NOTES-W13D` (the arm's composition + attempt #1),
`NOTES-W13F` (attempt #2), `NOTES-W13I` (attempt #3 — ran healthy to two
accepted layers before being killed externally; its procedures are this wave's
template).

**Status: LAUNCHED CLEAN, DIED SILENTLY AT 25 MINUTES — and the "it was
attached to a session" hypothesis is now falsified.** Launched
2026-08-05T22:01:35Z, run started 22:01:45.574Z, **PID `3454887`, PPID `1`
(systemd), own session `3454853`, no controlling terminal — detachment proved
from `/proc` (§4)**. It cleared every wall the previous three attempts died on,
reached closed-loop **layer 0's `pre_producer` capture in 20m17s** (W13I needed
1h08m46s to the same point), and then **vanished between 22:26:41 and 22:26:56**
with zero failure events, zero traceback, no OOM, no `systemd-oomd` record and
no crash file (§8).

It was **not** killed by session or process-group teardown: it was provably
outside both. Something on that host is killing these runs, and this wave
installed the instrument that will name it (§8.4).

The box is now occupied by the peer's `a0026-…-w13l-cold`, launched
2026-08-05T22:33Z from a `b65c81f5` worktree deployment, on the same two GPUs
this arm needs. **No relaunch was attempted into that contention** (§9).

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.
Run directory
`/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold`.

---

## 1. Deployment bump: 8a4ed56f → 6f178f14

```
bundle  8a4ed56f46dc74b6de6d3278687ca966c7351201..codex/lrc-blank-compression
        (12 commits; e15eb9fe … 6f178f14)
size    217,747 bytes
sha256  c38785c5b4d1334d1382f04dab54f20325fc0058f7c5a7263432c230df79e840   (identical both sides)
server  /home/cereal/nf-w13k/nf-w13k.bundle
```

`git bundle verify` clean on both sides, then
`git fetch <bundle> refs/heads/codex/lrc-blank-compression:refs/heads/w13k-import`
and `git reset --hard 6f178f14…`. Pre-reset state recorded in
`/home/cereal/nf-w13k/HEAD-before.txt` (`8a4ed56f…`) and `status-before.txt`
(**0 bytes** — the tree was clean, nothing was overwritten).

Deployment `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`:

| | |
|---|---|
| `git rev-parse HEAD` | `6f178f14add54ec882c09ad183f69a626a0cfb48` |
| `git status --porcelain` | **empty** |

Deployed working-tree digests, each compared against the **local committed
blob** at both `8a4ed56f` and `6f178f14` (`git cat-file blob`, not inferred
from commit titles):

| file | deployed sha256 | vs `8a4ed56f` |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **SAME — frozen collector holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | **SAME** |
| `hf_lfm2_streaming_vocabulary_margin_adapter.py` | `2a949a3076af23b829b31af2410d245f2d3f25355472a13740f3dd7879e051d6` | SAME |
| `hf_lfm2_balanced_circuit_basis.py` | `e1091d33fe03c95d338d4d967f1750e3d47f21a0223eaba367c317abb55c4108` | SAME |
| `hf_lfm2_balanced_basis_phase_cache.py` | `6bb971d558a573a5d1dba9dcaf11220b8cc8294d457258fc26cbcf08435b872f` | SAME |
| `hf_lfm2_sealed_evidence_set.py` | `8e6e608908e6eaee384cb54f52ab064bc0f3806621d89d8239b2d84683b2358c` | SAME |
| `experiments/run_r12_on_policy_round.py` | `ad5e72a54880360af3887a4d4641968d6dc9fe7a5299d69000343ca7dcc000ec` | SAME |
| `experiments/run_lfm2_dynamic_catalog.py` | `2af6de96d8f147dbaf03db91a41545c3724179ba53859dfbb312a391aea04ee7` | DIFF |
| `hf_lfm2_dense_budget_model_fold.py` | `c2797d77d041bb9cadac9b578dbc914ae59f37084846d1ea0e10397e37002a3f` | DIFF |
| `dynamic_closed_loop_fold.py` | `2747ed84f5bff89b1adab1a8565e900a6d4a1f46a900dc43ced1eec4a845ecdc` | DIFF |
| `hf_lfm2_child_closed_loop_refinement.py` | `bd4b0cd957251c7060e130d3c28a5156fac0e434b9825f2a24e3b1cbe52bebe0` | DIFF |
| `hf_lfm2_closed_loop_restart_capsule.py` | `793ed8cc1876ac953520acdde3f8bcb95ad3ff4d7ea9b6b46e123bbc80b5e90d` | DIFF |
| `hf_lfm2_dynamic_closed_loop_adapter.py` | `9606e1e459e5f4619882f255ed3a034480488986177edf7a7fe2accec299700d` | DIFF |
| `hf_lfm2_child_replay_phase_cache.py` | `c326f5a84a18b63931fb4a0683a8c8aeebb1015c96d6affc7ecbaff592fccc0b` | new file |
| `hf_lfm2_child_replay_batch_checkpoint.py` | `b5078a1b4910117c22bbd9cbcdf6e9ec4470b296a8a5d99de3871498dec94c67` | new file |

Every deployed digest equals its local committed blob exactly.

**The frozen-collector rule still holds.** `16271526…` is byte-identical at
`13f71132`, `53b353f0`, `84acded2`, `8a4ed56f` and now `6f178f14`;
`hf_lfm2_canonical.py` likewise (`789c6a6f…`). Neither file appears in
`git diff --name-only 8a4ed56f 6f178f14` at all. The two files `NOTES-W13G`
moved (`…margin_adapter`, `…balanced_circuit_basis`) did **not** move again, so
the balanced-basis implementation fingerprint that rotated at
`84acded2`→`8a4ed56f` is unchanged here.

`git diff --name-only 8a4ed56f 6f178f14` is 33 files: 15 runtime source files
(above), 17 test files, and `NOTES-W13J.md`.

### 1.1 One thing the bump exposed that is worth writing down

The venv's default `sys.path` still resolves `neural_forge` to a **stale
checkout**, `/home/cereal/model-surgery-lab/neural-forge-src/src/neural_forge/`,
which does not carry the frozen collector's current API — running the runner
without the launcher's environment dies with

```
ImportError: cannot import name 'Lfm2CircuitTraceSelectiveAuthenticatedSession'
  from 'neural_forge.hf_lfm2_circuit_trace_cache'
  (/home/cereal/model-surgery-lab/neural-forge-src/src/neural_forge/hf_lfm2_circuit_trace_cache.py)
```

The launcher's `export PYTHONPATH="$RUNTIME/src:$RUNTIME"` (line 137) is what
puts the deployment ahead of it, and it was verified rather than assumed:

```
pkg : …/deployments/neural-forge-postpub-e893767a/src/neural_forge/__init__.py
coll: …/deployments/neural-forge-postpub-e893767a/src/neural_forge/hf_lfm2_circuit_trace_cache.py
```

Any ad-hoc check of this runtime must export that `PYTHONPATH` or it is testing
the wrong tree. (W13I's `--help` reconciliation implicitly did; this note makes
it explicit, because the first attempt at it here did not and failed loudly.)

---

## 2. Launcher verification — unchanged, and re-audited flag by flag

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`

```
sha256  e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda   <- the expected v2-ON form
cmp     byte-identical to nf-launch-r12-round-refold.sh.bak-20260805T104136Z
bash -n clean
size    25,122 bytes, 490 lines
```

**Nothing had to be restored.** The file `NOTES-W13I` installed is still in
place; no `.bak-*` fallback was needed. The five backups
(`…bak-20260805T042559Z`, `…083111Z`, `…104136Z`, `…bak-v2off-20260805T123653Z`)
are untouched.

### 2.1 The r13 arm, read off the installed file (refold stage, lines 289–490)

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
comment  --balanced-basis-existing-phase-key-sha256   line 82 only:
         "#     No --balanced-basis-existing-phase-key-sha256 is pinned (r11's"
```

**No phase key is pinned.** The only occurrence of that spelling in the file is
the comment that says so.

### 2.3 Flag/`--help` reconciliation at 6f178f14

`experiments/run_lfm2_dynamic_catalog.py` moved in this bump, so the check was
redone against the deployed runner rather than inherited:

```
deployed --help at 6f178f14   181 options parsed   (/home/cereal/nf-w13k/runner-help-6f178f14.txt, 52,062 bytes, exit 0)
refold stage passes           108 distinct flags
used flags absent from --help 0
```

The option set grew by exactly four against `8a4ed56f`'s 178 — and all four are
new levers this arm deliberately does **not** pull:

```
--child-replay-phase-cache-root
--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only     <- the new relaxation lever, OFF
--dense-closed-loop-intermediate-numeric-checks-diagnostic-only         <- OFF
--dense-terminal-only-restart-resume                                    <- OFF
```

Each was confirmed twice: present in `--help` (so the spelling is real and the
absence is not a typo) and absent from the launcher's argument list. The one
option `comm` reported as "removed" (`--dense-closed-`) is a line-wrap artifact
of W13I's parsing regex, not an option; no option the launcher uses disappeared.

The 108-flag set is byte-for-byte the set `NOTES-W13I` §2.1 verified.

---

## 3. The killed partial run, preserved

`NOTES-W13I`'s run directory was moved, not deleted:

```
from  …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
to    …/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold.killed-external-220056
mv -n, exit 0, at 2026-08-05T22:00:56Z
du -sb before 1,236,990,855   after 1,236,990,855   (identical)
```

It holds attempt #3's `run.log` (545,288 bytes, **2 `validation_accepted:true`
layers**), its `restart-capsules/`, `snapshots/`, and its
`balanced-basis-phase-events.jsonl`.

It also holds evidence this wave did not create and did not touch: a
`resume-468ce5e1.log` (16,969 bytes), `resume-source-commit-468ce5e1.txt`
(`468ce5e1ead7e958871dcaf5e851eb5bc91ffdc7`), `run-resume-468ce5e1.pid` and a
`run.pid.w13i-stopped` holding W13I's `1745945` — i.e. **someone stopped W13I's
run and started a terminal-only resume against it at 2026-08-05T20:35Z**, from
a different commit than the one W13I launched. That resume is also dead. The
whole directory moved as one unit so that record stays intact.

The two older preserved failures
(`…failed-merged-basis-104326`, `…failed-config-streaming-v2-104727`) and
Codex's two cold run directories (`a0024-…-w13j-cold`, `a0025-…-w13k-cold`)
were **not touched, not read into, not moved**.

---

## 4. Detachment — the thing attempt #4 exists to fix

Two peer cold runs (a0024, a0025) died silently today, most likely launched
attached to an interactive session. This launch is detached three ways and the
detachment was **proved after the fact**, not assumed:

```bash
cd /home/cereal/model-surgery-lab/deployments
export SOURCE_COMMIT=6f178f14add54ec882c09ad183f69a626a0cfb48
setsid nohup bash ./nf-launch-r12-round-refold.sh refold \
    > /home/cereal/nf-w13k/launch-refold.out 2>&1 < /dev/null &
disown
```

* `setsid` — the launcher shell became a **session leader** in a brand-new
  session with no controlling terminal (`ps` state `Ss`, `tty ?`, `sess ==
  pid == 3454853`). A SIGHUP to the sshd session's group cannot reach it.
* `nohup` + `> …launch-refold.out 2>&1 < /dev/null` — the launcher's own
  pattern, which additionally frees the SSH channel so the client can close.
* `disown` — nothing left in the invoking shell's job table.

The launcher then background-forks the runner itself (`nohup "$VENV_PYTHON" -u
… &`) and exits, so the runner is reparented. **Measured 14 seconds after
launch, from `/proc`:**

```
run.pid                                   3454887
ps  -o pid,ppid,pgid,sess,tty,stat        3454887  1  3454853  3454853  ?  Rl
/proc/3454887/stat                        pid=3454887 ppid=1 pgrp=3454853 session=3454853 tty_nr=0
parent pid 1                              /usr/lib/systemd/systemd --system --deserialize=158 splash
launcher shell 3454853                    already exited
```

**PPID is 1 (systemd), not an sshd session**, `tty_nr` is 0, and the session id
is the setsid session, not the login session. That is the evidence the previous
two silent deaths lacked.

---

## 5. Launch timeline (UTC, 2026-08-05)

| | |
|---|---|
| 21:55 | `nvidia-smi`: cuda:0 308 MiB, cuda:4 499 MiB, both 0% — the idle r12 pair. (GPU 6, an RTX 8000, is a ComfyUI instance at 100%; untouched.) |
| 21:56 | bundle built (`c38785c5…`, 217,747 B), uploaded, `git bundle verify` clean both sides |
| 21:58 | deployment fetched + `reset --hard` to `6f178f14`, tree clean, 15 file digests checked against local blobs |
| 21:59–22:00 | launcher sha/`cmp`/`bash -n`; 108/108 flags reconciled against the deployed `--help`; must-be-absent list checked by name; phase-cache root inventoried (`phase-cache-before.txt`) |
| **22:00:56** | **killed partial moved aside** as `…killed-external-220056` (byte counts equal) |
| **22:01:35.437** | **`refold` stage launched** under `setsid`, `SOURCE_COMMIT=6f178f14add54ec882c09ad183f69a626a0cfb48` |
| 22:01:3x | preflight OK — child bridge reference, three corpus digests, both base caches' selection `f229e5f3…`, round-0 preimage (115 batches), both rebound envelopes binding `parent_cache_key c29b6737…` and carrying the sealed pack rows |
| 22:01:45.574 | `run_started` + `dense_replay_sharding_requested`; `configuration_sha256 c79cbe9704438328d56d77ae5403ea50d7a3f7c52a28956724fa50dcad6e38fc` |
| 22:01:45.609 | `directories_ready`, `balanced_basis_phase_cache_runtime_verified` |
| 22:01:45.996 | `local_artifacts_verified` |
| **22:01:53.020** | **`parent_verified` — clause A cleared in 7.4 s** (W13F died at this wall; W13I took 17 s) |
| 22:01:49 | detachment proved from `/proc` (§4) |
| 22:07:01.733 | `parent_cache_verified`, `tokenizer_bridge_started` |
| 22:07:29.451 | `tokenizer_bridge_verified` |
| 22:07:33.859 | `carrier_verified` (5m48s in) |
| **22:07:34.030–.122** | **three `global_topology_phase_cache_hit`, 0 misses** (§6.1) |
| 22:07:34.409 | `global_topology_selected_width_reconstructed` — `selected_hidden_size 816`, `frontier_sha256 8a4a1ac8…`; `global_frontier_preplanned` |
| 22:12:30.592 | `parent_rows_materialized` — `trace_rows_sha256 5046c639…` (4m56s) |
| 22:13:20.137 | `tokenizer_bridge_on_policy_preimage_complete` |
| **22:13:20.151** | **`on_policy_extension_evidence_materialized`** — 2 members, 371 batches (§6.2) |
| 22:13:41.180–.268 | the same three global-topology entries looked up and resumed a second time, post-evidence — 3 more hits, still 0 misses |

_Timeline continues in §6–§8._

### 5.1 The run configuration digest moved, and that is expected

```
W13D / W13I    b13987d675d5252cc46fce3407619bfbc38720fe5b4fa9fdccae91b0373c9d0d
W13K (this)    c79cbe9704438328d56d77ae5403ea50d7a3f7c52a28956724fa50dcad6e38fc
```

Same 108 flags, same values — the digest covers the runtime's own configuration
surface, which the twelve new commits widened (four new options, §2.3). The
externally-started resume in the moved-aside directory logged a third value
(`045bd8e1…`) at `468ce5e1`, which is the same effect one commit earlier.

---

## 6. Phase observations

### 6.1 Global topology — REUSED again, 6 hits / 0 misses

The task expected the balanced-basis phase keys to have rotated with the new
commits, with the global-topology entries *maybe* still hitting. Measured:

```
22:07:34.030  lookup  lfm2-global-capacity-union-v1      a5eaf8da157297c60403c661a66e6a6d0277c3bc4e6d723467df85c9147a6e43
22:07:34.043  HIT     resumed=true  payload_bytes=326983  completion ec925599…
22:07:34.065  lookup  lfm2-global-functional-width-v1    9c9d568fb71cb37efd530bb64b9cb9831e99a815eb7d704cf97fe9e2d310552a
22:07:34.066  HIT     resumed=true  payload_bytes=2946    completion ca043086…
22:07:34.120  lookup  lfm2-global-frontier-selection-v1  edea7406a4b4d3123fcd119930dc9454e474d5c87fd50f5fd9e1c9e9be40d201
22:07:34.122  HIT     resumed=true  payload_bytes=5786    completion ecec1dbd…
   … and the same three looked up and resumed again after the evidence set
     opened, at 22:13:41.180/.217/.266 -> hits at .181/.219/.268
```

Six lookups, **six hits, zero misses**, ~92 ms for the first triple. The three
phase keys are character-for-character the ones `NOTES-W13F` §4.5 recorded W13D
publishing and `NOTES-W13I` §4.1 recorded being reused — so they survived twelve
further commits *and* a moved `configuration_sha256` (§5.1). `NOTES-W13G` §4's
reading is confirmed a second time: these keys bind the **evidence identity**,
not the runtime's configuration surface. W13D paid **38m06s** to compute them
cold; this run paid milliseconds, twice.

The reconstruction reproduces every prior wave exactly: `selected_hidden_size
816`, `frontier_sha256 8a4a1ac8e0959f1c55ce81e7f0f8efba208a1c16fb9fdb95fee986fa4077cdc7`.

**Nothing was pinned.** No `--balanced-basis-existing-phase-key-sha256` is
passed (§2.2); every key above was resolved by content.

### 6.2 The evidence set opened — identical identity, three waves running

22:13:20.151Z, `on_policy_extension_evidence_materialized`:

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

Cache key, identity digest and merged-row digest reproduce `NOTES-W13D` §6.2 and
`NOTES-W13I` §5 character-for-character. The task's "73,989 rows" is the same
corpus: `base_path_count 67,583` + `extension_path_count 6,406` = 73,989 paths
over 371 merged batches, and the row digest `5046c639…` is byte-identical to
W13I's.

### 6.3 The merged balanced basis was a phase-cache HIT, not a cold rebuild — and the entry is the peer's

This is the one place where what happened differs from what the task expected.

```
22:17:47.764  balanced_basis_merged_evidence_bound
                merged_sealed_evidence_sha256      d4518f0512db27719185ec7e9d2f5ad9cfa8944caae0e410b6b63a688028ccc3
                evidence_set_cache_key             f62110d7…   on_policy_evidence_identity 75217282…
                member_count 2   base_batch_count 256   round_count 1
                all_extension_rows_are_fit_rows true   parent_forward_calls 0
22:17:53.416  balanced_basis_live_graph_authenticated
22:17:54.905  balanced_basis_phase_cache_lookup    52cbe3fd96650d284b2e2a03395515c3fb467ce846a014712f5d5c1a4e8f683d
22:18:11.715  balanced_basis_merged_evidence_bound   (re-authenticated, identical composite)
22:18:11.723  balanced_basis_live_graph_authenticated
22:18:13.173  balanced_basis_phase_cache_HIT       52cbe3fd…
                completion_sha256 975e828634a3be4e9bfd3df2e4bb23f21e86e3f9ba606262bd52507bd2d20ac3
                parent_forward_calls 0   trace_tensor_loads 0   training_steps 0
```

* The key **did** rotate as predicted (`6e423f93…` at `8a4ed56f` -> `52cbe3fd…`
  here). The entry's own `complete.json` shows why:
  `numeric_contract.basis_config.adapter_source_sha256 = c2797d77…`, which is
  exactly this deployment's `hf_lfm2_dense_budget_model_fold.py` digest (§1) —
  that file moved in this bump, so every balanced-basis key moved with it.
* The rotated key nevertheless **hit**, because the entry already existed. It
  was published at **2026-08-05T21:26Z, 35 minutes before this launch, by the
  peer's `a0025-…-w13k-cold` run** — whose own
  `balanced-basis-phase-events.jsonl` records `miss -> begin -> complete` on the
  same key `52cbe3fd…` with the same `completion_sha256 975e8286…`,
  `payload_tree_sha256 f0d52a5a…`, `total_bytes 85,984,075`.
* The entry is bound to **this run's evidence**, not merely to a similar one.
  Its `phase_key.dependency_sha256s` name
  `on_policy_evidence_identity = 75217282…`,
  `merged_sealed_evidence = d4518f05…` and `trace_cache = f62110d7…` — the three
  digests this run computed independently before the lookup.

So the reuse is content-addressed, HMAC-chained
(`balanced_basis_phase_cache_runtime_verified` at 22:01:45, one `hit` event
appended to this run's own event log with `previous_event_sha256 000…0`), and
**nothing was pinned to force it** — the launcher passes no phase key. But it
does mean **this run did not rebuild the merged basis**; it resumed the peer's.
Cost: **26 s** from `merged_evidence_bound` to `hit`, against W13I's 11m30s cold
build.

Recorded plainly because it matters for provenance: if the r13 arm ever needs a
basis it built itself end to end, that is not what this launch would have
produced.

### 6.4 What the twelve new commits actually bought — measured

| phase | W13I (`8a4ed56f`) | W13K (`6f178f14`) |
|---|---|---|
| import -> `parent_verified` | 17 s | **7.4 s** |
| carrier verification | 5m31s | 5m48s |
| global topology | 0.08 s (3 hits) | 0.09 s (3 hits, then 3 more) |
| parent row materialization | 5m13s | 4m56s |
| bridge on-policy preimage | 48 s | 50 s |
| merged balanced basis | 11m30s (cold build) | **26 s (phase hit, §6.3)** |
| basis install + structural pass, 14 layers | 25 s (structural only) | 1m11s (`phase_cache_hit` 22:18:13.173 -> `dense_structural_fold_completed` 22:19:24.710; no separate structural-start event was emitted, so this interval covers both) |
| layer 0 replay sharding | 70.2 s @ 1.981x | 73.4 s (22:20:29 -> 22:21:42) |
| **layer 0 `pre_producer` capture** | **39m47s** | **2m23s** |

The capture collapse is the headline: `child_closed_loop_layer_capture_started`
22:19:39.730 -> `child_closed_loop_layer_capture_completed` 22:22:02.849, 205
records, `capture_sha256 23ca0961…`. That is almost certainly `7ac864f5`
(*Avoid quadratic bridge verification during replay pairing*) landing at
production scale — W13I spent ~40 minutes per capture point and this spent ~2.

**Caveat:** one capture point on one layer. The run died before any producer
solve completed, so this measures the capture, not a layer. Nobody has yet seen
a layer complete under `6f178f14`.

---

## 7. RSS

Two independent measurements, per `NOTES-W13I` §7's method — and this time the
15-second sampler read **higher** than the runner's own stamp, because the run
died between stamped events:

```
runner peak (process_peak_rss_bytes)   40,899,932,160 B = 38.09 GiB   (stamped at the layer-0 capture)
sampler peak (15 s interval)           43.78 GiB over 97 samples
runner swap high-water                 0 bytes, every event
```

Growth curve from the run's own instrumentation:

| point | peak RSS |
|---|---|
| `dense_closed_loop_started` (22:19:25) | 34,101,673,984 B = **31.76 GiB** |
| `child_closed_loop_layer_started` layer 0 (22:19:39) | 31.76 GiB |
| layer-0 `pre_producer` capture completed (22:22:02) | 40,899,932,160 B = **38.09 GiB** |
| last live sample before death (22:26:41) | 45,902,028 kB = **43.78 GiB**, VSZ 195 GiB |

Comparable to W13I at the same point (32.71 GiB through the basis) and far below
its 68.63 GiB layer-0 validation peak. **Host memory was never a constraint**:
251 GiB total, 187–192 GiB available throughout, and the run's own
`process_swap_bytes` was 0 at every event. The host's 7.1/8.0 GiB of swap-in-use
is a pre-existing condition from other tenants (three ComfyUI instances holding
~42 GiB RSS) and was unchanged before, during and after.

RSS at each poll, as recorded:

| UTC | run RSS | note |
|---|---|---|
| 22:02:59 | 1.59 GiB | 42 events, `parent_verified` |
| 22:23:26 | 43.60 GiB | 512 events, layer-0 capture done, CPU 271%, 87 threads |
| 22:26:41 | 43.78 GiB | last live sample |
| 22:26:56 | 0 (`Zl`) | zombie |
| 22:27:11 | — | reaped |

---

## 8. The death, and what it was not

### 8.1 What the log shows

`run.log` ends at **22:22:02.849Z**, sequence 512, mid-run:

```
stage         child_closed_loop_layer_capture_completed
capture_point pre_producer     layer_index 0     record_count 205
capture_sha256 23ca0961c8f4a7d904dc0e3c2251171536127066e09ff89663a3fa448ddfa292
process_peak_rss_bytes 40899932160    process_swap_bytes 0
```

Then 4m39s of normal work with no events (the producer solve emits none), RSS
climbing smoothly 45.75 -> 45.90 GB across five 15-second samples, and then
nothing.

```
"stage":"failure" count   0
Traceback count           0
last log write            22:22:02
process state 22:26:41    Rl, 43.78 GiB, etime 25:05
process state 22:26:56    Zl, 0 KB   (zombie)
process state 22:27:11    gone
```

`stdout` and `stderr` were both redirected into `run.log` by the launcher
(`> "$RUN/run.log" 2>&1`) and Python ran with `-u`, so a Python exception, a
`MemoryError`, a C++ `abort()` message or a CUDA error **would have been
written**. Nothing was.

### 8.2 Ruled out, each with the evidence

| hypothesis | evidence against |
|---|---|
| **Session / process-group teardown** (the task's prior) | PPID `1` (systemd), `pgrp == sess == 3454853` (its own setsid session), `tty_nr 0`, launcher shell long gone — all measured from `/proc` at 22:01:49 (§4). The scope it lived in, `session-10810.scope`, is logged `Deactivated successfully … Consumed 1h 5min 36.672s CPU time` at 22:26:59 — **three seconds after** the process was already a zombie, i.e. the scope emptied *because* the process died, not the reverse. |
| **Kernel OOM killer** | `dmesg -T` has no entry of any kind after 21:04Z; no `Killed process`, no `oom-kill`. 187–192 GiB was available throughout. |
| **`systemd-oomd`** | active and enabled, but `journalctl -u systemd-oomd --since -8h` -> **"No entries"**, and no `Killed … due to memory pressure` line anywhere in the journal for the last 8 h. The run lived in `session-10810.scope` under `user-1000.slice`, whose `ManagedOOMSwap`/`ManagedOOMMemoryPressure` are `auto`; only `user@1000.service` carries `kill`, and the run was not in it. |
| **Crash signal from the kernel** (SIGSEGV/SIGBUS/SIGABRT) | `/var/crash/` is empty; `core_pattern` is apport and apport logged nothing; `dmesg` records no `segfault`/`traps` line for the pid. |
| **Fail-closed refusal** | 0 `"stage":"failure"` events, 0 tracebacks, and the last event is a *successful* capture. No receipt and no `diagnostics.json` were written, so **there is no receipt sha to quote**: the run never reached seal. |
| **Disk exhaustion** | 340 GiB free before, 335+ GiB after. |
| **The peer resetting the shared deployment underneath it** | checked directly: `neural-forge-postpub-e893767a` is still at `6f178f14`, `git status --porcelain` still empty, `hf_lfm2_circuit_trace_cache.py` still `16271526…`, and the deployment reflog's newest entry is still this wave's 21:57Z reset. The peer built a *separate* worktree (`deployments/neural-forge-b65c81f5`) instead. |

### 8.3 What remains

A silent disappearance with no kernel record and no userspace output is the
signature of **an external `SIGKILL` (or a `SIGTERM` with no handler) from
another process** — which Linux does not log anywhere by default. That is now
the fifth such death in one day:

| run | commit | died | log at death |
|---|---|---|---|
| `a0023-…-round-0-refold` (W13I) | `8a4ed56f` | ~2026-08-05T20:2xZ after ~8 h | 545,288 B, 2 accepted layers |
| its terminal-only resume | `468ce5e1` | after ~5 min | 16,969 B |
| `a0024-…-w13j-cold` (peer) | — | minutes | 5,595 B |
| `a0025-…-w13k-cold` (peer) | — | ~21:27Z | published the basis, then stopped |
| **`a0023-…-round-0-refold` (W13K, this)** | **`6f178f14`** | **22:26:41–56Z, 25 min in** | **405,927 B, 512 events** |

What is new here is the **negative** result: this one was provably detached, so
"it was launched attached to an interactive session" — the working hypothesis
this whole attempt was built around — **cannot explain this death**. Whatever is
doing it reaches processes that are session-detached, group-detached, tty-less
and reparented to pid 1.

Peer activity in the same minutes, from file timestamps only (the peer's
directories were read, never modified):

```
22:23Z  /home/cereal/nf-w13l/deploy-b65c81f5.sh + bundle written; a b65c81f5
        worktree deployment created from the shared .git of the postpub checkout
22:26:41-56Z   THIS RUN DIES
22:27Z  /home/cereal/nf-w13l/nf-launch-b65c81f5-w13l.sh, remote-test-b65c81f5.sh
22:29Z  peer pytest matrix starts (228 passed, 1 skipped, 137 s)
22:30Z  peer copy of nf-launch-r12-round-refold.sh (e652f576…) written
22:33Z  peer launches a0026-…-w13l-cold, pid 3551402
```

No peer script that can be read contains `kill`, `pkill` or a process sweep;
their deploy uses `git worktree add` and does not touch this deployment's HEAD or
working tree. The correlation is **timing only**, and is stated as timing only.

### 8.4 The instrument that will settle it

`/home/cereal/nf-w13k/killspy.bt` is now running as root, detached, bounded to
6 hours (`timeout 6h`), logging to `/home/cereal/nf-w13k/killspy.log`. It taps
`tracepoint:signal:signal_generate` and prints, for every
SIGKILL/SIGTERM/SIGINT/SIGHUP/SIGABRT the kernel generates, the **sender's** pid
and comm alongside the target's:

```
2026-08-05T16:39:04Z sig=15 sender_pid=3568908 sender_comm=bash -> target_pid=3568909 target_comm=sleep
```

That line is its **self-test** — a throwaway `sleep` was SIGTERMed on purpose and
the tracer named the sender. It is watching the peer's `a0026` run right now; if
that run dies the same way, the killer is named in `killspy.log`.

Two caveats on reading it: the printed timestamps are **server-local (UTC-6)**
despite the trailing `Z` in the format string, and it observes only signals — a
process that exits of its own accord leaves no line, which would itself be
diagnostic.

---

## 9. Current state, and why nothing was relaunched into it

At 22:39Z:

* **Deployment** `neural-forge-postpub-e893767a`: `6f178f14`, tree clean,
  collector frozen. Untouched by the peer's worktree.
* **Launcher**: `e496c102…`, the verified v2-ON r13 arm, unmodified.
* **Refold run path**: **clear**. This wave's dead partial was moved, not
  deleted, to
  `…/a0023-exact4x-h816-mgv3-r12-round-0-refold.killed-external-223936-w13k`
  (808,284,726 bytes before and after the move, 405,927-byte `run.log`, 512
  events, 771 MB of snapshots, the phase-event log with the `hit` record).
  W13I's earlier partial remains at `…killed-external-220056`.
* **Peer run `a0026-…-w13l-cold`** is live: pid `3551402`, started 22:33Z, from
  `deployments/neural-forge-b65c81f5` at `b65c81f5`, using **cuda:0 and cuda:4**
  as its replay pair — the same two GPUs this arm's launcher pins — and with
  `--balanced-basis-existing-phase-key-sha256 52cbe3fd…` **pinned** in its own
  launcher copy (`e652f576…`).

**No relaunch was attempted.** Two runs of this arm on one host would double-book
`cuda:0,cuda:4` and, more seriously, stack two closed-loop folds whose sibling
r12 peaked near 138 GB against 187 GiB of available RAM. The realistic failure
mode is a genuine kernel OOM that destroys **both** runs' evidence, including the
peer's. That is not a trade this wave is entitled to make unilaterally, and the
resource claim is a coordination artifact, not a lock. The box is primed instead:
every check in §§1–3 is green and the relaunch is one command (§10).

---

## 10. Relaunch and monitoring recipe

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
```

**Relaunch, once cuda:0/cuda:4 are free** — nothing else needs redoing; the
deployment, the launcher and the clear run path are all verified above:

```bash
cd /home/cereal/model-surgery-lab/deployments
export SOURCE_COMMIT=6f178f14add54ec882c09ad183f69a626a0cfb48
setsid nohup bash ./nf-launch-r12-round-refold.sh refold \
    > /home/cereal/nf-w13k/launch-refold.out 2>&1 < /dev/null &
disown
```

Then **prove the detachment before trusting it** — the check that made this
wave's negative result usable:

```
P=$(cat $RUN/run.pid)
ps -o pid=,ppid=,pgid=,sess=,tty=,stat= -p $P
awk '{print "ppid="$4" pgrp="$5" session="$6" tty="$7}' /proc/$P/stat
# want: ppid=1, tty=0, pgrp == session == the setsid session, launcher shell gone
```

Helpers, all installed under `/home/cereal/nf-w13k/`, all self-exiting when the
run pid does:

```bash
$SSH --command-timeout 900 --execute "bash /home/cereal/nf-w13k/poll.sh"      # one-shot full poll
$SSH --execute "cut -c1-200 /home/cereal/nf-w13k/milestones.log"              # decision lines only
$SSH --execute "python3 /home/cereal/nf-w13k/timeline.py $RUN/run.log"        # stage timeline
$SSH --execute "python3 /home/cereal/nf-w13k/timeline.py $RUN/run.log balanced_basis"   # + field detail
$SSH --execute "bash /home/cereal/nf-w13k/waitfor.sh '<stage regex>' 780"     # block until a stage appears
$SSH --execute "cat /home/cereal/nf-w13k/killspy.log"                         # who sent the signal (§8.4)
```

`rss-sampler.sh` and `milestones.sh` must be restarted after a relaunch — they
bind to the pid in `run.pid` when they start:

```
setsid nohup bash /home/cereal/nf-w13k/rss-sampler.sh > /home/cereal/nf-w13k/rss-sampler.log 2>&1 < /dev/null &
setsid nohup bash /home/cereal/nf-w13k/milestones.sh > /home/cereal/nf-w13k/milestones-runner.log 2>&1 < /dev/null &
```

Note `--command-timeout`: the client's default is 300 s, so any wait longer than
that must raise it explicitly.

### 10.1 What to expect, with this commit's measured pacing

| phase | expect |
|---|---|
| launch -> `parent_verified` | ~8 s |
| -> `carrier_verified` | ~6 min |
| global topology | 3 hits, ~0.1 s (then 3 more after the evidence set) |
| -> `on_policy_extension_evidence_materialized` | ~11.5 min, 2 members, 371 batches |
| merged balanced basis | **~30 s if `52cbe3fd…` is still cached**, ~12 min if rebuilt |
| basis install + structural pass, 14 layers | ~1m11s |
| layer 0 `pre_producer` capture | **~2.5 min** (not W13I's 40 min) |
| layer 0 producer solve | unmeasured at this commit — W13I paid ~17 min |
| **layer 0 decision** | unmeasured at this commit; W13I paid 3h04m for the whole layer with 40-minute captures, so expect materially less |

`validation_accepted` per layer remains the **only** live decision surface;
`NOTES-W13I` §8.2 still governs — the floor, boundary and margin numbers are
sealed into `diagnostics.json` at the end of the run and do not stream. The r13
arm's headline question (`refused_transition_indices`) is still unanswered, four
attempts in.

---

## 11. Caveats

* **The run did not fail; it was terminated.** There is no error, no receipt and
  no `diagnostics.json` to quote, because nothing failed closed. Do not read this
  wave as evidence against the r13 arm.
* **The detachment hypothesis is dead, and its replacement is not proven.** §8.2
  eliminates session teardown, OOM, oomd and crashes; what is left is an external
  signal from an unlogged sender. The correlation with the peer's W13L deployment
  window is **timing only** — no readable peer script kills anything, and the peer
  did not touch this deployment. `killspy.log` is the instrument that can convert
  suspicion into a named sender; until it fires, the cause is **unknown**.
* **The merged balanced basis was inherited, not built** (§6.3). Content-addressed
  and authenticated, nothing pinned — but it is the peer's `a0025` payload, and a
  future claim that this arm built its own basis would be false for this launch.
* **The 2m23s capture is one capture point, not a layer.** No producer solve
  finished under `6f178f14`. Anyone quoting a per-layer cost from this file is
  quoting a number that does not exist yet.
* **`heldout_extension_batch_count` was not observed** — the run died before the
  row-selection split and before any seal, so even W13I's arithmetic evidence
  (93 of 256 base batches) is absent here.
* **Nothing was weakened.** No flag relaxed, no fail-closed clause bypassed, no
  phase key pinned, no retry attempted, and the three new relaxation levers at
  this commit (`--dense-closed-loop-all-layer-finite-numeric-checks-diagnostic-only`,
  `--dense-closed-loop-intermediate-numeric-checks-diagnostic-only`,
  `--dense-terminal-only-restart-resume`) were verified present in `--help` and
  **absent** from the launcher.
* **Read-only artifacts untouched**: the sealed r12 run directory, `round-000`,
  the sealed 2026-07-17 pack and both base caches were never written to. The
  peer's `a0024`, `a0025` and `a0026` run directories and `/home/cereal/nf-w13l/`
  were **read** for the forensics in §8 and **never modified**. Both preserved
  W13D/W13F failures and both `killed-external` partials remain in place.
* **Local HEAD moved during the wave.** `codex/lrc-blank-compression` advanced
  `6f178f14 -> b65c81f5` (*Cache invariant folding group plans*, touching
  `hf_lfm2_structural_residual_attention_fold.py`) at ~22:20Z while this run was
  live. The deployment stayed pinned at `6f178f14` deliberately; the frozen
  collector is byte-identical at both commits.
* **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict.
* **Not committed.** No `git add`, no commit, no push.

---

## 12. Artifacts

Server, under `/home/cereal/nf-w13k/`:

| file | what it is |
|---|---|
| `nf-w13k.bundle` | the deployment bundle, `c38785c5…`, 217,747 B |
| `HEAD-before.txt`, `status-before.txt` | pre-reset deployment state (`8a4ed56f…`, clean / 0 bytes) |
| `runner-help-6f178f14.txt`, `help-flags.txt`, `used-flags.txt` | the 181-option `--help` and the 108/108 flag reconciliation |
| `flagcheck.sh`, `flagcheck2.sh`, `flagcheck2.out` | the flag audits (§§2.1–2.3). `flagcheck.sh`'s `--help` section is invalid — it omits `PYTHONPATH` (§1.1) — and is superseded by `flagcheck2.sh` |
| `phase-cache-before.txt` | phase-cache inventory taken before launch |
| `launch-start.txt`, `launch-refold.out` | launch timestamp and the launcher stage's own preflight stdout |
| `poll.sh`, `timeline.py`, `waitfor.sh`, `entryinfo.py` | the monitoring toolkit (§10) |
| `rss-sampler.sh`, `rss-samples.tsv` | the 15-second RSS sampler and its 97 samples, including the `Zl` transition |
| `milestones.sh`, `milestones.log` | the decision-event recorder |
| `killspy.bt`, `killspy.log` | the signal tracer and its log, running as root until ~04:39Z (§8.4) |

Preserved run directories under
`…/neural_forge_exact4x_20260730/catalog-integrated-carrier/` — none deleted:

| directory | |
|---|---|
| `…-round-0-refold.failed-merged-basis-104326` | W13D's fail-closed run |
| `…-round-0-refold.failed-config-streaming-v2-104727` | W13F's fail-closed run |
| `…-round-0-refold.killed-external-220056` | W13I's 8-hour run (2 accepted layers) + the `468ce5e1` resume attempt |
| `…-round-0-refold.killed-external-223936-w13k` | **this wave's run**, 512 events, died 22:26:5xZ |
| `a0024-…-w13j-cold`, `a0025-…-w13k-cold`, `a0026-…-w13l-cold` | peer runs — read for forensics, never modified |

Phase cache: **no new entry was published by this run.** It appended one `hit`
event to its own event log and resumed
`…/lfm2-circuit-balanced-residual-basis-v1/…/52cbe3fd…` (85,984,075 B, published
by the peer's a0025 at 21:26Z). The three global-topology entries were resumed,
not rewritten. `round-000`, the sealed r12 run directory, the sealed 2026-07-17
pack and both base caches: untouched.
