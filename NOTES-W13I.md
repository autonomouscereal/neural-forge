# NOTES-W13I — r13 refold, attempt #3: the deadlock is gone and the run is up

Wave 13I, branch `codex/lrc-blank-compression`. This deploys `8a4ed56f`
(*Authenticate merged evidence in margin-v2 and keep held-out child-free*),
restores `nf-launch-r12-round-refold.sh` to its **original v2-ON form**, and
relaunches the refold that `NOTES-W13D` and `NOTES-W13F` each failed closed.

**Status: RUNNING HEALTHY, past layer 0.** Launched 2026-08-05T12:38:23.857Z,
PID `1745945`. It cleared clause A in 17 seconds (the exact wall `NOTES-W13F`
died on), opened the 2-member / 73,989-row evidence set, **reused all three
global-topology phase entries** (79 ms against W13D's 38 minutes), did the thing
no run has ever done — **built and published a balanced basis from merged
base+extension evidence with streaming margin v2 on**, the phase `NOTES-W13D`
died entering and `NOTES-W13F` proved unreachable in either margin setting — and
then **completed closed-loop layer 0 with `validation_accepted: true`**.

Zero failures in 3h38m. RSS peak **68.63 GiB**, swap 0.

The one thing this wave still cannot answer: whether the reclosure acceptance
gate refuses r12's layer-0 writer blow-up. That number is sealed at the end of
the run, not streamed (§8.2).

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.
Run directory
`/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold`.

---

## 1. Deployment bump: 84acded2 → 8a4ed56f

```
bundle  84acded28acabcac1603fb7a46b0e85194d8f192..codex/lrc-blank-compression
        (2 commits; a8b65e15, 8a4ed56f)
size    62,039 bytes
sha256  643e92c37ac0c6ed127c74a391bbd67c37e9fff18ccbad475d7296e6223731f6   (identical both sides)
server  /home/cereal/nf-w13i/nf-w13i.bundle
```

`git bundle verify` clean on both sides, then
`git fetch <bundle> refs/heads/codex/lrc-blank-compression:refs/heads/w13i-import`
and `git reset --hard 8a4ed56f…`. Pre-reset state recorded in
`/home/cereal/nf-w13i/HEAD-before.txt` (`84acded2…`) and `status-before.txt`
(**0 bytes** — the tree was clean, nothing was overwritten).

Deployment `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`:

| | |
|---|---|
| `git rev-parse HEAD` | `8a4ed56f46dc74b6de6d3278687ca966c7351201` |
| `git status --porcelain` | **empty** |

Deployed file digests, compared against the local committed blobs at both
`84acded2` and `8a4ed56f` (`git cat-file blob`, not assumed from commit titles):

| file | deployed sha256 | vs `84acded2` |
|---|---|---|
| `hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | **SAME — frozen collector holds** |
| `hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | **SAME** |
| `hf_lfm2_streaming_vocabulary_margin_adapter.py` | `2a949a3076af23b829b31af2410d245f2d3f25355472a13740f3dd7879e051d6` | DIFF (W13G §2) |
| `hf_lfm2_balanced_circuit_basis.py` | `e1091d33fe03c95d338d4d967f1750e3d47f21a0223eaba367c317abb55c4108` | DIFF (W13G §3.2) |
| `experiments/run_lfm2_dynamic_catalog.py` | `0232c8a1e3e8a4f4a9c7ffb1de3067b3bbca27a3c219b27106c6fd7da06b423d` | DIFF (W13G §5.1) |
| `hf_lfm2_balanced_basis_phase_cache.py` | `6bb971d558a573a5d1dba9dcaf11220b8cc8294d457258fc26cbcf08435b872f` | SAME |
| `hf_lfm2_sealed_evidence_set.py` | `8e6e608908e6eaee384cb54f52ab064bc0f3806621d89d8239b2d84683b2358c` | SAME |
| `hf_lfm2_dense_budget_model_fold.py` | `ec9cc29d1800737f1aadb8cdc363c7fb4534bcfa1cbac0db3622c4c3c7596117` | SAME |
| `dynamic_closed_loop_fold.py` | `ab357dba7d52d59187dc1f4c0481f6d57d72a670d4c5c366880295b9474f7b09` | SAME |
| `experiments/run_r12_on_policy_round.py` | `ad5e72a54880360af3887a4d4641968d6dc9fe7a5299d69000343ca7dcc000ec` | SAME |

**The frozen-collector rule holds.** `16271526…` is byte-identical at
`13f71132`, `53b353f0`, `84acded2` and now `8a4ed56f`; `hf_lfm2_canonical.py`
likewise (`789c6a6f…`). Exactly the three W13G source files moved, and each
lands on the digest `NOTES-W13G` §6 recorded — the deployment carries W13G's
work byte for byte.

`git diff --name-only 84acded2 8a4ed56f` is 13 files: those three, five
native-lane test files and `test_hf_lfm2_*`/`test_run_lfm2_*` test files from
`a8b65e15` and `8a4ed56f`, plus `NOTES-W13G.md` and `NOTES-W13H.md`. No other
runtime source moved.

Codex's uncommitted local diffs to `hf_lfm2_dense_budget_model_fold.py` and
`dynamic_closed_loop_fold.py` are **not** deployed — the deployment carries the
committed blobs, which is why both show SAME above.

---

## 2. Launcher restored to the v2-ON form

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`

```
replaced   24c61cb32a5738eb13598ee658769dc64916df4445cacd49763212f4f5710633  (W13F's v2-OFF form)
backup     nf-launch-r12-round-refold.sh.bak-v2off-20260805T123653Z          (also /home/cereal/nf-w13i/…preinstall-v2off)
installed  e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda  (the W13D-composed v2-ON form)
```

Installed by copying `nf-launch-r12-round-refold.sh.bak-20260805T104136Z`
verbatim; `cmp` confirms the installed file is **byte-identical** to that
backup, and its sha256 is the expected `e496c102…`. `bash -n` clean.

`diff` of the replaced v2-OFF form against the installed v2-ON form shows
**exactly two changes and nothing else**: W13F's 51-line header block (which
documented the now-resolved mutual exclusion) is gone, and the two margin flags
are back:

```diff
     --dense-vocabulary-margin-observability \
+    --dense-streaming-vocabulary-margin-v2 \
+    --dense-streaming-vocabulary-margin-pair-chunk-size 256 \
     --sample-last-tokens 256 \
```

### 2.1 The r13 arm flags, re-verified on the installed file

Every item the task named, read off the installed launcher (line numbers are in
the `refold)` stage, launcher lines 373–489):

| requirement | flag as installed |
|---|---|
| margin observability | `--dense-vocabulary-margin-observability` |
| streaming margin v2 | `--dense-streaming-vocabulary-margin-v2` |
| margin pair chunk 256 | `--dense-streaming-vocabulary-margin-pair-chunk-size 256` |
| writer-reclosure-acceptance | `--dense-moving-gauge-writer-reclosure-acceptance require_boundary_improvement` |
| boundary regression tolerance | `--dense-moving-gauge-boundary-regression-tolerance 1e-6` |
| terminal floor | `--dense-closed-loop-terminal-composition-floor` + `--dense-closed-loop-terminal-regression-tolerance 0.25` + `--dense-closed-loop-terminal-absolute-floor-minimum-layer 0` |
| terminal tripwire (advisory) | `--dense-closed-loop-terminal-free-run-tripwire` |
| telemetry | `--dense-closed-loop-solver-conditioning-diagnostics`, `--dense-closed-loop-gate-operating-point-diagnostics`, `--dense-closed-loop-depth-telemetry` |
| EIV multiplier 0, ridge off | `--dense-closed-loop-input-error-multiplier 0`; `--dense-closed-loop-input-error-covariance` **absent** |
| margin row weights off | `--dense-closed-loop-margin-row-weights` **absent** |
| fingerprint install verification | `--dense-moving-gauge-install-verification fingerprint` |
| moving gauge v3 | `--dense-body-reclosure-mode moving_gauge_v3` |
| sharded replay pair | `--dense-replay-devices cuda:0,cuda:4` |
| merged captures | `--dense-closed-loop-merge-adjacent-layer-captures` |
| canonical prefix replay | `--dense-closed-loop-replay-mode prefix_by_prefix_canonical` |
| phase caches ON | `--derived-global-topology-cache-root`, `--balanced-basis-phase-cache-root`, `--balanced-basis-phase-event-log`, `--balanced-basis-phase-hmac-key-file`, `--balanced-basis-phase-hmac-key-id` |
| **no pinned phase key** | `--balanced-basis-existing-phase-key-sha256` appears **only** at launcher line 82, inside a comment stating that no phase key is pinned. It is not passed. |

The four "must be absent" spellings were checked by name:
`--dense-closed-loop-input-error-covariance`,
`--dense-closed-loop-margin-row-weights`,
`--balanced-basis-existing-phase-key-sha256` (comment only, see above) and
`--dense-closed-loop-terminal-free-run-stop`.

### 2.2 Flag/`--help` reconciliation at the new commit

`run_lfm2_dynamic_catalog.py` moved in this bump, so the check was redone rather
than inherited:

```
deployed --help at 8a4ed56f    178 options parsed   (/home/cereal/nf-w13i/runner-help-8a4ed56f.txt)
refold stage passes            108 distinct flags
used flags absent from --help  0
```

The option set is **identical** to `84acded2`'s: `comm` between the two parsed
lists is empty in both directions. W13F's "179" counted a bare `--` token as an
option; the real difference is zero. The 106 → 108 flag delta is exactly the two
restored margin flags, and nothing else.

---

## 3. Relaunch timeline (UTC, 2026-08-05)

| | |
|---|---|
| 12:3x | bundle built (`643e92c3…`), uploaded, `git bundle verify` clean; deployment reset to `8a4ed56f`, tree clean, digests checked |
| 12:36:53 | v2-OFF launcher backed up (`…bak-v2off-20260805T123653Z`, `24c61cb3…`); v2-ON form installed (`e496c102…`), `bash -n` clean, `cmp` identical |
| 12:37–12:38 | 108/108 flags confirmed against the deployed `--help`; phase-cache roots inventoried (`phase-cache-before.txt`, 21 paths) |
| **12:38:23.857** | **`refold` stage launched**, `SOURCE_COMMIT=8a4ed56f46dc74b6de6d3278687ca966c7351201` |
| 12:38:24.067 | launcher returned 0; preflight OK — both caches, both rebound envelopes, round preimage; identical to W13D §6.1 |
| 12:38:33.433 | `balanced_basis_phase_cache_runtime_verified` |
| **12:38:40.914** | **`parent_verified`** — 229,693,184 unique parameters, bf16. **Clause A cleared at ~17 s**, the wall W13F died on at 10.1 s |
| 12:44:11.736 | `carrier_verified` — 57,420,144 counted elements, 2 independent routes |
| **12:44:11.907–.986** | **three `global_topology_phase_cache_hit`, 0 misses** (§4) |
| 12:44:12.253 | `global_topology_selected_width_reconstructed` — `selected_hidden_size 816`, `frontier_sha256 8a4a1ac8…` |
| 12:44:12.254 | `global_frontier_preplanned` |
| 12:49:24.952 | `parent_rows_materialized` — 73,989 paths, 27,729,893,376 retained bytes, `trace_rows_sha256 5046c639…` |
| 12:50:13.013 | `tokenizer_bridge_on_policy_preimage_complete` — 371 pairs, 115 extension records, 0 parent forwards |
| **12:50:13.027** | **`on_policy_extension_evidence_materialized`** — 2 members, 73,989 rows (§5) |
| 12:50:46.620 | `wiring_planned` — `planned_target_abi_sha256 f04b6fda…` |
| 12:50:47.642 | `system_fold_started` |
| **12:54:34.421** | **`balanced_basis_merged_evidence_bound`** — the W13G fix firing at production scale (§6) |
| 12:54:40.181 | `balanced_basis_live_graph_authenticated` — 133 parameter specs, 0 trainable, topology exactly preserved |
| 12:54:43.000/.010 | `balanced_basis_phase_cache_lookup` / **`_miss`**, key `6e423f93…` (§4.2) |
| 12:55:05 → 12:58:29 | row-sliced selection: **fit 2,481 units, held-out 837 units** (§7) |
| 13:05:43.096, 13:06:02.641 | two further `balanced_basis_merged_evidence_bound` re-authentications, same composite |
| **13:06:04.205** | **`balanced_basis_phase_cache_published`** — `cold_reload_exact: true`, 85,984,075 bytes |
| 13:06:46.436 → 13:07:11.489 | structural pass, 14 layers in **25 s** — **past everything W13D and W13F reached** |
| 13:07:21.398 | `dense_restart_capsule_writer_bound` — capsule root bound, `rotation_keep 2`, `resumed false` |
| 13:07:22.761 | `dense_closed_loop_started` / `child_closed_loop_compiler_started` — 14 layers selected, `body_reclosure_enabled true`, `merge_adjacent_layer_captures_enabled true` |
| 13:07:22.761 | **`child_closed_loop_layer_started` layer 0** — fit 205 records, held-out 51 |
| 13:09:30.573 | `child_replay_sharding_completed` (pre_producer) — 2 workers on cuda:0+cuda:4, **1.981× observed speedup**, 70.2 s wall |
| 13:47:09.422 | layer 0 `pre_producer` capture completed (39m47s) |
| 14:04:19.835 / 14:04:22.345 | layer 0 producer compiled / installed |
| 14:05:34.171 | replay sharding (post_producer) — **1.988×**, 70.1 s |
| 14:43:31.174 | layer 0 `post_producer` capture completed (39m09s) |
| 14:43:43.012 / 14:43:45.473 | layer 0 consumer compiled / **installed** (`installed_state_sha256 cb966912…`) |
| 15:2x | layer 0 `post_consumer` capture completed |
| 15:3x | layer 0 downstream compiled / installed, merged adjacent-layer capture started |
| **16:11:20.536** | **`child_closed_loop_layer_validation_completed` layer 0 — `validation_accepted: true`** (§8) |
| 16:11:29.174 | `child_closed_loop_layer_completed` layer 0 + `child_closed_loop_restart_capsule_published` (`completed_layer_count 1`) |
| 16:11:29.174 | **`child_closed_loop_layer_started` layer 1** |
| 16:11:29.792 | layer 1 `pre_producer` **`child_closed_loop_layer_capture_served_by_merged_replay`** — no fresh capture, ~40 min saved |

```
PID   1745945          (launched 12:38:23.857Z)
RUN   /home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
log   $RUN/run.log     pid file $RUN/run.pid
```

Run configuration digest `b13987d675d5252cc46fce3407619bfbc38720fe5b4fa9fdccae91b0373c9d0d`
— **byte-identical to `NOTES-W13D` §5's**. The composed r13 arm is the same
configuration W13D launched; only the runtime under it changed.

GPU state: cuda:0 308→1,581 MiB, cuda:4 499 MiB (ComfyUI, coexisting), both at
0% during the CPU-bound evidence and basis phases. The idle r12 pair, reused as
instructed; the P40s and the RTX 8000 untouched.

---

## 4. Phase reuse vs recompute — the observation, recorded, nothing pinned

The task flagged that the balanced-basis implementation fingerprint rotated at
`84acded2`→`8a4ed56f` (`NOTES-W13G` §4) and asked which of W13D's three
global-topology entries survive. **All three were reused.** Measured, not
inferred:

### 4.1 Global topology — REUSED, 3 hits / 0 misses

```
12:44:11.907  lookup  lfm2-global-capacity-union-v1      a5eaf8da157297c60403c661a66e6a6d0277c3bc4e6d723467df85c9147a6e43
12:44:11.918  HIT     resumed=true  payload_bytes=326983  completion_sha256=ec925599…
12:44:11.938  lookup  lfm2-global-functional-width-v1    9c9d568fb71cb37efd530bb64b9cb9831e99a815eb7d704cf97fe9e2d310552a
12:44:11.939  HIT     resumed=true  payload_bytes=2946    completion_sha256=ca043086…
12:44:11.985  lookup  lfm2-global-frontier-selection-v1  edea7406a4b4d3123fcd119930dc9454e474d5c87fd50f5fd9e1c9e9be40d201
12:44:11.986  HIT     resumed=true  payload_bytes=5786    completion_sha256=ecec1dbd…
```

The three phase keys are **character-for-character** the three entries
`NOTES-W13F` §4.5 recorded W13D as publishing. All three resumed; the whole
lookup-and-resume sequence took **79 milliseconds**, against W13D's cold
compute of 08:43:28 → 09:21:34 = **38m06s**. The saving is real and was banked.

The reconstructed result reproduces W13D exactly:
`selected_hidden_size 816`, `frontier_sha256 8a4a1ac8e0959f1c55ce81e7f0f8efba208a1c16fb9fdb95fee986fa4077cdc7`
— the same frontier digest `NOTES-W13D` §6.3 recorded.

So `NOTES-W13G` §4's reading is confirmed on production evidence: the
global-topology phase keys are bound to the **evidence identity**, not to the
two modules whose fingerprint rotated. The task's "MAY no longer match" did not
happen.

**No phase key was pinned, either way.** The launcher passes no
`--balanced-basis-existing-phase-key-sha256`; every key above was resolved by
content from the phase-cache root. That is the r11 lesson `NOTES-W13D` §4 kept,
and it is why the reuse is trustworthy rather than asserted.

### 4.2 Balanced basis — COLD, as expected

```
12:54:43.000  lookup  6e423f9354c485b925fb1759a099499e45f285cb1f9f88592fb5ef5e372ab65c
12:54:43.010  MISS
```

Expected and correct: `_implementation_fingerprint` hashes both files W13G
edited, so every balanced-basis phase key rotates (`NOTES-W13G` §4). The two
pre-existing entries (`7367e3a1…` 2026-07-30, `ce0593c4…` 2026-07-31) are r11/r12
era and were neither read nor written. The run built the basis cold and
published a **new** entry under the rotated key — see §6.

---

## 5. The evidence set opened

12:50:13.027Z, `on_policy_extension_evidence_materialized`:

```
member_count                    2
evidence_set_cache_key          f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
evidence_identity_sha256        75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5
base_batch_count                256      base_path_count       67,583
extension_path_count            6,406    merged_row_count      73,989
merged_batch_count              371      round_count           1
all_extension_rows_are_fit_rows true
merge_member_rows_are_zero_copy_views  true
merge_member_sources_released_as_consumed true
merge_preallocated_once         true
merge_parent_forward_calls      0        parent_forwards       0
merged_trace_rows_sha256        5046c63988800bcc343277b1f34d943f95968a2b3ca004e0144c38f6f5041c61
```

Both the cache key and the identity digest reproduce `NOTES-W13D` §6.2
character-for-character, so the base-cache reconciliation and the rebound
envelopes are still the same evidence identity three waves later.

The merge's own RSS accounting, which is the cheapest independent read on the
memory question:

```
merge_before_process_rss_bytes           2,571,612,160   (2.39 GiB)
merge_after_process_peak_rss_bytes      30,310,998,016  (28.23 GiB)
merge_observed_process_rss_growth_bytes 27,739,385,856  (25.83 GiB)
merge_retained_bytes                    27,729,893,376  (25.83 GiB)
merge_streaming_merge_peak_bytes        27,828,836,352
merge_separate_materialization_peak_bytes 55,459,786,752
merge_peak_bytes_avoided                27,630,950,400  (25.73 GiB)
merge_peak_bytes_ratio_to_retained      1.0035680979605077
merge_maximum_transient_selection_bytes     98,942,976
merge_swap_bytes                                     0
```

The zero-copy merge holds the merged corpus at **1.0036× its retained size**
instead of the 2× a separate materialization would cost.

---

## 6. The merged-evidence balanced basis — the phase that had never run

This is the phase `NOTES-W13D` died entering (`trace_cache must be
Lfm2CircuitTraceCache.`), that `NOTES-W13E` fixed only at toy scale, and that
`NOTES-W13F` proved no configuration could reach. It ran.

### 6.1 `balanced_basis_merged_evidence_bound`, 12:54:34.421Z

```
merged_sealed_evidence_sha256      d4518f0512db27719185ec7e9d2f5ad9cfa8944caae0e410b6b63a688028ccc3
merged_session_sha256              baf0354ec0fd45d19f4f6172b85b56d421f618e5e2b11e824cde2426dedfae96
evidence_set_cache_key             f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
on_policy_evidence_identity_sha256 75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5
base_cache_key                     c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7
member_count 2   batch_count 371   base_batch_count 256   round_count 1
all_extension_rows_are_fit_rows  true
parent_forward_calls             0
```

`merged_sealed_evidence_sha256 d4518f05…` is the `NOTES-W13G` §2.2 composite —
the document that per-member-authenticates every sealed manifest digest and
records `sealed_merged_manifest_digest_exists: false` rather than inventing one.
This is its **first production value**.

It fired **three times** (12:54:34.421, 13:05:43.096, 13:06:02.641) — once at
build, twice more as the gate re-authenticated — and the composite is
**identical at all three**, with `member_count 2`, `batch_count 371` and
`parent_forward_calls 0` each time. That is W13G §2.3's enforcement point
behaving as designed: it recomputes the composite and compares, and it agreed.

`balanced_basis_live_graph_authenticated` (12:54:40.181) alongside it:
`runtime_parameter_spec_count 133`, `runtime_trainable_parameter_spec_count 0`,
`runtime_all_parameters_frozen true`, `structural_topology_exactly_preserved true`.

### 6.2 Published

```
13:06:04.205  balanced_basis_phase_cache_published
  phase_key_sha256   6e423f9354c485b925fb1759a099499e45f285cb1f9f88592fb5ef5e372ab65c
  completion_sha256  0318446985a8896af916e5ade966b1f1ed760899d373973cacd7533c1546616e
  cold_reload_exact  true
```

`$RUN/balanced-basis-phase-events.jsonl` — **non-empty for the first time**
(W13D's was zero bytes because the type check fired before the first event).
Three HMAC-chained events:

```
0  12:54:43.004878Z  miss      6e423f93…  entry_path …/lfm2-circuit-balanced-residual-basis-v1/6e423f93…
1  13:05:44.588163Z  begin     6e423f93…
2  13:05:46.303623Z  complete  6e423f93…  completion_sha256 03184469…
                                          payload_tree_sha256 c20d37a0…  total_bytes 85,984,075
   run_id lfm2-balanced-basis-b13987d675d5252cc46fce3407619bfbc38720fe5b4fa9fdccae91b0373c9d0d
   previous_event_sha256 chain starts at 000…0
```

**Phase wall clock:** `merged_evidence_bound` 12:54:34 → `published` 13:06:04 =
**11m30s**, against r12's ~9m48s cold single-cache build on 256 batches. On 371
batches and 73,989 rows with per-member authentication, that is cheap.

---

## 7. RSS — the number nobody had measured

`/home/cereal/nf-w13i/rss-sampler.sh` sampled `ps -o rss=,vsz=,stat=,etime=` on
the run pid every 15 s from 12:38:45, so the peak is recorded between operator
polls rather than only at them. Samples in
`/home/cereal/nf-w13i/rss-samples.tsv`.

**Two independent measurements, and they disagree — use the runner's.** The
15-second sampler necessarily misses transients; the runner stamps
`process_peak_rss_bytes` into its own events and that is authoritative.

```
sampler peak (15 s interval)             49.23 GiB   through layer 0
runner peak (process_peak_rss_bytes)     73,694,867,456 B = 68.63 GiB   at the layer-0 validation
swap                                     0 bytes, every single event
```

**RSS peak for this run so far: 68.63 GiB.** Reported by the run itself in the
`child_closed_loop_layer_validation_completed` event for layer 0.

The growth curve, all from the run's own instrumentation:

| point | peak RSS |
|---|---|
| before the evidence merge | 2.39 GiB |
| after the merge (retained 73,989-row corpus resident) | 28.23 GiB |
| through the merged balanced-basis build | **32.71 GiB** |
| layer 0 replay sharding resolved / completed | 35.99 → 38.15 GiB |
| layer 0 consumer installed | 50.58 GiB |
| **layer 0 validation (peak so far)** | **68.63 GiB** |

Two separate readings matter here:

1. **The balanced-basis phase — the question `NOTES-W13F` §5 left open — is
   answered: ~4.5 GiB over the resident corpus, not a second copy of it.** The
   28.23 → 32.71 GiB step is the whole cost of building a balanced basis from
   73,989 merged rows with streaming margin v2 on. The ~27.7 GB whole-batch arm
   the coordinator was prepared to pay for did not have to be paid.
2. **The closed-loop body fold, not the basis, is what actually drives RSS
   here** — 32.71 → 68.63 GiB across layer 0 alone. That is the number to watch
   from now on, and it is the one nobody had a figure for either. For scale, r12
   peaked at ~138 GB, so there is headroom, but layer 0 alone has already
   doubled the pre-fold footprint.

The mechanism evidence for (1) is in the
`balanced_basis_row_selection_progress` events (sample from the first and last
of 55):

```
authenticated_row_slice_path                    true
whole_tensor_cpu_clone_preceded_row_selection   false
all_snapshot_blocks_retained                    false
phase_windowed_fp64_retention                   true
payload_released                                true
maximum_materialized_source_rows                1
selected_row_count 42   selected_logical_bytes 86,016   safetensors_get_slice_calls 42
retained_fp64_logical_bytes  ~95 MB per phase buffer
```

`whole_tensor_cpu_clone_preceded_row_selection: false` with
`maximum_materialized_source_rows: 1` is the direct refutation of the ~27.7 GB
whole-batch arm: the build never materialized a batch's full tensor inventory.
Every payload was released (`payload_released: true` on every event).

### 7.1 The role-aware split, at r13 scale

Row selection ran in two passes:

```
split=fit      2,481 units   12:55:05.503 → (fit pass)
split=heldout    837 units   → 12:58:29.538
phases         prefill, decode.000 … decode.007   (9 per batch)
```

`837 / 9 = 93` batches held out, exactly. `NOTES-W13G` §3.2 predicted precisely
this: the held-out pool is narrowed to base members only, and the requested
fraction is preserved where the base can supply it — **"r13: 93 of 256 base."**
93 of 256 is what ran. Zero of the 115 child-generated extension batches entered
the held-out projection gate, which is the defect W13G §3 found and fixed.

**On `heldout_extension_batch_count`:** it does **not** appear in `run.log`. The
only `merged_sealed_evidence*` token in the stream is
`merged_sealed_evidence_sha256`. The enforced fields
(`heldout_extension_batch_count: 0`, `all_extension_rows_are_fit_rows`,
`extension_records_are_never_heldout`, `heldout_pool: "base-members-only"`) live
inside the composite document and the split report, which are sealed into
`receipt.json` / `diagnostics.json` at the end of the run — neither is written
yet. The 93-of-256 split above is the strongest live evidence available for the
same fact, and it is arithmetic, not a restated label.

---

## 8. Current state at handoff — layer 0 ACCEPTED, layer 1 in flight

As of 2026-08-05T16:16:48Z, **3h38m in, zero failures**
(`grep -c '"stage":"failure"'` = 0), 140% CPU, 87 threads.

### 8.1 The first r13 gate evidence ever produced

```
2026-08-05T16:11:20.536Z  child_closed_loop_layer_validation_completed
  layer_index               0
  validation_accepted       true
  validation_decision_sha256 c84628c14fa2223a797c5c8e7ef56b7f9b3db76bd09c8213976617428b22a966
  process_peak_rss_bytes    73694867456
  process_swap_bytes        0

2026-08-05T16:11:29.174Z  child_closed_loop_layer_completed          layer 0 of 14
2026-08-05T16:11:29.174Z  child_closed_loop_restart_capsule_published
  capsule_child_state_sha256 e7ee806722e6cd542a94b570337cc0f57622676e6367b9e786e3fbee1cd8d713
  completed_layer_count      1
```

Capsule on disk:
`$RUN/restart-capsules/capsule-000000-8051a02d5481b0fb5eeb46fcd91ce6a5c93dc765c24316fdd4bcb82fcceddd45`
(rotation keep 2).

**Layer 0 wall clock: 13:07:22.761 → 16:11:29.174 = 3h04m07s**, about 1.8× r12's
1h41m. That is the cost `NOTES-W13D` §7 predicted in words — "each layer capture
here is additionally widened by the terminal floor, so budget more" — now
measured. Its four capture points:

| capture point | window | |
|---|---|---|
| `pre_producer` | 13:07:22 → 13:47:09 | 39m47s, replay 70.2 s at 1.981× |
| `post_producer` | 14:04:22 → 14:43:31 | 39m09s, replay 70.1 s at 1.988× |
| `post_consumer` | 14:43:45 → ~15:2x | replay at 1.98× |
| merged downstream | ~15:3x → 16:11 | the `--dense-closed-loop-merge-adjacent-layer-captures` capture |

Layer 1 then took its `pre_producer` capture **from that merged replay**
(`child_closed_loop_layer_capture_served_by_merged_replay`, 16:11:29.792, 0.6 s
after the layer opened) instead of paying another ~40-minute capture. That is
the merge flag earning its place, and it means layers 1..13 should not each cost
3h04m.

### 8.2 The floor and gate numbers are NOT in the live stream — a correction

`NOTES-W13D` §7 says `child_closed_loop_layer_validation_completed` "is the
event that carries a layer's decision metrics." **On this commit it does not.**
Measured over the whole 505 KB log, the only matching token in the entire stage
stream is `reclosure_enabled` (twice):

```
grep -oE 'terminal_[a-z_]*|boundary_[a-z_]*|floor[a-z_]*|reclosure[a-z_]*|tripwire[a-z_]*|margin_[a-z_]*'
  -> 2  reclosure_enabled          (and nothing else)
```

There is no `terminal_composition_floor_engaged`, no `terminal_relative_rmse`,
no `terminal_baseline_relative_rmse`, no `boundary_acceptance_summary` and no
`terminal_free_run_tripwire_evaluated` in `run.log`. The event carries the
**decision** (`validation_accepted`) and a **digest of it**
(`validation_decision_sha256`); the metrics behind that decision are sealed into
`diagnostics.json` and the reclosure receipt, which are written at the **end** of
the run.

Operationally this means:

- `validation_accepted: true` for layer 0 is the strongest live statement
  available, and it means the terminal floor was **satisfied** and the layer's
  install was accepted — a floor violation fails the layer rather than passing it.
- **The r13 arm's headline question — whether the reclosure acceptance gate
  refuses r12's 4.73× layer-0 writer blow-up — cannot be read until the run
  seals.** It lives in
  `report["diagnostics"]["boundary_acceptance"]["refused_transition_indices"]`.
  Do not expect to see it stream by.
- `NOTES-W13D` §8's layer-0 caveat is therefore **unresolvable live** as well:
  if a future layer is refused, `terminal_baseline_relative_rmse` has to be read
  out of `diagnostics.json`, not out of `run.log`.

### 8.3 Phase-cache side effects

One new entry published, under the rotated balanced-basis key
(§4.2/§6.2). The three global-topology entries were read and resumed, not
rewritten. The two pre-existing balanced-basis entries (`7367e3a1…`,
`ce0593c4…`) were neither read nor written. `round-000`, the sealed r12 run
directory, the sealed 2026-07-17 pack and both base caches: untouched.

---

## 9. Monitoring recipe

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
```

**The poller does all of it in one shot** — alive/etime/RSS, the sampled RSS
peak, log size, the last stage kinds, every closed-loop/gate/floor stage seen,
the last decision lines, a failure count, artifact sizes and the two GPUs:

```bash
$SSH --command-timeout 900 --execute "bash /home/cereal/nf-w13i/poll.sh"
```

Note `--command-timeout` — the client's default is 300 s, which is why a naive
`sleep 420` poll aborts. Use `--command-timeout 900` for any wait.

A second background helper, `/home/cereal/nf-w13i/milestones.sh`, tails the log
and appends every layer-decision / capsule / failure line to
`/home/cereal/nf-w13i/milestones.log` as it appears, so a long polling interval
cannot miss one. Both helpers exit by themselves when the run pid does.

```bash
# the decisions, one line each, nothing else
$SSH --execute "cut -c1-200 /home/cereal/nf-w13i/milestones.log"
```

Individual reads:

```bash
# alive + RSS right now
$SSH --execute "ps -o pid=,etime=,rss=,pcpu=,stat= -p \$(cat $RUN/run.pid)"

# TRUE RSS peak -- the runner's own stamp, not the sampler (§7)
$SSH --execute "grep -oE '\"process_peak_rss_bytes\":[0-9]+' $RUN/run.log | cut -d: -f2 | sort -n | tail -1"

# layer decisions so far
$SSH --execute "grep -E 'layer_validation_completed|layer_completed|restart_capsule_published' $RUN/run.log | tail -20"

# fail-closed check
$SSH --execute "grep -E '\"stage\":\"failure\"|Traceback' $RUN/run.log | tail -5"
```

### 9.1 What to read next, in order

1. **`child_closed_loop_layer_validation_completed`** per layer:
   `validation_accepted` plus `validation_decision_sha256`. Layer 0 is
   `true`/`c84628c1…`. **These booleans are the only live decision surface**
   (§8.2) — the floor and boundary numbers are not in the stream.
2. **`child_closed_loop_restart_capsule_published`** follows each accepted
   layer; `completed_layer_count` is the progress counter, and the capsule is
   what a `--dense-resume-from-capsule` relaunch would use.
3. **Pacing:** layer 0 cost 3h04m with four capture points. Layer 1 opened with
   `child_closed_loop_layer_capture_served_by_merged_replay`, so it should be
   materially cheaper. Do not extrapolate 14 × 3h04m.
4. **At seal time — and only then** — read from `diagnostics.json`:
   - `report["diagnostics"]["boundary_acceptance"]["refused_transition_indices"]`.
     Empty means the reclosure earned every writer it published; **layer 0 in
     the list means r12's 4.73× writer blow-up was reproduced and refused** —
     the single result this arm exists to produce.
     `every_published_boundary_no_worse_than_seed` must be `True`.
   - `terminal_relative_rmse` / `terminal_baseline_relative_rmse` per layer.
     `NOTES-W13D` §8's caveat still governs their interpretation: if a layer is
     refused, read `terminal_baseline_relative_rmse` **before** concluding
     anything — a pre-install terminal already ≥ 1.0 is an artifact of the
     un-refined downstream stack and calls for a nonzero
     `--dense-closed-loop-terminal-absolute-floor-minimum-layer`, **not** a
     weakened floor.
   - the composite document's enforced fields (§7.1):
     `heldout_extension_batch_count: 0`, `extension_records_are_never_heldout`,
     `heldout_pool: "base-members-only"`.
   - `terminal_free_run_tripwire_evaluated` will be `false` unless a
     counterfactual free-run probe is supplied; the tripwire is advisory here.
5. "Sealed" looks like: `$RUN/receipt.json` + `$RUN/diagnostics.json` written,
   `$RUN/published-child/` populated, `$RUN/ledger.jsonl` closed, and
   `$RUN/routed-manual-dossier.json` written for the manual whole-response review
   that remains the intelligence authority.

Phase timings **measured on this run**, superseding `NOTES-W13D` §7 where they
overlap:

| phase | this run | note |
|---|---|---|
| import + config + parent verify | 17 s | clause A cleared here |
| carrier verification | 5m31s | |
| global topology | **0.08 s** | 3 phase hits; W13D paid 38m06s |
| parent row materialization | 5m13s | 27.7 GB retained |
| bridge on-policy preimage | 48 s | 371 pairs, 0 forwards |
| **merged balanced basis** | **11m30s** | never previously run; r12's single-cache cold build was 9m48s |
| structural pass, 14 layers | **25 s** | |
| **closed-loop layer 0** | **3h04m07s** | 4 capture points; r12's was 1h41m |
| closed-loop layer 1+ | cheaper, TBD | opened from the merged replay, no fresh capture |
| moving-gauge body reclosure | ~19h → less | `fingerprint` removes ~17.7h of r12's install sweep |

Any relaunch needs a clear run directory — the launcher exits 40 if
`…-round-0-refold` exists.

---

## 10. Caveats

- **The r13 gate's headline question is still unanswered.** Layer 0 was
  *accepted*, which is real evidence that the terminal floor is satisfiable and
  the arm runs — but whether the reclosure acceptance gate **refuses** r12's
  4.73× layer-0 writer blow-up lives in `refused_transition_indices`, which is
  only written at seal time (§8.2). Three waves in, that number still does not
  exist. Do not read "layer 0 accepted" as "the gate convicted the reclosure";
  they are different claims.
- **The floor/boundary metrics cannot be monitored live** (§8.2). This corrects
  `NOTES-W13D` §7, which said otherwise. Anyone told to "watch the floor events"
  will find none.
- **`heldout_extension_batch_count` not directly observed** (§7.1). The
  93-of-256 split is arithmetic evidence for the same invariant; the sealed field
  itself is only readable once the receipt is written.
- **The RSS peak is a moving number.** 68.63 GiB is the peak *through layer 0*,
  not for the run. It grew 32.71 → 68.63 GiB across that one layer, and 13 layers
  plus the reclosure, final-norm and output-interface phases remain. The sampler
  in `/home/cereal/nf-w13i/` under-reports (49.23 vs 68.63 GiB); use the runner's
  own `process_peak_rss_bytes`.
- **Layer 0's 3h04m is not a per-layer rate.** Layer 1 opened from the merged
  replay and skipped a ~40-minute capture. Extrapolating 14 × 3h04m would be
  wrong in the pessimistic direction; nobody has measured a merged-replay layer
  to completion yet.
- **The balanced-basis phase key rotated**, so the two pre-existing entries were
  not reusable. Expected (`NOTES-W13G` §4), and nothing was pinned to force reuse.
- **A merged basis solved by `8a4ed56f` differs numerically from one solved by
  `84acded2`**, because the fit/held-out split became role-aware
  (`NOTES-W13G` §3). No merged basis had ever been published before this run, so
  nothing existing is invalidated — but this run's basis is the first of its kind
  and has no predecessor to compare against.
- **Nothing was weakened.** No flag was relaxed, no fail-closed clause was
  bypassed, no phase key was pinned, and no retry was attempted.
- **Read-only artifacts untouched**: the sealed r12 run directory, `round-000`,
  the sealed 2026-07-17 pack and both base caches were never written to. The two
  preserved failed runs from W13D and W13F remain in place.
- **Numeric only.** Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`,
  nothing here is an intelligence verdict; the manual whole-response review
  remains the authority.
- **Not committed.** No `git add`, no commit, no push.

---

## 11. Artifacts

Server, under `/home/cereal/nf-w13i/` (kept):

| file | what it is |
|---|---|
| `nf-w13i.bundle` | the deployment bundle, sha256 `643e92c3…` |
| `HEAD-before.txt`, `status-before.txt` | pre-reset deployment state (`84acded2…`, clean) |
| `nf-launch-r12-round-refold.sh.preinstall-v2off` | the replaced v2-OFF launcher, `24c61cb3…` |
| `launch-refold.out` | the launcher stage's own stdout (preflight + `RUN=`/`PID=`) |
| `runner-help-8a4ed56f.txt`, `help-flags.txt`, `used-flags.txt` | the 178-option `--help` and the 108/108 flag check |
| `phase-cache-before.txt` | phase-cache root inventory taken at 12:38:07, before launch |
| `rss-sampler.sh`, `rss-samples.tsv`, `rss-sampler.log` | the 15-second RSS sampler and its samples (**under-reports**, see §7) |
| `milestones.sh`, `milestones.log` | the decision-event recorder (§9) |
| `poll.sh` | the monitoring poller (§9) |

Deployment: `nf-launch-r12-round-refold.sh.bak-v2off-20260805T123653Z`
(`24c61cb3…`, W13F's v2-OFF form) alongside W13F's `…bak-20260805T104136Z`
(`e496c102…`, which is what is now installed), W13D's `…bak-20260805T083111Z`
and the older `…bak-20260805T042559Z`.

Preserved failed runs, under
`…/neural_forge_exact4x_20260730/catalog-integrated-carrier/` — **neither
deleted, neither touched**:

| directory | |
|---|---|
| `…-round-0-refold.failed-merged-basis-104326` | W13D's run |
| `…-round-0-refold.failed-config-streaming-v2-104727` | W13F's run |

New phase-cache entry published by this run:
`…/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1/lfm2-circuit-balanced-residual-basis-v1/6e423f9354c485b925fb1759a099499e45f285cb1f9f88592fb5ef5e372ab65c`
(85,984,075 bytes) — the first merged-evidence balanced basis.
