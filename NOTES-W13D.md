# NOTES-W13D — r12→r13 bridged REFOLD: composed launch on the fixed solver stack

Wave 13D, branch `codex/lrc-blank-compression`. This deploys `53b353f0`, resolves
`NOTES-W12E` §6's open refold question, composes the r13 flag set from
`R13_PLAN.md` §3 + `NOTES-W13B` + `NOTES-W13C`, and launches the bridged refold
that consumes the sealed r12 on-policy round.

**Status: FAILED CLOSED at 09:25:50 UTC, 53m53s in, before the first body-layer
solve.** Not the terminal floor and not a gate — an infrastructure defect that
makes **on-policy extension evidence and the balanced-basis phase cache mutually
exclusive on this commit** (§6.4). The deployment bump, the base-evidence
reconciliation and the flag composition are all verified good: the merged
evidence set opened, both rebound envelopes authenticated, and the run reached
the system fold. The relaunch needs one configuration decision that is not
mine to take (§6.5).

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.
Run directory
`/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold`.

---

## 1. Deployment bump: 13f71132 → 53b353f0

Local `HEAD` was `8e08844c` (*Authenticate conditioned replay runtime captures
correctly*) at bundle time; the deployment was pinned to `53b353f0`
(*Make the reclosure prove it improved every boundary it overwrites*) as
instructed — `8e08844c` is deliberately **not** deployed.

```
bundle  13f711323e923b683347de1dfd315dfe4547f924..codex/lrc-blank-compression
        (12 commits; 3953ede4 … 8e08844c)
size    849,048 bytes
sha256  9448ae34e906b5e9c0f29cca30ff58992f0fe51540c4dfe077f904d21d8551b0   (identical both sides)
server  /home/cereal/nf-w13d/nf-w13d.bundle
```

`git bundle verify` clean, then
`git fetch <bundle> refs/heads/codex/lrc-blank-compression:refs/heads/w13d-import`
and `git reset --hard 53b353f0…`. Pre-reset `HEAD` and `status` recorded in
`/home/cereal/nf-w13d/HEAD-before.txt` (`13f71132…`) and `status-before.txt`
(0 bytes — the tree was already clean, so nothing was overwritten).

Deployment `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`:

| | |
|---|---|
| `git rev-parse HEAD` | `53b353f0ef5f2e105c35fc58c4ed28f3a6d4f2bf` |
| `git status --porcelain` | **empty** |

File-for-file against the local committed blobs at `53b353f0` — all seven
identical:

| file | sha256 |
|---|---|
| `src/neural_forge/hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` |
| `src/neural_forge/hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` |
| `src/neural_forge/hf_lfm2_on_policy_evidence_round.py` | `a3bf6d1f3e2f7bf88137b980918ba945a43ef35dd565521c138726ad64e24b86` |
| `src/neural_forge/hf_lfm2_moving_gauge_residual_reclosure.py` | `7202be1c75f713e129493453f7b66ad7a8d65541b70a6676a3934cba92e78da5` |
| `src/neural_forge/hf_lfm2_dynamic_closed_loop_adapter.py` | `ada3ea75e4a6694df12431389ddbef2ee664da59eb4a7881f1e2ffed701be0cf` |
| `experiments/run_lfm2_dynamic_catalog.py` | `1f2069489e4016794854a1a15187cc2fa44b9c188867c500e2cd838647652b0e` |
| `experiments/run_r12_on_policy_round.py` | `ad5e72a54880360af3887a4d4641968d6dc9fe7a5299d69000343ca7dcc000ec` |

**The frozen-collector rule holds.** `hf_lfm2_circuit_trace_cache.py` is
byte-identical at `13f71132` and `53b353f0` (`16271526…` in both), and so is
`hf_lfm2_canonical.py` (`789c6a6f…`). Checked against the committed blobs, not
assumed from the commit titles.

Runtime implementation identity, recomputed on the deployment after the bump
rather than copied from `NOTES-W12E`:

```
collector_source_sha256            16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945
canonical_contract_source_sha256   789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d
transformers_lfm2_source_sha256    2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150
torch 2.11.0+cu128  transformers 5.12.1  safetensors 0.8.0
implementation_sha256              732ca9ce7bd135d1e14d99fdc55e3aa9b1b40ad68c3e5a9d6c1dd2842b35c227
```

so the W12E recapture and the round's extension cache stay consumable:

| cache | `implementation_sha256` | `layer_topology_sha256` |
|---|---|---|
| recaptured base `c29b6737…` | `732ca9ce…` | `92166d4d…` |
| extension `084ab7dc…` (115 batches) | `732ca9ce…` | `92166d4d…` |
| **sealed 2026-07-17 base `76ff3cbc…`** | **`eb3bddeb…`** | `92166d4d…` |

---

## 2. The refold's base evidence: what it actually reads, and why the recapture
   was not optional

`NOTES-W12E` §6 left this open as a bookkeeping choice. It is not one — the old
base cache **cannot** be used at all.

### 2.1 The consumption chain, traced

The refold stage passes three things that touch base identity:

1. `--sealed-parent-cache-directory` → `resolve_production_parent_cache`
   (`run_lfm2_dynamic_catalog.py:4392`) opens it as `parent_cache`;
2. `--on-policy-round-manifest` → `_resolve_on_policy_consumed_rounds`
   (`:3941`) authenticates the round and resolves its extension cache;
3. `--cached-parent-outputs` (+`-sha256`) → `load_lfm2_cached_whole_output_baseline`
   (`:11414`), and `--tokenizer-bridge-mapped-parent-outputs` (+`-sha256`) →
   `_load_bridge_mapped_parent_baseline` (`:5616`).

(1) and (2) are then opened together as one `Lfm2SealedEvidenceSet`
(`_open_sealed_evidence_set`, `:4011`). `Lfm2SealedEvidenceSet.open`
(`hf_lfm2_sealed_evidence_set.py:698-707`) refuses:

> `Sealed evidence members disagree on implementation or layer topology; one
> closed-form solve cannot consume them.`

The sealed base is `eb3bddeb…` and the round's extension is `732ca9ce…`, so the
refold would have failed closed there. **The base cache had to become the W12E
recapture; everything else in §2 follows from that.**

(3) is the part that then had to move with it. Both envelopes bind the live base
cache by name:

- `load_lfm2_cached_whole_output_baseline` (`dynamic_lfm2_behavior.py:544-562`)
  requires `parent_cache_key == parent_cache.cache_key`;
- `_load_bridge_mapped_parent_baseline` (`:5660-5687`) requires the same, plus
  `original_parent_baseline_fingerprint` and
  `original_parent_baseline_envelope_sha256` of the *original* envelope — which
  move as soon as the original is re-sealed.

Phase caches are affected only in the intended way: `parent_cache_key` enters
`SEALED_CACHE_SPLIT_SCHEMA` (`:4465`), so the balanced-basis and
global-topology phases recompute for this evidence identity. No phase key is
pinned (see §4).

### 2.2 The producer, and what was actually re-run

`canonical_parent_outputs_bridge_bound.json` is written by
`experiments/prepare_lfm2_tokenizer_bridge_evidence.py`
(`ORIGINAL_ENVELOPE_FILENAME`, `:66`). Its own docstring names the operation:
step 4, *"rebinds an already-sealed canonical-parent behavior baseline to that
cache"*. The envelope is not derived evidence — it is the pack's
`canonical_parent_rows.json` (28 one-time parent responses, generated once in
2026-07) wrapped in identity bindings by
`seal_lfm2_cached_whole_output_envelope`
(`dynamic_lfm2_behavior.py:362`, *"Bind existing one-time parent responses to
their exact runtime identities"*). Same for `mapped_parent_outputs.json` over
`mapped_parent_rows.json` via
`seal_lfm2_tokenizer_bridge_mapped_parent_envelope`
(`run_lfm2_dynamic_catalog.py:5428`).

Running the producer verbatim was not an option — it would rebuild the parent
trace cache and re-generate the mapped rows through the parent, i.e. re-derive
evidence r12 sealed. So, exactly as `NOTES-W12E` §2.2 did for the recapture, a
driver outside the deployment re-runs **only the producer's sealing steps**,
importing them rather than reimplementing them:

`/home/cereal/nf-w13d/rebind_bridge_bound_outputs.py`
(log `rebind.log`, report `rebind-report.json`).

```
parent_model_loads 0   parent_forward_calls 0   parent_generation_calls 0
network_calls 0        total 669.7 s (bridge pairing 664.6 s of it)
```

Output — a **new** directory in the round root, nothing overwritten:

```
$ROUND_ROOT/bridge-bound-outputs-recaptured-w13d/
  canonical_parent_rows.json                  2f3f4085…  (byte copy of the pack's)
  canonical_parent_outputs_bridge_bound.json  fa97d9a8…  (re-sealed)
  mapped_parent_rows.json                     cfba4818…  (byte copy of the pack's)
  mapped_parent_outputs.json                  4fea48e1…  (re-sealed)
```

### 2.3 Old vs new binding, field by field

| | sealed pack | rebound | |
|---|---|---|---|
| `parent_cache_key` (both envelopes) | `76ff3cbc946ea52b…` | `c29b67370d5e33be…` | **DIFF — the point** |
| `canonical_parent_outputs_bridge_bound.json` file sha256 | `253be73e1b65a790…` | `fa97d9a8d70f2974…` | DIFF (follows) |
| … `envelope_sha256` | `83a730ed136aae7c…` | `985846d54682e332…` | DIFF (follows) |
| `mapped_parent_outputs.json` file sha256 | `3bbdffdea1d5a146…` | `4fea48e1494fa015…` | DIFF (follows) |
| … `envelope_sha256` | `90a7e5454f1ca6c5…` | `68b3770e9f6e4825…` | DIFF (follows) |
| … `original_parent_baseline_fingerprint` | `f35e04b1e0c64f31…` | `74b07c9b3e620065…` | DIFF (follows) |
| **`rows`** (28 cases, both envelopes) | | | **byte-identical** |
| `source_rows_sha256` | `2f3f4085696264ff…` | `2f3f4085696264ff…` | SAME |
| `parent_state_fingerprint` | `96a6779d927a8aca…` | `96a6779d927a8aca…` | SAME |
| `tokenizer_fingerprint` | `eb035507f87ad490…` | `eb035507f87ad490…` | SAME |
| `generation` protocol / `max_new_tokens` 64 | | | SAME |
| `bridge_artifact_sha256` | `5c1a452cbe6bb78c…` | `5c1a452cbe6bb78c…` | SAME |
| `transport_profile_sha256` | `e3fc98c666b70b36…` | `e3fc98c666b70b36…` | SAME |
| `trace_adapter_evidence_sha256` | `22d09d387457c7df…` | `22d09d387457c7df…` | SAME |
| `canonical_target_tokenization_evidence_sha256` | `649c466942ee6126…` | `649c466942ee6126…` | SAME |
| `canonical_behavior_input_evidence_sha256` | `e1ce71f61d39064f…` | `e1ce71f61d39064f…` | SAME |

The driver asserts this mechanically: the changed-key set of the original
envelope body is exactly `{parent_cache_key, envelope_sha256}` and of the mapped
envelope exactly
`{parent_cache_key, original_parent_baseline_fingerprint, original_parent_baseline_envelope_sha256, envelope_sha256}`,
with no key added or removed and both `rows` lists equal. It also re-resolves
the tokenizer bridge **against the recaptured cache** and refuses unless that
reproduces the pack's four bridge digests — which is what proves the recapture
carries the same corpus the bridge was authenticated against — and then loads
both envelopes back through the production loaders, including
`_load_bridge_mapped_parent_baseline`, whose normalization control comes out
`passed: true` with the pack's own numbers (`parent_nonempty_retention 0.875`,
`task_score_retention 0.75`, `correctness_retention 0.375`, 28 cases).

The 2026-07-17 pack was **not** written to: its files still hash to
`253be73e…` / `3bbdffde…` / `2f3f4085…` / `cfba4818…` and
`find $PACK -newermt "2026-08-05 01:00"` returns nothing.

---

## 3. Launcher changes

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`

```
backup       nf-launch-r12-round-refold.sh.bak-20260805T083111Z   (also /home/cereal/nf-w13d/…preinstall)
pre-edit     8d2e5a4d5b2c51c83a41f0aa219333fb5e11356ff05a3327c52ce5df234f7bd5
post-edit    e496c1022b37862af8c1770e23ede2e862f0cf56bbde22a2922f9e9a7806beda
```

`bash -n` clean locally and on the server, before and after install.

1. **New variables, no repoint of the old ones.** `BASE_CACHE` and `$PACK` stay
   exactly as they were; the corpus inputs (`--calibration`,
   `--target-tokenizer-calibration`) still come from the sealed pack, because
   the recapture preserved them bit for bit.

   ```
   REBOUND=$ROUND_ROOT/bridge-bound-outputs-recaptured-w13d
   REBOUND_PARENT_OUTPUTS_SHA256=fa97d9a8…
   REBOUND_MAPPED_PARENT_OUTPUTS_SHA256=4fea48e1…
   ```

2. **New refold-stage preflight** (`exit 49`), after the existing round-preimage
   check: both rebound envelopes are digest-checked, both are required to name
   `ROUND_BASE_CACHE`'s own `cache_key`, and both are required to carry rows
   equal to the sealed pack's. Two file reads; it turns a mixed-identity
   evidence set into a one-line refusal instead of a surprise after the parent
   and the pack are loaded.

3. **The refold command** — see §4.

4. Header rewritten: the "Refold caveat, NOT resolved here" paragraph is
   replaced by the W13D resolution and the r13 flag rationale.

`round-000`, the sealed caches, the sealed r12 run directory and the round stage
were not touched.

---

## 4. The composed r13 refold command

```
SOURCE_COMMIT=53b353f0ef5f2e105c35fc58c4ed28f3a6d4f2bf \
  bash /home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh refold
```

Every flag below was checked against the **deployed** runner's `--help` at
`53b353f0` before launch (181 options parsed out of `--help`, each of the 21
r13-relevant spellings confirmed present); `argparse` has no
`parse_known_args` on this path, so a typo aborts rather than no-ops.

Deltas from r12's refold flag set:

| flag | value | why |
|---|---|---|
| `--sealed-parent-cache-directory` | `$ROUND_BASE_CACHE` | §2.1 — the only base the round's extension can be opened with |
| `--cached-parent-outputs` / `-sha256` | `$REBOUND/canonical_parent_outputs_bridge_bound.json` / `fa97d9a8…` | §2.2 |
| `--tokenizer-bridge-mapped-parent-outputs` / `-sha256` | `$REBOUND/mapped_parent_outputs.json` / `4fea48e1…` | §2.2 |
| `--dense-moving-gauge-writer-reclosure-acceptance` | `require_boundary_improvement` | W13C §5. Not fail-closed: r12's damage was concentrated in the terminal-absorbing reverse steps and aborting the whole reclosure throws away the boundaries it helps |
| `--dense-moving-gauge-boundary-regression-tolerance` | `1e-6` | W13C §5's FP32-null-change caveat — at exactly 0.0 the publication cast can refuse a change that is numerically a no-op (observed at ratio 1+4e-9). Every real regression measured was 1.1x–4.2x |
| `--dense-moving-gauge-install-verification` | `fingerprint` | R13 §3's largest single win (−17.7h). **Decision recorded:** this arm optimizes wall-clock for the intelligence-review answer; the terminal capture still pins the full child `state_dict`, and a full-recapture certification pass is reserved for a child that reviews well |
| `--dense-closed-loop-terminal-composition-floor` | on | R13 item 2 |
| `--dense-closed-loop-terminal-regression-tolerance` | `0.25` | the flag's default, stated explicitly so the operating point is in the command |
| `--dense-closed-loop-terminal-absolute-floor-minimum-layer` | `0` | the flag's default, stated explicitly. See the caveat in §8 |
| `--dense-closed-loop-terminal-free-run-tripwire` | on (advisory) | R13 item 2(c). The stop rung is **not** set — R13 §2 specifies "opt-in stop flag, default off" |
| `--dense-closed-loop-solver-conditioning-diagnostics` | on | R13 item 8 |
| `--dense-closed-loop-gate-operating-point-diagnostics` | on | R13 item 8 |
| `--dense-closed-loop-depth-telemetry` | on | R13 item 8 — r12's compounding signature was unreconstructable |
| `--dense-closed-loop-input-error-multiplier` | `0` | **deliberate control.** `--dense-closed-loop-input-error-covariance` is NOT passed, so the EIV ridge is off and the multiplier is stated only to record the arm. W13B §5 also shows EIV is a structural no-op at layer 0 and never reaches an operator write at any depth |
| `--dense-closed-loop-margin-row-weights` | **not passed** | same attribution reason: this arm isolates the reclosure acceptance gate's effect on the composed result |

Unchanged from r12 / R13 §3: `--dense-body-reclosure-mode moving_gauge_v3`,
`--dense-closed-loop-replay-mode prefix_by_prefix_canonical`,
`--dense-closed-loop-merge-adjacent-layer-captures`,
`--dense-replay-devices cuda:0,cuda:4`, `--dense-restart-capsule-root`
+ `--dense-restart-capsule-keep 2`,
`--dense-closed-loop-performance-metrics-diagnostic-only`, the carrier package,
the counted budget, `--required-target-hidden-size 816`, and the balanced-basis
and global-topology phase caches with **no**
`--balanced-basis-existing-phase-key-sha256` (R13 §4's r11 lesson).

**Left out deliberately:** R13 §3's *terminal restart capsule* has no flag on
this runner — the deployed `--help` at `53b353f0` exposes only
`--dense-restart-capsule-root/-keep` and `--dense-resume-from-capsule`. That
pre-run engineering item never landed, so a crash in the terminal phase still
costs the terminal phase. Nothing was invented to cover it.

---

## 5. Launch timeline (UTC, 2026-08-05)

| | |
|---|---|
| 08:0x | bundle built (`9448ae34…`), uploaded, verified; deployment reset to `53b353f0`, tree clean |
| 08:1x | runtime `implementation_sha256` recomputed → `732ca9ce…`; base/extension/old-base fingerprints read back |
| 08:18–08:29 | envelope rebind: 669.7 s, 0 parent forwards, both envelopes re-sealed and re-loaded through the production loaders |
| 08:31:11 | launcher backed up (`…bak-20260805T083111Z`), edited launcher installed, `bash -n` clean |
| 08:31:2x | `preflight` stage → exit 0 on both caches |
| 08:31:45 | `refold` stage launched |
| 08:31:57 | `run_started`; replay sharding requested `cuda:0,cuda:4` |
| 08:32:07 | `parent_verified` (229,693,184 unique parameters, bf16) |
| 08:32–09:21 | evidence, bridge, carrier, global topology — all clean (§6.3) |
| 09:21:34 | `system_fold_started` |
| 09:25:50 | **failed closed** in the balanced-basis phase (§6.4); receipt written |

Run configuration digest: `b13987d675d5252cc46fce3407619bfbc38720fe5b4fa9fdccae91b0373c9d0d`.

```
PID   1053530  (exited 09:25:50 UTC, 53m53s wall clock)
RUN   /home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold
log   $RUN/run.log      pid file $RUN/run.pid
```

GPU state at launch: cuda:0 308 MiB, cuda:4 499 MiB (ComfyUI, coexisting), both
0% utilization; the P40s and the RTX 8000 idle and unused by this run.

---

## 6. Health evidence

### 6.1 Preflight and evidence authentication (all passed)

```
Child bridge reference matches the sealed bridge.
…lfm2_semantic_whole_response_prompts_v1.json: OK
…lfm2_on_policy_fit_prompts_v1.json: OK
…lfm2_manual_review_reserved_digests_v1.json: OK
Base capture selection f229e5f3… in …/parent_trace_cache/76ff3cbc…
Base execution 3d077b94… (eager on cuda) implementation eb3bddeb…
Base capture selection f229e5f3… in …/base-cache-recaptured-w12e/c29b6737…
Base execution 3d077b94… (eager on cuda) implementation 732ca9ce…
Preflight OK.
Round 0 carries a target-token preimage for 115 batches
  (target-tokenizer-canonical-then-one-row-mapped-to-donor).
…/bridge-bound-outputs-recaptured-w13d/canonical_parent_outputs_bridge_bound.json: OK
…/bridge-bound-outputs-recaptured-w13d/mapped_parent_outputs.json: OK
canonical_parent_outputs_bridge_bound.json binds parent_cache_key c29b6737…
mapped_parent_outputs.json binds parent_cache_key c29b6737…
canonical_parent_outputs_bridge_bound.json carries the sealed pack rows (canonical_parent_rows.json 2f3f4085…)
mapped_parent_outputs.json carries the sealed pack rows (mapped_parent_rows.json cfba4818…)
```

### 6.2 The evidence set opened — the reconciliation's load-bearing proof

08:43:2x, `on_policy_extension_evidence_materialized`:

```
member_count                    2
evidence_set_cache_key          f62110d77ddc3e1bbf646f633087702e913d2215fd08f313816e04500ce6b4e4
evidence_identity_sha256        75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5
base_batch_count                256      base_path_count       67,583
extension_path_count            6,406    merged_row_count      73,989
all_extension_rows_are_fit_rows true
merge_member_rows_are_zero_copy_views  true
merge_parent_forward_calls      0
```

`Lfm2SealedEvidenceSet.open` accepted both members. That is the check that
would have raised *"Sealed evidence members disagree on implementation or layer
topology"* against the old `76ff3cbc…` base, so it is the direct evidence that
§2's reconciliation was both necessary and correct.

Preceding it, 08:43:09 `tokenizer_bridge_on_policy_preimage_complete`:

```
authenticated_pair_count 371   extension_record_count 115   merged_record_count 371
merged_trace_adapter_evidence_sha256 aa0180c533cc9fe129f41f65a74e3533704c4258e297b8829da56ebcc6d58ca0
parent_forwards 0
```

and 08:42:20 `parent_rows_materialized`: 73,989 paths, 27,729,893,376 retained
bytes, `trace_rows_sha256 5046c63988800bcc343277b1f34d943f95968a2b3ca004e0144c38f6f5041c61`.
08:37:15 `carrier_verified`: 57,420,144 counted total parameter elements,
2 independent routes.

Both phase caches reported `global_topology_phase_cache_miss` and recomputed —
expected, and the intended consequence of the new evidence identity (§2.1); no
phase key is pinned.

### 6.3 What did complete

| UTC | |
|---|---|
| 08:31:57 | `run_started`, replay sharding requested `cuda:0,cuda:4` |
| 08:32:07 | `parent_verified` |
| 08:37:15 | `carrier_verified` — 57,420,144 counted elements, 2 routes |
| 08:42:20 | `parent_rows_materialized` — 73,989 paths, 27.7 GB retained |
| 08:43:09 | bridge on-policy preimage complete — 371 pairs, 0 parent forwards |
| 08:43:2x | evidence set opened, 2 members (§6.2) |
| 08:43:28 → 08:55 | global-topology capacity/frontier compute (both phase caches missed, as designed) |
| 09:21:20 | `global_topology_frontier_selection_completed` — `selected_hidden_size 816`, `frontier_sha256 8a4a1ac8e0959f1c55ce81e7f0f8efba208a1c16fb9fdb95fee986fa4077cdc7` |
| 09:21:33 | `wiring_planned` — `planned_target_abi_sha256 f04b6fda934188f9be3f7e0d003260180dea82870f07b3914442086a2156bf34` |
| 09:21:34 | `system_fold_started` |
| **09:25:50** | **`failure`** |
| 09:25:50 | `final_receipt` — `receipt_status failed_closed`, `receipt_sha256 8c826a07053cd081da00b5e851c6e33705119b8bcbd71a681b2915411d4d9b6a` |

The exact-4x layout re-solved to the same width r12 used (816) on the merged
evidence, which is the last thing this run proved.

### 6.4 The failure, exactly

```
{"stage":"failure","exception_type":"Lfm2BalancedBasisPhaseCacheError",
 "failure_category":"fold_or_infrastructure_failure","fail_closed":true,
 "timestamp_utc":"2026-08-05T09:25:50.169Z","sequence":99}

receipt.json:
  "failure": {"category": "fold_or_infrastructure_failure",
              "exception_chain": [{"type": "Lfm2BalancedBasisPhaseCacheError",
                                   "message": "trace_cache must be Lfm2CircuitTraceCache."}],
              "fail_closed": true},
  "status": "failed_closed"
```

Root cause, traced and then verified on the deployment rather than inferred:

1. `hf_lfm2_dense_budget_model_fold.py:474-505` `_resolve_dense_evidence_cache`
   — with any extension member present, the dense writers' cache handle is
   **replaced** by `evidence_set.merged_cache()`, "and every identity the writer
   binds becomes the merged evidence identity". That is
   `Lfm2MergedSealedEvidenceCache`.
2. `hf_lfm2_dense_budget_model_fold.py:2134` `_build_residual_basis`, branch at
   `:2189-2228` —
   when a balanced-basis phase-cache context exists it routes to
   `load_or_build_lfm2_circuit_balanced_basis`; with `phase_cache_context is
   None` it calls `build_lfm2_circuit_balanced_basis` directly.
3. `hf_lfm2_balanced_basis_phase_cache.py:777-783` `_trace_cache_binding`
   requires a concrete `Lfm2CircuitTraceCache` and raises on anything else.
4. `hf_lfm2_balanced_circuit_basis.py:3099-3101` — the **direct** builder types
   its parameter `trace_cache: Any` and imposes no such check.

Verified on the runtime at `53b353f0`:

```
issubclass(Lfm2MergedSealedEvidenceCache, Lfm2CircuitTraceCache)  ->  False
Lfm2MergedSealedEvidenceCache.__bases__                           ->  (object,)
```

So **the phase-cached balanced basis cannot accept merged on-policy evidence at
all**, and the direct build is the only path that can. The combination was
clearly *intended* — `run_lfm2_dynamic_catalog.py:9535`
`_compose_on_policy_phase_dependencies` exists precisely to "bind extension
evidence into the balanced-basis phase-cache key" — it has simply never been
exercised end to end. r12 never hit it: r12 consumed no extension evidence, so
its handle stayed a real `Lfm2CircuitTraceCache` even on its own
`balanced_basis_phase_cache_miss`.

**This is not caused by anything composed in this wave.** It does not touch the
reclosure-acceptance gate, the terminal floor, the telemetry flags, the install
verification mode, or the base-cache reconciliation; any refold consuming a
round with `--balanced-basis-phase-cache-root` set fails here, on either base
cache.

Side effects, checked:

- Nothing was written under the balanced-basis phase-cache root
  (`find … -newermt` empty), and `$RUN/balanced-basis-phase-events.jsonl` is
  empty — the type check fires before the first phase event.
- Three **global-topology** phase entries *were* published and are legitimate
  (`lfm2-global-capacity-union-v1`, `lfm2-global-frontier-selection-v1`,
  `lfm2-global-functional-width-v1`, all 08:55–09:21 UTC). They are authenticated
  and keyed on this evidence identity, so a relaunch will hit them and skip
  ~50 minutes of topology compute.
- `$RUN/ledger.jsonl` is empty, no child was published, `$RUN/diagnostics.json`
  (320 KB) and `$RUN/receipt.json` (284 KB) are written. The round, the caches
  and the sealed pack are untouched.

### 6.5 The decision the relaunch needs — not taken here

The only configuration in which this refold can run on `53b353f0` is with the
**balanced-basis phase cache off** (drop the four `--balanced-basis-phase-*`
flags, which drops `$BALANCED_BASIS_EVENTS` with them). That contradicts an
explicit instruction for this arm ("balanced-basis + global-topology phase
caches ON"), so it was not done unilaterally. The options, with what each costs:

1. **Run with `--balanced-basis-phase-cache-root` and friends removed.**
   `build_lfm2_circuit_balanced_basis` computes the identical basis from the
   same merged evidence — the phase cache is a cache, not a gate, and no
   numeric or identity clause weakens. What is lost is the HMAC-authenticated
   phase event log for this basis and any reuse of it by a later run; r12's
   cold build of this phase took ~9.5 minutes. Smallest change, available now.
2. **Fix the defect** — teach `_trace_cache_binding` the merged handle (it needs
   `cache_key`, `manifest`, `_completion`, the four fingerprints and
   `batch_count`, all of which `Lfm2MergedSealedEvidenceCache` already exposes),
   or have `_build_residual_basis` bind the phase cache to
   `evidence_set.base_cache` while building from the merged handle. That is a
   source change, a new commit and another deployment bump, and it changes what
   the phase key means for extension-bearing runs — a design call.
3. **Drop the round from this refold** (no `--on-policy-round-manifest`), which
   keeps every phase cache but throws away the entire point of the arm. Listed
   only for completeness.

A relaunch under any option must first move `$RUN` aside — the launcher exits
40 on an existing run directory, and the failed run's receipt and diagnostics
are the evidence for §6.4 and should be kept.

---

## 7. Monitoring recipe (for the relaunch)

```bash
SSH="python C:/Users/cereal/.claude/skills/server-manager/ssh_client.py --server ai"
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold

# alive?
$SSH --execute "ps -p \$(cat $RUN/run.pid) -o pid,etime,rss --no-headers"

# stage stream (one JSON object per line, 'stage' is the event name)
$SSH --execute "grep -o '\"stage\":\"[a-z_0-9]*\"' $RUN/run.log | tail -20"

# the events that matter for this arm
$SSH --execute "grep -E 'terminal_composition|terminal_absolute_floor|terminal_free_run|boundary_acceptance|refused|closed_loop_layer' $RUN/run.log | tail -40"
```

What to read, in order:

- **`child_closed_loop_layer_validation_completed`** is the event that carries a
  layer's decision metrics; `child_closed_loop_restart_capsule_published`
  follows it. On r12's pacing the first one lands ~1h41m after
  `dense_closed_loop_started`, so a healthy run reaches layer 2's decision
  roughly 5h after launch.
- **`terminal_composition_floor_engaged` / `terminal_relative_rmse` /
  `terminal_baseline_relative_rmse`** in each layer's validation metrics. A
  layer is stopped when `terminal_relative_rmse >= 1.0` (absolute wall) or when
  it regressed past `0.25` across the install. `terminal_composition_floor_failed`
  names it explicitly.
- **`boundary_acceptance_summary.refused_transition_indices`** in the reclosure
  receipt (`report["diagnostics"]["boundary_acceptance"]`). Empty means the
  reclosure earned every writer it published; **layer 0 in the list means r12's
  4.73x writer blow-up was reproduced and refused** — which is the single result
  this arm exists to produce.
- `every_published_boundary_no_worse_than_seed` must be `True`.
- `terminal_free_run_tripwire_evaluated` will be `false` unless a counterfactual
  free-run probe is supplied; the tripwire is advisory here either way.

"Sealed" for this run looks like: `$RUN/receipt.json` + `$RUN/diagnostics.json`
written, `$RUN/published-child/` populated, `$RUN/ledger.jsonl` closed, and
`$RUN/routed-manual-dossier.json` written for the manual whole-response review
that remains the intelligence authority.

Expected phases and rough timings (R13 §3's 35–38h estimate for one
body+reclosure block, with the install sweep collapsed):

| phase | rough | note |
|---|---|---|
| evidence authentication + bridge pairing | **~11 min, measured** | 08:32→08:43 this run |
| global-topology phases | **~38 min cold, ~0 on relaunch** | 08:43→09:21 this run; the three entries are now published and keyed on this evidence identity |
| balanced basis | ~10 min | r12's cold build was 9m48s |
| body fold, 14 layers | ~18h+ | r12's layer 0 ran 00:30:08 → 02:11:24 (**1h41m**) and layer 1 the same; each layer capture here is additionally widened by the terminal floor, so budget more |
| moving-gauge body reclosure | ~19h → less | r12's 19.4h included the install sweep; `fingerprint` removes ~17.7h of it |
| final-norm + output-interface solves | ~11.8h | the opaque block R13 §3 flags as the next audit target |
| evaluators + publication | ~11–14h | fresh-reload evaluation is still duplicated on this commit |

---

## 8. Caveats

- **None of the r13 gates were exercised.** The run died in the balanced-basis
  phase, before the structural pass and before any closed-loop layer, so this
  wave produced **zero** evidence about the reclosure acceptance gate, the
  terminal floor, or the telemetry. Everything in §4 is a composed and
  syntax/`--help`-verified configuration, not a measured one.
- **The absolute terminal wall is engaged from layer 0.** That is the flag's
  default and the strictest setting, and it is what makes a fast failure
  possible. But the adapter's own comment
  (`hf_lfm2_dynamic_closed_loop_adapter.py:1054-1058`) warns that a partially
  folded child still carries un-refined layers downstream, so an early terminal
  metric can legitimately sit above the no-branch baseline. If the run aborts at
  layer 0, read `terminal_baseline_relative_rmse` in that decision **before**
  concluding anything: if the *pre-install* terminal was already ≥ 1.0, the
  abort is an artifact of the un-refined downstream stack and the correct
  response is to re-launch with a nonzero
  `--dense-closed-loop-terminal-absolute-floor-minimum-layer`, not to weaken the
  floor. If the pre-install terminal was below 1.0 and the install pushed it
  through, that is a real conviction. Nobody has measured this quantity on a
  partially folded child — r12's receipt carries no terminal metric at all,
  which is why item 2 exists.
- **The reclosure gate is measured on the writer's own aligned FIT rows**, not
  held-out writer rows (W13C §2's stated limitation).
- **The EIV and margin-weight arms are not being tested here.** This is the
  multiplier=0 control by choice; their arms are a separate launch.
- **No terminal restart capsule** (§4).
- **Numeric only.** Per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`, nothing here is an
  intelligence verdict; the manual whole-response review remains the authority.
- **Not committed.** No `git add`, no commit, no push.

---

## 9. Artifacts

Server, under `/home/cereal/nf-w13d/` (kept):

| file | what it is |
|---|---|
| `nf-w13d.bundle` | the deployment bundle, sha256 `9448ae34…` |
| `HEAD-before.txt`, `status-before.txt` | pre-reset deployment state (`13f71132…`, clean) |
| `rebind_bridge_bound_outputs.py` | the envelope rebind driver |
| `rebind.log`, `rebind-report.json` | its run and its field-by-field report |
| `nf-launch-r12-round-refold.sh.new`, `…preinstall` | installed and pre-install copies of the launcher |

Round root:

| path | |
|---|---|
| `bridge-bound-outputs-recaptured-w13d/` | the rebound envelopes + copied rows (§2.2) |
| `base-cache-recaptured-w12e/c29b6737…` | untouched |
| `round-000/` | untouched (manifest `75a69021…`, extension cache `084ab7dc…`) |

Failed run (kept as the evidence for §6.4; must be moved aside before any
relaunch, the launcher exits 40 on an existing run directory):

```
…/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12-round-0-refold/
  run.log            336 KB, the full stage stream
  receipt.json       284 KB, status failed_closed, 8c826a07…
  diagnostics.json   320 KB
  ledger.jsonl       empty            balanced-basis-phase-events.jsonl  empty
  source-commit.txt  53b353f0…        consumed-round-manifest.sha256
```

Deployment: `nf-launch-r12-round-refold.sh.bak-20260805T083111Z`.

The sealed 2026-07-17 pack, the sealed `76ff3cbc…` base cache, `round-000` and
the sealed r12 run directory were never written to.
