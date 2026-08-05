# NOTES-W12E — r12 on-policy round: fix deployed, base cache recaptured, round sealed

Wave 12E, branch `codex/lrc-blank-compression`. This executes the coordinator's
decision on `NOTES-W12D` §5 — **option 1, recapture the base cache under the
current implementation identity** — and relaunches the round.

**Outcome: the round sealed its manifest and its 115-batch extension cache.
The runner's *receipt* did not get written, for a reason that has nothing to do
with the evidence (§5). No gate was weakened, and nothing was rerun to make a
number look better.**

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`.

---

## 1. What was deployed

Local `HEAD` = `13f711323e923b683347de1dfd315dfe4547f924`
(*Screen cached-decode admissibility instead of dying on bf16 physics*).

The server cannot reach the unpushed branch, so the deployment was updated from
a git bundle:

```
bundle  8369f42e3f57acafa6b1dbfbde2e78274a704881..codex/lrc-blank-compression
size    843,861,711 bytes
sha256  b21bb92ad5a7f89993af8131c16f54c1949d5f7ec6255bcb0d11c851d4cc7c66   (identical both sides)
server  /home/cereal/nf-w12e/nf-w12d.bundle
```

`git -C <runtime> fetch <bundle> refs/heads/codex/lrc-blank-compression:...`
then `git reset --hard 13f71132`. The debugger's dirty copy and its
`.w12d-backup` were **moved, not deleted**, to
`/home/cereal/nf-w12e/deploy-backup-20260805T041450Z/` together with
`HEAD-before.txt` and `status-before.txt`.

Deployment `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`:

| | |
|---|---|
| `git rev-parse HEAD` | `13f711323e923b683347de1dfd315dfe4547f924` |
| `git status --porcelain` | **empty** (re-checked after the round; still empty) |

File-for-file against the local committed blobs at `13f71132`:

| file | sha256 | |
|---|---|---|
| `src/neural_forge/hf_lfm2_circuit_trace_cache.py` | `16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945` | identical |
| `src/neural_forge/hf_lfm2_canonical.py` | `789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d` | identical |
| `src/neural_forge/hf_lfm2_on_policy_evidence_round.py` | `a3bf6d1f3e2f7bf88137b980918ba945a43ef35dd565521c138726ad64e24b86` | identical |
| `experiments/run_r12_on_policy_round.py` | `febc4bf4b769d3bc11ddb72386ab874375ba6f71590f9362b9c640b7c6998ed1` | identical |

**The collector invariant holds.** `hf_lfm2_circuit_trace_cache.py` is byte-identical
at `8369f42e` and at `13f71132` (`16271526dd6f…` in both), so local HEAD never
touched it and no already-sealed cache was retired by the deployment step.
`hf_lfm2_canonical.py` likewise did not move (`789c6a6f…`).

Runtime implementation identity, recomputed rather than trusted from notes:

```
collector_source_sha256            16271526dd6fb415b480d533f9305cf591c3e29e2056f7a7681790579cea0945
canonical_contract_source_sha256   789c6a6fdfca948f76a40c738d2690f29c735b85a4f4598ad80d9e4c5f13143d
transformers_lfm2_source_sha256    2ec8be8af6b65482f54d29e52da9765b863ce37b65846423dfbfda72f944f150
torch 2.11.0+cu128  transformers 5.12.1  safetensors 0.8.0
implementation_sha256              732ca9ce7bd135d1e14d99fdc55e3aa9b1b40ad68c3e5a9d6c1dd2842b35c227
```

which is exactly what `NOTES-W12D` §4.2 predicted.

### 1.1 Local `HEAD` moved during the run — harmlessly

Another agent landed seven commits on `codex/lrc-blank-compression` while this
was running; local `HEAD` is now `c0ce20f7b67353a65bc000ca42c760945411416d`
(*Checkpoint the immutable V5 review deployment*). `13f71132` is still an
ancestor, and — checked, not assumed — none of the four files that decide this
lane's identity moved:

```
c0ce20f7:src/neural_forge/hf_lfm2_circuit_trace_cache.py        16271526dd6fb415…  unchanged
c0ce20f7:src/neural_forge/hf_lfm2_canonical.py                  789c6a6fdfca948f…  unchanged
c0ce20f7:src/neural_forge/hf_lfm2_on_policy_evidence_round.py   a3bf6d1f3e2f7bf8…  unchanged
c0ce20f7:experiments/run_r12_on_policy_round.py                 febc4bf4b769d3bc…  unchanged
```

So a later runtime update to `c0ce20f7` (or beyond) still computes
`implementation_sha256 732ca9ce…`, and the recaptured base cache and the sealed
round stay consumable. That holds only while those two source files stay
untouched; any edit to either retires both caches, exactly as `NOTES-W12D` §4.2
established.

---

## 2. Recapturing the base cache

### 2.1 Which builder, and why not that one verbatim

The sealed base cache is
`…/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc…`,
85 GiB, 256 batches, schema `neural-forge-lfm2-parent-circuit-trace-cache-v1`.

Its builder is **`experiments/prepare_lfm2_tokenizer_bridge_evidence.py`**: it
writes `parent_trace_cache/` under the pack root (`PARENT_CACHE_ROOT_NAME`), and
its `prepare_bridge_evidence()` constructs the `Lfm2CircuitTraceCaptureConfig`
and calls `build_lfm2_circuit_trace_cache` with exactly the manifest keys the
sealed cache carries. Every other file in that pack is one of its named outputs.

It was **not** re-run, and that is deliberate. Re-running it re-derives the
corpus from the trajectory texts, the bridge artifact and the tokenizers, and
then does 256 parent generations behind retention gates. That is a rebuild of
the evidence, not a recapture of it: any drift in those inputs would silently
change what the base corpus *is*, which is the one thing option 1 must not do.

`experiments/recache_lfm2_complete_causal_evidence.py` is the right *shape* —
it reads a sealed cache's own authenticated input tensors and recaptures them —
but it hardcodes `sample_last_tokens = maximum_full_prefix_tokens` (264 here)
where the sealed selection is 256, and `_validate_complete_manifest` insists on
that widening. It therefore cannot reproduce `selection_sha256 f229e5f3…`, which
the launcher preflight, the round's comparable-capture clause and this task all
require. Using it would have changed the capture contract.

### 2.2 What was actually run

`/home/cereal/nf-w12e/recapture_r12_base_cache.py` (driver, kept **outside** the
deployment because the launcher exits 41 on any untracked file in the runtime).
It is `recache_lfm2_complete_causal_evidence.py`'s operation with the sealed
selection preserved, and it borrows the committed helpers rather than
reimplementing them:

- `experiments.recache_lfm2_complete_causal_evidence._source_input_batches` —
  authenticated selective session over the sealed cache, input tensors only,
  zero parent forwards against it, nothing written inside it;
- `experiments.run_r12_on_policy_round._capture_config_from_sealed_manifest` —
  rebuilds the selection from the sealed manifest and refuses unless the
  rebuild reproduces the sealed digest;
- `…_load_parent_matching_fingerprint` and `…_bind_sealed_execution_identity` —
  the same dtype search and execution binding the round itself performs.

A `--dry-run` proved every sealed identity was reproducible before 85 GiB was
committed to disk.

Command (env is the launcher's, verbatim — same `CUDA_VISIBLE_DEVICES` UUID
pinning, `CUBLAS_WORKSPACE_CONFIG`, offline flags):

```
/home/cereal/nf-w12e/run_recapture.sh          # log: /home/cereal/nf-w12e/recapture.log
                                               # report: /home/cereal/nf-w12e/recapture-report.json
```

Result: **85 GiB, 4608 parent forwards, 4469.6 s (74.5 min), 0 generations,
0 gradients, 0 optimizer steps, 0 network calls.** The recapture passed the
identical decode gate on all 256 base batches — the base regime never trips it
(`NOTES-W12D`: base p99 0.359, max 0.5625), so no screening of the BASE capture
was needed or applied.

New cache:

```
/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/
  a0023-exact4x-h816-mgv3-r12/base-cache-recaptured-w12e/
  c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7
```

The sealed 2026-07-17 cache was **not** written to, moved, or deleted.

### 2.3 Old vs new, field by field

Full machine-checked comparison in
`/home/cereal/nf-w12e/recapture-report.json`. Summary:

| block / field | sealed `76ff3cbc…` | recaptured `c29b6737…` | |
|---|---|---|---|
| `cache_key` | `76ff3cbc946ea52b…` | `c29b67370d5e33be…` | DIFF *(content-addressed; the implementation digest is part of the address)* |
| `manifest_sha256` | `5cef56936c8d407d…` | `2fe12a313a83c7f7…` | DIFF *(follows the above)* |
| **`corpus.corpus_sha256`** | `71884339978c7261…` | `71884339978c7261…` | **SAME** |
| `corpus.batch_count` | 256 | 256 | SAME |
| `corpus.ordered_batch_input_sha256` | 256 entries | 256 entries | SAME (list equal) |
| `corpus.prompt_token_counts` | 256 entries | 256 entries | SAME |
| `corpus.forced_decode_token_counts` | 256 entries | 256 entries | SAME |
| **`capture.selection_sha256`** | `f229e5f399858b7a…` | `f229e5f399858b7a…` | **SAME** |
| `capture.sample_last_tokens` | 256 | 256 | SAME |
| `capture.max_decode_steps` | 8 | 8 | SAME |
| `capture.max_capture_bytes` | 536870912 | 536870912 | SAME |
| `capture.logit_rtol` / `logit_atol` | 0.2 / 1.0 | 0.2 / 1.0 | SAME |
| `capture.require_top1_agreement` | true | true | SAME |
| `capture.allow_top1_disagreement_within_numeric_error_envelope` | true | true | SAME |
| `capture.cached_prefill_use_cache` / `forced_decode_use_cache` | true / true | true / true | SAME |
| `capture.full_prefix_verification_use_cache` / `prefill_use_cache` | false / false | false / false | SAME |
| `capture.capture_causal_circuit_evidence` + the 5 causal flags | all true | all true | SAME |
| `capture.inference_mode` / `model_eval_mode` / `gradients_used` / `training_steps` | true / true / false / 0 | same | SAME |
| **`parent.identity_sha256`** | `96a6779d927a8aca…` | `96a6779d927a8aca…` | **SAME** (bf16) |
| **`execution.execution_sha256`** | `3d077b94a3f7a490…` | `3d077b94a3f7a490…` | **SAME** |
| `execution.attention_backend` / `input_device_type` | eager / cuda | eager / cuda | SAME |
| `execution.fast_short_conv_available` / `_selected` | false / false | false / false | SAME |
| **`layer_topology.layer_topology_sha256`** | `92166d4d1dd6afe7…` | `92166d4d1dd6afe7…` | **SAME** |
| `implementation.torch_version` | 2.11.0+cu128 | 2.11.0+cu128 | SAME |
| `implementation.transformers_version` | 5.12.1 | 5.12.1 | SAME |
| `implementation.transformers_lfm2_source_sha256` | `2ec8be8af6b65482…` | `2ec8be8af6b65482…` | SAME |
| `implementation.safetensors_version` | 0.8.0 | 0.8.0 | SAME |
| **`implementation.collector_source_sha256`** | `71e99ee9b1c22ae8…` | `16271526dd6fb415…` | **DIFF — the point** |
| **`implementation.canonical_contract_source_sha256`** | `5956f90939ab57dc…` | `789c6a6fdfca948f…` | **DIFF — the point** |
| **`implementation.implementation_sha256`** | `eb3bddeb3414c80b…` | `732ca9ce7bd135d1…` | **DIFF — the point** |

Authenticated extraction receipt: `session_sha256 00271f4eea02eeb0…`.
So: same corpus, same selection, same parent, same execution, same topology.
Only the three implementation digests moved, and they moved to the runtime's own.

---

## 3. Launcher changes

`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`
Backup: `nf-launch-r12-round-refold.sh.bak-20260805T042559Z`
(pre-edit sha256 `b67f905f8fdf803c627d4e909c08ac123b6c248efca6e4e1ac94de990c7fa714`,
post-edit `8d2e5a4d5b2c51c83a41f0aa219333fb5e11356ff05a3327c52ce5df234f7bd5`).
`bash -n` clean locally and on the server.

1. **New variable, not a repoint.**

   ```
   ROUND_BASE_CACHE=$ROUND_ROOT/base-cache-recaptured-w12e/c29b6737…
   ```

   `BASE_CACHE` is left alone. The round's `--base-cache-directory` now uses
   `ROUND_BASE_CACHE`; the refold's `--sealed-parent-cache-directory` still uses
   `BASE_CACHE`. See §6 — repointing the refold would also require resealing
   `canonical_parent_outputs_bridge_bound.json`, whose sealed body names
   `parent_cache_key 76ff3cbc…`, and that is a coordinator call.

2. **The screen is on.** After `--truncate-forced-decode-to-capture-budget`:

   ```
       --screen-cached-decode-admissibility \
       --min-cached-decode-admissible-batches 100 \
   ```

3. **Preflight now reads both caches** and refuses if their
   `(corpus_sha256, parent identity)` pair differs, which is the cheapest place
   to catch a recapture that is not actually a copy. It also prints each
   cache's `implementation_sha256`.

4. **`SOURCE_COMMIT`** has no hardcoded default in this script — it is
   `: "${SOURCE_COMMIT:?…}"`, required from the environment. Nothing to update;
   the reviewed commit `13f71132…` is recorded in the header instead.

5. Header documents both W12E changes and the refold caveat.

`round-000` (the failed round holding the generation cache) was **moved, not
deleted**, to `round-000.failed-implementation-042559`, and the about-to-be-
clobbered `round.log`, `round.pid` and `source-commit.txt` were copied into it.

---

## 4. Relaunch

```
preflight  SOURCE_COMMIT=13f711323e923b683347de1dfd315dfe4547f924 \
           bash ./nf-launch-r12-round-refold.sh preflight        -> exit 0
round      SOURCE_COMMIT=13f711323e923b683347de1dfd315dfe4547f924 \
           bash ./nf-launch-r12-round-refold.sh round            -> PID 566823
```

Timeline (UTC):

| | |
|---|---|
| 04:14:50 | deployment dirty state backed up; reset to `13f71132`, tree clean |
| 04:20:00 – 04:24:38 | recapture dry run: every sealed identity reproducible |
| 04:25:18 – 04:29:52 | authenticated extraction of the 256-batch corpus |
| 04:25:59 | launcher backed up, failed `round-000` moved aside (concurrent) |
| 04:29:52 – 05:44:22 | base recapture, 4608 parent forwards, 85 GiB, 4469.6 s |
| ~05:45 | `recapture-report.json` written, all identities verified |
| 05:45:29 | edited launcher installed, `bash -n` clean |
| 05:45 | `preflight` stage exit 0 on both caches |
| 05:46 | round launched, PID 566823 |
| 05:47 | rollout done, `generation-cache.json` written |
| ~05:50 | admissibility screen done, extension capture starts |
| 05:59 | 115th batch captured, extension cache sealing |
| 06:02 | `round-manifest.json` sealed |
| 06:02 | runner exits 1 writing its receipt (§5) |

### What the round proves

```
round-000/round-manifest.json
  sealed manifest_sha256 (canonical body)  75a690219ad8821fbef7f1ffc8dc66611ece0f5f71c57bdbc8123b760b76433c
  file sha256                              27095c8f6ab5aec20cd0f5e4c02ebb990681162c194f794465246025a13a1c78
  request_sha256                           (see manifest)   rollout_sha256 0099bb61f0e77ac50137cfc80b85374ab1ef9bed97e593377daa7a7c819234f9
extension cache  084ab7dc8914097d04d0ab4ec0da8db75fa909db9f0b2269055c33692e11a561   115 batches
base cache       c29b67370d5e33be47a11daccb1e6c2144d6e85efb024621458dc14ca5bd5ef7   256 batches
```

Independently re-verified after the fact with
`open_lfm2_circuit_trace_cache(..., verify=True)` on the extension cache:

| | base | extension | |
|---|---|---|---|
| `implementation_sha256` | `732ca9ce7bd135d1…` | `732ca9ce7bd135d1…` | **AGREE** |
| `layer_topology_sha256` | `92166d4d1dd6afe7…` | `92166d4d1dd6afe7…` | **AGREE** |
| `execution_sha256` | `3d077b94a3f7a490…` | `3d077b94a3f7a490…` | **AGREE** |
| `parent_identity_sha256` | `96a6779d927a8aca…` | `96a6779d927a8aca…` | AGREE |
| `corpus_sha256` | `71884339978c7261…` | `24ea6bc49e22df26…` | distinct, `disjoint_from_base_corpus: true` |

That is the `NOTES-W12D` §5 blocker gone: the extension and base caches agree on
implementation, so one closed-form solve can consume them together.

### The admissibility screen

```
schema      neural-forge-lfm2-cached-decode-admissibility-v1
screened    true
candidates  119      admissible 115      excluded 4      floor 100  -> satisfied
screen_parent_forward_calls  2100
maximum_admissible_max_abs_error  1.1875
```

| excluded (`origin_prompt_indices`) | forced ids | phase | max abs error | violations | truncated |
|---|---|---|---|---|---|
| 0 | `[2,2,2,2,2,816,2,1000]` | decode.001 | 1.90625 | 190 | false |
| 1 | `[1,1,1,1,1,2038,11440,11440]` | decode.001 | 2.0 | 179 | false |
| 110 | `[1,1,1,1,1,1,1,1]` | decode.001 | 1.21875 | 1 | false |
| 113 | `[1,1,1,1,1,1,1,1]` | decode.001 | 1.9375 | 201 | false |

The debugger's survey predicted **capture-batch** indices `[0, 1, 109, 112]`;
the manifest records **fit-prompt** indices `[0, 1, 110, 113]`. These are the same
four trajectories. One prompt, `fit_prompt_index 4`, was rejected upstream with
`target_retokenization_shifted_the_prompt_boundary`, so every batch index past
4 is one lower than its prompt index: 110→109, 113→112, and 0,1 unchanged. The
forced token ids, error magnitudes and violation counts match the survey exactly
(1.90625/190, 2.0/179, 1.21875/1, 1.9375/201). No anomaly.

The round also reports honestly that the parent ran outside the sealed capture:

```
parent.executed_only_inside_sealed_capture            false
parent.executed_in_cached_decode_admissibility_screen true
parent.screen_parent_forward_calls                    2100
```

Counters: 120 child generate calls, 0 parent forwards during rollout, 115 parent
capture batches, 1 capture invocation, 0 training steps, 0 backward calls, 0
optimizer steps, 0 network calls, child and parent state unchanged.
Target preimage sealed for all 115 records under bridge `5c1a452c…`,
`parent_cache_input_mode = target-tokenizer-canonical-then-one-row-mapped-to-donor`.

---

## 5. The one thing that did not complete: the runner receipt

The library sealed the round. The **runner** then died writing its own receipt:

```
on-policy round failed closed: IR payload is not canonical JSON.
```

`$ROUND_ROOT/round-000-runner-receipt.json` **does not exist** and the process
exited 1.

This is not an evidence gate and not a numerical refusal. Reproduced exactly
(`/home/cereal/nf-w12e/probe_receipt.py`, which rebuilds the receipt body from
the sealed round):

```
FAIL cached_decode_admissibility  cause: TypeError: Object of type mappingproxy is not JSON serializable
     -> key 'excluded'          tuple of mappingproxy
FAIL rollout                      cause: TypeError: Object of type mappingproxy is not JSON serializable
     -> key 'rejected_prompts'  (mappingproxy({'fit_prompt_index': 4, 'reason': 'target_retokenization_shifted_the_prompt_boundary'}),)
WHOLE BODY FAILS: DynamicCircuitIRError: IR payload is not canonical JSON.
```

Cause: `load_lfm2_on_policy_evidence_round` returns a **deeply frozen** manifest
(`_deep_freeze`, `hf_lfm2_on_policy_evidence_round.py:232`), so nested rows are
`MappingProxyType`. `run_r12_on_policy_round.run()` thaws only the top level with
`dict(...)` (`experiments/run_r12_on_policy_round.py:393, 406, 407, 413`), and
`canonical_sha256` (`:414`) is a plain `json.dumps`, which refuses
`mappingproxy`.

It is latent, not new: it fires only when a manifest block has a **non-empty
nested list**. This is the first round that ever got far enough to have one —
and it now has two, `rollout.rejected_prompts` (1 rejected prompt) and
`capture.cached_decode_admissibility.excluded` (the screen's 4 exclusions). The
screen did not cause the bug; it made the bug reachable.

**Fix (not applied — no local commits were made, and the deployment must stay
clean for the launcher):** the round module already ships the right helper,
`_deep_thaw` (`hf_lfm2_on_policy_evidence_round.py:242`). In
`experiments/run_r12_on_policy_round.py`, import it alongside the other
round-module symbols and use it in place of `dict(...)` for every
manifest-derived block:

```python
"disjointness": _deep_thaw(manifest["disjointness"]),
"counters": _deep_thaw(manifest["counters"]),
"rollout": _deep_thaw(manifest["rollout"]),
...
body["cached_decode_admissibility"] = _deep_thaw(admissibility)
```

`execution_identity` is already plain but should use it for uniformity. This
changes no sealed digest: `runner_receipt_sha256` covers only the receipt body,
which is a wrapper artifact and is not referenced by the round manifest, the
extension cache, or the refold stage.

**No receipt was fabricated.** A reconstruction would be indistinguishable from
the runner's own output and should not exist until the runner can emit it.

---

## 6. Refold readiness

The refold's own inputs are satisfied: `$ROUND_MANIFEST` exists, seals a
`target_preimage`, and names bridge artifact `5c1a452cbe6bb78c…`, which is what
the refold preflight (`exit 48`) checks. **Do not run it yet** — two things need
a decision first.

1. **Which base cache does the refold consume?** The refold passes
   `--sealed-parent-cache-directory "$BASE_CACHE"` (still the sealed `76ff3cbc…`)
   together with `--cached-parent-outputs $PACK/canonical_parent_outputs_bridge_bound.json`,
   whose sealed body names `parent_cache_key 76ff3cbc…`. The round it will
   consume was sealed against `c29b6737…`. Repointing `BASE_CACHE` at the
   recapture without resealing that envelope will fail closed; leaving it
   means the refold mixes a `76ff3cbc…` base with a `c29b6737…`-keyed round.
   Both caches carry the same corpus and parent, so this is a bookkeeping
   decision about which identity the refold's evidence set claims — it is not
   a numerical one, and it is not mine to make.

2. **The runner receipt** (§5). Worth landing that three-line change before the
   refold, so the next round in this lane produces its receipt.

When both are settled:

```
SOURCE_COMMIT=<commit that carries the §5 receipt fix> \
  bash /home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh refold
```

The runtime must be re-updated from a new bundle and left clean first; the
launcher still exits 41 otherwise. If the refold runs against the runtime as it
stands right now, `SOURCE_COMMIT=13f711323e923b683347de1dfd315dfe4547f924` —
that is what the deployment is at, and it is what
`$ROUND_ROOT/source-commit.txt` records for the sealed round. Moving the runtime
forward to `c0ce20f7` is safe for this lane per §1.1, but re-verify those four
digests on whatever commit is chosen rather than trusting this note.

---

## 7. Artifacts

Server, under `/home/cereal/nf-w12e/` (kept, not cleaned):

| file | what it is |
|---|---|
| `nf-w12d.bundle` | the deployment bundle, sha256 `b21bb92a…` |
| `deploy-backup-20260805T041450Z/` | the debugger's dirty file, `.w12d-backup`, and pre-reset HEAD/status |
| `recapture_r12_base_cache.py` | the recapture driver |
| `run_recapture.sh` | its launcher, with the round's exact environment |
| `recapture.log`, `recapture-report.json` | the recapture run and its field-by-field report |
| `probe_receipt.py` | reproduces the §5 receipt failure from the sealed round |
| `nf-launch-r12-round-refold.sh.new`, `…preinstall` | staged and pre-install copies of the launcher |

Round root:

| path | |
|---|---|
| `base-cache-recaptured-w12e/c29b6737…` | recaptured base cache, 85 GiB, 256 batches |
| `round-000/round-manifest.json` | sealed; `manifest_sha256 75a69021…`, file sha256 `27095c8f…` |
| `round-000/extension-cache/084ab7dc…` | 115 batches, fully verified |
| `round-000/generation-cache.json` | rollout cache, `25362216…` |
| `round-000.failed-implementation-042559/` | the W12D failure, moved aside, with its `round.log` |
| `round-000.failed-truncation-001539/` | the earlier failure, untouched |

Sealed r12 run directory and the sealed `76ff3cbc…` base cache were never
written to. Disk after the round: 345 G free of 1.8 T.

`/home/cereal/nf-round-debug/` (the W12D scratch) is untouched.
