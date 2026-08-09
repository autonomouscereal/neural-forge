# NOTES-S2C — the stage-2 admission list, and the sampler stops re-reading itself

Clears NOTES-S2B's two named handoffs: **H1/H2** (the stage-2 admission path,
keyed on explicit labels) and **H4** (KV-cached on-policy sampling). Branch
`codex/lrc-blank-compression`, HEAD `d10f4877`. **Source and CPU-toy validation
only. No production training run, no GPU touched, no `git add`, no commit.**

Two sentences that matter more than the rest:

```
1. The stage-1 zero-operation audit guard is BYTE-UNTOUCHED.  An unlabelled
   caller opening a trained stage-2 checkpoint still meets
   "Packaged carrier routes do not have a zero-operation fold audit."
   -- proved on the same checkpoint the stage-2 lane admits, in one test.

2. Cached and uncached on-policy sampling emit IDENTICAL token sequences at
   one seed, greedy and sampled, on a real transformers.Lfm2ForCausalLM body
   and on the toy.  At the design's shape (768-token prefix + 256-token
   continuation) that costs 2.9x less wall on the real body, 4.3x on the toy.
```

---

## 1. Handoff 1 — the stage-2 admission list

### 1.1 What was added, and where

| path | what |
|---|---|
| `src/neural_forge/lfm2_carrier_child_package.py:226-455` | the sealed label structure, the four arm labels, the import-time seal, the lookup |
| `src/neural_forge/lfm2_carrier_child_package.py:987-1020` | `Lfm2CarrierChildPackageRegistration.admitted_stage2_checkpoint_plan(profile, label)` |
| `src/neural_forge/stage2_routed_decode_admission.py` | new — the decode lane: the stage-2 twin loader, the binder, `admit_stage2_rung_checkpoint` |
| `src/neural_forge/stage2_checkpoint_layout.py:98-110, 490-548` | the handoff record now names the landed module; checkpoints seal `initialization_mode` + `init_artifact_sha256` |
| `src/neural_forge/stage2_distillation_trainer.py:144-150, 707-709` | `Stage2CheckpointSettings.init_artifact_sha256`, threaded into every rung |
| `tests/stage2/test_stage2_routed_decode_admission.py` | new — 28 tests |

The stage-1 edit is **additive only**, measured rather than asserted:
`diff -u` against the pre-edit copy is **+261 / -0** — not one existing line of
`lfm2_carrier_child_package.py` was removed or rewritten. This is the
`ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES` pattern from NOTES-W13U applied a
second time, deliberately — same shape, same seal-at-import discipline, same
"coherent tuple, not per-field" matching, same identity-not-equality membership.

### 1.2 The label schema

```python
@dataclass(frozen=True)
class Lfm2Stage2CheckpointLabel:
    label_id: str
    arm_id: str                        # S2-A | S2-B | S2-C | S2-D
    initialization_mode: str           # "fold-init" | "random-init"
    init_artifact_sha256: str | None   # the design's declared init artifact
    route_coefficients_trained: bool
    counted_abi_profile_id: str | None # which sealed carrier ABI tuple, if any
```

`ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS` — the complete set, four entries, from
design section 1:

| label_id | arm | mode | init artifact | routes | carrier ABI |
|---|---|---|---|---|---|
| `stage2-s2a-fold-init-trained-routes` | S2-A | fold-init | `54b9c5f7…` (A9/A11 saved child state) | trained | *none* |
| `stage2-s2b-fold-init-trained-routes` | S2-B | fold-init | `f3f38cc3…` (a0028 2x published artifact tree) | trained | `exact-2x-h816` |
| `stage2-s2c-random-init-trained-routes` | S2-C | random-init | *none* | trained | *none* |
| `stage2-s2d-random-init-trained-routes` | S2-D | random-init | *none* | trained | `exact-2x-h816` |

Four seal invariants, enforced at import (`_seal_admitted_lfm2_stage2_checkpoint_labels`):

1. **label ids and arm ids are unique.**
2. **A fold arm must name an init artifact; a control must not.** A control's
   provenance is a recorded seed, not a digest; inventing one would be a false
   provenance claim, and *sharing* a fold arm's digest would erase the only
   thing separating the arm from its control.
3. **Every admitted label declares trained routes.** This lane exists only for
   checkpoints the stage-1 loader truthfully refuses. A frozen-route checkpoint
   already opens through the untouched stage-1 loader, so a frozen-route label
   would be an admission with nothing to admit — and would be a way to reach
   the stage-2 code path for an artifact that never needed it.
4. **A named carrier ABI profile must be in the stage-1 allowlist.**

Two entries name **no** carrier ABI, and that is a real fact, not a placeholder.
S2-A/S2-C run at the A11 publication topology — 166,778,624 physical +
3,867,776 route = 170,646,400 counted, 1.346018x — which is not
`exact-4x-h816` and not `exact-2x-h816`. They therefore cannot reach
`admitted_stage2_checkpoint_plan` at all. Giving them a fake profile id, or
loosening the check to "any admitted profile", would have re-created exactly
the cross-field mixture W13U §2 spent a wave making inexpressible.

### 1.3 The two-part key

A sealed arm label alone admits nothing, and a checkpoint's own digests alone
admit nothing. `admit_stage2_rung_checkpoint` requires both:

```python
admit_stage2_rung_checkpoint(
    rung_directory,
    label=LFM2_STAGE2_S2B_FOLD_LABEL,          # design-time, sealed, by identity
    identity=Stage2CheckpointIdentity(          # run-time, declared by the caller
        manifest_sha256=..., child_package_sha256=..., init_source_sha256=...,
    ),
    base=...,                                   # the checkpoint IO that loads the body
)
```

and then checks, in order: the label **is** one of the sealed objects (identity,
so an equal-valued forgery is refused); the rung manifest re-verifies against
every file digest **and** the declared manifest sha; the declared child-package
and init-source digests are this checkpoint's; the checkpoint's own sealed
stage-2 label block agrees with the presented label on `arm_id`,
`initialization_mode`, `init_artifact_sha256` and `route_coefficients_trained`;
the **outer child receipt's** label block agrees with the manifest's, and with
the presented label, independently. Only then is the route package
authenticated and bound.

The design-time half and the run-time half are separate because they have to
be: the fold arms' init artifacts are known now (they are sealed stage-1
outputs), the controls' are not (a control is a seed, and `init_source_sha256`
is derived per run and contains the run's own child directory). Pretending a
module constant could pin a run-time digest would have made the list either a
lie or a rubber stamp.

### 1.4 The parallel-lane proof

The test that carries the whole claim is
`test_the_same_checkpoint_is_still_refused_at_the_stage1_message`: it admits a
trained S2-B checkpoint through the stage-2 lane, gets a bound routed decode
with `semantic_step_generate` installed, and then — same directory, same bytes,
no label — sends it through both stage-1 entrypoints and requires the byte-exact
original refusal from each:

```
round_trip_stage2_rung_checkpoint(...)  -> refused_by_stage1_guard = True
                                          refusal_message == STAGE1_TRAINED_ROUTE_REFUSAL
load_packaged_lfm2_carrier_routes(...)  -> "Packaged carrier routes do not have a
                                            zero-operation fold audit."
```

The other four required proofs:

| requirement | test |
|---|---|
| stage-1 path byte-identical | every pre-existing suite green **unmodified** (§4); no stage-1 message, branch or constant was edited |
| labelled admission accepts the four arm labels | `test_labelled_admission_accepts_each_of_the_four_arms[S2-A…S2-D]` — all four bind a live routed decode |
| unlabelled trained-route checkpoint still refused | `test_the_same_checkpoint_is_still_refused_at_the_stage1_message` |
| label/checkpoint mismatch refused | wrong arm, fold-checkpoint-as-control, control claiming a fold artifact, missing init mode, frozen routes under a trained label, forged label object, and each of the three identity digests — 9 refusal tests |
| a label cannot admit through the stage-1 entrypoint | `test_a_stage2_label_cannot_admit_through_the_stage1_entrypoint` — `admitted_counted_plan(label)` refuses at the *unchanged* message ("Carrier counted plans are sealed to the admitted carrier ABI…"), and `admitted_stage2_checkpoint_plan(profile, profile)` refuses because a profile is not a sealed label |

### 1.5 The one uncomfortable part, stated plainly

`load_packaged_lfm2_carrier_routes` refuses at the audit **before** it
authenticates a single tensor, so there is no way to reach its structural checks
for a trained checkpoint without editing it — which is out of scope by design.
The two alternatives were:

* **Zero the audit** in the checkpoint, or in a temporary copy of it, and let
  the stage-1 loader run. Rejected: that manufactures an artifact whose
  `operation_audit` reads zero over trained state. That is the precise lie the
  stage-1 guard exists to catch, and a temp-dir copy of it is still a
  publishable file if the process dies mid-call.
* **Restate the authentication** in the stage-2 lane. Chosen, with the drift
  risk paid down rather than waved at.

`authenticate_stage2_packaged_carrier_routes` executes the same steps in the
same order, and every shared step is **the stage-1 verifier's own helper**
(`_sealed_json`, `_load_authenticated_tensors`, `_route_allocations`,
`_child_tables`, `_ordinary_file`, `_stable_file_identity`, `_strict_json_bytes`,
plus `_child_inventory`/`_strict_receipt` for the outer receipt), not a
look-alike. So a refusal raised inside a shared step is the stage-1 refusal,
type and message. That is measured, not assumed:

```
test_the_stage2_twin_matches_the_stage1_loader_where_stage1_will_run
  one student written twice -- routes declared trained, and frozen -- so the two
  trees are byte-identical except the audit and the label block.  Stage-1 loader
  on the frozen sibling vs stage-2 twin on the trained one:
    identical tensors (torch.equal, every key), identical allocations,
    identical receipt outside {operation_audit, package_sha256,
    stage2_route_provenance}

test_the_stage2_twin_refuses_exactly_what_the_stage1_loader_refuses
  [tensor-file-bytes | receipt-seal | extra-file | missing-tensor-file]
  applied to both siblings:  type(exc) equal AND str(exc) equal, all four
```

Two properties keep the twin from becoming a general-purpose relaxed loader:

* it refuses any package without a `stage2_route_provenance` block carrying
  `stage2_checkpoint: true` — **it cannot open a stage-1 published child at
  all** (`test_the_stage2_lane_refuses_an_ordinary_stage1_published_child`);
* it relaxes the **three training counters only**, and only to values equal to
  the checkpoint's own sealed `optimizer_steps`/`backward_calls`. The four
  credit counters (pruning, quantization, sparsification, sharing) must still be
  exactly zero — a checkpoint claiming pruning credit is refused
  (`test_a_pruning_or_quantization_credit_claim_is_never_admitted`), and an
  audit that disagrees with the checkpoint's record by one step is refused.

`bind_stage2_packaged_carrier_routes` exists for the same reason: the stage-1
binder deliberately re-opens the package with `load_packaged_lfm2_carrier_routes`
at bind time (so a retained mutable tensor mapping is never trusted), and that
re-open meets the audit guard again. The stage-2 binder keeps the re-open
discipline verbatim, swaps in the twin, and hands the tensors to the *unmodified*
`_bind_lfm2_carrier_runtime`.

---

## 2. Handoff 2 — KV cache for on-policy GKD sampling

### 2.1 The mechanism, taken from the runtime rather than invented

`hf_lfm2_semantic_step_model_runtime` (the deployed routed decode, lines
799-893) prefills the prompt with `use_cache=True`, then advances **one semantic
token per call** with `past_key_values=cache` and an attention mask covering the
whole context so far, and skips feeding the terminal token back because nothing
scores after it. The sampler now does exactly that, same call shape:

| path | what it owns |
|---|---|
| `stage2_carrier_student.py:353-366` | `Stage2CachedDecodeState` — the body's own cache plus the context length the mask is built from |
| `:739-791` | `body_hidden_states_with_cache(...)`, `donor_logits_from_hidden(...)` |
| `:812-874` | `start_cached_decode(prefix)` / `advance_cached_decode(state, token)` |
| `stage2_distillation_trainer.py:427-504` | `_draw(...)` (one selection rule for both paths) and `sample_on_policy_sequence(prefix, greedy=…, use_cache=…)` |
| `stage2_distillation_trainer.py:186-193` | `Stage2TrainerConfig.on_policy_sampling_uses_cache = True` — the uncached path is kept as the correctness reference |

Two savings, not one. The cache removes the O(L) body re-read per token; and
because a sampler only needs the *emitting* position, the cached path takes the
routed `C_out` readout once per token instead of at all `L` positions. At real
scale the second is the larger one: the uncached path was running a sparse mm
over 64,402 donor rows at ~1,023 positions per token that emit nothing.

The forward-call counter is unchanged in meaning — one body advance per emitted
token either way — so `on_policy_sample_forward_calls` in the receipt stays
comparable across the flag.

### 2.2 Correctness pin

`tests/stage2/test_stage2_cached_sampling.py`, 8 tests. The load-bearing one
runs on a **real `transformers.Lfm2ForCausalLM`** — 4 layers,
`layer_types=["conv","conv","full_attention","conv"]`, untied tables — wearing
the toy fixture's real packaged routes. That is where a cache is actually hard,
and it is the only surface on which a claim about "the LFM2 cache" is honest.

```
test_cached_decode_matches_the_uncached_path_on_a_real_lfm2_body[greedy]
test_cached_decode_matches_the_uncached_path_on_a_real_lfm2_body[sampled]
  torch.equal on the full token sequence, batch 2 x (6 prefix + 12 generated)
  max |logit delta| across all steps < 1e-4
  observed 3.7e-08 greedy / 4.8e-08 sampled -- float32 noise, not drift

test_the_real_lfm2_cache_actually_shortens_the_context_it_reads
  the body's forward is watched: inputs_embeds lengths == [6, 1, 1, 1, 1, 1]
  -- one prefill, five single-token advances.  A cache that silently re-read
  the prefix would pass an equality test alone.

test_the_trainers_sampler_is_identical_cached_and_uncached[greedy|sampled]
  same seed, flag off vs flag on, torch.equal; forward-call counts both 8

test_a_body_that_returns_no_cache_is_refused_rather_than_guessed
  a body that ignores use_cache raises rather than silently decoding wrong
```

The toy's own evidence bundle now carries the same equality
(`experiments/stage2_cpu_toy_validation.py::_measure_on_policy_sampling_cache`,
run before the first optimizer step so both rollouts see identical weights):

```
"on_policy_sampling_cache": {
  "prefix_tokens": 256, "continuation_tokens": 64, "sequences": 2,
  "greedy_tokens_identical": true,
  "sampled_tokens_identical": true,
  "uncached_seconds_per_rollout": 0.09996,
  "cached_seconds_per_rollout":   0.06333,
  "speedup": 1.578
}
```

The `_ToyBody` gained a real cache to make this possible: it now accepts
`past_key_values` and returns `last_hidden_state`/`past_key_values`, carrying
the running prefix sum and count its causal prefix-mean needs. Its uncached
branch is unchanged, so the training path is byte-identical to before.

### 2.3 Timing — measured, and the scaling measured too

Wall-clock per rollout, CPU, mean of 3 (uncached vs cached), tokens identical
(greedy and sampled) at every row:

**Toy body** (hidden 16, 24 donor rows, batch 2):

| prefix | continuation | uncached | cached | speedup |
|---|---|---|---|---|
| 64 | 32 | 0.0294 s | 0.0238 s | **1.24x** |
| 256 | 64 | 0.0741 s | 0.0487 s | **1.52x** |
| 768 | 256 | 0.8094 s | 0.1896 s | **4.27x** |

**Real `Lfm2ForCausalLM`** (4 layers, hidden 16, batch 2, mean of 2):

| prefix | continuation | uncached | cached | speedup |
|---|---|---|---|---|
| 64 | 32 | 0.1317 s | 0.1171 s | **1.12x** |
| 256 | 64 | 0.4333 s | 0.2564 s | **1.69x** |
| 768 | 256 | 3.4796 s | 1.1964 s | **2.91x** |

The last row of each table is the design's actual on-policy shape (D6: a
`1024 − 256` prefix plus a 256-token student sample). The speedup rises with
context in both, which is the predicted shape: the uncached cost is
O(continuation x context) and the cached cost is O(continuation).

**These numbers understate the production gain, and by a lot.** Both bodies here
are 16-wide over 24 donor rows, so the readout term — the one that is 64,402
rows wide in production and that the cached path evaluates 1,024x less often —
is essentially absent from the measurement. The honest statement is: *the shape
of the win is measured and the direction is certain; its production magnitude
is still the calibration run's job* (design §9 step 3, H4's original ask).

---

## 3. Files changed

| path | nature |
|---|---|
| `src/neural_forge/lfm2_carrier_child_package.py` | **stage-1, additive only** — the sanctioned admission-list addition + one new method |
| `src/neural_forge/stage2_routed_decode_admission.py` | new |
| `src/neural_forge/stage2_carrier_student.py` | cached decode surface; `body_hidden_states` refactored into shared helpers, same behaviour |
| `src/neural_forge/stage2_distillation_trainer.py` | cached sampler + flag; `init_artifact_sha256` threading |
| `src/neural_forge/stage2_checkpoint_layout.py` | two optional sealed label fields; handoff record updated |
| `src/neural_forge/stage2_training_receipt.py` | design-document digest re-pinned to the amended document (§5) |
| `experiments/stage2_cpu_toy_validation.py` | cache-capable toy body, the sampling-cache measurement, `source_artifact_sha256` in the fixture |
| `tests/stage2/test_stage2_routed_decode_admission.py` | new, 28 tests |
| `tests/stage2/test_stage2_cached_sampling.py` | new, 8 tests |
| `NOTES-S2C.md` | this file |

**No frozen-collector file was touched.** No stage-1 guard, refusal message,
branch or test was modified. Nothing was staged or committed.

Four files in `git status` are **not mine** — another agent is working in the
same tree, as NOTES-S2B §7 warned: `hf_lfm2_cached_global_topology_frontier.py`,
`hf_lfm2_global_topology_phase_compatibility.py` (new),
`tests/test_hf_lfm2_cached_global_topology_frontier.py` and
`tests/test_hf_lfm2_global_topology_phase_cache_exact4x.py`, all with mtimes
02:29-02:30 against this wave's 01:52-02:10. Do not attribute them here.

Pre-edit copies of every file this wave modified are at
`C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\a763fdb3-d614-4455-9c15-15950803b63f\scratchpad\backup-pre-s2c-20260809T015142\`.

---

## 4. Verification — what was run

Runner `.venv/Scripts/python.exe -m pytest … -q -p no:randomly` (Python 3.11.9,
torch 2.13.0+cpu, transformers 5.13.0; `C:/Python314` has no torch).

| suite | result |
|---|---|
| `tests/stage2` | **138 passed** (102 at wave start → +28 admission, +8 cached sampling) |
| `tests/test_lfm2_carrier_child_package_fresh_process.py` | **8 passed** (unmodified) |
| `+ …portable_runtime, …review_cli, …semantic_step_token_fold, …exact_candidate_frontier, …dynamic_counted_child_state` | **238 passed** total with `tests/stage2` |
| `tests/test_run_lfm2_dynamic_catalog.py` + `…_carrier_e2e.py` | **135 passed, 1 skipped** (7m 57s) |

Ruff:

```
ruff check src/neural_forge/ experiments/stage2_cpu_toy_validation.py tests/stage2/
  2 errors, both pre-existing, both in files this wave did not touch
  (hf_causal_kind_translation.py:419 F841, hf_lfm2_weight_complete_basis.py:685 F841)
ruff check --select E501 --line-length 88 <every file this wave touched>
  All checks passed!
```

Toy end to end:

```
.venv/Scripts/python.exe experiments/stage2_cpu_toy_validation.py \
  --workspace <dir> --steps 240 --routes frozen
  loss_decreased true; phase_switch_step 100; post_switch_on_policy_fraction 0.5
  route surface bitwise identical to the frozen runtime (both channels)
  3/3 rung checkpoints re-open through the review loader
  on_policy_sampling_cache: greedy and sampled tokens identical, 1.578x
```

Note the toy's `receipt_sha256` is not comparable to NOTES-S2B's
`5e9964db…`: the receipt seals wall-clock-derived `gpu_hours`, so it differs
per run, and it also seals the design-document digest, which moved (§5).

### The three carrier-e2e reds from NOTES-W13U are gone

`test_run_lfm2_dynamic_catalog_carrier_e2e.py` was 6 passed / 3 failed at W13U
(`Selected parent component payload changed after authenticated load.`). It is
now **9 passed**. Not fixed by this wave — fixed somewhere between `9326a564`
and `d10f4877`. Recorded so nobody re-opens it.

---

## 5. One pre-existing red, found and closed honestly

`test_the_design_and_its_preregistration_hash_to_their_sealed_digests` was
already failing at `d10f4877`, before any edit of mine:

```
assert observed["design_document_sha256"] == STAGE2_DESIGN_DOCUMENT_SHA256
  - 96b642cb4eebe7671228801ea98fb722cad9c913af7f63bbc264de8b15b24a04   (pinned)
  + 4a71d829934c8d07c127bcc09ba7dc59964c96135c0d523ef9c861ce2e0b83f8   (on disk)
```

Cause, verified by `git show 53d31156 -- docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`:
commit `53d31156` ("Generate the stage-2 corpus") appended **AMENDMENT 1**, 20
lines, entirely below section 12. The working tree is clean at that path, so
this is the committed document, not a local edit.

**Section 8 did not move.** The pre-registration digest
`70453e10f347b454443210ca26911e2e21a95c0c993dd6f1df02a1468d95d3d3` and its
3,337-byte length still verify — that assertion passes and is the one section 8
claims for itself. Only the whole-document digest changed, and an amendment
below §8 is a legitimate edit.

So `STAGE2_DESIGN_DOCUMENT_SHA256` was re-pinned to `4a71d829…` with both
digests and their commits recorded in the comment beside it
(`src/neural_forge/stage2_training_receipt.py:51-63`). The pin's purpose is that
a document change is *visible*, not that it is forbidden; silently deleting the
assertion would have removed the visibility, and leaving it red would have
buried the next real one.

---

## 6. Named handoffs, still open

**H1' — Cyril's approval is still the gate, and the adversarial review with it.**
Design §5.4 item 3 and §12 both say the stage-2 admission path "needs Cyril's
explicit approval" and "gets adversarial review". This wave built it behind that
gate; it has not been approved, and
`STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF["needs_cyril_approval"]` is still
`True`. The single highest-value thing for a reviewer to attack is §1.5 — the
restated route authentication in `stage2_routed_decode_admission.py`. The
equivalence tests cover the accepted case and four corruptions; they do not
cover every refusal branch of the 245-line stage-1 original.

**H3 (unchanged) — protected anchors are a design correction.** §5.4 item 1's
"both k32 route tensors as trainable parameters" still needs to read "…minus
the 4,096 protected injective anchors per channel".

**H4' — the calibration run still has to produce the production number.** The
cached sampler is correct and faster; *how much* faster at hidden 816 over
64,402 donor rows with a 230M teacher on a 3090 is unmeasured (§2.3). The
design's §9 step 3 calibration must still time the on-policy step separately
from the off-policy step.

**H8 (new) — the free-run probe still decodes uncached.** `stage2_freerun_probe`
takes a plain model callable and re-runs it per token, 64 prompts x 256 tokens
per probe. It is far cheaper than on-policy sampling (once per 1M tokens, not
once per step) so it was deliberately left alone, but it is the same fix:
`start_cached_decode`/`advance_cached_decode` are already the right interface
for it.

**H9 (new) — the toy has no admitted arm.** The toy writes checkpoints under
`arm_id="S2-TOY"`, which is deliberately not in the sealed label list, so the
toy exercises the *refusal* side end to end and the admission side is proved in
`tests/stage2/test_stage2_routed_decode_admission.py` on toy-built checkpoints
labelled S2-A…S2-D. Nothing at real scale — no real fold child, no
1,933,888-coefficient route channel, no 64,402-row donor vocabulary — has been
through either lane.

**H10 (new) — the label's init artifact is asserted, not derived.** A rung
checkpoint now *records* `init_artifact_sha256`, and admission requires it to
equal the sealed label's. But the value is supplied by the launcher
(`Stage2CheckpointSettings.init_artifact_sha256`); nothing re-derives the
initialization child's artifact tree digest at checkpoint-write time and
compares. A launcher that passes S2-B's digest to an S2-A run produces a
checkpoint that admits under the S2-B label. Closing it means hashing the
initialization child directory at write time — cheap, but it needs the same
artifact-tree hashing rule the publication used, which is not currently exposed
as a standalone helper.
