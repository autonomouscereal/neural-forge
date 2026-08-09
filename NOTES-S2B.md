# NOTES-S2B — the stage-2 distillation trainer, built and toy-proved

Lane 1 of `docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`. Branch
`codex/lrc-blank-compression`, local HEAD `fec82744`. **Implementation and
CPU-toy validation only. No production training run, no corpus generation, no
GPU touched, no git add, no git commit.** The design cross-review gate is still
the gate; this is the sanctioned preparation behind it.

**The three things this wave learned that the design does not say:**

```
1. 4,096 route coefficients per channel CANNOT be trained.  The routed decode
   runtime refuses to bind unless every target row's protected donor row is an
   exact one-hot at coefficient 1.0.  Training them yields a checkpoint that
   cannot decode at all -- no admission list repairs that.
     -> stage 2 trains 1,929,792 of 1,933,888 coefficients per channel.

2. A trained-route checkpoint is refused by the stage-1 guard at
   "Packaged carrier routes do not have a zero-operation fold audit."
   This is the lane-2 admission handoff, now pinned to a byte-exact message
   and reproduced on demand by a test.

3. The child-local loader authenticates internal consistency, NOT source-
   artifact identity of the coefficients.  A random-init control with frozen
   routes loads through it as if it were a published fold child.  Measured,
   not argued (tests/stage2/test_stage2_checkpoint_layout.py).
```

---

## 1. Module map

All new. No stage-1 file, no frozen collector, and no guard was edited.

| path | what it owns |
|---|---|
| `src/neural_forge/stage2_kd_objective.py` | forward-KL KD, generalized JSD (GKD), donor-vocabulary mask, the objective schedule as a pure function of the token counter |
| `src/neural_forge/stage2_carrier_student.py` | the differentiable routed student: coalesced route layouts, `C_in`/`C_out` in autograd, protected-anchor handling, fold-init vs random-init through one loader, the frozen-runtime equivalence assertion |
| `src/neural_forge/stage2_freerun_probe.py` | the 64-prompt seeded probe, the V5 disjointness screen, greedy free-run decode, token-match/lcp scoring |
| `src/neural_forge/stage2_training_receipt.py` | sealed canonical-JSON receipts, §8 pre-registration digest, corpus-consumption verification, optimizer-state digest |
| `src/neural_forge/stage2_checkpoint_layout.py` | rung checkpoints in the published-child layout, route-package re-seal with a truthful audit, round-trip through the review loader, the admission handoff record |
| `src/neural_forge/stage2_distillation_trainer.py` | the plain torch loop: grad accumulation, bf16 autocast, AdamW + schedule, GKD phase switch, probes, ladder, receipt |
| `experiments/stage2_cpu_toy_validation.py` | the CPU toy: a real sealed carrier child, a frozen toy teacher, a lane-2-shaped corpus manifest, the whole loop end to end |
| `tests/stage2/test_stage2_{kd_objective,carrier_student,freerun_probe,training_receipt,checkpoint_layout,distillation_trainer}.py` | 78 tests |

Key entry points:

* `Stage2DistillationTrainer(config=..., student=..., teacher_donor_logits=..., corpus=..., ...).run()`
  → the sealed receipt dict.
* `load_stage2_routed_student(child_dir, base=..., mode="fold-init"|"random-init", seed=...)`
  → `(student, decode_model, package)`. **One loader, one flag** — both arms
  read the same authenticated published child through
  `Lfm2CarrierChildCheckpointIO.from_child_directory(...).load(...)`, the loader
  the frozen review itself uses; the control then overwrites every dense
  parameter and both coefficient vectors from its recorded seed.
* `write_stage2_rung_checkpoint(...)` / `round_trip_stage2_rung_checkpoint(...)`.

---

## 2. Design deviations, each with its reason

**D1 — file layout.** The design (§9/§10) puts lane 1 in
`src/neural_forge/stage2/{distill_trainer,kd_objectives,training_receipts,freerun_probe}.py`.
The task specified flat `src/neural_forge/stage2_*.py`, which is also the
repository's existing convention (no `src/neural_forge/` subpackages exist).
Tests **are** under `tests/stage2/` as the design says, matching where Codex
put `test_stage2_generate_teacher_corpus.py`. There is one extra module beyond
the design's four (`stage2_carrier_student.py`, `stage2_checkpoint_layout.py`)
because the differentiable route surface and the checkpoint layout are each
large enough to test on their own.

**D2 — protected route anchors are not trainable (load-bearing).** §5.4 item 1
says the controls carry "both k32 route tensors as ordinary trainable
parameters". That cannot be taken literally.
`hf_lfm2_semantic_step_token_fold._validated_sparse_semantic_code` refuses to
bind a routed runtime unless, for every target row `t`, the protected donor row
`target_to_donor[t]` is an exact one-hot at target `t` with coefficient exactly
`1.0` — the strict bridge's 4,096 injective exact anchors. A checkpoint whose
optimizer moved one of them cannot decode at all, in any arm, for any admission
list. So `Stage2RoutedStudent` locates those positions
(`protected_anchor_edge_positions`) and holds them fixed with **two**
independent guards, because one is not enough:

* a gradient hook zeroing anchor gradients (keeps them out of Adam's moments), and
* a re-pin after every `optimizer.step()`, because AdamW's *decoupled* weight
  decay moves parameters that received no gradient at all.

The receipt records `protected_anchor_coefficient_elements` and
`trainable_route_coefficient_elements` so the counted-x claim stays honest:
counted route state is still 3,867,776; trainable route state is 3,859,584.

**D3 — 10M rung is 611 steps, not 610.** §3.1 states 610 / 6,104 / 61,036.
16,384 does not divide 10,000,000: 610 steps consume 9,994,240 tokens, i.e. the
rung would underspend its own budget. §5.2 says "the token budgets are exact
regardless", so `steps_for_tokens` rounds **up**; that reproduces 6,104 and
61,036 exactly and moves only the 10M rung. `STAGE2_DESIGN_STATED_RUNG_STEPS`
keeps the design's tuple visible in the receipt, and a test pins the difference.

**D4 — KD temperature.** The design names no separate distillation temperature;
its only temperature is the corpus sampler's 1.0, and §2.4 justifies forward KL
by that unbiasedness. `STAGE2_KD_TEMPERATURE = 1.0`, with the standard `T²`
gradient rescale carried explicitly (identity at 1.0) so a later sweep is one
constant.

**D5 — on-policy step selection is arithmetic, not a coin.** §2.4 gives a
*fraction* (50% after 10M tokens); a fraction has to become a concrete sequence
of steps. A sampled coin would let an arm and its control diverge in objective
mix under one seed, which §4 forbids ("identical objective mix and schedule").
The rule is `floor((k+1)f) > floor(k f)` over steps since the switch: exact
quota, no RNG, no drift. Measured at 0.25/0.5/0.75 in tests.

**D6 — on-policy sequence composition.** §2.4 says "sample a 256-token
continuation from the student on corpus prefixes" without naming the prefix
length. An on-policy step uses a `sequence_length − 256` corpus prefix plus a
256-token student sample, so it consumes exactly the same 16,384 tokens as an
off-policy step and the design's rung arithmetic stays exact. The GKD loss is
taken on the distributions that emitted the sampled tokens (positions
`prefix−1 … L−1`); the off-policy branch takes forward KL on all `L` positions.

**D7 — GKD endpoint semantics, corrected.** The generalized JSD in Agarwal et
al. **does not** reduce to forward or reverse KL at β=1 / β=0 — the raw form
vanishes at both endpoints. My first draft claimed it did; a test caught it.
The implementation is Agarwal's form with teacher `P`, student `Q`,
`M = βP + (1−β)Q`, and it is verified at β=0.5 against plain probability
algebra and bounded by `ln 2`. The wrong-limit claim is now pinned as a test so
nobody "fixes" it back.

**D8 — weight decay applies to every parameter in one group**, literally as
§3.1 pre-registers it, rather than the usual no-decay-on-norms split. Literal
beats conventional when the value is pre-registered.

**D9 — a checkpoint's route audit is written truthfully, so the design's
default arm is refused today.** See §3.

**D10 — corpus manifest interface.** Lane 2's manifest (Codex's
`experiments/stage2_generate_teacher_corpus.py`, landed during this wave) seals
shards by `shard_index`/`tokens_sha256` with the teacher under
`run.teacher_shard_sha256` — not the `relative_path`/`sha256` shape lane 1
first assumed. `corpus_manifest_shard_index` / `corpus_manifest_teacher_sha256`
read lane 2's actual shape (and a flat shape), and the toy writes lane 2's
shape so the cross-lane interface is exercised rather than invented.

---

## 3. The checkpoint ladder and the one thing it cannot do yet

A rung checkpoint is a complete carrier child directory —
`rung-<tokens>/child/{config.json, model.safetensors, tokenizer files,
semantic-carrier-routes/, semantic-source-lexicon/,
carrier-child-package-receipt.json}` — plus a sealed
`stage2-checkpoint-manifest.json` with every file digest and an optional
optimizer state file beside it. The outer child receipt's sealed inventory is
built by **the verifier's own** `_child_inventory`, so the writer cannot drift
from the loader.

Two cases, and the split is the point:

| routes | route `operation_audit` | stage-1 loader |
|---|---|---|
| frozen (`train_routes=False`, coefficients bit-identical to init) | truthfully zero | **loads end-to-end** |
| trained (design default) | truthfully non-zero | **refuses**: `Packaged carrier routes do not have a zero-operation fold audit.` |

Writing zeros for a trained checkpoint would be a lie about trained state, so
the writer refuses to: a frozen declaration whose coefficients moved raises
`Route coefficients moved in a run that declared frozen routes; a
zero-operation route audit would be false.`

The frozen-route case is what makes "published-child-compatible" a *measured*
claim: all three toy rung checkpoints re-open through
`Lfm2CarrierChildCheckpointIO.from_child_directory(...).load(...)` with
`semantic_step_generate` installed and the outer package sha matching the
manifest. The trained-route case is the lane-2 handoff.

Every stage-2 checkpoint is labelled `stage2_checkpoint: true` with its arm,
rung, step counts and `init_source_sha256` inside **both** sealed receipts
(outer child receipt and route receipt), because of finding 3: the child-local
loader cannot tell a control from a published fold child on its own.

---

## 4. Toy numbers (CPU, 240 steps, fold-init, frozen routes)

`experiments/stage2_cpu_toy_validation.py --workspace <dir> --steps 240 --routes frozen`
— 6.2 s wall for the loop, ~55 s including fixture build and three checkpoint
round trips. Toy scale: 24 donor rows, 12 target rows, hidden 16, 4 seqs × 12
tokens = 48 tokens/step, 1,768 counted parameters.

```
loss, split by branch (the two are NOT on one scale -- forward KL is unbounded,
the beta=0.5 JSD is bounded by ln 2 -- so a mixed curve alternates by design):

  forward-kl            34 logged points   first-8 mean 0.19342 -> last-8 mean 0.17011
  generalized-jsd-gkd   15 logged points   first-3 mean 0.04127 -> last-3 mean 0.03554
  mixed window means                       0.19232 -> 0.09564
  gradient norm                            0.1641 -> 0.0153

GKD phase switch
  switch threshold      4,800 tokens  (= 100 steps x 48 tokens)
  observed switch       step 100                         <- exact
  post-switch steps     140    on-policy 70    fraction  0.500000   <- exact

ladder (rung tokens, steps, on/off-policy steps in that rung)
  2,880    60 steps    0 on / 60 off
  6,912   144 steps   22 on / 62 off
 11,520   240 steps   48 on / 48 off
  tokens_consumed == optimizer_steps x tokens_per_step at every rung

free-run probe, 9 points, 4 prompts x 6 tokens, greedy both models
  token_match_fraction  0.0417 -> 0.125 -> 0.0833 (toy teacher is random; the
  mean_longest_common_prefix stays 0.0)             probe runs, it does not learn

receipts
  receipt_sha256        5e9964db9c998b771c7060b3d87a22bcc16d061d33618350174c2e39e3ff5264
  reload matches seal   true
  one-field tamper      refused: "Stage-2 receipt self-seal does not verify."
  corpus manifest       verified (2 shards, teacher matches manifest)
  gpu_hours             0.001718611 (measured wall x device count, not estimated)

route surface equivalence (before the first optimizer step)
  input  route bitwise identical to the frozen runtime : true
  output route bitwise identical to the frozen runtime : true
  protected anchors intact                             : true

checkpoint round trips (frozen routes)
  rung 1/2/3 -> loaded_through_review_loader = true, semantic_step_generate installed
```

All four cells were run: `{fold-init, random-init} × {frozen, trained} routes`.
Loss decreases in all four; phase switch and 0.5 quota are identical in all
four; round trips load in the two frozen cells and are refused at the pinned
message in the two trained cells.

---

## 5. Verification — what was run

```
.venv/Scripts/python.exe -m pytest tests/stage2 -q
  101 passed  (78 mine + 23 Codex's corpus tests)

.venv/Scripts/python.exe -m pytest tests/stage2 \
  tests/test_lfm2_carrier_child_package_fresh_process.py \
  tests/test_lfm2_carrier_portable_runtime.py \
  tests/test_lfm2_carrier_review_cli.py \
  tests/test_hf_lfm2_semantic_step_token_fold.py \
  tests/test_hf_exact_candidate_frontier.py \
  tests/test_dynamic_counted_child_state.py -q
  202 passed in 68.53s

.venv/Scripts/python.exe -m ruff check src/neural_forge/stage2_*.py \
  experiments/stage2_cpu_toy_validation.py tests/stage2/test_stage2_*.py
  All checks passed!   (also clean under --select E501 --line-length 88)
```

Runtime is the repository's own `.venv` (Python 3.11.9, torch 2.13.0+cpu,
transformers 5.13.0) because the suite needs torch; `C:/Python314/python.exe`
has no torch.

Checks worth naming individually:

* `test_trainable_route_surface_is_bitwise_identical_to_the_decode_runtime` —
  `torch.equal` against `Lfm2SemanticStepRoutingRuntime` through its dual-table
  session pair, both channels. Its negative twin moves one coefficient and
  requires the check to fail.
* `test_the_design_and_its_preregistration_hash_to_their_sealed_digests` —
  §8 = `70453e10f347b454443210ca26911e2e21a95c0c993dd6f1df02a1468d95d3d3`
  (3,337 bytes); design document =
  `96b642cb4eebe7671228801ea98fb722cad9c913af7f63bbc264de8b15b24a04`. An edited
  §8 is refused.
* `test_the_frozen_review_screen_catches_a_planted_overlap` — plants a real V5
  prompt into a copy of the fit-prompt pack; the probe loader refuses it.
* `test_no_stage2_module_defines_a_disqualified_proxy_metric` — AST scan of
  every `stage2_*.py`: no function, class, argument, name or attribute matches
  `teacher-forced | terminal-reconstruct | relative-rmse | rel-l2 | top-1`.
  Its companion walks the sealed toy receipt and requires that every key
  matching those patterns ends in `_used` and is `False`.

**Not verified.** Nothing here has met the real LFM2.5-230M teacher, the real
64,402-row donor vocabulary, the real 1,933,888-coefficient routes, a real
corpus shard, bf16 autocast on a real 3090, or a real 167M-parameter body. The
toy is 1,768 parameters. Throughput, memory, and the sampling cost in §6 are
unmeasured. The `Stage2HuggingFaceTeacher`-shaped adapter was deliberately
**not** written: the teacher is supplied as a plain
`[batch, length] -> [batch, length, rows]` callable, so the real adapter is one
short function written against the real checkpoint rather than a guess.

---

## 6. Named handoffs

**H1 — lane 2 / `routed_decode_admission.py` (needs Cyril's approval).**
Trained-route stage-2 checkpoints need the §5.4 item 3 admission path.
`STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF` in
`src/neural_forge/stage2_checkpoint_layout.py` carries the machine-readable
record: the guard
(`lfm2_carrier_bridge_publication.load_packaged_lfm2_carrier_routes`), the
byte-exact refusal, the key material an allowlist should use
(`stage2_checkpoint_manifest_sha256`, `child_package_sha256`, `config_sha256`,
`arm_id`, `rung_tokens`), and the standing rule that the stage-1 guard, its
refusal messages and its tests stay untouched. **I did not edit the guard or
the carrier admission list.**

**H2 — the admission list must key on labels, not on the loader.** The
child-local fresh-process loader authenticates internal consistency, not the
source-artifact identity of the coefficients, so a frozen-route control opens
through it. Test:
`test_every_stage2_checkpoint_labels_itself_in_both_sealed_receipts`.

**H3 — protected anchors are a design correction, not an implementation
detail.** §5.4 item 1's "both k32 route tensors as trainable parameters" should
read "…minus the 4,096 protected injective anchors per channel, which the
routed decode requires to remain exact one-hot at 1.0". Any lane that
re-implements route training must reproduce both guards (gradient mask *and*
post-step re-pin) or its checkpoints will not decode.

**H4 — on-policy sampling cost is unmeasured and probably dominant.**
`_sample_on_policy_sequence` re-runs the full student forward per generated
token with no KV cache: 256 forwards over a growing context per on-policy step.
The design's calibration run (§9 step 3) must measure the on-policy step
separately from the off-policy step, and a cached decode path for the sampler
is the obvious optimization. Nothing about correctness changes; the tok/s
estimate in §5.2 almost certainly does.

**H5 — corpus interface, if lane 2's manifest moves.** Lane 1 reads
`shards[*].{shard_index|relative_path}`, `shards[*].{tokens_sha256|sha256}` and
`run.teacher_shard_sha256` (or a flat `teacher_checkpoint_sha256`), and derives
shard paths from `STAGE2_CORPUS_SHARD_PATH_TEMPLATE =
"shards/shard-{index:05d}.tokens.u16"`. A `Stage2CorpusSource` supplies batches
plus per-batch shard records; lane 2's real loader only has to satisfy that
protocol.

**H6 — the free-run probe needs a donor tokenizer.** The trainer takes
`tokenize: str -> Sequence[int]`; the toy substitutes a content hash. The real
run passes the donor tokenizer, and the probe's prompt-to-id mapping should be
recorded in the receipt (currently only the prompt ids and `probe_sha256` are).

**H7 — reporting caveat.** The mixed-phase loss series is two objectives on two
scales. Report the branches separately (the receipt's per-point `objective` and
`on_policy` fields make that a one-line split) or the curve reads as noise.

---

## 7. If Codex takes this lane

Everything is untracked and self-contained. `git status` shows my six
`src/neural_forge/stage2_*.py`, `experiments/stage2_cpu_toy_validation.py`, six
`tests/stage2/test_stage2_*.py`, and this file. No tracked file was modified by
this lane.

Two tracked files **were** modified during this session by another agent
working in the same tree — `src/neural_forge/hf_lfm2_macro_cell_attractor_gate.py`
and `tests/test_hf_lfm2_macro_cell_attractor_gate.py` (mtimes 21:40–21:42). They
are not mine and are not part of stage 2; do not attribute them here.

Start points for a takeover:

1. `experiments/stage2_cpu_toy_validation.py::run_toy_validation` is the whole
   wiring diagram in one function — teacher, student, corpus, probe, objective,
   checkpoints, receipt.
2. To go to real scale, replace three things: the base checkpoint IO
   (`Lfm2DirectLocalCheckpointIO` instead of `_ToyBaseCheckpointIO`), the
   teacher callable (LFM2.5-230M forward, sliced to the 64,402 defined rows),
   and the corpus source (lane 2's loader satisfying `Stage2CorpusSource`).
   Nothing in `src/neural_forge/stage2_*.py` has toy assumptions in it.
3. Before any production launch: H1 must land, H4 must be calibrated, and
   `assert_preregistration_unedited` must be called (the trainer takes its
   result as a required argument, so a run cannot start without it).
