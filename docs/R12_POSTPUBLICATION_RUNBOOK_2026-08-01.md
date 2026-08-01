# r12 Post-Publication Runbook

Written: 2026-08-01
Staged against: `codex/lrc-blank-compression` at `e893767a`
Run: `a0023-exact4x-h816-mgv3-r12`

This is the ordered sequence to execute the moment r12 publishes. Everything it
needs is already staged; nothing below requires a design decision.

r1 through r11 all failed before publication, so **no exact-4x carrier lane has
ever reached a post-publication step**. Every stage here is first execution at
real scale. Treat each one as unproven and read the receipts.

---

## 0. Identities

| Thing | Value |
|---|---|
| Run directory (`$R12`) | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12` |
| Publication | `$R12/published-child` (standalone child at `$R12/published-child/child`) |
| r12 fold runtime | `/home/cereal/model-surgery-lab/deployments/neural-forge-exact4x-7b48e205` (commit `7b48e205`) |
| Post-publication runtime | `/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a` (detached at this staging commit, whose parent is `e893767a`; the directory is named for the base commit). Read the exact value with `git -C <runtime> rev-parse HEAD` and pass it as `SOURCE_COMMIT`. |
| Reserved review prompts | `research/lfm2_semantic_whole_response_prompts_v1.json`, file SHA-256 `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` |
| On-policy fit corpus | `experiments/lfm2_on_policy_fit_prompts_v1.json`, file SHA-256 `e85000326c5c39ae683506d69bc7e8edf3557845dc3b06e562f5ba6c23fe436f` |
| Fit corpus canonical body seal | `4bf7263dc5a6158c9a0ed4c2692ffd98b7ae4f56d5694dac6ddbd23e3071c452` |
| Round-sealed `prompt_corpus_sha256` | `834a95856f3c50f195e9ee51855b1a7bf0f05edcf80025d412f17b39f8389c26` |
| Reserved-digest manifest | `experiments/lfm2_manual_review_reserved_digests_v1.json`, file SHA-256 `d9f196956c6da0c7c4bf4d538005d17f2242f4c403f31014525f2318b88cf92c` |
| Sealed base parent cache | `$PACK/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95` |
| Base capture selection | `f229e5f399858b7a512ec0d7034fe4b3d76cf4dc3bb214ec461a8a6f7c000258` |
| Base execution identity | `3d077b94a3f7a4907f5280acfdc35284ed6e2ba109b48edf7111c45b74266f2c` (`eager` attention, `cuda` input device) |
| Bridge artifact / profile | `5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384` / `e3fc98c666b70b367962e400435afe7588a9f1a2b13619dffc6f40c6458c5ca1` |
| Staged launcher | `/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh` |

`$PACK` is
`/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5`.

**Two runtimes on purpose.** r12 was folded by `7b48e205`, which predates
`ab3ec99a` (dense/carrier lanes consume sealed evidence sets) and `09825ab2`
(bridged rounds seal the target-token preimage). A round or a refold has to run
from a runtime carrying both. r12's checkout, its run directory, and the shared
venv are never written by anything below; `neural-forge-postpub-e893767a` is a
separate clone and reuses `/home/cereal/model-surgery-lab/venv` read-only.

All remote commands go through the server-manager v2 client:

```
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '<script>'
```

---

## (a) Fresh-reload verification and the routed manual dossier

**The run does this itself.** There is no separate command. `run_lfm2_dynamic_catalog.py`
saves the child, reloads it standalone, re-proves the routed ABI on the reloaded
artifact, and only then generates the manual dossier from that reloaded child.

### Where it lands

| Artifact | Path |
|---|---|
| Routed manual dossier | `$R12/routed-manual-dossier.json` |
| Per-response review checkpoints | `$R12/routed-manual-checkpoints/` |
| Run receipt | `$R12/receipt.json` |
| Diagnostics | `$R12/diagnostics.json` |
| Published child | `$R12/published-child/child/` |
| Progress journal | `$R12/run.log` (one JSON object per line) |

### Confirm the run actually completed the sequence

```
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
R12=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12
/home/cereal/model-surgery-lab/venv/bin/python - <<PY
import json, pathlib
r = json.loads(pathlib.Path("'"$R12"'/receipt.json").read_text())
print("status", r["status"])
art = r["artifact"]
print("fresh_reload_accepted", art["fresh_reload_accepted"])
print("artifact_tree_sha256", art["artifact_tree_sha256"])
print("child_state_fingerprint", art["child_state_fingerprint"])
carrier = r["carrier_routes"]
print("route_smoke_phase_count", carrier["route_smoke_phase_count"])
print("in_memory_stage_and_final_reload_smoke_expected",
      carrier["in_memory_stage_and_final_reload_smoke_expected"])
print("high_cap_manual_dossier_sha256", carrier["high_cap_manual_dossier_sha256"])
print("intelligence_response_count", carrier["intelligence_response_count"])
print("safety_observation_count", carrier["safety_observation_count"])
print("initial_cap_censored", carrier["initial_cap_censored_response_count"])
print("adaptive_retry", carrier["adaptive_retry_response_count"])
print("remaining_cap_censored", carrier["remaining_cap_censored_response_count"])
PY'
```

Expected: `status = complete`, `fresh_reload_accepted = True`, three routed smoke
phases (in-memory, stage reload, final reload), `intelligence_response_count = 15`,
`safety_observation_count = 1`, and `adaptive_retry_response_count` exactly equal
to `initial_cap_censored_response_count` (the runner refuses any other relation).

The dossier's `dossier_sha256` must equal the receipt's
`high_cap_manual_dossier_sha256`. If it does not, stop: the dossier on disk is
not the one the run sealed.

### Pull the dossier local

```
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --download \
  "/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12/routed-manual-dossier.json" \
  "C:/Users/cereal/Documents/ai training/neural-forge/reports/a0023-r12"
```

Verify the local copy's SHA-256 against the receipt before reading it. Read the
complete decoded responses; the dossier holds them under
`intelligence_review_responses` and `safety_observations`, plus a
`response_evaluation_dispositions` row per prompt.

---

## (b) Manual review protocol

Per `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`. The manual
whole-response review is the only thing with verdict authority. Nothing in the
dossier grades the child, and the dossier says so in its own body
(`automatic_intelligence_verdict_performed: false`).

**Count note.** The task framing says "18 manual-review prompts". The corpus r12
actually seals at `0ee6bf14...` holds **16** prompts: 15 folding-quality prompts
plus 1 safety prompt (`safety-household-chemical`), which is why the receipt
reports `intelligence_response_count = 15`. The separate 18-prompt reserved set
is `research/lfm2_causal_support_future_manual_review_v1.json` (file SHA-256
`27b8f88b2e038c74e9f15b372eda0930e01ce24a0a142c2240125ff829c11559`); r12 does not
generate responses for it, and it is reserved for a later review. Both sets are
covered by the disjointness manifest the on-policy round consumes, so neither can
leak into fit evidence.

### Rules that are not negotiable

1. **Observations per dimension, not one verdict.** Record coherence, task
   correctness, reasoning, instruction following, usefulness, context retention,
   retained parent-demonstrated capability, and concrete failure modes
   separately. A single binary verdict has no authority.
2. **Safety is a parallel dimension.** `safety-household-chemical` is recorded in
   its own row. A safety observation is never a folding-quality failure and never
   an intelligence failure. Conversely, a response that violates an explicit task
   constraint (the multi-turn peanut correction is the canonical example) is a
   folding-quality miss and is **not** laundered through the safety dimension.
3. **Cap-censored is not failed.** A response that reached the evaluator cap is
   marked censored/incomplete. The run already applies one 2048-token retry to
   every and only the 1024-token cap hits. Anything still censored after that has
   an **unknown** endpoint. You may still record an observed useful/partial/miss
   prefix when the evidence before the cap supports it. EOS, cap hit, observed
   loop, and runtime error are four separate fields; a cap hit does not imply a
   loop and a loop is not inferred from length. Rerun a still-censored case from
   the same saved child at a larger allowance — never refold, never regenerate
   parent traces.
4. **No numeric threshold decides.** The ~90% retention objective is directional
   and holistic. No aggregate certifies anything.

### Worksheet

One row per prompt. `Disposition` is the dossier's own
`response_evaluation_dispositions[i].intelligence_endpoint_status`; the reviewer
fills everything after it.

| # | Prompt id | Category | Ended | Disposition | Coherence | Task correctness | Reasoning | Instruction following | Context retention | Usefulness | Failure modes observed | Folding-quality observation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `conversation-natural-greeting` | conversation | EOS / cap / error | | | | | | | | | useful / partial / miss |
| 2 | `instruction-three-actions` | instruction-following | | | | | | | | | | |
| 3 | `explanation-seasons` | knowledge-explanation | | | | | | | | | | |
| 4 | `arithmetic-multi-step` | reasoning | | | | | | | | | | |
| 5 | `logic-ordering` | reasoning | | | | | | | | | | |
| 6 | `context-retrieval` | context-use | | | | | | | | | | |
| 7 | `summarization-priorities` | summarization | | | | | | | | | | |
| 8 | `comparison-ram-rom` | conceptual-comparison | | | | | | | | | | |
| 9 | `coding-deduplicate` | coding | | | | | | | | | | |
| 10 | `debugging-off-by-one` | debugging | | | | | | | | | | |
| 11 | `writing-concise-email` | writing | | | | | | | | | | |
| 12 | `planning-dependencies` | planning | | | | | | | | | | |
| 13 | `uncertainty-insufficient-data` | epistemic-reasoning | | | | | | | | | | |
| 14 | `creative-continuity` | creative-writing | | | | | | | | | | |
| 15 | `multi-turn-correction` | multi-turn-conversation | | | | | | | | | | |

Safety, recorded separately and with no folding-quality authority:

| # | Prompt id | Ended | Disposition | Safety observation | Appropriateness | Notes |
|---|---|---|---|---|---|---|
| S1 | `safety-household-chemical` | EOS / cap / error | `excluded-safety-observation` | | | |

Summary block to record alongside the table:

- counts: `_` useful, `_` partial, `_` observed miss (out of 15)
- EOS `_`, still-censored after the 2048 retry `_`, runtime errors `_`
- of the censored rows: `_` useful prefix, `_` partial prefix, `_` already-observed miss
- strengths, regressions versus the A9/A11 same-body review, and hypotheses
- parent-relative note: the cached parent envelope is itself wrong on seasons,
  `94 / 5`, the ordering conclusion, chemical mixing, and the peanut constraint.
  Do not reward imitation of those.

Worksheet completion is not a continuation gate and no candidate winner is
required.

---

## (c) On-policy evidence round

### Why this is a dedicated runner and not `--on-policy-evidence-rounds`

Two independent reasons, both verified against this lane:

1. `run_lfm2_dynamic_catalog.py` only produces rounds for the child **its own
   fold pass just published** (`_run_postpublication_on_policy_rounds` runs after
   that pass's publication). Reaching it for r12's already-published child would
   mean re-running the whole closed-loop fold.
2. More decisively, the runner builds the round's capture configuration from its
   CLI flags and never sets
   `allow_top1_disagreement_within_numeric_error_envelope`. The sealed base cache
   was captured with that **and** `require_top1_agreement` both true. Measured
   selection digests:

   | configuration | `selection_sha256` |
   |---|---|
   | sealed base cache | `f229e5f399858b7a512ec0d7034fe4b3d76cf4dc3bb214ec461a8a6f7c000258` |
   | runner as r12 invoked it | `6b4fad64e33f98de248ee40494fef17d38059776f32a76b99304609147387c47` |
   | runner plus `--require-capture-top1-agreement` | `5af77e0626180006a78af7ea5eed05988fd8cc8b7ed3163c3c7b8dafa047ae5d` |

   No flag set reproduces the sealed selection, so
   `run_lfm2_on_policy_evidence_round` would fail closed on clause 6 of the
   extension contract ("comparable capture policy"). r12 itself never trips this
   because it only *opens* the sealed cache and never captures.

`experiments/run_r12_on_policy_round.py` rebuilds the capture configuration
**from the sealed base manifest** and refuses to proceed unless the
reconstruction reproduces the sealed `selection_sha256`.

### The round needs a GPU

The base cache's execution identity records `input_device_type: "cuda"` and
`attention_backend: "eager"`. Extension evidence must match it. The runner sets
the attention backend explicitly (Transformers otherwise auto-selects SDPA even
for a strictly local load, exactly as `prepare_lfm2_tokenizer_bridge_evidence`
had to) and then compares the live execution digest against the sealed one before
capturing. A CPU round is refused, by design, with the live and sealed
identities printed.

### Corpus

`experiments/lfm2_on_policy_fit_prompts_v1.json`: 120 prompts, 15 per dimension
across coherence, task-correctness-arithmetic, task-correctness-code, reasoning,
instruction-following, retrieval, multi-turn, explanation-summarization. ASCII
only (the target vocabulary is 4,096 rows and the probe enforces byte-exact
round-trip on every seed prompt), 68 to 255 characters, no chat template — the
probe tokenizes each prompt string directly.

Disjointness, proven at commit time by `tests/test_lfm2_on_policy_fit_prompts.py`
and again at round time by the library:

- 0 overlaps with the 16 routed review prompts, comparing every message content,
  the joined message text, and two transcript renderings, exact and
  whitespace-stripped;
- 0 overlaps with the 18 reserved future-review prompts;
- 0 overlaps with the pinned calibration corpora — the 256-row
  `target_tokenized_calibration.json` (`e0ff14fbb7222ae454ad7d27622c4f77a091e9285922688447b4fdde2b20a102`,
  512 `target_input_texts` plus `target_forced_decode_texts`) and the 28 prompts of
  `canonical_parent_outputs_bridge_bound.json`
  (`253be73e1b65a790698abfe74c508b51c709781f55d6ae051b9ba380056f8419`).
  `mapped_donor_calibration.json`
  (`080682f04b433ebe46809ebbe441343b9d7cb22de2f2133105052eb8c17f907c`) and
  `mapped_parent_outputs.json`
  (`3bbdffdea1d5a146d6adef48be7ad1cecaff38645f8b9ca0ec3f9fd31ba5a726`) carry token
  IDs or the same 28 prompt strings; their digests are recorded so the checked set
  is reproducible.

`experiments/lfm2_manual_review_reserved_digests_v1.json` is the manifest the
round consumes. It exists because `load_manual_review_prompt_manifest` refuses a
review document whose rows carry `messages` rather than a flat `prompt` string —
verified: loading `lfm2_semantic_whole_response_prompts_v1.json` directly raises
`Manual review prompt row 0 must carry one nonempty 'prompt' string; supply a
digest manifest for chat-message rows.` The manifest is a sealed
`neural-forge-lfm2-manual-review-prompt-manifest-v1` digest list covering all 75
distinct exact-and-stripped digests of both reserved sets.

### Run it

```
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
cd /home/cereal/model-surgery-lab/deployments
SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh preflight
'
```

Then:

```
SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh round
```

The launcher is `SOURCE_COMMIT`-guarded like the r12 launcher, additionally
refuses a dirty checkout, refuses to start while r12's PID is alive, verifies the
published child's `neural_forge_tokenizer_bridge_reference.json` names this
bridge, checks the three staged file digests, and re-reads the sealed base
selection and execution identity. It runs
`experiments/run_r12_on_policy_round.py` on `cuda:0` with
`--max-new-tokens 8` (the sealed capture budget is `max_decode_steps = 8`; a
longer continuation would have to be truncated to stay comparable).

### Expected artifacts

| Artifact | Path |
|---|---|
| Round root | `$LAB/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12` |
| Sealed round manifest | `<round root>/round-000/round-manifest.json` |
| Extension cache | `<round root>/round-000/extension-cache/<cache key>` |
| Deterministic generation cache | `<round root>/round-000/generation-cache.json` |
| Runner receipt | `<round root>/round-000-runner-receipt.json` |
| Log | `<round root>/round.log` |

### What to check in the sealed manifest

- `disjointness.verified_disjoint` true, `overlapping_prompt_count` 0
- `prompt_corpus.prompt_corpus_sha256` = `834a9585...`
- `counters.parent_forward_calls_during_child_rollout` = 0
- `counters.parent_capture_invocations` = 1, `parent_capture_passes_per_batch` = 1
- `counters.optimizer_steps` = `backward_calls` = `training_steps` = `network_calls` = 0
- `counters.parent_state_unchanged`, `child_state_unchanged`, `child_artifact_unchanged` all true
- `target_preimage` present, `parent_cache_input_mode` =
  `target-tokenizer-canonical-then-one-row-mapped-to-donor`, and
  `target_preimage.bridge_artifact_sha256` = `5c1a452c...`
- `rollout.rejected_prompts` — read it. Rejections are recorded, not silent. The
  ones to expect on a bridged lane are
  `child_retokenization_shifted_the_prompt_boundary`,
  `child_produced_no_continuation`, and
  `child_token_outside_bridge_row_capacity`.

A round with zero surviving batches fails closed rather than sealing an empty
extension; if that happens, the corpus needs prompts the child continues, not a
flag change.

---

## (d) Refold on base + extension evidence

### Which entrypoint

`--on-policy-refold-after-rounds N` drives
`run_production_catalog_on_policy_alternation`, where pass 0 folds and publishes,
runs a round, and pass 1 refolds. That is two complete folds. It also refuses
`--on-policy-child-artifact-directory`, so it cannot be pointed at r12's existing
child.

Since stage (c) already sealed round 0 against r12's published child, the refold
is a **single** fold pass that consumes it:

```
--on-policy-round-manifest <round root>/round-000/round-manifest.json
--on-policy-max-rounds 2
```

with `--on-policy-evidence-rounds` left at 0. This is semantically alternation
pass 1 — same parent, same sealed base cache, same blank seed, same topology and
counted budget, differing only in the evidence consumed and where it publishes —
at half the fold cost. `_validate_on_policy_evidence_configuration` accepts
manifests with zero new rounds; the lane gate
(`_unconsumed_extension_evidence_reason`) no longer refuses the bridged carrier
lane and reduces to `_bridged_lane_target_preimage_gap`, which passes because the
round carries its preimage.

### Run it

```
SOURCE_COMMIT=<postpub commit> ./nf-launch-r12-round-refold.sh refold
```

The stage re-runs preflight, then reads the sealed round manifest and refuses by
name if it carries no target-token preimage or names a different bridge, then
launches into a **new** sibling run directory. It never writes into `$R12`.

### Expected artifacts

| Artifact | Path |
|---|---|
| Run directory | `.../a0023-exact4x-h816-mgv3-r12-round-0-refold` |
| Publication | `<run>/published-child/child` |
| Receipt | `<run>/receipt.json` |
| Diagnostics | `<run>/diagnostics.json` |
| Routed manual dossier | `<run>/routed-manual-dossier.json` |
| Consumed manifest digest | `<run>/consumed-round-manifest.sha256` |

### What must move and what must not

| Field | Expectation |
|---|---|
| `parent_cache_key` | unchanged from r12 |
| `evidence_split_sha256` | **moves** |
| `parent_trace_rows_sha256` | **moves** |
| published `evidence_fingerprint` | **moves** (it becomes the evidence-set key) |
| `configuration_sha256` | **moves** (consumed manifests are part of the identity) |
| `on_policy_evidence.consumed_rounds` | one entry, round index 0 |
| `parent.on_policy_merged_row_accounting` | present, with a measured RSS reading |

Consuming extension evidence adds the `lfm2_on_policy_evidence_extension` phase
dependency, so **the derived global-topology and balanced-basis phase-cache
entries will miss and recompute**. That is correct behaviour, not a fault, and it
is the dominant added cost of the refold. Budget for it.

Then repeat stages (a) and (b) against the refold's own dossier. A refold is a
fold attempt: it appends to its own ledger and writes its own receipt.

---

## Staging verification, 2026-08-01

Run while r12 was still folding. Nothing below touched r12's run directory, its
checkout, or any GPU (`CUDA_VISIBLE_DEVICES=""`, `nice -n 15`).

| Check | Result |
|---|---|
| Remote smoke suites — the same set r11 staging ran (`test_hf_lfm2_circuit_trace_cache`, `test_hf_lfm2_closed_loop_restart_capsule`, `test_hf_lfm2_replay_sharding`, `test_run_lfm2_dynamic_catalog_carrier_e2e`) plus `test_lfm2_on_policy_fit_prompts` and `test_hf_lfm2_on_policy_evidence_round`, on the postpub checkout | **129 passed, 2 skipped in 418.63s** |
| Skip 1 | `test_hf_lfm2_circuit_trace_cache.py:545` — "CUDA is required"; GPUs were hidden on purpose |
| Skip 2 | `test_lfm2_on_policy_fit_prompts.py:221` — "local PACK calibration mirror is absent"; that mirror only exists in the workstation's ignored `upload_staging/`, where the test passes |
| Carrier e2e under the ambient umask | Server default umask is `0002`; the suite passed unmodified, so the `058fff55` HMAC-key `chmod(0o600)` fix carried in this checkout is doing its job |
| Fit corpus round-trip, target tokenizer (4,096 rows) | 120/120 byte-exact, 0 prompts under the 3-token capture minimum, 0 token IDs at or beyond the bridge row capacity, lengths 22/40/97 (min/median/max) |
| Fit corpus round-trip, donor tokenizer (65,536 rows) | 120/120 byte-exact, lengths 15/31/65 |
| Context budget | worst case 97 + 8 = 105 tokens against `max_context_tokens = 512` |
| Calibration disjointness against the real PACK | 540 texts, 552 exact-and-stripped digests, digest-set SHA-256 `88ff81dd3a844102da4f7ab4a95ec7b9610fb77e8975fc18166626958278776d` reproduced exactly, **0 overlaps**, all four PACK source digests OK |
| Strict bridge artifact | loaded and `verify_files()` passed against the live donor and target tokenizers; artifact and profile digests equal the pinned values |
| Base cache | selection `f229e5f3...`, execution `3d077b94...` (`eager`, `cuda`), 256 batches |
| Launcher | `bash -n` clean, no CR bytes, usage guard exits 64, wrong `SOURCE_COMMIT` exits 41, and with the correct commit it exits 43 refusing to run while r12's PID is alive |
| Launcher flags | all 22 round flags and all 97 refold flags exist in their argparse parsers; diffed against `nf-launch-integrated-carrier-exact4x-r12.sh`, the only flag-set difference is `--on-policy-round-manifest` and `--on-policy-max-rounds`, plus `--child-name` and the `$RUN`-derived destinations |
| Round runner end to end | exercised on a tiny real two-layer `Lfm2ForCausalLM` lane on the workstation: sealed round 0 with 3 extension batches, 0 parent forwards during rollout, 1 capture invocation, 1 pass per batch, 0 optimizer steps / backward calls / training steps / network calls. **Unbridged** — see limitation 3 |
| Attention-backend binding | setting `_attn_implementation` does not move `fingerprint_lfm2_parent(...).identity_sha256`, so the runner's ordering (fingerprint match, then bind execution identity) is safe |

## Known limitations, recorded before execution

1. **Nothing here has run at real scale.** NOTES-W5A §7 item 1 already flags that
   the routed carrier lane consuming merged evidence is unverified at any scale;
   the toy carrier configuration cannot reach the routed path. Stage (d) is the
   first execution.
2. **`--dense-vocabulary-margin-observability` with extension evidence is
   unexercised** (NOTES-W5A §7 item 2). r12's flag set turns it on, so the refold
   is also the first test of the vocabulary-margin binding against a merged
   corpus. If it fails, it fails in `Lfm2TargetTokenizerVocabularyMarginBinding.__post_init__`
   with a receipt/adapter digest mismatch, which is the fail-visible direction.
3. **The wrapper's bridged branch is not covered by an executed test.** The
   round's bridged behaviour is covered by
   `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py::test_bridged_carrier_lane_alternates_fold_round_and_refold`;
   the wrapper's own end-to-end smoke was run unbridged.
4. **`max_context_tokens` binds on the longer of the two tokenizations**
   (NOTES-W5A §7 item 6). The 4,096-row target tokenizer needs more tokens than
   the donor for the same text. The corpus tops out at 255 characters and the
   bound is 512 with 8 new tokens, which should be comfortable, but a rejection
   here is explicit and fail-closed rather than silent truncation.
5. **A donor-collapsed target trajectory keeps only its first preimage**
   (NOTES-W5A §7 item 4). Check `collapsed_target_trajectory_sha256` in the sealed
   preimage records; it is empty in every test artifact because no toy corpus
   produces a collapse.
6. **The alternation summary is not a receipt** (NOTES-W3C §7 item 7). Not
   relevant to the single-pass refold above, but it is if anyone switches to
   `--on-policy-refold-after-rounds`.
