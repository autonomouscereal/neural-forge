# NOTES-S2A — stage-2 teacher self-generation corpus (data preparation)

**Date:** 2026-08-09 · **Branch:** `codex/lrc-blank-compression` · **HEAD at start:** `fec82744` ·
**HEAD at handoff:** `ac3f7c9a` (moved by the other stage-2 lane, not by me) ·
**Authority:** `docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`
(sha256 `96b642cb4eebe7671228801ea98fb722cad9c913af7f63bbc264de8b15b24a04`) section 2.3.

> **Git note.** I ran no writing git command — only `log`, `status`, `rev-parse`, `ls-files`,
> `cat-file`, `branch`. While this work was in flight the lane-1 agent advanced the branch and, in
> commit `ae4b8fb5` ("Build the stage-2 distillation trainer, toy-proven end to end"), swept
> `tests/stage2/test_stage2_generate_teacher_corpus.py` into its commit. The committed bytes are
> identical to what I wrote (`git diff HEAD` on that path is empty) and the 23 tests still pass, but
> the file is now **tracked** rather than the untracked artifact this task called for. Flagging it
> because I cannot un-commit someone else's commit and because it means lane 1 is running
> `git add`-by-directory. `experiments/stage2_generate_teacher_corpus.py` and this file remain
> untracked.

**Scope:** corpus generation only. No training step ran, no optimizer was constructed, no stage-1
file was edited, and no stage-1 run directory was written to. Sealed stage-1 artifacts were read
(the PACK calibration corpus, the a7 selection evidence, the parent checkpoint) and never modified.
Everything the run writes lands under `/mnt/ai_data/neural-forge-stage2/`; `/home` (93% full) holds
nothing from this work.

---

## 1. Files

| path | state |
|---|---|
| `experiments/stage2_generate_teacher_corpus.py` | **new**, untracked; the generator |
| `tests/stage2/test_stage2_generate_teacher_corpus.py` | **new**; 23 CPU tests. Written untracked; committed by lane 1's `ae4b8fb5` mid-task (see the git note above), bytes unchanged |
| `upload_staging/canonical_parent_outputs_bridge_bound.json` | local mirror of the pinned PACK file (`upload_staging/` is gitignored); lets the calibration half of the screen be tested offline |

Server (all new):

```
/mnt/ai_data/neural-forge-stage2/
  code/stage2_generate_teacher_corpus.py     sha256 9bce297f6133f72a7e91dab7c53634e4b9dfcb00d3bedbcafea62155e126ff3e
  code/run_corpus.sh                          the launch wrapper (records an exit status)
  code/launch_detached.py                     copy of experiments/launch_detached.py
  code/verify_shard.py                        independent shard re-verifier
  inputs/prefix_estate.json                   sealed prefix estate bundle
  inputs/lfm2_semantic_whole_response_prompts_v1.json    sha256 0ee6bf14…  (the 16 V5 prompts)
  inputs/lfm2_causal_support_future_manual_review_v1.json sha256 27b8f88b…
  corpus/shards/shard-NNNNN.{tokens.u16,text.jsonl,manifest.json}
  corpus/inputs/prefix_estate.tokens.npz      tokenized estate cache, keyed by estate digest
  corpus/progress.json                        heartbeat: shards done, measured tok/s, projection
  corpus/corpus_manifest.json                 written when every shard is complete
  _launch/corpus-rtx8000-r4/                  job.log, job.pid, job.status
  smoke/ calib/ trial-shard0-16a5c153/        verification artifacts, kept as evidence
```

Note the deviation from design section 10, which places this code at
`src/neural_forge/stage2/corpus_generation.py`. The task specified
`experiments/stage2_generate_teacher_corpus.py` as a single self-contained module, and self-contained
matters here: the generator is deployed to the stage-2 directory on the server and imports nothing
from `neural_forge`, so no repo checkout has to exist next to it and nothing it does can reach
stage-1 code.

---

## 2. Script design

One file, four subcommands, no ORM/database anywhere (nothing here needs persistence beyond
content-addressed files).

**`estate`** — builds the sealed prefix estate from local sources, screens every text through the
frozen-review screen before admitting it, and writes one bundle sealed by `bundle_sha256` over its
own body.

**`generate`** — the run. Per shard:

* `shard_seed = shard_index`, recorded three ways in the manifest (`shard_seed`,
  `torch_manual_seed`, `prefix_rng_seed`). `torch.manual_seed(seed)` drives the sampler;
  `random.Random(seed)` drives prefix-window selection. A shard is therefore a deterministic
  function of its index given the same model and code.
* Batches are homogeneous by context length so no padding is ever needed: one batch of
  `batch_size/2` unconditional rows (`[BOS]`, 1023 sampled tokens) and one of `batch_size/2`
  prefix-seeded rows (`[BOS] + 64 estate tokens`, 959 sampled tokens). Both reach exactly 1,024.
* Sampling config is built from scratch, never inherited: the shipped `generation_config.json`
  carries `temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`, none of which is the design's
  sampler. The run pins `do_sample=True, temperature=1.0, top_p=0.995, top_k=0,
  repetition_penalty=1.0, eos_token_id=None, min_new_tokens == max_new_tokens`, and records all of
  it. Generation continues through `<|im_end|>` so every stored row is exactly 1,024 tokens.
* A `LogitsProcessor` masks the 1,134 undefined physical rows every step. The tokenizer's defined
  ids are asserted contiguous `0..64401` at load, and the counts are asserted against the sealed
  contract (64,402 defined / 1,134 undefined / 65,536 physical) or the run refuses.
* Curation: drop any sequence whose 3-gram self-repetition fraction exceeds 0.5.
* Contamination screen (section 3 below).
* **Seed-kind quota:** exactly half the accepted rows are `bos` and half are `prefix`. See section 7
  — the first trial shard came out 2,298/1,798 without it.
* Files are published by `write-temp → fsync → os.replace`, and the manifest is written **last**.
  The manifest's presence is therefore the commit record, which is what resume trusts.

**`seal`** — re-reads every shard manifest, verifies each seal (optionally re-hashes every data
file), and writes `corpus_manifest.json` sealed by `corpus_manifest_sha256` over its own body. It
carries the full shard inventory (index, split, seed, token count, `tokens_sha256`, `text_sha256`,
`manifest_sha256`), the aggregate filter statistics, and the run provenance (teacher digest,
tokenizer digest, estate digest, generator digest, design-document digest, both screen digest-set
shas). `generate` calls it automatically once every shard is complete.

**`self-test`** — CPU only, no model, no GPU: rebuilds both digest sets and proves the screen drops
16 spliced V5 prompts while leaving ordinary text alone.

### Manifest fields per shard

`schema, shard_index, shard_seed, torch_manual_seed, prefix_rng_seed, split, row_count,
sequence_length, token_count, token_dtype, tokens_sha256, tokens_bytes, text_sha256, text_bytes,
seed_kind_counts, seed_kind_quota, accepted_mean_repetition_fraction, filter_statistics{…},
generation{…}, run{…}, elapsed_seconds, tokens_per_second_accepted, tokens_per_second_sampled,
completed_at_utc, manifest_sha256`.

---

## 3. The contamination screen

Design section 2.3 says the exclusion is "the 16 V5 prompts and their 552-digest disjointness set
screened out of every prefix and generated sequence by the same utf8-sha256 exact-and-stripped
comparison the fit-prompt pack already uses". Two things in that sentence needed resolving, and both
resolutions are recorded in the manifest rather than buried in code:

**(a) An exact digest comparison is near-vacuous against generated text.** A 1,024-token teacher
sample will essentially never hash-equal a review prompt; what it *can* do is quote one. The screen
therefore runs both passes and counts them separately:

* `reserved_review_digest` — utf8-sha256, exact and whitespace-stripped, against the 75-digest set
  built from *every rendering* of both reserved review documents (individual messages, the joined
  transcript, and both `role: content` transcript forms) — the identical construction used by
  `tests/test_lfm2_on_policy_fit_prompts.py`. Rebuilt at run time and **asserted equal to the sealed
  set sha `b9d8ee479cf2f062f104e2a8d2b0b8ba245c8afc7772e30404f6c24fc26388d9`** that the fit-prompt
  pack recorded; a mismatch refuses the run.
* `reserved_review_containment` — substring containment of any reserved rendering (≥ 48 characters;
  the shortest V5 message is 65, so nothing is lost) in the raw text **and** in a
  whitespace-normalized copy of it, so a reflowed quotation still hits. This is the load-bearing
  pass and it is the one that actually fires.

**(b) The 552-digest set is the *calibration-corpus* disjointness set, not a V5-derived set** — and
the design draws prefixes out of that same bridge-folding corpus. Screening prefixes against it
would delete the prefix estate. So it is applied to generated sequences only, and the manifest says
so in words: `calibration_corpus_screen.applied_to_prefixes = false` with
`not_applied_to_prefixes_because`. The set is rebuilt at run time from the two pinned PACK files and
**asserted equal to the sealed sha `88ff81dd3a844102da4f7ab4a95ec7b9610fb77e8975fc18166626958278776d`
(552 digests from 540 texts)** before the run starts.

**Live proof, every batch.** `--inject-review-probes N` splices a real V5 message into filler text
and pushes N such probes through the *same* screen path every batch. If a probe ever survives, the
run raises `Stage2TeacherCorpusError` and stops. The counters `review_probe_injected` and
`review_probe_dropped` are in every shard manifest, so "the screen was live for this shard" is
readable evidence rather than a claim. The production run uses `--inject-review-probes 1`.

---

## 4. The prefix estate

`inputs/prefix_estate.json`, `estate_sha256 bdac5b56f698645d8e3bc8f4e47a1bdf32f858ac6b6fe0b45bfe7eb28552028d`,
1,331 texts, 0 screened out.

| source | admitted | provenance |
|---|---|---|
| `bridge_folding_corpus` | 512 | `target_tokenized_calibration.json` sha `e0ff14fb…`, PACK `lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5` |
| `on_policy_fit_prompts` | 120 | `experiments/lfm2_on_policy_fit_prompts_v1.json`, `corpus_sha256 4bf7263d…` (the digest the design quotes) |
| `a7_repository_corpus` | 699 of 942 | files named in a7's sealed `fold-fit-selection-evidence.json`; **only files whose bytes still match the sealed sha256 are admitted** — 243 have drifted since 2026-07-30, 0 missing, and the counts are in the bundle |

**`a8_natural_language_corpus` is absent, and that is a real gap.** a8's job log records
`external-fold-fit-selection-evidence-loaded file_count=1024 sampled_byte_count=80198516
evidence_sha256=0ea9140784502a9bf2ee76bf4c5f7d40f88b0194f720a5f57e4535f559827d1c`, but that evidence
file is not in the repo, not in `reports/carrier-replant-a8-…`, and not in the a8 artifact directory
on the server (`…/neural_forge_carrier_replant_20260730/a8-natural-language-2025-2026-corpus356-cout-lambda0001/`
holds only the carrier artifact, log, pid/status and dossier). Its 1,024-file inventory is therefore
not reconstructible from sealed evidence, so it is *not* in the estate. The prefix estate is the
design's list minus a8, and the bundle records that by omission — I did not substitute a
similar-looking corpus for it.

Prefix windows are drawn without crossing a document edge, weighted by each document's eligible
window count; documents shorter than 64 tokens are skipped and counted. The estate yields
**7,477,020 distinct 64-token windows** (recorded per shard as `run.prefix_estate_window_count`),
against 264,192 prefix rows for the whole 129-shard run — 3.5% of the window space, so reuse is
negligible and shard 0 drew 2,048 distinct windows for 2,048 prefix rows.

---

## 5. Teacher pinning

`/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors`
sha256 `f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd`.

The design requires this to equal the stage-1 parent digest in the sealed run receipts. It does:
`reports/sparse-semantic-a11-k32-split-cout-20260730/run-receipt.json` records exactly that digest at
`parent_tied_table_source_inspection/selected_shard_sha256`. The generator refuses to start if the
checkpoint's digest differs (`--allow-unpinned-teacher` exists but is not used). Tokenizer
`tokenizer.json` is digested into every manifest too.

**Dtype note:** the design's hardware table assumes bf16 is fine because the P40s are excluded for
lacking it. The Quadro RTX 8000 is Turing (sm_75) and has no native bf16 either, so generation runs
**fp16** and the manifest records `dtype: "float16"`. Sampling math is unaffected: `generate`
computes the logits in fp32 before the warpers.

---

## 6. Launch proof

Unit `nf-stage2-corpus.service` (transient, `--user`, `Restart=on-failure`, `RestartSec=30`),
started **2026-08-09 05:01:01 UTC**, run tag `corpus-rtx8000-r4`:

```
MainPID=1791550  ActiveState=active  SubState=running  Restart=on-failure
    PID    PPID    PGID     SID  ELAPSED STAT CMD
1791550    2402 1791550 1791550    00:10 Ss   /bin/bash …/code/run_corpus.sh
1791554                                        …/venv/bin/python -X faulthandler …/stage2_generate_teacher_corpus.py
```

`PGID == SID == PID` with `STAT=Ss` — its own session, its own process group, no controlling
terminal, no SSH session in its ancestry. GPU pinned by UUID
(`CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd` = Quadro RTX 8000). **The UUID
matters:** `CUDA_VISIBLE_DEVICES=6` selects a Tesla P40, not the RTX 8000 — nvidia-smi's PCI ordering
and CUDA's default `FASTEST_FIRST` ordering disagree on this host. Both RTX 3090s (idx 0, 4) were
idle before launch and are untouched; the P40 at idx 5 carries somebody else's 23 GB workload and was
not touched either.

### Why a systemd transient unit and not `launch_detached.py`

The repo's `experiments/launch_detached.py` (PPID 1, `start_new_session=True`) was tried first and
**died twice, silently, at ~10-11 minutes**, both times before the first shard landed:

| run | launcher | started | died | evidence |
|---|---|---|---|---|
| `corpus-rtx8000-r1` | `launch_detached.py`, python direct | ~03:56 UTC | < 04:06 | `job.log` 147 bytes (weight-loading bar only), no traceback |
| `corpus-rtx8000-r2` | `launch_detached.py`, bash wrapper | 04:11:04 UTC | < 04:22 | wrapper's `EXIT=` line never written, `job.status` never created — bash itself was SIGKILLed |

Ruled out: OOM (no `dmesg` OOM lines; 179 GB available of 251 GB), logind
(`KillUserProcesses=no`, `Linger=yes`, user manager active), and a blanket kill of detached processes
(a plain `setsid nohup` bash sentinel ran 11+ minutes untouched). **I did not root-cause it.** The
working hypothesis is that `launch_detached.py`'s child stays inside the SSH login session's
`session-N.scope` cgroup, which systemd tears down some minutes after the connection closes — the
one thing a transient unit under the lingering user manager is immune to. Recorded as unresolved.

The mitigation is sound independent of the cause: the job runs in its own cgroup outside any session
scope, `Restart=on-failure` brings it back after 30 s if it dies again, and because shards are
independent and resume is by missing manifest, a restart loses at most the shard in flight. The
wrapper writes `EXIT=<code>` and `_launch/corpus-rtx8000-r4/job.status` on any non-signal exit, so a
future death is diagnosable rather than silent.

---

## 7. Measured throughput, and the corpus-size decision

All numbers measured on the Quadro RTX 8000, fp16, this teacher.

**Batch scaling at short context (128 decode steps):**

| per-call batch | tok/s | ms/step | peak GB |
|---|---|---|---|
| 256 | 6,340 | 40.4 | 1.60 |
| 512 | 8,482 | 60.4 | 2.77 |
| 1,024 | 9,511 | 107.7 | 5.11 |
| 2,048 | 9,830 | 208.3 | 9.79 |

**At the real 1,024-token length, where the KV cache actually grows:**

| per-call batch | tok/s | wall | peak GB |
|---|---|---|---|
| 512 | 2,850 | 184 s | 14.87 |
| 1,024 | 2,915 | 360 s | 29.31 |

Throughput **saturates at ~2.9k tok/s** and batching past 512 buys 2% for 2x the memory. Two
concurrent workers on the same card were *worse* than one (1,130 + 1,129 = 2,259 tok/s combined vs
2,850 single), so the card is genuinely saturated, not launch-latency-bound. `cache_implementation=
"static"` — which would remove the per-step dynamic-cache concat — is **not available for LFM2's
hybrid conv/attention stack** in transformers 5.12.1 (`KeyError: 'conv'`,
`AttributeError: 'LinearAttentionLayer' object has no attribute 'max_batch_size'`). Production runs
at per-call batch 512 (`--batch-size 1024`).

**Measured end-to-end on a real shard:** 4,194,304 accepted tokens in 1,828.6 s = **2,294 tok/s
accepted** (2,872 tok/s sampled including rejects).

### Projection against the design

The design projected ~20k tok/s and ~15.5 GPU-h for 1.12B tokens. The measured rate is **8.7x
slower**, so:

| target | tokens | projected wall |
|---|---|---|
| design's 1.12B | 1,120,403,456 | **~136 h (5.7 days)** |
| **chosen: 129 shards** | **541,065,216** | **~65.5 h (2.7 days)** |
| a hard 24 h cap | ~198,000,000 (≈ 47 shards) | 24 h |

### The floor, and what the design actually says

**The design states no numeric minimum corpus size.** It states one volume (1.12B = 1.10B train +
20M held-out) and justifies it as covering "the 1B ladder rung with zero repeats". The floor has to
be derived from the design's own epoch rules, and it is *smaller* than 1.10B because 1.10B was sized
as though the whole student budget were off-policy:

* Section 2.4: forward-KL on the corpus for the first 10M student tokens, then a **50/50** mix of
  off-policy corpus and on-policy GKD rollouts. Only the off-policy half draws from this corpus.
* Section 4: all four arms share "identical corpus + shard order for the off-policy fraction", so
  the corpus is **not** multiplied by arm count.

Off-policy draw per rung: 10M rung → **10M**; 100M rung → 10M + 0.5x90M = **55M**; 1B rung →
10M + 0.5x990M = **505M**. Plus the 20M held-out split.

So the zero-repeat floor for the **whole ladder** is **505M train + 20M held-out = 525M**, and for a
ladder stopping at the 100M rung it is **75M**. (The brief's "≥300M for the 100M rung" figure is not
in the design document and I could not derive it from any reading of sections 2.4 / 3.1 / 4; the
100M rung's zero-repeat requirement is 55M off-policy tokens.)

### Decision

**Target 129 shards = 541,065,216 tokens: 5 held-out shards (20,971,520) + 124 train shards
(520,093,696).** That is the derived zero-repeat floor for the full 1B ladder with ~3% margin,
2.07x below the design's 1.12B, at a projected **~65.5 h**.

I did **not** cut further to fit 24 h. Going below 505M train tokens would force repeats at the 1B
rung, which changes the data regime for all four arms simultaneously — a design change, not a
scheduling choice, and not mine to make silently. The 24 h figure is honoured as a *decision point*
rather than a cap, and it is cheap to honour because of how the run is structured:

* **The held-out shards are generated first (indices 0-4).** The split rule is
  `holdout if index < holdout_shards else train`, so raising `--train-shards` later is purely
  additive: no already-sealed shard ever changes split, and this 129-shard target is a **strict
  prefix** of the design's full 267-shard / 1.12B plan. Extending to the full design target later
  regenerates nothing.
* **Stopping is free and safe at any shard boundary.** `systemctl --user stop nf-stage2-corpus`,
  then `seal --output-root …/corpus` produces a valid sealed corpus of whatever completed. At the
  24 h mark that is ~47 shards ≈ 198M tokens, which covers the 10M and 100M rungs zero-repeat with
  3.6x margin but not the 1B rung.

### The 50/50 fix (why one shard was thrown away)

The first production shard verified clean on every count except one: `seed_kind_counts` came out
**2,298 bos / 1,798 prefix**, not the design's 50/50. Cause: the shard stops at 4,096 accepted rows,
the unconditional batch is sampled before the prefix batch in each iteration, so the final
iteration's overflow was discarded entirely from the prefix side. The generator now enforces a
per-kind quota (2,048/2,048), skips generating a batch whose kind is already full, and records
`seed_kind_counts`, `seed_kind_quota`, and `surplus_over_kind_quota` in every manifest. Verified on a
16-row run: `{'bos': 8, 'prefix': 8}` against quota `{'bos': 8, 'prefix': 8}`, both shards. The fix
is cost-neutral (the true drop rate is ~6.5%, so each kind still needs 5 half-batches).

That first shard was generated by the previous generator sha and was **moved out of the corpus** to
`/mnt/ai_data/neural-forge-stage2/trial-shard0-16a5c153/` rather than left to contaminate an
inventory whose manifests would then disagree about which code produced them. It is kept as
evidence, not deleted.

---

## 8. Shard verification

Re-verified by `code/verify_shard.py`, which re-reads the files independently of the generator,
re-hashes each **twice**, and recomputes the curation statistic from the raw token file.

```
/home/cereal/model-surgery-lab/venv/bin/python \
  /mnt/ai_data/neural-forge-stage2/code/verify_shard.py \
  /mnt/ai_data/neural-forge-stage2/corpus <index>
```

| | shard 0 | shard 1 | shard 2 |
|---|---|---|---|
| split / seed | holdout / 0 | holdout / 1 | holdout / 2 |
| `manifest_seal_ok` | True | True | True |
| rows x length | 4,096 x 1,024 | 4,096 x 1,024 | 4,096 x 1,024 |
| `token_count` (file == manifest) | 4,194,304 | 4,194,304 | 4,194,304 |
| `tokens_sha256` (head) | `f78ccb2b21afb2dc…` | `e0f65c7606cefc61…` | `32330fdedf048e18…` |
| `text_sha256` (head) | `2ccb55342ab730db…` | `ddf17d559ebad648…` | `d6e42dcd2a4c31e3…` |
| stable across two independent re-reads | **True** | **True** | **True** |
| both digests match the manifest | **True** | **True** | **True** |
| min / max token id | 1 / 64,391 | 1 / 64,391 | 1 / 64,399 |
| any undefined id (>= 64,402) | **False** | **False** | **False** |
| every row starts with BOS | True | True | True |
| `seed_kind_counts` (manifest) | 2,048 / 2,048 | 2,048 / 2,048 | 2,048 / 2,048 |
| `seed_kind_counts` recounted from the text file | 2,048 / 2,048 | 2,048 / 2,048 | 2,048 / 2,048 |
| distinct prefix windows / prefix rows | 2,048 / 2,048 | 2,048 / 2,048 | 2,048 / 2,048 |
| max repetition fraction (<= 0.5) | 0.499022 | 0.500000 | 0.500000 |
| repetition recomputed from raw tokens (64-row sample) | 0.487280 | 0.498043 | 0.431507 |
| `dropped_repetition` | 334 | 301 | 309 |
| `dropped_reserved_review_containment` | 5 | 5 | 5 |
| `dropped_reserved_review_digest` | 0 | 0 | 0 |
| `dropped_calibration_corpus_digest` | 0 | 0 | 0 |
| `dropped_undefined_token_id` | 0 | 0 | 0 |
| `review_probe_injected` / `dropped` | **5 / 5** | **5 / 5** | **5 / 5** |
| `surplus_over_kind_quota` | 690 | 723 | 715 |
| elapsed / tok/s accepted / tok/s sampled | 1,825.3 s / 2,297.87 / 2,872.34 | 1,823.2 s / 2,300.56 / 2,875.70 | 1,823.4 s / 2,300.21 / 2,875.26 |

Every shard also carries, and was checked for, identical provenance:
`teacher_shard_sha256 f630da86…`, `estate_sha256 bdac5b56…`, `generator_sha256 9bce297f…`,
`design_document_sha256 96b642cb…`, `reserved_digest_set_sha256 b9d8ee47…`,
`calibration_digest_set_sha256 88ff81dd…`.

**Reading the screen numbers.** `review_probe_injected 5 / review_probe_dropped 5` is the live proof:
five real V5 review messages were spliced into candidate text during the shard and all five were
dropped by containment. Those five are also the whole of `dropped_reserved_review_containment`, i.e.
**no genuine teacher sample in shards 0-2 reproduced review-surface text** — the screen fired only on
the probes deliberately fed to it. `dropped_reserved_review_digest 0` and
`dropped_calibration_corpus_digest 0` are the expected near-vacuous exact-digest passes.

The only real curation loss is repetition, at 301-334 of 5,120 candidates (5.9-6.5%), which is a
sane rate for a 230M teacher at T=1.0 and well inside what the design anticipated.

`max_repetition_fraction` of exactly 0.500000 in shards 1 and 2 is correct, not a boundary bug: the
design drops sequences *above* 0.5, so a sequence sitting exactly on 0.5 is kept.

---

## 9. Tests

`tests/stage2/test_stage2_generate_teacher_corpus.py` — 23 tests, CPU only, no torch, no model,
nothing on disk over a few KB.

```
C:/Python314/python.exe -m pytest tests/stage2/test_stage2_generate_teacher_corpus.py -q
23 passed in 0.85s
```

Run the file by name, not the directory. `pytest tests/stage2/` now also collects lane 1's six
trainer test modules, five of which error at import under `C:/Python314` with
`ModuleNotFoundError: No module named 'torch'` / `'neural_forge'`. Those are not my files and not my
failures — that runtime has no torch and `src/neural_forge/stage2/` is empty in this checkout — but
the directory-level command is not green, so I am not claiming it is.

Coverage: both digest sets reproduce their sealed shas; all 16 V5 prompts drop when spliced into a
sequence; a reflowed V5 prompt still drops (the normalized pass); a whole V5 prompt drops by digest
*and* containment; ordinary text survives both passes; a tampered review pack is refused; the
calibration half is generated-only and does not touch prefixes; the estate screens a contaminated
source, seals its body, and refuses a tampered bundle; a repository corpus admits only byte-identical
files; the prefix sampler never crosses a document edge and is deterministic in its seed; the
repetition statistic's bounds; a shard manifest seal covers its body and a mutated field is refused;
resume treats only a sealed, consistent shard as done, catches truncation by byte length and
same-length corruption only when verifying digests, and regenerates a shard whose manifest is
missing; the corpus manifest seals the inventory, refuses a corrupted shard, an empty root and an
unknown split; and the donor vocabulary contract.

The server-side `self-test` subcommand ran on the deployed copy and reproduced both sealed digest
set shas with 16/16 probes dropped.

---

## 10. Resume recipe

Shards are independent. A shard counts as done only when its manifest exists, its seal covers its
own body, and both data files exist at the byte lengths the manifest records. Anything else is
regenerated from scratch under the same seed.

```bash
# status
systemctl --user status nf-stage2-corpus.service
cat /mnt/ai_data/neural-forge-stage2/corpus/progress.json
ls /mnt/ai_data/neural-forge-stage2/corpus/shards/*.manifest.json | wc -l

# resume after any kill, crash or reboot (re-runs only missing shards)
systemctl --user reset-failed nf-stage2-corpus.service
systemd-run --user --unit=nf-stage2-corpus --collect \
  --working-directory=/mnt/ai_data/neural-forge-stage2 \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd \
  --setenv=PYTHONUNBUFFERED=1 --setenv=TOKENIZERS_PARALLELISM=false \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  --property=StandardOutput=append:/mnt/ai_data/neural-forge-stage2/_launch/corpus-rtx8000-r4/job.log \
  --property=StandardError=append:/mnt/ai_data/neural-forge-stage2/_launch/corpus-rtx8000-r4/job.log \
  --property=Restart=on-failure --property=RestartSec=30 \
  -- /bin/bash /mnt/ai_data/neural-forge-stage2/code/run_corpus.sh

# stop early, then seal whatever completed
systemctl --user stop nf-stage2-corpus.service
/home/cereal/model-surgery-lab/venv/bin/python \
  /mnt/ai_data/neural-forge-stage2/code/stage2_generate_teacher_corpus.py \
  seal --output-root /mnt/ai_data/neural-forge-stage2/corpus --verify-seal

# paranoid resume: re-hash every completed shard before skipping it
#   add --verify-resume to the generate arguments in code/run_corpus.sh
```

Extending to the design's full 1.12B later: raise `--train-shards` from `124` to `262` in
`code/run_corpus.sh` and relaunch. The 129 sealed shards are reused untouched.

---

## 11. What I did not do, and what is not verified

* **No training, no trainer, no optimizer.** The training gate is closed and nothing here approaches
  it.
* **Nothing of mine is on `/home`.** Total local-disk footprint was a 12 KB `/tmp/nf-sentinel`
  diagnostic directory, since removed. `/` moved 93% → 94% during the session from unrelated
  workloads (Mailcow clamav signatures, a Suricata `eve.json`, an unrelated `model-surgery-lab`
  artifact cache), all confirmed by `find -mmin -240 -size +50M`. Corpus footprint is 23.7 MB/shard
  → ~3.1 GB at 129 shards, entirely on `/mnt/ai_data` (1.8 T free).
* **No git operations.** Two new untracked files; `HEAD` is still `fec82744`.
* **No stage-1 edits.** Sealed artifacts were read only. The one sealed-adjacent write is the local
  gitignored mirror in `upload_staging/`.
* **The r1/r2 silent kills are not root-caused** (section 6). Mitigated, hypothesized, not proven.
* **Bit-exact regeneration is not asserted.** Seeding is recorded and per-shard deterministic, but I
  did not regenerate a full shard twice to prove the token file is bit-identical; cuDNN/cuBLAS
  reduction order is not pinned. Digest *stability across re-read* is verified (section 8); digest
  stability across *re-generation* is not.
* **`a8_natural_language_corpus` is missing from the prefix estate** (section 4) — its file
  inventory is not recoverable from sealed evidence.
* **The 552-digest calibration screen has never fired and probably never will** — it is an exact
  digest match against 1,024-token samples. It is applied and counted because the design names it;
  the containment screen is what actually protects the review surface.
* **The corpus is not finished.** At handoff (2026-08-09 06:34 UTC) shards 0-2 of 129 are sealed and
  verified — the full 3-shard held-out prefix is nearly complete — and the run is steady at ~1,824 s
  per shard. The top-level `corpus_manifest.json` is written only when all 129 land, or on demand via
  `seal`. `seal` itself is proven working on two smaller roots (`smoke/`, 3 shards,
  `corpus_manifest_sha256 9826bd31…`; `diag/`, 2 shards, `d1989bbd…`), both with `--verify-seal`
  re-hashing every data file.
* **Projected completion: 2026-08-11 ~22:20 UTC** (65.4 h from 05:01 UTC on 2026-08-09) if run to the
  full 129-shard target. Nothing downstream is blocked by it — the training gate is closed and both
  RTX 3090s are free.
