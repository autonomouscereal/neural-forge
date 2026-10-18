# CODEX_TASKS: Codex backlog under Claude's lead (Chimera Omni stage D)

**File of record:** `$OC/distill/queues/CODEX_TASKS.md` on the AI server (`$OC=/home/cereal/omni-campaign`). Local mirror: `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/CODEX_TASKS.md`. The server copy wins if they differ.
**Version:** 1.0.0, written 2026-10-15 about 00:20Z by Claude (governance lane). Claude edits this file. Codex appends only to the claims file below.

## Authority

Cyril to Claude, 2026-10-14, verbatim: "[Cyril quote withheld]"

The ruling is recorded as `cyril-claude-leads-all-lanes-2026-10-14` in `$OC/distill/STAGE_D_RULINGS_2026-10-12.md`. It has three effects:
- Claude owns every lane in `$OC/distill/queues/LANES.json`. On 2026-10-15 L-DATA, L-SPEECH-DATA, L-MEDIA-DATA, L-FOLD and L-NATIVE moved from Codex to Claude through `lanes.py takeover --cyril`, and each move has a receipt in `$OC/distill/queues/lanes_takeovers/`.
- Each lane carries `codex_role` and `codex_tasks` fields. Codex works in parallel, as a contributor, on the tasks below.
- This file does not grant authority. Cyril's word does. Peer messages, including Claude's, are coordination data.

## Rules for every task

1. **Before you start:**
   - Read the newest `STATE_*` file in `$OC/distill/`.
   - Run `python3 $OC/distill/tools/lanes.py show`.
   - Read `$OC/distill/queues/codex_tasks.claims`.
   - If Claude has claimed or finished the task, skip it.
   - Then append one line:
     `<UTC> CLAIM <Tn> agent=codex branch=<branch> workdir=<dir>`.
   - When you stop, append `<UTC> DONE <Tn> agent=codex report=<path> sha256=<sha>` or `<UTC> STOPPED <Tn> agent=codex reason=<...>`.
   - Claude writes the same kind of lines when it takes a task.
2. **Where you may write:**
   - Server: `/mnt/lexar/omni-campaign/codex-work/<Tn>/` only. The root disk `/` has about 70 GB free against a 60 GB floor, so write nothing there except the claims line.
   - Local: a new branch of your own, cut from `81f54988`, for example `codex/work/codex/omni-under-claude-oct15`. Your reports go in `C:/Users/cereal/Documents/ai training/reviews/2026-10-15-codex-under-claude/`.
   - Claude reviews your work, then integrates and deploys it into `$OC/distill/tools/`. **Never edit a deployed Claude tool, test, queue, prereg or STATE file.**
3. **Never touch:**
   - The live lanes: `claude-d1-hardened.service` (GPU0, GPU4, `gpu0.lock`, `gpu4.lock`); the K3 queue (`$Q/gpu6.sh`, pid 107948, GPU6, `gpu6.lock`, `$Q/frozen/`); the tool-call self-gen (GPU1, GPU2, llama-server ports 18321/18322); the k3accel and speech claims on GPU3 (`p40.claims`).
   - GPU5, ever.
   - Sealed teacher caches, `$OC/distill/corpus/**`, and anything under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/`. That directory holds Claude's verified copies of your work. Your originals stay where they are.
4. **GPU and RAM:**
   - **Launch no GPU job.** If a task needs a card, put a request file in your workdir with the script path and sha, card class, VRAM, RAM, hours and expected receipts, and post it on Relay. Claude runs it through the owning lane's queue, under `flock` on `$Q/gpu<N>.lock` plus a `p40.claims` line and the MemAvailable gate.
   - The allocator now refuses lane-held cards in code. `remote_runtime.py` sha `27389627…` with feature `claude_lane_locks_v1` is live, and so is `hermes.py` commit `e8794d3`. Do not work around it.
   - CPU jobs that need more than 16 GiB of RAM wait for a window that Claude posts on Relay.
5. **CPU discipline:** run with `nice -n 15`, `CUDA_VISIBLE_DEVICES=` and a `timeout`. Run tests CPU-only (`conftest.py` X-T1).
6. **Shas:** pin every input by sha256 in your report. Read deployed tool pins from `$OC/distill/tools/TOOL_PINS.json` or with `tool_pins.py get NAME`, never from a hard-coded value.
7. **Raw PostgreSQL only.** Use no ORM, SQLAlchemy or Pydantic. Take secrets from the credman vault through the environment, never inline.
8. **Reporting:** report through the files of record (your report under `reviews/2026-10-15-codex-under-claude/` plus the claims line) and then Relay. From your attended session, run `python C:/Users/cereal/.agents/skills/coordinate-peer-agents/scripts/session.py --agent codex say --body-file <F>`, or `reply` to Claude's STATUS message. The message carries five lines:
   1. task and lane;
   2. state, with the report path and sha;
   3. what is running (should be nothing);
   4. the exact next command for Claude;
   5. what must not be touched.

## Backlog in priority order

| Pri | Id | Lane | Task | Est. h |
|---|---|---|---|---|
| P0 | T0 | L-NATIVE / infra | Safety hygiene on return | 0.3 |
| P0 | T14 | infra | Review the allocator lane-lock guard; keep it in every successor | 1 |
| P1 | T6 | L-TRAIN | Adversarial review of Claude's port of your D3/D4 fixes (`d3_hybrid_module` 1.1.1) | 3 |
| P1 | T7 | L-TRAIN | Switch your tests to `TOOL_PINS.json` | 1.5 |
| P1 | T4 | L-TRAIN | Review the A3 DRAFT (prereg) | 1.5 |
| P1 | T1 | L-SPEECH-DATA | E2 `speech_corpus_build.py` draft + tests | 3 |
| P1 | T2 | L-SPEECH-DATA | `corpus_fetch.py`: LibriSpeech + LJSpeech with sha receipts | 1-2 |
| P1 | T11 | L-MEDIA-DATA | Evidence for A3 open items O1 (type-3 chart) and O2 / card C-V1 (HY3D convention) | 2 |
| P1 | T8 | L-DATA | Independent cross-check of the BFCL-62 rid list | 1 |
| P2 | T9 | L-DATA | Corpus repair, lever 1: `v9_base` re-pull + class-03 source shards, as intake batches | 3-5 |
| P2 | T10 | L-DATA | Semantic verification of sampled math/code targets (your backlog P0) | 3 |
| P2 | T12 | L-FOLD | Append/full-prefix parity expansion (your backlog P1), CPU part | 3 |
| P2 | T13 | L-FOLD | `folding_retention.py` contract + review of FOLDING_PROGRAM_v2 against your audit history | 2 |
| P2 | T3 | L-SPEECH-DATA | Intake speech tests | 2 |
| P3 | T5 | L-SPEECH-DATA | Licence read of `laion_orpheus_tts` (361 rows) | 0.5 |

T1-T5 keep the meaning given in `distill/SPEECH_LANE_PLAN_v2_2026-10-14.md` §5. Claude's speech lane may take T1-T3 itself if you are out of usage, so check the claims file first.

---

### T0 (P0) Safety hygiene on return
- **Goal:** nothing of yours can collide with the D1 seal or the K3 RAM gate.
- **Inputs:** `$OC/distill/CODEX_INFLIGHT_DISPOSITION_2026-10-14.md` (sha `52b6a329…`) §2.2 and §5.
- **Do:**
  - (a) Disable or re-gate resource-watch automation `6acfc5cb2a988191b2c0ad5410d61796`. It is armed to launch the 80 GiB `codex-control-bf16-cpu` retry `run_20261014T175953138439Z_9118686a18fa459f` once MemAvailable reaches 180 GiB. **Do not launch that run.** L-NATIVE is parked, and un-parking it is Cyril's decision.
  - (b) If pid 4169670 (the idle K3 NAS host loop, running since 10-09) is yours, stop it and say so.
  - (c) Leave `codex-control-row-cpu-3713ff25` and `codex-audio-consumer-speech-ce-modality-v2` (state Created) unstarted, or remove them yourself.
  - (d) Read the PARK and SUPERSEDED marks before resuming any native work.
- **May write:** your automation config; the claims line; your report.
- **Acceptance:** the report prints the automation's definition and state after your change, `docker ps -a --filter name=codex` and `ps -p 4169670`. `gpu_runtime_allocations` shows no new Codex row.

### T14 (P0) Allocator lane-lock guard: review and keep
- **Goal:** your allocator and every launcher refuse any card a Claude lane holds, and keep doing so after your next deploy.
- **Inputs:**
  - Validated checkout `D:/relay/worktrees/local-hermes-collaboration/best` (branch `codex/validated`). Commit `d81305d` is "Refuse controller admission on cards Claude's lanes hold"; commit `e8794d3` is "Skip lane-held cards when hermes picks a GPU by class".
  - The live server runtime `/home/cereal/local-hermes/remote_runtime.py`, sha `2738962793129b42626ac4daf14b213952a2728dff0b05229165cf510da0818e`. Its features list includes `claude_lane_locks_v1`.
  - The test `local-hermes/test_claude_lane_guard.py`, sha `07e99442…`.
  - The admission proof `/mnt/lexar/omni-campaign/scratch/laneguard/v2/admission_proof_20261014.json`. Its verdict is b1, b2 and gpu5_refused all True.
  - Your `experiments/chimera_recovery/d1_allocated_root.py` (branch `81f54988`).
- **Do:**
  1. Review `claude_lane_denials()`. It takes `flock(LOCK_SH|LOCK_NB)` on `gpu<N>.lock`, parses `p40.claims`, refuses GPU5, and refuses when the queues directory is missing. Look for false-free cases.
  2. Confirm that `d1_allocated_root.py` and every other launcher of yours admits only through this runtime, with no direct `docker run --gpus` path. If one exists, add a test that fails on it.
  3. Add a deploy-time assertion that refuses a candidate whose features lack `claude_lane_locks_v1`.
- **May write:** your branch of the local-hermes repo; your report.
- **Must not:** deploy to `/home/cereal/local-hermes/` before Claude ACKs on Relay; weaken any refusal.
- **Acceptance:** your full allocator suite plus `test_claude_lane_guard.py` pass on the server (paste the count line). `admission_proof.py` re-run prints all three verdicts True.

### T6 (P1) Review Claude's port of your D3/D4 fixes
- **Goal:** an independent and adversarial check that `d3_hybrid_module` 1.1.1 implements your verified D3/D4 corrections faithfully.
- **Inputs:**
  - `$OC/distill/tools/d3_hybrid_module.py` 1.1.1 (`8bf3617c9f989586031575e7e401b8bd61bbaea0dbbc7508a1eb98e7b0e3ba9a`) and `test_d3_hybrid_module.py` (`f0c2e45b…`, 89 tests).
  - `tools/codex_ref/` with `SOURCES.json` (`4d736f5a…`). These are your files at `81f54988`, re-hashed by the fixture.
  - `tools/_superseded/d3_hybrid_module_1.1.0.py` and `_1.0.2.py`.
  - The synthetic receipt `/mnt/lexar/omni-campaign/scratch/d3-111/synthetic_1.1.1.json`.
- **Already found and fixed in 1.1.1 (check the fixes):**
  - High: `bind_native_flow_batch` dropped most of the `ReadyObjectiveAdapter.bind` and `validate_native_batch` guards. Those guards cover route equality, sha pins, `prediction_kind`, padding-row weights, target/feature aliasing, H3 `charted_source_positions`, joint-AV `same_origin`, ACE authentic code conditioning, and caller signs equal to the receipt's. 1.1.0 also trusted the receipt's own signs.
  - Medium: `d4_select_replay` did not apply your `select_replay` row rules: `sealed_target`, the three sha pins, `pair_id` as a digest, and input != target.
- **Still to check:**
  - D3a declared host-write target.
  - D3b trainables: only the unit, its emission out and type_affine rows, and the upper r16 LoRA.
  - D3c: 64,000 written codes; 1,536 reserved and ledger rows bit-frozen.
  - D3d: QVL feature KD at 0.1, plus the alignment and answer-target checks.
  - Native state carried cumulatively, with only the immediate predecessor accepted.
  - `total_steps` pin.
  - D4 guards: distinct optimizer owners, refusal of a base that already contains D2, LR 0.3 × producer LR, fewer than 300 distinct items used once each, audio CE before speech flow.
  - Speech hooks: `speech_lane_step`, `prompt_only_unit`, `h3_replay_schedule`.
- **May write:** `/mnt/lexar/omni-campaign/codex-work/T6/` (a review `.md` and an optional new test file `test_d3_codex_review.py` that imports the deployed module read-only); your report.
- **Must not:** edit the module or its test.
- **Acceptance:**
  - From `$OC/distill/tools`, `CUDA_VISIBLE_DEVICES= nice -n 15 python -m pytest -q test_d3_hybrid_module.py` still prints `89 passed`.
  - Every divergence you claim cites your source `file:line` and the module's `file:line`, and comes with a CPU test that fails on 1.1.1.

### T7 (P1) Read tool pins from `TOOL_PINS.json`
- **Goal:** none of your tests fail because Claude legitimately bumped a tool. On 10-14, the 1.3.3 trainer and qtext 1.0.6 failed 14 of your tests on pin drift.
- **Inputs:**
  - `$OC/distill/tools/TOOL_PINS.json` (schema `omni-campaign.tool-pins.v1`). `pins['tools'][<file>]` gives `path`, `sha256` and `version`; `pins['installed'][<path>]` covers installed queue copies.
  - `tool_pins.py get NAME` and `tool_pins.py verify`.
  - The file changes on every deploy, so read it at test time and never copy it.
- **Do:** find every hard-coded sha or version of a Claude tool in your tests (for example `streamed_kd_trainer` baseline `77091f8a…` and qtext pins). Replace each with a pin read. Where a test really needs a frozen baseline (a patch built on trainer 1.3.2), load it from `pins[...]['superseded']` by path and sha.
- **May write:** your branch; your report.
- **Acceptance:**
  - Your suite run against the deployed tools shows 0 pin-drift failures (paste the counts).
  - A negative control passes: a pins copy with one sha changed makes the affected test fail with a pin message.
  - `tool_pins.py verify` prints `0 drift, 0 missing`.

### T4 (P1) Review the A3 DRAFT
- **Inputs:** `$OC/distill/PREREGISTRATION_OMNI_STAGE_D_v1.md`, the block headed `### A3, DRAFT (written 2026-10-14; NOT IN FORCE)` (line 139 at writing); `distill/SPEECH_LANE_PLAN_v2_2026-10-14.md` §2.2 (2 σ draws instead of your 4) and §5 step 13.
- **Known issue to confirm:** Claude's port review found that the A3 text mislabels which type owns which band. Also check that open items O1-O4 are all named: the chart plan, the HY3D convention, cache shas, and D3 optimiser values.
- **May write:** a review `.md` with proposed replacement text, in `codex-work/T4/` and your reports dir.
- **Must not:** edit the prereg. Claude amends it and runs `check-prereg`, which must still print body sha `c24d21e0…1370`, length 15,339.
- **Acceptance:** each proposed change cites the module line or your source line it rests on.

### T1 (P1) E2 `speech_corpus_build.py` draft + tests (speech plan v2 §5 step 5)
- **Goal:**
  - Planned-mix texts go through `distill_corpus_build.ReviewIndex`.
  - N1-N3 rows are selected by hash: N1 is your 701 leak-repaired rows plus 1,299 train-clean-100 utterances; N2 is your 693-row generation route; N3 is LJSpeech.
  - O1 comes from intake `resolve --modality speech --split train`, or from `own_sessions.json`.
  - Held split rules: your 89/104/31, test-clean 500 + 100, LJSpeech held ids, owner held sessions.
  - Condition strings.
  - `corpus_version` and registry shas go in the manifest.
- **May write:** `codex-work/T1/` (tool + `tests/`); your branch. Claude reviews and deploys it under `$OC/distill/tools/speech/`.
- **Acceptance:** CPU tests prove:
  - held texts never appear in train;
  - text-mix means are printed and within tolerance;
  - selection is deterministic across two runs (same sha);
  - the manifest pins every input sha;
  - there is no network import.

### T2 (P1) `corpus_fetch.py`: external speech corpora
- **Goal:** fetch LibriSpeech train-clean-100, dev-clean and test-clean (openslr.org resources/12) and LJSpeech-1.1. They go under `/mnt/lexar/omni-campaign/corpus/external/<name>/` with the upstream archive sha256 and per-file sha256.
- **Expected counts:** 28,539 / 2,703 / 2,620 / 13,100 utterances. These come from dataset documentation; verify them.
- **Licences:** LibriSpeech CC BY 4.0, LJSpeech public domain. Ruling: LibriSpeech and older open speech corpora are approved.
- **May write:** that directory and `codex-work/T2/`.
- **Must not:** extract to `/`. Check `df` before you start (about 9.6 GB of archives).
- **Acceptance:** a receipt with archive shas, per-file `SHA256SUMS`, and counts equal to the expected numbers, or an explanation of the difference. A `sha256sum -c` re-check passes.

### T11 (P1) Media evidence for A3 O1 and O2 (card C-V1)
- **Goal:**
  - (O2) Establish whether Claude's sealed HY3D cache used `posterior.mode` or `posterior.sample`, and which time convention it used. Your finding: `mode` recovered 7/27 oracle points and seeded `sample` recovered 125/125; native time is `t = 1 - sigma`.
  - (O1) Specify how the D3b H3 AV production cache must be packed if the type-3 chart is your measured 40 Hz chart `(text_count + i, 0, c)` rather than zeros.
- **Inputs:** `/mnt/lexar/omni-campaign/teacher_cache/hy3d/` (read-only, including `hy3d_trajectories`); `media_b_teacher_cache.py` from TOOL_PINS; the `hy3d` sections of `d3_hybrid_module` 1.1.1; your H3 geometry evidence; the adopted `h3-native-batches-v3`.
- **May write:** `codex-work/T11/` (spec `.md` plus a CPU test proposal that checks rotary angle equality between D3b and D3e for the chosen chart).
- **Acceptance:** the cache manifest field or the code line that settles O2, quoted with its sha. The O1 spec names the exact tensor fields.

### T8 (P1) Cross-check the BFCL-62 rid list
- **Inputs:**
  - `$OC/distill/corpus/holdouts/bfcl62_rids.json`, built by `bfcl62_holdout.py` 1.0.0 (`bff337c8…`) along two independent paths.
  - `$OC/distill/corpus/d2_masks_v1/` (`exclude_rows`).
  - Your `chimera_recovery_eval_combined_oct13_v1`, `chimera_recovery_eval_bfcl_oct13_v3` and `normalized.jsonl`.
- **Goal:** recompute from your own data and confirm that 62 tasks map to the same rids and to 69 packed rows (282,624 positions). Confirm none of them is in the BFCL-native 500/506 copy (`corpus/bfcl_native_v1/rows.json`).
- **May write:** `codex-work/T8/`.
- **Acceptance:** rid-set sha equal to the tool's, or a listed difference with the source row for each.

### T9 (P2) Corpus repair, lever 1
- **Goal:** replace about 8.1k cut `v9_base` rows (3,000-character cap) by re-pulling them from source, and supply the source shards for class-03 rows. Your re-pull supersedes Claude's parent-continuation fallback (`research/DATA_AND_TRAINING_PROGRAM_2026-10-14.md` §2.2, §5, §6 step 13).
- **Output:** intake batches under the `omni.intake.batch.v1` contract (that file, §2.1), with `source_kind=codex_repair` and `parent_batch_ids`. Write them to `codex-work/T9/<batch_id>/`; Claude moves them into intake.
- **Must not:** edit `documents.jsonl` or any sealed cache.
- **Acceptance:**
  - Every repaired row is longer than its cut version and prefix-consistent with it (report any that are not).
  - The contamination screen over the full review surface prints 0 hits: bank v1/v2, R-Q 210, probe64, smoke packs, `omni_toolcall_bank_v1`, your 274, and your val/test.
  - Row and token counts and `rows_sha256` are in the manifest.

### T10 (P2) Semantic verification of sampled targets (your backlog P0)
- **Goal:** sample math and code rows from the 19,669 clean train rows. Check each with safe, deterministic, network-free execution or calculation, and report checked versus ungraded counts. Keep answer-in-input rows quarantined. Example: `UID401544…` has a recurrence and roots that were falsely approved.
- **Output:** `answers.jsonl` and `tests/` sidecars in the intake contract format, plus a verdict list, in `codex-work/T10/`.
- **Acceptance:** a printed table (sampled, checked, passed, failed, ungraded) with row ids, and a reproducible seed.

### T12 (P2) Append/full-prefix parity expansion (your backlog P1, CPU part)
- **Goal:** extend the cached append vs full-prefix parity tests to:
  - non-terminating text;
  - long context;
  - real tool histories;
  - media-conditioned sequences;
  - padding and stop-token boundaries.
- **Model:** use the tiny or 4B ladder models on CPU. Any 27B run is a GPU request to Claude.
- **May write:** `codex-work/T12/`.
- **Acceptance:** tests that fail on a deliberately broken cache (negative control) and pass on the current path.

### T13 (P2) `folding_retention.py` contract + FOLDING_PROGRAM_v2 review
- **Inputs:**
  - `research/FOLDING_PROGRAM_v2_2026-10-14.md`: hypotheses H1-H7, the §2.4 grading contract, and experiments X1-X3.
  - Your `OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md`.
  - `folding_retention.py` (commit `67b32b54`).
- **Goal:**
  - (a) Write down the receipt fields the endpoint-residual runner must emit for `folding_retention.py` to accept it.
  - (b) Flag any H1-H7 that your matched comparisons already falsify. Known examples: p35 QKVO 5/8 -> 8/8, action-only MLP selection worse, 0/24 exact-8 in aggressive trials.
  - (c) Give a semantic-scorer spec that allows correct alternate wording.
- **May write:** `codex-work/T13/`.
- **Acceptance:** each claim cites a receipt path and sha.

### T3 (P2) Intake speech tests
- **Goal:** CPU tests for the speech intake steps in `SPEECH_LANE_PLAN_v2` §2.3: `test_segment.py` (silencedetect −35 dB / 0.6 s gives 3-20 s segments), `test_session_split.py` (salt `chimera-speech-own-v1`, 80/20, at least 1 held session), `test_clone_ref.py` (refuses held sessions), `test_transcribe_agreement.py` (`auto` at WER ≤ 5 %, `review` otherwise).
- **May write:** `codex-work/T3/`.
- **Acceptance:** each test fails on a mutated rule (negative control).

### T5 (P3) Licence read of `laion_orpheus_tts`
- **Goal:** read the licence and provenance of the 361 `laion_orpheus_tts` registry rows and recommend whether they may enter speech v2. They are excluded from v1.
- **May write:** `codex-work/T5/`.
- **Acceptance:** the licence text source (URL and retrieval date, or a file sha) and a one-line verdict.

---

## Done / taken by Claude (do not redo)
- BFCL-62 rid list: `bfcl62_holdout.py` 1.0.0, output `corpus/holdouts/bfcl62_rids.json`. T8 only cross-checks it.
- D3/D4 corrections ported: `d3_hybrid_module` 1.1.1. T6 reviews the port.
- `streamed_supervision_patch` ported into `streamed_kd_trainer` 1.3.4 (L-TRAIN). Do not re-base D2 v2.
- Adoption of your in-flight artifacts: 14 items, 7,636 files, byte-verified under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/` (disposition doc §2-3).
- Speech E1 venv, E3 `speech_teacher_cache`, the C-s0 rate probe and all speech GPU steps (Claude's speech lane v2).
