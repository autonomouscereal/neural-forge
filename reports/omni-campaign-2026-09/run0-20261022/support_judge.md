# Run 0 support: the media judge on Strata (D2)

Date: 2026-10-28. The work ran 07:11-08:00Z, a usage limit stopped it, and it resumed 10:50-13:15Z.

- **Ruling:** `cyril-run0-decisions-2026-10-22` D2: "[Cyril quote withheld]".
- **Scout:** `support_scout_judge.md`.
- **Relay** (thread `52468f63…`):
  - CLAIM, event 204067, 07:11Z;
  - resume STATUS `8f7df1dd`, event 204072;
  - RELEASE, posted after this report.

## 0. Answer first

1. **Strata can see images now, and the projector is on the CPU, not the GPUs.**
   - Projector: `unsloth/Qwen3.8-Flash-Next-GGUF@766911a6` `mmproj-F16.gguf`. 904,004,000 B, sha256 `1f7b7f0b…a980`, Qwen Community licence. It sits on the Lexar with `SOURCE.json`, `SHA256SUMS` and `sha256.check`.
   - It runs with `--no-mmproj-offload`. `GGML_EXPERT_CACHE_MB` stays `30000,50000`, so Sprite's expert caches keep all their VRAM:
     - GPU0: 23,438 MiB used (23,782 before the change);
     - GPU6: 44,642 MiB used (44,970 before).
   - Live since 11:30Z.
2. **Host RAM cost: about 1.0 GiB.**
   - llama-server RSS is 105.91 GB fresh in text-only mode, and 106.97 GB fresh with the projector after the first image (VmHWM 106.97 GB).
   - The judge client itself is about 50 MB RSS. It runs under `MemoryMax=3G` with no swap and sees no CUDA device.
   - Mesh prep (numpy, CPU) runs inside the same cap. I did not measure its peak separately.
3. **The rollback was tested, and it is one command:** `/home/cereal/ai-serve/work/strata-judge-0c189f21/strata_vision.sh rollback`.
   - Test run 11:15-11:23Z. Every file came back to its backup sha.
   - After it: gate 1.2.3, launcher 1.3.0, text-only, background floor 40.
   - Text served afterwards (15.3 tok/s). Images got `400 no_vision`.
   - Then I re-applied (11:24-11:30Z).
4. **The judge runs on Strata through strata-gate at background priority.** `media_judge` 1.3.0, spec 1.3.0, sha `44147e9161a5…`.
   - Same rubric, schemas, thresholds, frame grids, mesh views, sidecars, `media_judgement` rows, omni_lab evals and publish gating as 1.2.0.
   - A preempted or busy answer is not a verdict: the group stays in `ready/`.
   - Music and H3-audio groups (CLAP/ASR) are parked: 586 groups.
5. **GPU5 is free for run 0.**
   - `media-judge.timer` is disabled.
   - `p40_cards` is empty, so judge 1.3.0 cannot claim any P40.
   - Nothing was started on GPU5 or on any P40.
6. **Calibration: 61 of 80 agree (76 %, kappa 0.525) against Qwen3.6-27B verdicts on the same items.** That breaks down as:
   - edits 19/20;
   - video 16/20;
   - images 13/20;
   - meshes 13/20.

   Strata is more lenient: 12 of the 40 old rejects passed, and 7 of the 40 old passes were rejected. In the disagreements I checked myself, the old judge was wrong about as often as Strata (§4). The number alone does not settle it.
7. **One real job ran:** 14 pending groups. 5 passed (4 t2i, 1 LTX video) and were published to Lexar intake by the 13:02Z `media-publish` run. 9 were rejected and moved to `rejected/`. No holds, no deferrals.
8. **Not done, on purpose: I did not re-enable `media-judge.timer`.**
   - Continuous judging means about 24 h of background time for the backlog (§5).
   - It also means accepting the leniency in §4 for training intake. That is a call for the lead or Cyril.
   - One command turns it on: `sudo systemctl enable --now media-judge.timer`. With the 1.3.0 drop-in it runs every 15 min, and it judges only when the gate admits background.

## 1. Where the earlier attempt stopped, and what I did about it

Found on resume. All of it was applied by session 0c189f21's earlier attempt, 07:11-07:57Z:

| Item | State found | Action |
|---|---|---|
| Projector download | Complete; `sha256.check` OK | Kept (no re-download) |
| `backups/strata-vision-20261028` | 13 files, `SHA256SUMS` all OK; `media-judge.timer.was-enabled.txt` = enabled/active at 07:54Z | Kept; it is the rollback source |
| strata-gate 1.3.0 + capture 1.3.0 + policy + `strata-gate.service.d/vision.conf` (`--vision-tiers background,cyril --mem-floor-gib 12`) | **Fully applied** at 07:57:50Z, healthy at 07:57:53Z | Kept, and its tests re-checked (§3). Not half-applied |
| strata-llama | **Untouched** (10-27 15:56 process, `/props` vision false). Launcher 1.4.0 and a GPU6 `vision.conf` were staged only | Re-planned for the CPU (below), then applied |
| media_judge 1.3.0 / spec 1.3.0 | Staged only | Applied 11:24Z |
| `media-judge.timer` | Disabled and inactive | Left disabled |

There was no half-applied Strata change.

**Change of plan from the earlier attempt.** It had planned the projector on GPU6 (`-mmdev CUDA1`), with GPU6's expert cache capped at 33,000 MiB to make room. That takes VRAM from Sprite's expert cache, and this task forbids it. So the drop-in now sets `STRATA_MMPROJ_DEV=none` and no expert-cache override. The launcher's default is also `none`. I tested the projector on the CPU first, as asked, and CPU encode is fast enough for a background judge (§2).

**Side effect of the gate step, already live since 07:58Z, flagged for the readiness step.**
- What it is: the gate's background admission floor went 40 → 12 GiB. That unblocked background work: 735 background requests up to 10:55Z (run0-strata-writer, gen-worker-text).
- Ownership: this is the admission floor only. The prompt-cache RAM change belongs to the run-0 readiness step / Strata lane, and I did not touch it.
- What to recheck: re-derive the floor when run 0 takes its ~20 GiB.
- Known issue: gen-worker-text still stamps the IQ3_S teacher id on rows made by the Q4 (from `support_scout_gen`). That is not this lane's to fix.
- Not changed: media-gate's floor is still 40.

## 2. What changed, and where

All server code is under `/home/cereal/ai-serve`. It is not a git repo, so the change follows its convention:
- dated backups in `backups/strata-vision-20261028/`, with `SHA256SUMS`;
- the work dir `/home/cereal/ai-serve/work/strata-judge-0c189f21/`, which holds the rollback script, verify scripts, test logs and staged trees.

Workstation copy: `C:/Users/cereal/Documents/ai training/ai-serve/`:
- `strata/`, `media/` and `systemd/*.d/` now hold the deployed versions;
- the helper scripts are in `strata/vision-20261028/`;
- the previous local copies are in `backups/strata-vision-20261028-local/`.

| File (server) | Version / sha256 (16) | What |
|---|---|---|
| `strata/strata_llama.sh` | 1.4.0 `4bc37480cd3b08c2` | `STRATA_MMPROJ` + `_SIZE` + `_SHA`: the projector is passed only when size, `SHA256SUMS` and `sha256.check` all agree, otherwise Strata starts **text-only** and says so. `STRATA_MMPROJ_DEV` (default `none` = `--no-mmproj-offload`). `STRATA_IMAGE_MAX_TOKENS` (1280). Without `STRATA_MMPROJ`, the command line is exactly 1.3.0's |
| `/etc/systemd/system/strata-llama.service.d/vision.conf` | new | The projector env above, CPU. Sorts after `q4kxl.conf` and overrides nothing in it |
| `strata/strata_gate.py` | 1.3.0 `dc5cb85e492928c9` | Images forwarded only for `--vision-tiers` AND while upstream `/props` says vision (re-read every 30 s). Sprite keeps `400 no_vision`. Video/audio parts → `400 unsupported_media`. `/v1/models` reports vision per tier |
| `strata/capture.py`, `capture_policy.json` | 1.3.0 `d0f875a86c64088c`, `4d2dd5d40d4b173f` | Image parts are stored as `{sha256, bytes, mime}`, not base64. `judge` is added to `blocked_source_pattern`, so judge prompts and captions are never exported |
| `/etc/systemd/system/strata-gate.service.d/vision.conf` | new | `--vision-tiers background,cyril --mem-floor-gib 12` |
| `media/media_judge.py` | 1.3.0 `955a9aa697c24d85` | Below |
| `media/media_judge_spec.json` | 1.3.0 `44147e9161a5a3a8` | Registers the judge, `strata-q4kxl-abl-v2+mmproj-f16-766911a6`: the Q4 `SHA256SUMS` hash and the mmproj sha. The identity is checked against the files and `/props` before every job. Rubrics etc. are byte-identical to 1.2.0 |
| `media/tests/test_media_judge_strata.py`, `strata/tests/test_gate_vision.py` | new | 13 + 10 tests |
| `/etc/systemd/system/media-judge.service.d/strata.conf` | new | `CUDA_VISIBLE_DEVICES=` (empty), `MemoryHigh=2G`, `MemoryMax=3G` |
| `/etc/systemd/system/media-judge.timer.d/strata.conf` | new | `*:05,20,35,50`. The timer itself stays **disabled** |

`media_judge.py` 1.3.0, in short:
- **`StrataJudge`** replaces the per-job llama-server:
  - `POST :18400/v1/chat/completions`, streamed, with the background key (`STRATA_GATE_BACKGROUND_KEY`, vault `strata_key_background`) and `X-Client-Source: media-judge`;
  - model `strata` (thinking off), `response_format` json_schema from the spec;
  - an answer that does not finish with `stop` is a hold, as in 1.2.0;
  - preempted / busy / thermal / starting answers raise `Deferred`: no verdict row, and the job stops.
- **`cycle`** checks `/gate/status` admission instead of claiming a card.
- **`calibrate`** re-judges old-spec verdicts and writes no verdicts.
- **The old 1.2.0 path** stays in the code as `judge_backend: "p40"`, for rollback only.

## 3. Evidence

**Tests (server, staged trees).**
- `media` judge suite: **32/32 OK** (19 existing + 13 new).
- strata-gate suite: 165 tests on the 1.3.0 tree.
  - At 10:58Z there were 4 failures (e.g. `test_upstream_slot_is_the_gates_choice` 503 ≠ 200). The 1.2.3 tree's 155 tests passed at 11:08Z.
  - Cause: these tests read the host's real `MemAvailable` against the default 40 GiB floor. At 10:58Z it was 29 GiB. At 11:08Z it was 150-233 GiB, because Strata was restarting.
  - Re-run of the 1.3.0 tree in the same high-RAM window: **0 failures**. It did show 2 `tearDownClass` errors (`test_gen_worker_text.TestQueue`, `test_sprite_capture_headers.TestOptOutDb`), and those two modules re-run alone pass **42/42**.
  - So the earlier failures are the environment, not 1.3.0. The two DB teardowns are flaky under load.

**Live checks, all through strata-gate at background priority** (`verify_vision.sh`, X-Client-Source `strata-vision-verify`, a blocked capture source):

| Moment | `/props` vision | Text (tok/s) | Image | Video part | GPU0 / GPU6 MiB | llama RSS |
|---|---|---|---|---|---|---|
| Before, 11:02Z | false | 10.4 (3 bg busy) | 400 no_vision | 400 unsupported_media | 23,782 / 44,970 | 74.2 GB + 6.1 GB swapped (old process, full cache) |
| Vision 1, 11:14Z | true | 13.8 | caption OK, 9.9 s (321 prompt tok in 7.5 s) | 400 | 23,466 / 44,650 | 106.98 GB |
| Rolled back, 11:23Z | false | 15.3 | 400 no_vision | 500 from llama (1.2.3 behaviour) | 23,466 / 44,648 | 105.91 GB |
| Final, 11:31Z | true | 12.9 | caption OK, 9.5 s | 400 | 23,438 / 44,642 | 106.09 → 106.97 GB after the image |

- Image test: COCO train `000000257036.jpg`, whose human captions are "a black cat, bicycle and backpack on the table". Strata: "A black and white cat with glowing eyes sits on a wooden surface behind a bicycle wheel, with a red backpack and floral curtains in the background."
- Journal: no Xid, illegal-access or abort lines since either restart. The last Xid is still 10-27 10:57 (IQ3_S).
- Outage: three strata-llama restarts of about 8 min each (11:06-11:14, 11:15-11:23, 11:24-11:30Z).
  - The writer and gen workers backed off on 503 `starting`.
  - The last sprite/cyril request was 10-27 22:04Z, re-checked before each restart. There was no PAUSE on Relay.
- Capture: the judge's rows are about 1.5 KB with no base64. Source `media-judge` is blocked from export.

**omni_lab.**
- Build run **40339**, 15 metrics.
- Calibration review run **40338**.
- Job evals: **40340** (edit), **40341** (t2i), **40343** (video_ltx), **40344** (mesh).
- Publish: **40342**.

**Receipts.**
- `/home/cereal/ai-serve/evidence/media-judge/calibrate-20261028T113200Z/calibration.json`
- `…/20261028T124435Z/receipt.json`
- `…/20261028T125346Z/receipt.json`

## 4. Calibration: Strata against Qwen3.6-27B on the same items

`media_judge.py calibrate --per-rule 20` took the most recent spec-1.2.0 verdicts (judged 10-26..10-28): 10 passes and 10 rejects per rule, each re-judged by Strata under the same rubric. The sample is balanced, so it does not reflect real pass rates.

| Rule | Agree | kappa | Old reject → Strata pass | Old pass → Strata reject | Mean score shift (Strata − old) |
|---|---|---|---|---|---|
| image | 13/20 | 0.30 | 5 | 2 | adherence +0.85, severity +0.85 |
| edit | 19/20 | 0.90 | 0 | 1 | applied −0.6, preserved −0.65 |
| video | 16/20 | 0.60 | 3 | 1 | action **+1.8**, adherence +1.1 |
| mesh | 13/20 | 0.30 | 4 | 3 | severity +1.1 |
| all | 61/80 (76 %) | 0.525 | 12 / 40 | 7 / 40 | |

**My own look at the disagreements.** I checked 8 of the 12 lenient flips: 7 by viewing the image or frames, and the mesh one from Strata's own caption. I also checked one strict flip, from the captions only:

| Item | Prompt asks | Old judge | Strata | My read |
|---|---|---|---|---|
| image b66a (interrogation room) | "heavy grain and high-contrast black-and-white" noir | reject: "heavy noise, underexposure" | pass | **Strata right**: the old judge penalised what was asked for |
| image f12c (fox in sink) | "silver fox … bright orange fur" (self-contradictory) | reject: red fox, not silver | pass | Defensible either way |
| image 246c (two foals) | foals in an acacia thicket | reject: legs merge | pass | Borderline (one foal's legs are a bit fused) |
| image 9fb3 (arm through onsen wall) | surreal muscular arm | reject: severity 4 | pass | Borderline |
| image 2be6 (1920s Shanghai sign) | bronze traffic sign, Shanghai alley, opium dens, art-deco type | reject: adherence 4 | pass | **Strata wrong**: garbled text, a generic European street, no alley |
| video 6ceb (Roman scribe) | camera tilts down to the writing | reject: action 4 | pass (action 10) | **Strata right**: the tilt from face to pen is clearly there |
| video b2ee (weeds at curb) | slow push-in to the tire | reject: action 2 | pass (action 9) | Mostly Strata: a clear push-in, with a jump between the first frames |
| mesh 6175 (hydrant + rocks), not viewed | hydrant "next to grey granite rocks" | reject: extraneous rocks | pass | **Strata wrong**: its own caption says "surrounded by rocks" but it marked no extraneous geometry |
| image 5401 (strict flip), not viewed | "four mismatched limbs" | pass | reject: "four arms and four legs" | Plausible strict call, from the captions alone |

**Reading.**
- Edits: the two judges agree.
- Video: Strata reads camera moves in the 8-frame grid better than the old judge, though it scores generously.
- Images and meshes: they disagree on borderline items near the thresholds. Strata is somewhat more lenient, and in 2 of the 8 lenient flips I checked it is clearly wrong.

The scout's suggested bar (≥ 85 % agreement, no lenient drift on known rejects) is **not met**. The cause is partly the old judge's own errors, but the drift is real. That is why I left the continuous timer off.

Possible next steps for the lead (none done here):
- A spec 1.3.1 that tightens the image adherence threshold or the mesh extraneous-geometry instruction. This is a new registration, so it must be re-calibrated.
- Or Cyril accepts 1.3.0 as is.

The other known risk, self-preference (Strata wrote many of these prompts), is not separable from leniency in this sample.

## 5. Throughput, RAM and what each part needs

- **Speed.** Median 39 s per member, 81 s at p90, while sharing the background tier with run0-strata-writer and gen-worker-text. Video and mesh items are the slowest. That is about 2× the scout's projection.
- **Backlog.**
  - Total: 2,342 pending groups under 1.3.0.
  - Parked: 586 audio groups (533 music, 53 H3 SOL), which need CLAP/ASR and so a GPU.
  - Remaining: 1,756 VL groups, about 2,200 members → **about 24 h** of free background time at one stream. Sprite or Cyril preempt it at any moment.
- **RAM.**

  | Part | Cost |
  |---|---|
  | Strata projector | +1.0 GiB, resident for as long as vision is on |
  | Judge client | ~50 MB, capped at 3 GiB |
  | Mesh prep | inside the same 3 GiB cap |

  There is no other host RAM and no VRAM cost.
- **Admission.** strata-gate refuses background below 12 GiB `MemAvailable`. MemAvailable was 28-29 GiB with a full prompt cache before the restarts, and about 120 GiB just after them, while the cache refills. With run 0's ~20 GiB taken it should still be above 12. The prompt-cache cap belongs to the readiness step.

## 6. Rollback

- **Everything:** `/home/cereal/ai-serve/work/strata-judge-0c189f21/strata_vision.sh rollback`.
  - It checks the backup `SHA256SUMS` first.
  - It removes the four drop-ins and restores the gate, capture, policy, launcher, judge and spec.
  - It restarts strata-llama (about 8 min) and then strata-gate.
  - It keeps `media-judge.timer` **disabled**, because the 1.2.0 judge would claim GPU5.
  - Tested 11:15-11:23Z.
  - Add `rollback --no-llama` to leave strata-llama running.
- **Vision only, keeping the gate:** delete `/etc/systemd/system/strata-llama.service.d/vision.conf`, run `systemctl daemon-reload`, restart strata-llama, then strata-gate. Going by the code, the 1.4.0 launcher without `STRATA_MMPROJ` passes the same arguments as 1.3.0, since its vision array is empty. That path itself was not run: the tested rollback restored the 1.3.0 launcher. The gate sees `/props` vision false within 30 s.
- **Status at any time:** `strata_vision.sh status`.

## 7. Not done / left for others

- **Continuous judging:** `media-judge.timer` stays disabled. This is the lead or Cyril's call, after §4.
- **Music and H3-audio judging** (CLAP/ASR, 586 groups): parked until a GPU is available for prep (a P40 after run 0, or a media-gate job). There is no CPU CLAP/ASR path, by choice, because of host RAM.
- **Sprite tier vision:** still `400 no_vision`. Opening it is the Sprite lead's call.
- **Native `input_video`:** not enabled. Video is still judged as one 8-frame grid image, as in spec 1.2.0.
- **Teacher id in gen-worker-text** (IQ3_S stamped on Q4 rows): not touched. It belongs to the gen lane.
- **Media-gate floor and Strata prompt-cache cap:** not touched. They belong to the run-0 readiness step.
- **Commit:** no git commit. ai-serve uses dated backups, and I wrote no chimera-omni or neural-forge code.
