# Run 0 support: Strata background data generation (D3)

Date: 2026-10-28.
- **First attempt, 06:20-08:00Z:** built, tested and installed the client.
- **Resume, 10:55-11:52Z:** verified the first batches, fixed the media lane and the edit check, and deployed 1.1.1.

Sources:
- Ruling `cyril-run0-decisions-2026-10-22` D3 ("[Cyril quote withheld]").
- Plan `RUN0_BUILD_PLAN.md` §2.3, §2.5, §3.3 and the Addendum.
- Scout report `support_scout_gen.md`.

## 0. Answer first

1. **`run0-strata-writer.service` is running and producing, and the first batch of every product is staged and verified.**
   - Code: chimera-omni **1.1.1 = `ce62959`**, branch `claude/run0-gen-20261028`, pushed to origin.
   - Deployed to `/home/cereal/omni-campaign/run0/gen-writer/code-ce62959`; `current` points at it.
   - It is a client only. It calls strata-gate and media-gate at the **background** tier (`strata_key_background` / `media_key_background`). No service was changed.
2. **Staged so far** under `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/batches/`:
   - 21 batches;
   - every batch's sha256 list re-hashed: 0 mismatches and 0 unlisted files.

| Product | Staged | What |
|---|---|---|
| `strata-intent-bank` | 256 decision twins over 11 slots; 16 clarify rows; 27 held ambiguous rows | COCO photo + Strata request → output decision |
| `strata-edit-instructions` | 322 | Grounded Qwen edits and phrasings of procedural operations |
| `coco-procedural-edits` | 128 pairs | PIL-rendered operation, Strata phrasing |
| `coco-edit-pairs` | 13 pairs (2 renders rejected) | Qwen-Image 2.1 edit renders via media-gate, `judged: false` |
| `strata-kd-gen` | 25 rows; 10,149 n_probs=64 positions | Verified math re-solves; token corpus + top-64 teacher cache |

3. **Since the first attempt, background admission has opened.** The 40 GiB floor is gone: strata-gate now reports `mem_floor_gib 12`, and media-gate admits background work again. Another lane made that RAM/Strata change. **I made no Strata or gate change, and I found none half-applied that belongs to this task.**

   The strata-llama restarts at 11:07 and 11:16Z were another lane's work. Strata came back healthy, and `/gate/status` now shows vision enabled for the background and cyril tiers (the D2 vision lane, CLAIM event 204067).
4. **Two defects found by reading real output were fixed:**
   - **1.1.0: no Qwen edit render ever ran under 1.0.0.** media-gate has one background slot, and gen-worker-media re-leases it as soon as a task ends. The once-a-minute check never found it free. With the media lane in its own thread, renders started within 10 minutes.
   - **1.1.1: the no-op check rejected correct small edits.** "Red gloves to bright blue" was correct but has a whole-image MAD of only 2.46.

## 1. What changed and where

| Path | What |
|---|---|
| `chimera-omni/scripts/run0_strata_writer.py` | 1.0.0 `b7f5142` (first attempt) → 1.1.0 `6c3f2ec` → 1.1.1 `ce62959` |
| `chimera-omni/tests/test_run0_strata_writer.py` | 13 → 16 tests |
| `chimera-omni/scripts/run0-strata-writer.service` | Unchanged. Installed at `/etc/systemd/system/` with Nice 15, idle IO class, MemoryMax 2G |
| server `/mnt/lexar/omni-campaign/run0-gen/strata-writer/` | State and journals. New file `media_state.json`, written only by the media lane |
| server `.../strata-writer/state.json.bak-*` | Backups taken before each restart and seal (11:14, 11:27, 11:47, 11:48Z) |
| omni_lab | Run 40193 (1.0.0) closed `pass` with 14 first-batch metrics. Run **40323** (`run0-strata-writer-6c3f2ec`, kind gen) is open for the continuing service |
| Relay | STATUS `6091cc00` on thread `52468f63…` describes the media-lane share and how to stop or roll back |

Changes in 1.1.0 and 1.1.1:

- **Media lane** (`render_edits` :1151, `_prepare_edit`, `_render_prepared`, `media_loop`; started in `serve`):
  - It runs in its own thread. A wait for GPU4 never holds up Strata units.
  - It polls media-gate `/gate/status` once a second. That call is side-effect free; a refused `POST /prompt` would count as a request. It polls only while all of these hold:
    - an edit is pending;
    - the caps allow a render;
    - the active profile is already `general`, so it never forces a switch from h3sol (H3 video) or speech (D4 TTS);
    - the busy slot is the only obstacle.

    Otherwise it polls every 30 s.
  - The source upload and graph are prepared before the gap opens.
  - Share caps: 15 renders/h and 0.2 duty (down from 30/h and 1/3).
  - In practice a render takes about 20 s, so the share is about 5 GPU4-minutes per hour, in a burst. gen-worker-media keeps the rest.
- **Edit check** (`image_checks` / `edit_changed`, about :769):
  - Pass when MAD ≥ 4, **or** when ≥ 0.2% of pixels (scaled to 256 px wide) changed by more than 80 levels.
  - Calibrated on the 15 real renders:
    - the two rejected correct local edits scored 0.0068 and 0.0048;
    - a resize + JPEG q60 round trip of the source scored 0.0000 (p99.9 diff ≤ 12).
- **Clarify screen.** A clarifying question may offer "a description". The leak screen for questions now drops only that one word (`QUESTION_LEAKS`). Requests keep the strict list.
- **Logging and cleanup.**
  - The "not admitted: idle" log line now says what it meant: gen-worker-text holds 3 background requests and the writer waits below 3.
  - Emptied ComfyUI nonce dirs are removed.

## 2. What I verified (commands and real output)

- **Tests.** 16 passed, rc 0, on the server, run from the deployed `code-ce62959` (Pascal venv, `CUDA_VISIBLE_DEVICES=""`):
  `python -m pytest -p no:cacheprovider -rA tests/test_run0_strata_writer.py`.
  - The deployed script's sha256 equals the git blob (`e07ec5f3…`).
  - The 1.0.0 deploy had been archived with CRLF line endings by `git archive` under autocrlf. That is harmless for Python. New deploys use `git -c core.autocrlf=false archive`.
- **Staging contract.** I re-hashed every file listed in all 21 batch manifests: `SHA_BAD_TOTAL 0`. Every manifest has schema `chimera_omni.run0.staging.v1` and `authorship_checked: true`.
- **KD batch** (`strata-kd-gen/v1/batches/b20261028T111504Z`), read back through **v2_trainer's own** `TokenCorpus` + `TopKTeacherCache`:
  - teacher `strata-q4kxl-abl-v2`, top_k 64, vocab 248,320;
  - 25 rows, 10,149 positions;
  - stored positions start exactly at the predictor of the first response token, with no gaps;
  - the target mask is off over the prompt and on over the response;
  - candidates are sorted;
  - the sampled next token is in the top-64 at **100%** of positions;
  - top-1 equals the next token at 97.0% (sampled at T 0.7);
  - top-64 probability mass at p5 is 1.000.

  Decoded rows are correct, e.g. pickles: 30 × 20 g = 600, matching the sandbox answer; LCM(18, 24).
- **Intent bank samples read** (3 per slot, plus one image across slots).
  - Example: "imagine if this were a cyberpunk scene, generate fresh art based on the layout" → `[image]` redraw. "Please remove the fence in the background." → `[image]` edit. "ambient audio for this scene, no music just room tone" → `[audio]` sound.
  - The blind label check agreed with the slot exactly on 243/248. The other 5 are joke vs describe, which are both `[text]`.
  - Real confusions are held out, e.g. "Add the sound of a dripping faucet to this image." was predicted as edit and went to `ambiguous.jsonl`.
- **Images viewed:**
  - 4 procedural pairs (hue_shift, invert, crop_center, mirror): all correct and matching their instruction.
  - 6 Qwen edit pairs:
    - recolour lettering, clock → moon, street → grass, walls → white: faithful local edits at the source aspect (1024x768, no crop);
    - "night → daylight": weak, but it passed (this needs the judge);
    - the rejected "gloves → blue": actually correct, which led to the 1.1.1 fix.
  - ComfyUI host output under `aiserve/run0-edit` holds 0 files after the renders, so `/` does not fill.
- **Live behaviour:**
  - With gen-worker-text holding 3 background requests, the writer yields: it starts only below 3.
  - gen-worker-text logged 0 "strata-gate busy" lines in the last 30 min.
  - KD: 25 of 42 units accepted; 16 rejections were the 512-token cap, 1 a wrong answer.
  - Writer RSS: 30 MB idle, 0.4-0.6 GB while working.

## 3. RAM each product needs (for the readiness step)

- **The writer:** at most 0.6 GB host RAM (cgroup MemoryMax 2G). No GPU.
- **Intent / edit-instruction / KD units:** ordinary Strata background requests. The 512-token KD responses carry n_probs (about 40x the gate's capture estimate per chunk). That is why `KD_N_PREDICT` stays at 512.
- **Qwen edit renders:** run inside the existing `general` ComfyUI container on GPU4. They add no host RAM beyond that container, but they need media-gate's admission. Since the RAM floor was removed, admission opens.

## 4. What I could not verify / left out

- **Throughput over a full day.** Before 11:07Z the gate's thermal level `no_new_background` paused Strata units often. Measured so far: about 9 intent, 30 edit-instruction and 14 KD units per Strata-busy hour, and up to 15 renders/h.
- **The 2 renders rejected under 1.0.x's check stay rejected** (gloves, soda cans). The new rule applies to new renders only.
- **Not built:**
  - showing the photo to Strata (vision now exists at the background tier; the intent bank still writes requests from the 5 COCO captions). A few requests name things the captions mention but that may sit elsewhere in the frame, e.g. "the menu card in the corner".
  - text-only intent inputs;
  - KD on code or long-context items;
  - TTS (D4 is a separate service).
- **Stop is not instant.** A `systemctl stop` during a long Strata stream can hit `TimeoutStopSec=120` and be SIGKILLed; it happened once at 11:27Z. State is written atomically and the unfinished unit is not journaled, so it reruns.
- **exFAT cluster overhead.** The Lexar's 1 MiB clusters inflate small files: about 0.45 GB on disk for about 60 MB of data so far. Expect roughly 1 GB/day on disk at the current rate.

## 5. Decisions and flags for the lead

1. **Clarify rows** (`clarify.jsonl`, decision `[text]` + Strata question) are staged with `needs_ruling`.
2. **Edit pairs are unjudged** (`judged: false`); area B decides the weight until D2's Strata judge lands. "Night → daylight" shows the pixel checks are not a meaning check.
3. **Edit-pair geometry.** Area B's `image_ref` puts a generated result into a cropped output bucket (e.g. source decodes at 512x416 uncropped, result at 512x384 `crop: true`), so pixels do not correspond exactly across the pair. This is area B's grammar choice; worth a look before edit training.
4. **Teacher identity.** Register `strata-q4kxl-abl-v2` as an omni_lab model key, and decide whether it shares the `strata` KD weight with the IQ3_S caches. There is no `lse.f32`: llama-server returns normalised log-probabilities.
5. **KD yield** is about 60%; the losses are the 512-token cap. Raising the cap needs the Strata lane to strip n_probs from capture first.
6. **gen-worker-text is still broken from the Q4 switch** (scout §2: the toolcall GGUF lock, and the IQ3_S teacher id stamped on its batches). Not fixed here; that is another lane's code.
7. **Media share.** 15 renders/h and 0.2 duty, posted on Relay for the media lane to change if it wants.

**Stop:** `sudo systemctl disable --now run0-strata-writer.service` (resumable).
**Rollback:** point `current` at `code-6c3f2ec` or `code-b7f5142`, then `systemctl restart`. Backups of `state.json` and `media_state.json` sit next to them.
