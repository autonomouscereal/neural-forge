# Run 0 support: D4 TTS caption renders (concept-interleaved)

Date: 2026-10-28, 07:00-07:50Z. Ruling: `cyril-run0-decisions-2026-10-22` D4 ("4 sure"). Plan: `RUN0_BUILD_PLAN.md` D4, §2.5 (concept-interleaved, 2 %), B-3 `build_concept`. Scout: `support_scout_gen.md` §5-6.

## 0. Answer first

1. **The job is built, verified offline and running.** It is the systemd user unit `run0-tts-captions.service`: enabled, survives reboot (linger is on), Nice 19, idle IO, MemoryMax 1G, swap 0. It uses about 75 MB RSS.
2. **It has not rendered a clip yet, and cannot until media-gate admits background work.** `/gate/status` says `background_admissible: false, "MemAvailable 28.8 GiB < 40"`. This is the same 40 GiB floor that has blocked every background lane since 01:46Z. The job polls `/gate/status` once a minute and sends nothing until both of these hold:
   - the gate says background work is admissible;
   - there has been no foreground request for 120 s.
3. **When run 0 starts, admission will not open by itself.** Run 0 is planned to take about 20 GiB of host RAM, which puts MemAvailable near 9 GiB. That is far under the 40 GiB admission floor and under the 20 GiB level where media-gate preempts background work. So D4 produces nothing until the media lane re-derives media-gate's `mem_floor_gib` for the new RAM layout. This is a media-gate config change: the task forbade it for me, and it is the media lane's call.
4. **Everything after the TTS call is proven on real data:**
   - plan selection;
   - the clip checks;
   - a lossless FLAC write;
   - the concept record, which passes area B's `check_records`;
   - area B's `load_media`, which decodes the speech ref to 38.4 kHz with the exact row count.

   The live TTS → ASR → WER loop has only been exercised as far as the gate's real refusal (the waiting path).

## 1. What was built

Code: chimera-omni branch `claude/run0-d4-tts-20261028` @ `2138486`. It was branched from area B's `96ed2b9` and pushed to origin.

- `scripts/run0_tts_captions.py`:
  - `plan`: `select_plan` at :219, `cmd_plan` at :707;
  - `run`: `Runner` at :377, with `why_not` :419, `gate_call` :449, `render` :474;
  - `manifest`: `write_manifest` :591;
  - `status` (:774) and `selftest` (:784).
- `tests/test_run0_tts_captions.py`: 14 tests.
- Deployed copy on the server: `/home/cereal/omni-campaign/run0/tts-d4/code-2138486/`. The script sha256 is `d2e75d3c…`. `run0_samples.py` is `a4517664…`, the same file area B built the COCO sources with.

How it works:

- **Plan (fixed, 12,000 items).** The input is the built `coco2017-caption` source. It is read only, and it was already screened against hold-out registry v3 (0 caption hits, 0 media hits).
  - Images are taken in order of `sha256(seed|coco2017:id)`, seed 20261028. This is a uniform sample, and any prefix of it is the same sample at a smaller n.
  - Each image gets one caption, chosen by seeded RNG among its captions of 5-30 words. Planned captions run 7-30 words, mean 10.5.
  - Voices are assigned round-robin in score order: ryan, aiden, serena and vivian get 3,000 each, and every prefix of the plan is balanced.
  - The spoken text is the caption's own words, with the first letter capitalised and a closing full stop added.
  - `plan.jsonl` sha256 is `56121b79…`.
- **Each clip.** The TTS call is `POST /v1/audio/speech` with model qwen3-tts, the planned voice, language english, a WAV response, a seed derived per item and attempt, and no streaming. The background key is `MEDIA_GATE_BACKGROUND_KEY` (vault `media_key_background`), read from `/home/cereal/.config/ai-serve/media.env`. The foreground cyril key is never used. The clip then goes through these checks in order:
  1. PCM16 mono at 24 kHz.
  2. `speech_e2e.pcm_stats`: RMS at least -45 dBFS, at least 40 % voiced frames, and a duration that fits the text.
  3. `POST /v1/audio/transcriptions` with Qwen3-ASR-1.7B through the same gate at background tier. WER is computed with speech_e2e's normaliser and number spelling, and the clip is kept only at WER ≤ 0.10. The ASR text is a check only and is never a target.
  4. FLAC encode, decoded back and required to give identical s16 samples.

  An item gets 2 attempts with different seeds. Then:
  - an accepted clip appends one row each to `items.jsonl` and `records.jsonl`, with fsync;
  - an item that fails both attempts goes to `rejects.jsonl` with each attempt's reason.

  The clip checks and normaliser are imported from `ai-serve/speech/speech_e2e.py` (sha `bfc519fc…`, recorded), not copied.
- **Gate answers** (`classify` :301):
  - 409 busy, and 503 preempted, timeout or error: wait `retry_after_s`, then retry the same attempt. Five non-admission gate errors fail the attempt.
  - 502 `bad_audio`: fails the attempt.
  - 400: rejects the item.
  - 401/403: exit 3, which the unit does not restart into.
  - A Sprite or cyril request preempts the synthesis inside the gate, and the client simply retries later.
- **Records.** They are built by `run0_samples.build_concept` (task `concept_interleaved`, provenance `media-teacher`), then `seal` with `origin coco2017:<id>`, then `check_record`.
  - The image ref is the COCO file in place, copied from the caption source.
  - The speech ref is `kind file` with `decode.native_rate 24000`. Its rows are the 80 ms units of the 38.4 kHz resample.
  - No source or origin tag matches the run 0 row denylist.
- **Duty cycle.** Clips run in blocks of 300, followed by a 3,600 s rest. The speech profile holds GPU4 for 300 s after its last use, so gen-worker-media's image jobs get 409 during a block. The block and rest lengths are flags. The media lane has been told on Relay and can ask for a different split.

## 2. Staging (the STAGING CONTRACT)

`/mnt/lexar/omni-campaign/run0-intake-staging/coco-tts-captions/v1/`. I created the `run0-intake-staging` root. Nothing is written to `/`, the NAS, or `run0/data/parts`.

| File | Content |
|---|---|
| `plan.json`, `plan.jsonl` | the fixed plan, its rules and seed, and the source `source.json` and `part.json` shas. Rewriting it is refused once any clip exists |
| `teacher.json` | full sha256 of every Qwen3-TTS-12Hz-1.7B-CustomVoice and Qwen3-ASR-1.7B `.safetensors`/`.json` file (TTS `model.safetensors` `38b1d597…`), media-gate 1.2.1 (`54642215…`), speech_worker (`28a47f8f…`) |
| `audio/<coco_id>-<voice>.flac` | 24 kHz mono s16. Expect about 70-110 KB per clip, roughly 1 GB for 12k |
| `items.jsonl` | per clip: image ref, caption, spoken text, voice, seed, attempt, file/PCM sha, duration/RMS/peak/voiced, ASR text + WER, gate request id, `X-Speech-Stats` |
| `records.jsonl` | `build_concept` records, ready for area B |
| `rejects.jsonl`, `state.json` | failures with reasons; live phase, wait reasons, counters |
| `manifest.json`, `README.md` | rewritten after every block and at exit. The manifest carries source, licences (COCO captions CC BY 4.0, Flickr images under Cyril's 10-14 ruling, Qwen3-TTS Apache-2.0), counts per modality/task/voice, bytes, the sha256 list, generator and teacher ids, sampling, hold-out v3, date, and `authorship_checked: true` |

There is no `READY.json`. Wiring the product into run 0 is area B's step: a `concept` subcommand of `run0_build_manifest.py` that reads `records.jsonl`, plus the `concept` share in `run0_build_manifest.py:836`, which is still 0.

## 3. Verification

- **Local tests:** `C:/Python314/python.exe -m pytest -q -rs tests/test_run0_tts_captions.py` gave 14 passed with 0 skipped. They cover:
  - the plan is deterministic, independent of n for its prefix, balanced across voices, and drops images that have no eligible caption;
  - the gate-answer mapping;
  - the speech-ref rows equal `media_sources.audio_rows` at 38.4 kHz;
  - the record passes `check_record` and `run0_data.check_records`;
  - a FLAC round trip is lossless, and a corrupted round trip leaves no file behind.
- **Server plan:** run as `systemd-run --user … run0_tts_captions.py plan --count 12000`. Selection took 6 s and hashing the teacher about 60 s. Checked results:
  - 12,000 unique ids;
  - 3,000 per voice;
  - hold-out block `{registry 3, caption_hits 0, media_hits 0}`.
- **Server selftest:** run as `run0_tts_captions.py selftest` on an existing Qwen3-TTS Ryan render, `cyril-speech/…-000-ryan/original.wav`. This is a validation artifact only, written to `/home/cereal/omni-campaign/run0/tts-d4/selftest/`.
  - The WAV is 24 kHz and 5.52 s long, at -29.6 dBFS with 52.5 % voiced frames, so the clip checks pass.
  - The FLAC is 110,254 B and gives identical PCM (`10b659ed…`).
  - The resulting record passes the grammar check.
- **Area B decode:** run with the Pascal venv, CPU only, under a 4G cgroup, from `code-b3`.
  - `check_records([record])` returns 1.
  - `load_media(speech_ref)` gives float32 (211,968,), which is 5.52 s at 38.4 kHz. Its 69 rows equal the ref's 69 rows.
  - The image ref decodes to (416, 640, 3).
  - The header is `speech 38400hz mono`.
- **Live gate:** the `status` command and the running unit both get the real refusal `MemAvailable 28.x GiB < 40`. The job logs it once and counts it in `state.json` `waits`.

## 4. Not verified

- **No real TTS or ASR clip has gone through the job.** The gate has not admitted background work at any point today. The first admitted block will be the first end-to-end proof. Check:
  - `items.jsonl` rows;
  - `rejects.jsonl` reasons;
  - the WER distribution in `manifest.json`;
  - listening to a few FLACs.
- **The host-RAM cost of loading the speech workers** (TTS and ASR, both 1.7B, on GPU4) on top of the parked ComfyUI's 57.6 GiB has not been measured. Media-gate preempts below 20 GiB.
- **Throughput is a projection:** about 5 s of TTS plus about 1-2 s of ASR per clip, so a 300-clip block takes about 40-60 min. At one block per ~2 h of admitted time, 12k clips need about 40 blocks.

## 5. Decisions for the lead

1. **The media-gate RAM floor.** Both of these are media-lane changes, not mine:
   - lower `mem_floor_gib` for the speech profile, or for everything, to match the Q4 Strata plus run 0 layout;
   - or free the parked ComfyUI's host-RAM cache.

   Until one happens, D4, gen-worker-media and the TTS ASR checks all stay at 0.
2. **The duty cycle against gen-worker-media**: the default is 300 clips, then 60 min of rest.
3. **The volume.** I planned 12k clips. The plan's need is at least about 3.6k unique a week, and its aim is 10k. A larger product would be a v2 plan, because v1's plan is fixed once rendering starts.
4. **WIT** was not built as a run 0 source, so this product is COCO only. That matches §2.5's "COCO image + caption".

## 6. Operating it

- Status: `/home/cereal/.venvs/ai-serve-media/bin/python /home/cereal/omni-campaign/run0/tts-d4/code-2138486/scripts/run0_tts_captions.py status`
  - or read `state.json` in the stage dir;
  - or `journalctl --user -u run0-tts-captions`.
- Stop cleanly: `touch <stage>/STOP`. The job exits 0, which is not restarted. To resume, remove `STOP` (or pass `run --clear-stop`), then `systemctl --user start run0-tts-captions`. It resumes from the ledgers.
- Remove: `systemctl --user disable --now run0-tts-captions`. The unit file is `~/.config/systemd/user/run0-tts-captions.service`.
- Re-hash: `… manifest --verify` re-hashes every FLAC against the ledger.
- omni_lab run **40192** (`run0-d4-tts-captions-v1`, kind gen) is open. Close it when the plan completes: `omni_lab.py log-run end run0-d4-tts-captions-v1 --status pass --metric accepted=<n> --metric wer_mean=<x>`.
- Relay STATUS posted to the media lane: message `cae96db0`, thread `52468f63`.

## 7. Re-check, 2026-10-28 10:47Z (workflow resume)

Nothing in sections 0-6 changed. I re-checked these against the live server; no code, unit or service was touched:

- `run0-tts-captions.service` has been active since 07:47:39Z. It uses 65 MB of its 1 G cap with swap 0, and 1.6 s of CPU in 3 h.
- `state.json` (updated 10:46:52Z): phase `waiting` on `coco2017:212834`, `accepted 0`, `rejected 0`, 180 one-minute waits. Every wait was `gate: MemAvailable N GiB < 40`; the last read was 28.9 GiB.
- The stage dir has 0 FLACs and no `items`, `records` or `rejects` rows. The manifest and plan are unchanged from 07:21Z.
- Branch `claude/run0-d4-tts-20261028` @ `2138486` matches on origin (`git ls-remote`).
- Host at 10:47Z:
  - MemAvailable 29.9 GiB; SwapFree 0.27 GiB of 7 GiB.
  - P40s GPU1/2/3/5 idle (5 MiB each), so run 0 had not started on them yet.
  - GPU4 ComfyUI held 18.6 GiB VRAM and 57 GiB RSS.
  - `/` had 113 GB free and the Lexar 227 GB. The Lexar was 381 GB the day before; the audio-gap downloads are the likely consumer. D4 writes are at most about 1 GB in total.
- SSH to the server timed out 5 times out of 6 after 10:50Z. The port-22 banner, ICMP and the :28400/:18400 TCP connects all answered, so I stopped polling to avoid adding load. A follow-up should check host memory pressure before blaming the job, which holds 65 MB.
- Relay: `session.py sync` returned only August posts, and `show --message-id cae96db0` returned `ValueError`. I could not confirm whether the media lane answered the floor request.

The blocker is the same as in §5.1: media-gate's 40 GiB background floor against about 29 GiB available. Once run 0 takes its about 20 GiB, the gap grows. The job needs no change when the floor moves; it starts its first 300-clip block within a minute of admission.
