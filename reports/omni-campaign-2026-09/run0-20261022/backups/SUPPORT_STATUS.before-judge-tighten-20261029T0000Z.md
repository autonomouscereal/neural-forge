# Run 0 support lanes: status of Cyril's six decisions

Checked live on the AI server at 2026-10-28 13:09Z. Covers ruling `cyril-run0-decisions-2026-10-22` (Cyril: "[Cyril quote withheld]").

Lane reports, all in this folder: `support_judge.md`, `support_audio.md` (+ `support_audio_verify_20261028T110828Z.json`), `support_gen.md`, `support_tts.md`, `support_corpus.md`.

Staging root: `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v<N>/`. Each product has `manifest.json` (source, licence, counts, sha256 list, generator/teacher ids, date) and `README.md`. Nothing was written to the trainer's live shard dirs, to `/` beyond small code dirs, or to the NAS.

Host at 13:09Z:
- MemAvailable 35.6 GiB, swap free 1.2 GiB.
- `/` 113 GB free, Lexar 208 GB free.
- GPU1/2/3/5 idle at 5 MiB, so run 0 is not on the P40s yet.
- GPU0 23.7 GiB and GPU6 44.9 GiB used (Strata).
- GPU4 11.2 GiB used (media-gate).

## Summary

| # | Decision | State | Blocker |
|---|---|---|---|
| D1 | use the K3 we have | owned by the run-0 workflow, no support product | L1 ratification not yet written in the rulings file |
| D2 | Strata as judge, vision first | **done**: vision live on CPU projector, judge ported | `media-judge.timer` left **disabled**; enabling it accepts the measured leniency (Cyril or lead decides) |
| D3 | Strata generates data, background | **running**: `run0-strata-writer.service` 1.1.1 | none hard; teacher-id registration and clarify-row ruling pending |
| D4 | TTS caption renders | **running**: first block done (278 accepted, 22 rejected) | media-gate's 40 GiB background floor; MemAvailable is 35.6 GiB now, so the next block will probably wait |
| Audio gap | download real SFX + AV data | **done**: 7 tier-1 sets, 4 products, verified with 0 problems | tier 2 not started (Lexar budget) |
| Corpus | train on our corpus | **done**: 3 derived views, sealed and registered | none |

None of these products is in a run-0 shard yet. Area B's tools wire all of them once the run-0 branch is integrated (see "Wiring" below).

## D1. "[Cyril quote withheld]"

**Done (plan level, not a support product).** Run 0 uses the cached K3: 1.89M positions, provisional bf16, at `corpus/k3/pass-0..3` plus the xtok caches. The run-0 plan's teacher weights are `k3present: k3 1.0, strata 0.3, q38 0.1`. The plan states that K3 is below 1% of the KD signal (Q1). It is recorded as a run-0-only deviation from "K3 primary" in the 10-28 entry of `STAGE_D_RULINGS_2026-10-12.md`.

**Running:** nothing. GPU6 is not borrowed.

**Open:** lead decision L1, ratifying the provisional bf16 K3 cache (gate: matched mass at least 0.90, measured 0.91), is not yet written in the rulings file. If it is revoked, the K3 bucket goes to 0. This belongs to the run-0 workflow.

## D2. Strata as the media judge, vision first

**Done.**
- **Strata vision is live** (since 11:30Z). The mmproj runs on the CPU (`--no-mmproj-offload`), so Sprite keeps its VRAM: GPU0 and GPU6 are unchanged.
  - Host RAM: about +1.0 GiB, held while vision is on.
  - Images work at background and cyril tier; video input is refused with 400.
  - The Sprite tier still returns `no_vision`. Opening it is the Sprite lead's call.
  - The rollback was tested once (11:15–11:23Z): every file returned to its backup sha, then the change was re-applied. Rollback command: `/home/cereal/ai-serve/work/strata-judge-0c189f21/strata_vision.sh rollback`. Backups: `/home/cereal/ai-serve/backups/strata-vision-20261028/`.
  - The earlier attempt's gate change (background RAM floor 40 → 12 GiB) was found fully applied. Nothing half-applied remains.
- **The judge runs on Strata** at background priority through strata-gate.
  - Same rubric, thresholds, outputs and intake gating as before.
  - It no longer claims any P40, so **GPU5 is free for run 0**.
  - A preempted or busy answer is never recorded as a verdict.
  - Judge suite 32/32.
- **Calibration** (80 items the old Qwen3.6-27B judge had already scored): 61/80 agree (76%, kappa 0.525).

  | Kind | Agree |
  |---|---|
  | edit | 19/20 |
  | video | 16/20 |
  | image | 13/20 |
  | mesh | 13/20 |

  Strata is more lenient: 12 of 40 old rejects passed, and 7 of 40 old passes were rejected. Of 8 lenient cases read by hand, Strata was clearly wrong on 2. The scout's bar (85%, no lenient drift) is **not met**.
- **One real run:** 14 groups judged. 5 passed and are published to Lexar intake (4 images, 1 video); 9 rejects moved to `rejected/`.

**Running:** nothing on a schedule. `media-judge.timer` is `disabled`/`inactive`.
- Check: `systemctl is-enabled media-judge.timer; systemctl list-timers media-judge.timer`

**Staged:** no staging product. Judged items publish into the existing Lexar intake (`/mnt/lexar/corpus-store/chimera-intake/`), which `run0_intake_sync.py` already scans.

**Blockers and decisions:**
- **Decision (Cyril or lead):** enable continuous judging with `sudo systemctl enable --now media-judge.timer`. It runs every 15 min, background only, with no P40. Enabling it accepts the leniency above for training intake. The alternative is a tightened image/mesh spec plus recalibration.
  - Backlog: 1,756 groups, about 24 h of background time (median 39 s per item).
  - Until the timer is on, run-0 media intake stays frozen (plan Q2).
- Music and H3-audio groups (586) stay parked, because their audio scoring needs a GPU.
- **RAM:** projector about 1.0 GiB; judge client about 50 MB (capped at 3 GiB); mesh prep is inside that cap (peak not measured). No VRAM.
- omni_lab: build 40339, calibration 40338, evals 40340/40341/40343/40344, publish 40342.

## D3. Strata generates run-0 data at background priority

**Done.**
- `run0_strata_writer.py` 1.1.1, chimera-omni branch `claude/run0-gen-20261028` at `ce62959`, pushed.
- Deployed sha matches the git blob; 16/16 tests pass on the server.
- It runs at background priority through strata-gate. Edit renders go through media-gate's Qwen-Image profile, capped at 15/h and only while that profile is already loaded.
- First batches are re-hashed with no mismatch.

**Running:** `run0-strata-writer.service` (system unit, `active`).
- Check: `systemctl status run0-strata-writer.service --no-pager; journalctl -u run0-strata-writer -n 20 --no-pager`
- Stop: `sudo systemctl disable --now run0-strata-writer.service`. It resumes where it left off.
- Roll back: point `/home/cereal/omni-campaign/run0/gen-writer/current` at `code-6c3f2ec` or `code-b7f5142` and restart.
- Throughput so far, per busy hour: about 9 intent units, 30 edit instructions, 14 KD units, and up to 15 renders.

**Staged** (counts as of the gen report, about 11:48Z; each product manifest lists its sealed batches by sha):

| Product | Manifest | Content |
|---|---|---|
| `strata-intent-bank` | `.../strata-intent-bank/v1/manifest.json` | 256 decision twins over 11 slots; 16 clarify rows (need a ruling); 27 held as ambiguous; blind label check 243/248 |
| `strata-edit-instructions` | `.../strata-edit-instructions/v1/manifest.json` | 322 Strata-written edit instructions |
| `coco-procedural-edits` | `.../coco-procedural-edits/v1/manifest.json` | 128 procedural COCO edit pairs |
| `coco-edit-pairs` | `.../coco-edit-pairs/v1/manifest.json` | 13 Qwen-Image edit pairs (2 rejected), **unjudged** |
| `strata-kd-gen` | `.../strata-kd-gen/v1/manifest.json` | 25 verified math rows, 10,149 positions with Strata top-64 probabilities (teacher `strata-q4kxl-abl-v2`); read back with the trainer's own readers |

**Blockers and decisions:**
- Register teacher id `strata-q4kxl-abl-v2` in omni_lab, and decide whether it shares the `strata` KD weight with the old IQ3_S caches.
- Clarify rows need a ruling before use.
- Edit pairs are unjudged. Area B sets their weight, or runs them through the Strata judge once the timer is on.
- Area B's crop gives 512x384 for the edited image but 512x416 for the source, so pixels do not align across a pair.
- The 512-token KD cap drops about 40% of math re-solves. Raising it first needs the Strata lane to stop the gate capturing `n_probs`.
- gen-worker-text is still broken from the Q4 switch, and its rows carry the old IQ3_S teacher id. That code belongs to another lane.
- **Plan deviation, for the lead:** plan §2.5 and §6 chose "no new strata-gate client" (jobs on the gen-worker-text queue). The writer is a new client, built because gen-worker-text could not carry this work.
- Not built: photo-grounded intent requests (now possible with vision), KD on code and long-context items, text-only intent inputs.
- **RAM:** writer at most 0.6 GB (30 MB idle). Renders run inside the existing GPU4 container.

## D4. Qwen3-TTS renders of COCO captions

**Done.**
- `run0_tts_captions.py`, chimera-omni branch `claude/run0-d4-tts-20261028` at `2138486`, pushed; 14 tests pass.
- Plan: 12,000 COCO images, one caption each, split 3,000 each over 4 voices; 0 hold-out hits.
- Every clip is checked: 24 kHz mono, loudness, voiced share, duration fit, a lossless FLAC write, and an ASR round-trip that keeps the clip only at WER ≤ 10%.
- **First real block finished** (11:36–12:50Z, while host RAM was high): 278 accepted, 22 rejected, 279 MB of FLAC.
  - Voices: ryan 68, aiden 72, serena 68, vivian 70.
  - Rejects are WER > 0.10 after retries, for example "racquet" vs "racket" (WER 0.111).
  - The clips have not been listened to yet.

**Running:** `run0-tts-captions.service` (user unit, `active`). Phase at 13:09Z: `resting`, a 60-min duty-cycle rest after a 300-clip block.
- Check: `systemctl --user status run0-tts-captions.service --no-pager; cat /mnt/lexar/omni-campaign/run0-intake-staging/coco-tts-captions/v1/state.json`

**Staged:** `coco-tts-captions/v1/manifest.json` reports planned 12,000, accepted 278, rejected 22, pending 11,700; task `concept_interleaved`.
- Licences: COCO captions CC BY 4.0, Flickr images under the 10-14 ruling, Qwen3-TTS Apache-2.0.
- It also holds `records.jsonl` (area B record schema), `items.jsonl`, `rejects.jsonl` and `teacher.json` (sha256 of every model file).

**Blockers:**
- **media-gate refuses background work while MemAvailable < 40 GiB.** It was 35.6 GiB at 13:09Z, so the next block will probably wait.
  - The readiness step owns the Strata RAM change that would admit it; the media lane owns the floor.
  - Once run 0 takes its planned ~20 GiB, the gap widens.
- At about 300 clips per 75-min busy period plus 60-min rest, 12k clips need about 40 block cycles. That is days of admitted GPU4 time.
- **RAM:** job at most 1 GB (65 MB now). The extra RAM the speech workers take on top of ComfyUI is not measured.

## Audio gap: real sound-effect and audio-video data

**Done.** Seven tier-1 datasets were downloaded from HF mirrors and official archives (no YouTube scraping) into 4 products.
- Raw audio: 150.3 GB on the Lexar, against a 152 GB budget.
- Code: chimera-omni `claude/run0-audio-20261028`, `0897c09` plus verifier `7c2fb16`, pushed.
- An independent verify run (11:08–11:15Z) found 0 problems:
  - all staged files match their manifests, and all counts recount exactly;
  - 34/34 raw archives re-hash correctly;
  - 45/45 random trainable clips decode through their real container;
  - publisher checksums match.
- omni_lab run 40189 closed as pass.

**Running:** nothing. Check: `python3 /home/cereal/omni-campaign/audio-gap/run0_audio_gap.py status`

**Staged:**

| Product | Manifest | Sources | Records | Trainable | Held out |
|---|---|---|---|---|---|
| `sfx_caption_v1` | `.../sfx_caption_v1/v1/manifest.json` | Clotho, AudioCaps | 52,435 | 50,494 | 1,941 |
| `video_caption_v1` | `.../video_caption_v1/v1/manifest.json` | MSR-VTT (no audio), AVCaps visual | 12,061 | 8,862 | 3,199 |
| `av_caption_v1` | `.../av_caption_v1/v1/manifest.json` | AVCaps, AudioCaps on AudioSet video, VALOR-32K | 41,048 | 31,950 | 9,098 |
| `av_label_v1` | `.../av_label_v1/v1/manifest.json` | VGGSound shards 00–01 | 20,000 | 16,801 | 1,536 (+2,360 `dup_of`) |

Notes:
- AVCaps' 2,061 clips are in both `video_caption_v1` and `av_caption_v1`; count them once.
- AudioCaps is 24 kHz.
- The MSR-VTT mirror has no audio track.
- VALOR loses 5,341 clips to the VGGSound-test hold-out (video-id match).
- All remaining captions are human-written.
- Most sources are non-commercial or research-only; licences are recorded per source.

**Blockers and decisions:**
- Tier 2 (FSD50K first) is not started, because the Lexar budget is spent.
- Cyril or the lead to decide:
  - loosen the VALOR/VGGSound hold-out to time-window overlap?
  - what generation weight for 24 kHz AudioCaps?
  - when to start tier 2?
- **RAM:** no standing RAM; readers stream one clip at a time, estimated well under 1 GiB per reader (not measured).

## Our corpus: "[Cyril quote withheld]"

**Done.** Three derived views, each with a manifest, README, check receipt and `sealed` marker, registered in omni_lab.
- Code: chimera-omni `claude/run0-corpus-views-20261028`, `47539bb` … `07b69bb`, pushed. 29 tests pass.
- A registry-media v1 bug was found and fixed: 21 held-out files and 1 orphan file sat unreferenced in its media folder. v1 was withdrawn before use, and v2 holds 2,401 files, all referenced, none held out.

**Running:** nothing. Check: `ls /mnt/lexar/omni-campaign/run0-intake-staging/{pdf-corpus,registry-text,registry-media}/*/sealed`

**Staged:**

| Product | Manifest | Content | omni_lab |
|---|---|---|---|
| `pdf-corpus/v1` | `.../pdf-corpus/v1/manifest.json` | 5,839 papers as text, 126.2M Qwen3.5 tokens (29 vendor model cards refused, 1 garbled) | dataset 560, version 569, run 40332 |
| `registry-text/v1` | `.../registry-text/v1/manifest.json` | 10,705 public-source documents + 70 math chat records, 5.99M tokens | dataset 561, version 570, run 40333 |
| `registry-media/v2` | `.../registry-media/v2/manifest.json` | 1,397 rows of public audio (SonicMaster 1,242, MTG-Jamendo 155), 2,401 files, 6.72 GB | dataset 562, version 571, run 40335 |

Already in run 0, so no new view was built:
- the curated set (same registry);
- `toolcall_flash_v1`, `bfcl_native_v1` and the Strata tool intake (2,775 records at the 08:13Z sync);
- Cyril's speech (382 records from 191 renders).

The memory tool calls stay excluded by §2.5 until Cyril confirms the compliant reading of "[Cyril quote withheld]".

**Decisions:**
- `registry-text/v1` **replaces** the allowed part of `text_clean_v1`; adding it on top would count about 10.7k documents twice.
- OpenR1-Math and DeepMath solutions are DeepSeek-R1 outputs. They are kept; drop them if "small-model generations" is read more widely.

**RAM:** none at rest. Build peaks were about 4 GB (6 PDF workers), under 1 GB, and about 0.33 GB.

## Wiring into run-0 shards (area B, after the run-0 branch is integrated)

The staging products sit outside the path `run0_intake_sync.py` scans (`/mnt/lexar/corpus-store/chimera-intake/*/<batch>/READY.json`). Each one has to be named as a source in `run0_build_manifest.py` and go through `final_steps.sh` (guard → verify → intake register → seal `run0_data_vN.json`), or be adopted later through `run0_intake_sync.py` → `NEXT_MANIFEST` at a checkpoint STOP.

| Product | Target bucket (§2.5) | What area B must add or decide |
|---|---|---|
| all four audio products | **no bucket yet** (Q7: "no sfx / joint AV") | new sfx-understanding and AV buckets plus shares; readers for zip, tar-offset, parquet-row, 7z and tar.gz members; `source_sr` (24 kHz AudioCaps); fold the hold-out lists into **registry v3 before any training use**; count AVCaps once |
| `coco-tts-captions` | concept-interleaved (3) | a `concept` subcommand that reads `records.jsonl`; the share is still 0 at `run0_build_manifest.py:836`; re-sync as blocks land |
| `strata-intent-bank` | decision twins + intent variants (4) | exclude the 16 clarify rows until ruled on; keep the 27 ambiguous rows held |
| `strata-edit-instructions`, `coco-procedural-edits` | image gen + edit (12) | fix the 512x384 vs 512x416 crop mismatch |
| `coco-edit-pairs` | image gen + edit | weight for unjudged pairs, or judge them first |
| `strata-kd-gen` | text KD (24) | `teacher_caches.strata` entry with id `strata-q4kxl-abl-v2`; teacher-weight decision |
| `registry-text/v1` | text KD (24) | **replace** the allowed part of `text_clean_v1`; it has no teacher caches yet (CE, or KD after a cache pass) |
| `pdf-corpus/v1` | text KD / long lane | no teacher caches; long docs fit the 16k lane only after S0 |
| `registry-media/v2` | music in/out (Jamendo); SonicMaster restore pairs have **no bucket** | a bucket decision for audio-restore pairs |
| judged media intake | image / video / 3D buckets | already on the `run0_intake_sync.py` path; flows only while the judge timer is on |

## Commits and records

- **Pushed to chimera-omni origin:**
  - `claude/run0-gen-20261028` @ `ce62959`
  - `claude/run0-d4-tts-20261028` @ `2138486`
  - `claude/run0-audio-20261028` @ `7c2fb16`
  - `claude/run0-corpus-views-20261028` @ `07b69bb`
- ai-serve changes use its dated-backup convention (`/home/cereal/ai-serve/backups/strata-vision-20261028/`).
- No weights were committed.
- **Not committed:** the lane reports and this file. The neural-forge checkout is on Codex branch `codex/lrc-blank-compression`, and `run0-20261022/` is untracked there.
- **omni_lab:** 40189 (audio), 40193 + 40323 (writer), 40192 (TTS, open), 40332/40333/40335 (corpus), 40338–40344 (judge).

## RAM ask for the readiness step

| Product | Standing host RAM |
|---|---|
| Strata vision projector | about 1.0 GiB while vision is on |
| judge client | ≤ 3 GiB cap (about 50 MB typical) |
| Strata writer | ≤ 0.6 GB |
| TTS job | ≤ 1 GB cap (65 MB now), plus speech workers inside media-gate (not measured) |
| audio / corpus products | none at rest; readers stream |

None of these products takes any VRAM of its own.

The binding constraint is media-gate's 40 GiB MemAvailable floor for background work. It gates D4, the edit renders and gen-worker-media. The readiness step's Strata RAM change should leave MemAvailable above 40 GiB after run 0 takes its share, or the media lane has to lower the floor.
