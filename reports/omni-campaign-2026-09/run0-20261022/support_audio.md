# Run 0 support: the audio gap (downloads, verification, staging)

Date: 2026-10-28. Ruling `cyril-run0-decisions-2026-10-22` ("fix the audio gap"). Scout: `support_scout_audio.md` (tier 1).

## 0. Answer first

- **All seven tier-1 datasets are downloaded, verified and staged. Nothing is still running.** The earlier attempt (cut off by the usage limit) had already finished the run at 08:15:56Z (`run: finished, 4/4 products staged, 0 error(s)`). This resume found that, did not re-download anything, and checked the result independently.
- **Independent check, 2026-10-28T11:08-11:15Z: 0 problems.** 45/45 sampled clips decode through their real container; 34/34 raw files re-hash to the manifest sha; every count in every manifest recounts exactly. Report: `support_audio_verify_20261028T110828Z.json` (sha256 `638fdc30…`).
- **Publisher checksums re-checked from the workstation:** the HF LFS sha256 of `vggsound_00/01.tar.gz`, `VALOR-32K-videos.tar`, `MSRVTT_Videos.zip` and AVCaps `train_videos.zip` match the manifests byte for byte.
- **Disk:** 150.3 GB of raw audio on the Lexar (budget 152). Lexar now has 227 GB free, `/` 113 GB (nothing was written to `/`), NAS untouched.
- **RAM:** the products need no standing RAM. See §6.
- **Four products** under `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/`, each with `records.jsonl`, `holdout/ids.jsonl`, `holdout/texts.jsonl`, `manifest.json` and `README.md`. Media stay packed in the raw archives; each record names its archive, the archive's sha256 and the member, tar offset or parquet row.

| product | sources | records | trainable | held out | dup_of | captions / labels |
|---|---|---|---|---|---|---|
| `sfx_caption_v1` | Clotho v2.1, AudioCaps | 52,435 | 50,494 | 1,941 | 0 | 81,279 audio captions |
| `video_caption_v1` | MSR-VTT, AVCaps (visual captions) | 12,061 | 8,862 | 3,199 | 0 | 192,967 visual captions |
| `av_caption_v1` | AVCaps, AudioCaps ∩ AudioSet video, VALOR-32K | 41,048 | 31,950 | 9,098 | 0 | 13,029 audio, 40,064 audio-visual, 7,599 visual |
| `av_label_v1` | VGGSound shards 00 + 01 | 20,000 | 16,801 | 1,536 | 2,360 | 20,000 class labels |

AVCaps appears in two products, as visual captions in `video_caption_v1` and as audio / AV captions in `av_caption_v1`. It is the same 2,061 clips, so count them once when you weight the mix.

## 1. Raw sources (all at `/mnt/lexar/omni-campaign/corpus/external/<ds>/`)

| ds | pinned source | files verified | bytes | publisher checksum | archive test | probe (measured) |
|---|---|---|---|---|---|---|
| clotho | Zenodo 4783391 (v2.1) | 10/10 | 7.06 GB | Zenodo md5 | `7z t` ok (3,839 + 1,045 + 1,045 wav) | 21/21 PCM 44.1 kHz mono, 15-29 s |
| audiocaps | HF `OpenSound/AudioCaps@b29b3243` | 475/475 | 43.94 GB | LFS sha256 | parquet metadata | **24 kHz** PCM, not 44.1 (see §4) |
| avcaps | HF `TUT-ARG/AVCaps@7a17eb6c` | 8/8 | 10.62 GB | LFS sha256 | zip ok | 21/21 h264 + AAC 44.1 kHz stereo, 20-90 s |
| msrvtt | HF `friedrichor/MSR-VTT@c1af215a` | 12/12 | 2.26 GB | LFS sha256 | zip ok | **no audio stream** (0/20), h264 |
| audioset_av | HF `chikuwa-AI/AudioSet_{balanced@92ad50b2,unbalanced@d931a088}_video`, only the AudioCaps ids | 6,689/6,689 | 11.96 GB | LFS sha256 per mp4 | every clip ffprobed | 6,670/6,689 have audio; AAC 44.1 kHz (5,226), 48 kHz (1,431), 32 kHz (13) |
| valor32k | HF `UnFaZeD07/VALOR-32K@25e55079` | 4/4 | 41.05 GB | LFS sha256 | tar headers ok, 32,327 mp4 | 20/20 h264 + AAC 44.1 kHz, ~10 s |
| vggsound | HF `Loie/VGGSound@2ca3012e` (shards 00, 01 + csv) | 4/4 | 33.36 GB | LFS sha256 | `pigz -t` ok, 10,000 mp4 each | 28/28 h264 + AAC 44.1 kHz, ~10 s |

Every file was hashed against its publisher checksum before it was renamed into place. Each dataset dir holds its `PLAN.json` (pinned revision, file list, checksums), `FETCHED.json` and `PROBE.json`.

## 2. Licences (also in each manifest's `licences`; record them on the run, as for MAESTRO)

| ds | licence |
|---|---|
| clotho | audio: per-file Freesound CC (listed in the metadata CSV); captions: Tampere University, non-commercial with attribution |
| audiocaps | CC BY-NC 4.0 (HF mirror); captions from cdjkim/audiocaps; audio is YouTube content |
| avcaps | CC BY-NC 4.0 (academic / non-commercial); videos from VidOR |
| msrvtt | Microsoft research use only |
| audioset_av | AudioSet labels CC BY 4.0; videos belong to their YouTube uploaders (the mirror states no licence); captions from AudioCaps (CC BY-NC 4.0) |
| valor32k | research only (upstream data terms are not stated; the HF mirror is tagged MIT); AudioSet-derived YouTube clips |
| vggsound | CC BY 4.0 (annotations); videos belong to their YouTube uploaders |

Most of this set is **non-commercial or research-only**. That is fine for the internal run, but it must go on the run record.

## 3. Provenance and hold-outs

- **Captions:** every caption is human-written (`author: public`). VGGSound class labels are `author: label`. The verifier counted authors across all records and found only those two.
- **Dropped:** 6,183 AVCaps GPT-4 summary captions were dropped at stage time and appear in no record. 20 AudioSet clips were dropped because they lack either an audio or a video stream.
- **Tags:** no product, source or hold-out bank name matches run 0's denied-tag guard (claude, codex, capture, refill, selfgen, sage, gpt, gemini, `memory_batch_`). This is checked both by the unit test and by the verifier.
- **Teachers:** none. No Strata, small-model, Claude/Codex or capture-spool text is in any product.
- **Hold-outs:** every test split is held out: Clotho evaluation, AudioCaps test, AVCaps test, MSR-VTT official test 7010-9999 (which contains test_1k), VALOR-32K test, AudioSet test and VGGSound test. So is any clip whose YouTube id is in another dataset's test split. `holdout/ids.jsonl` and `holdout/texts.jsonl` use the hold-out registry v3 row shapes; **area B must fold them into the next registry version before any training use.**
- **The cross-dataset rule costs VALOR-32K a lot:** 5,341 VALOR train/val clips are held out because their YouTube id is in VGGSound test, which is 17 % of VALOR's non-test clips. The rule matches on video id, not time window, so it is conservative. Loosening it to overlapping time windows is a lead decision; I left it strict.
- **`dup_of`** (`av_label_v1`, 2,360): VGGSound clips whose video is already used through a human-captioned product (AudioCaps 1,650, VALOR 706, MSR-VTT 4). The builder skips any record that has `dup_of`.

## 4. Measured caveats for the builder

- **AudioCaps:** the OpenSound mirror is **24 kHz** 16-bit PCM, band-limited to 12 kHz. The scout inferred 44.1 kHz; the probe measured 20/20 rows at 24 kHz, and so did the verifier (6/6). It is fine as audio input for `sound_caption`, but weak as a 38.4 kHz `sound_gen` target. There is no full-band copy without YouTube scraping. The note is in the manifest.
- **MSR-VTT:** this mirror's mp4s are **silent** (0/20 at probe, 0/6 at verify). Its records are `video_caption` only, never AV.
- **AVCaps** clips run 20-90 s (median 41 s). Crop or window them to the run's clip length.
- **AudioCaps coverage:** the mirror kept 45,178 of the 49,838 official train clips (91 %).
- Resample everything to 38.4 kHz mono (plan §2.4). Stream archive members; do not explode them onto the exFAT Lexar.

## 5. What this resume did

1. Found the earlier attempt's work:
   - branch `chimera-omni claude/run0-audio-20261028` @ `0897c09`, already pushed;
   - the deployed script `/home/cereal/omni-campaign/audio-gap/run0_audio_gap.py`, sha `6eff3d23…`, which matches `0897c09`;
   - `state/run.log`;
   - all 7 datasets `FETCHED` + `PROBE`d and all 4 products staged;
   - omni_lab run 40189, left `running`.

   There was no live process or unit and no partial `.part` file. The Lexar scratch dir is empty. It made no Strata change.
2. `run0_audio_gap.py status`: every dataset verified N/N, 0 in flight, 0 failed. Audio on disk is 150.3 GB against the 152 budget.
3. Wrote and ran an independent verifier, `scripts/run0_audio_gap_verify.py`. It is committed as `7c2fb16` on `claude/run0-audio-20261028`, pushed to origin, and deployed to `/home/cereal/omni-campaign/audio-gap/run0_audio_gap_verify.py` (sha `c0b94040…`). Under nice 19 / ionice idle it:
   - re-hashed every staged file against its manifest;
   - recounted records, trainable, hold-out, `dup_of`, splits and tasks;
   - checked every referenced archive (6,673 + 458 + 4 + 2) for presence, size and sha;
   - re-hashed 14 raw files plus 20 random AudioSet mp4s;
   - extracted and decoded 45 random trainable clips through their own container types: zip member, tar offset, parquet row, 7z member, plain file and tar.gz stream. Audio was decoded to PCM where the record claims audio, and video was decoded where it claims video.

   Result: 0 problems. The scratch dir was removed afterwards.
4. Checked the publisher checksums (HF `paths-info`, Zenodo API) from the workstation against the manifests.
5. Closed omni_lab run **40189** as `pass`. Its receipt is the verify report, and it has 9 metrics (datasets 7, products 4, records 125,544, trainable 108,107, held out 15,774, decode 45, re-hash 34, problems 0).

## 6. RAM and disk each product needs

- **Standing RAM: none.** Nothing is resident or running.
- **Reading a product** means streaming `records.jsonl` (18-56 MB per product) and pulling one member at a time from the packed archives.
  - Peak use is one clip buffer plus the decoder; the largest members are AVCaps clips of about 6-20 MB.
  - The Clotho 7z archives are solid, so pull their members in batches per archive rather than one at a time.
  - I did not measure a peak RSS. By design it is well under 1 GiB per reader process, and it does not need media-gate or Strata.
- **Disk:** 150.3 GB raw on the Lexar plus about 160 MB of staging files. No product needs `/`.

## 7. How to check it later

```
# on the AI server
cd /home/cereal/omni-campaign/audio-gap
python3 run0_audio_gap.py status
TS=$(date -u +%Y%m%dT%H%M%SZ); nice -n 19 ionice -c3 /home/cereal/model-surgery-lab/venv/bin/python -I \
  run0_audio_gap_verify.py /mnt/lexar/omni-campaign/corpus/external/_audio_gap_scratch/verify-$TS state/verify-$TS.json
```

The verifier takes about 7 min. It needs `pyarrow`, which the venv has and system python3 does not.

## 8. Not done / open

- **Tier 2 not started** (FSD50K 34.5 GB first, then more VGGSound shards and VATEX). The 152 GB budget is spent. The Lexar has 227 GB free, and run-0 checkpoints are expected to take about 100 GB of it. Starting tier 2 is a lead decision once checkpoint pruning shows the real headroom.
- **Wiring the products into run-0 shards** is area B's job. The hold-out lists also have to go into registry v3 first.
- omni_lab run 40189's `run_dir` was stored by the earlier attempt with an MSYS-mangled prefix (`C:/Program Files/Git/mnt/lexar/...`). The real path is `/mnt/lexar/omni-campaign/run0-intake-staging`. `log-run end` cannot rewrite `run_dir`; its `log_path` is now correct. `ended_at` is the close time (11:21Z), not the run's real finish (08:15:56Z).
- No Relay post. This lane touched no Strata or media-gate service.
- Lead decisions: the strict VALOR/VGGSound-test hold-out (§3), and how much `sound_gen` weight the 24 kHz AudioCaps gets.
