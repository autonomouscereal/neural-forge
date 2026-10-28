# Run 0 support scout: the audio gap (sound effects, paired audio-video, video understanding)

Date: 2026-10-27/28. Scout only. I read the HF Hub API, dataset cards, Zenodo and GitHub ID lists, and checked disk on the AI server. Nothing was downloaded to the server and no service was touched. The only local files were metadata (ID lists, CSVs) in my scratchpad, used to compute the overlaps below.

Ruling: `cyril-run0-decisions-2026-10-22`, "fix the audio gap". Downloads for real sound effects and paired audio-video are approved (RUN0_BUILD_PLAN Addendum).

## 0. Answer first

- **Disk today:**
  - `/` has 114 GB free (94 % used). That is already **below** the 150 GB the rules keep free for run-0 checkpoints, so **put nothing on `/`**.
  - Lexar (`/mnt/lexar`, exFAT) has 381 GB free. Run-0 checkpoints take about 100 GB of it.
  - NAS has 1 MB free. Write nothing there.
  - So the realistic cap is **about 150 GB on the Lexar, not 200 GB**. The tier-1 set below is 150.3 GB.
- **Tier-1 set: 7 sources, about 150 GB, all on the Lexar. It needs no YouTube scraping.**
  - Real sound effects with **human** captions: AudioCaps and Clotho.
  - Real paired audio-video with **human** captions: AVCaps, VALOR-32K, and the AudioCaps clips whose AudioSet video is on an HF mirror.
  - Real paired audio-video with labels: VGGSound (2 train shards).
  - Video understanding text, human-written: MSR-VTT, plus the AVCaps visual and audio-visual captions and the VALOR-32K captions.
- **Excluded on our data rules, not on size:**
  - Model-written caption sets: AudioSetCaps, Sound-VECaps, Auto-ACD and ASID-1M. AudioSetCaps was made with small open models; the others are LLM or MLLM output.
  - 16 kHz-only mirrors. The audio target is 38.4 kHz (§2.4), so 16 kHz audio is bandlimited and wrong as a generation target.
- **Licences:**
  - Most of the human-captioned sets are **non-commercial or research-only**. These are AudioCaps (NC on the HF mirror), Clotho captions (Tampere NC), AVCaps (CC BY-NC 4.0), MSR-VTT (research) and VALOR (terms not stated).
  - Record each one on the run, the same way MAESTRO is recorded.
  - VGGSound and FSD50K are CC BY 4.0. FSD50K also carries per-clip Freesound licences.

## 1. Server facts checked (2026-10-27 23:54 server time)

| Mount | Size | Free | Use | Note |
|---|---|---|---|---|
| `/` ext4 | 1.8 T | **114 G** | 94 % | Below the 150 GB checkpoint reserve already. Do not download here. |
| `/mnt/lexar` exFAT | 3.8 T | **381 G** | 91 % | No symlinks. Run-0 checkpoints need about 100 GB. Audio cap: 150 GB. |
| `/mnt/ai_data` NAS | 4.5 T | 1.0 M | 100 % | Write nothing. |

- **RAM:** 251 GB total, 27 GB available, swap 6/7 GB used. Downloads must stay small in RAM and must not flood the page cache next to strata-llama (see §5).
- **No audio datasets on disk yet.**
  - There is no VGGSound, AudioCaps, Clotho or WavCaps anywhere under `/` or `/mnt/lexar` (searched to depth 4).
  - `omnicoder_0.1.0/data/video` is 15 MB in total: annotations for MSR-VTT, VATEX, Charades, ActivityNet and others, with **no videos**.
  - `webvid` is 1 MB.
- **Tooling:**
  - `huggingface_hub` 1.21.0 is in `/home/cereal/model-surgery-lab/venv`.
  - `hf_transfer` is not installed.
  - `ffmpeg` and `ffprobe` are in `/usr/bin`.
  - `aria2c` and `yt-dlp` are not on the PATH. They are not needed.
- **Path correction for area B:**
  - The plan (§2.5) names the intake as `/mnt/lexar/corpus-store/chimera-intake`. The real path is **`/mnt/lexar/omni-campaign/corpus-store/chimera-intake/`**, with sub-dirs `audio`, `video`, `mixed`, and so on.
  - `run0-intake-staging` does not exist yet. Per the staging contract, use `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/`.

## 2. Candidates. Sizes are measured from the HF tree API or Zenodo listings.

"Human" means the captions were written by people. "LLM" means a model wrote them.

### 2a. Tier 1 (recommended)

| # | Dataset | Exact source | Size | Clips | Text | Video | Audio | Licence |
|---|---|---|---|---|---|---|---|---|
| 1 | **Clotho v2.1** | Zenodo record 4783391 (official): 3 × `.7z` + 3 caption CSVs + 3 metadata CSVs + LICENSE | **7.1 GB** | 6,974 clips, 15-30 s; dev 3,840 (+ val and eval) | 5 human captions per clip (34,870 in all), 8-20 words, high quality | no | Freesound originals, 44.1 kHz WAV inside the 7z | Audio: per-file Freesound CC (in the metadata CSV). Captions: Tampere University licence, **non-commercial with attribution** |
| 2 | **AudioCaps** (with audio) | HF `OpenSound/AudioCaps`, 473 parquet files, audio embedded (not gated) | **43.9 GB** | train 45,178 / val 2,223 / test 4,411 rows. The official v1 lists have 49,838 train, so about 91 % survived | Human AMT captions, one sentence each. The dataset's val and test have 5 captions per clip | no | About 0.85 MB per 10 s, which suggests 44.1 kHz 16-bit mono **(inference; check one row)** | HF tag CC BY-NC 4.0. Captions are from cdjkim/audiocaps. Audio is YouTube content |
| 3 | **AVCaps** | HF `TUT-ARG/AVCaps`: `train/val/test_videos.zip` + `*_captions.json` | **10.6 GB** | 1,661 / 200 / 200 = 2,061 clips, about 50 s average (28.8 h) | Per clip: up to 5 **audio**, 5 **visual** (written without sound) and 5 **audio-visual** captions, crowdsourced. Also 3 GPT-4 summaries, which we **drop** (LLM) | **yes**, VidOR mp4 with sound | in the mp4 | **CC BY-NC 4.0**, academic/non-commercial |
| 4 | **VALOR-32K** | HF mirror `UnFaZeD07/VALOR-32K`: `VALOR-32K-videos.tar` (41.05 GB) + annotations zip (1.5 MB). The official copy is BaiduDisk only | **41.0 GB** | 32k clips; 25k / 3.5k / 3.5k per the paper. 10 s AudioSet clips | **Human-annotated audio-visual captions**, as the paper states (arXiv 2304.08345) | **yes** | in the mp4, **to verify with ffprobe** | Mirror tagged MIT. The official repo's MIT covers the code; the data terms are not stated. Treat as **research-only** and record it |
| 5 | **AudioCaps ∩ AudioSet video** (selective) | HF `chikuwa-AI/AudioSet_balanced_video` + `chikuwa-AI/AudioSet_unbalanced_video`, one mp4 per YouTube ID, so selective `--include` works | **12.0 GB** | **6,689 clips**: train 6,625 (531 from the balanced repo) + test 64. Computed by joining the official AudioCaps CSV IDs with the mirror file lists (174,402 IDs) | AudioCaps human captions joined by ID → **real paired AV with human sound captions** | **yes**, 10 s | in the mp4, **to verify** | AudioSet labels CC BY 4.0, videos owned by their uploaders. The mirror states no licence. Captions as in row 2 |
| 6 | **VGGSound**, 2 of 20 shards | HF `Loie/VGGSound`: `vggsound_00.tar.gz`, `vggsound_01.tar.gz` (16.4 + 16.9 GB) + `vggsound.csv` | **33.4 GB** | about 20k of 199,467 clips (183,971 train / 15,496 test), 10 s, 310 classes | Class label only ("waterfall burbling"). Use it as label-conditioned AV and AV-correspondence, **not** as caption text | **yes**, mp4 with sound (the card says audio + video) | in the mp4 | **CC BY 4.0** for annotations. Videos owned by their uploaders |
| 7 | **MSR-VTT** | HF `friedrichor/MSR-VTT`: `MSRVTT_Videos.zip` (2.19 GB) + train_9k / train_7k / test_1k JSON | **2.2 GB** | 10,000 clips, 10-30 s | 20 human captions per clip (200k) → **video understanding text** | **yes** (most clips carry sound) | in the mp4 | Microsoft research-only. Record it |

**Tier-1 total: 7.1 + 43.9 + 10.6 + 41.0 + 12.0 + 33.4 + 2.2 = 150.2 GB**, all on the Lexar.

### 2b. Tier 2 (only if Lexar space frees up, e.g. after run-0 checkpoint pruning)

| Dataset | Source | Size | Why tier 2 |
|---|---|---|---|
| **FSD50K** | HF `Fhrozen/FSD50k` (per-file WAV; dev 25.6 GB + eval 8.8 GB) or Zenodo official | 34.5 GB | 51,197 clips, 44.1 kHz, human-verified AudioSet-ontology labels, plus human uploader titles, descriptions and tags in the metadata JSON. CC BY 4.0 with per-clip CC0/BY/BY-NC. The best pure-SFX set, but it has labels rather than captions. **First pick when space appears.** |
| More VGGSound shards | `Loie/VGGSound` `_02`..`_19` | 16.3-17.3 GB each | About 10k label-conditioned AV clips per shard. Holds the other 7,051 AudioCaps-matching IDs, but you cannot pick them out of a tar.gz. |
| **VATEX** (English) | HF `Blazewild/Vatex_reencoded` (20.5 GB mp4 + `VATEX_CAPTIONS.json`) | 20.5 GB | Human captions, 10 English per clip. **Unknown whether the re-encode kept audio.** Check one file first. |
| **Charades** | HF `linxxx3/Charades_v1_480` (16.3 GB zip) or `jinyoungkim/Charades` (59 GB mp4) | 16.3 GB | 9.8k indoor videos with human scripts and descriptions, and sound. AI2 **non-commercial** licence. |
| FineVideo subset | HF `HuggingFaceFV/finevideo` (**gated, auto-accept terms; needs Cyril to accept with his HF account**), 673 GB in about 0.5 GB parquet shards | e.g. 20-30 GB | 43-44k YouTube-Commons CC-BY videos **with audio**, about 3.4k h. Rich scene, mood and QA text, but **written by Gemini 1.5 Pro and reformatted by GPT-4o**. That is frontier-model text: not Claude/Codex and not a small model, so it is not barred by our rules. It still needs a lead ruling before it is used as a CE target. |
| WavCaps BBC / SoundBible / AudioSet_SL | HF `cvssp/WavCaps` `Zip_files/<src>` (BBC 130 GB, SoundBible 0.58 GB, AudioSet_SL 36 GB, FreeSound 653 GB) | - | Captions are **ChatGPT-rewritten** from human metadata. **Academic use only**; BBC carries its own RemArc NC terms. Usable as audio-only or as generation conditioning if the lead rules on it. Too large for this window. |

### 2c. Rejected, with the reason

| Dataset | Reason |
|---|---|
| AudioSetCaps (`baijs/AudioSetCaps`, 18.8 GB of CSV and `.pt` files) | Captions only; no audio. The captions come from large audio-language models; the paper names Qwen-Audio-Chat and a 7B LLM, and the card does not. That is **small-model generation**, which our rules forbid. Using it would also need 1.9M YouTube scrapes. Academic-only. |
| Sound-VECaps (`yyua8222/Sound-VECaps`, 401 to the API) and Auto-ACD | LLM-generated captions over AudioSet/VGGSound IDs. Same exclusion, and both need scraping. |
| ASID-1M (`AudioVisual-Caption/ASID-1M`, 514 GB) | Videos are included, but the 8-attribute captions come from an MLLM pipeline that does not name its models. Too big, and unknown generator provenance. |
| `Gray1y/VGGSound-50k` (24 GB) | Precomputed features only; no media. |
| `jp1924/AudioCaps` (808 GB, gated) | Far too large for AudioCaps. |
| `gijs/audiocaps` (17.5 GB), `txya900619/*-16k`, `quinnlue/FSD50K-16k`, `txya900619/wavcaps-10s-16k` (52 GB) | Smaller because they are resampled to 16 kHz or compressed. That is below the 38.4 kHz target, so they are not usable as generation targets. Fall back to them only if a full-band source fails. |
| `11hu83/vggsound` (62 GB) | It is VGGSound **test** only (15,445 rows), and it stores the same clips three times (parquet + mp4 + wav). VGGSound test is the standard video-to-audio benchmark: keep it **held out**, not trained on. |
| `agkphysics/AudioSet` (2.44 TB), `Muno459/AudioSet` (2.49 TB), `mitermix/audioset-with-grounded-captions` (220 GB, LLM captions), `CLAPv2/audioset_strong` (110 GB) | Size, and labels or LLM captions only. |
| `CLAPv2/Clotho` (12.5 GB parquet) | The same Clotho at 1.8× the size of the official Zenodo 7z. Use Zenodo. |

## 3. Recommended download order (all on the Lexar)

The order puts the fastest gap-closers first and the large single-file tars last.

1. **Clotho v2.1**, 7.1 GB, from Zenodo. Human-captioned SFX within the first hour.
2. **AudioCaps**, 43.9 GB, `OpenSound/AudioCaps`. The bulk of human-captioned SFX.
3. **AVCaps**, 10.6 GB. The first real joint AV, with audio, visual and AV captions, which are also video understanding text.
4. **MSR-VTT**, 2.2 GB. Video understanding text.
5. **AudioCaps ∩ AudioSet video**, 12.0 GB.
   - Selective `hf download --include` of the 6,689 mp4 paths. The path list can be rebuilt from the official AudioCaps CSVs and the two chikuwa file trees, using the scout's join.
   - These are human-captioned paired AV clips.
6. **VALOR-32K**, 41.0 GB.
   - It is one 41 GB tar, so it needs 41 GB contiguous free. Read it as a tar stream; do not extract 32k files onto exFAT.
7. **VGGSound shards 00 and 01**, 33.4 GB.
   - Drop the test-split IDs (per `vggsound.csv`) at staging, and add them to the hold-out registry.

Then, as space frees: FSD50K, then VATEX (after the audio check), then more VGGSound shards.

**Raw-source placement:** `/mnt/lexar/omni-campaign/corpus/external/<dataset>/<version>/`. This sits beside the existing `librispeech` and `modelnet10`.

**Derived products** go to `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/`, each with a manifest JSON and a README per the staging contract. Suggested products:
- `sfx_caption_v1`: Clotho + AudioCaps; audio ↔ caption, both directions.
- `av_caption_v1`: AVCaps + VALOR-32K + AudioCaps∩AudioSet video.
- `av_label_v1`: VGGSound.
- `video_caption_v1`: MSR-VTT + AVCaps visual captions.

## 4. Hold-outs and dedupe (enter these into hold-out registry v3 before any training use)

- **AudioCaps:** the test split (975 clips by official ID; 4,411 rows on the mirror). Keep val as dev.
- **Clotho:** the evaluation split.
- **VGGSound:** all 15,496 test IDs.
- **MSR-VTT:** test_1k.
- **VALOR-32K:** the test split.
- **AVCaps:** the test split (200).
- **Cross-dataset ID overlap**, all keyed by YouTube ID; dedupe on it:
  - AudioCaps train ∩ VGGSound = 7,051 IDs.
  - AudioSet-video mirror ∩ VGGSound = 4,749 IDs.
  - VALOR-32K is AudioSet-derived, so it will also overlap.

## 5. Staging and safety notes for whoever runs the downloads

- **exFAT:**
  - The HF cache uses symlinks, which exFAT does not support. Always use `hf download <repo> --repo-type dataset --local-dir <dir>` (real files), with `HF_HOME` on the Lexar.
  - Keep parquet, zip and tar archives packed, and stream-read them. Do not explode hundreds of thousands of small files onto exFAT.
- **RAM and page cache next to strata-llama** (27 GB available, swap almost full):
  - Run downloads with `nice -n 19 ionice -c3`, `--max-workers 2-4`, and no `hf_transfer`.
  - Watch `MemAvailable`, and pause if it drops below about 16 GB.
  - Large sequential writes fill the page cache. If the Strata experts are mmap'd rather than mlocked, a flood can evict them. This is not verified here: check how strata-llama holds its expert pages before a 150 GB pull, or drop the written files from the cache as you go (`dd ... oflag=direct`, or a `posix_fadvise DONTNEED` pass per finished file).
- **Verify artifacts, not exit codes.** For each source:
  - sha256 every archive against the HF LFS sha or the Zenodo md5;
  - `ffprobe` a random 20 clips: an audio stream must be present, plus the sample rate, channels and duration. This is mandatory for VALOR-32K, chikuwa and VATEX, where audio presence is unverified;
  - check counts against the table above.
- **Caption provenance tags for the B-1 guard:**
  - human captions: `public`;
  - AVCaps GPT-4 captions and any WavCaps/FineVideo LLM text: dropped, or a separate tag pending a lead ruling.
  - Nothing here comes from Claude, Codex, a small model or the capture spool.
- **Resampling:** all tier-1 audio is 44.1 or 48 kHz at source, or YouTube AAC. Polyphase resample it to 38.4 kHz mono, per §2.4.
- **Licence field per manifest source:**
  - Clotho: `freesound-per-file-cc + tampere-nc (captions)`;
  - AudioCaps: `cc-by-nc-4.0 (hf mirror)`;
  - AVCaps: `cc-by-nc-4.0`;
  - VALOR-32K: `research (unstated; mirror says MIT)`;
  - AudioSet video: `cc-by-4.0 labels; youtube content`;
  - VGGSound: `cc-by-4.0`;
  - MSR-VTT: `microsoft-research`.

## 6. Open items

- No HF token is needed for tier 1; every tier-1 repo is ungated. FineVideo (tier 2) is gated: Cyril has to accept its terms himself.
- One thing is not confirmed: that the chikuwa and VALOR mp4 files carry audio tracks. It is resolved by the ffprobe step on the first files downloaded, before the bulk pull.
- The AudioCaps mirror's sample rate is inferred from bytes per clip. Confirm it on one row.
