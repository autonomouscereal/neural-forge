# Run 0 data manifest (area B)

Status: 2026-10-28. Data for run 0 (co2-2b planted from Qwen3.5-2B, distilled on every modality on 4 P40s), built to `RUN0_BUILD_PLAN.md` revision 2 (§2.5, §3.3, §4 B-0..B-6). Two sealed manifest versions exist. **v2 is the one to train on.** v1 has two defects, listed below, that v2 fixes.

| Version | File (server) | sha256 | omni_lab | Code |
|---|---|---|---|---|
| **v2** (train) | `/home/cereal/omni-campaign/run0/data/run0_data_v2.json` | `04a0114b1c4de6a28a0e1111f68e3697e6ae0fab76f9d911e92d8f27ecd4a06f` | build run 40383; dataset `run0/data` (id 559), dataset_version v2 (id 572), run_dataset output | `9f5d5d5` |
| v1 (superseded) | `/home/cereal/omni-campaign/run0/data/run0_data_v1.json` | `0fed531b86e33c6797b89926b857e8f52c215716123d7e26b692698ca62851d1` | build run 40310; dataset_version v1 (id 568) | `dadee8b` |

- Code: chimera-omni branch `claude/run0-b-20261022`, pushed to origin. Every build ran from a commit-pinned copy, `/home/cereal/omni-campaign/run0/code-b-<commit>/` with `DEPLOYED.json` (commit plus tgz sha256, uploaded hash matched).
- Tests: `tests/test_run0_data_torch_optional.py`, run on the server Pascal venv (torch, scipy, ffmpeg), CPU only. At HEAD `ca66303`, 48 collected: 47 passed, 1 skipped, exit 0. The skipped one is the grammar test through the real layout API, which needs area A's A-2 (not on this branch). At `9f5d5d5` (the v2 build), 45 collected: 44 passed, 1 skipped.
- Every build was CPU only: `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `ionice idle`, `oom_score_adj 800` (the OOM killer takes the build before strata-llama), and a transient user unit capped by `MemoryMax`. No GPU was touched, and nothing was written to the NAS.
- Where the data lives: the 10-22 parts are in `/home/cereal/omni-campaign/run0/data/parts` (root, 16 GB). Everything built on 10-28 is in `/mnt/lexar/omni-campaign/run0/data/parts` (31 GB), so root stays at 113 GB free. Manifests and logs are in `/home/cereal/omni-campaign/run0/data/{run0_data_vN.json, logs/}`.

## What each manifest holds

Shares are position shares per optimizer step (plan §2.5). A share marked `(+x)` received bucket x's share because x has no live source. A source with a `fraction` is drawn with that probability inside its own bucket (lead call, below); the other sources in the bucket are drawn by size. "10-day supply" is the number of days before the bucket reaches its epoch cap at 33M or 46M positions per day. Text buckets count KD-only repeats up to 12 epochs per 7 days. "10+ d" means the bucket does not starve inside the 10-day run.

Unit counts are the plan's raw rows: each row is 3,072 values, which is 80 ms of 38.4 kHz audio, one 32x32 image patch, one 2-frame video tubelet at 12.5 fps (32x16 per frame), or one 8^3 block of a 64^3 volume. Positions are estimates: media rows, plus headers, plus text at about 3.5 characters per token. Corpus sources count their real token lengths exactly.

### v2 (train on this): buckets

| Bucket | Share | Kind | Cap (epochs/7d) | Sources | Records | Positions (est.) | Media rows | 10-day supply at 33M / 46M per day |
|---|---|---|---|---|---|---|---|---|
| text_kd | 0.26 (+long) | text_kd | 6 | 15 | 62.2k | 151.02M | - | 10+ d / 10+ d |
| k3 | 0.08 | text_kd | 6 | 4 | - | 1.99M | - | 3.3 d / 2.4 d (allowlisted positions; 9.1 / 6.5 before) |
| tool | 0.08 | text_kd | 6 | 4 | 3.3k | 10.50M | - | 10+ d / 10+ d |
| long | 0.00 | long | 8 | 0 | - | 0 | - | - / - |
| decision_twins | 0.04 | twin | 4 | 1 | 256 | 74.7k | image 65.0k | 10+ d / 10+ d |
| image_understand | 0.12 | understand | 4 | 1 | 118.3k | 38.14M | image 31.68M | 10+ d / 10+ d |
| image_gen | 0.12 | generate | 4 | 6 | 121.6k | 41.46M | image 35.06M | 10+ d / 10+ d |
| video_understand | 0.02 | understand | 4 | 3 | 39.7k | 82.22M | video 76.02M, audio 3.89M | 10+ d / 10+ d |
| video_gen | 0.04 | generate | 4 | 4 | 20.7k | 31.54M | video 29.74M, audio 480.1k | 10+ d / 10+ d |
| speech | 0.12 | mixed | 4 | 4 | 105.8k | 16.31M | audio 9.71M | 10+ d / 10+ d |
| music | 0.06 | mixed | 4 | 4 | 49.7k | 8.51M | audio 6.15M | 10+ d / 10+ d |
| 3d | 0.03 | mixed | 4 | 2 | 2.3k | 1.30M | image 28.4k, 3d 1.19M | 5.3 d / 3.8 d |
| concept | 0.03 | mixed | 4 | 2 | 168 | 68.6k | image 44.7k, audio 8.0k | 0.3 d / 0.2 d |
| **total** | | | | 51 | 523.9k | 383.15M | | |

The text floor holds for all 10 days at both rates. The text and tool share is 0.42, and served/floor is 1.05 every day. This is the corrected projection (`logs/supply_projection_v2_trainable.json`). It counts only the positions the author allowlist leaves trainable, and it pools the text buckets the way `v2_trainer.Run0Mixture` does, so a capped K3 is made up by text KD and the documents. The seal-time projection (`logs/supply_projection_run0_data_v2.json`) shows the same verdict but overstates K3 at 9.1 / 6.5 days. Code `ca66303` writes `positions_trainable_est` from v3 on.

Buckets that stop before day 10:
- **K3:** day 3.3 at 33M, day 2.4 at 46M. It has only 0.73M trainable positions: 37 % of its row memberships pass the allowlist. After that its share goes to text KD.
- **3d:** day 5.3 / 3.8.
- **concept:** day 0.3 / 0.2. It has 168 D4 records, and the TTS job keeps adding them.

### v2: per-task and per-modality totals

| Task | Records |
|---|---|
| image_gen | 120,833 |
| image_caption | 118,287 |
| document | 62,096 |
| asr | 55,182 |
| tts | 50,631 |
| sound_caption | 23,353 |
| av_gen | 19,976 |
| sound_gen | 19,911 |
| av_caption | 11,937 |
| video_caption | 8,309 |
| av_label | 7,949 |
| sound_label | 7,949 |
| music_gen | 4,934 |
| music_caption | 2,954 |
| tool_chat | 2,130 |
| music_label | 1,980 |
| text_chat | 1,124 |
| volume_text_gen | 1,111 |
| volume_label | 1,000 |
| image_edit | 790 |
| video_gen | 678 |
| intent_twin | 256 |
| concept_interleaved | 168 |
| volume_gen | 111 |
| volume_caption | 111 |
| temporal_order | 86 |
| math_chat | 70 |

| Modality | Raw 3,072-value rows (80 ms audio / 32x32 patch / 2-frame video tubelet / 8^3 volume) |
|---|---|
| video | 105,761,088 |
| image | 66,877,312 |
| audio | 20,235,433 |
| 3d | 1,194,496 |

### v2: every source

The text corpora come with their teacher caches: Q38-27BU HF bf16 top-64 + lse, the RVN IQ3_S Strata cache (not the live Q4_K_XL), and K3 provisional bf16 xtok, per ruling L1. `verify-kd` checked them on the real caches: the KD positions are finite, and top-10 mass is 0.99-1.00 (`logs/verify_kd_run0_data_v2.json`). Every record source was re-hashed against its `part.json` and re-checked record by record. 16 records per part were then decoded through the training loader and cut by the v2 rule (`final_steps.sh` step 4). Every media entry gave exactly the rows its record declares, at 3,072 values per row, all finite. Packet RMS after the gain was 0.43-0.53.

| Source | Bucket | Provenance | Records | Tasks | Positions (est.) | Media rows | Gain | sha256 (source.json or corpus) |
|---|---|---|---|---|---|---|---|---|
| `text_clean_v1#text-clean-00000` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `01d99603bd662630` |
| `text_clean_v1#text-clean-00001` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `1b5ecfc13a944ae6` |
| `text_clean_v1#text-clean-00002` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `36c04a2147fe139c` |
| `text_clean_v1#text-clean-00003` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `ee3d695861863e9c` |
| `text_clean_v1#text-clean-00004` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `8064467308227fcf` |
| `text_clean_v1#text-clean-00005` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `be62d718eec641ed` |
| `text_clean_v1#text-clean-00006` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `c52ae817de504174` |
| `text_clean_v1#text-clean-00007` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `f6d77e91f081d5ba` |
| `text_clean_v1#text-clean-00008` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `79abb869371198ac` |
| `text_clean_v1#text-clean-00009` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `d960ab6a1572eaeb` |
| `text_clean_v1#text-clean-00010` | text_kd | registry | - | all_tokens; teachers q38+strata | 1.05M | - | - | `18bee8fbc56f7de7` |
| `text_clean_v1#text-clean-00011` | text_kd | registry | - | all_tokens; teachers q38+strata | 814.8k | - | - | `9fbbb536ed0feb65` |
| `pdf-corpus-docs@20261028T122343Z` | text_kd | public | 51.5k | document 51.5k | 130.39M | - | - | `133b5d100ace5aad` |
| `registry-text-chat@20261028T122343Z` | text_kd | public | 70 | math_chat 70 | 24.4k | - | - | `275f4495b007691a` |
| `registry-text-docs@20261028T122343Z` | text_kd | registry | 10.6k | document 10.6k | 8.26M | - | - | `1eae2f1e65eaa361` |
| `k3#pass-0` | k3 | k3 | - | all_tokens; teachers q38+strata+k3 xtok | 498.3k | - | - | `2b35dd0935fd28de` |
| `k3#pass-1` | k3 | k3 | - | all_tokens; teachers q38+strata+k3 xtok | 498.0k | - | - | `1090d81cc87e0ed6` |
| `k3#pass-2` | k3 | k3 | - | all_tokens; teachers q38+strata+k3 xtok | 497.5k | - | - | `ad01fc7e0032f28e` |
| `k3#pass-3` | k3 | k3 | - | all_tokens; teachers q38+strata+k3 xtok | 497.7k | - | - | `756e386f65e23d28` |
| `toolcall_flash_v1` | tool | strata-gen | - | assistant_only; teachers strata | 3.00M | - | - | `1d3ca810c9bad0f4` |
| `bfcl_native_v1` | tool | public | - | assistant_only; teachers q38+strata | 415.5k | - | - | `b0e6aa63dba6c882` |
| `intake-strata-text` | tool | strata-gen | 479 | text_chat 148, tool_chat 331 | 975.9k | - | - | `50e7c94199bffe7d` |
| `intake-strata-text@20261028T081312Z` | tool | strata-gen | 2.8k | text_chat 976, tool_chat 1.8k | 6.11M | - | - | `786a519f3f664046` |
| `long_v1` (disabled) | long | registry | - | all_tokens; teachers q38 | 1.00M | - | - | `918f36f8e6042d5d` |
| `strata-intent-bank@20261028T122343Z` | decision_twins | strata-gen | 256 | intent_twin 256 | 74.7k | image 65.0k | image 0.84 | `d5332d144c27ab0d` |
| `coco2017-caption` | image_understand | public | 118.3k | image_caption 118.3k | 38.14M | image 31.68M | image 0.89 | `6aa49f8149d0ef9e` |
| `coco2017-t2i` | image_gen | public | 118.3k | image_gen 118.3k | 40.08M | image 34.00M | image 0.89 | `250cc96b2418ad0d` |
| `intake-image-edit` | image_gen | strata-gen | 32 | image_edit 32 | 18.1k | image 16.4k | image 0.79 | `603061a2df282798` |
| `intake-image-t2i` | image_gen | strata-gen | 247 | image_gen 247 | 88.8k | image 63.2k | image 0.79 | `d88f8ed3a64a588b` |
| `coco-procedural-edits@20261028T122343Z` | image_gen | procedural | 128 | image_edit 128 | 75.2k | image 69.6k | image 0.81 | `20e3836f2d64aad1` |
| `intake-image-edit@20261028T081312Z` | image_gen | strata-gen | 630 | image_edit 630 | 357.4k | image 322.6k | image 0.73 | `ae784e73bda58bcb` |
| `intake-image-t2i@20261028T081312Z` | image_gen | strata-gen | 2.3k | image_gen 2.3k | 840.5k | image 588.5k | image 0.72 | `40d04d7fade2d4f2` |
| `ext-av-understand@20261028T123014Z` | video_understand | public | 31.3k | av_caption 11.9k, sound_caption 3.4k, av_label 7.9k, sound_label 7.9k | 59.14M | video 53.45M, audio 3.89M | audio 3.65, video 0.86 | `c1e7ce6a83d3db00` |
| `ext-video-caption@20261028T123014Z` | video_understand | public | 8.2k | video_caption 8.2k | 22.44M | video 21.95M | video 0.85 | `ce9d67a98d4d0c38` |
| `intake-video-understand@20261028T081312Z` | video_understand | strata-gen | 229 | video_caption 143, temporal_order 86 | 636.5k | video 615.6k | video 0.75 | `d3bcdf7f838e79af` |
| `h3set-video` | video_gen | media-teacher | 506 | video_gen 506 | 855.5k | video 728.6k | video 0.59 | `f128c8389c852852` |
| `intake-video` | video_gen | strata-gen | 33 | video_gen 19, av_gen 14 | 51.4k | video 47.5k, audio 336 | video 0.77, audio 3.73 | `9f882a1f35ee8de7` |
| `ext-av-gen@20261028T123014Z` | video_gen | public | 19.9k | av_gen 19.9k | 30.28M | video 28.64M, audio 477.9k | audio 3.65, video 0.86 | `d30827200da88fce` |
| `intake-video@20261028T081312Z` | video_gen | strata-gen | 229 | av_gen 76, video_gen 153 | 356.2k | video 329.8k, audio 1.8k | video 0.76, audio 5.21 | `7b75901358fa99cc` |
| `cyril-speech` | speech | cyril | 382 | asr 191, tts 191 | 41.0k | audio 21.9k | audio 6.58 | `a65269e83a0dee79` |
| `librispeech-asr` | speech | public | 4.6k | asr 4.6k | 591.1k | audio 325.4k | audio 6.81 | `488e68ed554ee632` |
| `ljspeech` | speech | public | 26.2k | asr 13.1k, tts 13.1k | 3.73M | audio 2.17M | audio 5.53 | `800b115795799139` |
| `librispeech-train` | speech | public | 74.7k | asr 37.3k, tts 37.3k | 11.95M | audio 7.20M | audio 6.11 | `4f5b16bad8d25adf` |
| `intake-music-ace` | music | strata-gen | 568 | music_gen 284, music_caption 284 | 98.7k | audio 71.0k | audio 2.63 | `f81f3ffb8785407a` |
| `maestro-piano` | music | public | 4.0k | music_gen 2.0k, music_label 2.0k | 687.5k | audio 495.0k | audio 5.68 | `c059ca2b0df89fa3` |
| `ext-sound@20261028T123014Z` | music | public | 39.8k | sound_caption 19.9k, sound_gen 19.9k | 6.80M | audio 4.91M | audio 3.86 | `d61a043a5dfe89f7` |
| `intake-music-ace@20261028T081312Z` | music | strata-gen | 5.3k | music_gen 2.7k, music_caption 2.7k | 926.9k | audio 667.5k | audio 2.45 | `1b8c59fc2eff8447` |
| `hy3d-voxel64` | 3d | media-teacher | 333 | volume_gen 111, volume_text_gen 111, volume_caption 111 | 216.4k | image 28.4k, 3d 170.5k | 3d 1.00, image 0.56 | `fbf016dcd31af17b` |
| `modelnet10-64` | 3d | public | 2.0k | volume_label 1.0k, volume_text_gen 1.0k | 1.09M | 3d 1.02M | 3d 1.00 | `4ac68b237dc43678` |
| `coco-tts-concept@20261028T122343Z` | concept | media-teacher | 120 | concept_interleaved 120 | 49.0k | image 31.9k, audio 5.8k | image 0.84, audio 5.86 | `dde6c5d18203e36f` |
| `coco-tts-concept@20261028T123014Z` | concept | media-teacher | 48 | concept_interleaved 48 | 19.6k | image 12.8k, audio 2.2k | image 0.76, audio 5.80 | `8873808645259a73` |

Licences recorded on the run:
- COCO 2017: CC BY 4.0 annotations, Flickr images.
- LibriSpeech: CC BY 4.0.
- LJSpeech: public domain.
- MAESTRO: CC BY-NC-SA 4.0.
- ModelNet10: academic use.
- Hunyuan3D output: Tencent community licence.
- AudioCaps: CC BY-NC 4.0 captions; the audio is YouTube content.
- Clotho: per-file Freesound CC audio, Tampere NC-attribution captions.
- VGGSound: CC BY 4.0 annotations.
- AVCaps: CC BY-NC 4.0.
- VALOR-32K: research use.
- AudioSet: CC BY 4.0 labels.
- MSR-VTT: research use.
- PDF corpus: mixed copyright, internal training.

Cyril's 2026-10-14 ruling covers all of these ("[Cyril quote withheld]"). The full text is in each `source.json`.

### v1 (sealed 11:24Z, superseded)

| Bucket | Share | Kind | Cap (epochs/7d) | Sources | Records | Positions (est.) | Media rows | 10-day supply at 33M / 46M per day |
|---|---|---|---|---|---|---|---|---|
| text_kd | 0.26 (+long) | text_kd | 6 | 12 | - | 12.35M | - | 10+ d / 10+ d |
| k3 | 0.08 | text_kd | 6 | 4 | - | 1.99M | - | 9.1 d / 6.5 d |
| tool | 0.08 | text_kd | 6 | 4 | 3.3k | 10.50M | - | 10+ d / 10+ d |
| long | 0.00 | long | 8 | 0 | - | 0 | - | - / - |
| decision_twins | 0.00 | twin | 4 | 0 | - | 0 | - | - / - |
| image_understand | 0.19 (+decision_twins, concept) | understand | 4 | 1 | 118.3k | 38.14M | image 31.68M | 10+ d / 10+ d |
| image_gen | 0.12 | generate | 4 | 5 | 121.5k | 41.38M | image 34.99M | 10+ d / 10+ d |
| video_understand | 0.02 | understand | 4 | 1 | 229 | 636.5k | video 615.6k | 3.9 d / 2.8 d |
| video_gen | 0.04 | generate | 4 | 3 | 768 | 1.26M | video 1.11M, audio 2.2k | 3.8 d / 2.8 d |
| speech | 0.12 | mixed | 4 | 4 | 105.8k | 16.31M | audio 9.71M | 10+ d / 10+ d |
| music | 0.06 | mixed | 4 | 3 | 9.9k | 1.71M | audio 1.23M | 3.5 d / 2.5 d |
| 3d | 0.03 | mixed | 4 | 2 | 2.3k | 1.30M | image 28.4k, 3d 1.19M | 5.3 d / 3.8 d |
| concept | 0.00 | mixed | 4 | 0 | - | 0 | - | - / - |
| **total** | | | | 40 | 362.0k | 125.59M | | |

v1 has two defects. Both are fixed in v2, so do not launch on v1:
1. **No decision twins.** The 4 % decision-twin share was moved to `image_understand`. The cause: the twin bucket lists no sources of its own, because every generation source feeds it, so the manifest builder counted it as empty. Fixed in `8c3d999`: a twin bucket is live whenever a generation source exists.
2. **15 BFCL rows hold a held-out question.** They were in the tool bucket (details under Provenance below). Fixed in `9f5d5d5` with `deny_rows`.

v1 also starves video understanding, video generation and music by day 3-4, which the staging products fix in v2.

## Provenance and exclusion enforcement (B-0, B-1)

- **Paths.** A manifest path that matches the denylist is refused by name at validation, at load, and in every builder's `SourceWriter`. The denylist: `selfgen`, `tool-selfgen`, `toolcall/rescore`, `toolcall-q38`, `qtext_vl`, `vision_in`, `strata-capture`, `capture-export`, `d2_refill`, `refill-codex`, `sage`.
- **Rows.** A record or document whose source tag matches the row denylist is dropped. The row denylist: claude, codex, anthropic, openai, gpt, gemini, fable, opus, sonnet, grok, memory_batch_, trace_rewrite, memory_harness, selfgen, capture, refill, sage, qtext_vl, vision_in, toolcall-q38, q38-generated.
- **Authorship, by field.**
  - `text_clean_v1` and K3 rows are read through `author_allow_v1.json` (sha256 `94ad19b3…`). It holds the writer of every canonical row, taken from its author field, never from a known-row list.
  - Allowed writers: FineWeb-Edu, v9-web-text, OpenR1-Math, DeepMath, DAPO-Math, ARC-AGI-2.
  - Everything else is a non-target: unknown writers, Qwen teacher rewrites (a small model), agent traces, memory batches, Nemotron.
  - `text_clean_v1` keeps 5.40M of 12.35M tokens. In v2, 1,950 of its 3,015 packed rows are skipped (at least half denied) and 2,865 carry masked spans. Denied, by token count: unknown 3.04M, small-model rewrites 1.58M, agent traces 0.60M, memory_batch_ 0.29M, and more.
  - K3 keeps 1,511 of 4,135 row memberships, so 166 of its 487 packed rows are usable.
  - Every staged record and caption is checked by its `author` field: targets only from public, label, strata, cyril or registry. Template requests are conditioning only.
  - The AVCaps GPT-4 summaries were dropped upstream and checked again here: 1 refusal in the unit test, 0 in the products.
- **Small-model and agent data.** None is used. That covers selfgen, toolcall rescore, Q38-generated text, vision_in answers, media-judge captions, and the 674k + 206k memory tool-call events.
- **The raw Strata/Sprite capture spool is excluded by name.** It has 242 `strata-capture-*` batches. 3,580 of those items reached `intake-text` and were refused there.
- **Hold-out registry v3.** It screens every media sha256 and every request and target text: exact normalised text, plus 13-word shingles.
  - The staging products' own hold-out lists are also enforced. `holdout: true` and `dup_of` rows are skipped. A caption equal, after normalisation, to any held-out text of its product is refused: 10,688 MSR-VTT captions, 554 SFX and 150 AV.
  - Each dataset's `val` split is its dev/eval split and is not trained (2,614 clips).
  - Not done: folding the products' hold-out lists into a registry v4.
- **BFCL (`bfcl_native_v1`, B-0).** A whole-row hold-out screen hits all 500 rows, because the task template and the shared tool schemas recur in the held-out banks. Comparing the question alone against the registry questions finds:
  - 13 exact matches, all from the `codex_recovery_heldout` bank;
  - 2 more that share a 13-gram;
  - 0 rows from the `bfcl62` bank itself.
  
  Those 15 rows are refused through `deny_rows`, so 485 rows train. The tool schemas of held-out questions still appear in training prompts, which is accepted: schemas are part of the prompt at test time. BFCL scores on run 0 are not held-out evidence beyond the BFCL-62 split (Cyril, 10-17: "tool-bfcl-native stays as data").
- **Stale IQ3_S labels.** 134 of the 142 intake batches dated on or after 10-27 still name the old RVN IQ3_S in `_meta.json`. Sealed batches are never edited. The Strata teacher label in the manifest says what the caches are: "the RVN IQ3_S cache, NOT the live Q4_K_XL". New Strata KD from the support workflow (`strata-kd-gen`) carries `strata-q4kxl-abl-v2`.
- **Prompts.** Intake image, music and video items are trained only when their prompt is Strata-written (`prompt_source.kind=strata`). Per-batch refusal counts are in each intake `source.json`. Mesh items go to the hy3d path, and non-final H3 SOL stages are refused.
- The full B-0 report is `logs/provenance_audit_20261028.json`, with `logs/bfcl_holdout_rows.json` and `logs/bfcl_holdout_probe.json`.

## Streaming intake (B-6)

`run0_intake_sync.py sync` runs outside the trainer, at most once per 24 h. Each run:
1. exports the admitted train items of chimera_corpus (raw parameterised SQL, read-only session);
2. builds only what no run 0 source holds yet:
   - intake sources;
   - staging records (new batches only);
   - new documents;
   - a daily slice of the external clips not yet built (`--staging-clips 6000` per product, 4 workers);
3. seals `run0_data_v{N+1}.json`, chained by `parent_sha`, with the same author allowlist;
4. verifies the new sources and writes `NEXT_MANIFEST` for the supervisor to adopt at its next checkpoint;
5. registers the version in omni_lab (`dataset_version` and `run_dataset`).

A day with nothing new seals nothing.

What is still unbuilt at the source:
- about 29k of ~49k trainable SFX clips (Clotho + AudioCaps);
- about 16k of ~28k trainable AV-caption clips;
- about 8k of ~16k VGGSound clips.

All of them arrive through these daily slices. The D4 TTS concept records and new intent-bank, procedural-edit and edit-pair batches stream in the same way. Media that needs the judge waits while `media-judge.timer` is paused (Q2).

## Lead calls in this build (Claude leads Omni; Cyril may revise)

- **The music bucket is music plus sound effects** (6 %). `ext-sound` is drawn at a 0.40 fraction of the bucket.
- **Video understanding** (2 %) draws `ext-av-understand` at 0.45 and `ext-video-caption` at 0.35. The judged intake video gets the rest.
- **Video generation** (4 %) draws `ext-av-gen` (real joint audio-video, as sequential video-then-audio suffixes) at 0.40.
- **Decision twins** (4 %) draw the Strata-written intent bank at 0.25. The remainder are twins of every generation record.
- **Image generation** draws procedural edits at 0.05, and unjudged Qwen-Image edit pairs at 0.05 once they exist.
- **Text KD** draws the PDF corpus at 0.15 and registry web text at 0.05. These are CE-only document chunks of about 10k characters, with the first ~15 % shown as context. They stop at the CE cap, 4 epochs per 7 days.
- **First external slice** (v2): SFX 20,000 clips, AV caption 12,000, VGGSound 8,000, MSR-VTT/AVCaps video captions all of them (8,166 after screening). That took 1 h 50 min at 16 CPU workers. The rest streams daily.
- **Clip units:**
  - sound is cut to the first 10 s;
  - AV understanding pairs the middle 1.92 s of video (256x448, 24 frames at 12.5 fps) with up to 10 s of audio centred on it;
  - AV generation pairs the same window at 192x320 with its aligned 1.92 s of audio.
  
  Captions describe the whole clip, so a 1.92 s window is a sample of it. Longer video windows wait for the C-8 activation offload.
- **Intent rows (decision-only records) are served as twins:** they end after `<|/out|>`, never with `<|im_end|>`.
- **Q1 (run-0-only deviation):** every cached K3 target is used with the plan's repetition cap. In trainable positions K3 is under 1 % of the KD signal.

## Gaps (Q7, for Cyril)

- **3D** starves by day 4-5. hy3d-voxel64 still holds the 10-22 set (111 items); the judged GLBs from the 10-28 intake (740 mesh items) were not voxelised in this build.
- **Concept-interleaved data** is thin: 168 records. The D4 TTS job adds them as media-gate admits work.
- **K3** is capped by day 2-3.
- **No strata-kd-gen yet.** The writer's Q4 KD corpus, with no `lse`, is not wired. It needs a teacher-identity ruling and the `n_probs` check (C-15).
- **Edits are thin:** 790 image-edit records, plus 128 procedural.
- **Not trained:** the AudioCaps / Clotho / AVCaps `val` splits.

## Verification and commands

- Builds:
  - `run0_build_v1.sh dadee8b` (v1 seal);
  - `run0_build_v2.sh 9f5d5d5 16 sfx_caption_v1=20000 av_caption_v1=12000 av_label_v1=8000` (staging builds + v2 seal);
  - both run as `systemd-run --user` units with MemoryMax 10-12G and MemorySwapMax 0.
- The v2 build's cgroup measured about 3.7 GiB anon plus reclaimable cache when sampled, with memory pressure under 0.1 %. Strata kept its RAM. Host MemAvailable stayed at 34-84 GiB during the build; other lanes moved it.
- Disk: root 108 GB free (floor 75; run 0 data on root is still the 16 GB of 10-22 parts). The Lexar has 196 GB free (10-28 parts 31 GB). Nothing was written to the NAS.
- Earlier failures, now fixed:
  - the first probe build was killed by systemd-oomd at an 8 GiB peak (page cache charged to the unit, plus forked workers copying the planner state). Fixed with spawned workers and fadvise drops;
  - FLAC piped to stdout carried a 0-sample STREAMINFO. Fixed by encoding to a seekable file, checked by a test;
  - the 10-28 08:00Z seal attempt died on `toolcall_flash_v1`'s `lengths` key.
- Decode failures in v2: 203 of 48,166 planned clips (0.4 %). Most are Clotho 7z member names with HTML entities (fixed in `b0744f6` for the next sync), plus silent tracks and clips shorter than 1.92 s. Each one is counted in `source.json`, with examples.
- Not verified: the trainer serving these sources end to end through the real layout API. That needs area A's A-2 on the integration branch; the one skipped test pins it.
