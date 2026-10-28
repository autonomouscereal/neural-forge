# run0 survey: weights, teachers and data (refreshed 2026-10-28, read-only)

**Scope.** What exists now for run0: plant Qwen 2B, ablate it, then distil across all modalities on the v2 architecture using the 4 P40s.

**What I checked.**
- The AI server: root, Lexar and the NAS mount.
- The workstation C: and D: drives.
- HF Hub metadata only. No weights were downloaded.
- The teacher-cache manifests, the corpora and the run0 data parts.
- The intake drop zone, the media spool, the generator workers, and the llama.cpp server source.

**What I changed.** Nothing was launched, claimed or downloaded, and nothing in the campaign was modified. Two scratch scripts were uploaded to `/home/cereal/`, run, and deleted.

**Previous version.** The 2026-10-22 version of this file is at `backups/survey_data.2026-10-22.md`.

**Notation.**
- `$OC` = `/home/cereal/omni-campaign`
- `$L` = `/mnt/lexar/omni-campaign`
- `$T` = `$L/teacher_cache`
- `$OMC` = `/mnt/lexar/omnicoder_0.1.0`
- "Positions" = stored next-token teacher distributions.
- [V] = verified this session; [R] = carried over from the 10-22 survey or the docs, not re-read.

## 0. Bottom line

1. **The 2B is still not on disk.** Qwen3.5-2B is the exact co2-2b shape [V].
   - Not on the server: `/home/cereal/models`, `/mnt/ai_data/models/Qwen`, `/mnt/ai_data/hf_cache`, `~/.cache/huggingface/hub`, `/home/cereal/hf-downloads` or Lexar.
   - Not on the workstation: it has only `D:\SpriteBuild\models\vlm\Qwen3VL-2B-Instruct-Q4_K_M.gguf`, a different model.
   - Hub ids:
     - `Qwen/Qwen3.5-2B`: post-trained, sha `15852e8c`, modified 2026-03-02.
     - `Qwen/Qwen3.5-2B-Base`: sha `b1485b2f`, modified 2026-04-23.
   - Both are Apache-2.0, not gated, and one BF16 safetensors file of **4,548,221,488 B** (2,274,069,824 params). Each repo is 4.57 GB.
2. **One id space covers everything.** Qwen3.5-2B, Q38-27BU, the cached IQ3_S teacher and the **live Strata Q4_K_XL** share token ids [V].
   - The live GGUF `Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL` reports 248,320 tokens with pre-tokenizer `qwen35`. Its ids 0-248,069 equal Qwen3.5's exactly.
   - Q38-27BU differs from Qwen3.5 only by 7 added audio/TTS specials at 248,070-248,076. The base vocab (248,044) and the merges are identical.
   - So the existing top-64 caches apply to the planted 2B as they are.
3. **Text teacher signal in hand: about 26M distinct tokens.**

   | Teacher | Positions | Stored as |
   |---|---|---|
   | Q38-27BU | 37,206,719 | top-64 + lse |
   | Flash Next IQ3_S | 35,081,477 | top-64 + lse |
   | K3 | 1,889,438 K3 tokens | top-32 + lse |

   - At run0 speeds this is less than one day of training. The caches are unchanged since 10-21.
4. **Live Strata can give top-n logprobs only for tokens it generates.**
   - Requests: `n_probs`, or OAI `logprobs:true` + `top_logprobs` (default 20). Pre-sampling distributions when `post_sampling_probs=false`.
   - It refuses `echo`, so it cannot score corpus or prompt tokens [V, source].
   - Teacher-forced KD on fixed text needs the offline `teacher-score` binary on its own GPUs.
   - **Caveat:** Strata has served the Huihui abliterated-v2 Q4_K_XL since 10-27. That is a different finetune from the RVN IQ3_S that produced the 14 GB cache.
5. **Media grew about 10x since 10-22, but only on the intake side.**
   - The run0 data parts were built at 10-22 20:12Z and do not contain the growth.
   - Judged intake now: 4,071 images, 1,318 music clips, 414 meshes, 224 mp4 + 69 webp video. Speech/audio/documents: 0.
6. **New find: LibriSpeech clean train-100 + train-360 (about 460 h, multi-speaker) is on Lexar** [V].
   - Location: 64 orphaned HF parquet blobs, 29.65 GB, in `$OMC/models/hf/hub/datasets--librispeech_asr/blobs`.
   - The snapshot directories are empty because exFAT drops symlinks. All 64 blob names match the Hub LFS sha256 list at revision `71cacbfb`.
   - This closes the "train speech is dev/test only" gap.
7. **Same-concept sets.** Three-modality same-origin items exist only as generator by-products:
   - H3 SOL: prompt + video + audio;
   - HY3D: prompt + Qwen-Image picture + mesh;
   - edits: instruction + source + result.

   No generator produces concept sets (the same concept as image + text + audio). No `concept` logic exists in `ai-serve/gen` or `ai-serve/media` [V].
8. **Disk and RAM are tighter than on 10-22** [V]:
   - root **114 GB free** (94 %; was 330 GB);
   - Lexar **381 GB free** (91 %; was 435 GB);
   - NAS **1.0 MB free** (100 %);
   - host RAM 251 GB total, **30 GB available**. Strata `llama-server` has a 150 GB RSS (`--no-mmap`, `--cache-ram 49152`) and ComfyUI (GPU4) 60 GB.
9. **P40s.** GPU1/2/3/5 are idle (5 MiB each, no `gpu*.lock` held). However, `media-judge` (owner claude, lane L-EXP) claims **GPU5 every hour** for 20-30 min (`$OC/distill/queues/p40.claims`, last at 01:35Z). A 4-P40 run needs that job moved or paused first.

## 1. The 2B to plant

The HF config was read 2026-10-28 for both `Qwen3.5-2B` and `-Base`; they are identical [V].

| | Qwen3.5-2B | co2-2b (ARCH_V2 §3) | Match |
|---|---|---|---|
| Arch | `Qwen3_5ForConditionalGeneration` | ChimeraOmni v2 | — |
| Layers | 24, `full_attention_interval` 4 (3 linear : 1 full) | 24 = 18 GDN + 6 gated GQA | yes |
| Width / FFN | 2048 / 6144 | 2048 / 6144 | yes |
| Full attention | 8 q / 2 KV x 256, `attn_output_gate` true | 8 / 2 x 256, sigmoid output gate, QK-norm | yes |
| RoPE | partial 0.25 (64 dims), theta 1e7, mrope [11,11,10] interleaved | 64 of 256, theta 1e7, mRoPE P1 | yes |
| Linear (GDN) | 16 key / 16 value heads x 128, conv kernel 4 | 16 / 16 x 128, scalar GDN decay, short conv 4 | yes |
| Vocab / tie | 248,320, tied | 248,320 + 512 control, tied | +512 new rows |
| Context | 262,144 | 262,144 | yes |
| `rms_norm_eps` | 1e-6 | 1e-6 | yes |
| MTP | 1 layer (`mtp.*`, 15 tensors) | `mtp_heads=0` (training auxiliary only) | drop or use as an auxiliary |
| Vision tower | ViT depth 24, 1024 wide, patch 16, merge 2, temporal 2 (`model.visual.*`, 297 tensors) | none (raw 3,072-value boundary) | discard |
| Ours only | — | Block AttnRes (zero init), raw boundary 10.66M, x0 head 29.36M, A6b | new |

- **Tensors.** The weight map has `model.language_model.*` 320 tensors, `model.visual.*` 297 and `mtp.*` 15.
- **Parameters.** The repo holds 2.274 B. The text trunk without the ViT (about 0.3 B) and MTP is about 1.9 B, which is consistent with co2-2b's 1,923,133,041 including our new parts.
- **Fetch target.** `/home/cereal/models/Qwen3.5-2B` on root; 4.55 GB fits in 114 GB. Never the NAS.
- **Licence.** Apache-2.0 (Hub card). The `V2_INIT_RULE` note and the `$OC/rulings/arch_v2_init_source.json` ruling are still absent [V: `rulings/` holds only `OWNER-RULING-2026-10-12-STAGE-D-GO.json`]. Cyril's 10-22 order is the ruling to record.
- **Base vs post-trained.** The configs are identical. The post-trained checkpoint carries chat and tool behaviour, and the existing caches were rendered with the chat template (`enable_thinking=False`) [R]. This is the lead's call.

## 2. Text teachers and corpora that exist now

### 2a. Same-vocab and K3 caches [V, aggregated from every `manifest.json`]

Every qtext shard holds `ids.u32`, `pos.u32`, `topk_ids.i32 [P,64]`, `topk_logprobs.bf16`, `lse.f32` and `shard.json`. All are sealed and `all_finite`, with `head_rows [0, 248320)`.

| Teacher | Location | Caches / size | Positions | Per corpus |
|---|---|---|---|---|
| Q38-27BU (HF bf16, fp32 head, tool `qtext_teacher_cache` 1.0.5) | `$T/qtext/q38-27bu/<corpus_sha>/` | 42 / 14.73 GB | 37,206,719 | `text` 20,000,768; `text_clean_v1` 12,349,164; `long` 1,000,609; K3 rows pass-0..3 1,991,482; `bfcl_native_v1` 48,914; `d2_refill` 87,972; **`selfgen` 1,008,982 and `rescore` 718,828: excluded (§6)** |
| Flash Next Strata **IQ3_S RVN** (`cb5958e5…`, llama.cpp `gguf_teacher_cache`, `-ts 1,1,1,1` on 4 cards) | `$T/qtext/flash-next-strata-iq3s/` | 38 / 13.89 GB | 35,081,477 | `text` 20,000,768; `text_clean_v1` 12,349,164; K3 rows 1,991,482; `toolcall_flash_v1` 691,149 (of 3,000,787 forward); `bfcl_native_v1` 48,914 |
| Kimi-K3 provisional bf16 | `$T/k3/eb35851b2496a628-provisional-bf16/pass-{0..3}-of-4/` + `xtok/` | 4 passes / 824 MB | 473,702 + 473,294 + 473,124 + 473,973 = 1,894,093 | 4,135 docs, 7.86 MB of text, 1,889,438 K3 tokens; routes text/code/tool/long_context. Layout: `topk_ids int32 [P,32]`, `topk_logprobs bf16 [P,32]`, `lse` over 163,840, `target_logprob`. `xtok` maps it to Qwen spans |
| Q38-27BU VL | `$T/qtext_vl/q38-27bu/` [R] | 2 / 1.3 GB | 1,724,508 | vision_in answers: excluded as text |

**K3 notes.**
- The cache is provisional (bf16 numerics), and its top-32 is narrower than the Qwen-family top-64. Both still meet the top-10+ target.
- K3 itself runs only via the layer-streamed runner on GPU6. GPU6 is now Strata's, so **no K3 extension is possible without touching GPU6**. That is off-limits.

### 2b. Live Strata as a teacher

- **Model.** The live model is a different finetune from the cached teacher.
  - Live: `/home/cereal/models/gguf/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/` (4 files), arch `qwen4exp`, served on `:18401` behind the gate on `:18400`.
  - Cached teacher: `Qwen3.8-Flash-Next-Strata/RVN-…-IQ3_S-merged.gguf` (89 GB), which is still on disk.
- **llama-server, from the `llamacpp-xcache` source.**
  - `populate_token_probs` returns top-n pre-sampling logprobs per **generated** token. The only limit is the request size.
  - `echo` throws "Only no echo is supported", so there is no prompt scoring.
  - The idle gen workers do not request logprobs today [R].
- **Offline scoring.** `$OC/export/llamacpp/bin/teacher-score` uses libllama and writes qtext caches; set `LD_LIBRARY_PATH` to its bin.
  - It last ran IQ3_S across 4 P40s [R].
  - The Q4_K_XL is larger: about 95 GB derived from `du` of `models/gguf` = 218 GB, UNVERIFIED. It may not fit 4 x 24 GB with context.
  - Any scoring competes with run0 for the same P40s.
- **Provenance bug.** Intake `_meta.json` licence strings still say "model output of … RVN Strata IQ3_S" on batches dated 10-28, after the 10-27 Q4 switch. The model label in the capture and gen-worker exports is stale.

### 2c. Text and tool corpora

These are all Qwen3.8-tokenizer id views, which are the same ids as Qwen3.5.

| Corpus | Size [V] | Tokens | Notes |
|---|---|---|---|
| `$OC/distill/corpus/text` | 186 MB | 20,000,768 (4,883 x 4096) | Registry walk; routes text, tool, code, long, math, reasoning [R] |
| `text_clean_v1` | 120 MB | 12,349,164 | Held-out-clean **repack of `text`**, not new content. It is the run0 text source |
| `k3/pass-0..3` | 61 MB | 1.99M Qwen / 1.89M K3 | Dual-tokenized |
| `long` | 5.1 MB | 1,000,609 (31 docs x 32k) | Nothing at 50k or more |
| `toolcall_flash_v1` | 63 MB | 3,000,787 (691,149 supervised) | 1,599 Strata IQ3_S tool conversations, sealed `72053993…` on 10-20 |
| `bfcl_native_v1` | 6.2 MB | 415,513 (48,914 supervised) | BFCL v4, public |
| Canonical rows `distill/corpus/staging/rows.jsonl` | 267 MB | 91,984 records | Corpus v1 sealed 10-15 (`$L/corpus/v1/adopt_v1_index.json`). Routes: text 30,143; tool 14,329; code 11,169; reasoning 6,211; long_context 5,357; math 4,997; image 4,228; music 2,768; ocr 2,585; audio 1,784; tts 1,700; speech 1,378; video 1,336; others below 1.2k each. **About 6.6k tool rows are Claude/Codex memory traces; exclude them** (support_scout_corpus §0.4) |

**Idle-generator text in intake** (`$L/corpus-store/chimera-intake/text`) [V]:

| Kind | Rows now | On 10-22 | Size | Rule |
|---|---|---|---|---|
| `strata-toolcall-*` | 2,904 | 115 | 22.8 MB | Allowed (gen-worker-text, synthetic, held-out 13-gram screened) |
| `strata-reasoning-math-*` | 879 | 28 | 3.3 MB | Allowed |
| `strata-instruction-code-*` | 535 | 13 | 3.1 MB | Allowed |
| `strata-long-context-*` | 333 | 13 | 21.9 MB | Allowed |
| `strata-capture-*` (162 batches) | 19,548 | 1,199 | 153.7 MB | **Exclude** (capture spool, §6). Every batch says `outputs from {'background': N}`, none mentions Sprite, so this is likely idle-generator traffic that duplicates the rows above |

Not scored and not in run0: the NAS PDF corpus (5,869 PDFs, 36.5 GB, no text extracted), and omnicoder text/wiki [R].

## 3. Media teachers and data that exist now

### 3a. Teacher caches

These are [R] and unchanged since 10-22. The directory list was re-verified: `ace h3 hy3d k3 qimg qtext qtext_vl qvl32_vit qvl32_vit_ref`.

| Cache | Items | Size | Note |
|---|---|---|---|
| `ace/ace_lm_codes`, `ace/ace_dit_velocity` | 6,667 shards / 964 | 19 GB | Codec-space codes are a teacher target only; velocity is ACE-latent |
| `qimg/*` (Qwen-Image **old** fp8: TE, VAE, velocity) | 32 / 2,048 encode / 2,048 velocity targets | 13 GB | The old model; ARCH_V2 says not to use it |
| `qvl32_vit/qvl32_vit_l13` | 7,492 images | 73 GB | REPA feature targets |
| `hy3d/hy3d_trajectories` | 256 objects | 21 GB | Latent trajectories |
| `h3/*` | — | 39 MB | Practically empty |

All media teachers work in their own latent spaces. Against a raw student they provide **outputs (data), not logits**.

### 3b. Generator output: the gen-worker-media spool and judged intake [V]

| Kind | Spool `ready` (awaiting intake) | Spool `rejected` | Judged intake (`chimera-intake/*`) | On 10-22 |
|---|---|---|---|---|
| Image t2i + edit (Qwen-Image 2.1, 1024²) | 1,954 png (provenance: about 1,042 t2i, about 777 edit) | 3,281 png | **4,071 png**, 21 GB | 278 t2i + 45 edits |
| Music (ACE-Step 1.5) | 533 mp3 + 52 flac | 589 mp3 + 64 flac | **1,206 mp3 + 112 flac**, 5.2 GB | 129 |
| Video (H3 SOL with audio; LTX fallback) | 104 mp4 + 36 webp | 128 mp4 + 50 webp | **224 mp4 + 69 webp**, 1.5 GB | 18 + 5 |
| 3D (HY3D 2.1 GLB from a Qwen-Image picture) | 135 glb | 463 glb | **414 glb**, 1.7 GB | 141 |
| Speech / audio / documents / mixed | 0 | 0 | 0 | 0 |

- **Prompt provenance.** Prompts and edit instructions come from Strata (`prompt_source.kind=strata`), seeded from COCO captions (CC BY 4.0). Example: `ready/images/aiserve-image_edit-20261027/2270ed11….provenance.json`.
- The judge rejects about 60-75 % of items.
- `gen-worker-media` has no speech/TTS kind. `speech-worker@tts/asr` serve Sprite and write nothing to intake.

### 3c. run0 data parts already built

These are in `$OC/run0/data/parts`, built 2026-10-22 from the 10-22 intake snapshot [V]:

| Part | Provenance | Records (tasks) | Raw rows | Positions (est.) | Licence |
|---|---|---|---|---|---|
| coco2017-caption | public | 118,287 (image_caption) | 31,676,593 | 38.1M | annotations CC BY 4.0; Flickr images (ruled fine 10-14) |
| coco2017-t2i | public | 118,287 (image_gen) | 34,002,240 | 40.1M | same |
| ljspeech | public | 26,196 (13,098 asr + 13,098 tts) | 2,165,722 | 3.73M | public domain |
| librispeech-asr | public | 4,551 (asr; dev/test-clean only) | 325,439 | 0.59M | CC BY 4.0 |
| maestro-piano | public | 3,960 (1,980 gen + 1,980 label) | 495,000 | 0.69M | **CC BY-NC-SA 4.0** |
| modelnet10-64 | public | 2,000 (label + text_gen) | 1,024,000 | 1.09M | academic (ruled fine) |
| h3set-video | media-teacher | 506 (video_gen) | 728,640 | 0.86M | H3 outputs; **audio mismatched** [R] |
| hy3d-voxel64 | media-teacher | 333 (gen / text_gen / caption) | 198,912 | 0.22M | **Hunyuan non-commercial community licence** |
| intake-image-t2i | strata-gen | 247 | 63,232 | 0.09M | Qwen-Image 2.1 Apache-2.0 |
| intake-image-edit | strata-gen | 32 | 16,384 | 0.02M | same |
| intake-music-ace | strata-gen | 568 (284 gen + 284 caption) | 71,000 | 0.10M | ACE-Step 1.5 |
| intake-video | strata-gen | 33 (19 video + 14 AV) | 47,856 | 0.05M | H3/LTX |
| intake-strata-text | strata-gen | 479 (148 chat + 331 tool) | — | 0.98M | Strata, cleared 10-12 |
| cyril-speech | cyril | 382 (191 asr + 191 tts) | 21,896 | 0.04M | Cyril's own data |

The intake rows in this table are **5-15x behind** the current intake in §3b.

### 3d. Public sets on disk

| Set | Path | Size | Count | Licence |
|---|---|---|---|---|
| COCO 2017 | `$OMC/data/coco2017` | 135 GB | 118,287 images, about 5 captions each | CC BY 4.0 annotations; Flickr images |
| **LibriSpeech clean train-100 / train-360 / val / test (parquet)** | `$OMC/models/hf/hub/datasets--librispeech_asr/blobs` (orphaned blobs) | 6.43 + 23.22 + 0.34 + 0.35 GB (64 files) | about 100 h + 360 h + 5.4 h + 5.4 h | CC BY 4.0. **Not yet used** |
| LibriSpeech dev/test-clean (flac) | `$L/corpus/external/librispeech`, `$OMC/data/librispeech` | 6.8 GB / 3.2 GB | 5,323 items | CC BY 4.0 |
| LJSpeech 1.1 | `$OMC/data/ljspeech` | 16 GB | 13,100 clips, 24 h, one speaker | public domain |
| MAESTRO v3 | `$OMC/data/music/maestro-v3.0.0.zip` | 108 GB | about 200 h piano | CC BY-NC-SA 4.0 |
| ModelNet10 | `$L/corpus/external/modelnet10/ModelNet10.zip` | in 6.8 GB external | 4,899 meshes | academic |
| Lakh MIDI, WIT, vl_coco / vl_cc [R] | `$OMC/data/...` | 1.8 GB, 12 GB, 2.5 GB | — | CC BY 4.0 / CC BY-SA 3.0 |
| Common Voice 21/22, MusicCaps, laion "got talent", laion-coco | omnicoder HF caches | 40-44 KB each | **stubs, no data** | — |
| webvid / video QA dirs | `$OMC/data/webvid`, `data/video/*` | 1 MB | empty | — |

### 3e. Cyril's speech

- **Cleared data.** `$L/codex-adopted-2026-10-14/speech-native-caches-v2` (6.4 GB): `native-speech-ce-v2` 186, `native-audio-ce-v2` 194, `native-speech-flow-v2` 186 [R].
- **In run0.** Part `cyril-speech` holds 382 records (191 ASR + 191 TTS). Its own `source.json` describes them as "operator-authored lines … rendered by Qwen3-TTS voices", so they are his words in TTS voices.
- **Microphone recordings.** None. `chimera-intake/speech/` is empty, and no `own-*` directory exists anywhere on Lexar or `/home/cereal` [V].

### 3f. Speech and audio teacher models

All on `/mnt/lexar/models` [R]:
- Qwen3-TTS 12Hz: 0.6B and 1.7B;
- Qwen3-ASR: 0.6B and 1.7B;
- granite-speech: 3.3-2b and 4.1-2b;
- CosyVoice3-0.5B;
- Kokoro-82M;
- VoxCPM2.

The live `speech-worker@*` services are off-limits. Using any of these models for TTS pairs needs a P40 lane.

## 4. Per-modality table

| Modality | In / out | Source | Size now | Location | Teacher signal | Licence / rule notes |
|---|---|---|---|---|---|---|
| Text (general, code, math, reasoning) | in + out | Registry repack | 12.35M tok (clean); 20.0M raw | `$OC/distill/corpus/text_clean_v1` (`text`) | Q38 + Flash IQ3_S top-64 + lse | Q38 tokenizer = Qwen3.5 ids; drop Claude/Codex trace rows |
| Long context | in + out | Synthetic 32k concatenations + needles | 1.0M tok (31 docs); long-context gen 333 rows / 21.9 MB | `corpus/long`, intake/text | Q38 top-64 (`long` only) | Nothing at 50k or more; Cyril's floor is 50k |
| K3 knowledge rows | in + out | K3 passes 0-3 | 1.99M Qwen tok | `corpus/k3`, `$T/k3/...` | K3 top-32 + xtok; Q38; Flash | Provisional bf16; no extension possible (GPU6 off-limits) |
| Tool calls | in + out | Strata IQ3_S tool conversations; BFCL v4; gen-worker-text | 3.0M tok (691k supervised); 415k (49k); 2,904 rows (22.8 MB) | `toolcall_flash_v1`, `bfcl_native_v1`, intake/text | Q38 + Flash top-64 (first two only) | Q38 `rescore` excluded; capture batches excluded |
| Reasoning / math / code (generated) | in + out | gen-worker-text | 879 + 535 rows | intake/text | Strata answers (sequence-level) | Allowed |
| Image in (caption, VQA, OCR) | in | COCO 2017; registry image/ocr rows; vision_in images | 118,287 images; 4,228 + 2,585 rows; 7.5k images | `$OMC/data/coco2017`, `staging/rows.jsonl`, `corpus/vision_in` | COCO captions; QVL32 ViT-L13 features (7,492) | Real sizes (COCO about 640x480); vision_in answers excluded |
| Image out (t2i) | out | Qwen-Image 2.1 via media-gate; COCO | about 4k judged (t2i + edit) + 1,954 ready; COCO 118k as image_gen | intake/images, spool/ready | outputs only | Apache-2.0; judge captions (Qwen3.6-27B) are not training text |
| Image edit | in + out | Qwen-Image 2.1 edit triplets | part of the 4,071; about 777 ready | intake/images, spool/ready | outputs only | Strata-written instructions |
| Video in | in | H3 set, intake video, registry video rows | 512 clips; 224 mp4; 1,336 rows | `corpus/media/h3`, intake/video | none | H3 set audio is mismatched |
| Video + joint AV out | out | H3 SOL (draft + refine + muxed audio), LTX fallback | 224 mp4 + 69 webp judged; 104 + 36 ready | intake/video, spool/ready | outputs only | The only real same-origin AV source |
| Speech in (ASR) | in | **LibriSpeech train-100/360 (new)**, dev/test, LJSpeech, Cyril set | about 460 h + 10.8 h + 24 h + 186 items | `$OMC/.../librispeech_asr/blobs`, `$L/corpus/external`, `$OMC/data/ljspeech`, codex-adopted | transcripts; ASR models on Lexar | CC BY 4.0 / public domain / Cyril's |
| Speech out (TTS) | out | LJSpeech; Cyril flow targets; registry tts rows | 13,098 clips; 186 x 4 sigma; 1,700 rows | as above | Qwen3-TTS (not run for data) | One-speaker TTS only until TTS pairs are made |
| Music | in + out | ACE-Step 1.5 renders; MAESTRO; registry music rows; ACE caches | 1,318 judged + 585 ready; about 200 h piano; 2,768 rows | intake/music, spool, `$OMC/data/music`, `$T/ace` | ACE codes and velocity (latent; teacher only) | MAESTRO NC |
| 3D | in + out | HY3D 2.1 GLB; ModelNet10; voxel sets | 414 judged + 135 ready GLB; 4,899 meshes; 333 voxel records | intake/3d, `$L/corpus/derived/hy3d_voxel64/v2`, external | HY3D trajectories (latent) | HY3D non-commercial |
| Same-concept, 3+ modalities | mixed | H3 SOL (text + video + audio), HY3D (text + image + mesh), edits (text + image + image) | about 224 / 414 / about 800 | intake | — | No concept-set generator exists. Cheapest addition: COCO image + caption + Qwen3-TTS speech of the caption |

## 5. Disk, RAM and cards

| Mount | Size | Used | Free | Notes |
|---|---|---|---|---|
| `/` (ext4, `/dev/sda2`) | 1.8 T | 94 % | **114 G** | Qwen3.5-2B (4.55 GB) goes here; leave room for checkpoints |
| `/mnt/lexar` (exFAT) | 3.8 T | 91 % | **381 G** | No symlinks (hence the orphaned HF blobs); `$OMC` holds the public data |
| `/mnt/ai_data` (CIFS NAS) | 4.5 T | 100 % | **1.0 M** | Do not write |

**NAS top level.** With 8 s `du` per directory:
- Sized: `comfyui` 33 GB, `hf_cache` 6 GB, `neural-forge-ahb-cache-v1` 2 GB, `soc-archive` 2 GB, `chimera-intake` about 1 GB.
- Timed out, so these are the large ones: `backups`, `models`, `neural-forge`, `neural-forge-archives`, `neural-forge-matrix`, `neural-forge-sealed-caches`, `omni-campaign`, `omnicoder_2026`, `training_data` (the 36.5 GB PDF corpus) and `_dataset_intake_quarantine`.

A cleanup needs a proper per-directory size pass run on the NAS host.

**RAM and cards.**
- Host RAM: 251 GB total, 221 GB used, **30 GB available**. Strata llama-server has a 150 GB RSS (`--no-mmap`) and ComfyUI qwen21-gpu4 60 GB.
- GPU0: 23.8 of 24.6 GB used. GPU4: 18.8 GB. GPU6: 45.0 of 46.1 GB.
- GPU1/2/3/5: 5 MiB each.

## 6. Rule flags for run0 data selection

| Item | Why | Default |
|---|---|---|
| `$OC/distill/toolcall/rescore` (Q38 Q4_K_M tool calls) and its 718,828-position Q38 cache | Small-model generation | Exclude |
| `selfgen` (Q38 free-run) and its 1,008,982-position cache | Same | Exclude |
| `vision_in` answers and the `qtext_vl` cache | Answers written by Q38-27BU VL | Keep the images and prompts; drop the answers |
| `chimera-intake/text/strata-capture-*` (162 batches, 19,548 rows) and `ai-serve/gen/capture-export` | Standing rule: the raw Strata/Sprite capture spool is excluded until Cyril cleans it | Exclude. The provenance shows `background` only, so ask Cyril whether these idle-traffic captures can be released; they likely duplicate the allowed generator batches |
| Registry tool rows with Claude/Codex memory-trace ids (about 6.6k rows, 1.41M tok_chat) | Never train on Claude/Codex outputs | Exclude |
| Media-judge captions (Qwen3.6-27B Q4_K_M) | Small-model text | Not training text; prompts are fine |
| `d2_refill` (codex-native v7/v8) | Codex pipeline; provenance unclear | Check before use |
| Old `qimg` caches (old Qwen-Image) | Superseded teacher | Do not use (ARCH_V2 §9) |
| Intake `_meta` model label "RVN Strata IQ3_S" on post-10-27 batches | Stale label; the live model is Huihui Q4_K_XL | Fix the labels in capture_export / gen-worker before sealing a corpus version |
| MAESTRO (NC), HY3D outputs (Tencent non-commercial) | Licence | Record on the run |

## 7. Open points for the lead

1. **Fetch the 2B.** Get `Qwen/Qwen3.5-2B` or `-Base` (4.55 GB) into `/home/cereal/models/Qwen3.5-2B` and write `$OC/rulings/arch_v2_init_source.json` from Cyril's 10-22 order.
2. **Free GPU5.** Move or pause the hourly `media-judge` GPU5 claim before taking all four P40s for about 10 days. Otherwise the run must tolerate a fifth claimant.
3. **Refresh the intake parts.** Rebuild `intake-*` with the 10x growth, and add a LibriSpeech-train part from the orphaned parquet blobs. Map blob to split with the Hub tree API at `71cacbfb`; all 64 match.
4. **More text KD.** On-policy top-n capture in gen-worker-text means `n_probs` on Strata's own generations: a code change in ai-serve, no extra GPUs. Teacher-forced scoring would compete with run0 for the P40s. The live Q4 model is a different teacher from the cached IQ3_S; pick one or label both.
5. **Same-concept sets.** Nothing generates them. The cheapest path is COCO image + caption + TTS of the caption; this needs a TTS lane that is not `speech-worker@*`.
6. **NAS cleanup.** Run a full size inventory on the NAS host. From this host, CIFS `du` times out on every large directory.

**Sources read this session:**
- the `manifest.json` / `cache_config.json` of every qtext cache, and the K3 pass manifests and layout;
- `run0/data/parts/*/source.json`;
- the intake `_meta.json` and jsonl files, and the media spool provenance;
- the HF API and `config.json` for `Qwen/Qwen3.5-2B{,-Base}`, the weight index, and the LibriSpeech tree API;
- the tokenizer.json files of Qwen3.5-0.8B/4B/9B and Qwen3.8-27B-Uncensored, and the live GGUF tokenizer (gguf-py);
- `llamacpp-xcache/llama.cpp/tools/server/*.cpp`;
- `ps`, `free`, `nvidia-smi`, `df` and `p40.claims`.
