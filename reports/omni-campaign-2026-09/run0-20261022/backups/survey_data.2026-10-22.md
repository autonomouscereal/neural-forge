# run0 survey: weights, teachers and data (2026-10-22, read-only)

Scope: what exists now for run0 (plant Qwen 2B, ablate, then distil across all modalities on the v2 architecture on 4 P40s). I checked the AI server (root, Lexar, NAS mounts), the HF Hub metadata (no weights downloaded), the teacher-cache manifests, the corpora, the intake drop zone and the generator workers. Nothing was launched, claimed, downloaded or changed. One scratch script went into `$OC/lab/`, ran, and was deleted.

Notation: `$OC` = `/home/cereal/omni-campaign`, `$L` = `/mnt/lexar/omni-campaign`, `$T` = `$L/teacher_cache`. "Positions" = stored next-token teacher distributions.

## 0. Bottom line

1. **Qwen3.5-2B is the exact co2-2b shape. It is not on disk anywhere.** I checked `/home/cereal/models`, `/mnt/ai_data/models/Qwen`, `/mnt/ai_data/hf_cache`, `~/.cache/huggingface/hub`, `/home/cereal/hf-downloads` and `/mnt/lexar/models`. On disk: Qwen3.5-0.8B, 4B and 9B, plus Qwen3.8-27B.
   - Hub ids: `Qwen/Qwen3.5-2B` (post-trained, 2026-03-02) and `Qwen/Qwen3.5-2B-Base` (2026-04-23).
   - Both are Apache-2.0, not gated, and one safetensors file of **4,548,221,488 B** (2,274,069,824 params, BF16). The total repo is 4.57 GB.
2. **Strata, Q38-27BU, Qwen3.5-2B and the existing text caches share one id space.**
   - Vocab (248,044 BPE) and merges are identical, and special ids 248044-248069 are identical. The table is 248,320 rows.
   - Strata's GGUF reports `tokenizer.ggml.pre = qwen35` with 248,320 tokens.
   - Two differences:
     - Q38-27BU adds 7 audio/TTS specials at 248070-248076. These are untrained reserved rows in Qwen3.5.
     - The Q38 tokenizer.json has the older pre-tokenizer regex (no `\p{M}`). It segments text with combining marks differently. The ids stay valid.
   - So the existing top-64 caches apply to the planted 2B as they are.
3. **Text teacher signal in hand: about 26M distinct tokens.**
   - Each token has 1-3 same-vocab teachers stored as top-64 + lse: Q38-27BU 37.2M positions, Flash Next (Strata) 35.1M, K3 1.89M (top-32, K3 vocab, cross-mapped to Qwen ids).
   - At the hw survey's 33-46M tok/day, the cached text is **less than one day of run0**. Extending it needs either offline Strata scoring on the P40s, which competes with run0, or repeated epochs.
4. **llama-server (Strata, :18400 gate / :18401) can only give top-n logprobs for tokens it *generates*.**
   - Requests: `n_probs`, or OAI `top_logprobs` (default 20). Pre-sampling probabilities when `post_sampling_probs=false`.
   - It refuses `echo` ("Only no echo is supported"), so it cannot score prompt or corpus tokens.
   - Teacher-forced KD needs the offline `teacher-score` binary, which was last run with `-ts 1,1,1,1` on the four P40s.
   - The idle gen workers do not request logprobs today: no `logprob`/`n_probs` in the ai-serve gen, lanes or root `.py` files.
5. **Media teacher material is thin except for image-in and speech:**
   - image-in: COCO 118k images with captions, plus 7.5k ViT feature targets;
   - music: ACE codes for 6.7k shards, velocity for 964;
   - 3D: HY3D trajectories for 256 objects, plus 141 image to mesh pairs;
   - image generation: 256 Qwen-Image velocity items plus 278 judged t2i images;
   - image edits: 45 judged edit triples;
   - video: 18 H3 SOL clips with audio;
   - speech: 380 items from Cyril's set plus about 35 h of public speech.
6. **No three-modality same-concept sets exist.** All paired data is two-way: image-text, text-speech, text-music, image-mesh, video-audio-prompt.
7. **Rule flags (exclude by default).** These are covered in section 6:
   - the Q38-generated tool calls (`toolcall/rescore`), self-gen text and vision_in answers;
   - the strata-capture batches in intake/text;
   - judge captions written by Qwen3.6-27B Q4.
8. **Disk:**
   - root 330 GB free (82 % used);
   - Lexar 435 GB free (89 % used, exFAT);
   - NAS `/mnt/ai_data` **16 GB free (100 % used)**. Do not write there.

## 1. The 2B to plant

| | Qwen3.5-2B (HF config, read 2026-10-22) | co2-2b (`config.py` `chimera_omni_v2_2b_config`) | Match |
|---|---|---|---|
| Layers | 24, `layer_types` 3 linear : 1 full, `full_attention_interval` 4 | 24 = 18 GDN + 6 gated GQA, pattern kda,kda,kda,gated_gqa | yes |
| Width / FFN | 2048 / 6144, SiLU, `mlp_only_layers` [] | 2048 / 6144 dense SiLU | yes |
| Full attention | 8 q heads / 2 KV, head_dim 256, `attn_output_gate` true, partial rotary 0.25 (64 dims), theta 1e7, mrope [11,11,10] interleaved | 8/2 x 256, gqa_output_gate, qk_norm, rope_dim 64, theta 1e7, mrope (11,11,10) | yes |
| Linear attention (GDN) | 16 key heads / 16 value heads x 128, conv kernel 4; tensors `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj` | kda 16/16 x 128, decay "gdn" (scalar), short conv 4, silu output gate | yes |
| Vocab / tie | 248,320, tied | 248,320 + 512 control band, tied | +512 new rows |
| Context | 262,144 | 262,144 | yes |
| MTP | 1 layer (`mtp.*`, 15 tensors) | `mtp_heads=0` | drop it, or keep it as a training auxiliary |
| Vision tower | 24-block ViT, 1024 wide, patch 16, merge 2, temporal patch 2 (`model.visual.*`, 297 tensors) | none (raw 3,072-value boundary) | discard |
| Ours only | none | Block AttnRes (zero init), raw boundary and x0 flow head, A6b RMS unit norm | new |

- **Parameters.** The repo holds 2.274 B. Without the ViT (about 0.30 B) and the MTP layer, the text trunk is about 1.9 B, in line with co2-2b at 1,923,133,041 including the new heads.
- **Base vs post-trained.** The configs are identical and the weight files are the same size. The post-trained checkpoint carries the chat and tool-call behaviour, and the existing caches were rendered with `enable_thinking=False`. The choice is the lead's.
- **Fetch target.** `/home/cereal/models/Qwen3.5-2B` on ext4 root (330 GB free), not the NAS. This matches the hw survey and ARCH §13.
- **Licence.** Apache-2.0, verified both from the Hub card and from the Qwen3.5-4B/0.8B LICENSE on disk.
- **Policy note.** `V2_INIT_RULE` in `config.py` still says "fold-built (no direct placement of any existing model's weights)". Cyril's 10-22 order overrides this for run0, so the ruling file `$OC/rulings/arch_v2_init_source.json` should record that. The file is still absent per ARCH §14.

## 2. Text teachers that exist now

Each cache is one corpus file x one teacher. Every qtext shard has the same layout: `ids.u32`, `pos.u32`, `topk_ids.i32`, `topk_logprobs.bf16`, `lse.f32` and `shard.json`. All are sealed, `all_finite`, and `vocab_rows` 248,320. omni_lab tracks them as datasets `teacher_cache/*`.

| Teacher | Location | Format | Positions | Corpora covered |
|---|---|---|---|---|
| Q38-27BU (Qwen3.8-27B-Uncensored, HF bf16, fp32 head) | `$T/qtext/q38-27bu/<corpus_sha>/` (42 caches, 15 GB) | top-64 + lse, bf16 logprobs | 37,206,719 | text 20.0M; text_clean_v1 12.35M; K3 rows (Qwen ids) 1.99M; long (32k) 1.0M; bfcl 49k; d2_refill 88k; **selfgen 1.0M and toolcall/rescore 0.72M are excluded (section 6)** |
| Flash Next = Strata (`RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf`, sha `cb5958e5…`, arch `qwen4exp`, 48 blocks, 2560 wide, 512 experts top-10) | `$T/qtext/flash-next-strata-iq3s/` (38 caches, 15 GB) | top-64 + lse; llama.cpp logits, lse in double | 35,081,477 | text 20.0M; text_clean_v1 12.35M; K3 rows 1.99M; toolcall_flash_v1 0.69M; bfcl 49k |
| Q38-27BU VL | `$T/qtext_vl/q38-27bu/` (2 caches, 1.3 GB) | top-64 + lse over text positions of VL rows | 1,724,508 | vision_in (excluded as data, section 6) |
| K3 (Kimi-K3, provisional bf16, ruling `k3-b2-bf16-numerics-provisional-2026-10-17`) | `$T/k3/eb35851b2496a628-provisional-bf16/pass-{0..3}-of-4/shards/group-*.safetensors` (825 MB) | top-**32** in the K3 vocab (163,840) + lse + target_logprob, F32/BF16 | 1,894,093 (4 passes of about 474k) | `$OC/distill/corpus/k3/pass-*`: dual-tokenized rows, 122 rows x 4096 per pass |
| K3 to Qwen cross-map | `.../xtok/pass-N/xtok_target.safetensors` + `READY.json` | `qwen_ids[N,32]`, `k3_logprobs`, matched/tail mass | about 467k pairs per pass | pass 0: matched mass 0.908, top-1 mapped 96.1 %, next token identical 94.2 %; 121,755 of 163,840 K3 ids map to a Qwen id |
| Q3.5-4B tiny KD | `$OC/distill/trainer-lane/teacher_cache/q35-4b-tinykd` | — | 1.6 MB, smoke only | — |

- **Precision and parity.** The Flash scorer matches HF fp32 on a P40 (memory 10-18): top-1 100 %, top-64 overlap 99.98 %, KL 1.8e-4.
- **Throughput.** Q38 scored about 613 tok/s in one session.
- **K3 caveats.** The cache is provisional (bf16 numerics), and run 1 (fullkd_data_v3) already consumed it. Its top-32 is narrower than the top-64 of the Qwen-family caches. Cyril's "top-10+" target is still satisfied.
- **Strata live KD.** Strata can return top-n for its *own generations* at idle time, which gives on-policy data with teacher distributions. Prompt-token scoring needs `teacher-score` (`$OC/export/llamacpp/bin/teacher-score`, set `LD_LIBRARY_PATH` to the lane bin).

### Text and tool corpora (all Qwen3.8-tokenizer ids, packed)

| Corpus | Rows | Tokens | Supervised / KD positions | Source and notes |
|---|---|---|---|---|
| `$OC/distill/corpus/text` | 4,883 x 4096 | 20,000,768 | all | Registry walk of `/mnt/ai_data/omnicoder_2026/datasets` (640 shards, 94,318 rows, 35,985 documents). Routes: text 9.67M, tool 2.79M, code 2.67M, long_context 2.34M, math 1.29M, reasoning 0.83M, web_knowledge 0.20M, plus about 20 small routes |
| `text_clean_v1` | 3,015 | 12,349,164 | 16.58M clean KD positions (with kept rows) | The same registry with every held-out document removed (5,061 docs, 3.18M tokens). It is a **subset/repack of `text`, not new content** |
| `k3/pass-0..3` | 488 | 1.99M (Qwen) / 1.89M (K3) | all | Dual-tokenized K3 rows |
| `long` | 31 docs at 32,768 | 1,000,609 | all | Synthetic concatenations plus needle/retrieval tasks; the registry has no rows of 32k or more |
| `toolcall_flash_v1` | 1,599 x 8180 | 3,000,787 | 691,149 | Tool calls written by Flash Next (Strata) on 2026-10-18; categories single/parallel/sequential/multiturn/clarify/error/missing_tool/no_tool |
| `bfcl_native_v1` | 500 | 415,513 | 48,914 | BFCL v4 function calls (public); 62 held-out groups screened, 6 twins dropped |
| `d2_refill` (codex-native v7/v8) | — | 88k KD | 88k | Built by Codex's preparation pipeline; check its provenance before use |
| Canonical store `corpus/canonical` | 91,984 items | — | — | omni_lab dataset; tokenizer-agnostic records. Canonical text recovery was still owed (memory 10-18) |
| Holdout registry | 92,537 | — | — | `$L/corpus/holdouts/registry_v3`; it gates every intake |

**Distinct cached text:** about 20M registry + 2M K3 + 1M long + 3M Flash tool + 0.4M BFCL, so roughly **26M tokens**.

**Not yet scored:**
- the 34 GiB / 5,869-PDF corpus on the NAS (`\\…\ai\training_data\ai-training`);
- omnicoder `text` (4.1 GB) and `wiki` (6.2 GB).

The literature survey warns that raw web text replay is harmful at this size (survey_lit §221).

## 3. Media teachers and data that exist now

### Teacher caches (`$T`, omni_lab `teacher_cache/*`)

| Cache | Items | Size | Teacher | What it trains |
|---|---|---|---|---|
| `ace/ace_lm_codes` | 6,667 shards | 19 GB (with velocity) | ACE-Step 1.5 LM | Music codes. Indices are band-relative in [0, 64000). This is codec-space, so it is a teacher target only |
| `ace/ace_dit_velocity` | 964 | (in the 19 GB) | ACE-Step 1.5 DiT | Music flow velocity |
| `qimg/qimg-velocity-qwen_image_fp8_e4m3fn` | 256 | 13 GB (whole qimg) | Qwen-Image (fp8) | Image-generation velocity |
| `qimg/qimg-te-…`, `qimg/qimg-vae` | 32 / 32 | | Qwen2.5-VL-7B TE / Qwen-Image VAE | Conditioning / latents |
| `qvl32_vit/qvl32_vit_l13` | 7,492 images | 73 GB | Qwen-VL-32B ViT layer 13 | Image-in feature targets (a tower feature, usable only as a teacher target) |
| `hy3d/hy3d_trajectories` | 256 objects | 21 GB | Hunyuan3D 2.1 | 3D generation trajectories |
| `h3/h3-te-qwen3vl_32b…`, `h3/h3-vae` | — | 39 MB | MiniMax H3 | Video/AV: almost nothing cached |

### Media prompt sets (`$OC/distill/corpus/media`, 2.7 GB)

- QIMG: 2,048 prompts (857 description, 1,080 grounding, 111 OCR).
- HY3D: 256 object images.
- H3: 512 clips, 2 s at 24 fps, 320x512. **The audio is unrelated LibriSpeech**, so they are not joint AV, and H3 production stopped on 10-18.
- ACE: 1,024 clips/captions (891 v9 prompts, 132 Jamendo tags).
- ViT: 7,492 images.
- vision_in: 5,896 rows / 7,572 images (4.33M visual tokens), plus 512 video-in clips.

### Idle-generator output (gen-worker-media via media-gate GPU4, judged hourly; intake at `$L/corpus-store/chimera-intake`, mirrored on the NAS)

| Batch | Judged into intake |
|---|---|
| image t2i | 278 (8 on 10-21, 270 on 10-22) |
| image edit | 45 triples (source + instruction + result) |
| HY3D | 141 GLB + 140 source images (image to mesh pairs) |
| ACE music | 120 mp3 + 9 flac |
| H3 SOL video | 18 mp4 (+ 9 audio tracks under music/) |
| LTX video (fallback) | 5 |
| speech / audio / documents | 0 |

- Spool `/home/cereal/ai-serve/spool/media`: ready 34 files, incomplete 27, validation 22, **rejected 2,039**. The judge rejects most items.
- Raw ComfyUI output (unscreened): 381 png, 467 webp, 236 flac, 239 mp3, 57 glb, 6 mp4.
- `hy3d_voxel64` v1: 18 sealed volumes (17 train, 1 val) at `$L/corpus/derived/hy3d_voxel64/v1`.

### Text idle-generator output (gen-worker-text, Strata) that reached intake

| Generator | Rows |
|---|---|
| toolcall | 115 |
| reasoning-math | 28 |
| instruction-code | 13 |
| long-context | 13 |

`gen/toolcall/plan-strata-*.jsonl` (10k) and `plan-raw-*` are *plans*, not data.

### Public sets on disk

| Set | Path | Size | Count | Licence note |
|---|---|---|---|---|
| COCO 2017 train + captions/instances/keypoints | `/mnt/lexar/omnicoder_0.1.0/data/coco2017` | 135 GB | 118,287 images (about 5 captions each) | Annotations CC BY 4.0; images Flickr, mixed licences |
| vl_coco.jsonl / vl_cc / vl / vqa | `…/data/` | 1.5 GB + 1 GB + 0.2 GB | 100,000 / 1,406 / 1,000 / 201 files | COCO / CC3M-style |
| WIT | `…/data/wit` | 12 GB | 4 files | CC BY-SA 3.0 |
| LibriSpeech dev/test-clean | `$L/corpus/external/librispeech` (+ `…/data/librispeech` 3.2 GB) | 6.3 GB | 5,323 items (about 10.8 h) | CC BY 4.0 |
| LJSpeech | `…/data/ljspeech` | 16 GB | 13,100 clips (about 24 h, one speaker) | Public domain |
| MAESTRO v3 | `…/data/music/maestro-v3.0.0.zip` | 108 GB (zipped) | about 200 h piano audio + MIDI | **CC BY-NC-SA 4.0** |
| Lakh MIDI | `…/data/music/lmd_full.tar.gz` | 1.8 GB | — | CC BY 4.0 |
| ModelNet10 | `$L/corpus/external/modelnet10/ModelNet10.zip` | 453 MB | 4,899 CAD meshes | Academic use |
| Video QA dirs (msrvtt, webvid, activitynet …) | `…/data/video/*` | 15 MB total | empty | — |

Cyril ruled on 2026-10-14 that "[Cyril quote withheld]" (intake README). MAESTRO's NC clause is still worth recording on the run.

### Cyril's speech

**Cleared** in memory `omni-speech-data-is-cyrils`. Location: `$L/codex-adopted-2026-10-14/speech-native-caches-v2/payload` (6.4 GB).

| Cache | Items |
|---|---|
| `native-speech-ce-v2` (speech-in CE) | 186 |
| `native-audio-ce-v2` | 194 |
| `native-speech-flow-v2` (speech-out flow, 4 sigma levels each) | 186 |

- The clips are about 4 s at 24 kHz with transcripts.
- The source ids carry a TTS speaker name (`…-002-vivian`), so the audio looks like Qwen3-TTS renders of his authored lines rather than his microphone.
- `chimera-intake/speech/` has no `own-*` recordings dropped yet.

### Speech and audio teacher models on disk

All are at `/mnt/lexar/models`:
- Qwen3-TTS-12Hz 0.6B CustomVoice and 1.7B Base / CustomVoice / VoiceDesign;
- Qwen3-ASR 0.6B and 1.7B;
- granite-speech 3.3-2b and 4.1-2b;
- Fun-CosyVoice3-0.5B;
- Kokoro-82M;
- VoxCPM2.

The live `speech-worker@asr/tts` run on GPU4 and are off-limits.

## 4. Per-modality table

| Modality | In / out | Source | Size now | Location | Teacher signal | Licence / rule notes |
|---|---|---|---|---|---|---|
| Text (general, code, math, reasoning) | in + out | Registry walk (omnicoder_2026), clean repack | 20.0M tok (12.35M clean repack) | `$OC/distill/corpus/text{,_clean_v1}` | Q38 + Flash top-64 + lse; K3 top-32 on its own 2M | Q38-tokenized (old pre-tokenizer regex); retokenize new data with the Qwen3.5 tokenizer |
| Long context (32k) | in + out | Synthetic concatenation + needles | 1.0M tok | `…/corpus/long` | Q38 top-64 | Only 31 docs; nothing at 50k+ (Cyril's floor) |
| K3 knowledge rows | in + out | K3 corpus passes 0-3 | 1.99M tok | `…/corpus/k3`, `$T/k3/…` | K3 top-32 (+xtok to Qwen), Q38, Flash | K3 cache provisional bf16 |
| Tool calls | in + out | Flash Next generations; BFCL v4; Strata idle gen | 3.0M tok (0.69M supervised); 0.42M (49k); 115 rows | `toolcall_flash_v1`, `bfcl_native_v1`, intake/text | Q38 + Flash top-64 | Q38 toolcall/rescore (1,467 rows) excluded |
| Image in | in | COCO 2017, vl_coco, WIT, vision_in images | 118k images / 100k rows / 12 GB / 7.5k | omnicoder data, `…/corpus/vision_in` | QVL32 ViT-L13 features (7,492), COCO captions; vision_in answers excluded | Real sizes are available (COCO about 640x480) |
| Image out (t2i) | out | Qwen-Image teacher cache + judged t2i | 256 velocity items; 278 images (+ about 850 raw unscreened) | `$T/qimg`, intake/images | Qwen-Image velocity; prompts | Judge captions came from Qwen3.6-27B Q4 (do not train on them as text) |
| Image edit | in + out | Judged edit triples | 45 | intake/images/aiserve-image_edit-20261022 | none cached | Qwen-Image-Edit-2509 fp8 is local; production corpus still owed |
| Video in | in | H3 set clips, vision_in video | 512 clips (2 s, 320x512) | `…/corpus/media/h3`, vision_in | none usable | The omnicoder video QA dirs are empty |
| Video + AV out | out | H3 SOL via media-gate | 18 clips with audio (+5 LTX fallback) | intake/video | H3 te/vae (39 MB) only | The 512-clip H3 set has mismatched audio; not joint AV |
| Speech in (ASR) | in | LibriSpeech, LJSpeech, Cyril set | about 35 h public + 186/194 items | `$L/corpus/external`, omnicoder, codex-adopted | Transcripts; ASR teachers on Lexar | Cyril's set is cleared |
| Speech out (TTS) | out | Cyril set flow targets; LJSpeech | 186 x 4 sigma; 24 h | codex-adopted, omnicoder | Qwen3-TTS etc. on Lexar (not run yet) | No `own-*` mic recordings yet |
| Music | in + out | ACE caches, ACE renders, MAESTRO, Lakh | 6,667 code shards + 964 velocity; 1,024 prompts; 236+120 renders; about 200 h piano | `$T/ace`, comfyui/output, intake/music, omnicoder | ACE LM codes + DiT velocity | MAESTRO is NC; codes are codec-space (teacher only) |
| 3D | in + out | HY3D trajectories; image to mesh; voxel set; ModelNet10 | 256; 141; 18; 4.9k | `$T/hy3d`, intake/3d, `$L/corpus/derived/hy3d_voxel64`, external | HY3D trajectories | — |
| Interleaved / same concept | mixed | — | **none with 3+ modalities** | — | — | Cheapest path: COCO image + caption + Qwen3-TTS speech of the caption |

## 5. Disk

| Mount | Size | Used | Free | Notes |
|---|---|---|---|---|
| `/` (ext4) | 1.8 T | 82 % | **330 G** | Qwen3.5-2B (4.55 GB) goes here |
| `/mnt/lexar` (exFAT) | 3.8 T | 89 % | **435 G** | No symlinks; K3 passes need ≥ 200 GB free; omnicoder_0.1.0 holds 281 GB of public data |
| `/mnt/ai_data` (CIFS NAS) | 4.5 T | 100 % | **16 G** | Do not write; it hangs with two heavy writers |

## 6. Rule flags for run0 data selection

| Item | Why | Default |
|---|---|---|
| `$OC/distill/toolcall/rescore` (Q38 Q4_K_M generated tool calls, 1,467 rows, 0.72M supervised) and its Q38 cache | Generated by a small model (Qwen3.8-27B) | Exclude |
| `selfgen` (Q38 Q4_K_M free-run, 1.0M) and its cache `$T/qtext/inputs/selfgen` | Same | Exclude |
| `vision_in` answers and the `qtext_vl` cache (answers written by Q38-27BU VL) | Same | Exclude the answers; keep the images and prompts for re-answering by an allowed teacher |
| `chimera-intake/text/strata-capture-*` (16 batches, 1,199 rows) and `ai-serve/gen/capture-export` (1,205 rows) | The raw Strata/Sprite capture spool is excluded until Cyril cleans it | Exclude. The `strata-{toolcall,reasoning-math,instruction-code,long-context}-*` generator batches are allowed |
| Media-judge captions (Qwen3.6-27B Q4_K_M) | Small-model text | Do not use as training text; prompts are fine |
| `d2_refill` codex-native v7/v8 | Codex pipeline; provenance unclear | Check before use |
| ComfyUI raw output | Unjudged | Judge or skip |
| MAESTRO | NC licence | Record on the run |

## 7. Open points for the lead

1. Fetch Qwen3.5-2B (post-trained or Base) to `/home/cereal/models/Qwen3.5-2B`, then record the init ruling.
2. Decide whether new text gets scored. The P40s are shared with run0. The other option is Strata `n_probs` capture in gen-worker-text, which needs a code change and covers only its own generations.
3. Build the joint-concept set (COCO + TTS) and the real joint AV set (H3 SOL). Neither exists at volume today.
4. Re-render the text corpora with the Qwen3.5 tokenizer and chat template, or accept the Q38 regex difference for the first pass.

Sources read: the cache `manifest.json` / `cache_config.json` files, K3 shard headers and xtok receipts, the corpus manifests, omni_lab `dataset`/`dataset_version`, the Hub API and config for `Qwen/Qwen3.5-2B{,-Base}`, the llama.cpp server source at `$OC/export/llamacpp/llama.cpp/tools/server`, and `ps` of strata-llama.
