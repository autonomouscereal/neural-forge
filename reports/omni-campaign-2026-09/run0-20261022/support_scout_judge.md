# Support scout: the media judge and Strata vision (run 0, D2)

Scout, read-only, 2026-10-28 ~06:10Z (server clock 00:10 MDT). Nothing was changed on the server. Every number below was read from the server, the Hugging Face API or the llama.cpp source. Estimates are marked **(projection)**.

Ruling being served: D2, from `cyril-run0-decisions-2026-10-22`. Cyril said: "[Cyril quote withheld]". RUN0_BUILD_PLAN §0 records the effect: the judge moves to Strata at background priority, vision is enabled first, and GPU5 then joins run 0. Until that swap lands, `media-judge.timer` is paused for the run (option D2a).

## 0. Headlines

1. **Strata can be given vision with one ~0.9 GB file.**
   - Qwen3.8-Flash-Next is natively image+video. Its HF pipeline is `image-text-to-text`, and its config has a `vision_config` (27-block ViT, 1152 wide, patch 16, merge 2, out 2560).
   - The served Q4 GGUF also carries vision markers: the chat template has `<|vision_start|>` and image/video branches, `qwen4exp.ple.image_token_id` = 248056, and the rope sections are `[11,11,10,0]` (IMROPE).
   - The vision tower ships separately as an mmproj (`clip.projector_type = qwen3vl_merger`).
   - The live build (llamacpp-xcache, b96806d96 + xcache.patch) has `libmtmd.so`, `--mmproj`, `--no-mmproj-offload`, `-mmdev`, `--image-max-tokens`, and `input_video` support (ffmpeg-based; `/usr/bin/ffmpeg` is present). It maps `qwen3vl_merger` to `PROJECTOR_TYPE_QWEN3VL` and gives `LLM_ARCH_QWEN4EXP` IMROPE.
   - No mmproj is on the server for this model. The only mmproj files on the server are the Qwen3.6-27B judge's.
2. **The gate blocks images on purpose.**
   - `strata_gate.py` 1.2.3 returns `400 no_vision` for any `image_url` / `input_image` / `image` part, and `/v1/models` advertises `"vision": false`.
   - It does **not** block `input_video` parts. Today those reach llama-server and fail there.
3. **Blocker that is not about vision: all background work is stalled right now.**
   - `GET /gate/status` (background key) returns `background_admission: {allowed: false, why: "MemAvailable 29 GiB < 40"}`.
   - The same 40 GiB floor is in media-gate (`mem_floor_gib: 40`), in `lanes.py claim-run`, and in the Strata launcher's start check.
   - `ai_serve.request_log` has no row from either gate after 01:41-01:42Z on 10-28. Host MemAvailable is 27-29 GiB, swap is 6 of 7 GiB used, and strata-llama itself has 5.1 GB in swap (VmSwap).
   - Cause: the expert-cache layout (`-cmoe`, 71.7 GiB pinned experts plus `--cache-ram 49152`) fills RAM as the prompt cache grows.
   - Effect: a Strata judge at background priority would never be admitted under this floor. The same floor stalls Cyril's "[Cyril quote withheld]".
   - Who decides: the floor belongs to the Strata lane, and it must be re-derived for the new layout.
4. **The old judge has not run since 01:54Z, but its timer is still live.**
   - `media-judge.timer` is enabled and active. It fires hourly at :35 and tries GPU5 → GPU3 → GPU1 → GPU2.
   - Its last successful claim was GPU5, 01:35-01:54Z on 10-28. Every cycle since then has been refused by the 40 GiB RAM floor (rc 5).
   - D2a (pause the timer for run 0) has **not** been applied. Only the RAM floor keeps the judge off the P40s, and run 0 is not on the cards yet (all four P40s show 5 MiB).
   - The run-0 launch workflow should `systemctl disable --now media-judge.timer` before it claims GPU5.
5. **Backlog: 2,341 groups / 2,807 members are pending under spec `c9cc4c21…` (1.2.0).**
   - By kind: image_t2i 1042, image_edit 550, music_ace 533, mesh_hy3d 135, video_h3sol 52, video_ltx 29.
   - Publish reports `not_judged_pass: 2807`, so nothing new reaches intake until a judge runs.

## 1. What `media-judge.timer` runs today

| Item | Value |
|---|---|
| Timer | `/etc/systemd/system/media-judge.timer`, `OnCalendar=*-*-* *:35:00` |
| Service | oneshot, `flock -n media/.judge.lock …/ai-serve-media/bin/python /home/cereal/ai-serve/media/media_judge.py cycle`, Nice 10, IO idle, `TimeoutStartSec=4h`, holdouts and calibration banks `InaccessiblePaths` |
| Code | `media_judge.py` 1.2.0 (940 lines), `media_judge_prep.py` (CLAP/ASR/mesh), `media_sets.py`, spec `media_judge_spec.json` 1.2.0, sha256 `c9cc4c219a0ba0ed125602438d7673d84ece874a59a633cf7f9643277ba0ca6a` |
| cycle | If anything is pending, `lanes.py claim-run --lane L-EXP` on `cycle_cards` `[[5,""],[3,""],[1,"train-idle"],[2,"train-idle"]]`, then `job --ready-file …` under that claim |
| VL model | Qwen3.6-27B Q4_K_M + `mmproj-Qwen3.6-27B-F32-unsloth.gguf` in `/home/cereal/models/judge/qwen3.6-27b/` (SHA256SUMS + `.sha256.ok` stamps) |
| VL server | The job starts its own llama-server from `/home/cereal/omni-campaign/export/llamacpp/bin`: `-ngl 999 -c 12288 -np 1 --jinja`, 127.0.0.1:18460, one-time key file, stopped at job end |
| Prep | `media_judge_prep.py` in `/mnt/lexar/omni-campaign/runtimes/speech-asr` on **`cuda:0` of the claimed P40**. It runs CLAP `laion/larger_clap_general@ada0c23a`, ASR lyric recall (asr_two_judge A+B) and mesh stats (welded components, skirt flare, 4 grey renders). Last run: 93.6 s |
| Batch | `max_groups` 60 per job; a job took 20-31 min on GPU5 |
| Measured VL cost (job 20261028T013505Z) | median latency t2i 17.5 s (n 28), edit 19.6 s (n 21), mesh 14.5 s (n 12), video 12.9 s (n 1); P40 prompt read ~195 tok/s, decode 13.8 tok/s; t2i prompt ≈ 1,415 tokens (one 768-px image ≈ 576 image tokens + rubric) |

**What it judges** (spec `stage_rule`). A generation task is one group, and the group passes only if every member passes.

| Rule | Stages | Judge | Pass thresholds (pre-registered) |
|---|---|---|---|
| image | image_t2i, image_edit_source, mesh_hy3d_source | VL, one image at max side 768 | subject_present, adherence ≥ 7, defect_severity ≤ 3 |
| edit | image_edit | VL on before + after, plus measured high-frequency gain in the unchanged area | edit_applied ≥ 7, preserved ≥ 5, severity ≤ 3, HF gain ≤ 1.35 when the unchanged area is ≥ 10 % of the frame |
| video | video_ltx, h3sol_draft, ltx_refine | VL on **one grid image of 8 time-sampled frames** (4 columns, 384 px wide each, numbered) | subject_present, adherence/action/coherence ≥ 6, severity ≤ 4 |
| music | music_ace | CLAP rank vs 24 genre distractors; ASR lyric recall | rank_frac ≤ 0.25; recall ≥ 0.30 when the lyrics have ≥ 4 content words |
| sound | h3sol_audio | CLAP rank vs 24 sound distractors | rank_frac ≤ 0.25 |
| mesh | mesh_hy3d | VL on the source image + a sheet of 4 grey renders; mesh stats | object/shape ≥ 6, no extraneous geometry, severity ≤ 4, largest component ≥ 90 % of area, skirt flare ≤ 1.25 |

- **Prompt.** A system prompt ("strict quality inspector … JSON only") plus a per-rule rubric with `{prompt}` filled in. Temperature 0, seed 0, max_tokens 700, `enable_thinking: false`, `response_format` json_schema built from `vl.schemas`. An answer that does not end with finish_reason `stop` counts as an error, and the group is held.
- **Outputs.**
  - A `ai_serve.media_judgement` row for every member (scores including the VL caption, reasons, spec sha, judge model name and sha, omni_lab run id).
  - A `<file>.judge.json` sidecar.
  - Rejected groups move to `spool/media/rejected/<modality>/<batch>/` (kept, never deleted). Errors stay in `ready/`.
  - One omni_lab eval run per generator kind per job (`omni_lab.py log-eval`), and a `receipt.json` in `/home/cereal/ai-serve/evidence/media-judge/<utc>/`.
- **How verdicts gate intake.**
  - `media-publish` (`gen_worker_media.py publish`) moves an artifact only when it has `verdict = pass` and `group_verdict = pass` under the current spec sha, and its media set is complete.
  - Publish targets the NAS zone (100 % full, so refused) and overflows to `/mnt/lexar/omni-campaign/corpus-store/chimera-intake/`.
  - RUN0 §2.5 also excludes **media-judge captions** from training targets.
- **Totals so far (10-21 → 10-28).** 10,690 verdicts: t2i 2834 pass / 859 reject; edit 811 / 1133; mesh 473 / 405; music 1208 / 589; h3sol draft 128 / 48, audio 168 / 8, ltx_refine 133 / 43; video_ltx 70 / 50; 6 holds.

## 2. Can Strata see images and video?

**Not today. It can with a projector file plus a launcher change and a gate change.**

**Model directory.** `/home/cereal/models/gguf/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/`
- It holds 4 shards and SHA256SUMS. sha256.check is OK for all shards. There is no mmproj.
- Shard 1 is metadata only: arch `qwen4exp`, 48 blocks, embed 2560, quantized by Unsloth, 1224 tensors over the split.
- The vision tower is not in these shards, by llama.cpp convention.

**Projector files on HF.** Licence for all of them: `qwen-community-1.0`. Internal use is permitted. A separate licence is needed only for a commercial MaaS / AI-work-assistant business offered to third parties.

| File | Repo | Bytes | sha256 | Notes |
|---|---|---|---|---|
| `mmproj-F16.gguf` | `unsloth/Qwen3.8-Flash-Next-GGUF` (rev `766911a6b736`) | 904,004,000 | `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980` | **Recommended.** Same producer as our shards (Unsloth). F16 suits these CPUs: 2× E5-2699 v4, AVX2/FMA/F16C, no AVX-512 and no BF16. |
| `mmproj-BF16.gguf` | same | 907,542,944 | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` | BF16 is slower on CPU here |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` | Header says "GSQ-RCO (mmproj)", quantized by ISTA DASLab. A different tower than Unsloth's, not byte-identical. Abliteration does not touch the vision tower, so there is no reason to prefer it. |

- Both headers agree: `clip.projector_type qwen3vl_merger`, projection_dim 2560 (matches the LM's embedding_length), image_size 768, patch 16, 27 blocks, merge 2, no deepstack layers. Header checks were read via HTTP range.
- Because projection_dim matches the LM, the abliterated LM can take the original projector.
- Still to verify: a load plus a caption test. Nothing has been run with this pair.

**llama.cpp support** (source in `export/llamacpp-xcache/llama.cpp`, b96806d96, 2026-09-02; huihui tested with b11540).
- `qwen3vl_merger` → `PROJECTOR_TYPE_QWEN3VL`, mtmd MROPE positions, `LLM_ARCH_QWEN4EXP` → `LLAMA_ROPE_TYPE_IMROPE`.
- `/v1/chat/completions` accepts `image_url` (data URLs) and `input_video` (mtmd-helper decodes with ffmpeg, temporal merge). `allow_image` / `allow_video` turn on only when an mmproj is loaded.
- With an mmproj loaded, llama-server turns off `ctx_shift` and `cache_reuse`. Strata uses neither: its recurrent layers cannot shift, and the launcher comment says no `--cache-reuse`. `--cache-ram` and `--ctx-checkpoints` are separate paths and are not disabled.
- Untested risk: the xcache scheduler patch (`GGML_SCHED_OFFLOAD_FOLLOW`, `GGML_EXPERT_CACHE_MB`) is global ggml-backend code and also runs for the clip graph. With `--no-mmproj-offload` the clip graph is CPU-only, which should make the patch a no-op. This must be smoke-tested.

**strata-llama unit (live).**
- Base unit, plus `ctx256k.conf`, plus `q4kxl.conf`, giving: `-ngl 999 -sm layer -ts 16,32 -np 6 -c 262144 --kv-unified -cmoe -fa on -ctk/-ctv q8_0 -b/-ub 1024 --cache-ram 49152 --ctx-checkpoints 16 --jinja --numa distribute --no-mmap`.
- Environment: `GGML_EXPERT_CACHE_MB=30000,50000` and `GGML_SCHED_OFFLOAD_FOLLOW=1`.
- `strata_llama.sh` 1.3.0 has no mmproj knob. It passes `$STRATA_EXTRA`, so `--mmproj …` can go there, or into a new `STRATA_MMPROJ` with a sha receipt check to match its model checks.

**Where the projector runs.** VRAM is full by design: GPU0 23.6 of 24.0 GiB used, GPU6 44.8 of 45.0 GiB. The expert cache sizes itself to "free VRAM minus 768 MiB".

| Option | Cost | Notes |
|---|---|---|
| A. CPU: `--mmproj <F16> --no-mmproj-offload` | +~0.9 GB weights + compute buffer, ≈ 1.5-2 GiB host RAM **(projection)** | Simplest; leaves VRAM alone. But RAM is the scarcest resource now (MemAvailable 28 GiB, swap nearly full). Encode on 44 cores / AVX2: a 768-px image is 2,304 patches through ~450M params ≈ 2+ TFLOP, so **~5-15 s per image (projection)**, competing with Strata's `-t 16 -tb 32` threads. |
| B. GPU6: `--mmproj <F16> -mmdev CUDA1` (CUDA1 = GPU6 inside the unit's `CUDA_VISIBLE_DEVICES`) with `GGML_EXPERT_CACHE_MB=30000,48000` | ~2 GiB less expert cache on GPU6 (≈ 4 % of 50 GB); zero host RAM | Encode in about a second **(projection)**. Load order (mtmd init vs expert-cache sizing) is unknown, so the expert cache must be capped explicitly or the projector allocation may OOM. Adds load to the passive RTX 8000; gate thermal rules still apply. |

Recommendation: test B first, because RAM is the binding constraint and B adds no host RAM. Fall back to A only after the RAM floor is fixed. Add `--image-max-tokens 1024` either way so one oversized image cannot use up a slot.

**strata-gate pass-through.**
- `shape_openai_body` → `has_image_parts` → `400 no_vision` (by design, so the phone falls back to its on-device VL).
- Background tier: `/v1` requests must set `stream=true` (`background_must_stream`). `client_max_size` is 32 MiB, which is enough for base64 PNG pairs (≈ 1-3 MB per request).
- Capture: every upstream request is captured with its body (`capture.py`). Base64 image bytes would land in the capture store. A judge source must also be **held from training export**: the judge's captions and verdicts are excluded by §2.5, and the raw capture spool is never train-data.
  - Today `blocked_source_pattern` contains `review|verify|smoke`, but not `judge`.
  - `X-Sprite-Capture: off` works only for the sprite tier.

**How video would be judged.**
- Keep spec continuity: the 8-frame grid sheet is still **one `image_url`**, about 400-600 image tokens at 4×384 px wide. It needs no `input_video` path.
- Native `input_video` (temporal patches of 2, real motion cues) is an option for a later spec version. The gate would need to allow it deliberately; today it neither blocks nor tracks it.
- Audio: the projector has only a vision encoder (`clip.has_vision_encoder`, no audio). Music and sound stay on CLAP + ASR.

## 3. What exactly is needed to move the judge to Strata at background priority

Ownership: steps S1-S5 change strata-llama / strata-gate, so they are **Strata-lane work** (the Sprite lead's Strata sessions).
- Each needs a Relay STATUS on thread `52468f63-…`, a check for PAUSE, a quiet window (no sprite/cyril request for more than 2 min), a backup and a tested rollback.
- Cyril last used Strata at 22:04Z 10-27; the last Sprite request was 10-22 15:51Z.
- A strata-llama restart reloads 104 GiB, about 6 min of outage.

Steps J1-J6 are media-lane work in `ai-serve/media`. ai-serve is not a git repo; it uses dated `.bak` copies.

**Strata lane**
- **S0 (prerequisite, also unblocks gen workers) — re-derive the 40 GiB MemAvailable floors.**
  - Where: strata-gate `--mem-floor-gib`, media-gate `mem_floor_gib`, `lanes.py claim-run`, and `strata_llama.sh MEM_FLOOR_GIB`. The launcher's floor only applies at start, when Strata's own RAM is free again.
  - The floors were set before `-cmoe` + 48 GiB `--cache-ram`. Under the new layout MemAvailable sits at 27-29 GiB, so background has been refused since ~01:42Z.
  - Options: a lower floor sized to real headroom (decide it against swap use: 6 of 7 GiB is in use), or a smaller `--cache-ram`. Without one of these, no background judge or generator runs.
- **S1 — fetch the projector.**
  - File: `unsloth/Qwen3.8-Flash-Next-GGUF@766911a6` `mmproj-F16.gguf` (904,004,000 B, sha above).
  - Destination: Lexar, e.g. `/mnt/lexar/omni-campaign/models/strata-mmproj/`, with `SHA256SUMS`, a sha256.check receipt and a licence note. Use Lexar because `/` has 113 GB free, already under the 150 GB run-0 target. Do not use the NAS: it is 100 % full.
  - Size: 0.9 GB with `par_range_dl.py`, which already exists.
- **S2 — launcher.**
  - `strata_llama.sh` 1.4.0: `STRATA_MMPROJ` (path + size + sha receipt check, refuse without it) and `STRATA_MMPROJ_DEV` (`none` = `--no-mmproj-offload`, or `CUDA1`).
  - New drop-in `vision.conf`, sorted after `q4kxl.conf`, so its `STRATA_EXTRA` must repeat the q4kxl flags. Settings: option B → `GGML_EXPERT_CACHE_MB=30000,48000`, `--image-max-tokens 1024`.
  - Rollback: delete `vision.conf`, run `daemon-reload`, restart. Restart strata-gate after the swap (capture re-reads `/props`).
- **S3 — smoke in the quiet window.**
  - `/props` must show `modalities.vision: true`.
  - One caption of a known image and one 8-frame sheet, sent direct to `:18401` with the internal key.
  - Watch: Xid / illegal-access lines in the journal (the xcache build had an IQ3_S Xid-31 history), VRAM on GPU0/6, MemAvailable, and text-only bench parity (`bench_strata.py` write/read tok/s against the 24.1 / 258 baseline).
- **S4 — strata-gate 1.3.0.**
  - Allow image parts for the `background` (and `cyril`) tiers. Keep `400 no_vision` for `sprite` unless the Sprite lead opts in, because it changes the phone's VL fallback.
  - Handle `input_video` the same way (block or allow, not neither). Set `/v1/models` `"vision": true` for the tiers allowed.
  - Capture: store image parts as `{sha256, bytes, mime}` instead of base64. Hold source `media-judge` from export: add `judge` to `blocked_source_pattern`, or add an explicit rule.
  - Test via the unit tests in `strata/tests`. Roll back with the `.bak` copy of `strata_gate.py`.
- **S5 — announce the new teacher identity on Relay** (Strata + mmproj sha), as was done for the Q4 switch.

**Media lane (judge)**
- **J1 — `media_judge.py` 1.3.0: a `StrataJudge` client in place of `VLServer`.**
  - Request: `POST http://127.0.0.1:18400/v1/chat/completions`, `Authorization: Bearer $STRATA_GATE_BACKGROUND_KEY` (already in `media.env`), model `strata` (the gate injects thinking-off), `stream: true` (required for background).
  - Reassemble the SSE deltas and take `finish_reason` from the last chunk. Keep `response_format` json_schema; whether llama-server applies the grammar under streaming must be verified in S3.
  - Headers: `X-Client-Source: media-judge`, `X-Workflow-Id: media-judge-<stamp>`.
  - Treat `503 preempted / busy / thermal` as "retry later": the group stays in `ready/` and gets no verdict row, so it never becomes a permanent hold. Concurrency 1-2: `bg_max` 4 is shared with the gen workers.
- **J2 — `cycle` without a P40.**
  - Drop `cycle_cards` / `lanes.py claim-run`. Check `GET /gate/status` → `background_admission.allowed` before starting.
  - Keep the flock. Since a preemptible judge costs nothing when idle, the timer can move from hourly to every 10-15 min.
  - Disable the old timer's GPU path outright, so it can never claim GPU5 again (D2: GPU5 goes to run 0).
- **J3 — prep (CLAP / ASR / mesh) loses its GPU.**
  - Today it runs on `cuda:0` of the claimed P40. Without a P40 it falls back to CPU (`device = "cpu"` path exists). CLAP-general, two ASR judges and torch would add several GiB of host RAM, which conflicts with the "stay small in RAM" rule.
  - Options: (a) CPU with a hard RSS cap (systemd `MemoryMax=`, e.g. 6 GiB) and one model at a time, run only when the floor allows; (b) an ASR/CLAP job type behind media-gate (GPU4), which needs a media-gate change; (c) defer music/sound judging until run 0 frees a card.
  - Music (533 groups) and H3 audio depend on this choice. Images, edits, video and meshes do not.
- **J4 — spec 1.3.0, a new registration.**
  - Add `judge_model` (Strata Q4 abliterated SHA256SUMS hash + mmproj sha) to the spec, so the judge identity is pre-registered rather than only recorded per verdict. Rubrics, thresholds, frames and image sizes stay unchanged.
  - Effect: the 2,341 pending groups are judged under 1.3.0. Verdicts already published under 1.2.0 stay published, since publish only gates what is still in `ready/`.
  - Update the docstring and README lines saying "Strata is text-only".
- **J5 — calibration before cut-over.**
  - Method: `smoke`-style, no verdict writes, on a stratified sample of items already judged by Qwen3.6-27B, e.g. 50 each of image/edit/video/mesh with both passes and rejects. Include the known bad cases: the LTX boat clip, over-sharpened edits, skirted meshes.
  - Report agreement and kappa per rule plus caption spot checks. Log it in omni_lab (`omni_lab.py log-eval`, kind review).
  - Gate the switch on no lenient drift on known rejects. Suggested threshold: ≥ 85 % agreement, with Cyril/lead to confirm.
  - Two risks to cover: (i) self-preference — Strata wrote many of the prompts (`prompt_source: strata`) and would now judge their renders; (ii) the abliterated build may be more lenient. Calibration covers both.
- **J6 — throughput (projection).**
  - Per VL member ≈ image encode (option B ~1 s, A ~5-15 s), plus ~600-1,400 prompt tokens at ~230-258 tok/s, plus ~150-250 JSON tokens at ~14-24 tok/s background → **~10-20 s**. That is about the P40's 17.5 s, but preemptible and with no GPU5 claim.
  - Backlog: ~2,300 VL members ≈ 7-13 h of free background time at one stream.

**Ordering:** S0 → S1 → S2/S3 (quiet window) → S4 → J1/J2/J4 → J5 (calibration) → cut-over → J3 (audio prep) → retire the Qwen3.6-27B judge path. Until cut-over, run 0 keeps GPU5 and the timer stays paused (D2a), so no judged media reaches intake in that interval.

## Sources read
- Server: `/etc/systemd/system/media-judge.{timer,service}`; journal; `ai-serve/media/{media_judge.py, media_judge_spec.json, README.md, media_gate.py}`; evidence receipts 20261027T233504Z..20261028T013505Z; `media_judge.py pending`; `ai_serve.media_judgement` / `request_log` (read-only SELECTs).
- Server: `strata-llama.service` + drop-ins, `/proc/<pid>/cmdline,status`; `ai-serve/strata/{strata_gate.py 1.2.3, strata_llama.sh 1.3.0, capture.py, capture_policy.json}`; `/gate/status`; `nvidia-smi`; `free`; `lscpu`; `df`.
- Server: the Strata GGUF shard-1 metadata (gguf-py); llama.cpp source (`tools/mtmd`, `tools/server`, `src/llama-model.cpp`); `llama-server --help`.
- HF API / raw: `unsloth/Qwen3.8-Flash-Next-GGUF` and `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` trees, READMEs and mmproj headers (range reads); `Qwen/Qwen3.8-Flash-Next` config, preprocessor configs and LICENSE.
- Local: RUN0_BUILD_PLAN.md (§0 D2, §2.5, §6), STAGE_D_RULINGS (cyril-run0-decisions-2026-10-22), memory `strata-q4-switch-2026-10-27.md`.
