# Chimera Omni state, 2026-10-18 about 04:00Z: full-parameter KD launched (Claude session 753087a5)

**Read this first if you are picking up Chimera Omni.**

**The plan of record:**
- `REALIGNMENT_2026-10-17.md` (this folder), sections 1-7.
- The rulings in `distill/STAGE_D_RULINGS_2026-10-12.md`, under "Cyril's words and approved realignment, 2026-10-17".

**Memory:** `omni-full-distill-k3-primary`, `omni-corpus-tokenizer-agnostic`, `omni-avoid-waste-use-what-exists`, `omni-state-2026-10-17`.

## 1. What Cyril decided on 10-17 (his words, in order)

1. "[Cyril quote withheld]". Then: "[Cyril quote withheld]". Then: "[Cyril quote withheld]".
2. He approved the realignment: full-parameter [Cyril quote withheld]. Drop selfgen, tool-selfgen and refill-codex. K3 passes go on the 3090s until the trainer is ready.
3. "[Cyril quote withheld]", then "[Cyril quote withheld]".
   - Claude refused to train on the ~880k Claude/Codex tool calls in memory. Anthropic's and OpenAI's terms forbid training another model on their outputs, internal use included. Cyril's own prompts and the real tool results are fine.
4. "[Cyril quote withheld]".
5. "[Cyril quote withheld]".
6. "[Cyril quote withheld]".
7. "[Cyril quote withheld]".

## 2. Jobs running (server 192.168.50.222)

| Job | Card | Owner / entry | Status file / log | Stop |
|---|---|---|---|---|
| **Full-parameter KD run 1** | GPU4 (3090) | `$T/fullkd_queue.sh` (started 03:50:16Z) → `$T/streamed_full_kd_trainer.py` 1.0.0 train | `/mnt/lexar/omni-campaign/distill/fullkd/run1/{queue.status,queue.log,train-NNN.log,receipt-NNN.json}` | `touch run1/STOP` (between segments) or `kill -TERM <trainer pid>` (resumes from the last checkpoint) |
| **K3 teacher passes, provisional bf16** | GPU6 (RTX 8000) | `$Q/gpu6.sh` (`queue_gpu6.sh` 1.5.0). Pass 0 SEALED; pass 1 started 00:36:57Z (about 7.75 h per pass) | `$Q/gpu6.status`, `$Q/logs/gpu6/k3-pass-N.log` | as `queue_gpu6` documents |
| **K3 provisional xtok** | CPU | `$T/k3_prov_xtok.sh` (polls) | `$Q/logs/k3_prov_xtok.log`. Pass 0 READY: 467,231 pairs, xtok `fab52c57…` | kill |
| **Flash Next Strata IQ3_S download** | network → NAS | `/mnt/ai_data/models/flash-next-strata/download_strata_iq3s.sh` (resumed 03:52Z from 40.7 GB) | `.../download.log`; sha256 is checked at the end | kill |

**Stopped on purpose (do not resume):**
- LoRA D2 (`d2_launch.sh` 1.3.2).
- Qwen 3.8 27B tool-call generation (`run_toolcall_gen3_2026-10-17.sh`).
- The GPU0 K3 runner (`k3_multicard.sh` 1.1.0), stopped 03:50Z. The trainer is ready, and host RAM fits only one of {GPU0 K3 pass, trainer} beside GPU6. Its card-class eqcheck PASSED (`$Q/k3_cardclass_provisional/nvidia-geforce-rtx-3090-sm86/verdict.json`), so GPU0 can take K3 passes again whenever the trainer is not running.

**Idle:** GPU0 (see above) and the P40s GPU1/2/3/5, which wait for Flash Next (§4).

## 3. The trainer (`streamed_full_kd_trainer.py` 1.0.0, mirrored in `distill/tools/` here)

- **What it trains:** every text weight of OMNI1D1. That is P1 over `OC02-F-…OMNI1.clone-pre-D1` plus the D1 delta plan `$LEX/distill/d2/p1_plan.json`: 851 tensors, 47 GB bf16 master in host RAM.
- **How:** the streamed 1.3.4 engine. Per-layer fp32 gradient accumulation; a fused Adafactor update (no momentum, factored, RMS clip 1.0, decay -0.8, stochastic rounding to bf16); the head and norm after the head phase; sparse-row embedding updates.
- **Proof:**
  - CPU tests: 5/5.
  - `grad-parity` on Qwen3.5-0.8B fp32, all 320 tensors: worst 1.0e-7 (whole-layer) and 2.6e-5 (chunked).
  - Real-geometry probe: 17.3 GiB peak.
  - Real OMNI1D1 smoke: step 1 loss 0.363, checkpoint, resume, step 2 loss 0.934, 0 non-finite.
  - Receipts are in `$LEX/distill/fullkd/{parity,probe,smoke-20261017}`. The smoke checkpoints were deleted; logs and receipts are kept.
- **Run 1 settings:**
  - Data: `fullkd_data_v2.json`. That is the D2 registry text + `tool-bfcl-native`, with selfgen, tool-selfgen and refill-codex dropped.
  - K3 rows come from the provisional passes as they become READY: pass 0 at 0, 1 at 3M, 2 at 6.5M, 3 at 10M tokens, each at fixed fraction 0.07.
  - Objective weights: K3 0.8 and Q38 0.2 on K3 rows; Q38 0.25 on text rows; CE 0.1.
  - micro 8 × 4,096 (max width 5,282), 20M tokens (611 steps), lr 1e-5 with warmup 20, one rolling checkpoint every 60 min.
- **Rate (EST from the probe):** about 85-90 tok/s, so 20M tokens is about 2.7 days.
- **Exits:** 0 means DONE and the student is exported to `run1/out/student/` (qwen3_5-layout text model). If a K3 pass is not READY at its threshold, the trainer checkpoints and exits 4, and the queue waits for the marker.

## 4. Next (in order)

1. **Flash Next Strata to root.** After the download is verified (NAS):
   - move `omni-campaign/qwen-fold/children` (140 GB, cold since 09-03) to `/mnt/ai_data/backups/ai-server-cleanup-20261017/` with a symlink back, using `/home/cereal/cleanup-20261017/move_qwen_fold_children.sh`, which is verified byte-equal before the source goes;
   - then copy the GGUF to `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/` and remove the NAS duplicate.
   - **The NAS (CIFS) hangs under two concurrent big writers** (dmesg 10-17 18:20-18:39 local). Run ONE heavy NAS job at a time and use `--bwlimit`.
2. **Tool-call dataset.** Run `$T/flash_toolcall_gen.sh`:
   - one llama-server split over GPU1/2/3/5, 4 slots;
   - Flash Next writes the schemas, the plan (3,000 items) and the run (3M tokens);
   - finalize.
   - Then export → Q38 cache (P40s) → K3 passes over it (GPU6/GPU0) → add it as a source to the next data config.
   - Flash Next's tokenizer equals Q38's, so no translation step is needed.
3. **Flash Next as a teacher (logits)** next to K3: needs a teacher-forced scoring tool over llama.cpp (prompt logprobs), since the server does not return them.
4. **Media KD (D3) full-parameter.** H3 (fix the items-schema adapter first: `h3-encode` failed on `items.jsonl` with no top-level `id`), ACE-Step, Qwen-Image/Edit, STT/TTS (E2/E3 tools to build), 3D, VL.
5. **Tokenizer-agnostic canonical corpus.** Recover the source text of the registry into text/structured records. Every token view and teacher cache becomes derived per student tokenizer.
6. **Memory tool-call layer** (search/audit only): a typed `agent_memory_tool_calls` table and a turn view. It is NOT training data.
7. **Trainer speed-ups** for v1.1: pinned master, a second replica (shared host master), and a leaner optimizer pass.

## 4b. Update 2026-10-18 about 04:20Z

**Run 1 restart.** Segment 0 was REFUSED after loading at 04:04Z: `k3-pass-0` was declared `defer_until_tokens: 0`, and the trainer requires > 0 or absent. Pass 0 is READY, so `fullkd_data_v3.json` (sha `65849268…`) makes it a regular source. Passes 1-3 stay deferred at 3M/6.5M/10M. `fullkd_queue.sh` now defaults to v3. Segment 1 started 04:05:51Z (`run1/train-001.log`).

**Flash Next teacher logits: built and proven.**
- `distill/tools/teacher_score.cpp` 1.0.0 is a teacher-forced top-K scorer over the export lane's `libllama` (b96806d9). The binary is `$OC/export/llamacpp/bin/teacher-score`, with a receipt beside it; build it with `build_teacher_score.sh`.
  - It speaks a binary pipe protocol: lse in double; top-64 sorted by logit, ties to the lower id.
  - Each row is decoded in 1,024-token chunks on one sequence.
- `distill/tools/gguf_teacher_cache.py` 1.0.0 writes what the scorer returns as a qtext-format cache. It imports `qtext_teacher_cache`'s own shard, lock, session, digest and seal functions (nothing copied).
  - It needs `LD_LIBRARY_PATH` set to the lane's bin, because the `libggml*` RUNPATH does not reach `libggml-cpu`/`libggml-cuda`. The tool sets it.
  - The trainer's `QTextCache` reads these caches unchanged, including `verify_raw`.
- **Parity:** Qwen3.5-0.8B, f32 GGUF vs qtext HF fp32 (the production reference), 8 rows, 3,188 positions, store mask included. Positions are identical. Receipts are in `/mnt/lexar/omni-campaign/distill/fullkd/scorer-parity/compare-*.json`.

| arm | top-1 | top-64 overlap | \|dlse\| max | KL mean | KL p99 |
|---|---|---|---|---|---|
| CPU | 100 % | 99.97 % | 0.057 | 2.4e-4 | 1.2e-2 |
| P40 (GPU1) | 100 % | 99.98 % | 0.0026 | 1.8e-4 | 2.5e-4 |

The largest shared-id gap, 0.125, is one bf16 step at logprob ~ -20.

**Flash Next lane, unattended.** `$OC/distill/flash_lane.log` runs three steps in order:
1. `flash_stage_chain.sh`:
   - waits for the download, which `download_strata_iq3s.sh` sha-checks before renaming the `.part`;
   - moves `qwen-fold/children` (140 GB) to the NAS with a bandwidth cap and six retries, a size/list check and a checksum dry run, then symlinks it;
   - copies the GGUF to `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/` and checks the sha there. The NAS copy is KEPT.
   - Log: `/home/cereal/cleanup-20261017/flash_stage_chain.log`.
2. `flash_toolcall_gen.sh` (GPU1/2/3/5): Flash Next writes the tool-call dataset.
3. `flash_score_queue.sh` (GPU1/2/3/5, locks + `p40.claims`): Flash Next logits for three groups, one scorer load each:
   - G1: K3 pass corpora + `tool-bfcl-native`, about 2.4M tokens;
   - G2: `text_clean_v1`, about 12M;
   - G3: registry text, about 20M.
   - Caches go to the ROOT disk at `$OC/distill/teacher_cache/flash-next-strata-iq3s/`, because Lexar must keep 200 GB free for K3.

**Run 1 is training.** Step 1 at 04:23Z:
- 285.6 s for 32,768 tokens, which is **114.7 tok/s** (better than the 85-90 estimate). At that rate 20M tokens takes about 2.0 days.
- loss 0.415 (ce 0.180, q38 0.232); all 851 tensors updated, 0 non-finite, 0 without a gradient; GPU4 peak 19 GiB.
- Host MemAvailable is 71 GiB.

**Lexar relief.** `lexar_relief.sh` runs after `flash_stage_chain.sh`, because there is one NAS writer at a time. Log: `/home/cereal/cleanup-20261017/lexar_relief.log`.
- It moves the D1 transient scratch to `/mnt/ai_data/backups/lexar-relief-20261018/`: `scratch/d1_boundary` (77 GB) and `scratch/d1seal` (52 GB). Both are from 10-15, and nothing reads them; `p1_plan.json` reads only `clones/`.
- It uses `verified_move.sh` (rsync, list check, checksum dry run, symlink back).
- Why: Lexar has 256 GB free. The trainer's 47 GB checkpoint leaves about 209 GB, and `queue_gpu6` WAITS for ≥ 200 GB before each K3 pass (pass 2 at about 08:20Z).

**How Flash Next enters training.** Each sealed Flash Next cache is an ordinary `text` source over the same corpus as its Q38 cache, since the tokenizers are identical. The trainer applies `w_q38` to it, the "same-tokenizer teacher" weight.
- Adding sources to run 1 means a resume under `--allow-config-change --allow-identity-change`. The change is recorded in every later receipt, and the sampler keys sources by name, so a new source starts at 0.
- Trainer v1.1 should take two same-tokenizer teachers on ONE row (a Q38 + Flash Next mix) rather than alternating rows.

## 4c. Update 2026-10-18 about 06:30Z: media distillation (full-parameter) and the two-teacher mixture

**Media stage M1 is TRAINING on GPU0.** It started 06:17:49Z. Queue `$T/media_m1_queue.sh` (MODE=train), trainer `$T/media_full_kd_trainer.py` 1.1.0, logs in `/mnt/lexar/omni-campaign/distill/media_m1/run1/`.
- **What trains:** every weight of the five charted units (ACE DiT L12, QIMG MMDiT L30, HY3D DiT L10, H3 DiT L25, QVL32 ViT L13) plus the native media flow head. That is 622,973,233 parameters in 110 fp32 tensors, under the full trainer's streamed Adafactor.
- **What does not:** the text trunk is frozen at OMNI1D1 (clone-pre-D1 + p1_plan), because run 1 owns it. The media tensors are disjoint from run 1's 851 text tensors, so the two results join exactly.
- **The step:** D3b guided-velocity KD (x0 head) and D3e speech. It runs layer-major over groups of up to 30k tokens, so each trunk layer streams once per group. Boundaries stay on the GPU (`--offload cuda`, no host RAM). The text head is on the GPU for speech-in CE.
- **Proofs:**
  - `selftest` on a tiny Chimera child: group gradients are bit-identical (rel 0.0) to `d3_hybrid_module.d3b_forward_backward` per sample, over all 67 tensors.
  - Speech-out and speech-in are bit-identical to `d3.speech_lane_step`.
  - Loss falls 7.93 → 0.110.
  - Real-weights probe (lr 0, 3 groups): PASS, peak 5.5 GiB, RSS ≤ 16 GB, finite gradients on every route.
- **Data:** `media_data_v2.json`, Codex's sealed native batches, every tensor file sha-checked against its receipt. 16 held items per route are never trained.

| route | batches | fraction |
|---|---|---|
| music (ACE-Step 1.5, `native-ace-full-v5`) | 2,764 | 0.30 |
| image (Qwen-Image, `native-qimg-full-v3`) | 1,126 | 0.30 |
| joint AV (H3 parity set) | 32 | 0.05 |
| speech out (`native-speech-flow-v2`) | 744 | 0.20 |
| speech in (`native-speech-ce-v2` + `native-audio-ce-v2`) | 380 | 0.15 |

- **Speech provenance:** Cyril, 10-18: the speech data is HIS ("operator_authored" = Cyril). Memory `omni-speech-data-is-cyrils`.
- **Run settings:** 300 groups, lr 2e-5, 30-minute checkpoints (keep 2). The export is `out/export/media_modules.safetensors` in Chimera names.
- **Stop:** `touch run1/STOP`.

**Text teachers: Q38 + Flash Next on one row, with no trainer change.**
- `mix_teacher_cache.py` 1.0.0 writes a qtext-format cache of p̄ = 0.25·p_Q38 + 0.75·p_Flash on the union of both top-64 sets, cut to 64.
  - The cross-entropy term of a KL is linear in the teacher, so Σ wₜ∇KL(pₜ‖s) = (Σw)∇KL(p̄‖s) exactly. Only the top-64 truncation approximates.
  - `check` on the parity caches: 400 positions, worst 0.55 bf16 ulp, mass ≤ 1. The trainer's `QTextCache` and `verify_raw` read it.
- `fullkd_data_flash.py` has three commands:
  - `mix-ready` builds each mixture as its Flash Next cache seals;
  - `config` writes the next data config, with sealed mixtures replacing Q38 caches;
  - `status`.
- `flash_mix_queue.sh` polls `mix-ready` (log `$OC/distill/flash_score/mix_queue.log`).
- Adopting the mixtures in run 1 is a planned resume: `fullkd_data_flash.py config --out v4.json`, then TERM the trainer and restart `fullkd_queue.sh` with DATA=v4 plus `--allow-config-change --allow-identity-change`.

**Still owed on the media side:**
- The H3 production cache. `h3-encode` needs an ided items file, then h3-text and h3-velocity in the ComfyUI r5 image, then Codex's `h3_native` conversion. It needs a GPU, so it is queued behind M1 on GPU0.
- The HY3D geometry route (`recovery_geometry_flow_cache.v1`, 32 receipts, image-conditioned).
- D3d vision-in (QVL32 ViT + `qtext_vl` answers).
- D3c music band rows (ACE LM codes).
- Stage **M2**: after run 1 exports, every weight (trunk + media) trains on a text + media mixture from the merged student.

## 4d. Update 2026-10-18 about 07:00Z: M2 (every weight jointly), D3c music codes, reassembly

**M1 progress.** Steps 1-5 ran at 82-106 tok/s with 0 non-finite gradients. The music loss went 9.98 → 9.42 → 9.79 and the image loss 72.8 → 20.2.

**`media_full_kd_trainer.py` 1.2.0 adds D3c (music codes).**
- Route `music_codes`, kind `ace_codes`, over the ACE-Step 1.5 LM codes cache (6,667 items).
- The prompt is rendered exactly as Codex's ACE batches do it (checked token for token, 361 tokens equal), followed by the codes as C18 band ids.
- The teacher is ACE LM's top-64, renormalised onto the 64,000 written code rows.
- It trains the written code rows of the head (dense) and of the embedding (sparse), as fp32 masters. The ledger and reserved rows are digest-checked after every update.
- These groups run full depth (the gradient goes back to the input), with a per-route `group_tokens` of 12k.
- Selftest: D3c parity against d3's `full_depth_forward_backward` — head rel 0.0, embed rows 3.4e-8.
- Also new: a checkpoint on STOP, and resume keeps the base values for trainables the checkpoint lacks.
- `media_m1_switch_v3.sh` moves the running M1 to 1.2.0 + `media_data_v3.json` (music_codes at 0.15) at its first checkpoint.

**Stage M2 is built: `omni_full_kd_trainer.py` 1.1.0.**
- **What trains, together:** every weight. That is all text weights (one host master, `FullParamEngine`), the units, the bridge, the flow head and the music band.
- **The steps:** text steps (run 1's step) interleave with media groups by token share. A media group runs full depth: one fused update per trunk layer from fp32 gradients summed over the group (`OmniEngine.layer_backward_multi`), the units walked back between layers d and d-1, and sparse rows for the prompt embedding and the bridge.
- **Selftest (tiny child, CPU fp32), against ONE plain autograd graph with resident layers:**
  - media step: every gradient matches, worst 8.6e-9 over 131 tensors (all 53 trunk tensors, the embedding rows, units, bridge, head);
  - D3c inside the media step: exact (0.0) for the head rows, the embedding rows and the trunk;
  - joint training: media 6.25 → 0.24 and text 3.77 → 0.14.
- **Running it** needs run 1's final weights (`--init-text`) and M1's export (`--init-media`). It cannot be smoked on real weights until run 1 frees host RAM and a GPU.

**`chimera_reassemble.py` 1.0.0 writes ONE Chimera checkpoint from the trained parts.**
- Text tensors go back through `p1_qwen35_reader.importer_forward`, the importer's own inverse.
- A norm stored as 1+w is widened to fp32 when bf16 is inexact.
- Media tensors keep their Chimera names; music rows go into the C18 band of embed and lm_head; everything else stays the base value.
- `verify` re-reads the result with the P1 reader. Selftest on the tiny child: PASS.
- **First full-model artifact** (after run 1 + M1): `chimera_reassemble.py build --base clone-pre-D1 --plan p1_plan.json --text run1/out/student --media media_m1/run1/out/export/media_modules.safetensors --out ...`, then `verify`.

**Media routes still owed:**
- H3 production: the bf16 DiT needs GPU6 + a thermal ruling, so it will use the int8 DiT on a 3090 after M1, plus Codex's `h3_native` conversion. `items.ided.v1.jsonl` (512 rows) is built.
- Image edit: 32 `image_edit_cache.v1` batches, extracted to `media_m1/codex-image-edit-v1-extracted/`; they need a binder.
- HY3D geometry: 32 batches; they need a binder.
- D3d vision-in.

## 4e. Update 2026-10-18 about 11:30Z: the NAS is full and hung; what that blocks

**The NAS is the blocker.** The media server (192.168.50.250, smbd) re-exports an NFS pool from a VM, `192.168.122.170:/mnt/ai_pool/ai_dataset`.
- The pool is 4.5 TB and **98% full (117 GB free)**.
- Writes stall: an smbd worker sat in D-state holding 12 GB, and the box's load average is about 107.
- The CIFS session on the AI server died ("Host is down"); a remount now gets "Server abruptly closed the connection".
- On the media server itself, `ls` on its own NFS mount does not return.
- Claude did not touch the NAS VM, smbd or anything else on the media server (Plex and game servers run there).
- **Consequences:**
  - Moving cold data to the NAS (`qwen-fold/children` 140 GB, D1 scratch 129 GB) can never fit. Both moves are stopped; their sources are intact. A 42 GB partial copy of the children is on the NAS.
  - **The Flash Next GGUF's only copy is on the NAS** (verified sha `cb5958e5…`). It cannot be read now.
  - The root disk has 24 GB free. The memory-backup job is staging 26 GB in `/var/tmp/agent-memory-backups` for the NAS, which is not ours to touch.

**Done in this window (05:30-11:30Z):**
- Run 1 reached 3M tokens at step 94 (145 tok/s). K3 pass 1 sealed; its xtok was READY at 10:10Z, so the deferred source loads with no stop.
- K3 pass 2 started 10:09:51Z on GPU6, once Lexar was ≥ 200 GB.
- Lexar relief used local verified moves to root, because the NAS refuses writes:
  - `k3-fold-sep07-codex` (12 GB) and `distill/fullkd/scorer-parity` (3 GB) now live in `/home/cereal/lexar-relief-local/`.
  - Lexar is **exFAT, so it has no symlinks**. The old paths hold a `.MOVED.json` note instead; `verified_move.sh` now does this automatically.
- M1 media:
  - It died twice on bad ACE codes items: 07:11Z on an early EOS, and 07:34Z in the held eval, on an item with no codes before EOS.
  - 1.3.1 truncates at the first non-code token and skips unbindable items with a cap. 1.3.2 also skips them in the held eval.
  - It restarted 11:13Z on `media_data_v4.json`, 7 routes with vision included, from step 6. GPU0 sat idle 07:34-11:13Z.
  - 1.3.x also adds D3d vision-in (`VisionIndex` / `bind_vision`; 3,684 rows; selftest parity exact; a real-data `vision-check` PASS), a held-item eval every 25 groups (forward-only; parity exact), and per-route group token caps.
- H3 production capture: started on P40s, then stopped. The 2-second clips are shorter than the default window, and the corpus pairs **unrelated** LibriSpeech audio with silent generated video. The native joint-AV binder requires same-origin audio and video (Codex's contract). The H3 production cache therefore needs a real AV corpus first.

**11:40Z, M1 1.3.3: the ACE music-codes binder was wrong. This corrects the "early EOS" note above.**
- The ACE LM cache (`media_b_teacher_cache` ace_lm_codes, 6,667 shards) stores band-relative code indices in [0, 64000) for both the sequence and the top-64 ids. The EOS step is dropped, and min_tokens is 300, so no shard hits EOS.
- 1.2.0-1.3.2 compared those indices with the absolute band [151669, 215669) and refused every real item. So music_codes never trained, and M1 would have stopped at its first music_codes group (the 50-skip cap).
- 1.3.3 reads code indices (an absolute copy is shifted; anything else is refused). It also scales back any row whose top-k mass exceeds 1. That only happens from bf16 storage on rows where the conditional branch puts its mass on EOS (log p ≈ −32), 1-25 rows per shard.
- Checked on 8 real shards, including the held items M1 had skipped:
  - sequence round-trip exact;
  - the prompt decodes the caption;
  - positions, ids and finiteness all OK.
  Receipt: `/home/cereal/cleanup-20261017/ace_bind_check.out`.
- Selftest PASS, all parities exact (`media_m1/selftest-1.3.3.json`).
- M1 restarted on 1.3.3 at 11:41Z (log `train-005.log`), from the step-6 checkpoint.
- Also in this window: `flash_score_queue.sh` now takes GGUF, OK_MARK and OUT overrides. Caches default to Lexar beside the Q38 caches, and a group starts only with OUT_MIN_GB (215) free. When no OK_MARK exists, it hashes the GGUF itself. `fullkd_data_flash.py` roots default to Lexar too.

## 4f. Update 2026-10-18 about 17:00Z: drive cleanup, the throughput fixes, and the M1 OOM

**Cyril:** "[Cyril quote withheld]", with Flash Next to come after the drives are clean and training is running fully. Three agents did the cleanup, and each deletion was logged before it ran (`/home/cereal/cleanup-20261018/*-deleted.tsv`, with file lists under `lists/`).
- **Lexar: 192 → 339 GB free** (150.5 GB freed):
  - D1 transient scratch: `d1_boundary` 82 GB and `d1seal` 55 GB. D1 is sealed and reviewed; the registered delta and both clones stay.
  - Superseded LoRA D2 checkpoints, 6 GB.
  - 53 D1 rung files, 7 GB, sha-identical to `OMNI1D1-rungs-canonical`, which stays.
- **Root: 22 → 174 GB free** (about 163 GB freed):
  - unproven NF a0028 2× trace caches, 130 GB (frozen review 0/15; under Cyril's 08-19 rule, delete the unproven and keep the evidence);
  - the superseded C10 rehearsal partial child, 14 GB;
  - pytest temp files, 9.5 GB;
  - the unreferenced SonicMaster HF dataset, 8 GB;
  - journal, pip and apt caches.
- **Category B, Cyril's call; nothing deleted:**
  - On Lexar:
    - `omnicoder_0.1.0`, 693 GiB. About 170 GiB of it is re-downloadable caches and venvs; its `data/` holds LibriSpeech and LJSpeech.
    - Possible K3 fold duplicates, oct06/oct07, about 230 GiB; not hash-verified.
    - Codex's `native_executor/qimg_first32`, 122 GiB.
  - On root:
    - `qwen-fold/children`, 140 GB: registered `partially_capable` children; CHECKPOINT_POLICY R4/R5 apply.
    - About 28 GB of exited gauntlet/hunyuan containers.
- Stopped the idle `comfyui_qwen21_gpu5` (0 prompts in 72 h; the policy is stopped by default). That freed 6.4 GB of swap; restart it with `docker start comfyui_qwen21_gpu5`.
- **The NAS is still being repaired** (agent working); see §4e for the failure.

**M1 OOM, 15:19:50Z.** The first `vision_in` group ran at the default 30,000 tokens. vision_in is full-depth (all 64 layers of boundaries) like music_codes, and it overflowed the 24 GB 3090 in the text-head objective.
- `media_m1/media_data_v5.json` caps vision_in at `group_tokens` 10000.
- M1 restarted 16:48:47Z from step 33. GPU0 was idle 15:20-16:49.

**Throughput (py-spy on the live jobs).**
- **M1:**
  - 65% of each group was the P1 reader composing the frozen D1 overlay on the CPU; 40% was a 54 GB `isfinite` scan repeated every group.
  - **`p1_qwen35_reader` 1.0.6** checks finiteness once per tensor per reader. The inputs are sha-verified and the compose is deterministic, so the values are unchanged.
  - Two new tests were added; 1.0.5 fails exactly those two and 1.0.6 passes all 38.
  - `media_m1_queue.sh` now defaults to `OMP_NUM_THREADS` 32 (it was 8, on 88 cores).
- **Run 1:**
  - 46% of each step was the pageable host→device layer upload, 27% the pageable write-back, and 17% backward compute.
  - **`streamed_full_kd_trainer` 1.0.1** copies the update straight into the host master (same bits); its tests pass, 5 of 5.
  - `fullkd_queue.sh` takes `EXTRA`.
  - `run1_pin_restart.sh` waits for the next safety checkpoint, then STOPs and relaunches with `EXTRA="--pin-weights --allow-config-change"`. `--pin-weights` is in the config digest; base and data identity stay guarded.
- **Watch item:** at attempts 159-161, as K3 pass-2 rows joined (about 7,500 K3 positions per step), grad_norm reached 6.05, 679 tensors were clipped and the loss went 0.25 → 0.71. It recovered at 162.
- **Not ours, reported:** `clamd-mailcow` has restarted 175,730 times (exit 0, months-long) and costs one or two cores.

## 4g. Update 2026-10-18 about 21:30Z: what the fixes did, run 1's 3.5-hour stop, the NAS root cause, and Flash Next staging

- **M1 is about 3× faster.**
  - Layer load per group dropped from 300-400 s to 72-102 s, and groups take 95-135 s (from 270-420 s).
  - That came from reader 1.0.6, 32 OMP threads, and the GPU5 ComfyUI stop that freed swap.
  - M1 went from step 34 (16:58Z) to step 135 (21:10Z).
  - vision_in groups at `group_tokens` 10000 run without OOM.
- **Run 1 was down 17:55-21:21Z.**
  - The pinned restart (`ops-20261018/run1_pin_restart.sh`) worked as designed: checkpoint 163, then STOP, then trainer 1.0.1 with `--pin-weights`. The pinned 52 GB master loaded in 271 s.
  - The resume was then REFUSED: the base identity includes the p1 reader's sha and version, and reader 1.0.6 changed them.
  - `ops-20261018/base_identity_live.py` diffed the live identity against the checkpoint's stored one. The ONLY differences are `/p1_tool/sha256`, `/p1_tool/version` and `/summary/tool_version`. The clone, config, shards, parent config, D1 plan sha and root digest are all identical.
  - Relaunched 21:21Z with `--pin-weights --allow-config-change --allow-identity-change`; the record is `run1/IDENTITY_OVERRIDE_2026-10-18.json`.
  - **Lesson:** a reader change is a base-identity change for every job that streams through it; check the live jobs' resume guards BEFORE installing a shared module.
- **The NAS share is healthy again** (agent report, 21:15Z). Nothing was restarted or rebooted.
  - Root cause: the AI pool absorbs only about 5 MB/s of sustained writes. It sits on SMR ST5000LM000 disks, is 98% full, is attached to the VM as a queue-depth-1 emulated IDE disk, and the controller runs write-through.
  - Our move rsyncs plus the memory backup overran it, so NFS writes queued 240 s and that cascaded into smbd and CIFS.
  - The mirror looks DEGRADED: VM disk `sdb` (host `/dev/sdc`, controller volume reset at 12:29/12:30Z) takes no writes. **Cyril should check `zpool status ai_pool` on the NAS VM.** The ai-training corpus exists only on that pool.
  - Reads now run 16-25 MB/s from the AI server.
  - Freed 44 GB: our partial qwen-fold children copy.
  - Category B (Cyril): `omnicoder_2026` 610 GiB; agent-memory backup retention (about 27 GB/day × 30 days ≈ 810 GB; 7 daily + 4 weekly would free about 350 GiB); likely-duplicate NF sealed caches; old ollama/lmstudio, 37 GiB; `lexar-relief` partial copy, 38 GiB.
- **Flash Next:**
  - The GGUF is being copied NAS → root SSD (`ops-20261018/flash_gguf_stage.sh`; direct I/O, resumable, sha-checked, then the lane's OK mark). It started 21:22Z at about 16 MB/s, so it should be done around 22:55Z.
  - The P40 tool-call generation and scoring start after run 1's first steps are confirmed. Either:
- (a) free Lexar by deleting the D1 transient scratch (`scratch/d1_boundary` 77 GB + `scratch/d1seal` 52 GB; D1 sealed 10-15; transient scratch under STAGE_D_RULINGS) and re-download the GGUF from Hugging Face to Lexar (about 2.5 h at 10 MB/s); or
- (b) free NAS space and restart its SMB/NFS; or
- (c) delete the cold `qwen-fold/children` (140 GB on root).

Until then the P40s are idle.

## 5. Risks

- **Host RAM:** GPU6 K3 (~87-93 GiB) + trainer (~80 GB) + Flash Next llama-server (n-gram tables, if CPU-resident). Watch MemAvailable. K3 has `oom_score_adj` 500, so it dies first.
- **Lexar:** the K3 queue needs ≥ 200 GB free before each pass. The trainer keeps one 47 GB checkpoint (a peak of 94 GB while one is written).
- **Unverified:** the trainer's rate at micro 8 on the real model; K3-row behaviour in training (first K3 rows come at step 1 from pass 0); whether Flash Next fits 4 × P40 with its n-gram tables.
