# Codex Omni work in flight on 2026-10-14: disposition

**Written:** 2026-10-14, about 23:50Z, by Claude, lane `inflight`.
**Authority:** Cyril's ruling `cyril-claude-leads-all-lanes-2026-10-14` (`STAGE_D_RULINGS_2026-10-12.md`). Claude leads every lane. Codex is out of usage and contributes under Claude's direction when it has usage again.
**Inputs:** `STAGE_D_RECONCILED_PLAN_2026-10-14.md` (§1.3, §3-4, §7), `STATE_2026-10-14_WAVE3.md`, Codex's ledger `reviews/2026-10-13-folding/RECOVERY_STATUS_2026-10-13.md`, the 20 Codex Relay reports on thread `64abe991` (last one `a3a8e5d1`, 2026-10-14 00:17Z), Codex branch HEAD `81f54988` (read-only), and the Codex cloud-thread workspace `C:/Users/cereal/Documents/Codex/2026-10-05/task-4/oct14_real_qualification/` (thread `01a10f78`, the 10-14 D1/native work).

**Evidence marks:** **(m)** = I saw it printed in this lane, with the command or file named. **(cx)** = Codex reported it and nobody re-computed it (UNVERIFIED). **EST** = estimate.

**What this lane changed:** it wrote new files under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/`, `/mnt/lexar/omni-campaign/scratch/codex-inflight-*`, and three tool files plus this document under `$OC/distill/`. It deleted, moved, edited, started or stopped **no** Codex artifact, container, process or allocation. It touched no Claude live job (D1, K3 queue, tool self-gen, k3accel, speech lane, D2 prep).

---

## 0. Bottom line

1. **Nothing of Codex's is running.**
   - No Codex container is running: 92 `codex-*` containers exist, 0 running, 2 in state `Created` that never started (m, `docker ps -a --filter name=codex`, 23:38Z).
   - No GPU allocation is active: `gpu_runtime_allocations` has 0 rows in `starting`/`running`. The last Codex release was R6 at 16:29:00Z (m, read-only SQL via `agent_memory.connect()`, script `scratchpad/inflight/alloc_read.py`).
   - No Codex controller or watcher runs on this PC. The D1 controller PID 5160 and watcher PID 10308 named in Codex's runtime handoff are gone (m, `Get-Process -Id 5160` empty; the full `Win32_Process` list holds no Codex Python).
   - One old, idle host loop is alive on the server: pid 4169670 (item 18). I left it alone.
2. **Finished under Claude, CPU only:** 14 Codex artifacts (7,636 files, 3,884,097,474 B) are now byte-verified copies under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/`. Each has an `ADOPTION.json` and a `SHA256SUMS`, and Codex's originals are unchanged (§3). The three completed runs printed `9 adopted, 1 failed` (the failure was redone and passed in run b2), `2 adopted, 0 failed` and `3 adopted, 0 failed`. Two of the 14 are the root-owned speech/H3 native trees that Claude could not read before; they needed a root-side staging copy (items 4 and 5).
3. **No GPU job was queued.** No Codex item still needs a GPU for v1 (§4).
4. **One open risk needs a Relay message from the parent session.** Codex's cloud-thread resource-watch automation `6acfc5cb2a988191b2c0ad5410d61796` was armed to launch the 80 GiB `codex-control-bf16-cpu` retry once MemAvailable reaches 180 GiB (item 14). It cannot run while Codex has no usage. Once Codex has usage, the first moment it could fire is around the D1 seal (10-15 about 20-22Z, EST), which is exactly the K3 RAM-gate window.

---

## 1. Live check, 2026-10-14 23:04-23:45Z (m)

| What | Reading |
|---|---|
| GPU compute apps | GPU0/GPU4 pid 125746 (Claude hardened D1); GPU1/GPU2 pids 4054571/4058904 (Claude tool self-gen); GPU6 pid 306480 (Claude K3). GPU3 and GPU5: none. |
| `p40.claims` | Last line: GPU3 `k3accel` `k3-v130-parity` released at 21:55:53Z |
| MemAvailable | 94.7 GiB (23:04Z), then 72.2 GiB (23:38Z) |
| `/` | 70G free (97%). `/mnt/lexar`: 482-499G free |
| Codex containers | 0 running. `Created` (never started): `codex-control-row-cpu-3713ff25953e4dba`, `codex-audio-consumer-speech-ce-modality-v2` |
| Codex pid files under `$OC/distill/codex-preparation-01a1326f-20261013/` | 17 `.pid` files, every pid dead (m, `kill -0`) |
| Old host loop | pid 4169670, `k3_capacity_host_loop_proc_race_v4r9.py` (sha `9cffb4b7…`), up since 2026-10-09, sleeping (`hrtimer_nanosleep`), RSS 5.9 MB |

---

## 2. Disposition per item

Decisions: **FINISH** = completed under Claude now; **ADOPT** = taken as-is after sha verification; **PARK** = kept, not needed for v1; **SUPERSEDED** = kept, marked as replaced. Every Codex path below stays where it is.

### 2.1 Items adopted or finished for v1

| # | Codex work | State | v1 value (reconciled plan) | Decision | Receipt (all under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/`) |
|---|---|---|---|---|---|
| 1 | **Tool TRAIN 168-row Q38 capture** (`codex-qtext-tools-gpu4-oct14-v1`, allocation `554d9016`) and its packed corpus `native-text-v7-train` | Finished: exit 0, sealed 00:35Z (cx), allocation released 04:19:30Z (m) | Stage-T format fixture, not D2 KD data (§3.3.4) | **ADOPT** | `tool-train-168-cache`: 10 files, source pin `manifest.json` `367e509d…` matched, `shard.json` 5/5 refs. `tool-train-168-corpus`: 2 files, `assistant_text.safetensors` = `3ed3f303…` (the cache key). SHA256SUMS `1efeb8cb…` / `f73be494…` |
| 2 | **Speech teacher campaign BF16 v2** (`codex-chimera-speech-gpu4-oct13-v2`, allocation `13e72460`) | Finished: released 2026-10-13 20:04:44Z (m). 192 utterances, 186 exact (cx) | D3e seed rows (§5) | **ADOPT** | `speech-teacher-campaign-bf16-v2`: 583 files, 86,867,626 B, SHA256SUMS `43b59df9…` |
| 3 | **Held speech set** `speech-held-cpu-v1` | Finished: 16 pairs, 15 exact (cx) | D3e held probe | **ADOPT** | `speech-held-cpu-v1`: 54 files, SHA256SUMS `7f06538c…` |
| 4 | **Speech native caches**: CE 186 / flow 744 / combined CE 194 + replay index `3c99cfb7…` (`codex-native-audio-consumer-modality-v2`) | Finished 10-13 (collection 23:19Z). Tensor dirs are **root-owned 0700**, so cereal could not read them | D3e seed and parity reference (§5) | **FINISH** (CPU): root-side read-only staging copy, then adoption tied to root-side sums of the original | `speech-native-caches-v2`: 3,482 files, 3,539,620,308 B. All 3,482 equal the original's root-side sums (`cb1b004e…`). Pins: `replay-index.json` = `3c99cfb7…`, `collection-v3.tar.gz` = `1570f2b9…`. Replay index: 2,335 distinct refs, **1,882 ok**, 0 mismatched, 0 missing, 453 external (§6). SHA256SUMS `107b6686…` |
| 5 | **H3 native 32 batches** (`codex-h3-native-modality-v3`, manifest `ba5ac505…`, proof `b8f2812a…`) | Finished. The `.safetensors` are root 0600 | A3 type-3 chart input; parity reference for the H3 production cache (C3) | **FINISH** (CPU), as item 4 | `h3-native-batches-v3`: 150 files, equal to the root-side sums (`e5fafe41…`). Pins: manifest `ba5ac505…`, `proof.json` `b8f2812a…`. Manifest: 70 refs, **67 ok**, 3 external. SHA256SUMS `c8fea91e…` |
| 6 | **Collection-v3 tarball** (Codex's own root-written byte record of items 4+5, 2,334 files, archive `1570f2b9…`) | Finished | Provenance | **ADOPT** (metadata only; tensors come from items 4 and 5) | `speech-h3-native-collection-v3-meta`: 2,334 files equal to the index, with 3 payload pins (replay index, H3 manifest, proof). SHA256SUMS `3956ae0a…`. The first attempt (plan a) failed correctly: the replay index names 938 tensors that are not in the tarball |
| 7 | **H3 writer-repair BF16 capture**: velocity (`78cb8f3c`) and text (`0ffb3aa6`), 8 sources × 4 σ | Finished: released 21:18Z / 21:01Z 10-13 (m) | Parity reference for the H3 production cache and the A3 chart decision | **ADOPT** | `h3-bf16-velocity-v1`: 23 files, pin `…/c354051646c9aaf0/manifest.json` = `7770cf61…`, SHA256SUMS `35ee3605…`. `h3-bf16-text-v1`: 24 files, SHA256SUMS `c7072ccc…` |
| 8 | **Video top-64 K64 runtime**: server capture (`c9c4a743`) + stock control (`9cfedcb2`) `video-k64-singleton-v1`, and Codex's local-only validation | Finished: both exit 0, 792/792 exact (cx) | Parity and capacity reference for j6 | **ADOPT** | `video-k64-singleton-v1`: 896 files, pin `inputs/evidence/ingress.json` = `77aee60e…`, SHA256SUMS `2636c92a…`. `codex-local-video-k64-runtime-v1`: 65 files, payload pin `validation.json` = `d0bcdeab…`, SHA256SUMS `72b5762e…` |
| 9 | **Speech combined admission** (local `proof_outputs/chimera_speech_combined_admission_oct13_v1`) | Finished: 186 train / 15 held / 7 quarantined, canonical seal `f56c6968…` inside `preparation.json` (cx) | Provenance for the D3e seed rows | **ADOPT** | `codex-local-speech-combined-admission-oct13-v1`: 7 files, `preparation.json` = `dbfebd26…`, SHA256SUMS `9377f781…` |
| 10 | **Held-out evaluation, 274 tasks** (local `chimera_recovery_eval_combined_oct13_v1`) | Finished, frozen. It was **not on the server** before this lane (m, `find`) | Supplementary probe at every stage seal (§1.2, §3.2). Also the 13-gram contamination source for the tool self-gen `finalize` (§3.3.1) | **FINISH** (placed on the server) | `codex-local-recovery-eval-combined-oct13-v1`: 3 files. `tasks.jsonl` = manifest `holdout_sha256` `3795f333…`; `model_inputs.jsonl` = `inputs_sha256` `14929767…` (m). SHA256SUMS `76fe3ef2…` |
| 11 | **Execution-backed tool seed, 504 episodes** (local `chimera_tool_seed_oct13_v2`) | Finished. Not on the server before this lane | test 168 = held-out probe; val 168 = stage-T dev set; train 168 = format fixture (§3.3.4) | **FINISH** (placed on the server) | `codex-local-tool-seed-oct13-v2`: 3 files, `report.json` `3d66f503…`, `artifacts` table 2/2 refs matched. SHA256SUMS `dc01d001…` |

### 2.2 Items parked or superseded

| # | Codex work | State (m unless marked) | v1 value | Decision |
|---|---|---|---|---|
| 12 | **QIMG native full v3 reconversion** (`native-qimg-full-v3`, manifest `d33795a3…`) | Finished 2026-10-13 16:54Z, exit 0, `{"accepted_batches": 1126, "sealed": true}` | Codex native-trainer batch format. Claude's D3 trains from Claude's own sealed QIMG TE/VAE/velocity caches, which are valid with no recapture (§3.4) | **PARK**. Parity reference for the `d3_hybrid_module` port (§1.3). v2 (`native-qimg-full-v2`, 1,309 batches, manifest `d5868886…`) is **SUPERSEDED** by v3 |
| 13 | **ACE native full v4 conversion** (`native-ace-full-v4`, manifest `3d436b20…`) | Finished 16:40Z, exit 0, sealed 691 items / 2,764 batches | As item 12: Claude's ACE DiT and ACE LM caches are valid (§3.4) | **SUPERSEDED** by v5 (`native-ace-full-v5`, manifest `6291173d…`, finished 16:58Z, exit 0). v5 prints `"model_loaded": false, "native_tensor_revalidation_required": true`. That revalidation is a Codex native-consumer step, and v1 does not use that consumer. v5: **PARK** |
| 14 | **Native control-row repair, `codex-control-*`** (lane L-NATIVE). Tiny `8eaeed8e` exit 0; independent `c4b23840` exit 0; control-row qualification `15aeb3b7`/`7fe554d8` exit 0; **full 80 GiB `939ef3ab` exit 124** (the 1,800 s timeout; last marker `backward_start` at 986 s, no OOM, per Codex's `FULL_MODEL_BOUNDED_HANDOFF_V3.md`); retry `run_20261014T175953138439Z_9118686a18fa459f` **prepared, never launched** (container absent, m); `3713ff25` `Created`, never started | Stalled by design. Codex's gate needs MemAvailable ≥ 193,273,528,320 B | None. The native control band [248320, 248337) is unreachable in Claude's deliverable (plan §2.5): corpus refusal, GGUF truncation to 248,320 rows, and the p11 band mask | **PARK**. Never launch it outside a posted window (§7.4.3). **Risk:** Codex's resource-watch automation `6acfc5cb2a988191b2c0ad5410d61796`. Its last iteration (18:14:19Z) was `hold_resource_admission_unchanged`, short by 64,396,025,856 B. I found no local definition of it: it is not in `~/.codex/automations` and not in any `~/.codex/*.sqlite`, so it is probably a cloud-thread automation. I cannot disable it from here (§5) |
| 15 | **Native tool baseline and tiny runs** (L-NATIVE): `codex-p40-native-baseline-bc02df7c44fd` (allocations `87db9850`/`881dc49a`, released 13:57Z); tiny `095af76a`, `2dae057a`, `a996e473`; `codex-native-tool-tiny-cli-347f67117974` exit 1; `…-tiny-runtime-105275345fe1` exit 0; `codex-p40-unmasked2-trace-d4f4cccf7c88` (`evidence_only: true`, no GPU) | Finished. The baseline **failed behaviourally**: it generated `16371` then `279486` (music band) and stopped with `NativeProtocolError` (cx, Codex runtime handoff; report sha `5f19ace4…`) | Evidence only for the parked native protocol | **PARK** |
| 16 | **Codex D1 attempts** v31 → c36-r2 → c37-r3..r6 (allocations released; Lexar `d1-candidate-*`, `d1-continuation-source-*`, `d1-readback-*`; `$OC/distill/codex-preparation-01a10f78-20261014/`) | Over. Takeover receipt `$Q/d1_takeover/codex_d1_exit_20261014T174756Z.json` | L00-L10 are already the canonical prefix of the hardened D1 (§1.1) | **SUPERSEDED** by `claude-d1-hardened.service`. Keep everything |
| 17 | **K3 parity 012** (`codex-k3-parity-012-1394fd65a8f8`, allocation `1b39b4e5`, GPU6) and the 10-13 bisect | Finished: exit 0, 689.6 s, `training: false` | Diagnostic | **SUPERSEDED** by Claude's K3 L2 mechanism receipts (`c4c3ed81`, `82532ff0`, `e1ce01c3`; plan §6) |
| 18 | **Old K3 NAS window 61-72 host loop**, pid 4169670, run `a151e1a8…`, container `k3-nas61-72-a151e1a839b6486e` (Exited (2), 2026-10-09) | Alive and idle since 2026-10-09T10:28:44 (last log line: attempt `3bbd07a7` admitted). It polls `capacity_requests/` every 1 s. Its docstring says it only runs a read-only preflight, with "no allocator, provider, Docker start/stop/update" | None (Claude's K3 queue 1.3.0 owns K3) | **SUPERSEDED.** Left running: it is harmless, I did not start it, and stopping another agent's process is not this lane's call. Its owner is UNVERIFIED (the namespace style is Codex's). The parent may ask Codex to stop it |
| 19 | **Speech 1.7B comparison** (`codex-speech-1p7-comparison-modality-v1`) | Download complete: `{"complete": true, "all_source_bytes_verified": true}`, 9,223,333,259 B (m). CPU canary: ASR baseline/candidate/chain exit 0; **TTS exit 1** with `TimeoutError: 600 second CPU model-task budget exhausted` (m). State: partial | The 1.7B TTS is the planned data voice (§5). Its P40 rate (card C-s0) is unmeasured | The **parents are ADOPTed in place** (8.6 G; not copied). The CPU canary is **SUPERSEDED** by C-s0, which belongs to L-SPEECH-DATA (the speech lane v2 workflow, GPU3 through `p40.claims`). Not duplicated here |
| 20 | **Optional media lanes**: geometry flow v1-v5 / bridge / conditioning, HY3D target session, image-edit (32) and image-edit D4, temporal video (24), joint-audio and speech preparation trees | Finished (the last successful versions exit 0) | Optional D3 lanes under A3 (§1.3), L-MEDIA-DATA. HY3D mode/time is card C-V1 | **PARK** |
| 21 | **Failed CPU attempts** in `codex-preparation-01a1326f-20261013` (exit census, m): `ace_native_bounded_v1/v2`, `native_pack`/`v5`/`v6`, `d1-source-successor-v1`, `d4_h3_objectives_cpu_2111_v2`, `d4_speech_cpu_1934_v1`/`1938_v2`, `image_video_cpu_2253_v2`, `codex-h3-native-modality-v1/v2` | Failed, each with a later successful successor (`ace_native_bounded_v3`, `native_pack_v7`/`v8`, `d1-source-successor-v2`, `…_2110_v3`, `d4_speech_cpu_1951_v3`, `image_video_cpu_2300_v3`, `…-modality-v3`) | — | **SUPERSEDED.** Kept as Codex's failure receipts |
| 22 | **Failed attempts with no successor confirmed by me**: `d4_objectives_cpu_2109_v1`, `d4_writer_cpu_2000_v1`, `recovery_cpu_2310_v1`, `recovery_wiring_cpu_0026_v1` (exit 1), `speech_flow_cpu_1952_v1` (exit 1), `codex-audio-sampling-tests-modality-01a1326f-v1` (exit 139) | Failed | v1 uses Claude's engines (`d3_hybrid_module`, the D4 driver), not these Codex runners | **PARK** as failure receipts. Codex's ledger reports later passing suites (for example "audio stage-order repair validated with 28 actual native CPU passes", cx). I did not map each one |
| 23 | **H3 host reservations** `ice-roster-20260919-gpu3.json` / `-gpu5.json` | Unchanged since 2026-09-20 (m) | — | Outside this lane. Recorded because Codex listed them as unresolved |

---

## 3. What was executed (FINISH / ADOPT, CPU only)

### 3.1 Tools (deployed under `$OC/distill/tools/`, local mirror under `reports/omni-campaign-2026-09/distill/tools/`, sha parity checked on both sides)

| File | Version | sha256 | Tests |
|---|---|---|---|
| `codex_inflight_adopt.py` | 1.2.0 | `0605e54d38ad1482cfcfba5360bc07f297ce8e32b6b992615a1dd8c57ae43bee` | `test_codex_inflight_adopt.py` (`22ed9fb7…`): **17 passed** locally (Python 3.14) and on the server (venv) |
| `run_codex_inflight_rootcopy_2026-10-14.sh` | 1.0.0 | `0a9a33828e66b3ebcf57b89ecb9aacab62bc3d085930a18160cea99deda335ec` | `bash -n`, plus the real run below |

Superseded copies of the tool are in `tools/_superseded/` on the server and in `_superseded_local/` locally: 1.0.0 `56620a21…`, 1.1.0 `736d3e3b…`.

What the tool guarantees:
- Every source pin is checked before anything is copied.
- Tree copies are re-hashed. For tarballs, the archive sha must equal its index, members are extracted safely, and each must match the index exactly.
- Codex's own manifests are walked for `{path, sha256}` references, and every reference that resolves inside the copy must match.
- 1.1.0 adds `source_sums`: a staged copy must equal the root-side sums of the original exactly.
- 1.2.0 adds `move_staged`: only Claude's own staged copy may be renamed into place, and on any failure it is renamed back.
- Payloads are made read-only, an `ADOPTION.json` and `SHA256SUMS` are written per item, and a run receipt is written per run.
- Re-runs skip items that already passed. Nothing is ever overwritten.

Gates used on every run: `CUDA_VISIBLE_DEVICES=`, `nice -n 15`, `ionice` (idle, or best-effort level 7 for run b2), `OMP_NUM_THREADS=1` (single-threaded hashing), and MemAvailable ≥ 40 GiB checked before each item (72-94 GiB was available). Outputs go only to `/mnt/lexar/omni-campaign/`, never to `/`.

### 3.2 Runs and receipts

| Run | Plan (uploaded to `/mnt/lexar/omni-campaign/scratch/codex-inflight-uploads/`) | Result |
|---|---|---|
| a, 23:19Z, tool 1.0.0 | `codex_adopt_plan_2026-10-14.json` `7bcdad5e…` | `RESULT … 9 adopted, 1 failed`. Receipt `RECEIPT-2026-10-14T231922Z.json` (`a224bdac…`). The failure is item 6 with manifests: 938 tensor refs absent from the tarball. That is correct behaviour and is fixed by plan b |
| c, 23:37Z, tool 1.1.0 | `codex_adopt_plan_2026-10-14c.json` `7d23830f…` | `RESULT … 2 adopted, 0 failed`. Receipt `RECEIPT-2026-10-14T233744Z.json` (`2c88d5d5…`) |
| root copy, 23:27Z | driver 1.0.0, one `nginx:alpine` container (`--network none --cpus 2 --memory 2g`, both Codex dirs mounted `:ro`) | rc=0. Original shape: audio-consumer 3,482 files, H3 150 files, 0 symlinks, 0 special files. `SOURCE_SHA256SUMS.codex-native-audio-consumer-modality-v2` `cb1b004e…`, `SOURCE_SHA256SUMS.codex-h3-native-modality-v3` `e5fafe41…`. Staging: `/mnt/lexar/omni-campaign/scratch/codex-inflight-rootcopy-2026-10-14/`. busybox `cp -a` printed 3,467 "can't preserve ownership" warnings: Lexar does not keep Unix owners. Bytes are checked by the adopt step, not by `cp` |
| b, 23:28Z, tool 1.1.0 | `codex_adopt_plan_2026-10-14b.json` `28ca9437…` | **Stopped by me** after about 15 min. The 6.4 G staged tree was being read at about 6 MB/s under `ionice -c3`, and 1.1.0 would have hashed it three times. Only Claude's own `.partial` staging was left behind; run b2 cleared it on start |
| b2, 23:43Z, tool 1.2.0 | `codex_adopt_plan_2026-10-14b2.json` `24070521…` | `RESULT codex_inflight_adopt 1.2.0: 3 adopted, 0 failed`, finished 23:45:07Z. Receipt `RECEIPT-2026-10-14T234507Z.json` (`78f66512…`). Lines: `speech-native-caches-v2: files 3482 … replay-index.json refs 2335 ok 1882 external 453`; `h3-native-batches-v3: files 150 … refs 70 ok 67 external 3`; `speech-h3-native-collection-v3-meta: files 2334`. The staged trees were moved in, so `…/codex-inflight-rootcopy-2026-10-14/tree/` is now empty. The root-side sums files stay beside it as evidence |
| independent re-verify, 23:50Z | `codex_inflight_adopt.py verify-item` over every item directory (log `logs/verify-all-20261014T235034Z.log`) | 14 lines, every one `n/n files match SHA256SUMS`, 7,636 of 7,636 files in total (m) |

---

## 4. GPU work queued: none, and why

| Candidate | Why no GPU job |
|---|---|
| Tool TRAIN 168 capture | Already finished and sealed (item 1) |
| Speech 1.7B TTS rate / bulk TTS | Card C-s0..s4 belongs to L-SPEECH-DATA, which the speech lane v2 workflow is building now on GPU3 through `p40.claims`. A second queue here would compete for the same lock |
| 80 GiB control-bf16 retry | CPU job, PARKed (item 14) |
| ACE v5 revalidation | CPU, and PARKed (item 13) |
| H3 production cache / j6 | Claude's own card C3 (L-K3 / L-TRAIN), not Codex in-flight work |

---

## 5. Hand-offs and asks

**For the parent session to send on Relay** (sub-lanes never run `sync`). Five lines, per plan §7.3:
1. Lane: inflight (Codex in-flight disposition). Claude leads all lanes per `cyril-claude-leads-all-lanes-2026-10-14`.
2. State: this file. Adopted copies are under `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/`, receipts listed in §3.2.
3. Running: nothing of Codex's; the adoption tool has finished.
4. Next, for Codex when it has usage:
   - (a) Do **not** launch `run_20261014T175953138439Z_9118686a18fa459f`. Disable or re-gate resource-watch automation `6acfc5cb2a988191b2c0ad5410d61796` so it only fires inside a Claude-posted window (plan §7.4.3).
   - (b) Stop the idle host loop pid 4169670 if it is yours.
   - (c) Leave the two `Created` containers unstarted, or remove them yourself.
   - (d) Read the PARK/SUPERSEDED marks in §2.2 before resuming any native lane.
5. Do not touch: GPU0/4 (D1), GPU1/2 (self-gen), GPU6 (K3), GPU3 (k3accel / speech via `p40.claims`), GPU5 (never), and `/` (70G free, floor 60G).

**Files this lane deliberately did not edit:**
- `$Q/LANES.json`: owned by the wave-4 workflow. The Codex lanes there still read "awaiting Codex's first heartbeat". The owning workflow should record the Claude lead and point L-SPEECH-DATA and L-DATA at the receipts above.
- `$OC/distill/tools/TOOL_PINS.json`: owned by the wave-4 workflow. `codex_inflight_adopt.py` 1.2.0, its test and the root-copy driver are not pinned until its owner runs `tool_pins.py generate`.

---

## 6. What I could not verify

- Every (cx) number: the 186/192 and 15/16 speech splits, Codex's 792/792 video parity, the 1,126 and 691/2,764 batch counts as quality claims (only their seal flags and exit codes were read), and the native-baseline token IDs.
- Who owns host loop pid 4169670.
- Where automation `6acfc5cb…` is defined, and whether Codex's cloud thread will run it again.
- The 453 distinct replay-index references outside `/recovery` (by raw occurrence: `/joint/inputs` 940, `/work/inputs` 745, `/work/clean-speech-cache-v1` 402, `/work/clean-audio-cache-v1` 48), and the 3 `/recovery/inputs` references of the H3 manifest that did not resolve inside the copy. They point into Codex's speech/joint-audio preparation trees, which were not adopted. They are counted as external and not sha-checked here.
- Successors for the six failed attempts in item 22.
- The 1.7B parent file shas. Codex's `download.json` says verified; I did not re-hash the 9.2 G.
- That ACE v5's `native_tensor_revalidation_required` is harmless for any consumer other than Codex's native trainer.
