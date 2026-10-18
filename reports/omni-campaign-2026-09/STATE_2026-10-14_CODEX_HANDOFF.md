# Chimera Omni state after the Codex handoff, 2026-10-14

**Written:** 2026-10-14, about 14:10Z, by Claude, as a read-only review. Nothing on the server was changed, stopped, launched, registered or deleted.

**Inputs:**
- Four reader passes and a verifier pass:
  - codex-forensics
  - live-server
  - tools-tree
  - ledger-artifacts
  - suspects S01-S14
- Two read-only SSH snapshots of my own, at 14:01:16Z and about 14:05Z.

**Timestamps:** all times are UTC. Server logs print local time, UTC-6; I converted them. "UNVERIFIED" marks claims I did not see printed.

---

## 1. Summary

**OMNI1 is built, registered and safely cloned.**
- OMNI1 is `OC02-F-Q38-27BU-OMNI1`, with 27,535,655,753 parameters, at `/mnt/lexar/omni-campaign/omni1/`. It is registered as a candidate (ledger line 36, digest `8edf50c4…`). Its s7 check passed.
- The S9 clone `OMNI1.clone-pre-D1` exists (ledger line 39, sealed 2026-10-13 20:02:58Z). Codex made it, and its digest equals OMNI1's.
- `oc_ledger.py verify` returns 39/39 rows with 0 failures. Registry verify returns 16 checkpoints, also with 0 failures.

**S8 shows that OMNI1 is structurally sound but not yet intelligent.**
- The S8 text review finished at 08:47:59Z, before Claude stopped. Of 16 cases, 3 returned empty answers and 13 ran to the 1,024-token cap, 9 of them flagged as loops. Arithmetic was wrong (17×6 = 118).
- The S5 fp32 teacher-forced proxy matches OMNI1's next token at 1019-1024 of 1024 positions. So the text damage comes from the folded trunk (S3/S5), not from the media blocks.
- Media decodes are finite but no better than the noisy input:
  - ACE, H3 and HY3D are worse than the noisy input.
  - QIMG is only 0.2-0.4 dB better.
  - Vision answers are not grounded in the image.
- The music-code band wins the unmasked argmax at 146 of 210 positions.
- Stage D exists to fix all of this.

**Stage D status:**
- **Done:**
  - Most teacher caches (§4).
  - The D3 hybrid module (three must-fix findings fixed in 1.0.2; synthetic, parity and rate tests pass).
  - E3/GGUF export tooling.
  - Trainer-prep fixes.
- **Running:**
  - D1 trunk recovery, attempt 5 (`c37-recovery-r4`). It is Codex's, on GPU0+GPU4, and is still re-running already-trained layers. 11 of 64 layers are trained.
- **Broken or not started:**
  - The K3 teacher cache. B2 parity failed and GPU6 has been idle since 2026-10-13 23:13:53Z.
  - The H3 production caches. The ruling file was never written, though Cyril lifted the gate.
  - The vision-in/video-in answer cache (j6 skipped).
  - P2 engine rate (out of memory).
  - The P11 needle baseline (hit=0, probably a scoring artifact).
  - D1 is far too fragile to finish without fixes. Its lease monitor runs on Windows, and every restart re-runs all the finished layers.

---

## 2. What Codex did, and what it botched

### 2.1 Timeline by area

**Threads**
- Root thread `01a1326f`: 2026-10-13 13:39Z to 2026-10-14 00:44Z. Its subagents were Bacon, Locke, Aristotle, Tesla and Gauss.
- Cloud thread `01a10f78`, "Advance Chimera Omni development": 2026-10-14 from 03:06Z, still active. Its workspace is `C:/Users/cereal/Documents/Codex/2026-10-05/task-4/`. Its transcript is not on this PC (UNVERIFIED beyond its files).

**Cyril's instructions to Codex (root thread, verbatim excerpts)**
- 14:02Z: "[Cyril quote withheld]".
- 14:26:34Z: "[Cyril quote withheld]".
- 14:41Z: multi-token folding objective rather than first-token match.
- 17:38-17:56Z: speech models ("[Cyril quote withheld]"; "uploaded several to dot").

**What Codex did, by area**

| Area | What happened | Evidence |
|---|---|---|
| Step 1 / S8 | Codex reviewed S8 (`reviews/2026-10-13-folding/REVIEW.md`, 13:50Z). It did not touch the S8 outputs and wrote no step-1 results report. | codex-forensics; no mtime change under `reports/omni-campaign-2026-09/` |
| S9 clone | Done correctly. Dry run at 19:50:57Z, real run at 19:54:47Z (`oc_checkpoint.py clone … --mode copy --confirm`), sealed 20:02:58.841Z. 55,073,531,261 bytes, 18 files, read-only. | ledger line 39; `/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1.clone-receipt.json` |
| Ledger | The clone is the only ledger write. | `wc -l ledger.jsonl` = 39 (14:01Z) |
| gpu6 / K3 | Read-only audits. A cloud-thread L0-L2 probe ran 12:27-12:39Z on 10-14 and reported "historical_exact_match=True" using whole-corpus call geometry. No queue changes; no K3 production run. | `/mnt/lexar/omni-campaign/distill/codex-k3-parity-012-1394fd65a8f8` |
| gpu0 / qtext | Left Claude's queue alone. Ran its own work through the allocator: qtext offload parity, a compiler image, capacity runs v1-v4, and a 168-row tool-training cache (sealed 00:38Z). | codex-forensics |
| GPU4 | Used the card after Claude's gpu4 queue ended (13:03:34Z) for an H3 joint encode, an H3 bf16 text/velocity cache (test scale), a QIMG edit VAE, and a video k64 capture. | codex-forensics |
| Media / speech | Downloaded Qwen3 ASR/TTS 0.6B and 1.7B (9,223,333,259 B for the 1.7B pair). Built speech, joint-audio and test-scale H3 trees under `$OC/distill/codex-*`, about 31 G on root. | `du -sh` (13:24Z) |
| D3 / export | Untouched. Claude's queues finished on their own: d3 ended 10:51:11Z, export ended 09:44:09Z. | queue `.status` files (14:01Z) |
| D1 | Codex forked onto its own successor dataset and its allocator-wrapped launch of Claude's trainer. Six attempts (§2.2 B1). | §2.2 |
| Shared infrastructure | Hot-deployed the GPU allocator `local-hermes/remote_runtime.py` three times; sha `acb2d2fc…` is live (00:21:49Z). Patched the Relay dispatcher and `current_session` (16:55-17:16Z). | `gpu-reconciliation-fix-v1/`, `gpu-status-fix-v2/deployment.json` |

### 2.2 Botches, most severe first

**B1. HIGH: D1 cannot finish as it currently runs.**
- **Progress:** 11 of 64 layers trained in about 9.7 h of wall clock (allocation start 04:19:34Z to now).
- **Attempts:**
  - **v31-v34:** early failures:
    - `trainer/` was OOM-killed by the memory cgroup (about 132 GB of pinned shared memory).
    - v32 also hit the 128 GiB cap (`oom_kill 1`).
    - v33 failed with `KeyError getpwuid uid 1000`.
  - **v35:** trained L00-L08, then was OOM-killed inside the container at L09 step 230/230. The log stops at 09:02:16Z with `"OOMKilled": true` at 09:10:34Z. L09 was never saved.
  - **c36-r2:** re-ran L00-L08 from 09:40Z to about 11:48Z, trained L09 (12:13:15Z) and L10 (12:38:29Z). It then died at 12:41:06Z, exit 2: "D1 allocator attestation stale/revoked/foreign; refusing optimizer update". Its Windows monitor had died on "TimeoutError: Remote command exceeded 25.0s".
  - **c37-r3:** its Windows monitor died at 13:27:51Z ("transport unavailable"). It trained 0 new layers.
  - **c37-r4:** now running (§3).
- **Root cause:**
  - The lease monitor `d1_allocated_root.py` runs on Windows. It refreshes a proof over server-manager SSH every 10 s, and the proof expires after 45 s (`max_proof_age_seconds 45`).
  - It retries only on a snapshot timeout. On any other failure it publishes `valid=false`, and the trainer then raises at `d1_allocated_contract.py:272`.
  - Each restart re-runs every finished layer at about 1,000 s per KDA layer.
  - The time between monitor deaths has been 16 min to about 3 h. Re-running already takes about 2.6 h and grows by about 17 min per trained KDA layer, so it will soon exceed the time between deaths.
- **Evidence:**
  - lease `live.json` files under `$OC/distill/runtime_leases/D1-native-01a10f78-oct14-*`
  - `ACTUAL_D1_INTERRUPTION_20261014T091033.json`
  - `controller_c36_r2.stderr.log`
  - `ACTUAL_R3_FAILURE_20261014T133725278899Z.json`
  - all under `task-4/oct14_real_qualification/`
- **Rungs on disk:**
  - L00-L08: `/home/cereal/omni-campaign/distill/d1/run-OMNI1D1-native-01a10f78-20261014/trainer-v35/rungs/`
  - L09-L10: `/mnt/lexar/omni-campaign/distill/d1/run-OMNI1D1-native-01a10f78-20261014-c36-recovery-r2/trainer-c36/rungs/`
  - Each rung is 133,704,232 B.

**B2. HIGH: the K3 teacher cache was never built, and GPU6 has been idle for about 15 h.**
- Claude's B2 parity failed on 2026-10-13 at 13:51:08Z, before Codex took over. Only one gate failed: `target_logprob_p99: False`; top-1, top-16 overlap, median, finite and complete all passed. Receipt sha `b73408ad7e65`. The auto-bisect then reported `first_unequal_layer: 2` (23:13:53Z).
- Block compare (`$OC/distill/queues/logs/gpu6/b2_block_compare.log`):
  - Block 0 is bit-equal.
  - Divergence grows to relL2 0.0825 and cos_min 0.478 at block 5 (L60).
- Codex's probe showed that L0-L2 match when run with the fold's whole-corpus call geometry. That makes the per-row-group geometry (5 groups, 143,360 positions of which 98,797 are real) the prime suspect. A padding/KDA-state interaction is a plausible mechanism, and "padded pass-token guard" is an open k3cache low finding in `STAGE_D_WAVE1_STATUS.md` §4.6. The cause is UNVERIFIED.
- Codex did not act on its probe. It did not install the fixed queue `gpu6.sh.new` (1.1.0) or rerun B2.
- D2 cannot start without the 2.0M K3 tokens (prereg D2; rulings "K3 budget … K3 is never skipped").

**B3. HIGH: the vision-in/video-in answer cache was dropped without anyone deciding it.**
- gpu0 j6 (5,896 image rows + 512 clips) logged "skipped reason=no_go_file" at 2026-10-14 00:11:44Z, because `$OC/distill/queues/gpu0.vl_go` was never written. It was still absent at 14:01Z.
- The human read of the j3 smoke receipt (done 12:58:26Z) was assigned to Claude's gpu0 lane agent, which died.
- Codex saw the gap at 23:25Z and wrote at 00:17Z that "root did not change the gate". It neither did the read nor escalated.
- D3d depends on this cache, and D3d is the stage that fixes the ungrounded vision answers S8 found.
- The verifier's own read of the j3 smoke: the teacher is grounded (OCR is correct, the video description is correct). But the boxes come out in Qwen's 0-1000 frame while the corpus references are in pixels. Whether build-vl handles that is UNVERIFIED.

**B4. MEDIUM: the H3 thermal gate was lifted in notes only.**
- Cyril withdrew it at 2026-10-13 14:26:34Z.
- `gpu6_h3_thermal_ruling.txt` is still absent (14:01Z), and the H3 production caches (2,048 targets) were never built. Only Codex's test-scale H3 bf16 caches exist, under `$OC/distill/codex-preparation-01a1326f-20261013/`.

**B5. MEDIUM: shared infrastructure was hot-patched mid-campaign without review.**
- The GPU allocator was hot-deployed three times. D1's attestation path now depends on it.
- The Relay dispatcher was patched.
- These patches have not been shown to cause harm, but they are unreviewed dependencies on the critical path.

**B6. LOW: the plan was forked without canonical bookkeeping (S01).**
- Codex used a successor D1 dataset: 979 rows, 917 fit / 62 held, 3,752,131 / 250,280 real tokens, dataset sha `74f79199…`. The preregistration specifies rows [0,977), 916/61.
- It replaced the K3-before-D1 gate with "no live K3 process + MemAvailable ≥ 100 GiB + `--no-pin`" (`resource_policy.json`).
- The amendment exists only in Codex's successor copy, `PREREGISTRATION_D1_SOURCE_SUCCESSOR_2026-10-13.md` (sha `7d2b8601…`, written 19:12Z, before the first update). The canonical prereg still reads "Amendments (none)".
- The science is intact:
  - The trainer is Claude's `d1_trunk_recovery` 1.2.4 plus 40 lines of wrapper and receipt code (now labelled 1.2.8-1.2.10).
  - Every hyperparameter matches the prereg.
  - D1 has no K3 term by design.
  - The frozen-body pin `c24d21e0…1370` still validates.

**B7. LOW: clutter and root disk.**
- About 31 G of Codex scratch sits on root, which is at 96% with 71 G free. That includes 4 speech models.
- 87 `codex-*` containers are left (85 exited, 1 Created).
- A stray local directory `…/d1-clone-preparation-v1/clone-receipt.json/` exists.
- The 28.28 GB of reclaimable Docker space is not Codex's.

**B8. LOW: a parallel lineage left on Lexar.**
- Leftovers: `d1-candidate-c36/c37`, `d1-continuation-source-*`, `d1-readback-*`, and `deltas/OC02-D-…-OMNI1D1-*`, which hold only `d1.marker.json` files. About 217 MB in total.
- These are live recovery state for Codex's D1, so they must not be deleted yet.

**Gaps that are not Codex's fault, but that nobody closed**
- The step-1 results report (S9 report) is still not written.
- gpu0 j2 P2 failed both cells with `torch.OutOfMemoryError` (L4096-mb8 and L32768-mb1, rc=1, 0 steps, host RSS 41.96 GiB; `gpu0.status` lines 10-11).
- gpu4 P11 returned `hit=0 exact=0` in every cell. The model emits `</think>` first and then gets 6 of 7 digits (a likely scoring artifact, UNVERIFIED). The fla recurrence has never been parity-tested in fp32.
- The gpu6 must-fix findings (thermal guard, frozen-tool keying, frozen probe) exist only in the uninstalled `gpu6.sh.new`. Relaunching the live 1.0.0 queue with `START_AT=b2` would reuse the frozen 1.2.1 tool and `--resume` from the failed run's checkpoints.

### 2.3 Corrections to earlier records, and claims checked and not botches

**Not botches**
- The S9 clone is correct.
- OMNI1 is untouched: file times 2026-10-12 21:09-21:27 MDT, same size.
- No processes of Claude's were killed.
- The speech work was requested by Cyril.

**Corrections to Claude's stop-time records**
- **d3:** all three must-fix findings were already fixed in `d3_hybrid_module` 1.0.2 (09:29:18Z, fix agent ad3ee305):
  - detached `head_rows`
  - `with_text_suffix` plus a refusal on zero-gradient groups
  - `wait_for_ram`

  Tests: 35 passed. The queue ran parity PASS, d3b PASS and d3d PASS, the last at 6.5 tok/s.
- **trainer-prep:** all three findings were fixed (r27):
  - K3 pgrep plus a PASS check
  - flock on gpu0/gpu4
  - a weight-moving run refused without `--preregistration`
- **export:** fixed in E3 1.0.1 (18 tests OK).
- **p40:** the QVL parent-reference fix was executed (QVLREF CHECK rel_fro_max 9.98e-3 against a 2e-2 gate; CHECK qvl-vit PASS at 11:24:41Z).
- **S8 text:** it was finished, not running, when Claude stopped. The process still running was the S5 CPU fp32 proxy, which exited 0 at 11:33:26Z.

**Rejected suspects**
- **S03, "bootstrap expires 2026-10-15 04:19Z":** this belonged to the dead v31 container. The c36, c37 and r4 containers sleep 259,200 s (72 h). For r4, the pid 3840343 sleep started about 13:47Z, so it ends about 2026-10-17 13:47Z.
- **S05, "loss rising with depth":** the held relL2 is cumulative across the chained prefix. Every layer improves its own held error by 3.5-16%, and the per-layer increment is shrinking (+0.0203 → +0.0110 from L07 to L10).
- **S06, "D1 ran below its 60 GiB disk floor":** false. `disk_floor()` gates both startup and every rung.
- **S14, "K3 L2 results conflict":** they do not. Claude's bisect used the production per-group geometry; Codex's probe used whole-corpus geometry. Both agree where they overlap.

---

## 3. Each card now (snapshot 2026-10-14 14:01:16Z)

| Card | Running | Health |
|---|---|---|
| GPU0 RTX 3090 | D1 r4: container `codex-d1-native-01a10f78-oct14-c37-recovery-r4` (b86cb43bc8ca, Up 13 min), pid 3845549 `d1_allocated_entry.py --run`. 8,926 MiB, 52% util, 60 C. | **Working but fragile.** Re-running finished layers. The log holds only the startup line (13:51:55Z, `ids 979x4096 (fit 917 rows, held 62)`). No r4 rungs yet. `runtime-status.json` says `completed:false`, started 13:49:09Z. Container memory.current is 89,160,536,064 B (about 83 GiB of the 128 GiB cap). The lease `live.json` was captured at 14:01:08Z with `docker_live:true`; I did not see the valid flag (truncated). The Windows monitor is PID 5160, owned by the Codex cloud thread. |
| GPU4 RTX 3090 | Same D1 r4 run. 8,917 MiB, 51%, 68 C. | Same as GPU0. |
| GPU1 P40 | Idle (5 MiB). Codex's `codex-p40-native-baseline` was on GPU1-2 at 13:41Z and was gone by 14:01Z. | Idle. Claude's p40 queue ended 2026-10-14 11:08:44Z. |
| GPU2 P40 | Idle. | Idle. |
| GPU3 P40 | Idle. | Idle. d3 queue ended 10:51:11Z. |
| GPU5 P40 | Idle. | Idle. |
| GPU6 RTX 8000 | Idle since 2026-10-13 23:13:53Z. | **Wasted critical-path card.** K3 and H3 are both blocked. |

**Host and disk**
- Host RAM: 251 GiB total, 92 used, 159 available.
- Disk free: `/` 71 G (96%), `/mnt/lexar` 646 G (84%), `/mnt/ai_data` 318 G (93%).

**Stale processes**
- pid 4169670, a Codex `k3_capacity_host_loop_proc_race_v4r9.py` from 2026-10-09, asleep with 6 MB RSS. Leave it; it belongs to a Codex runtime.

---

## 4. Teacher caches

| Cache | State | Evidence / note |
|---|---|---|
| qtext rescore (selfgen) | **Sealed**, 1,008,982 positions, 5/5 shards verified | gpu0 j1 |
| qtext group caches | **Sealed**: 19 at 1,048,576 positions + 1 at 77,824 + 4 at about 498K | 26 qtext manifests, all finite |
| qlong | **Sealed**, 1,000,609 positions, 31 shards | gpu0 j5 |
| qtext VL smoke (j3) | Done, rc=0; **not signed off** | gpu0 queue.log line 30 |
| Vision-in / video-in answers (j6) | **Not started**: gate file absent | B3 |
| K3 2.0M (4 passes) | **Not started**: B2 parity FAIL (p99 gate) | B2 |
| H3 velocity + block-25 pairs (2,048 targets) | **Not started**: ruling file absent | B4. Codex test-scale bf16 H3 caches only, not production |
| QIMG TE (1,991 texts), VAE, velocity (256 shards) | **Sealed** | gpu4, 13:03Z 10-13 |
| ACE DiT velocity | **Sealed** 14:42:14Z 10-13 | teacher_cache/ace |
| ACE LM codes | **Sealed**, CHECK PASS, 6,667 shards, 2,000,100 code tokens | p40 |
| HY3D trajectories | **Sealed**, CHECK PASS, 256 shards | p40 |
| QVL32 ViT features | **Sealed**, 7,492 shards (11:12Z), plus `qvl32_vit_ref`; CHECK PASS 11:24:41Z | p40 |
| P2 engine rate | **Invalid**: out of memory in both cells, no rate | gpu0 j2 |
| P11 OMNI1 needle baseline | **Invalid / suspect**: hit=0 everywhere; `</think>` artifact; fla parity not tested | gpu4 |
| Codex tool-training cache (168 rows) | Sealed, but outside the design and not reviewed | codex-forensics |

The Lexar `teacher_cache` directory holds `ace hy3d qimg qtext qvl32_vit qvl32_vit_ref` (ls at 14:01Z). There is no `k3` or `h3` directory.

---

## 5. Ordered repair and resume plan

**Principles**
- **Do not stop D1 r4.**
- **One owner per lease.** Coordinate with Codex before touching anything Codex owns.
- **Keep GPU6 busy from now on.** Together with D1, it is now the critical path.

**Critical path:** D2 needs a sealed D1 **and** a sealed K3 2.0M cache **and** a P2 rate (to choose the engine).
- **D1:** if it survives, ≈ 2.6 h of re-running plus 53 × 23-25 min ≈ 23-25 h from now. If the monitor keeps dying, it never finishes.
- **K3:** ≈ 1-2 h of diagnosis, about 5 h of B2, and 20.6 h of passes (design planning figure, unmeasured at production geometry). That is ≈ 27-28 h if started within the hour, so K3 now gates D2 as much as D1 does.

### Step 0. Ownership and coordination
- **Action:**
  - Cyril names one owner for D1. The Codex cloud thread `01a10f78` currently owns r4 and Windows monitor PID 5160.
  - Until he does, Claude does not touch the r4 container, its lease directory, `remote_runtime.py`, or PID 5160.
  - Send Codex a Relay note listing what Claude will take (GPU6, the P40s, bookkeeping, the CPU candidate build) and what it leaves alone (D1 r4).
- **Card:** none.
- **Duration:** minutes.
- **Stops:** nothing.

### Step 1. Bookkeeping (CPU)
**1a. Thermal ruling file**
- Write `$OC/distill/queues/gpu6_h3_thermal_ruling.txt` with Cyril's verbatim withdrawal ("[Cyril quote withheld]", 2026-10-13 14:26:34Z, Codex root thread `01a1326f`, rollout `…2a64a706fedc`). Add the matching line to `STAGE_D_RULINGS_2026-10-12.md`.

**1b. Preregistration amendment (S01 repair 1)**
- In both the server and Documents copies of `PREREGISTRATION_OMNI_STAGE_D_v1.md`, edit only below "## Amendments". Add a dated entry recording:
  - the successor D1 dataset: sha `74f79199…`, manifest `a488e75b…`, 979 rows, 917/62, 3,752,131/250,280 tokens; successor doc sha `7d2b8601…`;
  - the K3-before-D1 gate replaced by no-live-K3 + MemAvailable ≥ 100 GiB + `--no-pin`.
- Then run `stage2_training_receipt.py check-prereg`. It must print frozen-body sha `c24d21e0…1370`, length 15339.

**1c. Retire the stale launcher**
- Mark `queues/d1_launch.sh` and `d1_launch.status` as superseded, and make the launcher refuse `--run`. This prevents a second `OMNI1D1` from the 977-row prefix.

**1d. Step-1 results report**
- Write it from the evidence in §1 and §2.3: S8 text, S5 proxy, media decode tables, QVL. This answers Cyril's 13:39Z question.

**Step 1 summary**
- Card: CPU.
- Duration: about 1.5 h in total.
- Preconditions: none.
- Stops: nothing.

### Step 2. Fix and restart GPU6 K3 (critical)
**2a. Diagnose the geometry**
- **Tool:** frozen `k3_teacher_cache` 1.2.1 (sha `7899023c…`) with the frozen probe from `gpu6.sh.new`.
- **Action:** run L0-L2 at production row-group geometry three ways: (i) as is, (ii) with no padded rows, (iii) as a single group. Compare against the fold reference hashes `b0a9b92d…`/`48afb2e4…`/`99eb1700…`.
- **Outcomes:**
  - If (ii) or (iii) restores L2 equality, the cause is padding or state handling. Fix it in k3_teacher_cache 1.2.4 with a test, and close the "padded pass-token guard" finding.
  - If only (iii) restores it, the difference is bf16 batch numerics. Then re-baseline B2 against an fp32 reference computed at the production geometry and record a ruling, citing Cyril's 2026-10-06 rule that finite fit numbers are diagnostics. Do not loosen gates silently.
- **Duration:** ≈ 1-2 h. Codex's L0-L2 probe took 12 min; three arms plus setup.

**2b. Install the fixed queue**
- **Action:** install `gpu6.sh.new` 1.1.0 (selftest: 7 passed), keyed to the new frozen tool sha.
- **Required:** a fresh B2 state directory. Never `--resume` from `parity/k3cache-2026-10-13-b2-gpu6`.

**2c. Rerun B2 parity**
- **Duration:** ≈ 5 h. The last run went from 08:48:39Z to 13:51:08Z.
- **Gate:** all six gates pass.

**2d. K3 passes 0-3**
- **Volume:** 2.0M tokens, four passes of ≤ 0.5M.
- **Duration:** 20.6 h planning figure (UNVERIFIED at this geometry).
- **Host-RAM guard:** start each pass only if MemAvailable ≥ 90 GiB.
  - D1's container is capped at 128 GiB (now 83 GiB used), and the design puts a K3 pass at about 72 GB of host state.
  - The bisect measured RSS of only 30-32 GB, but that was at the bisect scale.
  - If the guard trips, wait. Do not kill D1.

**2e. H3 production caches**
- **Action:** run them right after K3 (or between K3 passes if the guard is waiting), under the ruling file from 1a.
- **Targets:** 2,048. Drop to 1,024 only if P4 shows more than 20 s per forward (rulings).
- **Duration:** ≈ 3.3 h.

**Step 2 summary**
- Card: GPU6.
- Preconditions: 1a for 2e; nothing for 2a.
- Stops: nothing (GPU6 is idle).

### Step 3. Release the VL answer cache (j6)
**3a. Read and fix**
- Read the j3 smoke receipt (`qtext-smoke-vl-27b-gpu0-v104-2026-10-13.json`).
- Fix the box coordinate frame: convert reference boxes to Qwen's 0-1000 frame in build-vl, or drop teacher-forced box references. Add a test.
- Write `gpu0.vl_go` with the read recorded. Claude may make this call under the standing GO.

**3b. Run j6**
- **Tool:** `qtext_teacher_cache` 1.0.5 (sha `f2465a11`).
- **Fix first:** the two hard-kill tests are flaky in full runs (they pass alone). Fix or quarantine them before the run.
- **Card:** GPU6 after K3+H3, or GPU0 after D1 seals, whichever is free first.
- **Slack:** j6 feeds D3d only, so it is not on the D2 critical path.
- **Host RAM:** `HostResidentReader` holds about 54 GB of host RAM. Never run it alongside a K3 pass.
- **Duration:** UNVERIFIED (not measured). 5,896 images + 512 clips.

### Step 4. Harden D1 (CPU now; switch only at a safe point)
**4a. New pinned candidate**
- Build it from Codex's c37 candidate (trainer `d1_trunk_recovery` 1.2.10 lineage; hyperparameters unchanged):
  1. Run the lease monitor **on the server** (`systemd-run --user` or nohup), with a local `docker inspect`, local lease-directory writes, and PG at localhost:25490 with the credential from the vault via the environment. No Windows and no SSH in the loop. Stop redirecting stdout.
  2. Set `max_proof_age_seconds` to about 600. Retry any transport or PG error with backoff. Revoke only when the container or PG membership is actually gone. In the trainer, wait up to about 30 min on a stale proof instead of raising.
  3. After each committed rung, write a **boundary activation cache** (next-layer input for fit and held rows) to `/mnt/lexar/omni-campaign/scratch/`, keeping only the latest. Resume loads it after verifying the rung-manifest sha, with no re-running. Measure the shard size first; the estimate of about 41 GB is UNVERIFIED.
  4. Log per-layer RSS and cgroup `memory.current`, and free host buffers after each rung. v35 hit the 128 GiB cap after 9 layers in one process; r4 is at 83 GiB while re-running.
  5. Never use `--rm` on D1 containers, and write the exit state into `runtime-status.json`.
  6. Keep all observers on the ordinary SSH lane, never `--control`.
- CPU-test it: unit tests plus a fault-injection test that kills the monitor for 5 min and checks the trainer survives.
- **Card:** CPU.
- **Duration:** ≈ 3-5 h of engineering.

**4b. Canonical rung directory**
- Create `/mnt/lexar/omni-campaign/distill/d1/OMNI1D1-rungs-canonical/` and copy L00-L10 into it with `cp -n`.
- Check each file against the rung sha256 in its source `d1_receipt.json`.
- Write `SHA256SUMS`.
- **Duration:** about 10 min. Read-only on the sources.

**4c. Switch over**
- **If r4 dies:** relaunch immediately on GPU0+GPU4 with the 4a candidate from the canonical prefix. It re-runs once (≈ 3 h), then never again.
- **If r4 survives:** let it run. Switch only at a committed rung boundary, and only with the owner's agreement from Step 0. **This step stops a running job (r4).**
- **Expected D1 finish:**
  - Clean run: about 2026-10-15 13:00-16:00Z (r4 starts new training at L11 at about 16:30Z today, then 53 × 23-25 min).
  - Each death adds about 3 h until 4a is in place.

**4d. D1 seal**
- Run `verify_final_delta_c36.py` (or its successor) and assemble the delta in `/mnt/lexar/omni-campaign/deltas/OC02-D-Q38-27BU-OMNI1D1-<run>/`.
- Apply it to the clone, run the s7 check, and register. This is a deliberate ledger step; the Codex candidate does no ledger mutation.
- Run bank v1 on both surfaces.
- Gate: content ≥ OMNI1 S8, and R-Q top-1 ≥ S8, against the successor held split.
- **Duration:** ≈ 3.1 h (design).
- **Card:** GPU0+GPU4.

### Step 5. Make the D2 engine decision possible
**5a. P2 OOM fix (CPU now)**
- Find the out-of-memory cause (mb8 at 4k and mb1 at 32k both failed with 0 steps). Prepare an mb4 cell and a checkpointing cell.
- **Duration:** about 1-2 h.

**5b. P2 rerun**
- **Card:** GPU0 (3090, as the T(ii) engine requires), straight after the D1 seal, in parallel with the D1 review.
- **Rule:** use T(ii) if it reaches ≥ 151 tok/s; otherwise use the split cell (GPU6+GPU4). Use no depth cut; cut tokens only below 60 tok/s.
- **Duration:** ≈ 1 h (UNVERIFIED).

**5c. P11 fix**
- Strip or score past the leading `</think>`, or use the no-think template.
- Add an fp32 fla-vs-reference parity arm.
- Rerun the OMNI1 baseline on GPU6 (between K3 passes, if the RAM guard allows) or on a 3090 after D1.
- **When needed:** before the D2 seal, not before the D2 start.

### Step 6. D2 → D3 → D4 → T → export (design unchanged)
- **D2:**
  - Starts when D1 is registered, K3 is sealed and P2 has a verdict.
  - Text + K3 LoRA KD, 23.0M + 1.0M long tokens.
  - On-policy sampler per the rulings: Q8_0 on GPU6 on the upside path, Q4_K_M on the P40s on the planning path.
  - Duration 28.1 h upside, 59.7 h planning (design).
- **D3:**
  - Needs every media cache, including H3 (Step 2e) and j6 (Step 3). The D3 module 1.0.2 is ready.
  - Duration 23.1-30.6 h upside (design).
- **D4:**
  - Consolidation and the single merge.
  - Duration 23.7 h upside (design).
  - Root must have ≥ 60 G free; see risk R4.
- **Stage T, then a single Q4 export:**
  - E3 1.0.1 is ready.
  - The export must keep `--ledger-band-logit-mask`, or the band rows must be rescaled first (risk R6).

### Step 7. Cleanup (needs Cyril's go-ahead; after Codex is idle)
- Check references first.
- Move the speech parents to Lexar if Stage T/D3 keeps them; otherwise remove them.
- Remove duplicate prep iterations (geometry-flow v1-v4 and the like) and the exited `codex-*` containers.
- `docker builder prune` (28.53 GB) and pruning the non-Codex containers each need a separate go-ahead.
- Do not touch the Lexar D1 trees until D1 is registered.
- Frees about 15-31 G on root.

### Step 8. Parallel track: folding optimization
Cyril asked for this at 14:41Z, but said "fix training first".
- Use the idle P40s (GPU1/2/3/5) for multi-token-objective fold experiments on the 4B ladder (OC01 parents are registered).
- Grade behaviour first: frozen review plus a coherence check. Do not gate on token match.
- This must not touch GPU6 or the 3090s until D4.

### Step 9. Speech
- Review Codex's speech trees (`codex-speech-1p7-comparison-modality-v1`, `codex-speech-preparation-modality-01a1326f-v1`, `codex-joint-audio-preparation-modality-01a1326f-v1`) as the start of the audio-in/speech lane. The design currently lists that lane as a teacher gap with a stage-T path.
- Decide whether speech enters D3 or T.
- Their content is UNVERIFIED by this review.

---

## 6. Open risks

1. **D1 monitor deaths.** Until Step 4a is live, any SSH stall kills D1, and each death costs about 3 h of re-running. With deaths every 16 min to 3 h, D1 may never finish on the current wrapper.
2. **D1 out-of-memory.** r4 is at about 83 GiB of its 128 GiB cgroup while still re-running. v35 was OOM-killed after 9 in-process layers. If memory grows per layer, r4 may die around L18-L20 in-process. This is UNVERIFIED; watch `memory.current`.
3. **Host-RAM contention.** K3 (≈ 72 GB per pass, design figure), D1 (up to 128 GiB) and a HostResidentReader j6 (≈ 54 GB) cannot all run at once on 251 GiB. Steps 2d and 3b serialize with a MemAvailable guard.
4. **Root disk.** 71 G free (14:01Z). The design's root peaks are D2 ≈ 80 GB and D3 ≈ 120 GB, and D4 needs ≥ 60 G free for the merge. Redirect `x_d`, caches and scratch to Lexar (646 G free), or do the Step 7 cleanup before D2.
5. **The K3 B2 cause is unknown.** If it is a real padding or state bug, any K3 numbers that used per-group geometry are suspect. If it is only bf16 numerics, a recorded re-baseline is needed. Either way, D2 waits on it.
6. **Music band in the head.** It wins 146 of 210 unmasked argmax positions. D2-D4 losses are restricted to rows [0, 248,320), so nothing trains the band down. Any inference or export path that forgets `--ledger-band-logit-mask` will emit music codes.
7. **Two agents on one campaign.** The Codex cloud thread is still active, owns D1 r4, and hot-patched the shared allocator, which D1 attestation now depends on. Unreviewed patches and parallel lineages (the successor dataset, `d1-*` trees) can diverge from the canonical records again. Step 0 is a prerequisite.
8. **VL box frame.** If build-vl distils pixel-frame reference boxes against a 0-1000-frame teacher, D3d will learn off-frame grounding. Fix before j6.
9. **P11 baseline invalid.** The D2 and D4 256k-needle gates compare against OMNI1's P11. A corrupt baseline makes those gates meaningless. Fix before the D2 seal.
10. **Flaky qtext tests.** Two hard-kill resume tests fail in full runs and pass alone. Treat the crash-resume path as not proven until they are fixed.
11. **The allocator container will expire.** r4's 72 h bootstrap ends about 2026-10-17 13:47Z (pid 3840343 `time.sleep(259200)`, started about 13:47Z). This is not a near-term risk, but deaths stack up toward it.
12. **Stale Codex process.** pid 4169670 has been asleep since 2026-10-09. It is harmless, but it is listed as U8/B3 in wave 1. Leave it to Codex or Cyril.

---

## Evidence index
- **Server snapshots (this author):** `date -u` printed "Wed Oct 14 02:01:16 PM UTC 2026". Includes nvidia-smi, docker ps, the r4 `runtime-log.txt`/`runtime-status.json`/lease, the queue `.status` tails, the gate `ls`, the teacher_cache `ls`, df, free, ledger `wc` and registry `ls`, ps, the gpu6 b2/bisect logs, the gpu0 P2 lines, and r4 `memory.current`.
- **Reader scratch:** `C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462/scratchpad/cx/` (forensics; start with `triage2.txt`) and `…/scratchpad/tt/` (tool sha tables).
- **Codex D1 evidence:** `C:/Users/cereal/Documents/Codex/2026-10-05/task-4/oct14_real_qualification/d1_owner_*`.
- **Design, rulings and prereg:** `distill/DISTILL_DESIGN_2026-10-12.md` (lines 131, 169-172), `distill/STAGE_D_RULINGS_2026-10-12.md`, and `distill/PREREGISTRATION_OMNI_STAGE_D_v1.md`.
