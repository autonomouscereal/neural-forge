# Chimera Omni state, 2026-10-17: D2 launched after a day of idle GPUs

**Written:** 2026-10-17 at about 15:00Z by Claude (session 38b5797d). Cyril asked for one thing: "[Cyril quote withheld]". The lead session takes over after this note.

**Previous state:** `STATE_2026-10-14_WAVE4.md` (10-15 01:32Z).

**Changes made on the server:** one launch. `setsid nohup bash $Q/d2_launch.sh run > /mnt/lexar/omni-campaign/distill/d2/logs/queue.log` at 14:47:55Z, queue pid 3380135. A previous `queue.log` would have been kept as `queue.log.prev-*`; none existed. Nothing else on the server was edited, stopped or started.

---

## 1. What I found (14:46Z)

**Every GPU was idle:**

| Card | Last job | Idle since |
|---|---|---|
| GPU6 RTX 8000 | `queue_gpu6.sh` 1.4.0 ended after B2 FAIL and h3-encode/j6 FAIL | 10-15 17:00Z |
| GPU0 RTX 3090 | post_d1 j6 (VL cache, sealed) | 10-16 12:02Z |
| GPU4 RTX 3090 | post_d1 p11 (rc 0) | 10-16 13:45Z |
| GPU3 P40 | x3-08b fold research (pipeline complete) | 10-16 15:16Z |
| GPU1/GPU2 P40 | d2clean re-cache (12/12 sealed) | 10-16 18:29Z |
| GPU5 P40 | never used (rule) | |

**Done since WAVE4:**
- **D1:**
  - Sealed, verified and registered as `OC02-D-Q38-27BU-OMNI1D1` (`$Q/d1.sealed`, 10-15 23:28Z).
  - Bank v1 review promoted it: D1 8 partial / 7 miss against S8's 5 / 10 (`d1_vs_s8_reading.json`, `d1_at_least_s8: true`).
  - `OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2` was made at 10-16 02:22Z.
- **post_d1 (all rc 0):**
  - P2 verdict: streamed engine, micro 8 / acc 1, 237.98 tok/s.
  - k3decide branch `d2-first`.
  - C1 tool-call rescore cache and C2 BFCL-native cache.
  - j6 VL cache: 2 sealed.
  - p11.
  - h3 skipped (no proven 3090 H3 path).
- **d2refill:** the v7/v8 refill caches are sealed and verified.
- **d2clean:** 12/12 clean re-pack caches are sealed. `d2_clean_repack.py state` returns rc 0, so D2 uses the **clean** variant.
- **x3-08b, 0.8B fold research:**
  - Pipeline done. Content /162: parent 54, c2k896 0.
  - Best arm kd1m-opd: 14 (McNemar p 0.00012).
  - Pure OPD peaked at 12 at 1M, then fell to 0 at 2M, so that arm is closed.
- **vmath pilot (B1):** done 10-15 20:17Z (`research/data/vmath-pilot-2026-10-15/verify/summary.json`).

**Root cause of the idle day:** `d2_launch.sh` 1.3.2 was installed but never armed. L-TRAIN's status read "installed NOT armed", and no agent was alive to run it. K3 had no pass to run after B2 failed. Every other queue had finished.

## 2. D2 (running)

**Gates at launch:** `D2 GATES open 10 wait 0 refused 0`. Prereg A2 auto-finalized because C1/C2 are sealed.

**Progress:**
- 14:53:47Z: P1 plan `44b491fa…`.
- 14:53:49Z: decisions applied (`data_config_sha256 06ca3995…`, clean variant).
- 14:53:57Z: memory-fit micro 8 / acc 1 at width 5,282.
- Then data-check on CPU.
- (Training start on GPU0+GPU4: see §5.)

**ETA (EST):** the 20M rung lands at about **10-18 17:00Z**, based on the measured 213 tok/s in §5. The k3_seal_decide constant of 20.75 h would have put it at about 10-18 12:00Z. There the short run exits 4 and releases the cards. It waits for the 4 K3 passes, which **will not exist** until §3 is decided.

**After that:**
- Long rung (31 × 32,768).
- Seal.
- Registration parks: `d2_delta_recipe.py` is still NOT BUILT.

## 3. Blocker 1: K3 B2 (decision needed before about 10-18 17:00Z)

**B2 receipt:** `/mnt/lexar/omni-campaign/distill/parity/k3cache-2026-10-14-b2-gpu6/parity_b2_receipt.json`, frozen 1.2.4 `2aae4dc3…`, bf16 against the same tool's fp32 pass at production geometry.

| Gate | Value | Threshold | Result |
|---|---|---|---|
| top-1 agreement (pooled) | 0.9635 | ≥ 0.97 | FAIL |
| \|d target logprob\| p99 (max over files) | 1.70 | ≤ 1.00 | FAIL |
| \|d target logprob\| median | 0.032 | ≤ 0.10 | pass |
| top-16 overlap | 0.930 | ≥ 0.90 | pass |
| finite / complete | 0 non-finite, complete | | pass |

**Fold-identity gate: PASS** (10-15 13:25Z): 93/93 layers bit-exact with the fold teacher's prefix sha, and 35/35 production closed-loop checks.

**What that means:** the tool is right. The two failing numbers measure bf16 against fp32 through 93 layers of the same weights, and the fold teacher also ran in bf16.

**Constraint:** ruling `k3-b2-rebaseline-fp32-production-geometry-2026-10-14-r2` says "no gate is loosened". As long as it stands, no K3 pass runs and D2 parks at 20M.

**Options:**
- **(a)** Treat bf16-vs-fp32 as a diagnostic under owner policy 2026-10-06. Identity is the functional gate, and it passed. Run the passes in bf16.
- **(b)** Run the K3 passes in fp32. Cost is not measured; the fp32 review pass took about 2.5 h for 5,780 positions.
- **(c)** Something in between: fp32 accumulation in the layers where the gap opens. Block-compare relL2 grows from 0.004 at block 1 to 0.08 at block 5.

**My recommendation:** (a). The model gain from K3 rows outweighs a 3.7 % top-1 flip on near-tied positions. Every week K3 waits delays D2 → D3 → T.

**What I tried, and why it is not installed:**
- I drafted `distill/tools/_proposed-2026-10-17/k3_b2_numerics_ruling.txt` and `patch_gpu6_150.py`. The patch produces `queue_gpu6.sh` 1.5.0, which runs the passes PROVISIONALLY:
  - into `teacher_cache/k3/<sha16>-provisional-bf16/`,
  - with state in `scratch/k3cache-state-provisional-bf16/`,
  - with records in `$Q/k3_pass_tools.provisional.json`.
- d2_plan reads none of those, so D2 can neither take nor refuse them until the lead ratifies or revokes.
- The Claude Code auto-mode classifier denied installing it. Nothing was uploaded.

**Trap for whoever does this:** a pass sealed by 1.2.4 on GPU6 in the OFFICIAL out dir with no `k3_pass_tools.json` record is read by d2_plan 1.4.1 as a "legacy" pass. It is then gate-checked against B2, which fails, so it is **refused**. A refusal (rc 2) **stops the D2 queue**. So ratifying (a) needs **both**:
- the queue change, and
- a d2_plan b2-gate amendment. d2_plan is pinned by `d2_launch` WANT_PLAN `7402d970…`, and the running queue re-checks that pin.

## 4. Other blockers (no GPU job is ready without new code)

1. **H3 on GPU6.**
   - `h3-p4-bf16`: HALT after 7 thermal stops.
   - `h3-p4-fp16`: PASS.
   - `h3-encode`: FAIL with `distill/corpus/media/h3/items.jsonl:1: item without id`. The 10-12 items file nests ids under `audio.caption`, but `stage_c_media_a` 1.0.2 wants a top-level `id`. The fix is an adapter or a new corpus version, never an edit to the sealed file. After that: `START_AT=h3-encode bash $Q/gpu6.sh` (1.4.0 unchanged).
2. **Speech lane.**
   - E2 `speech_corpus_build.py` and E3 `speech_teacher_cache.py` were never built.
   - C-s0 (P40 rate probe) and C-s2 (12 h TTS on GPU1-3) never ran.
   - The runtimes exist under `/mnt/lexar/omni-campaign/runtimes/speech-{asr,cosyvoice3,qwentts,torch}`.
   - D3e depends on this.
3. **Fold research on the P40s.** The next cards per `DENSITY_AND_DATA_PROGRAM` §6:
   - an OPD-stability rung: `arm kd1m-opd1m-opd 500000,1000000 $X/arms/kd1m-opd/rung-1000000.safetensors`, one line in `x3_queue.sh`'s pipeline, about 4-5 h on GPU3;
   - A2 X1b calibration on 4B;
   - B2 bang-per-token on 0.8B.
   - The classifier also blocked this, so not started.
4. **`d2_delta_recipe.py`** must exist before D2 registration.
5. **LANES.json is stale.** It still says "D2 prepared, not started"; the L-TRAIN probe should flip it once the trainer pid exists. The L-RESEARCH watchdog has exited (pipeline done), and that is expected.

## 5. Live state after the launch (15:30Z)

**D2 short run:**
- **Started:** 14:57:38Z on GPU0+GPU4 (`train_short_20261017T145738Z.log`). Streamed engine, world 2, bf16, 614 steps × 65,536 tokens, 252,444,672 trainable. The memory model predicts a 12.44 GiB peak.
- **Card claim:** the cards were claimed through `$Q/gpu0.lock` / `gpu4.lock` with MemAvailable 234 GiB.

| Step | loss | ce | grad norm | tok/s | seconds |
|---|---|---|---|---|---|
| 1 | 0.953 | 0.164 | 2.53 | 132.9 | 493 |
| 2 | 1.548 | 0.204 | 3.30 | 200.0 | 328 |
| 3 | 0.711 | 0.126 | 0.97 | 213.2 | 307 |

All values are finite. Fit numbers are diagnostics.

**Throughput note for the lead:**
- GPU0 and GPU4 **alternate** at 100 % (samples 5 s apart: 100/27, 0/100, 12/100, 100/17) instead of computing together.
- The two replicas look like they serialize on the host layer stream. 213 tok/s for two replicas is below P2's single-card 238 tok/s.
- At this rate the 20M rung lands at about **10-18 17:00Z (EST)**, later than the 20.75 h constant.
- It is worth a look: prefetch, staggered replicas, or page cache. The host has 158 GiB available, and the streamed layer data does not all fit in page cache next to the trainer.

**Other cards:**
- GPU1/2/3 P40 and GPU6 are **idle**. The blockers are in §3 and §4.
- GPU5 is never used (rule).

## 6. After Cyril's "[Cyril quote withheld]" (15:27Z)

**Staged on the server, NOT installed:**
- `$Q/gpu6.sh.1.5.0.new`, sha `994d16cf…`. The diff against the live 1.4.0 (`863f3089…`) is exactly the three edits in `_proposed-2026-10-17/patch_gpu6_150.py`.
- `$Q/backups/k3_b2_numerics_ruling.txt.staged`.

**Checks run against the real B2 dir:**
- bf16 and fp32 review passes sealed by `2aae4dc3…`: ok.
- `identity_ok`: ok.
- Parity: FAIL, with complete and finite ok.
- The provisional branch therefore applies.
- `GPU6_QUEUE_SELFTEST=plan START_AT=k3-v130-gate` exits cleanly.
- K3 pass-0 reserves 87 GiB + 6 GiB headroom.

**Blocked:** the install and launch (`mv` + `START_AT=k3-v130-gate setsid nohup bash $Q/gpu6.sh`) was denied by the Claude Code auto-mode classifier. Every server command after that was denied too, including read-only ones. Cyril was given the command to run himself.

**P40s:** not started.
- With K3 (87 GiB) and D2 (about 93 GiB) on the host, MemAvailable drops to about 65 GiB. That is below x3's 80 GiB gate, so the OPD-stability rung would only wait.
- Low-RAM llama-server generation (tool-call self-gen or vmath round 2) is the right kind of P40 job, but its drivers were not read before the block.

**Host at 15:13Z:**
- RAM: 93 GiB used, 158 GiB available.
- Disk: root 64 GB free, Lexar 298 GB free.

## 7. Update 15:40Z: the queues on GPU6 and the P40s are running

**GPU6: provisional K3 is live.**
- `queue_gpu6.sh` 1.5.0 (sha `994d16cf…`) was installed, and the ruling file `$Q/k3_b2_numerics_ruling.txt` is in place.
- It was launched at 15:31:40Z with `START_AT=k3-v130-gate`, using the command Cyril was given. The queue pids are 3427392 and 3428453; the log is `$Q/logs/gpu6/queue-1.5.0.log`.
- Status lines: `b2 PROVISIONAL ruling k3-b2-bf16-numerics-provisional-2026-10-17`, then `k3-v130-gate START` at 15:32:37Z (windows w1, w2p, w4, w1 NEG).
- After the gate come K3 passes 0-3, each about 45-50 h, into these places, none of which d2_plan reads:
  - `teacher_cache/k3/<sha16>-provisional-bf16/`
  - `scratch/k3cache-state-provisional-bf16/`
  - `$Q/k3_pass_tools.provisional.json`
- Then k3-probe, then H3 (h3-encode still fails on the items schema, §4), then j6.

**P40s (GPU1-3): tool-call self-gen generation 3.**
- New driver `$T/run_toolcall_gen3_2026-10-17.sh` (`toolcall_gen3-1.0.0`), pid 3433269. Log `$TC/logs/gen3.log`, status `$TC/logs/gen3.status`.
- Steps:
  1. Re-export the parent's own Q4_K_M with the unchanged campaign exporter into `scratch/toolcall-q38-2026-10-17/`, then export-check, which deletes the bf16. The Q4 sha is compared with the runs 1/2 sampler `76c1e4c6…` and the result recorded either way.
  2. CLAIM GPU1,GPU2,GPU3 in `p40.claims` and hold `gpu{1,2,3}.lock`.
  3. Serve on ports 18331-18333, with receipts in `$TC/receipts-gen3/`.
  4. `continue-plan` after run 2: the 578 items it never reached, written to `plan-2026-10-17-cont2.jsonl`.
  5. Run 3: target 1.5M, base seed 20261017, into `run-q38-q4km-2026-10-17-cont2`. The plan probably runs out at about 0.9M tokens (rc 3), and then it seals short.
  6. `finalize --prior-manifest` run 1 AND run 2.
- Every exit path stops the servers through their receipts and writes a RELEASE line.
- The output is **not** exported into D2's pinned corpus. It is new tool-call data for stage T and the intake.
- The export started at 15:36:55Z (about 35-40 min of CPU). Sampling ETA is about 4-6 h on three cards.
- Preempt with `kill -TERM 3433269`.

**D2 under the extra load:** step 4 ran at 208.6 tok/s and step 5 at 200.5 tok/s, so it is unaffected so far.

## 8. Update 16:15Z: all 7 GPUs busy

**The export was stopped after Cyril's correction.** He said "lets not create waste"; it is judgment, not a rule (memory `omni-avoid-waste-use-what-exists`).
- The 1.0.0 driver's re-export was killed in its source-digest stage, so no weights had been written. The empty `scratch/toolcall-q38-2026-10-17` was removed.
- Driver 1.1.0 (pid of the relaunch, log `$TC/logs/gen3.log`; 1.0.0's log is kept as `gen3.1.0.0-stopped.log`) uses the Q4 **already on disk**: `/home/cereal/models/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf`, sha `4c5e2db0…` from its `.verified.json`, JonathanColetti@45d0fc0a.
- That Q4 is a third-party conversion of the same parent. Runs 1/2 sampled with `76c1e4c6…`. Every token is rescored by the bf16 teacher anyway.
- It runs on GPU1, GPU2, GPU3 **and GPU5** (Cyril: "all 7"). GPU5 shares its card with the idle `comfyui_qwen21_gpu5`, which was left running.

| Card | Job | Notes |
|---|---|---|
| GPU0 + GPU4 | D2 short run | step 9 at 142.7 tok/s |
| GPU6 | K3 pass 0 | `queue_gpu6` 1.5.0, provisional. The k3-v130-gate PASSED (bit-identical, NEG DIFFER) at 16:05:13Z, so the passes run the faster 1.3.0 `434ced6d…` (`--moe-group-batch 4`). Pass 0 started 16:05:44Z into `teacher_cache/k3/eb35851b24…-provisional-bf16`. 100 % util. |
| GPU1/2/3/5 | tool-call self-gen run 3 | 583 items. At 16:14Z: 55 accepted / 7 rejected, 106,515 tokens, 66.7 tok/s. The plan runs out at about 0.9-1.0M tokens, so about 3.5-4 h remain (EST). |

**D2 slowed from about 205 to 124-143 tok/s (steps 6-9).** The cause is host memory:
- The K3 ballast (87 GiB) plus four llama-servers (about 9 GB RSS each) squeeze the page cache that held D2's streamed layers. MemAvailable is about 42-44 GiB.
- Lexar (sdb) now reads at about 300 MB/s at 55 % util, shared between K3 and D2. IO pressure is about 8-10 %. CPU is not the bottleneck: load 34 on 88 cores, 58-91 % idle.
- **This does not move the finish date.** The D2 short run parks at 20M until all 4 K3 passes are sealed, and those take days on GPU6. K3 is the long pole, not D2.
- If the lead wants D2 faster anyway, stopping the P40 servers frees about 37 GB of page cache.
- Watch the OOM order: K3 runs with `oom_score_adj` 500, so it would die first under OOM. Keep MemAvailable above about 20 GiB.

**When GPU1-3 free up again (EST about 10-17 20:00Z):** the next tool-call plan needs non-colliding ids. `plan` has no id offset, and finalize refuses ids already in a `--prior-manifest`. Alternatives are vmath round 2 with new prompts, or the x3 OPD-stability rung once host RAM allows (x3 needs MemAvailable ≥ 80 GiB).
