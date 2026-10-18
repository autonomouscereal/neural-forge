# Stage D wave 1: integration status and completeness check

Written 2026-10-13. The live server read was at 07:39Z. The server logs use local time (UTC-6), so a log line stamped `[01:35]` means 07:35Z.

Binds: `DISTILL_DESIGN_2026-10-12.md` (§3, §4, §10) and `STAGE_D_RULINGS_2026-10-12.md`.

Rule for this file: every number below was either printed on the server during this read or is quoted from a lane's final report, and the source is named. Anything I did not see is marked **not verified**.

---

## 0. Your question: did we test our first Chimera Omni at all?

**Not with a full capability test yet. That test started at 07:05Z today and is about one third done.** Several earlier lane answers said OMNI1 had not been built. Those answers were out of date. Here is the state at 07:39Z (step 1's work; read only):

| What | State | Evidence |
|---|---|---|
| Assembly | Done, finished 03:27:11Z | `composite/s7/omni1/logs/c16_omni1.log` ends `end 2026-10-13T03:27:11Z` (quoted in the k3cache report) |
| Structural check (`s7_omni1_check.py` 1.0.0, GPU6) | PASS | `$OC/composite/s7/omni1/omni1_check.json`, verdict: "OMNI1 loads strict; identity vs S5 and accounting hold; text and every charted block's media forward finite; every charted block writes on its own segment and only there" |
| Registration | Done at 05:01:45Z | `register_confirm.log`: record `OC02-F-Q38-27BU-OMNI1-0001-register 3f38c5cf771fad29`, sealed at `$OC/registry/OC02-F-Q38-27BU-OMNI1/…-0001-register.json`, ledger `$OC/ledger.jsonl` sha256 `716581e6…`, `exit 0` |
| S8 media forward | Done, exit 0, 05:57–06:41Z | allocator `s8-media-forward-gpu0-claude.json`; `composite/s8/media_out/s8_media_forward.json` |
| S8 media decode (ACE, QIMG, H3, HY3D) | Done, exit 0, 06:42–06:49Z | `s8_media_decode.ace-qimg-h3-hy3d.json` (s8_media_decode 1.0.0). **I did not read the structural numbers** (rms, faces, frames) |
| S8 vision-in (QVL) answers | Done | `media_out/qvl/answers.json`, details below |
| S8 text frozen review (bank `lfm2_semantic_whole_response_prompts_v1`, 16 cases, thinking off, band mask on) | **Running**: pid 917640, GPU6+GPU4+GPU0, split 6/33/30. At 07:35Z it had finished case 6 of 16 | `composite/s8/logs/launch_omni1.out` → `s8/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json` |

**What OMNI1 actually produced so far** (observations only, not grades; Cyril grades S8, artifact-tolerant):

- **Loading and preflight checks passed:**
  - Loaded "27,535,655,753 parameters in 92.7s".
  - "decode-path preflight: agree=True".
  - "emit-zero floor: passed=True min|h|=137.3895 min logit sd=2.1020".
- **Text cases:**
  - Cases 1 (conversation-natural-greeting) and 6 (context-retrieval) returned **0 tokens**, `(complete)`. The model emitted end-of-turn immediately.
  - Cases 2–5 (instruction, explanation, arithmetic, logic) each hit the 1,024-token cap (`censored-at-cap`) at about 371–402 s per case.
  - Running out the cap does not count against a case under the design's tolerance rules. **An empty answer is a content miss.**
- **Vision-in:** the answers are fluent English but do not describe the images.
  - Aquarium photo: "…the image provided is a text string…".
  - Travel-bag photo: "The image is a text block…".
  - Both images with their features zeroed: "The image is a placeholder. It is a generic, abstract representation of a generic, abstract representation…" (a loop).
  - No image: "…a **2x2 grid** of four squares…" (made up).
  - Flags: `answers_differ_between_images True`, `image_answer_equals_zeroed_answer [False, False]`. So image tokens do reach the stack, but their content is not understood yet. That matches the "noise is the expected S8 state" line in DESIGN §2 G4. D3d is the stage meant to fix it.

**What no Stage D lane did:** none of them loaded or scored OMNI1. They tested the teachers (Qwen parent, K3, media parents), the S5 child (P1 reader parity, seam smoke) and tiny Qwen3.5-0.8B proxies.

---

## 1. Hardware and disk at the read (07:39Z)

- **GPUs:**
  - GPU0 3090: 21,890 MiB. GPU4 3090: 21,603 MiB. GPU6 RTX 8000: 11,328 MiB. All three are held by step 1's S8 review, pid 917640.
  - GPU1, GPU2 and GPU3 (P40s): **5 MiB each, so all three are free.** Self-gen has finished and stopped its servers.
  - GPU5: 5 MiB, reserved, never used.
- **Disk** (`df -BG`):

  | Mount | Free | Used |
  |---|---|---|
  | `/` | 118 G | 94% |
  | `/mnt/lexar` | **44 G** | 99% |
  | `/mnt/ai_data` (NAS) | 345 G | 93% |

  Lexar has fallen from the disk lane's 95 G reading to 44 G.
- **Codex `k3_capacity_host_loop_proc_race_v4r9.py`, pid 4169670, is still running.** Its owner is still unconfirmed (U8).

---

## 2. Per-lane results

**sha256 parity.** I ran `sha256sum` on the server over every `.py` in `$OC/distill/tools/` and compared it with the workstation copies in `distill/tools/`. **All 22 files match.**

### 2.1 qtext: Q-text / Q-long / self-gen rescoring writer, plus T(i) (streamed VL teacher)

- **Tool:** `qtext_teacher_cache.py` **1.0.3**, `9d6c06ed5bbda74597bdfe2df3704d2b456d4c1a72a86868281cfcc977cfb446`.
  - Tests: `test_qtext_teacher_cache.py`, `3cc2ccec62f614a2f06db9e851a6be5d60ccd7e5d65d8a4f3c4149edec5a9f78`.
  - Diagnostic: `qtext_vl_divergence_diag.py` 1.0.0, `bb4ae870…`.
  - Engine: `distill/tps_probe.py`, `449eecc4acf98555…` (unchanged).
  - Launchers: `run_qtext_cpu_v103_2026-10-12.sh` `c552b792…` and `run_qtext_seal_interrupt_check_2026-10-12.sh` `b9e99b5a…`.
- **Tests** (server CPU, `PYTHONPATH=$OC/distill/tools`, `CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src`): **"25 passed in 188.79s (0:03:08)"**. Log: `$OC/distill/logs/qtext_cpu_v103.log`.
- **Parity at 1.0.3** (Qwen3.5-0.8B, fp32, CPU). The gate is the same top-64 ids in the same order. All three arms pass with lse diff 0.0:
  - Text: "positions=6900 topk_ids_equal=True mismatch_positions=0 top1_equal=6900/6900". Receipt `$OC/distill/tps/qtext-parity-text-08b-fp32-cpu-v103.json` (`b02dd3af…`).
  - VL, 2 images: 101/101. Receipt `…-vl-…-v103.json` (`d87157e8…`).
  - Video, 8 frames: 66/66. Receipt `…-video-…-v103.json` (`6a9591c4…`).
  - verify: "1 shards PASS, 0 FAIL" on each cache.
- **Real-teacher interrupt check** (copy of the Qwen3.8-27B-U smoke cache):
  - 1.0.2 returned 2: "teacher digests differ … None".
  - 1.0.3 printed "SEALED … positions=1698" and verify gave "2 shards PASS, 0 FAIL".
  - Log: `$OC/distill/logs/qtext_seal_interrupt_v103.log`.
- **Review findings:**
  - **Fixed:** H1, a killed session made the cache impossible to seal. M1, a crash was logged as allocator "exit 0".
  - **Open (low; not rejected, not fixed):**
    - L1: `--processor-dir` is not part of the locked cache semantics.
    - L2: `store_mask` is indexed by input position, an off-by-one hazard for self-gen rescoring.
    - L3: the head output is rounded to bf16 before the fp32 upcast; this is a recorded fidelity choice.
- **ready_for_stage_c: true** (lane's verdict).
- Production rate on a 3090 (bf16, fla, mb8, L4096) is **unmeasured**.
- No P40 run uses 1.0.3. The P40 parity receipts are still at 1.0.1.

### 2.2 k3cache: K3 teacher cache

- **Tool:** `k3_teacher_cache.py` **1.2.0**, `7e7aca260fdb923c70a75593a3f17c7ffe633d85faa39590f5b6a037ab2c5738`.
  - Tests: `test_k3_teacher_cache.py`, `5fc937550b6b1db85a109b0cfea2cd5b5558ffa27c8480cbfb85facb39715599`.
  - Driver: `run_k3cache_rerun_v120_2026-10-13.sh`, `76414347…`.
  - Engine pin: k3_dense_pass 1.2.5 `fd7dbf8d`.
  - Previous version kept at `$OC/distill/tools/_superseded/`.
- **Tests:**
  - **"17 passed in 27.07s"** (venv-pascal-cu118-torch260) and **"17 passed in 15.95s"** (lab venv).
  - Negative control, the new tests run against 1.1.0: "4 failed, 13 passed in 11.51s".
- **Parity** (CPU re-run on 1.2.0; receipts in `$OC/distill/parity/k3cache-2026-10-12/rerun-1.2.0/`):
  - **Full 93-layer pass: FAIL on 2 of 6 gates**, exit 3. Receipt `parity_receipt.json` (`eddc1ec7…`).
    - review: top-1 0.9617, p99 1.0679.
    - native: top-1 0.9589, p99 1.6092.
    - The failing gates are top-1 (gate 0.97) and p99 (gate 1.0).
  - **D2** (from the fold engine's state at L61): PASS 6/6. Receipt `parity_d2_receipt.json` (`168c6123…`).
  - **D2b**, which finished after the lane report: I read "PARITY PASS gates {… all True}" in `logs/k3cache/parity_d2b_compare.log`. Receipt `$OC/distill/parity/k3cache-2026-10-12/parity_d2b_receipt.json` (sha256 `675d2ee7d8f4…`).
    - review: top-1 0.9766, p99 0.4729.
    - native: top-1 0.9700, p99 0.6154.
    - Driver: "D2b attempt 2 exit=0 2026-10-13T05:42:17Z".
  - **P3**, which also finished after the lane report: "P3 attempt 5 exit=0 2026-10-13T05:50:41Z", "verdict sealed".
    - Receipt `$OC/distill/tps/k3cache_p3_p40_2026-10-12/probe_receipt.json` (`42d0436c912c…`).
    - Line read: "L03 MLA: read 64.3s (292.8 MB/s), compute 53.2s (engine dequant 21.9s), blocks 8, device peak 21,992,605,184 B, rss 24,407,891,968 B".
    - It ran on a P40, not GPU6. **I did not read its T.**
  - xtok: matched mass 0.8743, top-1 mapped 0.9710. Probe bits per byte: bank_v1 0.6897, bank_v2 0.7402.
- **Review findings:**
  - **Fixed:** H, the wrong OMNI1 answer. M, an inline digest could seal with nothing checked, and a missing fold receipt was skipped. M, a resume re-read the whole checkpoint.
  - **Open (low):**
    - `--state-dir` is wiped with rmtree after sealing.
    - The `--max-pass-tokens` guard counts padding: fill is 0.962, so `--pass-count 4` is refused at startup for 2.0M tokens.
    - Per-layer timings are labelled as 5,868 positions but cover 12,552.
    - The d2 and d2b drivers had no lock on GPU3. That is now moot: both drivers have ended.
- **ready_for_stage_c: false.** The full-pass parity fails 2 of 6 gates. D2 and D2b passing points at bf16 drift through L00–60, but that is not proven. **The deciding run is the full pass in the reference environment: GPU6, lab venv, `--row-chunk-kda 1`.**

### 2.3 plumbing: P1 reader and delta mode

- **Tools:**
  - `p1_qwen35_reader.py` **1.0.4**, `086dd91b44f0be2f25de33e224d33c062e1ddb0d4448e7f549f9be4a3c797f14`. Tests `test_p1_qwen35_reader.py` `607cf2e1…`.
  - `ledger/oc_checkpoint.py` 1.1.0, `af43cecb…` (unchanged; it holds `register --delta-base` and `delta-compose`). Tests `test_oc_ledger.py` `6a61df8f…`.
  - `ledger/README.md` `8bac5d2a…`.
- **Tests:**
  - **"Ran 31 tests in 9.260s / OK"** (reader).
  - **"Ran 125 tests in 3.325s / OK (skipped=1)"** (ledger).
  - Negative control, the new tests run against 1.0.3: "FAILED (failures=2)".
- **Parity** (S5 child, CPU, read-only):
  - **"P1 PARITY PASS: 851/851 tensors, 26,141,023,744 elements, 0 failure(s)"**. Receipt `$OC/distill/tps/p1_reader_s5_parity_2026-10-12_r4.json` (`a7623634…`).
  - Seam smoke: **"P1 SEAM SMOKE PASS: 851/851 tensors strict-loaded by tps_probe … forward finite"**. Receipt `p1_seam_smoke_s5_2026-10-12_r2.json` (`bf987cf8…`).
  - Delta-plan parity has been proven only on the tiny real-ledger fixture.
- **Review findings: both fixed.** The delta root check now fails closed, and `parity --plan` now checks against an independently composed reference.
- **ready_for_stage_c: true.**
- **P1 has not been run on OMNI1.** It was run on S5 only. OMNI1 is now registered, so this can run (see §4).

### 2.4 trainer: T(ii), the streamed fwd+bwd KD trainer

No lane report was in my inputs. This section is read from the server only.

- **Tool:** `streamed_kd_trainer.py` **1.3.0**, `66b0854ffe8a5f0cbd652bf12954ef506494f5b2ae922d3255376f8055613b52`. Tests `test_streamed_kd_trainer.py` `e703a867…`.
- **Tests:** **"25 passed, 17 warnings in 232.63s (0:03:52)"**. Log: `$OC/distill/trainer-lane/logs/cpu_tests_main_venv_v130.log`.
- **Parity on a P40:**
  - "SELF-TEST PASS (5/5)".
  - **"GRAD-PARITY PASS: worst relative difference 0.000e+00 (threshold 1.0e-04)"**, with cross-device worst 2.46e-4 (threshold 1e-3).
  - Model Qwen3.5-0.8B, fp32, torch 2.6.0+cu118.
  - Receipt `$OC/distill/trainer-lane/receipts/final-v1.3.0-66b0854f/grad_parity_q35_08b_p40.json`.
- **Not verified:** the review findings and their disposition, and the P2 rate (§4).

### 2.5 d1: D1 per-layer trunk recovery

Server read only.

- **Tools:**
  - `d1_trunk_recovery.py` **1.2.4**, `aeb90e2b2964f2fd0dd0d003bf54b70c760c4f34c5beb0b339e84c51aefec967`.
  - `d1_dp_bench.py` 1.0.0, `75182666…`.
  - Tests `test_d1_trunk_recovery.py`, `e73ce95b…`.
- **Tests:** "16 passed in 20.55s" (lab venv) and "16 passed in 17.97s" (Pascal venv). Log: `distill/scratch/logs/d1_tests_v124.log`.
- **Runs:**
  - L39 on a P40 (tool 1.2.3): "1 layer(s) trained, streams finite through text layer 63; delta written". 4,090.6 s, peak RSS 17.334 GB, 8,192 tokens per step.
  - DP bench (1.2.4 against 1.2.3, one card, synthetic tensors): speedup 1.154 (2.359 against 2.722 s per step).
  - A first bench attempt exited 9. That was the launcher's card guard ("GPU 3 has … MiB used; refusing"). The waiter's retry ended "bench end exit 0 05:51:59".
- **Not verified:** review findings, and any two-card DP run on the 3090s.

### 2.6 media-a: H3 and Qwen-Image teacher caches

Server read only.

- **Tool:** `stage_c_media_a.py` **1.0.2**, `fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf`.
  - Integration test `itest_stage_c_media_a.py`, `f69342ee…`.
  - Unit tests `test_stage_c_media_a.py` `b99c1847…` and `test_run_media_a_wave2.py` `3ce870ed…`.
- **Integration test on GPU3:** **"Ran 12 tests in 494.312s / OK"**. Log: `distill/scratch/media-a-itest-20261013j/itest.log`. The qimg-vae cache drop came back "all_finite": true, "teacher_unchanged": true.
- **P4 for H3:** only a tiny timing run, `p4_h3_int8_tiny.json`. It reports 2.043 s per forward with "seeded gaussian … (timing only)", cfg 1. **That is not the production H3 bf16 measurement on GPU6.**
- **Not verified:** the unit test count line and the review findings.

### 2.7 media-b: ACE DiT and LM, HY3D, QVL32 ViT teacher caches

Server read only.

- **Tool:** `media_b_teacher_cache.py` **1.0.7**, `1b17f5026c4013153f5be65b37d07940dbdcb9d3f6513c6bd4b7e68e2511a739`. Tests `test_media_b_teacher_cache.py` `89c419b0…`.
- **v1.0.7 driver** (`$OC/distill/media_b/driver_status_v107.txt`, 07:29–07:36Z):
  - "38 passed, 1 skipped in 2.38s".
  - ACE encode and ACE DiT on two caches: exit 0.
  - Seal and verify: exit 0.
  - **compare "18 True, 0 False"**: the two independently built ACE DiT caches are bitwise equal.
  - verify-1.0.7 exit 0 on the v105 mini caches (ace_lm_codes, hy3d_trajectories, and qvl32_vit_l13 ×2).
  - "44 passed in 3.49s" with the mini caches attached.
- **P4 receipts:** `$OC/distill/tps/p4_ace_p40_fp32_2026-10-12.json` and `p4_hy3d_p40_fp32_2026-10-12.json`. I did not read their numbers.
- **Not verified:** review findings.

### 2.8 self-gen: parent Q4_K_M sampling (1.0M tokens)

- **Tool:** `selfgen_sampler.py` **1.2.0**, `f77198b2808f56da62fd7108bb267e77fc9d52b3adb2b0930fd6621e242cbb7c`. Tests `test_selfgen_sampler.py` `817813f4…`. **I did not read the test count line.**
- **Result: complete.** This was my one check of the server for this lane. From `$OC/distill/selfgen/logs/run.log`:
  - "manifest … sha fe1f0784777b8fdf: 1168 rows, 1,008,982 tokens".
  - "run rc=0".
  - "verify …: 19 shards, 1,008,982 response tokens, 0 failing, manifest sealed fe1f0784777b8fdf".
  - "verify rc=0".
- Output: `$OC/distill/selfgen/run-q38-27bu-q4km-2026-10-13/`.
- **Teardown:**
  - Both llama-servers were stopped through their launch receipts: GPU1 pid 510296 at 07:32:48Z, GPU2 pid 499410 at 07:32:51Z.
  - "removed transient /mnt/lexar/omni-campaign/scratch/selfgen-q38-2026-10-13/Q38-27BU-parent.Q4_K_M.gguf".
  - The scratch directory now holds only the two export receipts.
- **Earlier receipts:**
  - P5: `$OC/distill/tps/p5-selfgen-q38-27bu-q4km-p40-gpu2.json`.
  - Export check: `selfgen/receipts/export_check_Q38-27BU-parent.json`.
  - Parent tensor-map dry run: `tensor_map_parent_dryrun.json`.
- **Next step: rescoring.** It has not been run (see §4).

### 2.9 corpus

- **Tool:** `distill_corpus_build.py` **1.0.1**, `696559136fb4698c6b92de47ed320c993aba5c3a065a13f18e78cd5c24a7ab94`. Tests `2de2b83a…`.
- **Tests:** **28 passed.** Negative control on 1.0.0: "1 failed, 27 passed".
- **Contamination screen:** 0 hits over 91,984 rows. Seal `6ad8d85d…`.
- **verify:** re-hashed 2,107 files, 0 mismatches.
- **Sets** under `$OC/distill/corpus/`:

  | Set | Contents |
  |---|---|
  | text | 20,000,768 tokens |
  | long | 31 × 32,768 tokens, with 124 answers |
  | K3 | 4 passes; **8,518 Qwen tokens short of 2.0M** |
  | vision_in | 5,896 rows, 4,333,147 visual tokens |
  | video-in | 512 clips, 410,992 visual tokens |
  | media | QIMG 2,048 / HY3D 256 / H3 512 / ACE 1,024 / ViT 7,492 |
  | self-gen prompts | 2,532 |

- The eval split is 1,948 rows.

### 2.10 disk

- **Nothing was deleted.** Deleting the omnicoder weights (`du` printed 1.5T on `/mnt/lexar/omnicoder_0.1.0`) is waiting on Cyril's direct yes.
- **Read-only checks:** no process, lsof, fuser, cron, systemd or ledger reference to the tree.
- Another agent's inventory already exists: `$OC/composite/receipts/disk/oct12/CLEANUP_SUMMARY.md`.
- **This now blocks U9** (§4).

### 2.11 rulings

Cyril's 2026-10-12 go-ahead was sealed as a standalone ruling record using the ledger's own seal function. **The full record path was cut off in my input** ("/home/cereal/omni-c…"), so I have not re-read it.

---

## 3. Stage C readiness, one line per cache

| Cache (DESIGN §3) | Tool | Proven | Production started? | Ready |
|---|---|---|---|---|
| Q-text (text + K3 anchors + vision-in + video-in) | qtext 1.0.3 | CPU and P40 parity on 0.8B; 27B smoke on P40 | No | Yes for text. For VL, fix L1 first |
| Q-long (32k) | qtext 1.0.3 (mb1) | Not run at 32k | No | Yes, but the first job is a 32k memory measurement |
| Self-gen sampling | selfgen 1.2.0 | — | **DONE**: 1,008,982 tokens, sealed `fe1f0784…` | — |
| Self-gen rescoring | qtext 1.0.3 | — | No | **Fix L2 (store_mask) first** |
| K3 (2.0M, 4 passes) | k3cache 1.2.0 | D2/D2b PASS, full pass FAIL 2/6 | No | **No** |
| H3 velocity | media-a 1.0.2 | itest 12 OK, tiny P4 only | No | Pending the P4 H3 bf16 measurement on GPU6 |
| QIMG text-encoder + velocity | media-a 1.0.2 | itest (qimg-vae drop finite) | No | Pending P4 QIMG FP8 on a 3090 |
| HY3D trajectories | media-b 1.0.7 | mini cache verified, P4 on P40 | No | **Yes, and the P40s are free now** |
| ACE DiT + LM | media-b 1.0.7 | bitwise 18/18, mini LM cache verified | No | Yes |
| QVL32 ViT features | media-b 1.0.7 | mini caches verified | No | Yes |
| Vision- and video-in answers | qtext 1.0.3 build-vl (T(i)) | VL and video parity on 0.8B | No | After L1 |

---

## 4. Missing for wave 2

This list was checked against DISTILL_DESIGN §3, §4 and §10. Card rules: GPU0, GPU4 and GPU6 belong to step 1 until its S8 review (pid 917640) ends; it was at case 6 of 16, at roughly 6.5 min per case. The P40s GPU1–3 are free. GPU5 is never used.

### 4.1 Blockers that need Cyril or another owner

| # | Missing | Why it blocks | Exact next action | Card |
|---|---|---|---|---|
| B1 | **Space for `OMNI1.clone-pre-D1`** (U9, G0) | The ruling puts clones on Lexar after the omnicoder blobs are removed. Lexar has **44 G** free; the clone is 55.1 GB, and the D1 delta plus its clone need another 17.1 GB | **Cyril must give a direct yes** to delete the omnicoder weights under `/mnt/lexar/omnicoder_0.1.0`. Then the disk lane reads `CLEANUP_SUMMARY.md`, writes the manifest, copies and checksums the evidence to the NAS, deletes only the weight files and writes a deletion record. After that, step 1 runs S9 `clone OC02-F-Q38-27BU-OMNI1 → OMNI1.clone-pre-D1`. Fallback if he says no: rulings option (a), adding `/mnt/ai_data/omni-campaign/clones` to `REGISTRABLE_UNDER_PROTECTED` with a test | CPU / IO |
| B2 | **K3 full-pass parity verdict** | Stage C for K3 cannot start while 2 of 6 gates fail | After S8 ends, run the full 93-layer review pass in the reference environment: GPU6, lab venv, `--row-chunk-kda 1`, k3_teacher_cache 1.2.0. If it passes, the drift is bf16 across hardware and the P40/CPU caches are accepted. If it fails, bisect L00–60 against the fold engine's per-layer state | GPU6 |
| B3 | Codex pid 4169670 (`k3_capacity_host_loop_proc_race_v4r9.py`) is still alive (U8) | Uses host RAM while K3 needs 97–130 GB | Confirm the owner and whether it can be stopped, through the Relay coordd service | — |

### 4.2 Caches (§3)

| # | Missing | Exact next action | Card |
|---|---|---|---|
| C1 | Q-text production | When GPU0 frees: one-shard `build-text` over `corpus/text` at bf16, fla, mb8, L4096. Read tok/s and peak VRAM, which fills the "production rate unmeasured" gap. Then the full 20M + 2.0M K3 anchors. Only one Qwen streamer runs beside K3 | GPU0 |
| C2 | Q-long | `build-text` mb1 over `corpus/long`. First measure one 32,768-token document for VRAM; it has never run at 32k | GPU0 |
| C3 | Self-gen rescoring (1,008,982 tokens) | qtext **1.0.4**: fix L2. Accept a `target_mask` and shift it inside the tool, and add a prompt+completion test asserting that the first stored position is the last prompt token. Then rescore `selfgen/run-q38-27bu-q4km-2026-10-13` | GPU0 (0.4 h); P40 fallback GPU1 now, slower |
| C4 | K3 2.0M cache | After B2: fix the pass-token guard (count real tokens, or run `--pass-count 5`), scope `--state-dir` to `<state-dir>/<pass_sha[:16]>/`, then run the 4–5 passes | GPU6 |
| C5 | Vision- and video-in answers (T(i) at 27B) | qtext 1.0.4: fix L1 (lock the processor files' sha256 in cache_config). Then a 27B `build-vl` smoke on 2 images and 1 clip, then production over `corpus/vision_in` (4,333,147 visual tokens plus 410,992 video) | GPU0 |
| C6 | **HY3D trajectories** (256 shapes × 30 steps, about 7.7 h) | **Can start now:** `media_b_teacher_cache.py` 1.0.7 production over `corpus/media` HY3D, split across the three P40s, with no P2P | **GPU1–3, free now** |
| C7 | ACE DiT (1,024 clips × 4 σ) + LM (2M codes) | The design puts it on GPU6 after H3. The ACE P4 receipt exists for a P40 fp32, so it can start on a P40 now if HY3D leaves one free; otherwise GPU6 later | GPU3 now, or GPU6 |
| C8 | QVL32 ViT features (block 13) | Production run with media-b 1.0.7, under 1 h | GPU6, or a P40 |
| C9 | H3 velocity (2,048 targets) | First P4 for H3 bf16 against a guarded fp16 cast on GPU6 (the tiny int8 P40 run is timing only). Then production with media-a 1.0.2. If over 20 s per forward, drop to 1,024 targets (ruling) | GPU6 |
| C10 | QIMG text-encoder pass + QIMG velocity | First P4 for QIMG FP8 with per-layer cast on a 3090, plus the `qwen_2.5_vl_7b_fp8_scaled` encoder. Fall back to GPU6 if it needs more than 22 GB | GPU4 |

### 4.3 S0 probes (§4)

| Probe | State | Exact next action | Card |
|---|---|---|---|
| **P1** OMNI1 → qwen3_5, tensor-exact | **Missing on OMNI1** (S5 851/851 PASS) | Run `run_p1_104_tests_and_s5_r4_2026-10-12.sh` with the model path set to `/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1`. OMNI1 is registered, so this is G0's P1 item. Also confirm uniform FFN 16,640 and no gain tensor | CPU now |
| T(i) | 0.8B parity PASS (text, VL, video) | 27B VL smoke: C5 | GPU0 |
| T(ii) | 0.8B grad parity PASS | — | — |
| **P2** T(ii) rate cell | Missing | lr = 0, L4096 mb8 and L32768 mb1 on one 3090. The engine choice depends on this: at least 151 tok/s means the streamed upside path | GPU0 |
| P3 | Done on a P40, receipt sealed | Read and record its T, and whether 8 AttnRes blocks were resident at T = 65,536 as the design requires (RSS 24.4 GB was printed) | — |
| **P4** | Partial: ACE and HY3D (P40 fp32), H3 tiny int8 only | H3 bf16 and fp16 on GPU6; QIMG FP8 + text encoder on a 3090; ACE LM; ViT | GPU6, GPU4 |
| P5 | Done | — | — |
| P6 NAS write rate | Missing | Measured on the S9 clone write itself, after B1 | IO |
| **P7** GradScaler at lr = 0 with K3 present | Missing | 100 steps at `init_scale 2^10` on the split cell (GPU6 + GPU4) with OMNI1. This decides the 81 tok/s question | GPU6 + GPU4 |
| P8 | **Done**: "no lm_head: a text-encoder copy … ViT feature source only" | — (U5 closed) | — |
| **P9** GGUF tensor map on the P1 reader | Missing for OMNI1 (only the parent's dry run exists) | Build the per-tensor GGUF converter on P1 (§4.4 E3) and dry-run its map | CPU |
| P10 visual-token counts | Effectively done by the corpus lane (4,333,147 image and 410,992 video tokens at default max pixels) | Write a P10 receipt that cites the corpus `vision_in/manifest.json` and states that the media sidecar byte sizes were measured | CPU |
| **P11** 256k needle baseline | Missing; no engine | Engine E2 first, then the needle read at 32k/128k/256k × 2 depths on OMNI1 | GPU0 + GPU4 |

### 4.4 Engines and new tools

| # | Missing | Exact next action | Card |
|---|---|---|---|
| E1 | **Hybrid D3 module** (qwen3_5 layers + Chimera unit at depth d + interface rows + flow head; D3a–D3d) | Not started anywhere. Build `distill/tools/d3_hybrid_module.py`. Its parity arm is the hybrid forward against the Chimera charted forward on OMNI1 for each unit (ACE, QIMG, HY3D, H3, QVL32), with zero writes on rows a unit does not own. It also needs D3c (music band sparse-row Adam) and D3d (QVL32 + C11 rows) | CPU, then P40 |
| E2 | **P11 needle engine** | Extend `composite/tools/run_omni_frozen_review.py --stream --teacher-forced` with exact sequence-chunked FFN and projections, plus a parity arm against the unchunked path at 4k | GPU0 + GPU4 |
| E3 | **Per-tensor GGUF converter** (Chimera checkpoint via P1 → qwen3_5 GGUF, text rows truncated to 248,320, no merged HF copy) | Build it on top of `$OC/export/export_hf_to_gguf.py` (llama.cpp b10752) using the P1 reader as the tensor source. Needed for P9, the H checks, the on-policy option (i) sampler and the final Q4 | CPU |
| E4 | D1 at production | Two-card DP on GPU0 + GPU4 with OMNI1.clone-pre-D1 as the base (after B1); measure seconds per layer against the planned 359 s | GPU0 + GPU4 |
| E5 | K3 cross-tokenizer loss in the trainer | Verify that `streamed_kd_trainer.py` 1.3.0 consumes the xtok target (ids of -1, tail bucket, aligned boundaries) and the 0.5/0.5 anchor on K3 rows. Nobody has checked this yet | CPU |
| E6 | **`PREREGISTRATION_OMNI_STAGE_D_v1.md`** | Absent from the server (`find` returned nothing). Write it (rungs, losses, band mask [0, 248,320), gates) and pin its sha in `stage2_training_receipt.py` receipts before any D1 step | — |
| E7 | Delta mode | `oc_checkpoint.py` 1.1.0 `register --delta-base` and `delta-compose`, plus reader `parity --plan`, are proven on the fixture. **No real delta yet.** First real use is the D1 delta; then run `parity --plan` on OMNI1.clone-pre-D1 | CPU |
| E8 | **H checks** (D4: GPU6 Chimera load of the merged model; P40 load of the truncated text GGUF) | Depend on E3. The media CUDA K-quant dequant and Pascal bf16 handling for the "Chimera-Q4" package (§8) come after stage T, not in wave 2 | GPU6, P40 |

### 4.5 §10 unknowns

| U | State |
|---|---|
| U1 free space | Read 07:39Z: root 118 G, Lexar **44 G**, NAS 345 G. Read again before D2 and D4 |
| U2 delta mode | Built, fixture-proven (E7) |
| U4 S0 rates | P2, P4 (partial), P6, P7 and P11 are still missing |
| U5 | Closed by P8 |
| U6 | Media sidecars counted by the corpus lane. 7 LTX-suspect v9_base video rows were excluded |
| U7 | All rulings recorded on 2026-10-12 |
| U8 | Codex pid 4169670 is still alive (B3) |
| U9 | Ruled for Lexar, but blocked on space (B1) |
| U10 | S8 runs on GPU6/4/0, not GPU5. Satisfied |

### 4.6 Open low findings carried into wave 2

- qtext L1: `--processor-dir` is not locked. Fix before C5.
- qtext L2: the `store_mask` convention. Fix before C3.
- qtext L3: bf16 logit rounding. State it in the manifest, or compute the head in fp32.
- k3cache: `--state-dir` rmtree; the padded pass-token guard; mislabelled timings. Fix before C4.

---

## 5. What I could not verify

- The S8 text review's final result: it was at case 6 of 16 when I read it.
- The S8 media decode's structural numbers (rms, pixels, faces, frames). I saw only exit 0 and the schema.
- Review findings and dispositions for the trainer, d1, media-a, media-b and self-gen lanes. No reviewer verdicts for them were in my inputs.
- The test count lines for selfgen 1.2.0 and the media-a unit tests.
- P3's T, and whether the P4 ACE and HY3D receipts meet the design's per-forward budgets.
- The rulings record's full path, which was cut off in my input.
- Whether the earlier S5 `vcheck_written_child.json` run (pid 95797) has finished.
