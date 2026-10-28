# run0 hardware and lanes survey (refreshed 2026-10-28, read-only)

Scope: the AI server (`cereal-X10DRG`, 192.168.50.222). I checked the P40s (GPU1/2/3/5), the lanes, torch, host RAM, CPU, PCIe and disk for run0. Run0 is the planted Qwen3.5-2B all-modality distill on all 4 P40s, run for up to 10 days (Cyril, 2026-10-27).

Probes ran at about 05:59-06:20Z on 2026-10-28. The server clock is MDT (UTC-6). Nothing was launched, claimed or changed on the server. The 2026-10-22 version of this file is in `backups/survey_hw.20261022.md` (sha256 0fcc754e...). Its measured throughput (section 6) still stands, because no training run has happened since.

## 1. Bottom line

- **The four P40s are free.** GPU1/2/3/5 each show 5 MiB used, 24,435 MiB free, 0 % utilisation and 22-25 °C. A `flock -n` probe on gpu1/2/3/5.lock succeeded on all four. The only lock holder is Strata (`llama-server` pid 149633 on gpu0.lock and gpu6.lock). The only open `p40.claims` line is GPU0+GPU6 (`serve`, strata-llama).
- **RAM is the blocker. With RAM as it is now, run0 cannot launch and could not stay running.**
  - MemAvailable is **28.2-28.5 GiB**. Swap is 6 of 7 GiB used, and 5.2 GiB of that is Strata's.
  - `run0_queue.sh` claims with `--mem-floor-gib 60`. The trainer pauses below 60 GiB and resumes at 75 GiB.
  - The 4 Adafactor ranks add about 41 GiB of host RAM. MemAvailable therefore has to be about **100 GiB or more before launch**, which means freeing about 72 GiB.
  - The same floor is already hurting other work. Since about 02:35Z on 10-28, every hourly media-judge has been refused: "host MemAvailable 28.2 GiB < floor 40.0 GiB". 2,341 media groups are waiting for review.
- **Why RAM is short: Strata's Q4 expert cache (live since 10-27 21:56Z).** `llama-server` is now 144 GiB RSS, where on 10-22 it was 46 GiB. That breaks down into:
  - 72.9 GiB of shmem: the `-cmoe` experts, pinned in host RAM;
  - 71.1 GiB anonymous, which includes the `--cache-ram 49152` prompt cache of up to 48 GiB;
  - plus 5.2 GiB swapped out.

  The GPU4 ComfyUI (`comfyui_qwen21_gpu4`, `--cache-lru 10 --async-offload 2`) holds another 57 GiB anonymous. Section 5 lists the levers.
- **Disk on `/` is down to 113 GB free** (it was 330 GB on 10-22). The 104 GB Q4 Strata download filled it. Run0's run directory is on `/`, and so is the 60 GB `--disk-floor-gb`. Over 10 days the checkpoints need about 80-100 GB (section 5), so the floor would stop checkpoint writes around day 4-6. **The NAS is 100 % full (1.0 MB free)**, so run0 must not write there.
- **Nothing for run0 is on the server yet beyond the data.**
  - `/home/cereal/omni-campaign/run0/data` is built (19 GB), along with `data-smoke` (450 MB) and the `code-b*` trees.
  - The following are missing: `/home/cereal/models/Qwen3.5-2B` (needed for the tokenizer and weights; `run0_queue.sh` expects `.../Qwen3.5-2B/tokenizer.json`), `$R0/ablate+align` (the init), `/home/cereal/omni-campaign/chimera-omni-v2` with `DEPLOYED.json` (the queue script checks this), and `dump_spec.json`.
- **The resume fix is written but not proven on a GPU.** Commit `e52fa93` on `claude/run0-c-20261022` (C-1) adds:
  - CPU-mapped loads;
  - a rank-0 broadcast of each tensor;
  - an sha check before step 1;
  - `replica_sha.json`, `--allow-world-change` and staggered loads.

  It passed 35 tests on the Windows CPU. The `S4 resume` gate (RUN0_BUILD_PLAN.md:464) has never run on the P40s. omni_lab has no training run after 38325, which was the 2-rank smoke on 10-22.
- **Throughput (unchanged): about 380-535 tok/s on 4 P40s**, which is 33-46 M sequence positions per day. Over 10 days that is about **330-460 M positions**. A 4-rank gloo job has never been measured (section 6).

## 2. P40 state and topology

| GPU | UUID | Bus | Lane | NUMA / CPUs | Free | PCIe max | ECC |
|---|---|---|---|---|---|---|---|
| 1 | GPU-c95fb5b3 | 05:00.0 | L-TRAIN primary | node 0 / 0-21,44-65 | 24,435 MiB | gen3 x16 | off |
| 2 | GPU-a92b0dfe | 08:00.0 | L-TRAIN primary | node 0 | 24,435 MiB | gen3 x16 | off |
| 3 | GPU-08b615f9 | 09:00.0 | L-TRAIN flex + L-EXP flex | node 0 | 24,435 MiB | gen3 x16 | off |
| 5 | GPU-ce4a1da0 | 85:00.0 | L-EXP primary | **node 1** / 22-43,66-87 | 24,435 MiB | gen3 x16 | off |

Every card has a 250 W power limit and a 1,531 MHz maximum SM clock. The driver is 580.178.04.

`nvidia-smi topo -m` has not changed:
- GPU1 is PIX with GPU0 (the Strata 3090).
- GPU2 and GPU3 are PIX with each other.
- GPU1 to GPU2/3 is PHB.
- **GPU5 is SYS (across QPI) to GPU1/2/3.** It shares node 1 with GPU4 (ComfyUI) and GPU6 (Strata).

**P2P.** It is never used. The v2/run0 trainer averages gradients device → pinned host → gloo → device. NCCL 2.21.5 is installed, but if NCCL is ever tried, `NCCL_P2P_DISABLE=1` is mandatory, because NCCL would pick P2P on its own for GPU1-3. NCCL is untested here; gloo is the proven path.

**Other things that can land on the P40s.**
- No running container uses a P40.
- Stopped containers bound to P40s:
  - `comfyui_qwen21_gpu1`, `_gpu2` and `_gpu5` (restart `unless-stopped`, stopped by hand);
  - `comfyui_h3_ice_roster_gpu{1,2,3,5}_20260919` (restart `no`);
  - many `chimera-p40-*` containers and one `nervous_galileo` on GPU1, all in the `exited` or `created` state since 10-09/10-10, restart `no`;
  - `gauntlet-blender-gpu3`.
- None of these takes a lane lock. If anyone starts one during run0, it will collide with the run.
- There are no waiting P40 queue processes. The x3/vmath/watchdog jobs from the old L-RESEARCH lane are gone.

**Claims since 10-22** (270 new lines, 489 in total, all owner=claude):
- 138 hourly media-judge claims on GPU5, which spill to GPU3 or GPU1 when GPU5 is busy;
- sprite-strata(-engine) tests on GPU3 (9);
- arch-v2 and v2fold-proxy experiments on GPU3/5 (10);
- the three v2trainer smokes on GPU1/2.

The judge holds GPU5 for 20-31 min each hour; the last cycle that ran was 01:35-01:54Z on 10-28.

## 3. Lanes: how jobs are submitted and how a 4-card job is claimed

- **`lanes.py`** is the registry and launcher: `/home/cereal/omni-campaign/distill/tools/lanes.py`, registry `distill/queues/LANES.json` revision 2713, pulsed every 10 min from cron. The card map:
  - GPU1, GPU2: L-TRAIN primary;
  - GPU3: L-TRAIN flex and L-EXP flex;
  - GPU5: L-EXP primary;
  - GPU0, GPU6: L-SERVE, ask-first;
  - GPU4: L-MEDIA, ask-first.
- The L-TRAIN and L-EXP heartbeats are fresh (0.1 h). Every other lane has ended, is idle, or is stale.
- **`lane-runner@train` and `lane-runner@exp`** are both active. They run `lane_runner.py run --lane %i --poll 60`, and their jobs are rows in `ai_serve.lane_job`.
  - `enqueue --gpus-needed` is capped at 1-3, and each runner only tries cards from its own pool. **A 4-card job cannot go through the lane-runner queue.**
  - The runner's default Python is the cu128 venv, which has no SM6x kernels.
- **`claim-run`** is the single launcher. For each card it:
  - takes a non-blocking flock on `gpu<N>.lock` (with `--wait N`, it waits);
  - refuses if the card has an open claim, uses more than 2,000 MiB, or if MemAvailable is below `--mem-floor-gib`;
  - sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` to the card UUIDs;
  - writes the CLAIM and RELEASE lines;
  - logs an omni_lab run.

  A card from another lane needs `--borrow`, and a serving card needs `--cyril "<verbatim>"`.
- **The run0 claim is already scripted** (`scripts/run0_queue.sh` on `claude/run0-c-20261022`):
  - `claim-run --lane L-TRAIN --gpus 1,2,3,5 --borrow "run0 2B distill on all 4 P40s (cyril-run0-plant-qwen-2026-10-22)" --wait 3600 --mem-floor-gib 60` → `run0_supervisor.py` → `v2_trainer train`;
  - one rank per card, the GPU5 rank pinned to node 1;
  - `--threads 10`, `--oom-score-adj 500`, `ACCUM=4`, `--ckpt-every 50 --keep 3 --keep-every 500`;
  - `--mem-floor-gib 60 --mem-resume-gib 75 --disk-floor-gb 60`;
  - `RUN0_GPUS=1,2,3` is the 3-card fallback.

  `--wait 3600` waits out a running judge cycle; it never takes a held lock.
- **The judge and run0 compete for GPU5.** While run0 holds all four cards, every hourly judge is refused (it is refused on RAM already). Media publishing to the corpus stalls for the length of the run unless the judge moves somewhere else. This is a decision for the lead.

## 4. Python and torch on SM61

| Interpreter | torch | CUDA | Arch list | dist / gloo / NCCL |
|---|---|---|---|---|
| `/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python` (**use this**; the smokes ran on it) | 2.6.0+cu118 | 11.8 | sm_37, 50, **60**, 70, 75, 80, 86, 90 (sm_60 code runs on sm_61) | True / True / True, NCCL 2.21.5 |
| `/home/cereal/model-surgery-lab/venv` | 2.11.0+cu128 | 12.8 | sm_75 and up, **unusable on a P40** | |
| `ai-benchmark-lab/envs/lm-eval` | 2.12.1+cu130 | 13.0 | CUDA 13 dropped Pascal, **unusable** | |

Notes:
- The Pascal venv also has numpy 2.4.4, safetensors 0.8.0, transformers 5.12.1 and triton 3.2.0. Triton does not support SM61, so there are no compiled kernels.
- `pydantic` 2.13.5 is installed in it as a transitive dependency. Run0 code must not import it.
- There is no system `nvcc`.
- The P40 has no bf16 and runs fp16 at 1/64 rate, so fp32 is the only training dtype.
- The trainer is multi-process: spawn, one rank per card, `init_process_group("gloo")`.
- A 4-rank `torch.distributed` job has never run here (only 2-rank, omni_lab 38325). The run0 `--smoke` mode (100 steps, accum 1, no restarts) is the first 4-rank test.
- Host-side omni_lab calls need `/home/cereal/.venvs/ai-serve/bin/python`, because the system `python3` has no psycopg2.

## 5. Host RAM, CPU and disk

**RAM.** MemTotal is 251.7 GiB. `free -g` shows 223 used, 106 buff/cache (79 GiB of that is shmem) and **28 available**. Each NUMA node has about 1 GB free. Swap is 6.0 of 7.0 GiB used.

| Process | RSS | Breakdown | Notes |
|---|---|---|---|
| Strata `llama-server` pid 149633 (GPU0+GPU6) | **144 GiB** (+5.2 GiB swap) | 72.9 GiB shmem (pinned `-cmoe` experts, the 71.7 GiB expert set) + 71.1 GiB anonymous (`--cache-ram 49152` prompt cache of up to 48 GiB, plus context checkpoints and buffers) | `--no-mmap --lazy-mode off`; cgroup MemoryMax=infinity; oom_score_adj 0 |
| ComfyUI `comfyui_qwen21_gpu4` python pid 329587 | 57.9 GiB | 57.0 GiB anonymous | `--cache-lru 10 --async-offload 2 --reserve-vram 2` |
| Postgres (checkpointer, writer, backends) | about 8 GiB | | |
| gen workers, gates, clamd, dockerd and the rest | about 3 GiB | | |

**What run0 adds.** Adafactor was measured at about 10.9 GB per rank at world 2 (1.9 GB base plus the 7.7 GB pinned gradient buffer). At 4 ranks that is **about 44 GB (41 GiB)**; data-builder memory per rank comes on top. AdamW-offload would be about 110 GB and is out of the question.

**Budget.**
- Launch needs MemAvailable ≥ 60 GiB at the claim. After the ranks allocate, it has to stay ≥ 60 GiB or the trainer pauses, and it resumes only at 75 GiB.
- So the target is **≥ about 100 GiB available before launch and ≥ 60 GiB steady**.
- That requires freeing **about 72 GiB** from the current state, or lowering the floors.

**Levers, largest first.** None is applied; the lead decides. All but the last touch GPU0/4/6 services, which are serving.

1. **Strata experts read from disk** (Cyril, 10-27: "[Cyril quote withheld]").
   - Drop `--no-mmap` for the expert tensors, or move only some layers' experts to mmap. Their pages then become reclaimable page cache instead of 72.9 GiB of pinned shmem, which frees up to about 70 GiB.
   - Cost:
     - an xcache miss reads pageable or disk memory instead of pinned memory;
     - `/` is a **Crucial BX500 SATA SSD** (DRAM-less, about 500 MB/s, weak sustained reads), so a cold miss that goes to disk is slow;
     - Strata's prompt read (258 tok/s) and decode (24 tok/s) would drop by an unmeasured amount.
   - The xcache patch's pinned-host requirement is not checked here. Test before switching.
2. **Strata prompt cache `STRATA_CACHE_RAM` 49152 → 8192-16384.** This frees up to about 32-40 GiB if the cache is full (71 GiB anonymous says it largely is). Cyril has said a disk-backed prompt cache is fine. This is the cheapest lever, and Strata keeps all its experts.
3. **ComfyUI GPU4 `--cache-lru 10` → 2-3.** This frees a share of its 57 GiB. It is a GPU4 service, so ask first.
4. **Lower the run0 floors.** For example, `--mem-floor-gib 30 --mem-resume-gib 40`, and the claim floor to match. This is only safe if Strata's footprint is capped first (levers 1-2); otherwise the prompt-cache growth plus 4 ranks invites an OOM. The ranks run at `oom_score_adj` 500, so the OOM killer takes a trainer rank before Strata.

Levers 2 and 3 together give about 60-75 GiB, which is borderline. Levers 1 and 2 together give about 100 GiB, which is comfortable. Freeing RAM also unblocks the media judge, whose floor is 40 GiB.

**CPU.** Two Xeon E5-2699 v4 sockets: 44 cores, 88 threads. The load average is about 5 now (ComfyUI, gen workers, clamd, mail stack). 4 ranks × 10 threads, with the GPU5 rank on node 1, fits.

**Disk.**

| Mount | Device / FS | Free | Use |
|---|---|---|---|
| `/` (`/home/cereal`, run0 run dir `$R0/run`) | Crucial BX500 2 TB SATA SSD, ext4 | **113 GB** (94 % used) | checkpoints; `--disk-floor-gb 60`, resume at 80 |
| `/mnt/lexar` | Lexar SL600 USB HDD, **exFAT** (no symlinks) | 381 GB (91 %) | milestone or overflow target |
| `/mnt/ai_data` (NAS) | CIFS | **1.0 MB (100 %)** | do not write; NAS cleanup is a separate item |

**What run0 needs on `/`:**
- the Qwen3.5-2B download, about 4.5 GB;
- the plant and ablate+align output, about 7.7 GB each in fp32 (1.923 B × 4 B);
- rolling checkpoints: 3 × about 8 GB with Adafactor (the estimate scales from the proxy's 740 MB; a 2B checkpoint has not been measured);
- milestone checkpoints: `--keep-every 500` at about 270 s/step is one every 37 h, so about 6 in 10 days, about 48 GB;
- dumps.

That totals **about 80-100 GB**, against 53 GB of headroom above the floor. Two ways out:
- send the milestones to the Lexar;
- free `/`. Candidates for the lead: the IQ3_S Strata rollback copy, 83 GB at `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata`; and the two 27B GGUFs, 32 GB, which have been retired from data generation.

## 6. Throughput estimate for co2-2b (1.923 B params) on 4 P40s in fp32

**Peak.** The Tesla P40 has 3,840 CUDA cores at 1,531 MHz: 2 × 3,840 × 1.531 GHz = **11.76 TFLOP/s fp32** per card (NVIDIA lists about 12). Memory bandwidth is 346 GB/s. Four cards give 47 TFLOP/s at peak.

**Measured on 10-22.** v2_trainer, co2-2b, seq 4,096, Adafactor, checkpointing, head_chunk 256, proxy media. These are the only 2B measurements; nothing has run since.

| omni_lab run | Cards | Tokens / step | Step time | fwd+bwd | Host all-reduce | tok/s | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 38311 | 1 (GPU1) | 9,109 | 59-62 s | 58-60 s | 1-8 s | **148-154** | 20.6 GB |
| 38304 (adamw-offload) | 1 | 9,109 | 64-78 s | 59 s | 4-10 s | 117-142 | 20.6 GB |
| 38325 | 2 (GPU1+2) | 18,208 | 78-81 s | 58-60 s | **17.7-20.5 s** | **226-234** | 20.7 GB |

**What that implies:**
- About 152 tok/s per card, at 8N FLOPs per token (6N plus 2N for recompute), is about 2.3 TFLOP/s per card: **about 20 % MFU**.
- Going from 1 to 2 cards gives 1.53x. The loss is the gloo host all-reduce of 7.7 GB of fp32 gradients.

**Projection for 4 cards.** A ring all-reduce moves 2(N-1)/N × 7.7 GB = 11.5 GB per rank. That takes about 27-31 s, or 30-35 s with GPU5 across QPI.

| Setup | Tokens / step | Step time | tok/s | Tokens / day | 10 days |
|---|---|---|---|---|---|
| 4 cards, accum 1 | 36.4k | about 95 s | about 380 | about 33 M | about 330 M |
| **4 cards, accum 4 (run0_queue.sh default)** | 145.7k | about 272 s | **about 535** | **about 46 M** | **about 460 M** |
| 4 cards, all-reduce fully hidden | | | 608 | 52.5 M | 525 M |
| 3 cards (1/2/3, node 0), accum 4 | 109.3k | about 266 s | about 410 | about 35 M | about 350 M |

**What a position costs.** One real-size 1024² image is about 1,024 positions (3,072-value units), so a full day of images would be about 45k images; video costs far more. The ARCH §9 "86M tok/day" figure needs about 1,000 tok/s, which is out of reach without kernel work (a fused GDN, less recompute, all-reduce overlapped with backward, or NCCL-SHM). None of that has been measured. Re-measure tok/s in the first hour of run0 and log it in omni_lab.

## 7. "Make sure it completes before spending 10 days": what has to be true first

1. **RAM** (section 5): at least about 100 GiB available before launch, through levers 1-2 or a lowered floor. Without this, `claim-run` refuses (rc 5) or the trainer pauses within minutes.
2. **Disk** (section 5): about 100 GB usable above the floor, or milestones on the Lexar.
3. **Missing inputs on the server:** Qwen3.5-2B (check its licence), the plant → ablate+align init, the `chimera-omni-v2` deploy with `DEPLOYED.json`, and `dump_spec.json`.
4. **Run `run0_queue.sh --dry-run`, then `--smoke` (100 steps on 4 cards).** The smoke is the first ever 4-rank gloo run and the first GPU test of the C-1 resume fix. Then run RUN0_BUILD_PLAN's S4 resume gate: stop at step 100, `--resume auto` for 20 steps, `kill -9` one rank. It passes only if every rank has the same replica sha.
5. **Measure the real step time and the all-reduce on 4 cards** before committing 10 days, against the projections above.
6. **Pick 4 cards or 3.** With 4, the judge gets no card and media publishing stops. With 3, the run stays on node 0 at about 77 % of the 4-card rate.
7. **Keep unlocked containers off the P40s:** `comfyui_qwen21_gpu1/2/5`, `ice_roster`, `chimera-p40-*` and `nervous_galileo` all bypass the locks.
8. **After launch, watch Strata decode tok/s.** GPU1 shares a PCIe switch with GPU0.

## Evidence (2026-10-28 05:59-06:20Z)

- `nvidia-smi` (query, compute-apps, `topo -m`, `-L`, PCIe/ECC fields).
- `free`, `/proc/meminfo`, `numastat -m`, `/proc/149633/{status,smaps,cmdline}`, `/proc/329587/status`, `ps --sort=-rss`, `lscpu`, `lsblk`, `df`.
- `systemctl cat strata-llama` (drop-ins `ctx256k.conf` and `q4kxl.conf`), `systemctl list-units` and `list-timers`, `crontab -l`, `journalctl -u media-judge` (refusal lines).
- `lanes.py show` and `cards`; `p40.claims` (489 lines, tail and per-lane counts); `lsof` and `flock -n` on the `gpu*.lock` files.
- `docker ps -a` and `inspect` (device requests, restart policy).
- The Pascal venv: an import-only probe with `CUDA_VISIBLE_DEVICES=` (torch arch flags, dist/gloo/nccl), plus `pip list`.
- omni_lab `list runs` and `sql` (no training run after 38325).
- The server `run0/` tree; `/home/cereal/models` (no Qwen3.5-2B); `du` of `models/gguf`.
- The local branch `claude/run0-c-20261022` (`e52fa93`, `scripts/run0_queue.sh`).
- The NAS `du` timed out after 80 s; only `df` is reported.
