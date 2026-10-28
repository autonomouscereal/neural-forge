# run0 hardware and lanes survey (2026-10-22, read-only)

Scope: the AI server (`cereal-X10DRG`, 192.168.50.222). I checked the P40s (GPU1/2/3/5), the lanes, the torch builds, host RAM, CPU, PCIe and disk for run0. Run0 is the 2B Qwen-planted all-modality distill on all 4 P40s.

Nothing was launched, claimed or changed. The first probe was taken at about 17:15Z; it was re-checked at 18:45-18:54Z after the usage-limit pause (section 1a). The server clock is MDT, UTC-6.

## 1. Bottom line

- **The four P40s are free.** GPU1/2/3/5 each show 5 MiB used and 24,435 MiB free, at 0 % utilisation and 25-31 °C. All four `gpu<N>.lock` files are free (no `fuser` holders). The only open `p40.claims` line is GPU0+GPU6 for Strata.
- **The lane runner cannot launch a 4-card job.** `lane_runner.py enqueue` caps `--gpus-needed` at 1-3. The L-TRAIN pool is only {1, 2, flex 3}, and GPU5 belongs to L-EXP. Run0 has to start through `lanes.py claim-run` directly and borrow GPU5 (section 3).
- **The media judge needs GPU5 every hour.** `media-judge.timer` runs at :35 each hour. It claims GPU5, then GPU3, then GPU1/GPU2 if they are idle. While run0 holds all four cards, every judge cycle is refused. The judge gates media publishing to the corpus, so new generated media stops reaching intake for the whole run (52 groups were pending at 16:35Z). This is a decision for the lead (section 7).
- **The Qwen3.5-2B weights are not on disk.** `/home/cereal/models` has only `Qwen3.5-0.8B` and `Qwen3.5-4B`, and the Lexar has neither. The 2B must be downloaded (about 4-5 GB in bf16) before planting.
- **Checkpoint resume is broken.** In smoke3 C3b (omni_lab 38324, status `fail`), the job resumed from step 3, then refused at step 6: "replicas diverged (state sha256 per rank [...d5ec80f3..., ...161ce7c0...])", rc 2. A multi-day run depends on resume, so fix this first.
- **Realistic throughput is about 380-535 tok/s on 4 P40s (33-46 M tokens/day).** The measured 2B rate is about 150 tok/s per P40, and gradient averaging through host memory costs about 18-20 s per step at 2 ranks. The ARCH §9 figure of "86M tok/day" is not reachable on these numbers (section 6).

## 1a. Re-check at 18:45-18:54Z (what changed since 17:15Z)

- **The P40s are still free.** GPU1/2/3/5 show 5 MiB and 0 % at 18:54Z. `lsof` on `gpu*.lock` shows only Strata's llama-server (pid 829994) holding `gpu0.lock` and `gpu6.lock`. A non-blocking `flock -n` probe on gpu1/2/3/5 succeeded on all four; it held each lock for an instant and released it. `lane_runner.py list` is still empty.
- **`p40.claims` now has 219 lines.** The last line is `18:40:57Z RELEASE GPU5 ... media-judge-20261022T183500Z released (rc 0)`. The only open claim is GPU0+GPU6 (Strata). LANES.json is at revision 1671. L-TRAIN and L-EXP heartbeats are fresh (0.1 h).
- **The media-judge on GPU5 holds the card for 6-35 min each hour.** Today's cycles ran 12:35-12:57, 13:35-14:02, 14:35-15:10, 15:35-16:08, 16:35-17:02, 17:35-17:51 and 18:35-18:40Z. The next one starts at 19:35Z. A 4-card `claim-run --wait` started between cycles gets every lock at once. One started during a cycle waits for the judge's RELEASE.
- **RAM moves more than the 17:15Z snapshot showed.** MemAvailable read 124 GiB (`free -g`) and then 155 GiB (`/proc/meminfo`) within the same few minutes.
  - The GPU4 ComfyUI is a new process: container `comfyui_h3_sol_gpu4`, pid 1227862, up since about 17:33Z (`--reserve-vram 12 --async-offload 2`). The old pid 1119983 is gone.
  - Its RSS went from 33.9 GiB to 17.1 GiB between two reads 3 minutes apart, because it offloads to host RAM on demand.
  - Strata stays fixed at 46.0 GiB RSS (45.4 GiB anonymous, VmLck 0).
  - Worst case: Strata 46 + ComfyUI 34 + Postgres 8 + gen workers about 2, plus 4 Adafactor ranks at about 44, leaves about 75-80 GiB available. Adafactor still fits. adamw-offload (about 110 GB) still does not.
- **CPU.** The load average is 13.4. ComfyUI is using 275 % CPU, dockerd 172 % and clamd 100 % (clamd restarted at 18:45Z). That leaves about 70 threads for 4 ranks, so 10-12 threads per rank still holds.
- **Qwen3.5-2B is still not on disk.** `/home/cereal/models` has only `Qwen3.5-0.8B` and `Qwen3.5-4B`, and `/mnt/lexar/models` has no Qwen3.5.
- **Containers.** No change. `comfyui_qwen21_gpu1/2/5` are `unless-stopped` and stopped by hand; `ice_roster_gpu3/5` and `chimera-p40-*` are restart `no`. `comfyui_qwen21_gpu4` exited about 17:50Z, which is GPU4 and not a P40.
- **Disk.** `/` has 330 GB free, the Lexar 435 GB and the NAS 16 GB (100 %).
- **Torch, re-checked without a GPU** (`CUDA_VISIBLE_DEVICES=` and `torch._C._cuda_getArchFlags()`):
  - `venv-pascal-cu118-torch260` is 2.6.0+cu118, sm_37/50/60/70/75/80/86/90, gloo and NCCL 2.21.5.
  - `venv` is 2.11.0+cu128, sm_75 and up only.
  - `ai-benchmark-lab/envs/lm-eval` is 2.12.1+cu130, sm_75 and up. It is also **unusable on a P40**, because CUDA 13 dropped Pascal.
  - The Pascal venv also has `pydantic` 2.13.5 installed, pulled in as a dependency. run0 code must not import it (standing rule; runbook item 5 is still open with Cyril).

## 2. P40 state and topology

| GPU | Bus | Lane (LANES.json) | NUMA / CPUs | Free | Power limit | PCIe |
|---|---|---|---|---|---|---|
| 1 | 05:00.0 | L-TRAIN primary | node 0 / 0-21,44-65 | 24,435 MiB | 250 W | gen3 x16 max (gen1 at idle) |
| 2 | 08:00.0 | L-TRAIN primary | node 0 | 24,435 MiB | 250 W | gen3 x16 |
| 3 | 09:00.0 | L-TRAIN flex + L-EXP flex | node 0 | 24,435 MiB | 250 W | gen3 x16 |
| 5 | 85:00.0 | L-EXP primary | **node 1** / 22-43,66-87 | 24,435 MiB | 250 W | gen3 x16 |

ECC is disabled on the P40s. The driver is 580.178.04. The maximum SM clock is 1,531 MHz.

`nvidia-smi topo -m`:
- GPU1 is PIX with GPU0, the Strata 3090, so they share one PCIe switch.
- GPU2 and GPU3 are PIX with each other.
- GPU1 to GPU2/3 is PHB.
- GPU5 is SYS to GPU1/2/3: it crosses QPI to the other socket. It sits on node 1 with GPU4 (ComfyUI) and GPU6 (Strata).

A 4-card job therefore spans both sockets. A 3-card job on GPU1/2/3 stays on node 0.

**P2P.** `nvidia-smi topo -p2p r` reports **OK** between GPU1, GPU2 and GPU3, and TNS for GPU5. Do not trust that OK. The standing rule and ARCH §9 [R] say P2P between P40s is broken. The v2 trainer never uses P2P: it averages gradients device → pinned host → gloo → device (`HostAllReduce`, v2_trainer.py:1207). If anyone tries NCCL, NCCL will pick P2P on its own for GPU1-3, so `NCCL_P2P_DISABLE=1` is mandatory (host SHM transport). NCCL is untested on these cards; gloo is the proven path.

**What else can land on the P40s:**
- No running container uses a P40.
- Stopped containers with P40 device bindings:
  - `comfyui_qwen21_gpu1`, `_gpu2` and `_gpu5` (restart `unless-stopped`; they were stopped by hand, so a Docker restart will not bring them back);
  - `comfyui_h3_ice_roster_gpu{1,2,3,5}_20260919` (restart `no`, exited 2-4 weeks ago).
- These containers take no lane locks. If anyone starts a qwen21 P40 instance "on demand" during run0, it will collide with the run.

**p40.claims history.** The file has 215 lines, all owner=claude. Lanes seen: L-EXP 36, d2clean 17, media-m1 11, sprite-strata-engine 6, serve 5, toolcall 4, fold-research 4, sprite-strata 3, L-TRAIN 3, and others.

Today's P40 use:
- v2 trainer smokes on GPU1/2, ending 12:16Z;
- sprite-strata / sprite-strata-engine throwaway llama tests on GPU1,2,3, 12:38Z-15:40Z, now released;
- the hourly media-judge on GPU5, last released 17:02Z.

No other agent holds or has queued P40 work. `lane_runner.py list` shows an empty `ai_serve.lane_job` table.

## 3. Lanes: how jobs are submitted and how a 4-card job is claimed

- **`lanes.py`** is the registry and launcher: `$OC/distill/tools/lanes.py`, registry `$OC/distill/queues/LANES.json` revision 1658, pulsed every 10 min from cron. The card map:
  - GPU1, GPU2: L-TRAIN primary;
  - GPU3: L-TRAIN flex and L-EXP flex;
  - GPU5: L-EXP primary;
  - GPU0, GPU6: L-SERVE, ask-first;
  - GPU4: L-MEDIA, ask-first.
- **`lane-runner@train` and `lane-runner@exp`** are system units, both active. They run `/home/cereal/.venvs/ai-serve/bin/python /home/cereal/ai-serve/lanes/lane_runner.py run --lane %i --poll 60` with `KillMode=process`. Jobs are rows in `ai_serve.lane_job` (raw psycopg2; credentials come from `~/.config/ai-serve/lanes.env`, vault `ai_serve_pg`).
- **Enqueueing:** `lane_runner.py enqueue --lane train|exp --name N --tool T [--gpus-needed 1-3] [--exclusive] [--argv JSON] [--env JSON]`. T must be a `TOOL_PINS.json` tool, and its sha256 is checked. The runner tries card sets from the lane's own pool only, then calls `lanes.py claim-run`.
  - **Trap:** the runner's `DEFAULT_PYTHON` is `/home/cereal/model-surgery-lab/venv`, which is torch cu128 with no SM6x kernels. A `.py` tool queued without `--env '{"PYTHON": ".../venv-pascal-cu118-torch260/bin/python"}'` would fail on a P40.
- **`claim-run`** is the single launcher. For each card it:
  - takes a non-blocking flock on `gpu<N>.lock` (or waits up to `--wait` seconds);
  - refuses if the card has an open claims line, or uses more than `--busy-limit` (default 2,000 MiB), or if MemAvailable is below `--mem-floor-gib` (default **40**);
  - sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<the cards' UUIDs>`;
  - writes the CLAIM line with a preempt command and the RELEASE line on exit;
  - logs an omni_lab run.

  A card belonging to another lane needs `--borrow REASON`. A serving or unassigned card needs `--cyril "<verbatim>"`.
- **The 4-card run0 claim** (GPU3 is L-TRAIN's flex card, so only GPU5 is borrowed):
  ```
  /usr/bin/python3 /home/cereal/omni-campaign/distill/tools/lanes.py claim-run --lane L-TRAIN --agent claude \
    --gpus 1,2,3,5 --borrow "run0 2B distill on all 4 P40s (Cyril 2026-10-22)" --job run0-co2-2b-<stamp> \
    --log <run>/claim.log --wait 3600 --mem-floor-gib 60 --omni-lab-kind distill \
    --ready-file <run>/ready.json --exit-file <run>/exit.json -- \
    /home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python .../v2_trainer.py train ... --devices cuda:0,cuda:1,cuda:2,cuda:3
  ```
  Use `--wait` so the claim waits out a running media-judge cycle and never steals its lock. The smoke jobs used exactly this pattern with `--gpus 1,2` (`smoke_job3.sh`).

## 4. Python and torch on SM61

| Interpreter | Python | torch | CUDA | Arch list | dist / gloo / NCCL |
|---|---|---|---|---|---|
| `/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python` (**use this one**; the v2 smokes ran on it) | 3.12.3 | 2.6.0+cu118 | 11.8 | sm_37, 50, **60**, 70, 75, 80, 86, 90 (sm_60 code runs on sm_61) | yes / yes / NCCL 2.21.5 |
| `/home/cereal/model-surgery-lab/venv` | 3.12.3 | 2.11.0+cu128 | 12.8 | sm_75 and up only, **unusable on a P40** | yes / yes / 2.28.9 |
| `.venvs/ai-serve`, `ai-serve-media`, `chimera-intake`, `hy3d-voxel` | | no torch | | | |

- There is no system `nvcc` and no `/usr/local/cuda*`.
- Triton does not support SM61, so there are no fused or compiled kernels.
- The P40 has no bf16. Its fp16 rate is 1/64, so fp32 is the only training dtype.
- The v2 trainer is multi-process: `torch.multiprocessing` spawn, one process per device, `init_process_group("gloo")`. `world` comes from `--devices`, so 4 devices need no code change.
- The trainer's replica check hashes the full state on every rank. Every rank builds every micro-batch (v2_trainer.py:1739), so data-build CPU grows with world × accum.

I did not run a 4-rank `torch.distributed` test (no heavy runs were allowed). The 2-rank gloo path is measured (omni_lab 38325).

## 5. Host RAM, CPU and disk

**RAM.** MemTotal is 251.7 GiB. Measured now: used 95 GiB, buff/cache 138 GiB, **MemAvailable 156 GiB**, swap 7 GiB with 1.8 GiB in use. Committed_AS is 108 GiB. Per NUMA node: node 0 has 13.9 GB free, node 1 has 11.3 GB free; the rest is page cache.

The large resident processes:
- **Strata `llama-server` (pid 829994): 45.7 GiB RSS**, all anonymous. It runs `--no-mmap --lazy-mode off`; of the 89 GB IQ3_S model, about 66 GB sits on GPU0 (23.8 GB) and GPU6 (42.5 GB) and the rest is resident in RAM. This is a fixed footprint. It does not grow with load, because the KV cache is on the GPUs. It is also the largest RSS on the box, so it is the OOM killer's first target.
- ComfyUI on GPU4 (python pid 1119983 at 17:15Z): 31.7 GiB. At 18:5xZ it was pid 1227862, swinging between 17 and 34 GiB (section 1a).
- Postgres: about 8 GiB.

What the trainer would add, measured per rank:

| Optimizer | Host RSS per rank (measured) | 4 ranks | Fits? |
|---|---|---|---|
| Adafactor | about 10.9 GB at world 2 (run 38325: the 1.9 GB base plus the 7.7 GB pinned gradient buffer) | about 44 GB | yes; MemAvailable stays above 100 GiB |
| adamw-offload | 19.5 GB at world 1 (run 38304), so about 27 GB at world > 1 | about 110 GB | **no**; MemAvailable would drop to about 45 GiB with Strata and ComfyUI resident |

Recommendations:
- Use Adafactor (or another factored optimizer) for run0.
- Raise `--mem-floor-gib` to 60 or more.
- Give the rank processes `oom_score_adj` +500, so an OOM kills the trainer and not Strata.

**CPU.** Two Xeon E5-2699 v4 sockets: 44 cores, 88 threads. Load average is already about 11 from Strata, the gen workers and the mail stack. The smokes used `--threads 16` per rank; with 4 ranks, use 10-12 threads each. Pin the GPU5 rank to node-1 CPUs. Use preferred memory placement, not `--membind`, because node 1 has only about 11 GB free.

**Disk:**

| Mount | FS | Free | Use |
|---|---|---|---|
| `/` (`/home/cereal`) | ext4 | 331 GB (82 % used) | fine for checkpoints |
| `/mnt/lexar` | **exFAT** (no symlinks or hardlinks) | 436 GB (89 %) | the trainer's current run dir (`/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/runs`) |
| `/mnt/ai_data` (NAS) | CIFS | **16 GB (100 %)** | do not write |

Checkpoint size: the 191M proxy checkpoint with Adafactor is 740 MB, which is fp32 weights plus a tiny factored state. A co2-2b checkpoint is therefore about 7.7-8 GB with Adafactor, or about 23 GB with AdamW. Keeping the last 3 plus milestone checkpoints is about 40-60 GB, which fits on either disk.

## 6. Throughput estimate for co2-2b (1.923 B params) on 4 P40s in fp32

**Peak.** The Tesla P40 has 3,840 CUDA cores at a 1,531 MHz boost. That is 2 × 3,840 × 1.531 GHz = **11.76 TFLOP/s fp32** (NVIDIA lists about 12). Memory bandwidth is about 346 GB/s. Four cards give 47 TFLOP/s fp32 at peak.

**Measured (v2_trainer 1.0.0, co2-2b, seq 4,096, Adafactor, checkpointing on, head_chunk 256, MIXTURE_V1 on the proxy npz):**

| omni_lab run | Cards | Tokens / step | Step time | fwd+bwd | Host all-reduce | tok/s | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 38311 (smoke2 D) | 1 (GPU1) | 9,109 | 59-62 s | 58-60 s | (1-8 s of sync) | **148-154** | 20.6 GB |
| 38304 (smoke E, adamw-offload) | 1 | 9,109 | 64-78 s | 59 s | 4-10 s | 117-142 | 20.6 GB |
| 38325 (smoke3 D3) | 2 (GPU1+2) | 18,208 | 78-81 s | 58-60 s | **17.7-20.5 s** | **226-234** | 20.7 GB |

**What the measurements show:**
- **Per-card efficiency.** About 152 tok/s per card at 8 × N FLOPs per token (6N, plus 2N for checkpoint recompute) is about 2.3 TFLOP/s, so roughly 20 % MFU on the P40. The GDN chunk scan is plain torch and there is no Triton.
- **Scaling.** Going from 1 to 2 cards gives 1.53x (76 % efficiency). The loss is entirely the gloo host all-reduce of 7.7 GB of fp32 gradients. At 2 ranks that is about 0.4 GB/s effective, far below what PCIe gen3 allows.
- **VRAM.** Peak is 20.6-20.7 GB of 24.4 GB at seq 4,096. Longer sequences (32-64k, from ARCH S1-S4) will not fit without more sequence chunking or offload.

**Projection for 4 cards (GPU1/2/3/5):**
- A ring all-reduce moves 2(N-1)/N × 7.7 GB per rank: 11.5 GB at N=4 against 7.7 GB at N=2. That gives about 27-31 s, or 30-35 s with GPU5 across QPI.

| Setup | Tokens / step | Step time | tok/s | Tokens / day |
|---|---|---|---|---|
| 4 cards, accum 1 | 36.4k | about 95 s | **about 380** | about 33 M |
| 4 cards, accum 4 (one all-reduce per 4 micro-batches) | 145.7k | about 272 s | **about 535** | about 46 M |
| 4 cards, ideal (all-reduce fully hidden) | | | 608 | 52.5 M |
| 3 cards (1/2/3, node 0, GPU5 left for the judge), accum 4 | 109.3k | about 266 s | about 410 | about 35 M |

At 46 M tokens/day, 300 M tokens takes about a week and 1 B tokens about 22 days. "Tokens" here means sequence positions. One real-size 1024² image is about 1,024 positions (32×32×3 = 3,072-value units), so a day of pure images would be about 45k images; video costs far more.

To get above about 600 tok/s, the per-card MFU has to rise. The candidates are a fused P40 GDN kernel, less recompute, and overlapping the all-reduce with backward. A shared-memory all-reduce, or NCCL with `NCCL_P2P_DISABLE=1`, may cut the 18-30 s all-reduce, but both are untested here. **None of these were measured.** The ARCH §9 "86M tok/day" needs about 1,000 tok/s, which is about 1.6x the 4-card ceiling.

**Caveats:**
- These numbers come from 3-step probes at random init on proxy media (64×64 images, 4-frame video).
- The real-data mix changes the cost of the data builder, not the per-position trunk FLOPs.
- Attention cost grows with sequence length.
- Re-measure tok/s in the first hour of run0 and log it in omni_lab.

## 7. Risks and decisions for the lead

1. **4 cards against the media judge.**
   - 4 cards gives about +30 % throughput, but the hourly judge gets refused, so media corpus publishing stalls.
   - 3 cards (1/2/3) keeps everything on NUMA node 0, avoids GPU5's cross-socket all-reduce, and leaves GPU5 for the judge and L-EXP.
   - Cyril said "all 4 p40s", so if 4 cards is chosen, pause or redirect the judge knowingly.
2. **Fix the resume divergence (38324) before launch.** The fix needs an identical replica sha after a resume.
3. **Download Qwen3.5-2B** to `/home/cereal/models` (ext4) and check its licence.
4. **Do not use the lane-runner queue for run0.** Use a direct `claim-run` with `--borrow` for GPU5 and `--wait`; never take a held lock.
5. **Protect Strata.** Set a mem floor of 60 GiB or more, raise the trainer's oom_score_adj, use Adafactor rather than adamw-offload, and run 10-12 threads per rank.
6. **Watch for unlocked containers.** Nobody should start `comfyui_qwen21_gpu1/2/5` while run0 runs, because those containers bypass the locks.
7. **Do not touch GPU0, GPU4, GPU6 or their services.** GPU1's host traffic shares a PCIe switch with GPU0 (Strata). This was not measured as a problem, but keep an eye on Strata decode tok/s after launch.

## Evidence

- Server reads at about 17:15Z:
  - `nvidia-smi`, `topo -m`, `topo -p2p r`, the `--query-gpu` PCIe fields;
  - `/proc/meminfo`, `free`, `numactl -H`, `lscpu`, `df -T`;
  - `ps` RSS and `/proc/829994/status`;
  - `docker ps -a` and `inspect`;
  - `systemctl cat lane-runner@.service media-judge.service strata-llama.service`;
  - `lanes.py show|cards|claim-run --help`, `p40.claims` (215 lines), `lane_runner.py` (source and an empty list), `media_judge.py:31,100,844`.
- Smoke runs: `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/runs/smoke{,2,3}-*`. The D3 and C3b logs and `smoke_job3.sh` were read.
- omni_lab metric `tok_per_s` for runs 38301-38325.
- The torch probe was an import-only `-c` on GPU1 for each venv; no CUDA work was run.
- Re-check at 18:45-18:54Z:
  - `nvidia-smi` (query, topo, p2p), `free`, `/proc/meminfo`, `ps`, `lscpu`, `df`;
  - `lanes.py show|cards|claim-run --help`, the `p40.claims` tail (219 lines), `lsof` on the locks, `flock -n` probes;
  - `systemctl list-timers`, `systemctl cat lane-runner@train media-judge`;
  - `docker ps -a` and restart policies;
  - torch arch flags with no GPU visible.
