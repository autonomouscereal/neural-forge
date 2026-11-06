# AI server memory: what the "cache" is, what was wrong, and the rules from now on (2026-11-04)

Server `192.168.50.222` (cereal-X10DRG): 2 NUMA nodes, 251.7 GiB RAM, kernel 7.0.0-38, Ubuntu 24.04, booted 2026-11-03 15:16Z. Author: Claude, AI-server memory side session (`bea06e`) for the Omni lead (`local_89e897e3`). Relay CLAIM 204306, STATUS 204309, 204311, 204315 and 204319. Status: LIVE. Every change below is applied and has a one-command rollback.

Cyril, 2026-11-04:
- "[Cyril quote withheld]"
- "[Cyril quote withheld]"

## 1. The plain answer

**Almost none of the "cache" was GPU or prefill cache.** `free` showed 183-191 GiB of `buff/cache`. That number adds up three different things, and only one of them is a cache in the everyday sense.

Snapshot 16:51-16:57Z, from `/proc/meminfo` and per-service cgroup `memory.stat`:

| `free` calls it | GiB | What it really is |
|---|---:|---|
| `buff/cache`: plain file cache | ~138 | Copies of files read or written recently. 98 % was cold (`Inactive(file)` 139 GiB, `Active(file)` 2.2 GiB). The kernel hands it back the moment a program asks for memory, which is why `available` was 160 GiB. |
| `buff/cache`: `Shmem` | 47 | Program memory that Linux files under "shared". 41.1 GiB was Strata's pinned host buffers, 4 GiB the agent-memory Postgres shared buffers, ~2 GiB run 0 and the 27B judge. It is not a cache and cannot be dropped. |
| `buff/cache`: reclaimable slab | ~10 | Kernel metadata: inodes, dentries. |
| `used` | ~89 | The same 47 GiB of `Shmem` again (procps counts it in both columns), plus ~30 GiB of ordinary program memory. |
| `available` | 160 | What programs can actually still get. |

**Strata held 45 GiB of host RAM against a 103.7 GiB model** (4 GGUF parts, 111,334,654,784 bytes).
- 39.5 GiB was pinned host memory holding the experts that did not fit in GPU6's VRAM: 142 per-weight blocks of 251-475 MiB, counted from `/proc/<pid>/smaps`.
  - It reached that size on 11-03, when Strata moved to GPU6 alone.
  - It is part of the model, not a cache. Without it, experts would stream from disk on every token.
- 1.6 GiB was pinned compute and output buffers. 3.8 GiB was ordinary memory: heap and the prompt cache.
- The GPU-layout session restarted Strata on GPU0+GPU6 at 17:54Z. Its pinned tier is now 8.35 GiB (GPU0) + 13.16 GiB (GPU6) = **21.5 GiB**.

**Strata's prefill (prompt) cache is the only real "prefill cache".**
- It is capped at 16 GiB (`--cache-ram 16384`). It held under 3.8 GiB and had never evicted (`llamacpp:prompt_cache_evict_total 0`).
- The per-slot KV cache lives in VRAM, not host RAM.

**Who owned the ~138 GiB of plain file cache**, by the cgroup that read it (cgroup v2 charges a page to the reader):

| Owner | GiB | Why |
|---|---:|---|
| agent-memory Postgres container | 38-42 | Its 52 GB database. The `pg_dump` backup, running since ~16:05Z, reads every table once. |
| agent-memory backup session | 20.7 | The dump file it writes, and its rsync to the media server. |
| Lexar-to-NAS mover (disk-cleanup session) | 15.8 | 126 GiB of `.git`/`.venv` files copied, sha256 on both sides. |
| run-0 session | 9.9 | Training data parts and checkpoints. |
| finished SSH sessions (re-parented to `user-1000.slice`) | ~46 | Earlier copies, builds, scans. |
| Strata | 1.6 | Lazily read rows of the n-gram (PLE) table in GGUF part 2. |

**Per-file check (`pcache_scan.py`).**
- The first scan (16:55Z, all filesystems) died on a CIFS listing error after an hour. The rerun is per filesystem.
- `/mnt/lexar` at 18:05Z held 9.7 GiB in cache. 7.6 GiB of that was run-0 data parts (`run0/data/parts`, read by training); 0.9 GiB was the granite-speech and Qwen3-ASR weights.
- `/` is still running and writes `evidence/memory-20261104/pcache-scan-root.txt`.
- By then the picture had already changed. Between 17:10 and 17:16Z, kernel reclaim dropped ~80 GiB of cold cache: 64 GiB from finished sessions, 17 GiB from Postgres, 12 GiB each from the mover and the backup. That is the "reclaimable" in practice.

**Are we doing it wrong? Yes, in two places, and neither of them is "too much cache".**

1. **Swap was on with `vm.swappiness=60`.** The kernel pushed program memory to disk while 138 GiB of cold file copies stayed in RAM.
   - Since boot: 59 GiB written to swap and 22 GiB read back (`pswpout` 15.4 M, `pswpin` 5.9 M pages).
   - This is the "page filing" Cyril saw: 4 GiB in swap at 16:51Z. The biggest occupants were an idle ComfyUI (1.5 GiB), 7 orphaned run-0-era worker processes (1.1 GiB) and keycloak (0.45 GiB).
2. **Pinned GPU host memory cannot move, and three kernel features kept trying to move it.**
   - The features were proactive compaction, watermark boosting and automatic NUMA balancing.
   - From boot to 18:06Z on 11-04: `kcompactd1` 8.4 h of CPU, `kcompactd0` 3.8 h. 82 % of page migrations failed (7.59 B failed, 1.68 B ok). NUMA balancing took 5.9 B hint faults and moved 534 M pages (2.0 TiB) between the sockets.
   - The pinned-memory churn behind it came from ComfyUI (section 2.3).

**The approach change:**
- no swap;
- an explicit OOM order instead of swap;
- pinned memory sized once and never churned;
- the kernel told not to chase memory that cannot move;
- one-time big readers kept from filling the cache.

Cold file cache is harmless once swap is off. **The number to watch is `MemAvailable`, not `buff/cache`.**

## 2. Measurements

Read-only tools, in `/home/cereal/ai-serve/memory/` (repo: chimera-omni `ops/ai-serve/memory/`, branch `claude/ai-server-memory-20261104`). Evidence: `/home/cereal/ai-serve/evidence/memory-20261104/`.

| Tool | What it does |
|---|---|
| `memwatch.py` 1.0.0 | 10 s sampler, user unit `memwatch-20261104`. Records meminfo per node; compaction, migration, NUMA, swap, reclaim and FOLL_PIN counters; kswapd/kcompactd CPU ticks; node-1 high-order free blocks; every cgroup above 512 MiB. |
| `memwatch_report.py` | Rates over a window. |
| `perf_window.py` | Strata decode/prompt tok/s from llama-server `print_timing`, and run-0 step times. Report only. |
| `pinwatch.bt` | bpftrace: `pin_user_pages*` / `unpin_user_*` per process, in 5-min intervals. |
| `wakewatch.bt` | bpftrace: kswapd and kcompactd wakeups by node, order and process; direct compaction by order and process. |
| `pcache_scan.py` | `mincore` on every file of 1 MiB or more. Shows which files hold page cache; reads and evicts nothing. |
| `comfy_jobs.py` | ComfyUI job durations, grouped by the models each job loaded. |
| `cache_cap_test.sh` | The capped-scope proof (rule 4). |

### 2.1 Before and after, per change

Each window is 10 minutes or longer. Other sessions changed the server during them, as noted.

| Window (11-04, UTC) | State | kcompactd1 | compaction wakes | migrations ok / failed per s | NUMA PTE updates per s | swap out pages/s | MemAvailable min / mean GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 16:54-17:06 | baseline | 14.9 % | 72.2 /s | 1,949 / 30,328 | 31,270 | 29.9 | 155.0 / 159.8 |
| 17:07-17:18 | + `compaction_proactiveness=0` | 47.1 % | 1.5 /s | 18,733 / 65,257 | 56,717 | 8.2 | 151.2 / 158.5 |
| 17:20-17:33 | + `watermark_boost_factor=0` | 19.1 % | 0.1 /s | 11,066 / 47,619 | 17,644 | 174.1 | 144.5 / 151.0 |
| 17:36-17:46 | + `numa_balancing=0` | 0.9 % | 0.1 /s | 4,007 / 8,963 | 0 | 51.8 | 146.2 / 164.5 |
| 18:14-18:27 | all changes, Strata on GPU0+6, ComfyUI on unpinned h3sol | 0.0 % | 0.0 /s | 226 / 1.8 | 0 | 0 (no swap) | 146.2 / 161.2 |
| 18:41-18:56 | + ComfyUI general unpinned, backup cache dropped | 0.2 % | - | - / 162 | 0 | 0 | 162.3 / 172.9 |

Confounders:
- **17:10-17:16Z:** the GPU-layout session stopped the 27B judge, media-judge and K3. Kernel reclaim then dropped ~80 GiB of file cache in ~6 min, with kcompactd1 at 100 % trying to assemble large blocks from the freed memory.
- **From 17:20Z:** media generation restarted (gen-worker-media), bringing ComfyUI's pin churn (section 2.3).
- **17:52-~18:10Z:** Strata restarted with a new layout.

The clean comparisons are against the 16:54-17:06Z baseline.
- 17:36-17:46Z: kcompactd1 went from 14.9 % to 0.9 % of a core, and failed migrations from 30.3 k/s to 9.0 k/s.
- 18:14-18:27Z, with everything in place: kcompactd1 0.0 %, failed migrations 1.8/s, NUMA page migrations 0.4/s, no swap traffic.
  - Compaction succeeds again (0.4 successes/s). Node 1 reached 43,648 free blocks of 256 KiB or more, against 0 in every baseline sample.
- Proactive compaction wakes fell from 72/s to 0.1/s the moment it was switched off.

The remaining file churn in the last window (73 k pages/s scanned, 59 k/s refaulted, ~290 MB/s read) came from readers, not from the kernel. Part of it was the run-0 CPU eval unit re-reading a checkpoint under its own `MemoryHigh=24G` until 18:24Z (section 4, change 6).

Speed, report only:
- Strata decode:
  - baseline: 1.6 tok/s token-weighted, median 2.0 per request (GPU6 alone);
  - 17:36-17:46Z: 3.0 tok/s, median 4.3;
  - 18:14-18:27Z: 15.8 tok/s, median 16.9 (5 requests). That is mostly the GPU0+6 layout, with background paused.
  - Prompt: 68-83 tok/s before the restart, 162 tok/s after.
- Run-0 step time: steps 1227-1234 took 206-258 s, steps 1238-1243 took 212-267 s and steps 1250-1253 took 211-265 s. No change attributable to these settings.

### 2.2 Who wakes kswapd and kcompactd (wakewatch, 17:20-17:30Z)

- High-order allocations from `python` processes, order 7-10 (512 KiB-4 MiB contiguous) on both nodes: ComfyUI/CUDA and the trainer.
- `python3` at order 9: 2 MiB transparent huge pages.
- `rsync` at order 6: the NAS copies.
- `khugepaged` at order 9.

Every such request on node 1 meets a node whose movable blocks are studded with pinned pages: zero free blocks of 256 KiB or more in every baseline sample. kswapd then reclaims file cache and kcompactd scans without being able to assemble a block.

### 2.3 Pinned memory: who pins and unpins (pinwatch, 4 KiB pages per 5-min interval)

| Interval end (UTC) | Process | pinned | unpinned |
|---|---|---:|---:|
| 17:16:57 | 27B judge (llama-server, stopped) | 0 | 184,791 |
| 17:21:57 | ComfyUI general (pid 13215) | 3,744,693 | 0 |
| 17:26:57 | ComfyUI general | 14,979,816 | 14,727,608 |
| 17:31:57 | ComfyUI general | 8,123,728 | 10,786,812 |
| 17:56:57 | Strata, old process exiting / new process | 171,556 (new) | 10,775,430 (old) |

- ComfyUI 0.37 general pins model weights for async offload, up to its 24 GiB cap.
- gen-worker-media alternates Qwen-Image and ACE-Step every ~60 s job, so every job unpins one model and pins the next. That is 30-57 GiB pinned and unpinned per 5 min. Every pinned page is unmovable while pinned, and it sits on node 1 (GPU4).
- Strata pins its tier once, at start, and keeps it.
- Run 0's ranks hold ~0.3 GiB each of pinned CUDA host memory and did not churn.
- Transient `O_DIRECT` pins (`dd`, 256 MiB each) are normal and are released at once.

## 3. Rules of the road for this server

1. **No swap.**
   - RAM is 251 GiB, and every big tenant has a measured, bounded footprint.
   - Swap only moved serving and training memory onto a SATA disk while cold file copies stayed in RAM.
2. **Under pressure, the OOM killer chooses, in this order.**
   - Batch and data lanes die first: `OOMScoreAdjust=+500`, plus a `MemoryMax` from their measured peak so a runaway job dies inside its own cgroup.
   - Run-0 evals and archivers next: +300.
   - Run 0 (queue, supervisor, trainer, ranks): -300 (lead decision 11-04).
   - Serving gates and Strata survive longest: -500, because Sprite depends on them. sshd's listener sets -1000 itself.
   - Never put a `MemoryMax` on Strata or run 0.
   - The kernel OOM killer is the only killer: systemd-oomd's pressure kills are off for `user@1000` (change 6).
   - Without swap, `MemoryHigh` must sit above a job's anon peak, or the job stalls in reclaim.
3. **Watch `MemAvailable`.**
   - `buff/cache` is mostly reclaimable, and `Shmem` inside it is program memory.
   - Alert below 60 GiB.
   - Gate new big jobs on `MemAvailable`, as K3 does.
4. **Cache discipline for one-time big readers:** model loads, copies and moves, backups, checkpoint copies, archives.
   - Run them in a capped scope: `/home/cereal/ai-serve/memory/capped.sh --high 4G -- <cmd>` (a wrapper around `systemd-run --user --scope -p MemoryHigh=4G -p MemorySwapMax=0`; it works from cron). The kernel recycles their page cache inside the cap.
     - Proof: 502 MiB cached instead of 4,096 MiB for a 4 GiB read (`cache_cap_test.txt`).
     - Cost: at a 512M cap the read took 35.5 s instead of 18.2 s, measured during Strata's tier fill. Use 2-4G, not 512M.
   - Or read with `O_DIRECT`, or `posix_fadvise(DONTNEED)` after use. `k3_multicard` (fadvise drop), the 27B judge (`--load-mode dio`) and Strata's tier fill (fadvise after fill) already do this.
   - Never drop caches globally on a timer.
   - A one-off, targeted reclaim of one cgroup is fine: `echo 20G > /sys/fs/cgroup/<cg>/memory.reclaim`.
5. **Pinned memory is allocated once and kept.**
   - Pinned pools are sized at service start and live as long as the process. Strata's expert tier works this way.
   - A service that pins and unpins per job runs unpinned instead: ComfyUI `--disable-pinned-memory`, accepting the speed cost.
   - Every pinned byte is unmovable and fragments its NUMA node; keep it on the GPU's own node.
6. **The kernel does not chase what cannot move.**
   - `vm.compaction_proactiveness=0`, `vm.watermark_boost_factor=0`, `kernel.numa_balancing=0`.
   - On-demand compaction, for an allocation that needs it, still works.
   - llama.cpp itself warns when NUMA balancing is on.

## 4. Changes rolled out

Script: `/home/cereal/ai-serve/memory/memory_policy.sh`, with `status`, `sysctl <key>`, `sysctl-rollback <key>`, `persist`, `swap-off`, `swap-rollback`, `oom-apply`, `oom-rollback`, `caps-apply`, `caps-rollback`, `oomd-apply`, `oomd-rollback`, `run0-protect`, `run0-protect-rollback` and `rollback-all`. Wrapper: `capped.sh`. Backups: `/home/cereal/ai-serve/backups/memory-20261104/`. Log: `evidence/memory-20261104/policy.log`.

| # | When (UTC) | Change | Rollback |
|---|---|---|---|
| 1a | 17:07:22 | `vm.compaction_proactiveness` 20 -> 0 | `memory_policy.sh sysctl-rollback vm.compaction_proactiveness` |
| 1b | 17:20:19 | `vm.watermark_boost_factor` 15000 -> 0 | `... sysctl-rollback vm.watermark_boost_factor` |
| 1c | 17:35:56 | `kernel.numa_balancing` 1 -> 0 | `... sysctl-rollback kernel.numa_balancing` |
| - | 18:00:50 | All three persisted in `/etc/sysctl.d/90-ai-server-memory.conf`, boot values in its header. Nothing else on the box sets them. | `rollback-all` removes the file |
| 2 | 17:46:35-17:50:02 | `swapoff /swap.img`. 2.6 GiB came back; MemAvailable went from 174 to 168 GiB. The `/etc/fstab` line is commented (backup `fstab.before`) and `swap.img.swap` is gone. `/swap.img` is kept until **2026-11-11**, then deleted by change 13. | `memory_policy.sh swap-rollback` |
| 3 | 17:52:35 | OOM order: drop-ins `90-oom-order.conf` plus live `/proc` writes, no restarts. Old live values are in `oom-live-before.tsv`.<br>Batch +500. System units: `omni-lane-*`, `gen-worker-*`, `gpu4-recaption`, `run0-eval-watch`, `media-judge`, `gpu4-judge`. User units: `omni-lane-*`, `omni-lane@`, `omni-run0-evals(-cpu)`, `omni-run0-milestones`, `omni-data-index`, `run0-tts-captions`.<br>Protected -500: `strata-gate`, `media-gate`. | `memory_policy.sh oom-rollback` |
| 4 | 17:53:13 | MemoryHigh 3G / MemoryMax 4G on the uncapped batch clients: `omni-lane-source-audit-strata` (peak 25 MB) and `gen-worker-text` (inactive). `judge2` is masked and was skipped. Every other lane already had MemoryMax 4G. | `memory_policy.sh caps-rollback` |
| 5 | 17:59:51 | ComfyUI `comfyui_qwen21_gpu4` registry: `--disable-pinned-memory`. General was down (the gate was running h3sol). The gate's next switch runs `manage_comfy_instances.py up --adopt`, which recreates the container from the registry. No forced restart. | `/home/cereal/ai-serve/memory/comfy_pins.sh rollback` (quiet window) |
| 6 | 18:28 | systemd-oomd: `user@1000.service` `ManagedOOMMemoryPressure` kill -> auto (set-property). See below. | `memory_policy.sh oomd-rollback` |
| 7 | 18:35:49 | Run 0 to -300, live: root `/proc` write on the whole run-0 tree (queue, claim-run, supervisor, trainer, tracker, 4 ranks). The ranks were at +500, the rest at 0. Lead decision. No relaunch. | `memory_policy.sh run0-protect-rollback` |
| 8 | 18:37-18:44 | Run 0's launch default, chimera-omni `claude/run0-blocks-20261029`, deployed with `deploy_live.sh` (DEPLOYED.json 84b3e58). It takes effect at the next supervisor start.<br>d27941d: `run0_queue.sh --oom-score-adj -300`; `v2_trainer._apply_rank_os` falls back to `sudo -n tee` when lowering is refused; with a test.<br>84b3e58: the queue lowers itself first, so the whole tree inherits -300 instead of the omni-run0 unit's +200. | `deploy_live.sh` with the `chimera-omni-v2.prev-43959a1` backups |
| 9 | 18:40:14 | Evals and archivers to +300: system `run0-eval-watch`, `omni-run0-evals-cpu`; user `omni-run0-evals(-cpu)`, `omni-run0-milestones`. | `memory_policy.sh oom-rollback` |
| 10 | 18:50 | Backups leave no cold copy.<br>Both backup crons (agent-memory 02:20, omni_lab/chimera_corpus 02:50 local) run through `capped.sh --high 4G --reclaim-container agent-memory-postgres --keep-gib 8`.<br>`install-cron` in both backup scripts writes the same line (`CACHE_WRAP`, canonical skill `.agents/skills/agent-memory-backup/scripts/` and the server `bin/` copies).<br>One-off: `DONTNEED` on the in-flight 29 GiB dump (12 GiB released), and the Postgres container reclaimed 24.3 -> 8.0 GiB. Cached went from 143.8 to 114.6 GiB; MemAvailable was unchanged at 184 GiB. | crontab `crontab.before-capped`; scripts `*.before-capped` in the backups dir; or `CACHE_WRAP= ... install-cron` |
| 11 | 18:56 | Milestone archiver (user `omni-run0-milestones`): MemoryHigh 4G. Its checkpoint copies keep their cache inside 4 GiB; its anon peak is 14 MB. | `rm ~/.config/systemd/user.control/omni-run0-milestones.service.d/50-MemoryHigh.conf`, then `systemctl --user daemon-reload` |
| 12 | 11-05 00:56 | **SUPERSEDED 11-05 ~01:20Z** (Cyril: "[Cyril quote withheld]"). Registry-only h3sol `--cache-ram 4 16` -> `--cache-ram 12 16` (lead GO, Relay 204374). It never ran: h3sol did not start before it was superseded. At 01:24:01Z my mistaken test call of the old `h3sol_cache.sh rollback` put the registry back to `4 16` (Relay 204383), the exact 00:56Z backup. `h3sol_cache.sh` 1.0.1 now refuses apply and rollback. The GPU-layout session owns the h3sol args. | none: do not edit the h3sol registry args |
| 13 | 11-05 00:56 | `/swap.img` (8 GiB) deletion armed: system timer `swap-img-delete-20261111.timer` (2026-11-11 18:00Z, Persistent) runs root-owned `/usr/local/sbin/swap_img_delete.sh`. It refuses while the file is active swap or named in fstab, and logs to `policy.log`. Size recorded in `swap-img-bytes.txt`. `memory_policy.sh` 1.1.0 `swap-rollback` recreates the file if it is gone. | Cancel: `sudo systemctl disable --now swap-img-delete-20261111.timer` |
| 14 | 11-05 01:09 | Orphan sweep (lead request; read-only, never kills).<br>Server: user timer `orphan-sweep.timer` (:05/:35) runs `orphan_sweep.py`. It reports python processes with a dead parent (ppid 1 or a `systemd --user` subreaper) that are not a service's MainPID, are above 500 MB RSS and used 0 CPU over 60 s. Small processes in the same scope and cwd (resource trackers) are listed as siblings, and secrets in command lines are redacted. Each run logs a line in `evidence/orphan-sweep/orphan-sweep.log`, and groups not seen before go to `outbox/`.<br>Workstation: task `OrphanSweepRelay` (:10/:40, `C:\Users\cereal\Documents\ai training\orphan-sweep-relay\orphan_sweep_relay.py`) posts each outbox report to Relay as a STATUS and moves it to `outbox/posted/`. The server holds no Relay credential. Verified end to end (Relay 204378). | `systemctl --user disable --now orphan-sweep.timer`; `Unregister-ScheduledTask OrphanSweepRelay` |

**Change 5, the trial (lead-approved).**
- General came back unpinned at 18:44:30Z, through the gate's own switch, with no forced restart. The process runs `--disable-pinned-memory`, and its log says "RAM limited by cgroup to 40960 MB". The "Enabled pinned memory" lines at its start are the image replaying the old comfyui.log.
- Baseline, pinned, all gen-worker-media tasks 11-01..11-04 18:41Z (`gen-tasks-before.txt`, task start to done, validity = every step ok):

  | Task | n | Median | Min-max | All steps ok |
  |---|---:|---:|---|---:|
  | image_edit | 88 | 69.5 s | 24-172 | 88/88 |
  | image_t2i | 140 | 18.3 s | -3482 (task-id reuse) to 94 | 140/140 |
  | mesh_hy3d | 48 | 126.9 s | 85-202 | 46/48 |
  | music_ace | 189 | 45.5 s | 9-218 | 189/189 |
  | video_h3sol | 14 | 518.7 s | 457-780 | 14/14 |
  | video_ltx | 12 | 187.6 s | 172-272 | 12/12 |

- After, unpinned, 11-04 18:44:30Z to 11-05 00:33Z (~6 h, `gen_tasks.py "2026-11-04 18:44:30 UTC"`):

  | Task | n | Median | Min-max | All steps ok | vs pinned |
  |---|---:|---:|---|---:|---:|
  | image_edit | 62 | 36.4 s | 21-124 | 62/62 | -48 % |
  | image_t2i | 119 | 15.2 s | 15-79 | 119/119 | -17 % |
  | mesh_hy3d | 31 | 104.4 s | 76-230 | 31/31 | -18 % |
  | music_ace | 101 | 21.5 s | 12-239 | 101/101 | -53 % |
  | video_h3sol (control) | 19 | 417.9 s | 337-578 | 19/19 | -19 % |
  | video_ltx | 4 | 196.6 s | 145-495 | 4/4 | +5 % |

  The table counts only finished tasks. One more h3sol attempt (task 15769) failed when the h3sol container hit its own memory cap at 23:58:33Z. It was requeued with a 5-min backoff and reruns in the next h3sol stint. That container is not part of this change (section 5).
- Model-loading jobs, by ComfyUI's own timer. This is the fairer test of the flag, because batching only removes reloads:
  - Qwen-Image load: 61.8 s pinned (n=5, 17:15-17:37Z). Unpinned: 25.6 s (n=10, first container), 24.2 s (n=7, second).
  - ACE-Step load: 54.1 s pinned (n=3). Unpinned: 127.1 s (n=3, range 33-179), 27.9 s (n=2).
  - LTX load: 267 s pinned (n=1). Unpinned: 491.6 s with its text encoder, 229.2 s without (n=1 each). LTX is the only type that may be slower. Its 4 tasks all ran in the first hour, which is too few to call.
- Pins: the host acquired 0.41 GiB of FOLL_PIN pages in the hour to 00:37Z, and 0 MiB in a 30 s check at 00:34Z. Pinned, it was 114.8 GiB in 13 min (17:20-17:33Z). The snapshots' absolute "acquired - released" figure goes negative, so only the rate is usable.
- General container: anon 16.6-16.8 GiB, file 22.9-23.3 GiB, shmem 0.2 GiB. Its file cache recycles inside the 40 GiB cap.
- The flag survives the gate's switches. The container stopped for h3sol stints and came back at 21:11:43Z and again at 00:07:10Z, still running `--disable-pinned-memory`.
- Confounders:
  - gen-worker-media 1.5.0 (GPU-layout session) went live at the same time. It batches 12 tasks or 15 min per model family, so most jobs no longer reload. Compare the per-type medians and the model-loading jobs, not the throughput.
  - h3sol already ran unpinned (`--disable-pinned-memory` since 10-29), so video_h3sol is a control. It too got 19 % faster, which shows how much the batching moves per-task times.
  - TTS runs in the speech workers, not ComfyUI, and is unaffected.
- **Verdict: keep.**
  - Every type except LTX is as fast or faster. Most of the gain comes from batching, which hides any per-load cost.
  - All finished tasks are valid, and the pin churn is gone.
  - Re-check LTX once n >= 5. The watcher keeps appending until ~06:51Z.
- If the cost is unacceptable: `comfy_pins.sh rollback` (quiet window).

**Change 6, why.**
- Ubuntu's `user@.service.d/10-oomd-user-service-defaults.conf` lets systemd-oomd kill the user unit with the most reclaim when `user@1000.service` stays above 50 % memory pressure for 20 s. It ignores `oom_score_adj`.
- It killed `omni-run0-evals-cpu` twice at step 1250, at 18:18Z and 18:24Z (Relay 204313/204314). That unit's `MemoryHigh=24G` sat below its ≥ 24.7 GiB checkpoint load, so it stalled in reclaim.
- Since 18:19Z run 0 itself runs there (`omni-run0.service` ADOPT), so it was exposed to the same kill.
- Now `oomctl` monitors no cgroup, and the kernel OOM order is the only killer.
- Rule for unit owners: **with swap off, a `MemoryHigh` below a job's anon peak no longer swaps; it stalls the job.** Set `MemoryHigh` above the measured peak, or use only `MemoryMax`.
- Since change 6: no systemd-oomd kill. The 18:14:54Z and 18:18:33Z kills in its log (evals-cpu, source-audit-worker) came before it.

**Re-check, 11-05 00:37Z (the last hour of memwatch; `hold_check.py 60`).**
- Swap is 0. kcompactd0 uses 0.5 % of a core and kcompactd1 0.01 %; kswapd uses 0.3 % per node. NUMA migrations: 0.02/s.
- Failed migrations: 725/s. They come from direct compaction (compact_stall 0.34/s) during the busiest hour so far:
  - MemAvailable fell to 122 GiB (now 170).
  - Node 1's free order>=6 blocks briefly reached 0 (now 12,657).
  - That is higher than the 1.8-162/s of the earlier after-windows, but still 42x below the 30,328/s before.
- Shmem 27.9 GiB; Cached 145.9 GiB.
- Run-0 CPU evals (Hermes session's system unit; first full run at step 1300, 23:36-00:28Z):
  - memory.peak 35.3 GiB.
  - anon held 8.1-8.8 GiB, with bursts to 16.5 and 26.8 GiB. file was 7.8 GiB (the checkpoint).
  - MemoryHigh 44G leaves ~17 GiB above the anon peak, so the limits stay.
  - The checkpoint's 7.8 GiB of cache used to stay charged to the unit after the run. **DONE by the Hermes session, 11-05 ~00:45Z:**
    - `run0/evals/eval_step_cpu.sh` now calls `posix_fadvise(DONTNEED)` on every checkpoint file once all of a step's probes are done.
    - It doesn't do this after each probe, because the six probes re-read the checkpoint from the slow Lexar.
    - Tested on step 1300: the unit's MemoryCurrent went from 8.43 GB to 0.13 GB.
    - Backup: `eval_step_cpu.sh.bak-20261105T0045Z`.

**Root-disk I/O, 11-05 02:12-02:16Z (lead request; read-only; `io_writers.py`, report `evidence/io-20261105/io-writers-0212.txt`).**
- The disk: `/` is a Crucial BX500 2 TB (`CT2000BX500SSD1`), DRAM-less SATA, 94 % full (108 GiB free). It holds /home (the run-0 run dir, lanes, Hermes), Docker (agent-memory Postgres), journald and the ComfyUI models. Run-0 checkpoints go to the Lexar (`/mnt/lexar`, sda2 exfat).
- Over 240 s:
  - 69 % busy; writes 28.2 MB/s at 115/s (avg 239 KiB, await 84 ms); reads 76 MB/s; average queue depth 33.
  - Flushes 1.9/s at 88 ms each.
  - 8 MiB `dd conv=fsync` probes: 0.04 s, 6.76 s and 7.43 s.
- Writers:
  - 23.3 MB/s (83 %): the LTX fp8 download (curl, user unit `ltx-fp8-get-20261105`).
  - 3.1 MB/s: `omni-lane-stories-personality-gen`.
  - 0.73 MB/s at 13.6 w/s: agent-memory Postgres. It makes ~10 sync_file_range, 1.1 fdatasync and 0.3 fsync per s, so it is the one that feels the stalls.
  - 0.54 MB/s: journald.
- Reads: ~75 MB/s from the h3sol container re-reading weights with caching off.
- Verdict: the stalls are queueing behind one streaming writer on a slow, nearly full DRAM-less SSD. Writeback is already small (`dirty_bytes` 512 MiB, background 128 MiB; Dirty peaked at 299 MiB), so the fix is to rate-limit big writers, not to tune sysctls.
- Done by the GPU-layout session at 02:31Z: the download was restarted at `--limit-rate 10M`.
- Run 0's step-1350 checkpoint did not stall: 8.3 GB in 63 s (02:33:04Z), against 64 s at 1260 and 58 s at 1300. Its sample dump to `/` took 84 s (81-130 s before).
- Proposed: delete the 43 GiB BF16 LTX once the fp8 A/B passes; the lead decides. Optionally move the stories lane's output to the Lexar.

## 5. Held back, and proposals for their owners

- **Run 0's OOM score: DONE** (changes 7 and 8). Run 0's oom_score is now 467-474. The batch lanes (+500) come before it in the OOM line; Strata (score ~450, -500) dies after it.
- **gen-worker-media job order: DONE by the GPU-layout session** (1.5.0, batches by model family).
- **Lexar-to-NAS mover (disk-cleanup session): DONE by that session.**
  - Since 18:34Z its moves run one at a time through `run0/maint/disk-cleanup-20261104/move_queue.sh`, each in a `systemd-run --user --scope -p MemoryHigh=4G` (memory.high verified).
  - The uncapped movers of 15:44-17:44Z (the 16 GiB seen at 16:51Z) are stopped.
  - Its ad-hoc copies use `capped.sh`.
- **Strata (GPU-layout session owns its restarts): no change needed.**
  - The 21.5 GiB pinned tier is allocated once. `--cache-ram 16384` held 5.2 GiB.
  - Worst-case non-reclaimable memory is ~46 GiB, under MemoryHigh 64G.
  - The GGUF load already leaves only 1.7 GiB cached (1.4 GiB of hot PLE rows).
  - The cold 0.65 GiB of mmproj cache was dropped by that session with a one-off fadvise; it noted an ExecStartPost version for the next restart.
- **h3sol container OOM kill, 23:58:33Z. My fix (change 12) was the WRONG APPROACH and is superseded. The GPU-layout session owns the real fix.**
  - What happened: `comfyui_h3_sol_gpu4` hit its own 40 GiB cap (memory.max 41,943,040 kB, failcnt 461,448) with anon 39.7 GiB, file 0.2 and shmem 0.2. The kernel killed its python. gen-worker-media logged task 15769 (video_h3sol) as "lost", but it was requeued (attempts 1, 5-min backoff). It is the only memcg OOM kill since 11-01.
  - It is not the pins trial, because h3sol has run `--disable-pinned-memory` since 10-29. Two things changed under it on 11-04:
    - Swap is off. The container's memory.swap.max is unlimited, so an overshoot above memory.max used to swap; now it is a kill.
    - gen-worker-media 1.5.0's batching keeps the container up across many h3sol tasks. The last stint ran 82 min and 9 tasks.
    - GPU-layout session's count from media-gate and worker leases, 10-29..11-05: h3sol tasks per container stint were 3-8 (median 4) before 1.5.0, and 8 and 10 with it. The 22:36Z stint also ran 5 music_ace and 1 t2i, because every GPU4 task runs in whichever container is up.
  - memwatch over today's three h3sol stints: anon peaks of 26.4, 36.4 and 39.5 GiB. In the last one, anon climbed from 3.4 to 33.3 GiB in ~50 min (about 4 GiB per task), held at 33.3 for ~15 min, then rose 38.2 -> 39.5 -> kill.
  - Why:
    - `--cache-ram 4 16` tells ComfyUI's RAM-pressure cache to free its active cache only when less than 4 GB is free.
    - Inside a cgroup, ComfyUI counts free as memory.max - (memory.current - inactive_file) (`comfy/system_memory.py`).
    - So the cache fills the cap to within 4 GB, and a task that needs more than that overshoots. This one needed at least 6.2 GiB above the plateau.
  - The fix that replaces mine, from Cyril (11-05, verbatim): "[Cyril quote withheld]" The GPU-layout session (998ab6) owns it:
    - `--disable-smart-memory --cache-none` in place of any `--cache-ram`;
    - compact weights only;
    - shorter clips.
    - It writes the registry args itself. This session makes no h3sol edit.
  - Why the `--cache-ram` tuning was the wrong approach:
    - It kept a RAM cache in a capped batch container and only moved the point where the cache stops growing.
    - Every task is a new prompt, so cached node outputs are almost never reused. On this server the rule is "don't keep caching": with no swap, a cache that fills a cgroup is a kill waiting for one large task.
    - Turning the cache off removes the growth instead of tuning its ceiling.
    - Lesson: for a batch media container under a hard cap with no swap, turn caching off; don't tune the headroom.
  - Replaying the 22:36-23:58Z stint through `h3sol_watch.py` 1.1.0 (memwatch data) supports this:
    - 14 tasks ran: 9 video_h3sol and 5 music_ace.
    - anon at each task start (GiB; v = video, m = music): 9.5v 13.2v 16.5v 19.9v 23.2v 26.7v 29.9v 33.3m 33.3m 33.3m 33.1m 33.4m 33.4v 38.2v.
    - Each video task added about 3.4 GiB and nothing was given back; the music tasks added nothing. That is per-prompt caching, not a leak in the music path.
  - Watch, for measurement only: user unit `h3sol-watch-20261105`, `h3sol_watch.py` 1.1.0 (read-only, MemoryMax 256M).
    - It samples the container's cgroup every 10 s into `evidence/memory-20261104/h3sol-watch.tsv`.
    - At each stint's end it writes to `h3sol-stints.txt`: the anon peak, the anon at each task start, task starts, durations and validity per type, errors, memcg OOM kills, and the memory flags the container ran with.
    - Every stint before 998ab6's change lands is the "before".
    - It stops after the first stint on `--cache-none` with >= 6 tasks (lead 11-05), or after 7 days.
    - The GPU-layout session reads the same file.
  - **Result: the no-cache fix works for RAM.** The watcher stopped itself at 02:29:22Z after the first `--cache-none` stint.

    | Stint | Flags | Tasks | anon at video starts (GiB) | anon peak | memcg OOM | Video median |
    |---|---|---|---|---|---|---|
    | 11-04 22:36-23:58Z | `--cache-ram 4 16` | 9 h3sol + 5 music | 9.5 -> 38.2, +3.4 per task | 39.5 GiB (cap 40) | 1 kill, 1 failed attempt | 397 s (n=8) |
    | 11-05 01:26-02:27Z | `--disable-smart-memory --cache-none` | 6 h3sol + 2 music + 1 edit | 6.9, 5.5, 6.6, 6.9, 6.9 (flat) | 6.9 GiB | none, 0 errors | 632 s (n=6, 367-896) |

    - Every video task was valid (6/6). The GPU-layout session's clip trial (5 s / 4 s / 3 s) made no RAM difference.
    - The video tasks were slower, from two causes:
      - With caching off, anon+file sits at the 40 GiB cap, so every stage re-reads its weights from the BX500 (~75 MB/s of reads). H3 is 19.5 + 14.6 GiB and the BF16 LTX refine is 43 GiB, all inside a 40 GiB cgroup.
      - From ~01:43Z the LTX fp8 download (23 MB/s of writes) made those reads slow. The GPU-layout session's split: draft 200-217 s -> 333-510 s, refine 148-179 s -> 232-368 s.
    - The two tasks before the download took 367 and 440 s, which matches the pre-change median of ~418 s.
    - Expected next: the download is capped at 10M since 02:31Z and ends ~03:00Z. After that, the fp8 LTX (half the bytes per refine load) should bring task times back.
- **Orphans: DONE by the lead.**
  - 11-04: it SIGTERMed the v30 leakcheck pool (3021880-3021888) and the long-derive / ext_spread pools (231748/231749, 1242771, 1242815-1242819).
  - 11-05 ~01:00Z: three more mix-monitor pools (v36 derive, v37 and v38 leakcheck; ~3.2 GiB), found by my scan. The lead checked ppid 1 and 0 CPU over 8 s first; MemAvailable went from 164.7 to 167.7 GiB. Logged in `run0/mixmon/ORPHANS_KILLED.log`.
  - From now on change 14 reports them on Relay, and the lead decides.
- **Not changed, on purpose:**
  - `vm.swappiness`: irrelevant without swap.
  - `vm.min_free_kbytes` (88 MiB): no allocation stalls were measured (`allocstall` ~0/s), so no evidence for raising it.
  - THP (`madvise`/`madvise`).
  - Strata's `MemoryHigh=64G` (set-property). It caps how much of the tier fill's page cache Strata can hold, which is the behaviour rule 4 wants.
  - sysmon, the Sysmon-for-Linux EDR sensor (0.2-0.75 of a core): the lead's call.
- **Strata's 39.5 -> 21.5 GiB** came from the GPU-layout session's move back to GPU0+GPU6 (Cyril's 11-04 layout), not from this session.
