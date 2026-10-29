# Strata host-RAM budget: phase 2 design and staged plan (2026-10-28, FINAL)

Status: PLAN, nothing applied. Supersedes `STRATA_HOST_RAM_BUDGET_2026-10-28.md.draft.md` (kept for the critic trail). Phase 1 (Relay CLAIM 204084 / STATUS 204096, plan `STRATA_RAM_2026-10-28.md`, build `llamacpp-pcache`) owns `strata-llama` and `strata-gate` until it posts DONE or RELEASE on thread `52468f63-fbb0-429e-a316-1d17607d3912`; every stage here builds on its drop-ins, patches and backups and never reverts them. Section 9 records how each of the 13 critic findings was applied or rejected.

## 0. Order (Cyril, verbatim)

- 2026-10-28: "[Cyril quote withheld]"
- 2026-10-28, earlier: "[Cyril quote withheld]"
- 2026-10-28, priority (supersedes any speed gate): "[Cyril quote withheld]"
- 2026-10-27: "[Cyril quote withheld]"; expert cache: "[Cyril quote withheld]".

Reading: Strata gets a hard, bounded host-RAM budget. Everything that does not fit lives in VRAM (GPU0 3090 + GPU6 RTX 8000) or on disk, streamed through the GPUs; all math stays on the GPUs; the rest of system memory stays usable for run 0 (~20 GiB), media-gate background work (MemAvailable >= 40 GiB), agent-memory Postgres and workflows. Speed is measured and reported at every stage and is never a reason to keep RAM or roll back a RAM lever. Functional constraints still bind: correct output (greedy-identical to the reference build or explained), ~256k context, no output cap, Sprite/cyril succeed and preempt background in ms, follow-ups restore from cache tiers, no crashes/Xid, no swap growth.

## 1. Problem

Strata's non-reclaimable host footprint is ~147 GiB resident + 5.8 GiB swap (cgroup 158.1 GB, peak 190 GiB). 98.5 GiB of it is two fixed tensors no existing knob touches: a 71.7 GiB pinned `cudaMallocHost` copy of every expert (`-cmoe`, ~76 % of it duplicated in VRAM) and the 26.8 GiB per-layer n-gram table (`per_layer_token_embd`) held anonymous by `--lazy-mode off`. Phase 1 caps the prompt cache (48 -> 24 GiB) but cannot get below ~123 GiB. The target is ~48 GiB non-reclaimable with MemAvailable ~100 GiB while run 0 and media background run.

## 2. Measurements (2026-10-28, 20:41-21:45Z, read-only scouts; artifacts `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase2-scout/`, `/tmp/xc_gguf_probe.out`, `/tmp/xc_dmon.txt`)

Process: `-np 6 -c 262144 --kv-unified -ctk/-ctv q8_0 --ctx-checkpoints 16 --cache-ram 49152 -b/-ub 1024 -cmoe --no-mmap --lazy-mode off --numa distribute`, mmproj on CPU, `-ts 16,32`, `GGML_EXPERT_CACHE_MB=30000,50000`.

| Host part | GiB | Scales with |
|---|---:|---|
| Pinned experts: 144 `blk.N.ffn_{gate,up,down}_exps.weight` tensors, one 74,094 MiB `/dev/zero` mapping (cgroup charges 73.2 GiB shmem; RssShmem under-reports at 35.8) | 71.73 | fixed; not np/ckpt/b/ub/cache/context |
| `token_embd` Q8_0, same pinned mapping | 0.63 | fixed |
| Pinned output/CPU compute buffer (802 MiB) + 12 small pinned mappings | 0.81 | `-b`, `-ub` |
| PLE n-gram table, anon, `[160, 320001536]` IQ4_NL (22.0 in RAM + 4.83 in swap) | 26.82 | fixed while `--lazy-mode off`; tensor is `TENSOR_READ_LAZY` |
| RAM prompt cache (`--cache-ram 49152`, at cap) | 48.0 | `--cache-ram`; entry = (1 + checkpoints) x 112.2 MiB + 17.5 KiB/token |
| mmproj F16 on CPU + compute | ~1.3 | `--image-max-tokens` |
| Slot checkpoints (np x ckpt x 112.2 MiB worst), CPU compute, heap, CUDA runtime | ~3.6 | np, ckpt |
| Kernel (page tables, slab) | 0.49 | mapped RSS |
| Total | 147.2 resident + 5.76 swap | |

Derived: recurrent state per sequence = one checkpoint = 112.2 MiB; KV = 17.5 KiB/token (13,056 main q8_0 + 4,896 indexer). One expert = ~2.93 MiB per layer, 143.5 MiB across 48 layers; 512 experts, 10 active per token.

VRAM: GPU0 23,604/343 MiB used/free (layers 0-15), GPU6 44,812/390 MiB (16-47 + output). Expert slot caches ~18.3-19.1 GiB on GPU0 (~390-405 of 512 slots at 48.25 MiB) and ~34.7-35.7 GiB on GPU6 (~373-384 slots at 95.2 MiB); cap is free VRAM - 768 MiB, not the MB budgets. Indexer V cache ~0.8 GiB total allocated (`llama-kv-cache.cpp:231-234`) with no reader found in `qwen4exp.cpp:475-500`. QSA is active on this GGUF (compress_ratios = 4 on the 12 attention layers).

xcache behaviour (live tree `/home/cereal/omni-campaign/export/llamacpp-xcache/llama.cpp`, b96806d96 + `xcache.patch`; local copy `D:/SpriteBuild/strata-xcache-20261027/xcache.patch`): small path (<= 64 tokens) LRU per weight, miss = one h2d copy; large path (prefill) fills free slots only, never evicts, streams the cold set from host every batch. Measured PCIe during an 83.5k prefill: ~3.9 GB/s mean h2d, ~19 GiB per 1024-token batch, one batch per ~5.25 s, GPU0 and GPU6 never active in the same sample. Decode: 11.4 tok/s, 20-40 MB/token h2d, hit rate ~0.97 for one warm stream (3 samples). Coverage ~0.59 GPU0 / ~0.81 GPU6 / ~0.73 overall; cold set ~19-20 GiB.

Disk: no NVMe. Root BX500 SATA: 475 MB/s seq, 417-459 MB/s random 1-8 MiB, 3.9k/22.9k IOPS 4 KiB QD1/QD16 (p50 247/662 us), 108 GiB free. Lexar SL600 USB, exFAT (`/dev/sdb2`, `fmask=0022`): 385-446 MB/s seq, 314-447 MB/s mixed, 156 GiB free. exFAT cannot hold 0700 directories, so phase 1's disk tier is OFF in production (journal 21:21:10Z "not ... mode 0700; RAM prompt cache only"; `z-ramcap.conf` at 15:39 MDT no longer sets `STRATA_PCACHE_DIR`).

Swap: 8 GiB `/swap.img` ~full, swappiness 60; Strata holds 5.76 GiB (4.83 of it PLE pages). MemAvailable 21.9 GiB of 251.7. Largest non-Strata user: `comfyui_qwen21_gpu4` at 62.4 GiB anon at 20:41Z and 4.12 GiB at ~21:45Z (it swings). Strata `oom_score` 922 with `oom_score_adj` 0.

Incident during the scouts: 21:37Z the `llamacpp-pcache` build (PID 1531122) died with `CUDA error: an illegal memory access` and kernel `NVRM: Xid 31 ... MMU Fault` on GPU0; `Restart=always` brought it back ~2 min later (PID 1543369). Owner: phase 1. Gate for S2 (section 6).

## 3. Budget

"Non-reclaimable" = cgroup anon + shmem + swap of `strata-llama.service`. Page cache is outside the budget by definition (it is what "usable for everything else" means).

| Component | Today (GiB) | Budget (GiB) | Cap / mechanism | Stage |
|---|---:|---:|---|---|
| Expert host tier (experts not in VRAM; pinned; exclusive with VRAM) | 71.7 | 20 expected, 22 max | option (b) tier, `GGML_EXPERT_HOST_MB=22528` warning cap | S2 (+S3) |
| PLE n-gram table | 26.8 (4.8 swap) | 0 (page cache, <= 27 reclaimable) | drop `--lazy-mode off` + `ple-prefetch.patch` | S4 |
| RAM prompt cache | 48 | 16 (24 until phase 1's disk tier is proven) | `--cache-ram` | S1/S3 |
| Spill queue (phase 1) | n/a | 2 (phase 1 ships 4) | `STRATA_PCACHE_SPILL_QUEUE_GIB` | S1 |
| Slot checkpoints | <= 10.5 worst | 3.5 worst (4 x 8), 1-2 typical | `-np 4 --ctx-checkpoints 8` | S1 |
| Pinned output/compute, CUDA runtime, heap, disk index, page tables | ~4.1 | 5 | measured, not capped | - |
| mmproj + compute | ~1.3 | 1.5 | unchanged | - |
| **Total** | **~147 + 5.8 swap** | **~48 worst / ~44 typical** | | |

Progression (measured at each gate, never assumed): today ~153 -> phase 1 ~123 (re-measured in step 0; the disk tier is currently off) -> S1 ~110 -> S2 ~59 -> S3 ~57 -> S4 ~47 worst / ~44 typical.

MemAvailable: 251.7 - Strata 48 - ComfyUI at its measured peak 62.4 - run 0 20 - other/kernel/watermarks ~16 = ~105 GiB. Budgeted against ComfyUI's peak, not its average (critic 7). The >= 100 target holds only while that container stays at or below 62.4 GiB; it is outside this lane and is reported to the media lane.

Floor statement: S1-S3 only (table resident) floors at ~57 GiB. ~48 requires S4. If S4 is blocked: `--cache-ram 8192` (-> ~51), `-np 2` (-> -1.75 worst), or S5 disk bottom tier (-> ~41 with every prompt batch reading disk, ~2.3 s per batch per GiB removed).

Settings decided (each a one-line revert):
- `-np 4` (was 6); gate `--slots 4 --bg-max 2` (was 6/4): Sprite + cyril pinned slots + 2 background. `-np 2` is in reserve, taken if the S4 soak needs it (saves ~1.75 GiB worst checkpoints + 0.22 GiB VRAM).
- `--ctx-checkpoints 8` (was 16, hard-coded at `strata_llama.sh:114`).
- `-b 1024 -ub 1024` kept unless S1's verbose start shows `-ub 512` frees >= 1 GiB VRAM (prefill PCIe per token doubles otherwise, for an unmeasured gain).
- `--cache-ram 16384` once phase 1's disk restores are proven; 24576 until then.
- KV q8_0, `-c 262144`, `--kv-unified`, no output cap: unchanged.

## 4. Mechanism

### 4.1 Expert tiering: exclusive pinned host tier under the VRAM LRU (option b); no disk on the hot path

Invariant: every expert is in exactly one of {VRAM slot, pinned host slot}. Per weight: `n_slots` VRAM slots + 1 spare VRAM slot; a pinned host tier of `n_expert - n_slots` slots + 512 B trailing padding (critic 13: the large path reads 512 B past each expert at `xcache.patch:342, 383`). The loader's full host copy no longer exists; experts are file-backed only during the startup fill.

Data movement:
- Startup fill (first sizing, triggered by llama's warmup graph): experts 0..n_slots-1 -> VRAM slots (h2d from the mapping), the rest -> host tier (memcpy from the mapping); per weight range `madvise(MADV_WILLNEED)`/`MADV_SEQUENTIAL` first so the fill runs at ~475 MB/s sequential (~2.5-3 min for 72 GiB) instead of 4 KiB faults under the `--numa` `MADV_RANDOM`. After the fill: `madvise(MADV_DONTNEED)` drops the PTEs and the loader-side hook `llama_model_drop_mmap_ranges()` issues `posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)` per expert range so the 72 GiB does not sit in page cache displacing other services' cache (critic 10; `MADV_DONTNEED` alone does not drop page cache on a `MAP_SHARED` mapping). The mapping is never read again in steady state; the tier is the source of truth.
- Decode miss (small path): expert Y in host slot `hY`; victim X = LRU VRAM slot `s` not used this step and **not the spare** (critic 13). On the backend's stream, in order: (1) h2d `host[hY] -> vram[spare]`; (2) d2h `vram[s] -> host[hY]`. Bookkeeping: `slot_of[Y]=spare`, `expert_in[spare]=Y`, `loc[X]={host,hY}`, `spare=s`. Both copies on one stream, so no event. d2h rides the idle direction of the full-duplex link (~20-40 MB/token).
- Prefill (large path): cached experts d2d as today; cold experts copied one by one from their host slots (adjacent host slots coalesced when contiguous), replacing the contiguous-run copy at ~2122-2139. ~21k copies per 1024-token batch, ~0.2 s overhead. Free-slot fill unchanged; prefill still never evicts.
- NUMA: each GPU's tier allocated after `numa_set_preferred(node_of(gpu))` (GPU0 node 0, GPU6 node 1), verified with `numastat -p`. Refinement, not a gate.
- Debug-build asserts on every slot and host-slot index, on `spare != victim`, on `loc[]` consistency after each miss, and on tier bytes <= allocation (critic 3, 13).

Fatal at startup, never WARN (critic 4): `n_slots <= 0` for any weight; VRAM slot-cache allocation failure (`entry_get` nullptr); `GGML_EXPERT_TIER=1` with any `_exps` tensor not file-backed (the tier would duplicate the pinned pool: +20 GiB, critic 1); `GGML_EXPERT_TIER=0` with any `_exps` tensor file-backed (every miss would be a serial 4 KiB fault through `MADV_RANDOM`: ~16 MB/s, ~20 min per batch). The old whole-weight copy fallback is removed from the tier path.

Loader patch `exps-mmap.patch` (`src/llama-model-loader.cpp` ~1081-1107 and 1228-1270, `src/llama-mmap.cpp`, `common/arg.cpp`): flag `--mmap-tensors <regex>` makes matching tensors file-backed (plain CPU buffer type, `MAP_SHARED`) while everything else keeps today's allocation, overriding `--no-mmap` for those tensors only. Strata passes `--mmap-tensors '\.ffn_(up|gate|down)_exps\.weight$'` (critic 1: the draft's `_exps$` matches nothing; tensor names are `blk.N.ffn_gate_exps.weight` etc.). At the end of load the loader logs `mmap-tensors: matched N tensors, B bytes` and exits non-zero if N differs from `--mmap-tensors-expect` (144 for this GGUF). `GGML_CUDA_REGISTER_HOST` stays unset. `--no-host` / `GGML_CUDA_NO_PINNED` are not used: they only turn the pool into anon memory (not reclaimable, swap full).

Knobs: `GGML_EXPERT_TIER=1` (0 = old behaviour, only valid with `--no-mmap`; used for the S2 smoke, not as a rollback), `GGML_EXPERT_HOST_MB` (warning cap; a tier above it logs WARN with the measured number, never refuses, because the correct tier size is dictated by VRAM), `GGML_EXPERT_CACHE_STATS=60`.

Stats: `xtier.patch` promotes the sizing line and `ggml_sched_xcache_stats` to `GGML_LOG_WARN` (xcache's INFO maps to TRACE and is filtered at default verbosity) and adds `d2h_bytes`, `tier_bytes`, `n_slots` per GPU, `fill_s`, `disk_misses` (S5 only).

Expected: host 72.4 -> ~20 GiB pinned; h2d bytes unchanged; d2h +20-40 MB/token in decode; outputs token-identical (same experts, same kernels, placement only). Startup +2.5-3 min.

Code: `xtier.patch` over `ggml/src/ggml-backend.cpp` (~450-700 lines), `exps-mmap.patch` (~80-120 lines incl. the fadvise hook), applied by `apply_xcache.py` after `xcache.patch` and phase 1's `strata-pcache.patch`, in `/home/cereal/omni-campaign/export/llamacpp-tier/`.

Why not the simpler mmap-only route (option a without a tier) given "don't worry about speed": it puts ~19 GiB of disk reads on every prompt batch when the page cache is cold (~45 s per 1024 tokens, the 10-21 failure mode), and a background batch mid-read cannot be preempted, which breaks the Sprite functional constraint, not just speed. Option (b) keeps today's byte traffic and has no disk on the hot path.

### 4.2 VRAM reclaim (each GiB of VRAM = 1 GiB less host tier)

- `idx-nov.patch` behind `LLAMA_QWEN4EXP_IDX_NO_V=1`: skip the indexer V allocation (~0.8 GiB total) if step 0 confirms no reader. Greedy equality is the guard. Dropped if a reader is found.
- `-np 6 -> 4`: 0.22 GiB.
- `-ub 512`: decided from S1's measured compute-buffer line; applied only if >= 1 GiB freed.
- Rejected: smaller context or KV type (behaviour change); reducing the 768 MiB keep headroom (cuBLAS workspaces); mmproj to VRAM (zero-sum).

### 4.3 N-gram table: lazy mmap + per-ubatch row prefetch (S4)

The table is a CPU gather, not math. Row ids are computed on the CPU in `llm_graph_input_ple::set_input` (`qwen4exp.cpp` ~1048-1108) before the graph runs (critic confirmed); k = `ple_n_heads` = (`ngram_size` - 1) x `heads_per_ngram` per token, one PLE layer only. The numeric values of `ngram_size` and `heads_per_ngram` are read in step 0 from the GGUF KV with the header-only probe (`/tmp/xc_gguf_probe.py`; the server venv has no numpy).

Mechanism: drop `--lazy-mode off` (the loader then mmaps this `TENSOR_READ_LAZY` tensor, `llama-model-loader.cpp:1081-1105, 1335`), and `ple-prefetch.patch` in `set_input` computes the distinct 4 KiB pages of the ubatch's rows and issues one `madvise(MADV_WILLNEED)` batch (or `readahead()`) before the graph runs, so cold rows arrive at queue depth instead of serially through the fault handler. Cost bound: decode <= k x 0.25 ms/token when every row misses (k <= 8 -> <= 2 ms against ~42-88 ms/token); prefill <= 0.4 s per 1024-token batch at k = 8, QD16. If k > 16 or the ids are not available on the CPU, S4 is re-planned, not forced.

Rejected: keep anon (floor ~57); table in VRAM (zero-sum against expert slots); `MemoryLow=`/`MemoryMin=` (makes it non-reclaimable in effect while MemAvailable still counts it); hand-rolled pinned row cache (the page cache is one).

### 4.4 Prompt cache and slots (S1), config only

`-np 4 --ctx-checkpoints 8`, `--cache-ram 24576` (16384 once disk restores are proven), spill queue 2 GiB; gate `ExecStart` overridden to carry `--slots 4 --bg-max 2` (critic 2: the gate hard-codes `slots=6` at `strata_gate.py:482`, computes `first_bg = slots - bg_max` at `:323-324`, and llama-server wraps `id_slot % slots.size()` at `server-context.cpp:1548-1550`, so without `--slots 4` background lands on slots 0 and 1, the Sprite and cyril slots, defeating preemption and evicting their prefixes). Gate 1.4.1 reads llama-server's `/slots` count at startup and refuses to start when it differs from `--slots`.

Phase 1's disk entries carry `llama_build_info()` and `np` in their ident (`strata-pcache.patch:69-73`), so S1 and every new build invalidate all existing entries; the stale files keep counting toward `MAX_GIB`/`MAX_FILES` for 168 h. Phase 1 is asked (step 0) to purge stale-ident entries at startup; if it declines, `strata-pcache.patch` v2 on the tier branch does it. One round of foreground re-reads after each restart is expected and reported (critic 12).

### 4.5 Enforcement: caps, pre-flight, watchdog; never OOM-kill

- Hard caps in llama-server: `--cache-ram`, spill queue, `np x ckpt`, host tier (fixed allocation sized once), mmproj. After S4 the only unbounded host memory is page cache (reclaimable) and heap growth (a bug, caught by the watchdog).
- systemd: no `MemoryMax=` (OOM-kills the service), no `MemoryHigh=` (throttles the serving loop). `OOMScoreAdjust=-500` in `z2-budget.conf` so the kernel OOM killer does not pick Strata first (critic 7; today `oom_score` 922).
- Launcher `strata_llama.sh` 1.6.0 computes the budget from the actual flags per stage, not from a fixed list (critic 8): `cache-ram + spill + np x ckpt x 112 MiB + (72.4 GiB if --no-mmap and TIER=0, else GGML_EXPERT_HOST_MB) + (26.8 GiB if --lazy-mode off) + 8`, logs it, and refuses to start when it exceeds `STRATA_RAM_BUDGET_GIB` (set per stage: S1 130, S2 70, S3 65, S4 52). A refusal writes `journalctl -p err` plus `/home/cereal/ai-serve/strata/REFUSED` with the computed number. Because `Restart=always`/`RestartSec=20`/`StartLimitIntervalSec=0` would otherwise retry silently forever, the same formula runs first as `strata_ram.sh check` against the drop-ins before `daemon-reload`, and the watchdog alerts when the unit restarts more than 3 times in 10 min or the REFUSED marker exists.
- Launcher asserts that the final argument string contains `-cmoe`, `--kv-unified`, `-c 262144` and, from S2, `--mmap-tensors` (critic 6: `STRATA_EXTRA` is replaced by the last `Environment=` line, not appended; every stage writes the full string).
- Watchdog `strata_ram_watch.py` (`/home/cereal/ai-serve/strata/`, systemd timer every 60 s; raw `psycopg2`, manual connect/commit/close; DB credential from the environment file `capture.py` already uses, vault key name verified in step 0, never on a command line): samples `memory.stat` of `strata-llama.service`, `/proc/meminfo`, phase 1's `/metrics` gauges, the latest `xtier:` stats line, `majflt` delta, GPU0/GPU6 used MiB; one row per sample (section 5). Alerts (journal + Relay STATUS, one per condition per 30 min): non-reclaimable > budget + 4 GiB for 5 samples; MemAvailable < 60 GiB for 5 samples; swap growth > 1 GiB over 60 min; `majflt` > 5/s over 5 min after S4; `_spill_dropped_total` growth; unit restart storm; REFUSED marker. It never signals, stops or restarts anything.

### 4.6 Measurement hooks

- S1 restart only: `Environment=LLAMA_ARG_LOG_VERBOSITY=4` (reveals slot counts per GPU, compute-buffer sizes, baseline hit %); removed at S2 when the WARN-level stats land.
- Independent of logs: `nvidia-smi dmon -i 0,6 -s t` over decode-only and prefill-only windows; `print_timing` tok/s; `/proc/<pid>/stat` majflt; `numastat -p`.
- Prefill is also measured with run 0 active on GPU1, because GPU0 and GPU1 share a PCIe switch (`nvidia-smi topo` PIX) and run 0's traffic competes with both the h2d stream and (b)'s d2h demotions (critic 9). Reported, not gated.
- Bench: `bench_kv.py` 1.1.0 / `bench_strata.py` through the gate on the cyril tier (`X-Client-Source strata-ram-bench`, vault key `strata_key_cyril`), 3 runs, medians; results under `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase2/`.

## 5. Schema

One table in the ai_serve PostgreSQL database (name and the `service_event` table verified in step 0; alerts reuse `service_event` with kind `ram_watch_alert`). Raw parameterized SQL through `psycopg2`, `%s` tuples, manual connection management; migration file in `db/migrations/`, applied with `psql`.

```sql
CREATE TABLE IF NOT EXISTS strata_ram_sample (
    sampled_at        timestamptz NOT NULL DEFAULT now(),
    llama_pid         integer,
    anon_bytes        bigint NOT NULL,
    shmem_bytes       bigint NOT NULL,
    file_bytes        bigint NOT NULL,
    swap_bytes        bigint NOT NULL,
    mem_available_kib bigint NOT NULL,
    swap_free_kib     bigint NOT NULL,
    cache_ram_bytes   bigint,
    cache_disk_bytes  bigint,
    checkpoint_bytes  bigint,
    tier_bytes_gpu0   bigint,
    tier_bytes_gpu6   bigint,
    xc_hit_pct        real,
    xc_h2d_gib        real,
    xc_d2h_gib        real,
    xc_disk_misses    bigint,
    majflt_delta      bigint,
    gpu0_used_mib     integer,
    gpu6_used_mib     integer,
    unit_restarts_10m integer,
    stage             text NOT NULL,
    PRIMARY KEY (sampled_at)
);
CREATE INDEX IF NOT EXISTS strata_ram_sample_stage_idx ON strata_ram_sample (stage, sampled_at DESC);
```

Retention: `DELETE FROM strata_ram_sample WHERE sampled_at < now() - interval '30 days'`, daily from the same script.

## 6. Stages, gates, rollbacks

Preconditions for any restart: phase 1 has posted DONE/RELEASE and nothing restarted Strata for 15 min; the 21:37Z Xid 31 on the `llamacpp-pcache` build is root-caused by phase 1 and that build has soaked >= 12 h without a CUDA error or Xid (critic 3); quiet window (> 2 min with no sprite/cyril request, re-checked right before); Relay STATUS posted before and after; timestamped sha-listed backups under `/home/cereal/ai-serve/backups/strata-tier-<UTC>/`; rollback target tested in `strata_ram.sh` before the window. Never GPU1/2/3/5 (run 0; `lanes.py` lock + `p40.claims` if ever needed) or GPU4 (media-gate).

Reference outputs: captured once in step 0 from today's production xcache build (not stage-vs-stage, critic 11) in a quiet window with background paused (`--bg-max 0` via a temporary gate drop-in, then removed): temperature 0, fixed seed, 256 tokens on 4 prompts (10k, 66k, 150k, a tool-call turn). Every later equality run uses the same paused-background procedure and compares against this reference set, so drift cannot accumulate.

**Step 0: prerequisites (no service touched)**
- Relay: post this plan's summary as STATUS; tell phase 1 that `/mnt/lexar` is exFAT (0700 impossible, disk tier currently off), that its MIN_FREE/MAX_GIB test (`strata-pcache.patch:915-916`, `STRATA_PCACHE_MIN_FREE_GIB=60`) can never pass inside a dedicated image, and propose (its call): a fully pre-allocated (`fallocate`, not sparse) 48 GiB ext4 loop image `/mnt/lexar/omni-campaign/strata-kv.img` mounted 0700 at `/mnt/strata-kv`, with `STRATA_PCACHE_MIN_FREE_GIB=3` and `STRATA_PCACHE_MAX_GIB=44` (critic 5); and the stale-ident purge at startup (critic 12). Honour PAUSE.
- Re-pin source lines in the live tree: small/large paths and victim scan in `ggml/src/ggml-backend.cpp`; loader mmap selection; `llama-mmap.cpp` numa/`MADV_RANDOM`; `llm_graph_input_ple::set_input`; indexer V readers; the unit's `TimeoutStartSec` (must be >= 15 min for the fill). Read `ngram_size` and `heads_per_ngram` from the GGUF KV. Record in `.../phase2/source_pins.md`.
- Verify the ai_serve DB name, `service_event` table, the vault key `capture.py` uses; confirm `strata_gate.py` 1.4.0's `ExecStart` flags and `/slots` shape.
- Re-measure the post-phase-1 baseline (cgroup anon/shmem/swap) and capture the reference outputs.
- Git: in the chimera-omni checkout, branch `claude/strata-tier-20261028` from `claude/strata-ram-20261028`; patches, launcher, gate, watchdog, bench, migration live there; pushed to the GitHub origin after each stage (no weights, no KV files).

**S1: config only (one strata-llama + one strata-gate restart)**
- Drop-in `/etc/systemd/system/strata-llama.service.d/z2-budget.conf` (sorts after `z-ramcap.conf`): `STRATA_NP=4`, `STRATA_CKPT_N=8`, `STRATA_CACHE_RAM=24576` (16384 only if phase 1's disk restores are proven), `STRATA_PCACHE_SPILL_QUEUE_GIB=2`, `GGML_EXPERT_CACHE_STATS=60`, `LLAMA_ARG_LOG_VERBOSITY=4`, `STRATA_RAM_BUDGET_GIB=130`, `OOMScoreAdjust=-500`, and the full `STRATA_EXTRA` string (`--no-mmap --lazy-mode off --kv-unified -b 1024 -ub 1024 -cmoe` plus whatever phase 1 ships, copied verbatim).
- Gate drop-in `/etc/systemd/system/strata-gate.service.d/slots4.conf`: `ExecStart=` (blank) then `ExecStart=<same command> --slots 4 --bg-max 2`.
- Launcher 1.6.0 (`STRATA_CKPT_N`, flag-derived budget, REFUSED marker, flag assertions); gate 1.4.1 (slot-count check); `strata_ram.sh check` and `rollback-s1`.
- Expected freed: ~10-13 GiB (checkpoints, prompt cache if lowered, spill). Speed: none expected; reported.
- Gate: start line shows the values; `/props` n_ctx 262144 and a 250k prompt returns 200; `/slots` shows 4 with background only on 2 and 3; greedy 4-case identical to the reference set (single-request batch composition is unchanged, so identical is expected; any divergence explained before proceeding); preemption `freed_s` unchanged; no Xid/CUDA; swap flat 1 h; verbose journal captured (slot counts, compute buffers, hit %).
- Rollback: `strata_ram.sh rollback-s1` = delete both drop-ins, `daemon-reload`, restart both.

**S2: build + exclusive tier (one restart)**
- Build in `/home/cereal/omni-campaign/export/llamacpp-tier/` via `_container_build.sh` (sm 75 + 86) in tmux; record the build log and `sha256sum bin/llama-server`.
- Smoke inside the window, in order: (i) new binary, S1 flags unchanged (`--no-mmap`, `GGML_EXPERT_TIER=0`): must reproduce the reference set; (ii) new binary with `--mmap-tensors '\.ffn_(up|gate|down)_exps\.weight$'` and `GGML_EXPERT_TIER=1`: the loader line must report 144 matched tensors and the `xtier:` sizing lines per-GPU slots and tier bytes; then the reference set again.
- `z2-budget.conf`: `STRATA_LLAMA_BIN=/home/cereal/omni-campaign/export/llamacpp-tier/bin`, `GGML_EXPERT_TIER=1`, `GGML_EXPERT_HOST_MB=22528`, `STRATA_RAM_BUDGET_GIB=70`, full `STRATA_EXTRA` rewritten with `--mmap-tensors ...` added and `--no-mmap` kept (the regex overrides it for matching tensors only); `LLAMA_ARG_LOG_VERBOSITY` removed.
- Expected freed: ~51 GiB (72.4 -> ~20-21 pinned). Speed (reported): startup +2.5-3 min; prefill +~0.2 s/batch; decode unchanged; d2h 20-40 MB/token; prefill with run 0 on GPU1 measured separately.
- Gate: `smaps` shows no 72 GiB `/dev/zero` mapping and the exps file mapping near-zero RSS after the fill; cgroup `file` not inflated by the fill (fadvise worked); tier bytes sum <= 22528 MiB; greedy identical to the reference set (any difference rolls back); 256k; Sprite preempts in ms; no Xid/CUDA over 2 h; swap flat.
- Rollback: `strata_ram.sh rollback-s2` = `STRATA_LLAMA_BIN` back to the phase-1 `llamacpp-pcache/bin`, `STRATA_EXTRA` back to the S1 string (no `--mmap-tensors`), `GGML_EXPERT_TIER`/`HOST_MB` lines removed, restart. `GGML_EXPERT_TIER=0` alone is NOT a rollback (the binary refuses it with mmap'd experts, by design).

**S3: VRAM reclaim (one restart)**
- `idx-nov.patch` behind `LLAMA_QWEN4EXP_IDX_NO_V=1` (only if no reader was found); `-ub 512` only if S1's measured compute buffers show >= 1 GiB; `STRATA_CACHE_RAM=16384` if deferred at S1; `STRATA_RAM_BUDGET_GIB=65`.
- Expected freed: 1-2.5 GiB host (the tier shrinks by the VRAM gained) + 8 if the cache drop was deferred.
- Gate: greedy identical; QSA still active (compress_ratios in the start log); slot counts per GPU increased by the expected amount.
- Rollback: env line off / `-ub` back / `STRATA_CACHE_RAM` back, restart.

**S4: n-gram table (one restart)**
- `ple-prefetch.patch`; `STRATA_EXTRA` rewritten in full without `--lazy-mode off`; `STRATA_RAM_BUDGET_GIB=52`.
- Expected freed: 26.8 GiB non-reclaimable (table becomes page cache; its 4.8 GiB of swap disappears on restart).
- Speed (reported): ms/token and s/batch vs S3, `majflt` per token, on a quiet machine and with run 0 + media background admitted.
- Gate: greedy identical; Sprite succeeds and preempts; `majflt` rate bounded; no swap growth; MemAvailable steady over 2 h with run 0 running.
- Rollback: re-add `--lazy-mode off` to the full string, restart.

**S5: only if S4 fails a functional gate**: `--cache-ram 8192`, then `-np 2` (gate `--slots 2 --bg-max 0` or 1), then the disk bottom tier under (b) (`GGML_EXPERT_HOST_MB` below the cold set; O_DIRECT 4 KiB-aligned reads from the GGUF into pinned staging, ~500-900 lines; `disk_misses` counter). Reported honestly as ~2.3 s per prompt batch per GiB removed.

**Step 6: watchdog and soak**: `strata_ram_watch.py` + `strata-ram-watch.service`/`.timer`, migration applied; 24 h soak after S4 with run 0 and media background admitted; final numbers fixed; update `strata-q4-switch-2026-10-27.md`, the runbook's Strata section, Relay DONE. Needs Cyril: `vm.swappiness=10` via `/etc/sysctl.d/90-strata.conf` (root).

Restart count: 4 strata-llama (~6-9 min each from S2 on, due to the fill) + 2 strata-gate (S1, and S5 if np changes), each in a quiet window with Sprite falling back meanwhile.

## 7. Bench and functional protocol (every stage)

- Greedy equality: background paused (`--bg-max 0` temporary gate drop-in), temperature 0, fixed seed, 256 tokens, 4 prompts, compared to the step-0 reference set; pass = token-identical; any divergence is explained in the stage report before the window closes or the stage rolls back.
- Capacity: `/props` n_ctx 262144; a 250k prompt returns 200; no `max_tokens` injected.
- Sprite/cyril path: 3 foreground requests during background load -> `preempt_freed` logged, TTFT recorded; follow-up `cache_n >= 0.98 x prompt_n`, or a `source:disk` prefetch then `cache_n >= 0.98 x n_match`.
- Stability: `journalctl -k` + `-u strata-llama` grep `Xid|CUDA error|illegal`; `memory.swap.current` flat 60 min; watchdog rows; unit restart count 0.
- Speed (reported only, never a gate): short write, write at 66k, prompt read at 10k/66k, 10k follow-up TTFT; 3 runs, medians, quiet machine; PCIe h2d/d2h in decode and prefill windows; prefill with run 0 active on GPU1.

## 8. Risks and approvals

- Touches two running production services across 4-6 restarts. Gated on phase 1 DONE/RELEASE + Xid root cause + 12 h clean soak + quiet windows + Relay posts + tested rollbacks. Never GPU1/2/3/5 or GPU4.
- Tier correctness (S2): a bookkeeping error silently uses the wrong expert. Guards: startup fatal checks, debug asserts, TIER=0 smoke on the same binary, reference-set equality with background paused.
- Base-build stability: the pcache build crashed with Xid 31 today; under S2 every crash-restart also re-reads 72 GiB (~3 min outage). S2 does not start until phase 1 has root-caused it.
- Fill time: 72 GiB from the BX500 every restart; `TimeoutStartSec` verified >= 15 min in step 0; `RestartSec` interacts with the ~3 min fill (watchdog restart-storm alert).
- S4 depends on k and the CPU-side ids; k > 16 or GPU-only ids re-plans the stage. One-flag rollback.
- ComfyUI `comfyui_qwen21_gpu4` swings 4-62 GiB anon; the >= 100 GiB MemAvailable target is budgeted against its peak and is outside this lane.
- Phase 1's disk tier is off (exFAT). Until it works, `--cache-ram` stays 24576 to avoid foreground re-reads; the loop-image proposal and its MIN_FREE fix are phase 1's call.
- Every restart and new build invalidates phase 1's disk entries (ident); one round of re-reads per restart is expected.
- GPU0 shares a PCIe switch with run 0's GPU1; prefill under run 0 load is measured and reported.
- Swap 8 GiB ~full, swappiness 60; S2 and S4 remove the two swapped mappings; the sysctl needs Cyril.
- Approvals needed from Cyril: `vm.swappiness`; the ext4 loop image on the Lexar; acknowledgement of the measured speed numbers after S4 (reported; his standing order says they do not gate).

## 9. Critic findings: applied or rejected

| # | Finding | Decision | Where |
|---|---|---|---|
| 1 | `--mmap-tensors _exps$` matches no tensor; S2 would add ~20 GiB | Applied: regex `\.ffn_(up\|gate\|down)_exps\.weight$`; loader logs and enforces 144 matched tensors; tier refuses non-file-backed experts (fatal) | 4.1, S2 |
| 2 | `-np 4` without gate `--slots 4` wraps background onto slots 0/1 | Applied: gate `ExecStart` override with `--slots 4 --bg-max 2`; gate 1.4.1 refuses a slot-count mismatch; `/slots` check in the S1 gate | 4.4, S1 |
| 3 | pcache build crashed with Xid 31 today; S2 cannot attribute crashes | Applied: S2 gated on phase 1's root cause + >= 12 h clean soak of the base build; debug asserts on every index in xtier | 6 preconditions, 4.1 |
| 4 | `xe == nullptr` and `TIER=0` fallbacks read experts through the `MADV_RANDOM` mapping (~20 min/batch) | Applied: all three cases fatal at startup; whole-weight fallback removed from the tier path; rollback = binary + flag swap, TIER=0 is not a rollback | 4.1, S2 rollback |
| 5 | 48 GiB loop image never stores anything under `MIN_FREE_GIB=60`; disk tier is off today; draft baseline assumed it | Applied: proposal now carries `MIN_FREE 3 / MAX_GIB 44`, fully pre-allocated image; baseline re-measured in step 0; `--cache-ram` stays 24576 until restores are proven | step 0, 3, S1 |
| 6 | `STRATA_EXTRA` replaces, does not append; S2/S4 would drop `-cmoe`/`--kv-unified` | Applied: every stage writes the full string; launcher asserts `-cmoe`, `--kv-unified`, `-c 262144`, and `--mmap-tensors` from S2 | 4.5, S1/S2/S4 |
| 7 | Kernel OOM killer would pick Strata first; ComfyUI swings | Applied: `OOMScoreAdjust=-500`; MemAvailable budgeted against ComfyUI's peak | 4.5, 3 |
| 8 | Budget refusal bounds nothing before S4 and fails silently | Applied: flag-derived formula including the 72.4 pool and 26.8 table when present, per-stage `STRATA_RAM_BUDGET_GIB`, pre-flight `strata_ram.sh check` before `daemon-reload`, REFUSED marker + restart-storm alert | 4.5 |
| 9 | GPU0 shares a switch with run 0's GPU1 | Applied as measurement: prefill benchmarked with run 0 active; reported, not gated (speed never gates) | 4.6, 7 |
| 10 | `MADV_DONTNEED` does not drop page cache on `MAP_SHARED` | Applied: loader-side `posix_fadvise(POSIX_FADV_DONTNEED)` per range after the fill, plus `madvise(MADV_DONTNEED)` for PTEs; S2 gate checks cgroup `file` | 4.1, S2 |
| 11 | Equality runs need background paused and a fixed reference | Applied: `--bg-max 0` during equality runs; reference set captured once from today's production build; every stage compares to it | 6, 7 |
| 12 | Each restart invalidates phase 1's disk entries | Applied: stale-ident purge at startup requested from phase 1 (fallback: v2 of the patch on the tier branch); re-reads per restart expected and reported | 4.4, step 0 |
| 13 | Victim scan must exclude the spare; large path reads 512 B past an expert | Applied: spare excluded from the victim scan with an assert; 512 B trailing padding on the tier array | 4.1 |

Nothing was rejected outright. Finding 9 is accepted as a measurement rather than a gate because Cyril's priority removes speed as a gate; finding 8's "alert on refusal" is implemented through the pre-flight check and the watchdog rather than by changing `Restart=always`, because the service must keep self-healing from ordinary crashes.

## 10. Not done here, and why

- No service, file, drop-in, patch or build on the server was changed; phase 1 owns both units and has not posted DONE/RELEASE. Nothing in this document is applied.
- No Relay post was made from this lane; the step-0 STATUS post is the first action of execution.
- The expert-tier code (`xtier.patch`, `exps-mmap.patch`, `idx-nov.patch`, `ple-prefetch.patch`) is not written; this is a design with source pins to re-verify in step 0.
- The 21:37Z Xid 31 is not root-caused here; it belongs to phase 1 and gates S2.
- ComfyUI's 62 GiB peak is not addressed; it is outside this lane and is reported to the media lane.
- `ngram_size`, `heads_per_ngram`, exact per-GPU slot counts and compute-buffer sizes, the multi-stream decode hit rate, and the indexer V reader check are unread; each is a named step-0 or S1 measurement with a stated consequence if it comes out differently.
