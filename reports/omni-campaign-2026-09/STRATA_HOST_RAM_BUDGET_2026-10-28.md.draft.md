# Strata host-RAM budget: phase 2 design and staged plan (2026-10-28, DRAFT)

Status: DESIGN. Nothing in this document has been applied. Phase 1 (Relay CLAIM 204084 / STATUS 204096, plan `STRATA_RAM_2026-10-28.md`, build `llamacpp-pcache`) owns strata-llama and strata-gate until it posts DONE or RELEASE; every stage here builds on its drop-ins, patches and backups and never reverts them. Relay 204097 carries Cyril's 10-28 priority to the phase-1 lane.

Order being executed (Cyril, verbatim): "[Cyril quote withheld]"; "[Cyril quote withheld]"; 10-27: "[Cyril quote withheld]", "[Cyril quote withheld]".

Evidence base: the two 10-28 scouts (`measure_ram` 20:41-20:44Z, `measure_xcache` ~20:40-20:50Z; artifacts under `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase2-scout/` and `/tmp/xc_*`), the live tree `/home/cereal/omni-campaign/export/llamacpp-xcache/llama.cpp` (b96806d96 + `xcache.patch`; local copy `D:/SpriteBuild/strata-xcache-20261027/xcache.patch`), `q4kxl.conf`, `strata_llama.sh` 1.3.0 (1.4.0/1.5.0 on the server), `STRATA_RAM_2026-10-28.md`, memory `strata-q4-switch-2026-10-27.md`, `strata-speed-findings-2026-10-22.md`, `ai-server-serving-layout.md`. Source line numbers are those the scouts cited in the live patched tree; re-pin them in step 0 before editing.

## 1. Problem

Strata's non-reclaimable host footprint is ~147 GiB resident + 5.8 GiB swap, and 98.5 GiB of it is two fixed tensors that no existing knob touches: a 71.7 GiB pinned copy of every expert (`-cmoe`, duplicated ~76 % in VRAM) and the 26.8 GiB per-layer n-gram table held anonymous by `--lazy-mode off`. Phase 1 caps the prompt cache (48 -> 24 GiB) but cannot get below ~123 GiB. The order is a hard bound of about 48 GiB with MemAvailable about 100 GiB while run 0 (~20 GiB) and media background (needs >= 40 GiB) run, speed reported but never gating.

## 2. Budget (decisions with defaults)

Target state after stages S1-S4. "Non-reclaimable" = cgroup anon + shmem + swap of strata-llama; page cache is reclaimable by definition and is not in the budget.

| Component | Today (GiB) | Budget (GiB) | Mechanism / cap | Stage |
|---|---:|---:|---|---|
| Expert host tier (experts NOT in VRAM, pinned, exclusive with VRAM) | 71.7 (all experts pinned) | **20.0** expected, 22 max | option (b): pinned tier = n_expert - n_slots per weight; VRAM slots ~54 GiB today + ~1-2 GiB reclaimed; `GGML_EXPERT_HOST_MB=22528` is a logged warning cap | S2, S3 |
| Per-layer n-gram table (`per_layer_token_embd`, IQ4_NL) | 26.8 anon (4.8 in swap) | **0 non-reclaimable** (page cache, hot rows only; ~27 reclaimable worst) | drop `--lazy-mode off` (tensor is `TENSOR_READ_LAZY`, lazily mmapped) + per-ubatch row prefetch patch | S4 |
| RAM prompt cache (phase 1) | 48 | **16** (phase 1 ships 24; lowered after its soak proves disk restores) | `--cache-ram 16384` | S1 |
| Spill queue (phase 1) | n/a | **2** (phase 1 ships 4) | `--prompt-cache-spill-queue-gib 2` | S1 |
| Slot context checkpoints | <= 10.5 worst (6 x 16 x 112.2 MiB) | **3.5 worst** (4 x 8), ~1-2 typical | `-np 4 --ctx-checkpoints 8` | S1 |
| Compute/staging: pinned output buffer (~0.8), CPU compute, CUDA runtime, heap, disk index (<= 0.5), page tables | ~3.6 + 0.5 | **5.0** | fixed by `-b/-ub`, `-np`; measured, not capped | - |
| mmproj F16 on CPU + its compute buffer | ~1.3 | **1.5** | unchanged (`--no-mmproj-offload`; moving it to VRAM is zero-sum against expert slots) | - |
| **Total non-reclaimable** | **~147 + 5.8 swap** | **~48 worst, ~44 typical** | | |

Progression: today ~153 -> after phase 1 ~123 -> S1 ~110 -> S2 ~59 -> S3 ~57 -> S4 ~47 (worst-case checkpoints) / ~44 typical.

MemAvailable arithmetic (MemTotal 251.7): 251.7 - Strata 48 - ComfyUI (`comfyui_qwen21_gpu4`) 62.4 - run 0 20 - other users/kernel/watermarks ~16 = **~105 GiB**. The >= 100 target holds only while ComfyUI stays at or below its measured 62.4 GiB of anon; that container is outside this lane and is the single biggest variable (section 8). Page cache from the n-gram table (<= 27 GiB) and from the exps mmap (dropped with `DONTNEED` after fill) counts as available.

Floor statement, honest: with S1-S3 only (table resident) the floor is ~57 GiB; the ~48 target needs S4. If S4 is blocked (section 8), the alternatives are an 8-12 GiB prompt cache (-> ~51) or a disk bottom tier for experts (S5, -> ~41 but every prompt batch reads from disk).

Settings decided (defaults; each is a one-line revert):
- `-np 4` (was 6), gate `--bg-max 2` (was 4): Sprite + Cyril pinned slots + 2 background. `-np 2` is held in reserve: it saves only 1.75 GiB worst-case checkpoints + 0.22 GiB VRAM and halves background throughput; taken if the 24 h soak needs it.
- `--ctx-checkpoints 8` (was 16): a checkpoint is 112.2 MiB; 8 still gives per-user-message rollback points for a typical conversation.
- `-b 1024 -ub 1024` kept. `-ub 512` halves VRAM compute buffers (size unmeasured; each GiB freed = 1 GiB less host tier) but doubles the per-token PCIe cost of prefill (each batch streams the whole ~19 GiB cold set). Measured at S3, applied only if it frees >= 1 GiB of VRAM.
- `--cache-ram 16384`, spill queue 2 GiB.
- KV type q8_0, `-c 262144`, `--kv-unified`, no output cap: unchanged (functional constraints).

## 3. Design

### 3.1 Expert tiering: exclusive pinned host tier under the VRAM LRU (option b), no disk on the hot path

Today (xcache.patch): every expert lives in a 72.4 GiB pinned `cudaMallocHost` pool; VRAM holds LRU slot caches per weight (~390-405 of 512 slots per weight on GPU0, ~373-384 on GPU6); decode misses copy host->GPU (small path, patch lines ~2017-2032), prefill assembles cached experts d2d and the cold rest from the host pool in contiguous runs (large path, ~2122-2139), never evicting. VRAM therefore duplicates ~51-54 GiB of the pinned pool.

New invariant: **every expert is in exactly one of VRAM slot or pinned host slot.** Per weight: `n_slots` VRAM slots + 1 spare VRAM slot, and a pinned host tier of `n_expert - n_slots` slots. The loader's full host copy exists only during startup.

Data movement:
- Startup fill (first sizing, `ggml_sched_xcache_size` + `entry_get`): experts 0..n_slots-1 -> VRAM slots (h2d from the loader mapping), the rest -> host tier (memcpy from the mapping); then `madvise(MADV_DONTNEED)` on that weight's range of the mapping. Peak host during fill = tier + the page cache of the mapping (reclaimable), not 72 + 20.
- Decode miss (small path): expert Y needed, in host slot `hY`; victim X = LRU VRAM slot `s` (not used this step). Enqueue on the backend's CUDA stream, in this order: (1) h2d `host[hY] -> vram[spare]`; (2) d2h `vram[s] -> host[hY]`. Then bookkeeping: `slot_of[Y]=spare`, `expert_in[spare]=Y`, `loc[X]={host,hY}`, `spare=s`. Because X stays in slot `s` until a later miss uses `s` as the spare, and both copies sit on one stream, no event is needed: a later h2d into `s` is ordered after the d2h out of it. The d2h rides the other direction of a full-duplex PCIe gen3 x16 link (~20-40 MB/token measured); cost is negligible.
- Prefill (large path): cached experts d2d as today; cold experts copied one by one from their host slots (`tensor_set_async(input_cpy, host + hE*expert_size, e*expert_size, expert_size)`), replacing the contiguous-run copy. ~21k copies per 1024-token batch, ~0.2 s overhead (measure_xcache). Optional: sort cold experts by host slot and coalesce adjacent host runs (the tier starts contiguous in expert order). Free-slot fill unchanged (none free in steady state).
- `xe == nullptr` fallback (VRAM cache allocation failed for a weight): the generic path copies the whole weight from `input->data`, now a page-cache-backed mapping -> correct but slow; logged WARN. Not expected (allocation succeeds today).
- NUMA: the tier for each GPU is allocated after `numa_set_preferred(node_of(gpu))` (GPU0 node 0, GPU6 node 1) with `cudaMallocHost`, restoring the policy after; verified by `numastat -p`. Refinement, not a gate.

Why the loader copy must become an mmap (the "loader patch"): with `-cmoe --no-mmap` the pinned pool is allocated before the scheduler sizes anything (sizing is lazy at the first graph), so without it startup peaks at 72 + 20 GiB and the 72 stays. `--no-host`/`GGML_CUDA_NO_PINNED` only turn the pool into malloc'd anon (swappable, not reclaimable; swap is full) - they save nothing. So: in `llama-model-loader.cpp` (~1081-1107, 1228-1270) add a per-tensor mmap selection, flag `--mmap-tensors <regex>` (default off; Strata passes `_exps$`), so only `*_exps` tensors are file-backed while `token_embd`, the n-gram table (until S4) and everything else keep today's allocation. `--numa distribute` applies `MADV_RANDOM` over the whole file (`llama-mmap.cpp:472, 507-509`); the one-time fill reads each weight's range with an explicit `MADV_WILLNEED`/`MADV_SEQUENTIAL` on that range first, so the fill runs at the BX500's sequential ~475 MB/s (~2.5 min for 72 GiB, on top of today's ~6 min start) instead of 4 KiB faults. `GGML_CUDA_REGISTER_HOST` stays unset.

Expected: host 72.4 -> ~20 GiB pinned (exactly 71.7 x (1 - coverage), coverage ~0.73 measured by PCIe sampling, 0.76 by VRAM accounting); decode h2d bytes unchanged; prefill h2d bytes unchanged (+0.2 s/batch of launch overhead); outputs token-identical (same experts, same GPU kernels, only data placement changes).

Code: `xtier.patch` over `ggml/src/ggml-backend.cpp` (replaces the small-path miss block ~2017-2032, the large-path host copy ~2122-2139, `entry_get`/`size` for the tier allocation, free, stats), ~450-700 lines; `exps-mmap.patch` over `src/llama-model-loader.cpp`, `src/llama-mmap.cpp`, `common/arg.cpp` (~60-100 lines). Both live beside `xcache.patch`, applied in order by `apply_xcache.py`.

New env knobs (read in `ggml_sched_xcache_init`): `GGML_EXPERT_HOST_MB` (warning cap for the summed tier; never refuses), `GGML_EXPERT_TIER=1` (0 = today's behaviour on the same binary: the rollback inside the build), `GGML_EXPERT_CACHE_STATS` as today.

### 3.2 VRAM reclaimed for more slots (each GiB of VRAM = 1 GiB less host tier)

- Indexer V cache (~0.8 GiB across both cards): the indexer KV is allocated with V (`llama-kv-cache.cpp:231-234`, `has_v = !is_mla`) but `qwen4exp.cpp:475-500` only sets `k_idxs`; no reader found. `idx-nov.patch`: construct the indexer cache with `has_v=false` behind env `LLAMA_QWEN4EXP_IDX_NO_V=1`; the greedy-equality test is the guard. measure_ram notes QSA IS active on the Q4 GGUF (compress_ratios = 4 on the 12 attention layers), so this is on the live path; if any reader is found in step 0, the patch is dropped.
- `-np 6 -> 4`: 0.22 GiB (recurrent state per sequence 112.2 MiB).
- `-ub 1024 -> 512`: compute buffers roughly halve; size read from the verbose start line at S1; applied in S3 only if >= 1 GiB.
- Rejected: shrinking context/KV type (changes behaviour), reducing the 768 MiB `keep` headroom (cuBLAS workspaces), moving mmproj off CPU (zero-sum).

Expected: ~1-2.5 GiB more slots (~20-50 slots per card), host tier 20 -> ~18-19.

### 3.3 N-gram table: lazy mmap + per-ubatch row prefetch (S4)

The table is a CPU gather (not math) on rows selected from token ids. It is `TENSOR_READ_LAZY` (`qwen4exp.cpp:188`); with the default `--lazy-mode auto` the loader mmaps it instead of reading it (`llama-model-loader.cpp:1081-1105, 1335`). Strata passes `--lazy-mode off` because the 10-21 incident ("weights faulting in from disk on every token", 26-77 s turns) was the whole 89 GB IQ3_S model faulting through a file mapping, and the table was swept into the same fix. The table's access pattern is different: k row gathers per token (row = 160 x IQ4_NL ~ 90 B, one 4 KiB page each), and 4.8 GiB of it already sits in swap with zero major faults in a 15 s sample, i.e. most of it is cold.

Mechanism: drop `--lazy-mode off` (table becomes a file mapping, hot rows live in page cache, reclaimable) and add `ple-prefetch.patch`: in the CPU-side input setter that fills the table's row ids for a ubatch, compute the distinct row addresses and issue one `madvise(MADV_WILLNEED)` per distinct page (or a single `readahead()` batch) BEFORE the graph runs, so cold rows are fetched asynchronously at queue depth instead of serially by the fault handler (`MADV_RANDOM` from `--numa` disables readahead, which is what we want: no neighbour pages). If the row ids are computed inside the graph rather than in the setter, the hash is replicated in the setter from the model source (~50 lines). Measured disk pattern: random 4 KiB on the BX500 at 3.9k IOPS QD1 / 22.9k QD16, p50 247/662 us. Cost bounds: decode <= k x 0.25 ms per token when every row misses (k <= 8 -> <= 2 ms against 42 ms/token today); prefill 1024 x k reads at QD16 <= 0.4 s per batch at k = 8. k and the id source are the two facts step 0 reads from `qwen4exp.cpp`.

Rejected: keeping the table anon (floor ~57); putting it in VRAM (26.8 GiB of VRAM displaces 26.8 GiB of expert slots -> host tier grows by the same amount: zero-sum); `MemoryLow=`/`MemoryMin=` to protect its page cache (makes it non-reclaimable in effect while MemAvailable still counts it; defeats the purpose); a hand-rolled pinned row cache (the page cache already is one, and the kernel's LRU is good enough for a 4K-row table).

### 3.4 Prompt cache and slots (S1) - config only, on top of phase 1

`--cache-ram 16384`, spill queue 2 GiB, `-np 4 --ctx-checkpoints 8`, gate `--bg-max 2`. The disk tier from phase 1 absorbs what no longer fits in RAM; foreground follow-ups restore through phase 1's prefetch route. Nothing here changes outputs.

### 3.5 Enforcement: caps + watchdog, never OOM-kill

- Hard caps in llama-server: `--cache-ram`, spill queue, `-np x --ctx-checkpoints`, the host tier (a fixed allocation sized once at start), mmproj. After S4 the only unbounded host memory is page cache (reclaimable) and heap growth (a bug, caught by the watchdog).
- systemd: no `MemoryMax=` (OOM-kills strata-llama: forbidden) and no `MemoryHigh=` (throttles the serving process under reclaim). `MEM_FLOOR_GIB=40` start refusal in `strata_llama.sh` stays. Launcher 1.6.0 additionally refuses a start whose configured budget (`cache-ram + spill + np x ckpt x 112 MiB + GGML_EXPERT_HOST_MB + 8`) exceeds `STRATA_RAM_BUDGET_GIB` (default 52): a mis-edited drop-in cannot silently raise the bound.
- Watchdog `strata_ram_watch.py` (`/home/cereal/ai-serve/strata/`, systemd timer every 60 s, raw `psycopg2`, manual connection, DB credential from the environment file capture.py already uses, key name verified in step 0; never on a command line): samples `memory.stat` (anon, shmem, file, swap) of `strata-llama.service`, `/proc/meminfo` MemAvailable/SwapFree, `/metrics` gauges from phase 1, the latest `xtier:` stats line from the journal (hit %, h2d/d2h GiB, tier bytes, disk misses), `majflt` delta of the llama PID, and GPU0/GPU6 used MiB. Writes one row per sample (DDL in section 4). Alerts (journal line + Relay STATUS, rate-limited to one per 30 min per condition) when: Strata non-reclaimable > budget + 4 GiB for 5 consecutive samples; MemAvailable < 60 GiB for 5 samples; swap used grows > 1 GiB over 60 min; `majflt` > 5 per second averaged over 5 min after S4; `_spill_dropped_total` grows. It never signals, stops or restarts anything.

### 3.6 Measurement hooks

- `xcache.patch` sizing line and `ggml_sched_xcache_stats` print at ggml INFO, which llama-server maps to TRACE (filtered). `xtier.patch` promotes those two lines to `GGML_LOG_WARN` and adds: `d2h_bytes`, `tier_bytes` per GPU, `n_slots` per GPU, `disk_misses` (S5 only), `fill_s`. `GGML_EXPERT_CACHE_STATS=60`.
- For the S1 restart (config-only, before the new build exists): `Environment=LLAMA_ARG_LOG_VERBOSITY=4` on the drop-in, removed again at S2; it reveals the current slot counts per card, compute buffer sizes and the hit rate for the baseline. Journal volume is modest (measure_xcache).
- Independent of logs: `nvidia-smi dmon -i 0,6 -s t` PCIe sampling over a decode-only and a prefill-only window; `print_timing` tok/s from the gate bench; `/proc/<pid>/stat` majflt.
- Bench: `bench_kv.py` 1.1.0 / `bench_strata.py` cases, through the gate on the cyril tier with `X-Client-Source strata-ram-bench` (never exported by capture), 3 runs, medians, quiet machine; results into `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase2/`.

## 4. Schema

One narrow table in the ai_serve PostgreSQL database (same DB capture.py uses; the name of the existing `service_event` table is verified in step 0 and reused for alerts with kind `ram_watch_alert`). Raw SQL, parameterized, `psycopg2`, manual connect/commit/close. No ORM, no Pydantic, no SQLite.

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
    stage             text NOT NULL,
    PRIMARY KEY (sampled_at)
);
CREATE INDEX IF NOT EXISTS strata_ram_sample_stage_idx ON strata_ram_sample (stage, sampled_at DESC);
```

Insert shape: `INSERT INTO strata_ram_sample (...) VALUES (%s, %s, ...)` with a tuple; retention: `DELETE FROM strata_ram_sample WHERE sampled_at < now() - interval '30 days'` once a day from the same script.

## 5. Trade-offs

| Choice | Rejected | Why |
|---|---|---|
| Expert tier = exclusive pinned host tier under VRAM LRU (option b) | (a) mmap-backed experts + bounce pool; (c) O_DIRECT reads on miss | (a)/(c) put ~19 GiB of disk reads on every prompt batch (BX500/Lexar ~0.45 GB/s -> ~45 s per 1024 tokens) and a background batch mid-read cannot be preempted; (b) keeps today's byte traffic, adds only the d2h demotion on the idle link direction, and has no disk on the hot path. Cost: ~500-800 lines of scheduler + loader code and a correctness risk guarded by greedy equality. |
| `--mmap-tensors _exps$` loader patch | plain `--load-mode mmap` / dropping `--no-mmap`; `--no-host`; `GGML_CUDA_NO_PINNED` | plain mmap also page-cache-backs the n-gram table and token_embd without prefetch (the 10-21 pattern); the two env/flag knobs only make the pool anon, saving nothing. |
| N-gram table lazily mmapped + WILLNEED row prefetch (S4) | keep anon (floor ~57); table in VRAM; cgroup memory protection; custom pinned row cache | only S4 reaches ~48; VRAM is zero-sum against expert slots; protection defeats reclaim; the page cache is already a row cache. Risk is bounded by k x 0.25 ms/token and rolled back by one flag. |
| `-np 4`, `--bg-max 2` | keep 6 (phase 1's D3); go to 2 now | Cyril allowed 2/4; 4 keeps Sprite + Cyril pinned slots plus two background slots for corpus generation; 2 saves only ~1.75 GiB more and is held for the soak. |
| `--ctx-checkpoints 8` | 16 | halves worst-case checkpoint RAM; 8 rollback points still cover per-user-message restores. |
| `--cache-ram 16` after phase 1's soak | 24 (phase 1) / 8-12 | 16 is about one Sprite-hour of foreground entries; the disk tier covers older ones; 8-12 is the fallback if S4 is blocked. |
| Watchdog + launcher budget refusal | `MemoryMax=` / `MemoryHigh=` on the unit | the order forbids an OOM-kill of strata-llama; MemoryHigh throttles the serving loop under reclaim. The budget is enforced by fixed allocations + a start-time check + an alert. |
| WARN-level stats lines in the patch | `LLAMA_ARG_LOG_VERBOSITY=4` permanently | level 4 also traces every request; used only for the S1 baseline restart. |
| One restart per stage, each stage one lever | one big restart | each stage has its own GiB number, speed cost and one-line rollback; a functional regression is attributable. |
| `-b/-ub 1024` kept unless `-ub 512` frees >= 1 GiB VRAM | `-ub 512` by default | prefill PCIe cost per token doubles for an unmeasured VRAM gain; measured first. |
| Branch `claude/strata-tier-20261028` in chimera-omni, from phase 1's `claude/strata-ram-20261028` | same branch | phase 1 must stay revertible without the tier patches. |

## 6. Plan (ordered; nothing restarts strata-llama/strata-gate before phase 1 posts DONE/RELEASE and 15 quiet minutes pass)

0. Prerequisites and reads (no service touched):
   - Relay: post this draft's summary as STATUS on thread `52468f63-fbb0-429e-a316-1d17607d3912`; tell the phase-1 lane that `/mnt/lexar` (`/dev/sdb2`) is exFAT (`fmask=0022`): its launcher's 0700 check will never pass there, so the disk tier either never turns on or loses its privacy modes. Proposed fix for phase 1 (its call): a 48 GiB ext4 loop image on the Lexar (`/mnt/lexar/omni-campaign/strata-kv.img`, mounted 0700 at `/mnt/strata-kv`), the pattern the amem image already uses. Honour PAUSE.
   - Re-pin source lines in `/home/cereal/omni-campaign/export/llamacpp-xcache/llama.cpp`: small/large paths in `ggml/src/ggml-backend.cpp`; loader mmap selection `src/llama-model-loader.cpp` (~1081-1107, 1228-1270); `src/llama-mmap.cpp:472, 507-509`; the n-gram gather in `src/models/qwen4exp.cpp` (row-id source, k per token); indexer cache V readers (`qwen4exp.cpp:475-500`, `llama-kv-cache.cpp:231-234`); the unit's `TimeoutStartSec`. Record in `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase2/source_pins.md`.
   - Verify the ai_serve DB name, the `service_event` table name (`db/migrations/`), and the vault key name of the credential `capture.py` uses; confirm `strata_gate.py` 1.4.0 reads `--bg-max` from an env var (if not, gate 1.4.1 adds `STRATA_BG_MAX` so the drop-in stays `Environment=`-only).
   - Git: in the chimera-omni checkout, `git fetch`, branch `claude/strata-tier-20261028` from `claude/strata-ram-20261028`; patches, launcher, watchdog and bench live there and are pushed to the GitHub origin after each stage (no weights, no KV files).
1. S1 config-only (one strata-llama + one strata-gate restart). Files: new drop-in `/etc/systemd/system/strata-llama.service.d/z2-budget.conf` (sorts after `z-ramcap.conf`; header states what/why/rollback): `Environment=STRATA_NP=4`, `STRATA_CKPT_N=8` (launcher 1.6.0 maps it to `--ctx-checkpoints`, today hard-coded 16 at `strata_llama.sh:114`), `STRATA_CACHE_RAM=16384`, `STRATA_PCACHE_SPILL_QUEUE_GIB=2`, `GGML_EXPERT_CACHE_STATS=60`, `LLAMA_ARG_LOG_VERBOSITY=4` (this stage only), `STRATA_RAM_BUDGET_GIB=52`; gate drop-in `/etc/systemd/system/strata-gate.service.d/slots4.conf` with `Environment=STRATA_BG_MAX=2`. Launcher `strata_llama.sh` 1.6.0 (`/home/cereal/ai-serve/strata/`, mirror in the branch) adds `STRATA_CKPT_N`, the budget refusal, and logs the computed budget. Backups: every touched file to `/home/cereal/ai-serve/backups/strata-tier-<UTC stamp>/` (sha-listed); phase 1's `strata_ram.sh` gains a `rollback-s1` target, tested against the backup copies before the window. Expected freed: ~10-13 GiB (cache 8, spill 2, checkpoints ~2-7). Speed cost: none expected; reported from the bench. Functional checks: start line shows the four values; 256k prompt accepted; Sprite + cyril + 2 background slots observed in `/slots`; preemption `freed_s` unchanged; greedy 4-case equality vs the phase-1 build; no Xid/CUDA line; swap not growing over 1 h. Read from the verbose journal: slot counts per GPU, compute buffer sizes, baseline hit %. Rollback: delete `z2-budget.conf` and `slots4.conf`, `daemon-reload`, restart both. If phase 1's disk-tier restores are not proven at this point, `STRATA_CACHE_RAM` stays 24576 in S1 and drops to 16384 in S3.
2. S2 build + tier (one restart). In `/home/cereal/omni-campaign/export/llamacpp-tier/` (a copy of `llamacpp-xcache/` with phase 1's `strata-pcache.patch`): `xtier.patch` (3.1, 3.6), `exps-mmap.patch` (3.1), `apply_xcache.py` list extended, build in tmux via `_container_build.sh` (sm 75 + 86), record build log and `sha256sum bin/llama-server`. A CPU-only smoke cannot exercise a CUDA scheduler path, so the smoke runs on GPU0+GPU6 inside the restart window: `GGML_EXPERT_TIER=0` on the new binary must reproduce S1's greedy outputs (proves the build), then `GGML_EXPERT_TIER=1`. `z2-budget.conf` gains `STRATA_LLAMA_BIN=/home/cereal/omni-campaign/export/llamacpp-tier/bin`, `GGML_EXPERT_TIER=1`, `GGML_EXPERT_HOST_MB=22528`, and a `STRATA_EXTRA` that adds `--mmap-tensors _exps$` and keeps `--no-mmap` (the loader patch makes the regex override `--no-mmap` for matching tensors only); `LLAMA_ARG_LOG_VERBOSITY=4` removed. Expected freed: ~51 GiB (72.4 -> ~20-21 pinned). Speed cost (reported): startup +2-3 min; prefill +~0.2 s per 1024-token batch; decode unchanged; measured via bench + PCIe sampling (h2d unchanged, d2h ~20-40 MB/token). Functional checks: `xtier:` WARN lines show per-GPU slots and tier bytes with the sum <= 22528 MiB; greedy outputs token-identical to S1 on 4 cases (10k, 66k, 150k prefix, Sprite-shaped tool-call turn) - any difference is a bookkeeping bug and rolls back; 256k; no Xid/CUDA; `/proc/<pid>/smaps` shows no 72 GiB `/dev/zero` mapping and the exps mapping with RSS near 0 after fill; swap flat; preemption unchanged. Rollback: `STRATA_LLAMA_BIN` back to `llamacpp-pcache/bin`, drop the three new lines, restart (`GGML_EXPERT_TIER=0` is the in-binary fallback).
3. S3 VRAM reclaim (one restart): `idx-nov.patch` behind `LLAMA_QWEN4EXP_IDX_NO_V=1` (only if step 0 found no V reader); decide `-ub 512` from S1's measured compute buffer sizes; `STRATA_CACHE_RAM=16384` if deferred from S1. Expected freed: 1-2.5 GiB host (the tier shrinks by exactly the VRAM gained; read from the sizing line) plus 8 GiB if the cache drop was deferred. Speed cost: `-ub 512` only, reported. Functional: greedy identical; QSA still active (compress_ratios 4) and outputs unchanged. Rollback: env line off / `-ub` back.
4. S4 n-gram table (one restart): `ple-prefetch.patch`; `STRATA_EXTRA` drops `--lazy-mode off`. Expected freed: 26.8 GiB non-reclaimable (page cache grows by the hot set, reclaimable). Speed cost (reported): ms/token decode and s/batch prefill vs S3, `majflt` per token, under (i) a quiet machine and (ii) with run 0 and media background admitted (the pressure case). Functional: greedy identical; Sprite requests succeed and preempt; no swap growth; MemAvailable steady. Rollback: re-add `--lazy-mode off` (the patch is inert without the mapping).
5. S5 only if S4 is blocked by a functional failure: disk bottom tier for experts under (b) (`GGML_EXPERT_HOST_MB` below the cold set; O_DIRECT 4 KiB-aligned reads from the GGUF into the pinned staging on the scheduler thread, ~500-900 lines) OR `--cache-ram 8192` OR `-np 2`. Each is its own stage with the same checks. Reported honestly as "every prompt batch reads ~2.3 s per GiB removed from disk".
6. Watchdog and soak: `strata_ram_watch.py` plus `strata-ram-watch.service` and `strata-ram-watch.timer` (system units, `Environment=` from the ai-serve env file), the `CREATE TABLE` from section 4 applied with `psql` (migration file in `db/migrations/`), 24 h soak after S4 with run 0 training and media background admitted; alert thresholds per 3.5. Then fix the final numbers, update `strata-q4-switch-2026-10-27.md`, the runbook's Strata section, Relay DONE, and (needs Cyril) `vm.swappiness=10` via `/etc/sysctl.d/90-strata.conf`.

Restarts: S1-S4 = 4 strata-llama restarts (~6-9 min each with the mmap fill) + 4 strata-gate restarts, each in a quiet window (> 2 min with no sprite/cyril request, re-checked right before the restart), each Relay-posted, each with a tested rollback. Sprite falls back during each window.

## 7. Bench and functional protocol (every stage)

- Greedy equality: temperature 0, fixed seed, 256 tokens on 4 prompts (Sprite-shaped 10k, 66k prefix + question, 150k, a tool-call turn); pass = token-identical to the previous stage (S2-S4 change data placement only, so identity is expected; S1 changes batch composition for background only, so near-tie divergence there is explained, not failed).
- Capacity: a 256k-token window accepted (`n_ctx` from `/props`; a 250k prompt returns 200); no `max_tokens` injected by the gate.
- Sprite/cyril path: 3 foreground requests during background load -> `preempt_freed` in the gate log, TTFT recorded; follow-up turn `cache_n >= 0.98 x prompt_n` (RAM) or a `source:disk` prefetch then `cache_n >= 0.98 x n_match`.
- Stability: `journalctl -k` + `-u strata-llama` grep `Xid|CUDA error|illegal`; `memory.swap.current` flat over 60 min; watchdog rows.
- Speed (reported only): short write tok/s, write at 66k, prompt read at 10k/66k, TTFT of a 10k follow-up, 3 runs, medians, quiet machine; PCIe h2d/d2h GB/s during decode and prefill windows.

## 8. Risks and gates

- Touches two running production services (strata-llama, strata-gate) 4 times; gated on phase 1 DONE/RELEASE + 15 quiet minutes, Relay posts, quiet windows, timestamped backups, tested rollback. Never GPU1/2/3/5 (run 0) or GPU4 (media).
- Correctness of the tier (S2): a bookkeeping error silently uses the wrong expert. Guard: `GGML_EXPERT_TIER=0` on the same binary reproduces S1, then `=1` must be token-identical on 4 greedy cases before the window closes; any difference = rollback.
- Startup: the mmap fill reads 72 GiB from the BX500 (~2.5 min) on every restart, with `MADV_SEQUENTIAL` so it is not 4 KiB faults; `TimeoutStartSec` is checked in step 0.
- N-gram table (S4) depends on two unread facts: where row ids are computed and k per token. If k is large (> 16) or ids exist only on the GPU inside the graph, S4's cost model is wrong and the stage is re-planned, not forced. Rollback is one flag.
- ComfyUI `comfyui_qwen21_gpu4` holds 62.4 GiB of anon RAM, the largest non-Strata user; the >= 100 GiB MemAvailable target depends on it staying there. Outside this lane; reported to the media lane on Relay. Nothing in this plan touches it.
- Phase 1's disk tier on exFAT (0700 impossible): if phase 1 ships without a working disk tier, `--cache-ram 16` would cause re-reads for evicted foreground conversations (functional: "follow-up turns restore from the cache tiers instead of re-reading"). S1 keeps 24 until the disk-tier restores are proven.
- Swap: 8 GiB, 98.6 % full, `vm.swappiness` 60. S2 and S4 remove the two swapped mappings; the sysctl change needs Cyril's approval (root). Until then "no swap growth" is checked per stage.
- Approvals needed: Cyril for `vm.swappiness`, for the ext4 loop image on the Lexar (phase 1's directory), and to acknowledge the measured speed numbers after S4 (reported; his standing order says they do not gate). The repo home (chimera-omni) is the one phase 1 already uses.
- Not measured yet, fixed in step 0/S1: exact slot counts and compute buffer sizes per GPU, the decode hit rate across 6 (then 4) streams, k and the id source for the table, the indexer V reader check.
