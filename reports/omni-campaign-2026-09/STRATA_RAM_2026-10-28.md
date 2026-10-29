# Strata host-RAM bound: final design and plan (2026-10-28)

## Outcome (2026-10-29 ~00:45Z): PARTIAL

**In one line:** Strata's host RAM is now bounded and the pinned-slot restore bug is fixed (live since 2026-10-28 21:44:09Z). The disk tier is not on, though, so when a Sprite or Cyril state has been evicted from the 24 GiB RAM cache, the follow-up still re-reads its whole prompt. The disk tier and a preemption fix are built, CPU-tested and staged. They wait for the phase-2 lane (Relay 204110 CLAIM) to release strata-llama and strata-gate.

### What changed (live)
- **strata-llama** runs `/home/cereal/omni-campaign/export/llamacpp-pcache/bin` (xcache + cacheslots + pcache-disk patches) via the drop-in `/etc/systemd/system/strata-llama.service.d/z-ramcap.conf` (sha256 19e26408...):
  - `--cache-ram 24576` (was 49152).
  - Background requests are never stored in the prompt cache and never evict a cached entry (the gate sets `cache_store none` on them).
  - An empty pinned slot now loads from the RAM cache. The old build gave `cache_n` 0; the new build restores the full prompt.
  - `UMask=0077` and `MemorySwapMax=0`. The swap cap stops new swap-outs; it does not pull back pages that are already in swap.
  - Unchanged: 256k context (`n_ctx` 262,144 on all 6 slots), `-np 6`, no output cap (`n_predict -1`).
- **strata-gate 1.4.1** with `kv_tier.py` and the drop-in `strata-gate.service.d/kvtier.conf` (sha 99b7280d...). It sets `cache_store`/`cache_tag` on each request, strips any cache fields a client sends, and skips prefetch while llama reports `disk.on false`.
- `capture_admin.py` 1.3.0 (`delete-member` also calls `/prompt-cache/delete`), `strata_llama.sh` 1.5.0, `strata_ram.sh` 1.0.1.
- The vision lane's `vision.conf` drop-ins are kept, not reverted.
- **Installed but not used yet:** `strata-kv-store.service` (`/usr/local/sbin/strata_kv_store.sh`). It builds this chain:
  - a 24 GiB image on the Lexar, `/mnt/lexar/omni-campaign/strata-kv-store/store.img`;
  - a loop device with direct I/O;
  - dm-crypt aes-xts with a fresh random key at every start, held only in the kernel;
  - a fresh ext4 at `/mnt/strata-kv` (nosuid/nodev/noexec), with `pcache/` as cereal 0700.

  Stopping the unit or rebooting erases every state on it. The chain exists because the Lexar is exFAT and cannot hold 0700/0600 files.
- **Code:** `autonomouscereal/chimera-omni`, branch `claude/strata-ram-20261028`, pushed to GitHub. Phase 1 is commits 934afef and fa5848b; the fix pass is 20afbd9 and 1677f0e.

### Before / after (through the gate, cyril tier, medians of 3)
The host was busier during BEFORE (loadavg ~18-24 vs ~11), so most of the speed difference comes from host load, not the change. Speed regressions are report-only, per Cyril (relayed in 204097).

| Speed | Before (old build, 48 GiB cache) | After (new build, 24 GiB cache) |
|---|---|---|
| short write | 18.5 tok/s | 26.2 tok/s |
| write at 83.5k ctx | 11.7 tok/s | 11.5 tok/s |
| prompt read 10.7k / 83.5k / 193k | 188 / 203 / 142 tok/s | 271 / 203 / 151 tok/s |
| warm follow-up TTFT 10.7k / 83.5k / 193k (restored) | 1.29 / 1.31 / 1.92 s | 1.05 / 1.75 / 1.98 s |
| pinned follow-up after another slot launched | 40.2 s TTFT, `cache_n` 0 (2 of 3 re-read) | 1.54 s, `cache_n` 10,694 (3 of 3); at 83.5k: 2.6 s, 83,578 restored |
| cold read 8.8k (verifier) | not measured | 28.6 s (thinking off) / 34.9 s (thinking on) |

| RAM | Before | After |
|---|---|---|
| MemAvailable | 21.5 GiB, swap full | 127 GiB right after restart; 85-94 GiB with media's h3sol running (verifier minimum 85.0) |
| llama-server RssAnon / VmSwap | 77.4 / 6.0 GiB | 30.3 / 0 after restart; 41.9-53.3 / 3.8-5.5 after the bench and the verifier's cache fill |
| RssAnon + VmSwap minus cache bytes | not measured | flat at 29.8 GiB across a full cache fill (29.77 -> 29.80) |
| prompt cache max | 48 GiB, refilled ~70 min after each restart | peak 23.92 GiB vs the 24.00 cap; never went over |
| VRAM GPU0 / GPU6 | 23.8 / 45.0 GiB | 23.4-23.8 / 44.6-45.0 GiB |

Snapshot now (2026-10-29 00:44Z): MemAvailable 23.1 GiB, swap free 2.0 of 8 GiB, llama RssAnon 50.9 GiB, VmSwap 3.6 GiB. MemAvailable is low because media-gate's qwen21 ComfyUI on GPU4 holds ~87 GiB, not because of Strata. strata-gate was "deactivating" at that moment, which is the phase-2 lane's work.

### How a Sprite follow-up behaves now
- **State still in the 24 GiB RAM cache, including a pinned slot that another request took over:** it is restored. TTFT is ~1.0-2.6 s up to 83.5k and ~2 s at 193k. Greedy output matched a fresh run in 4 of 4 runs.
- **State evicted from RAM:** it is dropped, because the disk tier is off (`evict_total` equals `evict_dropped_total`). The follow-up re-reads its whole prompt: ~29-35 s at 8.8k, minutes at 66k and above.
  - The cache is half its old size, but background no longer churns it, so only foreground entries compete for the space.
  - For evicted conversations this does NOT yet meet the Sprite rule that a follow-up restores instead of re-reading.
- **Preemption** works but takes seconds, not ms: 1.8-7.4 s from preempt until the slot is free, and one case over 10 s. That is about the same as before the change (1.4-6.0 s).
  - Cause: `server_response::recv_with_timeout` restarted its 1 s timeout on every token notify, so llama did not notice a closed stream until that request's own first token.
  - Fixed in `strata-preempt.patch`, which is not deployed.
  - Even with the fix, the batch already running finishes first, so expect ~2.5-4 s at `-ub 1024`.

### What is bounded, and by what
- **Prompt cache:** bounded by `--cache-ram 24576`. llama evicts before it allocates and skips any entry larger than the cap. Verified at 23.92 of 24.00 GiB.
- **Background:** cannot enter or evict the cache, because the gate sets `cache_store none`. Verified: `store_none` rose from 189 to 203 with 0 background evictions.
  - This depends on `kvtier.conf`. If that file is removed while the pcache build runs, llama's default store applies and background churns the cache again.
- **Swap:** `MemorySwapMax=0` blocks new swap-outs. The 3.6-5.5 GiB already in swap stays there until a restart or until those pages are touched.
- **Not bounded:** strata-llama has no cgroup `memory.max` or `memory.high`. The bound comes only from `--cache-ram` plus checkpoint counts. With swap capped, llama-server is the OOM killer's natural target under host pressure.
- **Expected total:** ~131 GiB typical, ~151 GiB worst case:
  - experts 71.7
  - n-gram ~27
  - cache <= 24
  - checkpoints
  - mmproj ~1

  Measured anon outside the cache: ~30 GiB.

### Rolled back / not done
- **Rolled back:** at 00:31:58Z, after phase 2's claim, the fix-pass drop-ins were staged and daemon-reload ran. No service restarted. At 00:57Z the exact phase-1 files were put back (sha-checked), again with no restart. Backup: `/home/cereal/ai-serve/backups/strata-ram-20261029T0031Z/`.
- **Not done:**
  - enabling the disk tier;
  - the eviction -> disk-restore test;
  - the RAM-vs-disk greedy-equality test on the production model;
  - deploying the preempt patch;
  - testing the `strata_ram.sh` 1.1.0 rollback;
  - `vm.swappiness=10`, which needs Cyril;
  - the 24 h soak;
  - experts-to-mmap, which is a contingency only.

### Open items
1. Wait for phase 2 (204110) to post DONE or RELEASE, plus 15 quiet minutes. Then rebuild `strata-preempt.patch` on top of phase 2's `llamacpp-tier` tree (`git apply --check` already passes).
2. Apply the staged drop-ins from `/home/cereal/ai-serve/work/strata-ram-fix-20261029/stage/` with one strata-llama restart and one strata-gate restart in a quiet window.
   - `z-ramcap.conf`: Wants/After strata-kv-store, `STRATA_PCACHE_DIR=/mnt/strata-kv/pcache`, max 22 GiB, min free 0.5 GiB.
   - `kvtier.conf`: MIN_TOKENS 1024.
3. Run three checks, in order:
   - `vfy_fix.py disk`: evict -> disk restore with greedy equality, size and file bounds, 0700/0600 permissions, per-tag delete;
   - `vfy_fix.py preempt`: 6 probes under real background load;
   - the swap check: VmSwap 0 after the restart.
4. Re-run the CPU preempt smoke with the corrected bound (commit 1677f0e). The last run took 2.904 s against the old flat 2.0 s bound.
5. Test the `strata_ram.sh` 1.1.0 rollback. It has only passed a syntax check.
6. For Cyril: `vm.swappiness=10`, and whether to add a cgroup `memory.high` on strata-llama.
7. Other lanes:
   - Capture export is probably broken: `gen_worker_text.py` no longer defines `MODEL_ID`, which `capture_export.py:539` reads (Relay 204101).
   - Xid 31 on GPU0 at 21:37:19Z on the rolled-back old build. The current build shares its code base. It is the only Xid since 11:00Z.
8. Run the 24 h soak with hourly `service_event` `kv_soak` rows, then decide between a 24 and a 32 GiB cache.

### Rollback
- **Full rollback to the pre-change stack:** post on Relay first, then in a quiet window run `/home/cereal/ai-serve/strata/strata_ram.sh rollback`. It works from `/home/cereal/ai-serve/backups/strata-ram-20261028T2120Z/` (MANIFEST):
  - sha-checks every file;
  - removes `z-ramcap.conf` and `kvtier.conf`;
  - restores gate 1.3.0 (`q4kxl.conf` still sets 49152 and `llamacpp-xcache/bin`);
  - runs daemon-reload and restarts both services;
  - deletes the cache files again once the old build is running.

  Tested 21:31-21:37Z: it printed "every file matches MANIFEST" and the stack was healthy. Re-applied 21:39-21:44Z.
- **Swap cap only:** remove the `MemorySwapMax=0` line from `z-ramcap.conf`, then run `systemctl daemon-reload` and `systemctl set-property --runtime strata-llama MemorySwapMax=infinity`. No restart is needed; this was tested live.
- **Do not** remove `kvtier.conf` on its own while the pcache build runs, or background will enter the cache.
- **kv store:** `systemctl stop strata-kv-store` erases every disk state. `strata_ram.sh` 1.1.0 covers the store once it is tested.

**Evidence:**
- `/home/cereal/ai-serve/evidence/strata-ram-20261028/` (phase 0 and the bench)
- `/home/cereal/ai-serve/work/strata-ram-20261028/vfy/run1.jsonl` and `run2.jsonl`
- `/home/cereal/ai-serve/work/strata-ram-fix-20261029/smoke_preempt.log`
- CPU smoke 14/14 for `llamacpp-pcache`
- Runbook sections 12 and 12.1

**Relay:** 204084 (claim); f917bafc, 006d14fa, f0f1e668, 95f58a58 (phase-1 posts, ending in the RELEASE); 204101; 204111 (fix CLAIM); 204114 (YIELD).

---


Status at the time the plan was written: PLAN, nothing applied. The Outcome section above says what was applied later. The vision lane (Relay 204067 CLAIM / 204072 STATUS) owns strata-llama and strata-gate until it posts DONE or RELEASE; this plan builds on its drop-ins and backups and never reverts them.

Supersedes `STRATA_RAM_2026-10-28.md.draft.md` (same directory). Changes from the draft are driven by the critic review; section 11 lists every critic finding with applied / rejected and the reason.

## 1. Order and constraints (verbatim)

Cyril, 2026-10-28 (relayed by the Sprite lead, Relay event 204074): "[Cyril quote withheld]"

Cyril, 2026-10-27 (direct): "[Cyril quote withheld]". Same day: a disk-backed prompt cache is fine ("[Cyril quote withheld]").

Sprite constraints (Sprite lead, 204074, plus Cyril's 10-22 binding order): Strata keeps ~256k context and NO output-token cap; Sprite's slot must stay warm between turns - a disk-backed prompt cache (llama.cpp slot save/restore, or a smaller --cache-ram with spill to the Lexar) is fine as long as a follow-up turn RESTORES instead of re-reading. Sprite/cyril foreground preempts background in ms; that must keep working. Decode speed and prompt-read speed must not regress beyond noise for Sprite turns (measure under a quiet machine; single runs are noise).

Other RAM users: media-gate (GPU4, :28400) refuses all background work while MemAvailable < 40 GiB; the run-0 trainer on the four P40s needs ~20 GiB host RAM and launches within a day or so; strata-gate's own floor is `--mem-floor-gib 12`.

Rules that bind every step: SSH only through the server-manager v2 client; long work in tmux or `systemd-run`; read and post Relay thread 52468f63-fbb0-429e-a316-1d17607d3912; never touch GPU1/2/3/5 (run 0) or GPU4 (media); GPU0/GPU6 only as Strata itself; before any strata-llama / strata-gate change: Relay post, honour PAUSE, quiet window (> 2 min with no sprite or cyril request through the gate, re-checked right before each restart), timestamped backup of every file touched, tested rollback. Keys from the credman vault (`strata_key_cyril`, `strata_key_sprite`; names only). No hardcoded secrets. Raw parameterized SQL only. Code goes to its owning git repo on a `claude/*` branch and is pushed to its GitHub origin; no weights or KV files in git. KV / prompt-cache files hold user conversation content: 0700 directory, bounded size, bounded age, deletable per user.

## 2. Evidence base

Scouts `scout_ram`, `scout_llama`, `scout_usage` (10-28 ~06:40), the critic's read-only source and host checks (10-28 ~07:00-07:15), my own read-only check (10-28), the live drop-ins under `/etc/systemd/system/strata-llama.service.d/` and `strata-gate.service.d/`, `strata_gate.py` 1.3.0, `strata_llama.sh` 1.4.0, `capture.py` / `capture_admin.py`, and the source tree `/home/cereal/omni-campaign/export/llamacpp-xcache/llama.cpp/` (HEAD `b96806d96`) plus `xcache.patch`. Source line numbers below are in `tools/server/server-context.cpp` of that tree unless another file is named.

Related memory: `C:/Users/cereal/.claude/projects/C--Users-cereal/memory/strata-q4-switch-2026-10-27.md`, `strata-speed-findings-2026-10-22.md`, `ai-server-serving-layout.md`. Runbook: `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/AI_SERVER_RUNBOOK_2026-10-21.md`.

## 3. Problem

Strata's host footprint is unbounded in practice. The 48 GiB RAM prompt cache refills within ~70 min of every restart (sar: MemAvailable 130 -> 75 GiB after the 05:30 restart; 28 GiB at 01:30 and 03:30 before it), the 8 GiB swap is 98.6 % full with 4.3-5.0 GiB of llama-server pages in it, and ~59 background prompts/hour (212/h peak) that never reuse a cached prompt (hit ratio 0-3 %) churn the cache and evict Sprite's and Cyril's states, each background entry costing ~230-480 MiB of recurrent state plus checkpoints. The media gate stops all background work under 40 GiB MemAvailable and run 0 needs ~20 GiB, so Strata must be bounded at a level that leaves >= 40 GiB with margin, and a foreground follow-up after an eviction must restore from disk instead of re-reading.

Two defects found in the source make the current cache worse than its size suggests, and both are fixed here:

- A pinned slot that is empty never loads from the RAM cache (`:1588-1596`): when the gate sets `id_slot` and the idle-slot loop (`:2409-2423`) has already saved and cleared that slot, `f_keep = 0/0` is NaN, `NaN < 0.5f` is false, and `prompt_load` (`:1638`) never runs. This matches scout_usage's 58/94 Sprite follow-ups that missed with `cache_n = 0` and no background eviction in between. The 10-27 "cache reuse works" bench was served straight from the slot (no other launch between slot-1 tasks 805 and 2408), so the RAM load path has never been proven for pinned slots.
- The idle-slot save keys on the slot, never on the task; `cache_prompt` is read only at `:3190` (prefix reuse). So there is no runtime flag that keeps background prompts out of the cache; it needs a server patch.

## 4. RAM accounting (10-28)

Host: MemTotal 251.7 GiB; MemAvailable 74.6-74.7 GiB at 06:40-07:00 (falling: sar 85.4 -> 76.1 GiB 06:30-07:00 while Strata's anon+swap stayed flat at ~79 GiB, so ~9 GiB grew elsewhere and is unattributed); Shmem 77.3 GiB; Cached 144.8 GiB; AnonPages 85.2 GiB; swap 8 GiB, 0.1-0.15 GiB free; `vm.swappiness` 60; `vm.dirty_bytes` 536870912, `vm.dirty_background_bytes` 134217728 (global, shared with run-0 and intake writes to the Lexar). Committed_AS 292.8 GiB.

strata-llama (PID 1985658 since 05:30:24 MDT, `-np 6 -c 262144 --kv-unified -ctk q8_0 -ctv q8_0 --ctx-checkpoints 16 --cache-ram 49152 -cmoe --no-mmap --mmproj ... --no-mmproj-offload`): unit MemoryCurrent 187-189 GiB, of which ~40 GiB is reclaimable shard page cache from the `--no-mmap` load; RSS 131.6 GiB = RssAnon 74.9 + RssShmem 56.6 + RssFile 0.17; VmSwap 4.3-5.0 GiB; Pss_Shmem 59.3 GB. Eviction lines ("making room for prompt cache entry, removing oldest entry"): 858 since 10-27 17:00, avg 289 MiB, p50 237, p90 377, p99 685, max 1,710 MiB; 17 in the 15 min before my check; ~477 MiB entries at 07:08-07:09.

Strata's host RAM by part:

| Part | GiB | Nature |
|---|---|---|
| Pinned experts (`cudaMallocHost`, `/dev/zero` shared mapping, `-cmoe`) | 71.7 (55.8 visible in one mapping; rest in RssShmem/swap) | Model. Not swappable, not reclaimable. Stays. |
| Per-layer n-gram tables + token embeddings (`--lazy-mode off`, read per token on CPU) | ~27 anon | Model. Must stay resident. |
| RAM prompt cache (`--cache-ram 49152`) | <= 48 anon | The knob. RssAnon 74.9 = 27 + 48 to within a GiB: at cap, not a leak. |
| Per-slot context checkpoints (outside `--cache-ram`), compute buffers | ~3-5 typical; 22 worst (6 slots x 16 x ~0.23) | Bounded by `-np` and `--ctx-checkpoints`. |
| mmproj F16 on CPU + its compute buffer | ~1 | Vision lane; stays. |
| Shard page cache from load | ~40 | Reclaimable; inside MemAvailable already. Judge the bound by MemAvailable, not MemoryCurrent. |

Other consumers: agent-memory-postgres cgroup 15.8 GiB (~4 GiB RSS, rest page cache); user.slice 14.4 GiB (9 GiB cache, 2.8 anon, includes the run-0 staging build); ComfyUI GPU4 3.0; speech workers 1.6 each; everything else ~3 GiB together. None of these is the lever.

Disks: `/` Crucial CT2000BX500 SATA (DRAM-less), 113 GiB free at 94 % (at its run-0 floor; no cache tier goes here). `/mnt/lexar` (`sdb2`, Lexar SL600 3.8 TB USB 3.2 SSD; `ROTA=1` is the USB bridge's report) 208-216 GiB free at 95 %, shrinking (run 0, intake, audio, and now Strata's mmproj load from `/mnt/lexar/omni-campaign/models/strata-mmproj/`). Lexar throughput is NOT measured; it is step 1.

## 5. Decisions

The cache is the only large knob. Experts and n-gram tables are the model and stay. The design: (D1) the RAM cache becomes foreground-only, correct for pinned slots, and capped at 24 GiB; (D2) a disk tier under it is implemented inside llama-server as spill-on-evict of the RAM entries themselves (which carry the context checkpoints), written from a writer thread and promoted back into RAM by a gate-triggered prefetch that runs in llama's HTTP thread, so the server loop never does disk I/O and never restores over a live slot; (D3) slots unchanged; (D4) experts-to-mmap not done; (D5) swappiness, umask, metrics.

### D1. RAM prompt cache: foreground-only, pinned-slot load fix, 24 GiB cap. Patch `cacheslots.patch`; one strata-llama restart, shipped together with D2.

(a) Per-task cache tag instead of a slot-id split. The gate sets a new request field `cache_store` on every request: `"none"` for the background tier, `"ram"` for foreground members who are opted out of capture or send `X-Sprite-Capture: off`, `"disk"` otherwise (default when the field is absent: `"disk"` if a disk dir is configured, else `"ram"`). The slot stores the launching task's value. Both save sites honour it: the launch-time save (`:1636`) and the idle-slot loop (`:2414`) call `prompt_save` only when the slot's `cache_store != none`; `prompt_load` stays allowed on every slot, so an unpinned foreground request (a third concurrent foreground request, gate 1.2.2) still loads from the global cache. Background slots are still `prompt_clear`ed under unified KV as today.

(b) Empty-pinned-slot fix: `update_cache` becomes true when `ret->prompt.tokens.empty()` (add `|| ret->prompt.tokens.empty()` to the condition at `:1588-1596`), so a pinned slot that was round-tripped into the cache by a background launch loads back. Verified on the live service BEFORE the restart (step 1, probe ii) so the before/after is measured, not inferred.

(c) `STRATA_CACHE_RAM=24576` (24 GiB; was 49152). Sizing from scout_usage with foreground-only entries at ~13 KB/token + 0.3-0.5 GiB state: Sprite ~0.4 GiB each (max 39/h, 48/day captured; up to ~4x more uncaptured), Cyril p50 0.7, p90 1.6, max 2.6 GiB (max 10/h, 12/day). A 1-hour horizon is ~16-24 GiB; the disk tier covers everything older. Fallback if D2 slips: 32 GiB (frees 16 instead of 24).

(d) Metrics in the patch (`/metrics`): `llamacpp:prompt_cache_ram_bytes`, `_ram_entries`, `_disk_bytes`, `_disk_files`, `_checkpoint_bytes` (sum over slots), counters `_spill_total`, `_spill_dropped_total`, `_prefetch_ram_total`, `_prefetch_disk_total`, `_prefetch_none_total`, `_delete_total`, `_store_none_total`, `_image_uncacheable_total`. No gate-side mirror of the cache exists; the gate reads these.

Rejected: `--prompt-cache-slots N` by slot id (draft) - unpinned foreground requests would lose `prompt_load` (critic 5). Rejected: `--no-cache-idle-slots` (slots keep their KV in the single 262,144-token pool; Cyril at 178k + Sprite + 4 background slots overflows it; the overflow behaviour in this build is unverified). Rejected: a second llama-server for background (+100 GiB host RAM). Rejected: the `cache_prompt:false` probe (the source answers it; critic 7).

### D2. Disk tier: llama-side spill-on-evict with checkpoints, writer thread, HTTP-thread prefetch. Patch `pcache-disk.patch`.

Why not `/slots` save/restore (the draft): a restored slot has no checkpoints (`:2604`), and on this hybrid model the recurrent layers keep only their last position, so any token difference inside the saved reply (the chat template trims content, re-renders tool calls with `'\n\n<tool_call>\n<function='` and `tojson`, and BPE re-tokenizes; `render_content(...)|trim`) gives `pos_min >= pos_min_thold` -> no checkpoint -> `n_past = 0` -> full re-read (`:3321-3355`). The RAM cache survives this today only because its entries carry the two end-of-prompt checkpoints (`:3538`, offsets `4+n_ubatch` and `4`) plus one per user-message start (`:3522-3527`). Save/restore also runs synchronously in `process_single_task` (`:2518-2625`), stalling every slot and the gate's preemption probe for the write time, which on this host is throttled past 0.5 GiB by the global dirty budget; and restore loads over whatever the slot holds, which with shared slots 0/1 clobbers an unsaved conversation (critic 1, 3, 4). So the disk tier is the RAM cache's own entries, serialized intact.

Mechanics (all in `tools/server/server-task.{h,cpp}` `server_prompt_cache`, `server-context.cpp`, `server-http.cpp`, `common/arg.cpp`, `common/common.h`):

- Spill. In `update()`, when `states.pop_front()` evicts an entry (oldest-first) and the entry's `cache_store == disk`, the entry (tokens, state blob, checkpoints with `pos_min/pos_max`, tag, build ident) is moved - O(1), no copy - onto a bounded spill queue (`--prompt-cache-spill-queue-gib 4`; when full the entry is dropped and `_spill_dropped_total` counts). Entries dropped because a new entry fully contains them (supersession) are NOT spilled. A single writer thread pops the queue and writes `<dir>/<tag>/<sha256(tokens)[:32]>.spc` with `O_CREAT|O_EXCL`, mode 0600, as 64 MiB chunks each followed by `sync_file_range(SYNC_FILE_RANGE_WRITE|WAIT_AFTER)` so dirty pages stay under the 128 MiB background threshold, then `fdatasync`, then `posix_fadvise(DONTNEED)` on the whole file so the spilled state does not sit in page cache. File header `SPC1`: build string (llama build number + model `general.name` + file sha from `/props`), `n_ctx`, `type_k/type_v`, tag, `n_tokens`, tokens, state size, checkpoint count and each checkpoint's `pos_min/pos_max/size`, then the blobs, then a CRC32 of the body. The RAM is released the moment the entry leaves `states`; the queue bound is the only extra RAM (<= 4 GiB).
- Disk index. In-memory, owned by `server_prompt_cache`: tag, path, tokens (kept in RAM for LCP matching; bounded by `--prompt-cache-disk-max-files 400` x <= 262,144 x 4 B = 420 MB worst, ~50 MB typical), bytes, created, last_used. Rebuilt at startup by reading headers only; files whose build string, `n_ctx` or KV types differ from the running model are deleted (they are a cache, never a source of truth). A spill that fully contains an indexed entry's tokens deletes the older file.
- Bounds and janitor (writer thread, after every spill and every 10 min): `--prompt-cache-disk-max-gib 40`, `--prompt-cache-disk-max-files 400`, `--prompt-cache-disk-max-age-h 168`, `--prompt-cache-disk-min-free-gib 60` (`statvfs` before every write; below it the entry is dropped and counted, never blocks). Prune order: age, then LRU by `last_used`, then size. ENOSPC / EIO: count, log once per minute, keep serving.
- Promote (restore). New route `POST /prompt-cache/prefetch` (internal key; the gate does NOT expose it, any `/prompt-cache/*` route, or `POST /slots` to clients; `NATIVE_ROUTES` keeps `GET /slots` only). Body: the same OpenAI chat body the gate is about to forward. The HTTP thread applies the chat template and tokenizes exactly as the completion path does (reuse `oaicompat_chat_params_parse` + `tokenize_input_prompts`), then under the cache mutex: LCP over RAM entries; if the best RAM match has `f_keep >= 0.25`, return `{"source":"ram","n_match":N}` and do nothing. Else LCP over the disk index; if the best disk match has `f_keep >= 0.25` and a longer match than RAM: read the file in the HTTP thread (page cache `DONTNEED` after), verify CRC, build the entry, insert it as the newest RAM entry (evicting oldest as usual, which may spill), touch `last_used`, return `{"source":"disk","n_match":N,"bytes":B,"t_read_ms":T}`. Else `{"source":"none"}`. The gate then forwards the completion; the loop's existing `load()` finds the entry in RAM, with its checkpoints, so divergence inside the reply rolls back to the nearest checkpoint exactly as the RAM tier does today. The loop never reads disk. Only the one request that needed the restore waits for the file read.
- Gate side (`strata_gate.py` 1.4.0, new `strata/kv_tier.py`, no PG index): for foreground requests with an estimated prompt >= `STRATA_KV_MIN_TOKENS` (2048), call prefetch when the conversation is not known to be warm: first sight of this conversation key (sha256 chain over canonical `messages[0..i]`: role, content, tool_calls, tool_call_id) since the gate started, or last turn > `STRATA_KV_PREFETCH_AFTER_S` (600) ago, or a llama restart observed since (`/props` build start or `llamacpp:prompt_tokens_total` going backwards). A wrong skip costs exactly today's behaviour (re-read); a wrong call costs one tokenize in the HTTP thread (~10-30 ms at 10k, ~0.3-0.5 s at 150k). After each foreground turn the gate compares `timings.cache_n` with `prompt_n` and with the prefetch's `n_match`; `kv_prefetch_lost` counts a disk promote whose completion then read with `cache_n < 0.5 x n_match` (race with a concurrent eviction; expected rare with a foreground-only cache).
- Delete per user. `POST /prompt-cache/delete {"tag": "<tag>"}` (internal key) removes RAM entries, queued spills, index entries and files for that tag, so a deleted member leaves nothing behind (critic 8). Tag = `sha256(member_key)[:16]` where member_key is capture's (`cyril`, `sprite:<24hex>`; `member_and_source`, `capture.py:650`); the gate sets `cache_tag` on every foreground request. `capture_admin.py` 1.2.0 `delete-member` calls this endpoint (internal key from `/home/cereal/.config/ai-serve/strata.env`, read by the script, never printed) and records a `service_event` row (`kv_delete_member`, detail: tag, files, bytes - never prompt text).
- Admin flush. `POST /prompt-cache/flush` (internal key) spills every RAM entry with `cache_store == disk` to the queue (bounded by the queue; the call returns the count). Used by the bench to force "evicted" without flooding, and nothing else in phase 1.
- Images. `serialize()` refuses prompts with images (`:2537-2542`) and checkpoints are not taken after media chunks (`:3594-3595`); such slots are never cached today and stay that way; counted (`_image_uncacheable_total`). media_judge is background anyway.
- Privacy. Directory `/mnt/lexar/omni-campaign/strata-kv` created by the launcher with mode 0700, owned by `cereal`; the launcher refuses to pass `--prompt-cache-disk` if the dir is not 0700/cereal; `UMask=0077` on `strata-llama.service` (today 0022, so llama-created files would be 0644 - critic 8); files 0600; nothing leaves the host; bounded size and age; per-member delete covers RAM and disk; opt-out = RAM only. Token ids in clear in the headers: same class as the capture exports already on the Lexar. Needs Cyril's acknowledgement (section 10).
- strata-think. Entries carry checkpoints, so think conversations get the disk tier like any other; when the client echoes `reasoning_content` the template's `preserve_thinking` (default true) gives an exact match, otherwise the checkpoint at the last user-message start limits the re-read to the final turn, as in RAM today (critic 10 applied; the draft's RAM-only rule for think is dropped).

Expected numbers (to be measured in step 6): entry sizes 10k ~0.4 GiB, 66k ~1.1 GiB, 150k ~2.2 GiB. Prefetch from the Lexar: at 300 MB/s ~1.3 s (Sprite 10k), ~4 s (66k), ~7.5 s (150k), added only to that one request's TTFT; re-read today is ~43 s / ~290 s / ~660 s at 227-258 tok/s. Spill bytes: only true evictions, typically < 24 GiB/day (vs ~160 GB/day for save-every-turn; critic 13). Disk reads in the loop: zero. Slot stalls from the tier: zero by design (the writer thread and HTTP thread do the I/O); the mutex is measured.

Rejected: gate-side `/slots` save/restore with a PG index and a RAM mirror (draft D2; critic 1, 3, 4, 9); tmpfs staging (nothing is written from the loop any more); a synchronous disk lookup inside `load()` (would stall all slots for a multi-second USB read; the prefetch route gives the same restore without the stall); keying restores on `X-Sprite-Delegation` / `X-Sprite-Run` (not every turn has them); spill-on-shutdown (phase 2, section 12).

### D3. Slots: keep `-np 6`, `--bg-max 4`. No change.

Under `--kv-unified` the slot count adds almost no KV memory (ctx256k.conf: +490..728 MiB per card for the whole pool). Slots multiply context checkpoints (16 x ~0.23-0.5 GiB worst per slot), but background prompts (p50 272 tokens) create 0-1; the measured maximum entry of 1.7 GiB including checkpoints says the real total is a few GiB. `_checkpoint_bytes` in `/metrics` measures it in the soak; if it exceeds 12 GiB sustained, a per-task checkpoint cap for background is a phase-2 item. Fewer slots would cut Sprite's concurrency and change the preemption path the Sprite lead measured; not worth single-digit GiB.

### D4. Experts to mmap / page cache: NOT done. Contingency only.

Not needed once D1 lands (section 6). If ever needed: `--no-host` or `GGML_CUDA_NO_PINNED=1` plus `--load-mode mmap` makes the experts file-backed (`src/llama-model.cpp:1026-1056`, `ggml/src/ggml-cuda/ggml-cuda.cu:1279-1293, 5024-5037`); xcache still recognizes host-resident experts (`ggml_backend_buffer_is_host`). Costs to measure first: pageable `cudaMemcpy` on expert misses and page-cache misses hitting the BX500 (~500 MB/s) during prompt batches; decode would mostly survive on the 30+50 GB VRAM LRU caches, prompt read is the exposed path. Only after the soak misses its target, with a quiet-machine A/B (3 runs each).

### D5. Other accounting.

- `vm.swappiness=10` via `/etc/sysctl.d/90-strata.conf` (root; needs approval). Keeps Strata's anon (cache entries, checkpoints) in RAM ahead of stale page cache; today 4.3-5.0 GiB of llama-server anon is swapped out and a swapped entry stalls a load.
- `UMask=0077` in the new strata-llama drop-in (D2 privacy).
- Attribute the unattributed ~9 GiB/30 min drift (step 1, probe iv) before quoting the final target: per-cgroup `memory.stat` anon deltas over 30 min for system.slice units, docker scopes and user.slice; candidates are the run-0 staging build and intake workers. Not Strata (its anon+swap was flat).
- `MemoryHigh=` on the unit: rejected for now (would reclaim the 40 GiB shard page cache early, but throttles the process if anon ever crosses it; measure first).
- agent-memory-postgres and user.slice page cache: reclaimable, no action.
- `/gate/status` gains a `kv` block (the `/metrics` gauges above, the gate's own `kv_prefetch_{ram,disk,none,lost}` counters, `last_foreground_age_s`) and `mem_available_gib`, so the soak is one GET.

### Persistence

No new PostgreSQL table. The disk index is owned by llama-server and rebuilt from file headers at startup; a PG copy would be a second source of truth that drifts (the draft's `strata_kv_state` DDL is dropped). Audit rows go to the existing ai_serve `service_event` table (verify the table name in `db/migrations/` before use) with kinds `kv_delete_member` and `kv_soak` (hourly snapshot of the `kv` block, written by the soak script with raw parameterized `psycopg2`, detail jsonb, never prompt text).

## 6. Target and expected accounting

Bound on Strata's host RAM after D1+D2: experts 71.7 + n-gram/embeddings ~27 + RAM cache <= 24 + spill queue <= 4 + checkpoints and compute ~3-5 (22 worst) + mmproj ~1 + disk index <= 0.5 = ~131 GiB typical, ~151 GiB worst. Every part is a fixed tensor or a capped structure; it cannot grow past that. Page cache from spilled files is released with `DONTNEED`.

MemAvailable: 74.7 GiB now with the 48 GiB cache full; +24 after D1 = ~98; run 0 (-20) = ~78; TTS caption renders / media data generation on GPU4 (-5..-8) = ~70; minus whatever the 9 GiB drift turns out to be if it is a steady consumer. Target: >= 60 GiB steady with Strata serving (cache full), run 0 training and media background running; hard floor 40 GiB (media-gate, unchanged); soak alarm at < 50 GiB. The 24 h soak (step 8) fixes the final number and decides 24 vs 32 GiB.

## 7. Implementation steps (in order; nothing restarts strata-llama / strata-gate before step 5)

0. Relay. Post this plan as STATUS on thread 52468f63 (`python C:/Users/cereal/.agents/skills/coordinate-peer-agents/scripts/session.py --agent claude say --thread-id 52468f63-fbb0-429e-a316-1d17607d3912 --type STATUS --body-file <file>`). Read the vision lane's state with `--event-id 204067`, `204072` and newer (`show --history` returned `event_or_message_id_required`). Wait for its DONE or RELEASE and 15 quiet minutes on strata-llama (`journalctl -u strata-llama --since -15min` shows no start). Honour any PAUSE. Ask the Sprite lead for the first differing message in a few of the 58/94 Sprite misses (diagnostic only; probe ii below is the likely cause). Ask the run-0 and intake lanes for their Lexar budgets.

1. Phase-0 probes (read-only or single requests; today; evidence to `/home/cereal/ai-serve/evidence/strata-ram-20261028/phase0/`):
   - (i) Lexar throughput: `dd` 2 GiB write (`oflag=direct`) and read (`iflag=direct`) under `/mnt/lexar/omni-campaign/`, 3 runs each, noting what else is writing (`iostat -x 1`). Decides the prefetch numbers and whether the Lexar is the spill dir at all (if < 150 MB/s read, raise it with Cyril: a cold 150k restore would be ~15 s, still 40x better than a re-read, but the call is his).
   - (ii) Empty-slot probe on the live service, cyril tier via the gate: 10k prompt (bench `prefix10k-q1`), then one background-tier request, then the follow-up `-q1-again`. Expect `cache_n = 0` today (the bug) and ~10k after the patch. Repeat 3x. This is the before-number for D1(b).
   - (iii) Measure the exact-prefix rate on captured turns (scout script dir `/home/cereal/omni-campaign/scratch_scout/`, ai_serve `capture_request.body_z`): for consecutive turns of a conversation, `/apply-template` + `/tokenize` the previous request+reply and the next request; fraction where the previous is an exact token prefix. Informational: with checkpoints the tier works either way, but the number says how often the rollback path is taken.
   - (iv) Attribute the 9 GiB drift (D5): two `memory.stat` snapshots 30 min apart for every cgroup, diff anon.
   - (v) `ls /home/cereal/models/gguf` for a small GGUF for the CPU smoke test; `grep -n` the gate's `NATIVE_ROUTES`, `_generate`, `pick_slot` lines to pin the hook points for 1.4.0.

2. Build (tmux on the server, `_container_build.sh`, no service touched). In `/home/cereal/omni-campaign/export/llamacpp-xcache/`:
   - `cacheslots.patch` (D1): `tools/server/server-context.cpp` (`update_cache` empty-slot condition ~`:1588-1596`; `cache_store` honoured at `:1636` and `:2414`; metrics), `tools/server/server-task.{h,cpp}` (task field `cache_store` parsed from the request JSON, enum none/ram/disk; stored on the slot at launch).
   - `pcache-disk.patch` (D2): `server-task.{h,cpp}` (spill queue, writer thread, disk index, file format, janitor, `prefetch()`, `delete_tag()`, `flush()`, mutex over `states`), `server-context.cpp` + `server-http.cpp` (routes `POST /prompt-cache/prefetch|delete|flush`, internal-key gated like `/slots`), `common/arg.cpp` / `common/common.h` (`--prompt-cache-disk <dir>`, `--prompt-cache-disk-max-gib`, `--prompt-cache-disk-max-files`, `--prompt-cache-disk-max-age-h`, `--prompt-cache-disk-min-free-gib`, `--prompt-cache-spill-queue-gib`).
   - `apply_xcache.py` extended to apply a list in order: `xcache.patch` (only `ggml/src/ggml-backend.cpp`; no overlap), `cacheslots.patch`, `pcache-disk.patch`. Build twice: `bin-cs/` (xcache + cacheslots) and `bin-pc/` (all three); `bin/` untouched (rollback target). Record the build log and `sha256sum` of both `llama-server` binaries.
   - CPU smoke with a small GGUF, `-np 4 --cache-ram 512 --prompt-cache-disk <0700 tmp dir> --prompt-cache-disk-max-gib 2`: (a) two requests with `cache_store:none` add no entries (`/metrics`); (b) pinned-slot empty case: prompt on slot 0, a launch on slot 2, follow-up on slot 0 -> `cache_n ~ prompt length`; (c) fill past 512 MiB -> files appear, 0600, headers parse, `_spill_total` counts; (d) `POST /prompt-cache/flush` then prefetch -> `source:disk`, then the completion's `cache_n >= 0.98 x n_match`, greedy output identical to the RAM-served run; (e) `delete {"tag"}` removes RAM + files; (f) restart with the dir populated: index rebuilt, a file with a forged build string deleted; (g) `min-free-gib` above the free space -> spills dropped and counted, serving continues.

3. Gate and tools (local mirror `C:/Users/cereal/Documents/ai training/ai-serve/strata/`, deployed to `/home/cereal/ai-serve/strata/` hash-matched):
   - `kv_tier.py` (new): conversation keys, warm-table (key -> last_seen, bounded LRU of 10k keys), prefetch decision, `cache_store` / `cache_tag` assignment, counters.
   - `strata_gate.py` 1.4.0: reads `STRATA_KV_DIR`, `STRATA_KV_MIN_TOKENS` (2048), `STRATA_KV_PREFETCH_AFTER_S` (600), `STRATA_KV_TIERS` (`sprite,cyril`) from the environment (argparse defaults from `os.environ.get(..., "")`; no ExecStart change, critic 12); sets `cache_store` and `cache_tag` in `_generate` next to the existing `id_slot` assignment (~`:1036`); calls prefetch before `pump()` for eligible requests; post-turn `cache_n` accounting; `/gate/status` `kv` block and `mem_available_gib`; `NATIVE_ROUTES` unchanged (no `POST /slots`, no `/prompt-cache/*`).
   - `capture_admin.py` 1.2.0: `delete-member` also calls `POST /prompt-cache/delete` for the member's tag and writes the `service_event` row (raw parameterized SQL, `psycopg2`, manual connection).
   - `strata_llama.sh` 1.5.0: `STRATA_PCACHE_DIR` -> `--prompt-cache-disk` only if the dir exists, is owned by `cereal` and is mode 0700 (else log and start without it); `STRATA_PCACHE_MAX_GIB`, `_MAX_FILES`, `_MAX_AGE_H`, `_MIN_FREE_GIB`, `_SPILL_QUEUE_GIB` -> their flags; `STRATA_LLAMA_BIN` as today.
   - Tests: `tests/test_kv_tier.py` (keys, warm-table, prefetch decision, store/tag assignment, opt-out, header `X-Sprite-Capture: off`), the existing strata suite green; a fake-llama HTTP stub for the prefetch call.
   - Git: `/home/cereal/ai-serve` is not a git repo and the local mirror's parent (`Documents/ai training`) is a repo with no commits and no remote, so the gate code has no GitHub origin today. The patches' tree (`llamacpp-xcache/llama.cpp`) is an upstream checkout; the patches live outside it. Proposal (needs Cyril's one-word call): branch `ai-serve` of `autonomouscereal/chimera-omni` for the gate, launcher, patches and `apply_xcache.py` (the omni-lab-in-neural-forge pattern), work on `claude/strata-ram-20261028`, pushed to GitHub. Until he answers, the off-server copy `D:/SpriteBuild/strata-xcache-20261027` is updated with the patches. No KV files or weights in git.

4. Drop-ins and files (staged, not loaded; backups first):
   - Backups: every touched file copied to `/home/cereal/ai-serve/backups/strata-ram-20261028T<hhmm>Z/` with the timestamp in the name; `strata_ram.sh rollback` (new, in `/home/cereal/ai-serve/strata/`) restores them and is tested against the backup copies before the window.
   - `/etc/systemd/system/strata-llama.service.d/z-ramcap.conf` (sorts last; header states what, why, rollback): `Environment=STRATA_CACHE_RAM=24576`, `STRATA_LLAMA_BIN=/home/cereal/omni-campaign/export/llamacpp-xcache/bin-pc`, `STRATA_PCACHE_DIR=/mnt/lexar/omni-campaign/strata-kv`, `STRATA_PCACHE_MAX_GIB=40`, `STRATA_PCACHE_MAX_FILES=400`, `STRATA_PCACHE_MAX_AGE_H=168`, `STRATA_PCACHE_MIN_FREE_GIB=60`, `STRATA_PCACHE_SPILL_QUEUE_GIB=4`; `UMask=0077`. The vision lane's drop-ins are untouched.
   - `/etc/systemd/system/strata-gate.service.d/kvtier.conf`: `Environment=` lines only (`STRATA_KV_DIR`, `STRATA_KV_MIN_TOKENS=2048`, `STRATA_KV_PREFETCH_AFTER_S=600`, `STRATA_KV_TIERS=sprite,cyril`). No `ExecStart=`.
   - `/etc/sysctl.d/90-strata.conf`: `vm.swappiness=10` (root; after approval).
   - `mkdir -m 0700 /mnt/lexar/omni-campaign/strata-kv` as `cereal`.

5. Apply (one strata-llama restart, one strata-gate restart). Preconditions re-checked right before each restart: vision lane DONE/RELEASE and 15 quiet minutes; no PAUSE; `/gate/status` `last_foreground_age_s > 120`; Relay STATUS posted; run 0 not yet launched if at all possible (else coordinate a pause of its checkpoint writes for the window). Then: `systemctl daemon-reload`; `systemctl restart strata-llama` (~6 min load); confirm the journal start line shows `--cache-ram 24576`, the `bin-pc` path, `--prompt-cache-disk /mnt/lexar/omni-campaign/strata-kv`, and `/metrics` exposes the new gauges; `systemctl restart strata-gate` (capture re-reads `/props`); `sysctl --system`; confirm `cat /proc/sys/vm/swappiness` = 10 and `systemctl show strata-llama -p UMask` = 0077.

6. Correctness and speed (section 8).

7. Report on Relay with numbers and paths; write the memory file; update the runbook's Strata section and `strata-q4-switch-2026-10-27.md`'s rollback notes.

8. Soak 24 h (script `strata_ram_soak.py` in `/home/cereal/ai-serve/strata/`, systemd timer hourly, raw `psycopg2` into `service_event` kind `kv_soak`): MemAvailable (`/proc/meminfo`), `systemctl show strata-llama -p MemoryCurrent`, swap used, `/gate/status` `kv` block, eviction lines/hour, `_spill_total` and bytes, `_prefetch_*`, `kv_prefetch_lost`, Lexar free space, `_checkpoint_bytes`. Alarm (Relay STATUS) if MemAvailable < 50 GiB or `_spill_dropped_total` grows. Then fix the final target and decide `--cache-ram` 24 vs 32 GiB; D4 only if the soak misses the target.

## 8. Bench and correctness protocol (step 6)

Quiet machine: `gen-worker-text` and `gen-worker-media` stopped, run 0 not writing checkpoints (or paused), media idle; every case 3 runs, medians; bench `/home/cereal/ai-serve/strata/xcache/bench_strata.py` 1.1.0 (new cases below; keys via `STRATA_KEY_FILE`, never on a command line), results `bench_kv.jsonl`, evidence `/home/cereal/ai-serve/evidence/strata-ram-20261028/`, registered as an omni_lab probe run.

- Pinned-slot fix: probe (ii) repeated after the restart: 10k prompt, one background request, follow-up -> `cache_n >= 0.98 x prompt_n` (before: 0).
- Background isolation: 50 background-tier prompts -> `_ram_entries` unchanged, zero "making room" lines, `_store_none_total` +50.
- Greedy equality through the disk tier: prefix P (~66k, `--long 50000`) + question Q, temperature 0, fixed seed, 256 tokens -> A_ram. `POST /prompt-cache/flush` (RAM -> disk), confirm `_ram_entries` dropped and the file exists. Re-send P+Q+A_ram+Q2 -> gate prefetch returns `source:disk`; completion `cache_n >= 0.98 x n_match`; compare with the same request served from RAM (no flush): outputs identical token for token. Secondary: P+Q' (diverging question) after a flush -> rollback to the Q-start checkpoint, `cache_n ~ |P|`, not 0. Repeat at ~150k (`--long 115000`) and Sprite-shaped 10k. Also A_reread (cache deleted by tag, no disk) vs A_ram: first divergence >= 64 tokens (batch composition can move near-ties; informational).
- Prefetch timings from the route's `t_read_ms` and `bytes` at 10k / 66k / 150k, cold (after `DONTNEED`) and warm; TTFT of the prefetched request vs today's re-read TTFT.
- Stall check: a concurrent Sprite-tier decode stream during a 150k spill write and during a 150k prefetch: worst inter-token interval vs the same stream with the tier idle (pass: within noise of the 3-run medians); preemption `freed_s` during a spill (pass: unchanged); repeat the spill with a concurrent 4 GiB write to the Lexar from another process (run-0 checkpoint stand-in).
- Sprite regressions vs the 10-27 baseline (24.2 tok/s short write; 258 / 231 tok/s prompt read at 10k / 66k; 10k follow-up TTFT; `freed_s`): pass within +-5 % of the median or +-0.3 s TTFT.
- Delete: `capture_admin.py delete-member --member sprite:<test member>` removes RAM entries and files (verify `_ram_entries`, `ls`); an opted-out test member produces no file after a flush; a `cache_store:none` request produces no entry.
- Bounds: on the CPU smoke build set `max-gib 1` and confirm pruning; on the production build confirm the flags in the journal start line and that the janitor log line appears.

## 9. Rollback (each independent; all within a quiet window, Relay-posted)

- D1+D2 (llama): delete `z-ramcap.conf`, `daemon-reload`, restart strata-llama -> q4kxl.conf's 48 GiB cache on `bin/`, no disk tier, umask 0022. `bin-cs/` and `bin-pc/` stay inert. Partial: keep `z-ramcap.conf` but point `STRATA_LLAMA_BIN` at `bin-cs` and unset `STRATA_PCACHE_DIR` -> D1 only. `rm -rf /mnt/lexar/omni-campaign/strata-kv` (cache files only).
- Gate: delete `kvtier.conf`, restart strata-gate -> 1.4.0 code with `STRATA_KV_DIR` unset is a no-op (no `cache_store`/`cache_tag` fields, no prefetch); or restore `strata_gate.py`, `capture_admin.py`, `strata_llama.sh` from the backup dir via `strata_ram.sh rollback` and remove `kv_tier.py`.
- D5: delete `90-strata.conf`, `sysctl vm.swappiness=60`.

## 10. Risks and gates

- Touches two running production services and puts user conversation content on disk: gated on the vision lane's DONE, a Relay post, a quiet window, backups and the tested rollback before step 5; Cyril's approval for the sysctl, for the repo home, and for keeping conversation states (token ids in clear in the file headers) on the unencrypted Lexar, bounded and deletable, same class as the capture exports already there.
- Server-side patch surface: `pcache-disk.patch` is a real feature (writer thread, mutex over the cache, three routes, file format), larger than `cacheslots.patch`. Mitigations: CPU smoke matrix (step 2) before any production restart; `bin-cs` as the D1-only fallback; the loop never touches disk; a failure in the writer thread degrades to "no disk tier" (counted), never to a crash by design (all I/O errors are caught and counted; `std::bad_alloc` on promote returns `none`).
- Mutex contention: the loop's `load()`/`update()` and the HTTP thread's prefetch share one mutex; held only for container operations and the promote insert (a move), not for file reads. Measured by the stall check.
- Race: a promoted entry can be evicted by a concurrent launch before the completion loads it (`kv_prefetch_lost`). With a foreground-only cache at 24 GiB this needs ~24 GiB of new foreground entries in the gap; counted; the gate forwards the completion immediately after the prefetch returns.
- Tokenization parity between the prefetch route and the completion path: both use the same helpers, but a mismatch would only cost a `none` (today's behaviour); the bench compares the route's `n_match` to the completion's `cache_n`.
- Hybrid-model rollback inside a restored entry depends on the checkpoints in the file being the ones the RAM entry had; the file format stores them with `pos_min/pos_max`; section 8's greedy-equality and diverging-question cases prove it.
- Lexar: shared, 95 % full, USB. `min-free-gib 60` and the drop-and-count path protect the other lanes; a USB drop makes the tier read-fail (counted) and does not stop serving. The other lanes' Lexar budgets are collected before fixing 40 GiB.
- Dirty-page budget: the writer's 64 MiB chunked `sync_file_range` keeps Strata's dirty pages under the 128 MiB background threshold so run-0 checkpoint writes are not throttled by the spill; verified in the stall check with a concurrent write.
- Swap: until the restart, 4-5 GiB of llama-server anon is swapped; the restart clears it and `swappiness=10` keeps it out.
- The unattributed ~9 GiB/30 min drift is not Strata's; if probe (iv) finds a steady consumer, the target in section 6 moves by that amount and the owner lane is told.
- Wear: spill-on-evict writes only evicted states (typically < 24 GiB/day); no BX500 staging at all; the soak reports actual bytes.
- Not yet measured: Lexar throughput, real entry sizes and spill/prefetch timings, the mutex stall, the exact-prefix rate, the drift's owner (all in steps 1 and 6).

## 11. Critic findings: applied or rejected

| # | Finding | Decision | Reason |
|---|---|---|---|
| 1 | Disk restore has no checkpoints; any reply re-render difference -> full re-read; the step-6 test would fail | APPLIED | D2 is now spill-on-evict of RAM entries with their checkpoints; `/slots` save/restore is not used at all. The exact-prefix rate is measured (probe iii) for information only. |
| 2 | Empty pinned slot never loads from RAM (`:1588-1596`, NaN `f_keep`) | APPLIED | D1(b) adds the empty-slot condition; probe (ii) measures before, section 8 after. D1 sizing is revisited after the soak. |
| 3 | Save/restore stall the loop; page-cache write is throttled at 0.5 GiB on this host | APPLIED (by removing the mechanism) | No loop I/O remains: spills are written by a thread with chunked `sync_file_range`; restores are read in the HTTP thread; tmpfs staging is unnecessary and rejected. |
| 4 | Restore wipes a live slot; deferred save captures the wrong conversation | APPLIED (by design) | No slot restore exists; promotion inserts a RAM entry and the normal `load()` moves it into the slot at launch; no debounce, no slot lock. |
| 5 | Slot-id split hurts unpinned foreground requests | APPLIED | Per-task `cache_store` field stored on the slot at launch, honoured at both save sites `:1636` and `:2414`; `prompt_load` allowed on every slot. `--prompt-cache-slots` dropped. |
| 6 | Lexar has no free-space floor | APPLIED | `--prompt-cache-disk-min-free-gib 60` with `statvfs` before every write, drop-and-count, janitor prune; max 40 GiB; other lanes' budgets asked on Relay. |
| 7 | Skip the `cache_prompt:false` probe | APPLIED | Removed; the patch is required. |
| 8 | Umask 0022 -> 0644 files; `POST /slots` exposure; RAM copies survive delete | APPLIED | `UMask=0077` + explicit 0600; gate exposes no `POST /slots` or `/prompt-cache/*`; delete-by-tag removes RAM entries, queued spills and files. |
| 9 | Bound and RAM mirror underestimate; 9 GiB drift elsewhere | APPLIED | No mirror: the gate decides by warmth heuristics and llama answers from its real index; `/metrics` exposes RAM/disk/checkpoint bytes; drift attributed in probe (iv); bound restated with the 4 GiB spill queue and 22 GiB checkpoint worst case. |
| 10 | strata-think regresses | APPLIED | Think conversations get the disk tier (entries carry checkpoints; `preserve_thinking` gives exact matches when reasoning is echoed). |
| 11 | Large restore can fail on the unified pool | APPLIED (resolved by design) | Promotion goes RAM-first; the slot load uses the existing launch path and its space handling; no `/slots` restore. |
| 12 | `kvtier.conf` duplicates vision.conf's ExecStart; images uncacheable | APPLIED | Gate reads `STRATA_KV_*` from the environment; drop-in has `Environment=` only. Image prompts counted, not cached (unchanged behaviour). |
| 13 | Wear: save-every-turn x near-full USB SSD | APPLIED | Spill-on-evict only; superseded entries not spilled; no BX500 staging. |
| Re-plan | Ship 2 + 5 with the cap; disk tier as llama-side spill-on-evict with lookup on RAM miss | APPLIED with one change | Lookup is not done inside the loop's `load()` (that would stall all slots for a multi-second USB read); it is a gate-triggered `POST /prompt-cache/prefetch` served in the HTTP thread, which promotes the entry into RAM before the completion is forwarded. The loop stays I/O-free. |

## 12. What is NOT done, and why

- Experts to mmap / page cache (D4): not needed for the target; costs unmeasured prompt-read speed; contingency after the soak only.
- Spill-on-shutdown (flush RAM entries to disk on SIGTERM so restarts keep foreground states): useful but adds restart-time risk (24 GiB at Lexar speed is ~80 s against `TimeoutStopSec`); phase 2 after the tier has soaked.
- Per-task checkpoint cap for background slots: only if `_checkpoint_bytes` exceeds 12 GiB sustained in the soak.
- A PG index of cache files: rejected (second source of truth; the file headers are the index).
- `/slots` save/restore and `--slot-save-path`: not enabled; the mechanism cannot carry checkpoints and stalls the loop.
- Fewer slots (`-np`): no RAM gain worth Sprite's concurrency and the measured preemption path.
- `MemoryHigh=` on the unit: throttling risk; measure first.
- The Sprite-side prefix mismatch (58/94 misses): most likely the empty-slot bug (finding 2) and fixed here; any remainder after section 8 goes to the Sprite lead with the first differing message.
- Fixing the ~9 GiB/30 min drift: attributed in probe (iv), handed to its owner lane; not Strata's.
- Any change before the vision lane posts DONE/RELEASE, or without Cyril's calls on the sysctl, the repo home, and conversation states on the Lexar.
