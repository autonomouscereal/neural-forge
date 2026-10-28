# Strata host-RAM bound: design and plan (draft, 2026-10-28)

Order (Cyril, 10-28): "[Cyril quote withheld]". Constraints (Sprite lead, Relay 204074): 256k context, no output cap, Sprite's slot warm between turns, a disk-backed cache is fine if a follow-up RESTORES instead of re-reading, preemption in ms, decode and prompt-read within noise. Nothing here is applied; the vision lane (Relay 204067/204072) owns strata-llama until it posts DONE.

Evidence base: scouts `scout_ram`, `scout_llama`, `scout_usage` (10-28 ~06:40), my read-only SSH check on 10-28 (below), the live drop-ins, `strata_gate.py` 1.3.0, `strata_llama.sh` 1.4.0, `capture.py` / `capture_admin.py`.

## 1. Problem

Strata's host footprint is unbounded in practice: the 48 GiB RAM prompt cache fills in ~70 min after every restart (MemAvailable 130 -> 75 GiB; 28 GiB before the 05:30 restart), the 8 GiB swap is full with 4.3 GiB of llama-server pages in it, and the cache is churned by ~59 background prompts/hour that never reuse anything (hit ratio 0-3 %) while they evict Sprite's and Cyril's states (every background entry costs ~230 MiB of recurrent state). The media gate refuses all background work under 40 GiB MemAvailable and run 0 needs ~20 GiB, so Strata must be bounded at a level that leaves >= 40 GiB with margin.

Live numbers (10-28, my check): MemTotal 251.7 GiB, MemAvailable 74.7 GiB, Shmem 77.3 GiB, SwapFree 0.15 of 8 GiB, `vm.swappiness` 60, strata-llama PID 1985658 since 05:30:24 MDT, unit MemoryCurrent 187 GiB (of which ~40 GiB is reclaimable shard page cache), 17 "making room for prompt cache entry" evictions in the last 15 min. `/` (Crucial BX500 SATA SSD) 113 GiB free at 94 %; `/mnt/lexar` (`sdb2`, Lexar SL600: a USB 3.2 portable SSD; `ROTA=1` is the USB bridge's report, measure before trusting) 216 GiB free at 95 %.

Strata's host RAM today, by part: pinned experts 71.7 GiB (`/dev/zero` shared mapping, `cudaMallocHost`, not swappable); per-layer n-gram table + token embeddings ~27 GiB anon (`--lazy-mode off`, read per token on the CPU, must stay resident); RAM prompt cache up to 48 GiB anon; per-slot context checkpoints and compute buffers a few GiB; mmproj F16 ~0.9 GiB. RssAnon 74.9 = 27 + 48 to within a GiB, which says the anon part is the cache at its cap, not a leak.

## 2. Design (decisions, ranked by least harm to Sprite)

The cache is the knob. Pinned experts and the n-gram table are the model; they stay. The design makes the RAM cache foreground-only and smaller, and puts a disk tier under it so that a Sprite or Cyril follow-up after an eviction (or a restart) restores instead of re-reading.

### D1. RAM prompt cache: foreground-only (item d) + 24 GiB cap (item a). One llama restart, shipped together.

Shipping (a) without (d) harms Sprite: a 24 GiB cache churned by background turns over in 0.5-1.7 h instead of 1-3.5 h. So they go in one restart.

(d) Background prompts must not enter the cache. In this build the cache is global (`server_prompt_cache`, `server-context.cpp:1359`), and `--cache-idle-slots` (default on) saves EVERY idle slot into it at every task launch (`server-context.cpp:2409-2423`), then clears the slot under `--kv-unified`. The gate already pins background to slots 2-5 (`slot_plan`, `strata_gate.py:306-317`), so the cheapest correct cut is by slot id:

- New flag `--prompt-cache-slots N` (default: all): in the idle-slot loop, `prompt_save(slot)` only when `slot.id < N`; `prompt_load` likewise. Slots >= N are still `prompt_clear`ed under unified KV as today. ~15 lines in `tools/server/server-context.cpp` + `common/arg.cpp` + `common/common.h`. Patch file `cacheslots.patch` beside `xcache.patch` in `/home/cereal/omni-campaign/export/llamacpp-xcache/` (xcache.patch only touches `ggml/src/ggml-backend.cpp`; no overlap). Build to a NEW `bin-cs/` so rollback is `STRATA_LLAMA_BIN` back to `bin`.
- Pre-check first (no restart, 10 min): send one background-tier request with `"cache_prompt": false` and watch the `/metrics` prompt-cache gauges and the journal for a new entry. If `cache_prompt:false` already stops the idle save, the gate sets it for the background tier (one line in `_generate`) and the patch is not needed. Expectation from the source (the idle-slot loop keys on the slot, not the task): it does not; the patch is the plan.
- Rejected: `--no-cache-idle-slots` (slots keep their KV in the unified 262,144-token pool; Cyril at 178k + Sprite + 4 background slots overflows the pool and the failure mode in this build is unverified: unacceptable for Sprite). Rejected: a second llama-server for background (another 100 GiB of host RAM).

(a) `STRATA_CACHE_RAM=24576` (24 GiB, was 49152). Sizing from `scout_usage`: foreground entries only, 13 KB/token + ~0.3 GiB state: Sprite ~0.37 GiB each (max 39/h, 48/day), Cyril p50 0.7, p90 1.6, max 2.6 GiB (max 10/h, 12/day). 1-hour horizon ~24 GiB; 24-hour horizon ~30 GiB. With the disk tier (D2) behind it, 24 GiB covers the 1-hour horizon at the observed peaks and the disk tier covers the rest (Cyril's gap p99 is 4.7 h). If D2 slips, use 32 GiB (frees 16 instead of 24).
- Frees: 24 GiB. Sprite effect: none when D2 is live; the RAM tier still serves every follow-up inside ~1 h.

### D2. Disk tier on the Lexar for foreground conversations (item b): gate-side slot save/restore.

The build has no disk tier for the RAM cache (entries are `std::vector<uint8_t>`, `server-task.h:597-610`); the only disk path is `--slot-save-path` + `POST /slots/{id}?action=save|restore` (`server-context.cpp:2518-2625`), currently disabled. The save file uses the same hybrid `state_write` (attention KV + recurrent state, `llama-memory-hybrid.cpp:190-194`) that the working RAM cache uses, minus the context checkpoints. That is the strongest correctness argument: a restore from disk exercises the serialization the RAM cache already proves every hour.

Mechanics (gate `strata_gate.py` 1.4.0 + new `strata/kv_tier.py`; raw asyncpg for the index; file I/O in `asyncio.to_thread`):

- Keys. For a request's `messages`, the gate computes running sha256 hashes over the canonical JSON of messages[0..i] (role, content, tool_calls, tool_call_id; nothing else). A saved state is keyed by the hash of the request's messages plus the assistant reply appended as a message. On a new foreground request the key of messages[0..n-2] (everything but the new user/tool message) is looked up. Exact continuation only in phase 1 (the restored slot has no checkpoints, so a divergence inside the saved prefix re-reads anyway).
- Save. After a foreground turn (tiers sprite, cyril), alias `strata` (non-think; see risks), outcome ok, `prompt_tokens + completion_tokens >= --kv-disk-min-tokens 2048`, member not opted out, slot pinned: debounce 3 s (Sprite tool loops: hit-gap p50 0.01 s), then `POST /slots/{slot}?action=save {"filename": "<key32>.gsq"}` into the staging dir, check `n_saved >= 0.95 x expected tokens` (0 means another launch already round-tripped the slot through the RAM cache: skip, count `kv_save_empty`), move the file to `/mnt/lexar/omni-campaign/strata-kv/<sha256(member_key)[:16]>/<key32>.gsq` (copy + fsync + unlink, in a thread), upsert `strata_kv_state`. The gate keeps the ticket's slot held during the save so no foreground request is placed on it meanwhile.
- Why stage on `/`: the save runs inside llama-server's task loop and blocks every slot while it writes. Writes into page cache on the SATA SSD cost about a memcpy (2.2 GiB ~1-2 s, the D2H copy included); writing straight to the USB disk would be throttled by per-device dirty-page balancing to that device's speed. The staging dir is bounded: files leave within seconds; a janitor deletes anything older than 10 min; at most 3 files / 8 GiB (`/` is at its run-0 floor).
- Restore. On a foreground request whose key is indexed: decide RAM-resident vs not with the gate's mirror of the RAM cache (the gate is llama's only client, so it sees every entry llama makes: per conversation one entry, replaced each turn, evicted oldest-first; sizes estimated at 13 KB/token + 300 MiB; "resident" only if the entry sits within the newest 0.7 x cache_ram and no llama restart since, detected by `llamacpp:prompt_tokens_total` in `/metrics` going backwards). If resident: forward as today. Else: copy the file Lexar -> staging in a thread (page cache warm), `POST /slots/{slot}?action=restore`, check `n_restored`, then forward with the same `id_slot`. After the turn, compare `timings.cache_n` to `prompt_n`; a resident verdict that produced `cache_n < 0.5 x prompt_n` counts `kv_mirror_miss` and lowers the factor by 0.1 (floor 0.3). A failed restore forwards anyway (re-read) and counts `kv_restore_failed`. Unpinned tickets (every foreground slot held) get no restore.
- Bounds and privacy. Directory 0700 (`cereal`), files 0600, created by the gate with umask 077; `--kv-disk-max-gib 48`, `--kv-disk-max-age-h 168`, `--kv-disk-max-files 400`; a janitor every 10 min prunes by age then LRU (`last_used_at`). The files hold the token ids in clear (the GGSQ header stores the token list) plus KV: capture-grade data. Per-user delete: `capture_admin.py delete-member` 1.2.0 also deletes the member's `strata_kv_state` rows and files; members with `capture_member.consent = 'opted_out'` and requests with `X-Sprite-Capture: off` get no disk tier (RAM only). Member key = capture's (`cyril`, `sprite:<24hex>`; `member_and_source`, `capture.py:650`).
- Expected numbers. Sizes: 10k ~0.36 GiB, 66k ~1.1 GiB, 150k ~2.2 GiB (13 KB/token for 12 q8_0 attention layers + ~0.23 GiB recurrent state; consistent with the measured entry distribution, max 1.7 GiB at the ~100k conversations). Restore time = file read (USB SSD: 2.2 GiB at 400 MB/s ~6 s, at a pessimistic 150 MB/s ~15 s; Sprite 0.36 GiB ~1-3 s) + the in-loop load (~1-2 s at 150k, ~0.3 s at 10k). Re-read today: 10k ~43 s, 66k ~290 s, 150k ~660 s at 227-258 tok/s. Follow-up gain after an eviction: 10-60x. Cost when the RAM tier has it: zero (no restore issued).
- Frees: 0 GiB by itself; it is what makes D1's smaller cache safe for Sprite.
- Rejected: saving every turn straight to the Lexar (writer throttling stalls the server loop); keying on `X-Sprite-Delegation` / `X-Sprite-Run` (not every turn has them, Cyril's clients have none); a built-in RAM -> disk spill (does not exist in b96806d96; it would be a server-side patch of `server_prompt_cache` far larger than D1's and would still need the LRU/age/delete policy).

### D3. Slot count (item c): keep `-np 6`, `--bg-max 4`. No change.

With `--kv-unified` the slot count adds almost no KV memory (ctx256k.conf: +490..728 MiB per card for the whole pool). What slots multiply is per-slot context checkpoints: `--ctx-checkpoints 16` x ~0.23 GiB = 3.7 GiB worst case per slot, ~22 GiB across 6 in theory, but checkpoints are only made as a prompt is read, so background slots (p50 272 tokens) hold ~1 and the measured entry maximum (1.7 GiB including checkpoints) says the real total is a few GiB. Fewer slots would cut Sprite's concurrency and change the preemption path the Sprite lead measured; not worth single-digit GiB. Background work through the gate adds no host RAM after D1 (its states are never cached), so the gate's `--mem-floor-gib 12` (support lane) stays.

### D4. Cold experts to mmap / page cache (item e): NOT in this plan; contingency only.

Not needed once D1 lands (section 4 shows ~75 GiB MemAvailable with run 0). If it is ever needed: `--no-host` or `GGML_CUDA_NO_PINNED=1` plus `--load-mode mmap` makes the expert tensors file-backed (`llama-model.cpp:1026-1056`, `ggml-cuda.cu:1279-1293, 5024-5037`); xcache still recognizes host-resident experts (`ggml_backend_buffer_is_host`). Costs to measure before any use: pageable `cudaMemcpy` for expert misses (slower than pinned), and page-cache misses hitting the BX500 at ~500 MB/s during prompt batches ("a prompt is assembled from the slots plus host copies"). Decode would mostly survive because the VRAM LRU caches (30+50 GB) hold nearly all 71.7 GiB of experts; prompt read is the exposed path. Measure on a quiet machine: prompt read tok/s at 10k/66k and decode at 0/66k, `-cmoe` pinned vs mmap, 3 runs each; adopt only if both are within noise. Potential: 55-72 GiB becomes reclaimable.

### D5. Other accounting our lanes own (item f).

- Swap: `vm.swappiness=10` via `/etc/sysctl.d/90-strata.conf` (root; small change, needs approval). Today 4.3 GiB of llama-server anon (cache entries / checkpoints) is swapped out; a swapped entry read back stalls a restore. After the restart those pages are gone; the lower swappiness keeps Strata's anon in RAM ahead of stale page cache.
- Page cache attributed to strata-llama (~40 GiB from the shard load under `--no-mmap`) is reclaimable and already inside MemAvailable; judge the bound by MemAvailable, not `MemoryCurrent`. Rejected for now: `MemoryHigh=` on the unit (it would reclaim that cache early, but it throttles the process if anon ever crosses it; measure first).
- agent-memory-postgres (15.8 GiB cgroup, ~4 GiB RSS, the rest page cache) and user.slice (14.4 GiB, 9 GiB cache) are reclaimable; no action.
- mmproj on CPU (~0.9 GiB + compute buffer): known, bounded, stays (vision lane).
- strata-gate `/gate/status` gains a `kv` block (counters, RAM-mirror size, disk bytes/files) and `MemAvailable`, so the soak is one GET.

## 3. Schema (ai_serve, raw SQL; migration `db/migrations/0009_strata_kv_state.sql`)

```sql
CREATE TABLE IF NOT EXISTS strata_kv_state (
    key           text        PRIMARY KEY,              -- sha256 hex of the canonical message prefix incl. the reply
    parent_key    text,                                 -- the key this turn extended (NULL for a first turn)
    member_key    text        NOT NULL,                 -- 'cyril' | 'sprite:<24hex>' (capture_member.member_key)
    tier          text        NOT NULL CHECK (tier IN ('sprite', 'cyril')),
    alias         text        NOT NULL,                 -- 'strata' (phase 1), 'strata-think' later
    n_tokens      integer     NOT NULL,
    bytes         bigint      NOT NULL,
    path          text        NOT NULL,                 -- under /mnt/lexar/omni-campaign/strata-kv/
    build         text        NOT NULL,                 -- llama-server build + model ident from /props
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_used_at  timestamptz NOT NULL DEFAULT now(),
    restores      integer     NOT NULL DEFAULT 0,
    deleted_at    timestamptz
);
CREATE INDEX IF NOT EXISTS strata_kv_state_member_idx ON strata_kv_state (member_key) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS strata_kv_state_lru_idx    ON strata_kv_state (last_used_at) WHERE deleted_at IS NULL;
-- service_event kinds added: kv_save, kv_restore, kv_prune, kv_delete_member (detail jsonb; never prompt text)
```

`build` guards restores across model or build swaps (a state from another GGUF or KV type is never restored). The files are a cache and the index is rebuilt from the directory if lost.

## 4. Target and expected accounting

Bound on Strata's host RAM after D1: pinned experts 71.7 + n-gram/embeddings ~27 + cache <= 24 + checkpoints and compute ~3-5 (worst case ~22 if six slots each hold 16 checkpoints) + mmproj ~1 = ~128 GiB typical, ~150 GiB worst. It cannot grow past that: every part is a fixed tensor or a capped cache.

MemAvailable: 74.7 GiB now with the 48 GiB cache full; +24 after D1 = ~98 GiB; run 0 (-20) = ~78; TTS caption renders / media data generation on GPU4 (-5..-8) = ~70. Target: >= 60 GiB steady with Strata serving (cache full), run 0 training and media background running; hard floor 40 GiB (media-gate, unchanged); alarm in the soak if < 50 GiB. Refine after the 24 h soak (section 5, step 8).

## 5. Plan (ordered; nothing restarts strata-llama / strata-gate before step 5)

0. Relay. Post this plan as STATUS on thread 52468f63 (`session.py ... say --type STATUS --body-file`); read the vision lane's state with `--event-id` (my `show --history` call returned `event_or_message_id_required`; use 204067 / 204072 and newer). Wait for its DONE or RELEASE and 15 quiet minutes on strata-llama. Honour any PAUSE.
1. Phase-0 probes (read-only, today). (i) Lexar throughput: `dd` 2 GiB write and read under `/mnt/lexar/omni-campaign/` with `oflag=direct` / `iflag=direct`, 3 runs. (ii) `cache_prompt:false` probe through the background key (one request; `/metrics` + journal). (iii) Ask the Sprite lead for the first differing message in the 58/94 Sprite follow-ups that missed with 0 background requests in between (`scout_usage`): a changing prefix defeats both tiers and must be fixed on the Sprite side; not a blocker for this plan. (iv) List small GGUFs on the server for the CPU smoke test (`ls /home/cereal/models/gguf`).
2. Build. In `/home/cereal/omni-campaign/export/llamacpp-xcache/`: write `cacheslots.patch` (`tools/server/server-context.cpp` idle-slot loop ~2409-2423 and `prompt_load` ~1626-1642; `common/arg.cpp` new `--prompt-cache-slots`; `common/common.h` `prompt_cache_slots`), apply after `xcache.patch` through `apply_xcache.py` (extend it to take a list), build with `_container_build.sh` into `bin-cs/` (`bin/` untouched), in tmux. Smoke on CPU with a small GGUF: `-np 4 --cache-ram 512 --prompt-cache-slots 2 --slot-save-path <0700 dir>`: slots 2-3 add no entries (`/metrics`), slots 0-1 do; a save/restore round trip on slot 0 returns `n_restored == n_saved` and an identical greedy continuation.
3. Gate code (local mirror `C:/Users/cereal/Documents/ai training/ai-serve/strata/`, deployed to `/home/cereal/ai-serve/strata/` hash-matched): `kv_tier.py` (new); `strata_gate.py` 1.4.0 (flags `--kv-disk-dir`, `--kv-stage-dir`, `--kv-disk-max-gib 48`, `--kv-disk-max-age-h 168`, `--kv-disk-max-files 400`, `--kv-disk-min-tokens 2048`, `--kv-disk-tiers sprite,cyril`, `--kv-disk-aliases strata`; hooks in `_generate` before `pump()` and after the turn; `/gate/status` `kv` block); `capture_admin.py` 1.2.0 (delete-member removes kv rows and files); `db/migrations/0009_strata_kv_state.sql`; `strata_llama.sh` 1.5.0 (`STRATA_SLOT_SAVE_DIR`: passed only if the dir exists, is owned by cereal and is mode 0700; `STRATA_CACHE_SLOTS` -> `--prompt-cache-slots`); tests `tests/test_kv_tier.py` (fakes: keys, mirror, debounce, prune, opt-out, delete) plus the existing strata suite green. Git: `/home/cereal/ai-serve` is not a git repo and the local mirror's parent (`Documents/ai training`) is a repo with no commits and no remote, so ai-serve has no GitHub origin today. Proposal: branch `ai-serve` of `autonomouscereal/chimera-omni` (the omni-lab-in-neural-forge pattern), work on `claude/strata-ram-20261028`; needs Cyril's one-word call. No KV files or weights in git.
4. Drop-ins (staged, not loaded): `/etc/systemd/system/strata-llama.service.d/z-ramcap.conf` (sorts last): `STRATA_CACHE_RAM=24576`, `STRATA_LLAMA_BIN=/home/cereal/omni-campaign/export/llamacpp-xcache/bin-cs`, `STRATA_CACHE_SLOTS=2`, `STRATA_SLOT_SAVE_DIR=/home/cereal/ai-serve/run/kvstage` (header: what, why, rollback). `/etc/systemd/system/strata-gate.service.d/kvtier.conf`: vision.conf's ExecStart plus the `--kv-*` flags (one `ExecStart=` reset; sorts after vision.conf). `/etc/sysctl.d/90-strata.conf`: `vm.swappiness=10`. Directories `/home/cereal/ai-serve/run/kvstage` and `/mnt/lexar/omni-campaign/strata-kv`, 0700. Backups of every touched file in `/home/cereal/ai-serve/backups/strata-ram-20261028T<hhmm>Z/` with timestamps; the rollback script `strata_ram.sh rollback` tested against the backup copies before the window.
5. Apply (one strata-llama restart + one strata-gate restart). Quiet window: no sprite or cyril request through the gate for > 2 min (`/gate/status` `last_foreground_age_s`), re-checked right before each restart; Relay STATUS before. `daemon-reload`, `systemctl restart strata-llama` (~6 min load); confirm the journal start line shows `cache-ram 24576`, the `bin-cs` path, `--prompt-cache-slots 2`, `--slot-save-path`; apply migration 0009; `systemctl restart strata-gate` (capture re-reads `/props`). Then `sysctl --system`.
6. Correctness and speed (quiet machine: gen-worker-text / gen-worker-media stopped, before run 0 launches if at all possible; 3 runs each, medians; bench `/home/cereal/ai-serve/strata/xcache/bench_strata.py` 1.1.0 with new cases, results `bench_kv.jsonl`, evidence `/home/cereal/ai-serve/evidence/strata-ram-20261028/`, registered as an omni_lab probe run):
   - Greedy equality: prefix P (~66k, `--long 50000`) + question Q at temperature 0, fixed seed, 256 tokens -> A_ram (served from the RAM tier). Evict (restart strata-llama inside the window, or flood 100 distinct foreground-tier prompts), re-send -> A_disk through a restore. Require A_disk == A_ram token for token and `cache_n >= 0.98 x n_restored`. Also A_reread (cache cleared, no restore) as a secondary: first divergence >= 64 tokens (batch composition can move near-ties). Repeat at ~150k (`--long 115000`) and at a Sprite-shaped 10k.
   - Background isolation: 50 background prompts -> `/metrics` prompt-cache entry count unchanged, zero "making room" lines.
   - Save and restore timings from the `/slots` response (`n_bytes`, `t_ms`) at 10k / 66k / 150k, and the worst token interval of a concurrent decode stream during a save or restore (the stall).
   - Sprite regressions: short-write tok/s, 10k prompt read tok/s, 10k follow-up TTFT, preemption `freed_s` (n=3 each) against the 10-27 baseline (24.2 tok/s write; 258 / 231 tok/s read at 10k / 66k). Pass: within +-5 % of the median or +-0.3 s TTFT.
   - Delete hook: `capture_admin.py delete-member --member sprite:<test member>` removes rows and files; an opted-out member gets no file.
7. Report on Relay with numbers and paths; write the memory file; update the runbook's Strata section.
8. Soak 24 h: `sar -r` MemAvailable, `systemctl show strata-llama -p MemoryCurrent`, eviction lines per hour, `/gate/status` kv counters (`kv_save`, `kv_save_empty`, `kv_restore`, `kv_restore_failed`, `kv_mirror_miss`, disk bytes), swap used. Then fix the final target number and decide `--cache-ram` 24 vs 32 GiB. D4 only if the soak misses the target.

## 6. Rollback (each step independent)

- D1: delete `z-ramcap.conf`, `daemon-reload`, restart strata-llama in a quiet window -> q4kxl.conf's 48 GiB cache on `bin/`. `bin-cs/` stays (inert).
- D2: delete `kvtier.conf`, restart strata-gate (-> the 1.3.0 flags; 1.4.0 code with `--kv-disk-dir` unset is a no-op), or restore `strata_gate.py` / `capture_admin.py` from the backup dir; `rm -rf` the two 0700 directories; migration 0009 may stay (harmless) or `DROP TABLE strata_kv_state`.
- D5: delete `90-strata.conf`, `sysctl vm.swappiness=60`.

## 7. Risks and gates

- Touches a running production service (strata-llama, strata-gate) and puts user conversation content on disk: needs the vision lane's DONE, a Relay post, a quiet window and backups before step 5; Cyril's approval for the sysctl, for the repo home, and for keeping conversation states on the unencrypted Lexar (the same class as the capture exports already there).
- Hybrid restore correctness: the file path and the RAM cache share `state_write` / `state_read` (attention KV + recurrent state), but the restored slot has NO checkpoints, so only exact continuations restore; any divergence re-reads from zero (the same as today after an eviction, never worse). `strata-think` follow-ups diverge by design (the template strips reasoning from history), so think conversations stay RAM-only in phase 1 (the RAM entry's end-of-prompt checkpoint covers that case). Phase 3 idea, not planned: write the checkpoints into the save file.
- Server-loop stall: save and restore run inside llama-server's task loop (~0.3 s at 10k, ~1-2 s at 150k from page cache). Mitigated by staging on the SSD, the 2048-token minimum and the 3 s debounce; measured in step 6. If the 150k stall hurts a concurrent Sprite stream, raise `--kv-disk-min-tokens` for cyril or save only when the RAM mirror predicts eviction.
- Race: a task launched between the turn's end and the save round-trips the slot through the RAM cache (`n_saved = 0`): detected, skipped, retried next turn.
- SSD wear: ~270 foreground turns/day x ~0.6 GiB = ~160 GB/day through the BX500 staging and again onto the Lexar (~8 % of a 720 TBW rating per year). Acceptable; the soak reports actual bytes.
- Privacy: token ids in clear inside the files; 0700 / 0600, bounded size and age, per-member delete, opt-out honoured. Nothing leaves the host.
- The Sprite-side prefix mismatch (58/94 misses with no eviction) limits the value of both tiers for Sprite until the Sprite lead fixes it; this plan does not depend on it.
- Not measured yet: Lexar throughput (step 1), real save sizes and stall (step 6), the `cache_prompt:false` behaviour (step 1).
