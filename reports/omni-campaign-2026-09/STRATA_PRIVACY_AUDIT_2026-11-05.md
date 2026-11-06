# Strata privacy audit for Sprite private turns (2026-11-05)

GPU-layout side session (998ab6), for the Omni lead and the Sprite lead. This audit was **read-only**: nothing was changed.

**The promise.** A Sprite private-tab turn carries `X-Sprite-Capture: off` and `X-Sprite-Train: off`, and nothing of its prompt or answer text is kept anywhere. Metadata is fine.

## Verdict

For the real private path (the Sprite key, sprite tier, capture off), none of the persistent stores keeps any text:
- the capture tables keep nothing;
- request_log holds metadata only;
- the logs and the disk prompt cache hold nothing.

Text does stay in three places:
1. **The RAM prompt cache.** The gate deliberately sends capture-off turns `cache_store: "ram"`, so the turn's tokens and KV sit in llama-server's 16 GiB RAM cache until they are evicted or the server restarts. A "pinned" slot is an upstream llama-server slot the gate assigns to a tier; Sprite's is slot 0. The turn's KV also stays in slot 0's VRAM until the next Sprite request. **Fixes F2 and F3.**
2. **Media.** If a private turn delegates media, ComfyUI writes the prompt into the output file's metadata. The files are kept indefinitely (`comfyui/output/sprite`, back to 10-15), the prompt JSON stays in ComfyUI history in RAM, and the media gate has no capture-off handling. **Fix F4**, or Sprite never delegates media from a private tab.
3. **The lab key.** The background tier ignores `X-Sprite-*`, so every lab-key request is captured in full, prompt and answer, even when sent with capture off. The canary proved this. **Fix F1.**

The canary test used `strata_key_sprite_lab`, background tier, as instructed. It therefore exercised the background path, not the private sprite path. The sprite-tier rows below come from the code. A sprite-tier canary inside the approved Phase 0 window would confirm them empirically.

## Canary test (11-05, 09:14-09:20Z)

- **Canary:** sha256 `c85533159b5486e024e90ee19c28d8e7bd0de06147e7e8070e49b11856371ae5`. It was a random string, never printed, put in a command line or written to a file by the test.
  - The server-side scanner (`work/privacy-audit-20261105/privacy_scan.py`) read it from stdin.
  - A first run (sha256 `87694c8f…499e`) was stopped before its scan. Its request was found later by sha256 in the same capture rows.
- **Request A:** one streamed request on the lab key with `X-Sprite-Capture: off` and `X-Sprite-Train: off`, asking the model to repeat the canary. Request d66c62d5: done, the answer contained the canary, 193 s mostly spent queueing behind the data lanes.
- **Scan 1** ran 30 s after A. Then **request B**, a normal next turn on the same key ("2 + 2"): its answer held no canary. **Scan 2** ran 30 s after B.

| Store scanned | Scan 1 | Scan 2 |
|---|---|---|
| journald, every unit | 0 | 0 |
| Files modified since the test on the request path (`ai-serve` incl. the capture spool, `local-hermes`, `comfyui` minus models, `.config`, both prompt-cache disk dirs, `/tmp`, `/var/tmp`, `/var/log`, `/run/user/1000`, a 3-deep home sweep) | 0 of 27 | 0 of 30 |
| `request_log` | 0 of 7 rows | 0 of 14 |
| `capture_request` | **1 row** (A, `body_z`) | **1** (A only) |
| `capture_response` | **1 row** (A, `content`); found by request id | A only; B clean |
| `capture_blob`, `capture_opt_out`, `service_event` | 0 | 0 |
| llama-server `/slots` | no match | no match |
| strata-gate `/gate/status` | no match | no match |

## Per store

| Store | Keeps prompt/answer text? | Retention | Capture-off honoured? |
|---|---|---|---|
| strata-gate `request_log` | No: tier, client id, model, token counts, timings, outcome, error code (canary 0) | indefinite (metadata) | n/a |
| strata-gate journal | No: `access_log=None`; events carry counts only (canary 0) | journald rotation | n/a |
| capture, **sprite tier** (`capture_request` / `_response` / `_blob`, spool `capture-pending`) | **No.** An opted-out handle holds only the member key, the gate never attaches the answer, and only an hourly count goes to `capture_opt_out`. No sprite-tier capture rows exist at all since 10-22. | none kept | **Yes** (code, capture 1.2.0) |
| capture, **background tier** (data lanes and the **lab key**) | **YES**: full prompt and answer (canary in `body_z` and `content`) | indefinite (oldest 10-22, 84.7k rows; deleted only by `capture_admin delete-member`) | **No**: `X-Sprite-*` is read on the sprite tier only. Lab-key rows so far: 7 background-unlabelled, sprite-test-node-mcp, and my tests and canaries. Held from export, not exported. |
| capture export (hourly timer) | Background rows leave only if their source is labelled exportable; Claude, Codex and test sources are excluded | — | n/a |
| llama-server journal | No at verbosity 3 (canary 0) | journald | n/a |
| llama-server `/slots` (cyril tier and the gate) | No text fields: counts, sampler parameters, the template's generation prompt | — | n/a |
| llama-server **RAM prompt cache** (`--cache-ram 16384`) | **YES**: token ids and KV of capture-off sprite turns (`cache_store: "ram"`). No API returns text from it. A later request reuses an entry only for an identical token prefix. | until LRU eviction or restart | **Partly**: kept off disk, not out of RAM |
| llama-server slot KV in VRAM (sprite pinned to slot 0) and ctx checkpoints | **YES**, in VRAM/RAM | until the next request on that slot, or the idle-slot save (which moves a `ram` state into the RAM cache) | n/a |
| prompt-cache **disk** tier (`/mnt/lexar/omni-campaign/strata-kv`, `/mnt/strata-kv`) | No: OFF (no `STRATA_PCACHE_DIR`), both dirs empty; capture-off maps to `ram` even when it is on | — | Yes (policy) |
| pinned host tier (`GGML_EXPERT_TIER`) | No: expert weights only | — | n/a |
| Hermes front (`127.0.0.1:18490`; ledger, provenance, factual-drops) | Keeps lane text (factual-drops stores whole requests), but it is bound to loopback. Sprite (on .84) cannot reach it; Sprite calls the gate on `:18400`. Canary 0. | — | n/a (not on Sprite's path) |
| media gate + ComfyUI (only if a private turn delegates media) | **YES**: an embedded `prompt` key in every Sprite output checked (PNG, WebP; 24 files since 10-15, uuid names, no cleanup); ComfyUI `/history` in RAM; reference uploads in `comfyui/input`; no `--disable-metadata` on any GPU4 instance | files indefinite; history until a container restart | **No**: no capture-off handling in `media_gate.py` |
| request headers | `X-Sprite-*` is never forwarded to llama-server (gate 1.2.1) | — | — |

## Fix proposals: none applied, each needs the Omni lead's go

The four Strata fixes (F1, F2, F3, F5) are one strata-gate code change, in two files. Ship them as one quiet-window gate restart (llama-server untouched), together with the pending lab-key rotation (runbook 15.3).

**F1. Capture-off on every tier** (`capture.py`).
- Change: read `X-Sprite-Capture` and `X-Sprite-Train` on every tier.
  - `X-Sprite-Capture: off` → the opt-out handle (keep nothing, count only). Opting out only lowers retention, so honouring it from any key is safe.
  - `X-Sprite-Train: off` → source `sprite-test`, never exported.
  - Member selection stays sprite-only, so no other key can pick a member.
- Effect: the lab key stops being captured when it asks.
- Rollback: the previous `capture.py` plus a gate restart.

**F2. Capture-off is not cached** (`kv_tier.decide`).
- Change: `store = STORE_NONE` when the header says capture off, instead of `STORE_RAM`.
- Per `strata-pcache.patch`, the turn's state then never enters the RAM cache, and the idle-slot save empties the slot.
- Cost: a `none` request does not load from the cache either. It still reuses slot 0's in-place prefix, which is normally the Sprite system prompt from the last family turn. TTFT changes only when slot 0 holds another conversation.
- Members opted out by consent (not by header): `ram` or `none` is the lead's call.
- Rollback: the previous `kv_tier.py` plus a gate restart.

**F3. Erase the slot after a capture-off turn** (gate).
- Change: after the response ends, `POST /slots/{id}?action=erase` on the pinned slot. The private KV leaves VRAM at once instead of waiting for the next Sprite request.
- Cost: the next Sprite turn on slot 0 re-prefills from the RAM cache.
- Rollback: the previous gate.

**F4. Media from a private turn** (Sprite or the media gate).
- Cheapest: Sprite never delegates media from a private tab. **Sprite lead: does it today?**
- Otherwise: the media gate honours `X-Sprite-Capture: off`. Once the client has the outputs, it deletes the output and input files and `DELETE`s the ComfyUI history entry.
- Also add `--disable-metadata` to the general ComfyUI instance (registry only, at the next switch). The training lane records its provenance itself; check `media_validate` first.
- Rollback: the previous `media_gate.py` and the registry arg.

**F5. Clean SSE error boundary** (Sprite lead's finding).
- Problem: the gate relays raw upstream pieces (`iter_any`), so a pre-empted stream can end mid-event, and the `data: {"error": …}` line is glued onto it (`strata_gate.py` near line 1506).
- Change: relay only complete events (buffer up to the last `\n\n`). The error event then always starts on a clean boundary. Prefixing a bare `\n\n` would still leave a truncated JSON event behind.
- Rollback: the previous gate.

**F6. Test residue.** My lab-key test rows (5) and the earlier lab-node rows hold synthetic text only. Delete them by client 6 / request id on the lead's go: `capture_admin` gets a `delete-request` / `delete-client` action, or raw parameterized SQL.

**Verification after the change.**
- Run the same canary at the sprite tier, during the Phase 0 window or with the lead's OK on the Sprite key, plus a next turn. Expect 0 in every store.
- Read the llama RAM cache entry count for the member tag.
- Check `/slots` after F3.

## Method notes

- The canary lived only in the test process and the request body. The scanner took it on stdin, because `--script` runs land in `/tmp/.ssh_client_cmd_*` and Sysmon logs command lines.
- Each scan held the shared SSH lease only for about 70 s.
- `request_log` has no IP column and the gate keeps no access log, so attribution is by request id and timing.
