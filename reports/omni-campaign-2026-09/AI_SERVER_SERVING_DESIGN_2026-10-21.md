# AI Server Serving Design: Strata brain, media worker, idle-time corpus generation, P40 lanes

Date: 2026-10-21. Status: DESIGN (nothing deployed; nothing on the server changed). Author: architect lane.
Inputs: the verbatim instruction from Cyril of 2026-10-21, the GPU policy from the lead, the read-only server survey of 17:25Z,
`OMNI_LAB_GUIDE.md`, the `comfyui-deployment` skill, memories `qwen-image-2-1-deployment` and `gauntlet-film-pipeline`.
Every number marked **est.** is computed from tensor sizes or scaled from the 4-P40 run; nothing has been load-tested.

## 1. Problem

The seven-GPU AI server must become a permanently-on service: Strata (Qwen3.8-Flash-Next, qwen4exp A3B MoE) on
GPU0+GPU6+RAM as the backup brain for Sprite, one 3090 (GPU4) as an always-available media worker, and both must do
corpus-generation work whenever no user is waiting, dropping that work the instant a Sprite or Cyril request
arrives. The four P40s become a training lane and an experiments lane with GPU5 released. Today nothing runs on
GPU0/4/6, there are no systemd units, the Sprite stage points media at a stopped P40 container, and the Sprite
backend has no way to preempt anything, so all preemption has to live on the AI server.

## 2. Design

### 2.1 Component map

```
phone (Sprite)  --LAN-->  Sprite stage 192.168.50.84:23499  --LAN-->  AI server 192.168.50.222
                           (provider registry, delegations)            |
Cyril (curl/clients) ------------------------------------------------->|  strata-gate   :18400  (aiohttp, OpenAI-compatible, priorities)
                                                                       |     |__ llama-server 127.0.0.1:18401  GPU0+GPU6+RAM  (strata-llama.service)
                                                                       |
                                                                       |  media-gate    :28400  (aiohttp, ComfyUI-compatible, priorities, profile switch)
                                                                       |     |__ comfyui_qwen21_gpu4 :28334  profile "general" (image, edit, music, LTX, 3D if nodes load)
                                                                       |     |__ comfyui_h3_sol_gpu4 :28245  profile "h3sol"   (H3 Turbo draft + LTX refine)
                                                                       |
                                                                       |  gen-worker-text.service   -> strata-gate (background key)  -> intake drop zone
                                                                       |  gen-worker-media.service  -> media-gate  (background key)  -> intake drop zone
                                                                       |  lane-runner@train.service  GPU1,GPU2(,GPU3)   lane_job queue
                                                                       |  lane-runner@exp.service    GPU3,GPU5          lane_job queue
                                                                       |
                                                 PostgreSQL 25490:  ai_serve (new DB: clients, request_log, gen_task, lane_job)
                                                                    omni_lab (run registration via log-run), chimera_corpus (intake)
```

All services run as user `cereal` under systemd (`Restart=always`, journald logs, `ExecStartPre` health gates).
Credentials come from one env file `/home/cereal/.config/ai-serve/env` (mode 0600), generated from the vault
exactly like `~/.config/omni-lab/env`. The gateways bind the LAN ports listed and nothing else; the host firewall
restricts 18400/28400 to `192.168.50.0/24`. Nothing is exposed off-LAN.

Python runtime for everything new: a fresh venv `/home/cereal/.venvs/ai-serve` containing only `aiohttp`,
`asyncpg` and `psycopg2-binary` (intake compatibility). No FastAPI (it pulls pydantic), no ORM. `pip freeze` of
this venv is part of the acceptance evidence.

### 2.2 Strata serving: `strata-llama.service` (GPU0 + GPU6 + RAM)

**Model:** the IQ3_S already on disk (`/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf`,
sha `cb5958e5...`). Binary: `$OC/export/llamacpp/bin/llama-server` (b10752, sm_61/75/86, FA on). No rebuild.

**Placement (est.):** the 48 layers (55.2 GiB) + output (0.49 GiB) go on the two cards; the 26.8 GiB n-gram
table `per_layer_token_embd` and `token_embd` stay in host RAM (the llama.cpp default for that tensor class; the
running P40 job confirms it with a 30.9 GB RSS). A 16/32 layer split puts ~18.6 GiB on the 3090 and ~37.6 GiB on
the RTX 8000, leaving ~4.4 GiB and ~6.4 GiB for KV, SSM state and compute buffers.

**Unit `ExecStart` (one line in the unit; shown as a flag list):**

```
Environment=LD_LIBRARY_PATH=/home/cereal/omni-campaign/export/llamacpp/bin
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561,GPU-f5a09535-fc85-a027-50f7-734b344747dd
ExecStart=/home/cereal/omni-campaign/export/llamacpp/bin/llama-server
    -m /home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf
    --alias strata --host 127.0.0.1 --port 18401 --api-key-file /home/cereal/.config/ai-serve/llama_internal_key
    -ngl 999 -sm layer -ts 16,32 -fit off
    -np 6 -c 196608 -cb -fa on -ctk q8_0 -ctv q8_0
    -b 2048 -ub 512 -t 16 -tb 32
    --ctx-checkpoints 16 --cache-reuse 256 --cache-ram 16384
    --jinja --reasoning-format auto
    --slots --metrics --no-webui --prio 2 --numa distribute
TimeoutStartSec=1200
Restart=always
RestartSec=20
```

Flag rationale:
- `-ts 16,32 -fit off`: deterministic split. The first bring-up runs once with `-fit on` to read the proposal
  llama.cpp makes and the per-device buffer sizes from the startup log; the unit then pins what was observed.
- `-np 6 -c 196608`: six slots x 32k tokens. KV (12 full-attention layers, 24 KiB/token f16) at q8_0 is
  ~2.4 GiB for 196k tokens; SSM state 6 x 112 MiB = 0.67 GiB; compute buffers ~4 GiB at `-ub 512` (4.2 GiB
  measured on the P40 run). Total ~7 GiB of the ~10.8 GiB headroom. Verify `n_ctx_slot = 32768` in the log.
  Fallback if quantized KV misbehaves on the qwen4exp attention or indexer: `-ctk f16 -ctv f16 -np 4`.
- `-cb`: continuous batching, so a Sprite request joins the running decode batch instead of waiting.
- `--ctx-checkpoints 16`: 36 of 48 layers are recurrent, so prefix reuse only works from saved SSM checkpoints;
  this is what keeps multi-turn Sprite chats from re-prefilling from scratch every turn. `--cache-ram 16384`
  keeps evicted prompt caches in host RAM (16 GiB cap).
- `-b 2048 -ub 512`: the batch sizes the P40 run proved; a larger `-ub` costs compute-buffer VRAM we do not have.
- `-t 16 -tb 32`: the n-gram gather and sampling run on CPU; 44 physical cores, the rest stay for ComfyUI,
  intake and training. `--numa distribute` interleaves the 27 GiB table over both NUMA nodes (measure; drop it
  if tok/s is lower).
- `--jinja --reasoning-format auto`: tool calls and `reasoning_content` separation. Thinking on/off is decided
  per request by the gateway (`chat_template_kwargs.enable_thinking`), not here.
- `--prio 2`: above-normal scheduling; gen workers and training jobs run under `nice`.
- No `--sleep-idle-seconds`, no `--mlock`: the model must never unload (6-minute reload), and mmap + page cache
  is what already keeps the file hot.
- `--api-key-file`: an internal key (vault `strata_llama_internal_key`) so only the gateway can reach 18401.

**Throughput (est.; measured in plan step 6):** the 4-P40 layer-split run gave 15-26 tok/s per slot with 4 slots
busy and 170-250 tok/s prefill. The 3090 (936 GB/s) + RTX 8000 (672 GB/s) pair reads the ~1.5 GB of active
weights per token roughly 2-2.5x faster than four 346 GB/s P40s in series, so the expectation is 30-50 tok/s per
slot single-stream, 80-150 tok/s aggregate at 6 slots, 500-1,000 tok/s prefill. A Sprite turn with an 8k prompt
should see its first token in ~10-16 s.

**Measured 2026-10-21 18:2xZ (strata-llama.service, `-ts 16,32 -np 6 -c 196608`, q8_0 KV; the P40 teacher-score job
was running at the same time; evidence `/home/cereal/ai-serve/evidence/live_gate_check_20261021T1824Z.json`):**
`n_ctx_slot = 32768` confirmed; VRAM GPU0 22.4 GB / GPU6 40.7 GB; load 32 s with a hot page cache. Through the
gateway over the LAN: single-stream decode 44.3 tok/s (256 tokens), 6 concurrent slots 15.9 tok/s each = 88.5 tok/s
aggregate; prefill 124 tok/s on a 30-token prompt (long-prompt prefill and the 6-turn checkpoint-reuse test are
NOT measured yet). Preemption: 4 background streams answered 503 `preempted` 0.01 s after a Sprite request arrived,
the Sprite reply took 1.9 s, `/slots` showed 0 busy slots afterwards. `--prio 2` is refused for a non-root user, so
the unit uses `Nice=-5`; `--cache-reuse` is dropped (KV shifting does not apply to the recurrent layers).

**End-to-end run 2026-10-21 21:00-22:25Z (omni_lab runs 38044-38049; evidence
`/home/cereal/ai-serve/evidence/e2e-20261021/`; harness `ai-serve/e2e/e2e_serving.py`):**
- **Unit change.** The unit now runs with `--no-mmap --lazy-mode off`. This llama.cpp build reads tensors over 4 GiB
  (the per-layer n-gram embeddings, read on the CPU for every token) lazily from the mmapped GGUF.
  - Under file I/O those pages were evicted: 797,851 major faults in 3 h, and real Sprite turns of 5-10k prompt
    tokens took 26-77 s, decoding at 0.8-1.1 tok/s.
  - `--no-mmap` alone still left the lazy map in place.
  - With both flags the process holds 31 GB resident and takes 10 major faults.
  - Prefill went from 294/440 tok/s to 823/862/871 tok/s (2k/6k/9.6k-token prompts). Decode went from 37-40 to
    45-47 tok/s.
  - A clean restart is back in 152 s; after a SIGKILL it is back in 270 s.
- **Concurrency.** Measured through the gateway: 1 stream 47.9 tok/s; 4 streams 14.9 tok/s each (56.7 aggregate);
  6 streams 18.2 each (100.3 aggregate); 8 streams, 6 at 18.1 while 2 wait about 26 s for a slot (83.5 aggregate).
- **GPU6 under sustained load.** With 6 decode streams for 12 min, or 2 Sprite-shaped streams (6.8k-token prompts)
  for 8 min, GPU6 holds 81-82 C. Software thermal slowdown is active in 79 % of samples, at SM clock p50
  1522-1545 MHz, with no refusals; the gateway's 84 C level was never reached.
  - During the real Sprite traffic before the fix, the clock p50 was 405 MHz (slowdown in 97 % of samples).
  - Sprite-shaped turns are now 15.1 s to the first token (p50) and 20.7 s total.

**Q3 vs Q4 decision: IQ3_S now, IQ4_XS as a gated phase 2.**
- IQ3_S (3.44 bpw) is on disk, is the quant the Strata finetune actually published, and is the only size whose
  weights fit entirely on the two cards (no expert layers on CPU). It already produced 1,599 sealed tool-call
  conversations through llama-server, so it is a known quantity.
- IQ4_XS (90.8 GiB, ~64 GiB of GPU-resident layers) would need ~8 expert layers on CPU (`-ncmoe 8`). The expert
  FFN is tiny (width 640, 10 of 512 active), so the per-token CPU cost is small (est. ~200 MB of RAM reads per
  token, ~3 ms on this host) but batched prefill for those layers runs on AVX2 (no AVX-512), and the download
  needs ~95 GB on a root disk with 64 GB free. Also: the Q4 files on Hugging Face are a different repo
  (`0bserverx/...-Abliterated-Uncensored-GGUF`); the survey says same finetune, but identity must be proven from
  GGUF metadata (`general.source.*` / `general.base_model.*`) before a byte of it is served or trained on.
- Phase 2 trigger: disk cleanup finished AND an A/B on the held-out tool-call screens (`toolcall_heldout_screen.py`,
  `bfcl62_holdout.py`) shows IQ4_XS measurably better. Until then Q4 is not downloaded.
- Q4_K_M (111 GiB) is rejected: 15-17 expert layers on CPU plus AVX2 prefill makes "super responsive" unreachable.
- The unsloth 2.8 GB MTP draft model is an experiments-lane item (speculative decoding on top of IQ3_S), not part
  of the serving unit until measured.

**GPU6 thermal note:** the Quadro RTX 8000 is passively cooled and throttles under sustained load; its 250 W
limit stays. The gateway admission loop reads `nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active`
for GPU6 every 5 s: at >= 83 C it admits no new background requests; at >= 88 C it cancels running background
requests; at >= 92 C it returns 503 to everyone and logs a `thermal` event; it never kills `llama-server`.
Whatever thermal watchdog script already protects GPU6 is kept as-is (the survey did not locate it by name;
plan step 1 finds it). Two-thirds of every token of weight reads hit GPU6, so it is the card to watch during the
first 24 h soak.

### 2.3 `strata-gate.service` (priority gateway, LAN :18400)

A single-process `aiohttp` server (no FastAPI/pydantic). It exposes OpenAI-compatible `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, plus `GET /healthz` and `GET /gate/status`.

**Clients and priorities.** `Authorization: Bearer <key>`; keys are looked up in `ai_serve.api_client` (sha256 of
the key, never the key), loaded into memory at start and reloaded on SIGHUP. Three tiers:

| tier | priority | vault key | who |
|---|---|---|---|
| sprite | 0 (highest) | `strata_key_sprite` | the Sprite stage (`SPRITE_PROVIDER_STRATA_KEY`) |
| cyril | 1 | `strata_key_cyril` | clients Cyril runs himself, the omni_lab `gguf` engine, ad-hoc tooling |
| background | 2 | `strata_key_background` | `gen-worker-text` only |

**Served model ids:** `strata` (thinking off: the gateway injects `chat_template_kwargs: {"enable_thinking": false}`)
and `strata-think` (thinking on, `reasoning_content` returned). The Sprite provider entry names `strata`; the
phone cannot pass extra body fields, so the model id is how thinking is controlled.

**Admission rules (the whole point of the service):**
1. Slot budget: `N = 6` llama slots. Foreground (sprite + cyril) may use all 6. Background may use at most
   `N - 2 = 4`, and only when no foreground request has been seen for `idle_grace_s` = 20 s, GPU6 is below the
   thermal thresholds and host `MemAvailable` >= 40 GiB.
2. Preemption: when a foreground request arrives and no slot is free, the gateway immediately aborts enough
   background upstream connections (newest first) to free one. llama-server cancels a task when its client
   disconnects; background requests are therefore always sent upstream with `stream: true`, so the abort takes
   effect at the next token. The aborted background request gets HTTP 409
   `{"error": {"code": "preempted", "retry_after_s": N}}` and the gen worker requeues the task. The gateway
   confirms release via `GET /slots`; if the slot is still busy after 3 s it logs `preempt_stuck` (observed
   only; there is no other cancel API in llama-server).
3. Foreground never queues behind background, and sprite never queues behind cyril: the gateway orders its own
   wait queue by priority then arrival. With 2 slots always reserved for foreground, waiting should be rare.
4. Streaming: foreground requests pass `stream` through unchanged (Sprite sends `stream: false`; clients Cyril
   runs may stream). Response bodies for sprite are capped at 2 MB to match the Sprite backend cap.
5. Vision: a request carrying `image_url` parts is answered 400 `{"code": "no_vision"}` immediately, so the phone
   falls back to its on-device VL model instead of waiting for a text-only model to fail at depth.
6. Limits: `max_tokens` defaults to 4096 for sprite, 8192 for cyril, task-defined for background; per-request
   hard timeout 600 s (the Sprite provider `timeout_s`).

**Logging.** Every request writes one `ai_serve.request_log` row (client, tier, model, prompt/completion token
counts from the upstream `usage`, latency, first-token latency, slot id, outcome `ok|preempted|error|timeout|rejected`,
`preempted_count`). No prompt text is stored here (prompts for background tasks live in `gen_task`; Sprite
conversations belong to the phone). The gateway registers itself in omni_lab at start
(`log-run start --kind serve --name strata-gate --lane L-SERVE --gpus 0,6`) and posts hourly aggregate metrics
(`tok_s`, `requests`, `preemptions`) with `log-run metric`, always with `|| true` as the guide requires.

### 2.4 `media-gate.service` (ComfyUI priority gateway for GPU4, LAN :28400)

Same aiohttp pattern, speaking the ComfyUI HTTP API so the existing Sprite backend `ComfyClient` needs no change.

**Profiles (one GPU4 container each; both containers stay up, only one holds models):**

| profile | container | port | image | workflows |
|---|---|---|---|---|
| `general` (default, kept warm) | `comfyui_qwen21_gpu4` (exists, stopped) | 28334 | `claude-qwen-image-2.1-comfy037-torch210cu130-sm75sm86-20260920` | Qwen-Image 2.1 t2i + edit (`qwen21_t2i_v1`, `qwen21_edit_v1`), ACE-Step (`ace15_t2m_v1`), LTX 2.3 (`ltx23_t2v_v2`), Hunyuan3D 2.1 only if `/object_info` on this image lists the Hunyuan3d-2-1 nodes (host-mounted custom node; verified in plan step 2) |
| `h3sol` | `comfyui_h3_sol_gpu4` (new registry entry) | 28245 | `gauntlet-sol-sm86-v001-20260913` (sha `bdf75f0f...`) | H3 Turbo 4-step draft 896x512 + LTX 2.3 refine (`GauntletH3Sol`; SM80+ only, so GPU4 is the only card that can run it) |
| `hy3d` (only if `general` lacks the nodes) | `comfyui_burst_gpu4` (exists, disabled) | 28190 | its current image | Hunyuan3D 2.1 |

**TTS/speech: no profile.** No TTS model or node exists in any ComfyUI image on the server. It is a follow-up
(see Risks); it must not be promised to Sprite until a model is chosen and a container passes a canary.

**Routing.** The gateway maps each incoming `/prompt` graph to a profile by inspecting node `class_type`s
(`MiniMaxH3*` / `GauntletH3Sol` -> `h3sol`; `Hunyuan3D*` -> `hy3d` or `general`; everything else -> `general`).
The fixed Sprite graphs are known (`sprite_workflows.py`) and `h3_fl2va_v1` is `available=False` today, so every
Sprite job currently lands on `general`. Background tasks name their profile explicitly.

**Profile switch** (`general` -> `h3sol` or back): POST `/interrupt` and `DELETE /queue` for background items on
the vacated container, POST `/free` with `{"unload_models": true, "free_memory": true}` (its response body is
EMPTY: never JSON-parse it), poll `nvidia-smi` until GPU4 used < 1.5 GiB (timeout 60 s, else
`manage_comfy_instances.py restart <name> --ack-quiesced`), then forward. If the target container is not
running, start it through the manager (`up NAME --adopt`) and wait for `/system_stats`. H3 on the 3090 keeps the
proven `--reserve-vram 12`. `general` is the warm default; `h3sol` is unloaded again after 5 min idle. Each
container keeps its own `--user-directory`, pre-created under `/home/cereal/comfyui/user/` (otherwise supervisor
marks comfyui FATAL).

**Priority and preemption.** Tiers are identified by: source IP `192.168.50.84` (the Sprite stage) or
`extra_data.sprite_job` present -> `sprite`; header `X-Serve-Key` matching vault `media_key_cyril` -> `cyril`;
`media_key_background` -> `background`; anything else -> 403. The Sprite `ComfyClient` sends no headers, hence
the IP rule. On a foreground `/prompt`: `/interrupt` the running background prompt, delete pending background
prompt_ids from `/queue`, switch profile if needed, forward, and record the prompt_id as foreground. Background
tasks are admitted only when no foreground job has been seen for 60 s and the host RAM floor holds.

**Views Sprite depends on:** `GET /queue` is rewritten to show only foreground entries (Sprite treats any
`queue_running` / `queue_pending` as busy and would otherwise stall in `waiting_capacity`). `GET /system_stats`
and `GET /object_info` are served from a cache (refreshed every 60 s and immediately after a switch) so they
answer within the 2 s Sprite timeout even mid-swap; `/history`, `/view` and `/upload/image` pass straight through
to the container of the active profile. Upload filenames are rewritten to be content-addressed
(`<sha16>_<name>`), the lesson from the 2026-09-10 shared-slot collision.

**Logging:** one `request_log` row per `/prompt` (tier, profile, workflow id when `extra_data.sprite_job` is
present, duration, outcome), `service_event` rows for switches and preemptions, omni_lab registration like
strata-gate (`--lane L-MEDIA --gpus 4`).

### 2.5 Idle-time corpus generation: `gen-worker-text` and `gen-worker-media`

Both are small loops in the `ai-serve` venv that lease tasks from `ai_serve.gen_task` with raw SQL
(`FOR UPDATE SKIP LOCKED`), run them through the gateways with the background key, and hand finished outputs to
the intake. They never talk to llama-server or ComfyUI directly, so the gateway preemption rules are the only
scheduling rules.

**Text worker.** Keeps up to 4 concurrent background requests open (the gateway enforces the real cap).
Task kinds and their verification:

| kind | prompt source (never Claude/Codex outputs) | verifier before intake |
|---|---|---|
| `toolcall_conv` | `toolcall_selfgen.py` scenarios + tool schemas from `$OC/distill/tools` (Strata writes the tool calls, per the 2026-10-17 corpus ruling) | `toolcall_selfgen.grade_case` (executes the tool, pass/fail) |
| `instruction_code` | task templates with asserts (the `student_test` harness format) | run the asserts in a sandboxed subprocess; keep only passing samples, park failing ones under `_review` |
| `reasoning_math` | generated problems with a programmatic answer (exact arithmetic / sympy) | exact match of the final answer; `strata-think` output keeps `reasoning_content` as a separate field |
| `rewrite_pair` | public-domain or already-ingested canonical text records from `chimera_corpus` | length/language sanity only, flagged `verified=false` |

Each finished task writes one JSONL record to
`/mnt/ai_data/chimera-intake/text/strata-selfgen/<yyyy-mm-dd>/<task_kind>.jsonl` (spooled locally under
`/home/cereal/ai-serve/spool/` while the NAS is at 100%; see Risks) with provenance
`{model: "strata/IQ3_S", model_sha256, gateway_request_id, prompt_source, params: {temp, top_p, top_k, max_tokens, thinking}, verifier, verifier_result, synthetic: true}`.
The held-out screens (`toolcall_heldout_screen.py` for tool calls, `bfcl62_holdout.py`) run on every batch before
`intake.py` is invoked; a batch that overlaps a held-out item is moved to `_rejected` with the reason. Prompt seeds
come only from the sources in the table; the eval banks under `$OC/calibration/banks/` are not readable by the
worker process (enforced by file mode, not just policy).

**Media worker.** Task kinds: `image_t2i`, `image_edit`, `video_h3sol`, `video_ltx`, `music_ace`, `mesh_hy3d`.
Prompt sources: the caption/prompt pools already in the intake `text` lane and parameter sweeps over them. Each
output is paired with its full generation record (prompt, negative, seed, steps, cfg, resolution, profile,
container image sha, workflow graph sha) and dropped in the matching modality drop zone with `synthetic: true`,
so raw-modality training can weight or exclude synthetic data. Captions are the generation prompts; a VL
recaption pass is an experiments-lane job, not part of this worker.
*Built 2026-10-21/22 as `media/media_judge.py`:* an experiments-lane P40 job (`media-judge.timer` -> `lanes.py claim-run`). It
scores every spooled pair against a pre-registered spec (`media_judge_spec.json`), using Qwen3.6-27B VL on images, frame grids and
mesh renders, CLAP + ASR on audio, and mesh component and skirt metrics. Publish moves only pairs that pass. The VL caption is
kept in every verdict and in the published provenance (`judge.scores.caption`). Runbook sections 6 and 9 have the details.

**Requeue on preemption:** a 409 from a gateway resets the task to `queued` with `preemptions + 1` and a 2-minute
backoff; after 5 preemptions the task is `parked` and shows in `/gate/status`. A task that errors twice in a row
is `failed` with the error stored.

**Every batch is a run in omni_lab** (`log-run start --kind gen --lane L-SERVE|L-MEDIA`) with the intake batch id
in `--config`, so the lab can trace a training row back to the request that produced it.

### 2.6 P40 lanes: `lane-runner@train` (GPU1, GPU2, +GPU3 on demand) and `lane-runner@exp` (GPU3, GPU5)

One runner script, two systemd template instances, one `lane_job` table. A job row carries the lane, the command
(a pinned tool from `TOOL_PINS.json` plus args), the GPU count it needs and an optional `exclusive` flag. The runner:
1. Polls `lane_job` for its lane (priority, then created_at) and takes one with `FOR UPDATE SKIP LOCKED`.
2. Claims cards exactly as `omni_lab_run.py` does: `flock` on `$OC/distill/queues/gpu<N>.lock`, `nvidia-smi` busy
   check (< 2000 MiB), host `MemAvailable` >= 40 GiB, a `p40.claims` line, a LANES.json heartbeat.
3. Runs the command under `nice -n 10` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<uuids>`,
   logs to `/home/cereal/ai-serve/lane-logs/<job>.log`, registers `log-run start/end` in omni_lab.
4. GPU3 is flex: the train lane takes it when a queued training job declares `gpus_needed=3`; the exp lane takes it
   otherwise. The exp lane yields GPU3 only between jobs. Nothing on the P40s is ever preempted: training and
   experiments are checkpointed by their own tools, not killed by a scheduler.

The lane runner is deliberately boring: it does not replace the lock/claim/LANES conventions, it drives them, so
a hand-launched job and a queued job cannot collide. Until the training job for the v2 on-device model exists,
the train lane stays empty; the running teacher-score job finishes on its own (~19:30Z) and its claim plus the
dead `toolcall-extend` claim in `p40.claims` are cleaned in plan step 1.

**Allocator guard and LANES.json changes:**
- `/home/cereal/local-hermes/remote_runtime.py`: `CLAUDE_FORBIDDEN_GPU_INDICES` becomes `frozenset()`; a new
  `CLAUDE_ASK_FIRST_GPU_INDICES = frozenset(("0", "4", "6"))` makes the serving cards an explicit-approval set
  ("with my approval we can pull them"). The workstation copy under
  `C:/Users/cereal/Documents/claude_codex communication/local-hermes/` is brought back to the same sha so the
  drift the survey found stops.
- `omni_lab_run.py`: `--allow-gpu5` becomes a kept-for-compatibility no-op; a new `--allow-serving-gpus` gates
  0/4/6 the way GPU5 used to be gated. `tests/test_runner.py` updated.
- `$OC/distill/queues/LANES.json`: new revision with lanes `L-SERVE {0,6}`, `L-MEDIA {4}`, `L-TRAIN {1,2}`,
  `L-FLEX {3}`, `L-EXP {5}`; the L-RESEARCH `do_not_touch` text loses "GPU5 (never)" and gains "GPU0/4/6 (serving;
  ask Cyril)". Stale heartbeats stay (they are history); the `lanes` ingester re-runs so the omni_lab `lane`
  table matches.
- `/home/cereal/comfyui/comfy-gpu-owners.json`: GPU4 owner `ai-serve`; `set-owners comfyui_qwen21_gpu4=ai-serve
  comfyui_h3_sol_gpu4=ai-serve`; a GPU5 entry (`experiments`, P40 cu126 image only).

### 2.7 Sprite integration

**Provider entry** (stage file `D:/SpriteBuild/stage/sprite_providers.json`, hot-reloaded on mtime, no restart):

```
{"alias": "aiserver.strata", "label": "Strata (AI server)", "kind": "openai_compat",
 "base_url": "http://192.168.50.222:18400/v1", "model": "strata",
 "api_key_env": "SPRITE_PROVIDER_STRATA_KEY", "capabilities": ["llm.chat"],
 "egress": "lan", "enabled": true, "timeout_s": 600, "concurrency": 4, "cost": {}}
```
plus: `aiserver.comfy.image`, `aiserver.comfy.music` and `aiserver.comfy.video.ltx` repointed from `:28335` to
`http://192.168.50.222:28400`; `aiserver.comfy.video.h3` stays disabled until `h3_fl2va_v1` is ported;
`lan.lmstudio` (`:1234`, dead) removed.

**Key:** `start-stage.ps1` exports `SPRITE_PROVIDER_STRATA_KEY` from `credman.py get strata_key_sprite` before
uvicorn starts. This is the only step that needs a stage restart (`-Restart`), which interrupts the phone: take
the `stage23499` lease and post on Relay topics `db1fe673-e64b-4a44-bcf8-4401abebbbdb` and
`52468f63-fbb0-429e-a316-1d17607d3912` first.

**Routing rule (on-device first, big brain when needed or wanted):**
- wanted: the user picks engine `server:aiserver.strata` in the app (it appears automatically, labelled LAN).
- needed: the on-device agent `delegate.llm` tool chooses provider `aiserver.strata` for a task it judges beyond
  the phone (long context, multi-step tool use, code). This path exists and runs without an approval prompt
  (`llm.chat` over `lan` defaults to allow).
- automatic fallback (on-device OOM/timeout/context overflow -> retry on `aiserver.strata`): not in the phone
  today. Optional phone change on a new branch from `sprite/claude-20261019`, coordinated with the Sprite-lead
  session via Relay; not required for the brain to be usable.
- media: `delegate.image/video/music` go to the media gate unchanged; policy stays "ask" for GPU jobs.

**Network and auth path:** phone -> stage (LAN, authenticated Sprite session) -> gateway (LAN, Bearer key from the
vault). The phone never holds the Strata key. Off-LAN use would mean deploying the Sprite backend to production
`:23456` plus the portal; not in this design, needs a go-ahead from Cyril.

**Repo changes (optional, new branch `sprite/claude-strata-provider-20261021` from `sprite/claude-20261019`,
pushed to GitHub origin only, as autonomouscereal):** add the disabled `aiserver.strata` template and fix the
stale ComfyUI ports in `backend/app/sprite_providers.default.json`; port `h3_fl2va_v1` in `sprite_workflows.py`
for the `h3sol` profile. TTS/3D capabilities are a larger backend+phone change and wait for a real TTS profile.

## 3. Schema (new database `ai_serve`, PostgreSQL 16 on `192.168.50.222:25490`, raw SQL only)

Roles: `ai_serve` (owner, vault key `ai_serve_pg`) and `ai_serve_ro` (vault key `ai_serve_ro_pg`). Migrations are
numbered files applied by a ~40-line `ai_serve_db.py migrate` (psycopg2, forward-only, sha recorded, the same
pattern as omni_lab; the pattern is copied, the omni_lab code is not imported).

```sql
-- 0001_core.sql
CREATE TABLE schema_migration (
  version     integer PRIMARY KEY,
  filename    text NOT NULL,
  sha256      text NOT NULL,
  applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_client (
  id          bigserial PRIMARY KEY,
  name        text NOT NULL UNIQUE,                 -- 'sprite', 'cyril', 'background-text', ...
  service     text NOT NULL CHECK (service IN ('strata','media')),
  tier        text NOT NULL CHECK (tier IN ('sprite','cyril','background')),
  key_sha256  text NOT NULL UNIQUE,                 -- never the key itself
  vault_key   text NOT NULL,                        -- e.g. 'strata_key_sprite'
  enabled     boolean NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE request_log (
  id                bigserial PRIMARY KEY,
  service           text NOT NULL CHECK (service IN ('strata','media')),
  client_id         bigint REFERENCES api_client(id),
  tier              text NOT NULL,
  request_id        uuid NOT NULL UNIQUE,
  model_or_profile  text NOT NULL,                  -- strata | strata-think | general | h3sol | hy3d
  workflow_id       text,                           -- Sprite workflow id or background task kind
  prompt_tokens     integer,
  completion_tokens integer,
  slot_id           integer,
  received_at       timestamptz NOT NULL DEFAULT now(),
  first_token_at    timestamptz,
  finished_at       timestamptz,
  outcome           text NOT NULL CHECK (outcome IN ('ok','preempted','error','timeout','rejected')),
  preempted_count   integer NOT NULL DEFAULT 0,     -- background jobs this request displaced
  error_text        text,
  gen_task_id       bigint                          -- set for background requests
);
CREATE INDEX request_log_received_idx ON request_log (service, received_at DESC);
CREATE INDEX request_log_tier_idx     ON request_log (tier, received_at DESC);

CREATE TABLE service_event (
  id      bigserial PRIMARY KEY,
  service text NOT NULL,
  kind    text NOT NULL,                            -- start, stop, preempt, preempt_stuck, thermal, profile_switch, ram_floor, error
  detail  jsonb NOT NULL DEFAULT '{}'::jsonb,
  at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX service_event_at_idx ON service_event (service, at DESC);

-- 0002_gen_queue.sql
CREATE TABLE gen_task (
  id           bigserial PRIMARY KEY,
  service      text NOT NULL CHECK (service IN ('strata','media')),
  kind         text NOT NULL,                       -- toolcall_conv | instruction_code | reasoning_math | rewrite_pair | image_t2i | image_edit | video_h3sol | video_ltx | music_ace | mesh_hy3d
  priority     smallint NOT NULL DEFAULT 100,       -- lower first
  spec         jsonb NOT NULL,                      -- prompt(s) or graph, params, prompt_source, seed
  spec_sha256  text NOT NULL UNIQUE,                -- idempotent enqueue
  status       text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','leased','done','failed','parked','rejected')),
  attempts     integer NOT NULL DEFAULT 0,
  preemptions  integer NOT NULL DEFAULT 0,
  leased_by    text,
  lease_until  timestamptz,
  not_before   timestamptz NOT NULL DEFAULT now(),
  created_at   timestamptz NOT NULL DEFAULT now(),
  finished_at  timestamptz,
  last_error   text
);
CREATE INDEX gen_task_pick_idx ON gen_task (service, status, priority, not_before, id) WHERE status = 'queued';

CREATE TABLE gen_output (
  id              bigserial PRIMARY KEY,
  task_id         bigint NOT NULL REFERENCES gen_task(id),
  request_id      uuid REFERENCES request_log(request_id),
  output_path     text NOT NULL,                    -- spool or drop-zone path
  output_sha256   text NOT NULL,
  provenance      jsonb NOT NULL,                   -- model, model_sha256, params, prompt_source, verifier, synthetic: true
  verifier        text,
  verifier_result jsonb,
  screen_result   text CHECK (screen_result IN ('pass','heldout_overlap','not_run')),
  intake_batch_id text,                             -- filled when intake.py accepts it
  omni_lab_run_id bigint,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX gen_output_task_idx ON gen_output (task_id);

-- 0003_lanes.sql
CREATE TABLE lane_job (
  id              bigserial PRIMARY KEY,
  lane            text NOT NULL CHECK (lane IN ('train','exp')),
  name            text NOT NULL,
  priority        smallint NOT NULL DEFAULT 100,
  gpus_needed     smallint NOT NULL CHECK (gpus_needed BETWEEN 1 AND 3),
  exclusive       boolean NOT NULL DEFAULT false,
  tool            text NOT NULL,                    -- pinned tool name from TOOL_PINS.json
  argv            jsonb NOT NULL,
  env             jsonb NOT NULL DEFAULT '{}'::jsonb, -- non-secret only
  status          text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','done','failed','cancelled')),
  gpus_used       text[],                           -- UUIDs actually claimed
  pid             integer,
  omni_lab_run_id bigint,
  log_path        text,
  exit_code       integer,
  created_at      timestamptz NOT NULL DEFAULT now(),
  started_at      timestamptz,
  finished_at     timestamptz
);
CREATE INDEX lane_job_pick_idx ON lane_job (lane, status, priority, id) WHERE status = 'queued';

-- 0004_grants.sql
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ai_serve_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE ai_serve IN SCHEMA public GRANT SELECT ON TABLES TO ai_serve_ro;
```

Why a separate database instead of tables inside `omni_lab`: the omni_lab migrations are sha-locked and owned by
the lab; a serving queue that mutates thousands of rows an hour does not belong under the same owner. Runs still
register in omni_lab through `log-run`, which is the contract the guide defines.

## 4. Trade-offs

| Choice | Rejected alternative | Why |
|---|---|---|
| llama.cpp `llama-server` with layer split on GPU0+GPU6 | vLLM / SGLang / TensorRT-LLM | qwen4exp with an n-gram table and SSM layers is a llama.cpp-first architecture; the GGUF, the build (sm_75+sm_86, FA) and a prior successful serve all exist. Other engines would need a new container, a new quant and sm_75 support. Boring wins. |
| IQ3_S now, IQ4_XS gated | download Q4 now | Disk (64 GB free), unproven identity of the Q4 repo, and full-GPU residency only works at IQ3_S. |
| All layers on GPU, n-gram table in RAM | `-ncmoe` expert offload to free VRAM for more slots | Expert offload is cheap per token here (width-640 experts) but prefill on AVX2 is not; 6 x 32k slots fit without it. Revisit for IQ4_XS. |
| Preemption by closing the upstream stream | a patched llama-server with a cancel endpoint | Client-disconnect cancellation is upstream behaviour; a fork would need maintaining. Two reserved foreground slots make the cancel latency irrelevant for responsiveness. |
| Background requests always `stream: true` upstream | non-streaming background | Non-streaming cancellation on disconnect is not guaranteed to take effect before the response completes; streaming makes the abort observable at the next token. |
| Separate `strata-gate` process in front of llama-server | the llama-server `--api-key` list + `-np` and nothing else | llama-server has no priorities, no preemption and no per-client logging. The gateway is ~600 lines of aiohttp; no framework. |
| aiohttp + asyncpg in a clean venv | FastAPI | FastAPI depends on pydantic, which is forbidden; the two existing envs that carry pydantic are neither touched nor reused. |
| Media gateway speaks the ComfyUI API | a new media API plus a Sprite backend change | The Sprite `ComfyClient`, its health probes and job lookup already work against `/prompt`, `/queue`, `/history`; the gateway only filters `/queue` and caches `/system_stats` and `/object_info`. Zero Sprite code. |
| Two GPU4 containers kept up, `/free` on switch | stop/start containers per profile | Cold container start plus model load is 2-4 min; `/free` plus reload is 1-2 min and keeps supervisor state; the gateway falls back to a manager restart only if VRAM is not released. |
| `general` profile warm by default | H3 warm by default | Image, edit and music are the Sprite capabilities that exist; H3 is `available=False` in Sprite today. |
| No TTS profile | shipping a TTS container now | Nothing exists on the server; choosing a model is a decision for Cyril, not a side effect of this design. |
| Lane runner drives the existing locks/claims/LANES.json | a new scheduler that replaces them | Hand-launched jobs (Codex, Claude, Cyril) use the locks; replacing them re-creates the GPU collisions the locks were built to stop. |
| Training and experiment jobs are never preempted | preempt P40 jobs for serving spikes | Serving never touches the P40s; killing a training step mid-checkpoint loses work for no responsiveness gain. |
| New `ai_serve` database | tables in `omni_lab` | Ownership, migration locking and write rate differ; omni_lab keeps its role as the record of runs via `log-run`. |
| Local spool for generation outputs | writing straight to the NAS drop zones | `/mnt/ai_data` has 2.5 GB free; spool-then-move avoids losing output during the parallel cleanup. |
| GPU5 released; GPU0/4/6 become ask-first | keep "GPU5 never" | The instruction releases GPU5 and makes the three fast cards the ones that need approval to pull. |

## 5. Adversarial critique and revisions

What I attacked in the first draft and what changed:

1. **Cancel-on-disconnect is a behaviour, not an API.** If a future llama.cpp build stops honouring it, background
   work would hog slots. Revision: two slots are always reserved for foreground regardless of cancellation, the
   gateway verifies release via `/slots` and logs `preempt_stuck`, and the soak (plan step 6) measures cancel
   latency. If it exceeds 2 s, background concurrency drops to 2 until the build is investigated.
2. **Recurrent layers break prompt-cache reuse.** A Sprite conversation re-prefilled every turn would wait
   10-16 s for its first token at 8k context. Revision: `--ctx-checkpoints 16` + `--cache-ram`, and the soak
   measures a 6-turn conversation; if checkpoints do not restore, the fallback is slot pinning per conversation
   (`id_slot` from a hash the gateway derives from the first user message) so the slot state itself is reused.
3. **q8_0 KV on a hybrid SSM/attention model** might not be supported for the attention layers or the indexer.
   Revision: the f16 fallback (`-np 4`) is written into the plan, not left to discovery at 02:00.
4. **GPU6 carries two-thirds of the weights and is passively cooled.** Under a 6-slot decode storm it will warm
   up. Revision: thermal admission in the gateway (2.2) and a 24 h soak with temperature logged every 5 s before
   Sprite is pointed at it. If GPU6 throttles, the split moves toward GPU0 (`-ts 18,30`) at the cost of KV
   headroom on the 3090, and `-np` drops to 5.
5. **Host RAM.** Strata ~31 GB + two ComfyUI containers (20-65 GB each, the ceiling the gauntlet memory records)
   + a training job + page cache for an 83 GB model file. Revision: a 40 GiB `MemAvailable` floor is enforced in
   both gateways and the lane runner (the number omni_lab already uses), and the H3 container is unloaded after
   5 min idle. The 251 GiB host holds this; it would not hold a third ComfyUI container, which the design forbids
   on GPU4.
6. **Startup is 6 minutes and the root disk is 97% full.** A restart during a Sprite session is a 6-minute
   outage. Revision: `TimeoutStartSec=1200`, no idle unload, `Restart=always` with `RestartSec=20`, and the Sprite
   provider health check (2 s on `/models`) makes the phone show the engine unavailable instead of hanging;
   `/v1/models` is served by the gateway only while the llama-server `/health` is `ok`.
7. **IP-based Sprite detection in the media gateway** means anything on the stage host is "sprite". Revision:
   accepted for LAN-only (the stage host is the PC Cyril uses), plus `extra_data.sprite_job` as a second signal
   and `X-Serve-Key` for everything else. Off-LAN exposure is explicitly out of scope.
8. **Held-out leakage.** A generation worker that can read the eval banks could, through prompt seeding,
   reproduce held-out items. Revision: prompt sources are enumerated, the bank directory is unreadable to the
   worker (separate Unix user `ai-serve-gen` with no read on `$OC/calibration/banks/`), and the screens run on
   every batch before `intake.py`.
9. **Synthetic media in the raw-modality corpus** could dominate the real data. Revision: `synthetic: true` in
   every provenance record and a `dataset_version` tag in omni_lab so training recipes can weight or exclude it.
10. **The manager owner policy** reserves GPU4 for `neural-forge` and will refuse `ai-serve` containers.
    Revision: `set-owners` and the owners-file edit are an explicit plan step with the registry diff kept.
11. **Q4 identity.** The same-finetune claim for the 0bserverx repo rests on a name match. Revision: phase 2
    requires the source model id/hash in the GGUF metadata to match the Strata file before any download.
12. **One gateway is a single point of failure for the big brain.** Accepted: it is one process under systemd
    with `Restart=always`; the phone keeps working on-device when it is down, which is the stated primary mode.
13. **Sprite `GLOBAL_CAP = 4` and per-provider `concurrency` 1..4** mean at most 4 Sprite requests at once; with
    6 slots and 2 reserved, Sprite alone can never exhaust the server. Confirmed consistent; no change.

## 6. Plan (ordered; every step leaves evidence under `/home/cereal/ai-serve/evidence/`)

Prerequisites: the teacher-score job (pid 1183917) has exited (~19:30Z) or Cyril says to stop it; the disk
cleanup workflow has not pushed root free space below 20 GB (nothing here downloads anything; the new venv is
under 200 MB).

1. **Residue and locks** (server, no services yet): locate the existing GPU6 thermal watchdog
   (`grep -rl temperature.gpu /home/cereal/*.sh /home/cereal/omni-campaign/distill/tools`), confirm pid 1183917
   is gone, remove the stale `toolcall-extend` (pid 1912373) line and the finished `flash-score` claim from
   `$OC/distill/queues/p40.claims`, verify `gpu3.lock` is free. Evidence: before/after of `p40.claims`.
2. **ComfyUI node inventory on GPU4 (canary):** `manage_comfy_instances.py up comfyui_qwen21_gpu4 --adopt`,
   `GET :28334/object_info`, record whether `Hunyuan3D*`, ACE-Step and LTX node classes are present, then
   `/free` and `stop --ack-quiesced`. Decides whether `hy3d` is a separate profile.
3. **Database:** as the `agent_memory` admin: `CREATE ROLE ai_serve LOGIN`, `CREATE ROLE ai_serve_ro LOGIN`,
   `CREATE DATABASE ai_serve OWNER ai_serve`; store both passwords in the vault (`ai_serve_pg`, `ai_serve_ro_pg`)
   before they are used anywhere; apply migrations 0001-0004 from
   `C:/Users/cereal/Documents/ai training/ai-serve/migrations/` (new local repo, mirrored to
   `/home/cereal/ai-serve/` with a sha check like omni_lab).
4. **Keys and env file:** `credman.py set` for `strata_key_sprite`, `strata_key_cyril`, `strata_key_background`,
   `strata_llama_internal_key`, `media_key_cyril`, `media_key_background` (random 32-byte hex each); generate
   `/home/cereal/.config/ai-serve/env` (0600) and `/home/cereal/.config/ai-serve/llama_internal_key` (0600) from
   the vault with the generator pattern used for `~/.config/omni-lab/env`; insert `api_client` rows with the
   sha256 of each key. Evidence: `ls -l` of the config dir, `SELECT name, tier, enabled FROM api_client`.
5. **Venv:** `python3 -m venv /home/cereal/.venvs/ai-serve && pip install aiohttp asyncpg psycopg2-binary`;
   `pip freeze` saved as evidence and grepped for `pydantic|sqlalchemy|sqlmodel` (must be empty).
6. **Strata bring-up and measurement (GPU0+GPU6):** run the `ExecStart` line by hand once with `-fit on` in a
   detached `screen`, capture the split and buffer sizes from the log, then with the pinned `-ts 16,32 -fit off`.
   Measure with scripts in `/home/cereal/ai-serve/bench/`: single-stream tok/s, 6-slot aggregate, prefill tok/s
   at 2k/8k/32k, a 6-turn conversation (checkpoint reuse), q8_0 vs f16 KV, GPU6 temperature every 5 s, cancel
   latency on client disconnect. Record in omni_lab (`log-run --kind smoke`). Write the numbers into 2.2 of this
   document in place of the **est.** values.
7. **`strata-llama.service`:** install `/etc/systemd/system/strata-llama.service` (sudo; flagged), enable, start,
   confirm `/health` and the per-slot context in the journal.
8. **`strata-gate.service`:** code at `/home/cereal/ai-serve/strata_gate.py` (aiohttp; `/v1/models`,
   `/v1/chat/completions`, `/v1/completions`, `/healthz`, `/gate/status`); unit with `After=strata-llama.service`.
   Tests in `tests/test_strata_gate.py` against a fake upstream (priority ordering, preemption, 409 shape, vision
   rejection, thermal gate). Live test: a cyril-key request while 4 background streams run; preemption latency
   read from `request_log`.
9. **Firewall:** allow 18400 and 28400 from `192.168.50.0/24` only (sudo; flagged). Evidence: `ufw status` or
   `iptables -S`.
10. **Media profiles:** `set-owners` + `comfy-gpu-owners.json` edit (diff kept); register `comfyui_h3_sol_gpu4`
    on port 28245 with the sol image and `--reserve-vram 12`, user dir pre-created; canary each profile with the
    `comfyui-validation` scripts (one image, one 5 s H3 clip, one 10 s ACE clip, one mesh if `hy3d`), confirming
    nonempty artifacts. Leave `general` running, `h3sol` stopped.
11. **`media-gate.service`:** code at `/home/cereal/ai-serve/media_gate.py`; tests for queue filtering, profile
    detection, the switch sequence (`/free` empty body) and preemption; live test: a background H3 task
    interrupted by a `qwen21_t2i_v1` prompt carrying `extra_data.sprite_job`.
12. **Generation workers:** `/home/cereal/ai-serve/gen_worker_text.py`, `gen_worker_media.py`, enqueue scripts
    per kind, spool dir, the held-out screen call, the `intake.py` invocation with provenance. Run one small batch
    of each kind end to end and show the `gen_output -> intake_batch_id -> omni_lab run` chain. Create Unix user
    `ai-serve-gen` (sudo; flagged) with no read access to `$OC/calibration/banks/`.
13. **Lane runner:** `/home/cereal/ai-serve/lane_runner.py` + `lane-runner@.service`; LANES.json revision;
    `remote_runtime.py` guard edit on both the server and workstation copies (same sha afterwards);
    `omni_lab_run.py` `--allow-serving-gpus` + test update; re-run `omni_lab.py ingest lanes`. Enqueue one no-op
    experiment job on GPU5 to prove the claim/lock/log-run path.
14. **Sprite stage:** edit `D:/SpriteBuild/stage/sprite_providers.json` (hot reload): add `aiserver.strata`,
    repoint media to `:28400`, remove `lan.lmstudio`; edit `start-stage.ps1` to export
    `SPRITE_PROVIDER_STRATA_KEY` from the vault; post the Relay note; take the `stage23499` lease; restart the
    stage; confirm the phone lists `server:aiserver.strata` and a `delegate.llm` round trip appears in
    `request_log` with `tier = sprite`.
15. **Optional repo branch** `sprite/claude-strata-provider-20261021` from `sprite/claude-20261019`: default
    provider template + port fix; push to GitHub origin as autonomouscereal. Coordinate via Relay; never touch
    the branch of the Sprite lead.
16. **24 h soak:** gateways + both workers running, GPU6 temperature and `request_log` outcomes reviewed; only
    after this does background concurrency go from 2 to 4.

## 7. Risks and gates

| Risk | What it touches | Gate / mitigation |
|---|---|---|
| GPU6 thermal throttling or shutdown under sustained decode | running service, hardware | 24 h soak before the Sprite cutover; thermal admission in the gateway; the existing watchdog stays; the GPU6 power limit is unchanged. **Approval from Cyril** before any power-limit change. |
| 6-minute model reload on every restart | Sprite availability | systemd `Restart=always`; the phone shows the engine unavailable via the 2 s health check and keeps running on-device. |
| Root disk at 97% | everything; the venv needs ~200 MB, logs grow | No downloads in this design; journald `SystemMaxUse=2G` for the new units; Q4 waits for the cleanup workflow. |
| NAS `/mnt/ai_data` at 100% | intake drop zones | Local spool under `/home/cereal/ai-serve/spool/` with a mover that refuses to run while NAS free < 20 GB. **The NAS cleanup must land** before intake can accept generated data. |
| Stale `p40.claims` and registry entries (47 ComfyUI entries, dead claims) | lane correctness, manager preflight | Step 1 cleans claims; the registry is touched only for the two GPU4 entries (a full registry cleanup is separate work). |
| Manager owner policy refuses `ai-serve` on GPU4 | ComfyUI bring-up | `set-owners` + owners-file edit, diff kept. **Flag: changes a production policy file.** |
| H3 SOL node is SM80+ only | media profile placement | Only GPU4 hosts `h3sol`; never schedule it on GPU6 or a P40. |
| Preempting a 2-minute H3 draft wastes GPU time | background media throughput | Accepted; parked after 5 preemptions; background media admitted only after 60 s of foreground silence. |
| Sprite `ComfyClient` treats any queue entry as busy | Sprite media UX | `/queue` filtered to foreground entries; cached `/system_stats` and `/object_info`. |
| Image parts sent to a text-only model | Sprite UX | 400 `no_vision` from the gateway; the phone VL model stays on-device. |
| Key handling | credentials | Keys only in the vault and in the 0600 env file; `api_client` stores sha256; `start-stage.ps1` reads the vault at start. No key in any repo or document. |
| Held-out contamination by generated data | corpus integrity | Enumerated prompt sources, bank directory unreadable by the gen user, screens before intake, `synthetic: true` provenance. |
| Teacher-score job still running when step 6 starts | P40 lane, host RAM | Step 6 does not need the P40s; the lane runner is not started until pid 1183917 is gone. **Do not kill it.** |
| Stage restart interrupts the phone | the work of the Sprite lead | Relay note + `stage23499` lease before `-Restart`; a single restart, only for the key export. |
| Off-LAN access to the brain | security | Not in scope; firewall limits 18400/28400 to the LAN; **approval from Cyril** required to go further. |
| sudo steps (systemd units, firewall, new Unix user) | server configuration | Flagged for the caller; everything else runs as `cereal`. |
| A later llama.cpp rebuild changes cancel-on-disconnect or flag names | gateway correctness | The build is pinned (b10752); any rebuild re-runs the preemption test from step 8. |
| TTS requested by Cyril but absent | media scope | Needs a model decision (candidates: a ComfyUI TTS node pack or a standalone OpenAI-compatible TTS server behind the media gate) and a canary; a follow-up design, not a silent gap. |

Files this design creates (all new; none exist yet):
- `C:/Users/cereal/Documents/ai training/ai-serve/` with `strata_gate.py`, `media_gate.py`, `gen_worker_text.py`,
  `gen_worker_media.py`, `lane_runner.py`, `ai_serve_db.py`, `migrations/0001-0004`, `units/*.service`, `tests/`;
  mirrored to `/home/cereal/ai-serve/`
- `/home/cereal/.config/ai-serve/env` and `/home/cereal/.config/ai-serve/llama_internal_key` (0600, generated from the vault)
- `/home/cereal/.venvs/ai-serve/`

Files this design edits (each flagged for the caller):
- `/home/cereal/local-hermes/remote_runtime.py` and `C:/Users/cereal/Documents/claude_codex communication/local-hermes/remote_runtime.py` (guard)
- `C:/Users/cereal/Documents/ai training/omni-lab/omni_lab_run.py` and `tests/test_runner.py` (GPU5 -> serving-GPU gate), server mirror `/home/cereal/omni-campaign/lab/`
- `/home/cereal/omni-campaign/distill/queues/LANES.json` and `p40.claims`
- `/home/cereal/comfyui/comfy-instances.json` (two GPU4 entries, via the manager) and `/home/cereal/comfyui/comfy-gpu-owners.json`
- `D:/SpriteBuild/stage/sprite_providers.json` and `D:/SpriteBuild/stage/start-stage.ps1`

Touches production data, credentials or running services (gate each): the new PostgreSQL roles/database on
`agent-memory-postgres`; six new vault keys; the owner policy and registry of the ComfyUI manager; systemd units
and firewall rules (sudo); the Sprite stage restart; the allocator guard that every Claude/Codex GPU launch passes
through; LANES.json. Nothing in `agent_memory`, `chimera_corpus` or `omni_lab` schemas is altered.
