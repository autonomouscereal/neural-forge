# AI Server Runbook: Strata brain, media worker, idle-time generation, P40 lanes

Date: 2026-10-21. State is from 22:26-22:35Z live checks on the AI server (192.168.50.222) unless a line says otherwise.
Design: `AI_SERVER_SERVING_DESIGN_2026-10-21.md` (same folder). Code: `C:/Users/cereal/Documents/ai training/ai-serve/`,
mirrored to `/home/cereal/ai-serve/` (hash-matched at deploy). Server copy of this file:
`/home/cereal/ai-serve/AI_SERVER_RUNBOOK_2026-10-21.md`.
**Hardening review, 2026-10-22 01:15Z:** section 10 lists the open must-fix defects from the six lane reviews and
corrects statements in sections 3, 4 and 9 that the reviews proved wrong. Read it before relying on those sections.
**Status 2026-10-22 06:20Z:** section 10 items 1-4 and 6 are fixed and reviewed, item 5 (pydantic) waits on Cyril;
GPU6 is capped at 200 W with `--bg-max 4`; the Q4 swap is cancelled (section 10 table and "Also done").

Cyril's instruction (2026-10-21), in short: the 3090 + RTX 8000 + system RAM run Strata with batching, as Sprite's
backup brain and Cyril's own model, generating corpus data when idle and dropping that work at once when a user
asks. The other 3090 runs ComfyUI for every media type, with the same preemption rule. The P40s train (2-3) and
run experiments (1-2). The three fast cards stay on Sprite + generation by default; pulling them needs his approval.

## 1. GPU map

| GPU | Card | Lane (LANES.json) | What runs | Live 22:26Z |
|---|---|---|---|---|
| 0 | RTX 3090 (`GPU-4ead7f3a`) | L-SERVE, ask-first | Strata llama-server, layers 1-16 | 22.4 GB used, 52 C |
| 6 | Quadro RTX 8000 46 GB (`GPU-f5a09535`), passive cooling | L-SERVE, ask-first | Strata llama-server, layers 17-48 | 40.7 GB used, 72 C |
| host RAM | 2 NUMA nodes, 44 cores | (L-SERVE) | Strata's 26.8 GiB per-layer n-gram table + token embeddings (~31 GB resident) | MemAvailable 127 GiB |
| 4 | RTX 3090 (`GPU-2ce6a95c`) | L-MEDIA, ask-first | ComfyUI `general` profile (`comfyui_qwen21_gpu4`), warm | 20.6 GB used |
| 1, 2 | Tesla P40 | L-TRAIN | lane-runner@train; queue empty (the v2 on-device training job is still being designed) | idle, 5 MiB |
| 3 | Tesla P40 | flex (L-TRAIN or L-EXP) | arch-v2 proxy r4 probe (L-EXP claim 21:47Z) | 7.9 GB, busy |
| 5 | Tesla P40 | L-EXP (no longer reserved for ICE) | arch-v2 proxy r4 probe (L-EXP claim 21:47Z) | 8.5 GB, busy |

Card rules live in LANES.json (`python3 $OC/distill/tools/lanes.py cards`), where `$OC = /home/cereal/omni-campaign`.
The allocator guard in `/home/cereal/local-hermes/remote_runtime.py` now has `CLAUDE_FORBIDDEN_GPU_INDICES=frozenset()`, so GPU5 is no longer blocked.

## 2. Services (all systemd, user `cereal`, `Restart=always`, logs in journald)

| Unit | Listens | Does | Key settings |
|---|---|---|---|
| `strata-llama.service` | 127.0.0.1:18401 (internal key) | llama.cpp `llama-server` b10752, `RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf` (sha `cb5958e5...`, size and sha checked at every start) | `-ts 16,32 -np 6 -c 196608` (6 slots x 32k), `-cb`, q8_0 KV, FA on, `--ctx-checkpoints 16 --cache-ram 16384`, `--no-mmap --lazy-mode off`, `Nice=-5`. Takes `gpu0.lock` + `gpu6.lock` and writes CLAIM/RELEASE lines in `$OC/distill/queues/p40.claims`. Launch script: `strata/strata_llama.sh` |
| `strata-gate.service` | LAN :18400 | OpenAI-compatible priority gateway (aiohttp + asyncpg) | `--bg-max 4` since 2026-10-22 05:44Z with GPU6 capped at 200 W (omni_lab 38195; was 2 until the soak review). An iptables rule (`lan_only.sh`) drops :18400 from outside 192.168.50.0/24 |
| `gen-worker-text.service` | none | Idle-time tool-call corpus generation through strata-gate (background key) | `Nice=10`, idle I/O class |
| `media-gate.service` | LAN :28400 | ComfyUI-compatible priority gateway for GPU4; switches profiles | config `media/media_gate.json`; only accepts LAN and loopback |
| `media-gate-health.timer` | none | Probes `/healthz` every 2 min; restarts media-gate after 3 failures in a row | |
| `speech-worker@tts.service`, `speech-worker@asr.service` | 127.0.0.1:28411 / :28412 only | GPU4 speech models for media-gate's `speech` profile (`speech/speech_worker.py`, stdlib HTTP) | runtimes `/mnt/lexar/omni-campaign/runtimes/speech-qwentts` / `speech-asr` (pydantic-free, `evidence/speech/pip_freeze_*_20261022.txt`); CUDA-free until the first request; `/unload` exits the process (`Restart=always`, `RestartSec=2`); TTS: `SPEECH_CUDA_GRAPHS=1`, `SPEECH_MAX_RESIDENT=3`. Added 2026-10-21, before the media-gate 1.2.0 restart at 23:28Z |
| `comfy-gateway-only.service` | none | Oneshot (`PartOf=docker.service`): `media/gateway_only.sh ensure` makes the GPU4 ComfyUI ports (every profile port in `media_gate.json`) host-only, so the LAN reaches ComfyUI only through media-gate | REJECT tcp-reset in iptables `DOCKER-USER` (by conntrack original port) and iptables/ip6tables `INPUT` (not `lo`). Added 2026-10-21 22:40Z |
| `gen-worker-media.service` | none | Idle-time media generation through media-gate; leases in the MIXTURE_V1 mix order (`media_prompts.lease_order`) | eval banks, corpus holdouts and the refill's held-out index cache are `InaccessiblePaths` |
| `media-refill.timer` | none | 10 min after each pass: `media_prompts.py refill` keeps the GPU4 queue above low water (screened, deduplicated prompts; mix below). Added 2026-10-22 00:16Z (lane mediaqueue) | oneshot, `TimeoutStartSec=1800`, pg advisory lock; reads the holdout registry to screen, never to seed |
| `media-publish.timer` | none | Hourly: judged-pass media spool to intake drop zone | refuses while the NAS has < 50 GiB free; moves nothing without a `media_judgement` pass under the current judge spec. Enabled and active again since 00:37:24Z (checked 01:15Z); every run so far refused at the NAS floor (42 G free at 01:15Z) |
| `media-judge.timer` | none (its llama-server binds 127.0.0.1:18460 only while a job runs) | Hourly :35: meaning check of unjudged spool groups (`media_judge.py cycle`) | claims a P40 via `lanes.py claim-run` (L-EXP GPU5, GPU3; borrows an idle L-TRAIN card only while the train queue is empty); never touches GPU0/4/6 |
| `lane-runner@train.service`, `lane-runner@exp.service` | none | P40 queues: `ai_serve.lane_job` rows started through `lanes.py claim-run` | `KillMode=process` (stopping a runner never kills its job) |

ComfyUI profiles on GPU4 (only one container runs at a time; the manager refuses a second CUDA process on the card):

| Profile | Container : port | Image | Workflows |
|---|---|---|---|
| `general` (default, warm) | `comfyui_qwen21_gpu4` : 28334 | ComfyUI 0.37 cu130 (sm75/sm86) | Qwen-Image 2.1 generate + edit, ACE-Step 1.5 music, LTX 2.3 video, Hunyuan3D 2.1 mesh |
| `h3sol` (on demand) | `comfyui_h3_sol_gpu4` : 28245 | `gauntlet-sol-sm86-v001-20260913` | H3 SOL video: MiniMax H3 Turbo-4 draft 896x512 with audio, GauntletH3Sol attention (SM80+, so GPU4 only), then LTX 2.3 refine to 1344x768 |

A switch takes 32-47 s. `h3sol` is stopped after 5 min idle, then `general` comes back and is re-warmed.

**Speech profile** (`speech`, media-gate 1.2.0, 2026-10-22; section 3, "Speech"):
- **What it is:** not a container. Two loopback `speech_worker` processes: Qwen3-TTS-12Hz-1.7B CustomVoice /
  VoiceDesign / Base, and Qwen3-ASR-1.7B (Apache-2.0).
- **Entering it:** ComfyUI is freed (`/free {"unload_models": true}`) and its container stays up ("parked"); the switch
  takes 2.2-4.4 s.
- **Leaving it:** the workers exit to drop their CUDA contexts, and ComfyUI reloads its models on its next prompt.
- **Idle:** after 300 s idle the card returns to `general`, which is re-warmed.

Status at a glance (no key needed from the server itself):
```
systemctl status strata-llama strata-gate gen-worker-text media-gate gen-worker-media lane-runner@train lane-runner@exp
curl -s http://127.0.0.1:18400/healthz ; curl -s http://127.0.0.1:28400/healthz
python3 $OC/distill/tools/lanes.py show ; tail -20 $OC/distill/queues/p40.claims
```

## 3. Calling the brain and the media lane

### Credentials (vault keys only; never pasted into code)

| Vault key | Used by | Gateway |
|---|---|---|
| `strata_key_sprite` | Sprite stage (`SPRITE_PROVIDER_STRATA_KEY`, read from the vault by `scripts/sprite/start-sprite-stage.ps1`) | strata-gate, tier `sprite` |
| `strata_key_cyril` | Cyril's own clients, omni_lab, ad-hoc tools | strata-gate, tier `cyril` |
| `strata_key_background` | gen-worker-text, media prompt seeding | strata-gate, tier `background` |
| `media_key_cyril` | Cyril's own ComfyUI clients (`X-Serve-Key` header) | media-gate, tier `cyril` |
| `media_key_background` | gen-worker-media | media-gate, tier `background` |
| `strata_llama_internal_key` | strata-gate only (llama-server on 127.0.0.1:18401) | internal |
| `ai_serve_pg`, `ai_serve_ro_pg` | services, read-only queries | PostgreSQL `ai_serve` on :25490 |

Only sha256 hashes of the keys are in `ai_serve.api_client`. On the server, keys reach the services through
mode-0600 env files generated from the vault: `~/.config/ai-serve/strata.env`, `media.env`, `lanes.env`,
`llama_internal_key`. If you rotate a key: `credman.py set`, regenerate the env file
(`strata/deploy_secrets.py`, `lanes/deploy_lanes_env.py`), update the hash row, then `systemctl reload strata-gate` (SIGHUP).

### Strata (http://192.168.50.222:18400/v1)

- Model `strata`: thinking off. Model `strata-think`: thinking on, with `reasoning_content` returned separately.
- Endpoints: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`.
  - The cyril tier also gets the native `/completion /tokenize /props /slots`.
  - `GET /healthz` (LAN, no key) and `GET /gate/status` (any key).
- Images in a request get 400 `no_vision` at once (text-only model). Default `max_tokens`: sprite 4096, cyril 8192.
  Cap: 32768 (one slot). Hard timeout: 600 s.
- Error codes clients should handle:
  - `starting` (503: model loading or llama-server down);
  - `thermal` (503: GPU6 at or above 84 C);
  - `preempted` (503: background only);
  - `busy` (503: background not admitted).

Cyril, from the workstation (PowerShell); this exact call returned "ready" in 1.13 s at 22:33Z:
```powershell
$k = (python "C:/Users/cereal/.claude/skills/server-manager/credman.py" get strata_key_cyril | Select-Object -Last 1).Trim()
$body = @{model="strata"; messages=@(@{role="user"; content="hello"}); max_tokens=256} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://192.168.50.222:18400/v1/chat/completions -Method Post -Headers @{Authorization="Bearer $k"} -ContentType application/json -Body $body
```
Any OpenAI-compatible client works with base URL `http://192.168.50.222:18400/v1`, the key from `strata_key_cyril`,
and model `strata` or `strata-think`.

### Sprite

- **Path:** phone (on-device model first) -> Sprite stage `192.168.50.84:23499` (authenticated Sprite session) -> strata-gate (Bearer `strata_key_sprite`). The phone never holds the key.
- **Providers:** `aiserver.strata` and `aiserver.strata.think` in `D:/SpriteBuild/stage/sprite_providers.json`, enabled, LAN egress, `vision: false`, timeouts 600 s.
  - The code is in commit `e050c30`, "Sprite: AI-server Strata brain and media gate as providers", on `sprite/claude-20261019`.
  - Docs: `docs/SPRITE_BRAIN_SERVER_20261021.md` in the Family AI repo.
- **When Strata is used:**
  - The user picks the engine.
  - The on-device agent's `delegate.llm` tool delegates to it.
  - Automatic fallback on on-device failure is not built.
- **Real Sprite traffic** reached Strata at 20:00-22:22Z: 119 ok; 44 got `starting` while llama-server restarted; 1 client disconnect.
- **Media:** `aiserver.comfy.image`, `.music` and `.video.ltx` point at `http://192.168.50.222:28400`.
  - The gate treats requests from `192.168.50.84`, or carrying `extra_data.sprite_job`, as sprite tier. No Sprite code change was needed.
  - `aiserver.comfy.video.h3` stays disabled until Sprite's `h3_fl2va_v1` graph is ported to the `h3sol` profile.

### Media lane (http://192.168.50.222:28400, ComfyUI API)

- `POST /prompt` with a ComfyUI API graph, header `X-Serve-Key: <media_key_cyril>`.
- Then `GET /history/{prompt_id}` and `GET /view?...` exactly as with plain ComfyUI.
- `GET /gate/status` shows the active profile, jobs, admission and GPU temperature. `GET /gate/prompt/{id}` shows one job.
- The gate routes a graph to the profile whose nodes it uses (`GauntletH3Sol` goes to `h3sol`, everything else to `general`) and switches if needed.
- Working graph builders: `media/media_graphs.py` (`qwen_graph`, `ace_graph`, `ltx_graph`, `h3sol_draft_graph`, `h3sol_refine_graph`, `hy3d_graph`; ported from Sprite's `sprite_workflows.py` and the Gauntlet Sol driver).

### Speech (http://192.168.50.222:28400/v1/audio/..., OpenAI audio API shape)

- **Auth:** same tiers as `/prompt`. The Sprite stage host (192.168.50.84) needs no key. Everyone else sends
  `X-Serve-Key` or `Authorization: Bearer` with `media_key_cyril` (or `media_key_background` for idle-time work).
- **Endpoints:**
  ```
  POST /v1/audio/speech         {"model":"qwen3-tts","voice":"ryan","input":"...","language":"en",
                                 "response_format":"wav"|"flac"|"pcm","stream_format":"audio" (pcm only),
                                 "instructions":"...","seed":7}
        model qwen3-tts-voicedesign: "instructions" = the voice description (required)
        model qwen3-tts-base:        "ref_audio" (base64, 3-15 s) + "ref_text", or "voice" = stored clone id
  POST /v1/audio/transcriptions multipart file=@clip [language=en] [response_format=json|text|verbose_json]
  GET  /v1/audio/voices         speakers (vivian serena uncle_fu dylan eric ryan aiden ono_anna sohee), languages
  GET  /v1/models
  ```
- **Output:** 24 kHz mono 16-bit. A stream is raw s16le, one chunk per text segment; the first segment ends at the
  first clause break.
- **Quality gate:** every segment is checked before a byte is sent (finite samples, RMS >= -45 dBFS, a duration that
  fits the text). It is regenerated once on failure, then the request fails with 502 `bad_audio`.
- **Preemption:** a Sprite/cyril request cancels a running background synthesis at its next 12 Hz codec token; the
  background caller gets 503 `preempted`. Speech jobs never appear in `/queue`.
  - **Streams (fixed 2026-10-22 02:14Z, media-gate 1.2.1 + speech_worker 1.3.0; section 10 item 1):** a stream
    answers 200 only when its first audio chunk exists, so a failure before that gets the same 503/502 JSON as a
    non-streamed call. After audio has gone out, a preempted, `bad_audio`, cancelled or crashed stream ends without
    its closing chunk: the caller's client raises (httpx `RemoteProtocolError`, aiohttp `ClientPayloadError`), and
    the reason is at `X-Speech-Outcome` (`GET /gate/prompt/<id>`, field `code`). HTTP/1.0 streaming gets 400.
    Contract: `ai-serve/speech/STREAM_CONTRACT.md`.
- **Records:** `request_log` rows with `model_or_profile = speech`, `workflow_id` `tts.<variant>` or
  `asr.qwen3-asr-1.7b`; `first_token_at` = the first audio byte.
- **Measured** (2026-10-21/22, from 192.168.50.84 through the gate; harness `ai-serve/speech/speech_e2e.py`; receipts
  `evidence/speech-e2e-*`):
  - **Run on the deployed code (omni_lab 38127, pass, 02:15-02:32Z, gate 1.2.1 / worker 1.3.0, receipt
    `evidence/speech-e2e-20261022T0218Z-streamfix/`):** 30/30 clips non-silent and plausible. Round-trip WER through
    Qwen3-ASR-1.7B: 5.87 % raw on 494 words (29 edits), 0.40 % with the ASR's digits spelled out (2 edits). The new
    `streamcut` phase: a background stream preempted 1 s after its first audio got HTTP 200, then `ClientPayloadError`
    after 61,440 bytes; `X-Speech-Outcome` and `request_log` say `preempted`. The non-streamed background request
    got 503 `preempted`; the Sprite stream that displaced it got its first audio in 2.70 s.
  - Earlier run 38079 (29/29, WER 5.41 % / 0.42 %) was on media-gate 1.2.0 and is superseded.
  - **Time to first audio, warm:** non-streaming sentence median 4.4 s; streamed paragraph 2.81 s (while preempting a
    background synthesis) and 2.36 s (run 38096). Cold, switch from `general` plus model load: 19.3-22.4 s.
  - **Compute per second of audio** (CustomVoice, warm, same seeds, same audio): 3.42 s upstream; 2.53 s with the plain
    predictor loop; 1.05 s with it CUDA-graphed (runs 38074 / 38076 / 38077).
  - **Card return:**
    - Idle: back on `general` 446.2 s after the last speech request; the next image took 24.7 s.
    - An image arriving while `speech` holds the card: 167.6-240.7 s with `free_memory`, 79.3 s with `park_keep_cache`
      (not a clean A/B: ACE-Step background jobs held the card before). Warm image: 13.7 s.
- **Sprite provider (documented, not applied):** see section 9. Sprite's provider schema has no speech capability yet.

## 4. Preemption and idle work

**Strata (`strata/strata_gate.py`):**
- **Slots and tiers:** 6 upstream slots. Foreground (sprite, then cyril) can use all 6 and never queues behind background.
- **Preemption:** when a foreground request arrives, every running background stream is closed. llama-server cancels on disconnect; background requests are always streamed upstream. The caller gets 503 `preempted`.
- **Background admission:** only when all of these hold:
  - no foreground request is running or waiting;
  - none has finished for 20 s;
  - fewer than 2 background requests are running (`--bg-max 2`);
  - MemAvailable >= 40 GiB;
  - the thermal level allows it.
- **Thermal levels for GPU6,** derived from the card's own limits (82 C max operating, 84 C slowdown, 87 C shutdown):

  | From | Effect |
  |---|---|
  | 77 C | no new background |
  | 82 C | running background is cancelled |
  | 84 C | new requests of every tier get 503 `thermal` |
  | 85 C | everything running is cancelled |

  Each level clears 3 C lower. llama-server is never killed. Unreadable limits hold background off and are retried.
- **GPU0 thresholds:** 88/93/95/96 C.
- **Thermal log:** samples every 5 s in `/home/cereal/ai-serve/logs/thermal.csv`.

**Text worker (`strata/gen_worker_text.py`):**
- Leases `toolcall_conv` batches (24 conversations each) from `ai_serve.gen_task`.
- Runs the existing `toolcall_selfgen.py run`. Strata writes the user turns, simulated backends and tool calls, per the 2026-10-17 corpus ruling.
- Each batch then goes through `finalize` (review-surface and duplicate screens) and `toolcall_batch_screen.py` (held-out screen, `toolcall_heldout_screen` 1.1.0).
- A preempted request is retried by the selfgen client. A batch whose clients all give up (about 10 min of sustained foreground) is requeued and resumes from its journal; nothing is regenerated.
- Queue at 22:27Z: 3 batches done, 1 leased (task 6), 97 queued.
- **1.3.0 (2026-10-22 00:43Z, lane mediaqueue; code in `strata/text_refill.py` and `strata/text_tasks.py`).** Three
  more verified kinds. In each, Strata (Cyril 2026-10-17: "[Cyril quote withheld]") writes the user turn and the
  answer:
  - `reasoning_math` (16 items/batch). Strata writes a problem, a Python program and an answer. The program runs in
    `sandbox_exec`, then `strata-think` solves the problem cold. Kept only when all three answers agree exactly.
  - `instruction_code` (16). Strata writes a task, 4-10 asserts and a reference. Only asserts the reference passes
    count (>= 4 and >= 70 %), and a stub must fail them. The solver never sees the tests and must pass them all.
  - `long_context` (8). A ~10k-word context of LJSpeech public-domain passages. The question and a 2-12 word quote
    are written from the target passage only; the quote must occur there and nowhere else. The full-context answer
    must contain it.
  - Every batch then goes through `text_batch_screen.py` (the same `toolcall_heldout_screen` 1.1.0 indexes).
    Seeds (COCO train captions, LJSpeech) pass the media refill's held-out screen before Strata sees them.
  - The serve loop refills every 10 min: each kind is kept at 3-8 queued batches; toolcall takes the next slices of
    the newest plan, or a new plan with the next seed. The lease takes the kind furthest behind its share of the
    last 24 h's finished batches. Shares: toolcall 0.40, reasoning 0.25, code 0.20, long 0.15.
  - First refill at 00:43:31Z: tasks 372-395 (8 batches per new kind); 372 (`reasoning_math`) was leased first.
  - **Production path verified** on a 2-item `long_context` task, run by hand with `run-task 396`
    (base seed 20261022). Result:
    - 1 kept, 1 `answer_misses_quote`;
    - screen `pass`, `gen_output` 62;
    - omni_lab run 38097 `pass` on L-SERVE;
    - spool folder `text/strata-long-context-20261022-t000396/` holding `long_context.jsonl`, `.sha256`,
      `_meta.json` (route `text`) and `.provenance.json`.
  - Staging live runs before deploy (real Strata, never published), accepted items:
    - math 1 of 4;
    - code 0 of 4 before the reference cross-check, then 2 of 6 after;
    - long context 3 of 3.

    `text_batch_screen` kept every accepted row.
  - **Code sandbox: own-user isolation since 2026-10-22 02:25Z (sandbox lane; section 10 item 4 fixed).** Versions
    since the adversarial review (03:05Z): `sandbox_exec` 2.1.0, `text_tasks` 1.2.0, `gen_worker_text` 1.4.0,
    wrapper `strata-sbx-exec` 1.1.0 (sha256 `153c9cb1...`). The running gen-worker-text process still has
    `sandbox_exec` 2.0.0 imported for its own 15-min hold-back check until its next restart; every code batch runs
    the 2.1.0 check itself (11 probes) before any program runs.
    - **Adversarial review, 2026-10-22 02:35-03:09Z** (evidence `evidence/adv-sandbox-20261022/`, omni_lab run
      38140). Every attack below went through the real `sudo -n` wrapper with the audit hook switched off.
      - **Held:** strata.env and `/proc/1/root/...` unreadable, no network, no writes outside /work, no escalation,
        limits, a SIGTERM-ignoring double-forked orphan did not outlive its unit, 3 simultaneous callers ran one
        after another (no overlap), only `lo` in `/sys/class/net`.
      - **Found and fixed (wrapper 1.1.0):** `io_uring_setup` succeeded inside the unit (`@system-service`
        includes it; the host has `kernel.io_uring_disabled=0`), so it and `io_uring_enter`/`io_uring_register`
        are filtered now; `/work` was mounted without noexec (a copied binary ran from it), so it is noexec now;
        the setgid `utempter` helper was in the view (NoNewPrivileges already defeated it), so it is hidden; an
        orphan that ignores SIGTERM held the turn for 30 s and came back `timed_out`, now `TimeoutStopSec=2`
        kills it in about 3 s (the run fails, rc 1); the turn is only taken once no `strata-sbx` process is left
        anywhere on the host (tested: a stray `strata-sbx` sleep outside any unit made the next caller wait 40 s,
        then it ran).
      - **Found and fixed (text_tasks 1.2.0), the serious one:** `run_tests` counted every stdout line starting
        `PASS`, including the solution's own prints, and accepted on "PASS count == tests". On the server, a
        `grade()` solution with the wrong threshold that prints `PASS` while it decides was **accepted while it
        failed a test** (marks PASS x4 + FAIL), and a program that printed four `PASS` lines and exited before
        any test was also accepted. Tests now report `<random per-run marker> <index> PASS|FAIL` and marks are
        read per test; both cases are rejected now (`verifier_demo_before.log` / `verifier_demo_after.log`).
        A program written to read the marker out of its own source could still forge it (same process).
      - **Existing corpus:** all 27 `instruction_code` records accepted under 1.1.0 (tasks 380, 382, 383) were
        re-run through the 1.2.0 verifier: 27 OK, 0 false accepts (`reverify_code.json`).
      - **Provenance:** text_tasks now hashes itself and `sandbox_exec.py` when a batch starts, not at the end,
        so a file replaced mid-batch no longer stamps the manifest with code that batch did not run.
      - **Deploy:** code kinds paused by policy 03:00:54Z (worker not restarted; leased task 383 finished
        03:01:11Z under the old code), files installed 03:02:56Z, resumed 03:05:36Z. Backup
        `/home/cereal/ai-serve/backups/sandbox-adv-20261022T030054Z/` (including the installed 1.0.0 wrapper).
        Isolation check 11/11, `tests.test_sandbox_isolation` 8/8, strata suite 81 OK, staging
        `gen/staging-adv-20261022` on real Strata 3 of 4 accepted (never spooled). No gate or llama service
        touched.
      - **Left open (low):** the host cgroup tree under `/sys/fs/cgroup` is readable (read-only) inside the unit;
        `memfd_create` and `pidfd_open` are allowed (no gain without ptrace/kill rights); a SIGKILLed wrapper drops
        the lock early, which the leftover check now covers.
    - **Where it came from.** It reuses the Agentic Operations platform's least-privilege rule
      (`SOC_TESTING/soc-dashboard/api/services/agent_harness.py`, `docs/HERMES_HARNESS.md` "Sudo Policy"): drop to a
      fixed uid with all groups cleared, no sudo for the worker, no secrets in the child's environment, and elevation
      only through one narrow root-owned adapter. That platform's boundary itself does not meet the requirements, so
      it was not reused as it is:
      - its uid is 1000, which is `cereal` on this host;
      - its network guard is a curl wrapper on PATH;
      - the agent sees the whole container filesystem;
      - its Codex harness runs `danger-full-access` because bwrap namespaces are blocked on this host.
    - **Who runs the program.** Every Strata program runs as `strata-sbx` (uid 995, gid 984, no supplementary group,
      nologin, password locked). The path is: `sudo -n /usr/local/libexec/strata-sbx/strata-sbx-exec` (root-owned;
      `/etc/sudoers.d/strata-sbx`) -> a transient systemd unit. `cereal`'s groups (docker, sudo) never reach the child.
    - **What the unit allows** (the properties are fixed in the root-owned wrapper; the caller can only lower the
      limits within bounds):
      - Network: `PrivateNetwork` (only `lo`), `IPAddressDeny=any`, AF_UNIX sockets only.
      - Filesystem: an empty read-only root holding python3.12, timeout, the stdlib, the shared libraries and
        `ld.so.cache`. No /home, /mnt, /tmp, /var or /etc files. `/run` is read-only and empty. The host
        `/dev/shm`, `/dev/hugepages` and `/dev/mqueue` are inaccessible. `/work` is a private 64 MiB tmpfs and the
        only writable place.
      - Privileges: `NoNewPrivileges`, an empty capability set, `RestrictSUIDSGID`, `RestrictNamespaces`. seccomp is
        `@system-service` minus `@privileged` and similar groups; a filtered call fails with EPERM.
      - Processes: `ProtectProc=invisible`.
      - Limits: MemoryMax/AS 1 GiB, no swap, CPUQuota 100 %, TasksMax 8, CPU 10 s, FSIZE 10 MiB, a 20 s wall clock.
      - Output: stdout over `--max-output` is drained and dropped.
      - Concurrency: units run one at a time under a root flock, because all sandboxes share one uid.
    - **No switch-off.** `run_python` only goes through the wrapper. If the wrapper is missing or fails, the result is
      `SandboxUnavailable`, never a fallback. The 1.x audit hook still runs inside the unit, so verifier results do
      not move, but it is no longer the boundary.
      - A code batch (`reasoning_math`, `instruction_code`) starts only after `require_isolation()` passes every
        probe through the real wrapper. A failed check, or the sandbox going away mid-batch, ends the batch with
        exit 3 (outcome `sandbox`): it is requeued with a 30 min backoff, and the item that hit the failure is not
        journaled.
      - The worker also holds the code kinds back (no lease, no refill, `run-task` refuses them) while the check
        fails. It re-checks every 15 min.
      - `strata/gen_worker_text.json` `paused_kinds` is a policy pause only. A config that cannot be read pauses
        every code kind. Removing a kind from the list does not skip the check.
    - **Isolation check** (`python3 sandbox_exec.py`, 10 probes; each attack disarms the audit hook first):
      - runs, and hook still on;
      - identity: uid 995, groups [984], NoNewPrivs 1, CapEff/CapPrm/CapBnd 0, seccomp 2, 2 pids visible;
      - secrets, 18 refused: strata/media/lanes env files, `llama_internal_key`, `/etc/shadow`, the caller's
        `/proc/<pid>/environ` and `/proc/<pid>/root/...strata.env`, and listing /home/cereal, ~/.ssh, /root and /mnt;
      - network, 16 refused: LAN 18400/28400/25490, loopback, DNS, UDP, docker.sock, the system bus, the journal
        socket; interfaces `lo` only;
      - writes outside /work: none, also checked from the host;
      - escalation: setuid, setgroups, user and net namespaces, chroot, and sudo/su/docker are not in the view;
      - limits: a 1.5 GiB allocation fails, forks stop at 6, /work stops at 64 MiB, a CPU spin is killed;
      - a 64 MiB stdout flood is cut to 4 KiB.
    - **Run history.** The first run failed for real: a `/dev/shm` file written in the sandbox appeared on the host,
      and seccomp SIGSYS killed the escalation probe. The second run failed closed: `ReadOnlyPaths=/run` gives
      226/NAMESPACE on systemd 255. The third run passed 10/10. Evidence: `evidence/sandbox-20261022/`
      (`isolation_check_{1,2,3}.json`, `deployed_SHA256SUMS`, staging manifests and journals); omni_lab run 38129
      (`pass`); the pause was run 38102, ended at the resume.
    - **Tests:**
      - `tests.test_sandbox_isolation` (real wrapper) 7/7;
      - `TestVerifiers` + `TestBatchResume` on the real wrapper 3/3;
      - `tests.test_sandbox_gate` (fakes) 12/12;
      - the strata suite on the deployed files (`test_sandbox_gate`, `test_sandbox_isolation`, `test_text_tasks`
        including `TestRefillDb.test_lease_skips_held_kinds`, `test_gen_worker_text`, `test_gen_worker_failures`):
        53 tests OK at 02:30Z.
    - **Staging through the sandbox** (real Strata, never published; `gen/staging-sandbox-20261022`):
      - math 0 of 4: 3 author answers disagree with their own program; 1 solver answered `\(\frac{1}{4}\)`, which
        `parse_number` does not read (verifier gap, not the sandbox);
      - code 2 of 4: 2 `author_malformed`. The two accepted items had a stub fail all tests and the solution pass
        all of them (9/9, 8/8).
      - No item failed because of the sandbox.
    - **Costs:** about 0.25 s per program and about 6 s per isolation check. Each run writes one sudo line to
      auth.log and a few systemd journal lines.
    - **Undo:** `sudo rm /etc/sudoers.d/strata-sbx /usr/local/libexec/strata-sbx/strata-sbx-exec`. The code kinds then
      hold back by themselves, because the check fails closed. The 1.3.0 files are in
      `backups/sandbox-20261022T013724Z/`.

**Media gate (`media/media_gate.py`):**
- **Preemption:** a foreground `/prompt` interrupts the running background prompt, deletes pending background prompts, and goes to the front.
- **Background admission:** only when all of these hold:
  - no foreground job is in flight;
  - no foreground request has arrived for 60 s;
  - at most 1 background job is running;
  - GPU4 is below 82 C;
  - MemAvailable >= 40 GiB.
- **Thermal:** at 88 C, background is interrupted.
- **What Sprite sees:** `/queue` hides background entries from Sprite, and `/system_stats` and `/object_info` come from a cache, so Sprite's 2 s health check never waits on a switch.
- **Requeue:** the worker requeues a preempted task.

**Media worker (`media/gen_worker_media.py`):**
- Task kinds: `image_t2i`, `image_edit`, `video_ltx`, `video_h3sol`, `music_ace`, `mesh_hy3d`.
- Prompts come from Strata (background key) or from a file. Every artifact is validated (decodes, not flat, motion, loudness, mesh not a backdrop) before it is spooled.
- **The queue refills itself** (worker 1.2.0 + `media_prompts.py` 1.0.0, since 2026-10-22 00:16Z). The full spec is
  in `media/README.md`, "Automatic queue refill".
  - **Low water.** A kind is refilled when its queued GPU seconds fall below 3 h x its share, up to 8 h x its share.
    Seconds per task are measured from `request_log`. Under 1 h of total work, COCO prompts fill `image_t2i` without
    Strata.
  - **Mix,** as shares of GPU4 seconds (MIXTURE_V1, ARCHITECTURE_V2 14.4):

    | Kind | Share |
    |---|---|
    | `image_t2i` | 5 |
    | `image_edit` | 5 |
    | `video_h3sol` | 6 |
    | `video_ltx` | 2 |
    | `music_ace` | 7 |
    | `mesh_hy3d` | 5 |

    Speech (15) is not in it: no speech task kind exists in the worker yet.
  - **Lease order.** The worker leases the kind furthest behind its share of the last 24 h first. Explicit seeds
    (priority 100) still run before refill tasks (priority 200). h3sol runs in bursts: switch in when it is 1 task
    behind, out when it is 2 tasks ahead.
  - **Sources.**
    - COCO 2017 *train* captions: 30 % of `image_t2i`, used verbatim (prompt source `dataset`).
    - Strata briefs, seeded from COCO captions, COCO/ModelNet10 object names, LJSpeech transcripts, the corpus ACE
      captions (music only) and intake `.txt/.md` drops.
    - Edits of existing validated t2i images.
  - **Screens.** Held-out (registry v3: word 8-grams, whole held texts and sentences, literals, media sha256; liveness
    48/48). Dedup against every media task ever enqueued (exact, plus 3-gram Jaccard >= 0.6). A seed whose 6-word run
    is copied is dropped. Scene-swap edits are refused.
  - **First live pass, 00:33Z (omni_lab 38087):** 16 Strata briefs. Enqueued: t2i 108, music 48, mesh 36, edit 18,
    LTX 16, h3sol 11. 5 held seeds and 3 scene swaps were refused. GPU4 started on music at 00:33Z (task 135); tasks
    137-141 finished `music_ace` validated, about 20 s each.
  - **Seeding by hand** still works and runs first: `gen_worker_media.py seed --kind K --source strata|file ...`
    (priority 100).
  ```
  cd /home/cereal/ai-serve/media && AI_SERVE_ENV_FILE=/home/cereal/.config/ai-serve/media.env \
    /home/cereal/.venvs/ai-serve-media/bin/python media_prompts.py plan     # shares, queue, need, lease order
  ```

## 5. Pausing background work, taking a card

**Pause all background generation (serving stays up):**
```
sudo systemctl stop gen-worker-text gen-worker-media media-publish.timer
# resume
sudo systemctl start gen-worker-text gen-worker-media media-publish.timer
```
- gen-worker-text stops within 120 s; its batch resumes from the journal on the next start.
- Do NOT pause by disabling the background key. A 401 makes the selfgen client treat every sample as rejected, so the batch in flight ends with 0 rows and is marked `done`.

**Take a serving card (GPU0/GPU6 or GPU4): Cyril's approval first.** The `p40.claims` CLAIM line carries the preempt command.
- **Strata cards (GPU0 + GPU6):**
  ```
  sudo systemctl stop gen-worker-text strata-gate strata-llama
  ```
  - The RELEASE line is written and `gpu0.lock`/`gpu6.lock` are freed.
  - Start your job with `lanes.py claim-run --lane L-SERVE --gpus 0,6 --cyril "<his words, verbatim>" ...`.
  - When done, `sudo systemctl start strata-llama strata-gate gen-worker-text`.
  - The model is back in about 152 s (270 s after a crash).
  - If another holder keeps the lock, strata-llama exits 75 and retries every 20 s without alerting, so watch `journalctl -u strata-llama`.
- **Media card (GPU4):**
  ```
  sudo systemctl stop gen-worker-media media-gate-health.timer media-gate
  python3 /home/cereal/comfyui/manage_comfy_instances.py stop comfyui_qwen21_gpu4
  ```
  - GPU4's owner in `/home/cereal/comfyui/comfy-gpu-owners.json` is `ai-serve`, so the manager refuses other owners' containers there.
  - Reverse the steps when done.

**Experiments and training on P40s** (no approval needed in their own lane):
```
# queued (one job at a time per lane; the tool must be pinned in $OC/distill/tools/TOOL_PINS.json)
/home/cereal/.venvs/ai-serve/bin/python /home/cereal/ai-serve/lanes/lane_runner.py enqueue --lane exp --name my-probe --tool <pinned tool> --gpus-needed 1 --argv '["..."]'
/home/cereal/.venvs/ai-serve/bin/python /home/cereal/ai-serve/lanes/lane_runner.py list
# by hand (same locks and claims)
python3 $OC/distill/tools/lanes.py claim-run --lane L-EXP --agent claude --gpus 5 --job NAME --log /path/job.log --omni-lab-kind probe -- CMD ARGS
```
- The queued and by-hand commands share the same locks and claims. `--borrow REASON` takes another lane's P40.
- Nothing on a P40 is ever preempted by a scheduler. Stop a job only with the `preempt:` command in its CLAIM line.
- For hard work, a queued job with `--exclusive` takes every card of its own lane. To use another lane's P40, run by hand with `--borrow REASON`.

## 6. How outputs reach the corpus

1. **Spool:** each batch is spooled locally, with provenance in every row and a whole-batch `.provenance.json`:
   - text in `/home/cereal/ai-serve/spool/text/strata-toolcall-<date>-t<task>/`;
   - media in `/home/cereal/ai-serve/spool/media/`.

   Provenance fields:
   - model and GGUF sha;
   - prompt source;
   - sampling params;
   - gateway request ids;
   - verifier and screen results;
   - `synthetic: true`.
2. **Mover to the NAS drop zone** `/mnt/ai_data/chimera-intake/` (copy as `.part`, re-hash, rename; the spool copy is deleted only after it re-hashes equal):
   - The text mover runs between batches and needs >= 20 GB free on the NAS. It writes to `text/`.
   - `media-publish.timer` needs >= 50 GiB free. It writes to `<modality>/aiserve-<kind>-<date>/`.
   - **Media publishes only through the meaning check** (since 2026-10-21 ~23:30Z; `gen_worker_media` >= 1.1.0). An artifact moves
     only when `ai_serve.media_judgement` says `verdict = pass` and `group_verdict = pass` under the sha256 of
     `media/media_judge_spec.json`. The judge is `media-judge.timer` (hourly :35) -> `media_judge.py cycle` -> `lanes.py claim-run`
     on a P40 -> Qwen3.6-27B VL + CLAP + ASR + mesh checks; see `media/README.md`, section "Meaning check". Rejected groups are kept
     in `spool/media/rejected/` with their reasons.
3. **Intake:** `intake_cron.sh` (cron every 30 min, `$OC/distill/tools/`) runs `intake.py` over the drop zones.
   - Prompts are screened against the holdout registry.
   - Logs go to `$OC/distill/queues/logs/intake.log`.
4. **What is excluded:**
   - Only `strata` and `file` prompt sources are published. Claude-written validation and warmup prompts stay in the spool.
   - Claude/Codex outputs never enter.
   - The media worker cannot read the eval banks or holdouts (systemd `InaccessiblePaths`). The text path relies on the fixed prompt sources plus both screens on every batch.

**State at 22:27Z:**
- The NAS has **42 GB free** (disk cleanup is running in another workflow), up from 2.4-3.0 GB earlier today.
- **Text:** 3 batches (task 1: 6 of 8 kept, 11,431 tokens; tasks 2-3) sit in the spool. All screened `pass`.
  - Batch t000001 was copied to the NAS at 22:06Z, but the database connection was cut mid-move (the DB-outage test).
  - Its spool copy and the `gen_output` row were left behind. The next mover pass will see the equal hash, update the row and delete the spool copy.
- **Media:** 41 validated outputs (91 MB) wait in the spool until the NAS has 50 GiB free.
- **Not yet ingested:** `gen_output.intake_batch_id` is still empty for all rows.

## 7. Records: omni_lab and the `ai_serve` database

**omni_lab** (PostgreSQL `omni_lab` on :25490; read-only role `omni_lab_ro`, `omni_lab.py sql` from the workstation):
- **Gateway runs:** each gateway start registers a `serve` run with periodic metrics.
  - Current runs: strata-gate 38025, media-gate 38042.
  - Earlier starts: 37897, 37898, 37901, 37902, 37904, 37905, 37997, 38021, 38024, 38030.
- **Text batches:** one `gen` run per batch: 37899, 37903, 38009 (pass); 38016, 38031, 38043 (running; see section 9).
- **Media:**
  - validations 38003-38007;
  - Sprite-path preemption probe 38008;
  - one `gen` run per published batch, written when publishing starts.
- **End-to-end tests:** 38044-38049 (all pass).
- **Lanes:** lane-runner jobs register through `claim-run`. omni_lab lane ids: 4 L-TRAIN, 56 L-EXP, 57 L-SERVE, 58 L-MEDIA. The arch-v2 proxy runs 38033-38041 are on L-EXP.

Useful query:
```
C:/Python314/python.exe omni_lab.py sql "SELECT id, kind, name, status, started_at FROM run WHERE started_at > now() - interval '1 day' AND name ~* 'strata|media|gen|e2e' ORDER BY id"
```

**`ai_serve` database** (raw SQL; migrations 0001-0004 in `ai-serve/db/migrations/`). Tables:
- `api_client`: key hashes;
- `request_log`: one row per request, no prompt text;
- `service_event`: preempt, thermal, profile_switch, container_lost and similar;
- `gen_task` / `gen_output`: the generation queue and its outputs, with provenance, screen result and omni_lab run id;
- `lane_job`: the P40 queue.

**Request totals to 22:27Z:**

| Gateway | Tier | ok | Other outcomes |
|---|---|---|---|
| Strata | sprite | 119 | 44 `starting`, 1 error |
| Strata | cyril | 179 | |
| Strata | background | 688 | 52 preempted, 236 not admitted (mostly GPU6 thermal) |
| Media | sprite | 8 | 1 error (ComfyUI crash test) |
| Media | cyril | 27 | 4 errors |
| Media | background | 47 | 7 preempted |

**Other logs:**
- `journalctl -u <unit>`
- `/home/cereal/ai-serve/logs/thermal.csv`
- `/home/cereal/ai-serve/lane-logs/`
- evidence in `/home/cereal/ai-serve/evidence/`:
  - `live_gate_check_20261021T1824Z.json`
  - `media-validation/`
  - `e2e-20261021/`
  - the `pip_freeze_*.txt` files

## 8. Measured performance (2026-10-21, omni_lab 38044-38049 unless noted)

| What | Result |
|---|---|
| Strata decode, 1 stream | 47.9 tok/s, first token 0.94 s |
| Strata, 6 concurrent | 18.2 tok/s each, 100.3 tok/s aggregate, first token <= 2.2 s |
| Strata, 4 concurrent | 14.9 tok/s each (56.7 aggregate): slower than 6; ran once, unexplained |
| Strata, 8 concurrent | 6 run at 18.1 tok/s; 2 wait about 26 s for a slot |
| Strata prefill | 823 / 862 / 871 tok/s at 2k / 6k / 9.6k-token prompts; decode 45-47 tok/s after |
| Sprite-shaped turn (6.8k-token prompt) | 15.1 s to first token (p50), 20.7 s total |
| Short Sprite chat | first token 0.20-0.51 s at the server, <= 0.87 s total |
| Strata preemption | background cancelled in 2-3 ms; background resumes 20.7-21.7 s later |
| Strata restart | clean 152 s; after SIGKILL 270 s (Sprite gets 503 `starting` in 0.01 s meanwhile) |
| Media preemption | Sprite job starts in 1.0-1.2 s, done in 16.1 s; background LTX interrupted in < 10 ms |
| Media time to artifact | image 15.2 s; edit 30.3 s; music (20 s clip) 42.3 s; LTX video 189.5 s; Hunyuan3D 150.3 s; H3 SOL draft + audio + refine 479.3 s (32 s of it the profile switch) |
| ComfyUI crash | Sprite image still delivered in 87 s (either profile) |
| Database down 90-280 s | 100 % of health checks answered, 9/9 Sprite chats ok, request rows buffered and written afterwards |
| GPU6 under 6 streams for 12 min | 81-82 C, SM clock p50 1522-1545 MHz, software thermal slowdown in 79 % of samples, 0 refusals |

## 9. Known limits and open items

- **GPU6 headroom is thin.**
  - It idles at about 72-74 C with the model loaded.
  - The "no new background" level (77 C) was hit 173 times today, plus 29 background cancels at 82 C. Background therefore runs in bursts.
  - The 24 h soak (`thermal.csv`) decides whether `--bg-max` goes from 2 to 4.
  - A power or clock limit change on GPU6 needs Cyril's approval.
  - **Superseded 2026-10-22 05:45Z:** Cyril approved a safe cap; GPU6 runs at 200 W (`gpu6-power.service`, 76-77 C
    under 6-stream load, no throttling) and `--bg-max` is 4 (section 10, "Also done").
- **GPU4 also throttles:** 79 C, with software slowdown in 26.5 % of busy samples during LTX.
- **Speech is served (2026-10-22), Sprite is not wired to it.**
  - Sprite's `sprite_providers.py` (branch `sprite/claude-20261019`) knows no speech capability: its KINDS are
    `openai_compat`, `codex_responses`, `ai_task` and `comfyui`, and its CAPABILITIES lack speech. Wiring speech is a
    backend + phone change for the Sprite lead (Relay note 2026-10-22).
  - Proposed entry for `D:/SpriteBuild/stage/sprite_providers.json`, once the backend has the kind:
    `{"alias": "aiserver.speech", "label": "Speech (AI server)", "kind": "openai_audio",
    "base_url": "http://192.168.50.222:28400/v1", "model": "qwen3-tts", "api_key_env": null,
    "capabilities": ["speech.synthesize", "speech.transcribe"], "egress": "lan", "enabled": true, "timeout_s": 600,
    "concurrency": 2, "cost": {}}`
    - No key is needed: the stage host is the sprite tier by IP, as for the comfy providers.
    - Synthesis: `POST /audio/speech` with `response_format=pcm`, `stream_format=audio` and a voice from
      `GET /audio/voices`.
    - Transcription: `POST /audio/transcriptions` (multipart; the worker decodes M4A/AAC/WebM through ffmpeg).
    - Health check: `GET /healthz` on :28400.
- **Speech costs the next image a reload:** an image arriving while `speech` holds GPU4 waits for the switch plus a
  ComfyUI model reload (measured 79.3-240.7 s, section 3).
  - Untested remedy: TTS beside a warm `general` (VRAM permitting). The workload is bound by kernel launches, not by
    the card.
  - **Placement decided (Cyril 2026-10-22):** "[Cyril quote withheld]".
    Speech stays on the shared GPU4 lane, no P40, mainly for idle-time training data. Sprite speech runs on the phone
    by default; the server TTS is an optional fallback (Relay note to the Sprite lead, 2026-10-22).
- **The TTS talker is not graphed:** 28 eager layers per 12 Hz frame, so about 1 s of compute per second of audio.
  qwen-tts 0.1.1 has no token streaming: the first streamed chunk waits for its whole first segment.
- **The Q4 Strata (IQ4_XS) is downloaded and proven identical, but not served.** IQ3_S is still served (lane q4, 2026-10-22).
  - Files: `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata-IQ4_XS/` (8 shards from `0bserverx/...-GGUF` @75358160; each sha256 = LFS; `SHA256SUMS`).
  - Identity: all header fields match. All 608 tensors that kept their type (52.5 GB) are byte-identical to the served IQ3_S. Evidence: `evidence/q4-20261021/q4_receipt.json`; omni_lab run 38065.
  - It does not fit fully on GPU0+GPU6 at 6 x 32k; the estimate is `-ts 17,31 -ncmoe 3`.
  - The A/B (on P40s) and the swap (staged in `strata/q4_swap/`, needs Cyril) are in `STRATA_Q4_PLAN_2026-10-21.md`.
  - Q4_K_M is rejected.
  - **Closed 2026-10-22 04:31Z:** Cyril said not to use the Q4; IQ3_S stays and the IQ4_XS shards are deleted (omni_lab
    38172; section 10, "Also done").
- **Media meaning check (added 2026-10-21/22, Claude):**
  - `media-publish.timer` was stopped and disabled at 22:34Z (generation kept spooling) and re-enabled behind the judge;
    it has been active since 00:37:24Z. Nothing has published: every run refused at the NAS floor.
  - Publish now needs a `media_judgement` pass (see section 6 and `media/README.md`). Judge spec 1.2.0 is sha256 `c9cc4c21...`; its
    `history` field records why 1.0.0 and 1.1.0 changed. Neither change moved a threshold, and only validation files had been scored
    when they were made.
  - First real job (spec 1.1.0, 23:56-00:20Z, GPU1 borrowed from the idle L-TRAIN, omni_lab review run 38075, eval runs 38080-38085),
    by group (pass / reject / hold):

    | Kind | Pass | Reject | Hold |
    |---|---|---|---|
    | image_t2i | 8 | 0 | 0 |
    | image_edit | 0 | 2 | 3 |
    | mesh_hy3d | 1 | 1 | 3 |
    | music_ace | 2 | 2 | 0 |
    | video_ltx | 1 | 1 | 0 |
    | video_h3sol | 0 | 2 | 0 |

    The six holds were pairs generated before the worker kept the stage-0 image. Spec 1.2.0 rejects such incomplete pairs.
  - Rejection reasons seen:
    - **edit:** the change appears only in one corner; over-sharpened and crunchy texture.
    - **mesh:** base plate.
    - **music:** vocals missing (ASR recall 0.05); CLAP rank 8 of 25.
    - **LTX:** no motion.
    - **H3:** the LTX refine adds garbled text and morphing; the subject is missing.
  - Judge time: about 12-30 s per image on a P40, about 3 min per edit pair, 5 min of CPU ASR per music batch, and 1-2 min to load the VL model.
- **Text generation kinds:**
  - Built: `toolcall_conv`, `reasoning_math`, `instruction_code` and `long_context` (1.3.0, section 4).
  - `rewrite_pair` from the design is not built: it would be unverified.
  - Acceptance per kind is measured only on the small staging runs so far. Read `rejected_by_reason` in each batch's
    `gen_output.verifier_result` before raising batch counts.
- **The media mix has no speech kind.** gen_worker_media has no TTS task kind. The `speech` profile on media-gate was
  being tested by another lane on 2026-10-22 00:01-00:12Z (`tts.customvoice`, `asr.qwen3-asr-1.7b`, cyril tier).
  When it lands, add a speech task kind with share 15 to `media_prompts.MIX_V1`.
  - Update 2026-10-22 (TTS lane): the profile has landed (section 3, "Speech"). Background-key `/v1/audio/speech`
    calls are admitted and preempted like `/prompt` jobs. A speech task kind is still not built: no gen_worker_media
    change was made.
- **Refill prompts have no meaning check before generation.** The media judge checks the outputs before publish.
- **The train lane has no job.** GPU1/GPU2 sit idle until the v2 on-device model's training job is defined and enqueued.
- **Direct ComfyUI ports 28334/28245: closed to the LAN since 2026-10-21 22:40Z** (`comfy-gateway-only.service`; omni_lab run 38052; evidence `evidence/gateway-only-20261021/receipt.json`; iptables backup `backups/gateway-only-20261021T2240Z/`).
  - Before: 28334 answered 200 from the workstation over IPv4 and over IPv6 link-local (`fe80::ae1f:6bff:fe8a:4e00`, docker-proxy on `[::]`).
  - After, from the workstation: IPv4 refused on both ports; IPv6 link-local times out (dropped, the reset does not arrive); media-gate `/healthz`, `/gate/status`, `/system_stats`, `/queue` and `/history` all 200. On the server, `127.0.0.1:28334` and host-originated `192.168.50.222:28334` still answer, so the gate and local tools are unaffected.
  - Only the two serving ports are covered. Other registry ports (P40 instances, stale gauntlet/ICE entries) are unchanged and none of them is running.
  - The GPU4/GPU0 ICE ports 28191/28195 have no live user: their containers exited in early September, Codex's Relay claims on them expired 2026-08-28, and their cards are now media-gate's and strata-llama's. Any new GPU4 work goes through `:28400`.
  - Open the ports again only with `sudo /home/cereal/ai-serve/media/gateway_only.sh remove` (stopping the unit leaves the rules in place). Not tested: re-apply after a Docker restart or a reboot, and the `wg0` path.
- **`gpu4.lock` is not held by media-gate.** GPU4 is protected by the ComfyUI owners file, not by the lane lock.
- **Stale omni_lab rows: fixed 2026-10-21 by the storage lane.** Requeued batches now end `stopped` and the next lease
  lists them (`resumes_omni_lab_runs`); 38016, 38031, 38055 are `stopped`; 12 older runs got lane L-SERVE (backup in
  `/mnt/lexar/omni-campaign/scratch/storage-lane-20261021/backups/`). strata-gate 1.1.1 (passes `--lane L-SERVE`) is on
  disk only; `/healthz` still reports 1.1.0 at 01:15Z, so it takes effect at the next quiet-window restart.
- **Text worker free-space floor: added** (45 GiB on work dir and spool; storage lane).
- **Ownership text: fixed** (`comfy-gpu-owners.json` GPU0/4/6 = `ai-serve`; L-D1 / L-K3 re-registered as ended).
- **Pydantic in two data-path environments (owner rule, section 10 item 5):** `/home/cereal/model-surgery-lab/venv`
  (`pydantic==2.13.5`, via compressed-tensors; `TOOLCALL_PY` for selfgen, finalize and both batch screens) and the
  system `/usr/bin/python3` (`pydantic==2.12.5`; runs the lanes pulse cron and `intake_cron.sh`). Neither path imports
  it. The new `ai-serve` venvs and both speech runtimes are clean (`evidence/pip_freeze_*.txt`).
- **Not measured:** checkpoint reuse across multi-turn chats; the q8_0 vs f16 KV quality A/B; a full 24 h soak.
- **Sprite does not fall back automatically:** there is no automatic retry on Strata after an on-device failure. That would be a phone change on a new branch, coordinated with the Sprite-lead session.

## 10. Hardening review (2026-10-22, 00:55-01:15Z; status updated 06:20Z)

Six lanes (tts, quality, mediaqueue, ports, storage, q4) each had an adversarial review. The reviewers were
read-only and changed nothing. Items marked "checked 01:15Z" were re-read on the server while writing this section.
Status at 01:15Z: strata-gate `/healthz` ok (1.1.0), media-gate ok (1.2.0, profile `general`), and every serving,
worker, speech, timer and firewall unit active. No code fix below had been applied at 01:15Z.

**Status at 06:20Z (fix lanes speech, media-sets, sandbox, pydantic, gpu6, each with its own adversarial review):**
items 1, 2, 3, 4 and 6 are fixed and reviewed; item 5 (pydantic) is open and needs Cyril. Live check 06:17Z:
strata-gate `/healthz` `{"status": "ok", "upstream": "ok", "thermal": "ok", "version": "1.2.0"}`; media-gate
`{"ok": true, "version": "1.2.1", "active": "general", "db": true}`; `gpu6-power`, `strata-gate`, `media-gate`,
`gen-worker-text`, `gen-worker-media`, `media-publish.timer` and `comfy-gateway-only` all `active`; GPU6 at 200 W,
persistence on, 68 C, no throttle reason; `gen_worker_text.json` `paused_kinds: []`.

| # | Defect | Status 06:20Z | Evidence (omni_lab runs; files under `/home/cereal/ai-serve/`) |
|---|---|---|---|
| 1 | Started speech streams end as a clean 200 on failure | **Fixed 02:14Z, reviewed** | fix + e2e 38127; review 38133 (7 attacks, all as specified); old gate fails the new tests 7F/3E |
| 2 | TTS headline run not on deployed code | **Done** | 38127 on gate 1.2.1, 0 restarts during the run |
| 3 | Media sets publish with members missing | **Fixed 01:56Z; review fix 02:23Z** | publishes 38123, 38141, 38168, 38184, 38203 (182 sets, all zone `lexar`); judge job 38131 rejected task 202 under rule `set` |
| 4 | Code sandbox can be switched off | **Fixed 02:25Z; review fixes 03:05Z** | 38102 (pause), 38129 (install + check), 38140 (review); 11/11 probes |
| 5 | Pydantic in two data-path environments | **Open: Cyril's call** | audit 38114 (+ review); recommendation below |
| 6 | Stale qwen-text-to-image skill | **Fixed 01:27-01:29Z, verified** | `SKILL.md:54` "via media-gate ... :28400 only"; script exits 6 without a key |

### Must fix

1. **Speech streams fail silently** (`media/media_gate.py`, the two `return out` branches in the stream handler,
   about lines 1498 and 1502; checked 01:15Z).
   - Once the response is prepared, a preempted, crashed or `bad_audio` stream returns `out` and aiohttp sends a
     normal end. The reviewer reproduced it with the repo's fakes on aiohttp 3.13.5 and 3.14.5: HTTP 200, 960 of
     1440 bytes, no client error, while the gate logged `error`.
   - Effect: Sprite plays cut-off speech with no error, and a streaming background caller gets 200 with partial
     audio, not 503 `preempted`.
   - Fix: abort the connection after a started stream fails (so the client sees an incomplete chunked body), and add
     a fake-worker test for both tiers.
   - **FIXED 2026-10-22 02:14Z** (speech lane): media-gate 1.2.1 (sha256 `54642215...`) + speech_worker 1.3.0
     (`28a47f8f...`), contract `speech/STREAM_CONTRACT.md`. The gate holds its 200 until the first audio chunk; after
     that a failure closes the connection without the closing chunk (`abort_stream`) and records `code` for
     `X-Speech-Outcome`. A preemption now ends the job before `/cancel`, so a worker answer that beats the cancel reply
     is still `preempted` (a race the old code lost: new test fails on it, 'cancelled' != 'preempted'). Tests:
     `media/tests/test_media_gate_speech.py` `TestSpeechStreamFailures` (httpx 0.28.1 and aiohttp clients, sprite /
     cyril / background tiers) and `speech/tests/test_speech_worker_text.py` `TestStreamFraming` (real worker bytes).
     51/51 gate tests on the server (aiohttp 3.14.5) and locally (3.13.5); 13/13 worker tests. The 1.2.0 gate fails the
     new tests (httpx reads a clean 200). Backup of the replaced files: `backups/speech-stream-20261022T0214Z/`.
   - **Adversarial review 02:50-03:01Z (omni_lab 38133, pass; no code change).** On the live gate, with cyril and
     background keys only (no Sprite-tier traffic; quiet 17.8 min before start):
     - clean stream: 200, closing chunk present, outcome `ok` (cold status line at 24.2 s);
     - HTTP/1.0 streaming: `400` in 0.008 s;
     - own `/interrupt` after first audio: httpx `RemoteProtocolError` (incomplete chunked read), code `cancelled`;
     - caller hang-up: code `client_gone`, worker freed (next cyril call served in 2.6 s);
     - background stream preempted after first audio: 200, no closing chunk, code `preempted`; the cyril call got
       200 in 3.6 s;
     - preempted before any audio: `503` `code: preempted`, no 200 first;
     - `kill -9` on `speech-worker@tts` mid-stream: `RemoteProtocolError`, gate code `error` 0.2 s after the kill,
       systemd restarted the worker (1.3.0 healthy).
     - The backed-up 1.2.0 gate run against the new tests on the server: FAILED (failures=7, errors=3); 1.2.1 passes
       10/10. media-gate and strata-gate 0 restarts; only the deliberate kill restarted `speech-worker@tts`.
     - ffmpeg decode of caller audio (the worker's only "untrusted input"): local-file and loopback-URL playlists
       refused, concat refused, loopback listener got 0 hits.
     - Corrected in `speech/STREAM_CONTRACT.md`: `X-Request-Id` is only on 200 answers, and the contract now states
       the cold-start status-line latency (up to about 24 s).
   - **Not verified:** httpx cases skip on the server (no httpx in the venvs; tested locally only); OkHttp behaviour
     in the contract is untested.
2. **The TTS lane's headline run is not on the deployed code.** The review says the final e2e run (omni_lab 38079)
   predates the `media_gate.py` now deployed. Re-run `speech/speech_e2e.py` on the deployed gate after fix 1 and
   quote that run.
   - **DONE:** omni_lab 38127 (pass, 02:15-02:32Z, on gate 1.2.1 with 0 restarts during the run; `speech_e2e` 1.1.0
     with a new `streamcut` phase). Numbers in section 3, "Speech". Receipt
     `evidence/speech-e2e-20261022T0218Z-streamfix/speech_e2e_receipt.json` (workstation `ai-serve/evidence/`, sha
     `80f75371...`): 30/30 clips, WER 5.87 % raw on 494 words (0.40 % with digits spelled out); streamcut: background
     stream got 200 then `ClientPayloadError` after 61,440 bytes, outcome `preempted`. Re-checked by the review
     (hash, 30/30, WER 0.0587).
3. **Media groups can publish with members missing** (`media/gen_worker_media.py` `complete()`,
   `media/media_judge.py` `scan_ready`). The worker spools only the members that passed validation and marks the task
   `done`; the judge builds a group from whatever is in `ready/`. An H3 task whose draft failed could publish its
   audio alone, and an edit or mesh task whose output failed could publish its source image alone.
   - Exposure at 01:15Z: none (the review's DB query found no task with both good and failed `gen_output` rows).
   - **FIXED 2026-10-22 01:56Z (media-sets lane).** `media/media_sets.py` defines each set (H3 = draft + audio +
     refine; edit = source + edit; mesh = source image + mesh). `gen_worker_media` 1.3.0 `complete()` puts a partial
     set in `spool/media/incomplete/` and fails the task; `publish_set` undoes a failed publish fully.
     `media_judge` 1.2.0 `set_checks` rejects an incomplete set with rule `set` before any VL call, and waits on a
     task that is still writing. `media_prompts` 1.0.1 finds edit sources in both zones. Tests: 56 on the server
     (`evidence/media-sets-20261022/tests_server.txt`). Backup `backups/media-sets-20261022T0156Z/`.
   - **Adversarial review, HIGH found and fixed 02:23Z.** A set with an *extra* member could publish, and one did
     reach `ready/`: mesh task 202 held two `mesh_hy3d` GLBs (the lane's own 02:01Z worker restart killed a run that
     ComfyUI finished anyway; the history-less fallback globbed `aiserve/mesh_hy3d/202*`, which also matches tasks
     2020-2029). Fix: `media_sets` 1.1.0 (one output per stage; every member must be a recorded passing output of its
     task), `gen_worker_media` 1.3.1 (per-run prefix `aiserve/<kind>/<task>/<nonce>/`, fallback reads only that
     folder, publish UPDATE also matches `task_id`). 60 tests on the server
     (`evidence/media-sets-review-20261022/tests_server_review2.txt`); the new tests fail 4/4 on the old code. Break
     cases (`break.py`, scratch spool, read-only DB): set missing its GLB or its source -> REJECT/HELD; task 202 ->
     REJECT "has 2 outputs"; a relabelled sibling GLB -> REJECT "not a recorded passing output". Backup
     `backups/media-sets-review-20261022T0223Z/`.
   - **Live:** judge job 02:35Z (review 38131, evals 38134-38137) logged `task 202 mesh_hy3d: reject [... incomplete
     set: mesh_hy3d has 2 outputs ...]`; its files are in `spool/media/rejected/` with `rule: set`. Every publish since
     went to the Lexar zone only: 38123 (48 sets, 02:03Z), 38141 (35, 03:04Z), 38168 (37), 38184 (32), 38203 (30,
     06:02Z); omni_lab `config.sets[].zone` is `lexar` for all 182 (query 06:20Z). Intake picks them up: intake runs
     337-343 (02:30-06:00Z) ingested 181 assets with 0 errors (`$OC/distill/queues/logs/intake.log`).
4. **The code sandbox can be switched off** (`strata/sandbox_exec.py`: `ARMED = [False]` line 42, checked line 48,
   set line 70; checked 01:15Z).
   - Through the real `run_python` on the server, the reviewer disarmed it (one assignment to `ARMED[0]` through
     `sys.modules`, or a frame walk) and then read `strata.env` (`AI_SERVE_PG_PASSWORD`, `STRATA_BACKGROUND_KEY`),
     connected to `docker.sock` and wrote outside the scratch dir. The child runs as `cereal`, who is in the
     `docker` and `sudo` groups.
   - Patching the hook is not enough: it calls `os.path.realpath`, which the program can replace.
   - Fix: OS-level isolation (dedicated uid, no `docker`/`sudo`, no `~/.config/ai-serve`, no network).
   - **Fixed 2026-10-22 02:25Z (sandbox lane).** Programs now run as the `strata-sbx` uid in a transient systemd unit
     through a root-owned sudo wrapper. The unit has no network and a read-only minimal view, so neither `/home` nor
     `strata.env` is in it, and it has limits. Code batches start only after the isolation check passes, and the
     check cannot be switched off. The same one-line disarm now reaches nothing: 10/10 probes and 7/7 isolation
     tests pass (section 4, "Code sandbox"). The design reuses the agentic IT platform's least-privilege rule
     (`SOC_TESTING/soc-dashboard` `agent_harness.py`: fixed uid, groups cleared, no sudo) and adds what it lacked.
     - The code kinds were paused 01:24:31Z-02:25:36Z: gen-worker-text was stopped 01:24-02:01Z, then ran with
       `paused_kinds` until 02:25Z. Task 381 (`instruction_code`) was stopped mid-batch, journaled and requeued.
       omni_lab runs: 38102 (pause), 38129 (install + check).
     - **Adversarial review 03:05Z (omni_lab 38140):** the isolation held (secrets, network, writes, setuid/namespace/
       mount/bpf/ptrace escalation, leftover double-forked child, one-at-a-time ordering;
       `evidence/adv-sandbox-20261022/run_adv.log`). Wrapper 1.1.0 closes io_uring (`EPERM`), exec from `/work`
       (noexec), the visible `utempter` helper, a SIGTERM-ignoring child holding the slot (`TimeoutStopSec=2`) and the
       lock gap. `sandbox_exec` 2.1.0 adds an 11th probe (`kernel_surface`); 11/11 pass.
     - **HIGH found and fixed in the review:** the verifier (`text_tasks.run_tests`) counted the solution's own `PASS`
       prints, so a wrong solution was accepted. text_tasks 1.2.0 reads a random per-run marker per test index; the
       27 `instruction_code` records accepted before it re-verify with 0 false accepts (`reverify_code.json`). The
       deployed text_tasks is 1.3.0 (capture lane, 06:17Z check) and keeps the per-run marker (`run_tests`, line 359).
     - **Live 06:17Z:** `paused_kinds: []`; code batches pass on the isolated path: 38130 (task 382), 38132, 38138,
       38142, 38170, 38188. Two ended `stopped`: 38148 (task 384, 03:42Z) and 38171 (task 376, 04:41Z, during the
       capture deploy 38176); not investigated in this update.
     - **Still open (low):** a program written to read the marker from its own `/work/program.py` could still forge a
       pass, since it runs in the same process as the test harness.
5. **Pydantic is installed in two data-path environments** (owner rule: none in any environment). Checked 01:15Z:
   `model-surgery-lab/venv` has `pydantic==2.13.5` (via compressed-tensors; text worker `TOOLCALL_PY`), and the
   system `/usr/bin/python3` has `pydantic==2.12.5` (lanes pulse cron, `intake_cron.sh`). Neither path imports it.
   - **OPEN: needs Cyril.** Audit omni_lab 38114 (receipt
     `evidence/pydantic-audit-20261022/receipt_lane_pydantic.json`, sha `f572ac57...`); the review re-checked it and
     changed nothing. Nothing was uninstalled.
   - `model-surgery-lab/venv`: pulled in 2026-09-01 by compressed-tensors 0.18.0 (K3 install). The text worker runs
     identically without it (hardlinked venv copy: data files byte-identical, all exits 0, no import attempt). But
     `k3/harness/k3_common.py` `mxfp4_reference_check` loads compressed_tensors -> pydantic: with it
     `available: true, bit_exact: true`; without it `available: false` (the K3 4-bit decode reference becomes a gap).
     Recommendation: freeze the library's reference output as a fixed value in the K3 harness test, then uninstall
     pydantic, pydantic-core and compressed-tensors together between text batches.
   - System Python: `pip --user` install in `~/.local` (2026-04-07, not apt), a hard dependency of fastapi 0.135.3,
     web3 7.15.0, eth-account 0.13.7, eth-utils 6.0.0 (likely users: benchmark sources under `~/ai-benchmark-lab`,
     `~/.appworld`). Nothing on our data path imports it; `PYTHONNOUSERSITE=1` was shown safe for the three cron
     lines and `siem-ticket-bridge`. Recommendation: add `PYTHONNOUSERSITE=1` there; for zero pydantic, move the five
     packages into their own venv rather than deleting them from `~/.local`.
   - No SQLAlchemy/ORM installed or imported anywhere in `ai-serve`, `distill/tools`, `lab`, `agent-memory-backup` or
     the workstation tree; none of the 32 live Python processes has pydantic or `~/.local` loaded.
   - Section 11.7 item 6 (the capture export's held-out screen runs in this venv) depends on the same decision.
6. **A stale skill points agents at the closed port.** `C:/Users/cereal/.agents/skills/qwen-text-to-image/SKILL.md`
   line 27 listed `comfyui_qwen21_gpu4` at `:28334` (checked 01:15Z), and `qwen21_t2i.py` sent no `X-Serve-Key`.
   - **FIXED 01:27-01:29Z, verified by the pydantic lane and its review.** `SKILL.md:54` now reads "via media-gate
     http://192.168.50.222:28400 only (+ `X-Serve-Key`)" (re-read 06:20Z); `qwen21_t2i.py` sends `X-Serve-Key` from
     `MEDIA_GATE_KEY` (default empty) and exits 6 without one; with the `media_key_cyril` key a 512x512 8-step image
     came through the gate in 5.1 s. Five more skills got a "current route" section (qwen-image-editing,
     text-to-music, ltx-video-gen, comfyui-validation, comfyui-workflow-ops; the last converted to LF).

### Also done in this round (GPU6, Q4)

- **GPU6 (RTX 8000) power cap: 200 W, persistent** (Cyril 2026-10-22: "[Cyril quote withheld]"). Only `nvidia-smi -pl` / `-lgc` inside the card's own
  limits (150-250 W, 300-1620 MHz); no BIOS, firmware or voltage change. Sweep review 38149 with the decision rule
  registered before the first measurement; probe runs 38150, 38155, 38160, 38169, 38173, 38177, 38178, 38185, 38187;
  `evidence/gpu6-tune-20261022/sweep.json` (sha `07d1d267...`).
  - 200 W: 100.6 tok/s at 6 streams, 47.2 tok/s single (server), 76-77 C, 0 % thermal slowdown. Stock throttled in
    one of its two runs (27.8 % of samples, SM down to 1140 MHz, 81-82 C); against the unthrottled stock run 200 W is
    1.1 % slower at 6 streams and the same single-stream. The cap buys no throttling and 4-5 C, not speed.
  - `gpu6-power.service` (omni_lab 38194) sets it at boot by UUID with persistence mode, after `gpu-unlock-verify`,
    before `strata-llama` (which does not depend on it: a failure leaves the card stock). Revert:
    `sudo systemctl disable --now gpu6-power`.
  - strata-gate `--bg-max 2 -> 4` (omni_lab 38195; unit `ExecStart ... --bg-max 4`, checked 06:17Z).
  - Review 38204: fixed `e2e/gpu6_tune.py` 1.0.1 so a sweep's end restores `gpu6-power` instead of stock.
  - **Not tested:** a real reboot; 4 background streams at once (background demand never exceeded 2 while watched).
- **Q4 closed** (Cyril 2026-10-22: with Q3 at about 50 tok/s single and 100 tok/s at 6 parallel, "[Cyril quote withheld]"). IQ3_S stays. IQ4_XS shards deleted 04:31Z (97.5 GB; omni_lab 38172, deletion manifest under
  `evidence/q4-20261021/`); `ls /home/cereal/models/gguf/*IQ4*` finds nothing (06:17Z). Left behind: the staged,
  unused `strata/q4_swap/` (`q4-iq4xs.conf`, `strata_llama.sh`).

### Statements corrected

- `media-publish.timer` is enabled and active (since 00:37:24Z), not inactive (sections 2 and 9).
- Streamed speech did not get 503 `preempted` or 502 `bad_audio` once it had started (section 3; item 1; fixed 02:14Z).
- The sandbox was not a second layer before 02:25Z (section 4; item 4; now OS-isolated).
- GPU6 runs at a 200 W cap with `--bg-max 4`, not stock with `--bg-max 2` (sections 2 and 9 predate 05:45Z).
- The Q4 swap is cancelled, not pending (section 9).

### Worth fixing next (not blocking; status 06:20Z)

- **Lenient CLAP gate** (`media_judge_spec.json`): a rank of 7 or better out of 25 passes, so an unrelated clip passes
  by chance about 28 % of the time. For instrumental music and `h3sol_audio`, CLAP is the only meaning check.
  Changing the threshold means registering a new judge spec. *Open.*
- **An edit's "before" image need not have passed the judge** (`media_prompts.py` picks unjudged `ready/` images).
  *Open:* `media_prompts` 1.0.1 only made the source search cover both zones.
- **A hold that keeps failing the same way is retried every hour** and reloads the VL model each time. *Open.*
- **Firewall rule scope** (`media/gateway_only.sh`): the DOCKER-USER reject matches any destination on ports
  28334/28245, not only this host. Add `--ctorigdst 192.168.50.222`. *Open* (no `ctorigdst` in the script, 06:17Z).
- **The firewall fails open without an alert.** A malformed `media_gate.json` gives an empty port list; the unit fails
  and the ports stay open after a reboot. Add `OnFailure=` or a check in `media-gate-health`. *Open* (no `OnFailure`
  in `comfy-gateway-only.service`, 06:17Z).
- **`comfy-gateway-only` can read inactive while its rules are loaded** (Docker stopped, then started), and there are
  a few seconds at boot before the rules load. *Open.*
- **Quiet windows:** the ports lane measured its window on media only (strata-gate had a cyril request 11 min before
  the change: within the rule, barely). The mediaqueue lane restarted the two background workers without a 10-minute
  quiet window (no gate or ComfyUI touched). The `comfy-gateway-only` restart at 00:50:44Z came under a minute after
  Sprite strata traffic; it only adds rules and interrupted nothing. This round: the speech deploy waited 12.9 min
  quiet; the media-sets lane restarted only `gen-worker-media` (cut background mesh task 202, which caused the HIGH in
  item 3).
- **Sandbox marker forge** (item 4, low) and the two `stopped` code batches (38148, 38171). *Open.*

### Formerly "Not done"

- **Media publish to the Lexar: DONE** (lead decision; item 3). NAS 42 G free (06:17Z), still under its 50 GiB floor;
  the Lexar zone has 453 G free (89 % used).
- **Text-worker line endings: DONE.** `gen_worker_text.py` 1.4.0 is saved LF (`file`: "Python script, ASCII text
  executable", 06:17Z).
- **q4 lane: CLOSED** by Cyril (see "Also done").

### Needs Cyril

- **Pydantic** (item 5): approve the recommendation (freeze the K3 reference, then remove pydantic + pydantic-core +
  compressed-tensors from `model-surgery-lab/venv`; `PYTHONNOUSERSITE=1` for cron/siem jobs; move the five `~/.local`
  packages to their own venv), or exempt in writing. Section 11.7 item 6 rides on this.
- **NAS space (longer term):** media now goes to the Lexar (453 G free); the NAS stays at 42 G under its floor.
- **GPU6 (optional):** 200 W is applied. 175 W runs 5-6 C cooler (70-71 C) for about 2 % less 6-stream throughput,
  if he prefers margin; better airflow on the passive card is the only fix beyond a cap.
- ~~Code-running text kinds while the sandbox is open~~: resolved 02:25Z (item 4).
- ~~Q4 A/B and swap~~: Cyril, 2026-10-22: don't use the Q4. Closed.
- ~~GPU6 `--bg-max` and power~~: Cyril approved a safe cap; 200 W + `--bg-max 4` applied 05:45Z.
- ~~Speech placement~~: decided 2026-10-22 (shared GPU4 for training data, no P40; Sprite on device by default; the
  server TTS is an optional fallback; section 9).

## 11. Training-data capture of Strata traffic (2026-10-22, capture lane)

Cyril (2026-10-22): "[Cyril quote withheld]"

**State (Sprite headers, 2026-10-22 07:12Z; details in 11.8):**
- strata-gate 1.2.1 (sha `6aa7fc3f...`), `capture.py` 1.2.0 (`584d3648...`) and `capture_admin.py` 1.2.0 (`6f679ff3...`)
  are live since the quiet-window restart at 07:11:59Z (980 s quiet, healthy at 07:12:14Z). `capture_export.py` 1.1.1
  is unchanged.
- Migration `0008_capture_opt_out.sql` (`446fef42...`) was applied at 07:11:54Z. It adds one table; no existing row
  changed.
- The gate honours Sprite's per-request opt-out `X-Sprite-Capture: off`, and checks every `X-Sprite-*` header strictly,
  from the Sprite key only.
- Backup (server): `backups/sprite-headers-20261022T071154Z/` (the 1.1.1 files, a COPY of `capture_member`, receipt,
  this runbook before the edit). Workstation: `ai-serve/backups/sprite-headers-20261022-local/`.
- Evidence: `evidence/sprite-headers-20261022/`.
- Rollback: copy the `strata/` files back from that backup, delete `strata/tests/test_sprite_capture_headers.py`, and
  restart strata-gate in a quiet window. 0008 can stay: nothing else reads the table.

**State (adversarial review, 2026-10-22 05:20-06:25Z):**
- `capture.py` 1.1.0 (sha `7a742cb4...`; CRLF line endings by mistake, harmless, gone in 1.1.1) is live in
  strata-gate since 05:43:46Z (quiet-window restart; the gate file is unchanged, 1.2.0 sha `c0781ae7...`). 1.0.0 ran
  04:32:48-05:43:41Z.
- `capture_export.py` 1.1.1 (`7e9bc28e...`) and `capture_admin.py` 1.1.1 (`025772c1...`) are installed since
  06:00Z (a oneshot job and a CLI: no restart needed; the exporter re-installed at 06:09Z with LF line endings).
- `capture.py` 1.1.1 (`870807d7...`), staged in `staging/capture-review-111/` (server suite 128 OK / 1 skipped), went
  live at 06:33:32Z.
  - The detached waiter (`staging/cr-tools/run_all_111.sh` -> `capture_review_deploy_111.py`) found 617 s of quiet and
    restarted strata-gate. It was healthy with capture 1.1.1 at 06:34:03Z (`deploy_receipt_111.json`).
  - The live probe was captured, and none of its four planted secrets was stored (`probe_check_111.json`).
  - 1.2.0 superseded it at 07:11:59Z (above).
- Migration `0007_capture_not_target.sql` was applied at 05:37:55Z (only the CHECK on `capture_export_item.decision`
  changed; 240 rows untouched).
- Evidence: `evidence/capture-review-20261022/` (test logs, deploy receipts, live probe, completeness, intake
  extractor check). Backups (server): `backups/capture-review-20261022T053755Z/` (1.0.0 files + a COPY of
  `capture_export_item`), `backups/capture-review-20261022T055801Z/` (the 1.1.0 files) and
  `backups/capture-review-20261022T061024Z/` (what the 1.1.1 waiter rolls back to); workstation:
  `ai-serve/backups/capture-review-20261022-local/`.
- Rollback: copy the `strata/` capture files back from one of those backups and restart strata-gate (0007 can stay:
  it only widens the CHECK), or restart with `--no-capture`.
- Note: the gpu6 lane restarted strata-gate at 05:44:04Z with `--bg-max 4` in the unit; that unit change is theirs.

### 11.1 What is stored, where, and for how long

| Where | What | Access |
|---|---|---|
| PostgreSQL `ai_serve`, `capture_request` | the request as sent upstream: messages (a long system prompt as a `capture_blob` ref), params, model alias, tier, member, source, gen task, provenance (gate/capture version+sha, llama build blob, Sprite headers) | `ai_serve` role only; `ai_serve_ro` has no access |
| `capture_response` | content, `reasoning` (reasoning_content), tool_calls, finish_reason, usage, timings, slot, outcome, `partial`, redaction counts | same |
| `capture_blob` | tools catalogs, long system prompts, llama build records, each once by sha256 | same |
| `capture_member` | consent and export flag per member (`cyril`, `background`, `sprite:<pseudonym>`, `sprite:unmarked`) | counts/flags only |
| `capture_opt_out` (0008, capture 1.2.0) | for `X-Sprite-Capture: off` requests, a counter only: member, UTC hour, request count, `opt_out = true`. Nothing else of those requests is stored anywhere | `ai_serve` role only; revoked from `ai_serve_ro` |
| `capture_export`, `capture_export_item` | every export batch and the decision about every capture (no text; a MinHash signature) | |
| `/home/cereal/ai-serve/spool/capture-pending/` (0700, files 0600, 2 GiB cap) | `pending-*.jsonl`: redacted records waiting for PostgreSQL (replayed every 30 s); `rejected-*.jsonl`: records PostgreSQL refused (kept for a look, never replayed); `*.bad`: unreadable spool files set aside | user `cereal` |
| Lexar `/mnt/lexar/omni-campaign/corpus-store/chimera-intake/text/strata-capture-<date>-<nnn>-<text\|tools>/` | exported records only: cleared member, labelled source, trainable target, passed the screens | exFAT: no owner-only files |

**Retention: there is no automatic expiry anywhere yet.** Captures stay in PostgreSQL until `delete-member` removes
them; spool files stay until replayed (pending) or removed by hand (rejected, .bad); exported Lexar batches are corpus
assets and are not rewritten. Size so far: 3.0 MB of tables for 484 captures at 05:47Z (`capture_admin.py stats`).
A retention period for household captures that are never cleared for export is a decision for Cyril (11.7 item 7).

### 11.2 Opting a member out, deleting a member, clearing a member for training

```
cd /home/cereal/ai-serve/strata && export AI_SERVE_ENV_FILE=/home/cereal/.config/ai-serve/strata.env
PY=/home/cereal/.venvs/ai-serve/bin/python
$PY capture_admin.py members                    # member keys, consent, export flag, captured/exported/opt-out counts
$PY capture_admin.py stats --hours 24           # counts by tier/source/outcome, reasoning chars, table size, opt-outs
$PY capture_admin.py opt-out --member sprite:<x> --by "who decided, their words"   # stops capture at once
$PY capture_admin.py opt-in  --member sprite:<x> --by "..."
$PY capture_admin.py set-export --member sprite:<x> --on --by "Cyril: <his words>"  # allow training export
$PY capture_admin.py delete-member --member sprite:<x> --by "..."          # dry run: rows + batches it would hit
$PY capture_admin.py delete-member --member sprite:<x> --by "..." --yes    # hard delete, member stays opted out
```
- **Opt-out** takes the exclusive capture lock; the gate's writer checks consent inside every insert transaction under
  the shared lock, so nothing of an opted-out member is written after the command returns (spooled records included:
  replay checks again). Since 1.1.0 an opt-out works before the member's first request (the row is created opted out).
- **Per-request opt-out (1.2.0):** when a member turns capture off in Sprite settings, the Sprite backend sends
  `X-Sprite-Capture: off`.
  - The request is served as usual and nothing of it is stored. Only `capture_opt_out` counts it per member and hour
    (`members` shows `opt_out_requests`; 11.8).
  - An opt-out made here (`opt-out`) is the stronger one: that member gets no counter either.
- **Delete** removes the member's `capture_request`/`capture_response` rows, the blobs only they used and (1.2.0)
  their `capture_opt_out` counter rows. It keeps the member row as `opted_out`, and lists the export batches that
  already carried their rows.
  - Those batch files are in the Lexar zone and the intake, and must be removed there by hand.
  - Household members are never exported until Cyril clears them, so for them the list is empty unless he did.
- **Member keys (capture_admin 1.2.0):** `cyril`, `background`, `sprite:unmarked`, or `sprite:<24 hex>`. The 24 hex
  are the Sprite backend's HMAC pseudonym; hex is lower-cased, and any other form is refused, so a typo cannot create a
  second, wrong member.
- **Per-person opt-out needs the Sprite backend to send `X-Sprite-Member`.** Sprite commit e9cfef3 sends it. The stage
  `:23499` ran `claude-20261019` without that commit at 07:15Z, so stage traffic is still one member, `sprite:unmarked`
  (held from export). Opting out `sprite:unmarked` stops capture of all unmarked Sprite traffic.

### 11.3 How exports reach the corpus

1. `strata-capture-export.timer` (hourly at :20) runs `capture_export.py run` (nice 10, idle I/O).
2. It reads undecided captures of cleared members whose source is labelled exportable, oldest first, by keyset pages
   (1.1.0: rows that can only be held are filtered in SQL, so they cannot fill the page and starve later captures;
   `held_unlabelled` in the run summary counts them by tier:source).
3. Each becomes a `strata-capture-conv-v1` record: the conversation as Strata saw it, every earlier message marked
   context (`origin` strata/unknown/tool/client, `target: false`), Strata's own answer last (`target: true`,
   reasoning in `reasoning_content`), the tools catalog, provenance, and (1.1.0) a top-level `response`.
4. **Only trainable answers are exported** (1.1.0): outcome ok, a finish reason that is not `length`/`limit`, not
   empty, accepted by the batch verifier for idle-time generation. Partial, preempted, cut, rejected or empty answers
   are decided `not_target` and stay in PostgreSQL.
5. Every record is redacted again with the current rules, then screened: `text_batch_screen.py` (held-out index +
   review surface, incl. tool names, call arguments and catalogs; refuses the batch above a 20 % hit rate), then exact
   and MinHash near-duplicate removal against every earlier export.
6. The batch goes to the Lexar zone (`strata_capture.jsonl` + `.sha256` + `_meta.json` + `.provenance.json`), with a
   `capture_export` row, a `gen_task`/`gen_output` row and an omni_lab `export` run on lane L-SERVE.
7. `intake_cron.sh` (every 30 min) runs `intake.py` over the drop zones. Its text index keeps instruction/response per
   row (`_record_row` -> `distill_corpus_build.extract_*`): **response = the record's `response`** = Strata's target
   rendered as the model emits it (`<think>`reasoning`</think>`, content, `<tool_call>` blocks); instruction = the
   system + user turns. The full record (context flags, tool results, provenance) stays in the stored asset file.

### 11.4 History: the 1.0.0 deploy (04:32:48Z) and its design

**Live since 04:32:48Z:** strata-gate 1.2.0 (sha `c0781ae7...`) with `strata/capture.py` 1.0.0. Migration
`0006_capture.sql` was applied at 04:09Z. The `strata-capture-export.timer` runs hourly at :20. The restart also
brought in the 1.1.1 lane change. Evidence: `/home/cereal/ai-serve/evidence/capture-20261022/`. Backup of the replaced
files: `backups/capture-20261022T040912Z/`.

**What is captured.** Every request that reaches llama-server, from every tier (sprite, cyril, background), on
`/v1/chat/completions`, `/v1/completions` and `/completion`. Requests the gate refuses (auth, `busy`, `no_vision`,
`starting`, `thermal`) keep their `request_log` row only. For each request:
- **Request:** the body as sent upstream, the model alias the client asked for, the tools catalog and a long system
  prompt (each stored once by sha256 in `capture_blob`), and the params.
- **Answer:** content, `reasoning_content`, tool calls (streams rebuilt by index), finish reason, usage, timings,
  slot and outcome.
- **Partial answers:** a preempted, thermal, timeout, client-gone or error answer is kept with `partial = true`.
  It is never a training target.
- **/completion prompts** (token ids) are turned into text through llama's `/detokenize`, which uses no slot.
- **Provenance:** gate and capture version and sha, the llama build (`/props`: build_info, model path, template sha),
  `X-Client-Source`, `X-Workflow-Id`, `X-Gen-Task`, `X-Gen-Kind`, and the Sprite headers (below).
- Every answer carries `X-Request-Id`, which equals `request_log.request_id` and `capture_request.request_id`.

**Never in Sprite's way.**
- The hook only keeps references and calls `put_nowait` on a bounded queue (2,000 items, 256 MiB). Measured cost:
  about 13 µs per request.
- Reassembly, redaction and zlib run in one spawned child process (nice 10). The same work in a thread of the gate
  added about 19 ms to every Sprite-shaped request through GIL contention (measured on the workstation, then fixed).
- Capture work waits while a sprite or cyril request is running, for up to 30 s per group of 8.
- Writes use the capture's own two-connection asyncpg pool. With PostgreSQL down, redacted rows go to a 0600 spool
  (`/home/cereal/ai-serve/spool/capture-pending/`, 2 GiB cap) and are replayed later. A stopping gate drains its
  queue there.
- Every failure is counted, never raised. The counts are in `/gate/status` (`capture`) and in `service_event` kind
  `capture`, once a minute.
- `tests.test_capture.TestLatency` on the server (final run, `evidence/capture-20261022/test_suite_final.log`), with
  35 KB / 63-tool Sprite bodies through the real gate in alternating blocks of 40:
  - p50 10.01 ms off vs 9.94 ms on, and p95 14.93 vs 13.76 ms;
  - event-loop lag while the child prepares 100 captures back to back: max 6.9 ms, p99 4.3 ms (idle: max 1.9 ms,
    p99 1.8 ms). That burst only happens while no user request runs.
  - Earlier runs showed a 36-86 ms maximum. That was the test building 100 bodies on the event loop, not the capture
    (fixed in the test).

**Redaction, before anything is stored or spooled.**
- Removed: bearer tokens, `sk-`, JWT, AWS, GitHub, Slack, Google and Stripe keys, PEM private keys, URL credentials,
  password, passcode, PIN, OTP and CVV values, credential-named JSON fields, and long high-entropy strings (upper,
  lower, at least 2 digits, at least 4.2 bits per character; hex such as shas and uuids is kept).
- Sensitive tools (sign-in, vault, credential, OTP and 2FA, plus `sensitive_tools_pattern`) have their arguments and
  results blanked.
- Emails and phone numbers are kept (`redact_contacts: false`).
- No secret value is ever loaded: redaction works by shape.
- Each row stores counts by kind in `redactions`. Planted-secret and keep-ordinary-text tests are in
  `TestRedaction`.
- 1.1.0 added the shapes 1.0.0 missed (env-style names, passphrases, short codes, more token kinds, prefixed JSON
  fields): 11.5 item 2.

**Members and consent** (`capture_member`; `strata/capture_admin.py`). All of these are defaults written by the
capture lane, stated in `set_by` and in `capture_policy.json`, for Cyril to confirm or change:
- `cyril`, `background`: captured and exportable.
- `sprite:<X-Sprite-Member>` (an HMAC pseudonym from the Sprite backend), or `sprite:unmarked` when the header is
  missing: captured, **not exported for training** until Cyril runs `set-export --on` for that member.
- `opt-out` stops capture at once: the writer checks it inside each insert transaction, under an advisory lock that
  the admin command takes exclusively. `delete-member --yes` hard-deletes the member's rows and the blobs only they
  used, keeps the member opted out, and lists any export batches that already carried their rows (those must be
  removed from the intake by hand).
  ```
  cd /home/cereal/ai-serve/strata && AI_SERVE_ENV_FILE=/home/cereal/.config/ai-serve/strata.env /home/cereal/.venvs/ai-serve/bin/python capture_admin.py members|stats
  ... capture_admin.py set-export --member sprite:<x> --on --by "Cyril: <his words>"
  ... capture_admin.py opt-out|opt-in|delete-member --member <key> --by "..."
  ... capture_admin.py label-source --tier cyril --from unknown --to cyril-manual --since <ISO> --by "..."
  ```
- `ai_serve_ro` has no access to `capture_request`, `capture_response` or `capture_blob`.

**Export** (`strata/capture_export.py`, hourly). Captures become `strata-capture-conv-v1` conversation records:
- **Messages:** reasoning is kept apart in `reasoning_content`, tool calls are kept, the tools catalog is kept, and
  each message carries `origin` and `target`.
- **Targets:** only Strata's own answer to that request, and only when it is complete (ok, with a finish reason that
  is not `length`/`limit`). For idle-time batches it must also be accepted by the batch's verifier (text kinds: the
  gateway request ids in the journal; `toolcall_conv`: the seeds of accepted conversations).
- **Earlier assistant turns** in a request are context only. They are `origin: strata`, linked by `message_sha256`
  with that capture's reasoning, or `origin: unknown` (phone model, cloud, Codex).
- **Held** (looked at again every run):
  - members not cleared for export;
  - unlabelled sources (cyril `unknown`, `background-unlabelled`, `sprite-unmarked`);
  - generation tasks that are still open;
  - conversations with third-party tool results (`sms_*`, `messages_*`, `contacts_*`, `notifications_*`, `email_*`,
    `calendar_*`, `files_read`, `ui_screenshot`);
  - conversations with delegated cloud/Codex output (`delegate_*`).
- **Final:** Claude/Codex/test/bench traffic (`blocked_source_pattern` on the source or the workflow, for example
  `gpu6-tune`) is `excluded_source`, never exported.
- **Screens:** `text_batch_screen.py` 1.1.0 (now also reads tool names, tools catalogs and call arguments; unchanged
  for text_tasks rows, re-checked on the task 396 batch), then exact and MinHash (5-gram, Jaccard 0.9) dedup against
  every earlier export.
- **Output:** Lexar zone only, as
  `/mnt/lexar/omni-campaign/corpus-store/chimera-intake/text/strata-capture-<date>-<nnn>-<text|tools>/`, with
  `strata_capture.jsonl`, `.sha256`, `_meta.json` (route `text` or `tool_native`) and `.provenance.json`.
- **Records:** one `capture_export` row, one `gen_task` (kind `strata_capture_export`, status done) plus `gen_output`
  row, and one omni_lab `export` run on lane L-SERVE per batch (1.1.1: no omni_lab run, no batch).
- Superseded by 11.3 where they differ: 1.0.0 also wrote non-trainable records and had no top-level `response`.

**gen-worker-text keeps the traces** (text_tasks 1.3.0, text_refill, selfgen_via_gate; worker restarted 04:19Z with
no task leased):
- Every request sends `X-Client-Source gen-worker-text`, `X-Gen-Task`, `X-Gen-Kind` and
  `X-Workflow-Id text_tasks.<kind>.<role>`. The journal's `requests` are now the gateway ids; llama ids are in
  `completion_ids`.
- Kept answers carry `reasoning_content` whenever Strata produced one.
- Every second new `instruction_code` / `long_context` batch has `solver_model: strata-think`, so its reasoning is
  generated (max_tokens 8192 / 4096).
- Author calls and rejected items are not lost: the gateway capture holds them. Since capture_export 1.1.0 they stay
  in PostgreSQL (`not_target`) and are not written to the intake.
- `toolcall_conv` still renders with thinking off: the pinned `toolcall_selfgen` was not changed.

**Live check (04:32-04:59Z, counts only; `evidence/capture-20261022/live_verify.json`):**
- 146 requests reached upstream and all 146 were submitted to the capture. 140 were written by 04:59Z; the other 6
  were deferred behind the six running cyril streams (queue 5, no drops, no errors).
- Token counts agree with `request_log` in 140 of 140.
- All were `gpu6-tune` bench requests (cyril tier, `finish_reason` `length`). The first export runs marked them
  `excluded_source`.
- **Not seen live yet:** background, Sprite or reasoning captures. The bench held the foreground for the whole window,
  so no background was admitted, and no Sprite request arrived. `staging/capture_live_verify.py` (detached) re-checks
  every 2 min for up to 4 h and stops once background captures with reasoning appear.

**Incidents during the deploy.**
- `tests/test_strata_failures.py` builds a gate with `db=True`, which enabled the capture against production
  credentials. One fake-llama "sprite" chat was spooled into the production spool at 04:11Z, and the new gate
  replayed it at 04:33Z. It was deleted (copy kept in `evidence/capture-20261022/leaked_test_row_backup.json`). That
  test now sets `capture=False`.
- The quiet-window check read `request_log`, which gets its row only when a request finishes. The restart at
  04:31:49Z therefore came 6 s after a `gpu6-tune` cyril request started. The old gate drained for 59 s: all 6
  in-flight cyril requests finished `ok`, but new connections were refused from 04:31:49 to 04:32:48. There was no
  Sprite traffic (the last Sprite request was at 02:01Z). Next time, read `/gate/status` active and waiting counts as
  well.
- While polling for the window, an SSH session held the workstation's shared SSH lease for about 10 minutes before
  it was killed. Use a detached server-side wait, as was done for the restart.

### 11.5 Adversarial review (2026-10-22 05:20Z on): what broke, what was fixed

Items 1, 2, 3 and 5 were reproduced on the server (1 with the intake's own extractor, 2 by planting secrets, 3 against
the real PostgreSQL: `nul_surrogate_check.txt`, SQLSTATE 22021 / 22P05 / 22000, 5 seen live); 4, 6 and 7 are from
reading the code. Every item has a test in `tests/test_capture.py` (44 capture tests), plus
`tests/test_capture_review.py` (a client hanging up mid-stream is stored partial with what was said; a full spool
and a disk full mid-write drop and count, leave no torn file, and serving still answers 200; passes on the live 1.1.0
tree and the staged 1.1.1 tree, `test_capture_review.log`). Server suites on the staged trees: 1.1.0 122 OK /
1 skipped (`evidence/capture-review-20261022/test_suite_final.log`), 1.1.1 128 OK / 1 skipped (`test_suite_111.log`,
the 3 review tests included). The skip is the intake-extractor test, which needs numpy (not in the ai-serve venv);
the same check ran in the intake's own venv (`dcb_check.json`, `batch_check.json`).

1. **Critical: every exported record would have entered the intake with a garbage response.** The intake's text index
   (`intake.py` `_record_row` -> `distill_corpus_build.extract_response`) takes a top-level `response` first. 1.0.0
   records had none, so it fell to the `target` metadata dict and indexed its JSON (`{"finish_reason": ..., "index": 4,
   "trainable": true, ...}`) as the response. Without that dict it joins every assistant message, so cloud/Codex
   context turns and partial answers become responses. Run with the intake's own interpreter
   (`chimera-intake` venv): `dcb_check.json`. Fix: `response` = Strata's target only; `not_target` records are not
   exported (migration 0007). No batch had been written yet (the Lexar zone held only `_README.txt`).
2. **High: secrets in common shapes were stored verbatim.** Planted and checked:
   - `DB_PASSWORD=`, `POSTGRES_PASSWORD=`, `PGPASSWORD=`, `OPENAI_API_KEY=` (a leading `\b` never matches after `_`);
   - `X-Serve-Key: <hex>`, `Authorization: Token <hex>`, `Authorization: Basic <short>`;
   - unquoted passphrases (`password: correct horse battery staple` kept 3 of 4 words);
   - OTPs and door codes (`482913 is your Google verification code`, `Your code is 771204`,
     `the passcode for the front door is 7741`), `pw: ...`;
   - GitLab `glpat-`, Hugging Face `hf_`, xAI, npm and Telegram tokens; a 32+ hex value named as a key or token;
   - JSON fields with a prefix (`db_password`, `newPassword`, `x_api_key`, `apiToken`, `user_pin`).
   All are redacted now. Code stays readable: `def check_password(password: str)`, `user.password == hashed` and
   `password = os.environ.get(...)` are kept (`TestRedaction.KEPT`). Exports re-run the current rules on every
   record, so rows captured under 1.0.0 leave with today's redaction.
3. **High: one bad value cost a whole batch, then blocked the spool forever.** NUL or a lone surrogate in a message
   (both legal in JSON) made PostgreSQL refuse the batch of up to 8; it was spooled, and the replay stopped at the
   first failing file on every round, so every later spool file waited behind it. Now: NUL/surrogates are scrubbed
   in `prepare`; a data error (SQLSTATE 22/23) retries record by record and only the bad one goes to
   `rejected-*.jsonl`; an unreadable spool file is renamed `.bad` and the replay continues.
4. **High: a hung capture child would have stopped all capture until restart.** No timeout on the child call. Now
   120 s, then the child is killed and restarted and the group retried one request at a time.
5. **Medium, seen live: a restart lost the batch the writer had in hand** (only the queue was drained). The 1.0.0 ->
   1.1.0 restart at 05:43:41Z lost 2 background answers that had finished at 05:43:30 and 05:43:38Z
   (`c359a237...`, `62d87d24...`; `completeness.json`): everything else since 04:32:48Z was captured. Root cause in
   the journal: systemd's stop SIGTERMs the whole cgroup, so the capture child died mid-group (`BrokenProcessPool` at
   05:43:43Z) and 1.0.0 counted the group lost. 1.1.0 drains the in-flight batch and retries a dead child's group one
   by one in a fresh child; the gpu6 lane's restart at 05:44:04Z hit the same SIGTERM and 1.1.0 lost nothing ("retrying
   1 one by one", then `written 4`). 1.1.1 also makes the child ignore SIGTERM and prepares the remainder in the gate
   process during a stop. Whether the 1.1.1 restart loses anything: run `staging/cr-tools/completeness.py` after
   it (restart time in `deploy_receipt_111.json`; requests that finished just before it must all be in
   `capture_request`).
6. **Medium: held rows could starve the export.** Undecided rows that can only be held (cyril source `unknown`, open
   gen tasks) were fetched first, oldest first, `LIMIT 5000`; past 5,000 of them nothing newer would ever export. Now
   filtered in SQL, keyset paging.
7. **Medium: the queue's byte cap under-counted streams 4-5x** (256 B per parsed chunk; measured ~1.2 KB), and a long
   foreground run (the `gpu6-tune` bench kept six cyril streams open for hours) let capture make one group per 30 s.
   Now 1 KiB per chunk, and with the queue a quarter full the wait for user requests is at most 2 s.
8. **Low: opt-out needed an existing member row**; now it creates it. `delete-member` (1.1.1) also removes the
   member's lines from `rejected-*.jsonl`; `stats` counts pending/rejected/.bad spool files.
9. **Low (1.1.1, found from the live probe's redaction counts):** a passphrase followed on the same line by another
   recognised secret kept every word after the first. Fixed and tested (`passphrase_then_token`).
10. **Low (1.1.1):** a batch was written even when its omni_lab run could not be opened ("every batch registered").
    Now no omni_lab run, no batch (retried next hour), and each batch commits on its own.

**Latency (server, fake upstream, 35 KB / 63-tool Sprite bodies, `tests.test_capture` TestLatency and the new
TestStreamingTTFT):**
- Non-streamed total, alternating blocks of 40 (1.1.1 run): p50 9.12 ms off vs 9.13 ms on, p95 13.70 vs 11.05 ms;
  `begin` + `submit` 16 µs per request.
- **TTFT**, streamed, 60 requests each: capture off / on / on while the child is busy with a backlog of 8
  Sprite-sized captures when the request arrives:

  | run (log) | p50 ms | p95 ms |
  |---|---|---|
  | before the insert also waited for user requests (`test_suite_staging.log`) | 19.95 / 20.26 / 23.43 | 43.46 / 23.10 / 25.98 |
  | after (`test_capture_rerun.log`) | 22.08 / 21.85 / 22.79 | 31.38 / 24.50 / 42.27 |
  | 1.1.0 final (`test_suite_final.log`) | 25.44 / 23.99 / 28.00 | 61.41 / 32.18 / 38.01 |
  | 1.1.1, CRLF build (overwritten log) | 21.15 / 21.89 / 23.30 | 47.29 / 25.55 / 26.22 |
  | 1.1.1 final (`test_suite_111.log`) | 20.98 / 21.47 / 22.83 | 30.56 / 27.38 / 25.93 |

  Capture on vs off is within noise; the backlog case adds about 0.7-2.6 ms at p50 and happens only right after a
  burst of background captures. The p95 column moves by tens of ms between identical runs (3 samples make it), so
  read it as noise. The fake upstream shares the gate's event loop, so these numbers bound the gate's share only.
- Not measured: TTFT against the live llama-server with capture switched off (needs a gate restart with
  `--no-capture` in a quiet window).

**Still open (not fixed here):**
- Redaction is by shape. A password with no label and no digits or symbols ("hunter"), or one bare lowercase word
  after `password:` (kept on purpose so code is not mangled), is stored. Vault values are not compared: the gate never
  loads them. If Cyril wants a stronger guarantee, a salted-hash list of vault values checked per token is possible
  (a vault-derived file on the server; his call).
- A crash between writing a batch to the Lexar and the database commit would export the same captures again in the
  next batch (the dedup reads only committed decisions).

### 11.6 Live checks after the review deploys (`evidence/capture-review-20261022/`)

- **1.1.0 restart:** quiet window 621 s since the last sprite/cyril request, none active or waiting in `/gate/status`;
  restart 05:43:41Z, healthy with capture 1.1.0 enabled at 05:43:46Z (`deploy_receipt.json`). No Sprite request was
  touched (none since 02:01Z).
- **Live probe through the real gateway** (05:43:46Z, background tier, `strata-think`, source `claude-review-probe` =
  never exported; `probe_check.json`): planted `PGPASSWORD=...`, a `glpat-` token, a garage PIN and an `X-Serve-Key`
  hex. Stored row: `finish_reason stop`, not partial, 494 chars of reasoning and 37 of content captured; none of the
  four planted values is anywhere in the stored body, content or reasoning (redaction counts pin 5, provider_token 1,
  credential_value 2).
- **Live traffic by 05:47Z** (`capture_admin.py stats`): 137 complete gen-worker-text captures (6,074 reasoning
  chars) and 3 partial ones, 331 cyril `unknown` (the gpu6-tune bench), 12 `background-unlabelled`; 3.0 MB of tables.
  Export plan with 1.1.0: 18 trainable records to screen, 8 `not_target`, 92 `excluded_source`, 12 held unlabelled.
- **Sprite, first live traffic since the capture went in** (05:50-06:00Z, counts only): while Sprite requests were
  running, capture work waited (`deferred_ms` 55,833 at 05:52Z) and the rows appeared after each burst. At 06:01Z
  41 of 41 Sprite requests were captured: 39 as `sprite:unmarked` and 2 under a member pseudonym
  (`sprite:cd8ee033...`: the Sprite backend has started sending `X-Sprite-Member`). Both members are held from
  export (household default). Since 04:32:48Z, every request that reached Strata is captured except the 2 lost at
  the 05:43:41Z restart (11.5 item 5): background 232/234 ok + 6/6 error + 6/6 preempted (all 12 stored `partial`),
  cyril 331/331, sprite 41/41; token counts agree with `request_log` for every complete one (`completeness.json`).
- **First real export** (06:20:54Z, exporter 1.1.1): 309 candidates, 136 screened, 135 exported:
  `strata-capture-20261022-001-text` (77 rows, sha `11f8a7a3...`, omni_lab run 38205, gen_output 574) and
  `strata-capture-20261022-002-tools` (58 rows, `1a69fff9...`, run 38206, gen_output 575). Decisions: 135 exported,
  1 `dedup_near`, 81 `not_target` (67 verifier rejected/unused, 5 preempted, 3 error, 5 rejected by the program
  checks, 1 empty), 92 `excluded_source` (91 gpu6-tune bench, 1 review probe); 95 held by consent (Sprite) and 20
  held unlabelled. All 135 rows are background `gen-worker-text`. The intake's own extractor (chimera-intake venv,
  `batch_check.json`) takes exactly the target view as the response for 135 of 135 (8 with `<think>` reasoning, 34
  with `<tool_call>` blocks) and never an earlier assistant turn; a rescan of both files with the current redaction
  rules finds nothing left to redact (`batch_redaction_rescan.json`).

### 11.7 Needs Cyril (capture)

1. **Household members (default, recorded in `capture_policy.json` and `capture_member.set_by`):** captured, but
   **held from training export until you confirm each one** (`capture_admin.py set-export --member sprite:<x> --on`).
   `cyril` and `background` are captured and exportable. Confirm or change.
2. Third-party tool results (SMS, messages, contacts, notifications, email, calendar, files, screens): conversations
   holding them are held from export. Keep holding, or allow.
3. Delegated cloud/Codex output in Sprite contexts: held (so it cannot be laundered into a target). Keep holding, or
   allow with context-only marking.
4. Emails and phone numbers: kept (default) or redacted (`redact_contacts`).
5. The Lexar is exFAT, so files cannot be owner-only. Only cleared, trainable records go there; everything else stays
   in PostgreSQL. Accept, or ask for encryption on the Lexar.
6. The export's held-out screen runs in the `model-surgery-lab` venv, which still has pydantic (section 10 item 5).
   Remove it, or exempt it in writing.
7. **Retention (new):** nothing expires today. Proposed: household captures never cleared for export are deleted
   after 90 days; exported records stay as corpus. Pick a period or keep everything.
8. **Sprite backend (Sprite lane):** a request without `X-Sprite-Member` (an HMAC pseudonym) is `sprite:unmarked`, and
   a single household member can only be opted out by pseudonym.
   - Status 07:15Z: the headers and the settings switch are in Sprite commits e9cfef3/f3cf2c2, which are not on the
     stage yet.
   - The gate side, including the per-request opt-out, is live (11.8).
   - The Sprite lead keeps the settings row hidden until `SPRITE_STRATA_CAPTURE_OPTOUT=1` is set on the stage.
9. Optional: redaction of vault values by salted hash (catches an unlabelled password from the vault). Needs a
   vault-derived file on the server.

### 11.8 Sprite capture headers and the per-request opt-out (capture 1.2.0, strata-gate 1.2.1; live 07:11:59Z)

Asked by the Sprite lead on Relay `52468f63`: event 204015 (option (b), a per-request header opt-out), 204017 (the
headers shipped in Sprite e9cfef3) and 204018 (record the received switch on capture rows; announce with a proof row).

**What the gate accepts.** Headers are read from the sprite tier only, which is the `strata-sprite` key
(`api_client` id 1). They are never forwarded to llama-server.

| Header | Format accepted (anything else is dropped) | Stored as |
|---|---|---|
| `X-Sprite-Member` | 24 hex (HMAC-SHA256 of the user uuid, Sprite vault salt), lower-cased | member `sprite:<hex>`; malformed or missing -> `sprite:unmarked` |
| `X-Sprite-Delegation`, `X-Sprite-Run` | canonical uuid, lower-cased | `provenance.sprite.delegation/run`; with either one the source is `sprite-app`, else `sprite-unmarked` (held) |
| `X-Sprite-App-Version` | git sha of 7-40 hex with optional `+dirty`, or a version number such as `1.4.2` | `provenance.sprite.app_version` |
| `X-Sprite-Prompt-Template` | 64 hex (sha256) | `provenance.sprite.prompt_template` |
| `X-Sprite-Capture` | `off` / `on` (no header = on); any other value, or two values, fails closed as off | see below; captured rows carry `provenance.sprite_capture` `on` or `absent` |

- **Malformed values** (wrong length or characters, over 128, or sent twice with different values) are dropped.
  Only the header name goes into `provenance.sprite_invalid`; the value is never stored. The gate status counts
  `sprite_headers_invalid`.
- **Other keys** (cyril, background) set nothing with these headers: no member and no opt-out. The request is
  captured as that tier's member. The header names go into `provenance.sprite_ignored`, counted as
  `sprite_headers_ignored`.

**`X-Sprite-Capture: off`.**
- The request is served exactly as usual: non-streamed, streamed, preempted or cut by the client.
- Its capture handle holds the member key only: no body reference, no headers, no answer. It is never queued, prepared
  or spooled.
- The writer adds it to `capture_opt_out` (member, UTC hour, count, `opt_out = true`) under the same lock and consent
  check as captures. A member opted out with `capture_admin.py` gets no counter either.
- With PostgreSQL down, the counts wait in memory, and a stopping gate stores them.
- Counted in the gate status: `opt_out_requests`, `opt_out_counted`, `opt_out_lost`.
- What remains of such a request is its `request_log` row, the same as every request since 2026-10-21. That row holds
  tier, client, request id, model, token counts, timings and outcome; no prompt text.
- An `off` request therefore has no capture row. Its proof is: a `request_log` row with that `X-Request-Id`, no
  `capture_request` row for it, and the member's counter.

**Tests.**
- `tests/test_sprite_capture_headers.py`, 23 tests:
  - 6 parsing tests: formats, case, duplicates, the switch, other keys, member keys.
  - 8 tests that opt-out requests are served but not kept: non-streamed, streamed, the writer's own round, a
    malformed member, a garbled switch, an admin opt-out winning, database down (nothing spooled), and stop.
  - 3 preempted and cut tests: a thermal-cancelled opt-out stream, a client hang-up, and background dumped by an
    opted-out request.
  - 4 spoofing and provenance tests: the cyril and background keys, contract provenance, malformed headers.
  - 2 tests on real PostgreSQL in a throwaway database: counter rows only, `ai_serve_ro` cannot read them, upserts,
    consent, delete-member removing the counters, the CHECKs.
- `tests/test_capture.py` now uses contract-format Sprite values. It also has `drop_test_db()`, which retries
  `DROP DATABASE ... WITH (FORCE)` on SQLSTATE 42501.
  - The first server run hit that once in teardown: an autovacuum worker in the fresh test database cannot be
    signalled by the `ai_serve` role.
  - That log is kept as `test_suite_120_run1_teardown_error.log`, and the database it left was dropped.
- Server suite on the final staged tree: 151 OK / 1 skipped (`test_suite_120.log`; the skip is the numpy-dependent
  intake extractor test).
- TTFT p50, capture off / on / on with a backlog: 21.43 / 20.73 / 23.75 ms.
- begin + submit costs 20 us with all six Sprite headers present (1.1.1: 16 us).

**Deploy (`staging/cr-tools-120/run_all_120.sh`, detached).**
- 07:11:54Z: the staged tree was checked against `staged_SHA256SUMS`, then backed up, and 0008 was applied.
- Quiet window: 980 s since the last sprite/cyril request, none active or waiting.
- 07:11:59Z: files installed and strata-gate restarted. At 07:12:14Z it was healthy, with gate 1.2.1 and capture
  1.2.0 enabled (`deploy_receipt_120.json`). The rollback was not needed.
- Completeness (`completeness_120.json`): since the 1.1.1 gate came up at 06:34:03Z, every request that reached
  Strata is captured, 299 of 299 (the two opt-out checks below are left out of the count). That includes 14 background answers that finished in the 60 s before the 07:11:59Z restart.

**Live checks.**
- `probe_check_120.json`, secrets probe: the background planted-secrets probe was captured, and none of the four
  planted values was stored.
- `probe_check_120.json`, spoof probe: the background key sent emutest's pseudonym, every Sprite header and
  `X-Sprite-Capture: off`.
  - It was captured as member `background`, with `provenance.sprite` null.
  - All six header names are in `sprite_ignored`; the pseudonym is not stored.
  - emutest's counter stayed 0.
- `live_optout_check.json`: two Sprite-key requests as the stage member emutest (`sprite:cd8ee033...`) with
  `X-Sprite-Capture: off`, one non-streamed and one streamed, at 07:13:40Z. Request ids `3ad2a463...` and
  `28516083...`.
  - Both were served: HTTP 200, and `request_log` has tier sprite, outcome ok.
  - 0 `capture_request` and 0 `capture_response` rows.
  - Each prompt's nonce is in no capture and no spool file.
  - `capture_opt_out` for emutest, hour 07:00Z: 0 -> 2. Gate status: `opt_out_requests` 2, `opt_out_counted` 2.

**Open.**
- The stage `:23499` still runs `claude-20261019`, which has neither e9cfef3 nor f3cf2c2. Stage traffic carries no
  `X-Sprite-*` headers until the lead merges.
- Test row `206075f8...` (emutest; sent with `off` before 1.2.0 existed) is still stored, and held from export.
  - Removing a single row has no admin command.
  - `delete-member` would opt emutest out for good.
- Four `ai_serve_test_*` databases predate this lane: `bae4c3ff22`, plus three marked invalid by an interrupted drop
  (`6632280401`, `a796031e`, `ad5531db`). They were left alone.
