# Run 0 support scout: Strata background data generation (D3) and TTS caption renders (D4)

Date: 2026-10-28, 06:00-06:30Z (server clock UTC; journald prints MDT). This was a read-only scout. I read the code and checked the live state over the server-manager client. I made no service change and sent no generation request, and nothing was written on the server: two GGUF header reads ran from a temp dir under `/tmp`, which I deleted. Database queries ran with `SET default_transaction_read_only = on`.

Rulings: `cyril-run0-decisions-2026-10-22` D3 ("[Cyril quote withheld]") and D4 ("4 sure"), in `distill/STAGE_D_RULINGS_2026-10-12.md`. Plan: `RUN0_BUILD_PLAN.md` §2.3, §2.5, §3.2-3.3, B-3, B-5, B-6 and the Addendum.

## 0. Answer first

1. **Strata background work is stopped right now, and has been since 01:46Z.**
   - `GET /gate/status` (06:2xZ) returns `background_admission: {"allowed": false, "why": "MemAvailable 29 GiB < 40"}`.
   - The last foreground request was 29,667 s (8.2 h) ago, so this is not foreground use. The last background request was at 01:41:39Z.
   - media-gate has the same 40 GiB floor (`mem_floor_gib` 40). It also preempts running background work below 20 GiB.
   - Host RAM (06:1xZ):

     | What | Amount |
     |---|---|
     | MemAvailable | 29-31 GiB |
     | Swap | full: 3.8 MB free of 8 GiB |
     | Shmem | 81 GB |
     | strata-llama RSS | 150 GB |
     | `comfyui_qwen21_gpu4` container (ComfyUI `general` profile) | 57.6 GiB |

   - **Nothing in this note produces a row until one of two things happens:**
     - the Strata lane lowers `--mem-floor-gib`. This is a `strata-gate` change, so it goes through Relay, a quiet window and a rollback.
     - RAM is freed, for example by the 57.6 GiB ComfyUI process. That is the media lane's call.
   - Run 0 starting on the P40s will push MemAvailable lower still.
2. **gen-worker-text is partly broken by the 10-27 Q4 switch.**
   - `toolcall_conv` fails 100 %: 117 failed tasks, the last at 01:23Z. toolcall_selfgen locks the GGUF and refuses the new one:
     ```
     FAILED: http://127.0.0.1:18400 serves .../Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/...-00001-of-00004.gguf,
     the locked GGUF is .../Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf
     ```
   - The `text_tasks` kinds still run: `reasoning_math`, `instruction_code` and `long_context` finished 111, 89 and 57 batches in all.
   - **Their provenance is wrong after 11:04 MDT on 10-27.** `gen_worker_text.py:87-89` hardcodes `GGUF`, `GGUF_SHA = cb5958e5…` and `MODEL_ID` to the IQ3_S, and `text_refill.py:215,227` copies those into every batch's `.provenance.json` and every row.
   - The Q4 is a different checkpoint: `gguf_ident compare` says DIFFERENT, `qwen4exp.attention.compress_ratios` differs, and the base and imatrix differ. So rows made since 10-27 11:04 MDT carry the wrong teacher id.
   - Fixing either problem is a gen-worker-text code change. That is not a Strata service change, but it is another lane's code. Flag it to the lead.
3. **strata-gate 1.2.3 passes `n_probs` / `logprobs` / `top_logprobs` through untouched.**
   - `shape_openai_body` changes only `model`, `chat_template_kwargs.enable_thinking` and token caps above 262,144.
   - Native routes (`/completion`, `/tokenize`, `/detokenize`, `/apply-template`, `/props`) are open to the background tier, and the gate forwards their body as-is, except `stream` and `id_slot`.
   - For a background `/completion` with `stream:false`, the gate streams upstream and rebuilds one JSON. `merge_completion_chunks` concatenates `completion_probabilities` and `tokens` in order.
   - So **D3's `n_probs=64` capture needs no gate change.**
   - Background `/v1/*` must use `stream:true` (otherwise 400 `background_must_stream`). llama-server also refuses `logprobs` together with tools and streaming.
4. **Top-64 from llama-server are pre-sampling, T=1, full-vocab probabilities.** With `post_sampling_probs:false` (the default), they are what the existing KD caches hold. I checked this in the served xcache build's source (`get_token_probabilities`: partial_sort over all 248,320 logits, softmax over all of them). Two things differ from the cached shards:
   - **lse:** llama-server returns normalised `log p`, not the raw-logit lse that the teacher-score shards store in `lse.f32`.
   - **Coverage:** only tokens Strata *generates* get a distribution. `echo` is refused.
5. **n_probs only pays on assistant tokens.** Area B's grammar makes every user turn conditioning (`loss_mask` false), and edit instructions and intent requests are user turns. So:
   - Ask for **no** probabilities on the intent and edit-instruction writes.
   - Ask for `n_probs=64` only on answers that become assistant targets (§3).
6. **Edit renders and TTS renders run through the gates at background tier with no service change**, but the queue is not a clean drop-in:
   - **Edits:**
     - `image_edit` gen_tasks of a COCO photo (`params.source_path` + `source_sha256`) run as they are.
     - They can **never publish**: `media_sets` holds an edit until its source has a media-judge pass, and COCO photos are never judged.
     - Their 1024² output square-crops COCO's 4:3. The task_spec helper forces 1024×1024.
     - Priority 100 would starve the media mix, because `lease()` orders by priority before mix order.
     - Recommendation: a **standalone media-gate client** that writes straight to Lexar staging (§4).
   - **TTS:** speech has no queue. Calls go straight to `POST :28400/v1/audio/speech` with **`media_key_background`** (§5).
7. **Use the background keys:** vault `strata_key_background` and `media_key_background`. Both exist (`credman.py list`, names only).
   - The task text lists `strata_key_cyril` and `media_key_cyril`. Those are the **foreground** `cyril` tier: they preempt every background job and spoil the "quiet window" signal the Strata lane relies on.
8. **Disk:**
   - `/` has 113 GB free, **below** the 150 GB the rules keep for run 0. Write nothing to `/`. Note that gen-worker-media spools to `/home/cereal/ai-serve/spool/` on `/`.
   - Lexar has 381 GB free.
   - The NAS is 100 % full.
   - The intake-staging dir in the task's staging contract does not exist yet. It is `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/`. The plan names no other intake-staging location:
     - area B's `/home/cereal/omni-campaign/run0/data/staging/` holds raw source assets and sits on `/`;
     - `/home/cereal/omni-campaign/run0/data/parts/` holds live shards.

## 1. gen-worker-text: how it works

| Item | Fact (source) |
|---|---|
| Versions on the server | `gen_worker_text.py` 1.4.0, `text_tasks.py` 1.3.0, `text_refill.py`, `selfgen_via_gate.py`; sha matches the workstation copy `Documents/ai training/ai-serve/strata/` (strata_gate.py is the exception: server 1.2.3 `157f251c…`, workstation 1.2.1) |
| Queue | `ai_serve.gen_task` (service `strata`); lease 1800 s; `FOR UPDATE SKIP LOCKED`; parks after 5 preemptions, fails after 2 errors |
| Kinds and mix | `toolcall_conv` 0.40 (24 conv/batch, `toolcall_selfgen.py run` → `finalize` → `toolcall_batch_screen.py`), `reasoning_math` 0.25 (16), `instruction_code` 0.20 (16), `long_context` 0.15 (8). Refill every 10 min keeps 3-8 queued per kind; the lease takes the kind furthest behind its 24 h share |
| Who writes what | Strata writes the user turns and the answers. Seeds: COCO 2017 **train** captions (CC BY 4.0) and LJSpeech 1.1 (public domain), screened against the held-out registry before Strata sees them. The instructions to Strata are templates and are never training text |
| Verifiers | math: program in `strata-sbx` sandbox == author answer == cold `strata-think` answer. code: asserts that pass the reference and fail a stub; an independent solution must pass them all. long_context: a verbatim quote, unique to the target passage, appears in the full-context answer. Then `text_batch_screen.py` (`toolcall_heldout_screen` 1.1.0, 13-gram) |
| Gate calls | `text_tasks.Gate.chat`: `/v1/chat/completions`, `stream:true`, temperature 0.3-0.9, headers `X-Client-Source: gen-worker-text`, `X-Gen-Task`, `X-Gen-Kind`, `X-Workflow-Id text_tasks.<kind>.<role>`. It retries 409/429/503 using `retry_after_s` and gives up after `GIVE_UP_S`. **No logprobs anywhere** |
| Spool | `/home/cereal/ai-serve/spool/text/strata-<kind>-<date>-t<task>/` holding `<kind>.jsonl`, `.sha256`, `_meta.json` (intake keys: source, licence, route, group_by) and `.provenance.json` (whole batch). One `gen_output` row and one omni_lab `gen` run (L-SERVE) per batch |
| Move | The mover copies to the NAS zone while it has >= 20 GB free, else to the Lexar overflow `/mnt/lexar/omni-campaign/corpus-store/chimera-intake/` (`.part` → re-hash → rename; the spool copy is deleted after it re-hashes equal). The NAS is full, so it goes to the Lexar |
| Intake | `intake_cron.sh` every 30 min → `intake.py` → `chimera_corpus` DB. Run 0 reads only ADMITTED/train items through `run0_intake_sync.py` → `run0_build_manifest.py intake-text`, which accepts only batches starting `text/strata-` without `capture`. Those become `build_chat` records with every turn authored `strata`: **CE only, no teacher** |
| Floors | 45 GiB local (gen dir + spool on `/`); the worker checks gate admission before leasing |
| Counts (all time, 06:05Z) | toolcall_conv 162 done / 117 failed / 8 queued; reasoning_math 111 / 3 rejected; instruction_code 89 / 4 rejected; long_context 57 / 16 rejected; strata_capture_export 237 done |

## 2. strata-gate 1.2.3 background API

- **Auth:** `Authorization: Bearer <strata_key_background>`. The tier is set by the key's sha in `ai_serve.api_client`.
- **Routes:**
  - Background may use `/v1/chat/completions` and `/v1/completions` (both must stream).
  - It may also use native `/completion` (stream optional; the gate always streams upstream), `/tokenize`, `/detokenize`, `/apply-template` and `/props`.
  - `/slots` is cyril-only.
- **Admission:** all of these must hold, otherwise the request waits up to `--bg-wait-s` 30 s and then gets `503 {"error":{"code":"busy","retry_after_s":N}}`:
  - no foreground request running or waiting;
  - none finished in the last 20 s;
  - fewer than `--bg-max 4` background requests running (pinned to slots 2-5);
  - MemAvailable >= 40 GiB;
  - thermal OK. GPU6 takes no new background from 70 C and cancels it at 75 C (`--bg-headroom-c 12`). 101 of the background refusals in 12 h were thermal.
- **Preemption:** a sprite or cyril request closes every background stream within about 3 ms.
  - A non-streamed caller gets `503 preempted` and loses the whole answer.
  - A streamed caller gets an SSE `{"error":{"code":"preempted"}}` and EOF.
  - The deadline for background is 1800 s.
- **Body passthrough:** no default `max_tokens` (1.2.3), and the gate owns `id_slot`. Every other field, `n_probs`, `post_sampling_probs`, `return_tokens`, `cache_prompt`, `logprobs` and `top_logprobs` included, reaches llama-server unchanged.
- **Capture:** every request that reaches llama-server is captured as member `background`. Capture keeps the content, reasoning, tool calls and tokens; `completion_probabilities` are not stored.
  - **RAM caveat:** the capture queue estimates 1 KB per streamed chunk (`CHUNK_BYTES`). With `n_probs=64`, a parsed chunk holds about 64 nested dicts, roughly 40 KB.
  - So the 256 MiB queue cap really allows about 40 times that while the capture writer is behind, which is unsafe at 29 GiB available. Capture was healthy at 06:2xZ (queue 0, 0 drops).
  - **Keep KD generations short:** `n_predict` <= 512.
  - The fix would be to strip probabilities before capture or count them. That is a Strata-lane change: Relay first.
- **Captured twice:** these outputs will also show up in `strata-capture-*` exports. Run 0's guard refuses those (`DEFAULT_DENYLIST` "strata-capture", row pattern "capture"), so nothing is counted twice.

**Recommended KD request (native, teacher and student see identical prompt tokens):**

```
POST http://192.168.50.222:18400/completion
Authorization: Bearer <strata_key_background>
X-Client-Source: run0-strata-writer      X-Workflow-Id: run0.<product>.<role>
{"prompt": [<token ids rendered with the Qwen3.5-2B tokenizer.json + chat template, enable_thinking=False>],
 "n_predict": 384, "temperature": 0.7, "top_p": 0.95, "seed": <per item>,
 "n_probs": 64, "post_sampling_probs": false, "return_tokens": true, "cache_prompt": true, "stream": false}
-> {"content", "tokens": [...], "stop_type", "timings",
    "completion_probabilities": [{"id", "token", "bytes", "logprob", "top_logprobs": [{"id","token","bytes","logprob"} x64]}, ...]}
```

- **Why native and token ids:** Strata and Qwen3.5-2B share tokens, merges, the pre-tokenizer (`qwen35`) and bos/eos/pad. `gguf_ident compare` on the two Strata GGUFs: only `chat_template` and `add_bos_token` differ.
  - The served Q4 carries the Unsloth chat template, which differs from the IQ3_S one (sha `1479547f` vs `924d3fcc`). Strata's `/v1` rendering may not match the student's.
  - Sending the student's own rendered ids removes that risk. The response ids are Strata's own sampled ids, which the student must use as-is: re-tokenizing the decoded text can give a different segmentation.
- **Thinking:** a native call ignores `model`. Render the empty `<think>\n\n</think>\n\n` that `enable_thinking=False` adds, exactly as the student's template does.
- **Cost:** 64 entries per token, about 4.5 KB of JSON. Stored it is about 400 B per position, so 1M positions take about 0.4 GB.
- **Server CPU:** the server does an O(V log 64) partial sort plus a full softmax per token per slot. That is small next to the 24 tok/s decode, but measure it. With n_probs set, llama-server turns backend sampling off for that slot.

## 3. KD target format for Strata's own generations (`n_probs=64`)

The aim is to match what area B's `TextKDSource` already reads: the qtext shard layout and the `Run0Item.teacher` dict in §3.3. The new teacher gets its **own id**, for example `strata-q4kxl-abl-v2`, keyed to the 4 shard sha256 values in `.../Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL-V2/SHA256SUMS`. It must never be filed under `flash-next-strata-iq3s`.

```
<staging>/strata-kd-gen/v1/
  records.jsonl            chimera_omni.run0.record.v1 (build_chat; user turns author "strata" or "public", assistant author "strata"),
                           each with "kd": {"row": r, "ids_sha256", "prompt_len", "template_sha256"}
  shard-00000/
    ids.u32                every token of row r (prompt + response), Qwen3.5 ids
    pos.u32                predictor positions p = prompt_len + i - 1 for response token i (distribution over token p+1)
    topk_ids.i32           [P, 64]   sorted by logprob desc
    topk_logprobs.bf16     [P, 64]   normalised log p (llama-server pre-sampling, T=1)
    sampled_logprob.f32    [P]       log p of the token Strata actually emitted (it may be outside the top-64)
    lse.f32                [P]       NaN: not available from llama-server; flag below
    shard.json             row_table, checks {finite, positive_logprobs 0, unsorted_rows 0, topk_ids_out_of_range 0,
                           topk_mass_mean, top1_logprob_mean}, file sha256s
  cache_config.json        schema omni-distill-teacher-cache-config-v1 + "source": "llama-server n_probs on own generations",
                           "lse_semantics": "absent (normalised logprobs only)", "post_sampling_probs": false,
                           gguf shard shas, llama.cpp commit b96806d96 + xcache.patch sha, strata-gate version, sampling params
  manifest.json, README.md (staging contract)
```

**Open points for area B and area C:**

- **lse.** Does C-3's top-K bucket forward KL read `lse`? `TextKDSource` loads it through `teacher_lse_row`.
  - If it only needs normalised top-K plus the tail mass, these rows drop straight in.
  - If it reconstructs logits, the n_probs rows need a "normalised" flag path.
- **EOS.** Check in a smoke test whether the stop token (`<|im_end|>` 248046) gets a `completion_probabilities` entry when `stop_type` is `eos`. If it does not, the last target is CE only.
- **Sampling temperature.** Sample at T ≈ 0.7 for variety. The stored targets stay T=1 regardless.
- **What to generate.** It only pays to capture n_probs where the tokens are assistant targets.
  - **New standalone answers:** Strata answers text requests seeded from COCO train captions, LJSpeech and our canonical corpus. Use the same blind-check discipline as text_tasks, for example a math or quote check, wherever a check exists. Unchecked free-form answers are CE+KD targets of unknown quality, which is a lead call.
  - **Re-generations of verified gen-worker-text items:** re-ask the solver turn with n_probs and keep it only if it re-verifies. This uses the batches in the intake, so no gen-worker change is needed.
  - **Avoid** `strata-think` with probabilities: reasoning runs to thousands of tokens, with the RAM risk from §2.
- **Yield is modest.** Background ran about 50-120 Strata requests an hour while it was admitted. At about 300 response tokens each, that is about 15-35k KD positions an hour of idle time.
  - The text-KD bucket (22 % of about 33-46M positions a day) needs about 8-10M positions a day.
  - So this stream adds **a few percent a day** of fresh own-distribution KD. It does not replace the cached 35M.

## 4. Intent / output-decision examples and edit instructions

**What the plan needs (§2.3, §2.5, B-3).**
- **Rows.** Request text changes while the input stays the same, and each row is labelled `<|out|>[<|edit|>]<|m…|><|/out|>`. The label comes from the template slot, never from a model. There is no router.
- **Intents.** Six of them, over COCO items (human captions) and judged generation items:

  | Intent | Decision label |
  |---|---|
  | describe | `[text]` |
  | caption-joke | `[text]` |
  | edit | `[edit, image]` |
  | redraw | `[image]` |
  | animate | `[video]` |
  | sound | `[audio]` |

  Mixed `[text, image]` is allowed.
- **Twins.** Most of these can be **decision-only twins**: request plus `<|im_start|>assistant\n<|out|>…<|/out|>`, about 12 supervised tokens, in bucket 4 % "decision twins". No answer media is needed.
- **Authorship.** Intent phrasings come only from Strata or human COCO captions. Area B's own `REQUESTS` table is lead-written (author `template`, conditioning only).

**Strata write (no probabilities; `/v1/chat/completions` model `strata`, stream; or native).**
- **Input:**
  - the item's 5 COCO train captions, as a stand-in for what the image shows (Strata cannot see it);
  - the intent slot list;
  - a style list: terse, polite, indirect, no modality keyword, misleading keyword. "What's in this picture?" → text and "Make it look like night" → edit are hard negatives against keyword routing.
- **Output:** a JSON array of `{slot, request}` with 2-3 requests per slot. One call over one item gives about 15-20 requests in about 400 tokens.
- **Screens (client side, before staging):**
  - JSON parse;
  - 3-60 words;
  - no 6-word run copied from a caption, as `media_prompts` does;
  - held-out registry v3 text screen (`/mnt/lexar/omni-campaign/corpus/holdouts/HOLDOUT_REGISTRY_v3.json` + `registry_v3/`; word 8-gram, whole held sentences, liveness probe; reuse `media_prompts`' screen code read-only);
  - dedup by word 3-gram Jaccard >= 0.6 per item and globally.
- **Label check (recommended):** a second, blind Strata request classifies each request into an output set. Keep the row only when it equals the slot. Disagreements go to an `ambiguous/` pool and never become training rows unless the lead rules otherwise.
- **Ambiguous cases** ("do something with this") need a ruling. The proposal is decision `[text]` with a Strata-written clarifying question as the target (author strata, CE+KD with n_probs). The alternative is to exclude them.
- **Record:** `chimera_omni.run0.record.v1`, task `intent_twin`. User parts are `{"t":"media","ref":0}` plus `{"t":"text","text":<request>,"author":"strata"}`; the assistant has `out`, `edit` and no parts, and the twin flag is set. Provenance is `strata-gen`, with the label source `template-slot`.

**Edit instructions to be rendered (Qwen-Image 2.1 edit, media-gate GPU4).**
- **What Strata writes:** given the captions, Strata writes 1-3 feasible, grounded edit instructions, 5-20 words each, about objects the captions name. It also writes the same phrasings for **procedural** edits (mirror, crop, grayscale, hue), which PIL renders deterministically with no GPU.
- **Sources:** COCO train2017 photos at `/mnt/lexar/omnicoder_0.1.0/data/coco2017/train2017/` (118,287 jpg; annotations CC BY 4.0; images under Flickr terms, record per image). Use train only; run the media sha check against registry v3.
- **Rendering path. Recommended: a standalone media-gate client.**
  - `POST :28400/prompt` with `X-Serve-Key: <media_key_background>`, built with `media_graphs.qwen_graph(prefix, {"prompt", "reference_name", "width", "height", "seed", "steps": 25})` after `POST /upload/image`.
  - Set width × height **per source aspect**, at about 1 MP in multiples of 64, for example 1024×768 for 4:3, so run 0 can bucket it to 512×384.
  - Download with `/history` and `/view` straight to Lexar staging.
  - media-gate allows **1 background job at a time** (`bg_max_inflight` 1), so this client and gen-worker-media take turns. Cap the client's duty cycle, for example at most 1 job in 3, or set windows with the media lane.
  - Gate rules: a foreground `/prompt` interrupts the job, so requeue it. GPU4 at >= 82 C admits no background. The grace after Sprite use is 60 s.
- **Alternative: insert `image_edit` gen_tasks.** Use `params.source_path`, `source_sha256`, `width`, `height` and `prompt_source{kind:"strata", ...}` at **priority 200**. Priority 100 runs ahead of every refill task, because `lease()` orders `priority` before the mix. The drawbacks are why the standalone client is preferred:
  - the outputs spool on `/`;
  - they **never publish**, because the COCO source has no `media_judgement`;
  - they bypass `media_prompts`' held-out screen and dedup;
  - gen-worker-media cannot read the holdouts (`InaccessiblePaths`).
- **Checks before staging:**
  - the output decodes and has the expected size;
  - it is not near-identical to the source (pixel/pHash distance above a floor, which catches no-op edits);
  - it is not flat.
  - **No meaning check exists yet.** The media judge is the Qwen3.6-27B on GPU5, which belongs to run 0 now; D2 moves it to Strata only after vision is enabled. Stage these as `judged: false` and let area B decide the weighting.
- **Record:** `run0_samples.build_edit(source_ref, result_ref, instruction, instruction_author="strata")`, provenance `media-teacher`, with `teacher: qwen-image-2.1-edit` and the graph sha.
- **Throughput (projection):** about 14-20 s per warm edit, but GPU4 is shared, so perhaps 50-100 an hour of idle time.

## 5. TTS caption renders (D4, concept-interleaved)

- **Endpoint:** `POST http://192.168.50.222:28400/v1/audio/speech` with `Authorization: Bearer <media_key_background>`. The body:
  ```
  {"model":"qwen3-tts", "voice":<speaker>, "input":<caption>, "language":"en", "response_format":"wav"|"flac", "seed":<n>}
  ```
  - Get speakers from `GET /v1/audio/voices`: vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee. Choose English-capable ones and record the speaker per clip.
  - Use non-streamed calls. With a non-streamed background call, a preemption gives a clean 503 `preempted`; a streamed one is cut mid-body.
- **No queue, no gen_task kind:** gen_worker_media has no TTS kind, and speech jobs never appear in `/queue`. The "idle queue" here means the background tier.
  - Background speech is admitted like `/prompt`: no foreground for 60 s, 1 background in flight, GPU4 below 82 C, MemAvailable >= 40.
  - A Sprite or cyril request stops the synthesis at its next 12 Hz codec token.
- **Profile switch cost:**
  - Entering `speech` frees ComfyUI ("parked", `park_keep_cache`: the weights stay in container RAM) in 2.2-4.4 s.
  - The next image after speech waits 79-241 s for the reload.
  - `speech` holds the card for 300 s after its last use (`bg_hold_s`), so a continuous TTS run blocks gen-worker-media's ComfyUI jobs for its duration.
  - **Run TTS in long blocks** (hundreds of clips), at times agreed with the media lane, not interleaved.
- **RAM:** parking keeps ComfyUI's host-RAM cache, and that process is already 57.6 GiB. Look at RSS before and after the first block. Stop if MemAvailable falls toward 20 GiB, where media-gate preempts all background work.
- **Output:** 24 kHz mono s16. Run 0 resamples to 38.4 kHz (B-2 polyphase).
- **Built-in check:** finite samples, RMS >= -45 dBFS and a duration that fits the text, with one retry on failure; otherwise 502 `bad_audio`.
- **Add a round-trip check:** ASR through the same gate (`/v1/audio/transcriptions`, Qwen3-ASR-1.7B), keeping a clip only at normalised WER <= 10 %. The deployed speech_e2e run measured 5.87 % raw WER. The ASR text is a check only, never a target.
- **Inputs:** COCO train captions, one per image, chosen at random from the 5. Run the held-out text screen. COCO images pair only with caption-style requests (critic item 15).
- **Record:** `run0_samples.build_concept(image_ref, caption (author public), speech_ref)`, provenance `media-teacher` (Qwen3-TTS-12Hz-1.7B CustomVoice, Apache-2.0). Area B's builder has `concept: 0` until the set exists (`run0_build_manifest.py:836`).
- **Volume needed (projection):**
  - One sample is about 300 image rows (640×480), a 15-token caption, about 44 audio rows (3.5 s) and turn overhead, so about 390 positions.
  - The 2 % bucket at 33-46M positions a day comes to about 2k samples a day. Under the 4-epoch cap per 7 days, that needs at least about 3.6k unique clips a week; aim for 10k.
  - At about 5 s per clip plus ASR, that is about 14 h of idle GPU4 time. Storage is about 1-2 GB in FLAC.

## 6. Staging layout (all products)

Root: `/mnt/lexar/omni-campaign/run0-intake-staging/<product>/v1/`. The Lexar is exFAT: no symlinks, no owner-only modes.

| Product | Contents | Provenance tag |
|---|---|---|
| `strata-intent-bank` | `records.jsonl` (intent twins), `ambiguous.jsonl` (held, not for training) | `strata-gen` |
| `strata-edit-instructions` | `instructions.jsonl` (COCO id, source sha, instruction, kind qwen/procedural) | `strata-gen` |
| `coco-edit-pairs` | `images/<coco_id>-<n>.png` + source sha, `records.jsonl` (build_edit) | `media-teacher` (+ `procedural`) |
| `coco-tts-captions` | `audio/<coco_id>.flac`, `records.jsonl` (build_concept), ASR receipts | `media-teacher` |
| `strata-kd-gen` | §3 layout | `strata-gen` |

Each product needs a `manifest.json` with:
- source, licence, counts per modality and task, bytes, a sha256 list;
- generator and teacher ids: the Q4 GGUF shard shas, the llama.cpp commit, the gate version, the Qwen-Image / Qwen3-TTS ids and graph shas;
- the sampling parameters, the held-out registry version, the date;
- `authorship_checked: true`.

Each also needs a README.

**Naming:** no product name, batch id, `source_id` or `origin` may contain `claude`, `codex`, `capture`, `refill`, `selfgen` or `sage`. Run 0's row guard matches these case-insensitively (`DEFAULT_ROW_DENY`) and drops the row. Note that text_refill's own `prompt_source` string contains "no Claude/Codex text". Keep such notes out of tag fields.

**Wiring:** later, through area B's tools: `run0_build_manifest.py` new sources, then a manifest bump. Never write into `run0/data/parts/` or the live intake drop zones.

## 7. Blockers and decisions for the lead

1. **Background admission is blocked by the 40 GiB RAM floor.** MemAvailable is 29 GiB and swap is full. Before any client runs, choose one:
   - the Strata lane lowers `--mem-floor-gib` (a `strata-gate` restart: Relay STATUS on thread `52468f63…`, a quiet window, backup and rollback);
   - the media lane frees the 57.6 GiB ComfyUI RSS.

   Both gates need the same decision.
2. **gen-worker-text after the Q4 switch:**
   - the toolcall_selfgen GGUF lock fails every toolcall batch;
   - the hardcoded IQ3_S `GGUF_SHA` / `MODEL_ID` mislabels every text_tasks batch since 10-27 11:04 MDT.

   Fix the worker and relabel those batches. The text-intake manifest licence string also says "RVN Strata".
3. **Teacher identity:** register `strata-q4kxl-abl-v2` as a new teacher in omni_lab. Decide whether run 0 mixes it with the IQ3_S caches under the one `strata` weight (they are different checkpoints).
4. **Capture RAM:** n_probs responses make capture's queue estimate about 40× too low. Either keep `n_predict` <= 512 and watch gate RSS, or ask the Strata lane to strip probabilities before capture.
5. **Unjudged COCO edits:** with no vision judge until D2 lands, decide whether `judged: false` edit pairs train at full weight, at token-presence weight only, or wait.
6. **Ambiguous intents:** a `[text]` clarifying question written by Strata, or excluded.
7. **Media-lane agreement:** the duty cycle for edit renders and the TTS block windows on GPU4.

Not checked: recent Relay posts for a live Strata PAUSE (`session.py replay` returned HTTP 422 here). The builder must read the hook context or sync before any Strata-lane action. No omni_lab run was logged, because this scout generated nothing.
