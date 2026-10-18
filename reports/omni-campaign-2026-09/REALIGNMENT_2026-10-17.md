# Chimera Omni realignment, 2026-10-17

**Written:** 2026-10-17, about 20:45Z, by Claude (session 753087a5).

**Basis:** all 275 of Cyril's Chimera Omni and folding prompts in the PostgreSQL memory (2026-07-01 to 2026-10-17), read in full. Agent-written briefs and relay messages were excluded. The harvest is reproducible: `agent_memory_events`, `event_type='user_prompt'`, topic filter, de-duplicated. The reading file is in this session's scratchpad as `omni_prompts_final.txt`.

**Status:** this replaces the D2 plan in `STAGE_D_RECONCILED_PLAN_2026-10-14.md` and `STATE_2026-10-17_D2_LAUNCH.md` §2.

---

## 1. What Cyril has asked for (his words, dated)

1. **The model.**
   - **2026-08-04:** combine "[Cyril quote withheld]".
   - Add whisper/STT+TTS, acestep, the video model (H3 since 08-28 and 09-01), qwen image and qwen image edit "[Cyril quote withheld]".
2. **The shape.**
   - **2026-09-01:** a dense Chimera Omni that "[Cyril quote withheld]", input and output of all modalities.
   - **2026-09-01:** "[Cyril quote withheld]".
   - **08-28:** encoder-free.
3. **The order.**
   - **2026-09-01:** fold → "[Cyril quote withheld]" → finetune on the curated dataset. Checkpoint and clone before every stage.
   - **2026-10-07:** "[Cyril quote withheld]".
4. **Today (2026-10-17).**
   - "[Cyril quote withheld]"
   - "[Cyril quote withheld]"
   - "[Cyril quote withheld]"
   - "[Cyril quote withheld]"
   - "[Cyril quote withheld]"
5. **Folding continues in parallel.**
   - **10-07, 10-14:** keep optimizing folding to raise intelligence retention and density. Astra-style work runs on the P40s.
   - **10-13:** the first trunk optimized first-token match, but "[Cyril quote withheld]" is the better objective.
6. **Standing rules.**
   - Finite numbers are diagnostics. Judge intelligence by real outputs (07-21, 08-11, 10-06).
   - No quantization counts as folding (07-17, 08-12).
   - No dense-to-MoE tricks (08-12).
   - Cache parent outputs once (07-02, 07-16).
   - Use every GPU; Chimera has priority (09-07, 10-06).
   - Don't create waste (10-17).

## 2. Where the campaign drifted (honest list)

| # | What the queues did | What Cyril asked for |
|---|---|---|
| 1 | D1 and D2 were **LoRA**. D2 trained 252,444,672 adapter parameters on the folded OMNI1D1. The base weights of Chimera Omni never moved. | Full distillation of every weight. |
| 2 | The D2 teacher signal was about **91 % Qwen 3.8 27B** (17.5M registry tokens + 1.15M selfgen + 3.0M tool-selfgen, all with Q38 logits) and about **9 % Kimi K3** (2.0M tokens across 4 passes, deferred to the 20M rung). | Kimi K3 takes precedence ("99% kimi k3"). Qwen is a "smidge for personality". |
| 3 | `selfgen` (1,145,085 tokens) and `tool-selfgen` (3,002,570 tokens) were **written by Qwen 3.8 27B Q4** through llama-server. Today's run 3 was too, and it was stopped at 20:19Z. | "[Cyril quote withheld]". |
| 4 | The roughly 880k real tool calls already in memory (frontier models: GPT-6 Astra, Sol 6.1, Opus/Fable/Sonnet 5.x) were never used. Instead, the parent invented new ones. | "[Cyril quote withheld]". |
| 5 | The student is the fold-built OMNI1/OMNI1D1. Its bank-v1 review was **0 useful / 8 partial / 7 miss**, so it is a weak base. LoRA on a weak base can only nudge it. | Initialize from the folds, then **distill from the original parents** with every weight trainable. |
| 6 | Agents kept building gates, receipts and coordination layers. GPUs sat idle for a day while a fully ready D2 waited to be armed. | Ship, use the GPUs, don't gate on numbers. |

Nothing in rows 1-4 was Cyril's instruction. They came from agent-written plans (DISTILL_DESIGN 10-12, rulings 10-14/15) that inherited a LoRA engine because it was the only training cell measured on the 3090s. They were never checked against his words.

## 3. Corrected pipeline

| Stage | What | Student | Teachers (full original parents, outputs cached once) | Trainable |
|---|---|---|---|---|
| **F** (exists) | Composite fold of all sources into one dense Chimera Omni | — | — | — |
| **KD-text** | Full-parameter logit distillation, text + tool use + long context | Chimera Omni, initialized from the folds (OMNI1 / OMNI1D1 composed) | **Kimi K3 (primary weight)** + Qwen 3.8 27B Uncensored (secondary weight) | **all weights** |
| **KD-media** | Full-parameter distillation of every modality in the same single model | same model | H3 (video+audio), Qwen-Image / Edit, ACE-Step 1.5 (music), STT/TTS teachers, 3D teacher, Qwen VL tower (vision-in) | all weights |
| **T** | Finetune on our curated corpus, including the harvested tool calls (ground truth) | same | — | all weights |
| **Q** | Q4, then a 3090 test at 256k | — | — | — |

Clone before every stage, as before.

**Data rules from here:**
- Training text comes from:
  - **our own corpus** (the registry, about 17.5M tokens, plus the 100k+ rows Cyril has curated);
  - **tool-call trajectories harvested from memory**, written by frontier models.
- Teachers only *score* that text (logits). No small or local model writes training data.
  - Qwen 3.8 27B counts as small for generation.
  - Kimi K3 cannot free-run here: 92 × 14.8 GB of packed experts is 1.37 TB, more than the host RAM. It stays a teacher-forced logit teacher.
  - Qwen 3.8 Flash Next is not on disk.
- **Remove from every training mix:**
  - `selfgen` (Q38-generated);
  - `tool-selfgen` runs 1-3 (Q38-generated);
  - `refill-codex-v7/v8` (scripted, templated assistant turns, 168 two-tool scenarios), unless Cyril says otherwise.
- Keep `tool-bfcl-native` (BFCL ground truth, 0.4M tokens) as data, never as an eval.

**K3 coverage is the bottleneck.**
- Measured today: K3 pass 0 on GPU6 with tool 1.3.0 runs about 5 min per layer, so about 7.75 h per 0.5M tokens.
- The current K3 set is 2.0M tokens (4 passes, about 31 h).
- To make K3 the main signal over the roughly 17.5M-token registry plus the harvested tool calls, K3 has to score about 20M+ tokens. That is about 40 passes, or about 13 days on GPU6 alone.
- Running K3 on the 3090s too (the card-class gate exists, 1.3.0 is bit-identical on GPU6) cuts that roughly in thirds.
- Planning answer: start KD-text as soon as the trainer exists, weighting K3 on every row it covers, and keep extending K3 coverage in the background. Later epochs get more K3 rows.

## 4. Engineering that does not exist yet

1. **Full-parameter streamed distillation trainer.** Today `streamed_kd_trainer.py` 1.3.4 is LoRA-only (`--lora-rank`).
   - **Needed:**
     - layer-streamed fwd+bwd on the 3090s with every student weight trainable;
     - bf16 weights resident in host RAM;
     - an optimizer whose state fits next to the K3 job (Adafactor with stochastic-rounded bf16, or 8-bit AdamW with CPU-resident state, about 60-110 GB host);
     - per-layer gradient transfer and update;
     - a multi-teacher KL that weights K3 rows above Q38 rows;
     - rolling safety checkpoints (54 GB each, so keep 2) and exact resume.
   - **EST throughput:** about 100-150 tok/s on 2×3090 (P2's LoRA cell was 238 tok/s on one card; full parameters add the dW matmuls and the update traffic).
   - That is about 2 days per 20M tokens. A probe on 2 layers comes first.
2. **Memory tool-call layer (Cyril's ask).**
   - **What memory has today:** 673,808 Codex and 205,635 Claude Code `tool_call` events with tool input and response.
     - Codex events also carry the model and `turn_id`.
     - Claude Code events carry neither a model nor a turn id; their `prompt_id` is only inside the payload JSON.
     - Assistant text between calls is not in the events; it is in the transcripts (`transcript_path`).
   - **Build:**
     - a typed table `agent_memory_tool_calls` (event_id, agent, model, session, turn/prompt id, sequence, tool_name, arguments jsonb, result, is_error, duration) through raw SQL, backfilled from the 880k rows;
     - a turn-assembly view (user prompt → ordered calls + results → final answer);
     - a hook update so new calls arrive typed, with the Claude Code model resolved from the transcript;
     - an exporter that turns turns into native Qwen3-Coder XML tool conversations through the intake, with credential/PII screening and the held-out screen.
3. **K3 on the 3090s.** `k3_multicard.sh` plus the card-class gate need the same provisional branch that `gpu6.sh` 1.5.0 got.
4. **Media KD** in full-parameter form, the H3 items adapter (`items.jsonl` has no top-level `id`), and the speech E2/E3 tools.

## 5. GPUs now and next

| Card | Now (20:45Z) | Next |
|---|---|---|
| GPU6 RTX 8000 | **K3 teacher pass 0, resumed 20:36Z from saved state** (full Kimi K3 parent, provisional bf16, approved by Cyril 15:27Z) | Passes 1-3, then more K3 passes over the registry and the harvested tool calls |
| GPU0 + GPU4 3090 | idle (LoRA D2 stopped 20:19Z) | 2-layer probe of the full-parameter trainer, then **KD-text** with all weights trainable. Until then, K3 passes if the card-class gate is extended. |
| GPU1/2/3/5 P40 | idle (Q38 tool generation stopped 20:19Z) | Q38 (secondary teacher) logits over the harvested tool-call corpus once it exists (53 tok/s per P40 measured). Folding optimization with a multi-token objective (Cyril 10-13). |

## 6. Decisions for Cyril

1. **Student.**
   - Recommended: initialize from the folds (OMNI1D1 composed) and train every weight. This is his 10-07 plan.
   - Alternative: OMNI1 without the D1 LoRA deltas.
2. **Teacher weights.**
   - Recommended: K3 1.0 on every K3-covered row and Q38 0.2-0.3 everywhere ("smidge").
   - Or K3-only rows until K3 coverage is large.
3. **`refill-codex-v7/v8`.** Recommended: drop them; they are scripted.
4. **K3 on the 3090s before the trainer is ready?** Recommended: yes. It is idle time otherwise, and K3 is the bottleneck.

## 7. Addendum 2026-10-17 (about 23:45Z): approved and in progress

**Approved by Cyril.** Full-parameter [Cyril quote withheld]. Drop selfgen, tool-selfgen and refill-codex. K3 passes go on the 3090s until the trainer is ready.

**Full-parameter trainer** (`distill/tools/streamed_full_kd_trainer.py` 1.0.0).
- **Engine:** the 1.3.4 streamed engine with every text weight trainable. The bf16 master lives in host RAM (about 47 GB, 851 tensors for OMNI1D1). Gradients accumulate in fp32 per layer, and the Adafactor update (no momentum, factored, stochastic rounding) is fused into each layer's backward. The head and norm are updated after the head phase, and the embedding by sparse rows.
- **CPU tests:** 5/5 pass (SR unbiased, Adafactor = reference, blockwise = unblocked, exact resume, store round trip).
- **grad-parity on Qwen3.5-0.8B, fp32, P40, all 320 tensors, tied head:**

  | Path | Worst relative difference |
  |---|---|
  | whole-layer | 1.0e-7 |
  | chunked (row groups + FFN slices) | 2.6e-5 |

- **Real-geometry probe (GPU4, 4 layers at FFN 16,640, 8 × 4,096):** 23.5 s/step, peak 17.3 GiB, optimizer state 3.3 MiB. About 120-125 tok/s on one 3090 extrapolated to 64 layers (EST).
- **Data:** `fullkd_data_v2.json` is the D2 registry text + `tool-bfcl-native`, with `selfgen` / `tool-selfgen` / `refill-codex-v7/v8` dropped. It adds K3 rows from the provisional passes, staggered as each pass gets its xtok `READY` (pass 0 at 0, 1 at 3M, 2 at 6.5M, 3 at 10M tokens), each at fixed fraction 0.07. On K3 rows K3 has weight 0.8 and Q38 0.2; on text rows Q38 has 0.25; CE is 0.1.
- **Queues:** `distill/tools/fullkd_queue.sh` (GPU4, resume auto, waits on K3 markers) and `distill/tools/k3_prov_xtok.sh` (builds and verifies xtok for each provisional pass as it seals).
- **Host RAM:** two K3 passes (~87 GiB each) plus the trainer (~75-80 GB) do not fit in 251 GB. So the GPU0 K3 runner yields when training starts, and GPU6 keeps producing K3 passes.

**Tool-call data (Cyril: build it locally with Qwen 3.8 Flash Next and Kimi K3).**
- Claude/Codex-written tool calls are NOT used. Anthropic's and OpenAI's terms forbid training another model on their outputs, internal use included.
- **Writer:** `Qwen3.8-Flash-Next` abliterated IQ3_S, the Strata repack `jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata`.
  - The file is 88,882,765,184 B, sha256 `cb5958e5…`.
  - The license is Apache-2.0 / Qwen Community License 1.0, which has no output-training restriction.
  - It is being downloaded to `/mnt/ai_data/models/flash-next-strata/` and will be moved to the root disk after cleanup.
  - Its tokenizer is IDENTICAL to Qwen 3.8 27B's (248,077 ids, 247,587 merges, every special id, same Qwen3-Coder XML tool format), so `toolcall_selfgen.py` drives it unchanged.
- **Seeds:** Cyril's own prompts from memory (his words, legally his).
- **Execution:** real or sandboxed tools.
- **Scorers:** Kimi K3 (logits, primary; it cannot free-run here because 1.37 TB of experts would stream per token) and Q38. The tool schemas are re-written by Flash Next too.

**Tokenizer-agnostic corpus (Cyril 2026-10-17).** "[Cyril quote withheld]".
- The canonical corpus becomes TEXT and structured records: conversations with roles, tool calls and results as structured JSON, media as content-addressed files with captions and transcripts, and provenance, license and split per record.
- Every tokenized safetensors corpus and every teacher cache is a DERIVED view, keyed by (corpus version, student tokenizer + chat template, teacher).
- Rebuild rule: a new Chimera tokenizer re-renders the views from the canonical records. Nothing in the canonical store is ever stored as token ids only.
- Existing views (registry text, the Q38 caches) stay valid for the current student. Their source text has to be recovered into canonical records. That is pending work: `text_clean_v1/documents.jsonl` gives doc provenance; the original document text comes from the registry sources.
