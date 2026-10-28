# Support scout: "[Cyril quote withheld]" (run 0, 2026-10-28, read-only)

**Task.** Cyril's 2026-10-22 order (ruling `cyril-run0-decisions-2026-10-22`) was "[Cyril quote withheld]". This scout lists everything that phrase covers and says whether run 0's area-B pipeline already takes it in.

**How it was done.** Read-only throughout:
- The plan: §0, §2.5, §3 and the Addendum.
- `survey_data.md`.
- The worktree `claude/run0-b-20261022` at HEAD `96ed2b9`.
- The AI server: listings, manifests, `source.json` files, and read-only SQL against `omni_lab` and `chimera_corpus`.

Nothing was written on the server apart from one helper at `/tmp/scout_q.py`. No GPUs were touched.

**Notation.**
- `$OC` = `/home/cereal/omni-campaign`
- `$L` = `/mnt/lexar/omni-campaign`
- `$R` = `$OC/run0/data` (area-B root)

**Disk now:**

| Mount | Free | Note |
|---|---|---|
| `/` | 114 GB | |
| Lexar | 381 GB | |
| NAS | 1 MB | 100 % full: read only |

Host RAM available is 28 GiB.

**Staging location.** Neither §2.5 nor §3 names an intake staging directory. Area B writes to `$R/parts/<source>` and `$R/staging/<source>`. So per the contract, staging for new products is `$L/run0-intake-staging/<product>/v1/`, which does not exist yet.

## 0. Bottom line

1. **The canonical store is only partly canonical.**
   - `corpus/canonical` v1 is `chimera_corpus.corpus_version` #1. It is sealed 2026-10-15 and holds 91,984 records, "adopted by record".
   - In practice it is the registry staging file `$OC/distill/corpus/staging/rows.jsonl` (279 MB) plus a dedupe index.
   - Only **one** corpus version has ever been sealed. None of the roughly 16k intake items admitted since 10-14 is in a sealed version.
   - Run 0 does not read the canonical records. It reads the derived token view `text_clean_v1`.
2. **Area B already takes in our text corpus, but only its text part:**
   - `text_clean_v1`: 12.35M tokens of registry text, with agent-trace documents row-guarded out;
   - the K3 rows;
   - `toolcall_flash_v1`;
   - `bfcl_native_v1`;
   - the 10-22 snapshot of the Strata intake;
   - Cyril's speech.
3. **Four parts of "our corpus" are not in run 0 at all:**
   - the registry's **~15.1k media rows** (image/OCR/music/audio/TTS/speech/video/sound);
   - the **NAS PDF corpus** (5,869 PDFs, 36.5 GB, no text extracted yet);
   - **intake growth since 10-22**: about 10x more judged images, about 4x music, about 5x video, about 6x Strata text;
   - a **memory-harvested tool-call corpus**, which was never built (see §3.3).
4. **About half of the registry's tool route is agent-written.**
   - 6,567 of 14,329 tool rows carry Claude/Codex/frontier memory-trace dataset ids. Registry-wide that is 1.41M of 44.0M `tok_chat`.
   - Inside `text_clean_v1`: 3,705 of 22,970 documents, 386,146 tokens.
   - The run-0 row guard drops these by `dataset_id` (`DEFAULT_ROW_DENY`: `claude`, `codex`, `memory_batch_` …). That guard is correct and must stay. It also means the "tool" part of our own corpus is thinner than its row count suggests.
5. **"Cyril's speech" is TTS.** It is Cyril's authored lines rendered by Qwen3-TTS voices (Ryan, Aiden, Vivian, Serena…), not his microphone.
   - Both `speech/` drop zones (Lexar and NAS) are empty. No `own-*` session exists.
   - It is cleared and already in run 0: 191 train lines → 382 records.

## 1. Inventory

The "Run 0?" column means: **IN** = in an area-B part or manifest builder now; **PARTIAL**; **NO**.

| # | Item | Location | Size / count | Format | Licence / consent | Exclusions inside it | Run 0? |
|---|---|---|---|---|---|---|---|
| 1 | Canonical corpus v1 (`corpus/canonical`, chimera_corpus v1) | `$OC/distill/corpus/staging/rows.jsonl` (279 MB); index `$L/corpus/v1/adopt_v1_index.json` (91,984 rows, 15,938 media, 123,637 text keys); manifests `$OC/distill/.scratch/corpus_manifests_1.0.1.tgz` | 91,984 records; 44.0M `tok_chat` (Q38 tokenizer counts) | JSONL records with instruction/response/render, route, licence, `license_tier`, sidecars | `license_tier`: blank 51,094; explicit_allowlisted 14,754; permissive 12,344; internal_train_candidate 7,227; attribution 4,547; PD 1,669; SA 167; owned 182. Cyril 10-14: "[Cyril quote withheld]" | Agent traces 1.41M tok (6,567 tool rows) | **PARTIAL**: only via the `text_clean_v1` view. Canonical text recovery into role-structured records is still owed (memory `omni-corpus-tokenizer-agnostic`) |
| 2 | Curated high-value set = omnicoder_2026 registry (`chimera_omni/data/curated_shards.py` reads `MASTER_SHARD_INDEX_POINTER.json`; this is the "neural-forge registry") | NAS `/mnt/ai_data/omnicoder_2026/datasets/` (pointer: 631 entries, 640 shards); mirror on root `$OC/distill/corpus/registry_mirror` (525 MB, sha-equal to the 09-02 walk) | 94,318 rows: trained_base 28,025, v37 8,913, addenda 4,964, ready_untrained 52,416 | JSONL shards + media sidecars | as #1 | as #1 | Same rows as #1. Text routes **IN** (below). Media routes **NO** |
| 2a | Registry text view `corpus/text` | `$OC/distill/corpus/text` | 4,883 × 4096 = 20.0M tok (text 9.67M, tool 2.79M, code 2.67M, long 2.34M, math 1.29M, reasoning 0.83M…) | Q38-id safetensors + Q38/Strata top-64 caches | — | contains the held-out documents | NO (superseded by 2b) |
| 2b | `text_clean_v1` (held-out removed repack of 2a) | `$OC/distill/corpus/text_clean_v1` (119 MB + `documents.jsonl`) | 3,015 rows, 12.35M tok, 22,970 documents | as 2a; `documents.jsonl` gives `dataset_id` per document | — | 3,705 agent-trace documents / 386,146 tok. Top source: `memory_batch_000233_claude_ice_agent…` (141k), then `manual_cleanup_batch_000156_v10_memory_harness` (68k) and `codex_*` sets (~28k) | **IN**: bucket text_kd 22 %, cap 6, row guard drops the traces |
| 2c | Registry media rows | sidecars: 12,680 refs on the NAS, 2,003 on root `$OC/distill/corpus/media_store`, 445 other | ~15.1k rows with media: image 3,767, music 2,619, OCR 2,305, audio 1,636, TTS 1,541, speech 1,281, video 1,082, sound 444 | rows + sidecar files | per row (`license_tier`) | check authorship of the response text per row; drop agent traces (60 image, 30 OCR, 20 video) | **NO**. This is the only on-disk **sound** (444) and **OCR** (2,305) material |
| 3 | K3 rows | `$OC/distill/corpus/k3/pass-0..3` + `$L/teacher_cache/k3/eb35851b2496a628-provisional-bf16` | 488 rows, 1.99M Qwen tok / 1.89M K3 positions | dual-tokenized safetensors + xtok | registry-derived | — | **IN**: bucket k3 10 % (D1: K3 we have) |
| 4 | `toolcall_flash_v1` | `$OC/distill/corpus/toolcall_flash_v1` (65 MB) | 1,599 rows, 3.0M tok, 0.69M supervised | Q38-id safetensors + `rows.json` | Strata output (Cyril 10-12 cleared) | — | **IN**: tool bucket 8 % |
| 5 | `bfcl_native_v1` | `$OC/distill/corpus/bfcl_native_v1` | 500 rows, 0.42M tok (49k supervised) | safetensors | BFCL v4, public | 62 held-out groups screened | **IN** |
| 6 | Intake (chimera_corpus `item`, train split, `admitted`), all from gen-worker-text/media | drop zone `$L/corpus-store/chimera-intake/` (NAS mirror `/mnt/ai_data/chimera-intake`); DB `chimera_corpus` | text: `strata-toolcall` 2,130, `strata-reasoning-math` 680, `strata-instruction-code` 327, `strata-long-context` 117 (= 3,254 usable); image: t2i 2,546, edit 757, HY3D source 360, LTX 62; mesh 380; music: ACE 1,096 + H3 SOL audio 45; video H3 SOL 155 | content-addressed assets + provenance sidecars | local teacher outputs (Qwen-Image 2.1, ACE-Step, H3, LTX, Hunyuan3D non-commercial); Strata text cleared 10-12 | **`strata-capture` 3,580 admitted text items = raw capture spool: EXCLUDE** (denylisted); media-judge captions (Qwen3.6-27B) never used as text | **PARTIAL**: parts were built 2026-10-22 from export `intake_export_20261022T201247Z.json`: t2i 247, edit 32, ACE 284, video 33 (19 + 14 AV), strata-text 479 (148 chat + 331 tool). The rest waits for `run0_intake_sync.py sync` |
| 7 | Cyril's speech | built part `$R/parts/cyril-speech`; staged `$R/staging/cyril-speech` (85 MB, 192 `original.wav`, `speech_plan.json`, `tts_receipt.json`); source `$L/codex-adopted-2026-10-14/speech-native-caches-v2/payload` (6.4 GB: 186 CE, 194 audio CE, 186 flow) | 191 train lines → 382 records (191 ASR + 191 TTS), 41k positions | WAV ~4 s, 24 kHz + transcript | Cyril's own (10-18: "[Cyril quote withheld]") | none. The audio is Qwen3-TTS voice renders of his authored lines | **IN** (speech bucket) |
| 7b | Cyril's own microphone recordings | `speech/own-*` in either drop zone | **0** (both empty; NAS `training_data/rvc` empty) | — | consent = owner | — | n/a: nothing to take in |
| 8 | NAS ai-training PDF corpus `ai-frontier-research-2026` | `/mnt/ai_data/training_data/ai-training/ai-frontier-research-2026/pdfs` (NAS only); `ai-training/SHA256SUMS` = 5,869 lines | 5,869 PDFs, 36,535,248,604 B; 235 top-level entries (~127 named papers/model cards + 108 `pass_*` dirs). Token count unknown: about 150M tokens if an average PDF holds about 25k tokens **(projection)** | PDF only; no extracted text, no records, not in chimera_corpus (`documents/` drop zone empty) | third-party papers and model cards, mixed copyright (arXiv licences vary); falls under Cyril's 10-14 licence ruling; record per file | documents, not model outputs, so not excluded. Screen against hold-out v3. The literature survey (§221) warns against raw web-text replay at 2B; these are curated papers, so check the mix share and cap rather than skip | **NO** |
| 9 | Memory-harvested tool calls | PostgreSQL `agent_memory` (`agent_memory_events`, 1.22M events). There is no `agent_memory_tool_calls` table; the backend code has none | 673,808 Codex + 205,635 Claude Code `tool_call` events with input/response (REALIGNMENT_2026-10-17 §98) | event rows + redacted payload archives | assistant turns are Claude/Codex/GPT/Sol output; provider terms forbid training on them (STATE_2026-10-18 item 1: Claude declined; the 10-18 plan made it search/audit only) | everything except Cyril's prompts and the real tool **results** | **NO**, and it must not go in as written (see §3.3) |
| 10 | Q38-written tool calls (`$OC/distill/toolcall/rescore`, `run-q38-q4km-*`), `selfgen` | `$OC/distill/toolcall`, `$OC/distill/corpus/selfgen` | 1,467 rows / 0.72M; 2,532 rows / 1.0M | — | small-model output | all of it | **EXCLUDED** (denylist) |
| 11 | Codex-adopted tool/eval sets (`tool-train-168-*`, `codex-local-tool-seed-oct13-v2`, `codex-local-recovery-eval-*`) | `$L/codex-adopted-2026-10-14/` | not counted | caches | Codex-written | treat as Codex output | NO; keep excluded |
| 12 | `d2_refill` (codex-native v7/v8) | `$OC/distill/corpus/d2_refill` | 88k KD positions | — | Codex pipeline | all | **EXCLUDED** (denylist) |
| 13 | `vision_in` | `$OC/distill/corpus/vision_in` | 5,896 rows / 7,572 images (+512 video-in clips) | — | answers written by Q38-VL | answers excluded; the images and prompts are reusable | NO (denylisted by path). It could come back as images + prompts re-answered by Strata once the D2 vision fix lands |
| 14 | Media prompt sets `corpus/media` | `$OC/distill/corpus/media` (QIMG 2,048 prompts, ACE 1,024, HY3D 256, H3 512) | 2.7 GB | — | B-0 authorship check | Claude/Codex-written prompts are conditioning-only | h3set **IN** (conditioning-only, 506); the others are only through their teacher caches |
| 15 | Public text on the Lexar (`omnicoder_0.1.0/data/text`: `fineweb_edu.txt`, `synth.txt`, `wikitext2.txt`, 4.1 GB; `wiki/enwiki…bz2`, 6.2 GB) | `/mnt/lexar/omnicoder_0.1.0/data/` | 10.3 GB raw | plain text / XML dump | FineWeb-Edu ODC-By; Wikipedia CC BY-SA; **`synth.txt` author unknown** | `synth.txt` is excluded until its provenance is known | NO (and arguably not "ours"; listed for completeness) |
| 16 | Public media already in run 0 | COCO 2017 (118,287 × 2 tasks), LJSpeech (26,196), LibriSpeech (4,551), MAESTRO (3,960), ModelNet10 (2,000), hy3d_voxel64 (333) | — | — | as in `source.json` (MAESTRO NC recorded) | — | **IN** |

**omni_lab registry** (38 datasets, 131 versions). Every dataset above that is a token view or teacher cache is registered:
- `corpus/canonical` v1;
- `corpus/distill/{text, text_clean_v1, k3, long, media/*, selfgen, vision_in, staging, eval_split, d2_masks_v1/v2}`;
- `corpus/external/{librispeech, modelnet10}`;
- `corpus/derived/hy3d_voxel64` v1;
- `corpus/holdout_registry` v3 (92,537);
- 14 `teacher_cache/*`;
- `training_config/{fullkd_data v3, media_data v5}`.

**Not registered:**
- the PDF corpus;
- the registry media rows as a dataset;
- the intake as versions (only corpus_version 1 exists);
- run 0's own `run0/data` (`run0_intake_sync.py register` has not run).

## 2. What area B already does (worktree `claude/run0-b-20261022`, HEAD `96ed2b9`)

**Builders.** `scripts/run0_build_manifest.py` has these subcommands: `coco`, `intake`, `intake-text`, `ljspeech`, `librispeech`, `cyril-speech`, `maestro`, `modelnet`, `hy3d`, `h3set`, `manifest`.

**What `manifest` adds** on top of the built parts:
- `text_clean_v1` (all shards, `documents.jsonl` row guard, Q38 + Strata caches);
- `k3#pass-0..3` (+xtok);
- `toolcall_flash_v1`;
- `bfcl_native_v1`;
- `long_v1` (present but `enabled: false`).

**Not built yet.** `$R` holds 14 built parts but **no `run0_data_v*.json` yet**.

**Guard** (`training/run0_data.py`). Path denylist: `selfgen`, `tool-selfgen`, `toolcall/rescore`, `toolcall-q38`, `qtext_vl`, `vision_in`, `strata-capture`, `capture-export`, `d2_refill`, `refill-codex`, `sage`. Row deny: `claude`, `codex`, `anthropic`, `openai`, `gpt`, `gemini`, `fable`, `opus`, `sonnet`, `grok`, `memory_batch_`, `trace_rewrite`, `memory_harness`, `capture`, `refill` …

Note the frontier-model sets named `sol_*` and `nemotron_*` are caught only because their ids start with `memory_batch_`. A new source with a bare `sol_…` id would pass the guard.

**Streaming intake.** `scripts/run0_intake_sync.py` reads only `admitted` train items. It builds `<base>@<tag>` parts with the same guard, bumps the manifest and registers it in omni_lab. It covers row 6 of §1.

**No builder exists for:**
- the registry media rows (2c);
- the canonical records as records (1);
- the PDF corpus (8);
- any memory-derived tool data (9);
- intake documents/speech (both empty anyway).

## 3. Gaps and what each would take (for the lead; nothing here was started)

### 3.1 Intake growth since 10-22 (cheapest, biggest)

Run `run0_intake_sync.py sync` (area B's own tool). The text part gives about 3,254 Strata conversations vs 479 now.

### 3.2 Registry media rows, about 15.1k

This needs:
- a new area-B builder `registry-media`, reading sidecars from the root `media_store` (2,003 refs) and staging the 12,680 NAS-resident ones to `$L/run0-intake-staging/registry-media/v1/` (read-only NAS reads, CPU, nice/ionice);
- a per-row authorship check on response text, plus the agent-trace guard;
- a hold-out v3 screen.

It is the only source of sound/environmental audio (444) and OCR (2,305) we hold. That bears directly on the "audio gap".

### 3.3 Tool calls from memory

Cyril's 10-17 words are "[Cyril quote withheld]". The 10-18 decision (Claude declined on provider terms, recorded in `STATE_2026-10-18_FULLKD_LAUNCH.md`) and the run-0 row guard both keep the Claude/Codex assistant turns out.

The compliant form:
- Cyril's own prompts and the real tool results as context, `loss_mask` false;
- assistant turns **re-written by Strata** (D3, background priority) through the strata-gate client (B-6), so that only Strata turns are targets.

This needs:
- a raw-SQL extractor over `agent_memory_events` (no typed tool-call table exists);
- redaction (secrets were redacted at capture, so verify);
- the hold-out screen;
- staging at `$L/run0-intake-staging/memory-toolcall/v1/`.

Flag this to Cyril as the reading of his order, not as a rule change.

### 3.4 PDF corpus

This needs:
- CPU text extraction (e.g. `pdftotext`/PyMuPDF, small RAM, `nice`) reading the NAS and writing canonical records to `$L/run0-intake-staging/pdf-corpus/v1/` (estimate a few GB of text **(projection)**);
- a per-file sha checked against `SHA256SUMS`;
- per-file licence recorded;
- a hold-out v3 screen and near-dedupe;
- a token view with the Qwen3.5 tokenizer.

There is no teacher signal for it: Strata cannot score prompt tokens (no echo), and `teacher-score` needs the P40s that run 0 holds. So it enters as CE-only text, or waits for idle scoring. Cap its share.

### 3.5 Canonical v2

Seal a chimera_corpus version 2 (v1 + admitted intake) so run 0 pins a corpus version rather than the live export. Recovering the registry text into role-structured canonical records remains the owed tokenizer-agnostic work.

## 4. Sources read

- `RUN0_BUILD_PLAN.md` (§0, §2.5, §3, area B, Addendum);
- `survey_data.md`;
- `STAGE_D_RULINGS_2026-10-12.md` (10-22 entry);
- `REALIGNMENT_2026-10-17.md`;
- `STATE_2026-10-18_FULLKD_LAUNCH.md`;
- `distill/SPEECH_LANE_PLAN_v2_2026-10-14.md`;
- memory files `ai-training-corpus-location`, `omni-speech-data-is-cyrils`, `omni-full-distill-k3-primary`, `omni-corpus-tokenizer-agnostic`, `omni-data-grows-over-time`, `omni-lab-db`;
- worktree files `run0_build_manifest.py`, `run0_intake_sync.py`, `run0_stage_data.sh`, `run0_data.py`, `curated_shards.py`;
- on the server:
  - `$R/parts/*/source.json`;
  - `text_clean_v1/manifest.json` and `documents.jsonl`;
  - `corpus/staging/stage_receipt.json` and `rows.jsonl`;
  - `MASTER_SHARD_INDEX_POINTER.json`;
  - the intake `_README.txt`;
  - SQL: `omni_lab` `dataset`/`dataset_version`, and `chimera_corpus` `item`/`asset`/`corpus_version`;
  - `agent_memory.py --json status`.
