# Run 0 support: "our corpus" as run-0 views

Date: 2026-10-28, 07:20-08:00Z (first attempt, cut off by the usage limit) and 10:45Z onward (resumed). Ruling `cyril-run0-decisions-2026-10-22` ("[Cyril quote withheld]"). Plan: `RUN0_BUILD_PLAN.md` §2.5 (the text-KD row names "the canonical `staging/rows.jsonl` derived view") and the Addendum. Scout: `support_scout_corpus.md`.

Notation: `$ST` = `/mnt/lexar/omni-campaign/run0-intake-staging`, `$OC` = `/home/cereal/omni-campaign`.

## 0. Answer first

1. **Three staging products exist, verified and registered in omni_lab:**

   | Product | What | Size | omni_lab |
   |---|---|---|---|
   | `$ST/pdf-corpus/v1` | NAS PDF corpus as extracted text: 5,839 documents, 126.2M Qwen3.5 tokens, 141,746 pages | 471 MB | `corpus/run0-views/pdf-corpus` v1 (dataset 560, version 569, sealed), build run 40332 |
   | `$ST/registry-text/v1` | canonical corpus v1 text rows as tokenizer-agnostic documents + chat records, under an authorship allowlist: 10,705 documents + 70 records, 5.99M Qwen3.5 tokens | 36 MB | `corpus/run0-views/registry-text` v1 (dataset 561, version 570, sealed), build run 40333 |
   | `$ST/registry-media/v2` | canonical corpus v1 media rows whose media and words are public: 1,397 rows (SonicMaster 1,242, MTG-Jamendo 155), 2,401 audio files | 6.72 GB | `corpus/run0-views/registry-media` v2 (dataset 562, version 571, sealed), build run 40335 |

   Each has `manifest.json` (source, licence, counts per modality/task, bytes, sha256 of every file, generator + teacher ids, authorship policy, hold-out screen, date), `README.md`, and now `verify_receipt.json` + a `sealed` marker. `verify` re-hashes every listed file and fails on unlisted files; `dataset_version.sealed` is set from that receipt for exactly the manifest's sha256.
2. **Nothing else in the corpus scout needed a new view.** Every other item is already in run 0 or is excluded by the plan (§2 below):
   - curated high-value set = the same registry rows as canonical v1, so the two registry products cover it;
   - tool-call corpora: `toolcall_flash_v1`, `bfcl_native_v1` and the intake `strata-toolcall` items are in run 0 (area B's intake sync at 08:13Z: `intake-strata-text@20261028T081312Z`, 2,775 records, 1,799 of them tool chats); the registry's own tool rows are all refused by the allowlist; memory tool calls are excluded wholesale by plan §2.5;
   - Cyril's speech: all of it is already in run 0 (191 of the 192 plan rows → 382 ASR + TTS records; one render failed). No microphone recordings exist (both `speech/own-*` drop zones are empty).
3. **No GPU, no Strata, no media-gate, no NAS writes.** All builds are CPU-only under `nice -n 19 ionice -c3`, read the NAS read-only, write only `$ST`. Nothing was written into run 0's shard dirs; wiring is area B's job.
4. **RAM.** None of these products needs RAM while it waits. Build-time peaks were measured or bounded (§5). At train time they cost what area B's `RecordSource` costs (byte offsets, records parsed when served).
5. **Code:** chimera-omni branch `claude/run0-corpus-views-20261028`, commits `47539bb` (first attempt), `e1b0ab8`, `786bbc7`, `07b69bb`; pushed to origin. Deployed at `$OC/run0/code-corpus-views-07b69bb` (and `-786bbc7`), each with `DEPLOYED.json`.

## 1. Products

### 1.1 `pdf-corpus/v1` (built 07:20-07:52Z by the first attempt; verified and registered now)

- Source: `/mnt/ai_data/training_data/ai-training/ai-frontier-research-2026` (NAS, read-only), `SHA256SUMS` 5,869 entries (sha256 `db09c31c…`).
- **Every file was hashed and compared with its `SHA256SUMS` line before extraction:** 0 mismatches.
- File status (`files.jsonl`, one line per `SHA256SUMS` entry): ok 5,839; row_guard 29; garbled 1.
  - The 29 refused files are vendor system / model cards whose file names match the run 0 row guard (Anthropic, OpenAI, Google Gemini, xAI Grok). They quote model transcripts, so they stay out. The full list is in `files.jsonl`.
  - **Skipped as unreadable** (`unreadable.txt`): one file, `pdfs/pass_094_arxiv_backlog/arxiv-2603-05232v1.pdf` (garbled: under 50 % letters after extraction). No file failed `pdftotext`, timed out, or had no text.
- Extraction: `pdftotext 24.02.0 -enc UTF-8`, 6 workers, 3 GiB address-space cap and 240 s per file; reflowed paragraphs, de-hyphenation, page-number lines dropped. Exact duplicates and older arXiv versions collapse to one document.
- Hold-out registry v3, per paragraph (exact text + 13-word shingles): 770 paragraphs dropped.
- Tokens per document (Qwen3.5): p10 9,454, median 18,709, p90 34,802, max 376,036.
- Samples read (four random documents, head and middle): clean paper text with title, authors, abstract, tables flattened to lines, references kept. Prompt appendices inside papers (e.g. a D&D multi-agent system prompt in arXiv 2604.27660) are the paper's own content.
- Licence: third-party papers and model cards, mixed copyright; per-paper arXiv licences not fetched; internal training under Cyril's 2026-10-14 ruling. Each document carries its NAS path, sha256 and arXiv URL.
- Teacher: none. It is CE-only text (Strata cannot score text it did not write; `teacher-score` needs the P40s run 0 holds). Area B must cap its share.

### 1.2 `registry-text/v1` (built 07:54-07:55Z by the first attempt; verified and registered now)

This is the plan's "canonical `staging/rows.jsonl` derived view": canonical corpus v1 (`chimera_corpus` corpus_version 1, sealed 10-15; `$OC/distill/corpus/staging/rows.jsonl`, sha256 `cf64d6f2…`), text routes only, rebuilt as role-structured, tokenizer-agnostic records. Corpus v1 itself is not modified.

- Kept, by public set: v9 web-text 10,529 documents; FineWeb-Edu 176 documents; DAPO-Math 40, OpenR1-Math 24, DeepMath 6 chat records. 5,986,487 Qwen3.5 tokens.
- **Authorship is an allowlist, not a denylist.** The registry's dataset ids do not say who wrote a row, so a row is a target only when its `source_dataset` is a named public set, or it is the registry's v9 web-text sample. Codex-written lead-ins are user-side `agent-template` parts (conditioning only) or stripped. Area B's rev 2 adopted the same allowlist (`run0_build_manifest.py` "authorship" block, which names this script).
- Refused (rows / `tok_chat`):

  | Reason | Rows | tok_chat |
  |---|---|---|
  | no public source recorded (Codex sub-agent batch or omnicoder pipeline row) | 25,844 | 10.41M |
  | small-model output (Qwen teacher rewrites) | 5,743 | 5.31M |
  | agent-written / procedural (Codex pipeline batch under its own source label) | 11,534 | 4.77M |
  | agent tool trace (apply_patch / hook events) | 1,874 | 1.26M |
  | model-generated text in a public set (olmOCR, OpenThoughts, Nemotron, …) | 1,081 | 1.17M |
  | run 0 row guard (`memory_batch_` 6,558, `codex` 820, `memory_harness` 379) | 7,757 | 0.86M |
  | BFCL ground truth in a Codex JSON wrapper (BFCL is in run 0 as `bfcl_native_v1`) | 1,122 | 0.43M |
  | benchmark task specs rewritten by an agent | 446 | 0.22M |

  Per dataset id: `refused_by_dataset.json`.
- **Hold-out v3 dropped 8,237 rows, and that is correct.** Attribution of the 13-gram hits (first attempt's `s10.py`, re-run now): most hits are against `codex_recovery_heldout` (18,561 texts) and `eval_split` (3,832), which were built from these same v9 web-text and FineWeb rows. The rest are Codex template phrases ("separate evidence from interpretation, note assumptions or boundary conditions…") inside FineWeb rows. Duplicates dropped: 576.
- **Kept text checked for agent text:** a marker scan over all 10,705 kept documents (Codex header lines, the template phrases above, "Claude", "Codex", "as an AI", "2026") found only ordinary prose (e.g. "Claude Shannon", "icodextrin", "as an airman certificate", a 2025-2026 school year). None of the 176 kept FineWeb documents has a hit.
- The 70 records pass `check_records` from both the build's data contract (96ed2b9) and area B rev 2 (deployed `code-b-2d4c4d1`).
- **Overlap with what run 0 already has:** these are the same canonical rows that `text_clean_v1` tokenized. `audit-text-clean` puts 9.38M of `text_clean_v1`'s 12.35M tokens on rows this allowlist keeps. Area B should treat `registry-text/v1` as the tokenizer-agnostic replacement for the allowlisted part of `text_clean_v1`, not as an addition: serving both double-counts about 10.7k documents. `text_clean_v1` keeps its value as the view that already has Q38 + Strata top-64 caches.
- Licences: FineWeb-Edu ODC-By-1.0; ARC-AGI-2 Apache-2.0; OpenR1-Math Apache-2.0 and DeepMath MIT (solutions written by DeepSeek-R1, a large public model, not a small one); DAPO-Math Apache-2.0; the v9 web text has no source id recorded in the registry and falls under the 10-14 licence ruling.

### 1.3 `registry-media/v2` (built now)

Source: the same canonical rows (`rows.jsonl`, sha256 `cf64d6f2…`), rows with media sidecars. Media are copied off the NAS (read-only; Jamendo from `/home`) into a content-addressed store `media/<sha2>/<sha256>.<ext>`; every copy is hashed and checked against the row's recorded sha where one exists.

- Kept: 1,397 rows, 2,401 unique audio files, 6,717,661,121 B.
  - **SonicMaster** (`amaai-lab/SonicMasterDataset`, CC BY 2.0 recorded per row), 1,242 rows: `audio_restore_pair` 1,165 (degraded input + clean ground truth, 30 s 44.1 kHz stereo FLAC), `audio_degraded_clip` 71 (degraded input only: understanding input, never a generation target), `audio_clip` 6. No text target: the registry instruction is a Codex template and is not copied.
  - **MTG-Jamendo** (v9 music rows), 155 rows, task `music_tags`: real CC-licensed tracks (44.1 kHz stereo MP3, per-track CC BY / BY-SA licence URL in the row). Text target = the public metadata (artist, album, title, duration, human tags). Only 53 of 155 have any genre / instrument / mood tag, so most targets are artist/title only; the audio is the main value.
- Refused (rows): agent-written / procedural Codex batches 8,187 (`goodall_*` visual-document and image-OCR-video batches, Operator renders); every OCR row is among the refused (only SonicMaster and Jamendo rows are kept), so no OCR material is staged; no public source recorded 1,931; covered by run 0's `librispeech-train` 1,763 (the registry's LibriSpeech rows are train-clean-100, which area B B-2 already holds whole); model-generated text in a public set 586 (the classifier's list: Orpheus TTS, JamendoMaxCaps, BLIP3o, olmOCR, …; 554 of them in `v36_addenda_gated_4964`); hold-out v3 382; row guard (`codex`) 110; other 3. Per dataset id: `refused_by_dataset.json`.
- **v1 was withdrawn before anyone used it.** Sidecars without a recorded sha (Jamendo) were copied before their sha was known, so 21 hold-out files and 1 unreferenced sibling sat in v1's `media/` with no row pointing at them (2,423 stored for 2,402 referenced). Fixed in `07b69bb`: unreferenced store files are pruned before the manifest is written and counted (`counts.pruned_from_store`: holdout 21, unreferenced 1); a row needs every sidecar copied ok. v1 was never registered; its manifest, receipt and a `v1-WITHDRAWN.txt` note are in `registry-media/logs/`, and the v1 directory was deleted after v2 verified. **v2 is the product.** A check over v2: 2,401 store files, 2,401 referenced, 0 hold-out hits, 0 unreferenced.
- Samples read: 3 Jamendo + 3 SonicMaster rows. Every file's sha256 matches its row, and ffprobe decodes each (MP3 44.1 kHz stereo 252-528 s; FLAC 44.1 kHz stereo 30 s).
- Not staged here: the registry's sound (444) and OCR (2,305) rows. All of them are refused as procedural or agent-written, so this product does not close the sound gap; the audio-gap downloads do.

## 2. Items with no new view, and why

| Scout item | Decision | Evidence |
|---|---|---|
| 1. Canonical corpus v1 | **registry-text/v1 + registry-media/v2** | §1.2, §1.3 |
| 2. Curated high-value set (omnicoder_2026 `MASTER_SHARD_INDEX`, 94,318 rows) | Covered by the two registry products | It is the same registry as canonical v1 (91,984 records after dedupe; scout §1 row 2) |
| 2a/2b. `corpus/text`, `text_clean_v1` | Already in run 0 | text_kd bucket; area B rev 2 applies the same allowlist to it |
| 3. K3 rows | Already in run 0 | k3 bucket |
| 4-5. `toolcall_flash_v1`, `bfcl_native_v1` | Already in run 0 | tool bucket |
| 6. Intake growth since 10-22 | Area B's tool did it | `run0_intake_sync.py sync` at 08:13Z: `$L/run0/data/parts/intake-*@20261028T081312Z` (image t2i/edit, music, video, video-understand, Strata text 2,775 records) |
| 7. Cyril's speech | Already in run 0, all of it | `speech_plan.json`: 192 rows = 48 lines × 4 Qwen3-TTS voices; part `cyril-speech`: 382 records (191 ASR + 191 TTS), hold-out 0 hits, 1 failure. Transcripts are paired: each ASR/TTS record carries its line |
| 7b. Cyril's microphone recordings | Nothing exists | both `speech/own-*` drop zones empty |
| 8. NAS PDF corpus | **pdf-corpus/v1** | §1.1 |
| 9. Memory tool calls | Not built: plan §2.5 excludes "the harvested memory `tool_call` events (674k Codex, 206k Claude) … wholesale" | The compliant form (Cyril's prompts + real tool results as context, Strata rewrites the assistant turns) still needs Cyril to confirm it is what "[Cyril quote withheld]" meant, and Strata background admission (blocked by the 40 GiB floor) |
| 10-13. Q38 tool calls, selfgen, Codex sets, `d2_refill`, `vision_in` answers | Excluded | small-model / Codex output |
| 15. Public text on the Lexar (`fineweb_edu.txt`, `wikitext2.txt`, enwiki) | Not built | not "ours"; `synth.txt` has unknown authorship |
| Canonical corpus v2 seal | Not done | a `chimera_corpus` write owned by the intake system; run 0 pins the intake by its `@<tag>` parts instead |

## 3. What area B needs to do to use these

- Read the documents (`chimera_omni.corpus.document.v1`) as CE-only text sources with a share cap; read `records-*.jsonl` (`chimera_omni.run0.record.v1`) through the existing record path; read `registry-media` rows (`chimera_omni.run0.registry_media_row.v1`) with a new builder; `media[].store` is relative to the version directory (`media/<sha2>/<sha256>.<ext>`).
- Swap, do not add: `registry-text/v1` documents replace the allowlisted part of `text_clean_v1` only once a token view and caches exist for them.
- Pin by manifest sha256 (in omni_lab `dataset_version.src_sha256`) and check `verify_receipt.json`.
- Product, batch and source names avoid the row-guard words (`claude`, `codex`, `capture`, `refill`, `selfgen`, `sage`).

## 4. Provenance caveats

- **`pdf-corpus/v1` records the code sha read at finish time, not at start.** (`registry-text/v1` started after the swap; its recorded `7b2d1495…` is the file it ran, still on disk in `$OC/run0/code-corpus-views`.) The first attempt replaced `run0_corpus_views.py` on the server while the PDF build was running, so the PDF manifest's `run0_corpus_views.py` sha (`6b556854…`) names the file on disk at 07:52Z, not the one the build imported at 07:20Z. Checked empirically: 14 randomly sampled v1 documents re-extracted with the deployed `786bbc7` code give byte-identical text and identical token counts (14/14, `scratch/reextract.py`). The data-contract files are exact: `run0_data.py` `be79a767…` and `run0_samples.py` `a4517664…` are the 96ed2b9 blobs (CRLF checkout). `e1b0ab8` fixes this: `code_sha()` is now taken once at start. The finished versions were not edited; the caveat is stored on the omni_lab versions (`--note`).
- The v9 web-text rows have no source id in the registry (licence tier blank or attribution).
- OpenR1-Math and DeepMath solutions are DeepSeek-R1 outputs (public, large model). They are kept as public-set text; drop them in area B if the lead reads "small-model generations" more widely.

## 5. RAM

| Product | Build-time RAM | Note |
|---|---|---|
| pdf-corpus | about 0.64 GB RSS per process (measured, `/usr/bin/time`, smoke run); 6 workers about 4 GB | done; MemAvailable stayed 25-30 GiB during the build |
| registry-text | under 1 GB (hold-out index + tokenizer; single process) | done |
| registry-media | about 0.33 GB RSS, single process (measured, `ps`, v1 and v2 builds) | done |
| verify / register | under 100 MB | streaming hashes |

Nothing runs now. At train time the cost is area B's.

Disk: the three products use about 8.2 GB on the Lexar (exFAT 1 MB clusters: pdf-corpus 0.47 GB, registry-text 0.04 GB, registry-media 7.7 GB on disk for 6.72 GB of files). Lexar free after: 219 GB. Nothing on `/` except the two code dirs (under 1 MB each).

## 6. Verification run

- Tests: `tests/test_run0_corpus_views.py`, 29 passed (the new store-prune test fails on the pre-fix code) (locally on Python 3.14 and on the server venv `venv-pascal-cu118-torch260`, Python 3.12).
- `run0_corpus_views.py verify --manifest $ST/<product>/v<N>/manifest.json` for all three: ok, 0 bad, 0 unlisted (pdf-corpus v1: 3 files, 470,791,206 B; registry-text v1: 3 files, 36,488,757 B; registry-media v2: 2,403 files, 6,717,661,121 B).
- Samples read: 4 PDF documents, 4 registry-text documents, 3 records, 3 Jamendo + 3 SonicMaster rows (sha256 + ffprobe of every file), 14 PDFs re-extracted.
- Rev-2 grammar: `check_records` from `code-b-2d4c4d1` on all 70 registry-text records.
- omni_lab, read back with read-only SQL (`dataset` ⨝ `dataset_version` ⨝ `run_dataset`):

  | dataset | version | items | bytes | sealed | build run |
  |---|---|---|---|---|---|
  | 560 `corpus/run0-views/pdf-corpus` | 569 v1 | 5,839 | 470,791,206 | true | 40332 output pass |
  | 561 `corpus/run0-views/registry-text` | 570 v1 | 10,775 | 36,488,757 | true | 40333 output pass |
  | 562 `corpus/run0-views/registry-media` | 571 v2 | 1,397 | 6,717,661,121 | true | 40335 output pass |

## 7. Not verified / left out

- Per-paper arXiv licences were not fetched (each document has its URL).
- The 13-gram hold-out screen is conservative. It drops some rows that only share Codex template phrases with the eval banks. That loses yield; it does not leak hold-outs.
- No token view or teacher cache was built for any product (no GPU; P40s belong to run 0).
- No Relay post: this work touched no Strata or media-gate service. Another lane was restarting strata-llama during it (a `/health` wait loop on `:18401`; MemAvailable swung 75-165 GiB); the builds stayed at nice 19 / idle IO with a 16 GiB MemAvailable gate.
- This report is not committed: the whole `run0-20261022` report directory is untracked in neural-forge and that checkout is on a Codex branch (`codex/lrc-blank-compression`). The code is committed and pushed (chimera-omni).
