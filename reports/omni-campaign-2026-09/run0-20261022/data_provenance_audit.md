# Run 0 data provenance audit (B-0)

Status: 2026-10-28. This is plan item B-0 of `RUN0_BUILD_PLAN.md` revision 2. The raw reports are on the AI server under `/home/cereal/omni-campaign/run0/data/`:
- `logs/provenance_audit_20261028.json` (`run0_provenance_audit.py`, 12:24Z);
- `author_allow_v1.json` (the author audit, `run0_build_manifest.py author-audit`, 08:27Z);
- `logs/bfcl_holdout_rows.json` and `logs/bfcl_holdout_probe.json`.

The code is chimera-omni branch `claude/run0-b-20261022`. The data it feeds is described in `data_manifest.md`.

## Acceptance

- **Every manifest source has `authorship_checked: true` or is conditioning-only.** `run0_data.validate_manifest` refuses any other source, at seal and at every trainer load.
- **Excluded turns are counted per source.** Each record source's `source.json` holds a `refused` map. The corpus sources hold the author tallies below.

## 1. Authorship by field (text_clean_v1, K3, canonical rows)

The author of every canonical row that the cached corpora reference is read from its author field in `distill/corpus/staging/rows.jsonl`, never from a list of known rows. Only these writers are allowed: FineWeb-Edu, v9-web-text, OpenR1-Math, DeepMath, DAPO-Math and ARC-AGI-2. Everything else is a non-target: unknown writers, Qwen teacher rewrites (a small model), agent traces, memory batches and Nemotron.

| Corpus | Unit | Allowed | Total | Share |
|---|---|---|---|---|
| `text_clean_v1` | tokens (documents kept) | 5,398,520 | 12,349,164 | 44 % |
| K3 `pass-0..3` | row memberships | 1,511 | 4,135 | 37 % |

Largest authors in `text_clean_v1`, by tokens:

| Author | Tokens | Verdict |
|---|---|---|
| v9-web-text | 3,327,283 | allowed |
| unknown | 3,035,824 | denied |
| small-model: qwen-teacher-rewrite | 1,577,674 | denied (small model) |
| arc_agi_2_local_training_split_2026 | 925,463 | allowed |
| FineWeb-Edu | 831,366 | allowed |
| agent-trace | 595,719 | denied |
| memory_batch_* | 290,994 | denied |
| nvidia/Nemotron-Terminal-Corpus | 264,160 | denied |
| memory_harness | 67,563 | denied |
| codex | 44,902 | denied |

Largest authors in K3, by row memberships: v9-web-text 1,119; unknown 1,057 (denied); memory_batch_ 500 (denied); qwen-teacher-rewrite 296 (denied); FineWeb-Edu 277; agent-trace 136 (denied); codex 35 (denied).

- **Effect in training.** A denied document is masked span by span: its tokens are never a CE or KD target. A packed row that is at least half denied is skipped.
  - `text_clean_v1`: 1,950 of 3,015 rows are skipped, and 2,865 carry masked spans.
  - K3: 166 of 487 packed rows are usable.
- **The `text` registry part (20M positions)** is not a run 0 source. Registry text reaches run 0 only through the staging products `registry-text-docs` and `registry-text-chat`, and every turn of those is checked by its `author` field.
- **Harvested memory `tool_call` events** (674k Codex, 206k Claude) are excluded wholesale.

## 2. Prompt authorship per intake batch

An intake image, music or video item is trained only when its prompt is Strata-written (`prompt_source.kind = strata`). Everything else is refused. The refusal counts are per export; both exports refuse the same reasons.

| Source | Records | Intake items |
|---|---|---|
| intake-image-t2i | 247 | 247 |
| intake-image-edit | 32 | 32 |
| intake-music-ace | 568 | 105 |
| intake-video | 33 | 26 |
| intake-strata-text | 479 | 479 |
| intake-image-t2i@20261028T081312Z | 2,299 | 2,299 |
| intake-image-edit@20261028T081312Z | 630 | 630 |
| intake-music-ace@20261028T081312Z | 5,340 | 991 |
| intake-video@20261028T081312Z | 229 | 181 |
| intake-video-understand@20261028T081312Z | 229 | 143 |
| intake-strata-text@20261028T081312Z | 2,775 | 2,775 |

Refusals in the 10-28 export:
- mesh items, routed to the hy3d voxel path instead: 740;
- edit source is a held item: 70;
- edit source missing or held out: 25;
- H3 SOL video routed elsewhere: 38;
- non-final H3 SOL stages: 55.

Text intake refusals (capture spool or a non-generator batch): 1,038 in the 10-22 export, 3,580 in the 10-28 export.

Meshes are checked the same way. `hy3d-voxel64` and `hy3d-voxel64@v3` use a volume only when its mesh's prompt is Strata-written and misses the hold-out registry. Otherwise the whole volume is refused, and the source.json counts the "no Strata prompt" refusals.

## 3. The raw Strata/Sprite capture spool

It is excluded by name at path, row and batch level. There are 242 `strata-capture-*` batches, from `strata-capture-20261022-001-text` to `strata-capture-20261028-008-text`. None of them is used.

## 4. WIT load check

WIT does **not** load. `/mnt/lexar/omnicoder_0.1.0/data/wit` (12 GB) holds TSV metadata only: page and image URLs, captions and context. There are no image files, and one shard is an incomplete aria2 download. Using WIT needs a fetch pipeline, so it is not a run 0 source. Image understanding is COCO 2017, whose 10-day supply holds.

## 5. BFCL-62 held-out rows in `bfcl_native_v1`

- A whole-row screen against hold-out registry v3 hits all 500 rows, because the task template and the shared tool schemas recur in the held-out banks.
- Comparing the question alone gives 13 exact matches (rows 52, 70, 76, 87, 89, 311, 355, 401, 423, 425, 436, 468, 499). All 13 come from the `codex_recovery_heldout` bank.
- 2 more rows (138, 271) share a 13-gram.
- No row comes from the `bfcl62` bank itself.

The 15 rows are refused through `deny_rows`, so 485 train. BFCL tool scores on run 0 are not held-out evidence beyond the BFCL-62 split. Cyril ruled on 10-17 that "tool-bfcl-native stays as data".

## 6. gen-worker-text job types and `n_probs`

The Strata-written data (D3) comes from the support workflow's `run0-strata-writer.service` (branch `claude/run0-gen-20261028`), not from a B-5 script in this branch. It covers the intent bank, edit pairs and `n_probs` KD.

Its Q4_K_XL KD corpus (`strata-kd-gen`, no `lse`) is **not** wired into run 0. That needs two things first: a teacher-identity ruling, and the C-15 check of `n_probs` against the cached Strata teacher (which is the RVN IQ3_S cache).

## 7. Stale IQ3_S labels

134 of the 142 intake batches dated on or after the 10-27 switch still name RVN IQ3_S in `_meta.json`. Sealed batches are never edited. The manifest's Strata teacher label states what the caches are ("the RVN IQ3_S cache, NOT the live Q4_K_XL"), and new Strata KD carries `strata-q4kxl-abl-v2`.

## 8. The top-10 alarm's held-out text slice (added 10-28, manifest v3)

`holdout_text` holds 16 `text_clean_v1` rows. Each is a guard-served row covered by both teacher caches. They are picked round-robin by `sha256(source:row)`, so every seal of the same corpora and allowlist picks the same rows. They are in the trained shards' `deny_rows`, and `validate_manifest` refuses any manifest in which a trained source could serve one of them.
