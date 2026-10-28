# Chimera recovery corpus preparation, 2026-10-13

The complete 91,984-row staging file was scanned and normalized. The prepared corpus contains 24,724 rows that pass the structural, provenance, contamination, tool-schema, and supervision checks. This is corpus preparation evidence; answers have not been semantically graded and this does not establish model intelligence or successful training.

Source implementation commit: `78024056045e3fbdb65989c8a3786736d4e26e33`. The implementation is `experiments/chimera_recovery/corpus.py`; 14 focused tests pass. No model, GPU workload, tool call, download, or SSH operation was run by this preparation code.

| Prepared file | Rows | Purpose |
|---|---:|---|
| `train.jsonl` | 19,669 | Recovery training only |
| `validation.jsonl` | 2,506 | New source-group-isolated validation holdout |
| `test.jsonl` | 2,549 | New source-group-isolated test holdout |
| `trainable.jsonl` | 24,724 | All three splits, with split fields retained |
| `normalized.jsonl` | 91,984 | All source rows including quarantined examples |
| `blocked_media.jsonl` | 17,740 | Seven metadata streams; overlapping media entries, not unique samples |

Use `train.jsonl` or explicitly filter `split=train`. `trainable.jsonl` contains holdouts. Canonical rows use `recovery_corpus.v1`, boolean assistant-only message masks, and `provenance.recovery_group_id` as the connected source-event/media/content group. `provenance.split` equals the top-level split.

| Canonical primary route | Prepared rows across all splits |
|---|---:|
| Text instruction | 14,713 |
| Text reasoning | 5,991 |
| Text code | 3,361 |
| Text arithmetic, classified from explicit source calculation task types | 31 |
| Tool call | 373 |
| Tool parallel | 255 |

The 628 tool examples are BFCL examples with authentic schema JSON retained under the explicit `Available tool schemas:` label and acceptable ground-truth alternatives retained in response JSON. The normalizer chooses only schema-valid values present in those alternatives; recorded optional empty alternatives become omitted optional parameters. It preserves the original schemas, alternatives, and selections in `provenance.bfcl_schema_target_adaptation`. BFCL type spellings are transparently adapted to JSON Schema. Missing transport call IDs are derived from explicit message/call positions and labeled as derived; existing IDs are retained. No execution outcome is invented. These rows teach selecting and formatting calls; they do not provide successful tool execution, retries, or multistep outcome supervision. Tool results, retries, tool multiturn, and text multiturn still need genuine additional examples. Unsupported schema variants are quarantined.

All source rows are bound to staging SHA-256 `cf64d6f25b3b59dee2de8dacfbad65109ecce3bc7d293dc68137e0aeba25d1bd`, the allocation's exact 35,985-row text allowlist, and canonical carve seal `7781ff6c47b89835b55cd161d8ff1d8fb313eb8699373dadb96f9759a60a0a14`. Only original carve assignments of `train` can enter prepared output. The original frozen evaluation file remains separate: 1,948 rows, SHA-256 `626093ca6024424f0ddc1ab397376dd5822d10bafb9818fd3a366777e743c4e2`, matching its source manifest. An additional normalized exact-task and greater-than-50% shared 13-gram screen removes task near duplicates; shared tool catalogs are excluded from task text. Existing split conflicts within connected components are blocked.

Independent output revalidation found zero invalid rows, zero duplicate IDs, and zero source-ID or recovery-group intersections between train, validation, and test. `verification.json` records every output hash and sample rows. The stable training-file SHA-256 is `b468d931353854a3d922277a8d46271a378ccd8be73e2c5722cc6da5783c579b`; the all-splits file SHA-256 is `2d33b44581be480aa920c0a116bf79d6088cc0938dbb9128de6a30bdfda97a57`.

Quarantine counts are overlapping reasons, not a partition. In the full staging scan, 55,999 rows are outside the pinned text allocation, 19,741 carry frozen `eval_near_duplicate` assignments, and 5,744 expose `CURRENT_TARGET` answers inside teacher-rewrite prompts. Such rewrite tasks do not demonstrate learning to solve the original problem from its inputs. They are excluded from this recovery answer corpus. Exact source dates are collection-receipt dates; original authorship dates are unknown and are not inferred to be 2026.

| Metadata stream | Actual entries | Remaining preparation issue |
|---|---:|---|
| ACE | 1,024 | Real audio target/teacher cache and checkpoint provenance |
| QIMG | 2,048 | Real image target/teacher cache and checkpoint provenance |
| H3 | 512 | Audio explicitly comes from a deterministic unrelated assignment; synchronized same-origin AV must replace it |
| HY3D | 256 | Image/icon targets are 2D proxies; genuine geometry or volume supervision is absent from these items |
| ViT | 7,492 | Image metadata alone has no text response target; authentic teacher outputs are needed |
| Vision input | 5,896 | Linked image/OCR targets and teacher caches; 18 explicit generation edit pairs retain source-input/target roles |
| Video input | 512 | Linked video target responses and teacher caches |

All seven metadata files match the hashes in their original campaign manifests. Their media paths remain linked to actual remote artifact locations. The local metadata mirror lacks the underlying media payload files, so local verification cannot mark them usable; this does not assert that those files are absent on the server. Metadata does not substitute for a teacher tensor/cache. Blocked media is separate from text/tool training output.

The preparation API `prepare_av_pair` accepts already-extracted real audio/video targets with identical `source_media_sha256` and `start_seconds`. Validation requires matching source origin and timestamps, synchronized alignment metadata, actual file/hash checks, and teacher provenance. It rejects the original unrelated pairings and clean-target conditioning leakage. The caller must extract both streams from a real shared source interval and materialize the correct H3 teacher/cache representation before use.

`manifest.json` records sources, receipts, all 20 route coverage counters, per-row blocker totals, original shard paths needing richer tool structures, manifest inventory, and output hashes. `diagnostics.jsonl` retains a per-row explanation. No thermal gate is part of this corpus preparation.
