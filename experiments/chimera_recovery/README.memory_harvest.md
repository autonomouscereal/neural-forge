# Memory corpus harvest

`memory_harvest.py` reads the installed PostgreSQL hook audit trail through raw
asyncpg. It does not read local harness transcripts or execute historical tools.
The vault password remains inside the process. Database errors are reported by
exception type only.

## Bounded collection

Inspect the recorded structure before choosing a tranche:

```powershell
python experiments/chimera_recovery/memory_harvest.py inspect --limit 4000 --since 2026-09-01T00:00:00+00:00 --until 2026-10-01T00:00:00+00:00 --event-types user_prompt,session_stop,assistant_message
```

Inspection reports counts and field shapes, never message bodies or identity
values. `--agent` and `--space` accept exact recorded values; `--before-id` permits
repeatable older windows. All date ranges must remain within 2025–2026.

The query selects a bounded latest-ID range before joining matching archived
payloads. It runs in a repeatable-read, read-only transaction with a 15-second
statement timeout and a 45-second operation deadline. Connections are terminated
on exit. Tranches have at most 10,000 events, 32 MiB of compressed payloads,
32 MiB of archived character counts, and 1,000 exported candidates. Each archive
also has independent size limits and must match its event ID, event UID,
decompressed SHA-256, codec, and UTF-8 character count.

```powershell
python experiments/chimera_recovery/memory_harvest.py harvest --limit 4000 --max-rows 500 --since 2026-09-01T00:00:00+00:00 --until 2026-10-01T00:00:00+00:00 --event-types user_prompt,session_stop,assistant_message --exclude-prompts frozen_prompts.json --out new_tranche
```

The output directory must be new. `candidates.jsonl` contains redacted records;
`ready.jsonl` contains only individually reviewed records that pass every gate.
`report.json` contains counts, query bounds, limitations, and artifact hashes.
Do not commit corpus bodies or credentials.

## Record contract and gates

Each `recovery_corpus.v1` row preserves `messages`, `tool_calls`, real call IDs,
`tool_call_id`, tool schemas, and structured outcome flags. Routes are
`tool_calling` or `text_reasoning`; memory harvesting does not fabricate media.
`supervision.message_mask` and `supervised_assistant_messages` select assistant
messages only. System, developer, user, and tool-result tokens remain context.

Single-hook calls/results are paired inside the same verified archive. An
exact session/turn match can link a prompt, but missing prior context is never
reconstructed by adjacency. Recorded message arrays retain retry decisions,
parallel call/result ordering, and error flags. Parallel success/failure does
not count as a later repair. Success requires a recorded code or success flag;
opaque prose stays unknown.

`quality.format_ready` covers complete recorded message structure, schemas,
assistant targets, and omission/redaction gates. `quality.train_ready` also
requires individual content approval. Unreviewed raw memories stay candidates,
even when their format is valid. An insertion timestamp establishes ingestion
time; `provenance.source_date_basis` discloses when original recording time is
unavailable. `source_verified` means event/archive relation and byte integrity,
not proof of original full context, semantic correctness, or model performance.

Captured tool schemas are validated against actual arguments. A missing schema
is never inferred from an argument example. Optional `--tool-catalog` requires:

```json
{"schema":"recovery_tool_catalog.v1","provenance":{"authority":"operator_reviewed","source":"reviewed schema source","source_sha256":"<64 hex characters>"},"tools":[]}
```

This catalog records an explicit operator attestation and file hash. It does
not authenticate itself merely by claiming a source. External schema references
are refused, and absent JSON Schema validation leaves the schema gate closed.

## Individual review

Review well-posed examples and their recorded results before approving them.
`--review-manifest` accepts `recovery_memory_review.v1` with a reviewer, a
timezone-aware 2025–2026 `reviewed_at`, and individual decisions:

```json
{"schema":"recovery_memory_review.v1","reviewer":"reviewer identity","reviewed_at":"2026-10-13T00:00:00Z","rows":[{"id":"memory-...","row_sha256":"<review_digest(row)>","source_ids":[],"archive_payload_sha256s":[],"decision":"approve","reason":"specific review evidence","standalone_prompt_context":false}]}
```

The row digest binds the actual messages, schemas, media, and supervision mask;
source IDs and archive hashes must also match. A review can explicitly approve
the suitability of a linked prompt/answer as standalone training context. That
does not claim that the original full context was captured. Rejected records
and unreviewed records remain candidates. Existing redacted assistant targets,
missing schemas, and missing prompt linkage cannot be waived by this review.

## Leakage isolation

All rows from the same agent/session share one group. Exact duplicates, repeated
nontrivial user prompts, and near duplicates form connected components across
sessions. Components share one deterministic train/validation/test split;
exact duplicate targets have only one ready representative. Frozen evaluation
prompts are excluded after normalization, by near overlap, and when a longer
prompt contains a nontrivial frozen question. The exclusion-file SHA-256 is
recorded. Provide the actual frozen bank rather than an invented substitute.

Run the offline integrity and supervision suite:

```powershell
python -m unittest discover -s tests -p test_chimera_memory_harvest.py -v
```
