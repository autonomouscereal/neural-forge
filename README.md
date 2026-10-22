# Omni Lab

One PostgreSQL database (`omni_lab`) for every test, eval, fold, distill, cache and probe result of the
AI-model program (Chimera Omni, Neural Forge, K3, speech, media, corpus), plus the code that writes to it.
Built 2026-10-21 from the Omni Lab design and its critique (cases are global, every run records its software environment (`env`), media outputs go in `eval_artifact`, the tensor census in `model_tensor_group`, and there is a separate read-only role).

It has four parts: the database core (roles, migrations, `omni_lab_db.py`), the ingesters that backfill
every existing result format, the runner (`omni_lab.py run` / `compare` / `list` / `add-suite` / `add-cases` /
`grade` / `log-run`, in `omni_lab_run.py`) and its GPU worker (`omni_lab_worker.py` + `adapters/` + `graders/`).

## Where it lives

| What | Where |
|---|---|
| Database | `omni_lab` on `agent-memory-postgres`, `192.168.50.222:25490` (PostgreSQL 16.13) |
| Write role | `omni_lab` (owner of the database, CREATEDB for throwaway test databases). Vault key `omni_lab_pg` |
| Read-only role | `omni_lab_ro` (SELECT on every table and view). Vault key `omni_lab_ro_pg`. Point DBeaver/psql here |
| Code (workstation) | `C:/Users/cereal/Documents/ai training/omni-lab/` (local git, never pushed) |
| Code (AI server) | `/home/cereal/omni-campaign/lab/` (same bytes; check with `sha256sum`) |
| Server runtime env | `/home/cereal/.config/omni-lab/env`, mode 600. Generated from the vault, same pattern as `/home/cereal/.config/chimera-intake/env` |

## Connecting (no secrets in this repo)

`omni_lab_db.py` resolves each profile from `OMNI_LAB_PG_*` (rw), `OMNI_LAB_RO_PG_*` (ro) and
`OMNI_LAB_ADMIN_PG_*` (bootstrap only). Each profile takes `HOST`, `PORT`, `DB`, `USER` and one of
`PASSWORD`, `PASSWORD_FILE` or `PASSWORD_COMMAND`.

- **Workstation:** copy `.env.example` to `.env` (gitignored). It holds only credential-helper commands
  (`{python} .../credman.py get omni_lab_pg`).
- **AI server:** `set -a; . /home/cereal/.config/omni-lab/env; set +a`, then use
  `/home/cereal/.venvs/chimera-intake/bin/python` (it has psycopg2 2.9.13). Do not install psycopg2
  into `model-surgery-lab/venv`. The GPU worker never imports `omni_lab_db`.
- **psql / DBeaver:** host `192.168.50.222`, port `25490`, database `omni_lab`, user `omni_lab_ro`,
  password from the vault (`credman.py get omni_lab_ro_pg`).

## Commands

```
C:/Python314/python.exe omni_lab.py db check              # schema version, migration sha parity, row counts, last ingest per source
C:/Python314/python.exe omni_lab.py db migrate [--dry-run]
C:/Python314/python.exe omni_lab.py last [N]              # newest N rows of v_results
C:/Python314/python.exe omni_lab.py sql "SELECT * FROM v_results"   # runs as omni_lab_ro
C:/Python314/python.exe omni_lab.py compare 30658 30666 30676       # runs x cases pivot (see "Running a test")
$PY omni_lab.py run --model M --suite S --gpus 6,4,0 | --cpu       # AI server: evaluate -> console + database
C:/Python314/python.exe -m unittest discover -s tests -v  # throwaway omni_lab_test_<hex> DBs, created and dropped
```

## Schema at a glance

Migrations are forward-only and numbered (`migrations/NNNN_name.sql`). `schema_migration` records
the version and sha256 of each applied file. `migrate` refuses to run if an applied file has been edited.

- **Core (0001):**
  - Provenance: `ingest_batch`.
  - Hardware and software: `project`, `host`, `gpu` (keyed by UUID), `env` (software stack, deduplicated by hash), `tool`, `tool_version`.
  - Models: `model`, `model_lineage`.
  - Runs: `run` (with `config_sha256`), `run_device` (GPU, kernel path, full `kernels` dict, layer span), `run_model`, `metric` (curves at step>0, summaries at step 0 with a `denominator`), `run_event`.
  - Tests and evals: `test_suite`, `test_case` (global, keyed by `(case_id, content_sha256)`), `suite_member`, `grader`, `test` (a named, re-runnable test definition), `eval_run`, `eval_result`, `eval_grade`, `eval_artifact` (image/audio/video/mesh outputs).
  - Experiments: `technique`, `experiment` (fold/arch/distill/data/kernel), `experiment_eval`.
- **Registry (0002):** `lane`, `ledger_event` (mirror of `$OC/ledger.jsonl`; the ledger stays the authority), `dataset`, `dataset_version`, `run_dataset`.
- **Knowledge (0003):** `directive` (Cyril's words, verbatim), `ruling`, `document`, `model_tensor_group` (tensor census with `modality_scope`), `arch_requirement`, `arch_check`.
- **Views (0004):**
  - `v_results`: one row per eval run, laid out as "test | configs | results".
  - `v_eval_matrix`: per case, long form.
  - `v_effective_grade`: a human grade wins over a machine grade.
  - `v_run_summary`, `v_latest_models`.
  - `v_model_modality_coverage`: which modalities a model actually handled, measured from results.
  - `v_arch_conformance`, `v_directives_by_topic`.
- **Grants (0005):** SELECT for `omni_lab_ro`, including tables created later.
- **Folding record (0007):** `fold_arm` (one row per fold arm, 1:1 with `experiment`), `fold_component`, `fold_behaviour`, `fold_diagnostic`, `fold_source`, `experiment_alias`, `experiment_link`, `fold_finding` (+ `_experiment`, `_source`), `fold_shard`, `fold_coverage`; technique gains `code_location`, `first_date`, `status_text`, `detail`; experiment `evidence_level` also allows `notes`, `transcript`, `commit_msg`. Views `v_fold_arms`, `v_technique_scoreboard`, `v_findings`, `v_fold_coverage`. See "Neural Forge folding record".

A blank cell is a missing row. Add cases by creating a new suite version that links the old `test_case` rows plus the new ones. Older runs then show no result for the new cases. An edited prompt under the same `case_id` becomes a separate `test_case` row, so old and new answers are never compared as if they answered the same question.

## How the tables join

Everything hangs off `run`. A run is one fold, distill, cache, gen, eval, probe, build, export, review or smoke job.

```
project ─< lane ─< run >─ model ─< model_lineage >─ model          (lineage: fold/clone/distill/merge/export/source)
                    │  └─ tool_version >─ tool                     (which pinned script, by sha256)
                    │  └─ env                                      (torch/fla/transformers/driver stack)
                    ├─< run_device >─ gpu >─ host                  (cards by UUID, kernel path, kernels dict, layer span)
                    ├─< run_dataset >─ dataset_version >─ dataset  (corpora, teacher caches, banks)
                    ├─< run_model >─ model                         (teacher / baseline / output / base)
                    ├─< metric                                     (curves at step>0, summaries at step 0)
                    ├─< run_event                                  (parsed *.status lines)
                    └─1 eval_run ─< eval_result >─ test_case >─< suite_member >─ test_suite
                                       ├─< eval_grade >─ grader    (machine or human; human wins in v_effective_grade)
                                       └─< eval_artifact           (decoded wav/png/glb/latents)
model ─< ledger_event                                              (mirror of $OC/ledger.jsonl, one row per line)
technique ─< experiment ─< experiment_eval >─ eval_run             (fold / arch / distill experiments and their evals)
directive, ruling, document, arch_requirement ─< arch_check >─ model   (Cyril's words, decisions, docs, conformance)
every domain row: src_path, src_sha256, src_host, ingest_batch_id ─> ingest_batch   (provenance)
```

Questions and where to look:

| Question | Start here |
|---|---|
| "What did test X give on model Y with which GPUs and kernels?" | `v_results` (one line per eval run: test, configs, results) or `omni_lab.py last` |
| "Compare runs case by case" | `omni_lab.py compare <run ids>` or `--suite S --last N` |
| "What did training job Z do, on which cards, with what loss?" | `run` + `run_device` + `metric` (`v_run_summary`) |
| "What is this model, where is it, and what is it made of?" | `model`, `model_lineage`, `ledger_event`, `v_latest_models` |
| "Which fold techniques were tried, and how did they turn out?" | `technique`, `experiment`, `experiment_eval` |
| "What did Cyril say about X?" | `directive` (verbatim, dated, with the source event id), `v_directives_by_topic` |
| "Which modalities does a model really handle?" | `v_model_modality_coverage` (measured from results, not from config) |

## What is in it today (2026-10-21, from `omni_lab.py db check`)

Schema version 5. All five migrations are `ok` on both hosts. There are 86 ingest batches across 22 sources. Two
batches failed with a lock timeout (`dossiers` #27, `corpora` #65) while ingesters ran in parallel; both sources
were re-run cleanly afterwards.

| Area | Rows |
|---|---|
| Models | `model` 1,671 (1,663 present; 18 with a ledger checkpoint id), `model_lineage` 33, `ledger_event` 42 |
| Runs | `run` 2,367, `metric` 19,354, `run_event` 1,051, `run_device` 244, `run_model` 93, `run_dataset` 94 |
| Run kinds | fold 1,691 · cache 383 · probe 82 · build 74 · distill 42 · smoke 29 · review 26 · eval 24 · gen 10 · export 6 |
| Tests and evals | `test_suite` 16, `test_case` 562, `suite_member` 1,078, `eval_run` 24, `eval_result` 390, `eval_grade` 241, `eval_artifact` 79, `grader` 6, `test` 1 |
| Fold research | `technique` 43, `experiment` 43, `experiment_eval` 12 |
| Data | `dataset` 36, `dataset_version` 109 |
| Hardware and tools | `host` 1, `gpu` 7, `env` 16, `tool` 189, `tool_version` 330, `lane` 11 |
| Knowledge | `directive` 136, `ruling` 34, `document` 1,927 |

The comparison of the S0 parent import, OMNI1 and the full-KD run-1 student on bank_v1
(`compare 30658 30666 30676`) took 0.48 s wall from the workstation.

## Gaps (not done yet)

- **Architecture conformance (seeded 2026-10-21, arch v2).** `arch_requirement` holds R1-R16, each citing `directive` rows (`ingest arch_requirements --inventory inventory/2026-10-21-arch-v2`). `omni_lab.py arch-check --report R.json [--model KEY]` loads a `chimera_omni.conformance` report (chimera-omni branch `claude/arch-v2-20261021`) into `model_tensor_group` + `arch_check` + a `probe` run. Checked so far: the v2 profiles `arch:co2-2b`, `arch:co2-4b`, `arch:co2-proxy` (design-stage rows, layout `chimera.config.v1`, no weights), the legacy `arch:k3-test` toy, and `OC02-F-Q38-27BU-OMNI1` (header-only census: fails R1, R3, R4, R5, R7, R9, R13). Only the proxy has dynamic probes (2B/4B R3/R5 stay `partial`).
- **Most fold runs have no outcome.** 1,323 of the 1,691 fold runs have status `unknown`, because their receipts carry no terminal verdict. Unknown means "no record", not "still running".
- **Only one saved `test`** (`smoke-bank2-cpu`) and one runner-produced eval run (37881, a CPU smoke test on PARENT-Q35-0.8B).
- **Two engines have not run end to end:** `chimera` and `gguf`.
- **No media suites can run through the runner yet.** Media evals exist only as ingested S8 results (`s8_media_smoke`, `s8_qvl_describe`).
- **No cron is installed.** Ingest is run by hand. No existing training job calls `log-run` yet, so new runs appear only after an `ingest`.
- **Not ingested:** Codex sessions (`.codex/sessions`).
- **Folding record not loaded yet.** Migration 0007 is applied (2026-10-22) but the foldrec shards are still being extracted; `fold_arm` and its tables are empty until `ingest foldrec` runs. Until then the Neural Forge history in the database is only `foldhist` (43 summary rows) and `fold_receipts`.
- **GPU mapping is approximate for status-line runs.** GPU indices from status lines are mapped through today's `nvidia-smi` order (`run_device.detail.mapping = 'index-from-status'`).

## Ingesters: models and runs

Run on the AI server (the sources live there), from `/home/cereal/omni-campaign/lab`:

```
/home/cereal/.venvs/chimera-intake/bin/python omni_lab.py ingest models-runs --keep-going   # the whole group, ~5 min
/home/cereal/.venvs/chimera-intake/bin/python omni_lab.py ingest trainer_runs status_runs    # any subset, in that order
```

Each source writes one `ingest_batch` row (committed first, so a failure stays on record) and all of its
rows in one transaction. Every row is keyed on a natural key, so a re-run rewrites the same rows. The
ingesters only read their sources.

| Source (`ingest/<name>.py`) | Reads | Writes |
|---|---|---|
| `hardware` | `nvidia-smi` on the host it runs on | `host`, `gpu` (UUID is the key, index refreshed) |
| `tool_pins` | `$OC/distill/tools/TOOL_PINS.json` | `tool`, `tool_version` (pinned / superseded) |
| `lanes` | `$OC/distill/queues/LANES.json` | `lane` |
| `oc_ledger` | `$OC/ledger.jsonl`, `$OC/registry/**`, OMNI1 `composite_manifest.json` | `model` (key = checkpoint id), `ledger_event`, `model_lineage` (incl. OMNI1's 7 `source` edges), one `run` per sealed record with review columns as `metric` |
| `model_scan` | every weight directory under `$OC`, `$L`, `/home/cereal/models`, `/home/cereal/model-surgery-lab` (safetensors headers only) | `model` (key `path:<sha16 of host:path>` unless the ledger owns the path; census in `config`) |
| `teacher_caches` | `$L/teacher_cache/**/{manifest,pass_receipt.partial,plan}.json` | `dataset`, `dataset_version`, one `cache` run per pass + `run_dataset` |
| `trainer_runs` | full-KD and media-M1 run dirs, M1/M2 self-tests, `d2.status` + its log, D1 run dirs | parent run + one run per segment/attempt, per-step `metric` curves from `STEP {json}` log lines and receipt histories, `run_device`, `env`, student / M1-export `model` rows with `distill` lineage |
| `status_runs` | queue `*.status` files (not the trainer-owned ones) | `run_event` per timestamped line; `run` per paired job (heuristic: see module docstring) |
| `fold_receipts` | `*receipt*.json` under fold / K3 / import / media-fold / LFM2 paths | one `run` per receipt (kind fold, cache, probe or build); content-addressed cache entries and copied trees are excluded, out-of-scope rows are pruned |

Conventions worth knowing when reading the tables:
- GPU placement from status lines is by **index** (`run_device.detail.mapping = 'index-from-status'`), mapped through today's `nvidia-smi` order. Runs whose lines carry GPU UUIDs use them (`uuid-from-status`).
- `run.status = 'unknown'` means the source has no terminal record (for example, an open status line or `completed=false`). It does not mean the process is still alive.
- Times taken from file mtimes are marked in `run.notes` or `config.time_source`.

## Ingesters: test suites and eval results

Server sources (run in `/home/cereal/omni-campaign/lab`), then the Codex task sets on the workstation:

```
/home/cereal/.venvs/chimera-intake/bin/python omni_lab.py ingest evals     # banks dossiers toolcall_bankruns student_tests s8_media p11_reads
C:/Python314/python.exe omni_lab.py ingest codex_tasksets                  # workstation: proof_outputs/chimera_recovery_eval_*
```

| Source | Reads | Writes |
|---|---|---|
| `banks` | `$OC/calibration/banks/*.json`, R-Q `PROMPT_PACK` (by AST) in `import/rq27b/tree_c5e6052/scripts/import_receipt_qwen35.py` | suites `bank_v1@1` (16), `probe@1` (8), `smoke@1` (3), `toolcall_v1@1` (185), `bank_v2_draft@draft-<sha12>` (40, DRAFT), `rq_prompt_pack@c5e6052` (8) |
| `codex_tasksets` | workstation `proof_outputs/chimera_recovery_eval_*/tasks.jsonl` | `codex_recovery_{bfcl,tools_executed,frozen,text,combined}` (62/168/230/44/274 members, 274 distinct cases: combined and frozen link the same rows); oracle in `test_case.reference` |
| `dossiers` | `$OC/composite/{s0,s3,s8,d1}/review/oc-dossier-*.json` + `.finite_guard.json` + `review-*.json` | per dossier up to 3 eval runs (`bank_v1`, `smoke`, `probe`), placement per card with layer span, upper-bound columns as `metric upper_bound/*` |
| `toolcall_bankruns` | `$OC/distill/toolcall/bank-runs/*.grade.json` + responses + serve/export receipts | eval run on the GGUF (engine `gguf`), server cards by UUID, `toolcall_selfgen.grade_case` pass/fail per case |
| `student_tests` | `$L/scratch/student-tests-*/results.json` + `grades.json` | suites `student_tests_<date>@1`, `student_new_<date>@1`; the student eval run; reader grades (`human:claude`) for the student AND for the parent / OMNI1 runs it compared against |
| `s8_media` | `$OC/composite/s8/media_out/{s8_media_forward,s8_media_decode.*}.json`, `qvl/answers.json` | `s8_media_smoke@1.1.0` (13 segments, artifacts = decoded wav/png/glb + latents), `s8_qvl_describe@1` (5) |
| `p11_reads` | `$OC/distill/tps/**/p11_*_summary.json` | `p11_needles@v1-<needles sha12>`, one eval run per read |

How eval rows are keyed and graded:
- The model under test is the ledger checkpoint whose review record names the dossier's `dossier_sha256`, else the ledger directory, else a `path:<sha16>` row (same key `model_scan` uses). So the S3 native dossier lands on `OC02-F-Q38-27BU-C2K4352` and the composed D1 dossiers on `OC02-D-Q38-27BU-OMNI1D1`. The S0 parent dossier ran a Chimera import of the parent; it is its own row (stage `S0`) with an `export` lineage edge to `PARENT-Q38-27BU`.
- A recorded prompt that hashes differently from its bank row becomes its own case and is named on the batch (`prompt-mismatch`); none did on 2026-10-21.
- Grades are only what a file recorded: machine graders (`toolcall_selfgen.grade_case`, `student_test.code_check`, `p11_needle_engine.passkey_exact_teacher_forced`, `s8_media_decode.beats_noisy_input`) and reader grades from `grades.json` (`human:claude`: correct / partially correct / content miss / no content -> correct / partial / miss / none). A `not run` cell writes nothing, so it stays blank.
- Artifact flags come from recorded fields only: `loop_detected` -> `loop`, empty response -> `empty`, cap termination -> `cap-ride`, a non-CONTENT `oc_calib.classify_form` box -> its name.

Parent vs OMNI1 vs student, one line per case (`-` = the model never ran the case; `ungraded` = it ran, nobody graded it):

```
C:/Python314/python.exe omni_lab.py sql "SELECT tc.case_id, max(COALESCE(vm.grade,'ungraded')) FILTER (WHERE er.label='PARENT-Q38-27BU-NATIVE-NOTHINK') parent, max(COALESCE(vm.grade,'ungraded')) FILTER (WHERE er.label='OC02-OMNI1-NATIVE-NOTHINK') omni1, max(COALESCE(vm.grade,'ungraded')) FILTER (WHERE er.engine='qwen35-student') student FROM test_case tc JOIN suite_member sm ON sm.case_pk=tc.id JOIN test_suite ts ON ts.id=sm.suite_id LEFT JOIN v_eval_matrix vm ON vm.case_pk=tc.id LEFT JOIN eval_run er ON er.run_id=vm.eval_run_id WHERE ts.name IN ('bank_v1','student_tests_20261021') GROUP BY tc.case_id ORDER BY 1"
```

## Neural Forge folding record (foldrec, migration 0007)

Added 2026-10-22 because the 43 `foldhist` rows were summaries: months of folding work (May-Oct 2026) had no
per-arm record. Extraction agents read the notes, docs, reports, receipts, commits and memory and write one
shard each to `inventory/2026-10-22-foldrec/<shard>.json` (schema `foldrec-v1`). `ingest/foldrec.py` loads
every shard in that folder.

```
C:/Python314/python.exe omni_lab.py ingest foldrec --dry-run      # parse, validate, merge, resolve read-only; writes nothing
C:/Python314/python.exe omni_lab.py ingest foldrec                # one ingest_batch; re-runnable
C:/Python314/python.exe omni_lab.py sql "SELECT key, program, components, behaviour_score, scorer, verdict_class FROM v_fold_arms ORDER BY date"
C:/Python314/python.exe omni_lab.py sql "SELECT * FROM v_technique_scoreboard WHERE tested_at_4x ORDER BY best_behaviour_at_4x DESC NULLS LAST"
```

Run it on the workstation: the shards live there, and directive quotes are checked against the cited files
and `agent_memory_events`.

| Table / view | One row per |
|---|---|
| `fold_arm` | Fold arm, child or attempt. It is 1:1 with `experiment` (key = the source's own id) and holds program, wave, owner, parent, child, axes, ratios, objective, training, verdict, Cyril's reaction, every raw shard record (`record`) and the shards' disagreements (`conflicts`). |
| `fold_component` / `fold_behaviour` / `fold_diagnostic` / `fold_source` | Component (kind, layers, technique, ratio, rank/k), behaviour score (suite, scorer, `12/15` as written plus parsed numbers), diagnostic (value as written plus `value_num` when it is one number), cited source (path, lines, commit, event, quote). `shards` says which shard gave each row. |
| `experiment_alias` / `experiment_link` | Other id of an arm. Link between experiments: `same_as`, `rerun_of` (`KEY#x` or `KEY@w27` of `KEY`), `summarizes` (a foldhist id that several arms carry), `part_of`. |
| `fold_finding` (+ `_experiment`, `_source`) | Law or conclusion with status (`proven` ... `closed_path`), scope, the arms it rests on and its sources. |
| `fold_shard` / `fold_coverage` | Shard file, and each file it read (lines) or skipped (reason). |
| `v_fold_arms` | Arm on one line: components, techniques, ratios, objective, first behaviour score + scorer, diagnostics, verdict, aliases, links. |
| `v_technique_scoreboard` | Technique x component kind: arms, best behaviour, whether it was tested at >= 4x (ratio <= 0.25), best behaviour at >= 4x, parents tested, verdict counts, status. |
| `v_findings`, `v_fold_coverage` | Finding with evidence counts. Source path with the shards that read or skipped it, and `cited_not_declared_read`. |

Rules the ingester applies:

- **Nothing is paraphrased.** Text columns hold the shard's text. JSON numbers keep their digits (`0.0310` stays
  `0.0310`). Numeric columns are parsed copies, NULL when the text is not one number.
- **Merging.** Records merge only when they have the same key. An alias never merges two records, because the
  extractors give each attempt its own row and list the base id as an alias. When shards disagree, every variant
  is kept in `fold_arm.conflicts` with its shard and sources. A disagreeing number or `verdict_class` is left
  NULL in its column; nothing is picked.
- **Existing rows.** An arm whose key is already an experiment from another ingester (for example the
  `OC01-F-Q35-4B-*` foldhist rows) attaches to that row, and the other ingester's columns are left as they were.
  Technique names match case- and space-insensitively; unknown names become new `technique` rows.
- **Directives.** A Cyril quote is written to `directive` only after it is found in its source (a user-prompt
  event, or the cited file; `...` quotes are checked fragment by fragment, in order). It is skipped when a
  directive from another ingester already contains it, or when it is part of a longer foldrec quote. Keys
  follow `ingest/directives.py`, so the same words are never stored twice.
- **Refusals.** A shard that is not valid `foldrec-v1` JSON, or that reuses another shard's name, makes the ingest
  refuse. An enum value outside the format becomes NULL (`other` for a component kind), and the raw value stays
  in `record` and on the batch notes.

Dry run on production, 2026-10-22, with extraction still running: 23 shards, 6,782 records, 2,743 arms. 465
keys are described by more than one shard, and all of them carry kept conflicts. 9 arms attach to the foldhist
`OC01-F-Q35-4B-*` rows. Also 3,508 techniques, 1,239 findings and 378 directives: 194 would be written; 30
quotes are not found in their cited source; 12 cite a git commit message, which the ingester cannot read.
Ingesting a frozen copy of those 23 shards into a throwaway database wrote 40,783 rows in 338 s. A second pass
wrote 0.

## Running a test and comparing (the runner)

One command runs any model on any suite, prints the results, writes them to the database and prints the
comparison. Run it on the AI server from `/home/cereal/omni-campaign/lab` with the intake venv (it has
psycopg2; the worker it starts uses `model-surgery-lab/venv`, which has torch):

```
PY=/home/cereal/.venvs/chimera-intake/bin/python
$PY omni_lab.py run --model PARENT-Q35-0.8B --suite bank_v1 --cases arithmetic-multi-step,context-retrieval --cpu --cap 96
$PY omni_lab.py run --model /mnt/lexar/omni-campaign/distill/fullkd/run1/out/student --suite student_tests_20261021 --gpus 6,4,0
$PY omni_lab.py run --model OC02-F-Q38-27BU-OMNI1 --suite bank_v1 --gpus 0,4 --split 22,22
$PY omni_lab.py run ... --save-test bank-nothink      # name the suite+engine+gen+cards+case filter
$PY omni_lab.py run --test bank-nothink --model <other model>   # rerun the same test on another model
$PY omni_lab.py run ... --dry-run                      # resolve and print everything; claim and start nothing
```

What `run` does, in order:
1. It resolves the model (a ledger checkpoint id, a `model_key`, or a path, which gets a `path:<sha16>` row), the
   suite (`name` or `name@version`; newest by default) and the cases (`--cases` globs, `--category`, `--limit`).
2. It picks the engine from the layout: `hf` (stock transformers, one card or CPU), `qwen35-student` (any
   directory with `student_manifest.json`), `chimera` (`chimera.checkpoint.v1`) or `gguf` (needs `--server-url`
   of a running llama-server). `--engine` overrides. The tokenizer defaults to the model directory, else to the
   one the model's previous eval recorded.
3. It checks the cards. Each `--gpus` card is locked with `flock` on `$OC/distill/queues/gpu<N>.lock`
   (non-blocking: a held lock refuses the run), then refused if it uses more than `--busy-limit` MiB (2000).
   GPU5 is refused without `--allow-gpu5`, and host MemAvailable must stay above `--mem-floor-gib` (40).
   `--dry-run` applies the same refusals without locking.
4. It writes the `run` row (status `running`) and `spec.json` into
   `/mnt/lexar/omni-campaign/scratch/omni-lab-runs/<utc>-<engine>-<model>-<suite>/`, then starts the worker
   under `nice` with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` set to the claimed cards.
   It holds the locks until the worker exits.
5. The worker loads the model through its adapter and generates every case with `oc_calib.generate_greedy`.
   It applies the machine graders and writes `results.jsonl` (one line per case, as each finishes) and
   `receipt.json` (runtime versions, card UUIDs as torch sees them, placement, kernels, tokenizer and template
   sha, thinking requested vs. took effect, stops, tool sha256s).
6. `ingest/lab_runs.py` loads the run directory into the database. The output is a per-case console table,
   the summary line, and a pivot against up to two earlier runs that share cases with this one (a lineage
   parent first). `--baseline 30666,30658` picks them yourself; `--baseline none` skips the pivot.

If the database is down or the runner is killed after the worker, the run directory is still the record.
`omni_lab.py ingest lab_runs` loads it later with the same keys.

What each engine reuses (nothing is reimplemented):

| Engine | Loader | Notes |
|---|---|---|
| `hf` | `oc_run_frozen_bank.load_model`, `stop_token_ids` | On CPU, `qf_arch.force_torch_kernels` runs first. One device only. |
| `qwen35-student` | `student_test.load_student` (strict load, `model.language_model.` -> `model.`, accelerate dispatch) | `device_plan` generalised to 1..N cards. With 3 cards it is `student_test`'s own plan. The S8 ledger-band mask is on unless `--no-band-mask`. |
| `chimera` | `run_omni_frozen_review.py` 1.2.0 as a subprocess, on a generated bank file | Tool cases, raw continuations and non-text inputs are recorded as unsupported. Not yet run end to end. |
| `gguf` | `toolcall_selfgen.Renderer` + `Client` (bank-run greedy sampling, `n_predict` = `--cap`) | Never starts a llama-server. Not yet run end to end. |
| `echo` | none: a mechanics bed that returns the prompt (tests only) | |

Graders (worker side, `graders/registry.py`):

| Grader | Applies to | Result |
|---|---|---|
| `toolcall_selfgen.grade_case` | Tool cases with `expected` | pass/fail |
| `student_test.code_check` | Cases whose `reference.asserts` exists | Uses `student_test`'s harness with the case's own asserts. Score = fraction of asserts passed. |
| `chimera_recovery.evaluate` | Cases with an `oracle` | Vendored Codex grader `evaluation.grade_semantics`, sha `6a84626d…`, at `$L/research/fold/tools/vendor/chimera_recovery`. When the oracle says ungraded, no grade row is written. |

Bank v1 has no machine grader by design. Every result carries `oc_calib.classify_form`'s box and the loop/cap
flags as artifact flags, never as grades. A reader grades afterwards, and a human grade outranks a machine one
everywhere:

```
$PY omni_lab.py grade 37881 context-retrieval correct --why "names Saturday" --grader human:cyril
$PY omni_lab.py grade 37881 --file grades.json --grader human:claude    # {case_id: grade} or {case_id: {grade, why}}
```

Comparing takes well under a second (0.4 s wall from the workstation on 2026-10-21). In the pivot, `C/P/M/N` are
reader grades, `pass/FAIL` are machine grades, `?` means ran but ungraded, `-` means that run never had the case,
`*` marks a human grade, and `~loop ~cap ~empty ~echo ~unsup` are artifact flags:

```
C:/Python314/python.exe omni_lab.py compare 30658 30666 30676          # parent (S0 import) vs OMNI1 vs full-KD student
C:/Python314/python.exe omni_lab.py compare --suite bank_v1 --last 12   # every run that touched bank_v1 cases
C:/Python314/python.exe omni_lab.py compare --model OC02-F-Q38-27BU-OMNI1 --show text
C:/Python314/python.exe omni_lab.py list suites | tests | runs | models qwen | cases bank_v1
```

Adding tests never breaks old comparisons. `add-suite NAME --file F` reads every bank layout already in the
estate: `{"prompts": [...]}`, `{"cases": [...]}`, a list of strings or objects, or Codex `tasks.jsonl`. Per
case it takes `prompt` | `messages` | `raw_text` | `input`, `tools`, `media`, `expected`, `asserts` and `oracle`,
and infers the grader. `add-cases SUITE --file F` makes the next suite version: the old cases are carried
forward, a reused `case_id` with a new prompt replaces that case in the new version only, and every earlier run
shows `-` for the new cases.

Training, fold and cache jobs log themselves without the worker:

```
RID=$($PY omni_lab.py log-run start --kind distill --name "fullkd run2" --model OC02-F-Q38-27BU-OMNI1 \
      --lane L-TRAIN --gpus 4,0 --config @trainer_config.json --log run.log) || true
$PY omni_lab.py log-run metric $RID --metric loss=2.31@100 || true
$PY omni_lab.py log-run end $RID --status pass --exit-code 0 --receipt receipt-001.json --metrics-jsonl steps.jsonl || true
```

`log-run` uses a 10 s statement timeout. Keep the `|| true` so a database outage never stops a job; the
existing ingesters still pick its receipt up later.

## Rollback

```
DROP DATABASE omni_lab; DROP ROLE omni_lab_ro; DROP ROLE omni_lab;     -- as agent_memory
credman.py rm omni_lab_pg; credman.py rm omni_lab_ro_pg
```

Nothing in `agent_memory` or `chimera_corpus` was altered by the bootstrap.
