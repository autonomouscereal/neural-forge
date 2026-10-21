# Omni Lab

One PostgreSQL database (`omni_lab`) for every test, eval, fold, distill, cache and probe result of the
AI-model program (Chimera Omni, Neural Forge, K3, speech, media, corpus), plus the code that writes to it.
Built 2026-10-21 from the Omni Lab design and its critique (cases are global, every run records its software environment (`env`), media outputs go in `eval_artifact`, the tensor census in `model_tensor_group`, and there is a separate read-only role).

This step is the database core only: roles, database, migrations, the shared module, and tests.
The ingesters, the eval runner and the GPU worker come next and use `omni_lab_db.py`.

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
C:/Python314/python.exe -m unittest discover -s tests -v  # throwaway omni_lab_test_<hex> DB, created and dropped
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

A blank cell is a missing row. Add cases by creating a new suite version that links the old `test_case` rows plus the new ones. Older runs then show no result for the new cases. An edited prompt under the same `case_id` becomes a separate `test_case` row, so old and new answers are never compared as if they answered the same question.

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

## Rollback

```
DROP DATABASE omni_lab; DROP ROLE omni_lab_ro; DROP ROLE omni_lab;     -- as agent_memory
credman.py rm omni_lab_pg; credman.py rm omni_lab_ro_pg
```

Nothing in `agent_memory` or `chimera_corpus` was altered by the bootstrap.
