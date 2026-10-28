# Cleanup plan: experimental weights, 2026-10-21

**Status: plan only.** Nothing has been deleted, moved or written to omni_lab. The machine-readable plan is
`C:/Users/cereal/Documents/ai training/omni-lab/cleanup_plan.json` (same bytes in the session scratchpad `lab/cleanup_plan.json`).

Owner order (Cyril, 2026-10-21, verbatim): "[Cyril quote withheld]"

Inputs:
- server_inventory: `C:\Users\cereal\AppData\Local\Temp\claude\C--Users-cereal\0c189f21-3de3-4c40-b42c-db54c9d84462\scratchpad\lab\cleanup_server.json` sha256 `1ecb8e6915fd56db...`
- workstation_inventory: `C:\Users\cereal\Documents\ai training\omni-lab\cleanup_workstation.json` sha256 `86f8d30bb1b3f157...`

## 1. What this frees

| Disk | Free at scan | Freed by DELETE | Held back (needs a ruling) |
|---|---|---|---|
| AI server `/` (home) | 64G (97% used) | 612.8 GB | 0 |
| AI server `/mnt/lexar` | 317G (92%) | 880.2 GB | 69.0 GB |
| NAS `/mnt/ai_data` | 2.5G (100%) | 1043.0 GB | 0 |
| Workstation C: | 1.4 GB | 69.16 GiB (4 of these GiB wait on Codex) | 0 |
| Workstation D: | 36 GB | 14.03 GiB | 0 |
| Workstation E: | 1.3 TB | 0 | 0 |

Server total 2536.1 GB across 248 units; workstation 83.19 GiB across 89 items. Server sizes are `du -sb` per unit (decimal GB); workstation sizes are weight bytes (GiB). Removing whole worktrees frees more than the weight bytes; that extra is not counted.

The UNSURE lists (section 10) hold another ~2.7 TB on the server, mostly on the NAS. None of it is deleted by this plan.

## 2. Lead changes to the inventories

| Path | Inventory said | Plan says | Why |
|---|---|---|---|
| `/mnt/lexar/omni-campaign/distill/fullkd/run1/safety` (52.32 GB) | DELETE | HOLD | Resume point (ckpt-00000611: weights + optimizer/trainer state) of the KEEP-best run-1 student; inventory shows its first shard hashes differently from out/student, so it is NOT a duplicate. Touched 45 h ago. Deleting it ends any chance of resuming run 1. Needs an explicit owner/lead ruling that run 1 will never be resumed (v2 is a new 2B/4B model, so the expected ruling is delete). |
| `/mnt/lexar/omni-campaign/distill/media_m1/run1/out/checkpoints` (10.23 GB) | DELETE | HOLD | M1 intermediate checkpoints (step-0000294/0000300 media.safetensors + optimizer.pt) beside the kept M1 export; touched 55 h ago. Same question as run1/safety: resume point vs stale. Delete after the ruling. |
| `/mnt/lexar/omni-campaign/research/fold/x3-08b` (6.39 GB) | DELETE | HOLD | Named in LANES.json do_not_touch of L-RESEARCH (lane parked, not ended). The keep rules forbid deleting anything referenced by an active config; release the lane (lanes.py) first, then it moves to DELETE. |
| `/home/cereal/omni-campaign/import/rq27b` (0.35 GB) | DELETE | KEEP-live | flash_score_queue.sh (pid 4009138, running) exports CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src; the inventory missed this reference because the queue uses $OC-relative paths. Never delete while the queue runs. |
| `/mnt/ai_data/neural-forge-stage2/corpus` (0.8 GB) | DELETE | UNSURE | Neural Forge stage-2 corpus (0.8 GB). Corpora are KEEP by rule even when the experiment is dead; owner decides. |
| `/home/cereal/omni-campaign/composite/corpus` (0.03 GB) | DELETE | UNSURE | Named 'corpus' (superseded k3/qwen input_ids; 4.7 MB of weight-format files, 0.03 GB du). Corpora are KEEP by rule; no space to gain. Keep. |
| `/mnt/ai_data/neural-forge-stage2/calib` (0.03 GB) | DELETE | UNSURE | Neural Forge stage-2 calibration data (tiny). Data, not weights; keep with the stage-2 corpus decision. |

Other findings that change how the deletion runs:

1. **`oc_checkpoint.py delete` cannot be used.** Its R4/R6 clauses refuse anything under `/mnt` (Lexar included), refuse stage `clone`, and accept only status `bad` with content misses (`oc_checkpoint.py` 1.1.0, `_delete_conditions`). The 14 ledger-registered checkpoints in the DELETE list get an appended ledger record through `set-status --cyril-ruling "<owner order>" --notes "<deletion record>"` with the status left unchanged, and the deletion itself is recorded in omni_lab (section 6). This bypasses the policy's per-checkpoint R4 gate on the owner's explicit order. If the lead prefers, add a reviewed `retire` subcommand (event `delete`, `deleted: true`, owner ruling, no R4) to oc_checkpoint 1.2.0 instead.
2. **omni_lab would undo the deletions.** `omni_lab_db.upsert_model` sets `status = EXCLUDED.status`, `ingest/oc_ledger.py` writes `present`/`missing` from path presence, and `ingest/model_scan.py` resets existing rows to `present` (and absent ones to `missing`). A re-ingest would turn `deleted` into `missing`. Patch all three to never change a `deleted` row before any row is marked (W0).
3. **omni_lab has no deletion columns.** `model.status` already allows `deleted`, but there is no `deleted_at`, reason or recipe path. Migration 0006 adds them (section 6).
4. **omni_lab has no backup.** The agent-memory cron (`normal_backup_runner`, 02:20) dumps only the `agent_memory` database (`backup_agent_memory_pg.sh` uses `-d ${POSTGRES_DB:-agent_memory}`). `omni_lab` (229 MB today) and `chimera_corpus` live in the same container and are not dumped (section 5).
5. **The qwen-fold children are not on the NAS.** The only copy is `/home/cereal/omni-campaign/qwen-fold` (153.6 GB). It is still DELETE (OC01 ladder children), so after this plan there is no copy anywhere. The recipes and the 9 OC01 ledger records (QF-* children) keep the rebuild path; the qwen-fold tree is 25 units / 153.5 GB in W2.
6. **The NAS is full (2.5 GB free), so it cannot hold recipe copies yet.** Recipes go to Lexar first and are mirrored to the NAS after W1 frees space there.
7. **The workstation has no omni_lab rows** (0 of 1,665 rows are on host `workstation`, and there is no `workstation` host row). W0 creates the host row.

## 3. Order of work

| Wave | What | Gate |
|---|---|---|
| W0 | DB backups (section 5) plus a pre-cleanup dump of omni_lab, agent_memory and chimera_corpus. Migration 0006, the three ingester patches, the `workstation` host row, and the recipe archive root `/mnt/lexar/omni-campaign/archive/recipes/`. | Dump restores into a scratch DB (`pg_restore --list` and row counts match) |
| W1-NAS | NAS deletes (frees the full share first; unblocks NAS writes): 71 units, 1043.0 GB | Per-unit procedure (section 4); omni_lab dump after the wave |
| W2-HOME | AI server `/` deletes: 118 units, 612.8 GB | Per-unit procedure (section 4); omni_lab dump after the wave |
| W3-LEXAR | Lexar deletes: 59 units, 880.2 GB | Per-unit procedure (section 4); omni_lab dump after the wave |
| W4 | Workstation: git rm of the 7 Neural Forge children, then 78 clean worktrees removed with `git worktree remove`; the 4 worktrees with untracked files go to Codex on Relay | Relay notice to Codex first (the worktrees are Codex agent worktrees); re-check processes |
| W5 | HOLD items, once ruled; UNSURE items the owner approves | Written ruling recorded in omni_lab `model_deletion.reason` |

Do not delete anything while these are running, and re-check them before each wave:
- Flash teacher scoring: pids 1183917 (teacher-score), 1179896 (gguf_teacher_cache), 4009138 (flash_score_queue.sh). They use the Strata GGUF, `export/llamacpp`, `distill/queues`, `distill/corpus` and `import/rq27b`.
- K3 host loop: pid 4169670 (about 12 days) in `omni-campaign/runtime/k3-nas-window61-72-oct07-a151e1a8-...`. This is UNSURE / never-delete.
- Crons: agent-memory backup (02:20), `lanes.py pulse` (*/10) and `intake_cron.sh` (*/30, reads the NAS and Lexar intake drop zones).

## 4. Per-unit procedure (executor contract)

One executor runs this for each unit, holding one lock and one SSH lease per batch, and stops the wave on the first failure.

1. **Preflight.**
   - No pid has a file under the unit open (`/proc/*/fd`, `/proc/*/maps` and cwd, read as root; on Windows, Win32_Process command lines).
   - Neither the crontab, the running cmdlines, `flash_score_queue.sh`, `intake_cron.sh`, the K3 host-loop config nor LANES.json `do_not_touch` names the unit or a parent of it.
   - The unit holds no non-weight file of 50 MB or more (such a file may be data: HOLD the unit).
   - The unit holds no `teacher_cache`-format manifest and no `corpus`/`holdouts`/`receipts` directory (HOLD if found).
   - For the clones, the ledger `directory_sha256` of `OC02-F-Q38-27BU-OMNI1.clone-pre-D1` equals the one for `OC02-F-Q38-27BU-OMNI1`. If they differ, the clone is not a byte clone: HOLD it.
   - Record the preflight JSON.
2. **Measure.**
   - File list with size and mtime. `layout`: `hf` if `config.json` is present, `chimera.checkpoint.v1` if a chimera manifest is present, `gguf`, otherwise `safetensors-raw`.
   - `parameter_count`: sum of tensor numel from the safetensors headers (header read only) or from the GGUF metadata.
   - **Digest.** Units under 20 GB: full sha256 of every file. Units of 20 GB or more: sha256 of every index/config/manifest JSON, a sha256 of each safetensors header (the 8-byte length plus the JSON), and a listing digest = sha256 over the sorted `relpath\tsize\tsha256-or-header_sha256` lines.
   - Full hashing covers {'home': 322.9, 'lexar': 130.9, 'nas': 130.3} GB.
   - The NAS share has been measured hashing below 3 MB/s over CIFS (oc_checkpoint.py docstring), so the NAS share of that hashing (~130 GB) can take about 12 h. Run W1 detached, or hash on the NAS host itself.
3. **Archive the recipe.**
   - Copy every non-weight file under 50 MB (configs, `*.index.json`, manifests, tokenizer files, receipts, `*.status`, logs, recipe/command files) to `/mnt/lexar/omni-campaign/archive/recipes/<model>/tree/`, keeping relative paths.
   - Write `<model>/DELETION_RECEIPT.json` (all fields below) and `<model>/MANIFEST.tsv`, then sha256 the archive directory.
   - `<model>` = the ledger checkpoint id when the unit has exactly one, else `path-<sha16>__<last two path parts>` (sha16 = `sha256('<host>:<path>')[:16]`, the key omni_lab already uses).
4. **Record in omni_lab**, in one transaction, before the unlink:
   - Ensure a unit row exists (`model_key` `path:<sha16>`, role from `what`, project `neural-forge` for NF units, `k3` for K3 units and `chimera-omni` otherwise).
   - Set `status='deleted'`, `deleted_at`, `delete_reason` and `rebuild_recipe_path` on the unit row and on every existing row inside the unit.
   - Insert one `model_deletion` row (section 6).
5. **Ledger** (only for the 14 checkpoint ids listed in section 8): `oc_checkpoint.py set-status --checkpoint-id ID --cyril-ruling "<order>" --notes "deleted <date>; reason; recipe <path>; omni_lab model <id>" --confirm`. Store the sealed `receipt_sha` in `model_deletion.ledger_receipt_sha`.
6. **Delete:** `rm -rf --one-file-system <unit>`. On the workstation use `git worktree remove` or `git rm` as listed.
7. **Verify:** the path is absent, `df` moved by about the unit size, and `model_deletion.executed_at` / `verified_absent_at` are set. Mirror the recipe directory to `/mnt/ai_data/backups/omni-lab-recipes/` once the NAS has space.
8. After each wave: run an omni_lab dump, then `omni_lab.py db check`.

## 5. DB backups (owner asked: "[Cyril quote withheld]")

Reuse the proven agent-memory pipeline instead of writing a new one:

- **Parametrise the script.** Make `backup_agent_memory_pg.sh` take the database name as `DB_NAME`, plus its own lock and `DEST_ROOT`. Today the database name is hard-coded in the dump, the manifest `"database": "agent_memory"` and the restore test.
  - It stays raw `pg_dump --format=custom` inside `agent-memory-postgres`, with no credentials on the command line (container env).
  - Keep the existing sha256 publication and `pg_restore --list` check.
- **Databases:** `omni_lab` and `chimera_corpus`. `agent_memory` keeps its existing job.
- **Destinations:** `/mnt/lexar/backups/postgres/<db>/` (local, works now) and `/mnt/ai_data/backups/omni-lab-postgres/<db>/` (the NAS: enable after W1 frees space).
  - Media-server copy via the existing rsync/bwlimit path.
  - 30-day retention, the same as `RETENTION_DAYS=30` on agent-memory.
- **Schedule:** a cron at 02:50 daily (after the 02:20 agent-memory run), `flock`ed, plus on-demand runs before W1, after each wave and before any omni_lab migration.
- **Proof:** each run restores into `omni_lab_restore_test`, compares row counts of `model`, `run`, `metric`, `eval_result`, `ledger_event` and `model_deletion` against the source, then drops the test database.
- **Skill:** add an `omni-lab` section to the `agent-memory-backup` skill, or a sibling skill `omni-lab-backup`, so the job is discoverable.

## 6. omni_lab changes (migration 0006, raw SQL)

```sql
-- 0006_deletions.sql
ALTER TABLE model ADD COLUMN deleted_at TIMESTAMPTZ,
                  ADD COLUMN delete_reason TEXT,
                  ADD COLUMN rebuild_recipe_path TEXT;
CREATE TABLE model_deletion (                 -- ledger-equivalent record of every weight dir removed
    id                   BIGSERIAL PRIMARY KEY,
    model_id             BIGINT NOT NULL REFERENCES model(id),
    duplicate_of_model_id BIGINT REFERENCES model(id),   -- workstation worktree copies -> the 7 child rows
    host_id              BIGINT NOT NULL REFERENCES host(id),
    unit_path            TEXT NOT NULL,
    plan_wave            TEXT NOT NULL,
    size_bytes_du        BIGINT NOT NULL,
    weight_bytes         BIGINT,
    file_count           INTEGER,
    layout               TEXT,
    parameter_count      BIGINT,
    digest_mode          TEXT NOT NULL CHECK (digest_mode IN ('full-sha256','index+headers+listing','git-blob')),
    listing_sha256       TEXT NOT NULL,
    file_digests         JSONB NOT NULL,
    recipe_archive_path  TEXT NOT NULL,
    recipe_archive_sha256 TEXT,
    recipe_files         JSONB NOT NULL,
    producing_run_ids    BIGINT[],
    ledger_checkpoint_id TEXT,
    ledger_receipt_sha   TEXT,
    reason               TEXT NOT NULL,
    owner_order          TEXT NOT NULL,
    preflight            JSONB NOT NULL,
    recorded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    executed_at          TIMESTAMPTZ,
    verified_absent_at   TIMESTAMPTZ,
    UNIQUE (host_id, unit_path)
);
CREATE INDEX model_deletion_model ON model_deletion(model_id);
GRANT SELECT ON model_deletion TO omni_lab_ro;
```

Also in W0:
- `INSERT INTO host(name) VALUES ('workstation') ON CONFLICT (name) DO NOTHING`.
- Patch `omni_lab_db.upsert_model` to `status = CASE WHEN model.status = 'deleted' THEN 'deleted' ELSE EXCLUDED.status END`.
- Add the same guard to `ingest/model_scan.py` (the `UPDATE ... status = 'present'` at line ~199 and the `missing` sweep at line ~240) and to `ingest/oc_ledger.py`.
- Add a unit test for each and copy the code to `$OC/lab` (sha256 parity).

Rows: the plan creates 213 unit rows (units with no row at their exact path) and marks 509 existing rows deleted. All existing matched rows lie inside their unit; no row for a parent directory is touched. Workstation: 7 child rows (one per distinct blob) plus 89 `model_deletion` rows.

## 7. Recipe archive

- Root: `/mnt/lexar/omni-campaign/archive/recipes/<model>/`, holding `tree/`, `MANIFEST.tsv` and `DELETION_RECEIPT.json`. Mirror to the NAS after W1.
- Rebuild sources that stay: the parents (section 11); the 14 sealed ledger chains (`$OC/ledger.jsonl` plus the registry); the run rows, metrics and evals already in omni_lab; neural-forge `.git` for the workstation children; and the k3-window61-72 repository `.git`.
- Archive contents are per unit, as section 4 step 3 defines. The plan does not list file-level recipe sets because the server scan recorded weight files only; the executor enumerates them at run time.

## 8. Ledger-registered checkpoints in the DELETE list

- `OC01-F-Q35-4B-C2K1152`
- `OC01-F-Q35-4B-C2K2304`
- `OC01-F-Q35-4B-G5D32`
- `OC01-F-Q35-4B-X00`
- `OC01-F-Q35-4B-X03`
- `OC01-F-Q35-4B-X08`
- `OC01-F-Q35-4B-X15`
- `OC01-F-Q35-4B-X16`
- `OC01-F-Q35-4B-X31`
- `OC02-D-Q38-27BU-OMNI1D1`
- `OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2`
- `OC02-F-Q38-27BU-C2K4352`
- `OC02-F-Q38-27BU-C2K4352K3U`
- `OC02-F-Q38-27BU-OMNI1.clone-pre-D1`

`OC02-D-Q38-27BU-OMNI1D1` is a delta whose base is `OC02-F-Q38-27BU-OMNI1.clone-pre-D1`. Both are deleted together, so no chain is left half-standing. `OC02-F-Q38-27BU-OMNI1` (KEEP-best) is not touched.

## 9. DELETE list

### Server

Columns: GB = du; rows = existing omni_lab rows inside the unit (all marked deleted); new = unit row to create; ledger = checkpoint id; digest = full or index+headers+listing.

#### W1-NAS: 71 units, 1043.0 GB

| GB | what | rows | new | ledger | digest | path |
|---:|---|---:|:-:|---|---|---|
| 110.11 | fold-scratch | 0 | y |  | idx+hdr | `/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139` |
| 100.29 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge-sealed-caches/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-w13m-firstcomposed-cache` |
| 100.29 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge-sealed-caches/a0023-exact4x-h816-mgv3-r12-round-0-refold.sealed-armA-complete-cache` |
| 100.29 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge-archives/a0010-lfm2-exact4x-h816-ahb-policy-resume` |
| 98.85 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge-archives/neural_forge_on_policy_rounds_v1` |
| 97.78 | fold-scratch | 0 | y |  | idx+hdr | `/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c` |
| 91.51 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge-archives/a0023-exact4x-h816-mgv3-r12` |
| 57.49 | fold-scratch | 0 | y |  | idx+hdr | `/mnt/ai_data/omni-campaign/k3-run-oct09-claude-archive` |
| 55.4 | model | 0 | y |  | idx+hdr | `/mnt/ai_data/omni-campaign/s0-probe-sep04` |
| 41.23 | cache | 0 | y |  | idx+hdr | `/mnt/ai_data/backups/lexar-relief-20261018/omni-campaign` |
| 34.9 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710` |
| 24.65 | nf-experiment | 0 | y |  | idx+hdr | `/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710` |
| 8.49 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/progressive_stage1_20260709` |
| 8.05 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/qwen3_4b_compensated_dynamic_v10_20260716_diagnostic_pid947219` |
| 7.75 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave11` |
| 5.8 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/native_parent_cache_20260710` |
| 4.58 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/same_token_tokenrows_outputgate_20260710` |
| 4.58 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/real_medoid_dual_20260710` |
| 4.57 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave13` |
| 4.56 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/progressive_closed_form_20260710` |
| 4.22 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave25` |
| 3.49 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave8` |
| 3.22 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave24` |
| 3.16 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave28` |
| 3.12 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-ahb-cache-v1` |
| 3.04 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave27` |
| 2.89 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave31` |
| 2.83 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/canonical_blank_targets` |
| 2.82 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/compiled_child_cache` |
| 2.8 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave15` |
| 2.63 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave14` |
| 2.61 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave17` |
| 2.52 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave7` |
| 2.44 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave26` |
| 2.24 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/cross_family_blank_20260710` |
| 2.21 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave29` |
| 2.15 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave20` |
| 2.14 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave18` |
| 2.12 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave16` |
| 2.01 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/archive` |
| 1.96 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/dynamic_snapshots` |
| 1.77 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave21` |
| 1.76 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave19` |
| 1.76 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave30` |
| 1.6 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave9` |
| 1.51 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave23` |
| 1.45 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave10` |
| 1.4 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/runs` |
| 1.32 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave34` |
| 1.16 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/trajectory_subset_targetrows_20260710` |
| 1.12 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave22` |
| 1.09 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/backup` |
| 1.08 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave32` |
| 0.95 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave36` |
| 0.91 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave35` |
| 0.9 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave40` |
| 0.82 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave12` |
| 0.64 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave43` |
| 0.6 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/cross_token_real_medoid_20260710` |
| 0.58 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave33` |
| 0.5 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave37` |
| 0.49 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave38` |
| 0.37 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave41` |
| 0.37 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave42` |
| 0.29 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/qwen_basis_checkpoint_20260710` |
| 0.2 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/cells` |
| 0.2 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave45` |
| 0.18 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave44` |
| 0.17 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-matrix/wave6` |
| 0.03 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge/audits` |
| 0.03 | nf-experiment | 0 | y |  | full | `/mnt/ai_data/neural-forge-stage2/smoke` |

#### W2-HOME: 118 units, 612.8 GB

| GB | what | rows | new | ledger | digest | path |
|---:|---|---:|:-:|---|---|---|
| 102.43 | fold-scratch | 12 | y |  | idx+hdr | `/home/cereal/omni-campaign/k3-run-oct09-claude/fold` |
| 53.89 | model | 1 | y | OC02-F-Q38-27BU-C2K4352K3U | idx+hdr | `/home/cereal/omni-campaign/composite/c10/S5-C2K4352K3U` |
| 43.63 | model | 3 | y |  | idx+hdr | `/home/cereal/omni-campaign/composite/c10/design_sweep` |
| 37.08 | scratch | 12 | y |  | idx+hdr | `/home/cereal/omni-campaign/composite/s7` |
| 32.21 | media-unit | 12 | y |  | idx+hdr | `/home/cereal/omni-campaign/composite/c17/trunk_taps` |
| 20.69 | scratch | 12 | y |  | idx+hdr | `/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013` |
| 18.8 | media-unit | 12 | y |  | full | `/home/cereal/omni-campaign/composite/c17/charts` |
| 12.82 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730` |
| 12.04 | fold-scratch | 0 | y |  | full | `/home/cereal/lexar-relief-local/omni-campaign/k3-fold-sep07-codex` |
| 9.23 | scratch | 3 | y |  | full | `/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1` |
| 9.1 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-T9` |
| 8.89 | model | 1 |  | OC01-F-Q35-4B-X15 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X15` |
| 8.89 | model | 1 |  | OC01-F-Q35-4B-X03 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X03` |
| 8.89 | model | 1 |  | OC01-F-Q35-4B-X31 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X31` |
| 8.88 | model | 1 |  | OC01-F-Q35-4B-X16 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X16` |
| 8.88 | model | 1 |  | OC01-F-Q35-4B-X00 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X00` |
| 8.88 | model | 1 |  | OC01-F-Q35-4B-X08 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X08` |
| 8.71 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810` |
| 8.42 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-T9.gguf` |
| 8.21 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X03.gguf` |
| 8.21 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X15.gguf` |
| 8.21 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X31.gguf` |
| 8.2 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/neural-forge-src/runs` |
| 8.2 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X16.gguf` |
| 8.2 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X00.gguf` |
| 8.2 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-X08.gguf` |
| 7.59 | model | 1 |  | OC01-F-Q35-4B-G5D32 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-G5D32` |
| 6.91 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-G5D32.gguf` |
| 6.7 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_moving_gauge_v5_20260804` |
| 6.46 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neo_native_dynamic_6143_catalog_a4_prod_20260716_r6` |
| 6.29 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/composite/scratch` |
| 5.89 | scratch | 12 |  |  | full | `/home/cereal/omni-campaign/distill/trainer-lane` |
| 5.85 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/distill/parity` |
| 5.7 | model | 1 |  | OC01-F-Q35-4B-C2K2304 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-C2K2304` |
| 5.66 | media-unit | 5 | y |  | full | `/home/cereal/omni-campaign/output/native_executor/qimg_parallel` |
| 5.63 | nf-experiment | 0 | y |  | full | `/home/cereal/nf-mxe/evidence-cache` |
| 5.14 | model | 1 |  | OC01-F-Q35-4B-C2K1152 | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-C2K1152` |
| 4.46 | model | 1 |  |  | full | `/home/cereal/omni-campaign/qwen-fold/children/QF-C2K1152.gguf` |
| 4.45 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_layer0_study_v1` |
| 4.4 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1` |
| 3.88 | scratch | 2 | y |  | full | `/home/cereal/omni-campaign/import-lane/s4` |
| 3.46 | model | 2 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/dryrun` |
| 3.42 | scratch | 2 | y |  | full | `/home/cereal/omni-campaign/import-lane/s5` |
| 3.32 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1` |
| 3.14 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_production_20260728` |
| 2.05 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_interface_frontier_20260723` |
| 1.7 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_ahb_cache_local_v1` |
| 1.68 | scratch | 1 | y |  | full | `/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1` |
| 1.68 | scratch | 1 |  |  | full | `/home/cereal/omni-campaign/composite/s0/qwen35_0p8b_chimera_bf16_music` |
| 1.57 | scratch | 1 | y |  | full | `/home/cereal/omni-campaign/import-lane/s4-fix` |
| 1.54 | scratch | 1 |  |  | full | `/home/cereal/omni-campaign/composite/s0/qwen35_0p8b_chimera_bf16_s0` |
| 1.53 | scratch | 1 | y |  | full | `/home/cereal/omni-campaign/import-lane/s5b` |
| 1.4 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_large_ratio_20260716` |
| 1.23 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717` |
| 1.21 | checkpoint | 1 | y |  | full | `/home/cereal/omni-campaign/distill/d1` |
| 1.13 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716` |
| 1.08 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/runs/diagnostic_smokes` |
| 1.08 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003` |
| 0.96 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_resume_20260718` |
| 0.89 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/distill/media_b` |
| 0.83 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_heterogeneous_20260729` |
| 0.81 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/distill/scratch` |
| 0.78 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_causal_20260804` |
| 0.78 | nf-experiment | 0 | y |  | full | `/home/cereal/nf-mxd/evidence-cache` |
| 0.76 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_ahb_20260808` |
| 0.69 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_causal_head_frontier_20260724` |
| 0.66 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/composite/scratch_review_tests` |
| 0.65 | nf-experiment | 2 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_fused_component_diagnostics_20260727` |
| 0.61 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_partial_neuron_bank_20260717` |
| 0.57 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_real_attention_20260719` |
| 0.54 | nf-experiment | 1 |  |  | full | `/home/cereal/model-surgery-lab/neural-forge-src/$OUT` |
| 0.51 | model | 1 | y |  | full | `/home/cereal/omni-campaign/composite/c10/rehearsal-k3-1-60` |
| 0.5 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_parent_free_causal_resume_20260729` |
| 0.45 | nf-experiment | 4 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_causal_support_20260730` |
| 0.44 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_automatic_dense_budget_20260717` |
| 0.43 | scratch | 1 |  |  | full | `/home/cereal/omni-campaign/composite/media/h3-native-curve6-sep07` |
| 0.33 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_bridge_only_control_20260723` |
| 0.3 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_postfold_20260721` |
| 0.26 | nf-experiment | 4 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neo_native_dynamic_6143_prod_20260716_r5` |
| 0.22 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/review-s3` |
| 0.22 | nf-experiment | 0 | y |  | full | `/home/cereal/nf-round-debug/screened-round` |
| 0.21 | scratch | 8 | y |  | full | `/home/cereal/omni-campaign/composite/logs` |
| 0.2 | nf-experiment | 3 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neo_native_dynamic_6143_cache_portfolio_20260716_v1` |
| 0.18 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/import/omni-composite` |
| 0.13 | scratch | 12 | y |  | full | `/home/cereal/omni-campaign/composite/scratch_review_tests_s8` |
| 0.13 | scratch | 6 | y |  | full | `/home/cereal/omni-campaign/composite/scratch_review_tests_r2` |
| 0.12 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_target_lattice_qk_20260724` |
| 0.12 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721` |
| 0.12 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260720` |
| 0.11 | nf-experiment | 3 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_carrier_replant_20260730` |
| 0.1 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_protected_short_conv_diagnostic_20260727` |
| 0.1 | scratch | 4 | y |  | full | `/home/cereal/omni-campaign/composite/s8` |
| 0.1 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_progressive_component_ablation_20260727` |
| 0.07 | scratch | 11 | y |  | full | `/home/cereal/omni-campaign/composite/scratch_review_tests_s8g` |
| 0.07 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neo_native_dynamic_6143_prod_20260716_r3` |
| 0.06 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/src` |
| 0.06 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/src_review_s4fix` |
| 0.06 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/src_committed` |
| 0.06 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/review_s3` |
| 0.06 | model | 12 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/.review-20260902` |
| 0.06 | model | 12 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/.work-k2kv` |
| 0.06 | model | 12 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/.work-blockerfix` |
| 0.06 | model | 12 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/.work` |
| 0.06 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-review-s2/s2` |
| 0.05 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-omni-import/review_s1` |
| 0.05 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-review-s2/s1` |
| 0.05 | scratch | 0 | y |  | full | `/home/cereal/omni-campaign/chimera-import-s1/base` |
| 0.04 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_layer0_mlp_diagnostic_20260721` |
| 0.04 | nf-experiment | 2 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_partial_short_conv_frontier_20260727` |
| 0.03 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_layer0_current_normalized_20260721` |
| 0.03 | nf-experiment | 1 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730` |
| 0.02 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_statistics_20260721` |
| 0.01 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_crevf_hj8_20260717` |
| 0.01 | scratch | 8 | y |  | full | `/home/cereal/omni-campaign/composite/s0/scratch_review_s0_fable` |
| 0.01 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_mapped_causal_target_20260729` |
| 0.01 | model | 1 | y |  | full | `/home/cereal/omni-campaign/qwen-fold/.work-k2kv-probe` |
| 0.0 | nf-experiment | 0 | y |  | full | `/home/cereal/model-surgery-lab/artifacts/lfm2_circuit_cache_smoke_20260716` |
| 0.0 | scratch | 1 | y |  | full | `/home/cereal/omni-campaign/distill/codex-native-tool-tiny-cli-347f67117974` |

#### W3-LEXAR: 59 units, 880.2 GB

| GB | what | rows | new | ledger | digest | path |
|---:|---|---:|:-:|---|---|---|
| 122.71 | media-unit | 1 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/native_executor/qimg_first32` |
| 110.02 | fold-scratch | 12 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318/fold` |
| 89.46 | fold-scratch | 12 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1/fold` |
| 82.2 | fold-scratch | 12 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-continuation-oct08-claude/fold` |
| 55.07 | model | 1 |  | OC02-F-Q38-27BU-OMNI1.clone-pre-D1 | idx+hdr | `/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1` |
| 50.52 | fold-scratch | 10 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1/ancestry` |
| 33.74 | fold-scratch | 12 |  |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1/body` |
| 33.74 | fold-scratch | 12 |  |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318/body` |
| 30.68 | model | 3 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd` |
| 29.07 | model | 1 | y | OC02-F-Q38-27BU-C2K4352 | idx+hdr | `/mnt/lexar/omni-campaign/s3-fold-sep07-codex` |
| 24.02 | fold-scratch | 10 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-fold-sep07-codex-scorefix/fold` |
| 23.91 | media-unit | 3 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/h3-joint50-fixed3584-native-01a11929-de0a9de7-873d-4441-aa8f-4d9f25935af3` |
| 23.51 | fold-scratch | 2 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-c17-capture-l3-l4-sep07` |
| 20.55 | fold-scratch | 1 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318/ancestry` |
| 20.03 | fold-scratch | 1 | y |  | idx+hdr | `/mnt/lexar/omni-campaign/large-fold-opt-oct07` |
| 16.44 | fold-scratch | 1 | y |  | full | `/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-d70092c8-55c4-4b20-a516-ac33bf64206d/fold` |
| 11.44 | fold-scratch | 4 | y |  | full | `/mnt/lexar/omni-campaign/k3/scratch_veto` |
| 9.13 | media-unit | 12 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks` |
| 8.71 | fold-scratch | 6 |  |  | full | `/mnt/lexar/omni-campaign/k3-fold-sep07-codex-scorefix/body` |
| 8.56 | checkpoint | 1 |  |  | full | `/mnt/lexar/omni-campaign/distill/d1/OMNI1D1-rungs-canonical` |
| 8.56 | model | 1 |  | OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2 | full | `/mnt/lexar/omni-campaign/clones/OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2` |
| 8.56 | delta | 1 |  | OC02-D-Q38-27BU-OMNI1D1 | full | `/mnt/lexar/omni-campaign/deltas/OC02-D-Q38-27BU-OMNI1D1-hardened-claude-20261014` |
| 8.46 | fold-scratch | 1 |  |  | full | `/mnt/lexar/omni-campaign/k3/snapshots` |
| 6.75 | fold-scratch | 3 | y |  | full | `/mnt/lexar/omni-campaign/k3/veto_diag` |
| 6.19 | media-unit | 8 | y |  | full | `/mnt/lexar/omni-campaign/qimg-broad-capture-sep07` |
| 5.82 | scratch | 4 | y |  | full | `/mnt/lexar/omni-campaign/distill/parity/k3cache-2026-10-14-b2-gpu6` |
| 4.7 | fold-scratch | 1 |  |  | full | `/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-d70092c8-55c4-4b20-a516-ac33bf64206d/body` |
| 4.1 | media-unit | 10 | y |  | full | `/mnt/lexar/omni-campaign/h3-noise-evaluation-native-01a11929-516e8e90-70f5-4cd5-a28c-b16d9a4a4e13` |
| 3.52 | media-unit | 5 | y |  | full | `/mnt/lexar/omni-campaign/media/h3_blocks_v2` |
| 3.11 | media-unit | 5 | y |  | full | `/mnt/lexar/omni-campaign/media/h3_blocks` |
| 3.11 | media-unit | 2 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/final-dense-assembly-905bd7e6-edb0-42ac-b8b5-a91b6f212573` |
| 3.08 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/receipt-q27` |
| 2.32 | fold-scratch | 2 | y |  | full | `/mnt/lexar/omni-campaign/k3/dense` |
| 2.04 | media-unit | 4 | y |  | full | `/mnt/lexar/omni-campaign/c17-smoke` |
| 1.56 | media-unit | 2 | y |  | full | `/mnt/lexar/omni-campaign/qimg-fit-held-sep07` |
| 1.41 | media-unit | 9 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/completed-blocks` |
| 0.66 | fold-scratch | 1 | y |  | full | `/mnt/lexar/omni-campaign/k3/resid` |
| 0.47 | media-unit | 3 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/initial-gpu4-attempt2` |
| 0.42 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/qimg-source-capture-canary-sep07` |
| 0.3 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-parent-batch1-sep07` |
| 0.27 | checkpoint | 1 | y |  | full | `/mnt/lexar/omni-campaign/distill/d1/run-OMNI1D1-native-01a10f78-20261014-c36-recovery-r2` |
| 0.26 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/qimg-ownerpolicy-v4-ee1a6166-3e91-405d-8381-3e3ff4c91194` |
| 0.25 | fold-scratch | 1 |  |  | full | `/mnt/lexar/omni-campaign/k3/veto_diag_h2` |
| 0.17 | media-unit | 2 | y |  | full | `/mnt/lexar/omni-campaign/native_executor/qimg_parallel` |
| 0.12 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/media/h3_block_coreset_v2` |
| 0.12 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/media/h3_block_coreset_v1fix` |
| 0.12 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/media/h3_block_coreset` |
| 0.06 | scratch | 3 | y |  | full | `/mnt/lexar/omni-campaign/scratch/toolscreen-20261015` |
| 0.05 | scratch | 3 | y |  | full | `/mnt/lexar/omni-campaign/scratch/d2prep` |
| 0.03 | scratch | 1 | y |  | full | `/mnt/lexar/omni-campaign/scratch/toolscreen-review-20261015` |
| 0.02 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/source-composed-validation-820e29e1-a97a-4e02-9268-a1de1ce73a44` |
| 0.02 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/composed-validation-f0773c8d-d8db-4277-bef2-33d3f586bf1b` |
| 0.02 | scratch | 2 | y |  | full | `/mnt/lexar/omni-campaign/scratch/refill-20261015` |
| 0.01 | scratch | 2 | y |  | full | `/mnt/lexar/omni-campaign/scratch/qtext-gpu0-2026-10-13` |
| 0.0 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/codex-p40-native-baseline-bc02df7c44fd` |
| 0.0 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/codex-p40-native-tiny-2dae057a5f9e` |
| 0.0 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/codex-p40-native-tiny-a996e4738380` |
| 0.0 | media-unit | 1 | y |  | full | `/mnt/lexar/omni-campaign/codex-p40-native-tiny-095af76aea97` |
| 0.0 | scratch | 1 |  |  | full | `/mnt/lexar/omni-campaign/scratch/p2-v133-gpu0` |

### Workstation

**Neural Forge children (7, 1.0 GiB, C:).** `git rm` in `C:/Users/cereal/Documents/ai training/neural-forge`, one commit. The bytes stay in the repo history. One omni_lab row each (git blob sha1 + sha256).

- 0.523 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\smolvlm-real-fold-20260804\L3-prefix\child`
- 0.097 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260804-pretrained\L6-prefix-random-control\child`
- 0.097 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260803\L6-prefix-fold\child`
- 0.075 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260804-pretrained\L3-prefix-random-control\child`
- 0.075 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260804-pretrained\L3-prefix-pretrained\child`
- 0.068 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260803\L2-full-rank-parent-control\child`
- 0.068 GiB `C:\Users\cereal\Documents\ai training\neural-forge\reports\cross-family-real-20260803\L2-exact-subspace-parent\child`

**Worktree duplicates (82, 82.19 GiB).** Each carries the same 7 fold-child blobs as neural-forge HEAD (all 84 copies byte-verified). 68 are on C: under `Documents\ai training\.agent-worktrees\neural-forge-a495ffdd2f` (68.16 GiB). 14 are on D: under `D:\codex-owned-source-repositories\k3-window61-72-9d98292e804f\worktrees` (14.03 GiB). 78 are clean and get `git worktree remove` (their branches stay). These 4 have untracked files and go to Codex on Relay:

- `C:\Users\cereal\Documents\ai training\.agent-worktrees\neural-forge-a495ffdd2f\workers\codex\hy3d-parent-capture-bridge-3ff6b89b17a1` (untracked 2, branch `codex/work/codex/hy3d-parent-capture-bridge-3ff6b89b17a1`)
- `C:\Users\cereal\Documents\ai training\.agent-worktrees\neural-forge-a495ffdd2f\workers\codex\chimera-trainer-readiness-oct13-3e25f2da3b32` (untracked 1, branch `codex/work/codex/chimera-trainer-readiness-oct13-3e25f2da3b32`)
- `C:\Users\cereal\Documents\ai training\.agent-worktrees\neural-forge-a495ffdd2f\workers\codex\chimera-memory-corpus-oct13-9c19be622841` (untracked 3, branch `codex/work/codex/chimera-memory-corpus-oct13-9c19be622841`)
- `C:\Users\cereal\Documents\ai training\.agent-worktrees\neural-forge-a495ffdd2f\workers\codex\chimera-image-edit-d4-oct13-3e25f2da3b32` (untracked 1, branch `codex/work/codex/chimera-image-edit-d4-oct13-3e25f2da3b32`)

The full list of the 82 is in `cleanup_plan.json` -> `workstation_delete`.

## 10. HOLD and UNSURE (not deleted)

### HOLD (would be DELETE; needs a ruling)

| GB | path | release condition |
|---:|---|---|
| 52.32 | `/mnt/lexar/omni-campaign/distill/fullkd/run1/safety` | Resume point (ckpt-00000611: weights + optimizer/trainer state) of the KEEP-best run-1 student; inventory shows its first shard hashes differently from out/student, so it is NOT a duplicate. Touched 45 h ago. Deleting it ends any chance of resuming run 1. Needs an explicit owner/lead ruling that run 1 will never be resumed (v2 is a new 2B/4B model, so the expected ruling is delete). |
| 10.23 | `/mnt/lexar/omni-campaign/distill/media_m1/run1/out/checkpoints` | M1 intermediate checkpoints (step-0000294/0000300 media.safetensors + optimizer.pt) beside the kept M1 export; touched 55 h ago. Same question as run1/safety: resume point vs stale. Delete after the ruling. |
| 6.39 | `/mnt/lexar/omni-campaign/research/fold/x3-08b` | Named in LANES.json do_not_touch of L-RESEARCH (lane parked, not ended). The keep rules forbid deleting anything referenced by an active config; release the lane (lanes.py) first, then it moves to DELETE. |

### UNSURE, server (lead recommendation; owner decides)

Group sizes: NAS duplicates of parents (Kimi K3 1,561 GB, Flash Strata 88.9, 27B-U 55.6, Qwen3.5 4B/0.8B) and dead omnicoder trees (541.2 GB NAS, 22.7 GB Lexar, 15.8 GB home in this list) dominate.

| GB | path | recommendation |
|---:|---|---|
| 1561.0 | `/mnt/ai_data/models/moonshotai/Kimi-K3` | KEEP as the only second copy of a 1.56 TB parent (Lexar is one USB SSD). Revisit only if NAS space is still short after wave 1 frees ~1.04 TB there. |
| 198.16 | `/mnt/ai_data/omnicoder_2026/weights_archive` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 190.04 | `/mnt/ai_data/models/film2/hunyuan` | film2 Hunyuan (190 GB, Feb 2026) - possibly the Gauntlet film pipeline; owner decides. |
| 141.86 | `/mnt/ai_data/omnicoder_2026/local_root_legacy_payloads` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 88.88 | `/mnt/ai_data/models/flash-next-strata/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata` | Backup copy of a KEEP parent on a second device; recommend KEEP (cheap insurance). |
| 55.59 | `/mnt/ai_data/models/Qwen/Qwen3.8-27B` | Base of the 27B-Uncensored parent; re-downloadable; recommend DELETE after recording unless a base-vs-U teacher comparison is planned. |
| 55.58 | `/mnt/ai_data/models/JonathanColetti/Qwen3.8-27B-Uncensored` | Backup copy of a KEEP parent on a second device; recommend KEEP (cheap insurance). |
| 55.29 | `/mnt/ai_data/omnicoder_2026/checkpoint_archives` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 52.45 | `/mnt/ai_data/omnicoder_2026/training_runs` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 44.04 | `/mnt/ai_data/omnicoder_2026/checkpoints` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 39.07 | `/mnt/ai_data/omnicoder_2026/root_pressure_archive_20260604T0438Z` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 19.33 | `/mnt/ai_data/models/Qwen/Qwen3.5-9B` | Not on the keep list and no v2 role named; recommend DELETE after recording (re-downloadable). |
| 16.81 | `/home/cereal/models/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | Q4 quant of a parent, rebuildable with llama.cpp quantize from the kept parent; recommend DELETE after recording (frees space on / at 97%). |
| 16.55 | `/home/cereal/models/gguf/Qwen3.6-27B-Q4_K_M.gguf` | Q4 quant of a parent, rebuildable with llama.cpp quantize from the kept parent; recommend DELETE after recording (frees space on / at 97%). |
| 15.79 | `/home/cereal/omnicoder_2026_work/weights` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 13.01 | `/mnt/ai_data/models/comfyui/checkpoints` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 11.78 | `/mnt/lexar/models/ibm-granite/granite-speech-3.3-2b` | Speech model not named on the keep list; recommend KEEP until the v2 speech teacher set is fixed. |
| 9.75 | `/mnt/lexar/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Speech model not named on the keep list; recommend KEEP until the v2 speech teacher set is fixed. |
| 9.34 | `/mnt/ai_data/models/Qwen/Qwen3.5-4B` | Backup copy of a KEEP parent on a second device; recommend KEEP (cheap insurance). |
| 8.61 | `/home/cereal/omni-campaign/runtime/speech-src-01a114aa-9772e4d5-d09b-4c54-9650-699586a9529d` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 8.06 | `/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 7.67 | `/home/cereal/omni-campaign/runtime/speech-native-01a11531-c0bf-74f0-9ec5-d7d7072790b8-880527b5-9d0f-4ba4-a8e0-f83aef34123e` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 5.8 | `/home/cereal/model-surgery-lab/models/NEO1_5-2B-SFT-materialized-20260710` | Neural Forge parent/materialized model (LFM2.5-230M / NEO1.5-2B). Parents are never deleted; recommend KEEP (small). |
| 5.8 | `/mnt/ai_data/hf_cache/models--Paranioar--NEO1_5-2B-SFT` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 5.56 | `/mnt/lexar/omnicoder_0.1.0/models/hf/hub/models--microsoft--phi-2` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 5.56 | `/mnt/lexar/omnicoder_0.1.0/models/hf/models--microsoft--phi-2` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 5.48 | `/mnt/lexar/omnicoder_0.1.0/models/hf/hub/models--runwayml--stable-diffusion-v1-5` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 5.17 | `/mnt/ai_data/omnicoder_2026/weights` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 5.07 | `/home/cereal/omni-campaign/runtime/ace24-native-cpu-01a11929-c8eb6b59-2975-4f9f-9c5b-8c072bea397c` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 4.96 | `/mnt/lexar/models/openbmb/VoxCPM2` | Speech model not named on the keep list; recommend KEEP until the v2 speech teacher set is fixed. |
| 4.6 | `/mnt/lexar/omnicoder_0.1.0/weights` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 4.6 | `/mnt/ai_data/backups/omnicoder_0.1.0-evidence/weights` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 3.64 | `/home/cereal/omni-campaign/runtime/ace24-native-cpu-01a11929-c7bbf8ab-a1e3-49fe-973a-4601e59dae21` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 3.51 | `/home/cereal/omni-campaign/runtime/ace-oct06-native-492961ec-82c3-4bed-98ce-cb6c01af973e` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 3.34 | `/mnt/ai_data/models/comfyui/animatediff_models` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 3.11 | `/home/cereal/lexar-relief-local/omni-campaign/distill` | scorer-parity 0.8B GGUF next to the live flash queue's parity receipts; keep while the queue runs. |
| 2.89 | `/mnt/ai_data/models/comfyui/controlnet` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 2.62 | `/home/cereal/omni-campaign/export/gguf` | Not clearly in a category; keep until the owner rules. |
| 2.53 | `/mnt/ai_data/models/comfyui/ipadapter` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 2.46 | `/home/cereal/omni-campaign/media/out` | Not clearly in a category; keep until the owner rules. |
| 2.07 | `/home/cereal/omni-campaign/media/pairs` | Not clearly in a category; keep until the owner rules. |
| 1.77 | `/mnt/ai_data/models/Qwen/Qwen3.5-0.8B` | Backup copy of a KEEP parent on a second device; recommend KEEP (cheap insurance). |
| 1.62 | `/home/cereal/comfyui/models/skintoken/experiments` | Not clearly in a category; keep until the owner rules. |
| 1.53 | `/home/cereal/embeddinggemma2-preprod-20261006-task3/assets` | Not clearly in a category; keep until the owner rules. |
| 1.52 | `/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 1.38 | `/home/cereal/omni-campaign/runtime/k3-nas-window61-72-oct07-a151e1a8-39b6-486e-a4aa-124865332425` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 1.24 | `/home/cereal/omni-campaign/runtime/qvl32-cpu-01a10f94-29cc8a2c-2fc0-47b4-b98c-7cdd392d09be` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 1.2 | `/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 1.03 | `/home/cereal/omni-campaign/runtime/h3_joint50-capture-native-cpu-01a11929-8f899bc7-cce0-4519-a3d9-c4d970abd998` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 1.03 | `/home/cereal/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 1.02 | `/home/cereal/omni-campaign/runtime/qwen333-native-cpu-01a10f94-136a6909-c556-4146-8fe3-a89e0e3e0f3f` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.83 | `/home/cereal/omni-campaign/runtime/qvl32-1075-cpu-01a10f94-2b148ef9-817e-464f-b28c-746319b5ac77` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.81 | `/mnt/lexar/omnicoder_0.1.0/models--google--siglip-base-patch16-224` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 0.8 | `/mnt/ai_data/neural-forge-stage2/corpus` | Neural Forge stage-2 corpus (0.8 GB). Corpora are KEEP by rule even when the experiment is dead; owner decides. |
| 0.69 | `/home/cereal/omni-campaign/runtime/k3-nas-window61-72-oct07-def64e2b-a609-495d-846e-4167d125d99e` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.68 | `/home/cereal/omni-campaign/runtime/ace-audio-vae-cpu-01a10f94-5bafc3da-32a4-4e83-894a-7e293616823e` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.62 | `/home/cereal/omni-campaign/runtime/h3_audio-native-cpu-01a10f94-3181d72e-f9a3-48cc-a8bd-ea17bf2a7a5f` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.61 | `/mnt/lexar/omnicoder_0.1.0/models/hf/hub/models--timm--vit_base_patch32_clip_224.openai` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 0.59 | `/home/cereal/omni-campaign/runtime/k3-nas-window61-72-oct07-2ceb7b30-6b44-436e-b161-bac8b7f29caa` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.58 | `/home/cereal/omni-campaign/runtime/qwen333-fixed1075-cpu-01a10f94-27588780-c763-4489-89cf-1f296ae1cc89` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.55 | `/home/cereal/.cache/huggingface/hub/models--openai-community--gpt2` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 0.46 | `/home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 0.46 | `/home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M-Base` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 0.46 | `/home/cereal/model-surgery-lab/models/LFM2.5-230M` | Neural Forge parent/materialized model (LFM2.5-230M / NEO1.5-2B). Parents are never deleted; recommend KEEP (small). |
| 0.43 | `/mnt/ai_data/models/comfyui/insightface` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 0.43 | `/mnt/ai_data/omnicoder_2026/benchmarks` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 0.38 | `/home/cereal/.cache/huggingface/hub/models--EleutherAI--pythia-160m` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 0.36 | `/mnt/lexar/models/hexgrad/Kokoro-82M` | Speech model not named on the keep list; recommend KEEP until the v2 speech teacher set is fixed. |
| 0.27 | `/home/cereal/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M` | Small public base models (possible v2 proxies); recommend KEEP (small). |
| 0.12 | `/mnt/ai_data/omnicoder_2026/benchmarks_2026` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 0.1 | `/home/cereal/omni-campaign/runtime/h3_text-native-cpu-01a10f94-1c8778bb-7ea3-43ec-9f89-a2ef517b9adb` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.1 | `/mnt/lexar/omnicoder_0.1.0/inception-2015-12-05.pt` | Dead omnicoder project (pre-campaign). Not in any keep/delete category; recommend owner approves DELETE after recipes+receipts are archived. |
| 0.05 | `/home/cereal/chimera-proof-c5a5c18-v4/proof_outputs` | Not clearly in a category; keep until the owner rules. |
| 0.04 | `/mnt/lexar/omni-campaign/distill/media_m1/codex-image-edit-v1-extracted` | Not clearly in a category; keep until the owner rules. |
| 0.03 | `/home/cereal/omni-campaign/composite/corpus` | Named 'corpus' (superseded k3/qwen input_ids; 4.7 MB of weight-format files, 0.03 GB du). Corpora are KEEP by rule; no space to gain. Keep. |
| 0.03 | `/mnt/ai_data/neural-forge-stage2/calib` | Neural Forge stage-2 calibration data (tiny). Data, not weights; keep with the stage-2 corpus decision. |
| 0.02 | `/mnt/ai_data/backups/ice-cleanup-20261001-selftest/tar` | ICE game backup: out of scope, leave. |
| 0.02 | `/mnt/ai_data/backups/ice-cleanup-20261001-selftest/d` | ICE game backup: out of scope, leave. |
| 0.01 | `/home/cereal/omni-campaign/runtime/disk-backup-exact2x-20261007T0418-approved-54c54ee2` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.0 | `/mnt/ai_data/models/comfyui/upscale_models` | Legacy ComfyUI SD-era models on the NAS (not the production ComfyUI tree); owner decides. |
| 0.0 | `/home/cereal/omni-campaign/distill/p11` | Not clearly in a category; keep until the owner rules. |
| 0.0 | `/home/cereal/omni-campaign/runtime/h3-cpu-01a10f94-19c8cec0-dbd0-4fee-8066-7b63f33fa48b` | Codex runtime outputs: NEVER-delete class (Codex runtime dirs). Keep; ask Codex on Relay if space is needed. |
| 0.0 | `/mnt/ai_data/backups/ice-agent-simulator-git-20261001/dotgit-cursor` | ICE game backup: out of scope, leave. |

Container: the exited container `comfyui` (image `comfyui:codex-3d-hunyuan-skintoken-20260618`, exited 2026-09-01) holds a 22.4 GB ACE-Step captioner HF cache in its writable layer. It is UNSURE. Recommendation: owner decides; if the captioner is wanted as a teacher, copy the cache out first. The 92 exited or created `codex-*` containers hold no weights (0.03 GB writable), so they are outside this order (a docker prune can come later).

### UNSURE, workstation

| GiB | path | recommendation |
|---:|---|---|
| 2.879 | `C:\Users\cereal\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 2.381 | `D:\SpriteBuild\converted-model-candidates\d47e7eddb8b01dbd20dde269c174b77cb8bbdbb7fa1983e0ea1b0d474be507ec` | Verified sha256 duplicate of models\derived\aef54865...; delete after the 72 h Sprite window passes (after 2026-10-22T21:56-06:00) if Sprite does not reference it. |
| 1.627 | `C:\Users\cereal\.cache\huggingface\hub\models--desklib--ai-text-detector-v1.01` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 1.625 | `C:\Users\cereal\.cache\huggingface\hub\models--danibor--oculus-v2.0-multilingual` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 1.325 | `C:\Users\cereal\Documents\ai training\.validation-cache\composite-sep07\qimg\source-qimg-sep07.tar` | Evidence tar for the Sep-07 composite run; recommend KEEP (receipt). |
| 0.575 | `C:\Users\cereal\.cache\huggingface\hub\models--laion--clap-htsat-fused` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 0.523 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\smolvlm-real-fold-20260804\L3-prefix\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.453 | `C:\Users\cereal\.cache\huggingface\hub\models--Systran--faster-whisper-small` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 0.45 | `D:\CodexData\workspaces\2026-08-05\go-download-this-video-and-all\artifacts\h3\reviews\official20-vs-turbo8-blind-sanitized-v2\reviewer-scratch-quarantine-20260809T0755Z\.reviewer2-asr-cache\hub\models--Systran--faster-whisper-small.en\snapshots\d1d751a5f8271d482d14ca55d9e2deeebbae577f` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.323 | `C:\Users\cereal\.cache\huggingface\hub\models--MIT--ast-finetuned-audioset-10-10-0.4593` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 0.303 | `D:\SpriteBuild\models\speech\kokoro-multi-lang-v1_0-fp32-modelonly` | Sprite speech model only in SHA256SUMS.txt; Sprite owner (Claude, Sprite lead) decides; recommend KEEP (0.5 GiB total). |
| 0.243 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\native_executor\qimg_parallel\results_v2\native_subset\replay` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.238 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\native_executor\refold\output\fold` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.106 | `D:\SpriteBuild\models\speech\kokoro-int8-multi-lang-v1_0` | Sprite speech model only in SHA256SUMS.txt; Sprite owner (Claude, Sprite lead) decides; recommend KEEP (0.5 GiB total). |
| 0.097 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260804-pretrained\L6-prefix-random-control\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.097 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260803\L6-prefix-fold\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.08 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\native_executor\qimg_parallel\results_v2\one_held_block30_v1` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.075 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260804-pretrained\L3-prefix-random-control\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.075 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260804-pretrained\L3-prefix-pretrained\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.073 | `C:\Users\cereal\.cache\huggingface\hub\models--Systran--faster-whisper-tiny.en` | Public HF cache, re-downloadable; not campaign weights; recommend leave (C: is full, but these total ~7.5 GiB and may back tools). |
| 0.068 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260803\L2-full-rank-parent-control\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.068 | `C:\Users\cereal\Documents\Codex\2026-10-05\task-4\owner_fold_policy_oct06\worktree\reports\cross-family-real-20260803\L2-exact-subspace-parent\child` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |
| 0.065 | `D:\SpriteBuild\models\speech\sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Sprite speech model only in SHA256SUMS.txt; Sprite owner (Claude, Sprite lead) decides; recommend KEEP (0.5 GiB total). |
| 17.85 | `C:\Users\cereal\Documents\ai training\.git` | Codex workspace / Codex refs: never-delete class; ask Codex on Relay. |

## 11. KEEP list

### Server

| GB | category | path | reason |
|---:|---|---|---|
| 1561.0 | KEEP-parent | `/mnt/lexar/models/moonshotai/Kimi-K3` | Kimi K3 parent/teacher (K3 primary) |
| 88.88 | KEEP-parent | `/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata` | Flash Next Strata teacher GGUF; open by live teacher-score/gguf_teacher_cache (flash_score_queue.sh) |
| 65.45 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qvl32_vit/qvl32_vit_l13` | teacher cache |
| 55.58 | KEEP-parent | `/home/cereal/models/Qwen3.8-27B-Uncensored` | Qwen3.8-27B-Uncensored parent |
| 55.07 | KEEP-best | `/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1` | OMNI1 composite (keep exactly one copy) |
| 52.28 | KEEP-best | `/mnt/lexar/omni-campaign/distill/fullkd/run1/out` | full-KD run-1 student (keep exactly one copy) |
| 51.51 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 46.15 | KEEP-parent | `/home/cereal/comfyui/models/checkpoints/ltx-2.3-22b-distilled.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 40.23 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 24.41 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 21.04 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/hy3d/hy3d_trajectories` | teacher cache |
| 20.97 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 20.97 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 20.43 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 20.43 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 19.75 | KEEP-parent | `/mnt/ai_data/models/lmstudio/unsloth` | LM Studio archive (NAS) |
| 17.48 | KEEP-parent | `/mnt/ai_data/models/lmstudio/lmstudio-community` | LM Studio archive (NAS) |
| 15.69 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 14.23 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/qwen_image_2.1_bf16.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 9.45 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 9.38 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 9.35 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 9.34 | KEEP-parent | `/home/cereal/models/Qwen3.5-4B` | Qwen3.5 small base (v2 init / proxy, OC01 ladder parent) |
| 9.12 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qimg/qimg-te-qwen_2.5_vl_7b_fp8_scaled` | teacher cache |
| 7.61 | KEEP-parent | `/home/cereal/comfyui/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 7.37 | KEEP-parent | `/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 7.26 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 5.21 | KEEP-parent | `/home/cereal/comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 5.11 | KEEP-best | `/mnt/lexar/omni-campaign/distill/media_m1/run1/out/export` | media M1 export (media_modules.safetensors) |
| 4.87 | KEEP-parent | `/mnt/lexar/models/ibm-granite/granite-speech-4.1-2b` | granite-speech-4.1 speech teacher |
| 4.79 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/acestep-v15-turbo` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 4.79 | KEEP-parent | `/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 4.7 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-ASR-1.7B` | Qwen3 ASR/TTS speech teacher |
| 4.54 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen3 ASR/TTS speech teacher |
| 4.52 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Qwen3 ASR/TTS speech teacher |
| 4.52 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Qwen3 ASR/TTS speech teacher |
| 4.51 | KEEP-data | `/home/cereal/omni-campaign/distill/corpus` | distill corpus |
| 3.76 | KEEP-parent | `/home/cereal/comfyui/models/ACE-Step/acestep-5Hz-lm-1.7B` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 3.71 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen_1.7b_ace15.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 3.54 | KEEP-data | `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/speech-native-caches-v2` | adopted Codex speech/media caches (L-SPEECH-DATA) |
| 2.69 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qimg/qimg-velocity-qwen_image_fp8_e4m3fn` | teacher cache |
| 2.5 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Qwen3 ASR/TTS speech teacher |
| 2.39 | KEEP-data | `/home/cereal/omni-campaign/export/llamacpp` | llama.cpp build used by the live teacher-score process (not model weights) - NEVER delete |
| 2.39 | KEEP-data | `/home/cereal/omni-campaign/export/llamacpp-imatrix` | llama.cpp build used by the live teacher-score process (not model weights) - NEVER delete |
| 2.05 | KEEP-parent | `/mnt/ai_data/models/lmstudio/DevQuasar` | LM Studio archive (NAS) |
| 1.9 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/ace/ace_dit_velocity` | teacher cache |
| 1.88 | KEEP-parent | `/mnt/lexar/models/Qwen/Qwen3-ASR-0.6B` | Qwen3 ASR/TTS speech teacher |
| 1.77 | KEEP-parent | `/home/cereal/models/Qwen3.5-0.8B` | Qwen3.5 small base (v2 init / proxy, OC01 ladder parent) |
| 1.19 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/qwen_0.6b_ace15.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 1.19 | KEEP-parent | `/home/cereal/comfyui/models/text_encoders/Qwen3-Embedding-0.6B` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 1.1 | KEEP-parent | `/mnt/ai_data/models/lmstudio/JonathanColetti` | LM Studio archive (NAS) |
| 1.09 | KEEP-parent | `/home/cereal/comfyui/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 1.0 | KEEP-parent | `/home/cereal/comfyui/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 1.0 | KEEP-parent | `/home/cereal/comfyui/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.8 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/k3/eb35851b2496a628-provisional-bf16` | teacher cache |
| 0.78 | KEEP-parent | `/home/cereal/comfyui/models/loras/h3` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.68 | KEEP-parent | `/home/cereal/comfyui/models/vae/qwen_image_2.1_vae_bf16.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.66 | KEEP-parent | `/home/cereal/comfyui/models/vae/Hunyuan3D-vae-v2-1-fp16.ckpt` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.61 | KEEP-parent | `/home/cereal/comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.54 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qimg/qimg-vae` | teacher cache |
| 0.35 | KEEP-live | `/home/cereal/omni-campaign/import/rq27b` | flash_score_queue.sh (pid 4009138, running) exports CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src; the inventory missed this reference because the queue uses $OC-relative paths. Never delete while the queue runs. |
| 0.34 | KEEP-parent | `/home/cereal/comfyui/models/vae/ace_1.5_vae.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.34 | KEEP-parent | `/home/cereal/comfyui/models/vae/vae` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.3 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qvl32_vit_ref/qvl32_vit_l13` | teacher cache |
| 0.26 | KEEP-parent | `/home/cereal/comfyui/models/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.25 | KEEP-parent | `/home/cereal/comfyui/models/vae/qwen_image_vae.safetensors` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.2 | KEEP-data | `/home/cereal/omni-campaign/distill/toolcall` | tool-call corpus/caches (modified < 72 h) |
| 0.16 | KEEP-data | `/mnt/lexar/omni-campaign/corpus/holdouts` | corpus / data registry |
| 0.13 | KEEP-data | `/mnt/lexar/omni-campaign/research/data/quality-v1.0.1-registry-2026-10-15` | corpus / data registry |
| 0.13 | KEEP-data | `/mnt/lexar/omni-campaign/research/data/quality-v1-registry-2026-10-15` | corpus / data registry |
| 0.13 | KEEP-data | `/mnt/lexar/omni-campaign/research/data/quality-v1.0.3-registry-2026-10-15` | corpus / data registry |
| 0.13 | KEEP-data | `/mnt/lexar/omni-campaign/research/data/quality-v1.0.4-registry-2026-10-15` | corpus / data registry |
| 0.1 | KEEP-data | `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/h3-bf16-velocity-v1` | adopted Codex speech/media caches (L-SPEECH-DATA) |
| 0.06 | KEEP-data | `/mnt/lexar/omni-campaign/teacher_cache/qtext/inputs` | teacher cache |
| 0.04 | KEEP-data | `/home/cereal/omni-campaign/distill/queues` | live queue dir (flash_score_queue.sh, gguf_teacher_cache) - NEVER delete |
| 0.03 | KEEP-data | `/home/cereal/omni-campaign/receipts/codex-seven-native-01a11bd5-20261014-v28` | receipts |
| 0.02 | KEEP-data | `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/h3-bf16-text-v1` | adopted Codex speech/media caches (L-SPEECH-DATA) |
| 0.02 | KEEP-parent | `/home/cereal/comfyui/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.0 | KEEP-parent | `/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo` | ComfyUI production model (Qwen-Image 2.1 / MiniMax H3 / LTX / ACE-Step / Hunyuan3D / Qwen3-VL-32B text encoder); GPU-serving path |
| 0.0 | KEEP-data | `/home/cereal/omni-campaign/receipts/codex-native-qimg-01a11bd5-20261013-v2` | receipts |
| 0.0 | KEEP-data | `/mnt/lexar/omni-campaign/codex-adopted-2026-10-14/tool-train-168-corpus` | adopted Codex speech/media caches (L-SPEECH-DATA) |

The keep rules also name the following. The server inventory has no separate unit for them beyond the ComfyUI files above:
- a full Qwen3-VL-32B HF directory (only the ComfyUI text encoders were found);
- a separate Hunyuan3D 2.1 tree (only the ComfyUI checkpoint and VAE were found);
- Qwen3.5-2B (only 0.8B and 4B under `~/models`, plus 0.8B, 4B and 9B on the NAS).

Nothing in this plan deletes any of these. Check whether v2 needs a 2B base download.

### Workstation

| category | items | GiB |
|---|---:|---:|
| KEEP_DATA | 5 | 0.3 |
| KEEP_PARENT_OR_TEACHER | 3 | 4.66 |
| KEEP_REBUILD_SOURCE | 2 | 0.0 |
| KEEP_SPRITE_ACTIVE | 30 | 57.06 |
| KEEP_SYSTEM | 4 | 1.05 |
| NOT_WEIGHTS | 45 | 3.31 |

- KEEP_SPRITE_ACTIVE: everything under `D:\SpriteBuild` that is in `model_catalog.json`, referenced, or touched in the last 72 h. `b6d8afa5...` is kept but flagged unqualified by its own receipt.
- KEEP_PARENT_OR_TEACHER: acestep-5Hz-lm-1.7B, SmolVLM2-256M-Video-Instruct and SmolLM2-135M under `Documents\ai training\models`.
- KEEP_REBUILD_SOURCE: neural-forge `.git` and the k3-window61-72 repository `.git`.
- KEEP_SYSTEM: the NVIDIA, whisper and chroma caches. NOT_WEIGHTS: Flutter/Gradle/Unreal `.bin` files.

## 12. Evidence

- Server inventory and scan data: `cleanup_server.json` (scratchpad `lab/`); raw data on the server in `/home/cereal/claude-scratch/cleanup-inv-20261021/`.
- Live references: `lab/cinv/live_refs.txt`. The `CHIMERA_OMNI_SRC=$OC/import/rq27b/...` line is in the `flash_score_queue.sh` section.
- Workstation inventory: `C:/Users/cereal/Documents/ai training/omni-lab/cleanup_workstation.json`; raw data in the session scratchpad.
- Plan generators: `lab/cinv/build_plan.py` and `lab/cinv/build_plan_md.py` (scratchpad).
- `oc_checkpoint.py` R4/R6: `neural-forge/reports/omni-campaign-2026-09/ledger/oc_checkpoint.py` (`_delete_conditions`, `PROTECTED_PREFIXES`, `STATUSES`).
- Status overwrite: `omni-lab/omni_lab_db.py` `upsert_model` (`status = EXCLUDED.status`), `ingest/model_scan.py`:199/240, `ingest/oc_ledger.py`:137.
- Backup scope: `C:/Users/cereal/.agents/skills/agent-memory-backup/scripts/backup_agent_memory_pg.sh`:260/279.
