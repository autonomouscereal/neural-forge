## W0: database backups and omni_lab changes (done before any deletion)

**Backups (owner: "[Cyril quote withheld]").**
- New script `backup_pg_db.sh` 1.0.0: skill source `C:/Users/cereal/.agents/skills/agent-memory-backup/scripts/backup_pg_db.sh`, deployed to `/home/cereal/agent-memory-backup/bin/backup_pg_db.sh`. It is a sibling of the agent_memory job, which is unchanged.
  - Raw `pg_dump --format=custom` inside `agent-memory-postgres`, with no credentials on the command line.
  - Each run restores the dump into `<db>_restore_test`, compares exact row counts for every public table, then drops the test database.
  - Retention is 30 days. When the NAS has more than 20 GiB free, the run also mirrors to `/mnt/ai_data/backups/omni-lab-postgres/<db>/` (best effort).
- Cron, AI server: `50 2 * * *`, `run-all` over `omni_lab chimera_corpus` (block `# BEGIN pg-db-backup`). The crontab before the change is saved at `/home/cereal/agent-memory-backup/crontab.before-pg-db-backup-20261021.txt`.
- Pre-cleanup dumps, both with a passing restore test (row counts match):

  | Database | Dump | Size | sha256 | Restore test |
  |---|---|---|---|---|
  | omni_lab | `/mnt/lexar/backups/postgres/omni_lab/dumps/omni_lab_20261021T183825Z.dump` | 4,520,113 B | `f201ce5f872ec975fa0a6b10185d329ba0485cdc206f3a35a888b4fd7732a95b` | 38 tables, 30,771 rows |
  | chimera_corpus | `/mnt/lexar/backups/postgres/chimera_corpus/dumps/chimera_corpus_20261021T183835Z.dump` | 44,629 B | `58a062d31812498930b71354a0ace3523d3847be6a78f2326a63946b425c2132` | 11 tables, 324 rows |

  agent_memory was not dumped again. Its daily job had already produced `agent_memory_20261021T082002Z` (27.6 GB, sha256 `144c0607…`) the same day, and this cleanup does not touch that database.
- Skill: the `agent-memory-backup` SKILL.md has a new section, "Other Databases In The Same Container".

**omni_lab (repo `C:/Users/cereal/Documents/ai training/omni-lab`, commits `1db2dc8` and `5173782`, server copy `$OC/lab` with the same bytes).**
- Migration `0006_deletions` (sha `cda82d37fe70af75`) is applied; schema version 6.
  - `model` gains `deleted_at`, `delete_reason`, `rebuild_recipe_path` and a `deleting` status.
  - New `model_deletion` table: one row per removed directory, holding digests, recipe archive, preflight and df.
  - New `workstation` host row.
- Ingester guard: `omni_lab_db.upsert_model` and `ingest/model_scan.py` never change a `deleting` or `deleted` status. `ingest/oc_ledger.py` goes through `upsert_model`, so it is covered too.
  - Tests: `test_schema` 17 and `test_ingest_models_runs` 08 pass, 33/33 on the server and 18/18 on the workstation.
  - Negative control: test 08 fails against the unpatched code.
- Order per unit: the row is set to `deleting` before the `rm`, and to `deleted` only after the path is verified absent.

## How each unit was removed

Server units ran through `omni-lab/cleanup/cleanup_exec.py`, version 1.0.0 for home and Lexar and 1.2.0 for the NAS (1.1.0 added the tarball and a STOP file; 1.2.0 added `--hash-threads`): three detached workers, one per volume, under `nice`/`ionice -c3`. Each unit was gated again immediately before deletion:
- **Open files:** nothing under the unit open by any process, from a root scan of `/proc` covering fd, maps, cwd and exe.
- **References:** no reference to the unit, to anything inside it, or to a non-generic parent of it. The sources searched were:
  - process cmdlines and environ (read as root)
  - the cereal and root crontabs and `/etc/cron.d`
  - docker mounts
  - `distill/tools/*.sh` and `distill/queues/*.json`
  - Lexar `distill/**/*.json`
  - every JSON in the K3 runtime folder

  `$VARS` were expanded, and matches were checked in both directions.
- **Protected paths:** no keep, hold, unsure or review-excluded path overlaps the unit.
- **Data:** no non-weight file of 50 MB or more; no `corpus`, `holdouts` or `receipts` directory; no teacher-cache manifest.
- **Recency:** nothing modified in the last 24 h.

Validation: this gate refuses all 27 units that the lead review removed from DELETE (`gate_validation27.json`), mostly on references inside them.

Then, per unit:
1. Measure every file. Units under 20 GB get a full sha256. Larger units get sha256 for every file under 1 GB, a safetensors header sha256 for the rest, and a 64 MiB head sha256 for other large files. Parameter counts come from the headers.
2. Pack every non-weight file into `/mnt/lexar/omni-campaign/archive/recipes/<model>/tree.tar.gz`, re-hashing every member against the measured sha256. Write `MANIFEST.tsv` and `DELETION_RECEIPT.json` beside it.
   - The first 129 units (Lexar wave, home wave) were archived by 1.0.0 as loose files under `tree/`. Lexar is exFAT with 1 MiB clusters, so those loose files occupied about 40 GB.
   - `recipe_pack.py` then packed all 129 into `tree.tar.gz`, verified every member against MANIFEST.tsv (0 skips, 0 failures) and removed `tree/`, recording this in `PACKED.txt`. The recipe archive is now 1.2 GB.
3. Record the unit in omni_lab.
4. For ledger-registered checkpoints, append an `oc_checkpoint.py set-status --cyril-ruling` record with the status unchanged.
5. Run `rm -rf --one-file-system` with an argument list, so no shell sees the path. This matters for `$OUT`.
6. Verify the path is absent and record df.

`verify_deleted.py` then re-checked every row independently: path absent, MANIFEST sha, receipt listing sha, recipe files present, and model rows marked `deleted`.

NAS worker restarts. I stopped the NAS worker twice while it was still in the read-only measure step of its first unit, before anything was recorded (`operator-stop` lines in `exec_nas.out`):
- first, to switch the recipe format to the tarball;
- then, to add `--hash-threads 3`. Threads did not raise CIFS throughput: the storage behind the share delivers about 22-25 MB/s, measured from both the AI server and the media host.

A second NAS worker (`exec_nas_small.out`, `--only 70 69 68 67 66`) ran the five smallest NAS units to prove the NAS path end to end early. The main worker skips those units as `absent-already`.

Workstation: `omni-lab/cleanup/ws_cleanup.py` 1.0.0. Codex got a Relay notice first (message `4f2776d5`, thread `52468f63`).
- 62 clean Codex agent worktrees were removed with `git worktree remove` (never `--force`), and their branches were kept.
  - Each one was checked clean with `git status --ignored`.
  - Its 7 fold-child blobs were verified equal to neural-forge HEAD with `git hash-object`.
  - No running process named it.
- The 7 Neural Forge fold-child `model.safetensors` files were `git rm`'d in neural-forge commit `d71f4962` (pathspec-limited; the other 330 uncommitted entries in that shared repo were not touched).
  - The blobs stay in history; restore with `git show 1779b7b9:<path>`.
  - Config and tokenizer files stay in place.
- I used `git worktree remove` instead of `Remove-Item` so git's worktree metadata stays consistent. It deletes exactly the listed path.
- Detached heads. 18 of the removed worktrees were `candidates/*` checkouts on a detached HEAD.
  - 16 of those commits are contained in `codex/validated` or `codex/c17-paired-row-diagnostics-04aee676`.
  - The other 2 (`cc0488822f11`, `c61a2c4dac95`) were in no ref, so they could later have been garbage-collected. I pinned them as `refs/cleanup-2026-10-21/candidates/<id>` in neural-forge and pointed `model_deletion` rows 51 and 57 at those refs.
