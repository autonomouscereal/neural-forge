# NAS cleanup 2026-10-28, round 2

Round 1: NAS_CLEANUP_2026-10-27.md. Sections are appended by each workflow.

## Round 3: D4 Ollama

Owner approval (Cyril 2026-10-28, verbatim): "[Cyril quote withheld]". It is recorded per unit as a `protected_waivers` entry for the `/mnt/ai_data/models` root guard, in
`round3-D4/plan_nas_20261028_D4_round3.json`. That plan is the round-1 D4 units unchanged except wave, waiver and
rebuild text.

- **Wave** N3-NAS-20261028, tool `nas_group_delete 1.0.0` (unchanged, same sha256 as commit 43a4587). Run 2026-10-28 20:00:54-20:01:28Z.
- **Pre-checks (19:55Z):** the ollama unit is inactive, there is no ollama binary, process or container, and no container mounts the stores. Config, cron and omni_lab references: 0. Inbound symlinks: 0. The dry run cleared all 3 units.
- **omni_lab dump before rows:** `/mnt/lexar/omni-campaign/backups/omni_lab/dumps/omni_lab_20261028T200038Z.dump`
  (sha256 5e360143aa14dbf9..., restore-tested, NAS copy verified).

| model_deletion | unit | files | bytes | recipe |
|---|---|---|---|---|
| 223 | `/mnt/ai_data/models/ollama/deepseek/models` | 6 | 19,851,339,037 | `archive/recipes/path-c6f26f9241549ae0__nas20261028-ollama-deepseek` |
| 224 | `/mnt/ai_data/models/ollama/glm/models` | 6 | 20,652,491,852 | `archive/recipes/path-140a64ad0f404975__nas20261028-ollama-glm` |
| 225 | `/mnt/ai_data/models/ollama/qwen/models` | 12 | 38,152,094,825 | `archive/recipes/path-41c8794a752d0315__nas20261028-ollama-qwen` |
| **total** | | 24 | **78,655,925,714** | |

Affected model rows: 2573, 2574 and 2575, all now `deleted`.

- **NAS free space** (`df -B1 /mnt/ai_data`): 603,195,441,152 B (88%) before, then 681,808,232,448 B (86%) after, a gain of 78,612,791,296 B.
- **verify_deleted.py:** `verified ok 155 / 155 server model_deletion rows; problems 0`.
- **Kept:** the three `id_ed25519` / `id_ed25519.pub` pairs (6 of 6 present after the run, not read).
- **Rebuild:** each recipe holds MANIFEST.tsv (blob digests), REBUILD.md (`ollama pull` lines and, for GLM, the
  `hf download unsloth/GLM-4.6V-Flash-GGUF` line), DELETION_RECEIPT.json and tree.tar.gz (manifests and small blobs). Lexar
  `/mnt/lexar/omni-campaign/archive/recipes/`, mirrored byte-identical (sha256) to
  `/mnt/ai_data/backups/nas-cleanup-20261028/recipes/`; run dir mirrored to `.../manifests/round3-D4/`.

## Round 2: agent-memory backups, U3 Neural Forge caches, U4 Qwen3.5-9B (wave N2-NAS-20261028)

**Owner rulings** (Cyril 2026-10-28, verbatim):
- (3) "[Cyril quote withheld]"
- (4) "[Cyril quote withheld]"

Both are recorded verbatim as `owner_order` and as per-unit `protected_waivers` in `round2-N2/plan_nas_20261028_N2.json`
(sha256 5b5d2a71a014607d...). The preconditions come from checks run the same day: the BACKUP CHECK said "SAFE: YES"
(the 10-22 dump was fully restore-tested) and the GIT CHECK said "PASS".

**Not touched:** D4 (Ollama) and U5 (omnicoder hf_cache).
- The D4 stores had already been deleted at 20:00Z by the separate round-3 workflow above, which quotes a different
  owner approval.
- This round's instructions said D4 was not approved. The two records disagree, so Cyril should confirm which applies.

**Resume check:** the earlier attempt of this step left nothing behind, so nothing was redone.
- There were no N2 omni_lab rows, no N2 recipes and no N2 run dir.
- Every backup was still present, and no `nas_group_delete` process was running.
- No Relay CLAIM or STATUS was found, and none was posted.

**Tool:** `nas_group_delete` **1.1.0** (sha256 17c1ecf444fcaf46...). It is identical on the workstation and in
`/home/cereal/omni-campaign/lab/cleanup/`. The 1.0.0 copy is kept at
`/home/cereal/nas-cleanup-20261028/nas_group_delete.py.lab-1.0.0.bak`. Version 1.1.0 adds three optional fields, all off by
default:
- `unreadable_in_place`: files cereal cannot open are recorded by size and mtime only. Such a file must not be planned,
  and `open()` must really fail on it.
- `volume`: df is taken on the unit's own filesystem, and the unit must sit on that filesystem.
- plan `owner_order` / `reason_prefix`.

Tests are in `tests/test_nas_group_delete.py`. On the AI server all 8 pass (run with `-W error::ResourceWarning`). On the
workstation 5 pass and 3 are skipped because they need POSIX.

**Run:** 2026-10-28 22:46:06-23:15:17Z, systemd user unit `n2-real`.
- Groups ran one at a time, and a non-zero exit stopped the chain.
- Fresh symlink scans ran at 22:21-22:43Z: 8,726 links in home, 4 in the denied dirs, 0 in the NAS trees and `/var/tmp`.
- Dry runs for all 5 groups cleared at 22:44-22:45Z.
- In the real pass, the full gate re-ran immediately before each unit. It checked:
  - open files (root `/proc` scan);
  - references in cmdline, environ, cron, docker and campaign files;
  - inbound symlinks and omni_lab rows;
  - the size and mtime of every planned file.

**omni_lab dumps:**
- Before the rows: `/mnt/lexar/omni-campaign/backups/omni_lab/dumps/omni_lab_20261028T224552Z.dump` (sha256
  02ee4d108ad22783...). The restore test matched exactly (51 tables, 158,853 rows), and the NAS copy was verified.
- After the rows: `omni_lab_20261028T231600Z.dump` (sha256 47b5e094578c811f...). The restore test matched exactly (51
  tables, 158,909 rows), and the NAS copy was verified.

### Deleted

R = `/mnt/lexar/omni-campaign/archive/recipes/`, mirrored to `/mnt/ai_data/backups/nas-cleanup-20261028/recipes/` (an
rsync checksum dry-run showed 0 differences). Each recipe holds MANIFEST.tsv, REBUILD.md, DELETION_RECEIPT.json and
tree.tar.gz.

Paths are relative to `/mnt/ai_data` unless they start with `/`.

| row | unit | mode | files deleted | bytes | kept in place |
|---:|---|---|---:|---:|---:|
| 226-231 | `backups/agent-memory-postgres/incoming/agent_memory_2026072{7T211503Z,7T220103Z,7T223604Z,7T233328Z,8T002947Z,8T010138Z}` (D6) | tree | 6 | 19,164,865,482 | 0 |
| 232 | `.../incoming/agent_memory_20261018T224932Z` (U2) | tree | 3 | 27,410,218,651 | 0 |
| 233 | `.../incoming/agent_memory_20261002T025338Z_retained_retry` (empty) | tree | 0 | 0 | 0 |
| 234 | `.../dumps` (U1 plus the other old nightlies, 09-17..10-21) | files | 37 | 712,263,027,336 | 2 |
| 235 | `.../manifests` | files | 37 | 23,810 | 2 |
| 236 | `.../restore-lists` | files | 37 | 911,245 | 2 |
| 237 | `/var/tmp/agent-memory-backups/agent_memory_20261018T091251Z.PwMwDs.reliability-retained` (root disk) | tree | 3 | 27,393,349,251 | 0 |
| 238 | `/var/tmp/agent-memory-backups/agent_memory_20261018T224932Z.AT3bUz` (root disk) | tree | 3 | 27,410,218,651 | 0 |
| 239 | `/home/cereal/agent-memory-backup/local-safety` (07-27, 07-28 and 10-02 dumps with manifests and lists; root disk) | files | 9 | 22,781,429,218 | 2 |
| 240 | `neural-forge/dynamic_snapshots` (U3) | files | 472 | 1,960,166,784 | 4 |
| 241 | `neural-forge/qwen3_4b_compensated_20260710` (U3) | files | 3 | 24,134,884,104 | 879 |
| 242 | `neural-forge/runs` (U3) | files | 14 | 1,395,267,984 | 40 |
| 243 | `models/Qwen/Qwen3.5-9B` (U4, 4 shards) | files | 4 | 19,306,310,880 | 30 |
| | **NAS total** | | **655** | **805,635,676,276** | |
| | **root-disk total** | | **15** | **77,584,997,120** | |

- **Recipe dirs:** 18 dirs named `R/path-<16hex>__nas20261028-<slug>`. The slugs are `agent-memory-*`,
  `agent-memory-staging-*`, `agent-memory-local-safety`, `nf-*` and `qwen3.5-9b`. The largest is 44 MB (the compensated
  evidence tar).
- **Model rows:** 2576-2593 were created for the units and are now `deleted`. No pre-existing model row named any of
  these paths.
- **Dump metadata:** the manifests and restore lists of every deleted dump are inside the `agent-memory-manifests` and
  `agent-memory-restore-lists` recipe tars.
- **Rebuild:**
  - Agent-memory units: none needed. They are superseded snapshots of the live database, and the restore point is
    listed under Kept.
  - U3: the launch commands and configs are in the kept evidence files (in place and in tree.tar.gz). The code is on
    GitHub (`autonomouscereal/neural-forge`: archive tags, `47145a30`, `backup/20261028/*`), and bundles are in
    `/mnt/ai_data/backups/nf-git-20261028/`.
  - U4: `hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir /mnt/ai_data/models/Qwen/Qwen3.5-9B`.
    The revision comes from the kept HF download metadata, and the shard header sha256s are in MANIFEST.tsv.
- **Backup-tool state:** two files were removed outside the tool. Each was first copied to `round2-N2/state-retired/`
  (and the NAS mirror) and compared:
  - `~/agent-memory-backup/state/retention-hold` (sha256 f27399745d019b34...);
  - the stale pointer `state/pending-stage.recovered-49a541199a9d.txt` (sha256 fd3925d06390c2d7...), which named the
    deleted AT3bUz dir.

  The live pointer `state/pending-stage.txt` (pointing to `X7RcbN`) was not touched. The backup script and cron were not
  changed.

### Kept

- **The safe backup:** `dumps/agent_memory_20261022T082002Z.dump` with its manifest and restore list, plus
  `latest_backup_id.txt`, which still names it. It was re-hashed after all
  deletions (23:20:48-23:38:26Z): sha256 `4849ecac190cb7f9dcb525eda855d9687e5fd140a88f6800813617fd64e367be`, equal to its
  manifest (`round2-N2/safe_dump_sha256_after.txt`).
- **Waiting for tonight's 08:20Z run to publish 10-23:** all three are present and unchanged.
  - `incoming/agent_memory_20261023T082002Z/`, a 16,578,248,704 B partial;
  - `/var/tmp/agent-memory-backups/agent_memory_20261023T082002Z.X7RcbN/`;
  - `state/pending-stage.txt`.
- **Evidence:**
  - Every non-blob file of the U3/U4 units stays in place and is in the recipe tars: 4, 879, 40 and 30 files, counts
    re-checked after the run.
  - The 3 `progressive_stage1_20260709/*/run_progressive_fold.sh` files are untouched; that unit had no deletions this
    round.
  - `local-safety/latest_backup_id.txt` and `local-safety/state/.../latest.json` stay.
- **Symlink notes:** no symlink in the fresh scans resolves into a round-2 unit, so nothing was left dangling. The known
  link `model-surgery-lab/neural-forge-src/runs/residual-attention-20260711` points to
  `neural-forge/qwen3_4b_residual_attention_20260711`, which is not a U3 unit, and it still resolves.

### Skipped, with reasons

- **The 22 foldrec-cited U3 blobs (80,196,246,372 B).** All 22 are still present at their recorded sizes.
  - They are cited by exact path in the `2026-10-22-foldrec` shards: 9 in `gap1-child-deletion-cleanup-records`, 12 in
    `docs-crosswidth-kernel-modality` and 1 in `gap1-surgerylab-july-failed-launches`.
  - Nobody has ruled whether those citations are provenance sources or only records.
  - Because of this hold, 13 of the 16 U3 units had nothing to delete.
  - If the citations are ruled to be records only, rerun `make_plan_n2.py` against `DELETE_ALL.tsv`.
- **The published 10-18T224932Z dump, manifest and list (27,410,218,651 B), and 5 unreadable incoming dirs:**
  `*.recovery-20261019T021903Z-ef32908e934a`, `*.recovery-20261019T024007Z-49a541199a9d` and
  `transport-fixture-ef32908e934a{,-explicit-v2,-owner-v3}`.
  - They are mode 0600/0700 and owned by nobody or root; the dump also has 2 hard links. cereal can neither open nor
    delete them.
  - The 3 files are recorded as `unreadable_in_place` in rows 234-236.
  - Removing them needs root on the ZFS host (192.168.122.170) or sudo on the media server. This round's rules allowed
    SSH to the `ai` server only.
- **D4 (Ollama) and U5 (omnicoder hf_cache):** not approved for this round, so not touched.

### Free space

| | before (22:45:51Z) | after (23:20:32Z) | change |
|---|---:|---:|---:|
| NAS `/mnt/ai_data` | 679,962,738,688 (86%) | 1,482,089,824,256 (70%) | +802,127,085,568 |
| AI root `/` | 114,371,690,496 | 187,598,163,968 | +73,226,473,472 |
| Lexar `/mnt/lexar` | 208,270,262,272 | 203,508,678,656 | -4,761,583,616 |

- **NAS:** it gained 3.5 GB less than the 805.6 GB deleted. That fits ZFS freeing space in the background plus other
  writers.
- **Root disk:** it gained 4.4 GB less than the 77.6 GB deleted, because of other writers.
- **Lexar:** free space fell by 4.8 GB, but the recipes take under 0.1 GB, so most of the drop came from other lanes (run
  0, Strata, intake). The Lexar is still well above the 60 GiB Strata floor.

### Verification

- `verify_deleted.py` 1.1.0 (read-only role) reported **"verified ok 173 / 173 server model_deletion rows; problems 0"**
  (`round2-N2/verify_deleted_round2.txt`).
- **Kept checks** (`round2-N2/check_n2.log`):
  - the safe dump, manifest and list are present with unchanged size and mtime;
  - the 10-23 partial and X7RcbN are present;
  - the held foldrec blobs are 22/22 present;
  - kept-file counts equal the plan;
  - 0 symlinks point into round-2 units.
- **Run dir:** `/mnt/lexar/omni-campaign/archive/cleanup-2026-10-28/round2-N2/` holds the plan, dry and real logs, exec
  jsonl, df snapshots, symlink scans and the state-retired copies. It is mirrored to
  `/mnt/ai_data/backups/nas-cleanup-20261028/manifests/round2-N2/` with 0 differences.
- **Scripts:** `/home/cereal/nas-cleanup-20261028/round2-N2/`.

### Watch items for Cyril and the backup lane

1. **The retention hold is gone, so the nightly prune is now active (`RETENTION_DAYS=30`).** It removes only dumps older
   than 30 days. At about 28 GB a night the share will hold about 30 dumps (~840 GB). That is far fewer revisions than
   before, but still more than "1 safe backup". No keep-newest-N setting was added, and the job is unchanged.
2. **The prune will fail around 2026-11-18.** On that night `find ... -mtime +30 -delete` reaches the undeletable
   10-18T224932Z files.
   - The script runs under `set -Eeuo pipefail` and prunes after it publishes.
   - That run would therefore exit non-zero before `BACKUP_COMPLETED=1` and before it clears `pending-stage.txt`.
   - Remove those 3 files as root before then.
3. **Keep 10-22 until 10-23 is proven.** That means tonight's 08:20Z run publishes 10-23 and it passes the same restore
   proof. The restore must use `--shm-size` >= `maintenance_work_mem` or set `max_parallel_maintenance_workers=0`;
   otherwise both HNSW indexes fail.
