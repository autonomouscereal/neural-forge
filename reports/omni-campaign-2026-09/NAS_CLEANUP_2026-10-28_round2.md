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
