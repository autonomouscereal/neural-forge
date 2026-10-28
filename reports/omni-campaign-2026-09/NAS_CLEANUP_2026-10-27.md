# NAS cleanup, 2026-10-27 order (executed 2026-10-28)

Owner order (Cyril, 2026-10-27, relayed by the NAS-cleanup workflow): "[Cyril quote withheld]".
The NAS is `//192.168.50.250/ai`, mounted at `/mnt/ai_data` on the AI server. The media server re-exports it from the
NFS/ZFS dataset `192.168.122.170:/mnt/ai_pool/ai_dataset` (4.5 TB).

## Result

| | Free bytes (`df -B1 /mnt/ai_data`) | Use |
|---|---:|---:|
| Before (2026-10-28 ~11:50Z) | 1,048,576 | 100% |
| After (2026-10-28T12:29:19Z) | **603,220,606,976** | 88% |
| **Freed (df)** | **603,219,558,400 (603.2 GB)** | |

The deleted files total 605,302,541,947 B (apparent size). `df` gained 603.2 GB. The ~2 GB gap is consistent with ZFS
compression and other writers on the share.

Four of the six proposed groups were deleted: D1, D2, D3 and D5, as 13 units with omni_lab `model_deletion` rows 209 and
211-222, all `deleted`. Two groups were **held**: D4 (Ollama stores) and D6 (failed July agent-memory transfers). Both sit
under a `never-delete` entry of the 2026-10-21 protected list. They are listed under UNSURE below.

## Verification

- `cleanup/verify_deleted.py` (1.1.0, read-only role) reported **"verified ok 152 / 152 server model_deletion rows;
  problems 0"**: the 139 earlier rows plus these 13. Output is at
  `/mnt/lexar/omni-campaign/archive/cleanup-2026-10-28/verify_deleted_20261028.txt`.
- **File-subset units (D1, D2):** 0 files over 50 MB remain in any unit. The kept-file counts (36, 14, 21, 983, 144 and
  9) equal the pre-delete recon exactly.
- **Tree units (D3, D5):** all 7 paths are absent.
- **Untouched (confirmed present):**
  - `omnicoder_2026/datasets` and `omnicoder_2026/weights/training_data`;
  - `models/lmstudio` and `models/moonshotai/Kimi-K3`;
  - `training_data/ai-training`;
  - `omni-campaign/k3-run-oct09-claude-archive` and `omni-campaign/s0-probe-sep04`;
  - `backups/omnicoder_0.1.0-evidence` and `backups/agent-memory-postgres/dumps`;
  - `models/film2/{.hf,manifests}`, `models/comfyui/{upscale,VideoHelperSuite}`;
  - the three Ollama `id_ed25519` key pairs (not read).
- **Database backups:**
  - before any row was written: `omni_lab_20261028T112826Z` (13,421,462 B, sha `4972919c750b5f5b...`, every table's
    restore count ok);
  - after the deletions: `omni_lab_20261028T122905Z` (sha `73dc77043049d662...`);
  - both are in `/mnt/lexar/omni-campaign/backups/omni_lab/dumps/`.

## How it was done

The tool is a new `omni-lab/cleanup/nas_group_delete.py` 1.0.0 (sha `0fe9082726b99b77...`, identical on the workstation and
in `/home/cereal/omni-campaign/lab/cleanup/`). It reuses `cleanup_exec.py` 1.3.0's gate primitives and the migration-0006
record flow. `cleanup_exec.py` itself could not be used: it removes whole directories (D1/D2 must keep their evidence in
place), and it blocks any non-weight file of 50 MB or more (the D6 dumps).

**Per unit, in order:**
1. **Gate, re-run immediately before deletion.** Each check below blocks the unit:
   - a protected-list overlap;
   - a file open in any process (root `/proc` scan);
   - a path token in process cmdline/environ, crontabs, docker mounts or the campaign queue/tool/JSON sources;
   - an inbound symlink, from scans of `/home/cereal` (8,675 links, plus the 35 unreadable dirs via sudo) and of the NAS
     trees `omnicoder_2026`, `models`, `omni-campaign` and `neural-forge` (0 links);
   - any omni_lab row in any table naming the path;
   - a planned file missing or changed in size;
   - a new file in a tree unit;
   - anything modified after 2026-10-27T00:00Z.
2. **Measure.** Small files get a full sha256. Safetensors get a header sha256. Other files of 1 GB or more get a sha256
   of their first 64 MiB.
3. **Archive.** Every non-weight file under 50 MiB goes into a recipe tar, plus `MANIFEST.tsv` (with `kept_in_place` and
   `deleted` columns), `REBUILD.md` and `DELETION_RECEIPT.json`.
4. **Record.** One transaction writes the unit model row and the `model_deletion` row (`plan_wave` = `N1-NAS-20261028`),
   both in status `deleting`.
5. **Delete.** D1/D2 unlink only the listed files, never `rm -r`. D3/D5 use `rm -rf --one-file-system`.
6. **Verify.** The tool confirms the deleted files are absent, then sets the rows to `deleted`.

`verify_deleted.py` is now **1.1.0**. It also honours `preflight.kept_in_place`, the evidence files that stay at a deleted
unit's path. The 1.0.0 copy is backed up at `/home/cereal/nas-cleanup-20261028/verify_deleted.py.server-1.0.0.bak`.

**Where everything is:**
- Recipe dirs: `/mnt/lexar/omni-campaign/archive/recipes/path-<16hex>__nas20261028-<slug>/`. They are mirrored to
  `/mnt/ai_data/backups/nas-cleanup-20261028/recipes/` (rsync checksum dry-run: 0 differences).
- Run log, plan, symlink scans, recon listings and df snapshots: `/mnt/lexar/omni-campaign/archive/cleanup-2026-10-28/`,
  mirrored to `/mnt/ai_data/backups/nas-cleanup-20261028/manifests/`.

### Surprises (each stopped the run until it was understood)

1. **Every group overlapped the 2026-10-21 protected list** (`cleanup/inputs/rescope-20261021/protected_rescope.json`).
   No verifier had flagged this.
   - D1/D2 overlap only `plan-UNSURE` entries ("owner decides").
   - D3/D5 overlap their own `plan-UNSURE` entry and also the coarse root guard `never-delete /mnt/ai_data/models`.
   - D4 overlaps only that root guard.
   - D6 overlaps `never-delete /mnt/ai_data/backups/agent-memory-postgres`.

   **Ruling:**
   - **D1, D2, D3 and D5 proceed.** Their most specific classification is `plan-UNSURE`, and the owner has ruled. The rulings
     are in memory: omnicoder 2026-08-19 / 2026-09-03 / Stage D; HunyuanVideo "OUT entirely" 2026-09-01; plus the 10-27
     order. The same list classifies `film2/hunyuan` and the comfyui subdirs as UNSURE inside the models root, which shows
     the root guard protects the parents and was not a ruling on these subtrees.
   - The waivers are recorded per exact protected path, with a reason, in each row's `preflight.protected_waived`. A
     protected entry **inside** a unit can never be waived.
   - **D4 and D6 are held**, because they have no more specific classification than a never-delete entry.
2. **The first unlink failed with `ENOSPC`.** At 1 MiB free, the ZFS dataset behind the share could not record a remove.
   - The tool gained a fallback: on ENOSPC, truncate the file to 0, then unlink. The first truncation in a run must show
     the space back in `df` within 180 s, or the run stops; that guards against snapshot pinning.
   - The truncate itself also *returned* ENOSPC, but it took effect on the server: the file read as 0 B, and ZFS freed
     11.07 GB asynchronously over about a minute. That proves no snapshot pins the blocks.
   - The tool gained a resume path for that case. A `failed` row is resumed against its original pre-deletion record: the
     manifest sha and receipt listing are re-verified, the full gate re-runs, and only a file the record already lists may
     be missing or 0 B. It is never re-measured.
   - The truncated file was `weights_archive/.../checkpoints/.05_image.step288.copying.670730/rank00000.pt`, a partial
     copy and the first file of D1. After about 11 GB was free, plain unlinks worked for everything else.
   - Row 210 is an unused sequence number from the conflicting re-insert.

## Deleted (with rebuild recipe)

R = `/mnt/lexar/omni-campaign/archive/recipes/` (mirror: `/mnt/ai_data/backups/nas-cleanup-20261028/recipes/`)

### D1: omnicoder superseded run blobs (349,013,382,872 B, 56 files)
Only `*.pt` rank shards and partial temp blobs were deleted. All json/jsonl/log/cmd/txt evidence stays in place and is
copied into the recipe tar.

| row | unit | files | bytes | recipe |
|---:|---|---:|---:|---|
| 209 | `omnicoder_2026/weights_archive/training_runs/full_v9_plus_qwen_external_after_text_reusepreflight_20260604T075031Z` | 14 | 198,158,927,909 | `R/path-fe9d228071c9d553__nas20261028-omnicoder-full-v9-weights-archive` |
| 211 | `omnicoder_2026/checkpoint_archives/full_v9_plus_qwen_external_after_text_reusepreflight_20260604T075031Z` | 4 | 55,291,386,248 | `R/path-f62140913e67b163__nas20261028-omnicoder-full-v9-checkpoint-archives` |
| 212 | `omnicoder_2026/training_runs/fulltrain_v7_93e7417_inbackward_20260530T055809Z` | 6 | 52,452,441,353 | `R/path-f44e9533df237571__nas20261028-omnicoder-fulltrain-v7` |
| 213 | `omnicoder_2026/root_pressure_archive_20260604T0438Z` | 29 | 38,888,161,564 | `R/path-1c00c2cd60feedf9__nas20261028-omnicoder-root-pressure-archive` |
| 214 | `omnicoder_2026/weights/overfit_proof_fastcards` | 3 | 4,222,465,798 | `R/path-22e705ddefa8fee7__nas20261028-omnicoder-overfit-fastcards` |

- The 56 files are 51 `*.pt` plus 5 partial or temp blobs: `.rank00002.pt.lEbqsL`, `.rank00000.pt.hzLdlq` and
  `01_text.step32/.rank0000{0,1,2}.pt.19X.tmp`. The plan's 349 GB claim needed those 5 blobs to be valid; matching only
  `*.pt` would have freed 304 GB.
- The directories named `*.pt/` were never matched. Only regular files were.
- **Rebuild:** not needed, because this is a dead project. Retraining would mean rerunning the `omnicoder_2026_work`
  launchers on the kept `omnicoder_2026/datasets`, which is not planned. Per-file size, mtime and digest are in each
  `MANIFEST.tsv`.

### D2: omnicoder retained v38 checkpoint (44,037,203,404 B, 3 files)
| row | unit | files | bytes | recipe |
|---:|---|---:|---:|---|
| 215 | `omnicoder_2026/checkpoints/retained_v38_clean2_step1184_loss2p451_20260710` | 3 | 44,037,203,404 | `R/path-9b136cad291e42e2__nas20261028-omnicoder-retained-v38` |

- **What it was:** rank00000-00002.pt, preset omnicoder2026_20b_1m, step 1184, loss 2.451.
- **Kept in place:** `manifest.json`, `.complete.json`, `rank*.complete.json`, `.checkpoint_save_attempt.json` and the
  `checkpoint_io` jsonl.
- **Breadcrumb symlink:** `/home/cereal/omnicoder_2026_work/weights/training_runs/full_v9_v38_clean2_uncapped_20260607T1504Z/checkpoints/12_v38_clean2.pt`
  in the dead work tree now dangles. It was left as it is.
- **Rebuild:** not needed (dead project; the metadata stays).

### D3: film2 HunyuanVideo-1.5 (190,042,562,782 B, 55 files)
| row | unit | files | bytes | recipe |
|---:|---|---:|---:|---|
| 216 | `models/film2/hunyuan` | 55 | 190,042,562,782 | `R/path-3ade64d6f486afb9__nas20261028-film2-hunyuanvideo-1.5` |

- **Why:** Cyril ruled on 2026-09-01 that HunyuanVideo is "OUT entirely".
- **Note:** `transformer/480p_t2v_distilled` was a truncated partial (18.1 of 33.3 GB).
- **Rebuild:** `hf download tencent/HunyuanVideo-1.5 --revision 9b49404b3f5df2a8f0b31df27a0c7ab872e7b038 --local-dir /mnt/ai_data/models/film2/hunyuan`.
  To fetch one variant only, add `--include 'transformer/480p_t2v/*' 'vae/*' 'scheduler/*'`.
- **Untouched:** `film2/.hf` and `film2/manifests`.

### D5: legacy ComfyUI SD-era models on the NAS (22,209,392,889 B, 17 files)
| row | unit | files | bytes |
|---:|---|---:|---:|
| 217 | `models/comfyui/checkpoints` (SDXL base + refiner) | 2 | 13,014,060,264 |
| 218 | `models/comfyui/animatediff_models` (mm-Stabilized mid/high) | 2 | 3,344,209,558 |
| 219 | `models/comfyui/controlnet` (openpose/canny SD1.5, plus 2 x 15 B "Entry not found") | 4 | 2,890,314,278 |
| 220 | `models/comfyui/ipadapter` (image encoder, plus 1 x 15 B junk) | 3 | 2,528,373,478 |
| 221 | `models/comfyui/insightface` (antelopev2, 5 onnx) | 5 | 427,550,200 |
| 222 | `models/comfyui/upscale_models` (realesr-general-x4v3) | 1 | 4,885,111 |

- **Recipes:** `R/path-*__nas20261028-comfyui-<dir>`.
- **Rebuild:** run `hf download` against these sources; exact names, sizes and digests are in each `MANIFEST.tsv`:
  - `stabilityai/stable-diffusion-xl-base-1.0` and `stabilityai/stable-diffusion-xl-refiner-1.0`;
  - `manshoety/AD_Stabilized_Motion`;
  - `lllyasviel/control_v11p_sd15_openpose` and `lllyasviel/control_v11p_sd15_canny`;
  - `h94/IP-Adapter` (`models/image_encoder/model.safetensors`);
  - `DIAMONIK7777/antelopev2`;
  - Real-ESRGAN `realesr-general-x4v3.pth`.
- **Untouched:** production ComfyUI (`/home/cereal/comfyui/models`), and `models/comfyui/{upscale,VideoHelperSuite}` on the
  NAS.

## Kept, and why

- **Everything on the keep list:**
  - parents (Kimi K3, the Qwen3.8 family, Qwen-Image 2.1, H3/LTX, ACE-Step, Hunyuan3D, speech teachers);
  - protected fold artifacts (OMNI1, S5, S3, `k3_dense_1_92`, `k3-run-oct09-claude` and its NAS archive with K3 `fold/body`
    L61-92, the s7 units, the run-1 student);
  - teacher caches;
  - corpora and datasets (`training_data/ai-training` is the only copy; `omnicoder_2026/datasets` holds
    `MASTER_SHARD_INDEX_POINTER.json`);
  - sealed corpora and intake versions;
  - backups (agent-memory, omni-lab and chimera_corpus dumps, family-ai, `omnicoder_0.1.0-evidence`);
  - LM Studio archive dirs (`plan-KEEP`).

  None of these was in any deletion set, and the gate's protected list covered them.
- **D4 (Ollama stores) and D6 (failed July transfers) were held, not deleted.** They are under `never-delete` entries
  with no more specific classification; see UNSURE.
- **Not assessed in this pass (left alone):** `models/film2/.hf` (an HF/xet cache), the `models/comfyui/upscale` and
  `VideoHelperSuite` dirs, and the rest of the share outside the proposed groups.

## UNSURE: for Cyril (nothing here was deleted)

| id | path | size | what decides it |
|---|---|---:|---|
| **U1** | `backups/agent-memory-postgres/dumps/`: the 10 nightly dumps 09-17..09-26, with their manifests and restore lists | 146.16 GB | Confirm the 2026-10-18 recovery validation is finished, then remove `state/retention-hold`. The tool's own `RETENTION_DAYS=30` prune then removes them on the next successful run. **Do not rm them by hand.** Alternative: adopt the 14-daily + 8-weekly policy already used for omni_lab, which keeps 17 dumps (409.9 GB) and drops 22 (357.5 GB). |
| **U2** | `backups/agent-memory-postgres/incoming/agent_memory_20261018T224932Z` | 27.41 GB | It has the same byte size (27,410,190,156) as the published `dumps/` copy. Delete it only after both sha256s match the manifest: about 40 min of niced hashing over CIFS. |
| **U3** | `neural-forge/{qwen3_4b_codebook, qwen3_4b_compensated*, progressive_*, real_medoid_dual, same_token_tokenrows_outputgate, native_parent_cache, canonical_blank_targets, compiled_child_cache, cross_family_blank, dynamic_snapshots, runs, trajectory_subset_targetrows, cross_token_real_medoid, qwen_basis_checkpoint}_202607*` | 107.74 GB | Whether the v2 fold-init ruling ("built entirely by folding our existing folds", 2026-10-21) needs any July NF unit as a fold source. Otherwise they are unproven NF checkpoints and fall under delete-with-evidence. One live link exists: `model-surgery-lab/neural-forge-src/runs/residual-attention-20260711`. `neural-forge/archive` is kept regardless. |
| **U4** | `models/Qwen/Qwen3.5-9B` | 19.33 GB | Whether v2 has any role for Q35-9B. It is a registered parent code (`CAMPAIGN.md` R1, `CHECKPOINT_POLICY.md`, `qf_register.py`); the 10-21 plan recommended DELETE. Rebuild: `hf download Qwen/Qwen3.5-9B`. |
| **U5** | `omnicoder_2026/hf_cache` (Qwen3-4B-Base 11.31 GB, SmolLM2-135M, about 100 dataset stubs) | 11.58 GB | Decide together with U3. Qwen3-4B is the parent of those NF units. |
| **D4** | `models/ollama/{deepseek,glm,qwen}/models` | 78.66 GB | Only blocked by the coarse `never-delete /mnt/ai_data/models` root guard. Evidence otherwise says delete: Ollama is absent on the AI server; the media-server Ollama uses `/usr/share/ollama` and lists 0 models; the registry/HF digests match exactly. A one-line OK lets `run_group.sh D4-ollama-model-stores` run once a `protected_waivers` entry is added in the plan. Keep the `id_ed25519` pairs. |
| **D6** | `backups/agent-memory-postgres/incoming/agent_memory_2026072{7T211503Z,7T220103Z,7T223604Z,7T233328Z,8T002947Z,8T010138Z}` | 15.54 GB on disk (19.16 GB apparent) | These are failed partial uploads, never published: no manifest, restore.list or sha. But they sit inside the `never-delete` agent-memory backup tree and the keep list. The backup-lane owner should rule together with U1/U2. |

Total held in UNSURE: about 406 GB (U1-U5 plus D4 and D6, on-disk figures).

**Watch item:** agent-memory backups have failed since 10-23 (ENOSPC, then "destination capacity remained unavailable").
With 603 GB free they should succeed again tonight. Under the retention hold, the share refills at about 28 GB per night,
which gives roughly 3 weeks before the next full share unless U1 is ruled.

## Not verified / left out

- **The owner order's exact wording:** I could not find the 10-27 text in agent memory (the grep timed out; search
  returned nothing). The rows quote it as relayed by the workflow. The owner rulings each group relies on *were* checked in
  the memory files.
- **SMB locks:** `smbstatus` needs root on the media server. Instead, port-445 sessions were checked: only .222 and .84.
- **The rest of the share:** `omnicoder_2026/{local_root_legacy_payloads, benchmarks*}`,
  `backups/omnicoder_0.1.0-evidence/weights`, `neural-forge-stage2/*` and `hf_cache/models--Paranioar--NEO1_5-2B-SFT` are
  plan-UNSURE in the 10-21 list. They were not part of this task's groups and were not touched or assessed.
- **Git:** the omni-lab changes (`cleanup/nas_group_delete.py`, `cleanup/verify_deleted.py` 1.1.0) are not committed.

## Memory-index-worthy facts

- **NAS cleanup 2026-10-28.** The share went from 1 MiB to 603.2 GB free (100% to 88%). Deleted: omnicoder `.pt` blobs (D1
  349 GB, D2 44 GB; evidence kept in place), HunyuanVideo-1.5 (190 GB) and NAS legacy ComfyUI SD models (22 GB). The
  records are omni_lab `model_deletion` rows 209 and 211-222 (`plan_wave` `N1-NAS-20261028`). The report is
  `neural-forge/reports/omni-campaign-2026-09/NAS_CLEANUP_2026-10-27.md`.
- **The NAS is ZFS behind NFS behind Samba** (`192.168.122.170:/mnt/ai_pool/ai_dataset`, re-exported by the media server).
  At 100% full, **unlink fails with ENOSPC**. Truncate-to-0 also *reports* ENOSPC but takes effect, and ZFS frees the
  space asynchronously over about a minute. Fix: truncate first, wait for `df`, then unlink. No snapshot pins the blocks.
  Never let the share reach 100%.
- **The 2026-10-21 protected list is layered.** Coarse `never-delete` root guards (e.g. `/mnt/ai_data/models`,
  `/mnt/ai_data/backups/agent-memory-postgres`) sit above specific `plan-UNSURE` entries. A deletion under a root guard
  needs a recorded per-path waiver (`nas_group_delete.py` `protected_waivers`) and an owner ruling.
- **Tools:**
  - `omni-lab/cleanup/nas_group_delete.py` 1.0.0 handles group deletions on the NAS: file-subset units, an inbound-symlink
    gate, an omni_lab row scan, ENOSPC truncate fallback and resume of failed rows.
  - `verify_deleted.py` 1.1.0 honours `kept_in_place`.
  - The runner is `/home/cereal/nas-cleanup-20261028/run_group.sh <group>`.
- **Held for Cyril:** D4 Ollama (79 GB) and D6 July partials (16 GB), plus U1-U5. The biggest lever is U1, the
  agent-memory retention hold: 146 GB now, growing about 28 GB per night.
