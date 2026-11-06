# AI server disk cleanup, 2026-11-04 to 05

Owner order (Cyril, 2026-11-04, relayed by the Chimera Omni lead): "[Cyril quote withheld]".
Host: 192.168.50.222. Disks: `/` (sdb2, ext4, 1.97 TB) and `/mnt/lexar` (sda2, exFAT USB, 4.10 TB).
Work dir, logs and scan outputs: `/home/cereal/omni-campaign/run0/maint/disk-cleanup-20261104/`.
Tools and evidence: chimera-omni branch `claude/disk-cleanup-20261104`, directory `ops/run0/maint/disk-cleanup-20261104/`.
DATA_INDEX change: `claude/run0-blocks-20261029` bff3d5b, deployed to `tools/data_index/data_index.py` (sha `0e14f402…`).
Relay: CLAIM at 204301, then STATUS at 204320, 204321, 204334, 204338, 204340 and 204349.

**Update 2026-11-05 ~03:00Z:** the NAS recovered at 02:54:36Z (Relay 204405), but the lead said **NO GO**:
- WCJ79SX7, the other ai_pool drive, is at its wear rating and still shows slow minutes (78-128 ms).
- Cyril chooses between replacing the drives and letting the chunked copy trickle through. Only his answer, relayed by
  the lead, starts the queue.
- The queue stays armed as is. If it starts after 11-06 02:15Z, the NAS session first extends its probe and mirror units.

**Final state (stood down 2026-11-05 ~02:00Z, lead decision):**
- The agent-memory backup is published and verified twice.
- The Lexar meets its target.
- `/` does not. Its last move (comfyui/output) and the Lexar `.venv` move are on HOLD: the NAS's SMR drives collapse under
  sustained writes.
- The moves resume only when Cyril changes the NAS disks or the lead says so.
- Nothing on the AI server is deleted to make room.

## Result

| Disk | Free before (`df -B1`, 11-04 15:29Z) | Free after (11-05 01:46Z) | Net change | Target |
|---|---:|---:|---:|---:|
| `/` | 98,563,633,152 (91.8 GiB) | 122,688,016,384 (114.3 GiB) | +24,124,383,232 | 150 GB: **not met** |
| `/mnt/lexar` | 223,050,989,568 (207.7 GiB) | 300,525,027,328 (279.9 GiB) | +77,474,037,760 | 300 GB: **met** |

- **`/`:** this session freed ~30.3 GB (table below). Other sessions wrote ~6 GB over the same 10 h: run-0 data, lanes,
  logs.
- **Lexar:** freed 75.6 GB of file bytes, plus exFAT cluster slack (1 MiB clusters).
- **With the two held moves:**
  - `/` would gain 83.3 GB, to ~206 GB free.
  - The Lexar would gain ~59 GiB of `.venv` clusters, to ~360 GB free.

## What freed the space

| Disk | Item | Bytes | How |
|---|---|---:|---|
| Lexar | `/mnt/lexar/omnicoder_0.1.0/.git` (2,348 files, the 2025 repo's history) | 68,995,845,797 | Moved to NAS `omni-campaign/lexar-relief-20261104/omnicoder_0.1.0/.git` with `lexar_move_v3` (rsync per file, sha256 on both sides, source removed only after every file verified). Done 23:18:50Z. Breadcrumb `/mnt/lexar/omnicoder_0.1.0/.git.MOVED.json`. |
| Lexar | `omnicoder_0.1.0/data/wiki/enwiki-latest-pages-articles.xml.bz2` | 6,578,241,536 | Deleted: all zeros (`cmp` against /dev/zero), no references. |
| `/` | `/home/cereal/ai-benchmark-lab` (445,037 files, of which 211k are hardlinks; June benchmark install, cold since 07-27) | 23,588,302,848 on disk (49,975,174,145 counted per path) | Moved as `root-relief-20261104/tar/ai-benchmark-lab.part-000.tar` + `part-001.tar` (23.3 GB) with `tar_move.py` 1.1.0. All 494,937 members were sha256-checked by reading the parts back from the NAS before the source was removed. Done 00:20:01Z. Breadcrumb `/home/cereal/ai-benchmark-lab.MOVED.json`. |
| `/` | 10 unused docker images (gauntlet/codex comfyui builds, ice-h3-te-release, omni-qimg-capture, llama.cpp server-cuda) | 4,396,531,712 (df) | `docker rmi`. Checked first: no container, compose file, unit, lane, skill or ICE repo references them. |
| `/` | journal older than 7 days | ~2.1 GB | `journalctl --vacuum-time=7d` |
| `/` | apt and pip caches | 245,544,279 | `apt-get clean`, `pip cache purge` |

No omni_lab `model_deletion` rows were written: no model or checkpoint was deleted.

## Still on disk (held), with re-run commands

| Item | Size | State |
|---|---:|---|
| `comfyui/output`: the 59,774 audited MOVE entries (134,303 files) | 83,328,810,634 B | Untouched. The tar move stopped cleanly at 01:43:18Z when the NAS stalled. |
| Lexar `omnicoder_0.1.0/.venv` (52,082 files, a Windows venv) | 2,363,433,680 B of files, ~59 GiB of exFAT clusters | Not started (tar move). |

Re-run only after BOTH of these: the NAS session reports the SMR drive "recovered", and the lead says go (or Cyril changes
the NAS disks). One command; nothing else needs to change:

```bash
setsid nohup /home/cereal/omni-campaign/run0/maint/disk-cleanup-20261104/move_queue.sh > /home/cereal/omni-campaign/run0/maint/disk-cleanup-20261104/move_queue.out 2>&1 < /dev/null &
```

What the queue does:
- It waits for any agent-memory backup runner.
- It skips every step whose breadcrumb says `state=moved`: .git and ai-benchmark-lab are done.
- It runs comfyui/output, then the Lexar .venv, one at a time:
  - `tar_move.py` in a `capped.sh --high 4G` scope;
  - `--rate-kbps 20000`, fsync every 64 MiB;
  - every member sha256-checked from the NAS before the source is removed;
  - the run stops at the first failed write (no retry loop);
  - parts of 20 GB or less. After each part (and before each NAS step and before verification) it waits on
    `nas_wait_ok.sh`. This is the NAS session's chunk-and-pause rule (tar_move 1.2.0 `--between-parts-cmd`, 744bee3;
    host gate 16a9a8f). Two gates must both pass:
    1. **Host gate:** the media host's ai_pool latency state (`OK|BUSY …`, from O_DIRECT reads on each drive), read from
       its mirror on the share at `NAS_STATE_FILE=/mnt/ai_data/.nas-tuning/ai_pool_state`. It must say OK, be under
       3 min old, and have held OK for 10 straight minutes. A missing, stale or BUSY file waits, with no probe write.
       The mirror is live: unit `ai-pool-state-mirror-20261105` on .250, written once a minute. A dead probe writes
       `BUSY … probe-stale`. Read from the AI server at 02:17Z 11-05: `BUSY … WCJBH6XQ_med=81ms p90=363ms`, 41 s old.
       **Both the probe unit and the mirror unit stop around 02:15Z 11-06.** After that the file goes stale and the queue
       waits forever (fail-safe), so a later re-run needs the NAS session to extend them first.
    2. **Write probe:** 3 straight 8 MiB fsync'd probe writes to the share under 1.5 s, one per minute.
- Then it deletes the partial NAS copies listed below.

The comfyui re-run rewrites its part files from scratch. Its list is `comfy_move_list.frozen.txt`; entries gone since the
audit are dropped and logged, and files dated on or after the cutoff refuse the run.

Watch progress in `queue.log`, `tar_move_comfy.log` and `tar_move_venv.log`.

**Also walking the NAS:** the `omni-data-index` user timer (every 30 min) runs `du -sb` on each NAS catalog path. Its
reads land on the cleaning drive too. Pausing it is the lead's call; at 02:00Z 11-05 I stopped only my own manual refresh.

**Caution:** the ai_pool drives are drive-managed SMR (ST5000LM000, a mirror). They take ~30 GB at ~10 MB/s, then drop to
~1 MB/s while the drive rewrites its shingled bands, which also slows Plex and every other NAS user. The 83 GB comfy move
will hit that unless the disks change. This is the lead's reason for the hold.

## NAS leftovers from the stopped attempts (no breadcrumb points at them; the queue's last step deletes them)

- `/mnt/ai_data/root-relief-20261104/tar/comfyui-output-20261104.part-000.tar`: 20.0 GB, complete but unverified.
- `/mnt/ai_data/root-relief-20261104/tar/comfyui-output-20261104.part-001.tar`: ~13 GB, partial. A re-run overwrites
  both.
- `/mnt/ai_data/root-relief-20261104/ai-benchmark-lab/`: a partial per-file rsync copy (17:33Z). The tar parts replace it.
- `/mnt/ai_data/root-relief-20261104/comfyui-output/`: a partial per-file rsync copy.
- `/mnt/ai_data/omni-campaign/lexar-relief-20261104/omnicoder_0.1.0/.venv/`: a partial per-file copy.

## Agent-memory backup

| Time (Z) | Event |
|---|---|
| 11-04 15:57 | Run started (lead decision a). Preflight passed. |
| 17:12 | Dump done, 31,209,873,398 B. |
| 17:14 | Local verify ok. |
| 18:10, 18:46 | `close()` EIO on the stalling NAS. |
| 18:54 | Stopped. The verified stage was kept as pending; the failed copy was deleted by the runner itself. |
| 18:59 | Cron run HELD, with an expiry (`state/run-hold.json`, auto-lift by 11-05 18:00Z). |
| 19:40 | NAS OK for rate-limited writes. Hold lifted (`--now`). Publish started at 20 MB/s, one attempt. |
| 20:16 | Paused on Cyril's word ("[Cyril quote withheld]") for the vm1 virtio/RAM window. Stage kept. |
| 20:41 | NAS DONE. Publish re-run. |
| 21:40:55 | `verify destination: ok` (sha256 `880364ca34ad51cf…`, 351 TOC entries), then `published verified backup: agent_memory_20261104T155710Z`, then the 11-03 dump, manifest and restore list were deleted. |
| 22:03:11 | An independent `verify` of the published copy passed. |

Now:
- `latest_backup_id.txt` and `dumps/` hold only `agent_memory_20261104T155710Z`: 1,283,908 events, latest_event_id
  1341318.
- The nightly cron is active.
- Exit 3 is the known warning: the two root-owned 10-18 `incoming/` recovery dirs (0 bytes) need sudo on .250.
- `/`'s low point during the dump was 71.3 GiB, never under the lead's 60 GiB floor.

## What used the 64 GB on `/` overnight (11-03 18:15Z to 11-04 04:15Z)

Every file on `/` was summed by change time. It was all live work, not core dumps:

| When (Z) | What | GB |
|---|---|---:|
| ~19:00 | `models/judge/qwen3-omni-30b-a3b-instruct-awq4` (new judge model download) | 25.7 |
| ~20:00 | `~/.venvs/vllm-omni` | 9.9 |
| 01:00-03:00 | `run0/data/parts` (6.75) + `run0/staging/swe-windows` (4.8) | 11.5 |
| 18:00-03:00 | agent-memory PG volume (55 GB now) | ≤ 4 |
| 02:00-03:00 | docker overlay, `export/llamacpp-tier-mtmd` build, `strata-pcache-disk` tar | 6.1 |
| 04:10 | `/var/crash` crash report | 1.4 |

**Core dumps:** apport writes one report per executable and skips later crashes while that one is "unseen".
`/var/lib/systemd/coredump` is empty.

**KEPT, untouched:** `/var/crash/_usr_bin_python3.12.1000.crash` (1,509,328,348 B). It is the run-0 trainer SIGSEGV of
11-04 04:10Z; the 06:28, 08:37, 10:54 and 13:10Z crashes were skipped by apport. It is the newest run-0 core that
exists.

## comfyui/output audit (read-only subagent; files in `comfyui-output-audit/`)

- **MOVE: 83.33 GB.** Every file is older than 2026-10-28. Each either sha-matches intake, corpus or an ICE NAS archive,
  or no live consumer references it (gen-worker-media, the media judge, the media-gate spool, run0 manifests, DATA_INDEX,
  ICE tools, units, crontab).

  | Part | Size |
  |---|---:|
  | `minimax_h3` ICE run dirs | 26.5 GB |
  | `ice_h3_realesrgan` | 18.6 GB |
  | `h3_run` matched part | 14.7 GB |
  | old `aiserve` tasks | 19.7 GB |
  | small dirs and loose files | 3.9 GB |

- **KEEP: 29.8 GiB.**
  - Gauntlet clips, 25.5 GB: Cyril's 09-13 keep order.
  - Pending or recent `aiserve` tasks.
  - `h3_run` files with no copy elsewhere.
  - `afterimage-vault-*`.
  - `sprite/`.
  - `quarantine_h3_public_license_20260828`.
  - 109 receipt-pinned files.
- `build_comfy_list.py` rebuilt the set independently: no overlap with any KEEP or pinned item, and the newest file is
  from 10-27T19:53Z.

## Tools written (all on `claude/disk-cleanup-20261104`)

- **`tar_move.py` 1.1.0:** moves small-file trees, or listed entries, as GNU tar parts.
  - sha256 of every source file, then streams each part back from the NAS with the page cache dropped.
  - The read-back part hash must equal the bytes written.
  - `--rate-kbps` cap, fsync every 64 MiB, crash-safe.
- **`tree_move.py` 1.2.0:** per-file rsync with symlinks, hardlinks and `--list` mode. Suits trees of a few big files only.
- **`lexar_move_v3.py` copy:** with `LM3_REASON`, `LM3_BWLIMIT` and `LM3_STOP_ON_FAIL`.
- **`move_queue.sh`:** re-runnable.
- **`backup_hold_set.sh` / `backup_hold_lift.sh`:** a time-limited hold on the backup cron.
- **`build_comfy_list.py`:** the independent rebuild of the comfy MOVE set.

## Looked at and left alone

- **`~/pydantic-audit-20261022/venv-nopyd`:** a `cp -al` hardlink copy, so ~0 B to gain.
- **HF caches (13 GB):** all in use by neural-forge-src and the run0 good_view evals.
- **`run0/keep` (step 663) and `run0/rollback` (step 400):** unique checkpoints.
- **2,655 stopped gauntlet-blender containers:** 0.19 GB.
- **snapd, `/tmp`:** nothing worth taking.
- **Protected and never-delete trees** (10-21 protected list and rescope): `models`, `comfyui/models`,
  `omni-campaign/runtime`, `k3-run-oct09-claude`, composite c10/s7, `lexar-relief-local`, the distill trees; Kimi-K3,
  teacher_cache, the K3 folds, native_executor, clones, omni1, corpus and run0 on the Lexar.

## Held for Cyril

| Item | Size | Why it is held |
|---|---:|---|
| comfyui/output Gauntlet clips | 25.5 GB | 09-13 keep order |
| Stopped compose container `comfyui` (exited 09-01) | 21 GB | Its writable layer holds an `ACE-Step/acestep-captioner` HF download. `docker rm comfyui` would free it, but the container is compose-defined. |
| `models/gguf/Qwen3.8-Flash-Next-Strata` | 83 GB | The lead says keep (plan-KEEP rollback). |
| composite c10 + s7 | 86 GB | The lead says keep (protected fold trees). |
| Lexar `omnicoder_0.1.0/models/hf` | 277 GB | 2025 HF caches, partly plan-UNSURE. |
| Lexar `omni-campaign/scratch/{arch-v2-trainer, arch-v2-proxy-proof, v2-fold, k3accel}` | 21 GB | Not confirmed finished. |
| `ai-serve/spool/media` | 15 GB | Sprite/media spool. |
