# Operations scripts, 2026-10-18 (Claude 753087a5)

These are one-off operations scripts, each run on the AI server from `/home/cereal/cleanup-20261018/`. Each is kept here for provenance.

| Script | What it did |
|---|---|
| `prof_parse.py` | Ranks a py-spy raw profile by the deepest trainer frame and by the leaf frame. This found the run-1 host-copy bottleneck (73% of a step) and the M1 overlay-compose bottleneck (65%). |
| `ace_bind_check.py` | Real-shard check of the 1.3.3 ACE code-index binder: the sequence round-trips, the prompt decodes, positions and ids are in range, and top-k mass is at most 1. |
| `make_v5.py` | Writes `media_m1/media_data_v5.json` (vision_in `group_tokens` 10000) after the 15:19:50Z OOM. |
| `run1_pin_restart.sh` | Waits for run 1's next safety checkpoint, STOPs the queue, installs trainer 1.0.1, and relaunches with `--pin-weights`. |
| `base_identity_live.py` | Diffs run 1's live base identity against the stored identity of the checkpoint the queue refused. Only the p1 reader's sha and version differ, which licenses `--allow-identity-change`. |
| `flash_gguf_stage.sh` | Copies the Flash Next GGUF from the NAS to the root SSD (direct I/O, resumable), checks its sha256, and writes the Flash lane's OK mark. |
