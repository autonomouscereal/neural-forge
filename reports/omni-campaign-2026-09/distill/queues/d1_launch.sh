#!/bin/bash
# SUPERSEDED 2026-10-14 (Claude, plan STATE_2026-10-14_CODEX_HANDOFF.md Step 1c; PREREGISTRATION_OMNI_STAGE_D_v1 amendment A1):
# D1 runs on Codex's source-isolated successor dataset (979 rows, 917/62) through its allocator-wrapped launch, not on
# the rows [0, 977) prefix below. 1.1.1 refuses every weight-moving run (no argument, --run, anything but --dry-run /
# --check-gates) so a second OMNI1D1 can never come from the old prefix. --dry-run / --check-gates are kept as evidence.
# $OC/distill/queues/d1_launch.sh -- stage D1 production launcher (trainer-prep lane, Claude, 2026-10-13; E4).
#
# What it runs: d1_trunk_recovery.py 1.2.4, two-card host-reduced data parallel on GPU0 + GPU4, every text layer
# 0..63, base = OC02-F-Q38-27BU-OMNI1.clone-pre-D1 (directory read from the ledger), the first 977 rows of
# distill/corpus/text (4,001,792 tokens, PREREGISTRATION_OMNI_STAGE_D_v1 section 3 D1), then
# d1_delta_recipe.py (D1_delta.json -> DELTA.json) and oc_checkpoint.py register --delta-base (dry run, then
# --confirm) as OC02-D-Q38-27BU-OMNI1D1, then delta-compose --deep --plan-out.  The R2 clone
# OMNI1D1.clone-pre-D2 and the review package are NOT done here (they belong to the D1 seal / D2 start).
#
#   d1_launch.sh --dry-run   argument parsing + preflight only: CPU, no GPU touched, nothing written except
#                            $Q/d1/preflight/ (the receipt and the 31 MB ids file the real run reuses)
#   d1_launch.sh --check-gates   print the gate verdict once (and whether either card lock is held, read from
#                            /proc/locks); exit 0 if every gate holds and both locks are free, else 1.  Waits for
#                            nothing and takes no lock (a probe flock could make a gpu0.sh start refuse).
#   d1_launch.sh             waits (inside this script) for every gate below, then runs.  DO NOT start it before
#                            the K3 cache has ended and both 3090 queues (gpu0.sh, gpu4.sh) have finished.
#
# Gates (real run; each re-checked every 60 s until all hold): step 1's S8 review pid 917640 has exited; no K3
# teacher pass under ANY copy of the tool (`k3_teacher_cache*.py run`: the gpu6 lane and its bisect run the frozen
# $Q/frozen/k3_teacher_cache-gpu6.py), every k3-pass-k job's latest state in $K3_STATUS is PASS or DONE, and
# >= K3_SEALED_MIN sealed K3 manifests under $K3_ROOT (D1 pins ~82 GB of host buffers, the K3 pass 97-130 GB:
# they never overlap); no live gpu0*.sh / gpu4*.sh queue under any path; GPU0 and GPU4 each < 2000 MiB used with
# no compute process on either; MemAvailable >= 100 GiB.  Then the card locks are taken without blocking:
# $Q/gpu0.lock (gpu0.sh takes it with `flock -n` and refuses to start while D1 holds it) and $Q/gpu4.lock
# (advisory only: gpu4.sh 1.0.2 takes no lock).  Once: the pre-registration verifies against its pin, the ledger
# base exists, root and Lexar keep their 60 GiB floors.  Then EVERY gate is re-checked with the locks held,
# immediately before d1 starts; a failure releases the locks and waits again.  The locks are held until d1 exits.
set -u
LAUNCHER_VERSION=d1_launch-1.1.1-superseded
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
EF=$OC/import/rq27b/tree_ef1ffb9
L=$OC/ledger/oc_checkpoint.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
BASE_ID=OC02-F-Q38-27BU-OMNI1.clone-pre-D1
OMNI1_ID=OC02-F-Q38-27BU-OMNI1
DELTA_ID=OC02-D-Q38-27BU-OMNI1D1
PARENT=/mnt/ai_data/omni-campaign/s0-probe-sep04        # the parent as a Chimera checkpoint (S0 import), read only
PARENT_HF=/home/cereal/models/Qwen3.8-27B-Uncensored
PREREG=$OC/distill/PREREGISTRATION_OMNI_STAGE_D_v1.md
TEXT=$OC/distill/corpus/text
ROWS=977                                                # 977 x 4096 = 4,001,792 tokens
RUN=$OC/distill/d1/run-OMNI1D1                          # --out: rung files (8.6 GB, root, released after registration)
DELTA=/mnt/lexar/omni-campaign/deltas/$DELTA_ID         # --delta-out (8.56 GB; registered, permanent)
PRE=$Q/d1/preflight
IDS=$PRE/text_rows0-977.safetensors
LOG=$Q/d1/d1.log
STATUS=${D1_LAUNCH_STATUS:-$Q/d1_launch.status}   # override only for the 1.1.1 guard test
K3_ROOT=${K3_ROOT:-/mnt/lexar/omni-campaign/teacher_cache/k3}
K3_SEALED_MIN=${K3_SEALED_MIN:-4}
K3_STATUS=${K3_STATUS:-$Q/gpu6.status}                  # lane gpu6's queue status: "<utc> <job> <state> <note>"
K3_PASSES=${K3_PASSES:-0 1 2 3}
S8_PID=917640
export CHIMERA_SRC=$EF/src CHIMERA_OMNI_SRC=$EF PYTHONPATH=$T:$OC/ledger:$EF/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DRY=0; CHECK=0
[ "${1:-}" = "--dry-run" ] && DRY=1
[ "${1:-}" = "--check-gates" ] && CHECK=1
[ "$DRY" = 0 ] && [ "$CHECK" = 0 ] && { echo "$(date -u +%FT%TZ) d1_launch REFUSED ${1:-<no-arg>}: $LAUNCHER_VERSION (successor D1, prereg amendment A1)" | tee -a "$STATUS" >&2; exit 4; }
mkdir -p "$PRE" "$Q/d1"
status() { echo "$(date -u +%FT%TZ) d1_launch $*" >> "$STATUS"; echo "$(date -u +%FT%TZ) $*"; }

D1_ARGS=(--child "@BASE@" --parent "$PARENT" --ids "$IDS" --lengths-key lengths --layers all
         --devices cuda:0,cuda:1 --dtype bfloat16 --train-dtype float32 --master-dtype float32
         --epochs 2 --tokens-per-step 32768 --micro-tokens 8192 --lr 2e-5 --warmup-steps 10 --min-lr-frac 0.1
         --clip 1.0 --seed 0 --loss-site residual --base-digest full --min-free-gib 60 --resume
         --out "$RUN" --delta-out "$DELTA")

base_dir() {   # the ledger's directory for $1, or nothing
  $PY $L show --checkpoint-id "$1" 2>/dev/null | $PY -c 'import json,sys; print(json.load(sys.stdin)["checkpoint"]["directory"])' 2>/dev/null
}

# ---------------------------------------------------------------- preflight (both modes; CPU only)
preflight() {   # $1 = base directory to check against; $2 = label
  CUDA_VISIBLE_DEVICES="" nice -n 15 $PY - "$1" "$2" "$IDS" "$TEXT" "$ROWS" "$PRE/preflight.json" "$PREREG" "${D1_ARGS[@]}" <<'EOF'
import hashlib, json, os, shutil, sys, time
from pathlib import Path
base, label, ids_path, text_dir, rows, receipt_path, prereg = sys.argv[1:8]
d1_args = [base if a == "@BASE@" else a for a in sys.argv[8:]]
rows = int(rows)
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import d1_trunk_recovery as d1
import stage2_training_receipt as s2

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

out = {"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_label": label, "base": base,
       "d1_tool": {"version": d1.TOOL_VERSION, "sha256": sha(d1.__file__)}}
# 1. the pre-registration against its pin
out["preregistration"] = s2.assert_preregistration_unedited(Path(prereg))
print("PREREG OK", out["preregistration"]["preregistration_sha256"][:16], flush=True)
# 2. the ids: corpus/text rows [0, rows), every shard checked against the corpus manifest first
manifest = json.loads((Path(text_dir) / "manifest.json").read_text())
parts, lens, used, have = [], [], [], 0
for shard in manifest["shards"]:
    if have >= rows:
        break
    if sha(shard["path"]) != shard["sha256"]:
        raise SystemExit(f"{shard['path']} does not match the corpus manifest")
    with safe_open(shard["path"], "pt") as h:
        ids = h.get_tensor("input_ids").to(torch.int64)
        real = h.get_tensor("real_lengths").to(torch.int64)
    take = min(rows - have, ids.shape[0])
    parts.append(ids[:take]); lens.append(real[:take]); have += take
    used.append({"path": shard["path"], "sha256": shard["sha256"], "rows": take})
ids, lengths = torch.cat(parts), torch.cat(lens)
if os.path.exists(ids_path):
    with safe_open(ids_path, "pt") as h:
        same = torch.equal(h.get_tensor("input_ids"), ids) and torch.equal(h.get_tensor("lengths"), lengths)
    if not same:
        raise SystemExit(f"{ids_path} exists with other content; remove it deliberately")
else:
    save_file({"input_ids": ids, "lengths": lengths}, ids_path, metadata={"source": json.dumps(used), "rows": str(rows)})
out["ids"] = {"path": ids_path, "sha256": sha(ids_path), "rows": int(ids.shape[0]), "seq": int(ids.shape[1]),
              "tokens": int(ids.numel()), "valid_tokens": int(lengths.sum()), "from_shards": used,
              "corpus_manifest_sha256": manifest.get("manifest_sha256")}
print(f"IDS {ids.shape[0]}x{ids.shape[1]} = {ids.numel():,} tokens, valid {int(lengths.sum()):,}", flush=True)
# 3. d1's own argument parser and its read-only preflight functions
args = d1.build_parser().parse_args(d1_args)
tree = d1.bind_chimera()
child, parent = d1.open_checkpoint(args.child), d1.open_checkpoint(args.parent)
d1.guard_out(Path(args.out).absolute(), [child.root, parent.root], bool(args.resume))
d1.guard_delta_out(Path(args.delta_out).absolute(), Path(args.out).absolute(), [child.root, parent.root])
lay = d1.union_layout(child, args.trunk_width, args.k3_channels)
if (lay.width, lay.trunk, lay.k3, lay.pad) != (16640, 4352, 12288, 0):
    raise SystemExit(f"union layout {lay} is not the prereg's (16640, 4352, 12288, 0)")
if child.hidden != parent.hidden or len(child.text_blocks) != 64 or len(parent.text_blocks) != 64:
    raise SystemExit("child / parent geometry")
for i, (j, k) in enumerate(zip(child.text_blocks, parent.text_blocks)):
    if child.cfg.block_kind(j) != parent.cfg.block_kind(k):
        raise SystemExit(f"text layer {i}: block kinds differ")
if str(child.cfg.activation_kind) != "silu" or parent.charted_blocks:
    raise SystemExit("activation / parent charted blocks")
text_rows = min(d1.TEXT_ROWS, int(child.cfg.ledger_band_begin()), int(parent.cfg.ledger_band_begin()))
got, got_len, rec = d1.load_ids(args.ids, args.seq_len, args.max_rows, args.lengths_key)
valid = torch.arange(got.shape[1]).unsqueeze(0) < got_len.unsqueeze(1)
if int(got[valid].max()) >= text_rows or int(got[valid].min()) < 0 or int(got_len.min()) < 1:
    raise SystemExit("ids outside the text rows")
R, Lq, h = int(got.shape[0]), int(got.shape[1]), int(child.hidden)
held = max(1, R // 16)
free_out = d1.disk_floor(Path(args.out).absolute(), float(args.min_free_gib), 64 * 134_000_000)
free_delta = d1.disk_floor(Path(args.delta_out).absolute(), float(args.min_free_gib), 8_560_000_000)
mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable")}
buffers = 2 * R * Lq * h * 2
trainable = len(child.text_blocks) * (3 * h * lay.trunk + h)
out.update({"chimera": tree, "d1_argv": d1_args, "layout": lay.__dict__, "text_rows": text_rows,
            "child": {"root": str(child.root), "text_blocks": len(child.text_blocks), "charted_blocks": child.charted_blocks,
                      "identity": child.identity()},
            "parent": {"root": str(parent.root), "text_blocks": len(parent.text_blocks)},
            "rows": {"total": R, "held": held, "fit": R - held, "fit_tokens": (R - held) * Lq},
            "trainable_parameters": trainable, "host_buffers_bytes": buffers,
            "host_mem_kib": mem, "free_gib": {"out": round(free_out, 1), "delta": round(free_delta, 1)},
            "verdict": "preflight pass"})
Path(receipt_path).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
print(f"LAYOUT {lay}; child {child.root} text {len(child.text_blocks)} charted {child.charted_blocks}; parent {parent.root}", flush=True)
print(f"ROWS fit {R - held} held {held}; trainable {trainable:,}; host buffers {buffers / 1e9:.1f} GB; "
      f"MemAvailable {mem['MemAvailable'] / 2**20:.1f} GiB; free out {free_out:.0f} GiB delta-dir fs {free_delta:.0f} GiB", flush=True)
print(f"PREFLIGHT PASS -> {receipt_path}", flush=True)
EOF
}

BASE=$(base_dir $BASE_ID)
if [ "$DRY" = 1 ]; then
  status "dry-run start ($LAUNCHER_VERSION)"
  $PY $T/d1_trunk_recovery.py --help > /dev/null || { status "dry-run FAIL: d1 --help"; exit 2; }
  $PY $T/d1_delta_recipe.py --version && $PY $T/stage2_training_receipt.py --version
  if [ -z "$BASE" ]; then
    BASE=$(base_dir $OMNI1_ID)
    LABEL="$OMNI1_ID (stand-in: $BASE_ID is not in the ledger yet; the clone is byte-identical by construction)"
  else
    LABEL=$BASE_ID
  fi
  [ -n "$BASE" ] || { status "dry-run FAIL: neither $BASE_ID nor $OMNI1_ID resolves in the ledger"; exit 2; }
  echo "base: $LABEL -> $BASE"
  preflight "$BASE" "$LABEL"; RC=$?
  status "dry-run end rc=$RC base=$LABEL"
  exit $RC
fi

# ---------------------------------------------------------------- gates (real run and --check-gates)
k3_pass_state() {   # latest state of job k3-pass-$1 in $K3_STATUS, or nothing
  awk -v j="k3-pass-$1" '$2 == j {s = $3} END {print s}' "$K3_STATUS" 2>/dev/null
}
gates() {
  local why="" g k st used apps
  kill -0 $S8_PID 2>/dev/null && why="$why s8-pid-$S8_PID-alive"
  pgrep -f 'k3_teacher_cache[^ /]*\.py run' > /dev/null && why="$why k3-run-live"
  for k in $K3_PASSES; do
    st=$(k3_pass_state $k)
    case "$st" in PASS|DONE) ;; *) why="$why k3-pass-$k-${st:-none}" ;; esac
  done
  local sealed=$(grep -rlE --include=manifest.json '"verdict": ?"sealed"' "$K3_ROOT" 2>/dev/null | wc -l)
  [ "$sealed" -ge "$K3_SEALED_MIN" ] || why="$why k3-sealed-$sealed-of-$K3_SEALED_MIN"
  pgrep -f '(^|[ /])gpu0[^ /]*\.sh( |$)' > /dev/null && why="$why gpu0-queue-live"
  pgrep -f '(^|[ /])gpu4[^ /]*\.sh( |$)' > /dev/null && why="$why gpu4-queue-live"
  for g in 0 4; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
    { [ -n "$used" ] && [ "$used" -lt 2000 ]; } || why="$why gpu$g-${used:-?}-MiB"
    apps=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
    [ "$apps" -eq 0 ] || why="$why gpu$g-$apps-compute-procs"
  done
  local avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] || why="$why memavail-${avail}GiB"
  echo "$why"
}
take_card_locks() {   # never blocks: a lock held by anyone else means a lane is live on that card
  exec 8>> "$Q/gpu0.lock"
  flock -n 8 || { exec 8>&-; return 1; }
  exec 7>> "$Q/gpu4.lock"
  flock -n 7 || { exec 7>&- 8>&-; return 1; }
}
release_card_locks() { exec 7>&- 8>&-; }
lock_holders() {   # card locks someone holds now, read from /proc/locks (takes nothing)
  local f ino held=""
  for f in gpu0 gpu4; do
    [ -e "$Q/$f.lock" ] || continue
    ino=$(stat -c %i "$Q/$f.lock")
    awk -v i="$ino" '{n = split($6, a, ":"); if (a[n] == i) f = 1} END {exit !f}' /proc/locks && held="$held $f.lock"
  done
  echo "$held"
}

if [ "$CHECK" = 1 ]; then
  WHY=$(gates)
  HELD=$(lock_holders)
  echo "$LAUNCHER_VERSION gates:${WHY:- all hold}; card locks held:${HELD:- none}"
  [ -z "$WHY" ] && [ -z "$HELD" ] && exit 0
  exit 1
fi

# ---------------------------------------------------------------- real run: wait for every gate (never before)
status "waiting for gates ($LAUNCHER_VERSION)"
PREFLIGHT_DONE=0
while true; do
  while true; do
    WHY=$(gates)
    [ -z "$WHY" ] && break
    [ "${LAST_WHY:-}" != "$WHY" ] && status "blocked:$WHY"
    LAST_WHY=$WHY
    sleep 60
  done
  if ! take_card_locks; then
    [ "${LAST_WHY:-}" != " card-lock-held" ] && status "blocked: card-lock-held"
    LAST_WHY=" card-lock-held"
    sleep 60
    continue
  fi
  if [ "$PREFLIGHT_DONE" = 0 ]; then
    BASE=$(base_dir $BASE_ID)
    [ -n "$BASE" ] || { status "FAIL: $BASE_ID is not in the ledger (step 1 S9 has not cloned OMNI1)"; exit 2; }
    preflight "$BASE" "$BASE_ID" || { status "FAIL: preflight"; exit 2; }
    PREFLIGHT_DONE=1
  fi
  WHY=$(gates)                      # again, locks held, right before the cards are loaded
  [ -z "$WHY" ] && break
  status "re-gate after preflight:$WHY (locks released; waiting again)"
  LAST_WHY=$WHY
  release_card_locks
  sleep 60
done

D1_RUN_ARGS=("${D1_ARGS[@]}")
for i in "${!D1_RUN_ARGS[@]}"; do [ "${D1_RUN_ARGS[$i]}" = "@BASE@" ] && D1_RUN_ARGS[$i]=$BASE; done
START=$(date -u +%FT%TZ)
for g in 0 4; do
  printf '{"lane":"D1 OMNI1D1 (trainer-prep launcher)","gpu":%s,"card":"%s","by":"claude trainer-prep lane","started_utc":"%s","python":"%s","log":"%s","status":"running"}\n' \
    $g "$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)" "$START" "$PY" "$LOG" > $OC/composite/receipts/allocator/d1-omni1d1-gpu$g-claude.json
done
status "d1 start base=$BASE ($LAUNCHER_VERSION; card locks held)"
cd $T
CUDA_VISIBLE_DEVICES=0,4 nice -n 15 /usr/bin/time -v $PY d1_trunk_recovery.py "${D1_RUN_ARGS[@]}" >> "$LOG" 2>&1
RC=$?
release_card_locks
for g in 0 4; do
  printf '{"lane":"D1 OMNI1D1 (trainer-prep launcher)","gpu":%s,"card":"%s","by":"claude trainer-prep lane","started_utc":"%s","ended_utc":"%s","exit":%s,"python":"%s","log":"%s"}\n' \
    $g "$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)" "$START" "$(date -u +%FT%TZ)" $RC "$PY" "$LOG" > $OC/composite/receipts/allocator/d1-omni1d1-gpu$g-claude.json
done
status "d1 end rc=$RC"
[ $RC -eq 0 ] || exit $RC

$PY $L show --checkpoint-id $BASE_ID > $Q/d1/base_record.json
$PY $T/d1_delta_recipe.py --delta-dir "$DELTA" --base-record $Q/d1/base_record.json --base-checkpoint-id $BASE_ID >> "$LOG" 2>&1 \
  || { status "FAIL: d1_delta_recipe"; exit 2; }
REG=(register --checkpoint-id $DELTA_ID --stage distill --directory "$DELTA" --delta-base $BASE_ID
     --parent-directory $PARENT_HF --parent-code Q38-27BU
     --recipe-command "$Q/d1_launch.sh (d1_trunk_recovery ${D1_RUN_ARGS[*]})")
$PY $L "${REG[@]}" >> "$LOG" 2>&1 || { status "FAIL: register dry run"; exit 2; }
$PY $L "${REG[@]}" --confirm >> "$LOG" 2>&1 || { status "FAIL: register --confirm"; exit 2; }
status "registered $DELTA_ID on $BASE_ID"
$PY $L delta-compose --checkpoint-id $DELTA_ID --deep --plan-out $Q/d1/OMNI1D1_plan.json >> "$LOG" 2>&1 \
  || { status "FAIL: delta-compose --deep"; exit 2; }
status "done: plan $Q/d1/OMNI1D1_plan.json (next: R2 clone OMNI1D1.clone-pre-D2 + review package, not done here)"
