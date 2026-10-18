#!/usr/bin/env bash
# vmath_q4_copy.sh -- keep a byte-identical copy of the parent's own Q4_K_M for the verified-math pilot.
#
# Research lane (Claude), 2026-10-15.  The toolcall driver (run_toolcall_selfgen_2026-10-14.sh step 8) deletes its Q4
# from declared scratch once its run is exported; the verified-math pilot starts only after that release, so it needs
# its own copy of the same bytes.  The copy goes to a NEW directory in the declared scratch
# (/mnt/lexar/omni-campaign/scratch, STAGE_D_RULINGS "Transient scratch") and is checked against the sha256 that the
# toolcall lane's export-check receipt records; the pilot queue deletes it when the pilot ends.  Read-only on the
# source.  Idle I/O class + nice 19 so D1 / K3 reads are not slowed.
#
#   setsid nohup bash vmath_q4_copy.sh > $R/logs/q4_copy.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
CHECK=$OC/distill/toolcall/receipts/export_check_Q38-27BU-parent_toolcall.json
DEST_DIR=/mnt/lexar/omni-campaign/scratch/research-vmath-2026-10-15
NEED_GB=40
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
mkdir -p "$DEST_DIR"
SRC=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); assert r['parent_unchanged']; print(r['q4']['path'])" "$CHECK") || { stamp "REFUSED: $CHECK unreadable or parent moved"; exit 3; }
WANT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['q4']['sha256'])" "$CHECK")
DEST=$DEST_DIR/$(basename "$SRC")
stamp "source $SRC want sha256 $WANT"
if [ -s "$DEST_DIR/q4_copy_receipt.json" ] && grep -q "\"sha256\": \"$WANT\"" "$DEST_DIR/q4_copy_receipt.json"; then
  stamp "already copied (receipt present)"; exit 0
fi
[ -f "$SRC" ] || { stamp "REFUSED: $SRC is gone (the toolcall driver removed it); re-export with run_toolcall_export_2026-10-14.sh semantics"; exit 4; }
free_gb=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$NEED_GB" ] || { stamp "REFUSED: /mnt/lexar has ${free_gb} GB free, need ${NEED_GB}"; exit 3; }
nice -n 19 ionice -c3 cp -- "$SRC" "$DEST.part" || { stamp "copy failed"; rm -f -- "$DEST.part"; exit 5; }
GOT=$(nice -n 19 ionice -c3 sha256sum "$DEST.part" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ]; then stamp "MISMATCH: copy hashes $GOT, receipt says $WANT"; rm -f -- "$DEST.part"; exit 6; fi
mv -- "$DEST.part" "$DEST"
python3 - "$DEST" "$GOT" "$SRC" "$CHECK" "$DEST_DIR/q4_copy_receipt.json" <<'PY'
import hashlib, json, os, sys, time
dest, sha, src, check, out = sys.argv[1:]
rec = {"schema": "omni-research.vmath-q4-copy.v1", "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "q4": {"path": dest, "sha256": sha, "bytes": os.path.getsize(dest)},
       "copied_from": src, "export_check_receipt": {"path": check, "sha256": hashlib.sha256(open(check, "rb").read()).hexdigest()},
       "rule": "byte-identical to the export-check receipt's q4.sha256; transient, deleted by vmath_queue.sh when the pilot ends"}
json.dump(rec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
print(json.dumps(rec))
PY
stamp "done"
