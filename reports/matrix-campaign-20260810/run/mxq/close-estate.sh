#!/bin/bash
# Wave 13 estate close: backup, merge exactly once, verify, report.
set -u
M=/mnt/ai_data/neural-forge-matrix
W=/home/cereal/nf-mxq
STAMP=$(date -u +%Y%m%dT%H%M%S)
mkdir -p $M/backup
echo "=== 1. BEFORE ==="
sha256sum $M/ledger.jsonl
wc -l < $M/ledger.jsonl
echo "=== 2. BACKUP (sha AND cmp) ==="
cp -p $M/ledger.jsonl $M/backup/ledger.jsonl.pre-wave13-$STAMP
cp -p $M/ledger.jsonl $M/wave13/ledger.jsonl.pre-merge-backup
for b in $M/backup/ledger.jsonl.pre-wave13-$STAMP $M/wave13/ledger.jsonl.pre-merge-backup; do
  sha256sum $b
  cmp $M/ledger.jsonl $b && echo "cmp OK: $b"
done
echo "=== 3. MERGE (exactly once) ==="
if [ -f $W/.MERGED_ONCE ]; then
  echo "REFUSED: $W/.MERGED_ONCE exists; the wave-13 shards are already merged"
  exit 0
fi
SHARDS=$(ls $W/ledger-*.jsonl 2>/dev/null)
echo "shards: $SHARDS"
python3 $W/merge_shards_mxq.py $M/ledger.jsonl $SHARDS || exit 1
touch $W/.MERGED_ONCE
echo "=== 4. AFTER ==="
sha256sum $M/ledger.jsonl
wc -l < $M/ledger.jsonl
echo "=== 5. PREFIX RE-VERIFY (independent) ==="
python3 - "$M/ledger.jsonl" "$M/wave13/ledger.jsonl.pre-merge-backup" <<"PY"
import sys
after=open(sys.argv[1],"rb").read(); before=open(sys.argv[2],"rb").read()
print("before bytes", len(before), "after bytes", len(after), "appended", len(after)-len(before))
print("prefix holds:", after.startswith(before))
PY
