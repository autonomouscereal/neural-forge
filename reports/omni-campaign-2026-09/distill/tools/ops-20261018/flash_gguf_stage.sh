#!/bin/bash
# flash_gguf_stage.sh -- copy the Flash Next GGUF NAS -> root SSD (read-only on the NAS, direct I/O both ends so the
# training jobs' page cache is not churned), then sha256 it on root and write the OK mark the Flash lane reads.
# NAS reads are safe (the 10-18 wedge was sustained WRITES into the SMR pool); one reader, no NAS writes.
set -u
SRC="/mnt/ai_data/models/flash-next-strata/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf"
DIR=/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata; DST=$DIR/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf
SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6; SIZE=88882765184
MARK=/home/cereal/cleanup-20261017/flash_root.sha256.ok
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*"; }
mkdir -p "$DIR"
[ "$(df -B1 --output=avail / | tail -1)" -gt $((SIZE + 60000000000)) ] || { say "STOP: root would drop under 60 GB"; exit 2; }
got=$(stat -c %s "$SRC") || { say "STOP: cannot stat $SRC"; exit 3; }
[ "$got" = "$SIZE" ] || { say "STOP: NAS size $got != $SIZE"; exit 3; }
if [ "$(stat -c %s "$DST.part" 2>/dev/null || echo 0)" -lt "$SIZE" ]; then
  have=$(( $(stat -c %s "$DST.part" 2>/dev/null || echo 0) / 16777216 ))
  say "copy from block $have (16 MiB blocks)"
  ionice -c3 nice -n 10 dd if="$SRC" of="$DST.part" bs=16M iflag=direct oflag=direct skip="$have" seek="$have" conv=notrunc status=none
  rc=$?
  [ $rc -eq 0 ] || { say "STOP: dd rc=$rc (rerun resumes)"; exit 4; }
fi
[ "$(stat -c %s "$DST.part")" = "$SIZE" ] || { say "STOP: copied size $(stat -c %s "$DST.part") != $SIZE"; exit 5; }
say "hashing the root copy"
h=$(nice -n 10 sha256sum "$DST.part" | cut -d' ' -f1)
[ "$h" = "$SHA" ] || { say "STOP: root copy sha $h != $SHA (kept as .part for inspection)"; exit 6; }
mv "$DST.part" "$DST"
echo "$SHA  $DST  verified $(utc) (copied from the NAS copy, download-verified 10-18)" > "$MARK"
say "DONE: $DST verified; mark $MARK"
