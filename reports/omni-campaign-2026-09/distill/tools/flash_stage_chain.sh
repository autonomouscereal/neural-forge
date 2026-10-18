#!/bin/bash
# flash_stage_chain.sh -- put Qwen3.8-Flash-Next Strata IQ3_S on the AI server root, unattended and NAS-gentle.
# Cyril 2026-10-17: "[Cyril quote withheld]".
#   setsid nohup bash flash_stage_chain.sh > /home/cereal/cleanup-20261017/flash_stage_chain.log 2>&1 < /dev/null &
# Order (ONE heavy NAS job at a time -- the CIFS share hung under two concurrent writers on 10-17):
#   1. wait for the download (curl to the NAS) to end, then check the NAS file's sha256;
#   2. move omni-campaign/qwen-fold/children (140 GB, cold since 09-03) root -> NAS: rsync with a bandwidth cap and
#      retries, size+list check, checksum dry run must list nothing, then the source becomes a symlink (every old path
#      still resolves; the bytes live on the NAS, verified equal);
#   3. copy the GGUF NAS -> root, check its sha256 on root.  The NAS copy is KEPT (a second verified copy).
# Each step is skipped when its result is already in place, so a rerun resumes.
set -u
C=/home/cereal/cleanup-20261017
NASDIR=/mnt/ai_data/models/flash-next-strata
NAME=RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf
SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6
ROOTDIR=/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata
S=/home/cereal/omni-campaign/qwen-fold/children
D=/mnt/ai_data/backups/ai-server-cleanup-20261017/omni-campaign/qwen-fold/children
BW=${BW:-60000}   # rsync --bwlimit, KiB/s
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*"; }
root_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }

# 1. the download
while pgrep -f download_strata_iq3s.sh > /dev/null || pgrep -f "curl.*Flash-Next" > /dev/null; do
  say "waiting for the download: $(ls -la "$NASDIR"/*/ 2> /dev/null | grep -i gguf | awk '{s+=$5} END {printf "%.1f GB", s/1e9}')"
  sleep 300
done
NASFILE=$(find "$NASDIR" -name "$NAME" 2> /dev/null | head -1)
[ -n "$NASFILE" ] || { say "STOP: no $NAME under $NASDIR after the download (download_strata_iq3s.sh renames the .part only on a sha match; see $NASDIR/download.log)"; exit 2; }
# download_strata_iq3s.sh hashes the .part and renames it only when it matches: its VERIFIED line is the NAS check
if [ ! -s "$C/flash_nas.sha256.ok" ] && grep -q "VERIFIED sha256 $SHA" "$NASDIR/download.log"; then
  echo "$SHA  $NASFILE  (download_strata_iq3s.sh VERIFIED line)" > "$C/flash_nas.sha256.ok"
fi
if [ ! -s "$C/flash_nas.sha256.ok" ]; then
  say "sha256 of $NASFILE (NAS read)"
  got=$(nice -n 10 sha256sum "$NASFILE" | cut -d" " -f1)
  [ "$got" = "$SHA" ] || { say "STOP: NAS file sha $got != $SHA (the download did not finish clean; see $NASDIR/download.log)"; exit 3; }
  echo "$got  $NASFILE" > "$C/flash_nas.sha256.ok"
fi
say "NAS copy verified: $NASFILE"

# 2. the cold fold children -> NAS
if [ ! -L "$S" ]; then
  mkdir -p "$D"
  (cd "$S" && find . -type f -printf "%s\t%p\n" | sort -k2) > "$C/qwen_fold_children.manifest.tsv"
  ok=0
  for attempt in 1 2 3 4 5 6; do
    say "rsync attempt $attempt -> $D (bwlimit $BW KiB/s)"
    nice -n 10 ionice -c3 rsync -a --partial --inplace --timeout=600 --bwlimit="$BW" "$S/" "$D/"
    rc=$?
    say "rsync rc=$rc"
    [ $rc -eq 0 ] && { ok=1; break; }
    sleep 120
  done
  [ $ok = 1 ] || { say "STOP: rsync did not complete in 6 attempts; the source is untouched"; exit 4; }
  (cd "$D" && find . -type f -printf "%s\t%p\n" | sort -k2) > "$C/qwen_fold_children.nas.tsv"
  cmp -s "$C/qwen_fold_children.manifest.tsv" "$C/qwen_fold_children.nas.tsv" || { say "STOP: size/list mismatch; source untouched"; exit 5; }
  say "checksum dry run (must list nothing)"
  DIFF=$(nice -n 10 ionice -c3 rsync -a --checksum --dry-run --itemize-changes "$S/" "$D/")
  rc=$?
  [ $rc -eq 0 ] && [ -z "$DIFF" ] || { say "STOP: content check rc=$rc, differences: $(echo "$DIFF" | head -c 1500)"; exit 6; }
  cp "$C/qwen_fold_children.manifest.tsv" "$D/../children.manifest.tsv"
  mv "$S" "$S.moved-to-nas-20261017" && ln -s "$D" "$S" && rm -rf "$S.moved-to-nas-20261017"
  say "children moved (verified byte-equal), symlinked; root free $(root_gb) GB"
fi

# 3. GGUF NAS -> root
mkdir -p "$ROOTDIR"
DEST=$ROOTDIR/$NAME
if [ ! -s "$C/flash_root.sha256.ok" ]; then
  need=$(( $(stat -c %s "$NASFILE") / 1000000000 + 20 ))
  [ "$(root_gb)" -ge "$need" ] || { say "STOP: root has $(root_gb) GB, need $need GB for the GGUF"; exit 7; }
  say "copy $NASFILE -> $DEST"
  nice -n 10 ionice -c3 rsync -a --partial --inplace --timeout=600 "$NASFILE" "$DEST" || { say "STOP: copy failed rc=$?"; exit 8; }
  got=$(sha256sum "$DEST" | cut -d" " -f1)
  [ "$got" = "$SHA" ] || { say "STOP: root copy sha $got != $SHA"; exit 9; }
  echo "$got  $DEST" > "$C/flash_root.sha256.ok"
fi
say "DONE: $DEST verified ($SHA); NAS copy kept at $NASFILE; root free $(root_gb) GB"
