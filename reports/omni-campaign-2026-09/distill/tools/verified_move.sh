#!/bin/bash
# verified_move.sh SRC DEST -- move a cold directory to another filesystem WITHOUT destroying anything:
# rsync (bandwidth cap, retries) -> size+file-list equality -> checksum dry run lists nothing -> only then the source
# is replaced by a symlink to DEST, so every old path still resolves.  Cyril 2026-10-17: "don't delete anything
# destructively".  Run ONE such job against the NAS at a time (the CIFS share hung under two writers on 10-17).
# Exit 0 = moved and verified (or already a symlink); any other exit leaves the source untouched.
set -u
S=${1:?source dir}; D=${2:?dest dir}
BW=${BW:-60000}   # KiB/s
LOGDIR=${LOGDIR:-/home/cereal/cleanup-20261017}
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*"; }
[ -L "$S" ] && { say "$S is already a symlink -> $(readlink "$S")"; exit 0; }
[ -d "$S" ] || { say "STOP: $S is not a directory"; exit 2; }
if lsof +D "$S" > /dev/null 2>&1; then say "STOP: files under $S are open"; exit 3; fi
tag=$(echo "$S" | tr '/' '_')
mkdir -p "$D" "$LOGDIR"
(cd "$S" && find . -type f -printf "%s\t%p\n" | sort -k2) > "$LOGDIR/$tag.src.tsv"
say "move $S -> $D: $(wc -l < "$LOGDIR/$tag.src.tsv") files, $(du -sh "$S" | cut -f1)"
ok=0
for attempt in 1 2 3 4 5 6; do
  nice -n 10 ionice -c3 rsync -a --partial --inplace --timeout=600 --bwlimit="$BW" "$S/" "$D/"
  rc=$?
  say "rsync attempt $attempt rc=$rc"
  [ $rc -eq 0 ] && { ok=1; break; }
  sleep 120
done
[ $ok = 1 ] || { say "STOP: rsync did not complete; source untouched"; exit 4; }
(cd "$D" && find . -type f -printf "%s\t%p\n" | sort -k2) > "$LOGDIR/$tag.dst.tsv"
cmp -s "$LOGDIR/$tag.src.tsv" "$LOGDIR/$tag.dst.tsv" || { say "STOP: size/list mismatch; source untouched"; exit 5; }
DIFF=$(nice -n 10 ionice -c3 rsync -a --checksum --dry-run --itemize-changes "$S/" "$D/")
rc=$?
{ [ $rc -eq 0 ] && [ -z "$DIFF" ]; } || { say "STOP: checksum check rc=$rc: $(echo "$DIFF" | head -c 1500)"; exit 6; }
cp "$LOGDIR/$tag.src.tsv" "$D.manifest.tsv"
moved="$S.moved-$(date -u +%Y%m%d)"
mv "$S" "$moved" || { say "STOP: could not rename $S"; exit 7; }
if ln -s "$D" "$S" 2> /dev/null; then
  rm -rf "$moved"
  say "DONE: $S -> $D (verified byte-equal, symlinked)"
else
  # exFAT (the Lexar drive) has no symlinks: the verified copy is the canonical location; a note keeps the old path's answer
  printf '{"moved_to": "%s", "verified": "rsync list + checksum dry run equal", "manifest": "%s", "at_utc": "%s"}
'     "$D" "$D.manifest.tsv" "$(utc)" > "$S.MOVED.json"
  rm -rf "$moved"
  say "DONE: $S -> $D (verified byte-equal; no symlinks on this filesystem, note $S.MOVED.json)"
fi
