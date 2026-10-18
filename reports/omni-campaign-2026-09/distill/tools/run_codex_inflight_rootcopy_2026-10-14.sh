#!/usr/bin/env bash
# run_codex_inflight_rootcopy_2026-10-14.sh -- lane inflight: stage a read-only copy of Codex's root-owned speech/H3
# native trees so codex_inflight_adopt.py (>= 1.1.0) can adopt them as cereal.
#
# Why: Codex's root containers wrote codex-native-audio-consumer-modality-v2/native-{audio-ce,speech-ce,speech-flow}-v2
# and codex-h3-native-modality-v3/native-h3-v1/*.safetensors as root, mode 0700/0600; cereal cannot read them, so the
# 10-14 adoption of the collection tarball failed with 938 missing tensor references.  This driver runs ONE short
# root container (no network, 2 CPUs, 2 GiB, both Codex dirs mounted read-only) that
#   1. hashes every file of each ORIGINAL tree            -> $STAGE/SOURCE_SHA256SUMS.<dir> (paths relative to <dir>)
#   2. copies the trees                                   -> $STAGE/tree/<dir>
#   3. hands the staging to cereal (chown 1000:1000)
# Nothing in the Codex dirs is written, moved or deleted.  The adopt tool then proves staged == original sums and
# checks Codex's own replay-index / manifest shas.  Gates: MemAvailable >= 40 GiB, refuses an existing $STAGE.
#
# Written 2026-10-14 by Claude (lane inflight).
set -euo pipefail
VERSION="1.0.0"   # 1.0.0 (2026-10-14): first version
D=/home/cereal/omni-campaign/distill
A=$D/codex-native-audio-consumer-modality-v2
H=$D/codex-h3-native-modality-v3
STAGE=${STAGE:-/mnt/lexar/omni-campaign/scratch/codex-inflight-rootcopy-2026-10-14}
IMAGE=${IMAGE:-nginx:alpine}
NAME=claude-inflight-rootcopy-$(date -u +%Y%m%dT%H%M%SZ)

avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
if [ "$avail_kb" -lt $((40 * 1024 * 1024)) ]; then echo "REFUSE: MemAvailable ${avail_kb} kB < 40 GiB"; exit 3; fi
if [ -e "$STAGE" ]; then echo "REFUSE: $STAGE exists"; exit 3; fi
mkdir -p "$STAGE"
echo "rootcopy $VERSION start $(date -u +%FT%TZ) container $NAME image $IMAGE"
rc=0
timeout 900 docker run --rm --name "$NAME" --network none --cpus 2 --memory 2g --memory-swap 2g \
  --label owner=claude --label lane=inflight \
  -v "$A":/src/codex-native-audio-consumer-modality-v2:ro \
  -v "$H":/src/codex-h3-native-modality-v3:ro \
  -v "$STAGE":/dst \
  --entrypoint sh "$IMAGE" -c '
set -eu
for d in codex-native-audio-consumer-modality-v2 codex-h3-native-modality-v3; do
  cd /src/$d
  find . -type f -exec sha256sum {} + | sort -k2 > /dst/SOURCE_SHA256SUMS.$d
  echo "$d files $(find . -type f | wc -l) symlinks $(find . -type l | wc -l) special $(find . ! -type f ! -type d ! -type l | wc -l)" >> /dst/SOURCE_SHAPE.txt
done
cd /src
mkdir /dst/tree
cp -a /src/. /dst/tree/
chown -R 1000:1000 /dst
chmod -R u+rwX /dst
' || rc=$?
echo "rootcopy $VERSION done rc=$rc $(date -u +%FT%TZ)"
cat "$STAGE/SOURCE_SHAPE.txt"
sha256sum "$STAGE"/SOURCE_SHA256SUMS.*
exit $rc
