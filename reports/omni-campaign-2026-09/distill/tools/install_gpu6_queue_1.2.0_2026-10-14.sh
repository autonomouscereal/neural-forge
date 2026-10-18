#!/bin/bash
# install_gpu6_queue_1.2.0_2026-10-14.sh -- lane gpu6, plan Step 2b/2c: install queue_gpu6.sh 1.2.0 as the live GPU6
# queue and launch it detached. Run AFTER k3_b2_geometry_diag has written its verdict (the queue reads the B2
# re-baseline ruling file once, at start).
#   WANT_QUEUE_SHA=<sha> WANT_TOOL_SHA=<sha> setsid nohup bash install_gpu6_queue_1.2.0_2026-10-14.sh \
#     > $QL/install-1.2.0.log 2>&1 < /dev/null &        # backups, install, selftest (~4 min), launch
# Backups (nothing deleted): $Q/gpu6.sh -> $Q/gpu6.sh.1.0.0-live-20261013, $Q/gpu6.sh.new -> $Q/gpu6.sh.new.1.1.0,
# the 2026-10-13 B2 queue logs -> $QL/archive-2026-10-13-queue-1.0.0/ (moved, so the new queue never appends to them).
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
Q=$OC/distill/queues
QL=$Q/logs/gpu6
WANT_QUEUE_SHA=${WANT_QUEUE_SHA:?sha256 of the reviewed tools/queue_gpu6.sh}
WANT_TOOL_SHA=${WANT_TOOL_SHA:?sha256 of the reviewed tools/k3_teacher_cache.py}
utc() { date -u +%FT%TZ; }
[ "$(sha256sum $T/queue_gpu6.sh | cut -c1-64)" = "$WANT_QUEUE_SHA" ] || { echo "refuse: tools/queue_gpu6.sh is not $WANT_QUEUE_SHA"; exit 2; }
[ "$(sha256sum $T/k3_teacher_cache.py | cut -c1-64)" = "$WANT_TOOL_SHA" ] || { echo "refuse: tools/k3_teacher_cache.py is not $WANT_TOOL_SHA"; exit 2; }
if pgrep -f "distill/queues/gpu6.sh" > /dev/null; then echo "refuse: a gpu6 queue is running: $(pgrep -af distill/queues/gpu6.sh | head -2)"; exit 2; fi
if pgrep -f "k3_teacher_cache-gpu6.py run" > /dev/null; then echo "refuse: a frozen K3 run is live"; exit 2; fi
[ -e $Q/gpu6.sh.1.0.0-live-20261013 ] || cp -p $Q/gpu6.sh $Q/gpu6.sh.1.0.0-live-20261013
[ -e $Q/gpu6.sh.new ] && mv -n $Q/gpu6.sh.new $Q/gpu6.sh.new.1.1.0
A=$QL/archive-2026-10-13-queue-1.0.0
mkdir -p $A
for f in b2_pass.log b2_parity.log b2_block_compare.log queue.log; do [ -e $QL/$f ] && mv -n $QL/$f $A/; done
cp -p $T/queue_gpu6.sh $Q/gpu6.sh
chmod +x $Q/gpu6.sh
sha256sum $Q/gpu6.sh $Q/gpu6.sh.1.0.0-live-20261013 $Q/gpu6.sh.new.1.1.0 2>/dev/null
ls -la $A
echo "--- selftest (installed copy)"
ST=$(bash $T/test_queue_gpu6.sh $Q/gpu6.sh 2>&1); STRC=$?
echo "$ST" | grep -E "^ST |QUEUE SELFTEST|MISSING"
[ $STRC -eq 0 ] || { echo "refuse: selftest failed; queue NOT launched"; exit 3; }
echo "--- launch $(utc)"
REFREEZE=1 REFREEZE_PROBE=1 START_AT=b2 setsid nohup bash $Q/gpu6.sh > $QL/queue-1.2.0.log 2>&1 < /dev/null &
echo "queue pid $!"
sleep 20
head -14 $QL/queue-1.2.0.log
sha256sum $Q/frozen/*.py
tail -3 $Q/gpu6.status
