#!/bin/bash
# Re-run both M-B cells under the content-addressed split manifest.
#
# RSTO reuses all three conditioned captures already on disk, so it is minutes.
# RBEH reuses the first (both recipes fold C03 then C14, and the C14 refit is
# below bfloat16 resolution so the capture child is the same object) and cuts
# new captures from its own divergent children.
set -u
cd /home/cereal/nf-mxh
./run-mxh.sh COMP.MB.RSTO M-B RSTO 4
echo "=== RSTO done, starting RBEH ==="
./run-mxh.sh COMP.MB.RBEH M-B RBEH 4
echo "=== both done ==="
