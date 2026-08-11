#!/bin/bash
# Wait for the M-B RSTO run to exit, then start M-B RBEH on the same GPU.
#
# Sequential on purpose: the /mnt/ai_data share writes at ~20 MB/s and one
# conditioned capture is 8.1 GB, so two runs in parallel halve each other.
# RBEH reuses the first two captures through the shared content-addressed root
# -- both recipes fold C03 then C14, so the partially folded child is the same
# object and build_lfm2_circuit_trace_cache reuses the cache key rather than
# recapturing it.
set -u
while ps -ef | grep -q "[r]un_lfm2_matrix_composition.py --cell-id COMP.MB.RSTO"; do
  sleep 30
done
echo "$(date -u +%H:%M:%S) RSTO finished; starting RBEH"
/home/cereal/nf-mxh/run-mxh.sh COMP.MB.RBEH M-B RBEH 4
