#!/usr/bin/env bash
# run_p2_memory_proof_p40_2026-10-14.sh -- lane p2p11, Plan Step 5a: prove streamed_kd_trainer 1.3.3's P2 OOM fix and
# its memory model on ONE P40 (GPU1), before any 3090 time is spent.  Run detached:
#   setsid nohup bash run_p2_memory_proof_p40_2026-10-14.sh > $OUT/driver.log 2>&1 < /dev/null &
# Jobs, in order (each prints one result line; receipts in $OUT):
#   A  grad-parity, Qwen3.5-0.8B fp32, kernels torch, batch 2 x 256, with the 1.3.3 chunked path FORCED small
#      (--ffn-chunk-tokens 96 --recompute-tokens 256: 2 row groups, 3 FFN slices per row) + the resident arm with a host
#      hop.  Gate: worst relative difference <= 1e-4 (the 1.3.0 receipt's threshold).  The engine changed, so T(ii)'s
#      parity arm is re-earned here.
#   B  memory-probe, Qwen3.5-0.8B fp16 (2-byte activations = bf16 bytes): the 1.3.2 whole-layer path and three 1.3.3
#      granularities, measured per phase beside memory_model() -- the scaled-shape check of the model.
#   C  memory-probe, a random-weight layer pair at the OMNI1 width (H 5,120, FFN 16,640, one GDN + one gated-GQA layer,
#      the real head rows 248,320) fp16 at 32,768 tokens per micro-batch: the 1.3.2 path (expected to OOM: the P2
#      failure, reproduced on a 24 GB card) and the 1.3.3 cells P2 will run (mb8 / mb4 x 4,096, mb1 x 32,768).
# The P40 runs torch kernels (no triton on sm61): its GDN core is the fp32-upcast torch rule, heavier than fla on
# the 3090, so C's GDN phases are an upper bound for the 3090; the FFN, gated-GQA and head phases are byte-exact.
# v2 (2026-10-14 15:10Z): every probe runs --sdpa-gqa repeat.  Run 1 (receipts memory_probe_B1..B4) put the P40's attention
# on SDPA's MATH kernel: with no mask transformers passes enable_gqa, which only flash / math accept, and the P40 has no
# flash -> [heads, L, L] tensors (9.36 GiB for a 0.8B 8,192-token attention backward; OOM at 16,384).  The 3090 ran the
# 1.3.2 recompute through attention at 32k and died later in its FFN, so it was not on the math kernel; repeat puts the
# P40 on the memory-efficient kernel, which makes its attention phases comparable.  Grad parity runs in both modes.
# Card rule: P40_GPU (default GPU3, the card the toolcall lane's 14:28Z claim leaves for tests), announced in
# $Q/p40.claims; refuses if any other lane's claim line names that card, or if it holds >= 2,000 MiB.  Host: <= ~15 GB.
set -uo pipefail
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
D=${TRAINER_DIR:-/mnt/lexar/omni-campaign/scratch/p2p11/mirror/distill/tools}
T=$D/streamed_kd_trainer.py
WANT_T=${WANT_T:?set WANT_T to the trainer sha256 under test}
OUT=${OUT:-$OC/distill/tps/p2p11}
PY=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
GPU=${P40_GPU:-3}
mkdir -p $OUT
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
HAVE=$(sha256sum $T | cut -c1-64)
[ "$HAVE" = "$WANT_T" ] || { echo "refuse: $T sha256 $HAVE != $WANT_T"; exit 3; }
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')
[ "${used:-99999}" -lt 2000 ] || { echo "refuse: GPU$GPU holds ${used} MiB"; exit 4; }
#: the card list is the token after CLAIM ("<ts> CLAIM GPU1,GPU2 ...") or the second field ("<ts> GPU3 lane=..."); free text
#: later on the line ("GPU3 NOT claimed") is not a claim
foreign=$(awk -v g="GPU$GPU" '!/lane=p2p11/ && !/released/ {f = ($2 == "CLAIM") ? $3 : $2; n = split(f, a, ","); for (i = 1; i <= n; i++) if (a[i] == g) print}' $Q/p40.claims 2>/dev/null | tail -1)
[ -z "$foreign" ] || { echo "refuse: GPU$GPU is claimed by another lane: $foreign"; exit 4; }
echo "$(stamp) GPU$GPU lane=p2p11 owner=claude job=p2-memory-proof trainer=${WANT_T:0:16} pid=$$ log=$OUT/driver.log" >> $Q/p40.claims
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$D PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
echo "$(stamp) start: trainer $T ${WANT_T:0:16}; $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i $GPU)"


# v3 (15:35Z): STAGES=C runs only the OMNI1-width cells, one step each (the P40's memory-efficient attention made
# R1 122 s / step at 8,192 tokens; two steps at 32,768 x 24 heads would take hours).  AdamW moments are in the
# model's static set analytically; one step measures every phase.  R5 (0.8B x 32,768) was dropped for the same reason.
STAGES=${STAGES:-A R C}
[[ " $STAGES " == *" A "* ]] && for G in auto repeat; do
  echo "== A grad-parity sdpa-gqa $G $(stamp)"
  nice -n 15 $PY $T grad-parity --model $M08 --dtype float32 --kernels torch --trunk-ffn-width 1024 --device cuda:0 \
    --batch 2 --seq-len 256 --ffn-chunk-tokens 96 --recompute-tokens 256 --also-resident --sdpa-gqa $G \
    --receipt $OUT/grad_parity_q35_08b_p40_v133_chunked_gqa-$G.json > $OUT/A_grad_parity_gqa-$G.log 2>&1
  echo "A-$G rc=$? $(grep -E '^GRAD-PARITY|^ARM' $OUT/A_grad_parity_gqa-$G.log | tr '\n' ' ')"
done

probe() {  # probe <tag> <probe args...>   (fp16 = the bf16 byte sizes; repeat = the memory-efficient SDPA kernel)
  local tag=$1; shift
  echo "== $tag $(stamp)"
  nice -n 15 $PY $T memory-probe --device cuda:0 --dtype float16 --kernels torch --sdpa-gqa repeat "$@" \
    --receipt $OUT/memory_probe_$tag.json > $OUT/$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^MEMORY-PROBE' $OUT/$tag.log)"
  grep -E '^  PHASE' $OUT/$tag.log
}
Q08=(--model $M08 --trunk-ffn-width 1024)
if [[ " $STAGES " == *" R "* ]]; then
probe R1_q08_L8192_whole    "${Q08[@]}" --batch 1 --seq-len 8192 --ffn-chunk-tokens 0 --recompute-tokens 0 --head-chunk-positions 1024
probe R2_q08_L8192_ffn1024  "${Q08[@]}" --batch 1 --seq-len 8192 --ffn-chunk-tokens 1024 --recompute-tokens 8192
probe R3_q08_mb4_L2048      "${Q08[@]}" --batch 4 --seq-len 2048 --ffn-chunk-tokens 1024 --recompute-tokens 2048
probe R4_q08_L16384_ffn2048 "${Q08[@]}" --batch 1 --seq-len 16384 --ffn-chunk-tokens 2048 --recompute-tokens 16384
probe R5_q08_L32768_ffn4096 "${Q08[@]}" --batch 1 --seq-len 32768 --ffn-chunk-tokens 4096 --recompute-tokens 4096
fi

SYN=(--synthetic-config $M27 --synthetic-ffn 16640 --synthetic-layer-types linear_attention,full_attention --trunk-ffn-width 4352 --steps ${C_STEPS:-1})
if [[ " $STAGES " == *" C "* ]]; then
probe C1_omni1w_L32768_whole_v132 "${SYN[@]}" --batch 1 --seq-len 32768 --ffn-chunk-tokens 0 --recompute-tokens 0 --head-chunk-positions 1024 --steps 1
probe C2_omni1w_mb8_L4096   "${SYN[@]}" --batch 8 --seq-len 4096
probe C3_omni1w_mb4_L4096   "${SYN[@]}" --batch 4 --seq-len 4096
probe C4_omni1w_mb1_L32768  "${SYN[@]}" --batch 1 --seq-len 32768
fi
echo "$(stamp) GPU$GPU lane=p2p11 job=p2-memory-proof released" >> $Q/p40.claims
echo "$(stamp) done"
