#!/bin/sh
# C10 D4 design sweep (2026-10-11, Claude): c10_union_install.py >= 1.5.0 --sweep-targets, OPEN LOOP.
# Ten union targets fitted on the same captured rows per trunk layer; writes shared up + per-target down
# under $D/sweep (never a child).  Run one process per trunk-layer range:
#   GPU=0 RANGE=0-21  setsid nohup sh launch_sweep.sh > logs/sweep_0-21.log  2>&1 < /dev/null &
#   GPU=4 RANGE=22-43 setsid nohup sh launch_sweep.sh > logs/sweep_22-43.log 2>&1 < /dev/null &
#   GPU=6 RANGE=44-63 setsid nohup sh launch_sweep.sh > logs/sweep_44-63.log 2>&1 < /dev/null &
# GPU is the nvidia-smi index (check < 2000 MiB first).  Re-running the same command resumes (completed
# layers are skipped).  8 BLAS threads per process (3 processes = 24), nice 15.  --heavy-lock serializes the
# host-RAM-heavy pool-Gram/selection phase across the range processes (a two-K3-layer pool peaks ~75 GB RSS).
OC=/home/cereal/omni-campaign
D=$OC/composite/c10/design_sweep
EF=$OC/import/rq27b/tree_ef1ffb9
K3D=$OC/k3-run-oct09-claude/fold/dense
if [ ! -d $D/k3_dense_1_92 ]; then
  for l in $(seq -w 1 92); do mkdir -p $D/k3_dense_1_92/L$l; ln -sfn $K3D/L$l/ffn.safetensors $D/k3_dense_1_92/L$l/ffn.safetensors; done
fi
TARGETS="legacy=parent/additive:1;a0=trunk/additive:0;a003=trunk/additive:0.03;a01=trunk/additive:0.1;a03=trunk/additive:0.3;b01=trunk/convex:0.1;b025=trunk/convex:0.25;b05=trunk/convex:0.5;g01=trunk/rms:0.1;g03=trunk/rms:0.3"
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $OC
exec nice -n 15 /usr/bin/time -v /home/cereal/model-surgery-lab/venv/bin/python composite/tools/c10_union_install.py \
  --trunk /mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native \
  --parent-trunk /mnt/ai_data/omni-campaign/s0-probe-sep04 \
  --k3-dense $D/k3_dense_1_92 --k3-form E --k3-layers 1-92 \
  --k3-config /mnt/lexar/models/moonshotai/Kimi-K3/config.json \
  --charts $OC/composite/c17/charts --chart-arm ridge --chart-bias-ceiling 0 \
  --evidence $OC/qwen-fold/evidence/omni-evidence-27bu-v1.json \
  --k3-channels 12288 --parent-parameter-count 27781427952 \
  --device cuda:0 --selection-device cuda:0 --solve-device cuda:0 \
  --engine streamed --group-size 3 --chunk 8192 --gram-block 16384 \
  --out $D/sweep --sweep-targets "$TARGETS" --open-loop --trunk-layers ${RANGE:?set RANGE} --min-free-gib 62 \
  --heavy-lock $D/sweep.heavy.lock
