#!/usr/bin/env bash
# End-to-end driver: envelope -> parity -> stop ONLY the server this script started.
#
# The envelope run keeps its server alive so parity can reuse it, so the shutdown happens
# outside the envelope receipt. It is therefore done BY PID, read from the receipt's
# `server_pid`, and only after /proc/<pid>/cmdline is confirmed to be our own binary on our
# own port. The previous version ended with
#     pkill -f "llamacpp/bin/llama-server .*--port 18095"
# which is a pattern kill on a box where two peer lanes run their own llama-servers. That
# particular pattern could not have matched them (they run /app/llama-server inside
# containers, on other ports) but a pattern kill is the wrong instrument here: it decides
# what to kill from a string instead of from the pid we started.
#
# Because this driver passes --keep-server, an envelope receipt it produced records
# teardown {"stopped": false, "reason": "--keep-server"} and the shutdown appears in THIS
# script's log instead. A receipt showing teardown stopped:true was NOT produced by this
# driver -- it came from a direct envelope_bench.py call. Read the two together.
#
#   run_e2e_0p8b.sh [TAG] [GPU_UUID] [GPU_INDEX] [PORT]
#
set -u
cd /home/cereal/omni-campaign/export
VENV=/home/cereal/model-surgery-lab/venv/bin/python
TAG=${1:-Q35-08B-Q4KM-gpu0-q4kv-e2e}
GPU_UUID=${2:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}   # GPU0, RTX 3090
GPU_INDEX=${3:-0}
PORT=${4:-18096}
MODEL=/home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf
RECEIPT="envelope/envelope-${TAG}.json"

# Port guard. envelope_bench now aborts if the child dies while a stranger answers the
# port, but the cheaper move is not to pick an occupied port at all: 18080 and 18081 are
# peer-owned on this box.
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "PORT ${PORT} IS ALREADY LISTENING; refusing to start. Pass a free port."
  exit 4
fi

echo "### ENVELOPE $(date -u +%H:%M:%S)  gpu${GPU_INDEX} port ${PORT}"
timeout 3000 $VENV envelope_bench.py --tag "${TAG}" \
  --model "${MODEL}" \
  --runner host --gpu-uuid "$GPU_UUID" --gpu-index "$GPU_INDEX" \
  --kv q4_0 --ctx 262144 --port ${PORT} --bind 127.0.0.1 \
  --prefill-points 4096,32768 --needle-points 32768 --needle-depths 0.1,0.9 \
  --decode-tokens 64 --load-timeout 600 --request-timeout 900 --keep-server

echo "### PARITY $(date -u +%H:%M:%S)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" timeout 2400 $VENV parity_check.py \
  --tag "${TAG}-parity" --server-url http://127.0.0.1:${PORT} \
  --gguf-path "${MODEL}" \
  --hf-dir /home/cereal/models/Qwen3.5-0.8B --hf-device cuda:0 --hf-dtype bfloat16 \
  --out-dir /home/cereal/omni-campaign/export/parity

echo "### STOP OUR OWN SERVER, BY PID $(date -u +%H:%M:%S)"
PID="$($VENV -c "import json,sys; print(json.load(open('${RECEIPT}')).get('server_pid') or '')" 2>/dev/null)"
if [ -z "${PID}" ]; then
  echo "no server_pid in ${RECEIPT}; stopping NOTHING (a pattern kill is not a fallback)"
elif [ ! -r "/proc/${PID}/cmdline" ]; then
  echo "pid ${PID} is already gone"
else
  CMD="$(tr '\0' ' ' < /proc/${PID}/cmdline)"
  case "${CMD}" in
    *"llamacpp/bin/llama-server"*"--port ${PORT}"*)
      kill -TERM "${PID}" && echo "sent SIGTERM to ${PID}: ${CMD}" ;;
    *)
      echo "pid ${PID} is NOT our server (cmdline: ${CMD}); stopping nothing" ;;
  esac
fi
echo "### DONE $(date -u +%H:%M:%S)"
