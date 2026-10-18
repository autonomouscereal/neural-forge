#!/usr/bin/env bash
# Re-measure the IQ4_XS parity receipts through the provenance-carrying parity_check.py
# (EXPORT_LANE_SPEC.md sec 5.3 D4). The shipped receipts borrowed their HF column from
# parity-Q35-08B-Q4KM.json and said nothing about it; only a re-run puts the provenance
# into the artifacts a reader actually has.
#
# It also runs the SAME bank twice against the SAME running server, because the first
# re-run's greedy answers (temperature 0, top_k 1) did not reproduce the earlier ones.
#
# It starts ONE llama-server of its own on a free P40 and stops that same unit at the end.
# It touches nothing it did not start.
set -u

EXPORT=/home/cereal/omni-campaign/export
VENV=/home/cereal/model-surgery-lab/venv/bin/python
GGUF=$EXPORT/gguf/Q35-08B-stock.IQ4_XS.gguf
REF=$EXPORT/parity/parity-Q35-08B-Q4KM.json
PORT=18097
UNIT=oc-parity-iq4xs-reprov
GPU=GPU-ce4a1da0-baa8-3d62-419c-a28f6565610f   # GPU5, Tesla P40, idle
LOG=$EXPORT/logs/parity-iq4xs-reprov.log

mkdir -p "$EXPORT/logs"

if systemctl --user is-active --quiet "$UNIT"; then
  echo "FATAL: unit $UNIT is already active; refusing to touch it" >&2
  exit 2
fi

# Flags match the run EXPORT_LANE_SPEC.md sec 2.3 documents (-c 8192 -fa on, q4_0/q4_0 KV).
# They are not cosmetic: an f16-KV pass over this same file changed 10 of 15 answers under
# identical greedy settings, so a receipt measured under other flags is a different
# measurement, not a re-measurement.
echo "== launching our own llama-server (unit $UNIT, GPU5, IQ4_XS, -c 8192, q4_0/q4_0 KV) =="
systemd-run --user --unit="$UNIT" --collect \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --setenv=CUDA_VISIBLE_DEVICES="$GPU" \
  --setenv=LD_LIBRARY_PATH="$EXPORT/llamacpp/bin" \
  --property=StandardOutput=append:"$LOG" \
  --property=StandardError=append:"$LOG" \
  "$EXPORT/llamacpp/bin/llama-server" -m "$GGUF" \
  -c 8192 -fa on -np 1 -ngl 999 -ctk q4_0 -ctv q4_0 --host 127.0.0.1 --port "$PORT" || exit 3

healthy=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then healthy=1; break; fi
  sleep 2
done
if [ "$healthy" != "1" ]; then
  echo "FATAL: our server never became healthy; log tail:" >&2
  tail -30 "$LOG" >&2
  systemctl --user stop "$UNIT"
  exit 4
fi
echo "== healthy =="

rc=0
echo "== pass 1: raw-prompt parity, HF column borrowed from $REF =="
timeout 900 "$VENV" "$EXPORT/parity_check.py" --tag Q35-08B-IQ4XS \
  --server-url "http://127.0.0.1:$PORT" --gguf-path "$GGUF" --skip-hf \
  --reference "$REF" --out-dir "$EXPORT/parity" || rc=5

echo "== pass 2: identical arguments, same running server (determinism check) =="
timeout 900 "$VENV" "$EXPORT/parity_check.py" --tag Q35-08B-IQ4XS-REPEAT \
  --server-url "http://127.0.0.1:$PORT" --gguf-path "$GGUF" --skip-hf \
  --reference "$REF" --out-dir "$EXPORT/parity" || rc=7

echo "== chat-rendered parity (no chat-rendered HF reference exists: compared stays 0) =="
timeout 900 "$VENV" "$EXPORT/parity_check.py" --tag Q35-08B-IQ4XS-CHAT \
  --server-url "http://127.0.0.1:$PORT" --gguf-path "$GGUF" --skip-hf --chat \
  --hf-dir /home/cereal/models/Qwen3.5-0.8B --out-dir "$EXPORT/parity" || rc=6

echo "== stopping the unit WE started =="
systemctl --user stop "$UNIT" 2>/dev/null
sleep 2
systemctl --user is-active "$UNIT" || echo "unit stopped"

sha256sum "$EXPORT/parity/parity-Q35-08B-IQ4XS.json" \
          "$EXPORT/parity/parity-Q35-08B-IQ4XS-REPEAT.json" \
          "$EXPORT/parity/parity-Q35-08B-IQ4XS-CHAT.json"

echo "== do the two identical passes agree row for row? =="
"$VENV" "$EXPORT/compare_parity_passes.py" \
  "$EXPORT/parity/parity-Q35-08B-IQ4XS.json" \
  "$EXPORT/parity/parity-Q35-08B-IQ4XS-REPEAT.json"
exit $rc
