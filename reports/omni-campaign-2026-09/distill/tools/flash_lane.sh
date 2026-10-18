#!/bin/bash
# flash_lane.sh -- the whole Qwen3.8-Flash-Next lane on the four P40s, unattended:
#   (1) wait until flash_stage_chain.sh has the GGUF verified on root;
#   (2) flash_toolcall_gen.sh -- Flash Next writes the tool-call dataset (schemas -> plan -> 3M-token run -> finalize);
#   (2b) toolcall_selfgen.py export -- the sealed run -> $OC/distill/corpus/toolcall_flash_v1/toolcall-flash-v1.safetensors
#        (input_ids / lengths / target_mask, rendered with the Q38 tokenizer = Flash Next's);
#   (3) flash_score_queue.sh  -- Flash Next teacher logits over that export (G0), the K3 pass corpora, tool-bfcl-native,
#       text_clean, text.
#   setsid nohup bash flash_lane.sh > /home/cereal/omni-campaign/distill/flash_lane.log 2>&1 < /dev/null &
# Stop: kill -TERM the step's own driver (each is resumable), then this script.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
OK_MARK=/home/cereal/cleanup-20261017/flash_root.sha256.ok
SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6
GW=$OC/distill/toolcall/flash-gen-2026-10-18
SW=$OC/distill/flash_score
PY=/home/cereal/model-surgery-lab/venv/bin/python
TCC=$OC/distill/corpus/toolcall_flash_v1/toolcall-flash-v1.safetensors
utc() { date -u +%FT%TZ; }
echo "$(utc) flash lane start"
until [ -s "$OK_MARK" ] && grep -q "$SHA" "$OK_MARK"; do
  pgrep -f flash_stage_chain.sh > /dev/null || { echo "$(utc) STOP: flash_stage_chain.sh is not running and the GGUF is not verified on root"; tail -5 /home/cereal/cleanup-20261017/flash_stage_chain.log; exit 2; }
  sleep 300
done
echo "$(utc) GGUF verified on root; tool-call generation"
mkdir -p "$GW" "$SW"
# the root copy's sha256 was just checked by flash_stage_chain.sh: seed the generator's marker instead of re-hashing 89 GB
[ -s "$GW/gguf.sha256.ok" ] || echo "$SHA" > "$GW/gguf.sha256.ok"
bash "$T/flash_toolcall_gen.sh" > "$GW/driver.log" 2>&1 < /dev/null
rc=$?
echo "$(utc) flash_toolcall_gen.sh exit $rc: $(tail -2 $GW/status 2>/dev/null | tr '\n' ' ')"
if [ -s "$GW/run/manifest.json" ] && [ ! -s "$TCC" ]; then
  PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src "$PY" "$T/toolcall_selfgen.py" export --out-dir "$GW/run"     --dest "$TCC" > "$GW/export.log" 2>&1
  rc=$?
  echo "$(utc) export exit $rc: $(tail -1 $GW/export.log)"
fi
bash "$T/flash_score_queue.sh" > "$SW/queue.log" 2>&1 < /dev/null
rc=$?
echo "$(utc) flash_score_queue.sh exit $rc: $(tail -2 $SW/status 2>/dev/null | tr '\n' ' ')"
