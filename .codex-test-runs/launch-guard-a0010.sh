set -euo pipefail

LAUNCHER=/home/cereal/model-surgery-lab/launchers/launch-lfm2-exact4x-ahb-a0010-policy-resume.sh
GPU0=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561
GPU4=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
CLAUDE_PID=1791554

test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$GPU0"
test "$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)" = "$GPU4"
gpu0_memory=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
gpu4_memory=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$gpu0_memory" -lt 256
test "$gpu4_memory" -lt 256
ps -p "$CLAUDE_PID" -o args= | grep -q '[s]tage2_generate_teacher_corpus.py'
test -z "$(pgrep -f '[r]un_lfm2_dynamic_catalog.py' || true)"

printf 'launch-guard-ok gpu0_memory=%s gpu4_memory=%s claude_pid=%s\n' \
  "$gpu0_memory" "$gpu4_memory" "$CLAUDE_PID"
bash "$LAUNCHER"
