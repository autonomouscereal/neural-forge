#!/usr/bin/env bash
# test_d2_launch.sh -- d2_launch.sh's trainer exit-code recovery (1.1.2, adversarial review 2026-10-15).
#   bash test_d2_launch.sh [script under test, default $Q/d2_launch.sh]
# The streamed engine runs the trainer under `python -m torch.distributed.run`, which exits 1 for ANY failing worker,
# so the queue's case on 4 (K3 deferred) / 3 (halted) / 2 (refused) must read the trainer's own lines.  Real torchrun
# (CPU only: CUDA hidden, nice 15, two workers that never initialise a process group) runs stand-in workers that print
# exactly the trainer's outcome lines and exit with its codes.  Nothing touches a card, a lock or the queue dir.
#   rc0        a clean exit stays 0
#   deferred   worker exit 4 + "TRAIN DEFERRED ..." -> torchrun rc 1 (NEGATIVE CONTROL: the raw code 1.1.1 used) -> 4
#   halted     exit 3 + "TRAIN HALTED ..." -> 3;  refused  exit 2 + "REFUSED: ..." on stderr -> 2
#   crash      an exception (no outcome line) -> stays 1 (the retry path)
#   wired      train() maps through trainer_rc; a k3-seal refusal stops the queue (exit 2), only rc 10 waits
S=${1:-/home/cereal/omni-campaign/distill/queues/d2_launch.sh}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
TD=/mnt/lexar/omni-campaign/scratch/d2launch-test-$$
mkdir -p "$TD"
pass=0; fail=0
check() { if [ "$2" = 0 ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1 :: $3"; fi; }
eval "$(sed -n '/^trainer_rc() {/,/^}/p' "$S")"
type trainer_rc > /dev/null 2>&1; check "trainer_rc defined in $S" $? "no trainer_rc() in the script"

run_tr() {  # $1 tag $2 python body -> runs it under torchrun with 2 workers; prints the raw rc; log in $TD/$1.log
  printf '%s\n' "$2" > "$TD/$1.py"
  ( cd "$TD" && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 300 nice -n 15 "$PY" -m torch.distributed.run --standalone \
      --nproc_per_node 2 "$TD/$1.py" > "$TD/$1.log" 2>&1 ); echo $?
}
W='import os, sys
rank = int(os.environ.get("RANK", "0"))'

raw=$(run_tr rc0 "$W
sys.exit(0)"); got=$(trainer_rc "$raw" "$TD/rc0.log")
check "rc0: raw $raw -> $got" $([ "$raw" = 0 ] && [ "$got" = 0 ]; echo $?) "raw $raw got $got"

raw=$(run_tr deferred "$W
if rank == 0: print('TRAIN DEFERRED at step 305, tokens 20,001,234: k3-pass-0: sealed marker x does not exist yet', flush=True)
sys.exit(4)"); got=$(trainer_rc "$raw" "$TD/deferred.log")
check "deferred NEGATIVE CONTROL: torchrun hides the trainer's 4 (raw $raw, what 1.1.1 dispatched on)" $([ "$raw" = 1 ]; echo $?) "raw $raw"
check "deferred: raw $raw -> $got" $([ "$got" = 4 ]; echo $?) "got $got"

raw=$(run_tr halted "$W
if rank == 0: print('TRAIN HALTED verdict halted_nonfinite at step 9: x', flush=True)
sys.exit(3)"); got=$(trainer_rc "$raw" "$TD/halted.log")
check "halted: raw $raw -> $got" $([ "$raw" = 1 ] && [ "$got" = 3 ]; echo $?) "raw $raw got $got"

raw=$(run_tr refused "$W
print('REFUSED: source x: exclude_rows removed every row', file=sys.stderr, flush=True)
sys.exit(2)"); got=$(trainer_rc "$raw" "$TD/refused.log")
check "refused: raw $raw -> $got" $([ "$raw" = 1 ] && [ "$got" = 2 ]; echo $?) "raw $raw got $got"

raw=$(run_tr crash "$W
print('STEP {\"step\": 3}', flush=True)
raise RuntimeError('CUDA error: an illegal memory access')"); got=$(trainer_rc "$raw" "$TD/crash.log")
check "crash: raw $raw -> $got (retry path)" $([ "$raw" != 0 ] && [ "$got" = "$raw" ]; echo $?) "raw $raw got $got"

body=$(sed -n '/^train() {/,/^}/p' "$S")
echo "$body" | grep -q 'rc=$(trainer_rc "$raw" "$log")' && echo "$body" | grep -q 'return $rc'
check "wired: train() returns the recovered code" $? "train() does not map through trainer_rc"
grep -q '\[ $krc -eq 10 \] || {.*exit 2; }' "$S"
check "wired: a k3-seal refusal stops the queue; only rc 10 waits" $? "k3-seal loop not as expected"
bash -n "$S"; check "bash -n $S" $?

rm -rf "$TD"
echo "TEST d2_launch: $pass passed, $fail failed"
[ $fail -eq 0 ]
