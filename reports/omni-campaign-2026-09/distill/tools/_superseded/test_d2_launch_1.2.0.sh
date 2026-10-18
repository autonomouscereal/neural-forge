#!/usr/bin/env bash
# test_d2_launch.sh -- d2_launch.sh's trainer exit-code recovery (1.1.2) and the 1.2.0 memory-fit / register-check wiring
# (adversarial review 2026-10-15).
#   bash test_d2_launch.sh [script under test, default $Q/d2_launch.sh]
# The streamed engine runs the trainer under `python -m torch.distributed.run`, which exits 1 for ANY failing worker,
# so the queue's case on 4 (K3 deferred) / 3 (halted) / 2 (refused) must read the trainer's own lines.  Real torchrun
# (CPU only: CUDA hidden, nice 15, two workers that never initialise a process group) runs stand-in workers that print
# exactly the trainer's outcome lines and exit with its codes.  Nothing touches a card, a lock or the queue dir.
#   rc0        a clean exit stays 0
#   deferred   worker exit 4 + "TRAIN DEFERRED ..." -> torchrun rc 1 (NEGATIVE CONTROL: the raw code 1.1.1 used) -> 4
#   halted     exit 3 + "TRAIN HALTED ..." -> 3;  refused  exit 2 + "REFUSED: ..." on stderr -> 2
#   crash      an exception (no outcome line) -> stays 1 (the retry path)
#   wired      train() maps through trainer_rc
#   k3_wait    (1.2.0) rc 10 waits, a k3-seal refusal (2) stops at once, any other rc is a crash retried $RETRIES times
#   1.2.0      knob_args turns a fit's knobs into trainer flags (none for a fit without knobs); the run reads micro /
#              accumulation from $D2/memory_fit.json BEFORE the row plan; COMMON carries the knobs, the long rung its
#              own; register-check and site-check sit between the DRY RUN and --confirm, pinned by sha
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
# k3_wait (1.2.0) with a scripted `plan k3-seal` (a sequence of exit codes), no sleeping, in a subshell
eval "$(sed -n '/^k3_wait() {/,/^}/p' "$S")"
type k3_wait > /dev/null 2>&1; check "k3_wait defined in $S" $? "no k3_wait() in the script"
k3_run() {  # $1 the k3-seal exit codes in order -> "<k3_wait rc> <k3-seal calls>"
  ( K3SEQ=($1); n=0; : > "$TD/k3calls"
    plan() { local r=${K3SEQ[$n]:-10}; n=$((n + 1)); echo $n > "$TD/k3calls"; return $r; }
    status() { :; }; sleep() { :; }
    LOGS=$TD RETRIES=3 K3_POLL_S=0 SHORT_SAFETY=x
    k3_wait; echo "$? $(cat "$TD/k3calls")" )
}
got=$(k3_run "10 10 0"); check "k3_wait: waits on 10, returns 0 once sealed ($got)" $([ "$got" = "0 3" ]; echo $?) "got $got"
got=$(k3_run "10 2"); check "k3_wait: a k3-seal REFUSAL (2) stops at once ($got)" $([ "$got" = "2 2" ]; echo $?) "got $got"
got=$(k3_run "1 10 1 0"); check "k3_wait: a crash (1) is retried, not fatal ($got)" $([ "$got" = "0 4" ]; echo $?) "got $got"
got=$(k3_run "1 1 1 1 0"); check "k3_wait: RETRIES+1 crashes stop it ($got)" $([ "$got" = "2 4" ]; echo $?) "got $got"
grep -q '^       k3_wait || exit 2$' "$S"; check "wired: the short run's exit 4 goes through k3_wait" $? "k3_wait not called on exit 4"
bash -n "$S"; check "bash -n $S" $?

# ---- 1.2.0
PY=$PY eval "$(sed -n '/^knob_args() {/,/^}/p' "$S")"
printf '%s' '{"knobs": {"ffn_chunk_tokens": 2048, "recompute_tokens": 4096, "head_chunk_mib": 128.0, "head_chunk_positions": 0}, "long_knobs": null}' > "$TD/fit.json"
got=$(knob_args "$TD/fit.json" knobs)
check "knob_args: the P2 retry knobs -> flags" $([ "$got" = "--ffn-chunk-tokens 2048 --recompute-tokens 4096 --head-chunk-mib 128.0 --head-chunk-positions 0" ]; echo $?) "got '$got'"
got=$(knob_args "$TD/fit.json" long_knobs)
check "knob_args: no long knobs -> no flags (trainer defaults)" $([ -z "$got" ]; echo $?) "got '$got'"
line() { grep -n -m1 -F -- "$1" "$S" | cut -d: -f1; }
fit=$(line 'FIT=$D2/memory_fit.json'); rows=$(line 'ROWS_PER_STEP=$((MB * ACC * WORLD))'); dc=$(line 'data-check --data-config "$DATA"')
mb=$(line 'MB=$($PY -c '"'"'import json,sys; print(json.load(open(sys.argv[1]))["micro_batch"])'"'"' "$FIT")')
check "wired: micro / acc from the fit, before the row plan and the data-check" $([ -n "$fit" ] && [ -n "$mb" ] && [ "$fit" -lt "$mb" ] && [ "$mb" -lt "$rows" ] && [ "$rows" -lt "$dc" ]; echo $?) "fit $fit mb $mb rows $rows data-check $dc"
grep -q -- '--out "$FIT"' "$S" && grep -q 'if \[ ! -s "$FIT" \]' "$S"
check "wired: the fit is written once (a resume keeps its micro-batch)" $? "no write-once guard"
grep -q -- '--log-every 1 $KNOBS)' "$S" && grep -q -- '--total-steps "$LSTEPS" $LKNOBS)' "$S"
check "wired: COMMON carries the short knobs, the long rung its own" $? "knobs not in COMMON / long COMMON"
dry=$(line 'register_dryrun.log" 2>&1'); rc1=$(line 'd2_register_check.py log'); rc2=$(line 'd2_register_check.py sites'); conf=$(line '"${ARGS[@]}" --confirm')
check "wired: register-check and site-check between the DRY RUN and --confirm" $([ -n "$dry" ] && [ -n "$rc1" ] && [ -n "$rc2" ] && [ "$dry" -lt "$rc1" ] && [ "$rc1" -lt "$rc2" ] && [ "$rc2" -lt "$conf" ]; echo $?) "dry $dry log $rc1 sites $rc2 confirm $conf"
grep -qE '^WANT_RCHK=\$\{WANT_RCHK:-[0-9a-f]{64}\}' "$S" && grep -qE '^WANT_PLAN=\$\{WANT_PLAN:-[0-9a-f]{64}\}' "$S" && grep -qF '= "$WANT_RCHK" ]' "$S"
check "pinned: d2_plan and d2_register_check by sha" $? "WANT_PLAN / WANT_RCHK not a sha or not compared"

rm -rf "$TD"
echo "TEST d2_launch: $pass passed, $fail failed"
[ $fail -eq 0 ]
