#!/bin/bash
# run_k3cache_v130_parity_gpu3_2026-10-14.sh -- lane k3accel: k3_teacher_cache 1.3.0 (--accel on) vs the FROZEN 1.2.4
# (2aae4dc3) on GPU3 (P40), identical inputs and geometry, every output tensor hashed by k3_cache_parity.
#   per window: A = frozen 1.2.4, B = 1.3.0, C = 1.3.0 --stage-timers (stage split; its digests must match too)
#   compare A-B and A-C (BIT-IDENTICAL or DIFFER, exit 3); NEG = 1.3.0 with +1 ulp on prefix[0,0] after the window's
#   first layer: compare A-NEG must report DIFFER (the negative control)
# Windows (tag:layers:rows:truncate:synthetic_blocks), real rows of corpus/k3 pass-0, --group-tokens 65536,
# --row-chunk-* 1 --token-chunk 1024 (queue_gpu6.sh 1.3.0's pass geometry):
#   w1   L84-85  16 x 4096 (T trims to 3,997)  7 synthetic blocks: KDA + the L84 block push (7 -> 8), one 65,536 group
#   w2   L83     32 x 2048 truncated           7 synthetic blocks: MLA (T=4096 MLA does not fit a P40: math SDPA)
#   w3   L00-03  64 x 2048 truncated           real from the embedding: L00 push + dense MLP, KDA L01-02, MLA L03,
#                                              TWO row groups
#   w4   L84-85  16 x 4096, --group-tokens 16384 = FOUR row groups, 7 synthetic blocks: the multi-group MoE
#                                              (--moe-group-batch auto shares one dequant per expert across groups)
#   b2geo L83-85 the B2 geometry (k3_review + k3_review_native, --no-trim, --group-tokens 32768: 45,056 positions in
#                                              2 groups), 7 synthetic blocks: the P40 against GPU6's own B2 log
#   mini L00-11   2 x 2048 truncated           real: 4,096 tokens through 12 layers (3 MLA, 9 KDA, the L00 push)
#   head61 L61-92 + HEAD, the first 8 rows of k3_review from the fold engine's OWN teacher pair entering L61
#                 (before_L61.safetensors, the D2 layout): 32 real layers with the real 5-block stack (pushes at L72,
#                 L84), MLA + KDA, and every head output tensor (topk ids / logprobs, lse, target_logprob, rows)
# Each run starts cold: the window's shards are dropped from the page cache (fadvise DONTNEED) first -- unless
# NO_DROP=1 (the page cache is global: a drop also evicts what a concurrent pass on another card had prefetched).
# Writes ONLY under /mnt/lexar/omni-campaign/scratch/k3accel/.  Claims GPU3 in $Q/p40.claims and releases it.
# Host-RAM rule: start at MemAvailable >= START_GIB (60); the monitor SIGTERMs a run below 40 GiB.  nice 15, ionice 7.
set -u
OC=/home/cereal/omni-campaign; Q=$OC/distill/queues; T=$OC/distill/tools
S=/mnt/lexar/omni-campaign/scratch/k3accel
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
R=$S/runs/v130-$STAMP
PP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
SRC=/mnt/lexar/models/moonshotai/Kimi-K3
W124=2aae4dc3134dda725e3ee35572584fbf35532e0c4a7a5721a0b0c817fc2c4f9d
OLD=$S/src/k3_teacher_cache-1.2.4.${W124:0:16}.py
W130=$(sha256sum "$T/k3_teacher_cache_v130.py" | cut -c1-64)
NEW=$S/src/k3_teacher_cache-1.3.0.${W130:0:16}.py
START_GIB=${START_GIB:-60}; FLOOR_GIB=${FLOOR_GIB:-40}
mkdir -p "$R" "$S/src"
exec >> "$R/driver.log" 2>&1
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "== k3_teacher_cache 1.3.0 parity on GPU3 $(utc) run dir $R"
[ "$(sha256sum "$OLD" | cut -c1-64)" = "$W124" ] || { echo "REFUSED: $OLD is not the frozen 1.2.4"; exit 2; }
cp -f "$T/k3_teacher_cache_v130.py" "$NEW.tmp" && mv -f "$NEW.tmp" "$NEW" && chmod a-w "$NEW"
[ "$(sha256sum "$NEW" | cut -c1-64)" = "$W130" ] || { echo "REFUSED: copy of 1.3.0 changed"; exit 2; }
echo "A $OLD $W124"; echo "B $NEW $W130"
echo "parity $(sha256sum "$T/k3_cache_parity.py")"; echo "profile $(sha256sum "$T/k3_accel_profile.py")"

# ---- CPU: the 1.3.0 suite (incl. the whole 1.2.4 suite bound to 1.3.0), the parity harness's own tests, and two
# negative controls -- copies of 1.3.0 with a deliberate 1-ulp change, which the bit-identity tests must fail
CPUPY=/home/cereal/model-surgery-lab/venv/bin/python
cpu_tests() {  # $1 dir $2 log, then pytest args
  local dir=$1 log=$2; shift 2
  (cd "$dir" && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$dir timeout 900 nice -n 15 "$CPUPY" -m pytest -o addopts="" \
    -p no:cacheprovider "$@" > "$log" 2>&1); local rc=$?
  echo "$(basename "$log") exit=$rc : $(tail -1 "$log")"
}
export TMPDIR=$S/tests
cpu_tests "$T" "$R/cpu_test_v130.log" test_k3_teacher_cache_v130.py -q
cpu_tests "$T" "$R/cpu_test_parity.log" test_k3_cache_parity.py -q
for NC in weight prefix; do
  N=$S/negctl-$NC-$STAMP; rm -rf "$N"; mkdir -p "$N"
  cp "$T/k3_teacher_cache.py" "$T/test_k3_teacher_cache.py" "$T/test_k3_teacher_cache_v130.py" "$N/"
  "$CPUPY" - "$T/k3_teacher_cache_v130.py" "$N/k3_teacher_cache_v130.py" "$NC" <<'PY'
import sys
src, dst, kind = sys.argv[1:4]
s = open(src).read()
if kind == "weight":    # +1 fp32 ulp on every routed top-k weight
    old = "                    weight = plan[\"flat_w\"][sel]\n"
    new = "                    weight = torch.nextafter(plan[\"flat_w\"][sel], torch.full_like(plan[\"flat_w\"][sel], 2.0))  # NEGATIVE CONTROL\n"
else:                   # +1 ulp on one element of the new prefix of the first group at L02
    old = "                    prefix[g0:g1].copy_(new_prefix)  # card -> host state directly: the same bytes as .to(cpu)\n"
    new = ("                    if layer_idx == 2 and g0 == 0:  # NEGATIVE CONTROL\n"
           "                        new_prefix = new_prefix.clone(); new_prefix.view(-1)[:1].view({2: torch.int16, 4: torch.int32}[new_prefix.element_size()]).add_(1)\n" + old)
assert s.count(old) == 1, kind
open(dst, "w").write(s.replace(old, new))
PY
  cpu_tests "$N" "$R/cpu_negctl_$NC.log" test_k3_teacher_cache_v130.py -q -k "bit_identical or sorted_dispatch or moe_group"
done

quiet=0
while [ $quiet -lt 2 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  [ "$used" -le 2000 ] && quiet=$((quiet + 1)) || quiet=0
  [ $quiet -lt 2 ] && sleep 20
done
wait_ram() {
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge "$START_GIB" ] && break
    echo "$(utc) MemAvailable $avail GiB < $START_GIB; waiting"; sleep 60
  done
}
wait_ram
echo "$(utc) GPU3 ${used} MiB, MemAvailable ${avail} GiB"
echo "$(utc) GPU3 lane=k3accel owner=claude job=k3-v130-parity pid=$$ log=$R/driver.log" >> "$Q/p40.claims"
release() { echo "$(utc) GPU3 lane=k3accel job=k3-v130-parity released" >> "$Q/p40.claims"; }
trap release EXIT
# the monitor starts each run in its own session: a stop of this driver must reach it too
MON=""
trap 'pkill -TERM -f "k3_cache_parity.py digest --tool $S/src"; [ -n "$MON" ] && kill -TERM $MON 2>/dev/null; exit 143' TERM INT

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

echo "== GPU unit tests (graph replay vs eager, a forced mismatch falls back; exact dequant on the card) $(utc)"
# the tools dir carries the oc_cuda_guard conftest (2026-10-14): pytest sees no card unless OC_TEST_ALLOW_GPU=1 with
# an explicit CUDA_VISIBLE_DEVICES (GPU3, claimed above); a skip here would mean the CUDA cases did not run
cd "$T" && OC_TEST_ALLOW_GPU=1 PYTHONPATH=$T timeout 900 nice -n 15 "$PP" -m pytest -o addopts="" -p no:cacheprovider \
  test_k3_teacher_cache_v130.py -q -rs -k "kda_graph or exact_dequant" > "$R/gpu_unit_tests.log" 2>&1
GRC=$?; echo "gpu unit tests exit=$GRC : $(tail -1 "$R/gpu_unit_tests.log")"
grep -q " skipped" "$R/gpu_unit_tests.log" && echo "WARNING gpu unit tests skipped a case: the CUDA cases did not all run"

drop_cache() {  # $1 layer spec a-b: fadvise DONTNEED every shard those layers read (a cold start for each run)
  "$PP" - "$SRC" "$1" <<'PY'
import json, os, sys
src, spec = sys.argv[1], sys.argv[2]
a, b = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
shards = sorted({f for k, f in wm.items() if any(f".layers.{i}." in k for i in range(a, b + 1))})
for s in shards:
    fd = os.open(os.path.join(src, s), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
print(f"page cache dropped for {len(shards)} shard(s) of L{a}-L{b}")
PY
}
side() { nvidia-smi dmon -i 3 -s puc -d 2 -o T > "$R/dmon_$1.txt" 2>&1 & DMON=$!; }
unside() { kill $DMON 2>/dev/null; wait $DMON 2>/dev/null; }
one() {  # $1 tag $2 tool $3 sha $4 layers, then extra parity args (e.g. --inject-ulp N) and -- tool args
  local tag=$1 tool=$2 sha=$3 layers=$4; shift 4
  wait_ram; [ "${NO_DROP:-0}" = 1 ] || drop_cache "$layers"; side "$tag"
  echo "-- $tag $(basename "$tool") $(utc)"
  "$PP" "$T/k3_accel_profile.py" monitor --out "$R/monitor_$tag.jsonl" --interval 5 --mem-floor-gib "$FLOOR_GIB" -- \
    nice -n 15 ionice -c2 -n7 "$PP" "$T/k3_cache_parity.py" digest --tool "$tool" --tool-sha "$sha" \
    --out "$R/digest_$tag.json" "$@" > "$R/$tag.log" 2>&1 &
  MON=$!; wait $MON  # in the background, so a TERM to this driver runs its trap at once
  local rc=$?; unside
  echo "$tag exit=$rc $(utc) : $(grep -h 'digest 1\.' "$R/$tag.log" | tail -1)"
  grep -h "^\[k3_teacher_cache .*\] L[0-9][0-9] " "$R/$tag.log" | sed "s/^/   $tag /"
  [ $rc -eq 75 ] && { echo "$tag stopped by the RAM floor; stopping"; exit 75; }
  return $rc
}
cmpd() {  # $1 a $2 b $3 name
  "$PP" "$T/k3_cache_parity.py" compare "$R/digest_$1.json" "$R/digest_$2.json" --out "$R/compare_$3.json" \
    --lines "$R/compare_$3.lines.txt" --quiet
  echo "compare $3 exit=$?"
}

WINDOWS=${WINDOWS:-"w1 w2 w3 w4 b2geo mini"}
P0=$OC/distill/corpus/k3/pass-0/k3_input_ids.safetensors
SNAP61=/mnt/lexar/omni-campaign/k3-continuation-oct08-claude/fold/snapshots/before_L61.safetensors
COMMON=(--probe --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto
  --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --pass-count 1 --pass-index 0)
for TAG in $WINDOWS; do
  case $TAG in
    w1)    LAYERS=84-85; SPEC=(--input-ids "$P0" --row-limit 16 --layers 84-85 --synthetic-blocks 7 --group-tokens 65536) ;;
    w2)    LAYERS=83;    SPEC=(--input-ids "$P0" --row-limit 32 --truncate-seq 2048 --layers 83 --synthetic-blocks 7 --group-tokens 65536) ;;
    w3)    LAYERS=0-3;   SPEC=(--input-ids "$P0" --row-limit 64 --truncate-seq 2048 --layers 0-3 --synthetic-blocks 0 --group-tokens 65536) ;;
    w4)    LAYERS=84-85; SPEC=(--input-ids "$P0" --row-limit 16 --layers 84-85 --synthetic-blocks 7 --group-tokens 16384) ;;
    b2geo) LAYERS=83-85; SPEC=(--input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors"
             --input-ids "$OC/composite/corpus/k3_review_native_input_ids.safetensors" --no-trim
             --layers 83-85 --synthetic-blocks 7 --group-tokens 32768) ;;
    mini)  LAYERS=0-11;  SPEC=(--input-ids "$P0" --row-limit 2 --truncate-seq 2048 --layers 0-11 --synthetic-blocks 0 --group-tokens 65536) ;;
    head61) LAYERS=61-92; SPEC=(--input-ids "$OC/composite/corpus/k3_review_input_ids.safetensors" --row-limit 8
             --layers 61-92 --start-snapshot "$SNAP61" --snapshot-layout "k3_review_input_ids=98304:512"
             --group-tokens 32768) ;;
    tests) continue ;;  # WINDOWS=tests: the CPU / negative-control / GPU unit tests only
    *) echo "unknown window $TAG"; continue ;;
  esac
  ARGS=(run "${SPEC[@]}" "${COMMON[@]}")
  [ "$TAG" = head61 ] || ARGS+=(--no-head)  # only a window from the reference snapshot to L92 runs the head
  FIRST=${LAYERS%%-*}
  echo "== $TAG ${SPEC[*]} $(utc)"
  one "${TAG}_A" "$OLD" "$W124" "$LAYERS" -- "${ARGS[@]}" --pass-dir "$R/${TAG}_A-probe"
  one "${TAG}_B" "$NEW" "$W130" "$LAYERS" -- "${ARGS[@]}" --pass-dir "$R/${TAG}_B-probe"
  cmpd "${TAG}_A" "${TAG}_B" "${TAG}_A_B"
  case $TAG in w1|w2|w3|w4)
    one "${TAG}_C" "$NEW" "$W130" "$LAYERS" -- "${ARGS[@]}" --stage-timers --pass-dir "$R/${TAG}_C-probe"
    cmpd "${TAG}_A" "${TAG}_C" "${TAG}_A_C" ;;
  esac
  if [ "$TAG" = w1 ]; then
    one "${TAG}_NEG" "$NEW" "$W130" "$LAYERS" --inject-ulp "$FIRST" -- "${ARGS[@]}" --pass-dir "$R/${TAG}_NEG-probe"
    cmpd "${TAG}_A" "${TAG}_NEG" "${TAG}_A_NEG"
  fi
done
sha256sum "$R"/digest_*.json "$R"/compare_*.json 2>/dev/null
echo "== done $(utc)"
