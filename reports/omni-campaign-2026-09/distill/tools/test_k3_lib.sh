#!/bin/bash
# test_k3_lib.sh -- k3_lib.sh's K3 pass contract on temp dirs and a FAKE k3 tool (no GPU work; reads GPU6's name and
# temperature only).  Usage: bash test_k3_lib.sh [k3_lib.sh, default beside this file]
#   class      k3_card_class 6 = quadro-rtx-8000-sm75
#   review     k3_v130_review_ok: absent / wrong line 1 / wrong sha refused, the exact file accepted
#   choose     k3_choose_tool: v130 only with review + frozen copy at the pin + a gate PASS for that sha on that class
#   pick       sealed passes skipped, a live claim is busy, another class's geometry is bound, a free pass is claimed;
#              the claimant re-picks its own claim; a second runner sees it busy; a seal by an unknown tool is foreign
#   geometry   recorded at the first start, equal accepted, a different group-token count refused
#   oom-v130   OOM at --moe-group-batch 4 -> 1, one plain retry, then sealed: group tokens 65536 on every attempt
#   oom-v124   OOM twice -> HALT after 2 attempts, group tokens never changed
#   refuse     a resume at other group tokens is refused before the tool runs (rc 65)
#   fail       a tool exit 7 is FAIL with its exit, no retry -- even with an earlier attempt's OOM lines in the same log
#   record / legacy / all-sealed
# NEGATIVE CONTROL: bash test_k3_lib.sh --negative runs the same cases on a copy of the lib that changes the group
# tokens on an OOM (what queue_gpu6.sh <= 1.3.2 did); it must FAIL.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIB=${1:-$HERE/k3_lib.sh}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
if [ "$LIB" = --negative ]; then
  LIB=$D/k3_lib_negctl.sh
  sed 's|moeb=1; "$K3_WAIT_CARD"; sleep|moeb=1; GT=32768; "$K3_WAIT_CARD"; sleep|' "$HERE/k3_lib.sh" > "$LIB"
  grep -q 'GT=32768; "$K3_WAIT_CARD"' "$LIB" || { echo "negative control: mutation not applied"; exit 2; }
  echo "NEGATIVE CONTROL: lib copy that resumes an OOM at 32768 group tokens"
fi
utc() { date -u +%FT%TZ; }
status() { echo "STATUS $1 $2 ${3:-}" >> "$D/status.log"; }
alloc() { :; }
lane_hb() { :; }
GPU=6; Q=$D/q; QL=$D/ql; OC=$D/oc; KC=$D/kc; PY=bash
K3_OVERHEAD_GIB=0; RAM_FLOOR_GIB=0; K3_OOM_ADJ=500; THERMAL_RETRIES=1; K3_OOM_SLEEP_S=0
mkdir -p "$Q" "$QL" "$KC/pass-3" "$D/out"
. "$LIB"
STOP_C=150; HARD_C=200; RESUME_C=0; T_MAXOP=90; T_SLOW=95; T_SHUT=100   # never fires: the thermal path is not under test
pass=0; fail=0
check() {  # $1 name $2 got $3 want
  if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1"; else fail=$((fail + 1)); echo "FAIL $1: got '$2' want '$3'"; fi
}

# class
check class "$(k3_card_class 6)" quadro-rtx-8000-sm75

# review
PIN=$(printf 'fake 1.3.0\n' > "$D/v130.py"; sha256sum "$D/v130.py" | cut -c1-64)
r() { k3_v130_review_ok "$1" "$PIN" && echo 1 || echo 0; }
check review-absent "$(r "$D/none")" 0
printf 'k3-v130-review-passed\nsha256=%s\n' "$PIN" > "$D/rv1"; check review-line1 "$(r "$D/rv1")" 0
printf 'k3-v130-review-pass\nsha256=%s0\n' "${PIN:0:63}" > "$D/rv2"; check review-sha "$(r "$D/rv2")" 0
printf 'k3-v130-review-pass\nsha256=%s\nby=selftest\n' "$PIN" > "$D/rv"; check review-ok "$(r "$D/rv")" 1

# choose
gate() { printf '{"verdict": "%s", "tool_sha256": "%s", "card_class": "%s"}\n' "$1" "$2" "$3" > "$D/gate.json"; }
gate pass "$PIN" quadro-rtx-8000-sm75
check choose-v130 "$(k3_choose_tool "$D/rv" "$PIN" "$D/v130.py" "$D/gate.json" quadro-rtx-8000-sm75)" v130
check choose-no-review "$(k3_choose_tool "$D/none" "$PIN" "$D/v130.py" "$D/gate.json" quadro-rtx-8000-sm75 | cut -d' ' -f1-3)" "v124 no 1.3.0"
check choose-other-class "$(k3_choose_tool "$D/rv" "$PIN" "$D/v130.py" "$D/gate.json" nvidia-geforce-rtx-3090-sm86 | cut -c1-4)" v124
gate differ "$PIN" quadro-rtx-8000-sm75
check choose-gate-differ "$(k3_choose_tool "$D/rv" "$PIN" "$D/v130.py" "$D/gate.json" quadro-rtx-8000-sm75 | cut -c1-4)" v124
gate pass "$PIN" quadro-rtx-8000-sm75; printf 'changed\n' > "$D/v130b.py"
check choose-copy-changed "$(k3_choose_tool "$D/rv" "$PIN" "$D/v130b.py" "$D/gate.json" quadro-rtx-8000-sm75 | cut -c1-4)" v124

# pick
O=$D/out; A=aaaa1111; B=bbbb2222
mkdir -p "$O/pass-0-of-4" "$O/pass-1-of-4" "$O/pass-2-of-4"
echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$A\"}" > "$O/pass-0-of-4/manifest.json"
setsid sleep 611.7 < /dev/null > /dev/null 2>&1 & LIVE=$!
echo "{\"pid\": $LIVE, \"match\": \"sleep 611.7\", \"gpu\": \"4\"}" > "$O/pass-1-of-4.claim"
echo '{"tool_sha256": "aaaa1111", "card_class": "nvidia-geforce-rtx-3090-sm86", "group_tokens": 32768}' > "$O/pass-2-of-4.geometry"
check pick-free "$(k3_pick_pass "$O" "0 1 2 3" quadro-rtx-8000-sm75 "$A" 65536 "$A,$B" test_k3_lib.sh)" "PICK 3"
check pick-claim-written "$(python3 -c "import json;c=json.load(open('$O/pass-3-of-4.claim'));print(c['pid'],c['match'],c['group_tokens'])")" "$$ test_k3_lib.sh 65536"
check pick-own-claim "$(k3_pick_pass "$O" "0 1 2 3" quadro-rtx-8000-sm75 "$A" 65536 "$A,$B" test_k3_lib.sh)" "PICK 3"
check pick-other-runner "$(k3_pick_pass "$O" "3 2 1 0" quadro-rtx-8000-sm75 "$A" 65536 "$A,$B" other_runner.sh)" \
  "NONE sealed=0 busy=3:gpu6,1:gpu4 bound=2:nvidia-geforce-rtx-3090-sm86/aaaa1111/32768 foreign=-"
check pick-bound-same-class "$(k3_pick_pass "$O" "2" nvidia-geforce-rtx-3090-sm86 "$A" 32768 "$A,$B" other_runner.sh)" "PICK 2"
kill $LIVE; wait $LIVE 2>/dev/null
check pick-dead-claim "$(k3_pick_pass "$O" "1" quadro-rtx-8000-sm75 "$A" 65536 "$A,$B" test_k3_lib.sh)" "PICK 1"
echo '{"verdict": "sealed", "tool_sha256": "cccc3333"}' > "$O/pass-0-of-4/manifest.json"
check pick-foreign "$(k3_pick_pass "$O" "0" quadro-rtx-8000-sm75 "$A" 65536 "$A,$B" test_k3_lib.sh)" "NONE sealed=- busy=- bound=- foreign=0:cccc3333"
k3_release_claim "$O/pass-1-of-4"; check release "$([ -e "$O/pass-1-of-4.claim" ] && echo kept || echo gone)" gone

# geometry
G=$D/g/pass-0-of-4; mkdir -p "$G"
k3_geometry "$G" "$A" quadro-rtx-8000-sm75 65536 > /dev/null; check geometry-first "$?" 0
check geometry-file "$(python3 -c "import json;g=json.load(open('$G.geometry'));print(g['group_tokens'],g['card_class'])")" "65536 quadro-rtx-8000-sm75"
k3_geometry "$G" "$A" quadro-rtx-8000-sm75 65536 > /dev/null; check geometry-same "$?" 0
WHY=$(k3_geometry "$G" "$A" quadro-rtx-8000-sm75 32768); check geometry-differs "$?:$WHY" "1:group_tokens 65536 != 32768"

# the fake tool: one outcome per attempt from $D/outcomes (oom | ok | fail7); every call's args into $D/calls
cat > "$D/fake_tool.sh" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_DIR/calls"
n=$(wc -l < "$FAKE_DIR/calls")
what=$(sed -n "${n}p" "$FAKE_DIR/outcomes")
pd=""; prev=""
for a in "$@"; do [ "$prev" = --pass-dir ] && pd=$a; prev=$a; done
case "$what" in
  ok) mkdir -p "$pd"; echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$FAKE_SHA\"}" > "$pd/manifest.json"; exit 0 ;;
  oom) echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.20 GiB"; exit 1 ;;
  *) echo "fake failure"; exit 7 ;;
esac
EOF
export FAKE_DIR=$D/fake FAKE_SHA=$A
runpass() {  # $1 kind $2 outcomes (space separated) $3 GT $4 pass dir
  rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"; printf '%s\n' $2 > "$FAKE_DIR/outcomes"; : > "$FAKE_DIR/calls"
  PASSKIND=$1; PASSTOOL=$D/fake_tool.sh; PASSSHA=$A; GT=$3; MOEB=4; CARD_CLASS=quadro-rtx-8000-sm75; DIGEST=""
  K3_STATE=$D/state; K3_GATE=0; K3_RES=1; K3_SLOT=1; K3_WAIT_CARD=true
  mkdir -p "$4"
  k3_run_pass 3 "$4" > /dev/null 2>&1
}
calls_field() {  # $1 flag: its value on every call, comma separated
  awk -v f="$1" '{v="-"; for (i = 1; i < NF; i++) if ($i == f) v = $(i + 1); printf "%s%s", (NR > 1 ? "," : ""), v} END {print ""}' "$FAKE_DIR/calls"
}
runpass v130 "oom oom ok" 65536 "$D/p1/pass-3-of-4"
check oom-v130-end "$K3_END:$RC" "PASS:0"
check oom-v130-moeb "$(calls_field --moe-group-batch)" "4,1,1"
check oom-v130-gt "$(calls_field --group-tokens)" "65536,65536,65536"
check oom-v130-no-fadvise "$(grep -c -- '--no-fadvise-drop --kda-graph-keep kda-only' "$FAKE_DIR/calls")" 3
runpass v124 "oom oom oom" 65536 "$D/p2/pass-3-of-4"
check oom-v124-end "$K3_END" HALT
check oom-v124-gt "$(calls_field --group-tokens)" "65536,65536"
check oom-v124-no-accel "$(grep -c -- '--moe-group-batch' "$FAKE_DIR/calls")" 0
runpass v130 "ok" 32768 "$D/p1/pass-3-of-4"     # p1 began at 65536
check refuse-rc "$RC:$K3_END" "65:FAIL"
check refuse-no-call "$(wc -l < "$FAKE_DIR/calls" | tr -d ' ')" 0
runpass v124 "fail7 ok" 65536 "$D/p3/pass-3-of-4"
check fail-end "$K3_END:$RC:$(wc -l < "$FAKE_DIR/calls" | tr -d ' ')" "FAIL:7:1"

# record / legacy / all-sealed
mkdir -p "$D/r/pass-3-of-4"; echo '{"verdict": "sealed"}' > "$D/r/pass-3-of-4/manifest.json"
GPU=4 k3_record_pass 3 "$D/r/pass-3-of-4" "$A" "k3_teacher_cache 1.3.0" nvidia-geforce-rtx-3090-sm86 "$D/eqv.json"
check record "$(python3 -c "import json;p=json.load(open('$Q/k3_pass_tools.json'))['passes']['3'];print(p['tool_sha256'],p['card_class'],p['gpu'])")" "$A nvidia-geforce-rtx-3090-sm86 4"
L=$D/legacy; mkdir -p "$L/pass-2-of-4"; touch "$L/pass-2-of-4/head_progress.json"; echo "$B" > "$L/pass-2-of-4.queue_tool.sha256"
printf '== k3-pass-2 attempt 1 group_tokens 65536 x\n== k3-pass-2 attempt 2 group_tokens 32768 x\n' > "$QL/k3-pass-2.log"
k3_legacy_geometry "$L" > /dev/null
check legacy "$(python3 -c "import json;g=json.load(open('$L/pass-2-of-4.geometry'));print(g['tool_sha256'],g['card_class'],g['group_tokens'])")" "$B quadro-rtx-8000-sm75 32768"
S=$D/s; for k in 0 1 2 3; do mkdir -p "$S/pass-$k-of-4"; echo "{\"verdict\": \"sealed\", \"tool_sha256\": \"$([ $k = 2 ] && echo $B || echo $A)\"}" > "$S/pass-$k-of-4/manifest.json"; done
k3_all_sealed "$S" "$A,$B"; check all-sealed "$?" 0
k3_all_sealed "$S" "$A"; check all-sealed-one-tool "$?" 1

echo "K3 LIB TEST ($K3LIB_VERSION, $(sha256sum "$LIB" | cut -c1-16)): $pass passed, $fail failed"
[ $fail -eq 0 ]
