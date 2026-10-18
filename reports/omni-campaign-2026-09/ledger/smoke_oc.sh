#!/usr/bin/env bash
# End-to-end CLI smoke for the Chimera Omni ledger lane, run on the AI server.
# Builds throwaway fixtures under $S, exercises every command, and prints the
# exit code of each step.  Nothing outside $S is written and nothing is deleted
# except the fixture the R4 path is supposed to delete.
set -u

S=/home/cereal/omni-campaign/ledger/.smoke-2026-09-02
L=/home/cereal/omni-campaign/ledger
PY=${PY:-python3}

rm -rf "$S"
mkdir -p "$S"
R="$S/registry"
G="$S/ledger.jsonl"
# --registry / --ledger belong to the SUBCOMMAND parser, so they go last.
oc() { $PY "$L/oc_checkpoint.py" "$@" --registry "$R" --ledger "$G"; }

# ---- fixtures: real safetensors headers, no torch ------------------------
timeout 60 $PY - "$S" <<'PYFIX'
import json, struct, sys
from pathlib import Path
root = Path(sys.argv[1])
def st(path, tensors):
    header, off, blob = {}, 0, bytearray()
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + size]}
        blob += bytes(size); off += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(blob))
def model(name, tensors):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    st(d / "model.safetensors", tensors)
    (d / "config.json").write_text('{"model_type":"qwen3_5"}')
    (d / "tokenizer.json").write_text('{"version":"1"}')
model("parents/Q35-4B", {"w": [64, 64]})     # 4096 parameters
model("children/OC01-F-Q35-4B-C2S4L03", {"w": [16, 16]})   # 256 -> 16x
model("children/OC01-F-Q35-4B-LIED1", {"w": [60, 64]})     # 3840 -> 0.9375
model("children/OC01-D-Q35-4B-C2S4L03", {"w": [16, 16]})   # the trained child
print("fixtures built")
PYFIX

P="$S/parents/Q35-4B"
C="$S/children/OC01-F-Q35-4B-C2S4L03"
BAD="$S/children/OC01-F-Q35-4B-LIED1"

step() { echo; echo "### $* ###"; }

step "1 register DRY RUN (writes nothing)"
oc register --checkpoint-id OC01-F-Q35-4B-C2S4L03 --stage fold \
  --directory "$C" --parent-directory "$P" --parent-code Q35-4B \
  --component "folded:4096:256:w" --recipe-command "mlpq.sh C2S4L03" \
  | tail -6
echo "exit=$?  registry exists: $( [ -d "$R" ] && echo YES || echo NO )"

step "2 register --confirm"
oc register --checkpoint-id OC01-F-Q35-4B-C2S4L03 --stage fold \
  --directory "$C" --parent-directory "$P" --parent-code Q35-4B \
  --component "folded:4096:256:w" --recipe-command "mlpq.sh C2S4L03" --confirm \
  | grep -E "parameters|net_ratio|ledger sha256|sealed "
echo "exit=$?"

step "3 BLOCKER 2 -- a typed count the headers contradict"
oc register --checkpoint-id OC01-F-Q35-4B-LIED1 --stage fold \
  --directory "$BAD" --parent-directory "$P" --parent-code Q35-4B \
  --parameter-count 256 --parent-parameter-count 4096 \
  --component "folded:4096:256" --recipe-command "x" --confirm 2>&1 | tail -2
echo "exit=${PIPESTATUS[0]}"

step "4 BLOCKER 4 -- register under /mnt and under the model library"
oc register --checkpoint-id OC01-F-Q35-4B-SHAR1 --stage fold \
  --directory /mnt/ai_data/models/whatever --parent-directory "$P" \
  --parent-code Q35-4B --recipe-command x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"
oc register --checkpoint-id OC01-F-Q35-4B-SHAR2 --stage fold \
  --directory /home/cereal/models/Qwen3.5-4B --parent-directory "$P" \
  --parent-code Q35-4B --recipe-command x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"

step "4b BLOCKER (round 2) -- a spelling may not walk past the guard"
# `..` out of the campaign root and into the source model library
oc register --checkpoint-id OC01-F-Q35-4B-DOTS1 --stage fold \
  --directory /home/cereal/omni-campaign/../models/Qwen3.5-4B \
  --parent-directory "$P" --parent-code Q35-4B --recipe-command x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"
# a campaign-looking path whose real location is the CIFS share
ln -sfn /mnt/ai_data "$S/share-link"
oc register --checkpoint-id OC01-F-Q35-4B-LINK1 --stage fold \
  --directory "$S/share-link/models/whatever" \
  --parent-directory "$P" --parent-code Q35-4B --recipe-command x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"
# the child registered in step 2, spelled again -- one directory, not two ids
oc register --checkpoint-id OC01-F-Q35-4B-SPELL2 --stage fold \
  --directory "$S/children/../children/OC01-F-Q35-4B-C2S4L03" \
  --parent-directory "$P" --parent-code Q35-4B \
  --component "folded:4096:256:w" --recipe-command x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"
# and the record from step 2 names the LOCATION, not the spelling
$PY - "$L" "$R" <<'PYSEAL'
import sys
sys.path.insert(0, sys.argv[1])
import oc_checkpoint
print("sealed directory:",
      oc_checkpoint.current(sys.argv[2], "OC01-F-Q35-4B-C2S4L03")["checkpoint"]["directory"])
PYSEAL

step "5 record-review (status computed)"
cat > "$S/rev.json" <<'PYREV'
{"reviewer":"smoke","dossier_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
 "verdict_receipt_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
 "bank1":{"lanes_total":16,"useful":2,"partial":3,"observed_miss":10,
   "parent_useful_or_partial":12,"sealed_verdict":"alive-degraded",
   "canonical_signal":4,"canonical_artifact_miss":6,"canonical_content_miss":5},
 "columns":{"column_a":5,"column_b":4},"bank2":null,
 "media_out":[{"prompt_id":"av-kettle-A","non_empty":true,"audio_rms":0.24,"cyril_score":null}],
 "smoke_verdict":"coherent","safety":"safe-refusal/redirect","degeneracy_reading":"none",
 "notes":"smoke"}
PYREV
oc record-review --checkpoint-id OC01-F-Q35-4B-C2S4L03 --review-json "$S/rev.json" --confirm \
  | grep -E "status|canonical signal|ledger sha256"
echo "exit=$?"

step "6 BLOCKER-adjacent -- a silent WAV may not claim to be non-empty"
sed 's/"audio_rms":0.24/"audio_rms":0.0/' "$S/rev.json" > "$S/rev-silent.json"
oc record-review --checkpoint-id OC01-F-Q35-4B-C2S4L03 --review-json "$S/rev-silent.json" --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"

step "7 review package"
$PY $L/oc_package.py build --registry "$R" --ledger "$G" \
  --checkpoint-id OC01-F-Q35-4B-C2S4L03 --out "$S/human-review/OC01-F-Q35-4B-C2S4L03" \
  | grep -E "files|package_sha256|re-verified"
echo "exit=$?"
$PY $L/oc_package.py verify --package "$S/human-review/OC01-F-Q35-4B-C2S4L03" | tail -2
echo "exit=$?"

step "8 a mistyped --dossier is a refusal, not a traceback"
$PY $L/oc_package.py build --registry "$R" --ledger "$G" \
  --checkpoint-id OC01-F-Q35-4B-C2S4L03 --out "$S/pkg-missing" \
  --dossier "$S/nope.json" 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"

step "9 clone before training (R2)"
oc clone --source OC01-F-Q35-4B-C2S4L03 --stage-about-to-run D --confirm \
  | grep -E "clone digest|sealed|status"
echo "exit=$?"

step "10 BLOCKER 4 -- a clone may not land in the model library"
oc clone --source OC01-F-Q35-4B-C2S4L03 --stage-about-to-run D --clone-index 9 \
  --dest /home/cereal/models/clone-x --confirm 2>&1 | tail -1
echo "exit=${PIPESTATUS[0]}"

step "11 a distill child off a NON-clone is refused"
oc register --checkpoint-id OC01-D-Q35-4B-C2S4L03 --stage distill \
  --directory "$S/children/OC01-D-Q35-4B-C2S4L03" \
  --parent-checkpoint-id OC01-F-Q35-4B-C2S4L03 --parent-directory "$P" \
  --parent-code Q35-4B --recipe-command "distill.sh" --confirm 2>&1 | tail -2
echo "exit=${PIPESTATUS[0]}"

step "12 delete a partially-capable child -- every clause printed"
oc delete --checkpoint-id OC01-F-Q35-4B-C2S4L03 \
  --reason "smoke: this exists only to show the fourteen R4 clauses printing, and it is not a real retention decision at all." \
  --rebuild-command "mlpq.sh C2S4L03" --retained-source "$P" --confirm 2>&1 \
  | grep -E "^    \[|REFUSED"
echo "exit=${PIPESTATUS[0]}"
echo "clauses printed: $(oc delete --checkpoint-id OC01-F-Q35-4B-C2S4L03 --reason "$(head -c 80 /dev/zero | tr '\0' 'x')" --rebuild-command x --retained-source "$P" 2>/dev/null | grep -c '^    \[')"
echo "child still on disk: $( [ -d "$C" ] && echo YES || echo NO )"

step "13 dashboards"
OC_ROOT=$S $PY $L/oc_dashboard.py --registry "$R" --ledger "$G" --root "$S" | tail -3
echo "exit=$?"
grep -m1 "ratio from" "$S/SCORECARD.md"
grep -m1 "| headers |" "$S/SCORECARD.md" | cut -c1-120

step "14 integrity, both directions"
$PY $L/oc_ledger.py verify --ledger "$G" --registry "$R" | tail -2
echo "exit=$?"
oc verify | tail -2
echo "exit=$?"
oc verify --deep | tail -2
echo "exit=$?"
$PY $L/oc_ledger.py digest --ledger "$G" | tail -1

step "15 BLOCKER 1 -- merge --apply appends to a 1,056-row authority"
timeout 120 $PY - "$S" "$L" <<'PYMERGE'
import sys, io, contextlib
from pathlib import Path
S, L = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(L))
import oc_ledger, oc_seal, inspect
def row(i, digest, stamp):
    r = {c: None for c in oc_ledger.CHECKPOINT_ROW_COLUMNS}
    r.update({"schema": oc_ledger.CHECKPOINT_ROW_SCHEMA, "campaign": oc_seal.CAMPAIGN,
              "checkpoint_id": "A", "record_instance_id": i, "record_seq": 1,
              "event": "register", "stage": "fold", "status": "candidate",
              "status_reason": "-", "sealed_at": stamp, "receipt_sha": digest,
              "notes": "", "media_out_artifacts": 0, "media_out_scored": 0})
    return r
auth, shard, rep = S / "m.jsonl", S / "s.jsonl", S / "r.json"
oc_ledger.append_rows(auth, [row(f"A-{i:04d}-register", f"{i:064x}", "2026-09-01T00:00:00.000Z")
                             for i in range(1, 1057)])
before = auth.read_bytes()
oc_ledger.append_rows(shard, [row("Z-0001-register", "f" * 64, "2026-09-02T00:00:00.000Z")])
with contextlib.redirect_stdout(io.StringIO()):
    code = oc_ledger.main(["merge", "--authority", str(auth), "--shard", str(shard),
                           "--report", str(rep), "--apply"])
after = auth.read_bytes()
src = inspect.getsource(oc_ledger.cmd_merge)
print(f"exit={code} rows {len(before.splitlines())} -> {len(after.splitlines())} "
      f"prior-bytes-in-place={after[:len(before)] == before}")
print(f"authority.write_bytes( in cmd_merge: {'authority.write_bytes(' in src}  "
      f"open(\"ab\") in cmd_merge: {'open(\"ab\")' in src}")
PYMERGE

step "16 real models: counts from headers, no torch"
timeout 200 $PY - <<'PYREAL'
import sys, time
sys.path.insert(0, "/home/cereal/omni-campaign/ledger")
from oc_seal import safetensors_parameter_count, directory_digest
for d in ("/home/cereal/models/Qwen3.5-0.8B", "/home/cereal/models/Qwen3.5-4B"):
    t = time.time()
    try:
        n, per = safetensors_parameter_count(d)
        print(f"{d}  params={n:,}  shards={len(per)}  {time.time()-t:.2f}s")
    except Exception as exc:
        print(f"{d}  UNAVAILABLE: {type(exc).__name__}: {exc}")
t = time.time()
try:
    dg, entries = directory_digest("/home/cereal/models/Qwen3.5-0.8B")
    total = sum(int(e["bytes"]) for e in entries)
    dt = time.time() - t
    print(f"digest {dg}  files={len(entries)}  {total/2**30:.2f} GiB  {dt:.1f}s "
          f"= {total/dt/1e6:.0f} MB/s")
except Exception as exc:
    print(f"digest UNAVAILABLE: {type(exc).__name__}: {exc}")
PYREAL

echo
echo "### SMOKE COMPLETE -- fixtures left at $S ###"
