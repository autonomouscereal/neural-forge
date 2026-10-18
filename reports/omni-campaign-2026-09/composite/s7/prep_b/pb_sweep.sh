set -u
# PREP-B time-row weight sweep: ridge-only chart fits on scratch copies of the new unit (never the real unit dir)
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
PB=$S7/prep_b
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
PY="nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python"
WEIGHTS="1 64 512 4096"
fit() {
  W=$1; D=$PB/sweep/w$W/unit
  rm -rf $PB/sweep/w$W; mkdir -p $D
  cp -p $S7/units/hy3d/hy3d_dit_L10/charted_block.json $S7/units/hy3d/hy3d_dit_L10/charted_block.safetensors $S7/units/hy3d/hy3d_dit_L10/hy3d_time_token_reference.safetensors $D/
  $PY $OC/composite/tools/s7_chart_fit.py --unit hy3d_dit_L10 --unit-dir $D --arms ridge --time-row-weight $W > $PB/logs/sweep_w$W.log 2>&1 < /dev/null
  echo "w$W exit=$?" >> $PB/logs/sweep_w$W.log
}
set -- $WEIGHTS
while [ $# -gt 0 ]; do
  fit $1 & P1=$!
  if [ $# -gt 1 ]; then fit $2 & P2=$!; wait $P2; shift; fi
  wait $P1; shift
done
python3 - <<P
import json
rows=[]
for w in [int(x) for x in "$WEIGHTS".split()]:
    r=json.load(open(f"$PB/sweep/w{w}/unit/chart_receipt.json"))
    t=r["diagnostics"]["time_rows"]
    rows.append({"weight":w,
      "held_forward_time":{k:t["split_ridge_held_forward"]["time"][k] for k in ("cos","rel_l2","row_rel_l2","row_cos")},
      "held_forward_content":{k:t["split_ridge_held_forward"]["content"][k] for k in ("cos","rel_l2")},
      "held_backward_time":{k:t["split_ridge_held_backward"]["time"][k] for k in ("cos","rel_l2")},
      "held_backward_content":{k:t["split_ridge_held_backward"]["content"][k] for k in ("cos","rel_l2")},
      "held_roundtrip_all":r["diagnostics"]["ridge_split_held"]["roundtrip"],
      "record_in_sample_forward_time":{k:t["record_in_sample_forward"]["time"][k] for k in ("cos","rel_l2")},
      "ridge_rel":r["charts"]["ridge_rel"], "seconds":r["seconds"], "receipt":f"$PB/sweep/w{w}/unit/chart_receipt.json"})
base=rows[0]["held_forward_content"]["rel_l2"]
ok=[x for x in rows if x["held_forward_content"]["rel_l2"] <= base*1.01]
best=min(ok, key=lambda x:(round(x["held_forward_time"]["rel_l2"],4), x["weight"]))
out={"schema":"omni-campaign.prepb_time_row_weight_sweep.v1","unit":"hy3d_dit_L10","arms":"ridge only (split fit on fit groups, held = 2 objects)",
     "rule":"DESIGN (fixed before the sweep ran): the weight with the lowest held time-row forward relL2 (4 d.p.) among weights whose held content-row forward relL2 is within +1% of the weight-1 value; ties -> the smaller weight",
     "sweep":rows,"chosen_weight":best["weight"],"policy":"design parameter chosen from measurements (owner 2026-10-06); the numbers are diagnostics, never gates"}
json.dump(out, open("$PB/time_row_weight_sweep.json","w"), indent=1)
print(json.dumps({"chosen":best["weight"],"table":[(x["weight"],round(x["held_forward_time"]["rel_l2"],4),round(x["held_forward_time"]["cos"],4),round(x["held_forward_content"]["rel_l2"],4),round(x["held_forward_content"]["cos"],4)) for x in rows]}))
P
sha256sum $PB/time_row_weight_sweep.json > $PB/time_row_weight_sweep.json.sha256
echo SWEEP_DONE
