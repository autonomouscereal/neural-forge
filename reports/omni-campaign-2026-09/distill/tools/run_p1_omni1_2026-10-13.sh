#!/bin/bash
# S0-P1 on OMNI1 (G0 item), p1_qwen35_reader 1.0.5 (charted blocks skipped in the text map): the reader suite, then the
# headers-only map (FFN width, channel partition, extras = band rows + media
# blocks), the tensor-exact parity of every mapped text tensor against a direct
# safe_open read, and the tps_probe seam smoke (strict load + 8-token finite
# forward).  CPU only, nice 15, 8 threads; read-only on OMNI1 and the parent.
# A stat of every OMNI1 file is taken before and after.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
EF=$OC/import/rq27b/tree_ef1ffb9
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
P=/home/cereal/models/Qwen3.8-27B-Uncensored
OUT=$OC/distill/tps
TAG=p1_reader_omni1_2026-10-13
echo "start $(date -u +%FT%TZ) host $(hostname)"
stat -c '%n %s %Y' $OMNI1/* > $OUT/${TAG}_stat_before.txt
cd $OC/distill/tools
echo "== P1 reader suite"
CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$OC/distill/tools nice -n 15 $PY test_p1_qwen35_reader.py 2>&1 | grep -E "^Ran|^OK|^FAILED"
echo "P1_TESTS_EXIT=${PIPESTATUS[0]}"
echo "== map (headers only)"
nice -n 15 $PY p1_qwen35_reader.py map --chimera $OMNI1 --parent-config $P --json $OUT/${TAG}_map.json --quiet
echo "MAP_EXIT=$?"
$PY - "$OUT/${TAG}_map.json" <<'EOF'
import json, sys, collections
m = json.load(open(sys.argv[1]))
cov = m["coverage"]
print("map: text layers", m["num_layers"], "blocks", m["num_blocks"], "charted", m["charted_blocks"], "text_blocks[30:34]", m["text_blocks"][30:34], "ffn_width", m["ffn_width"], "channel_partition", m["channel_partition"],
      "base_vocab", m["base_vocab"], "vocab", m["vocab"], "tied", m["tied"], "kinds", m["kinds"])
print("map: layer_types", dict(collections.Counter(m["layer_types"])),
      "qwen35 intermediate_size", m["qwen35_config"]["text_config"]["intermediate_size"])
print("coverage: chimera_tensors", cov["chimera_tensors"], "chimera_elements", f"{cov['chimera_elements']:,}",
      "mapped_tensors", cov["mapped_tensors"], "mapped_elements", f"{cov['mapped_elements']:,}",
      "extra_entries", cov["extra_entries"], "extra_elements", f"{cov['extra_elements']:,}",
      "miscovered", len(cov["miscovered"]))
EOF
echo "== extras by group (served by Chimera name, never mixed into qwen3_5 names)"
$PY - "$OMNI1" <<'EOF'
import sys, collections
sys.path.insert(0, ".")
import p1_qwen35_reader as p1
r = p1.open_reader(sys.argv[1])
groups = collections.Counter()
gain = []
for name in r.extras():
    head = name.split(".")[0] if not name.endswith("[base:]") else name
    if name.startswith("blocks."):
        head = "blocks.*." + name.split(".")[2]
    groups[head] += 1
    if "gain" in name.lower() or "scale" in name.lower():
        gain.append(name)
for k, v in sorted(groups.items()):
    print(f"  extra group {k}: {v}")
print("extras named gain/scale:", gain[:20], "count", len(gain))
ffn_extra = [n for n in r.shards.names() if ".ffn." in n and not n.endswith(("ffn.up.weight", "ffn.down.weight"))]
print("ffn tensors other than up/down:", ffn_extra[:10], "count", len(ffn_extra))
r.close()
EOF
echo "== OMNI1 parity"
nice -n 15 /usr/bin/time -v $PY p1_qwen35_reader.py parity --chimera $OMNI1 --parent-dir $P \
  --receipt $OUT/${TAG}_parity.json --threads 8 --progress 2>&1 | grep -v "^  [0-9]*/851" | grep -E "P1 PARITY|receipt|FAIL|REFUSED|Maximum resident|Exit status|Elapsed"
echo "PARITY_EXIT=${PIPESTATUS[0]}"
echo "== seam smoke"
rm -rf $OC/distill/.scratch/p1_qwen35_config_omni1
nice -n 15 /usr/bin/time -v $PY p1_qwen35_reader.py seam-smoke --chimera $OMNI1 --parent-dir $P \
  --config-dir $OC/distill/.scratch/p1_qwen35_config_omni1 --tps-probe-dir $OC/distill \
  --forward-tokens 8 --receipt $OUT/${TAG}_seam_smoke.json --threads 8 2>&1 | grep -E "P1 SEAM|receipt|FAIL|REFUSED|Maximum resident|Exit status"
echo "SMOKE_EXIT=${PIPESTATUS[0]}"
stat -c '%n %s %Y' $OMNI1/* > $OUT/${TAG}_stat_after.txt
cmp $OUT/${TAG}_stat_before.txt $OUT/${TAG}_stat_after.txt && echo "OMNI1_STAT_UNCHANGED files=$(wc -l < $OUT/${TAG}_stat_after.txt)"
sha256sum $OUT/${TAG}_map.json $OUT/${TAG}_parity.json $OUT/${TAG}_seam_smoke.json
echo "ALL_DONE $(date -u +%FT%TZ)"
