#!/usr/bin/env bash
# selftest.sh - end-to-end run of cleanup_exec.py on synthetic units against a throwaway omni_lab_test_<hex> database.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . /home/cereal/.config/omni-lab/env; set +a
PY=/home/cereal/.venvs/chimera-intake/bin/python
HEX=$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')
T=/mnt/lexar/omni-campaign/scratch/cleanup-selftest-$HEX
DB=omni_lab_test_$HEX
mkdir -p "$T/unitA/sub" "$T/recipes"
$PY - "$T" <<'PY'
import json, struct, sys, os
t = sys.argv[1]
hdr = json.dumps({"w": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128]}}).encode()
with open(f"{t}/unitA/model.safetensors", "wb") as fh:
    fh.write(struct.pack("<Q", len(hdr)) + hdr + b"\0" * 128)
json.dump({"model_type": "toy"}, open(f"{t}/unitA/config.json", "w"))
open(f"{t}/unitA/sub/run.log", "w").write("step 1 loss 2.0\n")
# (Lexar is exFAT: no symlinks to test here)
def s(b): return struct.pack("<Q", len(b)) + b
g = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + s(b"general.name") + struct.pack("<I", 8) + s(b"toy") \
    + s(b"tok") + struct.pack("<I", 2) + struct.pack("<QQ", 3, 5) + struct.pack("<IQ", 0, 0)
open(f"{t}/unitB.gguf", "wb").write(g + b"\0" * 64)
units = [{"path": f"{t}/unitA", "volume": "lexar", "size_bytes_du": 4096, "size_gb": 0.0, "what": "scratch",
          "reason": "selftest", "omni_lab": {"ledger_checkpoint_ids": []}, "recipe_archive": f"{t}/x/path-aaaa__unitA/",
          "plan_index": 9001},
         {"path": f"{t}/unitB.gguf", "volume": "lexar", "size_bytes_du": 200, "size_gb": 0.0, "what": "model",
          "reason": "selftest gguf", "omni_lab": {"ledger_checkpoint_ids": []}, "recipe_archive": f"{t}/x/path-bbbb__unitB/",
          "plan_index": 9002}]
json.dump(units, open(f"{t}/units.json", "w"))
json.dump([{"class": "never-delete", "path": f"{t}/recipes"}], open(f"{t}/protected.json", "w"))
json.dump(["/", "/mnt", "/mnt/lexar", "/mnt/lexar/omni-campaign", "/mnt/lexar/omni-campaign/scratch", t], open(f"{t}/generic.json", "w"))
PY
find "$T" -exec touch -h -d '2026-10-01 00:00' {} +
$PY - "$DB" <<'PY'
import sys; sys.path.insert(0, "..")
import omni_lab_db as db
from psycopg2 import sql
a = db.connect("rw", app="selftest", autocommit=True)
a.cursor().execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(sys.argv[1]))); a.close()
c = db.connect("rw", dbname=sys.argv[1], app="selftest"); db.migrate(c)
c.cursor().execute("INSERT INTO host (name) VALUES ('ai')"); c.commit(); c.close()
PY
export OMNI_LAB_PG_DB=$DB
rc=0
$PY cleanup_exec.py execute --units "$T/units.json" --protected "$T/protected.json" --generic "$T/generic.json" \
   --volume lexar --run-dir "$T/run" --recipe-root "$T/recipes" --hash-threads 2 > "$T/out.jsonl" 2>&1 || rc=$?
echo "exec rc=$rc"; grep -o '"step": "[A-Za-z-]*"' "$T/out.jsonl" | sort | uniq -c
grep FAILED "$T/out.jsonl" | cut -c1-1500 || true
ls -la "$T"; find "$T/recipes" -type f | sort; head -5 "$T"/recipes/*/MANIFEST.tsv
$PY - "$DB" <<'PY'
import sys; sys.path.insert(0, "..")
import omni_lab_db as db
c = db.connect("rw", dbname=sys.argv[1], app="selftest")
cur = c.cursor()
cur.execute("SELECT model_key, role, layout, status, parameter_count, delete_reason IS NOT NULL, rebuild_recipe_path IS NOT NULL, deleted_at IS NOT NULL FROM model ORDER BY id")
for r in cur.fetchall(): print("model", r)
cur.execute("SELECT unit_path, status, digest_mode, file_count, parameter_count, layout, jsonb_object_keys(file_digests) FROM model_deletion ORDER BY id")
for r in cur.fetchall(): print("del", r)
cur.execute("SELECT count(*) FILTER (WHERE executed_at IS NOT NULL AND verified_absent_at IS NOT NULL AND df_after IS NOT NULL) FROM model_deletion")
print("verified", cur.fetchone())
c.close()
from psycopg2 import sql
a = db.connect("rw", dbname="omni_lab", app="selftest-drop", autocommit=True)
a.cursor().execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(sys.argv[1]))); a.close()
print("dropped", sys.argv[1])
PY
rm -rf --one-file-system -- "$T"
