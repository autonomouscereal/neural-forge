#!/usr/bin/env bash
# selftest_130.sh - cleanup_exec.py 1.3.0 additions (--overrides, --pre-delete-digest) end to end on synthetic units
# against a throwaway omni_lab_test_<hex> database. Expected: C (waived receipts dir) and A (survivor verified)
# DELETED; D (receipts dir, no waiver) BLOCKED; E (survivor digest mismatch) FAILED with nothing recorded and E
# still on disk; the gate-level reference waiver clears only the named source.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . /home/cereal/.config/omni-lab/env; set +a
PY=/home/cereal/.venvs/chimera-intake/bin/python
HEX=$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')
T=/mnt/lexar/omni-campaign/scratch/cleanup-selftest130-$HEX
DB=omni_lab_test_$HEX
mkdir -p "$T/unitA" "$T/unitC/receipts" "$T/unitD/receipts" "$T/unitE" "$T/survivor" "$T/recipes"
$PY - "$T" <<'PY'
import json, struct, sys
t = sys.argv[1]
hdr = json.dumps({"w": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128]}}).encode()
for u in ("unitA", "unitC", "unitD", "unitE", "survivor"):
    with open(f"{t}/{u}/model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)) + hdr + b"\0" * 128)
for u in ("unitC", "unitD"):
    json.dump({"ok": True}, open(f"{t}/{u}/receipts/r.json", "w"))
def unit(name, idx):
    return {"path": f"{t}/{name}", "volume": "lexar", "size_bytes_du": 4096, "size_gb": 0.0, "what": "scratch",
            "reason": "selftest 1.3.0", "omni_lab": {"ledger_checkpoint_ids": []},
            "recipe_archive": f"{t}/x/path-{idx}__{name}/", "plan_index": idx}
json.dump([unit("unitA", 9101), unit("unitC", 9103), unit("unitD", 9104)], open(f"{t}/units.json", "w"))
json.dump([unit("unitE", 9105)], open(f"{t}/units_e.json", "w"))
json.dump([{"class": "never-delete", "path": f"{t}/recipes"}, {"class": "never-delete", "path": f"{t}/survivor"}],
          open(f"{t}/protected.json", "w"))
json.dump(["/", "/mnt", "/mnt/lexar", "/mnt/lexar/omni-campaign", "/mnt/lexar/omni-campaign/scratch", t],
          open(f"{t}/generic.json", "w"))
json.dump({"units": {"9103": {"datadir_prefixes": {"receipts": "selftest: receipts dir archived in the recipe tar"}}}},
          open(f"{t}/overrides.json", "w"))
PY
find "$T" -exec touch -h -d '2026-10-01 00:00' {} +
SURV=$($PY -c "import sys; sys.path.insert(0,'/home/cereal/omni-campaign/ledger'); from oc_seal import directory_digest; print(directory_digest('$T/survivor')[0])")
echo "survivor digest $SURV"
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
   --overrides "$T/overrides.json" --pre-delete-digest "9101:$T/survivor:$SURV" \
   --volume lexar --run-dir "$T/run" --recipe-root "$T/recipes" > "$T/out.jsonl" 2>&1 || rc=$?
echo "exec rc=$rc"; grep -o '"step": "[A-Za-z-]*"' "$T/out.jsonl" | sort | uniq -c
rc2=0
$PY cleanup_exec.py execute --units "$T/units_e.json" --protected "$T/protected.json" --generic "$T/generic.json" \
   --pre-delete-digest "9105:$T/survivor:$(printf '0%.0s' $(seq 64))" \
   --volume lexar --run-dir "$T/run_e" --recipe-root "$T/recipes" > "$T/out_e.jsonl" 2>&1 || rc2=$?
echo "exec(E, wrong survivor digest) rc=$rc2"; grep -o '"step": "[A-Za-z-]*"' "$T/out_e.jsonl" | sort | uniq -c
$PY - "$DB" "$T" <<'PY'
import json, os, sys; sys.path.insert(0, "..")
import omni_lab_db as db
t = sys.argv[2]
fails = []
def want(cond, what):
    print(("PASS " if cond else "FAIL ") + what)
    if not cond: fails.append(what)
c = db.connect("rw", dbname=sys.argv[1], app="selftest")
cur = c.cursor()
cur.execute("SELECT unit_path, status, preflight FROM model_deletion ORDER BY id")
rows = {os.path.basename(r[0]): (r[1], r[2]) for r in cur.fetchall()}
want(rows.get("unitA", ("",))[0] == "deleted" and not os.path.lexists(f"{t}/unitA"), "A deleted after survivor check")
want(rows.get("unitA", (None, {}))[1].get("surviving_copies_verified", [{}])[0].get("dir") == f"{t}/survivor",
     "A preflight records the verified survivor")
want(rows.get("unitC", ("",))[0] == "deleted" and not os.path.lexists(f"{t}/unitC"), "C deleted under datadir waiver")
want("receipts" in rows.get("unitC", (None, {}))[1].get("overrides_applied", {}).get("datadirs", {}),
     "C preflight records the waiver")
want("unitD" not in rows and os.path.isdir(f"{t}/unitD"), "D (no waiver) blocked, untouched, unrecorded")
want("unitE" not in rows and os.path.isdir(f"{t}/unitE"), "E (survivor mismatch) untouched, unrecorded")
want(os.path.isdir(f"{t}/survivor"), "survivor untouched")
c.close()
# gate-level reference waiver: real gate(), environment scanners replaced by fixed token lists
sys.path.insert(0, ".")
import cleanup_exec as X
X.crontab_tokens = lambda: []
X.docker_mount_tokens = lambda: []
class Refs:
    cache = {}
    def file_tokens(self):
        return [("/fake/receipt.json", f"{t}/unitD/model.safetensors"), ("/fake/live_tool.sh", f"{t}/unitD")]
u = {"path": f"{t}/unitD", "plan_index": 1}
scan = {"open": [], "strings": [], "errors": 0}
ov = {1: {"ref_sources": {"/fake/receipt.json": "historical receipt, not a reader"},
          "datadir_prefixes": {"receipts": "selftest waiver for the receipts dir"}}}
g = X.gate(u, [], set(), Refs(), scan, ov)
want(len(g["reasons"]) == 1 and "live_tool.sh" in g["reasons"][0] and "receipt.json" not in g["reasons"][0],
     f"ref waiver clears only the named source: {g['reasons']}")
want(g["checked"]["overrides_applied"]["ref_sources"]["/fake/receipt.json"]["tokens"] == 1, "ref waiver recorded")
from psycopg2 import sql
a = db.connect("rw", dbname="omni_lab", app="selftest-drop", autocommit=True)
a.cursor().execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(sys.argv[1]))); a.close()
print("dropped", sys.argv[1])
print("SELFTEST", "FAIL" if fails else "PASS", len(fails))
PY
rm -rf --one-file-system -- "$T"
