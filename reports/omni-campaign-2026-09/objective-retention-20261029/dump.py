"""Read-only dump of fold_arm / fold_behaviour / fold_diagnostic / fold_component for the
objective-vs-retention analysis.  Raw parameterised SQL through omni_lab_db (role omni_lab_ro)."""
import json, sys
sys.path.insert(0, r"C:/Users/cereal/Documents/ai training/omni-lab")
import omni_lab_db as db

OUT = r"C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462/scratchpad/ora/dump.json"
Q = {
 "arm": "SELECT experiment_id, arm_key, title, program, wave, owner, parent_name, child_ref, ratio_body, ratio_net, ratio_text, "
        "objective_fit, objective_target, objective_k, objective_horizon, objective_text, training_used, training_text, "
        "verdict_class, verdict_text, axes FROM fold_arm",
 "beh": "SELECT id, experiment_id, ordinal, suite, scorer, score, score_value, score_denominator, useful, partial, miss, total, "
        "column_b, parent_score, text FROM fold_behaviour",
 "diag": "SELECT id, experiment_id, ordinal, name, value_text, value_num, value_unit, horizon, denominator, split, note FROM fold_diagnostic",
 "comp": "SELECT experiment_id, ordinal, kind, layers, technique_text, ratio, ratio_text, rank_or_k FROM fold_component",
}
conn = db.connect("ro", app="objective-retention-dump")
out = {}
try:
    with conn.cursor() as cur:
        for k, q in Q.items():
            cur.execute(q)
            cols = [d[0] for d in cur.description]
            out[k] = [dict(zip(cols, r)) for r in cur.fetchall()]
            print(k, len(out[k]))
    conn.rollback()
finally:
    conn.close()
json.dump(out, open(OUT, "w", encoding="utf-8"), default=str)
