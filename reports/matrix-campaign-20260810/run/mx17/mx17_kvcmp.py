import json, sys
W = "/mnt/ai_data/neural-forge-matrix/wave17"
for tag in ("KV6A", "KV6J", "KV6JQ", "KV4JQ", "X13KV6J", "X13KV6JQ", "F1301KV6", "F1301KV6Q"):
    r = json.load(open("%s/children/MX17-%s/mx17-build-receipt.json" % (W, tag)))
    ax = [a for a in r["axes"] if a["axis"] == "num_attention_heads"][0]
    sites = ax["heldout"]["per_site"]
    vs = [s.get("value_solve") or {} for s in sites]
    installed = sum(1 for v in vs if v.get("installed"))
    refused = sum(1 for v in vs if v.get("refused"))
    gates = [v.get("attention_recompute_relL2") for v in vs if v.get("attention_recompute_relL2") is not None]
    merged = [m for s in sites for m in s["census"]["merged"] if m.get("merged")]
    joint_ok = sum(1 for m in merged if m.get("joint_installed"))
    avg = [m["relation_relL2_average"] for m in merged]
    jnt = [m.get("relation_relL2_joint") for m in merged if m.get("relation_relJ2_joint") is None]
    jnt = [m.get("relation_relL2_joint") for m in merged]
    print("%-10s tensors %s median_cos %.4f  value installed %d/%d refused %d  gate %s  joint pairs %d/%d" % (
        tag, r["child_tensors_sha256"][:16], ax["heldout"]["median_cos"], installed, len(vs), refused,
        ("%.2e-%.2e" % (min(gates), max(gates))) if gates else "-", joint_ok, len(merged)))
    if merged and jnt[0] is not None:
        print("           relation relL2 average median %.4f -> joint median %.4f" % (
            sorted(avg)[len(avg)//2], sorted(jnt)[len(jnt)//2]))
