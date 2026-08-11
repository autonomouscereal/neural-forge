#!/usr/bin/env python3
"""Print one composition receipt the way a reader needs it."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    b = json.loads(path.read_text(encoding="utf-8"))
    c = b["composition"]
    print(f"cell {b['cell_id']} status={b['outcome']['status']} "
          f"fdm={b['outcome']['first_divergent_metric']} "
          f"receipt={b['receipt_sha'][:16]} wall={b['wall_time_seconds']:.1f}s")
    print(f"arch {b['code_identity']['compute_architecture']} "
          f"src_tree {b['code_identity']['src_tree_sha'][:16]} "
          f"repo {b['code_identity']['repo_head']}")
    fr = b["behavior"]["free_run"]
    print(f"COMPOSED free-run {fr['token_match_vs_control']:.6f} "
          f"lcp {fr['mean_lcp']:.6f}   global drift "
          f"{b['composed']['vs_copy_exact_control_delta']:.6f}   "
          f"emit-zero {b['composed']['emit_zero_floor_passed']}   "
          f"norm-ratio {b['composed']['norm_ratio_per_layer_minmax']}")
    bd = b["boundary"]
    print(f"boundary worst write cos {bd['write_heldout_cos']} "
          f"({bd.get('worst_write_component_id')})  worst macro "
          f"{bd['post_operator_heldout_relL2']} "
          f"({bd.get('worst_macrocell_component_id')})  band "
          f"{bd['endpoint_band_le_0p4']}")
    print(f"calibration {json.dumps(b['calibration'])}")
    s = c["storage"]
    print(f"STORAGE child {s['child_parameter_count']} / parent "
          f"{s['parent_parameter_count']} = {s['net_ratio']:.6f}x "
          f"delta {s['net_delta']:+d} contracts={s['contracts']}  "
          f"folded tensors {s['folded_tensors_stored_parameter_count']}/"
          f"{s['folded_tensors_parameter_count']} = "
          f"{(s['folded_tensors_ratio'] or 0):.4f}x")
    print()
    head = ("comp", "cell", "iso-fr", "post-fr", "drift", "dDrift", "wcos",
            "macro", "solve-s", "fdm")
    print(f"{head[0]:5}{head[1]:24}{head[2]:>8}{head[3]:>9}{head[4]:>11}"
          f"{head[5]:>11}{head[6]:>9}{head[7]:>8}{head[8]:>9}  {head[9]}")
    for e in c["components"]:
        a = e["after_install"]
        bo = e["boundary"]
        cond = "*" if e["evidence"].get("conditioned") else " "
        print(
            f"{e['component_id']:5}{e['isolation_cell_id']:24}"
            f"{(e['isolation'].get('free_run') or 0):8.4f}"
            f"{(a['free_run_token_match_vs_control'] or 0):9.4f}"
            f"{a['terminal_relL2_vs_control']:11.6f}"
            f"{a['delta_terminal_vs_previous_step']:+11.6f}"
            f"{(bo['write_heldout_cos'] or 0):9.5f}"
            f"{(bo['post_operator_heldout_relL2'] or 0):8.4f}"
            f"{e['solve'].get('solve_wall_time_seconds', 0):9.1f}"
            f" {cond}{e['first_divergent_metric']}"
        )
    print()
    for e in c["components"]:
        st = e["storage"]
        print(
            f"{e['component_id']:5} stored={st['stored_parameter_count']:>9} "
            f"installed={st['installed_parameter_count']:>9} "
            f"owned={st['owned_parameter_count']:>9} "
            f"declared={st['declared_ratio']:.4f} "
            f"net={st['stored_parameter_count'] - st['installed_parameter_count']:>+9}"
        )
    print()
    for e in c["components"]:
        ev = e["evidence"]
        print(
            f"{e['component_id']:5} conditioned={ev.get('conditioned')} "
            f"cache={str(ev.get('trace_cache_key'))[:16]} "
            f"split={str(ev.get('evidence_split_manifest_sha256'))[:16]} "
            f"rows={ev.get('fit_rows')}/{ev.get('heldout_rows')} "
            f"upstream={len(ev.get('upstream_installed') or [])} "
            f"capture_s={ev.get('capture_wall_time_seconds')}"
        )
    print()
    print("passed through:")
    for e in c["passed_through"]:
        print(f"  {e['component_id']:5} {e['reason']}")
    print()
    for index, sm in enumerate(b["behavior"]["decode_smoke"], start=1):
        print(f"--- smoke {index} [{sm['terminated']}] "
              f"{sm.get('generated_tokens')} tokens")
        print(sm["output_verbatim"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
