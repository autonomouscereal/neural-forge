#!/usr/bin/env python3
"""Merge the before-fix and after-fix probe runs into one receipt.

Writes ``logs/probe-K2KV-L27.json``: the four measured stages (kv-only and
full-group at layer 27, before and after the fix), the parent's own per-layer
norm-gain census that says WHY layer 27 and not 3..23, and a finding block whose
every number is a leaf of one of those.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

W = Path("/home/cereal/omni-campaign/qwen-fold")
PARENT = Path("/home/cereal/models/Qwen3.5-4B")
FULL_POSITIONS = (3, 7, 11, 15, 19, 23, 27, 31)


def gain_census() -> list[dict]:
    """``1 + weight`` per full-attention layer, read off the parent's own files."""

    index = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    for layer in FULL_POSITIONS:
        row: dict = {"layer": layer}
        for which in ("q_norm", "k_norm"):
            name = f"model.language_model.layers.{layer}.self_attn.{which}.weight"
            if name not in index:
                name = f"model.layers.{layer}.self_attn.{which}.weight"
            with safe_open(str(PARENT / index[name]), framework="pt") as handle:
                weight = handle.get_tensor(name)
            gain = 1.0 + weight.to(torch.float32)
            zero = torch.nonzero(gain == 0).flatten()
            row[which] = {
                "dtype": str(weight.dtype),
                "min_abs_effective_gain": float(gain.abs().min()),
                "coordinates_with_gain_exactly_zero": zero.tolist(),
            }
        rows.append(row)
    return rows


def main() -> int:
    before = json.loads((W / "logs/probe-K2KV-L27-before-fix.json").read_text())
    after = json.loads((W / "logs/probe-K2KV-L27-fixed.json").read_text())
    toggle = json.loads((W / "logs/probe-K2KV-L27-fix-toggle.json").read_text())

    out = dict(before)
    out["stages"] = dict(before["stages"])
    out["stages"].update(after["stages"])
    out["fix_toggle"] = toggle
    out["runs"] = {
        "before_fix": {
            "stages": ["A", "B"],
            "qf_attn_sha256": "6dceff7503f302b0b8c90a359ace5120fe751295d1d6d08b062dfe1b825b6406",
            "outcome": "both arms reproduced the crash at layer 27",
        },
        "after_fix": {
            "stages": ["A_fixed", "B_fixed"],
            "qf_attn_sha256": "142a5a239ad18ea1a7b4cd141fbc7f05909e328817a059a088263663be951faf",
            "outcome": (
                "NO DIAGNOSTICS -- both stages raised TypeError inside the PROBE, "
                "not inside the lane: qf_attn._resolve_writer had gained a `log=` "
                "keyword and this probe's monkeypatch was still pinned to the old "
                "arity.  40 minutes of GPU produced two stage_error entries and "
                "nothing else.  They are kept here rather than deleted because a "
                "receipt that hides its own wasted runs is not a receipt.  The "
                "after-fix evidence is `fix_toggle` below (the same two lines, on "
                "the parent's own layer-27 weights, CPU only) and the K2KV build "
                "in logs/build-K2KV.log."
            ),
        },
    }
    out["stage_C_not_run"] = (
        "The planned third stage -- layers 3,7,11,15,19,23,27 merged "
        "sequentially, the build's own order -- was started and stopped after "
        "stages A and B had already reproduced the identical failure at layer 27 "
        "with layer 27 as the ONLY merged layer.  The pathology is therefore not "
        "accumulated drift from the upstream folds; it is local to layer 27's own "
        "merge, and the parent gain census below says why.  Stage C would have "
        "cost ~2 h of the queue's GPU to confirm a result already isolated."
    )
    out["parent_norm_gain_census"] = gain_census()
    out["finding"] = {
        "root_cause": (
            "qf_attn.kv_head_merge divides the solved query rows by the q_norm "
            "EFFECTIVE gain (1 + weight, the factor Qwen3_5RMSNorm.forward "
            "applies).  Qwen3.5-4B layer 27's q_norm has one coordinate -- index "
            "1 -- whose bf16 weight is exactly -1.0, so that gain is exactly 0.  "
            "Coordinate 1 is the real half of rotary plane 1, so the gain-scaled "
            "row is zero while the plane is not, and the complex coefficient "
            "qf_solve.joint_relation_solve applies to the plane returns a "
            "non-zero value into that row.  non-zero / 0 = +-Inf."
        ),
        "chain": [
            "child q_proj carries +-Inf in one row per query head (kv-only: 16 "
            "rows x 2560 = 40960 entries; full-group: 8 x 2560 = 20480, and the "
            "energy-weighted sum of +Inf and -Inf makes those NaN)",
            "q_proj_out is non-finite for that coordinate at every token",
            "q_norm's rsqrt(mean(x^2)) over the head is then Inf, and Inf * 0 is "
            "NaN, so the whole head's query is NaN",
            "logits, softmax and the attention output are NaN, so 100% of the "
            "gated o_proj input stream is non-finite",
            "A^T A is all NaN; lam = 1e-6 * mean(diag(G)) is NaN; G + lam I is "
            "all NaN; cholesky reports 'the leading minor of order 1 is not "
            "positive-definite', which is about channel 0 only because that is "
            "where LAPACK looks first",
        ],
        "why_layer_27_and_not_3_to_23": (
            "parent_norm_gain_census: layer 27 is the ONLY full-attention layer "
            "of this parent with an effective q_norm gain of exactly zero"
        ),
        "both_arms_affected": True,
        "fix": (
            "qf_attn.unfold_gain installs the PARENT's own raw row at a "
            "zero-gain coordinate instead of dividing.  The score is unchanged "
            "(the norm multiplies that coordinate by zero either way) and the "
            "per-token RMS the norm divides the head by is the parent's.  "
            "Recorded per layer as zero_gain_coordinates."
        ),
        "defence_in_depth": (
            "qf_solve.prior_anchored_gram now refuses with the offending "
            "channels named; law_abiding_solve* fall back per rung to "
            "eigen_anchored_gram with an explicit recorded eigenvalue floor "
            "(solver_fallback); qf_attn._resolve_writer drops and counts "
            "non-finite captured rows (stream_repair)."
        ),
    }

    target = W / "logs/probe-K2KV-L27.json"
    target.write_text(json.dumps(out, indent=2), encoding="ascii")
    print(f"wrote {target}")
    print(f"stages: {sorted(out['stages'])}")
    t = out["fix_toggle"]
    print(f"  toggle: old q_hats/gq -> {t['old_line_q_hats_over_gq']['total_nonfinite']} non-finite; "
          f"unfold_gain -> {t['new_line_unfold_gain']['total_nonfinite']}")
    for layer in out["parent_norm_gain_census"]:
        print(
            f"  L{layer['layer']:03d}  q_norm min|1+w| {layer['q_norm']['min_abs_effective_gain']:<12g}"
            f" zeros {layer['q_norm']['coordinates_with_gain_exactly_zero']}"
            f"   k_norm min|1+w| {layer['k_norm']['min_abs_effective_gain']:<12g}"
            f" zeros {layer['k_norm']['coordinates_with_gain_exactly_zero']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
