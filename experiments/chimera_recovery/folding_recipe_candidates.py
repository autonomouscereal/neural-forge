"""Generate separate, pinned future recipe corrections; never fold/load weights."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

BASELINES = {
    "receipt": "c7fb1b0245d556c3af304ca6aeb9d78d9c294d39c682bcf93576e43488017257",
    "unit": "36df14de22b363e665d130fc5d6619ceb40e5c57ba98ec4c5ac22cfd72624c49",
}
# Independently reviewed 2026-10-13 live capture. Its only receipt diff is the
# absence of the mirror's FP32 nonpersistent RoPE-buffer preservation; the
# compare_prompts body is unchanged. This candidate fixes measurement admission
# only and preserves each baseline's block construction without choosing a RoPE
# policy. Unknown source bytes still fail closed.
REVIEWED_BASELINES = {
    "receipt": {
        BASELINES["receipt"]: "historical_2026_campaign_mirror_fp32_rope_preservation",
        "9fdf2db07651dd424e90d089fa528e59ea9c52844338d02bb9d2e988c089274e":
            "live_2026_10_13_capture_no_fp32_rope_preservation",
    },
    "unit": {BASELINES["unit"]: "reviewed_2026_campaign_unit_builder"},
}


class Refusal(ValueError):
    pass


MEASUREMENT_HELPER = '''
def validate_prompt_measurement(ids, reference, produced_full, reference_taps, produced_taps, *, base_vocab, layers_reported):
    """Reject incomplete/non-finite parity observations before maxima/argmax."""
    import torch
    if len(reference.shape) != 3 or len(produced_full.shape) != 3:
        raise Refusal("parity logits must have [B,T,V] shape")
    if reference.shape[:2] != produced_full.shape[:2] or tuple(reference.shape[:2]) != (1, len(ids)):
        raise Refusal("parity logit token geometry differs")
    if base_vocab < 1 or reference.shape[-1] < base_vocab or produced_full.shape[-1] < base_vocab:
        raise Refusal("parity base vocabulary is incomplete")
    if not bool(torch.isfinite(reference[..., :base_vocab]).all()) or not bool(torch.isfinite(produced_full[..., :base_vocab]).all()):
        raise Refusal("non-finite base logits cannot produce a parity receipt")
    band = produced_full[..., base_vocab:]
    if band.numel() and not bool(torch.isneginf(band).all()):
        raise Refusal("import-only band mask must be exactly negative infinity; NaN/+Inf/finite are refused")
    if type(layers_reported) is not int or layers_reported < 0:
        raise Refusal("invalid requested hidden tap count")
    required = set(range(layers_reported))
    if not required.issubset(reference_taps) or not required.issubset(produced_taps):
        raise Refusal("requested hidden-state tap coverage is incomplete")
    for layer in sorted(required):
        left, right = reference_taps[layer], produced_taps[layer]
        if (left is None or right is None or left.ndim != 3 or right.ndim != 3 or left.shape[-1] < 1 or
                left.shape != right.shape or tuple(left.shape[:2]) != (1, len(ids))):
            raise Refusal("requested hidden-state tap geometry differs")
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            raise Refusal("non-finite hidden-state tap cannot produce a parity receipt")

'''


def patch_source(kind: str, source: Path) -> tuple[str, dict]:
    return patch_bytes(kind, source.read_bytes())


def patch_bytes(kind: str, raw: bytes) -> tuple[str, dict]:
    actual = hashlib.sha256(raw).hexdigest()
    if kind not in REVIEWED_BASELINES or actual not in REVIEWED_BASELINES[kind]:
        raise Refusal("source baseline differs; inspect before generating a correction")
    before = raw.decode().replace("\r\n", "\n")
    if kind == "receipt":
        replacements = [
            ("def compare_prompts(\n", MEASUREMENT_HELPER + "def compare_prompts(\n"),
            ("    prompts: list[dict[str, Any]] = []\n    worst_logit = 0.0\n",
             '    if not packs or any(len(rows) != len(packs) for rows in (reference_logits, reference_hidden, produced_logits, produced_hidden)):\n        raise Refusal("parity observation count differs from requested prompt pack")\n    prompts: list[dict[str, Any]] = []\n    worst_logit = 0.0\n'),
            ('        produced_full = produced_logits[prompt_index]\n        produced = produced_full[..., :base_vocab]\n',
             '        produced_full = produced_logits[prompt_index]\n        validate_prompt_measurement(ids, reference, produced_full, reference_hidden[prompt_index], produced_hidden[prompt_index], base_vocab=base_vocab, layers_reported=layers_reported)\n        produced = produced_full[..., :base_vocab]\n'),
            ('            if left is None or right is None:\n                continue\n',
             '            if left is None or right is None:\n                raise Refusal("requested hidden-state tap missing after validation")\n'),
        ]
    else:
        replacements = [
            ('               block0_capture: Path | None = None) -> dict[str, Any]:\n',
             '               block0_capture: Path | None = None, time_reference_parent: Path | None = None) -> dict[str, Any]:\n'),
            ('    token = hy3d_time_token(parent, C16._torch().tensor(HY3D_TIME_TOKEN_SIGMAS))\n',
             '    time_parent = parent if time_reference_parent is None else time_reference_parent\n    token = hy3d_time_token(time_parent, C16._torch().tensor(HY3D_TIME_TOKEN_SIGMAS))\n    if token["time_token"].shape[-1] != width or not bool(torch.isfinite(token["time_token"]).all()):\n        raise C16.Refusal("HY3D time-reference parent width/finite contract differs from installed block")\n'),
            ('        "reference_weights": token["weights"],\n',
             '        "reference_weights": token["weights"],\n        "reference_source_weights": str(time_parent),\n        "reference_source_weights_sha256": sha256_file(time_parent),\n'),
            ('"reference_vs_block0_capture": (hy3d_reference_vs_capture(parent, block0_capture)',
             '"reference_vs_block0_capture": (hy3d_reference_vs_capture(time_parent, block0_capture)'),
            ('                                block0_capture=Path(args.hy3d_block0_capture) if args.hy3d_block0_capture else None)\n',
             '                                block0_capture=Path(args.hy3d_block0_capture) if args.hy3d_block0_capture else None,\n                                time_reference_parent=Path(args.parent) if args.parent else None)\n'),
        ]
    after = before
    for old, new in replacements:
        if after.count(old) != 1:
            raise Refusal("patch anchor missing or ambiguous: " + old[:80])
        after = after.replace(old, new, 1)
    ast.parse(after)
    return after, {"schema": "chimera_recovery.future_recipe_candidate.v1", "kind": kind,
                   "baseline_sha256": actual, "baseline_revision": REVIEWED_BASELINES[kind][actual],
                   "candidate_sha256": hashlib.sha256(after.encode()).hexdigest(),
                   "status": "private source candidate for review", "fold_launched": False,
                   "weights_changed": False, "recovery_checkpoint_changed": False}


def emit(kind: str, source: Path, out: Path) -> dict:
    after, receipt = patch_source(kind, source)
    if any(p.exists() for p in (out, out.with_suffix(".patch"), out.with_suffix(".receipt.json"))) or out.resolve() == source.resolve():
        raise Refusal("new candidate output required; existing recipe is read-only")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(after.encode())
    before = source.read_bytes().decode().replace("\r\n", "\n")
    patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile=source.name + " (pinned)", tofile=out.name))
    out.with_suffix(".patch").write_bytes(patch.encode())
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=sorted(BASELINES))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        print(json.dumps(emit(args.kind, args.source, args.out), sort_keys=True))
        return 0
    except (Refusal, OSError) as e:
        print(json.dumps({"status": "refused", "reason": str(e), "fold_launched": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
