"""Prepare a reviewed D1 receipt fix without overwriting the live campaign.

The patch pins the inspected 2026-10-13 source, attaches preregistration to the
resume identity, and writes the first sealed run receipt before training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BASELINE = {
    'tools/d1_trunk_recovery.py': 'aeb90e2b2964f2fd0dd0d003bf54b70c760c4f34c5beb0b339e84c51aefec967',
    'queues/d1_launch.sh': 'eb98df206152473df1ddf342795ecbe33df727250018e78e5b41b5a693a458d2',
}

PIN_HELPER = '''def preregistration_record(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind production runs to the frozen body and any dated amendments.

    Small standalone experiments can omit the argument; the production launcher
    always supplies it. Verification happens before opening checkpoints or
    creating output directories.
    """
    if not args.preregistration:
        return None
    import stage2_training_receipt as s2
    try:
        record = s2.assert_preregistration_unedited(Path(args.preregistration))
    except s2.Stage2TrainingReceiptError as exc:
        raise Refusal(f"preregistration refused: {exc}") from exc
    return {**record, "path": str(Path(args.preregistration).resolve())}


'''


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError('source anchor is missing or ambiguous')
    return text.replace(old, new, 1)


def patch_trainer(text: str) -> str:
    text = replace_once(text, 'TOOL_VERSION = "1.2.4"', 'TOOL_VERSION = "1.2.5"')
    text = replace_once(text, 'def run(argv: list[str] | None = None) -> RunState:', PIN_HELPER + 'def run(argv: list[str] | None = None) -> RunState:')
    text = replace_once(text, '    args = build_parser().parse_args(argv)\n    started = time.time()', '    args = build_parser().parse_args(argv)\n    preregistration = preregistration_record(args)\n    started = time.time()')
    text = replace_once(text, '"ids": ids_record, "held_rows": held_n, "layout": layout.__dict__,', '"ids": ids_record, "held_rows": held_n, "layout": layout.__dict__,\n                "preregistration": preregistration,')
    text = replace_once(text, '"run_identity_sha256": rs.chain, "devices": devices, "text_rows": text_rows,', '"run_identity_sha256": rs.chain, "devices": devices, "text_rows": text_rows,\n                  "preregistration": preregistration,')
    text = replace_once(text, '    for d in devices:\n        if d.startswith("cuda"):\n            props = torch.cuda.get_device_properties', '    # Persist the pinned initial receipt before any training step can run.\n    rs.receipt["initial_receipt_before_first_step"] = True\n    save_receipt(rs)\n    for d in devices:\n        if d.startswith("cuda"):\n            props = torch.cuda.get_device_properties')
    text = replace_once(text, '    ap.add_argument("--parent", required=True,', '    ap.add_argument("--preregistration", default=None, help="frozen stage-D document; mandatory in production launcher")\n    ap.add_argument("--parent", required=True,')
    ast.parse(text)
    return text


def patch_launcher(text: str) -> str:
    text = replace_once(text, 'LAUNCHER_VERSION=d1_launch-1.1.0', 'LAUNCHER_VERSION=d1_launch-1.1.1')
    return replace_once(text, 'D1_ARGS=(--child "@BASE@" --parent "$PARENT"', 'D1_ARGS=(--child "@BASE@" --preregistration "$PREREG" --parent "$PARENT"')


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('candidate output must be outside the campaign source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('candidate directory must be empty')
    inputs = {}
    for relative, expected in BASELINE.items():
        raw = (source / relative).read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise ValueError(f'baseline SHA changed: {relative}; review the new source before repatching')
        inputs[relative] = raw
    # Verify both inputs before writing either candidate.
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'chimera.d1-receipt-fix.v1', 'source': str(source), 'files': []}
    for relative, raw in inputs.items():
        fn = patch_trainer if relative.endswith('.py') else patch_launcher
        patched = fn(raw.decode('utf-8')).encode('utf-8')
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(patched)
        receipt['files'].append({'path': relative, 'baseline_sha256': BASELINE[relative], 'candidate_sha256': hashlib.sha256(patched).hexdigest()})
    (output/'patch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
