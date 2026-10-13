"""Prepare additive, source-pinned D1 candidate files; never deploy or launch.

The predecessor must be the exact reviewed v2 launcher/trainer. Runtime queue,
gates, exclusive GPU locks and original optimizer are retained. The source
extractor is replaced with admission of the independently sealed successor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import tempfile

try:
    from . import d1_source_contract as C
except ImportError:
    import d1_source_contract as C

PREDECESSOR = {"queues/d1_launch.sh": "3700300bd3ddd49c3d8bcb9c0965bc821ecf650a8b282f61f32708d074414772",
               "tools/d1_trunk_recovery.py": "5f63aab8d795b484b93287d70831871545ee0c8a59565e218351ffc497519b7f"}


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("reviewed D1 source anchor differs; do not adapt unknown runtime code")
    return text.replace(before, after, 1)


def trainer_candidate(raw):
    text = raw.decode().replace("\r\n", "\n")
    text = replace_once(text, 'TOOL_VERSION = "1.2.5"', 'TOOL_VERSION = "1.2.6-source-successor"')
    text = replace_once(text, "    preregistration = preregistration_record(args)\n    started = time.time()",
                        "    preregistration = preregistration_record(args)\n"
                        "    import d1_source_contract as source_contract\n"
                        "    # Source admission runs before checkpoint loading/output creation/updates.\n"
                        "    source_record = source_contract.validate_args(args, Path(__file__))\n"
                        "    started = time.time()")
    text = replace_once(text, '                "preregistration": preregistration,\n                "hyper":',
                        '                "preregistration": preregistration, "source_contract": source_record,\n                "hyper":')
    text = replace_once(text, '                  "preregistration": preregistration,\n                  "rows":',
                        '                  "preregistration": preregistration, "source_contract": source_record,\n                  "rows":')
    anchor = '    ap.add_argument("--held-rows", type=int, default=None, help="last N rows held out of training (default max(1, rows // 16))")'
    text = replace_once(text, anchor, anchor + '\n    ap.add_argument("--source-manifest", default=None)\n'
                        '    ap.add_argument("--source-manifest-sha256", default=None)\n'
                        '    ap.add_argument("--source-validator-sha256", default=None)')
    compile(text, "d1_trunk_recovery.py", "exec")
    return text.encode()


def launcher_candidate(raw, manifest, source_sha, runtime_manifest, runtime_candidate, validator_sha, trainer_sha):
    text = raw.decode().replace("\r\n", "\n")
    text = replace_once(text, "LAUNCHER_VERSION=d1_launch-1.1.1", "LAUNCHER_VERSION=d1_launch-source-successor-20261013")
    start, end = text.index("PREREG=$OC/distill/PREREGISTRATION"), text.index("K3_ROOT=")
    block = (f"CANDIDATE_ROOT={shlex.quote(runtime_candidate)}\nCT=$CANDIDATE_ROOT/tools\n"
             f"SOURCE_MANIFEST={shlex.quote(runtime_manifest)}\nSOURCE_MANIFEST_SHA={source_sha}\n"
             f"SOURCE_VALIDATOR_SHA={validator_sha}\nD1_TRAINER_SHA={trainer_sha}\n"
             f"PREREG={shlex.quote(manifest['preregistration']['path'])}\n"
             f"IDS={shlex.quote(manifest['dataset']['path'])}\n"
             f"ROWS={manifest['actual_counts']['total_rows']}\nHELD_ROWS={manifest['actual_counts']['held_rows']}\n"
             "RUN=$OC/distill/d1/run-OMNI1D1-source-successor-20261013\n"
             "DELTA=/mnt/lexar/omni-campaign/deltas/${DELTA_ID}-source-successor-20261013\n"
             "PRE=$CANDIDATE_ROOT/preflight\nLOG=$CANDIDATE_ROOT/d1.log\nSTATUS=$CANDIDATE_ROOT/d1_launch.status\n")
    text = text[:start] + block + text[end:]
    text = replace_once(text, "PYTHONPATH=$T:$OC/ledger:$EF/src", "PYTHONPATH=$CT:$T:$OC/ledger:$EF/src")
    text = replace_once(text, 'mkdir -p "$PRE" "$Q/d1"', 'mkdir -p "$PRE"')
    text = replace_once(text, '--ids "$IDS" --lengths-key lengths --layers all',
                        '--ids "$IDS" --seq-len 4096 --lengths-key lengths --held-rows "$HELD_ROWS" --layers all\n'
                        '         --source-manifest "$SOURCE_MANIFEST" --source-manifest-sha256 "$SOURCE_MANIFEST_SHA"\n'
                        '         --source-validator-sha256 "$SOURCE_VALIDATOR_SHA"')
    text = replace_once(text, '"$IDS" "$TEXT" "$ROWS" "$PRE/preflight.json"', '"$IDS" "$SOURCE_MANIFEST" "$ROWS" "$PRE/preflight.json"')
    text = replace_once(text, 'base, label, ids_path, text_dir, rows, receipt_path, prereg = sys.argv[1:8]',
                        'base, label, ids_path, source_manifest, rows, receipt_path, prereg = sys.argv[1:8]')
    text = replace_once(text, 'import d1_trunk_recovery as d1\n', 'import d1_trunk_recovery as d1\nimport d1_source_contract as contract\n')
    lo, hi = text.index('# 2. the ids:'), text.index('# 3. d1\'s own argument parser')
    text = text[:lo] + ('# 2. admit the exact successor; never extract or overwrite original text rows\n'
                       'args = d1.build_parser().parse_args(d1_args)\n'
                       'out["source_contract"] = contract.validate_args(args, Path(d1.__file__))\n'
                       'ids, lengths, rec = d1.load_ids(args.ids, args.seq_len, args.max_rows, args.lengths_key)\n'
                       'out["ids"] = {"path": ids_path, "sha256": sha(ids_path), "rows": int(ids.shape[0]),\n'
                       '              "seq": int(ids.shape[1]), "positions": int(ids.numel()), "valid_tokens": int(lengths.sum())}\n'
                       'print(f"IDS {ids.shape[0]}x{ids.shape[1]}, real {int(lengths.sum()):,}", flush=True)\n') + text[hi:]
    text = replace_once(text, 'held = max(1, R // 16)', 'held = int(args.held_rows)')
    text = replace_once(text, '"fit_tokens": (R - held) * Lq',
                        '"fit_tokens": int(got_len[:R-held].sum()), "held_tokens": int(got_len[R-held:].sum()),\n'
                        '                     "padded_positions": R * Lq - int(got_len.sum())')
    # Recheck exact tool bytes and every source immediately before invoking trainer.
    function = ('verify_source() {\n'
                '  CUDA_VISIBLE_DEVICES="" $PY - "$CT/d1_source_contract.py" "$SOURCE_VALIDATOR_SHA" "$CT/d1_trunk_recovery.py" "$D1_TRAINER_SHA" <<\'SOURCE_CODE\'\n'
                'import hashlib, sys\nfrom pathlib import Path\n'
                'for path, expected in zip(sys.argv[1::2], sys.argv[2::2]):\n'
                '    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:\n'
                '        raise SystemExit("D1 candidate code pin changed: " + path)\nSOURCE_CODE\n'
                '  [ $? -eq 0 ] || return 2\n'
                '  CUDA_VISIBLE_DEVICES="" $PY "$CT/d1_source_contract.py" --manifest "$SOURCE_MANIFEST" --sha256 "$SOURCE_MANIFEST_SHA" > "$PRE/source_contract.json"\n}\n\n')
    text = replace_once(text, '# ---------------------------------------------------------------- preflight', function + '# ---------------------------------------------------------------- preflight')
    text = replace_once(text, 'preflight() {   # $1 = base directory to check against; $2 = label\n',
                        'preflight() {   # $1 = base directory to check against; $2 = label\n'
                        '  verify_source || return 2\n')
    text = text.replace('$PY $T/d1_trunk_recovery.py --help', '$PY $CT/d1_trunk_recovery.py --help')
    text = replace_once(text, 'D1_RUN_ARGS=("${D1_ARGS[@]}")',
                        'verify_source || { status "FAIL: source/code changed after resource wait"; release_card_locks; exit 2; }\n'
                        'D1_RUN_ARGS=("${D1_ARGS[@]}")')
    text = replace_once(text, 'cd $T\nCUDA_VISIBLE_DEVICES=0,4', 'cd "$CT"\nCUDA_VISIBLE_DEVICES=0,4')
    text = text.replace('$Q/d1/base_record.json', '$CANDIDATE_ROOT/base_record.json')
    text = text.replace('$Q/d1/OMNI1D1_plan.json', '$CANDIDATE_ROOT/OMNI1D1_plan.json')
    text = text.replace('$Q/d1_launch.sh (d1_trunk_recovery', '$CANDIDATE_ROOT/queues/d1_launch.sh (d1_trunk_recovery')
    text = text.replace('DELTA_ID=OC02-D-Q38-27BU-OMNI1D1', 'DELTA_ID=OC02-D-Q38-27BU-OMNI1D1-source-successor-20261013')
    text = text.replace('${DELTA_ID}-source-successor-20261013', '$DELTA_ID')
    # Replace stale extractor documentation, keeping complete runtime gate documentation.
    text = text.replace('the first 977 rows of\n# distill/corpus/text (4,001,792 tokens, PREREGISTRATION_OMNI_STAGE_D_v1 section 3 D1)',
                        'the pinned source-isolated successor with explicit real lengths and held rows')
    text = text.replace('$Q/d1/preflight/ (the receipt and the 31 MB ids file the real run reuses)',
                        '$CANDIDATE_ROOT/preflight/ (receipts only; successor tensors remain immutable)')
    return text.encode()


def prepare(baseline, source_manifest, source_sha, runtime_manifest, runtime_candidate, out):
    if out.exists():
        raise ValueError("additive new candidate directory required")
    for key, expected in PREDECESSOR.items():
        if C.sha(baseline / key) != expected:
            raise ValueError("reviewed D1 predecessor SHA differs")
    for value in (runtime_manifest, runtime_candidate):
        if not PurePosixPath(value).is_absolute() or "\n" in value or "\r" in value:
            raise ValueError("explicit absolute remote runtime paths required")
    manifest = C.manifest_record(source_manifest, source_sha)
    trainer = trainer_candidate((baseline / "tools/d1_trunk_recovery.py").read_bytes())
    validator = Path(C.__file__).read_bytes()
    files = {"tools/d1_trunk_recovery.py": trainer, "tools/d1_source_contract.py": validator}
    files["queues/d1_launch.sh"] = launcher_candidate((baseline / "queues/d1_launch.sh").read_bytes(), manifest, source_sha,
                                                       runtime_manifest, runtime_candidate, hashlib.sha256(validator).hexdigest(),
                                                       hashlib.sha256(trainer).hexdigest())
    receipt = dict(schema="chimera_recovery.d1_candidate.v1", predecessor=PREDECESSOR, successor_manifest_sha256=source_sha,
                   successor_seal_sha256=manifest["seal_sha256"], runtime_manifest=runtime_manifest, runtime_candidate=runtime_candidate,
                   actual_counts=manifest["actual_counts"], files={key: hashlib.sha256(raw).hexdigest() for key, raw in files.items()},
                   training_launched=False, runtime_source_changed=False,
                   qtext_optional_corpus_args=["--corpus", manifest["dataset"]["path"], "--ids-key", "input_ids", "--lengths-key", "lengths"],
                   qtext_cache_scope="optional separate KD; no historical cache may be reused against successor SHA",
                   d1_teacher_scope="direct immutable parent hidden-state forwards; no qtext logit-cache dependency")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as temporary:
        root = Path(temporary)
        for key, raw in files.items():
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "candidate_receipt.json").write_bytes(C.canonical(receipt) + b"\n")
        root.rename(out)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "source-manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha", "runtime-manifest", "runtime-candidate"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
