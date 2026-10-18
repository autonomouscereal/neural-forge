#!/usr/bin/env python3
"""s7_dryrun.py -- prove S7 media units load, verify and account through c16_install_media.

Lane E4, Claude, 2026-10-09. NOT an install into the real trunk: a TINY synthetic Chimera
child (c16_media_fixtures.write_child, hidden 128, 8 layers kda,kda,kda,gated_gqa,
residual_mode none, SiLU) stands in for the 15B S3 trunk. Hidden 128 is chosen so the ACE
run's own ridge chart (fitted on a 128-wide toy host) can be installed verbatim as one arm.
Every other unit has no chart at any trunk width, so SYNTHETIC paired rows (seeded gaussian
host rows, parent = host @ W + noise) are written under <scratch>/tap/<unit>/ and fitted by
c16's own --tap-rows path (mediafold fit_charts ridge). The resulting charts are meaningless
numbers; what the run proves is the container contract: tensor sets, shapes, declared sha256,
time-reference gates, pairing, header-count accounting and the 4x component split.

    CHIMERA_OMNI_SRC=<tree> PYTHONPATH=<tree>/src python s7_dryrun.py --units U1 U2 ... --scratch DIR

1.1.0 (S7-on-S5 media lane, Claude, 2026-10-13): the interface and band inputs ride along. `--c15` / `--c11`
/ `--c18` are passed to c16 unchanged, and the tiny child's geometry is set from the command line
(`--feature`, `--music-band`, `--layers`, `--intermediate`, `--native-type-count`) so the REAL row sets
(native_feature_dim 6144, music_codes band 65,536 rows at hidden 5120) can install into it. Synthetic tap
rows are written only for a unit that ships no charts.safetensors (c16 never reads them otherwise).
Defaults reproduce the 1.0.0 child.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

TOOL_NAME = "s7_dryrun"
TOOL_VERSION = "1.1.0"
HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--host128-ace", help="ACE unit dir whose charts_host128.reference.safetensors gets an overlay arm")
    parser.add_argument("--time-reference-rel-tol", type=float, default=1.0e-5)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--build-child", action="store_true")
    parser.add_argument("--charted-dtype", default="match", choices=("match", "fp32"))
    parser.add_argument("--c15", help="interface row set (companions.json dir) passed to c16 --c15")
    parser.add_argument("--c11", help="C11 row set passed to c16 --c11")
    parser.add_argument("--c18", help="music_codes band table passed to c16 --c18")
    parser.add_argument("--feature", type=int, default=48, help="tiny child native_feature_dim")
    parser.add_argument("--music-band", type=int, default=24, help="tiny child music_codes band width")
    parser.add_argument("--layers", type=int, default=8, help="tiny child layer count")
    parser.add_argument("--intermediate", type=int, default=96, help="tiny child FFN width")
    parser.add_argument("--native-type-count", type=int, default=None, help="tiny child native_type_count (default: config default)")
    args = parser.parse_args(argv)

    fixtures = _load("c16_media_fixtures")
    c16 = _load("c16_install_media")
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    child = scratch / "tiny_child"
    report: dict = {"run": args.run_name, "hidden": args.hidden, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION}}
    if args.build_child:
        extra = {} if args.native_type_count is None else {"native_type_count": int(args.native_type_count)}
        report["child"] = fixtures.write_child(child, seed=0, hidden=args.hidden, feature=int(args.feature), music_band=int(args.music_band),
                                               residual_mode="none", intermediate=int(args.intermediate), num_layers=int(args.layers), **extra)
    blocks = []
    tap = scratch / "tap"
    for unit in args.units:
        unit_dir = Path(unit)
        body = json.loads((unit_dir / "charted_block.json").read_text(encoding="utf-8"))
        width = int(body["spec"]["width"])
        rows_file = tap / unit_dir.name / "paired_rows.safetensors"
        if not (unit_dir / "charts.safetensors").is_file() and not rows_file.is_file():
            fixtures.write_tap_rows(tap, unit_dir.name, hidden=args.hidden, width=width, rows=args.rows, seed=len(blocks) + 11)
        blocks.append(str(unit_dir))
    if args.host128_ace:
        src = Path(args.host128_ace)
        overlay = scratch / "overlay" / (src.name + "_host128charts")
        overlay.mkdir(parents=True, exist_ok=True)
        for name in ("charted_block.json", "charted_block.safetensors", "time_reference.safetensors"):
            link = overlay / name
            if not link.exists():
                link.symlink_to((src / name).resolve())
        charts = overlay / "charts.safetensors"
        if not charts.exists():
            charts.symlink_to((src / "charts_host128.reference.safetensors").resolve())
        blocks.append(str(overlay))
    out = scratch / f"out_{args.run_name}"
    cli = [
        "--child", str(child), "--out", str(out), "--blocks", *blocks, "--tap-rows", str(tap),
        "--pair-by", "depth-fraction", "--arm", "insert",
        "--chimera-src", os.environ["CHIMERA_OMNI_SRC"],
        "--mediafold-root", os.environ.get("MEDIAFOLD_ROOT", "/home/cereal/omni-campaign/media"),
        "--time-reference-rel-tol", repr(args.time_reference_rel_tol),
        "--checkpoint-id", f"DRYRUN-S7-{args.run_name}", "--parent-checkpoint-id", "DRYRUN-TINY-CHILD",
        "--min-free-gib", "60", "--overwrite", "--charted-dtype", args.charted_dtype,
    ]
    for flag, value in (("--c15", args.c15), ("--c11", args.c11), ("--c18", args.c18)):
        if value:
            cli += [flag, str(value)]
    report["c16_argv"] = cli
    code = c16.main(cli)
    report["c16_exit"] = code
    manifest = out / "composite_manifest.json"
    if manifest.is_file():
        report["manifest"] = str(manifest)
    print(json.dumps(report, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
