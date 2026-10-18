#!/usr/bin/env python3
"""Print the v2 linear-attention kernel report for one checkpoint, and stop.

WHY THIS IS A SEPARATE COMMAND.  The kernel path is a validity element -- it sits
in `lane_digest_validity` beside the dtype -- and the first version of it was
wrong for a whole pass of this lane because it was *inferred* from the package
inventory instead of read off the model.  A field like that needs to be
answerable in thirty seconds, on a CPU, without a card and without a review, or
nobody checks it and the next wrong inference lives just as long.

It loads the checkpoint on the CPU by default, so it can run while a card is
busy, and it writes nothing unless asked.  It grades nothing and refuses nothing.

    $PY oc_kernel_probe.py --model-dir /home/cereal/models/Qwen3.5-4B \\
        --out out/kernel-probe-v2-Q35-4B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oc_calib import kernel_path_report, log, resolve_dtype  # noqa: E402


def run(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  -- imported for the side effect of a clear error
    import transformers

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    declared = str(config["architectures"][0])
    loader = getattr(transformers, declared, None)
    if loader is None:
        raise SystemExit(
            f"stock transformers {transformers.__version__} has no class "
            f"{declared!r}; this probe never uses trust_remote_code."
        )
    log(f"loading {model_dir} as {declared} on {args.device} in {args.dtype}")
    model = loader.from_pretrained(
        str(model_dir), dtype=resolve_dtype(str(args.dtype)), trust_remote_code=False
    )
    if str(args.device) != "cpu":
        model = model.to(str(args.device))
    model.eval()

    report = kernel_path_report(model)
    report["probe_context"] = {
        "model_directory": str(model_dir),
        "loaded_class": type(model).__name__,
        "transformers_version": transformers.__version__,
        "device": str(args.device),
        "dtype": str(args.dtype),
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
        )
        log(f"wrote {args.out}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path)
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
