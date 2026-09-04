#!/usr/bin/env python3
"""The parent-calibration entry point: decide what has landed, then run it.

WHAT IT DECIDES, AND WHY IT DECIDES IT RATHER THAN BEING TOLD.  The campaign's
parents are being downloaded while the instruments are being written, so "run the
frozen bank on Qwen3.5-4B" is conditional on a race.  This file resolves that
race by looking at the disk -- not at the download log's last line, which says
`[done]` the moment the loop exits and says nothing about whether the shards are
whole.  `--download-log` is read and REPORTED because it is useful evidence, and
it decides nothing.

The landing test is three facts about the directory:

  1. `config.json` parses and names an architecture stock transformers has,
  2. every file `model.safetensors.index.json` maps a tensor to exists, and
  3. each of those files is WHOLE by its own safetensors header -- the header's
     `data_offsets` say how many bytes the file must have, and the file has at
     least that many.  A flat byte floor was the old test and it was wrong: a
     5.3 GB shard truncated to 2 KiB passed it.

A directory that fails any of them is NOT landed, and the runner falls back to the
random-init mechanics bed rather than half-loading a truncated shard and reporting
whatever comes out.  That fallback is labelled everywhere it appears: a
`is_random_init` dossier cannot be turned into a scoring sheet without an explicit
override, because untrained text is not evidence of anything.

USAGE (server)

  $PY oc_parent_calibration.py \
      --model-dir /home/cereal/models/Qwen3.5-4B --label Q35-4B \
      --download-log /home/cereal/hf-downloads/dl_qwen35_4b_now.log \
      --work /home/cereal/omni-campaign/calibration/out \
      --device cuda:0 --require-compute-architecture sm86 --dtype bfloat16

  add --bank-v2 <bank json> to run the draft bank's TEXT lanes as a second pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oc_calib import CalibrationError, file_sha256, log  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

#: A safetensors file is an 8-byte little-endian header length, that many bytes
#: of JSON, then the tensor data.  Eight bytes is the floor below which there is
#: not even a length to read.  IT IS NOT THE COMPLETENESS TEST -- see
#: `shard_completeness`, which reads the file's own header and computes the size
#: the file must have.  An earlier version compared against a flat 1,024 bytes,
#: which called a 5.3 GB shard truncated to 2 KiB "landed" and pushed the failure
#: down into the loader, where it reads as a code fault rather than a bad
#: download.
MINIMUM_SAFETENSORS_BYTES = 8

#: The largest safetensors JSON header this file will read before refusing.  A
#: real header for a 100-shard checkpoint is a few hundred kilobytes; a header
#: length of 10^12 is a corrupt or hostile file, not a big model.
MAXIMUM_HEADER_BYTES = 256 * 1024 * 1024


def shard_completeness(path: Path) -> dict[str, Any]:
    """Whether one safetensors file is WHOLE, decided from its own header.

    The header's `data_offsets` name the last byte any tensor in the file needs,
    so the minimum whole size of a safetensors file is computable from the first
    few hundred kilobytes of the file itself -- no index, no manifest, no
    download log, nothing to trust.  This reads the header only; it never loads a
    tensor, so it costs the same on a 2 KiB stub and on a 5 GB shard.

    Returns the numbers rather than a verdict alone, because "required
    5,329,398,688, observed 2,048" is the sentence that tells a reader the
    download died, and `ok: false` on its own is not.
    """

    result: dict[str, Any] = {
        "path": path.name,
        "observed_bytes": None,
        "required_bytes": None,
        "header_bytes": None,
        "tensors": None,
        "ok": False,
        "why": None,
    }
    if not path.is_file():
        result["why"] = "missing"
        return result
    size = path.stat().st_size
    result["observed_bytes"] = size
    if size < MINIMUM_SAFETENSORS_BYTES:
        result["why"] = "shorter than the 8-byte header length itself"
        return result
    try:
        with path.open("rb") as handle:
            header_length = int.from_bytes(handle.read(8), "little", signed=False)
            if header_length <= 0 or header_length > MAXIMUM_HEADER_BYTES:
                result["why"] = (
                    f"declared header length {header_length} is not a safetensors "
                    "header"
                )
                return result
            result["header_bytes"] = header_length
            if size < 8 + header_length:
                result["required_bytes"] = 8 + header_length
                result["why"] = "the header itself is truncated"
                return result
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["why"] = f"header unreadable: {type(exc).__name__}: {exc}"
        return result
    if not isinstance(header, dict):
        result["why"] = "header is not a JSON object"
        return result
    last_byte = 0
    tensors = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            result["why"] = f"header entry {name!r} is not an object"
            return result
        offsets = entry.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            result["why"] = f"header entry {name!r} has no usable data_offsets"
            return result
        tensors += 1
        last_byte = max(last_byte, int(offsets[1]))
    result["tensors"] = tensors
    result["required_bytes"] = 8 + header_length + last_byte
    result["ok"] = size >= result["required_bytes"]
    if not result["ok"]:
        result["why"] = (
            f"truncated: the header needs {result['required_bytes']} bytes and "
            f"the file is {size}"
        )
    return result


def landing_report(model_dir: Path) -> dict[str, Any]:
    """Three facts about the directory.  No opinion, no log parsing."""

    report: dict[str, Any] = {
        "model_directory": str(model_dir),
        "exists": model_dir.is_dir(),
        "config_ok": False,
        "architecture": None,
        "index_ok": False,
        "shards_expected": 0,
        "shards_present": 0,
        "shards_undersized": [],
        "shard_completeness": [],
        "landed": False,
        "why_not": [],
    }
    if not report["exists"]:
        report["why_not"].append("the directory does not exist")
        return report
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        report["why_not"].append("no config.json")
        return report
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["why_not"].append(f"config.json does not parse: {exc}")
        return report
    architectures = config.get("architectures") or []
    report["architecture"] = architectures[0] if architectures else None
    report["config_ok"] = bool(architectures)
    if not report["config_ok"]:
        report["why_not"].append("config.json declares no architectures")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
        report["shards_expected"] = len(files)
        present = 0
        for name in files:
            path = model_dir / name
            if not path.is_file():
                report["why_not"].append(f"index names {name}, which is missing")
                continue
            present += 1
            whole = shard_completeness(path)
            report["shard_completeness"].append(whole)
            if not whole["ok"]:
                report["shards_undersized"].append(name)
        report["shards_present"] = present
        report["index_ok"] = present == len(files) and not report["shards_undersized"]
        for whole in report["shard_completeness"]:
            if not whole["ok"]:
                report["why_not"].append(
                    f"{whole['path']}: {whole['why']} "
                    f"(observed {whole['observed_bytes']}, "
                    f"required {whole['required_bytes']})"
                )
    else:
        single = model_dir / "model.safetensors"
        report["shards_expected"] = 1
        report["shards_present"] = 1 if single.is_file() else 0
        if single.is_file():
            whole = shard_completeness(single)
            report["shard_completeness"].append(whole)
            report["index_ok"] = bool(whole["ok"])
            if not whole["ok"]:
                report["why_not"].append(
                    f"{whole['path']}: {whole['why']} "
                    f"(observed {whole['observed_bytes']}, "
                    f"required {whole['required_bytes']})"
                )
        else:
            report["index_ok"] = False
        if not report["index_ok"]:
            report["why_not"].append("no shard index and no model.safetensors")
    report["landed"] = bool(report["config_ok"] and report["index_ok"])
    return report


def read_download_log(path: Path | None) -> dict[str, Any]:
    """Evidence, not a decision."""

    if path is None or not Path(path).is_file():
        return {"path": str(path) if path else None, "present": False}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "path": str(path),
        "present": True,
        "bytes": len(text),
        "last_lines": lines[-6:],
        "decides_nothing": (
            "a download log says [done] when its loop exited; the disk says "
            "whether the shards are whole.  The disk decides."
        ),
    }


def run_step(command: list[str], *, label: str, log_dir: Path) -> dict[str, Any]:
    """Run one child, streaming its output to a log AND keeping it in memory.

    `capture_output=True` alone made a forty-minute GPU run invisible while it
    ran, which is exactly when you want to see it.  The child's stdout and stderr
    go to `<log_dir>/step-<label>.log` as they are produced, so `tail -f` works,
    and the file is read back afterwards so the failure path still prints
    everything rather than a summary of it.
    """

    log(f"$ {' '.join(command)}")
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = log_dir / f"step-{label}.log"
    started = time.monotonic()
    with open(stream, "w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.monotonic() - started, 2)
    captured = stream.read_text(encoding="utf-8", errors="replace")
    tail = captured.strip().splitlines()
    if result.returncode != 0:
        sys.stderr.write(captured)
        raise CalibrationError(
            f"{label} exited {result.returncode} after {elapsed}s.  Its full "
            f"output is above and in {stream}; it is not summarised away."
        )
    payload: dict[str, Any] = {}
    if tail:
        try:
            payload = json.loads(tail[-1])
        except json.JSONDecodeError:
            payload = {"stdout_tail": tail[-5:]}
    return {"label": label, "seconds": elapsed, "result": payload,
            "log": str(stream), "output_tail": tail[-8:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--download-log", type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--require-compute-architecture")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--thinking",
        default="template-default",
        choices=["template-default", "off"],
        help=(
            "passed through to every step so ALL of them share one surface.  "
            "On Qwen3.5 the template default opens a think block and the token "
            "budget goes to chain-of-thought: the first Q35-4B pass ended 13 of "
            "16 lanes at the cap with no answer written.  `off` produces "
            "answers.  Which surface the omni protocol adopts is a proposed "
            "decision line for Cyril; this only makes the choice sayable."
        ),
    )
    ap.add_argument(
        "--bank-v1",
        type=Path,
        default=REPO / "research" / "lfm2_semantic_whole_response_prompts_v1.json",
        help=(
            "the frozen 15-case pack.  The repo path is the default; on the "
            "server, where there is no checkout, pass the uploaded copy -- its "
            "digest is checked against 0ee6bf14... either way."
        ),
    )
    ap.add_argument(
        "--probe-prompts",
        type=Path,
        default=REPO / "experiments" / "matrix_campaign_probe_prompts_v1.json",
    )
    ap.add_argument(
        "--smoke-prompts",
        type=Path,
        default=REPO / "experiments" / "matrix_campaign_smoke_prompts_v1.json",
    )
    ap.add_argument("--bank-v2", type=Path, help="also run the draft bank's TEXT lanes")
    ap.add_argument("--bank-v2-lanes", help="comma-separated subset of bank-v2 case ids")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--force-tiny",
        action="store_true",
        help="use the random-init mechanics bed even if the parent has landed",
    )
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    landing = landing_report(Path(args.model_dir))
    download = read_download_log(args.download_log)
    tiny = args.force_tiny or not landing["landed"]
    log(
        f"landing: landed={landing['landed']} "
        f"shards {landing['shards_present']}/{landing['shards_expected']} "
        f"arch={landing['architecture']}"
    )
    if tiny and not args.force_tiny:
        log(
            "PARENT HAS NOT LANDED -- falling back to the RANDOM-INIT mechanics "
            f"bed.  Reasons: {landing['why_not']}"
        )

    label = args.label if not tiny else f"{args.label}-TINYRANDOM"
    dossier = work / f"oc-dossier-{label}.json"
    common = [
        args.python,
        str(HERE / "oc_run_frozen_bank.py"),
        "--model-dir", str(args.model_dir),
        "--label", label,
        "--parent-code", args.label,
        "--stage", "parent",
        "--device", str(args.device),
        "--dtype", str(args.dtype),
        "--maximum-new-tokens", str(args.maximum_new_tokens),
        #: One surface for every step of a calibration.  A bank-v1 pass and a
        #: bank-v2 pass taken on different thinking settings are not two
        #: measurements of one parent, and this is where that would slip in.
        "--thinking", str(args.thinking),
        "--probe-prompts", str(args.probe_prompts),
        "--smoke-prompts", str(args.smoke_prompts),
    ]
    if args.require_compute_architecture:
        common += ["--require-compute-architecture", args.require_compute_architecture]
    if tiny:
        common += ["--tiny-random"]

    steps = [
        run_step(
            common + ["--bank", str(args.bank_v1), "--out", str(dossier)],
            label="bank-v1",
            log_dir=work,
        )
    ]

    assessment = work / f"assessment-{label}.json"
    sheet = work / f"SCORING-SHEET-{label}.md"
    assess_command = [
        args.python,
        str(HERE / "oc_assess.py"),
        "--dossier", str(dossier),
        "--bank", str(args.bank_v1),
        "--out-json", str(assessment),
        "--out-markdown", str(sheet),
    ]
    if tiny:
        assess_command += ["--allow-random-init"]
    steps.append(run_step(assess_command, label="assess-v1", log_dir=work))

    if args.bank_v2:
        dossier2 = work / f"oc-dossier-b2-{label}.json"
        command = common + [
            "--bank", str(args.bank_v2),
            "--text-only",
            "--out", str(dossier2),
            "--skip-smoke",
        ]
        if args.bank_v2_lanes:
            command += ["--lanes", args.bank_v2_lanes]
        steps.append(run_step(command, label="bank-v2-text", log_dir=work))
        assessment2 = work / f"assessment-b2-{label}.json"
        sheet2 = work / f"SCORING-SHEET-B2-{label}.md"
        command = [
            args.python,
            str(HERE / "oc_assess.py"),
            "--dossier", str(dossier2),
            "--bank", str(args.bank_v2),
            "--out-json", str(assessment2),
            "--out-markdown", str(sheet2),
        ]
        if tiny:
            command += ["--allow-random-init"]
        steps.append(run_step(command, label="assess-v2", log_dir=work))

    body = json.loads(dossier.read_text(encoding="utf-8"))
    first_three = [
        {
            "id": record["id"],
            "termination": record["termination"],
            "generated_tokens": record["generated_tokens"],
            "response_verbatim": record["response_verbatim"],
        }
        for record in body["child_responses"][:3]
    ]
    summary = {
        "schema": "neural-forge-omni-parent-calibration-run-v1",
        "label": label,
        "used_random_init_bed": tiny,
        "landing": landing,
        "download_log": download,
        "dossier": str(dossier),
        "dossier_file_sha256": file_sha256(dossier),
        "dossier_sha256": body["dossier_sha256"],
        "lane_digest": body["lane_digest"],
        "lane_digest_validity": body["lane_digest_validity"],
        "emit_zero_floor_passed": body["emit_zero_floor"]["passed"],
        "decode_path_agree": body["decode_path_preflight"]["agree"],
        "steps": steps,
        "first_three_cases": first_three,
    }
    out = work / f"calibration-run-{label}.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
