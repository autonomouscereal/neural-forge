#!/usr/bin/env python3
"""omni_lab_worker -- the GPU side of `omni_lab.py run` (torch, transformers, fla; never psycopg2).

    <gpu-venv>/bin/python omni_lab_worker.py --spec <run-dir>/spec.json --run-dir <run-dir>

Reads the spec the control CLI wrote (model, engine, cases from the DB, generation settings, card list,
tool paths), loads the model through one adapter (adapters/*), generates every case, applies the machine
graders (graders/registry.py) and writes, in the run directory:

    results.jsonl   one line per case, appended as each case finishes (a crash keeps what was done)
    receipt.json    what actually ran: runtime versions, cards (with UUIDs when torch exposes them),
                    placement, kernels, tokenizer, thinking, stops, tool sha256s, timing, status

and prints one console line per case.  The control CLI ingests the two files into the database; the same
files re-ingest later with `omni_lab.py ingest lab_runs` if the database was down.  Exit 0 = every case
has a result row; 2 = load refused/failed; 1 = some case raised.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapters import common as A  # noqa: E402

TOOL = "omni_lab_worker"
VERSION = "1.0.0"
RECEIPT_SCHEMA = "omni-lab-worker-receipt-v1"
RESULT_SCHEMA = "omni-lab-worker-result-v1"


def write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def code_shas() -> dict:
    files = [HERE / "omni_lab_worker.py", HERE / "graders" / "registry.py"] + sorted((HERE / "adapters").glob("*.py"))
    return {str(p.relative_to(HERE)).replace("\\", "/"): A.file_sha256(p) for p in files}


def form_fields(record: dict) -> tuple[str | None, dict | None]:
    form = record.get("form")
    if isinstance(form, dict):
        return form.get("form_box"), form
    return None, None


def console_line(i: int, n: int, rec: dict, grades: list[dict]) -> str:
    g = ",".join(f"{x['grader'].split('.')[0]}={x['grade']}" for x in grades if x.get("grade")) or "-"
    box, _ = form_fields(rec)
    tps = rec.get("decode_tok_per_s")
    head = (rec.get("response_verbatim") or "").replace("\n", " ")[:90]
    return (f"[{i}/{n}] {rec['id']}: {rec.get('generated_tokens')} tok {rec.get('termination')} form={box or '-'} "
            f"grade={g} {rec.get('wall_time_seconds')} s" + (f" {tps} tok/s" if tps else "") + f" | {head!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    run_dir = args.run_dir
    results_path = run_dir / "results.jsonl"
    receipt_path = run_dir / "receipt.json"
    if results_path.exists():
        print(f"REFUSED: {results_path} exists (a run directory is written once)", flush=True)
        return 2
    started = A.utc()
    t0 = time.monotonic()
    receipt = {"schema": RECEIPT_SCHEMA, "tool": TOOL, "version": VERSION, "code_sha256": code_shas(),
               "run_key": spec["run_key"], "engine": spec["engine"], "model_path": spec.get("model_path"),
               "started_utc": started, "status": "loading", "argv": sys.argv, "pid": os.getpid(),
               "env_vars": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                                                          "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "TRITON_CACHE_DIR")}}
    write_json_atomic(receipt_path, receipt)
    A.log(f"{TOOL} {VERSION}: {spec['engine']} {spec.get('model_path')} -- {len(spec['cases'])} cases, "
          f"device {spec['device']['kind']} {spec['device'].get('cards') or ''}")

    torch = None
    if spec["engine"] != "echo":
        import torch  # noqa: F811 -- the GPU venv; the echo bed runs without it
    from adapters import load as load_adapter
    from graders import registry as G
    try:
        session = load_adapter(spec["engine"]).Session(spec)
    except Exception as exc:  # noqa: BLE001 -- recorded in the receipt and on the console
        receipt.update(status="load-failed", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc()[-4000:], ended_utc=A.utc(),
                       wall_seconds=round(time.monotonic() - t0, 3), tool_sha256=A.TOOL_SHAS)
        if torch is not None:
            receipt["runtime"] = A.torch_env(torch)
        write_json_atomic(receipt_path, receipt)
        A.log(f"LOAD FAILED: {receipt['error']}")
        return 2
    load_s = round(time.monotonic() - t0, 3)
    A.log(f"loaded in {load_s} s")
    receipt.update(status="running", load_seconds=load_s)
    if torch is not None:
        receipt["runtime"] = A.torch_env(torch)
    write_json_atomic(receipt_path, receipt)

    n = len(spec["cases"])
    by_id = {c["case_id"]: c for c in spec["cases"]}
    done = {"n": 0, "errors": 0}

    def emit(rec: dict) -> None:
        case = by_id[rec["id"]]
        grades, gerrors = ([], []) if rec.get("termination") in ("unsupported-modality", "other", "error") and \
            not rec.get("response_verbatim") else G.grade_all(case, rec.get("response_verbatim") or "", spec, run_dir)
        box, form = form_fields(rec)
        row = {"schema": RESULT_SCHEMA, "case_pk": case["case_pk"], "case_id": case["case_id"],
               "content_sha256": case["content_sha256"], "termination": rec.get("termination"),
               "runtime_error": rec.get("runtime_error"), "output_text": rec.get("response_verbatim"),
               "output_tokens": rec.get("generated_tokens"), "prompt_tokens": rec.get("prompt_tokens"),
               "loop_detected": rec.get("loop_detected") if rec.get("loop_detected") is not None
               else (form or {}).get("loop_detected"),
               "form_box": box, "form": form, "wall_time_seconds": rec.get("wall_time_seconds"),
               "decode_tok_per_s": rec.get("decode_tok_per_s"), "prefill_seconds": rec.get("prefill_seconds"),
               "generated_token_ids": rec.get("generated_token_ids"),
               "generated_ids_match_runner_sha": rec.get("generated_ids_match_runner_sha"),
               "response_sha256": rec.get("response_sha256"), "unsupported_reason": rec.get("unsupported_reason"),
               "server": rec.get("server"), "grades": grades, "grader_errors": gerrors}
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        done["n"] += 1
        print(console_line(done["n"], n, rec, grades), flush=True)
        for e in gerrors:
            A.log(f"    grader error: {e}")

    gen = spec["gen"]
    if hasattr(session, "run_all"):
        try:
            session.run_all(spec["cases"], gen, run_dir, emit)
        except Exception as exc:  # noqa: BLE001
            done["errors"] += 1
            A.log(f"run_all raised: {type(exc).__name__}: {exc}")
            receipt["error"] = f"{type(exc).__name__}: {exc}"
    else:
        for case in spec["cases"]:
            try:
                rec = session.generate(case, gen)
            except Exception as exc:  # noqa: BLE001 -- one case failing must not lose the others
                done["errors"] += 1
                rec = A.unsupported_record(case, "adapter raised", "error")
                rec["runtime_error"] = f"{type(exc).__name__}: {exc}"
                A.log(f"{case['case_id']}: {rec['runtime_error']}")
            emit(rec)
    try:
        report = session.report()
    except Exception as exc:  # noqa: BLE001
        report = {"report_error": f"{type(exc).__name__}: {exc}"}
    receipt.update(report=report, tool_sha256=A.TOOL_SHAS, n_cases=n, n_results=done["n"], case_errors=done["errors"],
                   ended_utc=A.utc(), wall_seconds=round(time.monotonic() - t0, 3),
                   status="pass" if done["n"] == n and not done["errors"] else "fail")
    if torch is not None:
        receipt["peak_allocated_mib"] = A.peak_mib(torch)
    write_json_atomic(receipt_path, receipt)
    A.log(f"DONE: {done['n']}/{n} results, {done['errors']} case errors, {receipt['wall_seconds']} s -> {run_dir}")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
