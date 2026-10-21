"""graders.registry -- machine graders the worker applies after generation.  Every one wraps a grader that
already exists; nothing here decides correctness by a new rule.

    name (= grader.name in the DB)   wraps                                              applies when
    toolcall_selfgen.grade_case      distill/tools/toolcall_selfgen.grade_case          case.grader names it (toolcall_v1)
    student_test.code_check          student_test.extract_code + its assert harness,    case.reference.asserts exists
                                     generalised from second_largest to any asserts
    chimera_recovery.evaluate        vendored chimera_recovery/evaluation.grade_semantics case.reference.oracle exists
                                     (sha 6a84626d..., Codex's oracle grader)

Bank v1 and the other reader-graded suites get NO machine grade: a reader grades them artifact-tolerant
afterwards (`omni_lab.py grade`).  Every result also carries oc_calib.classify_form's box and the loop flag,
which are artifact flags, never grades.  A grader that cannot run (tool file missing) records why in the
result's raw.grader_errors and writes no grade: the cell stays 'ungraded', never a guessed pass/fail.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adapters import common as A  # noqa: E402

FUNCTION_RE = re.compile(r"^\s*(?:not\s+)?([A-Za-z_]\w*)\s*\(")


def applicable(case: dict) -> list[str]:
    names = []
    if case.get("grader") in GRADERS:
        names.append(case["grader"])
    ref = case.get("reference") if isinstance(case.get("reference"), dict) else {}
    if ref.get("asserts") and "student_test.code_check" not in names:
        names.append("student_test.code_check")
    if ref.get("oracle") and "chimera_recovery.evaluate" not in names:
        names.append("chimera_recovery.evaluate")
    return names


def grade_toolcall(case: dict, response: str, spec: dict, work_dir: Path) -> dict | None:
    sg = A.load_tool("toolcall_selfgen", A.tool_path(spec, "toolcall_selfgen"))
    ref = case.get("reference") or {}
    if not isinstance(ref.get("expected"), dict) or not case.get("tools"):
        return None
    g = sg.grade_case({"id": case["case_id"], "lane": case.get("lane"), "tools": case["tools"],
                       "expected": ref["expected"]}, response)
    return {"grade": "pass" if g["passed"] else "fail", "score": 1.0 if g["passed"] else 0.0,
            "why": "; ".join(g.get("problems") or []) or None, "detail": {"calls": g.get("calls")}}


def grade_code_asserts(case: dict, response: str, spec: dict, work_dir: Path) -> dict | None:
    """student_test.run_code_check's harness with the case's own asserts: the first block defining the asserted
    function (student_test.extract_code), each assert run in an isolated interpreter (-I, 30 s)."""
    st = A.load_tool("student_test", A.tool_path(spec, "student_test"))
    asserts = [a for a in ((case.get("reference") or {}).get("asserts") or []) if isinstance(a, str) and a.strip()]
    if not asserts:
        return None
    m = FUNCTION_RE.match(asserts[0])
    if not m:
        return {"grade": "fail", "score": 0.0, "why": f"cannot read the function name from {asserts[0]!r}", "detail": {}}
    function = m.group(1)
    code = st.extract_code(response, function)
    if code is None:
        return {"grade": "fail", "score": 0.0, "why": f"no code defining {function} in the response",
                "detail": {"code_found": False, "of": len(asserts)}}
    harness = [code, "", "import json as _j", "_r = []"]
    for expr in asserts:
        harness += ["try:", f"    assert {expr}", f"    _r.append([{expr!r}, 'pass', None])",
                    "except Exception as _e:", f"    _r.append([{expr!r}, 'fail', type(_e).__name__ + ': ' + str(_e)])"]
    harness += ["print('ASSERTS ' + _j.dumps(_r))"]
    out_dir = work_dir / "code_check" / re.sub(r"[^A-Za-z0-9_.-]", "_", case["case_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "check.py"
    path.write_text("\n".join(harness) + "\n", encoding="utf-8")
    try:
        done = subprocess.run([sys.executable, "-I", str(path)], cwd=str(out_dir), capture_output=True, text=True,
                              timeout=30)
    except subprocess.TimeoutExpired:
        return {"grade": "fail", "score": 0.0, "why": "assert harness timed out (30 s)", "detail": {"harness": str(path)}}
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("ASSERTS ")), None)
    results = json.loads(line[len("ASSERTS "):]) if line else []
    passed = sum(1 for r in results if r[1] == "pass")
    frac = passed / len(asserts)
    return {"grade": "pass" if passed == len(asserts) else "fail", "score": frac,
            "why": f"{passed}/{len(asserts)} asserts pass" + ("" if line else f"; harness exit {done.returncode}"),
            "detail": {"asserts": results, "harness": str(path), "stderr_tail": done.stderr[-800:]}}


def grade_recovery(case: dict, response: str, spec: dict, work_dir: Path) -> dict | None:
    vendor = Path(A.tool_path(spec, "recovery_vendor"))
    if str(vendor.parent) not in sys.path:
        sys.path.insert(0, str(vendor.parent))
    import importlib
    ev = importlib.import_module(f"{vendor.name}.evaluation")
    A.TOOL_SHAS.setdefault("chimera_recovery.evaluation", {"path": ev.__file__, "sha256": A.file_sha256(ev.__file__)})
    ref = case.get("reference") or {}
    messages = A.messages_of(case)
    task = {"id": case["case_id"], "oracle": ref.get("oracle"), "input": {"messages": messages, "tools": case.get("tools") or []}}
    g = ev.grade_semantics(task, {"messages": messages + [{"role": "assistant", "content": response}]})
    status = g.get("status")
    if status not in ("pass", "fail"):
        return {"grade": None, "why": g.get("reason") or "ungraded by the oracle grader", "detail": g}
    return {"grade": status, "score": 1.0 if status == "pass" else 0.0,
            "why": g.get("reason") or ("; ".join(g.get("errors") or []) or None), "detail": g}


GRADERS: dict[str, Callable[[dict, str, dict, Path], Any]] = {
    "toolcall_selfgen.grade_case": grade_toolcall,
    "student_test.code_check": grade_code_asserts,
    "chimera_recovery.evaluate": grade_recovery,
}


def grade_all(case: dict, response: str, spec: dict, work_dir: Path) -> tuple[list[dict], list[str]]:
    """([{grader, grade, score, why, detail}], [errors]) -- a grade of None (oracle says ungraded) is kept
    with grade None so the reason is recorded, and is never written as a grade row."""
    grades, errors = [], []
    for name in applicable(case):
        try:
            g = GRADERS[name](case, response, spec, work_dir)
        except Exception as exc:  # noqa: BLE001 -- a grader that cannot run leaves the cell ungraded, with the reason
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if g is not None:
            grades.append({"grader": name, **g})
    return grades, errors
