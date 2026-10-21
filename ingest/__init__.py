"""Omni Lab ingesters.  `omni_lab.py ingest <source>|models-runs` runs them; see ingest/common.py.

ORDER is the dependency order of the models/runs group: hardware and pins and lanes first (runs point
at gpus, tool versions and lanes), the ledger before the scan (ledger rows keep their ids), caches
before the status files (runs link to the cache versions they name).
"""
from __future__ import annotations

import importlib

MODELS_RUNS_ORDER = ("hardware", "tool_pins", "lanes", "oc_ledger", "model_scan", "teacher_caches", "trainer_runs",
                     "status_runs", "fold_receipts")
# evals: banks first; dossiers and toolcall_bankruns before student_tests (its reader grades for the parent
# and OMNI1 columns attach to those runs).  codex_tasksets reads the WORKSTATION's proof_outputs: run it there.
EVALS_ORDER = ("banks", "dossiers", "toolcall_bankruns", "student_tests", "s8_media", "p11_reads")
GROUPS = {"models-runs": MODELS_RUNS_ORDER, "evals": EVALS_ORDER}


def load(name: str):
    """The ingester module ingest.<name> (must define SOURCE and run)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"bad ingester name {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(module, "SOURCE") or not hasattr(module, "run"):
        raise ValueError(f"ingest/{name}.py is not an ingester (needs SOURCE and run)")
    return module


def resolve(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for m in GROUPS.get(n, (n,)):
            if m not in out:
                out.append(m)
    return out
