"""Omni Lab model adapters (GPU worker side: torch, never psycopg2).

Each adapter is a thin wrapper over a loader the campaign already uses; it returns a Session whose
`generate(case)` gives one record in oc_calib.generate_greedy's shape (plus the generated ids), and
whose `report()` says what actually ran (placement, kernels, load facts).

    engine            module           loader reused
    hf                adapters.hf      oc_run_frozen_bank.load_model (stock transformers, one device)
    qwen35-student    adapters.qwen35  student_test.load_student (+ its device_plan, generalised to 1-N cards)
    chimera           adapters.chimera run_omni_frozen_review.py as a subprocess (dossier -> records)
    gguf              adapters.gguf    toolcall_selfgen.Renderer + Client against a running llama-server
    echo              adapters.echo    MECHANICS BED: no model; returns the last user turn (tests only)
"""
from __future__ import annotations

import importlib

ENGINES = ("hf", "qwen35-student", "chimera", "gguf", "echo")
_MODULES = {"hf": "hf", "qwen35-student": "qwen35", "chimera": "chimera", "gguf": "gguf", "echo": "echo"}


def load(engine: str):
    if engine not in _MODULES:
        raise ValueError(f"unknown engine {engine!r}; one of {ENGINES}")
    return importlib.import_module(f"{__name__}.{_MODULES[engine]}")
