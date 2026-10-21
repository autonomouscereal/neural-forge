"""adapters.chimera -- a chimera_omni.checkpoint.v1 directory (OMNI1, S5, D1-composed) through the native
runner run_omni_frozen_review.py, unchanged, as a subprocess.

The worker writes the run's chat cases as a bank file in the run directory (bank v1 layout: id, category,
messages), passes its sha256 as --bank-sha256, and reads the dossier's child_responses back as records.
What the native runner cannot take is recorded, not skipped: raw continuations (its bank needs chat
messages), tool cases (its renderer has no tools= path) and non-text inputs (--vision-cards is not wired
here) come back as termination 'unsupported-modality' / 'other' with the reason.
Placement: --split from the spec (GiB per visible card, the runner's own flag); the dossier's placement
and kernel report go into the receipt as written.  UNVERIFIED end to end on 2026-10-21 (no 27B run was
allowed); the command line is unit-tested.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import common as A


def build_command(spec: dict, run_dir: Path, bank_path: Path, bank_sha: str, lanes: list[str]) -> list[str]:
    gen = spec["gen"]
    p = spec["paths"]
    cmd = [sys.executable, p["chimera_runner"], "--model", spec["model_path"], "--label", spec["label"],
           "--stage", spec.get("stage_arg") or "distill", "--bank", str(bank_path), "--bank-sha256", bank_sha,
           "--lanes", ",".join(lanes), "--probe-prompts", p["probe_prompts"], "--skip-smoke",
           "--thinking", gen["thinking"], "--out", str(run_dir / "dossier.json"), "--device", "cuda:0",
           "--dtype", spec["dtype"], "--maximum-new-tokens", str(int(gen["cap"])),
           "--extra-eos-tokens", ",".join(gen["extra_eos_tokens"]), "--calibration-dir", p["calibration_dir"],
           "--text-only"]
    if spec.get("tokenizer_dir"):
        cmd += ["--tokenizer-dir", spec["tokenizer_dir"]]
    if p.get("chimera_src"):
        cmd += ["--chimera-src", p["chimera_src"]]
    if spec["device"].get("split"):
        cmd += ["--split", spec["device"]["split"]]
    if gen.get("ledger_band_mask", True):
        cmd += ["--ledger-band-logit-mask"]
    return cmd


class Session:
    engine = "chimera"

    def __init__(self, spec: dict):
        if spec["device"]["kind"] != "cuda":
            raise RuntimeError("engine chimera runs on CUDA cards")
        self.spec = spec
        self.dossier: dict = {}
        self.command: list[str] = []
        self.returncode = None
        A.TOOL_SHAS["run_omni_frozen_review"] = {"path": spec["paths"]["chimera_runner"],
                                                 "sha256": A.file_sha256(spec["paths"]["chimera_runner"])}

    def run_all(self, cases: list[dict], gen: dict, run_dir: Path, emit) -> None:
        runnable, held = [], []
        for case in cases:
            missing = A.non_text_inputs(case)
            if missing:
                held.append(A.unsupported_record(case, f"native runner wired text-only here; case needs {missing}"))
            elif A.is_raw_text(case):
                held.append(A.unsupported_record(case, "native runner banks are chat messages; raw continuation", "other"))
            elif case.get("tools"):
                held.append(A.unsupported_record(case, "native runner renders no tools=", "other"))
            else:
                runnable.append(case)
        for rec in held:
            emit(rec)
        if not runnable:
            return
        bank = {"schema": "omni-lab-generated-bank-v1", "prompts": [
            {"id": c["case_id"], "category": c.get("category") or "uncategorised", "messages": A.messages_of(c)}
            for c in runnable]}
        bank_path = run_dir / "bank.json"
        bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        self.command = build_command(self.spec, run_dir, bank_path, A.file_sha256(bank_path),
                                     [c["case_id"] for c in runnable])
        A.log("chimera: " + " ".join(self.command))
        env = dict(os.environ)
        if self.spec["paths"].get("chimera_src"):
            env["CHIMERA_OMNI_SRC"] = self.spec["paths"]["chimera_src"]
        done = subprocess.run(self.command, env=env)
        self.returncode = done.returncode
        out = run_dir / "dossier.json"
        if out.is_file():
            self.dossier = json.loads(out.read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in self.dossier.get("child_responses") or []}
        for case in runnable:
            rec = by_id.get(case["case_id"])
            if rec is None:
                rec = A.unsupported_record(case, f"native runner exit {done.returncode}; no response for this case",
                                           "error")
                rec["runtime_error"] = f"run_omni_frozen_review exit {done.returncode}"
            emit(dict(rec))

    def report(self) -> dict:
        d = self.dossier
        return {"load": {k: d.get(k) for k in ("runner", "model", "checkpoint", "is_random_init") if k in d},
                "kernels": d.get("kernels") or d.get("linear_attention_kernels"),
                "kernel_path": (d.get("kernels") or {}).get("kernel_path") if isinstance(d.get("kernels"), dict) else None,
                "placement": {k: d.get(k) for k in ("placement", "devices", "split", "block_device") if k in d},
                "thinking": d.get("thinking"), "command": self.command, "runner_exit": self.returncode,
                "dossier": "dossier.json" if d else None}
