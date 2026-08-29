#!/usr/bin/env python3
"""Generate the ADDENDUM 16.1 EXTENSION-surface dossier for one sealed child.

This is the second review surface and it is nothing like a second judge.  Its
whole job is resolution: five of eight round-2 kills rode one-case margins
against a fifteen-point instrument, and a wave that cannot tell a real +1 from an
instrument tick cannot spend GPU rationally.  So thirty extra prompts are decoded
under the *identical* decode path and the result is reported beside the sealed
count, never instead of it.

**How the decode path is kept identical.**  Not by copying it.  This file
imports ``run_lfm2_composition_frozen_review`` -- the sealed runner, unmodified
by this wave and by this mechanism forever -- and calls its own ``_generate``,
``_load_standard_shape_child`` and ``_reconstitute``.  If the sealed runner's
greedy loop, chat-template handling, cap accounting or termination taxonomy ever
changes, this surface changes with it in the same commit, because there is only
one copy of that code.  What this file adds is a different prompt list and a
different output schema; it adds no generation behaviour at all.

**What it refuses.**  The bank must digest to ``extension-bank-seal.json``'s
recorded ``bank_sha256``, and the seal must name the same sealed pack digest the
protocol seal names.  A bank that has drifted is not the frozen bank and the
dossier would be about nothing.  For a child, the ARCH receipt's own binding
checks run unchanged (file digest, loaded class, live parameter count), because a
dossier that cannot prove which child it graded is worthless on any surface.

**What it never does.**  It writes no verdict, no count, no ledger row and no
ARCH seal.  ``seal_matrix_behavior_review.py`` never sees this file's output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DOSSIER_SCHEMA = "neural-forge-lfm2-matrix-extension-surface-dossier-v1"


def _load_runner(deployment: Path):
    for sub in ("src", "experiments"):
        path = str(deployment / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import run_lfm2_composition_frozen_review as runner  # noqa: E402

    return runner


def _authenticated_bank(bank_path: Path, seal_path: Path, which: str) -> list[dict[str, Any]]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != "neural-forge-extension-bank-seal-v1":
        raise SystemExit(f"REFUSED: {seal_path} is not an extension-bank seal")
    # WAVE 34 adds `extension2` and adds NOTHING else to this file.  The v2 bank
    # is a different prompt list under the identical decode path; a new surface
    # that reimplemented the decode would be a new instrument and not a new bank.
    # The v1 key is untouched, so every one of the sixty committed v1
    # observations still authenticates against exactly the digest it did.
    key = {
        "extension": "bank_sha256",
        "extension2": "bank2_sha256",
        "paraphrase": "paraphrase_sha256",
    }[which]
    observed = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if observed != seal[key]:
        raise SystemExit(
            f"REFUSED: {bank_path} digests {observed}; the seal pinned {seal[key]}.  "
            "A drifted bank is not the frozen bank."
        )
    doc = json.loads(bank_path.read_text(encoding="utf-8"))
    if which in ("extension", "extension2"):
        rows = doc["prompts"]
    else:
        rows = [row for lane in doc["triplets"].values() for row in lane]
        for row in rows:
            row.setdefault("category", "paraphrase")
    if not rows:
        raise SystemExit("REFUSED: empty prompt surface")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--seal", type=Path, required=True)
    ap.add_argument(
        "--which",
        choices=("extension", "extension2", "paraphrase"),
        default="extension",
    )
    ap.add_argument("--receipt", type=Path, help="the sealed ARCH cell receipt, when there is one")
    ap.add_argument("--child-model", type=Path)
    ap.add_argument("--composed-tensors", type=Path)
    ap.add_argument("--child-tokenizer", type=Path)
    ap.add_argument("--parent-model", type=Path, required=True)
    ap.add_argument("--label", required=True, help="tag written into the dossier, e.g. V19 or PARENT")
    ap.add_argument("--parent-only", action="store_true")
    ap.add_argument("--skip-parent", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--require-compute-architecture", default="sm86")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    runner = _load_runner(args.deployment)
    import torch  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()
    cell_runner = runner.cell_runner

    architecture = cell_runner.compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise SystemExit(
            f"REFUSED: device reports {architecture}, this wave grades on "
            f"{args.require_compute_architecture}"
        )

    rows = _authenticated_bank(args.bank, args.seal, args.which)
    bank_digest = hashlib.sha256(args.bank.read_bytes()).hexdigest()
    print(f"surface {args.which}: {len(rows)} prompts, bank {bank_digest[:12]}")

    dtype = torch.float32
    parent_tokenizer = AutoTokenizer.from_pretrained(str(args.parent_model))
    child_tokenizer = parent_tokenizer
    if args.child_tokenizer is not None:
        child_tokenizer = AutoTokenizer.from_pretrained(str(args.child_tokenizer))

    child_records: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {}
    if not args.parent_only:
        if args.receipt is None:
            raise SystemExit("REFUSED: a child dossier needs --receipt")
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        if args.child_model is not None:
            child = runner._load_standard_shape_child(
                receipt=receipt,
                child_model=args.child_model,
                device=str(args.device),
                dtype=dtype,
            )
        elif args.composed_tensors is not None:
            child, _ = runner._reconstitute(
                parent_model=args.parent_model,
                receipt=receipt,
                tensors_path=args.composed_tensors,
                device=str(args.device),
                dtype=dtype,
            )
        else:
            raise SystemExit("REFUSED: --child-model or --composed-tensors")
        child_records = runner._generate(
            child,
            child_tokenizer,
            rows,
            cap=int(args.maximum_new_tokens),
            device=str(args.device),
            eos_id=getattr(child_tokenizer, "eos_token_id", None),
            label=f"child:{args.label}",
        )
        del child
        torch.cuda.empty_cache()

    parent_records: list[dict[str, Any]] = []
    if not args.skip_parent:
        parent = cell_runner._load_parent(args.parent_model, str(args.device), dtype)
        parent_records = runner._generate(
            parent,
            parent_tokenizer,
            rows,
            cap=int(args.maximum_new_tokens),
            device=str(args.device),
            eos_id=getattr(parent_tokenizer, "eos_token_id", None),
            label="parent",
        )
        del parent
        torch.cuda.empty_cache()

    body = {
        "schema": DOSSIER_SCHEMA,
        "surface": f"{args.which}-v1",
        "purpose": (
            "ADDENDUM 16.1 second review surface.  This file PRICES AND ORDERS.  It "
            "carries no verdict, no count and no gate; the sealed fifteen remain the "
            "only judge of any frontier claim."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "campaign_wave": 45,
        "cell_id": receipt.get("cell_id"),
        "cell_instance_id": receipt.get("cell_instance_id"),
        "receipt_sha256": (
            hashlib.sha256(args.receipt.read_bytes()).hexdigest()
            if args.receipt is not None else None
        ),
        "bank_path": str(args.bank),
        "bank_sha256": bank_digest,
        "seal_path": str(args.seal),
        "seal_sha256": hashlib.sha256(args.seal.read_bytes()).hexdigest(),
        "prompt_count": len(rows),
        "device_compute_architecture": architecture,
        "dtype": "float32",
        "maximum_new_tokens": int(args.maximum_new_tokens),
        "child_tokenizer": str(args.child_tokenizer) if args.child_tokenizer else "shared-with-parent",
        "child_responses": child_records,
        "parent_responses": parent_records,
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(child_records)} child, {len(parent_records)} parent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
