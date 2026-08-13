#!/usr/bin/env python3
"""The frozen review, run against a PROBE rather than a sealed ARCH child.

Two of wave 20's cells cannot be sealed as architecture children and it is worth
saying exactly why before a driver is written that pretends otherwise:

* **CBT** installs a dense tensor.  A stock `Lfm2ForCausalLM` loads it and counts
  the parent's own 229,693,184 parameters, so its ADDENDUM 12 net ratio would be
  1.000000x and its ledger row would read as a frontier claim at no compression.
  The compression is real and it is in the STORAGE (245,824 numbers determine the
  tensor); R3 counts the loader, so the honest place for it is a probe.
* **F-LANDMARK** replaces the attention block's forward with a landmark memory.
  There is no config that expresses it, `Lfm2ForCausalLM.from_pretrained` cannot
  produce it, and R3 makes custom runtime research-only by construction.

What is NOT weakened is the review.  The protocol seal, the prompt pack digest,
the sixteen cases, the chat template, greedy decode, the 1,024-token cap and the
termination classes are the same objects `run_lfm2_composition_frozen_review.py`
uses, imported from it rather than reimplemented, so a probe dossier is
comparable to a sealed one case for case.  The parent lane is generated in the
same process and its digest is checked against the campaign's calibration.

Usage::

    mx20_probe_review.py --probe cbt --child <dir> --out <dossier.json>
    mx20_probe_review.py --probe landmark --landmark-state <dir> --out <...>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
#: This file sits at `<deploy>/reports/matrix-campaign-20260810/run/mx20/`, and
#: `Path.parents[0]` is already one level UP, so the deployment root is
#: `parents[3]`: run -> matrix-campaign-20260810 -> reports -> <deploy>.  Worth
#: spelling out, because the protocol seal and the prompt pack are resolved from
#: it and an off-by-one here is not an import error, it is a review that
#: authenticates against the wrong tree.
_DEPLOY = _HERE.parents[3]
for extra in (_DEPLOY / "src", _DEPLOY / "experiments", str(_HERE)):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_lfm2_composition_frozen_review as frozen  # noqa: E402
import run_lfm2_matrix_cell as cell_runner  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402

import mx20_build as mx  # noqa: E402
import mx20_parent_identity as parent_identity  # noqa: E402


def mx20_parent_identity_digest(lane: list[dict[str, Any]]) -> str:
    """The campaign's own parent-lane digest, computed by the campaign's own code.

    `mx20_parent_identity.lane_digest` takes a path; the probe has the lane in
    memory and must not write a file to hash it, so the payload construction is
    called out of that module rather than copied.
    """

    import hashlib
    import json as _json

    payload = _json.dumps(
        [[row["id"], row["response_verbatim"]] for row in lane],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    assert parent_identity is not None
    return hashlib.sha256(payload).hexdigest()[:24]

SCHEMA = "neural-forge-matrix-wave20-probe-frozen-review-dossier-v1"
#: The parent's own sixteen frozen-review responses digest to this on every
#: dossier since wave 7.  A probe run that does not reproduce it is not being
#: read against the same parent, and the driver says so instead of continuing.
#:
#: **The recipe is imported, not reinvented, and this file learned why.**  Its
#: first version digested the per-response `response_sha256` values instead, got
#: a number that was stable across every dossier in the campaign and equal to
#: none of the published ones, and reported "PARENT LANE MOVED" on a lane that
#: was byte-identical to wave 19's.  A calibration constant is only a
#: calibration against the function that produced it.
PARENT_LANE_DIGEST = "865f006e3813712ec202cb23"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["cbt", "landmark"])
    ap.add_argument("--child", type=Path, help="a stock checkpoint directory (CBT)")
    ap.add_argument(
        "--landmark-state", type=Path, help="the F-LANDMARK install artifact"
    )
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--parent-model", type=Path, default=mx.PARENT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument("--require-compute-architecture")
    ap.add_argument("--skip-parent", action="store_true")
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    started = time.monotonic()

    architecture = cell_runner.compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise SystemExit(
            f"--device {args.device} resolves to {architecture!r}, not "
            f"{args.require_compute_architecture!r}"
        )

    seal_path = _DEPLOY / "reports" / "r12-certified-20260804" / "PROTOCOL_SEAL.json"
    prompts_path = _DEPLOY / "research" / "lfm2_semantic_whole_response_prompts_v1.json"
    seal = frozen._authenticated_protocol(seal_path)
    rows, surface = frozen._authenticated_prompts(
        prompts_path, expected_file_sha256=str(seal["prompt_pack_sha256"])
    )

    probe_receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    dtype = cell_runner._dtype(str(args.dtype))
    tokenizer = cell_runner._load_tokenizer(Path(args.parent_model))
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if args.probe == "cbt":
        from transformers import Lfm2ForCausalLM

        tensors = Path(args.child) / "model.safetensors"
        observed = cell_runner._file_sha256(tensors)
        declared = str(probe_receipt["child_tensors_sha256"])
        if observed != declared:
            raise SystemExit(
                f"{tensors} digests {observed}; the probe receipt sealed {declared}"
            )
        child = (
            Lfm2ForCausalLM.from_pretrained(str(args.child), dtype=dtype)
            .to(args.device)
            .eval()
        )
        frozen._log(
            f"CBT probe loaded by the STOCK loader: {type(child).__name__}, "
            f"{sum(p.numel() for p in child.parameters()):,} parameters"
        )
        runtime = "stock transformers.Lfm2ForCausalLM"
    else:
        import mx20_probe_landmark as landmark

        child = landmark.load_probe(
            Path(args.landmark_state),
            parent=Path(args.parent_model),
            device=str(args.device),
            dtype=dtype,
            receipt=probe_receipt,
        )
        runtime = "custom: mx20_probe_landmark.LandmarkAttention installed at one block"

    child_records = frozen._generate(
        child,
        tokenizer,
        rows,
        cap=int(args.maximum_new_tokens),
        device=str(args.device),
        eos_id=eos_id,
        label="probe",
    )
    del child
    torch.cuda.empty_cache()

    parent_records: list[dict[str, Any]] = []
    parent_digest = None
    if not args.skip_parent:
        from transformers import Lfm2ForCausalLM

        parent = (
            Lfm2ForCausalLM.from_pretrained(str(args.parent_model), dtype=dtype)
            .to(args.device)
            .eval()
        )
        parent_records = frozen._generate(
            parent,
            tokenizer,
            rows,
            cap=int(args.maximum_new_tokens),
            device=str(args.device),
            eos_id=eos_id,
            label="parent",
        )
        del parent
        torch.cuda.empty_cache()
        parent_digest = mx20_parent_identity_digest(parent_records)
        frozen._log(
            f"parent lane digest {parent_digest} "
            f"({'matches' if parent_digest == PARENT_LANE_DIGEST else 'DIFFERS FROM'} "
            f"the campaign calibration {PARENT_LANE_DIGEST})"
        )

    dossier = {
        "schema": SCHEMA,
        "probe": args.probe,
        "status": "probe",
        "research_only": True,
        "runtime": runtime,
        "probe_receipt": probe_receipt,
        "protocol": {
            "protocol_sha256": str(seal["protocol_sha256"]),
            "prompt_pack_sha256": str(seal["prompt_pack_sha256"]),
        },
        "protocol_sha256": str(seal["protocol_sha256"]),
        "prompt_pack_sha256": str(seal["prompt_pack_sha256"]),
        "prompt_surface_sha256": surface,
        "compute_architecture": architecture,
        "dtype": str(args.dtype),
        "maximum_new_tokens": int(args.maximum_new_tokens),
        "parent_lane_digest": parent_digest,
        "parent_lane_digest_expected": PARENT_LANE_DIGEST,
        "parent_lane_matches_calibration": parent_digest == PARENT_LANE_DIGEST,
        "child_responses": child_records,
        "parent_responses": parent_records,
        "wall_time_seconds": round(time.monotonic() - started, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dossier, indent=1))
    frozen._log(f"probe dossier -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
