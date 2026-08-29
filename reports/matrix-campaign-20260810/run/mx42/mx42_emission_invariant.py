#!/usr/bin/env python3
"""The rotation-invariant re-measurement of wave 29's massive-activation finding.

Wave 29 (`NOTES-MX-AE` section 6) read a correlation off six children: this
parent carries exactly one massive-activation dimension, coordinate **294**, and
the only two children that "lose" it -- ``W720`` and ``MM768`` -- are exactly the
two whose generative margin collapses to a third of the parent's.  The finding
was adopted as this campaign's emission mechanism and the field synthesis's #2
was scheduled against it.

**The correlation is an artifact of the measurement's coordinate system.**
``mx29_emission.py`` compares massive-activation dimensions *by index*: the
parent's ``294`` against the child's own top-energy coordinates.  A width fold
rewrites the residual stream as ``z = E^T x`` -- ``fold_writer`` in
``mx29_build.py`` is literally ``E^T W`` -- so the child's coordinate 294 is not
the parent's coordinate 294, and an index comparison across that rotation cannot
mean what it says.  ``W720`` and ``MM768`` are the only two children in that
table whose stream is rotated, so they are the only two the index comparison can
fail on, and it failed on both.  The correlation is with *"is a rotating fold"*,
not with *"loses the massive channel"*.

Two quantities are invariant under the rotation and this script measures both.

**1. Is the direction still representable?**  ``||E^T e_294||^2`` -- computed
free from the sealed basis artifacts by ``mx42_massive_probe.py``, which reports
0.9960 at width 720 and 0.9917 at width 512.  Every width basis on the ladder
keeps the direction to three decimals.  There is nothing to carve out.

**2. Is the direction still written?**  The child's residual lifted back into
parent coordinates, ``x_hat = E z``, and the energy it carries along ``e_294``
against the parent's own.  Representable and written are different claims and
only the second one is about emission; a subspace can hold a direction the
folded model no longer drives.

The script also reports the *invariant* peakedness -- the top residual
coordinate's energy over the median in parent coordinates -- so "one massive
channel" versus "eleven smeared ones" is read after the rotation is undone
rather than through it.  Sink heads need no such correction: a width fold does
not change ``num_attention_heads`` and head index means the same thing on both
sides, so wave 29's sink numbers stand as measured.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

#: The parent's one massive-activation coordinate, sealed in wave 29's pre-read
#: at 884.97 mean square against a residual median of 5.34 -- 165.8x.
PARENT_MASSIVE = 294
SCHEMA = "neural-forge-matrix-wave42-emission-invariant-v1"


def load_probes(pack: Path, limit: int | None) -> list[dict[str, Any]]:
    """`mx29_emission.load_probes` verbatim.

    The pack is a JSON DOCUMENT with a `prompts` array (or a `triplets` map of
    lanes), not a JSONL stream.  The first draft of this file read it a line at
    a time and every child in the ladder failed identically on character 1 --
    which at least failed loudly, on all seven, rather than on some.
    """

    doc = json.loads(pack.read_text(encoding="utf-8"))
    rows = doc["prompts"] if "prompts" in doc else [
        r for lane in doc["triplets"].values() for r in lane
    ]
    return rows[:limit] if limit else rows


def residual_profile(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: str,
    basis: torch.Tensor | None,
) -> dict[str, Any]:
    """Mean-square residual energy per PARENT coordinate, plus the peakedness.

    ``basis`` is the child's ``E`` (1024 x width) when the stream is rotated and
    ``None`` when it is not.  With a basis the child's hidden state is lifted,
    ``x_hat = E z``, before anything is measured; without one the stream is
    already in parent coordinates and the lift would be the identity.
    """

    total: torch.Tensor | None = None
    rows_seen = 0
    for row in rows:
        messages = [dict(m) for m in row["messages"]]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover
            text = "\n".join(m["content"] for m in messages)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1][0].float()  # [T, H_child]
        if basis is not None:
            if hidden.shape[1] != basis.shape[1]:
                raise SystemExit(
                    f"basis is {basis.shape[0]}x{basis.shape[1]} and the child's "
                    f"stream is {hidden.shape[1]}-wide; these are not the same fold"
                )
            hidden = hidden @ basis.T.to(hidden.dtype)  # [T, 1024], x_hat = E z
        energy = (hidden ** 2).mean(dim=0)
        total = energy if total is None else total + energy
        rows_seen += 1

    energy = total / max(rows_seen, 1)
    median = float(energy.median())
    order = torch.argsort(energy, descending=True)
    top = int(order[0])
    # How many coordinates clear the same 20x-median bar wave 29 used, now that
    # the bar is applied in one coordinate system for every child.
    above = [int(i) for i in order[:64].tolist() if float(energy[int(i)]) > 20.0 * max(median, 1e-9)]
    return {
        "coordinates": int(energy.shape[0]),
        "median_energy": round(median, 6),
        "massive_dim_energy": round(float(energy[PARENT_MASSIVE]), 4),
        "massive_dim_peakedness": round(float(energy[PARENT_MASSIVE]) / max(median, 1e-9), 2),
        "massive_dim_rank": int((energy > energy[PARENT_MASSIVE]).sum()),
        "top_coordinate": top,
        "top_energy": round(float(energy[top]), 4),
        "top_peakedness": round(float(energy[top]) / max(median, 1e-9), 2),
        "dims_above_20x_median": above,
        "dims_above_20x_count": len(above),
        "top8": {str(int(i)): round(float(energy[int(i)]), 4) for i in order[:8].tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--child-tokenizer", type=Path)
    ap.add_argument(
        "--basis",
        type=Path,
        help=(
            "the sealed basis artifact this child's width fold was built "
            "against.  Required for a rotated stream and REFUSED when the "
            "child's width already equals the parent's, because a lift that is "
            "not needed is a lift that can only be wrong"
        ),
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import os
    import sys

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for sub in ("src", "experiments"):
        p = str(args.deployment / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from safetensors.torch import load_file  # noqa: E402
    from transformers import AutoTokenizer, Lfm2ForCausalLM  # noqa: E402

    torch.set_grad_enabled(False)
    started = time.monotonic()
    rows = load_probes(args.probes, args.limit)

    parent_cfg = json.loads((args.parent / "config.json").read_text())
    child_cfg = json.loads((args.child / "config.json").read_text())
    p_hidden = int(parent_cfg["hidden_size"])
    c_hidden = int(child_cfg["hidden_size"])

    basis: torch.Tensor | None = None
    if args.basis is not None:
        if p_hidden == c_hidden:
            raise SystemExit(
                f"REFUSED: --basis given but the child's hidden_size is {c_hidden}, "
                f"the parent's own.  This stream is not rotated and lifting it "
                f"through a basis it was not folded with would manufacture the "
                f"very artifact this script exists to remove"
            )
        basis = load_file(str(args.basis))["basis"].to(torch.float32).to(args.device)
        if basis.shape != (p_hidden, c_hidden):
            raise SystemExit(
                f"REFUSED: basis is {tuple(basis.shape)} and the fold is "
                f"{p_hidden}->{c_hidden}"
            )
    elif p_hidden != c_hidden:
        raise SystemExit(
            f"REFUSED: the child's stream is {c_hidden}-wide against the parent's "
            f"{p_hidden} and no --basis was given.  Measuring it by index is "
            f"exactly wave 29's error and this script will not repeat it"
        )

    parent = Lfm2ForCausalLM.from_pretrained(
        str(args.parent), dtype=torch.float32
    ).to(args.device).eval()
    parent_tok = AutoTokenizer.from_pretrained(str(args.parent))
    parent_profile = residual_profile(parent, parent_tok, rows, args.device, None)
    del parent
    torch.cuda.empty_cache()

    child = Lfm2ForCausalLM.from_pretrained(
        str(args.child), dtype=torch.float32
    ).to(args.device).eval()
    child_tok = (
        AutoTokenizer.from_pretrained(str(args.child_tokenizer))
        if args.child_tokenizer
        else parent_tok
    )
    child_profile = residual_profile(child, child_tok, rows, args.device, basis)
    del child
    torch.cuda.empty_cache()

    # The two claims, separated.  `representable` is a property of the fold's
    # subspace and costs no forward pass; `written` is a property of the folded
    # model and costs the two above.
    representable = None
    if basis is not None:
        e = torch.zeros(p_hidden, dtype=torch.float64)
        e[PARENT_MASSIVE] = 1.0
        coeff = basis.double().cpu().T @ e
        representable = round(float((coeff ** 2).sum()), 6)

    written = child_profile["massive_dim_energy"] / max(
        parent_profile["massive_dim_energy"], 1e-9
    )
    report = {
        "schema": SCHEMA,
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_pack": str(args.probes),
        "probes_used": len(rows),
        "parent_hidden": p_hidden,
        "child_hidden": c_hidden,
        "basis": str(args.basis) if args.basis else None,
        "lifted_to_parent_coordinates": basis is not None,
        "parent": parent_profile,
        "child": child_profile,
        "invariant": {
            "massive_direction_representable": representable,
            "massive_direction_energy_retained": round(written, 4),
            "massive_direction_peakedness_parent": parent_profile["massive_dim_peakedness"],
            "massive_direction_peakedness_child": child_profile["massive_dim_peakedness"],
            "massive_dim_is_still_the_top_coordinate": (
                child_profile["top_coordinate"] == PARENT_MASSIVE
            ),
            "dims_above_20x_parent": parent_profile["dims_above_20x_count"],
            "dims_above_20x_child": child_profile["dims_above_20x_count"],
        },
        "reading": [],
        "wall_time_seconds": round(time.monotonic() - started, 2),
    }
    inv = report["invariant"]
    if inv["massive_dim_is_still_the_top_coordinate"] and written > 0.5:
        inv_read = (
            f"MASSIVE CHANNEL INTACT: coordinate {PARENT_MASSIVE} is still this "
            f"child's top residual coordinate and carries {written:.2f}x the "
            f"parent's energy on it"
        )
    elif written > 0.5:
        inv_read = (
            f"MASSIVE CHANNEL PRESENT BUT DEMOTED: coordinate {PARENT_MASSIVE} "
            f"carries {written:.2f}x the parent's energy and now ranks "
            f"{child_profile['massive_dim_rank']}"
        )
    else:
        inv_read = (
            f"MASSIVE CHANNEL LOST: coordinate {PARENT_MASSIVE} retains only "
            f"{written:.3f} of the parent's energy on it"
        )
    report["reading"].append(inv_read)
    if basis is not None:
        report["reading"].append(
            "measured in PARENT coordinates: the child's stream was lifted through "
            "its own fold basis before anything was read, so this is not wave 29's "
            "index comparison"
        )
    print(json.dumps(report["invariant"], indent=1))
    print(inv_read)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
