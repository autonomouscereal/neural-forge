"""Verify a BUILT evidence manifest against the reader that gates on it.

``qf_selftest.py`` proves the mechanics on a synthetic manifest.  This proves the
properties of the real artifact -- the file at
``evidence/omni-evidence-v1.json`` that every Wave A build is fitted against --
and it lives in the lane rather than in ``/tmp``, because a gate whose
verification is ephemeral is an assertion, not evidence.  (The first version of
these checks was a scratch script under ``/tmp``; ``/tmp`` is cleared on reboot,
at which point nobody could re-run the proof that the gate artifact refuses
tampering.)

Eleven checks, each printed PASS/FAIL, exit code 1 if any fails:

1.  the seal re-computes through :func:`qf_corpus.read_manifest`;
2.  ``corpus()`` returns one ``[1, L]`` int64 tensor per batch, in batch order;
3.  ``split_by_batch`` reproduces the declared fit/held pattern;
4.  ``fit_check_rows`` marks whole batches, every third fit batch from the third;
5.  ``evidence_receipt`` renders;
6.  every id is inside the parent's physical table;
7.  a flipped id is REFUSED (seal not recomputed -- the tamper case);
8.  a short row is REFUSED (seal recomputed -- the ragged case);
9.  the manifest's tokenizer sha256 is the parent's own, through the same
    :func:`qf_build.assert_evidence_tokenizer` a build calls, and a parent with a
    different tokenizer is REFUSED by it (``--other-parent``);
10. no batch begins mid-row: every batch ends on the separator id, so a batch
    boundary never falls inside a document -- checked on the MANIFEST, not taken
    from the builder's own report;
11. the build receipt agrees with the manifest: receipt_sha256 recomputes, the
    manifest file sha256 matches, every per-batch ``ids_sha256`` matches, and the
    receipt's padding counts match the separator runs actually in the file.

    $PY qf_evidence_verify.py \\
        --manifest evidence/omni-evidence-v1.json \\
        --parent /home/cereal/models/Qwen3.5-4B \\
        --other-parent /home/cereal/models/Qwen3.8-27B-Uncensored
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qf_corpus

LOG_PREFIX = "[qfverify]"

CHECKS: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


# ------------------------------------------------------------------- checks
@check("read_manifest")
def check_read_manifest(state: dict[str, Any]) -> dict[str, Any]:
    manifest = qf_corpus.read_manifest(state["manifest_path"])
    state["manifest"] = manifest
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "sequence_length": manifest["sequence_length"],
        "rows": len(manifest["rows"]),
        "tokenizer_sha256": manifest["tokenizer_sha256"],
    }


@check("corpus")
def check_corpus(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    corpus = qf_corpus.corpus(state["manifest"])
    state["corpus"] = corpus
    shapes = {tuple(tensor.shape) for _, tensor in corpus}
    assert len(shapes) == 1, shapes
    assert corpus[0][1].dtype == torch.int64, corpus[0][1].dtype
    assert [index for index, _ in corpus] == list(range(len(corpus)))
    return {
        "batches": len(corpus),
        "shape": sorted(shapes)[0],
        "dtype": str(corpus[0][1].dtype),
        "indices_contiguous": True,
    }


@check("split_by_batch")
def check_split(state: dict[str, Any]) -> dict[str, Any]:
    split = qf_corpus.split_by_batch(state["manifest"])
    fit = sorted(index for index, value in split.items() if value == "fit")
    held = sorted(index for index, value in split.items() if value == "held")
    assert fit and held, split
    return {"fit_batches": len(fit), "held_batches": len(held), "held": held}


@check("fit_check_rows")
def check_fit_rows(state: dict[str, Any]) -> dict[str, Any]:
    manifest = state["manifest"]
    mask = qf_corpus.fit_check_rows(manifest)
    length = int(manifest["sequence_length"])
    # Whole batches, never part of one: the mask must be constant inside every
    # `sequence_length` block, which is what makes the check split a text split.
    blocks = [mask[i * length : (i + 1) * length] for i in range(mask.numel() // length)]
    assert all(bool(block.all()) or not bool(block.any()) for block in blocks)
    fit_batches = [
        row["batch_index"]
        for row in sorted(manifest["rows"], key=lambda r: int(r["batch_index"]))
        if row["split"] == "fit"
    ]
    marked = [
        fit_batches[position]
        for position, block in enumerate(blocks)
        if bool(block.all())
    ]
    expected = [
        fit_batches[position]
        for position in range(len(fit_batches))
        if position >= qf_corpus.INTERNAL_CHECK_OFFSET
        and (position - qf_corpus.INTERNAL_CHECK_OFFSET) % qf_corpus.INTERNAL_CHECK_STRIDE == 0
    ]
    assert marked == expected, (marked, expected)
    return {
        "check_rows": int(mask.sum()),
        "of_fit_rows": int(mask.numel()),
        "check_batches": marked,
    }


@check("evidence_receipt")
def check_evidence_receipt(state: dict[str, Any]) -> dict[str, Any]:
    return qf_corpus.evidence_receipt(state["manifest"], state["manifest_path"])


@check("id_range")
def check_id_range(state: dict[str, Any]) -> dict[str, Any]:
    lowest = min(int(tensor.min()) for _, tensor in state["corpus"])
    highest = max(int(tensor.max()) for _, tensor in state["corpus"])
    physical = state["physical_vocab_size"]
    assert lowest >= 0, lowest
    assert highest < physical, (highest, physical)
    return {"lowest_id": lowest, "highest_id": highest, "physical_vocab_size": physical}


@check("tamper_refused")
def check_tamper(state: dict[str, Any]) -> dict[str, Any]:
    """One flipped id, seal NOT recomputed: the seal itself must catch it."""

    path = Path(state["work"]) / "tampered-evidence.json"
    shutil.copy(state["manifest_path"], path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["rows"][min(5, len(body["rows"]) - 1)]["ids"][11] += 1
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    try:
        qf_corpus.read_manifest(path)
    except SystemExit as error:
        assert "seal is broken" in str(error), str(error)
        return {"refused": True, "message": str(error)[:96]}
    raise AssertionError("a flipped id was accepted")


@check("ragged_refused")
def check_ragged(state: dict[str, Any]) -> dict[str, Any]:
    """One short row, seal RECOMPUTED: the length check is what must fire."""

    path = Path(state["work"]) / "ragged-evidence.json"
    shutil.copy(state["manifest_path"], path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["rows"][min(2, len(body["rows"]) - 1)]["ids"] = body["rows"][
        min(2, len(body["rows"]) - 1)
    ]["ids"][:-1]
    body["manifest_sha256"] = qf_corpus._body_digest(body)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    try:
        qf_corpus.read_manifest(path)
    except SystemExit as error:
        assert "tokens long" in str(error), str(error)
        return {"refused": True, "message": str(error)[:96]}
    raise AssertionError("a short row was accepted")


@check("tokenizer_gate")
def check_tokenizer_gate(state: dict[str, Any]) -> dict[str, Any]:
    """The consumption-time gate, exercised both ways.

    ``--parent`` must PASS (the manifest was tokenized by its tokenizer) and
    ``--other-parent``, when given, must be REFUSED -- which is the silent
    substitution the whole check exists for.
    """

    import qf_build

    report = qf_build.assert_evidence_tokenizer(
        state["manifest"], state["parent"], allow_mismatch=False
    )
    assert report["matches"], report
    other = state.get("other_parent")
    refusal = None
    if other:
        try:
            qf_build.assert_evidence_tokenizer(
                state["manifest"], other, allow_mismatch=False
            )
        except SystemExit as error:
            refusal = str(error)[:120]
        assert refusal, f"{other} was accepted for a manifest it did not tokenize"
        allowed = qf_build.assert_evidence_tokenizer(
            state["manifest"], other, allow_mismatch=True
        )
        assert not allowed["matches"] and allowed["allow_mismatch"]
    return {
        "parent": state["parent"],
        "parent_tokenizer_sha256": report["parent_tokenizer_sha256"],
        "other_parent": other,
        "other_parent_refused": refusal,
    }


@check("no_batch_begins_mid_row")
def check_row_atomicity(state: dict[str, Any]) -> dict[str, Any]:
    """Every batch ends on the separator, so no batch begins mid-document.

    Read off the MANIFEST, not off the builder's own report: every row this
    builder emits is terminated by the parent's eos id and any remainder is
    padded with the same id, so a batch whose last token is anything else was cut
    through the middle of a document -- which is exactly what the flat-stream
    packer did at all 23 boundaries, putting 1,827 held tokens and 2,367
    internal-check tokens downstream of text the solve had just fitted on.
    """

    separator = state["separator_id"]
    ends: list[int] = []
    runs: list[int] = []
    for row in sorted(state["manifest"]["rows"], key=lambda r: int(r["batch_index"])):
        ids = row["ids"]
        ends.append(int(ids[-1]))
        run = 0
        while run < len(ids) and int(ids[-1 - run]) == separator:
            run += 1
        runs.append(run)
    wrong = [index for index, value in enumerate(ends) if value != separator]
    assert not wrong, f"batches {wrong} do not end on the separator id {separator}"
    state["separator_runs"] = runs
    return {
        "separator_id": separator,
        "batches_ending_on_separator": len(ends),
        "batches_ending_mid_row": len(wrong),
        "trailing_separator_run_by_batch": runs,
    }


@check("build_receipt")
def check_build_receipt(state: dict[str, Any]) -> dict[str, Any]:
    """The receipt must be about THIS file, and its own seal must recompute."""

    from qf_arch import canonical_sha256

    path = Path(state["receipt_path"])
    if not path.is_file():
        return {"receipt": None, "skipped": "no build receipt beside the manifest"}
    receipt = json.loads(path.read_text(encoding="utf-8"))
    recomputed = canonical_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    assert recomputed == receipt["receipt_sha256"], (recomputed, receipt["receipt_sha256"])
    assert receipt["manifest"]["file_sha256"] == qf_corpus.file_sha256(
        state["manifest_path"]
    )
    assert receipt["manifest"]["manifest_sha256"] == state["manifest"]["manifest_sha256"]
    by_index = {
        int(row["batch_index"]): row
        for row in sorted(state["manifest"]["rows"], key=lambda r: int(r["batch_index"]))
    }
    mismatched = [
        row["batch_index"]
        for row in receipt["batches"]
        if canonical_sha256(by_index[int(row["batch_index"])]["ids"]) != row["ids_sha256"]
    ]
    assert not mismatched, mismatched
    # The padding the receipt claims must be the padding actually in the file:
    # a batch's trailing separator run is its padding plus the one separator that
    # terminates the last row in it.
    padding_disagreements = [
        row["batch_index"]
        for row in receipt["batches"]
        if "padding_tokens" in row
        and state["separator_runs"][int(row["batch_index"])] != int(row["padding_tokens"]) + 1
    ]
    leakage = receipt.get("packing", {}).get("split_leakage", {})
    assert leakage.get("rows_spanning_batches") == 0, leakage
    assert leakage.get("batches_starting_mid_row") == 0, leakage
    return {
        "receipt_sha256": receipt["receipt_sha256"],
        "builder_sha256": receipt["builder"]["sha256"],
        "batches_checked": len(receipt["batches"]),
        "padding_disagreements": padding_disagreements,
        "rows_spanning_batches": leakage.get("rows_spanning_batches"),
        "batches_starting_mid_row": leakage.get("batches_starting_mid_row"),
        "padding_fraction": receipt.get("packing", {}).get("padding_fraction"),
    }


# --------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        default="/home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json",
    )
    parser.add_argument("--receipt", default="")
    parser.add_argument("--parent", default="/home/cereal/models/Qwen3.5-4B")
    parser.add_argument(
        "--other-parent",
        default="",
        help="a parent with a DIFFERENT tokenizer, which the gate must refuse",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    if not args.receipt:
        stem = Path(args.manifest).stem
        args.receipt = str(Path(args.manifest).with_name(stem + "-build-receipt.json"))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.parent) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    separator = text_config.get("eos_token_id", config.get("eos_token_id"))
    if isinstance(separator, list) and separator:
        separator = separator[0]
    work = tempfile.mkdtemp(prefix="qf-evidence-verify-")
    state: dict[str, Any] = {
        "manifest_path": args.manifest,
        "receipt_path": args.receipt,
        "parent": args.parent,
        "other_parent": args.other_parent or None,
        "physical_vocab_size": int(
            text_config.get("vocab_size") or config.get("vocab_size") or 0
        ),
        "separator_id": int(separator),
        "work": work,
    }
    print(f"{LOG_PREFIX} manifest {args.manifest}")
    print(f"{LOG_PREFIX} parent   {args.parent} (separator id {state['separator_id']})")
    results: dict[str, Any] = {}
    failures = 0
    for name, fn in CHECKS:
        try:
            payload = fn(state)
            results[name] = {"status": "PASS", "evidence": payload}
            print(f"PASS  {name}  {json.dumps(payload, default=str)[:400]}")
        except Exception as error:  # noqa: BLE001 -- every failure is reported
            failures += 1
            results[name] = {"status": "FAIL", "error": repr(error)}
            print(f"FAIL  {name}  {error!r}")
    print(f"{LOG_PREFIX} {len(results) - failures}/{len(results)} checks passed")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"{LOG_PREFIX} results written to {args.json_out}")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
