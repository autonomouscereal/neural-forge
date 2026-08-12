#!/usr/bin/env python3
"""Generate the frozen-protocol review surface for one sealed composed child.

The campaign's authoritative behavioural judge is the frozen V5 manual review
(`reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md`,
digest-pinned by `PROTOCOL_SEAL.json`), and ADDENDUM 2 records the criterion it
supplies: a child's fifteen intelligence cases are graded under that protocol
and compared with a reference band -- the sealed A9 child's 8 of 15.

This script produces the *evidence* that review is taken on, and nothing else.
It grades nothing, it writes no verdict, and it cannot: the verdict is sealed by
`experiments/seal_matrix_cell_review.py` as a separate instance, because §5 rule
1 makes the review a judgement of verbatim text and a runner may not certify its
own behaviour.

Three things are pinned before a token is generated, in this order:

1. **The protocol.**  `PROTOCOL_SEAL.json` must re-seal to its own recorded
   `protocol_sha256`, and the protocol file must digest to it.  The seal exists
   precisely because the grading criteria were pinned nowhere and a post-hoc
   softening would have been undetectable.
2. **The prompt surface.**  The prompt pack must digest to the seal's
   `prompt_pack_sha256` -- the exact byte-form the seal names, not the
   `ff6fa27e…` fork `NOTES-MX-A` §8 reported -- and every row must pass the
   estate's own `lfm2_carrier_review_cli._load_prompts` validation.  The
   protocol also publishes a *normalized* surface digest (`5615b9e4…`) computed
   by machinery this script does not reproduce; a surface digest is recorded
   here under its own name and is **not** claimed to be that number.  The raw
   byte-form check is the binding one and it is the seal's own.
3. **The child.**  The composed child is reconstituted from the parent plus the
   sealed `composed-child-tensors.pt`, and every folded tensor's sha256 must
   equal the digest the composition receipt sealed for it.  A dossier that
   cannot prove it graded the sealed child is worthless, so this is a refusal,
   not a warning.

**ADDENDUM 12.**  A standard-shape child (`ARCH.<TAG>`) is not reconstituted at
all -- it is a directory that stock `transformers` loads, which is the whole
content of the claim -- so ``--child-model`` replaces ``--composed-tensors`` for
those receipts and the binding check moves with it: the directory's
``model.safetensors`` must digest to the ``architecture.child_tensors_sha256``
the receipt sealed, the loader must return the class the receipt sealed, and the
live parameter count must equal the sealed one.  Everything downstream -- the
protocol, the prompt surface, the 1,024-token cap, the same-process parent
control, the dossier schema and its ``receipt_sha256`` binding -- is byte-for-byte
the same, which is the point: a narrowed child and a parent-shaped one are graded
on one surface or their scores are not comparable.

The parent is decoded on the same prompts, in the same process, with the same
settings, because the protocol's parent-comparison section needs a control that
differs from the child in exactly one thing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import torch  # noqa: E402

import run_lfm2_matrix_cell as cell_runner  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256  # noqa: E402
from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CellId,
    schema_version_of,
    verify_seal,
)

DOSSIER_SCHEMA = "neural-forge-lfm2-matrix-composition-frozen-review-dossier-v1"

_log = cell_runner._log


class FrozenReviewError(RuntimeError):
    """The review surface could not be established.  Always fatal."""


def _authenticated_protocol(seal_path: Path) -> dict[str, Any]:
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    if seal.get("schema") != "neural-forge-manual-review-protocol-seal-v1":
        raise FrozenReviewError(f"{seal_path} is not a protocol seal.")
    protocol = Path(seal_path).parent / Path(seal["protocol_file"]).name
    if not protocol.is_file():
        raise FrozenReviewError(f"Protocol document missing at {protocol}.")
    observed = cell_runner._file_sha256(protocol)
    if observed != str(seal["protocol_sha256"]):
        raise FrozenReviewError(
            f"Protocol digest {observed} is not the sealed "
            f"{seal['protocol_sha256']}; a review claiming to follow the frozen "
            "protocol must be judged against that digest."
        )
    _log(f"protocol authenticated: {observed[:16]} ({protocol.name})")
    return seal


def _authenticated_prompts(
    path: Path, *, expected_file_sha256: str
) -> tuple[tuple[dict[str, Any], ...], str]:
    from neural_forge.lfm2_carrier_review_cli import _load_prompts

    rows = _load_prompts(path, expected_file_sha256=expected_file_sha256)
    surface = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-matrix-composition-prompt-surface-v1",
            "prompts": [
                {
                    "id": str(row["id"]),
                    "category": str(row["category"]),
                    "prompt": row.get("prompt"),
                    "messages": row.get("messages"),
                }
                for row in rows
            ],
        }
    )
    _log(f"prompt surface: {len(rows)} cases, surface digest {surface[:16]}")
    return rows, surface


def _messages_for(row: Mapping[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), Sequence) and not isinstance(
        row.get("messages"), (str, bytes)
    ):
        return [dict(message) for message in row["messages"]]
    return [{"role": "user", "content": str(row["prompt"])}]


@torch.no_grad()
def _generate(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    device: str,
    eos_id: int | None,
    label: str,
) -> list[dict[str, Any]]:
    """Greedy decode one response per case, recording termination separately.

    The protocol records termination as `complete` / `censored-at-cap` /
    `runtime-error` and says a cap hit is never itself a failure, so the three
    outcomes are distinguished here rather than inferred by a reader from the
    text's length.
    """

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        started = time.monotonic()
        messages = _messages_for(row)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - the LFM2 tokenizer carries a template
            text = "\n".join(message["content"] for message in messages)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generated: list[int] = []
        past = None
        current = ids
        termination = "censored-at-cap"
        try:
            for _step in range(cap):
                result = model(
                    input_ids=current, past_key_values=past, use_cache=True
                )
                past = result.past_key_values
                next_id = int(result.logits[0, -1].argmax().item())
                if eos_id is not None and next_id == eos_id:
                    termination = "complete"
                    break
                generated.append(next_id)
                current = torch.tensor(
                    [[next_id]], device=device, dtype=ids.dtype
                )
        except RuntimeError as exc:  # noqa: BLE001 - recorded, never swallowed
            termination = "runtime-error"
            _log(f"{label} {row['id']}: {type(exc).__name__}: {exc}")
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        records.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "prompt_messages": messages,
                "response_verbatim": decoded,
                "response_sha256": canonical_sha256({"text": decoded}),
                "generated_tokens": len(generated),
                "termination": termination,
                "wall_time_seconds": round(time.monotonic() - started, 3),
            }
        )
        _log(
            f"{label} {index}/{len(rows)} {row['id']}: {len(generated)} tokens "
            f"({termination}) in {records[-1]['wall_time_seconds']:.1f}s"
        )
    return records


def _expected_digests(receipt: Mapping[str, Any]) -> tuple[dict[str, str], Any]:
    """The folded tensors this receipt sealed, and the file digest it declared.

    Two receipt shapes reach this script and they seal the same fact in two
    places.  A **composition** receipt folds many components and lists each
    one's installed digests under ``composition.components``; a **cell**
    receipt folds exactly one component and lists its digests under
    ``solve.installed_tensor_digests`` (ADDENDUM 6, `--save-installed-tensors`).
    Reading both here is what lets one review surface -- one protocol digest,
    one prompt pack, one 1,024-token cap, one same-process parent control --
    grade a composed child and a single-component child on the same terms.
    That comparability is the whole reason ADDENDUM 6 can put a cell's score
    beside the parent's 12/15 at all.
    """

    expected: dict[str, str] = {}
    composition = receipt.get("composition")
    if isinstance(composition, Mapping):
        for entry in composition["components"]:
            for record in entry["installed_tensor_digests"]:
                key, _, digest = str(record).partition("=")
                expected[key] = digest
        return expected, composition.get("composed_child_tensors_sha256")
    records = receipt.get("solve", {}).get("installed_tensor_digests")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise FrozenReviewError(
            f"{receipt['cell_id']} sealed no installed tensor digests, so the "
            "child it is about cannot be reconstituted.  Re-run the cell with "
            "--save-installed-tensors."
        )
    for record in records:
        key, _, digest = str(record).partition("=")
        expected[key] = digest
    if not expected:
        raise FrozenReviewError(
            f"{receipt['cell_id']} sealed an empty installed tensor digest list."
        )
    return expected, receipt["solve"].get("installed_tensors_sha256")


def _reconstitute(
    *,
    parent_model: Path,
    receipt: Mapping[str, Any],
    tensors_path: Path,
    device: str,
    dtype: torch.dtype,
    corrections_path: Path | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Parent + the sealed folded tensors, proven equal to the sealed child.

    **ADDENDUM 10.**  A child that carries corrective routes is not its folded
    tensors: the routes are additive hooks the ``state_dict`` cannot express, so
    a review that loaded only `composed-child-tensors.pt` would grade a
    *different model* than the one the receipt seals, and would grade it as if it
    were the sealed one.  So the rule is symmetric with the folded half and just
    as hard: if the receipt seals a corrections block, the corrections file is
    mandatory, the file's own digest must match, every route tensor's digest must
    match, and the bank is attached before a single token is generated.
    """

    expected, declared = _expected_digests(receipt)
    observed_file = cell_runner._file_sha256(tensors_path)
    if declared is not None and str(declared) != observed_file:
        raise FrozenReviewError(
            f"{tensors_path} digests {observed_file}, the receipt sealed "
            f"{declared}."
        )
    model = cell_runner._load_parent(parent_model, device, dtype)
    folded = torch.load(tensors_path, weights_only=True)
    if sorted(folded) != sorted(expected):
        raise FrozenReviewError(
            f"The saved tensors {sorted(folded)} are not the receipt's folded "
            f"set {sorted(expected)}."
        )
    cell_runner._install_component(model, folded)
    state = model.state_dict()
    for key, digest in sorted(expected.items()):
        observed = cell_runner._tensor_sha256(state[key])
        if observed != digest:
            raise FrozenReviewError(
                f"Reconstituted {key} digests {observed}; the receipt sealed "
                f"{digest}.  This dossier would not be about the sealed child."
            )
    composition = receipt.get("composition")
    sealed_corrections = (
        composition.get("corrections") if isinstance(composition, Mapping) else None
    )
    corrections_summary: dict[str, Any] | None = None
    if sealed_corrections is not None:
        from neural_forge.matrix_corrective_routes import load_bank_from_receipt

        if corrections_path is None or not Path(corrections_path).is_file():
            raise FrozenReviewError(
                f"{receipt['cell_id']} seals "
                f"{sealed_corrections['route_count']} corrective route(s) but no "
                "--composed-corrections file was supplied; the child cannot be "
                "reconstituted and no dossier may be written."
            )
        declared_file = sealed_corrections.get("corrections_tensors_sha256")
        observed_corrections = cell_runner._file_sha256(Path(corrections_path))
        if declared_file is not None and str(declared_file) != observed_corrections:
            raise FrozenReviewError(
                f"{corrections_path} digests {observed_corrections}, the receipt "
                f"sealed {declared_file}."
            )
        payload = torch.load(Path(corrections_path), weights_only=True)
        bank = load_bank_from_receipt(
            receipt=receipt, tensors=payload["dense"]
        )
        bank.attach(model)
        corrections_summary = {
            "route_count": len(bank),
            "route_ids": [route.text for route in bank.routes],
            "route_tensor_digests": bank.digests(),
            "stored_parameter_count": bank.stored_parameter_count(),
            "corrections_tensors_sha256": observed_corrections,
            "corrections_tensors_path": str(corrections_path),
        }
        _log(
            f"child reconstituted: {len(bank)} corrective route(s) attached, "
            "digests match"
        )
    _log(f"child reconstituted: {len(expected)} folded tensors, digests match")
    return model, corrections_summary


def _load_standard_shape_child(
    *,
    receipt: Mapping[str, Any],
    child_model: Path,
    device: str,
    dtype: torch.dtype,
) -> Any:
    """ADDENDUM 12: load the sealed child directory the way R3 says it loads.

    Three refusals, in the order that makes each one cheap: the bytes, then the
    loader, then the count.  There is no tensor-by-tensor digest check here
    because there is no per-tensor claim to check -- the receipt seals the file,
    and the file is the child.
    """

    block = receipt.get("architecture")
    if not isinstance(block, Mapping):
        raise FrozenReviewError(
            f"{receipt['cell_id']} carries no architecture block, so there is no "
            "standard-shape child to load."
        )
    tensors = Path(child_model) / "model.safetensors"
    if not tensors.is_file():
        raise FrozenReviewError(f"No model.safetensors under {child_model}.")
    observed = cell_runner._file_sha256(tensors)
    declared = block.get("child_tensors_sha256")
    if declared is not None and str(declared) != observed:
        raise FrozenReviewError(
            f"{tensors} digests {observed}, the receipt sealed {declared}.  This "
            "dossier would not be about the sealed child."
        )
    from transformers import Lfm2ForCausalLM

    model = Lfm2ForCausalLM.from_pretrained(str(child_model), dtype=dtype)
    sealed_class = str(block.get("export", {}).get("loaded_class"))
    if type(model).__name__ != sealed_class:
        raise FrozenReviewError(
            f"{child_model} loaded as {type(model).__name__}; the receipt sealed "
            f"{sealed_class}."
        )
    live = int(sum(p.numel() for p in model.parameters()))
    sealed_count = int(block["export"]["parameters_counted_live"])
    if live != sealed_count:
        raise FrozenReviewError(
            f"{child_model} loaded {live} parameters; the receipt sealed "
            f"{sealed_count}."
        )
    _log(
        f"standard-shape child loaded: {type(model).__name__}, {live:,} "
        f"parameters, no trust_remote_code, digest matches"
    )
    return model.to(device).eval()


def run(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    started = time.monotonic()

    architecture = cell_runner.compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise FrozenReviewError(
            f"--device {args.device} resolves to {architecture!r}, not "
            f"{args.require_compute_architecture!r}."
        )

    seal = _authenticated_protocol(Path(args.protocol_seal))
    rows, surface = _authenticated_prompts(
        Path(args.prompts_json),
        expected_file_sha256=str(seal["prompt_pack_sha256"]),
    )

    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    verify_seal(receipt)
    # The version is read off the body, never assumed: ADDENDUM 12's id form is
    # illegal under v1 and v2 and `CellId.parse` says so.
    cell = CellId.parse(
        str(receipt["cell_id"]), schema_version=schema_version_of(receipt)
    )
    if cell.kind not in ("composition", "cell", "architecture"):
        raise FrozenReviewError(
            f"{cell.text} is not a composed child, a component cell, or a "
            "standard-shape child."
        )

    dtype = cell_runner._dtype(str(args.dtype))
    tokenizer = cell_runner._load_tokenizer(Path(args.parent_model))
    eos_id = getattr(tokenizer, "eos_token_id", None)

    corrections_summary: dict[str, Any] | None = None
    if cell.kind == "architecture":
        if args.child_model is None:
            raise FrozenReviewError(
                f"{cell.text} is a standard-shape child and needs --child-model; "
                "it has no folded-tensor file to reconstitute from."
            )
        if args.composed_tensors is not None:
            raise FrozenReviewError(
                "--composed-tensors is meaningless for a standard-shape child: "
                "there are no parent-shaped folded tensors to install."
            )
        child = _load_standard_shape_child(
            receipt=receipt,
            child_model=Path(args.child_model),
            device=str(args.device),
            dtype=dtype,
        )
    else:
        if args.composed_tensors is None:
            raise FrozenReviewError(
                f"{cell.text} is a {cell.kind} cell and needs --composed-tensors."
            )
        child, corrections_summary = _reconstitute(
            parent_model=Path(args.parent_model),
            receipt=receipt,
            tensors_path=Path(args.composed_tensors),
            device=str(args.device),
            dtype=dtype,
            corrections_path=(
                Path(args.composed_corrections)
                if args.composed_corrections is not None
                else None
            ),
        )
    child_records = _generate(
        child,
        tokenizer,
        rows,
        cap=int(args.maximum_new_tokens),
        device=str(args.device),
        eos_id=eos_id,
        label="child",
    )
    del child
    torch.cuda.empty_cache()

    parent_records: list[dict[str, Any]] = []
    if not args.skip_parent:
        parent = cell_runner._load_parent(
            Path(args.parent_model), str(args.device), dtype
        )
        parent_records = _generate(
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

    body: dict[str, Any] = {
        "schema": DOSSIER_SCHEMA,
        "purpose": (
            "The verbatim review surface for one sealed composed child under "
            "the digest-pinned frozen V5 protocol.  Grading happens elsewhere; "
            "this file grades nothing."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_id": cell.text,
        "cell_kind": cell.kind,
        "composition_mode": (
            receipt["composition"]["mode"]
            if isinstance(receipt.get("composition"), Mapping)
            else None
        ),
        "composition_variant": (
            receipt["composition"].get("variant")
            if isinstance(receipt.get("composition"), Mapping)
            else None
        ),
        # ADDENDUM 6 grades the whole model with ONE component folded, so the
        # dossier says which component and at what storage, or a reader cannot
        # tell a parent-band cell at 0.25x from a parent-band cell at 3.28x.
        "component": receipt["component"],
        "technique": receipt["technique"],
        "level": receipt["level"],
        "site": receipt["site"],
        "stored_parameter_count": receipt["solve"].get("stored_parameter_count"),
        "parent_parameter_count": receipt["solve"].get("parent_parameter_count"),
        "storage_bound_ratio": receipt["solve"].get("storage_bound_ratio"),
        "storage_bound_satisfied": receipt["solve"].get("storage_bound_satisfied"),
        "token_match_vs_control_diagnostic": receipt["behavior"]["free_run"][
            "token_match_vs_control"
        ],
        "receipt_sha256": str(receipt["receipt_sha"]),
        "receipt_path": str(args.receipt),
        # ADDENDUM 12: a standard-shape child's "folded tensors" are its own
        # checkpoint, so the same two fields carry the directory and its digest
        # and the dossier schema does not move.
        "folded_tensors_path": str(
            args.composed_tensors if cell.kind != "architecture" else args.child_model
        ),
        "composed_child_tensors_sha256": cell_runner._file_sha256(
            Path(args.composed_tensors)
            if cell.kind != "architecture"
            else Path(args.child_model) / "model.safetensors"
        ),
        "standard_shape_child": (
            None
            if cell.kind != "architecture"
            else {
                "child_model": str(args.child_model),
                "net_ratio": receipt["architecture"]["storage"]["net_ratio"],
                "config_diff": receipt["architecture"]["config_diff"],
                "export": receipt["architecture"]["export"],
            }
        ),
        # Present and non-null exactly when the child carries ADDENDUM 10 routes,
        # so a reader can tell a repaired child from an unrepaired one without
        # opening the receipt.
        "corrective_routes": corrections_summary,
        "protocol": {
            "seal_path": str(args.protocol_seal),
            "protocol_sha256": str(seal["protocol_sha256"]),
            "prompt_pack_sha256": str(seal["prompt_pack_sha256"]),
            "dossier_prompt_surface_sha256": surface,
            "dossier_prompt_surface_note": (
                "this script's own digest over the validated rows; it is NOT "
                "the protocol's published normalized surface 5615b9e4..., "
                "which is computed by machinery this script does not "
                "reproduce.  The binding check is the raw byte-form, which "
                "equals the seal's prompt_pack_sha256 exactly."
            ),
            "reference_band": (
                "ADDENDUM 2: the sealed A9 child's 8 of 15 useful-or-partial, "
                "tolerance +/-2 cases (the frontier re-score's own "
                "reviewer-sensitivity interval on this 15-case surface)"
            ),
        },
        "generation": {
            "maximum_new_tokens": int(args.maximum_new_tokens),
            "decoding": "greedy, argmax, no sampling",
            "device": str(args.device),
            "compute_architecture": architecture,
            "dtype": str(args.dtype),
            "cases": len(rows),
            "parent_control_generated": not bool(args.skip_parent),
        },
        "child_responses": child_records,
        "parent_responses": parent_records,
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    body["dossier_sha256"] = canonical_sha256(body)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(body))
    temporary.replace(destination)
    _log(
        f"sealed review surface {body['dossier_sha256'][:16]} -> {destination} "
        f"in {body['wall_time_seconds']:.1f}s"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--composed-tensors",
        type=Path,
        help=(
            "The sealed folded tensors of a parent-shaped child.  Mandatory for "
            "a composition or cell receipt; refused for an ADDENDUM 12 "
            "standard-shape child, which has none."
        ),
    )
    parser.add_argument(
        "--child-model",
        type=Path,
        help=(
            "ADDENDUM 12: the standard-shape child's own checkpoint directory, "
            "loaded by stock transformers with no trust_remote_code.  Mandatory "
            "for an ARCH.<TAG> receipt and meaningless for any other."
        ),
    )
    parser.add_argument(
        "--composed-corrections",
        type=Path,
        help=(
            "ADDENDUM 10: the sealed corrective-route tensors, mandatory for a "
            "child whose receipt carries a composition.corrections block and "
            "meaningless for one that does not."
        ),
    )
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument(
        "--protocol-seal",
        type=Path,
        default=_REPO_ROOT / "reports" / "r12-certified-20260804" / "PROTOCOL_SEAL.json",
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=_REPO_ROOT
        / "research"
        / "lfm2_semantic_whole_response_prompts_v1.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-compute-architecture")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--maximum-new-tokens", type=int, default=1024)
    parser.add_argument("--skip-parent", action="store_true")
    return parser


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
