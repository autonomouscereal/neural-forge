#!/usr/bin/env python3
"""Seal ADDENDUM 12's receipt for one **standard-shape** child.

`NOTES-MX-R` §6 is the reason this file exists.  Wave 14 built the first children
in this campaign that are architecturally smaller rather than smaller-in-storage,
and then could not seal a machine verdict for any of them: a narrowed child
replaces no parent tensor shape-for-shape, carries no ``weight_keys``, and is
therefore not a `CellId` under receipt schema v1 or v2.  Three children were
graded by a human and nothing was appended, which is the correct refusal and a
dead end -- every other child in the campaign carries a sealed verdict, and a
frontier claim that cannot be sealed cannot be compared with the rows it claims
to beat.

ADDENDUM 12 (receipt schema v3) closes that with one new id form, ``ARCH.<TAG>``,
and one new mandatory block.  This script is the runner side of it: it reads a
build receipt written by a wave-15 builder, **loads the child the way ADDENDUM 11
R3 says it must be loadable** -- `Lfm2ForCausalLM.from_pretrained`, no
``trust_remote_code`` -- and measures, in one process, everything the schema's
chain needs:

* the free-run diagnostic against the parent on the sealed probe set (ADDENDUM 8:
  a diagnostic, never a gate, and it rides in ``behavior.free_run`` where the
  chain's sixth link reads it);
* the decode smoke, verbatim, unscored here;
* the terminal-state comparison against **the parent**, which for a
  shape-changing child *is* the copy-exact control: there is no narrower
  copy-exact child to compare to, and the receipt says so in its own notes
  rather than leaving the field null.

**Wave 19 (`hidden_size`).**  One child form breaks that last comparison: a child
whose RESIDUAL is narrower than the parent's has terminal rows of a different
length, and ``candidate - reference`` is not an operation on them.  The honest
repair is not to skip the measurement but to take it where it is defined.  A
``hidden_size`` child is built against one shared orthonormal basis ``E``
(1024 x H), its build receipt names that artifact and its digest, and the child's
terminal row is meant to approximate ``E^T r_parent`` exactly -- so the parent's
row is projected through the child's own declared basis and the comparison is the
same metric on the same object.  If the widths differ and the receipt declares no
basis, or the basis on disk does not digest to what the receipt claims, this
script REFUSES rather than reporting a number it cannot define.  The per-layer
norm ratio needs no repair: it is a ratio of two scalars, and on this axis it
reads as the retained energy it is.

What it refuses:

* a child whose directory digests differently from the build receipt's claim;
* a child stock `transformers` will not load, or loads as the wrong class;
* a parameter count that disagrees with the build receipt's arithmetic;
* a build receipt with no per-axis provenance -- the axes array is mandatory for
  the same reason ``composition.components[]`` is: a shape change nobody can
  attribute is not a localizable result.

Usage::

    python experiments/seal_matrix_architecture_child.py \
        --cell-id ARCH.FF640C \
        --cell-instance-id mx15-ff640c-0001 \
        --child-model /mnt/.../children/MX15-CL640 \
        --build-receipt /mnt/.../children/MX15-CL640/mx15-build-receipt.json \
        --parent-model /home/cereal/model-surgery-lab/models/LFM2.5-230M \
        --campaign-wave 15 --device cuda:0 \
        --out /mnt/.../cells/ARCH.FF640C/cell-receipt.json \
        --ledger-shard /mnt/.../shards/arch-ff640c.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_REPO_ROOT / "src", _REPO_ROOT / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import run_lfm2_matrix_cell as cell_runner  # noqa: E402

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    ARCHITECTURE_MIN_SCHEMA_VERSION,
    CellId,
    MatrixCellReceiptError,
    append_ledger,
    blank_receipt,
    seal_receipt,
    write_receipt,
)

BUILD_RECEIPT_SCHEMAS = (
    "neural-forge-matrix-wave14-narrow-child-v1",
    "neural-forge-matrix-wave15-standard-shape-child-v1",
    "neural-forge-matrix-wave16-standard-shape-child-v1",
    "neural-forge-matrix-wave17-standard-shape-child-v1",
    "neural-forge-matrix-wave18-standard-shape-child-v1",
    "neural-forge-matrix-wave19-standard-shape-child-v1",
    "neural-forge-matrix-wave20-standard-shape-child-v1",
    "neural-forge-matrix-wave21-standard-shape-child-v1",
    "neural-forge-matrix-wave22-standard-shape-child-v1",
    "neural-forge-matrix-wave23-standard-shape-child-v1",
    "neural-forge-matrix-wave24-standard-shape-child-v1",
    "neural-forge-matrix-wave25-standard-shape-child-v1",
    "neural-forge-matrix-wave26-standard-shape-child-v1",
    "neural-forge-matrix-wave27-standard-shape-child-v1",
)


class ArchitectureSealError(RuntimeError):
    """A standard-shape child cannot be sealed as measured.  Always fatal."""


def _residual_basis(build: dict, parent_hidden: int, child_hidden: int) -> Any:
    """The child's own declared basis, verified by digest, or a refusal.

    Only reached when the child's residual is narrower than the parent's.  The
    basis is the build receipt's, not this script's: re-deriving it here would
    measure the child against a basis it was not built with, which is exactly
    the mistake this comparison exists to avoid.
    """

    declared = (build.get("residual_basis") or {})
    path = declared.get("basis_path")
    digest = declared.get("basis_sha256")
    if not path or not digest:
        raise ArchitectureSealError(
            f"the child's residual is {child_hidden} wide and the parent's is "
            f"{parent_hidden}; the terminal comparison needs the basis the child "
            "was built against and the build receipt declares none."
        )
    basis_path = Path(path)
    if not basis_path.is_file():
        raise ArchitectureSealError(f"{basis_path} (the declared basis) is missing.")
    measured = cell_runner._file_sha256(basis_path)
    if measured != digest:
        raise ArchitectureSealError(
            f"{basis_path} digests {measured}; the build receipt claims {digest}."
        )
    from safetensors import safe_open

    with safe_open(str(basis_path), "pt") as handle:
        basis = handle.get_tensor("basis").to(torch.float64)
    if tuple(basis.shape) != (parent_hidden, child_hidden):
        raise ArchitectureSealError(
            f"{basis_path} is {tuple(basis.shape)}; this child needs "
            f"({parent_hidden}, {child_hidden})."
        )
    orthonormality = float(
        (basis.T @ basis - torch.eye(child_hidden, dtype=torch.float64)).abs().max()
    )
    if orthonormality > 1e-8:
        raise ArchitectureSealError(
            f"{basis_path} is not orthonormal (max off-identity {orthonormality:.3e})."
        )
    _log(
        f"residual basis {basis_path.name} verified: {tuple(basis.shape)}, "
        f"orthonormal to {orthonormality:.2e}"
    )
    return basis


def _log(message: str) -> None:
    print(f"[arch] {message}", flush=True)


def _child_config_diff(parent_config: dict, child_config: dict) -> list[dict[str, Any]]:
    """Every config field whose value differs, both sides, sorted by field name.

    Recomputed from the two files rather than trusted from the build receipt:
    the config diff *is* the child under R3, so a builder that mis-stated it
    would be mis-stating the whole claim.
    """

    fields = sorted(set(parent_config) | set(child_config))
    diff: list[dict[str, Any]] = []
    for field in fields:
        left = parent_config.get(field)
        right = child_config.get(field)
        if left != right:
            diff.append({"field": field, "parent": left, "child": right})
    return diff


@torch.no_grad()
def _free_run(model: Any, tokenizer: Any, prompts, *, tokens: int, device: str) -> list[list[int]]:
    """Greedy continuation ids for each probe prompt, the runner's own way."""

    out: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated: list[int] = []
        past = None
        current = ids
        for _step in range(int(tokens)):
            result = model(input_ids=current, past_key_values=past, use_cache=True)
            past = result.past_key_values
            next_id = int(result.logits[0, -1].argmax().item())
            generated.append(next_id)
            current = torch.tensor([[next_id]], device=device, dtype=ids.dtype)
        out.append(generated)
    return out


def run(args: argparse.Namespace) -> int:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    started = time.monotonic()

    child_dir = Path(args.child_model)
    parent_dir = Path(args.parent_model)
    build = json.loads(Path(args.build_receipt).read_text(encoding="utf-8"))
    if build.get("schema") not in BUILD_RECEIPT_SCHEMAS:
        raise ArchitectureSealError(
            f"{args.build_receipt} declares schema {build.get('schema')!r}, "
            f"which is not one of {BUILD_RECEIPT_SCHEMAS}."
        )

    cell = CellId.parse(
        str(args.cell_id), schema_version=ARCHITECTURE_MIN_SCHEMA_VERSION
    )
    if cell.kind != "architecture":
        raise ArchitectureSealError(
            f"{cell.text} is a {cell.kind} cell; this script seals ARCH.<TAG> only."
        )

    architecture = cell_runner.compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise ArchitectureSealError(
            f"--device {args.device} resolves to {architecture!r}, not "
            f"{args.require_compute_architecture!r}."
        )

    probe_prompts, probe_sha = cell_runner._load_prompt_set(
        Path(args.probe_prompts), "free-run-probe"
    )
    smoke_prompts, smoke_sha = cell_runner._load_prompt_set(
        Path(args.smoke_prompts), "decode-smoke"
    )

    # ---------------------------------------------------------------- export
    # R3's claim, executed rather than asserted.  The digest is taken first so
    # the file that loads is provably the file the build receipt was written for.
    tensors_path = child_dir / "model.safetensors"
    child_tensors_sha = cell_runner._file_sha256(tensors_path)
    declared_tensors_sha = build.get("child_tensors_sha256")
    if declared_tensors_sha is not None and str(declared_tensors_sha) != child_tensors_sha:
        raise ArchitectureSealError(
            f"{tensors_path} digests {child_tensors_sha}; the build receipt "
            f"claims {declared_tensors_sha}."
        )
    from transformers import AutoModelForCausalLM, Lfm2ForCausalLM

    child = Lfm2ForCausalLM.from_pretrained(str(child_dir), dtype=torch.float32)
    live_parameters = int(sum(p.numel() for p in child.parameters()))
    loaded_class = type(child).__name__
    # The same load through the generic entry point, because "stock transformers
    # loads it" is a claim about the ordinary path and not about the private one.
    auto = AutoModelForCausalLM.from_pretrained(str(child_dir), dtype=torch.float32)
    if type(auto).__name__ != loaded_class:
        raise ArchitectureSealError(
            f"AutoModelForCausalLM returned {type(auto).__name__} and "
            f"Lfm2ForCausalLM returned {loaded_class}."
        )
    del auto
    if live_parameters != int(build["child_parameters"]):
        raise ArchitectureSealError(
            f"The loader counted {live_parameters} parameters; the build receipt "
            f"claims {build['child_parameters']}."
        )
    _log(f"stock load OK: {loaded_class}, {live_parameters:,} parameters")
    child = child.to(str(args.device)).eval()

    tokenizer = cell_runner._load_tokenizer(parent_dir)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # ADDENDUM 15 section 15.3 (wave 23, the `vocab_size` axis).  A child whose
    # vocabulary is a dense remap of a subset of the parent's does not read the
    # parent's ids -- the same id names a different token in the two tables -- so
    # it gets its own tokenizer and the parent keeps the parent's.  The gate is a
    # gate rather than a preference: the flag is refused when the vocabularies
    # already agree, and its absence is refused when they do not, so neither
    # direction can be reached by forgetting an argument.
    parent_vocab_size = int(
        json.loads((parent_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    )
    child_vocab_size = int(
        json.loads((child_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    )
    child_tokenizer = tokenizer
    if child_vocab_size != parent_vocab_size:
        if args.child_tokenizer is None:
            raise ArchitectureSealError(
                f"the child declares vocab_size {child_vocab_size} against the "
                f"parent's {parent_vocab_size} and no --child-tokenizer was "
                "given; the parent's ids do not name the child's tokens."
            )
        child_tokenizer = cell_runner._load_tokenizer(Path(args.child_tokenizer))
        if int(len(child_tokenizer)) > child_vocab_size:
            raise ArchitectureSealError(
                f"--child-tokenizer carries {len(child_tokenizer)} tokens and the "
                f"child's table has {child_vocab_size} rows."
            )
        _log(
            f"child tokenizer: {args.child_tokenizer} "
            f"({len(child_tokenizer)} tokens, child table {child_vocab_size})"
        )
    elif args.child_tokenizer is not None:
        raise ArchitectureSealError(
            "--child-tokenizer was given for a child whose vocab_size equals the "
            "parent's; there is nothing for it to repair and it can only "
            "introduce a difference the receipt would not name."
        )
    child_eos_id = getattr(child_tokenizer, "eos_token_id", None)

    child_ids = _free_run(
        child,
        child_tokenizer,
        probe_prompts,
        tokens=int(args.probe_tokens),
        device=str(args.device),
    )
    smoke = cell_runner._decode_smoke(
        child,
        child_tokenizer,
        smoke_prompts,
        cap=128,
        device=str(args.device),
        eos_id=child_eos_id,
    )
    child_terminal, child_norms = cell_runner._terminal_states(
        child, child_tokenizer, probe_prompts, device=str(args.device)
    )
    del child
    torch.cuda.empty_cache()

    parent = cell_runner._load_parent(parent_dir, str(args.device), torch.float32)
    parent_parameters = int(sum(p.numel() for p in parent.parameters()))
    parent_ids = _free_run(
        parent, tokenizer, probe_prompts, tokens=int(args.probe_tokens), device=str(args.device)
    )
    parent_terminal, parent_norms = cell_runner._terminal_states(
        parent, tokenizer, probe_prompts, device=str(args.device)
    )
    del parent
    torch.cuda.empty_cache()

    if parent_parameters != int(build["parent_parameters"]):
        raise ArchitectureSealError(
            f"The parent loaded with {parent_parameters} parameters; the build "
            f"receipt claims {build['parent_parameters']}."
        )

    # The free-run token match is [DIAGNOSTIC ONLY] and has been since ADDENDUM
    # 6 retired it as a gate.  Across two vocabularies it is not merely
    # uninformative, it is undefined -- id 4,211 is not the same token in both
    # tables -- so for a vocab child it is taken where it IS defined: the child's
    # continuation is decoded through the child's tokenizer and re-encoded with
    # the PARENT's, which compares the two models on the strings they actually
    # emitted.  The receipt says which of the two definitions produced the
    # number, and it is still never a gate.
    token_match_basis = "parent-token-ids, both models"
    if child_tokenizer is not tokenizer:
        token_match_basis = (
            "the child's continuation DECODED with the child tokenizer and "
            "RE-ENCODED with the parent's, compared against the parent's own ids "
            "-- a string-level comparison, because the two id spaces are "
            "different tables"
        )
        child_ids = [
            list(
                tokenizer(
                    child_tokenizer.decode(row, skip_special_tokens=False),
                    add_special_tokens=False,
                ).input_ids
            )
            for row in child_ids
        ]
    match, lcp = cell_runner._token_match(child_ids, parent_ids)
    parent_hidden = int(parent_terminal.shape[-1])
    child_hidden = int(child_terminal.shape[-1])
    terminal_basis = None
    if child_hidden != parent_hidden:
        terminal_basis = _residual_basis(build, parent_hidden, child_hidden)
        terminal = cell_runner._row_metrics(
            child_terminal, parent_terminal.to(torch.float64) @ terminal_basis
        )
    else:
        terminal = cell_runner._row_metrics(child_terminal, parent_terminal)
    terminal["reference"] = (
        "the parent's own terminal rows"
        if terminal_basis is None
        else "the parent's terminal rows PROJECTED through the child's declared basis"
    )
    # The per-layer norm ratio needs a layer correspondence, and a child that
    # excised blocks does not have the parent's depth.  The build receipt knows
    # exactly which parent blocks survived, so the ratio is taken against those
    # and the receipt says which ones; a child of a different depth compared
    # position-by-position would be comparing two different layers and calling
    # the difference drift.
    excised = [int(x) for x in build.get("excised", [])]
    surviving = [l for l in range(parent_norms.shape[1] - 1) if l not in excised]
    if child_norms.shape[1] != len(surviving) + 1:
        raise ArchitectureSealError(
            f"the child has {child_norms.shape[1] - 1} blocks and the build "
            f"receipt says {len(surviving)} survived."
        )
    parent_columns = [0] + [l + 1 for l in surviving]
    ratios = (
        child_norms / parent_norms[:, parent_columns].clamp_min(1e-30)
    ).reshape(-1)
    _log(
        f"free-run token match {match:.4f} (diagnostic), terminal relL2 "
        f"{terminal['mean_relative_l2']:.4f} cos {terminal['mean_cosine']:.4f}"
    )

    # ---------------------------------------------------------------- receipt
    parent_config = json.loads((parent_dir / "config.json").read_text(encoding="utf-8"))
    child_config = json.loads((child_dir / "config.json").read_text(encoding="utf-8"))
    diff = _child_config_diff(parent_config, child_config)
    if not diff:
        raise ArchitectureSealError(
            "The child's config equals the parent's; nothing about this child is "
            "a shape change and it is not an ARCH cell."
        )

    receipt = blank_receipt(
        cell_id=cell.text,
        cell_instance_id=str(args.cell_instance_id),
        campaign_wave=int(args.campaign_wave),
        repo_head=str(args.repo_head or cell_runner._repo_head(_REPO_ROOT)),
        src_tree_sha=cell_runner._src_tree_sha256(_REPO_ROOT),
        probe_set_sha=probe_sha,
        smoke_prompt_sha=smoke_sha,
        compute_architecture=architecture,
        schema_version=ARCHITECTURE_MIN_SCHEMA_VERSION,
    )
    receipt["evidence"].update(
        {
            "parent_trace_cache_sha": build.get("evidence", {}).get("trace_cache_key"),
            "heldout_split": {
                "fit_rows": int(build.get("evidence", {}).get("fit_rows", 0) or 0),
                "heldout_rows": int(build.get("evidence", {}).get("heldout_rows", 0) or 0),
            },
            "parent_model_forward_calls": int(
                build.get("evidence", {}).get("parent_model_forward_calls", 0) or 0
            ),
            "split_authority": build.get("evidence", {}).get("split_authority"),
            "conditioned": bool(build.get("evidence", {}).get("conditioned", False)),
        }
    )
    receipt["calibration"] = {
        "copy_exact_reproduction_relL2": None,
        "passed": False,
        "applicable": False,
        "inapplicable_reason": (
            "A standard-shape child has no copy-exact reproduction gate: the "
            "gate multiplies held-out input rows by ONE parent weight and "
            "compares them to that tensor's own output, and this child replaces "
            "no parent tensor shape-for-shape.  The parent itself is the only "
            "copy-exact control a shape change has, and it is used as such in "
            "the composed block below."
        ),
    }
    receipt["solve"].update(
        {
            "solver_path": str(build.get("solver_path", "")) or None,
            "ridge_relative": build.get("ridge_relative"),
            "condition_number": build.get("condition_number"),
            "condition_cap_saturated": bool(build.get("condition_cap_saturated", False)),
            "prior_norm": build.get("prior_norm"),
            "correction_norm": build.get("correction_norm"),
            "requested_clamp_ratio": build.get("requested_clamp_ratio", 0.0),
            "allowed_clamp_ratio": build.get("allowed_clamp_ratio", 0.0),
            "clamp_engaged": bool(build.get("clamp_engaged", False)),
            "stored_parameter_count": live_parameters,
            "parent_parameter_count": parent_parameters,
        }
    )
    worst = build.get("worst_site")
    receipt["boundary"] = {
        "applicable": bool(worst is not None),
        "local_solve_applicable": bool(worst is not None),
        "macrocell_endpoint": None if worst is None else str(worst.get("endpoint")),
        "write_heldout_relL2": None if worst is None else float(worst["relL2"]),
        "write_heldout_cos": None if worst is None else float(worst["cos"]),
        "post_operator_heldout_relL2": None if worst is None else float(worst["relL2"]),
        "post_operator_heldout_cos": None if worst is None else float(worst["cos"]),
        "endpoint_band_le_0p4": (
            None if worst is None else bool(float(worst["relL2"]) <= 0.4)
        ),
    }
    receipt["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": terminal["mean_relative_l2"],
        "emit_zero_floor_passed": bool(terminal["mean_relative_l2"] < 1.0),
        "vs_copy_exact_control_delta": terminal["mean_relative_l2"],
        "norm_ratio_per_layer_minmax": [
            float(ratios.min().item()),
            float(ratios.max().item()),
        ],
        "norm_ratio_parent_blocks": surviving,
    }
    receipt["behavior"] = {
        "free_run": {
            "probe_prompts": len(probe_prompts),
            "tokens_per_prompt": int(args.probe_tokens),
            "token_match_vs_control": match,
            "mean_lcp": lcp,
            "token_match_basis": token_match_basis,
            "child_tokenizer": (
                None if child_tokenizer is tokenizer else str(args.child_tokenizer)
            ),
        },
        "decode_smoke": smoke,
    }
    receipt["architecture"] = {
        "schema": "neural-forge-lfm2-matrix-addendum12-standard-shape-child-v1",
        "standard_shape": True,
        "tag": cell.architecture_tag,
        "rung": cell.architecture_rung,
        "child_directory": str(child_dir),
        "child_tensors_sha256": child_tensors_sha,
        "build_receipt_sha256": cell_runner._file_sha256(Path(args.build_receipt)),
        "config_diff": diff,
        "export": {
            "loader": "transformers.Lfm2ForCausalLM.from_pretrained",
            "also_loaded_by": "transformers.AutoModelForCausalLM.from_pretrained",
            "trust_remote_code": False,
            "loaded_class": loaded_class,
            "parameters_counted_live": live_parameters,
            "transformers_version": __import__("transformers").__version__,
            "statement": str(build.get("export_path", "")),
        },
        "storage": {
            "parent_parameter_count": parent_parameters,
            "child_parameter_count": live_parameters,
            "net_ratio": live_parameters / parent_parameters,
        },
        "axes": list(build["axes"]),
        "passed_through": list(build.get("passed_through", [])),
        "terminal_vs_parent": terminal,
    }
    receipt["outcome"]["notes"] = (
        "ADDENDUM 12 standard-shape child.  The composed block compares this "
        "child's terminal states to THE PARENT's, because a shape-changing child "
        "has no copy-exact control child; the free-run token match is the "
        "ADDENDUM 8 diagnostic and gates nothing.  The grade is the frozen "
        "15-prompt review, sealed separately."
    )
    receipt["wall_time_seconds"] = round(time.monotonic() - started, 3)

    sealed = seal_receipt(receipt)
    destination = Path(args.out)
    write_receipt(destination, sealed)
    _log(
        f"sealed {sealed['cell_id']} {sealed['receipt_sha'][:12]} "
        f"net {live_parameters / parent_parameters:.6f}x status "
        f"{sealed['outcome']['status']} first-divergent "
        f"{sealed['outcome']['first_divergent_metric']}"
    )
    _log(f"receipt -> {destination}")
    if args.ledger_shard is not None:
        append_ledger(Path(args.ledger_shard), sealed)
        _log(f"ledger shard -> {args.ledger_shard}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--cell-instance-id", required=True)
    parser.add_argument("--child-model", type=Path, required=True)
    parser.add_argument(
        "--child-tokenizer",
        type=Path,
        help=(
            "the child's OWN tokenizer directory.  Required for, and only for, a "
            "child whose `vocab_size` differs from the parent's: its ids are a "
            "dense remap of a subset of the parent's and feeding it the parent's "
            "ids would be feeding it a different sentence.  Omitted, this script "
            "behaves exactly as it has since wave 15 -- one tokenizer, the "
            "parent's, for both models -- so no sealed child's numbers move."
        ),
    )
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--campaign-wave", type=int, required=True)
    parser.add_argument("--repo-head")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-compute-architecture")
    parser.add_argument(
        "--probe-prompts", type=Path, default=cell_runner.PROBE_PROMPTS_FILE
    )
    parser.add_argument(
        "--smoke-prompts", type=Path, default=cell_runner.SMOKE_PROMPTS_FILE
    )
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger-shard", type=Path)
    return parser


def main() -> int:
    try:
        return run(_parser().parse_args())
    except (ArchitectureSealError, MatrixCellReceiptError) as exc:
        print(f"[arch] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
