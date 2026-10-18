"""One child, one transaction, one receipt.  The CLI every queue calls.

Usage is one-axis-at-a-time BY CONVENTION for the ladder -- a child built by two
folds cannot be attributed by a review, and the campaign's own record says the
second structural step is where things die.  Pairs are built by running this
twice, the second time against the first child as its parent, which is also what
makes the lineage in the receipt a real chain.

What is CONVENTION and what is ENFORCED are different things and this file says
which is which:

* structural axes (C2/C3/C4/C5/C6/C7/C9) may be combined in one call -- the
  self-test's end-to-end child is an excision plus an MLP fold -- and each one
  is recorded separately in ``axes``;
* ``--vocab-keep`` may NOT be combined with anything, and that IS enforced
  (:func:`refuse_illegal_combinations`).  A vocabulary fold renames every
  embedding row, so the sealed id stream stops being a stream of THIS model's
  tokens the moment it lands: a later capture either dies on an out-of-range id
  or, worse, silently solves every site against rows belonging to different
  tokens than the manifest names.  The child's remapped manifest is written
  beside the child so the next build in the chain has a legal evidence file;
* ``--carry-mtp`` is refused when any axis moved a shape the MTP block reads;
* the sealed evidence must have been tokenized by THIS parent's tokenizer, and
  that IS enforced (:func:`assert_evidence_tokenizer`).  ``qf_evidence_build.py``
  pins the tokenizer it builds against so a swapped tokenizer cannot change every
  id silently -- the check was missing one step downstream, where it matters:
  the ids in a manifest are legal for any parent with a wide enough table, so
  ``--parent Qwen3.8-27B-Uncensored --evidence omni-evidence-v1.json`` ran to
  completion with the seal verified, every ridge fitted on activations from a
  stream that parent decodes as different text, and a receipt naming the
  manifest.  ``--allow-tokenizer-mismatch`` is the recorded escape.

    qf_build.py --parent DIR --out DIR --evidence manifest.json --tag C2S4L03 \\
                --mlp-keep 2304

Every ``--*-keep``/``--*-heads`` number is a rung, not a threshold.  Every
number the receipt prints is a proposed decision line for Cyril.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qf_arch
import qf_attn
import qf_corpus
import qf_excise
import qf_export
import qf_gdn
import qf_mlp
import qf_termgain
import qf_vocab
from qf_harness import Harness, harness_parity

LOG_PREFIX = "[qf]"
RECEIPT_NAME = "qf-build-receipt.json"


def log(message: str) -> None:
    print(f"{message}" if message.startswith(LOG_PREFIX) else f"{LOG_PREFIX} {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qwen3_5 single-transaction fold builder")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", default="fold", choices=("fold", "distill", "finetune"))
    parser.add_argument("--parent-code", default="")
    parser.add_argument("--lineage", default="", help="comma-separated ancestor digests")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32", "float16"))

    parser.add_argument("--vocab-keep", type=int, default=None, help="C1")
    parser.add_argument("--mlp-keep", type=int, default=None, help="C2")
    parser.add_argument("--mlp-group", type=int, default=3)
    parser.add_argument("--mlp-select", default="evidence", choices=("evidence", "reader-metric"))
    parser.add_argument("--excise", default="", help="C3/C4: comma-separated layer indices")
    parser.add_argument("--gdn-value-heads", type=int, default=None, help="C5 arm A")
    parser.add_argument("--gdn-key-heads", type=int, default=None, help="C5 arm A")
    parser.add_argument("--gdn-key-head-dim", type=int, default=None, help="C5 arm B")
    parser.add_argument("--gdn-value-head-dim", type=int, default=None, help="C5 arm B")
    parser.add_argument("--gdn-pairing", default="index", choices=("index", "census"))
    parser.add_argument("--gdn-drop-arm", action="store_true")
    parser.add_argument("--nope-keep", type=int, default=None, help="C6")
    parser.add_argument("--kv-heads", type=int, default=None, help="C7")
    parser.add_argument("--kv-arm", default="kv-only", choices=("kv-only", "full-group"))
    parser.add_argument("--terminal-gain", action="store_true", help="C9")

    parser.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        help="build against evidence tokenized by a DIFFERENT tokenizer (recorded)",
    )
    parser.add_argument("--no-lambda-split", action="store_true")
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument("--carry-mtp", action="store_true")
    parser.add_argument("--drop-mtp", action="store_true")
    parser.add_argument("--gguf", default="", help="llama.cpp checkout; empty = command only")
    parser.add_argument("--gguf-out", default="")
    parser.add_argument("--review-script", default="")
    parser.add_argument("--review-out", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}

#: Config fields whose values determine the SHAPES of the tensors an MTP block
#: reads.  Carrying `mtp.*` through verbatim is only legal while every one of
#: them is still the parent's; the rule used to live in a comment that said "the
#: caller decides", and the caller is this file.
MTP_SHAPE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
)


def structural_axis_flags(args: argparse.Namespace) -> dict[str, bool]:
    """Which non-vocabulary axes this invocation asked for."""

    return {
        "--mlp-keep": args.mlp_keep is not None,
        "--excise": bool(args.excise),
        "--gdn-*": any(
            v is not None
            for v in (
                args.gdn_value_heads, args.gdn_key_heads,
                args.gdn_key_head_dim, args.gdn_value_head_dim,
            )
        ),
        "--nope-keep": args.nope_keep is not None,
        "--kv-heads": args.kv_heads is not None,
        "--terminal-gain": bool(args.terminal_gain),
    }


def refuse_illegal_combinations(args: argparse.Namespace) -> None:
    """Refuse before anything is loaded, let alone mutated.

    The vocabulary rule is the one that matters: after ``vocab_axis`` the live
    model's rows are renumbered and the sealed manifest is not.  Every capture
    after it would be taken on a stream this model cannot read -- an index error
    if the stream contains a dropped id (a device-side assert on CUDA, i.e. a
    dead process, AFTER the vocabulary transaction has already mutated the
    model), and silence if it does not, which is worse: every downstream site
    would be solved against rows belonging to different tokens than the manifest
    names, under a receipt that says the evidence was the manifest.
    """

    if args.carry_mtp and args.drop_mtp:
        raise SystemExit("--carry-mtp and --drop-mtp are mutually exclusive")
    if args.vocab_keep is None:
        return
    also = sorted(flag for flag, asked in structural_axis_flags(args).items() if asked)
    if also:
        raise SystemExit(
            f"--vocab-keep cannot be combined with {', '.join(also)}: the fold "
            f"renames every embedding row and the sealed evidence stream is in the "
            f"PARENT's id space, so the next capture would be solved against the "
            f"wrong tokens.  Build the vocabulary child alone; it writes "
            f"evidence-remapped.json beside itself, and the next build uses that "
            f"child as --parent and that manifest as --evidence."
        )


def assert_evidence_tokenizer(
    manifest: dict[str, Any], parent_dir: str | Path, *, allow_mismatch: bool
) -> dict[str, Any]:
    """The stream must be in THIS parent's id space.  Refuses when it is not.

    The manifest names the sha256 of the tokenizer.json its ids came out of.  A
    parent whose tokenizer.json hashes differently spells those ids as different
    tokens: nothing fails, no id is out of range, the seal still verifies, and
    every solve in the build is fitted on activations taken over text the parent
    never wrote.  On this box that is not hypothetical -- the Qwen3.5 0.8B and 4B
    share one tokenizer and the 27B does not.

    A C1 child is the legal exception in the other direction: its tokenizer is
    rewritten by the fold, and ``qf_vocab.remap_manifest`` seals the remapped
    stream under the CHILD's tokenizer sha, so this check passes on the chained
    build without an escape.
    """

    declared = str(manifest.get("tokenizer_sha256", ""))
    path = Path(parent_dir) / "tokenizer.json"
    measured = qf_corpus.file_sha256(path) if path.is_file() else None
    matches = measured is not None and declared == measured
    report = {
        "manifest_tokenizer_sha256": declared,
        "parent_tokenizer_json": str(path) if path.is_file() else None,
        "parent_tokenizer_sha256": measured,
        "matches": matches,
        "allow_mismatch": bool(allow_mismatch),
        "statement": (
            "the sealed stream was tokenized by the tokenizer this parent carries"
            if matches
            else "the sealed stream was NOT tokenized by this parent's tokenizer"
        ),
    }
    if matches or allow_mismatch:
        return report
    raise SystemExit(
        f"the evidence was tokenized by {declared or '(no sha recorded)'} and "
        f"{path} is {measured or '(missing)'}.  Every id in the manifest means a "
        f"different token to this parent, and nothing downstream would notice: "
        f"the seal verifies, the ids are in range, and the child receipt would "
        f"name the manifest.  Build the evidence against this parent "
        f"(qf_evidence_build.py --parent {parent_dir}) or pass "
        f"--allow-tokenizer-mismatch to have the exception recorded in the receipt."
    )


def assert_mtp_carry_legal(parent_config: Any, child_config: Any) -> list[str]:
    """The shapes an MTP block reads must still be the parent's.  Returns them."""

    parent_tc = qf_arch.text_config(parent_config)
    child_tc = qf_arch.text_config(child_config)
    moved = [
        f"{field} {getattr(parent_tc, field, None)} -> {getattr(child_tc, field, None)}"
        for field in MTP_SHAPE_FIELDS
        if getattr(parent_tc, field, None) != getattr(child_tc, field, None)
    ]
    if moved:
        raise SystemExit(
            "--carry-mtp is not legal for this child: the mtp.* tensors are copied "
            "verbatim and these shapes moved -- " + "; ".join(moved) + ".  Carry-"
            "through is only for axes that change no shape the MTP block reads "
            "(layer excision, terminal gain); otherwise drop the block."
        )
    return list(MTP_SHAPE_FIELDS)


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    refuse_illegal_combinations(args)
    dtype = DTYPES[args.dtype]
    manifest = qf_corpus.read_manifest(args.evidence)
    # BEFORE the parent is loaded and before a single capture: an evidence file
    # from another tokenizer is a silent substitution, not a tolerance question.
    evidence_tokenizer = assert_evidence_tokenizer(
        manifest, args.parent, allow_mismatch=args.allow_tokenizer_mismatch
    )
    parent_config = qf_arch.load_config(args.parent)
    parent_config_snapshot = copy.deepcopy(parent_config)

    log(f"parent {args.parent}")
    harness = Harness(args.parent, args.device, dtype, manifest)
    parent_counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    log(f"parent parameters {parent_counts['live']:,} (closed form agrees)")

    receipts: dict[str, Any] = {
        "parity": None,
        "causal_masks": None,
        "evidence_tokenizer": evidence_tokenizer,
    }
    log(
        f"evidence tokenizer {evidence_tokenizer['manifest_tokenizer_sha256'][:16]} "
        f"vs parent {str(evidence_tokenizer['parent_tokenizer_sha256'])[:16]}: "
        f"{'match' if evidence_tokenizer['matches'] else 'MISMATCH (allowed)'}"
    )
    if not args.no_parity:
        receipts["parity"] = harness_parity(harness)
        log(
            f"parity: deterministic, hook placement worst relL2 "
            f"{receipts['parity']['hook_placement_worst']:.3e}, logits bit-identical"
        )

    axes: list[dict[str, Any]] = []
    lambda_split = not args.no_lambda_split

    # C1 first: the vocabulary fold renames rows, and every capture after it is
    # on the child's own table.
    vocab_report: dict[str, Any] | None = None
    if args.vocab_keep is not None:
        vocab_report = qf_vocab.vocab_axis(
            harness, keep=args.vocab_keep,
            tokenizer_path=Path(args.parent) / "tokenizer.json", log=log,
        )
        # `tokenizer_body` is the child's whole tokenizer and `kept_ids` is the
        # keep set; the first belongs in tokenizer.json and the second in the
        # axis, and neither belongs a SECOND time in the preconditions block.
        axes.append({k: v for k, v in vocab_report.items() if k != "tokenizer_body"})
        receipts["vocab_report"] = {
            k: v for k, v in vocab_report.items()
            if k not in ("tokenizer_body", "kept_ids")
        }

    if args.excise:
        excised = [int(v) for v in args.excise.split(",") if v.strip()]
        report = qf_excise.excise_live(harness.model, excised)
        report["parameters_removed"] = qf_excise.parameters_removed(
            qf_arch.text_config(parent_config_snapshot), report["excised_kinds"]
        )
        # Before ANY capture on an excised model: the mask dispatch is
        # positional and this is the only cheap check that sees it.
        receipts["causal_masks"] = qf_excise.assert_causal_masks(harness)
        axes.append(report)
        log(
            f"C3/C4 excised {report['excised']} "
            f"({report['full_layers_removed']} full, {report['linear_layers_removed']} linear); "
            f"above-diagonal mass 0 on every surviving attention layer"
        )

    if args.mlp_keep is not None:
        metric = None
        if args.mlp_select == "reader-metric":
            metric, census = qf_mlp.reader_metric(harness, positions=2048, topk=64)
            receipts["reader_metric"] = census
        axes.append(
            qf_mlp.mlp_axis(
                harness, args.mlp_keep, group_size=args.mlp_group, metric=metric,
                lambda_split=lambda_split, dtype=dtype, log=log,
            )
        )

    if args.gdn_value_heads or args.gdn_key_heads or args.gdn_key_head_dim or args.gdn_value_head_dim:
        axes.append(
            qf_gdn.gdn_axis(
                harness,
                value_heads=args.gdn_value_heads, key_heads=args.gdn_key_heads,
                key_head_dim=args.gdn_key_head_dim, value_head_dim=args.gdn_value_head_dim,
                pairing=args.gdn_pairing, drop_arm=args.gdn_drop_arm,
                lambda_split=lambda_split, dtype=dtype, log=log,
            )
        )

    if args.nope_keep is not None:
        axes.append(
            qf_attn.nope_dim_contraction(
                harness, keep_nope=args.nope_keep, lambda_split=lambda_split,
                dtype=dtype, log=log,
            )
        )

    if args.kv_heads is not None:
        axes.append(
            qf_attn.kv_head_merge(
                harness, kv_heads=args.kv_heads, arm=args.kv_arm,
                lambda_split=lambda_split, dtype=dtype, log=log,
            )
        )

    if args.terminal_gain:
        parent_harness = Harness(args.parent, args.device, dtype, manifest)
        axes.append(
            qf_termgain.terminal_gain(harness, parent_harness, which="final",
                                      lambda_split=lambda_split, log=log)
        )
        del parent_harness

    if not axes:
        raise SystemExit("no axis requested; this would build a copy, not a child")

    # The live model must agree with its own config before it is written, and
    # the closed-form count must agree with the live one after every axis.
    qf_arch.assert_layer_types_agree(harness.model)
    child_counts = qf_arch.assert_parameter_count(harness.model, harness.model.config)
    log(f"child parameters {child_counts['live']:,} (closed form agrees)")
    if args.carry_mtp:
        # Checked against the CONFIGS, not against the flags: an axis that moved
        # a shape without being asked for by name is caught here too.
        receipts["mtp_carry_legality"] = {
            "checked_fields": assert_mtp_carry_legal(
                parent_config_snapshot, harness.model.config
            ),
            "statement": "every shape the mtp block reads is still the parent's",
        }

    export = qf_export.export_child(
        harness.model, args.out, parent_dir=args.parent, sidecars=True
    )
    if vocab_report is not None:
        written = qf_vocab.write_tokenizer(vocab_report, args.out, args.parent)
        export["sidecars_copied"] = sorted(set(export["sidecars_copied"]) | set(written))
        export["tokenizer"] = "remapped by C1"
        # The evidence for THIS child's descendants, in this child's id space.
        # Written rather than assumed, and a refusal is recorded rather than
        # raised: the child on disk is already exported and verified, and the
        # honest answer to an unspellable stream is to re-tokenize the source
        # text, which is a corpus job and not this build's.
        remapped_path = Path(args.out) / "evidence-remapped.json"
        try:
            remapped = qf_vocab.remap_manifest(
                manifest,
                vocab_report["kept_ids"],
                remapped_path,
                # The child's tokenizer.json has just been written; sealing the
                # remapped stream under ITS sha is what lets the next build's
                # tokenizer check pass on the truth instead of on an exception.
                child_tokenizer_sha256=qf_corpus.file_sha256(
                    Path(args.out) / "tokenizer.json"
                ),
            )
            receipts["evidence_for_descendants"] = {
                "path": str(remapped_path),
                "manifest_sha256": remapped["manifest_sha256"],
                "statement": (
                    "the sealed stream renamed into this child's id space; any build "
                    "that uses this child as --parent must use this as --evidence"
                ),
            }
            log(f"remapped evidence written to {remapped_path}")
        except SystemExit as error:
            receipts["evidence_for_descendants"] = {
                "path": None,
                "refused": str(error),
                "statement": (
                    "the parent's stream cannot be renamed into this child's id "
                    "space; re-tokenize the source text with the child's tokenizer"
                ),
            }
            log(f"remapped evidence REFUSED: {error}")

    mtp = {"mtp": "not present in the parent index"}
    if args.carry_mtp:
        mtp = qf_export.carry_through(args.parent, args.out)
        mtp["mtp"] = "carried verbatim"
        # The digest must be recomputed: a carried shard is part of the child.
        digest, files = qf_export.directory_digest(args.out)
        export["child_tensors_sha256"] = digest
        export["weight_files"] = files
        export["tensor_sha256"] = qf_export.tensor_digests(args.out)
    elif args.drop_mtp:
        mtp = {"mtp": "dropped", "note": "mtp_num_hidden_layers set to 0 in the child config"}
        config_path = Path(args.out) / "config.json"
        body = json.loads(config_path.read_text(encoding="utf-8"))
        body["mtp_num_hidden_layers"] = 0
        config_path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    verify, reloaded = qf_export.verify_child(args.out, device="cpu", dtype=torch.float32)
    log(
        f"verified: {verify['loaded_class']} re-loaded from disk, "
        f"{verify['parameters_counted_live']:,} parameters, "
        f"digest {verify['child_tensors_sha256'][:16]}"
    )
    if verify["child_tensors_sha256"] != export["child_tensors_sha256"]:
        raise SystemExit("the child changed between export and verification")

    gguf = qf_export.convert_to_gguf(
        args.out,
        args.gguf_out or (Path(args.out).parent / f"{Path(args.out).name}.gguf"),
        llama_cpp=args.gguf or "/home/cereal/llama.cpp",
        dry_run=not args.gguf,
    )
    review = qf_export.frozen_review(
        review_script=args.review_script or "run_omni_frozen_review.py",
        receipt=Path(args.out) / RECEIPT_NAME,
        child_dir=args.out,
        parent_dir=args.parent,
        out=args.review_out or (Path(args.out) / "frozen-review-dossier.json"),
        device=args.device,
        dtype="bfloat16" if args.dtype == "bfloat16" else args.dtype,
        dry_run=not args.review_script,
    )

    receipt = qf_export.shape_receipt(
        parent_config=parent_config_snapshot,
        child_config=reloaded.config,
        parent_dir=args.parent,
        export=export,
        verify=verify,
        axes=axes,
        evidence=qf_corpus.evidence_receipt(manifest, args.evidence),
        stage=args.stage,
        parent_code=args.parent_code,
        lineage=[v for v in args.lineage.split(",") if v.strip()],
        notes=args.notes,
    )
    receipt["tag"] = args.tag
    receipt["passed_through"] = mtp
    receipt["preconditions"] = receipts
    receipt["gguf"] = gguf
    receipt["review_hook"] = review
    receipt["wall_time_seconds"] = round(time.time() - started, 3)
    receipt["forward_calls"] = harness.forward_calls
    receipt["receipt_sha256"] = qf_arch.canonical_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    written = qf_export.write_receipt(receipt, Path(args.out) / RECEIPT_NAME)
    log(
        f"child {args.out}  net stored ratio "
        f"{receipt['storage']['net_stored_ratio']:.4f} "
        f"({receipt['storage']['net_fold']:.2f}x)  receipt "
        f"{written['receipt_sha256'][:16]}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
