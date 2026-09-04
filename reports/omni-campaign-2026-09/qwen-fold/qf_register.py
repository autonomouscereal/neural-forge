#!/usr/bin/env python3
"""One child, one ledger row.  Reads `qf-build-receipt.json`, writes nothing.

`qf_build.py` produces a child and a receipt; `oc_checkpoint.py register` seals
that child into the campaign ledger.  Between them sits exactly one judgement
call -- WHICH `--component NAME:PARENT:CHILD[:PREFIX]` lines describe the fold --
and this file makes it mechanically, from the receipt and from the safetensors
headers, so that the same receipt always produces the same command.

Three things it does that a hand-typed register does not:

* **It picks the granularity.**  `oc_checkpoint._components` refuses any folded
  component whose ratio is above `MAX_COMPONENT_RATIO` (0.25) -- the >=4x rule is
  per component.  A receipt's own `per_component_ratios` are whole-model rollups
  (`mlp`, `gdn_mixer`, `attention_mixer`, `interface`, `vision`), and for several
  Wave A axes the rollup is diluted by tensors the axis never touched: C1 at
  62,080 folds `embed_tokens` exactly 4x but the `interface` rollup also carries
  the final norm, so the rollup ratio is 0.2500004 and the ledger refuses it by a
  hair.  So the tensor index is cut at four widths -- `...layers.7.`, then
  `...layers.7.mlp.`, then `...layers.7.mlp.gate_proj.`, then the tensor itself --
  and the COARSEST cut whose every changed piece clears the ledger's ratio is the
  one that gets registered.  Each of those carries a TENSOR_PREFIX, so the ledger
  re-derives both counts from the parent's and the child's headers and refuses
  them if they disagree with what this script computed.  The receipt's own
  rollups are the last resort, because they are asserted rather than derived.
  Nothing is rounded and nothing is waived: if no width clears the gate, this
  refuses and prints the offending ratios, so a 2x rung is a REFUSAL here rather
  than a component the ledger silently accepts.

* **It never lets a piece hide.**  Whatever granularity is chosen, the sum of
  `parent - child` over the pieces must equal the receipt's own total component
  saving, which is itself cross-checked against `sum(axes[].parameters_removed)`.
  Three independent closed forms have to agree before a command is printed.

* **It knows layer excision renames rows.**  C3/C4 re-indexes: the child's
  `layers.3.*` is the parent's `layers.4.*`, so a tensor-name prefix derived from
  the child would compare two different layers and a "derived" component would be
  a derived lie.  The excision path therefore names one asserted component per
  excised layer out of `axes[].parameters_removed` and passes NO prefix, and it
  refuses to run at all if excision was combined with a shape axis.

Usage
-----

    qf_register.py C2K2304 --recipe-command './bodyq.sh C2K2304 --mlp-keep 2304'
    qf_register.py C2K2304 --recipe-command '...' --confirm

Without `--confirm` the ledger's own dry run is executed (it prints the plan and
writes nothing); with `--confirm` the same command is run for real.  `--print-only`
prints the command and runs nothing at all.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

RECEIPT_NAME = "qf-build-receipt.json"
RECEIPT_SCHEMA = "neural-forge-omni-qwen3_5-standard-shape-child-v1"

# oc_checkpoint.MAX_COMPONENT_RATIO.  Duplicated deliberately as a FLOOR for the
# search, never as an authority: the ledger re-checks every component it is
# handed, and a divergence between these two numbers can only make this script
# refuse something the ledger would have taken, never the other way round.
LEDGER_MAX_COMPONENT_RATIO = 0.25

RECEIPT_TOP_LEVEL = (
    "schema", "family", "stage", "parent_code", "parent_model", "lineage",
    "quantization", "standard_shape", "config_diff", "storage",
    "per_component_ratios", "axes", "evidence", "export", "verification",
    "notes", "thresholds", "receipt_sha256", "tag", "passed_through",
    "preconditions", "gguf", "review_hook", "wall_time_seconds", "forward_calls",
)
STORAGE_KEYS = (
    "parent_parameter_count", "child_parameter_count", "net_stored_ratio", "net_fold",
)
COMPONENT_KEYS = ("component", "parent_parameters", "child_parameters", "ratio", "fold")

EXCISION_AXIS = "layer_excision"

# Axis label -> the qf_build flag that MUST appear in --recipe-command.  This is
# the guard against registering child B's recipe against child A's receipt: the
# receipt does not record argv, so the recipe line is operator input and the only
# thing that can check it is the axes the receipt DOES record.
AXIS_FLAG = {
    "vocab_size": "--vocab-keep",
    "intermediate_size": "--mlp-keep",
    EXCISION_AXIS: "--excise",
    "linear_num_value_heads": "--gdn-",
    "linear_value_head_dim": "--gdn-",
    "head_dim (NoPE coordinates)": "--nope-keep",
    "num_key_value_heads": "--kv-heads",
    "num_key_value_heads+num_attention_heads": "--kv-heads",
    "terminal_gain": "--terminal-gain",
}


def log(message: str) -> None:
    print(f"[qfreg] {message}", flush=True)


# --------------------------------------------------------------------------
# the receipt, checked before it is believed
# --------------------------------------------------------------------------


def _dump_and_refuse(receipt: dict[str, Any], why: str) -> None:
    """Print what the receipt actually contains, then stop.

    A structural surprise is not something to work around: the whole point of
    deriving the components mechanically is that the derivation is checkable, and
    a receipt this file does not recognise has to be READ by a human before any
    row is written about it.
    """

    print(f"\nREFUSED: {why}", file=sys.stderr)
    print(f"\nreceipt top-level keys:\n  {sorted(receipt)}", file=sys.stderr)
    for block in ("storage", "per_component_ratios", "fold"):
        if block in receipt:
            print(
                f"\nreceipt[{block!r}]:\n"
                + json.dumps(receipt[block], indent=2, sort_keys=True),
                file=sys.stderr,
            )
    axes = receipt.get("axes")
    if isinstance(axes, list):
        print("\nreceipt['axes'] (heads only):", file=sys.stderr)
        for axis in axes:
            if isinstance(axis, dict):
                head = {
                    k: v for k, v in axis.items()
                    if k in ("axis", "kind", "arm", "sites", "component_ratio",
                             "component_fold", "parameters_removed", "keep",
                             "excised", "excised_kinds", "parent_vocab", "child_vocab")
                }
                print("  " + json.dumps(head, sort_keys=True), file=sys.stderr)
            else:
                print(f"  {axis!r}", file=sys.stderr)
    raise SystemExit(3)


def read_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"REFUSED: no receipt at {path}; the child is not built")
    try:
        receipt = json.loads(path.read_text(encoding="ascii"))
    except (ValueError, UnicodeDecodeError) as error:
        raise SystemExit(f"REFUSED: {path} is not the ascii JSON this lane writes: {error}")
    if not isinstance(receipt, dict):
        raise SystemExit(f"REFUSED: {path} is not a JSON object")

    missing = [key for key in RECEIPT_TOP_LEVEL if key not in receipt]
    if missing:
        _dump_and_refuse(receipt, f"the receipt is missing {missing}")
    if receipt["schema"] != RECEIPT_SCHEMA:
        _dump_and_refuse(
            receipt, f"schema is {receipt['schema']!r}, this file reads {RECEIPT_SCHEMA!r}"
        )
    if receipt["stage"] != "fold":
        _dump_and_refuse(receipt, f"stage is {receipt['stage']!r}; this file registers folds")
    if receipt["quantization"] != "none":
        _dump_and_refuse(
            receipt,
            f"quantization is {receipt['quantization']!r}; the ledger refuses a "
            "quantized fold and so does this",
        )

    storage = receipt["storage"]
    if not isinstance(storage, dict) or any(k not in storage for k in STORAGE_KEYS):
        _dump_and_refuse(receipt, f"storage does not carry {list(STORAGE_KEYS)}")
    components = receipt["per_component_ratios"]
    if not isinstance(components, list) or not components:
        _dump_and_refuse(receipt, "per_component_ratios is not a non-empty list")
    for entry in components:
        if not isinstance(entry, dict) or any(k not in entry for k in COMPONENT_KEYS):
            _dump_and_refuse(receipt, f"a per_component_ratios entry lacks {list(COMPONENT_KEYS)}")
    axes = receipt["axes"]
    if not isinstance(axes, list) or not axes:
        _dump_and_refuse(receipt, "axes is not a non-empty list")
    for axis in axes:
        if not isinstance(axis, dict) or "axis" not in axis or "parameters_removed" not in axis:
            _dump_and_refuse(receipt, "an axis entry lacks 'axis' / 'parameters_removed'")
    export = receipt["export"]
    if not isinstance(export, dict) or "child_tensors_sha256" not in export:
        _dump_and_refuse(receipt, "export does not carry child_tensors_sha256")
    verification = receipt["verification"]
    if not isinstance(verification, dict) or "child_tensors_sha256" not in verification:
        _dump_and_refuse(receipt, "verification does not carry child_tensors_sha256")
    if verification["child_tensors_sha256"] != export["child_tensors_sha256"]:
        _dump_and_refuse(
            receipt,
            "export and verification disagree about the child digest "
            f"({export['child_tensors_sha256'][:16]} vs "
            f"{verification['child_tensors_sha256'][:16]}); the child changed "
            "between writing and re-reading",
        )
    return receipt


# --------------------------------------------------------------------------
# safetensors headers -- the same numbers the ledger will re-read
# --------------------------------------------------------------------------


def tensor_counts(directory: Path, oc_seal: Any) -> dict[str, int]:
    return oc_seal.safetensors_tensor_counts(directory)


def _cut(name: str, depth: int) -> str:
    """The prefix that keeps the first `depth` segments of `name`.

    `model.language_model.layers.7.mlp.gate_proj.weight` at depth 5 is
    `model.language_model.layers.7.mlp.`.  The trailing dot matters: without it
    `...layers.1.` prefix-matches `...layers.11.` and one component would
    silently sum eleven layers.  A name with no more segments than `depth` is
    its own prefix, exactly (no trailing dot), which is what keeps
    `lm_head.weight` from being cut down to `lm_head.` and colliding with
    nothing useful.

    Cutting from the FRONT rather than from the back is the whole point: a
    back-cut puts `...layers.7.input_layernorm.weight` (six segments) and
    `...layers.7.mlp.gate_proj.weight` (seven) in different groups at the same
    setting, and the shallower group's prefix then reaches into the deeper one's
    tensors.  A front cut is closed by construction -- every tensor whose name
    starts with the prefix has those leading segments, so it IS in the group.
    """

    parts = name.split(".")
    if depth >= len(parts):
        return name
    return ".".join(parts[:depth]) + "."


def _prefix_is_unambiguous(prefix: str, names: Iterable[str], covered: set[str]) -> bool:
    """No tensor outside `covered` may start with `prefix`.

    `oc_seal.prefix_parameter_count` matches by `startswith`, so a prefix that
    reaches a tensor the component does not own would make the ledger's derived
    count larger than the one this script computed -- and the ledger would refuse
    the component with a confusing message instead of this one.
    """

    return all(name in covered for name in names if name.startswith(prefix))


# --------------------------------------------------------------------------
# component derivation
# --------------------------------------------------------------------------


class Piece:
    """One `--component` line, before it is a string."""

    def __init__(self, name: str, parent: int, child: int, prefix: str | None) -> None:
        self.name = name
        self.parent = int(parent)
        self.child = int(child)
        self.prefix = prefix

    @property
    def ratio(self) -> float:
        return self.child / float(self.parent) if self.parent else 0.0

    @property
    def saving(self) -> int:
        return self.parent - self.child

    def spec(self) -> str:
        base = f"{self.name}:{self.parent}:{self.child}"
        return f"{base}:{self.prefix}" if self.prefix else base

    def __repr__(self) -> str:
        source = "derived" if self.prefix else "asserted"
        return f"{self.name} {self.parent:,}->{self.child:,} ({self.ratio:.6f}, {source})"


def receipt_component_saving(receipt: dict[str, Any]) -> int:
    return sum(
        int(c["parent_parameters"]) - int(c["child_parameters"])
        for c in receipt["per_component_ratios"]
    )


def assert_axes_agree_with_components(receipt: dict[str, Any]) -> int:
    """`sum(axes[].parameters_removed)` vs the rolled-up component saving.

    Two closed forms written by two different modules from two different objects
    (the axis reports from the live geometry, the components from the two
    configs).  They are the same number or the receipt is not describing one
    transaction, and neither of them is worth registering until they agree.
    """

    axis_removed = sum(int(a["parameters_removed"]) for a in receipt["axes"])
    component_saving = receipt_component_saving(receipt)
    if axis_removed != component_saving:
        _dump_and_refuse(
            receipt,
            f"sum(axes[].parameters_removed) = {axis_removed:,} but the "
            f"per_component_ratios roll up to a saving of {component_saving:,}; "
            "the receipt's two closed forms disagree about what this fold removed",
        )
    return component_saving


def unchanged_rollups(receipt: dict[str, Any]) -> list[str]:
    """The receipt's whole-model components the fold did not touch.

    This is what `--passed-through` gets at every granularity.  The alternative
    -- listing the several hundred individual tensors a fold left alone -- is a
    record nobody reads; the rollup names are the ones `per_component_ratios`
    already speaks, and the tensor-level arithmetic is in `fold_accounting`.
    """

    return [
        str(entry["component"])
        for entry in receipt["per_component_ratios"]
        if int(entry["parent_parameters"]) > 0
        and int(entry["parent_parameters"]) == int(entry["child_parameters"])
    ]


def pieces_from_components(receipt: dict[str, Any]) -> list[Piece]:
    """Last resort: the receipt's own whole-model rollups, asserted."""

    return [
        Piece(
            str(entry["component"]),
            int(entry["parent_parameters"]),
            int(entry["child_parameters"]),
            None,
        )
        for entry in receipt["per_component_ratios"]
        if int(entry["parent_parameters"]) > 0
        and int(entry["parent_parameters"]) != int(entry["child_parameters"])
    ]


def pieces_from_headers(
    parent_counts: dict[str, int],
    child_counts: dict[str, int],
    *,
    depth: int,
) -> list[Piece] | None:
    """One cut of the tensor index, every piece DERIVED from a prefix.

    Only tensors the CHILD has are considered.  A tensor the parent has and the
    child does not is not a folded component -- on this parent it is the MTP
    block, which the build declines to carry rather than folds -- and naming it
    as a component would credit the fold with a saving it did not make.  Those
    names go to the notes and to `fold_accounting.unexplained` instead.

    A front cut is closed among the child's own names, but the PARENT may hold
    names the child does not, and a prefix that reaches one of those would make
    the ledger derive a bigger parent count than this script computed.  Such a
    group falls back to its members' own full names, which are exact prefixes of
    themselves.  If even that is ambiguous -- one tensor name being a strict
    prefix of another -- the cut is unusable and returns None.
    """

    all_names = set(parent_counts) | set(child_counts)
    groups: dict[str, list[str]] = {}
    for name in sorted(child_counts):
        groups.setdefault(_cut(name, depth), []).append(name)

    resolved: list[tuple[str, list[str]]] = []
    for prefix in sorted(groups):
        members = groups[prefix]
        if _prefix_is_unambiguous(prefix, all_names, set(members)):
            resolved.append((prefix, members))
            continue
        for name in members:
            if not _prefix_is_unambiguous(name, all_names, {name}):
                return None
            resolved.append((name, [name]))

    changed: list[Piece] = []
    for prefix, members in resolved:
        parent = sum(int(parent_counts.get(n, 0)) for n in members)
        child = sum(int(child_counts[n]) for n in members)
        if parent == child:
            continue
        changed.append(Piece(_short_name(prefix), parent, child, prefix))
    return changed


def _short_name(prefix: str) -> str:
    """A readable component NAME.  No colon, because the spec splits on colons."""

    name = prefix.rstrip(".")
    for noisy in ("model.language_model.", "model.", "language_model."):
        if name.startswith(noisy):
            name = name[len(noisy):]
            break
    return name.replace(":", "_") or prefix


def granularities(child_counts: dict[str, int]) -> list[tuple[str, int | None]]:
    """Coarsest first, and every header cut before the receipt's asserted rollups.

    A component the ledger re-derives from both sets of headers is worth more
    than one it has to take this script's word for, so the depth ladder is
    exhausted before `per_component_ratios` is even considered.
    """

    deepest = max((len(name.split(".")) for name in child_counts), default=1)
    ladder: list[tuple[str, int | None]] = [
        (f"headers/depth-{depth}", depth) for depth in range(1, deepest + 1)
    ]
    ladder.append(("receipt rollups (asserted)", None))
    return ladder


def choose_granularity(
    receipt: dict[str, Any],
    parent_counts: dict[str, int],
    child_counts: dict[str, int],
) -> tuple[str, list[Piece]]:
    """Coarsest granularity whose every changed piece clears the ledger's gate."""

    target = assert_axes_agree_with_components(receipt)
    child_only = sorted(set(child_counts) - set(parent_counts))
    if child_only:
        raise SystemExit(
            "REFUSED: the child carries tensors the parent does not "
            f"({child_only[:6]}{'...' if len(child_only) > 6 else ''}); a fold that "
            "invents tensor names cannot be described by prefixes shared with its "
            "parent"
        )

    attempts: list[tuple[str, list[Piece], str]] = []
    seen: set[tuple[str, ...]] = set()
    for label, depth in granularities(child_counts):
        if depth is None:
            changed: list[Piece] | None = pieces_from_components(receipt)
        else:
            changed = pieces_from_headers(parent_counts, child_counts, depth=depth)
            if changed is not None:
                signature = tuple(sorted(p.spec() for p in changed))
                if signature in seen:
                    continue          # this depth cuts the index exactly as the last did
                seen.add(signature)
        if changed is None:
            attempts.append((label, [], "one tensor name is a strict prefix of another"))
            continue
        if not changed:
            attempts.append((label, changed, "no piece changed at this width"))
            continue
        derived = sum(p.saving for p in changed)
        if derived != target:
            attempts.append((
                label, changed,
                f"the pieces sum to a saving of {derived:,}, the receipt says {target:,}",
            ))
            continue
        over = [p for p in changed if p.ratio > LEDGER_MAX_COMPONENT_RATIO]
        if over:
            attempts.append((
                label, changed,
                "above the ledger's per-component ratio "
                f"{LEDGER_MAX_COMPONENT_RATIO}: "
                + ", ".join(f"{p.name} {p.ratio:.6f}" for p in over[:8])
                + (f" (+{len(over) - 8} more)" if len(over) > 8 else ""),
            ))
            continue
        return label, changed

    print(
        "\nREFUSED: no granularity of this fold satisfies the ledger's per-component "
        f">=4x rule (ratio <= {LEDGER_MAX_COMPONENT_RATIO}).  Tried, coarsest first:",
        file=sys.stderr,
    )
    for label, changed, why in attempts:
        print(f"\n  [{label}] {why}", file=sys.stderr)
        for piece in changed[:12]:
            print(f"      {piece!r}", file=sys.stderr)
        if len(changed) > 12:
            print(f"      ... {len(changed) - 12} more", file=sys.stderr)
    print(
        "\nThis is the campaign's own rule, not a bug in the build: CHECKPOINT_POLICY\n"
        "makes >=4x per component the definition of a fold, and Wave A deliberately\n"
        "builds 2x rungs as well.  The child on disk and its receipt are untouched;\n"
        "what is missing is a ledger stage for a rung that folds by less than four.\n"
        "Take it to Cyril rather than widening the rule here.",
        file=sys.stderr,
    )
    raise SystemExit(3)


def excision_pieces(receipt: dict[str, Any]) -> tuple[list[Piece], list[str]]:
    """C3/C4: one asserted component per excised layer, and never a prefix.

    The child re-indexes, so `...layers.3.` names the parent's layer 3 and the
    child's layer 4.  A TENSOR_PREFIX makes the ledger derive both counts from
    that one string; here the string would name two different layers and the
    "derived" number would be a derived falsehood.  The counts therefore come
    from `qf_excise.parameters_removed`, which is closed form off the parent's
    own geometry, and the record says `asserted-by-operator` -- which is the
    truth about them.
    """

    axes = receipt["axes"]
    excision = [a for a in axes if a.get("axis") == EXCISION_AXIS]
    if len(axes) != 1:
        _dump_and_refuse(
            receipt,
            "layer excision was combined with another axis in one build.  Excision "
            "re-indexes every layer, so this file cannot tell which of the other "
            "axis's tensors moved because they were folded and which moved because "
            "they were renumbered.  Register that child by hand.",
        )
    axis = excision[0]
    for key in ("excised", "excised_kinds", "parent_layers", "child_layers"):
        if key not in axis:
            _dump_and_refuse(receipt, f"the excision axis lacks {key!r}")
    excised = [int(i) for i in axis["excised"]]
    kinds = [str(k) for k in axis["excised_kinds"]]
    removed = int(axis["parameters_removed"])
    if len(excised) != len(kinds):
        _dump_and_refuse(receipt, "excised and excised_kinds are different lengths")
    if removed <= 0:
        _dump_and_refuse(receipt, f"the excision axis removed {removed} parameters")

    if len(excised) == 1:
        name = f"excised.L{excised[0]:02d}.{kinds[0]}"
        pieces = [Piece(name, removed, 0, None)]
    else:
        # Two excised layers of different kinds cannot be split out of one total,
        # so they register as ONE component that removed all of them.  Wave A
        # excises one at a time; this branch exists so a later wave does not get
        # a wrong per-layer split instead of an honest aggregate.
        label = "+".join(f"L{i:02d}" for i in excised)
        pieces = [Piece(f"excised.{label}", removed, 0, None)]
    carried = [
        f"{int(axis['child_layers'])} of {int(axis['parent_layers'])} layers verbatim"
    ]
    return pieces, carried


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------


def walk(obj: Any, path: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), value
            yield from walk(value, here)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            here = f"{path}[{index}]"
            yield from walk(value, here)


def frame_mismatches(receipt: dict[str, Any]) -> list[tuple[str, str, int]]:
    """Every `*_frame_mismatch` the receipt names, deduplicated with its count."""

    seen: dict[tuple[str, str], int] = {}
    for _path, key, value in walk(receipt):
        if key.endswith("_frame_mismatch") and isinstance(value, str) and value.strip():
            seen[(key, value.strip())] = seen.get((key, value.strip()), 0) + 1
    return [(key, value, count) for (key, value), count in sorted(seen.items())]


def fit_relL2_range(receipt: dict[str, Any]) -> tuple[float, float, int] | None:
    values: list[float] = []
    for path, key, value in walk(receipt):
        if key == "relL2" and isinstance(value, (int, float)) and ".scores.fit." in path:
            values.append(float(value))
    if not values:
        return None
    return min(values), max(values), len(values)


def causal_summary(receipt: dict[str, Any]) -> str:
    block = receipt["preconditions"].get("causal_masks")
    if not block:
        return "causal_masks not run (only the excision axis runs it)"
    full = [v for v in block.values() if isinstance(v, dict) and "above_diagonal_mass" in v]
    linear = [v for v in block.values() if isinstance(v, dict) and "masks_seen" in v]
    worst = max((float(v["above_diagonal_mass"]) for v in full), default=0.0)
    masks = sorted({str(m) for v in linear for m in v["masks_seen"]})
    return (
        f"causal_masks: {len(block)} layers, {len(full)} full-attention "
        f"(max above-diagonal mass {worst:.6e}), {len(linear)} GDN "
        f"(masks seen {masks or ['(none)']})"
    )


def parity_summary(receipt: dict[str, Any]) -> str:
    block = receipt["preconditions"].get("parity")
    if not block:
        return "parity: not run (--no-parity)"
    return (
        f"parity: deterministic={block.get('deterministic')}, "
        f"hook_placement_worst={float(block.get('hook_placement_worst', 0.0)):.3e} "
        f"over {len(block.get('hook_placement_relL2') or {})} sites, "
        f"logits_bit_identical_under_hooks={block.get('logits_bit_identical_under_hooks')}"
    )


def build_notes(
    receipt: dict[str, Any],
    *,
    granularity: str,
    parent_only: list[str],
    header_parent: int,
    header_child: int,
) -> str:
    storage = receipt["storage"]
    parts = [
        f"qf-build-receipt {receipt['receipt_sha256'][:16]} "
        f"(child_tensors_sha256 {receipt['export']['child_tensors_sha256'][:16]})",
        f"axes {[a['axis'] for a in receipt['axes']]}",
        f"closed-form storage {int(storage['parent_parameter_count']):,} -> "
        f"{int(storage['child_parameter_count']):,} "
        f"(net_stored_ratio {float(storage['net_stored_ratio']):.6f}, "
        f"{float(storage['net_fold']):.4f}x)",
        f"components registered at {granularity} granularity",
        parity_summary(receipt),
        causal_summary(receipt),
    ]
    fit = fit_relL2_range(receipt)
    if fit:
        parts.append(f"per-site fit relL2 {fit[0]:.4f}-{fit[1]:.4f} over {fit[2]} sites")
    mismatches = frame_mismatches(receipt)
    if mismatches:
        parts.append(
            "frame mismatches: "
            + "; ".join(f"{key} x{count}: {value}" for key, value, count in mismatches)
        )
    else:
        parts.append("frame mismatches: none named in this receipt")
    gguf = receipt.get("gguf") or {}
    if gguf.get("ran"):
        parts.append(f"gguf: ran, returncode {gguf.get('returncode')}")
    else:
        parts.append(f"gguf: NOT RUN -- {gguf.get('reason')}")
    if parent_only:
        unexplained = header_parent - header_child - receipt_component_saving(receipt)
        parts.append(
            f"the child carries none of {len(parent_only)} parent tensors "
            f"({parent_only[:4]}{'...' if len(parent_only) > 4 else ''}); the header "
            f"drop exceeds the receipt's component saving by {unexplained:,} "
            f"parameters, which is what fold_accounting.unexplained will show, and "
            f"receipt['passed_through'] says {receipt.get('passed_through')}"
        )
    parts.append(f"evidence manifest {receipt['evidence']['manifest_sha256'][:16]}")
    return " | ".join(parts)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def assert_recipe_matches_axes(receipt: dict[str, Any], command: str) -> None:
    tag = str(receipt["tag"])
    if tag not in command.split():
        raise SystemExit(
            f"REFUSED: --recipe-command {command!r} does not carry the tag {tag!r} as "
            "its own word; a recipe that names another child cannot rebuild this one"
        )
    for axis in receipt["axes"]:
        label = str(axis["axis"])
        flag = AXIS_FLAG.get(label)
        if flag is None:
            raise SystemExit(
                f"REFUSED: axis {label!r} has no known qf_build flag in AXIS_FLAG, so "
                "the recipe command cannot be checked against it"
            )
        if flag not in command:
            raise SystemExit(
                f"REFUSED: the receipt has a {label!r} axis but --recipe-command has "
                f"no {flag!r}; the recipe does not rebuild the child it is filed under"
            )


def register_command(
    *,
    ledger_lane: Path,
    python: str,
    root: Path,
    checkpoint_id: str,
    tag: str,
    wave: str,
    child: Path,
    parent_directory: str,
    parent_code: str,
    parent_checkpoint_id: str | None,
    pieces: list[Piece],
    passed_through: list[str],
    recipe: Path,
    recipe_command: str,
    notes: str,
    confirm: bool,
) -> list[str]:
    # `--root` belongs to the TOP-LEVEL parser and `--registry` / `--ledger` to the
    # subcommand's, so the root goes before the word `register` and an argparse
    # "unrecognized arguments" is what a caller gets for putting it after.
    argv = [
        python, str(ledger_lane / "oc_checkpoint.py"),
        "--root", str(root),
        "register",
        "--checkpoint-id", checkpoint_id,
        "--stage", "fold",
        "--directory", str(child),
        "--parent-directory", parent_directory,
        "--parent-code", parent_code,
        "--tag", tag,
        "--wave", wave,
        "--parameter-count", "auto",
        "--parent-parameter-count", "auto",
        "--quantization", "none",
    ]
    if parent_checkpoint_id:
        argv += ["--parent-checkpoint-id", parent_checkpoint_id]
    for piece in pieces:
        argv += ["--component", piece.spec()]
    for name in passed_through:
        argv += ["--passed-through", name]
    # NO `--lineage`.  `oc_checkpoint.cmd_register` binds the child's directory
    # digest to `digest` at line 1289 and then REBINDS the same name inside the
    # `--lineage ID:DIGEST` loop at line 1356 (`ancestor, _, digest = ...`), so
    # the value that reaches `"directory_sha256": digest` at line 1407 -- and the
    # value printed in the plan at 1445, and the value written into the ledger
    # row -- is the LAST lineage entry's digest instead of the directory's.
    # Reproduced against the fixture campaign: three different fixture children
    # registered with `--lineage EVIDENCE-...:eeee...` all sealed
    # `directory_sha256 = "e" * 64`.  Until that shadow is fixed in the ledger
    # lane, this file states the evidence manifest in the NOTES, where a wrong
    # value cannot be mistaken for a digest of the bytes, and the registered
    # parent still reaches the lineage block through --parent-checkpoint-id
    # (which is appended after the loop and does not rebind anything).
    argv += [
        "--recipe", str(recipe),
        "--recipe-command", recipe_command,
        "--notes", notes,
    ]
    if confirm:
        argv.append("--confirm")
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="the child's TAG, e.g. C2K2304")
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parent,
                        help="the qwen-fold lane root")
    parser.add_argument("--root", type=Path, default=Path("/home/cereal/omni-campaign"),
                        help="OC_ROOT; the registry and ledger.jsonl live under it")
    parser.add_argument("--ledger-lane", type=Path, default=None,
                        help="where oc_checkpoint.py / oc_seal.py live (default <root>/ledger)")
    parser.add_argument("--child", type=Path, default=None,
                        help="the child directory (default <work>/children/QF-<TAG>)")
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter that runs oc_checkpoint.py")
    parser.add_argument("--wave", default="oc01")
    parser.add_argument("--parent-checkpoint-id", default="PARENT-Q35-4B",
                        help="the registered parent; '' to omit it")
    parser.add_argument("--recipe-command", required=True,
                        help="the exact bodyq.sh line that built this child")
    parser.add_argument("--recipe", type=Path, default=None,
                        help="the recipe file to digest (default <work>/bodyq.sh)")
    parser.add_argument("--confirm", action="store_true",
                        help="run the register for real; without it the LEDGER's own "
                             "dry run is executed and nothing is written")
    parser.add_argument("--print-only", action="store_true",
                        help="print the command and run nothing at all")
    parser.add_argument("--no-postcheck", action="store_true",
                        help="skip re-hashing the child against the sealed record "
                             "after a confirmed register")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work.resolve()
    ledger_lane = (args.ledger_lane or (args.root / "ledger")).resolve()
    child = (args.child or (work / "children" / f"QF-{args.tag}")).resolve()
    recipe = (args.recipe or (work / "bodyq.sh")).resolve()

    sys.path.insert(0, str(ledger_lane))
    try:
        import oc_seal  # noqa: E402  -- the ledger lane's pure-stdlib header reader
    except ImportError as error:
        raise SystemExit(f"REFUSED: cannot import oc_seal from {ledger_lane}: {error}")

    receipt = read_receipt(child / RECEIPT_NAME)
    if str(receipt["tag"]) != args.tag:
        raise SystemExit(
            f"REFUSED: {child}'s receipt is tagged {receipt['tag']!r}, not {args.tag!r}"
        )
    assert_recipe_matches_axes(receipt, args.recipe_command)

    parent_directory = str(receipt["parent_model"])
    parent_code = str(receipt["parent_code"])
    if not parent_code:
        _dump_and_refuse(receipt, "the receipt carries no parent_code")
    checkpoint_id = f"OC{args.wave[2:]}-F-{parent_code}-{args.tag}"

    parent_counts = tensor_counts(Path(parent_directory), oc_seal)
    child_counts = tensor_counts(child, oc_seal)
    header_parent = sum(int(v) for v in parent_counts.values())
    header_child = sum(int(v) for v in child_counts.values())

    axis_labels = [str(a["axis"]) for a in receipt["axes"]]
    parent_only = sorted(set(parent_counts) - set(child_counts))
    if EXCISION_AXIS in axis_labels:
        assert_axes_agree_with_components(receipt)
        pieces, passed_through = excision_pieces(receipt)
        granularity = "excised layer (asserted; C3/C4 re-indexes, so no prefix is true)"
    else:
        granularity, pieces = choose_granularity(receipt, parent_counts, child_counts)
        passed_through = unchanged_rollups(receipt)

    log(f"child            {child}")
    log(f"checkpoint id    {checkpoint_id}")
    log(f"axes             {axis_labels}")
    log(
        f"headers          parent {header_parent:,} -> child {header_child:,} "
        f"(net {header_child / header_parent:.6f}, {header_parent / header_child:.4f}x)"
    )
    log(
        f"closed form      parent {int(receipt['storage']['parent_parameter_count']):,} "
        f"-> child {int(receipt['storage']['child_parameter_count']):,} "
        f"(net {float(receipt['storage']['net_stored_ratio']):.6f})"
    )
    if parent_only:
        log(
            f"parent-only      {len(parent_only)} tensors the child does not carry: "
            f"{parent_only[:4]}{'...' if len(parent_only) > 4 else ''}"
        )
    log(f"granularity      {granularity}  ({len(pieces)} components)")
    for piece in pieces[:16]:
        log(f"  component      {piece!r}")
    if len(pieces) > 16:
        log(f"  component      ... {len(pieces) - 16} more")
    log(f"passed through   {passed_through}")

    notes = build_notes(
        receipt,
        granularity=granularity,
        parent_only=parent_only,
        header_parent=header_parent,
        header_child=header_child,
    )
    command = register_command(
        ledger_lane=ledger_lane,
        python=args.python,
        root=args.root,
        checkpoint_id=checkpoint_id,
        tag=args.tag,
        wave=args.wave,
        child=child,
        parent_directory=parent_directory,
        parent_code=parent_code,
        parent_checkpoint_id=args.parent_checkpoint_id or None,
        pieces=pieces,
        passed_through=passed_through,
        recipe=recipe,
        recipe_command=args.recipe_command,
        notes=notes,
        confirm=bool(args.confirm),
    )

    print("\n" + " \\\n  ".join(shlex.quote(part) for part in command) + "\n", flush=True)
    if args.print_only:
        return 0
    log(f"running the register {'FOR REAL' if args.confirm else 'as a DRY RUN'}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        log(f"oc_checkpoint.py register exited {completed.returncode}")
        return completed.returncode
    if args.confirm and not args.no_postcheck:
        postcheck(oc_seal, args.root, checkpoint_id, child)
    return 0


def postcheck(oc_seal: Any, root: Path, checkpoint_id: str, child: Path) -> None:
    """Re-read the sealed record and re-hash the child.  Trust the row, not the run.

    `register` prints a plan and then a `sealed <path>`, and both come out of the
    same in-memory dict; neither of them is evidence that what landed on disk
    describes these bytes.  This re-opens the record the registry actually holds
    and re-computes the directory digest from the files, which is how the
    `--lineage` variable shadow above was found in the first place.
    """

    registry = Path(root) / "registry" / checkpoint_id
    records = sorted(registry.glob(f"{checkpoint_id}-*-register.json"))
    if not records:
        raise SystemExit(f"REFUSED: register reported success but {registry} holds no record")
    sealed = json.loads(records[-1].read_text(encoding="utf-8"))["checkpoint"]
    live, _entries = oc_seal.directory_digest(child)
    if sealed["directory_sha256"] != live:
        raise SystemExit(
            f"REFUSED: {records[-1].name} seals directory_sha256 "
            f"{sealed['directory_sha256']} but {child} digests {live}.  The record "
            "does not name the bytes it claims to; do not review this checkpoint "
            "until the registry row is repaired."
        )
    log(
        f"postcheck ok: {records[-1].name} names the child's own digest "
        f"{live[:16]}, net_ratio {float(sealed['net_ratio']):.10f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
