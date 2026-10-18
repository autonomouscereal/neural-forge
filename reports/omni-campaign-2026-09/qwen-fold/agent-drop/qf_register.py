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
  rollups are used ONLY when no header cut could be constructed at all (every
  depth was name-ambiguous); a header cut that was constructible and failed --
  on ratio, or on disagreeing with the receipt -- is a REFUSAL, never a fall-back
  to an asserted number the bytes contradict.  Nothing is rounded and nothing is
  waived: a 2x rung is a refusal here rather than a component the ledger
  silently accepts.

* **It never lets a piece hide.**  Whatever granularity is chosen, the sum of
  `parent - child` over the pieces must equal the receipt's own total component
  saving, which is itself cross-checked against `sum(axes[].parameters_removed)`.
  Three independent closed forms have to agree before a command is printed.

* **It knows layer excision renames rows.**  C3/C4 re-indexes: the child's
  `layers.3.*` is the parent's `layers.4.*`, so a tensor-name prefix derived from
  the child would compare two different layers and a "derived" component would be
  a derived lie.  The excision path therefore names one asserted component per
  excised layer out of `axes[].parameters_removed`, passes NO prefix, and proves
  that number against the PARENT's headers (`...layers.<i>.` of the parent index)
  -- the check the ledger cannot make, because the child has re-indexed.  The
  receipt's rollups are known to fall short of that number by exactly the two
  per-layer RMSNorms (no rollup carries them), and that shortfall is asserted to
  be exactly the norms read from the headers.  It refuses to run at all if
  excision was combined with a shape axis.

Two more guards, both about lineage:

* the parent code comes from the receipt when the receipt carries one, else from
  the operator (`--parent-code` / `--parent-checkpoint-id`), else from the parent
  directory's name through an explicit table; a disagreement between any two of
  those is a refusal, and a code the receipt did not carry is said so in --notes;
* the registered parent (`PARENT-<code>`) is READ from the registry and its sealed
  `checkpoint.directory` must resolve to the receipt's `parent_model`, so the
  lineage link and the header-derived parent count name the same bytes.

Usage
-----

    qf_register.py C2K2304 --recipe-command './bodyq.sh C2K2304 --mlp-keep 2304'
    qf_register.py C2K2304 --recipe-command '...' --confirm

Without `--confirm` the ledger's own dry run is executed (it prints the plan and
writes nothing); with `--confirm` the same command is run for real.  `--print-only`
prints the command and runs nothing at all.

Exit status: 0 registered (or already registered and verifying clean); 3 REFUSED
(nothing written, the reason is on stderr as `REFUSED: ...`); 4 SEALED BUT THE
POSTCHECK FAILED (the registry row exists; do not treat it as a refusal).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, NoReturn

RECEIPT_NAME = "qf-build-receipt.json"
RECEIPT_SCHEMA = "neural-forge-omni-qwen3_5-standard-shape-child-v1"

EXIT_REFUSED = 3
EXIT_POSTCHECK_FAILED = 4

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

# Axis label -> the qf_build flags that carry its VALUE(s).  The receipt does not
# record argv, so the recipe line is operator input and the only thing that can
# check it is the axes the receipt DOES record -- by value, not by flag name:
# `--mlp-keep 2304` filed under the C2K1152 receipt would otherwise be sealed as
# the rebuild recipe (and its recipe_sha256) of a child it cannot rebuild.
AXIS_FLAGS: dict[str, tuple[str, ...]] = {
    "vocab_size": ("--vocab-keep",),
    "intermediate_size": ("--mlp-keep",),
    EXCISION_AXIS: ("--excise",),
    "linear_num_value_heads": ("--gdn-value-heads", "--gdn-key-heads",
                               "--gdn-key-head-dim", "--gdn-value-head-dim"),
    "linear_value_head_dim": ("--gdn-value-heads", "--gdn-key-heads",
                              "--gdn-key-head-dim", "--gdn-value-head-dim"),
    "head_dim (NoPE coordinates)": ("--nope-keep",),
    "num_key_value_heads": ("--kv-heads", "--kv-arm"),
    "num_key_value_heads+num_attention_heads": ("--kv-heads", "--kv-arm"),
    "terminal_gain": ("--terminal-gain",),
}
GDN_FLAG_FIELDS = (
    ("--gdn-value-heads", "num_v_heads"),
    ("--gdn-key-heads", "num_k_heads"),
    ("--gdn-key-head-dim", "head_k_dim"),
    ("--gdn-value-head-dim", "head_v_dim"),
)
#: Every qf_build flag that MOVES a shape.  A recipe that carries one of these
#: for an axis the receipt does not have did not build this child.
ALL_AXIS_FLAGS = sorted({flag for flags in AXIS_FLAGS.values() for flag in flags})

# Parent directory basename -> R1 parent code, for a receipt that carries no
# parent_code and an operator who passed none.  Explicit and small on purpose:
# a parent that is not in this table has to be STATED, never guessed.
PARENT_DIRECTORY_CODES = {
    "Qwen3.5-0.8B": "Q35-0.8B",
    "Qwen3.5-2B": "Q35-2B",
    "Qwen3.5-4B": "Q35-4B",
    "Qwen3.5-9B": "Q35-9B",
}

#: Checkpoint tensors the parent index carries and transformers never loads
#: (`qf_arch.carried_prefixes`): the child lacks them unless the build carried
#: them by hand, and their absence is a drop, not a fold.
CARRIED_PREFIXES = ("mtp.",)

_WAVE = re.compile(r"^oc(\d{2})$")


def log(message: str) -> None:
    print(f"[qfreg] {message}", flush=True)


def refuse(why: str) -> NoReturn:
    """The lane's one refusal contract: `REFUSED: <why>` on stderr, exit 3.

    The same sentence shape and the same status `oc_checkpoint.py` uses, so the
    queue's `grep REFUSED:` and a human reading `rc` see one thing for
    "refused" whether it was this file or the ledger that said no.
    """

    print(f"REFUSED: {why}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_REFUSED)


# --------------------------------------------------------------------------
# the receipt, checked before it is believed
# --------------------------------------------------------------------------


def _dump_and_refuse(receipt: dict[str, Any], why: str) -> NoReturn:
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
                             "excised", "excised_kinds", "parent_vocab", "child_vocab",
                             "child_kv_heads", "child_geometry", "parent_geometry",
                             "child_head_dim", "rotary_dim")
                }
                print("  " + json.dumps(head, sort_keys=True), file=sys.stderr)
            else:
                print(f"  {axis!r}", file=sys.stderr)
    print("", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_REFUSED)


def read_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        refuse(f"no receipt at {path}; the child is not built")
    try:
        receipt = json.loads(path.read_text(encoding="ascii"))
    except (ValueError, UnicodeDecodeError) as error:
        refuse(f"{path} is not the ascii JSON this lane writes: {error}")
    if not isinstance(receipt, dict):
        refuse(f"{path} is not a JSON object")

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
    if not isinstance(receipt["parent_model"], str) or not receipt["parent_model"]:
        _dump_and_refuse(receipt, "parent_model is not a directory path")
    return receipt


def total_axis_removed(receipt: dict[str, Any]) -> int:
    return sum(int(a["parameters_removed"]) for a in receipt["axes"])


# --------------------------------------------------------------------------
# the parent: its code, and its registered record
# --------------------------------------------------------------------------


def resolve_parent_code(
    receipt: dict[str, Any], cli_code: str | None, cli_parent_id: str | None
) -> tuple[str, str, str | None]:
    """``(code, source, note)``: the parent code every party agrees on.

    The receipt's own value wins when it has one.  Every Wave A receipt built
    through `bodyq.sh` carries `""` (bodyq passes no --parent-code and
    qf_build defaults it empty), so the operator's value is the next word, then
    the code inside `--parent-checkpoint-id PARENT-<code>`, then the parent
    directory's basename through PARENT_DIRECTORY_CODES.  Two stated values
    that disagree are a refusal; a value the receipt did not carry is said so
    in the notes, because the row should not read as if the receipt had it.
    """

    receipt_code = str(receipt.get("parent_code") or "").strip()
    id_code = None
    if cli_parent_id:
        if not cli_parent_id.startswith("PARENT-") or len(cli_parent_id) <= len("PARENT-"):
            refuse(f"--parent-checkpoint-id {cli_parent_id!r} is not PARENT-<CODE>")
        id_code = cli_parent_id[len("PARENT-"):]
    directory_code = PARENT_DIRECTORY_CODES.get(Path(str(receipt["parent_model"])).name)

    stated = [
        (label, value) for label, value in (
            ("receipt", receipt_code or None),
            ("--parent-code", cli_code or None),
            ("--parent-checkpoint-id", id_code),
        ) if value
    ]
    distinct = sorted({value for _label, value in stated})
    if len(distinct) > 1:
        refuse(
            "the parent code is stated more than one way: "
            + ", ".join(f"{label} says {value!r}" for label, value in stated)
            + "; nothing is registered until they agree"
        )
    if stated:
        code = stated[0][1]
        if directory_code and directory_code != code:
            refuse(
                f"parent code {code!r} (from {stated[0][0]}) but the receipt's "
                f"parent_model {receipt['parent_model']!r} is the "
                f"{directory_code!r} parent by name; the receipt does not describe "
                "a child of the parent the command names"
            )
        if receipt_code:
            return code, "receipt", None
        source = stated[0][0]
        return code, source, (
            f'parent_code {code} taken from {source} (receipt carries "")'
        )
    if directory_code:
        return code_from_directory(directory_code, receipt)
    refuse(
        'the receipt carries no parent_code (""), no --parent-code / '
        "--parent-checkpoint-id was given, and the parent directory "
        f"{receipt['parent_model']!r} is not one of "
        f"{sorted(PARENT_DIRECTORY_CODES)}; state the code"
    )


def code_from_directory(code: str, receipt: dict[str, Any]) -> tuple[str, str, str]:
    return code, "parent directory name", (
        f"parent_code {code} derived from the parent directory name "
        f'{Path(str(receipt["parent_model"])).name} (receipt carries "", none stated)'
    )


def assert_registered_parent(
    oc_checkpoint: Any,
    *,
    root: Path,
    parent_checkpoint_id: str,
    parent_code: str,
    parent_directory: str,
    header_parent: int,
) -> dict[str, Any]:
    """The registered parent's sealed record, proven to be the receipt's parent.

    `oc_checkpoint register` only checks that `--parent-checkpoint-id` is
    registered.  It then seals THAT record's `directory_sha256` into the lineage
    block while `parent_parameter_count` comes from `--parent-directory`'s own
    headers, and nothing compares the two.  A child of the 0.8B filed under
    PARENT-Q35-4B would seal a lineage link to bytes it never came from.  So
    the parent's current record is read here and three things must agree: its
    directory is the receipt's `parent_model` (resolved, so a symlink or a
    `..` spelling cannot hide a second copy), its code is the code in the
    child's id, and its sealed parameter count is what the headers say now.
    """

    registry = Path(root) / "registry"
    try:
        record = oc_checkpoint.current(registry, parent_checkpoint_id)
    except (oc_checkpoint.OcCheckpointError, oc_checkpoint.OcSealError) as error:
        refuse(f"--parent-checkpoint-id {parent_checkpoint_id}: {error}")
    checkpoint = record["checkpoint"]
    if checkpoint.get("stage") != "parent":
        refuse(
            f"{parent_checkpoint_id} is registered at stage {checkpoint.get('stage')!r}, "
            "not as a parent"
        )
    if checkpoint.get("deleted"):
        refuse(f"{parent_checkpoint_id} is sealed as deleted")
    sealed_code = str(checkpoint.get("parent_code") or "")
    if sealed_code != parent_code:
        refuse(
            f"{parent_checkpoint_id} is sealed with parent_code {sealed_code!r}, the "
            f"child id would carry {parent_code!r}"
        )
    sealed_dir = oc_checkpoint.resolve_path(checkpoint["directory"])
    receipt_dir = oc_checkpoint.resolve_path(parent_directory)
    if sealed_dir != receipt_dir:
        refuse(
            f"{parent_checkpoint_id} seals directory {sealed_dir} but the receipt's "
            f"parent_model is {receipt_dir}; the lineage link and the header parent "
            "would name different bytes"
        )
    sealed_count = int(checkpoint.get("parameter_count") or 0)
    if sealed_count != header_parent:
        refuse(
            f"{parent_checkpoint_id} seals parameter_count {sealed_count:,} but the "
            f"headers under {receipt_dir} say {header_parent:,} now; the parent "
            "directory has changed since it was registered"
        )
    return checkpoint


# --------------------------------------------------------------------------
# safetensors headers -- the same numbers the ledger will re-read
# --------------------------------------------------------------------------


def tensor_counts(directory: Path, oc_seal: Any) -> dict[str, int]:
    try:
        return oc_seal.safetensors_tensor_counts(directory)
    except oc_seal.OcSealError as error:
        refuse(f"cannot read safetensors headers under {directory}: {error}")


def text_prefix(names: Iterable[str]) -> str:
    """``model.language_model.`` or ``model.`` -- how these headers name the text body."""

    for name in names:
        if name.startswith("model.language_model."):
            return "model.language_model."
    return "model."


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
    configs).  For a SHAPE axis they are the same number or the receipt is not
    describing one transaction.  (Layer excision is the known exception -- the
    axis counts the two per-layer norms and no rollup does -- and takes the
    header-based path in `excision_pieces` instead of this one.)
    """

    axis_removed = total_axis_removed(receipt)
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
    split_ambiguous: bool,
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
    depth is unusable (None) -- EXCEPT at the deepest rung of the ladder, where
    `split_ambiguous` lets a group fall back to its members' own full names,
    which is the intended granularity there and not an inversion of "coarsest
    first".  If even a full name is a strict prefix of another, the cut is
    unusable and returns None.
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
        if not split_ambiguous:
            return None
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
    """Coarsest granularity whose every changed piece clears the ledger's gate.

    The asserted rollups are reached ONLY when no header depth produced a cut at
    all.  A header cut that exists and fails -- on the ledger's ratio, or on a
    sum that disagrees with the receipt -- is a header/receipt disagreement, and
    that is exactly the case a human must read: it is refused with the table of
    attempts, never papered over with `asserted-by-operator` counts the bytes
    contradict at tensor level.
    """

    target = assert_axes_agree_with_components(receipt)
    child_only = sorted(set(child_counts) - set(parent_counts))
    if child_only:
        refuse(
            "the child carries tensors the parent does not "
            f"({child_only[:6]}{'...' if len(child_only) > 6 else ''}); a fold that "
            "invents tensor names cannot be described by prefixes shared with its "
            "parent"
        )

    ladder = granularities(child_counts)
    deepest = max((d for _label, d in ladder if d is not None), default=1)
    attempts: list[tuple[str, list[Piece], str]] = []
    seen: set[tuple[str, ...]] = set()
    header_cut_existed = False
    for label, depth in ladder:
        if depth is None:
            if header_cut_existed:
                attempts.append((
                    label, [],
                    "not reached: a header cut was constructible above and failed; "
                    "an asserted rollup may not overrule the bytes",
                ))
                break
            changed: list[Piece] | None = pieces_from_components(receipt)
        else:
            changed = pieces_from_headers(
                parent_counts, child_counts, depth=depth,
                split_ambiguous=(depth == deepest),
            )
            if changed is not None:
                header_cut_existed = True
                signature = tuple(sorted(p.spec() for p in changed))
                if signature in seen:
                    continue          # this depth cuts the index exactly as the last did
                seen.add(signature)
        if changed is None:
            attempts.append((
                label, [],
                "a prefix at this depth reaches parent-only tensors (or one tensor "
                "name is a strict prefix of another); no prefix here is closed",
            ))
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
        f">=4x rule (ratio <= {LEDGER_MAX_COMPONENT_RATIO}) while agreeing with the "
        "receipt.  Tried, coarsest first:",
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
        file=sys.stderr, flush=True,
    )
    raise SystemExit(EXIT_REFUSED)


def excision_pieces(
    receipt: dict[str, Any],
    parent_counts: dict[str, int],
    child_counts: dict[str, int],
) -> tuple[list[Piece], list[str], list[str], str]:
    """C3/C4: one asserted component per excised layer, never a prefix, and the
    number proven against the PARENT's headers.

    The child re-indexes, so `...layers.3.` names the parent's layer 3 and the
    child's layer 4.  A TENSOR_PREFIX makes the ledger derive both counts from
    that one string; here the string would name two different layers and the
    "derived" number would be a derived falsehood.  The counts therefore come
    from `qf_excise.parameters_removed`, which is closed form off the parent's
    own geometry, and the record says `asserted-by-operator` -- which is the
    truth about them.

    What this file CAN prove, and the ledger cannot: the parent index still has
    `...layers.<i>.*` for every excised i, so `parameters_removed` is checked
    against those tensors' element counts.  The receipt's rollups are known to
    fall short of that number by exactly the two per-layer RMSNorms
    (`qf_arch.text_parameter_count` puts `2*H` per layer in `per_*_layer` but
    `qf_export.component_totals` has no bucket for them), and that shortfall is
    asserted to be exactly the norm tensors read from the same headers.  The
    child is then walked through the RE-INDEX MAP (child layer j is parent
    layer surviving[j]) and must be the parent minus the excised layers tensor
    for tensor -- names cannot be compared directly, because excising a
    full-attention layer moves a linear layer into its index -- and the total
    header drop must be the excised layers plus the tensors the child dropped
    outside the layer stack (the MTP block).

    Returns ``(pieces, passed_through, dropped_parent_only, note)``.
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
    parent_layers = int(axis["parent_layers"])
    child_layers = int(axis["child_layers"])
    if len(excised) != len(kinds):
        _dump_and_refuse(receipt, "excised and excised_kinds are different lengths")
    if removed <= 0:
        _dump_and_refuse(receipt, f"the excision axis removed {removed} parameters")
    if parent_layers - child_layers != len(excised):
        _dump_and_refuse(
            receipt,
            f"parent_layers {parent_layers} - child_layers {child_layers} != "
            f"{len(excised)} excised",
        )

    # The re-index map.  Tensor NAMES cannot be compared across an excision:
    # excising the full-attention layer 3 of the 4B moves a linear layer into
    # index 3, so the child has `layers.3.linear_attn.*` the parent never had
    # and lacks `layers.3.self_attn.*` the parent has.  What is true is the
    # MAP: child layer j is parent layer surviving[j], tensor for tensor.
    excised_set = set(excised)
    surviving = [i for i in range(parent_layers) if i not in excised_set]
    if len(surviving) != child_layers:
        _dump_and_refuse(receipt, f"{parent_layers} layers minus {sorted(excised_set)} is not {child_layers}")
    if "surviving_parent_positions" in axis:
        stated = [int(i) for i in axis["surviving_parent_positions"]]
        if stated != surviving:
            _dump_and_refuse(
                receipt,
                f"surviving_parent_positions {stated} is not {surviving} "
                f"(range({parent_layers}) minus {sorted(excised_set)})",
            )

    prefix = text_prefix(parent_counts)
    layers_prefix = f"{prefix}layers."
    mapped_parent: set[str] = set()
    mismatched: list[str] = []
    child_indices: set[int] = set()
    for name in sorted(child_counts):
        if name.startswith(layers_prefix):
            head, _, rest = name[len(layers_prefix):].partition(".")
            try:
                j = int(head)
            except ValueError:
                refuse(f"cannot read a layer index out of the child tensor {name!r}")
            child_indices.add(j)
            if j >= child_layers:
                mismatched.append(f"{name}: child index {j} >= child_layers {child_layers}")
                continue
            parent_name = f"{layers_prefix}{surviving[j]}.{rest}"
        else:
            parent_name = name
        if parent_name not in parent_counts:
            mismatched.append(f"{name}: the parent has no {parent_name}")
            continue
        if int(parent_counts[parent_name]) != int(child_counts[name]):
            mismatched.append(
                f"{name}: {int(child_counts[name]):,} elements, parent {parent_name} "
                f"has {int(parent_counts[parent_name]):,}"
            )
            continue
        mapped_parent.add(parent_name)
    if child_indices != set(range(child_layers)):
        refuse(
            f"the child names layer indices {sorted(child_indices)}, not the contiguous "
            f"0..{child_layers - 1} the axis says it re-indexed to"
        )
    if mismatched:
        refuse(
            "the child is not the parent minus the excised layers, tensor for tensor "
            f"through the re-index map: {mismatched[:6]}"
            f"{'...' if len(mismatched) > 6 else ''}"
        )
    unmapped = sorted(set(parent_counts) - mapped_parent)
    excised_prefixes = tuple(f"{layers_prefix}{i}." for i in excised)
    stray_layer = [
        n for n in unmapped
        if n.startswith(layers_prefix) and not n.startswith(excised_prefixes)
    ]
    if stray_layer:
        refuse(
            f"the parent's surviving layers carry tensors the child dropped: "
            f"{stray_layer[:6]}{'...' if len(stray_layer) > 6 else ''}; that is not a "
            "layer excision"
        )
    dropped = [n for n in unmapped if not n.startswith(layers_prefix)]

    layer_totals: dict[int, int] = {}
    norm_totals: dict[int, int] = {}
    for index in excised:
        layer_prefix = f"{layers_prefix}{index}."
        layer_n = layer_t = 0
        norms = 0
        norm_tensors = 0
        for name, elements in parent_counts.items():
            if not name.startswith(layer_prefix):
                continue
            layer_n += int(elements)
            layer_t += 1
            if name.startswith(f"{layer_prefix}input_layernorm.") or name.startswith(
                f"{layer_prefix}post_attention_layernorm."
            ):
                norms += int(elements)
                norm_tensors += 1
        if layer_t == 0:
            refuse(
                f"the parent headers under {receipt['parent_model']} have no tensor "
                f"under {layer_prefix!r}; the excised index {index} is not a parent layer"
            )
        if norm_tensors != 2:
            refuse(
                f"parent layer {index} carries {norm_tensors} RMSNorm weights under "
                f"{layer_prefix!r}, this file expects exactly input_layernorm and "
                "post_attention_layernorm"
            )
        layer_totals[index] = layer_n
        norm_totals[index] = norms

    header_removed = sum(layer_totals.values())
    header_norms = sum(norm_totals.values())
    if removed != header_removed:
        refuse(
            f"the excision axis says parameters_removed {removed:,} but the parent "
            f"headers hold {header_removed:,} parameters under "
            f"{[f'{layers_prefix}{i}.' for i in excised]}; the asserted component "
            "would not be what the bytes say was removed"
        )
    rollup_saving = receipt_component_saving(receipt)
    if rollup_saving != removed - header_norms:
        refuse(
            f"the receipt's per_component_ratios roll up to a saving of "
            f"{rollup_saving:,}; for {len(excised)} excised layer(s) this file expects "
            f"exactly parameters_removed {removed:,} minus the per-layer RMSNorms "
            f"{header_norms:,} (no rollup carries the norms) = {removed - header_norms:,}"
        )

    header_parent = sum(int(v) for v in parent_counts.values())
    header_child = sum(int(v) for v in child_counts.values())
    dropped_params = sum(int(parent_counts[n]) for n in dropped)
    if header_parent - header_child != removed + dropped_params:
        refuse(
            f"the headers drop {header_parent - header_child:,} parameters, but the "
            f"excised layer(s) account for {removed:,} and the {len(dropped)} "
            f"parent-only tensors outside the layer stack for {dropped_params:,}; "
            f"{header_parent - header_child - removed - dropped_params:,} parameters "
            "moved somewhere this receipt does not describe"
        )

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
    carried = [f"{child_layers} of {parent_layers} layers verbatim", *unchanged_rollups(receipt)]
    note = (
        f"excision proven against the parent headers: "
        + ", ".join(
            f"{layers_prefix}{i}. = {layer_totals[i]:,} ({norm_totals[i]:,} in its two RMSNorms)"
            for i in excised
        )
        + f"; the receipt's rollups save {rollup_saving:,}, short by exactly the "
        f"{header_norms:,} per-layer norm parameters, which live in no receipt rollup; "
        f"child layer j is parent layer {surviving}[j] tensor for tensor, so no "
        "tensor-name prefix is true across the re-index"
    )
    return pieces, carried, dropped, note


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


def passed_through_contradiction(
    receipt: dict[str, Any], dropped_parent_only: list[str]
) -> str | None:
    """The receipt's `passed_through` statement checked against the parent index.

    `qf_build` writes `{"mtp": "not present in the parent index"}` whenever
    neither --carry-mtp nor --drop-mtp was passed (bodyq.sh passes neither), and
    on this parent that sentence is false: `mtp.*` IS in the parent index and the
    child simply does not carry it.  The row must not repeat a sentence the
    headers beside it contradict without saying so.
    """

    statement = receipt.get("passed_through")
    mtp_dropped = [n for n in dropped_parent_only if n.startswith(CARRIED_PREFIXES)]
    if not mtp_dropped or not isinstance(statement, dict):
        return None
    if str(statement.get("mtp", "")).strip() == "not present in the parent index":
        return (
            "receipt passed_through statement is WRONG for this parent: mtp.* "
            f"({len(mtp_dropped)} tensors) is present in the parent index and was "
            "neither carried nor declared dropped; the child does not carry it"
        )
    return None


def build_notes(
    receipt: dict[str, Any],
    *,
    granularity: str,
    dropped_parent_only: list[str],
    header_parent: int,
    header_child: int,
    claimed_saving: int,
    parent_code_note: str | None,
    parent_link: str,
    extra: list[str],
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
        parent_link,
    ]
    if parent_code_note:
        parts.append(parent_code_note)
    parts += [parity_summary(receipt), causal_summary(receipt)]
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
    parts += extra
    if dropped_parent_only:
        unexplained = header_parent - header_child - claimed_saving
        parts.append(
            f"the child carries none of {len(dropped_parent_only)} parent tensors "
            f"({dropped_parent_only[:4]}{'...' if len(dropped_parent_only) > 4 else ''}); "
            f"the header drop exceeds the claimed component saving by {unexplained:,} "
            f"parameters, which is what fold_accounting.unexplained will show, and "
            f"receipt['passed_through'] says {receipt.get('passed_through')}"
        )
        contradiction = passed_through_contradiction(receipt, dropped_parent_only)
        if contradiction:
            parts.append(contradiction)
    parts.append(f"evidence manifest {receipt['evidence']['manifest_sha256'][:16]}")
    return " | ".join(parts)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def _flag_values(tokens: list[str]) -> dict[str, list[str | None]]:
    """``{flag: [value, ...]}`` for every `--flag value` / `--flag=value` / bare `--flag`."""

    out: dict[str, list[str | None]] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" in token:
                flag, value = token.split("=", 1)
                out.setdefault(flag, []).append(value)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.setdefault(token, []).append(tokens[i + 1])
                i += 1
            else:
                out.setdefault(token, []).append(None)
        i += 1
    return out


def _one_value(values: dict[str, list[str | None]], flag: str) -> str | None:
    """The single value of `flag`, or None when absent; twice is a refusal."""

    got = values.get(flag)
    if not got:
        return None
    if len(got) > 1:
        refuse(f"--recipe-command carries {flag} {len(got)} times; a recipe states each value once")
    return got[0]


def _int_value(values: dict[str, list[str | None]], flag: str) -> int | None:
    raw = _one_value(values, flag)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        refuse(f"--recipe-command gives {flag} the non-integer value {raw!r}")


def assert_recipe_matches_axes(receipt: dict[str, Any], command: str) -> None:
    """The recipe line rebuilds THIS child: tag, every axis flag, every VALUE.

    The ledger seals the string verbatim and hashes it into `recipe_sha256`, and
    CHECKPOINT_POLICY R4 makes it the rebuild path after a DROP.  A flag name is
    not enough: the receipt records the values (`keep`, `child_vocab`,
    `excised`, `child_kv_heads` + `arm`, `child_geometry`, `child_head_dim` -
    `rotary_dim`), so each is compared with the value after its flag and any
    disagreement is printed with both numbers.  A shape flag the receipt has no
    axis for is refused too: that recipe built a different child.
    """

    try:
        tokens = shlex.split(command)
    except ValueError as error:
        refuse(f"--recipe-command {command!r} does not tokenise: {error}")
    tag = str(receipt["tag"])
    if tag not in tokens:
        refuse(
            f"--recipe-command {command!r} does not carry the tag {tag!r} as "
            "its own word; a recipe that names another child cannot rebuild this one"
        )
    values = _flag_values(tokens)
    accounted: set[str] = set()

    def mismatch(label: str, flag: str, expected: Any, got: Any) -> NoReturn:
        refuse(
            f"the receipt's {label!r} axis says {flag} should be {expected!r} but "
            f"--recipe-command says {got!r}; the recipe does not rebuild the child "
            "it is filed under"
        )

    for axis in receipt["axes"]:
        label = str(axis["axis"])
        flags = AXIS_FLAGS.get(label)
        if flags is None:
            refuse(
                f"axis {label!r} has no known qf_build flag in AXIS_FLAGS, so "
                "the recipe command cannot be checked against it"
            )
        accounted.update(flags)
        if label == "intermediate_size":
            got = _int_value(values, "--mlp-keep")
            if got is None or got != int(axis["keep"]):
                mismatch(label, "--mlp-keep", int(axis["keep"]), got)
        elif label == "vocab_size":
            got = _int_value(values, "--vocab-keep")
            if got is None or got != int(axis["child_vocab"]):
                mismatch(label, "--vocab-keep", int(axis["child_vocab"]), got)
        elif label == EXCISION_AXIS:
            raw = _one_value(values, "--excise")
            if raw is None:
                mismatch(label, "--excise", sorted(int(i) for i in axis["excised"]), None)
            try:
                got_list = sorted(int(v) for v in raw.split(",") if v.strip())
            except ValueError:
                refuse(f"--recipe-command gives --excise the non-integer list {raw!r}")
            if got_list != sorted(int(i) for i in axis["excised"]):
                mismatch(label, "--excise", sorted(int(i) for i in axis["excised"]), got_list)
        elif label in ("num_key_value_heads", "num_key_value_heads+num_attention_heads"):
            got = _int_value(values, "--kv-heads")
            if got is None or got != int(axis["child_kv_heads"]):
                mismatch(label, "--kv-heads", int(axis["child_kv_heads"]), got)
            arm = _one_value(values, "--kv-arm") or "kv-only"      # qf_build's default
            if arm != str(axis.get("arm")):
                mismatch(label, "--kv-arm", str(axis.get("arm")), arm)
        elif label in ("linear_num_value_heads", "linear_value_head_dim"):
            child = axis.get("child_geometry") or {}
            parent = axis.get("parent_geometry") or {}
            present = [flag for flag, _field in GDN_FLAG_FIELDS if flag in values]
            if not present:
                mismatch(label, "--gdn-*", child, None)
            for flag, field in GDN_FLAG_FIELDS:
                got = _int_value(values, flag)
                if got is None:
                    # qf_gdn defaults an absent flag to the parent's value, so the
                    # child must then still carry the parent's value there.
                    if int(child[field]) != int(parent[field]):
                        mismatch(label, flag, int(child[field]), None)
                elif got != int(child[field]):
                    mismatch(label, flag, int(child[field]), got)
        elif label == "head_dim (NoPE coordinates)":
            keep_nope = int(axis["child_head_dim"]) - int(axis["rotary_dim"])
            got = _int_value(values, "--nope-keep")
            if got is None or got != keep_nope:
                mismatch(label, "--nope-keep", keep_nope, got)
        elif label == "terminal_gain":
            if "--terminal-gain" not in values:
                mismatch(label, "--terminal-gain", "present", "absent")
    stray = sorted(flag for flag in ALL_AXIS_FLAGS if flag in values and flag not in accounted)
    if stray:
        refuse(
            f"--recipe-command carries {stray} but the receipt has no axis those "
            "flags build; the recipe built a different child"
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
    # digest to `digest` and then REBINDS the same name inside the
    # `--lineage ID:DIGEST` loop (`ancestor, _, digest = ...`), so the value that
    # reaches `"directory_sha256": digest` -- and the value printed in the plan,
    # and the value written into the ledger row -- is the LAST lineage entry's
    # digest instead of the directory's.  Reproduced against the fixture
    # campaign: three different fixture children registered with
    # `--lineage EVIDENCE-...:eeee...` all sealed `directory_sha256 = "e" * 64`.
    # Until that shadow is fixed in the ledger lane, this file states the
    # evidence manifest in the NOTES, where a wrong value cannot be mistaken for
    # a digest of the bytes, and the registered parent still reaches the lineage
    # block through --parent-checkpoint-id (which is appended after the loop and
    # does not rebind anything).
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
    parser.add_argument("--wave", default="oc01", help="ocNN; the id is OC<NN>-F-<code>-<TAG>")
    parser.add_argument("--parent-code", default=None,
                        help="R1 parent code (e.g. Q35-4B); needed only when the receipt "
                             "carries none and the parent directory name is not known here")
    parser.add_argument("--parent-checkpoint-id", default=None,
                        help="the registered parent (default PARENT-<parent code>); "
                             "'' to omit the lineage link, which is then said in the notes")
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
    parser.add_argument("--strict-receipt", action="store_true",
                        help="refuse (instead of noting) a receipt whose passed_through "
                             "statement the parent headers contradict")
    args = parser.parse_args(argv)
    match = _WAVE.match(args.wave)
    if not match:
        parser.error(f"--wave {args.wave!r} is not ocNN (e.g. oc01)")
    args.wave_number = match.group(1)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work.resolve()
    ledger_lane = (args.ledger_lane or (args.root / "ledger")).resolve()
    child = (args.child or (work / "children" / f"QF-{args.tag}")).resolve()
    recipe = (args.recipe or (work / "bodyq.sh")).resolve()

    sys.path.insert(0, str(ledger_lane))
    try:
        import oc_seal  # noqa: E402  -- the ledger lane's pure-stdlib header reader
        import oc_checkpoint  # noqa: E402  -- `current` / `resolve_path`, the registry reader
    except ImportError as error:
        refuse(f"cannot import the ledger lane from {ledger_lane}: {error}")

    receipt = read_receipt(child / RECEIPT_NAME)
    if str(receipt["tag"]) != args.tag:
        refuse(f"{child}'s receipt is tagged {receipt['tag']!r}, not {args.tag!r}")
    assert_recipe_matches_axes(receipt, args.recipe_command)

    axis_labels = [str(a["axis"]) for a in receipt["axes"]]
    if total_axis_removed(receipt) == 0:
        refuse(
            f"this child removed no parameters (axes {axis_labels} report "
            "parameters_removed 0): a values-only repair such as terminal_gain is a "
            "repair, not a fold; oc_checkpoint stage fold requires >=1 --component "
            "and a smaller child, so it cannot be registered as a fold -- it needs a "
            "Cyril ruling on a repair/no-fold stage"
        )

    parent_directory = str(receipt["parent_model"])
    parent_code, parent_code_source, parent_code_note = resolve_parent_code(
        receipt, args.parent_code, args.parent_checkpoint_id or None
    )
    checkpoint_id = f"OC{args.wave_number}-F-{parent_code}-{args.tag}"
    if args.parent_checkpoint_id is None:
        parent_checkpoint_id: str | None = f"PARENT-{parent_code}"
    else:
        parent_checkpoint_id = args.parent_checkpoint_id or None

    parent_counts = tensor_counts(Path(parent_directory), oc_seal)
    child_counts = tensor_counts(child, oc_seal)
    header_parent = sum(int(v) for v in parent_counts.values())
    header_child = sum(int(v) for v in child_counts.values())

    if parent_checkpoint_id:
        parent_record = assert_registered_parent(
            oc_checkpoint,
            root=args.root,
            parent_checkpoint_id=parent_checkpoint_id,
            parent_code=parent_code,
            parent_directory=parent_directory,
            header_parent=header_parent,
        )
        parent_link = (
            f"lineage parent {parent_checkpoint_id} seals directory "
            f"{parent_record['directory']} sha256 {parent_record['directory_sha256']} "
            f"({int(parent_record['parameter_count']):,} header parameters) = the "
            "receipt's parent_model and the header parent above"
        )
    else:
        parent_record = None
        parent_link = "NO registered parent link (--parent-checkpoint-id '' was passed)"

    extra_notes: list[str] = []
    if EXCISION_AXIS in axis_labels:
        pieces, passed_through, dropped_parent_only, excision_note = excision_pieces(
            receipt, parent_counts, child_counts
        )
        extra_notes.append(excision_note)
        granularity = "excised layer (asserted; C3/C4 re-indexes, so no prefix is true)"
    else:
        granularity, pieces = choose_granularity(receipt, parent_counts, child_counts)
        passed_through = unchanged_rollups(receipt)
        dropped_parent_only = sorted(set(parent_counts) - set(child_counts))
    claimed_saving = sum(p.saving for p in pieces)

    contradiction = passed_through_contradiction(receipt, dropped_parent_only)
    if contradiction and args.strict_receipt:
        refuse(contradiction + " (--strict-receipt)")

    log(f"child            {child}")
    log(f"checkpoint id    {checkpoint_id}")
    log(f"parent code      {parent_code}  (from {parent_code_source})")
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
    log(f"header parent    {parent_directory}  (receipt parent_model)")
    if parent_record is not None:
        log(
            f"lineage parent   {parent_checkpoint_id} -> {parent_record['directory']}  "
            f"directory_sha256 {parent_record['directory_sha256']}  "
            f"parameter_count {int(parent_record['parameter_count']):,}  (same directory: yes)"
        )
    else:
        log("lineage parent   (none; --parent-checkpoint-id '')")
    if dropped_parent_only:
        log(
            f"parent-only      {len(dropped_parent_only)} tensors the child does not carry: "
            f"{dropped_parent_only[:4]}{'...' if len(dropped_parent_only) > 4 else ''}"
        )
    if contradiction:
        log(f"receipt          {contradiction}")
    log(f"granularity      {granularity}  ({len(pieces)} components)")
    for piece in pieces[:16]:
        log(f"  component      {piece!r}")
    if len(pieces) > 16:
        log(f"  component      ... {len(pieces) - 16} more")
    log(f"passed through   {passed_through}")

    notes = build_notes(
        receipt,
        granularity=granularity,
        dropped_parent_only=dropped_parent_only,
        header_parent=header_parent,
        header_child=header_child,
        claimed_saving=claimed_saving,
        parent_code_note=parent_code_note,
        parent_link=parent_link,
        extra=extra_notes,
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
        parent_checkpoint_id=parent_checkpoint_id,
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
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    if completed.returncode != 0:
        if "is already registered" in completed.stderr:
            # A re-run after a reboot mid-commit, or after the queue lost the
            # rc: the chain exists.  Idempotent means "verify it, and say so",
            # which also catches a sealed record whose ledger row never landed
            # (verify names the `repair-row` to run).
            return verify_existing(
                python=args.python, ledger_lane=ledger_lane, root=args.root,
                checkpoint_id=checkpoint_id,
            )
        log(f"oc_checkpoint.py register exited {completed.returncode}")
        return completed.returncode
    if args.confirm and not args.no_postcheck:
        return postcheck(oc_seal, args.root, checkpoint_id, child)
    return 0


def verify_existing(
    *, python: str, ledger_lane: Path, root: Path, checkpoint_id: str
) -> int:
    log(f"{checkpoint_id} is already registered; verifying the chain instead of refusing")
    verify = [
        python, str(ledger_lane / "oc_checkpoint.py"), "--root", str(root),
        "verify", "--checkpoint-id", checkpoint_id,
    ]
    print(" ".join(shlex.quote(part) for part in verify), flush=True)
    completed = subprocess.run(verify)
    if completed.returncode == 0:
        log(f"ALREADY REGISTERED: {checkpoint_id} verifies clean; nothing to do")
        return 0
    log(
        f"ALREADY REGISTERED but verify exited {completed.returncode}: read the PROBLEM "
        "lines above (a missing ledger row is repaired with `oc_checkpoint.py --root "
        f"{root} repair-row --checkpoint-id {checkpoint_id} --confirm`)"
    )
    return completed.returncode


def postcheck(oc_seal: Any, root: Path, checkpoint_id: str, child: Path) -> int:
    """Re-read the sealed record and re-hash the child.  Trust the row, not the run.

    `register` prints a plan and then a `sealed <path>`, and both come out of the
    same in-memory dict; neither of them is evidence that what landed on disk
    describes these bytes.  This re-opens the record the registry actually holds
    and re-computes the directory digest from the files, which is how the
    `--lineage` variable shadow above was found in the first place.

    A failure here is NOT a refusal: the record and the ledger row are already
    sealed.  It exits 4 with a `SEALED: ... POSTCHECK FAILED` line so the queue
    does not file the child under "register refused" while the registry holds
    it, and names the verify to run.
    """

    registry = Path(root) / "registry" / checkpoint_id
    records = sorted(registry.glob(f"{checkpoint_id}-*-register.json"))
    verify_hint = (
        f"the registry row exists; run oc_checkpoint.py --root {root} verify "
        f"--checkpoint-id {checkpoint_id}"
    )
    if not records:
        print(
            f"SEALED: {checkpoint_id} -- POSTCHECK FAILED: register reported success "
            f"but {registry} holds no record; {verify_hint}",
            flush=True,
        )
        return EXIT_POSTCHECK_FAILED
    try:
        sealed = json.loads(records[-1].read_text(encoding="utf-8"))["checkpoint"]
        live, _entries = oc_seal.directory_digest(child)
    except (OSError, ValueError, KeyError, oc_seal.OcSealError) as error:
        print(
            f"SEALED: {records[-1]} -- POSTCHECK FAILED: cannot re-read the record or "
            f"re-hash {child}: {error}; {verify_hint}",
            flush=True,
        )
        return EXIT_POSTCHECK_FAILED
    if sealed["directory_sha256"] != live:
        print(
            f"SEALED: {records[-1]} -- POSTCHECK FAILED: the record seals "
            f"directory_sha256 {sealed['directory_sha256']} but {child} digests {live}; "
            "the record does not name the bytes it claims to; do not review this "
            f"checkpoint until the registry row is repaired; {verify_hint}",
            flush=True,
        )
        return EXIT_POSTCHECK_FAILED
    log(
        f"postcheck ok: {records[-1].name} names the child's own digest "
        f"{live[:16]}, net_ratio {float(sealed['net_ratio']):.10f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
