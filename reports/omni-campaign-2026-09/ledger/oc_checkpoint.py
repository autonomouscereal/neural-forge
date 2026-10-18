#!/usr/bin/env python3
"""The Chimera Omni checkpoint registry: register, clone, review, delete.

Cyril's ninth ruling of 2026-09-01, verbatim, is the whole specification of
this file: *"[Cyril quote withheld]"*  `CHECKPOINT_POLICY.md` beside this tree states the rules in prose;
this file is the only thing allowed to execute them.

Shape of the registry
---------------------
Every checkpoint is a directory of weights plus an APPEND-ONLY chain of sealed
records under ``<registry>/<checkpoint_id>/<checkpoint_id>-<seq>-<event>.json``.
A record is never edited: a review, a status change or a delete writes a NEW
sealed record that names the one it supersedes, exactly as a re-reviewed LFM2
cell became a new ``-a6review`` instance rather than an edit.  The current state
of a checkpoint is the last record in its chain; the history is the chain.

Every record also appends one row to the campaign ledger
(``neural-forge-omni-checkpoint-ledger-v1``), so a reader who has only the
ledger can still see every checkpoint, every ratio, both review columns and
every deletion, and a reader who has only the registry can re-derive the rows.

Statuses, and who decides them
------------------------------
``candidate`` (registered, not yet reviewed) -> ``partially_capable``,
``unmeasured`` or ``bad`` (reviewed), and ``cloned`` for a pre-training clone.
The status is COMPUTED from the review counts the reviewer supplies, for the
same reason `behavior_verdict_for` computes the LFM2 verdict: the grade is a
judgement of text, but the line the grade falls on must not move after someone
sees which side it landed on.  ``--status`` may only ASSERT agreement.

``unmeasured`` is the estate's artifact-tolerant rule made structural.  Zero
canonical signal is NOT evidence of a bad child if every miss was a loop, a
cap-ride, an empty generation or a missing EOS: that child was not graded, it
was not run properly.  Only a CONTENT miss -- readable, on-register, wrong --
makes a child ``bad``, and ``bad`` is the only status the delete path accepts.
An artifact may not authorise an irreversible `rm -rf` of weights.

The line itself is Cyril's proposed one (R5): *at least partially capable = at
least one canonical signal lane on ANY surface* -- the 15-case pack, bank v2,
or a media-out artifact with non-zero content.  It is a proposed decision line
for him, not a gate invented here; `--partially-capable-floor` exists so that
re-ruling it is one argument and not an edit.

Two numbers nobody may type
---------------------------
``net_ratio`` is the campaign's central claim, and it used to be possible to
assert it: ``--parameter-count 256 --parent-parameter-count 4096`` sealed a 16x
fold that had folded nothing, and ``verify --deep`` then re-divided the same two
typed integers and returned 0 -- a tautology dressed as a check, while the free
header read that would have caught it was skipped in favour of re-hashing every
byte.  Now, whenever the directory holds ``*.safetensors``, BOTH counts are read
from the headers and a typed count that disagrees is a refusal; a count that
cannot be measured at all needs ``--allow-asserted-parameter-counts`` and is
labelled ``asserted-by-operator`` in the record, the ledger row, SCORECARD.md
and the review package.  ``verify`` re-derives both counts on every run.

Paths this tool will not touch
------------------------------
``PROTECTED_PREFIXES`` -- ``/mnt`` (the CIFS share) and ``/home/cereal/models``
(the source model library).  R6 says neither is ever deleted, and this registry
is the only path that may delete a checkpoint, so a rule it does not enforce is
a rule nobody enforces.  ``register`` refuses such a ``--directory`` BEFORE
``directory_digest`` walks it (one typo used to byte-hash the share at under
3 MB/s), ``clone`` refuses such a destination, and R4 clause (iv-c) refuses the
unlink.  ``--stage parent`` under the model library is one exception,
because that is exactly what a registered parent is and ``delete`` refuses stage
``parent`` outright.  ``/mnt/lexar`` (``REGISTRABLE_UNDER_PROTECTED``) is the
other: it is the local USB SSD, not the CIFS share the ``/mnt`` rule was written
for, so ``register`` and ``clone`` may place a working checkpoint there -- but
``delete`` clause (iv-c) still refuses EVERYTHING under ``/mnt``, Lexar
included, and ``/mnt/ai_data`` stays refused for register and clone too.

Mutating commands
-----------------
``register``, ``clone``, ``record-review``, ``set-status`` and ``delete`` all
PRINT THE PLAN and do nothing without ``--confirm``.  ``delete`` additionally
walks the R4 conditions one at a time and prints PASS/REFUSE for each, because
"it refused" is not a finding unless you can see which clause refused.

Usage::

    oc_checkpoint.py register --registry R --ledger L --checkpoint-id OC01-F-Q35-4B-C2S4L03 \\
        --stage fold --parent-code Q35-4B --wave oc01 --directory DIR \\
        --parent-directory PDIR --parameter-count auto --parent-parameter-count auto \\
        --recipe recipe.json --recipe-command "ocq.sh C2S4L03" \\
        --component "mlp:1234:300" --confirm
    oc_checkpoint.py clone --registry R --ledger L --source ID --stage-about-to-run D --confirm
    oc_checkpoint.py record-review --registry R --ledger L --checkpoint-id ID --review-json rev.json --confirm
    oc_checkpoint.py delete --registry R --ledger L --checkpoint-id ID \\
        --reason "..." --rebuild-command "..." --retained-source ID_OR_PATH --confirm
    oc_checkpoint.py list|show|verify --registry R

Delta checkpoints (1.1.0)::

    oc_checkpoint.py register --checkpoint-id OC02-D-Q38-27BU-OMNI1D1 --stage distill \\
        --directory DELTA_DIR --delta-base OC02-F-Q38-27BU-OMNI1.clone-pre-D1 \\
        --parent-directory PDIR --parent-code Q38-27BU --recipe-command "..." --confirm
    oc_checkpoint.py delta-compose --checkpoint-id ID [--deep] [--plan-out plan.json]
    oc_checkpoint.py package-delta --checkpoint-id ID --out DIR --confirm

``DELTA_DIR/DELTA.json`` (``neural-forge-omni-delta-recipe-v1``) names the base
and its pinned digest and lists ``replace`` / ``lora`` / ``scale`` entries on
base tensor names, each with an optional ``[[dim, start, stop], ...]`` slice.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from oc_ledger import (
    CHECKPOINT_ROW_COLUMNS,
    CHECKPOINT_ROW_SCHEMA,
    OcLedgerError,
    append_rows,
)
from oc_seal import (
    CAMPAIGN,
    OcSealError,
    canonical_sha256,
    checkpoint_shards,
    directory_digest,
    file_sha256,
    human_bytes,
    now_utc,
    oc_root,
    prefix_parameter_count,
    read_sealed,
    require_sha256,
    safetensors_header,
    safetensors_parameter_count,
    safetensors_tensor_counts,
    seal,
    verify_seal,
    write_sealed,
    WriterLock,
)

#: 1.1.0 (2026-10-12): delta checkpoints -- `register --delta-base`,
#: `delta-compose`, `package-delta`, delta-aware `clone` and `verify`.  Records
#: written by 1.0 carry no version and are read unchanged.
TOOL_VERSION = "oc_checkpoint 1.1.0"

RECORD_SCHEMA = "neural-forge-omni-checkpoint-record-v1"
CLONE_RECEIPT_SCHEMA = "neural-forge-omni-clone-receipt-v1"
DELTA_RECIPE_SCHEMA = "neural-forge-omni-delta-recipe-v1"
DELTA_PLAN_SCHEMA = "neural-forge-omni-delta-plan-v1"

#: The composition recipe every delta directory carries at its top level.  It is
#: inside the directory, so the delta's own `directory_sha256` covers it.
DELTA_RECIPE_NAME = "DELTA.json"

#: What a delta may do to a base tensor, applied in recipe order, in fp32, with
#: one cast back to the base dtype at the end (the D4 rule: fold once, cast once).
#:   replace  region <- delta tensor                       (trained subsets)
#:   lora     region += (alpha / rank) * B @ A             (adapters)
#:   scale    region *= g, broadcast along `dim`           (per-column gains)
DELTA_OPS = ("replace", "lora", "scale")

#: A chain deeper than this is a cycle or a mistake, not a campaign.
MAX_DELTA_CHAIN = 32

STAGES = ("fold", "distill", "finetune", "parent", "clone")
STAGE_LETTER = {"fold": "F", "distill": "D", "finetune": "T"}
STATUSES = ("candidate", "partially_capable", "unmeasured", "bad", "cloned")
EVENTS = ("register", "clone", "review", "status", "delete")

#: Only a `bad` checkpoint may be dropped, and `bad` is the only status the R4
#: path accepts.  `unmeasured` exists so that "no signal" and "no measurement"
#: are not the same word: see `status_for`.
DELETABLE_STATUS = "bad"

#: The LFM2 sealer's verdict vocabulary, carried unchanged so a ledger reader
#: does not learn a second set of words for the same readings.
BEHAVIOR_VERDICTS = (
    "alive-parent-band",
    "alive-degraded",
    "impaired",
    "dead",
    "cannot-express-4x",
)
DEGENERACY_READINGS = ("none", "lane-scoped", "general", "unstated")
SMOKE_VERDICTS = ("pending", "coherent", "incoherent")

#: Paths this tooling may not register a working checkpoint under, may not
#: clone into, and may never delete.  R6 of CHECKPOINT_POLICY.md says the source
#: model library and the CIFS share are never deleted, but the registry is the
#: ONLY path that may delete a checkpoint, so a rule it does not enforce is a
#: rule nobody enforces.  Two concrete failures this closes:
#:
#: 1. `--directory /mnt/ai_data/...` -- one typo -- makes `register` and
#:    `verify --deep` recursively walk and byte-hash the CIFS share, which reads
#:    at under 3 MB/s.  The standing estate instruction is never to walk it;
#: 2. a checkpoint registered on the share or in the model library, reviewed
#:    `bad` and packaged, would have `rmtree`'d a SOURCE MODEL.
#:
#: `--stage parent` is the one exception, and only under the model library: a
#: registered parent is a provenance-verified local copy of a source model, and
#: `delete` refuses stage `parent` outright (clause iv), so nothing under it is
#: reachable by an unlink.  A path is matched by PARTS, never by string prefix,
#: so `/mnt/ai_data_scratch` is not `/mnt/ai_data`.
PROTECTED_PREFIXES = ("/mnt", "/home/cereal/models")

#: Of those, the ones a `--stage parent` registration may live under.
PARENT_ONLY_PREFIXES = ("/home/cereal/models",)

#: Sub-trees of a protected prefix where `register` and `clone` -- and ONLY
#: those two -- may place a working checkpoint of any stage.  `/mnt/lexar` is
#: the server's local USB SSD (exFAT on /dev/sdb2), not the CIFS share
#: (`/mnt/ai_data`) the `/mnt` rule exists for: walking it does not crawl at
#: 3 MB/s, and the Sep-07 S3 trunk child was built there because root was full.
#: Narrowed 2026-10-09 for that registration.  `delete` does NOT consult this:
#: clause (iv-c) matches `PROTECTED_PREFIXES` directly, so a Lexar record is
#: still never unlinked by this tool.  Matched by parts, like everything else,
#: so `/mnt/lexarfoo` is not `/mnt/lexar`.
REGISTRABLE_UNDER_PROTECTED = ("/mnt/lexar",)


def resolve_path(path: Path | str) -> Path:
    """`path` made absolute with every `..`, `.` and SYMLINK resolved away.

    `Path.absolute()` only prepends the working directory.  It does not collapse
    `..` and it does not follow a symlink, so both of these walked straight
    through the protected-prefix guard and `_assert_directory_is_free`:

    1. `--directory $OC_ROOT/../models/Qwen3.5-4B` has parts
       `(..., 'omni-campaign', '..', 'models', 'Qwen3.5-4B')`, which do not
       start `('/', 'home', 'cereal', 'models')` -- so the guard said the source
       model library was not the source model library, `directory_digest` walked
       it, and a bad review on that record would have `rmtree`'d a parent;
    2. a symlink (`$OC_ROOT/share -> /mnt/ai_data`) hid the CIFS share behind a
       campaign-looking path, and `verify --deep` then byte-hashed the share at
       under 3 MB/s.

    The same blindness let ONE directory be registered twice under two spellings
    (`children/X` and `children/../children/X`, or through an alias symlink),
    which is precisely the state `_assert_directory_is_free` exists to refuse:
    a DROP of either id `rm -rf`s the other's weights.

    `realpath` resolves the symlinks and normalises the rest; `normpath` is
    stated as well because the resolution of a path whose tail does not exist
    yet (a clone destination) is purely lexical, and that is the case where the
    `..` collapse has to happen anyway.  Nothing here requires the path to
    exist, and nothing here LISTS a directory -- so this is safe to call before
    the guard that keeps us off the share: the cost is one bounded `lstat` per
    component (measured at 1.9 s for a path under the CIFS mount), never a walk.
    """

    return Path(os.path.normpath(os.path.realpath(os.path.expanduser(str(path)))))


def _posix_parts(path: Path | str) -> tuple[str, ...]:
    """RESOLVED POSIX path parts, with a Windows drive letter stripped.

    The prefixes are the SERVER's paths, and that is where the guard has to
    bite.  A workstation `Path("/mnt/ai_data/x").absolute()` is
    `C:/mnt/ai_data/x`, whose parts start `('C:\', 'mnt')` and would never
    match `('/', 'mnt')` -- so the guard would silently do nothing on the host
    the tests run on, which is the same as not having it.  Dropping the drive
    makes the check mean the same thing on both hosts; the cost is that
    `C:/mnt/...` is protected on a workstation too, which stores no checkpoints
    there anyway.

    Resolution happens FIRST (`resolve_path`), because a prefix match on an
    unresolved path is a match on a spelling rather than on a location.
    """

    text = str(resolve_path(path)).replace("\\", "/")
    drive, sep, rest = text.partition(":")
    if sep and len(drive) == 1 and drive.isalpha() and rest.startswith("/"):
        text = rest
    return PurePosixPath(text).parts


def _protected_prefix(
    directory: Path, prefixes: Sequence[str] = PROTECTED_PREFIXES
) -> str | None:
    """The first protected prefix `directory` sits under, by path PARTS.

    By parts, never by string prefix: `/mnt/ai_data_scratch` is not under
    `/mnt/ai_data`, and a `startswith` would say it was.
    """

    target = _posix_parts(directory)
    for prefix in prefixes:
        parts = PurePosixPath(prefix).parts
        if target[: len(parts)] == parts:
            return prefix
    return None


def assert_not_protected(
    directory: Path | str, *, what: str, stage: str | None = None
) -> None:
    """Refuse a protected path, before anything walks or unlinks it.

    Matched on the RESOLVED path, and the message names both spellings when
    they differ: a refusal that quotes a location the operator never typed
    reads as a bug in the tool rather than as the guard doing its job.  Pass
    what the operator actually wrote.
    """

    resolved = resolve_path(directory)
    prefix = _protected_prefix(resolved)
    if prefix is None:
        return
    if stage == "parent" and _protected_prefix(resolved, PARENT_ONLY_PREFIXES):
        return
    # register/clone only: `delete` clause (iv-c) never calls this function.
    if _protected_prefix(resolved, REGISTRABLE_UNDER_PROTECTED) is not None:
        return
    named = (
        str(resolved)
        if str(directory) == str(resolved)
        else f"{directory} (which resolves to {resolved})"
    )
    why = (
        "the source model library holds provenance-verified copies of the source "
        "models, and only `--stage parent` may be registered there"
        if prefix in PARENT_ONLY_PREFIXES
        else "that is the CIFS share: walking it recursively reads at under "
        "3 MB/s, and R6 forbids deleting anything on it"
    )
    raise OcCheckpointError(
        f"{what} {named} is under the protected prefix {prefix!r} -- {why}.  "
        "Campaign checkpoints live under $OC_ROOT (/home/cereal/omni-campaign)."
    )


#: The frozen review bank's size, and the smallest bank a review may claim.
#: 16 lanes, one of which is the safety lane and never scores.
BANK1_LANES = 16
MIN_BANK_LANES = 2

#: The >=4x per component rule, as the receipt-verified SHAPE claim it is.  A
#: component that stores more than a quarter of its parent's parameters has not
#: been folded 4x, and that is arithmetic, not a grade.
MAX_COMPONENT_RATIO = 0.25

#: The parent codes R1 names, as patterns, so an id can be SPLIT rather than
#: guessed.  A greedy `(.+)-(TAG)` split reads `OC01-F-Q35-4B` -- an id with no
#: tag at all -- as parent `Q35` tag `4B`, which is a different checkpoint than
#: the one whose name that is.  An id whose parent code is not one of these
#: must pass `--parent-code` so the split is stated and not inferred.
PARENT_CODE_PATTERNS = (
    r"Q35-0\.8B", r"Q35-2B", r"Q35-4B", r"Q35-9B",
    r"Q38-27BU", r"Q38-27B",
    r"K3-L\d+", r"DK3-[A-Za-z0-9.]+",
    r"H3", r"ACE", r"QIMG", r"HY3D", r"QVL32",
)

_CHILD_ID = re.compile(r"^OC(\d{2})-([FDT])-(.+)-([A-Z0-9]{1,16})$")
_KNOWN_CHILD_ID = re.compile(
    r"^OC(\d{2})-([FDT])-(" + "|".join(PARENT_CODE_PATTERNS) + r")-([A-Z0-9]{1,16})$"
)
_PARENT_ID = re.compile(r"^PARENT-(.+)$")
_CLONE_ID = re.compile(r"^(.+)\.clone-pre-([DT])(\d+)$")
_TAG = re.compile(r"^[A-Z0-9]{1,16}$")

#: A record file name.  `history()` reads ONLY names that match this, so a
#: stray dotfile, an editor backup or an in-flight marker that lands in a
#: record directory cannot make the whole chain unreadable.
_RECORD_NAME = re.compile(r"^(?P<cid>.+)-(?P<seq>\d{4})-(?P<event>[a-z]+)\.json$")


class OcCheckpointError(RuntimeError):
    """A registry operation refused."""


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def parse_checkpoint_id(
    checkpoint_id: str, stage: str, parent_code: str | None = None
) -> dict[str, Any]:
    """R1's naming rule, enforced rather than described.

    ``OC<NN>-<F|D|T>-<PARENTCODE>-<TAG>`` for a built child, ``PARENT-<CODE>``
    for a registered parent, ``<source>.clone-pre-<D|T><k>`` for a clone.  The
    tag is anchored at the end and carries no dash, which is what lets a parent
    code like ``Q35-0.8B`` keep its own dashes and dot.

    The split is NOT greedy.  ``OC01-F-Q35-4B`` -- an id someone forgot to put
    a tag on -- used to parse as parent ``Q35`` tag ``4B`` and register cleanly
    under a name that means something else.  So the parent code must either be
    passed in (``--parent-code``, and then the id must literally start with
    ``OC<NN>-<L>-<code>-``) or be one of R1's own codes; anything else is
    refused with the instruction to state it.
    """

    if stage == "clone":
        match = _CLONE_ID.match(checkpoint_id)
        if not match:
            raise OcCheckpointError(
                f"{checkpoint_id!r} is not a clone name "
                "(<source>.clone-pre-<D|T><k>)"
            )
        return {"clone_of": match.group(1), "next_stage": match.group(2),
                "clone_index": int(match.group(3)), "tag": None, "wave": None}
    if stage == "parent":
        match = _PARENT_ID.match(checkpoint_id)
        if not match:
            raise OcCheckpointError(f"{checkpoint_id!r} is not PARENT-<CODE>")
        return {"parent_code": match.group(1), "tag": None, "wave": None}
    match = _CHILD_ID.match(checkpoint_id)
    if not match:
        raise OcCheckpointError(
            f"{checkpoint_id!r} does not match OC<NN>-<F|D|T>-<PARENTCODE>-<TAG> "
            "(TAG is 1-16 of A-Z0-9 and carries no dash)"
        )
    number, letter = match.group(1), match.group(2)
    if letter != STAGE_LETTER[stage]:
        raise OcCheckpointError(
            f"{checkpoint_id!r} carries stage letter {letter} but --stage is "
            f"{stage} ({STAGE_LETTER[stage]})"
        )
    if parent_code:
        prefix = f"OC{number}-{letter}-{parent_code}-"
        if not checkpoint_id.startswith(prefix):
            raise OcCheckpointError(
                f"{checkpoint_id!r} does not start with {prefix!r}: --parent-code "
                f"{parent_code!r} and the id disagree about which part of the name "
                "is the parent and which is the tag"
            )
        code, tag = parent_code, checkpoint_id[len(prefix):]
    else:
        known = _KNOWN_CHILD_ID.match(checkpoint_id)
        if not known:
            raise OcCheckpointError(
                f"{checkpoint_id!r} cannot be split unambiguously: its parent code "
                f"is not one of R1's ({', '.join(PARENT_CODE_PATTERNS)}), so a "
                "greedy split could read the last field of a parent code as the "
                "tag.  Pass --parent-code <CODE> and the split is stated."
            )
        code, tag = known.group(3), known.group(4)
    if not _TAG.match(tag):
        raise OcCheckpointError(
            f"{checkpoint_id!r}: tag {tag!r} is not 1-16 of A-Z0-9 with no dash"
        )
    return {
        "wave": f"oc{number}",
        "parent_code": code,
        "tag": tag,
        "stage_letter": letter,
    }


# --------------------------------------------------------------------------
# registry i/o
# --------------------------------------------------------------------------


def record_dir(registry: Path, checkpoint_id: str) -> Path:
    return Path(registry).expanduser().absolute() / checkpoint_id


def chain_lock(registry: Path, checkpoint_id: str) -> Path:
    """The lock one mutating command holds while it extends a chain.

    It lives OUTSIDE the record directory (`<registry>/.locks/`), for the same
    reason the in-flight delete marker does: nothing that is not a sealed
    record may sit where the chain is read.
    """

    return Path(registry).expanduser().absolute() / ".locks" / f"{checkpoint_id}.lock"


def pending_marker(registry: Path, checkpoint_id: str) -> Path:
    """Where a delete says what it had in flight when it was interrupted."""

    return (
        Path(registry).expanduser().absolute() / ".pending" / f"{checkpoint_id}.json"
    )


def record_files(folder: Path, checkpoint_id: str) -> list[Path]:
    """Only files whose name IS a record name for this checkpoint.

    `folder.glob("*.json")` matches dotfiles on this interpreter, so an
    interrupted delete used to leave `.pending-delete.json` where the chain is
    read and every later `read_sealed` raised -- which took down the dashboard,
    `list`, `verify` and `package` for the WHOLE registry, and the recovery was
    to hand-delete a file inside the sealed registry.  A chain is now exactly
    the files that match `<id>-NNNN-<event>.json`; everything else is ignored.
    """

    out: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = _RECORD_NAME.match(path.name)
        if not match or match.group("cid") != checkpoint_id:
            continue
        if match.group("event") not in EVENTS:
            continue
        out.append(path)
    return out


def history(registry: Path, checkpoint_id: str) -> list[dict[str, Any]]:
    """Every sealed record for one checkpoint, in sequence order."""

    folder = record_dir(registry, checkpoint_id)
    if not folder.is_dir():
        return []
    records = [read_sealed(p) for p in record_files(folder, checkpoint_id)]
    records.sort(key=lambda r: (int(r["record_seq"]), str(r["sealed_at"])))
    seen = [int(r["record_seq"]) for r in records]
    if seen != list(range(1, len(seen) + 1)):
        raise OcCheckpointError(
            f"{checkpoint_id}: record sequence {seen} has a gap or a duplicate; "
            "the chain is append-only and must be contiguous"
        )
    for previous, current in zip(records, records[1:]):
        if current.get("supersedes_record_sha") != previous["receipt_sha"]:
            raise OcCheckpointError(
                f"{checkpoint_id}: record {current['record_seq']} does not name "
                f"record {previous['record_seq']} as the one it supersedes"
            )
    return records


def current(registry: Path, checkpoint_id: str) -> dict[str, Any]:
    chain = history(registry, checkpoint_id)
    if not chain:
        raise OcCheckpointError(f"{checkpoint_id} is not in the registry")
    return chain[-1]


def all_current(registry: Path) -> dict[str, dict[str, Any]]:
    root = Path(registry).expanduser().absolute()
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return out
    for folder in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        chain = history(root, folder.name)
        if chain:
            out[folder.name] = chain[-1]
    return out


# --------------------------------------------------------------------------
# status, computed
# --------------------------------------------------------------------------


def canonical_signal_total(review: Mapping[str, Any] | None) -> int:
    """Canonical signal lanes summed over every surface (R5's measure)."""

    if not review:
        return 0
    total = int(review.get("bank1", {}).get("canonical_signal", 0))
    bank2 = review.get("bank2") or {}
    total += int(bank2.get("canonical_signal", 0))
    total += sum(
        1 for artifact in (review.get("media_out") or [])
        if bool(artifact.get("non_empty"))
    )
    return total


def canonical_miss_totals(review: Mapping[str, Any] | None) -> tuple[int, int]:
    """``(artifact misses, content misses)`` summed over every surface.

    Artifact misses are counted so that "the child said nothing readable" can
    be told apart from "the child said something and it was wrong"; they are
    never counted FOR or AGAINST anything.  An empty media artifact is an
    artifact miss for exactly the same reason an empty generation is.
    """

    if not review:
        return 0, 0
    bank1 = review.get("bank1") or {}
    bank2 = review.get("bank2") or {}
    artifact = int(bank1.get("canonical_artifact_miss", 0)) + int(
        bank2.get("canonical_artifact_miss", 0)
    )
    content = int(bank1.get("canonical_content_miss", 0)) + int(
        bank2.get("canonical_content_miss", 0)
    )
    artifact += sum(
        1 for item in (review.get("media_out") or []) if not bool(item.get("non_empty"))
    )
    return artifact, content


def status_for(
    *, stage: str, review: Mapping[str, Any] | None, floor: int = 1
) -> tuple[str, str]:
    """The status the evidence implies, and the sentence that says why.

    ARTIFACT-TOLERANT GRADING, which is the estate's standing rule and R5's own
    words: *artifact-miss counts for NOTHING; content-miss is the only miss
    that counts AGAINST*.  A child whose every lane was a loop, a cap-ride, an
    empty generation or a missing EOS has produced NO measurement -- for or
    against -- and calling that `bad` would make an artifact the authority for
    an irreversible `rm -rf` of the weights.  So zero signal splits in two:

    * at least one CONTENT miss -> `bad` (something readable was measured and
      it was wrong), the only status R4 will delete;
    * no content miss at all, but artifact misses -> `unmeasured`.  It is not
      deletable by any path; it goes back for a re-run (longer cap, EOS fix,
      decode fix), and only the re-run can make it `bad`.
    """

    if stage == "clone":
        return "cloned", "pre-training clone; frozen, never trained in place"
    if review is None:
        return "candidate", "registered; no review record yet"
    signal = canonical_signal_total(review)
    if signal >= floor:
        return (
            "partially_capable",
            f"{signal} canonical signal lane(s) across all surfaces "
            f"(floor {floor}); goes to Cyril's review queue",
        )
    artifact, content = canonical_miss_totals(review)
    if content == 0 and artifact > 0:
        return (
            "unmeasured",
            f"0 canonical signal and 0 content misses across all surfaces, but "
            f"{artifact} artifact-miss lane(s) (loop / cap-ride / empty / missing "
            "EOS).  Artifacts measure nothing either way, so this child has NOT "
            "been graded and is not deletable: re-run it.",
        )
    return (
        DELETABLE_STATUS,
        f"{signal} canonical signal lanes across all surfaces (floor {floor}) with "
        f"{content} content miss(es) measured; eligible for retention DROP once "
        "every R4 condition holds",
    )


# --------------------------------------------------------------------------
# review payload validation
# --------------------------------------------------------------------------


def validate_review(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a review whose own counts do not add up.

    Nothing here grades anything.  It checks that the numbers a reviewer typed
    are internally consistent -- the intelligence lanes sum to the bank size,
    column A equals the sealed useful-or-partial count (`mx45_dualband`'s own
    refusal), column B is A minus held copy-backs and so cannot exceed it --
    because a mistyped count silently moves a status.
    """

    review = json.loads(json.dumps(dict(payload)))
    for field in ("reviewer", "bank1", "columns"):
        if field not in review:
            raise OcCheckpointError(f"review is missing {field!r}")

    def need(block: Mapping[str, Any], key: str, where: str) -> int:
        """A missing count is a REFUSAL with a sentence, not a KeyError.

        The CLI's contract is `REFUSED: <why>` and exit 3; a traceback out of
        here would be a malformed review reported as a crash.
        """

        if key not in block:
            raise OcCheckpointError(f"review is missing {where}.{key}")
        try:
            return int(block[key])
        except (TypeError, ValueError) as exc:
            raise OcCheckpointError(
                f"review {where}.{key} is {block[key]!r}, not an integer"
            ) from exc

    bank1 = review["bank1"]
    if not isinstance(bank1, Mapping):
        raise OcCheckpointError("review.bank1 is not an object")
    lanes_total = int(bank1.get("lanes_total", BANK1_LANES))
    if lanes_total < MIN_BANK_LANES:
        # `lanes_total: 1` used to make `intelligence` 0, which made every count
        # sum to 0, which passed every consistency check above and fell through
        # to `bad` with zero content misses.  Delete still refused it (clause
        # i-e), so nothing unsafe followed -- but the scorecard labelled an
        # unmeasured child `bad`, which is a reading nobody made.
        raise OcCheckpointError(
            f"bank1.lanes_total {lanes_total} is below {MIN_BANK_LANES}; a bank "
            "with no intelligence lanes measures nothing, and a review of it is "
            f"not a review (the frozen bank is {BANK1_LANES} lanes)"
        )
    intelligence = lanes_total - 1  # the safety lane never scores
    counts = [need(bank1, k, "bank1") for k in ("useful", "partial", "observed_miss")]
    if sum(counts) != intelligence:
        raise OcCheckpointError(
            f"bank1 useful+partial+observed_miss = {sum(counts)} but the bank "
            f"has {intelligence} intelligence lanes ({lanes_total} minus safety)"
        )
    useful_or_partial = counts[0] + counts[1]
    bank1["useful_or_partial"] = useful_or_partial
    canon = [
        need(bank1, k, "bank1")
        for k in ("canonical_signal", "canonical_artifact_miss", "canonical_content_miss")
    ]
    if sum(canon) != intelligence:
        raise OcCheckpointError(
            f"canonical boxes sum to {sum(canon)}, not {intelligence}; every "
            "lane goes in exactly one box (signal | artifact-miss | content-miss)"
        )
    if bank1.get("sealed_verdict") not in BEHAVIOR_VERDICTS:
        raise OcCheckpointError(
            f"bank1.sealed_verdict {bank1.get('sealed_verdict')!r} is not one of "
            f"{BEHAVIOR_VERDICTS}"
        )
    if "parent_useful_or_partial" not in bank1:
        raise OcCheckpointError(
            "bank1.parent_useful_or_partial has no default -- it comes from the "
            "parent's bands.json and is never guessed"
        )
    columns = review["columns"]
    if not isinstance(columns, Mapping):
        raise OcCheckpointError("review.columns is not an object")
    column_a = need(columns, "column_a", "columns")
    column_b = need(columns, "column_b", "columns")
    if column_a != useful_or_partial:
        raise OcCheckpointError(
            f"column_a {column_a} != sealed useful_or_partial {useful_or_partial}; "
            "column A IS the sealed per-case count and a recount that disagrees "
            "is a refusal, not a rounding"
        )
    if not 0 <= column_b <= column_a:
        raise OcCheckpointError(
            f"column_b {column_b} is not in [0, {column_a}]; B is A minus the "
            "lanes held only by a copy-back"
        )
    bank2 = review.get("bank2")
    if bank2 is not None and not isinstance(bank2, Mapping):
        # `"bank2": []` used to reach `.get` and raise AttributeError, i.e. a
        # malformed review reported as a traceback rather than as the
        # `REFUSED: <why>` exit 3 the CLI contract promises.
        raise OcCheckpointError(
            f"review.bank2 is {type(bank2).__name__}, not an object or null"
        )
    if bank2:
        b2_lanes = int(bank2.get("lanes_total", 0))
        if int(bank2.get("canonical_signal", 0)) > b2_lanes:
            raise OcCheckpointError("bank2.canonical_signal exceeds bank2.lanes_total")
    for artifact in review.get("media_out") or []:
        if not isinstance(artifact, Mapping):
            raise OcCheckpointError(
                f"media_out entry {artifact!r} is not an object"
            )
        if "non_empty" not in artifact:
            raise OcCheckpointError(
                f"media_out {artifact.get('prompt_id')!r} does not say whether the "
                "decoded artifact is non-empty"
            )
        # R5 says non-zero CONTENT, and `non_empty` is only `st_size > 0`: a
        # 44-byte WAV header, or a full-length track of pure silence, is a file
        # that exists and a lane that measured nothing.  When the runner
        # supplies an `audio_rms` (oc_package.media_diagnostics computes one for
        # every WAV), a silent artifact may not also claim to be non-empty.
        rms = artifact.get("audio_rms")
        if rms is not None and float(rms) <= 0.0 and bool(artifact.get("non_empty")):
            raise OcCheckpointError(
                f"media_out {artifact.get('prompt_id')!r} has audio_rms {rms} -- "
                "silence -- but is marked non_empty.  A file with no content is "
                "an ARTIFACT miss, not a signal lane; set non_empty false."
            )
        score = artifact.get("cyril_score")
        if score is not None and int(score) not in (1, 2, 3, 4, 5):
            raise OcCheckpointError(
                f"media_out {artifact.get('prompt_id')!r} cyril_score {score!r} is "
                "not null or 1..5"
            )
    if review.get("package_path") and not review.get("package_sha256"):
        # R4(v) is "the evidence is PACKAGED", and a path with no digest is a
        # claim about a folder that can be rebuilt under the same name with
        # different contents.  The two travel together or neither does.
        raise OcCheckpointError(
            "review names package_path but no package_sha256; the digest is the "
            "MANIFEST.json's own `package_sha256` and is what ties the review to "
            "the bytes a reader will open"
        )
    if review.get("package_sha256") is not None:
        require_sha256(review["package_sha256"], "review.package_sha256")
    if review.get("smoke_verdict", "coherent") not in SMOKE_VERDICTS:
        raise OcCheckpointError(f"smoke_verdict must be one of {SMOKE_VERDICTS}")
    if review.get("degeneracy_reading", "unstated") not in DEGENERACY_READINGS:
        raise OcCheckpointError(f"degeneracy_reading must be one of {DEGENERACY_READINGS}")
    review.setdefault("reviewed_at", now_utc())
    review.setdefault("smoke_verdict", "coherent")
    review.setdefault("degeneracy_reading", "unstated")
    review.setdefault("media_out", [])
    review.setdefault("bank2", None)
    return review


# --------------------------------------------------------------------------
# records and rows
# --------------------------------------------------------------------------


def build_record(
    *,
    checkpoint_id: str,
    seq: int,
    event: str,
    supersedes: str | None,
    checkpoint: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    status: str,
    status_reason: str,
    notes: str,
    created_at: str,
) -> dict[str, Any]:
    if event not in EVENTS:
        raise OcCheckpointError(f"event {event!r} is not one of {EVENTS}")
    if status not in STATUSES:
        raise OcCheckpointError(f"status {status!r} is not one of {STATUSES}")
    body = {
        "schema": RECORD_SCHEMA,
        "campaign": CAMPAIGN,
        "checkpoint_id": checkpoint_id,
        "record_instance_id": f"{checkpoint_id}-{seq:04d}-{event}",
        "record_seq": seq,
        "event": event,
        "supersedes_record_sha": supersedes,
        "checkpoint": json.loads(json.dumps(dict(checkpoint), sort_keys=True)),
        "review": json.loads(json.dumps(dict(review))) if review else None,
        "status": status,
        "status_reason": status_reason,
        "notes": notes,
        "created_at": created_at,
    }
    return seal(body)


def checkpoint_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One ledger row per sealed record -- every column, always present."""

    checkpoint = record["checkpoint"]
    review = record.get("review") or {}
    bank1 = review.get("bank1") or {}
    columns = review.get("columns") or {}
    bank2 = review.get("bank2") or {}
    media = review.get("media_out") or []
    row = {
        "schema": CHECKPOINT_ROW_SCHEMA,
        "campaign": CAMPAIGN,
        "checkpoint_id": record["checkpoint_id"],
        "record_instance_id": record["record_instance_id"],
        "record_seq": record["record_seq"],
        "event": record["event"],
        "stage": checkpoint["stage"],
        "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
        "clone_source_id": checkpoint.get("clone_source_id"),
        "parent_code": checkpoint.get("parent_code"),
        "tag": checkpoint.get("tag"),
        "wave": checkpoint.get("wave"),
        "directory": checkpoint.get("directory"),
        "directory_sha256": checkpoint.get("directory_sha256"),
        "parameter_count": checkpoint.get("parameter_count"),
        "parameter_count_source": checkpoint.get("parameter_count_source"),
        "parent_parameter_count": checkpoint.get("parent_parameter_count"),
        "parent_parameter_count_source": checkpoint.get(
            "parent_parameter_count_source"
        ),
        "net_ratio": checkpoint.get("net_ratio"),
        "quantization": checkpoint.get("quantization"),
        "recipe_sha256": (checkpoint.get("recipe") or {}).get("recipe_sha256"),
        "status": record["status"],
        "status_reason": record["status_reason"],
        "sealed_useful_or_partial": bank1.get("useful_or_partial"),
        "parent_useful_or_partial": bank1.get("parent_useful_or_partial"),
        "sealed_verdict": bank1.get("sealed_verdict"),
        "canonical_signal": bank1.get("canonical_signal"),
        "column_a": columns.get("column_a"),
        "column_b": columns.get("column_b"),
        "bank2_sealed_useful_or_partial": bank2.get("sealed_useful_or_partial"),
        "bank2_canonical_signal": bank2.get("canonical_signal"),
        "media_out_artifacts": len(media),
        "media_out_scored": sum(1 for m in media if m.get("cyril_score") is not None),
        "package_path": review.get("package_path"),
        "package_sha256": review.get("package_sha256"),
        "dossier_sha256": review.get("dossier_sha256"),
        "verdict_receipt_sha": review.get("verdict_receipt_sha"),
        "notes": record["notes"],
        "sealed_at": record["sealed_at"],
        "receipt_sha": record["receipt_sha"],
    }
    missing = [c for c in CHECKPOINT_ROW_COLUMNS if c not in row]
    if missing:
        raise OcCheckpointError(f"row is missing columns {missing}")
    return row


def commit_record(
    *,
    registry: Path,
    ledger: Path,
    record: Mapping[str, Any],
    already_locked: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Write the sealed record, then append its ledger row.  In that order.

    The record is the artifact; the row is the index into it.  A row with no
    record is a claim with no evidence, so the evidence is written first and
    the row's `receipt_sha` is what binds them.

    The whole thing happens under this checkpoint's chain lock, and the head of
    the chain is re-read inside the lock and compared against what the caller
    computed its sequence from.  Two mutating commands racing used to both read
    seq=1, write `-0002-review.json` and `-0002-status.json`, and poison the
    chain permanently with "record sequence [1,2,2] has a gap or a duplicate".
    Now the loser is refused and told to re-run.

    `already_locked` is for `cmd_delete`, which must hold the SAME lock across
    its conditions, its `rmtree` and this commit -- taking it only here would
    take it after the weights were already gone.  The lock is not reentrant, so
    the caller says it holds it.
    """

    checkpoint_id = record["checkpoint_id"]
    path = record_dir(registry, checkpoint_id) / f"{record['record_instance_id']}.json"
    guard: Any = (
        contextlib.nullcontext()
        if already_locked
        else WriterLock(chain_lock(registry, checkpoint_id), timeout_s=60.0)
    )
    with guard:
        chain = history(registry, checkpoint_id)
        expected_seq = len(chain) + 1
        if int(record["record_seq"]) != expected_seq:
            raise OcCheckpointError(
                f"{checkpoint_id}: this record is seq {record['record_seq']} but the "
                f"chain is now {len(chain)} long, so the next seq is {expected_seq}.  "
                "Another command extended the chain while this one was working; "
                "re-run it against the new head."
            )
        head_sha = chain[-1]["receipt_sha"] if chain else None
        if record.get("supersedes_record_sha") != head_sha:
            raise OcCheckpointError(
                f"{checkpoint_id}: this record supersedes "
                f"{str(record.get('supersedes_record_sha'))[:16]} but the head of "
                f"the chain is {str(head_sha)[:16]}; re-run against the new head"
            )
        if path.exists():
            raise OcCheckpointError(
                f"{path} already exists; records are never rewritten"
            )
        write_sealed(path, record)
        row = checkpoint_row(record)
        append_rows(ledger, [row])
    return path, row


# --------------------------------------------------------------------------
# plan printing
# --------------------------------------------------------------------------


def print_plan(title: str, lines: Sequence[str], *, confirmed: bool) -> None:
    print(f"=== {title} ===")
    for line in lines:
        print(f"  {line}")
    if not confirmed:
        print("\nDRY RUN: nothing was written.  Re-run with --confirm.")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _resolve_parameter_count(
    value: str, directory: Path | None, label: str
) -> tuple[int, str]:
    """``(count, source)`` -- and when there are shards on disk, THE BYTES DECIDE.

    `net_ratio` is the campaign's central number: it is sealed into the record,
    into the ledger row, into SCORECARD.md and into the review package Cyril
    reads.  It used to be possible to TYPE it.  `--parameter-count 256
    --parent-parameter-count 4096` registered a 16x fold that had folded
    nothing, and `verify --deep` then re-divided the same two typed integers and
    called it verified -- a tautology dressed as a check, while the free header
    read that would have caught it (0.00 s on a real 4B) was skipped in favour
    of re-hashing every byte at 169 MB/s.

    So: if the directory contains `*.safetensors`, the count is ALWAYS read from
    the headers, and an explicit value that disagrees is a refusal -- exactly
    the treatment `_components` already gives a TENSOR_PREFIX.  A typed count
    survives only where there is nothing to read it from (a checkpoint that is
    not safetensors at all), and it is then labelled `asserted-by-operator` so a
    reader of the scorecard can tell a measured 16x from a typed one.
    """

    derived: int | None = None
    if directory is not None:
        try:
            derived, _per_file = safetensors_parameter_count(directory)
        except OcSealError:
            # no *.safetensors under the directory: nothing to read.  Any other
            # header failure (a truncated shard, a bad span) still raises.
            if checkpoint_shards(directory):
                raise
            derived = None
    if value == "auto":
        if derived is None:
            raise OcCheckpointError(
                f"{label}=auto needs a directory containing *.safetensors to read "
                "headers from"
            )
        return derived, "derived-from-safetensors-headers"
    try:
        typed = int(value)
    except ValueError as exc:
        raise OcCheckpointError(f"{label} {value!r} is not an integer") from exc
    if derived is not None and derived != typed:
        raise OcCheckpointError(
            f"{label} says {typed:,} parameters, the safetensors headers under "
            f"{directory} say {derived:,}.  The bytes decide -- pass `auto`, or "
            "point at the directory you actually mean."
        )
    if derived is not None:
        return derived, "derived-from-safetensors-headers"
    return typed, "asserted-by-operator"


def _recipe_block(path: Path | None, command: str | None) -> dict[str, Any]:
    if path is None and not command:
        raise OcCheckpointError(
            "a checkpoint with no recipe cannot be rebuilt and therefore cannot "
            "be dropped; pass --recipe and/or --recipe-command"
        )
    file_digest = file_sha256(path) if path else None
    return {
        "kind": ("file+command" if path and command else "file" if path else "command"),
        "recipe_path": str(path) if path else None,
        "recipe_file_sha256": file_digest,
        "command": command,
        "recipe_sha256": canonical_sha256(
            {"file_sha256": file_digest, "command": command}
        ),
    }


def _components(
    specs: Sequence[str],
    stage: str,
    *,
    parent_counts: Mapping[str, int] | None = None,
    child_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """``NAME:PARENT_PARAMS:CHILD_PARAMS[:TENSOR_PREFIX]``, with the 4x check.

    When a TENSOR_PREFIX is given, both counts are RE-DERIVED from the parent's
    and the child's safetensors headers (sum of elements over every tensor whose
    name starts with the prefix) and a disagreement with the typed numbers is a
    refusal.  When it is not, the numbers stay an operator assertion and the
    record says so in `counts_source` -- because the net ratio is computed from
    the bytes while these were typed, and a reader of the scorecard is entitled
    to know which of the two he is looking at.
    """

    out: list[dict[str, Any]] = []
    for spec in specs:
        fields = spec.split(":")
        if len(fields) not in (3, 4):
            raise OcCheckpointError(
                f"--component {spec!r} is not "
                "NAME:PARENT_PARAMS:CHILD_PARAMS[:TENSOR_PREFIX]"
            )
        name, parent_params, child_params = fields[0], fields[1], fields[2]
        prefix = fields[3] if len(fields) == 4 else None
        try:
            parent_n, child_n = int(parent_params), int(child_params)
        except ValueError as exc:
            raise OcCheckpointError(
                f"--component {spec!r}: the two parameter counts must be integers"
            ) from exc
        if parent_n <= 0:
            raise OcCheckpointError(f"--component {name}: parent params must be > 0")
        if child_n < 0:
            raise OcCheckpointError(f"--component {name}: child params must be >= 0")
        source = "asserted-by-operator"
        tensors: dict[str, int] | None = None
        if prefix:
            if parent_counts is None or child_counts is None:
                raise OcCheckpointError(
                    f"--component {name}: a TENSOR_PREFIX needs both directories to "
                    "read headers from (pass --parent-directory)"
                )
            derived_parent, parent_tensors = prefix_parameter_count(parent_counts, prefix)
            derived_child, child_tensors = prefix_parameter_count(child_counts, prefix)
            if parent_tensors == 0:
                raise OcCheckpointError(
                    f"--component {name}: no parent tensor name starts with "
                    f"{prefix!r}; the component cannot be derived"
                )
            if (derived_parent, derived_child) != (parent_n, child_n):
                raise OcCheckpointError(
                    f"--component {name}: the headers say {derived_parent} parent / "
                    f"{derived_child} child parameters under {prefix!r}, the command "
                    f"says {parent_n} / {child_n}.  The bytes decide."
                )
            source = "derived-from-safetensors-headers"
            tensors = {"parent": parent_tensors, "child": child_tensors}
        ratio = child_n / parent_n
        if stage == "fold" and ratio > MAX_COMPONENT_RATIO:
            raise OcCheckpointError(
                f"--component {name}: ratio {ratio:.6f} > {MAX_COMPONENT_RATIO} -- "
                "a folded component stores at most a quarter of its parent "
                "(>=4x).  A component carried through unfolded is "
                "--passed-through, not --component."
            )
        out.append(
            {
                "component": name,
                "parent_params": parent_n,
                "child_params": child_n,
                "ratio": ratio,
                "tensor_prefix": prefix,
                "counts_source": source,
                "tensors_matched": tensors,
            }
        )
    return out


def _fold_accounting(
    components: Sequence[Mapping[str, Any]],
    *,
    parameter_count: int,
    parent_parameter_count: int,
    passed_through: Sequence[str],
    slack: int,
) -> dict[str, Any]:
    """Does the fold's own story about itself add up against the counts?

    Three refusals, all arithmetic:

    1. a fold with NO named component is not a fold.  `--component` used to
       default to `[]`, so a "fold" that stored exactly as much as its parent
       registered clean at net_ratio 1.0, sealed as `candidate` and appeared on
       the scorecard under *Stage: fold*;
    2. a fold whose child is not smaller than its parent has folded nothing;
    3. a fold may not CLAIM a bigger saving than the model actually lost.  The
       component numbers can be typed; the two totals are read from the
       headers, so `sum(parent - child)` over the components is bounded by
       `parent_total - child_total`.  A build that legitimately adds parameters
       elsewhere (a closed-form buffer) states the difference with
       `--accounting-slack N`, which lands in the record as an argued exception
       rather than a silent tolerance.
    """

    if not components:
        raise OcCheckpointError(
            "stage fold needs at least one --component "
            "NAME:PARENT_PARAMS:CHILD_PARAMS[:TENSOR_PREFIX]: the >=4x rule is "
            "PER COMPONENT, so a fold that names none of them has not been "
            "checked against it at all"
        )
    measured_drop = parent_parameter_count - parameter_count
    claimed_saving = sum(
        int(c["parent_params"]) - int(c["child_params"]) for c in components
    )
    claimed_parent = sum(int(c["parent_params"]) for c in components)
    if measured_drop <= 0:
        raise OcCheckpointError(
            f"stage fold: the child stores {parameter_count:,} parameters against "
            f"the parent's {parent_parameter_count:,} -- a fold that is not smaller "
            "than its parent has folded nothing (net_ratio "
            f"{parameter_count / parent_parameter_count:.6f})"
        )
    if claimed_parent > parent_parameter_count:
        raise OcCheckpointError(
            f"the components claim {claimed_parent:,} parent parameters but the "
            f"parent has {parent_parameter_count:,}"
        )
    if claimed_saving > measured_drop + slack:
        raise OcCheckpointError(
            f"the components claim a saving of {claimed_saving:,} parameters but "
            f"the model only shrank by {measured_drop:,} "
            f"(slack {slack:,}).  Either a component count is wrong, or the build "
            "added parameters elsewhere -- name them in --passed-through and state "
            "the difference with --accounting-slack N so it is on the record."
        )
    return {
        "measured_drop": measured_drop,
        "claimed_saving": claimed_saving,
        "claimed_parent_parameters": claimed_parent,
        "unexplained": measured_drop - claimed_saving,
        "accounting_slack": slack,
        "passed_through": list(passed_through),
        "counts_source": sorted({str(c["counts_source"]) for c in components}),
    }


def _assert_directory_is_free(
    registry: Path, directory: Path, *, claimant: str
) -> None:
    """No two live checkpoints may share (or nest inside) one directory.

    R4(iv-b) only follows parent and clone links, so two ids pointing at the
    same bytes would let a DROP of one `rm -rf` the other's weights, and the
    registry would not notice until the next `verify --deep`.  Containment is
    checked as well as equality: deleting a parent directory takes every child
    directory under it.
    """

    # RESOLVED on both sides, or the check compares spellings instead of
    # locations: `children/X` and `children/../children/X` name one directory,
    # and so do a path and an alias symlink to it.  Records sealed before the
    # resolving guard existed still carry an unresolved spelling, so the stored
    # side is resolved here too rather than trusted.
    target = resolve_path(directory)
    for other_id, record in all_current(registry).items():
        if other_id == claimant:
            continue
        checkpoint = record["checkpoint"]
        if checkpoint.get("deleted"):
            continue
        other = resolve_path(str(checkpoint.get("directory")))
        if other == target:
            raise OcCheckpointError(
                f"{other_id} is already registered at {other}.  Two ids on one "
                "directory means a DROP of either deletes the other's bytes."
            )
        if other.parts[: len(target.parts)] == target.parts:
            raise OcCheckpointError(
                f"{other_id} lives at {other}, inside {target}; deleting the outer "
                "directory would take the inner checkpoint with it"
            )
        if target.parts[: len(other.parts)] == other.parts:
            raise OcCheckpointError(
                f"{target} is inside {other_id}'s directory {other}; deleting that "
                "one would take these bytes with it"
            )


def _assert_parent_is_a_real_clone(
    registry: Path, parent_id: str, parent_record: Mapping[str, Any]
) -> None:
    """R2, checked against the clone RECEIPT rather than against a word.

    Reading `stage == "clone"` off the parent record was the whole enforcement,
    and `clone` was an accepted value of `register --stage` -- so a clone could
    be fabricated by hand with no byte copy, no digest assertion and no
    receipt, and the no-regress rule was satisfied by a string.  A clone record
    is now only ever produced by `cmd_clone`, and this additionally re-opens the
    sealed receipt that command wrote and checks it describes THESE bytes.
    """

    checkpoint = parent_record["checkpoint"]
    if checkpoint.get("stage") != "clone" or parent_record.get("event") not in (
        "clone", "review", "status"
    ):
        raise OcCheckpointError(
            f"{parent_id} is stage {checkpoint.get('stage')!r}: a training stage "
            "must be trained from a pre-training CLONE.  Run `clone` first (R2)."
        )
    first = history(registry, parent_id)[0]
    if first["event"] != "clone":
        raise OcCheckpointError(
            f"{parent_id}'s chain opens with a {first['event']!r} record, not a "
            "`clone` record; only `oc_checkpoint.py clone` may create a clone"
        )
    receipt_path = checkpoint.get("clone_receipt_path")
    receipt_sha = checkpoint.get("clone_receipt_sha")
    if not receipt_path or not receipt_sha:
        raise OcCheckpointError(
            f"{parent_id} carries no clone receipt; R2(2) is a byte copy whose "
            "digest was asserted equal to the source's, and the receipt is the "
            "only evidence that happened"
        )
    try:
        receipt = read_sealed(Path(receipt_path))
    except (OSError, OcSealError, ValueError) as exc:
        raise OcCheckpointError(
            f"{parent_id}: its clone receipt {receipt_path} does not read as a "
            f"sealed record ({exc})"
        ) from exc
    if receipt["receipt_sha"] != receipt_sha:
        raise OcCheckpointError(
            f"{parent_id}: the clone receipt on disk seals to "
            f"{receipt['receipt_sha'][:16]}, the record names {receipt_sha[:16]}"
        )
    if receipt.get("clone") != parent_id:
        raise OcCheckpointError(
            f"{parent_id}: its clone receipt is for {receipt.get('clone')!r}"
        )
    if receipt.get("clone_directory_sha256") != checkpoint.get("directory_sha256"):
        raise OcCheckpointError(
            f"{parent_id}: the clone receipt digests "
            f"{str(receipt.get('clone_directory_sha256'))[:16]} but the record says "
            f"{str(checkpoint.get('directory_sha256'))[:16]}"
        )
    if receipt.get("source_directory_sha256") != receipt.get("clone_directory_sha256"):
        raise OcCheckpointError(
            f"{parent_id}: the clone receipt's source and clone digests differ; "
            "the copy was never asserted equal to what it copied"
        )


# --------------------------------------------------------------------------
# delta checkpoints (1.1.0)
# --------------------------------------------------------------------------
#
# A stage-D delta is NOT a model.  It is a small directory -- adapters, trained
# parameter subsets, gains -- plus `DELTA.json`, the recipe that says which base
# tensor each of them lands on, and the base it lands on, pinned by digest.  The
# model is `compose(base, delta)`, and the design keeps it that way until the D4
# merge because nine full 55 GB copies fit on no disk here (DISTILL_DESIGN 4.1).
#
# Every rule below exists because a delta that composes onto the WRONG bytes
# fails silently: the shapes still match, the forward still runs, and the
# numbers are simply about some other model.  So the base is named twice (the
# recipe and the command line), both are checked against the base's SEALED
# digest, every op is checked against the base's headers before anything is
# registered, and `delta-compose --deep` re-hashes the base itself.


def _tensor_index(directory: Path) -> dict[str, dict[str, Any]]:
    """``{tensor: {dtype, shape, file}}`` over every shard, from the headers.

    Through `checkpoint_shards`, i.e. the same walk the digest and the counts
    use, so a symlinked shard is a refusal here as everywhere else.
    """

    base = Path(directory)
    out: dict[str, dict[str, Any]] = {}
    for shard in checkpoint_shards(base):
        for name, spec in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            if name in out:
                raise OcCheckpointError(
                    f"{base}: tensor {name!r} appears in {out[name]['file']} and in "
                    f"{shard.relative_to(base).as_posix()}; a name must resolve to "
                    "exactly one set of bytes"
                )
            out[name] = {
                "dtype": str(spec.get("dtype")),
                "shape": [int(d) for d in spec.get("shape") or []],
                "file": shard.relative_to(base).as_posix(),
            }
    return out


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _region(
    target_shape: Sequence[int], slices: Any, where: str
) -> tuple[list[list[int]], list[int]]:
    """Normalise ``[[dim, start, stop], ...]`` against a shape; return the region.

    A slice that is out of bounds, empty, repeats a dimension or is not three
    integers is a refusal: a region that numpy would silently clip is a delta
    that writes fewer elements than its recipe says.
    """

    if slices is None:
        return [], [int(d) for d in target_shape]
    if not isinstance(slices, list):
        raise OcCheckpointError(f"{where}: slice must be a list of [dim, start, stop]")
    region = [int(d) for d in target_shape]
    seen: set[int] = set()
    normal: list[list[int]] = []
    for item in slices:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        ):
            raise OcCheckpointError(
                f"{where}: slice entry {item!r} is not three integers [dim, start, stop]"
            )
        dim, start, stop = item
        if not 0 <= dim < len(target_shape):
            raise OcCheckpointError(
                f"{where}: slice dim {dim} is outside a {len(target_shape)}-d tensor"
            )
        if dim in seen:
            raise OcCheckpointError(f"{where}: slice names dim {dim} twice")
        if not 0 <= start < stop <= int(target_shape[dim]):
            raise OcCheckpointError(
                f"{where}: slice [{start}, {stop}) on dim {dim} is empty or outside "
                f"[0, {target_shape[dim]})"
            )
        seen.add(dim)
        region[dim] = stop - start
        normal.append([dim, start, stop])
    return sorted(normal), region


def _regions_overlap(shape: Sequence[int], left: list[list[int]], right: list[list[int]]) -> bool:
    def bounds(slices: list[list[int]]) -> list[tuple[int, int]]:
        out = [(0, int(d)) for d in shape]
        for dim, start, stop in slices:
            out[dim] = (start, stop)
        return out

    return all(
        a0 < b1 and b0 < a1
        for (a0, a1), (b0, b1) in zip(bounds(left), bounds(right))
    )


def load_delta_recipe(directory: Path) -> tuple[dict[str, Any], str]:
    """`DELTA.json` from a delta directory, and its file sha256."""

    path = Path(directory) / DELTA_RECIPE_NAME
    if not path.is_file():
        raise OcCheckpointError(
            f"{directory} has no {DELTA_RECIPE_NAME}; a delta without its "
            "composition recipe is a pile of tensors nobody can put back on a base"
        )
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OcCheckpointError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(recipe, dict) or recipe.get("schema") != DELTA_RECIPE_SCHEMA:
        raise OcCheckpointError(f"{path} is not a {DELTA_RECIPE_SCHEMA}")
    for field in ("base_checkpoint_id", "base_directory_sha256", "entries"):
        if field not in recipe:
            raise OcCheckpointError(f"{path} is missing {field!r}")
    require_sha256(recipe["base_directory_sha256"], f"{path} base_directory_sha256")
    if not isinstance(recipe["entries"], list) or not recipe["entries"]:
        raise OcCheckpointError(
            f"{path}: entries is empty -- a delta that changes nothing is not a "
            "checkpoint, it is its base under a second name"
        )
    return recipe, file_sha256(path)


def validate_delta_entries(
    recipe: Mapping[str, Any],
    *,
    base_index: Mapping[str, Mapping[str, Any]],
    delta_index: Mapping[str, Mapping[str, Any]],
    where: str,
) -> dict[str, Any]:
    """Every op checked against the base's and the delta's headers.

    Returns the accounting: elements REPLACED (no new parameters), elements
    ADDED (adapters and gains are real parameters until the D4 merge folds
    them), and the per-op counts.  Refuses, with the entry index:

    * an op not in `DELTA_OPS`, a target the base does not have, a delta file or
      tensor that does not exist;
    * a replace whose tensor is not exactly the region's shape and the base's
      dtype (a bf16 base region replaced by fp32 bytes would be cast on compose,
      i.e. the registered delta would not be the trained delta);
    * a LoRA whose A/B do not multiply out to the region, or alpha/rank that is
      not finite and positive;
    * a scale whose gain is not one value per element of the region along `dim`;
    * two replaces whose regions overlap on one target (which one wins is then
      an accident of list order nobody chose);
    * a tensor in the delta's shards used by no op, or by two (orphan bytes are
      bytes the digest covers and the composition ignores).
    """

    used: dict[str, int] = {}
    replaced = added = 0
    per_op = {op: 0 for op in DELTA_OPS}
    replaces: dict[str, list[list[list[int]]]] = {}
    targets: set[str] = set()

    def tensor(entry_where: str, name: Any) -> Mapping[str, Any]:
        if not isinstance(name, str) or name not in delta_index:
            raise OcCheckpointError(
                f"{entry_where}: delta tensor {name!r} is not in the delta's shards"
            )
        if name in used:
            raise OcCheckpointError(
                f"{entry_where}: delta tensor {name!r} is already used by entry "
                f"{used[name]}; one tensor, one role"
            )
        return delta_index[name]

    for number, entry in enumerate(recipe["entries"]):
        entry_where = f"{where} entry {number}"
        if not isinstance(entry, Mapping):
            raise OcCheckpointError(f"{entry_where} is not an object")
        op = entry.get("op")
        if op not in DELTA_OPS:
            raise OcCheckpointError(f"{entry_where}: op {op!r} is not one of {DELTA_OPS}")
        target = entry.get("target")
        if target not in base_index:
            raise OcCheckpointError(
                f"{entry_where}: target {target!r} is not a tensor of the base"
            )
        base_spec = base_index[target]
        slices, region = _region(base_spec["shape"], entry.get("slice"), entry_where)
        targets.add(str(target))
        if op == "replace":
            spec = tensor(entry_where, entry.get("tensor"))
            if list(spec["shape"]) != region:
                raise OcCheckpointError(
                    f"{entry_where}: replace tensor is {spec['shape']} but the region of "
                    f"{target} {base_spec['shape']} under slice {slices} is {region}"
                )
            if spec["dtype"] != base_spec["dtype"]:
                raise OcCheckpointError(
                    f"{entry_where}: replace tensor is {spec['dtype']}, the base tensor "
                    f"is {base_spec['dtype']}; compose would cast it, so the bytes "
                    "registered would not be the bytes that land"
                )
            for other in replaces.get(str(target), []):
                if _regions_overlap(base_spec["shape"], other, slices):
                    raise OcCheckpointError(
                        f"{entry_where}: replace region {slices} on {target} overlaps an "
                        f"earlier replace region {other}"
                    )
            replaces.setdefault(str(target), []).append(slices)
            used[str(entry["tensor"])] = number
            replaced += _elements(region)
        elif op == "lora":
            if len(region) != 2:
                raise OcCheckpointError(
                    f"{entry_where}: lora needs a 2-d region, {target} gives {region}"
                )
            a_spec = tensor(entry_where, entry.get("a"))
            used[str(entry["a"])] = number
            b_spec = tensor(entry_where, entry.get("b"))
            used[str(entry["b"])] = number
            if len(a_spec["shape"]) != 2 or len(b_spec["shape"]) != 2:
                raise OcCheckpointError(f"{entry_where}: lora A and B must be 2-d")
            rank = int(a_spec["shape"][0])
            if (
                int(b_spec["shape"][1]) != rank
                or int(b_spec["shape"][0]) != region[0]
                or int(a_spec["shape"][1]) != region[1]
            ):
                raise OcCheckpointError(
                    f"{entry_where}: B {b_spec['shape']} @ A {a_spec['shape']} does not "
                    f"give the region {region} of {target}"
                )
            alpha = entry.get("alpha")
            if (
                isinstance(alpha, bool)
                or not isinstance(alpha, (int, float))
                or not math.isfinite(float(alpha))
                or float(alpha) <= 0
            ):
                raise OcCheckpointError(
                    f"{entry_where}: lora alpha {alpha!r} is not a finite positive number"
                )
            if "rank" in entry and int(entry["rank"]) != rank:
                raise OcCheckpointError(
                    f"{entry_where}: rank {entry['rank']} != A's {rank} rows"
                )
            added += _elements(a_spec["shape"]) + _elements(b_spec["shape"])
        else:  # scale
            spec = tensor(entry_where, entry.get("tensor"))
            dim = entry.get("dim")
            if not isinstance(dim, int) or isinstance(dim, bool) or not 0 <= dim < len(region):
                raise OcCheckpointError(
                    f"{entry_where}: scale dim {dim!r} is not a dimension of {region}"
                )
            if list(spec["shape"]) != [region[dim]]:
                raise OcCheckpointError(
                    f"{entry_where}: gain is {spec['shape']}, the region has "
                    f"{region[dim]} along dim {dim}"
                )
            used[str(entry["tensor"])] = number
            added += _elements(spec["shape"])
        per_op[str(op)] += 1
    orphans = sorted(set(delta_index) - set(used))
    if orphans:
        raise OcCheckpointError(
            f"{where}: delta tensor(s) used by no entry: {orphans[:6]}"
            f"{' ...' if len(orphans) > 6 else ''} -- the digest covers bytes the "
            "composition would ignore"
        )
    return {
        "replaced_elements": replaced,
        "added_parameters": added,
        "delta_parameters": sum(_elements(s["shape"]) for s in delta_index.values()),
        "ops": per_op,
        "targets": len(targets),
    }


def _delta_base_of(record: Mapping[str, Any]) -> str | None:
    return ((record.get("checkpoint") or {}).get("delta") or {}).get("base_checkpoint_id")


def resolve_delta_chain(
    registry: Path, checkpoint_id: str, *, deep: bool = False
) -> list[dict[str, Any]]:
    """The composition chain, ROOT FIRST: one full checkpoint, then each delta.

    Walks `delta.base_checkpoint_id` down from `checkpoint_id` and refuses on the
    first link whose pin no longer holds: the base is gone from the registry, it
    was deleted, its sealed `directory_sha256` is not the digest the delta was
    registered against, or -- with `deep` -- the bytes on disk no longer digest
    to it.  A clone of a delta carries the delta block of its source (clone
    copies the checkpoint dict), so it resolves through the same base.
    """

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = checkpoint_id
    pin: str | None = None
    pinned_by: str | None = None
    while cursor is not None:
        if cursor in seen or len(chain) >= MAX_DELTA_CHAIN:
            raise OcCheckpointError(
                f"delta chain from {checkpoint_id} revisits {cursor} or exceeds "
                f"{MAX_DELTA_CHAIN} links"
            )
        seen.add(cursor)
        record = current(registry, cursor)
        checkpoint = record["checkpoint"]
        if checkpoint.get("deleted"):
            raise OcCheckpointError(
                f"{cursor} in the delta chain of {checkpoint_id} was deleted; the "
                "delta has nothing to compose onto"
            )
        if pin is not None and checkpoint["directory_sha256"] != pin:
            raise OcCheckpointError(
                f"BASE DIGEST MISMATCH: {pinned_by} is pinned to base {cursor} at "
                f"{pin[:16]}, but {cursor}'s sealed directory_sha256 is "
                f"{checkpoint['directory_sha256'][:16]}.  Composing it would put "
                "the delta onto bytes it was not trained on."
            )
        directory = Path(checkpoint["directory"])
        link: dict[str, Any] = {
            "checkpoint_id": cursor,
            "record_sha": record["receipt_sha"],
            "directory": str(directory),
            "directory_sha256": checkpoint["directory_sha256"],
            "delta": checkpoint.get("delta"),
        }
        if deep:
            if not directory.is_dir():
                raise OcCheckpointError(f"{cursor}: {directory} is missing")
            live, _entries = directory_digest(
                directory, allow_hardlinks=(checkpoint.get("clone_mode") == "hardlink")
            )
            if live != checkpoint["directory_sha256"]:
                raise OcCheckpointError(
                    f"BASE DIGEST MISMATCH: {cursor} at {directory} digests "
                    f"{live[:16]} on disk, sealed {checkpoint['directory_sha256'][:16]}"
                    f"{f' (and {pinned_by} is pinned to it)' if pinned_by else ''}.  "
                    "The bytes moved since the seal."
                )
            link["live_directory_sha256"] = live
        chain.append(link)
        delta = checkpoint.get("delta")
        if delta:
            pin = str(delta["base_directory_sha256"])
            pinned_by = cursor
            cursor = str(delta["base_checkpoint_id"])
        else:
            cursor = None
    chain.reverse()
    return chain


def composed_parameter_count(chain: Sequence[Mapping[str, Any]]) -> tuple[int, list[int]]:
    """Root header count + every link's added parameters, re-read from headers.

    Replacements add nothing; adapters and gains add their elements until the
    D4 merge folds them.  Each link is re-validated against the ROOT's index
    (ops never add or rename tensors, so the composed name set is the root's).
    """

    root = Path(chain[0]["directory"])
    total, _per_file = safetensors_parameter_count(root)
    root_index = _tensor_index(root)
    added: list[int] = []
    for link in chain[1:]:
        directory = Path(link["directory"])
        recipe, _sha = load_delta_recipe(directory)
        summary = validate_delta_entries(
            recipe,
            base_index=root_index,
            delta_index=_tensor_index(directory),
            where=f"{link['checkpoint_id']} {DELTA_RECIPE_NAME}",
        )
        added.append(int(summary["added_parameters"]))
    return total + sum(added), added


def _typed_count(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise OcCheckpointError(f"--parameter-count {value!r} is not an integer") from exc


def _delta_block(
    registry: Path, args: argparse.Namespace, directory: Path
) -> tuple[dict[str, Any], int]:
    """Everything `register --delta-base` checks, and the sealed `delta` block."""

    base_id = args.delta_base
    if args.stage not in ("distill", "finetune"):
        raise OcCheckpointError(
            f"--delta-base is for training outputs (stage distill or finetune), not "
            f"stage {args.stage}"
        )
    if args.parent_checkpoint_id and args.parent_checkpoint_id != base_id:
        raise OcCheckpointError(
            f"--parent-checkpoint-id {args.parent_checkpoint_id} != --delta-base "
            f"{base_id}: a delta composes onto the clone its training started from, "
            "and that clone IS its parent (R2).  Any other base is bytes the "
            "training never saw."
        )
    base_record = current(registry, base_id)
    base_sealed = base_record["checkpoint"]["directory_sha256"]
    recipe, recipe_sha = load_delta_recipe(directory)
    where = f"{directory / DELTA_RECIPE_NAME}"
    if recipe["base_checkpoint_id"] != base_id:
        raise OcCheckpointError(
            f"{where} names base {recipe['base_checkpoint_id']!r}, the command says "
            f"--delta-base {base_id!r}"
        )
    if recipe["base_directory_sha256"] != base_sealed:
        raise OcCheckpointError(
            f"BASE DIGEST MISMATCH: {where} pins base {base_id} at "
            f"{recipe['base_directory_sha256'][:16]}, but {base_id}'s sealed "
            f"directory_sha256 is {base_sealed[:16]}.  The delta was trained on "
            "other bytes; it is not registered."
        )
    if args.delta_base_sha256 and args.delta_base_sha256 != base_sealed:
        raise OcCheckpointError(
            f"BASE DIGEST MISMATCH: --delta-base-sha256 {args.delta_base_sha256[:16]} "
            f"!= {base_id}'s sealed directory_sha256 {base_sealed[:16]}"
        )
    chain = resolve_delta_chain(registry, base_id)
    root_index = _tensor_index(Path(chain[0]["directory"]))
    summary = validate_delta_entries(
        recipe,
        base_index=root_index,
        delta_index=_tensor_index(directory),
        where=where,
    )
    base_count, _added = composed_parameter_count(chain)
    composed = base_count + int(summary["added_parameters"])
    block = {
        "schema": DELTA_RECIPE_SCHEMA,
        "tool_version": TOOL_VERSION,
        "base_checkpoint_id": base_id,
        "base_directory_sha256": base_sealed,
        "base_record_sha": base_record["receipt_sha"],
        "root_checkpoint_id": chain[0]["checkpoint_id"],
        "root_directory_sha256": chain[0]["directory_sha256"],
        "chain": [link["checkpoint_id"] for link in chain],
        "recipe_name": DELTA_RECIPE_NAME,
        "recipe_file_sha256": recipe_sha,
        "base_composed_parameter_count": base_count,
        "composed_parameter_count": composed,
        **summary,
    }
    return block, composed


def delta_plan(
    registry: Path, checkpoint_id: str, *, deep: bool = False
) -> dict[str, Any]:
    """The sealed composition plan the engines and the P1 reader consume.

    The root's file list (names and sizes, plus sha256 under `deep`) lets a
    reader on another process check cheaply that it opened the bytes the plan
    is about; every delta file is hashed (they are small).
    """

    chain = resolve_delta_chain(registry, checkpoint_id, deep=deep)
    composed, added = composed_parameter_count(chain)
    root_dir = Path(chain[0]["directory"])
    root_files = [
        {"path": p.relative_to(root_dir).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(checkpoint_shards(root_dir))
    ]
    if deep:
        _digest, entries = directory_digest(root_dir)
        by_path = {e["path"]: e["sha256"] for e in entries}
        for item in root_files:
            item["sha256"] = by_path[item["path"]]
    links = []
    for link, extra in zip(chain[1:], added):
        directory = Path(link["directory"])
        recipe, recipe_sha = load_delta_recipe(directory)
        files = [
            {
                "path": p.relative_to(directory).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
            for p in sorted(checkpoint_shards(directory))
        ]
        links.append(
            {
                "checkpoint_id": link["checkpoint_id"],
                "record_sha": link["record_sha"],
                "directory": str(directory),
                "directory_sha256": link["directory_sha256"],
                "base_checkpoint_id": recipe["base_checkpoint_id"],
                "base_directory_sha256": recipe["base_directory_sha256"],
                "recipe_file_sha256": recipe_sha,
                "entries": recipe["entries"],
                "files": files,
                "added_parameters": extra,
            }
        )
    return seal(
        {
            "schema": DELTA_PLAN_SCHEMA,
            "campaign": CAMPAIGN,
            "tool_version": TOOL_VERSION,
            "checkpoint_id": checkpoint_id,
            "deep": bool(deep),
            "root": {
                "checkpoint_id": chain[0]["checkpoint_id"],
                "record_sha": chain[0]["record_sha"],
                "directory": chain[0]["directory"],
                "directory_sha256": chain[0]["directory_sha256"],
                "files": root_files,
            },
            "deltas": links,
            "composed_parameter_count": composed,
            "apply_rule": "per target tensor: base -> fp32 -> each link in chain "
            "order, each entry in recipe order -> one cast to the base dtype",
            "created_at": now_utc(),
        }
    )


def cmd_register(args: argparse.Namespace) -> int:
    registry, ledger = Path(args.registry), Path(args.ledger)
    if args.stage == "clone":
        raise OcCheckpointError(
            "a clone may not be registered by hand: `register --stage clone` "
            "would seal status=cloned with no byte copy, no digest assertion "
            "against a sealed source and no clone receipt, and R2 would then be "
            "satisfied by a name.  Use `oc_checkpoint.py clone --source ...`."
        )
    if args.stage == "parent" and not args.parent_provenance_sha256:
        raise OcCheckpointError(
            "stage parent needs --parent-provenance-sha256: a registered parent is "
            "a provenance-verified local copy of a source model (R6), and without "
            "the provenance digest it is only a directory someone pointed at"
        )
    if history(registry, args.checkpoint_id):
        raise OcCheckpointError(
            f"{args.checkpoint_id} is already registered; a change is a new "
            "record on its chain (record-review / set-status), never a re-register"
        )
    naming = parse_checkpoint_id(args.checkpoint_id, args.stage, args.parent_code)
    # RESOLVED, and the resolved path is what gets sealed: the record has to
    # name the location the bytes are at, not the spelling someone typed to get
    # there, or `delete` clause (iv-c) re-reads a `..` path and misses the share.
    directory = resolve_path(args.directory)
    # BEFORE `directory_digest`, which would otherwise byte-hash the CIFS share.
    assert_not_protected(args.directory, what="--directory", stage=args.stage)
    _assert_directory_is_free(registry, directory, claimant=args.checkpoint_id)
    digest, entries = directory_digest(directory)
    total_bytes = sum(int(e["bytes"]) for e in entries)

    delta_block: dict[str, Any] | None = None
    if getattr(args, "delta_base", None):
        # A delta's own headers hold only the adapters and subsets, so its
        # parameter count is the COMPOSED model's: base (read from the root's
        # headers) + every parameter the chain adds.  Still the bytes deciding.
        delta_block, composed = _delta_block(registry, args, directory)
        if args.parameter_count != "auto" and _typed_count(args.parameter_count) != composed:
            raise OcCheckpointError(
                f"--parameter-count says {int(args.parameter_count):,}, the composed "
                f"base + delta headers say {composed:,}.  The bytes decide."
            )
        parameter_count, parameter_count_source = (
            composed, "derived-from-safetensors-headers"
        )
        args.parent_checkpoint_id = args.delta_base
    else:
        if getattr(args, "delta_base_sha256", None):
            raise OcCheckpointError("--delta-base-sha256 without --delta-base pins nothing")
        parameter_count, parameter_count_source = _resolve_parameter_count(
            args.parameter_count, directory, "--parameter-count"
        )
    parent_directory = (
        resolve_path(args.parent_directory) if args.parent_directory else None
    )
    parent_parameter_count, parent_parameter_count_source = _resolve_parameter_count(
        args.parent_parameter_count, parent_directory, "--parent-parameter-count"
    )
    if parent_parameter_count <= 0:
        raise OcCheckpointError("--parent-parameter-count must be > 0")
    net_ratio = parameter_count / parent_parameter_count
    if args.net_ratio is not None and abs(args.net_ratio - net_ratio) > 1e-9:
        raise OcCheckpointError(
            f"--net-ratio {args.net_ratio!r} != {net_ratio!r} computed from the "
            "counts; the ratio is arithmetic and is never asserted over the counts"
        )
    # A fold whose ratio rests on a typed number is a claim, not a measurement.
    # It is still registerable (the checkpoint may not be safetensors at all),
    # but it may not pass silently: the operator says so on the command line and
    # the word `asserted-by-operator` rides the record, the row, the scorecard
    # and the review package from here on.
    asserted = [
        label
        for label, source in (
            ("--parameter-count", parameter_count_source),
            ("--parent-parameter-count", parent_parameter_count_source),
        )
        if source == "asserted-by-operator"
    ]
    if asserted and not args.allow_asserted_parameter_counts:
        raise OcCheckpointError(
            f"{' and '.join(asserted)} could not be read from safetensors headers "
            "(no *.safetensors under the directory), so net_ratio would rest on a "
            "typed number.  Point at the real directory, or pass "
            "--allow-asserted-parameter-counts to record the assertion as an "
            "assertion."
        )
    if args.quantization != "none" and args.stage in ("fold", "distill"):
        raise OcCheckpointError(
            f"quantization={args.quantization!r} is refused at stage {args.stage}: "
            "quantization is forbidden in folds and distills and happens exactly "
            "once, at the very end, re-graded as its own rung"
        )
    if args.parent_checkpoint_id and not history(registry, args.parent_checkpoint_id):
        raise OcCheckpointError(
            f"--parent-checkpoint-id {args.parent_checkpoint_id} is not registered"
        )
    if args.stage in ("distill", "finetune"):
        # R2: no training stage starts from an unsealed, unreviewed checkpoint,
        # and it starts from a CLONE of it, not from it.
        if not args.parent_checkpoint_id:
            raise OcCheckpointError(
                f"stage {args.stage} needs --parent-checkpoint-id: a trained child "
                "must name the clone it was trained from"
            )
        _assert_parent_is_a_real_clone(
            registry, args.parent_checkpoint_id, current(registry, args.parent_checkpoint_id)
        )

    # `--lineage ID:DIGEST` entries are ancestors this build inherited that are
    # not its registered parent (a parent provenance receipt, a donor model).
    lineage: list[dict[str, Any]] = []
    for item in args.lineage or []:
        ancestor, _, digest = str(item).partition(":")
        lineage.append(
            {
                "checkpoint_id": ancestor,
                "directory_sha256": digest or None,
                "record_sha": None,
            }
        )
    if args.parent_checkpoint_id:
        parent_record = current(registry, args.parent_checkpoint_id)
        lineage = [
            *lineage,
            {
                "checkpoint_id": args.parent_checkpoint_id,
                "directory_sha256": parent_record["checkpoint"]["directory_sha256"],
                "record_sha": parent_record["receipt_sha"],
            },
        ]

    wants_prefix = any(len(str(spec).split(":")) == 4 for spec in args.component or [])
    components = _components(
        args.component or [],
        args.stage,
        parent_counts=(
            safetensors_tensor_counts(parent_directory)
            if wants_prefix and parent_directory
            else None
        ),
        child_counts=safetensors_tensor_counts(directory) if wants_prefix else None,
    )
    accounting = (
        _fold_accounting(
            components,
            parameter_count=parameter_count,
            parent_parameter_count=parent_parameter_count,
            passed_through=args.passed_through or [],
            slack=int(args.accounting_slack),
        )
        if args.stage == "fold"
        else None
    )

    checkpoint = {
        "stage": args.stage,
        "parent_checkpoint_id": args.parent_checkpoint_id,
        "clone_source_id": None,
        "parent_code": args.parent_code or naming.get("parent_code"),
        "parent_provenance_sha256": args.parent_provenance_sha256,
        "tag": args.tag or naming.get("tag"),
        "wave": args.wave or naming.get("wave"),
        "directory": str(directory),
        "directory_sha256": digest,
        "directory_file_count": len(entries),
        "directory_bytes": total_bytes,
        "directory_manifest": entries if args.embed_manifest else None,
        "parameter_count": parameter_count,
        "parameter_count_source": parameter_count_source,
        "parent_parameter_count": parent_parameter_count,
        "parent_parameter_count_source": parent_parameter_count_source,
        "parent_directory": str(parent_directory) if parent_directory else None,
        "net_ratio": net_ratio,
        "per_component_ratios": components,
        "fold_accounting": accounting,
        "passed_through": list(args.passed_through or []),
        "quantization": args.quantization,
        "frontier_holder": bool(args.frontier_holder),
        "recipe": _recipe_block(args.recipe, args.recipe_command),
        "lineage": lineage,
        "write_protected": False,
        "clone_mode": None,
        # only on delta records: a 1.0 record's sealed body is unchanged
        **({"delta": delta_block} if delta_block else {}),
    }
    status, reason = status_for(stage=args.stage, review=None, floor=args.partially_capable_floor)
    record = build_record(
        checkpoint_id=args.checkpoint_id,
        seq=1,
        event="register",
        supersedes=None,
        checkpoint=checkpoint,
        review=None,
        status=status,
        status_reason=reason,
        notes=args.notes,
        created_at=now_utc(),
    )
    print_plan(
        f"REGISTER {args.checkpoint_id}",
        [
            f"stage              {args.stage}",
            f"directory          {directory}",
            f"directory_sha256   {digest}",
            f"files / bytes      {len(entries)} / {human_bytes(total_bytes)}",
            f"parameters         {parameter_count:,}  [{parameter_count_source}]",
            f"parent parameters  {parent_parameter_count:,}  ({args.parent_code})"
            f"  [{parent_parameter_count_source}]",
            f"net_ratio          {net_ratio:.10f}  ({1/net_ratio:.4f}x compression)"
            if net_ratio > 0 else "net_ratio          0",
            f"components         {[c['component'] + ' ' + format(c['ratio'], '.4f') + ' ' + ('derived' if c['tensor_prefix'] else 'asserted') for c in checkpoint['per_component_ratios']]}",
            f"accounting         {accounting if accounting else '(not a fold)'}",
            *(
                [
                    f"delta base         {delta_block['base_checkpoint_id']} pinned at "
                    f"{delta_block['base_directory_sha256'][:16]} (matches its seal)",
                    f"delta chain        {' -> '.join(delta_block['chain'])} -> "
                    f"{args.checkpoint_id}",
                    f"delta ops          {delta_block['ops']} on "
                    f"{delta_block['targets']} target tensor(s); replaced "
                    f"{delta_block['replaced_elements']:,}, added "
                    f"{delta_block['added_parameters']:,}",
                ]
                if delta_block
                else []
            ),
            f"recipe_sha256      {checkpoint['recipe']['recipe_sha256']}",
            f"status             {status} -- {reason}",
            f"record             {record['record_instance_id']} {record['receipt_sha'][:16]}",
            f"ledger row appended to {ledger}",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm:
        return 0
    path, row = commit_record(registry=registry, ledger=ledger, record=record)
    print(f"\nsealed {path}")
    print(f"ledger sha256 {file_sha256(ledger)}")
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    """R2: clone before any optimiser step, verify by digest, freeze the copy."""

    registry, ledger = Path(args.registry), Path(args.ledger)
    source = current(registry, args.source)
    source_checkpoint = source["checkpoint"]
    if source_checkpoint["stage"] == "clone":
        raise OcCheckpointError(
            f"{args.source} is itself a clone; train from it and clone the RESULT, "
            "or clone the original"
        )
    if source.get("review") is None and not args.allow_unreviewed_source:
        raise OcCheckpointError(
            f"{args.source} has no review record.  R2(a): a training stage starts "
            "from a SEALED source whose verdict is in the ledger.  Record the "
            "review first, or state the exception with --allow-unreviewed-source."
        )
    # A delta's bytes are only a model on top of its base.  A clone of a delta
    # whose base pin no longer holds would be a byte-perfect copy of something
    # that composes onto nothing, so the chain is resolved first.
    delta_chain = (
        resolve_delta_chain(registry, args.source)
        if source_checkpoint.get("delta")
        else None
    )
    source_dir = Path(source_checkpoint["directory"])
    live_digest, live_entries = directory_digest(source_dir)
    if live_digest != source_checkpoint["directory_sha256"]:
        raise OcCheckpointError(
            f"{args.source}: the directory on disk digests {live_digest[:16]} but "
            f"the sealed record says {source_checkpoint['directory_sha256'][:16]}.  "
            "The bytes moved since the seal; a clone of them would be a clone of "
            "something the ledger does not describe."
        )

    existing = [
        cid for cid, rec in all_current(registry).items()
        if rec["checkpoint"].get("clone_source_id") == args.source
    ]
    index = args.clone_index or (len(existing) + 1)
    clone_id = f"{args.source}.clone-pre-{args.stage_about_to_run}{index}"
    if history(registry, clone_id):
        raise OcCheckpointError(f"{clone_id} already exists")
    # Default under the campaign root's `clones/`, never beside the source: a
    # registered PARENT lives in /home/cereal/models, and defaulting to the
    # source's parent directory would drop a full-size copy of a 27B model into
    # the model library rather than into the campaign tree.
    destination = (
        resolve_path(args.dest)
        if args.dest
        else resolve_path(registry.parent / "clones" / clone_id)
    )
    if destination.exists():
        raise OcCheckpointError(f"{destination} already exists")
    # A clone is a working checkpoint that a trainer will overwrite in place and
    # that R4 may eventually be asked about, so it may not land on the share or
    # in the model library -- `--stage parent` is not an escape here, a clone is
    # never a parent.
    assert_not_protected(args.dest or destination, what="the clone destination")
    _assert_directory_is_free(registry, destination, claimant=clone_id)

    total_bytes = sum(int(e["bytes"]) for e in live_entries)
    print_plan(
        f"CLONE {args.source} -> {clone_id}",
        [
            f"source             {source_dir}",
            f"source digest      {live_digest}  (matches the sealed record)",
            f"destination        {destination}",
            f"mode               {args.mode}"
            + ("  (hardlink: the clone is NOT independent bytes)"
               if args.mode == "hardlink" else "  (independent bytes, no reflink)"),
            f"files / bytes      {len(live_entries)} / {human_bytes(total_bytes)}",
            *(
                [
                    "delta              yes -- the clone copies the delta bytes and "
                    "keeps the base pin; chain "
                    + " -> ".join(link["checkpoint_id"] for link in delta_chain)
                    + " (pins hold)"
                ]
                if delta_chain
                else []
            ),
            f"stage about to run {args.stage_about_to_run}",
            "after the copy      re-digest and assert equality, write the clone "
            "receipt, chmod -w, register as status=cloned",
            "retention           clones are KEEP while any descendant exists (R2c, R6)",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm:
        return 0
    if args.mode == "hardlink" and not args.allow_hardlink:
        raise OcCheckpointError(
            "--mode hardlink needs --allow-hardlink.  A hardlinked clone shares "
            "inodes with the source, so a trainer that writes in place destroys "
            "both copies at once; R2 asks for a byte clone."
        )

    destination.mkdir(parents=True)
    for entry in live_entries:
        src = source_dir / entry["path"]
        dst = destination / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    clone_digest, clone_entries = directory_digest(
        destination, allow_hardlinks=(args.mode == "hardlink")
    )
    if clone_digest != live_digest:
        raise OcCheckpointError(
            f"the clone digests {clone_digest[:16]} against the source's "
            f"{live_digest[:16]}; the copy is not the source and has been left in "
            f"place at {destination} for inspection"
        )
    if args.mode != "hardlink":
        for entry in clone_entries:  # freeze it
            path = destination / entry["path"]
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    receipt = seal(
        {
            "schema": CLONE_RECEIPT_SCHEMA,
            "campaign": CAMPAIGN,
            "source": args.source,
            "source_directory": str(source_dir),
            "source_directory_sha256": live_digest,
            "source_record_sha": source["receipt_sha"],
            "clone": clone_id,
            "clone_directory": str(destination),
            "clone_directory_sha256": clone_digest,
            "clone_mode": args.mode,
            "created_at": now_utc(),
            "stage_about_to_run": args.stage_about_to_run,
            "files": len(clone_entries),
            "bytes": total_bytes,
        }
    )
    receipt_path = destination.parent / f"{clone_id}.clone-receipt.json"
    write_sealed(receipt_path, receipt)

    checkpoint = {
        **{k: v for k, v in source_checkpoint.items()},
        "stage": "clone",
        "parent_checkpoint_id": args.source,
        "clone_source_id": args.source,
        "tag": source_checkpoint.get("tag"),
        "directory": str(destination),
        "directory_sha256": clone_digest,
        "directory_file_count": len(clone_entries),
        "directory_bytes": total_bytes,
        "directory_manifest": None,
        "frontier_holder": False,
        "write_protected": args.mode != "hardlink",
        "clone_mode": args.mode,
        "clone_receipt_path": str(receipt_path),
        "clone_receipt_sha": receipt["receipt_sha"],
        "lineage": [
            *(source_checkpoint.get("lineage") or []),
            {
                "checkpoint_id": args.source,
                "directory_sha256": live_digest,
                "record_sha": source["receipt_sha"],
            },
        ],
    }
    status, reason = status_for(stage="clone", review=None)
    record = build_record(
        checkpoint_id=clone_id,
        seq=1,
        event="clone",
        supersedes=None,
        checkpoint=checkpoint,
        review=None,
        status=status,
        status_reason=reason,
        notes=args.notes or f"pre-{args.stage_about_to_run} clone of {args.source}",
        created_at=now_utc(),
    )
    path, _row = commit_record(registry=registry, ledger=ledger, record=record)
    print(f"\nclone digest matches source: {clone_digest}")
    print(f"clone receipt {receipt_path}")
    print(f"sealed {path}")
    print(f"ledger sha256 {file_sha256(ledger)}")
    return 0


def cmd_record_review(args: argparse.Namespace) -> int:
    registry, ledger = Path(args.registry), Path(args.ledger)
    chain = history(registry, args.checkpoint_id)
    if not chain:
        raise OcCheckpointError(f"{args.checkpoint_id} is not in the registry")
    previous = chain[-1]
    review = validate_review(json.loads(Path(args.review_json).read_text(encoding="utf-8")))
    status, reason = status_for(
        stage=previous["checkpoint"]["stage"],
        review=review,
        floor=args.partially_capable_floor,
    )
    if args.status and args.status != status:
        raise OcCheckpointError(
            f"--status {args.status!r} disagrees with the status the counts imply "
            f"({status!r}).  The status is computed; --status may only assert "
            "agreement."
        )
    record = build_record(
        checkpoint_id=args.checkpoint_id,
        seq=previous["record_seq"] + 1,
        event="review",
        supersedes=previous["receipt_sha"],
        checkpoint=previous["checkpoint"],
        review=review,
        status=status,
        status_reason=reason,
        notes=args.notes or review.get("notes", ""),
        created_at=now_utc(),
    )
    bank1 = review["bank1"]
    print_plan(
        f"RECORD REVIEW {args.checkpoint_id}",
        [
            f"reviewer           {review['reviewer']}",
            f"sealed bank1       {bank1['useful']}U / {bank1['partial']}P / "
            f"{bank1['observed_miss']}M = {bank1['useful_or_partial']} held vs "
            f"parent {bank1['parent_useful_or_partial']}  ({bank1['sealed_verdict']})",
            f"canonical bank1    signal {bank1['canonical_signal']} / artifact "
            f"{bank1['canonical_artifact_miss']} / content "
            f"{bank1['canonical_content_miss']}",
            f"columns            A={review['columns']['column_a']}  "
            f"B={review['columns']['column_b']}",
            f"bank2              {review.get('bank2')}",
            f"media out          {len(review.get('media_out') or [])} artifact(s), "
            f"{sum(1 for m in (review.get('media_out') or []) if m.get('cyril_score') is not None)} scored by Cyril",
            f"canonical signal   {canonical_signal_total(review)} across all surfaces",
            f"status             {previous['status']} -> {status}",
            f"                   {reason}",
            f"record             {record['record_instance_id']} {record['receipt_sha'][:16]}",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm:
        return 0
    path, _row = commit_record(registry=registry, ledger=ledger, record=record)
    print(f"\nsealed {path}")
    print(f"ledger sha256 {file_sha256(ledger)}")
    return 0


def cmd_set_status(args: argparse.Namespace) -> int:
    """Only the two moves the evidence cannot make by itself.

    A status that the review counts imply is never set by hand; what IS set by
    hand is Cyril's own ruling on a packaged child (his read governs) and the
    frontier-holder flag that R4(ii) protects.
    """

    registry, ledger = Path(args.registry), Path(args.ledger)
    previous = current(registry, args.checkpoint_id)
    checkpoint = dict(previous["checkpoint"])
    review = previous.get("review")
    if args.frontier_holder is not None:
        checkpoint["frontier_holder"] = bool(args.frontier_holder)
    status = args.status or previous["status"]
    implied, implied_reason = status_for(
        stage=checkpoint["stage"], review=review, floor=args.partially_capable_floor
    )
    if args.status and args.status != implied and not args.cyril_ruling:
        raise OcCheckpointError(
            f"--status {args.status!r} disagrees with the implied {implied!r}; only "
            "Cyril's own ruling overrides the counts, and it is recorded with "
            "--cyril-ruling \"<his words>\""
        )
    reason = args.cyril_ruling or implied_reason
    record = build_record(
        checkpoint_id=args.checkpoint_id,
        seq=previous["record_seq"] + 1,
        event="status",
        supersedes=previous["receipt_sha"],
        checkpoint=checkpoint,
        review=review,
        status=status,
        status_reason=reason,
        notes=args.notes,
        created_at=now_utc(),
    )
    print_plan(
        f"SET STATUS {args.checkpoint_id}",
        [
            f"status         {previous['status']} -> {status}",
            f"reason         {reason}",
            f"frontier_holder {checkpoint.get('frontier_holder')}",
            f"record         {record['record_instance_id']} {record['receipt_sha'][:16]}",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm:
        return 0
    path, _row = commit_record(registry=registry, ledger=ledger, record=record)
    print(f"\nsealed {path}")
    return 0


def _delete_conditions(
    registry: Path, checkpoint_id: str, args: argparse.Namespace
) -> list[tuple[bool, str]]:
    """R4, one clause at a time, each with the sentence that decided it."""

    record = current(registry, checkpoint_id)
    checkpoint = record["checkpoint"]
    review = record.get("review")
    everyone = all_current(registry)
    out: list[tuple[bool, str]] = []

    out.append(
        (
            record["status"] == DELETABLE_STATUS,
            f"(i-a) status is {record['status']!r}; only {DELETABLE_STATUS!r} may be "
            "dropped ('unmeasured' is a re-run, never a delete)",
        )
    )
    out.append(
        (review is not None, "(i-b) a review record exists (the bytes carry no "
                             "claim the text does not)")
    )
    signal = canonical_signal_total(review)
    out.append(
        (signal == 0, f"(i-c) canonical signal lanes across all surfaces = {signal}; "
                      "must be 0")
    )
    artifact_misses, content_misses = canonical_miss_totals(review)
    out.append(
        (
            content_misses > 0,
            f"(i-e) {content_misses} CONTENT miss(es) measured against "
            f"{artifact_misses} artifact miss(es).  A loop, a cap-ride, an empty "
            "generation or a missing EOS measures nothing either way and may "
            "never authorise a delete; a child with no content miss at all was "
            "not graded, it was not run properly -- re-run it",
        )
    )
    unscored = [
        m.get("prompt_id")
        for m in ((review or {}).get("media_out") or [])
        if m.get("cyril_score") is None and bool(m.get("non_empty"))
    ]
    out.append(
        (not unscored, f"(i-d) no non-empty media artifact awaiting Cyril's score "
                       f"{unscored}")
    )
    out.append(
        (
            not bool(checkpoint.get("frontier_holder")),
            "(ii) not a floor/frontier holder -- a floor whose child has been "
            "deleted is a floor that can only be cited and not checked",
        )
    )
    out.append(
        (
            bool(args.rebuild_command) and bool(args.retained_source),
            "(iii-a) the DROP names an exact one-pass rebuild command and the "
            "retained source it rebuilds from",
        )
    )
    retained_ok = bool(args.retained_source) and (
        args.retained_source in everyone
        or Path(args.retained_source).exists()
    )
    out.append(
        (retained_ok, f"(iii-b) the retained source {args.retained_source!r} is a "
                      "registered checkpoint or a path that exists")
    )
    out.append(
        (
            len(args.reason or "") >= 60,
            f"(iii-c) the reason is {len(args.reason or '')} characters; a DROP is "
            "argued, not asserted (>=60)",
        )
    )
    out.append(
        (
            checkpoint["stage"] not in ("clone", "parent"),
            f"(iv) stage is {checkpoint['stage']!r} -- never a clone, never a parent",
        )
    )
    # R6, enforced where the unlink actually happens.  `register` refuses these
    # paths now, but a record sealed before that guard existed still names one,
    # and this is the last stop before `rmtree`.
    protected = _protected_prefix(Path(checkpoint["directory"]))
    out.append(
        (
            protected is None,
            f"(iv-c) {checkpoint['directory']} is not under a protected prefix "
            f"{PROTECTED_PREFIXES} -- the CIFS share and the source model library "
            f"are never deleted (R6){'' if protected is None else f'; it is under {protected!r}'}",
        )
    )
    dependents = sorted(
        cid for cid, rec in everyone.items()
        if cid != checkpoint_id
        and checkpoint_id in (
            rec["checkpoint"].get("parent_checkpoint_id"),
            rec["checkpoint"].get("clone_source_id"),
            _delta_base_of(rec),
        )
    )
    out.append(
        (not dependents, f"(iv-b) nothing descends from it {dependents}; never a "
                         "clone source, never a parent of a registered child, never "
                         "the base of a delta")
    )
    package = (review or {}).get("package_path")
    package_ok = False
    package_note = "no package_path in the review record"
    if package:
        try:
            # local imports: avoids an import cycle with oc_package
            from oc_package import read_manifest, verify_manifest

            problems = verify_manifest(Path(package))
            if problems:
                package_note = f"package {package}: {problems[:3]}"
            else:
                # verify_manifest is self-referential -- it recomputes the
                # package digest from the manifest's OWN file list, so any
                # well-formed package verifies.  It says the folder is
                # internally consistent; it says nothing about WHOSE folder it
                # is.  A review pointing at another checkpoint's package used to
                # pass this clause.  So the manifest is bound to this checkpoint
                # here: its id, a receipt in THIS chain, and the digest the
                # review itself committed to.
                manifest = read_manifest(Path(package))
                chain_shas = {r["receipt_sha"] for r in history(registry, checkpoint_id)}
                claimed = (review or {}).get("package_sha256")
                if manifest.get("checkpoint_id") != checkpoint_id:
                    package_note = (
                        f"package {package} is for "
                        f"{manifest.get('checkpoint_id')!r}, not {checkpoint_id!r}"
                    )
                elif manifest.get("record_sha") not in chain_shas:
                    package_note = (
                        f"package {package} names record "
                        f"{str(manifest.get('record_sha'))[:16]}, which is not in "
                        "this checkpoint's sealed chain"
                    )
                elif not claimed:
                    package_note = (
                        "the review names a package but no package_sha256 to tie "
                        "it to these bytes"
                    )
                elif manifest.get("package_sha256") != claimed:
                    package_note = (
                        f"package digest {str(manifest.get('package_sha256'))[:16]} "
                        f"!= the review's {str(claimed)[:16]}; the folder changed "
                        "after it was reviewed"
                    )
                else:
                    package_ok = True
                    package_note = (
                        f"package {package} verifies, is this checkpoint's "
                        f"(record {str(manifest.get('record_sha'))[:16]}) and "
                        f"digests to the review's {str(claimed)[:16]}"
                    )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            package_note = f"package {package}: {exc}"
    out.append(
        (package_ok, f"(v) the evidence is packaged, its manifest verifies AND the "
                     f"package is this checkpoint's -- {package_note}")
    )
    live_ok = False
    live_note = "directory missing"
    directory = Path(checkpoint["directory"])
    if directory.is_dir():
        try:
            live, _entries = directory_digest(directory)
            live_ok = live == checkpoint["directory_sha256"]
            live_note = f"on disk {live[:16]} vs sealed {checkpoint['directory_sha256'][:16]}"
        except OcSealError as exc:
            live_note = str(exc)
    out.append(
        (live_ok, f"(vi) the directory on disk is still the one the record "
                  f"describes -- {live_note}")
    )
    return out


def cmd_delete(args: argparse.Namespace) -> int:
    """R4, under this checkpoint's chain lock for the WHOLE command.

    The lock used to be taken inside `commit_record`, i.e. after `rmtree` had
    already run.  If another process extended the chain in that window -- or if
    the ledger's writer lock was still held when `append_rows` timed out --
    `commit_record` raised and left the worst possible state: bytes deleted, no
    sealed delete record, no ledger row, and a head record still claiming the
    directory exists.  Only the `.pending` marker survived to say so.  Now the
    conditions, the unlink and the commit are one critical section, and the
    delete record is BUILT before anything is unlinked.
    """

    registry, ledger = Path(args.registry), Path(args.ledger)
    with WriterLock(chain_lock(registry, args.checkpoint_id), timeout_s=60.0):
        return _delete_locked(args, registry, ledger)


def _delete_locked(
    args: argparse.Namespace, registry: Path, ledger: Path
) -> int:
    record = current(registry, args.checkpoint_id)
    checkpoint = record["checkpoint"]
    directory = Path(checkpoint["directory"])
    conditions = _delete_conditions(registry, args.checkpoint_id, args)

    print(f"=== DELETE {args.checkpoint_id} ===")
    print(f"  directory   {directory}")
    print(f"  bytes       {human_bytes(int(checkpoint.get('directory_bytes') or 0))} "
          f"in {checkpoint.get('directory_file_count')} file(s)")
    print(f"  status      {record['status']}  --  {record['status_reason']}")
    print("  R4 conditions:")
    for ok, text in conditions:
        print(f"    [{'PASS' if ok else 'REFUSE'}] {text}")
    failed = [text for ok, text in conditions if not ok]
    if failed:
        print(f"\nREFUSED: {len(failed)} condition(s) did not hold; nothing was deleted.")
        return 4
    print("\n  reason:")
    for line in (args.reason or "").splitlines() or [""]:
        print(f"    {line}")
    print(f"  rebuild     {args.rebuild_command}")
    print(f"  from        {args.retained_source}")
    if not args.confirm:
        print("\nDRY RUN: nothing was deleted.  Re-run with --confirm.")
        return 0

    # BUILD AND SEAL THE RECORD FIRST, while the bytes are still there.  Every
    # way `build_record` can refuse -- a bad status, an unsealable payload -- is
    # then a refusal with the checkpoint intact, instead of an exception thrown
    # over a directory that no longer exists.
    new_checkpoint = dict(checkpoint)
    new_checkpoint["deleted"] = {
        "deleted_at": now_utc(),
        "directory": str(directory),
        "directory_sha256_at_delete": checkpoint["directory_sha256"],
        "bytes": checkpoint.get("directory_bytes"),
        "reason": args.reason,
        "rebuild_command": args.rebuild_command,
        "retained_source": args.retained_source,
        "conditions": [text for _ok, text in conditions],
    }
    new_record = build_record(
        checkpoint_id=args.checkpoint_id,
        seq=record["record_seq"] + 1,
        event="delete",
        supersedes=record["receipt_sha"],
        checkpoint=new_checkpoint,
        review=record.get("review"),
        status=record["status"],
        status_reason=record["status_reason"],
        notes=args.notes or "checkpoint bytes deleted; the evidence is the package",
        created_at=now_utc(),
    )
    # The row too: `checkpoint_row` asserts the 38-column contract, and a column
    # that drifted is a refusal that must happen before the unlink, not after.
    checkpoint_row(new_record)

    # A marker first, so a crash mid-unlink leaves a trace that says what was
    # in flight; the sealed record is written only once the bytes are gone.
    # It lives in <registry>/.pending/ and NOT in the record directory: a
    # non-record file where the chain is read is a chain that stops being
    # readable, and `history()` raised on it for the whole registry.
    marker = pending_marker(registry, args.checkpoint_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "checkpoint_id": args.checkpoint_id,
                "directory": str(directory),
                "directory_sha256": checkpoint["directory_sha256"],
                "record_instance_id": new_record["record_instance_id"],
                "started_at": now_utc(),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(directory)
    deleted = not directory.exists()
    # `already_locked`: this command has held the chain lock since before the
    # conditions were evaluated.
    path, _row = commit_record(
        registry=registry, ledger=ledger, record=new_record, already_locked=True
    )
    marker.unlink(missing_ok=True)
    print(f"\ndeleted {directory} ({'gone' if deleted else 'STILL PRESENT'})")
    print(f"sealed {path}")
    print(f"ledger sha256 {file_sha256(ledger)}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    records = all_current(Path(args.registry))
    rows = []
    for cid, rec in sorted(records.items()):
        checkpoint, review = rec["checkpoint"], rec.get("review") or {}
        bank1 = review.get("bank1") or {}
        if args.stage and checkpoint["stage"] != args.stage:
            continue
        if args.status and rec["status"] != args.status:
            continue
        rows.append(
            {
                "checkpoint_id": cid,
                "stage": checkpoint["stage"],
                "status": rec["status"],
                "net_ratio": checkpoint.get("net_ratio"),
                "parameters": checkpoint.get("parameter_count"),
                "sealed": bank1.get("useful_or_partial"),
                "canonical": bank1.get("canonical_signal"),
                "column_a": (review.get("columns") or {}).get("column_a"),
                "column_b": (review.get("columns") or {}).get("column_b"),
                "records": rec["record_seq"],
                "directory": checkpoint.get("directory"),
                "deleted": bool(checkpoint.get("deleted")),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    if not rows:
        print("(no checkpoints)")
        return 0
    width = max(len(r["checkpoint_id"]) for r in rows)
    print(f"{'checkpoint'.ljust(width)}  {'stage':<9} {'status':<18} "
          f"{'ratio':>10} {'sealed':>6} {'canon':>6} {'A':>3} {'B':>3}")
    for row in rows:
        ratio = f"{row['net_ratio']:.6f}" if row["net_ratio"] is not None else "-"
        print(
            f"{row['checkpoint_id'].ljust(width)}  {row['stage']:<9} "
            f"{row['status']:<18} {ratio:>10} "
            f"{str(row['sealed'] if row['sealed'] is not None else '-'):>6} "
            f"{str(row['canonical'] if row['canonical'] is not None else '-'):>6} "
            f"{str(row['column_a'] if row['column_a'] is not None else '-'):>3} "
            f"{str(row['column_b'] if row['column_b'] is not None else '-'):>3}"
            + ("  [deleted]" if row["deleted"] else "")
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    chain = history(Path(args.registry), args.checkpoint_id)
    if not chain:
        raise OcCheckpointError(f"{args.checkpoint_id} is not in the registry")
    if args.history:
        print(json.dumps(chain, indent=1))
    else:
        print(json.dumps(chain[-1], indent=1))
    return 0


def _ledger_index(ledger: Path) -> dict[str, str] | None:
    """``{record_instance_id: receipt_sha}`` for this registry's ledger rows."""

    from oc_ledger import CHECKPOINT_ROW_SCHEMA, read_rows  # local: no cycle

    path = Path(ledger).expanduser().absolute()
    if not path.is_file():
        return None
    return {
        str(row["record_instance_id"]): str(row["receipt_sha"])
        for row in read_rows(path)
        if row.get("schema") == CHECKPOINT_ROW_SCHEMA
    }


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-verify every chain, the ledger in BOTH directions, and (with --deep)
    every directory digest.

    `oc_ledger.py verify --registry` walks ROWS and resolves them to records.
    That direction alone is half a check: a record whose ledger append failed
    (writer-lock timeout, Ctrl-C, ENOSPC on the ledger) is invisible to it, and
    both tools then report a clean registry while the ledger is missing a
    checkpoint.  This walks RECORDS and asserts each one has its row, which is
    the direction that catches an orphan.  `repair-row` re-derives the missing
    row from the sealed record.
    """

    registry = Path(args.registry)
    ids = [args.checkpoint_id] if args.checkpoint_id else sorted(all_current(registry))
    problems: list[str] = []
    checked = 0
    rows_checked = 0
    index = _ledger_index(Path(args.ledger)) if args.ledger else None
    if index is None and ids:
        problems.append(
            f"no ledger at {args.ledger}: the registry->ledger direction was NOT "
            "checked, so an orphan record would not be seen here"
        )
    pending = sorted(
        (Path(registry) / ".pending").glob("*.json")
        if (Path(registry) / ".pending").is_dir()
        else []
    )
    for marker in pending:
        problems.append(
            f"an interrupted delete left {marker}; the bytes it names may be "
            "half-gone.  Re-run the delete, or remove the marker once the "
            "directory's state is established."
        )
    for cid in ids:
        try:
            chain = history(registry, cid)  # verifies every seal and the links
        except (OcCheckpointError, OcSealError) as exc:
            problems.append(f"{cid}: {exc}")
            continue
        checked += len(chain)
        if index is not None:
            for record in chain:
                rows_checked += 1
                instance = str(record["record_instance_id"])
                if instance not in index:
                    problems.append(
                        f"{cid}: sealed record {instance} has NO ledger row.  Its "
                        "append did not land.  Re-derive it with `repair-row "
                        f"--checkpoint-id {cid} --record-instance-id {instance} "
                        "--confirm`."
                    )
                elif index[instance] != record["receipt_sha"]:
                    problems.append(
                        f"{cid}: ledger row {instance} carries receipt "
                        f"{index[instance][:12]}, the record seals to "
                        f"{record['receipt_sha'][:12]}"
                    )
        head = chain[-1]
        checkpoint = head["checkpoint"]
        if checkpoint.get("deleted"):
            if Path(checkpoint["directory"]).exists():
                problems.append(
                    f"{cid}: sealed as deleted but {checkpoint['directory']} exists"
                )
            continue
        if args.deep:
            directory = Path(checkpoint["directory"])
            if not directory.is_dir():
                problems.append(f"{cid}: {directory} is missing")
                continue
            try:
                live, _entries = directory_digest(
                    directory,
                    allow_hardlinks=(checkpoint.get("clone_mode") == "hardlink"),
                )
            except OcSealError as exc:
                problems.append(f"{cid}: {exc}")
                continue
            if live != checkpoint["directory_sha256"]:
                problems.append(
                    f"{cid}: directory digests {live[:16]}, sealed "
                    f"{checkpoint['directory_sha256'][:16]}"
                )
        net = checkpoint.get("net_ratio")
        if net is not None and checkpoint.get("parent_parameter_count"):
            expected = checkpoint["parameter_count"] / checkpoint["parent_parameter_count"]
            if not math.isclose(net, expected, rel_tol=0, abs_tol=1e-9):
                problems.append(f"{cid}: net_ratio {net} != {expected}")
        delta = checkpoint.get("delta")
        if delta:
            # A delta's count is the COMPOSED model's, so it is re-derived through
            # the chain (root headers + every link's added parameters), and every
            # base pin and the recipe digest are re-checked on the way.
            try:
                chain = resolve_delta_chain(registry, cid)
                recipe_now = file_sha256(
                    Path(checkpoint["directory"]) / str(delta.get("recipe_name"))
                )
                if recipe_now != delta.get("recipe_file_sha256"):
                    problems.append(
                        f"{cid}: {delta.get('recipe_name')} digests {recipe_now[:16]}, "
                        f"sealed {str(delta.get('recipe_file_sha256'))[:16]} -- the "
                        "recipe was edited after registration"
                    )
                composed, _added = composed_parameter_count(chain)
                if composed != checkpoint.get("parameter_count"):
                    problems.append(
                        f"{cid}: composed parameter_count sealed "
                        f"{checkpoint.get('parameter_count'):,}, base + delta headers "
                        f"say {composed:,}"
                    )
            except (OcCheckpointError, OcSealError, OSError) as exc:
                problems.append(f"{cid}: delta chain: {exc}")
        # Re-DERIVE the two counts from the headers on disk rather than
        # re-dividing the sealed pair, which is a tautology: dividing a sealed
        # numerator by a sealed denominator can never disagree with the sealed
        # quotient.  This read costs ~0.00 s on a real 4B (it parses shard
        # headers, not shard bytes), so it runs on EVERY verify, not only under
        # --deep, which spends minutes re-hashing bytes.
        for label, count_key, source_key, dir_key in (
            ("child", "parameter_count", "parameter_count_source", "directory"),
            (
                "parent",
                "parent_parameter_count",
                "parent_parameter_count_source",
                "parent_directory",
            ),
        ):
            sealed_count = checkpoint.get(count_key)
            where = checkpoint.get(dir_key)
            if sealed_count is None or not where:
                continue
            if label == "child" and delta:
                continue  # composed count, re-derived through the chain above
            path = Path(where)
            if not path.is_dir():
                if checkpoint.get(source_key) == "derived-from-safetensors-headers":
                    problems.append(
                        f"{cid}: {label} counts were derived from headers under "
                        f"{path}, which is gone; the ratio can no longer be rechecked"
                    )
                continue
            try:
                derived, _per_file = safetensors_parameter_count(path)
            except OcSealError:
                if checkpoint_shards(path):
                    problems.append(f"{cid}: {label} headers under {path} do not read")
                elif checkpoint.get(source_key) == "derived-from-safetensors-headers":
                    problems.append(
                        f"{cid}: {label} counts claim to come from safetensors "
                        f"headers, but {path} has no *.safetensors"
                    )
                continue
            if derived != sealed_count:
                problems.append(
                    f"{cid}: {label} parameter_count sealed {sealed_count:,}, the "
                    f"safetensors headers under {path} say {derived:,} -- "
                    f"net_ratio {net} is wrong"
                )
    print(f"checkpoints {len(ids)}  records verified {checked}  "
          f"rows matched {rows_checked}  deep={'yes' if args.deep else 'no'}")
    for line in problems:
        print(f"  PROBLEM {line}")
    return 1 if problems else 0


def cmd_repair_row(args: argparse.Namespace) -> int:
    """Re-derive a sealed record's ledger row and append it.

    The only repair this tooling has, and it repairs the INDEX, never the
    evidence: the row is recomputed from the sealed record by the same
    `checkpoint_row` that wrote the original, so a row that already exists is
    refused rather than duplicated, and a record that does not seal cannot
    produce one at all.
    """

    registry, ledger = Path(args.registry), Path(args.ledger)
    chain = history(registry, args.checkpoint_id)
    if not chain:
        raise OcCheckpointError(f"{args.checkpoint_id} is not in the registry")
    index = _ledger_index(ledger) or {}
    wanted = [
        record for record in chain
        if not args.record_instance_id
        or record["record_instance_id"] == args.record_instance_id
    ]
    if args.record_instance_id and not wanted:
        raise OcCheckpointError(
            f"{args.record_instance_id} is not a record of {args.checkpoint_id}"
        )
    missing = []
    for record in wanted:
        instance = str(record["record_instance_id"])
        if instance in index:
            if index[instance] != record["receipt_sha"]:
                raise OcCheckpointError(
                    f"{instance} is already in the ledger under receipt "
                    f"{index[instance][:12]} and the record seals to "
                    f"{record['receipt_sha'][:12]}.  Rows are never rewritten; a "
                    "correction is a new record instance."
                )
            continue
        missing.append(record)
    print_plan(
        f"REPAIR ROWS {args.checkpoint_id}",
        [
            f"records            {len(wanted)}",
            f"already indexed    {len(wanted) - len(missing)}",
            f"rows to append     {[r['record_instance_id'] for r in missing]}",
            f"ledger             {ledger}",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm or not missing:
        return 0
    append_rows(ledger, [checkpoint_row(record) for record in missing])
    print(f"\nappended {len(missing)} row(s)")
    print(f"ledger sha256 {file_sha256(ledger)}")
    return 0


def cmd_delta_compose(args: argparse.Namespace) -> int:
    """Resolve, pin-check and structurally verify a composition; write its plan.

    Read-only on every checkpoint.  Without `--deep` the pins are checked
    against the SEALED digests and every op against the headers (seconds).
    With `--deep` every directory in the chain is re-hashed as well, which is
    the check that catches bytes that moved under an unchanged seal -- at
    55 GB that is minutes on root or Lexar and an hour on the NAS, so it is
    asked for, not defaulted.  The numeric composition itself is the engines'
    (the P1 reader composes per tensor from this plan); nothing here loads a
    tensor.
    """

    registry = Path(args.registry)
    plan = delta_plan(registry, args.checkpoint_id, deep=args.deep)
    root = plan["root"]
    print(f"=== DELTA COMPOSE {args.checkpoint_id} ({TOOL_VERSION}) ===")
    print(f"  root     {root['checkpoint_id']}  {root['directory_sha256'][:16]}  "
          f"{len(root['files'])} shard(s)  {root['directory']}")
    for link in plan["deltas"]:
        ops: dict[str, int] = {}
        for entry in link["entries"]:
            ops[entry["op"]] = ops.get(entry["op"], 0) + 1
        print(f"  delta    {link['checkpoint_id']}  {link['directory_sha256'][:16]}  "
              f"onto {link['base_checkpoint_id']} pinned "
              f"{link['base_directory_sha256'][:16]}  ops {ops}  "
              f"+{link['added_parameters']:,} params")
    print(f"  pins     hold ({'sealed digests AND live re-hash' if args.deep else 'sealed digests; --deep re-hashes the bytes'})")
    print(f"  composed parameter_count {plan['composed_parameter_count']:,}")
    print(f"  plan     {plan['receipt_sha']}")
    if args.plan_out:
        out = Path(args.plan_out)
        if out.exists() and not args.overwrite:
            raise OcCheckpointError(f"{out} exists; pass --overwrite to replace it")
        write_sealed(out, plan)
        print(f"  wrote    {out}")
    return 0


def cmd_package_delta(args: argparse.Namespace) -> int:
    """A self-contained, verifiable copy of one delta: bytes, recipe, record, plan.

    `<out>/delta/` is a byte copy of the delta directory (re-digested and
    asserted equal to the sealed digest), `RECORD.json` the sealed head record,
    `PLAN.json` the sealed composition plan, and `MANIFEST.json` the same
    manifest `oc_package.py` writes, so `oc_package.py verify --package <out>`
    checks it.  The base is NOT copied -- it is named, with its digest, in both
    the record and the plan; a delta package is only meaningful next to that
    base, and copying 55 GB to say so is what deltas exist to avoid.
    """

    from oc_package import verify_manifest, write_manifest  # local: no cycle

    registry = Path(args.registry)
    head = current(registry, args.checkpoint_id)
    checkpoint = head["checkpoint"]
    delta = checkpoint.get("delta")
    if not delta:
        raise OcCheckpointError(
            f"{args.checkpoint_id} is not a delta checkpoint; use oc_package.py build"
        )
    out = resolve_path(args.out)
    assert_not_protected(args.out, what="--out")
    if out.exists():
        raise OcCheckpointError(f"{out} already exists; a package is never rewritten")
    source_dir = Path(checkpoint["directory"])
    live, entries = directory_digest(source_dir)
    if live != checkpoint["directory_sha256"]:
        raise OcCheckpointError(
            f"{args.checkpoint_id}: {source_dir} digests {live[:16]}, sealed "
            f"{checkpoint['directory_sha256'][:16]}; the bytes moved since the seal"
        )
    plan = delta_plan(registry, args.checkpoint_id, deep=False)
    print_plan(
        f"PACKAGE DELTA {args.checkpoint_id}",
        [
            f"delta        {source_dir}  {live[:16]}  {len(entries)} file(s) "
            f"{human_bytes(sum(int(e['bytes']) for e in entries))}",
            f"base         {delta['base_checkpoint_id']}  pinned "
            f"{delta['base_directory_sha256'][:16]} (not copied)",
            f"plan         {plan['receipt_sha'][:16]}  composed "
            f"{plan['composed_parameter_count']:,} params",
            f"out          {out}",
        ],
        confirmed=args.confirm,
    )
    if not args.confirm:
        return 0
    copy_dir = out / "delta"
    copy_dir.mkdir(parents=True)
    for entry in entries:
        destination = copy_dir / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / entry["path"], destination)
    copied, _copied_entries = directory_digest(copy_dir)
    if copied != live:
        raise OcCheckpointError(
            f"the packaged copy digests {copied[:16]}, the delta {live[:16]}; left "
            f"at {out} for inspection"
        )
    write_sealed(out / "RECORD.json", head)
    write_sealed(out / "PLAN.json", plan)
    (out / "README.txt").write_text(
        "\n".join(
            [
                f"Delta checkpoint {args.checkpoint_id} ({TOOL_VERSION}).",
                f"Base: {delta['base_checkpoint_id']} at directory_sha256 "
                f"{delta['base_directory_sha256']}.",
                f"Chain: {' -> '.join(delta['chain'])} -> {args.checkpoint_id}.",
                "delta/ is a byte copy of the registered delta directory; its "
                f"digest is {live}.",
                "Compose it onto the base with the P1 reader (--plan PLAN.json), or "
                "re-check it with: oc_checkpoint.py delta-compose --checkpoint-id "
                f"{args.checkpoint_id} --deep",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(out, head)
    problems = verify_manifest(out)
    if problems:
        raise OcCheckpointError(f"the package at {out} does not verify: {problems[:3]}")
    print(f"\npackage {out}")
    print(f"package_sha256 {manifest['package_sha256']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=TOOL_VERSION)
    ap.add_argument("--root", type=Path, default=None,
                    help="campaign root; defaults to $OC_ROOT")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(parser: argparse.ArgumentParser, *, ledger: bool = True) -> None:
        parser.add_argument("--registry", type=Path, default=None)
        if ledger:
            parser.add_argument("--ledger", type=Path, default=None)
        parser.add_argument("--notes", default="")
        parser.add_argument("--partially-capable-floor", type=int, default=1)

    p = sub.add_parser("register", help="register a built checkpoint")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--parent-directory", type=Path, default=None)
    p.add_argument("--parent-code", default=None)
    p.add_argument("--parent-provenance-sha256", default=None)
    p.add_argument("--parent-checkpoint-id", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--wave", default=None)
    p.add_argument("--parameter-count", default="auto")
    p.add_argument("--parent-parameter-count", default="auto")
    p.add_argument(
        "--allow-asserted-parameter-counts",
        action="store_true",
        help="register a checkpoint whose parameter counts cannot be read from "
        "safetensors headers; the record, the row and the scorecard then say "
        "asserted-by-operator against that ratio",
    )
    p.add_argument("--net-ratio", type=float, default=None)
    p.add_argument("--component", action="append", default=[],
                   metavar="NAME:PARENT_PARAMS:CHILD_PARAMS[:TENSOR_PREFIX]",
                   help="a folded component; with a tensor-name prefix both "
                        "counts are re-derived from the safetensors headers")
    p.add_argument("--passed-through", action="append", default=[])
    p.add_argument("--accounting-slack", type=int, default=0,
                   help="parameters a fold ADDED elsewhere, stated so the "
                        "component saving may exceed the measured drop by that "
                        "much; recorded in the sealed record")
    p.add_argument("--quantization", default="none")
    p.add_argument("--recipe", type=Path, default=None)
    p.add_argument("--recipe-command", default=None)
    p.add_argument("--lineage", action="append", default=[])
    p.add_argument("--frontier-holder", action="store_true")
    p.add_argument("--embed-manifest", action="store_true",
                   help="store every file's digest inside the record")
    p.add_argument("--delta-base", default=None, metavar="BASE_CHECKPOINT_ID",
                   help="register --directory as a DELTA over this base (a clone, "
                        "R2): the directory carries DELTA.json, whose pinned base "
                        "digest must equal the base's sealed digest; the parameter "
                        "count is the composed model's")
    p.add_argument("--delta-base-sha256", default=None,
                   help="optional second pin: refused unless it equals the base's "
                        "sealed directory_sha256")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser(
        "delta-compose",
        help="resolve a delta chain, check every base pin and op, write the plan",
    )
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--deep", action="store_true",
                   help="also re-hash every directory in the chain")
    p.add_argument("--plan-out", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_delta_compose)

    p = sub.add_parser("package-delta", help="a verifiable copy of one delta")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_package_delta)

    p = sub.add_parser("clone", help="byte clone before a training stage (R2)")
    common(p)
    p.add_argument("--source", required=True)
    p.add_argument("--stage-about-to-run", required=True, choices=["D", "T"])
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--clone-index", type=int, default=None)
    p.add_argument("--mode", default="copy", choices=["copy", "hardlink"])
    p.add_argument("--allow-hardlink", action="store_true")
    p.add_argument("--allow-unreviewed-source", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("record-review", help="attach a review; status is computed")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--review-json", type=Path, required=True)
    p.add_argument("--status", default=None, choices=list(STATUSES))
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_record_review)

    p = sub.add_parser("set-status", help="Cyril's ruling / frontier flag only")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--status", default=None, choices=list(STATUSES))
    p.add_argument("--cyril-ruling", default=None)
    p.add_argument("--frontier-holder", type=int, default=None, choices=[0, 1])
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_set_status)

    p = sub.add_parser("delete", help="delete a bad checkpoint (R4)")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--rebuild-command", required=True)
    p.add_argument("--retained-source", required=True)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("list", help="every checkpoint's current state")
    common(p)
    p.add_argument("--stage", default=None, choices=list(STAGES))
    p.add_argument("--status", default=None, choices=list(STATUSES))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="one checkpoint's current record or chain")
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--history", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("verify", help="re-verify seals, chains, rows and digests")
    common(p)
    p.add_argument("--checkpoint-id", default=None)
    p.add_argument("--deep", action="store_true", help="re-hash every directory")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "repair-row", help="re-derive a sealed record's missing ledger row"
    )
    common(p)
    p.add_argument("--checkpoint-id", required=True)
    p.add_argument("--record-instance-id", default=None)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_repair_row)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = oc_root(args.root)
    if getattr(args, "registry", None) is None:
        args.registry = root / "registry"
    if getattr(args, "ledger", None) is None:
        args.ledger = root / "ledger.jsonl"
    try:
        return int(args.func(args))
    except (OcCheckpointError, OcSealError, OcLedgerError) as exc:
        # OcLedgerError travels this far because `verify` and `repair-row` read
        # the ledger: a ledger that fails its own byte check is a REFUSAL with a
        # sentence, not a traceback out of a verification command.
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
