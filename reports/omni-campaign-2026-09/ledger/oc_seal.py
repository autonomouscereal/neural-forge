#!/usr/bin/env python3
"""Sealing, digesting and locking primitives for the Chimera Omni campaign.

Pure standard library, on purpose.  The LFM2 campaign's sealer lives inside
`neural_forge`, whose package `__init__` imports torch; this tree has to run on
a workstation with no torch and inside a bare `python3` on the server before the
lab venv is warm, so nothing here imports anything the interpreter does not
ship.

Three things are copied EXACTLY from `neural_forge.dynamic_circuit_ir` and
`neural_forge.matrix_cell_receipt`, because a digest that does not reproduce the
ancestor campaign's digest is a different instrument wearing its name:

* `canonical_json_bytes` -- `json.dumps(v, allow_nan=False, ensure_ascii=True,
  separators=(",",":"), sort_keys=True).encode("ascii")`;
* the seal -- the body is digested with `receipt_sha` held at the literal
  ``"self"``, so a sealed file's digest is reproducible from the file by
  resetting one field;
* the write -- canonical bytes to ``<path>.partial``, then an atomic replace,
  so a reader never sees half a receipt.

`test_oc_ledger.py` pins the canonical byte string of a fixture by value, so a
"harmless" reformat of the dumps call is a test failure and not a silent fork.

What this module adds for the omni campaign, all of it forced by something the
LFM2 tooling could assume and this one cannot:

* **directory digests.**  A 27B child is sharded across
  `model-0000N-of-000NN.safetensors`; `child_tensors_sha256` over a single
  `model.safetensors` does not exist for it.  `directory_digest` is sha256 over
  the canonical JSON of the sorted ``[[relative path, file sha256], ...]``
  list, which is the form the LFM2 retention proof already published beside the
  single-file digest (`child_directory_sha256 24b8a5b1...`).
* **hardlink refusal.**  `deploy.py --verify` refuses `st_nlink != 1` because a
  file reachable under a second name is a file another process can rewrite
  under you.  A digest that is a claim about a checkpoint inherits that rule.
* **parameter counts from safetensors headers**, so a ratio's denominator can be
  computed without loading a model -- on the workstation there is no torch, and
  on the server loading 27B to count it is minutes of GPU for an integer.
* **a single-writer lock.**  The LFM2 ledger got its single-writer discipline
  from per-child shards plus one merge; the omni registry additionally has a
  live CLI that appends rows, so the append path takes an exclusive lock.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The campaign tag every omni row and record carries.  The LFM2 ledger's rows
#: never carry it; that is the point of a new column rather than a new schema.
CAMPAIGN = "omni-2026-09"

#: Ancestor digests, cited and never re-derived here (R3 spec section 0).
ANCESTOR = {
    "repo_head": "1779b7b9c64adedc2d803d5c4c48bd38369c8c97",
    "lfm2_ledger_sha256": (
        "d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea"
    ),
    "protocol_sha256": (
        "e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b"
    ),
    "prompt_pack_sha256": (
        "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Files that are never part of a checkpoint's identity: a crash-truncated
#: atomic write and the interpreter's own droppings.  NOTE that this is the one
#: place a directory digest is NOT "every byte under the root": a file whose
#: name ends in one of these suffixes is invisible to `directory_digest` and
#: therefore to `verify --deep`.  The list is deliberately short and is stated
#: in `CHECKPOINT_POLICY.md` R0 rather than left as a surprise.
DIGEST_EXCLUDE_SUFFIXES = (".partial", ".pyc", ".lock")
DIGEST_EXCLUDE_DIRS = ("__pycache__", ".git", ".ipynb_checkpoints")


class OcSealError(RuntimeError):
    """A seal, digest or lock refused."""


# --------------------------------------------------------------------------
# canonical json + the seal (byte-identical to the LFM2 campaign)
# --------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OcSealError("payload is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def now_utc() -> str:
    """UTC, milliseconds, `Z` -- the `sealed_at` format the ledger sorts by."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address a record.  `sealed_at` is set once and never moved."""

    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["sealed_at"] = sealed.get("sealed_at") or now_utc()
    sealed["receipt_sha"] = "self"
    sealed["receipt_sha"] = canonical_sha256(sealed)
    return sealed


def verify_seal(body: Mapping[str, Any]) -> str:
    """Recompute a sealed record's digest and fail closed on a mismatch."""

    claimed = body.get("receipt_sha")
    if not isinstance(claimed, str) or _SHA256.match(claimed) is None:
        raise OcSealError("record is not sealed")
    probe = json.loads(json.dumps(dict(body), sort_keys=True))
    probe["receipt_sha"] = "self"
    recomputed = canonical_sha256(probe)
    if recomputed != claimed:
        raise OcSealError(
            f"seal mismatch: claims {claimed}, recomputes {recomputed}"
        )
    return recomputed


def write_sealed(path: str | Path, body: Mapping[str, Any]) -> Path:
    """Atomically write one sealed record, then read it back and re-verify.

    The read-back is NOTES-MX-AO section 9 applied here: every artifact a
    command writes is parsed by the command that writes it, in the same
    invocation.  Wave 35's corruption bug was a file that serialised fine and
    was never re-read until the wave after.
    """

    verify_seal(body)
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(dict(body)))
    temporary.replace(destination)
    reread = read_sealed(destination)
    if reread["receipt_sha"] != body["receipt_sha"]:
        raise OcSealError(f"{destination} does not read back as written")
    return destination


def read_sealed(path: str | Path) -> dict[str, Any]:
    """Read a sealed record, refusing non-canonical bytes and broken seals.

    The byte check matters as much as the seal: a record that was hand-edited
    and re-indented would still verify if it were re-serialised on read, and
    then the file on disk and the digest in the ledger would disagree about
    nothing visible.
    """

    source = Path(path).expanduser().absolute()
    raw = source.read_bytes()
    body = json.loads(raw.decode("utf-8"))
    if canonical_json_bytes(body) != raw:
        raise OcSealError(f"{source} is not canonical JSON bytes on disk")
    verify_seal(body)
    return body


# --------------------------------------------------------------------------
# file and directory digests
# --------------------------------------------------------------------------


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _digest_files(root: Path) -> list[Path]:
    """Every ordinary file under `root`, sorted, refusing every symlink.

    THE ONE ENUMERATOR.  The directory digest, the parameter count and the
    per-tensor index all walk through here, so they can never disagree about
    which files are in a checkpoint.  They used to disagree: the digest walked
    with `os.walk` and the counts walked with `Path.rglob`, and NEITHER follows
    a symlinked directory, so a checkpoint whose shards lived under a symlinked
    `weights/` sealed a `directory_sha256` and a `parameter_count` that covered
    none of those bytes -- and `clone` then copied the same visible subset,
    re-digested it, found the two digests equal and wrote a receipt asserting a
    byte-identical clone that was missing the weights.

    A symlink is therefore refused here, file or directory, and refused loudly:
    a digest that silently skips bytes is a digest that can lie, and net_ratio
    is the campaign's central number.  Copy the tree in, or register its target
    directly.
    """

    out: list[Path] = []
    for base, dirs, names in os.walk(root):
        for name in sorted(dirs):
            if name in DIGEST_EXCLUDE_DIRS:
                continue
            path = Path(base) / name
            if path.is_symlink():
                raise OcSealError(
                    f"{path} is a symlinked directory; a digest may not follow "
                    "one, and skipping it would seal a count over bytes it "
                    "never read.  Copy the tree in or register its target."
                )
        dirs[:] = sorted(d for d in dirs if d not in DIGEST_EXCLUDE_DIRS)
        for name in sorted(names):
            if name.endswith(DIGEST_EXCLUDE_SUFFIXES):
                continue
            path = Path(base) / name
            if path.is_symlink():
                raise OcSealError(f"{path} is a symlink; a digest may not follow one")
            out.append(path)
    return out


def checkpoint_shards(root: Path) -> list[Path]:
    """The `*.safetensors` of a checkpoint, through `_digest_files`.

    Not `rglob`: see the note there.  The shard set and the digested file set
    are the same walk by construction.
    """

    return sorted(p for p in _digest_files(root) if p.name.endswith(".safetensors"))


def directory_manifest(
    root: str | Path, *, allow_hardlinks: bool = False
) -> list[dict[str, Any]]:
    """Every ordinary file under `root`, with size and digest, sorted by path.

    Symlinks are refused outright -- FILES AND DIRECTORIES ALIKE, in
    `_digest_files`, which is the single enumerator this and the parameter
    counts share -- and hardlinks are refused by default: a checkpoint
    reachable under a second name is a checkpoint another process can rewrite
    while this one holds its digest.  `allow_hardlinks` exists only for the
    hardlink clone mode, which records the fact rather than hiding it.

    The `path.is_symlink()` line below is now unreachable through
    `_digest_files` and is kept as an assertion: if a future enumerator stops
    refusing, this refuses.
    """

    base = Path(root).expanduser().absolute()
    if not base.is_dir():
        raise OcSealError(f"{base} is not a directory")
    entries: list[dict[str, Any]] = []
    for path in _digest_files(base):
        stat = path.lstat()
        if path.is_symlink():
            raise OcSealError(f"{path} is a symlink; a digest may not follow one")
        if stat.st_nlink != 1 and not allow_hardlinks:
            raise OcSealError(
                f"{path} has st_nlink={stat.st_nlink}; a file reachable under a "
                "second name is refused (deploy.py's rule)"
            )
        entries.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": stat.st_size,
                "sha256": file_sha256(path),
                "nlink": stat.st_nlink,
            }
        )
    if not entries:
        raise OcSealError(f"{base} contains no files to digest")
    return entries


def digest_of_manifest(entries: Sequence[Mapping[str, Any]]) -> str:
    """sha256 over the sorted ``[[path, sha256], ...]`` pairs, canonically."""

    pairs = sorted([entry["path"], entry["sha256"]] for entry in entries)
    return canonical_sha256(pairs)


def directory_digest(
    root: str | Path, *, allow_hardlinks: bool = False
) -> tuple[str, list[dict[str, Any]]]:
    entries = directory_manifest(root, allow_hardlinks=allow_hardlinks)
    return digest_of_manifest(entries), entries


# --------------------------------------------------------------------------
# parameter counts without torch
# --------------------------------------------------------------------------

#: Bits per element for every dtype safetensors names.  Used by
#: `_validate_header_spans`: the parameter count itself is the product of the
#: shape and needs no dtype, but a count read out of a header nobody checked is
#: a confident number taken off a truncated file.
_DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64,
}


def _shape_elements(path: Path, name: str, spec: Mapping[str, Any]) -> int:
    shape = spec.get("shape")
    if not isinstance(shape, list):
        raise OcSealError(f"{path}: tensor {name} has no shape")
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return elements


def _validate_header_spans(
    path: Path, header: Mapping[str, Any], *, data_start: int, file_size: int
) -> None:
    """Every tensor's byte span must match its shape and lie inside the file.

    This is the check `_DTYPE_BITS` exists for, and it is not decoration: a
    parameter count is the denominator of every ratio in the campaign, and a
    truncated or half-written shard still parses as a header and still yields a
    confident integer.  A dtype the table does not name (K3's MXFP4 experts,
    say) is still bounds-checked; only the exact-width test is skipped, and the
    element count never depended on the dtype anyway.
    """

    region = file_size - data_start
    if region < 0:
        raise OcSealError(f"{path}: the header runs past the end of the file")
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, Mapping):
            raise OcSealError(f"{path}: tensor {name} is not an object")
        offsets = spec.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise OcSealError(f"{path}: tensor {name} has no data_offsets pair")
        start, end = int(offsets[0]), int(offsets[1])
        if not 0 <= start <= end <= region:
            raise OcSealError(
                f"{path}: tensor {name} spans [{start},{end}) outside the "
                f"{region}-byte tensor region -- the file is truncated or corrupt"
            )
        bits = _DTYPE_BITS.get(str(spec.get("dtype")))
        if bits is None:
            continue
        expected = (_shape_elements(path, name, spec) * bits + 7) // 8
        if end - start != expected:
            raise OcSealError(
                f"{path}: tensor {name} declares {end - start} bytes for a "
                f"{spec.get('shape')} {spec.get('dtype')} tensor ({expected} "
                "expected); the header does not describe the bytes"
            )


def safetensors_header(path: str | Path) -> dict[str, Any]:
    """The JSON header of one safetensors file, read without loading tensors.

    The byte spans are validated against the file's real size before the header
    is returned, so a count taken off it is a count off a file that is at least
    as long as it claims to be.
    """

    source = Path(path)
    file_size = source.stat().st_size
    with open(source, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise OcSealError(f"{source} is too short to be safetensors")
        (length,) = struct.unpack("<Q", raw_length)
        if length <= 0 or length > (1 << 31):
            raise OcSealError(f"{source} declares an implausible header length {length}")
        blob = handle.read(length)
    if len(blob) != length:
        raise OcSealError(
            f"{source} declares a {length}-byte header but carries {len(blob)}"
        )
    header = json.loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise OcSealError(f"{source} header is not an object")
    _validate_header_spans(source, header, data_start=8 + length, file_size=file_size)
    return header


def safetensors_parameter_count(root: str | Path) -> tuple[int, list[dict[str, Any]]]:
    """Total elements over every ``*.safetensors`` under `root`.

    This is the denominator of every ratio in the campaign, so it is computed
    from the bytes on disk rather than from a config or a build receipt's own
    claim about itself.  Shared/tied tensors are counted once per tensor entry,
    which is what `parameters_counted_live` counts too when the tie is declared
    in the config; a child whose live count disagrees is a refusal at seal
    time, not a silent adjustment here.

    The shards come from `checkpoint_shards`, i.e. from the SAME walk the
    directory digest uses, so the count and the digest cannot cover different
    file sets -- and a symlinked shard directory is a refusal here too rather
    than a count that quietly omits it.
    """

    base = Path(root).expanduser().absolute()
    shards = checkpoint_shards(base)
    if not shards:
        raise OcSealError(f"{base} contains no *.safetensors")
    total = 0
    per_file: list[dict[str, Any]] = []
    for shard in shards:
        header = safetensors_header(shard)
        count = 0
        tensors = 0
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            count += _shape_elements(shard, name, spec)
            tensors += 1
        total += count
        per_file.append(
            {
                "path": shard.relative_to(base).as_posix(),
                "tensors": tensors,
                "parameters": count,
            }
        )
    return total, per_file


def safetensors_tensor_counts(root: str | Path) -> dict[str, int]:
    """``{tensor name: elements}`` over every shard under `root`.

    This is what lets a `--component` claim be DERIVED rather than asserted:
    given a tensor-name prefix, the parent's and the child's element counts for
    that component are sums over this index, read from the same headers the
    net ratio's denominator is read from.  A component with no prefix stays an
    operator assertion, and the record says which of the two it is.

    Same walk as `safetensors_parameter_count` and the directory digest; see
    `_digest_files`.
    """

    base = Path(root).expanduser().absolute()
    shards = checkpoint_shards(base)
    if not shards:
        raise OcSealError(f"{base} contains no *.safetensors")
    counts: dict[str, int] = {}
    for shard in shards:
        for name, spec in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            counts[name] = counts.get(name, 0) + _shape_elements(shard, name, spec)
    return counts


def prefix_parameter_count(counts: Mapping[str, int], prefix: str) -> tuple[int, int]:
    """``(parameters, tensors)`` for every tensor whose name starts with `prefix`."""

    total = tensors = 0
    for name, elements in counts.items():
        if name.startswith(prefix):
            total += int(elements)
            tensors += 1
    return total, tensors


# --------------------------------------------------------------------------
# single-writer lock
# --------------------------------------------------------------------------


class WriterLock:
    """An exclusive lock file for the one process allowed to append.

    `O_CREAT | O_EXCL` on a local filesystem, which is what the campaign root
    is (`/home/cereal/omni-campaign`, 615 GB local).  It is deliberately NOT
    made safe for the CIFS share -- nothing in this campaign writes there.

    A stale lock is never broken automatically.  The estate's SSH lease has the
    same shape: see `[SSH BUSY]`, wait, retry, and if it never clears a human
    looks at who holds it.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 0.0) -> None:
        self.path = Path(path).expanduser().absolute()
        self.timeout_s = float(timeout_s)
        self._fd: int | None = None

    def owner(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def __enter__(self) -> "WriterLock":
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
                os.write(
                    self._fd,
                    canonical_json_bytes(
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "acquired_at": now_utc(),
                        }
                    ),
                )
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if time.time() >= deadline:
                    held = self.owner()
                    raise OcSealError(
                        f"ledger writer lock held: {self.path} {held}.  One "
                        "process appends at a time; wait, or remove the lock by "
                        "hand after checking the owner is gone."
                    ) from exc
                time.sleep(0.25)

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def oc_root(explicit: str | Path | None = None) -> Path:
    """`--root`, else `$OC_ROOT`, else the server's campaign root."""

    if explicit:
        return Path(explicit).expanduser().absolute()
    return Path(os.environ.get("OC_ROOT", "/home/cereal/omni-campaign")).absolute()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.match(value) is None:
        raise OcSealError(f"{label} is not a sha256: {value!r}")
    return value


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:,.1f} TiB"


def iter_json_lines(path: str | Path) -> Iterable[tuple[int, bytes]]:
    """Yield ``(1-based line number, raw bytes)`` for a JSONL file."""

    raw = Path(path).read_bytes() if Path(path).is_file() else b""
    for number, line in enumerate(raw.split(b"\n"), start=1):
        if line.strip():
            yield number, line
