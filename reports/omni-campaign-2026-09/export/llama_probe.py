"""Shared measurement helpers for the Chimera Omni export + envelope lane.

Stdlib only (the lab venv is torch/transformers; this module must also run on a bare
python3.12 with nothing installed, so that the log parser and the receipt writer can be
unit-tested anywhere).

Everything here MEASURES and RECORDS. Nothing here decides. No function in this module
returns a pass/fail verdict; the callers write numbers into receipts and Cyril reads them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SCHEMA = "chimera_omni.export_lane.v1"

# --------------------------------------------------------------------------------------
# receipts
# --------------------------------------------------------------------------------------


def canonical_json_bytes(payload):
    """Canonical JSON: sorted keys, compact separators, UTF-8, trailing newline."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Weight files are the POINT of a checkpoint digest. The first version of this allowlist
# carried only (.safetensors, .json, .txt, .model, .jinja), so a checkpoint sharded as
# `pytorch_model-*.bin` digested to a hash covering nothing but its config and tokenizer:
# two different sets of weights under the same config collided on the lineage key. Every
# common weight container is included now, and the filter itself is recorded in the result
# so a reader can see which files a digest did and did not cover.
DIGEST_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npz", ".h5", ".msgpack",
    ".json", ".txt", ".model", ".jinja", ".py",
)


def directory_digest(root, suffixes=DIGEST_SUFFIXES):
    """sha256 over the sorted (relative path, file sha256) pairs of a checkpoint directory.

    Same construction as the R3 review-ledger protocol's checkpoint digest, so an export
    receipt can be joined to a fold/train receipt by digest.

    `suffixes=None` digests every file. Whatever filter is used is echoed back in
    `suffix_filter`, because a digest whose coverage is invisible is not a lineage key.
    """
    entries = []
    skipped = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if suffixes and not name.endswith(tuple(suffixes)):
                skipped.append(rel)
                continue
            entries.append((rel, sha256_file(full)))
    entries.sort()
    skipped.sort()
    joined = "\n".join("%s %s" % (rel, digest) for rel, digest in entries)
    return {
        "directory_sha256": sha256_bytes(joined.encode("utf-8")),
        "file_count": len(entries),
        "files": [{"path": rel, "sha256": digest} for rel, digest in entries],
        "suffix_filter": list(suffixes) if suffixes else None,
        "skipped_file_count": len(skipped),
        "skipped_files": skipped[:50],
    }


def write_receipt(path, payload, keep_versions=True):
    """Atomically write a canonical-JSON receipt; return its own sha256.

    A receipt is also written to an immutable sidecar `<stem>.<sha8><ext>`. The canonical
    name is `os.replace`d in place, so before this a receipt hash published in a document
    (or cited by ANOTHER receipt, as the R3 control cites the build receipt) stopped
    resolving the moment the tool was re-run: the export lane shipped with a control naming
    build receipt `c1ab3f9d` while the file on disk hashed to `4a9554fe`, and the older
    bytes survived only because someone had made a `.bak` by hand. The sidecar is written
    first, is never overwritten (its name IS its content hash), and makes any cited hash
    resolvable to bytes.
    """
    payload = dict(payload)
    payload.setdefault("schema", SCHEMA)
    blob = canonical_json_bytes(payload)
    digest = sha256_bytes(blob)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    def _atomic(target):
        tmp = target + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)

    if keep_versions:
        stem, ext = os.path.splitext(path)
        versioned = "%s.%s%s" % (stem, digest[:8], ext or ".json")
        if not os.path.exists(versioned):
            _atomic(versioned)
    _atomic(path)
    return digest


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_tag(tag):
    """Tags become filenames (and a docker container name); refuse separators/traversal.

    Every tool in this lane interpolates `--tag` straight into an output path, and
    `write_receipt` makedirs + os.replace, so `--tag ../../x` wrote outside `--out-dir`.
    Only `export_hf_to_gguf.py` validated it; this is the shared version the others use.
    """
    if not tag or tag in (".", ".."):
        raise ValueError("tag must be a non-empty name")
    if any(char in tag for char in ("/", "\\", os.sep, "\0")):
        raise ValueError("tag must not contain path separators: %r" % tag)
    if tag.startswith("-"):
        raise ValueError("tag must not start with '-': %r" % tag)
    return tag


# --------------------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------------------


def run(cmd, timeout=60, cwd=None, env=None):
    """Run a command with a hard timeout. Returns (rc, stdout, stderr).

    rc 124 mirrors coreutils `timeout` when we time out ourselves.
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err + "\n[timeout after %ss]" % timeout
    except FileNotFoundError as exc:
        return 127, "", str(exc)


# --------------------------------------------------------------------------------------
# llama.cpp server log analysis
#
# llama.cpp prints no line that says "the flash-attention kernel executed". The strongest
# available evidence, in order:
#   1. the context line `llama_context: flash_attn = enabled|1` (the request was accepted),
#   2. the ABSENCE of any fallback / unsupported warning mentioning flash attention,
#   3. quantized KV acceptance: llama.cpp refuses `-ctk/-ctv` != f16 unless flash_attn is
#      on ("K/V cache quantization requires flash_attn"), so a server that LOADED with
#      q8_0/q4_0 KV on CUDA is running the FA path.
# The parser reports all three as separate fields plus the verbatim evidence lines. It
# never returns a verdict.
# --------------------------------------------------------------------------------------

_RE_FLASH_ATTN = re.compile(r"flash_attn\s*=\s*(\S+)", re.IGNORECASE)
_RE_KV_TYPES = re.compile(r"type_k\s*=\s*'?([A-Za-z0-9_]+)'?.*?type_v\s*=\s*'?([A-Za-z0-9_]+)'?", re.IGNORECASE | re.DOTALL)
_RE_TYPE_K = re.compile(r"type_k\s*=\s*'?([A-Za-z0-9_]+)'?", re.IGNORECASE)
_RE_TYPE_V = re.compile(r"type_v\s*=\s*'?([A-Za-z0-9_]+)'?", re.IGNORECASE)
_RE_KV_BUF = re.compile(r"KV (?:self )?(?:buffer )?size\s*=\s*([0-9.]+)\s*MiB", re.IGNORECASE)
_RE_KV_BUF_DEV = re.compile(r"([A-Za-z0-9_]+)\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB", re.IGNORECASE)
_RE_OFFLOAD = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", re.IGNORECASE)
# n_ctx and n_ctx_slot are DIFFERENT numbers: with -np N the 2026 server prints only
# `n_ctx_slot = <total>/N`. Reading n_ctx_slot as n_ctx silently divides the context by the
# parallel count, so the two are parsed separately and the source is recorded.
_RE_NCTX = re.compile(r"\bn_ctx\s*=\s*(\d+)")
_RE_NCTX_SLOT = re.compile(r"\bn_ctx_slot\s*=\s*(\d+)")
_RE_NSLOTS = re.compile(r"\bn_slots\s*=\s*(\d+)")
_RE_KV_UNIFIED = re.compile(r"kv_unified\s*=\s*'?(\w+)'?", re.IGNORECASE)
_RE_LISTENING = re.compile(
    r"server is listening|HTTP server is listening|starting the main loop|listening on http",
    re.IGNORECASE,
)
_RE_SLOTS_IDLE = re.compile(r"all slots are idle", re.IGNORECASE)
_RE_MODEL_LOADED = re.compile(r"model loaded|loading model done|srv\s+load_model", re.IGNORECASE)
# `srv load_model: loading model '...'` is the START of the load and is also matched by
# _RE_MODEL_LOADED; timestamping off that would report a 0.4 s load for a 55 s load.
_RE_MODEL_LOADED_EVENT = re.compile(r"model loaded|loading model done", re.IGNORECASE)

# Anything on this list is recorded verbatim as fallback / unsupported evidence.
_FALLBACK_PATTERNS = (
    ("flash_attn_unsupported", re.compile(r"flash.?attn.*(not supported|unsupported|disabl|requires|incompatible)", re.IGNORECASE)),
    ("flash_attn_off", re.compile(r"flash.?attention.*(not supported|unsupported|disabled|falling back)", re.IGNORECASE)),
    ("cpu_fallback", re.compile(r"falling back to (the )?CPU|using CPU backend for|not supported.*fallback", re.IGNORECASE)),
    # The refusal string was guessed from older llama.cpp ("K/V cache quantization
    # requires flash_attn") until the negative control on commit b96806d9 printed the real
    # one: `E llama_init_from_model: quantized V cache requires flash_attn to be enabled`.
    # The guessed pattern would NOT have matched it. Both spellings are kept.
    ("kv_quant_refused", re.compile(
        r"cache quantization requires|quantized\s+[KV]{1,3}\s+cache requires|"
        r"requires flash_attn", re.IGNORECASE)),
    ("cuda_error", re.compile(r"CUDA error|ggml_cuda.*(error|not implemented|unsupported)", re.IGNORECASE)),
)


# The 2026 llama.cpp log prefixes every line with an uptime stamp `M.SS.mmm.uuu `
# (e.g. `0.55.687.716 I srv  llama_server: model loaded` = 55.687716 s after start).
_RE_LOG_TIME = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?\s")


def parse_log_time(line):
    """Seconds-since-start from a 2026 llama.cpp log prefix, or None if absent.

    This is the server's OWN clock, which is what "how long did the model take to load"
    should be read from — a health-poll interval is an upper bound, not a load time.
    """
    match = _RE_LOG_TIME.match(line or "")
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    millis = int(match.group(3))
    micros = int(match.group(4)) if match.group(4) else 0
    return round(minutes * 60 + seconds + millis / 1000.0 + micros / 1e6, 6)


def analyse_server_log(text):
    """Extract every FA / KV / offload signal from a llama-server log.

    Returns a dict of measurements plus verbatim evidence lines. `flash_attn_state` is one
    of "enabled" / "disabled" / "auto" / "unknown" and is a READING of the log, never a
    verdict about the run.
    """
    lines = text.splitlines()
    out = {
        "flash_attn_state": "unknown",
        "flash_attn_lines": [],
        "kv_type_k": None,
        "kv_type_v": None,
        "kv_buffer_mib_total": None,
        "kv_buffer_mib_by_device": {},
        "layers_offloaded": None,
        "layers_total": None,
        "n_ctx": None,
        "n_ctx_source": None,
        "n_ctx_slot": None,
        "n_slots": None,
        "n_ctx_from_slots": None,
        "kv_unified": None,
        "server_listening": False,
        "listening_at_log_s": None,
        "model_loaded_at_log_s": None,
        "last_log_time_s": None,
        "slots_idle_seen": False,
        "model_loaded_seen": False,
        "fallback_evidence": [],
        "kv_quantized": False,
        "kv_quant_implies_flash_attn": False,
        "line_count": len(lines),
    }

    kv_total = 0.0
    saw_kv_buf = False

    for line in lines:
        match = _RE_FLASH_ATTN.search(line)
        if match:
            raw = match.group(1).strip().strip(",'\"").lower()
            out["flash_attn_lines"].append(line.strip())
            if raw in ("1", "true", "enabled", "on", "yes"):
                out["flash_attn_state"] = "enabled"
            elif raw in ("0", "false", "disabled", "off", "no"):
                out["flash_attn_state"] = "disabled"
            elif raw == "auto":
                # keep an explicit enabled/disabled reading if one already appeared
                if out["flash_attn_state"] == "unknown":
                    out["flash_attn_state"] = "auto"

        if out["kv_type_k"] is None:
            mk = _RE_TYPE_K.search(line)
            if mk:
                out["kv_type_k"] = mk.group(1).lower()
        if out["kv_type_v"] is None:
            mv = _RE_TYPE_V.search(line)
            if mv:
                out["kv_type_v"] = mv.group(1).lower()

        mdev = _RE_KV_BUF_DEV.search(line)
        if mdev:
            device = mdev.group(1)
            mib = float(mdev.group(2))
            out["kv_buffer_mib_by_device"][device] = out["kv_buffer_mib_by_device"].get(device, 0.0) + mib
            kv_total += mib
            saw_kv_buf = True
        else:
            mbuf = _RE_KV_BUF.search(line)
            if mbuf:
                kv_total += float(mbuf.group(1))
                saw_kv_buf = True

        moff = _RE_OFFLOAD.search(line)
        if moff:
            out["layers_offloaded"] = int(moff.group(1))
            out["layers_total"] = int(moff.group(2))

        mctx = _RE_NCTX.search(line)
        if mctx and out["n_ctx"] is None:
            out["n_ctx"] = int(mctx.group(1))
            out["n_ctx_source"] = "n_ctx"
        mslot = _RE_NCTX_SLOT.search(line)
        if mslot and out["n_ctx_slot"] is None:
            out["n_ctx_slot"] = int(mslot.group(1))
        mslots = _RE_NSLOTS.search(line)
        if mslots and out["n_slots"] is None:
            out["n_slots"] = int(mslots.group(1))

        munified = _RE_KV_UNIFIED.search(line)
        if munified and out["kv_unified"] is None:
            out["kv_unified"] = munified.group(1).lower()

        stamp = parse_log_time(line)
        if stamp is not None:
            out["last_log_time_s"] = stamp
        if _RE_LISTENING.search(line):
            out["server_listening"] = True
            if out["listening_at_log_s"] is None and stamp is not None:
                out["listening_at_log_s"] = stamp
        if _RE_SLOTS_IDLE.search(line):
            out["slots_idle_seen"] = True
        if _RE_MODEL_LOADED.search(line):
            out["model_loaded_seen"] = True
        if _RE_MODEL_LOADED_EVENT.search(line):
            if out["model_loaded_at_log_s"] is None and stamp is not None:
                out["model_loaded_at_log_s"] = stamp

        for name, pattern in _FALLBACK_PATTERNS:
            if pattern.search(line):
                out["fallback_evidence"].append({"signal": name, "line": line.strip()})

    if saw_kv_buf:
        out["kv_buffer_mib_total"] = round(kv_total, 2)

    # n_ctx_slot is per-slot: with -np N the TOTAL context is n_ctx_slot * n_slots. Only
    # fall back to it when no bare `n_ctx =` line exists, and say so in n_ctx_source.
    if out["n_ctx_slot"] is not None:
        if out["n_slots"]:
            out["n_ctx_from_slots"] = out["n_ctx_slot"] * out["n_slots"]
        if out["n_ctx"] is None:
            out["n_ctx"] = out["n_ctx_from_slots"] or out["n_ctx_slot"]
            out["n_ctx_source"] = (
                "n_ctx_slot*n_slots" if out["n_ctx_from_slots"] else "n_ctx_slot"
            )

    quantized = {"q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
    if (out["kv_type_k"] in quantized) or (out["kv_type_v"] in quantized):
        out["kv_quantized"] = True
        # llama.cpp refuses quantized KV without flash_attn; a loaded server therefore ran
        # the FA path. This is INFERENCE FROM A REFUSAL, recorded as such.
        refused = any(item["signal"] == "kv_quant_refused" for item in out["fallback_evidence"])
        out["kv_quant_implies_flash_attn"] = bool(
            out["model_loaded_seen"] or out["server_listening"]
        ) and not refused

    return out


# --------------------------------------------------------------------------------------
# nvidia-smi sampling
# --------------------------------------------------------------------------------------


def parse_nvidia_smi_csv(text):
    """Parse `--query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
            used = float(parts[1])
        except ValueError:
            continue
        total = None
        if len(parts) >= 3:
            try:
                total = float(parts[2])
            except ValueError:
                total = None
        rows[index] = {"memory_used_mib": used, "memory_total_mib": total}
    return rows


_FLAG_ALIASES = {
    "flash_attn": ("-fa", "--flash-attn"),
    "type_k": ("-ctk", "--cache-type-k"),
    "type_v": ("-ctv", "--cache-type-v"),
    "n_ctx": ("-c", "--ctx-size"),
    "n_parallel": ("-np", "--parallel"),
    "n_gpu_layers": ("-ngl", "--n-gpu-layers", "--gpu-layers"),
    "model": ("-m", "--model"),
    "port": ("--port",),
}
_VALUELESS = {"on", "off", "auto"}


def parse_server_flags(command):
    """Extract the llama-server envelope flags from a launch command.

    `command` may be a list (docker inspect .Config.Cmd) or a string. This is where the
    REQUESTED configuration comes from; the log says what the server did with it.
    """
    if command is None:
        return {}
    tokens = list(command) if isinstance(command, (list, tuple)) else shlex.split(str(command))
    tokens = [str(token) for token in tokens]
    found = {}
    for key, aliases in _FLAG_ALIASES.items():
        for index, token in enumerate(tokens):
            if token not in aliases:
                continue
            value = tokens[index + 1] if index + 1 < len(tokens) else None
            if key == "flash_attn":
                # `-fa` alone (older syntax) means enable; `-fa on|off|auto` is current
                if value is None or value.startswith("-") or value.lower() not in _VALUELESS:
                    found[key] = "on_implicit"
                else:
                    found[key] = value.lower()
            elif value is not None and not value.startswith("-"):
                found[key] = value
            break
    for key in ("n_ctx", "n_parallel", "n_gpu_layers", "port"):
        if key in found:
            try:
                found[key] = int(found[key])
            except (TypeError, ValueError):
                pass
    return found


QUANTIZED_KV = ("q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl")

# Where the "requested" flags came from. Only the first two were OBSERVED off a running
# server; the third is what an operator typed on our command line and says nothing about
# the server that is actually answering.
FLAGS_SOURCES = ("own_launch_command", "docker_inspect", "cli_defaults_unverified", "unspecified")

R3_CONTROL_ENV = "OC_R3_CONTROL"
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
R3_CONTROL_CANDIDATES = (
    os.path.join(_MODULE_DIR, "fa-control", "fa-negative-control.json"),
    os.path.join(_MODULE_DIR, "receipts", "fa-control", "fa-negative-control.json"),
)


def build_identity(llamacpp_root=None, build_receipt_path=None, source=None):
    """Identify the llama.cpp build a measurement is about: commit + llama-server sha256.

    Returns None when neither a build receipt nor a binary can be found, which callers must
    report as "the R3 control could not be joined to this build" rather than assuming it.
    """
    if not build_receipt_path and llamacpp_root:
        build_receipt_path = os.path.join(llamacpp_root, "build-receipt.json")
    identity = {
        "source": source or ("build_receipt" if build_receipt_path else "unknown"),
        "build_receipt_path": build_receipt_path,
        "build_receipt_sha256": None,
        "llama_cpp_commit": None,
        "llama_server_sha256": None,
    }
    found = False
    if build_receipt_path and os.path.exists(build_receipt_path):
        try:
            with open(build_receipt_path, "r", encoding="utf-8") as handle:
                built = json.load(handle)
        except (ValueError, OSError):
            built = {}
        identity["build_receipt_sha256"] = sha256_file(build_receipt_path)
        identity["llama_cpp_commit"] = built.get("llama_cpp_commit")
        binaries = built.get("binaries") or {}
        server = binaries.get("llama-server") or {}
        identity["llama_server_sha256"] = server.get("sha256")
        found = True
    # The binary on disk beats the receipt's memory of it: a rebuild that failed to update
    # the receipt would otherwise be invisible.
    if llamacpp_root:
        binary = os.path.join(llamacpp_root, "bin", "llama-server")
        if os.path.exists(binary):
            identity["llama_server_sha256_on_disk"] = sha256_file(binary)
            if not identity["llama_server_sha256"]:
                identity["llama_server_sha256"] = identity["llama_server_sha256_on_disk"]
            found = True
    return identity if found else None


# The join below is only a licence when THIS field matches on both sides. A commit is not
# an identity: the same commit built twice (different --arch, different CUDA flags, a
# separate imatrix root) produces different binaries, and a control measured on one of
# them says nothing about the other.
REQUIRED_JOIN_FIELD = "llama_server_sha256"


def _join_control_to_build(control_payload, identity):
    """Does this control receipt describe the build we are about to measure?

    `matches` is True only when the llama-server sha256 is present on BOTH sides and
    equal, and nothing else that is comparable disagrees. None means the join could not be
    completed -- which is not a licence either, and specifically covers the case that used
    to slip through: a control that never recorded which binary it measured, joined on
    commit alone, licensing every binary ever built from that commit.
    """
    join = {
        "checked": bool(identity),
        "matches": None,
        "required_field": REQUIRED_JOIN_FIELD,
        "required_field_matched": False,
        "fields_compared": [],
        "fields_missing_on_control": [],
        "fields_missing_on_build": [],
        "reasons": [],
        "control_llama_cpp_commit": control_payload.get("llama_cpp_commit"),
        "control_llama_server_sha256": control_payload.get("llama_server_sha256"),
        "control_build_receipt_sha256": control_payload.get("build_receipt_sha256"),
        "build_llama_cpp_commit": (identity or {}).get("llama_cpp_commit"),
        "build_llama_server_sha256": (identity or {}).get("llama_server_sha256"),
        "build_receipt_sha256": (identity or {}).get("build_receipt_sha256"),
    }
    if not identity:
        join["reasons"].append(
            "no build identity was supplied, so the control could not be joined to the "
            "binary being measured"
        )
        return join
    mismatch = False
    for label, left, right in (
        ("llama_cpp_commit", join["control_llama_cpp_commit"], join["build_llama_cpp_commit"]),
        ("llama_server_sha256", join["control_llama_server_sha256"],
         join["build_llama_server_sha256"]),
    ):
        if not left:
            join["fields_missing_on_control"].append(label)
            # A field the build carries and the control does not is NOT a neutral gap: it
            # is the control declining to say which binary it measured. Skipping it used
            # to leave the commit as the only comparison, so a control with no
            # llama_server_sha256 (exactly the shape shipped with this lane) matched any
            # rebuild at the same commit.
            join["reasons"].append(
                "%s missing on the control side%s; the control does not record which "
                "binary it measured" % (
                    label, " while the build reports %s" % str(right)[:16] if right else ""))
            continue
        if not right:
            join["fields_missing_on_build"].append(label)
            join["reasons"].append(
                "%s missing on the build side; the join cannot be completed on it" % label)
            continue
        join["fields_compared"].append(label)
        if left != right:
            mismatch = True
            join["reasons"].append("%s differs: control %s != build %s" % (
                label, str(left)[:16], str(right)[:16]))
    join["required_field_matched"] = (
        REQUIRED_JOIN_FIELD in join["fields_compared"]
        and join["control_%s" % REQUIRED_JOIN_FIELD] == join["build_%s" % REQUIRED_JOIN_FIELD]
    )
    if mismatch:
        join["matches"] = False
    elif join["required_field_matched"]:
        join["matches"] = True
        join["reasons"].append("%d field(s) matched, including %s" % (
            len(join["fields_compared"]), REQUIRED_JOIN_FIELD))
    elif join["fields_compared"]:
        # Something matched, but not the binary hash. flash_attn_evidence() reads
        # matches=None as "could NOT be joined", which is the honest reading: a commit
        # match is a coincidence of source, not of the binary that ran.
        join["reasons"].append(
            "%s was never compared, so the control is not joined to this binary "
            "(matched only: %s)" % (REQUIRED_JOIN_FIELD, ", ".join(join["fields_compared"])))
    else:
        join["reasons"].append("nothing comparable on both sides")
    return join


def load_r3_control(path=None, identity=None, llamacpp_root=None, build_receipt_path=None):
    """Load the negative-control receipt that licenses (or voids) the R3 inference.

    R3 says: "llama.cpp refuses quantized KV without flash_attn, so a server that loaded
    with q4_0 KV ran the FA path". That is a claim about a SPECIFIC BUILD's behaviour, and
    this build already surprised us once (it prints no FA banner at all). The claim is
    therefore only usable when `fa_negative_control.py` has actually started this build
    with `-fa off -ctk q4_0 -ctv q4_0` and recorded what happened.

    Returns None when no control receipt exists, which callers must report as
    "R3 premise UNVERIFIED on this build" rather than silently assuming it.

    A control is a measurement of ONE BINARY. The receipt is therefore joined to the build
    under measurement (`identity`, or one derived from `llamacpp_root` /
    `build_receipt_path`) on the llama-server sha256 -- which is REQUIRED, a commit match
    alone is not a join -- and the result is returned as `build_match`. `flash_attn_evidence()` refuses to promote a reading when that join is
    anything other than a positive match: without it, the next
    `build_llamacpp_cuda.sh` run at a new commit would leave a stale control on disk that
    keeps licensing R3 for a binary it never touched.
    """
    if identity is None and (llamacpp_root or build_receipt_path):
        identity = build_identity(llamacpp_root=llamacpp_root,
                                  build_receipt_path=build_receipt_path)
    if path:
        # An explicitly named control that does not exist is NOT quietly replaced by a
        # default one: the caller asked for a specific observation and did not get it.
        if not os.path.exists(path):
            return None
        candidates = [path]
    else:
        candidates = []
        env_path = os.environ.get(R3_CONTROL_ENV)
        if env_path:
            candidates.append(env_path)
        candidates.extend(R3_CONTROL_CANDIDATES)
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            continue
        return {
            "path": candidate,
            "receipt_sha256": sha256_file(candidate),
            "premise_holds": payload.get("premise_holds"),
            "llama_cpp_commit": payload.get("llama_cpp_commit"),
            "llama_server_sha256": payload.get("llama_server_sha256"),
            "build_receipt_sha256": payload.get("build_receipt_sha256"),
            "observed": payload.get("observed"),
            "refusal_line": payload.get("refusal_line"),
            "negative_refusal_signals": payload.get("negative_refusal_signals"),
            "build_match": _join_control_to_build(payload, identity),
        }
    return None


def flash_attn_evidence(log_analysis, requested_flags, flags_source="unspecified",
                        r3_control=None):
    """Compose the strongest available reading of whether the FA path ran.

    llama.cpp emits NO line meaning "the flash-attention kernel executed", and recent
    builds no longer print the `flash_attn = ...` context banner to the server log at all.
    What is available:
      R1 the requested `-fa` flag, ONLY when it was observed off the running server
         (our own launch command, or `docker inspect` of the attached container),
      R2 an explicit `flash_attn = enabled/disabled` line, when the build prints one,
      R3 quantized-KV acceptance: llama.cpp refuses `-ctk/-ctv` != f16 without flash
         attention -- usable only when a negative control on THIS BUILD confirms the
         refusal (see `load_r3_control`),
      R4 any fallback / "not supported" / CUDA-error line, verbatim.

    `flags_source` must be one of FLAGS_SOURCES. When it is `cli_defaults_unverified` the
    requested flags are DISCARDED: they are what the operator typed, not what the server
    is running, and a receipt that presented them as observed would be a fabrication.

    Returns a labelled reading plus its basis. Never a verdict about the run's quality.
    """
    requested_flags = requested_flags or {}
    if flags_source not in FLAGS_SOURCES:
        flags_source = "unspecified"
    # Only two sources are OBSERVATIONS of the server that is answering. "unspecified" used
    # to fall on the observed side of this line, which is how two probe receipts shipped
    # with `requested_flags_source: null` and an FA reading nobody had licensed: a caller
    # that forgets the argument must not get its flags treated as measurements.
    flags_observed = flags_source in ("own_launch_command", "docker_inspect")

    if flags_observed:
        requested_fa = requested_flags.get("flash_attn")
        type_k = (requested_flags.get("type_k") or log_analysis.get("kv_type_k") or "").lower()
        type_v = (requested_flags.get("type_v") or log_analysis.get("kv_type_v") or "").lower()
    else:
        requested_fa = None
        type_k = (log_analysis.get("kv_type_k") or "").lower()
        type_v = (log_analysis.get("kv_type_v") or "").lower()

    ready = bool(log_analysis.get("model_loaded_seen") or log_analysis.get("server_listening"))
    refused = any(
        item["signal"] in ("kv_quant_refused", "flash_attn_unsupported", "flash_attn_off")
        for item in log_analysis.get("fallback_evidence", [])
    )
    quantized_kv = type_k in QUANTIZED_KV or type_v in QUANTIZED_KV

    # A control licenses R3 only for the binary it was measured on. `build_match` is the
    # join between the control and the build under measurement; anything short of a
    # positive match drops the premise back to unverified, however confident the control
    # itself sounds.
    build_match = (r3_control or {}).get("build_match") or {}
    build_join = build_match.get("matches")
    build_join_note = None
    if r3_control is None:
        r3_premise = "unverified_on_this_build"
    elif build_join is False:
        r3_premise = "unverified_on_this_build"
        build_join_note = (
            "R3 control was measured on a DIFFERENT build (%s); its licence does not "
            "transfer" % "; ".join(build_match.get("reasons") or ["mismatch"])
        )
    elif build_join is None:
        r3_premise = "unverified_on_this_build"
        build_join_note = (
            "R3 control could NOT be joined to the build being measured (%s); its licence "
            "is not applied here" % "; ".join(build_match.get("reasons") or ["not checked"])
        )
    elif r3_control.get("premise_holds") is True:
        r3_premise = "verified"
    elif r3_control.get("premise_holds") is False:
        r3_premise = "void"
    else:
        r3_premise = "unverified_on_this_build"

    basis = []
    if build_join_note:
        basis.append(build_join_note)
    if not flags_observed:
        basis.append(
            "R0 launch flags NOT OBSERVED (source=%s); requested flags discarded" % flags_source
        )
    if flags_source == "unspecified" and requested_flags:
        basis.append(
            "R0 flags_source was not stated by the caller; flags treated as UNOBSERVED"
        )
    if requested_fa:
        basis.append("R1 launch flag -fa=%s (source=%s)" % (requested_fa, flags_source))
    if log_analysis.get("flash_attn_state") not in (None, "unknown"):
        basis.append("R2 log line flash_attn=%s" % log_analysis["flash_attn_state"])
    if quantized_kv:
        basis.append(
            "R3 KV types %s/%s are quantized and the server reached ready=%s "
            "[premise %s]" % (type_k or "?", type_v or "?", ready, r3_premise)
        )
    if refused:
        basis.append("R4 a refusal/unsupported/fallback line is present")

    r3_usable = quantized_kv and ready and not refused and r3_premise == "verified"
    if log_analysis.get("flash_attn_state") == "disabled" or refused:
        reading = "flash attention did NOT run (explicit log evidence)"
    elif r3_usable:
        reading = (
            "flash-attention path ran: quantized KV was accepted at load, which this build "
            "refuses without flash_attn (negative control %s; inference from a refusal that "
            "did not occur)" % (r3_control or {}).get("receipt_sha256", "?")[:16]
        )
    elif quantized_kv and ready and r3_premise == "void":
        reading = (
            "R3 IS VOID on this build: the negative control loaded with quantized KV and "
            "-fa off, so KV acceptance is not evidence of the flash-attention path. "
            "Only the launch flag (R1) speaks here."
        )
    elif quantized_kv and ready:
        reading = (
            "quantized KV was accepted at load, but the R3 premise (llama.cpp refuses "
            "quantized KV without flash_attn) is UNVERIFIED on this build -- run "
            "fa_negative_control.py before reading this as flash attention"
        )
        if build_join_note:
            reading += ". " + build_join_note
    elif log_analysis.get("flash_attn_state") == "enabled":
        reading = "flash attention requested and accepted per the log banner"
    elif flags_observed and requested_fa in ("off",):
        # R1 read in the negative direction. An observed `-fa off` is exactly as much of an
        # observation as an observed `-fa on`, and the earlier version of this function
        # only ever used the flag affirmatively: an attached `-fa off -ctk f16` server
        # reported "no direct evidence either way" while its own basis line said -fa=off.
        reading = (
            "flash attention was NOT requested: the launch command observed on the running "
            "server carries -fa off (source=%s), and nothing in the log contradicts it" % flags_source
        )
    else:
        reading = "no direct evidence either way in this build's log"

    return {
        "flags_source": flags_source,
        "flags_observed": flags_observed,
        "requested_flash_attn": requested_fa,
        "requested_type_k": type_k or None,
        "requested_type_v": type_v or None,
        "kv_pair_matched": bool(type_k and type_v and type_k == type_v),
        "server_reached_ready": ready,
        "quantized_kv_accepted": bool(quantized_kv and ready and not refused),
        "refusal_or_fallback_seen": refused,
        "r3_premise": r3_premise,
        "r3_usable": bool(r3_usable),
        "r3_control_build_match": build_match or None,
        "r3_control_build_join": build_join,
        "r3_control": r3_control,
        "reading": reading,
        "basis": basis,
        "note": "llama.cpp prints no 'FA kernel executed' line; this is a reading, not a proof.",
    }


def container_launch_command(name, timeout=60):
    """entrypoint + cmd of a container, as one token list. Read-only."""
    rc, out, err = run(["docker", "inspect", name], timeout=timeout)
    if rc != 0:
        return []
    try:
        config = json.loads(out)[0].get("Config", {})
    except (ValueError, IndexError, KeyError):
        return []
    return list(config.get("Entrypoint") or []) + list(config.get("Cmd") or [])


def docker_logs(name, head=None, tail=None, timeout=180):
    """Read a container's log. `head` gets the LOAD BANNER (flash_attn / KV / offload
    lines are printed once at startup and scroll away behind per-request spam), `tail`
    gets the current state. Read-only: `docker logs` never touches the container."""
    quoted = shlex.quote(name)
    if head:
        cmd = ["bash", "-c", "docker logs %s 2>&1 | head -n %d" % (quoted, int(head))]
    else:
        cmd = ["bash", "-c", "docker logs --tail %d %s 2>&1" % (int(tail or 200), quoted)]
    rc, out, err = run(cmd, timeout=timeout)
    if rc != 0:
        return "[docker logs rc=%d] %s" % (rc, err.strip()[:300])
    return out


def props_n_ctx(props):
    """llama-server reports n_ctx at the top level in some builds and nested inside
    default_generation_settings in others."""
    if not isinstance(props, dict):
        return None
    if isinstance(props.get("n_ctx"), int):
        return props["n_ctx"]
    nested = props.get("default_generation_settings")
    if isinstance(nested, dict):
        if isinstance(nested.get("n_ctx"), int):
            return nested["n_ctx"]
        params = nested.get("params")
        if isinstance(params, dict) and isinstance(params.get("n_ctx"), int):
            return params["n_ctx"]
    return None


def parse_compute_apps_csv(text):
    """Parse `--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory`.

    This is the per-PROCESS view. Whole-card `memory.used` includes every other tenant of
    the card, so a run's OWN share must be measured here, never derived by subtracting a
    remembered number from the whole-card peak.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "[N/A]" == line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
            used = float(parts[3])
        except ValueError:
            continue
        rows.append({
            "gpu_uuid": parts[0],
            "pid": pid,
            "process_name": parts[2],
            "memory_used_mib": used,
        })
    return rows


class GpuSampler(object):
    """Background nvidia-smi poller recording peak memory per GPU index AND per PID.

    Whole-card peak answers "did this fit on the card"; per-PID peak answers "what did THIS
    run cost", which is the number a KV-footprint conclusion actually needs. Both are
    recorded; neither is derived from the other.
    """

    def __init__(self, interval=2.0, indices=None, track_pids=None):
        self.interval = interval
        self.indices = set(indices) if indices is not None else None
        self.peak = {}
        self.peak_by_pid = {}
        self.uuid_to_index = {}
        self.samples = 0
        self.compute_app_samples = 0
        self.compute_apps_supported = None
        self.errors = []
        self.compute_app_errors = []
        self.track_pids = set(int(pid) for pid in track_pids) if track_pids else set()
        self._stop = threading.Event()
        self._thread = None

    def watch_pid(self, pid):
        """Mark a PID as ours (recorded in the receipt so 'own share' is unambiguous)."""
        if pid is not None:
            self.track_pids.add(int(pid))

    def _refresh_uuid_map(self):
        rc, out, _ = run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=20
        )
        if rc != 0:
            return
        for line in out.splitlines():
            parts = [piece.strip() for piece in line.split(",")]
            if len(parts) >= 2:
                try:
                    self.uuid_to_index[parts[1]] = int(parts[0])
                except ValueError:
                    continue

    def _poll_compute_apps(self):
        rc, out, err = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if rc != 0:
            self.compute_apps_supported = False
            self.compute_app_errors.append((err or "").strip()[:200])
            return
        self.compute_apps_supported = True
        self.compute_app_samples += 1
        for row in parse_compute_apps_csv(out):
            index = self.uuid_to_index.get(row["gpu_uuid"])
            # When an index filter is set, a row whose GPU we could not map is DROPPED
            # rather than kept unattributed: an unattributable process must not land in a
            # per-PID peak that a receipt will read as "on our card".
            if self.indices is not None and index not in self.indices:
                continue
            key = str(row["pid"])
            slot = self.peak_by_pid.setdefault(key, {
                "peak_memory_used_mib": 0.0,
                "process_name": row["process_name"],
                "gpu_uuid": row["gpu_uuid"],
                "gpu_index": index,
                "tracked_as_ours": row["pid"] in self.track_pids,
            })
            slot["tracked_as_ours"] = slot["tracked_as_ours"] or row["pid"] in self.track_pids
            if row["memory_used_mib"] > slot["peak_memory_used_mib"]:
                slot["peak_memory_used_mib"] = row["memory_used_mib"]

    def _poll_once(self):
        rc, out, err = run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if rc != 0:
            self.errors.append((err or "").strip()[:200])
        else:
            rows = parse_nvidia_smi_csv(out)
            self.samples += 1
            for index, row in rows.items():
                if self.indices is not None and index not in self.indices:
                    continue
                slot = self.peak.setdefault(
                    index,
                    {"peak_memory_used_mib": 0.0, "memory_total_mib": row["memory_total_mib"]},
                )
                if row["memory_used_mib"] > slot["peak_memory_used_mib"]:
                    slot["peak_memory_used_mib"] = row["memory_used_mib"]
        self._poll_compute_apps()

    def _loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self):
        self._refresh_uuid_map()
        self._poll_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3)
        self._poll_once()
        ours = {
            pid: slot for pid, slot in self.peak_by_pid.items() if slot.get("tracked_as_ours")
        }
        own_peak = round(sum(slot["peak_memory_used_mib"] for slot in ours.values()), 2) if ours else None
        # Per-process used memory can never exceed whole-card used memory at the same
        # instant, so own_peak > every whole-card peak means the two series did not see the
        # same moment -- one of them missed the ramp. A receipt shipped with exactly that
        # pair (own 23,942 MiB against a whole-card peak of 15,245 MiB on the same card in
        # the same 32-sample run) and nothing said so. The two series are sampled by
        # different nvidia-smi queries, so the inconsistency is reported, never repaired.
        card_peaks = [info.get("peak_memory_used_mib") for info in self.peak.values()
                      if info.get("peak_memory_used_mib") is not None]
        max_card_peak = max(card_peaks) if card_peaks else None
        inconsistent = (
            own_peak is not None and max_card_peak is not None and own_peak > max_card_peak)
        return {
            "samples": self.samples,
            "interval_s": self.interval,
            "peak_by_index": {str(k): v for k, v in sorted(self.peak.items())},
            "peak_by_pid": {k: self.peak_by_pid[k] for k in sorted(self.peak_by_pid)},
            "tracked_pids": sorted(str(pid) for pid in self.track_pids),
            "own_peak_memory_used_mib": own_peak,
            "whole_card_peak_max_mib": max_card_peak,
            "own_peak_exceeds_whole_card_peak": bool(inconsistent),
            "peak_series_note": (
                "own_peak_memory_used_mib (%s MiB) EXCEEDS the largest whole-card peak "
                "(%s MiB); per-process used cannot exceed whole-card used at one instant, "
                "so one series missed the ramp. Both are reported as sampled."
                % (own_peak, max_card_peak)
            ) if inconsistent else None,
            "compute_apps_supported": self.compute_apps_supported,
            "compute_app_samples": self.compute_app_samples,
            "errors": self.errors[-5:],
            "compute_app_errors": self.compute_app_errors[-5:],
            "note": (
                "peak_by_index is WHOLE-CARD and includes every other tenant; "
                "own_peak_memory_used_mib is the measured sum over tracked PIDs, not a "
                "subtraction."
            ),
        }


# --------------------------------------------------------------------------------------
# llama-server HTTP client
# --------------------------------------------------------------------------------------


class LlamaServer(object):
    """Minimal llama-server client over stdlib urllib. Greedy by default."""

    def __init__(self, base_url, timeout=600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, route, payload, timeout=None):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, route, timeout=None):
        request = urllib.request.Request(self.base_url + route, method="GET")
        with urllib.request.urlopen(request, timeout=timeout or 30.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self, timeout=10.0):
        try:
            return self._get("/health", timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - health is a probe, report the failure
            return {"status": "unreachable", "error": repr(exc)}

    def props(self, timeout=20.0):
        try:
            return self._get("/props", timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def wait_healthy(self, deadline_s, poll_s=5.0):
        started = time.time()
        while time.time() - started < deadline_s:
            state = self.health()
            if state.get("status") == "ok":
                return {"healthy": True, "wait_s": round(time.time() - started, 2)}
            time.sleep(poll_s)
        return {"healthy": False, "wait_s": round(time.time() - started, 2)}

    def tokenize(self, content, timeout=300.0):
        payload = self._post("/tokenize", {"content": content}, timeout=timeout)
        return payload.get("tokens", [])

    def token_count(self, content, timeout=300.0):
        return len(self.tokenize(content, timeout=timeout))

    @staticmethod
    def greedy_payload(prompt, n_predict, cache_prompt=False, seed=1234, extra=None):
        """Deterministic decode settings. Greedy = temperature 0 + top_k 1."""
        payload = {
            "prompt": prompt,
            "n_predict": int(n_predict),
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": int(seed),
            "cache_prompt": bool(cache_prompt),
            "stream": False,
        }
        if extra:
            payload.update(extra)
        return payload

    def completion(self, prompt, n_predict=64, cache_prompt=False, seed=1234, timeout=None, extra=None):
        payload = self.greedy_payload(prompt, n_predict, cache_prompt, seed, extra)
        started = time.time()
        body = self._post("/completion", payload, timeout=timeout)
        wall = time.time() - started
        return summarize_completion(body, wall)


def summarize_completion(body, wall_s):
    """Flatten a llama-server /completion response into measured fields."""
    timings = body.get("timings") or {}
    return {
        "content": body.get("content", ""),
        "stop_type": body.get("stop_type"),
        "stopped_eos": body.get("stopped_eos"),
        "stopped_limit": body.get("stopped_limit"),
        "tokens_predicted": body.get("tokens_predicted"),
        "tokens_evaluated": body.get("tokens_evaluated"),
        "truncated": body.get("truncated"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "wall_s": round(wall_s, 3),
    }


# --------------------------------------------------------------------------------------
# needle-in-a-haystack construction
# --------------------------------------------------------------------------------------

# n_predict used for a needle question, and the extra tokens held back for whatever the
# server prepends (BOS / template). Both feed context_point_headroom().
NEEDLE_PREDICT_TOKENS = 32
CONTEXT_SAFETY_TOKENS = 64

NEEDLE_TEMPLATE = "The secret access code for vault {label} is {code}."
NEEDLE_QUESTION = (
    "\n\nQuestion: what is the secret access code for vault {label}? "
    "Answer with the code only.\nAnswer:"
)
FILLER_TEMPLATE = (
    "Line {index:06d}: the maintenance log records routine inspection of the storage "
    "corridor, temperature nominal, no anomalies observed.\n"
)


def build_haystack(line_count, start=1):
    return "".join(FILLER_TEMPLATE.format(index=i) for i in range(start, start + line_count))


def build_needle_prompt(line_count, depth_fraction, code, label="K7"):
    """Insert the needle at `depth_fraction` through `line_count` filler lines.

    The needle is an EXTRA line placed after `insert_at` filler lines, so the body is
    always `line_count + 1` lines and the extremes are exact: depth 0.0 puts the needle
    before the first filler line, depth 1.0 puts it after the last one. (The earlier
    construction inserted before filler index round(d*(n-1)), which made depth 1.0
    second-to-last and never last.)

    Returns (prompt, insert_line_index) with insert_line_index in [0, line_count].
    """
    if line_count < 1:
        raise ValueError("line_count must be >= 1")
    if not 0.0 <= depth_fraction <= 1.0:
        raise ValueError("depth_fraction must be in [0, 1]")
    insert_at = int(round(depth_fraction * line_count))
    parts = []
    for i in range(line_count):
        if i == insert_at:
            parts.append(NEEDLE_TEMPLATE.format(label=label, code=code) + "\n")
        parts.append(FILLER_TEMPLATE.format(index=i + 1))
    if insert_at >= line_count:
        parts.append(NEEDLE_TEMPLATE.format(label=label, code=code) + "\n")
    prompt = (
        "You are reading a maintenance log. Read it carefully; one line contains a secret "
        "access code.\n\n" + "".join(parts) + NEEDLE_QUESTION.format(label=label)
    )
    return prompt, insert_at


def score_needle(response_text, code):
    """Exact containment of the code, case-insensitive, whitespace tolerant.

    Reported as a measurement. A miss is recorded verbatim, not interpreted.
    """
    if not response_text:
        return {"hit": False, "normalized": ""}
    normalized = re.sub(r"\s+", " ", response_text).strip()
    return {
        "hit": code.lower() in normalized.lower(),
        "normalized": normalized[:400],
    }


def calibrate_line_count(server, target_tokens, code, label="K7", depth_fraction=0.5,
                         tolerance=0.02, max_rounds=6, tokens_per_line_guess=22.0,
                         max_tokens=None):
    """Find a filler line count whose tokenized prompt lands near `target_tokens`.

    Uses the server's own tokenizer through /tokenize (so it is exact for the loaded
    model), converging by proportional correction.

    `max_tokens` is a HARD ceiling the prompt may never cross. It exists because the
    sibling lane's 250k needle requests came back `HTTPError 400: Bad Request`: a prompt
    calibrated to `target` +/- 2% can land ABOVE the served context, and llama-server
    refuses the request outright. At the 262,144-token point of a `-c 262144` server that
    refusal is guaranteed -- the prompt must leave room for the tokens we ask it to
    generate. So overshoot is never inside tolerance when a ceiling is set: the loop
    converges from below, and a final shrink pass guarantees the returned prompt is under
    the ceiling (or says `over_cap` if it could not get there).

    Returns a dict of the measurement, including `requested_target_tokens` (what the caller
    asked for), `target_tokens` (what was aimed at after the ceiling was applied),
    `max_tokens`, `capped` and `over_cap`.
    """
    requested_target = int(target_tokens)
    ceiling = int(max_tokens) if max_tokens else None
    if ceiling is not None and ceiling < 8:
        raise ValueError("max_tokens must leave room for a prompt, got %r" % (max_tokens,))
    capped = bool(ceiling is not None and requested_target > ceiling)
    target_tokens = min(requested_target, ceiling) if ceiling is not None else requested_target

    lines = max(4, int(target_tokens / tokens_per_line_guess))
    history = []
    prompt = None
    insert_at = None
    measured = None
    for _round in range(max_rounds):
        prompt, insert_at = build_needle_prompt(lines, depth_fraction, code, label)
        measured = server.token_count(prompt)
        history.append({"lines": lines, "tokens": measured})
        over = ceiling is not None and measured > ceiling
        if not over and abs(measured - target_tokens) <= tolerance * target_tokens:
            break
        # 0.995 undershoots deliberately: landing exactly on the ceiling leaves no room for
        # the tokenizer's next-round jitter, and being 0.5% short of the cap costs nothing.
        goal = float(min(target_tokens, ceiling * 0.995)) if over else float(target_tokens)
        ratio = goal / float(max(measured, 1))
        new_lines = max(4, int(lines * ratio))
        if new_lines == lines:
            new_lines = lines + (1 if measured < target_tokens else -1)
        lines = max(4, new_lines)

    # Final guarantee: never hand back a prompt above the ceiling.
    shrink_rounds = 0
    while ceiling is not None and measured is not None and measured > ceiling and shrink_rounds < 8:
        shrink_rounds += 1
        ratio = (float(ceiling) * 0.995) / float(max(measured, 1))
        new_lines = max(4, min(lines - 1, int(lines * ratio)))
        if new_lines == lines:
            break
        lines = new_lines
        prompt, insert_at = build_needle_prompt(lines, depth_fraction, code, label)
        measured = server.token_count(prompt)
        history.append({"lines": lines, "tokens": measured, "shrink_round": shrink_rounds})

    return {
        "lines": history[-1]["lines"],
        "tokens": measured,
        "target_tokens": target_tokens,
        "requested_target_tokens": requested_target,
        "max_tokens": ceiling,
        "capped": capped,
        "over_cap": bool(ceiling is not None and measured is not None and measured > ceiling),
        "rounds": history,
        "prompt": prompt,
        "insert_line": insert_at,
    }


def context_point_headroom(n_ctx, decode_tokens, needle_tokens=NEEDLE_PREDICT_TOKENS,
                           safety_tokens=CONTEXT_SAFETY_TOKENS):
    """Tokens that must stay free inside `n_ctx` for the generations we ask for.

    The prefill request asks for 1 token, the decode request for `decode_tokens` (with
    ignore_eos, so it really generates all of them), and each needle request for
    `needle_tokens`. llama-server needs prompt + n_predict to fit in the slot's context,
    so the prompt ceiling is n_ctx minus the largest of those, minus a safety margin for
    BOS/template tokens the server may add on its own.
    """
    reserve = max(int(decode_tokens), int(needle_tokens), 1) + int(safety_tokens)
    if not n_ctx:
        return {"n_ctx": None, "reserve_tokens": reserve, "prompt_ceiling": None}
    ceiling = int(n_ctx) - reserve
    if ceiling < 8:
        raise ValueError("n_ctx %r is too small for decode_tokens %r" % (n_ctx, decode_tokens))
    return {"n_ctx": int(n_ctx), "reserve_tokens": reserve, "prompt_ceiling": ceiling}


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------


def parse_int_list(text):
    """'4096,32768' -> [4096, 32768]; sorted, deduped, positive only."""
    values = []
    for piece in str(text).replace(" ", "").split(","):
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError("context points must be positive, got %r" % value)
        values.append(value)
    return sorted(set(values))


def parse_float_list(text):
    values = []
    for piece in str(text).replace(" ", "").split(","):
        if not piece:
            continue
        value = float(piece)
        if not 0.0 <= value <= 1.0:
            raise ValueError("depth fractions must be in [0, 1], got %r" % value)
        values.append(value)
    return sorted(set(values))


def host_facts():
    rc, smi, _ = run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader"],
        timeout=30,
    )
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "python": sys.version.split()[0],
        "nvidia_smi_rc": rc,
        "gpus": [line.strip() for line in smi.splitlines() if line.strip()] if rc == 0 else [],
        "captured_at": utc_now(),
    }
