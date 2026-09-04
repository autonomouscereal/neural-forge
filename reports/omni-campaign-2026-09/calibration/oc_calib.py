#!/usr/bin/env python3
"""Omni-campaign parent calibration: the shared instrument.

Everything here is ported from the sealed LFM2 matrix tooling and says where
from, because the whole point of a calibration lane is that the omni parents are
read on the SAME surface the LFM2 children were read on.  A number measured with
a different instrument is not comparable with the campaign's record, and the
campaign's record is the only prior we have.

WHAT IS COPIED AND WHY IT IS COPIED RATHER THAN IMPORTED.  This file runs on the
AI server under `/home/cereal/omni-campaign/calibration/`, where the neural-forge
repository is not checked out.  Importing `neural_forge.dynamic_circuit_ir` there
would fail, and a calibration runner that cannot run is worth nothing.  So the
four load-bearing primitives are re-implemented from the sealed originals.

EXACTLY HOW MUCH OF THAT IS ENFORCED, STATED HONESTLY.  One of the four is
machine-checked: `test_oc_calibration.py::test_canonical_bytes_match_repo_when_
importable` re-runs `canonical_json_bytes` / `canonical_sha256` against the repo
copy when the repo is checked out, and SKIPS on the server, where it is not.  The
other three -- `load_bank`, `compute_architecture`, `generate_greedy` -- and the
mx45 measures below were ported and diffed by hand against the files named here;
no test pins them to the repo.  A silent divergence in those three is possible
and would have to be caught by reading, so the line numbers are given.

  * `canonical_json_bytes` / `canonical_sha256`
        `src/neural_forge/dynamic_circuit_ir.py` lines 82-96.
  * `load_bank`
        `src/neural_forge/lfm2_carrier_review_cli.py::_load_prompts` -- the
        digest check, the ordinary-file refusal (no symlink, `st_nlink == 1`),
        duplicate ids, exactly-one-prompt-form, and the chat-role validation.
  * `compute_architecture`
        `experiments/run_lfm2_matrix_cell.py::compute_architecture` -- reads the
        RESOLVED device, never an environment variable, because a wave-1 batch
        was once launched on what `nvidia-smi` called a 3090 and CUDA resolved
        to a P40.
  * `generate_greedy`
        `experiments/run_lfm2_composition_frozen_review.py::_generate` -- the
        manual argmax loop, `use_cache`, EOS stop, and the three termination
        states `complete` / `censored-at-cap` / `runtime-error`.  A cap hit is
        never a failure; the protocol says so and this records it separately
        rather than letting a reader infer it from a length.

  * `normalise`, `longest_common_substring_length`, `union_coverage`
        `reports/matrix-campaign-20260810/run/mx45/mx45_dualband.py`, arithmetic
        unchanged, `UNION_MIN_BLOCK = 8` and the 0.60 copy threshold unretuned.
        Retuning either of them here would make the omni columns incomparable
        with the LFM2 columns for no gain.

WHAT IS NEW, AND WHAT IT IS ALLOWED TO DECIDE.  `classify_form` puts one lane in
one of six boxes -- EMPTY, LOOP, ECHO, CAP_RIDE, MISSING_EOS, CONTENT -- which is
the machine half of the canonical artifact-tolerant lens
(`MC/ACTUAL_MATRIX_2026-08-28.md`).  It decides NOTHING about intelligence.  The
lens says an artifact-miss (empty, pure loop, indiscriminate copy, cap-ride with
no unique content) counts nothing and a content-miss counts against, and only a
human reading the text can tell "wrong answer" from "no answer".  So every
function in this file that touches a verdict returns a bound or a box, never a
grade, and every artifact it writes says so in its own text.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Canonical bytes.  src/neural_forge/dynamic_circuit_ir.py lines 82-96.
# --------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    """The estate's canonical serialisation.  Do not "improve" this."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


#: The file classes a model directory's identity is made of.  R3's port note (2)
#: says the single-file `model.safetensors` digest the LFM2 sealer used cannot
#: address a sharded parent, and every omni parent is sharded, so identity moves
#: to a digest over the sorted (relative path, sha256) pairs of the files that
#: decide what the model IS.  Caches, READMEs and licences are excluded on
#: purpose: they change without the weights changing.
DIRECTORY_DIGEST_SUFFIXES = (".safetensors", ".json", ".jinja", ".model", ".txt")
DIRECTORY_DIGEST_EXCLUDE_NAMES = ("README.md", "LICENSE", ".gitattributes")


def directory_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """(digest, per-file rows) over a model directory, deterministically.

    Hidden directories (`.cache`) are skipped: HuggingFace writes its own
    bookkeeping there and it is not part of the model.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.name in DIRECTORY_DIGEST_EXCLUDE_NAMES:
            continue
        if path.suffix not in DIRECTORY_DIGEST_SUFFIXES:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    digest = canonical_sha256(
        {
            "schema": "neural-forge-omni-model-directory-digest-v1",
            "files": rows,
        }
    )
    return digest, rows


# --------------------------------------------------------------------------
# Prompt banks.  lfm2_carrier_review_cli._load_prompts, refusals preserved.
# --------------------------------------------------------------------------


class CalibrationError(RuntimeError):
    """The calibration surface could not be established.  Always fatal."""


def ordinary_file(path: Path, label: str) -> Path:
    """Refuse a symlink or a hard link.

    `_load_prompts` refuses `st_nlink != 1` because a prompt pack that is a link
    can be re-pointed after its digest was checked, and the digest check is the
    only thing making the review's authority transferable.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise CalibrationError(f"{label} is not a file: {resolved}")
    if resolved.is_symlink():
        raise CalibrationError(f"{label} is a symlink: {resolved}")
    if resolved.stat().st_nlink != 1:
        raise CalibrationError(
            f"{label} has {resolved.stat().st_nlink} links: {resolved}.  A "
            "digest-pinned surface may not be reachable under a second name."
        )
    return resolved


def load_bank(
    path: Path, *, expected_file_sha256: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """(rows, payload, file digest) for a review bank, validated as the estate does.

    `expected_file_sha256` is optional HERE and mandatory in the runner for any
    SEALED bank: the draft bank v2 has no seal yet by construction, and pretending
    it does would be the exact "duplicated contract decay" failure.  The runner
    decides; this function only refuses malformed rows.

    ONE DELIBERATE DIVERGENCE FROM THE SEALED `_load_prompts`, STATED HERE
    BECAUSE THE REST OF THIS FUNCTION IS A FAITHFUL PORT.  The original requires
    every `content` to be a non-blank `str`.  Bank v2's vision cases carry
    content as a LIST of typed parts (`{"type": "image", ...}`), which the
    original would refuse outright, so a list is accepted -- and then held to the
    same "carries something" standard: a blank string and an EMPTY LIST are both
    refused.  Nothing else about the refusals moved.
    """

    source = ordinary_file(Path(path), "review prompt bank")
    observed = file_sha256(source)
    if expected_file_sha256 is not None and observed != str(expected_file_sha256):
        raise CalibrationError(
            f"{source} digests {observed}, the seal names {expected_file_sha256}."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("prompts") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise CalibrationError(f"{source} has no prompt rows.")
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CalibrationError(f"prompt row {index} must be an object.")
        prompt_id = row.get("id")
        category = row.get("category")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id.strip()
            or prompt_id in seen
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise CalibrationError(f"prompt row {index} has invalid identity metadata.")
        has_prompt = isinstance(row.get("prompt"), str) and bool(
            str(row["prompt"]).strip()
        )
        messages = row.get("messages")
        has_messages = (
            isinstance(messages, Sequence)
            and not isinstance(messages, (str, bytes))
            and bool(messages)
        )
        if has_prompt == has_messages:
            raise CalibrationError(
                f"prompt row {index} must have exactly one prompt form."
            )
        if has_messages and any(
            not isinstance(message, Mapping)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), (str, list))
            or (isinstance(message.get("content"), str) and not str(message["content"]).strip())
            or (isinstance(message.get("content"), list) and not message["content"])
            for message in messages
        ):
            raise CalibrationError(f"prompt row {index} has invalid chat messages.")
        seen.add(prompt_id)
        normalised.append(dict(row))
    return normalised, dict(payload), observed


def prompt_surface_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """The runner's own digest over the validated rows.

    Named separately from the pack digest for the reason the LFM2 runner names
    it: this is NOT the protocol's published normalized surface, and claiming it
    were would be a false pin.
    """

    return canonical_sha256(
        {
            "schema": "neural-forge-omni-calibration-prompt-surface-v1",
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


def messages_for(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("messages"), Sequence) and not isinstance(
        row.get("messages"), (str, bytes)
    ):
        return [dict(message) for message in row["messages"]]
    return [{"role": "user", "content": str(row["prompt"])}]


def user_turns_text(prompt_messages: Sequence[Mapping[str, Any]]) -> str:
    """Every user turn, concatenated.  mx45_dualband.lane_whole_prompt."""

    turns: list[str] = []
    for message in prompt_messages:
        role = message.get("role", "user")
        if role != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            turns.append(content)
        elif isinstance(content, Sequence):
            turns.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
    return "\n".join(turn for turn in turns if turn)


# --------------------------------------------------------------------------
# The copy measure.  mx45_dualband.py, arithmetic unchanged.
# --------------------------------------------------------------------------

UNION_MIN_BLOCK = 8
COPY_BACK_UNION_COVERAGE = 0.60
COPY_THRESHOLD_INVARIANCE_INTERVAL = (0.483461, 0.826972)

_MOD = (1 << 61) - 1
_BASE = 1_000_003


def normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def longest_common_substring_length(a: str, b: str) -> int:
    """mx45_dualband, verbatim: binary search over rolling hashes, verified."""

    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    n = len(a)

    def hashes(s: str, length: int) -> dict[int, int]:
        out: dict[int, int] = {}
        if length == 0 or len(s) < length:
            return out
        power = pow(_BASE, length, _MOD)
        acc = 0
        for i, ch in enumerate(s):
            acc = (acc * _BASE + ord(ch)) % _MOD
            if i >= length:
                acc = (acc - ord(s[i - length]) * power) % _MOD
            if i >= length - 1:
                out.setdefault(acc, i - length + 1)
        return out

    def common(length: int) -> bool:
        if length == 0:
            return True
        ha = hashes(a, length)
        if not ha:
            return False
        hb = hashes(b, length)
        for key, start_b in hb.items():
            start_a = ha.get(key)
            if start_a is None:
                continue
            if a[start_a : start_a + length] == b[start_b : start_b + length]:
                return True
        return False

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if common(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def union_coverage(payload: str, response: str) -> float:
    p, r = normalise(payload), normalise(response)
    if not p or not r:
        return 0.0
    matcher = difflib.SequenceMatcher(None, p, r, autojunk=False)
    covered = sum(
        block.size
        for block in matcher.get_matching_blocks()
        if block.size >= UNION_MIN_BLOCK
    )
    return min(1.0, covered / len(p))


# --------------------------------------------------------------------------
# The form boxes.  New, and deliberately not a grade.
# --------------------------------------------------------------------------

FORM_BOXES = ("EMPTY", "LOOP", "ECHO", "CAP_RIDE", "MISSING_EOS", "CONTENT")

#: An artifact box carries no information about intelligence either way -- it is
#: the canonical lens's "counts nothing" column.  CONTENT is the only box that
#: puts the lane in front of a reader; the reader then decides signal against
#: content-miss, and this module never does.
ARTIFACT_BOXES = ("EMPTY", "LOOP", "ECHO", "CAP_RIDE", "MISSING_EOS")


def detect_loop(ids: Sequence[int], *, minimum_period: int = 1) -> bool:
    """run_lfm2_matrix_cell._detect_loop, verbatim.

    True when the last 64 tokens are three or more repetitions of one window.
    """

    tail = list(ids)[-64:]
    if len(tail) < 8:
        return False
    for period in range(minimum_period, len(tail) // 3 + 1):
        window = tail[-period:]
        repeats = len(tail) // period
        if repeats < 3:
            continue
        if all(
            tail[-period * (r + 1) : len(tail) - period * r] == window for r in range(3)
        ):
            return True
    return False


def unique_content_ratio(text: str) -> float:
    """Distinct whitespace tokens over total.  A pure loop tends to zero."""

    words = normalise(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


#: A cap-ride is only an ARTIFACT when the text under the cap is repetition.  The
#: canonical lens is explicit that "looping after content is free" and that a
#: cap-ride WITH unique content is not an artifact-miss, so the box for a long
#: non-repeating answer that ran out of budget is CONTENT, and its termination
#: stays visible beside it.
CAP_RIDE_UNIQUE_RATIO = 0.35


def classify_form(
    *,
    response: str,
    payload: str,
    termination: str,
    token_ids: Sequence[int] | None = None,
    lane_ask: str = "PRODUCE",
) -> dict[str, Any]:
    """One lane, one box, with every number that produced it published.

    Order matters and is stated so a reader can re-decide any lane from the
    artifact: EMPTY first (nothing to read), then LOOP (nothing NEW to read),
    then ECHO (everything there is to read came from the prompt), then the two
    termination boxes, then CONTENT.
    """

    text = normalise(response)
    coverage = union_coverage(payload, response) if payload else 0.0
    unique = unique_content_ratio(response)
    looped = bool(token_ids is not None and detect_loop(token_ids))
    box = "CONTENT"
    reason = "readable text that is not a copy of the prompt"
    #: THE OUT-OF-MEMORY BOX IS TESTED FIRST, ahead of the plain empty box,
    #: because an empty lane's REASON is the whole measurement here.  Both land
    #: in the same artifact box and so both count nothing; but "the model emitted
    #: nothing" and "the prompt did not fit on this card" are different findings,
    #: and the second one is what the long-context rungs were run to discover.
    if termination == "out-of-memory":
        box, reason = (
            "EMPTY",
            "the prompt did not fit in memory on this device, so the model was "
            "never asked; this measures the envelope, not the model",
        )
    elif not text:
        box, reason = "EMPTY", "the model emitted no non-special text"
    elif looped and unique < CAP_RIDE_UNIQUE_RATIO:
        box, reason = "LOOP", "the tail is a repeated cycle and the text is repetition"
    elif payload and coverage >= COPY_BACK_UNION_COVERAGE:
        box, reason = (
            "ECHO",
            f"union coverage {coverage:.6f} of the lane payload, at or above "
            f"{COPY_BACK_UNION_COVERAGE}",
        )
    elif termination == "censored-at-cap" and unique < CAP_RIDE_UNIQUE_RATIO:
        box, reason = (
            "CAP_RIDE",
            f"hit the cap with unique-word ratio {unique:.4f} below "
            f"{CAP_RIDE_UNIQUE_RATIO}",
        )
    elif termination == "censored-at-cap":
        box, reason = (
            "CONTENT",
            "hit the cap while still saying new things; the protocol says a cap "
            "hit is never itself a failure",
        )
    elif termination == "runtime-error":
        box, reason = "EMPTY", "the decode raised and the text is what it got to"
        if text:
            box, reason = "CONTENT", "the decode raised after producing text"
    return {
        "form_box": box,
        "form_reason": reason,
        "is_artifact_box": box in ARTIFACT_BOXES,
        "payload_union_coverage": round(coverage, 6),
        "unique_word_ratio": round(unique, 6),
        "loop_detected": looped,
        "termination": termination,
        "missing_eos": termination == "censored-at-cap",
        "lane_ask": lane_ask,
        "copy_can_satisfy_this_ask": lane_ask == "PRESERVE",
        "decides_nothing_about_intelligence": True,
    }


# --------------------------------------------------------------------------
# Device and dtype.
# --------------------------------------------------------------------------


def compute_architecture(device: str) -> str:
    """`sm<major><minor> <name>`, read off the RESOLVED device."""

    if not str(device).startswith("cuda"):
        return f"cpu {sys.platform}"
    import torch

    if not torch.cuda.is_available():
        raise CalibrationError(f"--device {device} requested but CUDA is unavailable.")
    index = 0 if ":" not in str(device) else int(str(device).split(":", 1)[1])
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{major}{minor} {torch.cuda.get_device_name(index)}"


#: The linear-attention kernel path is PART OF THE SURFACE, not an environment
#: detail.  A Qwen3.5 parent is a hybrid: three quarters of its layers are Gated
#: DeltaNet, and each such layer binds FOUR primitives that transformers resolves
#: silently at import time -- either to an installed kernel or to its own torch
#: implementation.  The two are the same function in exact arithmetic and NOT the
#: same function in bf16: they accumulate the recurrence in a different order, so
#: a greedy argmax can land on the other side of a near-tie and one lane's text
#: moves.  That is why this belongs beside the dtype in `lane_digest_validity`.
#:
#: THE FIRST VERSION OF THIS FIELD GOT THE RULE WRONG AND THIS LANE'S OWN LOG
#: REFUTED IT.  It derived the label from the package INVENTORY -- "fla AND
#: causal-conv1d, else torch-fallback" -- on the strength of the transformers log
#: line "The fast path is not available because one of the required library is
#: not installed.  Falling back to torch implementation".  That line is printed
#: off `is_fast_path_available`, which is `all()` of four callables
#: (modeling_qwen3_5.py:217-219, :425-429) -- but the four are bound
#: INDEPENDENTLY (modeling_qwen3_5.py:420-423):
#:
#:     self.causal_conv1d_fn           = causal_conv1d_fn                       # None when absent
#:     self.causal_conv1d_update       = causal_conv1d_update or torch_causal_conv1d_update
#:     self.chunk_gated_delta_rule     = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
#:     self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
#:
#: so with `fla` installed and `causal_conv1d` missing the warning fires, the
#: convolution runs in torch, AND FLA'S TRITON CHUNKED GATED-DELTANET KERNEL
#: STILL RUNS.  This is not a reading of the source: out/r6-lc64k-Q35-4B.log
#: lines 137-152 are a live traceback through `modeling_qwen3_5.py:535 ->
#: self.chunk_gated_delta_rule -> fla/ops/gated_delta_rule/chunk.py:567`, taken
#: on the box whose every r6 log carries that same warning on line 50.  The old
#: label would have written `torch-fallback` into the validity tuple of a run
#: that used Triton kernels: the one field added to make a moved lane
#: ATTRIBUTABLE, supplying a false attribution.
#:
#: So the label is now taken from WHAT THE LOADED MODEL ACTUALLY BOUND, one entry
#: per primitive, and the package inventory is published beside it as the
#: separate and weaker thing it always was.
#:
#: It is deliberately NOT a refusal.  Which kernel path the campaign standardises
#: on is a decision for Cyril; the instrument's job is to say which one ran.

#: Bumped when the MEANING of `kernel_path` changes, because that string travels
#: in `lane_digest_validity` and a reader comparing two dossiers has to be able
#: to tell a moved surface from a moved definition.  v1 was the refuted inventory
#: rule above -- in a v1 dossier `"fla-0.5.2"` means *fla present and
#: causal_conv1d absent*, and `"torch-fallback"` was asserted rather than
#: measured, so NEITHER v1 string is comparable with a v2 string.  v2 is the
#: per-primitive probe of the loaded model.
KERNEL_REPORT_SCHEMA = "neural-forge-omni-kernel-path-v2"

#: The four primitives a Gated-DeltaNet layer binds, in the order transformers
#: binds them, with the slot names used in the `kernel_path` string.
GATED_DELTA_PRIMITIVES = (
    ("chunk_gated_delta_rule", "chunk"),
    ("recurrent_gated_delta_rule", "recurrent"),
    ("causal_conv1d_fn", "conv-prefill"),
    ("causal_conv1d_update", "conv-decode"),
)


def _unwrap_callable(value: Any) -> Any:
    """Follow `functools.wraps` / dynamo wrappers to the real function.

    `fla` hands its kernels back through `fla.ops.backends` and torch through
    `torch._dynamo.eval_frame`; both copy `__wrapped__`, so the provider is only
    reliably readable after unwrapping.  Bounded, because a cycle here would
    hang a diagnostic.
    """

    seen = 0
    while seen < 8:
        inner = getattr(value, "__wrapped__", None)
        if inner is None or inner is value:
            break
        value = inner
        seen += 1
    return value


def _binding_provider(value: Any) -> str:
    """Which library supplies this bound primitive.

    `None` IS A REAL ANSWER AND IT MEANS TORCH: transformers binds
    `causal_conv1d_fn` to None when the package is missing and then branches
    `if self.causal_conv1d_fn is not None` onto its own convolution
    (modeling_qwen3_5.py:488).  Reporting it as "missing" would describe the
    inventory again instead of the path.
    """

    if value is None:
        return "torch"
    module = str(getattr(_unwrap_callable(value), "__module__", "") or "")
    root = module.split(".", 1)[0]
    if root in ("fla", "causal_conv1d"):
        return root
    if root == "transformers":
        return "torch"
    return f"unknown:{module}" if module else "unknown"


def _binding_callable_name(value: Any) -> str | None:
    if value is None:
        return None
    target = _unwrap_callable(value)
    module = str(getattr(target, "__module__", "") or "?")
    name = str(
        getattr(target, "__qualname__", None) or getattr(target, "__name__", "?")
    )
    return f"{module}.{name}"


def gated_delta_layers(model: Any) -> list[Any]:
    """Every submodule that binds the Gated-DeltaNet primitives.

    Named and returned rather than inlined because "how many layers did you
    actually look at" is part of the answer: a report taken from zero layers is
    `not-applicable`, and must not be allowed to read as `torch`.
    """

    if model is None:
        return []
    try:
        candidates = list(model.modules())
    except (AttributeError, TypeError):  # a stub, or not an nn.Module
        candidates = [model]
    return [m for m in candidates if hasattr(m, "chunk_gated_delta_rule")]


def kernel_path_report(model: Any = None) -> dict[str, Any]:
    """Which linear-attention kernels the LOADED MODEL bound, per primitive.

    THE LABEL REPORTS WHAT RUNS.  The previous version inferred it from which
    packages imported, and that inference is refuted by this lane's own r6
    traceback (see the note above): the four primitives fall back independently,
    so a box with `fla` and without `causal_conv1d` runs Triton for the
    recurrence and torch for the convolution at the same time -- which is
    exactly the state of this server, and exactly the state the old rule called
    `torch-fallback`.

    `model=None` is honest rather than convenient: with no model there is
    nothing bound to probe, so `kernel_path` reads `not-probed` and no caller
    can mistake the inventory for a path.
    """

    found: dict[str, str | None] = {}
    for name in ("fla", "causal_conv1d"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 -- absence and breakage read the same
            found[name] = None
        else:
            found[name] = str(getattr(module, "__version__", "unknown"))
    versions = {"fla": found["fla"], "causal_conv1d": found["causal_conv1d"]}
    present = [f"{n}-{v}" for n, v in sorted(found.items()) if v is not None]
    #: What the refuted v1 rule WOULD have printed, kept so a reader holding a
    #: v1 dossier can line it up against a v2 one without guessing.
    inventory_label_v1 = (
        "+".join(present)
        if all(value is not None for value in found.values())
        else "torch-fallback"
    )

    layers = gated_delta_layers(model)
    bindings: dict[str, dict[str, Any]] = {}
    for attribute, slot in GATED_DELTA_PRIMITIVES:
        if not layers:
            continue
        bound = getattr(layers[0], attribute, None)
        provider = _binding_provider(bound)
        bindings[slot] = {
            "provider": provider,
            "callable": _binding_callable_name(bound),
            "version": versions.get(provider),
        }
    #: Every Gated-DeltaNet layer binds in the same `__init__`, so disagreement
    #: means something was patched per layer -- cheap to check, and a silent
    #: per-layer mix would make any single label a lie.
    uniform = (
        all(
            _binding_provider(getattr(layer, attribute, None))
            == bindings[slot]["provider"]
            for layer in layers
            for attribute, slot in GATED_DELTA_PRIMITIVES
        )
        if layers
        else None
    )

    if model is None:
        kernel_path = "not-probed"
    elif not layers:
        kernel_path = "not-applicable (no gated-deltanet layers)"
    else:
        kernel_path = "; ".join(
            "{}={}".format(
                slot,
                bindings[slot]["provider"]
                if bindings[slot]["version"] is None
                else "{}-{}".format(
                    bindings[slot]["provider"], bindings[slot]["version"]
                ),
            )
            for _attribute, slot in GATED_DELTA_PRIMITIVES
        )

    #: The flag the warning is printed off, read from the module that defines
    #: the layer.  It is NOT the path -- it is `all()` of the four -- and it is
    #: published so the log line and the path can be reconciled by a reader.
    fast_path_flag = None
    if layers:
        defining = sys.modules.get(type(layers[0]).__module__)
        flag = getattr(defining, "is_fast_path_available", None)
        fast_path_flag = None if flag is None else bool(flag)

    return {
        "kernel_report_schema": KERNEL_REPORT_SCHEMA,
        "fla_version": found["fla"],
        "causal_conv1d_version": found["causal_conv1d"],
        "packages_present": present,
        "gated_delta_layers_probed": len(layers),
        "bindings_uniform_across_layers": uniform,
        "bindings": bindings,
        "transformers_fast_path_flag": fast_path_flag,
        "inventory_label_v1_refuted": inventory_label_v1,
        "kernel_path": kernel_path,
        "note": (
            "`kernel_path` is the PER-PRIMITIVE binding read off the loaded "
            "model, not the package inventory.  transformers binds the four "
            "Gated-DeltaNet primitives independently, so fla's Triton chunked "
            "kernel runs even when causal_conv1d is absent and the 'fast path "
            "is not available' warning fires -- that warning is printed off "
            "is_fast_path_available, which is all() of the four.  "
            "`inventory_label_v1_refuted` is what the earlier, wrong rule would "
            "have printed, published only so v1 dossiers can be lined up "
            "against v2 ones.  Recorded, never enforced."
        ),
    }


#: R3 port note (3): float32 is impossible at 27B, so the omni review dtype is
#: bf16 on sm86 and fp16-after-overflow-check on sm75, LABELLED in the dossier.
#: Parent and child must be decoded in the same dtype in the same process or the
#: two texts are not comparable.
REVIEW_DTYPES = {"bfloat16", "float16", "float32"}


def resolve_dtype(name: str) -> Any:
    import torch

    if name not in REVIEW_DTYPES:
        raise CalibrationError(f"{name!r} is not one of {sorted(REVIEW_DTYPES)}.")
    return getattr(torch, name)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# Generation.  run_lfm2_composition_frozen_review._generate, port notes inline.
# --------------------------------------------------------------------------


THINKING_MODES = ("template-default", "off")


def chat_template_kwargs_for(thinking: str) -> dict[str, Any]:
    if thinking not in THINKING_MODES:
        raise CalibrationError(
            f"{thinking!r} is not one of {list(THINKING_MODES)}."
        )
    #: `enable_thinking` is Qwen's own template variable.  A template that does
    #: not define it ignores an unknown kwarg, so `off` degrades to the default
    #: rather than failing -- and `thinking_took_effect` below is what tells the
    #: reader which of the two actually happened.
    return {"enable_thinking": False} if thinking == "off" else {}


def render_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking: str = "template-default",
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs_for(thinking),
        )
    raise CalibrationError(
        "the tokenizer carries no chat template, and the frozen protocol grades "
        "chat-formatted prompts; a raw-concatenation fallback would silently "
        "grade a different surface."
    )


def thinking_report(tokenizer: Any, messages: Sequence[Mapping[str, Any]],
                    thinking: str) -> dict[str, Any]:
    """Did the requested thinking mode actually change the rendered prompt?

    A recorded setting that the template ignored is worse than no setting, so
    both renderings are produced once, on one case, and the dossier carries the
    ANSWER rather than the request.
    """

    default = render_prompt(tokenizer, messages, thinking="template-default")
    chosen = render_prompt(tokenizer, messages, thinking=thinking)
    return {
        "requested": thinking,
        "took_effect": chosen != default,
        "template_opens_think_block": default.rstrip().endswith("<think>"),
        "rendered_tail": chosen[-80:],
        "note": (
            "`took_effect` false with `requested: off` means the template has no "
            "`enable_thinking` variable and the run used the template default."
        ),
    }


def logits_to_keep_kwargs(model: Any) -> dict[str, Any]:
    """`{"logits_to_keep": 1}` when the model takes it, `{}` when it does not.

    WHY THIS IS NOT AN OPTIMISATION.  A bare `model(input_ids=prompt)` returns
    logits for EVERY position, and the omni parents have a 248,320-token
    vocabulary: an 8,192-token prefill materialises 8,192 x 248,320 x 2 bytes =
    4.07 GB, and the 200,000-token long-context lane would ask for 99 GB.  The
    long-context half of bank v2 is unrunnable without this, on any card in the
    building.  Greedy decoding only ever reads the LAST position, so keeping one
    changes no token -- `test_logits_to_keep_changes_no_token` pins that -- and
    the dossier records whether it was available, because a run that silently
    fell back would OOM on the lane it was built for.
    """

    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {}
    return {"logits_to_keep": 1} if "logits_to_keep" in parameters else {}


def generate_greedy(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    device: str,
    eos_id: int | None,
    label: str,
    extra_eos_ids: Sequence[int] = (),
    use_logits_to_keep: bool = True,
    thinking: str = "template-default",
) -> list[dict[str, Any]]:
    """Greedy argmax, `use_cache`, one response per row.

    `extra_eos_ids` exists because a chat model can end a turn on a token that is
    not `eos_token_id` (Qwen3.5 ends on `<|im_end|>`), and a runner that only
    watched `eos_token_id` would record every well-formed answer as
    `censored-at-cap`.  That is not a grading change -- the protocol says a cap
    hit is never a failure either way -- but it is a TERMINATION-ACCOUNTING
    change, so the ids used are written into the dossier.
    """

    import torch

    stops = {int(x) for x in extra_eos_ids}
    if eos_id is not None:
        stops.add(int(eos_id))
    keep = logits_to_keep_kwargs(model) if use_logits_to_keep else {}
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            started = time.monotonic()
            messages = messages_for(row)
            text = render_prompt(tokenizer, messages, thinking=thinking)
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            prompt_tokens = int(ids.shape[-1])
            generated: list[int] = []
            past = None
            current = ids
            #: `result` IS THE ROW'S SECOND REFERENCE TO ITS OWN KV CACHE and it
            #: has to be named out here so it can be dropped.  It is a
            #: function-scoped local rebound by the next forward, so without an
            #: explicit clear the PREVIOUS row's `result.past_key_values` stays
            #: alive through the whole of the next row's prefill: `past = None`
            #: frees nothing while `result` still holds it, and the
            #: `empty_cache()` in the handlers below then frees nothing either.
            #: On the 64k rung that is ~20 GiB of a 24 GiB card carried into the
            #: next lane, which would turn one genuine out-of-memory into three
            #: and make the published envelope depend on lane ORDER.  Cleared
            #: after every row and before every `empty_cache()`.
            result = None
            termination = "censored-at-cap"
            error_text = None
            try:
                for _step in range(cap):
                    result = model(
                        input_ids=current, past_key_values=past, use_cache=True, **keep
                    )
                    past = result.past_key_values
                    next_id = int(result.logits[0, -1].argmax().item())
                    if next_id in stops:
                        termination = "complete"
                        break
                    generated.append(next_id)
                    current = torch.tensor(
                        [[next_id]], device=device, dtype=ids.dtype
                    )
            #: AN OUT-OF-MEMORY IS A MEASUREMENT AND GETS ITS OWN TERMINATION.
            #: The long-context lanes exist to find the point where this runner
            #: stops fitting on this card, and "the 64k rung raised
            #: torch.OutOfMemoryError at N prompt tokens" is the ANSWER to that
            #: question, not a failure to answer it.  Folded in with generic
            #: runtime errors it would read as a broken run; recorded as its own
            #: state, with the prompt length beside it, it is the envelope.
            #: The cache is emptied so the rung's REMAINING cases still get their
            #: chance -- one lane that did not fit must not silently condemn the
            #: lanes after it.
            except torch.OutOfMemoryError as exc:
                termination = "out-of-memory"
                error_text = f"{type(exc).__name__}: {exc}"
                log(
                    f"{label} {row['id']}: OUT OF MEMORY at {prompt_tokens} "
                    f"prompt tokens -- recorded as the envelope, not as a "
                    f"failure.  Continuing."
                )
                result = None
                past = None
                current = None
                torch.cuda.empty_cache()
            except RuntimeError as exc:  # recorded, never swallowed
                termination = "runtime-error"
                error_text = f"{type(exc).__name__}: {exc}"
                log(f"{label} {row['id']}: {error_text}")
                #: Same clear on this branch: a CUDA runtime error that is not
                #: raised as torch.OutOfMemoryError still leaves a full cache
                #: referenced, and the next lane pays for it.
                result = None
                past = None
                current = None
                torch.cuda.empty_cache()
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            records.append(
                {
                    "id": str(row["id"]),
                    "category": str(row["category"]),
                    "prompt_messages": messages,
                    "prompt_tokens": prompt_tokens,
                    "response_verbatim": decoded,
                    "response_sha256": canonical_sha256({"text": decoded}),
                    "logits_to_keep_used": bool(keep),
                    "generated_tokens": len(generated),
                    "generated_token_ids_sha256": canonical_sha256(
                        {"ids": [int(x) for x in generated]}
                    ),
                    "loop_detected": detect_loop(generated),
                    "termination": termination,
                    "runtime_error": error_text,
                    "wall_time_seconds": round(time.monotonic() - started, 3),
                }
            )
            log(
                f"{label} {index}/{len(rows)} {row['id']}: prompt "
                f"{prompt_tokens} tok -> {len(generated)} tok ({termination}) "
                f"in {records[-1]['wall_time_seconds']:.1f}s"
            )
            #: END OF ROW: DROP THE CACHE BEFORE THE NEXT PREFILL.  This is the
            #: line that makes the long-context envelope a property of the lane
            #: instead of a property of the running order.  It is checked by
            #: `test_generate_greedy_holds_at_most_one_row_of_kv_cache`, which
            #: counts live forward outputs from inside the model.
            result = None
            past = None
            current = None
    return records


def decode_smoke(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    cap: int,
    device: str,
    eos_id: int | None,
    extra_eos_ids: Sequence[int] = (),
    thinking: str = "template-default",
) -> list[dict[str, Any]]:
    """The three-prompt coherence smoke, `eos|cap|loop|error`, 128-token cap.

    `experiments/matrix_campaign_smoke_prompts_v1.json` is the sealed set and its
    digest is checked by the caller; this only decodes it.  The verdict on the
    smoke is a READING (`coherent` / `incoherent`), taken by a human, exactly as
    `seal_matrix_behavior_review.py --smoke-verdict` takes it.
    """

    rows = [
        {"id": f"smoke-{i}", "category": "decode-smoke", "prompt": prompt}
        for i, prompt in enumerate(prompts, start=1)
    ]
    records = generate_greedy(
        model,
        tokenizer,
        rows,
        cap=cap,
        device=device,
        eos_id=eos_id,
        label="smoke",
        extra_eos_ids=extra_eos_ids,
        thinking=thinking,
    )
    for record in records:
        if record["termination"] == "complete":
            record["terminated"] = "eos"
        elif record["termination"] == "runtime-error":
            record["terminated"] = "error"
        elif record["loop_detected"]:
            record["terminated"] = "loop"
        else:
            record["terminated"] = "cap"
    return records


# --------------------------------------------------------------------------
# The emit-zero floor.
# --------------------------------------------------------------------------

#: WHAT THIS IS AND WHAT IT IS NOT.  In the LFM2 sealer the floor is
#: `terminal_heldout_relL2 < 1.0` -- a CHILD-versus-PARENT statement: a child
#: whose terminal state is the zero vector sits at exactly 1.0 relative L2 from
#: its parent, so anything under 1.0 proves the child still writes something into
#: the residual stream.  A PARENT has no parent, so that form does not exist for
#: a calibration run and inventing a number for it would be a fabricated pin.
#:
#: The parent-side floor is the same claim measured directly: the terminal hidden
#: state is finite and non-zero, the logits are finite and not constant, and at
#: least one probe emitted a non-stop token.  When a reference terminal-state
#: file IS supplied (a later fold child against this parent) the relL2 form is
#: computed too and BOTH are published; the child-side number is the sealed one.
EMIT_ZERO_FLOOR_SCHEMA = "neural-forge-omni-emit-zero-floor-v1"


def emit_zero_floor(
    model: Any,
    tokenizer: Any,
    probe_prompts: Sequence[str],
    *,
    device: str,
    reference_terminal: Any = None,
) -> dict[str, Any]:
    """Does this model write anything at all?

    A DIAGNOSTIC, AND NOTHING IN THIS LANE GATES ON IT.  It used to call itself
    "a hard floor", which was wrong twice over: no caller stops on
    `passed: false` (`oc_run_frozen_bank.run` records it and keeps going,
    `oc_assess` surfaces the boolean), and inventing a gate is exactly what the
    campaign's standing rule forbids.  A failed floor is the strongest single
    signal in the dossier that a checkpoint emits nothing -- and it is still the
    reader who decides what to do about it.
    """

    import torch

    rows: list[dict[str, Any]] = []
    terminals: list[Any] = []
    with torch.no_grad():
        for index, prompt in enumerate(probe_prompts, start=1):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            terminal = out.hidden_states[-1][0, -1].float()
            logits = out.logits[0, -1].float()
            terminals.append(terminal)
            rows.append(
                {
                    "probe_index": index,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "terminal_l2": float(terminal.norm().item()),
                    "terminal_finite": bool(torch.isfinite(terminal).all().item()),
                    "logit_std": float(logits.std().item()),
                    "logit_finite": bool(torch.isfinite(logits).all().item()),
                    "argmax_token_id": int(logits.argmax().item()),
                }
            )
    stacked = torch.stack(terminals)
    reference_rel_l2 = None
    if reference_terminal is not None:
        reference = reference_terminal.float().to(stacked.device)
        if reference.shape != stacked.shape:
            raise CalibrationError(
                f"reference terminal states are {tuple(reference.shape)} and this "
                f"model's are {tuple(stacked.shape)}; they are not comparable."
            )
        reference_rel_l2 = float(
            ((stacked - reference).norm() / reference.norm()).item()
        )
    distinct_argmax = len({row["argmax_token_id"] for row in rows})
    passed = (
        all(row["terminal_finite"] and row["logit_finite"] for row in rows)
        and min(row["terminal_l2"] for row in rows) > 0.0
        and min(row["logit_std"] for row in rows) > 0.0
        and (reference_rel_l2 is None or reference_rel_l2 < 1.0)
    )
    return {
        "schema": EMIT_ZERO_FLOOR_SCHEMA,
        "probes": rows,
        "probe_count": len(rows),
        "distinct_argmax_tokens": distinct_argmax,
        "min_terminal_l2": min(row["terminal_l2"] for row in rows),
        "min_logit_std": min(row["logit_std"] for row in rows),
        "reference_terminal_relative_l2": reference_rel_l2,
        "terminal_states_sha256": canonical_sha256(
            {"rows": [round(row["terminal_l2"], 6) for row in rows]}
        ),
        "passed": bool(passed),
        "definition": (
            "finite, non-zero terminal hidden state and non-constant finite "
            "logits on every probe.  When a reference terminal-state tensor is "
            "supplied the sealer's own form -- relative L2 < 1.0 against it -- "
            "is computed as well and BOTH are published.  A parent has no "
            "parent, so the relL2 form is null on a calibration run and is not "
            "faked."
        ),
    }


def write_canonical(path: Path, body: Mapping[str, Any]) -> str:
    """Write canonical bytes through `.partial`, then parse the file back.

    NOTES-MX-AO section 9: parse the artifact in the command that writes it.  An
    artifact that did not survive its own write is a silent data loss and this
    estate has been bitten by exactly that.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(body))
    temporary.replace(destination)
    reread = json.loads(destination.read_text(encoding="utf-8"))
    if reread.get("schema") != body.get("schema"):
        raise CalibrationError(f"{destination} did not survive its own write.")
    return file_sha256(destination)
