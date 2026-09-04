#!/usr/bin/env python
"""k3_corpus_build.py -- stage S1 of the composite step-1 run book: the shared
calibration corpus on the KIMI-K3 side of the fold.

What it produces (all under ``--out-dir``, default ``$OC/composite/corpus``):

* ``k3_input_ids.safetensors``   the EVIDENCE rows only, ``[rows, sequence_length]``
  (24 x 4,096 for the 27B manifest): the SAME raw text the sealed manifest
  carries as Qwen ids, spelled in K3's tiktoken ids, document separators mapped
  to K3's ``[EOS]`` and the tail of every row padded with K3's ``[PAD]``.  Beside
  ``input_ids`` the file carries the split IN-BAND so no consumer can fit on a
  row it should not: ``split`` int8 ``[rows]`` (0 fit / 1 held / 2 review),
  ``real_lengths`` int64 ``[rows]``, ``fit_mask`` bool ``[rows, T]`` (the real
  tokens of fit rows only) and ``held_mask`` bool ``[rows, T]``.
* ``qwen_input_ids.safetensors``  the Qwen side of the same rows (verbatim from
  the manifest) in the same layout with the same in-band tensors.
* ``k3_review_input_ids.safetensors`` / ``qwen_review_input_ids.safetensors``
  (with ``--append-bank-rows``) the review surface -- bank v1 thinking-off
  prompts, bank v2 draft text cases, the probe and smoke packs -- as MASKED rows
  (``split`` 2, ``fit_mask`` all False) in a SEPARATE short file with its own
  ``--review-sequence-length`` (default 512; the longest real review prompt is
  ~150 K3 tokens), so the S2 teacher stream emits K3's teacher-forced logits on
  the review prompts as one cheap extra batch instead of 48 rows of 4,096 that
  are 98 % padding.  These rows are the Qwen chat template spelled as literal
  text (the pairing with the Qwen side is exact; K3 never sees that markup as
  its own specials) -- the receipt says so under ``k3_review_render``.
* ``k3_review_native_input_ids.safetensors`` (``--k3-bank-render k3-native``,
  the default with ``--append-bank-rows``) the same chat prompts rendered by
  K3's OWN ``tokenization_kimi.TikTokenTokenizer.apply_chat_template(thinking=False)``
  with K3's own specials (this checkpoint's ``encoding_k3`` renders XTML:
  ``<|open|>`` 163587, ``<|sep|>`` 163589, ``<|close|>`` 163588,
  ``<|end_of_msg|>`` 163586 -- the ids the build reports per row): the
  K3-native review baseline.  K3 only -- review rows
  are never charted, so no Qwen pairing is needed or written.
* ``row_meta.json``               per row (``file`` + ``row_index``): source,
  origin, split, masking flag, real token counts on both sides, K3 truncation
  and UTF-8 facts, the sha256 of the raw bytes both tokenizations spell.
* ``boundary_map.json``           per PAIRED row the Qwen<->K3 POSITION pairs at
  common byte boundaries (token index on each side whose consumed bytes end at
  the same offset of the row's raw byte stream; separators paired explicitly),
  with the aligned-position fraction -- the ceiling on the rows the S4 charts
  can pair; carries the manifest seal and the sha256 of every ids file.
* ``k3_corpus_build_receipt.json`` canonical JSON with sha256 of every input and
  output, the tokenizer checks, counts, the S2 pricing (positions per file) and
  the aggregate alignment numbers.

Refusals (every one a ``SystemExit``, nothing written):

* the evidence manifest fails its own seal (``qf_corpus.read_manifest``);
* the manifest's ``tokenizer_sha256`` is not the sha256 of ``--qwen-dir``'s
  ``tokenizer.json`` (the ``qf_build.assert_evidence_tokenizer`` refusal carried
  here verbatim in spirit) -- ``--allow-tokenizer-mismatch`` is the recorded
  escape;
* that tokenizer.json is not the pinned 27B tokenizer ``06b95093...`` --
  ``--allow-unpinned-tokenizer``;
* ``tiktoken.model`` is not the pinned K3 rank file ``b6c497a7...`` --
  ``--allow-unpinned-k3-tokenizer``;
* the Qwen separator id cannot be read from BOTH the manifest's sibling build
  receipt (``packing.padding_token_id`` / ``tokenizer.separator_id``) and the
  parent's ``generation_config.json`` (``pad_token_id``), or the two disagree --
  there is no literal fallback; ``--qwen-separator-id`` is the explicit override;
* the default evidence resolves to the spec path
  ``composite/corpus/omni-evidence-27b-v1.json`` but the lane copy under
  ``qwen-fold/evidence/`` exists and differs -- name one with ``--evidence``;
* any input path (evidence, parent dir, K3 dir, every bank file, the qwen-fold
  lane, the chimera src tree, the calibration dir) under ``/mnt/ai_data`` while
  share reads are off (off by default whenever ``--k3-dir`` is NOT under the
  share) -- ``--allow-share-read`` permits the small files;
* an evidence row whose last document is not closed by the separator, or a
  document with invalid UTF-8 INSIDE it (named by batch index and byte offset);
* a K3 round trip that does not reproduce the raw bytes of a row (asserted for
  every row before anything is written).

Reuse: ``qwen-fold/qf_corpus.py`` (manifest reader, seal), ``qf_arch``
(canonical JSON), chimera-omni ``importing.tokenizer`` (``ChimeraTokenizer.from_qwen
/ from_k3`` -> ``JsonBpeTokenizer`` / ``TikTokenBpeTokenizer`` with the K3 split
pattern and reserved specials), ``importing.bridge.token_alignment``'s byte
round-trip idea extended from vocabulary level to POSITION level (byte offsets)
in :func:`align_positions`, K3's own ``tokenization_kimi.py`` for the native
review render, and ``calibration/oc_bank_v2_draft.materialise_haystack`` for
long-context cases when ``--include-overlong truncate`` asks for them.

Runs on CPU, no torch tensor work (torch is imported only because ``qf_corpus``
imports it), < 2 GB RAM, seconds of wall time.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_ROW_META = "neural-forge-k3-corpus-row-meta-v2"
SCHEMA_BOUNDARY_MAP = "neural-forge-k3-boundary-map-v2"
SCHEMA_RECEIPT = "neural-forge-k3-corpus-build-receipt-v2"
SCHEMA_IDS = "neural-forge-k3-corpus-ids-v2"

HERE = Path(__file__).resolve().parent
#: composite/tools -> the campaign root ($OC on the server, the report mirror locally).
CAMPAIGN_ROOT = HERE.parent.parent

#: Qwen3.8-27B-Uncensored tokenizer.json (C02).  The 4B/0.8B tokenizer hashes
#: 5f9e4d49...; a manifest carrying that sha is the 4B evidence, not this corpus.
EXPECTED_QWEN_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
QWEN_4B_TOKENIZER_SHA256_PREFIX = "5f9e4d49"
#: /mnt/lexar/models/moonshotai/Kimi-K3/tiktoken.model, measured 2026-09-02.
EXPECTED_K3_TIKTOKEN_SHA256 = "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"

SHARE_ROOT = "/mnt/ai_data"
DEFAULT_K3_DIR = "/mnt/lexar/models/moonshotai/Kimi-K3"
DEFAULT_QWEN_DIR = "/home/cereal/models/Qwen3.8-27B-Uncensored"
#: The run book's S3/S4/S5 commands name this path; S1's qf_evidence_build writes it.
DEFAULT_EVIDENCE_SPEC = CAMPAIGN_ROOT / "composite" / "corpus" / "omni-evidence-27b-v1.json"
#: The qwen-fold lane's copy of the same 27B manifest (built 2026-09-02).
DEFAULT_EVIDENCE_LANE = CAMPAIGN_ROOT / "qwen-fold" / "evidence" / "omni-evidence-27bu-v1.json"
DEFAULT_OUT_DIR = CAMPAIGN_ROOT / "composite" / "corpus"
DEFAULT_CALIBRATION_DIR = CAMPAIGN_ROOT / "calibration"
DEFAULT_BANKS = DEFAULT_CALIBRATION_DIR / "banks"
DEFAULT_BANK_V1 = DEFAULT_BANKS / "lfm2_semantic_whole_response_prompts_v1.json"
DEFAULT_BANK_V2 = DEFAULT_BANKS / "omni_review_bank_v2_draft.json"
DEFAULT_PROBE = DEFAULT_BANKS / "matrix_campaign_probe_prompts_v1.json"
DEFAULT_SMOKE = DEFAULT_BANKS / "matrix_campaign_smoke_prompts_v1.json"
DEFAULT_QWEN_FOLD_DIR = CAMPAIGN_ROOT / "qwen-fold"
#: The server git-archive tree only.  A workstation checkout is named through
#: ``CHIMERA_OMNI_SRC`` or ``--chimera-src``; no machine-specific path lives here.
CHIMERA_SRC_CANDIDATES = (CAMPAIGN_ROOT / "import" / "rq27b" / "tree_c5e6052" / "src",)
CHIMERA_SRC_ENV = "CHIMERA_OMNI_SRC"

BANK_RENDERS = ("qwen-chat", "plain")
K3_BANK_RENDERS = ("k3-native", "none")
OVERLONG_POLICIES = ("skip", "truncate")
DEFAULT_REVIEW_SEQUENCE_LENGTH = 512

SPLIT_CODES = {"fit": 0, "held": 1, "review": 2}
IN_BAND_TENSORS = ("input_ids", "split", "real_lengths", "fit_mask", "held_mask")

FILE_EVIDENCE = "k3_input_ids"
FILE_EVIDENCE_QWEN = "qwen_input_ids"
FILE_REVIEW = "k3_review_input_ids"
FILE_REVIEW_QWEN = "qwen_review_input_ids"
FILE_REVIEW_NATIVE = "k3_review_native_input_ids"

# Lane modules, bound by :func:`bind_lanes`.  Module globals so the helpers
# below can reach them without threading them through every signature.
qf_corpus: Any = None
qf_arch: Any = None
ctok: Any = None


def log(message: str) -> None:
    print(f"[k3_corpus_build {time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def resolve_chimera_src(chimera_src: str | Path | None) -> Path:
    return Path(chimera_src or os.environ.get(CHIMERA_SRC_ENV) or first_existing(CHIMERA_SRC_CANDIDATES))


def bind_lanes(
    *,
    qwen_fold_dir: str | Path | None = None,
    chimera_src: str | Path | None = None,
) -> None:
    """Import the lane modules this tool reuses (read, never edited)."""

    global qf_corpus, qf_arch, ctok
    fold = Path(qwen_fold_dir or DEFAULT_QWEN_FOLD_DIR)
    src = resolve_chimera_src(chimera_src)
    if not (fold / "qf_corpus.py").is_file():
        raise SystemExit(f"qwen-fold lane not found: {fold}/qf_corpus.py")
    if not (src / "chimera_omni" / "importing" / "tokenizer.py").is_file():
        raise SystemExit(
            f"chimera-omni source tree not found: {src}/chimera_omni/importing/tokenizer.py "
            f"(name it with --chimera-src or {CHIMERA_SRC_ENV})"
        )
    for entry in (str(fold), str(src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import qf_arch as _qf_arch  # noqa: E402
    import qf_corpus as _qf_corpus  # noqa: E402
    from chimera_omni.importing import tokenizer as _ctok  # noqa: E402

    qf_corpus, qf_arch, ctok = _qf_corpus, _qf_arch, _ctok


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(qf_arch.canonical_json_bytes(value)).hexdigest()


def under_share(path: str | Path) -> bool:
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = str(path)
    return resolved.startswith(SHARE_ROOT) or str(path).startswith(SHARE_ROOT)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Evidence resolution and the Qwen separator (findings 5 and 6).
# --------------------------------------------------------------------------


def resolve_evidence(argument: str | None, *, spec: Path | None = None, lane: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """The manifest to build on.

    An explicit ``--evidence`` wins.  Otherwise the run book's spec path is
    preferred; when it is missing the qwen-fold lane copy is used; when BOTH
    exist and are not the same bytes this refuses -- the K3 rows must match the
    manifest the trunk taps are run on, and only a named choice can say which.
    """

    spec = Path(spec or DEFAULT_EVIDENCE_SPEC)
    lane = Path(lane or DEFAULT_EVIDENCE_LANE)
    if argument is not None:
        return Path(argument), {"resolved_from": "argument", "spec_path": str(spec), "lane_path": str(lane)}
    spec_exists, lane_exists = spec.is_file(), lane.is_file()
    if spec_exists and lane_exists:
        spec_sha, lane_sha = file_sha256(spec), file_sha256(lane)
        if spec_sha != lane_sha:
            raise SystemExit(
                f"two 27B evidence manifests differ: spec path {spec} ({spec_sha[:16]}) vs lane copy "
                f"{lane} ({lane_sha[:16]}).  The K3 rows must correspond to the manifest the trunk "
                f"taps are run on; name it with --evidence."
            )
        return spec, {"resolved_from": "spec-path (lane copy identical)", "spec_path": str(spec), "lane_path": str(lane)}
    if spec_exists:
        return spec, {"resolved_from": "spec-path", "spec_path": str(spec), "lane_path": str(lane)}
    if lane_exists:
        return lane, {"resolved_from": "lane-copy (spec path absent)", "spec_path": str(spec), "lane_path": str(lane)}
    raise SystemExit(f"no evidence manifest at {spec} or {lane}; pass --evidence")


def sibling_build_receipt(evidence: Path) -> Path:
    return evidence.with_name(evidence.stem + "-build-receipt.json")


def _dig(payload: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def qwen_separator_from_sources(
    evidence: Path,
    qwen_dir: Path,
    *,
    manifest_sha256: str | None = None,
    override: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """The manifest packer's separator/pad id, from the authorities that exist.

    Sources, all recorded: the manifest's sibling build receipt
    (``packing.padding_token_id`` and ``tokenizer.separator_id``) and the
    parent's ``generation_config.json`` (``pad_token_id``).  Any disagreement is
    a refusal; no source at all is a refusal -- there is no literal default.
    ``--qwen-separator-id`` is the explicit override (recorded as such).
    """

    report: dict[str, Any] = {"build_receipt": None, "generation_config": None, "override": override}
    candidates: dict[str, int] = {}
    receipt_path = sibling_build_receipt(evidence)
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        entry: dict[str, Any] = {"path": str(receipt_path), "sha256": file_sha256(receipt_path)}
        declared_manifest = _dig(receipt, "manifest_sha256") or _dig(receipt, "manifest", "manifest_sha256") or _dig(receipt, "outputs", "manifest", "manifest_sha256")
        entry["manifest_sha256"] = declared_manifest
        if manifest_sha256 is not None and declared_manifest is not None and str(declared_manifest) != str(manifest_sha256):
            raise SystemExit(
                f"{receipt_path} describes manifest {str(declared_manifest)[:16]}, the manifest being built on "
                f"seals {manifest_sha256[:16]}; the sibling build receipt is not this manifest's"
            )
        for label, value in (
            ("packing.padding_token_id", _dig(receipt, "packing", "padding_token_id")),
            ("tokenizer.separator_id", _dig(receipt, "tokenizer", "separator_id")),
        ):
            if value is not None:
                entry[label] = int(value)
                candidates[f"build receipt {label}"] = int(value)
        report["build_receipt"] = entry
    generation = qwen_dir / "generation_config.json"
    if generation.is_file():
        payload = load_json(generation)
        value = payload.get("pad_token_id")
        report["generation_config"] = {"path": str(generation), "pad_token_id": value}
        if value is not None:
            candidates["generation_config pad_token_id"] = int(value)
    report["candidates"] = candidates
    if len(set(candidates.values())) > 1:
        raise SystemExit(f"the Qwen separator/pad id is ambiguous: {candidates}; pass --qwen-separator-id to choose")
    if override is not None:
        report["source"] = "--qwen-separator-id"
        if candidates and int(override) not in candidates.values():
            report["override_disagrees_with"] = candidates
        return int(override), report
    if not candidates:
        raise SystemExit(
            f"no authority for the Qwen separator id: neither {receipt_path} (packing.padding_token_id) "
            f"nor {generation} (pad_token_id) exists; pass --qwen-separator-id explicitly"
        )
    report["source"] = " == ".join(sorted(candidates))
    return next(iter(candidates.values())), report


# --------------------------------------------------------------------------
# Tokenizer sides.
# --------------------------------------------------------------------------


class QwenSide:
    """The Qwen parent's tokenizer as a BYTE spelling of every id in a row.

    A base token spells its own bytes (GPT-2 byte alphabet through
    ``JsonBpeTokenizer.token_bytes``).  The separator id (the parent's eos,
    248044 ``<|endoftext|>``, which the manifest packer also pads with) is a
    STRUCTURAL boundary, not text: it spells ``None``.  Every OTHER added/special
    token (``<|im_start|>``, ``<|im_end|>``, the tool-call markers the registry's
    tool rows carry) is document text -- it spells the literal bytes of its own
    name, so the K3 side reads the same markup as ordinary text.  That is the
    only rendering under which "same raw text under both tokenizers" is true
    for a row that contains chat markup; the bridge's vocabulary-level choice
    (specials unmapped) is the right one for an embedding table and the wrong
    one for a position-level corpus.
    """

    def __init__(self, chimera_tokenizer: Any, separator_id: int) -> None:
        self.chimera = chimera_tokenizer
        self.base = chimera_tokenizer.base
        self.separator_id = int(separator_id)
        self.pad_id = int(separator_id)  # the manifest packer pads with its separator
        self.special_ids = frozenset(int(i) for i in self.base.special_ids)
        self.vocab_size = int(self.base.vocab_size)
        self.name = str(self.base.name)

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.base.encode_ids(text)]

    def piece(self, token_id: int) -> str | None:
        value = self.base._tokenizer.id_to_token(int(token_id))
        return None if value is None else str(value)

    def token_bytes(self, token_id: int) -> bytes | None:
        token_id = int(token_id)
        if token_id == self.separator_id:
            return None
        if token_id in self.special_ids:
            piece = self.piece(token_id)
            if piece is None:
                raise SystemExit(f"qwen id {token_id} is a declared special with no piece")
            return piece.encode("utf-8")
        raw = self.base.token_bytes(token_id)
        if raw:
            return bytes(raw)
        piece = self.piece(token_id)
        if piece is None:
            raise SystemExit(
                f"qwen id {token_id} is not defined in tokenizer.json (a dead embedding "
                f"row); a stream carrying it was not written by this tokenizer"
            )
        return piece.encode("utf-8")


class K3Side:
    """K3 tiktoken: ordinary encode (no special matching), exact byte decode.

    ``separator_id`` (``[EOS]``) closes documents; ``pad_id`` (``[PAD]``) fills
    the tail of a row.  Both are reserved specials and both are recorded.
    """

    def __init__(self, chimera_tokenizer: Any, separator_id: int, pad_id: int) -> None:
        self.chimera = chimera_tokenizer
        self.base = chimera_tokenizer.base
        self.encoding = chimera_tokenizer.base._encoding
        self.separator_id = int(separator_id)
        self.pad_id = int(pad_id)
        self.special_ids = frozenset(int(i) for i in self.base.special_ids)
        self.vocab_size = int(self.base.vocab_size)
        self.name = str(self.base.name)
        for label, value in (("separator", self.separator_id), ("pad", self.pad_id)):
            if value not in self.special_ids:
                raise SystemExit(
                    f"K3 {label} id {value} is not a reserved special "
                    f"({min(self.special_ids)}..{max(self.special_ids)})"
                )

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.encoding.encode_ordinary(text)]

    def token_bytes(self, token_id: int) -> bytes | None:
        token_id = int(token_id)
        if token_id == self.separator_id:
            return None
        if token_id in self.special_ids:
            return b""
        return bytes(self.encoding.decode_single_token_bytes(token_id))

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.token_bytes(i) or b"" for i in ids)

    def decode_with_specials(self, ids: list[int]) -> bytes:
        """tiktoken's own decoder: specials spell their names (the native review rows)."""

        return bytes(self.encoding.decode_bytes([int(i) for i in ids]))


def k3_special_id(k3_dir: Path, name: str) -> int:
    """The id of a named K3 special from ``tokenizer_config.json``."""

    payload = load_json(k3_dir / "tokenizer_config.json")
    decoder = payload.get("added_tokens_decoder") or {}
    for raw_id, entry in decoder.items():
        if isinstance(entry, dict) and entry.get("content") == name:
            return int(raw_id)
    raise SystemExit(f"{k3_dir}/tokenizer_config.json does not declare a special named {name!r}")


def load_sides(
    *,
    qwen_dir: Path,
    k3_dir: Path,
    qwen_separator_id: int,
    k3_separator_id: int | None,
    k3_pad_id: int | None = None,
) -> tuple[QwenSide, K3Side, dict[str, Any]]:
    qwen_chimera = ctok.ChimeraTokenizer.from_qwen(qwen_dir)
    k3_chimera = ctok.ChimeraTokenizer.from_k3(k3_dir)
    k3_config = load_json(k3_dir / "tokenizer_config.json")
    eos_name = str(k3_config.get("eos_token") or "[EOS]")
    pad_name = str(k3_config.get("pad_token") or "[PAD]")
    separator = int(k3_separator_id) if k3_separator_id is not None else k3_special_id(k3_dir, eos_name)
    pad = int(k3_pad_id) if k3_pad_id is not None else k3_special_id(k3_dir, pad_name)
    qwen = QwenSide(qwen_chimera, qwen_separator_id)
    k3 = K3Side(k3_chimera, separator, pad)
    k3_names = {int(k): v.get("content") for k, v in (k3_config.get("added_tokens_decoder") or {}).items()}
    report = {
        "qwen": {
            "name": qwen.name,
            "tokenizer_json": str(qwen_dir / "tokenizer.json"),
            "tokenizer_json_sha256": file_sha256(qwen_dir / "tokenizer.json"),
            "defined_rows": qwen.vocab_size,
            "special_ids": len(qwen.special_ids),
            "separator_id": qwen.separator_id,
            "separator_piece": qwen.piece(qwen.separator_id),
            "pad_id": qwen.pad_id,
            "pad_note": "the manifest packer pads with its separator; real_lengths delimit the real tokens",
            "special_render": "literal name bytes (separator excepted: structural, zero bytes)",
        },
        "k3": {
            "name": k3.name,
            "tiktoken_model": str(k3_dir / "tiktoken.model"),
            "tiktoken_model_sha256": file_sha256(k3_dir / "tiktoken.model"),
            "tokenizer_config_sha256": file_sha256(k3_dir / "tokenizer_config.json"),
            "mergeable_ranks": int(k3.encoding.n_vocab) - ctok.K3_NUM_RESERVED_SPECIAL_TOKENS
            if hasattr(k3.encoding, "n_vocab")
            else None,
            "vocab_size": k3.vocab_size,
            "reserved_specials": ctok.K3_NUM_RESERVED_SPECIAL_TOKENS,
            "separator_id": k3.separator_id,
            "separator_name": k3_names.get(k3.separator_id),
            "pad_id": k3.pad_id,
            "pad_name": k3_names.get(k3.pad_id),
            "eos_token_named_by_tokenizer_config": eos_name,
            "pad_token_named_by_tokenizer_config": pad_name,
            "encode": "tiktoken encode_ordinary (no special matching; markup is text)",
        },
    }
    return qwen, k3, report


# --------------------------------------------------------------------------
# Row construction and position alignment.
# --------------------------------------------------------------------------


@dataclass
class Segment:
    """One document of a packed Qwen row: ``(global_index, id)`` tokens, and
    whether the packer closed it with the separator."""

    tokens: list[tuple[int, int]]
    terminated: bool


def qwen_segments(ids: list[int], separator_id: int) -> tuple[list[Segment], int]:
    """Split a Qwen row at the separator.  Returns ``(segments, padding_tokens)``.

    A separator that closes a nonempty run is that document's terminator; a
    separator that follows another separator is padding (the packer's eos fill).
    """

    segments: list[Segment] = []
    current: list[tuple[int, int]] = []
    padding = 0
    for index, token in enumerate(ids):
        token = int(token)
        if token == separator_id:
            if current:
                segments.append(Segment(current, True))
                current = []
            else:
                padding += 1
        else:
            current.append((index, token))
    if current:
        segments.append(Segment(current, False))
    return segments, padding


def utf8_complete_prefix(raw: bytes) -> tuple[str, int]:
    """Decode ``raw`` strictly, allowing ONLY a trailing incomplete character.

    Returns ``(text, dropped_trailing_bytes)``.  A packed document the manifest
    cut at ``sequence_length - 1`` ids can end on the first byte(s) of a
    multi-byte character; those bytes have no text and no K3 spelling, so they
    are dropped and COUNTED.  Any interior invalid byte raises
    ``UnicodeDecodeError`` (``build_row`` turns it into the named refusal): the
    registry rows are JSON strings, so it cannot happen on a clean stream, and a
    stream on which it happens is not this manifest.
    """

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    text = decoder.decode(raw, final=False)
    kept = len(text.encode("utf-8"))
    return text, len(raw) - kept


@dataclass
class RowBuild:
    k3_ids: list[int] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)
    raw: bytearray = field(default_factory=bytearray)
    qwen_real: int = 0
    qwen_padding: int = 0
    segments_total: int = 0
    segments_kept: int = 0
    separator_pairs: int = 0
    truncated: bool = False
    terminated: bool = False
    k3_tokens_dropped: int = 0
    raw_bytes_total: int = 0
    dropped_trailing_partial_utf8_bytes: int = 0

    @property
    def k3_real(self) -> int:
        return len(self.k3_ids)


def align_positions(
    qwen_offsets: list[tuple[int, int]],
    k3_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Position pairs at common byte boundaries.

    ``qwen_offsets`` / ``k3_offsets`` are ``(token_index, end_byte_offset)`` in
    token order.  Two positions pair when the bytes consumed through them end at
    the same offset.  When two tokens on one side end at the same offset (a
    zero-byte token) the LAST one carries the boundary, so the map stays a
    function; the intersection is walked in offset order, so both coordinates
    are strictly increasing by construction.

    This is ``importing.bridge.token_alignment`` taken from the vocabulary to the
    position: there a token is the bytes it stands for; here a position is the
    bytes consumed to reach it.
    """

    q_by_offset = {int(off): int(idx) for idx, off in qwen_offsets}
    k_by_offset = {int(off): int(idx) for idx, off in k3_offsets}
    common = sorted(set(q_by_offset) & set(k_by_offset))
    return [(q_by_offset[off], k_by_offset[off]) for off in common]


def build_row(
    qwen_ids: list[int],
    *,
    qwen: QwenSide,
    k3: K3Side,
    sequence_length: int,
    label: str = "row",
    require_terminated: bool = False,
) -> RowBuild:
    """One row: Qwen ids -> raw bytes per document -> K3 ids, pairs, facts.

    Truncation follows the manifest packer's own rule: a K3 row that would
    overflow ``sequence_length`` is cut ONCE to ``sequence_length - 1`` ids plus
    the separator (an unterminated final document -- a bank prompt -- is cut to
    ``sequence_length`` with no separator appended, and is normally skipped by the
    caller instead).  Pairs are only emitted over bytes the K3 side actually
    covers, and a separator is paired only when it closes the same document on
    both sides.

    ``require_terminated`` (evidence rows) refuses a row whose last document the
    packer did not close; an interior invalid UTF-8 byte is refused naming
    ``label`` and the byte offset in the row's raw stream.
    """

    build = RowBuild()
    segments, build.qwen_padding = qwen_segments(qwen_ids, qwen.separator_id)
    build.segments_total = len(segments)
    build.qwen_real = sum(len(seg.tokens) + int(seg.terminated) for seg in segments)
    if not segments:
        raise SystemExit(f"{label}: the row carries no document (all separator/padding)")
    build.terminated = bool(segments[-1].terminated)
    if require_terminated and not build.terminated:
        raise SystemExit(
            f"{label}: the last document is not closed by separator {qwen.separator_id}; the "
            f"evidence packer terminates every document, so this row was not packed by it"
        )
    row_offset = 0  # byte offset of the current document in the row's raw stream
    for seg_index, seg in enumerate(segments):
        if build.truncated:
            break
        raw = bytearray()
        q_offsets: list[tuple[int, int]] = []
        for global_index, token in seg.tokens:
            piece = qwen.token_bytes(token)
            if piece is None:
                raise SystemExit("separator inside a segment; qwen_segments is broken")
            raw += piece
            q_offsets.append((global_index, row_offset + len(raw)))
        build.raw_bytes_total += len(raw)
        try:
            text, dropped = utf8_complete_prefix(bytes(raw))
        except UnicodeDecodeError as exc:
            raise SystemExit(
                f"{label}: invalid UTF-8 inside document {seg_index} at row byte offset "
                f"{row_offset + exc.start} ({exc.reason}); the registry rows are JSON strings, "
                f"so this stream is not the sealed manifest's"
            ) from exc
        build.dropped_trailing_partial_utf8_bytes += dropped
        text_bytes = text.encode("utf-8")
        k3_ids = k3.encode(text)
        k3_pieces = [k3.token_bytes(i) or b"" for i in k3_ids]
        if b"".join(k3_pieces) != text_bytes:
            raise SystemExit(f"{label}: K3 tiktoken did not round-trip document {seg_index}'s bytes; refusing to write")

        start = len(build.k3_ids)
        room = sequence_length - start
        terminated = seg.terminated
        keep = len(k3_ids)
        if keep + int(terminated) > room:
            build.truncated = True
            if terminated:
                keep = max(room - 1, 0)
            else:
                keep = max(room, 0)
            build.k3_tokens_dropped += len(k3_ids) - keep
        kept_ids = k3_ids[:keep]
        kept_pieces = k3_pieces[:keep]
        k_offsets: list[tuple[int, int]] = []
        consumed = 0
        for local_index, piece in enumerate(kept_pieces):
            consumed += len(piece)
            k_offsets.append((start + local_index, row_offset + consumed))
        covered = consumed
        build.raw += text_bytes[:covered]
        build.k3_ids.extend(kept_ids)
        build.pairs.extend(align_positions(q_offsets, k_offsets))
        appended = keep > 0
        if terminated and build.k3_real < sequence_length:
            build.k3_ids.append(k3.separator_id)
            appended = True
            if keep == len(k3_ids) and dropped == 0 and seg.tokens:
                # Same document closed on both sides: the separators pair.
                build.pairs.append((seg.tokens[-1][0] + 1, len(build.k3_ids) - 1))
                build.separator_pairs += 1
        if appended:
            build.segments_kept += 1
        row_offset += len(raw)
    if build.k3_real > sequence_length:
        raise SystemExit(f"{label}: row overflowed sequence_length after truncation; build_row is broken")
    for (q0, k0), (q1, k1) in zip(build.pairs, build.pairs[1:]):
        if not (q1 > q0 and k1 > k0):
            raise SystemExit(f"{label}: boundary pairs are not monotone: ({q0},{k0}) -> ({q1},{k1})")
    return build


def pad(ids: list[int], sequence_length: int, pad_id: int) -> list[int]:
    if len(ids) > sequence_length:
        raise SystemExit(f"row of {len(ids)} ids exceeds sequence_length {sequence_length}")
    return list(ids) + [int(pad_id)] * (sequence_length - len(ids))


# --------------------------------------------------------------------------
# Bank rows.
# --------------------------------------------------------------------------


def render_plain(messages: list[dict[str, Any]]) -> str:
    """A tokenizer-neutral transcript: ``ROLE: content`` blocks and an assistant cue."""

    blocks = []
    for message in messages:
        role = str(message.get("role", "user")).strip().upper() or "USER"
        content = message.get("content")
        if not isinstance(content, str):
            raise SystemExit("plain render got a non-text content part")
        blocks.append(f"{role}: {content}")
    blocks.append("ASSISTANT:")
    return "\n\n".join(blocks)


class QwenChatRenderer:
    """The parent's own chat template with thinking OFF, through transformers."""

    def __init__(self, qwen_dir: Path) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - environment
            raise SystemExit(
                "--bank-render qwen-chat needs the transformers package for the chat "
                "template; pass --bank-render plain to render without it"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(str(qwen_dir))
        template = qwen_dir / "chat_template.jinja"
        self.template_sha256 = file_sha256(template) if template.is_file() else None

    def render(self, messages: list[dict[str, Any]]) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )


class K3NativeRenderer:
    """K3's OWN chat rendering: ``tokenization_kimi.TikTokenTokenizer`` from the
    K3 directory (the checkpoint's own module, imported from its file; no
    ``trust_remote_code`` machinery), ``apply_chat_template(thinking=False,
    add_generation_prompt=True)`` -- text AND ids, the ids carrying K3's own
    specials (whatever the checkpoint's ``encoding_k3`` emits: XTML ``<|open|>`` /
    ``<|sep|>`` / ``<|close|>`` / ``<|end_of_msg|>`` here; recorded per row).

    The ids are verified against the campaign's own K3 encoding
    (``TikTokenBpeTokenizer``): ``decode_bytes(ids)`` must equal the rendered
    text, so the two encodings agree on every rank and every special name.
    """

    def __init__(self, k3_dir: Path) -> None:
        module_file = k3_dir / "tokenization_kimi.py"
        if not module_file.is_file():
            raise SystemExit(f"{module_file} is missing: the K3-native render needs K3's own tokenizer module")
        if str(k3_dir) not in sys.path:
            sys.path.insert(0, str(k3_dir))
        try:
            module = importlib.import_module("tokenization_kimi")
        except ImportError as exc:
            raise SystemExit(
                f"cannot import tokenization_kimi from {k3_dir} ({exc}); pass --k3-bank-render none "
                f"to build without the K3-native review set (recorded)"
            ) from exc
        if Path(getattr(module, "__file__", "")).resolve() != module_file.resolve():
            raise SystemExit(f"tokenization_kimi resolved to {getattr(module, '__file__', None)}, not {module_file}")
        config = load_json(k3_dir / "tokenizer_config.json")
        decoder = config.get("added_tokens_decoder") or {}
        try:
            from tokenizers import AddedToken  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - environment
            raise SystemExit("the K3-native render needs the tokenizers package (tokenization_kimi imports it)") from exc
        # The checkpoint's own special-name table (the module's defaults name
        # every reserved id <|reserved_token_i|> without it).
        declared = {int(k): str(v.get("content")) for k, v in decoder.items() if isinstance(v, dict) and v.get("content")}
        added = {
            token_id: AddedToken(name, special=bool((decoder.get(str(token_id)) or {}).get("special", True)), normalized=False)
            for token_id, name in declared.items()
        }
        self.tokenizer = module.TikTokenTokenizer(
            str(k3_dir / "tiktoken.model"),
            bos_token=config.get("bos_token", "[BOS]"),
            eos_token=config.get("eos_token", "[EOS]"),
            unk_token=config.get("unk_token"),
            pad_token=config.get("pad_token"),
            added_tokens_decoder=added,
        )
        for token_id, name in declared.items():
            if self.tokenizer.special_tokens.get(name) != token_id:
                raise SystemExit(f"tokenization_kimi names special {token_id} {self.tokenizer.special_tokens.get(name)!r}, tokenizer_config says {name!r}")
        self.special_names = {token_id: name for name, token_id in self.tokenizer.special_tokens.items()}
        self.module_sha256 = file_sha256(module_file)
        self.encoding_module_sha256 = file_sha256(k3_dir / "encoding_k3.py") if (k3_dir / "encoding_k3.py").is_file() else None

    def render(self, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        text = str(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False))
        ids = [int(i) for i in self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, thinking=False)]
        return text, ids


def collect_bank_prompts(
    *,
    bank_v1: Path | None,
    bank_v2: Path | None,
    probe: Path | None,
    smoke: Path | None,
    sequence_length: int,
    include_overlong: str,
    calibration_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Every review prompt as ``{source, id, kind, messages|text, ...}`` plus the
    skipped ones by name and reason.  Long-context cases are judged by their
    declared ``prompt_tokens_target`` against the REVIEW sequence length BEFORE
    any haystack is materialised."""

    prompts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}

    if bank_v1 is not None:
        payload = load_json(bank_v1)
        inputs["bank_v1"] = {"path": str(bank_v1), "sha256": file_sha256(bank_v1), "schema": payload.get("schema"), "prompts": len(payload.get("prompts", []))}
        for index, entry in enumerate(payload.get("prompts", [])):
            prompts.append({"source": "bank_v1", "id": str(entry.get("id", index)), "kind": "chat", "messages": entry["messages"], "category": entry.get("category")})

    if bank_v2 is not None:
        payload = load_json(bank_v2)
        inputs["bank_v2"] = {"path": str(bank_v2), "sha256": file_sha256(bank_v2), "schema": payload.get("schema"), "prompts": len(payload.get("prompts", []))}
        for index, entry in enumerate(payload.get("prompts", [])):
            case_id = str(entry.get("id", index))
            if entry.get("modality_in", "text") != "text":
                skipped.append({"source": "bank_v2", "id": case_id, "reason": f"modality_in={entry.get('modality_in')!r} is not a text case"})
                continue
            if entry.get("modality_out", "text") != "text":
                skipped.append({"source": "bank_v2", "id": case_id, "reason": f"modality_out={entry.get('modality_out')!r}: the answer is a media artifact, not a text case"})
                continue
            if entry.get("runnable", True) is False:
                skipped.append({"source": "bank_v2", "id": case_id, "reason": "runnable:false"})
                continue
            target = int(entry.get("prompt_tokens_target") or 0)
            spec = entry.get("haystack_spec")
            if spec and target > sequence_length and include_overlong == "skip":
                # A long-context case's target IS its size (the haystack is
                # generated to it); judged here so the megabyte of filler is
                # never materialised.  A short case's 512 is a nominal budget:
                # it is judged on its actual encoded length below.
                skipped.append({"source": "bank_v2", "id": case_id, "reason": f"prompt_tokens_target {target} > review sequence_length {sequence_length}"})
                continue
            messages = [dict(m) for m in entry["messages"]]
            haystack_sha = None
            if spec:
                document = materialise_haystack_checked(spec, case_id, calibration_dir)
                haystack_sha = str(spec["haystack_sha256"])
                messages = [{**m, "content": str(m["content"]).replace("<<HAYSTACK>>", document)} for m in messages]
            for message in messages:
                if not isinstance(message.get("content"), str):
                    skipped.append({"source": "bank_v2", "id": case_id, "reason": "non-text content part"})
                    break
            else:
                prompts.append({"source": "bank_v2", "id": case_id, "kind": "chat", "messages": messages, "category": entry.get("category"), "lane": entry.get("lane"), "haystack_sha256_verified": haystack_sha, "prompt_tokens_target": target})

    if probe is not None:
        payload = load_json(probe)
        inputs["probe"] = {"path": str(probe), "sha256": file_sha256(probe), "schema": payload.get("schema"), "prompts": len(payload.get("prompts", []))}
        for index, text in enumerate(payload.get("prompts", [])):
            prompts.append({"source": "probe", "id": f"probe-{index}", "kind": "raw", "text": str(text)})

    if smoke is not None:
        payload = load_json(smoke)
        inputs["smoke"] = {"path": str(smoke), "sha256": file_sha256(smoke), "schema": payload.get("schema"), "prompts": len(payload.get("prompts", []))}
        for index, text in enumerate(payload.get("prompts", [])):
            prompts.append({"source": "smoke", "id": f"smoke-{index}", "kind": "chat", "messages": [{"role": "user", "content": str(text)}]})

    return prompts, skipped, inputs


def materialise_haystack_checked(spec: dict[str, Any], case_id: str, calibration_dir: Path | None = None) -> str:
    """``oc_bank_v2_draft.materialise_haystack`` with the bank's pinned digest checked."""

    directory = Path(calibration_dir or DEFAULT_CALIBRATION_DIR)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    try:
        from oc_bank_v2_draft import materialise_haystack  # noqa: E402
    except ImportError as exc:
        raise SystemExit(f"cannot import oc_bank_v2_draft.materialise_haystack from {directory}: {exc}") from exc
    document = materialise_haystack(spec)
    observed = canonical_sha256({"text": document})
    if observed != str(spec.get("haystack_sha256")):
        raise SystemExit(f"case {case_id}: regenerated haystack digests {observed}, the bank pinned {spec.get('haystack_sha256')}")
    return document


# --------------------------------------------------------------------------
# The build.
# --------------------------------------------------------------------------


def check_tokenizer_pins(
    manifest: dict[str, Any],
    *,
    qwen_dir: Path,
    k3_dir: Path,
    allow_tokenizer_mismatch: bool,
    allow_unpinned_tokenizer: bool,
    allow_unpinned_k3_tokenizer: bool,
) -> dict[str, Any]:
    """The ``qf_build.assert_evidence_tokenizer`` refusal, plus the two pins."""

    declared = str(manifest.get("tokenizer_sha256", ""))
    path = qwen_dir / "tokenizer.json"
    measured = file_sha256(path) if path.is_file() else None
    matches = measured is not None and declared == measured
    report: dict[str, Any] = {
        "manifest_tokenizer_sha256": declared,
        "parent_tokenizer_json": str(path) if path.is_file() else None,
        "parent_tokenizer_sha256": measured,
        "matches": matches,
        "allow_mismatch": bool(allow_tokenizer_mismatch),
        "pinned_sha256": EXPECTED_QWEN_TOKENIZER_SHA256,
        "pinned": measured == EXPECTED_QWEN_TOKENIZER_SHA256,
        "allow_unpinned": bool(allow_unpinned_tokenizer),
        "statement": (
            "the sealed stream was tokenized by the tokenizer this parent carries"
            if matches
            else "the sealed stream was NOT tokenized by this parent's tokenizer"
        ),
    }
    if not matches and not allow_tokenizer_mismatch:
        hint = ""
        if declared.startswith(QWEN_4B_TOKENIZER_SHA256_PREFIX):
            hint = "  That sha is the Qwen3.5 0.8B/4B tokenizer: this manifest is the 4B evidence, not the 27B corpus this stage builds."
        raise SystemExit(
            f"the evidence was tokenized by {declared or '(no sha recorded)'} and "
            f"{path} is {measured or '(missing)'}.  Every id in the manifest means a "
            f"different token to this parent, and nothing downstream would notice: "
            f"the seal verifies, the ids are in range, and the receipt would name the "
            f"manifest.{hint}  Build the evidence against this parent "
            f"(qf_evidence_build.py --parent {qwen_dir}) or pass --allow-tokenizer-mismatch "
            f"to have the exception recorded in the receipt."
        )
    if not report["pinned"] and not allow_unpinned_tokenizer:
        raise SystemExit(
            f"{path} hashes {measured}, the pinned 27B tokenizer is "
            f"{EXPECTED_QWEN_TOKENIZER_SHA256}.  This stage builds the 27B corpus; pass "
            f"--allow-unpinned-tokenizer to build against another parent (recorded)."
        )
    tik = k3_dir / "tiktoken.model"
    tik_sha = file_sha256(tik) if tik.is_file() else None
    report["k3_tiktoken_model_sha256"] = tik_sha
    report["k3_pinned_sha256"] = EXPECTED_K3_TIKTOKEN_SHA256
    report["k3_pinned"] = tik_sha == EXPECTED_K3_TIKTOKEN_SHA256
    report["k3_allow_unpinned"] = bool(allow_unpinned_k3_tokenizer)
    if tik_sha is None:
        raise SystemExit(f"no tiktoken.model under {k3_dir}")
    if not report["k3_pinned"] and not allow_unpinned_k3_tokenizer:
        raise SystemExit(
            f"{tik} hashes {tik_sha}, the pinned K3 rank file is {EXPECTED_K3_TIKTOKEN_SHA256}; "
            f"a swapped rank file changes every K3 id silently.  Pass "
            f"--allow-unpinned-k3-tokenizer to record the exception."
        )
    return report


def guard_share_reads(paths: dict[str, Path | None], *, allow_share_read: bool) -> dict[str, Any]:
    """Every input path this tool may open, guarded and recorded by role."""

    guarded = {role: str(path) for role, path in paths.items() if path is not None}
    on_share = [f"{role}={path}" for role, path in guarded.items() if under_share(path)]
    if on_share and not allow_share_read:
        raise SystemExit(
            f"share reads are OFF and these inputs live under {SHARE_ROOT}: {on_share}.  "
            f"Point --k3-dir at the Lexar copy ({DEFAULT_K3_DIR}) or pass --allow-share-read "
            f"(small files only; this tool never opens a shard)."
        )
    return {"share_root": SHARE_ROOT, "allow_share_read": bool(allow_share_read), "guarded_inputs": guarded, "inputs_on_share": on_share}


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, body: dict[str, Any], seal_field: str | None = None) -> str:
    if seal_field:
        body = {k: v for k, v in body.items() if k != seal_field}
        body[seal_field] = canonical_sha256(body)
    atomic_write_bytes(path, json.dumps(body, indent=2, ensure_ascii=False).encode("utf-8"))
    return body[seal_field] if seal_field else file_sha256(path)


def in_band_tensors(rows: list[list[int]], splits: list[str], real_lengths: list[int]) -> dict[str, np.ndarray]:
    """``input_ids`` plus the split carried IN-BAND: ``split`` int8 [rows]
    (0 fit / 1 held / 2 review), ``real_lengths`` int64 [rows], ``fit_mask`` /
    ``held_mask`` bool [rows, T] over the REAL tokens of fit / held rows only.
    A consumer that flattens ``input_ids`` and deals positions by index sees
    exactly which positions may enter a solve and which may not."""

    array = np.asarray(rows, dtype=np.int64)
    if array.ndim != 2:
        raise SystemExit("ids must be a rectangular [rows, sequence_length] table")
    if not (len(splits) == len(real_lengths) == array.shape[0]):
        raise SystemExit("split / real_lengths do not match the rows")
    split = np.asarray([SPLIT_CODES[s] for s in splits], dtype=np.int8)
    lengths = np.asarray(real_lengths, dtype=np.int64)
    if (lengths < 0).any() or (lengths > array.shape[1]).any():
        raise SystemExit("real_lengths outside [0, sequence_length]")
    position = np.arange(array.shape[1], dtype=np.int64)[None, :]
    real = position < lengths[:, None]
    fit_mask = real & (split == SPLIT_CODES["fit"])[:, None]
    held_mask = real & (split == SPLIT_CODES["held"])[:, None]
    return {"input_ids": array, "split": split, "real_lengths": lengths, "fit_mask": fit_mask, "held_mask": held_mask}


def content_sha256(array: np.ndarray) -> dict[str, str]:
    """Digests of the ids THEMSELVES, independent of the safetensors header
    (whose metadata map is serialized in hash order and so differs between two
    byte-identical builds): ``int64`` over the raw row-major bytes, ``fp32`` over
    the float32 cast -- the lane's ``k3_common.sha256_tensor`` convention, so the
    S2 receipt's ``input_ids_sha256`` can be checked against this receipt."""

    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return {
        "int64": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "fp32": hashlib.sha256(contiguous.astype(np.float32).tobytes()).hexdigest(),
    }


def write_ids(path: Path, tensors: dict[str, np.ndarray], *, metadata: dict[str, str]) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


@dataclass
class IdsTable:
    """One ids file (or a paired K3/Qwen pair of files) in the making."""

    name: str
    sequence_length: int
    paired: bool
    k3_rows: list[list[int]] = field(default_factory=list)
    qwen_rows: list[list[int]] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)
    maps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.k3_rows)

    def positions(self) -> int:
        return self.rows * self.sequence_length

    def real_k3_tokens(self) -> int:
        return sum(int(m["k3_tokens_real"]) for m in self.meta)

    def pricing(self) -> dict[str, Any]:
        positions = self.positions()
        real = self.real_k3_tokens()
        return {
            "rows": self.rows,
            "sequence_length": self.sequence_length,
            "positions": positions,
            "k3_tokens_real": real,
            "k3_padding_tokens": positions - real,
            "k3_padding_fraction": ((positions - real) / positions) if positions else 0.0,
            "rows_by_split": dict(sorted({s: sum(1 for m in self.meta if m["split"] == s) for s in {m["split"] for m in self.meta}}.items())),
        }


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    evidence, evidence_resolution = resolve_evidence(args.evidence)
    qwen_dir = Path(args.qwen_dir)
    k3_dir = Path(args.k3_dir)
    out_dir = Path(args.out_dir)
    calibration_dir = Path(args.calibration_dir)
    review_sequence_length = int(args.review_sequence_length)
    if review_sequence_length < 1:
        raise SystemExit("--review-sequence-length must be positive")
    # Share reads are OFF unless asked for, or unless --k3-dir itself points at
    # the share (then only its small tokenizer files are read) and --no-share-read
    # did not veto that.  EVERY path this tool may open is guarded.
    allow_share = bool(args.allow_share_read) or (under_share(k3_dir) and not args.no_share_read)
    guarded: dict[str, Path | None] = {
        "evidence": evidence,
        "evidence_build_receipt": sibling_build_receipt(evidence) if sibling_build_receipt(evidence).is_file() else None,
        "qwen_dir": qwen_dir,
        "k3_dir": k3_dir,
        "qwen_fold_dir": Path(args.qwen_fold_dir),
        "chimera_src": resolve_chimera_src(args.chimera_src),
        "calibration_dir": calibration_dir,
    }
    if args.append_bank_rows:
        for role in ("bank_v1", "bank_v2", "probe", "smoke"):
            value = getattr(args, role)
            guarded[role] = Path(value) if value else None
    share = guard_share_reads(guarded, allow_share_read=allow_share)  # before any lane import opens a path
    bind_lanes(qwen_fold_dir=args.qwen_fold_dir, chimera_src=args.chimera_src)
    for name in ("tiktoken.model", "tokenizer_config.json", "config.json"):
        if not (k3_dir / name).is_file():
            raise SystemExit(f"{k3_dir}/{name} is missing: copy the small K3 tokenizer files first")

    outputs: dict[str, Path] = {
        FILE_EVIDENCE: out_dir / f"{FILE_EVIDENCE}.safetensors",
        FILE_EVIDENCE_QWEN: out_dir / f"{FILE_EVIDENCE_QWEN}.safetensors",
        "row_meta": out_dir / "row_meta.json",
        "boundary_map": out_dir / "boundary_map.json",
        "receipt": out_dir / "k3_corpus_build_receipt.json",
    }
    if args.append_bank_rows:
        outputs[FILE_REVIEW] = out_dir / f"{FILE_REVIEW}.safetensors"
        outputs[FILE_REVIEW_QWEN] = out_dir / f"{FILE_REVIEW_QWEN}.safetensors"
        if args.k3_bank_render == "k3-native":
            outputs[FILE_REVIEW_NATIVE] = out_dir / f"{FILE_REVIEW_NATIVE}.safetensors"
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"outputs exist and --force was not given: {existing}")

    manifest = qf_corpus.read_manifest(evidence)
    sequence_length = int(manifest["sequence_length"])
    manifest_file_sha = file_sha256(evidence)
    log(f"manifest {evidence} ({evidence_resolution['resolved_from']}): {len(manifest['rows'])} rows x {sequence_length}, seal {manifest['manifest_sha256'][:16]}, tokenizer {manifest['tokenizer_sha256'][:16]}")
    tokenizer_check = check_tokenizer_pins(
        manifest,
        qwen_dir=qwen_dir,
        k3_dir=k3_dir,
        allow_tokenizer_mismatch=args.allow_tokenizer_mismatch,
        allow_unpinned_tokenizer=args.allow_unpinned_tokenizer,
        allow_unpinned_k3_tokenizer=args.allow_unpinned_k3_tokenizer,
    )
    log(f"tokenizer check: manifest vs parent {'match' if tokenizer_check['matches'] else 'MISMATCH (allowed)'}; 27B pin {'ok' if tokenizer_check['pinned'] else 'UNPINNED (allowed)'}; K3 pin {'ok' if tokenizer_check['k3_pinned'] else 'UNPINNED (allowed)'}")

    qwen_separator, separator_report = qwen_separator_from_sources(
        evidence, qwen_dir, manifest_sha256=str(manifest["manifest_sha256"]), override=args.qwen_separator_id
    )
    qwen, k3, sides = load_sides(
        qwen_dir=qwen_dir, k3_dir=k3_dir, qwen_separator_id=qwen_separator,
        k3_separator_id=args.k3_separator_id, k3_pad_id=args.k3_pad_id,
    )
    sides["qwen"]["separator_sources"] = separator_report
    log(f"qwen {qwen.name}: {qwen.vocab_size} defined ids, separator {qwen.separator_id} {sides['qwen']['separator_piece']} ({separator_report['source']}); k3 {k3.name}: {k3.vocab_size} ids, separator {k3.separator_id} {sides['k3']['separator_name']}, pad {k3.pad_id} {sides['k3']['pad_name']}")

    evidence_table = IdsTable(FILE_EVIDENCE, sequence_length, paired=True)
    review_table = IdsTable(FILE_REVIEW, review_sequence_length, paired=True)
    native_table = IdsTable(FILE_REVIEW_NATIVE, review_sequence_length, paired=False)

    def check_k3_ids(label: str, ids: list[int]) -> None:
        for token in ids:
            if not 0 <= token < k3.vocab_size:
                raise SystemExit(f"{label}: K3 id {token} outside {k3.vocab_size}")

    def commit(table: IdsTable, build: RowBuild, qwen_ids_real: list[int], meta: dict[str, Any]) -> None:
        index = table.rows
        label = f"{table.name}[{index}]"
        raw_bytes = bytes(build.raw)
        if k3.decode_bytes(build.k3_ids) != raw_bytes:
            raise SystemExit(f"{label}: K3 detokenize != raw bytes; refusing to write")
        check_k3_ids(label, build.k3_ids)
        table.k3_rows.append(pad(build.k3_ids, table.sequence_length, k3.pad_id))
        table.qwen_rows.append(pad(qwen_ids_real, table.sequence_length, qwen.pad_id))
        pairs = build.pairs
        q_real = build.qwen_real
        k_real = build.k3_real
        table.meta.append(
            {
                "file": table.name,
                "row_index": index,
                **meta,
                "terminated": build.terminated,
                "qwen_tokens_real": q_real,
                "qwen_padding_tokens": table.sequence_length - q_real,
                "k3_tokens_real": k_real,
                "k3_padding_tokens": table.sequence_length - k_real,
                "k3_over_qwen_token_ratio": (k_real / q_real) if q_real else None,
                "segments": build.segments_total,
                "segments_kept": build.segments_kept,
                "k3_truncated": build.truncated,
                "k3_tokens_dropped": build.k3_tokens_dropped,
                "raw_bytes_total": build.raw_bytes_total,
                "k3_covered_bytes": len(raw_bytes),
                "dropped_trailing_partial_utf8_bytes": build.dropped_trailing_partial_utf8_bytes,
                "raw_text_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "k3_round_trip_exact": True,
            }
        )
        table.maps.append(
            {
                "file": table.name,
                "row_index": index,
                "source": meta["source"],
                "split": meta["split"],
                "qwen_tokens_real": q_real,
                "k3_tokens_real": k_real,
                "aligned_pairs": len(pairs),
                "separator_pairs": build.separator_pairs,
                "aligned_fraction_qwen": (len(pairs) / q_real) if q_real else 0.0,
                "aligned_fraction_k3": (len(pairs) / k_real) if k_real else 0.0,
                "aligned_fraction": (len(pairs) / max(q_real, k_real)) if max(q_real, k_real) else 0.0,
                "pairs": [[int(q), int(k)] for q, k in pairs],
            }
        )

    def commit_native(table: IdsTable, k3_ids: list[int], text: str, meta: dict[str, Any]) -> None:
        index = table.rows
        label = f"{table.name}[{index}]"
        text_bytes = text.encode("utf-8")
        if k3.decode_with_specials(k3_ids) != text_bytes:
            raise SystemExit(f"{label}: K3-native ids do not decode to the rendered text; refusing to write")
        check_k3_ids(label, k3_ids)
        table.k3_rows.append(pad(k3_ids, table.sequence_length, k3.pad_id))
        specials = [i for i in k3_ids if i in k3.special_ids]
        table.meta.append(
            {
                "file": table.name,
                "row_index": index,
                **meta,
                "terminated": False,
                "qwen_tokens_real": None,
                "k3_tokens_real": len(k3_ids),
                "k3_padding_tokens": table.sequence_length - len(k3_ids),
                "k3_special_tokens": len(specials),
                "k3_special_ids_present": sorted(set(specials)),
                "k3_truncated": False,
                "k3_tokens_dropped": 0,
                "raw_bytes_total": len(text_bytes),
                "k3_covered_bytes": len(text_bytes),
                "dropped_trailing_partial_utf8_bytes": 0,
                "raw_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "k3_round_trip_exact": True,
            }
        )

    # -- evidence rows ----------------------------------------------------
    for row in sorted(manifest["rows"], key=lambda r: int(r["batch_index"])):
        ids = [int(i) for i in row["ids"]]
        batch_index = int(row["batch_index"])
        build = build_row(ids, qwen=qwen, k3=k3, sequence_length=sequence_length, label=f"evidence batch_index {batch_index}", require_terminated=True)
        commit(
            evidence_table,
            build,
            ids,  # already sequence_length long, padded by the packer with the separator
            {
                "source": "evidence",
                "origin": {"manifest": evidence.name, "batch_index": batch_index},
                "split": str(row["split"]),
                "is_review_prompt": False,
                "masked": False,
            },
        )
    log(f"evidence: {evidence_table.rows} rows x {sequence_length}; K3 real tokens {evidence_table.real_k3_tokens()} vs Qwen {sum(m['qwen_tokens_real'] for m in evidence_table.meta)}; truncated {sum(m['k3_truncated'] for m in evidence_table.meta)}")

    # -- review rows ------------------------------------------------------
    bank_inputs: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    native_skipped: list[dict[str, Any]] = []
    render_report: dict[str, Any] = {"mode": None}
    native_report: dict[str, Any] = {"mode": None}
    if args.append_bank_rows:
        prompts, skipped, bank_inputs = collect_bank_prompts(
            bank_v1=Path(args.bank_v1) if args.bank_v1 else None,
            bank_v2=Path(args.bank_v2) if args.bank_v2 else None,
            probe=Path(args.probe) if args.probe else None,
            smoke=Path(args.smoke) if args.smoke else None,
            sequence_length=review_sequence_length,
            include_overlong=args.include_overlong,
            calibration_dir=calibration_dir,
        )
        renderer: QwenChatRenderer | None = None
        render_report = {"mode": args.bank_render}
        if args.bank_render == "qwen-chat":
            renderer = QwenChatRenderer(qwen_dir)
            render_report["chat_template_sha256"] = renderer.template_sha256
            render_report["enable_thinking"] = False
            render_report["add_generation_prompt"] = True
        for prompt in prompts:
            if prompt["kind"] == "raw":
                text = str(prompt["text"])
                render = "raw"
            elif renderer is not None:
                text = renderer.render(prompt["messages"])
                render = "qwen-chat"
            else:
                text = render_plain(prompt["messages"])
                render = "plain"
            qwen_ids = qwen.encode(text)
            if qwen.separator_id in qwen_ids:
                skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": "rendered text contains the separator token"})
                continue
            if len(qwen_ids) > review_sequence_length:
                if args.include_overlong == "skip":
                    skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": f"{len(qwen_ids)} qwen tokens > review sequence_length {review_sequence_length}"})
                    continue
                qwen_ids = qwen_ids[:review_sequence_length]
            label = f"review {prompt['source']}/{prompt['id']}"
            build = build_row(qwen_ids, qwen=qwen, k3=k3, sequence_length=review_sequence_length, label=label)
            if build.truncated and args.include_overlong == "skip":
                skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": f"{build.k3_real + build.k3_tokens_dropped} k3 tokens > review sequence_length {review_sequence_length}"})
                continue
            meta = {
                "source": prompt["source"],
                "origin": {k: v for k, v in prompt.items() if k in ("id", "category", "lane", "haystack_sha256_verified", "prompt_tokens_target")},
                "split": "review",
                "is_review_prompt": True,
                "masked": True,
                "render": render,
                "k3_review_render": K3_REVIEW_RENDER_STATEMENTS[render],
                "rendered_text": text,
                "rendered_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "qwen_ids": qwen_ids,
            }
            commit(review_table, build, qwen_ids, meta)
        log(f"review rows: {review_table.rows} x {review_sequence_length} appended ({args.bank_render} render, spelled as literal text on the K3 side), {len(skipped)} skipped by name")

        native_report = {"mode": args.k3_bank_render}
        if args.k3_bank_render == "k3-native":
            native = K3NativeRenderer(k3_dir)
            native_report.update({
                "renderer": "tokenization_kimi.TikTokenTokenizer.apply_chat_template",
                "thinking": False,
                "add_generation_prompt": True,
                "tokenization_kimi_sha256": native.module_sha256,
                "encoding_k3_sha256": native.encoding_module_sha256,
                "statement": K3_REVIEW_RENDER_STATEMENTS["k3-native"],
                "raw_prompts": "not rendered here: a raw probe prompt has no chat markup and is identical to its paired review row",
            })
            for prompt in prompts:
                if prompt["kind"] != "chat":
                    native_skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": "raw prompt: no chat markup, identical to the paired review row"})
                    continue
                if any(s["source"] == prompt["source"] and s["id"] == prompt["id"] for s in skipped):
                    native_skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": "skipped in the paired review set"})
                    continue
                text, k3_ids = native.render(prompt["messages"])
                if len(k3_ids) > review_sequence_length:
                    if args.include_overlong == "skip":
                        native_skipped.append({"source": prompt["source"], "id": prompt["id"], "reason": f"{len(k3_ids)} k3 tokens > review sequence_length {review_sequence_length}"})
                        continue
                    k3_ids = k3_ids[:review_sequence_length]
                    text = k3.decode_with_specials(k3_ids).decode("utf-8", errors="strict")
                commit_native(
                    native_table,
                    k3_ids,
                    text,
                    {
                        "source": prompt["source"],
                        "origin": {k: v for k, v in prompt.items() if k in ("id", "category", "lane", "haystack_sha256_verified", "prompt_tokens_target")},
                        "split": "review",
                        "is_review_prompt": True,
                        "masked": True,
                        "render": "k3-native",
                        "k3_review_render": K3_REVIEW_RENDER_STATEMENTS["k3-native"],
                        "rendered_text": text,
                        "rendered_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    },
                )
            log(f"K3-native review rows: {native_table.rows} x {review_sequence_length} (specials {sorted({i for m in native_table.meta for i in m['k3_special_ids_present']})}), {len(native_skipped)} not rendered")
        else:
            native_report["statement"] = K3_REVIEW_RENDER_STATEMENTS["none"]
            log("K3-native review set: NONE (--k3-bank-render none); the S2 bank-row logits are NOT a K3-native baseline")

    tables = [t for t in (evidence_table, review_table, native_table) if t.rows]
    all_meta = [m for t in tables for m in t.meta]
    all_maps = [m for t in tables for m in t.maps]

    # -- aggregates -------------------------------------------------------
    by_split: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for meta in all_meta:
        by_split[meta["split"]] = by_split.get(meta["split"], 0) + 1
        by_source[meta["source"]] = by_source.get(meta["source"], 0) + 1
    if any(m["is_review_prompt"] and m["split"] != "review" for m in all_meta):
        raise SystemExit("a review prompt landed outside the review split; refusing to write")
    if any(m["split"] == "review" for m in evidence_table.meta) or any(m["split"] != "review" for m in review_table.meta + native_table.meta):
        raise SystemExit("splits crossed files; refusing to write")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = sum(r["aligned_pairs"] for r in rows)
        q = sum(r["qwen_tokens_real"] for r in rows)
        k = sum(r["k3_tokens_real"] for r in rows)
        return {
            "rows": len(rows),
            "aligned_pairs": pairs,
            "qwen_tokens_real": q,
            "k3_tokens_real": k,
            "aligned_fraction_qwen": (pairs / q) if q else 0.0,
            "aligned_fraction_k3": (pairs / k) if k else 0.0,
            "aligned_fraction": (pairs / max(q, k)) if max(q, k) else 0.0,
            "min_row_aligned_fraction": min((r["aligned_fraction"] for r in rows), default=0.0),
        }

    boundary_aggregate = {
        "all": aggregate(all_maps),
        "evidence": aggregate([r for r in all_maps if r["source"] == "evidence"]),
        "evidence_fit": aggregate([r for r in all_maps if r["source"] == "evidence" and r["split"] == "fit"]),
        "evidence_held": aggregate([r for r in all_maps if r["source"] == "evidence" and r["split"] == "held"]),
        "review": aggregate([r for r in all_maps if r["split"] == "review"]),
    }
    k3_real = [m["k3_tokens_real"] for m in all_meta]
    fit_tokens = sum(m["k3_tokens_real"] for m in evidence_table.meta if m["split"] == "fit")
    held_tokens = sum(m["k3_tokens_real"] for m in evidence_table.meta if m["split"] == "held")
    counts = {
        "rows": len(all_meta),
        "sequence_length": sequence_length,
        "review_sequence_length": review_sequence_length if args.append_bank_rows else None,
        "rows_by_split": dict(sorted(by_split.items())),
        "rows_by_source": dict(sorted(by_source.items())),
        "rows_by_file": {t.name: t.rows for t in tables},
        "evidence_rows": evidence_table.rows,
        "review_rows": review_table.rows,
        "review_native_rows": native_table.rows,
        "skipped_review_prompts": len(skipped),
        "native_not_rendered": len(native_skipped),
        "k3_tokens_real_total": sum(k3_real),
        "k3_tokens_real_min": min(k3_real) if k3_real else 0,
        "k3_tokens_real_max": max(k3_real) if k3_real else 0,
        "k3_fit_tokens": fit_tokens,
        "k3_held_tokens": held_tokens,
        "qwen_tokens_real_total": sum(m["qwen_tokens_real"] or 0 for m in all_meta),
        "k3_rows_truncated": sum(int(m["k3_truncated"]) for m in all_meta),
        "k3_tokens_dropped_total": sum(m["k3_tokens_dropped"] for m in all_meta),
        "dropped_trailing_partial_utf8_bytes_total": sum(m["dropped_trailing_partial_utf8_bytes"] for m in all_meta),
        "k3_padding_tokens_total": sum(m["k3_padding_tokens"] for m in all_meta),
        "k3_padding_fraction": sum(m["k3_padding_tokens"] for m in all_meta) / float(sum(t.positions() for t in tables)) if tables else 0.0,
        "review_rows_in_fit_split": sum(int(m["is_review_prompt"] and m["split"] == "fit") for m in all_meta),
        "review_rows_in_evidence_file": sum(int(m["split"] == "review") for m in evidence_table.meta),
    }
    pricing = {
        "note": (
            "S2 is priced per POSITION of each file it streams; the run book's single T=65,536 is replaced "
            "by these: the evidence file alone is what the folded-prefix fits see, the review files are "
            "one short extra batch each (logits only, no solve)."
        ),
        "files": {t.name: t.pricing() for t in tables},
        "positions_total": sum(t.positions() for t in tables),
        "evidence_positions": evidence_table.positions(),
        "review_positions": review_table.positions() + native_table.positions(),
    }

    # -- write --------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    common_meta = {
        "schema": SCHEMA_IDS,
        "evidence_manifest_sha256": str(manifest["manifest_sha256"]),
        "in_band_tensors": ",".join(IN_BAND_TENSORS),
        "split_codes": json.dumps(SPLIT_CODES, sort_keys=True),
    }
    k3_meta = {"tokenizer": "kimi_k3 tiktoken", "tiktoken_model_sha256": sides["k3"]["tiktoken_model_sha256"], "separator_id": str(k3.separator_id), "pad_id": str(k3.pad_id)}
    qwen_meta = {"tokenizer": "qwen3_5 tokenizer.json", "tokenizer_json_sha256": sides["qwen"]["tokenizer_json_sha256"], "separator_id": str(qwen.separator_id), "pad_id": str(qwen.pad_id)}
    written: dict[str, dict[str, Any]] = {}

    def emit(name: str, table: IdsTable, side: str) -> None:
        rows = table.k3_rows if side == "k3" else table.qwen_rows
        lengths = [int(m["k3_tokens_real"] if side == "k3" else m["qwen_tokens_real"]) for m in table.meta]
        tensors = in_band_tensors(rows, [m["split"] for m in table.meta], lengths)
        metadata = {**common_meta, **(k3_meta if side == "k3" else qwen_meta), "file": name, "rows": str(table.rows), "sequence_length": str(table.sequence_length), "role": "evidence" if table is evidence_table else "review (masked)"}
        write_ids(outputs[name], tensors, metadata=metadata)
        written[name] = {
            "path": str(outputs[name]),
            "sha256": file_sha256(outputs[name]),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
            "input_ids_sha256": content_sha256(tensors["input_ids"]),
            "fit_mask_sha256": hashlib.sha256(np.ascontiguousarray(tensors["fit_mask"]).tobytes()).hexdigest(),
            "shape": [table.rows, table.sequence_length],
            "dtype": "int64",
            "separator_id": k3.separator_id if side == "k3" else qwen.separator_id,
            "pad_id": k3.pad_id if side == "k3" else qwen.pad_id,
            "fit_mask_true": int(tensors["fit_mask"].sum()),
            "held_mask_true": int(tensors["held_mask"].sum()),
            "role": metadata["role"],
        }

    emit(FILE_EVIDENCE, evidence_table, "k3")
    emit(FILE_EVIDENCE_QWEN, evidence_table, "qwen")
    if args.append_bank_rows:
        emit(FILE_REVIEW, review_table, "k3")
        emit(FILE_REVIEW_QWEN, review_table, "qwen")
        if args.k3_bank_render == "k3-native":
            emit(FILE_REVIEW_NATIVE, native_table, "k3")
    if written[FILE_EVIDENCE]["fit_mask_true"] != fit_tokens or written[FILE_EVIDENCE]["held_mask_true"] != held_tokens:
        raise SystemExit("in-band fit/held masks do not count the fit/held real tokens; refusing to seal")

    evidence_block = {
        "manifest": str(evidence),
        "resolution": evidence_resolution,
        "file_sha256": manifest_file_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "source": manifest["source"],
        "batches": len(manifest["rows"]),
        "build_receipt": separator_report.get("build_receipt"),
    }
    k3_review_render = {
        "paired_review_rows": K3_REVIEW_RENDER_STATEMENTS[args.bank_render] if args.append_bank_rows else None,
        "native_review_rows": native_report.get("statement") if args.append_bank_rows else None,
        "s2_bank_row_logits_are_k3_native_baseline": bool(args.append_bank_rows and args.k3_bank_render == "k3-native"),
    }
    row_meta = {
        "schema": SCHEMA_ROW_META,
        "sequence_length": sequence_length,
        "review_sequence_length": review_sequence_length if args.append_bank_rows else None,
        "rows": len(all_meta),
        "files": {name: {"rows": written[name]["shape"][0], "sequence_length": written[name]["shape"][1], "role": written[name]["role"]} for name in written},
        "in_band_tensors": list(IN_BAND_TENSORS),
        "split_codes": SPLIT_CODES,
        "evidence": evidence_block,
        "tokenizers": sides,
        "tokenizer_check": tokenizer_check,
        "bank_inputs": bank_inputs,
        "bank_render": render_report,
        "k3_bank_render": native_report,
        "k3_review_render": k3_review_render,
        "include_overlong": args.include_overlong,
        "skipped_review_prompts": skipped,
        "native_not_rendered": native_skipped,
        "counts": counts,
        "pricing": pricing,
        "row_list": all_meta,
    }
    row_meta_sha = write_json(outputs["row_meta"], row_meta, seal_field="row_meta_sha256")
    boundary = {
        "schema": SCHEMA_BOUNDARY_MAP,
        "definition": (
            "pairs are [qwen_position, k3_position]: token indices into the same row (file, row_index) of "
            "the qwen / k3 ids files whose consumed bytes end at the same offset of the row's raw byte "
            "stream (document separators paired explicitly when they close the same document on both "
            "sides).  A pair is where a hidden state on one side has read exactly the bytes the other "
            "side's has; aligned_fraction = pairs / max(real tokens) is the ceiling on the rows a "
            "cross-width chart can pair.  Only paired files have rows here; the K3-native review file "
            "has no Qwen side and is never charted."
        ),
        "sequence_length": sequence_length,
        "review_sequence_length": review_sequence_length if args.append_bank_rows else None,
        "evidence_manifest_sha256": str(manifest["manifest_sha256"]),
        "evidence_manifest_file_sha256": manifest_file_sha,
        "evidence_manifest_path": str(evidence),
        "qwen_tokenizer_sha256": sides["qwen"]["tokenizer_json_sha256"],
        "k3_tiktoken_model_sha256": sides["k3"]["tiktoken_model_sha256"],
        "ids_files_sha256": {name: written[name]["sha256"] for name in written},
        "ids_content_sha256": {name: written[name]["input_ids_sha256"] for name in written},
        "aggregate": boundary_aggregate,
        "rows": all_maps,
    }
    boundary_sha = write_json(outputs["boundary_map"], boundary, seal_field="boundary_map_sha256")

    receipt = {
        "schema": SCHEMA_RECEIPT,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "builder": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "share_reads": share,
        "evidence": evidence_block,
        "tokenizers": sides,
        "tokenizer_check": tokenizer_check,
        "bank_inputs": bank_inputs,
        "bank_render": render_report,
        "k3_bank_render": native_report,
        "k3_review_render": k3_review_render,
        "skipped_review_prompts": skipped,
        "native_not_rendered": native_skipped,
        "counts": counts,
        "pricing": pricing,
        "boundary_aggregate": boundary_aggregate,
        "in_band_tensors": list(IN_BAND_TENSORS),
        "split_codes": SPLIT_CODES,
        "outputs": {
            **written,
            "row_meta": {"path": str(outputs["row_meta"]), "sha256": file_sha256(outputs["row_meta"]), "row_meta_sha256": row_meta_sha},
            "boundary_map": {"path": str(outputs["boundary_map"]), "sha256": file_sha256(outputs["boundary_map"]), "boundary_map_sha256": boundary_sha},
        },
        "wall_seconds": round(time.time() - started, 3),
    }
    receipt_sha = write_json(outputs["receipt"], receipt, seal_field="receipt_sha256")
    receipt["receipt_sha256"] = receipt_sha
    log(
        f"wrote {', '.join(f'{t.name} {t.rows}x{t.sequence_length}' for t in tables)} -> {out_dir}; splits {counts['rows_by_split']}; "
        f"positions {pricing['positions_total']} (evidence {pricing['evidence_positions']}, review {pricing['review_positions']}); "
        f"aligned fraction all {boundary_aggregate['all']['aligned_fraction']:.4f} "
        f"(evidence {boundary_aggregate['evidence']['aligned_fraction']:.4f}); receipt {receipt_sha[:16]}"
    )
    return receipt


K3_REVIEW_RENDER_STATEMENTS = {
    "qwen-chat": "qwen-chat markup as literal text (not K3-native): the parent's chat template spelled byte-for-byte on the K3 side with encode_ordinary; K3's own specials never appear",
    "plain": "plain ROLE: transcript as literal text (not K3-native)",
    "raw": "raw prompt text, no chat markup on either side",
    "k3-native": "K3-native: tokenization_kimi.TikTokenTokenizer.apply_chat_template(thinking=False, add_generation_prompt=True) with K3's own specials; the S2 logits on these rows are a K3-native baseline",
    "none": "NO K3-native review set was built (--k3-bank-render none): the S2 bank-row logits are NOT a K3-native baseline",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", default=None, help=f"sealed neural-forge-omni-evidence-v1 manifest (default: {DEFAULT_EVIDENCE_SPEC}, else the qwen-fold lane copy; refuses if both exist and differ)")
    parser.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR, help="parent directory with tokenizer.json, generation_config.json (+ chat_template.jinja for --bank-render qwen-chat)")
    parser.add_argument("--k3-dir", "--src-dir", dest="k3_dir", default=DEFAULT_K3_DIR, help="K3 directory with tiktoken.model, tokenizer_config.json, config.json, tokenization_kimi.py (Lexar copy by default)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--append-bank-rows", action="store_true", help="write the review files: bank v1 / v2 / probe / smoke prompts as masked review rows")
    parser.add_argument("--review-sequence-length", type=int, default=DEFAULT_REVIEW_SEQUENCE_LENGTH, help="sequence length of the review files (their own short T; default 512)")
    parser.add_argument("--bank-v1", default=str(DEFAULT_BANK_V1))
    parser.add_argument("--bank-v2", default=str(DEFAULT_BANK_V2))
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--smoke", default=str(DEFAULT_SMOKE))
    parser.add_argument("--bank-render", choices=BANK_RENDERS, default="qwen-chat", help="paired review rows: qwen-chat = the parent's chat template, thinking OFF (recorded); plain = ROLE: transcript")
    parser.add_argument("--k3-bank-render", choices=K3_BANK_RENDERS, default="k3-native", help="k3-native (default): also write k3_review_native_input_ids rendered by K3's own tokenization_kimi with thinking=False; none: record that the S2 bank-row logits are NOT a K3-native baseline")
    parser.add_argument("--include-overlong", choices=OVERLONG_POLICIES, default="skip", help="review prompts longer than the review row: skip by name (default) or truncate")
    parser.add_argument("--qwen-separator-id", type=int, default=None, help="the manifest packer's separator/pad id (default: the manifest's sibling build receipt AND the parent's generation_config pad_token_id, which must agree)")
    parser.add_argument("--k3-separator-id", type=int, default=None, help="K3 document separator id (default: tokenizer_config eos_token, [EOS] 163585)")
    parser.add_argument("--k3-pad-id", type=int, default=None, help="K3 row padding id (default: tokenizer_config pad_token, [PAD] 163839)")
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true", help="build although the manifest was tokenized by a DIFFERENT tokenizer (recorded)")
    parser.add_argument("--allow-unpinned-tokenizer", action="store_true", help="build against a parent tokenizer other than the pinned 27B one (recorded)")
    parser.add_argument("--allow-unpinned-k3-tokenizer", action="store_true", help="accept a tiktoken.model other than the pinned K3 rank file (recorded)")
    parser.add_argument("--allow-share-read", action="store_true", help=f"permit reading the small tokenizer files from {SHARE_ROOT}")
    parser.add_argument("--no-share-read", action="store_true", help="refuse every input under the share even when --k3-dir points there")
    parser.add_argument("--qwen-fold-dir", default=str(DEFAULT_QWEN_FOLD_DIR))
    parser.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR), help="calibration lane dir (oc_bank_v2_draft.materialise_haystack for --include-overlong truncate)")
    parser.add_argument("--chimera-src", default=None, help=f"chimera-omni src/ (default: ${CHIMERA_SRC_ENV}, else the server git-archive tree)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    receipt = build_corpus(parse_args(argv))
    print(json.dumps({
        "rows_by_file": receipt["counts"]["rows_by_file"],
        "rows_by_split": receipt["counts"]["rows_by_split"],
        "positions": {"evidence": receipt["pricing"]["evidence_positions"], "review": receipt["pricing"]["review_positions"]},
        "aligned_fraction": receipt["boundary_aggregate"]["all"]["aligned_fraction"],
        "k3_native_baseline": receipt["k3_review_render"]["s2_bank_row_logits_are_k3_native_baseline"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
