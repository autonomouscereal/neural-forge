#!/usr/bin/env python
"""distill_corpus_build.py -- the Stage C/D distillation corpora of
``distill/DISTILL_DESIGN_2026-10-12.md`` (§3 cache inputs, §4 Data columns, §5).

CPU only.  Nothing here loads a model weight or touches a GPU.  Every set lands
under ``--out-dir`` (default ``$OC/distill/corpus``) with its own
``manifest.json`` (counts, token totals, sha256 per shard, source rows,
licence/provenance notes).  Steps, run in this order (``all`` runs them all):

``stage``     mirror the 640 curated registry shards from the NAS to root,
              each one sha256-verified against the 2026-09-02 walk, then
              normalise every row with the loader's own rules
              (``curated_shards.normalize_row``) into ``staging/rows.jsonl``.
              Text is taken with the loader's FIELD order but WITHOUT its
              whitespace compaction and 8,192-character cut: code keeps its
              newlines and long rows keep their tails.
``screen``    the contamination screen and the eval carve-out, BEFORE any set
              is built.  Review surface = bank v1, the bank v2 draft (with its
              materialised haystacks and its five vision-card images), the R-Q
              210-position prompt pack, the 64-prompt probe's whole source
              pack (120 prompts; any seeded 64-draw is a subset), the probe and
              smoke packs, the needle sets (bank v2 needles + the export
              envelope's needle templates and codes), the export parity bank,
              the extension / paraphrase screen banks and the S8 media prompts
              (bank v2 media-out cases + the vision-card assets).  Rules:
              exact (whitespace/case-normalised, items >= 4 words), word
              13-gram overlap, whole-item containment for 5..12-word items,
              literal needle codes, and sha256 for media.  Liveness probes
              prove it fires.  The eval split (2 % per route by a salted
              hash, a per-route floor, the 28-row stage-T split of
              ``sample_balanced_records``, every duplicate of an eval row) is
              carved here and never enters a training set; training rows that
              share > 50 % of their 13-grams with eval rows are held out too.
``allocate``  disjoint row allocation over the training pool, in this order:
              (b) long, (c) K3, (a) text, (f) self-gen prompts.
``text``      (a) 20,000,768 Qwen tokens = 4,883 rows x 4,096, chat-rendered
              (thinking off), stream-packed, route- and document-labelled.
``long``      (b) 31 documents x 32,768: same-route rows concatenated as
              numbered sections with three needles and a passage-retrieval
              question; answers recorded with their token spans.
``k3``        (c) 4 passes x 122 rows x 4,096 dual-tokenized rows (K3 tiktoken
              163,840 + Qwen), raw-rendered, boundary map per pass -- the
              ``composite/corpus`` format of ``k3_corpus_build.py`` whose
              tokenizer and alignment code is imported, not copied.
``measure``   S0 P10: byte size, sha256, image dims / ffprobe facts of every
              image and video sidecar the media sets can use.
``vision``    (d) vision-in / video-in prompts with their media and the visual
              token count at the parent processor's defaults (transformers'
              own ``smart_resize``); answers are for the teacher to generate.
``media``     (e) H3 clips (2 s @ 24 fps, 512 W x 320 H, cut by ffmpeg, with a
              paired audio track), Qwen-Image prompts (+ negatives), Hunyuan3D
              object images, ACE-Step music clips + captions, ViT images.
``selfgen``   (f) prompts worth a 1.0M-token free-run sampling budget.
``verify``    re-hash every shard every manifest names.

Never invents data: a set that cannot be filled from local data is filled with
what exists and the shortfall is written into its manifest.  Synthetic text is
limited to what the design itself specifies (the long set's section headers,
needles and questions), and that text is screened like everything else.

    CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src \\
    nice -n 15 $PY distill_corpus_build.py all
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

VERSION = "1.0.1"
TOOL = "distill_corpus_build"
SCHEMA = "neural-forge-omni-distill-corpus-v1"

HERE = Path(__file__).resolve().parent
#: distill/tools -> the campaign root ($OC on the server, the report mirror locally).
CAMPAIGN_ROOT = HERE.parent.parent

DEFAULT_OUT_DIR = CAMPAIGN_ROOT / "distill" / "corpus"
DEFAULT_POINTER = "/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json"
DEFAULT_WALK = CAMPAIGN_ROOT / "datasets" / "registry-walk-2026-09-02.json"
DEFAULT_WALK_PROGRESS = CAMPAIGN_ROOT / "datasets" / "registry-walk-2026-09-02.progress.jsonl"
DEFAULT_QWEN_DIR = "/home/cereal/models/Qwen3.8-27B-Uncensored"
DEFAULT_K3_DIR = "/mnt/lexar/models/moonshotai/Kimi-K3"
CHIMERA_SRC_ENV = "CHIMERA_OMNI_SRC"
#: The tree the S7 tools use (s7_chart_fit.py --chimera-src default).
DEFAULT_CHIMERA_SRC = CAMPAIGN_ROOT / "import" / "rq27b" / "tree_ef1ffb9" / "src"
DEFAULT_BANKS = CAMPAIGN_ROOT / "calibration" / "banks"
DEFAULT_CALIBRATION_DIR = CAMPAIGN_ROOT / "calibration"
DEFAULT_SCREEN_BANKS = CAMPAIGN_ROOT / "qwen-fold" / "screen-banks"
DEFAULT_EXPORT_DIR = CAMPAIGN_ROOT / "export"
DEFAULT_EVIDENCE = CAMPAIGN_ROOT / "composite" / "corpus" / "omni-evidence-27b-v1.json"
DEFAULT_COMPOSITE_TOOLS = CAMPAIGN_ROOT / "composite" / "tools"
DEFAULT_QWEN_FOLD = CAMPAIGN_ROOT / "qwen-fold"
DEFAULT_ONPOLICY_PACK = HERE / "screen_sources" / "lfm2_on_policy_fit_prompts_v1.json"

#: Qwen3.8-27B-Uncensored tokenizer.json, the same pin k3_corpus_build carries.
QWEN_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
K3_TIKTOKEN_SHA256 = "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
#: sha256 of the 64-prompt probe's source pack (neural-forge experiments/, stage2_freerun_probe draws from it).
ONPOLICY_PACK_SHA256 = "e85000326c5c39ae683506d69bc7e8edf3557845dc3b06e562f5ba6c23fe436f"
#: Text vocab rows: every text loss and every id lives in [0, 248,320).
TEXT_VOCAB_ROWS = 248_320

# --- the design's set sizes ------------------------------------------------
TEXT_SEQ = 4_096
TEXT_ROWS = 4_883  # 4,883 x 4,096 = 20,000,768 >= 20M
TEXT_SHARD_ROWS = 256
LONG_SEQ = 32_768
LONG_DOCS = 31
LONG_RESERVE = 640  # header + questions + answers + chat markup
LONG_SECTION_OVERHEAD = 12  # "### Section N\n" + "\n\n" + boundary slack
K3_SEQ = 4_096
K3_PASSES = 4
K3_ROWS_PER_PASS = 122  # 122 x 4,096 = 499,712 K3 positions <= 0.5M per pass
K3_LOOKAHEAD = 4_096
SELFGEN_BUDGET = 1_000_000
SELFGEN_MIN_BUDGET = 64
SELFGEN_MAX_NEW = 2_048
SELFGEN_MAX_PROMPT = 3_072
H3_CLIPS = 512
H3_SECONDS = 2.0
H3_FPS = 24
H3_WIDTH = 512
H3_HEIGHT = 320
QIMG_PROMPTS = 2_048
HY3D_IMAGES = 256
ACE_CLIPS = 1_024
#: 313 Oobleck latent frames per clip in the design's D4 arithmetic / 25 frames per s.
ACE_SECONDS = 12.52
VIDEO_IN_CLIPS = 512

# --- screen and carve-out ----------------------------------------------------
NGRAM = 13
SHORT_ITEM_MIN_WORDS = 5
EXACT_ITEM_MIN_WORDS = 4
EVAL_SALT = "omni-distill-eval-v1"
EVAL_PER_10000 = 200  # 2 %
EVAL_FLOOR = 4
NEAR_DUP_FRACTION = 0.5
SEED = 20261012

MEDIA_FAMILY = frozenset({
    "image", "video", "audio", "music", "tts", "speech", "sound", "ocr", "multimodal",
    "media_flow", "video_media_flow", "image_grounding", "ocr_document",
})
IMAGE_EXT = frozenset({"png", "jpg", "jpeg", "ppm", "webp", "bmp", "gif"})
VIDEO_EXT = frozenset({"mp4", "webm", "mov", "mkv", "avi"})
AUDIO_EXT = frozenset({"flac", "wav", "mp3", "ogg", "m4a", "opus"})

#: Estate default negative for Qwen-Image (.agents/skills/qwen-text-to-image/scripts/qwen21_t2i.py --negative "").
QIMG_DEFAULT_NEGATIVE = ""
QIMG_NEGATIVE_SOURCE = ".agents/skills/qwen-text-to-image/scripts/qwen21_t2i.py --negative default (empty)"

NEGATIVE_FIELDS = ("negative_prompt", "negative", "neg_prompt")

CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿가-힯"
WORD_RE = re.compile(f"[{CJK}]|[^\\W_{CJK}]+")
_HEX_ONLY = re.compile(r"\A[0-9a-fA-F]{32,}\Z")


def log(message: str) -> None:
    print(f"[{TOOL} {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Refusal(SystemExit):
    """A named refusal: nothing past this point is written."""


# --------------------------------------------------------------------------
# Digests and IO.
# --------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """The estate's canonical encoding (``qf_arch.canonical_json_bytes``), unchanged."""

    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, body: dict[str, Any], seal_field: str | None = None) -> str:
    """Write ``body`` atomically; with ``seal_field`` the body carries its own canonical sha."""

    if seal_field:
        body = {k: v for k, v in body.items() if k != seal_field}
        body[seal_field] = canonical_sha256(body)
    atomic_write_bytes(path, json.dumps(body, indent=1, ensure_ascii=False).encode("utf-8"))
    return body[seal_field] if seal_field else file_sha256(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return file_sha256(path), count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tool_identity() -> dict[str, Any]:
    return {
        "tool": TOOL,
        "version": VERSION,
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(__file__),
        "host": platform.node(),
        "python": platform.python_version(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def salted_unit(salt: str, key: str) -> int:
    """A stable integer in [0, 10000) from ``salt:key`` (sha256, process independent)."""

    return int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:12], 16) % 10_000


def stable_shuffle(items: list[Any], key: Callable[[Any], str], salt: str) -> list[Any]:
    """A seeded order that does not depend on the input order or PYTHONHASHSEED."""

    return sorted(items, key=lambda item: hashlib.sha256(f"{salt}:{key(item)}".encode("utf-8")).hexdigest())


# --------------------------------------------------------------------------
# Paths and lanes.
# --------------------------------------------------------------------------


def resolve_chimera_src(value: str | None) -> Path:
    return Path(value or os.environ.get(CHIMERA_SRC_ENV) or DEFAULT_CHIMERA_SRC)


def import_curated_shards(chimera_src: Path) -> Any:
    if not (chimera_src / "chimera_omni" / "data" / "curated_shards.py").is_file():
        raise Refusal(f"chimera-omni source tree not found at {chimera_src} (set {CHIMERA_SRC_ENV} or --chimera-src)")
    if str(chimera_src) not in sys.path:
        sys.path.insert(0, str(chimera_src))
    from chimera_omni.data import curated_shards  # noqa: E402

    return curated_shards


def out_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "mirror": out_dir / "registry_mirror",
        "staging": out_dir / "staging",
        "rows": out_dir / "staging" / "rows.jsonl",
        "stage_receipt": out_dir / "staging" / "stage_receipt.json",
        "screen": out_dir / "contamination_screen.json",
        "carve": out_dir / "carve.json",
        "eval": out_dir / "eval_split",
        "allocation": out_dir / "allocation.json",
        "text": out_dir / "text",
        "long": out_dir / "long",
        "k3": out_dir / "k3",
        "measure": out_dir / "media_measurements.json",
        "media_store": out_dir / "media_store",
        "vision": out_dir / "vision_in",
        "media": out_dir / "media",
        "selfgen": out_dir / "selfgen",
        "verify": out_dir / "verify_receipt.json",
    }


# --------------------------------------------------------------------------
# Row text: the loader's field order, without its compaction.
# --------------------------------------------------------------------------


def _clean(value: Any) -> str:
    return str(value).replace("\x00", " ").strip()


def _jsonish(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _messages_text(messages: Any, prefer_roles: tuple[str, ...]) -> str:
    if not isinstance(messages, list):
        return ""
    preferred: list[str] = []
    fallback: list[str] = []
    roles = {role.lower() for role in prefer_roles}
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if str(item.get("role", "")).lower() in roles:
            preferred.append(content)
        else:
            fallback.append(content)
    if preferred:
        return "\n".join(preferred)
    return "\n".join(fallback[:2])


def _extract_from_payload(value: Any, prefer: tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        found = _messages_text(value.get("messages"), ("user", "assistant", "model", "system"))
        if found:
            return found
        for key in prefer:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item
        return _jsonish(value)
    if isinstance(value, list):
        return _messages_text(value, ("user", "assistant", "model", "system")) or _jsonish(value)
    return ""


def extract_instruction(row: Mapping[str, Any]) -> str:
    """``curated_shards._extract_instruction`` field for field, minus ``_compact``."""

    for key in ("instruction", "prompt", "question", "query", "task_prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
    for key in ("input_json", "input"):
        found = _extract_from_payload(row.get(key), ("prompt", "instruction", "question", "query", "content", "text"))
        if found:
            return _clean(found)
    found = _messages_text(row.get("messages"), ("user", "system"))
    if found:
        return _clean(found)
    return _clean(_jsonish(row.get("input_json") or row.get("messages") or ""))


def extract_response(row: Mapping[str, Any]) -> str:
    """``curated_shards._extract_response`` field for field, minus ``_compact``."""

    for key in ("response", "answer", "output", "completion", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)
    for key in ("target_json", "target", "output_json"):
        found = _extract_from_payload(row.get(key), ("response", "answer", "output", "completion", "content", "transcript", "text"))
        if found:
            return _clean(found)
    found = _messages_text(row.get("messages"), ("assistant", "model"))
    if found:
        return _clean(found)
    return _clean(_jsonish(row.get("target_json") or ""))


def raw_render(instruction: str, response: str) -> str:
    """The fold evidence's render (``qf_evidence_build.render_row_text``): instruction, blank line, response."""

    return "\n\n".join(part for part in (instruction, response) if part)


def media_ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def media_kind(path: str) -> str | None:
    ext = media_ext(path)
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


SIDECAR_KEYS = ("path", "uri", "kind", "role", "sha256", "byte_size", "width", "height", "duration_sec", "fps", "frame_count", "mime_type", "dataset", "license", "generator")


def sidecar_record(item: Mapping[str, Any], path: str) -> dict[str, Any]:
    out = {key: item.get(key) for key in SIDECAR_KEYS if item.get(key) is not None and key not in ("path", "uri")}
    out["path"] = path
    out["media"] = media_kind(path)
    for key in ("license", "generator", "dataset"):
        if key in out:
            out[key] = str(out[key])[:160]
    return out


def row_negative_prompt(row: Mapping[str, Any]) -> str | None:
    for container in (row, row.get("input_json") if isinstance(row.get("input_json"), dict) else {}, row.get("metadata") if isinstance(row.get("metadata"), dict) else {}):
        for key in NEGATIVE_FIELDS:
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, str):
                return value
    return None


def task_type_of(row: Mapping[str, Any]) -> str:
    value = row.get("task_type")
    if not value and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get("task_type")
    return str(value or "")


# --------------------------------------------------------------------------
# Words, n-grams and the review index.
# --------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Screen tokens: NFKC, lower case, letter/digit runs; every CJK character is its own word."""

    return WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())


def norm_key(text: str) -> str:
    return " ".join(words(text))


def harvest_strings(node: Any, out: list[str]) -> None:
    """Every string value of a pack and the transcript of every message list (keys excluded,
    pure hex digests excluded: they are provenance, not review content)."""

    if isinstance(node, str):
        if not _HEX_ONLY.match(node.strip()):
            out.append(node)
        return
    if isinstance(node, list):
        for item in node:
            harvest_strings(item, out)
        return
    if isinstance(node, Mapping):
        messages = node.get("messages")
        if isinstance(messages, list) and all(isinstance(m, Mapping) for m in messages):
            out.append("\n\n".join(str(m.get("content", "")) for m in messages))
        for value in node.values():
            harvest_strings(value, out)


class ReviewIndex:
    """The review surface as exact keys, 13-grams, short items, literal codes and media digests."""

    def __init__(self, ngram: int = NGRAM, short_min: int = SHORT_ITEM_MIN_WORDS, exact_min: int = EXACT_ITEM_MIN_WORDS) -> None:
        self.n = int(ngram)
        self.short_min = int(short_min)
        self.exact_min = int(exact_min)
        self.grams: dict[tuple[str, ...], tuple[str, str]] = {}
        self.gram_first: set[str] = set()
        self.short: dict[tuple[str, str], list[tuple[tuple[str, ...], str, str]]] = defaultdict(list)
        self.exact: dict[str, tuple[str, str]] = {}
        self.literals: dict[str, tuple[str, str]] = {}
        self.media: dict[str, tuple[str, str]] = {}
        self.items: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        self.sources: list[dict[str, Any]] = []

    def add_text(self, source: str, item_id: str, text: str) -> None:
        toks = words(text)
        if not toks:
            return
        if len(toks) >= self.exact_min:
            self.exact.setdefault(" ".join(toks), (source, item_id))
        if len(toks) >= self.n:
            for i in range(len(toks) - self.n + 1):
                gram = tuple(toks[i : i + self.n])
                if gram not in self.grams:
                    self.grams[gram] = (source, item_id)
                    self.gram_first.add(gram[0])
        elif len(toks) >= self.short_min:
            gram = tuple(toks)
            bucket = self.short[(gram[0], gram[1])]
            if all(existing != gram for existing, _, _ in bucket):
                bucket.append((gram, source, item_id))
        self.items[source].append((item_id, text, len(toks)))

    def add_literal(self, source: str, item_id: str, literal: str) -> None:
        self.literals.setdefault(literal.lower(), (source, item_id))
        self.items[source].append((item_id, literal, len(words(literal))))

    def add_media(self, source: str, item_id: str, sha256: str) -> None:
        self.media.setdefault(sha256.lower(), (source, item_id))

    def hits(self, text: str, exact_keys: Iterable[str] | None = None, toks: list[str] | None = None) -> list[dict[str, str]]:
        """Every independent reason ``text`` is review content (first hit per rule).

        ``exact_keys`` are normalised field keys (``norm_key``) compared whole; by default the
        whole text's key."""

        found: list[dict[str, str]] = []
        toks = words(text) if toks is None else toks
        for key in (exact_keys if exact_keys is not None else (" ".join(toks),)):
            if key in self.exact:
                source, item = self.exact[key]
                found.append({"rule": "exact", "source": source, "item": item})
                break
        n = self.n
        first = self.gram_first
        grams = self.grams
        for i in range(len(toks) - n + 1):
            if toks[i] in first:
                hit = grams.get(tuple(toks[i : i + n]))
                if hit is not None:
                    found.append({"rule": f"ngram{n}", "source": hit[0], "item": hit[1]})
                    break
        if self.short:
            short = self.short
            done = False
            for i in range(len(toks) - 1):
                candidates = short.get((toks[i], toks[i + 1]))
                if not candidates:
                    continue
                for gram, source, item in candidates:
                    if tuple(toks[i : i + len(gram)]) == gram:
                        found.append({"rule": "short_item", "source": source, "item": item})
                        done = True
                        break
                if done:
                    break
        if self.literals:
            lowered = text.lower()
            for literal, (source, item) in self.literals.items():
                if literal in lowered:
                    found.append({"rule": "literal", "source": source, "item": item})
                    break
        return found

    def summary(self) -> dict[str, Any]:
        return {
            "ngram": self.n,
            "short_item_min_words": self.short_min,
            "exact_item_min_words": self.exact_min,
            "ngram_count": len(self.grams),
            "short_items": sum(len(v) for v in self.short.values()),
            "exact_keys": len(self.exact),
            "literal_codes": len(self.literals),
            "media_digests": len(self.media),
            "items_by_source": {source: len(items) for source, items in self.items.items()},
        }


def parse_prompt_pack(script: Path) -> list[str]:
    """``PROMPT_PACK`` of ``scripts/import_receipt_qwen35.py`` read by AST (the script imports torch)."""

    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[Any] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "PROMPT_PACK":
                value = ast.literal_eval(node.value)
                return [str(v) for v in value]
    raise Refusal(f"{script} defines no PROMPT_PACK")


def module_constants(script: Path, names: Iterable[str]) -> dict[str, Any]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    wanted = set(names)
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    if missing:
        raise Refusal(f"{script} lacks {sorted(missing)}")
    return found


def materialise_haystack(spec: dict[str, Any], case_id: str, calibration_dir: Path) -> str:
    """``oc_bank_v2_draft.materialise_haystack`` with the bank's pinned digest checked."""

    if str(calibration_dir) not in sys.path:
        sys.path.insert(0, str(calibration_dir))
    from oc_bank_v2_draft import materialise_haystack as _materialise  # noqa: E402

    document = _materialise(spec)
    observed = canonical_sha256({"text": document})
    if observed != str(spec.get("haystack_sha256")):
        raise Refusal(f"bank v2 case {case_id}: regenerated haystack digests {observed}, the bank pinned {spec.get('haystack_sha256')}")
    return document


def build_review_index(args: argparse.Namespace) -> tuple[ReviewIndex, list[dict[str, Any]]]:
    """Every review surface the design names, each recorded with its sha256 and item count."""

    index = ReviewIndex()
    sources: list[dict[str, Any]] = []
    banks = Path(args.banks)
    chimera_src = resolve_chimera_src(args.chimera_src)

    def pack(name: str, path: Path, role: str) -> Any:
        if not path.is_file():
            raise Refusal(f"review source {name} missing at {path}; the screen would not cover the surface")
        payload = load_json(path)
        sources.append({"name": name, "role": role, "path": str(path), "sha256": file_sha256(path)})
        return payload

    def add_strings(name: str, payload: Any) -> int:
        strings: list[str] = []
        harvest_strings(payload, strings)
        before = len(index.items[name])
        for i, value in enumerate(strings):
            index.add_text(name, f"{name}#{i}", value)
        return len(index.items[name]) - before

    # bank v1 (frozen, sealed) -------------------------------------------------
    payload = pack("bank_v1", banks / "lfm2_semantic_whole_response_prompts_v1.json", "frozen review bank v1")
    sources[-1]["items"] = add_strings("bank_v1", payload)
    # bank v2 draft: every string, every materialised haystack, the asset images --
    payload = pack("bank_v2", banks / "omni_review_bank_v2_draft.json", "review bank v2 draft (text, vision, long-context, media-out lanes)")
    sources[-1]["items"] = add_strings("bank_v2", payload)
    haystacks = 0
    for case in payload.get("prompts", []):
        spec = case.get("haystack_spec")
        if spec:
            document = materialise_haystack(spec, str(case.get("id")), Path(args.calibration_dir))
            index.add_text("bank_v2_haystack", f"{case.get('id')}", document)
            haystacks += 1
    sources.append({"name": "bank_v2_haystack", "role": "materialised long-context haystacks (oc-filler-v1, digest-checked)", "path": str(banks / "omni_review_bank_v2_draft.json"), "items": haystacks})
    media_cases = [c for c in payload.get("prompts", []) if c.get("modality_out", "text") != "text" or c.get("modality_in", "text") != "text"]
    sources.append({"name": "s8_media_prompts", "role": "S8 media prompts = bank v2 media-out / media-in cases (inside bank_v2 strings)", "items": len(media_cases), "case_ids": [c.get("id") for c in media_cases]})
    asset_dir = banks / "assets"
    assets = sorted(asset_dir.glob("*")) if asset_dir.is_dir() else []
    for asset in assets:
        digest = file_sha256(asset)
        index.add_media("bank_v2_assets", asset.name, digest)
    sources.append({"name": "bank_v2_assets", "role": "vision-card images (S8 vision cards)", "path": str(asset_dir), "items": len(assets), "files": {a.name: file_sha256(a) for a in assets}})
    # probe and smoke packs --------------------------------------------------------
    for name, file in (("probe_pack", "matrix_campaign_probe_prompts_v1.json"), ("smoke_pack", "matrix_campaign_smoke_prompts_v1.json")):
        payload = pack(name, banks / file, "sealed probe/smoke pack")
        sources[-1]["items"] = add_strings(name, payload)
    # R-Q prompt pack ------------------------------------------------------------
    script = chimera_src.parent / "scripts" / "import_receipt_qwen35.py"
    if not script.is_file():
        raise Refusal(f"R-Q prompt pack script missing at {script}")
    prompts = parse_prompt_pack(script)
    for i, text in enumerate(prompts):
        index.add_text("rq_prompts", f"rq-{i}", text)
    sources.append({"name": "rq_prompts", "role": "R-Q 8 prompts / 210 positions (PROMPT_PACK, read by AST)", "path": str(script), "sha256": file_sha256(script), "items": len(prompts)})
    # the 64-prompt probe's source pack ---------------------------------------------
    payload = pack("probe64_pack", Path(args.onpolicy_pack), "64-prompt free-run probe source (all 120 prompts; any seeded 64-draw is a subset)")
    if sources[-1]["sha256"] != ONPOLICY_PACK_SHA256:
        raise Refusal(f"{args.onpolicy_pack} hashes {sources[-1]['sha256']}, the pinned 64-prompt probe source is {ONPOLICY_PACK_SHA256}")
    sources[-1]["items"] = add_strings("probe64_pack", payload.get("prompts", []))
    # needle sets -----------------------------------------------------------------
    export = Path(args.export_dir)
    probe_constants = module_constants(export / "llama_probe.py", ("NEEDLE_TEMPLATE", "NEEDLE_QUESTION", "FILLER_TEMPLATE"))
    for key, value in probe_constants.items():
        index.add_text("needle_sets", key, str(value))
    codes = module_constants(export / "envelope_bench.py", ("NEEDLE_CODES",))["NEEDLE_CODES"]
    for code in codes:
        index.add_literal("needle_codes", code, str(code))
    sources.append({"name": "needle_sets", "role": "export envelope needle templates (llama_probe.py)", "path": str(export / "llama_probe.py"), "sha256": file_sha256(export / "llama_probe.py"), "items": len(probe_constants)})
    sources.append({"name": "needle_codes", "role": "export envelope needle codes (envelope_bench.py NEEDLE_CODES), literal containment", "path": str(export / "envelope_bench.py"), "sha256": file_sha256(export / "envelope_bench.py"), "items": len(codes)})
    payload = pack("export_parity_bank", export / "parity_bank.json", "export fidelity prompts (probe surface)")
    sources[-1]["items"] = add_strings("export_parity_bank", payload.get("prompts", []))
    # extension / paraphrase screen banks -----------------------------------------------
    screen_banks = Path(args.screen_banks)
    for name, file in (("extension_bank_v2", "lfm2_extension_bank_prompts_v2.json"), ("paraphrase_triplets_v1", "lfm2_paraphrase_triplets_v1.json")):
        payload = pack(name, screen_banks / file, "qwen-fold screen bank")
        sources[-1]["items"] = add_strings(name, payload)
    s8_dir = CAMPAIGN_ROOT / "composite" / "s8"
    sources.append({"name": "composite_s8", "role": "S8 stimuli directory", "path": str(s8_dir), "exists_at_screen_time": s8_dir.exists(), "note": "absent at screen time: S8 media prompts are the bank v2 media cases; re-run `screen` if step 1 adds stimuli"})
    index.sources = sources
    return index, sources


def screen_liveness(index: ReviewIndex, probes_per_source: int = 8) -> dict[str, Any]:
    """Prove the screen fires on REAL review items (verbatim and re-wrapped), on short items and
    literal codes, and stays quiet on a 12-word fragment and on clean filler."""

    by_source: list[dict[str, Any]] = []
    totals = Counter()
    filler_head = "The following passage continues a longer maintenance document.\n"
    filler_tail = "\nThe document then moves on to an unrelated topic entirely."
    for source, items in sorted(index.items.items()):
        long_items = sorted((it for it in items if it[2] >= index.n), key=lambda it: (-it[2], it[0]))[:probes_per_source]
        short_items = [it for it in items if index.short_min <= it[2] < index.n][:probes_per_source]
        literal_items = [it for it in items if it[1].lower() in index.literals][:probes_per_source]
        row = Counter()
        for _, text, _ in long_items:
            row["long_probes"] += 1
            if index.hits(filler_head + text + filler_tail):
                row["long_caught"] += 1
            rewrapped = "Exported log:\n" + "\n".join(text.upper().split()) + "\nEnd."
            if index.hits(rewrapped):
                row["rewrapped_caught"] += 1
            toks = words(text)
            if len(toks) >= 2 * index.n:
                middle = len(toks) // 2
                fragment = " ".join(toks[middle : middle + index.n - 1])
                if not index.hits(fragment):
                    row["fragment_controls"] += 1
                    # Sentinel neighbours: an ordinary neighbour can rebuild a real 13-gram when the
                    # item's own next word is that neighbour (1.0.0 failed its control that way).
                    if index.hits(f"Entry note: the technician wrote qqzxv {fragment} vxzqq and closed it."):
                        row["fragment_controls_falsely_caught"] += 1
        for _, text, _ in short_items:
            row["short_probes"] += 1
            if index.hits(f"In passing, someone said {text} before leaving."):
                row["short_caught"] += 1
        for _, text, _ in literal_items:
            row["literal_probes"] += 1
            if index.hits(f"maintenance ticket reference {text} closed"):
                row["literal_caught"] += 1
        totals.update(row)
        by_source.append({"source": source, **dict(row)})
    clean_caught = 0
    clean = 32
    for i in range(clean):
        sentence = f"Batch {i}: the pump was serviced on the {i % 28 + 1}th and the reading settled at {i * 3 % 97} units after the valve was reseated."
        if index.hits(sentence):
            clean_caught += 1
    live = (
        totals["long_probes"] > 0
        and totals["long_caught"] == totals["long_probes"]
        and totals["rewrapped_caught"] == totals["long_probes"]
        and totals["short_caught"] == totals["short_probes"]
        and totals["literal_caught"] == totals["literal_probes"]
        and totals["fragment_controls_falsely_caught"] == 0
        and clean_caught == 0
    )
    return {"rule": "every long item caught verbatim and re-wrapped (upper case, one word per line); every short item and literal code caught; no 12-word fragment and no clean filler caught", "totals": dict(totals), "clean_controls": clean, "clean_controls_falsely_caught": clean_caught, "by_source": by_source, "screen_is_live": live}


# --------------------------------------------------------------------------
# Tokenizers.
# --------------------------------------------------------------------------


class QwenText:
    """The 27B parent's tokenizer.json (pinned) + its chat template (thinking off)."""

    def __init__(self, qwen_dir: Path, *, allow_unpinned: bool = False) -> None:
        from tokenizers import Tokenizer  # noqa: E402

        path = qwen_dir / "tokenizer.json"
        self.sha256 = file_sha256(path)
        if self.sha256 != QWEN_TOKENIZER_SHA256 and not allow_unpinned:
            raise Refusal(f"{path} hashes {self.sha256}, the pinned 27B tokenizer is {QWEN_TOKENIZER_SHA256}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        generation = load_json(qwen_dir / "generation_config.json")
        self.separator_id = int(generation["pad_token_id"])
        eos = generation.get("eos_token_id")
        eos_ids = eos if isinstance(eos, list) else [eos]
        if self.separator_id not in [int(e) for e in eos_ids]:
            raise Refusal(f"generation_config pad_token_id {self.separator_id} is not among eos {eos_ids}")
        if self.tokenizer.id_to_token(self.separator_id) != "<|endoftext|>":
            raise Refusal(f"separator {self.separator_id} is {self.tokenizer.id_to_token(self.separator_id)!r}, expected <|endoftext|>")
        self.qwen_dir = qwen_dir
        self._chat = None
        template = qwen_dir / "chat_template.jinja"
        self.chat_template_sha256 = file_sha256(template) if template.is_file() else None

    def _chat_tokenizer(self) -> Any:
        if self._chat is None:
            from transformers import AutoTokenizer  # noqa: E402

            self._chat = AutoTokenizer.from_pretrained(str(self.qwen_dir))
        return self._chat

    def chat(self, messages: list[dict[str, str]], *, generation_prompt: bool = False) -> str:
        return str(self._chat_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_many(self, texts: list[str], batch: int = 512) -> list[list[int]]:
        out: list[list[int]] = []
        for start in range(0, len(texts), batch):
            for encoded in self.tokenizer.encode_batch(texts[start : start + batch], add_special_tokens=False):
                out.append(list(encoded.ids))
        return out

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return list(encoded.ids), [tuple(o) for o in encoded.offsets]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)

    def report(self) -> dict[str, Any]:
        return {"tokenizer_json": str(self.qwen_dir / "tokenizer.json"), "sha256": self.sha256, "separator_id": self.separator_id, "separator_piece": "<|endoftext|>", "chat_template_sha256": self.chat_template_sha256, "chat_render": "apply_chat_template(enable_thinking=False); the template emits an empty <think></think> block in assistant turns", "add_special_tokens": False}


def chat_messages(instruction: str, response: str) -> list[dict[str, str]] | None:
    if instruction and response:
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
    return None


# --------------------------------------------------------------------------
# Step: stage.
# --------------------------------------------------------------------------


def mirror_shards(walk_records: list[dict[str, Any]], mirror: Path, *, timeout_s: float = 900.0) -> list[dict[str, Any]]:
    """Copy every walked shard to root, verifying its sha256 against the walk while reading."""

    shards: list[dict[str, Any]] = []
    by_sha: dict[str, Path] = {}
    for record in walk_records:
        entry_index = int(record["entry_index"])
        for shard in record.get("shards", []):
            source = Path(str(shard["path"]))
            expected = str(shard["sha256"])
            destination = mirror / f"{entry_index:03d}" / source.name
            status = "copied"
            if expected in by_sha and by_sha[expected] != destination:
                status = "same_bytes_as_" + str(by_sha[expected].relative_to(mirror))
            if destination.is_file() and file_sha256(destination) == expected:
                status = "present"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                digest = hashlib.sha256()
                started = time.time()
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    for block in iter(lambda: reader.read(1 << 22), b""):
                        digest.update(block)
                        writer.write(block)
                        if time.time() - started > timeout_s:
                            raise Refusal(f"copy of {source} exceeded {timeout_s} s")
                if digest.hexdigest() != expected:
                    temporary.unlink(missing_ok=True)
                    raise Refusal(f"{source} hashes {digest.hexdigest()}, the 2026-09-02 walk recorded {expected}: the registry changed under this build")
                os.replace(temporary, destination)
            by_sha.setdefault(expected, destination)
            shards.append({
                "entry_index": entry_index,
                "dataset_id": str(record.get("dataset_id", "")),
                "training_status": str(record.get("training_status", "")),
                "shard_id": shard.get("shard_id"),
                "source_path": str(source),
                "mirror_path": str(destination),
                "sha256": expected,
                "bytes": destination.stat().st_size,
                "status": status,
            })
    return shards


def stage(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    cs = import_curated_shards(resolve_chimera_src(args.chimera_src))
    sys.path.insert(0, str(Path(args.qwen_fold_dir)))
    from qf_evidence_build import row_marker_reasons  # noqa: E402

    walk = load_json(args.walk)
    pointer_now = file_sha256(args.pointer) if Path(args.pointer).is_file() else None
    pointer_walk = walk["context"]["pointer_sha256"]
    if pointer_now != pointer_walk and not args.allow_registry_drift:
        raise Refusal(f"registry pointer is {pointer_now}, the walk read {pointer_walk}: the registry moved since 2026-09-02 (pass --allow-registry-drift to record it)")
    records = read_jsonl(Path(args.walk_progress))
    records.sort(key=lambda r: int(r["entry_index"]))
    log(f"mirroring {sum(len(r.get('shards', [])) for r in records)} shards of {len(records)} entries to {out['mirror']}")
    shards = mirror_shards(records, out["mirror"])
    mirrored_bytes = sum(s["bytes"] for s in shards)
    log(f"mirror ok: {len(shards)} shards, {mirrored_bytes:,} B, every sha256 equal to the walk")

    qwen = QwenText(Path(args.qwen_dir))
    rows: list[dict[str, Any]] = []
    drops = Counter()
    seen_shard_sha: set[str] = set()
    for shard in shards:
        duplicate_shard = shard["sha256"] in seen_shard_sha
        seen_shard_sha.add(shard["sha256"])
        with Path(shard["mirror_path"]).open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    drops["unparsable"] += 1
                    continue
                if not isinstance(row, dict):
                    drops["not_an_object"] += 1
                    continue
                marker = row_marker_reasons(row)
                record = cs.normalize_row(row, source_dataset=shard["dataset_id"], training_status=shard["training_status"], shard_path=shard["source_path"], line_number=line_number)
                if record is None:
                    drops["normalize_row_refused"] += 1
                    continue
                instruction = extract_instruction(row)
                response = extract_response(row)
                sidecars = []
                for item in cs._extract_sidecars(row):
                    path = cs._sidecar_path(item)
                    if path:
                        sidecars.append(sidecar_record(item, path))
                compact_render = raw_render(record.instruction, record.response)
                rows.append({
                    "rid": f"{shard['entry_index']:03d}:{shard['shard_id']}:{line_number}",
                    "uid": record.row_sha256,
                    "entry_index": shard["entry_index"],
                    "dataset_id": shard["dataset_id"],
                    "training_status": shard["training_status"],
                    "shard_sha256": shard["sha256"],
                    "shard_mirror": shard["mirror_path"],
                    "line_number": line_number,
                    "duplicate_shard": duplicate_shard,
                    "route": record.route,
                    "raw_route": record.raw_route,
                    "family": "media" if record.route in MEDIA_FAMILY else "text",
                    "record_id": record.record_id,
                    "task_type": task_type_of(row),
                    "instruction": instruction,
                    "response": response,
                    "sidecars": sidecars,
                    "negative_prompt": row_negative_prompt(row),
                    "license": str(row.get("license") or "")[:200],
                    "license_tier": str(row.get("license_tier") or "")[:80],
                    "source_dataset": str(row.get("source_dataset") or row.get("dataset_name") or "")[:120],
                    "markers": list(marker),
                    "text_key": text_sha256(" ".join(raw_render(instruction, response).split())),
                    "evidence_key": text_sha256(" ".join(unicodedata.normalize("NFC", compact_render).split())),
                })
    log(f"normalised {len(rows):,} rows; drops {dict(drops)}")

    # Render and count (exact ids are recomputed by each builder from the same text).
    chats: list[str] = []
    prompts: list[str] = []
    for row in rows:
        messages = chat_messages(row["instruction"], row["response"])
        chats.append(qwen.chat(messages) if messages else raw_render(row["instruction"], row["response"]))
        prompts.append(qwen.chat([{"role": "user", "content": row["instruction"]}], generation_prompt=True) if row["instruction"] else "")
    raws = [raw_render(r["instruction"], r["response"]) for r in rows]
    responses = [r["response"] for r in rows]
    log("tokenizing chat / raw / prompt / response renders")
    for column, texts in (("tok_chat", chats), ("tok_raw", raws), ("tok_prompt", prompts), ("tok_response", responses)):
        encoded = qwen.encode_many(texts)
        for row, ids in zip(rows, encoded):
            row[column] = len(ids)
            if column in ("tok_chat", "tok_raw"):
                row[f"sep_in_{column[4:]}"] = qwen.separator_id in ids
                row[f"max_id_{column[4:]}"] = max(ids) if ids else -1
        log(f"  {column}: {sum(len(ids) for ids in encoded):,} tokens")
    for row, chat in zip(rows, chats):
        row["render"] = "chat" if chat_messages(row["instruction"], row["response"]) else "plain"

    rows_sha, rows_count = write_jsonl(out["rows"], rows)
    by_route = defaultdict(lambda: Counter())
    for row in rows:
        bucket = by_route[row["route"]]
        bucket["rows"] += 1
        bucket["tok_chat"] += row["tok_chat"]
        bucket["tok_raw"] += row["tok_raw"]
        bucket["rows_with_sidecars"] += int(bool(row["sidecars"]))
    receipt = {
        "schema": f"{SCHEMA}/stage",
        "identity": tool_identity(),
        "pointer": {"path": args.pointer, "sha256_now": pointer_now, "sha256_walk": pointer_walk, "equal": pointer_now == pointer_walk},
        "walk": {"path": str(args.walk), "sha256": file_sha256(args.walk), "progress": str(args.walk_progress), "progress_sha256": file_sha256(args.walk_progress), "total_rows": walk["context"]["total_rows"], "total_shards": walk["context"]["total_shards"]},
        "mirror": {"dir": str(out["mirror"]), "shards": len(shards), "bytes": mirrored_bytes, "every_sha256_equals_walk": True, "shards_list": shards},
        "loader": {"curated_shards": str(Path(cs.__file__).resolve()), "sha256": file_sha256(cs.__file__), "row_marker_reasons": "qwen-fold/qf_evidence_build.py"},
        "text_rule": "field order of curated_shards._extract_instruction/_extract_response; \\x00 -> space and outer strip only (no whitespace compaction, no 8,192-char cut); evidence_key uses the loader's compacted view to match the fold evidence",
        "tokenizer": qwen.report(),
        "rows": rows_count,
        "rows_sha256": rows_sha,
        "drops": dict(drops),
        "by_route": {route: dict(counter) for route, counter in sorted(by_route.items())},
        "tokens": {"chat": sum(r["tok_chat"] for r in rows), "raw": sum(r["tok_raw"] for r in rows)},
        "rows_with_marker": sum(1 for r in rows if r["markers"]),
        "rows_with_separator_inside_chat": sum(1 for r in rows if r["sep_in_chat"]),
        "rows_with_separator_inside_raw": sum(1 for r in rows if r["sep_in_raw"]),
        "wall_seconds": round(time.time() - started, 2),
    }
    write_json(out["stage_receipt"], receipt, seal_field="receipt_sha256")
    log(f"stage: {rows_count:,} rows -> {out['rows']} ({rows_sha[:16]}); chat tokens {receipt['tokens']['chat']:,}, raw {receipt['tokens']['raw']:,}")
    return receipt


def load_rows(out: dict[str, Path]) -> list[dict[str, Any]]:
    if not out["rows"].is_file():
        raise Refusal(f"{out['rows']} missing: run `stage` first")
    return read_jsonl(out["rows"])


# --------------------------------------------------------------------------
# Step: screen (contamination screen + eval carve-out).
# --------------------------------------------------------------------------


def evidence_documents(evidence_path: Path, qwen: QwenText) -> tuple[set[str], dict[str, Any]]:
    """Normalised-text digests of every document of the fold evidence manifest (decoded ids)."""

    manifest = load_json(evidence_path)
    keys: set[str] = set()
    documents = 0
    for row in manifest["rows"]:
        current: list[int] = []
        for token in [int(t) for t in row["ids"]] + [qwen.separator_id]:
            if token == qwen.separator_id:
                if current:
                    text = qwen.decode(current)
                    keys.add(text_sha256(" ".join(unicodedata.normalize("NFC", text).split())))
                    documents += 1
                    current = []
            else:
                current.append(token)
    return keys, {"path": str(evidence_path), "sha256": file_sha256(evidence_path), "manifest_sha256": manifest.get("manifest_sha256"), "tokenizer_sha256": manifest.get("tokenizer_sha256"), "documents_decoded": documents, "distinct_documents": len(keys)}


def stage_t_eval_uids(cs: Any, pointer: str) -> tuple[set[str], dict[str, Any]]:
    """The 28-row registry eval split stage T names: ``sample_balanced_records`` defaults (seed 29)."""

    started = time.time()
    try:
        _, eval_records, report = cs.sample_balanced_records(pointer)
    except Exception as exc:  # recorded, not hidden
        return set(), {"status": "failed", "error": repr(exc)}
    uids = {r.row_sha256 for r in eval_records}
    return uids, {"status": "ok", "rows": len(eval_records), "by_route": report.get("eval_by_route"), "wall_seconds": round(time.time() - started, 2), "call": "curated_shards.sample_balanced_records(pointer) defaults: train 6 / eval 2 per route, 14 PROOF_ROUTES, seed 29, max_entries 160"}


def dedup_key(row: Mapping[str, Any]) -> str:
    """Normalised text for text rows; for media rows the text AND the sidecar identities, so two
    rows that share a templated caption but show different images stay two rows."""

    if row["family"] != "media":
        return row["text_key"]
    media = sorted(str(s.get("sha256") or s.get("path")) for s in row["sidecars"])
    return text_sha256(row["text_key"] + "|" + "|".join(media))


def carve_eval(rows: list[dict[str, Any]], *, forced_eval_uids: set[str], evidence_keys: set[str]) -> dict[str, str]:
    """``rid -> 'eval' | 'train'`` before screening.  Salted hash (2 %), a per-route floor, the
    stage-T rows, then every duplicate (same normalised text) of an eval row joins the eval split.
    Fold-evidence documents are never eval: they are in-sample for the charts."""

    assignment: dict[str, str] = {}
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    for route, members in by_route.items():
        ordered = sorted(members, key=lambda r: (salted_unit(EVAL_SALT, r["dedup_key"]), r["rid"]))
        floor = min(EVAL_FLOOR, len(members) // 10)
        for position, row in enumerate(ordered):
            chosen = salted_unit(EVAL_SALT, row["dedup_key"]) < EVAL_PER_10000 or position < floor
            assignment[row["rid"]] = "eval" if chosen else "train"
    for row in rows:
        if row["uid"] in forced_eval_uids:
            assignment[row["rid"]] = "eval"
    eval_keys = {row["dedup_key"] for row in rows if assignment[row["rid"]] == "eval"}
    for row in rows:
        if row["dedup_key"] in eval_keys:
            assignment[row["rid"]] = "eval"
    for row in rows:
        if row["evidence_key"] in evidence_keys and assignment[row["rid"]] == "eval" and row["uid"] not in forced_eval_uids:
            assignment[row["rid"]] = "train"
    return assignment


def ngram_hashes(toks: list[str], n: int = NGRAM) -> list[int]:
    return [hash(tuple(toks[i : i + n])) for i in range(len(toks) - n + 1)]


def screen(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = load_rows(out)
    cs = import_curated_shards(resolve_chimera_src(args.chimera_src))
    index, sources = build_review_index(args)
    liveness = screen_liveness(index)
    log(f"review index: {index.summary()}; live={liveness['screen_is_live']}")
    if not liveness["screen_is_live"]:
        raise Refusal(f"the contamination screen did not behave: {json.dumps(liveness['totals'])}, clean caught {liveness['clean_controls_falsely_caught']}")
    qwen = QwenText(Path(args.qwen_dir))
    evidence_keys, evidence_report = evidence_documents(Path(args.evidence), qwen)
    forced, stage_t = stage_t_eval_uids(cs, args.pointer)
    log(f"fold evidence: {evidence_report['distinct_documents']} documents; stage-T eval rows: {stage_t.get('rows')} ({stage_t['status']})")

    # Review-surface hits, per row ------------------------------------------------
    hits: list[dict[str, Any]] = []
    contaminated: set[str] = set()
    toks_by_rid: dict[str, list[str]] = {}
    for row in rows:
        text = raw_render(row["instruction"], row["response"])
        toks_instruction, toks_response = words(row["instruction"]), words(row["response"])
        toks = toks_instruction + toks_response  # == words(text): the blank line is a word boundary
        toks_by_rid[row["rid"]] = toks
        keys = (" ".join(toks_instruction), " ".join(toks_response), " ".join(toks))
        found = index.hits(text, exact_keys=keys, toks=toks)
        for sidecar in row["sidecars"]:
            digest = str(sidecar.get("sha256") or "").lower()
            if digest and digest in index.media:
                source, item = index.media[digest]
                found.append({"rule": "media_sha256", "source": source, "item": item})
        if found:
            contaminated.add(row["rid"])
            hits.append({"rid": row["rid"], "uid": row["uid"], "record_id": row["record_id"], "dataset_id": row["dataset_id"], "route": row["route"], "reasons": found})
    log(f"review-surface hits: {len(hits)} rows")

    # Eval carve-out ---------------------------------------------------------------
    for row in rows:
        row["dedup_key"] = dedup_key(row)
    assignment = carve_eval(rows, forced_eval_uids=forced, evidence_keys=evidence_keys)
    for rid in contaminated:
        assignment[rid] = "contaminated"
    marker_rids = {row["rid"] for row in rows if row["markers"]}
    for rid in marker_rids:
        if assignment[rid] == "train":
            assignment[rid] = "marker_refused"
    eval_rows = [row for row in rows if assignment[row["rid"]] == "eval"]

    # Train rows that near-duplicate an eval row are held out --------------------------
    eval_grams: set[int] = set()
    for row in eval_rows:
        eval_grams.update(ngram_hashes(toks_by_rid[row["rid"]]))
    eval_media = {str(s.get("sha256") or "").lower() for row in eval_rows for s in row["sidecars"] if s.get("sha256")}
    near = Counter()
    for row in rows:
        if assignment[row["rid"]] != "train":
            continue
        grams = ngram_hashes(toks_by_rid[row["rid"]]) if row["family"] == "text" else []
        if grams:
            shared = sum(1 for g in grams if g in eval_grams)
            if shared / len(grams) > NEAR_DUP_FRACTION:
                assignment[row["rid"]] = "eval_near_duplicate"
                near[row["route"]] += 1
                continue
        if any(str(s.get("sha256") or "").lower() in eval_media for s in row["sidecars"] if s.get("sha256")):
            assignment[row["rid"]] = "eval_media_shared"
            near[f"{row['route']}:media"] += 1

    # Duplicates inside the training pool: one representative per normalised text ------------
    representative: dict[str, str] = {}
    duplicates = 0
    for row in rows:
        if assignment[row["rid"]] != "train":
            continue
        first = representative.setdefault(row["dedup_key"], row["rid"])
        if first != row["rid"]:
            assignment[row["rid"]] = "duplicate"
            duplicates += 1

    status_counts = Counter(assignment.values())
    by_route = defaultdict(Counter)
    for row in rows:
        by_route[row["route"]][assignment[row["rid"]]] += 1
    eval_tokens = defaultdict(int)
    for row in eval_rows:
        eval_tokens[row["route"]] += row["tok_chat"]
    in_sample = sum(1 for row in rows if row["evidence_key"] in evidence_keys)
    eval_with_evidence = sum(1 for row in eval_rows if row["evidence_key"] in evidence_keys)

    eval_sha, eval_count = write_jsonl(out["eval"] / "eval_rows.jsonl", (
        {k: row[k] for k in ("rid", "uid", "record_id", "dataset_id", "route", "raw_route", "task_type", "instruction", "response", "sidecars", "tok_chat", "render", "license", "license_tier")}
        for row in eval_rows
    ))
    eval_manifest = {
        "schema": f"{SCHEMA}/eval_split",
        "identity": tool_identity(),
        "rule": f"salted hash ({EVAL_SALT}) < {EVAL_PER_10000}/10000 per row identity (normalised text; + sidecar digests for media rows), per-route floor min({EVAL_FLOOR}, rows//10) lowest hashes, plus the 28-row stage-T split, plus every duplicate (same normalised text) of an eval row; fold-evidence documents excluded (in-sample for the charts); review-surface hits removed",
        "rows": eval_count,
        "file": {"path": str(out["eval"] / "eval_rows.jsonl"), "sha256": eval_sha},
        "rows_by_route": dict(sorted(Counter(r["route"] for r in eval_rows).items())),
        "chat_tokens_by_route": dict(sorted(eval_tokens.items())),
        "stage_t_split": stage_t,
        "stage_t_rows_in_eval": sum(1 for r in eval_rows if r["uid"] in forced),
        "eval_rows_that_are_fold_evidence": eval_with_evidence,
        "never_in": "any training set built by this tool (text, long, k3, self-gen, vision-in, media)",
    }
    write_json(out["eval"] / "manifest.json", eval_manifest, seal_field="manifest_sha256")
    carve = {"schema": f"{SCHEMA}/carve", "identity": tool_identity(), "rows_sha256": file_sha256(out["rows"]), "assignment": assignment}
    carve_sha = write_json(out["carve"], carve, seal_field="carve_sha256")
    report = {
        "schema": f"{SCHEMA}/contamination_screen",
        "identity": tool_identity(),
        "rows_screened": len(rows),
        "rows_sha256": file_sha256(out["rows"]),
        "method": {
            "words": "NFKC, lower case, letter/digit runs; each CJK character is a word",
            "exact": f"normalised instruction / response / whole row equal to a normalised review item of >= {EXACT_ITEM_MIN_WORDS} words",
            "ngram": f"any word {NGRAM}-gram shared with any review item of >= {NGRAM} words",
            "short_item": f"whole-item containment of review items of {SHORT_ITEM_MIN_WORDS}..{NGRAM - 1} words",
            "literal": "case-insensitive containment of the export envelope needle codes",
            "media": "sidecar sha256 equal to a bank v2 vision-card asset",
            "eval_near_duplicate": f"a text-family training row sharing > {NEAR_DUP_FRACTION:.0%} of its {NGRAM}-grams with the eval split is held out; any training row sharing a media sidecar with an eval row is held out",
            "dedup": "one representative per identity (normalised text; media rows add their sidecar digests); an eval row's duplicates all join the eval split",
        },
        "sources": sources,
        "index": index.summary(),
        "liveness": liveness,
        "fold_evidence": {**evidence_report, "rows_matching": in_sample, "eval_rows_matching": eval_with_evidence, "note": "the fold evidence is in-sample for the charts and is never in the eval split; its rows stay training-eligible"},
        "eval_split": {k: eval_manifest[k] for k in ("rule", "rows", "rows_by_route", "chat_tokens_by_route", "stage_t_rows_in_eval")},
        "review_hits": {"rows": len(hits), "by_source": dict(Counter(r["source"] for h in hits for r in h["reasons"])), "by_rule": dict(Counter(r["rule"] for h in hits for r in h["reasons"])), "by_route": dict(Counter(h["route"] for h in hits)), "rows_list": hits},
        "held_out_near_eval": dict(near),
        "training_pool_duplicates_removed": duplicates,
        "marker_refused": len(marker_rids),
        "status_counts": dict(status_counts),
        "status_by_route": {route: dict(counter) for route, counter in sorted(by_route.items())},
        "carve": {"path": str(out["carve"]), "carve_sha256": carve_sha},
        "wall_seconds": round(time.time() - started, 2),
    }
    report_sha = write_json(out["screen"], report, seal_field="screen_sha256")
    log(f"screen: {dict(status_counts)}; eval rows {eval_count}; report {report_sha[:16]}")
    return report


def load_carve(out: dict[str, Path]) -> dict[str, str]:
    if not out["carve"].is_file():
        raise Refusal(f"{out['carve']} missing: run `screen` first")
    carve = load_json(out["carve"])
    if carve.get("rows_sha256") != file_sha256(out["rows"]):
        raise Refusal("carve.json was built over a different rows.jsonl; re-run `screen`")
    return carve["assignment"]


# --------------------------------------------------------------------------
# Step: allocate.
# --------------------------------------------------------------------------


def largest_remainder(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    keys = sorted(weights)
    norm = sum(weights.values())
    if norm <= 0:
        return {k: 0 for k in keys}
    exact = {k: total * weights[k] / norm for k in keys}
    out = {k: int(math.floor(v)) for k, v in exact.items()}
    rest = total - sum(out.values())
    for key in sorted(keys, key=lambda k: (-(exact[k] - out[k]), k))[:rest]:
        out[key] += 1
    return out


def interleave_by_deficit(queues: dict[str, list[Any]], size_of: Callable[[Any], int], target: int, weights: Mapping[str, float] | None = None) -> list[Any]:
    """Take items route by route so that every prefix carries each route at its share.

    ``queues`` are consumed from the FRONT; the route whose emitted tokens lag its share of the
    running total the most goes next (ties by route name)."""

    if weights is None:
        weights = {route: float(sum(size_of(i) for i in q)) for route, q in queues.items()}
    norm = sum(weights.values()) or 1.0
    share = {route: weights[route] / norm for route in queues}
    pointer = {route: 0 for route in queues}
    emitted = Counter()
    total = 0
    out: list[Any] = []
    while total < target:
        best = None
        best_value = None
        for route in sorted(queues):
            if pointer[route] >= len(queues[route]):
                continue
            value = share[route] * (total + 1) - emitted[route]
            if best is None or value > best_value:
                best, best_value = route, value
        if best is None:
            break
        item = queues[best][pointer[best]]
        pointer[best] += 1
        size = size_of(item)
        emitted[best] += size
        total += size
        out.append(item)
    return out


def pack_rows_first_fit(items: list[Any], size_of: Callable[[Any], int], *, rows: int, capacity: int, lookahead: int) -> tuple[list[list[Any]], list[Any]]:
    """Whole items into ``rows`` bins of ``capacity``: each bin takes the next item that fits from a
    window of ``lookahead`` pending items, in order.  Returns (bins, unused items)."""

    pending = list(items)
    bins: list[list[Any]] = []
    for _ in range(rows):
        room = capacity
        current: list[Any] = []
        while True:
            pick = None
            for position in range(min(lookahead, len(pending))):
                if size_of(pending[position]) <= room:
                    pick = position
                    break
            if pick is None:
                break
            item = pending.pop(pick)
            current.append(item)
            room -= size_of(item)
        if not current:
            break
        bins.append(current)
    return bins, pending


def allocate(args: argparse.Namespace) -> dict[str, Any]:
    out = out_paths(Path(args.out_dir))
    rows = load_rows(out)
    assignment = load_carve(out)
    pool = [r for r in rows if r["family"] == "text" and assignment[r["rid"]] == "train" and not r["sep_in_chat"] and not r["sep_in_raw"] and r["tok_raw"] > 0 and r["max_id_chat"] < TEXT_VOCAB_ROWS]
    excluded_separator = sum(1 for r in rows if r["family"] == "text" and assignment[r["rid"]] == "train" and (r["sep_in_chat"] or r["sep_in_raw"]))
    by_rid = {r["rid"]: r for r in pool}
    available = set(by_rid)
    route_tokens_raw = Counter()
    for r in pool:
        route_tokens_raw[r["route"]] += r["tok_raw"]
    log(f"training pool: {len(pool):,} text rows, {sum(route_tokens_raw.values()):,} raw tokens over {len(route_tokens_raw)} routes")

    # (b) long: 31 documents of same-route rows ------------------------------------------------
    eligible_routes = {route: tokens for route, tokens in route_tokens_raw.items() if tokens >= 3 * LONG_SEQ}
    docs_per_route = largest_remainder(LONG_DOCS, eligible_routes)
    long_docs: list[dict[str, Any]] = []
    budget = LONG_SEQ - LONG_RESERVE
    for route in sorted(docs_per_route):
        candidates = stable_shuffle([r for r in pool if r["route"] == route], key=lambda r: r["rid"], salt=f"{SEED}:long:{route}")
        candidates.sort(key=lambda r: 0 if r["tok_raw"] >= 1024 else 1)
        for doc_index in range(docs_per_route[route]):
            room = budget
            chosen: list[str] = []
            for r in candidates:
                if r["rid"] not in available:
                    continue
                cost = r["tok_raw"] + LONG_SECTION_OVERHEAD
                if cost <= room:
                    chosen.append(r["rid"])
                    available.discard(r["rid"])
                    room -= cost
                if room < 64:
                    break
            long_docs.append({"route": route, "rows": chosen, "estimated_tokens": budget - room})
    log(f"long: {len(long_docs)} docs {dict(Counter(d['route'] for d in long_docs))}")

    # (c) K3: 4 x 122 rows of 4,096, whole raw documents ----------------------------------------------
    k3_target = K3_PASSES * K3_ROWS_PER_PASS * K3_SEQ
    queues = {}
    for route in sorted(route_tokens_raw):
        members = [by_rid[rid] for rid in sorted(available) if by_rid[rid]["route"] == route and by_rid[rid]["tok_raw"] + 1 <= K3_SEQ]
        queues[route] = stable_shuffle(members, key=lambda r: r["rid"], salt=f"{SEED}:k3:{route}")
    candidates = interleave_by_deficit(queues, lambda r: r["tok_raw"] + 1, int(k3_target * 1.06))
    bins, unused = pack_rows_first_fit(candidates, lambda r: r["tok_raw"] + 1, rows=K3_PASSES * K3_ROWS_PER_PASS, capacity=K3_SEQ, lookahead=K3_LOOKAHEAD)
    k3_rows = [[r["rid"] for r in b] for b in bins]
    for b in bins:
        for r in b:
            available.discard(r["rid"])
    k3_qwen_real = sum(r["tok_raw"] + 1 for b in bins for r in b)
    log(f"k3: {len(k3_rows)} rows, {k3_qwen_real:,} real qwen tokens of {len(k3_rows) * K3_SEQ:,} positions; {len(unused)} candidates returned")

    # (a) text: 4,883 x 4,096 stream-packed chat documents -----------------------------------------
    text_target = TEXT_ROWS * TEXT_SEQ
    queues = {}
    for route in sorted(route_tokens_raw):
        members = [by_rid[rid] for rid in sorted(available) if by_rid[rid]["route"] == route]
        queues[route] = stable_shuffle(members, key=lambda r: r["rid"], salt=f"{SEED}:text:{route}")
    text_docs = interleave_by_deficit(queues, lambda r: r["tok_chat"] + 1, text_target)
    for r in text_docs:
        available.discard(r["rid"])
    text_tokens = sum(r["tok_chat"] + 1 for r in text_docs)
    log(f"text: {len(text_docs):,} docs, {text_tokens:,} tokens for a {text_target:,} target")

    # (f) self-gen prompts: a 1.0M-token sampling budget ---------------------------------------------
    def selfgen_budget(r: Mapping[str, Any]) -> int:
        return max(SELFGEN_MIN_BUDGET, min(SELFGEN_MAX_NEW, int(r["tok_response"])))

    queues = {}
    for route in sorted(route_tokens_raw):
        members = [by_rid[rid] for rid in sorted(available) if by_rid[rid]["route"] == route and by_rid[rid]["instruction"] and by_rid[rid]["response"] and 0 < by_rid[rid]["tok_prompt"] <= SELFGEN_MAX_PROMPT]
        queues[route] = stable_shuffle(members, key=lambda r: r["rid"], salt=f"{SEED}:selfgen:{route}")
    selfgen = interleave_by_deficit(queues, selfgen_budget, SELFGEN_BUDGET)
    for r in selfgen:
        available.discard(r["rid"])
    log(f"selfgen: {len(selfgen):,} prompts, budget {sum(selfgen_budget(r) for r in selfgen):,}")

    leftover = Counter()
    for rid in available:
        leftover[by_rid[rid]["route"]] += by_rid[rid]["tok_chat"]
    allocation = {
        "schema": f"{SCHEMA}/allocation",
        "identity": tool_identity(),
        "rows_sha256": file_sha256(out["rows"]),
        "carve_sha256": load_json(out["carve"]).get("carve_sha256"),
        "seed": SEED,
        "order": "long -> k3 -> text -> selfgen; every set disjoint; text-family training rows only (media-family rows feed the vision-in and media sets)",
        "pool": {"rows": len(pool), "raw_tokens_by_route": dict(sorted(route_tokens_raw.items())), "excluded_separator_inside": excluded_separator},
        "long": {"docs": long_docs, "docs_per_route": docs_per_route, "budget_per_doc": budget},
        "k3": {"rows": k3_rows, "qwen_real_tokens": k3_qwen_real},
        "text": {"docs": [r["rid"] for r in text_docs], "tokens_available": text_tokens, "target": text_target},
        "selfgen": {"prompts": [r["rid"] for r in selfgen], "budget_tokens": sum(selfgen_budget(r) for r in selfgen)},
        "leftover_chat_tokens_by_route": dict(sorted(leftover.items())),
        "leftover_rows": len(available),
    }
    sets = [set(r for d in long_docs for r in d["rows"]), set(r for b in k3_rows for r in b), set(allocation["text"]["docs"]), set(allocation["selfgen"]["prompts"])]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise Refusal(f"allocation is not disjoint between sets {i} and {j}")
    sha = write_json(out["allocation"], allocation, seal_field="allocation_sha256")
    log(f"allocation sealed {sha[:16]}; leftover {len(available):,} rows")
    return allocation


def load_allocation(out: dict[str, Path]) -> dict[str, Any]:
    if not out["allocation"].is_file():
        raise Refusal(f"{out['allocation']} missing: run `allocate` first")
    allocation = load_json(out["allocation"])
    if allocation.get("rows_sha256") != file_sha256(out["rows"]):
        raise Refusal("allocation.json was built over a different rows.jsonl; re-run `allocate`")
    return allocation


def provenance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Licence / provenance notes by dataset, read off the rows themselves."""

    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_dataset.setdefault(row["dataset_id"], {"rows": 0, "training_status": row["training_status"], "licenses": Counter(), "license_tiers": Counter(), "source_datasets": Counter()})
        entry["rows"] += 1
        if row.get("license"):
            entry["licenses"][row["license"]] += 1
        if row.get("license_tier"):
            entry["license_tiers"][row["license_tier"]] += 1
        if row.get("source_dataset"):
            entry["source_datasets"][row["source_dataset"]] += 1
    return {
        "registry": "curated master shard registry (MASTER_SHARD_INDEX_POINTER.json, 631 entries / 94,318 rows / 640 shards, walked 2026-09-02)",
        "registry_notes": "row-inspected curation May-Aug 2026; transforms of OpenThoughts3 (Apache-2.0), OpenCodeReasoning-2 / OpenMathReasoning (CC-BY-4.0), Nemotron-SFT-Agentic-v2, memory-trace rewrites (STAGE_D_FACTS); 'trained_*' states refer to the dead omnicoder -- every row is untrained for Chimera; licences cleared by Cyril 2026-10-12",
        "by_dataset": {
            name: {"rows": e["rows"], "training_status": e["training_status"], "licenses": dict(e["licenses"].most_common(6)), "license_tiers": dict(e["license_tiers"].most_common(4)), "source_datasets": dict(e["source_datasets"].most_common(6))}
            for name, e in sorted(by_dataset.items())
        },
    }


# --------------------------------------------------------------------------
# Packed-output screen (cross-boundary hits only exist after packing).
# --------------------------------------------------------------------------


def screen_packed(index: ReviewIndex, texts: Iterable[tuple[str, str]]) -> dict[str, Any]:
    hits = []
    checked = 0
    for label, text in texts:
        checked += 1
        found = index.hits(text)
        if found:
            hits.append({"unit": label, "reasons": found})
    return {"units_checked": checked, "units_hit": len(hits), "hits": hits[:50]}


# --------------------------------------------------------------------------
# Step: text (a).
# --------------------------------------------------------------------------


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> str:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)
    return file_sha256(path)


def route_table(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    routes = sorted({r["route"] for r in rows})
    if len(routes) > 254:
        raise Refusal("more than 254 routes do not fit uint8 route ids")
    return {route: i for i, route in enumerate(routes)}


def stream_pack(documents: list[list[int]], *, rows: int, seq: int, pad_id: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Concatenate documents and cut every ``seq`` positions.  Returns ids [rows, seq], the
    document index per position (-1 = pad) and per document its global [start, end) and kept length."""

    total = rows * seq
    ids = np.full(total, pad_id, dtype=np.int32)
    doc_of = np.full(total, -1, dtype=np.int32)
    spans: list[dict[str, int]] = []
    position = 0
    for index, document in enumerate(documents):
        if position >= total:
            break
        take = min(len(document), total - position)
        ids[position : position + take] = np.asarray(document[:take], dtype=np.int32)
        doc_of[position : position + take] = index
        spans.append({"doc": index, "start": position, "end": position + take, "length": len(document), "kept": take})
        position += take
    return ids.reshape(rows, seq), doc_of.reshape(rows, seq), spans


def build_text(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = {r["rid"]: r for r in load_rows(out)}
    allocation = load_allocation(out)
    qwen = QwenText(Path(args.qwen_dir))
    index, _ = build_review_index(args)
    chosen = [rows[rid] for rid in allocation["text"]["docs"]]
    routes = route_table(rows[rid] for rid in rows if rows[rid]["family"] == "text")
    texts = [qwen.chat(chat_messages(r["instruction"], r["response"])) if r["render"] == "chat" else raw_render(r["instruction"], r["response"]) for r in chosen]
    encoded = qwen.encode_many(texts)
    documents = []
    for r, ids in zip(chosen, encoded):
        if qwen.separator_id in ids:
            raise Refusal(f"text doc {r['rid']} carries the separator inside; the stage flags disagree with this encode")
        if ids and max(ids) >= TEXT_VOCAB_ROWS:
            raise Refusal(f"text doc {r['rid']} has id {max(ids)} outside the text rows")
        documents.append(ids + [qwen.separator_id])
    ids, doc_of, spans = stream_pack(documents, rows=TEXT_ROWS, seq=TEXT_SEQ, pad_id=qwen.separator_id)
    route_ids = np.full(ids.shape, 255, dtype=np.uint8)
    flat_routes = route_ids.reshape(-1)
    for span in spans:
        flat_routes[span["start"] : span["end"]] = routes[chosen[span["doc"]]["route"]]
    real_positions = int((doc_of >= 0).sum())
    shards = []
    out["text"].mkdir(parents=True, exist_ok=True)
    for shard_index, start in enumerate(range(0, TEXT_ROWS, TEXT_SHARD_ROWS)):
        stop = min(start + TEXT_SHARD_ROWS, TEXT_ROWS)
        path = out["text"] / f"text-{shard_index:05d}.safetensors"
        metadata = {"schema": f"{SCHEMA}/text", "tool_version": VERSION, "rows": f"{start}:{stop}", "sequence_length": str(TEXT_SEQ), "tokenizer_sha256": qwen.sha256, "separator_id": str(qwen.separator_id), "pad_id": str(qwen.separator_id), "route_table": json.dumps(routes, sort_keys=True), "route_pad": "255", "doc_index_pad": "-1"}
        tensors = {"input_ids": ids[start:stop], "route_ids": route_ids[start:stop], "doc_index": doc_of[start:stop], "real_lengths": (doc_of[start:stop] >= 0).sum(axis=1).astype(np.int32)}
        sha = write_safetensors(path, tensors, metadata)
        shards.append({"path": str(path), "sha256": sha, "rows": [start, stop], "tokens": int((doc_of[start:stop] >= 0).sum())})
    docs_table = []
    for span in spans:
        r = chosen[span["doc"]]
        docs_table.append({"doc": span["doc"], "rid": r["rid"], "uid": r["uid"], "record_id": r["record_id"], "dataset_id": r["dataset_id"], "route": r["route"], "render": r["render"], "start": span["start"], "end": span["end"], "length": span["length"], "kept": span["kept"], "row_first": span["start"] // TEXT_SEQ, "row_last": (span["end"] - 1) // TEXT_SEQ})
    docs_sha, _ = write_jsonl(out["text"] / "documents.jsonl", docs_table)
    tokens_by_route = Counter()
    for doc in docs_table:
        tokens_by_route[doc["route"]] += doc["kept"]
    first4m = Counter()
    for doc in docs_table:
        if doc["start"] < 4_000_000:
            first4m[doc["route"]] += min(doc["end"], 4_000_000) - doc["start"]
    post = screen_packed(index, ((f"row {i}", qwen.decode(ids[i].tolist())) for i in range(ids.shape[0])))
    used_rows = [chosen[s["doc"]] for s in spans]
    shortfall = TEXT_ROWS * TEXT_SEQ - real_positions
    manifest = {
        "schema": f"{SCHEMA}/text",
        "identity": tool_identity(),
        "set": "(a) text: Q-text teacher cache input, D1 (first 4M tokens) and D2 off-policy data",
        "design": "DISTILL_DESIGN §3 Q-text '20M registry text', §4 D1 'First 4M tokens of the C registry text set, packed 4,096', D2 '18.85M registry off-policy + 1.15M'",
        "sequence_length": TEXT_SEQ,
        "rows": TEXT_ROWS,
        "positions": TEXT_ROWS * TEXT_SEQ,
        "real_tokens": real_positions,
        "shortfall_tokens": shortfall,
        "documents": len(spans),
        "documents_cut_at_end": sum(1 for s in spans if s["kept"] < s["length"]),
        "documents_spanning_rows": sum(1 for d in docs_table if d["row_first"] != d["row_last"]),
        "tokens_by_route": dict(sorted(tokens_by_route.items())),
        "documents_by_route": dict(sorted(Counter(d["route"] for d in docs_table).items())),
        "first_4m_tokens_by_route": dict(sorted(first4m.items())),
        "route_table": routes,
        "render": "parent chat template, thinking OFF, [user: instruction, assistant: response]; rows lacking one side as plain text; each document closed by the separator <|endoftext|>",
        "packing": "stream: documents concatenated in allocation order and cut every 4,096 positions (a document may continue into the next row; doc_index / documents.jsonl give exact spans for segment masks); route order interleaved by token deficit so every prefix carries each route at its pool share",
        "tensors": {"input_ids": "int32 [rows, 4096]", "route_ids": "uint8 [rows, 4096] (255 pad)", "doc_index": "int32 [rows, 4096] (-1 pad)", "real_lengths": "int32 [rows]"},
        "all_ids_below": TEXT_VOCAB_ROWS,
        "shards": shards,
        "documents_file": {"path": str(out["text"] / "documents.jsonl"), "sha256": docs_sha},
        "tokenizer": qwen.report(),
        "post_pack_screen": post,
        "inputs": {"rows_sha256": allocation["rows_sha256"], "carve_sha256": allocation["carve_sha256"], "allocation_sha256": allocation.get("allocation_sha256"), "screen_sha256": load_json(out["screen"]).get("screen_sha256")},
        "provenance": provenance(used_rows),
        "wall_seconds": round(time.time() - started, 2),
    }
    if post["units_hit"]:
        raise Refusal(f"text set: packed rows hit the review surface: {post['hits'][:3]}")
    sha = write_json(out["text"] / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"text: {real_positions:,} tokens in {TEXT_ROWS} x {TEXT_SEQ}, {len(shards)} shards, shortfall {shortfall:,}; manifest {sha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: long (b).
# --------------------------------------------------------------------------

LONG_HEADER = "Below is a long reference file made of numbered sections, with a few archive notes placed between them. Read all of it, then answer the questions that follow it.\n\n"
LONG_NEEDLE = "Archive note {code}: the checksum recorded for section {section} is {value}."
LONG_QUESTION_NEEDLE = "What checksum does archive note {code} record for section {section}?"
LONG_QUESTION_PASSAGE = 'Which section begins with the words "{quote}"?'
LONG_ANSWER_RULE = "Answer each question on its own numbered line, giving only the answer."
LONG_DEPTHS = (0.1, 0.5, 0.9)
LONG_QUOTE_WORDS = 12


def long_codes(rng: random.Random) -> tuple[str, str]:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    code = "AN-" + "".join(rng.choice(letters) for _ in range(3)) + "-" + str(rng.randint(100, 999))
    value = str(rng.randint(100000, 999999))
    return code, value


def compose_long(sections: list[str], route: str, doc_index: int) -> dict[str, Any]:
    """One long document: header, numbered sections with three needles at the design depths,
    the questions, and the answers.  Deterministic in (route, doc_index, sections)."""

    rng = random.Random(f"{SEED}:long:{route}:{doc_index}")
    count = len(sections)
    if count < 4:
        raise Refusal(f"long doc {doc_index} ({route}) has {count} sections; a needle task needs at least 4")
    needle_after = sorted({min(count - 1, max(0, int(round(d * count)) - 1)) for d in LONG_DEPTHS})
    while len(needle_after) < len(LONG_DEPTHS):
        needle_after = sorted(set(needle_after) | {rng.randrange(count)})
    needles = []
    for slot in needle_after:
        code, value = long_codes(rng)
        target = rng.randrange(count) + 1
        needles.append({"after_section": slot + 1, "code": code, "value": value, "section": target, "text": LONG_NEEDLE.format(code=code, section=target, value=value)})
    starts = []
    for i, body in enumerate(sections):
        toks = body.split()
        starts.append(" ".join(toks[:LONG_QUOTE_WORDS]) if len(toks) >= LONG_QUOTE_WORDS else None)
    unique = [i for i, q in enumerate(starts) if q and starts.count(q) == 1]
    passage = None
    if unique:
        pick = unique[rng.randrange(len(unique))]
        passage = {"section": pick + 1, "quote": starts[pick]}
    parts = [LONG_HEADER]
    needle_iter = {n["after_section"]: n for n in needles}
    for i, body in enumerate(sections):
        parts.append(f"### Section {i + 1}\n{body}\n\n")
        if (i + 1) in needle_iter:
            parts.append(needle_iter[i + 1]["text"] + "\n\n")
    questions = [LONG_QUESTION_NEEDLE.format(code=n["code"], section=n["section"]) for n in needles]
    answers = [n["value"] for n in needles]
    if passage:
        questions.append(LONG_QUESTION_PASSAGE.format(quote=passage["quote"]))
        answers.append(f"Section {passage['section']}")
    parts.append("Questions:\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) + "\n" + LONG_ANSWER_RULE)
    user = "".join(parts)
    answer_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(answers))
    return {"user": user, "assistant": answer_text, "needles": needles, "passage": passage, "questions": questions, "answers": answers}


def build_long(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = {r["rid"]: r for r in load_rows(out)}
    allocation = load_allocation(out)
    qwen = QwenText(Path(args.qwen_dir))
    index, _ = build_review_index(args)
    codes_seen = set(index.literals)
    docs = allocation["long"]["docs"]
    ids_table = np.full((len(docs), LONG_SEQ), qwen.separator_id, dtype=np.int32)
    answer_mask = np.zeros((len(docs), LONG_SEQ), dtype=np.uint8)
    real_lengths = np.zeros(len(docs), dtype=np.int32)
    routes = route_table(r for r in rows.values() if r["family"] == "text")
    route_of_doc = np.zeros(len(docs), dtype=np.int32)
    tasks = []
    dropped_rows: list[str] = []
    for d, doc in enumerate(docs):
        members = list(doc["rows"])
        while True:
            sections = [raw_render(rows[rid]["instruction"], rows[rid]["response"]) for rid in members]
            composed = compose_long(sections, doc["route"], d)
            text = qwen.chat([{"role": "user", "content": composed["user"]}, {"role": "assistant", "content": composed["assistant"]}])
            ids, offsets = qwen.encode_with_offsets(text)
            if len(ids) + 1 <= LONG_SEQ:
                break
            dropped_rows.append(members.pop())
        if qwen.separator_id in ids or max(ids) >= TEXT_VOCAB_ROWS:
            raise Refusal(f"long doc {d}: separator inside or id outside the text rows")
        full = ids + [qwen.separator_id]
        ids_table[d, : len(full)] = full
        real_lengths[d] = len(full)
        route_of_doc[d] = routes[doc["route"]]
        answer_start = text.rfind(composed["assistant"])
        spans = []
        cursor = answer_start
        for i, answer in enumerate(composed["answers"]):
            line = f"{i + 1}. {answer}"
            begin = text.index(line, cursor) + len(f"{i + 1}. ")
            end = begin + len(answer)
            cursor = end
            token_positions = [t for t, (a, b) in enumerate(offsets) if a < end and b > begin]
            spans.append({"answer": answer, "chars": [begin, end], "tokens": [token_positions[0], token_positions[-1] + 1]})
            answer_mask[d, token_positions[0] : token_positions[-1] + 1] = 1
        for needle in composed["needles"]:
            begin = text.index(needle["text"])
            token = next(t for t, (a, b) in enumerate(offsets) if b > begin)
            needle["token_position"] = token
            needle["depth_realised"] = round(token / len(ids), 4)
            if needle["code"].lower() in codes_seen:
                raise Refusal(f"long doc {d}: needle code {needle['code']} collides with a review needle code")
        hit = index.hits(composed["user"] + "\n" + composed["assistant"])
        if hit:
            raise Refusal(f"long doc {d}: the composed document hits the review surface: {hit}")
        tasks.append({
            "doc": d, "route": doc["route"], "rows": members, "sections": len(members), "tokens": len(full),
            "needles": composed["needles"], "passage": composed["passage"], "questions": composed["questions"],
            "answers": composed["answers"], "answer_spans": spans,
            "uids": [rows[rid]["uid"] for rid in members], "record_ids": [rows[rid]["record_id"] for rid in members],
        })
    path = out["long"] / "long-00000.safetensors"
    sha = write_safetensors(path, {"input_ids": ids_table, "answer_mask": answer_mask, "real_lengths": real_lengths, "route_index": route_of_doc}, {"schema": f"{SCHEMA}/long", "tool_version": VERSION, "sequence_length": str(LONG_SEQ), "tokenizer_sha256": qwen.sha256, "separator_id": str(qwen.separator_id), "pad_id": str(qwen.separator_id), "route_table": json.dumps(routes, sort_keys=True)})
    tasks_sha = write_json(out["long"] / "tasks.json", {"schema": f"{SCHEMA}/long_tasks", "tasks": tasks}, seal_field="tasks_sha256")
    real = int(real_lengths.sum())
    used = [rows[rid] for t in tasks for rid in t["rows"]]
    manifest = {
        "schema": f"{SCHEMA}/long",
        "identity": tool_identity(),
        "set": "(b) long: Q-long teacher cache input and the D2 32k rung",
        "design": "DISTILL_DESIGN §3 Q-long: '1.0M tokens at L = 32,768: 31 synthetic documents of concatenated same-route rows plus needle and retrieval tasks'; the registry has 0 rows >= 32,768",
        "documents": len(tasks),
        "sequence_length": LONG_SEQ,
        "positions": len(tasks) * LONG_SEQ,
        "real_tokens": real,
        "shortfall_vs_1M_tokens": max(0, 1_000_000 - real),
        "documents_by_route": dict(sorted(Counter(t["route"] for t in tasks).items())),
        "tokens_by_route": {route: sum(t["tokens"] for t in tasks if t["route"] == route) for route in sorted({t["route"] for t in tasks})},
        "sections_total": sum(t["sections"] for t in tasks),
        "rows_dropped_to_fit": dropped_rows,
        "tasks": {"needles_per_doc": len(LONG_DEPTHS), "depths": list(LONG_DEPTHS), "passage_retrieval_docs": sum(1 for t in tasks if t["passage"]), "answers_recorded": sum(len(t["answers"]) for t in tasks)},
        "synthetic_text": {"header": LONG_HEADER, "needle": LONG_NEEDLE, "needle_question": LONG_QUESTION_NEEDLE, "passage_question": LONG_QUESTION_PASSAGE, "answer_rule": LONG_ANSWER_RULE, "note": "the only text this tool authors; the design specifies needle and retrieval tasks; screened against the review surface (0 hits) and needle codes checked against the envelope codes"},
        "render": "parent chat template, thinking OFF: [user: header + sections (raw instruction/blank line/response) + needles + questions, assistant: numbered answers]; separator appended; answer_mask marks the answer tokens",
        "files": [{"path": str(path), "sha256": sha}, {"path": str(out["long"] / "tasks.json"), "tasks_sha256": tasks_sha, "sha256": file_sha256(out["long"] / "tasks.json")}],
        "tokenizer": qwen.report(),
        "inputs": {"allocation_sha256": allocation.get("allocation_sha256"), "screen_sha256": load_json(out["screen"]).get("screen_sha256")},
        "provenance": provenance(used),
        "wall_seconds": round(time.time() - started, 2),
    }
    msha = write_json(out["long"] / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"long: {len(tasks)} docs, {real:,} real tokens, dropped {len(dropped_rows)} rows to fit; manifest {msha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: k3 (c).
# --------------------------------------------------------------------------


def bind_k3_corpus_build(args: argparse.Namespace) -> Any:
    tools = Path(args.composite_tools)
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import k3_corpus_build as kcb  # noqa: E402

    kcb.bind_lanes(qwen_fold_dir=args.qwen_fold_dir, chimera_src=str(resolve_chimera_src(args.chimera_src)))
    return kcb


def build_k3(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = {r["rid"]: r for r in load_rows(out)}
    allocation = load_allocation(out)
    qwen = QwenText(Path(args.qwen_dir))
    index, _ = build_review_index(args)
    kcb = bind_k3_corpus_build(args)
    k3_dir = Path(args.k3_dir)
    tik_sha = file_sha256(k3_dir / "tiktoken.model")
    if tik_sha != K3_TIKTOKEN_SHA256:
        raise Refusal(f"{k3_dir}/tiktoken.model hashes {tik_sha}, pinned {K3_TIKTOKEN_SHA256}")
    q_side, k_side, sides = kcb.load_sides(qwen_dir=Path(args.qwen_dir), k3_dir=k3_dir, qwen_separator_id=qwen.separator_id, k3_separator_id=None, k3_pad_id=None)
    bins = allocation["k3"]["rows"]
    expected = K3_PASSES * K3_ROWS_PER_PASS
    passes = []
    totals = Counter()
    used_rows = []
    for p in range(K3_PASSES):
        pass_bins = bins[p * K3_ROWS_PER_PASS : (p + 1) * K3_ROWS_PER_PASS]
        k3_rows, qwen_rows, metas, pair_rows, pair_q, pair_k = [], [], [], [], [], []
        k3_real, qwen_real = [], []
        for r_index, members in enumerate(pass_bins):
            docs = [qwen.encode(raw_render(rows[rid]["instruction"], rows[rid]["response"])) for rid in members]
            ids: list[int] = []
            for doc in docs:
                if qwen.separator_id in doc:
                    raise Refusal(f"k3 pass {p} row {r_index}: separator inside a document")
                ids.extend(doc + [qwen.separator_id])
            if len(ids) > K3_SEQ:
                raise Refusal(f"k3 pass {p} row {r_index}: {len(ids)} qwen ids > {K3_SEQ}; the allocation disagrees with this encode")
            padded = kcb.pad(ids, K3_SEQ, qwen.separator_id)
            label = f"k3 pass {p} row {r_index}"
            build = kcb.build_row(padded, qwen=q_side, k3=k_side, sequence_length=K3_SEQ, label=label, require_terminated=True)
            raw_bytes = bytes(build.raw)
            if k_side.decode_bytes(build.k3_ids) != raw_bytes:
                raise Refusal(f"{label}: K3 detokenize != raw bytes; refusing to write")
            k3_rows.append(kcb.pad(build.k3_ids, K3_SEQ, k_side.pad_id))
            qwen_rows.append(padded)
            k3_real.append(build.k3_real)
            qwen_real.append(build.qwen_real)
            for q, k in build.pairs:
                pair_rows.append(r_index)
                pair_q.append(q)
                pair_k.append(k)
            metas.append({"row_index": r_index, "rids": members, "uids": [rows[rid]["uid"] for rid in members], "routes": [rows[rid]["route"] for rid in members], "qwen_tokens_real": build.qwen_real, "k3_tokens_real": build.k3_real, "k3_truncated": build.truncated, "k3_tokens_dropped": build.k3_tokens_dropped, "aligned_pairs": len(build.pairs), "separator_pairs": build.separator_pairs, "aligned_fraction": len(build.pairs) / max(build.qwen_real, build.k3_real, 1), "raw_text_sha256": hashlib.sha256(raw_bytes).hexdigest(), "dropped_trailing_partial_utf8_bytes": build.dropped_trailing_partial_utf8_bytes})
            used_rows.extend(rows[rid] for rid in members)
        splits = ["fit"] * len(k3_rows)
        pass_dir = out["k3"] / f"pass-{p}"
        common = {"schema": kcb.SCHEMA_IDS, "distill_schema": f"{SCHEMA}/k3", "tool_version": VERSION, "pass": str(p), "in_band_tensors": ",".join(kcb.IN_BAND_TENSORS), "split_codes": json.dumps(kcb.SPLIT_CODES, sort_keys=True), "split_note": "every row is a distillation TRAINING row (code 0); eval rows were carved out before allocation"}
        files = {}
        for name, table, lengths, meta in (
            ("k3_input_ids", k3_rows, k3_real, {"tokenizer": "kimi_k3 tiktoken", "tiktoken_model_sha256": tik_sha, "separator_id": str(k_side.separator_id), "pad_id": str(k_side.pad_id)}),
            ("qwen_input_ids", qwen_rows, qwen_real, {"tokenizer": "qwen3_5 tokenizer.json", "tokenizer_json_sha256": qwen.sha256, "separator_id": str(qwen.separator_id), "pad_id": str(qwen.separator_id)}),
        ):
            tensors = kcb.in_band_tensors(table, splits, lengths)
            path = pass_dir / f"{name}.safetensors"
            kcb.write_ids(path, tensors, metadata={**common, **meta, "file": name, "rows": str(len(table)), "sequence_length": str(K3_SEQ)})
            files[name] = {"path": str(path), "sha256": file_sha256(path), "input_ids_sha256": kcb.content_sha256(tensors["input_ids"]), "shape": [len(table), K3_SEQ], "real_tokens": int(sum(lengths))}
        offsets = np.zeros(len(k3_rows) + 1, dtype=np.int64)
        counts = Counter(pair_rows)
        for i in range(len(k3_rows)):
            offsets[i + 1] = offsets[i] + counts.get(i, 0)
        bpath = pass_dir / "boundary_map.safetensors"
        bsha = write_safetensors(bpath, {"row": np.asarray(pair_rows, dtype=np.int32), "qwen_position": np.asarray(pair_q, dtype=np.int32), "k3_position": np.asarray(pair_k, dtype=np.int32), "row_offsets": offsets}, {"schema": f"{SCHEMA}/k3_boundary_map", "definition": "pairs (row, qwen_position, k3_position): token indices of the same row whose consumed bytes end at the same offset (k3_corpus_build.align_positions); separators paired when they close the same document", "qwen_file_sha256": files["qwen_input_ids"]["sha256"], "k3_file_sha256": files["k3_input_ids"]["sha256"]})
        files["boundary_map"] = {"path": str(bpath), "sha256": bsha, "pairs": len(pair_rows)}
        meta_sha = write_json(pass_dir / "row_meta.json", {"schema": f"{SCHEMA}/k3_row_meta", "pass": p, "rows": metas}, seal_field="row_meta_sha256")
        files["row_meta"] = {"path": str(pass_dir / "row_meta.json"), "sha256": file_sha256(pass_dir / "row_meta.json"), "row_meta_sha256": meta_sha}
        post = screen_packed(index, ((f"pass {p} row {i}", qwen.decode(qwen_rows[i][: qwen_real[i]])) for i in range(len(qwen_rows))))
        if post["units_hit"]:
            raise Refusal(f"k3 pass {p}: packed rows hit the review surface: {post['hits'][:3]}")
        summary = {
            "pass": p, "rows": len(k3_rows), "k3_positions": len(k3_rows) * K3_SEQ, "k3_real_tokens": int(sum(k3_real)), "qwen_real_tokens": int(sum(qwen_real)),
            "k3_rows_truncated": sum(1 for m in metas if m["k3_truncated"]), "k3_tokens_dropped": sum(m["k3_tokens_dropped"] for m in metas),
            "aligned_pairs": len(pair_rows), "aligned_fraction": len(pair_rows) / max(sum(qwen_real), sum(k3_real), 1),
            "files": files, "post_pack_screen": post,
        }
        bsummary_sha = write_json(pass_dir / "boundary_map.json", {"schema": kcb.SCHEMA_BOUNDARY_MAP, "distill_schema": f"{SCHEMA}/k3", "pass": p, "aggregate": {k: summary[k] for k in ("rows", "k3_real_tokens", "qwen_real_tokens", "aligned_pairs", "aligned_fraction")}, "pairs_file": files["boundary_map"], "ids_files_sha256": {n: files[n]["sha256"] for n in ("k3_input_ids", "qwen_input_ids")}, "rows": [{k: m[k] for k in ("row_index", "qwen_tokens_real", "k3_tokens_real", "aligned_pairs", "separator_pairs", "aligned_fraction")} for m in metas]}, seal_field="boundary_map_sha256")
        files["boundary_map_json"] = {"path": str(pass_dir / "boundary_map.json"), "sha256": file_sha256(pass_dir / "boundary_map.json"), "boundary_map_sha256": bsummary_sha}
        passes.append(summary)
        for key in ("rows", "k3_positions", "k3_real_tokens", "qwen_real_tokens", "aligned_pairs", "k3_rows_truncated", "k3_tokens_dropped"):
            totals[key] += summary[key]
        log(f"k3 pass {p}: {summary['rows']} rows, K3 real {summary['k3_real_tokens']:,} / {summary['k3_positions']:,} positions, qwen real {summary['qwen_real_tokens']:,}, aligned {summary['aligned_fraction']:.4f}")
    route_tokens = Counter()
    for r in used_rows:
        route_tokens[r["route"]] += r["tok_raw"] + 1
    manifest = {
        "schema": f"{SCHEMA}/k3",
        "identity": tool_identity(),
        "set": "(c) K3 rows: dual-tokenized (K3 tiktoken 163,840 + Qwen) teacher-forced K3 cache input + Q-text K3-row anchors + D2 K3 rows",
        "design": "DISTILL_DESIGN §3 K3 row: 'over 2.0M dual-tokenized tokens (boundary_map), run as 4 passes of <= 0.5M tokens'; STAGE_D_RULINGS: 2.0M in four passes",
        "format": "composite/corpus dual-tokenized format of k3_corpus_build.py (in-band input_ids / split / real_lengths / fit_mask / held_mask, int64), one directory per pass; pairs as int32 arrays in boundary_map.safetensors (the JSON pair lists of the 24-row evidence would be ~40 MB per pass)",
        "passes": passes,
        "rows_expected": expected,
        "totals": dict(totals),
        "pass_rule": f"{K3_ROWS_PER_PASS} rows x {K3_SEQ} = {K3_ROWS_PER_PASS * K3_SEQ:,} K3 positions per pass <= 500,000 (host attention-residual state per pass, design §3)",
        "shortfall_vs_2M_qwen_tokens": max(0, 2_000_000 - totals["qwen_real_tokens"]),
        "qwen_tokens_by_route": dict(sorted(route_tokens.items())),
        "render": "raw: instruction, blank line, response (the fold evidence's render) -- no chat markup, so K3 reads natural text; whole documents packed first-fit, each closed by the separator on both sides",
        "tokenizers": sides,
        "reused_code": {"k3_corpus_build": {"path": str(Path(kcb.__file__).resolve()), "sha256": file_sha256(kcb.__file__), "functions": ["load_sides", "build_row", "align_positions", "pad", "in_band_tensors", "content_sha256", "write_ids"]}},
        "inputs": {"allocation_sha256": allocation.get("allocation_sha256"), "screen_sha256": load_json(out["screen"]).get("screen_sha256")},
        "provenance": provenance(used_rows),
        "wall_seconds": round(time.time() - started, 2),
    }
    sha = write_json(out["k3"] / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"k3: {dict(totals)}; manifest {sha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: measure (S0 P10).
# --------------------------------------------------------------------------


def ffprobe(path: str, timeout: float = 60.0) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames,avg_frame_rate,r_frame_rate,duration,sample_rate,channels:format=duration", "-of", "json", path]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()[:300]}
    payload = json.loads(result.stdout or "{}")
    video = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None)

    def rate(text: str | None) -> float | None:
        if not text or text in ("0/0",):
            return None
        a, _, b = text.partition("/")
        return float(a) / float(b or 1) if float(b or 1) else None

    duration = payload.get("format", {}).get("duration")
    facts: dict[str, Any] = {"ok": True, "duration": float(duration) if duration else None}
    if video:
        facts.update({"width": video.get("width"), "height": video.get("height"), "fps": rate(video.get("avg_frame_rate")) or rate(video.get("r_frame_rate")), "nb_frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None})
    if audio:
        facts.update({"audio_sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None, "audio_channels": audio.get("channels")})
    facts["has_video"] = video is not None
    facts["has_audio"] = audio is not None
    return facts


def measure_file(path: str, kind: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"path": path, "kind": kind}
    try:
        stat = os.stat(path)
    except OSError as exc:
        return {**facts, "exists": False, "error": repr(exc)}
    facts["exists"] = True
    facts["bytes"] = stat.st_size
    facts["sha256"] = file_sha256(path)
    if kind == "image":
        from PIL import Image  # noqa: E402

        try:
            with Image.open(path) as image:
                facts["width"], facts["height"] = image.size
                facts["mode"] = image.mode
        except Exception as exc:
            facts["image_error"] = repr(exc)[:200]
    elif kind in ("video", "audio"):
        try:
            facts.update({"probe": ffprobe(path)})
        except subprocess.TimeoutExpired:
            facts["probe"] = {"ok": False, "error": "ffprobe timeout"}
    return facts


def visual_tokens_image(width: int, height: int, processor: Mapping[str, Any]) -> dict[str, int]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize  # noqa: E402

    config = processor["image_processor"]
    patch = int(config["patch_size"])
    merge = int(config["merge_size"])
    h_bar, w_bar = smart_resize(int(height), int(width), factor=patch * merge, min_pixels=int(config["size"]["shortest_edge"]), max_pixels=int(config["size"]["longest_edge"]))
    tokens = (h_bar // patch) * (w_bar // patch) // (merge * merge)
    return {"resized_height": h_bar, "resized_width": w_bar, "visual_tokens": tokens}


def visual_tokens_video(width: int, height: int, total_frames: int, source_fps: float, processor: Mapping[str, Any]) -> dict[str, int]:
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize  # noqa: E402

    config = processor["video_processor"]
    patch = int(config["patch_size"])
    merge = int(config["merge_size"])
    temporal = int(config["temporal_patch_size"])
    # Qwen3VLVideoProcessor.sample_frames with the default fps.
    num_frames = int(total_frames / source_fps * float(config["fps"]))
    num_frames = min(max(num_frames, int(config["min_frames"])), int(config["max_frames"]), int(total_frames))
    h_bar, w_bar = smart_resize(num_frames=num_frames, height=int(height), width=int(width), temporal_factor=temporal, factor=patch * merge, min_pixels=int(config["size"]["shortest_edge"]), max_pixels=int(config["size"]["longest_edge"]))
    grid_t = math.ceil(num_frames / temporal)
    tokens = grid_t * (h_bar // patch) * (w_bar // patch) // (merge * merge)
    return {"sampled_frames": num_frames, "grid_t": grid_t, "resized_height": h_bar, "resized_width": w_bar, "visual_tokens": tokens}


def media_rows(rows: list[dict[str, Any]], assignment: Mapping[str, str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["family"] == "media" and assignment[r["rid"]] in ("train", "duplicate")]


def ltx_suspect(row: Mapping[str, Any]) -> bool:
    """v9_base video rows are generated ops-chat artifacts whose generator is unrecorded: the
    design's 'LTX-rollout v9_base rows' -- excluded from every media set."""

    return row["dataset_id"].startswith("v9_base") and any(s.get("media") == "video" for s in row["sidecars"])


def measure(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = load_rows(out)
    assignment = load_carve(out)
    targets: dict[str, str] = {}
    for row in media_rows(rows, assignment):
        if ltx_suspect(row):
            continue
        for s in row["sidecars"]:
            if s.get("media") in ("image", "video"):
                targets.setdefault(s["path"], s["media"])
    log(f"measure: {len(targets):,} image/video sidecars")
    results: dict[str, dict[str, Any]] = {}
    with futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        jobs = {pool.submit(measure_file, path, kind): path for path, kind in targets.items()}
        for job in futures.as_completed(jobs):
            path = jobs[job]
            try:
                results[path] = job.result(timeout=300)
            except Exception as exc:
                results[path] = {"path": path, "exists": False, "error": repr(exc)[:200]}
    payload = {"schema": f"{SCHEMA}/media_measurements", "identity": tool_identity(), "probe": "S0 P10: os.stat bytes, full-read sha256, PIL header dims, ffprobe streams", "files": dict(sorted(results.items())), "counts": dict(Counter(("ok" if r.get("exists") else "missing") + ":" + r.get("kind", "?") for r in results.values())), "wall_seconds": round(time.time() - started, 2)}
    write_json(out["measure"], payload, seal_field="measurements_sha256")
    log(f"measure: {payload['counts']} in {payload['wall_seconds']} s")
    return payload


def load_measure(out: dict[str, Path]) -> dict[str, dict[str, Any]]:
    if not out["measure"].is_file():
        raise Refusal(f"{out['measure']} missing: run `measure` first")
    return load_json(out["measure"])["files"]


def store_media(path: str, sha256: str, store: Path) -> str:
    """Content-addressed copy under ``media_store`` (sha256 verified)."""

    destination = store / sha256[:2] / f"{sha256}.{media_ext(path) or 'bin'}"
    if destination.is_file() and file_sha256(destination) == sha256:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(path, temporary)
    if file_sha256(temporary) != sha256:
        temporary.unlink(missing_ok=True)
        raise Refusal(f"{path} changed while it was being stored")
    os.replace(temporary, destination)
    return str(destination)


def usable(facts: Mapping[str, Any] | None, kind: str) -> bool:
    if not facts or not facts.get("exists"):
        return False
    if kind == "image":
        return bool(facts.get("width")) and bool(facts.get("height"))
    probe = facts.get("probe") or {}
    if kind == "video":
        return bool(probe.get("ok")) and bool(probe.get("has_video")) and bool(probe.get("width"))
    return bool(probe.get("ok")) and bool(probe.get("has_audio"))


# --------------------------------------------------------------------------
# Step: vision (d).
# --------------------------------------------------------------------------


def build_vision(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = load_rows(out)
    assignment = load_carve(out)
    facts = load_measure(out)
    processor = load_json(Path(args.qwen_dir) / "processor_config.json")
    qwen = QwenText(Path(args.qwen_dir))
    candidates = [r for r in media_rows(rows, assignment) if assignment[r["rid"]] == "train" and r["route"] in ("image", "ocr", "video", "multimodal", "media_flow", "image_grounding", "ocr_document") and not ltx_suspect(r)]
    items = []
    skipped = Counter()
    for row in candidates:
        media = [s for s in row["sidecars"] if s.get("media") in ("image", "video")]
        if not media or not row["instruction"]:
            skipped["no_media_or_no_instruction"] += 1
            continue
        entries = []
        ok = True
        for s in media:
            f = facts.get(s["path"])
            if not usable(f, s["media"]):
                ok = False
                break
            entry = {"path": s["path"], "media": s["media"], "role": s.get("role"), "bytes": f["bytes"], "sha256": f["sha256"], "declared_sha256": s.get("sha256"), "sha256_matches_declared": (s.get("sha256") or f["sha256"]).lower() == f["sha256"]}
            try:
                if s["media"] == "image":
                    entry.update({"width": f["width"], "height": f["height"], **visual_tokens_image(f["width"], f["height"], processor)})
                else:
                    probe = f["probe"]
                    frames = probe.get("nb_frames") or int(round((probe.get("duration") or 0) * (probe.get("fps") or 0)))
                    entry.update({"width": probe["width"], "height": probe["height"], "fps": probe.get("fps"), "frames": frames, "duration": probe.get("duration"), **visual_tokens_video(probe["width"], probe["height"], frames, probe.get("fps") or 24.0, processor)})
            except ValueError as exc:
                skipped[f"processor_refused:{str(exc)[:60]}"] += 1
                ok = False
                break
            entries.append(entry)
        if not ok:
            skipped["sidecar_missing_or_unreadable_or_refused"] += 1
            continue
        items.append({"rid": row["rid"], "uid": row["uid"], "record_id": row["record_id"], "dataset_id": row["dataset_id"], "route": row["route"], "task_type": row["task_type"], "instruction": row["instruction"], "reference_response": row["response"], "reference_response_note": "registry answer kept for stage T; the D3d KD target is the teacher's own answer (generated in the Q-text cache)", "media": entries, "has_video": any(e["media"] == "video" for e in entries)})
    image_items = [i for i in items if not i["has_video"]]
    video_items = stable_shuffle([i for i in items if i["has_video"]], key=lambda i: i["rid"], salt=f"{SEED}:video_in")
    seen_video: set[str] = set()
    video_in = []
    for item in video_items:
        key = next(e["sha256"] for e in item["media"] if e["media"] == "video")
        if key in seen_video:
            continue
        seen_video.add(key)
        video_in.append(item)
        if len(video_in) >= VIDEO_IN_CLIPS:
            break
    store = out["media_store"]
    for item in image_items + video_in:
        for entry in item["media"]:
            entry["stored_path"] = store_media(entry["path"], entry["sha256"], store)
    prompts = [qwen.chat([{"role": "user", "content": i["instruction"]}], generation_prompt=True) for i in image_items + video_in]
    prompt_tokens = qwen.encode_many(prompts)
    for item, ids in zip(image_items + video_in, prompt_tokens):
        item["prompt_text_tokens"] = len(ids)
        item["visual_tokens"] = sum(e["visual_tokens"] for e in item["media"])
    for item in video_in:
        item["design_video_tokens"] = 8 * 256
    vision_sha, _ = write_jsonl(out["vision"] / "vision_in.jsonl", image_items)
    video_sha, _ = write_jsonl(out["vision"] / "video_in.jsonl", video_in)
    vis_tokens = [i["visual_tokens"] for i in image_items]
    manifest = {
        "schema": f"{SCHEMA}/vision_in",
        "identity": tool_identity(),
        "set": "(d) vision-in and video-in prompts: the teacher (parent model.visual + streamed LLM, tool (i)) generates the answers in the Q-text cache; D3d trains on them",
        "design": "DISTILL_DESIGN §3 Q-text vision rows, §4 D3d, §5.6 (P10 measures image sizes; visual-token cap = the parent processor's default max pixels per STAGE_D_RULINGS)",
        "processor": {"path": str(Path(args.qwen_dir) / "processor_config.json"), "sha256": file_sha256(Path(args.qwen_dir) / "processor_config.json"), "image": processor["image_processor"], "video": processor["video_processor"], "rule": "transformers qwen2_vl smart_resize (image) / qwen3_vl smart_resize + sample_frames at fps 2 (video); tokens = grid / merge^2; vision_start/end and timestamp text tokens not counted"},
        "vision_in": {"rows": len(image_items), "images": sum(len(i["media"]) for i in image_items), "visual_tokens": int(sum(vis_tokens)), "visual_tokens_min_max_mean": [int(min(vis_tokens)) if vis_tokens else 0, int(max(vis_tokens)) if vis_tokens else 0, round(float(np.mean(vis_tokens)), 1) if vis_tokens else 0.0], "prompt_text_tokens": int(sum(i["prompt_text_tokens"] for i in image_items)), "bytes": int(sum(e["bytes"] for i in image_items for e in i["media"])), "rows_by_route": dict(Counter(i["route"] for i in image_items)), "rows_by_task_type": dict(Counter(i["task_type"] for i in image_items).most_common(40)), "file": {"path": str(out["vision"] / "vision_in.jsonl"), "sha256": vision_sha}},
        "video_in": {"clips": len(video_in), "target": VIDEO_IN_CLIPS, "shortfall": max(0, VIDEO_IN_CLIPS - len(video_in)), "visual_tokens_processor_default": int(sum(i["visual_tokens"] for i in video_in)), "visual_tokens_design_8x256": len(video_in) * 8 * 256, "prompt_text_tokens": int(sum(i["prompt_text_tokens"] for i in video_in)), "bytes": int(sum(e["bytes"] for i in video_in for e in i["media"])), "file": {"path": str(out["vision"] / "video_in.jsonl"), "sha256": video_sha}},
        "skipped": dict(skipped),
        "excluded": {"ltx_suspect_v9_base_video_rows": sum(1 for r in rows if r["family"] == "media" and ltx_suspect(r)), "eval_or_screened_rows": sum(1 for r in rows if r["family"] == "media" and assignment[r["rid"]] not in ("train", "duplicate"))},
        "media_store": str(store),
        "sha256_mismatch_vs_declared": sum(1 for i in image_items + video_in for e in i["media"] if not e["sha256_matches_declared"]),
        "provenance": provenance([r for r in rows if r["rid"] in {i["rid"] for i in image_items + video_in}]),
        "wall_seconds": round(time.time() - started, 2),
    }
    sha = write_json(out["vision"] / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"vision: {len(image_items)} image rows ({manifest['vision_in']['visual_tokens']:,} visual tokens), {len(video_in)} video clips; manifest {sha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: media (e).
# --------------------------------------------------------------------------


def ffmpeg(command: list[str], timeout: float = 300.0) -> None:
    result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", *command], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:400])


def cut_h3_clip(source: str, start: float, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(H3_SECONDS * H3_FPS))
    ffmpeg(["-ss", f"{start:.3f}", "-i", source, "-t", f"{H3_SECONDS:.3f}", "-an", "-vf", f"fps={H3_FPS},scale={H3_WIDTH}:{H3_HEIGHT}:force_original_aspect_ratio=increase,crop={H3_WIDTH}:{H3_HEIGHT}", "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(destination)])
    probe = ffprobe(str(destination))
    count = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(destination)], capture_output=True, text=True, timeout=120).stdout.strip()
    probe["frames_counted"] = int(count) if count.isdigit() else None
    return probe


def cut_audio(source: str, start: float, seconds: float, destination: Path, codec: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-ss", f"{start:.3f}", "-i", source, "-t", f"{seconds:.3f}", "-vn", "-map", "0:a:0", "-af", f"apad=whole_dur={seconds:.3f}", "-c:a", codec, str(destination)])
    return ffprobe(str(destination))


def centered_start(duration: float | None, seconds: float) -> float:
    if not duration or duration <= seconds:
        return 0.0
    return round((duration - seconds) / 2.0, 3)


def build_media(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = load_rows(out)
    assignment = load_carve(out)
    facts = load_measure(out)
    pool = [r for r in media_rows(rows, assignment) if assignment[r["rid"]] == "train" and not ltx_suspect(r)]
    excluded_ltx = [r["rid"] for r in rows if r["family"] == "media" and ltx_suspect(r)]
    store = out["media_store"]
    media_root = out["media"]
    report: dict[str, Any] = {}

    def images_of(row: Mapping[str, Any], roles: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        found = []
        for s in row["sidecars"]:
            if s.get("media") != "image":
                continue
            if roles is not None and s.get("role") not in roles:
                continue
            f = facts.get(s["path"])
            if usable(f, "image"):
                found.append({"path": s["path"], "sha256": f["sha256"], "bytes": f["bytes"], "width": f["width"], "height": f["height"], "role": s.get("role")})
        return found

    # --- QIMG: (image, caption) pairs, captions verbatim from the rows ----------------------
    def qimg_tier(row: Mapping[str, Any]) -> int:
        task = row["task_type"]
        if "description" in task:
            return 0
        if "grounding" in task:
            return 1
        if "ocr" in task or row["route"] == "ocr":
            return 2
        if "editing" in task:
            return 3
        return 4

    qimg: list[dict[str, Any]] = []
    used_sha: set[str] = set()
    for row in sorted(stable_shuffle(pool, key=lambda r: r["rid"], salt=f"{SEED}:qimg"), key=qimg_tier):
        if len(qimg) >= QIMG_PROMPTS:
            break
        if row["route"] not in ("image", "ocr", "image_grounding", "ocr_document") or not row["response"]:
            continue
        roles = ("edited", "target") if "editing" in row["task_type"] else None
        for image in images_of(row, roles):
            if image["sha256"] in used_sha:
                continue
            negative = row.get("negative_prompt")
            qimg.append({"rid": row["rid"], "uid": row["uid"], "dataset_id": row["dataset_id"], "route": row["route"], "task_type": row["task_type"], "tier": ["description", "grounding", "ocr", "editing", "other"][qimg_tier(row)], "prompt": row["response"], "prompt_source": "row response verbatim (the caption / description of this image)", "negative_prompt": negative if negative is not None else QIMG_DEFAULT_NEGATIVE, "negative_source": "row" if negative is not None else QIMG_NEGATIVE_SOURCE, "image": {**image, "stored_path": store_media(image["path"], image["sha256"], store)}, "target_resolution": [512, 512]})
            used_sha.add(image["sha256"])
            break
    report["qimg"] = {"items": len(qimg), "target": QIMG_PROMPTS, "shortfall": max(0, QIMG_PROMPTS - len(qimg)), "by_tier": dict(Counter(i["tier"] for i in qimg)), "rows_with_own_negative": sum(1 for i in qimg if i["negative_source"] == "row")}

    # --- HY3D: object images -------------------------------------------------------------
    def hy3d_tier(row: Mapping[str, Any], image: Mapping[str, Any]) -> int:
        if row["dataset_id"].startswith("v9_base"):
            return 0  # single-object application icons
        if "editing" in row["task_type"] and image.get("role") == "source":
            return 1  # panels built around one central object
        if "grounding" in row["task_type"]:
            return 2
        return 3

    hy_candidates = []
    for row in stable_shuffle([r for r in pool if r["route"] in ("image", "image_grounding")], key=lambda r: r["rid"], salt=f"{SEED}:hy3d"):
        for image in images_of(row):
            hy_candidates.append((hy3d_tier(row, image), row, image))
    hy_candidates.sort(key=lambda t: t[0])
    hy3d = []
    hy_seen: set[str] = set()
    for tier, row, image in hy_candidates:
        if len(hy3d) >= HY3D_IMAGES:
            break
        if image["sha256"] in hy_seen:
            continue
        hy_seen.add(image["sha256"])
        hy3d.append({"rid": row["rid"], "uid": row["uid"], "dataset_id": row["dataset_id"], "task_type": row["task_type"], "tier": ["icon", "edit_source_central_object", "grounding_panel", "other"][tier], "caption": row["response"], "image": {**image, "stored_path": store_media(image["path"], image["sha256"], store)}})
    report["hy3d"] = {"items": len(hy3d), "target": HY3D_IMAGES, "shortfall": max(0, HY3D_IMAGES - len(hy3d)), "by_tier": dict(Counter(i["tier"] for i in hy3d)), "overlap_with_qimg": len(hy_seen & used_sha), "note": "the registry carries no photographed-object corpus; object images are application icons, then edit-source panels built around one central object, then grounding panels"}

    # --- ViT feature images: every usable distinct image -------------------------------------
    vit = []
    vit_seen: set[str] = set()
    for row in stable_shuffle([r for r in pool if r["route"] in ("image", "ocr", "image_grounding", "ocr_document")], key=lambda r: r["rid"], salt=f"{SEED}:vit"):
        for image in images_of(row):
            if image["sha256"] in vit_seen:
                continue
            vit_seen.add(image["sha256"])
            vit.append({"rid": row["rid"], "dataset_id": row["dataset_id"], "route": row["route"], "image": {**image, "stored_path": store_media(image["path"], image["sha256"], store)}})
    report["vit"] = {"items": len(vit), "note": "every usable distinct curated image (image + ocr routes, eval-split images excluded); the QVL32 ViT feature cache reads block-13 in/out on these"}

    # --- H3: 2 s @ 24 fps 512x320 clips + a paired audio track -----------------------------------
    video_candidates = []
    video_seen: set[str] = set()
    for row in stable_shuffle([r for r in pool if r["route"] in ("video", "media_flow", "video_media_flow")], key=lambda r: r["rid"], salt=f"{SEED}:h3"):
        for s in row["sidecars"]:
            f = facts.get(s["path"])
            if s.get("media") == "video" and usable(f, "video") and f["sha256"] not in video_seen:
                video_seen.add(f["sha256"])
                video_candidates.append((row, s, f))
                break
    audio_rows = stable_shuffle([r for r in pool if r["route"] in ("audio", "speech", "tts", "sound")], key=lambda r: r["rid"], salt=f"{SEED}:h3audio")
    audio_iter = iter([(r, s) for r in audio_rows for s in r["sidecars"] if s.get("media") == "audio"])
    h3 = []
    h3_failures = []
    for row, s, f in video_candidates:
        if len(h3) >= H3_CLIPS:
            break
        probe = f["probe"]
        duration = probe.get("duration") or 0.0
        if duration < H3_SECONDS:
            h3_failures.append({"rid": row["rid"], "reason": f"source {duration:.2f} s < {H3_SECONDS} s"})
            continue
        clip_index = len(h3)
        clip_path = media_root / "h3" / "clips" / f"clip-{clip_index:04d}.mp4"
        start = centered_start(duration, H3_SECONDS)
        try:
            clip_probe = cut_h3_clip(s["path"], start, clip_path)
        except Exception as exc:
            h3_failures.append({"rid": row["rid"], "reason": repr(exc)[:200]})
            continue
        if clip_probe.get("width") != H3_WIDTH or clip_probe.get("height") != H3_HEIGHT or clip_probe.get("frames_counted") != int(H3_SECONDS * H3_FPS):
            h3_failures.append({"rid": row["rid"], "reason": f"clip probe {clip_probe}"})
            clip_path.unlink(missing_ok=True)
            continue
        audio_entry = None
        while audio_entry is None:
            try:
                arow, aside = next(audio_iter)
            except StopIteration:
                break
            if not Path(aside["path"]).is_file():
                continue
            apath = media_root / "h3" / "audio" / f"clip-{clip_index:04d}.wav"
            try:
                aprobe_src = ffprobe(aside["path"])
                astart = centered_start(aprobe_src.get("duration"), H3_SECONDS)
                aprobe = cut_audio(aside["path"], astart, H3_SECONDS, apath, "pcm_s16le")
            except Exception:
                continue
            audio_entry = {"source_rid": arow["rid"], "source_route": arow["route"], "source_path": aside["path"], "source_duration": aprobe_src.get("duration"), "start": astart, "path": str(apath), "sha256": file_sha256(apath), "probe": aprobe, "caption": arow["response"][:2000], "pairing": "deterministic assignment of a curated audio/speech/tts/sound sidecar (design §4 D3: 'audio/tts/speech paired into the H3 audio channel'); not temporally related to the video"}
        h3.append({"clip": clip_index, "rid": row["rid"], "uid": row["uid"], "dataset_id": row["dataset_id"], "task_type": row["task_type"], "prompt": row["response"], "prompt_source": "row response verbatim (the description of this clip)", "source": {"path": s["path"], "sha256": f["sha256"], "duration": duration, "fps": probe.get("fps"), "width": probe.get("width"), "height": probe.get("height")}, "start": start, "video": {"path": str(clip_path), "sha256": file_sha256(clip_path), "probe": clip_probe}, "audio": audio_entry})
    report["h3"] = {"items": len(h3), "target": H3_CLIPS, "shortfall": max(0, H3_CLIPS - len(h3)), "with_audio": sum(1 for c in h3 if c["audio"]), "distinct_source_videos_available": len(video_candidates), "failures": h3_failures[:40], "failure_count": len(h3_failures), "clip_spec": {"seconds": H3_SECONDS, "fps": H3_FPS, "width": H3_WIDTH, "height": H3_HEIGHT, "orientation_note": "the design's '320x512' read as H 320 x W 512 (landscape, the curated videos are landscape)", "transform": "centred 2 s window; fps filter to 24 (frames repeated from the 16 fps sources); scale to cover + centre crop; libx264 crf 16 yuv420p"}, "source_fps_seen": dict(Counter(str(c["source"]["fps"]) for c in h3))}

    # --- ACE: 1,024 music clips + captions ---------------------------------------------------
    def ace_tier(row: Mapping[str, Any]) -> int:
        task = row["task_type"]
        if task == "music_generation_and_understanding":
            return 0
        if task in ("music_structure_descriptor", "music_caption_and_prompt", "music_mastering_descriptor"):
            return 1
        if row["dataset_id"].startswith("v9_base"):
            return 2
        if "mastering" in task:
            return 3
        if "operator" in task or "synthetic" in task:
            return 5
        return 4

    ace = []
    ace_failures = []
    ace_seen: set[str] = set()
    music_rows = sorted(stable_shuffle([r for r in pool if r["route"] == "music"], key=lambda r: r["rid"], salt=f"{SEED}:ace"), key=ace_tier)
    for row in music_rows:
        if len(ace) >= ACE_CLIPS:
            break
        sidecars = [s for s in row["sidecars"] if s.get("media") == "audio" and s.get("role") not in ("degraded_input", "rough_mix", "stem")]
        if not sidecars:
            continue
        s = sidecars[0]
        key = str(s.get("sha256") or s["path"])
        if key in ace_seen or not Path(s["path"]).is_file():
            continue
        try:
            source_probe = ffprobe(s["path"])
            start = centered_start(source_probe.get("duration"), ACE_SECONDS)
            path = media_root / "ace" / "clips" / f"ace-{len(ace):04d}.flac"
            clip_probe = cut_audio(s["path"], start, ACE_SECONDS, path, "flac")
        except Exception as exc:
            ace_failures.append({"rid": row["rid"], "reason": repr(exc)[:200]})
            continue
        ace_seen.add(key)
        ace.append({"clip": len(ace), "rid": row["rid"], "uid": row["uid"], "dataset_id": row["dataset_id"], "task_type": row["task_type"], "tier": ["jamendo_tags", "descriptor_caption", "ace_prompt_v9_base", "mastering_clean_target", "other", "synthetic_operator"][ace_tier(row)], "caption": row["instruction"], "caption_source": "row instruction verbatim (tags / lyrics / descriptors)", "reference_response": row["response"][:2000], "source": {"path": s["path"], "declared_sha256": s.get("sha256"), "role": s.get("role"), "duration": source_probe.get("duration"), "sample_rate": source_probe.get("audio_sample_rate"), "channels": source_probe.get("audio_channels")}, "start": start, "seconds": ACE_SECONDS, "padded": bool(source_probe.get("duration") and source_probe["duration"] < ACE_SECONDS), "clip": {"path": str(path), "sha256": file_sha256(path), "probe": clip_probe}})
    report["ace"] = {"items": len(ace), "target": ACE_CLIPS, "shortfall": max(0, ACE_CLIPS - len(ace)), "by_tier": dict(Counter(c["tier"] for c in ace)), "padded_short_sources": sum(1 for c in ace if c["padded"]), "failures": ace_failures[:40], "failure_count": len(ace_failures), "clip_spec": {"seconds": ACE_SECONDS, "basis": "313 Oobleck latent frames per clip in the design's D4 arithmetic at 25 latent frames/s", "transform": "centred window, apad to full length, FLAC at the source rate"}, "note": "v9_base music rows are ACE-generated (music_tts_ace_fast_main): teacher-family rollouts, kept and labelled"}

    files = {}
    for name, items in (("qimg", qimg), ("hy3d", hy3d), ("vit", vit), ("h3", h3), ("ace", ace)):
        sha, count = write_jsonl(media_root / name / "items.jsonl", items)
        files[name] = {"path": str(media_root / name / "items.jsonl"), "sha256": sha, "items": count}
        sub_manifest = {"schema": f"{SCHEMA}/media/{name}", "identity": tool_identity(), "items_file": files[name], **report[name], "excluded": {"ltx_suspect_v9_base_video_rows": len(excluded_ltx), "eval_split_and_screened_rows": "never in pool"}, "provenance": provenance([r for r in rows if r["rid"] in {i["rid"] for i in items}])}
        if name in ("h3", "ace"):
            sub_manifest["clip_files"] = [{"path": i["video"]["path"] if name == "h3" else i["clip"]["path"], "sha256": i["video"]["sha256"] if name == "h3" else i["clip"]["sha256"]} for i in items]
            if name == "h3":
                sub_manifest["audio_files"] = [{"path": i["audio"]["path"], "sha256": i["audio"]["sha256"]} for i in items if i["audio"]]
        write_json(media_root / name / "manifest.json", sub_manifest, seal_field="manifest_sha256")
    manifest = {
        "schema": f"{SCHEMA}/media",
        "identity": tool_identity(),
        "set": "(e) media prompt / conditioning sets for the D3 teacher caches",
        "design": "DISTILL_DESIGN §3 (H3 512 clips 2 s @24 fps 320x512; QIMG 2,048 prompts + negatives; HY3D 256 object images; ACE 1,024 clips/captions; QVL32 ViT feature images); LTX-rollout v9_base rows excluded",
        "summary": {name: {k: v for k, v in report[name].items() if k in ("items", "target", "shortfall", "by_tier", "with_audio")} for name in report},
        "files": files,
        "media_store": str(store),
        "excluded_ltx_suspect_rows": excluded_ltx,
        "wall_seconds": round(time.time() - started, 2),
    }
    sha = write_json(media_root / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"media: {json.dumps(manifest['summary'])}; manifest {sha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: selfgen (f).
# --------------------------------------------------------------------------


def build_selfgen(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = out_paths(Path(args.out_dir))
    rows = {r["rid"]: r for r in load_rows(out)}
    allocation = load_allocation(out)
    qwen = QwenText(Path(args.qwen_dir))
    index, _ = build_review_index(args)
    items = []
    for rid in allocation["selfgen"]["prompts"]:
        r = rows[rid]
        messages = [{"role": "user", "content": r["instruction"]}]
        prompt = qwen.chat(messages, generation_prompt=True)
        ids = qwen.encode(prompt)
        budget = max(SELFGEN_MIN_BUDGET, min(SELFGEN_MAX_NEW, int(r["tok_response"])))
        items.append({"rid": rid, "uid": r["uid"], "record_id": r["record_id"], "dataset_id": r["dataset_id"], "route": r["route"], "messages": messages, "prompt_text": prompt, "prompt_tokens": len(ids), "reference_response_tokens": r["tok_response"], "budget_tokens": budget, "max_new_tokens": SELFGEN_MAX_NEW})
    post = screen_packed(index, ((i["rid"], i["prompt_text"]) for i in items))
    if post["units_hit"]:
        raise Refusal(f"selfgen prompts hit the review surface: {post['hits'][:3]}")
    sha, count = write_jsonl(out["selfgen"] / "prompts.jsonl", items)
    budget = sum(i["budget_tokens"] for i in items)
    manifest = {
        "schema": f"{SCHEMA}/selfgen",
        "identity": tool_identity(),
        "set": "(f) self-gen prompts: free-run sampling by the parent's Q4_K_M on the P40s, logits rescored in bf16",
        "design": "DISTILL_DESIGN §3 'Self-gen text: 1.0M sampled tokens'; STAGE_D_RULINGS U7",
        "prompts": count,
        "budget_tokens": budget,
        "budget_target": SELFGEN_BUDGET,
        "budget_rule": f"per prompt: the registry answer's token count clamped to [{SELFGEN_MIN_BUDGET}, {SELFGEN_MAX_NEW}] (the expected sample length); prompts are taken until the sum reaches 1.0M; max_new_tokens {SELFGEN_MAX_NEW}",
        "prompt_tokens": sum(i["prompt_tokens"] for i in items),
        "prompts_by_route": dict(sorted(Counter(i["route"] for i in items).items())),
        "budget_by_route": {route: sum(i["budget_tokens"] for i in items if i["route"] == route) for route in sorted({i["route"] for i in items})},
        "render": "parent chat template, thinking OFF, add_generation_prompt=True (the prompt ends with an empty <think></think> block)",
        "file": {"path": str(out["selfgen"] / "prompts.jsonl"), "sha256": sha},
        "post_screen": post,
        "disjoint_from": "text, long, k3 sets and the eval split",
        "inputs": {"allocation_sha256": allocation.get("allocation_sha256")},
        "provenance": provenance([rows[i["rid"]] for i in items]),
        "wall_seconds": round(time.time() - started, 2),
    }
    msha = write_json(out["selfgen"] / "manifest.json", manifest, seal_field="manifest_sha256")
    log(f"selfgen: {count} prompts, budget {budget:,}; manifest {msha[:16]}")
    return manifest


# --------------------------------------------------------------------------
# Step: verify.
# --------------------------------------------------------------------------


def collect_hashed(node: Any, found: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        path, sha = node.get("path"), node.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64 and not path.endswith("manifest.json"):
            found.append((path, sha))
        for value in node.values():
            collect_hashed(value, found)
    elif isinstance(node, list):
        for value in node:
            collect_hashed(value, found)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    out = out_paths(Path(args.out_dir))
    manifests = sorted(Path(args.out_dir).glob("*/manifest.json")) + sorted(Path(args.out_dir).glob("media/*/manifest.json"))
    checked = 0
    mismatches = []
    seals = {}
    for manifest_path in manifests:
        body = load_json(manifest_path)
        seal = body.get("manifest_sha256")
        recomputed = canonical_sha256({k: v for k, v in body.items() if k != "manifest_sha256"})
        seals[str(manifest_path)] = {"seal": seal, "recomputed_equal": seal == recomputed}
        pairs: list[tuple[str, str]] = []
        collect_hashed({k: v for k, v in body.items() if k not in ("provenance", "identity")}, pairs)
        for path, sha in pairs:
            if path.startswith("/mnt/") or not Path(path).is_file():
                continue  # sources on the share are recorded, not re-read here
            checked += 1
            if file_sha256(path) != sha:
                mismatches.append({"manifest": str(manifest_path), "path": path, "expected": sha})
    receipt = {"schema": f"{SCHEMA}/verify", "identity": tool_identity(), "manifests": seals, "files_rehashed": checked, "mismatches": mismatches, "ok": not mismatches and all(s["recomputed_equal"] for s in seals.values())}
    write_json(out["verify"], receipt, seal_field="receipt_sha256")
    log(f"verify: {len(manifests)} manifests, {checked} files re-hashed, {len(mismatches)} mismatches, ok={receipt['ok']}")
    if not receipt["ok"]:
        raise Refusal("verify failed: " + json.dumps(mismatches[:5]))
    return receipt


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

STEPS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "stage": stage,
    "screen": screen,
    "allocate": allocate,
    "text": build_text,
    "long": build_long,
    "k3": build_k3,
    "measure": measure,
    "vision": build_vision,
    "media": build_media,
    "selfgen": build_selfgen,
    "verify": verify,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pointer", default=DEFAULT_POINTER)
    parser.add_argument("--walk", default=str(DEFAULT_WALK))
    parser.add_argument("--walk-progress", default=str(DEFAULT_WALK_PROGRESS))
    parser.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR)
    parser.add_argument("--k3-dir", default=DEFAULT_K3_DIR)
    parser.add_argument("--chimera-src", default=None, help=f"chimera-omni src/ (default ${CHIMERA_SRC_ENV}, else {DEFAULT_CHIMERA_SRC})")
    parser.add_argument("--banks", default=str(DEFAULT_BANKS))
    parser.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    parser.add_argument("--screen-banks", default=str(DEFAULT_SCREEN_BANKS))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--onpolicy-pack", default=str(DEFAULT_ONPOLICY_PACK))
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--composite-tools", default=str(DEFAULT_COMPOSITE_TOOLS))
    parser.add_argument("--qwen-fold-dir", default=str(DEFAULT_QWEN_FOLD))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-registry-drift", action="store_true")
    parser.add_argument("--version", action="version", version=f"{TOOL} {VERSION}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log(f"{TOOL} {VERSION} step={args.step} out={args.out_dir}")
    steps = list(STEPS) if args.step == "all" else [args.step]
    for step in steps:
        started = time.time()
        STEPS[step](args)
        log(f"step {step} done in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
