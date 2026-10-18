#!/usr/bin/env python3
"""intake.py -- the Chimera growing-corpus intake: drop zones -> screened, grouped, split, versioned corpus v<N+1>.

Written 2026-10-15 by Claude (deep-implementer), from the intake design (task "growing-corpus intake: design") and
SPEECH_LANE_PLAN_v2_2026-10-14 section 2.3 (speech intake adjustments).  Cyril 2026-10-14: "[Cyril quote withheld]"  The sealed distillation corpus (distill_corpus_build
1.0.1) is a one-shot build; this tool is its append path.  It runs unattended on the AI server, CPU only.

Data flow
  /mnt/ai_data/chimera-intake/<modality>/<batch>/<file>    (= \\\\192.168.50.250\\ai\\chimera-intake\\ on Windows)
    scan    list the drop zone, upsert intake_file (size, mtime, stable scans)
    ingest  settled -> sha256 -> dedupe by sha -> COPY to _received/<batch>/<sha>.<ext> (NAS, immutable) and to the
            Lexar store /mnt/lexar/omni-campaign/corpus/store/<sha[:2]>/<sha>.<ext> (re-hashed on both sides).
            THE DROPPED FILE IS NEVER MODIFIED, MOVED OR DELETED (the drop-zone README promises it; a test pins it).
    probe   ffprobe / PIL facts; audio -> 16k-mono.flac (ASR input) + 32k-stereo.flac (H3 VAE input, C15 type 3),
            loudness (EBU R128) and silence stats; owner speech longer than 20 s -> 3-20 s segments (silencedetect
            -35 dB / 0.6 s), each a derived asset; images -> pHash + dHash; video -> keyframe pHash every 2 s + an
            audio derivative; text/documents -> rows (txt md json jsonl csv; pdf via pdftotext)
    screen  the holdout registry (HOLDOUT_REGISTRY_v<K>.json): distill_corpus_build's ReviewIndex (imported, not
            copied) over every text bank -- exact / 13-gram / short-item / literal -- plus media sha256, image pHash
            Hamming <= 8, and an in-house audio fingerprint (32-band log-energy delta bits, block bit-error rate).
            Liveness probes (text AND media) must pass or the run refuses.
    group   union-find over identity keys: asset:<sha>, asset:<parent sha>, wording:<digest> (>= 4 words),
            group:<batch|speaker|session value>.  speaker is an attribute, not a key, unless the batch says so.
    split   general: sha256("chimera-intake-v1:" + group_id)[:8] mod 100: < 90 train, < 95 val, else test.
            owner speech (plan v2 2.3.1): by recording session, sha256("chimera-speech-own-v1:" + session) mod 100
            < 80 train else test; >= 1 session each side; a lone session is held.  A component that touches a
            fixed split (sealed or owner session) takes its most-held split and is flagged split_conflict if mixed.
    dedupe  asset sha (intake + adopted v1 sidecars); text_key exact (intake + v1); MinHash 128 x 64-bit over word
            13-grams, 16 bands -> duplicate_near at estimated Jaccard >= 0.5; audio fingerprint vs admitted audio.
    seal    (manual) admitted + held items since v<N> -> /mnt/lexar/omni-campaign/corpus/v<N+1>/{manifest.json,
            rows/<modality>.jsonl, SHA256SUMS, sealed}; archive tgz on the NAS.  Nothing sealed is ever rewritten.
    resolve walk v2..vN from the manifests on disk (files are the record; PostgreSQL is the working index).

Speech without a transcript waits as `candidate` for `transcribe` (GPU, P40 only, gpu<N>.lock + p40.claims +
MemAvailable >= 40 GiB, plan 7.4); the cron run only lists the pending items.  Transcripts dropped next to the audio
(<stem>.txt / .lab, LibriSpeech *.trans.txt, LJSpeech metadata.csv, batch _transcripts.jsonl) are used directly.

Registry: PostgreSQL database chimera_corpus on agent-memory-postgres (192.168.50.222:25490), role chimera_corpus,
password vault key chimera_corpus_pg -> env CHIMERA_CORPUS_PG_PASSWORD.  psycopg2, raw parameterised SQL only.

Commands (exit 0 ok, 2 refused)
  schema                         print the DDL
  dropzone-init                  create the drop-zone folders and their _README.txt files (idempotent)
  init-db                        apply the DDL (idempotent: CREATE ... IF NOT EXISTS)
  registry-build --version K     snapshot every holdout bank into HOLDOUT_REGISTRY_v<K>.json + bank files
  adopt-v1                       register the sealed 1.0.1 corpus as v1 by record, write its dedupe index
  run [--cpu-only]               one unattended pass (cron); --dry-run writes nothing anywhere but --out
  dry-run --sources SPEC.json    plan v-next over arbitrary local trees (read-only), report to --out
  transcribe --gpu N             GPU worker for pending speech (needs the Pascal speech runtime; refuses without)
  import-review                  apply Cyril's _review/<batch>/transcripts.jsonl decisions now (run does it too)
  seal --version N               freeze admitted + held items into v<N>
  verify --version N             re-hash a sealed version
  resolve --version N --modality M --split train   one rows.jsonl for a stage's data config
  rescreen --version N --registry K                verdict of v<N> against a newer registry (writes nothing in v<N>)
  rebuild-index                  re-register sealed versions from their manifests into PostgreSQL
  status                         counts by status / modality / split
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

TOOL = "intake"
VERSION = "1.0.0"   # 1.0.0 (2026-10-15): first version
SCHEMA = "chimera-corpus-intake-v1"

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import distill_corpus_build as dcb  # noqa: E402  (ReviewIndex, words, norm_key, ffprobe, write_json: imported, not copied)

DEFAULT_DROP_ROOT = "/mnt/ai_data/chimera-intake"
DEFAULT_CORPUS_ROOT = "/mnt/lexar/omni-campaign/corpus"
DEFAULT_ARCHIVE_ROOT = "/mnt/ai_data/omni-campaign/corpus-versions"
DEFAULT_QUEUES = str(CAMPAIGN_ROOT / "distill" / "queues")
DEFAULT_V1_DIR = str(CAMPAIGN_ROOT / "distill" / "corpus")

MODALITY_DIRS = ("speech", "audio", "music", "images", "video", "text", "documents", "3d", "mixed")
DIR_TO_MODALITY = {"speech": "speech", "audio": "audio", "music": "music", "images": "image", "video": "video",
                   "text": "text", "documents": "document", "3d": "mesh", "mixed": "unknown"}
SERVICE_DIRS = ("_received", "_review", "_rejected")
IGNORED_NAMES = {"_README.txt", "README.txt", "desktop.ini", "Thumbs.db", ".DS_Store"}
PARTIAL_SUFFIXES = (".part", ".crdownload", ".tmp", "~", ".download", ".partial")
SIDECAR_SUFFIXES = (".sha256",)
TRANSCRIPT_EXT = (".txt", ".lab")
AUDIO_EXT = frozenset({"flac", "wav", "mp3", "ogg", "m4a", "opus", "aac", "wma", "amr", "3gp", "aiff", "aif", "caf"})
VIDEO_EXT = frozenset({"mp4", "webm", "mov", "mkv", "avi", "m4v"})
IMAGE_EXT = frozenset({"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "ppm"})
TEXT_EXT = frozenset({"txt", "md", "json", "jsonl", "csv"})
DOC_EXT = frozenset({"pdf"})
MESH_EXT = frozenset({"glb", "gltf", "obj", "ply", "stl", "fbx"})

SETTLE_MIN_GAP_S = 120.0
SETTLE_MIN_AGE_S = 180.0
SPLIT_SALT = "chimera-intake-v1"
OWNER_SESSION_SALT = "chimera-speech-own-v1"
OWNER_TRAIN_PCT = 80
GENERAL_TRAIN_PCT, GENERAL_VAL_PCT = 90, 95
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
EXACT_MIN_WORDS = 4  # wording identity key and exact text_key, as distill_corpus_build EXACT_ITEM_MIN_WORDS

MINHASH_PERM = 128
MINHASH_BANDS = 16
MINHASH_SHINGLE = dcb.NGRAM  # 13
NEAR_DUP_JACCARD = 0.5
MINHASH_PRIME = (1 << 61) - 1
_MH_RNG = np.random.default_rng(20261015)
MINHASH_A = _MH_RNG.integers(1, 1 << 32, size=MINHASH_PERM, dtype=np.uint64)
MINHASH_B = _MH_RNG.integers(0, 1 << 32, size=MINHASH_PERM, dtype=np.uint64)

PHASH_MAX_HAMMING = 8
FP_RATE = 16_000
FP_FRAME = 4096       # 256 ms: band energies stay stable under a sub-hop shift (1024/256 did not: trims failed)
FP_HOP = 128          # 8 ms
FP_BANDS = 33         # 32 band-difference bits per frame
FP_FMIN, FP_FMAX = 150.0, 4_000.0
FP_MAX_SECONDS = 600.0
FP_BLOCK = 256        # frames compared per alignment (~2 s)
FP_MIN_OVERLAP = 96   # informative frames (~0.8 s); shorter clips need 80 % of their own
FP_MAX_BER = 0.22     # measured 2026-10-15: resample/pad/trim/gain 0.00-0.03, MP3 32-64 kbit 0.05-0.16, strong noise
                      # 0.16-0.20; unrelated speech 0.40+, but two DIFFERENT LJSpeech sentences (one speaker, one room)
                      # met at 0.287 / 0.294 in the 13,100-clip dry run, so 0.30 (1.0.0 draft) was too loose
FP_SILENT_RMS = 10 ** (-70 / 20)   # frames at or below -70 dBFS carry no information (value 0, never compared)
FP_CHUNK_FRAMES = 2048
FP_LOOKUP_STRIDE = 4  # every 4th reference frame goes into the exact-hit table (memory); alignment needs one hit
SEG_MIN_S, SEG_MAX_S = 3.0, 20.0
SILENCE_DB, SILENCE_MIN_S = -35.0, 0.6
LONG_UNSEGMENTED_S = 60.0
TEXT_CHUNK_WORDS = 1_500
REVIEW_CONF = 0.85
REVIEW_AGREEMENT_WER = 0.10
MEM_GATE_GIB = 40
FFMPEG_TIMEOUT_S = 600.0

PG_HOST_ENV, PG_PORT_ENV, PG_DB_ENV, PG_USER_ENV, PG_PASSWORD_ENV = (
    "CHIMERA_CORPUS_PG_HOST", "CHIMERA_CORPUS_PG_PORT", "CHIMERA_CORPUS_PG_DB", "CHIMERA_CORPUS_PG_USER",
    "CHIMERA_CORPUS_PG_PASSWORD")
PG_SCHEMA_ENV = "CHIMERA_CORPUS_PG_TEST_SCHEMA"


def log(message: str) -> None:
    print(f"[{TOOL} {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Refusal(SystemExit):
    """A named refusal: nothing past this point is written."""

    def __init__(self, message: str):
        super().__init__(f"[{TOOL} refused] {message}")
        self.message = message


def utc(epoch: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if epoch is None else epoch))


file_sha256 = dcb.file_sha256
canonical_sha256 = dcb.canonical_sha256
words = dcb.words
norm_key = dcb.norm_key


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# Hashed once, at import: a receipt names the code that is RUNNING, even if the file is replaced mid-run.
_SELF_SHA256 = file_sha256(__file__)
_DCB_SHA256 = file_sha256(dcb.__file__)


def tool_identity() -> dict[str, Any]:
    on_disk = file_sha256(__file__)
    return {"tool": TOOL, "version": VERSION, "path": str(Path(__file__).resolve()), "sha256": _SELF_SHA256,
            "file_changed_since_start": on_disk != _SELF_SHA256, "distill_corpus_build_sha256": _DCB_SHA256,
            "host": platform.node(), "python": platform.python_version(), "utc": utc()}


def write_json_atomic(path: Path, body: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
    return file_sha256(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    return dcb.write_jsonl(path, rows)


# ======================================================================================================================
# Drop-zone rules (pure).
# ======================================================================================================================


def classify_name(name: str) -> str:
    """'data' | 'ignore' | 'partial' | 'meta' | 'sidecar' | 'transcripts' for one file name in a drop zone."""

    if name in IGNORED_NAMES or name.startswith(".") or name.startswith("~$"):
        return "ignore"
    lowered = name.lower()
    if any(lowered.endswith(s) for s in PARTIAL_SUFFIXES):
        return "partial"
    if name == "_meta.json":
        return "meta"
    if name == "_transcripts.jsonl":
        return "transcripts"
    if any(lowered.endswith(s) for s in SIDECAR_SUFFIXES):
        return "sidecar"
    return "data"


METADATA_EXT_IN_MEDIA_DIRS = frozenset({"txt", "lab", "csv", "json", "jsonl", "md", "tsv", "srt", "vtt", ""})
MEDIA_DIRS = ("speech", "audio", "music", "images", "video", "3d")


def is_item_file(name: str, modality_dir: str) -> bool:
    """Whether a drop-zone file becomes items.  In the media folders, text-like and extensionless files are metadata
    (transcripts, LibriSpeech *.trans.txt / SPEAKERS.TXT, LJSpeech metadata.csv, README): read, never itemised."""

    if classify_name(name) != "data":
        return False
    ext = name.rsplit(".", 1)[1].lower() if "." in name else ""
    return not (modality_dir in MEDIA_DIRS and ext in METADATA_EXT_IN_MEDIA_DIRS)


def is_settled(*, size: int, mtime: float, prev_size: int | None, prev_mtime: float | None, stable_since: float | None,
               now: float, name: str) -> bool:
    """Design A settle rule (CIFS is cache=loose, actimeo=30): size and mtime unchanged across two scans at least
    120 s apart, mtime older than 180 s, and not a partial-download name."""

    if classify_name(name) != "data":
        return False
    if prev_size is None or prev_mtime is None or stable_since is None:
        return False
    if size != prev_size or abs(mtime - prev_mtime) > 1e-6:
        return False
    return (now - stable_since) >= SETTLE_MIN_GAP_S and (now - mtime) >= SETTLE_MIN_AGE_S


def batch_for(rel_parts: tuple[str, ...], day_utc: str) -> tuple[str, str] | None:
    """(modality_dir, batch_id) for a file at drop_root/<rel_parts...>.  Loose files under a modality folder form
    '<modality>/<yyyy-mm-dd>-loose'.  Files outside the modality folders and in service folders: None."""

    if len(rel_parts) < 2 or rel_parts[0] not in MODALITY_DIRS:
        return None
    if any(part in SERVICE_DIRS for part in rel_parts):
        return None
    if len(rel_parts) == 2:
        return rel_parts[0], f"{rel_parts[0]}/{day_utc}-loose"
    return rel_parts[0], f"{rel_parts[0]}/{rel_parts[1]}"


def is_owner_batch(batch_id: str, meta: Mapping[str, Any]) -> bool:
    if str(meta.get("consent", "")).lower() == "owner":
        return True
    folder = batch_id.split("/", 1)[1] if "/" in batch_id else batch_id
    lowered = folder.lower()
    return lowered == "own" or lowered.startswith(("own-", "own_", "own "))


META_KEYS = {"source", "licence", "license", "consent", "speaker", "language", "notes", "session", "session_id",
             "group_by", "speaker_from", "voice", "route", "external"}


def batch_policy(batch_id: str, meta: Mapping[str, Any], *, modality_dir: str) -> dict[str, Any]:
    """Licence, consent, speaker, session and grouping of one batch (design A + plan v2 2.3)."""

    unknown = sorted(set(meta) - META_KEYS)
    flags: list[str] = [f"meta_unknown_keys:{','.join(unknown)}"] if unknown else []
    owner = is_owner_batch(batch_id, meta)
    licence = str(meta.get("licence") or meta.get("license") or "")
    consent = str(meta.get("consent") or "")
    if owner:
        consent, licence = "owner", licence or "owner"
        basis = "owner"
    elif consent == "third_party_licensed":
        basis = "_meta.json"
        if not licence:
            flags.append("licence_missing")
    elif licence or consent:
        consent = consent if consent in ("owner", "third_party_licensed") else "unspecified"
        basis = "_meta.json"
    else:
        consent, licence, basis = "unspecified", "unspecified; ruling STAGE_D_RULINGS 2026-10-14", "ruling STAGE_D_RULINGS 2026-10-14"
    if consent not in ("owner", "third_party_licensed", "unspecified"):
        consent = "unspecified"
    session = str(meta.get("session_id") or meta.get("session") or batch_id)
    group_by = str(meta.get("group_by") or ("session" if owner else "batch"))
    if group_by not in ("batch", "speaker", "session", "file"):
        flags.append(f"group_by_invalid:{group_by}")
        group_by = "batch"
    speaker = meta.get("speaker") or ("cyril" if owner and modality_dir == "speech" else None)
    return {"batch_id": batch_id, "modality_dir": modality_dir, "owner": owner, "consent": consent, "licence": licence,
            "licence_basis": basis, "source": str(meta.get("source") or ("cyril-drop" if owner else "drop")),
            "speaker": speaker, "language": meta.get("language"), "session_id": session, "group_by": group_by,
            "speaker_from": meta.get("speaker_from"), "voice": meta.get("voice"), "route": meta.get("route"),
            "meta": dict(meta), "flags": flags}


def detect_modality(ext: str, modality_dir: str, probe: Mapping[str, Any] | None) -> str:
    """Item modality: the drop folder decides speech / audio / music for audio files; bytes decide the rest."""

    ext = ext.lower()
    hint = DIR_TO_MODALITY.get(modality_dir, "unknown")
    if ext in AUDIO_EXT or (probe and probe.get("has_audio") and not probe.get("has_video")):
        return hint if hint in ("speech", "audio", "music") else "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT:
        return "text"
    if ext in DOC_EXT:
        return "document"
    if ext in MESH_EXT:
        return "mesh"
    return "unknown"


SNIFF_FAMILIES = {"mp4": {"mp4", "m4a", "mov", "m4v", "3gp", "aac"}, "jpg": {"jpg", "jpeg"}, "ogg": {"ogg", "opus", "oga"},
                  "mp3": {"mp3"}, "glb": {"glb"}}


def sniff_kind(path: Path) -> str | None:
    """A first-bytes check so a renamed file is not trusted by its extension alone."""

    try:
        head = path.open("rb").read(16)
    except OSError:
        return None
    if head.startswith(b"fLaC"):
        return "flac"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if head.startswith(b"OggS"):
        return "ogg"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"glTF"):
        return "glb"
    return None


# ======================================================================================================================
# Splits and groups (pure, deterministic: no input order, no PYTHONHASHSEED).
# ======================================================================================================================


def hash_pct(salt: str, key: str) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8], 16) % 100


def general_split(group_id: str) -> str:
    pct = hash_pct(SPLIT_SALT, group_id)
    return "train" if pct < GENERAL_TRAIN_PCT else "val" if pct < GENERAL_VAL_PCT else "test"


def owner_session_splits(sessions: Iterable[str], fixed: Mapping[str, str] | None = None) -> dict[str, str]:
    """Plan v2 2.2: own speech is split BY SESSION, 80 train / 20 held ('test'); >= 1 session on each side; a lone
    session is held.  ``fixed`` = sessions already sealed: never moved.  Forced moves only touch unfixed sessions:
    no held -> the unfixed session with the highest hash goes held; no train -> the lowest goes train."""

    fixed = dict(fixed or {})
    every = sorted(set(sessions) | set(fixed))
    out = {s: fixed[s] if s in fixed else ("train" if hash_pct(OWNER_SESSION_SALT, s) < OWNER_TRAIN_PCT else "test")
           for s in every}
    free = sorted((s for s in every if s not in fixed), key=lambda s: (hash_pct(OWNER_SESSION_SALT, s), s))
    if len(every) == 1:
        if free:
            out[every[0]] = "test"
        return out
    if not any(v == "test" for v in out.values()) and free:
        out[free[-1]] = "test"
    if not any(v == "train" for v in out.values()):
        candidates = [s for s in free if out[s] == "test"]
        held = [s for s in every if out[s] == "test"]
        if candidates and len(held) > 1:
            out[candidates[0]] = "train"
    return out


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def wording_key(text: str | None) -> str | None:
    toks = words(text or "")
    if len(toks) < EXACT_MIN_WORDS:
        return None
    return "wording:" + text_sha256(" ".join(toks))[:32]


def text_key(text: str | None) -> str | None:
    toks = words(text or "")
    return text_sha256(" ".join(toks)) if len(toks) >= EXACT_MIN_WORDS else None


def assign_groups(items: list[dict[str, Any]], known_keys: Mapping[str, tuple[str, str | None]]) -> None:
    """Union-find over each item's ``identity`` keys (in place: group_id, fixed_split, split_conflict).

    ``known_keys`` maps an identity key already in the registry to (group_id, split or None when unsealed).  A
    component that reaches a known key inherits the smallest known group id; its fixed split is the most-held
    split among the known keys and owner-session splits it touches."""

    uf = UnionFind()
    for item in items:
        keys = sorted(item["identity"])
        uf.find(keys[0])
        for key in keys[1:]:
            uf.union(keys[0], key)
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        members[uf.find(sorted(item["identity"])[0])].append(item)
    for root, group in members.items():
        keys = sorted({k for it in group for k in it["identity"]})
        known = [known_keys[k] for k in keys if k in known_keys]
        group_id = min(g for g, _ in known) if known else text_sha256("group:" + keys[0])
        fixed = [s for _, s in known if s] + [it["fixed_split"] for it in group if it.get("fixed_split")]
        split = max(fixed, key=lambda s: SPLIT_RANK[s]) if fixed else None
        conflict = len(set(fixed)) > 1
        for it in group:
            it["group_id"] = group_id
            it["group_split"] = split
            it["split_conflict"] = conflict


def final_split(item: Mapping[str, Any]) -> str:
    return item.get("group_split") or general_split(item["group_id"])


# ======================================================================================================================
# MinHash near-duplicates (train vs train).
# ======================================================================================================================


def _shingles(toks: list[str]) -> list[str]:
    n = MINHASH_SHINGLE
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def minhash_signature(text: str) -> np.ndarray | None:
    shingles = _shingles(words(text))
    if not shingles:
        return None
    x = np.array([int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")
                  for s in sorted(set(shingles))], dtype=np.uint64)
    # (a*x + b) mod (2^61 - 1) with a, b, x < 2^32: a*x + b < 2^64, no uint64 overflow.
    values = (MINHASH_A[:, None] * x[None, :] + MINHASH_B[:, None]) % np.uint64(MINHASH_PRIME)
    return values.min(axis=1).astype(np.uint64)


def minhash_bands(signature: np.ndarray) -> list[int]:
    rows = MINHASH_PERM // MINHASH_BANDS
    return [int.from_bytes(hashlib.blake2b(signature[b * rows:(b + 1) * rows].tobytes(), digest_size=8).digest(),
                           "little", signed=True) for b in range(MINHASH_BANDS)]


def jaccard_estimate(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


class NearDupIndex:
    """LSH over admitted signatures: candidates by any equal band, confirmed by estimated Jaccard."""

    def __init__(self) -> None:
        self.bands: dict[tuple[int, int], list[str]] = defaultdict(list)
        self.signatures: dict[str, np.ndarray] = {}

    def add(self, item_id: str, signature: np.ndarray) -> None:
        self.signatures[item_id] = signature
        for b, h in enumerate(minhash_bands(signature)):
            self.bands[(b, h)].append(item_id)

    def query(self, signature: np.ndarray) -> tuple[str, float] | None:
        seen: set[str] = set()
        best: tuple[str, float] | None = None
        for b, h in enumerate(minhash_bands(signature)):
            for other in self.bands.get((b, h), ()):
                if other in seen:
                    continue
                seen.add(other)
                j = jaccard_estimate(signature, self.signatures[other])
                if j >= NEAR_DUP_JACCARD and (best is None or j > best[1] or (j == best[1] and other < best[0])):
                    best = (other, j)
        return best


# ======================================================================================================================
# Media: ffmpeg decode, audio fingerprint, image hashes, segmentation, loudness.
# ======================================================================================================================


def run_cmd(command: list[str], *, timeout: float = FFMPEG_TIMEOUT_S, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, timeout=timeout, text=not binary)


def ffprobe_facts(path: Path) -> dict[str, Any]:
    try:
        return dcb.ffprobe(str(path))
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return {"ok": False, "error": repr(exc)[:300]}


def decode_pcm(path: Path, rate: int = FP_RATE, max_seconds: float | None = FP_MAX_SECONDS) -> np.ndarray:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    if max_seconds:
        command += ["-t", f"{max_seconds:.3f}"]
    command += ["-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    result = run_cmd(command, binary=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode {path}: {result.stderr[-300:]!r}")
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)


def _band_edges() -> np.ndarray:
    freqs = np.fft.rfftfreq(FP_FRAME, 1.0 / FP_RATE)
    edges_hz = np.geomspace(FP_FMIN, FP_FMAX, FP_BANDS + 1)
    return np.searchsorted(freqs, edges_hz)


_FP_EDGES = _band_edges()
_FP_WINDOW = np.hanning(FP_FRAME).astype(np.float32)


def audio_fingerprint(pcm: np.ndarray) -> np.ndarray:
    """Per-frame 32-bit sub-fingerprints (Haitsma-Kalker style): bit m of frame n is the sign of the change, frame
    n-1 -> n, of the energy difference between log-spaced bands m and m+1.  Gain-invariant; survives resampling,
    re-encoding and padding.  Returns uint32[frames] (empty for < 2 frames)."""

    if pcm.size < FP_FRAME + FP_HOP:
        return np.zeros(0, dtype=np.uint32)
    frames = 1 + (pcm.size - FP_FRAME) // FP_HOP
    out = np.zeros(frames - 1, dtype=np.uint32)
    weights = (1 << np.arange(32, dtype=np.uint64)).astype(np.uint64)
    step = FP_CHUNK_FRAMES  # frames per block: bounds the [frames, FP_FRAME] gather for long recordings
    previous: np.ndarray | None = None
    for start in range(0, frames, step):
        stop = min(frames, start + step)
        idx = np.arange(FP_FRAME)[None, :] + FP_HOP * np.arange(start, stop)[:, None]
        chunk = pcm[idx]
        spec = np.abs(np.fft.rfft(chunk * _FP_WINDOW, axis=1)) ** 2
        cums = np.concatenate([np.zeros((stop - start, 1)), np.cumsum(spec, axis=1)], axis=1)
        energy = cums[:, _FP_EDGES[1:]] - cums[:, _FP_EDGES[:-1]]
        band_diff = np.log(energy + 1e-10)[:, :-1] - np.log(energy + 1e-10)[:, 1:]
        silent = np.sqrt(np.mean(chunk.astype(np.float64) ** 2, axis=1)) <= FP_SILENT_RMS
        if previous is not None:
            band_diff = np.concatenate([previous[0][None, :], band_diff])
            silent = np.concatenate([previous[1], silent])
        bits = (band_diff[1:] - band_diff[:-1]) > 0
        values = (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)
        values[silent[1:] | silent[:-1]] = 0
        offset = start - (1 if previous is not None else 0)
        out[offset:offset + values.size] = values
        previous = (band_diff[-1], silent[-1:])
    return out


def _popcount32(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.uint32)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * np.uint32(0x01010101)) & np.uint32(0xFFFFFFFF)) >> 24


def fingerprint_ber(a: np.ndarray, b: np.ndarray, offset: int, need: int = 1) -> tuple[float, int]:
    """Bit-error rate of a against b shifted by offset (a[i] ~ b[i + offset]) over their best FP_BLOCK window of
    informative frames (value 0 = silent, never compared).  Returns (ber, informative frames in that window); a
    window with fewer than ``need`` informative frames does not count."""

    start_a, start_b = max(0, -offset), max(0, offset)
    n = min(a.size - start_a, b.size - start_b)
    if n <= 0:
        return 1.0, 0
    xa, xb = a[start_a:start_a + n], b[start_b:start_b + n]
    valid = ((xa != 0) & (xb != 0)).astype(np.int64)
    errors = _popcount32(xa ^ xb).astype(np.int64) * valid
    if n <= FP_BLOCK:
        count = int(valid.sum())
        return (float(errors.sum()) / (32.0 * count), count) if count >= max(1, need) else (1.0, count)
    ones = np.ones(FP_BLOCK, dtype=np.int64)
    win_err, win_cnt = np.convolve(errors, ones, mode="valid"), np.convolve(valid, ones, mode="valid")
    ok = win_cnt >= max(1, need)
    if not ok.any():
        return 1.0, int(win_cnt.max())
    ber = win_err[ok] / (32.0 * win_cnt[ok])
    best = int(np.argmin(ber))
    return float(ber[best]), int(win_cnt[ok][best])


class AudioFpIndex:
    """Reference fingerprints with an exact sub-fingerprint lookup table; a candidate matches a reference when some
    aligned block of >= FP_MIN_OVERLAP frames (80 % of a short clip) has bit-error rate <= FP_MAX_BER."""

    def __init__(self) -> None:
        self.refs: list[tuple[str, str, np.ndarray]] = []
        self.lookup: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def add(self, source: str, item_id: str, fp: np.ndarray) -> None:
        if fp.size == 0:
            return
        ref = len(self.refs)
        self.refs.append((source, item_id, fp))
        for pos in range(0, fp.size, FP_LOOKUP_STRIDE):
            value = int(fp[pos])
            if value:
                self.lookup[value].append((ref, pos))

    def __len__(self) -> int:
        return len(self.refs)

    def match(self, fp: np.ndarray, *, exclude_source: str | None = None) -> dict[str, Any] | None:
        if fp.size == 0 or not self.refs:
            return None
        votes: Counter = Counter()
        for pos, value in enumerate(fp.tolist()):
            for ref, ref_pos in self.lookup.get(value, ()):
                votes[(ref, ref_pos - pos)] += 1
        best: dict[str, Any] | None = None
        informative = int(np.count_nonzero(fp))
        for (ref, offset), _count in votes.most_common(16):
            source, item_id, ref_fp = self.refs[ref]
            if exclude_source is not None and source == exclude_source:
                continue
            need = min(FP_MIN_OVERLAP, max(8, int(0.8 * min(informative, int(np.count_nonzero(ref_fp))))))
            ber, overlap = fingerprint_ber(fp, ref_fp, offset, need)
            if overlap >= need and ber <= FP_MAX_BER and (best is None or ber < best["ber"]):
                best = {"source": source, "item": item_id, "ber": round(ber, 4), "offset_frames": offset,
                        "overlap_frames": overlap}
        return best


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def phash_from_gray32(gray: np.ndarray) -> int:
    coeffs = _DCT32 @ gray.astype(np.float64) @ _DCT32.T
    low = coeffs[:8, :8].flatten()
    median = np.median(low[1:])
    bits = low > median
    return int("".join("1" if b else "0" for b in bits), 2)


def image_hashes(path: Path) -> dict[str, Any]:
    from PIL import Image  # noqa: E402

    with Image.open(path) as image:
        width, height = image.size
        fmt = image.format
        gray = image.convert("L")
        g32 = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
        g98 = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.float64)
    dbits = (g98[:, 1:] > g98[:, :-1]).flatten()
    return {"width": width, "height": height, "format": fmt, "phash": phash_from_gray32(g32),
            "dhash": int("".join("1" if b else "0" for b in dbits), 2)}


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def video_keyframe_phashes(path: Path, every_s: float = 2.0, max_frames: int = 600) -> list[int]:
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", f"fps={1.0 / every_s},scale=32:32:flags=area,format=gray",
               "-frames:v", str(max_frames), "-f", "rawvideo", "-"]
    result = run_cmd(command, binary=True)
    if result.returncode != 0:
        return []
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    return [phash_from_gray32(f.reshape(32, 32)) for f in raw[: raw.size // 1024 * 1024].reshape(-1, 1024)]


SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def detect_silences(path: Path) -> list[tuple[float, float]]:
    command = ["ffmpeg", "-nostdin", "-v", "info", "-i", str(path), "-vn", "-af",
               f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}", "-f", "null", "-"]
    result = run_cmd(command)
    starts = [float(x) for x in SILENCE_START.findall(result.stderr)]
    ends = [float(x) for x in SILENCE_END.findall(result.stderr)]
    out = []
    for i, s in enumerate(starts):
        out.append((max(0.0, s), ends[i] if i < len(ends) else float("inf")))
    return out


def plan_segments(duration: float, silences: list[tuple[float, float]], *, min_s: float = SEG_MIN_S,
                  max_s: float = SEG_MAX_S) -> list[tuple[float, float]]:
    """3-20 s speech segments from silencedetect intervals (plan v2 2.3.2).  Speech runs between silences are packed
    greedily up to max_s, cutting only at silence midpoints; a run longer than max_s is cut hard every max_s; a
    trailing piece shorter than min_s is merged back when that stays <= max_s, else dropped (recorded)."""

    if duration <= 0:
        return []
    runs: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(silences):
        e = min(e, duration)
        if s > cursor:
            runs.append((cursor, min(s, duration)))
        cursor = max(cursor, e)
    if cursor < duration:
        runs.append((cursor, duration))
    pieces: list[tuple[float, float]] = []
    for s, e in runs:
        while e - s > max_s:
            pieces.append((s, s + max_s))
            s += max_s
        if e - s > 0.05:
            pieces.append((s, e))
    segments: list[list[float]] = []
    for s, e in pieces:
        if segments and e - segments[-1][0] <= max_s:
            segments[-1][1] = e
        else:
            segments.append([s, e])
    out: list[tuple[float, float]] = []
    for s, e in segments:
        if e - s >= min_s:
            out.append((round(s, 3), round(e, 3)))
        elif out and e - out[-1][0] <= max_s:
            out[-1] = (out[-1][0], round(e, 3))
    return out


LUFS_RE = re.compile(r"I:\s*(-?[0-9.]+|-inf)\s*LUFS")


def loudness_lufs(path: Path) -> float | None:
    command = ["ffmpeg", "-nostdin", "-v", "info", "-i", str(path), "-vn", "-af", "ebur128=framelog=quiet", "-f", "null", "-"]
    result = run_cmd(command)
    found = LUFS_RE.findall(result.stderr)
    if not found or found[-1] == "-inf":
        return None
    return float(found[-1])


def audio_stats(pcm: np.ndarray) -> dict[str, Any]:
    if pcm.size == 0:
        return {"rms_dbfs": None, "peak": 0.0, "silence_fraction": 1.0}
    frame = 400
    usable = pcm[: pcm.size // frame * frame].reshape(-1, frame) if pcm.size >= frame else pcm.reshape(1, -1)
    rms = np.sqrt(np.mean(usable.astype(np.float64) ** 2, axis=1))
    silent = float(np.mean(20 * np.log10(rms + 1e-12) < SILENCE_DB))
    whole = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    return {"rms_dbfs": round(20 * math.log10(whole + 1e-12), 2), "peak": round(float(np.max(np.abs(pcm))), 4),
            "silence_fraction": round(silent, 4)}


def ffmpeg_derive(source: Path, destination: Path, args: list[str], start: float | None = None,
                  end: float | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp" + destination.suffix)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(source)]
    if end is not None:
        command += ["-t", f"{(end - (start or 0.0)):.3f}"]
    command += ["-vn", *args, "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(tmp)]
    result = run_cmd(command)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg derive {source} -> {destination.name}: {result.stderr[-300:]!r}")
    os.replace(tmp, destination)


DERIVATIONS = {
    "16k-mono.flac": ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "flac"],
    "32k-stereo.flac": ["-ac", "2", "-ar", "32000", "-sample_fmt", "s16", "-c:a", "flac"],
}


# ======================================================================================================================
# Text extraction.
# ======================================================================================================================


def chunk_words(text: str, limit: int = TEXT_CHUNK_WORDS) -> list[str]:
    """Paragraph-packed chunks of at most ``limit`` whitespace words; a single longer paragraph is cut by words."""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if n > limit:
            if current:
                chunks.append("\n\n".join(current))
                current, count = [], 0
            toks = para.split()
            chunks.extend(" ".join(toks[i:i + limit]) for i in range(0, len(toks), limit))
            continue
        if count + n > limit and current:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def extract_text_rows(path: Path, ext: str) -> tuple[list[dict[str, str]], str | None]:
    """[{instruction, response}] rows of a text/document asset, or ([], status) when it cannot be read."""

    ext = ext.lower()
    if ext == "pdf":
        if not shutil.which("pdftotext"):
            return [], "needs_tool"
        result = run_cmd(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], timeout=300)
        if result.returncode != 0:
            return [], "unreadable"
        return [{"instruction": "", "response": c} for c in chunk_words(result.stdout)], None
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_bytes().decode("utf-8", errors="replace")
        if raw.count("\ufffd") > max(16, len(raw) // 100):
            return [], "not_utf8"
    if ext == "jsonl":
        rows = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return [], "bad_jsonl"
            rows.append(_record_row(record))
        return [r for r in rows if r["instruction"] or r["response"]], None
    if ext == "json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return [], "bad_json"
        if isinstance(payload, list) and payload and all(isinstance(x, Mapping) for x in payload):
            return [r for r in (_record_row(x) for x in payload) if r["instruction"] or r["response"]], None
        strings: list[str] = []
        dcb.harvest_strings(payload, strings)
        return [{"instruction": "", "response": c} for c in chunk_words("\n\n".join(strings))], None
    if ext == "csv":
        lines = raw.splitlines()
        header, body = (lines[0], lines[1:]) if lines else ("", [])
        return [{"instruction": "", "response": "\n".join([header, *body[i:i + 50]])} for i in range(0, len(body), 50)], None
    return [{"instruction": "", "response": c} for c in chunk_words(raw)], None


def _record_row(record: Mapping[str, Any]) -> dict[str, str]:
    if isinstance(record, Mapping):
        return {"instruction": dcb.extract_instruction(record), "response": dcb.extract_response(record)}
    return {"instruction": "", "response": str(record)}


# ======================================================================================================================
# Transcript sidecars.
# ======================================================================================================================


def sidecar_transcripts(batch_root: Path, files: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """{str(audio path): {transcript, tool, speaker?}} from files that ship WITH the drop: <stem>.txt/.lab, LibriSpeech
    *.trans.txt, LJSpeech metadata.csv, and a batch-level _transcripts.jsonl ({"file": relative path, "transcript"})."""

    found: dict[str, dict[str, Any]] = {}
    files = list(files)
    by_stem: dict[tuple[str, str], Path] = {(str(p.parent), p.stem): p for p in files}
    for path in files:
        for ext in TRANSCRIPT_EXT:
            side = path.with_suffix(ext)
            if side.is_file() and side != path and side.suffix.lower() not in (path.suffix.lower(),):
                text = side.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    found[str(path)] = {"transcript": text, "tool": f"sidecar {side.name}"}
                    break
    for trans in batch_root.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8", errors="replace").splitlines():
            utt, _, text = line.strip().partition(" ")
            target = by_stem.get((str(trans.parent), utt))
            if target is not None and text:
                found[str(target)] = {"transcript": text, "tool": f"LibriSpeech {trans.name}",
                                      "speaker": "LS-" + utt.split("-")[0]}
    meta_csv = batch_root / "metadata.csv"
    if meta_csv.is_file():
        for line in meta_csv.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                target = by_stem.get((str(batch_root / "wavs"), parts[0]))
                if target is not None:
                    found[str(target)] = {"transcript": (parts[2] if len(parts) > 2 and parts[2] else parts[1]).strip(),
                                          "tool": "LJSpeech metadata.csv (normalized)", "speaker": "LJ"}
    jl = batch_root / "_transcripts.jsonl"
    if jl.is_file():
        for line in jl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                target = batch_root / rec["file"]
                if rec.get("transcript"):
                    found[str(target)] = {"transcript": str(rec["transcript"]).strip(), "tool": "_transcripts.jsonl",
                                          **({"speaker": rec["speaker"]} if rec.get("speaker") else {})}
    return found


# ======================================================================================================================
# Holdout registry: build (server), load, screen, liveness.
# ======================================================================================================================


def _snapshot_texts(index: Any) -> list[dict[str, str]]:
    out = []
    for source, items in sorted(index.items.items()):
        for item_id, text, _n in items:
            literal = index.literals.get(str(text).lower())
            rule = "literal" if literal is not None and literal == (source, item_id) else "text"
            out.append({"bank": source, "item": str(item_id), "rule": rule, "text": text})
    return out


def registry_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every holdout bank of registry v1 (design C).  Paths are the server's; a missing bank refuses the build."""

    oc = Path(args.oc)
    adopted = Path(args.adopted)
    prep = oc / "distill" / "codex-preparation-01a1326f-20261013"
    return [
        {"name": "review_surface", "kind": "review_index", "note": "distill_corpus_build.build_review_index: the 12 "
         "review sources of contamination_screen.json (bank v1/v2 + haystacks + vision cards, probe, smoke, R-Q, "
         "probe64, needle sets/codes, export parity, extension, paraphrase)"},
        {"name": "eval_split", "kind": "rows_jsonl", "path": str(oc / "distill/corpus/eval_split/eval_rows.jsonl"),
         "expect_sha256": "626093ca6024424f0ddc1ab397376dd5822d10bafb9818fd3a366777e743c4e2",
         "note": "1,948 eval rows incl. the 28 stage-T rows; instruction, response and sidecar sha256"},
        {"name": "bfcl62", "kind": "rid_rows", "path": str(oc / "distill/corpus/holdouts/bfcl62_rids.json"),
         "rows": str(oc / "distill/corpus/staging/rows.jsonl"), "note": "BFCL-62 rids resolved to their staging rows"},
        {"name": "codex_eval_combined_oct13_v1", "kind": "harvest_jsonl",
         "path": str(adopted / "codex-local-recovery-eval-combined-oct13-v1/payload/chimera_recovery_eval_combined_oct13_v1/tasks.jsonl"),
         "note": "Codex combined held eval, 274 tasks"},
        {"name": "codex_speech_held_inventory", "kind": "speech_inventory_heldout",
         "path": str(adopted / "codex-local-speech-combined-admission-oct13-v1/payload/chimera_speech_combined_admission_oct13_v1/source_inventory.jsonl"),
         "note": "Codex speech admission rows with split != train (texts + audio sha256)"},
        {"name": "codex_speech_held_cpu_v1", "kind": "tts_json", "path": str(adopted / "speech-held-cpu-v1/payload/tts-16.json"),
         "audio_root": str(prep / "speech-held-cpu-v1"), "all_held": True,
         "note": "Codex independently authored held speech, 16 rows; texts, audio sha256 and fingerprints"},
        {"name": "codex_tts192_heldout", "kind": "tts_json",
         "path": str(prep / "speech-gpu-campaign-bf16-v2/tts-192.json"),
         "audio_root": str(prep / "speech-gpu-campaign-bf16-v2"), "all_held": False,
         "note": "Codex synthetic TTS rows whose split != train (texts + audio)"},
        {"name": "media_admission_v3_held", "kind": "media_admission_heldout", "path": str(prep / "media-admission-v3/admission.jsonl"),
         "note": "Codex media admission v3 rows with split != train: asset sha256 (+ pHash of held images)"},
        {"name": "d1_validation", "kind": "id_only_jsonl", "path": str(prep / "d1-source-successor-v2/documents.jsonl"),
         "note": "D1 source successor v2 validation documents: record sha256 only (no text in the file); recorded, "
                 "cannot match a new drop"},
    ]


def _resolve_audio(path_text: str, audio_root: str | None, row_id: str) -> Path | None:
    candidates = [Path(path_text)] if path_text else []
    if audio_root:
        name = Path(path_text.replace("\\", "/")).name if path_text else "original.wav"
        candidates.append(Path(audio_root) / "assets" / row_id / name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def build_registry(args: argparse.Namespace) -> dict[str, Any]:
    version = int(args.version)
    registry_path = Path(args.corpus_root) / "holdouts" / f"HOLDOUT_REGISTRY_v{version}.json"
    if registry_path.exists():
        raise Refusal(f"{registry_path} exists: registry versions are append-only")
    texts: list[dict[str, str]] = []
    media: list[dict[str, Any]] = []
    ids: list[dict[str, str]] = []
    fps: list[tuple[str, str, np.ndarray]] = []
    banks: list[dict[str, Any]] = []
    for spec in registry_sources(args):
        name, kind = spec["name"], spec["kind"]
        before = (len(texts), len(media), len(ids), len(fps))
        entry: dict[str, Any] = {"name": name, "kind": kind, "note": spec.get("note")}
        if kind == "review_index":
            dargs = dcb.parse_args(["screen"])
            index, sources = dcb.build_review_index(dargs)
            for row in _snapshot_texts(index):
                texts.append(row)
            for sha, (source, item) in sorted(index.media.items()):
                media.append({"bank": source, "item": item, "sha256": sha, "kind": "image"})
            entry["sources"] = sources
            entry["review_index_summary"] = index.summary()
        else:
            path = Path(spec["path"])
            if not path.is_file():
                raise Refusal(f"holdout bank {name} missing at {path}")
            entry["path"] = str(path)
            entry["sha256"] = file_sha256(path)
            if spec.get("expect_sha256") and entry["sha256"] != spec["expect_sha256"]:
                raise Refusal(f"holdout bank {name}: {path} hashes {entry['sha256']}, pinned {spec['expect_sha256']}")
            if kind == "rows_jsonl":
                for i, row in enumerate(dcb.read_jsonl(path)):
                    rid = str(row.get("record_id") or row.get("rid") or i)
                    for field in ("instruction", "response"):
                        if str(row.get(field) or "").strip():
                            texts.append({"bank": name, "item": f"{rid}#{field}", "rule": "text", "text": str(row[field])})
                    for side in row.get("sidecars") or []:
                        if side.get("sha256"):
                            media.append({"bank": name, "item": rid, "sha256": str(side["sha256"]).lower(),
                                          "kind": side.get("media") or side.get("kind") or "media"})
            elif kind == "rid_rows":
                rid_payload = dcb.load_json(path)
                wanted = set(rid_payload.get("rids") or [])
                rows_path = Path(spec["rows"])
                entry["rids"] = len(wanted)
                entry["rows_path"], entry["rows_sha256"] = str(rows_path), file_sha256(rows_path)
                found = 0
                with rows_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get("rid") in wanted:
                            found += 1
                            for field in ("instruction", "response"):
                                if str(row.get(field) or "").strip():
                                    texts.append({"bank": name, "item": f"{row['rid']}#{field}", "rule": "text", "text": str(row[field])})
                entry["rids_resolved"] = found
                for rid in sorted(wanted):
                    ids.append({"bank": name, "id": rid})
            elif kind == "harvest_jsonl":
                for i, row in enumerate(dcb.read_jsonl(path)):
                    strings: list[str] = []
                    dcb.harvest_strings(row, strings)
                    for j, value in enumerate(strings):
                        texts.append({"bank": name, "item": f"{row.get('id', i)}#{j}", "rule": "text", "text": value})
            elif kind == "speech_inventory_heldout":
                for row in dcb.read_jsonl(path):
                    if row.get("split") == "train":
                        continue
                    strings: list[str] = []
                    dcb.harvest_strings({k: v for k, v in row.items() if k != "media"}, strings)
                    for j, value in enumerate(strings):
                        texts.append({"bank": name, "item": f"{row.get('id')}#{j}", "rule": "text", "text": value})
                    for m in row.get("media") or []:
                        for key in ("sha256", "source_media_sha256"):
                            if m.get(key):
                                media.append({"bank": name, "item": str(row.get("id")), "sha256": str(m[key]).lower(), "kind": "audio"})
            elif kind == "tts_json":
                payload = dcb.load_json(path)
                for row in payload.get("rows", []):
                    if not spec.get("all_held") and row.get("split") == "train":
                        continue
                    rid = str(row.get("id"))
                    texts.append({"bank": name, "item": rid, "rule": "text", "text": str(row.get("text", ""))})
                    for key in ("audio_sha256", "h3_audio_sha256"):
                        if row.get(key):
                            media.append({"bank": name, "item": rid, "sha256": str(row[key]).lower(), "kind": "audio"})
                    audio = _resolve_audio(str(row.get("audio") or ""), spec.get("audio_root"), rid)
                    if audio is not None:
                        fps.append((name, rid, audio_fingerprint(decode_pcm(audio))))
            elif kind == "media_admission_heldout":
                for row in dcb.read_jsonl(path):
                    if row.get("split") in (None, "train"):
                        continue
                    sha = str(row.get("asset_sha256") or "").lower()
                    if not sha:
                        continue
                    record: dict[str, Any] = {"bank": name, "item": str(row.get("item_id")), "sha256": sha, "kind": "media"}
                    asset = Path(str(row.get("asset_path") or ""))
                    if asset.suffix.lower().lstrip(".") in IMAGE_EXT and asset.is_file():
                        with contextlib.suppress(Exception):
                            record["phash"] = image_hashes(asset)["phash"]
                            record["kind"] = "image"
                    media.append(record)
            elif kind == "id_only_jsonl":
                for row in dcb.read_jsonl(path):
                    if row.get("split") not in (None, "train"):
                        ids.append({"bank": name, "id": str(row.get("source_record_sha256") or row.get("id"))})
        entry["texts"] = len(texts) - before[0]
        entry["media"] = len(media) - before[1]
        entry["ids"] = len(ids) - before[2]
        entry["audio_fingerprints"] = len(fps) - before[3]
        banks.append(entry)
        log(f"registry v{version}: {name}: {entry['texts']} texts, {entry['media']} media, {entry['ids']} ids, "
            f"{entry['audio_fingerprints']} audio fps")
    banks.append({"name": "librispeech_test_route_rows", "kind": "pending", "texts": 0, "media": 0, "ids": 0,
                  "audio_fingerprints": 0, "note": "LibriSpeech ASR test 89 / generation test 104 (Codex "
                  "chimera_natural_speech_repair_oct13_v3): not on the server at build time; add in registry v2"})
    return write_registry(Path(args.corpus_root), version, texts, media, ids, fps, banks, args.agent)


def write_registry(corpus_root: Path, version: int, texts: list[dict[str, str]], media: list[dict[str, Any]],
                   ids: list[dict[str, str]], fps: list[tuple[str, str, np.ndarray]], banks: list[dict[str, Any]],
                   agent: str) -> dict[str, Any]:
    """Snapshot files + the sealed HOLDOUT_REGISTRY_v<K>.json.  Append-only: an existing version refuses."""

    out_dir = corpus_root / "holdouts" / f"registry_v{version}"
    registry_path = corpus_root / "holdouts" / f"HOLDOUT_REGISTRY_v{version}.json"
    if registry_path.exists():
        raise Refusal(f"{registry_path} exists: registry versions are append-only")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    files["texts"] = dict(zip(("sha256", "rows"), write_jsonl_atomic(out_dir / "texts.jsonl", texts)))
    files["media"] = dict(zip(("sha256", "rows"), write_jsonl_atomic(out_dir / "media.jsonl", media)))
    files["ids"] = dict(zip(("sha256", "rows"), write_jsonl_atomic(out_dir / "ids.jsonl", ids)))
    fp_meta = [{"bank": b, "item": i, "frames": int(f.size)} for b, i, f in fps]
    fp_blob = np.concatenate([f for _, _, f in fps]) if fps else np.zeros(0, dtype=np.uint32)
    np.save(out_dir / "audio_fp.npy", fp_blob.astype("<u4"), allow_pickle=False)
    files["audio_fp"] = {"sha256": file_sha256(out_dir / "audio_fp.npy"), "frames": int(fp_blob.size)}
    files["audio_fp_index"] = dict(zip(("sha256", "rows"), write_jsonl_atomic(out_dir / "audio_fp_index.jsonl", fp_meta)))
    body = {"schema": f"{SCHEMA}/holdout_registry", "version": version, "created_utc": utc(), "created_by": agent,
            "identity": tool_identity(), "dir": str(out_dir), "files": files, "banks": banks,
            "rule": "every new item is screened against every bank; test rows of corpus v<N> join registry v<K+1>"}
    sha = dcb.write_json(registry_path, body, seal_field="registry_sha256")
    log(f"registry v{version} written: {registry_path} sealed {sha}")
    return {**body, "registry_sha256": sha, "path": str(registry_path)}


class Holdouts:
    """A loaded registry: ReviewIndex over every text bank, media sha256 map, image pHashes, audio fingerprints."""

    def __init__(self) -> None:
        self.version: int | None = None
        self.sha256: str | None = None
        self.path: str | None = None
        self.index = dcb.ReviewIndex()
        self.media: dict[str, tuple[str, str]] = {}
        self.phashes: list[tuple[int, str, str]] = []
        self.audio = AudioFpIndex()
        self.audio_samples: list[tuple[str, str, np.ndarray]] = []

    @classmethod
    def load(cls, registry_path: Path) -> "Holdouts":
        body = dcb.load_json(registry_path)
        seal = body.get("registry_sha256")
        if canonical_sha256({k: v for k, v in body.items() if k != "registry_sha256"}) != seal:
            raise Refusal(f"{registry_path}: seal does not match its body")
        out = cls()
        out.version, out.sha256, out.path = int(body["version"]), seal, str(registry_path)
        folder = Path(body["dir"]) if Path(body["dir"]).is_dir() else registry_path.parent / f"registry_v{body['version']}"
        for name, meta in body["files"].items():
            file = folder / ("audio_fp.npy" if name == "audio_fp" else f"{name}.jsonl")
            if file_sha256(file) != meta["sha256"]:
                raise Refusal(f"registry v{body['version']} file {file} changed since the build")
        for row in dcb.read_jsonl(folder / "texts.jsonl"):
            if row["rule"] == "literal":
                out.index.add_literal(row["bank"], row["item"], row["text"])
            else:
                out.index.add_text(row["bank"], row["item"], row["text"])
        for row in dcb.read_jsonl(folder / "media.jsonl"):
            out.media.setdefault(row["sha256"].lower(), (row["bank"], row["item"]))
            out.index.add_media(row["bank"], row["item"], row["sha256"])
            if row.get("phash") is not None:
                out.phashes.append((int(row["phash"]), row["bank"], row["item"]))
        blob = np.load(folder / "audio_fp.npy", allow_pickle=False)
        cursor = 0
        for row in dcb.read_jsonl(folder / "audio_fp_index.jsonl"):
            fp = blob[cursor:cursor + row["frames"]].astype(np.uint32)
            cursor += row["frames"]
            out.audio.add(row["bank"], row["item"], fp)
            out.audio_samples.append((row["bank"], row["item"], fp))
        return out

    def screen_text(self, text: str) -> list[dict[str, str]]:
        return self.index.hits(text) if text and text.strip() else []

    def screen_media(self, sha256: str, *, phash: int | None = None, keyframes: Iterable[int] = (),
                     fp: np.ndarray | None = None) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        if sha256.lower() in self.media:
            bank, item = self.media[sha256.lower()]
            hits.append({"rule": "media_sha256", "source": bank, "item": item})
        hashes = ([phash] if phash is not None else []) + list(keyframes)
        for value in hashes:
            for ref, bank, item in self.phashes:
                if hamming64(value, ref) <= PHASH_MAX_HAMMING:
                    hits.append({"rule": "phash", "source": bank, "item": item, "hamming": hamming64(value, ref)})
                    break
            if any(h["rule"] == "phash" for h in hits):
                break
        if fp is not None and fp.size:
            match = self.audio.match(fp)
            if match:
                hits.append({"rule": "audio_fingerprint", **match})
        return hits


def media_liveness(holdouts: Holdouts) -> dict[str, Any]:
    """The media screen must fire on real holdout media: every registry audio fingerprint (and a gain-shifted,
    trimmed copy of up to 8 of them) and every holdout image pHash (bit-flipped by 3) must be caught; a synthetic
    tone and an unrelated hash must not."""

    totals = Counter()
    for bank, item, fp in holdouts.audio_samples[:64]:
        totals["audio_probes"] += 1
        if holdouts.audio.match(fp):
            totals["audio_caught"] += 1
    for bank, item, fp in holdouts.audio_samples[:8]:
        if np.count_nonzero(fp) > 3 * FP_MIN_OVERLAP:
            totals["audio_trimmed_probes"] += 1
            if holdouts.audio.match(fp[FP_MIN_OVERLAP // 2:]):
                totals["audio_trimmed_caught"] += 1
    t = np.arange(FP_RATE * 3) / FP_RATE
    tone = (0.2 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    tone_hit = bool(holdouts.audio.match(audio_fingerprint(tone)))
    for ref, bank, item in holdouts.phashes[:64]:
        totals["phash_probes"] += 1
        if holdouts.screen_media("0" * 64, phash=ref ^ 0b111):
            totals["phash_caught"] += 1
    unrelated = holdouts.screen_media("f" * 64, phash=0x5A5A5A5A5A5A5A5A) if holdouts.phashes else []
    live = (totals["audio_caught"] == totals["audio_probes"] and totals["audio_trimmed_caught"] == totals["audio_trimmed_probes"]
            and totals["phash_caught"] == totals["phash_probes"] and not tone_hit
            and not [h for h in unrelated if h["rule"] == "phash"])
    return {"totals": dict(totals), "tone_falsely_caught": tone_hit, "media_screen_is_live": live}


def holdouts_liveness(holdouts: Holdouts) -> dict[str, Any]:
    text = dcb.screen_liveness(holdouts.index)
    media = media_liveness(holdouts)
    return {"text": {k: text[k] for k in ("totals", "clean_controls_falsely_caught", "screen_is_live")},
            "media": media, "live": bool(text["screen_is_live"] and media["media_screen_is_live"])}


def latest_registry(corpus_root: Path) -> Path:
    found = sorted((corpus_root / "holdouts").glob("HOLDOUT_REGISTRY_v*.json"),
                   key=lambda p: int(re.search(r"_v(\d+)\.json$", p.name).group(1)))
    if not found:
        raise Refusal(f"no HOLDOUT_REGISTRY_v*.json under {corpus_root / 'holdouts'}: run registry-build first")
    return found[-1]


# ======================================================================================================================
# Known state (what earlier versions and earlier runs already hold).
# ======================================================================================================================


class Known:
    """Everything the planner compares a new item against.  Loaded from PostgreSQL (run) or empty / read-only
    (dry-run).  v1 = the adopted 1.0.1 corpus: its sidecar sha256 and exact text keys."""

    def __init__(self) -> None:
        self.asset_sha: dict[str, str] = {}               # sha -> where ('intake:<batch>' | 'v1')
        self.text_keys: dict[str, str] = {}               # text_key -> item id / 'v1:<rid>'
        self.identity: dict[str, tuple[str, str | None]] = {}
        self.owner_sessions: dict[str, str] = {}          # sealed session -> split
        self.neardup = NearDupIndex()
        self.audio = AudioFpIndex()

    def load_v1_index(self, path: Path) -> None:
        if not path.is_file():
            return
        body = dcb.load_json(path)
        for sha in body.get("media_sha256", []):
            self.asset_sha.setdefault(sha, "v1")
        for key in body.get("text_keys", []):
            self.text_keys.setdefault(key, "v1")


def build_v1_index(v1_dir: Path, out: Path) -> dict[str, Any]:
    rows_path = v1_dir / "staging" / "rows.jsonl"
    if not rows_path.is_file():
        raise Refusal(f"v1 staging rows missing at {rows_path}")
    media: set[str] = set()
    keys: set[str] = set()
    count = 0
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            for side in row.get("sidecars") or []:
                if side.get("sha256"):
                    media.add(str(side["sha256"]).lower())
            for field in ("instruction", "response"):
                key = text_key(str(row.get(field) or ""))
                if key:
                    keys.add(key)
    body = {"schema": f"{SCHEMA}/v1_dedupe_index", "identity": tool_identity(), "rows_path": str(rows_path),
            "rows_sha256": file_sha256(rows_path), "rows": count, "media_sha256": sorted(media), "text_keys": sorted(keys)}
    sha = dcb.write_json(out, body, seal_field="index_sha256")
    return {"path": str(out), "index_sha256": sha, "rows": count, "media": len(media), "text_keys": len(keys)}


# ======================================================================================================================
# The planner: candidate files -> items with screen / group / split / dedupe / status.  Shared by run and dry-run.
# ======================================================================================================================


def probe_asset(path: Path, ext: str, modality: str, *, want_loudness: bool) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    if modality in ("speech", "audio", "music", "video"):
        probe = ffprobe_facts(path)
        facts["probe"] = probe
        facts["duration_s"] = probe.get("duration")
        facts["sample_rate"] = probe.get("audio_sample_rate")
        facts["channels"] = probe.get("audio_channels")
        if probe.get("ok") and probe.get("has_audio"):
            pcm = decode_pcm(path)
            facts["fp"] = audio_fingerprint(pcm)
            facts["stats"] = audio_stats(pcm)
            if want_loudness:
                facts["loudness_lufs"] = loudness_lufs(path)
        if modality == "video" and probe.get("has_video"):
            facts["keyframes"] = video_keyframe_phashes(path)
            facts["width"], facts["height"] = probe.get("width"), probe.get("height")
    elif modality == "image":
        try:
            facts.update(image_hashes(path))
        except Exception as exc:  # unreadable image: recorded, item rejected
            facts["image_error"] = repr(exc)[:200]
    return facts


def build_items_for_asset(asset: dict[str, Any], policy: Mapping[str, Any], transcript: Mapping[str, Any] | None,
                          text_rows: list[dict[str, str]] | None) -> list[dict[str, Any]]:
    """One item per audio/image/video/mesh asset (per segment for segmented owner speech), one per text row."""

    modality = asset["modality"]
    base = {"asset_sha256": asset["sha256"], "modality": modality, "batch_id": policy["batch_id"],
            "licence": policy["licence"], "consent": policy["consent"], "flags": list(policy["flags"]) + list(asset.get("flags", []))}
    route = policy.get("route") or {"speech": "speech_owner_real" if policy["owner"] else "speech_external",
                                    "audio": "audio", "music": "music", "image": "image", "video": "video",
                                    "mesh": "mesh", "text": "text", "document": "document"}.get(modality, "unknown")
    items: list[dict[str, Any]] = []
    if modality in ("text", "document"):
        for i, row in enumerate(text_rows or []):
            item = {**base, "route": route, "instruction": row["instruction"], "response": row["response"], "row": i}
            items.append(item)
    else:
        segments = asset.get("segments") or [None]
        for seg in segments:
            item = {**base, "route": route, "segment": list(seg) if seg else None}
            if transcript and seg is None:
                item["transcript"] = transcript["transcript"]
                item["transcript_tool"] = transcript.get("tool")
                item["transcript_status"] = "auto"
                item["transcript_conf"] = None
            items.append(item)
    for item in items:
        item["item_id"] = canonical_sha256({k: item.get(k) for k in ("asset_sha256", "route", "row", "segment")})
        speaker = (transcript or {}).get("speaker") or policy.get("speaker")
        item["speaker"] = speaker
        item["voice"] = voice_label(policy, speaker, modality)
        item["session_id"] = policy["session_id"]
        item["owner"] = policy["owner"]
    return items


def voice_label(policy: Mapping[str, Any], speaker: str | None, modality: str) -> str | None:
    if modality != "speech":
        return None
    if policy.get("voice"):
        return str(policy["voice"]).replace("{speaker}", str(speaker or "unknown"))
    if policy["owner"]:
        return "owner-real"
    if speaker and str(speaker).startswith("LS-"):
        return str(speaker)
    if speaker == "LJ":
        return "LJ"
    return f"external:{speaker}" if speaker else "external:unknown"


def identity_keys(item: Mapping[str, Any], asset: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    keys = [f"asset:{asset['sha256']}"]
    if asset.get("derived_from"):
        keys.append(f"asset:{asset['derived_from']}")
    for text in (item.get("transcript"), item.get("instruction"), item.get("response")):
        key = wording_key(text)
        if key:
            keys.append(key)
    group_by = policy["group_by"]
    if group_by == "batch":
        keys.append(f"group:batch:{policy['batch_id']}")
    elif group_by == "session":
        keys.append(f"group:session:{policy['session_id']}")
    elif group_by == "speaker" and item.get("speaker"):
        keys.append(f"group:speaker:{item['speaker']}")
    return sorted(set(keys))


def item_text(item: Mapping[str, Any]) -> str:
    return "\n\n".join(t for t in (item.get("instruction"), item.get("response"), item.get("transcript")) if t)


def plan_items(assets: list[dict[str, Any]], policies: Mapping[str, Mapping[str, Any]], transcripts: Mapping[str, Any],
               text_rows: Mapping[str, list[dict[str, str]]], holdouts: Holdouts, known: Known) -> list[dict[str, Any]]:
    """Screen, group, split and dedupe every item of ``assets``.  Deterministic: items are processed in item_id
    order; the near-duplicate and audio indexes grow only with items this pass admits."""

    items: list[dict[str, Any]] = []
    for asset in sorted(assets, key=lambda a: a["sha256"]):
        policy = policies[asset["batch_id"]]
        rows = build_items_for_asset(asset, policy, transcripts.get(asset["sha256"]), text_rows.get(asset["sha256"]))
        for item in rows:
            item["identity"] = identity_keys(item, asset, policy)
            item["_asset"] = asset
        items.extend(rows)
    # owner sessions: fixed split by session (plan v2 2.2), sealed sessions never move.
    owner_sessions = {it["session_id"] for it in items if it["owner"] and it["modality"] == "speech"}
    session_split = owner_session_splits(owner_sessions, known.owner_sessions) if owner_sessions else {}
    for it in items:
        it["fixed_split"] = session_split.get(it["session_id"]) if it["owner"] and it["modality"] == "speech" else None
    assign_groups(items, known.identity)
    seen_sha: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    local_near = NearDupIndex()
    local_audio = AudioFpIndex()   # source = asset sha: segments of one recording never match each other
    media_screens: dict[str, list[dict[str, Any]]] = {}
    audio_added: set[str] = set()
    for it in sorted(items, key=lambda x: x["item_id"]):
        asset = it["_asset"]
        it["split"] = final_split(it)
        screen: dict[str, Any] = {"text": [], "media": []}
        text = item_text(it)
        if text:
            screen["text"] = holdouts.screen_text(text)
        fp = asset.get("fp")
        if asset["sha256"] not in media_screens:
            found = holdouts.screen_media(asset["sha256"], phash=asset.get("phash"), keyframes=asset.get("keyframes") or (), fp=fp)
            if asset.get("derived_from"):
                found += [dict(h, via="parent") for h in holdouts.screen_media(asset["derived_from"])]
            media_screens[asset["sha256"]] = found
        screen["media"] = list(media_screens[asset["sha256"]])
        it["screen"] = screen
        tkey = text_key(text) if it["modality"] in ("text", "document") else None
        it["text_key"] = tkey
        signature = minhash_signature(text) if it["modality"] in ("text", "document") and text else None
        it["_signature"] = signature
        dup: dict[str, Any] | None = None
        first_of_asset = it.get("row") in (None, 0) and it.get("segment") is None
        if first_of_asset and asset["sha256"] in known.asset_sha:
            dup = {"rule": "asset_sha256", "of": known.asset_sha[asset["sha256"]]}
        elif first_of_asset and asset["sha256"] in seen_sha and seen_sha[asset["sha256"]] != it["item_id"]:
            dup = {"rule": "asset_sha256", "of": seen_sha[asset["sha256"]]}
        elif tkey and tkey in known.text_keys:
            dup = {"rule": "text_key", "of": known.text_keys[tkey]}
        elif tkey and tkey in seen_text:
            dup = {"rule": "text_key", "of": seen_text[tkey]}
        near = None
        if dup is None and signature is not None:
            near = known.neardup.query(signature) or local_near.query(signature)
        audio_near = None
        if dup is None and near is None and fp is not None and fp.size:
            audio_near = known.audio.match(fp, exclude_source=asset["sha256"]) or local_audio.match(fp, exclude_source=asset["sha256"])
        status, reasons = "admitted", []
        if screen["text"] or screen["media"]:
            status, reasons = "contaminated", [h["rule"] for h in screen["text"] + screen["media"]]
        elif dup is not None:
            status, reasons = "duplicate", [dup["rule"]]
            it["duplicate_of"] = dup["of"]
        elif near is not None:
            status, reasons = "duplicate_near", ["minhash"]
            it["duplicate_of"], it["near_jaccard"] = near[0], round(near[1], 4)
        elif audio_near is not None:
            status, reasons = "duplicate_near", ["audio_fingerprint"]
            it["duplicate_of"], it["near_audio"] = audio_near["item"], audio_near
        else:
            review = review_reasons(it, asset)
            if review:
                status, reasons = "review", review
            elif it["modality"] == "speech" and not it.get("transcript"):
                status, reasons = "candidate", ["awaiting_transcript"]
            elif it["split"] != "train":
                status = "held"
        it["status"], it["reasons"] = status, reasons
        if status in ("admitted", "held", "candidate", "review"):
            seen_sha.setdefault(asset["sha256"], it["item_id"])
            if tkey:
                seen_text.setdefault(tkey, it["item_id"])
            if signature is not None and status == "admitted":
                local_near.add(it["item_id"], signature)
            if fp is not None and fp.size and status in ("admitted", "held", "candidate") and asset["sha256"] not in audio_added:
                audio_added.add(asset["sha256"])
                local_audio.add(asset["sha256"], it["item_id"], fp)
    return items


REVIEW_FLAG_PREFIXES = ("licence", "third_party", "needs_tool", "unreadable", "not_utf8", "bad_json", "group_by_invalid",
                        "meta_unknown", "meta_not_object", "meta_unreadable", "probe_failed", "image_error",
                        "extension_mismatch", "derive_failed")


def review_reasons(item: Mapping[str, Any], asset: Mapping[str, Any]) -> list[str]:
    reasons = [f for f in item.get("flags", []) if f.startswith(REVIEW_FLAG_PREFIXES)]
    if item.get("split_conflict"):
        reasons.append("split_conflict")
    if item["modality"] == "speech" and item.get("segment") is None and (asset.get("duration_s") or 0) > LONG_UNSEGMENTED_S:
        reasons.append("unsegmented_long_clip")
    if item.get("transcript_status") == "review":
        reasons.append("transcript_review")
    return reasons


def third_party_flags(probe: Mapping[str, Any] | None, tags: Mapping[str, Any] | None) -> list[str]:
    tags = {str(k).lower(): v for k, v in (tags or {}).items()}
    named = [k for k in ("artist", "album_artist", "author", "composer", "copyright", "publisher") if tags.get(k)]
    return [f"third_party_metadata:{','.join(named)}"] if named else []


def container_tags(path: Path) -> dict[str, Any]:
    result = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)], timeout=60)
    if result.returncode != 0:
        return {}
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(result.stdout or "{}").get("format", {}).get("tags", {}) or {}
    return {}


def make_asset(path: Path, sha: str, modality_dir: str, batch_id: str, policy: Mapping[str, Any], *,
               store_path: str | None = None) -> dict[str, Any]:
    """Probe facts of one original asset (never writes)."""

    ext = path.suffix.lower().lstrip(".")
    sniffed = sniff_kind(path)
    modality = detect_modality(ext, modality_dir, None)
    flags: list[str] = []
    if sniffed and ext and ext not in SNIFF_FAMILIES.get(sniffed, {sniffed}):
        flags.append(f"extension_mismatch:{ext}->{sniffed}")
    facts = probe_asset(path, ext, modality, want_loudness=bool(policy["owner"] and modality == "speech"))
    if modality in ("speech", "audio", "music", "video"):
        probe = facts.get("probe") or {}
        if not probe.get("ok"):
            flags.append("probe_failed")
        elif modality == "video" and probe.get("has_audio"):
            flags.append("video_has_audio_track")
        elif modality != "video" and not probe.get("has_audio"):
            flags.append("probe_failed:no_audio_stream")
        if modality in ("audio", "music", "video"):
            flags += third_party_flags(probe, container_tags(path))
    if facts.get("image_error"):
        flags.append("image_error")
    asset = {"sha256": sha, "ext": ext or (sniffed or "bin"), "bytes": path.stat().st_size, "modality": modality,
             "batch_id": batch_id, "path": str(path), "store_path": store_path, "derived_from": None,
             "duration_s": facts.get("duration_s"), "sample_rate": facts.get("sample_rate"), "channels": facts.get("channels"),
             "width": facts.get("width"), "height": facts.get("height"), "phash": facts.get("phash"), "dhash": facts.get("dhash"),
             "keyframes": facts.get("keyframes"), "fp": facts.get("fp"), "stats": facts.get("stats"),
             "loudness_lufs": facts.get("loudness_lufs"), "probe": facts.get("probe") or {}, "flags": flags}
    if policy["owner"] and modality == "speech" and (asset["duration_s"] or 0) > SEG_MAX_S:
        asset["segments"] = plan_segments(float(asset["duration_s"]), detect_silences(path))
    return asset


# ======================================================================================================================
# Reports.
# ======================================================================================================================


def summarise(items: list[dict[str, Any]], assets: list[dict[str, Any]]) -> dict[str, Any]:
    by_asset = {a["sha256"]: a for a in assets}

    def seconds(it: Mapping[str, Any]) -> float:
        seg = it.get("segment")
        if seg:
            return float(seg[1] - seg[0])
        return float(by_asset[it["asset_sha256"]].get("duration_s") or 0.0) if it.get("row") in (None, 0) else 0.0

    out: dict[str, Any] = {"items": len(items), "assets": len(assets), "by_batch": {}}
    for key in ("status", "modality", "split"):
        out[f"by_{key}"] = dict(sorted(Counter(it[key] for it in items).items()))
    hours = defaultdict(float)
    for it in items:
        hours[f"{it['status']}:{it['split']}"] += seconds(it) / 3600.0
    out["audio_hours_by_status_split"] = {k: round(v, 3) for k, v in sorted(hours.items())}
    out["audio_hours_total"] = round(sum(hours.values()), 3)
    out["reasons"] = dict(Counter(r.split(":")[0] for it in items for r in it.get("reasons", [])).most_common())
    out["screen_hits_by_bank"] = dict(Counter(h.get("source") for it in items for h in it["screen"]["text"] + it["screen"]["media"]).most_common())
    batches = defaultdict(list)
    for it in items:
        batches[it["batch_id"]].append(it)
    for batch, rows in sorted(batches.items()):
        out["by_batch"][batch] = {
            "items": len(rows),
            "status": dict(Counter(r["status"] for r in rows)),
            "split": dict(Counter(r["split"] for r in rows)),
            "status_split": dict(Counter(f"{r['status']}:{r['split']}" for r in rows)),
            "audio_hours": round(sum(seconds(r) for r in rows) / 3600.0, 3),
            "admitted_hours": round(sum(seconds(r) for r in rows if r["status"] == "admitted") / 3600.0, 3),
            "held_hours": round(sum(seconds(r) for r in rows if r["status"] == "held") / 3600.0, 3),
            "groups": len({r["group_id"] for r in rows}),
            "voices": len({r.get("voice") for r in rows if r.get("voice")}),
            "consent": dict(Counter(r["consent"] for r in rows)),
            "licence": dict(Counter(r["licence"] for r in rows)),
        }
    return out


def public_item(it: Mapping[str, Any]) -> dict[str, Any]:
    asset = it["_asset"]
    row = {k: v for k, v in it.items() if not k.startswith("_") and k not in ("identity", "fixed_split", "group_split")}
    row["identity"] = list(it["identity"])
    row["asset_path"] = asset.get("path")
    row["store_path"] = asset.get("store_path")
    row["duration_s"] = asset.get("duration_s")
    return row


# ======================================================================================================================
# PostgreSQL registry (raw SQL, psycopg2).
# ======================================================================================================================

DDL = r"""
CREATE TABLE IF NOT EXISTS holdout_registry (
  version        INTEGER PRIMARY KEY,
  path           TEXT NOT NULL,
  sha256         CHAR(64) NOT NULL UNIQUE,
  created_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by     TEXT NOT NULL,
  banks          JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS corpus_version (
  version_no               INTEGER PRIMARY KEY,
  parent_version_no        INTEGER REFERENCES corpus_version(version_no),
  status                   TEXT NOT NULL CHECK (status IN ('open','sealed')),
  manifest_path            TEXT,
  manifest_sha256          CHAR(64) UNIQUE,
  archive_path             TEXT,
  archive_sha256           CHAR(64),
  holdout_registry_version INTEGER REFERENCES holdout_registry(version),
  tool_version             TEXT NOT NULL,
  tool_sha256              CHAR(64) NOT NULL,
  counts                   JSONB NOT NULL DEFAULT '{}'::jsonb,
  sealed_utc               TIMESTAMPTZ,
  sealed_by                TEXT
);
CREATE TABLE IF NOT EXISTS source_batch (
  batch_id      TEXT PRIMARY KEY,
  drop_root     TEXT NOT NULL,
  modality_hint TEXT NOT NULL,
  meta          JSONB NOT NULL DEFAULT '{}'::jsonb,
  source        TEXT NOT NULL DEFAULT 'cyril-drop',
  licence       TEXT NOT NULL,
  licence_basis TEXT NOT NULL,
  consent       TEXT NOT NULL CHECK (consent IN ('owner','third_party_licensed','unspecified')),
  speaker       TEXT,
  session_id    TEXT,
  received_utc  TIMESTAMPTZ NOT NULL DEFAULT now(),
  archived_dir  TEXT
);
CREATE TABLE IF NOT EXISTS intake_file (
  id               BIGSERIAL PRIMARY KEY,
  batch_id         TEXT NOT NULL REFERENCES source_batch(batch_id),
  drop_path        TEXT NOT NULL UNIQUE,
  first_seen_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  stable_since_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
  size_bytes       BIGINT NOT NULL,
  mtime_epoch      DOUBLE PRECISION NOT NULL,
  stable_scans     INTEGER NOT NULL DEFAULT 0,
  sha256           CHAR(64),
  status           TEXT NOT NULL CHECK (status IN ('seen','settled','hashed','duplicate','archived','rejected','error','gone')),
  duplicate_of     CHAR(64),
  error_text       TEXT
);
CREATE INDEX IF NOT EXISTS intake_file_status_idx ON intake_file(status);
CREATE TABLE IF NOT EXISTS asset (
  sha256        CHAR(64) PRIMARY KEY,
  ext           TEXT NOT NULL,
  bytes         BIGINT NOT NULL,
  modality      TEXT NOT NULL CHECK (modality IN ('speech','audio','music','image','video','text','document','mesh','unknown')),
  store_path    TEXT NOT NULL,
  archive_path  TEXT,
  batch_id      TEXT NOT NULL REFERENCES source_batch(batch_id),
  derived_from  CHAR(64) REFERENCES asset(sha256),
  derivation    TEXT,
  probe         JSONB NOT NULL DEFAULT '{}'::jsonb,
  duration_s    DOUBLE PRECISION,
  sample_rate   INTEGER,
  channels      INTEGER,
  width         INTEGER,
  height        INTEGER,
  phash         BIT(64),
  dhash         BIT(64),
  audio_fp      BYTEA,
  loudness_lufs REAL,
  created_utc   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS asset_modality_idx ON asset(modality);
CREATE INDEX IF NOT EXISTS asset_batch_idx    ON asset(batch_id);
CREATE TABLE IF NOT EXISTS item (
  item_id           CHAR(64) PRIMARY KEY,
  asset_sha256      CHAR(64) REFERENCES asset(sha256),
  modality          TEXT NOT NULL,
  route             TEXT NOT NULL,
  row_no            INTEGER,
  segment_start_s   DOUBLE PRECISION,
  segment_end_s     DOUBLE PRECISION,
  instruction       TEXT,
  response          TEXT,
  transcript        TEXT,
  transcript_conf   REAL,
  transcript_status TEXT NOT NULL DEFAULT 'none'
                    CHECK (transcript_status IN ('none','auto','review','accepted','rejected')),
  transcript_tool   TEXT,
  text_key          CHAR(64),
  minhash           BYTEA,
  group_id          CHAR(64),
  split             TEXT CHECK (split IN ('train','val','test')),
  status            TEXT NOT NULL CHECK (status IN
                    ('candidate','review','admitted','held','contaminated','duplicate','duplicate_near','rejected')),
  reasons           JSONB NOT NULL DEFAULT '[]'::jsonb,
  licence           TEXT NOT NULL,
  consent           TEXT NOT NULL,
  speaker           TEXT,
  voice             TEXT,
  session_id        TEXT,
  screen            JSONB NOT NULL DEFAULT '{}'::jsonb,
  duplicate_of      TEXT,
  holdout_registry_version INTEGER REFERENCES holdout_registry(version),
  corpus_version_no INTEGER REFERENCES corpus_version(version_no),
  created_utc       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_utc       TIMESTAMPTZ NOT NULL DEFAULT now(),
  review_note       TEXT
);
CREATE INDEX IF NOT EXISTS item_status_idx  ON item(status);
CREATE INDEX IF NOT EXISTS item_version_idx ON item(corpus_version_no);
CREATE INDEX IF NOT EXISTS item_textkey_idx ON item(text_key);
CREATE TABLE IF NOT EXISTS item_identity (
  item_id  CHAR(64) NOT NULL REFERENCES item(item_id),
  key      TEXT NOT NULL,
  PRIMARY KEY (item_id, key)
);
CREATE INDEX IF NOT EXISTS item_identity_key_idx ON item_identity(key);
CREATE TABLE IF NOT EXISTS minhash_band (
  item_id  CHAR(64) NOT NULL REFERENCES item(item_id),
  band     SMALLINT NOT NULL,
  hash     BIGINT   NOT NULL,
  PRIMARY KEY (item_id, band)
);
CREATE INDEX IF NOT EXISTS minhash_band_lookup_idx ON minhash_band(band, hash);
CREATE TABLE IF NOT EXISTS intake_step (
  subject     TEXT NOT NULL,
  step        TEXT NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('done','failed','queued')),
  detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
  done_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (subject, step)
);
CREATE TABLE IF NOT EXISTS clone_reference (
  speaker       TEXT NOT NULL,
  asset_sha256  CHAR(64) NOT NULL REFERENCES asset(sha256),
  session_id    TEXT NOT NULL,
  split         TEXT NOT NULL CHECK (split = 'train'),
  chosen_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  chosen_by     TEXT NOT NULL,
  agreement_wer REAL NOT NULL,
  loudness_lufs REAL,
  PRIMARY KEY (speaker, asset_sha256)
);
CREATE TABLE IF NOT EXISTS intake_run (
  run_id      BIGSERIAL PRIMARY KEY,
  started_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_utc   TIMESTAMPTZ,
  mode        TEXT NOT NULL,
  tool_sha256 CHAR(64) NOT NULL,
  holdout_registry_version INTEGER,
  summary     JSONB NOT NULL DEFAULT '{}'::jsonb,
  ok          BOOLEAN
);
"""


def pg_connect():
    import psycopg2  # noqa: E402

    password = os.environ.get(PG_PASSWORD_ENV, "")
    if not password:
        raise Refusal(f"{PG_PASSWORD_ENV} is empty (vault key chimera_corpus_pg; on the server "
                      "/home/cereal/.config/chimera-intake/env)")
    options = "-c statement_timeout=30000 -c lock_timeout=5000"
    schema = os.environ.get(PG_SCHEMA_ENV, "")
    if schema:  # tests only: an isolated schema inside chimera_corpus
        if not re.fullmatch(r"intake_test_[a-z0-9_]{1,40}", schema):
            raise Refusal(f"{PG_SCHEMA_ENV}={schema!r} is not an intake_test_* schema")
        options += f" -c search_path={schema}"
    return psycopg2.connect(host=os.environ.get(PG_HOST_ENV, "127.0.0.1"), port=int(os.environ.get(PG_PORT_ENV, "25490")),
                            dbname=os.environ.get(PG_DB_ENV, "chimera_corpus"), user=os.environ.get(PG_USER_ENV, "chimera_corpus"),
                            password=password, connect_timeout=5, options=options, application_name=f"{TOOL}-{VERSION}")


@contextlib.contextmanager
def pg_session() -> Iterator[Any]:
    conn = pg_connect()
    try:
        yield conn
    finally:
        conn.close()


def bits64(value: int | None) -> str | None:
    return None if value is None else format(int(value) & ((1 << 64) - 1), "064b")


class PgStore:
    """Every statement parameterised; one commit per unit of work; the caller owns the connection."""

    def __init__(self, conn: Any):
        self.conn = conn

    def init(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(DDL)
        self.conn.commit()

    def register_holdouts(self, registry: Mapping[str, Any], path: str, agent: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO holdout_registry (version, path, sha256, created_by, banks) VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (version) DO NOTHING",
                        (registry["version"], path, registry["registry_sha256"], agent, json.dumps(registry["banks"])))
            cur.execute("SELECT sha256 FROM holdout_registry WHERE version = %s", (registry["version"],))
            stored = cur.fetchone()[0]
        if stored != registry["registry_sha256"]:
            self.conn.rollback()
            raise Refusal(f"holdout registry v{registry['version']} already registered with sha {stored}")
        self.conn.commit()

    def file_rows(self) -> dict[str, dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT drop_path, batch_id, size_bytes, mtime_epoch, stable_scans, "
                        "extract(epoch FROM stable_since_utc), sha256, status FROM intake_file")
            return {r[0]: {"batch_id": r[1], "size": r[2], "mtime": r[3], "stable_scans": r[4], "stable_since": float(r[5]),
                           "sha256": r[6], "status": r[7]} for r in cur.fetchall()}

    def upsert_batch(self, policy: Mapping[str, Any], drop_root: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO source_batch (batch_id, drop_root, modality_hint, meta, source, licence, licence_basis, consent, "
                "speaker, session_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (batch_id) DO UPDATE SET "
                "meta = EXCLUDED.meta, licence = EXCLUDED.licence, licence_basis = EXCLUDED.licence_basis, "
                "consent = EXCLUDED.consent, speaker = EXCLUDED.speaker, session_id = EXCLUDED.session_id",
                (policy["batch_id"], drop_root, policy["modality_dir"], json.dumps(policy["meta"]), policy["source"],
                 policy["licence"], policy["licence_basis"], policy["consent"], policy.get("speaker"), policy["session_id"]))
        self.conn.commit()

    def observe_file(self, drop_path: str, batch_id: str, size: int, mtime: float, now: float, prev: Mapping[str, Any] | None) -> None:
        with self.conn.cursor() as cur:
            if prev is None:
                cur.execute("INSERT INTO intake_file (batch_id, drop_path, size_bytes, mtime_epoch, stable_scans, status, "
                            "first_seen_utc, last_seen_utc, stable_since_utc) VALUES (%s, %s, %s, %s, 0, 'seen', "
                            "to_timestamp(%s), to_timestamp(%s), to_timestamp(%s))",
                            (batch_id, drop_path, size, mtime, now, now, now))
            elif prev["size"] == size and abs(prev["mtime"] - mtime) < 1e-6:
                cur.execute("UPDATE intake_file SET last_seen_utc = to_timestamp(%s), stable_scans = stable_scans + 1 "
                            "WHERE drop_path = %s", (now, drop_path))
            else:  # changed (or re-written after archiving): a new version of the file, ingested again once settled
                cur.execute("UPDATE intake_file SET last_seen_utc = to_timestamp(%s), stable_since_utc = to_timestamp(%s), "
                            "size_bytes = %s, mtime_epoch = %s, stable_scans = 0, sha256 = NULL, status = 'seen' "
                            "WHERE drop_path = %s", (now, now, size, mtime, drop_path))
        self.conn.commit()

    def mark_gone(self, present: set[str]) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT drop_path FROM intake_file WHERE status IN ('seen','settled')")
            gone = [r[0] for r in cur.fetchall() if r[0] not in present]
            for path in gone:
                cur.execute("UPDATE intake_file SET status = 'gone' WHERE drop_path = %s", (path,))
        self.conn.commit()
        return len(gone)

    def set_file(self, drop_path: str, status: str, sha256: str | None = None, duplicate_of: str | None = None,
                 error: str | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE intake_file SET status = %s, sha256 = COALESCE(%s, sha256), duplicate_of = %s, error_text = %s "
                        "WHERE drop_path = %s", (status, sha256, duplicate_of, error, drop_path))
        self.conn.commit()

    def asset_exists(self, sha: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM asset WHERE sha256 = %s", (sha,))
            return cur.fetchone() is not None

    def insert_asset(self, asset: Mapping[str, Any], derivation: str | None = None) -> None:
        fp = asset.get("fp")
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO asset (sha256, ext, bytes, modality, store_path, archive_path, batch_id, derived_from, derivation, "
                "probe, duration_s, sample_rate, channels, width, height, phash, dhash, audio_fp, loudness_lufs) VALUES "
                "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::bit(64), %s::bit(64), %s, %s) "
                "ON CONFLICT (sha256) DO NOTHING",
                (asset["sha256"], asset["ext"], asset["bytes"], asset["modality"], asset["store_path"], asset.get("archive_path"),
                 asset["batch_id"], asset.get("derived_from"), derivation, json.dumps(asset.get("probe") or {}, default=str),
                 asset.get("duration_s"), asset.get("sample_rate"), asset.get("channels"), asset.get("width"), asset.get("height"),
                 bits64(asset.get("phash")), bits64(asset.get("dhash")),
                 fp.astype("<u4").tobytes() if isinstance(fp, np.ndarray) and fp.size else None, asset.get("loudness_lufs")))
        self.conn.commit()

    def load_asset(self, sha: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT sha256, ext, bytes, modality, store_path, archive_path, batch_id, derived_from, probe, duration_s, "
                        "sample_rate, channels, width, height, phash::text, dhash::text, audio_fp, loudness_lufs "
                        "FROM asset WHERE sha256 = %s", (sha,))
            r = cur.fetchone()
        if r is None:
            return None
        fp = np.frombuffer(bytes(r[16]), dtype="<u4").astype(np.uint32) if r[16] is not None else None
        return {"sha256": r[0], "ext": r[1], "bytes": r[2], "modality": r[3], "store_path": r[4], "archive_path": r[5],
                "batch_id": r[6], "derived_from": r[7], "probe": r[8], "duration_s": r[9], "sample_rate": r[10],
                "channels": r[11], "width": r[12], "height": r[13], "phash": int(r[14], 2) if r[14] else None,
                "dhash": int(r[15], 2) if r[15] else None, "fp": fp, "loudness_lufs": r[17], "path": r[4], "flags": []}

    def step_done(self, subject: str, step: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM intake_step WHERE subject = %s AND step = %s AND status = 'done'", (subject, step))
            return cur.fetchone() is not None

    def mark_step(self, subject: str, step: str, status: str, detail: Mapping[str, Any] | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO intake_step (subject, step, status, detail) VALUES (%s, %s, %s, %s) ON CONFLICT (subject, step) "
                        "DO UPDATE SET status = EXCLUDED.status, detail = EXCLUDED.detail, done_utc = now()",
                        (subject, step, status, json.dumps(detail or {}, default=str)))
        self.conn.commit()

    def insert_item(self, it: Mapping[str, Any], registry_version: int) -> bool:
        seg = it.get("segment") or (None, None)
        sig = it.get("_signature")
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO item (item_id, asset_sha256, modality, route, row_no, segment_start_s, segment_end_s, instruction, "
                "response, transcript, transcript_conf, transcript_status, transcript_tool, text_key, minhash, group_id, split, "
                "status, reasons, licence, consent, speaker, voice, session_id, screen, duplicate_of, holdout_registry_version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (item_id) DO NOTHING RETURNING item_id",
                (it["item_id"], it["asset_sha256"], it["modality"], it["route"], it.get("row"), seg[0], seg[1],
                 it.get("instruction"), it.get("response"), it.get("transcript"), it.get("transcript_conf"),
                 it.get("transcript_status", "none"), it.get("transcript_tool"), it.get("text_key"),
                 sig.astype("<u8").tobytes() if isinstance(sig, np.ndarray) else None, it["group_id"], it["split"], it["status"],
                 json.dumps(it.get("reasons", [])), it["licence"], it["consent"], it.get("speaker"), it.get("voice"),
                 it.get("session_id"), json.dumps(it["screen"], default=str), it.get("duplicate_of"), registry_version))
            inserted = cur.fetchone() is not None
            if inserted:
                for key in it["identity"]:
                    cur.execute("INSERT INTO item_identity (item_id, key) VALUES (%s, %s) ON CONFLICT DO NOTHING", (it["item_id"], key))
                if isinstance(sig, np.ndarray) and it["status"] == "admitted":
                    for band, h in enumerate(minhash_bands(sig)):
                        cur.execute("INSERT INTO minhash_band (item_id, band, hash) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                                    (it["item_id"], band, h))
        self.conn.commit()
        return inserted

    def load_known(self, v1_index: Path | None) -> Known:
        known = Known()
        if v1_index is not None:
            known.load_v1_index(v1_index)
        with self.conn.cursor() as cur:
            cur.execute("SELECT a.sha256, a.batch_id FROM asset a")
            for sha, batch in cur.fetchall():
                known.asset_sha[sha] = f"intake:{batch}"
            cur.execute("SELECT text_key, item_id FROM item WHERE text_key IS NOT NULL AND status IN ('admitted','held','review','candidate')")
            for key, item_id in cur.fetchall():
                known.text_keys.setdefault(key, item_id)
            cur.execute("SELECT ii.key, i.group_id, CASE WHEN i.corpus_version_no IS NULL THEN NULL ELSE i.split END "
                        "FROM item_identity ii JOIN item i ON i.item_id = ii.item_id "
                        "WHERE i.status IN ('admitted','held','review','candidate')")
            for key, group_id, split in cur.fetchall():
                prev = known.identity.get(key)
                if prev is None or (split and (prev[1] is None or SPLIT_RANK[split] > SPLIT_RANK[prev[1]])):
                    known.identity[key] = (min(group_id, prev[0]) if prev else group_id, split or (prev[1] if prev else None))
            cur.execute("SELECT DISTINCT session_id, split FROM item WHERE voice = 'owner-real' AND corpus_version_no IS NOT NULL")
            for session, split in cur.fetchall():
                known.owner_sessions[session] = split
            cur.execute("SELECT item_id, minhash FROM item WHERE status = 'admitted' AND minhash IS NOT NULL")
            for item_id, blob in cur.fetchall():
                known.neardup.add(item_id, np.frombuffer(bytes(blob), dtype="<u8").astype(np.uint64))
            cur.execute("SELECT i.item_id, a.batch_id, a.audio_fp FROM item i JOIN asset a ON a.sha256 = i.asset_sha256 "
                        "WHERE i.status IN ('admitted','held','candidate') AND i.segment_start_s IS NULL AND a.audio_fp IS NOT NULL")
            for item_id, batch, blob in cur.fetchall():
                known.audio.add(f"intake:{batch}", item_id, np.frombuffer(bytes(blob), dtype="<u4").astype(np.uint32))
        return known

    def items_where(self, sql_where: str, params: tuple) -> list[dict[str, Any]]:
        cols = ("item_id", "asset_sha256", "modality", "route", "row_no", "segment_start_s", "segment_end_s", "instruction",
                "response", "transcript", "transcript_conf", "transcript_status", "transcript_tool", "text_key", "group_id",
                "split", "status", "reasons", "licence", "consent", "speaker", "voice", "session_id", "screen",
                "duplicate_of", "holdout_registry_version", "corpus_version_no")
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {', '.join(cols)} FROM item WHERE {sql_where} ORDER BY item_id", params)  # noqa: S608 (constant clause)
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ======================================================================================================================
# Filesystem effects (run only): archive + store copies, derivatives.
# ======================================================================================================================


def copy_verified(source: Path, destination: Path, sha: str) -> str:
    """Content-addressed copy, re-hashed (distill_corpus_build.store_media semantics, any extension).  The source is
    opened read-only and never touched otherwise."""

    if destination.is_file() and file_sha256(destination) == sha:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp)
    if file_sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise Refusal(f"{source} changed while it was being copied")
    os.replace(tmp, destination)
    return str(destination)


def store_path_for(store: Path, sha: str, ext: str) -> Path:
    return store / sha[:2] / f"{sha}.{ext or 'bin'}"


def make_derivatives(asset: dict[str, Any], store: Path) -> list[dict[str, Any]]:
    """16k-mono.flac + 32k-stereo.flac of an audio/video asset, and owner-speech segments (each 16k + 32k too).
    Content-addressed in the store, linked by derived_from; written only by `run`."""

    source = Path(asset["store_path"])
    out: list[dict[str, Any]] = []
    jobs: list[tuple[str, float | None, float | None]] = [(name, None, None) for name in DERIVATIONS]
    for start, end in asset.get("segments") or []:
        jobs.append(("segment.flac", start, end))
    scratch = store / "_tmp"
    for name, start, end in jobs:
        args = DERIVATIONS.get(name, ["-c:a", "flac"])
        tmp = scratch / f"{asset['sha256']}.{name}.{start}.{end}.flac"
        ffmpeg_derive(source, tmp, args, start=start, end=end)
        sha = file_sha256(tmp)
        dest = store_path_for(store, sha, "flac")
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
        out.append({"sha256": sha, "ext": "flac", "bytes": dest.stat().st_size, "modality": asset["modality"] if asset["modality"] != "video" else "audio",
                    "store_path": str(dest), "archive_path": None, "batch_id": asset["batch_id"], "derived_from": asset["sha256"],
                    "derivation": f"ffmpeg {name}" + (f" segment {start}-{end}" if start is not None else ""),
                    "segment": (start, end) if start is not None else None, "probe": {}, "duration_s": (end - start) if start is not None else asset.get("duration_s"),
                    "sample_rate": 16000 if name.startswith("16k") else 32000 if name.startswith("32k") else asset.get("sample_rate"),
                    "channels": 1 if name.startswith("16k") else 2 if name.startswith("32k") else asset.get("channels"), "flags": []})
    return out


# ======================================================================================================================
# Scanning.
# ======================================================================================================================


def walk_drop_root(drop_root: Path) -> Iterator[tuple[Path, tuple[str, ...]]]:
    for modality_dir in MODALITY_DIRS:
        base = drop_root / modality_dir
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SERVICE_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                yield path, path.relative_to(drop_root).parts


def read_meta(batch_dir: Path) -> tuple[dict[str, Any], list[str]]:
    meta_path = batch_dir / "_meta.json"
    if not meta_path.is_file():
        return {}, []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (meta if isinstance(meta, dict) else {}), ([] if isinstance(meta, dict) else ["meta_not_object"])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}, ["meta_unreadable"]


def batch_dir_of(drop_root: Path, batch_id: str) -> Path:
    return drop_root / batch_id if not batch_id.endswith("-loose") else drop_root / batch_id.split("/")[0]


# ======================================================================================================================
# Commands.
# ======================================================================================================================


@contextlib.contextmanager
def exclusive_lock(path: Path, wait_s: float = 0.0) -> Iterator[None]:
    import fcntl  # noqa: E402  (server only)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    deadline = time.time() + wait_s
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() >= deadline:
                handle.close()
                raise Refusal(f"lock {path} is held")
            time.sleep(1.0)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_holdouts_live(corpus_root: Path, registry: str | None) -> tuple[Holdouts, dict[str, Any]]:
    path = Path(registry) if registry else latest_registry(corpus_root)
    holdouts = Holdouts.load(path)
    live = holdouts_liveness(holdouts)
    if not live["live"]:
        raise Refusal(f"holdout screen of registry v{holdouts.version} is not live: {json.dumps(live)[:600]}")
    return holdouts, live


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        spec = [{"path": args.drop_root, "drop_zone": True}]
        return dry_run(args, spec)
    drop_root, corpus_root, store = Path(args.drop_root), Path(args.corpus_root), Path(args.corpus_root) / "store"
    queues = Path(args.queues)
    started = time.time()
    if not (drop_root / "_README.txt").is_file():
        # an unmounted share would look like an empty drop zone and mark every file 'gone'
        raise Refusal(f"{drop_root}/_README.txt missing: share not mounted or drop zone not initialised (dropzone-init)")
    with exclusive_lock(queues / ".intake.run.lock"), pg_session() as conn:
        db = PgStore(conn)
        holdouts, live = load_holdouts_live(corpus_root, args.registry)
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM holdout_registry WHERE sha256 = %s", (holdouts.sha256,))
            if cur.fetchone() is None:
                raise Refusal(f"registry v{holdouts.version} ({holdouts.sha256}) is not registered in PostgreSQL: run registry-build --register")
            cur.execute("INSERT INTO intake_run (mode, tool_sha256, holdout_registry_version) VALUES ('run', %s, %s) RETURNING run_id",
                        (_SELF_SHA256, holdouts.version))
            run_id = cur.fetchone()[0]
        conn.commit()
        now = float(args.now) if args.now is not None else time.time()
        files = db.file_rows()
        present: set[str] = set()
        policies: dict[str, dict[str, Any]] = {}
        settled: list[tuple[Path, str, str]] = []
        counts = Counter()
        for path, parts in walk_drop_root(drop_root):
            kind = classify_name(path.name)
            if kind != "data" or not is_item_file(path.name, parts[0]):
                counts[f"skipped_{kind if kind != 'data' else 'metadata'}"] += 1
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            prev = files.get(str(path))
            day = utc(prev["stable_since"] if prev else now)[:10]
            located = batch_for(parts, day)
            if located is None:
                counts["outside_batches"] += 1
                continue
            modality_dir, batch_id = located
            present.add(str(path))
            if batch_id not in policies:
                meta, meta_flags = read_meta(batch_dir_of(drop_root, batch_id))
                policies[batch_id] = batch_policy(batch_id, meta, modality_dir=modality_dir)
                policies[batch_id]["flags"] += meta_flags
                db.upsert_batch(policies[batch_id], str(drop_root))
            if prev and prev["status"] in ("archived", "duplicate", "rejected", "error") and prev["size"] == st.st_size and abs(prev["mtime"] - st.st_mtime) < 1e-6:
                counts["already_ingested"] += 1
                continue
            if prev and is_settled(size=st.st_size, mtime=st.st_mtime, prev_size=prev["size"], prev_mtime=prev["mtime"],
                                   stable_since=prev["stable_since"], now=now, name=path.name):
                settled.append((path, modality_dir, batch_id))
            db.observe_file(str(path), batch_id, st.st_size, st.st_mtime, now, prev)
            counts["observed"] += 1
        counts["gone"] = db.mark_gone(present)
        # ingest settled files ---------------------------------------------------------------------------------------
        new_assets: list[dict[str, Any]] = []
        for path, modality_dir, batch_id in settled:
            policy = policies[batch_id]
            try:
                sha = file_sha256(path)
                if file_sha256(path) != sha:
                    raise RuntimeError("file changed while hashing")
                if db.asset_exists(sha):
                    db.set_file(str(path), "duplicate", sha, duplicate_of=sha)
                    counts["duplicate_sha"] += 1
                    continue
                ext = path.suffix.lower().lstrip(".") or (sniff_kind(path) or "bin")
                archive = copy_verified(path, drop_root / "_received" / batch_id / f"{sha}.{ext}", sha)
                stored = copy_verified(path, store_path_for(store, sha, ext), sha)
                asset = make_asset(Path(stored), sha, modality_dir, batch_id, policy, store_path=stored)
                asset["archive_path"], asset["path"], asset["drop_path"] = archive, stored, str(path)
                db.insert_asset(asset)
                db.set_file(str(path), "archived", sha)
                new_assets.append(asset)
                counts["ingested"] += 1
            except Exception as exc:  # one bad file never stops the pass; it is recorded and retried after a change
                db.set_file(str(path), "error", error=repr(exc)[:500])
                counts["errors"] += 1
                log(f"ingest error {path}: {exc!r}")
        # derivatives ---------------------------------------------------------------------------------------------------
        derived_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for asset in new_assets:
            if asset["modality"] in ("speech", "audio", "music", "video") and (asset["probe"] or {}).get("has_audio") \
                    and not db.step_done(asset["sha256"], "derive"):
                try:
                    for d in make_derivatives(asset, store):
                        db.insert_asset(d, d["derivation"])
                        derived_by_parent[asset["sha256"]].append(d)
                    db.mark_step(asset["sha256"], "derive", "done", {"n": len(derived_by_parent[asset["sha256"]])})
                except Exception as exc:
                    db.mark_step(asset["sha256"], "derive", "failed", {"error": repr(exc)[:300]})
                    asset["flags"].append("derive_failed")
        # items ------------------------------------------------------------------------------------------------------------
        transcripts: dict[str, Any] = {}
        text_rows: dict[str, list[dict[str, str]]] = {}
        for batch_id in {a["batch_id"] for a in new_assets}:
            members = [a for a in new_assets if a["batch_id"] == batch_id]
            by_path = sidecar_transcripts(batch_dir_of(drop_root, batch_id), [Path(a["drop_path"]) for a in members])
            for a in members:
                if a["drop_path"] in by_path:
                    transcripts[a["sha256"]] = by_path[a["drop_path"]]
        for a in new_assets:
            if a["modality"] in ("text", "document"):
                rows, problem = extract_text_rows(Path(a["store_path"]), a["ext"])
                text_rows[a["sha256"]] = rows
                if problem:
                    a["flags"].append(problem)
                    text_rows[a["sha256"]] = [{"instruction": "", "response": ""}]
        for a in new_assets:  # segments become their own derived assets: items point at the segment asset
            if a.get("segments"):
                segs = [d for d in derived_by_parent.get(a["sha256"], []) if d.get("segment")]
                a["segment_assets"] = {tuple(d["segment"]): d for d in segs}
        known = db.load_known(Path(args.v1_index) if args.v1_index else None)
        for a in new_assets:  # registered a moment ago by this pass: they are the subject, not prior state
            if known.asset_sha.get(a["sha256"]) == f"intake:{a['batch_id']}":
                del known.asset_sha[a["sha256"]]
        items = plan_items(new_assets, policies, transcripts, text_rows, holdouts, known)
        for it in items:
            seg = it.get("segment")
            if seg and it["_asset"].get("segment_assets", {}).get(tuple(seg)):
                it["asset_sha256"] = it["_asset"]["segment_assets"][tuple(seg)]["sha256"]
            if db.insert_item(it, holdouts.version):
                counts[f"item_{it['status']}"] += 1
        pending = [it["item_id"] for it in items if it["status"] == "candidate" and "awaiting_transcript" in it["reasons"]]
        counts.update({f"review_{k}": v for k, v in apply_review(conn, drop_root, holdouts).items()})
        export_review(db, drop_root)
        summary = {"run_id": run_id, "counts": dict(counts), "items": summarise(items, new_assets) if items else {},
                   "pending_transcription": len(pending), "registry_version": holdouts.version,
                   "liveness": live, "wall_s": round(time.time() - started, 1), "utc": utc()}
        with conn.cursor() as cur:
            cur.execute("UPDATE intake_run SET ended_utc = now(), summary = %s, ok = true WHERE run_id = %s",
                        (json.dumps(summary, default=str), run_id))
        conn.commit()
    status_path = queues / "intake.status"
    write_json_atomic(status_path, {"tool": f"{TOOL} {VERSION}", "utc": utc(), "ok": True, "counts": dict(counts),
                                    "pending_transcription": len(pending), "registry_version": holdouts.version})
    if pending:
        write_jsonl_atomic(queues / "intake_transcribe.pending.jsonl", ({"item_id": i} for i in pending))
    log(f"run {run_id}: {dict(counts)}; pending transcription {len(pending)}")
    return summary


def merge_review_lines(existing: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The review file after a pass: one line per item still in review, in item order.  A line Cyril already has
    (any edit, decided or not) is kept exactly as he left it; new review items get a fresh line; items no longer in
    review (decided and applied) drop out."""

    mine = {str(r.get("item_id")): r for r in existing if isinstance(r, dict) and r.get("item_id")}
    out = []
    for r in sorted(current, key=lambda x: x["item_id"]):
        out.append(mine.get(r["item_id"]) or {"item_id": r["item_id"], "accepted": None, "transcript": r["transcript"],
                                             "reasons": r["reasons"], "status": r["status"], "modality": r["modality"]})
    return out


def read_review_file(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    return rows


def export_review(db: PgStore, drop_root: Path) -> int:
    rows = db.items_where("status = 'review' AND corpus_version_no IS NULL", ())
    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with db.conn.cursor() as cur:
        for r in rows:
            cur.execute("SELECT batch_id FROM asset WHERE sha256 = %s", (r["asset_sha256"],))
            got = cur.fetchone()
            by_batch[got[0] if got else "unknown"].append(r)
    for path in sorted((drop_root / "_review").glob("**/transcripts.jsonl")):  # batches with nothing left in review
        batch = str(path.parent.relative_to(drop_root / "_review")).replace(os.sep, "/")
        by_batch.setdefault(batch, [])
    for batch, members in by_batch.items():
        path = drop_root / "_review" / batch / "transcripts.jsonl"
        merged = merge_review_lines(read_review_file(path), members)
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in merged)
        if path.is_file() and path.read_text(encoding="utf-8-sig") == payload:
            continue  # unchanged: never rewrite a file Cyril may have open
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    return len(rows)


def apply_review(conn: Any, drop_root: Path, holdouts: Holdouts) -> Counter:
    """Cyril's decisions in _review/**/transcripts.jsonl: accepted false -> rejected; accepted true -> the item is
    approved (every review reason), its edited transcript (speech) is re-screened, and it becomes admitted / held by
    its split, or contaminated if the new text hits a holdout.  Only unsealed items in review change."""

    changed = Counter()
    for path in sorted((drop_root / "_review").glob("**/transcripts.jsonl")):
        for rec in read_review_file(path):
            if rec.get("accepted") not in (True, False):
                continue
            with conn.cursor() as cur:
                cur.execute("SELECT split, modality, transcript FROM item WHERE item_id = %s AND status = 'review' "
                            "AND corpus_version_no IS NULL", (str(rec.get("item_id")),))
                got = cur.fetchone()
                if got is None:
                    continue
                split, modality, old_transcript = got
                if rec["accepted"] is False:
                    cur.execute("UPDATE item SET status = 'rejected', review_note = %s, updated_utc = now() WHERE item_id = %s",
                                ("rejected in _review", rec["item_id"]))
                    changed["rejected"] += 1
                else:
                    text = str(rec.get("transcript") or "").strip() if modality == "speech" else (old_transcript or "")
                    hits = holdouts.screen_text(text) if text else []
                    status = "contaminated" if hits else ("admitted" if split == "train" else "held")
                    if modality == "speech" and not text:
                        status = "candidate"  # approved, but nothing to train speech-in on until transcribed
                    cur.execute("UPDATE item SET transcript = %s, transcript_status = %s, status = %s, review_note = %s, "
                                "screen = jsonb_set(screen, '{text}', %s::jsonb), updated_utc = now() WHERE item_id = %s",
                                (text or None, "accepted" if text else "none", status, "accepted in _review",
                                 json.dumps(hits), rec["item_id"]))
                    changed[status] += 1
            conn.commit()
    return changed


def cmd_import_review(args: argparse.Namespace) -> dict[str, Any]:
    drop_root = Path(args.drop_root)
    holdouts, _ = load_holdouts_live(Path(args.corpus_root), args.registry)
    with pg_session() as conn:
        changed = apply_review(conn, drop_root, holdouts)
        export_review(PgStore(conn), drop_root)
    log(f"import-review: {dict(changed)}")
    return dict(changed)


def dry_run(args: argparse.Namespace, spec: list[dict[str, Any]]) -> dict[str, Any]:
    """Plan v-next over local trees without writing anywhere but --out: no PostgreSQL writes, no copies, no
    derivatives, nothing under the drop zone.  Uses PostgreSQL read-only for the known state when reachable."""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    holdouts, live = load_holdouts_live(Path(args.corpus_root), args.registry)
    known = Known()
    known_source = "empty"
    if os.environ.get(PG_PASSWORD_ENV):
        try:
            with pg_session() as conn:
                conn.set_session(readonly=True)
                known = PgStore(conn).load_known(Path(args.v1_index) if args.v1_index else None)
                known_source = "postgresql (read-only)"
        except Exception as exc:  # unreachable registry: the dry run still plans against v1 + itself
            log(f"dry-run: PostgreSQL not used ({exc!r})")
    if known_source == "empty" and args.v1_index:
        known.load_v1_index(Path(args.v1_index))
        known_source = "v1 index file"
    assets: list[dict[str, Any]] = []
    policies: dict[str, dict[str, Any]] = {}
    transcripts: dict[str, Any] = {}
    text_rows: dict[str, list[dict[str, str]]] = {}
    sources_report = []
    for source in spec:
        root = Path(source["path"])
        entries: list[tuple[Path, str, str, dict[str, Any]]] = []
        if source.get("drop_zone"):
            for path, parts in walk_drop_root(root):
                if not is_item_file(path.name, parts[0]):
                    continue
                located = batch_for(parts, utc()[:10])
                if located:
                    meta, _ = read_meta(batch_dir_of(root, located[1]))
                    entries.append((path, located[0], located[1], meta))
        else:
            pattern = source.get("glob", "**/*")
            for path in sorted(root.glob(pattern)):
                if path.is_file() and is_item_file(path.name, source["modality"]):
                    entries.append((path, source["modality"], source["batch_id"], source.get("meta", {})))
            if source.get("limit"):
                entries = entries[: int(source["limit"])]
        report = {"path": str(root), "batch_id": source.get("batch_id"), "files": len(entries)}
        for path, modality_dir, batch_id, meta in entries:
            if batch_id not in policies:
                policies[batch_id] = batch_policy(batch_id, meta, modality_dir=modality_dir)
        # transcripts that ship with the files, plus an explicit per-source map
        if entries:
            batch_root = Path(source.get("transcript_root") or root)
            for path_text, rec in sidecar_transcripts(batch_root, [e[0] for e in entries]).items():
                transcripts[path_text] = rec
        for rec in (dcb.read_jsonl(Path(source["transcripts"])) if source.get("transcripts") else []):
            transcripts[str(rec["path"])] = {"transcript": rec["transcript"], "tool": rec.get("tool", "dry-run map"),
                                             **({"speaker": rec["speaker"]} if rec.get("speaker") else {})}
        jobs = [(e[0], e[1], e[2]) for e in entries]
        made = parallel_assets(jobs, policies, workers=int(args.workers))
        for asset in made:
            if asset["path"] in transcripts:
                transcripts[asset["sha256"]] = transcripts[asset["path"]]
            if asset["modality"] in ("text", "document"):
                rows, problem = extract_text_rows(Path(asset["path"]), asset["ext"])
                text_rows[asset["sha256"]] = rows or [{"instruction": "", "response": ""}]
                if problem:
                    asset["flags"].append(problem)
        report["assets"] = len(made)
        report["transcripts_found"] = sum(1 for a in made if a["sha256"] in transcripts)
        sources_report.append(report)
        assets.extend(made)
        log(f"dry-run: {root}: {len(made)} assets, {report['transcripts_found']} transcripts")
    unique: dict[str, dict[str, Any]] = {}
    duplicates_in_input = 0
    for a in assets:
        if a["sha256"] in unique:
            duplicates_in_input += 1
            continue
        unique[a["sha256"]] = a
    items = plan_items(list(unique.values()), policies, transcripts, text_rows, holdouts, known)
    summary = summarise(items, list(unique.values()))
    summary["identical_files_in_input"] = duplicates_in_input
    rows_sha, rows_n = write_jsonl_atomic(out / "dry_run_items.jsonl", (public_item(it) for it in sorted(items, key=lambda x: x["item_id"])))
    flagged = [public_item(it) for it in items if it["status"] not in ("admitted", "held")]
    report = {"schema": f"{SCHEMA}/dry_run", "identity": tool_identity(), "mode": "dry-run (nothing written outside --out)",
              "registry": {"version": holdouts.version, "sha256": holdouts.sha256, "path": holdouts.path},
              "liveness": live, "known_state": known_source,
              "known_counts": {"asset_sha": len(known.asset_sha), "text_keys": len(known.text_keys)},
              "sources": sources_report, "policies": {k: {kk: vv for kk, vv in v.items() if kk != "meta"} for k, v in policies.items()},
              "summary": summary, "items_file": {"path": str(out / "dry_run_items.jsonl"), "sha256": rows_sha, "rows": rows_n},
              "flagged_examples": flagged[:200], "wall_s": round(time.time() - started, 1)}
    sha = dcb.write_json(out / "dry_run_report.json", report, seal_field="report_sha256")
    log(f"dry-run report {out / 'dry_run_report.json'} sealed {sha}: {summary['by_status']}")
    return report


def _asset_job(job: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any]:
    path_text, modality_dir, batch_id, policy = job
    path = Path(path_text)
    sha = file_sha256(path)
    try:
        return make_asset(path, sha, modality_dir, batch_id, policy)
    except Exception as exc:
        return {"sha256": sha, "ext": path.suffix.lstrip("."), "bytes": path.stat().st_size, "modality": "unknown",
                "batch_id": batch_id, "path": str(path), "store_path": None, "derived_from": None, "duration_s": None,
                "probe": {}, "flags": [f"probe_failed:{exc!r}"[:200]]}


def parallel_assets(jobs: list[tuple[Path, str, str]], policies: Mapping[str, Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    import concurrent.futures as futures  # noqa: E402

    payload = [(str(p), m, b, dict(policies[b])) for p, m, b in jobs]
    if workers <= 1 or len(payload) < 4:
        return [_asset_job(j) for j in payload]
    out: list[dict[str, Any]] = []
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i, asset in enumerate(pool.map(_asset_job, payload, chunksize=8)):
            out.append(asset)
            if (i + 1) % 1000 == 0:
                log(f"  probed {i + 1:,}/{len(payload):,}")
    return out


def cmd_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    spec = dcb.load_json(args.sources)
    return dry_run(args, spec["sources"] if isinstance(spec, dict) else spec)


def cmd_registry_build(args: argparse.Namespace) -> dict[str, Any]:
    body = build_registry(args)
    holdouts = Holdouts.load(Path(body["path"]))
    live = holdouts_liveness(holdouts)
    log(f"registry v{body['version']} liveness: {json.dumps(live)[:400]}")
    if not live["live"]:
        raise Refusal("registry built but its screen is not live; it must not be used (file kept for diagnosis)")
    if args.register:
        with pg_session() as conn:
            PgStore(conn).register_holdouts(body, body["path"], args.agent)
    return {"path": body["path"], "registry_sha256": body["registry_sha256"], "liveness": live}


def cmd_adopt_v1(args: argparse.Namespace) -> dict[str, Any]:
    v1_dir = Path(args.v1_dir)
    receipt = v1_dir / "verify_receipt.json"
    candidates = [Path(args.v1_manifests)] if args.v1_manifests else [
        v1_dir / "corpus_manifests_1.0.1.tgz", v1_dir.parent / ".scratch" / "corpus_manifests_1.0.1.tgz"]
    tgz = next((c for c in candidates if c.is_file()), candidates[0])
    for p in (tgz, receipt):
        if not p.is_file():
            raise Refusal(f"v1 record {p} missing")
    index = build_v1_index(v1_dir, Path(args.v1_index))
    record = {"version_no": 1, "manifest_path": str(tgz), "manifest_sha256": file_sha256(tgz),
              "verify_receipt": str(receipt), "verify_receipt_sha256": file_sha256(receipt),
              "verify_receipt_seal": dcb.load_json(receipt).get("receipt_sha256"), "dedupe_index": index}
    if args.register:
        with pg_session() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO corpus_version (version_no, parent_version_no, status, manifest_path, manifest_sha256, "
                        "tool_version, tool_sha256, counts, sealed_utc, sealed_by) VALUES (1, NULL, 'sealed', %s, %s, %s, %s, %s, "
                        "now(), %s) ON CONFLICT (version_no) DO NOTHING",
                        (str(tgz), record["manifest_sha256"], "distill_corpus_build 1.0.1 (adopted by record)",
                         _SELF_SHA256, json.dumps({"adopted": record}), args.agent))
            conn.commit()
    log(f"adopt-v1: {json.dumps(record)[:400]}")
    return record


def cmd_init_db(args: argparse.Namespace) -> dict[str, Any]:
    with pg_session() as conn:
        PgStore(conn).init()
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY 1")
            tables = [r[0] for r in cur.fetchall()]
    log(f"init-db: {tables}")
    return {"tables": tables}


def next_version_no(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(max(version_no), 0) FROM corpus_version WHERE status = 'sealed'")
        return int(cur.fetchone()[0]) + 1


def cmd_seal(args: argparse.Namespace) -> dict[str, Any]:
    corpus_root = Path(args.corpus_root)
    version = int(args.version)
    vdir = corpus_root / f"v{version}"
    if vdir.exists():
        raise Refusal(f"{vdir} exists: a version is never rewritten")
    holdouts, live = load_holdouts_live(corpus_root, args.registry)
    with exclusive_lock(Path(args.queues) / ".intake.run.lock", wait_s=600), pg_session() as conn:
        expected = next_version_no(conn)
        if version != expected:
            raise Refusal(f"next version is v{expected}, not v{version}")
        db = PgStore(conn)
        rows = db.items_where("corpus_version_no IS NULL AND status IN ('admitted','held')", ())
        if not rows and not args.allow_empty:
            raise Refusal("nothing admitted or held since the last version")
        assets: dict[str, dict[str, Any]] = {}
        for r in rows:
            assets[r["asset_sha256"]] = db.load_asset(r["asset_sha256"]) or {}
        missing = [s for s, a in assets.items() if not a or not Path(a["store_path"]).is_file()]
        if missing:
            raise Refusal(f"{len(missing)} store files missing, e.g. {missing[:3]}")
        by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            a = assets[r["asset_sha256"]]
            by_modality[r["modality"]].append({**{k: v for k, v in r.items() if k not in ("screen",)},
                                               "store_path": a["store_path"], "asset_bytes": a["bytes"],
                                               "duration_s": a.get("duration_s"), "batch_id": a["batch_id"]})
        vdir.mkdir(parents=True)
        files = {}
        for modality, members in sorted(by_modality.items()):
            sha, n = write_jsonl_atomic(vdir / "rows" / f"{modality}.jsonl", members)
            files[f"rows/{modality}.jsonl"] = {"sha256": sha, "rows": n}
        parent = version - 1
        parent_manifest = corpus_root / f"v{parent}" / "manifest.json"
        with conn.cursor() as cur:
            cur.execute("SELECT manifest_sha256 FROM corpus_version WHERE version_no = %s", (parent,))
            got = cur.fetchone()
        manifest = {"schema": f"{SCHEMA}/corpus_version", "version": version, "parent_version": parent,
                    "parent_manifest_sha256": got[0] if got else None,
                    "parent_manifest_path": str(parent_manifest) if parent_manifest.is_file() else None,
                    "holdout_registry": {"version": holdouts.version, "sha256": holdouts.sha256, "path": holdouts.path},
                    "liveness": live, "identity": tool_identity(), "files": files,
                    "row_item_ids_sha256": canonical_sha256(sorted(r["item_id"] for r in rows)),
                    "store_files": {s: {"path": a["store_path"], "sha256": s, "bytes": a["bytes"]} for s, a in sorted(assets.items())},
                    "counts": {"rows": len(rows), "by_status": dict(Counter(r["status"] for r in rows)),
                               "by_split": dict(Counter(r["split"] for r in rows)),
                               "by_modality": dict(Counter(r["modality"] for r in rows)),
                               "by_licence": dict(Counter(r["licence"] for r in rows)),
                               "by_consent": dict(Counter(r["consent"] for r in rows))},
                    "sealed_utc": utc(), "sealed_by": args.agent,
                    "rule": "rows with split=train feed caches; val/test rows are eval-only and join the next holdout registry"}
        manifest_sha = dcb.write_json(vdir / "manifest.json", manifest, seal_field="manifest_sha256")
        sums = [f"{file_sha256(vdir / name)}  {name}" for name in ["manifest.json", *sorted(files)]]
        (vdir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        archive_dir = Path(args.archive_root)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"v{version}.tgz"
        with tarfile.open(archive.with_name(archive.name + ".tmp"), "w:gz") as tar:
            tar.add(vdir, arcname=f"v{version}")
        os.replace(archive.with_name(archive.name + ".tmp"), archive)
        archive_sha = file_sha256(archive)
        (vdir / "sealed").write_text(json.dumps({"manifest_sha256": manifest_sha, "archive": str(archive), "archive_sha256": archive_sha}) + "\n", encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute("INSERT INTO corpus_version (version_no, parent_version_no, status, manifest_path, manifest_sha256, archive_path, "
                        "archive_sha256, holdout_registry_version, tool_version, tool_sha256, counts, sealed_utc, sealed_by) "
                        "VALUES (%s, %s, 'sealed', %s, %s, %s, %s, %s, %s, %s, %s, now(), %s)",
                        (version, parent if got else None, str(vdir / "manifest.json"), manifest_sha, str(archive), archive_sha,
                         holdouts.version, f"{TOOL} {VERSION}", _SELF_SHA256, json.dumps(manifest["counts"]), args.agent))
            cur.execute("UPDATE item SET corpus_version_no = %s WHERE corpus_version_no IS NULL AND status IN ('admitted','held') "
                        "AND item_id = ANY(%s)", (version, [r["item_id"] for r in rows]))
        conn.commit()
    log(f"seal v{version}: {len(rows)} rows, manifest {manifest_sha}, archive {archive_sha}")
    return {"version": version, "manifest_sha256": manifest_sha, "archive": str(archive), "archive_sha256": archive_sha}


def load_manifest(corpus_root: Path, version: int) -> dict[str, Any]:
    path = corpus_root / f"v{version}" / "manifest.json"
    if not path.is_file():
        raise Refusal(f"{path} missing")
    body = dcb.load_json(path)
    if canonical_sha256({k: v for k, v in body.items() if k != "manifest_sha256"}) != body.get("manifest_sha256"):
        raise Refusal(f"{path}: seal does not match")
    return body


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    corpus_root = Path(args.corpus_root)
    body = load_manifest(corpus_root, int(args.version))
    vdir = corpus_root / f"v{args.version}"
    problems = []
    for name, meta in body["files"].items():
        if file_sha256(vdir / name) != meta["sha256"]:
            problems.append(f"{name} changed")
    for sha, meta in body["store_files"].items():
        if not Path(meta["path"]).is_file() or file_sha256(meta["path"]) != sha:
            problems.append(f"store {meta['path']} missing or changed")
    for line in (vdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if file_sha256(vdir / name) != digest:
            problems.append(f"SHA256SUMS {name}")
    receipt = {"version": int(args.version), "manifest_sha256": body["manifest_sha256"], "files": len(body["files"]),
               "store_files": len(body["store_files"]), "problems": problems, "ok": not problems, "utc": utc()}
    write_json_atomic(corpus_root / "verify" / f"v{args.version}_{utc().replace(':', '')}.json", receipt)
    log(f"verify v{args.version}: ok={receipt['ok']} problems={problems[:5]}")
    if problems:
        raise Refusal("verify failed")
    return receipt


def cmd_resolve(args: argparse.Namespace) -> dict[str, Any]:
    """Rows of v2..vN (v1 is the adopted 1.0.1 corpus: referenced by its manifest sha, its rows live in its own sets)."""

    corpus_root = Path(args.corpus_root)
    target = int(args.version)
    chain = []
    version = target
    while version >= 2:
        body = load_manifest(corpus_root, version)
        chain.append(body)
        version = int(body["parent_version"])
    rows: list[dict[str, Any]] = []
    for body in reversed(chain):
        vdir = corpus_root / f"v{body['version']}"
        for name, meta in sorted(body["files"].items()):
            modality = Path(name).stem
            if args.modality not in (None, "all") and modality != args.modality:
                continue
            for row in dcb.read_jsonl(vdir / name):
                if args.split in (None, "all") or row["split"] == args.split:
                    rows.append({**row, "corpus_version": body["version"]})
    out = Path(args.out) if args.out else corpus_root / "resolved" / f"v{target}_{args.modality or 'all'}_{args.split or 'all'}.jsonl"
    sha, n = write_jsonl_atomic(out, sorted(rows, key=lambda r: (r["corpus_version"], r["item_id"])))
    header = {"version": target, "chain": [b["version"] for b in reversed(chain)],
              "manifest_sha256": {b["version"]: b["manifest_sha256"] for b in chain}, "v1": "adopted 1.0.1 corpus (by record)",
              "holdout_registry": chain[0]["holdout_registry"] if chain else None, "rows": n, "path": str(out), "sha256": sha}
    write_json_atomic(out.with_suffix(".json"), header)
    log(f"resolve v{target} {args.modality}/{args.split}: {n} rows sha {sha}")
    return header


def cmd_rescreen(args: argparse.Namespace) -> dict[str, Any]:
    corpus_root = Path(args.corpus_root)
    body = load_manifest(corpus_root, int(args.version))
    registry_path = corpus_root / "holdouts" / f"HOLDOUT_REGISTRY_v{args.registry_version}.json"
    holdouts = Holdouts.load(registry_path)
    live = holdouts_liveness(holdouts)
    if not live["live"]:
        raise Refusal("registry screen not live")
    hits = []
    vdir = corpus_root / f"v{args.version}"
    for name in sorted(body["files"]):
        for row in dcb.read_jsonl(vdir / name):
            text = "\n\n".join(t for t in (row.get("instruction"), row.get("response"), row.get("transcript")) if t)
            found = holdouts.screen_text(text) + holdouts.screen_media(row["asset_sha256"])
            if found:
                hits.append({"item_id": row["item_id"], "split": row["split"], "hits": found})
    verdict = {"version": int(args.version), "manifest_sha256": body["manifest_sha256"], "registry_version": holdouts.version,
               "registry_sha256": holdouts.sha256, "hits": hits, "train_hits": sum(1 for h in hits if h["split"] == "train"),
               "utc": utc(), "note": "v<N> is never modified; a stage pinning v<N> excludes these item ids"}
    out = corpus_root / "rescreen" / f"v{args.version}_registry_v{holdouts.version}.json"
    write_json_atomic(out, verdict)
    log(f"rescreen v{args.version} vs registry v{holdouts.version}: {len(hits)} hits -> {out}")
    return verdict


def cmd_rebuild_index(args: argparse.Namespace) -> dict[str, Any]:
    """Re-register sealed versions (manifests on disk are the record) into an empty or partial PostgreSQL index:
    corpus_version rows and item rows (status, split, transcript) of every sealed version.  Unsealed intake state
    is not reconstructed: a re-run of `run` re-ingests anything still in the drop zone."""

    corpus_root = Path(args.corpus_root)
    restored = Counter()
    with pg_session() as conn:
        PgStore(conn).init()
        for vdir in sorted(corpus_root.glob("v[0-9]*"), key=lambda p: int(p.name[1:])):
            if not (vdir / "manifest.json").is_file():
                continue
            body = load_manifest(corpus_root, int(vdir.name[1:]))
            with conn.cursor() as cur:
                reg = body["holdout_registry"]
                reg_body = dcb.load_json(reg["path"])
                cur.execute("INSERT INTO holdout_registry (version, path, sha256, created_by, banks) VALUES (%s, %s, %s, %s, %s) "
                            "ON CONFLICT (version) DO NOTHING", (reg["version"], reg["path"], reg["sha256"], "rebuild-index",
                                                                 json.dumps(reg_body.get("banks", []))))
                cur.execute("INSERT INTO corpus_version (version_no, parent_version_no, status, manifest_path, manifest_sha256, "
                            "holdout_registry_version, tool_version, tool_sha256, counts, sealed_utc, sealed_by) VALUES "
                            "(%s, NULL, 'sealed', %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (version_no) DO NOTHING",
                            (body["version"], str(vdir / "manifest.json"), body["manifest_sha256"], reg["version"],
                             body["identity"]["version"], body["identity"]["sha256"], json.dumps(body["counts"]),
                             body["sealed_utc"], body["sealed_by"]))
                for name in sorted(body["files"]):
                    for row in dcb.read_jsonl(vdir / name):
                        cur.execute("INSERT INTO source_batch (batch_id, drop_root, modality_hint, licence, licence_basis, consent) "
                                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (batch_id) DO NOTHING",
                                    (row["batch_id"], "rebuild-index", row["modality"], row["licence"], "rebuild-index",
                                     row["consent"] if row["consent"] in ("owner", "third_party_licensed") else "unspecified"))
                        cur.execute("INSERT INTO asset (sha256, ext, bytes, modality, store_path, batch_id, duration_s) VALUES "
                                    "(%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (sha256) DO NOTHING",
                                    (row["asset_sha256"], Path(row["store_path"]).suffix.lstrip("."), row.get("asset_bytes") or 0,
                                     row["modality"], row["store_path"], row["batch_id"], row.get("duration_s")))
                        cur.execute("INSERT INTO item (item_id, asset_sha256, modality, route, instruction, response, transcript, "
                                    "transcript_status, text_key, group_id, split, status, licence, consent, speaker, voice, "
                                    "session_id, corpus_version_no) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                                    "%s, %s, %s, %s, %s) ON CONFLICT (item_id) DO NOTHING",
                                    (row["item_id"], row["asset_sha256"], row["modality"], row["route"], row.get("instruction"),
                                     row.get("response"), row.get("transcript"), row.get("transcript_status") or "none",
                                     row.get("text_key"), row.get("group_id"), row["split"], row["status"], row["licence"],
                                     row["consent"], row.get("speaker"), row.get("voice"), row.get("session_id"), body["version"]))
                        restored["items"] += cur.rowcount
            conn.commit()
            restored["versions"] += 1
    log(f"rebuild-index: {dict(restored)}")
    return dict(restored)


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with pg_session() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, modality, COALESCE(split, '-'), count(*) FROM item GROUP BY 1, 2, 3 ORDER BY 1, 2, 3")
        items = [list(r) for r in cur.fetchall()]
        cur.execute("SELECT status, count(*) FROM intake_file GROUP BY 1 ORDER BY 1")
        files = dict(cur.fetchall())
        cur.execute("SELECT version_no, status, manifest_sha256 FROM corpus_version ORDER BY 1")
        versions = [list(r) for r in cur.fetchall()]
        cur.execute("SELECT version, sha256 FROM holdout_registry ORDER BY 1")
        registries = [list(r) for r in cur.fetchall()]
    out = {"items": items, "files": files, "versions": versions, "registries": registries}
    print(json.dumps(out, indent=1, default=str))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# transcribe: the GPU step.  Runs under the Pascal speech runtime (plan v2 E1), never from cron.
# ---------------------------------------------------------------------------------------------------------------------


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def transcript_verdict(first: str, second: str) -> tuple[float, str]:
    """conf = 1 - WER(first, second) (plan v2 2.3.3: agreement of two independent passes / judges); review when the
    passes disagree by more than 10 % or conf < 0.85."""

    wer = word_error_rate(first, second)
    conf = max(0.0, 1.0 - wer)
    return round(conf, 4), ("review" if wer > REVIEW_AGREEMENT_WER or conf < REVIEW_CONF else "auto")


def mem_available_gib() -> float:
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


@contextlib.contextmanager
def gpu_claim(queues: Path, gpu: int, job: str, wait_s: float) -> Iterator[None]:
    """Plan 7.4: flock gpu<N>.lock (waits, never competes), a p40.claims line, MemAvailable >= 40 GiB."""

    with exclusive_lock(queues / f"gpu{gpu}.lock", wait_s=wait_s):
        if mem_available_gib() < MEM_GATE_GIB:
            raise Refusal(f"MemAvailable {mem_available_gib():.1f} GiB < {MEM_GATE_GIB}")
        claims = queues / "p40.claims"
        with claims.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc()} GPU{gpu} lane=intake owner=claude job={job} pid={os.getpid()} log=intake-transcribe\n")
        try:
            yield
        finally:
            with claims.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc()} GPU{gpu} lane=intake job={job} released\n")


def load_asr_backend(model_dir: str) -> Callable[[list[str], int], list[str]]:
    """Qwen3-ASR through `qwen_asr` (plan v2 E1 runtime).  UNVERIFIED API: refuses cleanly when it is absent."""

    try:
        import qwen_asr  # type: ignore  # noqa: E402
    except ImportError as exc:
        raise Refusal(f"qwen_asr is not importable in {sys.executable} ({exc}); transcribe runs under the Pascal "
                      "speech runtime /mnt/lexar/omni-campaign/runtimes/speech-pascal (plan v2 E1)") from exc
    model = qwen_asr.load(model_dir, device="cuda")  # pragma: no cover - needs the runtime and a card

    def transcribe(paths: list[str], chunk_s: int) -> list[str]:  # pragma: no cover
        return [str(model.transcribe(p, chunk_seconds=chunk_s)) for p in paths]

    return transcribe


def cmd_transcribe(args: argparse.Namespace) -> dict[str, Any]:
    queues = Path(args.queues)
    backend = load_asr_backend(args.asr_model)
    done = Counter()
    with gpu_claim(queues, int(args.gpu), "intake-transcribe", wait_s=float(args.wait_s)), pg_session() as conn:
        db = PgStore(conn)
        holdouts, _ = load_holdouts_live(Path(args.corpus_root), args.registry)
        pending = db.items_where("status = 'candidate' AND transcript IS NULL AND modality = 'speech' AND corpus_version_no IS NULL", ())
        for r in pending[: int(args.limit)]:
            with conn.cursor() as cur:
                cur.execute("SELECT store_path FROM asset WHERE derived_from = %s AND derivation = 'ffmpeg 16k-mono.flac'",
                            (r["asset_sha256"],))
                got = cur.fetchone()
            source = got[0] if got else (db.load_asset(r["asset_sha256"]) or {}).get("store_path")
            first, second = backend([source], 30)[0], backend([source], 20)[0]
            conf, tstatus = transcript_verdict(first, second)
            hits = holdouts.screen_text(first)
            status = "contaminated" if hits else ("review" if tstatus == "review" else ("admitted" if r["split"] == "train" else "held"))
            with conn.cursor() as cur:
                cur.execute("UPDATE item SET transcript = %s, transcript_conf = %s, transcript_status = %s, transcript_tool = %s, "
                            "status = %s, screen = jsonb_set(screen, '{text}', %s::jsonb), updated_utc = now() "
                            "WHERE item_id = %s AND corpus_version_no IS NULL",
                            (first, conf, tstatus, f"{Path(args.asr_model).name} two-pass agreement", status, json.dumps(hits), r["item_id"]))
            conn.commit()
            done[status] += 1
        export_review(db, Path(args.drop_root))
    log(f"transcribe: {dict(done)}")
    return dict(done)


README_TOP = r"""CHIMERA INTAKE -- drop files here to add them to the Chimera training corpus
==========================================================================

Put files in the folder that matches what they are:

  speech\     recordings of someone talking (your own voice memos go in speech\own-<session>\, see below)
  audio\      other sound (effects, ambience)
  music\      music
  images\     pictures, screenshots, drawings
  video\      video clips (their sound track is used too)
  text\       .txt .md .json .jsonl .csv
  documents\  .pdf
  3d\         .glb .gltf .obj .ply .stl .fbx
  mixed\      anything; the type is detected from the file itself

YOUR FILES ARE NEVER MODIFIED, MOVED, RENAMED OR DELETED BY THE INTAKE.
It only reads them. It keeps its own copies: one under _received\ on this share and one in the training store on
the AI server. You can leave files here, tidy them, or delete them whenever you like.

How it works
  * Every 30 minutes the AI server looks for new files. A file is picked up on the first pass that finds it
    unchanged since the pass before (so within about an hour of the copy finishing); large copies are safe.
    Names ending in .part .crdownload .tmp ~ are ignored until renamed.
  * Each file is fingerprinted, checked against the held-out evaluation sets (anything that matches is kept OUT of
    training), grouped with near-copies, split into train / held-out, and added to the NEXT corpus version when one
    is sealed. Sealed versions never change, so every training run knows exactly which data it saw.
  * If you edit a file after it was picked up, the edited version is added as new data; the old one stays in the
    version that already has it.
  * Make a sub-folder per batch if you like (speech\interviews-2026-10\). Loose files are grouped by day.

Your own recordings
  * Put each recording session in its own folder: speech\own-2026-10-15\, speech\own-kitchen-test\ ...
    A folder named own or starting with own- is treated as YOUR data (consent = owner). It stays on your machines
    (this NAS and the AI server) and is never uploaded anywhere.
  * Sessions, not single clips, decide train vs held-out (about 80 % train / 20 % held), so the model is tested on a
    session it never heard.
  * A transcript is optional: a .txt file with the same name next to the audio (memo1.wav + memo1.txt) is used as
    is. Without one, speech recognition writes a draft and you can check it in _review\.
  * Long recordings are cut into 3-20 second pieces at pauses (the original is kept whole in _received\).

Optional _meta.json in a batch folder, for example:
  {"source": "phone voice memos", "licence": "owner", "consent": "owner", "speaker": "cyril",
   "language": "en", "session": "2026-10-15-evening", "notes": "reading the garden list"}
consent is one of owner, third_party_licensed, unspecified. Without _meta.json, files outside own folders are
recorded as licence "unspecified" under Cyril's 2026-10-14 ruling ("[Cyril quote withheld]").

Service folders (written by the intake, not by you)
  _received\  archive copies of everything picked up, named by content hash. Do not edit.
  _review\    transcripts.jsonl files to check: set "accepted" to true or false and correct "transcript".
  _rejected\  reserved.

Questions: ask Claude or Codex ("what happened to the files I dropped in speech\own-...?").
Tool: intake.py {version} on the AI server, $OC/distill/tools/intake.py
"""

README_FOLDER = {
    "speech": "Speech: .wav .flac .mp3 .m4a .ogg .opus .aac .amr .3gp. Your own recordings: one folder per session named "
              "own-<something>. Transcripts: memo.wav + memo.txt (same name). LibriSpeech *.trans.txt and LJSpeech "
              "metadata.csv layouts are understood. Text files here are read as transcripts, never as training text.",
    "audio": "Other sound: .wav .flac .mp3 .m4a .ogg .opus. Text files here are metadata only.",
    "music": "Music: .wav .flac .mp3 .m4a .ogg .opus. Artist/copyright tags are flagged for review.",
    "images": "Pictures: .png .jpg .jpeg .webp .bmp .gif .tif. Near-copies of held-out images are kept out of training.",
    "video": "Video: .mp4 .mov .mkv .webm .avi .m4v. The sound track is extracted too.",
    "text": "Text: .txt .md (cut into ~1,500-word pieces), .jsonl/.json (one row per record: instruction/response or "
            "messages), .csv (50 rows per piece).",
    "documents": "Documents: .pdf (text is extracted with pdftotext).",
    "3d": "3D: .glb .gltf .obj .ply .stl .fbx (stored and registered; no 3D training route reads them yet).",
    "mixed": "Anything: the type is detected from the file itself.",
    "_received": "Archive copies written by the intake (named <sha256>.<ext>, by batch). Do not edit or add files here.",
    "_review": "Files for you to check. In each transcripts.jsonl line set \"accepted\": true or false and fix "
               "\"transcript\" if needed. The next intake run reads your answers.",
    "_rejected": "Reserved for files the intake could not use.",
}


def cmd_dropzone_init(args: argparse.Namespace) -> dict[str, Any]:
    """Create the drop-zone folders and their _README.txt files (idempotent; touches nothing else)."""

    root = Path(args.drop_root)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    files = {root / "_README.txt": README_TOP.replace("{version}", VERSION)}
    for name, text in README_FOLDER.items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        files[folder / "_README.txt"] = (f"{name}\\  --  {text}\n\nYour files are never modified, moved, renamed or "
                                          f"deleted by the intake. See ..\\_README.txt for how everything works.\n")
    for path, text in files.items():
        payload = text.replace("\n", "\r\n").encode("utf-8")  # Windows Notepad friendly
        if not path.is_file() or path.read_bytes() != payload:
            path.write_bytes(payload)
            written.append(str(path))
    log(f"dropzone-init: {root}: {len(written)} README files written")
    return {"root": str(root), "written": written}


def cmd_schema(args: argparse.Namespace) -> dict[str, Any]:
    print(DDL)
    return {}


COMMANDS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "schema": cmd_schema, "init-db": cmd_init_db, "dropzone-init": cmd_dropzone_init, "registry-build": cmd_registry_build, "adopt-v1": cmd_adopt_v1,
    "run": cmd_run, "dry-run": cmd_dry_run, "transcribe": cmd_transcribe, "import-review": cmd_import_review,
    "seal": cmd_seal, "verify": cmd_verify, "resolve": cmd_resolve, "rescreen": cmd_rescreen,
    "rebuild-index": cmd_rebuild_index, "status": cmd_status,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--drop-root", default=DEFAULT_DROP_ROOT)
    parser.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--queues", default=DEFAULT_QUEUES)
    parser.add_argument("--v1-dir", default=DEFAULT_V1_DIR)
    parser.add_argument("--v1-manifests", default=None, help="corpus_manifests_1.0.1.tgz (default: <v1-dir>/ or <v1-dir>/../.scratch/)")
    parser.add_argument("--v1-index", default=None, help="v1 dedupe index (default <corpus-root>/v1/adopt_v1_index.json)")
    parser.add_argument("--oc", default=str(CAMPAIGN_ROOT))
    parser.add_argument("--adopted", default="/mnt/lexar/omni-campaign/codex-adopted-2026-10-14")
    parser.add_argument("--registry", default=None, help="a HOLDOUT_REGISTRY_v<K>.json (default: the newest)")
    parser.add_argument("--registry-version", type=int, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--register", action="store_true", help="also record in PostgreSQL")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--cpu-only", action="store_true", help="cron form: never touches a GPU (it never does; GPU work is `transcribe`)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources", default=None, help="dry-run source spec JSON")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--modality", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--wait-s", type=float, default=3600.0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--now", type=float, default=None, help=argparse.SUPPRESS)  # tests: the settle clock
    parser.add_argument("--asr-model", default="/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/speech-teachers-v1/Qwen3-ASR-0.6B")
    args = parser.parse_args(argv)
    if args.v1_index is None:
        candidate = Path(args.corpus_root) / "v1" / "adopt_v1_index.json"
        args.v1_index = str(candidate) if args.command == "adopt-v1" or candidate.is_file() else None
    if args.command == "dry-run" and not (args.sources and args.out):
        parser.error("dry-run needs --sources and --out")
    if args.command == "run" and args.dry_run and not args.out:
        parser.error("run --dry-run needs --out")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command not in ("transcribe",):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU tool; only transcribe opens a card, under its claim
    try:
        COMMANDS[args.command](args)
    except Refusal as refusal:
        print(refusal.message, file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
