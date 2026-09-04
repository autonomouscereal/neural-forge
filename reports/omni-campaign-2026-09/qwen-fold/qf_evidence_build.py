"""Build the sealed evidence manifest every solve in this lane is fitted against.

``qf_corpus.py`` defines the schema and re-seals it on read; nothing wrote one.
This is that writer.  It draws real Qwen tokens out of the curated master shard
registry, screens every one of them against the frozen review surface, packs
them into ``--batches`` sequences of exactly ``--sequence-length`` tokens, and
hands the rows to :func:`qf_corpus.write_manifest` so the seal in the file is
computed by the same code that will refuse it later.

Three decisions are load-bearing and are stated here rather than buried:

* **ids inline, not ``ids_path``.**  PORT_SPEC 2.1 allows either, but the
  shipped ``qf_corpus.read_manifest`` indexes ``row["ids"]`` directly -- a
  manifest written with ``ids_path`` would be refused by its own reader.  24 x
  4,096 ids is ~700 KB of JSON; that is cheap enough to keep the evidence in one
  self-sealing file.
* **Batches MIX routes; the split is by batch index; rows are never cut.**  If
  each batch were route-pure, the fit/held split (and the every-third-from-the-
  third internal check) would also be a route split, and a held-out number would
  mean "this fold generalises to a route it never saw" rather than "to text it
  never saw".  Rows are interleaved across text/code/math/tool/reasoning by token
  deficit (see :func:`pack_batches`) so every batch carries all five and the five
  route totals end close to each other; the receipt prints the per-batch
  composition.  Rows are placed WHOLE: a row that does not fit in what is left of
  a batch waits for the next one and the remainder is padded with the parent's
  eos id.  The first cut of this manifest packed a flat stream and sliced it at
  fixed offsets, which put the head of a document in a fit batch and its tail in
  the held batch after it -- 1,827 of 24,576 held tokens were continuations of
  text the solve had just fitted on.  A held number computed that way is
  optimistic by construction, so the packer now costs a little padding
  (``packing.padding_fraction``, refused above ``--max-padding-fraction``) to buy
  a split that is a DOCUMENT split, and the receipt measures it
  (``packing.split_leakage``) rather than claiming it.
* **The contamination screen refuses BATCHES, not just rows.**  Rows are
  screened as they are collected, and then every packed batch is decoded back
  through the same tokenizer and screened again -- because packing concatenates
  rows, and a containment hit that only exists across a row boundary is still a
  review prompt sitting in the fit stream.  The needle is any 48-character
  window of any text in the review surface (bank v1, the bank v2 draft, the
  extension bank v2, the paraphrase triplets, the probe and smoke packs), which
  is the strict reading of "a >= 48-char substring of the bank": quoting a third
  of a review prompt is contamination, and only screening whole prompts would
  miss it.  Live probes (a real review prompt spliced into filler) prove the
  screen fires, and clean controls prove it does not fire on everything.

Bounded reads only.  The share is CIFS at 5-25 MB/s and is never walked: the
pointer is read once, entries are filtered on the ``modality_counts`` the
pointer already carries, and only the shards actually sampled are opened.

    /home/cereal/model-surgery-lab/venv/bin/python qf_evidence_build.py \\
        --out  /home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json

Every count the receipt prints is a measurement.  The rung numbers
(``--batches``, ``--sequence-length``, ``--held-every``) are the plan's, and are
proposed decision lines for Cyril, not thresholds this file invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qf_corpus
from qf_arch import canonical_json_bytes

LOG_PREFIX = "[qfev]"
RECEIPT_SCHEMA = "neural-forge-omni-evidence-build-receipt-v1"

#: The routes this manifest is drawn over.  Media routes are deliberately absent:
#: the fold sites this evidence feeds are all text-side, and a media row's text
#: is a caption of bytes the harness never sees.
DEFAULT_ROUTES = ("text", "code", "math", "tool", "reasoning")

#: Server defaults, so the tool runs correctly with no arguments in the lane.
DEFAULT_POINTER = "/mnt/ai_data/omnicoder_2026/datasets/MASTER_SHARD_INDEX_POINTER.json"
DEFAULT_CHIMERA_SRC = "/home/cereal/omni-campaign/chimera-omni-import/src/src"
DEFAULT_PARENT = "/home/cereal/models/Qwen3.5-4B"
DEFAULT_SIBLING_PARENTS = (
    "/home/cereal/models/Qwen3.5-0.8B",
    "/home/cereal/models/Qwen3.8-27B-Uncensored",
)
DEFAULT_OUT = "/home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json"

#: The tokenizer this campaign's Qwen3.5 parents share [verified 2026-09-02:
#: 0.8B and 4B tokenizer.json are byte-identical].  Pinned because a swapped
#: tokenizer changes every id in the manifest silently -- the seal would still
#: verify and every solve would be fitted against a stream the model cannot read.
EXPECTED_TOKENIZER_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
#: The PHYSICAL embedding rows of the Qwen3.5 parents.  The tokenizer defines
#: fewer, and the difference is MEASURED here rather than quoted: 248,070 defined
#: (248,044 base BPE rows + 26 added tokens) inside 248,320 physical = 250 dead
#: rows on this box, not the 276 PORT_SPEC 0.3 and the R5 digest quote.  276 is
#: 250 + those 26 added tokens, which is what you get by deriving the dead set as
#: ``range(config.vocab_size) - base_bpe_ids``; a Stage2DonorVocabulary port that
#: does so deletes every added token in C1 -- eos/bos 248,044, vision_start,
#: vision_end, image and video among them.  See ``vocabulary_arithmetic`` in the
#: receipt, which carries the measurement and this hazard together.
EXPECTED_VOCAB_SIZE = 248_320
#: What PORT_SPEC 0.3 / the R5 digest quote, kept only so the receipt can say
#: where the two numbers disagree and why.
PORT_SPEC_QUOTED_DEAD_ROWS = 276

#: The review surface the evidence must not contain.  Digests are pinned so a
#: re-cut bank cannot silently narrow the screen; `--allow-unsealed-screen-source`
#: records the exception instead of hiding it.
SCREEN_SOURCE_SHA256: dict[str, str] = {
    "lfm2_semantic_whole_response_prompts_v1.json": (
        "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
    ),
    "omni_review_bank_v2_draft.json": (
        "f26f36bf31cb2a062c3e75e9f1567b379ea886e6022e2afb93b572a269f4a20b"
    ),
    "matrix_campaign_probe_prompts_v1.json": (
        "6230f878be9287a9abf4ad129d586fc22346867897858b9f1a4f0b75d1f82f2e"
    ),
    "matrix_campaign_smoke_prompts_v1.json": (
        "82ebf70a1606ec8a57c4daa31218fe1d376a91f211246c78716615ca66d2ceca"
    ),
    "lfm2_extension_bank_prompts_v2.json": (
        "340694a64f7e886ecf1576dd9245a1445d8e763036d842f3730c4c8ab5c9c592"
    ),
    "lfm2_paraphrase_triplets_v1.json": (
        "f9ac00e6dd5212f04a9f06d2a177c233562b5276617b14cbab0aa353ab846095"
    ),
}

#: A needle shorter than this is not a quotation, it is a common phrase; the
#: constant is the LFM2 screen's (stage2_generate_teacher_corpus.py:85) so a
#: "clean" statement here means what it meant there.
MINIMUM_NEEDLE_CHARACTERS = 48

#: Registry row markers that take a row out regardless of the loader's own
#: ``training_allowed`` / ``use_policy`` rules.
NEGATIVE_ROW_MARKERS = ("contaminated", "quarantine", "do_not_train", "excluded", "fail")

_HEX_ONLY = re.compile(r"\A[0-9a-fA-F]{32,}\Z")


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


# ---------------------------------------------------------------- digests / io


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalized_whitespace(value: str) -> str:
    """Collapse whitespace so a rewrapped copy of a review prompt still hits."""

    return " ".join(value.split())


# ------------------------------------------------------- the review-pack screen


def harvest_texts(node: Any, out: list[str]) -> None:
    """Every string in a review pack, plus the transcript of every message list.

    Deliberately over-broad: a bank's rubric, its needle text and its lane
    description are all things a training row must not quote, and the cost of
    screening them too is a handful of extra needles.  Pure hex digests are the
    one exclusion -- they are provenance, not review content, and a 48-character
    window of a sha256 is a needle that means nothing.
    """

    if isinstance(node, str):
        if not _HEX_ONLY.match(node.strip()):
            out.append(node)
        return
    if isinstance(node, list):
        for item in node:
            harvest_texts(item, out)
        return
    if isinstance(node, Mapping):
        messages = node.get("messages")
        if isinstance(messages, list) and all(isinstance(m, Mapping) for m in messages):
            contents = [str(m.get("content", "")) for m in messages]
            out.append("\n\n".join(contents))
            out.append("\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages))
        for value in node.values():
            harvest_texts(value, out)


@dataclass(frozen=True)
class ScreenSource:
    name: str
    path: str
    sha256: str
    pinned: bool
    sha256_matches_pin: bool
    text_count: int
    needle_windows_added: int
    needle_windows_own: int


@dataclass(frozen=True)
class ReviewScreen:
    """Digest and >=48-char containment screen over the whole review surface."""

    digests: frozenset[str]
    windows: frozenset[str]
    window_source: Mapping[str, str]
    needle_characters: int
    sources: tuple[ScreenSource, ...]
    #: Whole texts read back out of each pack, used by :func:`screen_liveness` to
    #: probe with REAL review prompts instead of with the screen's own windows.
    source_texts: Mapping[str, tuple[str, ...]]

    def drop_reasons(self, text: str) -> tuple[str, ...]:
        """Every independent reason this text stays out of the evidence.

        Containment is the load-bearing half.  A digest can only fire when the
        text IS a review prompt; a row that quotes one is caught by the window
        pass, and quoting is the shape contamination actually takes.
        """

        reasons: list[str] = []
        stripped = text.strip()
        if text_sha256(text) in self.digests or text_sha256(stripped) in self.digests:
            reasons.append("review_pack_digest")
        window = self.first_window_hit(text)
        if window is not None:
            reasons.append(f"review_pack_containment:{self.window_source[window]}")
        return tuple(reasons)

    def first_window_hit(self, text: str) -> str | None:
        """The first >=48-char window of the review surface this text contains."""

        width = self.needle_characters
        for candidate in (text, normalized_whitespace(text)):
            if len(candidate) < width:
                continue
            for start in range(len(candidate) - width + 1):
                window = candidate[start : start + width]
                if window in self.windows:
                    return window
        return None

    def provenance(self) -> dict[str, Any]:
        return {
            "needle_characters": self.needle_characters,
            "digest_count": len(self.digests),
            "window_count": len(self.windows),
            "rule": (
                "a row or a packed batch is refused if its text (raw or "
                "whitespace-normalized) contains ANY window of "
                f"{self.needle_characters} characters taken from ANY string in "
                "ANY listed review pack, or if its digest is one of theirs"
            ),
            "sources": [
                {
                    "name": source.name,
                    "path": source.path,
                    "sha256": source.sha256,
                    "pinned": source.pinned,
                    "sha256_matches_pin": source.sha256_matches_pin,
                    "texts_harvested": source.text_count,
                    # ADDED is order-dependent (a window already contributed by
                    # an earlier pack is not counted twice), OWN is the pack's
                    # own unique-window count and is a property of the file.
                    "needle_windows_added": source.needle_windows_added,
                    "needle_windows_own": source.needle_windows_own,
                }
                for source in self.sources
            ],
        }


def build_review_screen(
    paths: Iterable[Path],
    *,
    needle_characters: int,
    allow_unsealed: bool,
    probe_texts_per_source: int = 8,
) -> ReviewScreen:
    digests: set[str] = set()
    windows: dict[str, str] = {}
    sources: list[ScreenSource] = []
    source_texts: dict[str, tuple[str, ...]] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(
                f"review pack {path} is missing; the screen would not cover the "
                f"surface the campaign grades on"
            )
        sha = file_sha256(path)
        pin = SCREEN_SOURCE_SHA256.get(path.name)
        matches = pin is None or sha == pin
        if pin is not None and not matches and not allow_unsealed:
            raise SystemExit(
                f"{path.name} is {sha}, the campaign pinned {pin}.  A re-cut bank "
                f"narrows the screen silently; re-pin the digest in "
                f"SCREEN_SOURCE_SHA256 or pass --allow-unsealed-screen-source to "
                f"record the exception in the receipt."
            )
        harvested: list[str] = []
        harvest_texts(json.loads(path.read_text(encoding="utf-8")), harvested)
        if not harvested:
            raise SystemExit(
                f"review pack {path} yielded no strings; a pack that contributes "
                f"nothing to the screen is a screen built over the wrong file"
            )
        own: set[str] = set()
        added = 0
        for value in harvested:
            digests.add(text_sha256(value))
            digests.add(text_sha256(value.strip()))
            for candidate in (value, normalized_whitespace(value)):
                for start in range(max(0, len(candidate) - needle_characters + 1)):
                    window = candidate[start : start + needle_characters]
                    own.add(window)
                    if window not in windows:
                        windows[window] = path.name
                        added += 1
        # Bounded sample of this pack's own texts, longest first, kept so the
        # liveness probes can splice REAL review prompts rather than windows the
        # screen was built from (which it catches by construction).
        probe_texts = tuple(
            sorted(
                {value for value in harvested if len(value.strip()) >= needle_characters},
                key=lambda value: (-len(value), value),
            )[:probe_texts_per_source]
        )
        if not probe_texts:
            raise SystemExit(
                f"review pack {path} carries no string of at least "
                f"{needle_characters} characters; the containment screen cannot be "
                f"probed against it and its coverage would be unmeasured"
            )
        source_texts[path.name] = probe_texts
        sources.append(
            ScreenSource(
                name=path.name,
                path=str(path),
                sha256=sha,
                pinned=pin is not None,
                sha256_matches_pin=matches,
                text_count=len(harvested),
                needle_windows_added=added,
                needle_windows_own=len(own),
            )
        )
        log(
            f"screen source {path.name} sha {sha[:16]} texts {len(harvested)} "
            f"windows added {added} own {len(own)}"
        )
    return ReviewScreen(
        digests=frozenset(digests),
        windows=frozenset(windows),
        window_source=dict(windows),
        needle_characters=needle_characters,
        sources=tuple(sources),
        source_texts=source_texts,
    )


def screen_liveness(screen: ReviewScreen, *, probes: int) -> dict[str, Any]:
    """Prove the screen fires on REAL review prompts and not on clean filler.

    The first version of this check spliced a string taken straight out of
    ``screen.windows`` into filler and counted the catches.  That is
    tautological: any non-empty screen contains its own windows, so it returned
    ``probes/probes`` for ANY pack set and could not tell a screen built over the
    wrong text from a screen built over the right one.

    These probes are whole texts read back out of each pack (``source_texts``),
    and each one is required to be caught AND attributed to the pack it came
    from, so a pack whose windows were never actually merged in fails here.
    Three arms:

    * **verbatim** -- the prompt spliced into filler; the raw containment arm;
    * **re-wrapped** -- the same prompt with every whitespace run replaced by a
      newline, which only the whitespace-normalised arm can catch, so a screen
      that lost that arm fails instead of passing on the verbatim probes;
    * **short-quote control** -- the same prompt cut to ``needle_characters - 1``
      characters, which CANNOT contain a needle and must NOT be caught, so a
      screen that fires on everything fails too.

    Clean filler controls stay.  No probe text is written anywhere -- only counts
    and the source each catch was attributed to.
    """

    if not screen.windows:
        raise SystemExit("the review screen carries no needles")
    width = screen.needle_characters
    by_source: list[dict[str, Any]] = []
    caught_total = 0
    probe_total = 0
    rewrapped_caught = 0
    rewrapped_total = 0
    short_caught = 0
    short_total = 0
    misattributed: list[str] = []
    uncovered: list[str] = []
    for source in screen.sources:
        texts = screen.source_texts.get(source.name, ())
        caught = 0
        attributed = 0
        rewrapped = 0
        short_hits = 0
        for index, text in enumerate(texts):
            probe = (
                "The following passage continues a longer maintenance document.\n"
                f"{text}\n"
                "The document then moves on to an unrelated topic entirely."
            )
            reasons = screen.drop_reasons(probe)
            probe_total += 1
            if reasons:
                caught += 1
                caught_total += 1
                window = screen.first_window_hit(probe)
                if window is not None and screen.window_source[window] == source.name:
                    attributed += 1
            rewrapped_probe = (
                "Maintenance log, wrapped by the exporter:\n"
                + "\n".join(text.split())
                + "\nEnd of entry."
            )
            rewrapped_total += 1
            if screen.drop_reasons(rewrapped_probe):
                rewrapped += 1
                rewrapped_caught += 1
            short = text.strip()[: width - 1]
            short_probe = (
                f"Entry {index}: the technician noted [{short}] and closed the ticket."
            )
            short_total += 1
            if screen.drop_reasons(short_probe):
                short_hits += 1
                short_caught += 1
        if texts and (caught < len(texts) or attributed < len(texts)):
            # ``attributed`` can fall below ``caught`` only when two packs share a
            # window; that is a real overlap and is reported, not hidden.
            uncovered.append(source.name)
        if attributed < caught:
            misattributed.append(source.name)
        by_source.append(
            {
                "source": source.name,
                "probe_texts": len(texts),
                "caught": caught,
                "attributed_to_this_source": attributed,
                "rewrapped_caught": rewrapped,
                "short_quote_controls_falsely_caught": short_hits,
            }
        )

    clean_caught = 0
    for index in range(probes):
        clean = (
            f"Batch {index}: the pump was serviced on the {index % 28 + 1}th and the "
            f"reading settled at {index * 3 % 97} units after the valve was reseated."
        )
        if screen.drop_reasons(clean):
            clean_caught += 1
    return {
        "rule": (
            "every pack must catch its own whole prompts spliced into filler, "
            "verbatim and re-wrapped, and must not catch a quote shorter than the "
            "needle or clean filler"
        ),
        "sources": len(screen.sources),
        "probes": probe_total,
        "spliced_probes_caught": caught_total,
        "spliced_probes_missed": probe_total - caught_total,
        "rewrapped_probes": rewrapped_total,
        "rewrapped_probes_caught": rewrapped_caught,
        "short_quote_controls": short_total,
        "short_quote_controls_falsely_caught": short_caught,
        "clean_controls": probes,
        "clean_controls_falsely_caught": clean_caught,
        "sources_without_full_coverage": uncovered,
        "sources_with_shared_windows": misattributed,
        "by_source": by_source,
        "screen_is_live": (
            probe_total > 0
            and caught_total == probe_total
            and rewrapped_caught == rewrapped_total
            and short_caught == 0
            and clean_caught == 0
            and not uncovered
        ),
    }


# ------------------------------------------------------------------- tokenizer


@dataclass(frozen=True)
class Tokenizer:
    encoder: Any
    path: str
    sha256: str
    defined_rows: int
    base_rows: int
    added_tokens: tuple[tuple[int, str], ...]
    config_vocab_size: int
    separator_id: int

    def vocabulary_arithmetic(self) -> dict[str, Any]:
        """The dead-row count, MEASURED, next to the number the specs quote.

        PORT_SPEC 0.3 and the R5 digest both say 276 dead rows.  On this box the
        tokenizer defines 248,070 rows (248,044 base BPE + 26 added) inside
        248,320 physical, which is 250 dead.  276 = 250 + 26, i.e. exactly what
        ``range(config.vocab_size) - base_bpe_ids`` returns when the added tokens
        are not in ``ids`` -- so a port that follows the spec's derivation
        classifies every added token as dead and C1 deletes them, eos/bos and the
        vision/image/video control tokens included.  Both numbers are printed so
        nobody has to guess which one their code reproduced.
        """

        dead = self.config_vocab_size - self.defined_rows
        return {
            "physical_rows": self.config_vocab_size,
            "defined_rows": self.defined_rows,
            "base_bpe_rows": self.base_rows,
            "added_token_rows": len(self.added_tokens),
            "added_tokens": [
                {"id": token_id, "content": content}
                for token_id, content in self.added_tokens
            ],
            "dead_rows_measured": dead,
            "dead_rows_quoted_by_port_spec": PORT_SPEC_QUOTED_DEAD_ROWS,
            "agrees_with_port_spec": dead == PORT_SPEC_QUOTED_DEAD_ROWS,
            "hazard": (
                "the quoted 276 is this measurement's 250 dead rows PLUS the "
                f"{len(self.added_tokens)} added tokens; deriving the dead set as "
                "range(config.vocab_size) - base_bpe_ids marks every added token "
                "dead and a Stage2DonorVocabulary/C1 port following PORT_SPEC 0.3 "
                "would delete them -- eos/bos and the vision/image/video control "
                "tokens among them.  Derive dead rows from the DEFINED table "
                "(base + added), never from the base alphabet alone."
            ),
        }

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.encoder.decode(ids, skip_special_tokens=False)


def load_tokenizer(parent_dir: Path, *, allow_unpinned: bool) -> Tokenizer:
    """The parent's own tokenizer.json, hashed, with the vocabulary asserted.

    ``tokenizers`` rather than ``AutoTokenizer`` on purpose: the file whose
    sha256 the manifest records is then exactly the file the ids came out of,
    with no tokenizer_config.json in between.
    """

    from tokenizers import Tokenizer as HFTokenizer

    tokenizer_path = parent_dir / "tokenizer.json"
    config_path = parent_dir / "config.json"
    sha = file_sha256(tokenizer_path)
    if sha != EXPECTED_TOKENIZER_SHA256 and not allow_unpinned:
        raise SystemExit(
            f"{tokenizer_path} is {sha}, the campaign pinned "
            f"{EXPECTED_TOKENIZER_SHA256}.  Every id in the manifest is defined by "
            f"this file; pass --allow-unpinned-tokenizer to build against a "
            f"different one and have the exception recorded."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    vocab_size = int(text_config.get("vocab_size") or config.get("vocab_size") or 0)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise SystemExit(
            f"{config_path} declares vocab_size {vocab_size}, expected "
            f"{EXPECTED_VOCAB_SIZE} (the physical embedding rows every Qwen3.5 "
            f"parent in this campaign carries)"
        )
    encoder = HFTokenizer.from_file(str(tokenizer_path))
    defined_rows = int(encoder.get_vocab_size(with_added_tokens=True))
    base_rows = int(encoder.get_vocab_size(with_added_tokens=False))
    # Read out of the FILE whose sha256 the manifest records, not out of a
    # tokenizer API: the added tokens are the rows the dead-row arithmetic
    # disagrees about, and this is the copy that arithmetic has to be about.
    body = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    added_tokens = tuple(
        sorted(
            (int(row["id"]), str(row["content"]))
            for row in (body.get("added_tokens") or [])
        )
    )
    separator = text_config.get("eos_token_id", config.get("eos_token_id"))
    if isinstance(separator, list) and separator:
        separator = separator[0]
    if separator is None:
        # ``or`` would treat id 0 as absent; the fallbacks are checked for None.
        for fallback in ("<|im_end|>", "<|endoftext|>"):
            candidate = encoder.token_to_id(fallback)
            if candidate is not None:
                separator = candidate
                break
    if separator is None:
        raise SystemExit(f"{config_path} names no eos token; rows cannot be delimited")
    return Tokenizer(
        encoder=encoder,
        path=str(tokenizer_path),
        sha256=sha,
        defined_rows=defined_rows,
        base_rows=base_rows,
        added_tokens=added_tokens,
        config_vocab_size=vocab_size,
        separator_id=int(separator),
    )


def sibling_tokenizer_report(paths: Iterable[str], sha: str) -> list[dict[str, Any]]:
    """Which other parents on this box share the tokenizer this manifest used.

    Recorded, never asserted.  The campaign's working assumption is that one
    tokenizer serves the whole Qwen3.5 ladder; whether a given parent on disk
    actually shares it is a measurement, and a manifest built on one tokenizer
    is not evidence for a parent that uses another.
    """

    rows: list[dict[str, Any]] = []
    for directory in paths:
        candidate = Path(directory) / "tokenizer.json"
        if not candidate.is_file():
            rows.append({"parent": str(directory), "tokenizer_json": None, "shares_tokenizer": None})
            continue
        other = file_sha256(candidate)
        rows.append(
            {
                "parent": str(directory),
                "tokenizer_json": str(candidate),
                "sha256": other,
                "shares_tokenizer": other == sha,
            }
        )
    return rows


# ------------------------------------------------------------ registry sampling


@dataclass
class Sampled:
    route: str
    ids: list[int]
    record_id: str
    dataset_id: str
    shard_path: str
    row_sha256: str
    characters: int


#: The curation fields whose values are read for a refusal (and counted in the
#: receipt whether they refuse or not).
MARKER_FIELDS = ("contamination_status", "promotion_status", "training_bucket", "split")

_NEGATION_PREFIXES = ("not", "non", "un")
_NEGATION_WORDS = frozenset({"not", "non", "no", "un", "never"})
#: Values that ARE a clean verdict, as opposed to values that merely contain the
#: word (``quarantine_clean_pending`` is not a clean verdict).
_CLEAN_VALUES = frozenset({"clean", "cleaned", "clean_verified", "verified_clean"})


def row_marker_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Registry-side refusals beyond ``normalize_row``'s own two.

    ``normalize_row`` already drops ``training_allowed is False`` and any
    ``use_policy`` containing ``do_not_train``.  These rows also carry the
    curation passes' own verdicts, and a row the curators marked is not a row
    this manifest takes.  The observed values on this registry are
    ``contamination_status='clean'``, ``training_bucket='train'``,
    ``split='train'`` and a pending-index promotion label [verified 2026-09-02
    over 600 rows of five entries], so this fires on nothing today -- it is here
    so that a future append which DOES carry a marker is refused rather than
    quietly packed.  ``clean`` and a negation prefix both veto a marker, because
    ``not_contaminated`` is the opposite of the thing being looked for.
    """

    reasons: list[str] = []
    for field in MARKER_FIELDS:
        value = row.get(field)
        if not isinstance(value, str):
            continue
        lowered = value.lower().strip()
        # The veto is for the value that IS "clean", not for any value with the
        # word in it: ``quarantine_clean_pending`` must still be refused.
        if lowered in _CLEAN_VALUES:
            continue
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        for position, token in enumerate(tokens):
            for marker in NEGATIVE_ROW_MARKERS:
                if not token.startswith(marker):
                    continue
                previous = tokens[position - 1] if position else ""
                if previous in _NEGATION_WORDS or token.startswith(_NEGATION_PREFIXES):
                    continue
                reasons.append(f"{field}:{marker}")
                break
    return tuple(reasons)


def render_row_text(record: Any) -> str:
    """The text a row contributes: its instruction and its response, in order.

    ``normalize_row`` has already compacted both and pulled them out of whatever
    schema era the row belongs to.  Nothing else is added -- no chat template,
    because the capture stream is raw ids and the fold sites are solved on the
    parent's own activations, not on a rendered conversation.
    """

    parts = [part for part in (record.instruction, record.response) if part]
    return "\n\n".join(parts)


def collect_rows(
    curated: Any,
    *,
    pointer: Path,
    routes: tuple[str, ...],
    tokens_per_route: int,
    tokenizer: Tokenizer,
    screen: ReviewScreen,
    max_entries: int,
    max_shards_per_entry: int,
    max_rows_per_shard: int,
    per_dataset_route_cap: int,
    deadline_seconds: float,
) -> tuple[dict[str, list[Sampled]], dict[str, Any]]:
    """Walk the registry, bounded, until every route has its token budget.

    Bounded means: the pointer once; per entry, only the entries whose own
    ``modality_counts`` (already in the pointer) touch a wanted route; per entry,
    only the shards whose own counts touch a route still short of budget; per
    shard, at most ``max_rows_per_shard`` lines, streamed.  Nothing is walked.
    """

    started = time.time()
    pools: dict[str, list[Sampled]] = {route: [] for route in routes}
    tokens: Counter[str] = Counter()
    per_dataset: Counter[tuple[str, str]] = Counter()
    drops: Counter[str] = Counter()
    scanned = Counter()
    marker_values: dict[str, Counter[str]] = {field: Counter() for field in MARKER_FIELDS}
    shards_used: dict[str, dict[str, Any]] = {}
    entries_visited: list[dict[str, Any]] = []

    def satisfied() -> bool:
        return all(tokens[route] >= tokens_per_route for route in routes)

    for entry in curated.iter_master_entries(pointer, max_entries=None):
        if satisfied() or scanned["entries"] >= max_entries:
            break
        if time.time() - started > deadline_seconds:
            drops["deadline_reached"] += 1
            break
        entry_routes = {
            curated.canonical_route(name) for name in (entry.get("modality_counts") or {})
        }
        wanted_here = {route for route in routes if tokens[route] < tokens_per_route}
        if not entry_routes & wanted_here:
            continue
        dataset_id = str(entry.get("dataset_id") or entry.get("label") or "unknown_dataset")
        status = str(entry.get("training_status") or "")
        scanned["entries"] += 1
        entry_rows = 0
        shard_index = 0
        for shard in curated.iter_shard_specs(entry):
            if satisfied() or shard_index >= max_shards_per_entry:
                break
            if time.time() - started > deadline_seconds:
                drops["deadline_reached"] += 1
                break
            wanted_here = {route for route in routes if tokens[route] < tokens_per_route}
            shard_routes = {
                curated.canonical_route(name) for name in (shard.get("modality_counts") or {})
            }
            if not shard_routes & wanted_here:
                continue
            shard_index += 1
            shard_path = str(shard.get("path") or "")
            scanned["shards"] += 1
            used_here = 0
            for row, line_number in curated.iter_rows_for_shard(shard):
                if used_here >= max_rows_per_shard or satisfied():
                    break
                scanned["rows"] += 1
                for field in MARKER_FIELDS:
                    marker_values[field][str(row.get(field))[:64]] += 1
                marker = row_marker_reasons(row)
                if marker:
                    drops[f"row_marker:{marker[0]}"] += 1
                    continue
                record = curated.normalize_row(
                    row,
                    source_dataset=dataset_id,
                    training_status=status,
                    shard_path=shard_path,
                    line_number=line_number,
                )
                if record is None:
                    drops["normalize_row_refused"] += 1
                    continue
                if record.route not in routes:
                    drops["route_not_wanted"] += 1
                    continue
                if tokens[record.route] >= tokens_per_route:
                    drops["route_budget_full"] += 1
                    continue
                key = (dataset_id, record.route)
                if per_dataset_route_cap > 0 and per_dataset[key] >= per_dataset_route_cap:
                    drops["per_dataset_route_cap"] += 1
                    continue
                text = render_row_text(record)
                if len(text) < 16:
                    drops["row_too_short"] += 1
                    continue
                reasons = screen.drop_reasons(text)
                if reasons:
                    drops[f"contamination:{reasons[0].split(':')[0]}"] += 1
                    continue
                ids = tokenizer.encode(text)
                if not ids:
                    drops["row_encoded_empty"] += 1
                    continue
                ids.append(tokenizer.separator_id)
                pools[record.route].append(
                    Sampled(
                        route=record.route,
                        ids=ids,
                        record_id=record.record_id,
                        dataset_id=dataset_id,
                        shard_path=shard_path,
                        row_sha256=record.row_sha256,
                        characters=len(text),
                    )
                )
                tokens[record.route] += len(ids)
                per_dataset[key] += 1
                used_here += 1
                entry_rows += 1
                shard_row = shards_used.setdefault(
                    shard_path,
                    {
                        "shard_path": shard_path,
                        "dataset_id": dataset_id,
                        "training_status": status,
                        "shard_fields": {
                            k: v
                            for k, v in shard.items()
                            if isinstance(v, (str, int, float, bool)) and k != "path"
                        },
                        "rows_used": 0,
                        "tokens_collected": 0,
                        "routes": {},
                    },
                )
                shard_row["rows_used"] += 1
                shard_row["tokens_collected"] += len(ids)
                shard_row["routes"][record.route] = shard_row["routes"].get(record.route, 0) + 1
        entries_visited.append(
            {
                "dataset_id": dataset_id,
                "training_status": status,
                "rows_taken": entry_rows,
                "shards_opened": shard_index,
            }
        )
        log(
            f"entry {dataset_id} rows {entry_rows} "
            f"tokens {dict((r, tokens[r]) for r in routes)}"
        )

    report = {
        "pointer": str(pointer),
        "deadline_seconds": deadline_seconds,
        "wall_seconds": round(time.time() - started, 2),
        "tokens_per_route_target": tokens_per_route,
        "entries_visited": entries_visited,
        "scanned": dict(scanned),
        "dropped_rows_by_reason": dict(sorted(drops.items())),
        "collected_tokens_by_route": {route: tokens[route] for route in routes},
        "collected_rows_by_route": {route: len(pools[route]) for route in routes},
        # Stated once, as totals, because the per-entry ``rows_taken`` list is a
        # sequence anyone can accidentally read a running total out of: COLLECTED
        # is what the registry gave, PACKED (in ``packing``) is what the manifest
        # actually carries, and the two are different numbers.
        "collected_rows_total": int(sum(len(pools[route]) for route in routes)),
        "shards_opened_total": len(shards_used),
        "datasets_used_total": len(
            {item.dataset_id for route in routes for item in pools[route]}
        ),
        "mean_characters_by_route": {
            route: (
                round(sum(item.characters for item in pools[route]) / len(pools[route]), 1)
                if pools[route]
                else 0.0
            )
            for route in routes
        },
        "datasets_by_route": {
            route: sorted({item.dataset_id for item in pools[route]}) for route in routes
        },
        "curation_field_values_seen": {
            field: dict(counter.most_common(12)) for field, counter in marker_values.items()
        },
        "source_shards": sorted(shards_used.values(), key=lambda row: row["shard_path"]),
        "satisfied": satisfied(),
    }
    return pools, report


# ------------------------------------------------------------------- packing


def pack_batches(
    pools: Mapping[str, list[Sampled]],
    *,
    routes: tuple[str, ...],
    sequence_length: int,
    batches: int,
    seed: int,
    separator_id: int,
    fit_window: int = 8,
) -> tuple[list[list[int]], list[list[str]], dict[str, Any]]:
    """Interleave the routes into batches, placing every row WHOLE.

    Each pool is shuffled with the recorded seed first: rows adjacent in a shard
    came out of one generation run and are stylistically the same row twice, so
    taking them adjacent would make a batch narrower than its token count says.

    The interleave is not a plain round robin.  Rows differ in length by a factor
    of three across routes (math rows are the longest on this registry), so one
    row per route per cycle balances ROWS and leaves the TOKENS lopsided -- the
    first cut of this manifest came out math 27,260 / reasoning 15,932.  The next
    row is therefore always taken from the route with the fewest tokens emitted
    so far, ties broken by the declared route order.  That is deterministic, it
    balances what the solves actually count, and it puts short-row routes into a
    batch more often, so a 4,096-token batch carries all five routes instead of
    whichever four fitted before the boundary.

    Rows are ATOMIC.  The previous version of this function concatenated every
    row into one flat stream and sliced it at fixed ``sequence_length`` offsets,
    which cut a document in half at every one of the 23 boundaries: 1,827 of the
    24,576 held tokens were the tails of documents the solve had just been fitted
    on, and 2,367 of the internal-check tokens continued a row from the fit batch
    immediately before.  A held or check number computed on that stream is
    optimistic by construction and nobody grading it can say by how much.  So a
    row is placed only if it fits in what is left of the current batch; when
    nothing fits, the remainder is padded with the parent's eos id and the batch
    closes.  The cost is ``padding_fraction`` (measured, and refused by the
    caller above ``--max-padding-fraction``); the property bought is that the
    fit/held split and the internal check split are DOCUMENT splits.

    A row longer than one batch cannot be placed atomically at all.  It is cut
    once, here, to ``sequence_length - 1`` ids plus the separator, and counted in
    ``rows_truncated_to_sequence_length``: a truncated row is still one document
    living in one batch, which is the property everything downstream depends on.

    ``fit_window`` bounds how far down a route's queue the packer may look for a
    row that fits the remainder.  Taking only the head row would pay the whole
    tail of a batch in padding whenever the head happens to be a long one; a
    bounded look-ahead spends that padding on real text instead.  It is bounded
    rather than a full best-fit so the choice stays a deficit-ordered one and the
    packing stays cheap and deterministic; the number of rows taken out of head
    order is recorded (``rows_taken_from_look_ahead``).
    """

    generator = random.Random(seed)
    queues: dict[str, list[Sampled]] = {}
    truncated: Counter[str] = Counter()
    for route in routes:
        items = list(pools.get(route, ()))
        generator.shuffle(items)
        prepared: list[Sampled] = []
        for item in items:
            if len(item.ids) > sequence_length:
                item = replace(
                    item, ids=item.ids[: sequence_length - 1] + [separator_id]
                )
                truncated[route] += 1
            prepared.append(item)
        queues[route] = prepared

    order = {route: index for index, route in enumerate(routes)}
    available_at_start = {route: len(queues[route]) for route in routes}
    emitted: Counter[str] = Counter({route: 0 for route in routes})
    consumed: Counter[str] = Counter()
    consumed_datasets: dict[str, set[str]] = {route: set() for route in routes}
    cut_ids: list[list[int]] = []
    cut_owners: list[list[str]] = []
    placements: list[dict[str, Any]] = []
    padding_by_batch: list[int] = []
    rows_by_batch: list[int] = []
    look_ahead_takes = 0

    for index in range(batches):
        ids: list[int] = []
        owners: list[str] = []
        rows_here = 0
        while len(ids) < sequence_length:
            room = sequence_length - len(ids)
            chosen: tuple[str, int] | None = None
            for route in sorted(routes, key=lambda name: (emitted[name], order[name])):
                queue = queues[route]
                for offset in range(min(fit_window, len(queue))):
                    if len(queue[offset].ids) <= room:
                        chosen = (route, offset)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                break
            route, offset = chosen
            row = queues[route].pop(offset)
            if offset:
                look_ahead_takes += 1
            placements.append(
                {
                    "batch_index": index,
                    "route": route,
                    "offset": len(ids),
                    "tokens": len(row.ids),
                    "dataset_id": row.dataset_id,
                    "row_sha256": row.row_sha256,
                }
            )
            ids.extend(row.ids)
            owners.extend([route] * len(row.ids))
            emitted[route] += len(row.ids)
            consumed[route] += 1
            consumed_datasets[route].add(row.dataset_id)
            rows_here += 1
        pad = sequence_length - len(ids)
        if pad:
            rows_left = sum(len(queues[route]) for route in routes)
            if rows_left == 0:
                raise SystemExit(
                    f"the registry sample ran out inside batch {index}: "
                    f"{len(ids):,} of {sequence_length:,} tokens placed with no rows "
                    f"left.  Raise --token-margin, --max-entries or "
                    f"--per-dataset-route-cap and re-run."
                )
            ids.extend([separator_id] * pad)
            owners.extend(["pad"] * pad)
        rows_by_batch.append(rows_here)
        padding_by_batch.append(pad)
        cut_ids.append(ids)
        cut_owners.append(owners)

    # Measured, not asserted: the two numbers the old packer got wrong are
    # recomputed from where the rows actually landed.
    spanning = sum(
        1 for row in placements if row["offset"] + row["tokens"] > sequence_length
    )
    opened_by_a_row = {row["batch_index"] for row in placements if row["offset"] == 0}
    padding_total = sum(padding_by_batch)
    report = {
        "seed": seed,
        "order": (
            "next row from the route with the fewest tokens emitted so far, ties "
            "by declared route order; each route's pool shuffled with the seed"
        ),
        "rule": (
            "rows are placed WHOLE; a row that does not fit in the remainder of a "
            "batch waits for the next batch and the remainder is padded with the "
            "parent's eos id, so no document spans a batch boundary and no batch "
            "begins mid-row"
        ),
        "separator": "each row's ids are followed by the parent's eos id",
        "padding_token_id": separator_id,
        "rows_consumed_by_route": {route: consumed[route] for route in routes},
        "rows_consumed_total": int(sum(consumed.values())),
        "rows_truncated_to_sequence_length": {
            route: truncated[route] for route in routes if truncated[route]
        },
        "datasets_consumed_by_route": {
            route: sorted(consumed_datasets[route]) for route in routes
        },
        "rows_available_by_route": available_at_start,
        "rows_left_unused_by_route": {route: len(queues[route]) for route in routes},
        "fit_window": fit_window,
        "rows_taken_from_look_ahead": look_ahead_takes,
        "rows_by_batch": rows_by_batch,
        "padding_tokens_by_batch": padding_by_batch,
        "padding_tokens": padding_total,
        "padding_fraction": round(padding_total / (sequence_length * batches), 6),
        "tokens_placed_by_route": {route: emitted[route] for route in routes},
        "tokens_used": sequence_length * batches,
        "split_leakage": {
            "measure": (
                "rows are located by (batch, offset); a row whose offset plus its "
                "length exceeds the batch would span a boundary, and a batch no row "
                "opens at offset 0 would begin mid-row"
            ),
            "rows_placed": len(placements),
            "rows_spanning_batches": spanning,
            "batches_starting_mid_row": batches - len(opened_by_a_row),
            "held_tokens_continuing_fit_rows": 0 if spanning == 0 else None,
            "check_tokens_continuing_non_check_rows": 0 if spanning == 0 else None,
            "statement": (
                "no token of any fit document appears in a held or internal-check "
                "batch, because no document appears in two batches"
            ),
        },
    }
    return cut_ids, cut_owners, report


def split_for(index: int, held_every: int) -> str:
    """``held`` on every ``held_every``-th batch, counted from the first.

    The same rule ``qf_corpus.synthetic_manifest`` uses, so the self-tests and a
    real manifest disagree about the corpus and about nothing else.
    """

    return "held" if index % held_every == held_every - 1 else "fit"


# ------------------------------------------------------------ overwrite guard


def guard_output(args: argparse.Namespace) -> dict[str, Any]:
    """Refuse to destroy a sealed manifest; back it up when --force says to.

    ``--out`` defaults to the live artifact Wave A is gated on and the documented
    invocation takes no arguments at all, so the default behaviour of this tool
    must not be "overwrite the evidence".  A re-run after the curated registry
    gains a shard writes a DIFFERENT seal over the old file, and every receipt
    already naming the old ``manifest_sha256`` -- the dry child's, and every
    child receipt after it -- becomes unresolvable with nothing warning anyone.

    So: existing file, no ``--force`` -> refusal naming the seal that is already
    there.  With ``--force`` -> the manifest, its build receipt and its log are
    COPIED into ``evidence/.superseded-<UTC>/`` before anything is written, and
    the copy is named in the new receipt.  The estate rule is that a destructive
    overwrite gets a backup first; this makes the tool keep it, not the operator.
    """

    out = Path(args.out)
    receipt = Path(args.receipt)
    report: dict[str, Any] = {
        "out": str(out),
        "existed": out.exists(),
        "forced": bool(args.force),
        "superseded_manifest_sha256": None,
        "backup_directory": None,
        "backed_up": [],
    }
    if not out.exists():
        return report
    try:
        previous = json.loads(out.read_text(encoding="utf-8")).get("manifest_sha256")
    except (OSError, ValueError):
        previous = None
    report["superseded_manifest_sha256"] = previous
    if not args.force:
        raise SystemExit(
            f"{out} already exists (manifest_sha256 {previous}).  Re-running this "
            f"builder writes a different seal over the artifact Wave A is gated on "
            f"and orphans every receipt that names the old one.  Pass --force to "
            f"supersede it (the existing manifest, receipt and log are copied to "
            f"evidence/.superseded-<UTC>/ first), or --out a new path."
        )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = out.parent / f".superseded-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for path in (out, receipt, Path(str(out) + ".log"), out.with_suffix(".log")):
        if path.is_file():
            shutil.copy2(path, backup / path.name)
            report["backed_up"].append(
                {"name": path.name, "sha256": file_sha256(backup / path.name)}
            )
    report["backup_directory"] = str(backup)
    log(f"superseding {out} (was {previous}); backup {backup}")
    return report


# --------------------------------------------------------------------- receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    supersedes = guard_output(args)
    sys.path.insert(0, str(Path(args.chimera_src)))
    from chimera_omni.data import curated_shards as curated

    routes = tuple(dict.fromkeys(curated.canonical_route(r) for r in args.routes.split(",") if r.strip()))
    if not routes:
        raise SystemExit("--routes resolved to nothing")

    parent = Path(args.parent)
    tokenizer = load_tokenizer(parent, allow_unpinned=args.allow_unpinned_tokenizer)
    log(
        f"tokenizer {tokenizer.path} sha {tokenizer.sha256[:16]} "
        f"defined rows {tokenizer.defined_rows:,} of {tokenizer.config_vocab_size:,} "
        f"physical; separator id {tokenizer.separator_id}"
    )
    vocabulary = tokenizer.vocabulary_arithmetic()
    if not vocabulary["agrees_with_port_spec"]:
        log(
            f"vocabulary: {vocabulary['dead_rows_measured']} dead rows measured, "
            f"PORT_SPEC 0.3 quotes {vocabulary['dead_rows_quoted_by_port_spec']} "
            f"(= measured + {vocabulary['added_token_rows']} added tokens).  A C1 "
            f"port deriving dead ids from the base alphabet alone would delete the "
            f"added tokens, eos/bos {tokenizer.separator_id} included."
        )
    siblings = sibling_tokenizer_report(args.sibling_parent, tokenizer.sha256)
    for row in siblings:
        log(f"sibling {row['parent']} shares_tokenizer={row['shares_tokenizer']}")

    screen_paths = [Path(p) for p in args.screen_pack]
    screen = build_review_screen(
        screen_paths,
        needle_characters=args.needle_characters,
        allow_unsealed=args.allow_unsealed_screen_source,
    )
    liveness = screen_liveness(screen, probes=args.screen_probes)
    if not liveness["screen_is_live"]:
        raise SystemExit(
            f"the contamination screen did not behave: {liveness}.  A manifest "
            f"built behind a screen that does not fire is not screened evidence."
        )
    log(
        f"screen live: {liveness['spliced_probes_caught']}/{liveness['probes']} spliced "
        f"probes caught, {liveness['clean_controls_falsely_caught']} clean controls "
        f"falsely caught, {len(screen.windows):,} needle windows"
    )

    needed = args.sequence_length * args.batches
    tokens_per_route = int(needed / len(routes) * args.token_margin) + 1
    pools, registry = collect_rows(
        curated,
        pointer=Path(args.pointer),
        routes=routes,
        tokens_per_route=tokens_per_route,
        tokenizer=tokenizer,
        screen=screen,
        max_entries=args.max_entries,
        max_shards_per_entry=args.max_shards_per_entry,
        max_rows_per_shard=args.max_rows_per_shard,
        per_dataset_route_cap=args.per_dataset_route_cap,
        deadline_seconds=args.deadline_seconds,
    )
    registry["pointer_sha256"] = file_sha256(args.pointer)
    summary = curated.summarize_master_pointer(args.pointer)
    registry["master_pointer"] = {
        "schema_version": summary.schema_version,
        "entry_count": summary.entry_count,
        "total_rows": summary.total_rows,
        "total_shards": summary.total_shards,
        "rows_by_state": summary.rows_by_state,
    }

    cut_ids, cut_owners, packing = pack_batches(
        pools,
        routes=routes,
        sequence_length=args.sequence_length,
        batches=args.batches,
        seed=args.seed,
        separator_id=tokenizer.separator_id,
        fit_window=args.fit_window,
    )
    leakage = packing["split_leakage"]
    if leakage["rows_spanning_batches"] or leakage["batches_starting_mid_row"]:
        raise SystemExit(
            f"packing put a document across a batch boundary "
            f"({leakage['rows_spanning_batches']} rows spanning, "
            f"{leakage['batches_starting_mid_row']} batches opening mid-row); the "
            f"fit/held split would not be a document split and nothing was written."
        )
    if packing["padding_fraction"] > args.max_padding_fraction:
        raise SystemExit(
            f"padding is {packing['padding_fraction']:.4f} of the stream, above "
            f"--max-padding-fraction {args.max_padding_fraction}.  Row-atomic "
            f"packing costs padding; this much of it means the rows are long "
            f"relative to --sequence-length, so raise the sequence length or the "
            f"row supply rather than accept a stream that is mostly eos."
        )
    log(
        f"packed {packing['rows_consumed_total']} rows into {args.batches} batches; "
        f"padding {packing['padding_tokens']:,} tokens "
        f"({packing['padding_fraction'] * 100:.3f}%); rows spanning a boundary "
        f"{leakage['rows_spanning_batches']}; batches opening mid-row "
        f"{leakage['batches_starting_mid_row']}"
    )

    # Every id must be spellable by the parent's table; an out-of-range id is a
    # device-side assert deep inside a capture, hours later.
    highest = max(max(batch) for batch in cut_ids)
    if highest >= tokenizer.config_vocab_size:
        raise SystemExit(
            f"the stream carries id {highest}, outside the parent's "
            f"{tokenizer.config_vocab_size} embedding rows"
        )

    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    tokens_by_route_split: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (ids, owners) in enumerate(zip(cut_ids, cut_owners)):
        split = split_for(index, args.held_every)
        composition = Counter(owners)
        rows.append({"batch_index": index, "split": split, "ids": ids})
        batch_rows.append(
            {
                "batch_index": index,
                "split": split,
                "tokens": len(ids),
                "rows": packing["rows_by_batch"][index],
                "padding_tokens": packing["padding_tokens_by_batch"][index],
                "opens_a_row": True,
                "tokens_by_route": dict(sorted(composition.items())),
                "ids_sha256": canonical_sha256(ids),
            }
        )
        for route, count in composition.items():
            tokens_by_route_split[route][split] += count

    # The batch-level half of the screen: rows were screened before packing, and
    # packing concatenates them, so a hit that only exists across a row boundary
    # has to be looked for here or it is never looked for.
    batch_hits: list[dict[str, Any]] = []
    for row in batch_rows:
        decoded = tokenizer.decode(cut_ids[row["batch_index"]])
        reasons = screen.drop_reasons(decoded)
        if reasons:
            batch_hits.append({"batch_index": row["batch_index"], "reasons": list(reasons)})
    if batch_hits:
        raise SystemExit(
            f"contamination screen REFUSES the manifest: {batch_hits}.  A packed "
            f"batch contains the review surface; nothing was written."
        )
    log(f"screened {len(batch_rows)} packed batches after decode: 0 hits")

    source = (
        f"qf_evidence_build.py; curated master registry {args.pointer} "
        f"({summary.entry_count} entries / {summary.total_rows} rows); routes "
        f"{','.join(routes)}; packed by token-deficit interleave to "
        f"{args.sequence_length} tokens; "
        f"screened against {len(screen.sources)} review packs at "
        f"{args.needle_characters}-char containment; seed {args.seed}; receipt "
        f"{Path(args.receipt).name}"
    )
    manifest = qf_corpus.write_manifest(
        rows, tokenizer_sha256=tokenizer.sha256, source=source, out=args.out
    )
    log(f"manifest {args.out} sealed {manifest['manifest_sha256'][:16]}")

    # Read it back through the reader that will refuse it later, and re-derive
    # every split the solves index, so "it loads" is a measurement here.
    reread = qf_corpus.read_manifest(args.out)
    check_mask = qf_corpus.fit_check_rows(reread)
    evidence = qf_corpus.evidence_receipt(reread, args.out)
    fit_positions = [row["batch_index"] for row in reread["rows"] if row["split"] == "fit"]
    internal_check_batches = [
        fit_positions[position]
        for position in range(len(fit_positions))
        if position >= qf_corpus.INTERNAL_CHECK_OFFSET
        and (position - qf_corpus.INTERNAL_CHECK_OFFSET) % qf_corpus.INTERNAL_CHECK_STRIDE == 0
    ]

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "argv": sys.argv[1:],
        },
        "manifest": {
            "path": str(args.out),
            "file_sha256": file_sha256(args.out),
            **evidence,
        },
        "supersedes": supersedes,
        "tokenizer": {
            "path": tokenizer.path,
            "sha256": tokenizer.sha256,
            "sha256_matches_pin": tokenizer.sha256 == EXPECTED_TOKENIZER_SHA256,
            "defined_rows": tokenizer.defined_rows,
            "physical_vocab_size": tokenizer.config_vocab_size,
            "dead_rows": tokenizer.config_vocab_size - tokenizer.defined_rows,
            "vocabulary_arithmetic": vocabulary,
            "separator_id": tokenizer.separator_id,
            "highest_id_in_stream": highest,
            "add_special_tokens": False,
            "sibling_parents": siblings,
        },
        "split": {
            "held_every": args.held_every,
            "rule": "held on every held_every-th batch counted from the first",
            "fit_batches": fit_positions,
            "held_batches": [
                row["batch_index"] for row in reread["rows"] if row["split"] == "held"
            ],
            "internal_check_stride": qf_corpus.INTERNAL_CHECK_STRIDE,
            "internal_check_offset": qf_corpus.INTERNAL_CHECK_OFFSET,
            "internal_check_batches": internal_check_batches,
            "internal_check_rows": int(check_mask.sum()),
            "internal_check_rows_total": int(check_mask.numel()),
        },
        "routes": {
            "requested": list(routes),
            "pad": (
                "'pad' in the per-batch composition is eos padding, not a route: "
                "it is what row-atomic packing leaves at the end of a batch no "
                "remaining row fits into"
            ),
            "tokens_by_route": {
                route: sum(counts.values()) for route, counts in sorted(tokens_by_route_split.items())
            },
            "tokens_by_route_by_split": {
                route: dict(sorted(counts.items()))
                for route, counts in sorted(tokens_by_route_split.items())
            },
        },
        "batches": batch_rows,
        "packing": packing,
        "registry": registry,
        "contamination_screen": {**screen.provenance(), "liveness": liveness, "batch_hits": batch_hits},
        "wall_time_seconds": round(time.time() - started, 3),
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    )
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    log(f"receipt {receipt_path} sha {receipt['receipt_sha256'][:16]}")
    log(
        f"{evidence['batches']} batches x {evidence['sequence_length']} tokens; "
        f"fit {evidence['fit_batches']} / held {evidence['held_batches']}; "
        f"internal check {evidence['internal_check_rows']:,} of {evidence['fit_rows']:,} fit rows"
    )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    banks = Path("/home/cereal/omni-campaign/calibration/banks")
    screen_default = [
        str(banks / "lfm2_semantic_whole_response_prompts_v1.json"),
        str(banks / "omni_review_bank_v2_draft.json"),
        str(banks / "matrix_campaign_probe_prompts_v1.json"),
        str(banks / "matrix_campaign_smoke_prompts_v1.json"),
        str(here / "screen-banks" / "lfm2_extension_bank_prompts_v2.json"),
        str(here / "screen-banks" / "lfm2_paraphrase_triplets_v1.json"),
    ]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="supersede an existing manifest (backed up to evidence/.superseded-<UTC>/ first)",
    )
    parser.add_argument("--receipt", default="")
    parser.add_argument("--parent", default=DEFAULT_PARENT, help="tokenizer.json + config.json come from here")
    parser.add_argument("--sibling-parent", action="append", default=list(DEFAULT_SIBLING_PARENTS))
    parser.add_argument("--pointer", default=DEFAULT_POINTER)
    parser.add_argument("--chimera-src", default=DEFAULT_CHIMERA_SRC)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--held-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--token-margin", type=float, default=1.35)
    #: Proposed decision line, not a threshold this file invented: row-atomic
    #: packing pays padding for a document split, and 2% of the stream is the
    #: point past which the rows are too long for --sequence-length to be the
    #: right rung.  Measured on this registry at 4,096: well under it.
    parser.add_argument("--max-padding-fraction", type=float, default=0.02)
    parser.add_argument(
        "--fit-window",
        type=int,
        default=8,
        help="how far down a route's queue to look for a row that fits the "
        "remainder of a batch before padding it (bounded, deterministic)",
    )
    parser.add_argument("--max-entries", type=int, default=60)
    parser.add_argument("--max-shards-per-entry", type=int, default=4)
    parser.add_argument("--max-rows-per-shard", type=int, default=400)
    parser.add_argument("--per-dataset-route-cap", type=int, default=10)
    parser.add_argument("--deadline-seconds", type=float, default=2400.0)
    parser.add_argument("--screen-pack", action="append", default=None)
    parser.add_argument("--needle-characters", type=int, default=MINIMUM_NEEDLE_CHARACTERS)
    parser.add_argument("--screen-probes", type=int, default=32)
    parser.add_argument("--allow-unsealed-screen-source", action="store_true")
    parser.add_argument("--allow-unpinned-tokenizer", action="store_true")
    args = parser.parse_args(argv)
    if args.screen_pack is None:
        args.screen_pack = screen_default
    if not args.receipt:
        args.receipt = str(Path(args.out).with_name(Path(args.out).stem + "-build-receipt.json"))
    if args.held_every < 2:
        raise SystemExit("--held-every below 2 leaves no fit batches")
    return args


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
