#!/usr/bin/env python3
"""Count how often each vocabulary id occurs in the campaign's sealed corpus.

C01's weighted arm minimizes ``sum_v f_v * ||E_v - Ehat_v||^2``, and ``f`` has to
come from somewhere defensible.  The only corpus the campaign has ever measured
anything against is the one the parent evidence base
``2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff`` was captured
over -- the 256 mapped-donor calibration batches whose per-batch input digests
are listed, in order, in that cache's own ``manifest.json``.  So this script:

* reads the corpus file,
* **refuses unless the sealed manifest's ordered batch digest list is exactly the
  prefix of this corpus's own ``mapped_donor_cache_input_sha256`` values**, which
  is what makes "the sealed evidence base's corpus" a check rather than a claim.
  The capture took the first 24 records; the corpus file holds 256, and the
  digests tie the two together record by record,
* counts ``input_ids`` under the attention mask (padding is not evidence) over
  **the whole 256-record corpus**, and reports the 24-record captured prefix's
  counts separately so a reader can see both.  The wider count is the one the
  weighting uses: a token-frequency prior over ~6,000 tokens would be a prior
  about one capture rather than about the corpus the campaign calibrates on,
* and writes a plain JSON table plus the digests of everything it read.

The kernel re-checks the file's own sha256 against the one the recipe names, so
a weighting that moved cannot be silently absorbed into a receipt.

Usage::

    python mxo_token_frequency.py \
        --corpus /home/.../mapped_donor_calibration.json \
        --trace-cache /home/.../evidence/2dec87ac... \
        --vocab-size 65536 \
        --out /mnt/ai_data/neural-forge-matrix/wave11/token-frequency-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-10a608d5")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_technique_token_embedding import (  # noqa: E402
    TOKEN_FREQUENCY_SCHEMA,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    corpus_path = Path(arguments.corpus)
    corpus_sha = _file_sha256(corpus_path)
    records = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{corpus_path} is not a nonempty list of batches.")

    manifest_path = Path(arguments.trace_cache) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = [str(v) for v in manifest["corpus"]["ordered_batch_input_sha256"]]
    observed = [str(r["mapped_donor_cache_input_sha256"]) for r in records[: len(sealed)]]
    if observed != sealed:
        # Not a warning.  A frequency table counted over a different corpus than
        # the evidence base is a weighting the campaign cannot cite, and the
        # whole reason to weight at all is that the corpus is the sealed one.
        divergence = next(
            (i for i, (a, b) in enumerate(zip(observed, sealed)) if a != b), None
        )
        raise SystemExit(
            "REFUSED: the corpus does not reproduce the sealed evidence base's "
            f"ordered batch digests ({len(sealed)} sealed, {len(observed)} "
            f"observed, first divergence at index {divergence})."
        )

    vocab = int(arguments.vocab_size)
    counts = [0] * vocab
    captured_prefix_counts = [0] * vocab
    tokens = 0
    captured_tokens = 0
    masked_out = 0
    out_of_range: set[int] = set()
    for index, record in enumerate(records):
        ids = record["input_ids"]
        mask = record["attention_mask"]
        # A record holds one sequence as a flat list, and older records in this
        # family hold a batch of them.  Both are accepted; nothing else is.
        rows = ids if ids and isinstance(ids[0], list) else [ids]
        mask_rows = mask if mask and isinstance(mask[0], list) else [mask]
        in_capture = index < len(sealed)
        for row, mask_row in zip(rows, mask_rows):
            for token, keep in zip(row, mask_row):
                if not int(keep):
                    masked_out += 1
                    continue
                value = int(token)
                if 0 <= value < vocab:
                    counts[value] += 1
                    tokens += 1
                    if in_capture:
                        captured_prefix_counts[value] += 1
                        captured_tokens += 1
                else:
                    out_of_range.add(value)
    if out_of_range:
        raise SystemExit(
            f"REFUSED: {len(out_of_range)} token ids fall outside 0..{vocab - 1}, "
            f"e.g. {sorted(out_of_range)[:5]}."
        )

    observed_types = sum(1 for value in counts if value > 0)
    body = {
        "schema": TOKEN_FREQUENCY_SCHEMA,
        "counts": counts,
        "vocab_size": vocab,
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_sha,
        "corpus_record_count": len(records),
        "corpus_token_count": tokens,
        "captured_prefix_record_count": len(sealed),
        "captured_prefix_token_count": captured_tokens,
        "captured_prefix_observed_token_types": sum(
            1 for value in captured_prefix_counts if value > 0
        ),
        "corpus_masked_positions": masked_out,
        "observed_token_types": observed_types,
        "trace_cache_key": str(manifest.get("trace_cache_key") or Path(arguments.trace_cache).name),
        "trace_cache_manifest_sha256": _file_sha256(manifest_path),
        "counting_rule": (
            "one count per unmasked input_ids position across every record of "
            "the mapped-donor calibration corpus whose first "
            f"{len(sealed)} mapped_donor_cache_input_sha256 values reproduce "
            "the sealed parent evidence base's ordered batch digest list "
            "exactly; padding is not evidence and is not counted"
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    arguments.out.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    print(f"{arguments.out}")
    print(f"  sha256                {digest}")
    print(f"  corpus                {corpus_sha[:16]}... ({len(records)} records)")
    print(
        f"  sealed capture prefix {len(sealed)} records verified digest-for-digest, "
        f"{captured_tokens:,} tokens"
    )
    print(f"  counted tokens        {tokens:,} ({masked_out:,} masked positions skipped)")
    print(
        f"  observed token types  {observed_types:,} of {vocab:,} "
        f"({observed_types / vocab:.4%})"
    )
    top = sorted(range(vocab), key=lambda i: -counts[i])[:10]
    print(f"  top ids               {[(i, counts[i]) for i in top]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
