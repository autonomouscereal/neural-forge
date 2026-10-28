from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


CORPUS = Path(
    r"C:\Users\cereal\Documents\ai training\ai-frontier-research-2026"
    r"\extracted_text"
)
TOKENIZER = REPOSITORY / ".codex-test-runs" / "lfm2-parent-tokenizer"
OUTPUT = (
    REPOSITORY
    / ".codex-test-runs"
    / "lfm2-natural-language-token-evidence-2025-2026.json"
)
PARTIAL = OUTPUT.with_suffix(".partial.json")
YEAR_PATTERN = re.compile(r"(?:^|[^0-9])(?:25|26)[0-9]{2}[-_][0-9]{4,5}(?:[^0-9]|$)")
MAXIMUM_FILES = 1_024
SAMPLE_BYTES_PER_FILE = 192 * 1024
SAMPLE_CHUNK_BYTES = SAMPLE_BYTES_PER_FILE // 3
CHECKPOINT_INTERVAL = 25
POLICY = {
    "schema": "neural-forge-natural-language-fold-fit-policy-v1",
    "source": "local-ai-frontier-research-2026-extracted-text",
    "allowed_publication_years": [2025, 2026],
    "filename_year_pattern": YEAR_PATTERN.pattern,
    "maximum_files": MAXIMUM_FILES,
    "deterministic_selection": "lowest-sha256-relative-path",
    "sample_bytes_per_file": SAMPLE_BYTES_PER_FILE,
    "sample_strategy": "head-middle-tail-equal-byte-chunks",
    "tokenizer_add_special_tokens": False,
    "review_prompt_tokens_used": False,
    "parent_response_tokens_used": False,
    "parent_forward_calls": 0,
    "training_steps": 0,
}


def emit(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sampled_payload(path: Path) -> bytes:
    size = path.stat().st_size
    if size <= SAMPLE_BYTES_PER_FILE:
        return path.read_bytes()
    offsets = [0, max(0, size // 2 - SAMPLE_CHUNK_BYTES // 2), size - SAMPLE_CHUNK_BYTES]
    chunks = []
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            chunks.append(handle.read(SAMPLE_CHUNK_BYTES))
    return b"\n".join(chunks)


def main() -> None:
    started = time.monotonic()
    candidates = [
        path
        for path in CORPUS.rglob("*.txt")
        if YEAR_PATTERN.search(path.stem)
    ]
    candidates.sort(
        key=lambda path: hashlib.sha256(
            path.relative_to(CORPUS).as_posix().encode("utf-8")
        ).hexdigest()
    )
    selected = candidates[:MAXIMUM_FILES]
    inventory = []
    emit(
        "inventory-start",
        eligible_file_count=len(candidates),
        selected_file_count=len(selected),
    )
    for index, path in enumerate(selected):
        inventory.append(
            {
                "path": path.relative_to(CORPUS).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(selected):
            emit(
                "inventory-progress",
                files_complete=index + 1,
                files_total=len(selected),
                elapsed_seconds=time.monotonic() - started,
            )
    inventory_sha256 = canonical_sha256({"policy": POLICY, "files": inventory})
    if OUTPUT.is_file():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        supplied_sha = existing.get("evidence_sha256")
        body = dict(existing)
        body.pop("evidence_sha256", None)
        if (
            supplied_sha == canonical_sha256(body)
            and existing.get("inventory_sha256") == inventory_sha256
        ):
            emit(
                "evidence-reused",
                evidence_sha256=supplied_sha,
                elapsed_seconds=time.monotonic() - started,
            )
            return
    counts: Counter[int] = Counter()
    files_complete = 0
    sampled_byte_count = 0
    if PARTIAL.is_file():
        partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
        if partial.get("inventory_sha256") == inventory_sha256:
            files_complete = int(partial["files_complete"])
            sampled_byte_count = int(partial["sampled_byte_count"])
            counts.update(
                {
                    int(token_id): int(count)
                    for token_id, count in partial["token_counts"]
                }
            )
            emit(
                "tokenization-resumed",
                files_complete=files_complete,
                token_occurrence_count=sum(counts.values()),
            )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        local_files_only=True,
        trust_remote_code=False,
    )
    for index in range(files_complete, len(selected)):
        payload = sampled_payload(selected[index])
        sampled_byte_count += len(payload)
        text = payload.decode("utf-8", errors="replace")
        counts.update(
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        )
        files_complete = index + 1
        if files_complete % CHECKPOINT_INTERVAL == 0 or files_complete == len(selected):
            partial = {
                "schema": "neural-forge-lfm2-natural-language-token-evidence-partial-v1",
                "inventory_sha256": inventory_sha256,
                "files_complete": files_complete,
                "sampled_byte_count": sampled_byte_count,
                "token_counts": [
                    [token_id, counts[token_id]] for token_id in sorted(counts)
                ],
            }
            PARTIAL.write_text(
                json.dumps(partial, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            emit(
                "tokenization-checkpoint",
                files_complete=files_complete,
                files_total=len(selected),
                sampled_byte_count=sampled_byte_count,
                token_occurrence_count=sum(counts.values()),
                unique_token_count=len(counts),
                elapsed_seconds=time.monotonic() - started,
            )
    body = {
        "schema": "neural-forge-lfm2-natural-language-token-evidence-v1",
        "status": "complete",
        "inventory_sha256": inventory_sha256,
        "policy": POLICY,
        "files": inventory,
        "sampled_byte_count": sampled_byte_count,
        "token_counts": [[token_id, counts[token_id]] for token_id in sorted(counts)],
        "token_occurrence_count": sum(counts.values()),
        "unique_token_count": len(counts),
    }
    evidence = {**body, "evidence_sha256": canonical_sha256(body)}
    OUTPUT.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    PARTIAL.unlink(missing_ok=True)
    emit(
        "evidence-complete",
        evidence_sha256=evidence["evidence_sha256"],
        elapsed_seconds=time.monotonic() - started,
        sampled_byte_count=sampled_byte_count,
        token_occurrence_count=sum(counts.values()),
        unique_token_count=len(counts),
    )


if __name__ == "__main__":
    main()
