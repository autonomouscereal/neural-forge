from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (  # noqa: E402
    _donor_prompt_ids,
)


TOKENIZER = REPOSITORY / ".codex-test-runs" / "lfm2-parent-tokenizer"
EVIDENCE = (
    REPOSITORY
    / ".codex-test-runs"
    / "lfm2-natural-language-token-evidence-2025-2026.json"
)
PROMPTS = REPOSITORY / "research" / "lfm2_semantic_whole_response_prompts_v1.json"


def main() -> None:
    tokenizer = load_lfm2_direct_local_tokenizer(TOKENIZER)
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    evaluation_counts: Counter[int] = Counter()
    for prompt in prompts:
        evaluation_counts.update(
            int(token_id)
            for token_id in _donor_prompt_ids(tokenizer, prompt)[0].tolist()
        )
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    corpus_counts = {
        int(token_id): int(count) for token_id, count in evidence["token_counts"]
    }
    ranked = sorted(corpus_counts, key=lambda token_id: (-corpus_counts[token_id], token_id))
    rows = []
    for budget in (356, 512, 1_024, 2_048, 3_072, 4_096, 8_192, 16_384):
        selected = set(ranked[:budget])
        rows.append(
            {
                "budget": budget,
                "exact_occurrences": sum(
                    count
                    for token_id, count in evaluation_counts.items()
                    if token_id in selected
                ),
                "total_occurrences": sum(evaluation_counts.values()),
                "exact_unique": sum(
                    token_id in selected for token_id in evaluation_counts
                ),
                "total_unique": len(evaluation_counts),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
