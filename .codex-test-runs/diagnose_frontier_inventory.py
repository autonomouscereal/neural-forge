from __future__ import annotations

import json
import sys
from pathlib import Path

from neural_forge.hf_lfm2_semantic_postfold_frontier import (
    Lfm2SemanticTerminalFrontierBankReference,
    load_lfm2_semantic_terminal_frontier_bank,
)


run = Path(sys.argv[1])
postfold = json.loads((run / "postfold-receipt.json").read_text(encoding="utf-8"))
raw = postfold["terminal_frontier_export"]
expected = Lfm2SemanticTerminalFrontierBankReference(
    path=Path(raw["path"]),
    receipt_sha256=raw["receipt_sha256"],
    receipt_file_sha256=raw["receipt_file_sha256"],
    manifest_sha256=raw["manifest_sha256"],
    preterminal_capsule_receipt_sha256=raw[
        "preterminal_capsule_receipt_sha256"
    ],
    terminal_result_sha256=raw["terminal_result_sha256"],
    source_roots_sha256=raw["source_roots_sha256"],
    candidate_ids=tuple(raw["candidate_ids"]),
)
bank = load_lfm2_semantic_terminal_frontier_bank(
    expected.path,
    expected_receipt_file_sha256=expected.receipt_file_sha256,
    expected_receipt_sha256=expected.receipt_sha256,
    expected_preterminal_capsule_receipt_sha256=(
        expected.preterminal_capsule_receipt_sha256
    ),
    expected_terminal_result_sha256=expected.terminal_result_sha256,
)
manifest = bank.manifest
checks = {
    "reference_equal": bank.reference.as_dict() == expected.as_dict(),
    "reference_actual": bank.reference.as_dict(),
    "reference_expected": expected.as_dict(),
    "manifest_ids_equal": manifest.get("structurally_eligible_candidate_ids")
    == list(expected.candidate_ids),
    "manifest_count_equal": manifest.get("structurally_eligible_candidate_count")
    == len(expected.candidate_ids),
    "all_exported": manifest.get("all_structurally_eligible_candidates_exported")
    is True,
    "ineligible_not_exported": manifest.get(
        "structurally_ineligible_candidates_exported"
    )
    is False,
    "structural_only": manifest.get("structural_safety_is_only_candidate_veto")
    is True,
    "metric_threshold_false": manifest.get(
        "candidate_metric_threshold_used_for_export"
    )
    is False,
    "candidate_ids_nonempty": bool(expected.candidate_ids),
}
print(json.dumps(checks, sort_keys=True, indent=2))
