from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.prepare_lfm2_semantic_qualitative_review import (
    prepare_qualitative_review,
)
from experiments.review_lfm2_semantic_frontier import review_frontier
from experiments.run_lfm2_semantic_fold_to_review import (
    _atomic_json,
    _continuous_compilation_receipt,
    _file_sha256,
    _materialize_complete_structural_frontier,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_postfold_frontier import (
    Lfm2SemanticTerminalFrontierBankReference,
    Lfm2SemanticTerminalFrontierSourceRoots,
)
from neural_forge.hf_lfm2_semantic_preterminal_capsule import (
    Lfm2SemanticPreterminalCapsuleReference,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _capsule(value: dict[str, object]) -> Lfm2SemanticPreterminalCapsuleReference:
    return Lfm2SemanticPreterminalCapsuleReference(
        path=Path(str(value["path"])),
        receipt_sha256=str(value["receipt_sha256"]),
        receipt_file_sha256=str(value["receipt_file_sha256"]),
        manifest_sha256=str(value["manifest_sha256"]),
        bindings_sha256=str(value["bindings_sha256"]),
        context_binding_sha256=str(value["context_binding_sha256"]),
    )


def _bank(value: dict[str, object]) -> Lfm2SemanticTerminalFrontierBankReference:
    return Lfm2SemanticTerminalFrontierBankReference(
        path=Path(str(value["path"])),
        receipt_sha256=str(value["receipt_sha256"]),
        receipt_file_sha256=str(value["receipt_file_sha256"]),
        manifest_sha256=str(value["manifest_sha256"]),
        preterminal_capsule_receipt_sha256=str(
            value["preterminal_capsule_receipt_sha256"]
        ),
        terminal_result_sha256=str(value["terminal_result_sha256"]),
        source_roots_sha256=str(value["source_roots_sha256"]),
        candidate_ids=tuple(str(item) for item in value["candidate_ids"]),
    )


def _roots(value: dict[str, object]) -> Lfm2SemanticTerminalFrontierSourceRoots:
    paths = value["role_paths"]
    if not isinstance(paths, dict):
        raise TypeError("frontier source roots omit role_paths")
    result = Lfm2SemanticTerminalFrontierSourceRoots(
        preterminal_capsule_directory=Path(paths["preterminal_capsule"][0]),
        semantic_cache_directory=Path(paths["semantic_cache"][0]),
        tokenizer_bridge_directory=Path(paths["tokenizer_bridge"][0]),
        parent_model_directory=Path(paths["parent_model"][0]),
        donor_tokenizer_directories=tuple(
            Path(item) for item in paths["donor_tokenizer"]
        ),
        target_tokenizer_directories=tuple(
            Path(item) for item in paths["target_tokenizer"]
        ),
    )
    if result.source_roots_sha256 != value["source_roots_sha256"]:
        raise ValueError("frontier source-root reconstruction changed authentication")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    run = args.run_directory.expanduser().absolute()
    suffix = f"-{args.suffix}" if args.suffix else ""
    postfold = _load(run / "postfold-receipt.json")
    capsule = _capsule(postfold["preterminal_capsule"])
    bank = _bank(postfold["terminal_frontier_export"])
    bank_receipt = _load(bank.path / "frontier-receipt.json")
    roots = _roots(bank_receipt["authenticated_source_roots"])
    compilation = _continuous_compilation_receipt(
        postfold=postfold,
        capsule=capsule,
        bank=bank,
    )

    frontier_receipt = run / f"frontier-receipt{suffix}.json"
    frontier = _materialize_complete_structural_frontier(
        capsule=capsule,
        bank_reference=bank,
        source_roots=roots,
        compilation_receipt=compilation,
        materializations_directory=run / f"materializations{suffix}",
        receipt_json=frontier_receipt,
        device=args.device,
        dtype="bfloat16",
    )
    dossier_path = run / f"whole-response-dossier{suffix}.json"
    prompts = args.prompts_json.expanduser().absolute()
    parent_envelope = Path(
        "/home/cereal/model-surgery-lab/artifacts/"
        "lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json"
    )
    dossier = review_frontier(
        frontier_run_receipt=frontier_receipt,
        expected_frontier_run_sha256=frontier["run_sha256"],
        expected_frontier_run_file_sha256=_file_sha256(frontier_receipt),
        prompts_json=prompts,
        output_json=dossier_path,
        candidate_ids=(),
        device=args.device,
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
        parent_envelope=parent_envelope,
        expected_parent_envelope_sha256=(
            "e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97"
        ),
        expected_parent_envelope_file_sha256=(
            "c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7"
        ),
    )
    recipe = postfold["body_recipe_id"]
    qualitative = prepare_qualitative_review(
        dossier_paths=(dossier_path,),
        expected_dossier_file_sha256s=(_file_sha256(dossier_path),),
        expected_dossier_sha256s=(dossier["dossier_sha256"],),
        lane_labels=(recipe,),
        merged_json=run / f"qualitative-merged{suffix}.json",
        markdown_path=run / f"qualitative-review{suffix}.md",
        verdict_template_json=run / f"manual-verdict-template{suffix}.json",
    )
    body = {
        "schema": "neural-forge-lfm2-a0005-frontier-resume-v1",
        "status": "complete-decoded-behavior-ready-for-holistic-inspection",
        "postfold_result_sha256": postfold["result_sha256"],
        "preterminal_capsule": capsule.as_dict(),
        "terminal_frontier_bank": bank.as_dict(),
        "continuous_compilation_sha256": compilation["compilation_sha256"],
        "frontier_run_sha256": frontier["run_sha256"],
        "dossier_sha256": dossier["dossier_sha256"],
        "candidate_count": dossier["reviewed_candidate_count"],
        "prompt_count": dossier["prompt_count"],
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "numeric_diagnostics_have_decision_authority": False,
        "binary_manual_acceptance_gate_required": False,
        "qualitative_artifacts": dict(qualitative),
    }
    receipt = {**body, "resume_sha256": canonical_sha256(body)}
    _atomic_json(run / f"resume-receipt{suffix}.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
