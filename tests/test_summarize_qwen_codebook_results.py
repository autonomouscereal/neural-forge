from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "summarize_qwen_codebook_results.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_qwen_codebook_results", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
CATEGORIES = ("math", "code", "reasoning", "commonsense", "model_surgery")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_items(
    prefix: str,
    passed: list[bool],
    *,
    candidate: bool,
) -> list[dict[str, Any]]:
    outputs = (
        "zzzzzz",
        "答案答案答案答案答案",
        "",
        "Because the premise supports it.",
        "Merge related neurons and preserve outgoing scale.",
    )
    result = []
    for index, category in enumerate(CATEGORIES):
        result.append(
            {
                "id": f"{prefix}-{category}",
                "category": category,
                "kind": "fixture",
                "prompt": f"English prompt for {category}",
                "passed": passed[index],
                "continuation": outputs[index]
                if candidate
                else f"parent answer for {category}",
            }
        )
    return result


def _evaluation(
    model: str,
    param_count: int,
    passed: list[bool],
    perplexity: float,
    *,
    candidate: bool,
) -> dict[str, Any]:
    behavior = _suite_items("behavior", passed, candidate=candidate)
    generation_items = _suite_items("generation", passed, candidate=candidate)
    generations = [
        {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "output": item["prompt"] + " " + item["continuation"],
        }
        for item in generation_items
    ]
    generation_diagnostics = [
        {
            "id": item["id"],
            "category": item["category"],
            "passed": item["passed"],
        }
        for item in generation_items
    ]
    choices = [
        {
            "id": f"choice-{category}",
            "category": category,
            "correct": passed[index],
            "prediction": index if passed[index] else (index + 1) % len(CATEGORIES),
        }
        for index, category in enumerate(CATEGORIES)
    ]
    return {
        "model": model,
        "param_count": param_count,
        "eval_suite": {
            "source": "/fixture/qwen_codebook_eval.json",
            "text_eval_count": 2,
            "choice_probe_count": len(choices),
            "generation_prompt_count": len(generations),
            "behavior_probe_count": len(behavior),
        },
        "perplexity": {"mean": perplexity, "items": [perplexity - 1, perplexity + 1]},
        "behavior_suite": {
            "correct": sum(passed),
            "total": len(behavior),
            "items": behavior,
        },
        "generation_health": {
            "correct": sum(passed),
            "total": len(generations),
        },
        "generations": generations,
        "generation_diagnostics": generation_diagnostics,
        "choice_accuracy": {
            "correct": sum(passed),
            "total": len(choices),
            "items": choices,
        },
    }


def _aggregate(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": report["model"],
        "param_count": report["param_count"],
        "behavior_correct": report["behavior_suite"]["correct"],
        "behavior_total": report["behavior_suite"]["total"],
        "generation_correct": report["generation_health"]["correct"],
        "generation_total": report["generation_health"]["total"],
        "choice_correct": report["choice_accuracy"]["correct"],
        "choice_total": report["choice_accuracy"]["total"],
        "perplexity_mean": report["perplexity"]["mean"],
    }


def _summary(
    parent_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    pass_key: str,
) -> dict[str, Any]:
    parent = {item["id"]: bool(item[pass_key]) for item in parent_items}
    candidate = {item["id"]: bool(item[pass_key]) for item in candidate_items}
    parent_pass = {item_id for item_id, value in parent.items() if value}
    candidate_pass = {item_id for item_id, value in candidate.items() if value}
    retained = parent_pass & candidate_pass
    return {
        "total_ids": len(set(parent) | set(candidate)),
        "shared_ids": len(set(parent) & set(candidate)),
        "parent_pass": len(parent_pass),
        "child_pass": len(candidate_pass),
        "retained_parent_pass": len(retained),
        "lost_parent_pass": len(parent_pass - candidate_pass),
        "child_only_or_parent_failed_gain": len(candidate_pass - parent_pass),
        "parent_pass_retention_rate": (
            len(retained) / len(parent_pass) if parent_pass else None
        ),
        "shared_parent_pass_retention_rate": (
            len(retained) / len(parent_pass) if parent_pass else None
        ),
    }


def _comparison_suite(
    parent_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    pass_key: str,
) -> dict[str, Any]:
    return {
        "overall": _summary(parent_items, candidate_items, pass_key),
        "by_category": {
            category: _summary(
                [item for item in parent_items if item.get("category") == category],
                [
                    item
                    for item in candidate_items
                    if item.get("category") == category
                ],
                pass_key,
            )
            for category in CATEGORIES
        },
    }


def _contract(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite_source": report["eval_suite"]["source"],
        "counts": {
            "text_evals": report["eval_suite"]["text_eval_count"],
            "choice_probes": report["eval_suite"]["choice_probe_count"],
            "generation_prompts": report["eval_suite"]["generation_prompt_count"],
            "behavior_probes": report["eval_suite"]["behavior_probe_count"],
        },
        "prompt_ids": {
            "choice_probes": [item["id"] for item in report["choice_accuracy"]["items"]],
            "generation_prompts": [item["id"] for item in report["generations"]],
            "behavior_probes": [item["id"] for item in report["behavior_suite"]["items"]],
        },
        "text_evals_sha256": "a" * 64,
        "settings": {"max_new_tokens": 64},
    }


def _make_manifest(
    checkpoint: Path, source_params: int, child_params: int, index_bytes: int
) -> dict[str, Any]:
    files = {
        "model.safetensors": b"fixture tensor payload",
        "config.json": b'{"model_type":"qwen3"}\n',
        "tokenizer_contract.json": b'{"vocab_size":32}\n',
    }
    checkpoint.mkdir(parents=True)
    for name, payload in files.items():
        (checkpoint / name).write_bytes(payload)
    manifest = {
        "schema": "neural-forge-qwen3-codebook-checkpoint-v1",
        "format": "safetensors",
        "files": {name: _sha256(checkpoint / name) for name in files},
        "state_key_sha256": "b" * 64,
        "assignment_hashes": {"assignments": "c" * 64},
        "prototype_source_hashes": {"prototypes": "d" * 64},
        "parameter_report": {
            "source_unique_parameter_count": source_params,
            "unique_parameter_count": child_params,
            "index_buffer_bytes": index_bytes,
        },
    }
    _write_json(checkpoint / "manifest.json", manifest)
    return manifest


def _make_run(
    root: Path,
    tag: str,
    parent_path: Path,
    parent: dict[str, Any],
    child_params: int,
    candidate_passed: list[bool],
    candidate_ppl: float,
) -> Path:
    run_dir = root / tag
    run_dir.mkdir(parents=True)
    candidate = _evaluation(
        f"candidate:{tag}",
        child_params,
        candidate_passed,
        candidate_ppl,
        candidate=True,
    )
    _write_json(run_dir / "candidate_eval.json", candidate)
    generation_parent = [
        {
            **generation,
            **diagnostic,
        }
        for generation, diagnostic in zip(
            parent["generations"], parent["generation_diagnostics"], strict=True
        )
    ]
    generation_candidate = [
        {
            **generation,
            **diagnostic,
        }
        for generation, diagnostic in zip(
            candidate["generations"],
            candidate["generation_diagnostics"],
            strict=True,
        )
    ]
    comparison = {
        "parent_eval": str(parent_path),
        "child_eval": str(run_dir / "candidate_eval.json"),
        "parent": _aggregate(parent),
        "child": _aggregate(candidate),
        "param_compression_ratio": parent["param_count"] / child_params,
        "behavior": _comparison_suite(
            parent["behavior_suite"]["items"],
            candidate["behavior_suite"]["items"],
            "passed",
        ),
        "generation": _comparison_suite(
            generation_parent, generation_candidate, "passed"
        ),
        "choice": _comparison_suite(
            parent["choice_accuracy"]["items"],
            candidate["choice_accuracy"]["items"],
            "correct",
        ),
    }
    comparison_path = run_dir / "parent_candidate_comparison.json"
    _write_json(comparison_path, comparison)
    index_bytes = 10
    manifest = _make_manifest(
        run_dir / "compact_child", parent["param_count"], child_params, index_bytes
    )
    checkpoint = {
        "artifact_directory": str(run_dir / "compact_child"),
        "manifest": manifest,
    }
    contract = _contract(parent)
    aggregate = {
        "parent": comparison["parent"],
        "child": comparison["child"],
        "param_compression_ratio": comparison["param_compression_ratio"],
        "behavior": comparison["behavior"]["overall"],
        "generation": comparison["generation"]["overall"],
        "choice": comparison["choice"]["overall"],
    }
    run_report = {
        "schema": "neural-forge-qwen3-codebook-memory-run-v1",
        "status": "complete",
        "comparison": str(comparison_path),
        "accounting": {
            "parameters": {
                "source_logical": parent["param_count"],
                "source_unique": parent["param_count"],
                "runtime_unique": child_params,
            },
            "indices": {"total_bytes": index_bytes},
            "ratios": {"source_to_runtime": parent["param_count"] / child_params},
        },
        "aggregate": aggregate,
        "candidate_eval": candidate,
        "certification_contract": {
            "contract": contract,
            "cached_parent": {
                "model": parent["model"],
                "eval_counts": contract["counts"],
                "prompt_ids": contract["prompt_ids"],
            },
        },
        "execution_configuration": {
            "dtype": "bfloat16",
            "parameter_dtypes": ["torch.bfloat16"],
            "compact_checkpoint": checkpoint,
        },
        "verification": {
            "full_depth_noop": {
                "maximum_absolute_logit_error": 0.125,
                "mean_absolute_logit_error": 0.01,
                "relative_l2_logit_error": 0.02,
                "top1_match_count": 7,
                "top1_count": 8,
            },
            "compact_checkpoint": checkpoint,
        },
    }
    _write_json(run_dir / "run_report.json", run_report)
    return run_dir


@pytest.fixture
def sweep(tmp_path: Path) -> dict[str, Any]:
    parent_passed = [True, True, False, False, True]
    parent = _evaluation(
        "Qwen/Qwen3-4B-Base",
        1000,
        parent_passed,
        10.0,
        candidate=False,
    )
    parent_path = tmp_path / "cached_parent_eval.json"
    _write_json(parent_path, parent)
    first = _make_run(
        tmp_path,
        "tag-b",
        parent_path,
        parent,
        500,
        [False, True, False, True, True],
        12.0,
    )
    second = _make_run(
        tmp_path,
        "tag-a",
        parent_path,
        parent,
        400,
        [True, False, False, True, True],
        15.0,
    )
    return {
        "parent": parent,
        "parent_path": parent_path,
        "first": first,
        "second": second,
    }


def test_summarizes_multiple_tags_deterministically_and_without_missing_metrics(
    sweep: dict[str, Any],
) -> None:
    summary = MODULE.summarize_evidence(
        [sweep["first"], sweep["second"]], parent_eval=sweep["parent_path"]
    )
    reversed_summary = MODULE.summarize_evidence(
        [sweep["second"], sweep["first"]], parent_eval=sweep["parent_path"]
    )

    assert MODULE.canonical_json(summary) == MODULE.canonical_json(reversed_summary)
    assert summary["schema"] == MODULE.SUMMARY_SCHEMA
    assert [candidate["tag"] for candidate in summary["candidates"]] == [
        "tag-a",
        "tag-b",
    ]
    candidate = summary["candidates"][1]
    parameters = candidate["parameters"]
    assert parameters["source_parameter_count"] == 1000
    assert parameters["unique_child_parameter_count"] == 500
    assert parameters["raw_compression_ratio"] == 2.0
    assert parameters["adjusted_ratio"] == pytest.approx(2000 / 1010)
    assert candidate["artifact_verification"]["status"] == "verified"
    assert candidate["artifact_verification"]["verified_file_count"] == 3

    behavior = candidate["evaluation"]["suites"]["behavior"]
    assert behavior["parent"] == {"passes": 3, "total": 5}
    assert behavior["candidate"] == {"passes": 3, "total": 5}
    assert behavior["retention"]["retained_parent_passes"] == 2
    assert behavior["retention"]["losses"] == 1
    assert behavior["retention"]["gains"] == 1
    assert candidate["evaluation"]["perplexity"] == {
        "parent_mean": 10.0,
        "candidate_mean": 12.0,
        "delta": 2.0,
        "ratio": 1.2,
    }
    assert (
        candidate["evaluation"]["logit_drift"]["maximum_absolute_logit_error"]
        == 0.125
    )
    categories = candidate["evaluation"]["categories"]
    assert categories["behavior"]["math"]["pass_delta"] == -1
    assert categories["generation"]["commonsense"]["pass_delta"] == 1
    assert categories["choice"]["code"]["pass_delta"] == 0

    heuristics = candidate["output_heuristics"]
    assert heuristics["outputs_analyzed"] == 10
    assert heuristics["repetition"]["flagged"] >= 2
    assert heuristics["language_switch"]["flagged"] >= 2
    assert heuristics["empty_output"]["flagged"] == 2
    assert set(candidate["manual_examples"]) == set(MODULE.TARGET_MANUAL_CATEGORIES)
    assert all(candidate["manual_examples"][name] for name in MODULE.TARGET_MANUAL_CATEGORIES)
    assert "params 1,000 -> 500" in MODULE.render_text(summary)


def test_refuses_different_cached_parent_artifacts(sweep: dict[str, Any]) -> None:
    second_parent = copy.deepcopy(sweep["parent"])
    second_parent["behavior_suite"]["items"][0]["continuation"] = "changed evidence"
    second_parent_path = sweep["parent_path"].with_name("different_parent_eval.json")
    _write_json(second_parent_path, second_parent)
    comparison_path = sweep["second"] / "parent_candidate_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["parent_eval"] = str(second_parent_path)
    _write_json(comparison_path, comparison)

    with pytest.raises(
        MODULE.EvidenceSummaryError, match="different cached parent"
    ):
        MODULE.summarize_evidence(
            [sweep["first"], sweep["second"]], parent_eval=sweep["parent_path"]
        )


def test_refuses_mismatched_eval_contracts(sweep: dict[str, Any]) -> None:
    report_path = sweep["second"] / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["certification_contract"]["contract"]["suite_source"] = "/other/suite.json"
    _write_json(report_path, report)

    with pytest.raises(MODULE.EvidenceSummaryError, match="declared/observed eval contract"):
        MODULE.summarize_evidence(
            [sweep["first"], sweep["second"]], parent_eval=sweep["parent_path"]
        )


def test_refuses_tampered_checkpoint_artifact(sweep: dict[str, Any]) -> None:
    (sweep["first"] / "compact_child" / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(MODULE.EvidenceSummaryError, match="checkpoint hash mismatch"):
        MODULE.summarize_evidence(
            [sweep["first"]], parent_eval=sweep["parent_path"]
        )


def test_comparison_only_omits_unavailable_adjusted_ratio_and_logit_drift(
    sweep: dict[str, Any],
) -> None:
    comparison = sweep["first"] / "parent_candidate_comparison.json"
    summary = MODULE.summarize_evidence(
        [comparison], parent_eval=sweep["parent_path"]
    )
    candidate = summary["candidates"][0]

    assert candidate["input"]["kind"] == "parent_candidate_comparison"
    assert candidate["parameters"]["raw_compression_ratio"] == 2.0
    assert "adjusted_ratio" not in candidate["parameters"]
    assert "logit_drift" not in candidate["evaluation"]
    assert candidate["artifact_verification"] == {"status": "not-declared"}
