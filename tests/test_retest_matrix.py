from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "retest_matrix.py"
    spec = importlib.util.spec_from_file_location("retest_matrix", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_manifest_skips_internal_stage_dirs_and_keeps_primary_models(tmp_path: Path) -> None:
    module = _load_module()
    run_root = tmp_path / "run"
    primary = run_root / "candidate_a" / "models" / "01_init"
    internal = primary / "multi_donor_stages" / "00_parent"
    layer_checkpoint = run_root / "candidate_b" / "layer_checkpoints" / "layer_0005"
    for model_dir in (primary, internal, layer_checkpoint):
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(json.dumps({"hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 16}))
        (model_dir / "tokenizer_config.json").write_text("{}")

    manifest = module.build_manifest(
        run_root=run_root,
        repo_root=tmp_path / "repo",
        include_layer_checkpoints=False,
        max_candidates=0,
    )

    labels = {item["label"] for item in manifest["candidates"]}
    assert "candidate_a__01_init" in labels
    assert all("multi_donor_stages" not in item["model_dir"] for item in manifest["candidates"])
    assert all("layer_checkpoints" not in item["model_dir"] for item in manifest["candidates"])


def test_summary_ranks_behavior_before_perplexity(tmp_path: Path) -> None:
    module = _load_module()
    good = tmp_path / "good.json"
    low_ppl_bad = tmp_path / "bad.json"
    good.write_text(
        json.dumps(
            {
                "model": "good",
                "param_count": 10,
                "behavior_suite": {"correct": 5, "total": 10, "items": []},
                "generation_health": {"correct": 1, "total": 10},
                "choice_accuracy": {"correct": 1, "total": 10, "items": []},
                "perplexity": {"mean": 1000.0},
            }
        ),
        encoding="utf-8",
    )
    low_ppl_bad.write_text(
        json.dumps(
            {
                "model": "bad",
                "param_count": 10,
                "behavior_suite": {"correct": 0, "total": 10, "items": []},
                "generation_health": {"correct": 10, "total": 10},
                "choice_accuracy": {"correct": 10, "total": 10, "items": []},
                "perplexity": {"mean": 1.0},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        module._row_from_entry({"label": "good", "report": str(good)}),
        module._row_from_entry({"label": "bad", "report": str(low_ppl_bad)}),
    ]
    rows.sort(key=module._rank_key)
    assert rows[0]["label"] == "good"


def test_summary_ranks_generation_before_choice_after_behavior_tie(tmp_path: Path) -> None:
    module = _load_module()
    exact_gen = tmp_path / "exact_gen.json"
    choice_only = tmp_path / "choice_only.json"
    exact_gen.write_text(
        json.dumps(
            {
                "model": "exact_gen",
                "behavior_suite": {"correct": 5, "total": 10, "items": []},
                "generation_health": {"correct": 8, "total": 10},
                "choice_accuracy": {"correct": 1, "total": 10, "items": []},
                "perplexity": {"mean": 1000.0},
            }
        ),
        encoding="utf-8",
    )
    choice_only.write_text(
        json.dumps(
            {
                "model": "choice_only",
                "behavior_suite": {"correct": 5, "total": 10, "items": []},
                "generation_health": {"correct": 1, "total": 10},
                "choice_accuracy": {"correct": 10, "total": 10, "items": []},
                "perplexity": {"mean": 1.0},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        module._row_from_entry({"label": "choice_only", "report": str(choice_only)}),
        module._row_from_entry({"label": "exact_gen", "report": str(exact_gen)}),
    ]
    rows.sort(key=module._rank_key)
    assert rows[0]["label"] == "exact_gen"


def test_summary_ranks_critical_behavior_before_generic_generation(tmp_path: Path) -> None:
    module = _load_module()
    balanced = tmp_path / "balanced.json"
    broad_only = tmp_path / "broad_only.json"

    balanced.write_text(
        json.dumps(
            {
                "model": "balanced",
                "behavior_suite": {
                    "correct": 5,
                    "total": 10,
                    "items": [
                        {"id": "math_1", "category": "math", "passed": True},
                        {"id": "math_2", "category": "math", "passed": True},
                        {"id": "code_1", "category": "code", "passed": True},
                        {"id": "code_2", "category": "code", "passed": False},
                    ],
                },
                "generation_health": {"correct": 1, "total": 10},
                "choice_accuracy": {"correct": 1, "total": 10, "items": []},
                "perplexity": {"mean": 1000.0},
            }
        ),
        encoding="utf-8",
    )
    broad_only.write_text(
        json.dumps(
            {
                "model": "broad_only",
                "behavior_suite": {
                    "correct": 5,
                    "total": 10,
                    "items": [
                        {"id": "math_1", "category": "math", "passed": False},
                        {"id": "math_2", "category": "math", "passed": False},
                        {"id": "code_1", "category": "code", "passed": False},
                        {"id": "code_2", "category": "code", "passed": False},
                    ],
                },
                "generation_health": {"correct": 10, "total": 10},
                "choice_accuracy": {"correct": 10, "total": 10, "items": []},
                "perplexity": {"mean": 1.0},
            }
        ),
        encoding="utf-8",
    )

    rows = [
        module._row_from_entry({"label": "broad_only", "report": str(broad_only)}),
        module._row_from_entry({"label": "balanced", "report": str(balanced)}),
    ]
    rows.sort(key=module._rank_key)

    assert rows[0]["label"] == "balanced"
    assert rows[0]["critical_behavior_min_rate"] == 0.5
    assert rows[1]["critical_behavior_min_rate"] == 0.0


def test_summary_uses_parent_retention_before_perplexity(tmp_path: Path) -> None:
    module = _load_module()
    retained = tmp_path / "retained.json"
    low_ppl = tmp_path / "low_ppl.json"
    for path, model, ppl in ((retained, "retained", 1000.0), (low_ppl, "low_ppl", 1.0)):
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "behavior_suite": {"correct": 5, "total": 10, "items": []},
                    "generation_health": {"correct": 5, "total": 10},
                    "choice_accuracy": {"correct": 5, "total": 10, "items": []},
                    "perplexity": {"mean": ppl},
                }
            ),
            encoding="utf-8",
        )
    retention_dir = tmp_path / "retention_retained"
    retention_dir.mkdir()
    (retention_dir / "summary.json").write_text(
        json.dumps(
            {
                "behavior": {"overall": {"shared_parent_pass_retention_rate": 0.8, "parent_pass": 10}},
                "generation": {"overall": {"shared_parent_pass_retention_rate": 0.7, "parent_pass": 10}},
                "choice": {"overall": {"shared_parent_pass_retention_rate": 0.6, "parent_pass": 10}},
            }
        ),
        encoding="utf-8",
    )
    weak_retention_dir = tmp_path / "retention_low_ppl"
    weak_retention_dir.mkdir()
    (weak_retention_dir / "summary.json").write_text(
        json.dumps(
            {
                "behavior": {"overall": {"shared_parent_pass_retention_rate": 0.2, "parent_pass": 10}},
                "generation": {"overall": {"shared_parent_pass_retention_rate": 0.1, "parent_pass": 10}},
                "choice": {"overall": {"shared_parent_pass_retention_rate": 0.9, "parent_pass": 10}},
            }
        ),
        encoding="utf-8",
    )

    rows = [
        module._row_from_entry(
            {"label": "low_ppl", "report": str(low_ppl), "retention_dir": str(weak_retention_dir)}
        ),
        module._row_from_entry(
            {"label": "retained", "report": str(retained), "retention_dir": str(retention_dir)}
        ),
    ]
    rows.sort(key=module._rank_key)
    assert rows[0]["label"] == "retained"


def test_summary_reads_compare_retention_schema(tmp_path: Path) -> None:
    module = _load_module()
    report = tmp_path / "child.json"
    report.write_text(
        json.dumps(
            {
                "model": "child",
                "behavior_suite": {"correct": 5, "total": 10, "items": []},
                "generation_health": {"correct": 6, "total": 10},
                "choice_accuracy": {"correct": 7, "total": 10, "items": []},
                "perplexity": {"mean": 10.0},
            }
        ),
        encoding="utf-8",
    )
    retention_dir = tmp_path / "retention_child"
    retention_dir.mkdir()
    (retention_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparisons": {
                    "parent_vs_child": {
                        "behavior": {"shared_parent_pass_retention_rate": 0.4, "parent_pass": 5},
                        "generation": {"shared_parent_pass_retention_rate": 0.5, "parent_pass": 6},
                        "choice": {"shared_parent_pass_retention_rate": 0.6, "parent_pass": 7},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    row = module._row_from_entry(
        {"label": "child", "report": str(report), "retention_dir": str(retention_dir)}
    )

    assert row["behavior_parent_retention"] == 0.4
    assert row["generation_parent_retention"] == 0.5
    assert row["choice_parent_retention"] == 0.6


def test_summary_reads_critical_category_retention_and_markdown(tmp_path: Path) -> None:
    module = _load_module()
    report = tmp_path / "child.json"
    report.write_text(
        json.dumps(
            {
                "model": "child",
                "behavior_suite": {
                    "correct": 2,
                    "total": 4,
                    "items": [
                        {"id": "math_1", "category": "math", "passed": True},
                        {"id": "math_2", "category": "math", "passed": False},
                        {"id": "tok_1", "category": "tokenizer", "passed": True},
                    ],
                },
                "generation_health": {"correct": 2, "total": 4},
                "generation_diagnostics": [
                    {"id": "code_1", "category": "code", "passed": True},
                    {"id": "code_2", "category": "code", "passed": False},
                ],
                "choice_accuracy": {"correct": 2, "total": 4, "items": []},
                "perplexity": {"mean": 10.0},
            }
        ),
        encoding="utf-8",
    )
    retention_dir = tmp_path / "retention_child"
    retention_dir.mkdir()
    (retention_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparisons": {
                    "parent_vs_child": {
                        "behavior": {
                            "overall": {"shared_parent_pass_retention_rate": 0.4, "parent_pass": 5},
                            "by_category": {
                                "math": {"retained_parent_pass": 1, "parent_pass": 4},
                                "tokenizer": {"retained_parent_pass": 3, "parent_pass": 3},
                            },
                        },
                        "generation": {
                            "overall": {"shared_parent_pass_retention_rate": 0.5, "parent_pass": 6},
                            "by_category": {"code": {"retained_parent_pass": 2, "parent_pass": 4}},
                        },
                        "choice": {"overall": {"shared_parent_pass_retention_rate": 0.6, "parent_pass": 7}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    row = module._row_from_entry(
        {"label": "child", "report": str(report), "retention_dir": str(retention_dir)}
    )
    markdown = module._summary_markdown(
        {"created_at": "now", "row_count": 1, "complete_count": 1, "rows": [row]}
    )

    assert row["critical_behavior"]["math"] == {"correct": 1, "total": 2}
    assert row["critical_generation"]["code"] == {"correct": 1, "total": 2}
    assert row["critical_behavior_retention_min_rate"] == 0.25
    assert row["critical_generation_retention_min_rate"] == 0.5
    assert "Critical B" in markdown
    assert "math:1/2" in markdown
    assert "code:1/2" in markdown
    assert "25.0%/50.0%" in markdown
