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
