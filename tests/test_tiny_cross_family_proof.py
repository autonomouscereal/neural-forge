from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "run_tiny_cross_family_proof.py"
    spec = importlib.util.spec_from_file_location("run_tiny_cross_family_proof", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tiny_cross_family_proof_dry_run_wires_before_after_commands(tmp_path: Path) -> None:
    module = _load_module()
    out = tmp_path / "proof"

    exit_code = module.main(
        [
            "--output-dir",
            str(out),
            "--donor-model",
            "donor-model",
            "--target-config-model",
            "target-config",
            "--target-tokenizer-model",
            "target-tokenizer",
            "--dry-run",
        ],
    )

    assert exit_code == 0
    summary = json.loads((out / "proof_summary.json").read_text(encoding="utf-8"))
    command_text = "\n".join(" ".join(command) for command in summary["commands"])
    assert "hf-build-eval-suite" in command_text
    assert "hf-build-recovery-calibration" in command_text
    assert "hf-eval donor-model" in command_text
    assert "hf-random-init" in command_text
    assert "hf-donor-project-init" in command_text
    assert "--fold-mode transport" in command_text
    assert "--transport-backend sinkhorn" in command_text
    assert "--transport-support quantile-medoid" in command_text
    assert "--activation-fold-assignment balanced-signed-similarity" in command_text
    assert "--token-init-mode donor-tokenized-weighted" in command_text
    assert "--token-fallback-mode donor-mean" in command_text
    assert "compare_eval_retention.py" in command_text
    assert "donor_eval.json" in command_text
    assert "folded_eval.json" in command_text
    assert "hf-rank-evals" in command_text
    assert summary["retention_reports"]["donor_vs_folded"].endswith("donor_vs_folded.json")


def test_tiny_cross_family_proof_dry_run_can_include_recovery(tmp_path: Path) -> None:
    module = _load_module()
    out = tmp_path / "proof"

    exit_code = module.main(
        [
            "--output-dir",
            str(out),
            "--recover-steps",
            "2",
            "--dry-run",
        ],
    )

    assert exit_code == 0
    summary = json.loads((out / "proof_summary.json").read_text(encoding="utf-8"))
    command_text = "\n".join(" ".join(command) for command in summary["commands"])
    assert "hf-xfamily-recover" in command_text
    assert "--steps 2" in command_text
    assert "--projected-vocab-kd-weight 0.05" in command_text
    assert "--answer-first-token-top-negatives 8" in command_text
    assert any(any(str(part).endswith("recovered_eval.json") for part in command) for command in summary["commands"])
    assert summary["retention_reports"]["donor_vs_recovered"].endswith("donor_vs_recovered.json")
    assert summary["retention_reports"]["folded_vs_recovered"].endswith("folded_vs_recovered.json")
