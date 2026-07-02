import json
import os
import subprocess
import sys
from pathlib import Path


def test_lm_eval_comparison_reuses_cached_model_result(tmp_path: Path) -> None:
    fake_lm_eval = tmp_path / "fake_lm_eval.py"
    counter = tmp_path / "counter.txt"
    fake_lm_eval.write_text(
        """
import json
import os
import sys
from pathlib import Path

counter = Path(os.environ["LM_EVAL_COUNTER"])
count = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(count + 1))
output_path = Path(sys.argv[sys.argv.index("--output_path") + 1])
output_path.mkdir(parents=True, exist_ok=True)
(output_path / "results.json").write_text(json.dumps({
    "results": {
        "arc_easy": {"acc": 0.5, "acc_stderr": 0.01},
        "piqa": {"acc_norm": 0.625},
    }
}))
""".strip(),
        encoding="utf-8",
    )
    env = {**os.environ, "LM_EVAL_COUNTER": str(counter)}
    command = [
        sys.executable,
        "experiments/run_lm_eval_comparison.py",
        "--output-dir",
        str(tmp_path / "first"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--tasks",
        "arc_easy,piqa",
        "--limit",
        "7",
        "--lm-eval-command",
        f'"{sys.executable}" "{fake_lm_eval}"',
        "--model",
        "parent=Qwen/Qwen3-4B-Base",
    ]
    first = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr

    second_command = list(command)
    second_command[second_command.index("--output-dir") + 1] = str(tmp_path / "second")
    second = subprocess.run(
        second_command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    assert second.returncode == 0, second.stderr
    assert counter.read_text() == "1"

    first_summary = json.loads((tmp_path / "first" / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((tmp_path / "second" / "summary.json").read_text(encoding="utf-8"))
    assert first_summary["models"][0]["cache_hit"] is False
    assert second_summary["models"][0]["cache_hit"] is True
    assert second_summary["models"][0]["results"]["arc_easy"]["acc"] == 0.5
    assert "acc_stderr" not in second_summary["models"][0]["results"]["arc_easy"]
