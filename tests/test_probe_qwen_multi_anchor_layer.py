from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "probe_qwen_multi_anchor_layer.py"
)
SPEC = importlib.util.spec_from_file_location("probe_qwen_multi_anchor_layer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_probe_parser_exposes_bounded_multi_anchor_work_controls(tmp_path: Path) -> None:
    args = MODULE._parse_args(
        [
            "--source-model",
            "model",
            "--calibration-jsonl",
            str(tmp_path / "calibration.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
            "--similarity-chunk-size",
            "128",
            "--max-similarity-products",
            "20000000",
            "--max-projection-fits",
            "300000",
        ]
    )

    assert args.similarity_chunk_size == 128
    assert args.max_similarity_products == 20_000_000
    assert args.max_projection_fits == 300_000
