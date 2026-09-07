"""Plan/pixel contract tests only; these fixtures prove no teacher capability."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_parent_images as tool


def plan():
    return json.loads((Path(__file__).parent / "qimg_parent_calibration_sep07.json").read_text())


def test_preregistered_plan_preserves_independent_images(tmp_path):
    path = tmp_path / "plan.json"
    body = plan()
    path.write_text(json.dumps(body))
    assert tool.read_plan(path) == body
    assert sum(e["split"] == "fit" for e in body["entries"]) == 6
    assert sum(e["split"] == "holdout" for e in body["entries"]) == 2


@pytest.mark.parametrize("mutation", ["schema", "steps", "duplicate_id", "duplicate_seed", "duplicate_prompt", "seed_bool", "split", "template", "count"])
def test_plan_refuses_implicit_or_ambiguous_generation(tmp_path, mutation):
    body = plan()
    if mutation == "schema":
        body["schema"] = "fixture"
    elif mutation == "steps":
        body["sampling"]["steps"] = 6
    elif mutation.startswith("duplicate_"):
        key = mutation.split("_", 1)[1]
        body["entries"][1][key] = body["entries"][0][key]
    elif mutation == "seed_bool":
        body["entries"][0]["seed"] = True
    elif mutation == "split":
        body["entries"][0]["split"] = "unknown"
    elif mutation == "template":
        body["entries"][0]["prompt"] = "<|im_start|>"
    else:
        body["entries"] = []
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        tool.read_plan(path)


@pytest.mark.parametrize("temporal", [True, False])
def test_native_rgb_save_conversion_without_resize(temporal):
    value = torch.tensor([0.0, 0.5, 1.0]).expand(1, 2, 2, 3)
    got = tool.png_pixels(value.unsqueeze(1) if temporal else value, size=(2, 2))
    assert got.shape == (2, 2, 3)
    assert got[0, 0].tolist() == [0, 127, 255]


@pytest.mark.parametrize("bad", [float("nan"), -0.01, 1.01])
def test_bad_native_pixels_are_not_silently_fixed(bad):
    with pytest.raises(ValueError):
        tool.png_pixels(torch.full((1, 2, 2, 3), bad), size=(2, 2))


def test_multiple_frames_refused():
    with pytest.raises(ValueError):
        tool.png_pixels(torch.zeros(1, 2, 2, 2, 3), size=(2, 2))
