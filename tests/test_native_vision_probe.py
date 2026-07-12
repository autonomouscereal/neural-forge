from __future__ import annotations

import json

import pytest


pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from neural_forge.native_vision_probe import (  # noqa: E402
    build_native_vision_probe,
    canonical_image_sha256,
)


def test_native_vision_probe_is_deterministic_and_split(tmp_path) -> None:
    first = build_native_vision_probe(
        tmp_path / "first",
        examples_per_category=2,
        seed=7,
    )
    second = build_native_vision_probe(
        tmp_path / "second",
        examples_per_category=2,
        seed=7,
    )

    assert first["total"] == 8
    assert second["total"] == first["total"]
    assert first["calibration_count"] == 4
    assert first["heldout_count"] == 4
    left = json.loads((tmp_path / "first" / "manifest.json").read_text())
    right = json.loads((tmp_path / "second" / "manifest.json").read_text())
    for left_record, right_record in zip(
        left["records"], right["records"], strict=True
    ):
        assert left_record["image_sha256"] == right_record["image_sha256"]
        assert left_record["answer"] == right_record["answer"]
        with Image.open(tmp_path / "first" / left_record["image"]) as image:
            assert canonical_image_sha256(image) == left_record["image_sha256"]
    assert b"\r\n" not in (tmp_path / "first" / "manifest.json").read_bytes()


def test_native_vision_probe_rejects_existing_output(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_native_vision_probe(output)
