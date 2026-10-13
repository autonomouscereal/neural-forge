"""Audited QVL pixel layout and authentic source asset/grid refusal."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/vision_cache.py"
spec = importlib.util.spec_from_file_location("recovery_vision_cache_test", PATH)
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


def test_exact_c_t_patch_order_and_merge_window_row_positions():
    torch = pytest.importorskip("torch")
    pixels = torch.arange(3 * 32 * 64, dtype=torch.float32).reshape(3, 32, 64)
    flat, grid, yx = V.patchify(pixels)
    assert grid.tolist() == [[1, 2, 4]] and yx.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    expected = torch.stack([pixels[:, y * 16:(y + 1) * 16, x * 16:(x + 1) * 16].unsqueeze(1).expand(3, 2, 16, 16).reshape(-1) for y, x in yx.tolist()])
    assert torch.equal(flat, expected)


def test_original_image_reconstruction_matches_sealed_grid_and_rejects_asset_change(tmp_path):
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    image = tmp_path / "image.png"
    Image.new("RGB", (256, 256), color=(40, 90, 210)).save(image)
    sha = V.R.sha(image)
    pixels = torch.tensor([40, 90, 210], dtype=torch.float32)[:, None, None].expand(3, 256, 256) / 255.
    flat, grid, yx = V.patchify((pixels - .5) / .5)
    positions = torch.zeros(256, 4)
    positions[:, 1], positions[:, 2] = yx[:, 0].float() / 15, yx[:, 1].float() / 15
    teacher = {"grid_thw": grid, "yx": yx, "positions": positions, "cu_seqlens": torch.tensor([0, 256], dtype=torch.int32),
               "block_in": torch.zeros(256, 1152), "block_out": torch.ones(256, 1152)}
    sidecar = {"resized_hw": [256, 256], "grid_thw": [1, 16, 16]}
    config = {"teacher": "qvl32_vit_l13", "knobs": {"block": 13, "resize_factor": 32, "dtype": "fp32", "max_pixels": 262144}}
    actual, actual_positions = V.reconstruct_image(image, sha, sidecar, teacher, config)
    assert torch.equal(actual, flat) and torch.equal(actual_positions, positions)
    teacher["yx"] = teacher["yx"].flip(0)
    with pytest.raises(V.Error, match="patch order"):
        V.reconstruct_image(image, sha, sidecar, teacher, config)
    image.write_bytes(b"changed")
    with pytest.raises(V.Error, match="image bytes changed"):
        V.reconstruct_image(image, sha, sidecar, teacher, config)
