"""Header/accounting tests only; these fixtures are not Qwen-Image capability evidence."""
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_preflight as tool


def source_header(tmp_path, mutation=None):
    width, inner, head_dim = 12, 48, 4
    shapes = {"img_in.weight": [width, 64]}
    for index in range(2):
        prefix = f"transformer_blocks.{index}."
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"):
            shapes[prefix + f"attn.{name}.weight"] = [width, width]
            shapes[prefix + f"attn.{name}.bias"] = [width]
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            shapes[prefix + f"attn.{name}.weight"] = [head_dim]
        for stream in ("img", "txt"):
            shapes.update({prefix + f"{stream}_mlp.net.0.proj.weight": [inner, width],
                           prefix + f"{stream}_mlp.net.0.proj.bias": [inner],
                           prefix + f"{stream}_mlp.net.2.weight": [width, inner],
                           prefix + f"{stream}_mlp.net.2.bias": [width],
                           prefix + f"{stream}_mod.1.weight": [6 * width, width],
                           prefix + f"{stream}_mod.1.bias": [6 * width]})
    for index, inputs in ((1, 256), (2, width)):
        shapes[f"time_text_embed.timestep_embedder.linear_{index}.weight"] = [width, inputs]
        shapes[f"time_text_embed.timestep_embedder.linear_{index}.bias"] = [width]
    if mutation == "missing_text":
        del shapes["transformer_blocks.1.txt_mlp.net.2.bias"]
    if mutation == "additional_time":
        shapes["time_text_embed.addition_t_embedding.weight"] = [2, width]
    if mutation == "wrong_qkv":
        shapes["transformer_blocks.1.attn.to_q.weight"] = [width + 1, width]
    body, offset = {}, 0
    for name, shape in sorted(shapes.items()):
        count = tool.math.prod(shape)
        body[name] = {"dtype": "F8_E4M3", "shape": shape, "data_offsets": [offset, offset + count]}
        offset += count
    if mutation == "overlap":
        body["img_in.weight"]["data_offsets"] = [1, 1 + tool.math.prod(shapes["img_in.weight"])]
    raw = json.dumps(body, separators=(",", ":")).encode()
    path = tmp_path / "source.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return path


def test_full_dual_stream_header_keeps_missing_evidence_explicit(tmp_path):
    body = tool.inspect_weights(source_header(tmp_path), 1)
    assert body["width"] == 12 and body["depth"] == 2 and body["heads"] == 3
    assert body["mlps"]["image"]["maximum_legal_keep"] == 11
    assert body["mlps"]["text"]["parent_parameters"] == 1212
    assert body["fold_ready"] is False and body["capability_claim"] == "none"
    assert body["source_identity"]["full_file_sha256"] is None
    assert "tensor bodies not read" in body["source_identity"]["scope"]


def test_real_width_biases_make_naive_quarter_inner_insufficient():
    assert tool.biased_gelu_mlp_parameters(3072, 12288) == 75_512_832
    assert tool.biased_gelu_mlp_parameters(3072, 3072) == 18_880_512
    assert 4 * tool.biased_gelu_mlp_parameters(3072, 3072) > 75_512_832
    assert tool.maximum_fourfold_keep(3072, 12288) == 3071
    assert 4 * tool.biased_gelu_mlp_parameters(3072, 3071) <= 75_512_832


@pytest.mark.parametrize("mutation", ["missing_text", "additional_time", "wrong_qkv", "overlap"])
def test_rejects_incomplete_or_changed_contract(tmp_path, mutation):
    with pytest.raises(ValueError):
        tool.inspect_weights(source_header(tmp_path, mutation), 1)


def test_rejects_truncation_without_reading_tensor_payload(tmp_path):
    path = source_header(tmp_path)
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="range"):
        tool.inspect_weights(path, 1)
