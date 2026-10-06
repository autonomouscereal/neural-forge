"""CPU arithmetic regressions; synthetic fixtures do not establish child capability."""
from pathlib import Path

import pytest
import torch

from test_c10_union_install import (
    c10, chimera_config, k3, run, trunks, work,
    read_safetensors,
)
from safetensors.torch import load_file


def native_forward(cfg, up, down, rows):
    # Independent native module, with the actual serialized coefficients.
    with torch.device("meta"):
        native = c10.FeedForward(cfg, hidden_size=up.shape[1],
                                 intermediate_size=up.shape[0] // 2,
                                 output_size=down.shape[0])
    native.up.weight = torch.nn.Parameter(up, requires_grad=False)
    native.down.weight = torch.nn.Parameter(down, requires_grad=False)
    with torch.no_grad():
        return native(rows.to(dtype=up.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_runtime_helper_matches_native_fused_feed_forward(dtype):
    cfg = chimera_config(24)
    generator = torch.Generator().manual_seed(901)
    up = (0.37 * torch.randn(48, 32, generator=generator)).to(dtype)
    down = (0.13 * torch.randn(32, 24, generator=generator)).to(dtype)
    rows = 0.7 * torch.randn(19, 32, generator=generator)
    state = torch.random.get_rng_state().clone()
    native = c10.build_runtime_ffn(cfg, up, down, torch.device("cpu"))
    assert torch.equal(state, torch.random.get_rng_state()), "scoring must not consume fitting RNG"
    actual = native(rows.to(dtype=dtype))
    expected = native_forward(cfg, up, down, rows)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    width = up.shape[0] // 2
    widened = ((torch.nn.functional.silu(rows @ up[:width].float().T)
                * (rows @ up[width:].float().T)) @ down.float().T)
    if dtype != torch.float32:
        difference = float((actual.float() - widened).norm() / widened.norm())
        print(f"{dtype}: native-versus-coefficient-cast FP32 relL2={difference:.9g}")
        assert difference > 1e-4


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_install_receipt_scores_exact_serialized_runtime_arithmetic(trunks, k3, work, dtype):
    out = work / f"runtime_arithmetic_{dtype}"
    receipt = run(trunks, k3["dir"], k3["charts"], out, 10,
                  ["--dtype", dtype, "--dump-captures"])
    child = read_safetensors(out, verify=True)
    cfg = chimera_config(34)
    differences = []
    for layer in receipt["layers"]:
        index = layer["layer"]
        up = child[f"blocks.{index}.ffn.up.weight"].torch()
        down = child[f"blocks.{index}.ffn.down.weight"].torch()
        captures = load_file(str(Path(receipt["captures_dir"]) / f"L{index:02d}.safetensors"))
        for split in ("fit", "held"):
            rows, target = captures[f"x_{split}"], captures[f"y_{split}"]
            prediction = torch.cat([native_forward(cfg, up, down, piece)
                                    for piece in rows.split(64)])
            expected = c10.write_scores(prediction, target)
            assert layer["scores"][split]["union"] == pytest.approx(expected, rel=1e-10, abs=1e-12)
            width = up.shape[0] // 2
            old_prediction = (torch.nn.functional.silu(rows @ up[:width].float().T)
                              * (rows @ up[width:].float().T)) @ down.float().T
            old_expected = c10.write_scores(old_prediction, target)
            assert layer["coefficient_cast_fp32_scores"][split]["union"] == pytest.approx(
                old_expected, rel=1e-6, abs=1e-8)
        assert layer["heldout_relL2"] == layer["scores"]["held"]["union"]["relL2"]
        assert layer["scores_dtype"] == dtype
        assert layer["runtime_arithmetic"]["input_dtype"] == dtype
        assert layer["runtime_arithmetic"]["weight_dtype"] == dtype
        assert layer["runtime_arithmetic"]["prediction_dtype"] == dtype
        assert layer["runtime_arithmetic"]["operator"] == "chimera_omni.modeling.layers.FeedForward.forward"
        assert layer["runtime_arithmetic"]["chunk_rows"] == 64
        assert layer["runtime_arithmetic"]["gpu_parity_verified"] is False
        assert layer["coefficient_cast_fp32_scores_dtype"] == "float32"
        differences.append(abs(layer["heldout_relL2"] - layer["coefficient_cast_fp32_heldout_relL2"]))
    print(f"{dtype}: maximum heldout relative-error difference={max(differences):.9g}")
    assert max(differences) > 1e-6


def test_runtime_nonfinite_output_refuses_instead_of_issuing_score():
    cfg = chimera_config(24)
    up = torch.full((48, 32), 100., dtype=torch.float16)
    down = torch.ones(32, 24, dtype=torch.float16)
    native = c10.build_runtime_ffn(cfg, up, down, torch.device("cpu"))
    with pytest.raises(SystemExit, match="nonfinite native runtime"):
        c10.runtime_ffn_prediction(native, torch.full((2, 32), 100.))


def test_runtime_scoring_version_is_bound_into_partial_identity(tmp_path):
    partial = c10.Partial(tmp_path / "result")
    partial.start({"tool_version": "1.2.0"})
    with pytest.raises(SystemExit, match="inputs differ"):
        partial.verify_identity({"tool_version": c10.TOOL_VERSION})


def test_native_source_and_runtime_are_bound_into_actual_partial(trunks, k3, work):
    out = work / "runtime_source_identity"
    run(trunks, k3["dir"], k3["charts"], out, 10,
        ["--stop-after-layer", "0"], expect=c10.EXIT_STOPPED)
    partial = c10.Partial(out)
    import json
    identity = json.loads(partial.inputs_path.read_text())["identity"]
    assert identity["runtime_ffn_source_sha256"] == c10.sha256_file(c10.FeedForward.forward.__code__.co_filename)
    assert identity["runtime_torch_version"] == str(torch.__version__)
    assert identity["runtime_device"] == "cpu"
    changed = {**identity, "runtime_ffn_source_sha256": "0" * 64}
    with pytest.raises(SystemExit, match="runtime_ffn_source_sha256"):
        partial.verify_identity(changed)
