import importlib.util
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k3_gpu_canary as canary
import k3_dense_pass as kdp
from k3_common import dequant_mxfp4


def test_synthetic_inputs_cover_every_packed_value_and_compare_on_cpu():
    packed, scale = canary.synthetic_dequant_inputs()
    assert packed.unique().numel() == 256
    assert scale.shape == (8, 2)
    result = canary.compare_dequant(dequant_mxfp4, packed, scale, "cpu")
    assert result["shape"] == [8, 64]
    assert result["byte_value_equal"] and result["max_abs_difference"] == 0


def test_dequant_comparison_refuses_nonfinite_values():
    packed, scale = canary.synthetic_dequant_inputs()
    with pytest.raises(RuntimeError, match="nonfinite"):
        canary.compare_dequant(lambda *args, **kw: torch.tensor([float("nan")]), packed, scale, "cpu")


def test_dequant_comparison_refuses_changed_values():
    packed, scale = canary.synthetic_dequant_inputs()
    calls = iter([torch.tensor([1.0]), torch.tensor([2.0])])
    with pytest.raises(RuntimeError, match="differs"):
        canary.compare_dequant(lambda *args, **kw: next(calls), packed, scale, "cpu")
