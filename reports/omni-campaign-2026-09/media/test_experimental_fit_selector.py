"""Synthetic CPU research checks; no QIMG captures or acceptance evidence."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediafold.experimental_fit_selector as selector
from mediafold.linalg import law_abiding_solve, pivoted_columns_gram


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def synthetic_case(constant=False):
    rows = [[100, 1], [100 if constant else 200, 1],
            [100, 3], [100 if constant else 200, 3]]
    features = torch.tensor([row + [0] * 6 for row in rows], dtype=torch.float64)
    readout = torch.tensor([[0.1 if constant else 0, 1] + [0] * 6], dtype=torch.bfloat16)
    bias = torch.tensor([[20 if constant else 2]], dtype=torch.bfloat16)
    targets = features @ readout.double().T + bias.double()
    return features, readout, bias, targets


def fit_selected(features, readout, bias, targets, selected):
    # The proposed Gram never replaces the original features or parent anchor.
    design = torch.cat((features[:, selected], torch.ones((len(features), 1))), dim=1)
    prior = torch.cat((readout[:, selected].double(), bias.double()), dim=1)
    fitted, notes = law_abiding_solve(design, targets, prior)
    residual = design @ fitted.T - targets
    return float(residual.norm() / targets.norm()), notes, design, prior


@pytest.mark.parametrize("constant", [False, True])
def test_actual_solver_counterexamples(constant):
    features, readout, bias, targets = synthetic_case(constant)
    old = pivoted_columns_gram(features.T @ features, 1)
    new = selector.select_fit_output_weighted_columns(features, readout, 1)
    old_error, _, _, _ = fit_selected(features, readout, bias, targets, old)
    new_error, notes, design, prior = fit_selected(features, readout, bias, targets, new)
    assert old.tolist() == [0]
    assert new.tolist() == [1]
    assert old_error > (0.02 if constant else 0.2)
    assert new_error < 3e-6
    assert notes["update_scale"] <= 1
    torch.testing.assert_close(design[:, 0], features[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(prior, torch.cat((readout[:, [1]], bias), dim=1).double(), rtol=0, atol=0)
    # Both source affine biases count; selection indices add no inference weights.
    assert (8 * 2 + 8 + 1 * 8 + 1) / (1 * 2 + 1 + 1 * 1 + 1) == 6.6


def test_returned_indices_preserve_verbatim_storage_and_inputs():
    features, readout, bias, _ = synthetic_case()
    features.requires_grad_(True)
    original_features, original_readout = features.clone(), readout.clone()
    selected = selector.select_fit_output_weighted_columns(features, readout, 1, chunk_rows=1)
    assert selected.dtype == torch.long and selected.device.type == "cpu"
    assert not selected.requires_grad and selected.tolist() == [1]
    torch.testing.assert_close(features, original_features, rtol=0, atol=0)
    torch.testing.assert_close(readout, original_readout, rtol=0, atol=0)
    torch.testing.assert_close(readout[:, selected].view(torch.uint8), readout[:, [1]].view(torch.uint8), rtol=0, atol=0)
    source_w1 = torch.arange(24, dtype=torch.float32).reshape(8, 3).to(torch.float8_e4m3fn)
    source_bytes = source_w1.view(torch.uint8).clone()
    selected_bytes = source_w1.view(torch.uint8).index_select(0, selected)
    assert torch.equal(selected_bytes, source_bytes[[1]])
    assert torch.equal(source_w1.view(torch.uint8), source_bytes)
    assert bias.dtype == torch.bfloat16


@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_chunked_result_matches_independent_centered_gram_reference(chunk):
    features = torch.tensor([[1, 0, 3], [4, 1, 1], [2, 7, 2], [9, 2, 0], [0, 5, 4]], dtype=torch.float64)
    readout = torch.tensor([[0.25, 2, 0.5], [0.5, 0.5, 1]], dtype=torch.bfloat16)
    centered = features - features.mean(dim=0)
    weighted = centered * readout.double().norm(dim=0)
    expected = pivoted_columns_gram(weighted.T @ weighted, 2).sort().values
    actual = selector.select_fit_output_weighted_columns(features, readout, 2, chunk_rows=chunk)
    assert torch.equal(actual, expected)


def test_noncontiguous_views_remain_immutable():
    features = torch.arange(60, dtype=torch.float64).reshape(10, 6)[:, ::2]
    features = features.clone().T.contiguous().T
    # Add independent signals; the original arange columns alone are collinear.
    features[:, 1] = torch.tensor([0, 1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=torch.float64)
    readout = torch.tensor([[1, 9, 2, 8, 3, 7], [4, 6, 5, 5, 6, 4]], dtype=torch.bfloat16)[:, ::2]
    feature_before, weight_before = features.clone(), readout.clone()
    assert not features.is_contiguous() and not readout.is_contiguous()
    selected = selector.select_fit_output_weighted_columns(features, readout, 2, chunk_rows=3)
    assert selected.numel() == selected.unique().numel() == 2
    assert torch.equal(selected, selected.sort().values)
    assert torch.equal(features, feature_before) and torch.equal(readout, weight_before)


def test_output_norm_uses_all_outputs_and_ignores_zero_columns():
    features = torch.tensor([[1, 1, 1000], [-1, 1, -1000], [1, -1, -1000], [-1, -1, 1000]], dtype=torch.float64)
    readout = torch.tensor([[0, 3, 0], [4, 0, 0]], dtype=torch.bfloat16)
    assert selector.select_fit_output_weighted_columns(features, readout, 1).tolist() == [0]
    assert selector.select_fit_output_weighted_columns(features, readout, 2).tolist() == [0, 1]


@pytest.mark.parametrize("factor", [0.25, 2, 8])
def test_moderate_global_output_scaling_keeps_selection(factor):
    features, readout, _, _ = synthetic_case()
    expected = selector.select_fit_output_weighted_columns(features, readout, 1)
    actual = selector.select_fit_output_weighted_columns(features, readout * factor, 1)
    assert torch.equal(actual, expected)


def test_compensated_column_scaling_keeps_selection():
    features = torch.tensor([[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=torch.float64)
    readout = torch.tensor([[4, 1]], dtype=torch.bfloat16)
    expected = selector.select_fit_output_weighted_columns(features, readout, 1)
    transformed_features, transformed_readout = features.clone(), readout.clone()
    transformed_features[:, 0] *= 8
    transformed_readout[:, 0] /= 8
    assert torch.equal(selector.select_fit_output_weighted_columns(transformed_features, transformed_readout, 1), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_real_feature_dtypes(dtype):
    features, readout, _, _ = synthetic_case()
    assert selector.select_fit_output_weighted_columns(features.to(dtype), readout, 1).tolist() == [1]


@pytest.mark.parametrize("features,readout,keep", [
    (torch.ones(4, 2), torch.ones(1, 2, dtype=torch.bfloat16), 1),
    (torch.tensor([[1., 2.], [2., 4.], [3., 6.], [4., 8.]]), torch.ones(1, 2, dtype=torch.bfloat16), 2),
    (torch.eye(4), torch.zeros(1, 4, dtype=torch.bfloat16), 1),
    (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bfloat16), 1),
])
def test_insufficient_rank_refuses(features, readout, keep):
    with pytest.raises(ValueError, match="rank"):
        selector.select_fit_output_weighted_columns(features, readout, keep)


def test_near_dependent_roundoff_rank_refuses():
    features = torch.tensor([[1., 2.], [2., 4.], [3., 6. + 1e-13], [4., 8.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="rank"):
        selector.select_fit_output_weighted_columns(features, torch.ones(1, 2, dtype=torch.bfloat16), 2)


@pytest.mark.parametrize("rows", [3, 7, 1000])
@pytest.mark.parametrize("value", [10000000000.1, 1e20 + 32768.0, 1e100])
@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_large_identical_columns_never_gain_false_centered_rank(rows, value, chunk):
    features = torch.full((rows, 1), value, dtype=torch.float64)
    original = features.clone()
    with pytest.raises(ValueError, match="rank"):
        selector.select_fit_output_weighted_columns(features, torch.ones(1, 1, dtype=torch.bfloat16), 1, chunk_rows=chunk)
    assert torch.equal(features, original)


@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_large_common_offsets_preserve_selection(chunk):
    # Integer offsets/signals below are exactly representable in FP64. This
    # tests numerical translation, not information already lost in input casts.
    features = torch.tensor([[1, 0, 3], [4, 1, 1], [2, 7, 2], [9, 2, 0], [0, 5, 4]], dtype=torch.float64)
    readout = torch.tensor([[0.25, 2, 0.5], [0.5, 0.5, 1]], dtype=torch.bfloat16)
    offsets = torch.tensor([1e12, -1e13, 2e14], dtype=torch.float64)
    baseline = selector.select_fit_output_weighted_columns(features, readout, 2, chunk_rows=chunk)
    shifted = selector.select_fit_output_weighted_columns(features + offsets, readout, 2, chunk_rows=chunk)
    assert torch.equal(shifted, baseline)


def test_large_offset_cannot_create_rank_in_duplicate_columns():
    features = torch.tensor([[1., 2.], [2., 4.], [3., 6.], [4., 8.]], dtype=torch.float64)
    features += torch.tensor([1e12, -1e13], dtype=torch.float64)
    with pytest.raises(ValueError, match="rank"):
        selector.select_fit_output_weighted_columns(features, torch.ones(1, 2, dtype=torch.bfloat16), 2)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 99])
def test_invalid_keep_refuses(value):
    features, readout, _, _ = synthetic_case()
    with pytest.raises(ValueError, match="keep"):
        selector.select_fit_output_weighted_columns(features, readout, value)


@pytest.mark.parametrize("value", [0, -2, False, 2.5])
def test_invalid_chunk_refuses(value):
    features, readout, _, _ = synthetic_case()
    with pytest.raises(ValueError, match="chunk_rows"):
        selector.select_fit_output_weighted_columns(features, readout, 1, chunk_rows=value)


@pytest.mark.parametrize("features,readout", [
    (torch.empty(0, 2), torch.ones(1, 2, dtype=torch.bfloat16)),
    (torch.empty(4, 0), torch.empty(1, 0, dtype=torch.bfloat16)),
    (torch.ones(4), torch.ones(1, 4, dtype=torch.bfloat16)),
    (torch.ones(4, 2), torch.empty(0, 2, dtype=torch.bfloat16)),
    (torch.ones(4, 2), torch.ones(1, 3, dtype=torch.bfloat16)),
    (torch.ones(4, 2), torch.ones(1, 2)),
    (torch.ones(4, 2, dtype=torch.int64), torch.ones(1, 2, dtype=torch.bfloat16)),
    (torch.ones(4, 2, dtype=torch.complex128), torch.ones(1, 2, dtype=torch.bfloat16)),
    (torch.ones(4, 2).to(torch.float8_e4m3fn), torch.ones(1, 2, dtype=torch.bfloat16)),
    (torch.ones(4, 2, device="meta"), torch.ones(1, 2, dtype=torch.bfloat16)),
    (torch.ones(4, 2).to_sparse(), torch.ones(1, 2, dtype=torch.bfloat16)),
])
def test_shape_dtype_device_layout_refuses(features, readout):
    with pytest.raises(ValueError):
        selector.select_fit_output_weighted_columns(features, readout, 1)


@pytest.mark.parametrize("where", ["features", "readout"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_refuse_in_later_chunk(where, bad):
    features, readout, _, _ = synthetic_case()
    if where == "features":
        features[-1, 1] = bad
    else:
        readout = readout.repeat(3, 1)
        readout[-1, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        selector.select_fit_output_weighted_columns(features, readout, 1, chunk_rows=1)


def test_finite_input_gram_overflow_refuses():
    features = torch.tensor([[1e200], [-1e200], [1e200], [-1e200]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite centered FIT Gram"):
        selector.select_fit_output_weighted_columns(features, torch.ones(1, 1, dtype=torch.bfloat16), 1)


def test_finite_input_translation_overflow_refuses():
    features = torch.tensor([[1e308], [-1e308], [1e308], [-1e308]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite translated FIT feature"):
        selector.select_fit_output_weighted_columns(features, torch.ones(1, 1, dtype=torch.bfloat16), 1)


def test_tiny_readout_scale_absolute_rank_floor_refuses():
    features = torch.tensor([[1.], [-1.], [1.], [-1.]], dtype=torch.float64)
    readout = torch.tensor([[1e-20]], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rank"):
        selector.select_fit_output_weighted_columns(features, readout, 1)


def test_nontensor_input_refuses():
    with pytest.raises(TypeError, match="tensor"):
        selector.select_fit_output_weighted_columns([[1.], [2.]], torch.ones(1, 1, dtype=torch.bfloat16), 1)
