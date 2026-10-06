"""Exact CPU score parity and fixed-target operation/allocation counts."""
import math
import struct

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from test_mediafold_law_statistics import tool, legacy_law, legacy_scores, problem


def assert_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key, wanted in expected.items():
        value = actual[key]
        if isinstance(wanted, float):
            if math.isnan(wanted):
                assert math.isnan(value), key
            else:
                assert struct.pack('d', value) == struct.pack('d', wanted), key
        else:
            assert value == wanted, key


def score_inputs(kind):
    generator = torch.Generator().manual_seed(53216)
    target = torch.randn(27, 5, dtype=torch.float64, generator=generator)
    prediction = target + torch.randn(27, 5, dtype=torch.float64, generator=generator) * .1
    if kind == "noncontiguous":
        prediction = prediction.T.contiguous().T
        target = target.T.contiguous().T
        assert not prediction.is_contiguous() and not target.is_contiguous()
    elif kind in {"bf16", "fp16", "fp32", "integer", "boolean"}:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
                 "integer": torch.int64, "boolean": torch.bool}[kind]
        prediction, target = prediction.to(dtype), target.to(dtype)
    elif kind == "constant":
        target.fill_(3)
    elif kind == "zero":
        target.zero_()
    elif kind == "single_row":
        prediction, target = prediction[:1], target[:1]
    elif kind == "zero_columns":
        prediction, target = prediction[:, :0], target[:, :0]
    elif kind in {"nan", "positive_inf", "negative_inf"}:
        target[2, 1] = {"nan": float('nan'), "positive_inf": float('inf'),
                        "negative_inf": -float('inf')}[kind]
    return prediction, target


@pytest.mark.parametrize("kind", ["random", "noncontiguous", "bf16", "fp16", "fp32",
                                  "integer", "boolean", "constant", "zero", "single_row",
                                  "zero_columns", "nan", "positive_inf", "negative_inf"])
def test_public_scores_preserve_independent_original_values(kind):
    prediction, target = score_inputs(kind)
    assert_exact(tool.write_scores(prediction, target), legacy_scores(prediction, target))


@pytest.mark.parametrize("shape", [(0, 5), (5,)])
def test_public_scores_preserve_unsupported_empty_or_vector_errors(shape):
    target = torch.empty(shape, dtype=torch.float64)
    with pytest.raises(IndexError) as old_error:
        legacy_scores(target, target)
    with pytest.raises(IndexError) as error:
        tool.write_scores(target, target)
    assert str(error.value) == str(old_error.value)


def test_public_scores_keep_shape_check_and_compute_fresh_target_values():
    prediction, target = score_inputs("random")
    with pytest.raises(ValueError, match='score shape mismatch'):
        tool.write_scores(prediction, target[:1])
    first = tool.write_scores(prediction, target)
    target.add_(2)
    second = tool.write_scores(prediction, target)
    assert_exact(second, legacy_scores(prediction, target))
    assert first != second


@pytest.mark.parametrize("kind", ["zero_target", "constant_target", "noncontiguous", "tie_rungs"])
def test_ladder_still_matches_independent_scores_and_ranking(kind):
    design, target, prior, ladder = problem("well_conditioned")
    if kind == "zero_target":
        target.zero_()
    elif kind == "constant_target":
        target.fill_(3)
    elif kind == "noncontiguous":
        design, target, prior = (item.T.contiguous().T for item in (design, target, prior))
    elif kind == "tie_rungs":
        ladder = (1., 1., .1, .1)
    expected, notes = legacy_law(design, target, prior, ladder)
    actual, actual_notes = tool.law_abiding_solve(design, target, prior, ladder=ladder)
    assert torch.equal(actual, expected)
    assert actual_notes == notes


def test_iterator_ladder_is_consumed_once_and_empty_iterator_keeps_prior():
    design, target, prior, ladder = problem("well_conditioned")
    expected, notes = legacy_law(design, target, prior, iter(ladder))
    actual, actual_notes = tool.law_abiding_solve(design, target, prior, ladder=iter(ladder))
    assert torch.equal(actual, expected) and actual_notes == notes
    actual, notes = tool.law_abiding_solve(design, target, prior, ladder=iter(()))
    assert torch.equal(actual, prior) and notes['fell_back_to_prior']


class TargetOperations(TorchDispatchMode):
    """Count actual target-only reductions and full-array temporary creations."""
    def __init__(self, target, design=None):
        super().__init__()
        self.pointer = target.data_ptr()
        self.shape = tuple(target.shape)
        self.design = design
        self.stats_before_prediction = None
        self.centered_pointer = self.squared_pointer = None
        self.counts = dict.fromkeys(('global_norm', 'row_norm', 'column_mean',
                                   'centered_allocation', 'square_allocation', 'variance_sum'), 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        is_target = args and isinstance(args[0], torch.Tensor) and args[0].data_ptr() == self.pointer
        if (self.design is not None and self.stats_before_prediction is None
                and func is torch.ops.aten.mm.default
                and args[0].data_ptr() == self.design.data_ptr()
                and tuple(args[0].shape) == tuple(self.design.shape)):
            self.stats_before_prediction = self.counts.copy()
        if func is torch.ops.aten.linalg_vector_norm.default and is_target:
            dims = args[2] if len(args) > 2 else kwargs.get('dim')
            self.counts['global_norm' if dims is None else 'row_norm'] += 1
        elif func is torch.ops.aten.mean.dim and is_target:
            self.counts['column_mean'] += 1
        centered = (func is torch.ops.aten.sub.Tensor and is_target
                    and tuple(args[1].shape) == (1, self.shape[1]))
        square = (func is torch.ops.aten.pow.Tensor_Scalar
                  and args[0].data_ptr() == self.centered_pointer)
        variance = (func is torch.ops.aten.sum.default
                    and args[0].data_ptr() == self.squared_pointer)
        result = func(*args, **kwargs)
        if centered:
            self.counts['centered_allocation'] += 1
            self.centered_pointer = result.data_ptr()
        elif square:
            self.counts['square_allocation'] += 1
            self.centered_pointer = None
            self.squared_pointer = result.data_ptr()
        elif variance:
            self.counts['variance_sum'] += 1
            self.squared_pointer = None
        return result


def test_nine_rungs_compute_target_statistics_and_variance_temporaries_once():
    design, target, prior, _ = problem("well_conditioned")
    old = TargetOperations(target)
    with old:
        expected, notes = legacy_law(design, target, prior)
    assert set(old.counts.values()) == {9}, old.counts
    current = TargetOperations(target, design)
    with current:
        actual, actual_notes = tool.law_abiding_solve(design, target, prior)
    assert set(current.counts.values()) == {1}, current.counts
    assert set(current.stats_before_prediction.values()) == {1}
    assert torch.equal(actual, expected) and actual_notes == notes


def test_empty_ladder_does_not_compute_target_statistics():
    design, target, prior, _ = problem("well_conditioned")
    counts = TargetOperations(target)
    with counts:
        actual, notes = tool.law_abiding_solve(design, target, prior, ladder=())
    assert set(counts.counts.values()) == {0}
    assert torch.equal(actual, prior) and notes['fell_back_to_prior']
