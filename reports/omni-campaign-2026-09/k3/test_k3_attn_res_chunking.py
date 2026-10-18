"""CPU-only regression coverage for the S2P AttnRes memory prerequisite.

The reference is the pre-chunking expression, independently retained here.
These tests do not load checkpoints, use GPUs, or train a model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location("k3_common_chunking_test", Path(__file__).with_name("k3_common.py"))
assert _SPEC is not None and _SPEC.loader is not None
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)


def reference(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def inputs(tokens, blocks, hidden, dtype, noncontiguous=False):
    rng = torch.Generator().manual_seed(20260904)
    stride = 2 if noncontiguous else 1
    prefix = torch.randn(tokens, hidden * stride, generator=rng).to(dtype)[..., ::stride]
    snapshots = torch.randn(tokens, blocks, hidden * stride, generator=rng).to(dtype)[..., ::stride]
    proj = torch.randn(1, hidden * stride, generator=rng).to(dtype)[..., ::stride]
    norm = torch.randn(hidden * stride, generator=rng).to(dtype)[::stride]
    return prefix, snapshots, proj, norm, 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1, 3, 16, 37, 64, 1024])
def test_chunking_preserves_original_expression(dtype, noncontiguous, chunk_size):
    args = inputs(37, 8, 67, dtype, noncontiguous)
    if noncontiguous:
        assert not args[0].is_contiguous() and not args[1].is_contiguous()
    with torch.no_grad():
        expected = reference(*args)
        actual = common.apply_attn_res(*args, token_chunk_size=chunk_size)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    # The reduction axes and numerical kernel are unchanged. On this CPU
    # fixture token chunking must therefore preserve the original bits.
    assert torch.equal(actual, expected), float((actual.float() - expected.float()).abs().max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,blocks", [(0, 0), (0, 8), (19, 0)])
@pytest.mark.parametrize("chunk_size", [None, 1, 7, 1024])
def test_empty_input_and_zero_snapshots(dtype, tokens, blocks, chunk_size):
    args = inputs(tokens, blocks, 32, dtype)
    with torch.no_grad():
        expected = reference(*args)
        actual = common.apply_attn_res(*args, token_chunk_size=chunk_size)
    assert actual.shape == (tokens, 32) and actual.dtype == dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, "3", True, False])
def test_invalid_chunk_size_is_refused_even_for_empty_input(chunk_size):
    with pytest.raises(ValueError, match="positive integer or None"):
        common.apply_attn_res(*inputs(0, 0, 4, torch.float32), token_chunk_size=chunk_size)


@pytest.mark.parametrize("snapshot_tokens", [36, 38])
def test_chunking_cannot_hide_mismatched_residual_token_counts(snapshot_tokens):
    prefix, _, proj, norm, eps = inputs(37, 2, 16, torch.float32)
    snapshots = torch.zeros(snapshot_tokens, 2, 16)
    with pytest.raises(ValueError, match="matching dimensions"):
        common.apply_attn_res(prefix, snapshots, proj, norm, eps, token_chunk_size=7)


@pytest.mark.parametrize("chunk_size", [7, 1024])
def test_token_slices_reach_cat_before_any_full_stack_allocation(monkeypatch, chunk_size):
    tokens = 2 * chunk_size + 3
    args = inputs(tokens, 8, 32, torch.bfloat16, noncontiguous=True)
    with torch.no_grad():
        expected = reference(*args)
    observed = []
    original_cat = torch.cat

    def watched_cat(tensors, *cat_args, **cat_kwargs):
        if len(tensors) == 2 and all(t.ndim == 3 for t in tensors):
            observed.append(tuple(t.shape[0] for t in tensors))
            assert all(t.shape[0] <= chunk_size for t in tensors)
            assert all(t.dtype == torch.bfloat16 for t in tensors)
        return original_cat(tensors, *cat_args, **cat_kwargs)

    monkeypatch.setattr(torch, "cat", watched_cat)
    with torch.no_grad():
        # Exercise the existing five-positional API for the default limit.
        actual = common.apply_attn_res(*args) if chunk_size == 1024 else common.apply_attn_res(
            *args, token_chunk_size=chunk_size
        )
    assert observed == [(chunk_size, chunk_size), (chunk_size, chunk_size), (3, 3)]
    assert torch.equal(actual, expected)


def test_unchunked_option_reaches_original_kernel_as_one_call(monkeypatch):
    args = inputs(2051, 2, 16, torch.float32)
    observed = []
    original = common._apply_attn_res_chunk

    def watched(prefix, snapshots, *weights):
        observed.append((prefix.shape[0], snapshots.shape[0]))
        return original(prefix, snapshots, *weights)

    monkeypatch.setattr(common, "_apply_attn_res_chunk", watched)
    with torch.no_grad():
        actual = common.apply_attn_res(*args, token_chunk_size=None)
        expected = reference(*args)
    assert observed == [(2051, 2051)]
    assert torch.equal(actual, expected)
