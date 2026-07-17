from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from experiments import run_lfm2_cached_causal_turing_eager_shard as module


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.config._attn_implementation == "eager":
            return self.projection(value)
        query = value[:, None, :, :]
        return functional.scaled_dot_product_attention(query, query, query)[:, 0]


class TinyLfm2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.attention = Lfm2Attention(self.config)
        self.register_buffer("runtime", torch.arange(4, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.attention(value)


class FakeTuringCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 6
        return (7, 5)

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 6
        return "Quadro RTX 8000"


def test_eager_binding_preserves_dense_tensor_state_and_avoids_sdpa() -> None:
    model = TinyLfm2()
    before = {
        name: value.detach().clone()
        for name, value in tuple(model.named_parameters())
        + tuple(model.named_buffers())
    }
    value = torch.randn(2, 3, 4)

    with module.forbid_sdpa_dispatch() as audit:
        binding = module.force_lfm2_eager_attention(model)
        output = model(value)

    assert output.shape == value.shape
    assert audit == {"dispatch_attempts": 0}
    assert binding == {
        "original_attention_backends": ["sdpa"],
        "effective_attention_backend": "eager",
        "attention_module_count": 1,
        "config_object_count": 1,
        "tensor_bindings_unchanged": True,
    }
    assert all(
        torch.equal(before[name], value)
        for name, value in tuple(model.named_parameters())
        + tuple(model.named_buffers())
    )


def test_sdpa_tripwire_proves_flash_dispatch_cannot_reenter() -> None:
    model = TinyLfm2()
    value = torch.randn(1, 2, 4)

    with module.forbid_sdpa_dispatch() as audit:
        with pytest.raises(
            module.Lfm2TuringEagerRuntimeError,
            match="SDPA dispatch is forbidden",
        ):
            model(value)

    assert audit == {"dispatch_attempts": 1}


def test_guarded_scope_accepts_only_gpu6_turing_shard2() -> None:
    argv = (
        "--mode",
        "shard",
        "--shard-index",
        "2",
        "--device",
        "cuda:6",
    )

    assert module.validate_guarded_scope(argv, cuda=FakeTuringCuda()) == {
        "mode": "shard",
        "shard_index": 2,
        "device": "cuda:6",
        "device_name": "Quadro RTX 8000",
        "compute_capability": [7, 5],
    }
    with pytest.raises(module.Lfm2TuringEagerRuntimeError):
        module.validate_guarded_scope(
            (*argv[:-1], "cuda:4"),
            cuda=FakeTuringCuda(),
        )
