from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureError,
    fingerprint_lfm2_parent,
)


class _Config:
    def __init__(self, *, block_ff_dim: int | None) -> None:
        self.block_ff_dim = block_ff_dim
        self.intermediate_size = 7

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model_type": "lfm2",
            "hidden_size": 4,
            "intermediate_size": self.intermediate_size,
            "_name_or_path": "ignored/local/path",
        }
        if hasattr(self, "block_ff_dim"):
            result["block_ff_dim"] = self.block_ff_dim
        return result


class _Model(torch.nn.Module):
    def __init__(self, *, block_ff_dim: int | None) -> None:
        super().__init__()
        self.config = _Config(block_ff_dim=block_ff_dim)
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("runtime_scale", torch.ones(1), persistent=False)


def test_parent_fingerprint_canonicalizes_redundant_ffn_alias() -> None:
    without_alias = _Model(block_ff_dim=None)
    del without_alias.config.block_ff_dim
    with_alias = _Model(block_ff_dim=7)
    with_alias.load_state_dict(without_alias.state_dict())

    assert fingerprint_lfm2_parent(with_alias) == fingerprint_lfm2_parent(without_alias)


@pytest.mark.parametrize(
    ("intermediate_size", "block_ff_dim"),
    [(7, True), (7, "7"), (7, 8), (0, 0), (-1, -1)],
)
def test_parent_fingerprint_rejects_invalid_or_conflicting_ffn_alias(
    intermediate_size: int,
    block_ff_dim: object,
) -> None:
    model = _Model(block_ff_dim=None)
    model.config.intermediate_size = intermediate_size
    model.config.block_ff_dim = block_ff_dim

    with pytest.raises(Lfm2CircuitTraceCaptureError, match="block_ff_dim"):
        fingerprint_lfm2_parent(model)
