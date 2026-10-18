#!/usr/bin/env python3
"""CPU tests for streamed_full_kd_trainer.py: stochastic rounding, the Adafactor update, blockwise == unblocked,
the sparse row update, and an exact HostWeightStore save/load round trip.  The engine's gradients are proven by the
tool's own `grad-parity` command against stock autograd (needs a small real model).

    CUDA_VISIBLE_DEVICES= python test_streamed_full_kd_trainer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamed_full_kd_trainer as fk  # noqa: E402

CPU = torch.device("cpu")


def test_stochastic_rounding_is_unbiased() -> None:
    x = torch.full((200_000,), 1.0 + 3.0 / 512.0, dtype=torch.float32)  # between two bf16 neighbours
    g = torch.Generator(device="cpu").manual_seed(1)
    r = fk.stochastic_round_bf16(x, g).float()
    lo, hi = 1.0 + 2.0 / 256.0 - 1.0 / 128.0, 1.0 + 1.0 / 128.0
    values = set(r.unique().tolist())
    assert values <= {1.0, 1.0078125}, values
    mean = float(r.mean())
    assert abs(mean - float(x[0])) < 2e-4, (mean, float(x[0]))
    neg = fk.stochastic_round_bf16(-x, torch.Generator(device="cpu").manual_seed(2)).float()
    assert abs(float(neg.mean()) + float(x[0])) < 2e-4
    exact = torch.tensor([0.5, -2.0, 3.0], dtype=torch.float32)
    assert torch.equal(fk.stochastic_round_bf16(exact, g).float(), exact)
    del lo, hi


def reference_factored(w: torch.Tensor, g: torch.Tensor, vr: torch.Tensor, vc: torch.Tensor, beta2: float, lr: float,
                       clip: float, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g2 = g.double().square() + eps
    vr = vr.double() * beta2 + (1 - beta2) * g2.mean(1)
    vc = vc.double() * beta2 + (1 - beta2) * g2.mean(0)
    v = torch.outer(vr, vc) / vr.mean()
    u = g.double() / v.sqrt()
    u = u / max(1.0, float(u.square().mean().sqrt()) / clip)
    return w.double() - lr * u, vr, vc


def test_factored_update_matches_reference_and_blocks() -> None:
    torch.manual_seed(0)
    w = torch.randn(37, 23, dtype=torch.float32)
    grads = [torch.randn(37, 23) * s for s in (1.0, 3.0, 0.1)]
    for block in (1 << 25, 23 * 5):  # unblocked, then 5-row blocks
        opt = fk.StreamedAdafactor(CPU, seed=3, block_elements=block)
        weight = w.clone()
        ref_w, vr, vc = w.double(), torch.zeros(37, dtype=torch.float64), torch.zeros(23, dtype=torch.float64)
        for step, g in enumerate(grads, start=1):
            beta2 = 1.0 - step ** -0.8
            opt.update("t", weight, g, 1e-2, step=step)
            ref_w, vr, vc = reference_factored(ref_w, g, vr, vc, beta2, 1e-2, 1.0, 1e-30)
        assert torch.allclose(weight.double(), ref_w, atol=1e-6), float((weight.double() - ref_w).abs().max())
        assert torch.allclose(opt.state["t"]["vr"].double(), vr, rtol=1e-5)


def test_unfactored_and_rows() -> None:
    opt = fk.StreamedAdafactor(CPU, seed=4)
    w = torch.randn(11)
    g = torch.randn(11)
    before = w.clone()
    opt.update("bias", w, g, 1e-2, step=1)
    # step 1: beta2 = 0, v = g^2 + eps, u = sign(g); rms(u) = 1 -> no clip
    assert torch.allclose(w, before - 1e-2 * torch.sign(g), atol=1e-6)
    emb = torch.randn(5, 4)
    ids = torch.tensor([1, 3])
    rows = emb[ids].clone()
    rg = torch.randn(2, 4)
    opt.update_rows("embed", rows, rg, ids, 5, 1e-2, step=1)
    vrow = opt.state["embed"]["vrow"]
    assert float(vrow[0]) == 0.0 and float(vrow[2]) == 0.0 and float(vrow[1]) > 0.0
    expect = emb[ids] - 1e-2 * (rg / (rg.square().mean(1, keepdim=True) + 1e-30).sqrt()) / max(
        1.0, float((rg / (rg.square().mean(1, keepdim=True) + 1e-30).sqrt()).square().mean().sqrt()))
    assert torch.allclose(rows, expect, atol=1e-6)


def test_bf16_update_moves_and_resumes_exactly() -> None:
    w0 = (torch.randn(64, 48) * 0.02).to(torch.bfloat16)
    g = torch.randn(64, 48)
    a = fk.StreamedAdafactor(CPU, seed=9)
    b = fk.StreamedAdafactor(CPU, seed=9)
    wa, wb = w0.clone(), w0.clone()
    a.update("w", wa, g, 1e-3, step=1)
    b.update("w", wb, g, 1e-3, step=1)
    assert torch.equal(wa, wb), "same seed, step and name must round identically"
    assert not torch.equal(wa, w0), "a 1e-3 step must move bf16 weights"
    c = fk.StreamedAdafactor(CPU, seed=9)
    c.load_state_dict(a.state_dict())
    a.update("w", wa, g * 0.5, 1e-3, step=2)
    c.update("w", wb, g * 0.5, 1e-3, step=2)
    assert torch.equal(wa, wb), "a resumed optimizer must continue bit for bit"


class TinySource:
    text_prefix = "model"

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(5)
        self.t = {"model.embed_tokens.weight": torch.randn(10, 4, generator=g), "model.norm.weight": torch.zeros(4),
                  "model.layers.0.mlp.up_proj.weight": torch.randn(6, 4, generator=g), "lm_head.weight": torch.randn(10, 4, generator=g)}

    def names_under(self, prefix: str) -> list[str]:
        return sorted(n for n in self.t if n.startswith(prefix))

    def has(self, name: str) -> bool:
        return name in self.t

    def get(self, name: str) -> torch.Tensor:
        return self.t[name]

    def config_dict(self) -> dict:
        return {"model_type": "tiny"}

    def identity(self) -> dict:
        return {"kind": "tiny"}


def test_store_round_trip() -> None:
    store = fk.HostWeightStore(TinySource(), dtype=torch.bfloat16)
    assert set(store.tensors) == {"model.embed_tokens.weight", "model.norm.weight", "model.layers.0.mlp.up_proj.weight", "lm_head.weight"}
    with tempfile.TemporaryDirectory() as tmp:
        saved = store.save(Path(tmp) / "w", shard_bytes=64)
        assert len(saved["files"]) > 1
        other = fk.HostWeightStore(TinySource(), dtype=torch.bfloat16)
        for t in other.tensors.values():
            t.zero_()
        other.load(Path(tmp) / "w")
        for name in store.tensors:
            assert torch.equal(store.tensors[name], other.tensors[name]), name


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
