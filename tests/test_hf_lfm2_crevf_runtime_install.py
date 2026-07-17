from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn

from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    build_crevf_exact_spec,
    decode_crevf_artifact,
    encode_crevf_artifact,
    encode_crevf_tensor,
)
from neural_forge.hf_lfm2_crevf_runtime_install import (
    CrevfPackedEmbedding,
    CrevfPackedLinear,
    Lfm2CrevfRuntimeInstallError,
    install_lfm2_crevf_runtime,
)
from neural_forge import hf_lfm2_crevf_runtime_install as installer


class Lfm2RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5

    def forward(self, inputs):
        normalized = inputs.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(inputs.dtype)


class _TinyLfm2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4, padding_idx=0).to(
            dtype=torch.bfloat16
        )
        self.model.norm = Lfm2RMSNorm(4)
        self.model.conv = nn.Conv1d(1, 1, 3, bias=False).to(dtype=torch.bfloat16)
        self.proj = nn.Linear(4, 4, bias=True).to(dtype=torch.bfloat16)
        self.lm_head = nn.Linear(4, 8, bias=False).to(dtype=torch.bfloat16)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids):
        hidden = self.model.embed_tokens(token_ids)
        hidden = self.model.norm(hidden)
        return self.lm_head(self.proj(hidden))


def _logical_and_aliases(model: nn.Module):
    logical = dict(model.named_parameters(remove_duplicate=False))
    aliases = {}
    canonical_by_identity = {}
    for name, parameter in logical.items():
        canonical = canonical_by_identity.setdefault(id(parameter), name)
        if name == "model.embed_tokens.weight":
            canonical_by_identity[id(parameter)] = name
            canonical = name
        aliases[name] = canonical
    aliases["lm_head.weight"] = "model.embed_tokens.weight"
    return logical, aliases


def _exact_runtime(model: nn.Module):
    logical, aliases = _logical_and_aliases(model)
    blobs = []
    for canonical in sorted(set(aliases.values())):
        blobs.append(
            encode_crevf_tensor(
                build_crevf_exact_spec(
                    name=canonical,
                    role="normalization" if "norm" in canonical else "control",
                    values=logical[canonical].detach(),
                )
            )
        )
    return decode_crevf_artifact(encode_crevf_artifact(blobs)), aliases


def test_install_exact_runtime_preserves_outputs_and_tied_store() -> None:
    torch.manual_seed(7)
    model = _TinyLfm2().eval()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    expected = model(token_ids)
    runtime, aliases = _exact_runtime(model)

    report = install_lfm2_crevf_runtime(model, runtime, aliases)
    observed = model(token_ids)

    assert torch.equal(observed, expected)
    assert isinstance(model.model.embed_tokens, CrevfPackedEmbedding)
    assert isinstance(model.lm_head, CrevfPackedLinear)
    assert (
        model.model.embed_tokens._packed_weight
        is model.lm_head._packed_weight
        is runtime.tensor("model.embed_tokens.weight")
    )
    assert report.canonical_tensor_count == len(runtime.tensor_names)
    assert report.replaced_linear_count == 2
    assert report.replaced_embedding_count == 1
    assert report.replaced_conv1d_count == 1
    assert report.replaced_norm_count == 1
    assert report.tied_vocabulary_stored_once is True
    assert report.persistent_dense_parameter_elements == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    assert tuple(model.state_dict()) == ("neural_forge_device_sentinel",)
    assert model.state_dict()["neural_forge_device_sentinel"].numel() == 0


def test_install_rejects_incomplete_alias_inventory() -> None:
    model = _TinyLfm2()
    runtime, aliases = _exact_runtime(model)
    aliases.pop("proj.bias")
    with pytest.raises(
        Lfm2CrevfRuntimeInstallError,
        match="logical alias inventory",
    ):
        install_lfm2_crevf_runtime(model, runtime, aliases)


def test_install_rejects_untied_vocabulary() -> None:
    model = _TinyLfm2()
    model.lm_head.weight = nn.Parameter(
        model.lm_head.weight.detach().clone(),
    )
    runtime, aliases = _exact_runtime(model)
    with pytest.raises(
        Lfm2CrevfRuntimeInstallError,
        match="multiple untied|not tied",
    ):
        install_lfm2_crevf_runtime(model, runtime, aliases)


def test_install_preflight_rolls_back_unsupported_parameter() -> None:
    model = _TinyLfm2()
    model.unsupported = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    runtime, aliases = _exact_runtime(model)
    original_embedding = model.model.embed_tokens
    original_head = model.lm_head
    with pytest.raises(
        Lfm2CrevfRuntimeInstallError,
        match="outside supported packed modules",
    ):
        install_lfm2_crevf_runtime(model, runtime, aliases)
    assert model.model.embed_tokens is original_embedding
    assert model.lm_head is original_head
    assert not hasattr(model, "neural_forge_crevf_runtime")
    assert not hasattr(model, "neural_forge_device_sentinel")


def test_install_rolls_back_a_late_accounting_failure(monkeypatch) -> None:
    model = _TinyLfm2().train()
    runtime, aliases = _exact_runtime(model)
    original_embedding = model.model.embed_tokens
    original_head = model.lm_head

    def fail_accounting(_runtime):
        raise RuntimeError("injected accounting failure")

    monkeypatch.setattr(
        type(runtime),
        "persistent_encoded_bytes",
        property(fail_accounting),
    )
    with pytest.raises(RuntimeError, match="injected accounting failure"):
        install_lfm2_crevf_runtime(model, runtime, aliases)
    assert model.model.embed_tokens is original_embedding
    assert model.lm_head is original_head
    assert model.training is True
    assert not hasattr(model, "neural_forge_crevf_runtime")
    assert not hasattr(model, "neural_forge_device_sentinel")


def test_runtime_installer_has_no_loader_training_or_generation_calls() -> None:
    source = inspect.getsource(installer)
    tree = ast.parse(source)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"from_pretrained", "generate", "backward", "step"}.intersection(
        attributes
    )
