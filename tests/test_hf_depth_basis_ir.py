from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_depth_basis_ir as depth_ir_module  # noqa: E402
from neural_forge.hf_arch_layout import (  # noqa: E402
    UnsupportedDecoderLayoutError,
    resolve_module,
)
from neural_forge.hf_depth_basis_ir import (  # noqa: E402
    PROJECTION_ROLES,
    DepthBasisSelector,
    FixedLayerDepthBasisLinear,
    compile_hf_depth_basis_ir,
    materialize_hf_depth_basis_ir,
)


def _tiny_llama(*, layers: int = 3, bias: bool = False):
    torch.manual_seed(710)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=37,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
            hidden_act="silu",
            attention_bias=bias,
            mlp_bias=bias,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _tiny_qwen3(*, layers: int = 3):
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(711)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=37,
            hidden_size=12,
            intermediate_size=16,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=16,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _inputs():
    return torch.tensor([[1, 4, 7, 9], [1, 3, 5, 0]], dtype=torch.long)


@pytest.mark.parametrize(
    "factory",
    [_tiny_llama, _tiny_qwen3],
    ids=["llama", "qwen3"],
)
def test_full_rank_runtime_is_exact_canonical_and_executes_every_layer(factory) -> None:
    model = factory()
    model.eval()
    input_ids = _inputs()
    attention_mask = input_ids.ne(0)
    with torch.no_grad():
        expected = model(input_ids, attention_mask=attention_mask).logits

    compiled = compile_hf_depth_basis_ir(model, feature_chunk_size=17)
    calls = [0] * compiled.layout.layer_count
    handles = [
        block.register_forward_hook(
            lambda _module, _args, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            actual = model(input_ids, attention_mask=attention_mask).logits
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert calls == [1] * compiled.layout.layer_count
    assert tuple(compiled.representations) == PROJECTION_ROLES
    assert compiled.report.all_logical_layers_executed is True
    assert compiled.report.compiled_logical_parameters == (
        compiled.report.source_logical_parameters
    )
    assert compiled.report.compiled_unique_parameters == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert len(compiled.report.aliases) == len(PROJECTION_ROLES) * 3

    state = model.state_dict()
    for role, role_report in compiled.role_reports.items():
        representation = compiled.representations[role]
        assert role_report.rank == 3
        assert role_report.basis.selection == "full_rank"
        assert role_report.basis.full_rank_reconstruction is True
        assert all(key in state for key in role_report.state_dict_keys)
        views = [resolve_module(model, path) for path in role_report.paths]
        assert all(isinstance(view, FixedLayerDepthBasisLinear) for view in views)
        assert all(view.representation is representation for view in views)
        assert sum(parameter.numel() for parameter in views[0].parameters()) == 0
    assert not any(
        key.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            )
        )
        for key in state
    )


def test_low_rank_selectors_save_unique_parameters_and_keep_logits_finite() -> None:
    model = _tiny_llama()
    model.eval()
    selectors = {role: 1 for role in PROJECTION_ROLES}

    compiled = compile_hf_depth_basis_ir(model, role_ranks=selectors)
    with torch.no_grad():
        logits = model(_inputs()).logits

    assert bool(torch.isfinite(logits).all())
    assert compiled.report.unique_parameters_saved > 0
    assert compiled.report.selected_unique_parameters < (
        compiled.report.selected_logical_parameters
    )
    assert all(report.rank == 1 for report in compiled.report.roles)
    assert all(report.basis.selection == "rank" for report in compiled.report.roles)
    assert all(
        report.basis.relative_frobenius_error >= 0.0 for report in compiled.report.roles
    )


def test_energy_and_parameter_budget_are_selected_per_role() -> None:
    model = _tiny_llama()
    depth = model.config.num_hidden_layers
    k_proj = model.model.layers[0].self_attn.k_proj
    k_features = k_proj.weight.numel()

    compiled = compile_hf_depth_basis_ir(
        model,
        role_specs={"q": DepthBasisSelector(energy=0.7)},
        role_parameter_budgets={"k": k_features + depth},
    )

    assert compiled.role_reports["q"].basis.selection == "energy"
    assert compiled.role_reports["q"].basis.retained_energy_fraction >= 0.7 - 1e-12
    assert compiled.role_reports["k"].basis.selection == "parameter_budget"
    assert compiled.role_reports["k"].rank == 1
    assert compiled.role_reports["k"].unique_parameter_count == k_features + depth
    assert compiled.role_reports["v"].basis.selection == "full_rank"


def test_bias_dtype_device_and_training_mode_are_preserved() -> None:
    model = _tiny_llama(bias=True).double()
    model.train()

    compiled = compile_hf_depth_basis_ir(model)

    for role, representation in compiled.representations.items():
        assert representation.has_bias is True
        assert representation.basis_weight.dtype == torch.float64
        assert representation.basis_weight.device.type == "cpu"
        assert representation.training is True
        for path in compiled.role_paths[role]:
            view = resolve_module(model, path)
            assert view.training is True
            assert view.has_bias is True

    model.eval()
    assert all(
        not representation.training
        for representation in compiled.representations.values()
    )
    materialize_hf_depth_basis_ir(compiled)
    for paths in compiled.role_paths.values():
        for path in paths:
            linear = resolve_module(model, path)
            assert linear.bias is not None
            assert linear.weight.dtype == torch.float64
            assert linear.training is False


@pytest.mark.parametrize(
    "factory",
    [_tiny_llama, _tiny_qwen3],
    ids=["llama", "qwen3"],
)
def test_materialization_preserves_forward_and_restores_independent_linears(
    factory,
) -> None:
    model = factory()
    model.eval()
    compiled = compile_hf_depth_basis_ir(
        model,
        role_ranks={role: 1 for role in PROJECTION_ROLES},
    )
    with torch.no_grad():
        expected = model(_inputs()).logits

    materialized = materialize_hf_depth_basis_ir(compiled)
    with torch.no_grad():
        actual = materialized(_inputs()).logits

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert compiled.materialized is True
    assert not hasattr(model, "_depth_basis_ir")
    for paths in compiled.role_paths.values():
        modules = [resolve_module(model, path) for path in paths]
        assert all(type(module) is torch.nn.Linear for module in modules)
        assert len({id(module.weight) for module in modules}) == len(modules)
        if modules[0].bias is not None:
            assert len({id(module.bias) for module in modules}) == len(modules)

    q_modules = [resolve_module(model, path) for path in compiled.role_paths["q"]]
    untouched = q_modules[1].weight.detach().clone()
    with torch.no_grad():
        q_modules[0].weight.add_(1.0)
    assert torch.equal(q_modules[1].weight, untouched)


def test_compilation_failure_rolls_back_every_projection(monkeypatch) -> None:
    model = _tiny_llama()
    paths = [
        f"model.layers.{index}.self_attn.{role}_proj"
        for index in range(model.config.num_hidden_layers)
        for role in ("q", "k", "v", "o")
    ] + [
        f"model.layers.{index}.mlp.{role}_proj"
        for index in range(model.config.num_hidden_layers)
        for role in ("gate", "up", "down")
    ]
    original_modules = {path: resolve_module(model, path) for path in paths}
    original_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    real_view = depth_ir_module.FixedLayerDepthBasisLinear
    calls = 0

    def fail_during_replacement(representation, layer_index):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected replacement failure")
        return real_view(representation, layer_index)

    monkeypatch.setattr(
        depth_ir_module,
        "FixedLayerDepthBasisLinear",
        fail_during_replacement,
    )
    with pytest.raises(RuntimeError, match="injected replacement failure"):
        compile_hf_depth_basis_ir(model)

    assert not hasattr(model, "_depth_basis_ir")
    assert all(
        resolve_module(model, path) is module
        for path, module in original_modules.items()
    )
    assert original_state.keys() == model.state_dict().keys()
    for key, expected in original_state.items():
        assert torch.equal(model.state_dict()[key], expected)
    with torch.no_grad():
        assert bool(torch.isfinite(model(_inputs()).logits).all())


def test_materialization_failure_rolls_back_to_executable_views(monkeypatch) -> None:
    model = _tiny_llama()
    model.eval()
    compiled = compile_hf_depth_basis_ir(
        model,
        role_ranks={role: 1 for role in PROJECTION_ROLES},
    )
    original_views = {
        path: resolve_module(model, path)
        for paths in compiled.role_paths.values()
        for path in paths
    }
    with torch.no_grad():
        expected = model(_inputs()).logits
    real_counts = depth_ir_module._parameter_counts

    def reject_dense_commit(candidate):
        logical, unique = real_counts(candidate)
        if not hasattr(candidate, "_depth_basis_ir"):
            unique += 1
        return logical, unique

    monkeypatch.setattr(depth_ir_module, "_parameter_counts", reject_dense_commit)
    with pytest.raises(
        depth_ir_module.HFDepthBasisCompilationError,
        match="did not restore the source unique parameter count",
    ):
        materialize_hf_depth_basis_ir(compiled)

    assert compiled.materialized is False
    assert model._depth_basis_ir is compiled.registry
    assert all(
        resolve_module(model, path) is view for path, view in original_views.items()
    )
    with torch.no_grad():
        actual = model(_inputs()).logits
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_gpt2_and_unknown_models_fail_closed_without_touching_conv1d() -> None:
    gpt2 = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=31,
            n_positions=16,
            n_embd=8,
            n_layer=2,
            n_head=2,
            n_inner=12,
            tie_word_embeddings=True,
        )
    )
    original = gpt2.transformer.h[0].attn.c_attn

    with pytest.raises(UnsupportedDecoderLayoutError, match="Llama and Qwen3"):
        compile_hf_depth_basis_ir(gpt2)
    assert gpt2.transformer.h[0].attn.c_attn is original
    assert type(original).__name__ == "Conv1D"
    assert not hasattr(gpt2, "_depth_basis_ir")

    unknown = torch.nn.Module()
    unknown.config = SimpleNamespace(model_type="mystery")
    with pytest.raises(UnsupportedDecoderLayoutError, match="supported model types"):
        compile_hf_depth_basis_ir(unknown)
    assert not hasattr(unknown, "_depth_basis_ir")
