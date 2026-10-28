from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_runtime_semantics_receipt as runtime  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


def _parent(*, seed: int = 7301):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def test_fresh_receipt_rejects_code_identical_foreign_globals_functions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r"""
        import types

        import torch
        from transformers.models.lfm2 import modeling_lfm2

        import neural_forge.hf_lfm2_runtime_semantics_receipt as runtime

        assert not runtime._CANONICAL_RECEIPT

        def clone_with_globals(original, globals_mapping):
            clone = types.FunctionType(
                original.__code__,
                globals_mapping,
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            clone.__module__ = original.__module__
            clone.__qualname__ = original.__qualname__
            clone.__kwdefaults__ = original.__kwdefaults__
            clone.__annotations__ = dict(getattr(original, "__annotations__", {}))
            return clone

        def expect_rejected(label, owner, name, replacement):
            original = getattr(owner, name)
            setattr(owner, name, replacement)
            try:
                runtime.lfm2_runtime_semantics_receipt_sha256()
            except runtime.Lfm2RuntimeSemanticsReceiptError as exc:
                assert "globals" in str(exc).lower(), str(exc)
                print(f"REJECTED {label}")
            else:
                raise AssertionError(f"foreign globals were accepted: {label}")
            finally:
                setattr(owner, name, original)
            assert not runtime._CANONICAL_RECEIPT

        rotate_half = modeling_lfm2.rotate_half
        rotate_globals = dict(rotate_half.__globals__)
        rotate_globals["torch"] = types.SimpleNamespace(
            cat=lambda values, dim=-1: torch.zeros_like(values[0])
        )
        expect_rejected(
            "lfm2_rotate_half",
            modeling_lfm2,
            "rotate_half",
            clone_with_globals(rotate_half, rotate_globals),
        )

        linear_forward = torch.nn.Linear.forward
        expect_rejected(
            "torch_linear_forward",
            torch.nn.Linear,
            "forward",
            clone_with_globals(linear_forward, dict(linear_forward.__globals__)),
        )

        silu = torch.nn.functional.silu
        expect_rejected(
            "torch_functional_silu",
            torch.nn.functional,
            "silu",
            clone_with_globals(silu, dict(silu.__globals__)),
        )
        """
    )
    environment = dict(os.environ)
    path_entries = [str(repo_root / "src"), str(repo_root)]
    existing = environment.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "REJECTED lfm2_rotate_half" in completed.stdout
    assert "REJECTED torch_linear_forward" in completed.stdout
    assert "REJECTED torch_functional_silu" in completed.stdout


def test_fresh_receipt_rejects_registered_module_and_transitive_global_substitution() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r"""
        import sys
        import types

        import torch
        from torch.nn.modules import linear as torch_linear
        from transformers.models.lfm2 import modeling_lfm2

        import neural_forge.hf_lfm2_runtime_semantics_receipt as runtime

        def expect_rejected(label, mutate, restore):
            mutate()
            try:
                runtime.lfm2_runtime_semantics_receipt_sha256()
            except runtime.Lfm2RuntimeSemanticsReceiptError:
                print(f"REJECTED {label}")
            else:
                raise AssertionError(f"runtime substitution was accepted: {label}")
            finally:
                restore()
            assert not runtime._CANONICAL_RECEIPT

        original_rotate = modeling_lfm2.rotate_half
        foreign_name = "neural_forge_test_foreign_lfm2_runtime"
        foreign = types.ModuleType(foreign_name)
        foreign.__file__ = original_rotate.__code__.co_filename
        foreign.__dict__.update(original_rotate.__globals__)
        foreign.__name__ = foreign_name
        foreign.__file__ = original_rotate.__code__.co_filename
        foreign.torch = types.SimpleNamespace(
            cat=lambda values, dim=-1: torch.zeros_like(values[0])
        )
        replacement = types.FunctionType(
            original_rotate.__code__,
            foreign.__dict__,
            original_rotate.__name__,
            original_rotate.__defaults__,
            original_rotate.__closure__,
        )
        replacement.__module__ = original_rotate.__module__
        replacement.__qualname__ = original_rotate.__qualname__
        sys.modules[foreign_name] = foreign
        expect_rejected(
            "registered_foreign_lfm2_module",
            lambda: setattr(modeling_lfm2, "rotate_half", replacement),
            lambda: (
                setattr(modeling_lfm2, "rotate_half", original_rotate),
                sys.modules.pop(foreign_name, None),
            ),
        )

        original_functional = torch_linear.F
        fake_functional = types.SimpleNamespace(
            linear=lambda input, weight, bias=None: torch.zeros_like(input)
        )
        expect_rejected(
            "torch_linear_transitive_functional",
            lambda: setattr(torch_linear, "F", fake_functional),
            lambda: setattr(torch_linear, "F", original_functional),
        )

        original_cat = torch.cat
        expect_rejected(
            "torch_native_cat_binding",
            lambda: setattr(torch, "cat", lambda values, dim=-1: torch.zeros_like(values[0])),
            lambda: setattr(torch, "cat", original_cat),
        )

        original_rsqrt = torch.rsqrt
        expect_rejected(
            "torch_native_rsqrt_binding",
            lambda: setattr(torch, "rsqrt", torch.sqrt),
            lambda: setattr(torch, "rsqrt", original_rsqrt),
        )
        """
    )
    environment = dict(os.environ)
    path_entries = [str(repo_root / "src"), str(repo_root)]
    existing = environment.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "REJECTED registered_foreign_lfm2_module" in completed.stdout
    assert "REJECTED torch_linear_transitive_functional" in completed.stdout
    assert "REJECTED torch_native_cat_binding" in completed.stdout
    assert "REJECTED torch_native_rsqrt_binding" in completed.stdout


def test_native_receipt_is_deterministic_complete_and_accepts_exact_model() -> None:
    from transformers.models.lfm2 import modeling_lfm2

    first = runtime.lfm2_runtime_semantics_receipt()
    second = runtime.lfm2_runtime_semantics_receipt()
    receipt_sha256 = runtime.lfm2_runtime_semantics_receipt_sha256()

    assert first == second
    assert first is not second
    assert canonical_sha256(first) == receipt_sha256
    assert first["schema"] == runtime.LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA
    assert {
        "Lfm2ForCausalLM.forward",
        "Lfm2Model.forward",
        "Lfm2MLP.forward",
        "Lfm2Attention.forward",
        "Lfm2ShortConv.forward",
    }.issubset(first["native_lfm2_methods"])
    assert {"rotate_half", "repeat_kv"}.issubset(first["native_lfm2_functions"])
    assert set(first["attention_registry"]["ordered_keys"]) == set(
        modeling_lfm2.ALL_ATTENTION_FUNCTIONS.keys()
    )
    assert len(runtime.lfm2_runtime_semantics_receipt_source_sha256()) == 64
    assert (
        runtime.verify_lfm2_capture_model_runtime(
            _parent(),
            expected_sha256=receipt_sha256,
        )
        == receipt_sha256
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "rotate_half",
        "repeat_kv",
        "native_class_module",
        "torch_class_module",
        "dynamic_cache_export",
        "dynamic_cache_method",
        "attention_local_mapping",
    ),
)
def test_live_receipt_rejects_transitive_class_and_registry_tamper(
    monkeypatch,
    tamper,
) -> None:
    from transformers.models.lfm2 import modeling_lfm2

    runtime.lfm2_runtime_semantics_receipt_sha256()

    def poisoned(*_args, **_kwargs):
        return torch.zeros(1)

    if tamper in {"rotate_half", "repeat_kv"}:
        monkeypatch.setattr(modeling_lfm2, tamper, poisoned)
    elif tamper == "native_class_module":
        monkeypatch.setattr(
            modeling_lfm2.Lfm2MLP,
            "__module__",
            "foreign.runtime",
        )
    elif tamper == "torch_class_module":
        monkeypatch.setattr(torch.nn.Linear, "__module__", "foreign.runtime")
    elif tamper == "dynamic_cache_export":
        original = modeling_lfm2.DynamicCache
        poisoned = type(
            original.__name__,
            (original,),
            {
                "__module__": original.__module__,
                "__qualname__": original.__qualname__,
            },
        )
        monkeypatch.setattr(modeling_lfm2, "DynamicCache", poisoned)
    elif tamper == "dynamic_cache_method":
        from transformers import cache_utils

        monkeypatch.setattr(
            cache_utils.Cache,
            "get_seq_length",
            lambda _self, *_args, **_kwargs: 0,
        )
    else:
        registry = modeling_lfm2.ALL_ATTENTION_FUNCTIONS
        monkeypatch.setitem(
            vars(registry)["_local_mapping"],
            "neural_forge_foreign",
            poisoned,
        )

    with pytest.raises(runtime.Lfm2RuntimeSemanticsReceiptError):
        runtime.verify_lfm2_runtime_semantics_receipt()


@pytest.mark.parametrize(
    "tamper",
    ("root_forward", "internal_forward", "internal_hook", "compiled_call"),
)
def test_exact_capture_model_walk_rejects_instance_execution_tamper(
    monkeypatch,
    tamper,
) -> None:
    model = _parent(seed=7302)
    if tamper == "root_forward":
        monkeypatch.setattr(
            model,
            "forward",
            types.MethodType(lambda _self, *_args, **_kwargs: torch.zeros(1), model),
        )
    else:
        norm = model.model.layers[0].ffn_norm
        if tamper == "internal_forward":
            monkeypatch.setattr(
                norm,
                "forward",
                types.MethodType(
                    lambda _self, value: torch.zeros_like(value),
                    norm,
                ),
            )
        elif tamper == "internal_hook":
            norm.register_forward_hook(
                lambda _module, _inputs, output: torch.zeros_like(output)
            )
        else:
            monkeypatch.setattr(
                norm,
                "_compiled_call_impl",
                lambda *_args, **_kwargs: torch.zeros(1),
            )

    with pytest.raises(runtime.Lfm2RuntimeSemanticsReceiptError):
        runtime.verify_lfm2_capture_model_runtime(model)


def test_fresh_source_alias_is_removed_on_exec_failure_and_collision(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "broken_runtime.py"
    source.write_text(
        "raise RuntimeError('forced fresh exec failure')\n", encoding="utf-8"
    )
    live = types.ModuleType("test_broken_runtime")
    live.__file__ = str(source)
    alias = "test_runtime_semantics_receipt_fresh_alias"
    assert alias not in sys.modules

    with pytest.raises(
        runtime.Lfm2RuntimeSemanticsReceiptError,
        match="independent source authentication failed",
    ):
        runtime.load_fresh_source_module(live, alias, "broken runtime")
    assert alias not in sys.modules

    sentinel = types.ModuleType(alias)
    monkeypatch.setitem(sys.modules, alias, sentinel)
    with pytest.raises(
        runtime.Lfm2RuntimeSemanticsReceiptError,
        match="cannot be independently loaded",
    ):
        runtime.load_fresh_source_module(live, alias, "colliding runtime")
    assert sys.modules[alias] is sentinel


def test_fresh_source_failure_rolls_back_dependencies_and_attention_registry(
    tmp_path,
) -> None:
    from transformers import modeling_utils

    registry = modeling_utils.ALL_ATTENTION_FUNCTIONS
    before_observable = dict(registry.items())
    before_local = dict(vars(registry)["_local_mapping"])
    dependency = "neural_forge_test_unexpected_fresh_dependency"
    alias = "test_runtime_semantics_receipt_side_effect_alias"
    assert dependency not in sys.modules
    assert alias not in sys.modules
    source = tmp_path / "side_effect_runtime.py"
    source.write_text(
        "\n".join(
            (
                "import sys",
                "import types",
                "from transformers import modeling_utils",
                f"sys.modules[{dependency!r}] = types.ModuleType({dependency!r})",
                "modeling_utils.ALL_ATTENTION_FUNCTIONS.register(",
                "    'neural_forge_forced_side_effect', lambda *args: None",
                ")",
                "raise RuntimeError('forced side-effect failure')",
                "",
            )
        ),
        encoding="utf-8",
    )
    live = types.ModuleType("test_side_effect_runtime")
    live.__file__ = str(source)

    with pytest.raises(
        runtime.Lfm2RuntimeSemanticsReceiptError,
        match="independent source authentication failed",
    ):
        runtime.load_fresh_source_module(live, alias, "side-effect runtime")

    assert alias not in sys.modules
    assert dependency not in sys.modules
    assert vars(registry)["_local_mapping"] == before_local
    after_observable = dict(registry.items())
    assert set(after_observable) == set(before_observable)
    assert all(
        after_observable[name] is before_observable[name] for name in before_observable
    )
