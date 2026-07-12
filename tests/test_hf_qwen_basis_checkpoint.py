from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_qwen_basis_checkpoint as checkpoint_module  # noqa: E402
from neural_forge.hf_depth_basis_ir import (  # noqa: E402
    PROJECTION_ROLES,
    FixedLayerDepthBasisLinear,
)
from neural_forge.hf_factorized_vocab import (  # noqa: E402
    FactorizedEmbedding,
    FactorizedLMHead,
)
from neural_forge.hf_qwen_basis_checkpoint import (  # noqa: E402
    MANIFEST_FILENAME,
    Qwen3BasisBudgetError,
    Qwen3BasisCheckpointError,
    Qwen3BasisFormatError,
    export_qwen3_basis_checkpoint,
    load_qwen3_basis_checkpoint,
    plan_qwen3_basis_checkpoint,
)


def _tiny_qwen3():
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(2401)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=31,
            hidden_size=12,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )
    return model.eval()


def _rank_ones() -> dict[str, int]:
    return dict.fromkeys(PROJECTION_ROLES, 1)


def _logical_parameter_count(model) -> int:
    return sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def test_qwen3_4b_reference_plan_hits_exact_cap_with_explicit_ranks() -> None:
    config = {
        "model_type": "qwen3",
        "tie_word_embeddings": True,
        "vocab_size": 151_936,
        "hidden_size": 2_560,
        "intermediate_size": 9_728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "attention_bias": False,
    }
    role_ranks = {
        "q": 1,
        "k": 6,
        "v": 6,
        "o": 1,
        "gate": 1,
        "up": 1,
        "down": 1,
    }

    plan = plan_qwen3_basis_checkpoint(
        config,
        role_ranks=role_ranks,
        parameter_budget=134_443_364,
    )

    assert plan.ranks.role_ranks == role_ranks
    assert plan.ranks.vocab == 46
    assert plan.fixed_parameter_count == 196_096
    assert plan.vocabulary_unique_parameter_count == 7_106_816
    assert sum(plan.role_counts.values()) == 127_140_452
    assert plan.unique_parameter_count == 134_443_364
    assert plan.logical_parameter_count == 141_550_180
    assert plan.budget_headroom == 0

    below = plan_qwen3_basis_checkpoint(
        config,
        role_ranks=role_ranks,
        parameter_budget=134_443_363,
    )
    assert below.ranks.vocab == 45
    assert below.unique_parameter_count <= 134_443_363
    with pytest.raises(Qwen3BasisBudgetError, match="exceeds"):
        plan_qwen3_basis_checkpoint(
            config,
            rank_plan=plan.ranks,
            parameter_budget=134_443_363,
        )


def test_plan_rejects_partial_or_untied_rank_contracts() -> None:
    config = _tiny_qwen3().config

    with pytest.raises(Qwen3BasisCheckpointError, match="exactly"):
        plan_qwen3_basis_checkpoint(
            config,
            role_ranks={"q": 1},
            vocab_rank=1,
        )
    config.tie_word_embeddings = False
    with pytest.raises(Qwen3BasisCheckpointError, match="tie_word_embeddings"):
        plan_qwen3_basis_checkpoint(
            config,
            role_ranks=_rank_ones(),
            vocab_rank=1,
        )

    biased = _tiny_qwen3().config
    biased.attention_bias = True
    with pytest.raises(Qwen3BasisCheckpointError, match="attention_bias=False"):
        plan_qwen3_basis_checkpoint(
            biased,
            role_ranks=_rank_ones(),
            vocab_rank=1,
        )


def test_atomic_safetensors_export_and_standalone_exact_reload(tmp_path) -> None:
    model = _tiny_qwen3()
    checkpoint = tmp_path / "basis-checkpoint"

    manifest = export_qwen3_basis_checkpoint(
        model,
        checkpoint,
        role_ranks=_rank_ones(),
        vocab_rank=3,
        feature_chunk_size=17,
        full_vocab_svd_max_elements=1,
        vocab_row_chunk_size=7,
        vocab_oversampling=2,
        vocab_power_iterations=1,
        vocab_seed=99,
    )
    input_ids = torch.tensor([[1, 4, 7]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = model(input_ids).logits
        expected_tokens = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
        )

    assert checkpoint.is_dir()
    assert {path.name for path in checkpoint.iterdir()} == {
        "config.json",
        "generation_config.json",
        MANIFEST_FILENAME,
        "model.safetensors",
    }
    assert manifest["serialization"] == {
        "python_pickle": False,
        "serialized_modules": False,
        "serialized_weakrefs": False,
        "tensor_format": "safetensors",
        "byte_order": "little",
    }
    assert manifest["standalone"] is True
    assert manifest["external_weight_dependencies"] == []
    assert manifest["behavioral_validation"]["status"] == "not_evaluated"
    assert manifest["behavioral_validation"]["structural_checks_only"] is True
    assert set(manifest["depth_factorization"]) == set(PROJECTION_ROLES)
    assert all(
        report["quality_metrics_available"] is True
        for report in manifest["depth_factorization"].values()
    )
    vocab_report = manifest["vocabulary_factorization"]
    assert vocab_report["algorithm"] == (
        "deterministic-chunked-randomized-svd-v1"
    )
    assert vocab_report["materialized_full_float64_source_matrix"] is False
    assert vocab_report["materialized_full_left_singular_matrix"] is False
    assert vocab_report["row_chunk_size"] == 7
    assert vocab_report["seed"] == 99
    assert manifest["state_dict"]["alias_count"] == 2
    aliases = manifest["state_dict"]["key_to_tensor"]
    assert aliases["model.embed_tokens.vocab_factor"] == aliases[
        "lm_head.vocab_factor"
    ]
    assert aliases["model.embed_tokens.hidden_factor"] == aliases[
        "lm_head.hidden_factor"
    ]

    config_artifact = manifest["artifacts"]["config"]
    config_bytes = (checkpoint / config_artifact["file"]).read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == config_artifact["sha256"]
    hash_payload = dict(manifest)
    manifest_hash = hash_payload.pop("manifest_sha256")
    assert hashlib.sha256(_canonical_json_bytes(hash_payload)).hexdigest() == (
        manifest_hash
    )

    safetensors = pytest.importorskip("safetensors")
    del safetensors
    from safetensors import safe_open

    tensor_path = checkpoint / manifest["artifacts"]["tensors"]["file"]
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        stored_keys = set(handle.keys())
    assert stored_keys == set(manifest["tensors"])
    assert len(stored_keys) == manifest["state_dict"]["stored_tensor_count"]
    assert len(aliases) == manifest["state_dict"]["logical_tensor_count"]

    loaded = load_qwen3_basis_checkpoint(checkpoint)
    with torch.no_grad():
        actual_logits = loaded(input_ids).logits
        actual_tokens = loaded.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
        )

    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_tokens, expected_tokens)
    assert loaded.neural_forge_load_report["missing_keys"] == []
    assert loaded.neural_forge_load_report["unexpected_keys"] == []
    assert isinstance(loaded.get_input_embeddings(), FactorizedEmbedding)
    assert isinstance(loaded.get_output_embeddings(), FactorizedLMHead)
    assert (
        loaded.get_input_embeddings().vocab_factor
        is loaded.get_output_embeddings().vocab_factor
    )
    assert (
        loaded.get_input_embeddings().hidden_factor
        is loaded.get_output_embeddings().hidden_factor
    )
    assert all(
        isinstance(layer.self_attn.q_proj, FixedLayerDepthBasisLinear)
        for layer in loaded.model.layers
    )
    assert not any(
        marker in name.lower()
        for name, _ in loaded.named_modules()
        for marker in ("adapter", "teacher", "parent")
    )
    counts = manifest["parameter_counts"]
    assert sum(parameter.numel() for parameter in loaded.parameters()) == counts[
        "unique_parameter_count"
    ]
    assert _logical_parameter_count(loaded) == counts["logical_parameter_count"]


def test_chunked_randomized_vocabulary_factors_are_bitwise_deterministic() -> None:
    weight = _tiny_qwen3().get_input_embeddings().weight

    first_vocab, first_hidden, first_report = (
        checkpoint_module._chunked_randomized_vocabulary_factors(
            weight,
            rank=3,
            row_chunk_size=5,
            oversampling=2,
            power_iterations=1,
            seed=401,
        )
    )
    second_vocab, second_hidden, second_report = (
        checkpoint_module._chunked_randomized_vocabulary_factors(
            weight,
            rank=3,
            row_chunk_size=5,
            oversampling=2,
            power_iterations=1,
            seed=401,
        )
    )

    assert torch.equal(first_vocab, second_vocab)
    assert torch.equal(first_hidden, second_hidden)
    assert first_report == second_report
    assert first_report["source_passes"] == 4
    assert first_report["tall_workspace_shape"] == [31, 5]
    assert first_report["chunk_workspace_shape"] == [5, 12]
    assert first_report["materialized_full_float64_source_matrix"] is False


def test_large_full_vocab_svd_request_fails_closed(tmp_path) -> None:
    destination = tmp_path / "unsafe-full"
    with pytest.raises(Qwen3BasisCheckpointError, match="Refusing full float64"):
        export_qwen3_basis_checkpoint(
            _tiny_qwen3(),
            destination,
            role_ranks=_rank_ones(),
            vocab_rank=2,
            vocab_algorithm="full",
            full_vocab_svd_max_elements=1,
        )
    assert not destination.exists()


def test_raw_fallback_is_pickle_free_and_exact(tmp_path) -> None:
    model = _tiny_qwen3()
    checkpoint = tmp_path / "raw-checkpoint"
    export_qwen3_basis_checkpoint(
        model,
        checkpoint,
        role_ranks=_rank_ones(),
        vocab_rank=2,
        feature_chunk_size=19,
        tensor_format="raw",
    )
    input_ids = torch.tensor([[1, 6, 8]], dtype=torch.long)
    with torch.no_grad():
        expected = model(input_ids).logits

    assert (checkpoint / "model.nfraw").is_file()
    assert not any(
        path.suffix.lower() in {".bin", ".pkl", ".pickle", ".pt", ".pth"}
        for path in checkpoint.iterdir()
    )
    loaded = load_qwen3_basis_checkpoint(checkpoint)
    with torch.no_grad():
        actual = loaded(input_ids).logits
    assert torch.equal(actual, expected)
    assert loaded.neural_forge_load_report["missing_keys"] == []
    assert loaded.neural_forge_load_report["unexpected_keys"] == []


def test_manifest_corruption_and_existing_paths_fail_closed(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("keep", encoding="ascii")
    with pytest.raises(Qwen3BasisCheckpointError, match="Refusing to replace"):
        export_qwen3_basis_checkpoint(
            _tiny_qwen3(),
            existing,
            role_ranks=_rank_ones(),
            vocab_rank=1,
        )
    assert sentinel.read_text(encoding="ascii") == "keep"

    checkpoint = tmp_path / "corrupt"
    export_qwen3_basis_checkpoint(
        _tiny_qwen3(),
        checkpoint,
        role_ranks=_rank_ones(),
        vocab_rank=1,
        feature_chunk_size=23,
    )
    manifest_path = checkpoint / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["standalone"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Qwen3BasisFormatError, match="Manifest SHA-256 mismatch"):
        load_qwen3_basis_checkpoint(checkpoint)


def test_failed_payload_write_leaves_no_partial_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    model = _tiny_qwen3()
    checkpoint_module.prepare_qwen3_basis_model(
        model,
        role_ranks=_rank_ones(),
        vocab_rank=1,
        feature_chunk_size=29,
    )

    def fail_payload(*_args, **_kwargs):
        raise RuntimeError("injected tensor write failure")

    monkeypatch.setattr(checkpoint_module, "_save_tensor_payload", fail_payload)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected tensor write failure"):
        export_qwen3_basis_checkpoint(model, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))
