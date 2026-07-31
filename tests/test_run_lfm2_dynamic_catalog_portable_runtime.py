from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from test_run_lfm2_dynamic_catalog import MODULE, _required_args


def _canonical_config(tmp_path: Path):
    return MODULE.ProductionLfm2CatalogConfig(
        parent_model=tmp_path / "parent",
        calibration=tmp_path / "calibration.json",
        cached_parent_outputs=tmp_path / "parent-outputs.json",
        cached_parent_outputs_sha256="d" * 64,
        cache_root=tmp_path / "cache",
        snapshot_root=tmp_path / "snapshots",
        publish_destination=tmp_path / "published",
        ledger_path=tmp_path / "ledger.jsonl",
        receipt_path=tmp_path / "receipt.json",
        compression_ratio=4.0,
        dense_heterogeneous_layout=True,
        device="cpu",
    )


def test_portable_request_is_excluded_from_canonical_fold_identity(
    tmp_path: Path,
) -> None:
    canonical = _canonical_config(tmp_path)
    portable = replace(
        canonical,
        portable_runtime_device="cpu",
        portable_runtime_dtype="bf16",
    )

    assert canonical.dtype == "float32"
    assert portable.dtype == "float32"
    assert canonical.uses_portable_runtime_relocation is False
    assert portable.uses_portable_runtime_relocation is True
    assert portable.identity() == canonical.identity()
    assert MODULE.canonical_sha256(portable.identity()) == (
        MODULE.canonical_sha256(canonical.identity())
    )
    assert "portable_runtime_device" not in str(portable.identity())
    assert "portable_runtime_dtype" not in str(portable.identity())


@pytest.mark.parametrize(
    "dtype_name",
    ("float16", "fp16", "bfloat16", "bf16", "float32", "fp32", "float64", "fp64"),
)
def test_portable_runtime_dtype_is_explicit_and_complete(
    dtype_name: str,
) -> None:
    assert MODULE._portable_runtime_dtype(dtype_name) in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }


def test_portable_runtime_config_rejects_partial_or_noncanonical_requests(
    tmp_path: Path,
) -> None:
    canonical = _canonical_config(tmp_path)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="requires both",
    ):
        replace(canonical, portable_runtime_device="cpu")
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="FP16, BF16, FP32, or FP64",
    ):
        replace(
            canonical,
            portable_runtime_device="cpu",
            portable_runtime_dtype="int8",
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="canonical FP32 heterogeneous",
    ):
        MODULE.ProductionLfm2CatalogConfig(
            parent_model=tmp_path / "parent",
            calibration=tmp_path / "calibration.json",
            cached_parent_outputs=tmp_path / "parent-outputs.json",
            cached_parent_outputs_sha256="d" * 64,
            cache_root=tmp_path / "ordinary-cache",
            snapshot_root=tmp_path / "ordinary-snapshots",
            publish_destination=tmp_path / "ordinary-published",
            ledger_path=tmp_path / "ordinary-ledger.jsonl",
            receipt_path=tmp_path / "ordinary-receipt.json",
            portable_runtime_device="cpu",
            portable_runtime_dtype="float32",
        )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="explicit accelerator index",
    ):
        replace(
            canonical,
            portable_runtime_device="cuda",
            portable_runtime_dtype="float32",
        )


def test_parser_exposes_opt_in_portable_runtime_without_changing_fold_dtype(
    tmp_path: Path,
) -> None:
    parser = MODULE.build_parser()
    args = parser.parse_args(
        _required_args(tmp_path)
        + [
            "--compression-ratio",
            "4",
            "--dense-heterogeneous-layout",
            "--portable-runtime-device",
            "cpu",
            "--portable-runtime-dtype",
            "float64",
        ]
    )
    config = MODULE.config_from_args(args)

    assert config.dtype == "float32"
    assert config.portable_runtime_device == "cpu"
    assert config.portable_runtime_dtype == "float64"
    assert config.uses_portable_runtime_relocation is True


def _tiny_lfm(*, tie_word_embeddings: bool = False) -> object:
    config = transformers.Lfm2Config(
        vocab_size=7,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=tie_word_embeddings,
    )
    config._attn_implementation = "eager"
    return transformers.Lfm2ForCausalLM(config).eval()


@pytest.mark.parametrize(
    "runtime_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_runner_direct_local_portable_method_executes_real_lfm_forward(
    tmp_path: Path,
    monkeypatch,
    runtime_dtype: torch.dtype,
) -> None:
    child = _tiny_lfm()
    canonical_count = sum(parameter.numel() for parameter in child.parameters())
    artifact_io = MODULE.DirectLocalArtifactIO(
        tokenizer_directory=tmp_path,
        tokenizer_fingerprint="a" * 64,
        device="cpu",
        dtype=torch.float32,
    )
    load_calls = []

    def load(_source: Path):
        load_calls.append(True)
        return child

    monkeypatch.setattr(artifact_io, "load", load)
    relocated, receipt = (
        artifact_io.load_authenticated_canonical_fp32_portable(
            tmp_path,
            runtime_device="cpu",
            runtime_dtype=runtime_dtype,
        )
    )
    body = receipt.as_dict()

    assert relocated is not child
    assert load_calls == [True]
    assert body["canonical_unique_parameter_elements"] == canonical_count
    assert body["runtime_unique_parameter_elements"] == canonical_count
    assert body["exact_parameter_count_invariant"] is True
    assert body["deterministic_post_cast_state_match"] is True
    assert body["state_names_shapes_and_alias_abi_preserved"] is True
    assert body["executable_forward_smoke"]["logits_finite"] is True
    assert body["executable_forward_smoke"]["torch_no_grad"] is True
    assert body["executable_forward_smoke"]["use_cache"] is False
    assert body["runtime_dtype_parameter_compaction_credit"] == 0
    assert body["runtime_device_parameter_compaction_credit"] == 0
    assert {
        value.dtype
        for value in relocated.state_dict().values()
        if value.is_floating_point()
    } == {runtime_dtype}
    assert {
        value.dtype
        for value in child.state_dict().values()
        if value.is_floating_point()
    } == {torch.float32}


def test_runner_portable_load_stages_canonical_model_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _tiny_lfm()
    observed_load_policies = []

    def load(self, _source: Path):
        observed_load_policies.append((self.device, self.dtype))
        return child

    monkeypatch.setattr(MODULE.DirectLocalArtifactIO, "load", load)
    artifact_io = MODULE.DirectLocalArtifactIO(
        tokenizer_directory=tmp_path,
        tokenizer_fingerprint="a" * 64,
        device="cuda:7",
        dtype=torch.float32,
    )

    relocated, receipt = (
        artifact_io.load_authenticated_canonical_fp32_portable(
            tmp_path,
            runtime_device="cpu",
            runtime_dtype=torch.float32,
        )
    )

    assert relocated is not child
    assert observed_load_policies == [("cpu", torch.float32)]
    assert receipt.as_dict()["canonical_device"] == "cpu"


def test_transactional_relocation_preserves_tied_alias_abi() -> None:
    child = _tiny_lfm(tie_word_embeddings=True)
    assert child.get_input_embeddings().weight is (
        child.get_output_embeddings().weight
    )

    relocated, receipt = (
        MODULE.relocate_authenticated_lfm2_canonical_fp32_model(
            child,
            provenance_resource_fingerprint="tied-real-lfm",
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            fresh_loaded_runtime_copy=True,
        )
    )

    assert relocated is not child
    assert relocated.get_input_embeddings().weight is (
        relocated.get_output_embeddings().weight
    )
    assert receipt.as_dict()[
        "state_names_shapes_and_alias_abi_preserved"
    ] is True
    assert child.get_input_embeddings().weight.dtype is torch.float32


def test_forward_failure_issues_no_runner_success_event_or_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = _tiny_lfm()

    def failing_forward(**_kwargs):
        raise RuntimeError("simulated post-storage LFM forward failure")

    monkeypatch.setattr(child, "forward", failing_forward)
    artifact_io = MODULE.DirectLocalArtifactIO(
        tokenizer_directory=tmp_path,
        tokenizer_fingerprint="a" * 64,
        device="cpu",
        dtype=torch.float32,
    )
    monkeypatch.setattr(artifact_io, "load", lambda _source: child)
    stream = io.StringIO()
    progress = MODULE._ProgressJournal("b" * 64, stream=stream)

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="could not be relocated",
    ):
        MODULE._load_verified_portable_direct_runtime(
            artifact_io,
            tmp_path,
            runtime_device="cpu",
            runtime_dtype=torch.float16,
            expected_unique_parameters=sum(
                parameter.numel() for parameter in child.parameters()
            ),
            progress=progress,
        )

    assert "portable_runtime_relocation_completed" not in stream.getvalue()
    assert "portable_runtime_relocation" not in stream.getvalue()
    assert {
        value.dtype
        for value in child.state_dict().values()
        if value.is_floating_point()
    } == {torch.float32}


def test_portable_carrier_smoke_executes_one_step_and_keys_review_identity(
    monkeypatch,
) -> None:
    observed = []

    def fake_smoke(model, prompts, **kwargs):
        observed.append((model, prompts, dict(kwargs)))
        return {
            "operation_audit": {"child_body_forward_calls": 1},
            "routing_session_receipt": {
                "input": {"routing_cache_use_count": 1},
                "output": {"routing_cache_use_count": 1},
            },
        }

    monkeypatch.setattr(MODULE, "run_lfm2_carrier_route_smoke", fake_smoke)
    model = _tiny_lfm()
    runtime_state_sha256 = (
        MODULE.fingerprint_lfm2_direct_local_runtime_tensor_state(model)[
            "state_sha256"
        ]
    )
    route_state = {
        "state_sha256": "7" * 64,
        "continuous_coefficients_remain_fp32": True,
    }
    receipt = {
        "direct_local_relocation": {
            "runtime_device": "cpu",
            "runtime_dtype": "torch.float32",
            "runtime_state_sha256": runtime_state_sha256,
        },
        "carrier_runtime_sha256": "2" * 64,
    }
    prompts = ({"id": "smoke", "prompt": "hello"},)

    smoke, first_identity = (
        MODULE._run_verified_portable_carrier_execution_smoke(
            model,
            prompts,
            maximum_prompt_tokens=128,
            portable_runtime_receipt=receipt,
            canonical_child_artifact_sha256="3" * 64,
            route_state_observer=lambda _model: route_state,
        )
    )
    changed_model = _tiny_lfm().to(dtype=torch.float64)
    changed_state_sha256 = (
        MODULE.fingerprint_lfm2_direct_local_runtime_tensor_state(
            changed_model
        )["state_sha256"]
    )
    changed_receipt = {
        **receipt,
        "direct_local_relocation": {
            **receipt["direct_local_relocation"],
            "runtime_dtype": "torch.float64",
            "runtime_state_sha256": changed_state_sha256,
        },
    }
    _changed_smoke, second_identity = (
        MODULE._run_verified_portable_carrier_execution_smoke(
            changed_model,
            prompts,
            maximum_prompt_tokens=128,
            portable_runtime_receipt=changed_receipt,
            canonical_child_artifact_sha256="3" * 64,
            route_state_observer=lambda _model: route_state,
        )
    )

    assert observed[0][2] == {
        "maximum_prompt_tokens": 128,
        "maximum_new_tokens": 1,
    }
    assert smoke["body_input_and_output_routes_executed"] is True
    assert smoke["child_body_forward_calls"] == 1
    assert smoke["input_route_use_count"] == 1
    assert smoke["output_route_use_count"] == 1
    assert smoke["model_parameters_and_all_buffers_unchanged"] is True
    assert smoke["owned_route_tensors_unchanged"] is True
    assert first_identity != second_identity


@pytest.mark.parametrize("mutation", ("model", "route"))
def test_portable_carrier_smoke_rejects_state_mutation(
    monkeypatch,
    mutation: str,
) -> None:
    model = _tiny_lfm()
    route_tensor = torch.tensor([1.0], dtype=torch.float32)

    def observe_route(_model):
        return {
            "state_sha256": MODULE.canonical_sha256(
                {"values": route_tensor.tolist()}
            ),
            "continuous_coefficients_remain_fp32": True,
        }

    def fake_smoke(live_model, _prompts, **_kwargs):
        with torch.no_grad():
            if mutation == "model":
                next(live_model.parameters()).add_(1.0)
            else:
                route_tensor.add_(1.0)
        return {
            "operation_audit": {"child_body_forward_calls": 1},
            "routing_session_receipt": {
                "input": {"routing_cache_use_count": 1},
                "output": {"routing_cache_use_count": 1},
            },
        }

    monkeypatch.setattr(MODULE, "run_lfm2_carrier_route_smoke", fake_smoke)
    runtime_state_sha256 = (
        MODULE.fingerprint_lfm2_direct_local_runtime_tensor_state(model)[
            "state_sha256"
        ]
    )
    receipt = {
        "direct_local_relocation": {
            "runtime_device": "cpu",
            "runtime_dtype": "torch.float32",
            "runtime_state_sha256": runtime_state_sha256,
        },
        "carrier_runtime_sha256": "2" * 64,
    }

    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="mutated full model or fixed owned route tensor state",
    ):
        MODULE._run_verified_portable_carrier_execution_smoke(
            model,
            ({"id": "smoke", "prompt": "hello"},),
            maximum_prompt_tokens=128,
            portable_runtime_receipt=receipt,
            canonical_child_artifact_sha256="3" * 64,
            route_state_observer=observe_route,
        )


def test_portable_manual_checkpoints_use_runtime_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = {}

    def fake_review(*_args, **kwargs):
        observed.update(kwargs)
        return {
            "responses": [
                {
                    "id": "x",
                    "category": "reasoning",
                    "text": "coherent",
                    "diagnostics": {"censored_by_output_cap": False},
                }
            ],
            "initial_maximum_new_tokens": 1_024,
            "retry_maximum_new_tokens": 2_048,
            "initial_cap_censored_response_count": 0,
            "initial_cap_censored_response_ids": [],
            "adaptive_retry_response_count": 0,
            "adaptive_retry_response_ids": [],
            "remaining_cap_censored_response_count": 0,
            "remaining_cap_censored_response_ids": [],
            "review_sha256": "5" * 64,
            "published_child_artifact_sha256": kwargs[
                "published_child_artifact_sha256"
            ],
            "checkpoint_row_receipts": [],
            "checkpoint_row_count": 0,
            "checkpoint_rows_reused": 0,
            "checkpoint_rows_executed": 1,
            "runtime_sha256s": ["6" * 64],
        }

    monkeypatch.setattr(
        MODULE,
        "run_lfm2_resumable_adaptive_carrier_manual_review",
        fake_review,
    )
    canonical_artifact = "3" * 64
    runtime_identity = "4" * 64
    dossier = MODULE._write_routed_manual_dossier(
        model=object(),
        prompts=({"id": "x", "category": "reasoning", "prompt": "x"},),
        config=SimpleNamespace(
            routed_manual_dossier_path=tmp_path / "dossier.json",
            routed_manual_checkpoint_directory=tmp_path / "checkpoints",
            routed_manual_maximum_prompt_tokens=8_192,
            routed_manual_initial_max_new_tokens=1_024,
            routed_manual_retry_max_new_tokens=2_048,
        ),
        counted_verification={"verification_sha256": "7" * 64},
        outer_inventory={"inventory_sha256": "8" * 64},
        route_package=SimpleNamespace(package_sha256="9" * 64),
        lexicon_package=SimpleNamespace(package_sha256="a" * 64),
        published_child_artifact_sha256=canonical_artifact,
        portable_runtime_review_identity_sha256=runtime_identity,
    )

    expected_namespace = (
        tmp_path / "checkpoints" / f"portable-runtime-{runtime_identity}"
    )
    assert observed["published_child_artifact_sha256"] == canonical_artifact
    assert observed["checkpoint_directory"] == expected_namespace
    assert dossier["published_child_artifact_sha256"] == canonical_artifact
    assert dossier["canonical_published_child_artifact_sha256"] == (
        canonical_artifact
    )
    assert dossier["portable_runtime_review_identity_sha256"] == (
        runtime_identity
    )
    assert dossier[
        "review_checkpoints_keyed_by_runtime_device_dtype_and_state"
    ] is True
    assert dossier["portable_runtime_checkpoint_namespace"] == str(
        expected_namespace
    )
