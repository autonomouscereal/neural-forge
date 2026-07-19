from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import torch as safetensors_torch

from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner
from neural_forge.dynamic_candidate_capture import (
    DynamicCandidateCaptureIntegrityError,
    fingerprint_quarantine_callable,
    load_quarantined_candidate,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES,
    Lfm2CpcfV5HoldoutEvaluator,
    Lfm2CpcfV5QuarantineConfig,
    Lfm2CpcfV5QuarantineConsumer,
    Lfm2CpcfV5QuarantineError,
    Lfm2CpcfV5QuarantineSaver,
    Lfm2CpcfV5StrictLocalLoader,
    build_lfm2_cpcf_v5_quarantine_runtime,
    capture_lfm2_cpcf_v5_quarantine,
)
from neural_forge.hf_lfm2_intelligence_gate import (
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_parent_envelope,
)
from neural_forge.hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
)


class _TinyConfig:
    model_type = "lfm2"

    def __init__(self, *, hidden_size: int = 3, intermediate_size: int = 2):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_ff_dim = intermediate_size
        self.block_auto_adjust_ff_dim = False
        self._name_or_path = "C:/models/cache/parent/checkpoint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "block_ff_dim": self.block_ff_dim,
            "block_auto_adjust_ff_dim": self.block_auto_adjust_ff_dim,
            "_name_or_path": self._name_or_path,
        }


class _TinyFeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = torch.nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.w2 = torch.nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.w3 = torch.nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(value)) * self.w3(value))


class _TinyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = _TinyFeedForward(hidden_size, intermediate_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.feed_forward(value)


class _TinyCore(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_TinyLayer(hidden_size, intermediate_size)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class _TinyLfm2(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = _TinyConfig()
        self.model = _TinyCore(
            self.config.hidden_size,
            self.config.intermediate_size,
        )
        self.output = torch.nn.Linear(
            self.config.hidden_size,
            4,
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.model(value))

    def generate(self) -> torch.Tensor:
        value = torch.ones((1, 1, self.config.hidden_size))
        logits = self(value)
        # Make the decoded behavior deterministic while still executing the
        # complete candidate root and all child modules.
        return logits.detach()

    def save_pretrained(
        self,
        destination: Path,
        *,
        safe_serialization: bool,
    ) -> None:
        assert safe_serialization is True
        path = Path(destination)
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        safetensors_torch.save_file(
            {name: value.detach().cpu() for name, value in self.state_dict().items()},
            path / "model.safetensors",
        )


class _TinyTokenizer:
    def __init__(self, *, target: bool):
        self._target = bool(target)
        self.eos_token_id = 1
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 3]

    @property
    def _tokens(self) -> tuple[str, ...]:
        return (
            "<bos>",
            "<eos>",
            "target-answer" if self._target else "parent-answer",
            "<unk>",
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self._tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._tokens[token_id]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        special = set(self.all_special_ids)
        return " ".join(
            self._tokens[token_id]
            for token_id in token_ids
            if not skip_special_tokens or token_id not in special
        )

    def save_pretrained(self, destination: Path) -> None:
        path = Path(destination)
        vocabulary = self.get_vocab()
        (path / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {"type": "WordLevel", "vocab": vocabulary},
                    "added_tokens": [],
                    "neural_forge_test_target": self._target,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "bos_token": "<bos>",
                    "bos_token_id": 0,
                    "eos_token": "<eos>",
                    "eos_token_id": 1,
                    "pad_token": "<eos>",
                    "pad_token_id": 1,
                    "unk_token": "<unk>",
                    "unk_token_id": 3,
                    "neural_forge_test_target": self._target,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (path / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "pad_token": "<eos>",
                    "unk_token": "<unk>",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _tiny_model_loader(
    directory: Path,
    *,
    device: str,
    dtype: str,
) -> _TinyLfm2:
    assert device == "cpu"
    assert dtype == "float32"
    config = json.loads((Path(directory) / "config.json").read_text())
    assert config["block_ff_dim"] == config["intermediate_size"] == 2
    model = _TinyLfm2(seed=99)
    state = safetensors_torch.load_file(Path(directory) / "model.safetensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _tiny_tokenizer_loader(directory: Path) -> _TinyTokenizer:
    payload = json.loads((Path(directory) / "tokenizer.json").read_text())
    return _TinyTokenizer(target=bool(payload["neural_forge_test_target"]))


def _tiny_candidate_rows(
    model: _TinyLfm2,
    tokenizer: _TinyTokenizer,
    manifest: Any,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for spec in manifest.cases:
        model.generate()
        output_ids = [2, tokenizer.eos_token_id]
        rows.append(
            {
                "case_id": spec.case_id,
                "input_token_count": 1,
                "output": tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ),
                "output_token_ids": output_ids,
                "generation_limit": spec.generation_limit,
                "topical_relevance": None,
            }
        )
    return tuple(rows)


def _digest(character: str) -> str:
    return character * 64


@pytest.fixture(scope="module")
def parent_envelope():
    manifest = load_lfm2_intelligence_holdout_manifest()
    assert len(manifest.cases) == LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES
    rows = [
        {
            "case_id": spec.case_id,
            "input_token_count": 1,
            "output": "parent-answer",
            "output_token_ids": [2, 1],
        }
        for spec in manifest.cases
    ]
    sealed = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        rows,
        parent_state_fingerprint=_digest("a"),
        parent_model_fingerprint=_digest("2"),
        tokenizer_fingerprint=_digest("c"),
        eos_token_ids=(1,),
    )
    return load_lfm2_intelligence_holdout_parent_envelope(
        sealed,
        expected_envelope_sha256=sealed["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=_digest("a"),
        expected_parent_model_fingerprint=_digest("2"),
        expected_tokenizer_fingerprint=_digest("c"),
    )


def _sealed(value: dict[str, Any], seal: str) -> dict[str, Any]:
    return {**value, seal: canonical_sha256(value)}


def _context(
    target: _TinyLfm2,
    parent: _TinyLfm2,
    tokenizer: _TinyTokenizer,
) -> runner.V5CompiledCandidateConsumerContext:
    plan = _digest("1")
    parent_sha = _digest("2")
    cache = _digest("3")
    bridge = _digest("4")
    corpus = _digest("5")
    selective_source = _digest("6")
    compiler_implementation = _digest("7")
    runner_source = hashlib.sha256(
        Path(runner.__file__).resolve().read_bytes()
    ).hexdigest()
    parent_tokenizer_semantics = _digest("c")
    target_tokenizer_semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
    compiler_source_files = [
        {
            "module": "tests.fake_v5_compiler",
            "filename": "fake_v5_compiler.py",
            "sha256": _digest("8"),
        }
    ]
    blank_body = {"target_plan_fingerprint": plan, "blank": True}
    blank = _sealed(blank_body, "report_sha256")
    operation = {
        name: 0
        for name in (
            "parent_model_forward_calls",
            "target_model_forward_calls",
            "generation_calls",
            "parent_cache_generation_calls",
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
            "fallback_paths",
        )
    }
    compiled_body = {
        "blank_target_report_sha256": blank["report_sha256"],
        "selective_source_sha256": selective_source,
        "compiler_implementation": {
            "implementation_sha256": compiler_implementation,
            "ordered_source_files": compiler_source_files,
        },
        "source_state_preserved": True,
        "operation_audit": operation,
    }
    compiled = _sealed(compiled_body, "result_sha256")
    parent_inventory = dict(independent_parameter_inventory(parent))
    target_inventory = dict(independent_parameter_inventory(target))
    target_state_sha = fingerprint_model_state(target)
    target_runtime_sha = fingerprint_model_runtime_state(target)
    parent_state_sha = fingerprint_model_state(parent)
    parent_runtime_sha = fingerprint_model_runtime_state(parent)
    zero = {
        "schema": runner.COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA,
        **{name: 0 for name in runner._CONSUMER_ZERO_COUNTER_FIELDS},
        "compiler_operation_audit": operation,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "live_parent_calls_used": False,
        "clean": True,
    }
    from neural_forge import dynamic_fold_executor

    guard_identity = runner._fold_only_guard_implementation_identity(
        dynamic_fold_executor
    )
    static_audit_body = {
        "schema": (
            "neural-forge-lfm2-cpcf-v5-static-fold-mutation-source-audit-v1"
        ),
        "compiler_implementation_sha256": compiler_implementation,
        "authenticated_compiler_source_files": compiler_source_files,
        "runner_source_sha256": runner_source,
        "audited_source_count": 2,
        "source_audits": [],
        "forbidden_findings": [],
        "forbidden_finding_count": 0,
        "audit_scope": "test authenticated sources",
        "clean": True,
    }
    static_audit = _sealed(static_audit_body, "report_sha256")
    guard_body = {
        "schema": "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1",
        "guard_implementation": dict(guard_identity),
        "static_pruning_quantization_source_audit": static_audit,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_grad_calls": 0,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parent_module_forward_calls": 0,
        "parent_root_forward_calls": 0,
        "parent_inference_entrypoint_calls": 0,
        "target_module_forward_calls": 2,
        "target_root_forward_calls": 0,
        "target_inference_entrypoint_calls": 0,
        "training_steps": 0,
        "training_steps_runtime_observed_directly": False,
        "training_steps_zero_derivation": (
            "zero optimizer/backward/autograd API attempts under no-grad"
        ),
        "pruning_observation_scope": (
            "common-public-pytorch-pruning-mutation-entrypoints"
        ),
        "quantization_observation_scope": (
            "common-public-pytorch-quantization-mutation-entrypoints"
        ),
        "ordinary_bfloat16_and_float32_casts_blocked": False,
        "clean": True,
    }
    guard_report = _sealed(guard_body, "report_sha256")
    zero["compiler_runtime_guard"] = guard_report
    source_binding = canonical_sha256(
        {
            "selective_source_sha256": selective_source,
            "compiler_implementation_sha256": compiler_implementation,
            "runner_source_sha256": runner_source,
            "fold_only_runtime_guard_identity_sha256": guard_identity[
                "identity_sha256"
            ],
        }
    )
    seam = {
        "target_plan_fingerprint": plan,
        "blank_target_report_sha256": blank["report_sha256"],
        "compiler_receipt_sha256": compiled["result_sha256"],
        "parent_sha256": parent_sha,
        "cache_sha256": cache,
        "bridge_sha256": bridge,
        "corpus_sha256": corpus,
        "source_sha256": selective_source,
        "compiler_sha256": compiler_implementation,
        "runner_source_sha256": runner_source,
        "source_binding_sha256": source_binding,
        "fold_only_runtime_guard_identity_sha256": guard_identity[
            "identity_sha256"
        ],
        "parent_tokenizer_semantics_sha256": parent_tokenizer_semantics,
        "target_tokenizer_semantics_sha256": target_tokenizer_semantics,
        "target_tokenizer_package_sha256": None,
        "target_state_sha256": target_state_sha,
        "target_runtime_sha256": target_runtime_sha,
        "parent_state_sha256": parent_state_sha,
        "parent_runtime_sha256": parent_runtime_sha,
        "zero_operation_audit": zero,
    }
    diagnostic_body = {
        "inputs": {
            "parent_fingerprint": parent_sha,
            "cache_key": cache,
            "bridge_artifact_sha256": bridge,
            "trajectory_corpus_sha256": corpus,
            "donor_tokenizer_semantics_sha256": (
                parent_tokenizer_semantics
            ),
            "target_tokenizer_semantics_sha256": (
                target_tokenizer_semantics
            ),
        },
        "blank_target": blank,
        "compiled_model_fold": compiled,
        "compiler_runtime_fold_only_guard": guard_report,
        "parent_parameter_inventory": parent_inventory,
        "blank_parameter_inventory": target_inventory,
        "compiled_parameter_inventory": target_inventory,
        "claims": {
            "child_checkpoint_saved": False,
            "publication_attempted": False,
        },
        "compiled_candidate_consumer_seam": seam,
    }
    diagnostic = _sealed(diagnostic_body, "receipt_sha256")
    return runner.V5CompiledCandidateConsumerContext(
        target_model=target,
        target_tokenizer=tokenizer,
        forbidden_parent_model=parent,
        target_plan_fingerprint=plan,
        blank_target_report=blank,
        compiled_model_fold_report=compiled,
        compiler_receipt_sha256=compiled["result_sha256"],
        compile_diagnostic=diagnostic,
        parent_sha256=parent_sha,
        cache_sha256=cache,
        bridge_sha256=bridge,
        corpus_sha256=corpus,
        source_sha256=selective_source,
        compiler_sha256=compiler_implementation,
        runner_source_sha256=runner_source,
        source_binding_sha256=source_binding,
        parent_tokenizer_semantics_sha256=parent_tokenizer_semantics,
        target_tokenizer_semantics_sha256=target_tokenizer_semantics,
        target_tokenizer_package_sha256=None,
        parent_parameter_inventory=parent_inventory,
        blank_parameter_inventory=target_inventory,
        compiled_parameter_inventory=target_inventory,
        zero_operation_audit=zero,
        target_state_sha256=target_state_sha,
        target_runtime_sha256=target_runtime_sha,
        parent_state_sha256=parent_state_sha,
        parent_runtime_sha256=parent_runtime_sha,
    )


def _config(
    destination: Path,
    envelope: Any,
    tokenizer: _TinyTokenizer,
) -> Lfm2CpcfV5QuarantineConfig:
    return Lfm2CpcfV5QuarantineConfig(
        destination=destination,
        parent_envelope=envelope,
        target_tokenizer_semantics_sha256=(
            fingerprint_hf_tokenizer_semantics(tokenizer)
        ),
        device="cpu",
        dtype="float32",
        model_loader=_tiny_model_loader,
        tokenizer_loader=_tiny_tokenizer_loader,
        candidate_generator=_tiny_candidate_rows,
    )


def test_authenticated_callbacks_do_not_hash_live_tokenizer_or_model(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=10).eval()
    parent = _TinyLfm2(seed=11).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    runtime = build_lfm2_cpcf_v5_quarantine_runtime(
        context,
        _config(tmp_path / "candidate", parent_envelope, tokenizer),
    )

    assert runtime.request.compiler_sha256 == context.compiler_receipt_sha256
    assert runtime.request.source_sha256 == context.source_binding_sha256
    assert runtime.request.expected_model_state_sha256 == (
        fingerprint_model_state(target)
    )
    assert runtime.request.prior_audit.as_dict()["clean"] is True
    for role, callback in (
        ("saver", runtime.saver),
        ("loader", runtime.loader),
        ("evaluator", runtime.evaluator),
    ):
        assert callback.resource_fingerprint
        assert fingerprint_quarantine_callable(callback, role=role) == getattr(
            runtime.request,
            f"{role}_sha256",
        )


def test_saver_loader_and_evaluator_preserve_target_tokenizer_semantics(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=20).eval()
    delattr(target.config, "block_ff_dim")
    original_source_path = target.config._name_or_path
    tokenizer = _TinyTokenizer(target=True)
    state_sha = fingerprint_model_state(target)
    inventory = dict(independent_parameter_inventory(target))
    context_sha = _digest("d")
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=tokenizer,
        context_binding_sha256=context_sha,
        expected_ffn_width=2,
        expected_model_state_sha256=state_sha,
        expected_parameter_inventory_sha256=inventory["inventory_sha256"],
        expected_tokenizer_semantics_sha256=(
            fingerprint_hf_tokenizer_semantics(tokenizer)
        ),
        tokenizer_loader=_tiny_tokenizer_loader,
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    saver(target, artifact)
    assert not hasattr(target.config, "block_ff_dim")
    assert target.config._name_or_path == original_source_path
    assert json.loads((artifact / "config.json").read_text())[
        "_name_or_path"
    ] == ""
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context_sha,
        device="cpu",
        dtype="float32",
        expected_ffn_width=2,
        expected_model_state_sha256=state_sha,
        expected_parameter_inventory_sha256=inventory["inventory_sha256"],
        expected_tokenizer_semantics_sha256=(
            fingerprint_hf_tokenizer_semantics(tokenizer)
        ),
        model_loader=_tiny_model_loader,
        tokenizer_loader=_tiny_tokenizer_loader,
    )
    reloaded = loader(artifact)
    assert reloaded.training is False
    assert all(
        parameter.grad is None and parameter.requires_grad is False
        for parameter in reloaded.parameters()
    )
    evaluator = Lfm2CpcfV5HoldoutEvaluator(
        target_tokenizer=tokenizer,
        parent_envelope=parent_envelope,
        context_binding_sha256=context_sha,
        expected_target_tokenizer_semantics_sha256=(
            fingerprint_hf_tokenizer_semantics(tokenizer)
        ),
        candidate_generator=_tiny_candidate_rows,
    )
    evidence = evaluator(reloaded)

    assert (artifact / "model.safetensors").is_file()
    assert (artifact / "tokenizer.json").is_file()
    assert (artifact / LFM2_CPCF_V5_ARTIFACT_METADATA).is_file()
    assert evidence.output["case_count"] == 64
    assert evidence.output["cross_tokenizer_semantics_active"] is True
    assert evidence.output["exact_parent_token_matching_used"] is False
    assert {
        row["output"] for row in evidence.output["rows"]
    } == {"target-answer"}
    assert evidence.evaluation["manual_review_status"] == "pending"
    assert evidence.evaluation["automatic_promotion_authorized"] is False
    assert evidence.evaluation["gate_report"]["parent_forward_passes"] == 0
    assert evidence.evaluation["gate_report"]["optimizer_steps"] == 0
    assert evidence.evaluation["gate_report"]["backward_calls"] == 0


def test_full_quarantine_is_immutable_reloadable_and_nonpromotable(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=30).eval()
    parent = _TinyLfm2(seed=31).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    config = _config(tmp_path / "quarantine", parent_envelope, tokenizer)
    runtime = build_lfm2_cpcf_v5_quarantine_runtime(context, config)
    captured = capture_lfm2_cpcf_v5_quarantine(context, config)

    assert captured.path == config.destination
    assert captured.promotable is False
    assert captured.publication_authorized is False
    assert captured.requires_reviewed_rerun is True
    receipt = captured.receipt.as_dict()
    assert receipt["capture_request"]["compiler_sha256"] == (
        context.compiler_receipt_sha256
    )
    assert receipt["capture_request"]["source_sha256"] == (
        context.source_binding_sha256
    )
    assert receipt["audit"]["combined_clean"] is True
    assert receipt["audit"]["capture_observed"]["parent_forward_calls"] == 0
    assert receipt["audit"]["capture_observed"]["optimizer_steps"] == 0
    assert receipt["audit"]["capture_observed"]["backward_calls"] == 0
    assert receipt["audit"]["capture_observed"]["pruning_operations"] == 0
    assert receipt["audit"]["capture_observed"]["quantization_operations"] == 0
    again = load_quarantined_candidate(
        captured.path,
        request=runtime.request,
        expected_receipt_sha256=captured.receipt.receipt_sha256,
        loader=runtime.loader,
        evaluator=runtime.evaluator,
        forbidden_parent_models=(parent,),
    )
    assert again.receipt.receipt_sha256 == captured.receipt.receipt_sha256
    assert fingerprint_model_state(again.model) == fingerprint_model_state(target)

    metadata = captured.artifact_path / LFM2_CPCF_V5_ARTIFACT_METADATA
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["materialized_ffn_width"] = 3
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynamicCandidateCaptureIntegrityError):
        load_quarantined_candidate(
            captured.path,
            request=runtime.request,
            expected_receipt_sha256=captured.receipt.receipt_sha256,
            loader=runtime.loader,
            evaluator=runtime.evaluator,
            forbidden_parent_models=(parent,),
        )


def test_fixed_consumer_returns_exact_runner_result(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=40).eval()
    parent = _TinyLfm2(seed=41).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    consumer = Lfm2CpcfV5QuarantineConsumer(
        _config(tmp_path / "consumer", parent_envelope, tokenizer)
    )
    result = consumer(context)

    assert type(result) is runner.V5CompiledCandidateConsumerResult
    assert result.compiler_receipt_sha256 == context.compiler_receipt_sha256
    assert result.source_binding_sha256 == context.source_binding_sha256
    assert result.capture_receipt_sha256 == (
        result.candidate.receipt.receipt_sha256
    )
    assert result.candidate.path == tmp_path / "consumer"
    assert result.candidate.promotable is False


def test_context_cannot_substitute_compiler_implementation_for_run_receipt(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=50).eval()
    parent = _TinyLfm2(seed=51).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    runtime = build_lfm2_cpcf_v5_quarantine_runtime(
        context,
        _config(tmp_path / "unused", parent_envelope, tokenizer),
    )

    assert runtime.request.compiler_sha256 != context.compiler_sha256
    assert runtime.request.compiler_sha256 == context.compiler_receipt_sha256
    assert runtime.request.source_sha256 != context.source_sha256
    assert runtime.request.source_sha256 == context.source_binding_sha256


def test_parent_envelope_must_match_exact_compiled_parent(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=60).eval()
    parent = _TinyLfm2(seed=61).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    foreign_parent = replace(
        parent_envelope,
        parent_model_fingerprint=_digest("e"),
    )

    with pytest.raises(
        Lfm2CpcfV5QuarantineError,
        match="different parent or donor tokenizer",
    ):
        build_lfm2_cpcf_v5_quarantine_runtime(
            context,
            _config(tmp_path / "foreign-parent", foreign_parent, tokenizer),
        )


def test_parent_envelope_must_match_bridge_donor_tokenizer(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    target = _TinyLfm2(seed=70).eval()
    parent = _TinyLfm2(seed=71).eval()
    tokenizer = _TinyTokenizer(target=True)
    context = _context(target, parent, tokenizer)
    foreign_tokenizer = replace(
        parent_envelope,
        tokenizer_fingerprint=_digest("f"),
    )

    with pytest.raises(
        Lfm2CpcfV5QuarantineError,
        match="different parent or donor tokenizer",
    ):
        build_lfm2_cpcf_v5_quarantine_runtime(
            context,
            _config(
                tmp_path / "foreign-tokenizer",
                foreign_tokenizer,
                tokenizer,
            ),
        )


def test_forged_same_name_context_type_is_rejected(
    tmp_path: Path,
    parent_envelope: Any,
) -> None:
    tokenizer = _TinyTokenizer(target=True)
    forged_type = type(
        "V5CompiledCandidateConsumerContext",
        (),
        {"__module__": "forged.runner"},
    )
    forged = forged_type()

    with pytest.raises(
        Lfm2CpcfV5QuarantineError,
        match="exact canonical",
    ):
        build_lfm2_cpcf_v5_quarantine_runtime(
            forged,
            _config(tmp_path / "forged", parent_envelope, tokenizer),
        )
