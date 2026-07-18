from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments import capture_lfm2_cpcf_holdout_candidate as runner
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_intelligence_gate import (
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_candidate_outputs,
)


BRIDGE_ARTIFACT = hashlib.sha256(b"capture-bridge-artifact").hexdigest()
BRIDGE_PROFILE = hashlib.sha256(b"capture-bridge-profile").hexdigest()
DONOR_SEMANTICS = hashlib.sha256(b"capture-donor-semantics").hexdigest()
TARGET_SEMANTICS = hashlib.sha256(b"capture-target-semantics").hexdigest()
TARGET_PACKAGE = hashlib.sha256(b"capture-target-package").hexdigest()
CHECKPOINT_PACKAGE = hashlib.sha256(
    b"capture-checkpoint-tokenizer-package"
).hexdigest()
STATE_BEFORE = hashlib.sha256(b"capture-state-before").hexdigest()
STATE_AFTER = hashlib.sha256(b"capture-state-after").hexdigest()
SPECIAL_IDENTITY = hashlib.sha256(b"capture-special-identity").hexdigest()
EOS_ID = 7
MODEL_BRIDGE_KEY = "_test_strict_bridge_artifact"
PARAMETER_INVENTORY = {
    "method": "test-independent-parameter-inventory",
    "unique_parameters": 1,
    "independent_methods_agree": True,
    "inventory_sha256": hashlib.sha256(b"test-inventory").hexdigest(),
}


class _FakeBridge:
    artifact_sha256 = BRIDGE_ARTIFACT
    profile_fingerprint = BRIDGE_PROFILE
    donor_tokenizer_semantics_sha256 = DONOR_SEMANTICS
    target_tokenizer_semantics_sha256 = TARGET_SEMANTICS
    transport_profile = SimpleNamespace(target_row_capacity=50000)

    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_files(self) -> None:
        self.verify_calls += 1


class _FakeTokenizer:
    eos_token_id = EOS_ID
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert prompt
        assert return_tensors == "pt"
        assert add_special_tokens is True
        index = self.calls
        self.calls += 1
        return {
            "input_ids": torch.tensor([[1, 100 + index]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return f"Coherent synthetic child response {token_ids[0]}."


class _FakeChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = SimpleNamespace(
            **{
                MODEL_BRIDGE_KEY: BRIDGE_ARTIFACT,
                "_neural_forge_tokenizer_bridge_required": True,
            }
        )
        self.generate_calls = 0
        self.observed_limits: list[int] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls += 1
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        assert kwargs["eos_token_id"] == EOS_ID
        assert kwargs["pad_token_id"] == 0
        input_ids = kwargs["input_ids"]
        limit = int(kwargs["max_new_tokens"])
        self.observed_limits.append(limit)
        continuation = torch.tensor(
            [[20000 + self.generate_calls, EOS_ID]],
            device=input_ids.device,
            dtype=torch.long,
        )
        return torch.cat((input_ids, continuation), dim=1)


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpcf_receipt(
    checkpoint: Path,
    *,
    marker: str | None = None,
) -> dict[str, Any]:
    inventory = list(runner._checkpoint_inventory(checkpoint))
    core: dict[str, Any] = {
        "schema": runner.cpcf_runner.RUN_SCHEMA,
        "status": "folded_child_requires_manual_intelligence_review",
        "inputs": {
            "bridge_artifact_sha256": BRIDGE_ARTIFACT,
            "bridge_profile_sha256": BRIDGE_PROFILE,
        },
        "saved_checkpoint_files": inventory,
        "reloaded_parameter_inventory": PARAMETER_INVENTORY,
        "save_reload_config_aliases": {
            "fresh_reload": {
                "materialized_ffn_width": 13,
            }
        },
        "fresh_reload_verified": True,
        "structural_parameter_compaction_proven": True,
        "parent_state_preserved": True,
        "claims": {
            "folding_only": True,
            "blank_target_filled": True,
            "training_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "parent_backed_fallback_used": False,
        },
        "operation_audit": {
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "parent_cache_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantized_parameters": 0,
            "pruned_parameters": 0,
            "fallback_model_calls": 0,
            "network_access": False,
        },
    }
    if marker is not None:
        core["copy_marker"] = marker
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _fixture(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "frozen-child"
    target = tmp_path / "target-tokenizer"
    bridge_directory = tmp_path / "bridge"
    checkpoint.mkdir()
    target.mkdir()
    bridge_directory.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"frozen-child-weights")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (bridge_directory / "bridge-data.bin").write_bytes(b"strict-bridge")
    document = _cpcf_receipt(checkpoint)
    internal = checkpoint / runner.INTERNAL_CPCF_RECEIPT
    external = tmp_path / "external-cpcf-receipt.json"
    _write_canonical(internal, document)
    _write_canonical(external, document)
    manifest = load_lfm2_intelligence_holdout_manifest()
    return {
        "checkpoint": checkpoint,
        "target": target,
        "bridge_directory": bridge_directory,
        "internal": internal,
        "external": external,
        "receipt": document,
        "manifest": manifest,
        "candidate": tmp_path / "outputs" / "candidate.json",
        "capture": tmp_path / "outputs" / "capture.json",
    }


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
    *,
    state_hashes: list[str] | None = None,
) -> tuple[_FakeChild, _FakeTokenizer, _FakeBridge]:
    child = _FakeChild()
    tokenizer = _FakeTokenizer()
    bridge = _FakeBridge()
    monkeypatch.setattr(
        runner,
        "load_strict_tokenizer_bridge_artifact",
        lambda directory, **kwargs: (
            bridge
            if (
                Path(directory) == fixture["bridge_directory"].resolve()
                and kwargs["expected_artifact_sha256"] == BRIDGE_ARTIFACT
            )
            else pytest.fail("unexpected bridge load")
        ),
    )
    monkeypatch.setattr(
        runner,
        "strict_tokenizer_bridge_configuration",
        lambda observed: (
            {MODEL_BRIDGE_KEY: BRIDGE_ARTIFACT}
            if observed is bridge
            else pytest.fail("unexpected bridge object")
        ),
    )
    monkeypatch.setattr(
        runner.posthoc,
        "fingerprint_bound_local_tokenizer",
        lambda directory: (
            (
                CHECKPOINT_PACKAGE
                if Path(directory) == fixture["checkpoint"].resolve()
                else TARGET_PACKAGE
            ),
            (
                {
                    "path": "tokenizer.json",
                    "bytes": 2,
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                },
            ),
        ),
    )
    monkeypatch.setattr(
        runner.posthoc,
        "load_bound_special_token_ids",
        lambda _directory, **_kwargs: {
            "special_token_ids": {"eos": [EOS_ID]},
            "special_token_identity_sha256": SPECIAL_IDENTITY,
        },
    )
    monkeypatch.setattr(
        runner.fold_runtime,
        "load_local_tokenizer",
        lambda _directory: tokenizer,
    )
    monkeypatch.setattr(
        runner,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: TARGET_SEMANTICS,
    )
    monkeypatch.setattr(
        runner.fold_runtime,
        "load_local_lfm2_model",
        lambda directory, **kwargs: (
            child
            if (
                Path(directory) == fixture["checkpoint"].resolve()
                and kwargs["device"] == "cpu"
                and kwargs["dtype"] == torch.float32
            )
            else pytest.fail("unexpected child load")
        ),
    )
    monkeypatch.setattr(
        runner.cpcf_runner,
        "_verify_saved_lfm2_ffn_config",
        lambda directory, **kwargs: {
            "scope": "saved_config_json",
            "config_json_sha256": _file_sha(
                Path(directory) / "config.json"
            ),
            "serialized_aliases": {
                "intermediate_size": kwargs["expected_width"],
                "block_ff_dim": kwargs["expected_width"],
                "block_auto_adjust_ff_dim": False,
            },
            "shape_stable": True,
            "report_sha256": hashlib.sha256(b"saved-alias").hexdigest(),
        },
    )
    monkeypatch.setattr(
        runner.cpcf_runner,
        "_restore_lfm2_ffn_config_alias",
        lambda _model, scope: {
            "scope": scope,
            "materialized_ffn_width": 13,
            "previous_block_ff_dim": None,
            "serialized_aliases": {
                "intermediate_size": 13,
                "block_ff_dim": 13,
                "block_auto_adjust_ff_dim": False,
            },
            "shape_stable": True,
            "report_sha256": hashlib.sha256(b"live-alias").hexdigest(),
        },
    )
    monkeypatch.setattr(
        runner,
        "independent_parameter_inventory",
        lambda _model: PARAMETER_INVENTORY,
    )
    hashes = iter(state_hashes or [STATE_BEFORE, STATE_BEFORE])
    monkeypatch.setattr(
        runner.fold_runtime,
        "model_state_sha256",
        lambda _model: next(hashes),
    )
    return child, tokenizer, bridge


def _config(fixture: dict[str, Any]) -> runner.CaptureConfig:
    return runner.CaptureConfig(
        checkpoint_directory=fixture["checkpoint"],
        target_tokenizer_directory=fixture["target"],
        bridge_directory=fixture["bridge_directory"],
        cpcf_receipt=fixture["external"],
        expected_cpcf_receipt_file_sha256=_file_sha(
            fixture["external"]
        ),
        expected_cpcf_receipt_sha256=fixture["receipt"]["receipt_sha256"],
        expected_manifest_sha256=fixture["manifest"].manifest_sha256,
        expected_target_tokenizer_fingerprint=TARGET_PACKAGE,
        expected_checkpoint_tokenizer_fingerprint=CHECKPOINT_PACKAGE,
        expected_target_tokenizer_semantics_sha256=TARGET_SEMANTICS,
        expected_bridge_artifact_sha256=BRIDGE_ARTIFACT,
        expected_bridge_profile_sha256=BRIDGE_PROFILE,
        candidate_output_path=fixture["candidate"],
        capture_receipt_path=fixture["capture"],
        expected_candidate_state_fingerprint=STATE_BEFORE,
        device="cpu",
        dtype="float32",
    )


def test_capture_authenticates_frozen_child_and_seals_child_domain_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    child, tokenizer, bridge = _patch_runtime(monkeypatch, fixture)

    receipt = runner.capture_lfm2_cpcf_holdout_candidate(
        _config(fixture)
    )

    assert child.generate_calls == 64
    assert tokenizer.calls == 64
    assert child.observed_limits == [
        spec.generation_limit for spec in fixture["manifest"].cases
    ]
    assert bridge.verify_calls >= 2
    assert fixture["candidate"].is_file()
    assert fixture["capture"].is_file()
    assert receipt["status"] == "captured_frozen_child_holdout"
    assert receipt["candidate"]["state_unchanged"] is True
    assert receipt["candidate"]["candidate_state_fingerprint"] == STATE_BEFORE
    assert receipt["target_tokenizer"][
        "source_package_fingerprint"
    ] == TARGET_PACKAGE
    assert receipt["target_tokenizer"][
        "checkpoint_package_fingerprint"
    ] == CHECKPOINT_PACKAGE
    assert receipt["target_tokenizer"][
        "source_and_checkpoint_packages_identical"
    ] is False
    assert receipt["target_tokenizer"]["semantics_sha256"] == TARGET_SEMANTICS
    assert receipt["generation_protocol"]["global_64_token_cap_used"] is False
    assert receipt["operation_audit"]["candidate_generate_invocations"] == 64
    for name in (
        "parent_model_loaded",
        "training_used",
        "optimizer_constructed",
        "gradients_used",
        "quantization_used",
        "pruning_used",
        "network_access",
        "hub_loader_used",
    ):
        assert receipt["operation_audit"][name] is False
    stored_candidate = json.loads(fixture["candidate"].read_text())
    loaded = load_lfm2_intelligence_holdout_candidate_outputs(
        stored_candidate,
        expected_candidate_outputs_sha256=stored_candidate[
            "candidate_outputs_sha256"
        ],
        expected_manifest_sha256=fixture["manifest"].manifest_sha256,
        expected_candidate_state_fingerprint=STATE_BEFORE,
        expected_tokenizer_fingerprint=CHECKPOINT_PACKAGE,
        eos_token_ids=[EOS_ID],
    )
    assert len(loaded.rows) == 64
    assert all(row["output_token_ids"][-1] == EOS_ID for row in loaded.rows)
    capture_core = dict(receipt)
    capture_seal = capture_core.pop("capture_receipt_sha256")
    assert capture_seal == canonical_sha256(capture_core)
    assert receipt["candidate"]["output_file_sha256"] == _file_sha(
        fixture["candidate"]
    )


def test_capture_rejects_checkpoint_file_not_in_cpcf_receipt_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    (fixture["checkpoint"] / "unreceipted.bin").write_bytes(b"mutation")
    monkeypatch.setattr(
        runner.fold_runtime,
        "load_local_lfm2_model",
        lambda *_args, **_kwargs: pytest.fail("child must not load"),
    )

    with pytest.raises(
        runner.Lfm2CpcfHoldoutCaptureError,
        match="files differ",
    ):
        runner.capture_lfm2_cpcf_holdout_candidate(_config(fixture))

    assert not fixture["candidate"].exists()
    assert not fixture["capture"].exists()


def test_capture_rejects_nonidentical_external_and_internal_receipts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    different = _cpcf_receipt(
        fixture["checkpoint"],
        marker="different-external-copy",
    )
    _write_canonical(fixture["external"], different)
    fixture["receipt"] = different

    with pytest.raises(
        runner.Lfm2CpcfHoldoutCaptureError,
        match="not identical",
    ):
        runner.capture_lfm2_cpcf_holdout_candidate(_config(fixture))

    assert not fixture["candidate"].exists()
    assert not fixture["capture"].exists()


def test_capture_rejects_child_state_mutation_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    child, _tokenizer, _bridge = _patch_runtime(
        monkeypatch,
        fixture,
        state_hashes=[STATE_BEFORE, STATE_AFTER],
    )

    with pytest.raises(
        runner.Lfm2CpcfHoldoutCaptureError,
        match="state or gradients changed",
    ):
        runner.capture_lfm2_cpcf_holdout_candidate(_config(fixture))

    assert child.generate_calls == 64
    assert not fixture["candidate"].exists()
    assert not fixture["capture"].exists()


def test_capture_rejects_child_without_required_bridge_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    child, _tokenizer, _bridge = _patch_runtime(monkeypatch, fixture)
    delattr(child.config, "_neural_forge_tokenizer_bridge_required")

    with pytest.raises(
        runner.Lfm2CpcfHoldoutCaptureError,
        match="not bound to the strict bridge",
    ):
        runner.capture_lfm2_cpcf_holdout_candidate(_config(fixture))

    assert child.generate_calls == 0
    assert not fixture["candidate"].exists()
    assert not fixture["capture"].exists()


def test_capture_source_has_no_parent_loader_training_or_network_path() -> None:
    source = inspect.getsource(runner)

    for forbidden in (
        "AutoModel",
        "AutoTokenizer",
        ".from_pretrained(",
        "requests.",
        "urllib.",
        "torch.optim",
        ".backward(",
        "load_local_parent",
        "build_lfm2_circuit_trace_cache",
    ):
        assert forbidden not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "load_capture_manifest" in source
    assert "seal_capture_candidate_outputs" in source
    assert "load_local_lfm2_model" in source
