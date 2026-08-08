from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics
from neural_forge.lfm2_carrier_bridge_publication import (
    STANDALONE_BRIDGE_SCHEMA,
    carrier_materialization_tensor_sha256,
)
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
)
from neural_forge.lfm2_carrier_child_package import (
    ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES,
    CARRIER_CHILD_PACKAGE_RECEIPT,
    CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
    LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
    LFM2_CARRIER_EXACT_4X_COUNTED_ABI,
    Lfm2CarrierChildCheckpointIO,
    Lfm2CarrierChildPackageError,
    Lfm2CarrierChildPackageRegistration,
    Lfm2CarrierCountedAbiProfile,
    admitted_lfm2_carrier_counted_abi_profile,
    load_authenticated_lfm2_carrier_child_package,
)


pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Child(nn.Module):
    def __init__(self, rows: int = 2, hidden_size: int = 3) -> None:
        super().__init__()
        self.input_table = nn.Embedding(rows, hidden_size)
        self.output_table = nn.Linear(hidden_size, rows, bias=False)
        self.model = nn.Identity()

    def get_input_embeddings(self) -> nn.Module:
        return self.input_table

    def get_output_embeddings(self) -> nn.Module:
        return self.output_table


class _BaseCheckpointIO:
    saver_id = "test-safetensors-tokenizer-v1"
    loader_id = "test-safetensors-tokenizer-v1"
    resource_fingerprint = "test-base-checkpoint"

    def __init__(self, tokenizer_source: Path | None) -> None:
        self.tokenizer_source = tokenizer_source

    def save(self, model: _Child, destination: Path) -> None:
        if self.tokenizer_source is None:
            raise RuntimeError("The read-only base loader cannot save.")
        destination.mkdir(parents=True)
        save_file(
            {
                "input_table.weight": model.input_table.weight.detach().cpu(),
                "output_table.weight": model.output_table.weight.detach().cpu(),
            },
            str(destination / "model.safetensors"),
        )
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(self.tokenizer_source / name, destination / name)

    def load(self, source: Path) -> _Child:
        state = load_file(str(Path(source) / "model.safetensors"), device="cpu")
        rows, hidden_size = state["input_table.weight"].shape
        model = _Child(int(rows), int(hidden_size))
        model.input_table.weight.data.copy_(state["input_table.weight"])
        model.output_table.weight.data.copy_(state["output_table.weight"])
        return model


def _tokenizer_package(
    path: Path,
    vocabulary: dict[str, int],
) -> tuple[Any, str]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir()
    backend = Tokenizer(
        WordLevel(
            vocabulary,
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"unk_token": "<unk>", "eos_token": "<eos>"}),
        encoding="utf-8",
    )
    tokenizer = load_lfm2_direct_local_tokenizer(path)
    return tokenizer, fingerprint_hf_tokenizer_semantics(tokenizer)


def _csr(
    rows: list[tuple[list[int], list[float]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [0]
    target_ids: list[int] = []
    coefficients: list[float] = []
    for ids, values in rows:
        target_ids.extend(ids)
        coefficients.extend(values)
        offsets.append(len(target_ids))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(target_ids, dtype=torch.int64),
        torch.tensor(coefficients, dtype=torch.float32),
    )


def _source_publication(root: Path) -> tuple[Path, str]:
    publication = root / "publication"
    publication.mkdir()
    input_code = _csr(
        [([0], [1.0]), ([1], [1.0]), ([0, 1], [0.4, 0.6])]
    )
    output_code = _csr(
        [([0], [1.0]), ([1], [1.0]), ([0, 1], [0.25, 0.75])]
    )
    tensors = {
        "target_table": torch.arange(10, dtype=torch.float32).reshape(2, 5),
        "target_to_donor": torch.tensor([0, 1], dtype=torch.int64),
        "input_offsets": input_code[0],
        "input_target_ids": input_code[1],
        "input_coefficients": input_code[2],
        "output_offsets": output_code[0],
        "output_target_ids": output_code[1],
        "output_coefficients": output_code[2],
    }
    tensor_path = publication / "bridge.safetensors"
    save_file(tensors, str(tensor_path))
    coefficient_count = int(input_code[2].numel() + output_code[2].numel())
    operation_audit = {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sparsification_calls": 0,
        "parameter_sharing_calls": 0,
        "runtime_sidecar_parameters": 0,
    }
    body = {
        "schema": STANDALONE_BRIDGE_SCHEMA,
        "status": "complete",
        "bridge_tensor_file": tensor_path.name,
        "bridge_tensor_file_sha256": _file_sha256(tensor_path),
        "runtime_requires_parent": False,
        "runtime_requires_external_sidecar": False,
        "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
        "integer_csr_routes_are_structural_metadata": True,
        "continuous_semantic_coefficient_count": coefficient_count,
        "physical_target_table_parameter_count": int(
            tensors["target_table"].numel()
        ),
        "target_table_dtype": str(tensors["target_table"].dtype),
        "operation_audit": operation_audit,
        "materialization_receipt": {
            "input_coefficients_sha256": carrier_materialization_tensor_sha256(
                input_code[2]
            ),
            "output_coefficients_sha256": carrier_materialization_tensor_sha256(
                output_code[2]
            ),
            "continuous_fold_coefficient_count": coefficient_count,
        },
    }
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    (publication / "bridge-receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return publication, str(receipt["artifact_sha256"])


def _build_child_package(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    semantic_tokenizer_source = tmp_path / "semantic-source-tokenizer"
    semantic_tokenizer, semantics = _tokenizer_package(
        semantic_tokenizer_source,
        {"<unk>": 0, "<eos>": 1, "hello": 2},
    )
    target_tokenizer_source = tmp_path / "target-tokenizer"
    target_tokenizer, _target_semantics = _tokenizer_package(
        target_tokenizer_source,
        {"<unk>": 0, "<eos>": 1},
    )
    publication, artifact_sha256 = _source_publication(tmp_path)
    spans = seal_lfm2_semantic_step_span_evidence(
        [[0], [1], [0, 1]],
        target_row_count=2,
        lineage={"test_fixture": True},
    )
    span_path = tmp_path / "spans.json"
    span_path.write_text(
        json.dumps(spans, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    registration = Lfm2CarrierChildPackageRegistration(
        publication_directory=publication,
        expected_artifact_sha256=artifact_sha256,
        span_evidence_json=span_path,
        expected_span_evidence_sha256=str(spans["evidence_sha256"]),
        expected_span_evidence_file_sha256=_file_sha256(span_path),
        expected_target_rows=2,
        expected_hidden_size=3,
        expected_input_route_elements=4,
        expected_output_route_elements=4,
    )
    writer = Lfm2CarrierChildCheckpointIO(
        base=_BaseCheckpointIO(target_tokenizer_source),
        registration=registration,
        semantic_source_tokenizer_directory=semantic_tokenizer_source,
        semantic_source_tokenizer_sha256=semantics,
        in_memory_semantic_source_tokenizer=semantic_tokenizer,
        in_memory_target_tokenizer=target_tokenizer,
    )
    child_directory = tmp_path / "child"
    writer.save(_Child(), child_directory)
    package = load_authenticated_lfm2_carrier_child_package(child_directory)
    assert package.receipt["external_source_publication_required_at_load"] is False
    assert package.receipt["external_source_tokenizer_required_at_load"] is False
    return child_directory, (
        publication,
        semantic_tokenizer_source,
        target_tokenizer_source,
        span_path,
    )


def _fresh_process_probe(child_directory: Path) -> dict[str, Any]:
    import neural_forge.lfm2_carrier_route_runtime as route_runtime
    from neural_forge.hf_lfm2_semantic_step_model_runtime import (
        LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
    )

    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_BaseCheckpointIO(None),
        child_directory=child_directory,
    )
    model = loader.load(child_directory)

    def fake_review(
        _model: Any,
        _runtime: Any,
        _semantic_source_tokenizer: Any,
        _target_tokenizer: Any,
        prompts: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
            "status": "complete-standalone-semantic-step-whole-response-review",
            "responses": [
                {**dict(prompt), "response": "hello", "diagnostics": {}}
                for prompt in prompts
            ],
            "input_output_tables_are_untied": True,
            "routing_session_receipt": {
                "input": {"route_role": "input", "routing_cache_use_count": 1},
                "output": {"route_role": "output", "routing_cache_use_count": 1},
            },
            "operation_audit": {
                "child_body_forward_calls": 1,
                "child_generate_calls": 0,
                "child_lm_head_forward_calls": 0,
                "routing_input_output_coefficient_storage_shared": False,
            },
            "runtime_sha256": "fresh-process-test-runtime",
        }

    route_runtime.run_lfm2_semantic_step_model_review = fake_review
    routed = model.semantic_step_generate(
        ({"id": "fresh", "category": "general", "prompt": "hello"},),
        maximum_new_tokens=2,
    )
    package = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        child_model=model,
        expected_package_sha256=loader.expected_child_package_sha256,
    )
    return {
        "public_entrypoint_callable": callable(
            getattr(model, "semantic_step_generate", None)
        ),
        "registration_is_absent": loader.registration is None,
        "package_sha256": package.package_sha256,
        "source_publication_required": package.receipt[
            "external_source_publication_required_at_load"
        ],
        "source_tokenizer_required": package.receipt[
            "external_source_tokenizer_required_at_load"
        ],
        "routed_operation_audit": routed["operation_audit"],
    }


def test_fresh_process_loads_only_from_sealed_child_directory(tmp_path: Path) -> None:
    child_directory, build_sources = _build_child_package(tmp_path)
    expected_package = load_authenticated_lfm2_carrier_child_package(
        child_directory
    ).package_sha256
    for path in build_sources:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    environment = dict(os.environ)
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source_root),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--fresh-probe", str(child_directory)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed == {
        "package_sha256": expected_package,
        "public_entrypoint_callable": True,
        "registration_is_absent": True,
        "routed_operation_audit": {
            "child_body_forward_calls": 1,
            "child_generate_calls": 0,
            "child_lm_head_forward_calls": 0,
            "routing_input_output_coefficient_storage_shared": False,
        },
        "source_publication_required": False,
        "source_tokenizer_required": False,
    }


@pytest.mark.parametrize("tamper_mode", ["addition", "byte-change"])
def test_child_local_outer_inventory_rejects_tampering(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    child_directory, _build_sources = _build_child_package(tmp_path)
    if tamper_mode == "addition":
        (child_directory / "unsealed.bin").write_bytes(b"not in outer inventory")
    else:
        with (child_directory / "model.safetensors").open("ab") as handle:
            handle.write(b"changed")

    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="differs from its sealed inventory",
    ):
        Lfm2CarrierChildCheckpointIO.from_child_directory(
            base=_BaseCheckpointIO(None),
            child_directory=child_directory,
        )


def test_child_local_outer_inventory_rejects_receipt_tampering(
    tmp_path: Path,
) -> None:
    child_directory, _build_sources = _build_child_package(tmp_path)
    receipt_path = child_directory / CARRIER_CHILD_PACKAGE_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["standalone_child_local_runtime"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="receipt or self-seal is invalid",
    ):
        Lfm2CarrierChildCheckpointIO.from_child_directory(
            base=_BaseCheckpointIO(None),
            child_directory=child_directory,
        )


def _production_route_registration(
    *,
    route_elements: int = CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
) -> Lfm2CarrierChildPackageRegistration:
    """A registration whose route allocations are stated rather than loaded.

    The counted re-pin is arithmetic over the published route sizes; this
    isolates that arithmetic from the publication loader so both admitted
    budgets can be exercised against the real 1,933,888-element channels.
    """

    class _StatedRoutes(Lfm2CarrierChildPackageRegistration):
        def allocations(self):
            return (
                ContinuousChildStateAllocation(
                    state_id="semantic-carrier-route.c_in",
                    parameter_elements=route_elements,
                    tensor_sha256="8" * 64,
                    artifact_sha256="9" * 64,
                ),
                ContinuousChildStateAllocation(
                    state_id="semantic-carrier-route.c_out",
                    parameter_elements=route_elements,
                    tensor_sha256="a" * 64,
                    artifact_sha256="9" * 64,
                ),
            )

    return _StatedRoutes(
        publication_directory=Path("carrier-route"),
        expected_artifact_sha256="4" * 64,
        span_evidence_json=Path("spans.json"),
        expected_span_evidence_sha256="5" * 64,
        expected_span_evidence_file_sha256="6" * 64,
        expected_target_rows=4_096,
        expected_hidden_size=816,
        expected_input_route_elements=route_elements,
        expected_output_route_elements=route_elements,
    )


def test_counted_plan_re_pin_admits_both_sealed_carrier_budgets() -> None:
    """The runtime re-pin follows the run's admitted tuple, not one constant."""

    registration = _production_route_registration()

    exact_4x = registration.admitted_counted_plan(
        LFM2_CARRIER_EXACT_4X_COUNTED_ABI
    )
    exact_2x = registration.admitted_counted_plan(
        LFM2_CARRIER_EXACT_2X_COUNTED_ABI
    )

    assert exact_4x.target_total_parameter_elements == 57_420_144
    assert exact_4x.dense_model_parameter_elements == 53_552_368
    assert exact_2x.target_total_parameter_elements == 114_846_592
    assert exact_2x.dense_model_parameter_elements == 110_978_816
    for plan in (exact_4x, exact_2x):
        assert plan.parent_unique_parameters == 229_693_184
        assert plan.continuous_parameter_elements == 3_867_776
    # The exact-4x wrapper the carrier lane has always called is the same
    # object, so nothing about the proven lane moved.
    assert registration.current_exact4x_counted_plan() == exact_4x


def test_counted_plan_re_pin_refuses_a_mismatched_package() -> None:
    """Routes of the wrong size no longer reserve either admitted body."""

    mismatched = _production_route_registration(route_elements=1_933_889)

    for profile in ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES:
        with pytest.raises(
            Lfm2CarrierChildPackageError,
            match="did not reserve its exact H816 dense body",
        ):
            mismatched.admitted_counted_plan(profile)


def test_counted_plan_re_pin_refuses_an_unlisted_profile() -> None:
    """An equal-looking tuple assembled by a caller is not admitted."""

    registration = _production_route_registration()
    forged = Lfm2CarrierCountedAbiProfile(
        profile_id="exact-4x-h816",
        compression_ratio=4.0,
        counted_target_total_parameter_elements=57_420_144,
        counted_dense_body_parameter_elements=53_552_368,
        hidden_size=816,
    )

    assert forged == LFM2_CARRIER_EXACT_4X_COUNTED_ABI
    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="sealed to the admitted carrier ABI",
    ):
        registration.admitted_counted_plan(forged)


def test_admitted_carrier_abi_lookup_is_coherent_not_per_field() -> None:
    """Every field has to come from the same admitted profile."""

    admitted = {
        (4.0, 57_420_144, 53_552_368, 816): "exact-4x-h816",
        (2.0, 114_846_592, 110_978_816, 816): "exact-2x-h816",
    }
    for tuple_, profile_id in admitted.items():
        ratio, total, body, hidden = tuple_
        profile = admitted_lfm2_carrier_counted_abi_profile(
            compression_ratio=ratio,
            counted_target_total_parameter_elements=total,
            counted_dense_body_parameter_elements=body,
            hidden_size=hidden,
        )
        assert profile is not None
        assert profile.profile_id == profile_id

    refused = (
        (2.0, 57_420_144, 53_552_368, 816),      # 2x ratio, 4x budget
        (4.0, 114_846_592, 110_978_816, 816),    # 4x ratio, 2x budget
        (2.0, 114_846_592, 110_978_816, 1_024),  # 2x budget, H1024
        (4.0, 57_420_144, 110_978_816, 816),     # mixed totals
        (None, 57_420_144, 53_552_368, 816),     # no declared ratio
        (4.0, 57_423_296, 53_555_520, 816),      # the exact quarter
    )
    for ratio, total, body, hidden in refused:
        assert (
            admitted_lfm2_carrier_counted_abi_profile(
                compression_ratio=ratio,
                counted_target_total_parameter_elements=total,
                counted_dense_body_parameter_elements=body,
                hidden_size=hidden,
            )
            is None
        )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--fresh-probe":
        raise SystemExit("usage: test file --fresh-probe CHILD_DIRECTORY")
    print(
        json.dumps(
            _fresh_process_probe(Path(sys.argv[2])),
            allow_nan=False,
            sort_keys=True,
        )
    )
