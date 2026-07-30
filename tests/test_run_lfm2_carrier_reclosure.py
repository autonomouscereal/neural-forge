from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments import run_lfm2_carrier_reclosure as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_carrier_allocation import (
    build_lfm2_semantic_carrier_demand_evidence,
    build_lfm2_semantic_carrier_inventory,
)
from neural_forge.tokenizer_control_contract import (
    TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
    compile_local_tokenizer_control_contract,
    inventory_local_tokenizer_control_sources,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_sealed_json(path: Path, body: dict[str, object]) -> dict[str, object]:
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    _write_json(path, receipt)
    return receipt


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


def _tokenizer(root: Path, *, reserved_carrier: bool) -> tuple[Path, dict[str, str]]:
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    carrier_text = "[unused0]" if reserved_carrier else "carrier"
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        carrier_text: 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    _write_json(
        tokenizer / "tokenizer.json",
        {
            "model": {"type": "BPE", "vocab": vocab},
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
        },
    )
    _write_json(
        tokenizer / "tokenizer_config.json",
        {
            "unk_token": "<unk>",
            "unk_token_id": 0,
            "bos_token": "<s>",
            "bos_token_id": 1,
            "eos_token": "</s>",
            "eos_token_id": 2,
            "added_tokens_decoder": {
                "0": {"content": "<unk>", "special": True},
                "1": {"content": "<s>", "special": True},
                "2": {"content": "</s>", "special": True},
            },
            "chat_template": "{{ '<s>' + messages[0]['content'] + '</s>' }}",
        },
    )
    (tokenizer / "chat_template.jinja").write_text(
        "{{ '<s>' + messages[0]['content'] + '</s>' }}",
        encoding="utf-8",
    )
    return tokenizer, inventory_local_tokenizer_control_sources(tokenizer)


def _case(
    tmp_path: Path,
    *,
    reserved_carrier: bool = False,
    reclaim_reserved_carrier: bool = False,
    positive_carrier_incidence: bool = True,
) -> dict[str, object]:
    target_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    parent_table = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.35, 0.25, 0.4],
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float64,
    )
    mapping = torch.arange(4, dtype=torch.int64)
    input_code = _csr(
        [
            ([0], [1.0]),
            ([1], [1.0]),
            ([2], [1.0]),
            ([3], [1.0]),
            (
                ([0, 3] if positive_carrier_incidence else [0, 1]),
                [0.6, 0.4],
            ),
            ([0, 1], [0.6, 0.4]),
            ([1, 2], [0.8, 0.2]),
            ([0, 2], [0.3, 0.7]),
        ]
    )
    output_code = _csr(
        [
            ([0], [1.0]),
            ([1], [1.0]),
            ([2], [1.0]),
            ([3], [1.0]),
            (
                ([1, 3] if positive_carrier_incidence else [1, 2]),
                [0.75, 0.25],
            ),
            ([0, 2], [0.3, 0.7]),
            ([1, 2], [0.9, 0.1]),
            ([0, 1], [0.8, 0.2]),
        ]
    )
    base_tensors = {
        "target_table": target_table,
        "target_to_donor": mapping,
        "input_offsets": input_code[0],
        "input_target_ids": input_code[1],
        "input_coefficients": input_code[2],
        "output_offsets": output_code[0],
        "output_target_ids": output_code[1],
        "output_coefficients": output_code[2],
    }
    base_path = tmp_path / "base.safetensors"
    save_file(base_tensors, str(base_path))
    parent_path = tmp_path / "parent.safetensors"
    save_file({"parent_table": parent_table}, str(parent_path))

    solver_config = {
        "support_size": 2,
        "rows_per_chunk": 1,
        "ridge": 1e-5,
        "selection_device": "cpu",
        "selection_dtype": "bfloat16",
        "solve_device": "cpu",
        "solve_dtype": "float64",
    }
    numeric_contract = runner._solver_math_contract(solver_config)
    numeric_sha = canonical_sha256(numeric_contract)
    inventory = build_lfm2_semantic_carrier_inventory(
        input_offsets=input_code[0],
        input_target_ids=input_code[1],
        input_coefficients=input_code[2],
        output_offsets=output_code[0],
        output_target_ids=output_code[1],
        output_coefficients=output_code[2],
        target_to_donor=mapping,
    )
    evidence_tensors = {
        "donor_occurrence_counts": torch.tensor(
            [0, 0, 0, 0, 2, 1, 100, 1], dtype=torch.int64
        ),
        "donor_winner_counts": torch.tensor(
            [0, 0, 0, 0, 0, 0, 10, 0], dtype=torch.int64
        ),
        "donor_reconstruction_errors": torch.tensor(
            [0, 0, 0, 0, 1, 1, 10, 1], dtype=torch.float64
        ),
        "donor_eviction_costs": torch.tensor(
            [1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float64
        ),
    }
    evidence_tensor_path = tmp_path / "carrier-evidence.safetensors"
    save_file(evidence_tensors, str(evidence_tensor_path))
    compiled_evidence = build_lfm2_semantic_carrier_demand_evidence(
        evidence_tensors["donor_occurrence_counts"],
        evidence_tensors["donor_winner_counts"],
        evidence_tensors["donor_reconstruction_errors"],
        evidence_tensors["donor_eviction_costs"],
        source_artifact_sha256=_sha("carrier-source"),
        source_corpus_sha256=_sha("fold-fit-corpus"),
        parent_identity_sha256=_sha("parent"),
        fit_partition_manifest_sha256=_sha("fit-partition"),
        producer_implementation_sha256=_sha("producer"),
        numeric_contract_sha256=numeric_sha,
        base_semantic_binding_sha256=canonical_sha256(
            inventory.as_dict()["base_binding"]
        ),
        partition_role="fold-fit",
        selection_allowed=True,
    )
    evidence_body: dict[str, object] = {
        "schema": runner.CARRIER_EVIDENCE_INPUT_SCHEMA,
        "partition_role": "fold-fit",
        "selection_allowed": True,
        "review_or_evaluation_evidence_allowed": False,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "tensor_file_sha256": _file_sha256(evidence_tensor_path),
        "source_artifact_sha256": _sha("carrier-source"),
        "source_corpus_sha256": _sha("fold-fit-corpus"),
        "parent_identity_sha256": _sha("parent"),
        "fit_partition_manifest_sha256": _sha("fit-partition"),
        "producer_implementation_sha256": _sha("producer"),
        "numeric_contract_sha256": numeric_sha,
        "base_semantic_binding_sha256": canonical_sha256(
            inventory.as_dict()["base_binding"]
        ),
        "compiled_evidence_artifact_sha256": compiled_evidence.artifact_sha256,
        **{
            f"{name}_sha256": runner._tensor_sha256(tensor)
            for name, tensor in evidence_tensors.items()
        },
    }
    evidence_receipt_path = tmp_path / "carrier-evidence.json"
    _write_sealed_json(evidence_receipt_path, evidence_body)

    metric_specs: dict[str, dict[str, object]] = {}
    for channel, metric in (
        ("input", torch.eye(3, dtype=torch.float32)),
        ("output", torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))),
    ):
        tensor_path = tmp_path / f"{channel}-metric.safetensors"
        save_file({"metric": metric}, str(tensor_path))
        receipt_path = tmp_path / f"{channel}-metric.json"
        body = {
            "schema": runner.SOLVER_EVIDENCE_SCHEMA,
            "channel": channel,
            "partition_role": "fold-fit",
            "cached_trace_evidence": True,
            "review_or_evaluation_evidence_allowed": False,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "metric_sha256": runner._tensor_sha256(metric),
            "numeric_contract_sha256": numeric_sha,
            "trace_cache_artifact_sha256": _sha(f"{channel}-trace-cache"),
            "fit_partition_manifest_sha256": _sha(f"{channel}-fit-partition"),
        }
        _write_sealed_json(receipt_path, body)
        metric_specs[channel] = {
            "tensors": {"path": str(tensor_path), "sha256": _file_sha256(tensor_path)},
            "receipt": {
                "path": str(receipt_path),
                "sha256": _file_sha256(receipt_path),
            },
        }

    tokenizer, tokenizer_hashes = _tokenizer(
        tmp_path, reserved_carrier=reserved_carrier
    )
    tokenizer_spec: dict[str, object] = {
        "artifact_directory": str(tokenizer),
        "expected_file_sha256": tokenizer_hashes,
        "manifest_files": [],
    }
    if reclaim_reserved_carrier:
        default_contract = compile_local_tokenizer_control_contract(
            tokenizer,
            expected_file_sha256=tokenizer_hashes,
        )
        policy_path = tokenizer / "reclaim-policy.json"
        _write_json(
            policy_path,
            {
                "schema": TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
                "source_inventory_sha256": default_contract.receipt[
                    "source_inventory_sha256"
                ],
                "release_reserved_token_ids": [3],
                "reason": "Authenticated toy proof that the runner composes reclaim policy.",
            },
        )
        tokenizer_spec.update(
            {
                "reclaim_policy_file": "reclaim-policy.json",
                "expected_reclaim_policy_sha256": _file_sha256(policy_path),
            }
        )

    return {
        "schema": runner.REQUEST_SCHEMA,
        "run_id": "toy-carrier-reclosure",
        "phase_cache_directory": str(tmp_path / "phase-cache"),
        "event_log": str(tmp_path / "events.jsonl"),
        "output_directory": str(tmp_path / "output"),
        "base_bridge": {"path": str(base_path), "sha256": _file_sha256(base_path)},
        "parent_table": {
            "path": str(parent_path),
            "sha256": _file_sha256(parent_path),
        },
        "tokenizer_control": tokenizer_spec,
        "carrier_evidence": {
            "tensors": {
                "path": str(evidence_tensor_path),
                "sha256": _file_sha256(evidence_tensor_path),
            },
            "receipt": {
                "path": str(evidence_receipt_path),
                "sha256": _file_sha256(evidence_receipt_path),
            },
        },
        "input_solver_evidence": metric_specs["input"],
        "output_solver_evidence": metric_specs["output"],
        "selection": {
            "promotion_count": 1,
            "winner_weight": 8,
            "allow_positive_incidence": True,
        },
        "solver": solver_config,
    }


def test_positive_incidence_compile_recloses_both_channels_and_publishes(
    tmp_path: Path,
) -> None:
    request = _case(tmp_path)
    output = io.StringIO()

    result = runner.run_carrier_reclosure(request, log_stream=output)

    artifact = load_file(str(tmp_path / "output" / "bridge.safetensors"))
    assert result["status"] == "complete"
    assert result["target_table_dtype"] == "torch.bfloat16"
    assert result["protected_donor_ids"] == [0, 1, 2]
    assert result["reclaimed_reserved_donor_ids"] == []
    assert result["input_and_output_routes_are_independent"] is True
    assert result["complete_reverse_incidence_closure_materialized"] is True
    assert result["runtime_requires_parent"] is False
    assert result["runtime_requires_external_sidecar"] is False
    assert result["materialization_receipt"]["required_rerouted_donor_count"] == 2
    assert artifact["target_table"].dtype is torch.bfloat16
    assert not torch.equal(
        artifact["input_coefficients"], artifact["output_coefficients"]
    )
    promoted_start = int(artifact["input_offsets"][6])
    promoted_stop = int(artifact["input_offsets"][7])
    assert artifact["input_target_ids"][promoted_start:promoted_stop].tolist() == [3]
    assert artifact["input_coefficients"][promoted_start:promoted_stop].tolist() == [1.0]
    assert all(value == 0 for value in result["operation_audit"].values())
    stages = [json.loads(line)["stage"] for line in output.getvalue().splitlines()]
    assert "solver-chunk-start" in stages
    assert "solver-chunk-progress" in stages
    assert "carrier-materialization-complete" in stages
    assert "standalone-bridge-published" in stages


def test_fresh_package_observer_counts_only_independent_continuous_routes(
    tmp_path: Path,
) -> None:
    request = _case(tmp_path)
    result = runner.run_carrier_reclosure(request, log_stream=io.StringIO())
    publication = tmp_path / "output"

    allocations = runner.carrier_bridge_allocations_from_publication(
        publication,
        expected_artifact_sha256=result["artifact_sha256"],
    )
    tensors = load_file(str(publication / "bridge.safetensors"), device="cpu")

    assert tuple(value.state_id for value in allocations) == (
        "semantic-carrier-route.c_in",
        "semantic-carrier-route.c_out",
    )
    assert tuple(value.parameter_elements for value in allocations) == (
        int(tensors["input_coefficients"].numel()),
        int(tensors["output_coefficients"].numel()),
    )
    assert sum(value.parameter_elements for value in allocations) == result[
        "continuous_semantic_coefficient_count"
    ]
    assert all(value.dtype == "float32" for value in allocations)
    assert all(
        value.artifact_sha256 == result["artifact_sha256"] for value in allocations
    )
    assert allocations[0].tensor_sha256 == result["materialization_receipt"][
        "input_coefficients_sha256"
    ]
    assert allocations[1].tensor_sha256 == result["materialization_receipt"][
        "output_coefficients_sha256"
    ]
    assert all(value.parent_runtime_dependency is False for value in allocations)
    assert all(value.quantized is False for value in allocations)


def test_package_observer_rejects_wrong_identity_and_tampered_payload(
    tmp_path: Path,
) -> None:
    request = _case(tmp_path)
    result = runner.run_carrier_reclosure(request, log_stream=io.StringIO())
    publication = tmp_path / "output"

    with pytest.raises(
        runner.CarrierReclosureRunnerError, match="different artifact"
    ):
        runner.carrier_bridge_allocations_from_publication(
            publication,
            expected_artifact_sha256=_sha("wrong-carrier-artifact"),
        )

    tensor_path = publication / "bridge.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(runner.CarrierReclosureRunnerError, match="SHA-256 mismatch"):
        runner.carrier_bridge_allocations_from_publication(
            publication,
            expected_artifact_sha256=result["artifact_sha256"],
        )


def test_changed_execution_chunk_size_reuses_canonical_solver_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _case(tmp_path)
    first = runner.run_carrier_reclosure(request, log_stream=io.StringIO())
    request["output_directory"] = str(tmp_path / "second-output")
    original_math_contract = runner._solver_math_contract(request["solver"])
    request["solver"]["rows_per_chunk"] = 97
    assert runner._solver_math_contract(request["solver"]) == original_math_contract

    def unexpected_solve(**_: object):
        raise AssertionError("An unchanged solver chunk was recomputed.")

    monkeypatch.setattr(runner, "_solve_chunk", unexpected_solve)
    output = io.StringIO()
    second = runner.run_carrier_reclosure(request, log_stream=output)

    assert second["artifact_sha256"] == first["artifact_sha256"]
    stages = [json.loads(line)["stage"] for line in output.getvalue().splitlines()]
    assert stages.count("phase-cache-hit") == 3
    assert "solver-chunk-start" not in stages
    assert (tmp_path / "second-output" / "bridge.safetensors").is_file()


def test_zero_incidence_mode_uses_only_the_evicted_direct_route(
    tmp_path: Path,
) -> None:
    request = _case(tmp_path, positive_carrier_incidence=False)
    request["selection"]["allow_positive_incidence"] = False  # type: ignore[index]

    result = runner.run_carrier_reclosure(request, log_stream=io.StringIO())

    assert result["selection_receipt"]["positive_incidence_target_rows_allowed"] is False
    assert result["selection_receipt"]["required_rerouted_donor_ids"] == [3]
    assert result["materialization_receipt"]["required_rerouted_donor_count"] == 1


def test_tampered_cached_trace_metric_fails_before_any_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _case(tmp_path)
    metric_path = Path(
        request["input_solver_evidence"]["tensors"]["path"]  # type: ignore[index]
    )
    metric_path.write_bytes(metric_path.read_bytes() + b"tamper")

    def unexpected_solve(**_: object):
        raise AssertionError("A solver ran after evidence authentication failed.")

    monkeypatch.setattr(runner, "_solve_chunk", unexpected_solve)
    with pytest.raises(runner.CarrierReclosureRunnerError, match="SHA-256 mismatch"):
        runner.run_carrier_reclosure(request, log_stream=io.StringIO())


def test_reserved_carrier_is_fail_closed_unless_authenticated_policy_releases_it(
    tmp_path: Path,
) -> None:
    blocked_root = tmp_path / "blocked"
    blocked_root.mkdir()
    blocked = _case(blocked_root, reserved_carrier=True)
    with pytest.raises(Exception, match="eligible unprotected target rows"):
        runner.run_carrier_reclosure(blocked, log_stream=io.StringIO())

    released_root = tmp_path / "released"
    released_root.mkdir()
    released = _case(
        released_root,
        reserved_carrier=True,
        reclaim_reserved_carrier=True,
    )
    result = runner.run_carrier_reclosure(released, log_stream=io.StringIO())

    assert result["protected_donor_ids"] == [0, 1, 2]
    assert result["reclaimed_reserved_donor_ids"] == [3]
    assert result["status"] == "complete"
