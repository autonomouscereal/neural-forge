from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_interface_frontier as frontier  # noqa: E402
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL,
    Lfm2CausalTrieEvidence,
    Lfm2ChildCausalResidualEvidence,
    Lfm2DenseMappedRootHybridCausalEvidence,
    Lfm2TerminalCacheEvidence,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_dense_mapped_root_hybrid_causal_evidence,
)
from neural_forge.hf_lfm2_mapped_causal_target_capsule import (  # noqa: E402
    Lfm2MappedCausalTargetCapsuleError,
    Lfm2MappedCausalTargetEvidence,
    authenticate_lfm2_mapped_causal_target_capsule,
    build_lfm2_mapped_causal_target_evidence,
    collect_lfm2_mapped_causal_target_evidence,
    load_lfm2_mapped_causal_target_capsule,
    publish_lfm2_mapped_causal_target_capsule,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seal(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: canonical_sha256(body)}


class _Tokenizer:
    def __init__(self) -> None:
        self.surfaces = ["<pad>", "<bos>", "<eos>", "x"]
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = None
        self.all_special_ids = [0, 1, 2]

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.surfaces)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.surfaces[token_id]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


def _target_package(path: Path) -> None:
    path.mkdir()
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )


def _root_record(index: int, token_id: int) -> dict[str, Any]:
    return {
        "terminal_row_index": index,
        "batch_index": 0,
        "phase": "cached_prefill",
        "phase_row_index": index,
        "donor_boundary_position": 0,
        "target_boundary_position": 0,
        "donor_prefix_token_ids": [token_id],
        "target_prefix_token_ids": [token_id],
        "semantic_alignment_sha256": _sha(f"root-alignment-{index}"),
    }


class _FakeSelectiveSession(AbstractContextManager):
    def __init__(
        self,
        input_sha256: str,
        logits: Any,
        receipt: dict[str, Any],
    ) -> None:
        self.batch_references = (
            SimpleNamespace(index=0, input_sha256=input_sha256),
        )
        self._logits = logits
        self._receipt = receipt
        self.loads = 0
        self.releases = 0
        self.verifications = 0

    def __enter__(self) -> "_FakeSelectiveSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def load_selected_tensors(self, index: int, names: tuple[str, ...]) -> Any:
        assert index == 0
        assert names == ("cached_prefill.logits",)
        self.loads += 1
        return SimpleNamespace(
            tensors={"cached_prefill.logits": self._logits.unsqueeze(0).clone()},
            logical_bytes=int(self._logits.numel() * self._logits.element_size()),
            selection_sha256=_sha("logits-only-selection"),
        )

    def release_selected_tensors(self, _payload: Any) -> None:
        self.releases += 1

    def verify(self) -> None:
        self.verifications += 1

    def as_dict(self) -> dict[str, Any]:
        return dict(self._receipt)


class _FakeCache:
    def __init__(
        self,
        *,
        cache_key: str,
        parent_fingerprint: str,
        session: _FakeSelectiveSession,
    ) -> None:
        self.cache_key = cache_key
        self.parent_fingerprint = parent_fingerprint
        self._session = session
        self.sessions = 0

    def selective_authenticated_session(self) -> _FakeSelectiveSession:
        self.sessions += 1
        return self._session


def _case(tmp_path: Path) -> dict[str, Any]:
    tokenizer = _Tokenizer()
    target_package = tmp_path / "target-tokenizer"
    _target_package(target_package)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        target_package,
        donor_row_capacity=6,
        target_row_capacity=4,
    )
    donor_canonical_input_sha256 = _sha("donor-canonical-cache-input")
    donor_logits = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        ],
        dtype=torch.float32,
    )
    hidden_rows = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
        dtype=torch.float32,
    )
    roots = (_root_record(0, 1), _root_record(1, 3))
    canonical_roots = frontier._causal_root_evidence_rows(
        roots,
        donor_row_count=6,
        target_row_count=4,
        terminal_row_count=2,
    )
    session_receipt = {
        "cache_key": _sha("cache-key"),
        "session_sha256": _sha("session"),
        "manifest_sha256": _sha("manifest"),
        "completion_sha256": _sha("completion"),
    }
    selected_batches = [
        {
            "batch_index": 0,
            "input_sha256": donor_canonical_input_sha256,
            "selection_sha256": _sha("terminal-selection"),
            "logical_bytes": int(
                donor_logits.numel() * donor_logits.element_size()
                + hidden_rows.numel() * hidden_rows.element_size()
            ),
            "terminal_rows": 2,
            "phases": [
                {
                    "phase": "cached_prefill",
                    "rows": 2,
                    "hidden_name": "cached_prefill.output.lm_head_input",
                    "logits_name": "cached_prefill.logits",
                    "causal_root_rows": 2,
                    "rows_without_exact_target_boundary": 0,
                }
            ],
        }
    ]
    zero_audit = {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_forward_calls": 0,
        "child_generate_calls": 0,
        "network_access": False,
    }
    terminal_body = {
        "schema": LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        "status": "complete-authenticated-terminal-cache-selection",
        "cache_key": session_receipt["cache_key"],
        "parent_identity_sha256": _sha("parent"),
        "selective_session_sha256": session_receipt["session_sha256"],
        "manifest_sha256": session_receipt["manifest_sha256"],
        "completion_sha256": session_receipt["completion_sha256"],
        "batch_count": 1,
        "selected_batches": selected_batches,
        "terminal_row_count": 2,
        "hidden_size": 3,
        "donor_vocabulary_size": 6,
        "target_vocabulary_size": 4,
        "hidden_rows_sha256": frontier._tensor_sha256(hidden_rows),
        "donor_logits_sha256": frontier._tensor_sha256(donor_logits),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "semantic_alignment_sha256": _sha("all-alignments"),
        "semantic_alignment_count": 2,
        "tokenizer_live_session_sha256": _sha("tokenizer-session"),
        "causal_root_row_count": 2,
        "causal_root_rows_sha256": canonical_sha256(list(canonical_roots)),
        "rows_without_exact_target_boundary_excluded_from_causal_replay": 0,
        "all_terminal_rows_retained_in_diagnostic_tensors": True,
        "all_eligible_exact_boundary_rows_retained_for_causal_replay": True,
        "unpaired_cached_prefill_logits_excluded": True,
        "sealed_parent_cache_reused_without_recapture": True,
        "operation_audit": zero_audit,
    }
    terminal = Lfm2TerminalCacheEvidence(
        hidden_rows=hidden_rows,
        donor_logits=donor_logits,
        causal_root_records=roots,
        receipt=_seal(terminal_body, "evidence_sha256"),
    )

    root_plans = []
    for index in range(2):
        plan_body = {
            "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
            "root_index": index,
            "terminal_row_index": index,
            "target_root_token_ids": list(roots[index]["target_prefix_token_ids"]),
            "append_only_candidates": [],
        }
        root_plans.append(_seal(plan_body, "root_transition_plan_sha256"))
    transition_body = {
        "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
        "status": "synthetic-contextual-roots",
        "root_transition_plans": root_plans,
        "target_row_count": 4,
    }
    transition = _seal(transition_body, "plan_sha256")
    prefixes = ((1,), (3,), (1, 3))
    source_logits = torch.tensor(
        [
            [0.0, 0.0, 99.0, 0.0],
            [0.0, -77.0, 0.0, 0.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    source_support = torch.tensor(
        [
            [False, False, True, False],
            [False, True, False, False],
            [True, False, True, False],
        ],
        dtype=torch.bool,
    )
    terminal_receipt = terminal.as_dict()
    trie_body = {
        "schema": LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        "status": "cache-only-bounded-contextual-causal-conditionals-compiled",
        "parent_terminal_evidence_sha256": terminal_receipt["evidence_sha256"],
        **{
            name: terminal_receipt[name]
            for name in (
                "cache_key",
                "parent_identity_sha256",
                "selective_session_sha256",
                "manifest_sha256",
                "completion_sha256",
                "semantic_alignment_sha256",
                "tokenizer_live_session_sha256",
                "causal_root_rows_sha256",
                "bridge_artifact_sha256",
                "bridge_profile_sha256",
                "donor_tokenizer_semantics_sha256",
                "target_tokenizer_semantics_sha256",
            )
        },
        "contextual_transition_plan_sha256": transition["plan_sha256"],
        "contextual_transition_plan": transition,
        "target_prefix_rows_sha256": canonical_sha256(
            [list(value) for value in prefixes]
        ),
        "target_logits_sha256": frontier._tensor_sha256(source_logits),
        "target_logit_support_mask_sha256": frontier._tensor_sha256(
            source_support
        ),
        "root_row_count": 2,
        "internal_prefix_row_count": 1,
        "total_row_count": 3,
        "missing_cache_statistics": ["synthetic-unexamined-branches"],
        "node_records": [],
    }
    trie = Lfm2CausalTrieEvidence(
        target_prefix_token_ids=prefixes,
        target_logits=source_logits,
        target_logit_support_mask=source_support,
        receipt=_seal(trie_body, "result_sha256"),
    )
    session = _FakeSelectiveSession(
        donor_canonical_input_sha256,
        donor_logits,
        session_receipt,
    )
    cache = _FakeCache(
        cache_key=session_receipt["cache_key"],
        parent_fingerprint=terminal_receipt["parent_identity_sha256"],
        session=session,
    )
    return {
        "bridge": bridge,
        "tokenizer": tokenizer,
        "target_package": target_package,
        "cache": cache,
        "session": session,
        "terminal": terminal,
        "trie": trie,
        "donor_logits": donor_logits,
    }


def _collect(case: dict[str, Any]) -> Any:
    return collect_lfm2_mapped_causal_target_evidence(
        case["cache"],
        terminal_evidence=case["terminal"],
        causal_trie_evidence=case["trie"],
        bridge=case["bridge"],
    )


def test_cache_only_producer_opens_once_and_projects_all_root_target_cells(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    evidence = _collect(case)

    assert case["cache"].sessions == 1
    assert case["session"].loads == 1
    assert case["session"].releases == 1
    assert case["session"].verifications == 1
    assert evidence.root_terminal_row_indices.tolist() == [0, 1]
    assert evidence.target_to_donor_row_indices.tolist() == [0, 1, 2, 3]
    assert torch.equal(
        evidence.root_target_logits,
        case["donor_logits"][:, :4],
    )
    audit = evidence.receipt["operation_audit"]
    assert audit["parent_cache_selective_authenticated_sessions"] == 1
    assert audit["parent_model_forward_calls"] == 0
    assert audit["parent_trace_recaptures"] == 0
    assert audit["training_steps"] == 0


def test_cache_only_producer_accepts_pinned_terminal_receipt_without_tensors(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    evidence = collect_lfm2_mapped_causal_target_evidence(
        case["cache"],
        terminal_evidence=case["terminal"].as_dict(),
        causal_trie_evidence=case["trie"],
        bridge=case["bridge"],
    )

    assert case["cache"].sessions == 1
    assert evidence.receipt["operation_audit"][
        "parent_cache_selective_authenticated_sessions"
    ] == 1
    assert torch.equal(evidence.root_target_logits, case["donor_logits"][:, :4])


def test_builder_rejects_projection_record_outside_terminal_batch_order(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    source = _collect(case)
    record_body = dict(source.receipt["projection_records"][0])
    record_body.pop("projection_sha256")
    record_body["input_sha256"] = _sha("different-donor-cache-input")
    tampered_record = _seal(record_body, "projection_sha256")

    with pytest.raises(
        Lfm2MappedCausalTargetCapsuleError,
        match="terminal, trie, bridge, or session lineage",
    ):
        build_lfm2_mapped_causal_target_evidence(
            case["donor_logits"],
            terminal_evidence=case["terminal"],
            causal_trie_evidence=case["trie"],
            bridge=case["bridge"],
            selective_session_receipt=case["session"].as_dict(),
            projection_records=(tampered_record,),
            selected_tensor_payload_count=1,
            selected_tensor_logical_bytes=source.receipt[
                "selected_tensor_logical_bytes"
            ],
        )


def test_builder_rejects_arbitrary_logits_under_authentic_cache_lineage(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    source = _collect(case)
    arbitrary = case["donor_logits"].clone()
    arbitrary[0, 0] += 1.0

    with pytest.raises(
        Lfm2MappedCausalTargetCapsuleError,
        match="terminal, trie, bridge, or session lineage",
    ):
        build_lfm2_mapped_causal_target_evidence(
            arbitrary,
            terminal_evidence=case["terminal"],
            causal_trie_evidence=case["trie"],
            bridge=case["bridge"],
            selective_session_receipt=case["session"].as_dict(),
            projection_records=source.receipt["projection_records"],
            selected_tensor_payload_count=1,
            selected_tensor_logical_bytes=source.receipt[
                "selected_tensor_logical_bytes"
            ],
        )


def test_evidence_rejects_invalid_projection_descriptor_sha256(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    source = _collect(case)
    body = source.as_dict()
    body.pop("evidence_sha256")
    record_body = dict(body["projection_records"][0])
    record_body.pop("projection_sha256")
    target_descriptor = dict(record_body["target_logits"])
    target_descriptor["sha256"] = "0" * 64
    record_body["target_logits"] = target_descriptor
    body["projection_records"] = [
        _seal(record_body, "projection_sha256")
    ]

    with pytest.raises(
        Lfm2MappedCausalTargetCapsuleError,
        match="receipt or tensor lineage",
    ):
        Lfm2MappedCausalTargetEvidence(
            root_target_logits=source.root_target_logits,
            root_terminal_row_indices=source.root_terminal_row_indices,
            target_to_donor_row_indices=source.target_to_donor_row_indices,
            receipt=_seal(body, "evidence_sha256"),
        )


def test_hybrid_compiler_reauthenticates_live_mapped_tensor_bytes(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    evidence = _collect(case)
    evidence.root_target_logits[0, 0] += 1.0

    with pytest.raises(
        frontier.Lfm2InterfaceFrontierError,
        match="failed live reauthentication",
    ):
        compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
            case["trie"],
            evidence,
        )


def test_hybrid_compiler_reauthenticates_live_trie_tensor_bytes(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    evidence = _collect(case)
    case["trie"].target_logits[0, 0] += 1.0

    with pytest.raises(
        frontier.Lfm2InterfaceFrontierError,
        match="failed live reauthentication",
    ):
        compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
            case["trie"],
            evidence,
        )


def test_donor_canonical_producer_succeeds_when_unrelated_calibration_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer is bound to cache order, never calibration-record identity."""

    case = _case(tmp_path)
    unrelated_calibration_inputs = tuple(
        _sha(f"unrelated-target-calibration-{index}") for index in range(28)
    )
    terminal_inputs = tuple(
        row["input_sha256"]
        for row in case["terminal"].as_dict()["selected_batches"]
    )
    assert len(unrelated_calibration_inputs) != len(terminal_inputs)
    assert set(unrelated_calibration_inputs).isdisjoint(terminal_inputs)

    producer_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "build_lfm2_mapped_causal_target_capsule.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mapped_target_direct_producer_test",
        producer_path,
    )
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    producer_parameters = inspect.signature(producer.build).parameters
    assert "target_token_calibration_json" not in producer_parameters
    assert "expected_trace_adapter_evidence_sha256" not in producer_parameters
    parser_destinations = {action.dest for action in producer._parser()._actions}
    assert "target_token_calibration_json" not in parser_destinations
    assert "expected_trace_adapter_evidence_sha256" not in parser_destinations

    def write_receipt(path: Path, value: dict[str, Any]) -> str:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    terminal_receipt = case["terminal"].as_dict()
    trie_receipt = case["trie"].as_dict()
    terminal_path = tmp_path / "terminal.json"
    trie_path = tmp_path / "causal-trie.json"
    terminal_file_sha256 = write_receipt(terminal_path, terminal_receipt)
    trie_file_sha256 = write_receipt(trie_path, trie_receipt)

    cache = case["cache"]
    cache.batch_count = len(terminal_inputs)
    cache.manifest = {
        "corpus": {"ordered_batch_input_sha256": list(terminal_inputs)}
    }
    monkeypatch.setattr(
        producer,
        "Lfm2CircuitTraceCache",
        SimpleNamespace(open=lambda *_args, **_kwargs: cache),
    )
    monkeypatch.setattr(
        producer,
        "load_lfm2_direct_local_tokenizer",
        lambda *_args, **_kwargs: case["tokenizer"],
    )
    monkeypatch.setattr(
        producer,
        "restore_lfm2_causal_trie_evidence_from_receipt",
        lambda _receipt: case["trie"],
    )

    result = producer.build(
        trace_cache_directory=tmp_path / "donor-canonical-cache",
        expected_cache_key=terminal_receipt["cache_key"],
        expected_parent_identity_sha256=terminal_receipt[
            "parent_identity_sha256"
        ],
        expected_cache_manifest_sha256=terminal_receipt["manifest_sha256"],
        expected_cache_completion_sha256=terminal_receipt[
            "completion_sha256"
        ],
        terminal_evidence_json=terminal_path,
        expected_terminal_evidence_file_sha256=terminal_file_sha256,
        expected_terminal_evidence_sha256=terminal_receipt["evidence_sha256"],
        causal_trie_json=trie_path,
        expected_causal_trie_file_sha256=trie_file_sha256,
        expected_causal_trie_sha256=trie_receipt["result_sha256"],
        donor_tokenizer_directory=case["target_package"],
        target_tokenizer_directory=case["target_package"],
        bridge_directory=case["target_package"],
        expected_bridge_artifact_sha256=case["bridge"].artifact_sha256,
        expected_bridge_profile_sha256=case["bridge"].profile_fingerprint,
        capsule_directory=tmp_path / "producer-capsule",
        reference_document=tmp_path / "producer-reference.json",
    )

    assert result["status"] == "immutable-mapped-causal-target-capsule-published"
    assert result["operation_audit"][
        "parent_cache_selective_authenticated_sessions"
    ] == 1
    assert result["operation_audit"]["parent_trace_recaptures"] == 0
    assert case["cache"].sessions == 1


def test_capsule_is_atomic_pinned_and_fails_closed_on_tamper_and_replacement(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    evidence = _collect(case)
    output = tmp_path / "mapped-capsule"

    reference = publish_lfm2_mapped_causal_target_capsule(
        evidence,
        output_directory=output,
    )
    assert {path.name for path in output.iterdir()} == {
        "capsule-manifest.json",
        "capsule-receipt.json",
        "mapped-causal-targets.safetensors",
    }
    authenticated = authenticate_lfm2_mapped_causal_target_capsule(reference)
    loaded = load_lfm2_mapped_causal_target_capsule(reference)
    assert authenticated["evidence_receipt"]["evidence_sha256"] == (
        evidence.receipt["evidence_sha256"]
    )
    assert torch.equal(loaded.evidence.root_target_logits, evidence.root_target_logits)
    with pytest.raises(Lfm2MappedCausalTargetCapsuleError, match="replace"):
        publish_lfm2_mapped_causal_target_capsule(
            evidence,
            output_directory=output,
        )

    stale = dataclasses.replace(reference, tensor_file_sha256=_sha("stale-tensor"))
    with pytest.raises(Lfm2MappedCausalTargetCapsuleError, match="pins"):
        authenticate_lfm2_mapped_causal_target_capsule(stale)

    tensor_path = output / "mapped-causal-targets.safetensors"
    with tensor_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 0x01]))
    with pytest.raises(Lfm2MappedCausalTargetCapsuleError, match="pins"):
        load_lfm2_mapped_causal_target_capsule(reference)


def test_publish_reauthenticates_live_evidence_before_creating_artifact(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    evidence = _collect(case)
    evidence.root_target_logits[0, 0] += 1.0
    output = tmp_path / "must-not-exist"

    with pytest.raises(
        Lfm2MappedCausalTargetCapsuleError,
        match="receipt or tensor lineage",
    ):
        publish_lfm2_mapped_causal_target_capsule(
            evidence,
            output_directory=output,
        )

    assert not output.exists()


def test_capsule_round_trip_accepts_many_target_rows_to_one_donor_row(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    source = _collect(case)
    duplicate_mapping = torch.tensor([0, 1, 1, 3], dtype=torch.int64)
    duplicate_logits = case["donor_logits"].index_select(
        1,
        duplicate_mapping,
    ).contiguous()
    body = source.as_dict()
    body.pop("evidence_sha256")
    body["mapping_sha256"] = canonical_sha256(duplicate_mapping.tolist())
    body["target_to_donor_row_indices_sha256"] = frontier._tensor_sha256(
        duplicate_mapping
    )
    body["root_target_logits_sha256"] = frontier._tensor_sha256(
        duplicate_logits
    )
    duplicate_mapping_sha256 = canonical_sha256(duplicate_mapping.tolist())
    body["mapping_sha256"] = duplicate_mapping_sha256
    updated_projection_records = []
    for record in body["projection_records"]:
        projection_body = dict(record)
        projection_body.pop("projection_sha256")
        projection_body["mapping_sha256"] = duplicate_mapping_sha256
        start = projection_body["terminal_row_start"]
        end = projection_body["terminal_row_end"]
        projected_rows = duplicate_logits[start:end].contiguous()
        projection_body["target_logits"] = {
            "dtype": str(projected_rows.dtype),
            "shape": list(projected_rows.shape),
            "numel": int(projected_rows.numel()),
            "size_bytes": int(
                projected_rows.numel() * projected_rows.element_size()
            ),
            "sha256": frontier._tensor_sha256(projected_rows),
        }
        updated_projection_records.append(
            _seal(projection_body, "projection_sha256")
        )
    body["projection_records"] = updated_projection_records
    duplicate = Lfm2MappedCausalTargetEvidence(
        root_target_logits=duplicate_logits,
        root_terminal_row_indices=source.root_terminal_row_indices,
        target_to_donor_row_indices=duplicate_mapping,
        receipt=_seal(body, "evidence_sha256"),
    )

    reference = publish_lfm2_mapped_causal_target_capsule(
        duplicate,
        output_directory=tmp_path / "duplicate-map-capsule",
    )
    loaded = load_lfm2_mapped_causal_target_capsule(reference)

    assert loaded.evidence.target_to_donor_row_indices.tolist() == [0, 1, 1, 3]
    assert torch.equal(loaded.evidence.root_target_logits, duplicate_logits)


def test_capsule_rejects_symlink_alias_when_platform_allows_it(tmp_path: Path) -> None:
    case = _case(tmp_path)
    reference = publish_lfm2_mapped_causal_target_capsule(
        _collect(case),
        output_directory=tmp_path / "capsule",
    )
    alias = tmp_path / "alias"
    try:
        os.symlink(reference.path, alias, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create directory symlinks.")
    linked = dataclasses.replace(reference, path=alias)
    with pytest.raises(Lfm2MappedCausalTargetCapsuleError, match="link"):
        authenticate_lfm2_mapped_causal_target_capsule(linked)


def test_dense_mapped_hybrid_overrides_roots_and_preserves_internal_bytes(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    mapped = _collect(case)
    source = case["trie"]

    hybrid = compile_lfm2_dense_mapped_root_hybrid_causal_evidence(source, mapped)

    assert isinstance(hybrid, Lfm2DenseMappedRootHybridCausalEvidence)
    assert bool(hybrid.target_logit_support_mask[:2].all()) is True
    root_source_support = source.target_logit_support_mask[:2]
    assert torch.equal(
        hybrid.target_logits[:2][root_source_support],
        source.target_logits[:2][root_source_support],
    )
    assert torch.equal(
        hybrid.target_logits[:2][~root_source_support],
        mapped.root_target_logits[~root_source_support],
    )
    assert torch.equal(hybrid.target_logits[2:], source.target_logits[2:])
    assert torch.equal(
        hybrid.target_logit_support_mask[2:],
        source.target_logit_support_mask[2:],
    )
    receipt = hybrid.as_dict()
    assert receipt["inverse_output_plan_used"] is False
    assert receipt["all_donor_mass_bucket_aggregation_used"] is False
    assert receipt["context_free_dense_root_prior_used"] is False
    assert receipt["operation_audit"]["training_steps"] == 0


def test_dense_mapped_hybrid_reauthenticates_each_root_plan_seal(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    mapped = _collect(case)
    transition = case["trie"].receipt["contextual_transition_plan"]
    transition["root_transition_plans"][0]["terminal_row_index"] = 1

    with pytest.raises(
        frontier.Lfm2InterfaceFrontierError,
        match="failed live reauthentication",
    ):
        compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
            case["trie"],
            mapped,
        )


def test_dense_mapped_hybrid_typed_boundary_rejects_nested_root_plan_tamper(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    hybrid = compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
        case["trie"],
        _collect(case),
    )
    body = json.loads(json.dumps(hybrid.as_dict()))
    body.pop("result_sha256")
    transition = body["contextual_transition_plan"]
    transition["root_transition_plans"][0]["terminal_row_index"] = 1
    transition_body = dict(transition)
    transition_body.pop("plan_sha256")
    transition["plan_sha256"] = canonical_sha256(transition_body)
    body["contextual_transition_plan_sha256"] = transition["plan_sha256"]

    with pytest.raises(
        frontier.Lfm2InterfaceFrontierError,
        match="receipt, lineage, or tensors",
    ):
        Lfm2DenseMappedRootHybridCausalEvidence(
            target_prefix_token_ids=hybrid.target_prefix_token_ids,
            target_logits=hybrid.target_logits,
            target_logit_support_mask=hybrid.target_logit_support_mask,
            receipt=_seal(body, "result_sha256"),
        )


def _child_evidence(
    hidden_rows: Any,
    trie: Lfm2CausalTrieEvidence,
    structural_sha256: str,
) -> Lfm2ChildCausalResidualEvidence:
    receipt = trie.as_dict()
    state = _sha("child-state")
    body = {
        "schema": frontier.LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
        "status": "complete-target-native-child-residual-replay",
        "evidence_role": "mapped-root-unit-test",
        "trie_evidence_sha256": receipt["result_sha256"],
        **{
            name: receipt[name]
            for name in (
                "parent_terminal_evidence_sha256",
                "cache_key",
                "parent_identity_sha256",
                "selective_session_sha256",
                "manifest_sha256",
                "completion_sha256",
                "semantic_alignment_sha256",
                "tokenizer_live_session_sha256",
                "causal_root_rows_sha256",
                "contextual_transition_plan_sha256",
                "bridge_artifact_sha256",
                "bridge_profile_sha256",
                "donor_tokenizer_semantics_sha256",
                "target_tokenizer_semantics_sha256",
                "target_prefix_rows_sha256",
            )
        },
        "child_hidden_rows_sha256": frontier._tensor_sha256(hidden_rows),
        "causal_input_table_sha256": structural_sha256,
        "causal_body_inventory_sha256": _sha("child-body"),
        "causal_input_table_after_sha256": structural_sha256,
        "causal_body_inventory_after_sha256": _sha("child-body"),
        "row_count": int(hidden_rows.shape[0]),
        "hidden_size": int(hidden_rows.shape[1]),
        "child_model_state_before_sha256": state,
        "child_model_state_after_sha256": state,
        "child_state_preserved": True,
        "input_and_body_coordinate_preserved": True,
        "batch_size": 1,
        "full_prefix_use_cache_false": True,
        "position_origin": 0,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": int(hidden_rows.shape[0]),
            "child_generate_calls": 0,
            "network_access": False,
        },
    }
    return Lfm2ChildCausalResidualEvidence(
        hidden_rows=hidden_rows,
        receipt=_seal(body, "evidence_sha256"),
    )


def test_dense_mapped_roots_feed_tiny_adaptive_bridge_without_training(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    hybrid = compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
        case["trie"],
        _collect(case),
    )
    structural = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    hidden = torch.eye(3, dtype=torch.float32)
    child = _child_evidence(
        hidden,
        hybrid,
        frontier._tensor_sha256(structural),
    )

    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child,
        hybrid,
        parent_terminal_evidence=case["terminal"].as_dict(),
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=0.01,
        residual_scale_ppm=1_000_000,
        unsupported_logit_policy=(
            LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
        ),
        solve_device="cpu",
    )

    receipt = result.as_dict()
    assert receipt["fully_unsupported_output_row_count"] == 0
    assert receipt["input_supported_logit_cell_count"] >= 8
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["parent_forward_calls"] == 0
