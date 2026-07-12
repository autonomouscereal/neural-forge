from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from neural_forge.surgery_meta_compiler import (
    AttemptVectorizer,
    DatasetIntegrityError,
    FEATURE_VERSION,
    MAX_DATASET_RECORDS,
    MAX_FEATURE_VALUES,
    MAX_SET_CONTEXT,
    POLICY_PURPOSE,
    SurgeryMetaCompilerError,
    TargetModelReferenceError,
    TrainingConfig,
    bootstrap_from_attempt,
    canonical_sha256,
    canonicalize_attempt,
    family_held_out_split,
    load_policy_checkpoint,
    read_jsonl,
    save_policy_checkpoint,
    train_policy,
    write_jsonl,
)


torch = pytest.importorskip("torch")


class _UnsafeCheckpointValue:
    pass


def _digest(*parts: object) -> str:
    return canonical_sha256(list(parts))


def _record(
    family: str,
    attempt: str,
    *,
    accepted: bool,
    savings: int,
    damage: float,
    model_id: str | None = None,
    eval_contract_sha256: str | None = None,
) -> dict:
    model = model_id or f"{family}-model"
    return canonicalize_attempt(
        {
            "schema_version": "neural-forge.surgery-attempt.v1",
            "attempt_id": attempt,
            "model": {
                "family_id": family,
                "model_id": model,
                "architecture_id": f"{family}-architecture",
                "tokenizer_id": f"{family}-tokenizer",
                "modality": "text",
            },
            "features": {
                "static": {
                    "weight": {"mean": 0.2, "std": 0.4},
                    "shape": [4, 8],
                    "role": "unit",
                },
                "observed": {
                    "activation": {"mean": 0.1, "std": 0.2},
                    "jvp": {"mean": 0.05, "std": 0.1},
                    "dual": {"mean": 0.1, "std": 0.1},
                    "output": {"mean": 0.2, "std": 0.3},
                },
            },
            "anchors": {"mode": "soft", "count": 1, "penalty": 0.1},
            "action": {"kind": "merge", "source_count": 2},
            "deployable_byte_savings": savings,
            "metrics": {
                "parent": {"damage": damage},
                "incumbent": {"score": 0.5},
                "candidate": {"score": 0.6},
                "behavior": {"match": float(accepted)},
                "generation": {"exact": float(accepted)},
                "choice": {"top1": float(accepted)},
                "manual": {"approved": accepted},
            },
            "outcome": {"accepted": accepted, "rolled_back": not accepted},
            "projection": {"coefficient": 0.25, "risk": min(1.0, damage)},
            "provenance": {
                "source_attempt_sha256": _digest("attempt", attempt),
                "source_dataset_sha256": _digest("dataset", family),
                "parent_snapshot_sha256": _digest("parent", model),
                "eval_contract_sha256": eval_contract_sha256
                or _digest("eval-contract", family),
                "feature_manifest_sha256": _digest("features", attempt),
                "feature_stage": "pre_decision",
                "feature_encoding": FEATURE_VERSION,
            },
        }
    )


def _records() -> list[dict]:
    records = []
    for family in ("family-a", "family-b", "family-holdout"):
        for index in range(4):
            good = index % 2 == 0
            records.append(
                _record(
                    family,
                    f"{family}-{index}",
                    accepted=good,
                    savings=600 if good else 10,
                    damage=0.05 if good else 0.95,
                )
            )
    return records


def _recanonicalize(record: dict) -> dict:
    changed = copy.deepcopy(record)
    changed["provenance"].pop("record_sha256", None)
    return canonicalize_attempt(changed, verify_hash=False)


def _numpy_state_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_canonical_jsonl_rejects_tampering_and_bootstraps_provenance(
    tmp_path: Path,
) -> None:
    record = _record("family-a", "one", accepted=True, savings=128, damage=0.1)
    path = tmp_path / "attempts.jsonl"
    dataset_hash = write_jsonl([record], path)
    assert len(dataset_hash) == 64
    assert read_jsonl(path) == (record,)

    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["deployable_byte_savings"] = 999
    path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(DatasetIntegrityError, match="record_sha256"):
        read_jsonl(path)

    attempt = {
        "attempt_index": 7,
        "attempt_sha256": "b" * 64,
        "transaction_sha256": "c" * 64,
        "accepted": True,
        "decision": {"metrics": {"manual_score": 1.0}},
        "actions": [
            {"kind": "project", "sources": ["x"], "details": {"score": 0.2}}
        ],
    }
    bootstrapped = bootstrap_from_attempt(
        attempt,
        context={
            "model": record["model"],
            "features": record["features"],
            "metrics": {"parent": {"damage": 0.2}},
            "provenance": {
                "source_dataset_sha256": "d" * 64,
                "parent_snapshot_sha256": "e" * 64,
                "eval_contract_sha256": "f" * 64,
                "feature_manifest_sha256": "1" * 64,
                "feature_stage": "pre_decision",
                "feature_encoding": FEATURE_VERSION,
            },
        },
    )
    assert bootstrapped["action"]["kind"] == "project"
    assert bootstrapped["provenance"]["source_attempt_sha256"] == "b" * 64
    assert bootstrapped["outcome"] == {"accepted": True, "rolled_back": False}


def test_schema_requires_bounded_summary_features_and_complete_provenance() -> None:
    record = _record("family-a", "bounded", accepted=True, savings=10, damage=0.1)

    raw_weights = copy.deepcopy(record)
    raw_weights["features"]["static"]["weight"] = [0.1, 0.2]
    raw_weights["provenance"].pop("record_sha256")
    with pytest.raises(DatasetIntegrityError, match="must be an object"):
        canonicalize_attempt(raw_weights, verify_hash=False)

    oversized_summary = copy.deepcopy(record)
    oversized_summary["features"]["static"]["shape"] = list(
        range(MAX_FEATURE_VALUES + 1)
    )
    oversized_summary["provenance"].pop("record_sha256")
    with pytest.raises(DatasetIntegrityError, match="summary limit"):
        canonicalize_attempt(oversized_summary, verify_hash=False)

    missing_provenance = copy.deepcopy(record)
    missing_provenance["provenance"].pop("source_dataset_sha256")
    missing_provenance["provenance"].pop("record_sha256")
    with pytest.raises(DatasetIntegrityError, match="provenance keys"):
        canonicalize_attempt(missing_provenance, verify_hash=False)

    post_decision = copy.deepcopy(record)
    post_decision["provenance"]["feature_stage"] = "post_decision"
    post_decision["provenance"].pop("record_sha256")
    with pytest.raises(DatasetIntegrityError, match="pre_decision"):
        canonicalize_attempt(post_decision, verify_hash=False)

    hidden_target_state = copy.deepcopy(record)
    hidden_target_state["target_state_dict"] = {"weight": [1.0]}
    hidden_target_state["provenance"].pop("record_sha256")
    with pytest.raises(DatasetIntegrityError, match="extra"):
        canonicalize_attempt(hidden_target_state, verify_hash=False)


def test_vectorizer_excludes_identity_outcome_metrics_and_projection_labels() -> None:
    first = _record(
        "family-a", "first", accepted=True, savings=200, damage=0.01
    )
    second = _record(
        "unseen-family",
        "second",
        accepted=False,
        savings=200,
        damage=0.99,
    )
    second = copy.deepcopy(second)
    second["metrics"]["candidate"] = {"score": -1000.0}
    second["metrics"]["manual"] = {"secret_outcome": "rejected"}
    second["projection"] = {"coefficient": -8.0, "risk": 1.0}
    second = _recanonicalize(second)

    vectorizer = AttemptVectorizer()
    assert np.array_equal(vectorizer.vectorize(first), vectorizer.vectorize(second))


def test_strict_family_model_and_eval_contract_holdout_isolation() -> None:
    records = _records()
    train, holdout = family_held_out_split(records, "family-holdout")
    assert {record["model"]["family_id"] for record in train} == {
        "family-a",
        "family-b",
    }
    assert {record["model"]["family_id"] for record in holdout} == {
        "family-holdout"
    }

    model_leak = copy.deepcopy(records)
    model_leak[0]["model"]["model_id"] = "family-holdout-model"
    model_leak[0] = _recanonicalize(model_leak[0])
    with pytest.raises(DatasetIntegrityError, match="model_ids"):
        family_held_out_split(model_leak, "family-holdout")

    contract_leak = copy.deepcopy(records)
    contract_leak[0]["provenance"]["eval_contract_sha256"] = _digest(
        "eval-contract", "family-holdout"
    )
    contract_leak[0] = _recanonicalize(contract_leak[0])
    with pytest.raises(DatasetIntegrityError, match="eval_contract_sha256"):
        family_held_out_split(contract_leak, "family-holdout")


def test_training_is_deterministic_without_process_global_side_effects() -> None:
    records = _records()
    config = TrainingConfig(
        seed=9, epochs=16, batch_size=3, hidden_dim=12, learning_rate=0.03
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )

    first_policy, first_report = train_policy(
        records, held_out_family="family-holdout", config=config
    )
    second_policy, second_report = train_policy(
        records, held_out_family="family-holdout", config=config
    )

    assert first_report == second_report
    assert first_report.recovery_status.endswith("does not transfer intelligence")
    assert first_report.held_out_model_count == 1
    assert first_report.held_out_eval_contract_count == 1
    assert first_policy.rank(records[-4:]) == second_policy.rank(records[-4:])
    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.get_num_threads() == threads
    if cuda_states:
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_states, torch.cuda.get_rng_state_all())
        )


def test_measured_better_action_ranks_higher_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    records = _records()
    config = TrainingConfig(
        seed=11, epochs=30, batch_size=4, hidden_dim=16, learning_rate=0.03
    )
    policy, report = train_policy(
        records, held_out_family="family-holdout", config=config
    )
    better = _record(
        "family-holdout", "better", accepted=True, savings=1000, damage=0.01
    )
    worse = _record(
        "family-holdout", "worse", accepted=False, savings=1, damage=1.0
    )
    ranking = policy.rank([worse, better])
    assert [item.attempt_id for item in ranking] == ["better", "worse"]
    assert 0.0 <= ranking[0].uncertainty <= 0.7

    path = tmp_path / "policy.pt"
    checkpoint_hash = save_policy_checkpoint(policy, report, path)
    torch_state = torch.random.get_rng_state().clone()
    restored, restored_report, restored_hash = load_policy_checkpoint(path)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert checkpoint_hash == restored_hash
    assert restored_report == report
    assert restored.rank([worse, better]) == ranking

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["purpose"] == POLICY_PURPOSE
    assert set(payload["state_dict"]) == set(policy.model.state_dict())
    assert not ({"records", "outcome", "metrics", "target_model"} & set(payload))


def test_checkpoint_rejects_retained_target_tensors_and_unsafe_pickle(
    tmp_path: Path,
) -> None:
    policy, report = train_policy(
        _records(),
        held_out_family="family-holdout",
        config=TrainingConfig(epochs=2, hidden_dim=8),
    )
    path = tmp_path / "policy.pt"
    save_policy_checkpoint(policy, report, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    payload["target_model_state"] = {"weight": torch.ones(1)}
    retained = tmp_path / "retained-target.pt"
    torch.save(payload, retained)
    with pytest.raises(DatasetIntegrityError, match="retained evidence"):
        load_policy_checkpoint(retained)

    unsafe = tmp_path / "unsafe.pt"
    torch.save({"unsafe": _UnsafeCheckpointValue()}, unsafe)
    with pytest.raises(DatasetIntegrityError, match="weights-only"):
        load_policy_checkpoint(unsafe)


def test_checkpoint_loader_fails_closed_when_weights_only_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "placeholder.pt"
    path.write_bytes(b"not-a-checkpoint")
    calls = []

    def incompatible_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("weights_only is unsupported")

    monkeypatch.setattr(torch, "load", incompatible_loader)
    with pytest.raises(DatasetIntegrityError, match="cannot guarantee"):
        load_policy_checkpoint(path)
    assert len(calls) == 1
    assert calls[0][1]["weights_only"] is True


def test_history_and_context_are_bounded() -> None:
    record = _record("family-a", "limit", accepted=True, savings=1, damage=0.1)
    with pytest.raises(SurgeryMetaCompilerError, match="history limit"):
        family_held_out_split(
            [record] * (MAX_DATASET_RECORDS + 1), "family-holdout"
        )

    policy, _report = train_policy(
        _records(),
        held_out_family="family-holdout",
        config=TrainingConfig(epochs=1, hidden_dim=8),
    )
    candidates = [
        _record(
            "family-holdout",
            f"candidate-{index}",
            accepted=index % 2 == 0,
            savings=index + 1,
            damage=0.1,
        )
        for index in range(MAX_SET_CONTEXT + 7)
    ]
    observed_context_sizes = []
    original_forward = policy.model.forward

    def recording_forward(pair_features, set_features=None):
        observed_context_sizes.append(set_features.shape[1])
        return original_forward(pair_features, set_features)

    policy.model.forward = recording_forward
    policy.rank(candidates, max_candidates=7)
    assert observed_context_sizes
    assert set(observed_context_sizes) == {MAX_SET_CONTEXT}


def test_evidence_adapter_rejects_target_model_objects_and_unknown_context() -> None:
    with pytest.raises(TargetModelReferenceError):
        bootstrap_from_attempt(
            {"attempt_sha256": "d" * 64, "accepted": False},
            context={"model": torch.nn.Linear(2, 2)},
        )
    with pytest.raises(DatasetIntegrityError, match="unsupported keys"):
        bootstrap_from_attempt(
            {"attempt_sha256": "d" * 64, "accepted": False},
            context={"target_model_weights": [1.0, 2.0]},
        )
