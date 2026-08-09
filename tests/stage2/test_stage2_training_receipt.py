from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from neural_forge.stage2_training_receipt import (
    STAGE2_CORPUS_SHARD_PATH_TEMPLATE,
    STAGE2_DESIGN_DOCUMENT_SHA256,
    STAGE2_DESIGN_FILENAME,
    STAGE2_PREREGISTRATION_BYTE_LENGTH,
    STAGE2_PREREGISTRATION_SHA256,
    STAGE2_RUNG_TOKENS,
    Stage2CorpusConsumption,
    Stage2RungRecord,
    Stage2ShardConsumption,
    Stage2TrainingReceiptBuilder,
    Stage2TrainingReceiptError,
    assert_preregistration_unedited,
    corpus_manifest_shard_index,
    corpus_manifest_teacher_sha256,
    load_stage2_training_receipt,
    optimizer_state_digest,
    stage2_preregistration_digest,
    verify_corpus_consumption,
    write_stage2_training_receipt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESIGN = REPOSITORY_ROOT / "docs" / STAGE2_DESIGN_FILENAME
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_TEACHER = "c" * 64


def _consumption(**overrides: Any) -> Stage2CorpusConsumption:
    arguments: dict[str, Any] = {
        "manifest_path": "/mnt/ai_data/neural-forge-stage2/corpus/manifest.json",
        "manifest_sha256": _DIGEST_A,
        "teacher_checkpoint_sha256": _TEACHER,
        "shard_order_seed": 7,
        "shards": (
            Stage2ShardConsumption(
                order_index=0,
                relative_path="shard-000.bin",
                sha256=_DIGEST_B,
                tokens_consumed=1_024,
            ),
        ),
    }
    arguments.update(overrides)
    return Stage2CorpusConsumption(**arguments)


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest = {
        "teacher_checkpoint_sha256": _TEACHER,
        "shards": [
            {
                "relative_path": "shard-000.bin",
                "sha256": _DIGEST_B,
                "token_count": 1_048_576,
            }
        ],
    }
    manifest.update(overrides)
    return manifest


def _builder(**overrides: Any) -> Stage2TrainingReceiptBuilder:
    arguments: dict[str, Any] = {
        "arm_id": "S2-B",
        "init_source_sha256": _DIGEST_A,
        "initialization": {"mode": "fold-init"},
        "teacher_checkpoint_sha256": _TEACHER,
        "corpus": _consumption(),
        "objective_schedule": {"gkd_beta": 0.5},
        "optimizer_configuration": {"optimizer": "AdamW"},
        "trainer_configuration": {"batch_sequences": 16},
        "preregistration": stage2_preregistration_digest(DESIGN),
        "device_inventory": ({"index": 0, "name": "NVIDIA GeForce RTX 3090"},),
        "seeds": {"run_seed": 3},
        "probe_surface": {"prompt_count": 64},
        "route_surface_equivalence": {"input": True},
        "counted_parameters": {"counted_total_parameter_elements": 114_846_592},
        "rungs": [
            Stage2RungRecord(
                rung_tokens=10_000_000,
                tokens_consumed=10_000_384,
                optimizer_steps=610,
                backward_calls=9_760,
                micro_batch_forward_calls=9_760,
                on_policy_steps=0,
                off_policy_steps=610,
                wall_clock_seconds=1.5,
                checkpoint_manifest_sha256=_DIGEST_A,
                checkpoint_directory="/mnt/ai_data/neural-forge-stage2/runs/S2-B",
                optimizer_state={"optimizer_state_sha256": _DIGEST_B},
                free_run_probe={"token_match_fraction": 0.01},
            )
        ],
        "wall_clock_seconds": 2.0,
    }
    arguments.update(overrides)
    return Stage2TrainingReceiptBuilder(**arguments)


def test_the_design_and_its_preregistration_hash_to_their_sealed_digests() -> None:
    """Section 8 says it may not be edited after sealing; this is that check."""

    observed = stage2_preregistration_digest(DESIGN)
    assert observed["preregistration_sha256"] == STAGE2_PREREGISTRATION_SHA256
    assert (
        observed["preregistration_byte_length"]
        == STAGE2_PREREGISTRATION_BYTE_LENGTH
    )
    assert observed["design_document_sha256"] == STAGE2_DESIGN_DOCUMENT_SHA256
    assert assert_preregistration_unedited(DESIGN) == observed


def test_an_edited_preregistration_is_refused(tmp_path: Path) -> None:
    edited = tmp_path / STAGE2_DESIGN_FILENAME
    raw = DESIGN.read_bytes()
    edited.write_bytes(
        raw.replace(b"wins >= 2 of 3 rungs", b"wins >= 1 of 3 rungs")
    )
    assert hashlib.sha256(edited.read_bytes()).hexdigest() != (
        STAGE2_DESIGN_DOCUMENT_SHA256
    )
    with pytest.raises(Stage2TrainingReceiptError, match="section 8 has changed"):
        assert_preregistration_unedited(edited)


def test_rung_tokens_are_the_designs_ladder() -> None:
    assert STAGE2_RUNG_TOKENS == (10_000_000, 100_000_000, 1_000_000_000)


def test_a_receipt_that_cannot_name_its_data_is_refused() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="at least one consumed"):
        _consumption(shards=())


def test_shard_order_must_be_dense_and_ascending() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="dense and ascending"):
        _consumption(
            shards=(
                Stage2ShardConsumption(
                    order_index=1,
                    relative_path="shard-000.bin",
                    sha256=_DIGEST_B,
                    tokens_consumed=8,
                ),
            )
        )


def test_corpus_verification_reads_lane_twos_actual_manifest_shape() -> None:
    """Lane 2 seals shards by index and the teacher under ``run``."""

    lane_two = {
        "schema": "neural-forge-stage2-corpus-manifest-v1",
        "run": {"stage": "stage2", "teacher_shard_sha256": _TEACHER},
        "shards": [
            {
                "shard_index": 0,
                "split": "train",
                "tokens_sha256": _DIGEST_B,
                "token_count": 1_048_576,
            }
        ],
    }
    assert corpus_manifest_teacher_sha256(lane_two) == _TEACHER
    assert corpus_manifest_shard_index(lane_two) == {
        STAGE2_CORPUS_SHARD_PATH_TEMPLATE.format(index=0): _DIGEST_B
    }
    report = verify_corpus_consumption(
        _consumption(
            shards=(
                Stage2ShardConsumption(
                    order_index=0,
                    relative_path=STAGE2_CORPUS_SHARD_PATH_TEMPLATE.format(index=0),
                    sha256=_DIGEST_B,
                    tokens_consumed=1_024,
                ),
            )
        ),
        manifest=lane_two,
        manifest_file_sha256=_DIGEST_A,
    )
    assert report["corpus_manifest_verified"] is True


def test_a_manifest_that_names_no_teacher_is_refused() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="does not name the teacher"):
        corpus_manifest_teacher_sha256({"shards": []})


def test_a_shard_row_with_neither_a_path_nor_an_index_is_refused() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="neither a path nor"):
        corpus_manifest_shard_index({"shards": [{"token_count": 4}]})


def test_corpus_verification_accepts_a_matching_manifest() -> None:
    report = verify_corpus_consumption(
        _consumption(),
        manifest=_manifest(),
        manifest_file_sha256=_DIGEST_A,
    )
    assert report == {
        "corpus_manifest_verified": True,
        "manifest_shard_count": 1,
        "consumed_shard_count": 1,
        "teacher_matches_manifest": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest-bytes", "Corpus manifest bytes differ"),
        ("teacher", "differs from the corpus manifest's teacher"),
        ("missing-shard", "is not in the manifest"),
        ("shard-digest", "differs from the manifest digest"),
    ],
)
def test_corpus_verification_fails_closed(mutation: str, message: str) -> None:
    consumption = _consumption()
    manifest = _manifest()
    manifest_sha256 = _DIGEST_A
    if mutation == "manifest-bytes":
        manifest_sha256 = "d" * 64
    elif mutation == "teacher":
        manifest["teacher_checkpoint_sha256"] = "e" * 64
    elif mutation == "missing-shard":
        manifest["shards"] = [
            {
                "relative_path": "shard-999.bin",
                "sha256": _DIGEST_B,
                "token_count": 1,
            }
        ]
    else:
        manifest["shards"] = [
            {
                "relative_path": "shard-000.bin",
                "sha256": "f" * 64,
                "token_count": 1,
            }
        ]
    with pytest.raises(Stage2TrainingReceiptError, match=message):
        verify_corpus_consumption(
            consumption,
            manifest=manifest,
            manifest_file_sha256=manifest_sha256,
        )


def test_receipt_seals_reloads_and_refuses_tampering(tmp_path: Path) -> None:
    receipt = _builder().seal()
    destination = tmp_path / "S2-B.json"
    seal = write_stage2_training_receipt(destination, receipt)
    reloaded = load_stage2_training_receipt(
        destination,
        expected_receipt_sha256=seal,
    )
    assert reloaded["receipt_sha256"] == seal
    assert reloaded["stage"] == "stage2"
    assert reloaded["provenance"]["stage1_artifacts_modified"] is False
    assert reloaded["metric_policy"]["teacher_forced_top1_used"] is False
    assert reloaded["metric_policy"]["terminal_reconstruction_used"] is False
    assert reloaded["diagnostic_held_out_nll"]["diagnostic_only"] is True
    assert reloaded["gpu_hours"] == pytest.approx(2.0 / 3600.0)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["rungs"][0]["tokens_consumed"] = 1
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage2TrainingReceiptError, match="self-seal does not verify"):
        load_stage2_training_receipt(destination)


def test_a_receipt_for_a_different_run_is_refused(tmp_path: Path) -> None:
    receipt = _builder().seal()
    destination = tmp_path / "S2-B.json"
    write_stage2_training_receipt(destination, receipt)
    with pytest.raises(Stage2TrainingReceiptError, match="different sealed run"):
        load_stage2_training_receipt(
            destination,
            expected_receipt_sha256="0" * 64,
        )


def test_writing_refuses_an_unverifiable_seal(tmp_path: Path) -> None:
    receipt = dict(_builder().seal())
    receipt["arm_id"] = "S2-D"
    with pytest.raises(Stage2TrainingReceiptError, match="seal does not verify"):
        write_stage2_training_receipt(tmp_path / "bad.json", receipt)


def test_a_receipt_without_a_rung_or_the_preregistration_is_refused() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="at least one ladder rung"):
        _builder(rungs=[]).seal()
    with pytest.raises(Stage2TrainingReceiptError, match="sealed section-8"):
        _builder(preregistration={"preregistration_sha256": "0" * 64}).seal()


def test_a_teacher_that_disagrees_with_the_corpus_is_refused() -> None:
    with pytest.raises(Stage2TrainingReceiptError, match="differs from its corpus"):
        _builder(teacher_checkpoint_sha256="9" * 64).seal()


def test_optimizer_state_digest_is_stable_and_moves_with_the_state() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    before = optimizer_state_digest(optimizer)
    assert before["optimizer_state_element_count"] == 0
    parameter.grad = torch.ones(4)
    optimizer.step()
    after = optimizer_state_digest(optimizer)
    assert after["optimizer_state_sha256"] != before["optimizer_state_sha256"]
    # exp_avg (4) + exp_avg_sq (4) + the scalar step tensor.
    assert after["optimizer_state_element_count"] == 9
    assert optimizer_state_digest(optimizer) == after
