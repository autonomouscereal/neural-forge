"""Sealed stage-2 training receipts: one JSON per run, one addendum per rung.

``docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`` section 5.1 lists what a
stage-2 receipt must carry, and section 8 is a pre-registration whose digest
"must be recorded in the first stage-2 training receipt before any training
step runs".  This module owns both: the canonical-JSON seal the estate already
uses for fold artifacts, and the fail-closed checks that make the seal mean
something.

The central refusal is deliberately blunt: **a receipt that cannot name its
exact data is refused.**  A run that consumed no named shard, that consumed a
shard absent from its corpus manifest, that quotes a manifest whose bytes have
moved, or that distilled from a teacher the manifest does not name, produces no
receipt at all.  Section 4 makes the comparison void unless compute is measured
on both sides; a receipt whose corpus is unnameable would make it void
silently, which is the failure the staged policy exists to prevent.

Per ceiling-report section 7.4 rule 5 the provenance chains stay separate:
every record here is labelled ``stage2`` and declares that it cites no stage-1
zero-training receipt and modified no stage-1 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


STAGE2_TRAINING_RECEIPT_SCHEMA = "neural-forge-stage2-training-receipt-v1"
STAGE2_RUNG_RECORD_SCHEMA = "neural-forge-stage2-rung-record-v1"
STAGE2_DESIGN_FILENAME = "STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md"
# The pre-registration is bytes, not a paraphrase.  Section 8 runs from its own
# heading to the byte before the section-9 heading, LF newlines, 3,337 bytes at
# design HEAD `fec82744`.  Recording the digest here means an edit to section 8
# after sealing is a test failure, which is the property section 8 claims for
# itself ("may not be edited afterward").
STAGE2_PREREGISTRATION_SECTION = "## 8. Pre-registered success / failure criteria"
STAGE2_PREREGISTRATION_NEXT_SECTION = "\n## 9. "
STAGE2_PREREGISTRATION_BYTE_LENGTH = 3_337
STAGE2_PREREGISTRATION_SHA256 = (
    "70453e10f347b454443210ca26911e2e21a95c0c993dd6f1df02a1468d95d3d3"
)
# The whole document *may* gain amendments; only section 8 may not move.  It
# gained one: commit `53d31156` appended "AMENDMENT 1 -- 2026-08-09: corpus
# arithmetic and throughput corrected" after section 12, 20 lines, entirely
# below section 8.  ``STAGE2_PREREGISTRATION_SHA256`` above is unchanged and
# still verifies, which is the property that actually matters; this digest
# tracks the document so an amendment is *visible* rather than silent.
#   design HEAD `fec82744`:
#     96b642cb4eebe7671228801ea98fb722cad9c913af7f63bbc264de8b15b24a04
#   amended `53d31156`:
#     4a71d829934c8d07c127bcc09ba7dc59964c96135c0d523ef9c861ce2e0b83f8
STAGE2_DESIGN_DOCUMENT_SHA256 = (
    "4a71d829934c8d07c127bcc09ba7dc59964c96135c0d523ef9c861ce2e0b83f8"
)
# Design section 3.1: rungs are checkpoints inside one run per arm.
STAGE2_RUNG_TOKENS = (10_000_000, 100_000_000, 1_000_000_000)
# Lane 2's shard layout (``experiments/stage2_generate_teacher_corpus.py``);
# lane 1 reads it, never writes it.
STAGE2_CORPUS_SHARD_PATH_TEMPLATE = "shards/shard-{index:05d}.tokens.u16"


class Stage2TrainingReceiptError(RuntimeError):
    """A stage-2 receipt, corpus consumption record, or seal is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage2TrainingReceiptError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2TrainingReceiptError(f"{name} must be a nonnegative integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage2TrainingReceiptError(f"{name} must be a lowercase SHA-256.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage2TrainingReceiptError(f"{name} must be a nonempty string.")
    return value.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stage2_preregistration_digest(design_markdown: Path) -> dict[str, Any]:
    """Hash section 8 out of the design document by its exact byte range."""

    path = Path(design_markdown)
    if path.is_symlink() or not path.is_file():
        raise Stage2TrainingReceiptError(
            "Stage-2 design document is missing or not a real file."
        )
    raw = path.read_bytes()
    marker = STAGE2_PREREGISTRATION_SECTION.encode("utf-8")
    stop_marker = STAGE2_PREREGISTRATION_NEXT_SECTION.encode("utf-8")
    start = raw.find(marker)
    if start < 0:
        raise Stage2TrainingReceiptError(
            "Stage-2 design document has no section 8 heading."
        )
    stop = raw.find(stop_marker, start)
    if stop < 0:
        raise Stage2TrainingReceiptError(
            "Stage-2 design document has no section 9 heading after section 8."
        )
    section = raw[start : stop + 1]
    return {
        "design_document": path.name,
        "design_document_sha256": hashlib.sha256(raw).hexdigest(),
        "preregistration_byte_length": len(section),
        "preregistration_sha256": hashlib.sha256(section).hexdigest(),
    }


def assert_preregistration_unedited(design_markdown: Path) -> dict[str, Any]:
    """Refuse to run if section 8 no longer hashes to its sealed digest."""

    observed = stage2_preregistration_digest(design_markdown)
    if (
        observed["preregistration_sha256"] != STAGE2_PREREGISTRATION_SHA256
        or observed["preregistration_byte_length"]
        != STAGE2_PREREGISTRATION_BYTE_LENGTH
    ):
        raise Stage2TrainingReceiptError(
            "Stage-2 pre-registration section 8 has changed since it was "
            f"sealed ({observed['preregistration_sha256']} / "
            f"{observed['preregistration_byte_length']} bytes)."
        )
    return observed


@dataclass(frozen=True, slots=True)
class Stage2ShardConsumption:
    """One corpus shard the student actually read, in the order it was read."""

    order_index: int
    relative_path: str
    sha256: str
    tokens_consumed: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.order_index, "order_index")
        _text(self.relative_path, "relative_path")
        _digest(self.sha256, "shard sha256")
        _positive_int(self.tokens_consumed, "tokens_consumed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_index": self.order_index,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "tokens_consumed": self.tokens_consumed,
        }


@dataclass(frozen=True, slots=True)
class Stage2CorpusConsumption:
    """The consumed-data half of a receipt, checked against lane 2's manifest.

    The manifest itself is produced by the data lane and read only here
    (design section 10, interface (i)).  This class does not care what else the
    manifest carries; it cares that every shard the trainer read is in it, with
    the same digest, and that the teacher the trainer distilled from is the
    teacher the manifest was generated by.
    """

    manifest_path: str
    manifest_sha256: str
    teacher_checkpoint_sha256: str
    shard_order_seed: int
    shards: tuple[Stage2ShardConsumption, ...]

    def __post_init__(self) -> None:
        _text(self.manifest_path, "manifest_path")
        _digest(self.manifest_sha256, "corpus manifest sha256")
        _digest(self.teacher_checkpoint_sha256, "teacher checkpoint sha256")
        _nonnegative_int(self.shard_order_seed, "shard_order_seed")
        if not self.shards:
            raise Stage2TrainingReceiptError(
                "A stage-2 receipt must name at least one consumed shard."
            )
        expected = tuple(range(len(self.shards)))
        if tuple(shard.order_index for shard in self.shards) != expected:
            raise Stage2TrainingReceiptError(
                "Consumed shard order indices must be dense and ascending."
            )

    @property
    def tokens_consumed(self) -> int:
        return sum(shard.tokens_consumed for shard in self.shards)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "shard_order_seed": self.shard_order_seed,
            "shard_count": len(self.shards),
            "shards_consumed_in_order": [shard.as_dict() for shard in self.shards],
            "tokens_consumed_from_shards": self.tokens_consumed,
        }


def corpus_manifest_teacher_sha256(manifest: Mapping[str, Any]) -> str:
    """Read the teacher digest out of a lane-2 corpus manifest.

    Lane 2 owns the manifest schema (design section 10, interface (i)) and
    records the teacher under ``run.teacher_shard_sha256``.  Both that spelling
    and a flat ``teacher_checkpoint_sha256`` are accepted; anything else is a
    refusal, because section 2.1 makes distilling from an unnamed teacher
    void the comparison.
    """

    if not isinstance(manifest, Mapping):
        raise Stage2TrainingReceiptError("Corpus manifest must be an object.")
    flat = manifest.get("teacher_checkpoint_sha256")
    if flat is not None:
        return _digest(flat, "manifest teacher_checkpoint_sha256")
    run = manifest.get("run")
    if isinstance(run, Mapping) and run.get("teacher_shard_sha256") is not None:
        return _digest(
            run.get("teacher_shard_sha256"),
            "manifest run.teacher_shard_sha256",
        )
    raise Stage2TrainingReceiptError(
        "Corpus manifest does not name the teacher it was generated from."
    )


def corpus_manifest_shard_index(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Normalize a lane-2 shard inventory to ``relative path -> sha256``.

    Lane 2 seals shards by index (``shard_index`` + ``tokens_sha256``) and lays
    them out at :data:`STAGE2_CORPUS_SHARD_PATH_TEMPLATE`; an explicit
    ``relative_path`` + ``sha256`` pair is also accepted so the receipt code
    does not depend on one lane's file-naming choice.
    """

    shards = manifest.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise Stage2TrainingReceiptError("Corpus manifest must list its shards.")
    index: dict[str, str] = {}
    for row in shards:
        if not isinstance(row, Mapping):
            raise Stage2TrainingReceiptError(
                "Corpus manifest shard entries must be objects."
            )
        if row.get("relative_path") is not None:
            relative_path = _text(row.get("relative_path"), "manifest relative_path")
            sha256 = _digest(row.get("sha256"), "manifest shard sha256")
        elif row.get("shard_index") is not None:
            relative_path = STAGE2_CORPUS_SHARD_PATH_TEMPLATE.format(
                index=_nonnegative_int(row.get("shard_index"), "shard_index")
            )
            sha256 = _digest(row.get("tokens_sha256"), "manifest tokens_sha256")
        else:
            raise Stage2TrainingReceiptError(
                "Corpus manifest shard names neither a path nor an index."
            )
        if relative_path in index:
            raise Stage2TrainingReceiptError(
                f"Corpus manifest repeats shard {relative_path!r}."
            )
        index[relative_path] = sha256
    if not index:
        raise Stage2TrainingReceiptError("Corpus manifest names no shards.")
    return index


def verify_corpus_consumption(
    consumption: Stage2CorpusConsumption,
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless every consumed byte is named by the manifest."""

    if not isinstance(consumption, Stage2CorpusConsumption):
        raise Stage2TrainingReceiptError(
            "consumption must be a Stage2CorpusConsumption."
        )
    if consumption.manifest_sha256 != _digest(
        manifest_file_sha256,
        "observed corpus manifest sha256",
    ):
        raise Stage2TrainingReceiptError(
            "Corpus manifest bytes differ from the digest the receipt quotes."
        )
    if corpus_manifest_teacher_sha256(manifest) != (
        consumption.teacher_checkpoint_sha256
    ):
        raise Stage2TrainingReceiptError(
            "Receipt teacher digest differs from the corpus manifest's teacher."
        )
    index = corpus_manifest_shard_index(manifest)
    for shard in consumption.shards:
        expected = index.get(shard.relative_path)
        if expected is None:
            raise Stage2TrainingReceiptError(
                f"Consumed shard {shard.relative_path!r} is not in the manifest."
            )
        if expected != shard.sha256:
            raise Stage2TrainingReceiptError(
                f"Consumed shard {shard.relative_path!r} differs from the "
                "manifest digest."
            )
    return {
        "corpus_manifest_verified": True,
        "manifest_shard_count": len(index),
        "consumed_shard_count": len(consumption.shards),
        "teacher_matches_manifest": True,
    }


def optimizer_state_digest(optimizer: Any) -> dict[str, Any]:
    """A reproducible digest of optimizer hyperparameters and moment state.

    ``torch`` is imported locally so a receipt can be built, sealed, reloaded,
    and audited on a machine with no training stack installed.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Stage2TrainingReceiptError(
            "PyTorch is required to digest optimizer state."
        ) from exc
    state = optimizer.state_dict()
    groups = state.get("param_groups")
    raw_state = state.get("state")
    if not isinstance(groups, list) or not isinstance(raw_state, dict):
        raise Stage2TrainingReceiptError(
            "Optimizer state_dict does not expose param_groups and state."
        )
    hyperparameters = [
        {
            key: value
            for key, value in sorted(group.items())
            if key != "params" and isinstance(value, (int, float, bool, str))
        }
        for group in groups
    ]
    tensor_digests: list[dict[str, Any]] = []
    element_count = 0
    for key in sorted(raw_state, key=lambda value: str(value)):
        entry = raw_state[key]
        if not isinstance(entry, Mapping):
            continue
        for name in sorted(entry):
            value = entry[name]
            if hasattr(value, "detach") and hasattr(value, "numel"):
                tensor = value.detach().to(device="cpu").contiguous()
                element_count += int(tensor.numel())
                payload = hashlib.sha256()
                payload.update(str(tensor.dtype).encode("ascii"))
                payload.update(b"\0")
                payload.update(str(tuple(tensor.shape)).encode("ascii"))
                payload.update(b"\0")
                payload.update(
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                )
                tensor_digests.append(
                    {
                        "parameter_key": str(key),
                        "state_name": str(name),
                        "sha256": payload.hexdigest(),
                    }
                )
            elif isinstance(value, (int, float, bool)):
                tensor_digests.append(
                    {
                        "parameter_key": str(key),
                        "state_name": str(name),
                        "scalar": value,
                    }
                )
    body = {
        "param_group_hyperparameters": hyperparameters,
        "state_entries": tensor_digests,
        "state_element_count": element_count,
    }
    return {
        "optimizer_state_sha256": canonical_sha256(body),
        "optimizer_state_element_count": element_count,
        "optimizer_param_group_count": len(hyperparameters),
    }


@dataclass(frozen=True, slots=True)
class Stage2RungRecord:
    """One ladder rung: what was consumed, what was written, what was probed."""

    rung_tokens: int
    tokens_consumed: int
    optimizer_steps: int
    backward_calls: int
    micro_batch_forward_calls: int
    on_policy_steps: int
    off_policy_steps: int
    wall_clock_seconds: float
    checkpoint_manifest_sha256: str | None
    checkpoint_directory: str | None
    optimizer_state: Mapping[str, Any]
    free_run_probe: Mapping[str, Any] | None
    loss_curve: tuple[Mapping[str, Any], ...] = ()
    diagnostic_held_out_nll: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_RUNG_RECORD_SCHEMA,
            "rung_tokens": self.rung_tokens,
            "tokens_consumed": self.tokens_consumed,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "micro_batch_forward_calls": self.micro_batch_forward_calls,
            "on_policy_steps": self.on_policy_steps,
            "off_policy_steps": self.off_policy_steps,
            "wall_clock_seconds": round(float(self.wall_clock_seconds), 6),
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_directory": self.checkpoint_directory,
            "optimizer_state": dict(self.optimizer_state),
            "free_run_probe": (
                None if self.free_run_probe is None else dict(self.free_run_probe)
            ),
            "loss_curve": [dict(row) for row in self.loss_curve],
            "diagnostic_held_out_nll": {
                "diagnostic_only": True,
                "acceptance_authority": "none",
                "series": [dict(row) for row in self.diagnostic_held_out_nll],
            },
        }


@dataclass(slots=True)
class Stage2TrainingReceiptBuilder:
    """Accumulate one arm's run record, then seal it once."""

    arm_id: str
    init_source_sha256: str
    initialization: Mapping[str, Any]
    teacher_checkpoint_sha256: str
    corpus: Stage2CorpusConsumption
    objective_schedule: Mapping[str, Any]
    optimizer_configuration: Mapping[str, Any]
    trainer_configuration: Mapping[str, Any]
    preregistration: Mapping[str, Any]
    device_inventory: tuple[Mapping[str, Any], ...]
    seeds: Mapping[str, Any]
    probe_surface: Mapping[str, Any]
    route_surface_equivalence: Mapping[str, Any]
    counted_parameters: Mapping[str, Any]
    rungs: list[Stage2RungRecord] = field(default_factory=list)
    loss_curve: list[Mapping[str, Any]] = field(default_factory=list)
    free_run_probe_series: list[Mapping[str, Any]] = field(default_factory=list)
    diagnostic_held_out_nll_series: list[Mapping[str, Any]] = field(
        default_factory=list
    )
    wall_clock_seconds: float = 0.0

    def body(self) -> dict[str, Any]:
        if not self.rungs:
            raise Stage2TrainingReceiptError(
                "A stage-2 receipt must carry at least one ladder rung."
            )
        if self.preregistration.get(
            "preregistration_sha256"
        ) != STAGE2_PREREGISTRATION_SHA256:
            raise Stage2TrainingReceiptError(
                "Receipt does not carry the sealed section-8 pre-registration."
            )
        if self.corpus.teacher_checkpoint_sha256 != _digest(
            self.teacher_checkpoint_sha256,
            "teacher_checkpoint_sha256",
        ):
            raise Stage2TrainingReceiptError(
                "Receipt teacher digest differs from its corpus record."
            )
        device_count = len(self.device_inventory)
        return {
            "schema": STAGE2_TRAINING_RECEIPT_SCHEMA,
            "stage": "stage2",
            "status": "complete",
            "arm_id": _text(self.arm_id, "arm_id"),
            "init_source_sha256": _digest(
                self.init_source_sha256,
                "init_source_sha256",
            ),
            "initialization": dict(self.initialization),
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "corpus": self.corpus.as_dict(),
            "objective_schedule": dict(self.objective_schedule),
            "optimizer_configuration": dict(self.optimizer_configuration),
            "trainer_configuration": dict(self.trainer_configuration),
            "counted_parameters": dict(self.counted_parameters),
            "route_surface_equivalence": dict(self.route_surface_equivalence),
            "free_run_probe_surface": dict(self.probe_surface),
            "seeds": dict(self.seeds),
            "preregistration": dict(self.preregistration),
            "device_inventory": [dict(row) for row in self.device_inventory],
            "wall_clock_seconds": round(float(self.wall_clock_seconds), 6),
            "gpu_hours": round(
                float(self.wall_clock_seconds) * device_count / 3600.0,
                9,
            ),
            "gpu_hours_are_measured_not_estimated": True,
            "rungs": [rung.as_dict() for rung in self.rungs],
            "loss_curve": [dict(row) for row in self.loss_curve],
            "free_run_probe_series": [
                dict(row) for row in self.free_run_probe_series
            ],
            "diagnostic_held_out_nll": {
                "diagnostic_only": True,
                "acceptance_authority": "none",
                "calibration_caveat": (
                    "ceiling report section 4.3 -- teacher-forced signals are "
                    "anti-correlated with free-run behaviour"
                ),
                "series": [
                    dict(row) for row in self.diagnostic_held_out_nll_series
                ],
            },
            "metric_policy": {
                "verdict_authority": "frozen-v5-manual-review",
                "curve_signal": "free-run-token-match-and-lcp",
                "terminal_reconstruction_used": False,
                "teacher_forced_top1_used": False,
                "cached_frame_metric_used": False,
            },
            "provenance": {
                "stage1_artifacts_modified": False,
                "stage1_receipt_cited_as_covering_trained_child": False,
                "chain": "stage2-only",
            },
        }

    def seal(self) -> dict[str, Any]:
        body = self.body()
        return {**body, "receipt_sha256": canonical_sha256(body)}


def write_stage2_training_receipt(
    destination: Path,
    receipt: Mapping[str, Any],
) -> str:
    """Atomically write one sealed receipt; refuse to overwrite."""

    path = Path(destination).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Stage2TrainingReceiptError(
            "Stage-2 receipt destination already exists."
        )
    body = dict(receipt)
    seal = body.pop("receipt_sha256", None)
    if not isinstance(seal, str) or seal != canonical_sha256(body):
        raise Stage2TrainingReceiptError(
            "Refusing to write a stage-2 receipt whose seal does not verify."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stage2-receipt.",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(receipt), allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return seal


def load_stage2_training_receipt(
    source: Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Reload a receipt, refusing any byte that moved after sealing."""

    path = Path(source).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Stage2TrainingReceiptError(
            "Stage-2 receipt is missing or not a real file."
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2TrainingReceiptError(
                    f"Stage-2 receipt repeats key {key!r}."
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Stage2TrainingReceiptError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2TrainingReceiptError(
            "Stage-2 receipt is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise Stage2TrainingReceiptError("Stage-2 receipt must be an object.")
    body = dict(payload)
    seal = body.pop("receipt_sha256", None)
    if (
        body.get("schema") != STAGE2_TRAINING_RECEIPT_SCHEMA
        or body.get("stage") != "stage2"
        or not isinstance(seal, str)
        or seal != canonical_sha256(body)
    ):
        raise Stage2TrainingReceiptError(
            "Stage-2 receipt self-seal does not verify."
        )
    if expected_receipt_sha256 is not None and seal != _digest(
        expected_receipt_sha256,
        "expected receipt sha256",
    ):
        raise Stage2TrainingReceiptError(
            "Stage-2 receipt belongs to a different sealed run."
        )
    return payload


__all__ = [
    "STAGE2_CORPUS_SHARD_PATH_TEMPLATE",
    "STAGE2_DESIGN_DOCUMENT_SHA256",
    "STAGE2_DESIGN_FILENAME",
    "STAGE2_PREREGISTRATION_BYTE_LENGTH",
    "STAGE2_PREREGISTRATION_SHA256",
    "STAGE2_RUNG_RECORD_SCHEMA",
    "STAGE2_RUNG_TOKENS",
    "STAGE2_TRAINING_RECEIPT_SCHEMA",
    "Stage2CorpusConsumption",
    "Stage2RungRecord",
    "Stage2ShardConsumption",
    "Stage2TrainingReceiptBuilder",
    "Stage2TrainingReceiptError",
    "assert_preregistration_unedited",
    "corpus_manifest_shard_index",
    "corpus_manifest_teacher_sha256",
    "file_sha256",
    "load_stage2_training_receipt",
    "optimizer_state_digest",
    "stage2_preregistration_digest",
    "verify_corpus_consumption",
    "write_stage2_training_receipt",
]
