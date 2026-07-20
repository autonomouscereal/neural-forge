"""Sequential target-runtime substrate for an LFM2 CPCF model fold.

This module joins three pieces that must not be treated as independent local
surgeries:

* a strict target-tokenizer record is expanded into the exact full prefix for
  every cached prefill/decode phase;
* the compiler-owned zero blank receives its tied causal interface and
  explicit neutral normalization coordinates; and
* every installed macrocell is replayed on the *actual folded-upstream child
  state* before the next component is compiled.

The component solvers remain separate.  This substrate deliberately cannot
run a parent, train, prune, quantize, or silently replace a failed child with a
parent-backed wrapper.  It exists to prevent the old failure mode where every
layer was fitted against an independently projected parent state and the
errors were never propagated through the child that would actually be saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (
    Lfm2CpcfMacrocellReplay,
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
)


LFM2_CPCF_TARGET_PHASE_SCHEMA = "neural-forge-lfm2-cpcf-target-phase-v1"
LFM2_CPCF_TARGET_RUNTIME_SCHEMA = "neural-forge-lfm2-cpcf-target-runtime-v1"
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfModelCompilerError(ValueError):
    """Target phase alignment or sequential child replay is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF model compilation."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfModelCompilerError(
            "CPCF target-runtime metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfModelCompilerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfModelCompilerError(f"{name} must be nonempty text.")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfModelCompilerError(f"{name} must be a nonnegative integer.")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2CpcfModelCompilerError(f"{name} must be a positive integer.")
    return result


def _integer_matrix(value: Any, name: str) -> Any:
    torch = _torch()
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or value.dtype not in integer_dtypes
        or value.requires_grad
    ):
        raise Lfm2CpcfModelCompilerError(
            f"{name} must be a nonempty dense rank-two integer tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.long).contiguous()
    if bool((result < 0).any().item()):
        raise Lfm2CpcfModelCompilerError(f"{name} contains a negative value.")
    return result.clone()


class _OwnedCpuPhaseStates(dict[tuple[int, str], Any]):
    """Private marker for CPU archives already owned by a phase-state bank."""


def _floating_states(
    value: Any,
    name: str,
    *,
    hidden_size: int | None = None,
    copy: bool = True,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 3
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfModelCompilerError(
            f"{name} must be a detached dense real rank-three tensor."
        )
    if hidden_size is not None and int(value.shape[-1]) != hidden_size:
        raise Lfm2CpcfModelCompilerError(
            f"{name} hidden width differs from {hidden_size}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise Lfm2CpcfModelCompilerError(f"{name} contains NaN or infinity.")
    # Phase banks are durable compiler archives, not live execution tensors.
    # Canonical CPU storage keeps accelerator residency independent of corpus
    # size and layer count.  The copy is synchronous and preserves the exact
    # dtype/shape/bytes authenticated by the bank seal.
    detached = value.detach()
    if not copy and detached.device.type == "cpu" and detached.is_contiguous():
        return detached
    return detached.to(
        device="cpu",
        non_blocking=False,
        copy=True,
        memory_format=torch.contiguous_format,
    )


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    detached = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": [int(size) for size in detached.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(detached.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2CpcfTargetPhase:
    """One target-token full prefix aligned to one authenticated parent phase."""

    batch_index: int
    record_id: str
    split: str
    phase: str
    donor_input_sha256: str
    target_alignment_sha256: str
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    sampled_suffix_tokens: int
    parent_evidence: Lfm2CpcfInstallablePhaseEvidence
    phase_sha256: str

    def __post_init__(self) -> None:
        batch_index = _nonnegative_int(self.batch_index, "batch_index")
        record_id = _text(self.record_id, "record_id")
        split = _text(self.split, "split")
        phase = _text(self.phase, "phase")
        donor_sha = _digest(
            self.donor_input_sha256,
            "donor_input_sha256",
        )
        alignment_sha = _digest(
            self.target_alignment_sha256,
            "target_alignment_sha256",
        )
        ids = _integer_matrix(self.input_ids, "input_ids")
        mask = _integer_matrix(self.attention_mask, "attention_mask")
        positions = _integer_matrix(self.position_ids, "position_ids")
        if (
            tuple(mask.shape) != tuple(ids.shape)
            or tuple(positions.shape) != tuple(ids.shape)
            or not bool(((mask == 0) | (mask == 1)).all().item())
            or bool((mask.sum(dim=1) == 0).any().item())
        ):
            raise Lfm2CpcfModelCompilerError(
                "Target phase IDs, mask, and positions are misaligned."
            )
        sampled = _positive_int(
            self.sampled_suffix_tokens,
            "sampled_suffix_tokens",
        )
        if sampled > int(ids.shape[1]):
            raise Lfm2CpcfModelCompilerError(
                "sampled_suffix_tokens exceeds the full target prefix."
            )
        evidence = self.parent_evidence
        if (
            not isinstance(evidence, Lfm2CpcfInstallablePhaseEvidence)
            or evidence.batch_index != batch_index
            or evidence.phase != phase
            or evidence.input_sha256 != donor_sha
        ):
            raise Lfm2CpcfModelCompilerError(
                "Parent evidence differs from the aligned target phase."
            )
        body = {
            "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
            "batch_index": batch_index,
            "record_id": record_id,
            "split": split,
            "phase": phase,
            "donor_input_sha256": donor_sha,
            "target_alignment_sha256": alignment_sha,
            "input_ids_sha256": _tensor_sha256(ids),
            "attention_mask_sha256": _tensor_sha256(mask),
            "position_ids_sha256": _tensor_sha256(positions),
            "batch_size": int(ids.shape[0]),
            "full_prefix_tokens": int(ids.shape[1]),
            "sampled_suffix_tokens": sampled,
            "parent_evidence_sha256": evidence.evidence_sha256,
            "numeric_token_id_equality_required": False,
            "actual_child_full_prefix_required": True,
        }
        expected = _sha(body)
        if self.phase_sha256 != expected:
            raise Lfm2CpcfModelCompilerError("Target phase evidence seal is invalid.")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "input_ids", ids)
        object.__setattr__(self, "attention_mask", mask)
        object.__setattr__(self, "position_ids", positions)

    @property
    def identity(self) -> tuple[int, str]:
        return self.batch_index, self.phase

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def full_prefix_tokens(self) -> int:
        return int(self.input_ids.shape[1])

    def suffix(self, value: Any) -> Any:
        if (
            not hasattr(value, "ndim")
            or value.ndim < 2
            or int(value.shape[0]) != self.batch_size
            or int(value.shape[1]) != self.full_prefix_tokens
        ):
            raise Lfm2CpcfModelCompilerError(
                "Cannot select an aligned suffix from this tensor."
            )
        return value[:, -self.sampled_suffix_tokens :, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
            "batch_index": self.batch_index,
            "record_id": self.record_id,
            "split": self.split,
            "phase": self.phase,
            "donor_input_sha256": self.donor_input_sha256,
            "target_alignment_sha256": self.target_alignment_sha256,
            "batch_size": self.batch_size,
            "full_prefix_tokens": self.full_prefix_tokens,
            "sampled_suffix_tokens": self.sampled_suffix_tokens,
            "parent_evidence_sha256": self.parent_evidence.evidence_sha256,
            "phase_sha256": self.phase_sha256,
        }


def _target_phase(
    *,
    record: Any,
    evidence: Lfm2CpcfInstallablePhaseEvidence,
) -> Lfm2CpcfTargetPhase:
    torch = _torch()
    prompt_ids = _integer_matrix(record.target_input_ids, "target_input_ids")
    prompt_mask = _integer_matrix(record.attention_mask, "attention_mask")
    forced = _integer_matrix(
        record.target_forced_decode_ids,
        "target_forced_decode_ids",
    )
    base_positions = getattr(record, "position_ids", None)
    if base_positions is None:
        prompt_positions = (
            torch.arange(
                int(prompt_ids.shape[1]),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(int(prompt_ids.shape[0]), -1)
        )
    else:
        prompt_positions = _integer_matrix(base_positions, "position_ids")
    if evidence.phase in {"prefill", "cached_prefill"}:
        ids = prompt_ids
        mask = prompt_mask
        positions = prompt_positions
    else:
        match = _DECODE_PHASE.fullmatch(evidence.phase)
        if match is None:
            raise Lfm2CpcfModelCompilerError(
                f"Unsupported parent phase {evidence.phase!r}."
            )
        step_count = int(match.group(1)) + 1
        if step_count > int(forced.shape[1]):
            raise Lfm2CpcfModelCompilerError(
                "Decode phase exceeds the target forced-decode record."
            )
        suffix_ids = forced[:, :step_count]
        ids = torch.cat((prompt_ids, suffix_ids), dim=1).contiguous()
        mask = torch.cat(
            (
                prompt_mask,
                torch.ones_like(suffix_ids, dtype=torch.long),
            ),
            dim=1,
        ).contiguous()
        start = prompt_positions[:, -1:] + 1
        offsets = torch.arange(step_count, dtype=torch.long).unsqueeze(0)
        decode_positions = start + offsets
        positions = torch.cat(
            (prompt_positions, decode_positions),
            dim=1,
        ).contiguous()
    sampled = int(evidence.layers[0].tensors["residual.pre"].shape[1])
    body = {
        "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
        "batch_index": evidence.batch_index,
        "record_id": str(record.record_id),
        "split": str(record.split),
        "phase": evidence.phase,
        "donor_input_sha256": evidence.input_sha256,
        "target_alignment_sha256": str(record.alignment_sha256),
        "input_ids_sha256": _tensor_sha256(ids),
        "attention_mask_sha256": _tensor_sha256(mask),
        "position_ids_sha256": _tensor_sha256(positions),
        "batch_size": int(ids.shape[0]),
        "full_prefix_tokens": int(ids.shape[1]),
        "sampled_suffix_tokens": sampled,
        "parent_evidence_sha256": evidence.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }
    return Lfm2CpcfTargetPhase(
        batch_index=evidence.batch_index,
        record_id=str(record.record_id),
        split=str(record.split),
        phase=evidence.phase,
        donor_input_sha256=evidence.input_sha256,
        target_alignment_sha256=str(record.alignment_sha256),
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        sampled_suffix_tokens=sampled,
        parent_evidence=evidence,
        phase_sha256=_sha(body),
    )


def build_lfm2_cpcf_target_phases(
    trace_adapter: Any,
    parent_phases: Sequence[Lfm2CpcfInstallablePhaseEvidence],
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Bind strict target IDs to every streamed installable parent phase."""

    verify = getattr(trace_adapter, "verify", None)
    records = getattr(trace_adapter, "records", None)
    if not callable(verify) or not isinstance(records, tuple):
        raise Lfm2CpcfModelCompilerError(
            "A verified strict tokenizer trace adapter is required."
        )
    verify()
    phases = tuple(parent_phases)
    if not phases or not all(
        isinstance(value, Lfm2CpcfInstallablePhaseEvidence) for value in phases
    ):
        raise Lfm2CpcfModelCompilerError(
            "parent_phases must contain installable phase evidence."
        )
    by_ordinal = {int(record.ordinal): record for record in records}
    if len(by_ordinal) != len(records):
        raise Lfm2CpcfModelCompilerError(
            "Tokenizer trace record ordinals are not unique."
        )
    result = []
    seen: set[tuple[int, str]] = set()
    for evidence in phases:
        record = by_ordinal.get(evidence.batch_index)
        if record is None:
            raise Lfm2CpcfModelCompilerError(
                f"No target tokenizer record for cache batch {evidence.batch_index}."
            )
        if record.donor_cache_input_sha256 != evidence.input_sha256:
            raise Lfm2CpcfModelCompilerError(
                "Target record is not bound to the parent cache input."
            )
        phase = _target_phase(record=record, evidence=evidence)
        if phase.identity in seen:
            raise Lfm2CpcfModelCompilerError(
                f"Duplicate target phase {phase.identity}."
            )
        seen.add(phase.identity)
        result.append(phase)
    trace_adapter.verify()
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfPhaseStateBank:
    """Full-prefix actual child residuals at one sequential model boundary."""

    boundary: str
    layer_index: int
    phases: tuple[Lfm2CpcfTargetPhase, ...]
    states: Mapping[tuple[int, str], Any]
    state_bank_sha256: str

    def __post_init__(self) -> None:
        boundary = _text(self.boundary, "boundary")
        layer_index = _nonnegative_int(self.layer_index, "layer_index")
        phases = tuple(self.phases)
        if not phases:
            raise Lfm2CpcfModelCompilerError("A phase-state bank cannot be empty.")
        expected = tuple(phase.identity for phase in phases)
        if len(set(expected)) != len(expected) or set(self.states) != set(expected):
            raise Lfm2CpcfModelCompilerError(
                "Phase-state identities are missing, duplicated, or reordered."
            )
        normalized: dict[tuple[int, str], Any] = {}
        hidden_sizes = set()
        dtypes = set()
        inventory = []
        owns_archive = isinstance(self.states, _OwnedCpuPhaseStates)
        for phase in phases:
            state = _floating_states(
                self.states[phase.identity],
                f"states[{phase.identity!r}]",
                copy=not owns_archive,
            )
            if tuple(state.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"State {phase.identity!r} differs from its full prefix."
                )
            hidden_sizes.add(int(state.shape[-1]))
            dtypes.add(state.dtype)
            normalized[phase.identity] = state
            inventory.append(
                {
                    "identity": [phase.batch_index, phase.phase],
                    "phase_sha256": phase.phase_sha256,
                    "state_sha256": _tensor_sha256(state),
                }
            )
        if len(hidden_sizes) != 1:
            raise Lfm2CpcfModelCompilerError(
                "Actual child residual widths differ by phase."
            )
        if len(dtypes) != 1:
            raise Lfm2CpcfModelCompilerError(
                "Actual child residual dtypes differ by phase."
            )
        body = {
            "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
            "scope": "phase_state_bank",
            "boundary": boundary,
            "layer_index": layer_index,
            "hidden_size": next(iter(hidden_sizes)),
            "ordered_states": inventory,
            "actual_folded_upstream_states": True,
        }
        if self.state_bank_sha256 != _sha(body):
            raise Lfm2CpcfModelCompilerError("Phase-state bank seal is invalid.")
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(
            self,
            "states",
            MappingProxyType(normalized),
        )

    @property
    def hidden_size(self) -> int:
        return int(next(iter(self.states.values())).shape[-1])

    def state(self, phase: Lfm2CpcfTargetPhase) -> Any:
        try:
            return self.states[phase.identity].clone()
        except KeyError as exc:
            raise Lfm2CpcfModelCompilerError(
                f"Phase {phase.identity!r} is absent from this bank."
            ) from exc

    def suffix(self, phase: Lfm2CpcfTargetPhase) -> Any:
        return phase.suffix(self.state(phase))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
            "scope": "phase_state_bank",
            "boundary": self.boundary,
            "layer_index": self.layer_index,
            "phase_count": len(self.phases),
            "hidden_size": self.hidden_size,
            "state_bank_sha256": self.state_bank_sha256,
        }


def _phase_state_bank(
    boundary: str,
    layer_index: int,
    phases: Sequence[Lfm2CpcfTargetPhase],
    states: Mapping[tuple[int, str], Any],
) -> Lfm2CpcfPhaseStateBank:
    ordered = tuple(phases)
    owned = _OwnedCpuPhaseStates()
    inventory = []
    for phase in ordered:
        state = _floating_states(
            states[phase.identity],
            f"states[{phase.identity!r}]",
            copy=False,
        )
        owned[phase.identity] = state
        inventory.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "phase_sha256": phase.phase_sha256,
                "state_sha256": _tensor_sha256(state),
            }
        )
    body = {
        "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
        "scope": "phase_state_bank",
        "boundary": boundary,
        "layer_index": layer_index,
        "hidden_size": int(next(iter(owned.values())).shape[-1]),
        "ordered_states": inventory,
        "actual_folded_upstream_states": True,
    }
    return Lfm2CpcfPhaseStateBank(
        boundary=boundary,
        layer_index=layer_index,
        phases=ordered,
        states=owned,
        state_bank_sha256=_sha(body),
    )


def select_lfm2_cpcf_phase_state_bank(
    bank: Lfm2CpcfPhaseStateBank,
    *,
    splits: Sequence[str],
) -> Lfm2CpcfPhaseStateBank:
    """Rebuild a phase-state bank from complete trajectories in chosen splits.

    Split selection is intentionally performed over trajectories keyed by
    ``(batch_index, record_id)``.  A trajectory is admitted only when it has
    one complete prefill followed by every contiguous decode phase present in
    the authenticated source bank, with unchanged causal prefixes.  The
    returned bank owns fresh state tensors and is resealed solely from the
    selected phases and states; excluded splits cannot affect its content or
    digest.
    """

    torch = _torch()
    if not isinstance(bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("bank must be Lfm2CpcfPhaseStateBank.")
    if not isinstance(splits, Sequence) or isinstance(splits, (str, bytes)):
        raise TypeError("splits must be a sequence of split names.")
    requested = []
    seen_splits: set[str] = set()
    for index, value in enumerate(splits):
        split = _text(value, f"splits[{index}]")
        if split in seen_splits:
            raise Lfm2CpcfModelCompilerError(
                f"Requested split {split!r} is duplicated."
            )
        seen_splits.add(split)
        requested.append(split)
    if not requested:
        raise Lfm2CpcfModelCompilerError(
            "At least one phase-state split must be requested."
        )

    available = {phase.split for phase in bank.phases}
    missing = [split for split in requested if split not in available]
    if missing:
        raise Lfm2CpcfModelCompilerError(
            "Requested phase-state splits are absent from the bank: "
            + ", ".join(repr(split) for split in missing)
            + "."
        )

    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    for phase in bank.phases:
        grouped.setdefault(
            (phase.batch_index, phase.record_id),
            [],
        ).append(phase)

    selected_trajectories: set[tuple[int, str]] = set()
    for trajectory_id, trajectory_values in grouped.items():
        trajectory = tuple(trajectory_values)
        trajectory_splits = {phase.split for phase in trajectory}
        if not trajectory_splits.intersection(seen_splits):
            continue
        if len(trajectory_splits) != 1:
            raise Lfm2CpcfModelCompilerError(
                f"Trajectory {trajectory_id!r} crosses phase-state splits."
            )
        trajectory_split = next(iter(trajectory_splits))
        if trajectory_split not in seen_splits:  # pragma: no cover - guarded above
            continue

        first = trajectory[0]
        if (
            first.phase not in {"prefill", "cached_prefill"}
            or first.sampled_suffix_tokens != first.full_prefix_tokens
        ):
            raise Lfm2CpcfModelCompilerError(
                f"Trajectory {trajectory_id!r} has no complete causal prefill."
            )
        previous = first
        for phase_index, phase in enumerate(trajectory[1:], start=1):
            expected_phase = f"decode.{phase_index - 1:03d}"
            if phase.phase != expected_phase:
                raise Lfm2CpcfModelCompilerError(
                    f"Trajectory {trajectory_id!r} skips or reorders "
                    f"causal phase {expected_phase!r}."
                )
            if (
                phase.batch_size != previous.batch_size
                or phase.donor_input_sha256 != previous.donor_input_sha256
                or phase.target_alignment_sha256 != previous.target_alignment_sha256
                or phase.full_prefix_tokens
                != previous.full_prefix_tokens + phase.sampled_suffix_tokens
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Trajectory {trajectory_id!r} changed identity or "
                    "target prefix geometry."
                )
            previous_tokens = previous.full_prefix_tokens
            if any(
                not torch.equal(
                    getattr(phase, field)[:, :previous_tokens],
                    getattr(previous, field),
                )
                for field in (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                )
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Trajectory {trajectory_id!r} changed an authenticated "
                    "target-token causal prefix."
                )
            previous_state = bank.states[previous.identity]
            current_state = bank.states[phase.identity]
            if (
                previous_state.dtype != current_state.dtype
                or tuple(previous_state.shape)
                != (
                    phase.batch_size,
                    previous_tokens,
                    bank.hidden_size,
                )
                or not torch.equal(
                    current_state[:, :previous_tokens, :],
                    previous_state,
                )
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Trajectory {trajectory_id!r} changed an authenticated "
                    "actual-child causal state prefix."
                )
            previous = phase
        selected_trajectories.add(trajectory_id)

    selected_phases = tuple(
        phase
        for phase in bank.phases
        if (phase.batch_index, phase.record_id) in selected_trajectories
    )
    if not selected_phases:
        raise Lfm2CpcfModelCompilerError(
            "Requested phase-state splits contain no complete trajectories."
        )
    # ``state()`` returns a clone.  Reconstructing from those clones prevents
    # the selected archive from sharing mutable tensor storage with the source
    # bank while preserving exact authenticated bytes.
    selected_states = {phase.identity: bank.state(phase) for phase in selected_phases}
    return _phase_state_bank(
        bank.boundary,
        bank.layer_index,
        selected_phases,
        selected_states,
    )


def rebind_lfm2_cpcf_phase_state_bank(
    bank: Lfm2CpcfPhaseStateBank,
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Lfm2CpcfPhaseStateBank:
    """Replace evidence-owning phase objects without copying archived states.

    The v5 selective compiler temporarily binds one authenticated source
    layer to otherwise tensor-free phase shells.  Once that layer has been
    compiled, this helper restores the tensor-free shells before the
    residency window releases the payload.  Only phase objects with the exact
    same canonical identities and phase seals are admissible, so the state
    bank seal must remain byte-identical.
    """

    if not isinstance(bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("bank must be Lfm2CpcfPhaseStateBank.")
    ordered = tuple(phases)
    if not ordered or any(
        not isinstance(phase, Lfm2CpcfTargetPhase) for phase in ordered
    ):
        raise TypeError("phases must contain Lfm2CpcfTargetPhase objects.")
    current_lineage = tuple(
        (phase.identity, phase.phase_sha256) for phase in bank.phases
    )
    replacement_lineage = tuple(
        (phase.identity, phase.phase_sha256) for phase in ordered
    )
    if replacement_lineage != current_lineage:
        raise Lfm2CpcfModelCompilerError(
            "Phase-state rebinding changed phase identity, order, or seal."
        )
    rebound = _phase_state_bank(
        bank.boundary,
        bank.layer_index,
        ordered,
        bank.states,
    )
    if rebound.state_bank_sha256 != bank.state_bank_sha256:
        raise Lfm2CpcfModelCompilerError(
            "Phase-state rebinding changed the authenticated bank seal."
        )
    return rebound


def _model_core(model: Any) -> Any:
    core = getattr(model, "model", None)
    if (
        core is None
        or getattr(getattr(model, "config", None), "model_type", None) != "lfm2"
    ):
        raise Lfm2CpcfModelCompilerError(
            "Target model must be a materialized LFM2 causal language model."
        )
    return core


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    """Return the one floating device/dtype used by a target layer."""

    torch = _torch()
    parameters = tuple(layer.parameters())
    if not parameters:
        raise Lfm2CpcfModelCompilerError(
            "Target layer exposes no materialized parameters."
        )
    if any(
        parameter.device.type == "meta"
        or not torch.is_floating_point(parameter)
        or parameter.is_complex()
        for parameter in parameters
    ):
        raise Lfm2CpcfModelCompilerError(
            "Target layer parameters must be materialized real floating tensors."
        )
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise Lfm2CpcfModelCompilerError(
            "Target layer must use one execution device and floating dtype."
        )
    return next(iter(devices)), next(iter(dtypes))


def _named_layer_tensors(
    layer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return alias-complete parameter and buffer inventories."""

    def values(method_name: str) -> dict[str, Any]:
        method = getattr(layer, method_name, None)
        if not callable(method):
            raise Lfm2CpcfModelCompilerError(
                f"Target layer must expose {method_name}()."
            )
        try:
            observed = tuple(method(remove_duplicate=False))
        except TypeError as exc:  # pragma: no cover - unsupported Torch boundary
            raise Lfm2CpcfModelCompilerError(
                f"Target layer {method_name}() does not expose the required "
                "alias-complete inventory."
            ) from exc
        names = [str(name) for name, _value in observed]
        if len(set(names)) != len(names):
            raise Lfm2CpcfModelCompilerError(
                f"Target layer {method_name}() contains duplicate names."
            )
        return {str(name): value for name, value in observed}

    parameters = values("named_parameters")
    buffers = values("named_buffers")
    if set(parameters) & set(buffers):
        raise Lfm2CpcfModelCompilerError(
            "Target layer parameter and buffer names collide."
        )
    return parameters, buffers


def _layer_tensor_alias_partition(
    values: Mapping[str, Any],
    *,
    storage: bool,
) -> tuple[tuple[str, ...], ...]:
    """Describe object or storage aliases without binding device addresses."""

    groups: dict[Any, list[str]] = {}
    for name, value in values.items():
        if storage and int(value.numel()) != 0:
            key = (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        elif storage:
            key = ("empty", id(value))
        else:
            key = id(value)
        groups.setdefault(key, []).append(name)
    return tuple(
        sorted(
            (tuple(sorted(names)) for names in groups.values()),
            key=lambda names: names,
        )
    )


def _unique_parameter_count(model: Any) -> int:
    seen: set[int] = set()
    count = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count += int(parameter.numel())
    return count


def install_lfm2_cpcf_tied_interface(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
) -> Mapping[str, Any]:
    """Install the one dense tied input/output table into the zero blank."""

    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    input_module = target_model.get_input_embeddings()
    output_module = target_model.get_output_embeddings()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or tuple(input_weight.shape) != tuple(interface.target_weight.shape)
    ):
        raise Lfm2CpcfModelCompilerError(
            "Target must expose one native tied table matching the interface."
        )
    identity = id(input_weight)
    data_pointer = int(input_weight.data_ptr())
    shape = tuple(int(size) for size in input_weight.shape)
    unique_before = _unique_parameter_count(target_model)
    with torch.no_grad():
        input_weight.copy_(
            interface.target_weight.to(
                device=input_weight.device,
                dtype=input_weight.dtype,
            )
        )
    installed_sha = _tensor_sha256(input_weight)
    expected_sha = _tensor_sha256(interface.target_weight.to(dtype=input_weight.dtype))
    accepted = (
        target_model.get_input_embeddings().weight
        is target_model.get_output_embeddings().weight
        and id(input_weight) == identity
        and int(input_weight.data_ptr()) == data_pointer
        and tuple(input_weight.shape) == shape
        and _unique_parameter_count(target_model) == unique_before
        and installed_sha == expected_sha
    )
    if not accepted:
        raise Lfm2CpcfModelCompilerError(
            "Installed tied interface failed identity or byte verification."
        )
    body = {
        "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
        "scope": "tied_interface_install",
        "accepted": True,
        "native_tie_preserved": True,
        "parameter_identity_preserved": True,
        "unique_parameter_count_preserved": True,
        "target_weight_sha256": installed_sha,
        "interface_report_sha256": interface.report["report_sha256"],
        "blank_target_values_read": False,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def install_lfm2_cpcf_neutral_norms(
    target_model: Any,
    *,
    attention_qk_norm_weights: Mapping[int, tuple[Any, Any]] | None = None,
) -> Mapping[str, Any]:
    """Install explicit functional coordinates for every zeroed RMSNorm.

    Residual and final normalizers receive unit gains.  Preserved-head-dimension
    backends may supply source-derived Q/K gains per attention layer; otherwise
    those gains are also unit vectors.  This is a deterministic installation,
    not a learned parameter update.
    """

    torch = _torch()
    core = _model_core(target_model)
    supplied = dict(attention_qk_norm_weights or {})
    layers = tuple(core.layers)
    if set(supplied) - set(range(len(layers))):
        raise Lfm2CpcfModelCompilerError(
            "Q/K norm weights reference an unknown target layer."
        )
    parameters: list[tuple[str, Any, Any]] = []
    for index, layer in enumerate(layers):
        for role in ("operator_norm", "ffn_norm"):
            module = getattr(layer, role, None)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                raise Lfm2CpcfModelCompilerError(
                    f"Target layer {index} lacks {role}.weight."
                )
            parameters.append((f"layers.{index}.{role}.weight", weight, None))
        if bool(getattr(layer, "is_attention_layer", False)):
            attention = getattr(layer, "self_attn", None)
            values = supplied.get(index, (None, None))
            for role, source in zip(
                ("q_layernorm", "k_layernorm"),
                values,
            ):
                weight = getattr(getattr(attention, role, None), "weight", None)
                if not isinstance(weight, torch.nn.Parameter):
                    raise Lfm2CpcfModelCompilerError(
                        f"Target layer {index} lacks {role}.weight."
                    )
                parameters.append(
                    (f"layers.{index}.self_attn.{role}.weight", weight, source)
                )
    final_weight = getattr(getattr(core, "embedding_norm", None), "weight", None)
    if not isinstance(final_weight, torch.nn.Parameter):
        raise Lfm2CpcfModelCompilerError(
            "Target LFM2 model lacks embedding_norm.weight."
        )
    parameters.append(("embedding_norm.weight", final_weight, None))
    identities = {
        name: (id(weight), int(weight.data_ptr()), tuple(weight.shape))
        for name, weight, _source in parameters
    }
    installed = {}
    with torch.no_grad():
        for name, weight, source in parameters:
            value = torch.ones_like(weight)
            if source is not None:
                if (
                    not isinstance(source, torch.Tensor)
                    or tuple(source.shape) != tuple(weight.shape)
                    or not torch.is_floating_point(source)
                    or not bool(torch.isfinite(source).all().item())
                ):
                    raise Lfm2CpcfModelCompilerError(
                        f"Source-derived {name} has the wrong shape or values."
                    )
                value = source.to(device=weight.device, dtype=weight.dtype)
            weight.copy_(value)
            installed[name] = _tensor_sha256(weight)
    if any(
        identities[name] != (id(weight), int(weight.data_ptr()), tuple(weight.shape))
        for name, weight, _source in parameters
    ):
        raise Lfm2CpcfModelCompilerError(
            "Neutral norm installation replaced a target Parameter."
        )
    body = {
        "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
        "scope": "neutral_norm_install",
        "accepted": True,
        "installed_norm_count": len(parameters),
        "source_derived_qk_norm_layer_indices": sorted(supplied),
        "parameter_identities_preserved": True,
        "installed_sha256": installed,
        "blank_target_values_read": False,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def seed_lfm2_cpcf_phase_states(
    target_model: Any,
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Lfm2CpcfPhaseStateBank:
    """Lookup the installed target table directly for every full prefix."""

    torch = _torch()
    ordered = tuple(phases)
    if not ordered:
        raise Lfm2CpcfModelCompilerError("Target phases cannot be empty.")
    weight = getattr(target_model.get_input_embeddings(), "weight", None)
    if not isinstance(weight, torch.nn.Parameter):
        raise Lfm2CpcfModelCompilerError(
            "Target has no materialized input embedding table."
        )
    states = {}
    with torch.no_grad():
        for phase in ordered:
            ids = phase.input_ids.to(device=weight.device)
            if bool((ids >= int(weight.shape[0])).any().item()):
                raise Lfm2CpcfModelCompilerError(
                    f"Target phase {phase.identity!r} exceeds the table."
                )
            live_state = (
                weight.index_select(0, ids.reshape(-1))
                .reshape(
                    phase.batch_size,
                    phase.full_prefix_tokens,
                    int(weight.shape[1]),
                )
                .detach()
            )
            states[phase.identity] = _floating_states(
                live_state,
                f"seed_states[{phase.identity!r}]",
                hidden_size=int(weight.shape[1]),
                copy=False,
            )
            del live_state, ids
    return _phase_state_bank(
        "input_embeddings",
        0,
        ordered,
        states,
    )


def _rope_from_target(
    target_model: Any,
    residual: Any,
    position_ids: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    rotary = getattr(_model_core(target_model), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", None)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or isinstance(scaling, bool)
    ):
        raise Lfm2CpcfModelCompilerError(
            "Target rotary embedding lacks a static inv_freq contract."
        )
    try:
        scale = float(scaling)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfModelCompilerError(
            "Target RoPE attention scaling is invalid."
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise Lfm2CpcfModelCompilerError(
            "Target RoPE attention scaling must be positive and finite."
        )
    positions = position_ids.to(device=residual.device, dtype=torch.float32)
    frequencies = positions.unsqueeze(-1) * inv_freq.to(
        device=residual.device, dtype=torch.float32
    ).reshape(1, 1, -1)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        (angles.cos() * scale).to(dtype=residual.dtype),
        (angles.sin() * scale).to(dtype=residual.dtype),
    )


def _additive_causal_mask(phase: Lfm2CpcfTargetPhase, residual: Any) -> Any:
    torch = _torch()
    sequence = phase.full_prefix_tokens
    allowed = torch.tril(
        torch.ones(
            (sequence, sequence),
            device=residual.device,
            dtype=torch.bool,
        )
    )
    active_keys = phase.attention_mask.to(
        device=residual.device,
        dtype=torch.bool,
    )[:, None, None, :]
    allowed = allowed[None, None, :, :] & active_keys
    minimum = torch.finfo(residual.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=residual.device, dtype=residual.dtype),
        torch.full(
            (),
            minimum,
            device=residual.device,
            dtype=residual.dtype,
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfLayerReplayBank:
    """Every actual-child replay for one installed target layer."""

    layer_index: int
    input_bank_sha256: str
    replays: Mapping[tuple[int, str], Lfm2CpcfMacrocellReplay]
    post_layer_bank: Lfm2CpcfPhaseStateBank
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonnegative_int(self.layer_index, "layer_index")
        _digest(self.input_bank_sha256, "input_bank_sha256")
        if set(self.replays) != set(self.post_layer_bank.states):
            raise Lfm2CpcfModelCompilerError(
                "Layer replay and post-layer state identities differ."
            )
        if not all(
            isinstance(value, Lfm2CpcfMacrocellReplay)
            for value in self.replays.values()
        ):
            raise Lfm2CpcfModelCompilerError(
                "Layer replay bank contains an invalid replay."
            )
        report = dict(self.report)
        report_body = dict(report)
        report_sha256 = report_body.pop("report_sha256", None)
        if report_sha256 != _sha(report_body):
            raise Lfm2CpcfModelCompilerError("Layer replay report seal is invalid.")

        def finite_nonnegative(value: Any, name: str) -> float:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"{name} must be finite and nonnegative."
                )
            return float(value)

        expected_replays = [
            {
                "identity": [phase.batch_index, phase.phase],
                "replay_sha256": self.replays[phase.identity].replay_sha256,
            }
            for phase in self.post_layer_bank.phases
        ]
        stitch_records = report.get("causal_stitch_records")
        if not isinstance(stitch_records, list) or len(stitch_records) != len(
            self.post_layer_bank.phases
        ):
            raise Lfm2CpcfModelCompilerError(
                "Layer replay report does not contain one ordered causal "
                "stitch record per phase."
            )
        torch = _torch()
        record_fields = {
            "identity",
            "source",
            "prefix_tokens",
            "suffix_tokens",
            "raw_prefix_relative_rmse",
            "raw_prefix_maximum_absolute_error",
            "raw_prefix_permitted_relative_error",
            "raw_prefix_permitted_maximum_absolute_error",
            "raw_replay_sha256",
        }
        trajectories: dict[
            tuple[int, str],
            tuple[Lfm2CpcfTargetPhase, Any],
        ] = {}
        relative_errors = []
        maximum_absolute_errors = []
        decode_count = 0
        prefill_count = 0
        for phase, raw_record in zip(
            self.post_layer_bank.phases,
            stitch_records,
            strict=True,
        ):
            if not isinstance(raw_record, dict):
                raise Lfm2CpcfModelCompilerError(
                    "A causal stitch record is not canonical metadata."
                )
            record = dict(raw_record)
            if set(record) != record_fields:
                raise Lfm2CpcfModelCompilerError(
                    "A causal stitch record has an invalid field inventory."
                )
            numeric_fields = (
                "raw_prefix_relative_rmse",
                "raw_prefix_maximum_absolute_error",
                "raw_prefix_permitted_relative_error",
                "raw_prefix_permitted_maximum_absolute_error",
            )
            numeric_values = {}
            for name in numeric_fields:
                numeric_values[name] = finite_nonnegative(
                    record.get(name),
                    f"causal_stitch_records.{name}",
                )
            identity = [phase.batch_index, phase.phase]
            replay = self.replays[phase.identity]
            canonical_state = self.post_layer_bank.states[phase.identity]
            raw_post_layer = replay.post_layer_residual
            if (
                record.get("identity") != identity
                or record.get("raw_replay_sha256") != replay.replay_sha256
                or tuple(canonical_state.shape) != tuple(raw_post_layer.shape)
                or canonical_state.dtype != raw_post_layer.dtype
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Causal stitch record {phase.identity!r} differs from "
                    "its archived replay geometry."
                )
            _digest(
                record["raw_replay_sha256"],
                "raw_replay_sha256",
            )
            trajectory_key = (phase.batch_index, phase.record_id)
            previous = trajectories.get(trajectory_key)
            prefix_tokens = _nonnegative_int(
                record.get("prefix_tokens"),
                "causal_stitch_records.prefix_tokens",
            )
            suffix_tokens = _positive_int(
                record.get("suffix_tokens"),
                "causal_stitch_records.suffix_tokens",
            )
            if phase.phase in {"prefill", "cached_prefill"}:
                if (
                    previous is not None
                    or record.get("source") != "complete_prefill_replay"
                    or prefix_tokens != 0
                    or suffix_tokens != phase.full_prefix_tokens
                    or any(numeric_values.values())
                    or not torch.equal(canonical_state, raw_post_layer)
                ):
                    raise Lfm2CpcfModelCompilerError(
                        f"Causal prefill stitch {phase.identity!r} is invalid."
                    )
                prefill_count += 1
            else:
                match = _DECODE_PHASE.fullmatch(phase.phase)
                if match is None or previous is None:
                    raise Lfm2CpcfModelCompilerError(
                        f"Causal decode stitch {phase.identity!r} has no "
                        "ordered predecessor."
                    )
                previous_phase, previous_state = previous
                previous_match = _DECODE_PHASE.fullmatch(previous_phase.phase)
                expected_step = (
                    0
                    if previous_phase.phase in {"prefill", "cached_prefill"}
                    else (
                        -1
                        if previous_match is None
                        else int(previous_match.group(1)) + 1
                    )
                )
                expected_prefix = phase.full_prefix_tokens - phase.sampled_suffix_tokens
                raw_prefix = raw_post_layer[:, :expected_prefix, :]
                delta = raw_prefix.to(torch.float64) - previous_state.to(torch.float64)
                reference_rms = (
                    previous_state.to(torch.float64)
                    .square()
                    .mean()
                    .sqrt()
                    .clamp_min(torch.finfo(torch.float64).tiny)
                )
                measured_relative_rmse = float(
                    (delta.square().mean().sqrt() / reference_rms).item()
                )
                measured_maximum_absolute_error = float(delta.abs().max().item())
                if raw_post_layer.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }:
                    measured_permitted_relative_error = float(
                        8.0 * torch.finfo(raw_post_layer.dtype).eps
                    )
                else:
                    measured_permitted_relative_error = 1e-5
                amplitude = max(
                    1.0,
                    float(previous_state.abs().max().item()),
                    float(raw_prefix.abs().max().item()),
                )
                measured_permitted_maximum_absolute_error = (
                    measured_permitted_relative_error * amplitude
                )
                if numeric_values != {
                    "raw_prefix_relative_rmse": measured_relative_rmse,
                    "raw_prefix_maximum_absolute_error": (
                        measured_maximum_absolute_error
                    ),
                    "raw_prefix_permitted_relative_error": (
                        measured_permitted_relative_error
                    ),
                    "raw_prefix_permitted_maximum_absolute_error": (
                        measured_permitted_maximum_absolute_error
                    ),
                }:
                    raise Lfm2CpcfModelCompilerError(
                        f"Causal decode stitch {phase.identity!r} "
                        "contains unauthenticated drift metrics."
                    )
                if (
                    int(match.group(1)) != expected_step
                    or phase.split != previous_phase.split
                    or previous_phase.full_prefix_tokens != expected_prefix
                    or not torch.equal(
                        phase.input_ids[:, :expected_prefix],
                        previous_phase.input_ids,
                    )
                    or not torch.equal(
                        phase.attention_mask[:, :expected_prefix],
                        previous_phase.attention_mask,
                    )
                    or not torch.equal(
                        phase.position_ids[:, :expected_prefix],
                        previous_phase.position_ids,
                    )
                    or record.get("source") != "ordered_decode_terminal_suffix"
                    or prefix_tokens != expected_prefix
                    or suffix_tokens != phase.sampled_suffix_tokens
                    or numeric_values["raw_prefix_permitted_relative_error"] <= 0.0
                    or numeric_values["raw_prefix_permitted_maximum_absolute_error"]
                    <= 0.0
                    or numeric_values["raw_prefix_relative_rmse"]
                    > numeric_values["raw_prefix_permitted_relative_error"]
                    or numeric_values["raw_prefix_maximum_absolute_error"]
                    > numeric_values["raw_prefix_permitted_maximum_absolute_error"]
                    or not torch.equal(
                        canonical_state[:, :expected_prefix, :],
                        previous_state,
                    )
                    or not torch.equal(
                        canonical_state[:, expected_prefix:, :],
                        raw_post_layer[:, expected_prefix:, :],
                    )
                ):
                    raise Lfm2CpcfModelCompilerError(
                        f"Causal decode stitch {phase.identity!r} is invalid."
                    )
                decode_count += 1
            trajectories[trajectory_key] = (phase, canonical_state)
            relative_errors.append(numeric_values["raw_prefix_relative_rmse"])
            maximum_absolute_errors.append(
                numeric_values["raw_prefix_maximum_absolute_error"]
            )
        decoded_relative_errors = [
            float(record["raw_prefix_relative_rmse"])
            for record in stitch_records
            if record["prefix_tokens"] > 0
        ]
        relative_mean = sum(decoded_relative_errors) / max(
            1,
            len(decoded_relative_errors),
        )
        stitch_bundle_sha256 = report.get("causal_stitch_bundle_sha256")
        _digest(
            stitch_bundle_sha256,
            "causal_stitch_bundle_sha256",
        )
        reported_trajectory_count = _nonnegative_int(
            report.get("causal_trajectory_count"),
            "causal_trajectory_count",
        )
        reported_decode_count = _nonnegative_int(
            report.get("decode_suffix_stitch_count"),
            "decode_suffix_stitch_count",
        )
        reported_relative_mean = finite_nonnegative(
            report.get("raw_prefix_relative_rmse_mean"),
            "raw_prefix_relative_rmse_mean",
        )
        reported_relative_worst = finite_nonnegative(
            report.get("raw_prefix_relative_rmse_worst"),
            "raw_prefix_relative_rmse_worst",
        )
        reported_absolute_worst = finite_nonnegative(
            report.get("raw_prefix_maximum_absolute_error_worst"),
            "raw_prefix_maximum_absolute_error_worst",
        )
        stitch_report_valid = (
            prefill_count == len(trajectories)
            and reported_trajectory_count == len(trajectories)
            and reported_decode_count == decode_count
            and reported_relative_mean == relative_mean
            and reported_relative_worst == max(relative_errors, default=0.0)
            and reported_absolute_worst == max(maximum_absolute_errors, default=0.0)
            and stitch_bundle_sha256 == _sha(stitch_records)
        )
        if (
            report.get("schema") != LFM2_CPCF_TARGET_RUNTIME_SCHEMA
            or report.get("scope") != "sequential_installed_macrocell_replay"
            or report.get("layer_index") != self.layer_index
            or report.get("input_bank_sha256") != self.input_bank_sha256
            or report.get("post_layer_bank_sha256")
            != self.post_layer_bank.state_bank_sha256
            or self.post_layer_bank.layer_index != self.layer_index + 1
            or self.post_layer_bank.boundary
            != f"layer.{self.layer_index:03d}.post_layer"
            or report.get("ordered_replay_sha256") != expected_replays
            or not stitch_report_valid
            or report.get("actual_folded_upstream_input") is not True
            or report.get("full_prefix_replay") is not True
            or report.get("raw_full_prefix_replay_archived") is not True
            or report.get("persistent_replay_storage_device") != "cpu"
            or report.get("completed_phase_accelerator_storage_retained") is not False
            or report.get("canonical_causal_post_layer_state_stitching") is not True
            or report.get("post_layer_state_assembly")
            != "complete_prefill_plus_ordered_decode_suffixes"
            or report.get("raw_recomputed_historical_prefix_used") is not False
        ):
            raise Lfm2CpcfModelCompilerError(
                "Layer replay archive differs from its authenticated report."
            )
        object.__setattr__(
            self,
            "replays",
            MappingProxyType(dict(self.replays)),
        )
        object.__setattr__(self, "report", _freeze_json(report))

    def as_dict(self) -> dict[str, Any]:
        return _thaw_json(self.report)


def replay_lfm2_cpcf_phase_bank(
    target_model: Any,
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    execution_layer: Any | None = None,
) -> Lfm2CpcfLayerReplayBank:
    """Replay one installed macrocell on all actual folded-upstream prefixes.

    ``execution_layer`` is an explicit layer-local execution override used by
    bounded candidate evaluators.  Omitting it retains the canonical serial
    path exactly: the layer is read from ``target_model`` as before.  The
    override never substitutes a model, changes the causal input bank, or
    permits a topology different from the canonical target layer.
    """

    torch = _torch()
    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("input_bank must be Lfm2CpcfPhaseStateBank.")
    index = _nonnegative_int(layer_index, "layer_index")
    layers = tuple(_model_core(target_model).layers)
    if index >= len(layers):
        raise Lfm2CpcfModelCompilerError("layer_index is outside the target model.")
    if input_bank.layer_index != index:
        raise Lfm2CpcfModelCompilerError(
            "Input state bank is not the next sequential layer boundary."
        )
    canonical_layer = layers[index]
    layer = canonical_layer if execution_layer is None else execution_layer
    if execution_layer is not None:
        torch = _torch()
        canonical_parameters, canonical_buffers = _named_layer_tensors(
            canonical_layer
        )
        execution_parameters, execution_buffers = _named_layer_tensors(layer)
        canonical_tensors = {
            **{
                f"parameter::{name}": value
                for name, value in canonical_parameters.items()
            },
            **{
                f"buffer::{name}": value
                for name, value in canonical_buffers.items()
            },
        }
        execution_tensors = {
            **{
                f"parameter::{name}": value
                for name, value in execution_parameters.items()
            },
            **{
                f"buffer::{name}": value
                for name, value in execution_buffers.items()
            },
        }
        execution_devices = {
            value.device for value in execution_tensors.values()
        }
        if (
            layer is canonical_layer
            or bool(getattr(layer, "is_attention_layer", False))
            != bool(getattr(canonical_layer, "is_attention_layer", False))
            or not canonical_parameters
            or set(execution_parameters) != set(canonical_parameters)
            or set(execution_buffers) != set(canonical_buffers)
            or set(execution_tensors) != set(canonical_tensors)
            or len(execution_devices) != 1
            or any(
                not isinstance(actual, torch.Tensor)
                or not isinstance(expected, torch.Tensor)
                or tuple(actual.shape) != tuple(expected.shape)
                or actual.dtype != expected.dtype
                or actual.layout != expected.layout
                or tuple(actual.stride()) != tuple(expected.stride())
                or int(actual.storage_offset())
                != int(expected.storage_offset())
                for name, expected in canonical_tensors.items()
                for actual in (execution_tensors[name],)
            )
            or _layer_tensor_alias_partition(
                canonical_tensors,
                storage=False,
            )
            != _layer_tensor_alias_partition(
                execution_tensors,
                storage=False,
            )
            or _layer_tensor_alias_partition(
                canonical_tensors,
                storage=True,
            )
            != _layer_tensor_alias_partition(
                execution_tensors,
                storage=True,
            )
            or any(
                bool(execution_parameters[name].requires_grad)
                != bool(canonical_parameters[name].requires_grad)
                for name in canonical_parameters
            )
        ):
            raise Lfm2CpcfModelCompilerError(
                "execution_layer must be one private topology-identical "
                "parameter-and-buffer target layer replica."
            )
        canonical_storage = {
            (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
            for value in canonical_tensors.values()
            if value.numel() != 0
        }
        execution_storage = {
            (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
            for value in execution_tensors.values()
            if value.numel() != 0
        }
        if canonical_storage & execution_storage:
            raise Lfm2CpcfModelCompilerError(
                "execution_layer shares parameter storage or buffer storage "
                "with the canonical target layer."
            )
        if any(
            value.device != next(iter(execution_devices))
            for value in execution_tensors.values()
        ):
            raise Lfm2CpcfModelCompilerError(
                "execution_layer parameters and buffers do not share one "
                "execution device."
            )
    runtime_device, runtime_dtype = _layer_runtime(layer)
    replays = {}
    post_states = {}
    causal_trajectories: dict[
        tuple[int, str],
        tuple[Lfm2CpcfTargetPhase, Any],
    ] = {}
    stitch_records = []
    for phase in input_bank.phases:
        # Stage exactly one phase on the selected execution device.  Both the
        # input bank and the returned replay are canonical CPU archives, so no
        # completed phase or historical layer remains accelerator resident.
        archived_state = input_bank.states[phase.identity]
        if archived_state.device.type != "cpu":
            raise Lfm2CpcfModelCompilerError(
                "Input phase bank is not a canonical CPU archive."
            )
        if archived_state.dtype != runtime_dtype:
            raise Lfm2CpcfModelCompilerError(
                "Archived state dtype differs from the target layer runtime."
            )
        residual = archived_state.to(
            device=runtime_device,
            non_blocking=False,
        )
        if bool(getattr(layer, "is_attention_layer", False)):
            position_embeddings = _rope_from_target(
                target_model,
                residual,
                phase.position_ids,
            )
            mask = _additive_causal_mask(phase, residual)
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=position_embeddings,
                additive_causal_mask=mask,
            )
            del position_embeddings, mask
        else:
            padding_mask = phase.attention_mask.to(
                device=residual.device,
                dtype=residual.dtype,
            )
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=padding_mask,
            )
            del padding_mask
        del residual
        if any(
            value.device.type != "cpu"
            for value in (
                replay.operator_input,
                replay.operator_delta,
                replay.post_operator_residual,
                replay.mlp_input,
                replay.mlp_delta,
                replay.post_layer_residual,
            )
        ):
            raise Lfm2CpcfModelCompilerError(
                "Completed replay endpoints retained execution-device storage."
            )
        replays[phase.identity] = replay
        trajectory_key = (phase.batch_index, phase.record_id)
        previous = causal_trajectories.get(trajectory_key)
        raw_post_layer = replay.post_layer_residual
        if phase.phase in {"prefill", "cached_prefill"}:
            if previous is not None:
                raise Lfm2CpcfModelCompilerError(
                    f"Target trajectory {trajectory_key!r} contains more than "
                    "one causal prefill."
                )
            canonical_post_layer = raw_post_layer
            stitch_records.append(
                {
                    "identity": [phase.batch_index, phase.phase],
                    "source": "complete_prefill_replay",
                    "prefix_tokens": 0,
                    "suffix_tokens": phase.full_prefix_tokens,
                    "raw_prefix_relative_rmse": 0.0,
                    "raw_prefix_maximum_absolute_error": 0.0,
                    "raw_prefix_permitted_relative_error": 0.0,
                    "raw_prefix_permitted_maximum_absolute_error": 0.0,
                    "raw_replay_sha256": replay.replay_sha256,
                }
            )
        else:
            match = _DECODE_PHASE.fullmatch(phase.phase)
            if match is None or previous is None:
                raise Lfm2CpcfModelCompilerError(
                    f"Target trajectory {trajectory_key!r} is missing its "
                    "ordered causal prefill/decode predecessor."
                )
            previous_phase, previous_post_layer = previous
            previous_step = _DECODE_PHASE.fullmatch(previous_phase.phase)
            expected_step = (
                0
                if previous_phase.phase in {"prefill", "cached_prefill"}
                else (-1 if previous_step is None else int(previous_step.group(1)) + 1)
            )
            step = int(match.group(1))
            previous_tokens = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            if (
                step != expected_step
                or phase.split != previous_phase.split
                or previous_phase.full_prefix_tokens != previous_tokens
                or tuple(previous_post_layer.shape)
                != (
                    phase.batch_size,
                    previous_tokens,
                    int(raw_post_layer.shape[2]),
                )
                or previous_post_layer.dtype != raw_post_layer.dtype
                or not torch.equal(
                    phase.input_ids[:, :previous_tokens],
                    previous_phase.input_ids,
                )
                or not torch.equal(
                    phase.attention_mask[:, :previous_tokens],
                    previous_phase.attention_mask,
                )
                or not torch.equal(
                    phase.position_ids[:, :previous_tokens],
                    previous_phase.position_ids,
                )
                or not torch.equal(
                    archived_state[:, :previous_tokens, :],
                    input_bank.states[previous_phase.identity],
                )
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Target trajectory {trajectory_key!r} changed or skipped "
                    "its authenticated causal prefix."
                )
            raw_prefix = raw_post_layer[:, :previous_tokens, :]
            delta = raw_prefix.to(torch.float64) - previous_post_layer.to(torch.float64)
            reference_rms = (
                previous_post_layer.to(torch.float64)
                .square()
                .mean()
                .sqrt()
                .clamp_min(torch.finfo(torch.float64).tiny)
            )
            prefix_relative_rmse = float(
                (delta.square().mean().sqrt() / reference_rms).item()
            )
            prefix_maximum_absolute_error = float(delta.abs().max().item())
            if raw_post_layer.dtype in {torch.float16, torch.bfloat16}:
                permitted_relative_error = float(
                    8.0 * torch.finfo(raw_post_layer.dtype).eps
                )
            else:
                permitted_relative_error = 1e-5
            amplitude = max(
                1.0,
                float(previous_post_layer.abs().max().item()),
                float(raw_prefix.abs().max().item()),
            )
            permitted_maximum_absolute_error = permitted_relative_error * amplitude
            if (
                not math.isfinite(prefix_relative_rmse)
                or not math.isfinite(prefix_maximum_absolute_error)
                or prefix_relative_rmse > permitted_relative_error
                or prefix_maximum_absolute_error > permitted_maximum_absolute_error
            ):
                raise Lfm2CpcfModelCompilerError(
                    f"Target phase {phase.identity!r} raw full-prefix replay "
                    "is not numerically invariant on its historical causal "
                    "rows "
                    f"(relative_rmse={prefix_relative_rmse:.9g}, "
                    f"relative_bound={permitted_relative_error:.9g}, "
                    f"maximum_absolute_error="
                    f"{prefix_maximum_absolute_error:.9g}, "
                    f"maximum_absolute_bound="
                    f"{permitted_maximum_absolute_error:.9g})."
                )
            canonical_post_layer = torch.cat(
                (
                    previous_post_layer,
                    raw_post_layer[:, -phase.sampled_suffix_tokens :, :],
                ),
                dim=1,
            ).contiguous()
            stitch_records.append(
                {
                    "identity": [phase.batch_index, phase.phase],
                    "source": "ordered_decode_terminal_suffix",
                    "prefix_tokens": previous_tokens,
                    "suffix_tokens": phase.sampled_suffix_tokens,
                    "raw_prefix_relative_rmse": prefix_relative_rmse,
                    "raw_prefix_maximum_absolute_error": (
                        prefix_maximum_absolute_error
                    ),
                    "raw_prefix_permitted_relative_error": (permitted_relative_error),
                    "raw_prefix_permitted_maximum_absolute_error": (
                        permitted_maximum_absolute_error
                    ),
                    "raw_replay_sha256": replay.replay_sha256,
                }
            )
        if tuple(canonical_post_layer.shape) != (
            phase.batch_size,
            phase.full_prefix_tokens,
            int(raw_post_layer.shape[2]),
        ):
            raise Lfm2CpcfModelCompilerError(
                f"Target phase {phase.identity!r} did not produce one complete "
                "canonical causal post-layer state."
            )
        post_states[phase.identity] = canonical_post_layer
        causal_trajectories[trajectory_key] = (
            phase,
            canonical_post_layer,
        )
    post_bank = _phase_state_bank(
        f"layer.{index:03d}.post_layer",
        index + 1,
        input_bank.phases,
        post_states,
    )
    body = {
        "schema": LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
        "scope": "sequential_installed_macrocell_replay",
        "layer_index": index,
        "input_bank_sha256": input_bank.state_bank_sha256,
        "post_layer_bank_sha256": post_bank.state_bank_sha256,
        "ordered_replay_sha256": [
            {
                "identity": [phase.batch_index, phase.phase],
                "replay_sha256": replays[phase.identity].replay_sha256,
            }
            for phase in input_bank.phases
        ],
        "actual_folded_upstream_input": True,
        "full_prefix_replay": True,
        "raw_full_prefix_replay_archived": True,
        "canonical_causal_post_layer_state_stitching": True,
        "post_layer_state_assembly": ("complete_prefill_plus_ordered_decode_suffixes"),
        "raw_recomputed_historical_prefix_used": False,
        "causal_trajectory_count": len(causal_trajectories),
        "decode_suffix_stitch_count": sum(
            record["source"] == "ordered_decode_terminal_suffix"
            for record in stitch_records
        ),
        "raw_prefix_relative_rmse_mean": (
            sum(
                record["raw_prefix_relative_rmse"]
                for record in stitch_records
                if record["prefix_tokens"] > 0
            )
            / max(
                1,
                sum(record["prefix_tokens"] > 0 for record in stitch_records),
            )
        ),
        "raw_prefix_relative_rmse_worst": max(
            (record["raw_prefix_relative_rmse"] for record in stitch_records),
            default=0.0,
        ),
        "raw_prefix_maximum_absolute_error_worst": max(
            (record["raw_prefix_maximum_absolute_error"] for record in stitch_records),
            default=0.0,
        ),
        "causal_stitch_records": stitch_records,
        "causal_stitch_bundle_sha256": _sha(stitch_records),
        "execution_device": str(runtime_device),
        "execution_dtype": str(runtime_dtype).removeprefix("torch."),
        "persistent_replay_storage_device": "cpu",
        "completed_phase_accelerator_storage_retained": False,
        "independent_parent_projection_substitution": False,
        "parent_forward_count": 0,
        "child_model_forward_count": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    if execution_layer is not None:
        body = {
            **body,
            "execution_layer_override": True,
            "canonical_model_layer_executed": False,
            "private_layer_replica_required": True,
            "private_layer_replica_storage_isolated": True,
            "private_layer_replica_parameter_and_buffer_topology_verified": (
                True
            ),
            "private_layer_replica_parameter_and_buffer_aliases_preserved": (
                True
            ),
            "private_layer_replica_parameter_and_buffer_storage_isolated": (
                True
            ),
        }
    report = {**body, "report_sha256": _sha(body)}
    return Lfm2CpcfLayerReplayBank(
        layer_index=index,
        input_bank_sha256=input_bank.state_bank_sha256,
        replays=replays,
        post_layer_bank=post_bank,
        report=report,
    )


__all__ = [
    "LFM2_CPCF_TARGET_PHASE_SCHEMA",
    "LFM2_CPCF_TARGET_RUNTIME_SCHEMA",
    "Lfm2CpcfLayerReplayBank",
    "Lfm2CpcfModelCompilerError",
    "Lfm2CpcfPhaseStateBank",
    "Lfm2CpcfTargetPhase",
    "build_lfm2_cpcf_target_phases",
    "install_lfm2_cpcf_neutral_norms",
    "install_lfm2_cpcf_tied_interface",
    "rebind_lfm2_cpcf_phase_state_bank",
    "replay_lfm2_cpcf_phase_bank",
    "select_lfm2_cpcf_phase_state_bank",
    "seed_lfm2_cpcf_phase_states",
]
