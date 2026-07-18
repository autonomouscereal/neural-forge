"""Opt-in bounded-residency LFM2 CPCF v5 model compiler.

The legacy v4 compiler intentionally remains untouched and is still the
default production lane.  This module is a separately fingerprinted v5 lane
which:

* keeps canonical target-phase metadata in tensor-free evidence shells;
* loads only one authenticated source layer plus its prefetched successor;
* fits each boundary chart immediately before compiling that same layer;
* drops tensor-owning replay results before advancing the residency window;
* optionally routes short-conv candidate replay through an injected
  authenticated evaluator; and
* continuously samples physical process RSS in addition to reporting logical
  evidence bytes.

It consumes an already committed local trace cache through
``Lfm2CpcfSelectiveEvidenceProvider``.  It never runs a parent, trains,
prunes, quantizes, or changes the v4 byte/provenance contract.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_evidence_residency import (
    DynamicEvidenceLayerAuthentication,
    DynamicEvidenceResidencyWindow,
)
from neural_forge.hf_cpcf_boundary_chart import fit_cpcf_boundary_chart
from neural_forge import hf_lfm2_cpcf_boundary_bundle as boundary
from neural_forge import hf_lfm2_cpcf_model_compiler as model_compiler
from neural_forge import hf_lfm2_cpcf_model_fold as legacy_model_fold
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    Lfm2CpcfCandidateReplayEvaluator,
)
from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (
    compile_and_install_lfm2_cpcf_layer,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierAdapter,
    Lfm2CpcfShortConvV5CandidateSelector,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    rescore_sealed_v4_short_conv_selection_frontier,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    LFM2_CPCF_TARGET_PHASE_SCHEMA,
    Lfm2CpcfTargetPhase,
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
    rebind_lfm2_cpcf_phase_state_bank,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    Lfm2CpcfSelectiveEvidenceCatalog,
    Lfm2CpcfSelectiveEvidenceProvider,
    Lfm2CpcfSelectiveFinalNormPayload,
    Lfm2CpcfSelectiveLayerPayload,
    lfm2_cpcf_selective_provider_implementation,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (
    fit_tied_causal_interface,
)


LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-compiler-implementation-v1"
)
LFM2_CPCF_V5_MODEL_FOLD_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-selective-model-fold-v1"
)
LFM2_CPCF_V5_RESUME_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-selective-resume-v1"
)
LFM2_CPCF_V5_ARTIFACT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-selective-artifact-v1"
)
LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
_SPLITS = ("selection", "fit", "validation")
_BOUNDARIES = ("residual.post_operator", "residual.post_layer")
_RSS_SAMPLE_INTERVAL_SECONDS = 0.05


class Lfm2CpcfV5ModelCompilerError(RuntimeError):
    """Selective evidence, physical residency, or v5 compilation failed."""


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5EndpointFirstCandidateSelector:
    """Apply the source-bound endpoint-first v5 policy before installation."""

    def select(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("request must be an authenticated mapping.")
        reports = request.get("candidate_selection_reports")
        if not isinstance(reports, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                "Candidate-selector request has no sealed selection frontier."
            )
        return rescore_sealed_v4_short_conv_selection_frontier(reports)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF v5 model compilation."
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
        raise Lfm2CpcfV5ModelCompilerError(
            "V5 compiler metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be within [0, 1]."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be within [0, 1]."
        ) from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be within [0, 1]."
        )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Cannot fingerprint compiler source {path}."
        ) from exc
    return digest.hexdigest()


def lfm2_cpcf_v5_compiler_implementation() -> Mapping[str, Any]:
    """Return the source-bound v5 compiler and resume/artifact contract."""

    from neural_forge import dynamic_evidence_residency
    from neural_forge import hf_lfm2_cpcf_candidate_replay
    from neural_forge import hf_lfm2_cpcf_layer_compiler
    from neural_forge import hf_lfm2_cpcf_model_compiler
    from neural_forge import hf_lfm2_cpcf_short_conv_v5
    from neural_forge import hf_lfm2_cpcf_short_conv_v4_fold
    from neural_forge import hf_lfm2_cpcf_selective_evidence

    modules = (
        sys.modules[__name__],
        boundary,
        dynamic_evidence_residency,
        hf_lfm2_cpcf_candidate_replay,
        hf_lfm2_cpcf_layer_compiler,
        hf_lfm2_cpcf_model_compiler,
        legacy_model_fold,
        hf_lfm2_cpcf_short_conv_v5,
        hf_lfm2_cpcf_short_conv_v4_fold,
        hf_lfm2_cpcf_selective_evidence,
    )
    files = []
    for module in modules:
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Compiler module {module.__name__!r} has no source file."
            )
        path = Path(raw).resolve()
        files.append(
            {
                "module": str(module.__name__),
                "filename": path.name,
                "sha256": _file_sha256(path),
            }
        )
    files.sort(key=lambda row: row["module"])
    short_conv_v5_source = next(
        row
        for row in files
        if row["module"]
        == "neural_forge.hf_lfm2_cpcf_short_conv_v5"
    )
    provider = dict(lfm2_cpcf_selective_provider_implementation())
    body = {
        "schema": LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA,
        "model_fold_schema": LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
        "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "artifact_schema": LFM2_CPCF_V5_ARTIFACT_SCHEMA,
        "ordered_source_files": files,
        "selective_provider_implementation_sha256": (
            provider["implementation_sha256"]
        ),
        "short_conv_v5_numerical_policy_schema": (
            hf_lfm2_cpcf_short_conv_v5.LFM2_CPCF_SHORT_CONV_V5_SCHEMA
        ),
        "short_conv_v5_numerical_policy_source_sha256": (
            short_conv_v5_source["sha256"]
        ),
        "legacy_v4_default_unchanged": True,
        "selective_evidence_required": True,
        "current_plus_next_residency_required": True,
        "physical_rss_gate_required": True,
        "short_conv_v5_numerical_policy_integration_status": (
            "generic_numerical_frontier_adapter_integrated_"
            "real_v5_candidate_generation_pending"
        ),
        "resume_implementation_status": "fail_closed_not_yet_integrated",
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType(
        {**body, "implementation_sha256": canonical_sha256(body)}
    )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RuntimePolicy:
    """Bounded evidence and physical-RSS controls for the opt-in v5 lane."""

    model_fold_policy: legacy_model_fold.Lfm2CpcfModelFoldPolicy
    maximum_resident_evidence_bytes: int
    full_phase_tree_logical_bytes: int
    minimum_physical_rss_reduction_fraction: float = 0.10
    rss_sample_interval_seconds: float = _RSS_SAMPLE_INTERVAL_SECONDS
    require_short_conv_v5_endpoint_first_selector: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_fold_policy,
            legacy_model_fold.Lfm2CpcfModelFoldPolicy,
        ):
            raise TypeError(
                "model_fold_policy must be Lfm2CpcfModelFoldPolicy."
            )
        _positive_int(
            self.maximum_resident_evidence_bytes,
            "maximum_resident_evidence_bytes",
        )
        _positive_int(
            self.full_phase_tree_logical_bytes,
            "full_phase_tree_logical_bytes",
        )
        object.__setattr__(
            self,
            "minimum_physical_rss_reduction_fraction",
            _fraction(
                self.minimum_physical_rss_reduction_fraction,
                "minimum_physical_rss_reduction_fraction",
            ),
        )
        interval = float(self.rss_sample_interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0 or interval > 1.0:
            raise Lfm2CpcfV5ModelCompilerError(
                "rss_sample_interval_seconds must be within (0, 1]."
            )
        object.__setattr__(self, "rss_sample_interval_seconds", interval)
        if not isinstance(
            self.require_short_conv_v5_endpoint_first_selector,
            bool,
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "require_short_conv_v5_endpoint_first_selector must be "
                "boolean."
            )

    @property
    def maximum_physical_rss_increase_bytes(self) -> int:
        retained = 1.0 - self.minimum_physical_rss_reduction_fraction
        return int(math.floor(self.full_phase_tree_logical_bytes * retained))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_fold_policy": self.model_fold_policy.as_dict(),
            "maximum_resident_evidence_bytes": (
                self.maximum_resident_evidence_bytes
            ),
            "full_phase_tree_logical_bytes": (
                self.full_phase_tree_logical_bytes
            ),
            "minimum_physical_rss_reduction_fraction": (
                self.minimum_physical_rss_reduction_fraction
            ),
            "maximum_physical_rss_increase_bytes": (
                self.maximum_physical_rss_increase_bytes
            ),
            "rss_sample_interval_seconds": (
                self.rss_sample_interval_seconds
            ),
            "require_short_conv_v5_endpoint_first_selector": (
                self.require_short_conv_v5_endpoint_first_selector
            ),
        }


def _windows_process_memory() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "GetProcessMemoryInfo failed."
        )
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass
    if os.name == "posix" and os.path.exists("/proc/self/statm"):
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            raise Lfm2CpcfV5ModelCompilerError(
                "/proc/self/statm lacks an RSS field."
            )
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    if os.name == "nt":
        return _windows_process_memory()[0]
    raise Lfm2CpcfV5ModelCompilerError(
        "Current process RSS is unavailable; v5 fails closed."
    )


def _process_peak_rss_bytes() -> int:
    if os.name == "nt":
        return _windows_process_memory()[1]
    if os.name == "posix" and os.path.exists("/proc/self/status"):
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    if os.name == "posix":
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    raise Lfm2CpcfV5ModelCompilerError(
        "Peak process RSS is unavailable; v5 fails closed."
    )


class _PhysicalRssGuard:
    """Continuously retain physical RSS peaks and enforce a hard delta cap."""

    def __init__(
        self,
        *,
        maximum_increase_bytes: int,
        full_phase_tree_logical_bytes: int,
        minimum_reduction_fraction: float,
        sample_interval_seconds: float,
    ) -> None:
        if (
            isinstance(maximum_increase_bytes, bool)
            or not isinstance(maximum_increase_bytes, int)
            or maximum_increase_bytes < 0
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "maximum_increase_bytes must be a nonnegative integer."
            )
        self.maximum_increase_bytes = maximum_increase_bytes
        self.full_phase_tree_logical_bytes = _positive_int(
            full_phase_tree_logical_bytes,
            "full_phase_tree_logical_bytes",
        )
        self.minimum_reduction_fraction = _fraction(
            minimum_reduction_fraction,
            "minimum_reduction_fraction",
        )
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.start_rss_bytes = _process_rss_bytes()
        self.start_lifetime_peak_bytes = _process_peak_rss_bytes()
        self.current_rss_bytes = self.start_rss_bytes
        self.peak_rss_bytes = self.start_rss_bytes
        self.checkpoints: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._violation: str | None = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="lfm2-cpcf-v5-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        self.checkpoint("rss_guard_start")

    @property
    def peak_increase_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)

    @property
    def current_increase_bytes(self) -> int:
        return max(0, self.current_rss_bytes - self.start_rss_bytes)

    def _sample(self) -> None:
        current = _process_rss_bytes()
        lifetime_peak = _process_peak_rss_bytes()
        new_lifetime_peak = (
            lifetime_peak
            if lifetime_peak > self.start_lifetime_peak_bytes
            else 0
        )
        with self._lock:
            self.current_rss_bytes = current
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                current,
                new_lifetime_peak,
            )
            increase = max(
                0,
                self.peak_rss_bytes - self.start_rss_bytes,
            )
            if (
                increase > self.maximum_increase_bytes
                and self._violation is None
            ):
                self._violation = (
                    f"physical RSS increase {increase} exceeded "
                    f"{self.maximum_increase_bytes}"
                )

    def _sample_loop(self) -> None:
        try:
            while not self._stop.wait(self.sample_interval_seconds):
                self._sample()
        except BaseException as exc:  # pragma: no cover - platform failure
            with self._lock:
                self._error = exc

    def _raise_failure(self, label: str) -> None:
        with self._lock:
            error = self._error
            violation = self._violation
        if error is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Physical RSS sampler failed at {label!r}."
            ) from error
        if violation is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                f"{violation} at checkpoint {label!r}."
            )

    def checkpoint(self, label: str) -> None:
        if self._stopped:
            raise Lfm2CpcfV5ModelCompilerError(
                "Cannot checkpoint a stopped physical RSS guard."
            )
        self._sample()
        self._raise_failure(label)
        self.checkpoints.append(
            {
                "label": str(label),
                "current_rss_bytes": self.current_rss_bytes,
                "peak_rss_bytes": self.peak_rss_bytes,
                "current_increase_bytes": self.current_increase_bytes,
                "peak_increase_bytes": self.peak_increase_bytes,
            }
        )

    def close(self) -> Mapping[str, Any]:
        if self._stopped:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS guard can be closed exactly once."
            )
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 4))
        if self._thread.is_alive():
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS sampler did not stop."
            )
        self._sample()
        self._stopped = True
        self._raise_failure("rss_guard_close")
        passed = (
            self.peak_increase_bytes <= self.maximum_increase_bytes
            and self.current_increase_bytes <= self.maximum_increase_bytes
        )
        body = {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
            "start_rss_bytes": self.start_rss_bytes,
            "current_rss_bytes": self.current_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "start_lifetime_peak_bytes": self.start_lifetime_peak_bytes,
            "current_increase_bytes": self.current_increase_bytes,
            "peak_increase_bytes": self.peak_increase_bytes,
            "full_phase_tree_logical_bytes": (
                self.full_phase_tree_logical_bytes
            ),
            "minimum_reduction_fraction": (
                self.minimum_reduction_fraction
            ),
            "maximum_increase_bytes": self.maximum_increase_bytes,
            "sample_interval_seconds": self.sample_interval_seconds,
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": list(self.checkpoints),
            "sampler_stopped": True,
            "physical_rss_gate_passed": passed,
            "comparison_basis": (
                "peak_process_rss_increase_from_precompiler_baseline_"
                "versus_full_phase_tree_logical_bytes"
            ),
        }
        if not passed:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS did not prove material reduction from the full "
                "phase-tree baseline."
            )
        return MappingProxyType(
            {**body, "report_sha256": canonical_sha256(body)}
        )

    def abort(self) -> None:
        if self._stopped:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 4))
        self._stopped = True


def _stub_layers(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    phase_index: int,
    *,
    active_layer_index: int | None = None,
    active_evidence: Lfm2CpcfInstallableLayerEvidence | None = None,
) -> tuple[Lfm2CpcfInstallableLayerEvidence, ...]:
    reference = catalog.phase_references[phase_index]
    result = []
    for layer_index, operator_kind in enumerate(
        reference.layer_operator_kinds
    ):
        if layer_index == active_layer_index:
            if (
                not isinstance(
                    active_evidence,
                    Lfm2CpcfInstallableLayerEvidence,
                )
                or active_evidence.layer_index != layer_index
                or active_evidence.operator_kind != operator_kind
                or active_evidence.source_width != catalog.source_width
                or active_evidence.evidence_sha256
                != reference.layer_evidence_sha256[layer_index]
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "Active layer evidence differs from its phase reference."
                )
            result.append(active_evidence)
        else:
            result.append(
                Lfm2CpcfInstallableLayerEvidence(
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    source_width=catalog.source_width,
                    tensors=MappingProxyType({}),
                    evidence_sha256=reference.layer_evidence_sha256[
                        layer_index
                    ],
                )
            )
    return tuple(result)


def _phase_shell(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    phase_index: int,
    *,
    active_layer_index: int | None = None,
    active_evidence: Lfm2CpcfInstallableLayerEvidence | None = None,
    final_norm_tensors: Mapping[str, Any] | None = None,
) -> Lfm2CpcfInstallablePhaseEvidence:
    reference = catalog.phase_references[phase_index]
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=catalog.cache_key,
        batch_index=reference.batch_index,
        input_sha256=reference.input_sha256,
        phase=reference.phase,
        layers=_stub_layers(
            catalog,
            phase_index,
            active_layer_index=active_layer_index,
            active_evidence=active_evidence,
        ),
        final_norm_tensors=(
            MappingProxyType({})
            if final_norm_tensors is None
            else MappingProxyType(dict(final_norm_tensors))
        ),
        evidence_sha256=reference.existing_phase_evidence_sha256,
    )


def _target_prefix(
    record: Any,
    *,
    phase_name: str,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    prompt_ids = legacy_model_fold._torch().as_tensor(
        record.target_input_ids
    ).detach().to(device="cpu", dtype=torch.long).contiguous()
    prompt_mask = legacy_model_fold._torch().as_tensor(
        record.attention_mask
    ).detach().to(device="cpu", dtype=torch.long).contiguous()
    forced = legacy_model_fold._torch().as_tensor(
        record.target_forced_decode_ids
    ).detach().to(device="cpu", dtype=torch.long).contiguous()
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise Lfm2CpcfV5ModelCompilerError(
            "Target tokenizer record IDs and attention mask are invalid."
        )
    raw_positions = getattr(record, "position_ids", None)
    if raw_positions is None:
        prompt_positions = (
            torch.arange(int(prompt_ids.shape[1]), dtype=torch.long)
            .unsqueeze(0)
            .expand(int(prompt_ids.shape[0]), -1)
            .contiguous()
        )
    else:
        prompt_positions = torch.as_tensor(raw_positions).detach().to(
            device="cpu",
            dtype=torch.long,
        ).contiguous()
    if phase_name in {"prefill", "cached_prefill"}:
        return prompt_ids, prompt_mask, prompt_positions
    match = _DECODE_PHASE.fullmatch(phase_name)
    if match is None:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Unsupported selective phase {phase_name!r}."
        )
    count = int(match.group(1)) + 1
    if forced.ndim != 2 or count > int(forced.shape[1]):
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective decode phase exceeds target forced-decode IDs."
        )
    suffix_ids = forced[:, :count]
    ids = torch.cat((prompt_ids, suffix_ids), dim=1).contiguous()
    mask = torch.cat(
        (
            prompt_mask,
            torch.ones_like(suffix_ids, dtype=torch.long),
        ),
        dim=1,
    ).contiguous()
    start = prompt_positions[:, -1:] + 1
    offsets = torch.arange(count, dtype=torch.long).unsqueeze(0)
    positions = torch.cat(
        (prompt_positions, start + offsets),
        dim=1,
    ).contiguous()
    return ids, mask, positions


def build_lfm2_cpcf_v5_target_phase_shells(
    trace_adapter: Any,
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    bootstrap_payload: Lfm2CpcfSelectiveLayerPayload,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Build canonical target phases with no resident parent tensor tree."""

    verify = getattr(trace_adapter, "verify", None)
    records = getattr(trace_adapter, "records", None)
    if not callable(verify) or not isinstance(records, tuple):
        raise Lfm2CpcfV5ModelCompilerError(
            "A verified strict tokenizer trace adapter is required."
        )
    verify()
    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise TypeError(
            "catalog must be Lfm2CpcfSelectiveEvidenceCatalog."
        )
    if not isinstance(
        bootstrap_payload,
        Lfm2CpcfSelectiveLayerPayload,
    ):
        raise TypeError(
            "bootstrap_payload must be Lfm2CpcfSelectiveLayerPayload."
        )
    if (
        len(bootstrap_payload.phases) != catalog.phase_count
        or bootstrap_payload.layer_index >= catalog.layer_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Bootstrap layer payload differs from the selective catalog."
        )
    by_ordinal = {int(record.ordinal): record for record in records}
    if len(by_ordinal) != len(records):
        raise Lfm2CpcfV5ModelCompilerError(
            "Tokenizer trace record ordinals are not unique."
        )
    result = []
    for phase_index, (reference, phase_payload) in enumerate(
        zip(
            catalog.phase_references,
            bootstrap_payload.phases,
            strict=True,
        )
    ):
        evidence = phase_payload.evidence
        if (
            phase_payload.phase_reference_sha256
            != reference.reference_sha256
            or evidence.layer_index != bootstrap_payload.layer_index
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Bootstrap layer phase order or evidence binding changed."
            )
        record = by_ordinal.get(reference.batch_index)
        if (
            record is None
            or record.donor_cache_input_sha256 != reference.input_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Target tokenizer record is not bound to the parent cache."
            )
        try:
            sampled = int(evidence.tensors["residual.pre"].shape[1])
        except (AttributeError, KeyError, TypeError) as exc:
            raise Lfm2CpcfV5ModelCompilerError(
                "Bootstrap evidence lacks residual.pre sampling geometry."
            ) from exc
        _positive_int(sampled, "sampled_suffix_tokens")
        ids, mask, positions = _target_prefix(
            record,
            phase_name=reference.phase,
        )
        shell = _phase_shell(catalog, phase_index)
        body = {
            "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
            "batch_index": reference.batch_index,
            "record_id": str(record.record_id),
            "split": str(record.split),
            "phase": reference.phase,
            "donor_input_sha256": reference.input_sha256,
            "target_alignment_sha256": str(record.alignment_sha256),
            "input_ids_sha256": model_compiler._tensor_sha256(ids),
            "attention_mask_sha256": model_compiler._tensor_sha256(mask),
            "position_ids_sha256": model_compiler._tensor_sha256(positions),
            "batch_size": int(ids.shape[0]),
            "full_prefix_tokens": int(ids.shape[1]),
            "sampled_suffix_tokens": sampled,
            "parent_evidence_sha256": shell.evidence_sha256,
            "numeric_token_id_equality_required": False,
            "actual_child_full_prefix_required": True,
        }
        result.append(
            Lfm2CpcfTargetPhase(
                batch_index=reference.batch_index,
                record_id=str(record.record_id),
                split=str(record.split),
                phase=reference.phase,
                donor_input_sha256=reference.input_sha256,
                target_alignment_sha256=str(record.alignment_sha256),
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                sampled_suffix_tokens=sampled,
                parent_evidence=shell,
                phase_sha256=model_compiler._sha(body),
            )
        )
    phases = tuple(result)
    if (
        tuple(phase.identity for phase in phases)
        != tuple(
            (reference.batch_index, reference.phase)
            for reference in catalog.phase_references
        )
        or any(
            any(layer.tensors for layer in phase.parent_evidence.layers)
            or bool(phase.parent_evidence.final_norm_tensors)
            for phase in phases
        )
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Tensor-free target phase shells are incomplete or retain tensors."
        )
    verify()
    return phases


def _copy_phase_with_evidence(
    phase: Lfm2CpcfTargetPhase,
    evidence: Lfm2CpcfInstallablePhaseEvidence,
) -> Lfm2CpcfTargetPhase:
    result = Lfm2CpcfTargetPhase(
        batch_index=phase.batch_index,
        record_id=phase.record_id,
        split=phase.split,
        phase=phase.phase,
        donor_input_sha256=phase.donor_input_sha256,
        target_alignment_sha256=phase.target_alignment_sha256,
        input_ids=phase.input_ids,
        attention_mask=phase.attention_mask,
        position_ids=phase.position_ids,
        sampled_suffix_tokens=phase.sampled_suffix_tokens,
        parent_evidence=evidence,
        phase_sha256=phase.phase_sha256,
    )
    if result.as_dict() != phase.as_dict():
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective phase materialization changed target phase metadata."
        )
    return result


def materialize_lfm2_cpcf_v5_layer_phases(
    phases: Sequence[Lfm2CpcfTargetPhase],
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    payload: Lfm2CpcfSelectiveLayerPayload,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Bind exactly one resident source layer to tensor-free target phases."""

    ordered = tuple(phases)
    if (
        not ordered
        or not isinstance(payload, Lfm2CpcfSelectiveLayerPayload)
        or len(ordered) != catalog.phase_count
        or len(payload.phases) != catalog.phase_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Layer phase materialization inputs are incomplete."
        )
    result = []
    for phase_index, (phase, phase_payload, reference) in enumerate(
        zip(
            ordered,
            payload.phases,
            catalog.phase_references,
            strict=True,
        )
    ):
        if (
            phase.identity != (reference.batch_index, reference.phase)
            or phase_payload.phase_reference_sha256
            != reference.reference_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Layer payload phase order differs from target phases."
            )
        evidence = _phase_shell(
            catalog,
            phase_index,
            active_layer_index=payload.layer_index,
            active_evidence=phase_payload.evidence,
        )
        result.append(_copy_phase_with_evidence(phase, evidence))
    materialized = tuple(result)
    for phase in materialized:
        tensor_layers = [
            layer.layer_index
            for layer in phase.parent_evidence.layers
            if bool(layer.tensors)
        ]
        if tensor_layers != [payload.layer_index]:
            raise Lfm2CpcfV5ModelCompilerError(
                "Materialized target phase owns more than its active layer."
            )
    return materialized


def materialize_lfm2_cpcf_v5_final_norm_phases(
    phases: Sequence[Lfm2CpcfTargetPhase],
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    payload: Lfm2CpcfSelectiveFinalNormPayload,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Bind only final-normalization evidence to target phase shells."""

    ordered = tuple(phases)
    if (
        not ordered
        or not isinstance(payload, Lfm2CpcfSelectiveFinalNormPayload)
        or len(ordered) != catalog.phase_count
        or len(payload.phases) != catalog.phase_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization phase materialization is incomplete."
        )
    result = []
    for phase_index, (phase, phase_payload, reference) in enumerate(
        zip(
            ordered,
            payload.phases,
            catalog.phase_references,
            strict=True,
        )
    ):
        if (
            phase.identity != (reference.batch_index, reference.phase)
            or phase_payload.phase_reference_sha256
            != reference.reference_sha256
            or phase_payload.evidence_sha256
            != reference.final_norm_evidence_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization payload phase binding changed."
            )
        evidence = _phase_shell(
            catalog,
            phase_index,
            final_norm_tensors=phase_payload.tensors,
        )
        result.append(_copy_phase_with_evidence(phase, evidence))
    materialized = tuple(result)
    if any(
        any(layer.tensors for layer in phase.parent_evidence.layers)
        or not phase.parent_evidence.final_norm_tensors
        for phase in materialized
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization phases retained layer evidence."
        )
    return materialized


class _SelectiveBoundaryCompiler:
    """Fit the exact v2 boundary bundle without an all-layer tensor tree."""

    def __init__(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        catalog: Lfm2CpcfSelectiveEvidenceCatalog,
        source_tied_table: Any,
        *,
        target_hidden_width: int,
        model_policy: legacy_model_fold.Lfm2CpcfModelFoldPolicy,
        preserve_vocabulary_mean: bool = True,
        metric_chunk_rows: int = 512,
    ) -> None:
        self.phases = tuple(phases)
        self.catalog = catalog
        self.table = boundary._matrix(
            source_tied_table,
            "source_tied_table",
        )
        self.source_width = int(self.table.shape[1])
        self.target_width = boundary._positive_int(
            target_hidden_width,
            "target_hidden_width",
        )
        if (
            self.target_width >= self.source_width
            or self.source_width != catalog.source_width
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary widths differ from source/catalog geometry."
            )
        if not isinstance(preserve_vocabulary_mean, bool):
            raise TypeError("preserve_vocabulary_mean must be boolean.")
        self.preserve_vocabulary_mean = preserve_vocabulary_mean
        self.metric_chunk_rows = boundary._positive_int(
            metric_chunk_rows,
            "metric_chunk_rows",
        )
        self.model_policy = model_policy
        self.layer_kinds = tuple(
            catalog.phase_references[0].layer_operator_kinds
        )
        if (
            len(self.layer_kinds) != catalog.layer_count
            or any(
                tuple(reference.layer_operator_kinds) != self.layer_kinds
                for reference in catalog.phase_references
            )
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective catalog layer topology changes across phases."
            )
        self.split_phases = MappingProxyType(
            {
                split: tuple(
                    phase for phase in self.phases if phase.split == split
                )
                for split in _SPLITS
            }
        )
        if not self.split_phases["fit"] or not self.split_phases["validation"]:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary fit needs fit and validation phases."
            )
        self.split_phase_sha256 = MappingProxyType(
            {
                split: boundary._sha256(
                    [
                        {
                            "ordinal": ordinal,
                            "batch_index": phase.batch_index,
                            "record_id": phase.record_id,
                            "phase": phase.phase,
                            "phase_sha256": phase.phase_sha256,
                            "parent_evidence_sha256": (
                                phase.parent_evidence.evidence_sha256
                            ),
                        }
                        for ordinal, phase in enumerate(
                            self.split_phases[split]
                        )
                    ]
                )
                for split in _SPLITS
            }
        )
        self.split_row_sha256 = MappingProxyType(
            {
                split: boundary._sha256(
                    [
                        row
                        for phase in self.split_phases[split]
                        for row in boundary._row_keys(phase)
                    ]
                )
                for split in _SPLITS
            }
        )
        self.split_rows = MappingProxyType(
            {
                split: sum(
                    phase.batch_size * phase.sampled_suffix_tokens
                    for phase in self.split_phases[split]
                )
                for split in _SPLITS
            }
        )
        if (
            self.split_phase_sha256["fit"]
            == self.split_phase_sha256["validation"]
            or self.split_row_sha256["fit"]
            == self.split_row_sha256["validation"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Fit and validation phase/row identities are not disjoint."
            )
        self.fit_weights: Any | None = None
        self.validation_weights: Any | None = None
        self.fit_weight_summary: list[dict[str, Any]] | None = None
        self.validation_weight_summary: list[dict[str, Any]] | None = None
        self.base_interface: Any | None = None
        self.anchor_fold: Any | None = None
        self.interface: Any | None = None
        self.anchor_split_reports: dict[str, Any] | None = None
        self.fit_future: Any | None = None
        self.validation_future: Any | None = None
        self.layer_results: list[Any] = []
        self._observation_sha: dict[tuple[int, str], str] = {}

    def _minimal_inventory(
        self,
        tensors: Mapping[tuple[tuple[int, str], str], Any],
    ) -> boundary._PhaseInventory:
        return boundary._PhaseInventory(
            phases=self.phases,
            layer_kinds=self.layer_kinds,
            source_width=self.source_width,
            tensors=MappingProxyType(dict(tensors)),
            split_phases=self.split_phases,
            split_phase_sha256=self.split_phase_sha256,
            split_row_sha256=self.split_row_sha256,
            split_rows=self.split_rows,
            source_observation_sha256="0" * 64,
        )

    def fit_interface(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        *,
        target_norm_epsilon: float,
    ) -> Any:
        if self.interface is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective interface can be fitted exactly once."
            )
        ordered = tuple(phases)
        if tuple(phase.phase_sha256 for phase in ordered) != tuple(
            phase.phase_sha256 for phase in self.phases
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization phase lineage changed."
            )
        final_field = "output.final_norm_output"
        tensors = {}
        for phase_index, phase in enumerate(ordered):
            try:
                value = phase.parent_evidence.final_norm_tensors[
                    final_field
                ]
            except KeyError as exc:
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Phase {phase.identity!r} lacks final norm evidence."
                ) from exc
            tensor = boundary._phase_tensor(
                value,
                f"{phase.identity}.{final_field}",
                phase=phase,
                source_width=self.source_width,
            )
            tensors[(phase.identity, final_field)] = tensor
            self._observation_sha[(phase_index, final_field)] = (
                boundary._tensor_sha256(tensor)
            )
        inventory = self._minimal_inventory(tensors)
        (
            self.fit_weights,
            self.fit_weight_summary,
        ) = boundary._balanced_phase_weights(inventory, "fit")
        (
            self.validation_weights,
            self.validation_weight_summary,
        ) = boundary._balanced_phase_weights(inventory, "validation")
        fit_final = boundary._rows(inventory, "fit", final_field)
        validation_final = boundary._rows(
            inventory,
            "validation",
            final_field,
        )
        self.base_interface = fit_tied_causal_interface(
            self.table,
            fit_final,
            target_width=self.target_width,
            validation_producer_rows=validation_final,
            fit_weights=self.fit_weights,
            validation_weights=self.validation_weights,
            preserve_vocabulary_mean=self.preserve_vocabulary_mean,
            target_norm_epsilon=target_norm_epsilon,
            minimum_validation_top1_agreement=0.0,
            metric_chunk_rows=self.metric_chunk_rows,
        )
        anchor_splits = boundary._causal_anchor_splits(inventory)
        self.anchor_fold = fit_tied_causal_interface_anchor_residual(
            self.base_interface,
            self.table,
            selection_split=anchor_splits["selection"],
            fit_split=anchor_splits["fit"],
            validation_split=anchor_splits["validation"],
            target_norm_epsilon=target_norm_epsilon,
            metric_chunk_rows=self.metric_chunk_rows,
        )
        self.interface = self.anchor_fold.interface
        self.fit_future = self.interface.encode_producer_rows(fit_final)
        self.validation_future = self.interface.encode_producer_rows(
            validation_final
        )
        self.anchor_split_reports = {
            split: anchor_splits[split].as_dict() for split in _SPLITS
        }
        del inventory, tensors, fit_final, validation_final, anchor_splits
        gc.collect()
        return self.interface

    def fit_layer(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        *,
        layer_index: int,
    ) -> Any:
        index = boundary._positive_int(layer_index + 1, "layer_index") - 1
        if (
            self.interface is None
            or self.fit_weights is None
            or self.validation_weights is None
            or index != len(self.layer_results)
            or index >= len(self.layer_kinds)
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective layer charts must be fitted in canonical order "
                "after the interface."
            )
        ordered = tuple(phases)
        if tuple(phase.phase_sha256 for phase in ordered) != tuple(
            phase.phase_sha256 for phase in self.phases
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Layer chart phase lineage changed."
            )
        values: dict[str, dict[tuple[int, str], Any]] = {
            name: {} for name in _BOUNDARIES
        }
        for phase_index, phase in enumerate(ordered):
            evidence = phase.parent_evidence.layers[index]
            if (
                evidence.layer_index != index
                or evidence.operator_kind != self.layer_kinds[index]
                or evidence.source_width != self.source_width
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Phase {phase.identity!r} active layer changed."
                )
            for name in _BOUNDARIES:
                try:
                    raw = evidence.tensors[name]
                except KeyError as exc:
                    raise Lfm2CpcfV5ModelCompilerError(
                        f"Phase {phase.identity!r} layer {index} lacks "
                        f"{name!r}."
                    ) from exc
                tensor = boundary._phase_tensor(
                    raw,
                    f"{phase.identity}.layer.{index:03d}.{name}",
                    phase=phase,
                    source_width=self.source_width,
                )
                values[name][phase.identity] = tensor
                field = f"layer.{index:03d}.{name}"
                self._observation_sha[(phase_index, field)] = (
                    boundary._tensor_sha256(tensor)
                )

        def rows(
            split: str,
            name: str,
            source_values: Mapping[
                str,
                Mapping[tuple[int, str], Any],
            ] = values,
        ) -> Any:
            torch = _torch()
            return torch.cat(
                [
                    source_values[name][phase.identity].reshape(
                        -1,
                        self.source_width,
                    )
                    for phase in self.split_phases[split]
                ],
                dim=0,
            ).contiguous()

        charts = {}
        for name in _BOUNDARIES:
            field = f"layer.{index:03d}.{name}"
            charts[name] = fit_cpcf_boundary_chart(
                field,
                rows("fit", name),
                self.fit_future,
                target_width=self.target_width,
                validation_parent_states=rows("validation", name),
                validation_future_coordinates=self.validation_future,
                fit_weights=self.fit_weights,
                validation_weights=self.validation_weights,
                state_reconstruction_weight=(
                    self.model_policy.state_reconstruction_weight
                ),
                ridge=self.model_policy.ridge,
                minimum_predictive_energy_fraction=(
                    self.model_policy.minimum_predictive_energy_fraction
                ),
                minimum_validation_block_retention=(
                    self.model_policy.minimum_validation_block_retention
                ),
                enforce_validation_block_floor=(
                    self.model_policy.enforce_validation_block_floor
                ),
            )
        result = boundary._layer_charts(
            layer_index=index,
            operator_kind=self.layer_kinds[index],
            post_operator=charts["residual.post_operator"],
            post_layer=charts["residual.post_layer"],
        )
        self.layer_results.append(result)
        del values, charts
        gc.collect()
        return result

    def finalize(self) -> boundary.Lfm2CpcfBoundaryBundle:
        if (
            self.interface is None
            or self.base_interface is None
            or self.anchor_fold is None
            or self.anchor_split_reports is None
            or len(self.layer_results) != len(self.layer_kinds)
            or self.fit_weight_summary is None
            or self.validation_weight_summary is None
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary bundle is incomplete."
            )
        final_field = "output.final_norm_output"
        observations = []
        for phase_index, phase in enumerate(self.phases):
            fields = [final_field]
            for layer_index in range(len(self.layer_kinds)):
                fields.extend(
                    f"layer.{layer_index:03d}.{name}"
                    for name in _BOUNDARIES
                )
            for field in fields:
                try:
                    tensor_sha = self._observation_sha[
                        (phase_index, field)
                    ]
                except KeyError as exc:
                    raise Lfm2CpcfV5ModelCompilerError(
                        "Selective source observation inventory is incomplete."
                    ) from exc
                observations.append(
                    {
                        "phase_sha256": phase.phase_sha256,
                        "field": field,
                        "tensor_sha256": tensor_sha,
                    }
                )
        layers = tuple(self.layer_results)
        phase_counts = {
            split: len(self.split_phases[split]) for split in _SPLITS
        }
        row_counts = dict(self.split_rows)
        split_phase_hashes = dict(self.split_phase_sha256)
        split_row_hashes = dict(self.split_row_sha256)
        expected_observations = len(self.phases) * (
            1 + 2 * len(self.layer_kinds)
        )
        predictive_floor = (
            self.model_policy.minimum_predictive_energy_fraction
        )
        validation_floor = (
            self.model_policy.minimum_validation_block_retention
        )
        body = {
            "schema": boundary.LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
            "method": (
                "tied_causal_interface_then_causal_anchor_residual_then_"
                "independent_predictive_boundary_charts"
            ),
            "source_width": self.source_width,
            "target_width": self.target_width,
            "vocabulary_size": int(self.table.shape[0]),
            "layer_count": len(layers),
            "layer_operator_kinds": list(self.layer_kinds),
            "interface_report_sha256": boundary._report_digest(
                self.interface.report,
                "interface",
            ),
            "base_interface_report_sha256": boundary._report_digest(
                self.base_interface.report,
                "base_interface",
            ),
            "causal_anchor_report_sha256": boundary._report_digest(
                self.anchor_fold.report,
                "causal_anchor",
            ),
            "interface_lineage": {
                "base_interface_report": dict(self.base_interface.report),
                "causal_anchor_report": dict(self.anchor_fold.report),
                "base_interface_report_sha256": boundary._report_digest(
                    self.base_interface.report,
                    "base_interface",
                ),
                "causal_anchor_report_sha256": boundary._report_digest(
                    self.anchor_fold.report,
                    "causal_anchor",
                ),
                "active_interface_report_sha256": boundary._report_digest(
                    self.interface.report,
                    "interface",
                ),
                "active_interface_is_anchor_repaired": True,
                "producer_encoder_fixed_by_anchor_repair": True,
                "additional_runtime_parameter_count": 0,
            },
            "causal_anchor_extraction": {
                "method": (
                    "one_last_sampled_final_norm_endpoint_per_record_causal_step"
                ),
                "label_source": "next_phase_newly_appended_target_token",
                "phase_label_contract": {
                    "prefill": "forced_decode[0]",
                    "cached_prefill": "forced_decode[0]",
                    "decode.k": "forced_decode[k+1]",
                },
                "final_unlabeled_phase_excluded": True,
                "canonical_prefill_or_cached_prefill_decode_order_required": True,
                "one_unique_row_id_per_record_batch_row_causal_step": True,
                "row_id_binds_batch_index_batch_row_and_prefix_position": True,
                "no_sampled_prefix_row_duplication": True,
                "split_inventory": dict(self.anchor_split_reports),
            },
            "layer_charts": [layer.as_dict() for layer in layers],
            "split_phase_counts": phase_counts,
            "split_row_counts": row_counts,
            "split_phase_sha256": split_phase_hashes,
            "split_row_sha256": split_row_hashes,
            "fit_validation_disjoint_phase_identities": True,
            "fit_validation_disjoint_row_identities": True,
            "selection": {
                "phase_count": phase_counts["selection"],
                "row_count": row_counts["selection"],
                "phase_sha256": split_phase_hashes["selection"],
                "row_sha256": split_row_hashes["selection"],
                "used_to_fit_base_interface": False,
                "used_to_fit_causal_anchor_values": False,
                "used_to_select_causal_anchor_ridge_margin": True,
                "used_to_fit_boundary_charts": False,
                "used_for_validation_gates": False,
                "authenticated_and_receipted": True,
            },
            "phase_weighting": {
                "method": "equal_phase_mass_then_uniform_rows_within_phase",
                "balanced_phase_weights": True,
                "fit_total_weight": float(self.fit_weights.sum().item()),
                "validation_total_weight": float(
                    self.validation_weights.sum().item()
                ),
                "fit_weights_sha256": boundary._tensor_sha256(
                    self.fit_weights
                ),
                "validation_weights_sha256": boundary._tensor_sha256(
                    self.validation_weights
                ),
                "fit_phases": self.fit_weight_summary,
                "validation_phases": self.validation_weight_summary,
            },
            "interface_top1": {
                "principal_objective": False,
                "capacity_gate_used": False,
                "fit_rmsnorm_parent_top1_agreement": (
                    self.base_interface.report["fit"][
                        "rmsnorm_parent_top1_agreement"
                    ]
                ),
                "validation_rmsnorm_parent_top1_agreement": (
                    self.base_interface.report["validation"][
                        "rmsnorm_parent_top1_agreement"
                    ]
                ),
                "diagnostic_only": True,
            },
            "predictive_heldout_gates": {
                "principal_boundary_objective": True,
                "minimum_predictive_energy_fraction": predictive_floor,
                "minimum_validation_block_retention": validation_floor,
                "validation_block_floor_enforced": (
                    self.model_policy.enforce_validation_block_floor
                ),
                "fit_rows_used_for_parameters": True,
                "validation_rows_used_for_parameters": False,
            },
            "source_coverage": {
                "phase_count": len(self.phases),
                "layer_count": len(self.layer_kinds),
                "boundary_fields_per_layer": list(_BOUNDARIES),
                "final_output_field": final_field,
                "expected_tensor_observations": expected_observations,
                "observed_tensor_observations": len(observations),
                "source_observation_sha256": boundary._sha256(
                    observations
                ),
                "all_phase_final_outputs_covered": True,
                "all_layer_post_operator_boundaries_covered": True,
                "all_layer_post_layer_boundaries_covered": True,
                "full_source_coverage": True,
            },
            "coordinate_contract": {
                "future_coordinates": (
                    "anchor_repaired_tied_interface_encoded_parent_final_norm_"
                    "output"
                ),
                "active_interface_is_anchor_repaired": True,
                "one_global_residual_basis_used": False,
                "independent_post_operator_and_post_layer_charts": True,
                "residual_write_formula": (
                    "encoded_parent_endpoint_minus_actual_child_pre_state"
                ),
            },
            "operation_audit": {
                "parent_forward_count": 0,
                "target_forward_count": 0,
                "parent_generation_count": 0,
                "parent_cache_regeneration_count": 0,
                "model_mutation_count": 0,
                "training_step_count": 0,
                "backward_call_count": 0,
                "optimizer_step_count": 0,
                "quantization_used": False,
                "pruning_used": False,
                "blank_target_values_read": False,
            },
        }
        if len(observations) != expected_observations:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective source observation count is incomplete."
            )
        bundle_sha = boundary._sha256(body)
        return boundary.Lfm2CpcfBoundaryBundle(
            interface=self.interface,
            layers=layers,
            split_phase_sha256=split_phase_hashes,
            split_row_sha256=split_row_hashes,
            report={**body, "bundle_sha256": bundle_sha},
            bundle_sha256=bundle_sha,
        )


def _authenticated_bootstrap_payload(
    provider: Lfm2CpcfSelectiveEvidenceProvider,
    *,
    layer_index: int,
) -> tuple[
    Lfm2CpcfSelectiveLayerPayload,
    DynamicEvidenceLayerAuthentication,
]:
    reference = provider.dynamic_catalog.references[layer_index]
    payload = provider.load_layer_payload(reference)
    try:
        authentication = provider.authenticate_layer_payload(
            reference,
            payload,
        )
    except BaseException:
        provider.release_layer_payload(reference, payload)
        raise
    if (
        not isinstance(
            authentication,
            DynamicEvidenceLayerAuthentication,
        )
        or authentication.layer_index != layer_index
        or authentication.evidence_sha256 != reference.evidence_sha256
        or authentication.logical_bytes != reference.logical_bytes
    ):
        provider.release_layer_payload(reference, payload)
        raise Lfm2CpcfV5ModelCompilerError(
            "Bootstrap evidence authentication differs from the catalog."
        )
    return payload, authentication


def compile_lfm2_cpcf_v5_selective_model_fold(
    source_model: Any,
    target_model: Any,
    *,
    trace_adapter: Any,
    evidence_provider: Lfm2CpcfSelectiveEvidenceProvider,
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    runtime_policy: Lfm2CpcfV5RuntimePolicy,
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None = None,
    candidate_selector: Lfm2CpcfShortConvV5CandidateSelector | None = None,
    numerical_frontier_adapter: (
        Lfm2CpcfShortConvNumericalFrontierAdapter | None
    ) = None,
    resume_state: Any | None = None,
) -> legacy_model_fold.Lfm2CpcfCompiledModelFold:
    """Compile one dense blank using bounded selective evidence.

    V5 resume artifacts are intentionally rejected until the disk publication
    bridge can serialize partial chart state under
    ``LFM2_CPCF_V5_RESUME_SCHEMA``.  Accepting a v4 resume here would silently
    mix compiler fingerprints, so it fails closed.
    """

    if resume_state is not None:
        raise Lfm2CpcfV5ModelCompilerError(
            "V5 selective resume is not yet integrated; refusing a v4 or "
            "untyped resume artifact."
        )
    if not isinstance(runtime_policy, Lfm2CpcfV5RuntimePolicy):
        raise TypeError("runtime_policy must be Lfm2CpcfV5RuntimePolicy.")
    if candidate_selector is not None and not callable(
        getattr(candidate_selector, "select", None)
    ):
        raise TypeError(
            "candidate_selector must implement select(request)."
        )
    if numerical_frontier_adapter is not None and not callable(
        getattr(numerical_frontier_adapter, "generate", None)
    ):
        raise TypeError(
            "numerical_frontier_adapter must implement generate(request)."
        )
    if (
        runtime_policy.require_short_conv_v5_endpoint_first_selector
        and not isinstance(
            candidate_selector,
            Lfm2CpcfV5EndpointFirstCandidateSelector,
        )
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "The required source-bound short-conv v5 endpoint-first selector "
            "needs Lfm2CpcfV5EndpointFirstCandidateSelector at the "
            "pre-install candidate frontier."
        )
    if not isinstance(
        evidence_provider,
        Lfm2CpcfSelectiveEvidenceProvider,
    ):
        raise TypeError(
            "evidence_provider must be Lfm2CpcfSelectiveEvidenceProvider."
        )
    if candidate_replay_evaluator is not None and not callable(
        getattr(candidate_replay_evaluator, "evaluate", None)
    ):
        raise TypeError(
            "candidate_replay_evaluator must implement evaluate()."
        )
    compiler_implementation = dict(
        lfm2_cpcf_v5_compiler_implementation()
    )
    catalog = evidence_provider.catalog
    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective provider has no authenticated LFM2 catalog."
        )
    authenticated_full_phase_tree_bytes = (
        catalog.final_norm_logical_bytes
        + sum(
            reference.logical_bytes
            for reference in catalog.dynamic_catalog.references
        )
    )
    if (
        runtime_policy.full_phase_tree_logical_bytes
        != authenticated_full_phase_tree_bytes
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Configured full phase-tree bytes differ from the authenticated "
            "selective evidence catalog."
        )
    if (
        catalog.final_norm_logical_bytes
        > runtime_policy.maximum_resident_evidence_bytes
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization evidence exceeds the configured resident "
            "evidence budget."
        )
    rss_guard = _PhysicalRssGuard(
        maximum_increase_bytes=(
            runtime_policy.maximum_physical_rss_increase_bytes
        ),
        full_phase_tree_logical_bytes=(
            runtime_policy.full_phase_tree_logical_bytes
        ),
        minimum_reduction_fraction=(
            runtime_policy.minimum_physical_rss_reduction_fraction
        ),
        sample_interval_seconds=runtime_policy.rss_sample_interval_seconds,
    )
    source_before = legacy_model_fold._snapshot_model_auth(source_model)
    target_before = legacy_model_fold._snapshot_model(target_model)
    provider_closed = False
    bootstrap_payload: Lfm2CpcfSelectiveLayerPayload | None = None
    final_payload: Lfm2CpcfSelectiveFinalNormPayload | None = None
    try:
        rss_guard.checkpoint("request_authenticated")
        target_inventory_before = (
            legacy_model_fold.independent_parameter_inventory(target_model)
        )
        target_count = int(
            target_inventory_before["unique_parameters"]
        )
        planned = _positive_int(
            planned_target_unique_parameters,
            "planned_target_unique_parameters",
        )
        if target_count != planned:
            raise Lfm2CpcfV5ModelCompilerError(
                "Live target count differs from the planned dense blank."
            )
        blank = legacy_model_fold._blank_receipt(
            blank_target_report,
            target_unique_parameters=target_count,
        )
        legacy_model_fold._assert_target_is_zero_blank(target_model)
        source_layers = tuple(
            legacy_model_fold._model_core(
                source_model,
                "source_model",
            ).layers
        )
        target_layers = tuple(
            legacy_model_fold._model_core(
                target_model,
                "target_model",
            ).layers
        )
        if (
            not source_layers
            or len(source_layers) != len(target_layers)
            or len(source_layers) != catalog.layer_count
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Source, target, and selective catalog depth differ."
            )
        parameter_receipt = (
            legacy_model_fold.audit_lfm2_cpcf_parameter_compaction(
                source_model,
                target_model,
                planned_target_unique_parameters=planned,
                promotion_compression_ratio=(
                    runtime_policy.model_fold_policy.promotion_compression_ratio
                ),
            )
        )
        if (
            runtime_policy.model_fold_policy.require_promotion_ratio
            and not parameter_receipt["promotion_ratio_met"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Target does not meet the required promotion ratio."
            )
        bootstrap_payload, bootstrap_authentication = (
            _authenticated_bootstrap_payload(
                evidence_provider,
                layer_index=0,
            )
        )
        phases = build_lfm2_cpcf_v5_target_phase_shells(
            trace_adapter,
            catalog,
            bootstrap_payload,
        )
        evidence_provider.release_layer_payload(
            catalog.dynamic_catalog.references[0],
            bootstrap_payload,
        )
        bootstrap_payload = None
        gc.collect()
        rss_guard.checkpoint("tensor_free_phase_shells_built")
        boundary_compiler = _SelectiveBoundaryCompiler(
            phases,
            catalog,
            source_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            model_policy=runtime_policy.model_fold_policy,
        )
        final_payload = evidence_provider.load_final_norm_payload()
        final_norm_authentication_sha256 = (
            evidence_provider.authenticate_final_norm_payload(final_payload)
        )
        final_phases = materialize_lfm2_cpcf_v5_final_norm_phases(
            phases,
            catalog,
            final_payload,
        )
        interface = boundary_compiler.fit_interface(
            final_phases,
            target_norm_epsilon=(
                legacy_model_fold._final_norm_epsilon(target_model)
            ),
        )
        del final_phases
        evidence_provider.release_final_norm_payload(final_payload)
        final_payload = None
        gc.collect()
        rss_guard.checkpoint("interface_fitted_final_norm_released")
        interface_install = install_lfm2_cpcf_tied_interface(
            target_model,
            interface,
        )
        neutral_norm_install = install_lfm2_cpcf_neutral_norms(
            target_model,
            attention_qk_norm_weights=(
                legacy_model_fold._preserved_qk_norm_weights(
                    source_model,
                    target_model,
                )
            ),
        )
        state_bank = seed_lfm2_cpcf_phase_states(
            target_model,
            phases,
        )
        completed_receipts = []
        residency_snapshot: Mapping[str, Any] | None = None
        with DynamicEvidenceResidencyWindow(
            catalog.dynamic_catalog,
            evidence_provider,
            maximum_resident_bytes=(
                runtime_policy.maximum_resident_evidence_bytes
            ),
        ) as residency:
            resident = residency.start(start_layer_index=0)
            rss_guard.checkpoint("residency_started")
            for layer_index in range(catalog.layer_count):
                if (
                    resident.layer_index != layer_index
                    or not isinstance(
                        resident.payload,
                        Lfm2CpcfSelectiveLayerPayload,
                    )
                ):
                    raise Lfm2CpcfV5ModelCompilerError(
                        "Residency window promoted the wrong LFM2 layer."
                    )
                materialized_phases = (
                    materialize_lfm2_cpcf_v5_layer_phases(
                        phases,
                        catalog,
                        resident.payload,
                    )
                )
                layer_bank = rebind_lfm2_cpcf_phase_state_bank(
                    state_bank,
                    materialized_phases,
                )
                charts = boundary_compiler.fit_layer(
                    materialized_phases,
                    layer_index=layer_index,
                )
                result = compile_and_install_lfm2_cpcf_layer(
                    source_model,
                    target_model,
                    layer_index=layer_index,
                    input_bank=layer_bank,
                    boundary_charts=charts,
                    ridge=runtime_policy.model_fold_policy.ridge,
                    short_conv_gate_policy=(
                        runtime_policy.model_fold_policy.resolved_short_conv_policy()
                    ),
                    short_conv_v4_solver_policy=(
                        runtime_policy.model_fold_policy.short_conv_v4_solver_policy
                    ),
                    candidate_replay_evaluator=(
                        candidate_replay_evaluator
                    ),
                    candidate_selector=candidate_selector,
                    numerical_frontier_adapter=(
                        numerical_frontier_adapter
                    ),
                )
                receipt = legacy_model_fold.Lfm2CpcfCompletedLayerReceipt(
                    result.report
                )
                next_bank = rebind_lfm2_cpcf_phase_state_bank(
                    result.next_bank,
                    phases,
                )
                if (
                    receipt.input_bank_sha256
                    != state_bank.state_bank_sha256
                    or receipt.next_bank_sha256
                    != next_bank.state_bank_sha256
                ):
                    raise Lfm2CpcfV5ModelCompilerError(
                        "Layer receipt differs from tensor-free state-bank "
                        "lineage."
                    )
                completed_receipts.append(receipt)
                state_bank = next_bank
                del (
                    result,
                    receipt,
                    layer_bank,
                    materialized_phases,
                    charts,
                )
                gc.collect()
                rss_guard.checkpoint(
                    f"layer_{layer_index:03d}_compiled_before_release"
                )
                following = residency.advance(
                    completed_layer_index=layer_index
                )
                rss_guard.checkpoint(
                    f"layer_{layer_index:03d}_released"
                )
                if layer_index + 1 < catalog.layer_count:
                    if following is None:
                        raise Lfm2CpcfV5ModelCompilerError(
                            "Residency ended before the final layer."
                        )
                    resident = following
                elif following is not None:
                    raise Lfm2CpcfV5ModelCompilerError(
                        "Residency retained evidence after the final layer."
                    )
            residency_snapshot = residency.snapshot()
        if residency_snapshot is None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Residency window emitted no completion receipt."
            )
        selector_applied_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if "preinstall_candidate_selector_receipt"
            in receipt.report["operator_artifact"]
        ]
        numerical_frontier_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if receipt.report["operator_artifact"].get(
                "numerical_frontier_adapter_active"
            )
            is True
        ]
        if (
            runtime_policy.require_short_conv_v5_endpoint_first_selector
            and not selector_applied_layer_indices
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "The required short-conv v5 endpoint-first selector was not "
                "applied to any compiled layer."
            )
        bundle = boundary_compiler.finalize()
        final_payload = evidence_provider.load_final_norm_payload()
        if (
            evidence_provider.authenticate_final_norm_payload(
                final_payload
            )
            != final_norm_authentication_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization evidence changed between interface and "
                "final-norm fitting."
            )
        final_phases = materialize_lfm2_cpcf_v5_final_norm_phases(
            phases,
            catalog,
            final_payload,
        )
        final_norm_report = legacy_model_fold._fit_and_install_final_norm(
            target_model,
            phases=final_phases,
            final_state_bank=rebind_lfm2_cpcf_phase_state_bank(
                state_bank,
                final_phases,
            ),
            boundary_bundle=bundle,
            ridge=runtime_policy.model_fold_policy.ridge,
        )
        del final_phases
        evidence_provider.release_final_norm_payload(final_payload)
        final_payload = None
        gc.collect()
        rss_guard.checkpoint("final_norm_fitted_evidence_released")
        target_inventory_after = (
            legacy_model_fold.independent_parameter_inventory(target_model)
        )
        legacy_model_fold._authenticate_model(source_model, source_before)
        if (
            int(target_inventory_after["unique_parameters"]) != target_count
            or legacy_model_fold._sha(
                legacy_model_fold._model_inventory(
                    target_model,
                    include_tensor_bytes=False,
                )
            )
            != target_before.structure_inventory_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Successful v5 fold changed source or target topology."
            )
        evidence_provider.close()
        provider_closed = True
        gc.collect()
        rss_guard.checkpoint("provider_closed")
        physical_rss_report = rss_guard.close()
        body = {
            "schema": LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
            "artifact_schema": LFM2_CPCF_V5_ARTIFACT_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "resume_supported": False,
            "resume_fail_closed": True,
            "compiler_implementation": compiler_implementation,
            "blank_target_report_sha256": blank["report_sha256"],
            "policy": runtime_policy.as_dict(),
            "phase_count": len(phases),
            "phase_sha256": [phase.phase_sha256 for phase in phases],
            "selective_catalog_sha256": catalog.catalog_sha256,
            "selective_source_sha256": catalog.source_sha256,
            "selective_provider_sha256": catalog.provider_sha256,
            "authenticated_full_phase_tree_logical_bytes": (
                authenticated_full_phase_tree_bytes
            ),
            "physical_rss_baseline_bound_to_authenticated_catalog": True,
            "bootstrap_authentication_sha256": (
                bootstrap_authentication.authentication_sha256
            ),
            "final_norm_authentication_sha256": (
                final_norm_authentication_sha256
            ),
            "residency_receipt": dict(residency_snapshot),
            "physical_rss_receipt": dict(physical_rss_report),
            "boundary_bundle_sha256": bundle.bundle_sha256,
            "legacy_boundary_bundle_schema_preserved": True,
            "interface_install_report_sha256": (
                interface_install["report_sha256"]
            ),
            "neutral_norm_install_report_sha256": (
                neutral_norm_install["report_sha256"]
            ),
            "ordered_layer_result_sha256": [
                receipt.result_sha256 for receipt in completed_receipts
            ],
            "final_state_bank_sha256": state_bank.state_bank_sha256,
            "final_norm_report_sha256": (
                final_norm_report["report_sha256"]
            ),
            "parameter_receipt": dict(parameter_receipt),
            "target_parameter_inventory_before": dict(
                target_inventory_before
            ),
            "target_parameter_inventory_after": dict(
                target_inventory_after
            ),
            "source_state_preserved": True,
            "target_parameter_count_preserved": True,
            "all_layers_compiled_in_order": True,
            "actual_folded_upstream_replay_used": True,
            "full_all_layer_phase_tree_materialized": False,
            "tensor_free_phase_shells_used": True,
            "active_layer_evidence_released_before_successor_compile": True,
            "final_norm_evidence_separately_bounded_and_reauthenticated": True,
            "candidate_replay_backend_injected": (
                candidate_replay_evaluator is not None
            ),
            "preinstall_candidate_selector_provided": (
                candidate_selector is not None
            ),
            "preinstall_candidate_selector_active": bool(
                selector_applied_layer_indices
            ),
            "preinstall_candidate_selector_layer_indices": (
                selector_applied_layer_indices
            ),
            "short_conv_v5_endpoint_first_selector_integrated": (
                bool(selector_applied_layer_indices)
                and isinstance(
                    candidate_selector,
                    Lfm2CpcfV5EndpointFirstCandidateSelector,
                )
            ),
            "short_conv_numerical_frontier_adapter_provided": (
                numerical_frontier_adapter is not None
            ),
            "short_conv_numerical_frontier_adapter_active": bool(
                numerical_frontier_layer_indices
            ),
            "short_conv_numerical_frontier_adapter_layer_indices": (
                numerical_frontier_layer_indices
            ),
            "short_conv_v5_numerical_candidate_generation_integrated": False,
            "short_conv_v5_numerical_policy_integrated": False,
            "short_conv_v5_post_install_rescoring_performed": False,
            "legacy_v4_default_or_fingerprint_changed": False,
            "operation_audit": {
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "generation_calls": 0,
                "parent_cache_generation_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "fallback_paths": 0,
            },
        }
        result_sha = _sha(body)
        return legacy_model_fold.Lfm2CpcfCompiledModelFold(
            boundary_bundle=bundle,
            layer_results=tuple(completed_receipts),
            final_state_bank=state_bank,
            parameter_receipt=parameter_receipt,
            interface_install_report=interface_install,
            neutral_norm_install_report=neutral_norm_install,
            final_norm_report=final_norm_report,
            report={**body, "result_sha256": result_sha},
            result_sha256=result_sha,
        )
    except BaseException as error:
        if bootstrap_payload is not None:
            try:
                evidence_provider.release_layer_payload(
                    catalog.dynamic_catalog.references[
                        bootstrap_payload.layer_index
                    ],
                    bootstrap_payload,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    "Bootstrap evidence release also failed: "
                    f"{cleanup_error}"
                )
        if final_payload is not None:
            try:
                evidence_provider.release_final_norm_payload(final_payload)
            except BaseException as cleanup_error:
                error.add_note(
                    "Final-normalization evidence release also failed: "
                    f"{cleanup_error}"
                )
        if not provider_closed:
            try:
                evidence_provider.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Selective provider close also failed: {cleanup_error}"
                )
        rss_guard.abort()
        try:
            legacy_model_fold._restore_model(target_model, target_before)
            rollback = legacy_model_fold._prove_exact_whole_model_rollback(
                source_model,
                target_model,
                source_snapshot_before=source_before,
                target_snapshot_before=target_before,
            )
        except BaseException as rollback_error:
            raise Lfm2CpcfV5ModelCompilerError(
                "V5 selective compilation failed and exact whole-model "
                "rollback could not be proven."
            ) from rollback_error
        if rollback.get("exact_whole_model_rollback_proven") is not True:
            raise Lfm2CpcfV5ModelCompilerError(
                "V5 selective compilation rollback receipt is incomplete."
            ) from error
        if not isinstance(error, Exception):
            raise
        raise Lfm2CpcfV5ModelCompilerError(
            "V5 selective compilation failed after exact whole-model "
            f"rollback: {error}"
        ) from error


__all__ = [
    "LFM2_CPCF_V5_ARTIFACT_SCHEMA",
    "LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA",
    "LFM2_CPCF_V5_MODEL_FOLD_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA",
    "LFM2_CPCF_V5_RESUME_SCHEMA",
    "Lfm2CpcfV5EndpointFirstCandidateSelector",
    "Lfm2CpcfV5ModelCompilerError",
    "Lfm2CpcfV5RuntimePolicy",
    "build_lfm2_cpcf_v5_target_phase_shells",
    "compile_lfm2_cpcf_v5_selective_model_fold",
    "lfm2_cpcf_v5_compiler_implementation",
    "materialize_lfm2_cpcf_v5_final_norm_phases",
    "materialize_lfm2_cpcf_v5_layer_phases",
]
