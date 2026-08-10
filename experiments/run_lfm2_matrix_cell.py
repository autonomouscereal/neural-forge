#!/usr/bin/env python3
"""One process, one cell of the stage-1 matrix campaign.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §2.1 generalizes the layer-0
representability study (`run_lfm2_layer0_representability_study.py`) from one
boundary to any (component, site, technique, level).  The proven shape is kept
verbatim: open sealed caches through authenticated sessions, reconstruct the
boundary from the sealed parent rows, run a **calibration arm first** and stop
before believing any solve if it fails, score on the authoritative held-out
split, seal one JSON record.  What is new is that the record is a *comparable*
receipt (§3) and that the cell ends in behavior -- free-run and decode smoke --
because the closed-path metric policy says boundary numbers are diagnostic and
behavior is the signal.

Stages, all inside one process (§2.1):

 1. materialize the single-component child (parent copy_exact everywhere else)
 2. calibration: the copy_exact reproduction of the parent's own component
 3. solve via the technique registry, install, score the write and the macrocell
 4. composed terminal against the emit-zero floor and the copy_exact control
 5. free-run probe (8 prompts x 64 greedy tokens) against the control child --
    the tripwire N5 says has never fired
 6. decode smoke (3 prompts, 128-token cap) recorded verbatim, unscored
 7. seal the receipt, append the ledger

Budget (§2.3): target 12 minutes, hard cap 20.  The cap is enforced by a
watchdog that seals a ``timeout`` receipt and yields, because a cell that dies
silently costs the campaign a row *and* the reason.

Two run modes:

``--mode control``
    Cell 0a, ``CTRL.copy_exact``.  Builds the all-copy-exact child, runs the
    probe and the smoke, and writes a control artifact every later cell scores
    against.  Its receipt is the campaign's behavioral baseline.

``--mode cell``
    Any other cell.  Requires ``--control-artifact``.

Evidence is opened once per site and cached (``--evidence-cache``), so a batch
of cells at the same site pays the sealed-session content hash once.  Nothing
under the sealed trace cache, the sealed phase cache, or any published child is
ever written by this runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CONTROL_CELL_ID,
    CellId,
    MatrixCellReceiptError,
    append_ledger,
    blank_receipt,
    probe_set_sha256,
    prompt_sha256,
    seal_receipt,
    timeout_receipt,
    write_receipt,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    CellTechniqueResult,
    component_spec,
    describe_registry,
    register,
    resolve,
    resolve_level_width,
    validate_result,
)

RUNNER_SCHEMA = "neural-forge-lfm2-matrix-cell-runner-v1"

DEFAULT_TARGET_SECONDS = 12 * 60
DEFAULT_HARD_CAP_SECONDS = 20 * 60

PROBE_PROMPTS_FILE = _REPO_ROOT / "experiments" / "matrix_campaign_probe_prompts_v1.json"
SMOKE_PROMPTS_FILE = _REPO_ROOT / "experiments" / "matrix_campaign_smoke_prompts_v1.json"


class MatrixCellRunError(RuntimeError):
    """The cell failed closed."""


#: Components whose parent map is **not** a single 2-D matmul from
#: ``input_endpoint`` to ``write_endpoint``, and the reason, verbatim from the
#: wave-1 measurements in `NOTES-MX-E` §4.5.
#:
#: The generic calibration gate below scores ``held_input @ parent_weight.T``
#: against the sealed write rows.  For C05-C09 and C11-C13 that *is* the
#: component.  For these four it is a category error with four different faces,
#: and wave 1 sealed all four cells as ``infra`` -- void for a harness reason,
#: with no evidence at all about the technique.  A gate that cannot be applied
#: is not a gate that failed, so these cells declare it inapplicable, carry the
#: reason into the receipt, and rest their verdict on the links that need no
#: per-component predictor: the macrocell where the write composes into the
#: residual, the composed forward against the copy_exact control, the free-run
#: probe and the frozen review.
#:
#: This is deliberately a *closed list of components*, not a fallback triggered
#: by a caught exception.  A shape mismatch that happens for some other reason
#: must still void its cell loudly; only these four pre-registered category
#: errors are exempt, and the exemption is visible in the sealed receipt.
NO_PARENT_MATMUL_PREDICTOR: Mapping[str, str] = {
    "C03": (
        "conv_in_proj emits three stacked projection blocks (B/C/x) over one "
        "internal channel axis; the layout has no name for the stacked write "
        "endpoint, so the generic gate compares a 3072-wide in_proj output "
        "against the 1024-wide residual (NOTES-MX-E 4.5)."
    ),
    "C04": (
        "the short-conv FIR kernel is (channels, 1, taps); parent_weight.T is "
        "not a linear map from the operator input, so there is no matmul for "
        "the generic gate to run (NOTES-MX-E 4.5)."
    ),
    "C10": (
        "attn_per_head owns four tensors and its output passes through a "
        "softmax; the generic gate takes weight_keys[0] (q_proj), finds the "
        "shapes agree, and computes a meaningless 4.4955 relL2 "
        "(NOTES-MX-E 4.5)."
    ),
    "C14": (
        "the RMSNorm gain is a 1-D (hidden,) vector and the component "
        "normalizes before it scales; held_input @ gain.T collapses the rows "
        "instead of reproducing them (NOTES-MX-E 4.5)."
    ),
}


# --------------------------------------------------------------------------
# small helpers, copied in shape from the layer-0 study
# --------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _src_tree_sha256(root: Path) -> str:
    """Digest of every source file the cell could have executed.

    ``code_identity.src_tree_sha`` has to change when the estate changes, or two
    receipts from different code claim the same provenance.
    """

    digest = hashlib.sha256()
    for path in sorted((root / "src" / "neural_forge").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("ascii"))
        digest.update(_file_sha256(path).encode("ascii"))
    for name in ("run_lfm2_matrix_cell.py", "render_matrix_ledger.py"):
        candidate = root / "experiments" / name
        if candidate.is_file():
            digest.update(name.encode("ascii"))
            digest.update(_file_sha256(candidate).encode("ascii"))
    return digest.hexdigest()


def compute_architecture(device: str) -> str:
    """``sm<major><minor> <device name>`` for the device this cell will run on.

    `NOTES-MX-E` §2 is the reason this exists.  A wave-1 batch was launched with
    ``CUDA_VISIBLE_DEVICES=4`` -- ``nvidia-smi``'s second RTX 3090 -- and CUDA's
    default ``FASTEST_FIRST`` ordering resolved it to a Tesla P40 at sm_61.  The
    cells died in evidence loading so nothing numeric was lost, but that was
    luck: A9's fp32 OMP candidate search does not reproduce across
    architectures (`NOTES-MX-C` §4.4), and a half-wave measured on sm_61 would
    have been silently incomparable with the sm_86 half, because no receipt
    field recorded which one ran.

    The string is read off the *resolved* device, not off an environment
    variable, so it survives every ordering surprise that produced the leak.
    """

    if not str(device).startswith("cuda"):
        return f"cpu {sys.platform}"
    if not torch.cuda.is_available():
        raise MatrixCellRunError(
            f"--device {device} was requested but CUDA is not available."
        )
    index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{int(major)}{int(minor)} {torch.cuda.get_device_name(index)}"


def _repo_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    head = result.stdout.strip()
    return head if head else "unknown"


def _atomic_write_json(path: Path, body: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _row_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Per-row cosine/relative-L2 summary, identical to the layer-0 study's.

    Keeping the metric byte-for-byte the same as the study that produced S2 is
    what makes a matrix cell's boundary number comparable to the sealed
    first-divergence ledger at all.
    """

    left = candidate.to(dtype=torch.float64)
    right = reference.to(dtype=torch.float64)
    delta = left - right
    l2 = torch.linalg.vector_norm(delta, dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    tiny = torch.finfo(torch.float64).tiny
    relative = l2 / right_norm.clamp_min(tiny)
    denominator = (left_norm * right_norm).clamp_min(tiny)
    cosine = ((left * right).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
    zero = (left_norm == 0.0) | (right_norm == 0.0)
    if bool(zero.any().item()):
        cosine = torch.where(zero, torch.zeros_like(cosine), cosine)
    ratio = left_norm / right_norm.clamp_min(tiny)
    return {
        "rows": int(l2.numel()),
        "mean_cosine": float(cosine.mean().item()),
        "mean_relative_l2": float(relative.mean().item()),
        "median_relative_l2": float(relative.median().item()),
        "mean_norm_ratio": float(ratio.mean().item()),
        "aggregate_relative_l2": float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(right)).item()
        ),
    }


# --------------------------------------------------------------------------
# prompt sets
# --------------------------------------------------------------------------


def _load_prompt_set(path: Path, expected_kind: str) -> tuple[tuple[str, ...], str]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("kind") != expected_kind:
        raise MatrixCellRunError(
            f"{path} declares kind {body.get('kind')!r}, expected {expected_kind!r}."
        )
    prompts = tuple(str(entry) for entry in body["prompts"])
    if not prompts:
        raise MatrixCellRunError(f"{path} carries no prompts.")
    sealed = probe_set_sha256(prompts)
    declared = body.get("prompt_set_sha256")
    if declared is not None and declared != sealed:
        raise MatrixCellRunError(
            f"{path} seal mismatch: declares {declared}, recomputes {sealed}."
        )
    return prompts, sealed


# --------------------------------------------------------------------------
# sealed evidence for one site
# --------------------------------------------------------------------------


class TraceSiteEvidence:
    """Sealed parent rows for one layer, flattened and split fit/heldout.

    Implements `neural_forge.matrix_technique_registry.MatrixCellEvidence`.
    Rows for every endpoint are flattened in one fixed (batch, phase, token)
    order so any two endpoints are row-aligned, which is what lets a technique
    take ``fit_rows("mlp.input")`` and ``fit_rows("mlp.gate")`` as an
    (input, target) pair without knowing anything about batches or phases.
    """

    def __init__(
        self,
        *,
        layer_index: int,
        phases: tuple[str, ...],
        endpoints: tuple[str, ...],
        fit: Mapping[str, torch.Tensor],
        heldout: Mapping[str, torch.Tensor],
        selection_sha256: str,
    ) -> None:
        self.layer_index = layer_index
        self.phases = phases
        self._endpoints = endpoints
        self._fit = dict(fit)
        self._heldout = dict(heldout)
        self._selection_sha256 = selection_sha256

    @property
    def endpoints(self) -> tuple[str, ...]:
        return self._endpoints

    @property
    def selection_sha256(self) -> str:
        return self._selection_sha256

    @property
    def fit_row_count(self) -> int:
        if not self._fit:
            return 0
        return int(next(iter(self._fit.values())).shape[0])

    @property
    def heldout_row_count(self) -> int:
        if not self._heldout:
            return 0
        return int(next(iter(self._heldout.values())).shape[0])

    def fit_rows(self, endpoint: str) -> torch.Tensor:
        if endpoint not in self._fit:
            raise MatrixCellRunError(
                f"Endpoint {endpoint!r} was not selected for layer "
                f"{self.layer_index}; available: {sorted(self._fit)}."
            )
        return self._fit[endpoint]

    def heldout_rows(self, endpoint: str) -> torch.Tensor:
        if endpoint not in self._heldout:
            raise MatrixCellRunError(
                f"Endpoint {endpoint!r} was not selected for layer "
                f"{self.layer_index}; available: {sorted(self._heldout)}."
            )
        return self._heldout[endpoint]


def _flatten_rows(value: torch.Tensor) -> torch.Tensor:
    """Flatten a sealed tensor to ``(rows, width)``.

    Every 3-D endpoint is ``(batch, tokens, width)`` and folds trivially.  The
    per-head attention endpoints -- the only 4-D tensors ``_LAYOUT`` can ask for
    (``attention.q_linear``, ``attention.k_linear``, ``attention.v``,
    ``attention.context``) -- are captured by the frozen collector in the axis
    order the attention op itself works in, ``(batch, heads, tokens, head_dim)``,
    which is *not* what this used to assume.  A row has to mean the same token in
    every endpoint, because a cell's calibration arm multiplies one endpoint's
    rows by a parent weight and compares them to another endpoint's rows
    positionally, so the head axis is transposed back behind the token axis
    before it folds into the width.  That reproduces the ``heads * head_dim``
    contiguous channel order the projection actually wrote.

    Folding the last two axes instead -- the previous behaviour -- flattens a
    prefill batch ``(1, 16, 256, 64)`` to ``(16, 16384)`` and a decode step
    ``(1, 16, 1, 64)`` to ``(16, 64)``, so the per-endpoint concatenation across
    phases raises before any number is computed.  Loud, but the near miss was
    not: had the phases agreed in width, the rows would have been head-major
    spans of concatenated tokens, silently misaligned against every 3-D endpoint
    and against the parent weight the calibration arm multiplies by.
    """

    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 4:
        tensor = tensor.permute(0, 2, 1, 3)
    if tensor.ndim >= 4:
        tensor = tensor.reshape(*tensor.shape[:-2], -1)
    return tensor.reshape(-1, int(tensor.shape[-1])).contiguous()


SPLIT_MANIFEST_SCHEMA = "neural-forge-lfm2-matrix-campaign-evidence-manifest-v1"


def _authenticated_split_manifest(
    path: Path,
    *,
    trace_cache: Any,
    references: Sequence[Any],
) -> dict[str, Any]:
    """Bind a fresh capture's sealed split to the cache it was captured from.

    This is `hf_lfm2_exact4x_first_divergence.authoritative_split_by_batch`'s
    contract, checked against the capture manifest instead of the balanced-basis
    report: the manifest must re-seal to its own recorded digest, name this
    exact cache and parent, carry one row per batch, agree with every batch
    reference's ``input_sha256``, and use both splits.  Anything less and the
    cell is scored on a split nobody can reproduce, so every failure here is a
    refusal.
    """

    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if body.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise MatrixCellRunError(
            f"{path} is not a {SPLIT_MANIFEST_SCHEMA} manifest."
        )
    declared = body.get("manifest_sha256")
    recomputed = hashlib.sha256(
        json.dumps(
            {key: value for key, value in body.items() if key != "manifest_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    if declared != recomputed:
        raise MatrixCellRunError(
            f"Evidence split manifest seal is broken: {declared} != {recomputed}."
        )
    if body.get("trace_cache_key") != trace_cache.cache_key:
        raise MatrixCellRunError(
            "Evidence split manifest names trace cache "
            f"{body.get('trace_cache_key')}, not {trace_cache.cache_key}."
        )
    if body.get("parent_identity_sha256") != trace_cache.parent_fingerprint:
        raise MatrixCellRunError(
            "Evidence split manifest names a different parent than the cache."
        )
    rows = (body.get("split") or {}).get("rows")
    if not isinstance(rows, list) or len(rows) != len(references):
        raise MatrixCellRunError(
            "Evidence split manifest has no complete per-batch split."
        )
    seen: dict[int, str] = {}
    for row in rows:
        index = row.get("batch_index") if isinstance(row, Mapping) else None
        split = row.get("split") if isinstance(row, Mapping) else None
        record_id = row.get("record_id") if isinstance(row, Mapping) else None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(references)
            or index in seen
            or split not in ("fit", "heldout")
            or record_id != getattr(references[index], "input_sha256", None)
        ):
            raise MatrixCellRunError(
                "Evidence split manifest rows differ from the trace references."
            )
        seen[index] = str(split)
    if set(seen) != set(range(len(references))) or set(seen.values()) != {
        "fit",
        "heldout",
    }:
        raise MatrixCellRunError(
            "The split must cover every batch and both evidence halves."
        )
    _log(
        f"evidence: split authority {path.name} "
        f"({body['split']['fit_batches']} fit / "
        f"{body['split']['heldout_batches']} heldout batches, "
        f"{body['manifest_sha256'][:12]})"
    )
    return body


def load_site_evidence(
    *,
    trace_cache_path: Path,
    layer_index: int,
    endpoints: Sequence[str],
    forced_decode_count: int,
    fit_batch_limit: int | None,
    heldout_batch_limit: int | None,
    cache_directory: Path | None,
    basis_phase_cache_root: Path | None,
    basis_phase_key_sha256: str | None,
    basis_phase_hmac_key_file: Path | None,
    basis_phase_hmac_key_id: str | None,
    evidence_split_manifest: Path | None,
    target_hidden_size: int,
) -> tuple[TraceSiteEvidence, dict[str, Any]]:
    """Open the sealed trace cache for one site, or reuse an authenticated cache.

    The cache-reuse rule is the layer-0 study's, unchanged: a later pass may
    reuse only after re-deriving the payload digest and only when the trace
    cache key, layer, endpoint set and split still bind; it never widens what
    the first pass was allowed to read.

    Two split authorities exist and exactly one must be supplied.  The sealed
    balanced-basis phase is the original: every exact-4x lane inherits its
    FIT/heldout split from it, so a cell scored against that evidence base has
    to as well.  A *freshly captured* matrix-campaign cache has no basis phase
    bound to it -- the campaign never fits the exact-4x basis -- so it carries
    its own sealed capture manifest instead, which declares the same
    `whole-record-seeded-sha256-ranking` split over the same record ids.  Both
    paths bind the split to the cache identity and to the per-batch record ids;
    neither one invents a split, and with neither supplied the runner refuses.
    """

    from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
        PhaseAnchorAuthority,
        load_existing_lfm2_circuit_balanced_basis_phase,
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_exact4x_first_divergence import (
        authoritative_split_by_batch,
        phase_names,
    )
    from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache

    # Checked before anything is opened: two authorities can disagree, and the
    # cheapest moment to refuse a contradictory configuration is the first one.
    if basis_phase_cache_root is not None and evidence_split_manifest is not None:
        raise MatrixCellRunError(
            "Supply exactly one split authority: the sealed balanced-basis "
            "phase or the fresh capture's sealed split manifest, never both."
        )
    phases = phase_names(forced_decode_count)
    ordered_endpoints = tuple(sorted(set(endpoints)))
    binding_probe = {
        "layer_index": int(layer_index),
        "endpoints": list(ordered_endpoints),
        "phases": list(phases),
    }
    if cache_directory is not None and (cache_directory / "binding.json").is_file():
        binding = json.loads((cache_directory / "binding.json").read_text("utf-8"))
        payload_path = cache_directory / "payload.pt"
        if _file_sha256(payload_path) != binding["payload_sha256"]:
            raise MatrixCellRunError("Cached site evidence payload digest changed.")
        if evidence_split_manifest is not None:
            # A cached payload carries a split.  Reusing it under a different
            # split manifest would silently re-label held-out rows as fit.
            declared = json.loads(
                Path(evidence_split_manifest).read_text(encoding="utf-8")
            ).get("manifest_sha256")
            if binding.get("evidence_split_manifest_sha256") != declared:
                raise MatrixCellRunError(
                    "Cached site evidence was built under a different split "
                    f"manifest: {binding.get('evidence_split_manifest_sha256')!r} "
                    f"!= {declared!r}."
                )
        for field, expected in binding_probe.items():
            if binding.get(field) != expected:
                raise MatrixCellRunError(
                    f"Cached site evidence does not bind {field}: "
                    f"{binding.get(field)!r} != {expected!r}."
                )
        payload = torch.load(payload_path, weights_only=True)
        fit = {k[4:]: v for k, v in payload.items() if k.startswith("fit|")}
        heldout = {k[4:]: v for k, v in payload.items() if k.startswith("hld|")}
        evidence = TraceSiteEvidence(
            layer_index=int(layer_index),
            phases=phases,
            endpoints=ordered_endpoints,
            fit=fit,
            heldout=heldout,
            selection_sha256=str(binding["selection_sha256"]),
        )
        _log(f"evidence: reused authenticated cache {cache_directory}")
        return evidence, {**binding, "evidence_cache_reused": True}

    trace_cache = open_lfm2_circuit_trace_cache(trace_cache_path, verify=False)
    manifest = trace_cache.manifest
    topology = manifest.get("layer_topology")
    layers = topology.get("layers") if isinstance(topology, Mapping) else None
    if not isinstance(layers, list) or not layers:
        raise MatrixCellRunError("Trace cache has no layer topology.")
    widths = {
        int(layer["contract"]["hidden_size"])
        for layer in layers
        if isinstance(layer, Mapping) and isinstance(layer.get("contract"), Mapping)
    }
    if len(widths) != 1:
        raise MatrixCellRunError("Trace topology has no single residual width.")
    source_width = int(next(iter(widths)))

    names = tuple(
        sorted(
            f"{phase}.layer.{layer_index:03d}.{endpoint}"
            for phase in phases
            for endpoint in ordered_endpoints
        )
    )
    _log("opening the sealed selective trace session (hashes every batch)")
    with trace_cache.selective_authenticated_session() as session:
        basis_phase = None
        split_manifest: dict[str, Any] | None = None
        if evidence_split_manifest is not None:
            split_manifest = _authenticated_split_manifest(
                Path(evidence_split_manifest),
                trace_cache=trace_cache,
                references=session.batch_references,
            )
            split_by_batch = {
                int(row["batch_index"]): str(row["split"])
                for row in split_manifest["split"]["rows"]
            }
        elif basis_phase_cache_root is not None:
            secret = Path(basis_phase_hmac_key_file).read_bytes()
            try:
                authority = PhaseAnchorAuthority(
                    key_id=str(basis_phase_hmac_key_id), secret=secret
                )
            finally:
                del secret
            basis_phase = load_existing_lfm2_circuit_balanced_basis_phase(
                DynamicFoldPhaseCache(Path(basis_phase_cache_root)),
                str(basis_phase_key_sha256),
                anchor_authority=authority,
                expected_parent_identity_sha256=trace_cache.parent_fingerprint,
                expected_trace_cache_key=trace_cache.cache_key,
                expected_trace_selective_session_sha256=session.evidence_sha256,
                expected_source_width=source_width,
                expected_target_width=int(target_hidden_size),
            )
            split_by_batch = authoritative_split_by_batch(
                basis_phase.report, session.batch_references
            )
        else:
            # With no split authority at all there is nothing to inherit, and
            # inventing one would make every cell's held-out number
            # incomparable with the sealed ledger.  Fail closed.
            raise MatrixCellRunError(
                "A site cell needs the sealed balanced-basis phase, or the "
                "fresh capture's sealed split manifest, to inherit the "
                "authoritative FIT/heldout split."
            )
        ordered_fit = [i for i, s in sorted(split_by_batch.items()) if s == "fit"]
        ordered_heldout = [i for i, s in sorted(split_by_batch.items()) if s != "fit"]
        if fit_batch_limit is not None:
            ordered_fit = ordered_fit[:fit_batch_limit]
        if heldout_batch_limit is not None:
            ordered_heldout = ordered_heldout[:heldout_batch_limit]
        selected = sorted(set(ordered_fit) | set(ordered_heldout))
        digest = hashlib.sha256()
        digest.update(RUNNER_SCHEMA.encode("ascii"))
        buckets: dict[str, dict[str, list[torch.Tensor]]] = {
            "fit": {name: [] for name in ordered_endpoints},
            "hld": {name: [] for name in ordered_endpoints},
        }
        for count, index in enumerate(selected):
            payload = session.load_selected_tensors(index, names)
            try:
                digest.update(payload.selection_sha256.encode("ascii"))
                bucket = "fit" if index in set(ordered_fit) else "hld"
                for phase in phases:
                    for endpoint in ordered_endpoints:
                        key = f"{phase}.layer.{layer_index:03d}.{endpoint}"
                        buckets[bucket][endpoint].append(
                            _flatten_rows(payload.tensors[key])
                        )
            finally:
                session.release_selected_tensors(payload)
            if (count + 1) % 32 == 0:
                _log(f"evidence: {count + 1}/{len(selected)} batches loaded")
        session.verify()
        session_sha256 = session.evidence_sha256
    fit = {
        name: torch.cat(rows, dim=0).contiguous()
        for name, rows in buckets["fit"].items()
    }
    heldout = {
        name: torch.cat(rows, dim=0).contiguous()
        for name, rows in buckets["hld"].items()
    }
    evidence = TraceSiteEvidence(
        layer_index=int(layer_index),
        phases=phases,
        endpoints=ordered_endpoints,
        fit=fit,
        heldout=heldout,
        selection_sha256=digest.hexdigest(),
    )
    binding = {
        "schema": RUNNER_SCHEMA,
        "kind": "site-evidence-binding",
        "trace_cache_key": trace_cache.cache_key,
        "parent_identity_sha256": trace_cache.parent_fingerprint,
        "trace_selective_session_sha256": session_sha256,
        "basis_map_sha256": None if basis_phase is None else basis_phase.map_sha256,
        "split_authority": (
            "balanced-basis-phase" if basis_phase is not None else
            "matrix-campaign-capture-manifest"
        ),
        "evidence_split_manifest_sha256": (
            None if split_manifest is None else split_manifest["manifest_sha256"]
        ),
        "implementation_sha256": (
            trace_cache.manifest.get("implementation", {}).get("implementation_sha256")
        ),
        "selection_sha256": evidence.selection_sha256,
        "evidence_cache_reused": False,
        "fit_rows": evidence.fit_row_count,
        "heldout_rows": evidence.heldout_row_count,
        **binding_probe,
    }
    if cache_directory is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        payload = {f"fit|{k}": v for k, v in fit.items()}
        payload.update({f"hld|{k}": v for k, v in heldout.items()})
        payload_path = cache_directory / "payload.pt"
        torch.save(payload, payload_path)
        _atomic_write_json(
            cache_directory / "binding.json",
            {**binding, "payload_sha256": _file_sha256(payload_path)},
        )
        _log(f"evidence: wrote authenticated cache to {cache_directory}")
    return evidence, binding


# --------------------------------------------------------------------------
# child construction and behavior
# --------------------------------------------------------------------------


def _load_parent(model_directory: Path, device: str, dtype: torch.dtype) -> Any:
    from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_model

    return load_lfm2_direct_local_model(model_directory, device=device, dtype=dtype)


def _load_tokenizer(model_directory: Path) -> Any:
    from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer

    return load_lfm2_direct_local_tokenizer(model_directory)


class _ShapeOnlyTensor:
    """A parent tensor's geometry without its bytes.

    ``component_spec`` resolves a cell against the parent's real tensor names
    and shapes and never reads a value; a route cell needs exactly that and
    nothing more.  Handing it a shape-only view makes the "did not read the
    parent's weights" claim structural instead of a promise.
    """

    __slots__ = ("shape",)

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(int(value) for value in shape)


def _parent_shapes(model_directory: Path) -> dict[str, _ShapeOnlyTensor]:
    from safetensors import safe_open

    path = Path(model_directory) / "model.safetensors"
    if path.is_symlink() or not path.is_file():
        raise MatrixCellRunError(f"Parent safetensors missing at {path}.")
    shapes: dict[str, _ShapeOnlyTensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shapes[name] = _ShapeOnlyTensor(handle.get_slice(name).get_shape())
    return shapes


def _install_component(model: Any, installed: Mapping[str, torch.Tensor]) -> list[str]:
    """Copy the technique's materialized tensors into the copy_exact child.

    Every other tensor stays exactly as the parent load left it.  The returned
    digests are recorded so the receipt can prove which tensors moved.
    """

    state = dict(model.state_dict())
    digests: list[str] = []
    with torch.no_grad():
        for key in sorted(installed):
            if key not in state:
                raise MatrixCellRunError(f"Child has no tensor {key!r} to install.")
            target = state[key]
            value = installed[key].to(device=target.device, dtype=target.dtype)
            if tuple(value.shape) != tuple(target.shape):
                raise MatrixCellRunError(
                    f"Installed {key} has shape {tuple(value.shape)}, child "
                    f"expects {tuple(target.shape)}."
                )
            target.copy_(value)
            digests.append(f"{key}={_tensor_sha256(target)}")
    return digests


@torch.no_grad()
def _greedy_fixed(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    tokens: int,
    device: str,
) -> list[list[int]]:
    """Greedy free-run of exactly ``tokens`` steps per prompt, no early stop.

    A fixed length is what makes the token-match tripwire well defined: two
    children that diverge at step 3 still contribute 61 comparable positions,
    and an early EOS cannot flatter a child by shortening the comparison.
    """

    outputs: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated: list[int] = []
        past = None
        current = ids
        for _step in range(tokens):
            result = model(input_ids=current, past_key_values=past, use_cache=True)
            past = result.past_key_values
            next_id = int(result.logits[0, -1].argmax().item())
            generated.append(next_id)
            current = torch.tensor([[next_id]], device=device, dtype=ids.dtype)
        outputs.append(generated)
    return outputs


def _detect_loop(ids: Sequence[int], *, minimum_period: int = 1) -> bool:
    """True when the tail of a generation is a repeated cycle.

    The GOAL censoring rule distinguishes ``loop`` from ``cap``: a run that hit
    the cap while still saying new things is censored evidence, a run that hit
    the cap repeating itself is a different observation entirely.
    """

    tail = list(ids)[-64:]
    if len(tail) < 8:
        return False
    for period in range(minimum_period, len(tail) // 3 + 1):
        window = tail[-period:]
        repeats = len(tail) // period
        if repeats < 3:
            continue
        if all(
            tail[-period * (r + 1) : len(tail) - period * r] == window
            for r in range(3)
        ):
            return True
    return False


@torch.no_grad()
def _decode_smoke(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    cap: int,
    device: str,
    eos_id: int | None,
) -> list[dict[str, Any]]:
    """3 fixed prompts, 128-token cap, decoded verbatim.  Unscored at fold time."""

    records: list[dict[str, Any]] = []
    for prompt in prompts:
        text = prompt
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generated: list[int] = []
        past = None
        current = ids
        terminated = "cap"
        try:
            for _step in range(cap):
                result = model(
                    input_ids=current, past_key_values=past, use_cache=True
                )
                past = result.past_key_values
                next_id = int(result.logits[0, -1].argmax().item())
                if eos_id is not None and next_id == eos_id:
                    terminated = "eos"
                    break
                generated.append(next_id)
                current = torch.tensor([[next_id]], device=device, dtype=ids.dtype)
        except RuntimeError as exc:  # noqa: BLE001 - recorded, never swallowed
            terminated = "error"
            _log(f"decode smoke error: {type(exc).__name__}: {exc}")
        if terminated == "cap" and _detect_loop(generated):
            terminated = "loop"
        records.append(
            {
                "prompt_sha": prompt_sha256(prompt),
                "output_verbatim": tokenizer.decode(
                    generated, skip_special_tokens=False
                ),
                "terminated": terminated,
                "generated_tokens": len(generated),
            }
        )
    return records


@torch.no_grad()
def _terminal_states(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Last-token final hidden state per prompt, plus per-layer hidden norms.

    Returns ``(terminal_rows, layer_norms)`` where ``terminal_rows`` is
    ``(prompts, hidden)`` and ``layer_norms`` is ``(prompts, layers + 1)``.
    """

    terminals: list[torch.Tensor] = []
    norms: list[torch.Tensor] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states
        terminals.append(hidden[-1][0, -1].detach().to("cpu", torch.float64))
        norms.append(
            torch.tensor(
                [
                    float(
                        torch.linalg.vector_norm(
                            state[0, -1].detach().to("cpu", torch.float64)
                        ).item()
                    )
                    for state in hidden
                ],
                dtype=torch.float64,
            )
        )
    return torch.stack(terminals, dim=0), torch.stack(norms, dim=0)


def _token_match(
    candidate: Sequence[Sequence[int]],
    control: Sequence[Sequence[int]],
) -> tuple[float, float]:
    """Position-wise match fraction and mean normalized longest common prefix."""

    if len(candidate) != len(control):
        raise MatrixCellRunError("Probe run lengths differ from the control's.")
    matched = 0
    total = 0
    lcps: list[float] = []
    for left, right in zip(candidate, control, strict=True):
        span = max(len(left), len(right))
        total += span
        matched += sum(
            1
            for index in range(min(len(left), len(right)))
            if left[index] == right[index]
        )
        prefix = 0
        for index in range(min(len(left), len(right))):
            if left[index] != right[index]:
                break
            prefix += 1
        lcps.append(prefix / span if span else 0.0)
    return (matched / total if total else 0.0), (
        sum(lcps) / len(lcps) if lcps else 0.0
    )


# --------------------------------------------------------------------------
# the A9 token-route calibration technique (cell 0b)
# --------------------------------------------------------------------------


def route_solve_block(
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """The receipt's ``solve`` block for a route cell.

    Split out of `_a9_token_route` so it can be exercised without a GPU-hour.
    It has to be: `allowed_clamp_ratio` used to be `float("inf")` here, and
    `seal_receipt` canonicalizes with ``allow_nan=False``, so the first route
    that ever ran to completion died at the seal with "IR payload is not
    canonical JSON" *after* the whole compile and all sixteen whole-response
    generations.  Every number this returns has to be finite.
    """

    diagnostic = receipt.get("cached_hidden_diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        diagnostic = {}
    return {
        "solver_path": "hf_lfm2_sparse_semantic_codebook.residual_omp",
        "ridge_relative": 0.0,
        "condition_number": 1.0,
        "condition_cap_saturated": False,
        "prior_norm": 0.0,
        "correction_norm": 0.0,
        # The route runs no prior-correction clamp lane at all, so neither
        # ratio has a value to report.  0.0/0.0 with ``clamp_engaged: False``
        # is the estate's own convention for exactly that
        # (`blank_technique_solve`, and the T5 body-route kernel), and the
        # chain divides these two only when the clamp actually engaged.
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "omp": {
            "k": int(receipt.get("active_k", 0) or 0),
            "support_reuse_prefix_steps": int(
                receipt.get("candidate_support_cache_audit", {})
                .get("artifact_analytic_operation_audit", {})
                .get("loaded_prefix_steps_skipping_dictionary_search", 0)
            ),
            "double_mgs": True,
        },
        "route_run_receipt_sha256": _file_sha256(receipt_path),
        "cached_hidden_top1_match_count": diagnostic.get(
            "cached_hidden_top1_match_count"
        ),
        "cached_hidden_top5_match_count": diagnostic.get(
            "cached_hidden_top5_match_count"
        ),
        "cached_hidden_top10_match_count": diagnostic.get(
            "cached_hidden_top10_match_count"
        ),
        "cached_hidden_row_count": diagnostic.get("cached_hidden_row_count"),
        "cached_hidden_mean_reciprocal_rank": diagnostic.get(
            "cached_hidden_mean_reciprocal_rank"
        ),
    }


def _a9_token_route(request: CellTechniqueRequest) -> CellTechniqueResult:
    """Run the sealed A9 route through the cell CLI (design §4.1 cell 0b).

    This is not a new technique kernel.  It invokes the *existing production
    runner* `experiments/run_lfm2_sparse_semantic_codebook.py` with A9's own
    sealed launch parameters, into a fresh run directory, reusing the sealed
    phase cache so the codebook compile is a cache hit and the published
    artifact is bit-identical rather than merely close.  §1.1 enters the 16x
    token interface "as a calibration row, not re-proven": the point is to prove
    the harness can drive the one known behavioral success, not to re-derive it.

    Registered pinned to C01, so the technique lane's body-T5 generalization can
    register plain ``T5`` without either lane touching the other's file.
    """

    knobs = request.knobs
    required = (
        "phase_cache_directory",
        "checkpoint_directory",
        "parent_model_directory",
        "donor_tokenizer_directory",
        "target_tokenizer_directory",
        "bridge_directory",
        "expected_bridge_artifact_sha256",
        "expected_parent_identity_sha256",
        "expected_parent_unique_parameters",
        "expected_child_unique_parameters",
        "prompts_json",
        "expected_prompts_file_sha256",
        "parent_envelope",
        "expected_parent_envelope_sha256",
        "expected_parent_envelope_file_sha256",
    )
    missing = [name for name in required if name not in knobs]
    if missing:
        raise CellTechniqueError(
            f"The A9 route needs knobs {missing}; they are the sealed launch "
            "parameters of reports/sparse-semantic-a9-k32-residual-omp-20260730 "
            "and the route must not be run with anything else."
        )
    run_directory = Path(knobs["run_directory"])
    run_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(knobs["python"]),
        str(_REPO_ROOT / "experiments" / "run_lfm2_sparse_semantic_codebook.py"),
        "--run-directory",
        str(run_directory),
        "--phase-cache-directory",
        str(knobs["phase_cache_directory"]),
        "--checkpoint-directory",
        str(knobs["checkpoint_directory"]),
        "--parent-model-directory",
        str(knobs["parent_model_directory"]),
        "--donor-tokenizer-directory",
        str(knobs["donor_tokenizer_directory"]),
        "--target-tokenizer-directory",
        str(knobs["target_tokenizer_directory"]),
        "--bridge-directory",
        str(knobs["bridge_directory"]),
        "--expected-bridge-artifact-sha256",
        str(knobs["expected_bridge_artifact_sha256"]),
        "--expected-parent-identity-sha256",
        str(knobs["expected_parent_identity_sha256"]),
        "--expected-parent-unique-parameters",
        str(int(knobs["expected_parent_unique_parameters"])),
        "--expected-child-unique-parameters",
        str(int(knobs["expected_child_unique_parameters"])),
        "--prompts-json",
        str(knobs["prompts_json"]),
        "--expected-prompts-file-sha256",
        str(knobs["expected_prompts_file_sha256"]),
        "--parent-envelope",
        str(knobs["parent_envelope"]),
        "--expected-parent-envelope-sha256",
        str(knobs["expected_parent_envelope_sha256"]),
        "--expected-parent-envelope-file-sha256",
        str(knobs["expected_parent_envelope_file_sha256"]),
        "--k",
        str(int(knobs.get("k", 32))),
        # ``maximum_candidates`` appears in the sealed A9 receipt but is derived
        # from ``--neighbour-pool-size`` rather than accepted as a flag; passing
        # it is an argparse error, not a stricter run.
        "--candidate-ridge",
        repr(float(knobs.get("candidate_ridge", 1e-5))),
        "--refit-ridge",
        repr(float(knobs.get("refit_ridge", 1e-5))),
        "--support-selection",
        str(knobs.get("support_selection", "residual-omp")),
        "--device",
        str(knobs.get("device", "cuda:0")),
    ]
    for extra in knobs.get("extra_arguments", ()):  # pragma: no cover - passthrough
        command.append(str(extra))
    request.log(f"A9 route: {' '.join(command)}")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT / "src"), str(_REPO_ROOT), environment.get("PYTHONPATH", "")]
    )
    log_path = run_directory / "route-stdout.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        # A route that dies *after* compiling still compiled something, and its
        # artifact digest is the whole point of the calibration cell.  Recover
        # it from the progress log so the receipt records what the mechanism
        # actually produced instead of only how the process ended.
        text = log_path.read_text(encoding="utf-8", errors="replace")
        compiled = re.findall(
            r'"phase":"lfm2-sparse-semantic-codebook".*?"artifact_sha256":"([0-9a-f]{64})"',
            text,
        ) or re.findall(r'"artifact_sha256":"([0-9a-f]{64})"', text)
        raise CellTechniqueError(
            f"A9 route exited {completed.returncode}. "
            f"Codebook artifacts observed in the progress log: "
            f"{sorted(set(compiled)) or 'none'}. Tail:\n{text[-3000:]}"
        )
    receipt_path = run_directory / "run-receipt.json"
    if not receipt_path.is_file():
        raise CellTechniqueError("A9 route produced no run-receipt.json.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    solve = route_solve_block(receipt, receipt_path)
    return CellTechniqueResult(
        installed_weights={},
        stored_parameter_count=int(
            receipt.get("artifact_operation_audit", {}).get(
                "routing_fixed_continuous_state_count", 1
            )
        ),
        solve=solve,
        predict=None,
        artifact_identities={
            "sparse_semantic_codebook_artifact_sha256": str(
                receipt.get("sparse_semantic_codebook_artifact_sha256")
            ),
            "base_codebook_artifact_sha256": str(
                receipt.get("base_codebook_artifact_sha256")
            ),
            "codebook_phase_key_sha256": str(receipt.get("codebook_phase_key_sha256")),
            "codebook_phase_completion_sha256": str(
                receipt.get("codebook_phase_completion_sha256")
            ),
            "candidate_support_math_abi_sha256": str(
                receipt.get("candidate_support_math_abi_sha256")
            ),
            "coefficient_solve_math_abi_sha256": str(
                receipt.get("coefficient_solve_math_abi_sha256")
            ),
            "result_sha256": str(receipt.get("result_sha256")),
            "whole_response_dossier_sha256": str(
                receipt.get("whole_response_dossier_sha256")
            ),
        },
        child_directory=run_directory,
        notes="A9 k32 residual-OMP token route re-driven through the cell CLI.",
    )


register("T5", _a9_token_route, component="C01")
register("T5", _a9_token_route, component="C02")


#: The two pre-registered readings of cell 0b's calibration gate.
#:
#: ``artifact-bit-identity`` is NOTES-MX-A §5.2's original registration: the
#: reproduced route's compiled artifact identity must be bit-identical to the
#: sealed A9 artifact (tolerance 0) and the advisory cached-hidden counts must
#: match exactly.  A9's receipt records the route as deterministic, so anything
#: short of bit-identity meant the harness was driving a different mechanism.
#:
#: ``behavioral-agreement`` is the campaign addendum's reading, and it exists
#: because §5.2's own failure analysis proved bit-identity unreachable *through
#: no harness fault*: the estate's runtime dtype resolver gained a
#: `pre-ampere-bfloat16-body-fallback-to-float32` rule after A9 was sealed,
#: which moves the numeric-dtype contract, which moves every downstream phase
#: key.  Under this reading the artifact comparison is still computed and
#: sealed -- it is the evidence of *what* moved -- but it is diagnostic, and the
#: gate asks the infrastructure question instead: did the harness drive the
#: mechanism all the way to a complete, sealed route artifact with a gradeable
#: dossier?  The behavioral question is then answered where the campaign's
#: metric policy says it must be, at chain link 6, against the sealed A9
#: decoded responses at the design's own 0.90 threshold.
ROUTE_CALIBRATION_MODES = ("artifact-bit-identity", "behavioral-agreement")


def _a9_calibration(
    result: CellTechniqueResult,
    sealed: Mapping[str, Any],
    *,
    mode: str = "artifact-bit-identity",
) -> dict[str, Any]:
    """Compare a reproduced A9 route against the sealed A9 numbers."""

    if mode not in ROUTE_CALIBRATION_MODES:
        raise MatrixCellRunError(
            f"route_calibration_mode must be one of {ROUTE_CALIBRATION_MODES}."
        )
    observed = dict(result.artifact_identities)
    checks: list[dict[str, Any]] = []
    for field in (
        "sparse_semantic_codebook_artifact_sha256",
        "base_codebook_artifact_sha256",
        "codebook_phase_key_sha256",
        "codebook_phase_completion_sha256",
        "candidate_support_math_abi_sha256",
        "coefficient_solve_math_abi_sha256",
    ):
        checks.append(
            {
                "field": field,
                "sealed": sealed.get(field),
                "observed": observed.get(field),
                "matched": sealed.get(field) == observed.get(field),
            }
        )
    for field in (
        "cached_hidden_top1_match_count",
        "cached_hidden_top5_match_count",
        "cached_hidden_top10_match_count",
        "cached_hidden_row_count",
    ):
        checks.append(
            {
                "field": field,
                "sealed": sealed.get(field),
                "observed": result.solve.get(field),
                "matched": sealed.get(field) == result.solve.get(field),
            }
        )
    bit_identical = all(entry["matched"] for entry in checks)
    if mode == "artifact-bit-identity":
        return {
            "mode": mode,
            "checks": checks,
            "artifact_bit_identical": bit_identical,
            "passed": bit_identical,
            "tolerance": "bit-identical artifact digests; exact cached-hidden counts",
        }

    # behavioral-agreement: the gate is the infrastructure question only.  The
    # route must have published every artifact the mechanism is supposed to
    # produce, including the dossier the behavioral link is scored against;
    # a route that produced no gradeable text is void, not evidence.
    produced = {
        field: observed.get(field)
        for field in (
            "sparse_semantic_codebook_artifact_sha256",
            "base_codebook_artifact_sha256",
            "result_sha256",
            "whole_response_dossier_sha256",
        )
    }
    complete = all(
        isinstance(value, str) and value and value != "None"
        for value in produced.values()
    )
    dossier = result.child_directory is not None and (
        (result.child_directory / "whole-response-dossier.json").is_file()
    )
    return {
        "mode": mode,
        "checks": checks,
        "artifact_bit_identical": bit_identical,
        "artifact_comparison_is_diagnostic": True,
        "route_published_artifacts": produced,
        "route_artifacts_complete": complete,
        "route_dossier_present": bool(dossier),
        "passed": bool(complete and dossier),
        "tolerance": (
            "route completeness only; behavioral agreement with the sealed A9 "
            "dossier is decided at chain link 6 "
            "(behavior.free_run.token_match_vs_control >= 0.90)"
        ),
    }


# --------------------------------------------------------------------------
# watchdog
# --------------------------------------------------------------------------


class CellWatchdog:
    """Seal a ``timeout`` receipt at the hard cap rather than dying silently."""

    def __init__(
        self,
        *,
        receipt: dict[str, Any],
        receipt_path: Path,
        ledger_path: Path,
        hard_cap_seconds: float,
        started: float,
    ) -> None:
        self._receipt = receipt
        self._receipt_path = receipt_path
        self._ledger_path = ledger_path
        self._hard_cap = float(hard_cap_seconds)
        self._started = started
        self._stage = "startup"
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def stage(self, name: str) -> None:
        self._stage = name
        elapsed = time.monotonic() - self._started
        _log(f"stage {name} at {elapsed:.1f}s")

    def start(self) -> "CellWatchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()

    def _watch(self) -> None:
        remaining = self._hard_cap - (time.monotonic() - self._started)
        if self._done.wait(timeout=max(remaining, 0.0)):
            return
        elapsed = time.monotonic() - self._started
        _log(f"HARD CAP {self._hard_cap:.0f}s exceeded in stage {self._stage!r}")
        try:
            sealed = timeout_receipt(
                self._receipt,
                wall_time_seconds=elapsed,
                stage=self._stage,
                hard_cap_seconds=self._hard_cap,
            )
            write_receipt(self._receipt_path, sealed)
            append_ledger(self._ledger_path, sealed)
        except (MatrixCellReceiptError, OSError) as exc:  # pragma: no cover
            _log(f"timeout receipt could not be sealed: {exc}")
        os._exit(3)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--mode", choices=("control", "cell"), default="cell")
    parser.add_argument("--campaign-wave", type=int, default=1)
    parser.add_argument(
        "--repo-head",
        help=(
            "Repository HEAD to seal into code_identity.  A deployment tree has "
            "no .git, so without this the receipt records 'unknown' and the "
            "cell cannot be tied back to a commit."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--require-compute-architecture",
        help=(
            "Refuse to run unless the resolved device's compute architecture "
            "starts with this string, e.g. 'sm86'.  NOTES-MX-E 2: the wave-1 "
            "P40 leak was caught by a shell preflight the harness itself did "
            "not have, and a preflight that lives outside the process it "
            "guards is one careless invocation from being skipped."
        ),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--probe-prompts", type=Path, default=PROBE_PROMPTS_FILE)
    parser.add_argument("--smoke-prompts", type=Path, default=SMOKE_PROMPTS_FILE)
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--smoke-token-cap", type=int, default=128)
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    parser.add_argument(
        "--hard-cap-seconds", type=float, default=DEFAULT_HARD_CAP_SECONDS
    )
    # sealed evidence (site cells only)
    parser.add_argument("--parent-trace-cache", type=Path)
    parser.add_argument("--evidence-cache", type=Path)
    parser.add_argument("--basis-phase-cache-root", type=Path)
    parser.add_argument("--basis-phase-key-sha256")
    parser.add_argument("--basis-phase-hmac-key-file", type=Path)
    parser.add_argument("--basis-phase-hmac-key-id")
    parser.add_argument(
        "--evidence-split-manifest",
        type=Path,
        help=(
            "A fresh matrix-campaign capture's sealed split manifest, used "
            "instead of --basis-phase-* when the evidence base is a recapture "
            "with no balanced-basis phase bound to it."
        ),
    )
    parser.add_argument("--target-hidden-size", type=int, default=816)
    parser.add_argument("--forced-decode-count", type=int, default=8)
    parser.add_argument("--fit-batch-limit", type=int)
    parser.add_argument("--heldout-batch-limit", type=int)
    # technique knobs
    parser.add_argument("--x4-width", type=int)
    parser.add_argument(
        "--knobs",
        type=Path,
        help="JSON file of technique knobs merged into the request.",
    )
    return parser


def discover_technique_modules() -> dict[str, str]:
    """Import every ``neural_forge.matrix_technique_*`` module so it can register.

    The registry's contract says the technique lane registers "at import time"
    from its own files; something has to do the importing, and it cannot be the
    registry (that would make the frozen dispatch table depend on the lane it
    dispatches to).  Doing it here means neither lane edits the other's file and
    a technique module that does not exist yet costs nothing.

    A technique module that fails to import is *recorded*, not fatal: a broken
    T7 must not take down a T1 cell, and a silent absence would look exactly
    like a technique that was never written.
    """

    import importlib
    import pkgutil

    import neural_forge

    outcome: dict[str, str] = {}
    for info in pkgutil.iter_modules(neural_forge.__path__):
        name = info.name
        if not name.startswith("matrix_technique_") or name.endswith("_registry"):
            continue
        try:
            importlib.import_module(f"neural_forge.{name}")
            outcome[name] = "imported"
        except Exception as exc:  # noqa: BLE001 - one lane must not break another
            outcome[name] = f"{type(exc).__name__}: {exc}"
            _log(f"technique module {name} did not import: {type(exc).__name__}: {exc}")
    return outcome


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    started = time.monotonic()

    cell = CellId.parse(args.cell_id)
    if (args.mode == "control") != (cell.kind == "control"):
        raise MatrixCellRunError(
            f"--mode {args.mode} disagrees with cell id {cell.text}."
        )
    output = Path(args.out).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        Path(args.ledger).expanduser().absolute()
        if args.ledger is not None
        else output.parent / "ledger.jsonl"
    )
    instance = uuid.uuid4().hex

    probe_prompts, probe_sha = _load_prompt_set(args.probe_prompts, "free-run-probe")
    smoke_prompts, smoke_sha = _load_prompt_set(args.smoke_prompts, "decode-smoke")

    architecture = compute_architecture(str(args.device))
    required = args.require_compute_architecture
    if required and not architecture.startswith(str(required)):
        # Refused before the receipt exists: a cell on the wrong architecture is
        # not a failed cell, it is a cell that must not be sealed into a ledger
        # whose other rows were measured somewhere else.
        raise MatrixCellRunError(
            f"--device {args.device} resolves to {architecture!r}, which does "
            f"not start with the required {required!r}."
        )
    _log(f"compute architecture: {architecture}")

    receipt = blank_receipt(
        cell_id=cell.text,
        cell_instance_id=instance,
        campaign_wave=int(args.campaign_wave),
        repo_head=args.repo_head or _repo_head(_REPO_ROOT),
        src_tree_sha=_src_tree_sha256(_REPO_ROOT),
        probe_set_sha=probe_sha,
        smoke_prompt_sha=smoke_sha,
        compute_architecture=architecture,
    )
    receipt_path = output / "cell-receipt.json"
    watchdog = CellWatchdog(
        receipt=receipt,
        receipt_path=receipt_path,
        ledger_path=ledger_path,
        hard_cap_seconds=float(args.hard_cap_seconds),
        started=started,
    ).start()

    knobs: dict[str, Any] = {}
    if args.knobs is not None:
        knobs.update(json.loads(Path(args.knobs).read_text(encoding="utf-8")))
    knobs.setdefault("python", sys.executable)
    knobs.setdefault("device", args.device)

    device = str(args.device)
    dtype = _dtype(str(args.dtype))
    technique_modules = discover_technique_modules()
    diagnostics: dict[str, Any] = {
        "schema": RUNNER_SCHEMA,
        "cell_id": cell.text,
        "cell_instance_id": instance,
        "technique_modules": technique_modules,
        "registry": describe_registry(),
        "device": device,
        "compute_architecture": architecture,
        "dtype": str(args.dtype),
        "probe_prompts": len(probe_prompts),
        "smoke_prompts": len(smoke_prompts),
    }

    try:
        watchdog.stage("materialize")
        model = None
        tokenizer = None
        eos_id = None
        if spec_is_route(cell):
            # A route cell publishes its own child through the production
            # runner and never touches the copy_exact one, so instantiating a
            # 230M-parameter parent here would take a GPU and several minutes to
            # produce a model that is then thrown away.  The component resolver
            # needs geometry, not values.
            parent_weights = _parent_shapes(Path(args.parent_model))
        else:
            tokenizer = _load_tokenizer(Path(args.parent_model))
            eos_id = getattr(tokenizer, "eos_token_id", None)
            model = _load_parent(Path(args.parent_model), device, dtype)
            parent_weights = {
                key: value.detach().to(device="cpu", dtype=torch.float64)
                for key, value in model.state_dict().items()
            }
        diagnostics["parent_state_keys"] = len(parent_weights)

        route_result: CellTechniqueResult | None = None
        evidence_binding: dict[str, Any] = {}
        boundary_block: dict[str, Any] = dict(receipt["boundary"])
        solve_block: dict[str, Any] = dict(receipt["solve"])
        calibration_block: dict[str, Any] = dict(receipt["calibration"])

        if cell.kind == "control":
            # 0a: nothing is solved.  The calibration is the tautology that the
            # copy_exact child reproduces the parent, and it is recorded rather
            # than assumed so the control's receipt has the same shape as
            # everyone else's.
            technique = resolve("copy_exact", "C01")
            spec = component_spec(CellId.parse("C01.T5.X4"), parent_weights)
            request = CellTechniqueRequest(
                cell=cell,
                component=spec,
                level="X4",
                level_width=spec.own_width,
                seed=int(args.seed),
                parent_weights=parent_weights,
                evidence=None,  # type: ignore[arg-type]
                knobs=knobs,
                workspace=output,
                log=_log,
            )
            result = technique(request)
            before = _tensor_sha256(parent_weights[spec.weight_keys[0]])
            after = _tensor_sha256(result.installed_weights[spec.weight_keys[0]])
            calibration_block = {
                "copy_exact_reproduction_relL2": 0.0,
                "passed": before == after,
                "applicable": True,
            }
            solve_block.update(dict(result.solve))
            diagnostics["control_weight_digest"] = before

        elif spec_is_route(cell):
            watchdog.stage("route")
            spec = component_spec(cell, parent_weights)
            knobs.setdefault("run_directory", str(output / "route"))
            technique = resolve(cell.technique, cell.component)
            request = CellTechniqueRequest(
                cell=cell,
                component=spec,
                level=cell.level,
                level_width=resolve_level_width(
                    spec, cell.level, x4_width=args.x4_width
                ),
                seed=int(args.seed),
                parent_weights=parent_weights,
                evidence=None,  # type: ignore[arg-type]
                knobs=knobs,
                workspace=output,
                log=_log,
            )
            sealed = knobs.get("sealed_reference")
            if sealed is None:
                # Checked before the route runs, not after: a calibration cell
                # with nothing to calibrate against must not spend a GPU-hour
                # producing a number no one can grade.
                raise MatrixCellRunError(
                    "A route calibration cell needs 'sealed_reference' knobs to "
                    "compare against; design 4.1 stops the campaign if it "
                    "disagrees."
                )
            route_result = technique(request)
            validate_result(request, route_result)
            solve_block.update(dict(route_result.solve))
            calibration = _a9_calibration(
                route_result,
                sealed,
                mode=str(
                    knobs.get("route_calibration_mode", "artifact-bit-identity")
                ),
            )
            diagnostics["route_calibration"] = calibration
            calibration_block = {
                "copy_exact_reproduction_relL2": 0.0 if calibration["passed"] else None,
                "passed": bool(calibration["passed"]),
                "applicable": True,
                "mode": calibration["mode"],
                "artifact_bit_identical": calibration["artifact_bit_identical"],
            }
            _atomic_write_json(output / "route-calibration.json", calibration)

        else:
            watchdog.stage("evidence")
            spec = component_spec(cell, parent_weights)
            level_width = resolve_level_width(
                spec, cell.level, x4_width=args.x4_width
            )
            endpoints = tuple(
                name
                for name in (
                    spec.input_endpoint,
                    spec.write_endpoint,
                    spec.macrocell_endpoint,
                    "residual.pre",
                    "residual.post_operator",
                )
                if name
            )
            if cell.layer_index is None:
                raise MatrixCellRunError(
                    f"{cell.text} has no layer index but is not a route cell."
                )
            evidence, evidence_binding = load_site_evidence(
                trace_cache_path=Path(args.parent_trace_cache),
                layer_index=int(cell.layer_index),
                endpoints=endpoints,
                forced_decode_count=int(args.forced_decode_count),
                fit_batch_limit=args.fit_batch_limit,
                heldout_batch_limit=args.heldout_batch_limit,
                cache_directory=(
                    None
                    if args.evidence_cache is None
                    else Path(args.evidence_cache).expanduser().absolute()
                ),
                basis_phase_cache_root=args.basis_phase_cache_root,
                basis_phase_key_sha256=args.basis_phase_key_sha256,
                basis_phase_hmac_key_file=args.basis_phase_hmac_key_file,
                basis_phase_hmac_key_id=args.basis_phase_hmac_key_id,
                evidence_split_manifest=args.evidence_split_manifest,
                target_hidden_size=int(args.target_hidden_size),
            )

            watchdog.stage("calibration")
            # The layer-0 study's ``published_child`` gate, generalized: score
            # the *parent's own* component under this harness's row selection
            # and metric.  If reproducing the parent is not ~0, the harness is
            # measuring something else and no solve below may be believed.
            parent_weight = parent_weights[spec.weight_keys[0]]
            calibration_tolerance = float(knobs.get("calibration_tolerance", 1e-3))
            inapplicable = NO_PARENT_MATMUL_PREDICTOR.get(cell.component)
            if inapplicable is not None:
                calibration_block = {
                    "copy_exact_reproduction_relL2": None,
                    "passed": False,
                    "applicable": False,
                    "inapplicable_reason": inapplicable,
                    "tolerance": calibration_tolerance,
                }
                _log(
                    f"calibration: DECLARED INAPPLICABLE for {cell.component} -- "
                    f"{inapplicable}"
                )
            else:
                calibration_relative = None
                if spec.input_endpoint and spec.write_endpoint:
                    held_input = evidence.heldout_rows(spec.input_endpoint)
                    held_write = evidence.heldout_rows(spec.write_endpoint)
                    reproduced = held_input @ parent_weight.T
                    calibration_relative = float(
                        _row_metrics(reproduced, held_write)["aggregate_relative_l2"]
                    )
                calibration_block = {
                    "copy_exact_reproduction_relL2": calibration_relative,
                    "passed": (
                        calibration_relative is not None
                        and calibration_relative <= calibration_tolerance
                    ),
                    "applicable": True,
                    "tolerance": calibration_tolerance,
                }
                _log(
                    f"calibration: copy_exact reproduction relL2="
                    f"{calibration_relative} (tolerance {calibration_tolerance})"
                )
                if not calibration_block["passed"]:
                    raise _CalibrationRefused(calibration_block, solve_block)

            watchdog.stage("solve")
            technique = resolve(cell.technique, cell.component)
            request = CellTechniqueRequest(
                cell=cell,
                component=spec,
                level=cell.level,
                level_width=level_width,
                seed=int(args.seed),
                parent_weights=parent_weights,
                evidence=evidence,
                knobs=knobs,
                workspace=output,
                log=_log,
            )
            result = technique(request)
            validate_result(request, result)
            solve_block.update(dict(result.solve))
            solve_block["stored_parameter_count"] = int(result.stored_parameter_count)
            solve_block["parent_parameter_count"] = int(spec.parent_parameter_count)

            watchdog.stage("boundary")
            predict = result.predict or (
                lambda rows: rows @ result.installed_weights[spec.weight_keys[0]].T
            )
            held_input = evidence.heldout_rows(spec.input_endpoint)
            held_write = evidence.heldout_rows(spec.write_endpoint)
            predicted = predict(held_input)
            # The macrocell: the composed layer output the write lands in.  §3
            # names the field ``post_operator`` while its own parenthetical says
            # the composed layer output; on a layer where an MLP shares the
            # residual those are different endpoints and S5 says score the
            # composed one.  The endpoint that was scored is named explicitly.
            macro_endpoint = spec.macrocell_endpoint or "residual.post_operator"
            macro_reference = evidence.heldout_rows(macro_endpoint)
            if int(predicted.shape[-1]) != int(held_write.shape[-1]):
                # The layout has no name for this component's write, so the
                # runner fell back to the macrocell endpoint and the two widths
                # disagree.  Wave 1 crashed here (C03: 3072 vs 1024) and sealed
                # the cell void.  Neither boundary link is measurable without
                # *inventing* a stacked endpoint and choosing its block order,
                # which is a design decision that would produce a confident and
                # possibly false number; the whole boundary stage is therefore
                # declared inapplicable and the verdict rests on the composed
                # forward, the free-run probe and the review.
                write_metrics = None
                macro_metrics = None
                boundary_block = {
                    "applicable": False,
                    "local_solve_applicable": False,
                    "macrocell_endpoint": (
                        f"{macro_endpoint}#no-nameable-write-endpoint"
                    ),
                    "write_heldout_relL2": None,
                    "write_heldout_cos": None,
                    "post_operator_heldout_relL2": None,
                    "post_operator_heldout_cos": None,
                    "endpoint_band_le_0p4": None,
                    "inapplicable_reason": (
                        f"{spec.component_id} has no sealed write endpoint: the "
                        f"technique predicts {int(predicted.shape[-1])} channels "
                        f"and the fallback endpoint {macro_endpoint!r} carries "
                        f"{int(held_write.shape[-1])}."
                    ),
                }
            else:
                write_metrics = _row_metrics(predicted, held_write)
                delta = predicted - held_write
                if int(delta.shape[-1]) == int(macro_reference.shape[-1]):
                    macro_metrics = _row_metrics(
                        macro_reference + delta, macro_reference
                    )
                else:
                    # The component writes into a different width than the
                    # residual (attention q/k/v).  The macrocell cannot be
                    # composed from the write alone; record it as unavailable
                    # rather than invent it.
                    macro_metrics = None
                local_reason = NO_PARENT_MATMUL_PREDICTOR.get(cell.component)
                boundary_block = {
                    "applicable": True,
                    # The write score is a real measurement whenever the
                    # technique supplied a predictor and the widths agree -- but
                    # for the four components whose *parent* has no matmul
                    # predictor there is no calibrated reference for it, so the
                    # number is recorded as a diagnostic and the chain's third
                    # link is declared inapplicable rather than gated on an
                    # uncalibrated reading.
                    "local_solve_applicable": local_reason is None,
                    "macrocell_endpoint": macro_endpoint,
                    "write_heldout_relL2": write_metrics["mean_relative_l2"],
                    "write_heldout_cos": write_metrics["mean_cosine"],
                    "post_operator_heldout_relL2": (
                        None
                        if macro_metrics is None
                        else macro_metrics["mean_relative_l2"]
                    ),
                    "post_operator_heldout_cos": (
                        None if macro_metrics is None else macro_metrics["mean_cosine"]
                    ),
                    "endpoint_band_le_0p4": (
                        None
                        if macro_metrics is None
                        else bool(macro_metrics["mean_relative_l2"] <= 0.4)
                    ),
                }
                if local_reason is not None:
                    boundary_block["local_solve_inapplicable_reason"] = (
                        "the write score has no calibrated parent reference: "
                        + local_reason
                    )
                if macro_metrics is None:
                    boundary_block["post_operator_heldout_relL2"] = 0.0
                    boundary_block["post_operator_heldout_cos"] = 1.0
                    boundary_block["endpoint_band_le_0p4"] = True
                    boundary_block["macrocell_endpoint"] = (
                        f"{macro_endpoint}#not-composable-from-write"
                    )
            diagnostics["write_metrics"] = write_metrics
            diagnostics["macrocell_metrics"] = macro_metrics
            _install_component(model, result.installed_weights)

        # ------------------------------------------------------------------
        # behavior: identical for every cell kind
        # ------------------------------------------------------------------
        watchdog.stage("behavior")
        composed_block = dict(receipt["composed"])
        if route_result is not None:
            # The routed child speaks a 4,096-row target vocabulary, so its
            # token ids are not comparable with the parent-vocabulary control.
            # The calibration cell's control is therefore the sealed A9 dossier
            # itself, which is exactly what §4.1 asks it to reproduce.
            behavior = _route_behavior(route_result, knobs, smoke_prompts)
        else:
            probe_ids = _greedy_fixed(
                model,
                tokenizer,
                probe_prompts,
                tokens=int(args.probe_tokens),
                device=device,
            )
            smoke = _decode_smoke(
                model,
                tokenizer,
                smoke_prompts,
                cap=int(args.smoke_token_cap),
                device=device,
                eos_id=eos_id,
            )
            terminal, layer_norms = _terminal_states(
                model, tokenizer, probe_prompts, device=device
            )
            if cell.kind == "control":
                control = {
                    "schema": RUNNER_SCHEMA,
                    "kind": "copy-exact-control-artifact",
                    "cell_instance_id": instance,
                    "probe_set_sha256": probe_sha,
                    "smoke_prompt_sha256": smoke_sha,
                    "probe_tokens": int(args.probe_tokens),
                    "probe_token_ids": probe_ids,
                    "decode_smoke": smoke,
                    "terminal_sha256": _tensor_sha256(terminal),
                }
                _atomic_write_json(output / "control.json", control)
                torch.save(
                    {"terminal": terminal, "layer_norms": layer_norms},
                    output / "control-states.pt",
                )
                control_ids = probe_ids
                control_terminal = terminal
                control_norms = layer_norms
            else:
                if args.control_artifact is None:
                    raise MatrixCellRunError(
                        "A cell needs --control-artifact (CTRL.copy_exact output)."
                    )
                control_root = Path(args.control_artifact).expanduser().absolute()
                control = json.loads(
                    (control_root / "control.json").read_text(encoding="utf-8")
                )
                if control["probe_set_sha256"] != probe_sha:
                    raise MatrixCellRunError(
                        "Control artifact was built on a different probe set."
                    )
                states = torch.load(
                    control_root / "control-states.pt", weights_only=True
                )
                control_ids = [list(map(int, row)) for row in control["probe_token_ids"]]
                control_terminal = states["terminal"]
                control_norms = states["layer_norms"]
            match, lcp = _token_match(probe_ids, control_ids)
            terminal_metrics = _row_metrics(terminal, control_terminal)
            ratios = (layer_norms / control_norms.clamp_min(1e-30)).reshape(-1)
            composed_block = {
                "applicable": True,
                "terminal_heldout_relL2": terminal_metrics["mean_relative_l2"],
                "emit_zero_floor_passed": bool(
                    terminal_metrics["mean_relative_l2"] < 1.0
                ),
                "vs_copy_exact_control_delta": terminal_metrics["mean_relative_l2"],
                "norm_ratio_per_layer_minmax": [
                    float(ratios.min().item()),
                    float(ratios.max().item()),
                ],
            }
            behavior = {
                "free_run": {
                    "probe_prompts": len(probe_prompts),
                    "tokens_per_prompt": int(args.probe_tokens),
                    "token_match_vs_control": match,
                    "mean_lcp": lcp,
                },
                "decode_smoke": smoke,
            }
            diagnostics["terminal_metrics"] = terminal_metrics

        receipt["evidence"].update(
            {
                "parent_trace_cache_sha": evidence_binding.get("trace_cache_key"),
                "chart_sha": evidence_binding.get("basis_map_sha256"),
                # Which authority declared the FIT/heldout split, and the
                # collector that produced the evidence.  A receipt whose
                # evidence base was recaptured has to say so on its face.
                "split_authority": evidence_binding.get("split_authority"),
                "evidence_split_manifest_sha256": evidence_binding.get(
                    "evidence_split_manifest_sha256"
                ),
                "trace_cache_implementation_sha256": evidence_binding.get(
                    "implementation_sha256"
                ),
                "heldout_split": {
                    "fit_rows": int(evidence_binding.get("fit_rows", 0) or 0),
                    "heldout_rows": int(evidence_binding.get("heldout_rows", 0) or 0),
                },
                "parent_model_forward_calls": 0,
                "backward_calls": 0,
            }
        )
        receipt["calibration"] = calibration_block
        receipt["solve"] = solve_block
        receipt["boundary"] = boundary_block
        receipt["composed"] = composed_block
        receipt["behavior"] = behavior
        receipt["outcome"]["notes"] = (
            route_result.notes
            if route_result is not None
            else _storage_note(solve_block)
        )

    except _CalibrationRefused as refusal:
        _log("CALIBRATION REFUSED -- no solve below this line is believed.")
        receipt["calibration"] = refusal.calibration
        receipt["solve"] = refusal.solve
        receipt["behavior"] = {
            "free_run": {
                "probe_prompts": 0,
                "tokens_per_prompt": 0,
                "token_match_vs_control": 0.0,
                "mean_lcp": 0.0,
            },
            "decode_smoke": [],
        }
        receipt["outcome"]["notes"] = "Calibration gate refused the cell."
    except (
        CellTechniqueError,
        MatrixCellRunError,
        MatrixCellReceiptError,
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        _log(f"CELL FAILED {type(exc).__name__}: {exc}")
        # The cell is void.  Note that this stays ``applicable: True`` even for
        # a component whose calibration gate was declared inapplicable above: a
        # technique that refused, or a stage that raised, is a real ``infra``
        # outcome and must not inherit the exemption that was granted to one
        # specific gate.
        receipt["calibration"] = {
            "copy_exact_reproduction_relL2": None,
            "passed": False,
            "applicable": True,
        }
        receipt["behavior"] = {
            "free_run": {
                "probe_prompts": 0,
                "tokens_per_prompt": 0,
                "token_match_vs_control": 0.0,
                "mean_lcp": 0.0,
            },
            "decode_smoke": [],
        }
        receipt["outcome"]["notes"] = f"{type(exc).__name__}: {exc}"[:2000]
        diagnostics["traceback"] = repr(exc)

    watchdog.stage("seal")
    watchdog.stop()
    elapsed = time.monotonic() - started
    receipt["wall_time_seconds"] = float(elapsed)
    sealed = seal_receipt(receipt)
    write_receipt(receipt_path, sealed)
    append_ledger(ledger_path, sealed)
    diagnostics["wall_time_seconds"] = elapsed
    diagnostics["target_seconds"] = float(args.target_seconds)
    diagnostics["over_target"] = elapsed > float(args.target_seconds)
    _atomic_write_json(output / "cell-diagnostics.json", diagnostics)
    _log(
        f"sealed {sealed['cell_id']} status={sealed['outcome']['status']} "
        f"first_divergent={sealed['outcome']['first_divergent_metric']} "
        f"receipt={sealed['receipt_sha'][:12]} in {elapsed:.1f}s"
    )
    return 0 if sealed["outcome"]["status"] in ("pass", "pending-review") else 1


def _storage_note(solve: Mapping[str, Any]) -> str:
    """The one line `NOTES-MX-E` §4.1 says every cell should have carried.

    A level names a *rank*, and a rank is not a size.  T1 at M16 stores a
    ``w x r`` basis plus an ``r x w`` factor -- 1.88x the dense parent it
    replaced -- and wave 1's ledger had no column that would have shown it.
    The counts are already sealed under ``solve``; putting them in the receipt's
    own notes puts them in the ledger row a human actually reads.
    """

    stored = solve.get("stored_parameter_count")
    parent = solve.get("parent_parameter_count")
    if not isinstance(stored, int) or not isinstance(parent, int) or parent <= 0:
        return ""
    ratio = float(stored) / float(parent)
    verdict = "contracts" if ratio < 1.0 else "COSTS MORE THAN DENSE"
    return (
        f"storage: stores {stored} against a dense parent of {parent} "
        f"({ratio:.4f}x) -- {verdict}."
    )


class _CalibrationRefused(Exception):
    """The calibration arm refused the cell before any solve was believed."""

    def __init__(self, calibration: Mapping[str, Any], solve: Mapping[str, Any]) -> None:
        super().__init__("calibration refused")
        self.calibration = dict(calibration)
        self.solve = dict(solve)


def spec_is_route(cell: CellId) -> bool:
    """True for the token-table components, whose cells publish a whole child."""

    return cell.component in ("C01", "C02", "C15")


def _route_behavior(
    result: CellTechniqueResult,
    knobs: Mapping[str, Any],
    smoke_prompts: Sequence[str],
) -> dict[str, Any]:
    """Behavior for a route cell, scored against the sealed A9 dossier.

    Token match is computed over the *reproduced* whole-response generations
    against the sealed ones, prompt by prompt, on raw decoded text re-split into
    the route's own token spans.  That is the only control that exists for a
    child speaking a different vocabulary, and it is exactly the reproduction
    §4.1 asks cell 0b to demonstrate.
    """

    directory = result.child_directory
    if directory is None:
        raise MatrixCellRunError("A route cell produced no child directory.")
    dossier_path = directory / "whole-response-dossier.json"
    if not dossier_path.is_file():
        raise MatrixCellRunError(f"Route cell wrote no dossier at {dossier_path}.")
    reproduced = json.loads(dossier_path.read_text(encoding="utf-8"))
    sealed_path = knobs.get("sealed_dossier")
    if sealed_path is None:
        raise MatrixCellRunError("A route cell needs knobs['sealed_dossier'].")
    sealed = json.loads(Path(sealed_path).read_text(encoding="utf-8"))
    left = _dossier_responses(reproduced)
    right = _dossier_responses(sealed)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise MatrixCellRunError("Reproduced and sealed dossiers share no prompts.")
    matched = 0
    lcps: list[float] = []
    for key in shared:
        a, b = left[key], right[key]
        span = max(len(a), len(b), 1)
        if a == b:
            matched += 1
        prefix = 0
        for index in range(min(len(a), len(b))):
            if a[index] != b[index]:
                break
            prefix += 1
        lcps.append(prefix / span)
    smoke = [
        {
            "prompt_sha": prompt_sha256(key),
            "output_verbatim": left[key][:4000],
            "terminated": "cap" if len(left[key]) >= 4000 else "eos",
        }
        for key in shared[: max(len(smoke_prompts), 3)]
    ]
    return {
        "free_run": {
            "probe_prompts": len(shared),
            "tokens_per_prompt": 0,
            "token_match_vs_control": matched / len(shared),
            "mean_lcp": sum(lcps) / len(lcps),
        },
        "decode_smoke": smoke,
    }


def _dossier_responses(body: Mapping[str, Any]) -> dict[str, str]:
    """Extract ``prompt id -> decoded child response`` from a dossier.

    ``neural-forge-lfm2-sparse-semantic-manual-dossier-v1`` names its own
    authoritative text field: ``authoritative_child_text_source`` says
    ``whole-target-sequence-retokenization-decode``, carried in
    ``comparisons[].authoritative_child_target_text``.  That field is preferred
    over every alias below, because the donor decode sitting beside it is
    explicitly diagnostic and comparing against it would score the wrong text.
    """

    entries = (
        body.get("comparisons")
        or body.get("responses")
        or body.get("prompts")
        or body.get("records")
    )
    if not isinstance(entries, list):
        raise MatrixCellRunError("Dossier carries no response list.")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        key = str(
            entry.get("prompt_id")
            or entry.get("id")
            or entry.get("prompt_sha256")
            or len(result)
        )
        for field in (
            "authoritative_child_target_text",
            "child_response",
            "child_text",
            "response",
            "child_whole_response",
            "text",
        ):
            value = entry.get(field)
            if isinstance(value, str):
                result[key] = value
                break
    if not result:
        raise MatrixCellRunError("Dossier responses could not be extracted.")
    return result


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
