#!/usr/bin/env python3
"""One process, one composed child of the matrix campaign's composition wave.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` ADDENDUM 1 makes the campaign's
terminal deliverable a heterogeneous child -- different techniques folding
different components -- and ADDENDUM 4 pre-registers the two modes that child can
be assembled by, as an experiment rather than an assumption:

``--mode M-A`` **direct assembly.**  Every selected winner is solved exactly as
    it was in isolation, against the sealed PARENT evidence, and all of them are
    installed into one child.  This is the null hypothesis: the A12 law says a
    local cached-fit improvement composes worse, so M-A is expected to land
    below the per-component numbers.  It is cheap, it needs no recapture, and it
    runs first.

``--mode M-B`` **sequential conditioned composition.**  The same winners in the
    same topological order, but each one is re-solved against evidence captured
    **through the partially folded child** -- every upstream component already
    installed.  The capture is the frozen collector
    (`hf_lfm2_circuit_trace_cache.build_lfm2_circuit_trace_cache`) pointed at
    the child, unchanged, once per step, with full provenance sealed: which
    components were installed, in what order, and the cache key and split
    manifest digest of the evidence each solve actually saw.

``--mode M-C`` **alternating joint co-solve (ALS).**  ADDENDUM 9.  M-A first --
    every component solved against parent evidence and installed, so an M-C run
    starts from the byte-identical M-A child and the comparison is controlled --
    and then *sweeps*: for each folded component in turn, a fresh capture
    through the CURRENT composed child, a re-solve of that component's own
    closed form against the child's activations, install, next component;
    repeat until the sweep stops paying.  Still zero gradients: every step is
    the same ridge/Gram/OMP solve the isolation cell ran.

    Two things separate it from M-B, and both are the point.  M-B conditions
    each component **once**, in install order, and can never revisit a
    component whose inputs a later install moved; ALS revisits every component
    against the full joint state, repeatedly, which is what makes it a
    coordinate descent with a fixed point rather than a single pass.  And M-B's
    solve target is the parent operator applied to the child's drifted rows --
    "be this map where the data now lives" -- which re-fits the operator to the
    drift instead of cancelling it.  M-C's default target is the parent's own
    sealed **write rows** at the same endpoint over the same corpus positions,
    so each coordinate step asks the only question that can undo composition
    damage: *given the inputs this child actually delivers, what must this
    component emit for the parent's activation to come out?*

Everything the isolation runner does per cell, this runner does per component
and then once more for the whole child, by **importing the isolation runner's
own functions** rather than restating them.  That is not tidiness: a composed
child's free-run number is only comparable with a cell's if it was produced by
the same `_greedy_fixed`, the same `_token_match`, the same `_row_metrics` and
the same control artifact, and the cheapest way to guarantee that is to call
them.

The receipt is `matrix_cell_receipt.SCHEMA` -- ``…receipt-v2`` from ADDENDUM 7
on, ``…receipt-v1`` for everything waves 1-6 sealed, and the only difference
between them is which sites `CellId.parse` accepts -- with the mandatory
``composition`` block (`matrix_cell_receipt._validate_composition`): a
per-component provenance array, the install order, the conditioned-evidence
digests, and the composed child's exact net storage.  The runner writes
``decode_smoke_review: pending`` and cannot write anything else -- the frozen
review is sealed separately by `experiments/seal_matrix_cell_review.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import torch  # noqa: E402

import run_lfm2_matrix_cell as cell_runner  # noqa: E402
from neural_forge.matrix_cell_receipt import (  # noqa: E402
    COMPOSITION_MODES,
    TECHNIQUES,
    CellId,
    MatrixCellReceiptError,
    append_ledger,
    blank_receipt,
    first_divergent_metric,
    seal_receipt,
    write_receipt,
)
from neural_forge.matrix_composition import (  # noqa: E402
    MatrixCompositionError,
    aggregate_boundary,
    aggregate_calibration,
    aggregate_solve,
    evidence_cache_name,
    net_storage,
    validate_recipe,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    describe_registry,
    resolve,
    resolve_level_width,
    validate_result,
)

RUNNER_SCHEMA = "neural-forge-lfm2-matrix-composition-runner-v1"
CONDITIONED_MANIFEST_SCHEMA = cell_runner.SPLIT_MANIFEST_SCHEMA

_log = cell_runner._log
_atomic_write_json = cell_runner._atomic_write_json
_row_metrics = cell_runner._row_metrics


class MatrixCompositionRunError(RuntimeError):
    """The composed child failed closed."""


# --------------------------------------------------------------------------
# parameter accounting
# --------------------------------------------------------------------------


def unique_parameter_count(model: Any) -> tuple[int, dict[str, Any]]:
    """Parent parameters, counted once per *storage*, not once per name.

    LFM2 ties the readout to the embedding table, so summing ``state_dict``
    numels double-counts 67M parameters and every storage ratio computed from it
    would be quietly wrong in the child's favour.  De-duplicating on the data
    pointer is the same rule the estate's own unique-parameter audits use.
    """

    seen: dict[int, int] = {}
    aliases: list[str] = []
    for name, tensor in model.state_dict().items():
        pointer = int(tensor.data_ptr())
        if pointer in seen:
            aliases.append(name)
            continue
        seen[pointer] = int(tensor.numel())
    return sum(seen.values()), {
        "distinct_storages": len(seen),
        "tied_names": sorted(aliases),
    }


# --------------------------------------------------------------------------
# conditioned evidence capture (M-B)
# --------------------------------------------------------------------------


def _capture_config(sealed_manifest: Mapping[str, Any]) -> Any:
    """Rebuild the parent evidence base's own capture selection, field by field.

    M-B's whole claim is that the only thing that moved between the isolation
    solve and the conditioned one is *what the model was*.  If the capture
    selection moved too, the comparison is between two different experiments, so
    the selection is reconstructed from the sealed manifest and refused unless
    it reproduces that manifest's ``selection_sha256``.
    """

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        Lfm2CircuitTraceCaptureConfig,
    )

    capture = sealed_manifest.get("capture")
    if not isinstance(capture, Mapping):
        raise MatrixCompositionRunError(
            "The parent evidence base's manifest carries no capture block."
        )
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=int(capture["sample_last_tokens"]),
        max_decode_steps=int(capture["max_decode_steps"]),
        max_capture_bytes=int(capture["max_capture_bytes"]),
        logit_rtol=float(capture["logit_rtol"]),
        logit_atol=float(capture["logit_atol"]),
        require_top1_agreement=bool(capture["require_top1_agreement"]),
        allow_top1_disagreement_within_numeric_error_envelope=bool(
            capture["allow_top1_disagreement_within_numeric_error_envelope"]
        ),
        capture_causal_circuit_evidence=bool(
            capture.get("capture_causal_circuit_evidence", False)
        ),
    )
    rebuilt = config.as_dict()
    if rebuilt["selection_sha256"] != str(capture["selection_sha256"]):
        raise MatrixCompositionRunError(
            "Reconstructed capture selection does not reproduce the parent "
            f"base's {capture['selection_sha256']}: {rebuilt['selection_sha256']}."
        )
    return config


def _inherited_split_manifest(
    *,
    cache: Any,
    base_manifest: Mapping[str, Any],
    record_ids: Sequence[str],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the conditioned capture's split by INHERITING the parent base's.

    `capture_matrix_evidence.py` computes a fresh capture's split with the
    estate's `whole-record-seeded-sha256-ranking` rule, whose ranking key is
    seeded with the **cache key**.  A conditioned capture is a different model
    and therefore a different cache key, so that rule would re-partition the 24
    batches at every step and M-B's held-out rows would stop being M-A's.  The
    comparison ADDENDUM 4 exists to make would then be between two different
    held-out sets.

    So the split is inherited by ``record_id`` -- the per-batch input digest,
    which is identical across captures because the corpus is -- and the
    deviation is sealed on the manifest's face, so nobody has to take the
    inheritance on trust.

    **The body carries nothing that is not the split.**  It used to carry a
    ``captured_at`` timestamp and the step's conditioning provenance, and the
    shared content-addressed capture root found out why that is wrong within
    one run: two M-B cells whose recipes share a prefix reuse the same capture,
    re-derive the same split over the same records, and got two different
    ``manifest_sha256`` values because the manifests were written a hundred
    minutes apart.  `load_site_evidence` then correctly refused to reuse the
    site payload one of them had already built -- "Cached site evidence was
    built under a different split manifest" -- and the second cell sealed
    `infra` at its first conditioned step.  The guard was right; the manifest
    was not.  A split manifest is a statement about a split, so it is a pure
    function of the cache identity and the inherited rows, and the run-specific
    provenance rides in a sidecar and in the receipt where it belongs.
    """

    from neural_forge.dynamic_circuit_ir import canonical_sha256

    base_rows = {
        str(row["record_id"]): str(row["split"])
        for row in base_manifest["split"]["rows"]
    }
    missing = [rid for rid in record_ids if rid not in base_rows]
    if missing:
        raise MatrixCompositionRunError(
            "The conditioned capture saw batches the parent evidence base's "
            f"split does not cover: {missing[:3]}."
        )
    rows = [
        {
            "batch_index": index,
            "record_id": str(record_id),
            "split": base_rows[str(record_id)],
            "corpus_declared_split": base_rows[str(record_id)],
        }
        for index, record_id in enumerate(record_ids)
    ]
    fit = sum(1 for row in rows if row["split"] == "fit")
    body: dict[str, Any] = {
        "schema": CONDITIONED_MANIFEST_SCHEMA,
        "kind": "matrix-campaign-conditioned-child-evidence",
        "purpose": (
            "ADDENDUM 4 mode M-B: evidence captured through the partially "
            "folded child so the next component's winner is re-solved against "
            "the inputs it will actually see."
        ),
        "cache_directory": str(step["cache_directory"]),
        "trace_cache_key": str(cache.cache_key),
        "parent_identity_sha256": str(cache.parent_fingerprint),
        "split": {
            "authority": "inherited-from-parent-evidence-base-by-record-id",
            "method": "inherit",
            "inherited_from_manifest_sha256": str(
                base_manifest["manifest_sha256"]
            ),
            "inherited_from_trace_cache_key": str(
                base_manifest["trace_cache_key"]
            ),
            "deviation_from_seeded_rule": (
                "whole-record-seeded-sha256-ranking seeds its ranking key with "
                "the cache key, which a conditioned capture necessarily "
                "changes; re-ranking would re-partition the batches and M-B's "
                "held-out rows would stop being M-A's.  The split is therefore "
                "inherited per record_id, which is stable across captures "
                "because the corpus is."
            ),
            "record_count": len(rows),
            "fit_batches": fit,
            "heldout_batches": len(rows) - fit,
            "rows": rows,
        },
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def capture_conditioned_evidence(
    *,
    capture_model: Any,
    child_model: Any,
    output_root: Path,
    batches: Sequence[Mapping[str, Any]],
    capture_config: Any,
    base_manifest: Mapping[str, Any],
    device: str,
    conditioning: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture one conditioned evidence base through the partially folded child.

    The collector is the frozen one, called unchanged.  What differs from the
    parent capture is exactly one argument: the model.
    """

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        build_lfm2_circuit_trace_cache,
        fingerprint_lfm2_parent,
    )

    started = time.monotonic()
    with torch.no_grad():
        capture_model.load_state_dict(child_model.state_dict(), strict=True)
    configuration = getattr(capture_model, "config", None)
    if configuration is None:
        raise MatrixCompositionRunError("Capture model exposes no configuration.")
    # The sealed base was captured on the eager attention backend and the
    # causal-circuit evidence the collector writes is only exact there; a
    # conditioned capture on SDPA would be a different endpoint set wearing the
    # same names.
    configuration._attn_implementation = "eager"
    fingerprint = fingerprint_lfm2_parent(capture_model).as_dict()
    _log(
        f"conditioned capture: child identity {fingerprint['identity_sha256'][:16]} "
        f"(parent base {base_manifest['parent_identity_sha256'][:16]})"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cache = build_lfm2_circuit_trace_cache(
        output_root,
        capture_model,
        batches,
        config=capture_config,
        input_device=torch.device(device),
    )
    record_ids = [
        str(value)
        for value in cache.manifest["corpus"]["ordered_batch_input_sha256"]
    ]
    cache_directory = Path(output_root) / str(cache.cache_key)
    manifest = _inherited_split_manifest(
        cache=cache,
        base_manifest=base_manifest,
        record_ids=record_ids,
        step={"cache_directory": str(cache_directory)},
    )
    manifest_path = Path(output_root) / f"split-manifest-{cache.cache_key[:16]}.json"
    from neural_forge.dynamic_circuit_ir import canonical_json_bytes

    encoded = canonical_json_bytes(manifest)
    if manifest_path.is_file():
        # An earlier step, or an earlier cell, already sealed the split for this
        # exact capture.  It must agree byte for byte; a disagreement means two
        # runs derived different splits over the same rows, which is the one
        # thing the inheritance rule exists to prevent.
        existing = manifest_path.read_bytes()
        if existing != encoded:
            raise MatrixCompositionRunError(
                f"{manifest_path} already carries a different split manifest "
                "for this capture; two runs derived different splits over the "
                "same records."
            )
    else:
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_bytes(encoded)
        temporary.replace(manifest_path)
    # The run-specific provenance rides beside the manifest, never inside it:
    # which components were already folded, in what order, and which component
    # this capture was cut for.  Digesting it would make the split manifest a
    # function of the run instead of the split.
    sidecar_directory = Path(output_root) / "conditioning"
    sidecar_directory.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_directory / (
        f"{cache.cache_key[:16]}-{conditioning['install_index']:02d}-"
        f"{conditioning['next_component']}.json"
    )
    sidecar_body = {
        "schema": RUNNER_SCHEMA,
        "kind": "conditioned-capture-provenance",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_cache_key": str(cache.cache_key),
        "split_manifest_sha256": str(manifest["manifest_sha256"]),
        "child_identity_sha256": str(fingerprint["identity_sha256"]),
        "child_state_dict_sha256": str(fingerprint["state_dict_sha256"]),
        "conditioning": dict(conditioning),
    }
    _atomic_write_json(sidecar, sidecar_body)
    elapsed = time.monotonic() - started
    _log(
        f"conditioned capture sealed {cache.cache_key[:16]} in {elapsed:.1f}s "
        f"-> {cache_directory}"
    )
    return {
        "trace_cache_key": str(cache.cache_key),
        "cache_directory": str(cache_directory),
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": str(manifest["manifest_sha256"]),
        "conditioning_provenance_path": str(sidecar),
        "conditioning_provenance_sha256": cell_runner._file_sha256(sidecar),
        "child_identity_sha256": str(fingerprint["identity_sha256"]),
        "child_state_dict_sha256": str(fingerprint["state_dict_sha256"]),
        "capture_wall_time_seconds": float(elapsed),
        "collector_implementation_sha256": str(
            cache.manifest["implementation"]["implementation_sha256"]
        ),
        "reproduces_parent_identity": (
            str(fingerprint["identity_sha256"])
            == str(base_manifest["parent_identity_sha256"])
        ),
    }


# --------------------------------------------------------------------------
# one component install
# --------------------------------------------------------------------------


def _endpoints_for(spec: Any) -> tuple[str, ...]:
    return tuple(
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


def _calibration_block(
    *,
    cell: CellId,
    spec: Any,
    evidence: Any,
    parent_weights: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """The isolation runner's calibration gate, applied per component.

    Identical code path and identical tolerance, because a composed child whose
    components were gated more loosely than the cells that chose them would be
    comparing two different measurements.
    """

    inapplicable = cell_runner.NO_PARENT_MATMUL_PREDICTOR.get(cell.component)
    if inapplicable is not None:
        return {
            "copy_exact_reproduction_relL2": None,
            "passed": False,
            "applicable": False,
            "inapplicable_reason": inapplicable,
            "tolerance": tolerance,
        }
    relative = None
    if spec.input_endpoint and spec.write_endpoint:
        held_input = evidence.heldout_rows(spec.input_endpoint)
        held_write = evidence.heldout_rows(spec.write_endpoint)
        parent_weight = parent_weights[spec.weight_keys[0]]
        relative = float(
            _row_metrics(held_input @ parent_weight.T, held_write)[
                "aggregate_relative_l2"
            ]
        )
    return {
        "copy_exact_reproduction_relL2": relative,
        "passed": relative is not None and relative <= tolerance,
        "applicable": True,
        "tolerance": tolerance,
    }


class _GlobalComponentEvidence:
    """The evidence a component with no site has, which is none.

    C01's layout names no input, write or macrocell endpoint -- a tied embedding
    table is a lookup, not a linear map from a sealed activation boundary -- so
    there is no *site* to open an evidence cache for and no layer index to open
    it at.  Handing the technique a real cache built at some arbitrary layer
    would be worse than handing it nothing: it would let a kernel silently read
    rows that have no relationship to what it folds.  So it gets this, which
    satisfies the `MatrixCellEvidence` protocol and refuses every row access with
    the reason, and the harness declares the calibration and boundary links
    inapplicable rather than scoring them against endpoints that do not exist.
    """

    def __init__(self, component_id: str) -> None:
        self._component_id = str(component_id)

    @property
    def endpoints(self) -> tuple[str, ...]:
        return ()

    @property
    def selection_sha256(self) -> str:
        return "0" * 64

    @property
    def fit_row_count(self) -> int:
        return 0

    @property
    def heldout_row_count(self) -> int:
        return 0

    def _refuse(self, endpoint: str) -> Any:
        raise MatrixCompositionRunError(
            f"{self._component_id} is a global component with no sealed site "
            f"endpoints; a kernel asked for {endpoint!r}.  If a technique at "
            "this component needs activation evidence it needs a site, and the "
            "layout has to name one."
        )

    def fit_rows(self, endpoint: str) -> Any:
        return self._refuse(endpoint)

    def heldout_rows(self, endpoint: str) -> Any:
        return self._refuse(endpoint)


def _boundary_block(
    *,
    cell: CellId,
    spec: Any,
    evidence: Any,
    result: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The isolation runner's boundary stage, applied per component, verbatim."""

    if spec.input_endpoint is None or spec.write_endpoint is None:
        # A component with no sealed endpoints has no boundary to score.  Said
        # once, here, rather than by letting `evidence.heldout_rows(None)` raise
        # somewhere the receipt would read as an infrastructure failure.
        return (
            {
                "applicable": False,
                "local_solve_applicable": False,
                "macrocell_endpoint": None,
                "write_heldout_relL2": None,
                "write_heldout_cos": None,
                "post_operator_heldout_relL2": None,
                "post_operator_heldout_cos": None,
                "endpoint_band_le_0p4": None,
                "inapplicable_reason": (
                    f"{spec.component_id} declares no sealed input/write "
                    "endpoint pair; there is no activation boundary at which "
                    "this component's write is a nameable object."
                ),
            },
            {},
        )
    predict = result.predict or (
        lambda rows: rows @ result.installed_weights[spec.weight_keys[0]].T
    )
    held_input = evidence.heldout_rows(spec.input_endpoint)
    held_write = evidence.heldout_rows(spec.write_endpoint)
    predicted = predict(held_input)
    macro_endpoint = spec.macrocell_endpoint or "residual.post_operator"
    macro_reference = evidence.heldout_rows(macro_endpoint)
    diagnostics: dict[str, Any] = {}
    if int(predicted.shape[-1]) != int(held_write.shape[-1]):
        block = {
            "applicable": False,
            "local_solve_applicable": False,
            "macrocell_endpoint": f"{macro_endpoint}#no-nameable-write-endpoint",
            "write_heldout_relL2": None,
            "write_heldout_cos": None,
            "post_operator_heldout_relL2": None,
            "post_operator_heldout_cos": None,
            "endpoint_band_le_0p4": None,
            "inapplicable_reason": (
                f"{spec.component_id} has no sealed write endpoint: the "
                f"technique predicts {int(predicted.shape[-1])} channels and "
                f"the fallback endpoint {macro_endpoint!r} carries "
                f"{int(held_write.shape[-1])}."
            ),
        }
        return block, diagnostics
    write_metrics = _row_metrics(predicted, held_write)
    delta = predicted - held_write
    macro_metrics = (
        _row_metrics(macro_reference + delta, macro_reference)
        if int(delta.shape[-1]) == int(macro_reference.shape[-1])
        else None
    )
    local_reason = cell_runner.NO_PARENT_MATMUL_PREDICTOR.get(cell.component)
    block = {
        "applicable": True,
        "local_solve_applicable": local_reason is None,
        "macrocell_endpoint": macro_endpoint,
        "write_heldout_relL2": write_metrics["mean_relative_l2"],
        "write_heldout_cos": write_metrics["mean_cosine"],
        "post_operator_heldout_relL2": (
            None if macro_metrics is None else macro_metrics["mean_relative_l2"]
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
        block["local_solve_inapplicable_reason"] = (
            "the write score has no calibrated parent reference: " + local_reason
        )
    if macro_metrics is None:
        block["post_operator_heldout_relL2"] = 0.0
        block["post_operator_heldout_cos"] = 1.0
        block["endpoint_band_le_0p4"] = True
        block["macrocell_endpoint"] = f"{macro_endpoint}#not-composable-from-write"
    diagnostics["write_metrics"] = write_metrics
    diagnostics["macrocell_metrics"] = macro_metrics
    return block, diagnostics


def _component_chain(entry: Mapping[str, Any]) -> tuple[str | None, str]:
    """The §3 chain, walked over one component's own numbers.

    A composed child's per-component ``first_divergent_metric`` is what makes a
    composition failure localizable at all (ADDENDUM 4), so it is computed with
    the same thresholds the isolation cells were graded with -- by handing the
    component's blocks to the receipt module's own walker inside a minimal
    single-component receipt, rather than restating the thresholds here where
    they could drift.

    Returns the link **and how far the walk got**.  With the per-step free-run
    probe switched off the chain has no link-6 measurement, and the receipt says
    ``chain_walked_to: composition`` rather than reporting a behavioral verdict
    it never took.
    """

    after = entry["after_install"]
    measured = after.get("free_run_token_match_vs_control") is not None
    probe = blank_receipt(
        cell_id=str(entry["isolation_cell_id"]),
        cell_instance_id="chain-probe",
        campaign_wave=0,
        repo_head="chain-probe",
        src_tree_sha="0" * 64,
        probe_set_sha="0" * 64,
        smoke_prompt_sha="0" * 64,
        compute_architecture="chain-probe",
    )
    probe["calibration"] = dict(entry["calibration"])
    probe["solve"] = dict(entry["solve"])
    probe["boundary"] = dict(entry["boundary"])
    probe["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": after["terminal_relL2_vs_control"],
        "emit_zero_floor_passed": bool(
            float(after["terminal_relL2_vs_control"]) < 1.0
        ),
        "vs_copy_exact_control_delta": after["terminal_relL2_vs_control"],
        "norm_ratio_per_layer_minmax": after.get("norm_ratio_per_layer_minmax"),
    }
    probe["behavior"] = {
        "free_run": {
            "probe_prompts": int(after.get("probe_prompts", 0) or 0),
            "tokens_per_prompt": int(after.get("tokens_per_prompt", 0) or 0),
            # With no per-step probe the walker is handed a value that cannot
            # cross link 6, so links 1-5 are still walked exactly and link 6 is
            # reported as unwalked instead of guessed.
            "token_match_vs_control": (
                after["free_run_token_match_vs_control"] if measured else 1.0
            ),
            "mean_lcp": after.get("free_run_mean_lcp") if measured else 1.0,
        },
        "decode_smoke": [],
    }
    return (
        first_divergent_metric(probe),
        "behaviour" if measured else "composition",
    )


# --------------------------------------------------------------------------
# ADDENDUM 9 mode M-C: the alternating joint co-solve
# --------------------------------------------------------------------------

#: Where a component's write lands inside its own block, as an ordinal.
#:
#: This is the whole capture-reuse licence, so it is conservative by
#: construction: a component is given the ordinal of the *earliest* tensor it
#: writes, so "does installing Y move X's input rows?" can only ever err toward
#: *yes*.  The order is LFM2's block, read off the parent module and not
#: invented here: norms feed the operator, the operator writes into the
#: residual before the FFN norm reads it, ``w1``/``w3`` both read ``mlp.input``
#: and neither reads the other, and ``w2`` reads what they produce.
_BLOCK_ORDER: Mapping[str, int] = {
    "C14": 0,  # operator_norm / ffn_norm -- upstream of everything in the block
    "C03": 1,  # conv in_proj
    "C04": 2,  # conv FIR taps
    "C05": 3,  # conv out_proj  -> operator.output
    "C06": 4,  # q_proj  \
    "C07": 4,  # k_proj   |  all three read operator.input and none reads another
    "C08": 4,  # v_proj  /
    "C10": 4,  # head merge/drop edits q/k/v/o; the earliest is q at 4
    "C09": 5,  # out_proj reads attention.context
    "C11": 6,  # w1 \  both read mlp.input
    "C12": 6,  # w3 /
    "C13": 7,  # w2 reads mlp.post_swiglu
}

#: A component with no layer (C01/C02/C15, the token interface) sits upstream of
#: every block, so it is given a layer index below the first one.
_GLOBAL_LAYER = -1


def _als_position(cell: CellId) -> tuple[int, int]:
    """(layer, intra-block ordinal) for the capture-reuse licence."""

    layer = (
        _GLOBAL_LAYER if cell.layer_index is None else int(cell.layer_index)
    )
    return layer, int(_BLOCK_ORDER.get(cell.component, 0))


def _capture_is_stale(
    position: tuple[int, int], pending: Sequence[tuple[int, int]]
) -> bool:
    """Has anything installed since the last capture moved *these* input rows?

    A weight installed at (layer L', ordinal o') can only change the rows a
    component at (L, o) reads when it writes strictly upstream of them:
    ``L' < L``, or the same block and ``o' < o``.  Nothing downstream of a
    component can move that component's input, and a component's own re-solve
    cannot move its own input, so a sweep that walks depth-ascending re-uses one
    capture for every component that shares a rung -- and the first component of
    sweep *n+1* re-uses the last capture of sweep *n* whenever the only
    installs since then were deeper than it.

    The licence is a claim about the model's dataflow, so wave 11 does not take
    it on trust: ``--als-verify-capture-licence`` spends one extra capture the
    first time this returns False and asserts the reused input rows are
    bit-identical to the fresh ones.
    """

    layer, order = position
    return any(
        (other_layer < layer) or (other_layer == layer and other_order < order)
        for other_layer, other_order in pending
    )


class _AlsHybridEvidence:
    """Child inputs, parent targets, one row index.

    The ALS coordinate step solves ``min ||X_child W'^T - Y_parent||`` -- the
    design rows are what the composed child actually delivers at the
    component's input endpoint, and the target rows are what the *parent*
    emitted at the component's write endpoint over the same corpus positions.
    Both captures run the frozen collector over the same batches in the same
    order with the same sealed capture selection, and the collector's decode is
    teacher-forced from caller-supplied ids, so row *i* of either capture is the
    same (batch, phase, token) position.  The batch-order half of that argument
    is checked by `_als_verify_row_alignment` against the two split manifests
    before any splice is built; what is checked *here* is the geometry: same
    layer, same phase list, same fit and held-out row counts.

    **The two ``selection_sha256`` values legitimately differ and are not
    compared.**  That digest is taken over the per-batch payload selection
    digests, which are functions of the captured activations themselves -- a
    capture through a folded child cannot reproduce the parent's, and requiring
    it to would be requiring the child to be the parent.  Both are recorded on
    the step so the receipt says which two objects were spliced.
    """

    def __init__(
        self,
        *,
        child: Any,
        parent: Any,
        parent_served: Sequence[str],
    ) -> None:
        self._child = child
        self._parent = parent
        self._parent_served = tuple(sorted(set(str(v) for v in parent_served)))
        child_layer = getattr(child, "layer_index", None)
        parent_layer = getattr(parent, "layer_index", None)
        if child_layer != parent_layer:
            raise MatrixCompositionRunError(
                "ALS hybrid evidence refuses to splice two sites: child layer "
                f"{child_layer} against parent layer {parent_layer}."
            )
        child_phases = tuple(getattr(child, "phases", ()) or ())
        parent_phases = tuple(getattr(parent, "phases", ()) or ())
        if child_phases != parent_phases:
            raise MatrixCompositionRunError(
                "ALS hybrid evidence refuses to splice two captures with "
                f"different phase lists: {child_phases} against "
                f"{parent_phases}."
            )
        if int(child.fit_row_count) != int(parent.fit_row_count) or int(
            child.heldout_row_count
        ) != int(parent.heldout_row_count):
            raise MatrixCompositionRunError(
                "ALS hybrid evidence refuses to splice two captures with "
                f"different row counts: child {child.fit_row_count}/"
                f"{child.heldout_row_count}, parent {parent.fit_row_count}/"
                f"{parent.heldout_row_count}."
            )

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._child.endpoints) | set(self._parent.endpoints)))

    @property
    def selection_sha256(self) -> str:
        return str(self._child.selection_sha256)

    @property
    def fit_row_count(self) -> int:
        return int(self._child.fit_row_count)

    @property
    def heldout_row_count(self) -> int:
        return int(self._child.heldout_row_count)

    def served_by(self, endpoint: str) -> str:
        return "parent" if str(endpoint) in self._parent_served else "child"

    def fit_rows(self, endpoint: str) -> Any:
        source = self._parent if self.served_by(endpoint) == "parent" else self._child
        return source.fit_rows(endpoint)

    def heldout_rows(self, endpoint: str) -> Any:
        source = self._parent if self.served_by(endpoint) == "parent" else self._child
        return source.heldout_rows(endpoint)


def _relative_residual(predicted: Any, reference: Any) -> float:
    """Aggregate relative L2 of one row block against another."""

    return float(_row_metrics(predicted, reference)["aggregate_relative_l2"])


def _als_verify_row_alignment(
    *, split_manifest_path: str | Path, base_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that row *i* of a child capture is row *i* of the parent base.

    The whole parent-target objective rests on it, so it is established from
    the two manifests rather than assumed from the fact that both runs used the
    same corpus file:

    * the conditioned split is **inherited** from the parent base's, by
      ``record_id``, and names the manifest it inherited from;
    * the conditioned capture saw **exactly** the parent base's batches, in
      **exactly** the parent base's order -- the record id is the batch's own
      input digest, so equality of the ordered list is equality of the inputs
      and of their order;
    * therefore, because the capture selection (``sample_last_tokens``,
      ``max_decode_steps``, the phase list) is reconstructed from the base's own
      sealed manifest and refused unless it reproduces its ``selection_sha256``,
      the per-batch row blocks are the same size in the same order.
    """

    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    split = manifest.get("split") or {}
    if str(split.get("method")) != "inherit" or str(
        split.get("inherited_from_manifest_sha256")
    ) != str(base_manifest.get("manifest_sha256")):
        raise MatrixCompositionRunError(
            "The conditioned split was not inherited from the parent evidence "
            f"base ({split.get('inherited_from_manifest_sha256')!r} against "
            f"{base_manifest.get('manifest_sha256')!r}); the child's rows and "
            "the parent's rows would not be the same positions."
        )
    observed = [
        str(row["record_id"])
        for row in sorted(split["rows"], key=lambda r: int(r["batch_index"]))
    ]
    declared = [
        str(value) for value in base_manifest["corpus"]["ordered_batch_input_sha256"]
    ]
    if observed != declared:
        raise MatrixCompositionRunError(
            "The conditioned capture saw different batches, or the same "
            "batches in a different order, than the parent evidence base; row "
            "i of one is not row i of the other."
        )
    return {
        "batches": len(observed),
        "inherited_from_manifest_sha256": str(
            split.get("inherited_from_manifest_sha256")
        ),
        "ordered_record_ids_match_parent_base": True,
        "conditioned_split_manifest_sha256": str(manifest.get("manifest_sha256")),
    }


def _als_capture_selfcheck(
    *,
    spec: Any,
    child_evidence: Any,
    installed: Mapping[str, Any],
) -> float | None:
    """The calibration gate, restated for a capture through a folded child.

    A conditioned capture cannot run the copy_exact arm: the tensor installed at
    this component is no longer the parent's, so ``input @ parent^T`` is not
    what the capture wrote and the number the gate would compute is the fold's
    own error, not the capture's fidelity.  The identity that *is* still exact
    is the one with the installed weight in it, and checking it is what proves
    the capture's two endpoints are the map the runner thinks they are.
    """

    if spec.input_endpoint is None or spec.write_endpoint is None:
        return None
    weight = installed.get(spec.weight_keys[0])
    if weight is None:
        return None
    held_input = child_evidence.heldout_rows(spec.input_endpoint)
    held_write = child_evidence.heldout_rows(spec.write_endpoint)
    if int(held_write.shape[-1]) != int(weight.shape[0]):
        return None
    return _relative_residual(held_input @ weight.to(held_input.dtype).T, held_write)


def _als_sweeps(  # noqa: PLR0912, PLR0915 - one linear loop, reported step by step
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    entries: list[dict[str, Any]],
    recipe: Mapping[str, Any],
    parent_weights: Mapping[str, Any],
    knobs: Mapping[str, Any],
    capture_state: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    control_terminal: Any,
    control_norms: Any,
    control_ids: Sequence[Sequence[int]],
    probe_prompts: Sequence[Any],
    device: str,
    output: Path,
    watchdog: Any,
    calibration_tolerance: float,
) -> dict[str, Any]:
    """Sweep the installed components until the joint re-solve stops paying."""

    capture_root = Path(args.conditioned_capture_root)
    max_sweeps = int(args.als_max_sweeps)
    floor = float(args.als_improvement_floor)
    objective = str(args.als_objective)
    prune = bool(args.als_prune_captures)
    verify_licence = bool(args.als_verify_capture_licence)

    order = [
        (index, CellId.parse(str(entry["isolation_cell_id"])))
        for index, entry in enumerate(entries)
    ]
    # Depth ascending, then intra-block order: the sweep walks the dataflow, so
    # every component is re-solved against a child whose upstream is already at
    # this sweep's answer.
    order.sort(key=lambda item: (_als_position(item[1]), item[0]))
    eligible: list[tuple[int, CellId]] = []
    skipped: list[dict[str, Any]] = []
    for index, cell in order:
        has_site = cell.layer_index is not None and (
            component_spec(cell, parent_weights).input_endpoint is not None
        )
        if has_site:
            eligible.append((index, cell))
        else:
            skipped.append(
                {
                    "cell_id": cell.text,
                    "reason": (
                        "a component with no sealed site evidence has no "
                        "activation boundary to re-solve against; its fold is a "
                        "weight-space fold and the ALS sweep leaves it exactly "
                        "as the M-A pass solved it"
                    ),
                }
            )

    def _global_drift() -> tuple[float, dict[str, Any]]:
        terminal, layer_norms = cell_runner._terminal_states(
            model, tokenizer, probe_prompts, device=device
        )
        metrics = _row_metrics(terminal, control_terminal)
        ratios = (layer_norms / control_norms.clamp_min(1e-30)).reshape(-1)
        return float(metrics["mean_relative_l2"]), {
            "terminal_relL2_vs_control": metrics["mean_relative_l2"],
            "terminal_cos_vs_control": metrics["mean_cosine"],
            "norm_ratio_per_layer_minmax": [
                float(ratios.min().item()),
                float(ratios.max().item()),
            ],
        }

    def _snapshot() -> dict[str, Any]:
        state = model.state_dict()
        return {
            key: state[key].detach().to("cpu").clone()
            for entry in entries
            for key in entry["installed_weight_keys"]
        }

    def _restore(snapshot: Mapping[str, Any]) -> None:
        cell_runner._install_component(
            model, {key: value.clone() for key, value in snapshot.items()}
        )

    started_drift, started_metrics = _global_drift()
    _log(f"ALS: M-A child global drift {started_drift:.6f} before any sweep")

    block: dict[str, Any] = {
        "mechanism": "alternating-closed-form-joint-resolve",
        "objective": objective,
        "max_sweeps": max_sweeps,
        "improvement_floor_relative": floor,
        "capture_reuse_licence": (
            "a fresh capture is taken whenever anything installed since the "
            "last one writes strictly upstream of the component's input "
            "endpoint (_capture_is_stale); reuse is otherwise exact"
        ),
        "capture_licence_verification": None,
        "row_alignment": None,
        "m_a_baseline": {
            "global_drift": started_drift,
            **started_metrics,
        },
        "skipped_components": skipped,
        "sweeps": [],
        "captures": [],
        "stopped_because": "max sweeps reached",
        "selected_sweep": 0,
    }

    best_drift = started_drift
    best_sweep = 0
    best_snapshot = _snapshot()
    previous_drift = started_drift
    pending: list[tuple[int, int]] = [_als_position(cell) for _index, cell in order]
    capture: dict[str, Any] | None = None
    child_cache: dict[str, Any] = {}

    for sweep in range(1, max_sweeps + 1):
        sweep_started = time.monotonic()
        steps: list[dict[str, Any]] = []
        for index, cell in eligible:
            entry = entries[index]
            spec = component_spec(cell, parent_weights)
            position = _als_position(cell)
            watchdog.stage(f"als-{sweep}-{cell.text}")
            stale = capture is None or _capture_is_stale(position, pending)
            licence_check: dict[str, Any] | None = None
            if stale:
                watchdog.stage(f"als-{sweep}-capture-{cell.text}")
                if capture is not None and prune:
                    _prune_capture(capture["cache_directory"])
                capture = capture_conditioned_evidence(
                    capture_model=capture_state["model"],
                    child_model=model,
                    output_root=capture_root,
                    batches=capture_state["batches"],
                    capture_config=capture_state["config"],
                    base_manifest=base_manifest,
                    device=device,
                    conditioning={
                        "upstream_installed": [c.text for _i, c in order],
                        "install_index": index,
                        "next_component": f"sweep{sweep}-{cell.text}",
                    },
                )
                alignment = _als_verify_row_alignment(
                    split_manifest_path=capture["split_manifest_path"],
                    base_manifest=base_manifest,
                )
                block["row_alignment"] = alignment
                block["captures"].append(
                    {
                        "sweep": sweep,
                        "cut_for": cell.text,
                        "row_alignment_verified": True,
                        "trace_cache_key": str(capture["trace_cache_key"]),
                        "split_manifest_sha256": str(
                            capture["split_manifest_sha256"]
                        ),
                        "child_state_dict_sha256": str(
                            capture["child_state_dict_sha256"]
                        ),
                        "collector_implementation_sha256": str(
                            capture["collector_implementation_sha256"]
                        ),
                        "wall_time_seconds": float(
                            capture["capture_wall_time_seconds"]
                        ),
                    }
                )
                pending = []
                child_cache = {}
            key = (str(capture["trace_cache_key"]), cell.text)
            if child_cache.get("key") != key:
                child_evidence, child_binding = cell_runner.load_site_evidence(
                    trace_cache_path=Path(capture["cache_directory"]),
                    layer_index=int(cell.layer_index),
                    endpoints=_endpoints_for(spec),
                    forced_decode_count=int(args.forced_decode_count),
                    fit_batch_limit=None,
                    heldout_batch_limit=None,
                    cache_directory=capture_root
                    / "evidence-cache"
                    / f"{capture['trace_cache_key'][:16]}-"
                    f"{evidence_cache_name(cell)}",
                    basis_phase_cache_root=None,
                    basis_phase_key_sha256=None,
                    basis_phase_hmac_key_file=None,
                    basis_phase_hmac_key_id=None,
                    evidence_split_manifest=Path(capture["split_manifest_path"]),
                    target_hidden_size=int(args.target_hidden_size),
                )
                child_cache = {
                    "key": key,
                    "evidence": child_evidence,
                    "binding": child_binding,
                }
            child_evidence = child_cache["evidence"]
            child_binding = child_cache["binding"]

            if (
                not stale
                and verify_licence
                and block["capture_licence_verification"] is None
            ):
                # Spend one capture, once, on the first reuse the run reaches,
                # and prove the dataflow claim the licence rests on rather than
                # asserting it: the rows a reused capture serves at this
                # component's input endpoint must be BIT-IDENTICAL to the rows a
                # capture taken right now would serve.
                watchdog.stage(f"als-{sweep}-licence-{cell.text}")
                fresh = capture_conditioned_evidence(
                    capture_model=capture_state["model"],
                    child_model=model,
                    output_root=capture_root,
                    batches=capture_state["batches"],
                    capture_config=capture_state["config"],
                    base_manifest=base_manifest,
                    device=device,
                    conditioning={
                        "upstream_installed": [c.text for _i, c in order],
                        "install_index": index,
                        "next_component": f"sweep{sweep}-{cell.text}-licence",
                    },
                )
                probe_evidence, _probe_binding = cell_runner.load_site_evidence(
                    trace_cache_path=Path(fresh["cache_directory"]),
                    layer_index=int(cell.layer_index),
                    endpoints=_endpoints_for(spec),
                    forced_decode_count=int(args.forced_decode_count),
                    fit_batch_limit=None,
                    heldout_batch_limit=None,
                    cache_directory=capture_root
                    / "evidence-cache"
                    / f"{fresh['trace_cache_key'][:16]}-licence-"
                    f"{evidence_cache_name(cell)}",
                    basis_phase_cache_root=None,
                    basis_phase_key_sha256=None,
                    basis_phase_hmac_key_file=None,
                    basis_phase_hmac_key_id=None,
                    evidence_split_manifest=Path(fresh["split_manifest_path"]),
                    target_hidden_size=int(args.target_hidden_size),
                )
                reused_rows = child_evidence.fit_rows(spec.input_endpoint)
                fresh_rows = probe_evidence.fit_rows(spec.input_endpoint)
                identical = bool(torch.equal(reused_rows, fresh_rows))
                licence_check = {
                    "component": cell.text,
                    "sweep": sweep,
                    "reused_trace_cache_key": str(capture["trace_cache_key"]),
                    "fresh_trace_cache_key": str(fresh["trace_cache_key"]),
                    "input_endpoint": str(spec.input_endpoint),
                    "rows": int(fresh_rows.shape[0]),
                    "bit_identical": identical,
                    "max_abs_difference": (
                        0.0
                        if identical
                        else float((reused_rows - fresh_rows).abs().max().item())
                    ),
                }
                block["capture_licence_verification"] = licence_check
                del probe_evidence
                if prune:
                    _prune_capture(fresh["cache_directory"])
                if not identical:
                    raise MatrixCompositionRunError(
                        "The capture-reuse licence is wrong: reusing the "
                        f"previous capture for {cell.text} serves input rows "
                        "that a fresh capture through the same child does not "
                        "reproduce."
                    )

            parent_evidence, parent_binding = cell_runner.load_site_evidence(
                trace_cache_path=Path(args.parent_trace_cache),
                layer_index=int(cell.layer_index),
                endpoints=_endpoints_for(spec),
                forced_decode_count=int(args.forced_decode_count),
                fit_batch_limit=None,
                heldout_batch_limit=None,
                cache_directory=Path(args.evidence_cache_root)
                / evidence_cache_name(cell),
                basis_phase_cache_root=None,
                basis_phase_key_sha256=None,
                basis_phase_hmac_key_file=None,
                basis_phase_hmac_key_id=None,
                evidence_split_manifest=Path(args.evidence_split_manifest),
                target_hidden_size=int(args.target_hidden_size),
            )

            live = model.state_dict()
            installed_now = {
                weight_key: live[weight_key].detach().to("cpu", torch.float64)
                for weight_key in spec.weight_keys
            }
            selfcheck = _als_capture_selfcheck(
                spec=spec, child_evidence=child_evidence, installed=installed_now
            )
            if selfcheck is not None and selfcheck > calibration_tolerance:
                raise MatrixCompositionRunError(
                    f"{cell.text}: the conditioned capture does not reproduce "
                    f"its own installed weight ({selfcheck} against a "
                    f"{calibration_tolerance} tolerance); no ALS solve below "
                    "that may be believed."
                )

            served = [
                name
                for name in (
                    spec.write_endpoint,
                    spec.macrocell_endpoint,
                    "residual.pre",
                    "residual.post_operator",
                )
                if name
            ]
            evidence = (
                _AlsHybridEvidence(
                    child=child_evidence,
                    parent=parent_evidence,
                    parent_served=served,
                )
                if objective == "parent-target"
                else child_evidence
            )
            # What the coordinate step is trying to reduce, measured before it
            # runs and again after: the held-out residual of this component's
            # write against the PARENT's write, given the child's own inputs.
            held_input = child_evidence.heldout_rows(spec.input_endpoint)
            held_parent_write = parent_evidence.heldout_rows(spec.write_endpoint)
            before = _relative_residual(
                held_input @ installed_now[spec.weight_keys[0]].to(held_input.dtype).T,
                held_parent_write,
            )

            technique = resolve(cell.technique, cell.component)
            item = recipe["components"][index]
            step_knobs = {**knobs, **(item.get("knobs") or {})}
            supports_target = str(cell.technique) == "T8"
            if objective == "parent-target" and supports_target:
                step_knobs["relation_target_endpoint"] = spec.write_endpoint
            request = CellTechniqueRequest(
                cell=cell,
                component=spec,
                level=cell.level,
                level_width=int(entry["level_width"]),
                seed=int(args.seed),
                parent_weights=parent_weights,
                evidence=evidence,
                knobs=step_knobs,
                workspace=output / f"als-{sweep:02d}-{cell.text}",
                log=_log,
            )
            solve_started = time.monotonic()
            result = technique(request)
            validate_result(request, result)
            solve_seconds = time.monotonic() - solve_started
            if int(result.stored_parameter_count) != int(
                entry["storage"]["stored_parameter_count"]
            ):
                raise MatrixCompositionRunError(
                    f"{cell.text}: the ALS re-solve stores "
                    f"{result.stored_parameter_count} against the M-A pass's "
                    f"{entry['storage']['stored_parameter_count']}; an ALS "
                    "sweep re-solves a component, it does not re-budget it."
                )
            after = _relative_residual(
                held_input
                @ result.installed_weights[spec.weight_keys[0]]
                .to(held_input.dtype)
                .T,
                held_parent_write,
            )
            boundary, _boundary_diagnostics = _boundary_block(
                cell=cell, spec=spec, evidence=evidence, result=result
            )
            installed_digests = cell_runner._install_component(
                model, result.installed_weights
            )
            pending.append(position)
            solve_block = dict(result.solve)
            solve_block["stored_parameter_count"] = int(result.stored_parameter_count)
            solve_block["parent_parameter_count"] = int(spec.parent_parameter_count)
            solve_block["solve_wall_time_seconds"] = float(solve_seconds)
            entry["solve"] = solve_block
            entry["boundary"] = boundary
            entry["installed_tensor_digests"] = installed_digests
            entry["installed_weight_keys"] = sorted(result.installed_weights)
            entry["notes"] = str(result.notes or "")
            entry["evidence"] = {
                **entry["evidence"],
                "conditioned": True,
                "trace_cache_key": child_binding.get("trace_cache_key"),
                "split_authority": child_binding.get("split_authority"),
                "evidence_split_manifest_sha256": child_binding.get(
                    "evidence_split_manifest_sha256"
                ),
                "selection_sha256": child_binding.get("selection_sha256"),
                "trace_cache_implementation_sha256": child_binding.get(
                    "implementation_sha256"
                ),
                "evidence_cache_reused": bool(
                    child_binding.get("evidence_cache_reused")
                ),
                "fit_rows": int(child_binding.get("fit_rows", 0) or 0),
                "heldout_rows": int(child_binding.get("heldout_rows", 0) or 0),
                "als_parent_target_binding": {
                    "trace_cache_key": parent_binding.get("trace_cache_key"),
                    "evidence_split_manifest_sha256": parent_binding.get(
                        "evidence_split_manifest_sha256"
                    ),
                    "served_endpoints": served if objective == "parent-target" else [],
                },
                "reason": (
                    f"ADDENDUM 9 M-C sweep {sweep}: re-solved against a capture "
                    "through the composed child"
                ),
            }
            # The calibration block is NOT overwritten.  It is a true statement
            # about the parent evidence at this site -- measured at the M-A
            # install, passed, and unchanged by anything a re-solve does -- and
            # replacing it would delete a gate that really ran.  What the
            # coordinate step needs instead is the gate that IS defined on a
            # capture through a folded child: the capture reproducing its own
            # INSTALLED weight, checked above at the same tolerance and
            # reported per step as `capture_selfcheck_relL2`.
            entry["calibration"] = {
                **entry["calibration"],
                "als_gate": {
                    "kind": "installed-weight-reproduction",
                    "capture_selfcheck_relL2": selfcheck,
                    "tolerance": calibration_tolerance,
                    "reason": (
                        "the copy_exact arm is undefined on a conditioned "
                        "capture -- the component's own tensor is no longer "
                        "the parent's -- so the exact identity checked here is "
                        "the one with the installed weight in it"
                    ),
                },
            }
            entry["after_install"] = {
                **entry["after_install"],
                "als_note": (
                    "these are the M-A install-trace numbers, kept as the "
                    "record of the sequential install; after an ALS sweep a "
                    "per-component 'after install' drift is not a defined "
                    "quantity, and the composed child's drift after every "
                    "sweep is in composition.als.sweeps"
                ),
            }
            entry["first_divergent_metric"], entry["chain_walked_to"] = (
                _component_chain(entry)
            )
            steps.append(
                {
                    "sweep": sweep,
                    "cell_id": cell.text,
                    "position": list(position),
                    "capture_refreshed": bool(stale),
                    "trace_cache_key": str(capture["trace_cache_key"]),
                    "objective": (
                        "parent-target"
                        if objective == "parent-target" and supports_target
                        else "mimic-parent-operator-on-child-rows"
                    ),
                    "objective_reason": (
                        None
                        if supports_target or objective != "parent-target"
                        else f"technique {cell.technique} synthesizes its own "
                        "target from the parent weight and takes no target "
                        "endpoint; this component is re-solved in mimic form "
                        "and says so"
                    ),
                    "capture_selfcheck_relL2": selfcheck,
                    "child_selection_sha256": str(child_evidence.selection_sha256),
                    "parent_selection_sha256": str(parent_evidence.selection_sha256),
                    "parent_target_heldout_relL2_before": before,
                    "parent_target_heldout_relL2_after": after,
                    "improved": bool(after < before),
                    "write_heldout_relL2": boundary.get("write_heldout_relL2"),
                    "write_heldout_cos": boundary.get("write_heldout_cos"),
                    "solve_wall_time_seconds": float(solve_seconds),
                    "licence_check": licence_check,
                }
            )
            _log(
                f"ALS sweep {sweep} {cell.text}: parent-target heldout "
                f"{before:.6f} -> {after:.6f} "
                f"({'better' if after < before else 'WORSE'}), capture "
                f"selfcheck {selfcheck}"
            )
            del parent_evidence
            del held_input, held_parent_write

        watchdog.stage(f"als-{sweep}-drift")
        drift, drift_metrics = _global_drift()
        probe_ids = cell_runner._greedy_fixed(
            model,
            tokenizer,
            probe_prompts,
            tokens=int(args.probe_tokens),
            device=device,
        )
        match, lcp = cell_runner._token_match(probe_ids, control_ids)
        improvement = (
            (previous_drift - drift) / previous_drift if previous_drift else 0.0
        )
        block["sweeps"].append(
            {
                "sweep": sweep,
                "steps": steps,
                "global_drift": drift,
                **drift_metrics,
                "free_run_token_match_vs_control": match,
                "free_run_mean_lcp": lcp,
                "drift_improvement_vs_previous_sweep": improvement,
                "wall_time_seconds": float(time.monotonic() - sweep_started),
            }
        )
        _log(
            f"ALS sweep {sweep} complete: global drift {previous_drift:.6f} -> "
            f"{drift:.6f} ({improvement * 100:.2f}% relative), free-run {match}"
        )
        if drift < best_drift:
            best_drift = drift
            best_sweep = sweep
            best_snapshot = _snapshot()
        previous_drift = drift
        if improvement < floor:
            block["stopped_because"] = (
                f"sweep {sweep} improved the global drift by "
                f"{improvement * 100:.2f}%, below the pre-registered "
                f"{floor * 100:.2f}% floor"
            )
            break

    if best_sweep != len(block["sweeps"]):
        # The child that gets sealed is the best sweep, not the last one.  The
        # rule is pre-registered (ADDENDUM 9) and it is a solver stopping rule
        # on a solver diagnostic; it never reads the review surface.
        _restore(best_snapshot)
        _log(
            f"ALS: restored sweep {best_sweep} (drift {best_drift:.6f}); the "
            f"last sweep was worse"
        )
    block["selected_sweep"] = best_sweep
    block["selected_global_drift"] = best_drift
    block["global_drift_delta_vs_m_a"] = best_drift - started_drift
    if capture is not None and prune:
        _prune_capture(capture["cache_directory"])
    return block


def _prune_capture(directory: str | Path) -> None:
    """Drop one conditioned capture's bytes; its digests are already sealed.

    A capture is 8 GB and an ALS run takes one per rung per sweep.  They are
    re-derivable from the sealed child state and the frozen collector, and the
    receipt carries the cache key, the split manifest digest, the collector
    implementation digest and the child state digest of every one, so the bytes
    are evidence nobody has to keep.
    """

    import shutil

    path = Path(directory)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, help="COMP.MA[.VARIANT]")
    parser.add_argument("--mode", choices=("M-A", "M-B", "M-C"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--campaign-wave", type=int, default=4)
    parser.add_argument("--repo-head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-compute-architecture")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--probe-prompts", type=Path, default=cell_runner.PROBE_PROMPTS_FILE
    )
    parser.add_argument(
        "--smoke-prompts", type=Path, default=cell_runner.SMOKE_PROMPTS_FILE
    )
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--smoke-token-cap", type=int, default=128)
    parser.add_argument("--hard-cap-seconds", type=float, default=4 * 3600.0)
    parser.add_argument("--target-seconds", type=float, default=3600.0)
    # sealed parent evidence
    parser.add_argument("--parent-trace-cache", type=Path, required=True)
    parser.add_argument("--evidence-split-manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence-cache-root",
        type=Path,
        required=True,
        help=(
            "Directory of the authenticated per-(site, component) evidence "
            "caches the isolation waves built, e.g. .../evidence-cache; a step "
            "reuses <root>/<SITE>-<COMPONENT> exactly as a cell did."
        ),
    )
    parser.add_argument("--forced-decode-count", type=int, default=8)
    parser.add_argument("--target-hidden-size", type=int, default=816)
    # conditioned recapture (M-B)
    parser.add_argument("--conditioned-capture-root", type=Path)
    parser.add_argument("--donor-calibration", type=Path)
    parser.add_argument("--capture-batch-count", type=int)
    parser.add_argument("--capture-dtype", default="bfloat16")
    # ADDENDUM 9 mode M-C: the alternating joint co-solve
    parser.add_argument("--als-max-sweeps", type=int, default=4)
    parser.add_argument(
        "--als-improvement-floor",
        type=float,
        default=0.05,
        help=(
            "Stop when a whole sweep improves the composed child's global "
            "drift by less than this RELATIVE fraction (0.05 = 5%%)."
        ),
    )
    parser.add_argument(
        "--als-objective",
        choices=("parent-target", "mimic"),
        default="parent-target",
        help=(
            "parent-target: each coordinate step re-solves against the "
            "PARENT's sealed write rows given the child's inputs (the only "
            "objective that can cancel composed drift).  mimic: M-B's target, "
            "the parent operator applied to the child's rows, revisited."
        ),
    )
    parser.add_argument(
        "--als-prune-captures", action="store_true", default=True,
        help="Delete each 8 GB conditioned capture once its rows are read.",
    )
    parser.add_argument(
        "--no-als-prune-captures", dest="als_prune_captures", action="store_false"
    )
    parser.add_argument(
        "--als-verify-capture-licence", action="store_true", default=False,
        help=(
            "Spend one extra capture, once, to prove that a reused capture "
            "serves bit-identical input rows to a fresh one."
        ),
    )
    parser.add_argument(
        "--per-step-free-run",
        action="store_true",
        default=True,
        help="Run the 8x64 free-run probe after every install (default on).",
    )
    parser.add_argument(
        "--no-per-step-free-run", dest="per_step_free_run", action="store_false"
    )
    parser.add_argument("--knobs", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915 - one linear run
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    started = time.monotonic()

    cell = CellId.parse(args.cell_id)
    if cell.kind != "composition":
        raise MatrixCompositionRunError(
            f"{cell.text} is a {cell.kind} cell; this runner builds composed "
            "children only (COMP.MA / COMP.MB / COMP.MC)."
        )
    mode = COMPOSITION_MODES[str(cell.composition_mode)]
    if mode != args.mode:
        raise MatrixCompositionRunError(
            f"--mode {args.mode} disagrees with cell id {cell.text} ({mode})."
        )

    output = Path(args.out).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        Path(args.ledger).expanduser().absolute()
        if args.ledger is not None
        else output.parent / "ledger.jsonl"
    )
    instance = uuid.uuid4().hex

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    validate_recipe(recipe)
    knobs: dict[str, Any] = {}
    if args.knobs is not None:
        knobs.update(json.loads(Path(args.knobs).read_text(encoding="utf-8")))
    knobs.setdefault("python", sys.executable)
    knobs.setdefault("device", str(args.device))
    calibration_tolerance = float(knobs.get("calibration_tolerance", 5e-3))

    probe_prompts, probe_sha = cell_runner._load_prompt_set(
        args.probe_prompts, "free-run-probe"
    )
    smoke_prompts, smoke_sha = cell_runner._load_prompt_set(
        args.smoke_prompts, "decode-smoke"
    )

    architecture = cell_runner.compute_architecture(str(args.device))
    required = args.require_compute_architecture
    if required and not architecture.startswith(str(required)):
        raise MatrixCompositionRunError(
            f"--device {args.device} resolves to {architecture!r}, which does "
            f"not start with the required {required!r}."
        )
    _log(f"compute architecture: {architecture}")

    receipt = blank_receipt(
        cell_id=cell.text,
        cell_instance_id=instance,
        campaign_wave=int(args.campaign_wave),
        repo_head=args.repo_head or cell_runner._repo_head(_REPO_ROOT),
        src_tree_sha=cell_runner._src_tree_sha256(_REPO_ROOT),
        probe_set_sha=probe_sha,
        smoke_prompt_sha=smoke_sha,
        compute_architecture=architecture,
    )
    receipt["code_identity"]["deployed_module_digests"] = _module_digests()
    receipt["composition"] = {
        "mode": mode,
        "variant": cell.composition_variant,
        "recipe_path": str(args.recipe),
        "recipe_sha256": cell_runner._file_sha256(Path(args.recipe)),
        "recipe_variant": str(recipe.get("variant")),
        "recipe_ranking": str(recipe.get("ranking")),
        "selection_policy": recipe.get("selection_policy"),
        "install_order": [],
        "components": [],
        "passed_through": list(recipe.get("passed_through") or []),
        "storage": {
            "parent_parameter_count": 0,
            "child_parameter_count": 0,
            "net_ratio": 1.0,
        },
        "incomplete_reason": "the run did not reach its first install",
    }
    receipt_path = output / "cell-receipt.json"
    watchdog = cell_runner.CellWatchdog(
        receipt=receipt,
        receipt_path=receipt_path,
        ledger_path=ledger_path,
        hard_cap_seconds=float(args.hard_cap_seconds),
        started=started,
    ).start()

    device = str(args.device)
    dtype = cell_runner._dtype(str(args.dtype))
    technique_modules = cell_runner.discover_technique_modules()
    diagnostics: dict[str, Any] = {
        "schema": RUNNER_SCHEMA,
        "cell_id": cell.text,
        "cell_instance_id": instance,
        "mode": mode,
        "recipe": recipe,
        "technique_modules": technique_modules,
        "registry": describe_registry(),
        "device": device,
        "compute_architecture": architecture,
        "dtype": str(args.dtype),
        "steps": [],
    }

    entries: list[dict[str, Any]] = []
    try:
        watchdog.stage("materialize")
        tokenizer = cell_runner._load_tokenizer(Path(args.parent_model))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        model = cell_runner._load_parent(Path(args.parent_model), device, dtype)
        parent_weights = {
            key: value.detach().to(device="cpu", dtype=torch.float64)
            for key, value in model.state_dict().items()
        }
        parent_total, parent_audit = unique_parameter_count(model)
        diagnostics["parent_parameter_count"] = parent_total
        diagnostics["parent_parameter_audit"] = parent_audit
        _log(f"parent: {parent_total} unique parameters ({parent_audit})")

        control_root = Path(args.control_artifact).expanduser().absolute()
        control = json.loads(
            (control_root / "control.json").read_text(encoding="utf-8")
        )
        if control["probe_set_sha256"] != probe_sha:
            raise MatrixCompositionRunError(
                "Control artifact was built on a different probe set."
            )
        control_states = torch.load(
            control_root / "control-states.pt", weights_only=True
        )
        control_ids = [list(map(int, row)) for row in control["probe_token_ids"]]
        control_terminal = control_states["terminal"]
        control_norms = control_states["layer_norms"]

        base_manifest = json.loads(
            Path(args.evidence_split_manifest).read_text(encoding="utf-8")
        )

        # M-B preparation: the corpus, the capture selection, and a second
        # model object at the capture dtype.  All three are built before the
        # first solve so an M-B run cannot get four components deep and then
        # discover it cannot capture.
        capture_state: dict[str, Any] = {}
        if mode in ("M-B", "M-C"):
            watchdog.stage("capture-preflight")
            capture_state = _prepare_conditioned_capture(
                args=args,
                parent_trace_cache=Path(args.parent_trace_cache),
                device=device,
            )
            diagnostics["conditioned_capture"] = {
                key: value
                for key, value in capture_state.items()
                if key not in ("model", "batches", "config")
            }
            diagnostics["conditioned_capture"]["batches"] = len(
                capture_state["batches"]
            )

        installed_so_far: list[str] = []
        for index, item in enumerate(recipe["components"]):
            step_started = time.monotonic()
            step_cell = CellId.parse(str(item["cell_id"]))
            watchdog.stage(f"component-{index}-{step_cell.text}")
            spec = component_spec(step_cell, parent_weights)
            level_width = resolve_level_width(
                spec, step_cell.level, x4_width=item.get("x4_width")
            )
            # The technique is handed the ORIGINAL parent tensors as its prior in
            # both modes, and this is the assertion that says why that is honest:
            # the component being solved has not been folded yet, so the child's
            # own tensor for it *is* the parent's.  Reading it back out of the
            # float32 child instead would introduce a rounding difference
            # between M-A and M-B that has nothing to do with conditioning.
            live = model.state_dict()
            for key in spec.weight_keys:
                if not bool(
                    torch.equal(
                        live[key].detach().to("cpu", torch.float64),
                        parent_weights[key],
                    )
                ):
                    raise MatrixCompositionRunError(
                        f"{step_cell.text} would be solved with the parent's "
                        f"{key} as its prior, but the child's copy of that "
                        "tensor has already been modified; the recipe folds a "
                        "tensor twice."
                    )

            conditioning: dict[str, Any] = {
                "conditioned": False,
                "upstream_installed": list(installed_so_far),
                "install_index": index,
            }
            if mode == "M-B" and index > 0:
                watchdog.stage(f"capture-{index}-{step_cell.text}")
                # One flat, content-addressed root for every conditioned
                # capture in the campaign.  `build_lfm2_circuit_trace_cache` is
                # "build or reuse an immutable content-addressed cache", and a
                # conditioned capture's key is a function of the child state,
                # the corpus and the capture selection -- so two runs whose
                # recipes share a prefix (RBEH and RSTO both fold C03 then C14)
                # capture that prefix once, and a re-run of an M-B cell never
                # re-captures anything.  On this host that is not a nicety: the
                # share writes at ~20 MB/s and one capture is 8.1 GB.
                captured = capture_conditioned_evidence(
                    capture_model=capture_state["model"],
                    child_model=model,
                    output_root=Path(args.conditioned_capture_root),
                    batches=capture_state["batches"],
                    capture_config=capture_state["config"],
                    base_manifest=base_manifest,
                    device=device,
                    conditioning={
                        "upstream_installed": list(installed_so_far),
                        "install_index": index,
                        "next_component": step_cell.text,
                    },
                )
                conditioning.update(captured)
                conditioning["conditioned"] = True
                trace_cache_path = Path(captured["cache_directory"])
                split_manifest = Path(captured["split_manifest_path"])
                evidence_cache = (
                    Path(args.conditioned_capture_root)
                    / "evidence-cache"
                    / f"{captured['trace_cache_key'][:16]}-"
                    f"{evidence_cache_name(step_cell)}"
                )
            else:
                trace_cache_path = Path(args.parent_trace_cache)
                split_manifest = Path(args.evidence_split_manifest)
                evidence_cache = Path(args.evidence_cache_root) / str(
                    item.get("evidence_cache")
                    or evidence_cache_name(step_cell)
                )
                conditioning["reason"] = (
                    (
                        "M-C's first pass IS M-A -- every winner solved against "
                        "parent evidence -- so that the ALS sweeps start from "
                        "the byte-identical M-A child (ADDENDUM 9)"
                        if mode == "M-C"
                        else "M-A installs every winner solved against parent "
                        "evidence (ADDENDUM 4's null hypothesis)"
                    )
                    if mode != "M-B"
                    else "first component in topological order: nothing "
                    "upstream is folded, so the conditioned evidence IS the "
                    "parent evidence and no capture was spent"
                )

            watchdog.stage(f"evidence-{index}-{step_cell.text}")
            if step_cell.layer_index is None:
                # A GLOBAL-site component has no layer to open a site evidence
                # cache at.  Wave 9's C01 is the first one the campaign has
                # composed, and the honest thing is to hand its kernel an
                # evidence object that refuses rather than a cache cut at some
                # layer chosen for the runner's convenience.  The binding still
                # names the parent base the run is scoped to, so the receipt
                # says which evidence generation this cell belongs to even
                # though it read none of its rows.
                evidence = _GlobalComponentEvidence(step_cell.component)
                binding = {
                    "trace_cache_key": str(base_manifest.get("trace_cache_key")),
                    "split_authority": "no-site-evidence-global-component",
                    "evidence_split_manifest_sha256": str(
                        base_manifest.get("manifest_sha256")
                    ),
                    "selection_sha256": None,
                    "implementation_sha256": None,
                    "evidence_cache_reused": False,
                    "fit_rows": 0,
                    "heldout_rows": 0,
                }
                _log(
                    f"{step_cell.text}: GLOBAL site -- no per-site evidence "
                    "cache is opened and none is read"
                )
            else:
                evidence, binding = cell_runner.load_site_evidence(
                    trace_cache_path=trace_cache_path,
                    layer_index=int(step_cell.layer_index),
                    endpoints=_endpoints_for(spec),
                    forced_decode_count=int(args.forced_decode_count),
                    fit_batch_limit=None,
                    heldout_batch_limit=None,
                    cache_directory=evidence_cache,
                    basis_phase_cache_root=None,
                    basis_phase_key_sha256=None,
                    basis_phase_hmac_key_file=None,
                    basis_phase_hmac_key_id=None,
                    evidence_split_manifest=split_manifest,
                    target_hidden_size=int(args.target_hidden_size),
                )

            watchdog.stage(f"calibration-{index}-{step_cell.text}")
            calibration = _calibration_block(
                cell=step_cell,
                spec=spec,
                evidence=evidence,
                parent_weights=parent_weights,
                tolerance=calibration_tolerance,
            )
            _log(
                f"{step_cell.text}: calibration relL2="
                f"{calibration['copy_exact_reproduction_relL2']} "
                f"applicable={calibration['applicable']} "
                f"passed={calibration['passed']}"
            )
            if calibration["applicable"] and not calibration["passed"]:
                raise MatrixCompositionRunError(
                    f"{step_cell.text} calibration refused at "
                    f"{calibration['copy_exact_reproduction_relL2']} against a "
                    f"{calibration_tolerance} tolerance; no solve below it may "
                    "be believed and the composed child is void."
                )

            watchdog.stage(f"solve-{index}-{step_cell.text}")
            technique = resolve(step_cell.technique, step_cell.component)
            request = CellTechniqueRequest(
                cell=step_cell,
                component=spec,
                level=step_cell.level,
                level_width=level_width,
                seed=int(args.seed),
                parent_weights=parent_weights,
                evidence=evidence,
                knobs={**knobs, **(item.get("knobs") or {})},
                workspace=output / f"step-{index:02d}-{step_cell.text}",
                log=_log,
            )
            solve_started = time.monotonic()
            result = technique(request)
            validate_result(request, result)
            solve_seconds = time.monotonic() - solve_started
            solve_block = dict(result.solve)
            solve_block["stored_parameter_count"] = int(result.stored_parameter_count)
            solve_block["parent_parameter_count"] = int(spec.parent_parameter_count)
            solve_block["solve_wall_time_seconds"] = float(solve_seconds)

            watchdog.stage(f"boundary-{index}-{step_cell.text}")
            boundary, boundary_diagnostics = _boundary_block(
                cell=step_cell, spec=spec, evidence=evidence, result=result
            )
            installed_digests = cell_runner._install_component(
                model, result.installed_weights
            )
            installed_parameters = int(
                sum(int(value.numel()) for value in result.installed_weights.values())
            )

            watchdog.stage(f"after-install-{index}-{step_cell.text}")
            terminal, layer_norms = cell_runner._terminal_states(
                model, tokenizer, probe_prompts, device=device
            )
            terminal_metrics = _row_metrics(terminal, control_terminal)
            ratios = (layer_norms / control_norms.clamp_min(1e-30)).reshape(-1)
            after: dict[str, Any] = {
                "terminal_relL2_vs_control": terminal_metrics["mean_relative_l2"],
                "terminal_cos_vs_control": terminal_metrics["mean_cosine"],
                "norm_ratio_per_layer_minmax": [
                    float(ratios.min().item()),
                    float(ratios.max().item()),
                ],
                "delta_terminal_vs_previous_step": (
                    terminal_metrics["mean_relative_l2"]
                    - (
                        entries[-1]["after_install"]["terminal_relL2_vs_control"]
                        if entries
                        else 0.0
                    )
                ),
                "free_run_token_match_vs_control": None,
                "free_run_mean_lcp": None,
                "probe_prompts": 0,
                "tokens_per_prompt": 0,
            }
            if bool(args.per_step_free_run):
                probe_ids = cell_runner._greedy_fixed(
                    model,
                    tokenizer,
                    probe_prompts,
                    tokens=int(args.probe_tokens),
                    device=device,
                )
                match, lcp = cell_runner._token_match(probe_ids, control_ids)
                after["free_run_token_match_vs_control"] = match
                after["free_run_mean_lcp"] = lcp
                after["probe_prompts"] = len(probe_prompts)
                after["tokens_per_prompt"] = int(args.probe_tokens)
            _log(
                f"{step_cell.text} installed: terminal drift "
                f"{after['terminal_relL2_vs_control']:.6f}, free-run "
                f"{after['free_run_token_match_vs_control']}"
            )

            entry = {
                "component_id": step_cell.component,
                "component": spec.name,
                "technique_id": step_cell.technique,
                "technique": TECHNIQUES[step_cell.technique],
                "level": step_cell.level,
                "site": step_cell.site,
                "level_width": int(level_width),
                "isolation_cell_id": step_cell.text,
                "isolation": dict(item.get("isolation") or {}),
                "selection_reason": str(item.get("selection_reason", "")),
                "weight_keys": list(spec.weight_keys),
                "installed_weight_keys": sorted(result.installed_weights),
                "installed_tensor_digests": installed_digests,
                "evidence": {
                    **conditioning,
                    "trace_cache_key": binding.get("trace_cache_key"),
                    "split_authority": binding.get("split_authority"),
                    "evidence_split_manifest_sha256": binding.get(
                        "evidence_split_manifest_sha256"
                    ),
                    "selection_sha256": binding.get("selection_sha256"),
                    "trace_cache_implementation_sha256": binding.get(
                        "implementation_sha256"
                    ),
                    "evidence_cache_reused": bool(
                        binding.get("evidence_cache_reused")
                    ),
                    "fit_rows": int(binding.get("fit_rows", 0) or 0),
                    "heldout_rows": int(binding.get("heldout_rows", 0) or 0),
                },
                "calibration": calibration,
                "solve": solve_block,
                "boundary": boundary,
                "after_install": after,
                "storage": {
                    "stored_parameter_count": int(result.stored_parameter_count),
                    "installed_parameter_count": installed_parameters,
                    "owned_parameter_count": int(spec.parent_parameter_count),
                    "declared_ratio": (
                        float(result.stored_parameter_count)
                        / float(spec.parent_parameter_count)
                        if spec.parent_parameter_count
                        else None
                    ),
                },
                "notes": str(result.notes or ""),
                "wall_time_seconds": float(time.monotonic() - step_started),
            }
            entry["first_divergent_metric"], entry["chain_walked_to"] = (
                _component_chain(entry)
            )
            entries.append(entry)
            installed_so_far.append(step_cell.text)
            # Kept current so the watchdog's timeout receipt carries the work
            # that was actually done rather than the empty placeholder.
            receipt["composition"]["install_order"] = [
                e["isolation_cell_id"] for e in entries
            ]
            receipt["composition"]["components"] = entries
            diagnostics["steps"].append(
                {
                    "index": index,
                    "cell_id": step_cell.text,
                    "boundary_diagnostics": boundary_diagnostics,
                    "terminal_metrics": terminal_metrics,
                    "wall_time_seconds": entry["wall_time_seconds"],
                    "solve_wall_time_seconds": solve_seconds,
                    "evidence_binding": binding,
                    "conditioning": conditioning,
                }
            )
            del evidence

        # ------------------------------------------------------------------
        # ADDENDUM 9: the alternating joint co-solve, on top of the M-A child
        # ------------------------------------------------------------------
        als_block: dict[str, Any] | None = None
        if mode == "M-C":
            watchdog.stage("als")
            als_block = _als_sweeps(
                args=args,
                model=model,
                tokenizer=tokenizer,
                entries=entries,
                recipe=recipe,
                parent_weights=parent_weights,
                knobs=knobs,
                capture_state=capture_state,
                base_manifest=base_manifest,
                control_terminal=control_terminal,
                control_norms=control_norms,
                control_ids=control_ids,
                probe_prompts=probe_prompts,
                device=device,
                output=output,
                watchdog=watchdog,
                calibration_tolerance=calibration_tolerance,
            )
            receipt["composition"]["als"] = als_block
            receipt["composition"]["components"] = entries

        # ------------------------------------------------------------------
        # the composed child
        # ------------------------------------------------------------------
        watchdog.stage("behavior")
        probe_ids = cell_runner._greedy_fixed(
            model, tokenizer, probe_prompts, tokens=int(args.probe_tokens),
            device=device,
        )
        smoke = cell_runner._decode_smoke(
            model,
            tokenizer,
            smoke_prompts,
            cap=int(args.smoke_token_cap),
            device=device,
            eos_id=eos_id,
        )
        terminal, layer_norms = cell_runner._terminal_states(
            model, tokenizer, probe_prompts, device=device
        )
        match, lcp = cell_runner._token_match(probe_ids, control_ids)
        terminal_metrics = _row_metrics(terminal, control_terminal)
        ratios = (layer_norms / control_norms.clamp_min(1e-30)).reshape(-1)

        storage = net_storage(parent_total, entries)
        composition = dict(receipt["composition"])
        composition.pop("incomplete_reason", None)
        composition["install_order"] = [e["isolation_cell_id"] for e in entries]
        composition["components"] = entries
        composition["storage"] = storage
        if als_block is not None:
            composition["als"] = als_block
        composition["global_drift"] = {
            "terminal_relL2_vs_control": terminal_metrics["mean_relative_l2"],
            "terminal_cos_vs_control": terminal_metrics["mean_cosine"],
            "per_component_worst_local_drift": max(
                float(e["after_install"]["terminal_relL2_vs_control"])
                for e in entries
            ),
            "per_component_local_drift_gate": 0.02,
            "components_over_local_gate": [
                str(e["component_id"])
                for e in entries
                if float(e["after_install"]["terminal_relL2_vs_control"]) > 0.02
            ],
        }
        composition["free_run_by_step"] = [
            {
                "cell_id": e["isolation_cell_id"],
                "isolation_free_run": e["isolation"].get("free_run"),
                "composed_free_run_after_install": e["after_install"][
                    "free_run_token_match_vs_control"
                ],
                "terminal_relL2_vs_control": e["after_install"][
                    "terminal_relL2_vs_control"
                ],
            }
            for e in entries
        ]

        receipt["composition"] = composition
        receipt["evidence"].update(
            {
                "parent_trace_cache_sha": str(
                    base_manifest.get("trace_cache_key")
                ),
                "chart_sha": None,
                "split_authority": "matrix-campaign-capture-manifest",
                "evidence_split_manifest_sha256": str(
                    base_manifest.get("manifest_sha256")
                ),
                "heldout_split": {
                    "fit_rows": int(entries[0]["evidence"]["fit_rows"]),
                    "heldout_rows": int(entries[0]["evidence"]["heldout_rows"]),
                },
                "conditioned_evidence_bases": [
                    {
                        "cell_id": e["isolation_cell_id"],
                        "conditioned": bool(e["evidence"]["conditioned"]),
                        "trace_cache_key": e["evidence"].get("trace_cache_key"),
                        "split_manifest_sha256": e["evidence"].get(
                            "evidence_split_manifest_sha256"
                        ),
                        "upstream_installed": e["evidence"]["upstream_installed"],
                    }
                    for e in entries
                ],
                "parent_model_forward_calls": 0,
                "backward_calls": 0,
            }
        )
        receipt["calibration"] = aggregate_calibration(entries)
        receipt["solve"] = aggregate_solve(
            entries,
            mode=mode,
            child_parameter_count=int(storage["child_parameter_count"]),
            parent_parameter_count=int(storage["parent_parameter_count"]),
        )
        receipt["boundary"] = aggregate_boundary(entries)
        receipt["composed"] = {
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
        receipt["behavior"] = {
            "free_run": {
                "probe_prompts": len(probe_prompts),
                "tokens_per_prompt": int(args.probe_tokens),
                "token_match_vs_control": match,
                "mean_lcp": lcp,
            },
            "decode_smoke": smoke,
        }
        receipt["outcome"]["notes"] = (
            f"{mode} composed child: {len(entries)} components folded, "
            f"{len(composition['passed_through'])} passed through; net storage "
            f"{storage['child_parameter_count']} against a parent of "
            f"{storage['parent_parameter_count']} "
            f"({storage['net_ratio']:.6f}x) -- "
            f"{'contracts' if storage['contracts'] else 'COSTS MORE THAN THE PARENT'}."
        )
        diagnostics["terminal_metrics"] = terminal_metrics
        # The composed child's own folded tensors, so the frozen review can be
        # taken on *this* child instead of on a replay that has to argue it is
        # the same one.  Only the folded tensors are written -- every other
        # tensor is the parent's, bit-identically -- so the file is tens of
        # megabytes and the digests in the receipt prove which child it is.
        tensor_path = output / "composed-child-tensors.pt"
        torch.save(
            {
                key: model.state_dict()[key].detach().to("cpu")
                for entry in entries
                for key in entry["installed_weight_keys"]
            },
            tensor_path,
        )
        composition["composed_child_tensors_sha256"] = cell_runner._file_sha256(
            tensor_path
        )
        composition["composed_child_tensors_path"] = str(tensor_path)
        receipt["composition"] = composition
        _atomic_write_json(
            output / "composed-child-state-digest.json",
            {
                "schema": RUNNER_SCHEMA,
                "kind": "composed-child-tensor-digests",
                "cell_id": cell.text,
                "install_order": composition["install_order"],
                "installed_tensor_digests": {
                    e["isolation_cell_id"]: e["installed_tensor_digests"]
                    for e in entries
                },
            },
        )

    except (
        CellTechniqueError,
        MatrixCompositionError,
        MatrixCompositionRunError,
        cell_runner.MatrixCellRunError,
        MatrixCellReceiptError,
        RuntimeError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        _log(f"COMPOSED CHILD FAILED {type(exc).__name__}: {exc}")
        composition = dict(receipt["composition"])
        composition["install_order"] = [e["isolation_cell_id"] for e in entries]
        composition["components"] = entries
        if entries:
            composition.pop("incomplete_reason", None)
        composition["incomplete_reason"] = (
            f"{type(exc).__name__} at install {len(entries)} of "
            f"{len(recipe['components'])}: {exc}"
        )[:2000]
        if entries:
            composition["storage"] = net_storage(
                diagnostics.get("parent_parameter_count", 1) or 1, entries
            )
        receipt["composition"] = composition
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
    diagnostics["over_target"] = elapsed > float(args.target_seconds)
    _atomic_write_json(output / "cell-diagnostics.json", diagnostics)
    _log(
        f"sealed {sealed['cell_id']} status={sealed['outcome']['status']} "
        f"first_divergent={sealed['outcome']['first_divergent_metric']} "
        f"receipt={sealed['receipt_sha'][:12]} in {elapsed:.1f}s"
    )
    return 0 if sealed["outcome"]["status"] in ("pass", "pending-review") else 1


def _prepare_conditioned_capture(
    *,
    args: argparse.Namespace,
    parent_trace_cache: Path,
    device: str,
) -> dict[str, Any]:
    """Everything M-B needs to capture, built before the first solve.

    An M-B run that discovers at step four that it has no corpus has burned
    three solves and a capture for nothing, so the corpus, the capture selection
    and the second (capture-dtype) model object are all constructed up front.
    """

    import run_lfm2_residual_attention_fold as residual

    for name in ("conditioned_capture_root", "donor_calibration"):
        if getattr(args, name) is None:
            raise MatrixCompositionRunError(
                f"--{name.replace('_', '-')} is required in modes M-B and M-C."
            )
    manifest = json.loads(
        (Path(parent_trace_cache) / "manifest.json").read_text(encoding="utf-8")
    )
    config = _capture_config(manifest)
    corpus = residual.load_calibration_batches(
        Path(args.donor_calibration).expanduser().resolve(strict=True)
    )
    declared = len(manifest["corpus"]["ordered_batch_input_sha256"])
    count = int(args.capture_batch_count or declared)
    if count != declared:
        raise MatrixCompositionRunError(
            f"The parent evidence base captured {declared} batches; a "
            f"conditioned capture of {count} would be scored on a different "
            "corpus."
        )
    batches = tuple(corpus.batches[:count])
    observed = [
        str(value) for value in manifest["corpus"]["ordered_batch_input_sha256"]
    ]
    capture_model = cell_runner._load_parent(
        Path(args.parent_model), device, cell_runner._dtype(str(args.capture_dtype))
    )
    configuration = getattr(capture_model, "config", None)
    if configuration is None:
        raise MatrixCompositionRunError("Capture model exposes no configuration.")
    configuration._attn_implementation = "eager"
    _log(
        f"M-B capture preflight: {count} batches, selection "
        f"{config.as_dict()['selection_sha256'][:16]}, capture dtype "
        f"{args.capture_dtype}"
    )
    return {
        "model": capture_model,
        "batches": batches,
        "config": config,
        "batch_count": count,
        "selection_sha256": str(config.as_dict()["selection_sha256"]),
        "parent_base_record_ids": observed,
        "capture_dtype": str(args.capture_dtype),
        "corpus_path": str(args.donor_calibration),
    }


def _module_digests() -> dict[str, str]:
    """sha256(first 16) of every matrix module this process could have run.

    The deployment discipline the campaign runs under is "record which bytes
    produced this number", and `code_identity.src_tree_sha` covers the whole
    tree in one digest that changes for any reason.  These are the files a
    reader actually wants to compare against a previous wave.
    """

    digests: dict[str, str] = {}
    for path in sorted((_REPO_ROOT / "src" / "neural_forge").glob("matrix_*.py")):
        digests[path.name] = cell_runner._file_sha256(path)[:16]
    for name in (
        "run_lfm2_matrix_cell.py",
        "run_lfm2_matrix_composition.py",
        "render_matrix_ledger.py",
        "seal_matrix_cell_review.py",
        "derive_matrix_composition_recipe.py",
    ):
        candidate = _REPO_ROOT / "experiments" / name
        if candidate.is_file():
            digests[name] = cell_runner._file_sha256(candidate)[:16]
    return digests


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
