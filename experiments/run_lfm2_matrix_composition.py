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

Everything the isolation runner does per cell, this runner does per component
and then once more for the whole child, by **importing the isolation runner's
own functions** rather than restating them.  That is not tidiness: a composed
child's free-run number is only comparable with a cell's if it was produced by
the same `_greedy_fixed`, the same `_token_match`, the same `_row_metrics` and
the same control artifact, and the cheapest way to guarantee that is to call
them.

The receipt is `neural-forge-lfm2-matrix-cell-receipt-v1` with the mandatory
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
    deviation is sealed on the manifest's face, together with what the seeded
    rule *would* have produced, so nobody has to take the inheritance on trust.
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
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_directory": str(step["cache_directory"]),
        "trace_cache_key": str(cache.cache_key),
        "parent_identity_sha256": str(cache.parent_fingerprint),
        "conditioning": dict(step["conditioning"]),
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
        step={
            "cache_directory": str(cache_directory),
            "conditioning": conditioning,
        },
    )
    manifest_path = Path(output_root) / f"split-manifest-{cache.cache_key[:16]}.json"
    from neural_forge.dynamic_circuit_ir import canonical_json_bytes

    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_bytes(canonical_json_bytes(manifest))
    temporary.replace(manifest_path)
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


def _boundary_block(
    *,
    cell: CellId,
    spec: Any,
    evidence: Any,
    result: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The isolation runner's boundary stage, applied per component, verbatim."""

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
        if mode == "M-B":
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
                captured = capture_conditioned_evidence(
                    capture_model=capture_state["model"],
                    child_model=model,
                    output_root=Path(args.conditioned_capture_root)
                    / f"step-{index:02d}-{step_cell.text}",
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
                    / f"step-{index:02d}-{step_cell.text}"
                    / f"evidence-cache-{step_cell.site}-{step_cell.component}"
                )
            else:
                trace_cache_path = Path(args.parent_trace_cache)
                split_manifest = Path(args.evidence_split_manifest)
                evidence_cache = Path(args.evidence_cache_root) / str(
                    item.get("evidence_cache")
                    or evidence_cache_name(step_cell)
                )
                conditioning["reason"] = (
                    "M-A installs every winner solved against parent evidence "
                    "(ADDENDUM 4's null hypothesis)"
                    if mode != "M-B"
                    else "first component in topological order: nothing "
                    "upstream is folded, so the conditioned evidence IS the "
                    "parent evidence and no capture was spent"
                )

            watchdog.stage(f"evidence-{index}-{step_cell.text}")
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
                f"--{name.replace('_', '-')} is required in mode M-B."
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
