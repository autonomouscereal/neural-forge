#!/usr/bin/env python3
"""M6 -- drift shopping: what does each candidate lane cost ON TOP OF the frontier?

`NOTES-MX-O` §10.2 names the one decision wave 11 took on autopilot: MXO4 added
L07 because it was next in depth, and the drift model ranks other lanes below
L07's unmeasured components.  §4c's own install trace is why that matters -- in
`COMP.MA.MXO2` the eleven lanes cost, in install order, 0.3659 / 0.1139 / 0.0421
/ 0.0687 / 0.0685 / 0.0530 / 0.0518 / 0.0297 / 0.0054 / 0.0105 / **-0.0061** --
so a lane's marginal drift is a function of *where in the stack it lands* and the
ledger's per-site medians, pooled across positions, cannot rank the next one.

This probe measures the number the decision actually needs: **the marginal drift
of one candidate lane installed on top of a FIXED base**, with every candidate
measured against the same base, on the same GPU, in one process.

It is exact for M-A and the reason is worth stating, because it is what makes the
probe affordable.  In M-A every component is solved against the **parent's**
sealed evidence, so a component's solved tensor does not depend on what else is
installed; the composed child is a function of the SET, not of the install order.
So installing a candidate lane after the base -- even at a layer shallower than
the base's -- yields exactly the tensors, and exactly the child, that a full
`run-mxp.sh` on the union recipe would produce.  Restoring the parent's own
tensors afterwards returns the model to the base exactly, and the probe asserts
that by re-measuring the base drift after every candidate.

What it is NOT: a sealed cell.  It writes no receipt and appends no ledger row.
It is a measurement that *chooses a recipe*, exactly as `mxo_sharing_probe.py`
was a falsification that *closed a mechanism*, and the recipe it chooses is then
run, sealed and reviewed the normal way.

Usage::

    python mxp_shop_probe.py --out /mnt/ai_data/neural-forge-matrix/wave12/probes/shop.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_DEPLOY = Path(
    "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-4d8774f6"
)
for _extra in (_DEPLOY / "src", _DEPLOY / "experiments"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import torch  # noqa: E402

import run_lfm2_matrix_cell as cell_runner  # noqa: E402
from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_composition import (  # noqa: E402
    evidence_cache_name,
    evidence_endpoints,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    resolve,
    resolve_level_width,
    storage_width_for_ratio,
    validate_result,
)

PROBE_SCHEMA = "neural-forge-matrix-wave12-lane-shopping-probe-v1"

#: The frontier's body, verbatim from `COMP.MA.MXO3`'s sealed recipe minus the
#: token interface.  The interface is left out on purpose: it is installed last,
#: it costs +0.040485 on this base (MXO3's own trace), and leaving it out saves a
#: 268 MB solve per probe run without changing any ranking.
BASE_LANES: tuple[int, ...] = (5, 6)
BASE_TECHNIQUE = "T8"
BASE_LEVEL = "S4"

#: The MLP lane, in install order within a block.
LANE_COMPONENTS: tuple[str, ...] = ("C11", "C12", "C13")

STORAGE_RATIOS = {"S4": 0.25, "S8": 0.125}

_log = cell_runner._log


def _drift(model, tokenizer, prompts, control_terminal, control_norms, device):
    terminal, layer_norms = cell_runner._terminal_states(
        model, tokenizer, prompts, device=device
    )
    metrics = cell_runner._row_metrics(terminal, control_terminal)
    ratios = (layer_norms / control_norms.clamp_min(1e-30)).reshape(-1)
    return float(metrics["mean_relative_l2"]), [
        float(ratios.min().item()),
        float(ratios.max().item()),
    ]


def _install_lane(
    *,
    model,
    parent_weights,
    layer: int,
    technique: str,
    level: str,
    knobs: dict[str, Any],
    args,
    output: Path,
) -> list[dict[str, Any]]:
    """Solve and install one whole SwiGLU lane, reporting each component."""

    installed: list[dict[str, Any]] = []
    for component in LANE_COMPONENTS:
        cell = CellId.parse(f"{component}.{technique}.{level}.L{layer:02d}")
        spec = component_spec(cell, parent_weights)
        ratio = STORAGE_RATIOS[level]
        width = storage_width_for_ratio(spec, technique, ratio)
        if width is None:
            installed.append(
                {
                    "cell_id": cell.text,
                    "cannot_express": True,
                    "reason": (
                        f"{technique} cannot store {component} at {level} at any "
                        "admissible width (ADDENDUM 6 cannot-express)"
                    ),
                }
            )
            continue
        level_width = resolve_level_width(spec, level, x4_width=int(width))
        evidence, binding = cell_runner.load_site_evidence(
            trace_cache_path=Path(args.parent_trace_cache),
            layer_index=layer,
            endpoints=evidence_endpoints(cell),
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
        started = time.monotonic()
        request = CellTechniqueRequest(
            cell=cell,
            component=spec,
            level=level,
            level_width=int(level_width),
            seed=int(args.seed),
            parent_weights=parent_weights,
            evidence=evidence,
            knobs=knobs,
            workspace=output / f"probe-{cell.text}",
            log=_log,
        )
        result = resolve(cell.technique, cell.component)(request)
        validate_result(request, result)
        cell_runner._install_component(model, result.installed_weights)
        bound = ratio * float(spec.parent_parameter_count)
        installed.append(
            {
                "cell_id": cell.text,
                "cannot_express": False,
                "level_width": int(level_width),
                "stored_parameter_count": int(result.stored_parameter_count),
                "parent_parameter_count": int(spec.parent_parameter_count),
                "declared_ratio": (
                    float(result.stored_parameter_count)
                    / float(spec.parent_parameter_count)
                ),
                "storage_bound": bound,
                "storage_bound_satisfied": bool(
                    int(result.stored_parameter_count) <= bound
                ),
                "installed_weight_keys": sorted(result.installed_weights),
                "solve_wall_time_seconds": round(time.monotonic() - started, 3),
                "evidence_selection_sha256": binding.get("selection_sha256"),
            }
        )
        del evidence
    return installed


def _restore(model, parent_weights, keys) -> None:
    cell_runner._install_component(
        model,
        {key: parent_weights[key].detach().clone() for key in keys},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--parent-model",
        type=Path,
        default=Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M"),
    )
    parser.add_argument(
        "--control-artifact",
        type=Path,
        default=Path("/mnt/ai_data/neural-forge-matrix/cells/CTRL.copy_exact"),
    )
    parser.add_argument("--parent-trace-cache", type=Path, required=True)
    parser.add_argument("--evidence-split-manifest", type=Path, required=True)
    parser.add_argument("--evidence-cache-root", type=Path, required=True)
    parser.add_argument("--knobs", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-compute-architecture", default="sm86")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forced-decode-count", type=int, default=8)
    parser.add_argument("--target-hidden-size", type=int, default=816)
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument(
        "--probe-prompts", type=Path, default=cell_runner.PROBE_PROMPTS_FILE
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LAYER:TECHNIQUE:LEVEL",
        help="A lane to price on top of the base, e.g. 7:T8:S4 or 13:T5:S4.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    started = time.monotonic()
    # The technique lane registers itself on import; the composition runner does
    # this before its first solve and a probe that skips it resolves nothing.
    technique_modules = cell_runner.discover_technique_modules()
    architecture = cell_runner.compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise SystemExit(
            f"--device {args.device} resolves to {architecture!r}, not "
            f"{args.require_compute_architecture!r}"
        )
    output = Path(args.workspace).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    knobs = json.loads(Path(args.knobs).read_text(encoding="utf-8"))
    knobs.setdefault("python", sys.executable)
    knobs.setdefault("device", str(args.device))

    prompts, probe_sha = cell_runner._load_prompt_set(
        args.probe_prompts, "free-run-probe"
    )
    control_root = Path(args.control_artifact)
    control = json.loads((control_root / "control.json").read_text(encoding="utf-8"))
    if control["probe_set_sha256"] != probe_sha:
        raise SystemExit("Control artifact was built on a different probe set.")
    control_states = torch.load(control_root / "control-states.pt", weights_only=True)
    control_ids = [list(map(int, row)) for row in control["probe_token_ids"]]

    tokenizer = cell_runner._load_tokenizer(Path(args.parent_model))
    model = cell_runner._load_parent(
        Path(args.parent_model), str(args.device), cell_runner._dtype(str(args.dtype))
    )
    parent_weights = {
        key: value.detach().to(device="cpu", dtype=torch.float64)
        for key, value in model.state_dict().items()
    }
    parent_runtime = {
        key: value.detach().to("cpu").clone()
        for key, value in model.state_dict().items()
    }

    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "purpose": (
            "M6 drift shopping: the marginal composed drift of one candidate "
            "SwiGLU lane installed on a fixed base, measured for every candidate "
            "against the same base in one process."
        ),
        "exactness_argument": (
            "M-A solves every component against the PARENT's sealed evidence, so "
            "a solved tensor does not depend on what else is installed and the "
            "composed child is a function of the SET.  A candidate installed "
            "after the base therefore equals the child a full run on the union "
            "recipe would build, and the base is restored exactly by writing the "
            "parent's own tensors back."
        ),
        "deployment": str(_DEPLOY),
        "compute_architecture": architecture,
        "technique_modules": technique_modules,
        "base": {
            "lanes": [f"L{layer:02d}" for layer in BASE_LANES],
            "technique": BASE_TECHNIQUE,
            "level": BASE_LEVEL,
            "source": "COMP.MA.MXO3's body, minus the C01 token interface",
        },
        "candidates": [],
    }

    _log("probe: installing the base")
    base_records = []
    for layer in BASE_LANES:
        base_records.extend(
            _install_lane(
                model=model,
                parent_weights=parent_weights,
                layer=layer,
                technique=BASE_TECHNIQUE,
                level=BASE_LEVEL,
                knobs=knobs,
                args=args,
                output=output,
            )
        )
    base_drift, base_norms = _drift(
        model,
        tokenizer,
        prompts,
        control_states["terminal"],
        control_states["layer_norms"],
        str(args.device),
    )
    base_ids = cell_runner._greedy_fixed(
        model, tokenizer, prompts, tokens=int(args.probe_tokens), device=str(args.device)
    )
    base_match, base_lcp = cell_runner._token_match(base_ids, control_ids)
    report["base"]["components"] = base_records
    report["base"]["global_drift"] = base_drift
    report["base"]["norm_ratio_per_layer_minmax"] = base_norms
    report["base"]["free_run_token_match_vs_control"] = base_match
    report["base"]["free_run_mean_lcp"] = base_lcp
    report["base"]["stored_parameter_count"] = sum(
        int(entry.get("stored_parameter_count", 0)) for entry in base_records
    )
    _log(f"probe: base global drift {base_drift:.6f} (MXO3 body reads 0.421325)")

    for raw in args.candidate:
        layer_text, technique, level = str(raw).split(":")
        layer = int(layer_text)
        started_candidate = time.monotonic()
        _log(f"probe: candidate L{layer:02d} {technique} {level}")
        keys = [
            key
            for component in LANE_COMPONENTS
            for key in component_spec(
                CellId.parse(f"{component}.{technique}.{level}.L{layer:02d}"),
                parent_weights,
            ).weight_keys
        ]
        record: dict[str, Any] = {
            "candidate": f"L{layer:02d}.{technique}.{level}",
            "layer": layer,
            "technique": technique,
            "level": level,
        }
        try:
            components = _install_lane(
                model=model,
                parent_weights=parent_weights,
                layer=layer,
                technique=technique,
                level=level,
                knobs=knobs,
                args=args,
                output=output,
            )
            drift, norms = _drift(
                model,
                tokenizer,
                prompts,
                control_states["terminal"],
                control_states["layer_norms"],
                str(args.device),
            )
            ids = cell_runner._greedy_fixed(
                model,
                tokenizer,
                prompts,
                tokens=int(args.probe_tokens),
                device=str(args.device),
            )
            match, lcp = cell_runner._token_match(ids, control_ids)
            stored = sum(
                int(entry.get("stored_parameter_count", 0)) for entry in components
            )
            covered = sum(
                int(entry.get("parent_parameter_count", 0)) for entry in components
            )
            record.update(
                {
                    "components": components,
                    "cannot_express": any(
                        bool(entry.get("cannot_express")) for entry in components
                    ),
                    "global_drift": drift,
                    "marginal_drift": drift - base_drift,
                    "norm_ratio_per_layer_minmax": norms,
                    "free_run_token_match_vs_control": match,
                    "free_run_mean_lcp": lcp,
                    "lane_stored_parameter_count": stored,
                    "lane_covered_parameter_count": covered,
                    "lane_removed_parameter_count": covered - stored,
                    "storage_bound_satisfied": all(
                        bool(entry.get("storage_bound_satisfied", True))
                        for entry in components
                    ),
                    "drift_per_million_removed": (
                        (drift - base_drift) * 1_000_000.0 / float(covered - stored)
                        if covered > stored
                        else None
                    ),
                }
            )
        except (CellTechniqueError, RuntimeError, ValueError, KeyError, OSError) as exc:
            record.update({"failed": f"{type(exc).__name__}: {exc}"[:600]})
        finally:
            _restore(model, parent_runtime, keys)
            check, _ = _drift(
                model,
                tokenizer,
                prompts,
                control_states["terminal"],
                control_states["layer_norms"],
                str(args.device),
            )
            record["base_drift_after_restore"] = check
            record["base_restored_exactly"] = bool(
                abs(check - base_drift) < 1e-9
            )
            record["wall_time_seconds"] = round(
                time.monotonic() - started_candidate, 3
            )
        report["candidates"].append(record)
        _log(
            f"probe: L{layer:02d}.{technique}.{level} marginal "
            f"{record.get('marginal_drift')} (base restored "
            f"{record['base_restored_exactly']})"
        )

    report["wall_time_seconds"] = round(time.monotonic() - started, 3)
    report["ranked"] = [
        {
            "candidate": entry["candidate"],
            "marginal_drift": entry.get("marginal_drift"),
            "removed": entry.get("lane_removed_parameter_count"),
        }
        for entry in sorted(
            [e for e in report["candidates"] if e.get("marginal_drift") is not None],
            key=lambda e: float(e["marginal_drift"]),
        )
    ]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _log(f"probe sealed -> {destination} in {report['wall_time_seconds']:.1f}s")
    for entry in report["ranked"]:
        print(
            f"{entry['candidate']:>18}  marginal drift {entry['marginal_drift']:+.6f}"
            f"  removes {entry['removed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
