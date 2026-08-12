#!/usr/bin/env python3
"""Wave 13 lane shopping: price a candidate lane on a fixed base, then keep going.

Wave 12's `mxp_shop_probe.py`, with the two things wave 13 needs it to do and
could not ask of it:

* **A lane is no longer only a SwiGLU triple.**  ``--candidate 12:T7:S4:attn``
  prices C10 -- the whole attention block, all four projections, merged onto
  shared head bases by the M5 kernel.  An attention lane removes 2,392,064
  parameters against a SwiGLU lane's 5,901,819, so the two are *not*
  comparable on marginal drift alone and this probe reports
  ``drift_per_million_removed`` beside it.  Ranking attention against MLP on
  raw marginal drift would be ranking a smaller purchase as a cheaper one.
* **A greedy ladder.**  ``--greedy-rounds N`` prices every candidate, installs
  the winner **permanently**, and re-prices what is left against the new base,
  N times.  `NOTES-MX-P` §6a's first open item is exactly this: the wave-12
  table priced the THIRD lane and the compounding rate past it was inferred
  from `COMP.MA.MXO2`'s trace rather than measured.  A ladder measures it.

The exactness argument is wave 12's and is unchanged: in M-A every component is
solved against the *parent's* sealed evidence, so a solved tensor does not
depend on what else is installed and the composed child is a function of the
SET.  A candidate installed after the base is the child a full run on the union
recipe would build.  The probe re-measures the base after every restore and
records whether it came back exactly.

**This probe decides recipes; it does not grade them.**  It seals no receipt,
appends no ledger row and makes no behavioural claim.  Under Cyril's wave-13
directive the drift it measures is a *search heuristic* -- which lane to try
first -- and never a verdict: a lane that prices badly here is still built and
still read by the frozen 15-prompt review, because `COMP.MA.MXO3` is alive at
drift 0.4618 against a nominal budget of 0.3765 and the residuals of an
R^2 = 0.62 law are not a gate.

Usage::

    python mxq_shop_probe.py --out .../probes/shop.json \\
        --base 5:T8:S4 --base 6:T8:S4 \\
        --candidate 12:T7:S4:attn --candidate 13:T5:S4 --greedy-rounds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_DEPLOY = Path(
    "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8"
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
from neural_forge.matrix_technique_head_merge import (  # noqa: E402
    c10_head_merge_admissible_width,
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

PROBE_SCHEMA = "neural-forge-matrix-wave13-lane-shopping-probe-v2"

#: kind -> the components a lane of that kind installs, in install order.
LANE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "mlp": ("C11", "C12", "C13"),
    "attn": ("C10",),
}

STORAGE_RATIOS = {"S2": 0.5, "S4": 0.25, "S8": 0.125}

_log = cell_runner._log


def _parse_lane(raw: str) -> dict[str, Any]:
    parts = str(raw).split(":")
    if len(parts) not in (3, 4):
        raise SystemExit(f"--candidate/--base {raw!r} is not LAYER:TECHNIQUE:LEVEL[:KIND]")
    layer, technique, level = parts[0], parts[1], parts[2]
    kind = parts[3] if len(parts) == 4 else "mlp"
    if kind not in LANE_COMPONENTS:
        raise SystemExit(f"lane kind {kind!r} is not one of {sorted(LANE_COMPONENTS)}")
    return {
        "layer": int(layer),
        "technique": technique,
        "level": level,
        "kind": kind,
        "text": f"L{int(layer):02d}.{technique}.{level}.{kind}",
    }


def _width_for(spec: Any, technique: str, component: str, ratio: float) -> int | None:
    """The width whose stored form fits the bound, aligned where the kernel needs it."""

    if (technique, component) == ("T7", "C10"):
        return c10_head_merge_admissible_width(spec, ratio)
    return storage_width_for_ratio(spec, technique, ratio)


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


def _lane_keys(lane: dict[str, Any], parent_weights) -> list[str]:
    return [
        key
        for component in LANE_COMPONENTS[lane["kind"]]
        for key in component_spec(
            CellId.parse(
                f"{component}.{lane['technique']}.{lane['level']}."
                f"L{lane['layer']:02d}"
            ),
            parent_weights,
        ).weight_keys
    ]


def _install_lane(
    *,
    model,
    parent_weights,
    lane: dict[str, Any],
    knobs: dict[str, Any],
    args,
    output: Path,
) -> list[dict[str, Any]]:
    """Solve and install one whole lane, reporting each component."""

    installed: list[dict[str, Any]] = []
    layer = int(lane["layer"])
    technique = str(lane["technique"])
    level = str(lane["level"])
    for component in LANE_COMPONENTS[lane["kind"]]:
        cell = CellId.parse(f"{component}.{technique}.{level}.L{layer:02d}")
        spec = component_spec(cell, parent_weights)
        ratio = STORAGE_RATIOS[level]
        width = _width_for(spec, technique, component, ratio)
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
        record = {
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
        # The head merge's own gates travel with the record, because a probe
        # that priced a lane whose identity gate had drifted would be pricing
        # something other than a merge.
        for field in (
            "identity_gate_max_abs_deviation",
            "write_heldout_relative_l2",
            "write_heldout_relative_l2_procrustes_only",
            "o_refit_adopted",
            "query_partition",
            "kv_partition",
        ):
            if field in result.solve:
                record[field] = result.solve[field]
        installed.append(record)
        del evidence
    return installed


def _restore(model, runtime_weights, keys) -> None:
    cell_runner._install_component(
        model,
        {key: runtime_weights[key].detach().clone() for key in keys},
    )


def _measure(model, tokenizer, prompts, control_states, control_ids, args):
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
    return drift, norms, match, lcp


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
        "--base",
        action="append",
        default=[],
        metavar="LAYER:TECHNIQUE:LEVEL[:KIND]",
        help="A lane in the fixed base, e.g. 5:T8:S4 (default: MXO3's body).",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LAYER:TECHNIQUE:LEVEL[:KIND]",
        help="A lane to price on top of the base, e.g. 12:T7:S4:attn.",
    )
    parser.add_argument(
        "--greedy-rounds",
        type=int,
        default=0,
        help=(
            "After ranking, install the cheapest candidate PERMANENTLY and "
            "re-price the rest against the new base, this many times."
        ),
    )
    parser.add_argument(
        "--rank-by",
        default="marginal_drift",
        choices=("marginal_drift", "drift_per_million_removed"),
        help=(
            "Which price the greedy ladder buys on.  Raw marginal drift ranks "
            "a smaller purchase as a cheaper one; per-parameter ranks what the "
            "storage target actually needs."
        ),
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(int(args.seed))
    started = time.monotonic()
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

    base_lanes = [_parse_lane(raw) for raw in (args.base or ["5:T8:S4", "6:T8:S4"])]
    candidates = [_parse_lane(raw) for raw in args.candidate]

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
            "Wave 13 lane shopping: the marginal composed drift of one candidate "
            "lane -- SwiGLU triple or M5 attention head merge -- installed on a "
            "fixed base, every candidate against the same base in one process, "
            "optionally repeated as a greedy ladder."
        ),
        "not_a_gate": (
            "Drift chooses the ORDER of the search and never the verdict.  Every "
            "lane this probe prices is still built and still read by the frozen "
            "15-prompt review; MXO3 is alive at drift 0.4618 against a nominal "
            "0.3765 budget, so the law's residuals are not a licence to refuse a "
            "child a reading."
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
        "rank_by": str(args.rank_by),
        "rounds": [],
    }

    installed_lanes: list[dict[str, Any]] = []
    installed_records: list[dict[str, Any]] = []
    _log("probe: installing the base")
    for lane in base_lanes:
        installed_records.extend(
            _install_lane(
                model=model,
                parent_weights=parent_weights,
                lane=lane,
                knobs=knobs,
                args=args,
                output=output,
            )
        )
        installed_lanes.append(lane)
    base_drift, base_norms, base_match, base_lcp = _measure(
        model, tokenizer, prompts, control_states, control_ids, args
    )
    report["base"] = {
        "lanes": [lane["text"] for lane in base_lanes],
        "components": installed_records,
        "global_drift": base_drift,
        "norm_ratio_per_layer_minmax": base_norms,
        "free_run_token_match_vs_control": base_match,
        "free_run_mean_lcp": base_lcp,
        "stored_parameter_count": sum(
            int(entry.get("stored_parameter_count", 0)) for entry in installed_records
        ),
        "removed_parameter_count": sum(
            int(entry.get("parent_parameter_count", 0))
            - int(entry.get("stored_parameter_count", 0))
            for entry in installed_records
        ),
    }
    _log(f"probe: base global drift {base_drift:.6f}")

    remaining = list(candidates)
    for round_index in range(int(args.greedy_rounds) + 1):
        occupied = {int(lane["layer"]) for lane in installed_lanes}
        round_record: dict[str, Any] = {
            "round": round_index,
            "base_lanes": [lane["text"] for lane in installed_lanes],
            "base_global_drift": base_drift,
            "candidates": [],
        }
        for lane in remaining:
            record: dict[str, Any] = {
                "candidate": lane["text"],
                "layer": lane["layer"],
                "technique": lane["technique"],
                "level": lane["level"],
                "kind": lane["kind"],
            }
            if int(lane["layer"]) in occupied:
                record["skipped"] = (
                    "NOTES-MX-K §2.1: block L%02d already hosts a folded lane; a "
                    "second lane in one block is refused unless a recipe says so "
                    "on purpose." % int(lane["layer"])
                )
                round_record["candidates"].append(record)
                continue
            keys = _lane_keys(lane, parent_weights)
            started_candidate = time.monotonic()
            _log(f"probe round {round_index}: candidate {lane['text']}")
            try:
                components = _install_lane(
                    model=model,
                    parent_weights=parent_weights,
                    lane=lane,
                    knobs=knobs,
                    args=args,
                    output=output,
                )
                drift, norms, match, lcp = _measure(
                    model, tokenizer, prompts, control_states, control_ids, args
                )
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
            except (
                CellTechniqueError,
                RuntimeError,
                ValueError,
                KeyError,
                OSError,
            ) as exc:
                record.update({"failed": f"{type(exc).__name__}: {exc}"[:600]})
            finally:
                _restore(model, parent_runtime, keys)
                check, _norms = _drift(
                    model,
                    tokenizer,
                    prompts,
                    control_states["terminal"],
                    control_states["layer_norms"],
                    str(args.device),
                )
                record["base_drift_after_restore"] = check
                record["base_restored_exactly"] = bool(abs(check - base_drift) < 1e-9)
                record["wall_time_seconds"] = round(
                    time.monotonic() - started_candidate, 3
                )
            round_record["candidates"].append(record)
            _log(
                f"probe round {round_index}: {lane['text']} marginal "
                f"{record.get('marginal_drift')} per-Mparam "
                f"{record.get('drift_per_million_removed')} (base restored "
                f"{record['base_restored_exactly']})"
            )

        priced = [
            entry
            for entry in round_record["candidates"]
            if entry.get(args.rank_by) is not None
        ]
        priced.sort(key=lambda entry: float(entry[args.rank_by]))
        round_record["ranked"] = [
            {
                "candidate": entry["candidate"],
                "marginal_drift": entry.get("marginal_drift"),
                "drift_per_million_removed": entry.get("drift_per_million_removed"),
                "removed": entry.get("lane_removed_parameter_count"),
            }
            for entry in priced
        ]
        report["rounds"].append(round_record)

        if round_index >= int(args.greedy_rounds) or not priced:
            break
        winner_text = priced[0]["candidate"]
        winner = next(lane for lane in remaining if lane["text"] == winner_text)
        _log(f"probe round {round_index}: installing winner {winner_text}")
        installed_records.extend(
            _install_lane(
                model=model,
                parent_weights=parent_weights,
                lane=winner,
                knobs=knobs,
                args=args,
                output=output,
            )
        )
        installed_lanes.append(winner)
        remaining = [lane for lane in remaining if lane["text"] != winner_text]
        base_drift, base_norms, base_match, base_lcp = _measure(
            model, tokenizer, prompts, control_states, control_ids, args
        )
        round_record["installed_winner"] = winner_text
        round_record["new_base_global_drift"] = base_drift

    report["final_base"] = {
        "lanes": [lane["text"] for lane in installed_lanes],
        "global_drift": base_drift,
        "stored_parameter_count": sum(
            int(entry.get("stored_parameter_count", 0)) for entry in installed_records
        ),
        "removed_parameter_count": sum(
            int(entry.get("parent_parameter_count", 0))
            - int(entry.get("stored_parameter_count", 0))
            for entry in installed_records
        ),
    }
    report["wall_time_seconds"] = round(time.monotonic() - started, 3)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _log(f"probe sealed -> {destination} in {report['wall_time_seconds']:.1f}s")
    for round_record in report["rounds"]:
        print(f"--- round {round_record['round']}  base {round_record['base_lanes']}")
        for entry in round_record["ranked"]:
            per_million = entry["drift_per_million_removed"]
            print(
                f"{entry['candidate']:>22}  marginal {entry['marginal_drift']:+.6f}"
                f"  per-Mparam "
                f"{'n/a' if per_million is None else format(per_million, '+.6f')}"
                f"  removes {entry['removed']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
