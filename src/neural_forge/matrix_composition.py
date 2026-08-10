#!/usr/bin/env python3
"""Recipe derivation and receipt aggregation for the matrix composition wave.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` ADDENDUM 1 makes the campaign's
terminal deliverable a *heterogeneous* child -- different techniques folding
different components -- and ADDENDUM 4 pre-registers the modes that child is
assembled by.  This module is everything about that assembly that is **not**
torch: which isolation winner each component ships, why, in what order they
install, what the composed child stores, and how a per-component array of
outcomes aggregates into the one failure chain §3 walks.

It is separate from `matrix_technique_registry` on purpose.  The registry is
the frozen interface between the harness lane and the technique lane, and the
technique surface that produced the isolation winners has to stay byte-identical
to itself or the winners stop being comparable with the child assembled from
them.  Nothing here is imported by a technique kernel; the one thing this module
borrows from the registry is `_LAYOUT`, read-only, because the parent tensors a
component owns are a property of the *layout* and a recipe has to know them
before any model is loaded (that is how a recipe that folds one tensor twice is
refused on a workstation instead of discovered on a GPU).

Three things live here:

* **Grading and eligibility** (ADDENDUM 3).  `grade` is the interpretation
  layer, `eligible` is the composition wave's admission rule -- promising or
  better, at a sealed site, with a coherent smoke review.
* **Selection** (ADDENDUM 1 + the storage-honesty criterion).  Two rankings,
  because the behavioural winner and the storage-honest winner are different
  cells at three components and running both is the only way to measure what
  the difference costs.  Every pick carries the sentence that justifies it into
  the receipt.
* **Aggregation.**  A composed child has one receipt and N component outcomes.
  The aggregate is deliberately pessimistic on every diagnostic link -- worst
  write cosine, worst macrocell -- because a composed child is exactly as sound
  as its worst install, and an averaged boundary would hide the component
  ADDENDUM 4 exists to localize.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .matrix_cell_receipt import (
    COMPONENTS,
    COMPOSITION_MODES,
    FREE_RUN_MIN_TOKEN_MATCH,
    CellId,
    MatrixCellReceiptError,
)
from .matrix_technique_registry import _C14_FINAL_KEYS, _LAYOUT

__all__ = [
    "ELIGIBLE_MIN_FREE_RUN",
    "RECIPE_SCHEMA",
    "STORAGE_TIEBREAK_FREE_RUN",
    "MatrixCompositionError",
    "aggregate_boundary",
    "aggregate_calibration",
    "aggregate_solve",
    "build_recipe",
    "candidate_from_receipt",
    "component_weight_keys",
    "eligible",
    "evidence_cache_name",
    "evidence_endpoints",
    "grade",
    "install_order_key",
    "net_storage",
    "validate_recipe",
]

RECIPE_SCHEMA = "neural-forge-lfm2-matrix-composition-recipe-v1"

#: ADDENDUM 3's `promising` floor.  0.90 is the aspirational "works outright"
#: line, not the dismissal line; 0.60 with a coherent smoke is the band the
#: composition wave is allowed to ship a component's winner from.
ELIGIBLE_MIN_FREE_RUN = 0.60

#: The storage-honesty band.  A variant that *contracts* is preferred over one
#: that costs more storage than the tensor it replaced whenever the two are
#: within this much free-run of each other.  It is not a free parameter: the
#: campaign's own storage-aware alternates table (`NOTES-MX-G` §5) lists three
#: components where the behavioural winner stores 1.4x-1.9x its parent, and the
#: measured cost of switching to the honest variant at C06 is 0.023 free-run --
#: inside this band -- while at C05 it is 0.123 -- outside it.  So the band
#: separates the two real cases in the sealed data and does not have to be
#: argued for.
STORAGE_TIEBREAK_FREE_RUN = 0.05

#: Topological rank *within* one layer: the pre-registered order is
#: input side -> output side, conv -> attention -> norms.
_KIND_RANK: Mapping[str, int] = {
    "short_conv": 0,
    "full_attention": 1,
    "mlp": 2,
    "norm": 3,
    "token": -1,
}


class MatrixCompositionError(RuntimeError):
    """A recipe or an aggregate is malformed.  Always fatal."""


# --------------------------------------------------------------------------
# grading (ADDENDUM 3)
# --------------------------------------------------------------------------


def grade(free_run: float | None, review: str) -> str:
    """ADDENDUM 3's interpretation layer, verbatim.

    `strong` is free-run >= 0.90 with a coherent smoke, `promising` >= 0.60
    with a coherent smoke, `weak` 0.20-0.60 or a degraded smoke, `dead` below
    0.20 or degenerate.  Nothing here is an acceptance clause -- §5 rule 1 is
    still the only one -- this is the adaptive policy's reading of a sealed
    number.
    """

    if free_run is None:
        return "dead"
    value = float(free_run)
    coherent = str(review) == "coherent"
    if value >= FREE_RUN_MIN_TOKEN_MATCH and coherent:
        return "strong"
    if value >= ELIGIBLE_MIN_FREE_RUN and coherent:
        return "promising"
    if value >= 0.20:
        return "weak"
    return "dead"


def eligible(candidate: Mapping[str, Any]) -> bool:
    """The composition wave's admission rule: promising or better, at a site.

    A cell that never reached behaviour (`infra`) is not a negative result and
    is not an eligible one either -- it carries no evidence in either
    direction, which is §3 link 1's whole point.
    """

    if str(candidate.get("status")) == "infra":
        return False
    return str(candidate.get("grade")) in ("promising", "strong")


# --------------------------------------------------------------------------
# what a component owns
# --------------------------------------------------------------------------


def component_weight_keys(cell: CellId) -> tuple[str, ...]:
    """The parent ``state_dict`` keys one cell's component would install.

    `matrix_technique_registry.component_spec` answers the same question but
    needs the parent's real tensors to do it, because it also resolves shapes.
    A recipe is built on a workstation with no model and no GPU and still has to
    refuse a recipe that folds one tensor twice, so the key half of that
    resolution is available here on its own.
    """

    if cell.component not in _LAYOUT:
        raise MatrixCompositionError(f"No layout for component {cell.component!r}.")
    layout = _LAYOUT[cell.component]
    templates: Sequence[str] = (
        _C14_FINAL_KEYS
        if cell.component == "C14" and cell.site == "FINAL"
        else layout["keys"]
    )
    keys: list[str] = []
    for template in templates:
        if "{layer}" in template:
            if cell.layer_index is None:
                raise MatrixCompositionError(
                    f"{cell.text} names no layer index but its component is "
                    "per-layer."
                )
            keys.append(template.format(layer=int(cell.layer_index)))
        else:
            keys.append(template)
    return tuple(keys)


def evidence_endpoints(cell: CellId) -> tuple[str, ...]:
    """The sorted endpoint set a cell at this component would ask evidence for.

    Mirrors `run_lfm2_matrix_cell.run`'s own construction -- input, write (or
    the macrocell when the layout has no name for the write), macrocell, plus
    the two residual endpoints every cell needs -- so a cache named after this
    set is a cache that binding check will accept.
    """

    layout = _LAYOUT[cell.component]
    return tuple(
        sorted(
            {
                name
                for name in (
                    layout["input"],
                    layout["write"] or layout["macrocell"],
                    layout["macrocell"],
                    "residual.pre",
                    "residual.post_operator",
                )
                if name
            }
        )
    )


def evidence_cache_name(cell: CellId) -> str:
    """``<site>-<canonical component>`` for the authenticated evidence cache.

    A site evidence cache binds a layer *and an endpoint set*, and two
    components can bind the same one: C10's endpoints moved to C09's in wave 3
    (`matrix_technique_registry:179`), which is why wave 3's own batch pointed
    every C10 cell at `L02-C09` and why the stale `L02-C10` directory on the run
    host still carries wave 1's `operator.*` endpoints and would be refused.
    Naming the cache after the *lowest component id that shares the endpoint
    set* makes that reuse mechanical instead of a thing somebody has to
    remember.
    """

    endpoints = evidence_endpoints(cell)
    sharing = [
        component
        for component in sorted(_LAYOUT)
        if component in COMPONENTS
        and _site_is_legal(component, cell.site)
        and evidence_endpoints(
            CellId(
                component=component,
                technique=cell.technique,
                level=cell.level,
                site=cell.site,
            )
        )
        == endpoints
    ]
    canonical = sharing[0] if sharing else cell.component
    return f"{cell.site}-{canonical}"


def _site_is_legal(component: str, site: str) -> bool:
    sites = COMPONENTS[component]["sites"]
    if sites == "global":
        return site == "GLOBAL"
    return site in {
        "FINAL" if entry == "final" else f"L{int(entry):02d}" for entry in sites
    }


def install_order_key(cell: CellId) -> tuple[int, int, str]:
    """Topological sort key: input side -> output side, conv -> attn -> norms.

    ADDENDUM 4 names the order for M-B and M-A inherits it so that the two modes
    differ in exactly one thing -- the evidence each solve saw -- and not in the
    order the installs landed.
    """

    kind = str(COMPONENTS[cell.component]["kind"])
    layer = cell.layer_index
    return (
        999 if layer is None else int(layer),
        _KIND_RANK.get(kind, 9),
        cell.component,
    )


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


def candidate_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one sealed cell receipt into a selection candidate.

    Reads the *receipt*, not the ledger row, on purpose.  Waves 1 and 2 sealed
    their storage counts under ``solve`` before `ledger_line` had columns for
    them, so a ledger-only derivation silently loses the storage honesty of
    every early cell -- including `C05.T5.M16.L05` (0.9375x) and
    `C03.T5.M16.L00` (0.9377x), two of the composition wave's picks.
    """

    cell = CellId.parse(str(receipt["cell_id"]))
    if cell.kind != "cell":
        raise MatrixCompositionError(
            f"{cell.text} is a {cell.kind} cell and is not a composition "
            "candidate; only single-component cells carry a winner."
        )
    solve = receipt["solve"]
    stored = solve.get("stored_parameter_count")
    parent = solve.get("parent_parameter_count")
    ratio = (
        float(stored) / float(parent)
        if isinstance(stored, int) and isinstance(parent, int) and parent > 0
        else None
    )
    free_run = receipt["behavior"]["free_run"]["token_match_vs_control"]
    review = receipt["outcome"]["decode_smoke_review"]
    return {
        "cell_id": cell.text,
        "component_id": cell.component,
        "component": str(receipt["component"]),
        "technique_id": cell.technique,
        "technique": str(receipt["technique"]),
        "level": cell.level,
        "site": cell.site,
        "campaign_wave": int(receipt["campaign_wave"]),
        "status": str(receipt["outcome"]["status"]),
        "first_divergent_metric": receipt["outcome"]["first_divergent_metric"],
        "decode_smoke_review": str(review),
        "free_run": None if free_run is None else float(free_run),
        "vs_control_delta": receipt["composed"].get("vs_copy_exact_control_delta"),
        "stored_parameter_count": stored,
        "parent_parameter_count": parent,
        "stored_ratio": ratio,
        "grade": grade(free_run, review),
        "receipt_sha": str(receipt["receipt_sha"]),
        "weight_keys": list(component_weight_keys(cell)),
    }


def _margin(candidate: Mapping[str, Any]) -> float:
    """Composed-terminal margin, ADDENDUM 1's tie-break.  Lower is better."""

    value = candidate.get("vs_control_delta")
    if value is None:
        return math.inf
    return float(value)


def _behaviour_first(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """ADDENDUM 1's rule with the storage-honesty tie-break applied on top.

    ADDENDUM 1: highest free-run-vs-control at the largest level that passes,
    ties broken by composed-terminal margin.  The coordinator's storage clause
    then overrides the head of that ranking when -- and only when -- an honest
    variant is within `STORAGE_TIEBREAK_FREE_RUN` of it, because a composed
    child that stores more than its parent is a failure regardless of
    behaviour.
    """

    ranked = sorted(
        candidates,
        key=lambda c: (-(c["free_run"] or 0.0), _margin(c), str(c["cell_id"])),
    )
    if not ranked:
        return []
    best = ranked[0]
    if (best.get("stored_ratio") or math.inf) < 1.0:
        return [dict(entry) for entry in ranked]
    honest = [
        entry
        for entry in ranked
        if entry.get("stored_ratio") is not None
        and float(entry["stored_ratio"]) < 1.0
        and (best["free_run"] or 0.0) - (entry["free_run"] or 0.0)
        <= STORAGE_TIEBREAK_FREE_RUN
    ]
    if not honest:
        return [dict(entry) for entry in ranked]
    promoted = honest[0]
    rest = [entry for entry in ranked if entry["cell_id"] != promoted["cell_id"]]
    return [dict(promoted)] + [dict(entry) for entry in rest]


def _storage_honest(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Only variants that genuinely contract, ranked by free-run.

    A candidate whose receipt never sealed a storage count is *not* admitted:
    unknown storage cannot be claimed honest, and the whole point of this
    variant is that its net ratio is defensible tensor by tensor.
    """

    honest = [
        entry
        for entry in candidates
        if entry.get("stored_ratio") is not None and float(entry["stored_ratio"]) < 1.0
    ]
    return [
        dict(entry)
        for entry in sorted(
            honest,
            key=lambda c: (-(c["free_run"] or 0.0), _margin(c), str(c["cell_id"])),
        )
    ]


RANKINGS = {
    "behaviour-first": _behaviour_first,
    "storage-honest": _storage_honest,
}


def build_recipe(
    candidates: Iterable[Mapping[str, Any]],
    *,
    variant: str,
    ranking: str,
    ledger_sha256: str,
    description: str = "",
) -> dict[str, Any]:
    """Assemble one composed-child recipe out of sealed single-component cells.

    Returns the recipe **and** the reasons: every component that ships names the
    cell it ships, the runner-up it beat and by how much; every component that
    does not ship names why, in the same array.  §5 rule 6 binds the result --
    twelve isolation winners are twelve isolation results -- so the recipe is a
    *hypothesis*, and the reasons are what make the composed receipt able to say
    which part of the hypothesis broke.
    """

    if ranking not in RANKINGS:
        raise MatrixCompositionError(f"Unknown ranking {ranking!r}.")
    rank = RANKINGS[ranking]
    by_component: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        by_component.setdefault(str(candidate["component_id"]), []).append(candidate)

    picks: list[dict[str, Any]] = []
    passed_through: list[dict[str, Any]] = []
    for component_id in sorted(by_component):
        entries = by_component[component_id]
        admitted = [entry for entry in entries if eligible(entry)]
        if not admitted:
            best = max(entries, key=lambda c: (c["free_run"] or 0.0))
            passed_through.append(
                {
                    "component_id": component_id,
                    "component": str(best["component"]),
                    "reason": (
                        f"no technique reached ADDENDUM 3 `promising` at this "
                        f"component: best sealed cell is {best['cell_id']} at "
                        f"free-run {best['free_run']:.4f} "
                        f"({best['grade']}), below the {ELIGIBLE_MIN_FREE_RUN:.2f} "
                        "admission floor.  Installed as parent weights "
                        "(copy_exact); no silent inclusion of a weak fold."
                    ),
                    "best_sealed_cell_id": str(best["cell_id"]),
                    "best_sealed_free_run": best["free_run"],
                    "best_sealed_grade": str(best["grade"]),
                }
            )
            continue
        ordered = rank(admitted)
        if not ordered:
            passed_through.append(
                {
                    "component_id": component_id,
                    "component": str(admitted[0]["component"]),
                    "reason": (
                        "every eligible variant at this component stores >= 1.0x "
                        "its parent tensor (best honest ratio: none sealed below "
                        "1.0), and this recipe variant admits only folds that "
                        "genuinely contract.  Installed as parent weights."
                    ),
                    "best_sealed_cell_id": str(admitted[0]["cell_id"]),
                    "best_sealed_free_run": admitted[0]["free_run"],
                    "best_sealed_grade": str(admitted[0]["grade"]),
                }
            )
            continue
        chosen = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        reason = (
            f"{ranking}: {chosen['cell_id']} is the best eligible sealed cell at "
            f"{component_id} -- free-run {chosen['free_run']:.4f} "
            f"({chosen['grade']}), stores "
            f"{_ratio_text(chosen['stored_ratio'])} of its parent tensor, at the "
            f"proven site {chosen['site']} only (ADDENDUM 5 rule 2-S)."
        )
        if runner_up is not None:
            reason += (
                f"  Runner-up {runner_up['cell_id']} at "
                f"{runner_up['free_run']:.4f} / "
                f"{_ratio_text(runner_up['stored_ratio'])}."
            )
        if (
            ranking == "behaviour-first"
            and (chosen.get("stored_ratio") or math.inf) < 1.0
            and runner_up is not None
            and (runner_up["free_run"] or 0.0) > (chosen["free_run"] or 0.0)
        ):
            reason += (
                f"  Promoted over the higher-free-run "
                f"{runner_up['cell_id']} by the storage-honesty clause: it gives "
                f"up {(runner_up['free_run'] or 0.0) - (chosen['free_run'] or 0.0):.4f} "
                f"free-run (inside the {STORAGE_TIEBREAK_FREE_RUN:.2f} band) and "
                f"stores {_ratio_text(chosen['stored_ratio'])} against "
                f"{_ratio_text(runner_up['stored_ratio'])}."
            )
        picks.append(
            {
                "cell_id": str(chosen["cell_id"]),
                "component_id": component_id,
                "selection_reason": reason,
                "evidence_cache": evidence_cache_name(
                    CellId.parse(str(chosen["cell_id"]))
                ),
                "evidence_endpoints": list(
                    evidence_endpoints(CellId.parse(str(chosen["cell_id"])))
                ),
                "isolation": {
                    key: chosen[key]
                    for key in (
                        "free_run",
                        "stored_ratio",
                        "stored_parameter_count",
                        "parent_parameter_count",
                        "grade",
                        "status",
                        "first_divergent_metric",
                        "receipt_sha",
                        "campaign_wave",
                        "vs_control_delta",
                    )
                },
                "weight_keys": list(chosen["weight_keys"]),
            }
        )

    picks, dropped = resolve_conflicts(picks, ranking=ranking)
    passed_through.extend(dropped)
    picks.sort(key=lambda entry: install_order_key(CellId.parse(entry["cell_id"])))
    passed_through.sort(key=lambda entry: str(entry["component_id"]))
    return {
        "schema": RECIPE_SCHEMA,
        "variant": str(variant),
        "ranking": ranking,
        "description": str(description),
        "derived_from_ledger_sha256": str(ledger_sha256),
        "selection_policy": {
            "admission": (
                "ADDENDUM 3 `promising` or better: free-run >= "
                f"{ELIGIBLE_MIN_FREE_RUN} and a coherent decode-smoke review"
            ),
            "ranking": ranking,
            "storage_tiebreak_free_run": STORAGE_TIEBREAK_FREE_RUN,
            "site_scope": (
                "ADDENDUM 5 rule 2-S: a winner installs at its sealed passing "
                "site only; every other site of the same component is passed "
                "through unfolded"
            ),
            "install_order": (
                "topological, input side -> output side; within a layer "
                "conv -> attention -> norms"
            ),
        },
        "install_order": [entry["cell_id"] for entry in picks],
        "components": picks,
        "passed_through": passed_through,
    }


def _ratio_text(value: Any) -> str:
    return "unsealed" if value is None else f"{float(value):.4f}x"


def resolve_conflicts(
    picks: Sequence[Mapping[str, Any]],
    *,
    ranking: str = "behaviour-first",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refuse two winners that own the same parent tensor, and say who won.

    C10 owns all four attention projections at its site, so it collides with
    C06/C07/C08/C09 at the same layer by construction.  Installing both would
    make the second install silently erase the first and score the survivor
    against evidence the loser invalidated.

    The tie is broken on **the recipe's own axis**, not on a third one imported
    for the occasion: a behaviour-first recipe keeps the higher free-run, a
    storage-honest recipe keeps the one that contracts more.  Deciding both
    variants the same way would collapse the only contrast the two recipes exist
    to measure -- at L02 the two axes name different winners (C09/C10), which is
    precisely the case worth seeing twice.
    """

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    owner: dict[str, str] = {}
    if ranking == "storage-honest":
        def order_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
            # Parameters removed, not the ratio.  The ratio is per-tensor and a
            # conflict is between folds of *different* tensor sets: at L02
            # C09.T5 stores 0.9375x of one 1024x1024 writer (-65,536 parameters)
            # while C10 stores 0.9583x of all four projections (-131,072).  The
            # child's weight is the second number.  ``parent_parameter_count``
            # stands in for the installed count here because the only component
            # whose kernel installs fewer tensors than it owns is C14, which
            # cannot collide with anything.
            stored = entry["isolation"].get("stored_parameter_count")
            owned = entry["isolation"].get("parent_parameter_count")
            saved = (
                math.inf
                if not isinstance(stored, int) or not isinstance(owned, int)
                else float(stored - owned)
            )
            return (
                saved,
                -(entry["isolation"].get("free_run") or 0.0),
                _margin(entry["isolation"]),
                str(entry["cell_id"]),
            )
    else:
        def order_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                -(entry["isolation"].get("free_run") or 0.0),
                _margin(entry["isolation"]),
                str(entry["cell_id"]),
            )

    ordered = sorted(picks, key=order_key)
    for entry in ordered:
        clash = [key for key in entry["weight_keys"] if key in owner]
        if clash:
            dropped.append(
                {
                    "component_id": str(entry["component_id"]),
                    "component": str(entry["component_id"]),
                    "reason": (
                        f"{entry['cell_id']} owns {sorted(clash)}, already folded "
                        f"by {owner[clash[0]]}; a composed child folds each "
                        "parent tensor at most once and the conflict was "
                        f"resolved on this recipe's own axis ({ranking}).  "
                        "Passed through unfolded in this recipe."
                    ),
                    "best_sealed_cell_id": str(entry["cell_id"]),
                    "best_sealed_free_run": entry["isolation"].get("free_run"),
                    "best_sealed_grade": str(entry["isolation"].get("grade")),
                    "conflicts_with": owner[clash[0]],
                }
            )
            continue
        for key in entry["weight_keys"]:
            owner[key] = str(entry["cell_id"])
        kept.append(dict(entry))
    return kept, dropped


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Fail closed on a recipe the runner could not honestly execute."""

    if recipe.get("schema") != RECIPE_SCHEMA:
        raise MatrixCompositionError(f"A recipe must declare {RECIPE_SCHEMA!r}.")
    components = recipe.get("components")
    if not isinstance(components, list) or not components:
        raise MatrixCompositionError("A recipe must fold at least one component.")
    if list(recipe.get("install_order") or []) != [
        str(entry["cell_id"]) for entry in components
    ]:
        raise MatrixCompositionError(
            "recipe.install_order must equal the components' cell ids in order."
        )
    owner: dict[str, str] = {}
    previous: tuple[int, int, str] | None = None
    for entry in components:
        cell = CellId.parse(str(entry["cell_id"]))
        declared = tuple(entry.get("weight_keys") or ())
        resolved = component_weight_keys(cell)
        if tuple(declared) != resolved:
            raise MatrixCompositionError(
                f"{cell.text} declares weight keys {declared} but its layout "
                f"resolves {resolved}."
            )
        for key in resolved:
            if key in owner:
                raise MatrixCompositionError(
                    f"{cell.text} installs {key!r}, already installed by "
                    f"{owner[key]}."
                )
            owner[key] = cell.text
        key_now = install_order_key(cell)
        if previous is not None and key_now < previous:
            raise MatrixCompositionError(
                f"{cell.text} breaks the topological install order "
                f"{previous} -> {key_now}."
            )
        previous = key_now
    if not isinstance(recipe.get("passed_through"), list):
        raise MatrixCompositionError("recipe.passed_through is mandatory.")


# --------------------------------------------------------------------------
# aggregation into the one chain §3 walks
# --------------------------------------------------------------------------


def aggregate_calibration(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The composed child's calibration gate: every applicable one must pass.

    A component whose gate is declared inapplicable (C03/C04/C10/C14, the four
    pre-registered category errors) contributes nothing in either direction --
    exactly as it does in an isolation cell.  If *every* component's gate is
    inapplicable the aggregate says so, with the reason, rather than claiming a
    pass it never measured.
    """

    applicable = [e for e in entries if bool(e["calibration"].get("applicable"))]
    if not applicable:
        reasons = "; ".join(
            f"{e['component_id']}: "
            f"{e['calibration'].get('inapplicable_reason', 'unstated')}"
            for e in entries
        )
        return {
            "copy_exact_reproduction_relL2": None,
            "passed": False,
            "applicable": False,
            "inapplicable_reason": (
                "no folded component in this recipe has a generic "
                f"parent-matmul predictor: {reasons}"
            ),
            "components_applicable": 0,
            "components_total": len(entries),
        }
    worst = max(
        applicable,
        key=lambda e: float(
            e["calibration"].get("copy_exact_reproduction_relL2") or 0.0
        ),
    )
    return {
        "copy_exact_reproduction_relL2": worst["calibration"].get(
            "copy_exact_reproduction_relL2"
        ),
        "passed": all(bool(e["calibration"].get("passed")) for e in applicable),
        "applicable": True,
        "worst_component_id": str(worst["component_id"]),
        "components_applicable": len(applicable),
        "components_total": len(entries),
    }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_solve(
    entries: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    child_parameter_count: int,
    parent_parameter_count: int,
) -> dict[str, Any]:
    """One ``solve`` block over N solves.  Pessimistic on every conditioning link.

    ``stored_parameter_count`` / ``parent_parameter_count`` carry the **net**
    child and parent totals rather than one component's, so the ledger's
    existing `stored_parameter_ratio` column is the composed child's net storage
    ratio without a new column and without a reader having to know that this row
    is different.
    """

    if mode not in COMPOSITION_MODES.values():
        raise MatrixCompositionError(f"Unknown composition mode {mode!r}.")
    conditions = [
        value
        for value in (_finite(e["solve"].get("condition_number")) for e in entries)
        if value is not None
    ]
    saturated = [
        e for e in entries if bool(e["solve"].get("condition_cap_saturated"))
    ]
    clamped = [e for e in entries if bool(e["solve"].get("clamp_engaged"))]
    # The clamp pair is reported for the *worst* engaged clamp, because that is
    # the one the chain's second link would trip on.  Reporting a mean would let
    # one healthy solve dilute one that saturated.
    requested = allowed = None
    if clamped:
        worst = max(
            clamped,
            key=lambda e: (
                (_finite(e["solve"].get("requested_clamp_ratio")) or 0.0)
                / max(_finite(e["solve"].get("allowed_clamp_ratio")) or 0.0, 1e-30)
            ),
        )
        requested = _finite(worst["solve"].get("requested_clamp_ratio"))
        allowed = _finite(worst["solve"].get("allowed_clamp_ratio"))
    return {
        "solver_path": f"composition:{mode}",
        "ridge_relative": max(
            (_finite(e["solve"].get("ridge_relative")) or 0.0) for e in entries
        ),
        "condition_number": max(conditions) if conditions else None,
        "condition_cap_saturated": bool(saturated),
        "condition_cap_saturated_components": [
            str(e["component_id"]) for e in saturated
        ],
        "prior_norm": sum(
            (_finite(e["solve"].get("prior_norm")) or 0.0) for e in entries
        ),
        "correction_norm": sum(
            (_finite(e["solve"].get("correction_norm")) or 0.0) for e in entries
        ),
        "requested_clamp_ratio": requested if clamped else 0.0,
        "allowed_clamp_ratio": allowed if clamped else 0.0,
        "clamp_engaged": bool(clamped),
        "clamp_engaged_components": [str(e["component_id"]) for e in clamped],
        "omp": None,
        "component_solves": len(entries),
        "stored_parameter_count": int(child_parameter_count),
        "parent_parameter_count": int(parent_parameter_count),
    }


def aggregate_boundary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One ``boundary`` block over N installs: worst write, worst macrocell.

    A composed child is exactly as sound as its worst install.  Averaging would
    let six clean attention solves hide one conv writer at 0.4, which is the
    component ADDENDUM 4 exists to localize.
    """

    measured = [e for e in entries if bool(e["boundary"].get("applicable"))]
    if not measured:
        return {
            "applicable": False,
            "local_solve_applicable": False,
            "macrocell_endpoint": None,
            "write_heldout_relL2": None,
            "write_heldout_cos": None,
            "post_operator_heldout_relL2": None,
            "post_operator_heldout_cos": None,
            "endpoint_band_le_0p4": None,
            "inapplicable_reason": (
                "no folded component in this recipe has a nameable sealed write "
                "endpoint; the verdict rests on the composed forward, the "
                "free-run probe and the frozen review"
            ),
        }
    local = [e for e in measured if bool(e["boundary"].get("local_solve_applicable"))]
    worst_write = (
        min(local, key=lambda e: float(e["boundary"]["write_heldout_cos"]))
        if local
        else None
    )
    worst_macro = max(
        measured, key=lambda e: float(e["boundary"]["post_operator_heldout_relL2"])
    )
    block: dict[str, Any] = {
        "applicable": True,
        "local_solve_applicable": bool(local),
        "macrocell_endpoint": (
            f"worst-of-{len(measured)}:{worst_macro['component_id']}:"
            f"{worst_macro['boundary'].get('macrocell_endpoint')}"
        ),
        "write_heldout_relL2": (
            None if worst_write is None else worst_write["boundary"]["write_heldout_relL2"]
        ),
        "write_heldout_cos": (
            None if worst_write is None else worst_write["boundary"]["write_heldout_cos"]
        ),
        "post_operator_heldout_relL2": worst_macro["boundary"][
            "post_operator_heldout_relL2"
        ],
        "post_operator_heldout_cos": worst_macro["boundary"][
            "post_operator_heldout_cos"
        ],
        "endpoint_band_le_0p4": bool(
            all(bool(e["boundary"].get("endpoint_band_le_0p4")) for e in measured)
        ),
        "worst_write_component_id": (
            None if worst_write is None else str(worst_write["component_id"])
        ),
        "worst_macrocell_component_id": str(worst_macro["component_id"]),
        "components_measured": len(measured),
        "components_total": len(entries),
    }
    if not local:
        block["local_solve_inapplicable_reason"] = (
            "no folded component in this recipe has a calibrated parent-matmul "
            "reference for its write, so the chain's third link has no "
            "aggregate to gate on"
        )
    return block


def net_storage(
    parent_parameter_count: int,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exact stored-parameter accounting for the composed child.

    The per-component ratio a cell seals is `stored / spec.parent_parameter_count`
    and it is **not** additive, because a kernel may install fewer tensors than
    its component owns and the two live cases disagree about which set its
    stored count is measured over:

    * **C14** owns `operator_norm` and `ffn_norm`, refits only the first, and
      seals ``stored = 1024`` against ``owned = 2048`` -- 0.5x, which reads as a
      50% saving over a pair of tensors the child still stores in full
      (`NOTES-MX-F` §4.1 called it a bookkeeping artifact and it is one).  Its
      stored count covers the **installed** tensor.
    * **C10** owns all four attention projections, installs `q_proj` and
      `out_proj`, and seals ``stored = 3,014,656`` -- which explicitly *includes*
      the untouched `k_proj`/`v_proj` numels, because "the kv rows a dead group
      would free are only a saving if the child actually stops storing them, and
      this child stores them" (`matrix_technique_head_drop`).  Its stored count
      covers the **owned** set.

    Charging both to the same formula gets one of them wrong by 917,504
    parameters, which is the difference between a composed child that contracts
    and one that does not.  The coverage is therefore *derived*, mechanically,
    from the sealed numbers themselves: a stored count that exceeds the tensors
    actually installed cannot be a statement about only those tensors, so it is
    read as covering the owned set; otherwise it covers the installed set.  When
    a kernel installs everything it owns -- every single-tensor component --
    both readings coincide and the rule is inert.
    """

    covered_total = 0
    stored = 0
    rows: list[dict[str, Any]] = []
    for entry in entries:
        installed = int(entry["storage"]["installed_parameter_count"])
        owned = int(entry["storage"]["owned_parameter_count"])
        component_stored = int(entry["storage"]["stored_parameter_count"])
        if installed != owned and component_stored > installed:
            covered, basis = owned, "owned"
        else:
            covered, basis = installed, "installed"
        covered_total += covered
        stored += component_stored
        rows.append(
            {
                "component_id": str(entry["component_id"]),
                "cell_id": str(entry["isolation_cell_id"]),
                "installed_parameter_count": installed,
                "stored_parameter_count": component_stored,
                "owned_parameter_count": owned,
                "covered_parameter_count": covered,
                "coverage_basis": basis,
                "declared_ratio": entry["storage"].get("declared_ratio"),
                "effective_ratio": (
                    None if covered <= 0 else float(component_stored) / float(covered)
                ),
                "effective_delta": component_stored - covered,
            }
        )
    replaced = covered_total
    child = int(parent_parameter_count) - replaced + stored
    if int(parent_parameter_count) <= 0:
        raise MatrixCompositionError("The parent parameter count must be positive.")
    return {
        "parent_parameter_count": int(parent_parameter_count),
        "child_parameter_count": int(child),
        "net_ratio": float(child) / float(parent_parameter_count),
        "folded_tensors_parameter_count": replaced,
        "folded_tensors_stored_parameter_count": stored,
        "folded_tensors_ratio": (
            None if replaced <= 0 else float(stored) / float(replaced)
        ),
        "net_delta": child - int(parent_parameter_count),
        "contracts": bool(child < int(parent_parameter_count)),
        "per_component": rows,
    }


def receipt_error(message: str) -> MatrixCellReceiptError:  # pragma: no cover
    return MatrixCellReceiptError(message)
