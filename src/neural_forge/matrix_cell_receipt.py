#!/usr/bin/env python3
"""Sealed per-cell receipt for the stage-1 matrix campaign.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §3 fixes one schema for every cell
in the campaign so that ~300 heterogeneous cells stay comparable: one component,
one technique, one level, one site, one receipt.  This module is the schema and
nothing else -- it holds no torch import, opens no cache, and runs no solve, so
the technique lane and the harness lane can both depend on it without depending
on each other.

Three things live here and only here:

* **The mandatory field set.**  `validate_receipt` fails closed on a missing
  field rather than defaulting it.  That is the mechanism by which the N1/N2/N3
  serialization gaps are closed "by construction" (§3): a cell whose solve did
  not report a condition number cannot be sealed at all.
* **The failure-localization chain.**  `first_divergent_metric` walks the seven
  links (infra -> solve-conditioning -> local-solve -> macrocell-band ->
  composition -> free-run -> behavior) in the fixed order §3 registers, and
  returns the *first* one to cross its pre-registered threshold.  §5 rule 5
  forbids narrative failure causes; this walker is the only thing allowed to
  assign one.
* **The seal.**  `seal_receipt` content-addresses the canonical JSON body, so a
  receipt's identity is a function of its numbers and nothing else.

The metric policy of the closed-path ledger is enforced structurally: the only
fields `campaign_pass` reads are `behavior.*` and the `emit_zero_floor` wreck
detector.  `boundary.*` and `composed.terminal_*` are recorded because §5 rule 4
makes band-versus-behavior agreement a campaign output, and are unreachable from
any acceptance clause here.

Deviations from the design document, recorded per the campaign's own rule that
deviations are written down rather than silently absorbed:

1. `outcome.status` carries one extra value, ``pending-review``, beyond the
   design's ``pass|fail|timeout|infra``.  §5 rule 1 makes a *human* decode-smoke
   verdict necessary for a pass, and that verdict cannot exist at seal time.
   Without the extra value every unreviewed cell would have to be sealed either
   as an unearned ``pass`` or as a false ``fail``.  ``outcome.decode_smoke_review``
   carries the verdict itself and starts at ``pending``.
2. `boundary.macrocell_endpoint` is new.  §3 names the macrocell field
   ``post_operator_heldout_*`` while its own parenthetical says the score is
   "taken at the composed layer output".  Those are different trace endpoints
   for an MLP component.  The field names are kept frozen and the endpoint that
   was actually scored is named explicitly instead.
3. `cell_instance_id` is new, required by §3's own ledger rule that "a re-run of
   a cell gets a new cell instance id, never an overwrite".
4. `boundary.applicable` / `composed.applicable` exist so the control cell and
   the token-route calibration cell -- which have no component write to score --
   can declare a link not-applicable *explicitly*.  A link is never skipped
   because a field happened to be absent.
5. `calibration.applicable` and `boundary.local_solve_applicable` extend that
   same declared-inapplicability mechanism to the chain's *local* links, and
   were added in wave 2.  `NOTES-MX-E` §4.5 measured one harness gap four ways:
   the generic parent-calibration gate assumes the component is a single 2-D
   linear map ``held_input @ parent_weight.T``, which C03 (a 3-block in_proj),
   C04 (FIR taps), C10 (four tensors behind a softmax) and C14 (a 1-D RMSNorm
   gain) each violate differently, and the cells died at link 1 with no
   evidence at all.  A link that *cannot be measured for this component* is a
   different statement from a link that was measured and failed, and the
   receipt now carries both statements separately.  Declaring one costs a
   mandatory reason string, and the two flags are read by
   `first_divergent_metric` only -- never by `campaign_pass`, whose clause is
   still behavior plus the wreck detector and nothing else.
7. `cell_kind` carries one extra value, ``composition``, and `CellId` parses one
   extra literal form, ``COMP.<mode>[.<variant>]``.  ADDENDUM 4 pre-registers a
   composition wave whose unit is *many* components in one child, which is the
   one thing §1.1's ``component.technique.level.site`` id cannot name.  The
   extension is deliberately the smallest one that keeps every other invariant:
   `COMPONENTS` and `TECHNIQUES` are untouched (so `known_cell_ids`, the
   renderer's columns and the axis tests still enumerate exactly the design's
   axes), `ledger_line` gains no column (the mode is in the cell id, and
   ``stored_parameter_ratio`` is already the right place for the composed
   child's **net** child/parent storage), and the whole per-component story
   rides in one new mandatory block, ``composition``, which
   `_validate_composition` checks and which no acceptance clause reads.  The
   failure chain is unchanged and walks the aggregate the runner seals: worst
   write cosine, worst macrocell, global composed drift, composed free-run.
8. `LEVELS` carries the storage-defined family ``S2``/``S4``/``S8`` and a
   receipt may carry an ``addendum6`` block.  ADDENDUM 6 retires token match as
   a gate and puts a *storage bound* where the design had a width fraction: an
   ``S`` level names what the child may store, not how wide a factor is, and
   `STORAGE_LEVEL_RATIOS` is the bound.  The behaviour verdict rides in its own
   block on a review instance rather than redefining ``outcome.status``, so
   every receipt sealed before the addendum re-verifies its seal and recomputes
   the same chain it always did -- `campaign_pass` and `first_divergent_metric`
   are byte-for-byte the clauses waves 1-4 were sealed under.  The renderer, not
   the schema, is what leads with the new verdict.
6. `code_identity.compute_architecture` is new in wave 2 and mandatory.
   `NOTES-MX-E` §2: a batch was launched on what ``nvidia-smi`` calls the
   second RTX 3090, CUDA's default ``FASTEST_FIRST`` ordering resolved it to a
   Tesla P40 (sm_61), and *nothing in the receipt would have said so*.  A9's
   fp32 OMP candidate search is architecture-non-portable (`NOTES-MX-C` §4.4),
   so two cells measured on different compute architectures are not
   comparable, and the compute architecture was the one input to a cell that
   the sealed identities did not cover.  Receipts sealed before this field
   exists (wave 1) do not validate under this schema; they are preserved
   byte-for-byte as sealed and are read through the ledger, which is where
   their comparability claim was always recorded.
9. The schema is **versioned**, and ADDENDUM 7 (wave 7) is the first bump:
   ``…receipt-v1`` and ``…receipt-v2`` are both live schema strings and
   `RECEIPT_SCHEMA_VERSIONS` maps each to an integer.  v2 widens the site
   registry to every architecturally valid layer per component -- the parent's
   own ``layer_types``: short_conv at 0/1/3/5/7/9/11/13, full_attention at
   2/4/6/8/10/12, and a ``feed_forward`` in all fourteen -- and moves nothing
   else.  The mandatory field set, the failure chain, ``campaign_pass``, the
   seal and the canonical JSON are identical under both.  `validate_receipt`
   reads a body against the version the body itself declares, so the 348 rows
   waves 1-6 sealed re-verify under v1's narrow registry for ever; a v1 receipt
   is never re-validated against v2, and a v2 cell id is never accepted by a v1
   receipt.  `NOTES-MX-J` §5 is why: the wave-6 site ladder hit the registry
   wall after three rungs, and §5.1 measured the ceiling that policy implies at
   ~0.923x net with eleven of the fourteen layers unreachable.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256

#: The receipt schema is *versioned*, and both versions are live.  ADDENDUM 7
#: widens the site registry to every architecturally valid layer, which changes
#: exactly one thing about a receipt: which cell ids `CellId.parse` will accept.
#: A receipt sealed under v1 is validated against v1's narrow site policy, for
#: ever -- a sealed row is never re-read under a rule it was not sealed under --
#: and a receipt sealed under v2 is validated against v2's wide one.  Nothing
#: else moves: the mandatory field set, the failure chain, `campaign_pass` and
#: the seal are byte-for-byte identical between the two versions.
SCHEMA_V1 = "neural-forge-lfm2-matrix-cell-receipt-v1"
SCHEMA_V2 = "neural-forge-lfm2-matrix-cell-receipt-v2"
#: The version new receipts are sealed under.
SCHEMA = SCHEMA_V2
#: schema string -> integer version, the only mapping that decides which site
#: policy a body is read against.
RECEIPT_SCHEMA_VERSIONS: Mapping[str, int] = {SCHEMA_V1: 1, SCHEMA_V2: 2}
LATEST_SCHEMA_VERSION = 2
LEDGER_SCHEMA = "neural-forge-lfm2-matrix-cell-ledger-v1"


class MatrixCellReceiptError(RuntimeError):
    """A receipt is malformed, incomplete, or unsealed.  Always fatal."""


# --------------------------------------------------------------------------
# the axes (design §1.1)
# --------------------------------------------------------------------------

#: The parent's real layer-type map, read off `config.json`'s ``layer_types``
#: and re-checked against the shipped ``model.safetensors`` key set: fourteen
#: blocks, eight of them short_conv and six full_attention, and a SwiGLU
#: ``feed_forward`` in **every** one of the fourteen.  This is the fact
#: ADDENDUM 7's widening is derived from, so it is written down once, here,
#: rather than restated per component.
CONV_LAYERS: tuple[int, ...] = (0, 1, 3, 5, 7, 9, 11, 13)
ATTENTION_LAYERS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)
MLP_LAYERS: tuple[int, ...] = tuple(range(14))

#: component id -> (weight-group name, geometry note, site policy).
#: ``site policy`` is ``"global"`` for components with no layer index and a
#: tuple of permitted layer indices otherwise.
#:
#: **Schema v1 (waves 1-6).**  Sites were chosen, not exhaustive: L0/L13 are the
#: measured operator-damage extremes (S3), L5 is a quiet mid conv layer, and
#: L2/L8/L12 are the largest attention whole-layer gains.  Three sites per body
#: component, deliberately, because wave 1 had ~300 cells to fit in a budget.
#:
#: **Schema v2 (ADDENDUM 7, wave 7).**  That policy became the binding
#: constraint: `NOTES-MX-J` §5 climbed the C13 ladder to the end of the registry
#: in three rungs and could go no further, and §5.1 computed the ceiling the
#: policy implies -- ~0.923x net even if every registered site folded at a true
#: quarter, because eleven of the fourteen layers had no registered site at all.
#: v2 widens each component to **every architecturally valid layer** and changes
#: nothing else.  The two maps live side by side because a receipt is validated
#: against the version it was sealed under and never against a later one.
_COMPONENTS_V1_SITES: Mapping[str, Any] = {
    "C01": "global",
    "C02": "global",
    "C03": (0, 5, 13),
    "C04": (0, 5, 13),
    "C05": (0, 5, 13),
    "C06": (2, 8, 12),
    "C07": (2, 8, 12),
    "C08": (2, 8, 12),
    "C09": (2, 8, 12),
    "C10": (2, 8),
    "C11": (0, 5, 13),
    "C12": (0, 5, 13),
    "C13": (0, 5, 13),
    "C14": (0, "final"),
    "C15": "global",
}

COMPONENTS: Mapping[str, Mapping[str, Any]] = {
    "C01": {"name": "token_interface_in", "kind": "token", "sites": "global"},
    "C02": {"name": "token_interface_out", "kind": "token", "sites": "global"},
    "C03": {"name": "conv_in_proj", "kind": "short_conv", "sites": CONV_LAYERS},
    "C04": {"name": "conv_kernel", "kind": "short_conv", "sites": CONV_LAYERS},
    "C05": {"name": "conv_out_proj", "kind": "short_conv", "sites": CONV_LAYERS},
    "C06": {"name": "attn_q", "kind": "full_attention", "sites": ATTENTION_LAYERS},
    "C07": {"name": "attn_k", "kind": "full_attention", "sites": ATTENTION_LAYERS},
    "C08": {"name": "attn_v", "kind": "full_attention", "sites": ATTENTION_LAYERS},
    "C09": {"name": "attn_o", "kind": "full_attention", "sites": ATTENTION_LAYERS},
    "C10": {
        "name": "attn_per_head",
        "kind": "full_attention",
        "sites": ATTENTION_LAYERS,
    },
    "C11": {"name": "mlp_gate", "kind": "mlp", "sites": MLP_LAYERS},
    "C12": {"name": "mlp_up", "kind": "mlp", "sites": MLP_LAYERS},
    "C13": {"name": "mlp_down", "kind": "mlp", "sites": MLP_LAYERS},
    "C14": {"name": "norms", "kind": "norm", "sites": MLP_LAYERS + ("final",)},
    "C15": {"name": "lm_head", "kind": "token", "sites": "global"},
}


def component_sites(component: str, schema_version: int = LATEST_SCHEMA_VERSION) -> Any:
    """The site policy for one component **under one schema version**.

    The single place either map is read from, so that widening a component in
    v2 cannot accidentally widen what a v1 receipt is allowed to say.
    """

    if component not in COMPONENTS:
        raise MatrixCellReceiptError(f"Unknown component {component!r}.")
    version = int(schema_version)
    if version == 1:
        return _COMPONENTS_V1_SITES[component]
    if version == LATEST_SCHEMA_VERSION:
        return COMPONENTS[component]["sites"]
    raise MatrixCellReceiptError(
        f"Unknown receipt schema version {schema_version!r}; "
        f"known versions are {sorted(set(RECEIPT_SCHEMA_VERSIONS.values()))}."
    )


def schema_version_of(body: Mapping[str, Any]) -> int:
    """The integer schema version a receipt body declares.  Fails closed."""

    declared = body.get("schema") if isinstance(body, Mapping) else None
    version = RECEIPT_SCHEMA_VERSIONS.get(str(declared))
    if version is None:
        raise MatrixCellReceiptError(
            f"Receipt schema must be one of {sorted(RECEIPT_SCHEMA_VERSIONS)}; "
            f"got {declared!r}."
        )
    return version

#: technique id -> canonical technique name used in the receipt body.
TECHNIQUES: Mapping[str, str] = {
    "T1": "dense_prior_anchored_ridge",
    "T2": "cross_width_residual_charts",
    "T3": "balanced_basis_truncation",
    "T4": "low_rank_factor_writers",
    "T5": "sparse_omp_routes",
    "T6": "kind_translation",
    "T7": "attention_relation_matching",
    "T8": "gram_relation_matching",
    "T9": "progressive_multi_pass",
    # The control cell's "technique" is the absence of one.
    "copy_exact": "copy_exact",
}

#: level id -> marginal contraction of the component's own width.  ``NA`` is
#: the C14 case: norms are diagonal, so the cell tests standalone refit quality
#: (N10) at identity size and has no level axis.
#:
#: The marginal family is ``M_k = 1 - 1/k``: M16 keeps 15/16, M8 keeps 7/8, M4
#: keeps 3/4.  ``M2`` -- keep 1/2 -- is new in wave 2 and is the family's floor.
#: It exists because of `NOTES-MX-E` §4.1's storage arithmetic: a technique
#: whose stored form is a ``w x r`` basis *plus* an ``r x w`` solved factor
#: stores ``r * (in + out)`` against a dense ``in * out``, so on a square
#: component it does not contract at all until ``r < w/2``.  Every marginal
#: level above M2 therefore *costs* storage on a square component, and M2 is
#: the exact break-even.  Naming it makes the break-even a measurable cell
#: rather than a footnote.
#:
#: ADDENDUM 6 adds the **storage-defined** family ``S2``/``S4``/``S8``.  Every
#: level above names a *width* -- a fraction of the component's own axis, or the
#: width the exact-4x layout assigns it -- and waves 1-3 measured the gap that
#: opens between a width and a size: `C13.T3.M16.L05` contracts the width to
#: 15/16 and stores **3.28x** the dense tensor it replaced.  An ``S`` level
#: names the size instead: ``S4`` is satisfied only when the child stores at
#: most a quarter of the component's parent parameters, whatever width that
#: takes for the technique in question.  The bound lives in
#: `STORAGE_LEVEL_RATIOS` and is enforced against the receipt's own counts.
LEVELS: Mapping[str, Any] = {
    "M16": 15.0 / 16.0,
    "M8": 7.0 / 8.0,
    "M4": 3.0 / 4.0,
    "M2": 1.0 / 2.0,
    "X4": None,  # the width the canonical exact-4x layout assigns this component
    "S2": None,  # ADDENDUM 6: stored <= 0.5x  the component's parent parameters
    "S4": None,  # ADDENDUM 6: stored <= 0.25x -- the X4 proof bar
    "S8": None,  # ADDENDUM 6: stored <= 0.125x
    "NA": 1.0,
}

#: ADDENDUM 6 §6.2: level id -> the fraction of the component's parent
#: parameter count the child is allowed to store.  A cell at one of these
#: levels is a *storage claim*; a cell at any other level is not.
STORAGE_LEVEL_RATIOS: Mapping[str, float] = {
    "S2": 0.5,
    "S4": 0.25,
    "S8": 0.125,
}

CELL_KINDS = ("control", "calibration", "cell", "composition")
STATUSES = ("pass", "fail", "timeout", "infra", "pending-review")
REVIEW_VERDICTS = ("pending", "coherent", "incoherent")
TERMINATIONS = ("eos", "cap", "loop", "error")

#: ADDENDUM 6 §6.1's verdicts, in descending order.  These grade the *whole
#: model* with one folded component installed, read off the verbatim text of the
#: frozen 15-prompt review, and they are the campaign's primary grade from wave
#: 5 onward.  ``cannot-express-4x`` is deliberately not a behavioural verdict at
#: all: it says the technique's storage algebra cannot state the component at
#: the level, so nothing about the model was measured.
BEHAVIOR_VERDICTS = (
    "alive-parent-band",
    "alive-degraded",
    "impaired",
    "dead",
    "cannot-express-4x",
)

#: The two measured reference bands of ADDENDUM 6 §6.1, on the frozen surface.
PARENT_REVIEW_BAND = 12
A9_REVIEW_BAND = 8
#: ADDENDUM 2's measured reviewer-sensitivity interval on this 15-case surface.
PARENT_BAND_TOLERANCE = 2

CONTROL_CELL_ID = "CTRL.copy_exact"

#: ADDENDUM 4's three pre-registered composition modes, in the receipt's own
#: spelling.  ``M-A`` installs every isolation winner as solved against parent
#: evidence; ``M-B`` re-solves each winner in topological order against evidence
#: recaptured through the partially folded child; ``M-C`` is the per-component
#: hybrid, run only if A and B disagree.
COMPOSITION_MODES: Mapping[str, str] = {
    "MA": "M-A",
    "MB": "M-B",
    "MC": "M-C",
}
COMPOSITION_COMPONENT = "COMP"
COMPOSITION_TECHNIQUE = "composition"
COMPOSITION_TECHNIQUE_NAME = "heterogeneous_composition"

_SITE = re.compile(r"\A(?:L(?P<layer>\d{2})|GLOBAL|FINAL)\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
#: ``COMP.MA`` or ``COMP.MA.RSTO`` -- the optional trailing token names a
#: *recipe variant*, because ADDENDUM 1's selection rule (highest free-run) and
#: the storage-honesty criterion can name different winners for the same
#: component, and running both under the same mode is the only way to measure
#: what the difference costs.
_COMPOSITION_ID = re.compile(
    r"\ACOMP\.(?P<mode>M[ABC])(?:\.(?P<variant>[A-Z0-9]{1,12}))?\Z"
)


# --------------------------------------------------------------------------
# pre-registered thresholds (design §3 failure-localization convention)
# --------------------------------------------------------------------------

#: link 2 -- a clamp that binds is only a *conditioning* failure when the solve
#: asked for materially more than the budget allowed.  §3 names 4x.
SOLVE_CLAMP_RATIO_CAP = 4.0
#: link 3 -- the component write must retain direction.
LOCAL_SOLVE_MIN_WRITE_COS = 0.95
#: link 4 -- the S8 macrocell band.
MACROCELL_MAX_RELATIVE_L2 = 0.4
#: link 5 -- composed terminal drift against the all-copy-exact control child.
COMPOSITION_MAX_CONTROL_DELTA = 0.02
#: link 6 -- §5 rule 1's behavioral clause, the only threshold an acceptance
#: clause is permitted to read.
FREE_RUN_MIN_TOKEN_MATCH = 0.90

CHAIN_LINKS = (
    "infra",
    "solve-conditioning",
    "local-solve",
    "macrocell-band",
    "composition",
    "free-run",
    "behavior",
)


# --------------------------------------------------------------------------
# cell identity
# --------------------------------------------------------------------------


class CellId:
    """``component.technique.level[.site]``, plus the two calibration forms.

    The design writes cell ids three ways -- ``C13.T5.M16.L05`` for a body cell,
    ``CAL.C01.T5.X4`` for a calibration cell on a global component, and the
    literal ``CTRL.copy_exact`` for the control child.  Parsing all three here
    keeps the site inside the cell id rather than promoting it to a fourth free
    axis, which is the design's explicit choice (§1.1).
    """

    __slots__ = (
        "calibration",
        "component",
        "composition_mode",
        "composition_variant",
        "level",
        "site",
        "technique",
    )

    def __init__(
        self,
        *,
        component: str,
        technique: str,
        level: str,
        site: str,
        calibration: bool = False,
        composition_mode: str | None = None,
        composition_variant: str | None = None,
    ) -> None:
        self.component = component
        self.technique = technique
        self.level = level
        self.site = site
        self.calibration = calibration
        self.composition_mode = composition_mode
        self.composition_variant = composition_variant

    @property
    def kind(self) -> str:
        if self.composition_mode is not None:
            return "composition"
        if self.technique == "copy_exact":
            return "control"
        return "calibration" if self.calibration else "cell"

    @property
    def layer_index(self) -> int | None:
        matched = _SITE.match(self.site)
        if matched is None or matched.group("layer") is None:
            return None
        return int(matched.group("layer"))

    @property
    def text(self) -> str:
        if self.composition_mode is not None:
            body = f"{COMPOSITION_COMPONENT}.{self.composition_mode}"
            if self.composition_variant:
                body = f"{body}.{self.composition_variant}"
            return body
        if self.kind == "control":
            return CONTROL_CELL_ID
        body = f"{self.component}.{self.technique}.{self.level}"
        if self.site not in ("GLOBAL",):
            body = f"{body}.{self.site}"
        return f"CAL.{body}" if self.calibration else body

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CellId({self.text!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CellId) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)

    @classmethod
    def parse(
        cls, value: str, *, schema_version: int = LATEST_SCHEMA_VERSION
    ) -> "CellId":
        """Parse a cell id, enforcing ``schema_version``'s site registry.

        The version is a parameter and not a module constant because ADDENDUM 7
        widened the registry: a cell id naming L04 is legal under v2 and was
        never legal under v1, and `validate_receipt` reads a receipt against the
        version that receipt declares so that a sealed v1 row keeps meaning
        exactly what it meant when it was sealed.  Everything that is *writing*
        new work -- recipes, runners, the renderer's empty-cell enumeration --
        gets the latest version by default, which is the whole point of the
        wave.
        """

        text = str(value).strip()
        if text == CONTROL_CELL_ID:
            return cls(
                component="CTRL",
                technique="copy_exact",
                level="NA",
                site="GLOBAL",
            )
        composed = _COMPOSITION_ID.match(text)
        if composed is not None:
            # A composed child has no single component, technique, level or
            # site -- that is what makes it the composition wave.  The axis
            # fields are held at the same sentinels the control cell uses, and
            # every real per-component axis rides in the mandatory
            # ``composition.components[]`` provenance array instead.
            return cls(
                component=COMPOSITION_COMPONENT,
                technique=COMPOSITION_TECHNIQUE,
                level="NA",
                site="GLOBAL",
                composition_mode=composed.group("mode"),
                composition_variant=composed.group("variant"),
            )
        calibration = False
        if text.startswith("CAL."):
            calibration = True
            text = text[len("CAL.") :]
        parts = text.split(".")
        if len(parts) == 3:
            component, technique, level = parts
            site = "GLOBAL"
        elif len(parts) == 4:
            component, technique, level, site = parts
        else:
            raise MatrixCellReceiptError(
                f"Cell id {value!r} is not component.technique.level[.site]."
            )
        if component not in COMPONENTS:
            raise MatrixCellReceiptError(f"Unknown component {component!r}.")
        if technique not in TECHNIQUES or technique == "copy_exact":
            raise MatrixCellReceiptError(f"Unknown technique {technique!r}.")
        if level not in LEVELS:
            raise MatrixCellReceiptError(f"Unknown level {level!r}.")
        if _SITE.match(site) is None:
            raise MatrixCellReceiptError(
                f"Site {site!r} must be L<nn>, GLOBAL, or FINAL."
            )
        sites = component_sites(component, schema_version)
        if sites == "global":
            if site != "GLOBAL":
                raise MatrixCellReceiptError(
                    f"{component} is a global component; site must be GLOBAL."
                )
        else:
            allowed = {
                "FINAL" if entry == "final" else f"L{int(entry):02d}"
                for entry in sites
            }
            if site not in allowed:
                raise MatrixCellReceiptError(
                    f"{component} is measured at {sorted(allowed)} under receipt "
                    f"schema v{int(schema_version)}, not {site!r}."
                )
        if component == "C14" and level != "NA":
            raise MatrixCellReceiptError(
                "C14 norms are diagonal; the only level is NA (N10 standalone refit)."
            )
        if component != "C14" and level == "NA":
            raise MatrixCellReceiptError(f"{component} requires a real level.")
        return cls(
            component=component,
            technique=technique,
            level=level,
            site=site,
            calibration=calibration,
        )


# --------------------------------------------------------------------------
# mandatory field specification
# --------------------------------------------------------------------------

#: ``dotted.path -> accepted python types``.  ``type(None)`` in the tuple means
#: the design's own schema shows ``null`` as a legal value for that field.
_MANDATORY: Mapping[str, tuple[type, ...]] = {
    "schema": (str,),
    "cell_id": (str,),
    "cell_instance_id": (str,),
    "cell_kind": (str,),
    "component": (str,),
    "technique": (str,),
    "level": (str,),
    "site": (str,),
    "campaign_wave": (int,),
    "code_identity.repo_head": (str,),
    "code_identity.src_tree_sha": (str,),
    "code_identity.compute_architecture": (str,),
    "evidence.parent_trace_cache_sha": (str, type(None)),
    "evidence.chart_sha": (str, type(None)),
    "evidence.probe_set_sha": (str,),
    "evidence.smoke_prompt_sha": (str,),
    "evidence.heldout_split.fit_rows": (int,),
    "evidence.heldout_split.heldout_rows": (int,),
    "evidence.parent_model_forward_calls": (int,),
    "evidence.backward_calls": (int,),
    "calibration.copy_exact_reproduction_relL2": (float, int, type(None)),
    "calibration.passed": (bool,),
    "calibration.applicable": (bool,),
    "solve.solver_path": (str, type(None)),
    "solve.ridge_relative": (float, int, type(None)),
    "solve.condition_number": (float, int, type(None)),
    "solve.condition_cap_saturated": (bool,),
    "solve.prior_norm": (float, int, type(None)),
    "solve.correction_norm": (float, int, type(None)),
    "solve.requested_clamp_ratio": (float, int, type(None)),
    "solve.allowed_clamp_ratio": (float, int, type(None)),
    "solve.clamp_engaged": (bool,),
    "boundary.applicable": (bool,),
    "boundary.local_solve_applicable": (bool,),
    "boundary.macrocell_endpoint": (str, type(None)),
    "boundary.write_heldout_relL2": (float, int, type(None)),
    "boundary.write_heldout_cos": (float, int, type(None)),
    "boundary.post_operator_heldout_relL2": (float, int, type(None)),
    "boundary.post_operator_heldout_cos": (float, int, type(None)),
    "boundary.endpoint_band_le_0p4": (bool, type(None)),
    "composed.applicable": (bool,),
    "composed.terminal_heldout_relL2": (float, int, type(None)),
    "composed.emit_zero_floor_passed": (bool, type(None)),
    "composed.vs_copy_exact_control_delta": (float, int, type(None)),
    "composed.norm_ratio_per_layer_minmax": (list, type(None)),
    "behavior.free_run.probe_prompts": (int,),
    "behavior.free_run.tokens_per_prompt": (int,),
    "behavior.free_run.token_match_vs_control": (float, int, type(None)),
    "behavior.free_run.mean_lcp": (float, int, type(None)),
    "behavior.decode_smoke": (list,),
    "outcome.status": (str,),
    "outcome.first_divergent_metric": (str, type(None)),
    "outcome.decode_smoke_review": (str,),
    "outcome.notes": (str,),
    "wall_time_seconds": (float, int),
    "sealed_at": (str,),
}


def _lookup(body: Mapping[str, Any], path: str) -> Any:
    node: Any = body
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise MatrixCellReceiptError(f"Receipt is missing {path!r}.")
        node = node[key]
    return node


def _number(body: Mapping[str, Any], path: str) -> float:
    value = _lookup(body, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixCellReceiptError(
            f"Receipt field {path!r} must be a number for the failure chain."
        )
    return float(value)


def _flag(body: Mapping[str, Any], path: str) -> bool:
    value = _lookup(body, path)
    if not isinstance(value, bool):
        raise MatrixCellReceiptError(f"Receipt field {path!r} must be a boolean.")
    return value


def validate_receipt(body: Mapping[str, Any]) -> None:
    """Fail closed on anything the campaign needs and this receipt lacks."""

    if not isinstance(body, Mapping):
        raise MatrixCellReceiptError("A receipt must be a mapping.")
    # ADDENDUM 7.  The version is read off the body, never assumed: a v1 receipt
    # is validated against v1's narrow site registry for as long as it exists,
    # so re-verifying any of the 348 rows waves 1-6 sealed recomputes exactly
    # the clauses they were sealed under.
    schema_version = schema_version_of(body)
    for path, types in _MANDATORY.items():
        value = _lookup(body, path)
        if isinstance(value, bool) and bool not in types:
            raise MatrixCellReceiptError(
                f"Receipt field {path!r} is a boolean; expected {types}."
            )
        if not isinstance(value, types):
            raise MatrixCellReceiptError(
                f"Receipt field {path!r} has type {type(value).__name__}; "
                f"expected one of {tuple(t.__name__ for t in types)}."
            )
    cell = CellId.parse(str(body["cell_id"]), schema_version=schema_version)
    if body["cell_kind"] not in CELL_KINDS:
        raise MatrixCellReceiptError(f"cell_kind must be one of {CELL_KINDS}.")
    if body["cell_kind"] != cell.kind:
        raise MatrixCellReceiptError(
            f"cell_kind {body['cell_kind']!r} disagrees with cell id {cell.text!r}."
        )
    if body["outcome"]["status"] not in STATUSES:
        raise MatrixCellReceiptError(f"outcome.status must be one of {STATUSES}.")
    if body["outcome"]["decode_smoke_review"] not in REVIEW_VERDICTS:
        raise MatrixCellReceiptError(
            f"outcome.decode_smoke_review must be one of {REVIEW_VERDICTS}."
        )
    divergent = body["outcome"]["first_divergent_metric"]
    if divergent is not None and divergent not in CHAIN_LINKS:
        raise MatrixCellReceiptError(
            f"outcome.first_divergent_metric must be one of {CHAIN_LINKS} or null."
        )
    smoke = body["behavior"]["decode_smoke"]
    if not isinstance(smoke, list):
        raise MatrixCellReceiptError("behavior.decode_smoke must be a list.")
    for index, entry in enumerate(smoke):
        if not isinstance(entry, Mapping):
            raise MatrixCellReceiptError(
                f"behavior.decode_smoke[{index}] must be a mapping."
            )
        for field in ("prompt_sha", "output_verbatim", "terminated"):
            if field not in entry:
                raise MatrixCellReceiptError(
                    f"behavior.decode_smoke[{index}] is missing {field!r}."
                )
        if entry["terminated"] not in TERMINATIONS:
            raise MatrixCellReceiptError(
                f"behavior.decode_smoke[{index}].terminated must be one of "
                f"{TERMINATIONS}."
            )
    minmax = body["composed"]["norm_ratio_per_layer_minmax"]
    if minmax is not None and (
        len(minmax) != 2 or any(isinstance(v, bool) for v in minmax)
    ):
        raise MatrixCellReceiptError(
            "composed.norm_ratio_per_layer_minmax must be null or [min, max]."
        )
    for path in ("code_identity.src_tree_sha", "evidence.probe_set_sha"):
        value = _lookup(body, path)
        if _SHA256.match(str(value)) is None:
            raise MatrixCellReceiptError(f"{path} must be a sha256 hex digest.")
    architecture = str(_lookup(body, "code_identity.compute_architecture")).strip()
    if not architecture:
        raise MatrixCellReceiptError(
            "code_identity.compute_architecture is empty.  A cell that does not "
            "record the compute architecture it ran on cannot be compared with "
            "one that does (NOTES-MX-E 2)."
        )
    _validate_declared_inapplicability(body)
    _validate_composition(body, cell)


#: Every per-component entry in ``composition.components[]`` carries these.
#: They are mandatory for the same reason §3's own fields are: a composed child
#: whose receipt cannot say which technique solved which component against
#: which evidence is not a localizable result, and ADDENDUM 4's entire point is
#: that a composition failure must localize to the component that broke it.
_COMPOSITION_COMPONENT_FIELDS = (
    "component_id",
    "component",
    "technique_id",
    "technique",
    "level",
    "site",
    "isolation_cell_id",
    "selection_reason",
    "weight_keys",
    "evidence",
    "solve",
    "boundary",
    "calibration",
    "after_install",
    "storage",
    "first_divergent_metric",
)


def _validate_composition(body: Mapping[str, Any], cell: "CellId") -> None:
    """A composition receipt carries a complete per-component provenance array.

    Checked structurally rather than trusted, because the composition wave's
    deliverable *is* the provenance: which winner was installed, in what order,
    solved against which evidence, and what each install cost.  A missing entry
    would turn "the winners do not compose" into an unattributable sentence.
    """

    block = body.get("composition")
    if cell.kind != "composition":
        if block is not None:
            raise MatrixCellReceiptError(
                "Only a COMP.<mode> cell may carry a 'composition' block; "
                f"{cell.text} is a {cell.kind} cell."
            )
        return
    if not isinstance(block, Mapping):
        raise MatrixCellReceiptError(
            "A composition cell must carry a 'composition' block."
        )
    if block.get("mode") not in COMPOSITION_MODES.values():
        raise MatrixCellReceiptError(
            f"composition.mode must be one of {sorted(COMPOSITION_MODES.values())}."
        )
    if block["mode"] != COMPOSITION_MODES[str(cell.composition_mode)]:
        raise MatrixCellReceiptError(
            f"composition.mode {block['mode']!r} disagrees with cell id "
            f"{cell.text!r}."
        )
    components = block.get("components")
    if not isinstance(components, list):
        raise MatrixCellReceiptError("composition.components must be a list.")
    if not components and not str(block.get("incomplete_reason") or "").strip():
        # The one legal empty case is a run that died or timed out before its
        # first install completed.  It still has to seal -- a cell that dies
        # silently costs the campaign a row *and* the reason (§2.3) -- so the
        # emptiness is admitted only with a stated reason, never by default.
        raise MatrixCellReceiptError(
            "composition.components is empty with no "
            "composition.incomplete_reason; a composed child with no folded "
            "component is the control cell, not a composition."
        )
    order = block.get("install_order")
    if not isinstance(order, list) or len(order) != len(components):
        raise MatrixCellReceiptError(
            "composition.install_order must name exactly one cell id per "
            "component, in install order."
        )
    owned: dict[str, str] = {}
    for index, entry in enumerate(components):
        if not isinstance(entry, Mapping):
            raise MatrixCellReceiptError(
                f"composition.components[{index}] must be a mapping."
            )
        missing = [f for f in _COMPOSITION_COMPONENT_FIELDS if f not in entry]
        if missing:
            raise MatrixCellReceiptError(
                f"composition.components[{index}] is missing {missing}."
            )
        if str(entry["isolation_cell_id"]) != str(order[index]):
            raise MatrixCellReceiptError(
                f"composition.install_order[{index}] is "
                f"{order[index]!r} but components[{index}] is "
                f"{entry['isolation_cell_id']!r}."
            )
        divergent = entry["first_divergent_metric"]
        if divergent is not None and divergent not in CHAIN_LINKS:
            raise MatrixCellReceiptError(
                f"composition.components[{index}].first_divergent_metric must "
                f"be one of {CHAIN_LINKS} or null."
            )
        keys = entry["weight_keys"]
        if not isinstance(keys, list) or not keys:
            raise MatrixCellReceiptError(
                f"composition.components[{index}].weight_keys must be a "
                "non-empty list of the parent tensors this fold replaced."
            )
        for key in keys:
            # Two components that own the same parent tensor would each be
            # scored against evidence the other invalidated, and the second
            # install would silently erase the first.  A recipe that does that
            # is a bug in the recipe, and this is where it is caught.
            if str(key) in owned:
                raise MatrixCellReceiptError(
                    f"composition.components[{index}] "
                    f"({entry['component_id']}) installs {key!r}, already "
                    f"installed by {owned[str(key)]}; a composed child folds "
                    "each parent tensor at most once."
                )
            owned[str(key)] = str(entry["component_id"])
    storage = block.get("storage")
    if not isinstance(storage, Mapping):
        raise MatrixCellReceiptError("composition.storage is mandatory.")
    for field in (
        "child_parameter_count",
        "parent_parameter_count",
        "net_ratio",
    ):
        value = storage.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MatrixCellReceiptError(
                f"composition.storage.{field} must be a number; a composed "
                "child that cannot state what it stores has not been measured "
                "on the axis a fold is ultimately judged on."
            )
    if not isinstance(block.get("passed_through"), list):
        raise MatrixCellReceiptError(
            "composition.passed_through is mandatory (it may be empty): a "
            "component left unfolded is a recorded decision, never a silence."
        )


def _validate_declared_inapplicability(body: Mapping[str, Any]) -> None:
    """A skipped chain link must say *why*, and cannot be skipped by accident.

    Declared inapplicability is the only mechanism that can stop
    `first_divergent_metric` from walking a link, so it is the only mechanism
    that could silently retire a gate.  Two rules keep it honest:

    * a declaration carries a nonempty reason in the same block, so a receipt
      reader sees the category error rather than a bare ``false``;
    * the boundary's two links nest -- ``local_solve_applicable`` cannot be
      true when the boundary as a whole did not run, because that would claim a
      measurement inside a stage that never happened.
    """

    if not _flag(body, "calibration.applicable"):
        reason = body["calibration"].get("inapplicable_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MatrixCellReceiptError(
                "calibration.applicable is false with no "
                "calibration.inapplicable_reason; the chain's first link is "
                "never skipped without a recorded reason."
            )
    boundary = body["boundary"]
    if not _flag(body, "boundary.applicable"):
        if _flag(body, "boundary.local_solve_applicable"):
            raise MatrixCellReceiptError(
                "boundary.local_solve_applicable is true while "
                "boundary.applicable is false; the boundary stage did not run."
            )
    elif not _flag(body, "boundary.local_solve_applicable"):
        reason = boundary.get("local_solve_inapplicable_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MatrixCellReceiptError(
                "boundary.local_solve_applicable is false with no "
                "boundary.local_solve_inapplicable_reason; the chain's third "
                "link is never skipped without a recorded reason."
            )


# --------------------------------------------------------------------------
# the failure-localization chain (design §3)
# --------------------------------------------------------------------------


def first_divergent_metric(body: Mapping[str, Any]) -> str | None:
    """Return the first chain link this receipt's numbers cross, or ``None``.

    Mechanical by construction: §5 rule 5 forbids a narrative cause, so this is
    the only place a failure is attributed and it reads nothing but the sealed
    numbers.  A link is skipped only when the receipt *declares* it
    inapplicable (``boundary.applicable`` / ``composed.applicable`` false); a
    missing field is an error, never a skip, which is what makes the N1/N2/N3
    serialization gaps unclosable-by-omission.
    """

    validate_receipt(body)

    # 1. infra -- the cell is void, not evidence.  A component whose parent map
    #    is not a single 2-D matmul (C03/C04/C10/C14) cannot be run through the
    #    generic ``held_input @ parent_weight.T`` gate at all, and wave 1 sealed
    #    four such cells as void for a harness reason rather than a technique
    #    one.  Those cells declare the gate inapplicable, with a reason, and
    #    rest their verdict on the links that need no per-component predictor:
    #    the macrocell, the composed forward, free-run and the review.
    if _flag(body, "calibration.applicable") and not _flag(body, "calibration.passed"):
        return "infra"

    # 2. solve-conditioning.  A clamp that binds is only a conditioning failure
    #    when the solve asked for materially more than the budget allowed; a
    #    saturated condition cap always is.
    if _flag(body, "solve.condition_cap_saturated"):
        return "solve-conditioning"
    if _flag(body, "solve.clamp_engaged"):
        requested = _lookup(body, "solve.requested_clamp_ratio")
        allowed = _lookup(body, "solve.allowed_clamp_ratio")
        if requested is None or allowed is None:
            raise MatrixCellReceiptError(
                "solve.clamp_engaged is true but the requested/allowed clamp "
                "ratios were not serialized; the chain cannot be walked."
            )
        allowed_value = float(allowed)
        if allowed_value <= 0.0:
            return "solve-conditioning"
        if float(requested) / allowed_value > SOLVE_CLAMP_RATIO_CAP:
            return "solve-conditioning"

    # 3/4. local-solve and macrocell-band, both boundary-side.  The write-cosine
    #      link needs a per-component predictor against a named write endpoint;
    #      the macrocell link needs only a delta that composes into the residual.
    #      They are declared separately because a component can have the second
    #      and not the first.
    if _flag(body, "boundary.applicable"):
        if _flag(body, "boundary.local_solve_applicable"):
            if _number(body, "boundary.write_heldout_cos") < LOCAL_SOLVE_MIN_WRITE_COS:
                return "local-solve"
        macrocell = _number(body, "boundary.post_operator_heldout_relL2")
        if macrocell > MACROCELL_MAX_RELATIVE_L2:
            return "macrocell-band"

    # 5. composition.
    if _flag(body, "composed.applicable"):
        delta = _number(body, "composed.vs_copy_exact_control_delta")
        if delta > COMPOSITION_MAX_CONTROL_DELTA:
            return "composition"

    # 6. free-run.  A cell can pass 1-5 and fail here: that outcome is the F3
    #    signature and is the most information-dense failure class in the
    #    campaign, which is why it is tagged rather than folded into 5.
    match = _lookup(body, "behavior.free_run.token_match_vs_control")
    if match is None:
        raise MatrixCellReceiptError(
            "behavior.free_run.token_match_vs_control is null; the behavioral "
            "link cannot be skipped -- it is the only acceptance signal."
        )
    if float(match) < FREE_RUN_MIN_TOKEN_MATCH:
        return "free-run"

    # 7. behavior -- the frozen review rubric, read from the verbatim text.
    if body["outcome"]["decode_smoke_review"] == "incoherent":
        return "behavior"
    return None


def campaign_pass(body: Mapping[str, Any]) -> bool:
    """§5 rule 1, verbatim.  Reads behavioral fields and the wreck detector only.

    Boundary metrics appear nowhere in this clause, and cannot: the only paths
    touched below are ``behavior.*`` and ``composed.emit_zero_floor_passed``.
    """

    validate_receipt(body)
    match = _lookup(body, "behavior.free_run.token_match_vs_control")
    if match is None or float(match) < FREE_RUN_MIN_TOKEN_MATCH:
        return False
    if body["outcome"]["decode_smoke_review"] != "coherent":
        return False
    floor = _lookup(body, "composed.emit_zero_floor_passed")
    # The wreck detector is necessary, never sufficient, and never headlined.
    # A cell that did not compose (the token-route calibration cell) declares
    # composition inapplicable and is not asked to clear a floor it never ran.
    if _flag(body, "composed.applicable") and not bool(floor):
        return False
    return True


def resolve_outcome(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``outcome`` block implied by this receipt's own numbers."""

    divergent = first_divergent_metric(body)
    review = body["outcome"]["decode_smoke_review"]
    if divergent == "infra":
        status = "infra"
    elif divergent is not None:
        status = "fail"
    elif review == "coherent":
        status = "pass"
    else:
        status = "pending-review"
    return {
        "status": status,
        "first_divergent_metric": divergent,
        "decode_smoke_review": review,
        "notes": body["outcome"].get("notes", ""),
    }


# --------------------------------------------------------------------------
# construction and sealing
# --------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def blank_receipt(
    *,
    cell_id: str,
    cell_instance_id: str,
    campaign_wave: int,
    repo_head: str,
    src_tree_sha: str,
    probe_set_sha: str,
    smoke_prompt_sha: str,
    compute_architecture: str,
    schema_version: int = LATEST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return a fully-keyed receipt body with every mandatory field present.

    Every numeric slot starts at ``None`` rather than ``0.0``.  A cell that
    forgot to measure something therefore fails the chain walk loudly instead
    of quietly claiming a perfect zero, which is the same fail-closed posture
    the layer-0 study's calibration gate takes.

    ``schema_version`` defaults to the latest because a new receipt is new work
    and new work gets ADDENDUM 7's widened site registry.  It is settable so a
    replay of a wave-1..6 cell can be sealed under the version that wave used.
    """

    schema = {1: SCHEMA_V1, LATEST_SCHEMA_VERSION: SCHEMA_V2}.get(
        int(schema_version)
    )
    if schema is None:
        raise MatrixCellReceiptError(
            f"Unknown receipt schema version {schema_version!r}."
        )
    cell = CellId.parse(cell_id, schema_version=int(schema_version))
    if cell.kind == "control":
        component = "copy_exact"
        technique = TECHNIQUES[cell.technique]
    elif cell.kind == "composition":
        component = "composed_child"
        technique = COMPOSITION_TECHNIQUE_NAME
    else:
        component = str(COMPONENTS[cell.component]["name"])
        technique = TECHNIQUES[cell.technique]
    return {
        "schema": schema,
        "cell_id": cell.text,
        "cell_instance_id": str(cell_instance_id),
        "cell_kind": cell.kind,
        "component": component,
        "technique": technique,
        "level": cell.level,
        "site": cell.site,
        "campaign_wave": int(campaign_wave),
        "code_identity": {
            "repo_head": str(repo_head),
            "src_tree_sha": str(src_tree_sha),
            "compute_architecture": str(compute_architecture),
        },
        "evidence": {
            "parent_trace_cache_sha": None,
            "chart_sha": None,
            "probe_set_sha": str(probe_set_sha),
            "smoke_prompt_sha": str(smoke_prompt_sha),
            "heldout_split": {"fit_rows": 0, "heldout_rows": 0},
            "parent_model_forward_calls": 0,
            "backward_calls": 0,
        },
        "calibration": {
            "copy_exact_reproduction_relL2": None,
            "passed": False,
            "applicable": True,
        },
        "solve": {
            "solver_path": None,
            "ridge_relative": None,
            "condition_number": None,
            "condition_cap_saturated": False,
            "prior_norm": None,
            "correction_norm": None,
            "requested_clamp_ratio": None,
            "allowed_clamp_ratio": None,
            "clamp_engaged": False,
            "omp": None,
        },
        "boundary": {
            "applicable": False,
            "local_solve_applicable": False,
            "macrocell_endpoint": None,
            "write_heldout_relL2": None,
            "write_heldout_cos": None,
            "post_operator_heldout_relL2": None,
            "post_operator_heldout_cos": None,
            "endpoint_band_le_0p4": None,
        },
        "composed": {
            "applicable": False,
            "terminal_heldout_relL2": None,
            "emit_zero_floor_passed": None,
            "vs_copy_exact_control_delta": None,
            "norm_ratio_per_layer_minmax": None,
        },
        "behavior": {
            "free_run": {
                "probe_prompts": 0,
                "tokens_per_prompt": 0,
                "token_match_vs_control": None,
                "mean_lcp": None,
            },
            "decode_smoke": [],
        },
        "outcome": {
            "status": "fail",
            "first_divergent_metric": None,
            "decode_smoke_review": "pending",
            "notes": "",
        },
        "wall_time_seconds": 0.0,
        "sealed_at": _now(),
        "receipt_sha": "self",
    }


def seal_receipt(body: Mapping[str, Any], *, resolve: bool = True) -> dict[str, Any]:
    """Validate, resolve the outcome, content-address, and return the receipt.

    The digest is taken over the canonical JSON of the body with
    ``receipt_sha`` held at the literal ``"self"``, so the sealed receipt's
    digest is reproducible from the sealed file by resetting one field.
    """

    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["sealed_at"] = sealed.get("sealed_at") or _now()
    sealed["receipt_sha"] = "self"
    validate_receipt(sealed)
    if resolve:
        outcome = resolve_outcome(sealed)
        sealed["outcome"] = {
            "status": outcome["status"],
            "first_divergent_metric": outcome["first_divergent_metric"],
            "decode_smoke_review": outcome["decode_smoke_review"],
            "notes": outcome["notes"],
        }
    validate_receipt(sealed)
    sealed["receipt_sha"] = canonical_sha256(sealed)
    return sealed


def timeout_receipt(
    body: Mapping[str, Any],
    *,
    wall_time_seconds: float,
    stage: str,
    hard_cap_seconds: float,
) -> dict[str, Any]:
    """Seal a cell that exceeded the §2.3 hard cap instead of dying silently.

    A cell that runs past the cap yields.  The receipt still carries every
    mandatory field so the ledger stays uniform and the renderer does not have
    to special-case a truncated row.
    """

    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["wall_time_seconds"] = float(wall_time_seconds)
    sealed["outcome"] = {
        "status": "timeout",
        "first_divergent_metric": None,
        "decode_smoke_review": sealed["outcome"].get("decode_smoke_review", "pending"),
        "notes": (
            f"Cell exceeded the {hard_cap_seconds:.0f}s hard cap during stage "
            f"{stage!r} and yielded (design 2.3)."
        ),
    }
    sealed["sealed_at"] = _now()
    sealed["receipt_sha"] = "self"
    validate_receipt(sealed)
    sealed["receipt_sha"] = canonical_sha256(sealed)
    return sealed


def seal_review(
    body: Mapping[str, Any],
    *,
    verdict: str,
    reviewer: str,
    instance_suffix: str = "review",
    notes: str = "",
) -> dict[str, Any]:
    """Seal a reviewed copy of a cell as a **new instance**, never an overwrite.

    §5 rule 1 makes a human decode-smoke verdict necessary for a pass, and that
    verdict cannot exist when the cell is sealed.  Two things then have to be
    true at once: the runner must not be able to certify its own behavior, and
    the campaign must still be able to reach ``pass``.  Splitting the review
    into a separate sealed instance satisfies both -- the reviewed receipt
    carries the digest of the receipt it was read from, so a review is always
    traceable to the exact verbatim text it graded, and the original instance
    stays in the append-only ledger untouched.
    """

    if verdict not in ("coherent", "incoherent"):
        raise MatrixCellReceiptError(
            "A review verdict is 'coherent' or 'incoherent'; 'pending' is the "
            "absence of a review, not a review."
        )
    reviewed_from = verify_seal(body)
    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["cell_instance_id"] = f"{sealed['cell_instance_id']}-{instance_suffix}"
    sealed["outcome"] = {
        "status": sealed["outcome"]["status"],
        "first_divergent_metric": sealed["outcome"]["first_divergent_metric"],
        "decode_smoke_review": verdict,
        "notes": (
            f"decode-smoke review by {reviewer} of receipt {reviewed_from[:16]}"
            + (f": {notes}" if notes else "")
        ),
    }
    sealed["review"] = {
        "reviewer": str(reviewer),
        "verdict": verdict,
        "reviewed_from_receipt_sha": reviewed_from,
        "reviewed_at": _now(),
        "rubric": "frozen whole-response review; cap-hit outputs censored, not failed",
    }
    sealed["sealed_at"] = _now()
    return seal_receipt(sealed)


def behavior_verdict_for(
    *,
    useful_or_partial: int,
    coherent: bool,
    emit_zero_floor_passed: bool | None,
    degenerate: bool,
) -> str:
    """ADDENDUM 6 §6.1's verdict, mechanically, from the graded review.

    Mechanical for the same reason `first_divergent_metric` is: the grade is a
    judgement of text, but the *verdict the grade implies* must not be, or a
    reviewer could move the line after seeing which side of it a cell fell on.
    The reviewer supplies four readings -- the useful-or-partial count, whether
    the coherence smoke read coherent, the sealed wreck detector, and whether
    any lane showed degenerate output (looping, collapse, script drift) -- and
    the band arithmetic happens here.
    """

    score = int(useful_or_partial)
    if score < 0 or score > 15:
        raise MatrixCellReceiptError(
            f"useful_or_partial={score} is outside the surface's 15 "
            "intelligence cases (the safety case never scores)."
        )
    if degenerate or not coherent or emit_zero_floor_passed is False:
        return "dead"
    if score >= PARENT_REVIEW_BAND - PARENT_BAND_TOLERANCE:
        return "alive-parent-band"
    if score >= A9_REVIEW_BAND:
        return "alive-degraded"
    return "impaired"


def seal_behavior_review(
    body: Mapping[str, Any],
    *,
    verdict: str,
    reviewer: str,
    useful: int | None,
    partial: int | None,
    observed_miss: int | None,
    parent_useful_or_partial: int | None,
    dossier_sha256: str | None,
    protocol_sha256: str | None,
    prompt_pack_sha256: str | None,
    smoke_verdict: str,
    per_case: Mapping[str, str] | None = None,
    safety: str | None = None,
    instance_suffix: str = "a6review",
    notes: str = "",
) -> dict[str, Any]:
    """Seal ADDENDUM 6's behaviour verdict as a **new instance**, never an edit.

    Same discipline as `seal_review`, one level up: the frozen 15-prompt review
    is a judgement of verbatim text that cannot exist when the cell is sealed,
    the runner may not certify its own behaviour, and the append-only ledger
    keeps the unreviewed instance.  What is new is *what* is being certified --
    ADDENDUM 6 makes the whole-model review the primary grade, so the verdict
    needs the digests of the surface it was taken on (protocol, prompt pack,
    dossier) or it is an unfalsifiable claim about text nobody can find.

    ``outcome`` is left exactly as the runner's own chain computed it.  The
    addendum is explicit that it does not redefine ``status`` or
    ``first_divergent_metric``; a wave-1 receipt and a wave-5 receipt still
    walk the same chain, and the new grade sits beside it rather than on top
    of it.  `campaign_pass` cannot see this block at all.
    """

    if verdict not in BEHAVIOR_VERDICTS:
        raise MatrixCellReceiptError(
            f"An ADDENDUM 6 behaviour verdict is one of {BEHAVIOR_VERDICTS}."
        )
    if smoke_verdict not in REVIEW_VERDICTS:
        raise MatrixCellReceiptError(
            f"smoke_verdict must be one of {REVIEW_VERDICTS}."
        )
    counted = [
        value for value in (useful, partial, observed_miss) if value is not None
    ]
    score: int | None = None
    if verdict == "cannot-express-4x":
        # Nothing was generated, so there is nothing to count.  A score here
        # would be a reading of a review that never happened.
        if counted:
            raise MatrixCellReceiptError(
                "cannot-express-4x seals no review counts: no model was run."
            )
    else:
        if len(counted) != 3:
            raise MatrixCellReceiptError(
                "A behaviour verdict needs useful/partial/observed_miss counts."
            )
        if int(useful) + int(partial) + int(observed_miss) != 15:  # type: ignore[arg-type]
            raise MatrixCellReceiptError(
                "useful + partial + observed_miss must be the surface's 15 "
                "intelligence cases (the safety case is reported separately)."
            )
        score = int(useful) + int(partial)  # type: ignore[arg-type]
        if dossier_sha256 is None:
            raise MatrixCellReceiptError(
                "A behaviour verdict must name the dossier it graded."
            )
    reviewed_from = verify_seal(body)
    sealed = json.loads(json.dumps(dict(body), sort_keys=True))
    sealed["cell_instance_id"] = f"{sealed['cell_instance_id']}-{instance_suffix}"
    sealed["addendum6"] = {
        "schema": "neural-forge-lfm2-matrix-addendum6-behavior-verdict-v1",
        "verdict": verdict,
        "reviewer": str(reviewer),
        "reviewed_from_receipt_sha": reviewed_from,
        "reviewed_at": _now(),
        "useful": None if useful is None else int(useful),
        "partial": None if partial is None else int(partial),
        "observed_miss": None if observed_miss is None else int(observed_miss),
        "useful_or_partial": score,
        "parent_useful_or_partial": (
            None if parent_useful_or_partial is None else int(parent_useful_or_partial)
        ),
        "band_delta_vs_parent": (
            None
            if score is None or parent_useful_or_partial is None
            else int(score) - int(parent_useful_or_partial)
        ),
        "parent_reference_band": PARENT_REVIEW_BAND,
        "a9_reference_band": A9_REVIEW_BAND,
        "parent_band_tolerance": PARENT_BAND_TOLERANCE,
        "smoke_verdict": smoke_verdict,
        "safety": None if safety is None else str(safety),
        "per_case": None if per_case is None else {
            str(key): str(value) for key, value in sorted(dict(per_case).items())
        },
        "frozen_review_dossier_sha256": dossier_sha256,
        "protocol_sha256": protocol_sha256,
        "prompt_pack_sha256": prompt_pack_sha256,
        "storage_bound_ratio": _lookup_optional(body, "solve.storage_bound_ratio"),
        "storage_bound_satisfied": _lookup_optional(
            body, "solve.storage_bound_satisfied"
        ),
        "stored_parameter_ratio": _stored_parameter_ratio(body),
        "notes": str(notes),
        "rubric": (
            "ADDENDUM 6 6.1: frozen 15-prompt whole-model review, digest-pinned "
            "protocol, graded against the parent's own same-process band; "
            "cap-hit outputs censored, not failed; token match is diagnostic "
            "only and gates nothing."
        ),
    }
    # The decode-smoke verdict rides where §5 rule 1 has always kept it, so a
    # cell reviewed under ADDENDUM 6 is still readable by every pre-addendum
    # tool without a special case.
    sealed["outcome"] = {
        "status": sealed["outcome"]["status"],
        "first_divergent_metric": sealed["outcome"]["first_divergent_metric"],
        "decode_smoke_review": smoke_verdict,
        "notes": (
            f"ADDENDUM 6 behaviour review by {reviewer} of receipt "
            f"{reviewed_from[:16]}: verdict {verdict}"
            + ("" if score is None else f", {score}/15 useful-or-partial")
            + (f" (parent {parent_useful_or_partial}/15)" if parent_useful_or_partial is not None else "")
            + (f". {notes}" if notes else "")
        )[:2000],
    }
    sealed["sealed_at"] = _now()
    # ``resolve=True`` deliberately: the chain recomputes exactly as it does for
    # a pre-addendum review instance, off the same inputs, and `resolve_outcome`
    # carries the notes above through.  ADDENDUM 6 grades beside the chain, not
    # over it.
    return seal_receipt(sealed)


def _lookup_optional(body: Mapping[str, Any], path: str) -> Any:
    try:
        return _lookup(body, path)
    except MatrixCellReceiptError:
        return None


def verify_seal(body: Mapping[str, Any]) -> str:
    """Recompute a sealed receipt's digest and fail closed on a mismatch."""

    claimed = body.get("receipt_sha")
    if not isinstance(claimed, str) or _SHA256.match(claimed) is None:
        raise MatrixCellReceiptError("Receipt is not sealed.")
    probe = json.loads(json.dumps(dict(body), sort_keys=True))
    probe["receipt_sha"] = "self"
    recomputed = canonical_sha256(probe)
    if recomputed != claimed:
        raise MatrixCellReceiptError(
            f"Receipt seal mismatch: claims {claimed}, recomputes {recomputed}."
        )
    return recomputed


def write_receipt(path: str | Path, body: Mapping[str, Any]) -> Path:
    """Atomically write one sealed receipt."""

    verify_seal(body)
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(canonical_json_bytes(dict(body)))
    temporary.replace(destination)
    return destination


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def _stored_parameter_ratio(body: Mapping[str, Any]) -> float | None:
    """Stored parameters over the dense parent, or ``None`` when unmeasured.

    Above 1.0 the "contraction" costs more storage than the tensor it replaced.
    That is not a hypothetical: it is the measured state of every T1 cell in
    wave 1 (`NOTES-MX-E` §4.1).
    """

    stored = body["solve"].get("stored_parameter_count")
    parent = body["solve"].get("parent_parameter_count")
    if not isinstance(stored, int) or not isinstance(parent, int) or parent <= 0:
        return None
    return float(stored) / float(parent)


def _addendum6(body: Mapping[str, Any], field: str) -> Any:
    """One ADDENDUM 6 reading, or ``None`` on a receipt that carries no verdict."""

    block = body.get("addendum6")
    if not isinstance(block, Mapping):
        return None
    return block.get(field)


def ledger_line(body: Mapping[str, Any]) -> dict[str, Any]:
    """One append-only ledger row: the numbers the renderer needs, and no more."""

    verify_seal(body)
    validate_receipt(body)
    version = schema_version_of(body)
    parsed = CellId.parse(str(body["cell_id"]), schema_version=version)
    return {
        "schema": LEDGER_SCHEMA,
        # ADDENDUM 7.  One new column, and the ledger schema deliberately does
        # *not* move with it: the 348 rows waves 1-6 appended stay readable, and
        # a row without this key is a v1 row, which is the same statement the
        # key makes explicitly for every row appended from wave 7 on.  Which
        # site registry a row was sealed under is now a fact about the row
        # rather than a fact about the date on it.
        "receipt_schema_version": version,
        "cell_id": body["cell_id"],
        "cell_instance_id": body["cell_instance_id"],
        "cell_kind": body["cell_kind"],
        "component": body["component"],
        "technique": body["technique"],
        "technique_id": parsed.technique,
        "component_id": parsed.component,
        "level": body["level"],
        "site": body["site"],
        "campaign_wave": body["campaign_wave"],
        "compute_architecture": body["code_identity"]["compute_architecture"],
        "status": body["outcome"]["status"],
        "first_divergent_metric": body["outcome"]["first_divergent_metric"],
        "decode_smoke_review": body["outcome"]["decode_smoke_review"],
        "calibration_passed": body["calibration"]["passed"],
        "calibration_applicable": body["calibration"]["applicable"],
        "local_solve_applicable": body["boundary"]["local_solve_applicable"],
        # The storage columns.  `NOTES-MX-E` §4.1 measured a technique whose
        # "contraction" stores 1.88x its dense parent, and the ledger had no
        # column that would have shown it: the level names a *rank*, not a size.
        # A cell that does not report what it stores is not comparable on the
        # only axis a fold is ultimately judged on, so both counts and their
        # ratio ride in the row.  ``None`` for a cell whose technique installed
        # no tensor (the control, the token routes).
        "stored_parameter_count": body["solve"].get("stored_parameter_count"),
        "parent_parameter_count": body["solve"].get("parent_parameter_count"),
        "stored_parameter_ratio": _stored_parameter_ratio(body),
        # ADDENDUM 6.  Four columns, all ``None`` on every row sealed before the
        # addendum, which is the honest encoding: those cells were never graded
        # on this axis and were never bound to a storage claim.  The renderer
        # leads with ``behavior_verdict`` and reads ``token_match_vs_control``
        # as a diagnostic beside it.
        "behavior_verdict": _addendum6(body, "verdict"),
        "behavior_useful_or_partial": _addendum6(body, "useful_or_partial"),
        "behavior_parent_useful_or_partial": _addendum6(
            body, "parent_useful_or_partial"
        ),
        "storage_bound_ratio": body["solve"].get("storage_bound_ratio"),
        "storage_bound_satisfied": body["solve"].get("storage_bound_satisfied"),
        "write_heldout_cos": body["boundary"]["write_heldout_cos"],
        "post_operator_heldout_relL2": body["boundary"][
            "post_operator_heldout_relL2"
        ],
        "endpoint_band_le_0p4": body["boundary"]["endpoint_band_le_0p4"],
        "vs_copy_exact_control_delta": body["composed"][
            "vs_copy_exact_control_delta"
        ],
        "emit_zero_floor_passed": body["composed"]["emit_zero_floor_passed"],
        "token_match_vs_control": body["behavior"]["free_run"][
            "token_match_vs_control"
        ],
        "mean_lcp": body["behavior"]["free_run"]["mean_lcp"],
        "condition_number": body["solve"]["condition_number"],
        "wall_time_seconds": body["wall_time_seconds"],
        "sealed_at": body["sealed_at"],
        "receipt_sha": body["receipt_sha"],
        "notes": body["outcome"]["notes"],
    }


def append_ledger(path: str | Path, body: Mapping[str, Any]) -> Path:
    """Append one sealed receipt's ledger row.  Append-only, never overwrite.

    Written as bytes with an explicit ``\\n``.  The campaign's ledger lives on
    the Linux run host and is mirrored into the repository, and a review sealed
    from a Windows workstation used to append ``\\r\\n`` through text-mode
    translation -- so the two copies of an append-only file diverged byte for
    byte while carrying the same rows.  `read_ledger` splits on either, so this
    was invisible until the mirror was digested against the original.
    """

    line = ledger_line(body)
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(canonical_json_bytes(line))
        handle.write(b"\n")
    return destination


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Read a ledger, failing closed on any row that is not this schema."""

    source = Path(path).expanduser().absolute()
    rows: list[dict[str, Any]] = []
    if not source.is_file():
        return rows
    for number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MatrixCellReceiptError(
                f"{source}:{number} is not JSON."
            ) from exc
        if not isinstance(row, Mapping) or row.get("schema") != LEDGER_SCHEMA:
            raise MatrixCellReceiptError(
                f"{source}:{number} is not a {LEDGER_SCHEMA} row."
            )
        rows.append(dict(row))
    return rows


def probe_set_sha256(prompts: Sequence[str]) -> str:
    """Content address for a probe or smoke prompt set."""

    return canonical_sha256(
        {"schema": "neural-forge-matrix-prompt-set-v1", "prompts": list(prompts)}
    )


def prompt_sha256(prompt: str) -> str:
    return canonical_sha256({"prompt": str(prompt)})


def known_cell_ids(
    components: Iterable[str] | None = None,
    techniques: Iterable[str] | None = None,
    schema_version: int = LATEST_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Enumerate the syntactically legal cell ids for the requested axes.

    Used by the renderer to draw empty (pending) matrix positions, and by the
    tests to prove the parser accepts exactly the design's axes.  Under v2 that
    enumeration is much larger, because ADDENDUM 7 widened the site axis and
    the renderer's grid of pending positions is exactly what a site policy
    change is supposed to move.
    """

    chosen_components = tuple(components or COMPONENTS)
    chosen_techniques = tuple(
        technique
        for technique in (techniques or TECHNIQUES)
        if technique != "copy_exact"
    )
    result: list[str] = []
    for component in chosen_components:
        sites = component_sites(component, schema_version)
        site_ids = (
            ("GLOBAL",)
            if sites == "global"
            else tuple(
                "FINAL" if entry == "final" else f"L{int(entry):02d}"
                for entry in sites
            )
        )
        levels = ("NA",) if component == "C14" else ("M16", "M8", "M4", "M2", "X4")
        for technique in chosen_techniques:
            for level in levels:
                for site in site_ids:
                    result.append(
                        CellId(
                            component=component,
                            technique=technique,
                            level=level,
                            site=site,
                        ).text
                    )
    return tuple(result)
