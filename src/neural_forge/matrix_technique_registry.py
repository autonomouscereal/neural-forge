#!/usr/bin/env python3
"""Dispatch table between the matrix harness and the technique kernels.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §4.3 splits the campaign into two
lanes that must not edit each other's files, joined by exactly two frozen
objects: the receipt schema (`matrix_cell_receipt`) and this dispatch table.
The harness knows how to open sealed evidence, materialize a copy_exact child,
score a boundary, free-run a probe set and seal a receipt.  It knows nothing
about how any technique solves anything.  A technique kernel knows how to solve
one component and nothing about caches, children, probes or receipts.

The contract, in one sentence: **a technique receives read-only parent weights
plus sealed evidence rows for one component at one site, and returns
parent-shaped materialized weights plus the telemetry the receipt's `solve`
block requires.**

Why parent-shaped output.  §1.1 says the unit is "one named weight group at one
named site; everything else in the child is copy_exact (bit-identical parent
weights, identity charts, untouched norms)".  That is only literally true if a
contracted component still *materializes* at the parent's shape -- the level is
the width of the **stored** form, and the installed tensor keeps the geometry
its neighbours were compiled against.  A cell that changed a shared dimension
would silently be a two-component cell, which §5 rule 6 forbids from licensing
anything.  `CellTechniqueResult.stored_parameter_count` is where the actual
compression is declared, and the harness cross-checks it against the level.

Registration
------------

    from neural_forge.matrix_technique_registry import register

    def t5_body_routes(request):
        ...
        return CellTechniqueResult(...)

    register("T5", t5_body_routes)                       # all components
    register("T5", t5_token_route, component="C01")      # one component wins

Resolution prefers the most specific entry: ``(technique, component)`` first,
then ``technique``.  That is how the token-interface T5 calibration route (the
sealed A9 mechanism, S4) and the T5 body-route generalization (the directive's
centerpiece) coexist under one technique id without either lane editing the
other's registration.

Built in here, and only these:

* ``copy_exact`` -- the control child's non-technique.  Returns the parent's own
  weights, and is what cell 0a runs.
* ``T1`` -- dense prior-anchored ridge contraction, the S2-exonerated production
  solver (`hf_lfm2_structural_residual_attention_fold._fit_linear`) driven
  unchanged.  It is the *reference implementation of this contract*, not a
  scientific centerpiece: it exists so the technique lane has a worked example
  of every field, and so every T1 column of the matrix has a real baseline to
  contrast the exotic techniques against.
* ``T1`` at ``C14`` -- the standalone norm refit (N10), which is a diagonal fit
  rather than a contraction and therefore needs its own entry.

Everything else -- T2, T3, T4, T5-body, T6, T7, T8, T9 -- is the technique
lane's, registered from `src/neural_forge/matrix_technique_*.py` at import time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .matrix_cell_receipt import COMPONENTS, LEVELS, CellId

__all__ = [
    "CellTechnique",
    "CellTechniqueError",
    "CellTechniqueRequest",
    "CellTechniqueResult",
    "ComponentSpec",
    "MatrixCellEvidence",
    "component_spec",
    "describe_registry",
    "register",
    "registered_pairs",
    "resolve",
    "resolve_level_width",
]


class CellTechniqueError(RuntimeError):
    """A technique could not run this cell.  The harness seals ``infra``."""


def _torch() -> Any:
    import torch

    return torch


# --------------------------------------------------------------------------
# what a component *is*, mechanically
# --------------------------------------------------------------------------

#: Component id -> the parent ``state_dict`` key templates it owns, the axis its
#: level contracts, and the sealed trace endpoints that carry its input, its
#: write, and the macrocell the write composes into.
#:
#: ``contraction_axis`` is one of:
#:   ``output``  the stored form factors the map's output width  (W = U @ V)
#:   ``input``   the stored form factors the map's input  width  (W = U @ V0.T)
#:   ``kernel``  the FIR taps themselves (C04) -- not a matrix factorization
#:   ``heads``   head count / head_dim (C10)
#:   ``diagonal``the component is diagonal (C14); the level axis is n/a
#:   ``route``   token-table routes (C01/C02/C15); the level is a route budget
_LAYOUT: Mapping[str, Mapping[str, Any]] = {
    "C01": {
        "keys": ("model.embed_tokens.weight",),
        "axis": "route",
        "input": None,
        "write": None,
        "macrocell": None,
    },
    "C02": {
        "keys": ("model.embed_tokens.weight",),
        "axis": "route",
        "input": None,
        "write": None,
        "macrocell": None,
    },
    "C03": {
        "keys": ("model.layers.{layer}.conv.in_proj.weight",),
        "axis": "output",
        "input": "operator.input",
        "write": None,
        "macrocell": "residual.post_layer",
    },
    "C04": {
        "keys": ("model.layers.{layer}.conv.conv.weight",),
        "axis": "kernel",
        "input": "operator.input",
        "write": "operator.output",
        "macrocell": "residual.post_layer",
    },
    "C05": {
        "keys": ("model.layers.{layer}.conv.out_proj.weight",),
        "axis": "input",
        "input": "short_conv.context",
        "write": "operator.output",
        "macrocell": "residual.post_layer",
    },
    "C06": {
        "keys": ("model.layers.{layer}.self_attn.q_proj.weight",),
        "axis": "output",
        "input": "operator.input",
        "write": "attention.q_linear",
        "macrocell": "residual.post_layer",
    },
    "C07": {
        "keys": ("model.layers.{layer}.self_attn.k_proj.weight",),
        "axis": "output",
        "input": "operator.input",
        "write": "attention.k_linear",
        "macrocell": "residual.post_layer",
    },
    "C08": {
        "keys": ("model.layers.{layer}.self_attn.v_proj.weight",),
        "axis": "output",
        "input": "operator.input",
        "write": "attention.v",
        "macrocell": "residual.post_layer",
    },
    "C09": {
        "keys": ("model.layers.{layer}.self_attn.out_proj.weight",),
        "axis": "input",
        "input": "attention.context",
        "write": "attention.o",
        "macrocell": "residual.post_layer",
    },
    "C10": {
        "keys": (
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            "model.layers.{layer}.self_attn.out_proj.weight",
        ),
        "axis": "heads",
        "input": "operator.input",
        "write": "operator.output",
        "macrocell": "residual.post_layer",
    },
    "C11": {
        "keys": ("model.layers.{layer}.feed_forward.w1.weight",),
        "axis": "output",
        "input": "mlp.input",
        "write": "mlp.gate",
        "macrocell": "residual.post_layer",
    },
    "C12": {
        "keys": ("model.layers.{layer}.feed_forward.w3.weight",),
        "axis": "output",
        "input": "mlp.input",
        "write": "mlp.up",
        "macrocell": "residual.post_layer",
    },
    "C13": {
        "keys": ("model.layers.{layer}.feed_forward.w2.weight",),
        "axis": "input",
        "input": "mlp.post_swiglu",
        "write": "mlp.down",
        "macrocell": "residual.post_layer",
    },
    "C14": {
        "keys": (
            "model.layers.{layer}.operator_norm.weight",
            "model.layers.{layer}.ffn_norm.weight",
        ),
        "axis": "diagonal",
        "input": "residual.pre",
        "write": "operator.input",
        "macrocell": "residual.post_layer",
    },
    "C15": {
        "keys": ("lm_head.weight",),
        "axis": "route",
        "input": None,
        "write": None,
        "macrocell": None,
    },
}

#: The final-norm variant of C14: the site is ``FINAL`` and the key is global.
_C14_FINAL_KEYS = ("model.embedding_norm.weight",)

#: X4 is "the width the canonical exact-4x layout assigns this component"
#: (§1.1), which is an option-bank lookup, not a formula.  Only the token
#: interface's X4 is a settled number -- 4,096, the proven A9/A11 route budget
#: (S4).  Every other component's X4 must be supplied explicitly by the caller;
#: guessing it would silently invent a layout the campaign never adopted.
_X4_WIDTHS: Mapping[str, int] = {"C01": 4096, "C02": 4096, "C15": 4096}


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One component at one site, resolved against the parent's real geometry."""

    component_id: str
    name: str
    kind: str
    site: str
    layer_index: int | None
    weight_keys: tuple[str, ...]
    contraction_axis: str
    parent_output_width: int
    parent_input_width: int
    parent_parameter_count: int
    input_endpoint: str | None
    write_endpoint: str | None
    macrocell_endpoint: str | None

    @property
    def own_width(self) -> int:
        """The width the level contracts: output for readers, input for writers."""

        if self.contraction_axis == "input":
            return self.parent_input_width
        return self.parent_output_width


def component_spec(
    cell: CellId,
    parent_weights: Mapping[str, Any],
) -> ComponentSpec:
    """Resolve a cell's component against the parent state dict.

    Fails closed when a named weight is absent: a cell that cannot name its own
    parent tensor is an infrastructure failure, not a null result.
    """

    layout = _LAYOUT[cell.component]
    layer = cell.layer_index
    if cell.component == "C14" and cell.site == "FINAL":
        templates: Sequence[str] = _C14_FINAL_KEYS
    else:
        templates = layout["keys"]
    keys: list[str] = []
    for template in templates:
        if "{layer}" in template:
            if layer is None:
                raise CellTechniqueError(
                    f"{cell.component} at site {cell.site!r} has no layer index."
                )
            keys.append(template.format(layer=layer))
        else:
            keys.append(template)
    missing = [key for key in keys if key not in parent_weights]
    if missing:
        raise CellTechniqueError(
            f"Parent state is missing {missing} for cell {cell.text}."
        )
    primary = parent_weights[keys[0]]
    shape = tuple(int(v) for v in primary.shape)
    output_width = shape[0]
    input_width = shape[1] if len(shape) > 1 else shape[0]
    parameters = sum(
        int(math.prod(tuple(int(v) for v in parent_weights[key].shape)))
        for key in keys
    )
    return ComponentSpec(
        component_id=cell.component,
        name=str(COMPONENTS[cell.component]["name"]),
        kind=str(COMPONENTS[cell.component]["kind"]),
        site=cell.site,
        layer_index=layer,
        weight_keys=tuple(keys),
        contraction_axis=str(layout["axis"]),
        parent_output_width=output_width,
        parent_input_width=input_width,
        parent_parameter_count=parameters,
        input_endpoint=layout["input"],
        write_endpoint=layout["write"] or layout["macrocell"],
        macrocell_endpoint=layout["macrocell"],
    )


def resolve_level_width(
    spec: ComponentSpec,
    level: str,
    *,
    x4_width: int | None = None,
) -> int:
    """Return the stored width the level asks of this component.

    Marginal levels are a fraction of the component's *own* width and round
    down, so M16 never accidentally widens a component whose width is not a
    multiple of 16.  X4 is a lookup, never a formula (see ``_X4_WIDTHS``).
    """

    if level not in LEVELS:
        raise CellTechniqueError(f"Unknown level {level!r}.")
    if spec.contraction_axis == "diagonal":
        if level != "NA":
            raise CellTechniqueError("Diagonal components have no level axis.")
        return int(spec.own_width)
    if level == "NA":
        raise CellTechniqueError(
            f"{spec.component_id} is not diagonal; NA is not a level for it."
        )
    if level == "X4":
        if x4_width is not None:
            return int(x4_width)
        settled = _X4_WIDTHS.get(spec.component_id)
        if settled is None:
            raise CellTechniqueError(
                f"X4 for {spec.component_id} is an option-bank lookup that this "
                "campaign has not settled; pass the layout's width explicitly."
            )
        return int(settled)
    fraction = float(LEVELS[level])
    width = int(math.floor(spec.own_width * fraction))
    if width < 1 or width >= spec.own_width:
        raise CellTechniqueError(
            f"Level {level} on width {spec.own_width} resolves to {width}, "
            "which is not a contraction."
        )
    return width


# --------------------------------------------------------------------------
# the evidence a technique may read
# --------------------------------------------------------------------------


@runtime_checkable
class MatrixCellEvidence(Protocol):
    """Read-only view over the sealed parent rows for one site.

    The harness opens the sealed trace cache once per site-batch and hands the
    same object to every cell at that site (§2.3: "evidence caches load once per
    worker process and are shared across cells at the same site").  A technique
    must not hold a reference past its own call and must not mutate a returned
    tensor.

    Rows are float64 CPU tensors of shape ``(rows, width)``, already flattened
    across batch and phase in a fixed order, so ``fit_rows(a)`` and
    ``fit_rows(b)`` are row-aligned for any two endpoints.
    """

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Every trace endpoint this evidence carries, e.g. ``mlp.input``."""

    @property
    def selection_sha256(self) -> str:
        """The authenticated digest of the selection this evidence came from."""

    @property
    def fit_row_count(self) -> int: ...

    @property
    def heldout_row_count(self) -> int: ...

    def fit_rows(self, endpoint: str) -> Any: ...

    def heldout_rows(self, endpoint: str) -> Any: ...


# --------------------------------------------------------------------------
# request / result
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellTechniqueRequest:
    """Everything a technique kernel is given, and nothing it is not.

    Deliberately absent: the child model, the probe prompts, the receipt, the
    ledger, any device other than CPU, and any writable view of the parent.
    A technique that needs one of those is doing the harness's job.
    """

    cell: CellId
    component: ComponentSpec
    level: str
    level_width: int
    seed: int
    parent_weights: Mapping[str, Any]
    evidence: MatrixCellEvidence
    knobs: Mapping[str, Any]
    workspace: Path
    log: Callable[[str], None]

    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs.get(name, default)


@dataclass(frozen=True, slots=True)
class CellTechniqueResult:
    """What a technique hands back.

    ``installed_weights`` maps parent ``state_dict`` keys to **parent-shaped**
    float64 CPU tensors.  Every key must be one of
    ``request.component.weight_keys``; touching any other key is a
    two-component fold and the harness rejects it.

    ``solve`` becomes the receipt's ``solve`` block.  These keys are mandatory
    because the receipt makes them mandatory (§3, closing N1/N2/N3 by
    construction):

        solver_path              str   -- which branch produced the coefficients
        ridge_relative           float -- effective ridge / sigma_max, or 0.0
        condition_number         float -- of the system actually solved
        condition_cap_saturated  bool
        prior_norm               float
        correction_norm          float
        requested_clamp_ratio    float
        allowed_clamp_ratio      float
        clamp_engaged            bool
        omp                      dict|None -- k, support_reuse_prefix_steps,
                                             double_mgs (T5 only)

    ``predict`` is optional and, when supplied, maps evidence rows at the
    component's ``input_endpoint`` to the component's write.  The harness uses
    it for the held-out boundary score; without it the harness falls back to the
    materialized weight, which is correct for a plain linear map and wrong for
    anything with internal state, so a stateful technique should supply it.
    """

    installed_weights: Mapping[str, Any]
    stored_parameter_count: int
    solve: Mapping[str, Any]
    predict: Callable[[Any], Any] | None = None
    artifact_identities: Mapping[str, str] = field(default_factory=dict)
    child_directory: Path | None = None
    notes: str = ""


@runtime_checkable
class CellTechnique(Protocol):
    """``(CellTechniqueRequest) -> CellTechniqueResult``."""

    def __call__(self, request: CellTechniqueRequest) -> CellTechniqueResult: ...


REQUIRED_SOLVE_FIELDS = (
    "solver_path",
    "ridge_relative",
    "condition_number",
    "condition_cap_saturated",
    "prior_norm",
    "correction_norm",
    "requested_clamp_ratio",
    "allowed_clamp_ratio",
    "clamp_engaged",
)


def validate_result(
    request: CellTechniqueRequest,
    result: CellTechniqueResult,
) -> None:
    """Reject a technique result the receipt could not honestly seal.

    Runs before anything is installed, so a contract violation costs a cell and
    not a misleading receipt.
    """

    torch = _torch()
    if not isinstance(result, CellTechniqueResult):
        raise CellTechniqueError("A technique must return a CellTechniqueResult.")
    allowed = set(request.component.weight_keys)
    delivered = set(result.installed_weights)
    if not delivered:
        # A *route* technique publishes a whole child rather than installing a
        # tensor into the copy_exact one -- the token interface is a table plus
        # a coefficient route, not a weight the parent's state dict can hold.
        # Anything else that installs nothing has silently done nothing.
        if result.child_directory is None:
            raise CellTechniqueError(
                "A technique installed no weights and published no child."
            )
        return
    stray = sorted(delivered - allowed)
    if stray:
        raise CellTechniqueError(
            f"{request.cell.text} would install {stray}, which its component "
            "does not own; a cell never folds two components (design 1.2)."
        )
    for key, value in result.installed_weights.items():
        expected = tuple(int(v) for v in request.parent_weights[key].shape)
        actual = tuple(int(v) for v in value.shape)
        if actual != expected:
            raise CellTechniqueError(
                f"{key} materialized at {actual}; the parent shape is "
                f"{expected} and every neighbour is copy_exact against it."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise CellTechniqueError(f"{key} materialized non-finite values.")
    missing = [name for name in REQUIRED_SOLVE_FIELDS if name not in result.solve]
    if missing:
        raise CellTechniqueError(
            f"{request.cell.text} solve telemetry is missing {missing}; the "
            "receipt schema makes these mandatory (design 3)."
        )
    if int(result.stored_parameter_count) <= 0:
        raise CellTechniqueError("stored_parameter_count must be positive.")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

_REGISTRY: dict[tuple[str, str | None], CellTechnique] = {}


def register(
    technique: str,
    call: CellTechnique,
    *,
    component: str | None = None,
    replace: bool = False,
) -> None:
    """Register a kernel for a technique, optionally pinned to one component."""

    from .matrix_cell_receipt import TECHNIQUES

    if technique not in TECHNIQUES:
        raise CellTechniqueError(f"Unknown technique id {technique!r}.")
    if component is not None and component not in COMPONENTS:
        raise CellTechniqueError(f"Unknown component id {component!r}.")
    if not callable(call):
        raise CellTechniqueError("A technique must be callable.")
    key = (technique, component)
    if key in _REGISTRY and not replace:
        raise CellTechniqueError(
            f"{key} is already registered; pass replace=True to override."
        )
    _REGISTRY[key] = call


def resolve(technique: str, component: str) -> CellTechnique:
    """Most-specific-wins lookup: ``(technique, component)`` then ``technique``."""

    for key in ((technique, component), (technique, None)):
        found = _REGISTRY.get(key)
        if found is not None:
            return found
    raise CellTechniqueError(
        f"No kernel registered for technique {technique!r} at component "
        f"{component!r}.  The technique lane registers into "
        "neural_forge.matrix_technique_registry.register()."
    )


def registered_pairs() -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted(_REGISTRY, key=lambda k: (k[0], k[1] or "")))


def describe_registry() -> dict[str, Any]:
    """A serializable inventory of who is wired, for the cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-registry-v1",
        "entries": [
            {
                "technique": technique,
                "component": component,
                "callable": getattr(call, "__name__", repr(call)),
                "module": getattr(call, "__module__", "?"),
            }
            for (technique, component), call in sorted(
                _REGISTRY.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
        ],
    }


# --------------------------------------------------------------------------
# built-in reference kernels
# --------------------------------------------------------------------------


def _no_solve(path: str) -> dict[str, Any]:
    """A solve block for a technique that did not solve anything."""

    return {
        "solver_path": path,
        "ridge_relative": 0.0,
        "condition_number": 1.0,
        "condition_cap_saturated": False,
        "prior_norm": 0.0,
        "correction_norm": 0.0,
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "omp": None,
    }


def copy_exact_technique(request: CellTechniqueRequest) -> CellTechniqueResult:
    """The control child's non-technique: hand the parent's weights straight back.

    Cell 0a's whole point is that nothing is solved, so its receipt's solve
    block has to say so in fields the schema still requires.  Reusing the same
    result object as every other cell is what makes the control comparable.
    """

    torch = _torch()
    installed = {
        key: request.parent_weights[key].detach().to(dtype=torch.float64).clone()
        for key in request.component.weight_keys
    }
    return CellTechniqueResult(
        installed_weights=installed,
        stored_parameter_count=int(request.component.parent_parameter_count),
        solve=_no_solve("copy_exact_no_solve"),
        predict=None,
        notes="copy_exact control: parent weights installed bit-identically.",
    )


def _pod_basis(rows: Any, width: int) -> Any:
    """Top-``width`` eigenvectors of ``rows.T @ rows``, deterministic in sign.

    The contraction basis has to be a function of the evidence alone -- no
    randomness, no ordering ambiguity -- or two runs of the same cell would not
    be the same cell.  Sign is pinned by the largest-magnitude entry, exactly as
    the layer-0 study's nested chart does.
    """

    torch = _torch()
    gram = rows.T @ rows
    gram = ((gram + gram.T) * 0.5).contiguous()
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)[:width]
    basis = vectors[:, order].contiguous()
    pivots = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivots, torch.arange(basis.shape[1])])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs).contiguous()


def t1_dense_prior_anchored_ridge(
    request: CellTechniqueRequest,
) -> CellTechniqueResult:
    """T1: contract one component's own width, then prior-anchored ridge refit.

    The stored form is a factorization of exactly one parent tensor:

        axis "output":  W = U0 @ V     U0 = (out x r) POD basis of the parent
                                            write on the FIT rows
                        V  solved by ``_fit_linear`` against ``U0.T @ P`` prior
        axis "input":   W = U @ V0.T   V0 = (in x r) POD basis of the component's
                                            own input rows
                        U  solved by ``_fit_linear`` against ``P @ V0`` prior

    Both arms call the production `_fit_linear` unchanged, with
    ``solver_diagnostics=True`` so the receipt gets the solver path and the
    conditioning of the system that was actually solved -- the telemetry
    NOTES-W13B §6 says was the only reason Stage A's 26x stayed unattributed.

    The materialized product is parent-shaped, so every neighbour stays
    copy_exact; ``stored_parameter_count`` is where the contraction is declared.

    One property worth knowing before reading a receipt: on the **output** axis
    the projected prior ``U0.T @ P`` is already the reduced-rank optimum for a
    linear component measured at its own exact write, so the solve's correction
    is identically zero and every such cell seals ``update_scale 1.0`` with a
    zero correction norm.  That is the right answer, not a dead fit -- the whole
    contraction lives in the basis.  The **input** axis is where the solver
    actually does work, because the reduced design ``X @ V0`` cannot reproduce
    ``X`` and the second factor has to be re-solved.
    """

    torch = _torch()
    from .hf_lfm2_structural_residual_attention_fold import _fit_linear

    spec = request.component
    if spec.contraction_axis not in ("output", "input"):
        raise CellTechniqueError(
            f"T1's reference kernel contracts linear maps; {spec.component_id} "
            f"has axis {spec.contraction_axis!r} and needs its own kernel."
        )
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input/write endpoints."
        )
    ridge = float(request.knob("ridge", 1e-6))
    relative_lambda = request.knob("relative_ridge_lambda", None)
    max_delta = request.knob("max_prior_delta_ratio", 1.0)
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    fit_input = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    fit_target = request.evidence.fit_rows(spec.write_endpoint).to(dtype=torch.float64)
    rank = int(request.level_width)
    request.log(
        f"T1 {request.cell.text}: axis={spec.contraction_axis} rank={rank} "
        f"rows={int(fit_input.shape[0])} in={int(fit_input.shape[1])} "
        f"out={int(fit_target.shape[1])}"
    )

    if spec.contraction_axis == "output":
        basis = _pod_basis(fit_target, rank)  # (out x r)
        prior = (basis.T @ parent).contiguous()  # (r x in)
        solved, report = _fit_linear(
            fit_input,
            fit_target @ basis,
            role=f"{spec.name}_t1_output_factor",
            ridge=ridge,
            prior=prior,
            max_prior_delta_ratio=max_delta,
            solver_diagnostics=True,
            relative_ridge_lambda=relative_lambda,
        )
        weight = (basis @ solved).contiguous()
        stored = int(basis.numel() + solved.numel())
    else:
        basis = _pod_basis(fit_input, rank)  # (in x r)
        prior = (parent @ basis).contiguous()  # (out x r)
        solved, report = _fit_linear(
            fit_input @ basis,
            fit_target,
            role=f"{spec.name}_t1_input_factor",
            ridge=ridge,
            prior=prior,
            max_prior_delta_ratio=max_delta,
            solver_diagnostics=True,
            relative_ridge_lambda=relative_lambda,
        )
        weight = (solved @ basis.T).contiguous()
        stored = int(basis.numel() + solved.numel())

    # ``Lfm2PriorCorrectionReport.as_dict`` nests its telemetry in three
    # optional blocks -- ``solver_conditioning`` (opt-in), ``prior_delta_clamp``
    # (emitted whenever the clamp was evaluated) and ``relative_ridge`` (only on
    # a relative arm) -- so that an unmodified fit seals the exact body it
    # sealed before the telemetry existed.  The receipt wants them flat, and
    # wants them *mandatory*: N1/N2/N3 stayed open for a year because these
    # numbers were computed and then dropped on the floor.
    telemetry = report.as_dict()
    conditioning = telemetry.get("solver_conditioning", {})
    clamp = telemetry.get("prior_delta_clamp", {})
    relative = telemetry.get("relative_ridge", {})
    condition = conditioning.get("system_condition_number")
    requested = clamp.get("requested_prior_delta_ratio")
    allowed = clamp.get("allowed_prior_delta_ratio")
    lambda_used = relative.get("relative_ridge_lambda")
    condition_value = float("inf") if condition is None else float(condition)
    saturated = not math.isfinite(condition_value)
    if lambda_used:
        # A relative ridge caps the solved condition number at 1/lambda; a solve
        # that lands on the cap was limited by the regularizer, not the
        # evidence, and the chain's second link exists to say so.
        saturated = saturated or condition_value >= 0.99 / float(lambda_used)
    solve = {
        "solver_path": conditioning.get("solver_path") or "unreported",
        "ridge_relative": float(lambda_used or 0.0),
        "condition_number": condition_value,
        "condition_cap_saturated": bool(saturated),
        "prior_norm": float(telemetry.get("prior_norm", 0.0)),
        "correction_norm": float(telemetry.get("correction_norm", 0.0)),
        "requested_clamp_ratio": float(requested) if requested is not None else 0.0,
        "allowed_clamp_ratio": (
            float(allowed) if allowed is not None else float("inf")
        ),
        "clamp_engaged": bool(clamp.get("clamp_engaged") or False),
        "omp": None,
        "prior_correction_report": telemetry,
        "contraction_basis_rank": rank,
        "effective_ridge": float(relative.get("effective_ridge", ridge)),
        "update_scale": float(telemetry.get("update_scale", 1.0)),
        "underdetermined": bool(telemetry.get("underdetermined", False)),
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ weight.T

    return CellTechniqueResult(
        installed_weights={key: weight},
        stored_parameter_count=stored,
        solve=solve,
        predict=predict,
        notes=(
            f"T1 dense prior-anchored ridge, {spec.contraction_axis} axis, "
            f"stored rank {rank} of {spec.own_width}."
        ),
    )


def t1_standalone_norm_refit(
    request: CellTechniqueRequest,
) -> CellTechniqueResult:
    """T1 at C14: the standalone RMSNorm gain refit nobody has ever measured (N10).

    Norms are diagonal, so there is no width to contract; the cell asks a
    different question -- how well does a *standalone* closed-form gain fit
    reproduce the parent's normalized activations, when it is not being fitted
    inside an interface solve that can hide its error in a neighbouring writer.
    """

    torch = _torch()
    from .hf_closed_form_fit import fit_diagonal_norm

    spec = request.component
    if spec.contraction_axis != "diagonal":
        raise CellTechniqueError("t1_standalone_norm_refit is C14's kernel only.")
    epsilon = float(request.knob("norm_epsilon", 1e-5))
    ridge = float(request.knob("norm_ridge", 1e-6))
    inputs = request.evidence.fit_rows("residual.pre").to(dtype=torch.float64)
    targets = request.evidence.fit_rows("operator.input").to(dtype=torch.float64)
    validation_inputs = request.evidence.heldout_rows("residual.pre").to(
        dtype=torch.float64
    )
    validation_targets = request.evidence.heldout_rows("operator.input").to(
        dtype=torch.float64
    )
    fitted = fit_diagonal_norm(
        inputs,
        targets,
        kind="rms_norm",
        epsilon=epsilon,
        has_bias=False,
        ridge=ridge,
        validation_input_rows=validation_inputs,
        validation_target_rows=validation_targets,
    )
    key = spec.weight_keys[0]
    weight = fitted.weight.detach().to(dtype=torch.float64).contiguous()
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    correction = float(torch.linalg.vector_norm(weight - parent).item())
    prior_norm = float(torch.linalg.vector_norm(parent).item())
    solve = {
        "solver_path": "fit_diagonal_norm_rms",
        "ridge_relative": 0.0,
        "condition_number": 1.0,
        "condition_cap_saturated": False,
        "prior_norm": prior_norm,
        "correction_norm": correction,
        "requested_clamp_ratio": (
            0.0 if prior_norm == 0.0 else correction / prior_norm
        ),
        "allowed_clamp_ratio": float("inf"),
        "clamp_engaged": False,
        "omp": None,
        "norm_fit_relative_rmse": float(fitted.train_metrics.relative_rmse),
        "norm_validation_relative_rmse": (
            None
            if fitted.validation_metrics is None
            else float(fitted.validation_metrics.relative_rmse)
        ),
    }

    def predict(rows: Any) -> Any:
        values = rows.to(dtype=torch.float64)
        normalized = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + epsilon
        )
        return (normalized * weight).contiguous()

    return CellTechniqueResult(
        installed_weights={key: weight},
        stored_parameter_count=int(weight.numel()),
        solve=solve,
        predict=predict,
        notes="C14 standalone RMSNorm gain refit (N10), identity size.",
    )


register("copy_exact", copy_exact_technique)
register("T1", t1_dense_prior_anchored_ridge)
register("T1", t1_standalone_norm_refit, component="C14")
