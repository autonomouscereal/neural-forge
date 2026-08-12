#!/usr/bin/env python3
"""ADDENDUM 10's corrective-component class: additive routes the parent lacks.

`NOTES-MX-O` §7 records why wave 11 did not run M2 and states the reason as a
constraint rather than an omission: a corrective route "at a block output" is a
tensor **the parent does not have**, the receipt schema binds every folded
component to the parent ``state_dict`` keys it replaces, and
`matrix_cell_receipt.COMPONENTS` is one of the two frozen objects the campaign's
two lanes share.  Expressing an additive corrective tensor needs a component id
that table cannot carry.

ADDENDUM 10 registers a **second, disjoint** table for exactly that, and this
module is its implementation.  Three properties are the whole design:

* **Disjoint id space.**  Corrective ids are ``X..``; folded component ids are
  ``C..``.  `COMPONENTS`, `TECHNIQUES`, `CellId` and `known_cell_ids` are
  untouched, so every sealed receipt re-verifies against the same objects it was
  sealed against and no wave-1..11 row changes meaning.
* **A corrective route replaces nothing.**  It owns no parent tensor, so it is
  never checked against ``weight_keys``, can never collide with a fold, and its
  storage is **added** to the child rather than swapped into it.  A rung that
  spends routes therefore pays for them at full price in the net ratio, which is
  the only accounting that could make the mechanism honest.
* **It is closed form.**  The route is `matrix_technique_body_routes.
  sparse_route_codes` -- residual-OMP support selection plus one ridge solve per
  target row -- run on the composed child's own activations against the
  **parent's** activations at the same corpus positions.  Zero gradients, zero
  backward calls, one extra conditioned capture per route.

**What X01 is, concretely.**  The collector proves ``residual.post_layer =
residual.post_operator + mlp.down`` for every block
(`hf_lfm2_circuit_trace_cache`, layer residual check), so a tensor added to the
``feed_forward`` module's output is added to the block's output residual exactly.
X01 reads the block's own FFN-norm output (``mlp.input``, the endpoint the folded
SwiGLU lane reads) and writes into the block output:

```
child block output  :=  child block output  +  mlp.input @ R
R solved from       :   min || X_child R  -  (Y_parent - Y_child) ||
                        X = child mlp.input rows, Y = residual.post_layer rows
```

`R` is a genuinely new operator: no parent tensor maps ``mlp.input`` linearly
into the residual, so this cannot be folded back into ``w1``/``w3``/``w2`` and it
is not M1's re-solve wearing a different name.  It is also the *only* linear
route that can cancel drift the block's own fold did not cause -- the target is
the whole block gap, so upstream damage that survives into this block's output is
inside the objective.

**Storage.**  A k-sparse code that does not store its atom ids is a code nobody
can decode; `matrix_technique_token_embedding.c01_sparse_route_storage` settled
that convention for the campaign in wave 10 and this module applies it verbatim:
``2 * nnz + (targets + 1)`` -- one coefficient and one atom id per nonzero, plus
the CSR row offsets, because residual OMP stops early and the supports are
ragged.  ``nnz`` is the **realized** support total, never ``targets * k``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .matrix_cell_receipt import MLP_LAYERS, MatrixCellReceiptError

__all__ = [
    "CORRECTIVE_COMPONENTS",
    "CORRECTIVE_ROUTE_SCHEMA",
    "CORRECTIVE_TECHNIQUES",
    "CorrectiveRoute",
    "CorrectiveRouteBank",
    "CorrectiveRouteError",
    "CorrectiveRouteId",
    "corrective_route_storage",
    "solve_corrective_route",
    "validate_corrections_block",
]

#: The block a corrective receipt entry declares, so a reader never has to infer
#: which generation of the mechanism produced it.  Versioned on its own axis:
#: the receipt schema stays v2 and old receipts, which carry no ``corrections``
#: key at all, are not re-read against anything new (ADDENDUM 10 §10.3).
CORRECTIVE_ROUTE_SCHEMA = "neural-forge-lfm2-matrix-corrective-route-v1"


class CorrectiveRouteError(RuntimeError):
    """A corrective route is malformed, or a bank was asked for the impossible."""


#: corrective component id -> what it reads, what it writes, and where it lives.
#:
#: **Disjoint from `matrix_cell_receipt.COMPONENTS` by construction.**  The ``X``
#: prefix is reserved for the corrective class and `CellId` rejects it, so a
#: corrective id can never be parsed as a folded component and a folded component
#: can never be sealed as a corrective one.
#:
#: ``module`` is the submodule path *within a decoder layer* whose forward output
#: the route is added to; ``reads`` is that submodule's own input endpoint and
#: ``writes`` is the sealed endpoint the addition lands in.  Both names are the
#: frozen collector's, so a route's objective is stated in the same vocabulary
#: every other cell in the campaign states its boundary in.
CORRECTIVE_COMPONENTS: Mapping[str, Mapping[str, Any]] = {
    "X01": {
        "name": "ffn_output_corrective_route",
        "kind": "corrective",
        "module": "feed_forward",
        "reads": "mlp.input",
        "writes": "residual.post_layer",
        "sites": MLP_LAYERS,
        "replaces_parent_tensor": False,
    },
}

#: The one technique a corrective route may be solved by.  T5 is the campaign's
#: sparse-OMP route solver and the mechanism ADDENDUM 9 §9.4 named for M2; a
#: second entry here would be a second mechanism and would need its own addendum.
CORRECTIVE_TECHNIQUES: Mapping[str, str] = {"T5": "sparse_omp_routes"}

_ROUTE_ID = re.compile(r"^(?P<component>X\d{2})\.(?P<technique>T\d)\.K(?P<k>\d+)\.L(?P<layer>\d{2})$")


@dataclass(frozen=True)
class CorrectiveRouteId:
    """``X01.T5.K64.L07`` -- component, technique, sparsity, site.

    The sparsity is *in the id* because it is the only knob that moves this
    mechanism's storage, and the campaign's rule is that a level claim is a claim
    about stored parameters.  Two routes at one site with different ``k`` are two
    different rungs and have to be two different ids.
    """

    component: str
    technique: str
    sparsity: int
    layer_index: int

    @property
    def site(self) -> str:
        return f"L{int(self.layer_index):02d}"

    @property
    def text(self) -> str:
        return (
            f"{self.component}.{self.technique}.K{int(self.sparsity)}."
            f"L{int(self.layer_index):02d}"
        )

    @classmethod
    def parse(cls, text: str) -> "CorrectiveRouteId":
        match = _ROUTE_ID.match(str(text).strip())
        if match is None:
            raise CorrectiveRouteError(
                f"{text!r} is not a corrective route id "
                "(X<nn>.T<n>.K<sparsity>.L<layer>)."
            )
        component = match.group("component")
        if component not in CORRECTIVE_COMPONENTS:
            raise CorrectiveRouteError(
                f"Unknown corrective component {component!r}; ADDENDUM 10 "
                f"registers {sorted(CORRECTIVE_COMPONENTS)}."
            )
        technique = match.group("technique")
        if technique not in CORRECTIVE_TECHNIQUES:
            raise CorrectiveRouteError(
                f"Corrective component {component} may not be solved by "
                f"{technique}; ADDENDUM 10 registers "
                f"{sorted(CORRECTIVE_TECHNIQUES)}."
            )
        layer = int(match.group("layer"))
        if layer not in tuple(CORRECTIVE_COMPONENTS[component]["sites"]):
            raise CorrectiveRouteError(
                f"{component} has no site at layer {layer}."
            )
        sparsity = int(match.group("k"))
        if sparsity <= 0:
            raise CorrectiveRouteError("A route sparsity must be positive.")
        return cls(
            component=component,
            technique=technique,
            sparsity=sparsity,
            layer_index=layer,
        )


def corrective_route_storage(*, nonzero_count: int, target_count: int) -> int:
    """``2 * nnz + (targets + 1)``, the wave-10 sparse-route convention verbatim.

    * ``nnz`` coefficients;
    * ``nnz`` atom ids, counted at parity with the value each one indexes --
      conservative, because a 1,024-atom dictionary needs ten bits and not
      thirty-two, and the campaign has counted indices this way since
      `c01_sparse_route_storage`;
    * ``targets + 1`` CSR row offsets, because residual OMP stops early on a row
      it has already explained and the supports are ragged.

    There is no head term: a corrective route has no verbatim rows.  Every
    parameter it stores is a parameter the child would not otherwise have, so the
    whole of this number is charged to the net ratio as an **addition**.
    """

    nnz = int(nonzero_count)
    targets = int(target_count)
    if nnz < 0:
        raise CorrectiveRouteError("nonzero_count must be nonnegative.")
    if targets <= 0:
        raise CorrectiveRouteError("target_count must be positive.")
    return 2 * nnz + (targets + 1)


def _torch() -> Any:
    import torch

    return torch


def _tensor_sha256(tensor: Any) -> str:
    """`run_lfm2_matrix_cell._tensor_sha256`, restated so this module imports no
    experiment script.  dtype, shape, then the raw bytes -- the same three fields
    in the same order, so a route digest and a folded-tensor digest are the same
    kind of object and can sit in one receipt without a footnote."""

    torch = _torch()
    contiguous = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _relative_l2(residual: Any, reference: Any) -> float:
    """Aggregate relative L2 of a residual block against a reference block."""

    numerator = float(residual.reshape(-1).double().norm().item())
    denominator = float(reference.reshape(-1).double().norm().item())
    if denominator <= 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


@dataclass(frozen=True)
class CorrectiveRoute:
    """One solved additive route, with everything a receipt has to state.

    ``dense`` is ``(reads, writes)`` -- the scatter of the sparse code, which is
    what the forward hook multiplies by.  The *stored* form is the sparse triple
    (``coefficients``, ``supports``, ``offsets``); the dense matrix is a runtime
    expansion of it and is never what the storage count is taken over, exactly as
    for every other T5 route the campaign has sealed.
    """

    route_id: CorrectiveRouteId
    reads: str
    writes: str
    module_path: str
    dense: Any
    supports: Any
    lengths: Any
    offsets: Any
    coefficients: Any
    nonzero_count: int
    stored_parameter_count: int
    fit_rows: int
    heldout_rows: int
    diagnostics: Mapping[str, Any]

    @property
    def text(self) -> str:
        return self.route_id.text

    def receipt_entry(self) -> dict[str, Any]:
        """The sealed record for one route.  Storage first, because it is added."""

        return {
            "schema": CORRECTIVE_ROUTE_SCHEMA,
            "route_id": self.text,
            "corrective_component_id": self.route_id.component,
            "corrective_component": str(
                CORRECTIVE_COMPONENTS[self.route_id.component]["name"]
            ),
            "technique_id": self.route_id.technique,
            "technique": CORRECTIVE_TECHNIQUES[self.route_id.technique],
            "site": self.route_id.site,
            "layer_index": int(self.route_id.layer_index),
            "module_path": self.module_path,
            "reads_endpoint": self.reads,
            "writes_endpoint": self.writes,
            "replaces_parent_tensor": False,
            "sparsity": int(self.route_id.sparsity),
            "nonzero_count": int(self.nonzero_count),
            "stored_parameter_count": int(self.stored_parameter_count),
            "stored_parameter_formula": "2 * nnz + (targets + 1)",
            "dense_shape": [int(self.dense.shape[0]), int(self.dense.shape[1])],
            "dense_parameter_count": int(self.dense.numel()),
            "fit_rows": int(self.fit_rows),
            "heldout_rows": int(self.heldout_rows),
            "route_tensor_sha256": _tensor_sha256(self.dense),
            "diagnostics": dict(self.diagnostics),
        }


def solve_corrective_route(
    *,
    route_id: CorrectiveRouteId | str,
    source_fit: Any,
    source_heldout: Any,
    gap_fit: Any,
    gap_heldout: Any,
    ridge_relative: float = 1e-10,
) -> CorrectiveRoute:
    """Solve one additive route in closed form, and measure it out of sample.

    ``source_*`` are the child's own rows at the route's read endpoint;
    ``gap_*`` are ``parent rows - child rows`` at the route's write endpoint,
    over the same corpus positions.  The fit rows solve, the held-out rows grade,
    and the number that decides whether the mechanism pays is
    ``heldout_gap_relative_after`` -- the fraction of the block's residual gap
    that survives the correction on rows the solve never saw.
    """

    torch = _torch()
    identifier = (
        route_id
        if isinstance(route_id, CorrectiveRouteId)
        else CorrectiveRouteId.parse(str(route_id))
    )
    registration = CORRECTIVE_COMPONENTS[identifier.component]
    for name, block in (
        ("source_fit", source_fit),
        ("source_heldout", source_heldout),
        ("gap_fit", gap_fit),
        ("gap_heldout", gap_heldout),
    ):
        if getattr(block, "ndim", None) != 2:
            raise CorrectiveRouteError(f"{name} must be a 2-D row block.")
        if getattr(block, "requires_grad", False):
            raise CorrectiveRouteError(
                f"{name} carries autograd; a corrective route is fold-only."
            )
    if int(source_fit.shape[0]) != int(gap_fit.shape[0]):
        raise CorrectiveRouteError(
            f"{identifier.text}: {int(source_fit.shape[0])} source fit rows "
            f"against {int(gap_fit.shape[0])} target fit rows; the route's two "
            "endpoints must be the same corpus positions."
        )
    if int(source_heldout.shape[0]) != int(gap_heldout.shape[0]):
        raise CorrectiveRouteError(
            f"{identifier.text}: held-out row counts disagree "
            f"({int(source_heldout.shape[0])} against "
            f"{int(gap_heldout.shape[0])})."
        )

    from .matrix_technique_body_routes import sparse_route_codes

    with torch.no_grad():
        design = source_fit.detach().to(torch.float64)
        target = gap_fit.detach().to(torch.float64)
        # `sparse_route_codes` indexes atoms and targets by their SHARED space,
        # which here is the corpus-position axis: atom j is input channel j read
        # down the fit rows, target i is the gap's channel i read down the same
        # rows.  Its `dense` is then (atoms, targets) = (reads, writes), which is
        # exactly the matrix the forward hook right-multiplies the child's rows
        # by.  This is the same construction `fit_body_component_routes` uses for
        # kept evidence channels; the only new thing is what the targets are.
        codes = sparse_route_codes(
            atoms=design.transpose(0, 1).contiguous(),
            targets=target.transpose(0, 1).contiguous(),
            sparsity=int(identifier.sparsity),
            ridge_relative=float(ridge_relative),
        )
        dense = codes.dense.detach().to(torch.float64)
        lengths = codes.lengths.detach().to("cpu", torch.int64)
        offsets = torch.zeros((int(lengths.shape[0]) + 1,), dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, dim=0)

        fit_before = _relative_l2(target, target)
        fit_after = _relative_l2(target - design @ dense, target)
        held_design = source_heldout.detach().to(torch.float64)
        held_target = gap_heldout.detach().to(torch.float64)
        held_before = _relative_l2(held_target, held_target)
        held_after = _relative_l2(held_target - held_design @ dense, held_target)
        # The gap is what the route is asked to explain, so "how much of the
        # block's own output did the gap amount to" is reported beside it: a
        # route that removes 30% of a gap worth 2% of the signal has bought
        # almost nothing, and the receipt should let a reader see that without
        # re-deriving it.
        gap_share_heldout = float(
            held_target.reshape(-1).norm().item()
        ) / max(float(held_design.reshape(-1).norm().item()), 1e-30)

    nonzero = int(codes.nonzero_count)
    stored = corrective_route_storage(
        nonzero_count=nonzero, target_count=int(dense.shape[1])
    )
    diagnostics = {
        "fit_gap_relative_before": fit_before,
        "fit_gap_relative_after": fit_after,
        "heldout_gap_relative_before": held_before,
        "heldout_gap_relative_after": held_after,
        "heldout_gap_explained_fraction": max(0.0, 1.0 - held_after),
        "heldout_gap_norm_over_source_norm": gap_share_heldout,
        "condition_proxy": float(codes.condition_proxy),
        "realized_mean_support": (
            float(nonzero) / float(dense.shape[1]) if int(dense.shape[1]) else 0.0
        ),
        "ridge_relative": float(ridge_relative),
        "solver": "residual-OMP support selection + ridge on the support",
        "gradients": 0,
    }
    return CorrectiveRoute(
        route_id=identifier,
        reads=str(registration["reads"]),
        writes=str(registration["writes"]),
        module_path=(
            f"model.layers.{int(identifier.layer_index)}."
            f"{registration['module']}"
        ),
        dense=dense,
        supports=codes.supports.detach().to("cpu"),
        lengths=lengths,
        offsets=offsets,
        coefficients=codes.coefficients.detach().to("cpu"),
        nonzero_count=nonzero,
        stored_parameter_count=int(stored),
        fit_rows=int(source_fit.shape[0]),
        heldout_rows=int(source_heldout.shape[0]),
        diagnostics=diagnostics,
    )


class CorrectiveRouteBank:
    """Every corrective route a child carries, and the hooks that realize them.

    A route is **not** a state_dict entry, so a bank has to be attached to every
    model object the child is realized in -- the generating model, the capture
    model an ALS/repair capture runs through, and the review script's
    reconstituted child.  `attach` is idempotent per (model, route) and
    `detach_all` is exact, because a bank that half-attaches is a child nobody
    can name.
    """

    def __init__(self) -> None:
        self._routes: dict[str, CorrectiveRoute] = {}
        self._handles: list[Any] = []
        self._attached: set[tuple[int, str]] = set()

    def __len__(self) -> int:
        return len(self._routes)

    @property
    def routes(self) -> tuple[CorrectiveRoute, ...]:
        return tuple(
            self._routes[key]
            for key in sorted(
                self._routes,
                key=lambda text: (
                    self._routes[text].route_id.layer_index,
                    text,
                ),
            )
        )

    def add(self, route: CorrectiveRoute) -> None:
        if route.text in self._routes:
            raise CorrectiveRouteError(
                f"{route.text} is already in this bank; a child carries one "
                "route per corrective id."
            )
        for existing in self._routes.values():
            if (
                existing.route_id.component == route.route_id.component
                and existing.route_id.layer_index == route.route_id.layer_index
            ):
                raise CorrectiveRouteError(
                    f"{route.text} and {existing.text} both correct "
                    f"{route.module_path}; two additive routes at one site are "
                    "one route with a bigger k, and the id has to say so."
                )
        self._routes[route.text] = route

    def attach(self, model: Any) -> None:
        """Register one forward hook per route on ``model``.

        The hook adds ``module input @ R`` to the module's own output, which the
        collector's own residual identity makes an addition to
        ``residual.post_layer`` exactly.  It runs in the module's runtime dtype
        and casts the float64 route once, per model, so the arithmetic the review
        sees is the arithmetic the generating run measured.
        """

        torch = _torch()
        for route in self.routes:
            key = (id(model), route.text)
            if key in self._attached:
                continue
            module = model.get_submodule(route.module_path)
            parameter = next(iter(module.parameters()), None)
            if parameter is None:
                raise CorrectiveRouteError(
                    f"{route.module_path} has no parameters, so a route cannot "
                    "resolve its runtime dtype and device."
                )
            weight = route.dense.detach().to(
                device=parameter.device, dtype=parameter.dtype
            )

            def _hook(_module: Any, args: Any, output: Any, _weight: Any = weight) -> Any:
                if not args:
                    raise CorrectiveRouteError(
                        "A corrective route needs its module's input; the hook "
                        "was called with no positional arguments."
                    )
                source = args[0]
                if isinstance(output, tuple):
                    raise CorrectiveRouteError(
                        "A corrective route may only be attached to a module "
                        "whose forward returns one tensor."
                    )
                with torch.no_grad():
                    return output + (source @ _weight)

            self._handles.append(module.register_forward_hook(_hook))
            self._attached.add(key)

    def detach_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._attached.clear()

    def stored_parameter_count(self) -> int:
        return int(sum(route.stored_parameter_count for route in self.routes))

    def dense_tensors(self) -> dict[str, Any]:
        return {route.text: route.dense.detach().to("cpu") for route in self.routes}

    def sparse_tensors(self) -> dict[str, Any]:
        """The *stored* form, for anyone who wants to re-derive the dense one."""

        payload: dict[str, Any] = {}
        for route in self.routes:
            payload[f"{route.text}/coefficients"] = route.coefficients
            payload[f"{route.text}/supports"] = route.supports
            payload[f"{route.text}/offsets"] = route.offsets
        return payload

    def digests(self) -> dict[str, str]:
        return {route.text: _tensor_sha256(route.dense) for route in self.routes}

    def receipt_block(self) -> dict[str, Any]:
        """The whole ``composition.corrections`` block, ready to seal."""

        entries = [route.receipt_entry() for route in self.routes]
        return {
            "schema": CORRECTIVE_ROUTE_SCHEMA,
            "mechanism": (
                "ADDENDUM 10 M2: additive sparse-OMP routes at folded block "
                "outputs, solved closed form against the parent's own rows over "
                "the composed child's own inputs"
            ),
            "registered_components": {
                key: dict(value) for key, value in CORRECTIVE_COMPONENTS.items()
            },
            "routes": entries,
            "route_count": len(entries),
            "stored_parameter_count": self.stored_parameter_count(),
            "replaces_parent_tensors": False,
            "storage_accounting": (
                "every corrective parameter is ADDED to the child; none of it "
                "replaces a parent tensor, so it is charged to the net ratio in "
                "full and no 4x claim is made for it"
            ),
        }


def validate_corrections_block(block: Any) -> None:
    """Fail closed on a ``composition.corrections`` block.

    Called by `matrix_cell_receipt._validate_composition` **only when the key is
    present**.  Every receipt sealed before ADDENDUM 10 has no such key and is
    not read against one line of this function, which is what "old receipts
    untouched" has to mean if it is to mean anything.
    """

    if not isinstance(block, Mapping):
        raise MatrixCellReceiptError(
            "composition.corrections must be a mapping when it is present."
        )
    if str(block.get("schema")) != CORRECTIVE_ROUTE_SCHEMA:
        raise MatrixCellReceiptError(
            f"composition.corrections.schema must be {CORRECTIVE_ROUTE_SCHEMA!r}."
        )
    routes = block.get("routes")
    if not isinstance(routes, list) or not routes:
        raise MatrixCellReceiptError(
            "composition.corrections.routes must be a non-empty list; a child "
            "with no corrective route seals no corrections block at all."
        )
    total = 0
    seen: set[str] = set()
    sites: set[tuple[str, int]] = set()
    for index, entry in enumerate(routes):
        if not isinstance(entry, Mapping):
            raise MatrixCellReceiptError(
                f"composition.corrections.routes[{index}] must be a mapping."
            )
        identifier = CorrectiveRouteId.parse(str(entry.get("route_id")))
        if identifier.text in seen:
            raise MatrixCellReceiptError(
                f"composition.corrections.routes lists {identifier.text} twice."
            )
        seen.add(identifier.text)
        site = (identifier.component, int(identifier.layer_index))
        if site in sites:
            raise MatrixCellReceiptError(
                f"Two {identifier.component} routes at layer "
                f"{identifier.layer_index}; a site carries one additive route."
            )
        sites.add(site)
        if bool(entry.get("replaces_parent_tensor", False)):
            raise MatrixCellReceiptError(
                f"{identifier.text} claims to replace a parent tensor; a "
                "corrective route replaces nothing, and a route that did would "
                "have to be a folded component with a C id."
            )
        for field in (
            "stored_parameter_count",
            "nonzero_count",
            "fit_rows",
            "heldout_rows",
        ):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MatrixCellReceiptError(
                    f"{identifier.text}.{field} must be a nonnegative integer."
                )
        declared = int(entry["stored_parameter_count"])
        shape = entry.get("dense_shape")
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or len(shape) != 2
        ):
            raise MatrixCellReceiptError(
                f"{identifier.text}.dense_shape must be [reads, writes]."
            )
        recomputed = corrective_route_storage(
            nonzero_count=int(entry["nonzero_count"]),
            target_count=int(shape[1]),
        )
        if declared != recomputed:
            raise MatrixCellReceiptError(
                f"{identifier.text} declares {declared} stored parameters; "
                f"2*nnz+(targets+1) is {recomputed}."
            )
        if int(entry["nonzero_count"]) > int(shape[1]) * int(identifier.sparsity):
            raise MatrixCellReceiptError(
                f"{identifier.text} realized more nonzeros than k allows."
            )
        total += declared
    if int(block.get("stored_parameter_count", -1)) != total:
        raise MatrixCellReceiptError(
            "composition.corrections.stored_parameter_count must be the sum of "
            f"its routes ({block.get('stored_parameter_count')} against "
            f"{total})."
        )


def corrections_manifest(bank: CorrectiveRouteBank, *, cell_id: str) -> dict[str, Any]:
    """The sidecar written beside the composed tensors, for the review script."""

    return {
        "schema": CORRECTIVE_ROUTE_SCHEMA,
        "kind": "composed-child-corrective-routes",
        "cell_id": str(cell_id),
        "routes": [route.receipt_entry() for route in bank.routes],
        "route_tensor_digests": bank.digests(),
        "stored_parameter_count": bank.stored_parameter_count(),
    }


def load_bank_from_receipt(
    *, receipt: Mapping[str, Any], tensors: Mapping[str, Any]
) -> CorrectiveRouteBank:
    """Rebuild a bank from a sealed receipt plus the saved dense tensors.

    Used by the frozen review, which must reconstitute the *whole* child -- a
    review that silently dropped the corrections would be a dossier about a
    different model than the one the receipt seals.  Every route's tensor is
    checked against the digest the receipt sealed before it is attached.
    """

    torch = _torch()
    composition = receipt.get("composition")
    block = (
        composition.get("corrections") if isinstance(composition, Mapping) else None
    )
    if block is None:
        raise CorrectiveRouteError(
            f"{receipt.get('cell_id')} seals no corrections block."
        )
    validate_corrections_block(block)
    bank = CorrectiveRouteBank()
    for entry in block["routes"]:
        identifier = CorrectiveRouteId.parse(str(entry["route_id"]))
        dense = tensors.get(identifier.text)
        if dense is None:
            raise CorrectiveRouteError(
                f"The corrections file carries no tensor for {identifier.text}."
            )
        observed = _tensor_sha256(dense)
        if observed != str(entry["route_tensor_sha256"]):
            raise CorrectiveRouteError(
                f"{identifier.text} digests {observed}; the receipt sealed "
                f"{entry['route_tensor_sha256']}.  This is not the sealed child."
            )
        shape = [int(value) for value in entry["dense_shape"]]
        if [int(dense.shape[0]), int(dense.shape[1])] != shape:
            raise CorrectiveRouteError(
                f"{identifier.text} is {tuple(dense.shape)}; the receipt sealed "
                f"{tuple(shape)}."
            )
        registration = CORRECTIVE_COMPONENTS[identifier.component]
        bank.add(
            CorrectiveRoute(
                route_id=identifier,
                reads=str(registration["reads"]),
                writes=str(registration["writes"]),
                module_path=str(entry["module_path"]),
                dense=dense.detach().to(torch.float64),
                supports=None,
                lengths=None,
                offsets=None,
                coefficients=None,
                nonzero_count=int(entry["nonzero_count"]),
                stored_parameter_count=int(entry["stored_parameter_count"]),
                fit_rows=int(entry["fit_rows"]),
                heldout_rows=int(entry["heldout_rows"]),
                diagnostics=dict(entry.get("diagnostics") or {}),
            )
        )
    return bank


def describe_corrective_registry() -> str:
    """One canonical JSON line, for the runner's diagnostics."""

    return json.dumps(
        {
            "schema": CORRECTIVE_ROUTE_SCHEMA,
            "components": {
                key: {
                    name: (list(value) if isinstance(value, tuple) else value)
                    for name, value in registration.items()
                }
                for key, registration in CORRECTIVE_COMPONENTS.items()
            },
            "techniques": dict(CORRECTIVE_TECHNIQUES),
        },
        sort_keys=True,
    )
