"""ADDENDUM 10's corrective-route class, tested where it could lie.

The mechanism's whole risk is bookkeeping, not arithmetic: an additive tensor
that is *not* charged to the child, or a route that quietly claims to replace a
parent tensor, would turn "repair bought drift back" into a free lunch nobody
measured.  So the tests here concentrate on the four claims a corrective route
makes about itself -- what it stores, that it replaces nothing, that it is a
disjoint id space, and that it actually computes ``source @ R`` at the module
boundary it says it does -- and on the two refusals that keep old receipts out of
the new validator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    COMPONENTS,
    CellId,
    MatrixCellReceiptError,
    _validate_composition,
)
from neural_forge.matrix_composition import (  # noqa: E402
    MatrixCompositionError,
    net_storage,
    validate_recipe,
)
from neural_forge.matrix_corrective_routes import (  # noqa: E402
    CORRECTIVE_COMPONENTS,
    CORRECTIVE_ROUTE_SCHEMA,
    CorrectiveRouteBank,
    CorrectiveRouteError,
    CorrectiveRouteId,
    corrective_route_storage,
    load_bank_from_receipt,
    solve_corrective_route,
    validate_corrections_block,
)


def _rows(count: int, width: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((count, width), generator=generator, dtype=torch.float64)


# --------------------------------------------------------------------------
# the id space
# --------------------------------------------------------------------------


def test_the_corrective_id_space_is_disjoint_from_the_frozen_component_table():
    """The reason wave 11 could not run M2 is the reason this has to hold."""

    assert not set(CORRECTIVE_COMPONENTS) & set(COMPONENTS)
    for identifier in CORRECTIVE_COMPONENTS:
        with pytest.raises(Exception):
            CellId.parse(f"{identifier}.T5.S4.L05")


def test_a_route_id_round_trips_and_carries_its_sparsity():
    parsed = CorrectiveRouteId.parse("X01.T5.K64.L07")
    assert parsed.component == "X01"
    assert parsed.technique == "T5"
    assert parsed.sparsity == 64
    assert parsed.layer_index == 7
    assert parsed.site == "L07"
    assert parsed.text == "X01.T5.K64.L07"


@pytest.mark.parametrize(
    "text",
    [
        "X01.T5.L07",
        "X99.T5.K64.L07",
        "X01.T8.K64.L07",
        "X01.T5.K0.L07",
        "C11.T8.S4.L05",
        "X01.T5.K64.L14",
    ],
)
def test_a_malformed_or_unregistered_route_id_is_refused(text: str):
    with pytest.raises(CorrectiveRouteError):
        CorrectiveRouteId.parse(text)


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def test_route_storage_counts_the_atom_ids_and_the_csr_offsets():
    """`c01_sparse_route_storage`'s convention, minus the head it does not have."""

    assert corrective_route_storage(nonzero_count=100, target_count=1024) == (
        2 * 100 + 1025
    )
    assert corrective_route_storage(nonzero_count=0, target_count=8) == 9
    with pytest.raises(CorrectiveRouteError):
        corrective_route_storage(nonzero_count=-1, target_count=8)
    with pytest.raises(CorrectiveRouteError):
        corrective_route_storage(nonzero_count=1, target_count=0)


def test_corrective_storage_is_ADDED_to_the_child_and_never_discounted():
    """A repaired rung must be comparable to an unrepaired one at full price."""

    entries = [
        {
            "component_id": "C11",
            "isolation_cell_id": "C11.T8.S4.L05",
            "storage": {
                "installed_parameter_count": 2_621_440,
                "owned_parameter_count": 2_621_440,
                "stored_parameter_count": 655_360,
                "declared_ratio": 0.25,
            },
        }
    ]
    plain = net_storage(229_693_184, entries)
    repaired = net_storage(
        229_693_184, entries, corrective_stored_parameter_count=66_561
    )
    assert repaired["child_parameter_count"] == plain["child_parameter_count"] + 66_561
    assert repaired["net_ratio"] > plain["net_ratio"]
    assert repaired["corrective_stored_parameter_count"] == 66_561
    # Wave 1-11's arithmetic is unchanged when nothing corrective is passed.
    assert plain["corrective_stored_parameter_count"] == 0
    assert plain["child_parameter_count"] == 229_693_184 - 2_621_440 + 655_360


def test_a_negative_corrective_count_is_refused():
    with pytest.raises(MatrixCompositionError):
        net_storage(
            229_693_184,
            [
                {
                    "component_id": "C11",
                    "isolation_cell_id": "C11.T8.S4.L05",
                    "storage": {
                        "installed_parameter_count": 4,
                        "owned_parameter_count": 4,
                        "stored_parameter_count": 1,
                        "declared_ratio": 0.25,
                    },
                }
            ],
            corrective_stored_parameter_count=-1,
        )


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------


def test_a_route_recovers_an_exactly_linear_gap_and_reports_it_out_of_sample():
    """If the gap really is ``X @ R`` for a sparse ``R``, OMP must find it.

    This is the mechanism's sanity arm, not its claim: a real composed gap is not
    linear in the block's input and the campaign's whole question is how much of
    it is.  What this pins is that a route which *could* cancel a gap does, on
    rows it never saw.
    """

    reads, writes, fit, held = 24, 6, 400, 120
    generator = torch.Generator().manual_seed(11)
    truth = torch.zeros((reads, writes), dtype=torch.float64)
    for column in range(writes):
        support = torch.randperm(reads, generator=generator)[:3]
        truth[support, column] = torch.randn(3, generator=generator, dtype=torch.float64)
    source_fit = _rows(fit, reads, seed=1)
    source_held = _rows(held, reads, seed=2)
    route = solve_corrective_route(
        route_id="X01.T5.K4.L05",
        source_fit=source_fit,
        source_heldout=source_held,
        gap_fit=source_fit @ truth,
        gap_heldout=source_held @ truth,
    )
    assert route.diagnostics["heldout_gap_relative_after"] < 1e-6
    assert route.diagnostics["fit_gap_relative_after"] < 1e-6
    assert route.nonzero_count <= writes * 4
    assert route.stored_parameter_count == corrective_route_storage(
        nonzero_count=route.nonzero_count, target_count=writes
    )
    assert torch.allclose(route.offsets[1:] - route.offsets[:-1], route.lengths)
    assert int(route.offsets[0].item()) == 0


def test_a_route_cannot_explain_an_unrelated_gap_and_says_so():
    """The honest failure mode: the number reported is out of sample, not in."""

    reads, writes, fit, held = 16, 8, 300, 90
    route = solve_corrective_route(
        route_id="X01.T5.K4.L06",
        source_fit=_rows(fit, reads, seed=3),
        source_heldout=_rows(held, reads, seed=4),
        gap_fit=_rows(fit, writes, seed=5),
        gap_heldout=_rows(held, writes, seed=6),
    )
    assert route.diagnostics["fit_gap_relative_after"] < 1.0
    assert route.diagnostics["heldout_gap_relative_after"] > 0.9


def test_a_route_refuses_misaligned_rows_and_autograd():
    source = _rows(40, 8, seed=7)
    gap = _rows(39, 4, seed=8)
    with pytest.raises(CorrectiveRouteError):
        solve_corrective_route(
            route_id="X01.T5.K2.L05",
            source_fit=source,
            source_heldout=_rows(10, 8, seed=9),
            gap_fit=gap,
            gap_heldout=_rows(10, 4, seed=10),
        )
    live = _rows(40, 8, seed=11).clone().requires_grad_(True)
    with pytest.raises(CorrectiveRouteError):
        solve_corrective_route(
            route_id="X01.T5.K2.L05",
            source_fit=live,
            source_heldout=_rows(10, 8, seed=12),
            gap_fit=_rows(40, 4, seed=13),
            gap_heldout=_rows(10, 4, seed=14),
        )


# --------------------------------------------------------------------------
# the bank, and what it does to a module
# --------------------------------------------------------------------------


class _FeedForward(torch.nn.Module):
    """A stand-in for the SwiGLU block: a nonlinearity, then ``w2``.

    ``w2`` is a named submodule because that is where the route attaches -- the
    endpoint the frozen collector captures as ``mlp.down`` and writes its
    residual identity over.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(width, width, bias=False)
        self.w2 = torch.nn.Linear(width, width, bias=False)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(rows)))


class _Block(torch.nn.Module):
    """A stand-in for one decoder layer."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.feed_forward = _FeedForward(width)


class _Stack(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(width) for _ in range(layers)])

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            rows = rows + layer.feed_forward(rows)
        return rows


class _Model(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.model = _Stack(width, layers)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.model(rows)


def _route(width: int, layer: int) -> object:
    source_fit = _rows(200, width, seed=20 + layer)
    truth = torch.zeros((width, width), dtype=torch.float64)
    truth[0, 0] = 0.5
    truth[1, 2] = -0.25
    return solve_corrective_route(
        route_id=f"X01.T5.K2.L{layer:02d}",
        source_fit=source_fit,
        source_heldout=_rows(60, width, seed=40 + layer),
        gap_fit=source_fit @ truth,
        gap_heldout=_rows(60, width, seed=40 + layer) @ truth,
    )


def test_attaching_a_bank_adds_exactly_source_at_R_to_w2s_output():
    """The write lands in ``mlp.down``, and it is exactly ``mlp.input @ R``.

    This is the property the frozen collector's residual identity rests on: if
    the correction were anywhere else in the block, ``layer_output ==
    post_operator + mlp.down`` would stop holding and the collector would refuse
    the next capture -- which is exactly what it did when wave 12 first attached
    the write to the ``feed_forward`` module.
    """

    width = 6
    model = _Model(width, layers=3).to(torch.float64)
    rows = _rows(5, width, seed=99)
    forward = model.get_submodule("model.layers.1.feed_forward")

    def _probe(store):
        return [
            forward.register_forward_pre_hook(
                lambda _m, args: store.__setitem__("mlp_input", args[0].clone())
            ),
            forward.w2.register_forward_hook(
                lambda _m, _a, out: store.__setitem__("mlp_down", out.clone())
            ),
        ]

    plain: dict[str, torch.Tensor] = {}
    probes = _probe(plain)
    with torch.no_grad():
        before = model(rows).clone()
    for handle in probes:
        handle.remove()

    route = _route(width, 1)
    bank = CorrectiveRouteBank()
    bank.add(route)
    bank.attach(model)
    # Registered AFTER the bank, which is the frozen collector's own situation:
    # its hooks go on when a capture session opens, and a capture only ever opens
    # on a child whose routes are already installed.  Torch runs forward hooks in
    # registration order, so an observer registered later sees the corrected
    # ``mlp.down`` -- which is exactly what the collector's residual identity
    # needs in order to keep holding.
    corrected: dict[str, torch.Tensor] = {}
    probes = _probe(corrected)
    with torch.no_grad():
        after = model(rows).clone()
    with torch.no_grad():
        expected = plain["mlp_down"] + (plain["mlp_input"] @ route.dense)
    assert torch.allclose(corrected["mlp_down"], expected)
    # And the identity the collector verifies still holds through the block.
    assert not torch.allclose(before, after)

    bank.detach_all()
    for handle in probes:
        handle.remove()
    with torch.no_grad():
        restored = model(rows).clone()
    assert torch.allclose(before, restored)


def test_an_observer_registered_before_the_bank_does_not_see_the_correction():
    """Hook order is load-bearing, so it is pinned rather than assumed.

    Wave 12 lost a run to the frozen collector's residual check and the fix was
    to move the write into ``mlp.down``.  That fix only works because the
    collector observes ``mlp.down`` *after* the route has added to it.  This test
    states the failure mode explicitly: an observer registered first sees the
    uncorrected value, and would compute an identity that does not hold.
    """

    width = 6
    model = _Model(width, layers=1).to(torch.float64)
    rows = _rows(4, width, seed=77)
    forward = model.get_submodule("model.layers.0.feed_forward")
    early: dict[str, torch.Tensor] = {}
    handle = forward.w2.register_forward_hook(
        lambda _m, _a, out: early.__setitem__("mlp_down", out.clone())
    )
    bank = CorrectiveRouteBank()
    bank.add(_route(width, 0))
    bank.attach(model)
    late: dict[str, torch.Tensor] = {}
    late_handle = forward.w2.register_forward_hook(
        lambda _m, _a, out: late.__setitem__("mlp_down", out.clone())
    )
    with torch.no_grad():
        model(rows)
    assert not torch.allclose(early["mlp_down"], late["mlp_down"])
    handle.remove()
    late_handle.remove()
    bank.detach_all()


def test_a_detached_bank_leaves_no_hook_behind():
    width = 4
    model = _Model(width, layers=2).to(torch.float64)
    bank = CorrectiveRouteBank()
    bank.add(_route(width, 0))
    bank.attach(model)
    bank.attach(model)  # idempotent per (model, route)
    forward = model.get_submodule("model.layers.0.feed_forward")
    assert len(forward._forward_pre_hooks) == 1
    assert len(forward.w2._forward_hooks) == 1
    bank.detach_all()
    assert len(forward._forward_pre_hooks) == 0
    assert len(forward.w2._forward_hooks) == 0


def test_two_routes_at_one_site_are_refused():
    width = 4
    bank = CorrectiveRouteBank()
    bank.add(_route(width, 0))
    with pytest.raises(CorrectiveRouteError):
        bank.add(_route(width, 0))


# --------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------


def _corrections_block(width: int = 4) -> dict:
    bank = CorrectiveRouteBank()
    bank.add(_route(width, 5))
    return bank.receipt_block()


def test_a_well_formed_corrections_block_validates():
    validate_corrections_block(_corrections_block())


def test_a_corrections_block_that_understates_its_storage_is_refused():
    block = _corrections_block()
    block["stored_parameter_count"] = int(block["stored_parameter_count"]) - 1
    with pytest.raises(MatrixCellReceiptError):
        validate_corrections_block(block)


def test_a_route_that_claims_to_replace_a_parent_tensor_is_refused():
    block = _corrections_block()
    block["routes"][0]["replaces_parent_tensor"] = True
    with pytest.raises(MatrixCellReceiptError):
        validate_corrections_block(block)


def test_a_route_whose_declared_storage_does_not_match_its_nnz_is_refused():
    block = _corrections_block()
    block["routes"][0]["stored_parameter_count"] = 12
    with pytest.raises(MatrixCellReceiptError):
        validate_corrections_block(block)


def test_a_composition_receipt_without_corrections_is_not_read_against_the_new_rules():
    """Old receipts untouched, stated as a test rather than as an intention."""

    body = {
        "composition": {
            "mode": "M-A",
            "components": [
                {
                    "component_id": "C11",
                    "isolation_cell_id": "C11.T8.S4.L05",
                    "first_divergent_metric": None,
                    "weight_keys": ["model.layers.5.feed_forward.w1.weight"],
                    "component": "mlp_gate",
                    "technique_id": "T8",
                    "technique": "gram_relation_matching",
                    "level": "S4",
                    "site": "L05",
                    "selection_reason": "fixture",
                    "evidence": {},
                    "solve": {},
                    "boundary": {},
                    "calibration": {},
                    "after_install": {},
                    "storage": {},
                }
            ],
            "install_order": ["C11.T8.S4.L05"],
            "storage": {
                "child_parameter_count": 1,
                "parent_parameter_count": 2,
                "net_ratio": 0.5,
            },
            "passed_through": [],
        }
    }
    _validate_composition(body, CellId.parse("COMP.MA.TEST"))
    body["composition"]["corrections"] = {"schema": "wrong"}
    with pytest.raises(MatrixCellReceiptError):
        _validate_composition(body, CellId.parse("COMP.MA.TEST"))


def test_the_review_rebuilds_a_bank_only_from_digest_matching_tensors():
    width = 4
    bank = CorrectiveRouteBank()
    route = _route(width, 5)
    bank.add(route)
    receipt = {"cell_id": "COMP.MA.TEST", "composition": {"corrections": bank.receipt_block()}}
    rebuilt = load_bank_from_receipt(receipt=receipt, tensors=bank.dense_tensors())
    assert [r.text for r in rebuilt.routes] == [route.text]
    tampered = {route.text: route.dense + 1.0}
    with pytest.raises(CorrectiveRouteError):
        load_bank_from_receipt(receipt=receipt, tensors=tampered)
    with pytest.raises(CorrectiveRouteError):
        load_bank_from_receipt(receipt=receipt, tensors={})


# --------------------------------------------------------------------------
# the recipe
# --------------------------------------------------------------------------


def _recipe(corrections: list | None) -> dict:
    from neural_forge.matrix_composition import RECIPE_SCHEMA, component_weight_keys

    cell = CellId.parse("C11.T8.S4.L05")
    recipe = {
        "schema": RECIPE_SCHEMA,
        "install_order": [cell.text],
        "components": [
            {
                "cell_id": cell.text,
                "component_id": cell.component,
                "weight_keys": list(component_weight_keys(cell)),
                "x4_width": 639,
            }
        ],
        "passed_through": [],
    }
    if corrections is not None:
        recipe["corrections"] = corrections
    return recipe


def test_a_recipe_may_request_corrective_routes_and_must_justify_each_one():
    validate_recipe(
        _recipe([{"route_id": "X01.T5.K64.L05", "selection_reason": "repair L05"}])
    )
    with pytest.raises(MatrixCompositionError):
        validate_recipe(_recipe([{"route_id": "X01.T5.K64.L05"}]))
    with pytest.raises(MatrixCompositionError):
        validate_recipe(
            _recipe(
                [
                    {"route_id": "X01.T5.K64.L05", "selection_reason": "a"},
                    {"route_id": "X01.T5.K64.L05", "selection_reason": "b"},
                ]
            )
        )
    with pytest.raises(MatrixCompositionError):
        validate_recipe(_recipe([]))
    # A recipe with no corrections key is exactly wave 1-11's recipe.
    validate_recipe(_recipe(None))


def test_the_runner_declares_the_corrective_registry_it_ran_under():
    from neural_forge.matrix_corrective_routes import describe_corrective_registry

    described = json.loads(describe_corrective_registry())
    assert described["schema"] == CORRECTIVE_ROUTE_SCHEMA
    assert described["components"]["X01"]["reads"] == "mlp.input"
    assert described["components"]["X01"]["writes"] == "mlp.down"
    assert (
        described["components"]["X01"]["writes_residual_endpoint"]
        == "residual.post_layer"
    )
    assert described["components"]["X01"]["module"] == "feed_forward.w2"
    assert described["components"]["X01"]["source_module"] == "feed_forward"
    assert described["components"]["X01"]["replaces_parent_tensor"] is False


def test_the_module_docstring_states_the_storage_formula_it_implements():
    source = Path(__file__).resolve().parents[1] / "src" / "neural_forge" / (
        "matrix_corrective_routes.py"
    )
    text = source.read_text(encoding="utf-8")
    assert re.search(r"2 \* nnz \+ \(targets \+ 1\)", text)
