"""Contract tests for wave 13's M5 kernel: the Procrustes head merge at C10.

`reports/matrix-campaign-20260810/wave12/NOTES-MX-P.md` §8 specified this kernel
and wave 12 could not run it.  What these tests pin is the set of ways a head
merge can be wrong in a manner a sealed receipt would still look plausible
under:

1. **The storage arithmetic is the reason the cell exists.**  C10's other kernel
   cannot express S4 at all; this one can only express it at ``g_q = 3``, and
   the difference between §8's published 0.21875x and the implementable
   0.239583x is a whole kv group.  A test pins both numbers and the correction
   between them, because a storage model that drifts by 8.7% is the difference
   between a frontier claim and a sub-4x diagnostic.
2. **A merge is a merge and not an edit.**  One basis per head has to give the
   parent back, over *all four* projections, through the same clustering,
   alternation and refit a real cell runs.  The identity gate is a measurement
   in the receipt and this is the test that it measures something.
3. **The recovery claim is exact where it should be.**  When the parent's heads
   really are rotations of ``g`` shared bases, the kernel must find that
   structure and reproduce the parent to machine precision.  A kernel that
   merged the wrong heads, or aligned them with the wrong orientation, would
   still produce finite tensors, a contracting storage count and a receipt.
4. **Both C10 kernels stay reachable.**  T7-at-C10 must not shadow T1-at-C10,
   and T7 elsewhere must still be the attention relation kernel.  Wave 13
   deliberately did not mint a new technique id -- `TECHNIQUES` and therefore
   `known_cell_ids` are what the campaign's 442 sealed rows verify against --
   so "the pin does not collide" is a claim that needs a test.
5. **A refusal is a result.**  Wrong component, a level that does not name whole
   groups, a level that contracts nothing, a parent that cannot replay its own
   write: each raises rather than returning a confident number about something
   else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_head_drop import (  # noqa: E402
    c10_head_drop_technique,
)
# Imported for its registration side effect: the claim under test is that
# pinning T7 to C10 does not disturb T7 where it already lives.
from neural_forge import matrix_technique_relation_matching  # noqa: E402,F401
from neural_forge.matrix_technique_head_merge import (  # noqa: E402
    c10_head_merge_admissible_width,
    HEAD_MERGE_HEAD_DIM,
    c10_head_merge_storage,
    c10_head_merge_technique,
    describe_head_merge_technique,
    head_merge_group_plan,
    head_merge_stored_parameters,
)
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    resolve,
    storage_model,
    storage_width_for_ratio,
    validate_result,
)

# The test fixture is a scale model of the parent: 8 query heads, 4 kv heads,
# head_dim 4.  Keeping the 2:1 GQA ratio is what makes the kv group arithmetic
# -- the thing §8 got wrong -- exercised at all.
HEADS = 8
KV_HEADS = 4
HEAD_DIM = 4
WIDTH = HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
FIT_ROWS = 96
HELD_ROWS = 48
LAYER = 2

# The real parent's numbers, which the S4 claim is actually made about.
LFM2_HIDDEN = 1024
LFM2_HEADS = 16
LFM2_KV_HEADS = 8
LFM2_HEAD_DIM = 64
LFM2_OWNED = 2 * LFM2_HIDDEN * LFM2_HIDDEN + 2 * (LFM2_KV_HEADS * LFM2_HEAD_DIM) * (
    LFM2_HIDDEN
)


class _Evidence:
    def __init__(self, fit, heldout):
        self._fit = dict(fit)
        self._heldout = dict(heldout)

    @property
    def endpoints(self):
        return tuple(sorted(self._fit))

    @property
    def selection_sha256(self):
        return "0" * 64

    @property
    def fit_row_count(self):
        return int(next(iter(self._fit.values())).shape[0])

    @property
    def heldout_row_count(self):
        return int(next(iter(self._heldout.values())).shape[0])

    def fit_rows(self, endpoint):
        return self._fit[endpoint]

    def heldout_rows(self, endpoint):
        return self._heldout[endpoint]


class _Spec:
    """A minimal C10 ComponentSpec stand-in for the storage-model tests.

    The storage model is planning arithmetic over four tensor *shapes*; making
    the planner allocate 25 MB of float64 to answer a question about integers
    would be the test costing more than the thing it tests.
    """

    def __init__(self, hidden, query_width, owned):
        self.component_id = "C10"
        self.parent_input_width = hidden
        self.parent_output_width = query_width
        self.parent_parameter_count = owned
        self.own_width = query_width


def _request(cell_text, state, evidence, *, level_width, knobs=None):
    cell = CellId.parse(cell_text)
    spec = component_spec(cell, state)
    return CellTechniqueRequest(
        cell=cell,
        component=spec,
        level=cell.level,
        level_width=level_width,
        seed=0,
        parent_weights=state,
        evidence=evidence,
        knobs=dict(knobs or {}),
        workspace=Path("."),
        log=lambda message: None,
    )


def _orthogonal(generator, size):
    matrix = torch.randn((size, size), dtype=torch.float64, generator=generator)
    left, _singular, right_t = torch.linalg.svd(matrix)
    return left @ right_t


def _random_state(generator, *, layer=LAYER):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {
        f"model.layers.{layer}.self_attn.q_proj.weight": randn(WIDTH, WIDTH),
        f"model.layers.{layer}.self_attn.k_proj.weight": randn(KV_WIDTH, WIDTH),
        f"model.layers.{layer}.self_attn.v_proj.weight": randn(KV_WIDTH, WIDTH),
        f"model.layers.{layer}.self_attn.out_proj.weight": randn(WIDTH, WIDTH),
        f"model.layers.{layer}.self_attn.q_layernorm.weight": torch.ones(
            HEAD_DIM, dtype=torch.float64
        ),
    }


def _mergeable_state(generator, *, query_groups, kv_groups, layer=LAYER):
    """A parent whose heads ARE rotations of ``g`` shared bases, exactly.

    Head ``h`` belongs to group ``h % g`` -- interleaved rather than contiguous,
    so a kernel that merged adjacent heads by position instead of by content
    would fail this test rather than pass it by luck.
    """

    q_bases = [
        torch.randn((HEAD_DIM, WIDTH), dtype=torch.float64, generator=generator)
        for _ in range(query_groups)
    ]
    o_bases = [
        torch.randn((WIDTH, HEAD_DIM), dtype=torch.float64, generator=generator)
        for _ in range(query_groups)
    ]
    k_bases = [
        torch.randn((HEAD_DIM, WIDTH), dtype=torch.float64, generator=generator)
        for _ in range(kv_groups)
    ]
    v_bases = [
        torch.randn((HEAD_DIM, WIDTH), dtype=torch.float64, generator=generator)
        for _ in range(kv_groups)
    ]
    q_proj = torch.zeros((WIDTH, WIDTH), dtype=torch.float64)
    o_proj = torch.zeros((WIDTH, WIDTH), dtype=torch.float64)
    k_proj = torch.zeros((KV_WIDTH, WIDTH), dtype=torch.float64)
    v_proj = torch.zeros((KV_WIDTH, WIDTH), dtype=torch.float64)
    for head in range(HEADS):
        group = head % query_groups
        rotation = _orthogonal(generator, HEAD_DIM)
        q_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM, :] = rotation @ q_bases[group]
        o_proj[:, head * HEAD_DIM : (head + 1) * HEAD_DIM] = o_bases[group] @ rotation.T
    for head in range(KV_HEADS):
        group = head % kv_groups
        rotation = _orthogonal(generator, HEAD_DIM)
        k_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM, :] = rotation @ k_bases[group]
        v_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM, :] = rotation @ v_bases[group]
    return {
        f"model.layers.{layer}.self_attn.q_proj.weight": q_proj,
        f"model.layers.{layer}.self_attn.k_proj.weight": k_proj,
        f"model.layers.{layer}.self_attn.v_proj.weight": v_proj,
        f"model.layers.{layer}.self_attn.out_proj.weight": o_proj,
        f"model.layers.{layer}.self_attn.q_layernorm.weight": torch.ones(
            HEAD_DIM, dtype=torch.float64
        ),
    }


def _evidence(generator, state, *, layer=LAYER, corrupt_write=False):
    """Context rows and the exact O-projection write they produce.

    ``attention.o = attention.context @ out_proj.T`` is the one exact matmul
    between two sealed endpoints an attention block has, and it is the relation
    the kernel's parent gate checks.
    """

    o_proj = state[f"model.layers.{layer}.self_attn.out_proj.weight"]
    fit_context = torch.randn(
        (FIT_ROWS, WIDTH), dtype=torch.float64, generator=generator
    )
    held_context = torch.randn(
        (HELD_ROWS, WIDTH), dtype=torch.float64, generator=generator
    )
    fit_write = fit_context @ o_proj.T
    held_write = held_context @ o_proj.T
    if corrupt_write:
        held_write = held_write + 0.5 * torch.randn(
            held_write.shape, dtype=torch.float64, generator=generator
        )
    return _Evidence(
        {"attention.context": fit_context, "attention.o": fit_write},
        {"attention.context": held_context, "attention.o": held_write},
    )


# ---------------------------------------------------------------- storage


def test_group_plan_makes_the_kv_term_integral():
    # §8 wrote the kv term as g/2 and evaluated it at 1.5 for g=3.  Eight kv
    # heads split into two groups or into one; never into one and a half.
    assert head_merge_group_plan(
        192, query_heads=16, kv_heads=8, head_dim=64
    ) == (3, 2)
    assert head_merge_group_plan(
        256, query_heads=16, kv_heads=8, head_dim=64
    ) == (4, 2)
    assert head_merge_group_plan(
        128, query_heads=16, kv_heads=8, head_dim=64
    ) == (2, 1)
    assert head_merge_group_plan(
        1024, query_heads=16, kv_heads=8, head_dim=64
    ) == (16, 8)


def test_storage_pins_the_spec_arithmetic_and_its_correction():
    spec = _Spec(LFM2_HIDDEN, LFM2_HIDDEN, LFM2_OWNED)
    assert LFM2_OWNED == 3_145_728

    # §8's g = 4 row, unchanged: over the S4 bar, so a diagnostic and not a
    # frontier claim.
    assert c10_head_merge_storage(spec, 256) == 884_736
    assert c10_head_merge_storage(spec, 256) / LFM2_OWNED == pytest.approx(0.28125)

    # §8's g = 3 row, corrected.  The published 688,128 assumed 1.5 kv groups.
    assert c10_head_merge_storage(spec, 192) == 753_664
    assert c10_head_merge_storage(spec, 192) != 688_128
    assert c10_head_merge_storage(spec, 192) / LFM2_OWNED == pytest.approx(
        0.2395833333, abs=1e-9
    )
    assert c10_head_merge_storage(spec, 192) <= 0.25 * LFM2_OWNED


def test_alignments_are_charged_in_full():
    # 24 orthogonal 64x64 blocks at 4,096 each.  An orthogonal matrix has 2,016
    # degrees of freedom; the child stores 4,096, so the model charges 4,096.
    spec = _Spec(LFM2_HIDDEN, LFM2_HIDDEN, LFM2_OWNED)
    bases_only = head_merge_stored_parameters(
        hidden_size=LFM2_HIDDEN,
        head_dim=LFM2_HEAD_DIM,
        query_heads=0,
        kv_heads=0,
        query_groups=3,
        kv_groups=2,
    )
    assert c10_head_merge_storage(spec, 192) - bases_only == 24 * 64 * 64


def test_registry_plans_s4_at_three_query_groups_on_the_real_parent():
    # The planner is what a recipe calls before a GPU is booked, so it is
    # exercised at the geometry the campaign actually plans against.
    spec = _Spec(LFM2_HIDDEN, LFM2_HIDDEN, LFM2_OWNED)
    assert storage_model("T7", "C10") is not None
    width = storage_width_for_ratio(spec, "T7", 0.25)
    assert width is not None
    assert storage_model("T7", "C10")(spec, width) <= 0.25 * LFM2_OWNED
    assert head_merge_group_plan(
        width, query_heads=LFM2_HEADS, kv_heads=LFM2_KV_HEADS, head_dim=LFM2_HEAD_DIM
    ) == (3, 2)
    # S2 buys seven query groups.  S8 is expressible -- which head DROP never
    # was at any level -- but only at one query basis and one kv basis for the
    # whole block, 0.114583x, where the 24 alignments are 27% of what is stored.
    assert head_merge_group_plan(
        storage_width_for_ratio(spec, "T7", 0.5),
        query_heads=LFM2_HEADS,
        kv_heads=LFM2_KV_HEADS,
        head_dim=LFM2_HEAD_DIM,
    ) == (7, 4)
    s8_width = storage_width_for_ratio(spec, "T7", 0.125)
    assert head_merge_group_plan(
        s8_width,
        query_heads=LFM2_HEADS,
        kv_heads=LFM2_KV_HEADS,
        head_dim=LFM2_HEAD_DIM,
    ) == (1, 1)
    assert c10_head_merge_storage(spec, s8_width) == 360_448
    assert c10_head_merge_storage(spec, s8_width) / LFM2_OWNED == pytest.approx(
        0.1145833333, abs=1e-9
    )


def test_the_planner_returns_a_width_the_kernel_will_accept():
    # storage_width_for_ratio answers 255 for S4 -- the widest integer that
    # fits, and a width the kernel refuses because it does not name whole
    # groups.  A planner that passed it through would book a GPU to receive a
    # refusal.
    spec = _Spec(LFM2_HIDDEN, LFM2_HIDDEN, LFM2_OWNED)
    assert storage_width_for_ratio(spec, "T7", 0.25) == 255
    assert c10_head_merge_admissible_width(spec, 0.25) == 192
    assert c10_head_merge_admissible_width(spec, 0.25) % LFM2_HEAD_DIM == 0
    assert c10_head_merge_admissible_width(spec, 0.5) == 448
    assert c10_head_merge_admissible_width(spec, 0.125) == 64
    assert c10_head_merge_admissible_width(spec, 0.05) is None


def test_storage_model_refuses_a_shape_that_is_not_qkvo():
    from neural_forge.matrix_technique_head_merge import _c10_geometry

    with pytest.raises(ValueError):
        _c10_geometry(_Spec(1024, 1024, 2 * 1024 * 1024), 64)


# ---------------------------------------------------------------- dispatch


def test_t7_at_c10_does_not_shadow_the_other_two_registrations():
    assert resolve("T7", "C10") is c10_head_merge_technique
    assert resolve("T1", "C10") is c10_head_drop_technique
    assert resolve("T7", "C06") is not c10_head_merge_technique


def test_description_is_sealable_and_names_the_mechanism():
    body = describe_head_merge_technique()
    assert body["mechanism"] == "M5"
    assert body["technique"] == "T7"
    assert body["component"] == "C10"
    assert body["head_dim_constant"] == HEAD_MERGE_HEAD_DIM
    json.dumps(body)


# ---------------------------------------------------------------- the merge


def _run(state, evidence, *, level_width, knobs=None):
    request = _request(
        f"C10.T7.S4.L{LAYER:02d}", state, evidence, level_width=level_width, knobs=knobs
    )
    result = c10_head_merge_technique(request)
    validate_result(request, result)
    return result


def test_merge_installs_all_four_parent_shaped_projections():
    generator = torch.Generator().manual_seed(3)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    installed = dict(result.installed_weights)
    assert sorted(installed) == sorted(
        f"model.layers.{LAYER}.self_attn.{name}.weight"
        for name in ("k_proj", "out_proj", "q_proj", "v_proj")
    )
    for key, value in installed.items():
        assert value.shape == state[key].shape
        assert bool(torch.isfinite(value).all())


def test_identity_gate_returns_the_parent_over_all_four_tensors():
    generator = torch.Generator().manual_seed(5)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    assert result.solve["identity_gate_max_abs_deviation"] < 1e-9


def test_exact_group_structure_is_recovered_to_machine_precision():
    generator = torch.Generator().manual_seed(7)
    state = _mergeable_state(generator, query_groups=4, kv_groups=2)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    for role, key in (
        ("q", "q_proj"),
        ("k", "k_proj"),
        ("v", "v_proj"),
        ("o", "out_proj"),
    ):
        assert result.solve["weight_reconstruction_relative_l2"][role] < 1e-9, role
    # And the grouping it found is the one the fixture built: heads h and h+4
    # share a basis, interleaved, which no positional heuristic would produce.
    partition = sorted(sorted(members) for members in result.solve["query_partition"])
    assert partition == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_the_write_refit_wins_on_the_rows_it_was_fit_on():
    # The invariant a least-squares solve actually guarantees, and the only one
    # a receipt may quote without an out-of-sample reading beside it.
    generator = torch.Generator().manual_seed(13)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    assert (
        result.solve["write_fit_relative_l2_refit"]
        <= result.solve["write_fit_relative_l2_procrustes_only"] + 1e-9
    )
    assert 0.0 < result.solve["write_heldout_relative_l2"] < 1.0


def test_a_refit_that_loses_the_inner_split_is_not_installed():
    # 96 fit rows against a 16-column design is exactly the regime where the
    # refit generalizes worse than the prior it started from.  The kernel has
    # to notice that on the FIT rows' own inner split and keep the prior; if it
    # instead adopted the refit, the child would carry a worse O than the
    # Procrustes merge it claims to improve on.
    generator = torch.Generator().manual_seed(13)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    assert result.solve["o_refit_inner_split_applicable"] is True
    adopted = result.solve["o_refit_adopted"]
    assert adopted is (
        result.solve["o_refit_inner_validation_relative_l2"]
        < result.solve["o_prior_inner_validation_relative_l2"]
    )
    if not adopted:
        assert result.solve["write_heldout_relative_l2"] == pytest.approx(
            result.solve["write_heldout_relative_l2_procrustes_only"], abs=1e-12
        )


def test_the_alternation_is_monotone_and_says_so():
    generator = torch.Generator().manual_seed(17)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    assert result.solve["query_family"]["monotone"] is True
    assert result.solve["kv_family"]["monotone"] is True
    for trace in result.solve["query_family"]["objective_trace"]:
        assert trace == sorted(trace, reverse=True)


def test_the_same_recipe_builds_the_same_child_twice():
    generator = torch.Generator().manual_seed(19)
    state = _random_state(generator)
    evidence = _evidence(generator, state)
    first = _run(state, evidence, level_width=4 * HEAD_DIM)
    second = _run(state, evidence, level_width=4 * HEAD_DIM)
    assert first.solve["query_partition"] == second.solve["query_partition"]
    assert first.solve["kv_partition"] == second.solve["kv_partition"]
    for key, value in first.installed_weights.items():
        assert torch.equal(value, second.installed_weights[key])


def test_stored_count_is_the_child_and_agrees_with_the_registered_model():
    generator = torch.Generator().manual_seed(23)
    state = _random_state(generator)
    result = _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM)
    spec = component_spec(CellId.parse(f"C10.T7.S4.L{LAYER:02d}"), state)
    assert result.stored_parameter_count == c10_head_merge_storage(
        spec, 4 * HEAD_DIM, head_dim=HEAD_DIM
    )
    assert result.stored_parameter_count < spec.parent_parameter_count
    assert result.solve["storage_realized_ratio"] < 1.0
    # The installed tensors are parent-shaped and are NOT what is stored; the
    # count has to be the basis-plus-alignment set.
    installed_numel = sum(
        int(value.numel()) for value in result.installed_weights.values()
    )
    assert installed_numel == spec.parent_parameter_count
    assert result.stored_parameter_count < installed_numel


def test_predict_is_the_installed_o_projection():
    generator = torch.Generator().manual_seed(29)
    state = _random_state(generator)
    evidence = _evidence(generator, state)
    result = _run(state, evidence, level_width=4 * HEAD_DIM)
    rows = evidence.heldout_rows("attention.context")
    child_o = result.installed_weights[
        f"model.layers.{LAYER}.self_attn.out_proj.weight"
    ]
    assert torch.allclose(result.predict(rows), rows @ child_o.T, atol=1e-12)


# ---------------------------------------------------------------- refusals


def test_refuses_a_component_it_does_not_model():
    generator = torch.Generator().manual_seed(31)
    state = _random_state(generator)
    request = _request(
        f"C09.T7.S4.L{LAYER:02d}",
        state,
        _evidence(generator, state),
        level_width=4 * HEAD_DIM,
    )
    with pytest.raises(CellTechniqueError, match="attn_per_head kernel only"):
        c10_head_merge_technique(request)


def test_refuses_a_level_that_is_not_whole_groups():
    generator = torch.Generator().manual_seed(37)
    state = _random_state(generator)
    with pytest.raises(CellTechniqueError, match="whole number"):
        _run(state, _evidence(generator, state), level_width=4 * HEAD_DIM + 1)


def test_refuses_a_level_that_contracts_nothing():
    generator = torch.Generator().manual_seed(41)
    state = _random_state(generator)
    with pytest.raises(CellTechniqueError, match="not a contraction"):
        _run(state, _evidence(generator, state), level_width=HEADS * HEAD_DIM)


def test_refuses_a_parent_that_cannot_replay_its_own_write():
    generator = torch.Generator().manual_seed(43)
    state = _random_state(generator)
    evidence = _evidence(generator, state, corrupt_write=True)
    with pytest.raises(CellTechniqueError, match="cannot reproduce"):
        _run(state, evidence, level_width=4 * HEAD_DIM)
