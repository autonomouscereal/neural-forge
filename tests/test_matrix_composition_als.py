"""Contract tests for ADDENDUM 9's alternating joint co-solve (mode M-C).

Wave 11 adds one mechanism and one one-line defect fix, and both of them are
the kind of change that a receipt would not show if it were wrong:

1. **The capture-reuse licence.**  An ALS sweep is only affordable because it
   re-uses one conditioned capture for every component whose input rows nothing
   installed since that capture could have moved.  The licence is a claim about
   LFM2's block dataflow, so the predicate is pinned here on every case that
   matters -- same rung, deeper rung, shallower rung, the token interface, and
   the sweep boundary, where the first component of sweep *n+1* legitimately
   re-uses the last capture of sweep *n*.  (The run itself spends one extra
   capture to check the same claim against real rows; this is the algebra.)

2. **The hybrid evidence object.**  A coordinate step splices the child's
   inputs onto the parent's targets.  If the two halves were ever taken at
   different sites, over different phase lists, over different row counts, or
   over a corpus the conditioned capture reordered, the fit would be against two
   different things and every number downstream of it would be fiction -- so
   each of those refuses rather than warns, and the routing (which endpoint
   comes from which capture) is asserted rather than assumed.  What is *not*
   compared is the two selection digests: they are taken over the captured
   activations themselves, so a capture through a folded child cannot reproduce
   the parent's, and wave 11's first plumbing run died on exactly that
   over-strict guard.

3. **T8's new target knob.**  With the knob absent the kernel must be the
   function every wave 1-10 receipt was sealed under -- byte-identical
   installed weights, not merely similar -- and with it present the fit must
   actually chase the supplied target.  Both directions are pinned, because a
   knob that quietly changes the default would restate 60 sealed children.

4. **The ('T5','C01') double registration**, NOTES-MX-N §9.2's open defect:
   importing the runner after the kernel must not raise, and the fix must not
   change which kernel a cell resolves to.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "experiments"))

torch = pytest.importorskip("torch")

from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueError,
    CellTechniqueRequest,
    component_spec,
    resolve,
    validate_result,
)
from neural_forge.matrix_technique_relation_matching import (  # noqa: E402
    t8_gram_relation_technique,
)

composition = importlib.import_module("run_lfm2_matrix_composition")

HIDDEN = 32
INTERMEDIATE = 80


# --------------------------------------------------------------- fixtures


class _Evidence:
    """The minimal `MatrixCellEvidence` the kernels and the splice need."""

    def __init__(
        self, fit, heldout, *, selection="0" * 64, layer_index=5, phases=("p0",)
    ):
        self._fit = dict(fit)
        self._heldout = dict(heldout)
        self._selection = selection
        self.layer_index = layer_index
        self.phases = tuple(phases)

    @property
    def endpoints(self):
        return tuple(sorted(self._fit))

    @property
    def selection_sha256(self):
        return self._selection

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


def _parent_state(generator):
    def randn(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator) / (
            shape[-1] ** 0.5
        )

    return {
        "model.layers.5.feed_forward.w1.weight": randn(INTERMEDIATE, HIDDEN),
        "model.layers.5.feed_forward.w3.weight": randn(INTERMEDIATE, HIDDEN),
        "model.layers.5.feed_forward.w2.weight": randn(HIDDEN, INTERMEDIATE),
        "model.layers.12.self_attn.k_proj.weight": randn(HIDDEN, HIDDEN),
    }


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


def _mlp_evidence(generator, *, rows=256, heldout=64, target=None):
    scale = torch.logspace(0.0, -1.5, HIDDEN, dtype=torch.float64)
    fit_input = (
        torch.randn((rows, HIDDEN), dtype=torch.float64, generator=generator) * scale
    )
    held_input = (
        torch.randn((heldout, HIDDEN), dtype=torch.float64, generator=generator) * scale
    )
    fit = {"mlp.input": fit_input}
    held = {"mlp.input": held_input}
    if target is not None:
        fit["mlp.gate"] = target[0]
        held["mlp.gate"] = target[1]
    return _Evidence(fit, held)


# ------------------------------------------------- the capture-reuse licence


def test_the_licence_refreshes_for_a_deeper_rung_in_the_same_block():
    # w2 reads what w1 and w3 produced, so a re-solved w1 invalidates its rows.
    w1 = composition._als_position(CellId.parse("C11.T8.S4.L05"))
    w2 = composition._als_position(CellId.parse("C13.T8.S4.L05"))
    assert composition._capture_is_stale(w2, [w1]) is True


def test_the_licence_re_uses_one_capture_across_a_shared_rung():
    # w1 and w3 both read mlp.input and neither reads the other, which is the
    # entire reason an MLP lane costs two captures per sweep and not three.
    w1 = composition._als_position(CellId.parse("C11.T8.S4.L05"))
    w3 = composition._als_position(CellId.parse("C12.T8.S4.L05"))
    assert composition._capture_is_stale(w3, [w1]) is False
    assert composition._capture_is_stale(w1, [w3]) is False


def test_the_licence_ignores_everything_downstream():
    # This is the sweep boundary: the last component of sweep n is the deepest,
    # and the first component of sweep n+1 is the shallowest, so the shallowest
    # legitimately re-uses the capture the deepest one was solved against.
    shallow = composition._als_position(CellId.parse("C11.T8.S4.L04"))
    deep = composition._als_position(CellId.parse("C07.T5.S4.L12"))
    assert composition._capture_is_stale(shallow, [deep]) is False
    assert composition._capture_is_stale(deep, [shallow]) is True


def test_the_licence_treats_the_token_interface_as_upstream_of_every_block():
    token = composition._als_position(CellId.parse("C01.T5.S4"))
    first = composition._als_position(CellId.parse("C03.T5.S4.L00"))
    assert token[0] == composition._GLOBAL_LAYER
    assert composition._capture_is_stale(first, [token]) is True
    assert composition._capture_is_stale(token, [first]) is False


def test_attention_projections_share_a_rung_and_out_proj_does_not():
    q = composition._als_position(CellId.parse("C06.T1.S4.L12"))
    k = composition._als_position(CellId.parse("C07.T5.S4.L12"))
    o_same_block = composition._als_position(CellId.parse("C09.T5.S4.L12"))
    o_shallower = composition._als_position(CellId.parse("C09.T5.S4.L02"))
    # q/k/v all read operator.input, so one capture serves the three of them.
    assert composition._capture_is_stale(k, [q]) is False
    # out_proj reads attention.context, which q has just moved.
    assert composition._capture_is_stale(o_same_block, [q]) is True
    # A shallower block is never invalidated by a deeper one.
    assert composition._capture_is_stale(o_shallower, [q]) is False


# ------------------------------------------------------- the hybrid evidence


def _hybrid_pair(generator):
    child = _mlp_evidence(generator)
    parent = _Evidence(
        {
            "mlp.input": torch.randn(
                (256, HIDDEN), dtype=torch.float64, generator=generator
            ),
            "mlp.gate": torch.randn(
                (256, INTERMEDIATE), dtype=torch.float64, generator=generator
            ),
        },
        {
            "mlp.input": torch.randn(
                (64, HIDDEN), dtype=torch.float64, generator=generator
            ),
            "mlp.gate": torch.randn(
                (64, INTERMEDIATE), dtype=torch.float64, generator=generator
            ),
        },
    )
    return child, parent


def test_the_splice_routes_inputs_to_the_child_and_targets_to_the_parent():
    generator = torch.Generator().manual_seed(11)
    child, parent = _hybrid_pair(generator)
    hybrid = composition._AlsHybridEvidence(
        child=child, parent=parent, parent_served=["mlp.gate"]
    )
    assert hybrid.served_by("mlp.input") == "child"
    assert hybrid.served_by("mlp.gate") == "parent"
    assert torch.equal(hybrid.fit_rows("mlp.input"), child.fit_rows("mlp.input"))
    assert torch.equal(hybrid.fit_rows("mlp.gate"), parent.fit_rows("mlp.gate"))
    assert torch.equal(
        hybrid.heldout_rows("mlp.gate"), parent.heldout_rows("mlp.gate")
    )
    assert hybrid.fit_row_count == 256
    assert hybrid.heldout_row_count == 64


def test_the_splice_does_not_compare_selection_digests():
    # The digest is taken over the captured activations themselves, so a
    # capture through a folded child cannot reproduce the parent's.  Requiring
    # it to would be requiring the child to be the parent, which is the one
    # thing an ALS step knows is false.  Wave 11's first plumbing run died on
    # exactly this over-strict guard; it is now a recorded field, not a gate.
    generator = torch.Generator().manual_seed(12)
    child, parent = _hybrid_pair(generator)
    other = _Evidence(
        {
            "mlp.input": parent.fit_rows("mlp.input"),
            "mlp.gate": parent.fit_rows("mlp.gate"),
        },
        {
            "mlp.input": parent.heldout_rows("mlp.input"),
            "mlp.gate": parent.heldout_rows("mlp.gate"),
        },
        selection="a" * 64,
    )
    hybrid = composition._AlsHybridEvidence(
        child=child, parent=other, parent_served=["mlp.gate"]
    )
    assert hybrid.selection_sha256 == child.selection_sha256


def test_the_splice_refuses_two_different_sites():
    generator = torch.Generator().manual_seed(31)
    child, parent = _hybrid_pair(generator)
    elsewhere = _Evidence(
        {
            "mlp.input": parent.fit_rows("mlp.input"),
            "mlp.gate": parent.fit_rows("mlp.gate"),
        },
        {
            "mlp.input": parent.heldout_rows("mlp.input"),
            "mlp.gate": parent.heldout_rows("mlp.gate"),
        },
        layer_index=4,
    )
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._AlsHybridEvidence(
            child=child, parent=elsewhere, parent_served=["mlp.gate"]
        )
    assert "two sites" in str(caught.value)


def test_the_splice_refuses_two_different_phase_lists():
    generator = torch.Generator().manual_seed(37)
    child, parent = _hybrid_pair(generator)
    other_phases = _Evidence(
        {
            "mlp.input": parent.fit_rows("mlp.input"),
            "mlp.gate": parent.fit_rows("mlp.gate"),
        },
        {
            "mlp.input": parent.heldout_rows("mlp.input"),
            "mlp.gate": parent.heldout_rows("mlp.gate"),
        },
        phases=("p0", "p1"),
    )
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._AlsHybridEvidence(
            child=child, parent=other_phases, parent_served=["mlp.gate"]
        )
    assert "phase lists" in str(caught.value)


def test_row_alignment_refuses_a_split_that_was_not_inherited(tmp_path):
    base = {
        "manifest_sha256": "b" * 64,
        "corpus": {"ordered_batch_input_sha256": ["r0", "r1"]},
    }
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": "c" * 64,
                "split": {
                    "method": "whole-record-seeded-sha256-ranking",
                    "inherited_from_manifest_sha256": None,
                    "rows": [
                        {"batch_index": 0, "record_id": "r0", "split": "fit"},
                        {"batch_index": 1, "record_id": "r1", "split": "heldout"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._als_verify_row_alignment(
            split_manifest_path=path, base_manifest=base
        )
    assert "not inherited" in str(caught.value)


def test_row_alignment_refuses_a_reordered_corpus(tmp_path):
    base = {
        "manifest_sha256": "b" * 64,
        "corpus": {"ordered_batch_input_sha256": ["r0", "r1"]},
    }
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": "c" * 64,
                "split": {
                    "method": "inherit",
                    "inherited_from_manifest_sha256": "b" * 64,
                    "rows": [
                        {"batch_index": 0, "record_id": "r1", "split": "fit"},
                        {"batch_index": 1, "record_id": "r0", "split": "heldout"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._als_verify_row_alignment(
            split_manifest_path=path, base_manifest=base
        )
    assert "different batches" in str(caught.value)


def test_row_alignment_accepts_the_inherited_split_in_corpus_order(tmp_path):
    base = {
        "manifest_sha256": "b" * 64,
        "corpus": {"ordered_batch_input_sha256": ["r0", "r1"]},
    }
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": "c" * 64,
                "split": {
                    "method": "inherit",
                    "inherited_from_manifest_sha256": "b" * 64,
                    "rows": [
                        {"batch_index": 0, "record_id": "r0", "split": "fit"},
                        {"batch_index": 1, "record_id": "r1", "split": "heldout"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    verified = composition._als_verify_row_alignment(
        split_manifest_path=path, base_manifest=base
    )
    assert verified["batches"] == 2
    assert verified["ordered_record_ids_match_parent_base"] is True


def test_the_splice_refuses_two_captures_with_different_row_counts():
    generator = torch.Generator().manual_seed(13)
    child, _parent = _hybrid_pair(generator)
    short = _Evidence(
        {"mlp.gate": torch.zeros((8, INTERMEDIATE), dtype=torch.float64)},
        {"mlp.gate": torch.zeros((4, INTERMEDIATE), dtype=torch.float64)},
    )
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._AlsHybridEvidence(
            child=child, parent=short, parent_served=["mlp.gate"]
        )
    assert "different row counts" in str(caught.value)


# ------------------------------------------------------------ T8's target


def test_t8_without_the_knob_is_byte_identical_to_the_sealed_kernel():
    generator = torch.Generator().manual_seed(17)
    state = _parent_state(generator)
    evidence = _mlp_evidence(generator)
    request = _request("C11.T8.S4.L05", state, evidence, level_width=12)
    result = t8_gram_relation_technique(request)
    validate_result(request, result)
    # The reference is the kernel's own documented default: the target is the
    # parent operator applied to the design rows.  Re-deriving it here and
    # feeding it back through the knob must reproduce the same tensor exactly.
    parent = state["model.layers.5.feed_forward.w1.weight"]
    target_fit = evidence.fit_rows("mlp.input") @ parent.T
    target_held = evidence.heldout_rows("mlp.input") @ parent.T
    explicit = _request(
        "C11.T8.S4.L05",
        state,
        _Evidence(
            {
                "mlp.input": evidence.fit_rows("mlp.input"),
                "mlp.gate": target_fit,
            },
            {
                "mlp.input": evidence.heldout_rows("mlp.input"),
                "mlp.gate": target_held,
            },
        ),
        level_width=12,
        knobs={"relation_target_endpoint": "mlp.gate"},
    )
    mirrored = t8_gram_relation_technique(explicit)
    key = "model.layers.5.feed_forward.w1.weight"
    assert torch.equal(result.installed_weights[key], mirrored.installed_weights[key])
    assert result.solve["relation_target_endpoint"] is None
    assert mirrored.solve["relation_target_endpoint"] == "mlp.gate"


def test_t8_with_the_knob_chases_the_supplied_target():
    generator = torch.Generator().manual_seed(19)
    state = _parent_state(generator)
    key = "model.layers.5.feed_forward.w1.weight"
    fit_input = torch.randn((256, HIDDEN), dtype=torch.float64, generator=generator)
    held_input = torch.randn((64, HIDDEN), dtype=torch.float64, generator=generator)
    # A target that is a DIFFERENT operator's output on the same rows: if the
    # knob is honoured the fit must land nearer that operator than the parent's.
    other = state["model.layers.5.feed_forward.w3.weight"]
    evidence = _Evidence(
        {"mlp.input": fit_input, "mlp.gate": fit_input @ other.T},
        {"mlp.input": held_input, "mlp.gate": held_input @ other.T},
    )
    request = _request(
        "C11.T8.S4.L05",
        state,
        evidence,
        level_width=HIDDEN,
        knobs={"relation_target_endpoint": "mlp.gate"},
    )
    result = t8_gram_relation_technique(request)
    validate_result(request, result)
    installed = result.installed_weights[key]
    to_target = float(
        (held_input @ installed.T - held_input @ other.T).norm()
        / (held_input @ other.T).norm()
    )
    to_parent = float(
        (held_input @ installed.T - held_input @ state[key].T).norm()
        / (held_input @ state[key].T).norm()
    )
    assert to_target < 0.05
    assert to_target < to_parent


def test_t8_refuses_a_target_that_is_not_row_aligned_with_the_design():
    generator = torch.Generator().manual_seed(23)
    state = _parent_state(generator)
    evidence = _Evidence(
        {
            "mlp.input": torch.randn(
                (256, HIDDEN), dtype=torch.float64, generator=generator
            ),
            "mlp.gate": torch.randn(
                (128, INTERMEDIATE), dtype=torch.float64, generator=generator
            ),
        },
        {
            "mlp.input": torch.randn(
                (64, HIDDEN), dtype=torch.float64, generator=generator
            ),
            "mlp.gate": torch.randn(
                (64, INTERMEDIATE), dtype=torch.float64, generator=generator
            ),
        },
    )
    request = _request(
        "C11.T8.S4.L05",
        state,
        evidence,
        level_width=12,
        knobs={"relation_target_endpoint": "mlp.gate"},
    )
    with pytest.raises(CellTechniqueError) as caught:
        t8_gram_relation_technique(request)
    assert "row-aligned" in str(caught.value)


# --------------------------------------------- the capture self-check gate


def test_the_capture_selfcheck_is_exact_for_the_installed_weight():
    generator = torch.Generator().manual_seed(29)
    state = _parent_state(generator)
    spec = component_spec(CellId.parse("C11.T8.S4.L05"), state)
    installed = torch.randn(
        (INTERMEDIATE, HIDDEN), dtype=torch.float64, generator=generator
    )
    held_input = torch.randn((64, HIDDEN), dtype=torch.float64, generator=generator)
    evidence = _Evidence(
        {"mlp.input": held_input, "mlp.gate": held_input @ installed.T},
        {"mlp.input": held_input, "mlp.gate": held_input @ installed.T},
    )
    exact = composition._als_capture_selfcheck(
        spec=spec,
        child_evidence=evidence,
        installed={"model.layers.5.feed_forward.w1.weight": installed},
    )
    assert exact < 1e-12
    # A capture that does not reproduce its own installed weight is exactly the
    # failure the gate exists to catch, and it has to be visible as a number.
    wrong = composition._als_capture_selfcheck(
        spec=spec,
        child_evidence=evidence,
        installed={"model.layers.5.feed_forward.w1.weight": state[spec.weight_keys[0]]},
    )
    assert wrong > 0.1


# ------------------------------------------------ the best-sweep restore


def test_adopting_a_sweeps_entries_replaces_them_in_place():
    # The receipt already holds references to these dicts, so a rebind would
    # seal the sweep that was discarded.  This is the wave-11 defect: MXO1
    # restored sweep 0's WEIGHTS and sealed sweep 1's DIGESTS, which describes
    # a child that does not exist.
    entries = [
        {
            "isolation_cell_id": "C11.T8.S4.L05",
            "installed_tensor_digests": ["w=sweep1"],
            "solve": {"rank": 182},
        }
    ]
    held = entries[0]
    composition._als_adopt_entries(
        entries,
        [
            {
                "isolation_cell_id": "C11.T8.S4.L05",
                "installed_tensor_digests": ["w=sweep0"],
                "solve": {"rank": 182},
            }
        ],
    )
    assert entries[0] is held
    assert held["installed_tensor_digests"] == ["w=sweep0"]


def test_adopting_refuses_a_reordered_or_resized_component_list():
    entries = [{"isolation_cell_id": "C11.T8.S4.L05"}]
    with pytest.raises(composition.MatrixCompositionRunError):
        composition._als_adopt_entries(entries, [])
    with pytest.raises(composition.MatrixCompositionRunError) as caught:
        composition._als_adopt_entries(entries, [{"isolation_cell_id": "C13.T8.S4.L05"}])
    assert "order moved" in str(caught.value)


# ------------------------------------------ the ('T5','C01') double registration


def _in_subprocess(body: str) -> str:
    """Run an import-order probe in its own interpreter.

    Both checks below are about *global* state -- the dispatch table -- and the
    suite already carries one order-dependent failure caused by a test that
    mutated it and never put it back (NOTES-MX-N §9.2, the four collateral
    victims).  Adding a second one to test the fix for the first would be
    absurd, so these run out of process and this process's registry is left
    exactly as it was found.
    """

    import subprocess

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(_ROOT / 'src')!r})\n"
        f"sys.path.insert(0, {str(_ROOT / 'experiments')!r})\n"
    ) + body
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return completed.stdout.strip()


def test_importing_the_runner_after_the_kernel_does_not_raise():
    # NOTES-MX-N §9.2: alphabetical collection of the whole tests/ tree imports
    # the kernel first, and the runner's bare register() then blew up at import
    # with ``('T5','C01') is already registered``.
    output = _in_subprocess(
        "import neural_forge.matrix_technique_token_embedding\n"
        "import run_lfm2_matrix_cell\n"
        "from neural_forge.matrix_technique_registry import resolve\n"
        "print(resolve('T5', 'C01').__module__)\n"
    )
    assert output


def test_the_fix_does_not_change_which_kernel_c01_resolves_to():
    # The order a real run has: the runner registers at import, then
    # discover_technique_modules() imports the technique lane, whose
    # registration replaces the runner's.  The resolved kernel is the technique
    # lane's -- which is what waves 9 and 10 sealed their C01 folds with.
    output = _in_subprocess(
        "import run_lfm2_matrix_cell as runner\n"
        "runner.discover_technique_modules()\n"
        "from neural_forge.matrix_technique_registry import resolve\n"
        "print(resolve('T5', 'C01').__module__)\n"
    )
    assert output == "neural_forge.matrix_technique_token_embedding"
