"""Contract tests for ADDENDUM 7: receipt schema v2 and the widened site registry.

Wave 6 ran a per-layer ladder and it stopped after three rungs, not because the
model refused but because `matrix_cell_receipt.COMPONENTS` registered three
sites per component and `_validate_cell` enforced it (`NOTES-MX-J` §5).  §5.1
then measured what that policy costs: ~0.923x net even if *every* registered
site folded at a true quarter, because eleven of the fourteen layers had no
registered site at all.  ADDENDUM 7 widens the registry to the parent's real
architecture and versions the schema so the widening cannot reach backwards.

Three things have to hold at once, and each one is a way this change could be
wrong rather than a restatement that it is right:

1. **v2 widened to the architecture, not to everything.**  A short_conv
   component is still refused at an attention layer, an attention component is
   still refused at a short_conv layer, and no component reaches layer 14.  If
   the widening were a blanket ``range(14)`` the second and third tests here
   would pass and the first would not.
2. **v1 is frozen and is still what a v1 row is read against.**  The 348 rows
   waves 1-6 sealed re-verify under the narrow registry for ever.  A receipt
   that declares v1 and names an L04 site is refused -- the widening is not
   retroactive, so a wave-1 row cannot acquire a meaning it was not sealed with.
3. **Only sites moved.**  Same mandatory fields, same chain, same
   ``campaign_pass``, same evidence-cache names -- that last one matters
   materially: the 21 authenticated site evidence caches on the run host are
   named by `evidence_cache_name`, and a registry change that renamed one would
   silently invalidate every cache the campaign has built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    ATTENTION_LAYERS,
    COMPONENTS,
    CONV_LAYERS,
    LATEST_SCHEMA_VERSION,
    MLP_LAYERS,
    RECEIPT_SCHEMA_VERSIONS,
    SCHEMA,
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    CellId,
    MatrixCellReceiptError,
    blank_receipt,
    campaign_pass,
    component_sites,
    first_divergent_metric,
    known_cell_ids,
    ledger_line,
    probe_set_sha256,
    schema_version_of,
    seal_receipt,
    verify_seal,
)
from neural_forge.matrix_composition import evidence_cache_name  # noqa: E402

#: The v1 registry, restated here as a literal and not imported, so that an
#: edit to the module's own frozen copy is a test failure rather than a silent
#: redefinition of what 348 sealed rows are validated against.
_V1_SITES = {
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


def _receipt(cell_id: str, schema_version: int = LATEST_SCHEMA_VERSION):
    """A minimal complete receipt at one cell id, sealed under one version."""

    body = blank_receipt(
        cell_id=cell_id,
        cell_instance_id="0" * 32,
        campaign_wave=7,
        repo_head="670ce320",
        src_tree_sha="a" * 64,
        probe_set_sha=probe_set_sha256(("probe",)),
        smoke_prompt_sha=probe_set_sha256(("smoke",)),
        compute_architecture="sm86 NVIDIA GeForce RTX 3090",
        schema_version=schema_version,
    )
    body["calibration"] = {
        "copy_exact_reproduction_relL2": 1e-9,
        "passed": True,
        "applicable": True,
    }
    body["solve"].update(
        {
            "solver_path": "t8_gram_relation",
            "ridge_relative": 0.0,
            "condition_number": 12.5,
            "condition_cap_saturated": False,
            "prior_norm": 3.0,
            "correction_norm": 0.5,
            "requested_clamp_ratio": 0.16,
            "allowed_clamp_ratio": 1.0,
            "clamp_engaged": False,
            "stored_parameter_count": 655_360,
            "parent_parameter_count": 2_621_440,
            "storage_bound_ratio": 0.25,
            "storage_bound_satisfied": True,
        }
    )
    body["boundary"] = {
        "applicable": True,
        "local_solve_applicable": True,
        "macrocell_endpoint": "residual.post_layer",
        "write_heldout_relL2": 0.2,
        "write_heldout_cos": 0.99,
        "post_operator_heldout_relL2": 0.09,
        "post_operator_heldout_cos": 0.995,
        "endpoint_band_le_0p4": True,
    }
    body["composed"] = {
        "applicable": True,
        "terminal_heldout_relL2": 0.004,
        "emit_zero_floor_passed": True,
        "vs_copy_exact_control_delta": 0.004,
        "norm_ratio_per_layer_minmax": [0.98, 1.02],
    }
    body["behavior"] = {
        "free_run": {
            "probe_prompts": 8,
            "tokens_per_prompt": 64,
            "token_match_vs_control": 0.5508,
            "mean_lcp": 0.52,
        },
        "decode_smoke": [
            {"prompt_sha": "b" * 64, "output_verbatim": "hello", "terminated": "eos"}
        ],
    }
    return seal_receipt(body)


# --------------------------------------------------------------------------
# 1. the versions themselves
# --------------------------------------------------------------------------


def test_both_schema_versions_are_live_and_new_work_is_v2():
    """ADDENDUM 12 re-pins this: v3 is live and is what new work seals under.

    The clause ADDENDUM 7 wrote here is unchanged in substance -- both of its
    versions are still live and still map to the same integers -- and the only
    edit is that a third version now exists.  v2's *content* is asserted below by
    `test_v3_reads_v2s_site_registry_unchanged`, which is the thing this test was
    really protecting.
    """

    assert RECEIPT_SCHEMA_VERSIONS == {SCHEMA_V1: 1, SCHEMA_V2: 2, SCHEMA_V3: 3}
    assert SCHEMA == SCHEMA_V3
    assert LATEST_SCHEMA_VERSION == 3
    assert schema_version_of({"schema": SCHEMA_V1}) == 1
    assert schema_version_of({"schema": SCHEMA_V2}) == 2
    assert schema_version_of({"schema": SCHEMA_V3}) == 3


def test_v3_reads_v2s_site_registry_unchanged():
    """ADDENDUM 12 bumped the version and touched no site policy."""

    for component in sorted(_V1_SITES):
        assert component_sites(component, 2) == component_sites(component, 3)


def test_an_unknown_schema_string_fails_closed():
    for declared in (None, "", "neural-forge-lfm2-matrix-cell-receipt-v4", 2):
        with pytest.raises(MatrixCellReceiptError):
            schema_version_of({"schema": declared})


def test_an_unknown_schema_version_cannot_name_sites():
    for version in (0, 4, -1):
        with pytest.raises(MatrixCellReceiptError):
            component_sites("C13", version)


# --------------------------------------------------------------------------
# 2. the widening is the architecture, and nothing wider
# --------------------------------------------------------------------------


def test_the_layer_type_map_is_the_parents_own():
    """`config.json`'s ``layer_types``, and the shipped tensor keys agree with it.

    short_conv at 0/1/3/5/7/9/11/13, full_attention at 2/4/6/8/10/12, and a
    SwiGLU ``feed_forward`` in every one of the fourteen -- which is why the MLP
    components widen to all fourteen layers and the operator components do not.
    """

    assert CONV_LAYERS == (0, 1, 3, 5, 7, 9, 11, 13)
    assert ATTENTION_LAYERS == (2, 4, 6, 8, 10, 12)
    assert MLP_LAYERS == tuple(range(14))
    assert set(CONV_LAYERS) | set(ATTENTION_LAYERS) == set(MLP_LAYERS)
    assert not set(CONV_LAYERS) & set(ATTENTION_LAYERS)


def test_v2_sites_are_the_components_own_block_kind():
    expected = {
        "C03": CONV_LAYERS,
        "C04": CONV_LAYERS,
        "C05": CONV_LAYERS,
        "C06": ATTENTION_LAYERS,
        "C07": ATTENTION_LAYERS,
        "C08": ATTENTION_LAYERS,
        "C09": ATTENTION_LAYERS,
        "C10": ATTENTION_LAYERS,
        "C11": MLP_LAYERS,
        "C12": MLP_LAYERS,
        "C13": MLP_LAYERS,
        "C14": MLP_LAYERS + ("final",),
    }
    for component, sites in expected.items():
        assert component_sites(component, 2) == sites, component
    for component in ("C01", "C02", "C15"):
        assert component_sites(component, 2) == "global"


def test_v2_still_refuses_the_wrong_block_kind_and_a_layer_that_does_not_exist():
    for text in (
        "C05.T8.S4.L04",  # conv_out_proj at an attention layer
        "C03.T1.S4.L02",  # conv_in_proj at an attention layer
        "C07.T5.S4.L05",  # attn_k at a short_conv layer
        "C06.T1.S4.L13",  # attn_q at a short_conv layer
        "C10.T1.S4.L07",  # attn_per_head at a short_conv layer
        "C13.T8.S4.L14",  # there is no fifteenth layer
        "C11.T8.S4.L99",
        "C01.T5.X4.L05",  # a global component still has no site
    ):
        with pytest.raises(MatrixCellReceiptError):
            CellId.parse(text, schema_version=2)


def test_v2_admits_exactly_the_new_rungs_the_wave_needs():
    for text in (
        "C13.T8.S4.L04",
        "C13.T8.S4.L06",
        "C13.T8.S4.L03",
        "C13.T8.S4.L07",
        "C07.T5.S4.L02",
        "C07.T5.S4.L04",
        "C07.T5.S4.L06",
        "C11.T8.S4.L05",
        "C11.T8.S4.L13",
        "C05.T8.S4.L07",
    ):
        assert CellId.parse(text, schema_version=2).text == text


# --------------------------------------------------------------------------
# 3. v1 is frozen, and the widening does not reach backwards
# --------------------------------------------------------------------------


def test_v1_registry_is_frozen_verbatim():
    for component, sites in _V1_SITES.items():
        assert component_sites(component, 1) == sites, component
    assert set(_V1_SITES) == set(COMPONENTS)


def test_v1_sites_are_a_subset_of_v2_sites_for_every_component():
    """v2 widens.  It never moves or drops a site a sealed row already names."""

    for component in COMPONENTS:
        one = component_sites(component, 1)
        two = component_sites(component, 2)
        if one == "global":
            assert two == "global", component
            continue
        assert set(one).issubset(set(two)), component


def test_a_v1_receipt_may_not_name_a_v2_only_site():
    body = _receipt("C13.T8.S4.L05", schema_version=1)
    assert body["schema"] == SCHEMA_V1
    assert verify_seal(body) == body["receipt_sha"]
    # The same cell id one layer over is a v2 site.  Sealing it as v1 must fail:
    # a wave-1..6 row is validated against the registry its wave was sealed
    # under, and nothing may write a new row that claims otherwise.
    with pytest.raises(MatrixCellReceiptError):
        _receipt("C13.T8.S4.L04", schema_version=1)
    # And a v1 body edited after the fact to carry a v2 site is refused too.
    forged = dict(body)
    forged["cell_id"] = "C13.T8.S4.L04"
    with pytest.raises(MatrixCellReceiptError):
        seal_receipt(forged)


def test_a_v1_receipt_still_seals_verifies_and_walks_the_same_chain():
    old = _receipt("C13.T8.S4.L05", schema_version=1)
    new = _receipt("C13.T8.S4.L05", schema_version=2)
    assert verify_seal(old) == old["receipt_sha"]
    assert verify_seal(new) == new["receipt_sha"]
    # Only the schema string differs, so the digest differs -- and every clause
    # that reads the numbers returns the same answer under both versions.
    assert first_divergent_metric(old) == first_divergent_metric(new)
    assert campaign_pass(old) == campaign_pass(new)
    assert old["outcome"] == new["outcome"]
    assert {k: v for k, v in old.items() if k not in ("schema", "receipt_sha", "sealed_at")} == {
        k: v for k, v in new.items() if k not in ("schema", "receipt_sha", "sealed_at")
    }


# --------------------------------------------------------------------------
# 4. only sites moved
# --------------------------------------------------------------------------


def test_every_v1_site_keeps_its_evidence_cache_name():
    """The 21 authenticated site caches on the run host must stay reachable.

    `evidence_cache_name` names a cache after the *lowest* component id sharing
    the endpoint set among the components legal at that site, so widening the
    legal set at a site is exactly the kind of change that could rename one.
    It does not, because every component v2 adds at an already-registered site
    has a higher id than the canonical one already there -- but that is an
    argument, and this is the check.
    """

    expected = {
        ("C03", "L00"): "L00-C03",
        ("C03", "L05"): "L05-C03",
        ("C03", "L13"): "L13-C03",
        ("C04", "L00"): "L00-C04",
        ("C04", "L05"): "L05-C04",
        ("C04", "L13"): "L13-C04",
        ("C05", "L00"): "L00-C05",
        ("C05", "L05"): "L05-C05",
        ("C05", "L13"): "L13-C05",
        ("C06", "L02"): "L02-C06",
        ("C06", "L08"): "L08-C06",
        ("C06", "L12"): "L12-C06",
        ("C07", "L02"): "L02-C07",
        ("C07", "L08"): "L08-C07",
        ("C07", "L12"): "L12-C07",
        ("C08", "L02"): "L02-C08",
        ("C08", "L08"): "L08-C08",
        ("C08", "L12"): "L12-C08",
        ("C09", "L02"): "L02-C09",
        ("C09", "L08"): "L08-C09",
        ("C09", "L12"): "L12-C09",
        ("C10", "L02"): "L02-C09",
        ("C10", "L08"): "L08-C09",
        ("C11", "L00"): "L00-C11",
        ("C11", "L05"): "L05-C11",
        ("C11", "L13"): "L13-C11",
        ("C12", "L00"): "L00-C12",
        ("C12", "L05"): "L05-C12",
        ("C12", "L13"): "L13-C12",
        ("C13", "L00"): "L00-C13",
        ("C13", "L05"): "L05-C13",
        ("C13", "L13"): "L13-C13",
        ("C14", "L00"): "L00-C03",
        ("C14", "FINAL"): "FINAL-C14",
    }
    for (component, site), name in expected.items():
        cell = CellId(
            component=component,
            technique="T8",
            level="NA" if component == "C14" else "S4",
            site=site,
        )
        assert evidence_cache_name(cell) == name, (component, site)


def test_a_new_site_gets_its_own_cache_name():
    for text, name in (
        ("C13.T8.S4.L04", "L04-C13"),
        ("C13.T8.S4.L06", "L06-C13"),
        ("C07.T5.S4.L04", "L04-C07"),
        ("C11.T8.S4.L05", "L05-C11"),
        ("C05.T8.S4.L07", "L07-C05"),
    ):
        assert evidence_cache_name(CellId.parse(text)) == name


def test_the_ledger_row_records_which_registry_sealed_it():
    old = ledger_line(_receipt("C13.T8.S4.L05", schema_version=1))
    new = ledger_line(_receipt("C13.T8.S4.L04", schema_version=2))
    assert old["receipt_schema_version"] == 1
    assert new["receipt_schema_version"] == 2
    assert new["site"] == "L04"
    # The ledger schema itself does NOT move: waves 1-6's rows stay readable and
    # a row without the key is a v1 row.
    assert old["schema"] == new["schema"]


def test_known_cell_ids_widen_and_all_reparse():
    one = known_cell_ids(schema_version=1)
    two = known_cell_ids(schema_version=2)
    assert set(one).issubset(set(two))
    assert len(two) > len(one)
    for text in two:
        assert CellId.parse(text, schema_version=2).text == text
    # And every v1 id is still a v1 id.
    for text in one:
        assert CellId.parse(text, schema_version=1).text == text
