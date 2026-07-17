from __future__ import annotations

import itertools
import math

import pytest
import torch

from neural_forge.hf_lfm2_causal_residual_entropy_value_fold import (
    CREVF_ARTIFACT_FIXED_BYTES,
    CREVF_RECORD_ENVELOPE_BYTES,
    CrevfBudgetError,
    CrevfError,
    CrevfAuxiliarySpec,
    account_crevf_artifact,
    build_crevf_exact_spec,
    build_crevf_quantized_spec,
    decode_crevf_artifact,
    encode_crevf_auxiliary,
    encode_crevf_tensor,
)
from neural_forge.hf_lfm2_crevf_budget import (
    CrevfSite,
    CrevfSiteOption,
    plan_crevf_sites,
    role_minimum_integer_width,
    role_requires_exact_bf16,
    semantic_minimum_integer_width,
    tensor_identity_minimum_width,
)


def _option(
    *,
    option_id: str,
    name: str,
    role: str,
    values: torch.Tensor,
    width: int,
    loss: int,
) -> CrevfSiteOption:
    spec = (
        build_crevf_exact_spec(name=name, role=role, values=values)
        if width == 16
        else build_crevf_quantized_spec(
            name=name,
            role=role,
            values=values,
            group_bit_widths=width,
            group_size=256,
        )
    )
    return CrevfSiteOption(
        option_id=option_id,
        tensor_blob=encode_crevf_tensor(spec),
        loss_units=loss,
    )


def _frontier_sites():
    generator = torch.Generator().manual_seed(4401)
    q_values = torch.randn(512, generator=generator).to(torch.bfloat16)
    norm_values = torch.linspace(0.75, 1.25, 32).to(torch.bfloat16)
    ffn_values = torch.randn(2048, generator=generator).to(torch.bfloat16)
    q6 = _option(
        option_id="q6-invalid",
        name="model.layers.0.self_attn.q_proj.weight",
        role="q_proj",
        values=q_values,
        width=6,
        loss=0,
    )
    q7 = _option(
        option_id="q7",
        name="model.layers.0.self_attn.q_proj.weight",
        role="q_proj",
        values=q_values,
        width=7,
        loss=1,
    )
    norm4 = _option(
        option_id="norm4-invalid",
        name="model.layers.0.operator_norm.weight",
        role="operator_norm",
        values=norm_values,
        width=4,
        loss=0,
    )
    norm_exact = _option(
        option_id="norm-exact",
        name="model.layers.0.operator_norm.weight",
        role="operator_norm",
        values=norm_values,
        width=16,
        loss=1,
    )
    ffn4 = _option(
        option_id="ffn4",
        name="model.layers.0.feed_forward.w2.weight",
        role="ffn_w2",
        values=ffn_values,
        width=4,
        loss=20,
    )
    ffn8 = _option(
        option_id="ffn8",
        name="model.layers.0.feed_forward.w2.weight",
        role="ffn_w2",
        values=ffn_values,
        width=8,
        loss=0,
    )
    return (
        CrevfSite(
            site_id="model.layers.0.self_attn.q_proj.weight",
            role="q_proj",
            options=(q6, q7),
        ),
        CrevfSite(
            site_id="model.layers.0.operator_norm.weight",
            role="operator_norm",
            options=(norm4, norm_exact),
        ),
        CrevfSite(
            site_id="model.layers.0.feed_forward.w2.weight",
            role="ffn_w2",
            options=(ffn4, ffn8),
        ),
    ), (q6, q7, norm4, norm_exact, ffn4, ffn8)


def test_actual_byte_planner_enforces_floors_and_exact_65_percent_ceiling() -> None:
    sites, options = _frontier_sites()
    _q6, q7, _norm4, norm_exact, ffn4, ffn8 = options
    low_checkpoint = (
        CREVF_ARTIFACT_FIXED_BYTES
        + q7.encoded_record_bytes
        + norm_exact.encoded_record_bytes
        + ffn4.encoded_record_bytes
    )
    high_checkpoint = (
        CREVF_ARTIFACT_FIXED_BYTES
        + q7.encoded_record_bytes
        + norm_exact.encoded_record_bytes
        + ffn8.encoded_record_bytes
    )
    assert low_checkpoint < high_checkpoint
    dense_reference = math.ceil(low_checkpoint / 0.35)
    ceiling = dense_reference * 350_000 // 1_000_000
    assert low_checkpoint <= ceiling < high_checkpoint

    plan = plan_crevf_sites(
        list(reversed(sites)),
        dense_reference_bytes=dense_reference,
        target_reduction=0.65,
    )
    assert plan.target_checkpoint_bytes == ceiling
    assert plan.checkpoint_physical_bytes <= ceiling
    assert plan.headroom_bytes == ceiling - plan.checkpoint_physical_bytes
    assert plan.selected_option_ids == {
        "model.layers.0.feed_forward.w2.weight": "ffn4",
        "model.layers.0.operator_norm.weight": "norm-exact",
        "model.layers.0.self_attn.q_proj.weight": "q7",
    }
    by_site = {choice.site_id: choice for choice in plan.choices}
    assert by_site["model.layers.0.self_attn.q_proj.weight"].minimum_bit_width == 7
    assert by_site["model.layers.0.operator_norm.weight"].exact_bf16 is True
    artifact = plan.build_artifact()
    assert len(artifact) == plan.artifact_physical_bytes
    accounting = account_crevf_artifact(
        artifact,
        dense_reference_bytes=dense_reference,
        target_reduction=0.65,
    )
    assert accounting.checkpoint_physical_bytes == plan.checkpoint_physical_bytes
    assert accounting.runtime_packed_bytes == plan.runtime_packed_bytes
    assert accounting.persistent_encoded_bytes == plan.persistent_encoded_bytes
    assert accounting.decoded_full_state_bytes == plan.decoded_full_state_bytes
    assert (
        accounting.maximum_layer_decode_transient_bytes
        == plan.maximum_layer_decode_transient_bytes
    )

    reordered = tuple(
        CrevfSite(site.site_id, site.role, tuple(reversed(site.options)))
        for site in sites
    )
    second = plan_crevf_sites(
        list(reversed(reordered)),
        dense_reference_bytes=dense_reference,
        target_reduction=0.65,
    )
    assert second.selected_option_ids == plan.selected_option_ids
    assert second.build_artifact() == artifact


def test_option_bytes_are_derived_from_authenticated_blob() -> None:
    values = torch.ones(256, dtype=torch.bfloat16)
    option = _option(
        option_id="w4",
        name="w",
        role="ffn_w2",
        values=values,
        width=4,
        loss=1,
    )
    assert option.encoded_record_bytes == (
        CREVF_RECORD_ENVELOPE_BYTES + len(option.tensor_blob)
    )
    assert option.runtime_packed_bytes == option.metadata.runtime_packed_bytes
    assert option.persistent_encoded_bytes == option.metadata.persistent_encoded_bytes
    assert option.decoded_full_state_bytes == option.metadata.decoded_full_state_bytes


@pytest.mark.parametrize(
    ("role", "exact", "minimum"),
    [
        ("q_proj", False, 7),
        ("attention.key.weight", False, 7),
        ("attention_qkvo_projection", False, 7),
        ("operator_norm", True, 16),
        ("chat_control_anchor", True, 16),
        ("ffn_w2", False, 4),
    ],
)
def test_role_precision_floors_are_non_bypassable(
    role: str,
    exact: bool,
    minimum: int,
) -> None:
    assert role_requires_exact_bf16(role) is exact
    assert role_minimum_integer_width(role) == minimum


def test_tensor_identity_rejects_mislabeled_or_fused_attention_bypass() -> None:
    values = torch.ones(256, dtype=torch.bfloat16)
    disguised = _option(
        option_id="bad4",
        name="model.layers.0.self_attn.q_proj.weight",
        role="ffn_w2",
        values=values,
        width=4,
        loss=0,
    )
    with pytest.raises(CrevfError, match="understates"):
        CrevfSite(
            "model.layers.0.self_attn.q_proj.weight",
            "ffn_w2",
            (disguised,),
        )
    for name in (
        "model.layers.0.self_attn.wqkv.weight",
        "model.layers.0.attention.c_attn.weight",
        "model.layers.0.attention.in_proj_weight",
    ):
        assert tensor_identity_minimum_width(name) == 7
    assert tensor_identity_minimum_width("transformer.h.0.ln_1.weight") == 16
    assert tensor_identity_minimum_width("transformer.ln_f.weight") == 16
    assert tensor_identity_minimum_width("vision.surface_normal_head.weight") == 4
    assert tensor_identity_minimum_width("vision.normal_projection.weight") == 4
    assert semantic_minimum_integer_width("attention_query") == 7
    assert semantic_minimum_integer_width("normalization") == 16

    catalog_norm = _option(
        option_id="catalog-norm-4",
        name="adapter.opaque_17.weight",
        role="ffn_w2",
        values=values,
        width=4,
        loss=0,
    )
    typed_site = CrevfSite(
        catalog_norm.metadata.name,
        catalog_norm.metadata.role,
        (catalog_norm,),
        semantic_kind="normalization",
    )
    with pytest.raises(CrevfBudgetError, match="exact BF16"):
        _ = typed_site.eligible_options


def test_planner_fails_closed_when_role_floor_or_bytes_are_impossible() -> None:
    values = torch.ones(256, dtype=torch.bfloat16)
    q6 = _option(
        option_id="q6",
        name="q",
        role="q_proj",
        values=values,
        width=6,
        loss=0,
    )
    q_site = CrevfSite("q", "q_proj", (q6,))
    with pytest.raises(CrevfBudgetError, match="7-bit"):
        plan_crevf_sites(
            [q_site],
            dense_reference_bytes=100_000,
            target_reduction=0.55,
        )

    exact = _option(
        option_id="exact",
        name="norm",
        role="rms_norm",
        values=values,
        width=16,
        loss=0,
    )
    norm_site = CrevfSite("norm", "rms_norm", (exact,))
    with pytest.raises(CrevfBudgetError, match="exceeding"):
        plan_crevf_sites(
            [norm_site],
            dense_reference_bytes=100,
            target_reduction=0.65,
        )


def test_exact_frontier_never_silently_truncates_to_an_approximation() -> None:
    generator = torch.Generator().manual_seed(98)
    values = torch.randn(2048, generator=generator).to(torch.bfloat16)
    low = _option(
        option_id="low",
        name="w",
        role="ffn_w2",
        values=values,
        width=4,
        loss=10,
    )
    high = _option(
        option_id="high",
        name="w",
        role="ffn_w2",
        values=values,
        width=8,
        loss=0,
    )
    assert low.encoded_record_bytes < high.encoded_record_bytes
    with pytest.raises(CrevfBudgetError, match="max_frontier_states"):
        plan_crevf_sites(
            [CrevfSite("w", "ffn_w2", (low, high))],
            dense_reference_bytes=100_000,
            target_reduction=0.55,
            max_frontier_states=1,
        )
    with pytest.raises(CrevfBudgetError, match="provisional"):
        plan_crevf_sites(
            [CrevfSite("w", "ffn_w2", (low, high))],
            dense_reference_bytes=100_000,
            target_reduction=0.55,
            max_frontier_states=10,
            max_provisional_states=1,
        )
    with pytest.raises(CrevfBudgetError, match="backpointer"):
        plan_crevf_sites(
            [CrevfSite("w", "ffn_w2", (low, high))],
            dense_reference_bytes=100_000,
            target_reduction=0.55,
            max_backpointer_nodes=1,
        )


def test_planner_independently_bounds_physical_baseline_and_decoded_runtime() -> None:
    generator = torch.Generator().manual_seed(109)
    values = torch.randn(2048, generator=generator).to(torch.bfloat16)
    low = _option(
        option_id="w4",
        name="w",
        role="ffn_w2",
        values=values,
        width=4,
        loss=10,
    )
    high = _option(
        option_id="w8",
        name="w",
        role="ffn_w2",
        values=values,
        width=8,
        loss=0,
    )
    site = CrevfSite("w", "ffn_w2", (low, high))
    unbounded = plan_crevf_sites(
        [site],
        dense_reference_bytes=100_000,
        target_reduction=0.55,
    )
    assert unbounded.selected_option_ids == {"w": "w8"}
    bounded = plan_crevf_sites(
        [site],
        dense_reference_bytes=100_000,
        target_reduction=0.55,
        must_beat_physical_bytes=unbounded.checkpoint_physical_bytes,
        max_persistent_encoded_bytes=low.persistent_encoded_bytes,
        max_decoded_full_state_bytes=low.decoded_full_state_bytes,
        max_layer_transient_bytes=low.decoded_code_transient_bytes,
    )
    assert bounded.selected_option_ids == {"w": "w4"}
    assert bounded.checkpoint_physical_bytes < unbounded.checkpoint_physical_bytes
    assert bounded.total_runtime_bytes <= low.runtime_packed_bytes
    assert bounded.effective_checkpoint_ceiling_bytes == (
        unbounded.checkpoint_physical_bytes - 1
    )
    assert bounded.runtime_headroom_bytes == 0
    artifact = bounded.build_artifact()
    accounting = account_crevf_artifact(
        artifact,
        dense_reference_bytes=100_000,
        target_reduction=0.55,
        must_beat_physical_bytes=unbounded.checkpoint_physical_bytes,
        max_persistent_encoded_bytes=low.persistent_encoded_bytes,
        max_decoded_full_state_bytes=low.decoded_full_state_bytes,
        max_layer_transient_bytes=low.decoded_code_transient_bytes,
    )
    assert accounting.checkpoint_physical_bytes == bounded.checkpoint_physical_bytes
    assert accounting.total_runtime_bytes == bounded.total_runtime_bytes
    assert (
        accounting.total_decoded_full_state_bytes
        == bounded.total_decoded_full_state_bytes
    )


def test_exact50_is_starting_line_and_exposes_four_independent_budgets() -> None:
    values = torch.zeros(2048, dtype=torch.bfloat16)
    option = _option(
        option_id="all8-repaired",
        name="model.layers.3.feed_forward.w2.weight",
        role="ffn_w2",
        values=values,
        width=8,
        loss=0,
    )
    plan = plan_crevf_sites(
        [CrevfSite(option.metadata.name, option.metadata.role, (option,))],
        dense_reference_bytes=464_128_028,
        target_reduction=0.50,
        max_persistent_encoded_bytes=232_064_014,
        max_decoded_full_state_bytes=232_064_014,
        max_layer_transient_bytes=option.decoded_code_transient_bytes,
    )
    assert plan.reduction_ppm == 500_000
    assert plan.target_checkpoint_bytes == 232_064_014
    assert plan.checkpoint_physical_bytes <= plan.target_checkpoint_bytes
    assert plan.total_persistent_encoded_bytes <= 232_064_014
    assert plan.total_decoded_full_state_bytes <= 232_064_014
    assert (
        plan.maximum_layer_decode_transient_bytes <= option.decoded_code_transient_bytes
    )


def test_multi_budget_frontier_finds_feasible_layer_local_repair_mix() -> None:
    generator = torch.Generator().manual_seed(4402)
    values_a = torch.randn(1024, generator=generator).to(torch.bfloat16)
    values_b = torch.randn(2048, generator=generator).to(torch.bfloat16)
    a4 = _option(
        option_id="a4",
        name="model.layers.2.feed_forward.w1.weight",
        role="ffn_w1",
        values=values_a,
        width=4,
        loss=5,
    )
    a8 = _option(
        option_id="a8",
        name="model.layers.2.feed_forward.w1.weight",
        role="ffn_w1",
        values=values_a,
        width=8,
        loss=0,
    )
    b4 = _option(
        option_id="b4",
        name="model.layers.2.feed_forward.w2.weight",
        role="ffn_w2",
        values=values_b,
        width=4,
        loss=6,
    )
    b8 = _option(
        option_id="b8",
        name="model.layers.2.feed_forward.w2.weight",
        role="ffn_w2",
        values=values_b,
        width=8,
        loss=0,
    )
    layer_cap = a4.decoded_code_transient_bytes + b8.decoded_code_transient_bytes
    decoded_cap = a4.decoded_full_state_bytes + b8.decoded_full_state_bytes
    plan = plan_crevf_sites(
        [
            CrevfSite(a4.metadata.name, a4.metadata.role, (a4, a8)),
            CrevfSite(b4.metadata.name, b4.metadata.role, (b4, b8)),
        ],
        dense_reference_bytes=100_000,
        target_reduction=0.50,
        max_persistent_encoded_bytes=100_000,
        max_decoded_full_state_bytes=decoded_cap,
        max_layer_transient_bytes=layer_cap,
    )
    assert plan.selected_option_ids == {
        a4.metadata.name: "a4",
        b4.metadata.name: "b8",
    }
    assert plan.maximum_layer_decode_transient_bytes == layer_cap
    assert plan.total_decoded_full_state_bytes == decoded_cap


def test_equal_byte_equal_loss_tie_uses_lexicographic_option_id() -> None:
    values = torch.ones(256, dtype=torch.bfloat16)
    blob = encode_crevf_tensor(
        build_crevf_quantized_spec(
            name="w",
            role="ffn_w2",
            values=values,
            group_bit_widths=4,
            group_size=256,
        )
    )
    z = CrevfSiteOption("z-option", blob, 3)
    a = CrevfSiteOption("a-option", blob, 3)
    plan = plan_crevf_sites(
        [CrevfSite("w", "ffn_w2", (z, a))],
        dense_reference_bytes=10_000,
        target_reduction=0.65,
    )
    assert plan.selected_option_ids == {"w": "a-option"}


def _bundle_option(
    *,
    option_id: str,
    site_id: str,
    main_values: torch.Tensor,
    main_width: int,
    loss: int,
    with_writer: bool,
) -> CrevfSiteOption:
    main = _option(
        option_id=f"{option_id}-main",
        name=site_id,
        role="ffn_w2",
        values=main_values,
        width=main_width,
        loss=loss,
    )
    if not with_writer:
        return CrevfSiteOption(option_id, main.tensor_blob, loss)
    writer_values = main_values[:256].clone()
    writer = _option(
        option_id=f"{option_id}-writer",
        name=f"{site_id}.writer.weight",
        role="residual_writer",
        values=writer_values,
        width=8,
        loss=0,
    )
    indices = encode_crevf_auxiliary(
        CrevfAuxiliarySpec(
            name=f"{site_id}.writer.indices",
            kind="writer_column_indices_u16",
            payload=b"\x03\x00\x09\x00\x11\x00\x1f\x00",
        )
    )
    return CrevfSiteOption(
        option_id,
        main.tensor_blob,
        loss,
        correction_tensor_blobs=(writer.tensor_blob,),
        auxiliary_blobs=(indices,),
    )


@pytest.mark.parametrize(
    ("reduction", "ppm"),
    (
        (0.50, 500_000),
        (0.55, 550_000),
        (0.60, 600_000),
        (0.65, 650_000),
    ),
)
def test_atomic_bundle_is_exactly_accounted_at_every_large_target(
    reduction: float,
    ppm: int,
) -> None:
    values = torch.zeros(2048, dtype=torch.bfloat16)
    option = _bundle_option(
        option_id="main-plus-writer",
        site_id="model.layers.3.feed_forward.w2.weight",
        main_values=values,
        main_width=8,
        loss=0,
        with_writer=True,
    )
    minimum_checkpoint = CREVF_ARTIFACT_FIXED_BYTES + option.encoded_record_bytes
    retained_ppm = 1_000_000 - ppm
    dense_reference = math.ceil(
        minimum_checkpoint * 1_000_000 / retained_ppm
    ) + 16
    plan = plan_crevf_sites(
        (CrevfSite(option.metadata.name, option.metadata.role, (option,)),),
        dense_reference_bytes=dense_reference,
        target_reduction=reduction,
        max_persistent_encoded_bytes=option.persistent_encoded_bytes,
        max_decoded_full_state_bytes=option.decoded_full_state_bytes,
        max_layer_transient_bytes=option.decoded_code_transient_bytes,
    )
    assert plan.reduction_ppm == ppm
    assert plan.artifact_physical_bytes == minimum_checkpoint
    assert plan.persistent_encoded_bytes == option.persistent_encoded_bytes
    assert plan.decoded_full_state_bytes == option.decoded_full_state_bytes
    assert (
        plan.maximum_tensor_decode_transient_bytes
        == option.maximum_tensor_decode_transient_bytes
    )
    assert (
        plan.maximum_layer_decode_transient_bytes
        == option.decoded_code_transient_bytes
    )
    assert (
        plan.maximum_tensor_decode_transient_bytes
        < plan.maximum_layer_decode_transient_bytes
    )
    artifact = plan.build_artifact()
    accounting = account_crevf_artifact(
        artifact,
        dense_reference_bytes=dense_reference,
        target_reduction=reduction,
        max_persistent_encoded_bytes=option.persistent_encoded_bytes,
        max_decoded_full_state_bytes=option.decoded_full_state_bytes,
        max_layer_transient_bytes=option.decoded_code_transient_bytes,
    )
    assert accounting.artifact_physical_bytes == plan.artifact_physical_bytes
    assert accounting.persistent_encoded_bytes == plan.persistent_encoded_bytes
    assert accounting.decoded_full_state_bytes == plan.decoded_full_state_bytes


def test_atomic_bundle_planner_matches_tiny_bruteforce_and_actual_artifact() -> None:
    generator = torch.Generator().manual_seed(771)
    values_a = torch.randn(2048, generator=generator).to(torch.bfloat16)
    values_b = torch.randn(3072, generator=generator).to(torch.bfloat16)
    a_plain = _bundle_option(
        option_id="a-plain",
        site_id="model.layers.0.feed_forward.w2.weight",
        main_values=values_a,
        main_width=4,
        loss=8,
        with_writer=False,
    )
    a_bundle = _bundle_option(
        option_id="a-bundle",
        site_id="model.layers.0.feed_forward.w2.weight",
        main_values=values_a,
        main_width=8,
        loss=0,
        with_writer=True,
    )
    b_plain = _bundle_option(
        option_id="b-plain",
        site_id="model.layers.1.feed_forward.w2.weight",
        main_values=values_b,
        main_width=4,
        loss=6,
        with_writer=False,
    )
    b_bundle = _bundle_option(
        option_id="b-bundle",
        site_id="model.layers.1.feed_forward.w2.weight",
        main_values=values_b,
        main_width=8,
        loss=0,
        with_writer=True,
    )
    sites = (
        CrevfSite(a_plain.metadata.name, "ffn_w2", (a_plain, a_bundle)),
        CrevfSite(b_plain.metadata.name, "ffn_w2", (b_plain, b_bundle)),
    )
    one_bundle_cap = CREVF_ARTIFACT_FIXED_BYTES + max(
        a_bundle.encoded_record_bytes + b_plain.encoded_record_bytes,
        a_plain.encoded_record_bytes + b_bundle.encoded_record_bytes,
    )
    combinations = tuple(itertools.product(*[site.eligible_options for site in sites]))
    persistent_cap = max(
        sum(option.persistent_encoded_bytes for option in values)
        for values in combinations
    )
    decoded_cap = max(
        sum(option.decoded_full_state_bytes for option in values)
        for values in combinations
    )
    layer_cap = max(
        option.decoded_code_transient_bytes
        for values in combinations
        for option in values
    )
    feasible = []
    for options in combinations:
        physical = CREVF_ARTIFACT_FIXED_BYTES + sum(
            option.encoded_record_bytes for option in options
        )
        persistent = sum(option.persistent_encoded_bytes for option in options)
        decoded = sum(option.decoded_full_state_bytes for option in options)
        if (
            physical <= one_bundle_cap
            and persistent <= persistent_cap
            and decoded <= decoded_cap
            and max(
                option.decoded_code_transient_bytes for option in options
            )
            <= layer_cap
        ):
            feasible.append(
                (
                    (
                        sum(option.loss_units for option in options),
                        physical,
                        persistent,
                        decoded,
                        tuple(option.option_id for option in options),
                    ),
                    options,
                )
            )
    expected_key, expected_options = min(feasible, key=lambda value: value[0])
    plan = plan_crevf_sites(
        list(reversed(sites)),
        dense_reference_bytes=one_bundle_cap * 4,
        target_reduction=0.50,
        must_beat_physical_bytes=one_bundle_cap + 1,
        max_persistent_encoded_bytes=persistent_cap,
        max_decoded_full_state_bytes=decoded_cap,
        max_layer_transient_bytes=layer_cap,
    )
    assert plan.total_loss_units == expected_key[0]
    assert plan.checkpoint_physical_bytes == expected_key[1]
    assert plan.persistent_encoded_bytes == expected_key[2]
    assert plan.decoded_full_state_bytes == expected_key[3]
    assert plan.selected_option_ids == {
        site.site_id: option.option_id
        for site, option in zip(sites, expected_options, strict=True)
    }
    artifact = plan.build_artifact()
    runtime = decode_crevf_artifact(artifact)
    assert tuple(runtime.parameters()) == ()
    assert runtime.state_dict() == {}
    assert len(runtime.tensor_names) == sum(
        len(option.tensor_metadata) for option in expected_options
    )
    assert len(runtime.auxiliary_names) == sum(
        len(option.auxiliary_metadata) for option in expected_options
    )
    accounting = account_crevf_artifact(
        artifact,
        dense_reference_bytes=one_bundle_cap * 4,
        target_reduction=0.50,
        must_beat_physical_bytes=one_bundle_cap + 1,
        max_persistent_encoded_bytes=persistent_cap,
        max_decoded_full_state_bytes=decoded_cap,
        max_layer_transient_bytes=layer_cap,
    )
    assert accounting.checkpoint_physical_bytes == plan.checkpoint_physical_bytes
    assert accounting.persistent_encoded_bytes == plan.persistent_encoded_bytes
    assert accounting.decoded_full_state_bytes == plan.decoded_full_state_bytes
    by_site = {choice.site_id: choice for choice in plan.choices}
    for option in expected_options:
        choice = by_site[option.metadata.name]
        assert choice.record_names == option.record_names
        assert choice.record_sha256s == option.record_sha256s
        assert choice.bundle_sha256 == option.bundle_sha256
        assert len(choice.bundle_sha256) == 64


def test_bundle_rejects_corruption_duplicate_ownership_and_cross_layer_transients() -> None:
    values = torch.ones(256, dtype=torch.bfloat16)
    main_a = _option(
        option_id="main-a",
        name="model.layers.0.feed_forward.w2.weight",
        role="ffn_w2",
        values=values,
        width=8,
        loss=0,
    )
    main_b = _option(
        option_id="main-b",
        name="model.layers.1.feed_forward.w2.weight",
        role="ffn_w2",
        values=values,
        width=8,
        loss=0,
    )
    with pytest.raises(CrevfError, match="repeat a record name"):
        CrevfSiteOption(
            "duplicate-inside",
            main_a.tensor_blob,
            0,
            correction_tensor_blobs=(main_a.tensor_blob,),
        )
    auxiliary = encode_crevf_auxiliary(
        CrevfAuxiliarySpec(
            name="shared.writer.indices",
            kind="writer_column_indices_u16",
            payload=b"\x01\x00",
        )
    )
    corrupt_auxiliary = bytearray(auxiliary)
    corrupt_auxiliary[-33] ^= 1
    with pytest.raises(CrevfError, match="auxiliary SHA-256"):
        CrevfSiteOption(
            "corrupt",
            main_a.tensor_blob,
            0,
            auxiliary_blobs=(bytes(corrupt_auxiliary),),
        )
    option_a = CrevfSiteOption(
        "a",
        main_a.tensor_blob,
        0,
        auxiliary_blobs=(auxiliary,),
    )
    option_b = CrevfSiteOption(
        "b",
        main_b.tensor_blob,
        0,
        auxiliary_blobs=(auxiliary,),
    )
    with pytest.raises(CrevfError, match="owned by both"):
        plan_crevf_sites(
            (
                CrevfSite(main_a.metadata.name, "ffn_w2", (option_a,)),
                CrevfSite(main_b.metadata.name, "ffn_w2", (option_b,)),
            ),
            dense_reference_bytes=100_000,
            target_reduction=0.50,
        )
    cross_layer_writer = _option(
        option_id="cross",
        name="model.layers.1.feed_forward.w2.weight.writer.weight",
        role="residual_writer",
        values=values,
        width=8,
        loss=0,
    )
    cross_layer = CrevfSiteOption(
        "cross-layer",
        main_a.tensor_blob,
        0,
        correction_tensor_blobs=(cross_layer_writer.tensor_blob,),
    )
    with pytest.raises(CrevfError, match="layer-transient group"):
        CrevfSite(main_a.metadata.name, "ffn_w2", (cross_layer,))
