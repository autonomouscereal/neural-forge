from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_short_conv_channel_qdeim import (
    LFM2_SHORT_CONV_CHANNEL_QDEIM_RANK_DEFICIT_SCHEMA,
    Lfm2CompiledShortConvChannelQdeim,
    Lfm2ShortConvChannelQdeimError,
    Lfm2ShortConvChannelQdeimInsufficientRankError,
    authenticate_lfm2_short_conv_channel_qdeim,
    compile_lfm2_short_conv_channel_qdeim,
    lfm2_short_conv_channel_qdeim_parameter_accounting,
    lfm2_short_conv_channel_qdeim_seam_state_sha256,
    lfm2_short_conv_channel_qdeim_source_state_sha256,
)


def _source(
    *,
    hidden: int = 7,
    channels: int = 7,
    kernel: int = 3,
    seed: int = 2707,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(3 * channels, hidden, generator=generator),
        torch.randn(channels, 1, kernel, generator=generator),
        torch.randn(hidden, channels, generator=generator),
    )


def _compile(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    target_channels: int,
    input_decoder: torch.Tensor | None = None,
    output_encoder: torch.Tensor | None = None,
    protected: tuple[int, ...] = (),
) -> Lfm2CompiledShortConvChannelQdeim:
    in_proj, conv, out_proj = source
    source_sha256 = lfm2_short_conv_channel_qdeim_source_state_sha256(
        in_proj,
        conv,
        out_proj,
    )
    seam_sha256 = None
    if input_decoder is not None:
        seam_sha256 = lfm2_short_conv_channel_qdeim_seam_state_sha256(
            source_hidden_size=int(in_proj.shape[1]),
            input_decoder=input_decoder,
            output_encoder=output_encoder,
        )
    return compile_lfm2_short_conv_channel_qdeim(
        source_in_proj_weight=in_proj,
        source_conv_weight=conv,
        source_out_proj_weight=out_proj,
        expected_source_state_sha256=source_sha256,
        target_internal_channels=target_channels,
        input_decoder=input_decoder,
        output_encoder=output_encoder,
        expected_seam_state_sha256=seam_sha256,
        protected_source_channels=protected,
    )


def _short_conv(
    hidden_states: torch.Tensor,
    *,
    in_proj: torch.Tensor,
    conv: torch.Tensor,
    out_proj: torch.Tensor,
) -> torch.Tensor:
    bcx = torch.nn.functional.linear(hidden_states, in_proj).transpose(-1, -2)
    b, c, x = bcx.chunk(3, dim=-2)
    convolved = torch.nn.functional.conv1d(
        b * x,
        conv,
        groups=int(conv.shape[0]),
        padding=int(conv.shape[-1]) - 1,
    )[..., : hidden_states.shape[1]]
    return torch.nn.functional.linear(
        (c * convolved).transpose(-1, -2).contiguous(),
        out_proj,
    )


def _alias_groups(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(name)
    return [names for names in groups.values() if len(names) > 1]


def test_full_width_is_literal_identity_and_behaviorally_exact() -> None:
    source = _source(hidden=6, channels=6, seed=2801)
    candidate = _compile(source, target_channels=6)
    in_proj, conv, out_proj = source
    assert candidate.selected_source_channels == tuple(range(6))
    assert candidate.report["selection_method"] == "full-width-canonical-identity"
    assert torch.equal(candidate.in_proj_weight, in_proj)
    assert torch.equal(candidate.conv_weight, conv)
    assert torch.equal(candidate.out_proj_weight, out_proj)
    torch.testing.assert_close(
        candidate.interpolation_reconstruction,
        torch.eye(6, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(2803)
    hidden_states = torch.randn(2, 8, 6, generator=generator, dtype=torch.float64)
    expected = _short_conv(
        hidden_states,
        in_proj=in_proj.double(),
        conv=conv.double(),
        out_proj=out_proj.double(),
    )
    actual = _short_conv(
        hidden_states,
        in_proj=candidate.in_proj_weight.double(),
        conv=candidate.conv_weight.double(),
        out_proj=candidate.out_proj_weight.double(),
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    authenticate_lfm2_short_conv_channel_qdeim(candidate)


def test_rank_deficient_full_width_is_still_literal_identity() -> None:
    hidden = channels = 5
    source = (
        torch.zeros(3 * channels, hidden, dtype=torch.float32),
        torch.ones(channels, 1, 3, dtype=torch.float32),
        torch.ones(hidden, channels, dtype=torch.float32),
    )
    candidate = _compile(source, target_channels=channels)
    assert candidate.report["observed_structural_rank"] == 0
    assert candidate.report["selected_context_interpolation_rank"] == 0
    assert candidate.report["structural_rank_required_for_materialization"] is False
    assert (
        candidate.report["selected_context_rank_required_for_materialization"] is False
    )
    assert candidate.selected_source_channels == tuple(range(channels))
    assert torch.equal(candidate.in_proj_weight, source[0])
    assert torch.equal(candidate.conv_weight, source[1])
    assert torch.equal(candidate.out_proj_weight, source[2])
    torch.testing.assert_close(
        candidate.interpolation_reconstruction,
        torch.eye(channels, dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    authenticate_lfm2_short_conv_channel_qdeim(candidate)


def test_compressed_candidate_selects_complete_b_c_x_fir_tuples() -> None:
    source = _source(hidden=8, channels=8, seed=2903)
    candidate = _compile(source, target_channels=4)
    in_proj, conv, out_proj = source
    source_b, source_c, source_x = in_proj.chunk(3, dim=0)
    candidate_b, candidate_c, candidate_x = candidate.in_proj_weight.chunk(3, dim=0)
    selected = torch.tensor(candidate.selected_source_channels, dtype=torch.long)
    torch.testing.assert_close(
        candidate_b,
        source_b.index_select(0, selected),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate_c,
        source_c.index_select(0, selected),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate_x,
        source_x.index_select(0, selected),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.conv_weight,
        conv.index_select(0, selected),
        rtol=0.0,
        atol=0.0,
    )
    expected_writer = (
        out_proj.double() @ candidate.interpolation_reconstruction.T
    ).float()
    torch.testing.assert_close(
        candidate.out_proj_weight,
        expected_writer,
        rtol=0.0,
        atol=0.0,
    )
    lineage = candidate.report["channel_tuple_lineage"]
    assert [row["source_channel_index"] for row in lineage] == list(
        candidate.selected_source_channels
    )
    assert all(
        row["producer_channel_dense_average_used"] is False
        and row["fir_channel_dense_average_used"] is False
        for row in lineage
    )
    assert candidate.report["channel_tuple_lineage_sha256"] == canonical_sha256(lineage)
    contract = candidate.report["intact_tuple_contract"]
    assert contract["B_C_X_rows_selected_from_one_identical_source_index"] is True
    assert contract["FIR_row_selected_from_that_identical_source_index"] is True
    assert contract["omitted_channel_action_reconstructed_only_in_dense_writer"]


def test_selection_is_deterministic_and_protected_pivots_are_retained() -> None:
    source = _source(hidden=9, channels=9, seed=3001)
    first = _compile(source, target_channels=5, protected=(7, 1))
    second = _compile(source, target_channels=5, protected=(1, 7))
    assert first.selected_source_channels == second.selected_source_channels
    assert first.selected_source_channels[:2] == (1, 7)
    assert first.report["protected_source_channels"] == [1, 7]
    assert first.report["result_sha256"] == second.report["result_sha256"]
    assert torch.equal(first.in_proj_weight, second.in_proj_weight)
    assert torch.equal(first.conv_weight, second.conv_weight)
    assert torch.equal(first.out_proj_weight, second.out_proj_weight)
    assert torch.equal(
        first.interpolation_reconstruction,
        second.interpolation_reconstruction,
    )


def test_structural_rank_failure_has_authenticated_fail_closed_receipt() -> None:
    hidden = channels = 5
    in_proj = torch.zeros(3 * channels, hidden, dtype=torch.float32)
    conv = torch.ones(channels, 1, 3, dtype=torch.float32)
    out_proj = torch.ones(hidden, channels, dtype=torch.float32)
    source_sha256 = lfm2_short_conv_channel_qdeim_source_state_sha256(
        in_proj,
        conv,
        out_proj,
    )
    with pytest.raises(Lfm2ShortConvChannelQdeimInsufficientRankError) as caught:
        compile_lfm2_short_conv_channel_qdeim(
            source_in_proj_weight=in_proj,
            source_conv_weight=conv,
            source_out_proj_weight=out_proj,
            expected_source_state_sha256=source_sha256,
            target_internal_channels=2,
        )
    report = dict(caught.value.report)
    assert report["schema"] == LFM2_SHORT_CONV_CHANNEL_QDEIM_RANK_DEFICIT_SCHEMA
    assert report["status"] == "insufficient-separable-gram-rank"
    assert report["observed_structural_rank"] == 0
    assert report["required_structural_rank"] == 2
    assert report["candidate_parameters_materialized"] is False
    assert report["interpolation_matrix_materialized"] is False
    assert report["random_mode_completion_permitted"] is False
    assert report["silent_pseudoinverse_permitted"] is False
    result_sha256 = report.pop("result_sha256")
    assert canonical_sha256(report) == result_sha256


def test_standalone_d_e_seams_are_absorbed_without_runtime_bridges() -> None:
    hidden = channels = 7
    child_hidden = 4
    source = _source(hidden=hidden, channels=channels, seed=3109)
    generator = torch.Generator(device="cpu").manual_seed(3119)
    decoder = torch.randn(child_hidden, hidden, generator=generator)
    encoder = torch.randn(hidden, child_hidden, generator=generator)
    candidate = _compile(
        source,
        target_channels=4,
        input_decoder=decoder,
        output_encoder=encoder,
    )
    in_proj, conv, out_proj = source
    source_b, source_c, source_x = in_proj.double().chunk(3, dim=0)
    selected = torch.tensor(candidate.selected_source_channels, dtype=torch.long)
    explicit_in = torch.cat(
        (
            source_b.index_select(0, selected) @ decoder.double().T,
            source_c.index_select(0, selected) @ decoder.double().T,
            source_x.index_select(0, selected) @ decoder.double().T,
        ),
        dim=0,
    )
    explicit_out = (
        encoder.double().T
        @ out_proj.double()
        @ candidate.interpolation_reconstruction.T
    )
    torch.testing.assert_close(
        candidate.in_proj_weight,
        explicit_in.float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        candidate.out_proj_weight,
        explicit_out.float(),
        rtol=0.0,
        atol=0.0,
    )
    generator.manual_seed(3137)
    child_states = torch.randn(
        2,
        6,
        child_hidden,
        generator=generator,
        dtype=torch.float64,
    )
    explicit = (
        _short_conv(
            child_states @ decoder.double(),
            in_proj=torch.cat(
                (
                    source_b.index_select(0, selected),
                    source_c.index_select(0, selected),
                    source_x.index_select(0, selected),
                ),
                dim=0,
            ),
            conv=conv.double().index_select(0, selected),
            out_proj=(out_proj.double() @ candidate.interpolation_reconstruction.T),
        )
        @ encoder.double()
    )
    folded = _short_conv(
        child_states,
        in_proj=explicit_in,
        conv=conv.double().index_select(0, selected),
        out_proj=explicit_out,
    )
    torch.testing.assert_close(folded, explicit, rtol=1e-11, atol=1e-11)
    assert candidate.report["seam_mode"] == "standalone-residual-seams-absorbed"
    assert candidate.report["parameter_storage"]["runtime_bridge_parameter_count"] == 0
    assert candidate.report["operation_audit"]["runtime_bridge_modules"] == 0
    assert candidate.report["operation_audit"]["parent_sidecars_required"] == 0


def test_requested_real_lfm2_dense_parameter_counts_are_exact() -> None:
    m776 = lfm2_short_conv_channel_qdeim_parameter_accounting(
        source_hidden_size=1024,
        source_internal_channels=1024,
        target_internal_channels=776,
        kernel_size=3,
    )
    assert m776["source_dense_parameter_count"] == 4_197_376
    assert m776["candidate_dense_parameter_count"] == 3_180_824
    assert m776["source_exactly_four_times_candidate"] is False
    m256 = lfm2_short_conv_channel_qdeim_parameter_accounting(
        source_hidden_size=1024,
        source_internal_channels=1024,
        target_internal_channels=256,
        kernel_size=3,
    )
    assert m256["candidate_dense_parameter_count"] == 1_049_344
    assert m256["source_exactly_four_times_candidate"] is True
    assert m256["candidate_times_four"] == m256["source_dense_parameter_count"]
    assert m256["parameter_sharing_used"] is False
    assert m256["sparse_storage_used"] is False
    assert m256["quantization_used"] is False


def test_materialized_candidate_has_three_unique_dense_parameter_storages() -> None:
    candidate = _compile(_source(seed=3203), target_channels=4)
    state = dict(candidate.state_dict())
    assert _alias_groups(state) == []
    assert candidate.report["parameter_storage"] == {
        "stored_dense_parameter_count": sum(
            tensor.numel() for tensor in state.values()
        ),
        "unique_dense_parameter_tensor_count": 3,
        "storage_alias_group_count": 0,
        "runtime_bridge_parameter_count": 0,
        "parent_sidecar_parameter_count": 0,
    }
    assert candidate.report["operation_audit"]["parameters_pruned"] == 0
    assert candidate.report["operation_audit"]["parameters_quantized"] == 0
    assert candidate.report["operation_audit"]["parameters_sparsified"] == 0
    assert (
        candidate.report["operation_audit"]["runtime_parameter_sharing_used"] is False
    )


def test_authentication_rejects_candidate_state_mutation() -> None:
    candidate = _compile(_source(seed=3301), target_channels=4)
    corrupted = replace(
        candidate,
        out_proj_weight=candidate.out_proj_weight.clone(),
    )
    corrupted.out_proj_weight[0, 0] += 1.0
    with pytest.raises(
        Lfm2ShortConvChannelQdeimError,
        match="state changed",
    ):
        authenticate_lfm2_short_conv_channel_qdeim(corrupted)
    corrupted_lineage = replace(
        candidate,
        interpolation_reconstruction=(candidate.interpolation_reconstruction.clone()),
    )
    corrupted_lineage.interpolation_reconstruction[0, 0] += 1.0
    with pytest.raises(
        Lfm2ShortConvChannelQdeimError,
        match="FP64.*lineage changed",
    ):
        authenticate_lfm2_short_conv_channel_qdeim(corrupted_lineage)


def test_source_and_seam_pins_fail_closed_before_materialization() -> None:
    source = _source(seed=3407)
    in_proj, conv, out_proj = source
    source_sha256 = lfm2_short_conv_channel_qdeim_source_state_sha256(
        in_proj,
        conv,
        out_proj,
    )
    changed = in_proj.clone()
    changed[0, 0] += 0.25
    with pytest.raises(Lfm2ShortConvChannelQdeimError, match="source.*changed"):
        compile_lfm2_short_conv_channel_qdeim(
            source_in_proj_weight=changed,
            source_conv_weight=conv,
            source_out_proj_weight=out_proj,
            expected_source_state_sha256=source_sha256,
            target_internal_channels=4,
        )
    generator = torch.Generator(device="cpu").manual_seed(3413)
    decoder = torch.randn(4, 7, generator=generator)
    encoder = torch.randn(7, 4, generator=generator)
    seam_sha256 = lfm2_short_conv_channel_qdeim_seam_state_sha256(
        source_hidden_size=7,
        input_decoder=decoder,
        output_encoder=encoder,
    )
    encoder = encoder.clone()
    encoder[0, 0] += 0.5
    with pytest.raises(Lfm2ShortConvChannelQdeimError, match="seam state changed"):
        compile_lfm2_short_conv_channel_qdeim(
            source_in_proj_weight=in_proj,
            source_conv_weight=conv,
            source_out_proj_weight=out_proj,
            expected_source_state_sha256=source_sha256,
            target_internal_channels=4,
            input_decoder=decoder,
            output_encoder=encoder,
            expected_seam_state_sha256=seam_sha256,
        )


def test_fp64_lineage_seals_every_requested_gram_and_single_fp32_materialization() -> (
    None
):
    candidate = _compile(_source(seed=3501), target_channels=4)
    lineage = candidate.report["fp64_lineage"]
    assert lineage["formulae"] == {
        "G_BX": ".5*((B@B.T)*(X@X.T)+(B@X.T)*(X@B.T))",
        "G_ctx": "(C@C.T)*(K@K.T)*G_BX",
        "G_full": "(O.T@O)*G_ctx",
        "interpolation": "solve(G_ctx[S,S],G_ctx[S,:])",
        "writer_prior": "O@R.T",
        "input_seam_absorption": "identity",
        "output_seam_absorption": "identity",
    }
    for key in (
        "G_BX",
        "G_ctx",
        "G_full",
        "structural_eigenvalues",
        "canonical_structural_basis",
        "qdeim_pivot_residual_norms",
        "sampled_context_gram",
        "sampled_to_all_context_gram",
        "interpolation_reconstruction",
        "candidate_in_proj_before_fp32",
        "candidate_out_proj_before_fp32",
    ):
        assert lineage[key]["dtype"] == "float64"
        assert len(lineage[key]["sha256"]) == 64
    assert lineage["solve_device"] == "cpu"
    assert lineage["solve_dtype"] == "float64"
    assert lineage["materialization_dtype"] == "float32"
    assert candidate.report["fp64_lineage_sha256"] == canonical_sha256(lineage)
    assert all(
        tensor.dtype == torch.float32 for tensor in candidate.state_dict().values()
    )
