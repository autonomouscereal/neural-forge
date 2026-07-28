from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_output_action_writer_fold as writer_fold  # noqa: E402
from neural_forge.hf_lfm2_output_action_subspace import (  # noqa: E402
    compile_lfm2_output_action_subspace,
    seal_lfm2_output_action_evidence,
    seal_lfm2_output_action_rows,
)
from neural_forge.hf_lfm2_output_action_writer_fold import (  # noqa: E402
    Lfm2OutputActionWriterFoldError,
    Lfm2OutputActionWriterFoldResult,
    compile_lfm2_output_action_writer_fold,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


_PARENT = "a" * 64
_CACHE = "b" * 64
_COMPONENT = "model.layers.0.conv.out_proj"


def _subspace():
    actions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fit = seal_lfm2_output_action_rows(
        role="fit",
        row_ids=("fit-0", "fit-1", "fit-2", "fit-3"),
        branch_action_rows=actions,
    )
    evidence = seal_lfm2_output_action_evidence(
        parent_state_sha256=_PARENT,
        cache_sha256=_CACHE,
        component_id=_COMPONENT,
        fit=fit,
    )
    return compile_lfm2_output_action_subspace(evidence, target_rank=2)


def _pins(subspace, parent_weight):
    return {
        "expected_parent_state_sha256": _PARENT,
        "expected_cache_sha256": _CACHE,
        "expected_subspace_result_sha256": subspace.result_sha256,
        "expected_basis_receipt_sha256": subspace.basis_receipt_sha256,
        "expected_projector_sha256": subspace.report["projector"]["sha256"],
        "expected_parent_out_proj_sha256": lfm2_authenticated_tensor_sha256(
            parent_weight.detach()
        ),
    }


def _compile(parent_weight=None):
    subspace = _subspace()
    if parent_weight is None:
        parent_weight = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=torch.float32,
        )
    return (
        subspace,
        parent_weight,
        compile_lfm2_output_action_writer_fold(
            subspace,
            parent_weight,
            **_pins(subspace, parent_weight),
        ),
    )


def test_absorbs_projector_into_writer_in_cpu_fp64_then_fp32() -> None:
    subspace, parent, result = _compile()
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(result.out_proj_weight, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        result.out_proj_weight,
        (subspace.projector.T @ parent.to(torch.float64)).to(torch.float32),
        atol=0,
        rtol=0,
    )
    assert result.out_proj_weight.dtype == torch.float32
    assert result.out_proj_weight.device.type == "cpu"
    assert result.out_proj_weight.grad_fn is None
    assert result.out_proj_weight.requires_grad is False
    assert result.verify() is True


def test_runtime_state_is_only_independent_out_proj_weight() -> None:
    _subspace_value, parent, result = _compile()
    state = result.state_dict()

    assert tuple(state) == ("out_proj.weight",)
    assert state["out_proj.weight"].shape == parent.shape
    assert (
        state["out_proj.weight"].untyped_storage().data_ptr()
        != result.out_proj_weight.untyped_storage().data_ptr()
    )
    state["out_proj.weight"].zero_()
    assert bool((result.out_proj_weight != 0.0).any().item())
    assert tuple(result.__dataclass_fields__) == ("out_proj_weight", "report")
    assert result.report["runtime_state_keys"] == ("out_proj.weight",)
    assert result.report["runtime_q_or_p_factors_retained"] is False
    assert result.report["parent_sidecar_required"] is False


def test_parameter_accounting_explicitly_makes_no_reduction_claim() -> None:
    _subspace_value, parent, result = _compile()
    accounting = result.report["parameter_accounting"]

    assert accounting["parent_parameter_count"] == parent.numel()
    assert accounting["folded_parameter_count"] == parent.numel()
    assert accounting["parameter_delta"] == 0
    assert accounting["parameter_ratio_parent_over_folded"] == 1.0
    assert accounting["parameter_reduction_claimed"] is False
    assert accounting["diagnostic_parent_shell_only"] is True


@pytest.mark.parametrize(
    "pin_name",
    (
        "expected_parent_state_sha256",
        "expected_cache_sha256",
        "expected_subspace_result_sha256",
        "expected_basis_receipt_sha256",
        "expected_projector_sha256",
    ),
)
def test_every_independent_subspace_pin_is_authenticated(pin_name) -> None:
    subspace = _subspace()
    parent = torch.randn(4, 3, dtype=torch.float32)
    pins = _pins(subspace, parent)
    pins[pin_name] = "f" * 64

    with pytest.raises(Lfm2OutputActionWriterFoldError, match="pins do not match"):
        compile_lfm2_output_action_writer_fold(subspace, parent, **pins)


def test_parent_writer_pin_is_authenticated_independently() -> None:
    subspace = _subspace()
    parent = torch.randn(4, 3, dtype=torch.float32)
    pins = _pins(subspace, parent)
    pins["expected_parent_out_proj_sha256"] = "f" * 64

    with pytest.raises(Lfm2OutputActionWriterFoldError, match="pin does not match"):
        compile_lfm2_output_action_writer_fold(subspace, parent, **pins)


def test_pin_syntax_and_writer_width_fail_closed() -> None:
    subspace = _subspace()
    parent = torch.randn(3, 2, dtype=torch.float32)
    pins = _pins(subspace, parent)
    pins["expected_cache_sha256"] = "not-a-digest"
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="SHA-256"):
        compile_lfm2_output_action_writer_fold(subspace, parent, **pins)

    pins = _pins(subspace, parent)
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="output width"):
        compile_lfm2_output_action_writer_fold(subspace, parent, **pins)


def test_parameter_input_is_detached_without_autograd() -> None:
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float32))
    _subspace_value, _parent, result = _compile(parameter)

    assert parameter.requires_grad is True
    assert result.out_proj_weight.requires_grad is False
    assert result.out_proj_weight.grad_fn is None
    assert result.report["training_or_optimization_used"] is False


@pytest.mark.parametrize("bad_dtype", (torch.float16, torch.bfloat16, torch.float64))
def test_non_fp32_writer_fails_closed(bad_dtype) -> None:
    subspace = _subspace()
    parent = torch.randn(4, 3, dtype=bad_dtype)
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="FP32"):
        compile_lfm2_output_action_writer_fold(
            subspace,
            parent,
            **{
                **_pins(subspace, parent.to(torch.float32)),
                "expected_parent_out_proj_sha256": "f" * 64,
            },
        )


def test_nonfinite_and_tensor_subclass_writers_fail_closed() -> None:
    subspace = _subspace()
    nonfinite = torch.randn(4, 3, dtype=torch.float32)
    nonfinite[0, 0] = torch.nan
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="NaN or infinity"):
        compile_lfm2_output_action_writer_fold(
            subspace,
            nonfinite,
            **_pins(subspace, nonfinite),
        )

    class _TensorSubclass(torch.Tensor):
        pass

    subclass = torch.Tensor._make_subclass(
        _TensorSubclass,
        torch.randn(4, 3, dtype=torch.float32),
        False,
    )
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="exact dense"):
        compile_lfm2_output_action_writer_fold(
            subspace,
            subclass,
            **_pins(subspace, subclass),
        )


def test_mutated_result_and_fabricated_receipt_fail_verification() -> None:
    _subspace_value, _parent, result = _compile()
    result.out_proj_weight[0, 0] += 1.0
    with pytest.raises(Lfm2OutputActionWriterFoldError):
        result.verify()

    _subspace_value, _parent, clean = _compile()
    report = clean.as_dict()
    report["parameter_accounting"]["parameter_reduction_claimed"] = True
    fabricated = Lfm2OutputActionWriterFoldResult(
        out_proj_weight=clean.out_proj_weight.clone(),
        report=report,
    )
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="accounting"):
        fabricated.verify()


def test_verify_accepts_and_rejects_independent_publication_pins() -> None:
    _subspace_value, _parent, result = _compile()
    report = result.report
    assert result.verify(
        expected_parent_state_sha256=report["parent_state_sha256"],
        expected_cache_sha256=report["cache_sha256"],
        expected_subspace_result_sha256=report["subspace_result_sha256"],
        expected_basis_receipt_sha256=report["basis_receipt_sha256"],
        expected_projector_sha256=report["projector_sha256"],
        expected_parent_out_proj_sha256=report["parent_out_proj_sha256"],
        expected_state_sha256=result.state_sha256,
        expected_lineage_sha256=result.lineage_sha256,
        expected_result_sha256=result.result_sha256,
    )
    with pytest.raises(Lfm2OutputActionWriterFoldError, match="does not match"):
        result.verify(expected_result_sha256="f" * 64)


def test_compilation_is_bit_deterministic_and_does_not_alias_parent() -> None:
    subspace = _subspace()
    parent = torch.randn(
        4,
        9,
        generator=torch.Generator().manual_seed(606),
        dtype=torch.float32,
    )
    first = compile_lfm2_output_action_writer_fold(
        subspace, parent, **_pins(subspace, parent)
    )
    second = compile_lfm2_output_action_writer_fold(
        subspace, parent, **_pins(subspace, parent)
    )

    assert torch.equal(first.out_proj_weight, second.out_proj_weight)
    assert first.result_sha256 == second.result_sha256
    assert first.lineage_sha256 == second.lineage_sha256
    assert (
        first.out_proj_weight.untyped_storage().data_ptr()
        != parent.untyped_storage().data_ptr()
    )


def test_operation_receipt_and_api_prove_fold_only_diagnostic_surface() -> None:
    _subspace_value, _parent, result = _compile()
    audit = result.report["operation_audit"]

    for key in (
        "model_objects_accepted",
        "parent_model_forward_calls",
        "candidate_model_forward_calls",
        "parent_trace_recaptures",
        "training_steps",
        "optimizer_constructions",
        "optimizer_steps",
        "autograd_grad_calls",
        "backward_calls",
        "parameters_pruned",
        "parameters_quantized",
        "parameters_sparsified",
        "runtime_q_factors",
        "runtime_p_factors",
        "runtime_bridge_modules",
        "parent_sidecars_required",
        "filesystem_reads",
        "filesystem_writes",
    ):
        assert audit[key] == 0
    assert audit["network_access"] is False
    assert audit["runtime_parameter_sharing_used"] is False

    signature = inspect.signature(compile_lfm2_output_action_writer_fold)
    assert "model" not in signature.parameters
    source = inspect.getsource(writer_fold)
    for forbidden in (
        "torch.optim",
        ".backward(",
        "autograd.grad",
        "open(",
        "Path(",
        "requests.",
        "urllib.",
        "socket.",
    ):
        assert forbidden not in source
