from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_model_fold as model_fold  # noqa: E402
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    Lfm2CpcfModelFoldError,
    Lfm2CpcfModelFoldPolicy,
    audit_lfm2_cpcf_parameter_compaction,
    independent_parameter_inventory,
)


class _TiedModel(torch.nn.Module):
    def __init__(self, rows: int, width: int) -> None:
        super().__init__()
        self.table = torch.nn.Embedding(rows, width)
        self.lm_head = torch.nn.Linear(width, rows, bias=False)
        self.lm_head.weight = self.table.weight
        self.config = SimpleNamespace(model_type="lfm2")


class _TransactionalModel(torch.nn.Module):
    def __init__(self, width: int, value: float) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.full((width,), value))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.model.embedding_norm = SimpleNamespace(variance_epsilon=1e-5)
        self.config = SimpleNamespace(
            model_type="lfm2",
            hidden_size=width,
        )


def test_independent_parameter_inventory_counts_native_tie_once() -> None:
    model = _TiedModel(17, 8)
    inventory = independent_parameter_inventory(model)
    assert inventory["unique_parameters"] == 17 * 8
    assert inventory["raw_named_parameter_numel"] == 2 * 17 * 8
    assert inventory["alias_group_count"] == 1
    assert inventory["independent_methods_agree"] is True


def test_parameter_receipt_allows_any_fold_but_marks_promotion() -> None:
    source = _TiedModel(128, 16)
    target = _TiedModel(8, 8)
    target_count = 8 * 8
    diagnostic = audit_lfm2_cpcf_parameter_compaction(
        source,
        target,
        planned_target_unique_parameters=target_count,
        promotion_compression_ratio=100.0,
    )
    assert diagnostic["realized_compression_ratio"] == 32.0
    assert diagnostic["promotion_ratio_met"] is False
    assert diagnostic["valid_lower_ratio_diagnostic_fold"] is True
    assert diagnostic["quantization_used"] is False


def test_parameter_receipt_rejects_noncompact_or_wrong_plan() -> None:
    source = _TiedModel(16, 8)
    target = _TiedModel(16, 8)
    with pytest.raises(Lfm2CpcfModelFoldError, match="does not compact"):
        audit_lfm2_cpcf_parameter_compaction(
            source,
            target,
            planned_target_unique_parameters=16 * 8,
        )
    smaller = _TiedModel(8, 8)
    with pytest.raises(Lfm2CpcfModelFoldError, match="catalog plan"):
        audit_lfm2_cpcf_parameter_compaction(
            source,
            smaller,
            planned_target_unique_parameters=63,
        )


def test_model_fold_policy_keeps_generic_budget_separate_from_promotion() -> None:
    generic = Lfm2CpcfModelFoldPolicy()
    production = Lfm2CpcfModelFoldPolicy(
        promotion_compression_ratio=100.0,
        require_promotion_ratio=True,
    )
    assert generic.promotion_compression_ratio == 1.0
    assert generic.require_promotion_ratio is False
    assert production.promotion_compression_ratio == 100.0
    assert production.require_promotion_ratio is True
    assert production.resolved_short_conv_policy().enforce is True


def test_whole_model_failure_restores_blank_gradient_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _TransactionalModel(2, 1.0)
    target = _TransactionalModel(1, 0.0)
    target.train()
    original_gradient = torch.tensor([7.0])
    target.scalar.grad = original_gradient

    class _Phase:
        def __init__(self) -> None:
            self.parent_evidence = SimpleNamespace(layers=(object(),))
            self.phase_sha256 = "1" * 64

    monkeypatch.setattr(model_fold, "Lfm2CpcfTargetPhase", _Phase)
    boundary = SimpleNamespace(
        interface=object(),
        layers=(object(),),
        bundle_sha256="2" * 64,
    )
    monkeypatch.setattr(
        model_fold,
        "fit_lfm2_cpcf_boundary_bundle",
        lambda *_args, **_kwargs: boundary,
    )

    def _mutating_interface(model, _interface):
        with torch.no_grad():
            model.scalar.fill_(3.0)
        model.eval()
        model.scalar.requires_grad_(False)
        model.scalar.grad = None
        return {"report_sha256": "3" * 64}

    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_tied_interface",
        _mutating_interface,
    )
    monkeypatch.setattr(
        model_fold,
        "install_lfm2_cpcf_neutral_norms",
        lambda *_args, **_kwargs: {"report_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        model_fold,
        "seed_lfm2_cpcf_phase_states",
        lambda *_args, **_kwargs: SimpleNamespace(layer_index=0),
    )

    def _fail_layer(*_args, **_kwargs):
        with torch.no_grad():
            target.scalar.fill_(9.0)
        raise RuntimeError("late layer failure")

    monkeypatch.setattr(
        model_fold,
        "compile_and_install_lfm2_cpcf_layer",
        _fail_layer,
    )
    blank_body = {
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "weights_installed": False,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "target_unique_parameters": 1,
    }
    blank_report = {
        **blank_body,
        "report_sha256": model_fold._sha(blank_body),
    }
    with pytest.raises(RuntimeError, match="late layer failure"):
        model_fold.compile_lfm2_cpcf_model_fold(
            source,
            target,
            phases=(_Phase(),),
            source_tied_table=torch.ones(3, 2),
            blank_target_report=blank_report,
            planned_target_unique_parameters=1,
        )
    assert torch.equal(target.scalar, torch.zeros_like(target.scalar))
    assert target.scalar.requires_grad is True
    assert target.scalar.grad is original_gradient
    assert torch.equal(target.scalar.grad, torch.tensor([7.0]))
    assert target.training is True
