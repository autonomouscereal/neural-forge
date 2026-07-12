from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.", allow_module_level=True
    )

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_residual_attention_probe as base  # noqa: E402
import neural_forge.hf_residual_attention_qk_anchor as qk_anchor  # noqa: E402
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
from neural_forge.hf_residual_attention_qk_anchor import (  # noqa: E402
    RealQKAnchor,
    ResidualAttentionQKAnchorError,
    audit_qk_anchor_assignments,
    install_real_qk_anchor,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 24

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


def _runtime(*, groups: int = 1, attention_bias: bool = False):
    torch.manual_seed(9696 + groups)
    heads = groups * 2
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=3,
            num_attention_heads=heads,
            num_key_value_heads=groups,
            head_dim=4 if groups == 1 else 2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            attention_bias=attention_bias,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    permutations = None
    if groups == 2:
        permutations = torch.tensor([[1, 0], [0, 1], [1, 0]])
    return prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=3,
        neuron_prototypes_per_neuron=3,
        group_permutations=permutations,
    )


def _proposed(runtime, redirect: base.AttentionRedirect):
    _bridge, _proposal, _candidates, tentative = base._tentative_codebook(
        runtime, (redirect,)
    )
    return tentative[0]


def _hash(value: torch.Tensor) -> str:
    return base._tensor_sha256(value)


def test_real_qk_anchor_copies_pair_only_and_rolls_back_byte_exactly() -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    before = {name: _hash(value) for name, value in registry.state_dict().items()}
    anchor_q = registry.prototype_q_weight[1].detach().clone()
    anchor_k = registry.prototype_k_weight[1].detach().clone()
    v_before = registry.prototype_v_weight.detach().clone()
    o_before = registry.prototype_o_weight.detach().clone()

    installed = install_real_qk_anchor(
        registry,
        RealQKAnchor(1, 0),
        destination_source_flat_index=0,
    )

    assert torch.equal(registry.prototype_q_weight[0], anchor_q)
    assert torch.equal(registry.prototype_k_weight[0], anchor_k)
    assert torch.equal(registry.prototype_v_weight, v_before)
    assert torch.equal(registry.prototype_o_weight, o_before)
    assert installed.report.anchor_slot == 1
    assert installed.report.destination_slot == 0
    assert installed.report.anchor_pair_sha256 == (
        installed.report.installed_pair_sha256
    )
    assert installed.report.non_qk_registry_tensors_byte_exact is True
    assert installed.report.unaffected_qk_slots_byte_exact is True
    installed.assert_live()

    installed.rollback()

    assert installed.is_applied is False
    assert {
        name: _hash(value) for name, value in registry.state_dict().items()
    } == before


def test_assignment_audit_honors_shared_sites_permutations_scales_and_norms() -> None:
    runtime = _runtime(groups=2)
    # Source coordinates are provenance layer:physical-group.  Redirecting
    # L2/G1 to L0/G1 makes every assignment for source 1 part of the audit.
    codebook = _proposed(runtime, base.AttentionRedirect(2, 1, 0, 1))
    registry = codebook.registry
    installation = install_real_qk_anchor(
        registry,
        RealQKAnchor(1, 0),
        destination_source_flat_index=1,
    )
    try:
        audit = audit_qk_anchor_assignments(
            runtime,
            registry,
            anchor=RealQKAnchor(1, 0),
            destination_source_flat_index=1,
        )
    finally:
        installation.rollback()

    assert audit["q_and_k_copied_from_same_real_active_slot"] is True
    assert audit["q_or_k_averaged_before_softmax"] is False
    assert audit["logical_layer_q_norm_k_norm_preserved"] is True
    assert audit["all_materialized_qk_slices_verified_exact"] is True
    assert audit["all_group_permutations_bijective"] is True
    assert audit["assignment_count"] >= 2
    assert [2, 1] in audit["assignment_coordinates"]
    for site in audit["assignments"]:
        assert site["q_norm"]["weight_sha256"]
        assert site["k_norm"]["weight_sha256"]
        assert sorted(site["permutation_row"]) == [0, 1]


def test_removed_or_nonexistent_anchor_is_rejected_without_mutation() -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    before = {name: _hash(value) for name, value in registry.state_dict().items()}

    with pytest.raises(ResidualAttentionQKAnchorError, match="active real survivor"):
        install_real_qk_anchor(
            registry,
            RealQKAnchor(2, 0),
            destination_source_flat_index=0,
        )

    assert {
        name: _hash(value) for name, value in registry.state_dict().items()
    } == before


def test_assignment_audit_rejects_a_named_anchor_that_was_not_installed() -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry

    with pytest.raises(
        ResidualAttentionQKAnchorError,
        match="not byte-identical to the named real anchor",
    ):
        audit_qk_anchor_assignments(
            runtime,
            registry,
            anchor=RealQKAnchor(1, 0),
            destination_source_flat_index=0,
        )


def test_assignment_audit_rejects_qk_tampering_after_install() -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    installation = install_real_qk_anchor(
        registry,
        RealQKAnchor(1, 0),
        destination_source_flat_index=0,
    )
    try:
        with torch.inference_mode():
            registry.prototype_k_weight[0, 0, 0].add_(1.0)
        with pytest.raises(
            ResidualAttentionQKAnchorError,
            match="not byte-identical to the named real anchor",
        ):
            audit_qk_anchor_assignments(
                runtime,
                registry,
                anchor=RealQKAnchor(1, 0),
                destination_source_flat_index=0,
            )
    finally:
        # Rollback deliberately accepts a dirty destination and restores its
        # complete Q/K affine backup byte-exactly.
        installation.rollback()


def test_ragged_anchor_copies_qk_biases_but_not_v_bias() -> None:
    runtime = _runtime(attention_bias=True)
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    assert registry.prototype_q_bias is not None
    assert registry.prototype_k_bias is not None
    assert registry.prototype_v_bias is not None
    anchor_q_bias = registry.prototype_q_bias[1].detach().clone()
    anchor_k_bias = registry.prototype_k_bias[1].detach().clone()
    v_bias_before = registry.prototype_v_bias.detach().clone()

    installation = install_real_qk_anchor(
        registry,
        RealQKAnchor(1, 0),
        destination_source_flat_index=0,
    )
    try:
        assert torch.equal(registry.prototype_q_bias[0], anchor_q_bias)
        assert torch.equal(registry.prototype_k_bias[0], anchor_k_bias)
        assert torch.equal(registry.prototype_v_bias, v_bias_before)
        audit = audit_qk_anchor_assignments(
            runtime,
            registry,
            anchor=RealQKAnchor(1, 0),
            destination_source_flat_index=0,
        )
        assert audit["destination_pair_byte_identical_to_anchor"] is True
    finally:
        installation.rollback()


def test_rectangular_cross_group_anchor_honors_permutation_and_scales() -> None:
    runtime = _runtime(groups=2, attention_bias=True)
    registry = runtime.attention_codebook.registry
    assert registry.prototype_q_weight.ndim == 4
    # Make role-scale semantics non-trivial.  The registry uses canonical
    # assignment axes before argsort(group_permutations) materializes physical
    # runtime groups.
    scales = torch.linspace(
        0.5,
        1.5,
        steps=registry.logical_layers * registry.key_value_heads * 4,
        dtype=registry.prototype_q_weight.dtype,
    ).reshape(registry.logical_layers, registry.key_value_heads, 4)
    registry.role_scales = torch.nn.Parameter(scales)
    protected = {
        name: _hash(value)
        for name, value in registry.state_dict().items()
        if name
        not in {
            "prototype_q_weight",
            "prototype_k_weight",
            "prototype_q_bias",
            "prototype_k_bias",
        }
    }
    # In this permutation fixture source L2/G1 occupies rectangular global
    # slot 2, while source L0/G0 occupies global slot 3.
    installation = install_real_qk_anchor(
        registry,
        RealQKAnchor(2, 1),
        destination_source_flat_index=0,
    )
    try:
        assert installation.report.anchor_slot == 2
        assert installation.report.destination_slot == 3
        audit = audit_qk_anchor_assignments(
            runtime,
            registry,
            anchor=RealQKAnchor(2, 1),
            destination_source_flat_index=0,
        )
        assert audit["assignment_coordinates"] == [[0, 0]]
        assert audit["assignments"][0]["canonical_assignment_axis"] == 1
        assert audit["assignments"][0]["q_role_scale"] == pytest.approx(
            float(scales[0, 1, 0])
        )
        assert audit["assignments"][0]["k_role_scale"] == pytest.approx(
            float(scales[0, 1, 1])
        )
        assert {
            name: _hash(value)
            for name, value in registry.state_dict().items()
            if name in protected
        } == protected
    finally:
        installation.rollback()


def test_rollback_fails_on_unrelated_tamper_and_is_retryable() -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    before = {name: _hash(value) for name, value in registry.state_dict().items()}
    v_backup = registry.prototype_v_weight.detach().clone()
    installation = install_real_qk_anchor(
        registry,
        RealQKAnchor(1, 0),
        destination_source_flat_index=0,
    )
    with torch.inference_mode():
        registry.prototype_v_weight[0, 0, 0].add_(1.0)
    with pytest.raises(
        ResidualAttentionQKAnchorError,
        match="rollback did not restore the registry byte-exactly",
    ):
        installation.rollback()
    assert installation.is_applied is True

    with torch.inference_mode():
        registry.prototype_v_weight.copy_(v_backup)
    installation.rollback()
    assert installation.is_applied is False
    assert {
        name: _hash(value) for name, value in registry.state_dict().items()
    } == before


def test_install_validation_failure_rolls_back_before_raising(monkeypatch) -> None:
    runtime = _runtime()
    codebook = _proposed(runtime, base.AttentionRedirect(2, 0, 0, 0))
    registry = codebook.registry
    before = {name: _hash(value) for name, value in registry.state_dict().items()}
    real_slots_sha256 = qk_anchor._slots_sha256
    calls = 0

    def fail_after_mutation(value, excluded_slot):
        nonlocal calls
        calls += 1
        digest = real_slots_sha256(value, excluded_slot)
        # Four Q/K entries are hashed before mutation; falsify the first
        # post-mutation unaffected-slot check to exercise immediate rollback.
        return "0" * 64 if calls == 5 else digest

    monkeypatch.setattr(qk_anchor, "_slots_sha256", fail_after_mutation)
    with pytest.raises(
        ResidualAttentionQKAnchorError,
        match="changed an unaffected Q/K prototype slot",
    ):
        install_real_qk_anchor(
            registry,
            RealQKAnchor(1, 0),
            destination_source_flat_index=0,
        )
    assert {
        name: _hash(value) for name, value in registry.state_dict().items()
    } == before
