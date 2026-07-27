from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_forge.hf_lfm2_progressive_component_ablation import (
    HYBRID_MANIFEST_FILENAME,
    LFM2_PROGRESSIVE_COMPONENT_INSTALL_SCHEMA,
    LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA,
    Lfm2ProgressiveComponentAblationError,
    Lfm2ProgressiveComponentCapsule,
    _model_state_sha256,
    _module_state_sha256,
    _parameter_alias_topology,
    _stored_tensor_elements,
    apply_lfm2_component_ablation_candidate,
    build_lfm2_progressive_component_ablation_schedule,
    install_lfm2_progressive_component,
    load_lfm2_progressive_hybrid_checkpoint,
    schedule_receipt,
    write_lfm2_progressive_hybrid_checkpoint,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _Attention(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.head_dim = 1
        self.layer_idx = 0
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        del position_embeddings, attention_mask, past_key_values, kwargs
        value = self.v_proj(hidden_states)
        return self.out_proj(value), None


class _Mlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden, intermediate, bias=False)
        self.w2 = torch.nn.Linear(intermediate, hidden, bias=False)
        self.w3 = torch.nn.Linear(hidden, intermediate, bias=False)

    def forward(self, value):
        return self.w2(torch.nn.functional.silu(self.w1(value)) * self.w3(value))


class _Layer(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.feed_forward = _Mlp(hidden, intermediate)


class _ToyModel(torch.nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        layer_count: int = 1,
    ) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_Layer(hidden, intermediate) for _ in range(layer_count)]
        )


def _capsule(
    *,
    component_id: str,
    kind: str,
    parent: torch.nn.Module,
    compact: torch.nn.Module,
    parent_path: str,
    compact_path: str,
    encoder: torch.Tensor,
    decoder: torch.Tensor | None,
) -> Lfm2ProgressiveComponentCapsule:
    return Lfm2ProgressiveComponentCapsule(
        component_id=component_id,
        kind=kind,
        parent_module_path=parent_path,
        compact_module_path=compact_path,
        input_encoder=encoder,
        output_decoder=decoder,
        expected_parent_module_sha256=_module_state_sha256(
            parent.get_submodule(parent_path)
        ),
        expected_compact_module_sha256=_module_state_sha256(
            compact.get_submodule(compact_path)
        ),
        expected_input_encoder_sha256=lfm2_authenticated_tensor_sha256(encoder),
        expected_output_decoder_sha256=(
            None
            if decoder is None
            else lfm2_authenticated_tensor_sha256(decoder)
        ),
        source_fold_receipt_sha256=_sha(f"fold-{component_id}"),
        source_compact_checkpoint_sha256=_sha("compact-checkpoint"),
    )


def test_parameter_alias_topology_detects_partial_shared_storage_views() -> None:
    class SharedViews(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            storage = torch.arange(12, dtype=torch.float32)
            self.left = torch.nn.Parameter(storage[:8].view(2, 4))
            self.right = torch.nn.Parameter(storage[4:].view(2, 4))

    model = SharedViews()
    topology = _parameter_alias_topology(model)
    assert len(topology) == 1
    storage_dtype, storage_nbytes, members = topology[0]
    assert storage_dtype == "torch.float32"
    assert storage_nbytes == 12 * 4
    assert [(member[0], member[1]) for member in members] == [
        ("left", 0),
        ("right", 4),
    ]
    assert _stored_tensor_elements(model) == 12


def test_attention_install_is_physical_counted_and_non_final() -> None:
    torch.manual_seed(7)
    parent = _ToyModel(hidden=4, intermediate=5).requires_grad_(False)
    compact = _ToyModel(hidden=2, intermediate=2).requires_grad_(False)
    encoder = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.5]]
    )
    decoder = torch.linalg.pinv(encoder)
    capsule = _capsule(
        component_id="layer0-attention",
        kind="attention",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    before = sum(value.numel() for value in parent.parameters())
    installation = install_lfm2_progressive_component(
        parent_model=parent,
        compact_model=compact,
        capsule=capsule,
        ablation_mode="isolated",
        step_index=0,
    )
    after = sum(value.numel() for value in parent.parameters())
    receipt = installation.receipt
    accounting = receipt["physical_parameter_accounting"]

    assert receipt["schema"] == LFM2_PROGRESSIVE_COMPONENT_INSTALL_SCHEMA
    assert receipt["status"].startswith("NON-FINAL")
    assert receipt["interpretation"]["standalone_child_compression_claimed"] is False
    assert accounting["whole_model_parameter_elements_before"] == before
    assert accounting["whole_model_parameter_elements_after"] == after
    assert accounting["whole_model_parameter_element_delta"] == after - before < 0
    assert accounting["counted_bridges_are_parameters"] is True
    assert receipt["operation_audit"]["parameters_pruned"] == 0
    assert receipt["operation_audit"]["training_steps"] == 0

    result, weights = parent.model.layers[0].self_attn(
        torch.randn(2, 3, 4),
        position_embeddings=None,
        attention_mask=None,
    )
    assert result.shape == (2, 3, 4)
    assert weights is None


def test_bad_bridge_pin_or_non_reducing_component_rolls_back() -> None:
    parent = _ToyModel(hidden=4, intermediate=5).requires_grad_(False)
    compact = _ToyModel(hidden=4, intermediate=5).requires_grad_(False)
    encoder = torch.eye(4)
    decoder = torch.eye(4)
    capsule = _capsule(
        component_id="nonreducing-attention",
        kind="attention",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    original = parent.model.layers[0].self_attn
    original_state = _module_state_sha256(original)
    with pytest.raises(
        Lfm2ProgressiveComponentAblationError,
        match="not physically smaller",
    ):
        install_lfm2_progressive_component(
            parent_model=parent,
            compact_model=compact,
            capsule=capsule,
            ablation_mode="isolated",
            step_index=0,
        )
    assert parent.model.layers[0].self_attn is original
    assert _module_state_sha256(parent.model.layers[0].self_attn) == original_state

    with pytest.raises(
        Lfm2ProgressiveComponentAblationError,
        match="Input encoder differs",
    ):
        Lfm2ProgressiveComponentCapsule(
            component_id="tampered",
            kind="attention",
            parent_module_path="model.layers.0.self_attn",
            compact_module_path="model.layers.0.self_attn",
            input_encoder=encoder + 1,
            output_decoder=decoder,
            expected_parent_module_sha256=original_state,
            expected_compact_module_sha256=_module_state_sha256(
                compact.model.layers[0].self_attn
            ),
            expected_input_encoder_sha256=lfm2_authenticated_tensor_sha256(
                encoder
            ),
            expected_output_decoder_sha256=lfm2_authenticated_tensor_sha256(
                decoder
            ),
            source_fold_receipt_sha256=_sha("fold"),
            source_compact_checkpoint_sha256=_sha("checkpoint"),
        )


def test_schedule_contains_isolated_and_ordered_cumulative_localization() -> None:
    parent = _ToyModel(hidden=4, intermediate=6).requires_grad_(False)
    compact = _ToyModel(hidden=2, intermediate=2).requires_grad_(False)
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    attention = _capsule(
        component_id="attention-0",
        kind="attention",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    mlp = _capsule(
        component_id="mlp-0",
        kind="mlp",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.feed_forward",
        compact_path="model.layers.0.feed_forward",
        encoder=encoder,
        decoder=decoder,
    )
    schedule = build_lfm2_progressive_component_ablation_schedule(
        (attention, mlp)
    )
    assert [value.candidate_id for value in schedule] == [
        "isolated--attention-0",
        "isolated--mlp-0",
        "cumulative--attention-0--mlp-0",
    ]
    receipt = schedule_receipt(schedule)
    assert receipt["candidate_count"] == 3
    assert receipt["topology_derived_grouping_capability_enabled"] is True
    assert receipt["topology_derived_grouping_applied"] is False
    assert receipt["duplicate_component_path_sets_emitted"] is False
    assert receipt["standalone_child_compression_claimed"] is False

    cumulative_parent = _ToyModel(hidden=4, intermediate=6).requires_grad_(False)
    cumulative_attention = _capsule(
        component_id="attention-0",
        kind="attention",
        parent=cumulative_parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    cumulative_mlp = _capsule(
        component_id="mlp-0",
        kind="mlp",
        parent=cumulative_parent,
        compact=compact,
        parent_path="model.layers.0.feed_forward",
        compact_path="model.layers.0.feed_forward",
        encoder=encoder,
        decoder=decoder,
    )
    cumulative_candidate = build_lfm2_progressive_component_ablation_schedule(
        (cumulative_attention, cumulative_mlp)
    )[-1]
    installations = apply_lfm2_component_ablation_candidate(
        parent_model=cumulative_parent,
        compact_model=compact,
        candidate=cumulative_candidate,
    )
    assert len(installations) == 2
    assert installations[1].receipt["prior_installation_sha256"] == (
        installations[0].receipt["installation_sha256"]
    )


def test_schedule_groups_every_layer_of_repeated_subsystems_without_duplicates() -> None:
    parent = _ToyModel(hidden=4, intermediate=6, layer_count=2).requires_grad_(
        False
    )
    compact = _ToyModel(hidden=2, intermediate=2, layer_count=2).requires_grad_(
        False
    )
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    capsules = []
    for layer_index in range(2):
        for kind, attribute in (
            ("attention", "self_attn"),
            ("mlp", "feed_forward"),
        ):
            capsules.append(
                _capsule(
                    component_id=f"layer-{layer_index}-{kind}",
                    kind=kind,
                    parent=parent,
                    compact=compact,
                    parent_path=f"model.layers.{layer_index}.{attribute}",
                    compact_path=f"model.layers.{layer_index}.{attribute}",
                    encoder=encoder,
                    decoder=decoder,
                )
            )
    schedule = build_lfm2_progressive_component_ablation_schedule(capsules)
    attention_group = next(
        value
        for value in schedule
        if value.candidate_id == "subsystem--all-attention"
    )
    mlp_group = next(
        value for value in schedule if value.candidate_id == "subsystem--all-mlp"
    )
    assert [value.component_id for value in attention_group.capsules] == [
        "layer-0-attention",
        "layer-1-attention",
    ]
    assert [value.component_id for value in mlp_group.capsules] == [
        "layer-0-mlp",
        "layer-1-mlp",
    ]
    component_path_sets = [
        frozenset(value.parent_module_path for value in candidate.capsules)
        for candidate in schedule
    ]
    assert len(component_path_sets) == len(set(component_path_sets))
    receipt = schedule_receipt(schedule)
    assert receipt["candidate_mode_counts"] == {
        "isolated": 4,
        "subsystem_group": 2,
        "subsystem_cumulative": 0,
        "cumulative": 3,
    }
    assert receipt["topology_derived_grouping_applied"] is True


def test_second_component_failure_atomically_restores_entire_parent() -> None:
    parent = _ToyModel(hidden=4, intermediate=6).requires_grad_(False)
    compact = _ToyModel(hidden=2, intermediate=2).requires_grad_(False)
    encoder = torch.randn(4, 2)
    decoder = torch.randn(2, 4)
    original_attention = parent.model.layers[0].self_attn
    original_mlp = parent.model.layers[0].feed_forward
    state_before = _model_state_sha256(parent)
    parameter_count_before = sum(value.numel() for value in parent.parameters())
    topology_before = tuple(
        (name, type(value).__module__, type(value).__qualname__)
        for name, value in parent.named_modules()
    )
    attention = _capsule(
        component_id="attention-first",
        kind="attention",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    valid_mlp = _capsule(
        component_id="mlp-second",
        kind="mlp",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.feed_forward",
        compact_path="model.layers.0.feed_forward",
        encoder=encoder,
        decoder=decoder,
    )
    invalid_mlp = Lfm2ProgressiveComponentCapsule(
        component_id=valid_mlp.component_id,
        kind=valid_mlp.kind,
        parent_module_path=valid_mlp.parent_module_path,
        compact_module_path=valid_mlp.compact_module_path,
        input_encoder=valid_mlp.input_encoder,
        output_decoder=valid_mlp.output_decoder,
        expected_parent_module_sha256=_sha("wrong-parent-module"),
        expected_compact_module_sha256=(
            valid_mlp.expected_compact_module_sha256
        ),
        expected_input_encoder_sha256=valid_mlp.expected_input_encoder_sha256,
        expected_output_decoder_sha256=(valid_mlp.expected_output_decoder_sha256),
        source_fold_receipt_sha256=valid_mlp.source_fold_receipt_sha256,
        source_compact_checkpoint_sha256=(
            valid_mlp.source_compact_checkpoint_sha256
        ),
    )
    candidate = build_lfm2_progressive_component_ablation_schedule(
        (attention, invalid_mlp)
    )[-1]

    with pytest.raises(
        Lfm2ProgressiveComponentAblationError,
        match="Parent or compact component differs",
    ):
        apply_lfm2_component_ablation_candidate(
            parent_model=parent,
            compact_model=compact,
            candidate=candidate,
        )

    assert parent.model.layers[0].self_attn is original_attention
    assert parent.model.layers[0].feed_forward is original_mlp
    assert _model_state_sha256(parent) == state_before
    assert sum(value.numel() for value in parent.parameters()) == parameter_count_before
    assert tuple(
        (name, type(value).__module__, type(value).__qualname__)
        for name, value in parent.named_modules()
    ) == topology_before
    assert parent.training is True
    assert all(not value.requires_grad for value in parent.parameters())


def _tiny_config(*, hidden: int, heads: int, intermediate: int):
    transformers = pytest.importorskip("transformers")
    return transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=1,
        max_position_embeddings=64,
        block_auto_adjust_ff_dim=False,
        layer_types=["full_attention", "conv"],
        tie_word_embeddings=True,
        conv_bias=False,
        conv_L_cache=3,
    )


def test_real_lfm2_hybrid_saves_fresh_reloads_and_preserves_state(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.manual_seed(11)
    parent = transformers.Lfm2ForCausalLM(
        _tiny_config(hidden=8, heads=2, intermediate=12)
    ).requires_grad_(False)
    compact = transformers.Lfm2ForCausalLM(
        _tiny_config(hidden=4, heads=1, intermediate=4)
    ).requires_grad_(False)
    parent.eval()
    compact.eval()
    aliases_before = _parameter_alias_topology(parent)
    assert tuple(
        tuple(member[0] for member in members)
        for _, _, members in aliases_before
    ) == (("lm_head.weight", "model.embed_tokens.weight"),)
    encoder = torch.randn(8, 4)
    decoder = torch.randn(4, 8)
    attention = _capsule(
        component_id="attention-layer-0",
        kind="attention",
        parent=parent,
        compact=compact,
        parent_path="model.layers.0.self_attn",
        compact_path="model.layers.0.self_attn",
        encoder=encoder,
        decoder=decoder,
    )
    candidate = build_lfm2_progressive_component_ablation_schedule((attention,))[0]
    installations = apply_lfm2_component_ablation_candidate(
        parent_model=parent,
        compact_model=compact,
        candidate=candidate,
    )
    assert _parameter_alias_topology(parent) == aliases_before
    accounting = installations[0].receipt["physical_parameter_accounting"]
    assert accounting["new_runtime_shared_parameters"] == 0
    assert accounting["native_parent_aliases_retained"] is True
    assert accounting["parameter_reduction_attributed_to_sharing"] is False
    source = tmp_path / "source-package"
    source.mkdir()
    (source / "tokenizer.json").write_text("{}", encoding="ascii")
    destination = tmp_path / "hybrid"
    manifest = write_lfm2_progressive_hybrid_checkpoint(
        model=parent,
        destination=destination,
        tokenizer_source_directory=source,
        candidate=candidate,
        installations=installations,
    )
    expected_state = _model_state_sha256(parent)
    assert manifest["schema"] == LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA
    assert manifest["status"].startswith("NON-FINAL")
    assert manifest["new_runtime_shared_parameters"] == 0
    assert manifest["native_parent_aliases_retained"] is True
    assert manifest["parameter_reduction_attributed_to_sharing"] is False
    assert [
        member["name"]
        for member in manifest["parameter_alias_topology"]["alias_groups"][0][
            "members"
        ]
    ] == ["lm_head.weight", "model.embed_tokens.weight"]
    assert (destination / HYBRID_MANIFEST_FILENAME).is_file()

    reloaded, reloaded_manifest = load_lfm2_progressive_hybrid_checkpoint(
        destination,
        expected_manifest_sha256=manifest["manifest_sha256"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert reloaded_manifest == manifest
    assert _model_state_sha256(reloaded) == expected_state
    assert _parameter_alias_topology(reloaded) == aliases_before
    assert sum(value.numel() for value in reloaded.parameters()) == (
        manifest["unique_parameter_elements"]
    )
    input_ids = torch.tensor([[1, 4, 7]], dtype=torch.long)
    with torch.inference_mode():
        before = parent(input_ids=input_ids, use_cache=False).logits
        after = reloaded(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_cross_vocab_readout_requires_counted_output_domain_bridge() -> None:
    class ReadoutModel(torch.nn.Module):
        def __init__(self, hidden: int, vocab: int) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    parent = ReadoutModel(8, 12).requires_grad_(False)
    compact = ReadoutModel(4, 5).requires_grad_(False)
    encoder = torch.randn(8, 4)
    capsule = Lfm2ProgressiveComponentCapsule(
        component_id="terminal",
        kind="terminal_readout",
        parent_module_path="lm_head",
        compact_module_path="lm_head",
        input_encoder=encoder,
        expected_parent_module_sha256=_module_state_sha256(parent.lm_head),
        expected_compact_module_sha256=_module_state_sha256(compact.lm_head),
        expected_input_encoder_sha256=lfm2_authenticated_tensor_sha256(encoder),
        source_fold_receipt_sha256=_sha("terminal-fold"),
        source_compact_checkpoint_sha256=_sha("terminal-checkpoint"),
        require_parameter_reduction=False,
    )
    original = parent.lm_head
    with pytest.raises(
        Lfm2ProgressiveComponentAblationError,
        match="Cross-tokenizer terminal replacement",
    ):
        install_lfm2_progressive_component(
            parent_model=parent,
            compact_model=compact,
            capsule=capsule,
            ablation_mode="isolated",
            step_index=0,
        )
    assert parent.lm_head is original

    output_bridge = torch.randn(5, 12)
    bridged = Lfm2ProgressiveComponentCapsule(
        component_id="terminal-bridged",
        kind="terminal_readout",
        parent_module_path="lm_head",
        compact_module_path="lm_head",
        input_encoder=encoder,
        output_domain_decoder=output_bridge,
        expected_parent_module_sha256=_module_state_sha256(parent.lm_head),
        expected_compact_module_sha256=_module_state_sha256(compact.lm_head),
        expected_input_encoder_sha256=lfm2_authenticated_tensor_sha256(encoder),
        expected_output_domain_decoder_sha256=lfm2_authenticated_tensor_sha256(
            output_bridge
        ),
        source_fold_receipt_sha256=_sha("terminal-fold"),
        source_compact_checkpoint_sha256=_sha("terminal-checkpoint"),
        require_parameter_reduction=False,
    )
    installation = install_lfm2_progressive_component(
        parent_model=parent,
        compact_model=compact,
        capsule=bridged,
        ablation_mode="isolated",
        step_index=0,
    )
    logits = parent.lm_head(torch.randn(2, 3, 8))
    assert logits.shape == (2, 3, 12)
    assert installation.receipt["physical_parameter_accounting"][
        "output_domain_bridge_parameter_elements"
    ] == 60
