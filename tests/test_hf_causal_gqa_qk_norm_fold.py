"""Per-head Q/K normalization support in the coupled GQA/RoPE kernel.

Two things have to hold at once.  A target that declares per-head Q/K RMS
normalization must be compiled into with the *target architecture's own*
semantics — gains applied per head before RoPE — and a target that does not
declare one must keep the exact bytes the kernel produced before this lane
existed.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    CausalGQAAttentionBranchOperator,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    FoldEvidence,
    FoldExecutionContext,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge import hf_causal_circuit_evidence as evidence_module  # noqa: E402
from neural_forge.hf_arch_layout import build_decoder_layout  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    HFCausalCircuitEvidenceError,
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_gqa_attention_fold import (  # noqa: E402
    HFCausalCoupledAttentionFold,
    HFCausalGQAAttentionFoldError,
    HFCausalGQABarycentricAssignment,
    HFCausalGQATopology,
)
from neural_forge.hf_transport_trace import collect_hf_transport_trace  # noqa: E402


TOKENIZER = canonical_sha256({"tokenizer": "shared-llama-qwen3"})

# Captured from the kernel at commit 29939aea, before the target Q/K
# normalization lane existed.  An identity-Q/K target must keep every one of
# these bytes: the projection solve, the replay order, and the sealed receipt
# are all unchanged for it.
_IDENTITY_GOLDEN = {
    "evidence_cache_key": (
        "f270c1d80b7452f272846fe6ee743c29a87f6583f38ec9d38e512607fb257b8d"
    ),
    "input_norm_report_sha256": (
        "fd1c8960204a8747c14bf07a922ac61bddd1b20cf6daafd6d5f526c4c352b83c"
    ),
    "attention_report_sha256": (
        "5f0d11c1fe7308ae73dc4f53f8bd40bc0c19cb6f3fb83a4634c40383fa3146e8"
    ),
    "candidate_state_fingerprint": (
        "095b4de28616d8cfd210e694e93c9b1c06c77289eb3f74c9cd0af5c4f11a704e"
    ),
    "installed": {
        "model.layers.0.input_layernorm.weight": (
            "221a58d7d42f954a30295c7810b237bf4499488f10cd5bf1a6d90e16e06fee85"
        ),
        "model.layers.0.self_attn.k_proj.weight": (
            "d96ca2178fcb347ad2f8f0fdd482252dee8734930c60a7cf5731a3111b322d59"
        ),
        "model.layers.0.self_attn.o_proj.weight": (
            "26a2e1843cfa5014dd7d4ea5d7a15f23b1838494d3c3bc7a6b082b77c75d1ca6"
        ),
        "model.layers.0.self_attn.q_proj.weight": (
            "2294698ad8ac603a0b725d2fd3f1c93ed7ba043fc57e76437152ad5aee7ffc8d"
        ),
        "model.layers.0.self_attn.v_proj.weight": (
            "ac6cff494230c86f121e8d6c94775812857a095c1eddd0f6531979f48fe13617"
        ),
    },
}


def _llama(seed: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**configuration))
    model.eval()
    return model


def _qwen3(seed: int, **changes: Any):
    torch.manual_seed(seed)
    configuration = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    configuration.update(changes)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**configuration))
    model.eval()
    return model


def _input_ids():
    return torch.arange(1, 49, dtype=torch.long).reshape(2, 24) % 32


def _payload_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _prepared_attention_branch(parent, child):
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=TOKENIZER,
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=TOKENIZER,
    )
    evidence = capture_hf_causal_circuit_evidence(
        parent,
        {"input_ids": _input_ids()},
        parent_graph_fingerprint=parent_graph.fingerprint,
        metadata={"test": "family-neutral-causal-branch-promotion"},
    )
    plan = DynamicFoldPlanner(
        enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
    ).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=fingerprint_model_state(child),
        evidence_fingerprint=evidence.cache_key,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    transaction = CausalGQAAttentionBranchOperator().prepare(
        action,
        context,
        FoldEvidence(
            evidence.cache_key,
            evidence,
            channels=action.evidence_requirements,
        ),
    )
    return evidence, transaction


def test_identity_query_key_target_fold_is_byte_identical_to_the_old_kernel() -> None:
    parent = _llama(3)
    child = _llama(7, intermediate_size=16)
    evidence, transaction = _prepared_attention_branch(parent, child)
    report = transaction.report

    assert evidence.cache_key == _IDENTITY_GOLDEN["evidence_cache_key"]
    assert report["input_norm_report_sha256"] == (
        _IDENTITY_GOLDEN["input_norm_report_sha256"]
    )
    assert report["attention_report_sha256"] == (
        _IDENTITY_GOLDEN["attention_report_sha256"]
    )
    assert transaction.candidate_state_fingerprint == (
        _IDENTITY_GOLDEN["candidate_state_fingerprint"]
    )
    assert {
        path: _payload_sha256(child.get_parameter(path))
        for path in report["installed_parameter_paths"]
    } == _IDENTITY_GOLDEN["installed"]
    # An identity target must not gain a single receipt key either, so a
    # persisted receipt from before this lane still reproduces exactly.
    assert not any(key.startswith("target_query_key_norm") for key in report)
    assert not any(key.endswith("_norm_gain_relative_rmse") for key in report)
    transaction.commit()


def _fold_from_installed_child(child, *, query_key_norm: str):
    """Read one installed attention branch back out as a replayable fold."""

    layout = build_decoder_layout(child)
    attention = layout.blocks[0].attention
    topology = HFCausalGQATopology(
        query_heads=attention.query_heads,
        key_value_heads=attention.key_value_heads,
        head_dim=attention.head_dim,
        rope_theta=float(layout.position.rope_theta),
        attention_scaling=(
            attention.scaling
            if attention.scaling is not None
            else attention.head_dim ** -0.5
        ),
        query_key_norm=query_key_norm,
        rms_norm_epsilon=attention.query_norm.epsilon,
    )
    gains = {}
    if query_key_norm == "rms_norm":
        gains = {
            "query_norm_weight": child.get_parameter(
                attention.query_norm.path + ".weight"
            ),
            "key_norm_weight": child.get_parameter(
                attention.key_norm.path + ".weight"
            ),
        }
    return HFCausalCoupledAttentionFold(
        query_weight=child.get_parameter(str(attention.query_path) + ".weight"),
        key_weight=child.get_parameter(str(attention.key_path) + ".weight"),
        value_weight=child.get_parameter(str(attention.value_path) + ".weight"),
        output_weight=child.get_parameter(str(attention.output_path) + ".weight"),
        source_topology=topology,
        target_topology=topology,
        assignment=HFCausalGQABarycentricAssignment(
            source_query_groups=((0,),),
            source_key_value_groups=((0,),),
            selected_rope_pairs_by_target_kv=((0,),),
        ),
        report={},
        **gains,
    )


def test_installed_qk_norm_child_replays_its_own_attention() -> None:
    """The compiled Q/K norm must be the target architecture's, not ours.

    The child is traced through its real ``Qwen3Attention`` after the install,
    and the installed tensors are read straight back out of the child.  Feeding
    the child's own pre-attention norm output through
    :meth:`HFCausalCoupledAttentionFold.replay` has to reproduce the delta the
    module actually computed, which is only true if the gains are applied per
    head, as a direct scale, and before RoPE.
    """

    parent = _llama(11)
    child = _qwen3(13)
    _evidence, transaction = _prepared_attention_branch(parent, child)
    transaction.commit()

    input_ids = _input_ids()
    trace = collect_hf_transport_trace(child, {"input_ids": input_ids})
    block = trace.blocks[0]
    sequence = int(input_ids.shape[1])
    positions = (
        torch.arange(sequence, dtype=torch.long)
        .reshape(1, sequence)
        .expand(int(input_ids.shape[0]), sequence)
    )
    observed = block.attention_delta.detach().to(dtype=torch.float64)

    replayed, scores, _probabilities = _fold_from_installed_child(
        child, query_key_norm="rms_norm"
    ).replay(
        block.pre_attention_norm_output,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        position_ids=positions,
    )
    normalized_error = _relative_rmse(replayed, observed)
    assert normalized_error < 1e-5

    # Control: reading the same installed tensors as an identity-Q/K attention
    # reproduces neither the module's residual write nor its score geometry, so
    # the gains are genuinely load-bearing rather than decorative.
    unnormalized, plain_scores, _probabilities = _fold_from_installed_child(
        child, query_key_norm="identity"
    ).replay(
        block.pre_attention_norm_output,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        position_ids=positions,
    )
    assert _relative_rmse(unnormalized, observed) > 100.0 * normalized_error
    assert _relative_rmse(plain_scores, scores) > 0.1


def test_evidence_capture_rejects_a_non_direct_scale_source_head_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source head gain is consumed as a direct scale, so prove it is one.

    The kernel multiplies the cached ``q_norm``/``k_norm`` weights straight
    into the source head coordinates.  A unit-offset gain stores ``1 + w``, so
    reading it directly is off by one per channel.  The family allowlist blocks
    gemma today, but this record must not depend on the allowlist for that.
    """

    parent = _qwen3(29)
    honest = build_decoder_layout(parent)
    forwards = 0

    def counted(_module, _args, _output):
        nonlocal forwards
        forwards += 1

    def _unit_offset_layout(model: Any):
        assert model is parent
        blocks = tuple(
            replace(
                block,
                attention=replace(
                    block.attention,
                    key_norm=replace(
                        block.attention.key_norm, weight_kind="unit_offset"
                    ),
                ),
            )
            for block in honest.blocks
        )
        return replace(honest, blocks=blocks)

    monkeypatch.setattr(evidence_module, "build_decoder_layout", _unit_offset_layout)
    handle = parent.register_forward_hook(counted)
    try:
        with pytest.raises(
            HFCausalCircuitEvidenceError,
            match="direct-scale RMSNorm gains",
        ):
            capture_hf_causal_circuit_evidence(
                parent,
                {"input_ids": _input_ids()},
                parent_graph_fingerprint=TOKENIZER,
            )
    finally:
        handle.remove()

    assert forwards == 0


def test_coupled_fold_rejects_gains_that_disagree_with_the_declared_target() -> None:
    identity = HFCausalGQATopology(
        query_heads=2,
        key_value_heads=1,
        head_dim=4,
        rope_theta=10000.0,
        attention_scaling=0.5,
    )
    normalized = HFCausalGQATopology(
        query_heads=2,
        key_value_heads=1,
        head_dim=4,
        rope_theta=10000.0,
        attention_scaling=0.5,
        query_key_norm="rms_norm",
    )
    assignment = HFCausalGQABarycentricAssignment(
        source_query_groups=((0,), (1,)),
        source_key_value_groups=((0,),),
        selected_rope_pairs_by_target_kv=((0, 1),),
    )
    weights = {
        "query_weight": torch.zeros(8, 6, dtype=torch.float64),
        "key_weight": torch.zeros(4, 6, dtype=torch.float64),
        "value_weight": torch.zeros(4, 6, dtype=torch.float64),
        "output_weight": torch.zeros(6, 8, dtype=torch.float64),
        "source_topology": identity,
        "assignment": assignment,
        "report": {},
    }
    gain = torch.ones(4, dtype=torch.float64)

    with pytest.raises(HFCausalGQAAttentionFoldError, match="exactly when"):
        HFCausalCoupledAttentionFold(target_topology=normalized, **weights)
    with pytest.raises(HFCausalGQAAttentionFoldError, match="exactly when"):
        HFCausalCoupledAttentionFold(
            target_topology=identity,
            query_norm_weight=gain,
            key_norm_weight=gain,
            **weights,
        )
    with pytest.raises(HFCausalGQAAttentionFoldError, match="wrong target head"):
        HFCausalCoupledAttentionFold(
            target_topology=normalized,
            query_norm_weight=torch.ones(3, dtype=torch.float64),
            key_norm_weight=gain,
            **weights,
        )

    fold = HFCausalCoupledAttentionFold(
        target_topology=normalized,
        query_norm_weight=gain,
        key_norm_weight=gain,
        **weights,
    )
    assert tuple(fold.query_norm_weight.shape) == (4,)
    assert tuple(fold.key_norm_weight.shape) == (4,)
