"""Cross-family execution proof: a llama parent folded into a qwen3 child.

Nothing here is a same-family shortcut.  The parent is a Llama decoder with
identity Q/K semantics, eight query heads, four key/value heads and its own
ByteLevel-BPE vocabulary.  The child is a blank qwen3-shaped decoder with
per-head Q/K RMS normalization, half the heads, a narrower MLP, and a
completely unrelated ByteLevel-BPE vocabulary of a different size.  The whole
child is installed by three promoted operators inside one
:class:`DynamicFoldExecutor` run, from one sealed parent evidence bundle, with
zero parent forwards during execution.

The token interface comes from a sealed *pair* bridge rather than the strict
donor-row bridge, which means it does not reach every target row.  Those rows
are the point of this file as much as the attention math is: they must keep the
blank child's incumbent value, not the zeros the transport builder returns.
"""

from __future__ import annotations

from typing import Any

import pytest


torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    ANCHOR_PREFERRED_PAIR_TRANSPORT,
    CausalGQAAttentionBranchOperator,
    CausalGatedMLPBranchOperator,
    DynamicCausalOperatorPromotionError,
    PairCodeTokenInterfaceBridgeOperator,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    StaticEvidenceProvider,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_planner import (  # noqa: E402
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_LAYER_BRANCH_METHODS,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)
from neural_forge.hf_tokenizer_pair_bridge import (  # noqa: E402
    compile_hf_tokenizer_pair_bridge,
)


TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"

_PARENT_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}
_CHILD_SPECIALS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad0>",
    "unk_token": "<unk0>",
}

_PARENT_CORPUS = (
    "the quick brown fox jumps over the lazy dog",
    "the compiler folds the parent model into the child model",
    "tokenizer bridges align vocabulary rows without touching weights",
    "the forest is quiet and the river runs deep tonight",
    "another folded child inherits the parent interface exactly",
)
_CHILD_CORPUS = (
    "quantum superposition collapses whenever an observer measures it",
    "les jardins fleurissent au printemps quand la pluie tombe",
    "matrix factorization recovers low rank structure from noise",
    "eigenvalues drift when the operator loses its symmetry",
)


def _train_byte_level_bpe(corpus, vocab_size: int, specials: dict[str, str]) -> Any:
    """Train one real ByteLevel-BPE tokenizer on a tiny corpus."""

    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(
        [line for line in corpus for _ in range(12)],
        trainer=tokenizers.trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=1,
            show_progress=False,
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=list(specials.values()),
        ),
    )
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, **specials)


@pytest.fixture(scope="module")
def tokenizer_pair() -> tuple[Any, Any]:
    parent = _train_byte_level_bpe(_PARENT_CORPUS, 330, _PARENT_SPECIALS)
    child = _train_byte_level_bpe(_CHILD_CORPUS, 300, _CHILD_SPECIALS)
    assert len(parent) != len(child)
    assert set(parent.get_vocab()) != set(child.get_vocab())
    return parent, child


@pytest.fixture(scope="module")
def pair_bridge(tokenizer_pair, tmp_path_factory: pytest.TempPathFactory):
    parent, child = tokenizer_pair
    root = tmp_path_factory.mktemp("cross-family-bridge") / "bridge"
    return compile_hf_tokenizer_pair_bridge(parent, child, root)


def _llama_parent(tokenizer, seed: int = 20260731):
    torch.manual_seed(seed)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
    )
    model.eval()
    return model


def _blank_qwen3_child(tokenizer, seed: int = 4711):
    """One blank qwen3-shaped child: contracted heads, per-head Q/K norm.

    The blank state is deliberately *not* zero.  A pair bridge leaves rows it
    cannot reach untouched, and the transport builder returns exact zeros for
    exactly those rows, so a zero blank would make "left alone" and "installed
    the builder's zeros" indistinguishable.
    """

    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )
    )
    model.eval()
    return model


def _input_ids(tokenizer):
    return torch.arange(3, 3 + 32, dtype=torch.long).reshape(2, 16) % len(tokenizer)


class _AcceptStructuralCandidate:
    """Behavior remains the outer authority; structure is admitted here."""

    def evaluate(self, context: Any) -> FoldDecision:
        return FoldDecision.accept("Structurally complete cached branch install.")


class _CrossFamilyFold:
    """One prepared cross-family plan plus everything it needs to execute."""

    def __init__(self, tokenizer_pair, pair_bridge) -> None:
        self.parent_tokenizer, self.child_tokenizer = tokenizer_pair
        self.pair_bridge = pair_bridge
        self.parent = _llama_parent(self.parent_tokenizer)
        self.child = _blank_qwen3_child(self.child_tokenizer)
        self.parent_graph = inspect_model_circuits(
            self.parent,
            model_name="parent",
            tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(
                self.parent_tokenizer
            ),
        )
        self.child_graph = inspect_model_circuits(
            self.child,
            model_name="child",
            tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(
                self.child_tokenizer
            ),
        )
        self.evidence = capture_hf_causal_circuit_evidence(
            self.parent,
            {"input_ids": _input_ids(self.parent_tokenizer)},
            parent_graph_fingerprint=self.parent_graph.fingerprint,
            metadata={"test": "cross-family-llama-to-qwen3"},
        )
        self.blank_state = {
            name: value.detach().clone()
            for name, value in self.child.state_dict().items()
        }
        self.blank_fingerprint = fingerprint_model_state(self.child)
        self.plan = DynamicFoldPlanner(
            enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
        ).plan(
            (self.parent_graph,),
            self.child_graph,
            child_state_fingerprint=self.blank_fingerprint,
            evidence_fingerprint=self.evidence.cache_key,
            parent_state_fingerprints={
                "parent": fingerprint_model_state(self.parent)
            },
        )

    @property
    def context(self) -> FoldExecutionContext:
        return FoldExecutionContext(
            parent_models={"parent": self.parent},
            child_model=self.child,
            parent_graphs={"parent": self.parent_graph},
            child_graph=self.child_graph,
        )

    def registry(self) -> FoldOperatorRegistry:
        # Nothing is registered globally.  The pair operator claims the same
        # already-active ``token_interface_bridge`` method the planner selects
        # for unrelated tokenizer identities, so no registry default changes.
        return FoldOperatorRegistry(
            (
                ExactCopyOperator(),
                CausalGQAAttentionBranchOperator(),
                CausalGatedMLPBranchOperator(),
                PairCodeTokenInterfaceBridgeOperator(
                    artifact=self.pair_bridge,
                    embedding_transport=ANCHOR_PREFERRED_PAIR_TRANSPORT,
                    lm_head_transport=ANCHOR_PREFERRED_PAIR_TRANSPORT,
                ),
            )
        )

    def execute(self, evaluator) -> Any:
        return DynamicFoldExecutor(
            operators=self.registry(),
            evidence_provider=StaticEvidenceProvider(
                self.evidence.cache_key, self.evidence
            ),
            evaluator=evaluator,
        ).execute(self.plan, self.context)


@pytest.fixture()
def cross_family(tokenizer_pair, pair_bridge) -> _CrossFamilyFold:
    return _CrossFamilyFold(tokenizer_pair, pair_bridge)


def test_cross_family_plan_is_gap_free_and_partitions_the_child(
    cross_family: _CrossFamilyFold,
) -> None:
    plan = cross_family.plan

    assert plan.ready
    assert not plan.gaps
    methods = [action.method_id for action in plan.actions]
    assert methods.count(TOKEN_INTERFACE_BRIDGE_METHOD) == 1
    assert methods.count(CAUSAL_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_GATED_MLP_BRANCH_METHOD) == 2

    # Every learned target tensor is owned by exactly one transaction.
    owned: list[str] = [
        path for action in plan.actions for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(cross_family.child_graph.parameter_paths)

    bridge = next(
        action
        for action in plan.actions
        if action.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )
    assert sorted(bridge.target_parameter_paths) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    attention = next(
        action
        for action in plan.actions
        if action.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    )
    configuration = dict(attention.configuration)
    assert configuration["source_family"] == "llama"
    assert configuration["target_family"] == "qwen3"
    assert "model.layers.0.self_attn.q_norm.weight" in (
        attention.target_parameter_paths
    )
    assert "model.layers.0.self_attn.k_norm.weight" in (
        attention.target_parameter_paths
    )


def test_cross_family_fold_executes_completely_without_a_parent_forward(
    cross_family: _CrossFamilyFold,
) -> None:
    child = cross_family.child
    parent_forwards = 0

    def counted(_module, _args, _output):
        nonlocal parent_forwards
        parent_forwards += 1

    handle = cross_family.parent.register_forward_hook(counted)
    try:
        run = cross_family.execute(_AcceptStructuralCandidate())
    finally:
        handle.remove()

    assert run.status == "complete"
    assert all(attempt.accepted for attempt in run.attempts)
    assert parent_forwards == 0
    assert run.observed_parent_root_forward_calls == 0
    assert run.observed_parent_module_forward_calls == 0
    for name in (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "pruning_operations",
        "quantization_operations",
    ):
        assert getattr(run, name) == 0
    assert run.executed_fold_static_zero_training_audit_passed is True
    assert set(run.completed_target_parameter_paths) == set(
        cross_family.child_graph.parameter_paths
    )
    # Every accepted attempt changed only its own declared tensors, and no
    # tensor was claimed twice.
    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    assert len(changed) == len(set(changed))
    assert set(changed) <= set(cross_family.child_graph.parameter_paths)

    assert fingerprint_model_state(child) != cross_family.blank_fingerprint
    with torch.no_grad():
        logits = child(
            input_ids=_input_ids(cross_family.child_tokenizer),
            use_cache=False,
        ).logits
    assert tuple(logits.shape) == (2, 16, len(cross_family.child_tokenizer))
    assert bool(torch.isfinite(logits).all())

    attention_errors = [
        attempt.transaction_report["validation_attention_delta_relative_rmse"]
        for attempt in run.attempts
        if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    ]
    assert len(attention_errors) == 2
    # Held-out attention-branch error on a real contraction (8 -> 4 query
    # heads, 4 -> 2 key/value heads) into a per-head Q/K normalized target.
    # Behavior, not this number, remains the acceptance authority.
    assert all(0.0 < value < 0.5 for value in attention_errors)
    for attempt in run.attempts:
        if attempt.method_id != CAUSAL_ATTENTION_BRANCH_METHOD:
            continue
        report = attempt.transaction_report
        assert report["target_query_key_norm"] == "rms_norm"
        assert report["parent_forward_calls"] == 0
        assert 0.0 < report["query_norm_gain_relative_rmse"] < 1.0
        assert 0.0 < report["key_norm_gain_relative_rmse"] < 1.0


def test_pair_bridge_leaves_uncovered_target_rows_at_their_blank_value(
    cross_family: _CrossFamilyFold,
) -> None:
    run = cross_family.execute(_AcceptStructuralCandidate())
    assert run.status == "complete"
    report = next(
        attempt.transaction_report
        for attempt in run.attempts
        if attempt.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )

    assert report["bridge_kind"] == "sealed-independent-tokenizer-pair-bridge"
    assert report["masked_row_policy"] == (
        "leave-uncovered-target-rows-at-incumbent-blank"
    )
    rows = report["target_row_capacity"]
    assert rows == len(cross_family.child_tokenizer)
    covered = report["covered_output_rows"]
    uncovered = report["uncovered_output_rows"]
    assert covered + uncovered == rows
    # A pair bridge is not a total function; if it were, this test would prove
    # nothing about the masked-row policy.
    assert uncovered > 0
    assert report["embedding_uncovered_target_rows"] == uncovered
    assert report["embedding_uncovered_target_row_indices_truncated"] is False
    blank_rows = sorted(report["embedding_uncovered_target_row_indices"])
    assert len(blank_rows) == uncovered
    assert len(report["embedding_evidence_mask_sha256"]) == 64

    for path in ("model.embed_tokens.weight", "lm_head.weight"):
        installed = cross_family.child.get_parameter(path).detach()
        blank = cross_family.blank_state[path]
        uncovered_index = torch.tensor(blank_rows, dtype=torch.long)
        covered_index = torch.tensor(
            sorted(set(range(rows)) - set(blank_rows)),
            dtype=torch.long,
        )
        # Left exactly alone -- not overwritten with the builder's zeros.
        assert torch.equal(
            installed.index_select(0, uncovered_index),
            blank.index_select(0, uncovered_index),
        )
        assert bool(
            (blank.index_select(0, uncovered_index) != 0.0).any()
        ), "the blank rows must be distinguishable from installed zeros"
        # And every covered row really was replaced.
        assert not torch.equal(
            installed.index_select(0, covered_index),
            blank.index_select(0, covered_index),
        )


def test_cross_family_rejection_rolls_back_the_whole_run(
    cross_family: _CrossFamilyFold,
) -> None:
    class _RejectTheSecondAttentionBranch:
        def __init__(self) -> None:
            self.seen = 0

        def evaluate(self, context: Any) -> FoldDecision:
            if context.action.method_id != CAUSAL_ATTENTION_BRANCH_METHOD:
                return FoldDecision.accept("Accepted.")
            self.seen += 1
            if self.seen == 2:
                return FoldDecision.reject("Rejected for the rollback contract.")
            return FoldDecision.accept("Accepted.")

    run = cross_family.execute(_RejectTheSecondAttentionBranch())

    assert run.status == "rejected"
    rejected = [attempt for attempt in run.attempts if not attempt.accepted]
    assert len(rejected) == 1
    assert rejected[0].method_id == CAUSAL_ATTENTION_BRANCH_METHOD
    assert rejected[0].rollback_exact is True
    assert (
        rejected[0].transaction_report["final_state_fingerprint"]
        == rejected[0].incumbent_state_fingerprint
    )
    # The rejected branch and every branch accepted before it are undone, so a
    # partially folded child never survives a rejected run.
    assert fingerprint_model_state(cross_family.child) == (
        cross_family.blank_fingerprint
    )
    for name, value in cross_family.child.state_dict().items():
        assert torch.equal(value, cross_family.blank_state[name])


def test_pair_bridge_installs_one_tied_storage_group_from_one_transport(
    tokenizer_pair, pair_bridge
) -> None:
    parent_tokenizer, child_tokenizer = tokenizer_pair
    parent = _llama_parent(parent_tokenizer)
    torch.manual_seed(97)
    child = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=len(child_tokenizer),
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
    )
    child.eval()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(parent_tokenizer),
    )
    child_graph = inspect_model_circuits(
        child,
        model_name="child",
        tokenizer_fingerprint=fingerprint_hf_tokenizer_semantics(child_tokenizer),
    )
    blank = child.get_parameter("model.embed_tokens.weight").detach().clone()
    plan = DynamicFoldPlanner(
        enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
    ).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=fingerprint_model_state(child),
        evidence_fingerprint=canonical_sha256({"cache": "tied-pair-bridge"}),
        parent_state_fingerprints={"parent": fingerprint_model_state(parent)},
    )
    action = next(
        value
        for value in plan.actions
        if value.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )

    transaction = PairCodeTokenInterfaceBridgeOperator(
        artifact=pair_bridge
    ).prepare(
        action,
        FoldExecutionContext(
            parent_models={"parent": parent},
            child_model=child,
            parent_graphs={"parent": parent_graph},
            child_graph=child_graph,
        ),
        FoldEvidence(
            canonical_sha256({"cache": "tied-pair-bridge"}),
            None,
            channels=action.evidence_requirements,
        ),
    )
    report = transaction.report

    assert report["target_tied_storage"] is True
    # One transport, one tensor: the tie is enforced by construction, not by
    # comparing two independently compiled matrices.
    assert report["lm_head_transport_report_sha256"] == (
        report["embedding_transport_report_sha256"]
    )
    assert report["installed_parameter_paths"] == sorted(
        action.target_parameter_paths
    )
    assert torch.equal(
        child.get_parameter("lm_head.weight"),
        child.get_parameter("model.embed_tokens.weight"),
    )
    uncovered = torch.tensor(
        sorted(report["embedding_uncovered_target_row_indices"]),
        dtype=torch.long,
    )
    assert int(uncovered.numel()) == report["uncovered_output_rows"] > 0
    assert torch.equal(
        child.get_parameter("lm_head.weight").detach().index_select(0, uncovered),
        blank.index_select(0, uncovered),
    )
    assert transaction.commit()["accepted"] is True


def test_pair_bridge_operator_fails_closed_on_a_foreign_tokenizer_identity(
    cross_family: _CrossFamilyFold,
) -> None:
    action = next(
        value
        for value in cross_family.plan.actions
        if value.method_id == TOKEN_INTERFACE_BRIDGE_METHOD
    )
    operator = PairCodeTokenInterfaceBridgeOperator(artifact=cross_family.pair_bridge)
    evidence = FoldEvidence(
        cross_family.evidence.cache_key,
        cross_family.evidence,
        channels=action.evidence_requirements,
    )
    before = fingerprint_model_state(cross_family.child)

    foreign_parent_graph = inspect_model_circuits(
        cross_family.parent,
        model_name="parent",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "not-the-source"}),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="source tokenizer differs",
    ):
        operator.prepare(
            action,
            FoldExecutionContext(
                parent_models={"parent": cross_family.parent},
                child_model=cross_family.child,
                parent_graphs={"parent": foreign_parent_graph},
                child_graph=cross_family.child_graph,
            ),
            evidence,
        )

    foreign_child_graph = inspect_model_circuits(
        cross_family.child,
        model_name="child",
        tokenizer_fingerprint=canonical_sha256({"tokenizer": "not-the-target"}),
    )
    with pytest.raises(
        DynamicCausalOperatorPromotionError,
        match="target tokenizer differs",
    ):
        operator.prepare(
            action,
            FoldExecutionContext(
                parent_models={"parent": cross_family.parent},
                child_model=cross_family.child,
                parent_graphs={"parent": cross_family.parent_graph},
                child_graph=foreign_child_graph,
            ),
            evidence,
        )

    assert fingerprint_model_state(cross_family.child) == before
