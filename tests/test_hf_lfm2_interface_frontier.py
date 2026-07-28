from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_interface_frontier as interface_module  # noqa: E402
from neural_forge.hf_lfm2_cpcf_blank_target import (  # noqa: E402
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    Lfm2DenseRootHybridCausalEvidence,
    Lfm2InterfaceFrontierError,
    Lfm2InterfaceRecipe,
    base_tied_target_budget_for_recipe,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_causal_trie_evidence,
    compile_lfm2_cached_logit_residual_head,
    compile_lfm2_dense_root_hybrid_causal_evidence,
    compile_lfm2_interface_sequence_plan,
    compile_lfm2_inverse_bucket_output_table,
    compile_lfm2_inverse_output_plan,
    fold_lfm2_interface_candidate,
    plan_lfm2_dense_untied_target_rebalance,
    target_unique_parameters_for_recipe,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)
def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokenizer(path: Path, vocabulary: dict[str, int]):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _no_unk_tokenizer(path: Path, vocabulary: dict[str, int]):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(WordLevel(vocabulary))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _bpe_tokenizer(
    path: Path,
    vocabulary: dict[str, int],
    merges: list[tuple[str, str]],
):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(
        BPE(vocab=vocabulary, merges=merges, unk_token="<unk>")
    )
    backend.decoder = decoders.Fuse()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _bpe_bridge_case(
    tmp_path: Path,
    *,
    donor_merges: list[tuple[str, str]],
    target_merges: list[tuple[str, str]],
    donor_row_capacity: int = 8,
):
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "a": 4,
        "b": 5,
        "ab": 6,
        "c": 7,
    }
    donor = _bpe_tokenizer(tmp_path / "bpe-donor", vocabulary, donor_merges)
    target_path = tmp_path / "bpe-target"
    target = _bpe_tokenizer(target_path, vocabulary, target_merges)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=donor_row_capacity,
        target_row_capacity=8,
    )
    return bridge, donor, target


def _parent():
    config = transformers.Lfm2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(91)
    return transformers.Lfm2ForCausalLM(config).eval()


def test_public_causal_body_fingerprint_excludes_untied_interfaces() -> None:
    model = _parent()
    model.set_output_embeddings(torch.nn.Linear(16, 40, bias=False))
    assert (
        model.get_input_embeddings().weight
        is not model.get_output_embeddings().weight
    )
    expected = canonical_sha256(
        list(
            interface_module._body_records(
                model,
                interface_storage_ids=interface_module._interface_storage_ids(model),
            )
        )
    )
    observed = interface_module.fingerprint_lfm2_causal_body_inventory(model)
    assert observed == expected
    assert "fingerprint_lfm2_causal_body_inventory" in interface_module.__all__
    with torch.no_grad():
        model.get_input_embeddings().weight.add_(1)
        model.get_output_embeddings().weight.sub_(1)
    assert interface_module.fingerprint_lfm2_causal_body_inventory(model) == observed
    interface_storage_ids = interface_module._interface_storage_ids(model)
    body_parameter = next(
        parameter
        for parameter in model.parameters()
        if interface_module._storage_identity(parameter) not in interface_storage_ids
    )
    with torch.no_grad():
        body_parameter.reshape(-1)[0].add_(1)
    assert interface_module.fingerprint_lfm2_causal_body_inventory(model) != observed


def test_public_terminal_cache_collector_returns_typed_reusable_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    cache_key = _sha("terminal-cache")
    parent_identity = _sha("terminal-parent")
    phase = types.SimpleNamespace(
        parent_sampled_suffix_tokens=2,
        target_position_origin=0,
        donor_full_prefix_tokens=2,
        donor_full_prefix_token_ids=(2, 4),
        target_full_prefix_token_ids=(2, 4),
        target_to_donor_boundary_positions=(0, 1),
    )
    alignment_digest = _sha("terminal-alignment")

    class Alignment:
        source_batch_index = 0
        alignment_sha256 = alignment_digest

        @staticmethod
        def get_phase(name: str):
            assert name == "decode"
            return phase

        @staticmethod
        def to_evidence_dict() -> dict[str, str]:
            return {"alignment_sha256": alignment_digest}

    hidden = torch.randn(1, 2, 16)
    logits = torch.randn(1, 2, 40)
    payload = types.SimpleNamespace(
        tensors={
            "decode.output.lm_head_input": hidden,
            "decode.logits": logits,
        },
        selection_sha256=_sha("selection"),
        logical_bytes=int(hidden.numel() + logits.numel()) * 4,
    )
    reference = types.SimpleNamespace(
        index=0,
        ordered_tensor_names=tuple(payload.tensors),
        input_sha256=_sha("input"),
    )

    class Session:
        batch_references = (reference,)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        @staticmethod
        def load_selected_tensors(index: int, names: tuple[str, ...]):
            assert index == 0
            assert names == tuple(sorted(payload.tensors))
            return payload

        @staticmethod
        def release_selected_tensors(observed) -> None:
            assert observed is payload

        @staticmethod
        def verify() -> None:
            return None

        @staticmethod
        def as_dict() -> dict[str, str]:
            return {
                "session_sha256": _sha("session"),
                "manifest_sha256": _sha("manifest"),
                "completion_sha256": _sha("completion"),
            }

    cache = types.SimpleNamespace(
        cache_key=cache_key,
        parent_fingerprint=parent_identity,
        selective_authenticated_session=Session,
    )

    def semantic_alignments_from_session(**_kwargs):
        return (Alignment(),), {"session": "live-local-tokenizers"}

    monkeypatch.setitem(
        sys.modules,
        "neural_forge.hf_lfm2_semantic_postfold",
        types.SimpleNamespace(
            _semantic_alignments_from_session=semantic_alignments_from_session
        ),
    )
    evidence = interface_module.collect_lfm2_terminal_cache_evidence(
        cache,
        expected_cache_key=cache_key,
        expected_parent_identity_sha256=parent_identity,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
    )
    assert isinstance(evidence, interface_module.Lfm2TerminalCacheEvidence)
    assert tuple(evidence.hidden_rows.shape) == (2, 16)
    assert tuple(evidence.donor_logits.shape) == (2, 40)
    assert len(evidence.causal_root_records) == 2
    receipt = evidence.as_dict()
    assert receipt["target_vocabulary_size"] == 16
    assert receipt["sealed_parent_cache_reused_without_recapture"] is True
    assert receipt["operation_audit"]["parent_forward_calls"] == 0


def _case(tmp_path: Path):
    donor_tokens = [
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
        "a",
        "!",
        *[f"t{index}" for index in range(6, 16)],
        "a!",
        *[f"t{index}" for index in range(17, 40)],
    ]
    donor_vocab = {token: index for index, token in enumerate(donor_tokens)}
    target_vocab = {
        token: index for index, token in enumerate(donor_tokens[:16])
    }
    donor = _tokenizer(tmp_path / "donor", donor_vocab)
    target_path = tmp_path / "target"
    target = _tokenizer(target_path, target_vocab)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=40,
        target_row_capacity=16,
    )
    parent = _parent()
    body_parameters = (
        sum(parameter.numel() for parameter in parent.parameters()) - 40 * 16
    )
    base_parameters = body_parameters + 16 * 16
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="interface-frontier-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    return parent, blank, bridge, donor, target, base_parameters


def test_context_plan_captures_non_strict_boundary_token(tmp_path: Path) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor,
        target,
        contexts=("", "a"),
    )
    exclamation = plan["records"][5]
    assert [5] in exclamation["donor_sequences"]
    assert [16] in exclamation["donor_sequences"]


def test_public_contextual_input_compiler_accepts_projected_child_width(
    tmp_path: Path,
) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor,
        target,
        contexts=("", "a"),
    )
    torch.manual_seed(107)
    projected_parent_rows = torch.randn(40, 7)
    public = interface_module.compile_lfm2_contextual_input_table(
        projected_parent_rows,
        plan,
        strict_anchor_ppm=500_000,
        preserve_anchor_norm=True,
    )
    compatibility = interface_module._compile_context_table(
        projected_parent_rows,
        plan,
        strict_anchor_ppm=500_000,
        preserve_anchor_norm=True,
    )
    assert tuple(public.shape) == (16, 7)
    torch.testing.assert_close(public, compatibility, rtol=0, atol=0)
    unsealed = dict(plan)
    unsealed["contexts"] = ["fabricated-context"]
    with pytest.raises(Lfm2InterfaceFrontierError, match="unsealed"):
        interface_module.compile_lfm2_contextual_input_table(
            projected_parent_rows,
            unsealed,
            strict_anchor_ppm=500_000,
            preserve_anchor_norm=True,
        )
    assert plan["rows_with_context_variants"] >= 1
    assert plan["plan_sha256"]


def test_inverse_output_plan_covers_every_donor_row_once(tmp_path: Path) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    flattened = [value for bucket in plan["buckets"] for value in bucket]
    assert sorted(flattened) == list(range(40))
    assert plan["assigned_donor_row_count"] == 40
    assert plan["donor_rows_assigned_exactly_once"] is True


def test_inverse_output_plan_explicitly_sinks_no_unk_capacity_rows(
    tmp_path: Path,
) -> None:
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        **{f"t{index}": index for index in range(3, 16)},
    }
    donor = _no_unk_tokenizer(tmp_path / "donor-no-unk", vocabulary)
    target_path = tmp_path / "target-no-unk"
    target = _no_unk_tokenizer(target_path, vocabulary)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=40,
        target_row_capacity=16,
    )
    assert donor.unk_token_id is None
    assert target.unk_token_id is None
    with pytest.raises(
        Lfm2InterfaceFrontierError,
        match="explicit fallback target row",
    ):
        compile_lfm2_inverse_output_plan(bridge, donor, target)
    plan = compile_lfm2_inverse_output_plan(
        bridge,
        donor,
        target,
        fallback_target_id=target.pad_token_id,
    )
    flattened = [value for bucket in plan["buckets"] for value in bucket]
    assert sorted(flattened) == list(range(40))
    assert plan["fallback_policy"] == "explicit-authenticated-target-row"
    assert plan["fallback_target_id"] == 0
    assert plan["donor_tokenizer_defined_row_count"] == 16
    assert plan["invalid_donor_capacity_row_count"] == 24
    assert plan["valid_empty_decomposition_count"] == 0
    assert plan["fallback_assignment_count"] == 24
    assert plan["empty_decomposition_count"] == 24
    assert set(range(16, 40)).issubset(plan["buckets"][0])


def test_context_tied_candidate_preserves_body_and_composes_rows(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    parent_before = fingerprint_model_state(parent)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="context-mean-norm-tied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    assert receipt["exact_parent_body_depth_width_preserved"] is True
    assert receipt["native_input_output_tying_preserved"] is True
    assert receipt["target_unique_parameters"] == base_parameters
    assert result.model.get_input_embeddings().weight is result.model.get_output_embeddings().weight
    assert not torch.equal(
        result.model.get_input_embeddings().weight[5],
        parent.get_input_embeddings().weight[5],
    )
    assert fingerprint_model_state(parent) == parent_before
    assert receipt["operation_audit"]["training_steps"] == 0


def test_asymmetric_candidate_materializes_dense_untied_output(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="context-input-strict-output-untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
        output_strict_anchor_ppm=1_000_000,
    )
    expected = target_unique_parameters_for_recipe(
        base_tied_target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        hidden_size=16,
        recipe=recipe,
    )
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    input_weight = result.model.get_input_embeddings().weight
    output_weight = result.model.get_output_embeddings().weight
    mapping = torch.tensor(
        bridge.transport_profile.target_to_donor_row_indices,
        dtype=torch.long,
    )
    assert input_weight is not output_weight
    assert receipt["native_input_output_tying_preserved"] is False
    assert receipt["compiler_owned_untied_blank"]["added_dense_parameters"] == 256
    assert receipt["target_unique_parameters"] == expected
    assert torch.equal(
        output_weight,
        parent.get_output_embeddings().weight.index_select(0, mapping),
    )
    assert receipt["operation_audit"]["runtime_shared_parent_parameters"] == 0


def test_inverse_bucket_candidate_uses_all_parent_output_rows(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="inverse-bucket-output",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="inverse-bucket-centroid",
        output_strict_anchor_ppm=0,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    )
    expected = base_parameters + 16 * 16
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    assert receipt["inverse_output_plan"]["assigned_donor_row_count"] == 40
    assert receipt["inverse_output_plan"]["donor_rows_assigned_exactly_once"] is True
    assert receipt["inverse_output_plan_sha256"]
    assert result.model.get_input_embeddings().weight is not result.model.get_output_embeddings().weight


def test_cached_logit_residual_head_is_closed_form_and_preserves_specials(
    tmp_path: Path,
) -> None:
    parent, _blank, bridge, donor, target, _base_parameters = _case(tmp_path)
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(18)
    hidden = torch.randn(8, 16)
    donor_logits = hidden.matmul(parent.get_output_embeddings().weight.t())
    result = compile_lfm2_cached_logit_residual_head(
        structural,
        hidden,
        donor_logits,
        inverse_plan,
        source_evidence_sha256=_sha("sealed-terminal-evidence"),
        ridge=1e-3,
        residual_scale_ppm=1_000_000,
        protected_target_ids=(0, 1, 2, 3),
        solve_device="cpu",
    )
    receipt = result.as_dict()
    assert receipt["relative_centered_error_after"] < receipt[
        "relative_centered_error_before"
    ]
    assert receipt["diagnostic_metrics_are_not_acceptance_gates"] is True
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["autograd_grad_calls"] == 0
    torch.testing.assert_close(result.table[:4], structural[:4], rtol=0, atol=0)


def test_cached_logit_residual_candidate_installs_as_dense_untied_head(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(19)
    hidden = torch.randn(8, 16)
    donor_logits = hidden.matmul(parent.get_output_embeddings().weight.t())
    residual_head = compile_lfm2_cached_logit_residual_head(
        structural,
        hidden,
        donor_logits,
        inverse_plan,
        source_evidence_sha256=_sha("sealed-terminal-evidence"),
        ridge=1e-2,
        residual_scale_ppm=500_000,
        solve_device="cpu",
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="cached-logit-residual",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="cached-logit-residual",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    )
    expected = base_parameters + 16 * 16
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
        cached_logit_residual_head=residual_head,
    )
    receipt = folded.as_dict()
    assert receipt["cached_logit_residual_head"]["result_sha256"] == (
        residual_head.as_dict()["result_sha256"]
    )
    assert folded.model.get_input_embeddings().weight is not folded.model.get_output_embeddings().weight
    torch.testing.assert_close(
        folded.model.get_output_embeddings().weight,
        residual_head.table,
        rtol=0,
        atol=0,
    )


def _plan_with_internal_target_prefixes(plan: dict) -> dict:
    body = dict(plan)
    body.pop("plan_sha256")
    donor_rows = int(body["donor_row_count"])
    target_rows = int(body["target_row_count"])
    decompositions = [[donor_id % target_rows] for donor_id in range(donor_rows)]
    for donor_id in range(donor_rows):
        if decompositions[donor_id][0] in {4, 7}:
            decompositions[donor_id] = [9]
    decompositions[0] = [4]
    decompositions[1] = [4, 5]
    decompositions[2] = [4, 6]
    decompositions[3] = [7, 8]
    buckets = [[] for _target_id in range(target_rows)]
    prefix_buckets = [[] for _target_id in range(target_rows)]
    for donor_id, decomposition in enumerate(decompositions):
        buckets[decomposition[0]].append(donor_id)
        prefix_buckets[decomposition[-1]].append(donor_id)
    body["donor_target_decompositions"] = decompositions
    body["buckets"] = buckets
    body["prefix_transition_buckets"] = prefix_buckets
    return {**body, "plan_sha256": canonical_sha256(body)}


def _causal_root(
    terminal_row_index: int,
    *,
    donor_prefix: tuple[int, ...] = (4,),
    target_prefix: tuple[int, ...] = (4,),
) -> dict:
    return {
        "terminal_row_index": terminal_row_index,
        "batch_index": 0,
        "phase": "decode.000",
        "phase_row_index": terminal_row_index,
        "donor_boundary_position": len(donor_prefix) - 1,
        "target_boundary_position": len(target_prefix) - 1,
        "donor_prefix_token_ids": donor_prefix,
        "target_prefix_token_ids": target_prefix,
        "semantic_alignment_sha256": _sha(f"alignment-{terminal_row_index}"),
    }


def _terminal_evidence(donor_logits, roots, bridge) -> dict:
    normalized = interface_module._causal_root_evidence_rows(
        roots,
        donor_row_count=int(donor_logits.shape[1]),
        target_row_count=bridge.transport_profile.target_row_capacity,
        terminal_row_count=int(donor_logits.shape[0]),
    )
    body = {
        "schema": "neural-forge-lfm2-interface-terminal-cache-evidence-v1",
        "cache_key": _sha("cache-key"),
        "parent_identity_sha256": _sha("parent-identity"),
        "selective_session_sha256": _sha("selective-session"),
        "manifest_sha256": _sha("manifest"),
        "completion_sha256": _sha("completion"),
        "semantic_alignment_sha256": _sha("semantic-alignments"),
        "tokenizer_live_session_sha256": _sha("tokenizer-live-session"),
        "terminal_row_count": int(donor_logits.shape[0]),
        "donor_vocabulary_size": int(donor_logits.shape[1]),
        "donor_logits_sha256": interface_module._tensor_sha256(donor_logits),
        "causal_root_row_count": len(normalized),
        "causal_root_rows_sha256": canonical_sha256(list(normalized)),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "operation_audit": {
            "training_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
        },
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _child_feature_evidence(
    child_hidden,
    trie,
    *,
    input_sha256: str = _sha("causal-input-table"),
    body_sha256: str = _sha("causal-body"),
) -> dict:
    trie_receipt = trie.as_dict()
    state = _sha("child-model-state")
    body = {
        "schema": "neural-forge-lfm2-child-causal-residual-evidence-v2",
        "status": "complete-target-native-child-residual-replay",
        "evidence_role": "unit-test",
        "trie_evidence_sha256": trie_receipt["result_sha256"],
        "parent_terminal_evidence_sha256": trie_receipt[
            "parent_terminal_evidence_sha256"
        ],
        "cache_key": trie_receipt["cache_key"],
        "parent_identity_sha256": trie_receipt["parent_identity_sha256"],
        "selective_session_sha256": trie_receipt["selective_session_sha256"],
        "manifest_sha256": trie_receipt["manifest_sha256"],
        "completion_sha256": trie_receipt["completion_sha256"],
        "semantic_alignment_sha256": trie_receipt[
            "semantic_alignment_sha256"
        ],
        "tokenizer_live_session_sha256": trie_receipt[
            "tokenizer_live_session_sha256"
        ],
        "causal_root_rows_sha256": trie_receipt["causal_root_rows_sha256"],
        "contextual_transition_plan_sha256": trie_receipt[
            "contextual_transition_plan_sha256"
        ],
        "bridge_artifact_sha256": trie_receipt["bridge_artifact_sha256"],
        "bridge_profile_sha256": trie_receipt["bridge_profile_sha256"],
        "donor_tokenizer_semantics_sha256": trie_receipt[
            "donor_tokenizer_semantics_sha256"
        ],
        "target_tokenizer_semantics_sha256": trie_receipt[
            "target_tokenizer_semantics_sha256"
        ],
        "target_prefix_rows_sha256": trie_receipt["target_prefix_rows_sha256"],
        "child_hidden_rows_sha256": interface_module._tensor_sha256(child_hidden),
        "causal_input_table_sha256": input_sha256,
        "causal_body_inventory_sha256": body_sha256,
        "causal_input_table_after_sha256": input_sha256,
        "causal_body_inventory_after_sha256": body_sha256,
        "row_count": int(child_hidden.shape[0]),
        "hidden_size": int(child_hidden.shape[1]),
        "child_model_state_before_sha256": state,
        "child_model_state_after_sha256": state,
        "child_state_preserved": True,
        "input_and_body_coordinate_preserved": True,
        "batch_size": 1,
        "full_prefix_use_cache_false": True,
        "position_origin": 0,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": int(child_hidden.shape[0]),
            "child_generate_calls": 0,
            "network_access": False,
        },
    }
    receipt = {**body, "evidence_sha256": canonical_sha256(body)}
    return interface_module.Lfm2ChildCausalResidualEvidence(
        hidden_rows=child_hidden,
        receipt=receipt,
    )


def _typed_causal_evidence(
    bridge,
    donor,
    target,
    *,
    evidence_rows: int,
    maximum_candidates: int = 1,
):
    donor_logits = torch.full((evidence_rows, 40), -8.0)
    donor_logits[:, int(donor.eos_token_id)] = 8.0
    roots = [_causal_root(index) for index in range(evidence_rows)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    trie = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=maximum_candidates,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=max(2, evidence_rows),
    )
    return trie, terminal


def test_causal_trie_retains_partial_contextual_root_support_and_eos(
    tmp_path: Path,
) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    donor_logits = torch.full((2, 40), -8.0)
    donor_logits[:, int(donor.eos_token_id)] = 9.0
    donor_logits[:, 4] = 8.0
    roots = [_causal_root(0), _causal_root(1)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    result = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=4,
        maximum_total_internal_prefixes=8,
    )
    receipt = result.as_dict()
    assert receipt["root_row_count"] == 2
    assert receipt["all_root_rows_retained"] is True
    assert receipt["root_support_is_bounded_partial_observation"] is True
    assert receipt["root_unknown_cells_receive_finite_floor"] is False
    assert bool(result.target_logit_support_mask[:2].all()) is False
    assert bool(
        result.target_logit_support_mask[:2, int(target.eos_token_id)].all()
    ) is True
    plans = receipt["contextual_transition_plan"]["root_transition_plans"]
    assert all(
        row["append_only_candidates"][0]["transition_kind"]
        == "authenticated-special-control"
        for row in plans
    )
    assert receipt["context_free_first_bucket_plan_used"] is False
    assert bool(torch.isfinite(result.target_logits).all()) is True


def test_causal_v2_quarantines_contextual_suffix_rewrite(
    tmp_path: Path,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[],
        target_merges=[("a", "b")],
    )
    assert donor.encode("a", add_special_tokens=False) == [4]
    assert donor.encode("b", add_special_tokens=False) == [5]
    assert target.encode("ab", add_special_tokens=False) == [6]
    donor_logits = torch.full((1, 8), -9.0)
    donor_logits[0, 5] = 9.0
    roots = [_causal_root(0)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    result = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=1,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=2,
    )
    receipt = result.as_dict()
    plan = receipt["contextual_transition_plan"]["root_transition_plans"][0]
    assert plan["quarantined_donor_candidate_ids"] == [5]
    assert plan["quarantined_candidates"][0]["reason"] == "target-suffix-rewrite"
    assert plan["quarantined_log_probability_mass"] == pytest.approx(0.0, abs=1e-6)
    assert receipt["root_rows_without_observed_append_support"] == 1
    assert bool(result.target_logit_support_mask[0].any()) is False


def test_causal_v2_rejects_resealed_context_free_plan(
    tmp_path: Path,
) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    context_free = compile_lfm2_inverse_output_plan(bridge, donor, target)
    body = dict(context_free)
    body.pop("plan_sha256")
    body["donor_target_decompositions"][5] = [4, 5]
    resealed = {**body, "plan_sha256": canonical_sha256(body)}
    donor_logits = torch.zeros((1, 40))
    roots = [_causal_root(0)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    with pytest.raises(Lfm2InterfaceFrontierError, match="context-free"):
        compile_lfm2_causal_trie_evidence(
            donor_logits,
            roots,
            resealed,
            donor,
            target,
            parent_terminal_evidence=terminal,
        )


def test_causal_v2_excludes_undefined_donor_rows_before_topk(
    tmp_path: Path,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[],
        target_merges=[],
        donor_row_capacity=10,
    )
    donor_logits = torch.full((1, 10), -9.0)
    donor_logits[0, 9] = 100.0
    donor_logits[0, int(donor.eos_token_id)] = 8.0
    roots = [_causal_root(0)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    result = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=1,
    )
    receipt = result.as_dict()
    transition = receipt["contextual_transition_plan"]
    root_plan = transition["root_transition_plans"][0]
    assert transition["undefined_donor_row_count"] == 2
    assert transition["all_undefined_donor_rows_excluded_before_topk"] is True
    assert root_plan["inspected_donor_candidate_ids"] == [int(donor.eos_token_id)]
    assert root_plan["undefined_donor_log_probability_mass_over_full_head"] == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_internal_prefix_allocation_is_invariant_to_root_logit_offset(
    tmp_path: Path,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[("a", "b")],
        target_merges=[],
    )
    roots = [
        _causal_root(0, donor_prefix=(7,), target_prefix=(7,)),
        _causal_root(1, donor_prefix=(7,), target_prefix=(7,)),
    ]

    def compile_with_offset(offset: float):
        donor_logits = torch.full((2, 8), -8.0)
        donor_logits[:, 6] = 8.0
        donor_logits[1] += offset
        terminal = _terminal_evidence(donor_logits, roots, bridge)
        return compile_lfm2_causal_trie_evidence(
            donor_logits,
            roots,
            bridge,
            donor,
            target,
            parent_terminal_evidence=terminal,
            maximum_donor_candidates_per_root=1,
            maximum_internal_prefixes_per_root=1,
            maximum_total_internal_prefixes=1,
        )

    baseline = compile_with_offset(0.0).as_dict()
    shifted = compile_with_offset(1_000.0).as_dict()
    assert baseline["node_records"][0]["root_index"] == 0
    assert shifted["node_records"][0]["root_index"] == 0
    assert shifted["cross_root_prefix_priority_uses_normalized_log_probability"] is True


def _dense_root_hybrid_case(tmp_path: Path):
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[("a", "b")],
        target_merges=[],
    )
    donor_logits = torch.tensor(
        [[-7.0, -6.0, -5.0, -4.0, 8.5, 7.0, 10.0, -3.0]],
        dtype=torch.float32,
    )
    roots = [
        _causal_root(
            0,
            donor_prefix=(7,),
            target_prefix=(7,),
        )
    ]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    causal = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=1,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=2,
    )
    assert causal.as_dict()["internal_prefix_row_count"] == 1
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    return causal, donor_logits, inverse_plan, terminal


def test_dense_root_hybrid_uses_all_mass_exact_override_and_sparse_internal_rows(
    tmp_path: Path,
) -> None:
    causal, donor_logits, inverse_plan, terminal = _dense_root_hybrid_case(tmp_path)
    first = compile_lfm2_dense_root_hybrid_causal_evidence(
        causal,
        donor_logits,
        inverse_plan,
        maximum_rows_per_aggregation_chunk=1,
    )
    second = compile_lfm2_dense_root_hybrid_causal_evidence(
        causal,
        donor_logits,
        inverse_plan,
        maximum_rows_per_aggregation_chunk=1,
    )
    receipt = first.as_dict()
    assert isinstance(first, Lfm2DenseRootHybridCausalEvidence)
    assert isinstance(first, interface_module.Lfm2CausalTrieEvidence)
    assert receipt["result_sha256"] == second.as_dict()["result_sha256"]
    torch.testing.assert_close(
        first.target_logits, second.target_logits, rtol=0, atol=0
    )
    assert torch.equal(
        first.target_logit_support_mask,
        second.target_logit_support_mask,
    )

    assignment = torch.full((int(donor_logits.shape[1]),), -1, dtype=torch.long)
    for target_id, bucket in enumerate(inverse_plan["buckets"]):
        if bucket:
            assignment[torch.tensor(bucket, dtype=torch.long)] = target_id
    assert bool((assignment >= 0).all())
    aggregated = interface_module._aggregate_donor_logit_mass(
        donor_logits,
        assignment,
        target_row_count=int(first.target_logits.shape[1]),
        maximum_rows_per_chunk=1,
    )
    nonempty = receipt["nonempty_root_prior_target_ids"]
    empty = receipt["empty_root_prior_target_ids"]
    source_root_support = causal.target_logit_support_mask[0]
    exact_target_id = int(torch.nonzero(source_root_support)[0].item())
    assert exact_target_id in nonempty
    assert first.target_logits[0, exact_target_id].item() == (
        causal.target_logits[0, exact_target_id].item()
    )
    assert first.target_logits[0, exact_target_id].item() != pytest.approx(
        aggregated[0, exact_target_id].item(),
        abs=1e-6,
    )
    dense_only_target_id = next(
        target_id for target_id in nonempty if not bool(source_root_support[target_id])
    )
    assert first.target_logits[0, dense_only_target_id].item() == pytest.approx(
        aggregated[0, dense_only_target_id].item(),
        abs=1e-6,
    )
    assert bool(first.target_logit_support_mask[0, nonempty].all()) is True
    assert bool(first.target_logit_support_mask[0, empty].any()) is False
    torch.testing.assert_close(
        first.target_logits[1:],
        causal.target_logits[1:],
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        first.target_logit_support_mask[1:],
        causal.target_logit_support_mask[1:],
    )
    assert receipt["assigned_donor_row_count"] == int(donor_logits.shape[1])
    assert receipt["all_donor_rows_assigned_exactly_once"] is True
    assert receipt["all_donor_logit_mass_aggregated_into_root_prior"] is True
    assert receipt["dense_root_prior_is_context_free"] is True
    assert receipt["dense_root_prior_is_exact_branch_equivalence"] is False
    assert receipt["context_free_bucket_aggregation_is_dense_root_prior_only"] is True
    assert receipt["exact_all_branch_autoregressive_equivalence_claimed"] is False
    assert receipt["exact_contextual_root_values_override_dense_prior"] is True
    assert receipt["root_empty_bucket_cells_use_structural_baseline"] is True
    assert receipt["internal_prefix_rows_preserved_exactly"] is True
    assert receipt["internal_unsupported_cells_use_structural_baseline"] is True
    audit = receipt["operation_audit"]
    assert audit["network_access"] is False
    assert all(
        audit[name] == 0
        for name in (
            "training_steps",
            "optimizer_constructions",
            "optimizer_steps",
            "autograd_grad_calls",
            "backward_calls",
            "parent_forward_calls",
            "parent_generate_calls",
            "parent_trace_recaptures",
            "child_forward_calls",
            "child_generate_calls",
            "parameters_pruned",
            "parameters_quantized",
            "parameters_sparsified",
        )
    )
    torch.manual_seed(1061)
    structural = torch.randn(8, 4)
    child_hidden = torch.randn(int(first.target_logits.shape[0]), 4)
    child_evidence = _child_feature_evidence(
        child_hidden,
        first,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    compiled_head = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_evidence,
        first,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("dense-root-sequence-plan"),
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
    )
    assert compiled_head.as_dict()["trie_conditional_evidence_sha256"] == receipt[
        "result_sha256"
    ]


def test_dense_root_hybrid_rejects_resealed_non_total_assignment(
    tmp_path: Path,
) -> None:
    causal, donor_logits, inverse_output_plan, _terminal = _dense_root_hybrid_case(
        tmp_path
    )
    body = json.loads(json.dumps(inverse_output_plan))
    body.pop("plan_sha256")
    donor_id = body["buckets"][0][0]
    body["buckets"][1].append(donor_id)
    resealed = {**body, "plan_sha256": canonical_sha256(body)}
    with pytest.raises(Lfm2InterfaceFrontierError, match="exactly once"):
        compile_lfm2_dense_root_hybrid_causal_evidence(
            causal,
            donor_logits,
            resealed,
        )


def test_dense_root_hybrid_rejects_tensor_and_resealed_semantic_tampering(
    tmp_path: Path,
) -> None:
    causal, donor_logits, inverse_plan, _terminal = _dense_root_hybrid_case(tmp_path)
    result = compile_lfm2_dense_root_hybrid_causal_evidence(
        causal,
        donor_logits,
        inverse_plan,
    )
    tampered_logits = result.target_logits.clone()
    tampered_logits[0, result.as_dict()["nonempty_root_prior_target_ids"][0]] += 1.0
    with pytest.raises(Lfm2InterfaceFrontierError, match="receipt, lineage"):
        Lfm2DenseRootHybridCausalEvidence(
            target_prefix_token_ids=result.target_prefix_token_ids,
            target_logits=tampered_logits,
            target_logit_support_mask=result.target_logit_support_mask,
            receipt=result.as_dict(),
        )

    tampered_body = result.as_dict()
    tampered_body.pop("result_sha256")
    tampered_body["dense_root_prior_is_exact_branch_equivalence"] = True
    resealed_receipt = {
        **tampered_body,
        "result_sha256": canonical_sha256(tampered_body),
    }
    with pytest.raises(Lfm2InterfaceFrontierError, match="receipt, lineage"):
        Lfm2DenseRootHybridCausalEvidence(
            target_prefix_token_ids=result.target_prefix_token_ids,
            target_logits=result.target_logits,
            target_logit_support_mask=result.target_logit_support_mask,
            receipt=resealed_receipt,
        )


def test_dense_root_hybrid_rejects_different_cached_logits(tmp_path: Path) -> None:
    causal, donor_logits, inverse_plan, _terminal = _dense_root_hybrid_case(tmp_path)
    changed_logits = donor_logits.clone()
    changed_logits[0, 0] += 1.0
    with pytest.raises(Lfm2InterfaceFrontierError, match="cached source"):
        compile_lfm2_dense_root_hybrid_causal_evidence(
            causal,
            changed_logits,
            inverse_plan,
        )


@pytest.mark.parametrize(
    ("evidence_rows", "expected_group_formulation"),
    ((8, "dual"), (20, "primal")),
)
def test_causal_residual_bridge_selects_bounded_closed_form_solve(
    tmp_path: Path,
    evidence_rows: int,
    expected_group_formulation: str,
) -> None:
    parent, _blank, bridge, donor, target, _base = _case(tmp_path)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = parent.get_input_embeddings().weight.detach().index_select(
        0,
        torch.tensor(mapping, dtype=torch.long),
    )
    torch.manual_seed(721 + evidence_rows)
    child_hidden = torch.randn(evidence_rows, 16)
    trie, terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=evidence_rows,
    )
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
        solve_device="cpu",
    )
    receipt = result.as_dict()
    assert receipt["solve_formulation"] == "grouped-masked-per-output-ridge"
    assert receipt[f"{expected_group_formulation}_support_mask_group_count"] == 1
    assert receipt["unsupported_cells_included_as_zero_residual_observations"] is False
    assert receipt["unsupported_cells_exactly_preserved_claimed"] is False
    assert receipt["unsupported_logit_drift_max_abs"] >= 0.0
    assert receipt["causal_child_residual_features_used"] is True
    assert receipt["compiled_against_parent_hidden_features"] is False
    assert receipt["diagnostic_metrics_are_not_acceptance_gates"] is True
    assert receipt["operation_audit"]["training_steps"] == 0


def test_protected_special_support_is_removed_before_residual_centering(
    tmp_path: Path,
) -> None:
    parent, _blank, bridge, donor, target, _base = _case(tmp_path)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = parent.get_input_embeddings().weight.detach().index_select(
        0,
        torch.tensor(mapping, dtype=torch.long),
    )
    torch.manual_seed(803)
    child_hidden = torch.randn(4, 16)
    trie, terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=4,
    )
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    eos = int(target.eos_token_id)
    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
        protected_target_ids=(eos,),
    )
    receipt = result.as_dict()
    assert receipt["protected_support_cells_removed_before_centering"] == 4
    assert receipt["effective_supported_logit_cell_count"] == 0
    assert receipt["rows_without_effective_support"] == 4
    torch.testing.assert_close(result.table, structural, rtol=0, atol=0)


def test_masked_output_ridge_records_real_unsupported_logit_drift(
    tmp_path: Path,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[],
        target_merges=[("a", "b")],
    )
    donor_logits = torch.full((2, 8), -9.0)
    donor_logits[0, int(donor.eos_token_id)] = 9.0
    donor_logits[0, 4] = 6.0
    donor_logits[1, 5] = 9.0
    donor_logits[1, 6] = 8.0
    roots = [_causal_root(0), _causal_root(1)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    trie = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=4,
    )
    child_hidden = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    )
    torch.manual_seed(919)
    structural = torch.randn(8, 4)
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=1e-3,
        residual_scale_ppm=1_000_000,
    )
    receipt = result.as_dict()
    assert receipt["each_output_row_fit_only_on_supported_feature_rows"] is True
    assert receipt["unsupported_cells_included_as_zero_residual_observations"] is False
    assert receipt["unsupported_cells_exactly_preserved_claimed"] is False
    assert receipt["unsupported_logit_drift_max_abs"] > 0.0


def test_sealed_background_causal_head_preserves_every_unsupported_logit_after_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[],
        target_merges=[("a", "b")],
    )
    donor_logits = torch.full((2, 8), -9.0)
    donor_logits[0, int(donor.eos_token_id)] = 9.0
    donor_logits[0, 4] = 6.0
    donor_logits[1, 5] = 9.0
    donor_logits[1, 6] = 8.0
    roots = [_causal_root(0), _causal_root(1)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    trie = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=4,
    )
    # Orthogonal evidence axes leave an exactly representable nullspace for
    # per-context corrections; the serialization check below exercises the
    # final target-dtype seal, not merely the pre-projection solve.
    child_hidden = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    torch.manual_seed(929)
    structural = torch.randn(8, 4).to(dtype=torch.bfloat16)
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=1e-3,
        residual_scale_ppm=1_000_000,
        row_norm_calibration_ppm=250_000,
        unsupported_logit_policy="sealed-background-nullspace",
    )
    checkpoint = tmp_path / "sealed-background-head.pt"
    torch.save(
        {"table": result.table, "receipt": result.as_dict()},
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = interface_module.Lfm2CausalResidualTokenBridge(
        table=payload["table"],
        receipt=payload["receipt"],
    )

    receipt = reloaded.as_dict()
    support = trie.target_logit_support_mask
    baseline_logits = child_hidden.matmul(structural.float().t())
    reloaded_logits = child_hidden.matmul(reloaded.table.float().t())
    unsupported_output_rows = ~support.any(dim=0)
    assert torch.equal(
        reloaded_logits[~support],
        baseline_logits[~support],
    )
    assert torch.equal(
        reloaded.table[unsupported_output_rows],
        structural[unsupported_output_rows],
    )
    assert receipt["unsupported_logit_policy"] == (
        "sealed-background-nullspace"
    )
    assert receipt["unsupported_cells_exactly_preserved_claimed"] is True
    assert receipt["unsupported_logit_drift_rms"] == 0.0
    assert receipt["unsupported_logit_drift_max_abs"] == 0.0
    assert receipt["vocabulary_row_coverage_complete"] is True
    assert receipt["all_vocabulary_rows_classified"] is True
    assert receipt["vocabulary_row_count"] == int(structural.shape[0])
    assert (
        receipt["contextually_supported_output_row_count"]
        + receipt["fully_unsupported_output_row_count"]
        == receipt["vocabulary_row_count"]
    )
    assert receipt["fully_unsupported_output_rows_equal_structural_baseline"] is True
    assert receipt["contextual_correction_output_row_count"] > 0
    assert receipt["background_constraint_sha256"]
    assert receipt["row_norm_calibration_formulation"] == (
        "background-preserving-delta-contraction"
    )
    assert reloaded.table.requires_grad is False
    audit = receipt["operation_audit"]
    assert audit["training_steps"] == 0
    assert audit["optimizer_constructions"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["autograd_grad_calls"] == 0
    assert audit["backward_calls"] == 0
    assert receipt["diagnostic_metrics_are_not_acceptance_gates"] is True
    assert receipt["diagnostic_metrics_used_as_compilation_gate"] is False
    assert receipt["manual_behavior_scores_used_as_compilation_gate"] is False
    assert receipt["behavior_retention_goal_is_not_a_structural_cutoff"] is True

    # The same row can round differently when projected in a support group
    # versus as one column of the complete vocabulary. Simulate that late
    # platform-dependent violation for a candidate which the grouped seal
    # retained, then prove the final full-table seal reverts it.
    corrected_with_background = (
        (reloaded.table != structural).any(dim=1)
        & (~support).any(dim=0)
    )
    late_target_id = int(
        torch.nonzero(
            corrected_with_background,
            as_tuple=False,
        )[0].item()
    )
    late_evidence_row = int(
        torch.nonzero(
            ~support[:, late_target_id],
            as_tuple=False,
        )[0].item()
    )
    original_matmul = torch.Tensor.matmul
    injected_full_projection_calls = 0

    def _late_full_projection_rounding(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal injected_full_projection_calls
        projected = original_matmul(left, right)
        is_full_vocabulary_projection = tuple(right.shape) == (
            int(structural.shape[1]),
            int(structural.shape[0]),
        )
        selected_row_changed = (
            is_full_vocabulary_projection
            and not torch.equal(
                right[:, late_target_id],
                structural.float()[late_target_id],
            )
        )
        if selected_row_changed:
            projected = projected.clone()
            projected[late_evidence_row, late_target_id] += 1.0
            injected_full_projection_calls += 1
        return projected

    monkeypatch.setattr(
        torch.Tensor,
        "matmul",
        _late_full_projection_rounding,
    )
    late_sealed = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=_sha("late-full-projection-sequence-plan"),
        ridge=1e-3,
        residual_scale_ppm=1_000_000,
        row_norm_calibration_ppm=250_000,
        unsupported_logit_policy="sealed-background-nullspace",
    )
    late_receipt = late_sealed.as_dict()
    late_logits = original_matmul(
        child_hidden,
        late_sealed.table.float().t(),
    )
    assert injected_full_projection_calls == 1
    assert late_receipt[
        "target_dtype_strict_cell_accounting_projection"
    ] == "union-of-per-group-and-final-full-vocabulary-matmul"
    assert late_receipt[
        "target_dtype_final_full_projection_late_failure_output_row_count"
    ] >= 1
    assert late_receipt[
        "target_dtype_strict_cell_failure_output_row_count"
    ] == (
        late_receipt[
            "target_dtype_group_strict_cell_failure_output_row_count"
        ]
        + late_receipt[
            "target_dtype_final_full_projection_late_failure_output_row_count"
        ]
    )
    assert torch.equal(
        late_sealed.table[late_target_id],
        structural[late_target_id],
    )
    assert torch.equal(late_logits[~support], baseline_logits[~support])
    assert late_receipt["unsupported_logit_drift_max_abs"] == 0.0


def test_adaptive_background_seal_preserves_fp32_corrections_after_strict_reversion(
    tmp_path: Path,
) -> None:
    bridge, donor, target = _bpe_bridge_case(
        tmp_path,
        donor_merges=[],
        target_merges=[("a", "b")],
    )
    donor_logits = torch.full((2, 8), -9.0)
    donor_logits[0, int(donor.eos_token_id)] = 9.0
    donor_logits[0, 4] = 6.0
    donor_logits[1, 5] = 9.0
    donor_logits[1, 6] = 8.0
    roots = [_causal_root(0), _causal_root(1)]
    terminal = _terminal_evidence(donor_logits, roots, bridge)
    trie = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        donor,
        target,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=4,
    )
    torch.manual_seed(20260728)
    child_hidden = torch.randn(2, 64, dtype=torch.float32)
    structural = torch.randn(8, 64, dtype=torch.float32)
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    common = {
        "parent_terminal_evidence": terminal,
        "sequence_plan_sha256": _sha("adaptive-fp32-sequence-plan"),
        "ridge": 1e-3,
        "residual_scale_ppm": 1_000_000,
        "row_norm_calibration_ppm": 250_000,
    }
    strict = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        unsupported_logit_policy=(
            interface_module.LFM2_UNSUPPORTED_LOGIT_POLICY_SEALED_BACKGROUND_NULLSPACE
        ),
        **common,
    )
    adaptive = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_receipt,
        trie,
        unsupported_logit_policy=(
            interface_module.LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
        ),
        **common,
    )
    strict_receipt = strict.as_dict()
    adaptive_receipt = adaptive.as_dict()
    supported_rows = trie.target_logit_support_mask.any(dim=0)
    fully_unsupported_rows = ~supported_rows

    assert strict_receipt["target_dtype_nonzero_candidate_output_row_count"] > 0
    assert strict_receipt["target_dtype_strict_cell_pass_output_row_count"] == 0
    assert strict_receipt["target_dtype_background_reverted_output_row_count"] == (
        strict_receipt["target_dtype_nonzero_candidate_output_row_count"]
    )
    assert strict_receipt["contextual_correction_output_row_count"] == 0
    torch.testing.assert_close(strict.table, structural, rtol=0, atol=0)

    candidate_count = adaptive_receipt[
        "target_dtype_nonzero_candidate_output_row_count"
    ]
    assert candidate_count == strict_receipt[
        "target_dtype_nonzero_candidate_output_row_count"
    ]
    assert adaptive_receipt["target_dtype_strict_cell_pass_output_row_count"] == 0
    assert adaptive_receipt[
        "target_dtype_adaptive_fallback_output_row_count"
    ] == candidate_count
    assert adaptive_receipt["target_dtype_background_reverted_output_row_count"] == 0
    assert adaptive_receipt["contextual_correction_output_row_count"] == candidate_count
    assert adaptive_receipt["adaptive_background_fallback_used"] is True
    assert (
        adaptive_receipt["adaptive_background_retained_every_nonzero_candidate"]
        is True
    )
    assert adaptive_receipt["unsupported_cells_exactly_preserved_claimed"] is False
    assert (
        adaptive_receipt["adaptive_fallback_suppresses_global_exact_cell_claim"]
        is True
    )
    assert adaptive_receipt["supported_row_off_support_logit_drift_max_abs"] > 0.0
    assert adaptive_receipt["fully_unsupported_output_row_logit_drift_max_abs"] == 0.0
    assert adaptive_receipt["relative_centered_error_after"] < adaptive_receipt[
        "relative_centered_error_before"
    ]
    assert adaptive_receipt["relative_centered_error_after"] < strict_receipt[
        "relative_centered_error_after"
    ]
    assert torch.equal(
        adaptive.table[fully_unsupported_rows],
        structural[fully_unsupported_rows],
    )
    assert bool((adaptive.table[supported_rows] != structural[supported_rows]).any())


def test_causal_residual_rejects_resealed_child_prefix_ancestry(
    tmp_path: Path,
) -> None:
    parent, _blank, bridge, donor, target, _base = _case(tmp_path)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = parent.get_input_embeddings().weight.detach().index_select(
        0,
        torch.tensor(mapping, dtype=torch.long),
    )
    child_hidden = torch.randn(3, 16)
    trie, terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=3,
    )
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=interface_module._tensor_sha256(structural),
    )
    tampered_body = child_receipt.as_dict()
    tampered_body.pop("evidence_sha256")
    tampered_body["target_prefix_rows_sha256"] = _sha("context-free-prefixes")
    tampered = interface_module.Lfm2ChildCausalResidualEvidence(
        hidden_rows=child_hidden,
        receipt={
            **tampered_body,
            "evidence_sha256": canonical_sha256(tampered_body),
        },
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="prefix order"):
        compile_lfm2_causal_residual_token_bridge(
            structural,
            tampered,
            trie,
            parent_terminal_evidence=terminal,
            sequence_plan_sha256=_sha("sequence-plan"),
            ridge=1e-2,
            residual_scale_ppm=1_000_000,
        )


def test_tied_causal_bridge_materializes_as_explicit_perturbative_ablation(
    tmp_path: Path,
) -> None:
    parent, control_blank, bridge, donor, target, base_parameters = _case(tmp_path)
    control_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-structural-control",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    control = fold_lfm2_interface_candidate(
        parent,
        control_blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=control_blank.as_dict(),
        recipe=control_recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    tied_structural = control.model.get_input_embeddings().weight.detach().clone()
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    torch.manual_seed(827)
    child_hidden = torch.randn(8, 16)
    trie, terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=8,
    )
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=control.as_dict()["input_table_sha256"],
        body_sha256=control.as_dict()["body_tensor_inventory_sha256"],
    )
    causal = compile_lfm2_causal_residual_token_bridge(
        tied_structural,
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=control.as_dict()["sequence_plan_sha256"],
        ridge=1e-2,
        residual_scale_ppm=100_000,
        solve_device="cpu",
    )
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="tied-causal-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-causal",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge-tied",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="differs"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "changed-context"),
            causal_residual_token_bridge=causal,
            precompiled_inverse_output_plan=inverse_plan,
        )
    altered_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-causal-altered-input",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge-tied",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="differs"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=altered_recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "a"),
            causal_residual_token_bridge=causal,
            precompiled_inverse_output_plan=inverse_plan,
        )
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
        causal_residual_token_bridge=causal,
        precompiled_inverse_output_plan=inverse_plan,
    )
    receipt = folded.as_dict()
    assert folded.model.get_input_embeddings().weight is (
        folded.model.get_output_embeddings().weight
    )
    torch.testing.assert_close(
        folded.model.get_input_embeddings().weight,
        causal.table,
        rtol=0,
        atol=0,
    )
    assert receipt["causal_feature_coordinate_mode"] == (
        "tied-shared-table-perturbative-ablation"
    )
    assert receipt["one_shot_tied_causal_self_consistency_claimed"] is False
    assert receipt["postinstall_tied_feature_drift_audit_required"] is True


def test_untied_causal_bridge_preserves_feature_replay_input_coordinate(
    tmp_path: Path,
) -> None:
    parent, control_blank, bridge, donor, target, base_parameters = _case(tmp_path)
    control_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-structural-control",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    control = fold_lfm2_interface_candidate(
        parent,
        control_blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=control_blank.as_dict(),
        recipe=control_recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    torch.manual_seed(911)
    child_hidden = torch.randn(8, 16)
    trie, terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=8,
    )
    child_receipt = _child_feature_evidence(
        child_hidden,
        trie,
        input_sha256=control.as_dict()["input_table_sha256"],
        body_sha256=control.as_dict()["body_tensor_inventory_sha256"],
    )
    causal = compile_lfm2_causal_residual_token_bridge(
        control.model.get_input_embeddings().weight.detach().clone(),
        child_receipt,
        trie,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=control.as_dict()["sequence_plan_sha256"],
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
        solve_device="cpu",
    )
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="untied-causal-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="untied-causal",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    expected_parameters = base_parameters + 16 * 16
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected_parameters,
        contexts=("", "a"),
        causal_residual_token_bridge=causal,
        precompiled_inverse_output_plan=inverse_plan,
    )
    receipt = folded.as_dict()
    assert folded.model.get_input_embeddings().weight is not (
        folded.model.get_output_embeddings().weight
    )
    torch.testing.assert_close(
        folded.model.get_input_embeddings().weight,
        control.model.get_input_embeddings().weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        folded.model.get_output_embeddings().weight,
        causal.table,
        rtol=0,
        atol=0,
    )
    assert receipt["input_table_sha256"] == control.as_dict()["input_table_sha256"]
    assert receipt["body_tensor_inventory_sha256"] == control.as_dict()[
        "body_tensor_inventory_sha256"
    ]
    assert receipt["causal_feature_coordinate_mode"] == (
        "untied-output-exact-causal-coordinate"
    )
    assert receipt["postinstall_tied_feature_drift_audit_required"] is False


def test_source_causal_child_replay_is_full_prefix_and_nonmutating(
    tmp_path: Path,
) -> None:
    model, _blank, bridge, donor, target, _base = _case(tmp_path)
    before = fingerprint_model_state(model)
    trie, _terminal = _typed_causal_evidence(
        bridge,
        donor,
        target,
        evidence_rows=3,
    )
    input_sha256 = interface_module._tensor_sha256(
        model.get_input_embeddings().weight
    )
    body_sha256 = canonical_sha256(
        list(
            interface_module._body_records(
                model,
                interface_storage_ids=interface_module._interface_storage_ids(model),
            )
        )
    )
    evidence = interface_module.collect_lfm2_child_causal_residual_evidence(
        model,
        trie,
        expected_input_table_sha256=input_sha256,
        expected_body_inventory_sha256=body_sha256,
        device="cpu",
        evidence_role="unit-test-causal-replay",
    )
    features = evidence.hidden_rows
    receipt = evidence.as_dict()
    assert tuple(features.shape) == (3, 16)
    assert bool(torch.isfinite(features).all()) is True
    assert fingerprint_model_state(model) == before
    assert receipt["full_prefix_use_cache_false"] is True
    assert receipt["operation_audit"]["child_forward_calls"] == 3
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["parent_forward_calls"] == 0
    assert receipt["batch_size"] == 1
    assert receipt["input_and_body_coordinate_preserved"] is True
    with pytest.raises(Lfm2InterfaceFrontierError, match="coordinate differs"):
        interface_module.collect_lfm2_child_causal_residual_evidence(
            model,
            trie,
            expected_input_table_sha256=_sha("wrong-input-coordinate"),
            expected_body_inventory_sha256=body_sha256,
            device="cpu",
            evidence_role="unit-test-wrong-coordinate",
        )


def test_asymmetric_candidate_fails_closed_above_target_size(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="target-size"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "a"),
        )


def test_exact4x_untied_interface_budget_rebalances_only_ffn_width() -> None:
    recipe = Lfm2InterfaceRecipe(
        candidate_id="exact4x-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="cached-logit-residual",
    )
    assert base_tied_target_budget_for_recipe(
        maximum_total_target_unique_parameters=57_423_296,
        target_vocabulary_size=4_096,
        hidden_size=776,
        recipe=recipe,
    ) == 54_244_800
    plan = plan_lfm2_dense_untied_target_rebalance(
        parent_unique_parameters=229_693_184,
        base_tied_target_unique_parameters=57_405_368,
        target_vocabulary_size=4_096,
        hidden_size=776,
        layer_count=14,
        current_intermediate_size=797,
        target_compaction_denominator=4,
    )
    assert plan["target_unique_parameter_ceiling"] == 57_423_296
    assert plan["added_dense_untied_output_parameters"] == 3_178_496
    assert plan["intermediate_size_reduction"] == 97
    assert plan["target_intermediate_size"] == 700
    assert plan["removed_body_parameters"] == 3_161_424
    assert plan["target_unique_parameters"] == 57_422_440
    assert plan["parameter_ceiling_spare"] == 856
    assert plan["residual_width_preserved"] is True
    assert plan["attention_geometry_preserved"] is True
