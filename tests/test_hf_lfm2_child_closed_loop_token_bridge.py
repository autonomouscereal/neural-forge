from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_lfm2_child_closed_loop_refinement import (  # noqa: E402
    Lfm2ChildClosedLoopRefinementError,
    pair_lfm2_parent_child_replay,
    project_lfm2_paired_parent_logits,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    capture_lfm2_child_replay,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


class _Tokenizer:
    def __init__(
        self,
        surfaces: list[str],
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        self.surfaces = list(surfaces)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = None
        self.all_special_ids = sorted(
            value
            for value in (bos_token_id, eos_token_id, pad_token_id)
            if value is not None
        )

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{token_id}-{surface}": token_id
            for token_id, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        matches = [
            token_id
            for token_id, surface in enumerate(self.surfaces)
            if surface == text
        ]
        if len(matches) != 1:
            raise KeyError(text)
        return matches


def _target_files(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )


def _model(*, vocab_size: int, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


@pytest.fixture()
def bridged_replay(tmp_path: Path):
    donor = _Tokenizer(
        ["<pad>", "<bos>", "A", "B", "<eos>", "C"],
        bos_token_id=1,
        eos_token_id=4,
        pad_token_id=0,
    )
    target = _Tokenizer(
        ["<bos>", "B", "<eos>", "<pad>"],
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=3,
    )
    target_root = tmp_path / "target-tokenizer"
    _target_files(target_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_root,
        donor_row_capacity=8,
        target_row_capacity=4,
    )
    records = (
        Lfm2TargetTokenizedRecord(
            record_id="fit-0",
            split="fit",
            input_ids=torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[1]], dtype=torch.long),
        ),
        Lfm2TargetTokenizedRecord(
            record_id="heldout-0",
            split="heldout",
            input_ids=torch.tensor([[0, 1, 0, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[2]], dtype=torch.long),
        ),
    )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=1,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    parent_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        _model(vocab_size=8, seed=1101),
        adapter.parent_cache_batches(),
        config=config,
        input_device="cpu",
    )
    child_replay = capture_lfm2_child_replay(
        _model(vocab_size=4, seed=1102),
        adapter.child_replay_batches(),
        selected_layer_indices=[1],
        config=config,
        input_device="cpu",
    )
    return adapter, parent_cache, child_replay


def test_pairs_distinct_donor_and_target_ids_through_authenticated_bridge(
    bridged_replay,
) -> None:
    adapter, parent_cache, child_replay = bridged_replay

    assert parent_cache.corpus_fingerprint != child_replay.corpus_fingerprint
    with pytest.raises(
        Lfm2ChildClosedLoopRefinementError,
        match="corpus identities differ",
    ):
        pair_lfm2_parent_child_replay(parent_cache, child_replay)

    paired = pair_lfm2_parent_child_replay(
        parent_cache,
        child_replay,
        tokenizer_bridge_trace_adapter=adapter,
    )

    assert paired.tokenizer_bridge_trace_adapter is adapter
    assert paired.tokenizer_bridge_evidence_sha256 is not None
    assert len(paired) == adapter.record_count == 2
    for index, aligned in enumerate(adapter.records):
        batch = paired.load_batch(index)
        assert batch.parent_input_sha256 == aligned.donor_cache_input_sha256
        assert batch.input_sha256 == aligned.target_cache_input_sha256
        assert batch.parent_input_sha256 != batch.input_sha256
        assert batch.tokenizer_bridge_record_id == aligned.record_id
        assert batch.tokenizer_bridge_input_evidence is not None
        assert (
            batch.tokenizer_bridge_input_evidence.numeric_token_id_equality_required
            is False
        )
        assert not torch.equal(batch.parent.input_ids, batch.child.input_ids)


def test_projects_sealed_parent_logits_into_child_vocabulary_without_mutation(
    bridged_replay,
) -> None:
    adapter, parent_cache, child_replay = bridged_replay
    paired = pair_lfm2_parent_child_replay(
        parent_cache,
        child_replay,
        tokenizer_bridge_trace_adapter=adapter,
    )
    mapping = torch.tensor(adapter.mapping, dtype=torch.long)

    for batch in paired.batches:
        for phase in batch.phase_order:
            donor_logits = batch.parent.tensors[f"{phase}.logits"]
            before = donor_logits.clone()
            projection = project_lfm2_paired_parent_logits(
                paired,
                batch_index=batch.index,
                phase=phase,
                top_k=3,
            )
            expected = donor_logits.index_select(-1, mapping)
            assert torch.equal(projection.target_logits, expected)
            assert tuple(projection.target_logits.shape) == tuple(
                batch.child.tensors[f"{phase}.logits"].shape
            )
            assert projection.record_id == adapter.records[batch.index].record_id
            assert projection.cache_key == parent_cache.cache_key
            assert projection.cache_batch_index == batch.index
            assert torch.equal(donor_logits, before)
