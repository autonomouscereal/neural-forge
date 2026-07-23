from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_bridge_only_control import (  # noqa: E402
    Lfm2BridgeOnlyControlError,
    fold_lfm2_bridge_only_control,
)
from neural_forge.hf_lfm2_cpcf_blank_target import (  # noqa: E402
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokenizer(path: Path, vocabulary: dict[str, int]):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
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


def _parent():
    config = transformers.Lfm2Config(
        vocab_size=32,
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
    torch.manual_seed(8128)
    return transformers.Lfm2ForCausalLM(config).eval()


def _case(tmp_path: Path):
    donor_vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        **{f"t{index}": index for index in range(4, 32)},
    }
    target_vocab = {
        token: index for index, token in enumerate(list(donor_vocab)[:16])
    }
    donor = _tokenizer(tmp_path / "donor", donor_vocab)
    target_path = tmp_path / "target"
    target = _tokenizer(target_path, target_vocab)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=32,
        target_row_capacity=16,
    )
    parent = _parent()
    body_parameters = (
        sum(parameter.numel() for parameter in parent.parameters()) - 32 * 16
    )
    target_parameters = body_parameters + 16 * 16
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="bridge-only-child",
        target_unique_parameters=target_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=7,
        device="cpu",
        dtype=torch.float32,
    )
    return parent, blank, bridge


def test_bridge_only_control_preserves_exact_body_and_restricted_logits(
    tmp_path: Path,
) -> None:
    parent, blank, bridge = _case(tmp_path)
    parent_before = fingerprint_model_state(parent)
    result = fold_lfm2_bridge_only_control(
        parent,
        blank.model,
        bridge=bridge,
        blank_target_receipt=blank.as_dict(),
    )
    receipt = result.as_dict()

    assert receipt["exact_parent_body_depth_width_preserved"] is True
    assert receipt["parameter_reduction_is_vocabulary_rows_only"] is True
    assert receipt["parent_unique_parameters"] > receipt["target_unique_parameters"]
    assert receipt["operation_audit"]["training_steps"] == 0
    assert fingerprint_model_state(parent) == parent_before
    mapping = torch.tensor(
        bridge.transport_profile.target_to_donor_row_indices,
        dtype=torch.long,
    )
    assert torch.equal(
        result.model.get_input_embeddings().weight,
        parent.get_input_embeddings().weight.index_select(0, mapping),
    )

    # The exact body plus row-selected tied table is functionally identical to
    # restricting the parent's logits to mapped donor rows for mapped inputs.
    target_ids = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    donor_ids = mapping.index_select(0, target_ids.reshape(-1)).view_as(target_ids)
    with torch.no_grad():
        parent_logits = parent(donor_ids).logits.index_select(-1, mapping)
        child_logits = result.model(target_ids).logits
    assert torch.equal(parent_logits, child_logits)


def test_bridge_only_control_rejects_nonblank_target(tmp_path: Path) -> None:
    parent, blank, bridge = _case(tmp_path)
    with torch.no_grad():
        next(blank.model.parameters()).fill_(1)
    with pytest.raises(Lfm2BridgeOnlyControlError, match="nonzero"):
        fold_lfm2_bridge_only_control(
            parent,
            blank.model,
            bridge=bridge,
            blank_target_receipt=blank.as_dict(),
        )


def test_bridge_only_control_derives_legacy_lfm2_head_dimension(
    tmp_path: Path,
) -> None:
    parent, blank, bridge = _case(tmp_path)
    delattr(parent.config, "head_dim")
    delattr(blank.model.config, "head_dim")
    result = fold_lfm2_bridge_only_control(
        parent,
        blank.model,
        bridge=bridge,
        blank_target_receipt=blank.as_dict(),
    )
    assert result.as_dict()["target_topology"]["head_dim"] == 8
