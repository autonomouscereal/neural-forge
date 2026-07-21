from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_long_horizon_terminal_refold as refold  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionResidualResult,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    fit_tied_causal_interface,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)


class _Tokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = None
        self.all_special_ids = [0, 1, 2]

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


def _tokenizer_files(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )


def _model(
    *,
    vocabulary: int,
    hidden: int,
    intermediate: int,
    seed: int,
    tied: bool,
    bfloat16: bool,
) -> Any:
    heads = 2 if hidden >= 8 else 1
    config = transformers.Lfm2Config(
        vocab_size=vocabulary,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=tied,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    if bfloat16:
        model.to(dtype=torch.bfloat16)
    model.eval()
    return model


def _fixture(
    tmp_path: Path,
    *,
    donor_vocabulary: int = 31,
    target_vocabulary: int = 31,
    include_unmapped_donor_row: bool = False,
) -> dict[str, Any]:
    parent = _model(
        vocabulary=donor_vocabulary,
        hidden=8,
        intermediate=12,
        seed=1701,
        tied=False,
        bfloat16=False,
    )
    batches = []
    for index in range(4):
        terminal = (
            donor_vocabulary - 1
            if include_unmapped_donor_row and index == 0
            else 4 + index
        )
        batches.append(
            {
                "input_ids": torch.tensor(
                    [[1, 2, 3, terminal]],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "forced_decode_ids": torch.tensor(
                    [[5 + index, 6 + index]],
                    dtype=torch.long,
                ),
            }
        )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "cache",
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=4,
        ),
        input_device="cpu",
    )
    parent_final_norm = refold.Lfm2SealedParentFinalRmsNorm(
        cache_key=cache.cache_key,
        parent_fingerprint=cache.parent_fingerprint,
        layer_topology_fingerprint=cache.layer_topology_fingerprint,
        weight=parent.model.embedding_norm.weight,
        epsilon=float(parent.model.embedding_norm.variance_epsilon),
    )

    source_table = (
        parent.get_output_embeddings()
        .weight[:target_vocabulary]
        .detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .clone()
    )
    generator = torch.Generator().manual_seed(1703)
    interface = fit_tied_causal_interface(
        source_table,
        torch.randn(48, 8, generator=generator),
        target_width=4,
        validation_producer_rows=torch.randn(24, 8, generator=generator),
        minimum_validation_top1_agreement=0.0,
        metric_chunk_rows=8,
    )
    child = _model(
        vocabulary=target_vocabulary,
        hidden=4,
        intermediate=8,
        seed=1705,
        tied=True,
        bfloat16=True,
    )
    assert child.get_input_embeddings().weight is child.get_output_embeddings().weight
    with torch.no_grad():
        child.get_input_embeddings().weight.copy_(
            interface.target_weight.to(torch.bfloat16)
        )

    donor_surfaces = [f"<{index}>" for index in range(donor_vocabulary)]
    target_surfaces = donor_surfaces[:target_vocabulary]
    tokenizer_root = tmp_path / "target-tokenizer"
    _tokenizer_files(tokenizer_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        _Tokenizer(donor_surfaces),
        _Tokenizer(target_surfaces),
        tokenizer_root,
        donor_row_capacity=donor_vocabulary,
        target_row_capacity=target_vocabulary,
    )
    return {
        "cache": cache,
        "bridge": bridge,
        "child": child,
        "interface": interface,
        "source_table": source_table,
        "parent_final_norm": parent_final_norm,
    }


def _accepted_terminal(seed, policy) -> Lfm2CpcfTerminalActionResidualResult:
    incumbent = seed.down_weight.detach().clone()
    selected = incumbent.clone()
    selected.reshape(-1)[0] = selected.reshape(-1)[0] + torch.tensor(
        0.125,
        dtype=torch.bfloat16,
    )
    assert not torch.equal(selected, incumbent)
    body = {
        "policy_sha256": policy.authentication_sha256,
        "selected_candidate_id": "test-accepted-candidate",
        "accepted": True,
        "output_state_sha256": {
            "incumbent": refold._tensor_sha256(incumbent),
            "selected": refold._tensor_sha256(selected),
            "published": refold._tensor_sha256(selected),
        },
    }
    seal = refold._canonical_sha256(body)
    return Lfm2CpcfTerminalActionResidualResult(
        policy=policy,
        incumbent_down_weight=incumbent,
        selected_down_weight=selected,
        published_down_weight=selected,
        selected_candidate_id="test-accepted-candidate",
        accepted=True,
        report={**body, "result_sha256": seal},
        result_sha256=seal,
    )


def test_replays_each_complete_trajectory_once_and_installs_only_accepted_w2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    child = values["child"]
    writer = child.model.layers[-1].feed_forward.w2.weight
    writer_id = id(writer)
    writer_ptr = int(writer.data_ptr())
    before = writer.detach().clone()

    def forbidden_model_forward(*_args, **_kwargs):
        raise AssertionError("model.forward must not be called")

    child.forward = forbidden_model_forward

    def accepted_compiler(**kwargs):
        return _accepted_terminal(kwargs["seed"], kwargs["policy"])

    monkeypatch.setattr(
        refold,
        "compile_lfm2_cpcf_terminal_action_residual",
        accepted_compiler,
    )
    result = refold.compile_install_lfm2_long_horizon_terminal_refold(
        cache=values["cache"],
        bridge=values["bridge"],
        target_model=child,
        interface=values["interface"],
        source_selected_tied_table=values["source_table"],
        parent_final_norm=values["parent_final_norm"],
    )

    assert result.accepted is True
    assert id(writer) == writer_id
    assert int(writer.data_ptr()) == writer_ptr
    assert not torch.equal(writer, before)
    report = result.as_dict()
    audit = report["operation_audit"]
    assert audit["parent_model_forward_calls"] == 0
    assert audit["child_model_forward_calls"] == 0
    assert audit["child_embedding_seed_lookups"] == 4
    assert audit["child_macrocell_replay_calls"] == 8
    assert audit["expected_child_macrocell_replay_calls"] == 8
    assert audit["training_steps"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["parameters_pruned"] == 0
    assert audit["parameters_quantized"] == 0
    assert report["split_rows"] == {
        "fit": 3,
        "fit_check": 3,
        "selection": 3,
        "validation": 3,
    }
    buckets = report["horizon_metrics"]["buckets"]
    assert buckets["0-7"]["available"] is True
    assert buckets["0-7"]["rows"] == 6
    assert buckets["0-7"]["incumbent"] is not None
    assert buckets["0-7"]["selected"] is not None
    assert buckets["8-31"] == {
        "available": False,
        "rows": 0,
        "incumbent": None,
        "selected": None,
    }
    assert buckets["32+"]["available"] is False
    assert len(report["trajectory_replay_receipts"]) == 4
    assert all(
        value["macrocell_replay_calls"] == 2
        for value in report["trajectory_replay_receipts"]
    )


def test_real_closed_form_selection_never_mutates_w2_when_rejected(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    child = values["child"]
    writer = child.model.layers[-1].feed_forward.w2.weight
    identity = (id(writer), int(writer.data_ptr()))
    before = writer.detach().clone()
    result = refold.compile_install_lfm2_long_horizon_terminal_refold(
        cache=values["cache"],
        bridge=values["bridge"],
        target_model=child,
        interface=values["interface"],
        source_selected_tied_table=values["source_table"],
        parent_final_norm=values["parent_final_norm"],
    )

    assert (id(writer), int(writer.data_ptr())) == identity
    assert result.terminal_result.verify_authentication()
    if result.accepted:
        assert not torch.equal(writer, before)
    else:
        assert torch.equal(writer, before)
        assert result.as_dict()["installation"]["installed"] is False
    assert result.as_dict()["operation_audit"]["training_steps"] == 0


def test_fails_closed_when_cached_donor_token_has_no_unique_target_row(
    tmp_path: Path,
) -> None:
    values = _fixture(
        tmp_path,
        donor_vocabulary=32,
        target_vocabulary=31,
        include_unmapped_donor_row=True,
    )
    writer = values["child"].model.layers[-1].feed_forward.w2.weight
    before = writer.detach().clone()
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="does not cover every cached prompt row",
    ):
        refold.compile_install_lfm2_long_horizon_terminal_refold(
            cache=values["cache"],
            bridge=values["bridge"],
            target_model=values["child"],
            interface=values["interface"],
            source_selected_tied_table=values["source_table"],
            parent_final_norm=values["parent_final_norm"],
        )
    assert torch.equal(writer, before)


def test_parent_norm_and_whole_trajectory_split_seals_fail_closed(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    seal = values["parent_final_norm"]
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="authentication is invalid",
    ):
        refold.Lfm2SealedParentFinalRmsNorm(
            cache_key=seal.cache_key,
            parent_fingerprint=seal.parent_fingerprint,
            layer_topology_fingerprint=seal.layer_topology_fingerprint,
            weight=seal.weight,
            epsilon=seal.epsilon * 2.0,
            authentication_sha256=seal.authentication_sha256,
        )
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="each need at least one complete trajectory",
    ):
        refold.Lfm2LongHorizonTrajectorySplitPlan(
            cache_key=values["cache"].cache_key,
            assignments=("fit", "fit_check", "selection", "selection"),
        )
