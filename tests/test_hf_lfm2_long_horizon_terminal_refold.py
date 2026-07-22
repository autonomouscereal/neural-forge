from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402

import neural_forge.hf_lfm2_long_horizon_terminal_refold as refold  # noqa: E402
import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionResidualResult,
    compile_lfm2_cpcf_terminal_action_residual as _compile_terminal_action_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
    fit_tied_causal_interface,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)
from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    compile_tied_residual_congruence_candidate,
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


class _GreedyTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = list(pieces)
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.all_special_ids: list[int] = []

    def __len__(self) -> int:
        return len(self.pieces)

    def get_vocab(self) -> dict[str, int]:
        return {piece: index for index, piece in enumerate(self.pieces)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[token_id]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        ranked = sorted(
            enumerate(self.pieces),
            key=lambda item: (-len(item[1]), item[0]),
        )
        result = []
        position = 0
        while position < len(text):
            for token_id, piece in ranked:
                if piece and text.startswith(piece, position):
                    result.append(token_id)
                    position += len(piece)
                    break
            else:
                raise KeyError(text[position:])
        return result


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
    parent_tied: bool = False,
    parent_bfloat16: bool = False,
) -> dict[str, Any]:
    parent = _model(
        vocabulary=donor_vocabulary,
        hidden=8,
        intermediate=12,
        seed=1701,
        tied=parent_tied,
        bfloat16=parent_bfloat16,
    )
    if parent_bfloat16:
        parent.config.dtype = torch.bfloat16
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
    source_lineage = refold.build_lfm2_long_horizon_terminal_source_lineage(
        cache=cache,
        bridge=bridge,
        parent_final_norm=parent_final_norm,
        interface=interface,
        source_selected_tied_table=source_table,
    )
    return {
        "parent": parent,
        "cache": cache,
        "bridge": bridge,
        "child": child,
        "interface": interface,
        "source_table": source_table,
        "source_lineage": source_lineage,
        "parent_final_norm": parent_final_norm,
    }


def test_interface_report_accepts_exact_tied_residual_congruence_and_rejects_source_tamper():
    generator = torch.Generator().manual_seed(1_711)
    source_table = torch.randn(
        (13, 7),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    base = fit_tied_causal_interface(
        source_table,
        torch.randn((31, 7), generator=generator, dtype=torch.float64),
        target_width=4,
        validation_producer_rows=torch.randn(
            (17, 7),
            generator=generator,
            dtype=torch.float64,
        ),
        metric_chunk_rows=7,
    )
    candidate = compile_tied_residual_congruence_candidate(
        base,
        source_table,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )

    verified = refold._verify_interface_report(candidate.interface, source_table)

    assert verified == dict(candidate.interface.report)
    assert (
        verified["mapped_parent_tied_table_sha256"]
        == (candidate.report["mapped_parent_tied_table_sha256"])
    )

    tampered = source_table.clone()
    tampered[0, 0] += 1.0
    assert not torch.equal(tampered, source_table)
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="differ from their authenticated lineage",
    ):
        refold._verify_interface_report(candidate.interface, tampered)


def test_long_horizon_source_lineage_requires_exact_anchor_transport_witness(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    source_interface = values["interface"]
    source_target = source_interface.target_weight.to(torch.float64)
    base = torch.zeros_like(source_target)
    residual = source_target.clone()
    source_body = {
        "schema": TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
        "target_weight_sha256": postfold._foundation_lineage_tensor_sha256(
            source_target
        ),
        "producer_encoder_sha256": postfold._foundation_lineage_tensor_sha256(
            source_interface.producer_encoder
        ),
    }
    anchored_source = TiedCausalInterfaceFold(
        target_weight=source_target,
        producer_encoder=source_interface.producer_encoder,
        report={
            **source_body,
            "report_sha256": refold._canonical_sha256(source_body),
        },
    )
    candidate = compile_tied_residual_congruence_candidate(
        anchored_source,
        values["source_table"],
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base,
        anchor_source_residual_weight=residual,
    )
    witness = candidate.anchor_transport_witness
    assert witness is not None

    lineage = refold.build_lfm2_long_horizon_terminal_source_lineage(
        cache=values["cache"],
        bridge=values["bridge"],
        parent_final_norm=values["parent_final_norm"],
        interface=candidate.interface,
        source_selected_tied_table=values["source_table"],
        anchor_transport_witness=witness,
    )
    lineage.verify_for(
        cache=values["cache"],
        bridge=values["bridge"],
        parent_final_norm=values["parent_final_norm"],
        interface=candidate.interface,
        source_selected_tied_table=values["source_table"],
        anchor_transport_witness=witness,
    )
    with pytest.raises(refold.Lfm2LongHorizonTerminalRefoldError):
        lineage.verify_for(
            cache=values["cache"],
            bridge=values["bridge"],
            parent_final_norm=values["parent_final_norm"],
            interface=candidate.interface,
            source_selected_tied_table=values["source_table"],
        )


def _semantic_fixture(
    tmp_path: Path,
    *,
    long_horizon: bool = False,
) -> dict[str, Any]:
    donor = _GreedyTokenizer(["<B>", "a", "b", "ab", " ", "c", " c", "!", "z", "x"])
    target = _GreedyTokenizer(["<B>", "a", "b", "ab", " ", "c", "!", "z"])
    parent = _model(
        vocabulary=len(donor),
        hidden=8,
        intermediate=12,
        seed=2701,
        tied=False,
        bfloat16=False,
    )
    batches = []
    for index in range(4):
        forced = [2, 6, 7]
        if long_horizon and index == 2:
            forced = [2, 6, 7, *([8] * 30)]
        batches.append(
            {
                "input_ids": torch.tensor([[0, 8, 1]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "forced_decode_ids": torch.tensor([forced], dtype=torch.long),
            }
        )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "semantic-cache",
        parent,
        batches,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=64,
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
    tokenizer_root = tmp_path / "semantic-target-tokenizer"
    _tokenizer_files(tokenizer_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        tokenizer_root,
        donor_row_capacity=len(donor),
        target_row_capacity=len(target),
    )
    mapping = torch.tensor(
        bridge.transport_profile.target_to_donor_row_indices,
        dtype=torch.long,
    )
    source_table = (
        parent.get_output_embeddings()
        .weight.index_select(0, mapping)
        .detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .clone()
    )
    generator = torch.Generator().manual_seed(2703)
    interface = fit_tied_causal_interface(
        source_table,
        torch.randn(48, 8, generator=generator),
        target_width=4,
        validation_producer_rows=torch.randn(24, 8, generator=generator),
        minimum_validation_top1_agreement=0.0,
        metric_chunk_rows=8,
    )
    child = _model(
        vocabulary=len(target),
        hidden=4,
        intermediate=8,
        seed=2705,
        tied=True,
        bfloat16=True,
    )
    with torch.no_grad():
        child.get_input_embeddings().weight.copy_(
            interface.target_weight.to(torch.bfloat16)
        )
    source_lineage = refold.build_lfm2_long_horizon_terminal_source_lineage(
        cache=cache,
        bridge=bridge,
        parent_final_norm=parent_final_norm,
        interface=interface,
        source_selected_tied_table=source_table,
    )
    return {
        "parent": parent,
        "cache": cache,
        "bridge": bridge,
        "donor_tokenizer": donor,
        "target_tokenizer": target,
        "child": child,
        "interface": interface,
        "source_table": source_table,
        "source_lineage": source_lineage,
        "parent_final_norm": parent_final_norm,
    }


def _accepted_terminal(**kwargs) -> Lfm2CpcfTerminalActionResidualResult:
    """Use the real compiler so accepted test results retain private provenance."""

    return _compile_terminal_action_residual(**kwargs)


def _near_tie_cached_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    cached = torch.tensor([[[13.3750, 13.3750, 9.0]]])
    full = torch.tensor([[[13.2500, 13.3125, 9.0]]])
    forced = torch.tensor([[2]], dtype=torch.long)
    config = Lfm2CircuitTraceCaptureConfig(
        logit_rtol=0.0,
        logit_atol=0.25,
        require_top1_agreement=True,
        allow_top1_disagreement_within_numeric_error_envelope=True,
    )
    prefill = {
        "phase": "cached_prefill",
        "prefix_length": 3,
        **trace_cache._compare_logits(
            torch,
            cached,
            full,
            config=config,
            name="cached prefill",
        ),
    }
    decode = {
        "phase": "decode.000",
        "prefix_length": 4,
        "forced_token_sha256": trace_cache._tensor_sha256(torch, forced),
        **trace_cache._compare_logits(
            torch,
            cached,
            full,
            config=config,
            name="decode.000",
        ),
    }
    tensors = {
        "cached_prefill.logits": cached,
        "prefill.logits": full,
        "decode.000.logits": cached.clone(),
        "decode.000.full_prefix_logits": full.clone(),
        "decode.000.forced_token_ids": forced,
    }
    capture = {
        "decode_steps": 1,
        "cached_full_prefix_verifications": [prefill, decode],
        "all_cached_logits_verified": True,
        "all_cached_top1_verified": False,
        "all_cached_top1_decisions_verified": True,
        "decode_trace_mode": trace_cache.CACHED_VERIFIED_TRACE_MODE,
        "canonical_trace_source": "verified_one_token_cache",
        "cached_logits_accepted": True,
        "cached_forward_count": 2,
    }
    return tensors, capture


def _material_cached_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    cached = torch.tensor([[[20.0, -5.0, 9.0]]])
    full = torch.tensor([[[-5.0, 20.0, 9.0]]])
    forced = torch.tensor([[2]], dtype=torch.long)
    # Ask the cache comparison helper for the complete, exact diagnostics under
    # a deliberately loose envelope, then state the real terminal tolerance
    # outcome explicitly.  The terminal verifier recomputes it at atol=0.25.
    diagnostic_config = Lfm2CircuitTraceCaptureConfig(
        logit_rtol=0.0,
        logit_atol=100.0,
        require_top1_agreement=True,
        allow_top1_disagreement_within_numeric_error_envelope=True,
    )

    def comparison(name: str) -> dict[str, Any]:
        result = trace_cache._compare_logits(
            torch,
            cached,
            full,
            config=diagnostic_config,
            name=name,
        )
        result["numerically_close"] = False
        return result

    prefill = {
        "phase": "cached_prefill",
        "prefix_length": 3,
        **comparison("material cached prefill"),
    }
    decode = {
        "phase": "decode.000",
        "prefix_length": 4,
        "forced_token_sha256": trace_cache._tensor_sha256(torch, forced),
        **comparison("material decode.000"),
    }
    tensors = {
        "cached_prefill.logits": cached,
        "prefill.logits": full,
        "decode.000.logits": cached.clone(),
        "decode.000.full_prefix_logits": full.clone(),
        "decode.000.forced_token_ids": forced,
    }
    capture = {
        "decode_steps": 1,
        "cached_full_prefix_verifications": [prefill, decode],
        "all_cached_logits_verified": False,
        "all_cached_top1_verified": False,
        "all_cached_top1_decisions_verified": True,
        "decode_trace_mode": trace_cache.CACHED_VERIFIED_TRACE_MODE,
        "canonical_trace_source": "verified_one_token_cache",
        "cached_logits_accepted": False,
        "cached_forward_count": 2,
    }
    return tensors, capture


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
        return _accepted_terminal(**kwargs)

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
        source_lineage=values["source_lineage"],
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
    compatibility = report["terminal_cache_compatibility"]
    assert compatibility["terminal_trajectory_compatibility_verified"] is True
    assert compatibility["top1_exactness_is_publication_gate"] is False
    assert compatibility["full_prefix_recache_required"] is False
    assert compatibility["preflight_io_contract"]["trace_file_sha256_passes"] == 0
    assert compatibility["preflight_io_contract"]["trace_payload_bytes_read"] == 0
    assert audit["consuming_selective_session_constructions"] == 1
    assert audit["consuming_trace_file_sha256_passes_per_batch"] == 1
    assert (
        audit[
            "additional_full_trace_file_sha256_passes_performed_by_this_terminal_compiler"
        ]
        == 0
    )
    payload_audit = report["terminal_cache_tensor_payload_load_audit"]
    assert payload_audit["trace_file_sha256_passes_accounted_here"] == 0
    assert payload_audit["selected_tensor_payload_load_count"] == 4
    assert audit["selected_tensor_payload_loads_total"] == 4
    assert (
        audit["selected_tensor_payload_logical_bytes_total"]
        == payload_audit["selected_tensor_payload_logical_bytes_total"]
    )
    assert all(
        value["all_logits_numerically_close"] is True
        and value["all_top1_disagreements_within_numeric_error_envelope"] is True
        for value in report["cached_full_prefix_payload_verification"]
    )


def test_metadata_preflight_reads_only_records_and_seals_zero_trace_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    observed = []
    original = refold._cache_file_sha256

    def record_hash_only(path):
        observed.append(Path(path).name)
        assert Path(path).name == trace_cache.BATCH_RECORD_FILENAME
        return original(path)

    monkeypatch.setattr(refold, "_cache_file_sha256", record_hash_only)
    receipt = refold.preflight_lfm2_long_horizon_terminal_cache(values["cache"])

    assert observed == [trace_cache.BATCH_RECORD_FILENAME] * 4
    io = receipt["preflight_io_contract"]
    assert io["record_file_sha256_passes_total"] == 4
    assert io["trace_file_sha256_passes"] == 0
    assert io["trace_payload_bytes_read"] == 0
    body = dict(receipt)
    supplied = body.pop("authentication_sha256")
    assert supplied == refold._canonical_sha256(body)


def test_forced_cached_near_ties_are_raw_rederived_not_exact_top1_gated() -> None:
    tensors, capture = _near_tie_cached_payload()

    metadata = refold._terminal_cache_capture_compatibility(
        capture,
        batch_index=0,
    )
    raw = refold._verify_cached_full_prefix_payload_equivalence(
        tensors,
        capture,
        batch_index=0,
        logit_rtol=0.0,
        logit_atol=0.25,
    )

    assert metadata["cached_top1_exact_for_all_phases"] is False
    assert metadata["top1_exactness_is_diagnostic_only"] is True
    assert metadata["top1_disagreement_count"] == 2
    assert raw["top1_disagreement_count"] == 2
    assert raw["all_logits_numerically_close"] is True
    assert raw["all_top1_disagreements_within_numeric_error_envelope"] is True
    assert all(
        value["forced_token_sha256"] is not None
        for value in raw["phase_verification"][1:]
    )


def test_material_cached_trace_is_diagnostic_but_integrity_still_fails_closed() -> None:
    material_tensors, material_capture = _material_cached_payload()
    metadata = refold._terminal_cache_capture_compatibility(
        material_capture,
        batch_index=0,
    )
    raw = refold._verify_cached_full_prefix_payload_equivalence(
        material_tensors,
        material_capture,
        batch_index=0,
        logit_rtol=0.0,
        logit_atol=0.25,
    )

    assert metadata["cached_logits_numerically_verified"] is False
    assert metadata["collector_cached_logits_accepted"] is False
    assert metadata["numeric_diagnostics_are_admission_authority"] is False
    assert raw["all_logits_numerically_close"] is False
    assert raw["top1_disagreement_count"] == 2
    assert raw["numeric_diagnostics_are_admission_authority"] is False

    tensors, capture = _near_tie_cached_payload()
    material = json.loads(json.dumps(capture))
    material["cached_full_prefix_verifications"][0][
        "top1_disagreement_within_numeric_error_envelope"
    ] = False
    diagnostic = refold._terminal_cache_capture_compatibility(material, batch_index=0)
    assert diagnostic["all_top1_disagreements_numeric_envelope_verified"] is False
    assert diagnostic["numeric_diagnostics_are_admission_authority"] is False

    incomplete = json.loads(json.dumps(capture))
    incomplete["cached_full_prefix_verifications"].pop()
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="coverage is incomplete",
    ):
        refold._terminal_cache_capture_compatibility(incomplete, batch_index=0)

    wrong_raw = dict(tensors)
    wrong_raw["decode.000.full_prefix_logits"] = torch.tensor([[[20.0, -5.0, 9.0]]])
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="do not reproduce the sealed numeric verification",
    ):
        refold._verify_cached_full_prefix_payload_equivalence(
            wrong_raw,
            capture,
            batch_index=0,
            logit_rtol=0.0,
            logit_atol=0.25,
        )

    wrong_forced = json.loads(json.dumps(capture))
    wrong_forced["cached_full_prefix_verifications"][1]["forced_token_sha256"] = (
        "0" * 64
    )
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="sealed forced token digest differs",
    ):
        refold._verify_cached_full_prefix_payload_equivalence(
            tensors,
            wrong_forced,
            batch_index=0,
            logit_rtol=0.0,
            logit_atol=0.25,
        )


def test_record_inventory_missing_raw_counterpart_fails_early(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    batch = values["cache"].load_batch(0)
    record = batch.record
    inventory = dict(record["tensors"])
    inventory.pop("decode.000.full_prefix_logits")
    compatibility = refold._terminal_cache_capture_compatibility(
        record["capture"],
        batch_index=0,
    )

    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="omits required terminal evidence",
    ):
        refold._terminal_cache_record_inventory_compatibility(
            inventory,
            compatibility,
            batch_index=0,
            final_layer_index=1,
        )


def test_metadata_preflight_failure_precedes_target_seed_replay_and_solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    events = []

    def fail_preflight(_cache):
        events.append("metadata_preflight")
        raise refold.Lfm2LongHorizonTerminalRefoldError("synthetic incompatibility")

    def forbidden(name):
        def call(*_args, **_kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran after incompatible preflight")

        return call

    monkeypatch.setattr(
        refold,
        "preflight_lfm2_long_horizon_terminal_cache",
        fail_preflight,
    )
    monkeypatch.setattr(refold, "_target_runtime_seed", forbidden("target_seed"))
    monkeypatch.setattr(refold, "_build_split_evidence", forbidden("child_replay"))
    monkeypatch.setattr(
        refold,
        "compile_lfm2_cpcf_terminal_action_residual",
        forbidden("closed_form_solve"),
    )

    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="synthetic incompatibility",
    ):
        refold.compile_install_lfm2_long_horizon_terminal_refold(
            cache=values["cache"],
            bridge=values["bridge"],
            target_model=values["child"],
            interface=values["interface"],
            source_selected_tied_table=values["source_table"],
            source_lineage=values["source_lineage"],
            parent_final_norm=values["parent_final_norm"],
        )
    assert events == ["metadata_preflight"]


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
        source_lineage=values["source_lineage"],
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


def test_semantic_span_replays_expansion_contraction_and_suffix_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _semantic_fixture(tmp_path)

    def forbidden_model_forward(*_args, **_kwargs):
        raise AssertionError("model.forward must not be called")

    values["child"].forward = forbidden_model_forward

    def accepted_compiler(**kwargs):
        return _accepted_terminal(**kwargs)

    monkeypatch.setattr(
        refold,
        "compile_lfm2_cpcf_terminal_action_residual",
        accepted_compiler,
    )
    result = refold.compile_install_lfm2_long_horizon_terminal_refold(
        cache=values["cache"],
        bridge=values["bridge"],
        target_model=values["child"],
        interface=values["interface"],
        source_selected_tied_table=values["source_table"],
        source_lineage=values["source_lineage"],
        parent_final_norm=values["parent_final_norm"],
        tokenizer_replay_mode="semantic_span",
        donor_tokenizer=values["donor_tokenizer"],
        target_tokenizer=values["target_tokenizer"],
    )

    report = result.as_dict()
    assert report["tokenizer_replay_mode"] == "semantic_span"
    assert report["bridge_inverse_contract"] == {
        "every_cached_donor_prompt_and_forced_id_covered": False,
        "every_used_donor_row_has_exactly_one_target_row": False,
        "exact_full_prefix_text_roundtrip_used": True,
        "suffix_rewrite_supported": True,
        "numeric_token_id_equality_required": False,
    }
    semantic = report["semantic_span_bridges"][0]
    assert [phase["target_full_prefix_token_ids"] for phase in semantic["phases"]] == [
        [0, 7, 1],
        [0, 7, 3],
        [0, 7, 3, 4, 5],
        [0, 7, 3, 4, 5, 6],
    ]
    assert semantic["phases"][1]["target_replaced_suffix_token_count"] == 1
    assert semantic["phases"][2]["target_span_token_count"] == 2
    assert semantic["expansion"]["full_horizon"]["retokenized_suffix_phase_count"] == 1
    receipts = report["trajectory_replay_receipts"]
    assert all(value["full_prefix_replay_calls"] == 2 for value in receipts)
    assert all(
        value["replay_strategy"] == "semantic-maximal-target-prefix-branch-cover-v1"
        for value in receipts
    )
    assert report["operation_audit"]["child_embedding_seed_lookups"] == 8
    assert report["operation_audit"]["child_macrocell_replay_calls"] == 16
    assert len(report["parent_action_projection_sha256"]) == 4
    assert report["operation_audit"]["training_steps"] == 0
    payload_audit = report["terminal_cache_tensor_payload_load_audit"]
    session_batches = report["cache_authentication"]["session"][
        "ordered_authenticated_batches"
    ]
    expected_bytes = sum(value["logical_bytes"] for value in session_batches)
    assert payload_audit["selected_tensor_payload_load_count"] == 4
    assert payload_audit["full_tensor_inventory_payload_load_count"] == 4
    assert payload_audit["every_batch_loaded_full_tensor_inventory_once"] is True
    assert payload_audit["full_tensor_inventory_payload_logical_bytes_total"] == (
        expected_bytes
    )
    assert (
        report["operation_audit"]["full_tensor_inventory_payload_logical_bytes_total"]
        == expected_bytes
    )


def test_semantic_span_populates_real_32_plus_horizon_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _semantic_fixture(tmp_path, long_horizon=True)

    def accepted_compiler(**kwargs):
        return _accepted_terminal(**kwargs)

    monkeypatch.setattr(
        refold,
        "compile_lfm2_cpcf_terminal_action_residual",
        accepted_compiler,
    )
    result = refold.compile_install_lfm2_long_horizon_terminal_refold(
        cache=values["cache"],
        bridge=values["bridge"],
        target_model=values["child"],
        interface=values["interface"],
        source_selected_tied_table=values["source_table"],
        source_lineage=values["source_lineage"],
        parent_final_norm=values["parent_final_norm"],
        tokenizer_replay_mode="semantic_span",
        donor_tokenizer=values["donor_tokenizer"],
        target_tokenizer=values["target_tokenizer"],
    )

    report = result.as_dict()
    bucket = report["horizon_metrics"]["buckets"]["32+"]
    assert bucket["available"] is True
    assert bucket["rows"] == 2
    selection_bridge = report["semantic_span_bridges"][2]
    assert selection_bridge["expansion"]["horizon_buckets"][2]["phase_count"] == 1


def test_semantic_span_is_explicit_and_strict_path_still_fails_unmapped_rows(
    tmp_path: Path,
) -> None:
    values = _semantic_fixture(tmp_path)
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="does not cover every cached forced-decode row",
    ):
        refold.compile_install_lfm2_long_horizon_terminal_refold(
            cache=values["cache"],
            bridge=values["bridge"],
            target_model=values["child"],
            interface=values["interface"],
            source_selected_tied_table=values["source_table"],
            source_lineage=values["source_lineage"],
            parent_final_norm=values["parent_final_norm"],
        )


def test_semantic_span_fails_closed_on_wrong_action_projection(
    tmp_path: Path,
) -> None:
    values = _semantic_fixture(tmp_path)
    wrong_source = values["source_table"].flip(0).contiguous()
    generator = torch.Generator().manual_seed(3703)
    wrong_interface = fit_tied_causal_interface(
        wrong_source,
        torch.randn(48, 8, generator=generator),
        target_width=4,
        validation_producer_rows=torch.randn(24, 8, generator=generator),
        minimum_validation_top1_agreement=0.0,
        metric_chunk_rows=8,
    )
    with torch.no_grad():
        values["child"].get_input_embeddings().weight.copy_(
            wrong_interface.target_weight.to(torch.bfloat16)
        )

    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="source lineage differs",
    ):
        refold.compile_install_lfm2_long_horizon_terminal_refold(
            cache=values["cache"],
            bridge=values["bridge"],
            target_model=values["child"],
            interface=wrong_interface,
            source_selected_tied_table=wrong_source,
            source_lineage=values["source_lineage"],
            parent_final_norm=values["parent_final_norm"],
            tokenizer_replay_mode="semantic_span",
            donor_tokenizer=values["donor_tokenizer"],
            target_tokenizer=values["target_tokenizer"],
        )


def test_large_projection_error_survives_with_exact_lineage_and_tamper_fails(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    donor_rows = values["bridge"].transport_profile.donor_row_capacity
    parent_rows = torch.zeros((2, 8), dtype=torch.bfloat16)
    logits = torch.linspace(-40.0, 40.0, donor_rows).reshape(1, 1, -1)
    tensors = {
        "prefill.logits": logits,
        "decode.000.logits": logits.flip(-1).contiguous(),
    }

    report = refold._verify_parent_action_projection(
        tensors,
        parent_rows,
        values["bridge"],
        values["source_table"],
        decode_steps=1,
        source_lineage=values["source_lineage"],
        batch_index=0,
        input_sha256="1" * 64,
        record_sha256="2" * 64,
        payload_binding_sha256="3" * 64,
    )

    assert report["relative_rmse"] > 0.02
    assert report["relative_rmse_diagnostic_only"] is True
    assert report["numeric_admission_threshold_applied"] is False
    assert report["numeric_projection_error_can_reject_or_rollback"] is False

    tampered = values["source_lineage"].as_dict()
    tampered["mapped_table_bfloat16_sha256"] = "0" * 64
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="source lineage authentication is invalid",
    ):
        refold.Lfm2LongHorizonTerminalSourceLineage(tampered)


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
            source_lineage=values["source_lineage"],
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


def test_parent_final_norm_runtime_replay_authenticates_cached_endpoints() -> None:
    seal = refold.Lfm2SealedParentFinalRmsNorm(
        cache_key="1" * 64,
        parent_fingerprint="2" * 64,
        layer_topology_fingerprint="3" * 64,
        weight=torch.tensor([1.0, 0.75, 1.25], dtype=torch.float32),
        epsilon=1e-5,
    )
    residual = torch.tensor(
        [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]],
        dtype=torch.float32,
    )
    replayed = refold._runtime_parent_rmsnorm(residual, seal)
    tensors = {}
    for index, phase in enumerate(("prefill", "decode.000")):
        endpoint_input = residual[index : index + 1].reshape(1, 1, -1)
        endpoint_output = replayed[index : index + 1].reshape(1, 1, -1)
        tensors[f"{phase}.output.final_norm_input"] = endpoint_input
        tensors[f"{phase}.output.final_norm_output"] = endpoint_output
        tensors[f"{phase}.output.lm_head_input"] = endpoint_output.clone()

    target, receipt = refold._verify_parent_final_norm_endpoint(
        tensors,
        residual,
        seal,
        decode_steps=1,
    )

    torch.testing.assert_close(
        target,
        replayed.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert receipt["final_residual_to_norm_input_exact"] is True
    assert receipt["runtime_replay_to_cached_output_verified"] is True
    assert receipt["authenticated_cached_output_used_as_compiler_target"] is True
    body = dict(receipt)
    authentication = body.pop("authentication_sha256")
    assert authentication == refold._canonical_sha256(body)

    wrong = refold.Lfm2SealedParentFinalRmsNorm(
        cache_key=seal.cache_key,
        parent_fingerprint=seal.parent_fingerprint,
        layer_topology_fingerprint=seal.layer_topology_fingerprint,
        weight=seal.weight * 2.0,
        epsilon=seal.epsilon,
    )
    wrong_target, wrong_receipt = refold._verify_parent_final_norm_endpoint(
        tensors,
        residual,
        wrong,
        decode_steps=1,
    )
    torch.testing.assert_close(
        wrong_target,
        replayed.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert wrong_receipt["runtime_replay_within_legacy_numeric_envelope"] is False
    assert wrong_receipt["runtime_replay_metrics_diagnostic_only"] is True
    assert wrong_receipt["runtime_replay_numeric_admission_threshold_applied"] is False


def test_loads_parent_final_norm_directly_from_identity_matched_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path / "fixture")
    parent = values["parent"]
    source = tmp_path / "direct-local-parent"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(parent.config.to_dict()),
        encoding="utf-8",
    )
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in parent.state_dict().items()
        },
        source / "model.safetensors",
    )

    loaded = refold.load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        values["cache"],
        source,
        tensor_key="model.embedding_norm.weight",
        epsilon_config_key="norm_eps",
    )

    assert torch.equal(
        loaded.final_norm.weight,
        parent.model.embedding_norm.weight.detach().to(device="cpu"),
    )
    assert loaded.final_norm.epsilon == float(parent.config.norm_eps)
    report = loaded.as_dict()
    assert report["tensor_key"] == "model.embedding_norm.weight"
    assert report["epsilon_config_key"] == "norm_eps"
    assert report["operation_audit"]["parent_model_objects_constructed"] == 0
    assert report["operation_audit"]["parent_model_forward_calls"] == 0
    assert (
        report["reconstructed_parent_identity"]["state_dict_sha256"]
        == values["cache"].manifest["parent"]["state_dict_sha256"]
    )

    def accepted_compiler(**kwargs):
        return _accepted_terminal(**kwargs)

    monkeypatch.setattr(
        refold,
        "compile_lfm2_cpcf_terminal_action_residual",
        accepted_compiler,
    )
    result = refold.compile_install_lfm2_long_horizon_terminal_refold(
        cache=values["cache"],
        bridge=values["bridge"],
        target_model=values["child"],
        interface=values["interface"],
        source_selected_tied_table=values["source_table"],
        source_lineage=values["source_lineage"],
        parent_final_norm=loaded,
    )
    assert (
        result.as_dict()["parent_final_norm_direct_local_source"]["result_sha256"]
        == loaded.result_sha256
    )


def test_mapped_tied_table_rejects_same_shape_different_parent_checkpoint(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path / "fixture", parent_tied=True)
    parent_a = values["parent"]

    def save_parent(model: Any, directory: Path) -> None:
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(model.config.to_dict()),
            encoding="utf-8",
        )
        save_file(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in model.state_dict().items()
                if name != "lm_head.weight"
            },
            directory / "model.safetensors",
        )

    checkpoint_a = tmp_path / "parent-a"
    save_parent(parent_a, checkpoint_a)
    authenticated_a = refold.load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        values["cache"],
        checkpoint_a,
        tensor_key="model.embedding_norm.weight",
        epsilon_config_key="norm_eps",
    )
    rows_a, _legacy_a, source_a = (
        postfold.load_lfm2_mapped_parent_tied_table_from_local_safetensors(
            checkpoint_a,
            cache=values["cache"],
            bridge=values["bridge"],
            authenticated_parent_final_norm=authenticated_a,
            include_source_lineage_report=True,
        )
    )
    assert source_a["source_checkpoint_exactly_matches_authenticated_parent"] is True
    assert source_a[
        "independently_authenticated_parent_identity_sha256"
    ] == values["cache"].parent_fingerprint
    assert torch.equal(
        rows_a,
        parent_a.get_input_embeddings().weight.detach().to(torch.bfloat16),
    )
    direct_lineage = refold.build_lfm2_long_horizon_terminal_source_lineage(
        cache=values["cache"],
        bridge=values["bridge"],
        parent_final_norm=authenticated_a,
        interface=values["interface"],
        source_selected_tied_table=rows_a,
        direct_local_mapped_source_report=source_a,
    )
    assert direct_lineage.direct_local_checkpoint_source_authenticated is True
    reconstruction = direct_lineage.document["direct_local_mapped_reconstruction"]
    assert reconstruction["exact_tensor_equality_verified"] is True
    assert reconstruction["operation_audit"]["parent_model_objects_constructed"] == 0
    assert reconstruction["operation_audit"]["parent_model_forward_calls"] == 0
    assert (
        reconstruction["reconstructed_mapped_table_descriptor"]
        == reconstruction["supplied_mapped_table_descriptor"]
    )

    parent_b = _model(
        vocabulary=31,
        hidden=8,
        intermediate=12,
        seed=99_999,
        tied=True,
        bfloat16=False,
    )
    checkpoint_b = tmp_path / "parent-b"
    save_parent(parent_b, checkpoint_b)
    assert not torch.equal(
        parent_a.get_input_embeddings().weight,
        parent_b.get_input_embeddings().weight,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="differs from the independently authenticated parent checkpoint",
    ):
        postfold.load_lfm2_mapped_parent_tied_table_from_local_safetensors(
            checkpoint_b,
            cache=values["cache"],
            bridge=values["bridge"],
            authenticated_parent_final_norm=authenticated_a,
            include_source_lineage_report=True,
        )

    # A source report has an ordinary caller-recomputable self-hash.  Prove
    # that substituting B's mapped bytes while retaining A's exact checkpoint
    # inventory cannot mint terminal lineage, even when the supplied causal
    # interface is internally consistent with B.
    rows_b = (
        parent_b.get_input_embeddings()
        .weight.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .clone()
    )
    generator = torch.Generator().manual_seed(1703)
    interface_b = fit_tied_causal_interface(
        rows_b,
        torch.randn(48, 8, generator=generator),
        target_width=4,
        validation_producer_rows=torch.randn(24, 8, generator=generator),
        minimum_validation_top1_agreement=0.0,
        metric_chunk_rows=8,
    )
    forged = dict(source_a)
    forged.pop("result_sha256")
    forged["source_tensor_sha256"] = refold._tensor_sha256(
        parent_b.get_input_embeddings().weight.detach().to(device="cpu"),
        "parent_tied_source_tensor",
    )
    forged["mapped_table_bfloat16_sha256"] = refold._tensor_sha256(
        rows_b,
        "mapped_parent_tied_table",
    )
    forged["mapped_table_interface_source_sha256"] = (
        refold._interface_tensor_sha256(rows_b.to(torch.float64))
    )
    forged["result_sha256"] = refold._canonical_sha256(forged)
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="not the exact bridge projection",
    ):
        refold.build_lfm2_long_horizon_terminal_source_lineage(
            cache=values["cache"],
            bridge=values["bridge"],
            parent_final_norm=authenticated_a,
            interface=interface_b,
            source_selected_tied_table=rows_b,
            direct_local_mapped_source_report=forged,
        )

    tampered_mapping = dict(source_a)
    tampered_mapping.pop("result_sha256")
    tampered_mapping["target_to_donor_row_indices_sha256"] = "0" * 64
    tampered_mapping["result_sha256"] = refold._canonical_sha256(tampered_mapping)
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="differs from the exact cache, bridge, checkpoint",
    ):
        refold.build_lfm2_long_horizon_terminal_source_lineage(
            cache=values["cache"],
            bridge=values["bridge"],
            parent_final_norm=authenticated_a,
            interface=values["interface"],
            source_selected_tied_table=rows_a,
            direct_local_mapped_source_report=tampered_mapping,
        )


def test_direct_local_parent_reconstructs_authenticated_bfloat16_rope_buffers(
    tmp_path: Path,
) -> None:
    values = _fixture(
        tmp_path / "fixture",
        parent_tied=True,
        parent_bfloat16=True,
    )
    parent = values["parent"]
    source = tmp_path / "direct-local-parent"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(parent.config.to_dict()),
        encoding="utf-8",
    )
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in parent.state_dict().items()
            if name != "lm_head.weight"
        },
        source / "model.safetensors",
    )

    loaded = refold.load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        values["cache"],
        source,
        tensor_key="model.embedding_norm.weight",
        epsilon_config_key="norm_eps",
    )

    report = loaded.as_dict()
    assert report["checkpoint_floating_dtypes"] == ["bfloat16"]
    assert report["runtime_buffer_dtype"] == "bfloat16"
    assert (
        report["reconstructed_parent_identity"]["runtime_buffer_sha256"]
        == values["cache"].manifest["parent"]["runtime_buffer_sha256"]
    )
    assert (
        report["reconstructed_parent_identity"]["state_dict_sha256"]
        == values["cache"].manifest["parent"]["state_dict_sha256"]
    )


def test_direct_local_parent_rejects_config_checkpoint_dtype_disagreement() -> None:
    with pytest.raises(
        refold.Lfm2LongHorizonTerminalRefoldError,
        match="config dtype and direct-local checkpoint floating dtypes disagree",
    ):
        refold._direct_local_parent_runtime_dtype(
            {"dtype": "bfloat16"},
            {torch.float32},
        )
