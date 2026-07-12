from __future__ import annotations

import hashlib
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

import run_qwen_residual_attention_commit as commit  # noqa: E402
import run_qwen_residual_attention_probe as probe  # noqa: E402
import neural_forge.hf_qwen_codebook_model as codebook_model  # noqa: E402
from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_residual_attention_fold import (  # noqa: E402
    ResidualAttentionGatePolicy,
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


def _runtime(seed: int = 1409):
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    runtime = prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=2,
        neuron_prototypes_per_neuron=2,
    )
    return runtime, tokenizer


def _contract(runtime, tokenizer: _Tokenizer) -> dict[str, object]:
    rows = [int(value) for value in runtime.parent_token_rows.detach().cpu().tolist()]
    return {
        "tokenizer_name_or_path": "tiny-residual-attention-commit-tokenizer",
        "vocab_size": len(tokenizer),
        "vocab_sha256": hashlib.sha256(
            codebook_model._canonical_json_bytes(tokenizer.get_vocab())
        ).hexdigest(),
        "special_token_ids": {
            name: getattr(runtime.config, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": hashlib.sha256(
            codebook_model._canonical_json_bytes(rows)
        ).hexdigest(),
    }


def _saved_incumbent(path: Path, seed: int = 1409):
    runtime, tokenizer = _runtime(seed)
    save_qwen3_codebook_checkpoint(
        runtime,
        path,
        tokenizer_contract=_contract(runtime, tokenizer),
    )
    return load_qwen3_codebook_checkpoint(path), tokenizer


def _limits() -> HFCachedAttentionTraceLimits:
    return HFCachedAttentionTraceLimits(
        max_prompts=1,
        max_horizon=1,
        max_prompt_tokens=16,
        max_layers=4,
        max_groups=1,
        max_model_devices=1,
        max_capture_bytes=2_000_000,
    )


def _permissive_gate() -> ResidualAttentionGatePolicy:
    # This only keeps the tiny serialization test independent of random model
    # quality.  Production CLI calls have no bypass and use the strict default.
    return ResidualAttentionGatePolicy(max_risk_score=1_000_000.0, minimum_horizon=1)


@pytest.mark.parametrize("compensation", ["scalar", "ridge"])
def test_experimental_commit_saves_reloads_and_never_changes_incumbent(
    tmp_path: Path,
    compensation: str,
) -> None:
    source = tmp_path / f"p2-{compensation}"
    output = tmp_path / f"p1-{compensation}"
    runtime, _tokenizer = _saved_incumbent(source)
    source_before = commit._checkpoint_inventory(source)
    parent = qwen3_attention_runtime_snapshot(runtime)
    calibration = torch.tensor([[1, 4, 6]], dtype=torch.long)
    heldout = torch.tensor([[1, 5, 7]], dtype=torch.long)

    report = commit.commit_qwen_residual_attention_redirect(
        runtime,
        (calibration,),
        (heldout,),
        (probe.AttentionRedirect(1, 0, 0, 0),),
        source_checkpoint=source,
        output_checkpoint=output,
        compensation=compensation,
        horizon=1,
        ridge=1e-4,
        trace_limits=_limits(),
        max_affected_sites=4,
        gate_policy=_permissive_gate(),
    )

    child = qwen3_attention_runtime_snapshot(runtime)
    assert child.prototype_count == parent.prototype_count - 1
    assert report["status"] == "experimental-uncertified"
    assert report["fold_committed"] is True
    assert report["compaction_certified"] is False
    assert report["both_splits_passed"] is True
    assert report["fit_data_access"]["heldout_rows_used"] == 0
    assert report["data_split"]["disjoint"] is True
    assert report["incumbent_byte_exact"] is True
    assert commit._checkpoint_inventory(source) == source_before
    assert report["physical_attention_bytes_saved"] > 0
    assert report["parameter_accounting"]["unique_parameter_delta"] < 0
    assert report["reload_audit"]["full_tensor_state_exact"] is True
    assert report["reload_audit"]["parent_runtime_reference_count"] == 0
    assert set(value["sha256"] for value in report["output_inventory"].values())
    assert report["edit_audit"]["unaffected_o_slots_byte_exact"] is True
    assert report["edit_audit"]["non_o_tensor_identity_and_version_exact"] is True
    fit = report["fits"]["0"]
    assert fit["affected_sites"] == [[0, 0], [1, 0]]
    assert fit["design_contract"].startswith("candidate-runtime")
    assert fit["target_contract"] == "parent-selected-group-O-contribution"

    reloaded = load_qwen3_codebook_checkpoint(output).eval()
    assert qwen3_attention_runtime_snapshot(reloaded) == child
    assert qwen3_codebook_parameter_report(reloaded).parent_runtime_reference_count == 0
    marker = getattr(reloaded.config, commit.CONFIG_MARKER)
    assert marker["status"] == "experimental-uncertified"
    assert marker["compaction_certified"] is False
    with torch.inference_mode():
        expected = runtime(calibration, use_cache=False).logits
        actual = reloaded(calibration, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


def test_failure_after_o_edit_rolls_back_exactly_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "p2"
    output = tmp_path / "must-not-exist"
    runtime, _tokenizer = _saved_incumbent(source, seed=1410)
    source_before = commit._checkpoint_inventory(source)
    snapshot_before = qwen3_attention_runtime_snapshot(runtime)
    report_before = qwen3_codebook_parameter_report(runtime)

    def fail(stage: str) -> None:
        if stage == "after_compensation":
            raise RuntimeError("injected post-O failure")

    with pytest.raises(RuntimeError, match="injected post-O failure"):
        commit.commit_qwen_residual_attention_redirect(
            runtime,
            (torch.tensor([[1, 4, 6]]),),
            (torch.tensor([[1, 5, 7]]),),
            (probe.AttentionRedirect(1, 0, 0, 0),),
            source_checkpoint=source,
            output_checkpoint=output,
            compensation="ridge",
            horizon=1,
            ridge=1e-4,
            trace_limits=_limits(),
            max_affected_sites=4,
            gate_policy=_permissive_gate(),
            fault_injector=fail,
        )

    assert not output.exists()
    assert commit._checkpoint_inventory(source) == source_before
    assert qwen3_attention_runtime_snapshot(runtime) == snapshot_before
    assert qwen3_codebook_parameter_report(runtime) == report_before
    assert not hasattr(runtime.config, commit.CONFIG_MARKER)
    assert not list(tmp_path.glob(".must-not-exist.staging.*"))


def test_split_and_destination_guards_fail_before_mutation(tmp_path: Path) -> None:
    source = tmp_path / "p2"
    runtime, _tokenizer = _saved_incumbent(source, seed=1411)
    snapshot = qwen3_attention_runtime_snapshot(runtime)
    prompt = torch.tensor([[1, 4, 6]])

    with pytest.raises(commit.QwenResidualAttentionCommitError, match="disjoint"):
        commit.commit_qwen_residual_attention_redirect(
            runtime,
            (prompt,),
            (prompt.clone(),),
            (probe.AttentionRedirect(1, 0, 0, 0),),
            source_checkpoint=source,
            output_checkpoint=tmp_path / "overlap-output",
            compensation="scalar",
            horizon=1,
            trace_limits=_limits(),
            gate_policy=_permissive_gate(),
        )
    assert qwen3_attention_runtime_snapshot(runtime) == snapshot

    existing = tmp_path / "existing-output"
    existing.mkdir()
    with pytest.raises(commit.QwenResidualAttentionCommitError, match="must not"):
        commit.commit_qwen_residual_attention_redirect(
            runtime,
            (prompt,),
            (torch.tensor([[1, 5, 7]]),),
            (probe.AttentionRedirect(1, 0, 0, 0),),
            source_checkpoint=source,
            output_checkpoint=existing,
            compensation="scalar",
            horizon=1,
            trace_limits=_limits(),
            gate_policy=_permissive_gate(),
        )
    assert qwen3_attention_runtime_snapshot(runtime) == snapshot
