from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

import neural_forge.hf_lfm2_on_policy_evidence_round as MODULE  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)


WORDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
)
FIT_PROMPTS = (
    "alpha beta gamma",
    "delta epsilon zeta",
    "eta theta iota",
)
MANUAL_PROMPTS = (
    "kappa lambda mu",
    "nu alpha beta",
)


def _tiny_lfm2_config() -> object:
    config = transformers.Lfm2Config(
        vocab_size=4 + len(WORDS),
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _tiny_lfm2_model(*, seed: int) -> object:
    config = _tiny_lfm2_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


def _write_word_tokenizer(directory: Path) -> None:
    vocabulary = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    for index, word in enumerate(WORDS):
        vocabulary[word] = 4 + index
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
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


def _publish_child(directory: Path, *, seed: int) -> object:
    """Write one standalone local LFM2 child checkpoint plus its tokenizer."""

    directory.mkdir(parents=True, exist_ok=True)
    model = _tiny_lfm2_model(seed=seed)
    (directory / "config.json").write_text(
        json.dumps(model.config.to_dict()), encoding="utf-8"
    )
    save_file(
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        str(directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    _write_word_tokenizer(directory)
    return model


def _capture_config() -> Lfm2CircuitTraceCaptureConfig:
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=4,
        max_decode_steps=8,
        max_capture_bytes=64 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )


def _rollout() -> MODULE.Lfm2OnPolicyRolloutConfig:
    return MODULE.Lfm2OnPolicyRolloutConfig(
        max_prompts=8,
        max_new_tokens=4,
        max_context_tokens=64,
        seed=0,
        max_capture_batches=16,
    )


def _write_manual_manifest(path: Path, prompts) -> Path:
    path.write_text(
        json.dumps(
            [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in prompts
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    """One published child, one sealed teacher-forced base cache, one parent."""

    root = tmp_path_factory.mktemp("lfm2-on-policy-round")
    parent = _tiny_lfm2_model(seed=701)
    child_directory = root / "child-a"
    _publish_child(child_directory, seed=811)
    capture_config = _capture_config()
    # A teacher-forced base corpus that no child rollout reproduces.
    base_cache = build_lfm2_circuit_trace_cache(
        root / "base-cache-root",
        parent,
        (
            {
                "input_ids": torch.tensor([[2, 4, 5]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "forced_decode_ids": torch.tensor([[11, 12]], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([[2, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "forced_decode_ids": torch.tensor([[13, 14]], dtype=torch.long),
            },
        ),
        config=capture_config,
        input_device="cpu",
    )
    manual_manifest = _write_manual_manifest(
        root / "manual-review-prompts.json", MANUAL_PROMPTS
    )
    return SimpleNamespace(
        root=root,
        parent=parent,
        parent_fingerprint=fingerprint_lfm2_parent(parent).identity_sha256,
        parent_tokenizer=MODULE.load_lfm2_direct_local_tokenizer(child_directory),
        child_directory=child_directory,
        base_cache=base_cache,
        capture_config=capture_config,
        manual_manifest=manual_manifest,
    )


def _run(lane, output_root: Path, **overrides):
    arguments = {
        "child_artifact_directory": lane.child_directory,
        "parent_model": lane.parent,
        "parent_tokenizer": lane.parent_tokenizer,
        "parent_fingerprint": lane.parent_fingerprint,
        "fit_prompts": FIT_PROMPTS,
        "manual_prompt_manifest_path": lane.manual_manifest,
        "base_cache_directory": lane.base_cache.path,
        "output_root": output_root,
        "capture_config": lane.capture_config,
        "rollout": _rollout(),
        "input_device": "cpu",
    }
    arguments.update(overrides)
    return MODULE.run_lfm2_on_policy_evidence_round(**arguments)


def test_round_seals_an_extension_cache_and_verifies(lane, tmp_path) -> None:
    result = _run(lane, tmp_path / "rounds")

    assert result.manifest["schema"] == MODULE.ROUND_MANIFEST_SCHEMA
    assert result.round_index == 0
    assert result.max_rounds == MODULE.DEFAULT_MAX_ROUNDS
    assert result.manifest_path.is_file()

    extension = open_lfm2_circuit_trace_cache(result.extension_cache_path, verify=True)
    assert extension.cache_key == result.extension_cache_key
    assert extension.parent_fingerprint == lane.parent_fingerprint
    assert extension.batch_count == result.batch_count == len(FIT_PROMPTS)
    assert extension.corpus_fingerprint != lane.base_cache.corpus_fingerprint

    capture = result.manifest["capture"]
    assert capture["batch_count"] == len(FIT_PROMPTS)
    assert capture["already_sealed_in_prior_evidence"] == ()
    for entry in capture["batches"]:
        # Each batch is one child rollout: a >=3-token prefill plus the child's
        # own greedy continuation, so every visited prefix is covered once.
        assert entry["prompt_token_count"] >= 3
        assert entry["forced_decode_token_count"] >= 1
        assert entry["forced_decode_truncated"] is False
        assert len(entry["visited_prefix_row_ids"]) == (
            entry["forced_decode_token_count"] + 1
        )

    covered = {
        row_id
        for entry in capture["batches"]
        for row_id in entry["visited_prefix_row_ids"]
    }
    rollout = result.manifest["rollout"]
    assert rollout["child_generate_calls"] == len(FIT_PROMPTS)
    assert rollout["visited_prefix_row_count"] == len(covered)
    assert rollout["rejected_prompts"] == ()
    assert Path(rollout["generation_cache_path"]).is_file()

    references = result.references()
    assert references.as_tuple() == (
        lane.base_cache.path,
        result.extension_cache_path,
    )
    assert len(references) == 2
    caches = MODULE.open_lfm2_on_policy_evidence_caches(references)
    assert [value.cache_key for value in caches] == [
        lane.base_cache.cache_key,
        extension.cache_key,
    ]


def test_sealed_batches_are_the_child_visited_trajectories(lane, tmp_path) -> None:
    """The parent is evaluated on states the child actually reaches."""

    result = _run(lane, tmp_path / "rounds")
    child = MODULE.load_lfm2_direct_local_model(
        lane.child_directory, device="cpu", dtype=torch.float32
    )
    tokenizer = MODULE.load_lfm2_direct_local_tokenizer(lane.child_directory)
    expected = set()
    with torch.inference_mode():
        for prompt in FIT_PROMPTS:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            generated = child.generate(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                attention_mask=torch.ones((1, len(prompt_ids)), dtype=torch.long),
                max_new_tokens=4,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            row = [int(value) for value in generated[0].tolist()]
            expected.add((tuple(prompt_ids), tuple(row[len(prompt_ids) :])))

    cache = open_lfm2_circuit_trace_cache(result.extension_cache_path, verify=False)
    sealed = set()
    for index in range(cache.batch_count):
        batch = cache.load_batch(index)
        sealed.add(
            (
                tuple(int(value) for value in batch.input_ids[0].tolist()),
                tuple(int(value) for value in batch.forced_decode_ids[0].tolist()),
            )
        )
    assert sealed == expected


def test_manifest_reloads_verifies_and_is_idempotent(lane, tmp_path) -> None:
    output_root = tmp_path / "rounds"
    first = _run(lane, output_root)

    reloaded = MODULE.load_lfm2_on_policy_evidence_round(
        first.manifest_path, verify_extension_cache=True
    )
    assert reloaded.manifest_sha256 == first.manifest_sha256
    assert reloaded.request_sha256 == first.request_sha256
    assert reloaded.extension_cache_path == first.extension_cache_path

    # A directory also resolves to the sealed manifest.
    assert (
        MODULE.load_lfm2_on_policy_evidence_round(first.round_directory).manifest_sha256
        == first.manifest_sha256
    )

    # Re-running the identical request reuses the sealed round; it must not
    # execute the parent again.
    second = _run(lane, output_root)
    assert second.manifest_sha256 == first.manifest_sha256

    references = MODULE.resolve_lfm2_on_policy_evidence_references(
        lane.base_cache.path,
        [first.manifest_path],
        expected_parent_fingerprint=lane.parent_fingerprint,
    )
    assert references.as_tuple() == (lane.base_cache.path, first.extension_cache_path)


def test_tampered_manifest_fails_closed(lane, tmp_path) -> None:
    result = _run(lane, tmp_path / "rounds")
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    document["counters"]["backward_calls"] = 1
    result.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundIntegrityError):
        MODULE.load_lfm2_on_policy_evidence_round(result.manifest_path)


def test_conflicting_request_on_the_same_round_directory_fails_closed(
    lane, tmp_path
) -> None:
    output_root = tmp_path / "rounds"
    _run(lane, output_root)
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundConflictError):
        _run(lane, output_root, fit_prompts=FIT_PROMPTS[:2])


def test_overlapping_fit_prompt_fails_closed(lane, tmp_path) -> None:
    output_root = tmp_path / "rounds"
    manual = _write_manual_manifest(
        tmp_path / "manual-overlap.json",
        (*MANUAL_PROMPTS, FIT_PROMPTS[1]),
    )

    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundDisjointnessError) as failure:
        _run(lane, output_root, manual_prompt_manifest_path=manual)
    assert "reserved manual-review prompts" in str(failure.value)
    assert not (output_root / "round-000").exists()


def test_overlap_is_detected_through_whitespace_normalization(lane, tmp_path) -> None:
    manual = tmp_path / "manual-rows.json"
    manual.write_text(
        json.dumps({"prompts": [{"id": "m0", "prompt": f"  {FIT_PROMPTS[0]}  "}]}),
        encoding="utf-8",
    )
    loaded = MODULE.load_manual_review_prompt_manifest(manual)
    assert loaded.form == "review-prompt-rows"
    assert loaded.prompt_count == 1

    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundDisjointnessError):
        _run(lane, tmp_path / "rounds", manual_prompt_manifest_path=manual)


def test_sealed_manual_manifest_form_is_authenticated(tmp_path) -> None:
    digests = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in MANUAL_PROMPTS
    ]
    body = {
        "schema": MODULE.MANUAL_PROMPT_MANIFEST_SCHEMA,
        "prompt_count": len(digests),
        "prompt_sha256": digests,
    }
    sealed = tmp_path / "sealed.json"
    sealed.write_text(
        json.dumps(
            {
                **body,
                "manifest_sha256": hashlib.sha256(
                    json.dumps(
                        body,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    loaded = MODULE.load_manual_review_prompt_manifest(sealed)
    assert loaded.form == "sealed-digest-manifest"
    assert loaded.digests == frozenset(digests)

    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps({**body, "manifest_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundIntegrityError):
        MODULE.load_manual_review_prompt_manifest(broken)


def test_second_round_respects_max_rounds(lane, tmp_path) -> None:
    output_root = tmp_path / "rounds"
    first = _run(lane, output_root)

    # Round 1 uses the refolded child, so its on-policy states are new.
    second_child = tmp_path / "child-b"
    _publish_child(second_child, seed=929)
    second = _run(
        lane,
        output_root,
        child_artifact_directory=second_child,
        round_index=1,
        previous_round_manifest_paths=[first.manifest_path],
    )
    assert second.round_index == 1
    assert second.extension_cache_path != first.extension_cache_path
    assert second.manifest["round"]["previous_rounds"][0]["manifest_sha256"] == (
        first.manifest_sha256
    )

    references = MODULE.resolve_lfm2_on_policy_evidence_references(
        lane.base_cache.path,
        [first.manifest_path, second.manifest_path],
    )
    assert references.as_tuple() == (
        lane.base_cache.path,
        first.extension_cache_path,
        second.extension_cache_path,
    )

    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundBudgetError) as failure:
        _run(
            lane,
            output_root,
            child_artifact_directory=second_child,
            round_index=2,
            previous_round_manifest_paths=[
                first.manifest_path,
                second.manifest_path,
            ],
        )
    assert "max_rounds=2" in str(failure.value)
    assert not (output_root / "round-002").exists()


def test_evidence_already_sealed_is_never_regenerated(lane, tmp_path) -> None:
    first = _run(lane, tmp_path / "rounds")

    # The round-0 extension cache now contains exactly this child's on-policy
    # trajectories.  Treating it as the base cache leaves no new evidence.
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundError) as failure:
        _run(
            lane,
            tmp_path / "rounds-repeat",
            base_cache_directory=first.extension_cache_path,
        )
    assert "closed path" in str(failure.value)


def test_zero_parent_forward_and_single_capture_pass_accounting(
    lane, tmp_path, monkeypatch
) -> None:
    forwards = {"calls": 0}

    def count(_module, _inputs, _output) -> None:
        forwards["calls"] += 1

    handle = lane.parent.register_forward_hook(count)
    observed: list[dict[str, object]] = []
    real_build = MODULE.build_lfm2_circuit_trace_cache

    def spy(cache_root, model, calibration_batches, **kwargs):
        batches = tuple(calibration_batches)
        observed.append(
            {
                "parent_forwards_before": forwards["calls"],
                "batch_count": len(batches),
                "model_is_parent": model is lane.parent,
            }
        )
        return real_build(cache_root, model, batches, **kwargs)

    monkeypatch.setattr(MODULE, "build_lfm2_circuit_trace_cache", spy)
    try:
        result = _run(lane, tmp_path / "rounds")
    finally:
        handle.remove()

    assert len(observed) == 1
    assert observed[0]["model_is_parent"] is True
    # The child-only rollout phase executed zero parent forwards.
    assert observed[0]["parent_forwards_before"] == 0
    assert observed[0]["batch_count"] == result.batch_count
    # The single sanctioned capture is the only parent execution of the round.
    assert forwards["calls"] > 0

    counters = result.manifest["counters"]
    assert counters["parent_capture_invocations"] == 1
    assert counters["parent_capture_passes_per_batch"] == 1
    assert counters["parent_capture_batches"] == result.batch_count
    assert counters["parent_forward_calls_during_child_rollout"] == 0
    assert counters["child_generate_calls"] == len(FIT_PROMPTS)
    assert counters["optimizer_constructed"] is False
    assert counters["optimizer_steps"] == 0
    assert counters["backward_calls"] == 0
    assert counters["training_steps"] == 0
    assert counters["network_calls"] == 0
    assert counters["grad_enabled_during_round"] is False
    assert counters["parent_state_unchanged"] is True
    assert counters["child_state_unchanged"] is True

    assert all(
        parameter.grad is None for parameter in lane.parent.parameters()
    )
    assert fingerprint_lfm2_parent(lane.parent).identity_sha256 == (
        lane.parent_fingerprint
    )


def test_rollout_config_rejects_sampling_and_unbounded_context() -> None:
    with pytest.raises(ValueError, match="greedy"):
        MODULE.Lfm2OnPolicyRolloutConfig(greedy=False)
    with pytest.raises(ValueError, match="max_context_tokens"):
        MODULE.Lfm2OnPolicyRolloutConfig(max_new_tokens=8, max_context_tokens=8)
    with pytest.raises(ValueError, match="max_prompts"):
        MODULE.Lfm2OnPolicyRolloutConfig(max_prompts=0)


def test_prompt_budget_and_parent_identity_fail_closed(lane, tmp_path) -> None:
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundBudgetError):
        _run(
            lane,
            tmp_path / "rounds-budget",
            rollout=MODULE.Lfm2OnPolicyRolloutConfig(
                max_prompts=2,
                max_new_tokens=4,
                max_context_tokens=64,
            ),
        )
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundIntegrityError):
        _run(lane, tmp_path / "rounds-parent", parent_fingerprint="a" * 64)


def test_capture_selection_must_match_the_sealed_base_cache(lane, tmp_path) -> None:
    with pytest.raises(MODULE.Lfm2OnPolicyEvidenceRoundIntegrityError) as failure:
        _run(
            lane,
            tmp_path / "rounds-selection",
            capture_config=Lfm2CircuitTraceCaptureConfig(
                sample_last_tokens=2,
                max_decode_steps=8,
                max_capture_bytes=64 * 1024 * 1024,
                logit_rtol=0.2,
                logit_atol=1.0,
                require_top1_agreement=False,
            ),
        )
    assert "same selection" in str(failure.value)


def test_an_unbridged_round_seals_no_target_preimage_and_says_so(
    lane, tmp_path
) -> None:
    """No bridge, no preimage — and the refusal names the manifest."""

    result = _run(lane, tmp_path / "rounds-unbridged")

    assert "target_preimage" not in result.manifest
    assert result.target_preimage is None
    assert result.carries_target_preimage is False
    assert result.parent_cache_input_mode == (
        MODULE.PARENT_CANONICAL_CAPTURE_INPUT_MODE
    )
    # The unbridged request is byte-identical to the pre-bridge revision, so
    # an already sealed round keeps its request digest and round directory.
    assert "tokenizer_bridge_artifact_sha256" not in result.manifest["request"]

    reason = MODULE.lfm2_on_policy_round_missing_target_preimage_reason(result)
    assert "seals no target-token preimage" in reason
    assert str(result.manifest_path) in reason

    extension = open_lfm2_circuit_trace_cache(
        result.extension_cache_path,
        verify=False,
    )
    with pytest.raises(
        MODULE.Lfm2OnPolicyEvidenceRoundBridgeError,
        match="seals no target-token preimage",
    ):
        MODULE.build_lfm2_on_policy_round_target_records(
            result,
            bridge=SimpleNamespace(artifact_sha256="a" * 64),
            extension_cache=extension,
        )


def test_a_bridge_must_be_a_verified_strict_artifact(lane, tmp_path) -> None:
    with pytest.raises(
        MODULE.Lfm2OnPolicyEvidenceRoundBridgeError,
        match="must be a StrictTokenizerBridgeArtifact",
    ):
        _run(
            lane,
            tmp_path / "rounds-fake-bridge",
            tokenizer_bridge=SimpleNamespace(artifact_sha256="a" * 64),
        )


# --- cached-decode admissibility screen -------------------------------------
#
# A degenerate child can visit prefixes where the parent's KV-cached decode
# logits genuinely leave the capture's envelope around the use_cache=False
# full-prefix recompute, at the parent's own runtime precision.  The capture is
# right to refuse to seal those.  These tests pin the round's behaviour around
# that refusal: off by default, and when enabled it excludes exactly the
# refused trajectories, records them, and tells the truth about the extra
# parent forwards it spent deciding.


def _admissibility_reports(models, *, inadmissible=()):
    """Verdicts shaped exactly like screen_lfm2_cached_decode_admissibility."""

    from neural_forge.hf_lfm2_circuit_trace_cache import _prepare_batches

    prepared = _prepare_batches(torch, list(models), _capture_config())
    reports = []
    for position, batch in enumerate(prepared):
        refused = position in inadmissible
        failure = {
            "phase": "decode.001",
            "prefix_length": int(batch.input_ids.shape[1]) + 2,
            "admissible": False,
            "capture_error": (
                "decode.001 cached logits differ from use_cache=False "
                "full-prefix logits (maximum error 1.90625)."
            ),
            "max_abs_error": 1.90625,
            "elementwise_tolerance_violations": 190,
            "compared_elements": 65536,
            "cached_logit_at_max_abs_error": -6.40625,
            "full_prefix_logit_at_max_abs_error": -8.3125,
            "logit_dtype": "bfloat16",
        }
        reports.append(
            {
                "schema": MODULE.CACHED_DECODE_ADMISSIBILITY_SCHEMA,
                "index": batch.index,
                "input_sha256": batch.input_sha256,
                "admissible": not refused,
                "cached_decode_verified": not refused,
                "decode_trace_mode": "cached_verified",
                "parent_forward_calls": 10,
                "phases": [
                    {
                        "phase": "cached prefill",
                        "prefix_length": int(batch.input_ids.shape[1]),
                        "admissible": True,
                        "max_abs_error": 0.0,
                        "elementwise_tolerance_violations": 0,
                        "compared_elements": 65536,
                        "cached_logit_at_max_abs_error": 0.0,
                        "full_prefix_logit_at_max_abs_error": 0.0,
                        "logit_dtype": "bfloat16",
                    },
                    *([failure] if refused else []),
                ],
                "first_inadmissible_phase": failure if refused else None,
            }
        )
    return tuple(reports)


def test_screen_reports_every_batch_instead_of_failing_the_capture_closed(
    lane,
) -> None:
    """The screen answers per batch where the capture refuses the whole set."""

    from neural_forge import hf_lfm2_circuit_trace_cache as CACHE

    batches: list[dict] = [
        {
            "input_ids": torch.tensor([[2, 4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[11, 12]], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[2, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[13, 14]], dtype=torch.long),
        },
    ]
    reports = MODULE.screen_lfm2_cached_decode_admissibility(
        lane.parent,
        batches,
        config=lane.capture_config,
        input_device="cpu",
    )
    assert len(reports) == 2
    for report in reports:
        assert report["admissible"] is True
        assert report["first_inadmissible_phase"] is None
        # One uncached and one cached prefill, then two forwards per decode step.
        assert report["parent_forward_calls"] == 2 + 2 * 2
        assert [value["phase"] for value in report["phases"]] == [
            "cached prefill",
            "decode.000",
            "decode.001",
        ]

    # The screen must not raise where the capture would: it reports the refusal
    # and hands the caller the measured evidence behind it.  The verdict is
    # taken from the collector's own comparison, which is what is stubbed here.
    original = MODULE._compare_lfm2_trace_logits
    assert original is CACHE._compare_logits

    def refuse(torch_module, cached, full_prefix, *, config, name):
        if name == "decode.001":
            raise CACHE.Lfm2CircuitTraceCaptureError(
                f"{name} cached logits differ from use_cache=False full-prefix "
                "logits (maximum error 1.90625)."
            )
        return original(torch_module, cached, full_prefix, config=config, name=name)

    MODULE._compare_lfm2_trace_logits = refuse
    try:
        refused = MODULE.screen_lfm2_cached_decode_admissibility(
            lane.parent,
            batches,
            config=lane.capture_config,
            input_device="cpu",
        )
    finally:
        MODULE._compare_lfm2_trace_logits = original

    assert [value["admissible"] for value in refused] == [False, False]
    for report in refused:
        failure = report["first_inadmissible_phase"]
        assert failure["phase"] == "decode.001"
        assert "maximum error" in failure["capture_error"]
        assert failure["compared_elements"] > 0
        # The chain keeps advancing after a refusal so later phases are still
        # judged on the cache state the capture would actually hold.
        assert [value["phase"] for value in report["phases"]] == [
            "cached prefill",
            "decode.000",
            "decode.001",
        ]


def test_the_screen_must_not_live_in_the_collector(lane) -> None:
    """The collector's source digest is part of every sealed cache's identity.

    ``implementation_identity`` hashes ``hf_lfm2_circuit_trace_cache.py`` itself,
    a round refuses an extension cache whose ``implementation_sha256`` differs
    from its base cache's, so *any* edit to that file -- even a purely additive
    one -- retires every sealed cache captured before it.  The screen therefore
    has to live here and import the collector's helpers.
    """

    from neural_forge import hf_lfm2_circuit_trace_cache as CACHE

    assert not hasattr(CACHE, "screen_lfm2_cached_decode_admissibility")
    assert (
        MODULE.screen_lfm2_cached_decode_admissibility.__module__
        == "neural_forge.hf_lfm2_on_policy_evidence_round"
    )
    assert (
        MODULE.CACHED_DECODE_ADMISSIBILITY_SCHEMA
        == "neural-forge-lfm2-cached-decode-admissibility-v1"
    )


def test_screen_is_off_by_default_and_the_round_still_fails_closed(
    lane, tmp_path
) -> None:
    result = _run(lane, tmp_path / "rounds-unscreened")
    admissibility = result.manifest["capture"]["cached_decode_admissibility"]

    assert admissibility["screened"] is False
    assert admissibility["excluded_batch_count"] == 0
    assert admissibility["screen_parent_forward_calls"] == 0
    assert result.manifest["parent"]["executed_only_inside_sealed_capture"] is True
    assert (
        result.manifest["parent"]["executed_in_cached_decode_admissibility_screen"]
        is False
    )


def test_disabled_screen_leaves_the_sealed_rollout_digest_untouched() -> None:
    """An unscreened round keeps the digests every already sealed round has."""

    unscreened = MODULE.Lfm2OnPolicyRolloutConfig(max_prompts=8, max_new_tokens=4)
    body = unscreened.as_dict()
    assert "screen_cached_decode_admissibility" not in body

    screened = MODULE.Lfm2OnPolicyRolloutConfig(
        max_prompts=8,
        max_new_tokens=4,
        screen_cached_decode_admissibility=True,
    )
    screened_body = screened.as_dict()
    assert screened_body["screen_cached_decode_admissibility"] is True
    assert screened_body["rollout_sha256"] != body["rollout_sha256"]


def test_screened_round_excludes_and_records_the_refused_trajectory(
    lane, tmp_path, monkeypatch
) -> None:
    captured = {}

    def screen(model, models, *, config, input_device):
        captured["count"] = len(models)
        return _admissibility_reports(models, inadmissible={1})

    monkeypatch.setattr(MODULE, "screen_lfm2_cached_decode_admissibility", screen)

    result = _run(
        lane,
        tmp_path / "rounds-screened",
        rollout=MODULE.Lfm2OnPolicyRolloutConfig(
            max_prompts=8,
            max_new_tokens=4,
            max_context_tokens=64,
            seed=0,
            max_capture_batches=16,
            screen_cached_decode_admissibility=True,
        ),
    )

    assert captured["count"] == len(FIT_PROMPTS)
    admissibility = result.manifest["capture"]["cached_decode_admissibility"]
    assert admissibility["screened"] is True
    assert admissibility["candidate_batch_count"] == len(FIT_PROMPTS)
    assert admissibility["admissible_batch_count"] == len(FIT_PROMPTS) - 1
    assert admissibility["excluded_batch_count"] == 1
    assert admissibility["screen_parent_forward_calls"] == 10 * len(FIT_PROMPTS)

    (excluded,) = admissibility["excluded"]
    assert excluded["reason"] == "cached_decode_not_verifiable_against_full_prefix"
    assert excluded["first_inadmissible_phase"]["phase"] == "decode.001"
    assert excluded["first_inadmissible_phase"]["max_abs_error"] == 1.90625
    assert excluded["forced_decode_token_ids"]

    # The refused trajectory is gone from the sealed evidence, and only it.
    assert result.batch_count == len(FIT_PROMPTS) - 1
    assert result.manifest["capture"]["batch_count"] == len(FIT_PROMPTS) - 1
    sealed = {
        value["trajectory_sha256"] for value in result.manifest["capture"]["batches"]
    }
    assert excluded["trajectory_sha256"] not in sealed
    assert len(sealed) == len(FIT_PROMPTS) - 1

    extension = open_lfm2_circuit_trace_cache(result.extension_cache_path, verify=True)
    assert extension.batch_count == len(FIT_PROMPTS) - 1

    # Screening costs parent forwards outside the sealed capture, and the
    # manifest says so rather than repeating the unqualified claim.
    parent = result.manifest["parent"]
    assert parent["executed_only_inside_sealed_capture"] is False
    assert parent["executed_in_cached_decode_admissibility_screen"] is True
    assert parent["screen_parent_forward_calls"] == 10 * len(FIT_PROMPTS)
    assert parent["unchanged_by_round"] is True


def test_screen_refuses_a_round_with_too_few_verifiable_trajectories(
    lane, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "screen_lfm2_cached_decode_admissibility",
        lambda model, models, *, config, input_device: _admissibility_reports(
            models, inadmissible=set(range(len(models)))
        ),
    )

    with pytest.raises(
        MODULE.Lfm2OnPolicyEvidenceRoundError,
        match="too degenerate to extend this evidence set",
    ):
        _run(
            lane,
            tmp_path / "rounds-all-refused",
            rollout=MODULE.Lfm2OnPolicyRolloutConfig(
                max_prompts=8,
                max_new_tokens=4,
                max_context_tokens=64,
                seed=0,
                max_capture_batches=16,
                screen_cached_decode_admissibility=True,
            ),
        )


def test_screen_refuses_a_verdict_for_a_batch_the_round_did_not_submit(
    lane, tmp_path, monkeypatch
) -> None:
    def screen(model, models, *, config, input_device):
        reports = list(_admissibility_reports(models))
        reports[0] = {**reports[0], "input_sha256": "f" * 64}
        return tuple(reports)

    monkeypatch.setattr(MODULE, "screen_lfm2_cached_decode_admissibility", screen)

    with pytest.raises(
        MODULE.Lfm2OnPolicyEvidenceRoundIntegrityError,
        match="a batch the round did not submit",
    ):
        _run(
            lane,
            tmp_path / "rounds-mismatched-screen",
            rollout=MODULE.Lfm2OnPolicyRolloutConfig(
                max_prompts=8,
                max_new_tokens=4,
                max_context_tokens=64,
                seed=0,
                max_capture_batches=16,
                screen_cached_decode_admissibility=True,
            ),
        )
