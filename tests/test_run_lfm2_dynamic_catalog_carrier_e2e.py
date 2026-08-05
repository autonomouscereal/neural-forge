"""Minutes-scale dress rehearsal for the exact4x integrated-carrier lane.

``tests/test_run_lfm2_dynamic_catalog.py`` proves the dense catalog lane with a
real tiny parent, and the carrier lane only has mocked component tests.  Real
carrier runs (``upload_staging/nf-launch-integrated-carrier-exact4x-r*.sh``)
therefore reach production hardware before anything executes the whole wiring,
and run r7 died inside the fold on a progress-callback field collision that a
toy full-pipeline run would have caught in minutes.

This module builds every carrier-lane input at toy scale on CPU:

* a real ByteLevel-BPE donor tokenizer package,
* a real ``Lfm2ForCausalLM`` parent that keeps the production residual width,
  attention geometry, and short-conv width lattice (see ``_PARENT_HIDDEN_SIZE``),
* a real strict tokenizer bridge compiled by
  ``experiments/compile_lfm2_target_tokenizer_bridge.py``,
* target-canonical and one-row-mapped donor calibrations,
* one sealed parent circuit-trace cache captured on the mapped donor IDs,
* the sealed canonical-parent and mapped-parent whole-output envelopes,
* a sealed routed manual-review prompt artifact,

and then runs the same ``run_production_catalog`` entrypoint ``main()`` uses
with the r10 launcher's flag set.

Two production pins keep the *fully* routed configuration out of reach at toy
scale (sealed 4096-row/H816 ABI and the registered 229,693,184-element parent);
they are recorded in ``NOTES-1E.md`` at the repository root together with the
defects this rehearsal found.  The reachable configuration is exercised as deep
as the lane currently goes — complete structural fold, complete closed loop with
moving-gauge-v3 reclosure, the cached-parent trajectory gate, and the dual
original/mapped whole-output behavior gate — and the routed configuration is
exercised up to its first production pin.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

import neural_forge.hf_lfm2_on_policy_evidence_round as MODULE_ROUND  # noqa: E402
from experiments import compile_lfm2_target_tokenizer_bridge as bridge_compiler  # noqa: E402
from experiments import run_lfm2_dynamic_catalog as MODULE  # noqa: E402
from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.dynamic_counted_child_state import (  # noqa: E402
    ContinuousChildStateAllocation,
    CountedChildStateError,
)
from neural_forge.dynamic_lfm2_behavior import (  # noqa: E402
    load_lfm2_cached_whole_output_baseline,
    seal_lfm2_cached_whole_output_envelope,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    derive_lfm2_automatic_mlp_width_policy,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


# ``_build_cached_global_heterogeneous_selection_input`` never threads the
# short-conv width or attention-geometry catalogs through the CLI, so every
# child the runner can plan uses ShortConv internal widths (704, 1024) and GQA
# options (12, 6, 64) / (16, 8, 64).  A toy parent must therefore keep the
# production residual width and head geometry; only depth, vocabulary, and MLP
# width shrink.  See NOTES-1E.md, blocker 3.
_PARENT_HIDDEN_SIZE = 1024
_PARENT_QUERY_HEADS = 16
_PARENT_KEY_VALUE_HEADS = 8
_PARENT_HEAD_DIM = 64
_PARENT_INTERMEDIATE_SIZE = 64
_PARENT_LAYER_TYPES = ("conv", "full_attention")
_PARENT_CONV_KERNEL = 3
_TARGET_HIDDEN_SIZE = 32
_SHORT_CONV_INTERNAL_WIDTHS = (704, 1024)
_ATTENTION_HEAD_OPTIONS = ((12, 6, 64), (16, 8, 64))

_GENERATION_LIMIT = 4
_MAX_DECODE_STEPS = 6
_SAMPLE_LAST_TOKENS = 4
_CAPTURE_BYTES = 256 * 1024 * 1024

_CALIBRATION_CORPUS = (
    "the quick brown fox jumps over the lazy dog",
    "a folded child keeps the parent behavior",
    "counted route tensors stay outside the dense body",
    "tiny models fold quickly on a local cpu",
    "the parent tokenizer bridges to a smaller target",
    "one sealed cache feeds every fold phase",
)

# (id, split, prompt text, forced-decode text)
_TRAJECTORIES = (
    ("traj-000", "selection", "the quick brown fox", " the"),
    ("traj-001", "fit", "a folded child keeps", " on"),
    ("traj-002", "validation", "counted route tensors stay", " at"),
    ("traj-003", "fit", "tiny models fold quickly", " to"),
)

# (id, prompt, category, expected response, parent response)
_BEHAVIOR_CASES = (
    ("case-000", "2 plus 2 is", "math", "4", "4"),
    ("case-001", "3 plus 4 is", "math", "7", "7"),
)

_MANUAL_PROMPTS = (
    ("manual-000", "reason", "counted route tensors stay"),
    ("manual-001", "safety", "my kitchen smells like gas"),
)

# Publication packages the child checkpoint and the sealed tokenizer into one
# directory, then re-runs whole-directory bridge authentication over it.
# NOTES-1E.md, blocker 8.
_PACKAGED_BRIDGE_BLOCKER = "Target tokenizer files differ from the sealed artifact."

# Failure messages this rehearsal already reached and reported.  Each entry is
# documented in NOTES-1E.md with the exact file, line, and minimal repair.  An
# unlisted failure means the lane broke somewhere new and the test fails.
_KNOWN_TOY_SCALE_BLOCKERS = (
    # NOTES-1E.md, blocker 4 (fixed in the working tree, still live at
    # 9c53a6e8): the bridge-aligned parent rows were recomputed and then
    # discarded before the span-mapped trajectory comparison.
    "Parent and hybrid-child trajectory paths, splits, or corpus differ.",
    # NOTES-1E.md, blocker 5: the per-block FP64 Gram is required to be
    # bitwise symmetric straight out of a threaded CPU GEMM.
    "Snapshot contribution receipt requires its exact symmetric FP64 Gram.",
    # NOTES-1E.md, blocker 8: `DirectLocalArtifactIO.save` writes the child
    # checkpoint into the publication directory and then asks the sealed
    # bridge to re-authenticate that whole directory as a tokenizer package.
    _PACKAGED_BRIDGE_BLOCKER,
)

_REQUIRED_STAGE_PREFIX = (
    "run_started",
    "directories_ready",
    "local_artifacts_verified",
    "parent_verified",
    "parent_cache_verified",
    "tokenizer_bridge_started",
    "tokenizer_bridge_live_tokenizers_authenticated",
    "tokenizer_bridge_complete",
    "tokenizer_bridge_verified",
    "carrier_verified",
    "global_frontier_planning_started",
    "parent_rows_materialization_started",
    "parent_rows_materialized",
    "wiring_planned",
    "parent_identity_digest_started",
    "system_fold_started",
)

# The fold itself, which no other test in this suite executes for the carrier
# lane.  Run r7 died inside this window on a relayed progress field, so these
# stages are required, not optional telemetry.
_REQUIRED_FOLD_STAGES = (
    "dense_structural_short_conv_action_evidence_completed",
    "dense_structural_token_transport_completed",
    "dense_structural_fold_completed",
    "dense_closed_loop_started",
    "dense_closed_loop_merge_capture_policy",
    "child_closed_loop_layer_producer_installed",
    "child_closed_loop_layer_consumer_installed",
    "child_closed_loop_layer_downstream_installed",
    "child_closed_loop_body_reclosure_started",
    "child_closed_loop_body_reclosure_installed",
    "child_closed_loop_body_reclosure_completed",
    "child_closed_loop_final_normalization_completed",
    "child_closed_loop_output_interface_completed",
    "dense_closed_loop_completed",
)


def _progress_rows(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") == MODULE.PROGRESS_JOURNAL_SCHEMA:
            rows.append(row)
    return rows


def _stages(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["stage"]) for row in rows]


def _assert_ordered_stages(
    observed: Sequence[str],
    required: Sequence[str],
    *,
    context: str = "",
) -> None:
    """Require every named stage, in order, allowing telemetry in between."""

    remaining = list(observed)
    for stage in required:
        if stage not in remaining:
            raise AssertionError(
                f"progress ledger never reached {stage!r}; observed "
                f"{observed!r}. {context}"
            )
        remaining = remaining[remaining.index(stage) + 1 :]


def _write_canonical_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="ascii",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_donor_tokenizer(directory: Path) -> Any:
    """Save a real ByteLevel-BPE donor package the bridge compiler accepts."""

    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: index for index, token in enumerate(alphabet)}
    merges: list[tuple[str, str]] = []
    for left, right in (
        ("t", "h"),
        ("th", "e"),
        ("o", "u"),
        ("i", "n"),
        ("e", "r"),
        ("a", "n"),
        ("r", "e"),
        ("o", "n"),
        ("a", "t"),
        ("e", "n"),
        ("Ġ", "t"),
        ("Ġt", "h"),
        ("Ġth", "e"),
        ("Ġ", "a"),
        ("Ġ", "o"),
        ("Ġ", "c"),
        ("Ġ", "f"),
        ("Ġ", "b"),
        ("Ġ", "p"),
        ("Ġ", "s"),
    ):
        merged = left + right
        if merged in vocabulary:
            continue
        merges.append((left, right))
        vocabulary[merged] = len(vocabulary)
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocabulary, merges=merges)
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    donor = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    directory.mkdir(parents=True, exist_ok=True)
    donor.backend_tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )
    return residual.load_local_tokenizer(directory)


def _build_tiny_parent(vocabulary_rows: int) -> Any:
    torch.manual_seed(20260731)
    parent = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=vocabulary_rows,
            hidden_size=_PARENT_HIDDEN_SIZE,
            intermediate_size=_PARENT_INTERMEDIATE_SIZE,
            num_hidden_layers=len(_PARENT_LAYER_TYPES),
            num_attention_heads=_PARENT_QUERY_HEADS,
            num_key_value_heads=_PARENT_KEY_VALUE_HEADS,
            head_dim=_PARENT_HEAD_DIM,
            layer_types=list(_PARENT_LAYER_TYPES),
            conv_L_cache=_PARENT_CONV_KERNEL,
            conv_bias=False,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    ).eval()
    # The installability evidence the frontier consumes needs the explicit
    # Q/K/softmax/V path, exactly like the production preparation command.
    parent.config._attn_implementation = "eager"
    return parent


def _exact_toy_body_budget(target_vocabulary_rows: int) -> int:
    """Return the smallest exact whole-child budget the H32 lattice can hit."""

    for target in range(200_000, 900_000):
        try:
            derive_lfm2_automatic_mlp_width_policy(
                layer_types=_PARENT_LAYER_TYPES,
                source_widths=[_PARENT_INTERMEDIATE_SIZE] * len(_PARENT_LAYER_TYPES),
                short_conv_internal_widths=_SHORT_CONV_INTERNAL_WIDTHS,
                attention_head_options=_ATTENTION_HEAD_OPTIONS,
                target_hidden_size=_TARGET_HIDDEN_SIZE,
                target_vocab_size=target_vocabulary_rows,
                exact_target_unique_parameters=target,
                conv_kernel_size=_PARENT_CONV_KERNEL,
                discover_exact_short_conv_writer_ranks=True,
            )
        except Exception:
            continue
        return target
    raise AssertionError("no exact toy dense-body budget exists for H32.")


@dataclass(frozen=True)
class _CarrierLanePack:
    """Every local input one integrated-carrier launcher passes the runner."""

    root: Path
    parent_directory: Path
    donor_directory: Path
    target_directory: Path
    target_calibration: Path
    donor_calibration: Path
    sealed_parent_cache: Path
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    mapped_parent_outputs: Path
    mapped_parent_outputs_sha256: str
    bridge_artifact_sha256: str
    bridge_profile_sha256: str
    manual_prompts: Path
    manual_prompts_sha256: str
    parent_unique_parameters: int
    donor_row_capacity: int
    target_row_capacity: int
    exact_body_parameters: int


def _build_carrier_lane_pack(root: Path) -> _CarrierLanePack:
    """Build one real toy-scale copy of the r10 launcher's evidence pack."""

    root.mkdir(parents=True, exist_ok=True)
    donor_directory = root / "donor-tokenizer"
    donor = _build_donor_tokenizer(donor_directory)
    donor_row_capacity = len(donor) + 4

    parent = _build_tiny_parent(donor_row_capacity)
    parent_directory = root / "parent"
    residual.atomic_save_local_checkpoint(
        parent,
        parent_directory,
        output_root=root,
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (parent_directory / name).write_bytes(
            (donor_directory / name).read_bytes()
        )

    corpus = root / "bridge-corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": value}) for value in _CALIBRATION_CORPUS)
        + "\n",
        encoding="utf-8",
    )
    target_directory = root / "target-tokenizer"
    compilation_receipt = bridge_compiler.compile_target_tokenizer(
        donor_tokenizer_directory=donor_directory,
        output_directory=target_directory,
        receipt_path=root / "target-tokenizer-receipt.json",
        target_vocab_size=len(donor) - 8,
        donor_row_capacity=donor_row_capacity,
        jsonl_paths=(corpus,),
        parquet_paths=(),
        maximum_texts=64,
        maximum_characters=4096,
    )
    bridge_artifact_sha256 = str(compilation_receipt["bridge_artifact_sha256"])
    bridge_profile_sha256 = str(compilation_receipt["bridge_profile_sha256"])
    target_row_capacity = int(compilation_receipt["target_vocabulary_size"])

    pack = root / "pack"
    target_tokenizer = residual.load_local_tokenizer(target_directory)
    artifact = load_strict_tokenizer_bridge_artifact(
        target_directory,
        donor_tokenizer=donor,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=bridge_artifact_sha256,
    )

    target_rows = []
    for record_id, split, prompt, forced in _TRAJECTORIES:
        input_ids = MODULE._tokenizer_input_ids(
            target_tokenizer,
            prompt,
            add_special_tokens=True,
        )
        forced_ids = MODULE._tokenizer_input_ids(
            target_tokenizer,
            forced,
            add_special_tokens=False,
        )
        assert len(forced_ids) <= _MAX_DECODE_STEPS
        target_rows.append(
            {
                "schema": "neural-forge-lfm2-target-tokenized-trajectory-v1",
                "id": record_id,
                "split": split,
                "target_tokenization_mode": "tokenizer-call-add-special-tokens",
                "target_forced_decode_tokenization_mode": (
                    "tokenizer-call-no-special-tokens"
                ),
                "target_input_texts": [prompt],
                "target_forced_decode_texts": [forced],
                "input_ids": list(input_ids),
                "attention_mask": [1] * len(input_ids),
                "forced_decode_ids": list(forced_ids),
            }
        )
    target_calibration_path = pack / "target_tokenized_calibration.json"
    _write_canonical_json(target_calibration_path, target_rows)
    target_calibration = residual.load_calibration_batches(
        target_calibration_path
    )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        artifact,
        MODULE._target_tokenized_records(target_calibration),
        expected_artifact_sha256=bridge_artifact_sha256,
    )

    donor_rows = []
    for source, record in zip(target_rows, adapter.records):
        parent_batch = record.as_parent_cache_batch()
        donor_rows.append(
            {
                "schema": "neural-forge-lfm2-mapped-donor-trajectory-v1",
                "id": source["id"],
                "split": source["split"],
                "input_ids": [
                    int(value) for value in parent_batch["input_ids"][0].tolist()
                ],
                "attention_mask": [
                    int(value)
                    for value in parent_batch["attention_mask"][0].tolist()
                ],
                "forced_decode_ids": [
                    int(value)
                    for value in parent_batch["forced_decode_ids"][0].tolist()
                ],
            }
        )
    donor_calibration_path = pack / "mapped_donor_calibration.json"
    _write_canonical_json(donor_calibration_path, donor_rows)

    parent_cache = build_lfm2_circuit_trace_cache(
        pack / "parent_trace_cache",
        parent,
        tuple(record.as_parent_cache_batch() for record in adapter.records),
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=_SAMPLE_LAST_TOKENS,
            max_decode_steps=_MAX_DECODE_STEPS,
            max_capture_bytes=_CAPTURE_BYTES,
            logit_rtol=MODULE.DEFAULT_CAPTURE_LOGIT_RTOL,
            logit_atol=MODULE.DEFAULT_CAPTURE_LOGIT_ATOL,
            require_top1_agreement=False,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device=torch.device("cpu"),
    )

    donor_fingerprint = MODULE.fingerprint_local_tokenizer(donor_directory)
    behavior_rows_path = root / "canonical-parent-rows.json"
    _write_canonical_json(
        behavior_rows_path,
        [
            {
                "id": case_id,
                "prompt": prompt,
                "category": category,
                "expected_response": expected,
                "parent_output_text": response,
                "parent_output_ids": list(
                    donor(response, add_special_tokens=False)["input_ids"]
                )[:_GENERATION_LIMIT],
            }
            for case_id, prompt, category, expected, response in _BEHAVIOR_CASES
        ],
    )
    cached_parent_outputs = pack / "canonical_parent_outputs_bridge_bound.json"
    original_receipt = seal_lfm2_cached_whole_output_envelope(
        behavior_rows_path,
        cached_parent_outputs,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=donor_fingerprint,
        generation_limit=_GENERATION_LIMIT,
    )
    original_baseline = load_lfm2_cached_whole_output_baseline(
        cached_parent_outputs,
        expected_sha256=original_receipt["file_sha256"],
        expected_parent_cache_key=parent_cache.cache_key,
        expected_parent_state_fingerprint=parent_cache.parent_fingerprint,
        expected_tokenizer_fingerprint=donor_fingerprint,
        expected_generation_limit=_GENERATION_LIMIT,
        fragment_minimum_characters=1,
    )
    bridge_runtime = MODULE.resolve_production_tokenizer_bridge(
        SimpleNamespace(
            uses_tokenizer_bridge=True,
            target_tokenizer_directory=target_directory,
            target_tokenizer_calibration=target_calibration_path,
            tokenizer_bridge_artifact_sha256=bridge_artifact_sha256,
            tokenizer_bridge_profile_sha256=bridge_profile_sha256,
        ),
        donor_tokenizer=donor,
        parent_cache=parent_cache,
        calibration_splits=target_calibration.splits,
    )
    assert bridge_runtime is not None

    mapping = artifact.transport_profile.target_to_donor_row_indices
    mapped_rows = []
    with torch.inference_mode():
        for case in original_baseline.cases:
            target_ids = MODULE._tokenizer_input_ids(
                target_tokenizer,
                case.prompt,
                add_special_tokens=True,
            )
            mapped_ids = tuple(mapping[token_id] for token_id in target_ids)
            input_ids = torch.tensor([mapped_ids], dtype=torch.long)
            generated = parent.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=_GENERATION_LIMIT,
                do_sample=False,
                use_cache=True,
                pad_token_id=int(donor.pad_token_id),
            )
            continuation = [
                int(value)
                for value in generated[0, int(input_ids.shape[1]) :].tolist()
            ]
            mapped_rows.append(
                {
                    "id": case.case_id,
                    "prompt": case.prompt,
                    "mapped_parent_output_text": donor.decode(
                        continuation,
                        skip_special_tokens=True,
                    ),
                    "mapped_parent_output_ids": continuation,
                    "target_prompt_input_ids": list(target_ids),
                    "mapped_parent_input_ids": list(mapped_ids),
                }
            )
    mapped_rows_path = root / "mapped-parent-rows.json"
    _write_canonical_json(mapped_rows_path, mapped_rows)
    mapped_parent_outputs = pack / "mapped_parent_outputs.json"
    mapped_receipt = MODULE.seal_lfm2_tokenizer_bridge_mapped_parent_envelope(
        mapped_rows_path,
        mapped_parent_outputs,
        original_parent_baseline=original_baseline,
        tokenizer_bridge=bridge_runtime,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
    )

    manual_prompts = pack / "routed_manual_prompts.json"
    manual_prompts_sha256 = _write_canonical_json(
        manual_prompts,
        {
            "schema": "neural-forge-lfm2-routed-manual-prompts-v1",
            "prompts": [
                {"id": prompt_id, "category": category, "prompt": prompt}
                for prompt_id, category, prompt in _MANUAL_PROMPTS
            ],
        },
    )

    return _CarrierLanePack(
        root=root,
        parent_directory=parent_directory,
        donor_directory=donor_directory,
        target_directory=target_directory,
        target_calibration=target_calibration_path,
        donor_calibration=donor_calibration_path,
        sealed_parent_cache=Path(parent_cache.path),
        cached_parent_outputs=cached_parent_outputs,
        cached_parent_outputs_sha256=str(original_receipt["file_sha256"]),
        mapped_parent_outputs=mapped_parent_outputs,
        mapped_parent_outputs_sha256=str(mapped_receipt["file_sha256"]),
        bridge_artifact_sha256=bridge_artifact_sha256,
        bridge_profile_sha256=bridge_profile_sha256,
        manual_prompts=manual_prompts,
        manual_prompts_sha256=manual_prompts_sha256,
        parent_unique_parameters=residual.unique_parameter_count(parent),
        donor_row_capacity=donor_row_capacity,
        target_row_capacity=target_row_capacity,
        exact_body_parameters=_exact_toy_body_budget(target_row_capacity),
    )


def _load_pack_bridge(pack: _CarrierLanePack) -> Any:
    """Reload the pack's sealed strict bridge from its own local packages."""

    return load_strict_tokenizer_bridge_artifact(
        pack.target_directory,
        donor_tokenizer=residual.load_local_tokenizer(pack.donor_directory),
        target_tokenizer=residual.load_local_tokenizer(pack.target_directory),
        expected_artifact_sha256=pack.bridge_artifact_sha256,
    )


@pytest.fixture(scope="module")
def carrier_lane_pack(tmp_path_factory) -> _CarrierLanePack:
    """Build the toy carrier-lane inputs once for every rehearsal here."""

    return _build_carrier_lane_pack(
        tmp_path_factory.mktemp("lfm2-carrier-lane-pack")
    )


def _lane_arguments(
    pack: _CarrierLanePack,
    run: Path,
    *,
    carrier_routes: bool,
    residual_basis_mode: str | None = None,
) -> list[str]:
    """Mirror nf-launch-integrated-carrier-exact4x-r10.sh at toy scale.

    ``residual_basis_mode`` defaults to the mode the routing itself implies.
    Naming it explicitly lets an unrouted lane compose the residual basis and
    the margin observability the carrier production pin requires, which is the
    only part of that pin the toy ABI can actually reach.
    """

    residual_basis_mode = residual_basis_mode or (
        "circuit_balanced" if carrier_routes else "weight_complete"
    )
    # Production keeps the derived topology and balanced-basis phase caches in
    # stable cross-run artifact roots; only per-run trees live beside them.
    stable = run.parent / "stable"
    arguments = [
        "--parent-model",
        str(pack.parent_directory),
        "--tokenizer-directory",
        str(pack.donor_directory),
        "--target-tokenizer-directory",
        str(pack.target_directory),
        "--target-tokenizer-calibration",
        str(pack.target_calibration),
        "--tokenizer-bridge-artifact-sha256",
        pack.bridge_artifact_sha256,
        "--tokenizer-bridge-profile-sha256",
        pack.bridge_profile_sha256,
        "--tokenizer-bridge-mapped-parent-outputs",
        str(pack.mapped_parent_outputs),
        "--tokenizer-bridge-mapped-parent-outputs-sha256",
        pack.mapped_parent_outputs_sha256,
        "--tokenizer-bridge-minimum-parent-task-retention",
        "0",
        "--calibration",
        str(pack.donor_calibration),
        "--sealed-parent-cache-directory",
        str(pack.sealed_parent_cache),
        "--cached-parent-outputs",
        str(pack.cached_parent_outputs),
        "--cached-parent-outputs-sha256",
        pack.cached_parent_outputs_sha256,
        "--cache-root",
        str(run / "cache"),
        "--derived-global-topology-cache-root",
        str(stable / "derived-global-topology"),
        "--snapshot-root",
        str(run / "snapshots"),
        "--publish-destination",
        str(run / "published-child"),
        "--ledger",
        str(run / "ledger.jsonl"),
        "--receipt",
        str(run / "receipt.json"),
        "--diagnostics",
        str(run / "diagnostics.json"),
        "--parent-name",
        "parent",
        "--child-name",
        "lfm2-toy-integrated-carrier-exact4x",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--compression-ratio",
        "4",
        "--dense-minimum-hidden-size",
        str(_TARGET_HIDDEN_SIZE),
        # Carrier mode derives its own H816 ABI pin and its own sealed body
        # budget; nothing smaller is expressible.  See NOTES-1E.md, blocker 1.
        *(
            []
            if carrier_routes
            else ["--required-target-hidden-size", str(_TARGET_HIDDEN_SIZE)]
        ),
        "--dense-heterogeneous-layout",
        "--dense-heterogeneous-global-hidden-frontier",
        "--dense-heterogeneous-factor-writer-rank-discovery",
        "--dense-heterogeneous-global-functional-rows-per-role",
        "16",
        "--dense-heterogeneous-target-parameters",
        str(
            MODULE.LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS
            if carrier_routes
            else pack.exact_body_parameters
        ),
        "--dense-residual-basis-mode",
        residual_basis_mode,
        "--dense-run-closed-loop",
        "--dense-body-reclosure-mode",
        "moving_gauge_v3",
        "--dense-closed-loop-replay-mode",
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        "--dense-closed-loop-performance-metrics-diagnostic-only",
        "--dense-attention-consumer-mode",
        "coupled-value-output",
        "--dense-attention-geometry-risk",
        "--sample-last-tokens",
        str(_SAMPLE_LAST_TOKENS),
        "--max-decode-steps",
        str(_MAX_DECODE_STEPS),
        "--max-capture-bytes",
        str(_CAPTURE_BYTES),
        "--child-decode-trace-mode",
        "full_prefix_canonical",
        "--capture-logit-rtol",
        "0.2",
        "--capture-logit-atol",
        "1",
        "--max-selection-rows",
        "16",
        "--max-fit-rows",
        "16",
        "--max-validation-rows",
        "16",
        "--max-endpoint-relative-rmse",
        "1000000",
        "--min-endpoint-cosine-similarity",
        "-1",
        "--min-teacher-top1",
        "0",
        "--min-free-run-match",
        "0",
        "--min-residual-cosine",
        "-1",
        "--whole-output-max-new-tokens",
        str(_GENERATION_LIMIT),
        "--whole-output-maximum-prompt-tokens",
        "32",
        "--whole-output-fragment-minimum-characters",
        "1",
        "--whole-output-maximum-new-empty-responses",
        "8",
        "--whole-output-maximum-new-fragment-responses",
        "8",
        "--whole-output-minimum-parent-nonempty-retention",
        "0",
        "--whole-output-minimum-task-score-retention",
        "0",
        "--whole-output-maximum-parent-task-score-regressions",
        "8",
        "--whole-output-maximum-new-generation-limit-saturations",
        "8",
        "--whole-output-maximum-repeated-ngram-share-regressions",
        "8",
        "--whole-output-maximum-repeated-bigram-share-delta",
        "1",
        "--final-performance-metrics-diagnostic-only",
    ]
    if residual_basis_mode == "circuit_balanced":
        # Streaming vocabulary-margin observability is only legal on the
        # circuit-balanced residual basis, and the carrier lane pins both.
        key_file = stable / "keys" / "balanced-basis.key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(secrets.token_bytes(32))
        # The runner rejects group/world-readable HMAC keys; the ambient
        # umask on shared hosts would otherwise leave 0o664 and fail the run.
        key_file.chmod(0o600)
        arguments.extend(
            [
                "--dense-vocabulary-margin-observability",
                "--dense-streaming-vocabulary-margin-v2",
                "--dense-streaming-vocabulary-margin-pair-chunk-size",
                "64",
                "--balanced-basis-phase-cache-root",
                str(stable / "balanced-basis"),
                "--balanced-basis-phase-event-log",
                str(run / "balanced-basis-events.jsonl"),
                "--balanced-basis-phase-hmac-key-file",
                str(key_file),
                "--balanced-basis-phase-hmac-key-id",
                "toy-balanced-basis-v1",
            ]
        )
    if carrier_routes:
        arguments.extend(
            [
                "--carrier-route-publication",
                str(pack.root / "carrier-route-publication"),
                "--carrier-route-artifact-sha256",
                "4" * 64,
                "--carrier-span-evidence",
                str(pack.root / "carrier-span-evidence.json"),
                "--carrier-span-evidence-file-sha256",
                "5" * 64,
                "--carrier-span-evidence-sha256",
                "6" * 64,
                "--semantic-source-tokenizer-directory",
                str(pack.donor_directory),
                "--routed-manual-prompts",
                str(pack.manual_prompts),
                "--routed-manual-prompts-sha256",
                pack.manual_prompts_sha256,
                "--routed-manual-dossier",
                str(run / "routed-manual-dossier.json"),
                "--routed-manual-checkpoint-directory",
                str(run / "routed-manual-checkpoints"),
                "--counted-target-total-parameters",
                str(MODULE.LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS),
                "--counted-dense-body-parameters",
                str(MODULE.LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS),
                "--carrier-expected-target-rows",
                str(MODULE.LFM2_CARRIER_TARGET_ROWS),
                "--carrier-expected-hidden-size",
                str(MODULE.LFM2_CARRIER_HIDDEN_SIZE),
                "--routed-smoke-maximum-prompt-tokens",
                "32",
                "--routed-smoke-max-new-tokens",
                str(MODULE.LFM2_CARRIER_SMOKE_MAXIMUM_NEW_TOKENS),
                "--routed-manual-maximum-prompt-tokens",
                "64",
                "--routed-manual-initial-max-new-tokens",
                str(MODULE.LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS),
                "--routed-manual-retry-max-new-tokens",
                str(MODULE.LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS),
            ]
        )
    return arguments


def _failure_messages(receipt: Mapping[str, Any]) -> list[str]:
    failure = receipt.get("failure")
    if not isinstance(failure, Mapping):
        return []
    return [
        str(entry.get("message"))
        for entry in failure.get("exception_chain", ())
        if isinstance(entry, Mapping)
    ]


def test_toy_carrier_lane_folds_through_every_reachable_stage(
    carrier_lane_pack,
    tmp_path,
    capsys,
) -> None:
    """Run the r10 flag set end to end on CPU over real toy artifacts."""

    pack = carrier_lane_pack
    run = tmp_path / "run"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _lane_arguments(pack, run, carrier_routes=False)
        )
    )

    assert config.uses_tokenizer_bridge is True
    assert config.uses_carrier_routes is False
    assert config.dense_heterogeneous_global_hidden_frontier is True
    assert config.dense_heterogeneous_factor_writer_rank_discovery is True
    assert config.dense_run_closed_loop is True
    assert config.dense_body_reclosure_mode == "moving_gauge_v3"
    assert config.dense_closed_loop_replay_capture_mode == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert config.dense_attention_consumer_mode == "coupled-value-output"
    assert config.dense_attention_geometry_risk_enabled is True
    assert config.required_target_hidden_size == _TARGET_HIDDEN_SIZE
    assert config.dtype == "float32"

    receipt = MODULE.run_production_catalog(config)
    progress = _progress_rows(capsys.readouterr().out)
    stages = _stages(progress)
    diagnostics = json.loads(
        config.diagnostics_path.read_text(encoding="ascii")
    )

    # The ledger is emitted in strictly increasing, scalar-only records even
    # when a downstream phase relays its own callback metadata.  Run r7 died
    # exactly here, on a relayed field that collided with a reserved name.
    messages = _failure_messages(receipt)
    context = f"status={receipt['status']} failure={messages!r}"
    assert [row["sequence"] for row in progress] == list(range(len(progress)))
    assert stages[-1] == "final_receipt"
    _assert_ordered_stages(stages, _REQUIRED_STAGE_PREFIX, context=context)
    _assert_ordered_stages(stages, _REQUIRED_FOLD_STAGES, context=context)
    # moving_gauge_v3 recaptures its own previews and installs; both windows
    # must appear or the pinned reclosure mode silently degraded.
    assert "moving_gauge_preview_recapture_completed" in stages
    assert "moving_gauge_install_recapture_completed" in stages

    # Strict tokenizer bridge: authenticated, complete, and never a forward.
    bridge_receipt = receipt["parent"]["tokenizer_bridge"]
    assert bridge_receipt["enabled"] is True
    assert bridge_receipt["artifact_sha256"] == pack.bridge_artifact_sha256
    assert bridge_receipt["transport_profile_sha256"] == (
        pack.bridge_profile_sha256
    )
    assert bridge_receipt["target_row_capacity"] == pack.target_row_capacity
    assert bridge_receipt["donor_row_capacity"] == pack.donor_row_capacity
    assert bridge_receipt["all_physical_target_rows_mapped"] is True
    assert bridge_receipt["paired_record_count"] == len(_TRAJECTORIES)
    assert bridge_receipt["parent_model_forwards"] == 0
    assert bridge_receipt["training_steps"] == 0
    assert diagnostics["tokenizer_bridge"]["enabled"] is True
    assert diagnostics["parent_traces_regenerated_after_cache"] is False
    assert diagnostics["training_steps"] == 0
    assert diagnostics["zero_gradient"] is True

    # Zero-training and zero-parent-forward invariants hold in both branches.
    assert receipt["folding_only"] is True
    assert receipt["training_steps"] == 0
    assert receipt["parent"]["parent_cache_build_called"] is False
    assert receipt["parent"]["parent_cache_build_forward_count"] == 0
    assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
    assert receipt["parent"]["parent_unique_parameters"] == (
        pack.parent_unique_parameters
    )
    assert receipt["closed_loop_performance_metrics_policy"]["training_steps"] == 0

    # This lane is the production carrier shape: heterogeneous child, canonical
    # prefix replay, moving_gauge_v3, and a selective authenticated session.
    # The adapter's own capture-merge decision is sealed here so an opted-in
    # run can never look effective when the hook actually declined.
    merged_policy = receipt["closed_loop_replay_policy"][
        "merge_adjacent_layer_captures_observed"
    ]
    assert merged_policy is not None, context
    assert merged_policy["requested"] is False
    assert merged_policy["selective_session_bound"] is True
    assert merged_policy["heterogeneous_child"] is True
    assert merged_policy["replay_capture_mode"] == (
        MODULE.CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert merged_policy["admitted_by_adapter"] is True, context

    if receipt["status"] == "complete" or _PACKAGED_BRIDGE_BLOCKER in messages:
        # Both branches mean the folded child cleared the cached-parent
        # trajectory gate and the dual original/mapped whole-output behavior
        # gate, and that publication had already started.
        accepted = [
            event
            for event in diagnostics["behavior_gate_events"]
            if event["phase"] == "accepted_in_memory_child"
        ]
        assert accepted, context
        assert accepted[0]["summary"]["passed"] is True, context
        assert accepted[0]["parent_forward_count"] == 0
        assert diagnostics["original_parent_whole_output_events"], context
        assert diagnostics["mapped_parent_whole_output_events"], context

    if receipt["status"] == "complete":
        _assert_ordered_stages(
            stages,
            (
                "system_fold_completed",
                "success",
                "final_receipt",
            ),
        )
        assert receipt["fold"]["optimizer_steps"] == 0
        assert receipt["fold"]["backward_calls"] == 0
        assert receipt["parent"]["parent_forward_count_after_cache"] == 0
        assert receipt["artifact"]["fresh_reload_accepted"] is True
        assert set(receipt["behavior"]) == {
            "accepted_in_memory_child",
            "freshly_reloaded_child",
        }
        assert (config.publish_destination / "certificate.json").is_file()
        assert (
            config.publish_destination / "child" / "tokenizer.json"
        ).is_file()
        # The published child speaks the compiled target vocabulary.
        assert receipt["target"]["exact_unique_parameters"] == (
            pack.exact_body_parameters
        )
        return

    # Deepest currently reachable configuration: the run must still fail
    # closed, and only on a blocker this rehearsal already reported.
    assert receipt["status"] == "failed_closed"
    assert receipt["failure"]["fail_closed"] is True
    messages = _failure_messages(receipt)
    assert any(
        message in _KNOWN_TOY_SCALE_BLOCKERS for message in messages
    ), (
        "the toy carrier lane failed on an unreported defect; record it in "
        f"NOTES-1E.md before pinning it: {messages!r} after stage "
        f"{stages[-3] if len(stages) > 2 else stages!r}"
    )


def test_toy_carrier_lane_merges_adjacent_layer_captures(
    carrier_lane_pack,
    tmp_path,
    capsys,
) -> None:
    """The r10 flag set plus the capture merge, on the production lane shape.

    The merge is only worth an operator's trust if the same lane reaches the
    same outcome with one fewer child replay per layer boundary, so this runs
    the identical arguments twice and compares the two runs, not a fixture.
    """

    pack = carrier_lane_pack
    baseline_run = tmp_path / "baseline"
    baseline_config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _lane_arguments(pack, baseline_run, carrier_routes=False)
        )
    )
    baseline_receipt = MODULE.run_production_catalog(baseline_config)
    baseline_stages = _stages(_progress_rows(capsys.readouterr().out))

    merged_run = tmp_path / "merged"
    merged_config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_lane_arguments(pack, merged_run, carrier_routes=False),
                "--dense-closed-loop-merge-adjacent-layer-captures",
            ]
        )
    )
    assert (
        merged_config.dense_closed_loop_merge_adjacent_layer_captures is True
    )
    merged_receipt = MODULE.run_production_catalog(merged_config)
    merged_progress = _progress_rows(capsys.readouterr().out)
    merged_stages = _stages(merged_progress)

    merged_policy = merged_receipt["closed_loop_replay_policy"][
        "merge_adjacent_layer_captures_observed"
    ]
    context = (
        f"baseline={baseline_receipt['status']} "
        f"merged={merged_receipt['status']} "
        f"failure={_failure_messages(merged_receipt)!r}"
    )
    assert merged_policy["requested"] is True, context
    assert merged_policy["admitted_by_adapter"] is True, context
    # Requested and admitted is not enough: the boundary replays must actually
    # have been served from one merged capture.
    assert "child_closed_loop_layer_merged_capture_completed" in merged_stages
    assert (
        "child_closed_loop_layer_capture_served_by_merged_replay"
        in merged_stages
    )
    assert (
        "child_closed_loop_layer_merged_capture_completed"
        not in baseline_stages
    )
    assert merged_receipt["status"] == baseline_receipt["status"], context
    _assert_ordered_stages(
        merged_stages,
        _REQUIRED_FOLD_STAGES,
        context=context,
    )
    assert _failure_messages(merged_receipt) == _failure_messages(
        baseline_receipt
    ), context


def test_toy_carrier_lane_counts_route_state_inside_the_target(
    carrier_lane_pack,
) -> None:
    """Two independent route channels must be counted, never free state."""

    pack = carrier_lane_pack
    route_elements = 4 * _TARGET_HIDDEN_SIZE * pack.target_row_capacity
    allocations = (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=route_elements,
            tensor_sha256="8" * 64,
            artifact_sha256="9" * 64,
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=route_elements,
            tensor_sha256="a" * 64,
            artifact_sha256="9" * 64,
        ),
    )
    counted_total = pack.exact_body_parameters + 2 * route_elements
    plan = MODULE.build_counted_child_state_plan(
        parent_unique_parameters=pack.parent_unique_parameters,
        target_total_parameter_elements=counted_total,
        continuous_allocations=allocations,
    )

    assert plan.dense_model_parameter_elements == pack.exact_body_parameters
    assert plan.continuous_parameter_elements == 2 * route_elements

    verification = MODULE.verify_counted_child_state(
        plan,
        physical_dense_parameter_elements=pack.exact_body_parameters,
        observed_allocations=allocations,
    )

    assert verification["target_total_parameter_elements"] == counted_total
    assert verification["physical_dense_parameter_elements"] == (
        pack.exact_body_parameters
    )
    assert verification["quantization_credit"] is False
    assert verification["parent_runtime_sidecar"] is False
    assert verification["realized_parameter_compaction_ratio"] == pytest.approx(
        pack.parent_unique_parameters / counted_total
    )
    # The realized ratio counts the routes, so it is strictly worse than the
    # dense-body-only ratio the frontier solves for.
    assert verification["realized_parameter_compaction_ratio"] < (
        pack.parent_unique_parameters / pack.exact_body_parameters
    )

    # A child that ships only the dense body cannot satisfy the counted plan.
    with pytest.raises(CountedChildStateError):
        MODULE.verify_counted_child_state(
            plan,
            physical_dense_parameter_elements=counted_total,
            observed_allocations=allocations,
        )
    with pytest.raises(CountedChildStateError):
        MODULE.verify_counted_child_state(
            plan,
            physical_dense_parameter_elements=pack.exact_body_parameters,
            observed_allocations=allocations[:1],
        )


def test_authenticated_carrier_routes_stop_at_the_sealed_production_abi(
    carrier_lane_pack,
    tmp_path,
    capsys,
) -> None:
    """The routed lane is pinned to the 4096-row H816 exact-4x publication."""

    pack = carrier_lane_pack
    run = tmp_path / "carrier-run"
    arguments = _lane_arguments(pack, run, carrier_routes=True)

    # Two config-time pins make a toy routed child inexpressible: the residual
    # width is forced to H816 and the dense body budget to the sealed
    # 53,552,368-element remainder.  See NOTES-1E.md, blocker 1.
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="equal carrier_expected_hidden_size",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *arguments,
                    "--required-target-hidden-size",
                    str(_TARGET_HIDDEN_SIZE),
                ]
            )
        )
    toy_budget_arguments = list(arguments)
    toy_budget_arguments[
        toy_budget_arguments.index("--dense-heterogeneous-target-parameters") + 1
    ] = str(pack.exact_body_parameters)
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="reserve both independent route channels",
    ):
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(toy_budget_arguments)
        )

    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(arguments)
    )

    # The parser derives the whole sealed carrier ABI from the same flags the
    # r10 launcher passes; none of it is reachable from a toy child.
    assert config.uses_carrier_routes is True
    assert config.carrier_expected_target_rows == 4_096
    assert config.carrier_expected_hidden_size == 816
    assert config.required_target_hidden_size == 816
    assert config.counted_target_total_parameters == 57_420_144
    assert config.counted_dense_body_parameters == 53_552_368
    assert (
        config.counted_target_total_parameters
        - config.counted_dense_body_parameters
        == 2 * MODULE.LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE
    )
    carrier_identity = config.identity()["carrier_routes"]
    assert carrier_identity["numeric_metrics_have_acceptance_authority"] is False
    assert carrier_identity[
        "manual_whole_response_intelligence_review_is_authoritative"
    ] is True
    assert config.routed_manual_dossier_path == (
        run / "routed-manual-dossier.json"
    )
    assert config.routed_manual_checkpoint_directory == (
        run / "routed-manual-checkpoints"
    )

    receipt = MODULE.run_production_catalog(config)
    stages = _stages(_progress_rows(capsys.readouterr().out))

    assert receipt["status"] == "failed_closed"
    assert receipt["failure"]["fail_closed"] is True
    assert receipt["folding_only"] is True
    assert receipt["training_steps"] == 0
    assert receipt["parent"]["parent_cache_build_forward_count"] == 0
    # Everything before the carrier publication gate is reachable at toy scale.
    _assert_ordered_stages(
        stages,
        (
            "run_started",
            "local_artifacts_verified",
            "parent_verified",
            "parent_cache_verified",
            "tokenizer_bridge_verified",
        ),
    )
    # The routed lane never reaches carrier verification, target synthesis, or
    # the manual-review dossier without the real authenticated publication.
    assert "carrier_verified" not in stages
    assert "system_fold_started" not in stages
    assert "manual_review_started" not in stages
    assert not config.routed_manual_dossier_path.exists()
    assert receipt["publication_present"] is False


# --- fold/round alternation on the carrier lane's dense core ----------------
#
# The routed carrier configuration is inexpressible at toy scale (the sealed
# 4096-row/H816 ABI, NOTES-1E blocker 1).  Everything below it is reachable,
# including the **bridged** configuration: a round captured against a bridged
# child now seals the child's own target tokenization of every extension batch
# next to the donor batch the parent was captured on, so the merged corpus has
# a target-token child replay and the lane consumes base+extension evidence.
# Both the unbridged dense core and the bridged lane are exercised here: the
# same H1024 parent, the same exact-4x dense budget, the same heterogeneous
# layout with the global hidden frontier and factor-writer rank discovery, the
# same moving_gauge_v3 closed loop, published, rolled out, and refolded on the
# child's own rollout evidence.

_REFOLD_FIT_PROMPTS = (
    "the quick brown fox",
    "a folded child keeps",
    "counted route tensors stay",
)
_REFOLD_MANUAL_PROMPTS = ("my kitchen smells like gas", "explain the parent")


def _alternation_arguments(
    pack: _CarrierLanePack,
    run: Path,
    arguments: list[str],
) -> list[str]:
    """Append the bounded one-round, one-refold alternation knobs."""

    return [
        *arguments,
        "--on-policy-evidence-rounds",
        "1",
        "--on-policy-refold-after-rounds",
        "1",
        "--on-policy-fit-prompts",
        str(pack.root / "on-policy-fit-prompts.json"),
        "--on-policy-manual-prompt-manifest",
        str(pack.root / "on-policy-manual-prompts.json"),
        "--on-policy-round-root",
        str(run / "on-policy-rounds"),
        "--on-policy-max-prompts",
        str(len(_REFOLD_FIT_PROMPTS)),
        "--on-policy-max-new-tokens",
        "2",
        # ByteLevel-BPE seed prompts are long in tokens; the bound only has to
        # leave room for the seed plus the two decoded tokens.
        "--on-policy-max-context-tokens",
        "64",
    ]


def _bridged_refold_lane_arguments(
    pack: _CarrierLanePack, run: Path
) -> list[str]:
    """The bridged carrier lane itself, with the alternation knobs."""

    arguments = _lane_arguments(pack, run, carrier_routes=False)
    # The sealed base cache was captured with the eager attention backend, and
    # so is the round's extension capture: causal-circuit installability
    # evidence requires the reconstructed score/probability operator to be the
    # one the parent actually executed.
    arguments[arguments.index("--parent-model") + 1] = str(
        pack.root / "parent-eager"
    )
    return _alternation_arguments(pack, run, arguments)


def _bridged_balanced_refold_lane_arguments(
    pack: _CarrierLanePack,
    run: Path,
) -> list[str]:
    """The bridged lane on the carrier lane's basis and margins, with a round.

    ``ProductionLfm2CatalogConfig.__post_init__`` pins carrier-routed production
    to the circuit-balanced residual basis with
    ``--dense-vocabulary-margin-observability`` **and**
    ``--dense-streaming-vocabulary-margin-v2`` together.  The counted carrier ABI
    is out of reach at toy scale, so this lane composes every other term of that
    pin -- bridge, circuit-balanced basis, streaming-v2 margins, the
    balanced-basis phase cache -- and adds the one-round/one-refold alternation,
    which is the combination that had no executable configuration.
    """

    arguments = _lane_arguments(
        pack,
        run,
        carrier_routes=False,
        residual_basis_mode="circuit_balanced",
    )
    # The sealed base cache was captured with the eager attention backend, and
    # so is the round's extension capture: causal-circuit installability
    # evidence requires the reconstructed score/probability operator to be the
    # one the parent actually executed.
    arguments[arguments.index("--parent-model") + 1] = str(
        pack.root / "parent-eager"
    )
    return _alternation_arguments(pack, run, arguments)


def _refold_lane_arguments(pack: _CarrierLanePack, run: Path) -> list[str]:
    """The carrier lane's dense core, unbridged, with the alternation knobs."""

    bridged = _lane_arguments(pack, run, carrier_routes=False)
    dropped_flags = {
        "--target-tokenizer-directory",
        "--target-tokenizer-calibration",
        "--tokenizer-bridge-artifact-sha256",
        "--tokenizer-bridge-profile-sha256",
        "--tokenizer-bridge-mapped-parent-outputs",
        "--tokenizer-bridge-mapped-parent-outputs-sha256",
        "--tokenizer-bridge-minimum-parent-task-retention",
        "--dense-heterogeneous-target-parameters",
    }
    arguments: list[str] = []
    skip = 0
    for value in bridged:
        if skip:
            skip -= 1
            continue
        if value in dropped_flags:
            skip = 1
            continue
        arguments.append(value)
    # The sealed base cache was captured with the eager attention backend, and
    # so is the round's extension capture: causal-circuit installability
    # evidence requires the reconstructed score/probability operator to be the
    # one the parent actually executed.
    arguments[arguments.index("--parent-model") + 1] = str(
        pack.root / "parent-eager"
    )
    # Without the bridge the child speaks the donor vocabulary, so the exact
    # whole-child budget is the one that vocabulary admits.
    arguments.extend(
        [
            "--dense-heterogeneous-target-parameters",
            str(_exact_toy_body_budget(pack.donor_row_capacity)),
        ]
    )
    return _alternation_arguments(pack, run, arguments)


@pytest.fixture(scope="module")
def refold_prompt_pack(carrier_lane_pack) -> _CarrierLanePack:
    """Seal the fit and reserved-review prompt corpora one round consumes."""

    pack = carrier_lane_pack
    eager_directory = pack.root / "parent-eager"
    if not eager_directory.exists():
        shutil.copytree(pack.parent_directory, eager_directory)
        config_path = eager_directory / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["attn_implementation"] = "eager"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
    (pack.root / "on-policy-fit-prompts.json").write_text(
        json.dumps(list(_REFOLD_FIT_PROMPTS)),
        encoding="utf-8",
    )
    (pack.root / "on-policy-manual-prompts.json").write_text(
        json.dumps(
            [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in _REFOLD_MANUAL_PROMPTS
            ]
        ),
        encoding="utf-8",
    )
    return pack


def test_bridged_carrier_lane_refuses_a_round_without_a_target_preimage(
    refold_prompt_pack,
    tmp_path,
) -> None:
    """A round with no sealed preimage is refused by name, at config time."""

    pack = refold_prompt_pack
    run = tmp_path / "bridged"
    missing = run / "on-policy-rounds" / "round-000" / "round-manifest.json"
    arguments = [
        *_lane_arguments(pack, run, carrier_routes=False),
        "--on-policy-round-manifest",
        str(missing),
    ]
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="no target-token preimage",
    ) as failure:
        MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))
    assert str(missing) in str(failure.value)

    # The lane itself is no longer the reason: with nothing to consume yet the
    # bridged lane is a full consumer, because the rounds this same invocation
    # produces are captured with the bridge in hand.
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            [
                *_lane_arguments(pack, run, carrier_routes=False),
                "--on-policy-evidence-rounds",
                "1",
                "--on-policy-fit-prompts",
                str(pack.root / "on-policy-fit-prompts.json"),
                "--on-policy-manual-prompt-manifest",
                str(pack.root / "on-policy-manual-prompts.json"),
            ]
        )
    )
    policy = MODULE._on_policy_evidence_policy(config)
    assert policy["consumes_extension_evidence"] is False
    assert policy["extension_evidence_consumer_gap"] is None


def test_bridged_carrier_lane_without_extension_evidence_is_unchanged(
    refold_prompt_pack,
    tmp_path,
) -> None:
    """Nothing this wave added is in a base-only bridged run's identity."""

    pack = refold_prompt_pack
    run = tmp_path / "bridged-base-only"
    arguments = _lane_arguments(pack, run, carrier_routes=False)
    config = MODULE.config_from_args(MODULE.build_parser().parse_args(arguments))

    # The fold identity — and therefore every derived phase-cache key and the
    # derived restart-capsule root — carries no target-preimage field at all.
    identity = json.dumps(config.identity(), sort_keys=True)
    assert "preimage" not in identity
    assert "merged_trace_adapter" not in identity
    # Rebuilding the same arguments reproduces the same configuration digest.
    rebuilt = MODULE.config_from_args(
        MODULE.build_parser().parse_args(arguments)
    )
    expected_configuration_sha256 = MODULE.canonical_sha256(config.identity())
    assert MODULE.canonical_sha256(rebuilt.identity()) == (
        expected_configuration_sha256
    )
    assert MODULE._unconsumed_extension_evidence_reason(config) is None

    receipt = MODULE.run_production_catalog(config)
    assert receipt["status"] == "complete", _failure_messages(receipt)
    assert receipt["configuration_sha256"] == expected_configuration_sha256

    # The bridge receipt body is the base-only one: no merged replay, no
    # preimage digests, and its own seal reproduces from that body.
    bridge_receipt = dict(receipt["parent"]["tokenizer_bridge"])
    assert not [name for name in bridge_receipt if name.startswith("on_policy")]
    sealed = bridge_receipt.pop("runtime_receipt_sha256")
    assert MODULE.canonical_sha256(bridge_receipt) == sealed
    assert receipt["on_policy_evidence"]["evidence_identity_sha256"] is None
    assert receipt["on_policy_evidence"]["round_count"] == 0
    certificate = json.loads(
        (config.publish_destination / "certificate.json").read_text(
            encoding="ascii"
        )
    )
    assert certificate["evidence_fingerprint"] == (
        receipt["parent"]["parent_cache_key"]
    )


def test_carrier_dense_core_alternates_fold_round_and_refold(
    refold_prompt_pack,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """publish -> real round -> refold on base+extension -> publish again."""

    pack = refold_prompt_pack
    run = tmp_path / "run"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(_refold_lane_arguments(pack, run))
    )

    assert config.uses_tokenizer_bridge is False
    assert config.uses_carrier_routes is False
    assert config.compression_ratio == 4.0
    assert config.dense_heterogeneous_layout is True
    assert config.dense_heterogeneous_global_hidden_frontier is True
    assert config.dense_heterogeneous_factor_writer_rank_discovery is True
    assert config.dense_run_closed_loop is True
    assert config.dense_body_reclosure_mode == "moving_gauge_v3"
    assert config.on_policy_refold_after_rounds == 1
    # This lane consumes base+extension evidence; nothing about it is gated.
    assert MODULE._unconsumed_extension_evidence_reason(config) is None

    # Observers only: the real loader and the real round still run.  They count
    # every parent forward and sample the counter around the sanctioned round.
    observed: list[dict[str, int | None]] = []
    real_loader = MODULE._load_cache_bound_parent
    real_round = MODULE.run_lfm2_on_policy_evidence_round

    def counting_loader(*args, **kwargs):
        model, dtype, name = real_loader(*args, **kwargs)
        record: dict[str, int | None] = {
            "forwards": 0,
            "before_round": None,
            "after_round": None,
        }
        observed.append(record)

        def count(_module, _inputs, _output):
            record["forwards"] = int(record["forwards"]) + 1

        model.register_forward_hook(count)
        return model, dtype, name

    def observing_round(**kwargs):
        record = observed[-1]
        record["before_round"] = record["forwards"]
        result = real_round(**kwargs)
        record["after_round"] = record["forwards"]
        return result

    monkeypatch.setattr(MODULE, "_load_cache_bound_parent", counting_loader)
    monkeypatch.setattr(
        MODULE,
        "run_lfm2_on_policy_evidence_round",
        observing_round,
    )
    summary = MODULE.run_production_catalog_on_policy_alternation(config)
    rows = _progress_rows(capsys.readouterr().out)

    assert summary["status"] == "complete", summary.get("failure")
    assert summary["executed_pass_count"] == summary["expected_pass_count"] == 2
    base_pass, refold_pass = summary["passes"]
    assert base_pass["status"] == refold_pass["status"] == "complete", summary

    base_receipt = json.loads(
        (run / "receipt.json").read_text(encoding="ascii")
    )
    refold_receipt = json.loads(
        (run / "receipt-round-0-refold.json").read_text(encoding="ascii")
    )
    round_manifest = json.loads(
        Path(base_pass["sealed_round_manifests"][0]).read_text(encoding="ascii")
    )
    extension_batches = int(round_manifest["extension_cache"]["batch_count"])

    # 1. The refold consumed exactly the round the first pass sealed.
    assert refold_pass["consumed_round_manifests"] == (
        base_pass["sealed_round_manifests"]
    )
    evidence = refold_receipt["on_policy_evidence"]
    assert evidence["consumed_by_this_solve_pass"] is True
    assert evidence["extension_batch_counts"] == [extension_batches]
    evidence_set = evidence["evidence_set"]
    assert evidence_set["member_count"] == 2
    assert evidence_set["batch_count"] == len(_TRAJECTORIES) + extension_batches

    # 2. Extension rows are fit rows only, in the sealed inventory and in the
    #    merged rows the closed-form writers actually read.
    assert evidence_set["extension_selection_or_validation_batch_count"] == 0
    assert evidence_set["extension_fit_batch_count"] == extension_batches
    assert evidence_set["parent_forward_calls"] == 0
    materialized = next(
        row
        for row in rows
        if row["stage"] == "on_policy_extension_evidence_materialized"
    )
    assert materialized["base_path_count"] > 0
    assert materialized["extension_path_count"] > 0
    assert int(materialized["merged_path_count"]) == (
        int(materialized["base_path_count"])
        + int(materialized["extension_path_count"])
    )
    assert materialized["parent_forwards"] == 0
    # 2b. The merge preallocated once instead of holding two live copies.
    assert materialized["merge_preallocated_once"] is True
    assert materialized["merge_member_sources_released_as_consumed"] is True
    assert int(materialized["merge_streaming_merge_peak_bytes"]) < int(
        materialized["merge_separate_materialization_peak_bytes"]
    )
    accounting = refold_receipt["parent"]["on_policy_merged_row_accounting"]
    assert accounting["merged_row_count"] == int(
        materialized["merged_path_count"]
    )
    assert accounting["peak_bytes_ratio_to_retained"] < 1.5

    # 3. Every derived identity moved with the extension evidence, and the
    #    sealed base cache itself is untouched and shared by both passes.
    assert base_receipt["on_policy_evidence"]["evidence_identity_sha256"] is None
    assert len(evidence["evidence_identity_sha256"]) == 64
    assert (
        refold_receipt["parent"]["evidence_split_sha256"]
        != base_receipt["parent"]["evidence_split_sha256"]
    )
    assert (
        refold_receipt["parent"]["parent_trace_rows_sha256"]
        != base_receipt["parent"]["parent_trace_rows_sha256"]
    )
    assert (
        refold_receipt["parent"]["parent_cache_key"]
        == base_receipt["parent"]["parent_cache_key"]
    )
    base_certificate = json.loads(
        (run / "published-child" / "certificate.json").read_text(
            encoding="ascii"
        )
    )
    refold_certificate = json.loads(
        (
            run / "published-child-round-0-refold" / "certificate.json"
        ).read_text(encoding="ascii")
    )
    assert base_certificate["evidence_fingerprint"] == (
        base_receipt["parent"]["parent_cache_key"]
    )
    assert (
        refold_certificate["evidence_fingerprint"]
        == evidence_set["evidence_set_cache_key"]
        != base_certificate["evidence_fingerprint"]
    )

    # 4. Zero parent forwards outside the round's single sanctioned capture.
    assert len(observed) == 2
    assert observed[0]["before_round"] == 0
    assert observed[0]["after_round"] == observed[0]["forwards"]
    assert int(observed[0]["forwards"]) > 0
    assert observed[1]["before_round"] is None
    assert observed[1]["forwards"] == 0
    assert round_manifest["counters"]["parent_capture_invocations"] == 1
    assert round_manifest["counters"]["optimizer_steps"] == 0
    assert round_manifest["counters"]["backward_calls"] == 0
    for receipt in (base_receipt, refold_receipt):
        assert receipt["parent"]["parent_forward_count_after_cache"] == 0
        assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
        assert receipt["fold"]["optimizer_steps"] == 0
        assert receipt["fold"]["backward_calls"] == 0
        assert receipt["training_steps"] == 0
    assert base_receipt["parent"]["on_policy_parent_capture_invocations"] == 1
    assert refold_receipt["parent"]["on_policy_parent_capture_invocations"] == 0

    # 5. Both children are real, distinct artifacts that reload standalone at
    #    the exact counted budget.
    for receipt in (base_receipt, refold_receipt):
        assert receipt["artifact"]["fresh_reload_accepted"] is True
    assert (
        base_receipt["target"]["exact_unique_parameters"]
        == refold_receipt["target"]["exact_unique_parameters"]
    )
    for directory in ("published-child", "published-child-round-0-refold"):
        assert (run / directory / "child" / "tokenizer.json").is_file()

    # 6. Each pass journals its own ordered progress into one shared ledger.
    for pass_record in (base_pass, refold_pass):
        pass_rows = [
            row
            for row in rows
            if row["configuration_sha256"] == pass_record["configuration_sha256"]
        ]
        assert [row["sequence"] for row in pass_rows] == list(
            range(len(pass_rows))
        )
        assert pass_rows[-1]["stage"] == "final_receipt"
        _assert_ordered_stages(
            [str(row["stage"]) for row in pass_rows],
            _REQUIRED_FOLD_STAGES,
            context=str(pass_record["status"]),
        )
    assert base_pass["configuration_sha256"] != (
        refold_pass["configuration_sha256"]
    )
    assert base_receipt["ledger"]["record_count"] < (
        refold_receipt["ledger"]["record_count"]
    )


def _round_manifest_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def _reseal_round_manifest(document: dict[str, Any], path: Path) -> Path:
    """Write one canonical round manifest with every seal recomputed."""

    preimage = document.get("target_preimage")
    if preimage is not None:
        body = {
            name: value
            for name, value in preimage.items()
            if name != "target_preimage_sha256"
        }
        document["target_preimage"] = {
            **body,
            "target_preimage_sha256": MODULE.canonical_sha256(body),
        }
    body = {
        name: value
        for name, value in document.items()
        if name != "manifest_sha256"
    }
    sealed = {**body, "manifest_sha256": MODULE.canonical_sha256(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            sealed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return path


def test_bridged_carrier_lane_alternates_fold_round_and_refold(
    refold_prompt_pack,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """Bridged publish -> real round with both preimages -> refold -> publish."""

    pack = refold_prompt_pack
    run = tmp_path / "run"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _bridged_refold_lane_arguments(pack, run)
        )
    )

    assert config.uses_tokenizer_bridge is True
    assert config.dense_heterogeneous_layout is True
    assert config.dense_run_closed_loop is True
    assert config.dense_body_reclosure_mode == "moving_gauge_v3"
    assert config.on_policy_refold_after_rounds == 1
    # Nothing about the bridged lane's shape gates consumption any more.
    assert MODULE._unconsumed_extension_evidence_reason(config) is None

    # Observers only: the real loader and the real round still run.
    observed: list[dict[str, int | None]] = []
    real_loader = MODULE._load_cache_bound_parent
    real_round = MODULE.run_lfm2_on_policy_evidence_round

    def counting_loader(*args, **kwargs):
        model, dtype, name = real_loader(*args, **kwargs)
        record: dict[str, int | None] = {
            "forwards": 0,
            "before_round": None,
            "after_round": None,
        }
        observed.append(record)

        def count(_module, _inputs, _output):
            record["forwards"] = int(record["forwards"]) + 1

        model.register_forward_hook(count)
        return model, dtype, name

    def observing_round(**kwargs):
        record = observed[-1]
        record["before_round"] = record["forwards"]
        result = real_round(**kwargs)
        record["after_round"] = record["forwards"]
        return result

    monkeypatch.setattr(MODULE, "_load_cache_bound_parent", counting_loader)
    monkeypatch.setattr(
        MODULE,
        "run_lfm2_on_policy_evidence_round",
        observing_round,
    )
    summary = MODULE.run_production_catalog_on_policy_alternation(config)
    rows = _progress_rows(capsys.readouterr().out)

    assert summary["status"] == "complete", summary.get("failure")
    assert summary["executed_pass_count"] == summary["expected_pass_count"] == 2
    base_pass, refold_pass = summary["passes"]
    assert base_pass["status"] == refold_pass["status"] == "complete", summary

    base_receipt = json.loads(
        (run / "receipt.json").read_text(encoding="ascii")
    )
    refold_receipt = json.loads(
        (run / "receipt-round-0-refold.json").read_text(encoding="ascii")
    )
    manifest_path = Path(base_pass["sealed_round_manifests"][0])
    round_manifest = _round_manifest_document(manifest_path)
    extension_batches = int(round_manifest["extension_cache"]["batch_count"])
    artifact = _load_pack_bridge(pack)
    mapping = list(artifact.transport_profile.target_to_donor_row_indices)

    # 1. The round sealed BOTH preimages of every extension batch: the donor
    #    batch the parent was actually captured on, and the child's own target
    #    tokenization it is the one-row image of.
    preimage = round_manifest["target_preimage"]
    assert preimage["schema"] == MODULE_ROUND.ROUND_TARGET_PREIMAGE_SCHEMA
    assert preimage["parent_cache_input_mode"] == (
        MODULE_ROUND.BRIDGE_MAPPED_CAPTURE_INPUT_MODE
    )
    assert preimage["bridge_artifact_sha256"] == pack.bridge_artifact_sha256
    assert preimage["transport_profile_sha256"] == pack.bridge_profile_sha256
    assert preimage["child_tokenizer_is_the_sealed_bridge_target"] is True
    assert preimage["alignment"]["identity"] == "exact-decoded-utf8-sha256"
    assert preimage["alignment"]["numeric_token_id_equality_required"] is False
    assert preimage["record_count"] == extension_batches
    assert len(preimage["records"]) == extension_batches
    donor_inputs = list(
        round_manifest["extension_cache"]["ordered_batch_input_sha256"]
    )
    for position, record in enumerate(preimage["records"]):
        assert record["index"] == position
        assert record["split"] == "fit"
        assert record["donor_cache_input_sha256"] == donor_inputs[position]
        # The parent-facing batch really is the row map applied to the sealed
        # target preimage, and the map really moved IDs.
        assert record["numerically_different_token_count"] > 0
        assert max(record["target_input_ids"]) < preimage["target_row_capacity"]
        donor_prompt = [mapping[value] for value in record["target_input_ids"]]
        assert donor_prompt != list(record["target_input_ids"])
        # Both preimages are present, and the parent tokenizer's own reading of
        # the same decoded text is recorded without being what was captured.
        parent_preimage = record["parent_tokenizer_preimage"]
        assert parent_preimage["used_for_parent_capture"] is False
        assert parent_preimage["token_ids_sha256"] != (
            record["target_token_ids_sha256"]
        )
    assert base_receipt["on_policy_evidence"][
        "postpublication_rounds_executed"
    ][0]["carries_target_token_preimage"] is True

    # 2. The refold consumed exactly that round, on base+extension evidence.
    assert refold_pass["consumed_round_manifests"] == (
        base_pass["sealed_round_manifests"]
    )
    evidence = refold_receipt["on_policy_evidence"]
    assert evidence["consumed_by_this_solve_pass"] is True
    assert evidence["extension_batch_counts"] == [extension_batches]
    evidence_set = evidence["evidence_set"]
    assert evidence_set["member_count"] == 2
    assert evidence_set["batch_count"] == len(_TRAJECTORIES) + extension_batches
    assert evidence_set["extension_selection_or_validation_batch_count"] == 0
    assert evidence_set["extension_fit_batch_count"] == extension_batches
    assert evidence_set["every_extension_carries_a_target_preimage"] is True
    assert evidence_set["extension_target_preimage_sha256"] == [
        preimage["target_preimage_sha256"]
    ]

    # 3. The bridge the refold bound replays the merged corpus, one
    #    target-token record per merged sealed batch, with zero parent forwards.
    bridge_receipt = refold_receipt["parent"]["tokenizer_bridge"]
    assert bridge_receipt["on_policy_merged_record_count"] == (
        evidence_set["batch_count"]
    )
    assert bridge_receipt["on_policy_extension_record_count"] == (
        extension_batches
    )
    assert bridge_receipt["on_policy_merged_paired_record_count"] == (
        evidence_set["batch_count"]
    )
    assert bridge_receipt["on_policy_target_preimage_sha256"] == [
        preimage["target_preimage_sha256"]
    ]
    assert bridge_receipt["parent_model_forwards"] == 0
    # The base-only pass carries none of those keys at all.
    assert not [
        name
        for name in base_receipt["parent"]["tokenizer_bridge"]
        if name.startswith("on_policy")
    ]
    stages = _stages(rows)
    assert "tokenizer_bridge_on_policy_preimage_complete" in stages

    # 4. Every derived identity moved with the extension evidence while the
    #    sealed base cache stayed exactly where it was.
    assert base_receipt["on_policy_evidence"]["evidence_identity_sha256"] is None
    assert len(evidence["evidence_identity_sha256"]) == 64
    assert (
        refold_receipt["parent"]["evidence_split_sha256"]
        != base_receipt["parent"]["evidence_split_sha256"]
    )
    assert (
        refold_receipt["parent"]["parent_trace_rows_sha256"]
        != base_receipt["parent"]["parent_trace_rows_sha256"]
    )
    assert (
        refold_receipt["parent"]["parent_cache_key"]
        == base_receipt["parent"]["parent_cache_key"]
    )
    refold_certificate = json.loads(
        (
            run / "published-child-round-0-refold" / "certificate.json"
        ).read_text(encoding="ascii")
    )
    assert refold_certificate["evidence_fingerprint"] == (
        evidence_set["evidence_set_cache_key"]
    )

    # 5. Zero parent forwards outside the round's single sanctioned capture.
    assert len(observed) == 2
    assert observed[0]["before_round"] == 0
    assert observed[0]["after_round"] == observed[0]["forwards"]
    assert int(observed[0]["forwards"]) > 0
    assert observed[1]["before_round"] is None
    assert observed[1]["forwards"] == 0
    assert round_manifest["counters"]["parent_capture_invocations"] == 1
    assert round_manifest["counters"]["optimizer_steps"] == 0
    assert round_manifest["counters"]["backward_calls"] == 0
    for receipt in (base_receipt, refold_receipt):
        assert receipt["parent"]["parent_forward_count_after_cache"] == 0
        assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
        assert receipt["training_steps"] == 0
    assert base_receipt["parent"]["on_policy_parent_capture_invocations"] == 1
    assert refold_receipt["parent"]["on_policy_parent_capture_invocations"] == 0

    # 6. Both children are real published artifacts on the target vocabulary.
    for directory in ("published-child", "published-child-round-0-refold"):
        assert (run / directory / "child" / "tokenizer.json").is_file()
    for receipt in (base_receipt, refold_receipt):
        assert receipt["artifact"]["fresh_reload_accepted"] is True
    assert (
        base_receipt["target"]["exact_unique_parameters"]
        == refold_receipt["target"]["exact_unique_parameters"]
    )

    # 7. Each pass journals its own ordered progress into one shared ledger.
    for pass_record in (base_pass, refold_pass):
        pass_rows = [
            row
            for row in rows
            if row["configuration_sha256"] == pass_record["configuration_sha256"]
        ]
        assert [row["sequence"] for row in pass_rows] == list(
            range(len(pass_rows))
        )
        assert pass_rows[-1]["stage"] == "final_receipt"
        _assert_ordered_stages(
            [str(row["stage"]) for row in pass_rows],
            _REQUIRED_FOLD_STAGES,
            context=str(pass_record["status"]),
        )
    assert base_pass["configuration_sha256"] != (
        refold_pass["configuration_sha256"]
    )

    # 8. A tampered target preimage fails closed even with every seal in the
    #    manifest recomputed: the live transport profile no longer reproduces
    #    the donor batch the sealed extension cache actually holds.
    tampered = _round_manifest_document(manifest_path)
    record = tampered["target_preimage"]["records"][0]
    original = int(record["target_input_ids"][0])
    replacement = next(
        value
        for value in range(len(mapping))
        if value != original and mapping[value] != mapping[original]
    )
    record["target_input_ids"][0] = replacement
    record["target_token_ids_sha256"] = MODULE_ROUND._target_token_ids_sha256(
        record["target_input_ids"],
        record["target_forced_decode_ids"],
    )
    tampered_path = _reseal_round_manifest(
        tampered,
        tmp_path / "tampered" / "round-manifest.json",
    )
    tampered_round = MODULE_ROUND.load_lfm2_on_policy_evidence_round(
        tampered_path
    )
    assert tampered_round.carries_target_preimage is True
    extension_cache = open_lfm2_circuit_trace_cache(
        tampered_round.extension_cache_path,
        verify=False,
    )
    with pytest.raises(
        MODULE_ROUND.Lfm2OnPolicyEvidenceRoundBridgeError,
        match="does not map onto sealed extension batch",
    ):
        MODULE_ROUND.build_lfm2_on_policy_round_target_records(
            tampered_round,
            bridge=artifact,
            extension_cache=extension_cache,
        )
    # The same tamper without recomputing the block seal is caught earlier, by
    # the seal itself.
    unsealed = _round_manifest_document(manifest_path)
    unsealed["target_preimage"]["records"][0]["target_input_ids"][0] = (
        replacement
    )
    body = {
        name: value
        for name, value in unsealed.items()
        if name != "manifest_sha256"
    }
    unsealed_path = tmp_path / "unsealed" / "round-manifest.json"
    unsealed_path.parent.mkdir(parents=True, exist_ok=True)
    unsealed_path.write_bytes(
        json.dumps(
            {**body, "manifest_sha256": MODULE.canonical_sha256(body)},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(
        MODULE_ROUND.Lfm2OnPolicyEvidenceRoundIntegrityError,
        match="target_preimage_sha256 does not match",
    ):
        MODULE_ROUND.load_lfm2_on_policy_evidence_round(unsealed_path)

    # 9. An old-format round — one captured before target preimages existed —
    #    is refused on this lane by name, at configuration time.
    legacy = _round_manifest_document(manifest_path)
    legacy.pop("target_preimage")
    legacy_path = _reseal_round_manifest(
        legacy,
        tmp_path / "legacy" / "round-manifest.json",
    )
    legacy_round = MODULE_ROUND.load_lfm2_on_policy_evidence_round(legacy_path)
    assert legacy_round.carries_target_preimage is False
    assert legacy_round.parent_cache_input_mode == (
        MODULE_ROUND.PARENT_CANONICAL_CAPTURE_INPUT_MODE
    )
    with pytest.raises(
        MODULE.ProductionLfm2CatalogError,
        match="seals no target-token preimage",
    ) as failure:
        MODULE.config_from_args(
            MODULE.build_parser().parse_args(
                [
                    *_lane_arguments(
                        pack,
                        tmp_path / "legacy-run",
                        carrier_routes=False,
                    ),
                    "--on-policy-round-manifest",
                    str(legacy_path),
                ]
            )
        )
    assert str(legacy_path) in str(failure.value)


def test_carrier_production_margin_pin_composes_through_a_round(
    refold_prompt_pack,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """The r13 deadlock: streaming-v2 margins on merged on-policy evidence.

    ``ProductionLfm2CatalogConfig.__post_init__`` requires
    ``dense_vocabulary_margin_observability`` and
    ``dense_streaming_vocabulary_margin_v2`` of every carrier-routed run, and
    the balanced-basis builder used to refuse streaming v2 on the merged session
    ``_resolve_dense_evidence_handles`` hands it whenever a round contributed
    extension evidence, so a carrier-routed refold consuming a round had no
    executable configuration in either margin setting.  This composes both pins
    with a real round and asserts the streaming adapter authenticated the merged
    handle it used to refuse.
    """

    pack = refold_prompt_pack
    run = tmp_path / "run"
    config = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _bridged_balanced_refold_lane_arguments(pack, run)
        )
    )

    # The two terms of the carrier production pin this lane can reach, and the
    # round consumption that used to be mutually exclusive with them.
    assert config.dense_residual_basis_mode == "circuit_balanced"
    assert config.dense_vocabulary_margin_observability is True
    assert config.dense_streaming_vocabulary_margin_v2 is True
    assert config.uses_tokenizer_bridge is True
    assert config.uses_balanced_basis_phase_cache is True
    assert config.on_policy_refold_after_rounds == 1
    assert MODULE._unconsumed_extension_evidence_reason(config) is None

    # The routed configuration itself still constructs with both pins intact,
    # so nothing here weakened the validator that named them.
    routed = MODULE.config_from_args(
        MODULE.build_parser().parse_args(
            _alternation_arguments(
                pack,
                tmp_path / "routed",
                _lane_arguments(
                    pack,
                    tmp_path / "routed",
                    carrier_routes=True,
                ),
            )
        )
    )
    assert routed.uses_carrier_routes is True
    assert routed.dense_vocabulary_margin_observability is True
    assert routed.dense_streaming_vocabulary_margin_v2 is True

    # Observer only: the real streaming adapter still runs and still seals its
    # own evidence; this records which handle each solve pass gave it.
    from neural_forge import (
        hf_lfm2_streaming_vocabulary_margin_adapter as margin_adapter,
    )

    real_adapter = (
        margin_adapter.build_lfm2_streaming_vocabulary_margin_adapter_v2
    )
    margin_builds: list[dict[str, Any]] = []

    def observing_adapter(trace_cache, selective_session, **kwargs):
        result = real_adapter(trace_cache, selective_session, **kwargs)
        authentication = result.as_dict()["cache_authentication"]
        margin_builds.append(
            {
                "cache": type(trace_cache).__name__,
                "session": type(selective_session).__name__,
                "cache_key": authentication["cache_key"],
                "manifest_sha256": authentication["manifest_sha256"],
                "merged": authentication.get(
                    margin_adapter._MERGED_EVIDENCE_FIELD
                ),
            }
        )
        return result

    monkeypatch.setattr(
        margin_adapter,
        "build_lfm2_streaming_vocabulary_margin_adapter_v2",
        observing_adapter,
    )

    summary = MODULE.run_production_catalog_on_policy_alternation(config)
    rows = _progress_rows(capsys.readouterr().out)

    assert summary["status"] == "complete", summary.get("failure")
    assert summary["executed_pass_count"] == summary["expected_pass_count"] == 2
    base_pass, refold_pass = summary["passes"]
    assert base_pass["status"] == refold_pass["status"] == "complete", summary

    base_receipt = json.loads((run / "receipt.json").read_text(encoding="ascii"))
    refold_receipt = json.loads(
        (run / "receipt-round-0-refold.json").read_text(encoding="ascii")
    )
    round_manifest = _round_manifest_document(
        Path(base_pass["sealed_round_manifests"][0])
    )
    extension_batches = int(round_manifest["extension_cache"]["batch_count"])

    # 1. The refold really consumed base+extension evidence.
    evidence = refold_receipt["on_policy_evidence"]
    assert evidence["consumed_by_this_solve_pass"] is True
    evidence_set = evidence["evidence_set"]
    assert evidence_set["member_count"] == 2
    assert evidence_set["batch_count"] == len(_TRAJECTORIES) + extension_batches
    assert evidence_set["extension_selection_or_validation_batch_count"] == 0
    assert base_receipt["on_policy_evidence"]["evidence_identity_sha256"] is None
    assert len(evidence["evidence_identity_sha256"]) == 64

    # 2. The base pass bound one sealed cache and the refold bound the merged
    #    evidence set -- the exact handle the adapter used to refuse by name.
    assert len(margin_builds) == 2
    base_build, refold_build = margin_builds
    assert base_build["cache"] == "Lfm2CircuitTraceCache"
    assert base_build["merged"] is None
    assert len(base_build["manifest_sha256"]) == 64
    assert refold_build["cache"] == "Lfm2MergedSealedEvidenceCache"
    assert refold_build["session"] == "Lfm2MergedSealedEvidenceSession"
    assert refold_build["cache_key"] == evidence_set["evidence_set_cache_key"]
    # No sealed manifest digest is invented for a corpus that has none.
    assert refold_build["manifest_sha256"] is None

    # 3. The margin evidence names which members and which composition it came
    #    from, with each member's own real sealed manifest digest.
    document = refold_build["merged"]
    assert document["schema"] == (
        margin_adapter.LFM2_STREAMING_VOCABULARY_MARGIN_MERGED_EVIDENCE_SCHEMA
    )
    assert document["evidence_set_cache_key"] == (
        evidence_set["evidence_set_cache_key"]
    )
    assert document["base_cache_key"] == refold_receipt["parent"][
        "parent_cache_key"
    ]
    assert document["on_policy_evidence_identity_sha256"] == (
        evidence["evidence_identity_sha256"]
    )
    assert document["member_count"] == 2
    assert document["batch_count"] == evidence_set["batch_count"]
    assert document["base_batch_count"] == len(_TRAJECTORIES)
    assert [value["role"] for value in document["ordered_members"]] == [
        "base",
        "on_policy_extension",
    ]
    assert [
        len(value["trace_cache"]["manifest_sha256"])
        for value in document["ordered_members"]
    ] == [64, 64]
    assert document["ordered_members"][1]["round_manifest_sha256"] == (
        round_manifest["manifest_sha256"]
    )
    assert document["ordered_members"][1]["all_rows_are_fit_rows"] is True
    assert MODULE.canonical_sha256(
        {
            name: value
            for name, value in document.items()
            if name != f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"
        }
    ) == document[f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"]

    # 4. The balanced basis was solved on that merged evidence through the phase
    #    cache, and only the refold pass bound merged evidence.  Each pass is
    #    its own compiler run, so each also gets its own phase event log; one
    #    log holding both run IDs is refused by the log itself.
    assert (run / "balanced-basis-events.jsonl").is_file()
    assert (run / "balanced-basis-events-round-0-refold.jsonl").is_file()
    bound = [
        row
        for row in rows
        if row["stage"] == "balanced_basis_merged_evidence_bound"
    ]
    assert bound
    assert {row["configuration_sha256"] for row in bound} == {
        refold_pass["configuration_sha256"]
    }
    # The phase binding and the margin evidence are two independent documents
    # over the same composition, so they agree on the identity and not on the
    # digest.
    assert bound[0]["evidence_set_cache_key"] == document[
        "evidence_set_cache_key"
    ]
    assert bound[0]["on_policy_evidence_identity_sha256"] == document[
        "on_policy_evidence_identity_sha256"
    ]
    assert bound[0]["member_count"] == document["member_count"]

    # 5. Neither pass regenerated a parent trace, and both children reload.
    for receipt in (base_receipt, refold_receipt):
        assert receipt["parent"]["parent_forward_count_after_cache"] == 0
        assert receipt["parent"]["parent_trace_regeneration_after_cache"] == 0
        assert receipt["training_steps"] == 0
        assert receipt["artifact"]["fresh_reload_accepted"] is True
    assert "on_policy_extension_evidence_materialized" in _stages(rows)
