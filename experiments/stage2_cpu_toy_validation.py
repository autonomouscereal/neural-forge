"""CPU toy validation for the stage-2 distillation trainer.

**No production training run happens here.**  This script builds a tiny
teacher/student pair with the *real* carrier ABI -- a sealed published child
directory with packaged ``C_in``/``C_out`` routes, a semantic-source lexicon,
and the outer child receipt -- trains it for a few hundred CPU steps, and
proves the six properties the trainer has to have before a 3090 is allowed
anywhere near it:

1. the trainable route surface is **bitwise identical** to the frozen decode
   runtime's route surface;
2. the loss decreases;
3. the GKD phase switches at the token threshold and the post-switch on-policy
   fraction is exactly the pre-registered one;
4. receipts seal, reload, and refuse tampering;
5. the free-run probe runs and reports only sanctioned metrics; and
6. a rung checkpoint round-trips through the loader the frozen review uses --
   and a trained-route checkpoint is refused by the stage-1 guard at its exact
   message, which is the handoff to lane 2's admission path.

Run it directly to print the JSON evidence bundle:

``.venv/Scripts/python.exe experiments/stage2_cpu_toy_validation.py``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)
from neural_forge.lfm2_carrier_bridge_publication import (  # noqa: E402
    STANDALONE_BRIDGE_SCHEMA,
    carrier_materialization_tensor_sha256,
)
from neural_forge.lfm2_carrier_child_package import (  # noqa: E402
    Lfm2CarrierChildCheckpointIO,
    Lfm2CarrierChildPackageRegistration,
    load_authenticated_lfm2_carrier_child_package,
)
from neural_forge.stage2_carrier_student import (  # noqa: E402
    STAGE2_FOLD_INITIALIZATION,
    STAGE2_RANDOM_INITIALIZATION,
    load_stage2_routed_student,
)
from neural_forge.stage2_checkpoint_layout import (  # noqa: E402
    STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF,
    round_trip_stage2_rung_checkpoint,
    stage2_route_provenance,
)
from neural_forge.stage2_distillation_trainer import (  # noqa: E402
    Stage2CheckpointSettings,
    Stage2CorpusBatch,
    Stage2DistillationTrainer,
    Stage2ProbeSettings,
    Stage2TrainerConfig,
)
from neural_forge.stage2_freerun_probe import (  # noqa: E402
    load_stage2_freerun_probe_prompts,
)
from neural_forge.stage2_kd_objective import (  # noqa: E402
    Stage2DonorVocabulary,
    Stage2ObjectiveSchedule,
)
from neural_forge.stage2_training_receipt import (  # noqa: E402
    STAGE2_CORPUS_SHARD_PATH_TEMPLATE,
    STAGE2_DESIGN_FILENAME,
    assert_preregistration_unedited,
    corpus_manifest_teacher_sha256,
    file_sha256,
    load_stage2_training_receipt,
    verify_corpus_consumption,
    write_stage2_training_receipt,
)


TOY_DONOR_ROWS = 24
TOY_TARGET_ROWS = 12
TOY_HIDDEN_SIZE = 16
TOY_TEACHER_HIDDEN_SIZE = 24
TOY_SEQUENCE_LENGTH = 12
TOY_CONTINUATION_TOKENS = 3
TOY_BATCH_SEQUENCES = 4
TOY_MICRO_BATCH_SEQUENCES = 2
TOY_PROBE_PROMPTS = 4
TOY_PROBE_NEW_TOKENS = 6


# ---------------------------------------------------------------------------
# A tiny causal body with the carrier ABI the routed runtime requires.
# ---------------------------------------------------------------------------


class _ToyBody(nn.Module):
    """Causal prefix-mean mixing plus a gated MLP, residual, one block.

    It is deliberately boring: the point of the toy is the *plumbing* and the
    accounting, not an architecture claim.  It accepts the exact keyword
    signature the carrier decode runtime calls bodies with.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.down = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Any = None,
    ) -> Tensor:
        del attention_mask, use_cache, past_key_values
        length = int(inputs_embeds.shape[1])
        weights = torch.tril(
            torch.ones(length, length, dtype=inputs_embeds.dtype)
        )
        weights = weights / weights.sum(dim=1, keepdim=True)
        pooled = weights @ inputs_embeds
        hidden = inputs_embeds + self.mix(self.norm(pooled))
        return hidden + self.down(torch.nn.functional.silu(self.up(hidden)))


class _ToyChild(nn.Module):
    """The carrier-ABI child: untied tables, callable body at ``model``."""

    def __init__(
        self,
        target_rows: int = TOY_TARGET_ROWS,
        hidden_size: int = TOY_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(target_rows, hidden_size)
        self.lm_head = nn.Linear(hidden_size, target_rows, bias=False)
        self.model = _ToyBody(hidden_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class _ToyTeacher(nn.Module):
    """A frozen donor-vocabulary language model with the same call contract."""

    def __init__(
        self,
        donor_rows: int = TOY_DONOR_ROWS,
        hidden_size: int = TOY_TEACHER_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(donor_rows, hidden_size)
        self.body = _ToyBody(hidden_size)
        self.head = nn.Linear(hidden_size, donor_rows, bias=False)

    def forward(self, donor_ids: Tensor) -> Tensor:
        return self.head(self.body(self.embed(donor_ids)))


class _ToyBaseCheckpointIO:
    """The base saver/loader the carrier child package wraps."""

    saver_id = "stage2-toy-safetensors-tokenizer-v1"
    loader_id = "stage2-toy-safetensors-tokenizer-v1"
    resource_fingerprint = "stage2-toy-base-checkpoint"

    def __init__(self, tokenizer_source: Path | None) -> None:
        self.tokenizer_source = tokenizer_source

    def save(self, model: _ToyChild, destination: Path) -> None:
        if self.tokenizer_source is None:
            raise RuntimeError("The read-only base loader cannot save.")
        destination = Path(destination)
        destination.mkdir(parents=True)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        save_file(state, str(destination / "model.safetensors"))
        (destination / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "lfm2",
                    "target_rows": int(model.embed_tokens.weight.shape[0]),
                    "hidden_size": int(model.embed_tokens.weight.shape[1]),
                },
                sort_keys=True,
            ),
            encoding="ascii",
        )
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(self.tokenizer_source / name, destination / name)

    def load(self, source: Path) -> _ToyChild:
        state = load_file(str(Path(source) / "model.safetensors"), device="cpu")
        rows, hidden_size = state["embed_tokens.weight"].shape
        model = _ToyChild(int(rows), int(hidden_size))
        model.load_state_dict(state)
        return model


# ---------------------------------------------------------------------------
# A real sealed published child, built from the real packaging code.
# ---------------------------------------------------------------------------


def _tokenizer_package(path: Path, vocabulary: dict[str, int]) -> tuple[Any, str]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir(parents=True)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"unk_token": "<unk>", "eos_token": "<eos>"}),
        encoding="utf-8",
    )
    tokenizer = load_lfm2_direct_local_tokenizer(path)
    return tokenizer, fingerprint_hf_tokenizer_semantics(tokenizer)


# Every target row is protected by one distinct donor row that must stay an
# exact one-hot at coefficient 1.0 (``_validated_sparse_semantic_code``).  The
# toy reproduces that contract exactly -- donor ``t`` anchors target ``t`` --
# because it is the contract that decides whether a *trained* route can decode
# at all.
TOY_TARGET_TO_DONOR = tuple(range(TOY_TARGET_ROWS))


def _toy_spans() -> list[list[int]]:
    """One nonempty ascending target span per donor row, anchors one-hot."""

    spans: list[list[int]] = []
    for donor in range(TOY_DONOR_ROWS):
        if donor < TOY_TARGET_ROWS:
            spans.append([donor])
            continue
        rows = {
            (donor * 5 + offset * 3 + 1) % TOY_TARGET_ROWS
            for offset in range(1 + (donor % 3))
        }
        spans.append(sorted(rows))
    return spans


def _csr_from_spans(
    spans: Sequence[Sequence[int]],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    offsets = [0]
    target_ids: list[int] = []
    coefficients: list[float] = []
    anchored = {int(donor): target for target, donor in enumerate(TOY_TARGET_TO_DONOR)}
    for donor, span in enumerate(spans):
        for row in span:
            target_ids.append(int(row))
        if donor in anchored:
            coefficients.append(1.0)
        else:
            values = torch.rand(len(span), generator=generator) * 0.8 + 0.2
            coefficients.extend(float(value) for value in values)
        offsets.append(len(target_ids))
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(target_ids, dtype=torch.int64),
        torch.tensor(coefficients, dtype=torch.float32),
    )


def _source_publication(
    root: Path,
    generator: torch.Generator,
) -> tuple[Path, str, list[list[int]]]:
    publication = root / "publication"
    publication.mkdir(parents=True)
    spans = _toy_spans()
    input_code = _csr_from_spans(spans, generator)
    output_code = _csr_from_spans(spans, generator)
    tensors = {
        "target_table": torch.randn(
            TOY_TARGET_ROWS,
            TOY_HIDDEN_SIZE,
            generator=generator,
        ),
        "target_to_donor": torch.tensor(
            TOY_TARGET_TO_DONOR,
            dtype=torch.int64,
        ),
        "input_offsets": input_code[0],
        "input_target_ids": input_code[1],
        "input_coefficients": input_code[2],
        "output_offsets": output_code[0],
        "output_target_ids": output_code[1],
        "output_coefficients": output_code[2],
    }
    tensor_path = publication / "bridge.safetensors"
    save_file(tensors, str(tensor_path))
    coefficient_count = int(input_code[2].numel() + output_code[2].numel())
    body = {
        "schema": STANDALONE_BRIDGE_SCHEMA,
        "status": "complete",
        "bridge_tensor_file": tensor_path.name,
        "bridge_tensor_file_sha256": file_sha256(tensor_path),
        "runtime_requires_parent": False,
        "runtime_requires_external_sidecar": False,
        "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
        "integer_csr_routes_are_structural_metadata": True,
        "continuous_semantic_coefficient_count": coefficient_count,
        "physical_target_table_parameter_count": int(
            tensors["target_table"].numel()
        ),
        "target_table_dtype": str(tensors["target_table"].dtype),
        "operation_audit": {
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameter_pruning_calls": 0,
            "parameter_quantization_calls": 0,
            "parameter_sparsification_calls": 0,
            "parameter_sharing_calls": 0,
            "runtime_sidecar_parameters": 0,
        },
        "materialization_receipt": {
            "input_coefficients_sha256": carrier_materialization_tensor_sha256(
                input_code[2]
            ),
            "output_coefficients_sha256": carrier_materialization_tensor_sha256(
                output_code[2]
            ),
            "continuous_fold_coefficient_count": coefficient_count,
        },
    }
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    (publication / "bridge-receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return publication, str(receipt["artifact_sha256"]), spans


def build_toy_published_child(root: Path) -> dict[str, Any]:
    """Produce a sealed carrier child directory using the real packaging path."""

    generator = torch.Generator().manual_seed(20260808)
    semantic_source = root / "semantic-source-tokenizer"
    semantic_tokenizer, semantics = _tokenizer_package(
        semantic_source,
        {"<unk>": 0, "<eos>": 1, **{f"w{index}": index + 2 for index in range(30)}},
    )
    target_source = root / "target-tokenizer"
    target_tokenizer, _target_semantics = _tokenizer_package(
        target_source,
        {"<unk>": 0, "<eos>": 1},
    )
    publication, artifact_sha256, spans = _source_publication(root, generator)
    evidence = seal_lfm2_semantic_step_span_evidence(
        spans,
        target_row_count=TOY_TARGET_ROWS,
        lineage={"stage2_cpu_toy": True},
    )
    span_path = root / "spans.json"
    span_path.write_text(
        json.dumps(evidence, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    input_elements = int(sum(len(span) for span in spans))
    registration = Lfm2CarrierChildPackageRegistration(
        publication_directory=publication,
        expected_artifact_sha256=artifact_sha256,
        span_evidence_json=span_path,
        expected_span_evidence_sha256=str(evidence["evidence_sha256"]),
        expected_span_evidence_file_sha256=file_sha256(span_path),
        expected_target_rows=TOY_TARGET_ROWS,
        expected_hidden_size=TOY_HIDDEN_SIZE,
        expected_input_route_elements=input_elements,
        expected_output_route_elements=input_elements,
    )
    writer = Lfm2CarrierChildCheckpointIO(
        base=_ToyBaseCheckpointIO(target_source),
        registration=registration,
        semantic_source_tokenizer_directory=semantic_source,
        semantic_source_tokenizer_sha256=semantics,
        in_memory_semantic_source_tokenizer=semantic_tokenizer,
        in_memory_target_tokenizer=target_tokenizer,
    )
    child_directory = root / "published-child"
    child = _ToyChild()
    with torch.no_grad():
        for parameter in child.parameters():
            parameter.copy_(
                torch.randn(
                    tuple(parameter.shape),
                    generator=generator,
                )
                * 0.2
            )
    writer.save(child, child_directory)
    package = load_authenticated_lfm2_carrier_child_package(child_directory)
    return {
        "child_directory": child_directory,
        "target_tokenizer_directory": target_source,
        "semantic_source_tokenizer_sha256": semantics,
        "lexicon_package_sha256": str(
            package.receipt["semantic_source_lexicon_package_sha256"]
        ),
        "package_sha256": package.package_sha256,
        "spans": spans,
    }


# ---------------------------------------------------------------------------
# A toy corpus with a real shard manifest.
# ---------------------------------------------------------------------------


class ToyCorpusSource:
    """Two sha-manifested shards of teacher-sampled donor ids, read in order.

    The manifest is written in **lane 2's** actual shape
    (``experiments/stage2_generate_teacher_corpus.py``): shards sealed by index
    with ``tokens_sha256``, and the teacher digest under
    ``run.teacher_shard_sha256``.  The toy therefore exercises the real
    cross-lane interface rather than a convenient invention of it.
    """

    def __init__(self, root: Path, teacher: _ToyTeacher, *, seed: int) -> None:
        shard_root = root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(seed)
        self.shard_order_seed = seed
        self._shards: list[dict[str, Any]] = []
        self._paths: list[str] = []
        self._tokens: list[Tensor] = []
        for index in range(2):
            rows = _sample_teacher_corpus(
                teacher,
                sequences=32,
                sequence_length=TOY_SEQUENCE_LENGTH,
                generator=generator,
            )
            relative_path = STAGE2_CORPUS_SHARD_PATH_TEMPLATE.format(index=index)
            path = root / relative_path
            save_file({"donor_ids": rows}, str(path))
            self._shards.append(
                {
                    "shard_index": index,
                    "split": "train",
                    "tokens_sha256": file_sha256(path),
                    "token_count": int(rows.numel()),
                    "row_count": int(rows.shape[0]),
                }
            )
            self._paths.append(relative_path)
            self._tokens.append(rows)
        manifest_body = {
            "schema": "neural-forge-stage2-toy-corpus-manifest-v1",
            "run": {
                "stage": "stage2",
                "teacher_shard_sha256": toy_teacher_digest(teacher),
                "generation_seed": seed,
            },
            "sampler": {"temperature": 1.0, "top_p": 0.995},
            "shards": self._shards,
        }
        self.manifest = manifest_body
        manifest_path = root / "corpus_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_body, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="ascii",
        )
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = file_sha256(manifest_path)
        self.teacher_checkpoint_sha256 = corpus_manifest_teacher_sha256(
            manifest_body
        )
        self._cursor = 0
        self._shard_index = 0

    def next_batch(
        self,
        *,
        sequences: int,
        sequence_length: int,
    ) -> Stage2CorpusBatch:
        rows = self._tokens[self._shard_index]
        if self._cursor + sequences > int(rows.shape[0]):
            self._cursor = 0
            self._shard_index = (self._shard_index + 1) % len(self._tokens)
            rows = self._tokens[self._shard_index]
        batch = rows[self._cursor : self._cursor + sequences, :sequence_length]
        self._cursor += sequences
        shard = self._shards[self._shard_index]
        return Stage2CorpusBatch(
            donor_ids=batch.clone(),
            shards=(
                {
                    "relative_path": self._paths[self._shard_index],
                    "sha256": shard["tokens_sha256"],
                    "tokens_consumed": int(batch.numel()),
                },
            ),
        )


def toy_teacher_digest(teacher: _ToyTeacher) -> str:
    return canonical_sha256(
        {
            name: [round(float(value), 8) for value in tensor.reshape(-1).tolist()]
            for name, tensor in sorted(teacher.state_dict().items())
        }
    )


def _sample_teacher_corpus(
    teacher: _ToyTeacher,
    *,
    sequences: int,
    sequence_length: int,
    generator: torch.Generator,
) -> Tensor:
    """Ancestral teacher self-generation, the design's corpus in miniature."""

    rows = torch.randint(
        0,
        TOY_DONOR_ROWS,
        (sequences, 1),
        generator=generator,
    )
    with torch.no_grad():
        while int(rows.shape[1]) < sequence_length:
            probabilities = torch.softmax(teacher(rows)[:, -1, :], dim=-1)
            drawn = torch.multinomial(probabilities, 1, generator=generator)
            rows = torch.cat((rows, drawn), dim=1)
    return rows


# ---------------------------------------------------------------------------
# The validation itself.
# ---------------------------------------------------------------------------


def run_toy_validation(
    workspace: Path,
    *,
    steps: int = 240,
    mode: str = STAGE2_FOLD_INITIALIZATION,
    train_routes: bool = True,
    switch_steps: int = 100,
) -> dict[str, Any]:
    """Train the toy pair and return the evidence bundle as plain JSON."""

    workspace = Path(workspace)
    built = build_toy_published_child(workspace / "fixture")
    torch.manual_seed(11)
    teacher = _ToyTeacher()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    vocabulary = Stage2DonorVocabulary(
        defined_row_count=TOY_DONOR_ROWS,
        physical_row_count=TOY_DONOR_ROWS + 3,
    )

    def teacher_donor_logits(donor_ids: Tensor) -> Tensor:
        # The physical readout is wider than the defined donor rows, exactly
        # like the real parent; the extra rows are pushed to -inf so a caller
        # that forgets the mask still cannot sample one.
        logits = teacher(donor_ids)
        padding = torch.full(
            (*logits.shape[:-1], vocabulary.undefined_row_count),
            float("-inf"),
        )
        return torch.cat((logits, padding), dim=-1)

    student, decode_model, package = load_stage2_routed_student(
        built["child_directory"],
        base=_ToyBaseCheckpointIO(None),
        mode=mode,
        donor_vocabulary=vocabulary,
        train_routes=train_routes,
        seed=4242 if mode == STAGE2_RANDOM_INITIALIZATION else None,
    )
    # The equivalence check is only meaningful while the student still carries
    # the sealed fold weights and the sealed coefficients -- a random-init
    # control has already replaced both, so the frozen runtime would be
    # comparing against numbers that no longer exist.
    verify_route_surface = None
    if mode == STAGE2_FOLD_INITIALIZATION:
        from neural_forge.lfm2_carrier_bridge_publication import (
            bind_packaged_lfm2_carrier_routes,
        )

        verify_route_surface = (
            bind_packaged_lfm2_carrier_routes(
                decode_model,
                package.routes,
                expected_target_rows=TOY_TARGET_ROWS,
                expected_hidden_size=TOY_HIDDEN_SIZE,
            ),
            tuple(range(TOY_DONOR_ROWS)),
            torch.randn(5, TOY_TARGET_ROWS),
        )
    corpus = ToyCorpusSource(workspace / "corpus", teacher, seed=777)
    probe_prompts = load_stage2_freerun_probe_prompts(
        REPOSITORY_ROOT / "experiments" / "lfm2_on_policy_fit_prompts_v1.json",
        review_prompts_json=(
            REPOSITORY_ROOT
            / "research"
            / "lfm2_semantic_whole_response_prompts_v1.json"
        ),
        seed=8,
        prompt_count=TOY_PROBE_PROMPTS,
    )

    def tokenize(text: str) -> Sequence[int]:
        # The toy has no donor tokenizer of its own; a stable content hash of
        # the prompt gives every probe row a distinct, reproducible prefix.
        digest = canonical_sha256({"prompt": text})
        return [
            int(digest[index : index + 2], 16) % TOY_DONOR_ROWS
            for index in (0, 2, 4)
        ]

    objective = Stage2ObjectiveSchedule(
        # The toy scales the design's 10M-token switch down to a threshold the
        # CPU can actually cross, and keeps the pre-registered fractions.
        switch_tokens=(
            TOY_BATCH_SEQUENCES * TOY_SEQUENCE_LENGTH * int(switch_steps)
        ),
        continuation_tokens=TOY_CONTINUATION_TOKENS,
    )
    checkpoints = Stage2CheckpointSettings(
        root=workspace / "runs" / "toy-arm",
        base=_ToyBaseCheckpointIO(built["target_tokenizer_directory"]),
        provenance=stage2_route_provenance(
            package,
            coefficients_from_source_artifact=(
                mode == STAGE2_FOLD_INITIALIZATION
            ),
            initial_input_coefficients=student.input_coefficients.detach(),
            initial_output_coefficients=student.output_coefficients.detach(),
        ),
        structural_route_tensors=package.routes.tensors,
        initialization_child_directory=built["child_directory"],
        semantic_source_tokenizer_sha256=built[
            "semantic_source_tokenizer_sha256"
        ],
        lexicon_package_sha256=built["lexicon_package_sha256"],
        save_optimizer=lambda optimizer, path: torch.save(
            optimizer.state_dict(),
            str(path),
        ),
    )
    tokens_per_step = TOY_BATCH_SEQUENCES * TOY_SEQUENCE_LENGTH
    config = Stage2TrainerConfig(
        arm_id=f"toy-{mode}-{'trained' if train_routes else 'frozen'}-routes",
        seed=2026,
        # The design's 10M/100M/1B ladder scaled to CPU-toy size: three rungs
        # inside one run, the last one at the end of the run.
        rung_tokens=(
            tokens_per_step * max(steps // 4, 1),
            tokens_per_step * max(steps * 3 // 5, 2),
            tokens_per_step * steps,
        ),
        batch_sequences=TOY_BATCH_SEQUENCES,
        micro_batch_sequences=TOY_MICRO_BATCH_SEQUENCES,
        sequence_length=TOY_SEQUENCE_LENGTH,
        warmup_steps=10,
        enable_autocast=False,
        loss_log_interval_steps=5,
        safety_checkpoint_seconds=0.0,
    )
    trainer = Stage2DistillationTrainer(
        config=config,
        student=student,
        teacher_donor_logits=teacher_donor_logits,
        teacher_checkpoint_sha256=corpus.teacher_checkpoint_sha256,
        corpus=corpus,
        donor_vocabulary=vocabulary,
        objective=objective,
        preregistration=assert_preregistration_unedited(
            REPOSITORY_ROOT / "docs" / STAGE2_DESIGN_FILENAME
        ),
        device=torch.device("cpu"),
        checkpoints=checkpoints,
        probe=Stage2ProbeSettings(
            prompts=probe_prompts,
            tokenize=tokenize,
            teacher_donor_logits=teacher_donor_logits,
            maximum_new_tokens=TOY_PROBE_NEW_TOKENS,
            interval_tokens=tokens_per_step * 40,
            end_of_sequence_ids=(),
        ),
        device_inventory=({"index": 0, "name": "cpu-toy", "device_type": "cpu"},),
    )
    receipt = trainer.run(verify_route_surface=verify_route_surface)
    equivalence = trainer.route_surface_equivalence or None
    corpus_check = verify_corpus_consumption(
        trainer.corpus_consumption(),
        manifest=corpus.manifest,
        manifest_file_sha256=corpus.manifest_sha256,
    )
    receipt_path = workspace / "receipts" / f"{config.arm_id}.json"
    sealed = write_stage2_training_receipt(receipt_path, receipt)
    reloaded = load_stage2_training_receipt(
        receipt_path,
        expected_receipt_sha256=sealed,
    )
    tampered = _tamper_refusal(receipt_path)
    round_trips = [
        round_trip_stage2_rung_checkpoint(
            Path(checkpoints.root) / f"rung-{rung['rung_tokens']}",
            base=_ToyBaseCheckpointIO(None),
        )
        for rung in receipt["rungs"]
    ]
    losses = [row["loss"] for row in receipt["loss_curve"]]
    window = max(len(losses) // 5, 1)
    # The two branches are not on one scale -- forward KL is unbounded while
    # the generalized JSD at beta=0.5 is bounded by ln 2 -- so a mixed-phase
    # curve alternates by construction.  The per-branch series is the one that
    # can be read as progress.
    by_objective = {
        name: [
            row["loss"]
            for row in receipt["loss_curve"]
            if row["objective"] == name
        ]
        for name in ("forward-kl", "generalized-jsd-gkd")
    }
    per_objective = {
        name: {
            "points": len(series),
            "first": None if not series else series[0],
            "last": None if not series else series[-1],
            "decreased": bool(len(series) >= 2 and series[-1] < series[0]),
        }
        for name, series in by_objective.items()
    }
    return {
        "mode": mode,
        "train_routes": train_routes,
        "route_training_mask": student.route_training_mask(),
        "route_surface_equivalence": equivalence,
        "loss_curve_points": len(losses),
        "first_window_mean_loss": sum(losses[:window]) / window,
        "last_window_mean_loss": sum(losses[-window:]) / window,
        "loss_decreased": (
            sum(losses[-window:]) / window < sum(losses[:window]) / window
        ),
        "loss_by_objective": per_objective,
        "phase_switch_step": trainer.phase_switch_step,
        "phase_switch_tokens": objective.switch_tokens,
        "on_policy_steps": trainer.on_policy_steps,
        "off_policy_steps": trainer.off_policy_steps,
        "post_switch_steps": trainer.steps_in_phase,
        "post_switch_on_policy_fraction": (
            trainer.on_policy_steps / max(trainer.steps_in_phase, 1)
        ),
        "tokens_consumed": trainer.tokens_consumed,
        "optimizer_steps": trainer.optimizer_steps,
        "rungs": [
            {
                "rung_tokens": rung["rung_tokens"],
                "tokens_consumed": rung["tokens_consumed"],
                "optimizer_steps": rung["optimizer_steps"],
                "checkpoint_manifest_sha256": rung["checkpoint_manifest_sha256"],
                "free_run_token_match_fraction": (
                    None
                    if rung["free_run_probe"] is None
                    else rung["free_run_probe"]["token_match_fraction"]
                ),
                "free_run_mean_lcp": (
                    None
                    if rung["free_run_probe"] is None
                    else rung["free_run_probe"]["mean_longest_common_prefix"]
                ),
            }
            for rung in receipt["rungs"]
        ],
        "free_run_probe_points": len(receipt["free_run_probe_series"]),
        "receipt_sha256": sealed,
        # The sealed artifact itself travels in the bundle: the evidence is the
        # receipt, not a summary of it.
        "receipt": receipt,
        "receipt_reload_matches": reloaded["receipt_sha256"] == sealed,
        "receipt_tamper_refused": tampered,
        "corpus_manifest_check": corpus_check,
        "checkpoint_round_trips": [
            {
                key: value
                for key, value in result.items()
                if key not in {"model", "package"}
            }
            for result in round_trips
        ],
        "routed_decode_admission_handoff": dict(
            STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF
        ),
    }


def _tamper_refusal(receipt_path: Path) -> dict[str, Any]:
    """Flip one byte of a sealed receipt and confirm the reload refuses."""

    original = receipt_path.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["arm_id"] = f"{payload['arm_id']}-tampered"
    receipt_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    try:
        load_stage2_training_receipt(receipt_path)
    except Exception as exc:  # noqa: BLE001 - the refusal is the result
        refusal = {"refused": True, "message": str(exc)}
    else:
        refusal = {"refused": False, "message": ""}
    receipt_path.write_text(original, encoding="ascii")
    return refusal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--switch-steps", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=(STAGE2_FOLD_INITIALIZATION, STAGE2_RANDOM_INITIALIZATION),
        default=STAGE2_FOLD_INITIALIZATION,
    )
    parser.add_argument(
        "--routes",
        choices=("trained", "frozen"),
        default="trained",
        help=(
            "trained routes exercise the design default and meet the stage-1 "
            "guard; frozen routes prove the published-child round trip today"
        ),
    )
    arguments = parser.parse_args(argv)
    result = run_toy_validation(
        arguments.workspace,
        steps=arguments.steps,
        mode=arguments.mode,
        train_routes=arguments.routes == "trained",
        switch_steps=arguments.switch_steps,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
