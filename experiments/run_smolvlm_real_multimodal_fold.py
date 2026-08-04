#!/usr/bin/env python3
"""Fold a real pretrained multimodal parent, end to end, and decode from it.

``reports/cross-family-real-20260804-pretrained`` proved the cross-width branch
kernels against a real pretrained *text* checkpoint and read decoded
continuations out of the folded child.  Every multimodal proof in this tree so
far -- NEO native vision, the audio/video/points3d lanes -- ran against
synthesized parents whose vision front ends were built by the test that folded
them.  Nothing here has ever folded a real, downloaded, pretrained multimodal
checkpoint and asked the child what it sees.

This runner does that.  The parent is
``SmolVLM2-256M-Video-Instruct``: a SigLIP-shaped ``smolvlm_vision`` tower, a
pixel-shuffle connector, and a llama text decoder at exactly the geometry the
cross-width charts and branch kernels are already proven on (H576, 30 layers,
9 query heads over 3 key/value heads, head_dim 64).  The fold is:

* the **decoder** crosses H576 -> H432 through the sealed per-boundary chart
  set and the two promoted causal branch kernels, one action per branch per
  layer, exactly as the cross-family runner does;
* the **token interface** (embedding, head, final norm) is one
  ``token_interface_bridge`` action.  Parent and child share one tokenizer
  here, so ``CrossWidthTokenHeadInterfaceOperator`` from ``src`` executes it
  directly -- the strict row-identity path -- and no local composition of the
  pair-code interface is needed;
* the **vision tower** is carried through unchanged, tensor for tensor;
* the **connector** cannot be carried unchanged, because it writes into the
  residual stream whose width just contracted.  It is carried *through the same
  interface-locked chart* the embedding rows use, which is the only projection
  consistent with them.  That is stated as its own sealed carry receipt rather
  than smuggled in as a copy.

Evidence is one decoder forward over a batch that holds **both** text-only
windows and image-conditioned windows, so the residual basis the charts are fit
on has actually seen image tokens.  The image rows carry real vision-tower
output: deterministic RGB probe images go through the real
``SmolVLMImageProcessorPil`` and the real
``transformers.models.smolvlm.processing_smolvlm.get_image_prompt_string``, the
real vision tower and connector produce the image embeddings, and the real
``SmolVLMModel.inputs_merger`` substitutes them at the image-token rows.

``SmolVLMProcessor`` itself cannot be constructed in this environment:
``transformers`` 5.13.0 gates ``SmolVLMImageProcessorPil`` behind a torchvision
backend requirement it does not actually use, and ``SmolVLMProcessor.__init__``
requires a ``BaseVideoProcessor`` whose class object is a torchvision
placeholder here.  The image half of ``SmolVLMProcessor.__call__`` is therefore
reproduced from its own components, call for call; the video half is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_forge.dynamic_causal_operator_promotion import (  # noqa: E402
    CausalGatedMLPBranchOperator,
    CausalGQAAttentionBranchOperator,
    CrossWidthCausalBranchOperator,
    CrossWidthTokenHeadInterfaceOperator,
    DynamicCausalOperatorPromotionError,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    DynamicFoldExecutionError,
    DynamicFoldExecutor,
    ExactCopyOperator,
    FoldDecision,
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
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    SmolVlmCircuitInspector,
    inspect_model_circuits,
)
from neural_forge.hf_causal_circuit_evidence import (  # noqa: E402
    capture_hf_causal_circuit_evidence,
)
from neural_forge.hf_causal_cross_width_charts import (  # noqa: E402
    ATTENTION_INPUT_BOUNDARY,
    CrossWidthResidualChartError,
    LiveChildBoundaryRecapture,
    ResidualBoundaryChartSet,
    compile_residual_boundary_chart_set,
    residual_boundary_id,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)
from neural_forge.native_vision_probe import canonical_image_sha256  # noqa: E402


RUNNER_SCHEMA = "neural-forge-smolvlm-real-multimodal-fold-runner-v1"
TOKEN_INTERFACE_BRIDGE_METHOD = "token_interface_bridge"
PROBE_SCHEMA = "neural-forge-smolvlm-rgb-probe-v1"

PRETRAINED_WEIGHT_FILE = "model.safetensors"
PRETRAINED_CONFIG_FILE = "config.json"
PRETRAINED_TOKENIZER_FILE = "tokenizer.json"
PRETRAINED_PREPROCESSOR_FILE = "preprocessor_config.json"
PRETRAINED_CHAT_TEMPLATE_FILE = "chat_template.json"

# The composite fields a SmolVLM2-256M checkpoint has to state before anything
# in this runner is allowed to touch it.  ``text_config`` is checked field by
# field because it is the half the charts and branch kernels are proven on.
DECLARED_COMPOSITE_CONFIG = {
    "model_type": "smolvlm",
    "architectures": ["SmolVLMForConditionalGeneration"],
    "scale_factor": 4,
    "image_token_id": 49190,
    "vocab_size": 49280,
    "tie_word_embeddings": False,
}
DECLARED_TEXT_CONFIG = {
    "model_type": "llama",
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "head_dim": 64,
    "vocab_size": 49280,
    "rms_norm_eps": 1e-05,
}
DECLARED_VISION_CONFIG = {
    "model_type": "smolvlm_vision",
    "hidden_size": 768,
    "image_size": 512,
    "patch_size": 16,
}

STAGES = (
    "configuration",
    "probe",
    "evidence",
    "charts",
    "carry",
    "fold",
    "fold-failure",
    "behavior",
)

# Eight integer colors, no floating point anywhere in probe construction.
PROBE_COLORS = (
    ("red", (220, 35, 45)),
    ("green", (25, 165, 80)),
    ("blue", (35, 90, 220)),
    ("yellow", (235, 205, 35)),
    ("purple", (145, 65, 190)),
    ("orange", (235, 120, 30)),
    ("black", (20, 20, 20)),
    ("white", (250, 250, 250)),
)
PROBE_EDGE = 64
PROBE_BACKGROUND = (245, 245, 245)


class SmolVlmFoldError(RuntimeError):
    """The run cannot proceed and nothing was written past the last seal."""


# ---------------------------------------------------------------------------
# Sealed, content-addressed I/O
# ---------------------------------------------------------------------------


def _canonical_bytes(body: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``body`` plus the canonical digest of everything else in it."""

    content = {key: value for key, value in body.items() if key != "receipt_sha256"}
    return {**content, "receipt_sha256": canonical_sha256(content)}


def verify_receipt(document: Mapping[str, Any]) -> bool:
    """Recompute a receipt seal over its own body."""

    content = {key: value for key, value in document.items() if key != "receipt_sha256"}
    return document.get("receipt_sha256") == canonical_sha256(content)


def _write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_receipt(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(sealed))
    return sealed


def _read_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SmolVlmFoldError(f"Receipt is not readable JSON: {path}.") from exc
    if not isinstance(document, dict):
        raise SmolVlmFoldError(f"Receipt is not a JSON object: {path}.")
    if not verify_receipt(document):
        raise SmolVlmFoldError(f"Receipt seal is broken: {path}.")
    return document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path, skip: Sequence[str] = ()) -> list[dict[str, Any]]:
    skipped = set(skip)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in skipped:
            continue
        rows.append(
            {
                "path": name,
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return rows


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _relative_rmse(prediction: Any, target: Any) -> float:
    difference = (prediction - target).double()
    reference = target.double()
    denominator = float(reference.square().mean().sqrt().item())
    if denominator <= 0.0:
        raise SmolVlmFoldError("The comparison target has no energy.")
    return float(difference.square().mean().sqrt().item()) / denominator


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


# ---------------------------------------------------------------------------
# Deterministic integer RGB probes
# ---------------------------------------------------------------------------


def _color_probe(index: int) -> tuple[Any, str, str, str]:
    from PIL import Image, ImageDraw

    name, color = PROBE_COLORS[index % len(PROBE_COLORS)]
    inset = 6 + (index % 5)
    image = Image.new("RGB", (PROBE_EDGE, PROBE_EDGE), PROBE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (inset, inset, PROBE_EDGE - 1 - inset, PROBE_EDGE - 1 - inset), fill=color
    )
    return (
        image,
        "color",
        "What color is the square? Answer with one word.",
        name,
    )


def _count_probe(index: int) -> tuple[Any, str, str, str]:
    from PIL import Image, ImageDraw

    count = 1 + index % 4
    name, color = PROBE_COLORS[(index // 4) % len(PROBE_COLORS)]
    image = Image.new("RGB", (PROBE_EDGE, PROBE_EDGE), PROBE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    radius = 7
    for position in ((14, 14), (46, 14), (14, 46), (46, 46))[:count]:
        left, top = position[0] - radius, position[1] - radius
        draw.ellipse(
            (left, top, left + 2 * radius, top + 2 * radius),
            fill=color,
            outline=(20, 20, 20),
        )
    return (
        image,
        "count",
        f"How many {name} circles are shown? Answer with one digit.",
        str(count),
    )


def _spatial_probe(index: int) -> tuple[Any, str, str, str]:
    from PIL import Image, ImageDraw

    relations = ("left", "right", "above", "below")
    relation = relations[index % len(relations)]
    placement = {
        "left": ((16, 32), (48, 32)),
        "right": ((48, 32), (16, 32)),
        "above": ((32, 16), (32, 48)),
        "below": ((32, 48), (32, 16)),
    }[relation]
    image = Image.new("RGB", (PROBE_EDGE, PROBE_EDGE), PROBE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(placement, ((220, 35, 45), (35, 90, 220))):
        draw.rectangle((x - 9, y - 9, x + 9, y + 9), fill=color)
    return (
        image,
        "spatial",
        (
            "Where is the red square relative to the blue square? Answer left, "
            "right, above, or below."
        ),
        relation,
    )


def _split_probe(index: int) -> tuple[Any, str, str, str]:
    from PIL import Image, ImageDraw

    top_name, top_color = PROBE_COLORS[index % len(PROBE_COLORS)]
    bottom_name, bottom_color = PROBE_COLORS[(index + 3) % len(PROBE_COLORS)]
    image = Image.new("RGB", (PROBE_EDGE, PROBE_EDGE), PROBE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, PROBE_EDGE - 1, PROBE_EDGE // 2 - 1), fill=top_color)
    draw.rectangle(
        (0, PROBE_EDGE // 2, PROBE_EDGE - 1, PROBE_EDGE - 1), fill=bottom_color
    )
    del bottom_name
    return (
        image,
        "split",
        "What color is the top half of the image? Answer with one word.",
        top_name,
    )


PROBE_BUILDERS = (_color_probe, _count_probe, _spatial_probe, _split_probe)


def build_probe_records(count: int) -> list[dict[str, Any]]:
    """Build ``count`` deterministic probes, round robin over the builders.

    No random source is consulted anywhere: every coordinate and every channel
    value is an integer function of the probe index, so the same ``count``
    always yields the same images with the same digests on any machine.
    """

    if isinstance(count, bool) or count <= 0:
        raise SmolVlmFoldError("probe count must be a positive integer.")
    records: list[dict[str, Any]] = []
    for index in range(count):
        builder = PROBE_BUILDERS[index % len(PROBE_BUILDERS)]
        image, category, prompt, answer = builder(index // len(PROBE_BUILDERS))
        records.append(
            {
                "probe_index": index,
                "probe_id": f"{category}_{index:03d}",
                "category": category,
                "prompt": prompt,
                "expected_answer": answer,
                "image": image,
                "image_sha256": canonical_image_sha256(image),
            }
        )
    return records


# ---------------------------------------------------------------------------
# The real SmolVLM image path, without the processor class
# ---------------------------------------------------------------------------


class RealSmolVlmImagePath:
    """``SmolVLMProcessor.__call__``'s image branch, from its own components."""

    def __init__(self, directory: Path, *, image_seq_len: int) -> None:
        pil_module = importlib.import_module(
            "transformers.models.smolvlm.image_processing_pil_smolvlm"
        )
        self.processing_module = importlib.import_module(
            "transformers.models.smolvlm.processing_smolvlm"
        )
        self.image_processor = pil_module.SmolVLMImageProcessorPil.from_pretrained(
            str(directory), local_files_only=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True
        )
        self.image_seq_len = int(image_seq_len)
        self.image_token = "<image>"
        self.fake_image_token = "<fake_token_around_image>"
        self.global_image_token = "<global-img>"
        self.image_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.image_token)
        )

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "image_processor_class": type(self.image_processor).__name__,
            "image_processor_backend": str(
                getattr(self.image_processor, "backend", "")
            ),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_rows": int(len(self.tokenizer)),
            "image_seq_len": self.image_seq_len,
            "image_token_id": self.image_token_id,
            "expansion_callable": (
                f"{self.processing_module.__name__}.get_image_prompt_string"
            ),
            "do_image_splitting": False,
        }

    def encode(self, image: Any, prompt: str) -> dict[str, Any]:
        """One image, one question, in the model's own chat surface form."""

        vision = self.image_processor(
            [[image]], return_row_col_info=True, do_image_splitting=False
        )
        rows = vision.pop("rows")
        cols = vision.pop("cols")
        expansion = self.processing_module.get_image_prompt_string(
            rows[0][0],
            cols[0][0],
            self.image_seq_len,
            fake_token_around_image=self.fake_image_token,
            image_token=self.image_token,
            global_image_token=self.global_image_token,
        )
        text = f"<|im_start|>User:{expansion}{prompt}<end_of_utterance>\nAssistant:"
        encoded = self.tokenizer(text, add_special_tokens=False)
        pixel_values = torch.as_tensor(vision["pixel_values"]).to(torch.float32)
        pixel_attention_mask = torch.as_tensor(vision["pixel_attention_mask"])
        input_ids = [int(value) for value in encoded["input_ids"]]
        if sum(1 for value in input_ids if value == self.image_token_id) != (
            self.image_seq_len
        ):
            raise SmolVlmFoldError(
                "The expanded image prompt does not carry exactly "
                f"{self.image_seq_len} image tokens."
            )
        return {
            "text": text,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }


# ---------------------------------------------------------------------------
# The decoder view the charts, planner and kernels operate on
# ---------------------------------------------------------------------------


class SmolVlmTextDecoderModule(torch.nn.Module):
    """The composite's own decoder modules, presented as a causal LM.

    ``model.text_model`` and ``lm_head`` are *referenced*, not copied, so every
    tensor this view exposes is the same storage the composite serves from.
    Folding through this view folds the composite in place.
    """

    def __init__(self, composite: Any) -> None:
        super().__init__()
        self.config = composite.config.text_config
        self.model = composite.model.text_model
        self.lm_head = composite.lm_head

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
        )


class ImageTokenMergeHook:
    """Substitute real image embeddings at the sealed batch's image rows.

    ``collect_hf_transport_trace`` and ``capture_hf_causal_circuit_evidence``
    both refuse ``inputs_embeds``, and they are right to: a full-model evidence
    bundle whose token rows are not token rows cannot be audited.  The image
    rows of this batch *are* token rows -- ``<image>`` is a real vocabulary row
    -- and SmolVLM replaces their embeddings with connector output before the
    first decoder layer.  This hook performs exactly that substitution, through
    the model's own ``inputs_merger``, and fails closed on any batch other than
    the one it was sealed against.
    """

    def __init__(
        self,
        composite: Any,
        *,
        input_ids: Any,
        image_features: Any,
    ) -> None:
        self.composite = composite
        self.input_ids = input_ids.detach().clone()
        self.image_features = image_features.detach().clone()
        self.calls = 0

    def __call__(self, _module: Any, args: Any, output: Any) -> Any:
        if not args:
            raise SmolVlmFoldError(
                "The image merge hook requires positional token ids."
            )
        ids = args[0]
        if tuple(ids.shape) != tuple(self.input_ids.shape) or not bool(
            torch.equal(ids.detach().cpu(), self.input_ids)
        ):
            raise SmolVlmFoldError(
                "The image merge hook was reached with token ids it was not "
                "sealed against; image conditioning cannot be claimed for them."
            )
        self.calls += 1
        return self.composite.inputs_merger(ids, output, self.image_features)


# ---------------------------------------------------------------------------
# Parent and child
# ---------------------------------------------------------------------------


def pretrained_parent_identity(directory: Path) -> dict[str, Any]:
    """Verify the checkpoint's declared composition and seal its identity."""

    resolved = directory.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise SmolVlmFoldError(f"{resolved} is not a directory.")
    required = (
        PRETRAINED_CONFIG_FILE,
        PRETRAINED_WEIGHT_FILE,
        PRETRAINED_TOKENIZER_FILE,
        PRETRAINED_PREPROCESSOR_FILE,
        PRETRAINED_CHAT_TEMPLATE_FILE,
    )
    for name in required:
        if not (resolved / name).is_file():
            raise SmolVlmFoldError(
                f"A local SmolVLM parent needs {name} at {resolved}; a sharded "
                "checkpoint is not supported here."
            )
    config = json.loads((resolved / PRETRAINED_CONFIG_FILE).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SmolVlmFoldError(f"{resolved / PRETRAINED_CONFIG_FILE} is not an object.")
    mismatches = []
    for name, want in DECLARED_COMPOSITE_CONFIG.items():
        got = config.get(name)
        if got != want:
            mismatches.append(f"{name}: declared {want!r}, checkpoint {got!r}")
    for section, declared in (
        ("text_config", DECLARED_TEXT_CONFIG),
        ("vision_config", DECLARED_VISION_CONFIG),
    ):
        observed = config.get(section)
        if not isinstance(observed, dict):
            mismatches.append(f"{section}: missing")
            continue
        for name, want in declared.items():
            got = observed.get(name)
            if isinstance(want, float):
                agrees = (
                    isinstance(got, (int, float)) and abs(float(got) - want) <= 1e-12
                )
            else:
                agrees = got == want
            if not agrees:
                mismatches.append(
                    f"{section}.{name}: declared {want!r}, checkpoint {got!r}"
                )
    if mismatches:
        raise SmolVlmFoldError(
            f"The SmolVLM checkpoint at {resolved} does not match the declared "
            "composition: " + "; ".join(mismatches) + "."
        )
    return {
        "directory": str(resolved),
        "model_type": config["model_type"],
        "architectures": list(config["architectures"]),
        "weight_file": PRETRAINED_WEIGHT_FILE,
        "weight_file_bytes": int((resolved / PRETRAINED_WEIGHT_FILE).stat().st_size),
        "weight_file_sha256": _file_sha256(resolved / PRETRAINED_WEIGHT_FILE),
        "config_file_sha256": _file_sha256(resolved / PRETRAINED_CONFIG_FILE),
        "tokenizer_file_sha256": _file_sha256(resolved / PRETRAINED_TOKENIZER_FILE),
        "preprocessor_file_sha256": _file_sha256(
            resolved / PRETRAINED_PREPROCESSOR_FILE
        ),
        "chat_template_file_sha256": _file_sha256(
            resolved / PRETRAINED_CHAT_TEMPLATE_FILE
        ),
        "verified_composite_fields": dict(DECLARED_COMPOSITE_CONFIG),
        "verified_text_fields": dict(DECLARED_TEXT_CONFIG),
        "verified_vision_fields": dict(DECLARED_VISION_CONFIG),
        "loaded_dtype": "torch.float32",
    }


def load_pretrained_parent(identity: Mapping[str, Any], *, layers: int) -> Any:
    """Load the sealed checkpoint and keep the requested decoder prefix."""

    directory = Path(identity["directory"])
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        directory, local_files_only=True, dtype=torch.float32
    )
    if type(model).__name__ != "SmolVLMForConditionalGeneration":
        raise SmolVlmFoldError(
            f"The checkpoint at {directory} loaded as {type(model).__name__}, "
            "not SmolVLMForConditionalGeneration."
        )
    depth = int(model.config.text_config.num_hidden_layers)
    if layers > depth:
        raise SmolVlmFoldError(
            f"The checkpoint has {depth} decoder layers; {layers} were requested."
        )
    if layers < depth:
        model.model.text_model.layers = torch.nn.ModuleList(
            list(model.model.text_model.layers[:layers])
        )
        model.config.text_config.num_hidden_layers = int(layers)
        model.model.text_model.config.num_hidden_layers = int(layers)
    model.eval()
    return model


def build_child(
    parent: Any,
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    layers: int,
    seed: int,
) -> Any:
    """One blank SmolVLM at a narrower decoder residual width."""

    payload = parent.config.to_dict()
    text = dict(payload["text_config"])
    text.update(
        {
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "num_attention_heads": int(query_heads),
            "num_key_value_heads": int(key_value_heads),
            "num_hidden_layers": int(layers),
        }
    )
    payload["text_config"] = text
    config = type(parent.config).from_dict(payload)
    torch.manual_seed(int(seed))
    child = type(parent)(config)
    child.eval()
    if int(child.config.text_config.hidden_size) != int(hidden_size):
        raise SmolVlmFoldError("The blank child did not adopt the requested width.")
    if child.model.text_model.embed_tokens.weight is child.lm_head.weight:
        raise SmolVlmFoldError(
            "The blank child tied its embedding and head; the chart-projected "
            "interface installs them from two different charts."
        )
    return child


class AcceptStructuralCandidate:
    """Behavior stays the outer authority; structure is admitted here.

    The behavioral evidence this run produces is the decoded continuation
    corpus at the end, read by a human, on text-only *and* image-conditioned
    prompts.  Nothing inside the executor substitutes a scalar for that.
    """

    def evaluate(self, context: Any) -> FoldDecision:
        del context
        return FoldDecision.accept("Structurally complete cross-width branch.")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> tuple[str, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("prompts") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise SmolVlmFoldError(
            f"{path} must hold a nonempty JSON list of prompt strings."
        )
    prompts = []
    for row in rows:
        value = row.get("prompt") if isinstance(row, dict) else row
        if not isinstance(value, str) or not value.strip():
            raise SmolVlmFoldError("Every calibration prompt must be nonempty text.")
        prompts.append(value)
    return tuple(prompts)


def _pad_rows(
    rows: Sequence[Sequence[int]],
    window: int,
    pad_id: int,
) -> tuple[Any, Any, tuple[int, ...]]:
    """Right pad to one fixed window.

    Under causal attention a real token never attends to a later position, so
    right padding leaves every real row identical to what an unpadded forward
    would produce, and the padded rows leave the evidence through the mask.
    """

    ids: list[list[int]] = []
    masks: list[list[int]] = []
    lengths: list[int] = []
    for row in rows:
        values = [int(value) for value in row]
        if len(values) > window:
            raise SmolVlmFoldError(
                f"A window row needs {len(values)} tokens, above the requested "
                f"window of {window}."
            )
        missing = window - len(values)
        lengths.append(len(values))
        ids.append(values + [pad_id] * missing)
        masks.append([1] * len(values) + [0] * missing)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        tuple(lengths),
    )


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


def greedy_continuation(
    model: Any,
    input_ids: Any,
    *,
    new_tokens: int,
    pixel_values: Any | None = None,
    pixel_attention_mask: Any | None = None,
) -> list[int]:
    """Greedy decode with a KV cache; no sampling, no temperature, no beams.

    The image crop is supplied once, on the prefill, exactly as the composite's
    own generation path supplies it: after the prefill the image rows are
    already in the cache and later steps are ordinary token steps.
    """

    ids = input_ids.reshape(1, -1)
    attention = torch.ones_like(ids)
    produced: list[int] = []
    with torch.no_grad():
        kwargs: dict[str, Any] = {}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            kwargs["pixel_attention_mask"] = pixel_attention_mask
        output = model(
            input_ids=ids, attention_mask=attention, use_cache=True, **kwargs
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        for _ in range(new_tokens):
            token = int(logits.argmax(dim=-1).item())
            produced.append(token)
            attention = torch.cat(
                [attention, torch.ones((1, 1), dtype=attention.dtype)], dim=1
            )
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits = output.logits[:, -1, :]
    return produced


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _configuration_stage(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    checkpoint = pretrained_parent_identity(args.parent_model_directory)
    prompts = load_prompts(args.calibration_prompts)
    calibration_count = int(args.calibration_prompts_count)
    holdout_count = int(args.holdout_prompts_count)
    if calibration_count + holdout_count > len(prompts):
        raise SmolVlmFoldError(
            f"{args.calibration_prompts} holds {len(prompts)} prompts; the run "
            f"needs {calibration_count + holdout_count}."
        )
    calibration = prompts[:calibration_count]
    holdout = prompts[calibration_count : calibration_count + holdout_count]
    layers = int(args.layers)
    if not 1 <= layers <= DECLARED_TEXT_CONFIG["num_hidden_layers"]:
        raise SmolVlmFoldError(
            "--layers must select a prefix of the checkpoint's 30 decoder layers."
        )
    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "configuration",
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "torch_threads": int(torch.get_num_threads()),
        "parent_checkpoint": checkpoint,
        "parent": {
            "family": "smolvlm",
            "decoder_family": "llama",
            "decoder_hidden_size": DECLARED_TEXT_CONFIG["hidden_size"],
            "decoder_layers_folded": layers,
            "decoder_full_depth": DECLARED_TEXT_CONFIG["num_hidden_layers"],
            "bounded_layer_prefix": layers < DECLARED_TEXT_CONFIG["num_hidden_layers"],
            "vision_hidden_size": DECLARED_VISION_CONFIG["hidden_size"],
        },
        "child": {
            "family": "smolvlm",
            "decoder_family": "llama",
            "decoder_hidden_size": int(args.child_hidden_size),
            "decoder_intermediate_size": int(args.child_intermediate_size),
            "decoder_query_heads": int(args.child_query_heads),
            "decoder_key_value_heads": int(args.child_key_value_heads),
            "decoder_layers": layers,
            "seed": int(args.child_seed),
        },
        "corpus": {
            "calibration_prompts_file": str(args.calibration_prompts),
            "calibration_prompts_file_sha256": _file_sha256(args.calibration_prompts),
            "calibration_text_prompts": calibration_count,
            "holdout_text_prompts": holdout_count,
            "evidence_window": int(args.evidence_window),
            "holdout_window": int(args.holdout_window),
        },
        "probes": {
            "schema": PROBE_SCHEMA,
            "total": int(args.probe_images),
            "calibration": int(args.calibration_probe_images),
            "holdout": int(args.probe_images) - int(args.calibration_probe_images),
            "edge_pixels": PROBE_EDGE,
        },
        "ridge": float(args.ridge),
        "decode_new_tokens": int(args.decode_new_tokens),
    }
    body["input_sha256"] = canonical_sha256(body)
    receipt = _write_receipt(root / "configuration.json", body)
    return {
        "receipt": receipt,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "holdout": holdout,
        "layers": layers,
    }


def _probe_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build, save and seal the probe images the fold and the decode both use."""

    total = int(args.probe_images)
    calibration = int(args.calibration_probe_images)
    if not 0 < calibration < total:
        raise SmolVlmFoldError(
            "--calibration-probe-images must leave at least one held-out probe."
        )
    directory = root / "probes"
    directory.mkdir(parents=True, exist_ok=True)
    records = build_probe_records(total)
    rows: list[dict[str, Any]] = []
    for record in records:
        name = f"{record['probe_id']}.png"
        path = directory / name
        record["image"].save(path, format="PNG", optimize=False)
        reloaded = __import__("PIL.Image", fromlist=["Image"]).open(path)
        if canonical_image_sha256(reloaded) != record["image_sha256"]:
            raise SmolVlmFoldError(
                f"The saved probe {name} does not round trip its pixels."
            )
        rows.append(
            {
                "probe_index": record["probe_index"],
                "probe_id": record["probe_id"],
                "category": record["category"],
                "prompt": record["prompt"],
                "expected_answer": record["expected_answer"],
                "split": (
                    "calibration" if record["probe_index"] < calibration else "heldout"
                ),
                "file": f"probes/{name}",
                "file_sha256": _file_sha256(path),
                "image_sha256": record["image_sha256"],
            }
        )
    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "probe",
        "input_sha256": configuration["input_sha256"],
        "probe_schema": PROBE_SCHEMA,
        "generation": "integer-only deterministic PIL drawing, no random source",
        "edge_pixels": PROBE_EDGE,
        "total": total,
        "calibration": calibration,
        "heldout": total - calibration,
        "records": rows,
    }
    receipt = _write_receipt(root / "probe.json", body)
    return {"receipt": receipt, "records": records, "rows": rows}


def _build_evidence_batch(
    image_path: RealSmolVlmImagePath,
    *,
    text_prompts: Sequence[str],
    probes: Sequence[Mapping[str, Any]],
    window: int,
    pad_id: int,
) -> dict[str, Any]:
    """One batch: text-only windows first, image-conditioned windows after."""

    rows: list[list[int]] = []
    kinds: list[str] = []
    for prompt in text_prompts:
        ids = [
            int(value)
            for value in image_path.tokenizer(prompt, add_special_tokens=False)[
                "input_ids"
            ]
        ][:window]
        rows.append(ids)
        kinds.append("text")
    pixel_values = []
    pixel_masks = []
    image_texts: list[str] = []
    for probe in probes:
        encoded = image_path.encode(probe["image"], probe["prompt"])
        rows.append(encoded["input_ids"])
        kinds.append("image")
        pixel_values.append(encoded["pixel_values"])
        pixel_masks.append(encoded["pixel_attention_mask"])
        image_texts.append(encoded["text"])
    input_ids, attention_mask, lengths = _pad_rows(rows, window, pad_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "kinds": tuple(kinds),
        "pixel_values": torch.cat(pixel_values, dim=0),
        "pixel_attention_mask": torch.cat(pixel_masks, dim=0),
        "image_texts": tuple(image_texts),
    }


def _image_features(composite: Any, batch: Mapping[str, Any]) -> Any:
    """Run the composite's own vision tower and connector once."""

    with torch.no_grad():
        outputs = composite.model.get_image_features(
            batch["pixel_values"],
            batch["pixel_attention_mask"],
        )
    features = outputs.pooler_output
    return features.detach().to(dtype=torch.float32)


def _chart_summary(chart_set: ResidualBoundaryChartSet) -> dict[str, Any]:
    cycle: list[float] = []
    fluctuation: list[float] = []
    fit_retained: list[float] = []
    heldout_retained: list[float] = []
    unabsorbed: list[float] = []
    for chart in chart_set.charts:
        diagnostics = dict(chart.report["diagnostics"])
        cycle.append(float(diagnostics["cycle_defect"]))
        fluctuation.append(float(diagnostics["norm_commutation_row_fluctuation"]))
        absorption = diagnostics.get("absorption_demand")
        if absorption is not None:
            unabsorbed.append(float(absorption["unabsorbed_energy_fraction"]))
        chosen = dict(chart.report["lineage"].get("selection", {}))
        for name, sink in (
            ("fit_retained_energy_fraction", fit_retained),
            ("heldout_retained_energy_fraction", heldout_retained),
        ):
            value = chosen.get(name)
            if isinstance(value, (int, float)):
                sink.append(float(value))
    return {
        "boundary_count": len(chart_set.charts),
        "interface_locked_boundary_ids": [
            chart.boundary_id for chart in chart_set.charts if chart.interface_locked
        ],
        "cycle_defect": _summarize(cycle),
        "norm_commutation_row_fluctuation": _summarize(fluctuation),
        "fit_retained_energy_fraction": _summarize(fit_retained),
        "heldout_retained_energy_fraction": _summarize(heldout_retained),
        "absorption_demand_unabsorbed_fraction": _summarize(unabsorbed),
    }


def _carry_vision_and_connector(
    parent: Any,
    child: Any,
    chart_set: ResidualBoundaryChartSet,
) -> dict[str, Any]:
    """Copy the tower exactly; project the connector through boundary 0.

    The tower's geometry does not change, so every one of its tensors is an
    exact copy and the receipt proves it by digest.  The connector writes into
    the residual stream at the same boundary the token embedding writes, so the
    only projection consistent with the installed embedding rows is the *same*
    interface-locked chart applied to the connector's output coordinates.
    """

    embedding_chart = chart_set.chart(residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY))
    if not embedding_chart.interface_locked:
        raise SmolVlmFoldError(
            "The connector carry requires an interface-locked embedding chart."
        )
    parent_vision = dict(
        parent.model.vision_model.named_parameters(recurse=True, remove_duplicate=False)
    )
    child_vision = dict(
        child.model.vision_model.named_parameters(recurse=True, remove_duplicate=False)
    )
    if set(parent_vision) != set(child_vision):
        raise SmolVlmFoldError(
            "The child's vision tower does not expose the parent's tensors."
        )
    copied = 0
    elements = 0
    with torch.no_grad():
        for name, source in sorted(parent_vision.items()):
            target = child_vision[name]
            if tuple(source.shape) != tuple(target.shape):
                raise SmolVlmFoldError(
                    f"Vision tensor {name!r} differs in shape between parent and "
                    "child; the tower cannot be carried unchanged."
                )
            target.copy_(source)
            copied += 1
            elements += int(source.numel())
        for name, source in sorted(parent_vision.items()):
            if not bool(torch.equal(child_vision[name], source)):
                raise SmolVlmFoldError(
                    f"Vision tensor {name!r} did not carry through exactly."
                )
        source_projection = parent.model.connector.modality_projection.proj.weight
        target_projection = child.model.connector.modality_projection.proj.weight
        projected = embedding_chart.fold_consumer_weight(
            source_projection.detach().to(dtype=torch.float64).transpose(0, 1)
        ).transpose(0, 1)
        if tuple(projected.shape) != tuple(target_projection.shape):
            raise SmolVlmFoldError("The chart-projected connector has the wrong shape.")
        target_projection.copy_(projected.to(dtype=target_projection.dtype))
    return {
        "vision_tensors_copied": copied,
        "vision_elements_copied": elements,
        "vision_carry_method": "copy_exact",
        "vision_state_sha256": canonical_sha256(
            {
                "kind": "vision_tower_state",
                "tensors": [
                    [
                        name,
                        hashlib.sha256(
                            child_vision[name].detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                    ]
                    for name in sorted(child_vision)
                ],
            }
        ),
        "connector_carry_method": "chart_projected_interface_boundary_zero",
        "connector_boundary_id": embedding_chart.boundary_id,
        "connector_boundary_chart_sha256": embedding_chart.chart_sha256,
        "connector_source_shape": [int(size) for size in source_projection.shape],
        "connector_target_shape": [int(size) for size in target_projection.shape],
        "connector_elements": int(target_projection.numel()),
    }


def _fold_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
    probe: Mapping[str, Any],
    calibration: Sequence[str],
    layers: int,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """One decoder forward, one chart set, one carry, one executor run."""

    receipt_path = root / "fold.json"
    child_directory = root / "child"
    input_sha256 = canonical_sha256(
        {
            "configuration": configuration["input_sha256"],
            "probe": probe["receipt"]["receipt_sha256"],
        }
    )
    existing = _read_receipt(receipt_path)
    if (
        existing is not None
        and existing.get("input_sha256") == input_sha256
        and (child_directory / "config.json").is_file()
    ):
        child = transformers.AutoModelForImageTextToText.from_pretrained(
            child_directory, local_files_only=True, dtype=torch.float32
        )
        child.eval()
        if fingerprint_model_state(child) != existing["folded_child_state_sha256"]:
            raise SmolVlmFoldError(
                "The saved folded child does not match its sealed fingerprint."
            )
        return {"receipt": existing, "child": child, "resumed": True}

    started = time.perf_counter()
    directory = Path(checkpoint["directory"])
    image_path = RealSmolVlmImagePath(directory, image_seq_len=args.image_seq_len)
    tokenizer = image_path.tokenizer
    tokenizer_fingerprint = fingerprint_hf_tokenizer_semantics(tokenizer)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    parent = load_pretrained_parent(checkpoint, layers=layers)
    child = build_child(
        parent,
        hidden_size=args.child_hidden_size,
        intermediate_size=args.child_intermediate_size,
        query_heads=args.child_query_heads,
        key_value_heads=args.child_key_value_heads,
        layers=layers,
        seed=args.child_seed,
    )
    parent_view = SmolVlmTextDecoderModule(parent)
    child_view = SmolVlmTextDecoderModule(child)
    parent_view.eval()
    child_view.eval()
    blank_state_fingerprint = fingerprint_model_state(child_view)

    calibration_probes = [
        record
        for record in probe["records"]
        if record["probe_index"] < int(args.calibration_probe_images)
    ]
    batch = _build_evidence_batch(
        image_path,
        text_prompts=calibration,
        probes=calibration_probes,
        window=int(args.evidence_window),
        pad_id=pad_id,
    )
    vision_started = time.perf_counter()
    parent_features = _image_features(parent, batch)
    vision_seconds = time.perf_counter() - vision_started

    parent_hook = ImageTokenMergeHook(
        parent.model,
        input_ids=batch["input_ids"],
        image_features=parent_features,
    )
    parent_handle = parent_view.model.embed_tokens.register_forward_hook(parent_hook)

    parent_graph = inspect_model_circuits(
        parent_view,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    child_graph = inspect_model_circuits(
        child_view,
        model_name="child",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    parent_composite_graph = inspect_model_circuits(
        parent,
        model_name="parent-composite",
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    child_composite_graph = inspect_model_circuits(
        child, model_name="child-composite", tokenizer_fingerprint=tokenizer_fingerprint
    )
    composite_inspector = SmolVlmCircuitInspector().name
    for graph in (parent_composite_graph, child_composite_graph):
        if dict(graph.metadata)["inspector"] != composite_inspector:
            raise SmolVlmFoldError(
                "The composite graphs were not compiled by the SmolVLM inspector."
            )

    try:
        capture_started = time.perf_counter()
        evidence = capture_hf_causal_circuit_evidence(
            parent_view,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            parent_graph_fingerprint=parent_graph.fingerprint,
            metadata={
                "runner": RUNNER_SCHEMA,
                "corpus": (
                    "kernel-stage2 calibration text windows plus deterministic "
                    "RGB probe windows whose image rows carry real vision-tower "
                    "and connector output"
                ),
            },
        )
        capture_seconds = time.perf_counter() - capture_started
    finally:
        parent_handle.remove()
    if parent_hook.calls != 1:
        raise SmolVlmFoldError(
            f"The parent image merge ran {parent_hook.calls} times; the evidence "
            "path takes exactly one decoder forward."
        )

    image_rows = sum(1 for kind in batch["kinds"] if kind == "image")
    image_token_positions = int(
        (batch["input_ids"] == image_path.image_token_id).sum().item()
    )
    evidence_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "evidence",
        "input_sha256": input_sha256,
        "cached_parent_evidence_sha256": evidence.cache_key,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "parent_composite_graph_fingerprint": parent_composite_graph.fingerprint,
        "child_composite_graph_fingerprint": child_composite_graph.fingerprint,
        "parent_decoder_forward_calls": int(evidence.parent_forward_calls),
        "parent_vision_tower_forward_calls": 1,
        "parent_image_merge_calls": parent_hook.calls,
        "vision_seconds": vision_seconds,
        "sequences": int(batch["input_ids"].shape[0]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "text_only_rows": int(batch["input_ids"].shape[0]) - image_rows,
        "image_conditioned_rows": image_rows,
        "image_token_positions": image_token_positions,
        "image_tokens_per_row": int(args.image_seq_len),
        "unpadded_token_lengths": list(batch["lengths"]),
        "padded_positions": int(batch["input_ids"].numel() - sum(batch["lengths"])),
        "valid_rows": int(evidence.valid_token_mask.sum().item()),
        "fit_rows": int(evidence.fit_token_mask.sum().item()),
        "validation_rows": int(evidence.validation_token_mask.sum().item()),
        "layer_count": len(evidence.layers),
        "capture_seconds": capture_seconds,
        "image_processor": image_path.identity,
        "image_prompt_texts": list(batch["image_texts"]),
        "evidence_input_ids_sha256": canonical_sha256(
            {
                "kind": "evidence_input_ids",
                "shape": [int(size) for size in batch["input_ids"].shape],
                "ids": [
                    int(value) for value in batch["input_ids"].reshape(-1).tolist()
                ],
            }
        ),
        "parent_image_feature_sha256": canonical_sha256(
            {
                "kind": "parent_image_features",
                "shape": [int(size) for size in parent_features.shape],
                "sha256": hashlib.sha256(
                    parent_features.cpu().numpy().tobytes()
                ).hexdigest(),
            }
        ),
    }
    _write_receipt(root / "evidence.json", evidence_body)

    chart_started = time.perf_counter()
    chart_set = compile_residual_boundary_chart_set(evidence, parent_view, child_view)
    chart_seconds = time.perf_counter() - chart_started
    _write_receipt(
        root / "charts.json",
        {
            "schema": RUNNER_SCHEMA,
            "stage": "charts",
            "input_sha256": input_sha256,
            "cached_parent_evidence_sha256": evidence.cache_key,
            "boundary_chart_set_sha256": chart_set.set_sha256,
            "boundary_chart_policy_sha256": chart_set.report["policy_sha256"],
            "source_width": chart_set.source_width,
            "target_width": chart_set.target_width,
            "boundary_ids": list(chart_set.boundary_ids),
            "compile_seconds": chart_seconds,
            "summary": _chart_summary(chart_set),
        },
    )

    carry_started = time.perf_counter()
    carry = _carry_vision_and_connector(parent, child, chart_set)
    carry["seconds"] = time.perf_counter() - carry_started
    child_features = _image_features(child, batch)
    projected_features = (
        chart_set.chart(residual_boundary_id(0, ATTENTION_INPUT_BOUNDARY))
        .encode(parent_features)
        .to(dtype=torch.float32)
    )
    carry["child_image_feature_vs_chart_encoded_parent_relative_rmse"] = _relative_rmse(
        child_features, projected_features
    )
    carry["child_image_feature_shape"] = [int(size) for size in child_features.shape]
    if fingerprint_model_state(child_view) != blank_state_fingerprint:
        raise SmolVlmFoldError(
            "The vision/connector carry changed the decoder the fold owns."
        )
    _write_receipt(
        root / "carry.json",
        {
            "schema": RUNNER_SCHEMA,
            "stage": "carry",
            "input_sha256": input_sha256,
            "boundary_chart_set_sha256": chart_set.set_sha256,
            **carry,
        },
    )

    child_hook = ImageTokenMergeHook(
        child.model,
        input_ids=batch["input_ids"],
        image_features=child_features,
    )
    child_handle = child_view.model.embed_tokens.register_forward_hook(child_hook)

    plan = DynamicFoldPlanner(
        enabled_experimental_method_ids=CAUSAL_LAYER_BRANCH_METHODS
    ).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=blank_state_fingerprint,
        evidence_fingerprint=evidence.cache_key,
        parent_state_fingerprints={"parent": fingerprint_model_state(parent_view)},
    )
    if not plan.ready or plan.gaps:
        child_handle.remove()
        raise SmolVlmFoldError(
            "The SmolVLM decoder cross-width plan is not gap free: "
            + ", ".join(sorted({gap.required_capability for gap in plan.gaps}))
        )
    recapture = LiveChildBoundaryRecapture(
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
    )
    registry = FoldOperatorRegistry(
        (
            ExactCopyOperator(),
            CrossWidthCausalBranchOperator(
                CausalGQAAttentionBranchOperator(ridge=args.ridge),
                chart_set=chart_set,
                recapture=recapture,
            ),
            CrossWidthCausalBranchOperator(
                CausalGatedMLPBranchOperator(ridge=args.ridge),
                chart_set=chart_set,
                recapture=recapture,
            ),
            CrossWidthTokenHeadInterfaceOperator(chart_set=chart_set, ridge=args.ridge),
        )
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent_view},
        child_model=child_view,
        parent_graphs={"parent": parent_graph},
        child_graph=child_graph,
    )
    parent_forwards = 0

    def counted(_module: Any, _args: Any, _output: Any) -> None:
        nonlocal parent_forwards
        parent_forwards += 1

    counter_handle = parent_view.register_forward_hook(counted)
    execute_started = time.perf_counter()
    try:
        run = DynamicFoldExecutor(
            operators=registry,
            evidence_provider=StaticEvidenceProvider(evidence.cache_key, evidence),
            evaluator=AcceptStructuralCandidate(),
        ).execute(plan, context)
    except DynamicCausalOperatorPromotionError as exc:
        # A fail-closed branch is a measurement, not a crash: seal where the
        # composition stopped so the depth it reached is on the record.
        _write_receipt(
            root / "fold-failure.json",
            {
                "schema": RUNNER_SCHEMA,
                "stage": "fold-failure",
                "input_sha256": input_sha256,
                "requested_layer_prefix": layers,
                "cached_parent_evidence_sha256": evidence.cache_key,
                "boundary_chart_set_sha256": chart_set.set_sha256,
                "child_recapture_calls": recapture.recapture_calls,
                "child_image_merge_calls": child_hook.calls,
                "failure_class": type(exc).__name__,
                "failure": str(exc),
                "child_decoder_state_rolled_back_to_blank": (
                    fingerprint_model_state(child_view) == blank_state_fingerprint
                ),
                "seconds": time.perf_counter() - execute_started,
            },
        )
        raise
    finally:
        counter_handle.remove()
        child_handle.remove()
    execute_seconds = time.perf_counter() - execute_started
    if run.status != "complete":
        raise SmolVlmFoldError(
            f"The fold run ended {run.status!r} rather than complete."
        )

    branches: list[dict[str, Any]] = []
    interface: dict[str, Any] = {}
    for attempt in run.attempts:
        report = dict(attempt.transaction_report)
        if attempt.method_id == TOKEN_INTERFACE_BRIDGE_METHOD:
            interface = {
                key: report[key]
                for key in (
                    "kind",
                    "shared_tokenizer_identity",
                    "final_norm_relative_rmse",
                    "boundary_chart_set_sha256",
                    "installed_parameter_paths",
                    "target_tied_storage",
                    "parent_residual_width",
                    "child_residual_width",
                )
                if key in report
            }
            continue
        if attempt.method_id not in {
            CAUSAL_ATTENTION_BRANCH_METHOD,
            CAUSAL_GATED_MLP_BRANCH_METHOD,
        }:
            continue
        key = (
            "validation_attention_delta_relative_rmse"
            if attempt.method_id == CAUSAL_ATTENTION_BRANCH_METHOD
            else "validation_mlp_delta_relative_rmse"
        )
        target = dict(report["sequential_branch_target"])
        row = {
            "method_id": attempt.method_id,
            "source_layer_index": report["source_layer_index"],
            "input_boundary_id": report["input_boundary_id"],
            "output_boundary_id": report["output_boundary_id"],
            "validation_endpoint_relative_rmse": report[key],
            "no_branch_endpoint_relative_rmse_baseline": report[
                "no_branch_endpoint_relative_rmse_baseline"
            ],
            "parent_residual_width": report["parent_residual_width"],
            "child_residual_width": report["child_residual_width"],
            "sequential_child_replay": report["sequential_child_replay"],
            "parent_forward_calls": report["parent_forward_calls"],
            "fit_upstream_error_energy_ratio": target[
                "fit_upstream_error_energy_ratio"
            ],
            "input_boundary_cycle_defect": report["input_boundary_chart"][
                "cycle_defect"
            ],
            "output_boundary_cycle_defect": report["output_boundary_chart"][
                "cycle_defect"
            ],
        }
        # A llama child normalizes neither queries nor keys per head, so the
        # coupled kernel reports ``identity`` and omits the fitted-gain fields
        # a qwen3 child would carry.  Record whichever the operator emitted.
        for optional in (
            "target_query_key_norm",
            "query_norm_gain_relative_rmse",
            "key_norm_gain_relative_rmse",
            "reverse_reclosure_sweep_executed",
            "parent_residual_coordinate_endpoints",
        ):
            if optional in report:
                row[optional] = report[optional]
        branches.append(row)

    child_directory.mkdir(parents=True, exist_ok=True)
    child.save_pretrained(child_directory, safe_serialization=True)
    tokenizer.save_pretrained(child_directory)
    folded_fingerprint = fingerprint_model_state(child)
    changed = [
        path
        for attempt in run.attempts
        for path in attempt.transaction_report["changed_state_paths"]
    ]
    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "fold",
        "input_sha256": input_sha256,
        "folded_layer_prefix": layers,
        "full_depth": DECLARED_TEXT_CONFIG["num_hidden_layers"],
        "bounded_layer_prefix": layers < DECLARED_TEXT_CONFIG["num_hidden_layers"],
        "plan": {
            "ready": bool(plan.ready),
            "gap_count": len(plan.gaps),
            "action_count": len(plan.actions),
            "method_counts": {
                method: sum(1 for action in plan.actions if action.method_id == method)
                for method in sorted({action.method_id for action in plan.actions})
            },
            "plan_fingerprint": plan.fingerprint,
            "partitions_every_learned_child_decoder_tensor": (
                sorted(
                    path
                    for action in plan.actions
                    for path in action.target_parameter_paths
                )
                == sorted(child_graph.parameter_paths)
            ),
        },
        "run": {
            "status": run.status,
            "fingerprint": run.fingerprint,
            "attempts": len(run.attempts),
            "accepted": sum(1 for attempt in run.attempts if attempt.accepted),
            "observed_parent_root_forward_calls": (
                run.observed_parent_root_forward_calls
            ),
            "observed_parent_module_forward_calls": (
                run.observed_parent_module_forward_calls
            ),
            "hooked_parent_forward_calls_during_execution": parent_forwards,
            "parent_forward_calls_during_evidence_capture": int(
                evidence.parent_forward_calls
            ),
            "optimizer_constructions": run.optimizer_constructions,
            "optimizer_steps": run.optimizer_steps,
            "backward_calls": run.backward_calls,
            "autograd_grad_calls": run.autograd_grad_calls,
            "pruning_operations": run.pruning_operations,
            "quantization_operations": run.quantization_operations,
            "executed_fold_static_zero_training_audit_passed": (
                run.executed_fold_static_zero_training_audit_passed
            ),
            "operator_static_zero_training_audit_passed": (
                run.operator_static_zero_training_audit_passed
            ),
            "child_recapture_calls": recapture.recapture_calls,
            "child_image_merge_calls": child_hook.calls,
            "changed_state_paths": len(changed),
            "changed_state_paths_unique": len(changed) == len(set(changed)),
            "completed_every_learned_child_decoder_tensor": (
                set(run.completed_target_parameter_paths)
                == set(child_graph.parameter_paths)
            ),
        },
        "graphs": {
            "parent_decoder_fingerprint": parent_graph.fingerprint,
            "child_decoder_fingerprint": child_graph.fingerprint,
            "parent_composite_fingerprint": parent_composite_graph.fingerprint,
            "child_composite_fingerprint": child_composite_graph.fingerprint,
            "composite_inspector": composite_inspector,
            "parent_composite_nodes": len(parent_composite_graph.nodes),
            "child_composite_nodes": len(child_composite_graph.nodes),
            "parent_composite_metadata": dict(parent_composite_graph.metadata),
            "child_composite_metadata": dict(child_composite_graph.metadata),
        },
        "parent": {
            "family": "smolvlm",
            "decoder_hidden_size": int(parent.config.text_config.hidden_size),
            "decoder_layers": int(parent.config.text_config.num_hidden_layers),
            "composite_parameters": _parameter_count(parent),
            "decoder_parameters": _parameter_count(parent_view),
            "vocab_size": int(parent.config.text_config.vocab_size),
            "decoder_state_sha256": fingerprint_model_state(parent_view),
            "pretrained": True,
            "pretrained_checkpoint": dict(checkpoint),
        },
        "child": {
            "family": "smolvlm",
            "decoder_hidden_size": int(child.config.text_config.hidden_size),
            "decoder_layers": int(child.config.text_config.num_hidden_layers),
            "composite_parameters": _parameter_count(child),
            "decoder_parameters": _parameter_count(child_view),
            "vocab_size": int(child.config.text_config.vocab_size),
            "blank_decoder_state_sha256": blank_state_fingerprint,
            "directory": child_directory.name,
        },
        "carry": carry,
        "branches": branches,
        "branch_summary": {
            "attention_validation_endpoint_relative_rmse": _summarize(
                [
                    row["validation_endpoint_relative_rmse"]
                    for row in branches
                    if row["method_id"] == CAUSAL_ATTENTION_BRANCH_METHOD
                ]
            ),
            "gated_mlp_validation_endpoint_relative_rmse": _summarize(
                [
                    row["validation_endpoint_relative_rmse"]
                    for row in branches
                    if row["method_id"] == CAUSAL_GATED_MLP_BRANCH_METHOD
                ]
            ),
        },
        "token_interface": interface,
        "folded_child_state_sha256": folded_fingerprint,
        "folded_child_decoder_state_sha256": fingerprint_model_state(child_view),
        "child_decoder_changed_by_fold": (
            fingerprint_model_state(child_view) != blank_state_fingerprint
        ),
        "seconds": {
            "vision_tower": vision_seconds,
            "evidence_capture": capture_seconds,
            "chart_compile": chart_seconds,
            "carry": carry["seconds"],
            "execute": execute_seconds,
            "stage_total": time.perf_counter() - started,
        },
    }
    receipt = _write_receipt(receipt_path, body)
    return {"receipt": receipt, "child": child, "resumed": False}


def _behavior_stage(
    args: argparse.Namespace,
    root: Path,
    configuration: Mapping[str, Any],
    probe: Mapping[str, Any],
    fold: Mapping[str, Any],
    holdout: Sequence[str],
    layers: int,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Teacher-forced agreement, then decoded continuations on both lanes."""

    receipt_path = root / "behavior.json"
    input_sha256 = canonical_sha256(
        {
            "configuration": configuration["input_sha256"],
            "probe": probe["receipt"]["receipt_sha256"],
            "fold": fold["receipt"]["receipt_sha256"],
            "decode_new_tokens": int(args.decode_new_tokens),
            "holdout_window": int(args.holdout_window),
        }
    )
    existing = _read_receipt(receipt_path)
    if existing is not None and existing.get("input_sha256") == input_sha256:
        return {"receipt": existing, "resumed": True}

    started = time.perf_counter()
    directory = Path(checkpoint["directory"])
    image_path = RealSmolVlmImagePath(directory, image_seq_len=args.image_seq_len)
    tokenizer = image_path.tokenizer
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    child = fold["child"]
    child.eval()
    parent = load_pretrained_parent(checkpoint, layers=layers)
    parent_view = SmolVlmTextDecoderModule(parent)
    if (
        fingerprint_model_state(parent_view)
        != (fold["receipt"]["parent"]["decoder_state_sha256"])
    ):
        raise SmolVlmFoldError(
            "The rebuilt parent does not match the parent the fold read."
        )
    # A layer prefix of a pretrained model is not a pretrained language model.
    # The untruncated checkpoint decodes the same prompts alongside it so the
    # distance between "reproduces its parent" and "is the model" is on the
    # record instead of being left to the reader.
    full_depth = int(DECLARED_TEXT_CONFIG["num_hidden_layers"])
    reference = (
        load_pretrained_parent(checkpoint, layers=full_depth)
        if layers < full_depth
        else None
    )

    holdout_probes = [
        record
        for record in probe["records"]
        if record["probe_index"] >= int(args.calibration_probe_images)
    ]
    text_rows = [
        [
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ][: int(args.holdout_window)]
        for prompt in holdout
    ]
    text_ids, text_mask, _lengths = _pad_rows(
        text_rows, int(args.holdout_window), pad_id
    )

    with torch.no_grad():
        child_logits = child(
            input_ids=text_ids, attention_mask=text_mask, use_cache=False
        ).logits
        parent_logits = parent(
            input_ids=text_ids, attention_mask=text_mask, use_cache=False
        ).logits
    valid = text_mask.reshape(-1).to(torch.bool)
    child_valid = child_logits.reshape(-1, child_logits.shape[-1])[valid]
    parent_valid = parent_logits.reshape(-1, parent_logits.shape[-1])[valid]
    text_teacher_forced = {
        "rows": int(valid.sum().item()),
        "logit_relative_rmse": _relative_rmse(child_valid, parent_valid),
        "centered_logit_relative_rmse": _relative_rmse(
            child_valid - child_valid.mean(dim=-1, keepdim=True),
            parent_valid - parent_valid.mean(dim=-1, keepdim=True),
        ),
        "top1_agreement": float(
            (child_valid.argmax(dim=-1) == parent_valid.argmax(dim=-1))
            .double()
            .mean()
            .item()
        ),
        "parent_distinct_top1_tokens": int(
            parent_valid.argmax(dim=-1).unique().numel()
        ),
    }
    modal = parent_valid.argmax(dim=-1)
    counts = torch.bincount(modal)
    text_teacher_forced["parent_modal_top1_token_id"] = int(counts.argmax().item())
    text_teacher_forced["parent_modal_top1_token"] = tokenizer.convert_ids_to_tokens(
        [int(counts.argmax().item())]
    )[0]
    text_teacher_forced["parent_modal_top1_floor"] = float(
        counts.max().item() / max(int(modal.numel()), 1)
    )

    image_encoded = [
        image_path.encode(record["image"], record["prompt"])
        for record in holdout_probes
    ]
    image_rows = [encoded["input_ids"] for encoded in image_encoded]
    image_ids, image_mask, _image_lengths = _pad_rows(
        image_rows, max(len(row) for row in image_rows), pad_id
    )
    image_pixels = torch.cat(
        [encoded["pixel_values"] for encoded in image_encoded], dim=0
    )
    image_pixel_mask = torch.cat(
        [encoded["pixel_attention_mask"] for encoded in image_encoded], dim=0
    )
    with torch.no_grad():
        child_image_logits = child(
            input_ids=image_ids,
            attention_mask=image_mask,
            pixel_values=image_pixels,
            pixel_attention_mask=image_pixel_mask,
            use_cache=False,
        ).logits
        parent_image_logits = parent(
            input_ids=image_ids,
            attention_mask=image_mask,
            pixel_values=image_pixels,
            pixel_attention_mask=image_pixel_mask,
            use_cache=False,
        ).logits
    image_valid = image_mask.reshape(-1).to(torch.bool)
    child_image_valid = child_image_logits.reshape(-1, child_image_logits.shape[-1])[
        image_valid
    ]
    parent_image_valid = parent_image_logits.reshape(-1, parent_image_logits.shape[-1])[
        image_valid
    ]
    image_modal = parent_image_valid.argmax(dim=-1)
    image_counts = torch.bincount(image_modal)
    image_teacher_forced = {
        "rows": int(image_valid.sum().item()),
        "logit_relative_rmse": _relative_rmse(child_image_valid, parent_image_valid),
        "centered_logit_relative_rmse": _relative_rmse(
            child_image_valid - child_image_valid.mean(dim=-1, keepdim=True),
            parent_image_valid - parent_image_valid.mean(dim=-1, keepdim=True),
        ),
        "top1_agreement": float(
            (child_image_valid.argmax(dim=-1) == parent_image_valid.argmax(dim=-1))
            .double()
            .mean()
            .item()
        ),
        "parent_distinct_top1_tokens": int(image_modal.unique().numel()),
        "parent_modal_top1_token_id": int(image_counts.argmax().item()),
        "parent_modal_top1_token": tokenizer.convert_ids_to_tokens(
            [int(image_counts.argmax().item())]
        )[0],
        "parent_modal_top1_floor": float(
            image_counts.max().item() / max(int(image_modal.numel()), 1)
        ),
    }

    def decode_pair(
        prompt_ids: Sequence[int],
        pixel_values: Any | None,
        pixel_attention_mask: Any | None,
    ) -> dict[str, Any]:
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
        child_tokens = greedy_continuation(
            child,
            ids,
            new_tokens=int(args.decode_new_tokens),
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )
        parent_tokens = greedy_continuation(
            parent,
            ids,
            new_tokens=int(args.decode_new_tokens),
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )
        reference_tokens = (
            None
            if reference is None
            else greedy_continuation(
                reference,
                ids,
                new_tokens=int(args.decode_new_tokens),
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
            )
        )
        matched = 0
        for left, right in zip(child_tokens, parent_tokens):
            if left != right:
                break
            matched += 1
        return {
            **(
                {}
                if reference_tokens is None
                else {
                    "full_depth_parent_continuation_text": tokenizer.decode(
                        reference_tokens
                    ),
                    "full_depth_parent_continuation_token_ids": reference_tokens,
                }
            ),
            "child_continuation_text": tokenizer.decode(child_tokens),
            "child_continuation_token_ids": child_tokens,
            "child_continuation_tokens": tokenizer.convert_ids_to_tokens(child_tokens),
            "parent_continuation_text": tokenizer.decode(parent_tokens),
            "parent_continuation_token_ids": parent_tokens,
            "parent_continuation_tokens": tokenizer.convert_ids_to_tokens(
                parent_tokens
            ),
            "child_parent_greedy_prefix_agreement_tokens": matched,
        }

    text_continuations: list[dict[str, Any]] = []
    for index, row in enumerate(text_rows):
        text_continuations.append(
            {
                "holdout_index": index,
                "lane": "text",
                "prompt_window_text": tokenizer.decode(row),
                "prompt_window_tokens": len(row),
                **decode_pair(row, None, None),
            }
        )
    image_continuations: list[dict[str, Any]] = []
    for index, (record, encoded) in enumerate(zip(holdout_probes, image_encoded)):
        image_continuations.append(
            {
                "holdout_index": index,
                "lane": "image",
                "probe_id": record["probe_id"],
                "category": record["category"],
                "probe_prompt": record["prompt"],
                "expected_answer": record["expected_answer"],
                "image_sha256": record["image_sha256"],
                "prompt_window_tokens": len(encoded["input_ids"]),
                **decode_pair(
                    encoded["input_ids"],
                    encoded["pixel_values"],
                    encoded["pixel_attention_mask"],
                ),
            }
        )

    body = {
        "schema": RUNNER_SCHEMA,
        "stage": "behavior",
        "input_sha256": input_sha256,
        "folded_layer_prefix": layers,
        "decode_policy": "greedy argmax, no sampling, no temperature, no beams",
        "decode_new_tokens": int(args.decode_new_tokens),
        "holdout_text_prompts": len(text_rows),
        "holdout_image_probes": len(holdout_probes),
        "holdout_window_tokens": int(args.holdout_window),
        "full_depth_parent_reference_decoded": reference is not None,
        "full_depth_parent_layers": full_depth,
        "text_teacher_forced": text_teacher_forced,
        "image_conditioned_teacher_forced": image_teacher_forced,
        "text_greedy_prefix_agreement_tokens": _summarize(
            [
                float(row["child_parent_greedy_prefix_agreement_tokens"])
                for row in text_continuations
            ]
        ),
        "image_greedy_prefix_agreement_tokens": _summarize(
            [
                float(row["child_parent_greedy_prefix_agreement_tokens"])
                for row in image_continuations
            ]
        ),
        "text_continuations": text_continuations,
        "image_continuations": image_continuations,
        "parent_pretrained_checkpoint_sha256": checkpoint["weight_file_sha256"],
        "seconds": time.perf_counter() - started,
    }
    receipt = _write_receipt(receipt_path, body)
    return {"receipt": receipt, "resumed": False}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_report(root: Path) -> dict[str, Any]:
    """Re-verify every receipt seal and every content digest in the manifest."""

    manifest_path = root / "manifest.json"
    manifest = _read_receipt(manifest_path)
    if manifest is None:
        raise SmolVlmFoldError(f"No sealed manifest at {manifest_path}.")
    problems: list[str] = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            problems.append(f"missing: {row['path']}")
            continue
        if _file_sha256(path) != row["sha256"]:
            problems.append(f"digest changed: {row['path']}")
    for stage in STAGES:
        path = root / f"{stage}.json"
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not verify_receipt(document):
            problems.append(f"receipt seal broken: {stage}.json")
    return {
        "root": str(root),
        "manifest_sha256": manifest["receipt_sha256"],
        "files": len(manifest["files"]),
        "problems": problems,
        "verified": not problems,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.child_seed))
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    root = args.report_root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)

    configuration = _configuration_stage(args, root)
    probe = _probe_stage(args, root, configuration["receipt"])
    fold = _fold_stage(
        args,
        root,
        configuration["receipt"],
        probe,
        configuration["calibration"],
        configuration["layers"],
        configuration["checkpoint"],
    )
    behavior = _behavior_stage(
        args,
        root,
        configuration["receipt"],
        probe,
        fold,
        configuration["holdout"],
        configuration["layers"],
        configuration["checkpoint"],
    )
    manifest_body = {
        "schema": RUNNER_SCHEMA,
        "stage": "manifest",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {
            "configuration": configuration["receipt"]["receipt_sha256"],
            "probe": probe["receipt"]["receipt_sha256"],
            "fold": fold["receipt"]["receipt_sha256"],
            "behavior": behavior["receipt"]["receipt_sha256"],
        },
        "resumed_stages": sorted(
            name
            for name, stage in (("fold", fold), ("behavior", behavior))
            if stage.get("resumed")
        ),
        "files": _tree_manifest(root, skip=("manifest.json",)),
    }
    manifest = _write_receipt(root / "manifest.json", manifest_body)
    return {
        "report_root": str(root),
        "manifest_sha256": manifest["receipt_sha256"],
        "fold_receipt_sha256": fold["receipt"]["receipt_sha256"],
        "behavior_receipt_sha256": behavior["receipt"]["receipt_sha256"],
        "verification": verify_report(root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fold a real pretrained SmolVLM2 multimodal parent across a decoder "
            "residual width contraction, carrying its vision tower and "
            "connector, then decode held-out text-only and image-conditioned "
            "continuations from the folded child."
        )
    )
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--parent-model-directory",
        type=Path,
        default=Path("C:/Users/cereal/Documents/ai training/models")
        / "SmolVLM2-256M-Video-Instruct",
        help=(
            "Directory holding the SmolVLM2-256M-Video-Instruct checkpoint.  "
            "Its declared composition is verified field by field before any "
            "compute happens."
        ),
    )
    parser.add_argument(
        "--calibration-prompts",
        type=Path,
        default=ROOT
        / "reports"
        / "kernel-stage2-20260801"
        / "calibration_prompts_64.json",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help=(
            "How many of the checkpoint's 30 decoder layers to fold.  A prefix "
            "is sealed as such in every receipt; full depth does not fit the "
            "CPU budget this runner is used under."
        ),
    )
    parser.add_argument("--child-hidden-size", type=int, default=432)
    parser.add_argument("--child-intermediate-size", type=int, default=1152)
    parser.add_argument("--child-query-heads", type=int, default=6)
    parser.add_argument("--child-key-value-heads", type=int, default=3)
    parser.add_argument("--child-seed", type=int, default=20260804)
    parser.add_argument("--calibration-prompts-count", type=int, default=36)
    parser.add_argument("--holdout-prompts-count", type=int, default=8)
    parser.add_argument("--probe-images", type=int, default=16)
    parser.add_argument("--calibration-probe-images", type=int, default=12)
    parser.add_argument("--image-seq-len", type=int, default=64)
    parser.add_argument("--evidence-window", type=int, default=96)
    parser.add_argument("--holdout-window", type=int, default=64)
    parser.add_argument("--decode-new-tokens", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-verify an existing sealed report and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            result = verify_report(args.report_root.expanduser().absolute())
        else:
            result = run(args)
    except (
        SmolVlmFoldError,
        DynamicCausalOperatorPromotionError,
        DynamicFoldExecutionError,
        CrossWidthResidualChartError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"smolvlm multimodal fold failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
