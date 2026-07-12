from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("PIL")
torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402


SCRIPT = (
    Path(__file__).resolve().parents[1] / "experiments" / "eval_neo_native_vision.py"
)
SPEC = importlib.util.spec_from_file_location("eval_neo_native_vision", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoded_hash(image: Image.Image) -> str:
    canonical = image.convert("RGB")
    payload = f"RGB:{canonical.width}x{canonical.height}\n".encode("ascii")
    return hashlib.sha256(payload + canonical.tobytes()).hexdigest()


def _build_manifest(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "probe"
    images = root / "images"
    images.mkdir(parents=True)
    definitions = (
        (
            "color",
            "What is the dominant colored shape? Answer with one word.",
            "red",
            (220, 35, 45),
        ),
        (
            "count",
            "How many blue circles are shown? Answer with one digit.",
            "2",
            (35, 90, 220),
        ),
        (
            "spatial",
            "Where is the red square relative to the blue square? Answer left, right, above, or below.",
            "left",
            (248, 248, 248),
        ),
        (
            "ocr",
            "Read the three-digit code. Answer with only the digits.",
            "012",
            (255, 255, 255),
        ),
    )
    records = []
    for category, prompt, answer, color in definitions:
        probe_id = f"{category}_0000"
        relative = f"images/{probe_id}.png"
        image = Image.new("RGB", (64, 64), color)
        image.save(root / relative, format="PNG", optimize=False)
        records.append(
            {
                "probe_id": probe_id,
                "category": category,
                "image": relative,
                "prompt": prompt,
                "answer": answer,
                "split": "heldout",
                "image_sha256": _decoded_hash(image),
            }
        )
    payload = {
        "schema": MODULE.PROBE_SCHEMA,
        "seed": 0,
        "image_format": "RGB uint8 64x64 PNG",
        "image_hash_basis": MODULE.IMAGE_HASH_BASIS,
        "examples_per_category": 1,
        "heldout_per_category": 1,
        "total": 4,
        "calibration_count": 0,
        "heldout_count": 4,
        "categories": list(MODULE.CATEGORIES),
        "records": records,
    }
    manifest = root / "manifest.json"
    manifest.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest, _sha256(manifest)


def _build_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    config = {
        "architectures": ["NEOChatModel"],
        "auto_map": {"AutoModel": "modeling_neo_chat.NEOChatModel"},
        "downsample_ratio": 0.5,
        "eos_token_id": 151645,
        "llm_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
        "min_pixels": 65536,
        "model_type": "neo_chat",
        "pad_token_id": 151643,
        "patch_size": 16,
        "template": "neo1_0",
        "vision_config": {
            "downsample_ratio": 0.5,
            "llm_hidden_size": 2048,
            "model_type": "neo_vision",
            "num_channels": 3,
            "patch_size": 16,
        },
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snapshot / "modeling_neo_chat.py").write_text(
        "# local remote code\n", encoding="ascii"
    )
    (snapshot / "model.safetensors").write_bytes(b"fake-local-weights")
    return snapshot


class FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return MODULE.SPECIAL_TOKEN_IDS[token]


class FakeNEOModel(torch.nn.Module):
    def __init__(
        self, *, fail_on_call: int | None = None, patch_size: int = 16
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.patch_size = patch_size
        self.downsample_ratio = 0.5
        self.config = SimpleNamespace(
            patch_size=patch_size,
            downsample_ratio=0.5,
            vision_config=SimpleNamespace(
                patch_size=patch_size,
                downsample_ratio=(0.5,),
            ),
        )
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        tokenizer: Any,
        *,
        pixel_values: Any,
        grid_hw: Any,
        question: str,
        generation_config: dict[str, Any],
        verbose: bool,
    ) -> str:
        call = {
            "question": question,
            "pixel_shape": tuple(pixel_values.shape),
            "pixel_dtype": pixel_values.dtype,
            "pixel_device": str(pixel_values.device),
            "grid": grid_hw.tolist(),
            "generation_config": dict(generation_config),
            "verbose": verbose,
        }
        self.calls.append(call)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("intentional interruption")
        if question.startswith("What is the dominant"):
            return '"red".'
        if question.startswith("How many"):
            return "2"
        if question.startswith("Where is"):
            return "left."
        if question.startswith("Read the"):
            return "012"
        raise AssertionError(question)


def _loader_for(model: FakeNEOModel):
    calls: list[dict[str, Any]] = []

    def loader(snapshot: Path, **kwargs: Any) -> tuple[FakeTokenizer, FakeNEOModel]:
        calls.append({"snapshot": snapshot, **kwargs})
        return FakeTokenizer(), model

    loader.calls = calls  # type: ignore[attr-defined]
    return loader


def _evaluate(
    tmp_path: Path,
    *,
    output_name: str = "eval",
    model: FakeNEOModel | None = None,
    loader: Any | None = None,
) -> tuple[dict[str, Any], FakeNEOModel, Any, Path, Path, str]:
    manifest, manifest_sha256 = _build_manifest(tmp_path)
    snapshot = _build_snapshot(tmp_path)
    selected_model = model or FakeNEOModel()
    selected_loader = loader or _loader_for(selected_model)
    output = tmp_path / output_name
    report = MODULE.evaluate_parent_baseline(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        model_snapshot=snapshot,
        output_dir=output,
        dtype="fp32",
        device="cpu",
        max_new_tokens=8,
        seed=11,
        manual_samples_per_group=1,
        model_loader=selected_loader,
        torch_module=torch,
    )
    return report, selected_model, selected_loader, manifest, snapshot, manifest_sha256


def test_decoded_rgb_hash_ignores_png_encoding_and_rejects_pixel_change(
    tmp_path: Path,
) -> None:
    manifest, digest = _build_manifest(tmp_path)
    verified = MODULE.verify_manifest(manifest, digest)
    assert verified["sha256"] == digest

    image_path = manifest.parent / "images" / "color_0000.png"
    original_encoded_hash = _sha256(image_path)
    with Image.open(image_path) as source:
        decoded = source.convert("RGB")
    decoded.save(image_path, format="PNG", optimize=True, compress_level=9)
    assert MODULE.verify_manifest(manifest, digest)["records"][0][
        "image_sha256"
    ] == _decoded_hash(decoded)
    assert _sha256(image_path) != ""  # The encoded bytes are deliberately irrelevant.
    assert original_encoded_hash != ""  # Do not make the test zlib-version-dependent.

    changed = decoded.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(image_path, format="PNG")
    with pytest.raises(
        MODULE.EvaluationContractError, match="decoded RGB SHA-256 mismatch"
    ):
        MODULE.verify_manifest(manifest, digest)


def test_manifest_rejects_old_or_ambiguous_hash_basis(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["image_hash_basis"] = "sha256(encoded PNG bytes)"
    manifest.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    with pytest.raises(MODULE.EvaluationContractError, match="image_hash_basis"):
        MODULE.verify_manifest(manifest, _sha256(manifest))

    del payload["image_hash_basis"]
    manifest.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    with pytest.raises(MODULE.EvaluationContractError, match="frozen schema"):
        MODULE.verify_manifest(manifest, _sha256(manifest))


def test_official_preprocessing_resizes_normalizes_and_flattens(tmp_path: Path) -> None:
    image_path = tmp_path / "uniform.png"
    Image.new("RGB", (64, 64), (255, 128, 0)).save(image_path)

    patches, grid_hw = MODULE.preprocess_neo_image(image_path, torch_module=torch)

    assert MODULE.smart_resize(64, 64) == (256, 256)
    assert tuple(patches.shape) == (256, 768)
    assert grid_hw.tolist() == [[16, 16]]
    expected = (
        (1.0 - 0.485) / 0.229,
        ((128.0 / 255.0) - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    )
    assert torch.allclose(patches[0, :256], torch.full((256,), expected[0]), atol=1e-6)
    assert torch.allclose(
        patches[0, 256:512], torch.full((256,), expected[1]), atol=1e-6
    )
    assert torch.allclose(patches[0, 512:], torch.full((256,), expected[2]), atol=1e-6)

    source = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
    flattened, small_grid = MODULE.preprocess_pixel_values(source, torch_module=torch)
    assert small_grid.tolist() == [[2, 2]]
    assert flattened[0, 0] == source[0, 0, 0]
    assert flattened[0, 255] == source[0, 15, 15]
    assert flattened[0, 256] == source[1, 0, 0]
    assert flattened[1, 0] == source[0, 0, 16]
    assert flattened[2, 0] == source[0, 16, 0]


@pytest.mark.parametrize(
    ("category", "raw", "expected"),
    (
        ("color", "red", "red"),
        ("color", '"RED".', "red"),
        ("color", "The answer is red", None),
        ("count", "3.", "3"),
        ("count", "three", None),
        ("spatial", "LEFT", "left"),
        ("spatial", "red is left", None),
        ("ocr", "012", "012"),
        ("ocr", "12", None),
        ("ocr", "code: 012", None),
    ),
)
def test_direct_answer_normalization_is_deliberately_strict(
    category: str,
    raw: str,
    expected: str | None,
) -> None:
    assert MODULE.normalize_direct_answer(category, raw) == expected


def test_evaluator_caches_every_output_and_complete_cache_skips_loader(
    tmp_path: Path,
) -> None:
    report, model, loader, manifest, snapshot, digest = _evaluate(tmp_path)

    assert len(loader.calls) == 1
    assert len(model.calls) == 4
    assert report["metrics"]["overall"] == {
        "accuracy": 1.0,
        "correct": 4,
        "direct_answer_rate": 1.0,
        "direct_answers": 4,
        "total": 4,
    }
    assert report["cache"]["cache_hits_this_invocation"] == 0
    assert report["cache"]["new_records_this_invocation"] == 4
    assert (
        len((tmp_path / "eval" / "parent_outputs.jsonl").read_text().splitlines()) == 4
    )
    for call in model.calls:
        assert call["pixel_shape"] == (256, 768)
        assert call["pixel_dtype"] == torch.float32
        assert call["pixel_device"] == "cpu"
        assert call["grid"] == [[16, 16]]
        assert call["generation_config"] == {
            "do_sample": False,
            "max_new_tokens": 8,
            "num_beams": 1,
        }
        assert call["verbose"] is False

    def forbidden_loader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("complete parent cache must not load or call the model")

    cached_report = MODULE.evaluate_parent_baseline(
        manifest=manifest,
        manifest_sha256=digest,
        model_snapshot=snapshot,
        output_dir=tmp_path / "eval",
        dtype="fp32",
        device="cpu",
        max_new_tokens=8,
        seed=11,
        manual_samples_per_group=1,
        model_loader=forbidden_loader,
        torch_module=torch,
    )
    assert cached_report["cache"]["cache_hits_this_invocation"] == 4
    assert cached_report["cache"]["new_records_this_invocation"] == 0


def test_interrupted_run_resumes_only_missing_manifest_suffix(tmp_path: Path) -> None:
    manifest, digest = _build_manifest(tmp_path)
    snapshot = _build_snapshot(tmp_path)
    output = tmp_path / "interrupted"
    first_model = FakeNEOModel(fail_on_call=3)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        MODULE.evaluate_parent_baseline(
            manifest=manifest,
            manifest_sha256=digest,
            model_snapshot=snapshot,
            output_dir=output,
            dtype="fp32",
            device="cpu",
            max_new_tokens=8,
            seed=11,
            model_loader=_loader_for(first_model),
            torch_module=torch,
        )
    assert len(first_model.calls) == 3
    assert len((output / "parent_outputs.jsonl").read_text().splitlines()) == 2
    failure = json.loads((output / "failure.json").read_text())
    assert failure["completed"] == 2

    resumed_model = FakeNEOModel()
    report = MODULE.evaluate_parent_baseline(
        manifest=manifest,
        manifest_sha256=digest,
        model_snapshot=snapshot,
        output_dir=output,
        dtype="fp32",
        device="cpu",
        max_new_tokens=8,
        seed=11,
        model_loader=_loader_for(resumed_model),
        torch_module=torch,
    )
    assert len(resumed_model.calls) == 2
    assert resumed_model.calls[0]["question"].startswith("Where is")
    assert resumed_model.calls[1]["question"].startswith("Read the")
    assert report["cache"]["cache_hits_this_invocation"] == 2
    assert report["cache"]["new_records_this_invocation"] == 2
    assert not (output / "failure.json").exists()


def test_cache_tampering_fails_before_model_loader(tmp_path: Path) -> None:
    _, _, _, manifest, snapshot, digest = _evaluate(tmp_path)
    cache_path = tmp_path / "eval" / "parent_outputs.jsonl"
    records = [json.loads(line) for line in cache_path.read_text().splitlines()]
    records[0]["raw_output"] = "blue"
    cache_path.write_text("\n".join(json.dumps(item) for item in records) + "\n")

    def forbidden_loader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid cache must fail before model loading")

    with pytest.raises(
        MODULE.EvaluationContractError, match="raw output hash mismatch"
    ):
        MODULE.evaluate_parent_baseline(
            manifest=manifest,
            manifest_sha256=digest,
            model_snapshot=snapshot,
            output_dir=tmp_path / "eval",
            dtype="fp32",
            device="cpu",
            max_new_tokens=8,
            seed=11,
            model_loader=forbidden_loader,
            torch_module=torch,
        )


def test_model_and_tokenizer_loading_is_strictly_local(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tokenizer_result = object()

    class AutoTokenizer:
        calls: list[tuple[str, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            cls.calls.append((path, kwargs))
            return tokenizer_result

    class LoadedModel:
        def __init__(self) -> None:
            self.to_calls: list[Any] = []
            self.eval_calls = 0

        def to(self, device: Any) -> "LoadedModel":
            self.to_calls.append(device)
            return self

        def eval(self) -> "LoadedModel":
            self.eval_calls += 1
            return self

    loaded_model = LoadedModel()

    class AutoModel:
        calls: list[tuple[str, dict[str, Any]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> LoadedModel:
            cls.calls.append((path, kwargs))
            return loaded_model

    tokenizer, model = MODULE.load_neo_model(
        snapshot,
        torch_module=torch,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModel,
    )

    assert tokenizer is tokenizer_result
    assert model is loaded_model
    assert AutoTokenizer.calls[0][1] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "use_fast": False,
    }
    assert AutoModel.calls[0][1] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32,
    }
    assert loaded_model.to_calls == [torch.device("cpu")]
    assert loaded_model.eval_calls == 1


def test_force_neo_eager_prefill_attention_disables_only_flex_mask(
    monkeypatch: Any,
) -> None:
    module_name = "fake_neo_qwen3_runtime"
    fake_module = SimpleNamespace(
        create_flex_mask_qwen3_style=lambda *args, **kwargs: ("compiled-mask", 17)
    )
    monkeypatch.setitem(MODULE.sys.modules, module_name, fake_module)

    language_model_cls = type("FakeLanguageModel", (), {"__module__": module_name})
    language_model = language_model_cls()
    language_model.config = SimpleNamespace(_attn_implementation="sdpa")
    model = SimpleNamespace(language_model=language_model)

    MODULE._force_neo_eager_prefill_attention(model)

    assert language_model.config._attn_implementation == "eager"
    assert fake_module.create_flex_mask_qwen3_style(object(), do_padding=True) == (
        None,
        0,
    )
    assert model._neural_forge_attention_runtime == MODULE.ATTENTION_RUNTIME


def test_loaded_model_preprocess_mismatch_fails_before_chat(tmp_path: Path) -> None:
    manifest, digest = _build_manifest(tmp_path)
    snapshot = _build_snapshot(tmp_path)
    model = FakeNEOModel(patch_size=14)
    with pytest.raises(MODULE.EvaluationContractError, match="patch_size"):
        MODULE.evaluate_parent_baseline(
            manifest=manifest,
            manifest_sha256=digest,
            model_snapshot=snapshot,
            output_dir=tmp_path / "mismatch",
            dtype="fp32",
            device="cpu",
            model_loader=_loader_for(model),
            torch_module=torch,
        )
    assert model.calls == []
    assert not (tmp_path / "mismatch" / "parent_outputs.jsonl").exists()
