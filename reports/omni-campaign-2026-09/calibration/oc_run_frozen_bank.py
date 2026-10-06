#!/usr/bin/env python3
"""Run a review bank on one omni parent (or child) and seal the review surface.

This is the omni analogue of `experiments/run_lfm2_composition_frozen_review.py`,
and the parts that matter are the parts that did NOT move:

  * the same 15-case frozen pack, digest-pinned to 0ee6bf14...,
  * the same chat template path with `add_generation_prompt=True`,
  * the same greedy argmax decode with `use_cache`,
  * the same 1,024-token cap,
  * the same three termination states, and a cap hit is still never a failure,
  * the same "grades nothing" rule -- this writes verbatim text, and a human
    reads it.

FOUR THINGS DID MOVE, EACH BECAUSE THE LFM2 RUNNER HARDCODED SOMETHING THAT ONLY
EXISTS ON LFM2.  R3's port note names all four and this file pays them:

 (1) **The loader class is not hardcoded.**  The LFM2 runner imports
     `Lfm2ForCausalLM` by name.  Here the class is read off the checkpoint's own
     `config.json["architectures"][0]` and resolved against stock `transformers`
     with `trust_remote_code=False`.  A model whose architecture string is not a
     stock class is REFUSED rather than loaded through remote code, because a
     review whose loader is arbitrary code is not a review of a standard
     checkpoint.  `--loader-class` can name it explicitly and is checked against
     the config, so the two can never silently disagree.

 (2) **Identity is a directory digest, not `model.safetensors`.**  Every omni
     parent is sharded, and `Qwen3.5-4B` is sharded under names
     (`model.safetensors-00001-of-00002.safetensors`) that no single-file check
     would even find.  `oc_calib.directory_digest` digests the sorted
     (relative path, sha256) pairs of the weight, config and tokenizer files.

 (3) **The review dtype is bf16, not float32.**  float32 at 27B does not fit and
     a calibration measured in one dtype cannot be compared with a review taken
     in another, so the dtype is a LABELLED part of the surface: it goes in the
     dossier, in the surface tuple, and into the parent-lane digest's validity
     conditions.  `--dtype` accepts float16 for sm75 (the RTX 8000 lane) and the
     dossier says which one ran.

 (4) **Termination accounting knows about `<|im_end|>`.**  A Qwen chat model ends
     a turn on a token that is not `eos_token_id`.  A runner that watched only
     `eos_token_id` would record every well-formed answer as `censored-at-cap`,
     which is not a grading error -- the protocol says a cap hit is never a
     failure -- but it IS an accounting error, and the artifact-tolerant lens
     reads terminations.  The stop ids used are written into the dossier.

AND ONE THING IS ADDED: A DECODE-PATH AGREEMENT PREFLIGHT.  The manual argmax
loop is the protocol's decode path and it is what runs.  But Qwen3.5 is an mrope
model, and an mrope model computes position ids inside `forward` in a way tuned
for `generate`, so "the manual loop is equivalent to greedy generate" is an
ASSUMPTION on this architecture where it was a fact on LFM2.  The preflight
decodes eight tokens both ways on the first case and writes the agreement into
the dossier.  It does not switch paths on disagreement and it does not fail the
run: it makes the assumption VISIBLE, so a reader is never grading text produced
by a decode path nobody checked.

USAGE (server, GPU0)

  $PY oc_run_frozen_bank.py \
      --model-dir /home/cereal/models/Qwen3.5-4B \
      --label Q35-4B --stage parent \
      --bank <repo>/research/lfm2_semantic_whole_response_prompts_v1.json \
      --bank-sha256 0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b \
      --device cuda:0 --require-compute-architecture sm86 --dtype bfloat16 \
      --out oc-dossier-Q35-4B.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oc_calib import (  # noqa: E402
    CalibrationError,
    canonical_sha256,
    decode_smoke,
    directory_digest,
    emit_zero_floor,
    file_sha256,
    generate_greedy,
    kernel_path_report,
    load_bank,
    log,
    messages_for,
    prompt_surface_sha256,
    render_prompt,
    thinking_report,
    resolve_dtype,
    write_canonical,
    compute_architecture,
)

DOSSIER_SCHEMA = "neural-forge-omni-parent-calibration-dossier-v1"

#: The frozen V5 pack.  Named here so a caller who forgets `--bank-sha256` on the
#: sealed bank is refused rather than silently reading an edited pack.
FROZEN_V1_PACK_SHA256 = (
    "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"
)
FROZEN_V1_PROTOCOL_SHA256 = (
    "e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b"
)

#: `experiments/matrix_campaign_{probe,smoke}_prompts_v1.json` at repo HEAD.
PROBE_PROMPTS_SHA256 = (
    "6230f878be9287a9abf4ad129d586fc22346867897858b9f1a4f0b75d1f82f2e"
)
SMOKE_PROMPTS_SHA256 = (
    "82ebf70a1606ec8a57c4daa31218fe1d376a91f211246c78716615ca66d2ceca"
)


def materialise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand `<<HAYSTACK>>` and refuse any row this runner cannot decode.

    TWO THINGS, BOTH REFUSALS RATHER THAN QUIET BEHAVIOUR.

    A long-context case carries a `haystack_spec` instead of a megabyte of
    filler.  The document is regenerated here and its digest is checked against
    the one the bank pinned, so the text decoded is provably the text the bank
    describes.  A mismatch means the generator or the spec moved and the case is
    refused, not silently run against different filler.

    A vision or audio-video case carries image/video parts and needs a processor,
    pixel batching and an asset loader that this text runner does not have.  It is
    REFUSED by name.  The gap is real and saying so beats decoding the text half
    of a vision prompt and filing the result as a vision result.

    A `runnable: false` case is refused for the mirror-image reason: its prompt is
    ordinary text but its ANSWER is a clip, an image or a track, so a text decoder
    produces prose where the case asked for an artifact.  `--text-only` drops
    those by name before they get here; a `--lanes` list that names one directly
    is refused rather than quietly boxed as a text lane.
    """

    from oc_bank_v2_draft import materialise_haystack

    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if row.get("runnable", True) is False:
            raise CalibrationError(
                f"case {row['id']} is marked runnable:false: it asks for a media "
                f"artifact (modality_out={row.get('modality_out')!r}) and this "
                "runner emits text.  Drop it with --text-only, which names it in "
                "the dossier, rather than filing prose as a generated clip."
            )
        for message in row.get("messages", []) or []:
            if not isinstance(message.get("content"), str):
                raise CalibrationError(
                    f"case {row['id']} carries non-text content parts "
                    f"(modality_in={row.get('modality_in')!r}).  This runner "
                    "decodes text only; the vision and audio-video splits need a "
                    "processor and an asset loader that are NOT built.  Select "
                    "the text lanes with --lanes."
                )
        spec = row.get("haystack_spec")
        if spec:
            document = materialise_haystack(spec)
            observed = canonical_sha256({"text": document})
            if observed != str(spec["haystack_sha256"]):
                raise CalibrationError(
                    f"case {row['id']}: the regenerated haystack digests "
                    f"{observed}, the bank pinned {spec['haystack_sha256']}."
                )
            row["messages"] = [
                {**m, "content": str(m["content"]).replace("<<HAYSTACK>>", document)}
                for m in row["messages"]
            ]
            row["haystack_sha256_verified"] = observed
        out.append(row)
    return out


def select_cases(
    rows: list[dict[str, Any]],
    *,
    lanes: str | None,
    text_only: bool,
    max_cases: int | None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Which cases this run decodes, and the NAME of every case it does not.

    TWO REASONS TO DROP A CASE UNDER `--text-only`, AND THEY ARE NOT THE SAME
    REASON.  `modality_in` catches a case whose PROMPT this runner cannot build
    (a vision or audio-video case needs a processor and an asset loader that are
    not written).  `runnable: false` catches a case whose ANSWER is a media
    artifact: the three media-out cases in bank v2 carry a plain string prompt
    and the default `modality_in: text`, so a modality filter alone KEPT them --
    a text LLM was asked to "generate a four-second clip of a blue kettle", its
    prose would have been boxed as a text lane, and the scored denominator would
    have been three lanes too large.  Both lists go into the dossier under their
    own keys, because "we could not build the prompt" and "we cannot produce the
    answer's medium" are different admissions.

    Kept separate from `run` so it can be tested with no torch and no GPU.
    """

    if lanes:
        wanted = set(str(lanes).split(","))
        missing = wanted - {str(row["id"]) for row in rows}
        if missing:
            raise CalibrationError(f"--lanes names unknown cases: {sorted(missing)}")
        rows = [row for row in rows if str(row["id"]) in wanted]
    skipped_modality: list[str] = []
    skipped_not_runnable: list[str] = []
    if text_only:
        keep: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("modality_in", "text")) != "text":
                skipped_modality.append(str(row["id"]))
            elif row.get("runnable", True) is False:
                skipped_not_runnable.append(str(row["id"]))
            else:
                keep.append(row)
        rows = keep
    if max_cases:
        rows = rows[: int(max_cases)]
    #: AN EMPTY SELECTION IS A REFUSAL, NOT A ZERO-CASE RUN.  `--lanes` names
    #: only vision cases, or only media-out cases, and `--text-only` then drops
    #: every one of them: a legal pair of flags with nothing left to decode.  The
    #: run used to reach `messages_for(rows[0])` and raise `IndexError: list
    #: index out of range`, which names neither flag and reads like a bug in the
    #: runner.  It is refused here, where both reasons are still in hand.
    if not rows:
        raise CalibrationError(
            "the case selection is empty: nothing would be decoded and the "
            "dossier would seal a review of no cases.  "
            f"--lanes={lanes!r}, --text-only={text_only}, "
            f"--max-cases={max_cases!r} dropped "
            f"{len(skipped_modality)} case(s) for an unsupported input modality "
            f"({skipped_modality}) and {len(skipped_not_runnable)} case(s) whose "
            f"ANSWER is a media artifact ({skipped_not_runnable})."
        )
    return rows, {
        "skipped_unsupported_modality": skipped_modality,
        "skipped_not_runnable": skipped_not_runnable,
    }


def resolve_loader_class(config: dict[str, Any], requested: str | None) -> Any:
    """Stock `transformers` class named by the checkpoint, or a refusal."""

    import transformers

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise CalibrationError(
            "the checkpoint's config.json declares no `architectures`, so there "
            "is no class to load it with and no claim to seal."
        )
    declared = str(architectures[0])
    if requested is not None and requested != declared:
        raise CalibrationError(
            f"--loader-class {requested!r} disagrees with the checkpoint's own "
            f"architectures[0] {declared!r}."
        )
    loader = getattr(transformers, declared, None)
    if loader is None:
        raise CalibrationError(
            f"stock transformers {transformers.__version__} has no class "
            f"{declared!r}.  A review may not load a checkpoint through remote "
            "code: trust_remote_code stays literally false."
        )
    return loader


PLACEMENT_STRATEGIES = ("auto", "balanced", "balanced_low_0", "sequential")


def available_host_memory_bytes() -> int:
    """Available physical RAM, without treating total RAM as available."""
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_pagefile",
                    "available_pagefile", "total_virtual", "available_virtual", "extended")
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise CalibrationError("could not measure available host RAM")
        return int(status.available_physical)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError) as exc:
        raise CalibrationError("could not measure available host RAM") from exc


def make_load_plan(
    device_map: str | None, max_memory: list[str] | None, *, device: str,
    tiny_random: bool = False,
) -> dict[str, Any] | None:
    """Validate explicit budgets before tokenizer or checkpoint loading.

    CUDA indices are relative to CUDA_VISIBLE_DEVICES. Every visible GPU needs
    a limit, including 0MiB for a GPU which must not be used. CPU is mandatory;
    cpu=0GiB prevents automatic spill. Limits reserve nothing and describe
    weight placement, not the KV cache or activation footprint.
    """
    if device_map is None:
        if max_memory is not None:
            raise CalibrationError("--max-memory requires --device-map")
        return None
    if device_map not in PLACEMENT_STRATEGIES or not max_memory or tiny_random:
        raise CalibrationError("explicit stock --device-map requires budgets and a real checkpoint")
    limits: dict[int | str, int] = {}
    for entry in max_memory:
        match = re.fullmatch(r"(cpu|0|[1-9][0-9]*)=(0|[1-9][0-9]*)(MiB|GiB)", entry)
        if match is None:
            raise CalibrationError(f"invalid --max-memory {entry!r}; use DEVICE=integerMiB/GiB")
        key: int | str = "cpu" if match[1] == "cpu" else int(match[1])
        if key in limits:
            raise CalibrationError(f"duplicate --max-memory device {key!r}")
        limits[key] = int(match[2]) * (1 << (20 if match[3] == "MiB" else 30))
    import torch
    visible = set(range(torch.cuda.device_count()))
    if set(limits) != visible | {"cpu"}:
        raise CalibrationError("--max-memory must name cpu and every visible CUDA index exactly once")
    target = "cpu" if device == "cpu" else (
        int(device.split(":")[1]) if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) else None
    )
    if target not in limits or limits[target] == 0:
        raise CalibrationError("--device must name a device with a positive explicit memory limit")
    available: dict[int | str, int] = {"cpu": available_host_memory_bytes()}
    for index in sorted(visible):
        if limits[index]:  # Do not initialize a context on excluded/busy GPUs.
            available[index] = int(torch.cuda.mem_get_info(index)[0])
    for key, measured in available.items():
        if limits[key] > measured:
            raise CalibrationError(f"--max-memory {key} budget {limits[key]} exceeds currently free {measured} bytes")
    return {"device_map": device_map, "max_memory": limits}


def mapped_placement_report(model: Any, limits: dict[int | str, int]) -> dict[str, Any]:
    """Report actual stock placement; refuse disk or incomplete meta weights."""
    def canonical(value: Any) -> str:
        if isinstance(value, bool):
            raise CalibrationError("boolean is not a device")
        value = f"cuda:{value}" if isinstance(value, int) else str(value)
        key: int | str = "cpu" if value == "cpu" else (
            int(value.split(":")[1]) if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) else "invalid"
        )
        if key not in limits or limits[key] <= 0:
            raise CalibrationError(f"loaded placement {value!r} is outside positive explicit budgets")
        return value
    actual = getattr(model, "hf_device_map", None)
    map_source = "hf_device_map"
    if not isinstance(actual, dict) or not actual:
        # Transformers 5.12.1 does not retain hf_device_map for a CPU-only
        # from_pretrained result. Resident parameter devices are direct
        # measurements; do not reconstruct an intended/inferred map instead.
        actual = {name: parameter.device for name, parameter in model.named_parameters()} if hasattr(model, "named_parameters") else {}
        map_source = "resident parameter devices"
        if not actual:
            raise CalibrationError("stock loader returned no measurable actual placement")
    placement = {str(name): canonical(target) for name, target in actual.items()}
    # Meta parameters in mixed CPU/GPU offload require a different execution
    # surface. This option deliberately supports resident weights only.
    dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        canonical(parameter.device)
        if parameter.is_floating_point():
            name = str(parameter.dtype)
            dtypes[name] = dtypes.get(name, 0) + int(parameter.numel())
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise CalibrationError("cannot discover the loaded input embedding device")
    input_device = canonical(embeddings.weight.device)
    return {"device_map": placement, "device_map_source": map_source, "input_device": input_device,
            "floating_parameter_dtypes": dtypes}


def load_model(
    model_dir: Path,
    *,
    dtype: Any,
    device: str,
    requested_class: str | None,
    load_plan: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    loader = resolve_loader_class(config, requested_class)
    log(f"loading {model_dir} as {loader.__name__} in {dtype}")
    started = time.monotonic()
    options = {"dtype": dtype, "trust_remote_code": False}
    if load_plan is not None:
        options.update(load_plan)
    model = loader.from_pretrained(str(model_dir), **options)
    placement = {}
    if load_plan is None:
        model = model.to(device)
    else:
        placement = mapped_placement_report(model, load_plan["max_memory"])
        placement["device_map_strategy"] = load_plan["device_map"]
        placement["max_memory_bytes"] = {str(k): v for k, v in load_plan["max_memory"].items()}
        placement["requested_dtype"] = str(dtype)
    model = model.eval()
    live = int(sum(p.numel() for p in model.parameters()))
    import transformers

    export = {
        "loader": "stock transformers",
        "loaded_class": type(model).__name__,
        "declared_class": str(config["architectures"][0]),
        "trust_remote_code": False,
        "parameters_counted_live": live,
        "transformers_version": transformers.__version__,
        #: PROBED OFF THIS MODEL, not off the package inventory: the four
        #: Gated-DeltaNet primitives fall back independently, so only the
        #: loaded module knows which of them is Triton and which is torch.
        "linear_attention_kernels": kernel_path_report(model),
        "load_wall_time_seconds": round(time.monotonic() - started, 3),
        **placement,
    }
    if export["loaded_class"] != export["declared_class"]:
        raise CalibrationError(
            f"{model_dir} loaded as {export['loaded_class']} but declares "
            f"{export['declared_class']}."
        )
    log(f"loaded: {live:,} parameters in {export['load_wall_time_seconds']:.1f}s")
    return model, export


def build_tiny_random(
    reference_dir: Path, *, dtype: Any, device: str, seed: int = 20260901
) -> tuple[Any, dict[str, Any]]:
    """A shrunken, randomly initialised model of the SAME class.

    THIS IS A MECHANICS BED AND IT IS LABELLED AS ONE EVERYWHERE.  It exists so
    the runner, the floors, the classifier and the assessment can be exercised
    end to end when the real parent has not landed, and every artifact it
    produces carries `is_random_init: true`.  Nothing measured on it is a
    calibration of anything, and `oc_assess.py` refuses to write a scoring sheet
    for Cyril from a random-init dossier.
    """

    import torch
    import transformers

    torch.manual_seed(seed)
    config = json.loads((Path(reference_dir) / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    text.update(
        {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
        }
    )
    for key in ("layer_types", "full_attention_interval", "linear_num_key_heads",
                "linear_num_value_heads", "linear_key_head_dim",
                "linear_value_head_dim", "decoder_sparse_step", "mlp_only_layers"):
        text.pop(key, None)
    if "vision_config" in config:
        config["vision_config"].update(
            {"depth": 2, "hidden_size": 64, "intermediate_size": 128,
             "num_heads": 4, "out_hidden_size": 128}
        )
    declared = str(config["architectures"][0])
    config_class = getattr(transformers, declared).config_class
    tiny = config_class.from_dict(config)
    model = getattr(transformers, declared)(tiny).to(dtype).to(device).eval()
    live = int(sum(p.numel() for p in model.parameters()))
    log(f"RANDOM-INIT mechanics model: {declared}, {live:,} parameters")
    return model, {
        "loader": "stock transformers, randomly initialised",
        "loaded_class": type(model).__name__,
        "declared_class": declared,
        "trust_remote_code": False,
        "parameters_counted_live": live,
        "transformers_version": transformers.__version__,
        "linear_attention_kernels": kernel_path_report(model),
        "is_random_init": True,
        "seed": seed,
        "warning": (
            "RANDOM INIT.  This model has never been trained.  Nothing measured "
            "on it is a calibration; it exercises the runner and the floors."
        ),
    }


def decode_path_agreement(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    device: str,
    steps: int = 8,
    stop_ids: set[int] | None = None,
    thinking: str = "template-default",
) -> dict[str, Any]:
    """Eight tokens, manual loop versus greedy `generate`.  Published, not acted on."""

    import torch

    text = render_prompt(tokenizer, messages_for(row), thinking=thinking)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    from oc_calib import logits_to_keep_kwargs

    keep = logits_to_keep_kwargs(model)
    manual: list[int] = []
    with torch.no_grad():
        past = None
        current = ids
        result = None
        for _ in range(steps):
            result = model(
                input_ids=current, past_key_values=past, use_cache=True, **keep
            )
            past = result.past_key_values
            next_id = int(result.logits[0, -1].argmax().item())
            manual.append(next_id)
            if stop_ids and next_id in stop_ids:
                break
            current = torch.tensor([[next_id]], device=device, dtype=ids.dtype)
        #: THE MANUAL LOOP'S CACHE MUST DIE BEFORE `generate` PREFILLS AGAIN.
        #: This diagnostic is the most expensive thing in the run -- two full
        #: prefills of the longest prompt -- and holding the first one's cache
        #: across the second is what made it, rather than the review, the thing
        #: that ran out of memory on the 64k rung.  `past` alone is not enough:
        #: `result` references the same Cache object.
        result = None
        past = None
        current = None
        torch.cuda.empty_cache()
        produced = model.generate(
            input_ids=ids,
            max_new_tokens=steps,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    reference = [int(x) for x in produced[0][ids.shape[-1] :].tolist()]
    common = min(len(manual), len(reference))
    agree = manual[:common] == reference[:common]
    return {
        "case_id": str(row["id"]),
        "steps": steps,
        "manual_loop_ids": manual,
        "greedy_generate_ids": reference,
        "agree": bool(agree),
        "logits_to_keep_available": bool(keep),
        "decode_path_used": "manual argmax loop (the frozen protocol's path)",
        "why_this_is_published": (
            "on LFM2 the manual loop and greedy generate are the same "
            "computation; on an mrope model the position ids are computed inside "
            "forward in a way tuned for generate, so the equality is an "
            "assumption here.  Disagreement does NOT switch paths and does NOT "
            "fail the run -- it is recorded so no reader grades text produced by "
            "an unchecked decode path."
        ),
    }


def stop_token_ids(tokenizer: Any, extra_tokens: list[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    names: list[str] = []
    for token in extra_tokens:
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None and value >= 0 and value != getattr(
            tokenizer, "unk_token_id", -1
        ):
            ids.append(int(value))
            names.append(token)
    return ids, names


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    torch.set_grad_enabled(False)
    started = time.monotonic()
    load_plan = make_load_plan(
        getattr(args, "device_map", None), getattr(args, "max_memory", None),
        device=str(args.device), tiny_random=bool(args.tiny_random),
    )

    architecture = compute_architecture(str(args.device))
    if args.require_compute_architecture and not architecture.startswith(
        str(args.require_compute_architecture)
    ):
        raise CalibrationError(
            f"--device {args.device} resolves to {architecture!r}, not "
            f"{args.require_compute_architecture!r}.  A review measured on one "
            "compute architecture is not comparable with one measured on "
            "another, and this estate has already been bitten by CUDA's device "
            "ordering once."
        )
    log(f"device {args.device} -> {architecture}")

    bank_path = Path(args.bank)
    expected = args.bank_sha256
    if expected is None and bank_path.name.startswith("lfm2_semantic_whole_response"):
        expected = FROZEN_V1_PACK_SHA256
    rows, bank_payload, bank_digest = load_bank(bank_path, expected_file_sha256=expected)
    rows, selection = select_cases(
        rows,
        lanes=args.lanes,
        text_only=bool(args.text_only),
        max_cases=args.max_cases,
    )
    skipped = selection["skipped_unsupported_modality"]
    skipped_not_runnable = selection["skipped_not_runnable"]
    if skipped:
        log(f"--text-only dropped {len(skipped)} non-text case(s): {skipped}")
    if skipped_not_runnable:
        log(
            f"--text-only dropped {len(skipped_not_runnable)} case(s) marked "
            f"runnable:false: {skipped_not_runnable}"
        )
    rows = materialise_rows(rows)
    surface = prompt_surface_sha256(rows)
    log(f"bank {bank_path.name}: {len(rows)} case(s), file {bank_digest[:16]}")

    dtype = resolve_dtype(str(args.dtype))
    model_dir = Path(args.model_dir)
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else model_dir
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=False
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    extra_ids, extra_names = stop_token_ids(
        tokenizer, [t for t in str(args.extra_eos_tokens).split(",") if t]
    )

    if args.tiny_random:
        model, export = build_tiny_random(model_dir, dtype=dtype, device=str(args.device))
        provenance = {
            "model_directory": str(model_dir),
            "directory_sha256": None,
            "files": [],
            "note": "config and tokenizer only; the weights are random",
        }
    else:
        model, export = load_model(
            model_dir,
            dtype=dtype,
            device=str(args.device),
            requested_class=args.loader_class,
            load_plan=load_plan,
        )
        digest, files = directory_digest(model_dir)
        provenance = {
            "model_directory": str(model_dir),
            "directory_sha256": digest,
            "files": files,
            "file_count": len(files),
        }
        log(f"directory digest {digest[:16]} over {len(files)} file(s)")

    decode_device = str(export.get("input_device", args.device))
    if load_plan is not None:
        architectures = {target: compute_architecture(target)
                         for target in sorted(set(export["device_map"].values()))}
        export["device_compute_architectures"] = architectures
        architecture = compute_architecture(decode_device)
        if args.require_compute_architecture and any(
            not value.startswith(str(args.require_compute_architecture))
            for value in architectures.values()
        ):
            raise CalibrationError("actual mapped devices disagree with --require-compute-architecture")
        log(f"mapped input embeddings -> {decode_device}; devices {architectures}")
    stops = set(extra_ids) | ({int(eos_id)} if eos_id is not None else set())
    #: Answer the thinking question ONCE, on a real case, before any decoding,
    #: so the dossier records what the template actually did rather than what
    #: the flag asked for.
    thinking = thinking_report(tokenizer, messages_for(rows[0]), str(args.thinking))
    log(
        f"chat template thinking: requested={thinking['requested']} "
        f"took_effect={thinking['took_effect']} "
        f"template_opens_think_block={thinking['template_opens_think_block']}"
    )
    if thinking["template_opens_think_block"] and str(args.thinking) == "template-default":
        log(
            "NOTE: this checkpoint's template opens a think block, so the "
            f"{args.maximum_new_tokens}-token budget is shared with chain-of-"
            "thought and lanes may terminate censored-at-cap before answering.  "
            "That is a SURFACE fact, recorded, not a failure and not corrected "
            "here; --thinking off is the other surface."
        )
    #: THE PREFLIGHT IS PUBLISHED, NOT ACTED ON -- so it must not be able to
    #: end the run, and until now it could.  It decodes eight tokens twice over
    #: `rows[0]`: once through the manual loop and once through `generate`, which
    #: is TWO full prefills of the first case.  On the bank-v2 64k rung the first
    #: case is ~82,000 tokens and that pair of prefills raised
    #: torch.OutOfMemoryError before a single review token had been produced, so
    #: the whole rung died with a traceback and no dossier -- a diagnostic taking
    #: the measurement down with it.  It is now caught and RECORDED as
    #: `agree: null` with the reason, and the review proceeds on the decode path
    #: the protocol names.  A reader sees "the check could not be run here"
    #: instead of nothing at all.
    try:
        preflight = decode_path_agreement(
            model,
            tokenizer,
            rows[0],
            device=decode_device,
            stop_ids=stops,
            thinking=str(args.thinking),
        )
    except RuntimeError as exc:
        import torch

        preflight = {
            "case_id": str(rows[0]["id"]),
            "agree": None,
            "not_run": f"{type(exc).__name__}: {exc}",
            "why_not_run": (
                "the preflight decodes the first case TWICE (manual loop and "
                "greedy generate) and so costs two prefills of the longest "
                "prompt in the run.  It could not complete here.  It is a "
                "published diagnostic and never gated anything, so the review "
                "continues on the protocol's manual argmax path; a reader "
                "should treat the decode-path equality as UNCHECKED for this "
                "dossier."
            ),
        }
        torch.cuda.empty_cache()
    log(f"decode-path preflight: agree={preflight['agree']}")

    probe_payload = json.loads(Path(args.probe_prompts).read_text(encoding="utf-8"))
    probe_digest = file_sha256(Path(args.probe_prompts))
    floor = emit_zero_floor(
        model, tokenizer, probe_payload["prompts"], device=decode_device
    )
    log(
        f"emit-zero floor: passed={floor['passed']} "
        f"min|h|={floor['min_terminal_l2']:.4f} "
        f"min logit sd={floor['min_logit_std']:.4f}"
    )

    responses = generate_greedy(
        model,
        tokenizer,
        rows,
        cap=int(args.maximum_new_tokens),
        device=decode_device,
        eos_id=eos_id,
        label=str(args.label),
        extra_eos_ids=extra_ids,
        use_logits_to_keep=not bool(getattr(args, "no_logits_to_keep", False)),
        thinking=str(args.thinking),
    )

    smoke_records: list[dict[str, Any]] = []
    smoke_digest = None
    if not args.skip_smoke:
        smoke_payload = json.loads(Path(args.smoke_prompts).read_text(encoding="utf-8"))
        smoke_digest = file_sha256(Path(args.smoke_prompts))
        smoke_records = decode_smoke(
            model,
            tokenizer,
            smoke_payload["prompts"],
            cap=int(args.smoke_maximum_new_tokens),
            device=decode_device,
            eos_id=eos_id,
            extra_eos_ids=extra_ids,
            thinking=str(args.thinking),
        )

    body: dict[str, Any] = {
        "schema": DOSSIER_SCHEMA,
        "purpose": (
            "The verbatim review surface for one omni model under a named review "
            "bank.  GRADING HAPPENS ELSEWHERE; this file grades nothing.  Only "
            "measured intelligence read by a human grades, and every machine "
            "column downstream of this file is a diagnostic."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": str(args.label),
        "parent_code": str(args.parent_code or args.label),
        "stage": str(args.stage),
        "campaign": "omni-2026-09",
        "is_random_init": bool(export.get("is_random_init", False)),
        "provenance": provenance,
        "export": export,
        "bank": {
            "path": str(bank_path),
            "file_sha256": bank_digest,
            "declared_schema": bank_payload.get("schema"),
            "declared_status": bank_payload.get("status"),
            "digest_was_pinned": expected is not None,
            "cases_run": len(rows),
            "cases_in_bank": len(bank_payload.get("prompts", [])),
            "cases_skipped_unsupported_modality": skipped,
            "cases_skipped_not_runnable": skipped_not_runnable,
            "dossier_prompt_surface_sha256": surface,
            "dossier_prompt_surface_note": (
                "this runner's own digest over the validated rows.  It is NOT "
                "the V5 protocol's published normalized surface 5615b9e4..., "
                "which is computed by machinery this file does not reproduce."
            ),
            "frozen_v1_protocol_sha256": FROZEN_V1_PROTOCOL_SHA256,
            "frozen_v1_pack_sha256": FROZEN_V1_PACK_SHA256,
        },
        "generation": {
            "maximum_new_tokens": int(args.maximum_new_tokens),
            "decoding": "greedy, argmax, no sampling",
            "decode_path": "manual argmax loop with use_cache",
            "device": decode_device,
            "compute_architecture": architecture,
            "dtype": str(args.dtype),
            "eos_token_id": None if eos_id is None else int(eos_id),
            "extra_stop_token_ids": extra_ids,
            "extra_stop_tokens": extra_names,
            "tokenizer_directory": str(tokenizer_dir),
            "chat_template_applied": True,
            "add_generation_prompt": True,
            "thinking": thinking,
            "smoke_thinking": (
                "the coherence smoke decodes on the SAME thinking setting as "
                "the bank, because a smoke taken on a different surface says "
                "nothing about the surface that was reviewed."
            ),
            "cases": len(rows),
            "runtime": "hf-transformers",
        },
        "decode_path_preflight": preflight,
        "emit_zero_floor": floor,
        "decode_smoke": {
            "prompts_file": str(args.smoke_prompts) if not args.skip_smoke else None,
            "prompts_file_sha256": smoke_digest,
            "prompts_file_sha256_expected": SMOKE_PROMPTS_SHA256,
            "digest_matches_sealed": smoke_digest == SMOKE_PROMPTS_SHA256,
            "maximum_new_tokens": int(args.smoke_maximum_new_tokens),
            "records": smoke_records,
            "verdict": "pending -- a human reads these three and says coherent or incoherent",
        },
        "probe_prompts": {
            "prompts_file": str(args.probe_prompts),
            "prompts_file_sha256": probe_digest,
            "prompts_file_sha256_expected": PROBE_PROMPTS_SHA256,
            "digest_matches_sealed": probe_digest == PROBE_PROMPTS_SHA256,
            "count": len(probe_payload["prompts"]),
        },
        "child_responses": responses,
        "parent_responses": [],
        "parent_control_note": (
            "A PARENT calibration run has no parent control: this model IS the "
            "control that later children are read against.  A child run of this "
            "same file supplies --parent-dossier to oc_assess.py, and the parent "
            "text it is compared with is the text in THIS file."
        ),
        "wall_time_seconds": round(time.monotonic() - started, 3),
    }
    body["lane_digest"] = canonical_sha256(
        [[record["id"], record["response_verbatim"]] for record in responses]
    )
    body["lane_digest_validity"] = {
        "note": (
            "the omni analogue of the LFM2 parent-lane anchor 865f006e...  It is "
            "valid ONLY for this exact tuple; any element moving moves the "
            "digest and a moved lane grades nothing."
        ),
        "bank_file_sha256": bank_digest,
        "prompt_surface_sha256": surface,
        "dtype": str(args.dtype),
        "compute_architecture": architecture,
        "transformers_version": export["transformers_version"],
        "maximum_new_tokens": int(args.maximum_new_tokens),
        "runtime": "hf-transformers",
        "decode_path": "manual argmax loop with use_cache",
        #: A hybrid parent's Gated-DeltaNet layers run through `fla`'s Triton
        #: kernels or through transformers' own torch implementations, per
        #: primitive, and transformers binds them silently.  The paths differ in
        #: bf16 accumulation order, so this belongs in the tuple beside the dtype
        #: for exactly the same reason the dtype does.
        "linear_attention_kernel_path": export.get(
            "linear_attention_kernels", {}
        ).get("kernel_path"),
        #: THE SCHEMA TRAVELS WITH THE STRING, because the string's MEANING
        #: changed once already.  Dossiers written before 2026-09-02 carry the
        #: v1 inventory rule, in which "fla-0.5.2" meant *fla present and
        #: causal_conv1d absent* and "torch-fallback" was inferred rather than
        #: measured -- and that inference was wrong.  Without this element a
        #: reader would see the kernel field move between a v1 dossier and a v2
        #: one and conclude the SURFACE moved, which is precisely the
        #: misattribution the field exists to prevent.  A v1 string and a v2
        #: string are not comparable; only two v2 strings are.
        "linear_attention_kernel_report_schema": export.get(
            "linear_attention_kernels", {}
        ).get("kernel_report_schema"),
        #: A reasoning parent renders a DIFFERENT prompt depending on this, so
        #: it sits beside the dtype for the same reason: two reviews taken on
        #: two settings are two surfaces, not one surface measured twice.
        "chat_template_thinking": thinking["requested"],
        "chat_template_thinking_took_effect": thinking["took_effect"],
    }
    if load_plan is not None:
        body["lane_digest_validity"]["mapped_placement"] = {
            "device_map": export["device_map"],
            "device_map_source": export["device_map_source"],
            "input_device": decode_device,
            "device_compute_architectures": export["device_compute_architectures"],
            "floating_parameter_dtypes": export["floating_parameter_dtypes"],
        }
    body["dossier_sha256"] = canonical_sha256(body)
    digest = write_canonical(Path(args.out), body)
    log(f"dossier {body['dossier_sha256'][:16]} -> {args.out}")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "file_sha256": digest,
                "dossier_sha256": body["dossier_sha256"],
                "lane_digest": body["lane_digest"],
                "cases": len(responses),
                "emit_zero_floor_passed": floor["passed"],
                "decode_path_agree": preflight["agree"],
                "is_random_init": body["is_random_init"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--tokenizer-dir", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--parent-code")
    ap.add_argument("--stage", default="parent", choices=["parent", "fold", "distill", "finetune"])
    ap.add_argument("--loader-class")
    ap.add_argument(
        "--bank",
        type=Path,
        default=repo / "research" / "lfm2_semantic_whole_response_prompts_v1.json",
    )
    ap.add_argument("--bank-sha256")
    ap.add_argument("--lanes", help="comma-separated case ids to run")
    ap.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "drop cases this text runner must not decode and name them in the "
            "dossier: a case whose modality_in is not text (its prompt cannot be "
            "built here) and a case marked runnable:false (its ANSWER is a media "
            "artifact, so decoding its text prompt would file words as a video, "
            "an image or a track).  Without this the runner REFUSES a multimodal "
            "case rather than decoding half of it."
        ),
    )
    ap.add_argument("--max-cases", type=int)
    ap.add_argument(
        "--probe-prompts",
        type=Path,
        default=repo / "experiments" / "matrix_campaign_probe_prompts_v1.json",
    )
    ap.add_argument(
        "--smoke-prompts",
        type=Path,
        default=repo / "experiments" / "matrix_campaign_smoke_prompts_v1.json",
    )
    ap.add_argument(
        "--thinking",
        default="template-default",
        choices=["template-default", "off"],
        help=(
            "`template-default` decodes exactly what the checkpoint's own chat "
            "template produces, which on Qwen3.5 OPENS a think block and spends "
            "the token budget on chain-of-thought.  `off` passes "
            "enable_thinking=False.  Both are recorded in the dossier and in the "
            "lane-digest validity tuple; which one the omni protocol adopts is a "
            "proposed decision line for Cyril, not a default this runner picks."
        ),
    )
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", choices=PLACEMENT_STRATEGIES,
                    help="optional direct stock-HF sharded placement; requires explicit --max-memory")
    ap.add_argument("--max-memory", action="append",
                    help="repeat DEVICE=integerMiB/GiB for cpu and each visible CUDA index; 0MiB excludes a device")
    ap.add_argument("--require-compute-architecture")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument("--smoke-maximum-new-tokens", type=int, default=128)
    ap.add_argument("--extra-eos-tokens", default="<|im_end|>,<|endoftext|>")
    ap.add_argument(
        "--no-logits-to-keep",
        action="store_true",
        help=(
            "force the prefill to materialise logits for EVERY position.  It "
            "exists so the claim 'keeping one position changes no token' can be "
            "A/B-measured on a REAL model rather than only on a stub, which is "
            "the only place a bf16 lm_head GEMM over [1,1,H] and over [1,L,H] "
            "can differ in the last bits and flip a near-tie argmax."
        ),
    )
    ap.add_argument(
        "--tiny-random",
        action="store_true",
        help=(
            "build a shrunken RANDOMLY INITIALISED model of the same class from "
            "--model-dir's config and tokenizer.  A mechanics bed; every artifact "
            "it produces is labelled is_random_init and cannot be scored."
        ),
    )
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
