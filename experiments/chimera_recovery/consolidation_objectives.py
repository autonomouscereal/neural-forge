"""Native D4 backward callbacks, with no optimizer creation or stepping.

All callbacks return the unscaled diagnostic mean and scale the actual gradient
by the entire accumulation window's measured normalization factor. Production
receipt/source admission belongs to consolidation_dispatch, never these tensor
primitives. Test tensors do not create teacher seals or accepted stage states.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import math

from . import training as R

Error = R.ReadinessError
TEXT = "qwen_kimi_assistant_kd"
GUIDED = "guided_velocity"
MUSIC = "music_code_band_kl"
VISION = "text_answer_kd_plus_vit_feature_mse"
AUDIO = "authored_audio_assistant_ce"
SPEECH = "supervised_speech_rectified_flow"
GEOMETRY = "supervised_geometry_rectified_flow"
OBJECTIVES = {TEXT, GUIDED, MUSIC, VISION, AUDIO, SPEECH, GEOMETRY}


def count_matches(expected, actual):
    if type(expected) is not int or expected < 1 or actual != expected:
        raise Error("D4 measured supervised denominator differs from sealed replay row")


def weight_value(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise Error("D4 objective requires a finite positive window weight")
    return float(weight)


def scalar_flow_count(types, mask, bands):
    import torch
    if mask.shape != types.shape or not bool(torch.isfinite(mask).all()) or bool(((mask != 0) & (mask != 1)).any()):
        raise Error("D4 flow scalar denominator needs exact binary owned rows")
    return sum(int(((types == kind) & (mask == 1)).sum()) * (band["emission"][1] - band["emission"][0])
               for kind, band in bands.items())


@contextmanager
def all_h3_prompt_only(hybrid, prompt_length):
    from . import train_audio as A
    containers = sorted({p.container for p in hybrid.layout if p.spec.source == "h3" and 3 in p.type_ids})
    if not containers:
        raise Error("D4 audio/speech replay requires actual H3 placement")
    with ExitStack() as stack:
        for container in containers:
            stack.enter_context(A.prompt_only_h3(hybrid, container, prompt_length))
        yield


def text_backward(module, hybrid, batch, objective_config, units, weight):
    import torch
    ids = batch["input_ids"]
    targets = {key[7:]: value for key, value in batch.items() if key.startswith("target.")}
    owned = batch["assistant_prediction_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or owned.shape != ids.shape or owned.dtype != torch.bool:
        raise Error("D4 text needs one native row and boolean assistant prediction ownership")
    count_matches(units, int(owned.sum()))
    length = ids.shape[1]
    if any(value.shape[0] != length for value in targets.values()):
        raise Error("D4 text target positions differ from native input")
    for key in ("q_w", "k_w", "ce_w", "eos_w", "ul_w"):
        if key in targets and (not bool(torch.isfinite(targets[key]).all()) or bool((targets[key] < 0).any())
                               or bool(targets[key][~owned[0]].any())):
            raise Error("D4 text teacher/CE objectives include unowned prompt or padding positions")
    config = hybrid.T.ObjectiveConfig(**objective_config)
    if config.text_vocab_rows != hybrid.base_vocab:
        raise Error("D4 text objective vocabulary differs from native text rows")
    def loss(normed, context):
        head = hybrid.engine.head_weight[:hybrid.base_vocab]
        parts, grad = hybrid.T.chunked_objective(normed.to(head.device, head.dtype), head, targets, config,
                                                denominator=float(units), backward=True)
        surrogate = (normed * grad.to(normed.device, normed.dtype)).sum()
        return surrogate - surrogate.detach() + torch.tensor(float(parts["total"]), device=normed.device)
    return module.full_depth_forward_backward(hybrid, {"ids": ids}, loss, loss_scale=weight_value(weight))[0]


def flow_backward(module, hybrid, segment, target, mask, receipt, units, weight):
    count_matches(units, scalar_flow_count(segment["types"], mask, hybrid.bands))
    allocation = {int(k): tuple(v) for k, v in receipt["noisy_feature_bands"].items()}
    if receipt["target_kind"] not in {"x0", "canonical_velocity"} or receipt["prediction_kind"] != "x0":
        raise Error("D4 flow requires explicit native x0/canonical velocity conversion")
    def loss(normed, context):
        return R.flow_velocity_loss(hybrid.emission(normed, context), target, segment["types"], mask, hybrid.bands,
                                    target_kind=receipt["target_kind"], param="x0",
                                    x_t=segment["features"], sigma=segment["sigma"], noisy_feature_bands=allocation)
    return module.full_depth_forward_backward(hybrid, segment, loss, loss_scale=weight_value(weight))[0]


def guided_backward(module, hybrid, batch, item, receipt, units, weight):
    """Keep genuine producer head/sign semantics and original pair-mean loss."""
    from . import train_media as T
    count_matches(units, 1)
    compute = R.ReadyObjectiveAdapter(receipt).bind(item["route"], batch, hybrid.bands,
                parent_signs={int(k): v for k, v in item.get("parent_signs", {}).items()})
    segment = T.make_segment(batch, media_placeholder=hybrid.media_placeholder)
    return module.full_depth_forward_backward(hybrid, segment,
        lambda normed, context: compute(hybrid.emission(normed, context)), loss_scale=weight_value(weight))[0]


def geometry_segment(batch, *, media_placeholder):
    """Feed native progress to HY3D; canonical sigma stays loss-only metadata."""
    from . import train_media as T
    segment = T.make_segment({k: v for k, v in batch.items() if k != "charted_source_positions"},
                             media_placeholder=media_placeholder)
    segment["source_positions"] = {"hy3d": batch["charted_source_positions"]}
    return segment


def geometry_backward(module, hybrid, batch, compute, units, weight):
    """Authenticated canonical velocity MSE, raw sign handled by source adapter.

    GeometryFlowAdapter owns source/target reconstruction and the fixed -1 raw
    head sign. No endpoint conversion or division by native progress occurs.
    """
    import torch
    types, mask = batch["native_media_type_ids"], batch["native_media_loss_weights"]
    if (hybrid.bands[7]["feature"] != (576, 640) or hybrid.bands[7]["emission"] != (320, 384)
            or not torch.equal(mask, batch["native_media_loss_mask"])
            or bool((types[mask != 0] != 7).any())):
        raise Error("D4 geometry needs the authentic type7 64-channel latent allocation")
    count_matches(units, scalar_flow_count(types, mask, {7: hybrid.bands[7]}))
    canonical, progress = batch["native_media_canonical_sigma"], batch["native_media_noise_levels"]
    owned = types == 7
    if (canonical.shape != types.shape or progress.shape != types.shape
            or not bool(torch.isfinite(canonical).all()) or bool(((canonical[owned] <= 0) | (canonical[owned] > 1)).any())
            or not torch.equal(progress[owned], 1 - canonical[owned])):
        raise Error("D4 geometry canonical sigma and native progress must remain separate exact fields")
    segment = geometry_segment(batch, media_placeholder=hybrid.media_placeholder)
    return module.full_depth_forward_backward(hybrid, segment,
        lambda normed, context: compute(hybrid.emission(normed, context)), loss_scale=weight_value(weight))[0]


def music_backward(module, hybrid, batch, receipt, units, weight):
    import torch
    ids, codes, probs = batch["input_ids"], batch["topk_codes"], batch["topk_logprobs_band"]
    prompt = receipt["prompt_length"]
    count_matches(units, int(codes.shape[0]))
    rows = hybrid._recovery_d4_music["contract"]["code_rows"]
    if (ids.dtype != torch.int64 or ids.shape != (1, prompt + units - 1) or prompt < 1
            or codes.ndim != 2 or probs.shape != codes.shape or not bool(torch.isfinite(probs).all())
            or bool(((codes < 0) | (codes >= rows)).any()) or bool((probs > 1e-5).any())
            or bool((probs.exp().sum(-1) > 1.00001).any()) or any(v.unique().numel() != v.numel() for v in codes)
            or bool(((ids[:, :prompt] < 0) | (ids[:, :prompt] >= hybrid.base_vocab)).any())
            or bool(((ids[:, prompt:] < hybrid.music_band_begin) | (ids[:, prompt:] >= hybrid.music_band_begin + rows)).any())):
        raise Error("D4 music requires authentic previous-code prefix and conditional64000-row top-k targets")
    positions = torch.arange(prompt - 1, prompt - 1 + units, device=hybrid.device)
    def loss(normed, context):
        head = hybrid._recovery_d4_music["head"].to(normed.device, normed.dtype)
        return module.band_kl(normed[0, positions], head, codes.to(normed.device), probs.to(normed.device))
    return module.full_depth_forward_backward(hybrid, {"ids": ids}, loss, loss_scale=weight_value(weight))[0]


def audio_backward(module, hybrid, segment, target_mask, receipt, units, weight):
    from . import train_audio as A
    prompt = receipt["prompt_length"]
    targets, actual = A.assistant_ce_targets(hybrid, segment, target_mask, prompt, receipt["completion_eos_id"])
    count_matches(units, actual)
    A.validate_clean_observation(hybrid, segment, prompt)
    with all_h3_prompt_only(hybrid, prompt):
        return module.full_depth_forward_backward(hybrid, segment, A.ce_head_loss(hybrid, targets, actual),
                                                  loss_scale=weight_value(weight))[0]


def speech_backward(module, hybrid, segment, target, mask, receipt, units, weight):
    from . import train_speech_flow as S
    S.validate_inputs(hybrid, segment, target, mask, receipt["prompt_length"])
    with all_h3_prompt_only(hybrid, segment["ids"].shape[1]):
        return flow_backward(module, hybrid, segment, target, mask, receipt, units, weight)


def vision_backward(module, hybrid, segment, targets, features, receipt, units, weight):
    from . import train_vision as V
    count_matches(units, 1)  # One combined mean text+feature objective per pair.
    positions = receipt["answer_positions"]
    if type(positions) is not int or positions < 1 or int((targets["q_w"] > 0).sum()) != positions:
        raise Error("D4 vision assistant target denominator differs")
    result = V.feature_forward_backward(module, hybrid, segment, module.d3d_head_loss(hybrid, targets, denominator=float(positions)),
                                       receipt["unit"], features, loss_scale=weight_value(weight))
    return result["loss"]
