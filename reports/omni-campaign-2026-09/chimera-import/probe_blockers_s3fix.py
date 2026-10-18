"""Measure the three S3 review blockers on the real assets, before and after.

Prints the pre-fix behaviour (reproduced from the exact rule the old code used)
next to the post-fix behaviour, so the fix is a measured change, not a claim.
"""

import json
import sys

from chimera_omni.importing.bridge import embedding_bridge, token_alignment
from chimera_omni.importing.profiles import qwen35_chimera_config
from chimera_omni.importing.tokenizer import (
    ChimeraTokenizer,
    JsonBpeTokenizer,
    TikTokenBpeTokenizer,
    bytes_to_unicode,
)

QWEN = "/home/cereal/models/Qwen3.5-0.8B"
K3 = "/mnt/ai_data/models/moonshotai/Kimi-K3"

print("---- BLOCKER 1: a flat-layout yarn rope ----")
text = json.load(open(QWEN + "/config.json"))["text_config"]
flat = dict(text)
rope = flat.pop("rope_parameters")
flat["rope_theta"] = rope["rope_theta"]
flat["partial_rotary_factor"] = rope["partial_rotary_factor"]
flat["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0}


def old_rule(cfg):
    """The pre-fix rule: rope_type ONLY from rope_parameters, theta/factor flat."""

    nested = cfg.get("rope_parameters")
    source = dict(nested) if isinstance(nested, dict) else {}
    for key in ("rope_theta", "partial_rotary_factor"):
        if key not in source and key in cfg:
            source[key] = cfg[key]
    return source.get("rope_type", "default")


print("pre-fix rope_type seen by the guard:", old_rule(flat), "-> scaled rope accepted")
try:
    qwen35_chimera_config({"text_config": flat})
    print("post-fix: NO REFUSAL (bad)")
except ValueError as exc:
    print("post-fix: ValueError", str(exc)[:110])

both = dict(text)
both["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0}
try:
    qwen35_chimera_config({"text_config": both})
    print("post-fix both-layouts: NO REFUSAL (bad)")
except ValueError as exc:
    print("post-fix both-layouts: ValueError", str(exc)[:110])

print("---- BLOCKER 2: control ids bridged by the spelling of their name ----")
from tokenizers import Tokenizer  # noqa: E402

raw = Tokenizer.from_file(QWEN + "/tokenizer.json")
old_base = JsonBpeTokenizer(raw, name="qwen_pre_fix", special_ids=[])
new_tok = ChimeraTokenizer.from_qwen(QWEN)
k3_tok = ChimeraTokenizer.from_k3(K3)

marker_id = 248053
print("pre-fix  token_bytes(%d) = %r" % (marker_id, old_base.token_bytes(marker_id)))
print("post-fix token_bytes(%d) = %r  special=%s"
      % (marker_id, new_tok.base.token_bytes(marker_id), new_tok.base.is_special_id(marker_id)))
pieces = k3_tok.encode(old_base.token_bytes(marker_id).decode("utf-8"))
print("pre-fix  destination pieces for that name:", pieces)

k3_reserved = k3_tok.base.vocab_size - 256
old_k3 = TikTokenBpeTokenizer(k3_tok.base._encoding, name="k3_pre_fix", vocab_size=k3_tok.base.vocab_size)
print("pre-fix  k3 token_bytes(%d) = %r" % (k3_reserved, old_k3.token_bytes(k3_reserved)))
print("post-fix k3 token_bytes(%d) = %r" % (k3_reserved, k3_tok.base.token_bytes(k3_reserved)))


class Window:
    """A view of the Qwen tokenizer restricted to the added-token id window."""

    def __init__(self, tokenizer, begin, end):
        self.base = tokenizer.base
        self.source = tokenizer.source
        self.base_vocab_size = tokenizer.base_vocab_size
        self._begin = begin
        self._end = end

    def encode(self, text):
        return list(self.base.encode_ids(text))

    def decode(self, ids):
        return self.base.decode_ids(list(ids))


def window_alignment(base, begin, end):
    aligned = 0
    unmapped = 0
    for token_id in range(begin, end):
        payload = base.token_bytes(token_id)
        if not payload:
            unmapped += 1
            continue
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            unmapped += 1
            continue
        aligned += 1
    return aligned, unmapped


print("ids 248040..248069  pre-fix  aligned=%d unmapped=%d" % window_alignment(old_base, 248040, 248070))
print("ids 248040..248069  post-fix aligned=%d unmapped=%d" % window_alignment(new_tok.base, 248040, 248070))

pairs, receipt = token_alignment(new_tok, k3_tok, ("<|im_start|>user\nhi<|im_end|>",))
print("corpus with markers: aligned=%s unmapped=%s special=%s"
      % (receipt["aligned_tokens"], receipt["unmapped_tokens"], receipt["special_tokens"]))
print("marker ids in pairs:", [i for i in new_tok.encode("<|im_start|>user") if new_tok.base.is_special_id(i)],
      "->", [i in pairs for i in new_tok.encode("<|im_start|>user") if new_tok.base.is_special_id(i)])

print("---- BLOCKER 3: segment packet width vs the patchifier ----")
from chimera_omni.ledger import build_fixed_band_ledger  # noqa: E402
from chimera_omni.native_media import volume_to_native_patches  # noqa: E402
from chimera_omni.native_segments import NATIVE_SEGMENT_SPECS, build_native_segment_packet  # noqa: E402
import torch  # noqa: E402

ledger = build_fixed_band_ledger(256)
packet = build_native_segment_packet(ledger, "volume", (33, 35, 37))
emitted = volume_to_native_patches(torch.zeros(1, 1, 33, 35, 37), patch=16)
print("pre-fix  volume packet feature_dim = 3072 (hard default)")
print("post-fix volume packet feature_dim = %d, patchifier emits %d, segments %d/%d"
      % (packet["feature_dim"], int(emitted.features.shape[-1]),
         packet["segment_count"], int(emitted.features.shape[1])))
print("other modalities unchanged:",
      {k: build_native_segment_packet(ledger, k, (64, 64) if v.patch_axes == ("height", "width")
                                      else (9216,) if v.patch_axes == ("samples",) else (2, 64, 64))["feature_dim"]
       for k, v in NATIVE_SEGMENT_SPECS.items()})
sys.stdout.flush()
