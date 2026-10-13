"""Native text projection and role/body/EOS ownership around observed media."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/audio_dialogue.py"
spec = importlib.util.spec_from_file_location("audio_dialogue_test", PATH)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


class Tokenizer:
    """Protocol fixture; actual parent tokenizer proof runs separately."""
    chat_template = "audited_fixture_with_native_eos_and_empty_think"
    bad_prefix = False
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **options):
        text = "".join("<|im_start|>" + message["role"] + "\n" +
                       ("<think>\n\n</think>\n\n" if message["role"] == "assistant" else "") +
                       message["content"] + "<|im_end|>\n" for message in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            if self.bad_prefix:
                text += "changed"
        return self(text, add_special_tokens=False) if tokenize else text
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        ids, offsets, index = [], [], 0
        while index < len(text):
            special = next((token for token in ("<|im_start|>", "<|im_end|>") if text.startswith(token, index)), None)
            if special:
                ids.append(1000 if special == "<|im_start|>" else 1001)
                offsets.append((index, index + len(special)))
                index += len(special)
            else:
                ids.append(ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
        return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}


def record():
    return {"route": "audio_understand", "messages": [{"role": "user", "content": "Listen to the tone."},
            {"role": "assistant", "content": "One tone."}], "supervision": {"policy": "assistant_only", "message_mask": [False, True]}}


def test_actual_role_anchor_preserves_every_text_id_and_body_eos_only():
    row, tokenizer = record(), Tokenizer()
    packed = P.pack_dialogue(row, tokenizer, media_rows=80, max_length=2048)
    ids, mask, indices = (packed[key] for key in ("input_ids", "assistant_target_mask", "media_indices"))
    assert len(indices) == 80 and all(ids[index] == P.PLACEHOLDER and mask[index] is False for index in indices)
    assert [token for index, token in enumerate(ids) if index not in indices] == packed["parent_text_ids"]
    assert ids[indices[-1] + 1] == 1001  # native user-close; media stayed inside user role
    owned = [token for token, selected in zip(ids, mask) if selected]
    assert owned == [*map(ord, "One tone."), 1001]
    assert sum(mask) == packed["assistant_targets"] == 10
    assert not any(mask[:packed["prompt_length"]]) and mask[packed["prompt_length"]]
    assert packed["next_token_mask"] == mask[1:] + [False]
    assert packed["production_training_ready"] is False


@pytest.mark.parametrize("problem,reason", [("prefix", "exact reference prefix"), ("role", "unambiguous"), ("length", "no waveform"), ("media_text", "ordinary parent text")])
def test_prefix_role_media_erasure_or_truncation_refuses(problem, reason):
    row, tokenizer, limit = record(), Tokenizer(), 2048
    if problem == "prefix":
        tokenizer.bad_prefix = True
    elif problem == "role":
        row["messages"][0]["content"] += "<|im_end|>"
    elif problem == "length":
        limit = 10
    else:
        row["messages"][0]["content"] = [{"type": "audio", "audio": "source.wav"}]
    with pytest.raises(P.Error, match=reason):
        P.pack_dialogue(row, tokenizer, media_rows=80, max_length=limit)
