"""Target-free native audio -> text sampling for the held-answer review.

Only the authentic generation prefix and clean observations enter inference.
Reference answer IDs/positions are discarded before validation or model access.
This helper neither grades answers nor grants a production/review pass.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('chimera_audio_sampling_driver', HERE / 'train_audio.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
R, Error = A.R, A.Error
POLICY = 'audio_native_prefix_only_greedy_v1'


def prefix_segment(model, segment, prompt_length):
    """Own clones, validate prefix text ordinals, and never read an answer tail."""
    import torch
    ids = segment['ids']
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64 or
            type(prompt_length) is not int or not 0 < prompt_length <= ids.shape[1]):
        raise Error('audio free answers require the authentic single-row generation prefix')
    value = {key: segment[key].detach().clone() for key in ('features', 'types', 'positions', 'indices', 'sigma')}
    value.update(ids=ids[:, :prompt_length].detach().clone(), media_rows=value['types'].numel())
    positions = segment.get('source_positions', {}).get('h3')
    if positions is None or positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[1] < prompt_length or positions.shape[2] != 3:
        raise Error('audio free answers need the source-pinned standalone H3 observation chart')
    value['source_positions'] = {'h3': positions[:, :prompt_length].detach().clone()}
    A.validate_clean_observation(model, value, prompt_length)
    media = set(value['indices'][0].tolist())
    text = [i for i in range(prompt_length) if i not in media]
    xyz = value['source_positions']['h3']
    expected = torch.zeros(1, len(text), 3, device=xyz.device, dtype=xyz.dtype)
    expected[0, :, 0] = torch.arange(len(text), device=xyz.device, dtype=xyz.dtype)
    if (not text or not torch.equal(xyz[:, text], expected) or
            bool(((value['ids'][0, text] < 0) | (value['ids'][0, text] >= model.base_vocab)).any())):
        raise Error('audio free-answer text slots must preserve the amended native text ordinal chart and base vocabulary')
    return value, len(text)


def sample_answer(engine, segment, *, prompt_length, completion_eos_id, max_new_tokens=256):
    """Greedy full-depth generation without target IDs, masks or teacher logits.

    The H3 unit can read clean audio and original prompt text. Generated answers
    enter only the causal text trunk, with newly constructed text coordinates.
    Use the pinned native tokenizer's completion EOS; caller separately records
    held source/checkpoint identities and grades the decoded answer.
    """
    import torch
    model = engine.hybrid
    if (type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096 or
            type(completion_eos_id) is not int or not 0 <= completion_eos_id < model.base_vocab or
            type(prompt_length) is not int or prompt_length + max_new_tokens > 65536):
        raise Error('audio free-answer EOS and bounded token budget must be explicit')
    value, text_count = prefix_segment(model, segment, prompt_length)
    P = A.T.load_module(HERE / 'h3_audio_probe.py', 'chimera_audio_sampling_tensor_digest')
    prefix_digest = R.digest({key: P.tensor_sha(value[key]) for key in ('ids', 'features', 'types', 'positions', 'indices', 'sigma')} |
                             {'source_positions': P.tensor_sha(value['source_positions']['h3'])})
    generated, stopped = [], False
    with torch.no_grad(), A.prompt_only_h3(model, engine.placement.container, prompt_length):
        for _ in range(max_new_tokens):
            out = model.forward(value)
            index = torch.tensor([value['ids'].shape[1] - 1], dtype=torch.int64, device=value['ids'].device)
            logits = model.text_logits(out['normed'], index)[0, :model.base_vocab]
            if logits.numel() != model.base_vocab or not bool(torch.isfinite(logits).all()):
                raise Error('audio free-answer native text logits are missing or nonfinite')
            token = int(logits.argmax())
            generated.append(token)
            if token == completion_eos_id:
                stopped = True
                break
            value['ids'] = torch.cat((value['ids'], torch.tensor([[token]], dtype=torch.int64, device=value['ids'].device)), dim=1)
            xyz = value['source_positions']['h3']
            next_position = torch.tensor([[[float(text_count), 0., 0.]]], device=xyz.device, dtype=xyz.dtype)
            value['source_positions']['h3'] = torch.cat((xyz, next_position), dim=1)
            text_count += 1
    return {'schema': 'chimera_recovery.audio_free_answer_sample.v1', 'policy': POLICY,
        'input_prefix_sha256': prefix_digest, 'prompt_length': prompt_length, 'generated_ids': generated,
        'completion_eos_id': completion_eos_id, 'stopped_on_eos': stopped, 'max_new_tokens': max_new_tokens,
        'teacher_forced': False, 'teacher_logits_used': False, 'target_tail_discarded': True,
        'read_policy': A.READ_POLICY, 'native_parent_xyz_parity': False,
        'engine_identity_sha256': R.digest(engine.identity), 'sampler_sha256': R.sha(Path(__file__)),
        'review_pass_granted': False, 'training_started': False}
