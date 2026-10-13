import ast
import __future__
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.chimera_recovery import train_text as T


def sources(rows=((10, 11, 12), (20, 21))):
    return [SimpleNamespace(name=f"source-{i}", row_ids=list(ids)) for i, ids in enumerate(rows)]


def test_one_pass_partial_batch_and_exact_resume_refuse_changed_plan():
    full = T.OnePassSampler(sources(), 20261013)
    first = full.next_rows(3)
    state = full.state()
    remainder = full.next_rows(3)
    assert len(remainder) == 2 and len(set(first + remainder)) == 5
    assert full.next_rows(3) == []
    resumed = T.OnePassSampler(sources(), 20261013)
    resumed.load_state(state)
    assert resumed.next_rows(3) == remainder
    with pytest.raises(T.ReadinessError, match="plan_sha256"):
        T.OnePassSampler(sources(((10, 11, 99), (20, 21))), 20261013).load_state(state)
    with pytest.raises(T.ReadinessError, match="seed"):
        T.OnePassSampler(sources(), 17).load_state(state)


def test_one_pass_invalid_identity_and_cursor():
    with pytest.raises(T.ReadinessError, match="repeats"):
        T.OnePassSampler(sources(((1, 1),)), 0)
    sampler = T.OnePassSampler(sources(), 0)
    with pytest.raises(T.ReadinessError, match="cursor"):
        sampler.load_state({**sampler.state(), "cursor": 6})


def test_resume_rejects_internal_progress_and_adam_inconsistency():
    import numpy as np
    sampler = T.OnePassSampler(sources(), 2)
    for source in sampler.sources:
        source.cache = SimpleNamespace(index={}, corpus_assistant_mask=np.zeros((22, 8), dtype=bool))
    for (source_index, row_id), length, supervised in zip(sampler.plan[:3], [7, 6, 6], [4, 4, 3]):
        cache = sampler.sources[source_index].cache
        cache.index[row_id] = ("synthetic", 0, 0, length)
        cache.corpus_assistant_mask[row_id, 1:1+supervised] = True
    sampler.next_rows(3)
    progress = {"rows": 3, "real_tokens": 19, "compute_tokens": 21, "supervised_positions": 11, "applied_steps": 1}
    optimizer = SimpleNamespace(state={"param": {"step": 1}})
    assert T.verify_progress(sampler, progress, 1, 19, 3, 1, optimizer) == progress
    with pytest.raises(T.ReadinessError, match="cursor"):
        T.verify_progress(sampler, progress, 2, 19, 3, 1, optimizer)
    with pytest.raises(T.ReadinessError, match="counters"):
        T.verify_progress(sampler, {**progress, "real_tokens": 20}, 1, 19, 3, 1, optimizer)
    with pytest.raises(T.ReadinessError, match="Adam"):
        T.verify_progress(sampler, progress, 1, 19, 3, 1, SimpleNamespace(state={"param": {"step": 2}}))
    with pytest.raises(T.ReadinessError, match="row-plan prefix"):
        T.verify_progress(sampler, {**progress, "real_tokens": 20}, 1, 20, 3, 1, optimizer)


def test_amendment_must_be_appended_and_pinned(tmp_path):
    amendment = tmp_path / "extra.md"
    amendment.write_bytes(b"dated recovery\n")
    document = tmp_path / "prereg.md"
    document.write_bytes(b"## 1. Scope\nfrozen\n## Amendments\ndated recovery\n")
    result = T.verify_amendment(document, amendment)
    assert result["document_sha256"] == T.sha(document)
    amendment.write_bytes(b"moved recovery\n")
    with pytest.raises(T.ReadinessError, match="absent"):
        T.verify_amendment(document, amendment)


def actual_builder():
    torch = pytest.importorskip("torch")
    raw = os.environ.get("CHIMERA_STREAMED_BASELINE")
    if not raw:
        pytest.skip("exact production streamed baseline is required")
    candidate, receipt = T.patch_trainer(Path(raw))
    node = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef) and n.name == "build_batch")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-patched-build-batch", "exec", flags=__future__.annotations.compiler_flag), ns)
    return ns["build_batch"], candidate, receipt


def cached_sources(torch):
    def row(width, ids, mask):
        real = len(ids)
        return {"ids": torch.tensor(ids + [0] * (width - real)), "length": real,
                "assistant_mask": torch.tensor(mask + [False] * (width - real)),
                "stored": torch.arange(width) < real - 1,
                "topk_ids": torch.zeros(width, 2, dtype=torch.int64),
                "topk_logprobs": torch.full((width, 2), -1.)}
    data = [{1: row(12, [1, 2, 3, 4], [False, False, True, True]),
             2: row(12, [5, 6, 7, 8, 9], [False, False, True, True, True])},
            {3: row(6, [10, 11, 12], [False, True, True])}]
    return [SimpleNamespace(name=f"cache-{i}", kind="text", xtok=None, row_ids=sorted(records),
                            cache=SimpleNamespace(row=lambda rid, records=records: records[rid]))
            for i, records in enumerate(data)]


def test_actual_builder_dynamic_width_and_accumulation_gradient_reference():
    torch = pytest.importorskip("torch")
    builder, _, _ = actual_builder()
    src = cached_sources(torch)
    obj = SimpleNamespace(w_q38=1., w_ce=.1, w_eos=.1, w_ul=0., eos_ids=(9,), onpolicy_divergence="jsd")
    ids, targets, counts = T.cropped_batch(src, [(0, 1), (1, 3)], obj, builder)
    assert ids.shape == (2, 4) and counts == {"rows": 2, "real_tokens": 7, "compute_tokens": 8, "supervised_positions": 4}
    assert targets["q_ids"].shape == (8, 2)
    assert targets["next_ids"].tolist() == [-1, 3, 4, -1, 11, 12, -1, -1]
    assert targets["ce_w"].count_nonzero() == 4 and targets["q_w"].count_nonzero() == 4
    sampler = T.OnePassSampler(src, 1)
    batches, stats = T.accumulation_window(src, sampler, 2, 2, obj, builder)
    assert stats["rows"] == 3 and stats["supervised_positions"] == 7 and len(batches) == 2
    # Unequal batch lengths/counts must share the entire window denominator.
    weights = torch.randn(13, 13, generator=torch.Generator().manual_seed(5), requires_grad=True)
    loss = torch.zeros(())
    for token_ids, target in batches:
        flat = token_ids.reshape(-1)
        good = target["next_ids"] >= 0
        loss = loss + torch.nn.functional.cross_entropy(weights[flat[good]], target["next_ids"][good], reduction="sum") / stats["supervised_positions"]
    loss.backward()
    grad = weights.grad.clone()
    ref = weights.detach().clone().requires_grad_()
    inputs, wanted = [], []
    for s in src:
        for r in s.row_ids:
            item = s.cache.row(r)
            for pos in range(item["length"] - 1):
                if item["assistant_mask"][pos + 1]:
                    inputs.append(item["ids"][pos]); wanted.append(item["ids"][pos + 1])
    reference = torch.nn.functional.cross_entropy(ref[torch.stack(inputs)], torch.stack(wanted))
    reference.backward()
    assert torch.allclose(loss, reference, atol=1e-7) and torch.allclose(grad, ref.grad, atol=1e-7)


def test_cropping_refuses_supervised_padding_and_empty_rows():
    torch = pytest.importorskip("torch")
    builder, _, _ = actual_builder()
    src = cached_sources(torch)
    src[0].cache.row(1)["assistant_mask"][-1] = True
    with pytest.raises(T.ReadinessError, match="padding"):
        T.cropped_batch(src, [(0, 1)], None, builder)


def test_exact_patch_opts_in_without_changing_legacy_sampler():
    _, candidate, receipt = actual_builder()
    assert 'if structured else MixtureSampler(sources, args.seed)' in candidate
    assert 'if not structured and sources[0].cache.seq_len != args.seq_len' in candidate
    assert 'denominator=step_denominator' in candidate and 'step_denominator = float(step_stats["supervised_positions"])' in candidate
    assert receipt["legacy_lane_preserved"] and receipt["production_training_ready"] is False
    assert 'file_sha256(Path(__file__))' in candidate and 'before_first_update' in candidate


def test_patch_refuses_unreviewed_baseline(tmp_path):
    bad = tmp_path / "changed.py"
    bad.write_text("pass\n")
    with pytest.raises(T.ReadinessError, match="baseline moved"):
        T.patch_trainer(bad)
