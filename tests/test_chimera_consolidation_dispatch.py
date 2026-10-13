"""Native D4 safety/flow tests; fixture outputs never certify production stages."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid

import pytest
from experiments.chimera_recovery import consolidation_dispatch as D, consolidation_objectives as O
from test_chimera_consolidation_runtime import fixture, replay, backward


def write(path, value):
    path.write_bytes(D.I.canonical(value) + b"\n")
    return D.I.pin(path)


def runtime_fixture():
    assignment = dict(schema="chimera_recovery.d4_allocated_runtime.v1", job_sha256="a" * 64,
        allocation_group_id="61852066-afdd-4f24-a4b3-06d5c9c830e0", container_id="c" * 64,
        container_name="fixture-only", image_id="sha256:" + "d" * 64, gpu_uuids=["GPU-2ce6a95c-7b1f-644a-3808-d781732a0582"],
        root_controller_sha256="b" * 64, source_tree_sha256="e" * 64, max_proof_age_seconds=45,
        torch_version="fixture", image_python="/fixture/python", minimum_mem_available_gib=2,
        mounts=[dict(source="/fixture/input", target="/inputs/fixture", read_only=True)])
    pg = [dict(group_id=assignment["allocation_group_id"], gpu_uuid=assignment["gpu_uuids"][0], state="running", owner="codex", server="ai",
               kind="generic", name="fixture-only", container_id=assignment["container_id"], details={"gpu_uuids": assignment["gpu_uuids"]})]
    docker = dict(Id=assignment["container_id"], Image=assignment["image_id"], Name="/fixture-only", State={"Running": True},
        Config={"Labels": {"ai.local.gpu-controller": "v1", "ai.local.allocation": assignment["allocation_group_id"],
            "ai.local.owner": "codex", "ai.local.kind": "generic", "ai.local.gpu-uuids": json.dumps(assignment["gpu_uuids"])}},
        HostConfig={"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": assignment["gpu_uuids"]}]},
        Mounts=[dict(Type="bind", Source="/fixture/input", Destination="/inputs/fixture", RW=False)])
    cuda = dict(uuids=assignment["gpu_uuids"], device_count=1, ordinals=[0], bf16_matmul_finite=True,
                torch_version="fixture", python_executable="/fixture/python")
    host = dict(mem_available_gib=4, foreign_gpu_processes=[], source_tree_sha256=assignment["source_tree_sha256"])
    return assignment, pg, docker, cuda, host


def test_live_root_proof_cannot_be_replaced_by_active_metadata_or_foreign_controller():
    assignment, pg, docker, cuda, host = runtime_fixture()
    now = datetime(2026, 10, 13, 21, 0, tzinfo=timezone.utc)
    proof = D.attest_runtime(assignment, pg, docker, cuda, host, now=now)
    assert D.runtime_gate(assignment, proof, assignment["job_sha256"], now)["job_sha256"] == "a" * 64
    for changes in ({"postgres_live": False}, {"host_budget_verified": False}, {"container_id": "f" * 64},
                    {"captured_at": (now - timedelta(seconds=46)).isoformat()}, {"captured_at": (now + timedelta(seconds=1)).isoformat()}):
        with pytest.raises(D.Error, match="runtime lease"):
            D.runtime_gate(assignment, dict(proof, **changes), assignment["job_sha256"], now)
    pg[0]["state"] = "starting"
    with pytest.raises(D.Error, match="PG lease"):
        D.attest_runtime(assignment, pg, docker, cuda, host, now=now)
    pg[0]["state"] = "running"
    docker["Mounts"][0]["RW"] = True
    with pytest.raises(D.Error, match="readonly"):
        D.attest_runtime(assignment, pg, docker, cuda, host, now=now)
    config = {"source": "a", "runtime_assignment": "before", "decode_review_pointer": "x"}
    assert D.job_digest(config) == D.job_digest(dict(config, runtime_assignment="after", decode_review_pointer="y"))
    assert D.job_digest(config) != D.job_digest(dict(config, source="b"))


def test_actual_cuda_uuid_format_canonicalization_keeps_authoritative_identity():
    assignment, pg, docker, cuda, host = runtime_fixture()
    now = datetime(2026, 10, 13, 21, 0, tzinfo=timezone.utc)
    bare = assignment['gpu_uuids'][0].removeprefix('GPU-')
    for value in (bare, 'GPU-' + bare, uuid.UUID(bare)):
        proof = D.attest_runtime(assignment, pg, docker, dict(cuda, uuids=[value]), host, now=now)
        assert proof['gpu_uuids'] == assignment['gpu_uuids']
    for value in ('GPU-invalid', None, '00000000-0000-0000-0000-000000000099'):
        with pytest.raises(D.Error, match='UUID|CUDA'):
            D.attest_runtime(assignment, pg, docker, dict(cuda, uuids=[value]), host, now=now)
    foreign_pg = copy.deepcopy(pg)
    foreign_pg[0]['gpu_uuid'] = bare
    with pytest.raises(D.Error, match='PG'):
        D.attest_runtime(assignment, foreign_pg, docker, dict(cuda, uuids=[bare]), host, now=now)


def test_unknown_cli_modes_and_missing_actual_states_refuse_before_torch(tmp_path):
    config = tmp_path / "config.json"
    write(config, {"schema": D.SCHEMA, "stage": "D4"})
    result = D.preflight(config)
    assert result["blockers"] and not result["candidate_preflight_passed"] and not result["production_ready"]
    with pytest.raises(D.Error, match="preflight blocked"):
        D.train(config, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
    with pytest.raises(SystemExit):
        D.main(["--config", str(config)])
    with pytest.raises(SystemExit):
        D.main(["preflight", "--config", str(config), "--unexpected"])


def test_h3_executing_source_helpers_and_sha_locator_are_pinned_before_model(tmp_path):
    from experiments.chimera_recovery import h3_native as H, train_media as T
    modules=(D.R,D.R.D,D.O,D.C,D.I,D.N,D)
    config={'helpers':{Path(module.__file__).name:D.I.pin(Path(module.__file__)) for module in modules}}
    rows=[{'lane':'h3'}]
    assert D.verify_helpers(config,tmp_path,[{'lane':'qimg'}]) is None
    with pytest.raises(D.Error,match='helper source'): D.verify_helpers(config,tmp_path,rows)
    config['helpers'].update({Path(module.__file__).name:D.I.pin(Path(module.__file__)) for module in (H,T)})
    alias=tmp_path/'aliases.json'; config['h3_source_aliases']=write(alias,{'a'*64:str((tmp_path/'mounted-original').resolve())})
    assert D.verify_helpers(config,tmp_path,rows)=={'a'*64:str((tmp_path/'mounted-original').resolve())}
    different=copy.deepcopy(config); different['helpers']['h3_native.py']['sha256']='f'*64
    with pytest.raises(D.Error,match='helper source'): D.verify_helpers(different,tmp_path,rows)
    alias.write_bytes(b'{}')
    with pytest.raises(D.I.Error): D.verify_helpers(config,tmp_path,rows)
    for bad in ({'invalid-sha':'/absolute/path'},{'a'*64:'relative/source'},['not','mapping']):
        config['h3_source_aliases']=write(alias,bad)
        with pytest.raises(D.Error,match='aliases'): D.verify_helpers(config,tmp_path,rows)


def test_guided_h3_dispatch_demands_full_source_guard_before_loss(tmp_path):
    """Dispatcher boundary fixture; real 32-target reconstruction is independently tested by H3 adapter."""
    from experiments.chimera_recovery import train_media as T
    helper=tmp_path/'h3_native.py'
    helper.write_text('''import hashlib
from pathlib import Path
def validate_batch(cpu,receipt,root,*,aliases=None):
    if not all(value.on_cpu for value in cpu.values()): raise ValueError("CPU source proof missing")
    if aliases != receipt["expected_aliases"]: raise ValueError("mounted source aliases differ")
    return {**receipt["proof"],"adapter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
''')
    class CPUProbe:
        on_cpu=False
        def detach(self): return self
        def cpu(self): self.on_cpu=True; return self
    aliases={'a'*64:str((tmp_path/'original').resolve())}
    callback=D.NativeBackward(None,tmp_path,{},h3_validator=D.I.pin(helper),config_root=tmp_path,h3_aliases=aliases)
    row={'lane':'h3','supervised_units':1}
    item={'route':'joint_av','source_id':'captured-source','recovery_group_id':'train-group'}
    proof={'verified':True,'source_id':'captured-source','recovery_group_id':'train-group',
           'supervised_units':1,'denominator_kind':'native_guided_pair_mean'}
    receipt={'proof':proof,'expected_aliases':aliases}
    assert callback.authenticate_guided_source(row,item,{'tensor':CPUProbe()},receipt,tmp_path)['sha256']==D.I.sha(helper)
    for change in ({'verified':False},{'source_id':'invented-target'},{'recovery_group_id':'held'},
                   {'supervised_units':32},{'denominator_kind':'scalar_mse'}):
        with pytest.raises(T.Error,match='ownership'):
            callback.authenticate_guided_source(row,item,{'tensor':CPUProbe()},dict(receipt,proof={**proof,**change}),tmp_path)
    with pytest.raises(D.Error,match='denominator'):
        callback.authenticate_guided_source(dict(row,supervised_units=32),item,{},receipt,tmp_path)
    with pytest.raises(D.Error,match='route'):
        callback.authenticate_guided_source(dict(row,lane='qimg'),item,{},receipt,tmp_path)
    missing=D.NativeBackward(None,tmp_path,{})
    with pytest.raises(T.Error,match='source-pinned'):
        missing.authenticate_guided_source(row,item,{},receipt,tmp_path)


def test_actual_native_x0_velocity_callback_and_window_weight_are_not_endpoint_mse(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    first, second = construct()["hybrid"], construct()["hybrid"]
    segment = M.tiny_segment(first.cfg, "h3", rows=2, seed=205)
    assert bool((segment["types"] == 3).all())
    target = torch.zeros(1, 2, M.TINY.FEATURE)
    lo, hi = first.bands[3]["emission"]
    mask = torch.ones_like(segment["types"], dtype=torch.float32)
    noisy_hi = first.bands[3]["feature"][1]
    receipt = dict(target_kind="x0", prediction_kind="x0", noisy_feature_bands={"3": [noisy_hi - (hi - lo), noisy_hi]})
    # True baseline uses the native forward and explicit canonical velocity;
    # the noisy x_t cancels algebraically when both target/prediction are x0.
    output = first.forward(segment)["emission"]
    expected = ((output[..., lo:hi].float() - target[..., lo:hi]) / segment["sigma"][..., None]).square().mean()
    units = 2 * (hi - lo)
    actual = O.flow_backward(M, first, segment, target, mask, receipt, units, 1.)
    scaled = O.flow_backward(M, second, segment, target, mask, receipt, units, .3)
    assert abs(actual - float(expected.detach())) < 1e-5 and abs(actual - scaled) < 1e-6
    groups = [*first.units.values(), *second.units.values()]
    assert groups
    a = {name: value for named in D.N.unit_parameters(first).items() for name, value in [named] if value.grad is not None}
    b = D.N.unit_parameters(second)
    assert a and any(bool(value.grad.any()) for value in a.values())
    for name, value in a.items():
        torch.testing.assert_close(b[name].grad, value.grad * .3, rtol=3e-4, atol=2e-5)
    with pytest.raises(O.Error, match="denominator"):
        O.flow_backward(M, first, segment, target, mask, receipt, units + 1, 1.)
    with pytest.raises(O.Error, match="explicit native"):
        O.flow_backward(M, first, segment, target, mask, dict(receipt, prediction_kind="raw"), units, 1.)


def test_actual_union_dispatch_quarter_review_then_exact_resume_full_answers(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    union = construct()
    rows, sources = replay(tmp_path, torch, M, union["hybrid"])
    engine = D.R.UnionEngine(union, rows, sources, tmp_path, accumulation=1)
    run = backward(torch, M)
    gate_calls = []
    stopped = D.execute(engine, run, tmp_path / "before", gate=lambda: gate_calls.append(engine.step), export=False)
    assert stopped["status"] == "awaiting_decode_review" and stopped["step"] == 1 and gate_calls == [0]
    assert engine.cursor == 1 and not (tmp_path / "before" / "checkpoint-final").exists()
    checkpoint = Path(stopped["checkpoint"])
    answer = tmp_path / "fixture-full-answer.json"
    write(answer, {"fixture": True, "answer": "Synthetic mechanical fixture; no actual model quality judgment"})
    review = tmp_path / "fixture-review.json"
    review_ref = write(review, dict(schema="chimera_recovery.d4_decode_review.v1", run_identity_sha256=D.I.digest(engine.identity),
        trained_step=1, meaningful_prompt_conditioned_content=True, verdict="PASS", reviewer="operator fixture only",
        checkpoint_receipt_sha256=D.I.sha(checkpoint / "receipt.json"), full_answers=D.I.pin(answer), outcomes=D.I.pin(answer),
        evaluation_protocol=D.I.pin(answer), baseline_answers=D.I.pin(answer)))
    pointer = tmp_path / "review-pointer.json"
    write(pointer, review_ref)
    restored = D.R.UnionEngine(construct(), rows, sources, tmp_path, accumulation=1)
    result = D.execute(restored, run, tmp_path / "after", gate=lambda: gate_calls.append(restored.step),
                       review_pointer=pointer, resume=checkpoint, export=False)
    assert result["training_complete"] and restored.cursor == 4 and restored.step == 4
    assert not result["accepted"] and not result["registered"] and gate_calls == [0, 1, 2, 3]
    reference = D.R.UnionEngine(construct(), rows, sources, tmp_path, accumulation=1)
    while reference.cursor < len(rows):
        reference.window(run)
    assert all(torch.equal(value, reference.census[name]) for name, value in restored.census.items())
    # Changing the durable safety policy cannot quietly erase the review/skip history.
    saved = tmp_path / "after" / "checkpoint-final"
    write(saved / "dispatch_policy.json", dict(consecutive_skips=0, window=[], attempts=0, decode_review=None))
    with pytest.raises(D.Error, match="missing or changed"):
        D.restored_policy(restored, saved)


def test_actual_guided_callback_preserves_producer_raw_signs_and_pair_mean(tmp_path):
    torch, M, _, construct = fixture(tmp_path, native_features=True)
    hybrid = construct()["hybrid"]
    segment = M.tiny_segment(hybrid.cfg, "h3", rows=4, seed=215)
    target = torch.zeros(1, 4, M.TINY.FEATURE)
    batch = dict(input_ids=segment["ids"], native_media_features=segment["features"],
        native_media_targets=target, native_media_type_ids=segment["types"], native_media_positions=segment["positions"],
        native_media_indices=segment["indices"], native_media_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        native_media_noise_levels=segment["sigma"], native_media_loss_mask=torch.ones(1, 4),
        charted_source_positions=segment["source_positions"]["h3"])
    allocation = {str(kind): [hybrid.bands[kind]["feature"][1] - (hybrid.bands[kind]["emission"][1] - hybrid.bands[kind]["emission"][0]),
                              hybrid.bands[kind]["feature"][1]] for kind in (2, 3)}
    receipt = dict(schema="chimera_recovery.native_flow_cache.v1", fixture=True, sealed=True, all_finite=True,
        all_structural_ok=True, guided_teacher=True, no_clean_target_in_inputs=True,
        cache_sha256="a" * 64, corpus_sha256="b" * 64, teacher_sha256="c" * 64, recovery_amendment_sha256="d" * 64,
        target_kind="canonical_velocity", prediction_kind="raw", input_layout="feature", target_layout="emission",
        route="joint_av", noisy_feature_bands=allocation, noisy_feature_bands_sha256=D.I.digest(allocation),
        parent_signs={"2": -1., "3": 1.}, av_alignment={"same_origin": True, "source_media_sha256": "e" * 64, "start_seconds": 0.})
    item = dict(route="joint_av", parent_signs=receipt["parent_signs"])
    output = hybrid.forward(segment)["emission"]
    expected = []
    for kind in (2, 3):
        lo, hi = hybrid.bands[kind]["emission"]
        expected.append(output[..., lo:hi][segment["types"] == kind].float().square().mean())
    expected = float(torch.stack(expected).mean().detach())
    actual = O.guided_backward(M, hybrid, batch, item, receipt, 1, .4)
    assert abs(actual - expected) < 1e-5 and receipt["prediction_kind"] == "raw"
    with pytest.raises(O.Error, match="signs differ"):
        O.guided_backward(M, hybrid, batch, dict(item, parent_signs={"2": 1., "3": 1.}), receipt, 1, 1.)


def test_actual_nonfinite_dispatch_retries_same_window_and_halts_without_updates(tmp_path):
    torch, M, _, construct = fixture(tmp_path)
    union = construct()
    rows, sources = replay(tmp_path, torch, M, union["hybrid"])
    engine = D.R.UnionEngine(union, rows, sources, tmp_path, accumulation=2)
    before = {name: value.detach().clone() for name, value in engine.census.items()}
    calls = []
    with pytest.raises(D.Error, match="persistent nonfinite"):
        D.execute(engine, lambda *_: float("nan"), tmp_path / "skips", gate=lambda: calls.append("should never apply"), export=False)
    assert engine.step == engine.cursor == 0 and not calls and not engine.optimizer.state
    assert all(torch.equal(value, before[name]) for name, value in engine.census.items())
    checkpoint = tmp_path / "skips" / "checkpoint-skip-halt-00000000"
    restarted = D.R.UnionEngine(construct(), rows, sources, tmp_path, accumulation=2)
    with pytest.raises(D.Error, match="cannot erase"):
        D.execute(restarted, backward(torch, M), tmp_path / "cannot-reset", gate=lambda: None, resume=checkpoint, export=False)
    assert not (tmp_path / "cannot-reset").exists()
