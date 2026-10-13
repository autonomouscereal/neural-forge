"""D4 regression fixtures: held leakage, optimizer aliases and state overwrite.

All files/tensor descriptors here are explicitly tiny synthetic contracts;
passing these tests does not prove native model/cache/merge parity.
"""
import copy
import json

import pytest

from experiments.chimera_recovery import consolidation as d4
from experiments.chimera_recovery.corpus import canonical, digest


def row(lane, number=0, count=4096):
    identity = dict(input_sha256=digest(f"x-{number}"), target_sha256=digest(f"y-{number}"), conditioning_sha256=digest(f"c-{number}"))
    return dict(id=f"{lane}-{number}", lane=lane, recovery_group_id=f"g{number}", split="train", held=False,
                sealed_target=True, pair_id=digest(identity), real_tokens=count, **identity,
                corpus_sha256="1" * 64, teacher_manifest_sha256="2" * 64, target_shard_sha256="3" * 64)


def optimizer_groups():
    return [dict(family=family, previous_base_lr="0.0001", lr="0.00003",
                 parameters=[dict(parameter_id=f"physical-{i}", numel=12)]) for i, family in enumerate(sorted(d4.OPTIMIZER_FAMILIES))]


def states():
    result, previous, bank = [], None, digest("initial-r16")
    for role in d4.STAGE_ORDER:
        item = dict(stage_role=role, state_sha256=digest(role), parent_state_sha256=previous, semantics="full_replacement")
        if role.startswith("D3"):
            item.update(shared_lora_semantics="unchanged" if role == "D3a" else "full_replacement",
                        shared_lora_parent_sha256=bank, shared_lora_state_sha256=bank if role == "D3a" else digest("bank-" + role))
            bank = item["shared_lora_state_sha256"]
        result.append(item)
        previous = item["state_sha256"]
    return result


def op(kind="lora", bank="D2.k3_lora", target="layers.0.ffn.down.weight", axis=1, start=4352, stop=16640, **extra):
    return dict(kind=kind, bank=bank, target=target, axis=axis, start=start, stop=stop,
                compute_dtype="float32", cast="defer_until_target_complete", adapter_state_id="layer0.k3", **extra)


def test_replay_preserves_whole_rows_and_exposes_overshoot_not_exact_budget():
    report = d4.select_replay("long_text", [row("long_text", n, 32768) for n in range(20)])
    assert len(report["selected_ids"]) == 16
    assert report["selected_amount"] == 524288
    assert report["overshoot"] == 24288
    assert report["deficit"] == 0
    assert "whole_row_token_budget_overshoot_requires_dated_amendment" in report["blockers"]
    assert report["production_ready"] is False


@pytest.mark.parametrize("change,error", [
    ({"split": "test"}, "heldout"), ({"held": None}, "heldout"),
    ({"sealed_target": False}, "sealed_targets"), ({"target_shard_sha256": "unsealed"}, "source_pin"),
    ({"real_tokens": True}, "token_count"), ({"lane": "short_text"}, "matching_existing"),
])
def test_replay_rejects_unproven_or_held_inputs(change, error):
    item = row("long_text")
    item.update(change)
    with pytest.raises(d4.Error, match=error):
        d4.select_replay("long_text", [item])


def test_same_actual_pair_cannot_repeat_but_genuine_distinct_sigma_pairs_can_count():
    first, second = row("ace", 0), row("ace", 1)
    second["pair_id"] = first["pair_id"]
    with pytest.raises(d4.Error, match="duplicate_native_pair"):
        d4.select_replay("ace", [first, second])
    with pytest.raises(d4.Error, match="heldout"):
        d4.select_replay("ace", [first], {first["recovery_group_id"]})
    report = d4.select_replay("ace", [first])
    assert report["deficit"] == 299 and report["unit"] == "pairs"
    second = row("ace", 1)
    second["recovery_group_id"] = first["recovery_group_id"]
    assert d4.select_replay("ace", [first, second])["selected_amount"] == 2
    second["pair_id"] = "invented-alias"
    with pytest.raises(d4.Error, match="must_bind_actual_input"):
        d4.select_replay("ace", [second])
    second["input_sha256"] = second["target_sha256"]
    second["pair_id"] = digest({key: second[key] for key in ("input_sha256", "target_sha256", "conditioning_sha256")})
    with pytest.raises(d4.Error, match="target_cannot_be"):
        d4.select_replay("ace", [second])


def test_optimizer_physical_alias_and_rate_vs_prior_base_rejected():
    groups = optimizer_groups()
    assert d4.optimizer_union(groups)["parameters"] == 9
    groups[1]["parameters"][0]["parameter_id"] = groups[0]["parameters"][0]["parameter_id"]
    with pytest.raises(d4.Error, match="duplicate_optimizer_owners"):
        d4.optimizer_union(groups)
    groups = optimizer_groups()
    groups[0]["lr"] = "0.000003"  # .3 of previous terminal .1-LR is not .3 of base LR.
    with pytest.raises(d4.Error, match="previous_base_lr"):
        d4.optimizer_union(groups)
    groups[0]["lr"] = "NaN"
    with pytest.raises(d4.Error, match="previous_base_lr"):
        d4.optimizer_union(groups)
    with pytest.raises(d4.Error, match="missing_D2_or_D3"):
        d4.optimizer_union(optimizer_groups()[:-1])


def test_full_shared_banks_require_actual_sequential_carry_not_sibling_addition():
    chain = states()
    assert d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))["stages"] == 9
    assert d4.STAGE_ORDER[-1] == "D3d_audio_ce"
    chain[3]["parent_state_sha256"] = chain[1]["state_sha256"]
    with pytest.raises(d4.Error, match="sequential_lineage"):
        d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))
    chain = states()
    chain[3]["shared_lora_parent_sha256"] = digest("initial-r16")
    with pytest.raises(d4.Error, match="continuation_not_proven"):
        d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))
    chain[3]["shared_lora_semantics"] = "add"
    with pytest.raises(d4.Error, match="cannot_be_summed"):
        d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))


def test_unit_refit_cannot_reset_bank_and_stage_order_cannot_hide_duplicate():
    chain = states()
    chain[1]["shared_lora_state_sha256"] = digest("reset-bank")
    with pytest.raises(d4.Error, match="unit_only_refit"):
        d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))
    chain = states()
    chain[2], chain[3] = chain[3], chain[2]
    with pytest.raises(d4.Error, match="order_once_each"):
        d4.sequential_states(chain, initial_shared_bank_sha256=digest("initial-r16"))


def test_k3_gain_once_after_all_lora_aliases_and_cast_once():
    shapes = {"layers.0.ffn.down.weight": [5120, 16640]}
    operations = [op(), op(kind="k3_gain", bank="D2.gain")]
    assert d4.merge_plan(operations, shapes)["gain_folds"] == 1
    # This hand-computed noncommuting example explains why the rejected order
    # matters: (2 + 3)*4=20; applying gain before the adapter gives 2*4+3=11.
    assert (2 + 3) * 4 == 20 and 2 * 4 + 3 == 11
    with pytest.raises(d4.Error, match="gain_must_fold_once_after"):
        d4.merge_plan(operations[::-1], shapes)
    extra = op(bank="D3.shared_lora", target="alias-down")
    extra["adapter_state_id"] = "layer0.shared"
    with pytest.raises(d4.Error, match="after_gain"):
        d4.merge_plan(operations + [extra], shapes, {"alias-down": "layers.0.ffn.down.weight"})
    with pytest.raises(d4.Error, match="gain_must_fold_once_after"):
        d4.merge_plan(operations + [operations[-1]], shapes)
    operations[0]["cast"] = "bf16"
    with pytest.raises(d4.Error, match="one_final_cast"):
        d4.merge_plan(operations, shapes)


def test_music_protected_rows_and_native_ownership_and_overwrites():
    shapes = {"embed.weight": [317952, 5120]}
    music = op(kind="replace_full_native_state", bank="D3.music_embed", target="embed.weight", axis=0, start=252416, stop=316416)
    owners = {"D3.music_embed": [dict(target="embed.weight", axis=0, start=252416, stop=316416)]}
    assert d4.merge_plan([music], shapes, protected_row_targets=["embed.weight"], owner_regions=owners)["targets"] == 1
    reserved = copy.deepcopy(music)
    reserved["stop"] = 317952
    with pytest.raises(d4.Error, match="reserved_rows"):
        d4.merge_plan([reserved], shapes)
    wrong_owner = copy.deepcopy(music)
    wrong_owner["bank"] = "D3.unit"
    with pytest.raises(d4.Error, match="protected_vocab"):
        d4.merge_plan([wrong_owner], shapes, protected_row_targets=["embed.weight"])
    with pytest.raises(d4.Error, match="outside_pinned_owner"):
        d4.merge_plan([music], shapes, owner_regions={})
    with pytest.raises(d4.Error, match="overwrite_prior_merge_write"):
        d4.merge_plan([music, music], shapes)
    bias = op(kind="replace_full_native_state", bank="D3.unit", target="unit.bias", axis=0, start=0, stop=4)
    assert d4.merge_plan([bias], {"unit.bias": [4]})["targets"] == 1
    with pytest.raises(d4.Error, match="every_owned_family"):
        d4.merge_plan([music], shapes, owner_regions=owners, require_complete=True)


def test_frozen_pin_matches_campaign_literal_marker_and_lf_policy(tmp_path, monkeypatch):
    import hashlib
    raw = b'The literal "## 1. Scope" and "## Amendments" are described here.\n\n## 1. Scope\nbody\n\n## Amendments\nnew\n'
    frozen = raw[raw.find(b"## 1. Scope"):raw.find(b"\n## Amendments") + 1]
    monkeypatch.setattr(d4, "FROZEN_BYTES", len(frozen))
    monkeypatch.setattr(d4, "FROZEN_SHA", hashlib.sha256(frozen).hexdigest())
    path = tmp_path / "prereg.md"
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert d4.preregistration(path)["frozen_body_sha256"] == hashlib.sha256(frozen).hexdigest()
    path.write_bytes(raw.replace(b"body", b"edited"))
    with pytest.raises(d4.Error, match="body_changed"):
        d4.preregistration(path)


def files(tmp_path, monkeypatch):
    frozen = b"## 1. Scope\nsynthetic test only\n"
    monkeypatch.setattr(d4, "FROZEN_BYTES", len(frozen))
    import hashlib
    monkeypatch.setattr(d4, "FROZEN_SHA", hashlib.sha256(frozen).hexdigest())
    prereg = tmp_path / "prereg.md"
    prereg.write_bytes(frozen + b"## Amendments\n")
    initial = tmp_path / "initial-bank.bin"
    initial.write_bytes(b"synthetic initial shared bank")
    chain, previous, bank = states(), None, d4.file_sha256(initial)
    for item in chain:
        state = tmp_path / (item["stage_role"] + ".bin")
        state.write_bytes(item["stage_role"].encode())
        item.update(state_file=d4.pin(state), state_sha256=d4.file_sha256(state), parent_state_sha256=previous)
        if item["stage_role"].startswith("D3"):
            if item["stage_role"] == "D3a":
                item.update(shared_lora_state_sha256=bank, shared_lora_parent_sha256=bank)
            else:
                shared = tmp_path / (item["stage_role"] + "-bank.bin")
                shared.write_bytes((item["stage_role"] + " bank").encode())
                item.update(shared_lora_file=d4.pin(shared), shared_lora_state_sha256=d4.file_sha256(shared), shared_lora_parent_sha256=bank)
                bank = item["shared_lora_state_sha256"]
        previous = item["state_sha256"]
    catalog = [row(lane, i, quota) for i, (lane, quota) in enumerate(d4.QUOTAS.items())]
    operations = [op(), op(kind="k3_gain", bank="D2.gain")]
    shapes = {"layers.0.ffn.down.weight": [5120, 16640]}
    for bank, target, shape in (("D2.lora", "layers.0.ffn.up.weight", [33280, 5120]),
                                 ("D3.shared_lora", "layers.1.ffn.up.weight", [33280, 5120]),
                                 ("D3.unit", "unit.bias", [4]), ("D3.emission", "flow_head.weight", [2560, 5120]),
                                 ("D3.bridge", "input_bridge.weight", [30720, 5120]),
                                 ("D3.music_embed", "embed.weight", [317952, 5120]), ("D3.music_head", "head.weight", [317952, 5120])):
        shapes[target] = shape
        music = bank in {"D3.music_embed", "D3.music_head"}
        item = op(kind="lora" if bank in {"D2.lora", "D3.shared_lora"} else "replace_full_native_state",
                  bank=bank, target=target, axis=0, start=252416 if music else 0, stop=316416 if music else shape[0])
        item["adapter_state_id"] = bank + ".site"
        operations.append(item)
    owners = {bank: [{key: item[key] for key in ("target", "axis", "start", "stop")} for item in operations if item["bank"] == bank] for bank in d4.OPTIMIZER_FAMILIES}
    values = dict(stage_states=chain, optimizer_groups=optimizer_groups(), merge_operations=operations,
                  tensor_inventory=dict(shapes=shapes, protected_vocab_targets=["embed.weight", "head.weight"]), owner_regions=owners)
    refs = dict(preregistration=d4.pin(prereg), initial_shared_bank=d4.pin(initial))
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.write_bytes(canonical(value))
        refs[name] = d4.pin(path)
    path = tmp_path / "catalog.jsonl"
    path.write_bytes(b"".join(canonical(item) + b"\n" for item in catalog))
    refs["replay_catalog"] = d4.pin(path)
    job = tmp_path / "job.json"
    job.write_bytes(canonical(dict(schema="chimera_recovery.d4_validation_plan.v1", sources=refs, heldout_groups=[])))
    return job, values, catalog


def test_executable_pinned_job_never_certifies_production_and_catches_changed_checkpoint(tmp_path, monkeypatch):
    job, values, _ = files(tmp_path, monkeypatch)
    report = d4.validate_job(job)
    assert report["production_ready"] is False and report["weights_changed"] is False
    assert report["lineage"]["stages"] == 9
    (tmp_path / "D3b.ace.bin").write_bytes(b"changed checkpoint")
    with pytest.raises(d4.Error, match="pinned_source_bytes_changed"):
        d4.validate_job(job)


def test_heldout_in_unused_suffix_cannot_hide_behind_replay_quota(tmp_path, monkeypatch):
    job, _, catalog = files(tmp_path, monkeypatch)
    held = row("long_text", 50)
    held["split"] = "test"
    path = tmp_path / "catalog.jsonl"
    path.write_bytes(b"".join(canonical(item) + b"\n" for item in catalog + [held]))
    config = json.loads(job.read_text())
    config["sources"]["replay_catalog"] = d4.pin(path)
    job.write_bytes(canonical(config))
    with pytest.raises(d4.Error, match="heldout.*catalog"):
        d4.validate_job(job)


def test_complete_merge_cannot_omit_an_owned_trained_tensor(tmp_path, monkeypatch):
    job, values, _ = files(tmp_path, monkeypatch)
    operations = values["merge_operations"][:-1]
    with pytest.raises(d4.Error, match="exact_pinned_owner_regions"):
        d4.merge_plan(operations, values["tensor_inventory"]["shapes"], owner_regions=values["owner_regions"], require_complete=True)


def prepared_inputs(tmp_path, monkeypatch):
    files(tmp_path, monkeypatch)
    stage = tmp_path / "stage.py"
    stage.write_text("# synthetic descriptor only, never executed\n")
    inventory, jobs = tmp_path / "inventory.json", tmp_path / "text_jobs.json"
    inventory.write_bytes(canonical(dict(manifests={})))
    jobs.write_bytes(canonical(dict(sources=[dict(split="train", shape=[1, 4096], rows=1, real_tokens=88, supervised_tokens=10, name="short_train")])))
    rows = tmp_path / "admission.jsonl"
    rows.write_bytes(canonical(dict(kind="qimg", split="train", train_admitted=True, recovery_group_id="g0")) + b"\n")
    admission = tmp_path / "admission.json"
    body = dict(outputs=dict(admission=d4.pin(rows)))
    admission.write_bytes(canonical({**body, "manifest_sha256": digest(body)}))
    native = tmp_path / "native.json"
    value = dict(schema="chimera_recovery.native_batches.v1", sealed=True, split="train", complete_source_traversal=True,
                 heldout_groups_excluded=True, corpus_sha256=d4.file_sha256(admission),
                 batches=[dict(id="fixture", recovery_group_id="g0", route="image_generate",
                               tensors=dict(path="fixture.safetensors", sha256="1" * 64), teacher_receipt=dict(path="fixture.json", sha256="2" * 64))])
    native.write_bytes(canonical(value))
    return [tmp_path / "prereg.md", inventory, jobs, admission, [stage], tmp_path / "prepared"], native, value


def test_preparation_counts_groups_and_preserves_old_artifact_without_fake_byte_proof(tmp_path, monkeypatch):
    args, native, _ = prepared_inputs(tmp_path, monkeypatch)
    report = d4.prepare(*args, native_manifests=[("qimg", native)])
    assert report["coverage"]["qimg"]["native_sealed_manifest_batches"] == 1
    assert report["coverage"]["qimg"]["native_manifest_distinct_groups"] == 1
    assert report["coverage"]["qimg"]["verified_native_pairs"] is None
    assert report["coverage"]["qimg"]["tensor_bytes_verified_in_this_local_audit"] is False
    assert report["text_input_totals"]["assistant_tokens"] == 10
    assert (args[-1] / "report.json").is_file()
    with pytest.raises(d4.Error, match="retain_prior"):
        d4.prepare(*args, native_manifests=[("qimg", native)])


@pytest.mark.parametrize("change,error", [
    ({"route": "music_generate"}, "route_does_not_match"),
    ({"recovery_group_id": "held-g1"}, "not_current_train_admitted"),
    ({"tensors": dict(sha256="1" * 64)}, "file_group_pins_missing"),
])
def test_sealed_manifest_label_cannot_override_route_or_train_group_or_file_pin(tmp_path, monkeypatch, change, error):
    args, native, value = prepared_inputs(tmp_path, monkeypatch)
    value["batches"][0].update(change)
    native.write_bytes(canonical(value))
    with pytest.raises(d4.Error, match=error):
        d4.prepare(*args, native_manifests=[("qimg", native)])
    assert not args[-1].exists()
