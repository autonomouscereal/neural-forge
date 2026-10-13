"""Pinned D4 candidate dispatcher; explicit preflight and train modes only.

Uses the genuine union engine and existing native objective validators. It does
not allocate GPUs, execute old commands, accept stages, publish model weights or
invent missing caches. Production requires a root-produced live PG/Docker proof
and actual accepted predecessor states. Checkpoint publication remains separate.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from . import consolidation as C, consolidation_runtime as R, consolidation_objectives as O

I, N, Error = R.I, R.N, R.Error
SCHEMA = "chimera_recovery.d4_dispatch_config.v1"
REPLAY = "chimera_recovery.d4_replay.v1"
CHECKPOINT = "chimera_recovery.d4_dispatch_checkpoint.v1"
SAFETY = dict(checkpoint_seconds=1800, keep_safety=2, consecutive_skips=8,
              skip_window=100, skip_fraction=.2, review_fraction=.25)
LANES = {"short_text": O.TEXT, "long_text": O.TEXT, "k3": O.TEXT,
         "vision": O.VISION, "ace": O.GUIDED, "qimg": O.GUIDED,
         "hy3d": O.VOLUME, "h3": O.GUIDED, "audio_ce": O.AUDIO, "speech_flow": O.SPEECH}
SOURCES = {O.TEXT: "chimera_recovery.text_replay_batches.v1", O.GUIDED: "chimera_recovery.native_batches.v1",
           O.MUSIC: "chimera_recovery.music_batches.v1", O.VISION: "chimera_recovery.vision_batches.v1",
           O.AUDIO: "chimera_recovery.audio_batches.v1", O.SPEECH: "chimera_recovery.speech_flow_batches.v1"}


def read(ref, root):
    path = I.pinned(ref, root)
    return json.loads(path.read_text(encoding="utf-8")), path


def sealed(value):
    if value.get("seal_sha256") != I.digest({k: v for k, v in value.items() if k != "seal_sha256"}):
        raise Error("D4 manifest seal differs")


def replay_catalog(entry, root, amendment, *, compatible_amendments=()):
    """Bind every selected row to one actual sealed manifest batch and receipt.

    Existing cache validators inspect tensor contents immediately before their
    backward. This first traversal rejects fabricated membership, changed files,
    holdout groups, missing quotas and admitted-source omissions before a model
    is opened. Token budgets and assistant/scalar denominators remain distinct.
    """
    value, path = read(entry, root)
    sealed(value)
    if (value.get("schema") != REPLAY or value.get("sealed") is not True
            or value.get("heldout_groups_excluded") is not True or value.get("split") != "train"
            or value.get("amendment_sha256") != amendment):
        raise Error("D4 replay needs a sealed train-only amended catalog")
    sources = dict(value.get("sources", {}), replay=I.pin(path))
    exclusion, _ = read(value["exclusions"], path.parent)
    if exclusion.get("connected_groups_verified") is not True:
        raise Error("D4 global connected-group exclusion proof missing")
    excluded = set(exclusion["excluded_recovery_group_ids"])
    sources["exclusions"] = value["exclusions"]
    rows, caches, seen_pairs = copy.deepcopy(value.get("rows", [])), {}, set()
    audio_index, _ = read(value["audio_speech_index"], path.parent)
    if (audio_index.get("schema") != "chimera_recovery.audio_speech_replay_index.v1"
            or any(audio_index.get(key) is not True for key in ("sealed", "passed", "verified", "complete_source_traversal", "source_reconstruction_bitwise_equal"))
            or audio_index.get("amendment_sha256") != amendment):
        raise Error("D4 audio/speech replay requires the fully reconstructed admitted native index")
    sources["audio_speech_index"] = value["audio_speech_index"]
    indexed = {row["id"]: row for row in audio_index["entries"]}
    if len(indexed) != len(audio_index["entries"]):
        raise Error("D4 audited audio/speech IDs duplicate")
    R.replay_identity(rows, sources, path.parent)
    counts = {lane: 0 for lane in LANES}
    admitted_sources = {"audio_ce": set(), "speech_flow": set()}
    for row in rows:
        lane, objective = row.get("lane"), row["objective"]
        if lane not in LANES or objective != LANES[lane] or row["recovery_group_id"] in excluded:
            raise Error("D4 replay lane/objective or global heldout exclusion differs")
        # Music token replay is an extra explicit objective, never silently
        # counted as one of the five frozen 300 flow/vision pair budgets.
        if objective not in SOURCES:
            raise Error("D4 unsupported cache/objective; new volume requires its own audited contract")
        ref = row["source_manifest"]
        key = ref["sha256"]
        if key not in caches:
            data, manifest_path = read(ref, path.parent)
            if (data.get("schema") != SOURCES[objective] or data.get("sealed") is not True
                    or data.get("split") != "train" or data.get("heldout_groups_excluded") is not True):
                raise Error("D4 source cache must be sealed train-only")
            items = data.get("batches", [])
            if not items or len({item.get("id") for item in items}) != len(items):
                raise Error("D4 source batch IDs missing or duplicate")
            caches[key] = data, manifest_path, {item["id"]: item for item in items}
            sources["cache:" + key] = I.pin(manifest_path)
        data, manifest_path, items = caches[key]
        item = items.get(row["source_batch_id"])
        source_group = row.get("source_recovery_group_id", row["recovery_group_id"])
        if (not item or item.get("recovery_group_id") != source_group
                or I.pinned(item["tensors"], manifest_path.parent) != I.pinned(row["input"], path.parent)
                or I.pinned(item["teacher_receipt"], manifest_path.parent) != I.pinned(row["target"], path.parent)
                or item.get("storage") != row.get("storage")):
            raise Error("D4 row differs from authentic source batch membership")
        receipt, _ = read(row["target"], manifest_path.parent)
        if (receipt.get("sealed") is not True or receipt.get("all_finite") is not True
                or receipt.get("split") != "train" or receipt.get("held") is not False
                or receipt.get("recovery_group_id") != source_group
                or receipt.get("tensors_sha256") != row["input"]["sha256"]
                or receipt.get("amendment_sha256", receipt.get("recovery_amendment_sha256")) not in {amendment, *compatible_amendments}
                or row.get("source_amendment_sha256") != receipt.get("amendment_sha256", receipt.get("recovery_amendment_sha256"))):
            raise Error("D4 native receipt does not bind exact row/tensor/group/amendment")
        if lane in {"short_text", "long_text", "k3"}:
            amount = row.get("real_tokens")
            if type(amount) is not int or amount < 1 or receipt.get("real_tokens") != amount:
                raise Error("D4 real token budget must match actual source receipt")
        else:
            amount = 1
            pair = (row["input"]["sha256"], row["target"]["sha256"])
            if pair in seen_pairs:
                raise Error("D4 duplicate physical native pair cannot fill replay quota")
            seen_pairs.add(pair)
        counts[lane] += amount
        if lane in admitted_sources:
            audited = indexed.get(row["source_batch_id"])
            if (not audited or audited["objective"] != objective or audited["union_recovery_group_id"] != row["recovery_group_id"]
                    or audited["recovery_group_id"] != source_group or audited["supervised_units"] != row["supervised_units"]
                    or audited["source_manifest"] != row["source_manifest"] or audited.get("storage") != row.get("storage")
                    or audited["tensors"] != item["tensors"] or audited["teacher_receipt"] != item["teacher_receipt"]):
                raise Error("D4 row differs from independently reconstructed audio/speech index")
            # Require every eligible source batch exactly once; the source seal
            # and existing validator bind source identity/teacher conditioning.
            if row["source_batch_id"] in admitted_sources[lane]:
                raise Error("D4 repeated audio/speech source batch")
            admitted_sources[lane].add(row["source_batch_id"])
    allowed = value.get("whole_row_token_overshoot", {})
    for lane, quota in C.QUOTAS.items():
        excess = counts[lane] - quota
        if excess < 0 or (excess and (lane not in {"short_text", "long_text", "k3"} or allowed.get(lane) != excess)):
            raise Error("D4 frozen replay quota missing or unamended whole-row excess: " + lane)
    if allowed:
        declaration, _ = read(value["budget_amendment"], path.parent)
        if (declaration.get("selected") is not True or declaration.get("dated") != "2026-10-13"
                or declaration.get("preregistration_amendment_sha256") != amendment
                or declaration.get("whole_row_token_overshoot") != allowed):
            raise Error("D4 whole-row overshoot needs an exact selected dated amendment")
        sources["budget_amendment"] = value["budget_amendment"]
    for lane in admitted_sources:
        full_ids = set()
        full_ids = {key for key, row in indexed.items() if row["objective"] == LANES[lane]}
        if not full_ids or admitted_sources[lane] != full_ids:
            raise Error("D4 selected amendment requires one replay of every admitted " + lane + " batch")
    sources = {key: I.pin(I.pinned(ref, path.parent)) for key, ref in sources.items()}
    return dict(rows=rows, sources=sources, root=str(path.parent), counts=counts,
                measured_supervised_units={objective: sum(row["supervised_units"] for row in rows if row["objective"] == objective)
                                           for objective in {row["objective"] for row in rows}})


class NativeBackward:
    """Dispatch trusted validators and native losses; never execute dataset code."""
    def __init__(self, module, root, objective_config):
        self.module, self.root, self.objective_config = module, Path(root), objective_config

    def text_batch(self, row, batch, receipt):
        import torch
        from . import train_text as X, training as Q
        T = self.module.trainer_module()
        data_path = I.pinned(receipt["text_data_config"], self.root)
        spec = json.loads(data_path.read_text(encoding="utf-8"))
        if (not spec.get("sources") or any(item.get("kind", "text") not in {"text", "k3"}
                or item.get("xtok_unverified_ok") is True for item in spec["sources"])):
            raise Error("D4 text must use authenticated Qwen/Kimi sources without unverified cross-token alignment")
        sources, _ = T.load_data_config(data_path)
        matches = [i for i, source in enumerate(sources) if source.name == receipt["text_source_name"]]
        if len(matches) != 1 or receipt["source_row_id"] not in sources[matches[0]].row_ids:
            raise Error("D4 text source row membership differs")
        source = sources[matches[0]]
        if (row["lane"] == "k3") != (source.kind == "k3"):
            raise Error("D4 Kimi replay cannot be substituted with Qwen-only targets")
        verifier = T.load_peer_module("qtext_teacher_cache")
        I.pinned(receipt["raw_cache_verifier"], self.root)
        if I.sha(Path(verifier.__file__)) != receipt["raw_cache_verifier"]["sha256"]:
            raise Error("D4 actual raw text verifier differs from sealed recipe")
        for entry in source.cache.manifest["shards"]:
            if Path(entry["shard"]).name != entry["shard"] or verifier.verify_shard(source.cache.dir / entry["shard"]).get("ok") is not True:
                raise Error("D4 original teacher raw bytes/contents changed")
        rid = receipt["source_row_id"]
        original = source.cache.row(rid)
        config = T.ObjectiveConfig(**self.objective_config)
        ids, targets, stats = X.cropped_batch(sources, [(matches[0], rid)], config, T.build_batch)
        assistant = original["assistant_mask"][:ids.shape[1]].unsqueeze(0)
        targets = Q.mask_streamed_targets(targets, assistant, torch.tensor([stats["real_tokens"]]))
        owned = torch.zeros_like(assistant)
        owned[:, :-1] = assistant[:, 1:]
        expected = {"input_ids": ids, "assistant_prediction_mask": owned, **{"target." + key: value for key, value in targets.items()}}
        if (receipt.get("denominator_kind") != "assistant_next_token_positions"
                or stats["real_tokens"] != row["real_tokens"] or stats["supervised_positions"] != row["supervised_units"]
                or set(batch) != set(expected) or any(not torch.equal(value.cpu(), expected[key].cpu()) for key, value in batch.items())):
            raise Error("D4 text input/masks/targets differ from actual sealed teacher source and recipe")

    def __call__(self, hybrid, row, weight):
        from . import train_media as T, train_stages as S
        import torch
        receipt_path = I.pinned(row["target"], self.root)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest_path = I.pinned(row["source_manifest"], self.root)
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = next(item for item in source["batches"] if item["id"] == row["source_batch_id"])
        # Canonical rows carry absolute pins, so the runtime's source guard can
        # hash every selected file independently of the old cache namespace.
        item = {**item, "tensors": row["input"], "teacher_receipt": row["target"]}
        batch = T.load_native_batch(item, manifest_path.parent, "cpu")
        objective, units = row["objective"], row["supervised_units"]
        if objective == O.TEXT:
            self.text_batch(row, batch, receipt)
            batch = {key: value.to(hybrid.device) for key, value in batch.items()}
            return O.text_backward(self.module, hybrid, batch, self.objective_config, units, weight)
        if objective == O.VISION:
            from . import vision_cache as V
            V.validate_vision_batch(batch, receipt, manifest_path.parent)
        elif objective == O.AUDIO:
            from . import audio_cache as A
            A.validate_batch(batch, receipt, manifest_path.parent)
        elif objective == O.SPEECH:
            from . import speech_flow as F
            F.validate_batch(batch, receipt, manifest_path.parent)
        elif objective == O.MUSIC:
            S.validate_music_batch(batch, receipt)
        elif objective != O.GUIDED:
            raise Error("D4 objective lacks an audited native callback")
        batch = {key: value.to(hybrid.device) for key, value in batch.items()}
        if objective == O.GUIDED:
            return O.guided_backward(self.module, hybrid, batch, item, receipt, units, weight)
        if objective == O.MUSIC:
            return O.music_backward(self.module, hybrid, batch, receipt, units, weight)
        segment = T.make_segment(batch, media_placeholder=hybrid.media_placeholder)
        if objective == O.VISION:
            targets = {key[7:]: value for key, value in batch.items() if key.startswith("answer.")}
            return O.vision_backward(self.module, hybrid, segment, targets, batch["teacher_features"], receipt, units, weight)
        if objective == O.AUDIO:
            return O.audio_backward(self.module, hybrid, segment, batch["assistant_target_mask"], receipt, units, weight)
        return O.speech_backward(self.module, hybrid, segment, batch["native_media_targets"], batch["native_media_loss_mask"], receipt, units, weight)


def job_digest(config):
    # Runtime assignment pins this immutable training plan; including that pin
    # in its own digest would require an impossible cryptographic fixed point.
    return I.digest({key: value for key, value in config.items() if key not in
                     {"runtime_assignment", "runtime_proof_path", "decode_review_pointer"}})


def runtime_gate(assignment, proof, job_sha, now=None):
    """Consume fresh root attestation, never an active:true metadata substitute.

    Root must obtain live parameterized PG ownership and Docker/image/device/
    readonly-mount evidence. The read-only proof is published by that controller;
    this process cannot renew its own authority. D4 has a distinct runtime schema
    and does not reuse D1-specific RAM or K3-cache completion gates.
    """
    now = now or datetime.now(timezone.utc)
    captured = datetime.fromisoformat(proof["captured_at"])
    if captured.tzinfo is None or now.tzinfo is None:
        raise Error("D4 root proof needs timezone-aware time")
    age = (now - captured).total_seconds()
    maximum = assignment.get("max_proof_age_seconds")
    if (assignment.get("schema") != "chimera_recovery.d4_allocated_runtime.v1"
            or type(maximum) is not int or not 1 <= maximum <= 45
            or not 0 <= age <= maximum or proof.get("schema") != "chimera_recovery.d4_live_runtime.v1"
            or proof.get("valid") is not True or proof.get("proof_producer") != "root_raw_pg_and_server_manager"
            or proof.get("job_sha256") != job_sha or assignment.get("job_sha256") != job_sha
            or any(proof.get(key) != assignment.get(key) for key in
                   ("allocation_group_id", "container_id", "image_id", "gpu_uuids", "root_controller_sha256", "source_tree_sha256"))
            or any(proof.get(key) is not True for key in
                   ("postgres_live", "docker_live", "readonly_sources_verified", "cuda_bf16_verified", "host_budget_verified"))
            or not assignment.get("gpu_uuids") or not assignment.get("allocation_group_id")
            or not I.valid_sha(job_sha) or not I.valid_sha(assignment.get("container_id"))
            or not I.valid_sha(str(assignment.get("image_id", "")).removeprefix("sha256:"))
            or not I.valid_sha(assignment.get("source_tree_sha256"))
            or not I.valid_sha(assignment.get("root_controller_sha256"))):
        raise Error("D4 runtime lease stale/revoked/foreign/unverified; no optimizer authority")
    return dict(assignment_sha256=I.digest(assignment), captured_at=proof["captured_at"], job_sha256=job_sha)


def attest_runtime(assignment, pg_rows, docker, cuda, host, *, now=None):
    """Root-side pure admission of actual bounded raw-PG/Docker/CUDA evidence.

    Collection belongs to the root controller. No DB write, Docker operation or
    quota/session action happens here; an unreconciled allocator timeout cannot
    satisfy these live exact-label checks.
    """
    import uuid
    now = now or datetime.now(timezone.utc)
    group, uuids = assignment["allocation_group_id"], assignment["gpu_uuids"]
    uuid.UUID(group)
    if (not uuids or len(set(uuids)) != len(uuids) or len(pg_rows) != len(uuids)
            or {row.get("gpu_uuid") for row in pg_rows} != set(uuids)):
        raise Error("D4 live PG group does not contain the exact assigned GPU members")
    for row in pg_rows:
        details = row.get("details", {})
        details = json.loads(details) if isinstance(details, str) else details
        if (str(row.get("group_id") or row.get("allocation_id")) != group or row.get("state") != "running"
                or row.get("owner") != "codex" or row.get("server") != "ai" or row.get("kind") != "generic"
                or row.get("name") != assignment["container_name"] or row.get("container_id") != assignment["container_id"]
                or details.get("gpu_uuids") != uuids):
            raise Error("D4 PG lease ownership/container/group differs")
    labels = docker.get("Config", {}).get("Labels", {})
    requests = docker.get("HostConfig", {}).get("DeviceRequests", [])
    if (docker.get("Id") != assignment["container_id"] or docker.get("Image") != assignment["image_id"]
            or docker.get("Name", "").lstrip("/") != assignment["container_name"] or docker.get("State", {}).get("Running") is not True
            or labels.get("ai.local.gpu-controller") != "v1" or labels.get("ai.local.allocation") != group
            or labels.get("ai.local.owner") != "codex" or labels.get("ai.local.kind") != "generic"
            or json.loads(labels.get("ai.local.gpu-uuids", "[]")) != uuids
            or len(requests) != 1 or requests[0].get("Driver") != "nvidia" or requests[0].get("DeviceIDs") != uuids):
        raise Error("D4 actual Docker image/device/controller labels differ")
    mounts = docker.get("Mounts", [])
    observed = {(item["Source"], item["Destination"], not item["RW"]) for item in mounts if item.get("Type") == "bind"}
    expected = {(item["source"], item["target"], item["read_only"]) for item in assignment["mounts"]}
    if len(mounts) != len(expected) or observed != expected or not any(item[2] for item in expected):
        raise Error("D4 live explicit readonly source mounts differ")
    minimum = assignment.get("minimum_mem_available_gib")
    if (type(minimum) not in (float, int) or not math.isfinite(minimum) or minimum <= 0
            or host.get("mem_available_gib", 0) < minimum or host.get("foreign_gpu_processes")
            or host.get("source_tree_sha256") != assignment["source_tree_sha256"]):
        raise Error("D4 fresh host memory/foreign process/source proof refused")
    if (cuda.get("uuids") != uuids or cuda.get("device_count") != len(uuids)
            or cuda.get("ordinals") != list(range(len(uuids))) or cuda.get("bf16_matmul_finite") is not True
            or cuda.get("torch_version") != assignment["torch_version"] or cuda.get("python_executable") != assignment["image_python"]):
        raise Error("D4 actual CUDA UUID/ordinal/Torch/BF16 runtime differs")
    proof = dict(schema="chimera_recovery.d4_live_runtime.v1", valid=True,
                 captured_at=now.isoformat(), proof_producer="root_raw_pg_and_server_manager",
                 postgres_live=True, docker_live=True, readonly_sources_verified=True,
                 cuda_bf16_verified=True, host_budget_verified=True,
                 **{key: assignment[key] for key in ("job_sha256", "allocation_group_id", "container_id", "image_id",
                                                      "gpu_uuids", "root_controller_sha256", "source_tree_sha256")})
    runtime_gate(assignment, proof, assignment["job_sha256"], now)
    return proof


def accepted_lineage(entries, root, context):
    if [entry.get("stage") for entry in entries] != list(C.STAGE_ORDER):
        raise Error("D4 actual predecessor order must include audio CE and distinct speech flow")
    previous_native, previous_shared, sources = None, None, {}
    for index, entry in enumerate(entries):
        acceptance, path = read(entry["acceptance"], root)
        if (acceptance.get("schema") != "chimera_recovery.accepted_stage.v1" or acceptance.get("accepted") is not True
                or acceptance.get("stage") != entry["stage"] or acceptance.get("checkpoint_state") != entry["checkpoint_state"]
                or acceptance.get("checkpoint_receipt") != entry["checkpoint_receipt"] or acceptance.get("config") != entry["config"]):
            raise Error("D4 predecessor requires actual checkpoint/config-bound stage acceptance")
        # Acceptance includes the full previously published answers/outcomes;
        # no finite-logit or structural receipt is an intelligence substitute.
        for name in ("full_answers", "outcomes", "evaluation_protocol"):
            I.pinned(acceptance[name], path.parent)
        for name in ("acceptance", "checkpoint_state", "checkpoint_receipt", "config"):
            I.pinned(entry[name], root)
            sources[f"stage:{index}:{name}"] = I.pin(I.pinned(entry[name], root))
        if entry["stage"] == "D2":
            continue
        shared = I.verify_entry(entry["shared_r16"], root, context)
        native = N.verify_entry(entry["native_state"], root, context, shared["manifest_sha256"])
        nm, sm = native["manifest"], shared["manifest"]
        if (nm["stage"] != entry["stage"].split(".")[0]
                or nm["producer"]["checkpoint_state"] != entry["checkpoint_state"]
                or nm["producer"]["checkpoint_receipt"] != entry["checkpoint_receipt"]
                or nm.get("parent_manifest_sha256") != previous_native
                or sm.get("parent_manifest_sha256") != previous_shared):
            raise Error("D4 actual cumulative native/shared predecessor chain differs")
        if "." in entry["stage"] and nm["producer"].get("unit") != entry.get("unit"):
            raise Error("D4 media stage unit differs from producing export")
        previous_native, previous_shared = entry["native_state"]["sha256"], entry["shared_r16"]["sha256"]
    return dict(native_sha256=previous_native, shared_sha256=previous_shared, sources=sources)


def preflight(config_path):
    from . import train_media as T
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    blockers, verified = [], {}
    def check(label, operation):
        try:
            verified[label] = operation()
        except (ValueError, KeyError, TypeError, OSError) as exc:
            blockers.append(label + ": " + str(exc))
    if config.get("schema") != SCHEMA or config.get("stage") != "D4":
        return dict(schema=SCHEMA, production_ready=False, training_started=False, blockers=["explicit D4 config required"])
    check("preregistration", lambda: T.verify_preregistration(config["preregistration"], path.parent))
    check("hybrid_module", lambda: str(I.pinned(config["hybrid_module"], path.parent)))
    check("immutable_base_receipt", lambda: read(config["immutable_base_receipt"], path.parent)[0])
    check("runtime_assignment", lambda: read(config["runtime_assignment"], path.parent)[0])
    check("d2_binding", lambda: read(config["d2_binding"], path.parent)[0])
    check("previous_rate_ledger", lambda: read(config["previous_rate_ledger"], path.parent)[0])
    try:
        if (config.get("safety") != SAFETY or config.get("dtype") not in {"float32", "bfloat16"}
                or type(config.get("seed")) is not int or type(config.get("accumulation")) is not int or config["accumulation"] < 1
                or config.get("kernels") not in {"torch", "triton"} or not str(config.get("device", "")).startswith("cuda:")
                or config.get("clip_norm") != 1.):
            raise Error("D4 explicit current runtime/seed/accumulation/frozen safety required")
        base = verified["immutable_base_receipt"]
        if (base.get("immutable_clone") is not True or base.get("registered") is not True
                or base.get("verified_equal_to_registered_parent") is not True
                or base.get("base_sha256") != base.get("clone_sha256") or base.get("clone_path") != config["omni"]):
            raise Error("D4 immutable registered D2 native clone receipt required")
        context = I.stage_context(config, base["base_sha256"], path.parent)
        verified["shared_r16"] = I.preflight_config(config, path.parent, base["base_sha256"])
        verified["native_state"] = N.preflight_config(config, path.parent, verified["shared_r16"])
        lineage = accepted_lineage(config["accepted_stages"], path.parent, context)
        if (lineage["native_sha256"] != verified["native_state"]["source_manifest_sha256"]
                or lineage["shared_sha256"] != verified["shared_r16"]["source_manifest_sha256"]):
            raise Error("D4 terminal exports differ from actual accepted lineage")
        verified["lineage"] = lineage
        source_amendments = set()
        selected = I.pinned(config["preregistration"], path.parent).read_bytes().replace(b"\r\n", b"\n")
        tail = selected[selected.find(b"\n## Amendments"):]
        for entry in config.get("compatible_cache_preregistrations", []):
            old = T.verify_preregistration(entry, path.parent)
            raw = I.pinned(entry, path.parent).read_bytes().replace(b"\r\n", b"\n")
            prior = raw[raw.find(b"\n## Amendments"):]
            if not tail.startswith(prior):
                raise Error("D4 source amendment is not an unchanged prefix of selected append-only amendment")
            source_amendments.add(old["amendment_sha256"])
        verified["replay"] = replay_catalog(config["replay_manifest"], path.parent,
                    verified["preregistration"]["amendment_sha256"], compatible_amendments=source_amendments)
        # Exact currently executing helper bytes are part of this run identity.
        helpers = config["helpers"]
        expected = {Path(module.__file__).name: I.sha(Path(module.__file__)) for module in (R, R.D, O, C, I, N)}
        expected[Path(__file__).name] = I.sha(Path(__file__))
        if any(name not in helpers or helpers[name]["sha256"] != digest or I.sha(I.pinned(helpers[name], path.parent)) != digest
               for name, digest in expected.items()):
            raise Error("D4 running union/native/objective/dispatch helper source differs")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        blockers.append("bindings: " + str(exc))
    return dict(schema=SCHEMA, production_ready=False, candidate_preflight_passed=not blockers,
                training_started=False, blockers=blockers, config=config, config_sha256=I.sha(path), verified=verified,
                limitation="preflight metadata alone does not prove actual native objective parity or live update authority")


def previous_rates(groups, entries, d2_binding, ledger, root):
    """Derive rates from real producing optimizer/options, then bind the ledger.

    Native compact interfaces can include several prior stage-owned regions. All
    observed contributing source rates must agree for a single physical leaf;
    a conflict refuses rather than averaging. Current selected recipes happen
    to agree. The last shared-bank producer and last full unit producer determine
    their own previous rates. No old optimizer state is imported into D4.
    """
    import torch
    from decimal import Decimal
    states, provenance = [], {}
    for entry in entries[1:]:
        path = I.pinned(entry["checkpoint_state"], root)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config, _ = read(entry["config"], root)
        if (state.get("step") != state.get("total_steps") or state.get("step", 0) < 1
                or state.get("total_steps") != config.get("steps")
                or state.get("identity", {}).get("config_sha256") != entry["config"]["sha256"]):
            raise Error("D4 actual predecessor did not complete its configured optimizer budget")
        states.append((entry, state))
    d2_optimizer = torch.load(I.pinned(d2_binding["producer"]["optimizer.pt"], root), map_location="cpu", weights_only=True, mmap=True)
    d2_groups = d2_optimizer["param_groups"]
    if len(d2_groups) != 2 or any("base_lr" not in group for group in d2_groups):
        raise Error("D4 actual D2 LoRA/gain optimizer groups differ from pinned producer recipe")
    expected_d2 = [sum(len(groups[key]) for key in ("D2.lora", "D2.k3_lora")), len(groups["D2.gain"])]
    if [len(group["params"]) for group in d2_groups] != expected_d2:
        raise Error("D4 actual D2 optimizer physical parameter membership differs")
    result = {}
    for family, named in groups.items():
        result[family] = {}
        for name in named:
            candidates = []
            if family.startswith("D2."):
                number = 1 if family == "D2.gain" else 0
                candidates = [("D2", d2_binding["producer"]["optimizer.pt"]["sha256"], d2_groups[number]["base_lr"])]
            elif family in {"D3.music_embed", "D3.music_head"}:
                for entry, state in states:
                    if entry["stage"] == "D3c" and "embed" in state and "head" in state:
                        candidates = [("D3c", entry["checkpoint_state"]["sha256"], state["options"]["band_lr"])]
            else:
                source_name = "lora." + name[3:] if family == "D3.shared_lora" else name
                category = "lora" if family == "D3.shared_lora" else family.split(".")[1]
                for entry, state in states:
                    owned = state.get("trainables", state.get("lora", {}))
                    if source_name not in owned:
                        continue
                    optimizer = state.get("optimizer", state.get("lora_optimizer", {}))
                    matching = [group for group in optimizer.get("param_groups", []) if group.get("name") == category]
                    if category == "lora" and not matching and entry["stage"] == "D3c":
                        rate = state["options"]["lora_lr"]
                    elif len(matching) == 1 and "base_lr" in matching[0]:
                        rate = matching[0]["base_lr"]
                    else:
                        raise Error("D4 actual producing optimizer has no unique pinned category base rate: " + name)
                    record = (entry["stage"], entry["checkpoint_state"]["sha256"], rate)
                    if category in {"bridge", "emission"}:
                        candidates.append(record)
                    else:
                        candidates = [record]
            if not candidates or len({Decimal(str(record[2])) for record in candidates}) != 1:
                raise Error("D4 physical leaf lacks a unique authentic previous rate; compact owners cannot average source rates: " + name)
            rate = Decimal(str(candidates[-1][2]))
            result[family][name] = str(rate)
            provenance[name] = dict(family=family, previous_base_lr=str(rate),
                                   producers=[dict(stage=stage, checkpoint_sha256=digest, base_lr=str(Decimal(str(value))))
                                              for stage, digest, value in candidates])
    if (ledger.get("schema") != "chimera_recovery.d4_previous_rates.v1" or ledger.get("entries") != provenance
            or ledger.get("lr_multiplier") != "0.3"):
        raise Error("D4 declared rate ledger differs from actual per-owner producing checkpoints")
    return result


def train(config_path, out, resume=None):
    check = preflight(config_path)
    if not check.get("candidate_preflight_passed"):
        raise Error("D4 candidate preflight blocked: " + "; ".join(check["blockers"]))
    import torch
    from . import train_media as T
    config, verified, root = check["config"], check["verified"], Path(config_path).parent
    assignment, _ = read(config["runtime_assignment"], root)
    def gate():
        # Each actual apply reopens the read-only root heartbeat, and checks its
        # current pins; a persisted active allocation label alone grants nothing.
        current, _ = read(config["runtime_assignment"], root)
        proof = json.loads(Path(config["runtime_proof_path"]).read_text(encoding="utf-8"))
        return runtime_gate(current, proof, job_digest(config))
    gate()
    if (str(torch.__version__) != assignment["torch_version"] or sys.executable != assignment["image_python"] or not torch.cuda.is_available()
            or torch.cuda.device_count() != len(assignment["gpu_uuids"])
            or [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())] != assignment["gpu_uuids"]):
        raise Error("D4 actual in-container Torch/CUDA membership differs")
    mountinfo = Path("/proc/self/mountinfo").read_text()
    mounts = {line.split()[4].replace("\\040", " "): "ro" in line.split()[5].split(",")
              for line in mountinfo.splitlines() if len(line.split()) >= 10 and "-" in line.split()}
    if any(mounts.get(item["target"]) is not item["read_only"] for item in assignment["mounts"]):
        raise Error("D4 actual in-container mount access differs from root readonly source attestation")
    module = T.load_module(Path(verified["hybrid_module"]), "chimera_recovery_d4_actual_hybrid")
    trainer = module.trainer_module()
    I.pinned(config["streamed_trainer"], root)
    if I.sha(Path(trainer.__file__)) != config["streamed_trainer"]["sha256"]:
        raise Error("D4 genuine streamed trainer differs from pinned native objective implementation")
    torch.manual_seed(config["seed"])
    shared_plan = trainer.SitePlanConfig(rank=16, alpha=16., k3_rank=0, gain=False, trunk_ffn_width=config["trunk_ffn_width"])
    hybrid = module.HybridD3(config["omni"], config["parent_config"], module.HybridOptions(dtype=getattr(torch, config["dtype"]),
                device=config["device"], kernels=config["kernels"], lora=shared_plan, lora_seed=config["seed"], plan=config.get("compose_plan")))
    shared = I.initialize_stage(hybrid, verified["shared_r16"], root)
    native = N.initialize_stage(hybrid, verified["native_state"], root)
    original = trainer.P1Source(config["pre_d2_omni"], config["parent_config"], config.get("pre_d2_compose_plan"))
    d2_plan = trainer.SitePlanConfig(rank=64, alpha=64., k3_rank=32, k3_alpha=32., gain=True, trunk_ffn_width=config["trunk_ffn_width"])
    d2 = trainer.AdapterBank(trainer.build_site_plan(hybrid.text_config, d2_plan), hybrid.bank.layer_types, config["device"], config["seed"])
    binding = R.D.verify_d2_source(config["d2_binding"], root, original, d2)
    rates = lambda groups: previous_rates(groups, config["accepted_stages"], binding, verified["previous_rate_ledger"], root)
    union = R.construct_union(module, hybrid, original, d2, binding,
                dict(source=config["native_state"]["source"], receipt=native),
                dict(source=config["shared_r16"]["source"], receipt=shared), rates)
    union["identity"]["dispatch"] = dict(config_sha256=check["config_sha256"], job_sha256=job_digest(config),
                preregistration=verified["preregistration"], runtime_assignment=config["runtime_assignment"],
                helpers=config["helpers"], candidate_preflight_passed=True, production_ready=False)
    replay = verified["replay"]
    replay["sources"].update(verified["lineage"]["sources"])
    replay["sources"].update({key: I.pin(I.pinned(ref, root)) for key, ref in
                            {"preregistration": config["preregistration"], "rates": config["previous_rate_ledger"],
                             "streamed_trainer": config["streamed_trainer"], **config["helpers"]}.items()})
    engine = R.UnionEngine(union, replay["rows"], replay["sources"], Path(replay["root"]),
                           accumulation=config["accumulation"], clip_norm=config["clip_norm"])
    engine.identity["replay"]["supervised_denominators"] = replay["measured_supervised_units"]
    return execute(engine, NativeBackward(module, Path(replay["root"]), config["text_objective"]), out,
                   gate=gate, review_pointer=config.get("decode_review_pointer"), resume=resume)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--config", type=Path, required=True)
        if name == "train":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.config)
        print(json.dumps({key: result[key] for key in ("production_ready", "candidate_preflight_passed", "blockers") if key in result}))
        return 0 if result.get("candidate_preflight_passed") else 2
    print(json.dumps(train(args.config, args.out, args.resume)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def policy_record(value):
    if (set(value) != {"consecutive_skips", "window", "decode_review", "attempts"}
            or type(value["consecutive_skips"]) is not int or not 0 <= value["consecutive_skips"] <= 8
            or type(value["attempts"]) is not int or value["attempts"] < 0
            or not isinstance(value["window"], list) or len(value["window"]) > 100
            or any(type(bit) is not bool for bit in value["window"])
            or (value["decode_review"] is not None and not isinstance(value["decode_review"], dict))):
        raise Error("D4 resumed dispatcher safety policy invalid")
    return copy.deepcopy(value)


def save(engine, path, policy):
    path = Path(path)
    engine.checkpoint(path)
    (path / "dispatch_policy.json").write_bytes(I.canonical(policy_record(policy)) + b"\n")
    receipt = dict(schema=CHECKPOINT, identity_sha256=I.digest(engine.identity),
                   engine_receipt=I.pin(path / "receipt.json"), policy=I.pin(path / "dispatch_policy.json"),
                   step=engine.step, cursor=engine.cursor)
    (path / "dispatch_receipt.json").write_bytes(I.canonical(receipt) + b"\n")


def restored_policy(engine, path):
    path = Path(path)
    value = json.loads((path / "dispatch_receipt.json").read_text(encoding="utf-8"))
    if (value.get("schema") != CHECKPOINT or value.get("identity_sha256") != I.digest(engine.identity)
            or value.get("step") != engine.step or value.get("cursor") != engine.cursor
            or I.pinned(value["engine_receipt"], path) != (path / "receipt.json").resolve()):
        raise Error("D4 resume dispatcher receipt/engine identity differs")
    policy, _ = read(value["policy"], path)
    return policy_record(policy)


def decode_review(entry, engine, root, checkpoint):
    value, path = read(entry, root)
    quarter = max(1, math.ceil(engine.total_steps * .25))
    if (value.get("schema") != "chimera_recovery.d4_decode_review.v1"
            or value.get("run_identity_sha256") != I.digest(engine.identity)
            or value.get("trained_step") != engine.step or engine.step < quarter
            or value.get("meaningful_prompt_conditioned_content") is not True
            or value.get("verdict") != "PASS" or not value.get("reviewer")
            or value.get("checkpoint_receipt_sha256") != I.sha(checkpoint / "receipt.json")):
        raise Error("D4 requires actual source-bound quarter decode review")
    for field in ("full_answers", "outcomes", "evaluation_protocol", "baseline_answers"):
        I.pinned(value[field], path.parent)
    return I.pin(path)


def execute(engine, backward, out, *, gate, review_pointer=None, resume=None, clock=time.monotonic, export=True):
    """One actual replay pass, bounded safety and exact checkpoint-policy resume.

    Caller supplies only the native callbacks below and a fresh root runtime
    gate. Explicit fixture=True engines can prove mechanics on actual tiny
    HybridD3; their receipts and exports remain marked fixture/unaccepted.
    """
    from . import train_media as T
    out = Path(out)
    if out.exists():
        raise Error("preserve D4 output; explicit resume requires a fresh run directory")
    policy = dict(consecutive_skips=0, window=[], decode_review=None, attempts=0)
    if resume:
        engine.resume(Path(resume))
        policy = restored_policy(engine, Path(resume))
    if policy["consecutive_skips"] >= 8 or (len(policy["window"]) == 100 and sum(policy["window"]) > 20):
        raise Error("D4 existing skip-halt needs a new ruling; exact resume cannot erase it")
    out.mkdir(parents=True)
    save(engine, out / "checkpoint-initial", policy)
    last_safety = clock()
    while True:
        quarter = max(1, math.ceil(engine.total_steps * .25))
        if engine.step >= quarter and policy["decode_review"] is None:
            checkpoint = out / f"checkpoint-review-{engine.step:08d}"
            save(engine, checkpoint, policy)
            if not review_pointer or not Path(review_pointer).is_file():
                return dict(status="awaiting_decode_review", step=engine.step, cursor=engine.cursor,
                            checkpoint=str(checkpoint), training_complete=False, production_ready=False)
            pointer = json.loads(Path(review_pointer).read_text(encoding="utf-8"))
            expected = Path(resume) if resume and engine.step == json.loads((Path(resume) / "receipt.json").read_text())["step"] else checkpoint
            policy["decode_review"] = decode_review(pointer, engine, out, expected)
        if engine.cursor >= len(engine.rows):
            break
        try:
            result = engine.window(backward, before_apply=gate)
        except BaseException:
            save(engine, out / f"checkpoint-failure-{engine.step:08d}", policy)
            raise
        skipped = result.get("applied") is not True
        policy["attempts"] += 1
        policy["consecutive_skips"] = policy["consecutive_skips"] + 1 if skipped else 0
        policy["window"] = (policy["window"] + [skipped])[-100:]
        with (out / "attempts.jsonl").open("ab") as stream:
            stream.write(I.canonical(dict(attempt=policy["attempts"], **result)) + b"\n")
        if policy["consecutive_skips"] >= 8 or (len(policy["window"]) == 100 and sum(policy["window"]) > 20):
            save(engine, out / f"checkpoint-skip-halt-{engine.step:08d}", policy)
            raise Error("D4 persistent nonfinite windows halted without consuming rows")
        if clock() - last_safety >= 1800:
            save(engine, out / f"checkpoint-safety-{policy['attempts']:08d}", policy)
            T.prune_safety_checkpoints(out, keep=2)
            last_safety = clock()
    checkpoint = out / "checkpoint-final"
    save(engine, checkpoint, policy)
    result = dict(status="complete", step=engine.step, cursor=engine.cursor, training_complete=True,
                  accepted=False, registered=False, production_ready=False, checkpoint=str(checkpoint),
                  supervised_denominators=engine.identity["replay"].get("supervised_denominators", {}))
    if export:
        from . import consolidation_writer as W
        result["candidate_replacements"] = W.export_candidate(engine, checkpoint, out / "candidate-replacements")
    return result
