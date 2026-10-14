"""Authenticate actual recovery targets and prepare their immutable handoff.

`inspect` inventories received bytes and missing mounts. `cpu-targets` runs the
existing native source/tensor consumers with CUDA hidden, without a model or an
optimizer. Neither operation grants an accepted checkpoint or runtime lease.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from . import consolidation_dispatch as D, stage_jobs as J

SCHEMA = "chimera_recovery.target_preparation.v1"
REQUEST = "chimera_recovery.target_preparation_request.v1"
HERE = Path(__file__).resolve().parent
ROLES = {
    "ace": ("D3b.ace", "guided_velocity", "chimera_recovery.native_batches.v1"),
    "qimg": ("D3b.qimg", "guided_velocity", "chimera_recovery.native_batches.v1"),
    "geometry": (D.GEOMETRY_STAGE, D.O.GEOMETRY, "recovery_geometry_flow_batches.v1"),
    "image_edit": (D.EDIT_STAGE, D.O.EDIT, "chimera_recovery.image_edit_batches.v1"),
    "h3": ("D3b.h3", "guided_velocity", "chimera_recovery.native_batches.v1"),
    "music_code": ("D3c", "music_code_band_kl", "chimera_recovery.music_batches.v1"),
    "vision": ("D3d", D.O.VISION, "chimera_recovery.vision_batches.v1"),
    "audio_ce": ("D3d_audio_ce", D.O.AUDIO, "chimera_recovery.audio_batches.v1"),
    "speech_flow": ("D3d_speech_flow", D.O.SPEECH, "chimera_recovery.speech_flow_batches.v1"),
}
ROUTES = {"ace": ["audio_generate", "music_generate"], "qimg": ["image_generate"],
    "geometry": ["volume_generate"], "image_edit": ["image_edit"], "h3": ["joint_av"],
    "music_code": ["music_prompt_to_code"], "vision": ["vision_grounding", "video_understand"],
    "audio_ce": ["audio_understand", "speech_understand"], "speech_flow": ["speech_generate"]}
CODE = ("recovery_readiness.py", "stage_jobs.py", "consolidation_dispatch.py", "consolidation_objectives.py",
    "train_media.py", "training.py", "media_cache.py", "stage_inheritance.py", "native_state_inheritance.py",
    "train_geometry.py", "geometry_flow.py", "train_image_edit.py", "image_edit.py", "h3_native.py",
    "train_audio.py", "audio_cache.py", "train_speech_flow.py", "speech_flow.py", "audio_replay_index.py",
    "train_stages.py", "train_music.py", "music_cache.py", "train_vision.py", "vision_cache.py")
SELECTED_POLICY = {
    "config": {"accumulation":1,"clip_norm":1.,"seed":20261013,"dtype":"bfloat16","kernels":"torch"},
    "optimizer": {"kind":"fresh_AdamW","betas":[.9,.95],"eps":1e-8,"weight_decay":0.,"schedule":"fixed_base_rates"},
    "lr_multiplier":"0.3",
    "expected_d4_rates": {"D2.lora":"0.00003","D2.k3_lora":"0.00003","D2.gain":"0.0003",
        "D3.shared_lora":"0.00003","D3.unit":"0.000006","D3.bridge":"0.000006",
        "D3.emission":"0.000006","D3.music_embed":"0.000006","D3.music_head":"0.000006"},
    "frozen_quotas":D.C.QUOTAS,
    "exact_batch_counts":{"geometry":32,"h3":32,"image_edit":32,"audio_ce":194,"speech_flow":744},
    "recorded_index_units":{D.O.AUDIO:2833,D.O.SPEECH:8776960},
    "music_code_membership":"every_final_admitted_native_batch_once_count_pinned_after_full_source_seal",
    "outer_objective_weight":1.,
    "source_order":"unchanged_manifest_order_one_pass_no_loss_ranking_or_padding",
    "text_budget":"whole_source_rows_at_least_quota_exact_overshoot_requires_pinned_selected_declaration",
    "intrinsic_losses":"unchanged_producer_objectives_and_denominators_including_vision_KD1_featureMSE0.1",
}


class Error(ValueError): pass


def sha(path): return J.sha(path)
def pin(path): return J.pin(path)
def canonical(value): return J.canonical(value)


class Locator:
    """Aliases locate byte-identical files; paths and producer objects stay intact."""
    def __init__(self, aliases=None):
        self.aliases = aliases or {}
        if (not isinstance(self.aliases, dict) or any(not isinstance(v, str) or not Path(v).is_absolute()
            or re.fullmatch('[a-f0-9]{64}', k) is None for k, v in self.aliases.items())):
            raise Error("aliases must map exact SHA256 to absolute actual paths")
        self.observed, self.missing, self.changed_graph_edges = {}, {}, {}

    def file(self, ref, root, *, required=True):
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise Error("actual path/hash reference required")
        digest = J.require_sha(ref.get("sha256"))
        original = Path(ref["path"])
        original = original if original.is_absolute() else Path(root) / original
        path = original
        if not path.is_file() and digest in self.aliases: path = Path(self.aliases[digest])
        if not path.is_file():
            self.missing[(ref["path"], digest)] = dict(ref)
            if required: raise FileNotFoundError("unchanged pinned file/mount absent: " + str(original))
            return None
        stat = path.stat(); identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        previous = self.observed.get(identity)
        if previous is None:
            actual = sha(path)
            if actual != digest: raise Error("pinned bytes changed: " + str(path))
            if "bytes" in ref and ref["bytes"] != stat.st_size: raise Error("pinned byte count changed")
            self.observed[identity] = {"original":dict(ref),"received":pin(path)}
        elif previous["received"]["sha256"] != digest: raise Error("contradictory file identity")
        return path.resolve()

    def json(self, ref, root):
        path = self.file(ref, root)
        return json.loads(path.read_text(encoding="utf-8")), path

    def graph(self, value, root, seen=None):
        """Authenticate supplied file edges, recurse JSON only; never invent pins."""
        seen = set() if seen is None else seen
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and re.fullmatch('[a-f0-9]{64}', str(value.get("sha256"))):
                try:
                    path = self.file(value, root, required=False)
                except Error as exc:
                    # A historical receipt may point at an editable worktree. The
                    # audit preserves its original pin and reports unavailable
                    # historical bytes; direct target/helper reads remain strict.
                    if not str(exc).startswith("pinned bytes changed:"): raise
                    original = Path(value["path"])
                    original = original if original.is_absolute() else Path(root)/original
                    self.changed_graph_edges[(value['path'],value['sha256'])] = {
                        'required':dict(value),'observed':pin(original),
                        'reason':'historical pinned bytes differ at the recorded path; immutable original required'}
                    return
                if path is not None and path.suffix == ".json" and value["sha256"] not in seen:
                    seen.add(value["sha256"])
                    self.graph(json.loads(path.read_text(encoding="utf-8")), path.parent, seen)
                return
            for child in value.values(): self.graph(child, root, seen)
        elif isinstance(value, list):
            for child in value: self.graph(child, root, seen)


def membership(role, data, path, locator):
    stage, objective, schema = ROLES[role]
    if data.get("schema") != schema or data.get("sealed") is not True or data.get("complete_source_traversal") is not True:
        raise Error("complete actual native cache required: " + role)
    if role not in {"geometry", "image_edit"} and (data.get("split") != "train" or data.get("heldout_groups_excluded") is not True):
        raise Error("native cache must explicitly exclude heldout: " + role)
    if role in {"geometry", "image_edit"} and data.get("stage") != stage:
        raise Error("distinct supervised stage differs")
    if data.get('objective') is not None and data['objective'] != objective:
        raise Error('native objective differs from its requested lane')
    if 'manifest_sha256' in data and data['manifest_sha256'] != hashlib.sha256(canonical({k:v for k,v in data.items() if k!='manifest_sha256'})).hexdigest():
        raise Error('native semantic manifest seal changed')
    items = data.get("batches", [])
    if not items: raise Error("actual target batch inventory missing")
    ids, pairs, sources, groups, units = set(), set(), set(), set(), Counter()
    received = 0; order=[]
    receipt_key = "receipt" if role == "geometry" else "teacher_receipt"
    for item in items:
        ident, group = item.get("id"), item.get("recovery_group_id")
        if not isinstance(ident, str) or not ident or ident in ids or not isinstance(group, str) or not group:
            raise Error("missing/duplicate batch or source group")
        if role != 'geometry' and item.get('route') not in ROUTES[role] and not (role=='music_code' and item.get('route')=='music_generate'):
            raise Error('native target routed to a different lane')
        ids.add(ident); groups.add(group);order.append(ident)
        source = item.get("source_id", ident if role=='audio_ce' else None)
        if source is not None:sources.add(source)
        receipt_path = locator.file(item[receipt_key], path.parent, required=False)
        if receipt_path is None: continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")); received += 1
        metadata=D.guided_receipt_metadata(role,receipt,item,data) if role in {'ace','qimg','h3'} else receipt
        tensor_sha = receipt.get("tensors", {}).get("sha256") if role == "geometry" else metadata.get("tensors_sha256")
        if tensor_sha != item["tensors"]["sha256"] or metadata.get("recovery_group_id") != group:
            raise Error("target receipt and manifest tensor/group differ: "+role+":"+ident)
        if receipt.get("sealed") is not True or metadata.get("split", "train") != "train" or metadata.get("held", False) is not False:
            raise Error("unsealed/held native target")
        if role in {"geometry", "image_edit", "speech_flow"}:
            if not isinstance(source,str) or not source:raise Error('explicit authentic source/sigma identity required')
            pair = source, receipt.get("sigma")
            if pair in pairs: raise Error("source/sigma target repeated")
            pairs.add(pair)
        if role == "audio_ce":
            count = receipt.get("assistant_targets")
            if type(count) is not int or count < 1: raise Error("recorded assistant denominator absent")
            units["assistant_next_token_positions"] += count
        if role == "image_edit":
            if receipt.get("supervised_units") != 65536: raise Error("image edit denominator changed")
            units["native_output_scalars"] += 65536
        locator.graph(receipt, receipt_path.parent)
    if pairs and received == len(items):
        sigmas = data.get("sigma_grid", data.get("sigmas"))
        if not isinstance(sigmas, list) or not sigmas or pairs != {(s,v) for s in sources for v in sigmas}:
            raise Error("native target source/sigma grid omitted or changed")
    locator.graph(data, path.parent)
    return dict(stage=stage, objective=objective, routes=ROUTES[role], batches=len(ids),
        source_ids=sorted(sources), recovery_groups=sorted(groups), batch_ids=order,
        recorded_supervised_units=dict(units), manifest=pin(path),
        received_receipts=received, metadata_membership_verified=received==len(items),
        tensor_reconstruction_performed=False, production_ready=False)


def audio_index(entry, root, locator, inventory):
    """Compare every actual receipt/member and independently sum recorded units."""
    index, path = locator.json(entry, root)
    if (index.get("schema") != "chimera_recovery.audio_speech_replay_index.v1" or
        any(index.get(k) is not True for k in ("sealed", "passed", "verified", "complete_source_traversal", "source_reconstruction_bitwise_equal"))):
        raise Error("complete independently reconstructed index required")
    objective_roles = {D.O.AUDIO:"audio_ce", D.O.SPEECH:"speech_flow"}
    amounts, counts, seen = Counter(), Counter(), set()
    for row in index.get("entries", []):
        ident = row.get("id"); role = objective_roles.get(row.get("objective"))
        if not role or ident in seen or not ident: raise Error("unknown/duplicate indexed target")
        seen.add(ident)
        manifest, mp = locator.json(row["source_manifest"], path.parent)
        matches = [i for i in manifest["batches"] if i["id"] == ident]
        if len(matches) != 1: raise Error("index target absent in actual source manifest")
        item = matches[0]
        if any(item.get(k) != row.get(k) for k in ("tensors", "teacher_receipt", "storage", "recovery_group_id")):
            raise Error("index target differs from producer membership")
        if row["source_manifest"]["sha256"] != inventory[role]["manifest"]["sha256"]:
            raise Error("index not bound to selected target manifest")
        receipt, _ = locator.json(row["teacher_receipt"], mp.parent)
        value = row.get("supervised_units")
        if type(value) is not int or value <= 0: raise Error("positive actual indexed denominator required")
        if role == "audio_ce":
            if value != receipt.get("assistant_targets") or row.get("denominator_kind") != "assistant_next_token_positions_including_completion_eos":
                raise Error("indexed assistant denominator differs")
        else:
            if (value != row.get("native_audio_rows", 0)*32 or row.get("native_audio_rows") != row.get("latent_frames_per_channel", 0)*2
                or row.get("sigma") != receipt.get("sigma") or row.get("noise_seed") != receipt.get("noise_seed")
                or row.get("denominator_kind") != "native_weighted_audio_rows_times32_emission_channels"):
                raise Error("indexed native scalar denominator/sigma differs")
        if row.get("source_row_sha256") != receipt.get("source_row_sha256"):
            raise Error("indexed source row identity differs")
        counts[row["objective"]] += 1; amounts[row["objective"]] += value
    expected_ids = set(inventory["audio_ce"]["batch_ids"]) | set(inventory["speech_flow"]["batch_ids"])
    if seen != expected_ids: raise Error("index omitted an actual eligible target")
    measured = {obj:dict(batches=counts[obj], supervised_units=amounts[obj]) for obj in objective_roles}
    if index.get("objectives") != measured: raise Error("index summary differs from actual target rows")
    locator.graph(index, path.parent)
    return dict(pin=pin(path), objectives=measured, source_identity_verified=True,
        tensor_reconstruction_repeated=False, production_ready=False)


def inspect(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != REQUEST: raise Error("explicit target preparation request required")
    aliases = {}
    for ref in request.get("alias_files", []):
        path = Locator().file(ref, request_path.parent)
        mapping = json.loads(path.read_text(encoding="utf-8"))
        Locator(mapping)
        for digest, value in mapping.items():
            if digest in aliases and aliases[digest] != value: raise Error("conflicting literal source aliases")
            aliases[digest] = value
    locator = Locator(aliases)
    # Execute only this source capsule. No untrusted code-path or historical argv.
    for name, ref in request.get("helpers", {}).items():
        if name not in CODE or ref["sha256"] != sha(HERE/name): raise Error("executing source pin differs: " + name)
        locator.file(ref, request_path.parent)
    if set(request.get("helpers", {})) != set(CODE): raise Error("complete executing source inventory required")
    inventory = {}
    absent = request.get("absent_targets", {})
    if (set(request.get("targets", {})) & set(absent) or set(request.get("targets", {})) | set(absent) != set(ROLES)
        or any(not isinstance(reason, str) or not reason.strip() for reason in absent.values())):
        raise Error("every target lane must have exact artifacts or an explicit genuine data dependency")
    for role, entry in request["targets"].items():
        data, path = locator.json(entry, request_path.parent)
        inventory[role] = membership(role, data, path, locator)
    replay = audio_index(request["audio_speech_index"], request_path.parent, locator, inventory)
    evidence = {}
    for key, entry in request.get("evidence", {}).items():
        path = locator.file(entry, request_path.parent)
        evidence[key] = pin(path)
    dependencies = request.get("execution_dependencies")
    if (not isinstance(dependencies, list) or not dependencies or any(not isinstance(v, dict)
        or set(v) != {"kind","reason"} or v["kind"] not in {"accepted_state", "teacher_capture", "native_cache", "parity", "runtime", "review", "selection"}
        or not isinstance(v["reason"], str) or not v["reason"].strip() for v in dependencies)):
        raise Error("typed genuine execution dependencies required")
    return dict(schema=SCHEMA, request=pin(request_path), source_commit=request["source_commit"],
        helpers=request["helpers"], alias_files=request.get("alias_files", []),
        native_targets=inventory, absent_targets=absent, audio_speech_index=replay, evidence=evidence,
        lineage_order=D.replay_lineage_order(), execution_dependencies=dependencies,
        authenticated_files=list(locator.observed.values()), missing_files=list(locator.missing.values()),
        changed_graph_edges=list(locator.changed_graph_edges.values()),
        full_source_graph_verified=not locator.missing and not locator.changed_graph_edges,
        metadata_membership_verified=all(v['metadata_membership_verified'] for v in inventory.values()),
        full_target_preflight_passed=False,
        accepted_states_created=False, runtime_acquired=False, training_launched=False, production_ready=False)


def cpu_targets(request_path, roles):
    """Use the real native validators; original producer paths must be mounted."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise Error("explicit CUDA-hidden CPU required")
    result = inspect(request_path)
    if not result['full_source_graph_verified']:
        raise Error('full unchanged source graph/mounts required before CPU target reconstruction')
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not roles or len(set(roles)) != len(roles) or set(roles)-set(request["targets"]): raise Error("explicit unique actual target roles required")
    import torch
    torch.set_num_threads(8)
    checks = {}
    for role in roles:
        ref = request["targets"][role]
        # Metadata aliases do not monkey-patch production consumers.
        path = Locator().file(ref, Path(request_path).parent)
        data = json.loads(path.read_text(encoding="utf-8")); count = 0
        if role == "geometry":
            from . import train_geometry as G
            G.cache_manifest(path, data["amendment"]["sha256"])
            for item in data["batches"]: G.load_row(item, path.parent); count += 1
        elif role == "image_edit":
            from . import train_image_edit as E
            E.cache_manifest(path, data["amendment_sha256"])
            for item in data["batches"]: E.load_row(item, path.parent); count += 1
        elif role == "audio_ce":
            from . import train_audio as A
            data = A.audio_batches(path, data["amendment_sha256"])
            C = A.T.load_module(HERE/"audio_cache.py", "readiness_authentic_audio_cache")
            for item in data["batches"]:
                batch=A.T.load_native_batch(item,path.parent,'cpu')
                receipt=A.T.read_json(A.T.pinned_file(item['teacher_receipt'],path.parent))
                C.validate_batch(batch,receipt,path.parent);count += 1
        elif role == "speech_flow":
            from . import train_speech_flow as W
            data = W.batches(path, data["amendment_sha256"])
            for item in data["batches"]:
                batch=W.T.load_native_batch(item,path.parent,'cpu')
                receipt=W.T.read_json(W.T.pinned_file(item['teacher_receipt'],path.parent))
                W.F.validate_batch(batch,receipt,path.parent);count += 1
        elif role in {"ace", "qimg", "h3"}:
            from . import train_media as T
            for item in data["batches"]:
                batch=T.load_native_batch(item,path.parent,'cpu')
                rp=T.pinned_file(item['teacher_receipt'],path.parent);receipt=T.read_json(rp)
                if role == "h3": T.authenticate_h3_batch(item,batch,receipt,path.parent,pin(HERE/'h3_native.py'),HERE)
                storage=item.get('storage', {})
                if not isinstance(storage.get('bands'),dict): raise Error('actual native storage/bands required for target-only binding')
                bands={int(k):v for k,v in storage['bands'].items()}
                T.R.ReadyObjectiveAdapter(receipt).bind(item['route'],batch,bands,
                    parent_signs={int(k):v for k,v in item.get('parent_signs',{}).items()});count += 1
        else: raise Error("native full consumer not prepared for this lane; use its actual stage preflight")
        checks[role] = dict(batches=count, actual_native_consumer_executed=True)
    if torch.cuda.is_initialized(): raise Error("CUDA initialized in CPU target preflight")
    result.update(full_target_preflight_passed=set(roles)==set(request['targets']),
        selected_target_preflight_passed=True, checked_roles=roles, cpu_target_checks=checks, model_loaded=False,
        optimizer_created=False, production_ready=False)
    return result


def verify_policy(entry, root, config=None):
    value,path=Locator().json(entry,root)
    if (value.get('schema')!='chimera_recovery.d4_selected_policy.v1' or value.get('selected') is not True
        or value.get('dated')!='2026-10-13' or value.get('constants')!=SELECTED_POLICY):
        raise Error('exact selected reproducible D4 policy required')
    document=Locator().file(value['preregistration'],path.parent)
    raw,_,tail=J.preregistration_parts(document.read_bytes())
    if hashlib.sha256(tail).hexdigest()!=value.get('amendment_sha256'):
        raise Error('selected combined amendment changed')
    if json.dumps(SELECTED_POLICY,sort_keys=True,indent=2).encode() not in raw:
        raise Error('combined document does not contain the exact selected D4 policy')
    if config is not None:
        if any(config.get(k)!=v for k,v in SELECTED_POLICY['config'].items()):
            raise Error('D4 execution scalars differ from selected policy')
        if config.get('preregistration',{}).get('sha256')!=value['preregistration']['sha256']:
            raise Error('D4 config uses another selected combined document')
    return value


def verify_replay_policy(policy, replay):
    mapping={'geometry':'hy3d','h3':'h3','image_edit':'image_edit','audio_ce':'audio_ce','speech_flow':'speech_flow'}
    for role,lane in mapping.items():
        if replay['counts'].get(lane)!=SELECTED_POLICY['exact_batch_counts'][role]:
            raise Error('D4 replay count differs from selected policy: '+role)
        if {r['source_manifest']['sha256'] for r in replay['rows'] if r['lane']==lane}!={policy['target_manifests'][role]['sha256']}:
            raise Error('D4 selected target source changed: '+role)
    for obj,count in SELECTED_POLICY['recorded_index_units'].items():
        if replay['measured_supervised_units'].get(obj)!=count:raise Error('D4 measured native replay denominator differs')
    if any(type(row.get('objective_weight')) not in (int,float) or row['objective_weight']!=1. for row in replay['rows']):
        raise Error('D4 outer objective weights differ from selected policy')
    return True


def verify_derived_rates(rates, policy):
    from decimal import Decimal
    expected=policy['constants']['expected_d4_rates']
    if set(rates)!=set(expected):raise Error('D4 actual producing rates omit a selected physical family')
    for family,values in rates.items():
        if not isinstance(values,dict) or not values:raise Error('D4 actual physical owner rate ledger required')
        for rate in values.values():
            if Decimal(str(rate))*Decimal('0.3')!=Decimal(expected[family]):
                raise Error('D4 actual producing base rate differs from selected policy: '+family)
    return rates


def combine_preregistration(request_path, selected_path, out, *, select_established_policy=False):
    """Append fixed root-selected D4 policy; no accepted state or target fabrication."""
    result = inspect(request_path)
    if Path(out).exists() or Path(out).with_suffix('.selection.json').exists():
        raise Error('fresh immutable combined document and selection namespace required')
    if select_established_policy:
        if any(result['native_targets'].get(role,{}).get('batches')!=count
               for role,count in SELECTED_POLICY['exact_batch_counts'].items()):
            raise Error('selected exact native target counts differ; new scope selection required')
        if any(result['audio_speech_index']['objectives'][obj]['supervised_units']!=count
               for obj,count in SELECTED_POLICY['recorded_index_units'].items()):
            raise Error('selected measured replay denominators differ')
    raw, body, tail = J.preregistration_parts(Path(selected_path).read_bytes())
    additions = b"".join(J.amendment(stage) for stage in J.STAGES)
    if any(J.amendment(stage) in raw for stage in J.STAGES): raise Error("stage contract already appended; no duplication")
    summary = {role:{"manifest":r["manifest"],"batches":r["batches"],"batch_ids":r["batch_ids"]}
        for role,r in result["native_targets"].items()}
    appendix = ("\n\n### 2026-10-13: combined D4 source-pinned recovery preparation candidate\n\n"+
        ("The reproducible policy below is explicitly selected for preparation; " if select_established_policy
         else "This candidate is unselected; ")+"it does not accept prior states or grant a runtime. "
        "Preserve every original native target, receipt and preregistration file. D4 accepts an earlier "
        "cache amendment only by proving its normalized complete document is an unchanged prefix of this "
        "combined document; it never relabels target receipts. Bind a selected exact replay-membership "
        "amendment and source-wide exclusion graph before training. Retain the original 2.5M short/0.5M "
        "long/1M K3 real-token budgets and 300 distinct vision/ACE/QIMG pairs; do not count padding or "
        "assistant/scalar denominators as real tokens. For supervised geometry, guided H3, image editing "
        "and music-code replay, use each actual admitted source batch exactly once, with an explicit "
        "source-membership amendment. Missing music/vision/text targets cannot be replaced by repetition "
        "or fixtures. Add every admitted audio CE row and speaking source/sigma pair once from the "
        "independently reconstructed audio/speech index. Rates are 0.3 times each physical owner's "
        "verified preceding base rate, with one fresh union optimizer. Preserve D2 r64/K3/gain exactly "
        "once, final cumulative native state and the final full shared-r16 replacement bank. No new "
        "full-trunk thaw or independent-bank sum is authorized by preparation.\n\n"
        "Selected reproducible constants (actual producer checkpoints must independently agree):\n\n```json\n"+
        json.dumps(SELECTED_POLICY,sort_keys=True,indent=2)+"\n```\n\n"
        "Exact current target preparation (receipt inventory only; full runtime preflight still required):\n\n```json\n"+
        json.dumps({"request":result["request"],"audio_speech_index":result["audio_speech_index"]["pin"],
        "lineage_order":result["lineage_order"],"current_native_targets":summary,
        "missing_target_dependencies":result["absent_targets"]},sort_keys=True,indent=2)+"\n```\n").encode()
    candidate = raw + additions + appendix
    _, after_body, after_tail = J.preregistration_parts(candidate)
    if body != after_body or not after_tail.startswith(tail): raise Error("append moved frozen/source contract")
    J.write_new(Path(out), candidate)
    selection = dict(schema='chimera_recovery.d4_selected_policy.v1',selected=select_established_policy,
        dated='2026-10-13',constants=SELECTED_POLICY,preregistration=pin(out),
        amendment_sha256=hashlib.sha256(after_tail).hexdigest(),
        audio_speech_index=result['audio_speech_index']['pin'],
        target_manifests={role:r['manifest'] for role,r in result['native_targets'].items()},
        replay_membership_complete=False,accepted_states_created=False,production_ready=False)
    selection_path=Path(out).with_suffix('.selection.json')
    J.write_new(selection_path,canonical(selection)+b'\n')
    return dict(candidate=pin(out), selection=pin(selection_path), selected=select_established_policy, production_ready=False,
        normalized_amendment_sha256=hashlib.sha256(after_tail).hexdigest(), predecessor=pin(selected_path),
        independent_stage_contracts=list(J.STAGES), full_target_preflight_passed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ("inspect", "cpu-targets", "combine-preregistration"):
        q=sub.add_parser(name);q.add_argument("--request",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
        if name=="cpu-targets":q.add_argument("--role",choices=sorted(ROLES),action="append",required=True)
        if name=="combine-preregistration":
            q.add_argument("--selected-preregistration",type=Path,required=True)
            q.add_argument('--select-established-policy',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.command=="combine-preregistration":result=combine_preregistration(a.request,a.selected_preregistration,a.out,
            select_established_policy=a.select_established_policy)
        else:
            result=inspect(a.request) if a.command=="inspect" else cpu_targets(a.request,a.role)
            J.write_new(a.out,canonical(result)+b'\n')
        print(json.dumps({"out":pin(a.out),"production_ready":False,"result":result.get("full_target_preflight_passed",False)}));return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(json.dumps({"status":"refused","reason":str(exc),"production_ready":False,"training_launched":False}));return 2


if __name__=="__main__":raise SystemExit(main())
