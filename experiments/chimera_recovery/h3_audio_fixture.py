"""Package actual standalone H3 source/probe/tensor identities, CPU file I/O.

This is a bounded diagnostic join, never a native batch or complete cache seal.
The native packer independently validates tensor values and all source rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from . import audio_admission as A
from .h3_audio_chart import validate_encoder_proof

TEACHER_SHA256="8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
STAGE_C_SHA256="fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf"
COMFY_PINS={"comfy/latent_formats.py":"f13259cbd815a339a474976522eb6aae10fb16c8397f13b41e50ddd107f8caf8",
            "comfy/ldm/minimax/audio_vae.py":"3e00d398b25d68dc39ca0eb589fc7549f3693b6bfaf08ef920e7ad2a7a803cba",
            "comfy/sd.py":"7efb10f22252446069a0b88f0a8bc2d94713696d9f946d7757a0981ae8101b58"}


def tensor_inventory(path,proof):
    if A.file_sha(path) != proof["tensors"]["sha256"]:
        raise ValueError("actual H3 tensor file differs from encoder receipt")
    size=path.stat().st_size
    with path.open("rb") as stream:
        raw=stream.read(8)
        if len(raw) != 8:
            raise ValueError("missing safetensors header")
        length=struct.unpack("<Q",raw)[0]
        if not 2 <= length <= min(size-8,1024*1024):
            raise ValueError("unbounded/invalid safetensors header")
        body=json.loads(stream.read(length))
    result={k:v for k,v in body.items() if k!="__metadata__"}
    if set(result) != set(proof["tensors"]["keys"]):
        raise ValueError("H3 tensor inventory differs from receipt")
    t=proof["input_contract"]["expected_latent_frames"]
    n=proof["input_contract"]["target_samples"]
    expected={"input_waveform_stereo32k":[1,2,n],"decoded_waveform":[1,2,n],
              "x0_raw_audio":[1,32,2,t],"x0_model_audio":[1,32,2,t],
              "packed_raw_audio":[2*t,32],"packed_model_audio":[2*t,32]}
    spans=[]
    for name,entry in result.items():
        shape=entry.get("shape");offset=entry.get("data_offsets")
        if (entry.get("dtype") != "F32" or shape != expected.get(name) or not isinstance(offset,list) or len(offset) != 2 or
            any(type(i) is not int or i<0 for i in offset) or offset[1]<offset[0]):
            raise ValueError("actual H3 tensor dtype/shape/offset contract differs")
        count=1
        for dim in shape:
            count*=dim
        if offset[1]-offset[0] != count*4 or offset[1] > size-8-length:
            raise ValueError("actual H3 tensor byte span is incomplete")
        spans.append(tuple(offset))
    spans.sort()
    if spans[0][0] != 0 or spans[-1][1] != size-8-length or any(a[1]!=b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError("tensor file contains unsealed gaps/overlaps/trailing bytes")
    return result


def package(*,source,probe,tensors,output,admission=None,expected_admission_sha256=None,
            derivation=None,original=None):
    if output.exists():
        raise ValueError("preserve old fixture evidence; use a new output file")
    proof=json.loads(probe.read_text(encoding="utf-8"))
    validate_encoder_proof(proof)
    if (proof.get("teacher_sha256_before") != TEACHER_SHA256 or proof.get("sources",{}).get("stage_c_sha256") != STAGE_C_SHA256 or
        any(proof.get("sources",{}).get("comfy_sources",{}).get(k) != v for k,v in COMFY_PINS.items())):
        raise ValueError("encoder source/teacher differs from reviewed native pins")
    measure,_=A.pcm(source)
    contract=proof["input_contract"]
    if (A.file_sha(source) != contract.get("asset_sha256") or measure["pcm_sha256"] != contract.get("pcm_sha256") or
        measure["sample_rate"] != contract.get("source_rate") or measure["channels"] != contract.get("source_channels") or
        measure["samples"] != contract.get("source_samples") or measure["duration_seconds"] != contract.get("duration_seconds") or
        proof.get("probe_passed") is not True or proof.get("teacher_unchanged") is not True or
        proof.get("teacher_sha256_before") != proof.get("teacher_sha256_after")):
        raise ValueError("actual source/encoder/teacher identity join differs")
    header=tensor_inventory(tensors,proof)
    joined=[];admission_pin=None
    if admission is not None:
        A.pinned({"path":str(admission),"sha256":expected_admission_sha256},"audio admission")
        body=json.loads(admission.read_text(encoding="utf-8"))
        if (body.get("schema") != A.SCHEMA or body.get("sealed") is not True or
            any(body.get(k) is not True for k in ("complete_source_traversal","frozen_exclusions_verified","connected_groups_verified")) or
            A.digest({k:v for k,v in body.items() if k!="manifest_sha256"}) != body.get("manifest_sha256")):
            raise ValueError("audio admission seal is not complete/global")
        ref=body["outputs"]["admission"];path=A.pinned(ref,"audio admission rows")
        recorded=list(A.rows(path))
        if len(recorded)!=ref["rows"]:
            raise ValueError("complete admission row count differs")
        joined=[r for r in recorded if r.get("asset_sha256")==contract["asset_sha256"]]
        if len(joined)!=1 or joined[0].get("id")!=joined[0].get("item_id"):
            raise ValueError("actual encoder source admission join missing/ambiguous")
        admission_pin={"path":str(admission.resolve()),"sha256":A.file_sha(admission),"manifest_sha256":body["manifest_sha256"]}
    derived=None
    if derivation is not None or original is not None:
        if derivation is None or original is None:
            raise ValueError("derived waveform requires both original and explicit derivation receipt")
        derived=A.validate_tail_derivation(json.loads(derivation.read_text(encoding="utf-8")),original,source)
        derived["receipt_sha256"]=A.file_sha(derivation)
    result={"schema":"chimera_recovery.h3_audio_fixture_evidence.v1","diagnostic_only":True,"complete_native_cache":False,
            "native_train_ready":False,"training_started":False,"code_sha256":A.file_sha(Path(__file__)),
            "source":{"path":str(source.resolve()),"sha256":A.file_sha(source),**measure},
            "encoder_probe":{"path":str(probe.resolve()),"sha256":A.file_sha(probe),"input_contract":contract},
            "actual_tensors":{"path":str(tensors.resolve()),"sha256":A.file_sha(tensors),"header":header},
            "teacher_sha256":proof["teacher_sha256_before"],"source_pins":proof["sources"],
            "admission":admission_pin,"joined_source_rows":[{k:r[k] for k in ("id","source_row_sha256","asset_sha256","split","recovery_group_id","train_admitted")} for r in joined],
            "explicit_tail_derivation":derived,
            "limitations":["File hashes/header/source coverage are verified here; native packer must validate actual finite tensor values and recorded raw/model/pack transformations.",
                           "One diagnostic source never seals the complete24-source encoder/cache inventory.",
                           "Standalone XYZ uses a separate dated observation-chart amendment; native parent XYZ parity is not claimed."]}
    result["manifest_sha256"]=A.digest(result)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(A.canonical(result)+b"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","probe","tensors","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("admission","derivation","original"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--expected-admission-sha256")
    args=vars(p.parse_args(argv));r=package(**args)
    print(json.dumps({"manifest_sha256":r["manifest_sha256"],"native_train_ready":False,"joined_sources":len(r["joined_source_rows"])}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
