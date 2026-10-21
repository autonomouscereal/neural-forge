"""ingest.s8_media -- OMNI1's S8 media checks -> two eval runs (media denoise smoke, QVL describe).

Sources ($OC/composite/s8/media_out/, read-only):
  s8_media_forward.json          s8_media_smoke: one charted-block denoise step per segment (ace/h3/qimg/hy3d),
                                 latent metrics vs the parent's ideal; also names the QVL answers file
  s8_media_decode.*.json         s8_media_decode (ComfyUI runtime): decoded wav/png/glb per segment + metrics
  qvl/answers.json               greedy 'Describe this image.' answers: image / zeroed-features / no-image
  <modality>/latents.json        the latents file each modality wrote (eval_artifact kind 'latent')

  'eval:s8-media:<started>:<forward sha12>'  suite s8_media_smoke@<tool version>, one case per segment.
       finite = input, final hidden, last logits and emission all finite.  Machine grade
       s8_media_decode.beats_noisy_input on x0_omni_final (h3: video AND audio) -- the decode's own
       diagnostic (owner policy 2026-10-06: every number here is a diagnostic, never a gate).
  'eval:s8-qvl:<started>:<answers sha12>'    suite s8_qvl_describe@1, one case per answer; ungraded.
Each segment's prompt is the stimulus spec (unit, sigma, rows), its media the parent pair capture.
"""
from __future__ import annotations

import glob
import os
from datetime import timedelta
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "s8_media"
MODALITY = {"ace": (["text", "audio"], ["audio"]), "h3": (["text", "video", "audio"], ["video", "audio"]),
            "qimg": (["text", "image"], ["image"]), "hy3d": (["text", "3d"], ["3d"])}


def media_dir() -> str:
    return f"{C.OC}/composite/s8/media_out"


def artifact_kind(modality: str, key: str, path: str) -> str:
    if modality == "h3":
        return "video" if key.startswith("video") else "audio"
    ext = Path(path).suffix.lower()
    return {".wav": "audio", ".png": "image", ".glb": "mesh", ".obj": "mesh", ".safetensors": "latent"}.get(ext, "other")


def chimera_commit(src: str | None) -> str | None:
    if not src:
        return None
    text = C.read_text(f"{src}/COMMIT").strip()
    return text.split()[0] if text else None


def beats(dec: dict, modality: str) -> tuple[bool | None, dict]:
    if modality == "h3":
        v = ((dec.get("video_vs_noisy_input") or {}).get("x0_omni_final") or {})
        a = ((dec.get("audio_vs_noisy_input") or {}).get("x0_omni_final") or {})
        if not isinstance(v.get("beats_noisy_input"), bool) or not isinstance(a.get("beats_noisy_input"), bool):
            return None, {}
        return v["beats_noisy_input"] and a["beats_noisy_input"], {"video": v, "audio": a}
    x = ((dec.get("vs_noisy_input") or {}).get("x0_omni_final") or {})
    if not isinstance(x.get("beats_noisy_input"), bool):
        return None, {}
    return x["beats_noisy_input"], {"x0_omni_final": x, "x0_parent": (dec.get("vs_noisy_input") or {}).get("x0_parent")}


def run(conn, batch: C.Batch, args) -> None:
    K.ensure_gpus(conn)
    fpath = f"{media_dir()}/s8_media_forward.json"
    batch.saw()
    if not os.path.isfile(fpath):
        batch.skip(f"missing {fpath}")
        return
    fprov = batch.prov(fpath)
    f = C.load_json(fpath) or {}
    decode = {}
    dpaths = sorted(glob.glob(f"{media_dir()}/s8_media_decode.*.json"))
    dprov = None
    for dp in dpaths:
        d = C.load_json(dp)
        if isinstance(d, dict) and d.get("schema") == "omni-campaign.s8_media_decode.v1":
            dprov = batch.prov(dp)
            decode = d
    omni = f.get("omni1") or {}
    model = K.ensure_model_at(conn, omni.get("directory") or "", role="composite", layout="chimera.checkpoint.v1",
                              stage="OMNI1", prov=fprov, batch_id=batch.id)
    tool = f.get("tool") or {}
    tool_v = C.tool_version_for(conn, tool.get("name") or "s8_media_smoke", tool.get("version"), tool.get("sha256"))
    src = (f.get("chimera_omni") or {}).get("chimera_src")
    env_id = K.env_for(conn, torch=f.get("torch"), code_commit=chimera_commit(src),
                       env_vars={"CUDA_VISIBLE_DEVICES": f.get("cuda_visible_devices")},
                       packages={"chimera_src": src, "decode_torch": (decode.get("runtime") or {}).get("torch"),
                                 "decode_comfy_sha256": (decode.get("runtime") or {}).get("sha256")})
    started = C.parse_ts(f.get("started_utc"))
    ended = started + timedelta(seconds=f["seconds"]) if started and isinstance(f.get("seconds"), (int, float)) else None
    common_config = {"dtype": f.get("dtype"), "device": f.get("device"), "cuda_device_name": f.get("cuda_device_name"),
                     "chimera_src": src, "charted_blocks": omni.get("charted_blocks"),
                     "omni1_manifest_sha256": omni.get("manifest_sha256"), "omni1_config_sha256": omni.get("config_sha256"),
                     "policy": f.get("policy"), "argv": f.get("argv")}

    def place(run_id: int, with_decode: bool) -> None:
        K.clear_devices(conn, run_id)
        mapped = K.visible_gpus(conn, f.get("cuda_visible_devices"), None)
        for gid, detail in mapped:
            if gid is not None:
                db.add_run_device(conn, run_id, gid, role="compute", kernel_path="chimera-reference",
                                  detail={**detail, "device_name_recorded": f.get("cuda_device_name")})
        rt = decode.get("runtime") or {}
        if with_decode and rt.get("device_uuid"):
            gid = C.gpu_for(conn, uuid="GPU-" + rt["device_uuid"])
            if gid is not None:
                db.add_run_device(conn, run_id, gid, role="decode", kernel_path="comfyui",
                                  detail={"mapping": "uuid", "comfy_root": rt.get("comfy_root"), "torch": rt.get("torch")})

    # --- media denoise smoke -------------------------------------------------------------------
    mods_in, mods_out, pks = set(), set(), []
    seg_rows = []
    pairs = f.get("pairs") or {}
    for modality, block in (f.get("modalities") or {}).items():
        m_in, m_out = MODALITY.get(modality, (["text"], ["other"]))
        mods_in.update(m_in)
        mods_out.update(m_out)
        pair = pairs.get(modality) or {}
        for seg in block.get("segments") or []:
            prompt = {"task": "one denoise step through the OMNI1 charted unit", "modality": modality,
                      "unit": seg.get("unit"), "sigma": seg.get("sigma"), "prompt_tokens": seg.get("prompt_tokens"),
                      "media_rows": seg.get("media_rows"), "time_rows": seg.get("time_rows"),
                      "pair_forward": seg.get("pair_forward")}
            media = [{"kind": "pair_capture", "path": pair.get("file"), "sha256": pair.get("sha256")}] if pair else None
            pk = db.ensure_case(conn, seg["segment"], prompt, media=media, category=f"media-{modality}",
                                lane=modality, tags=["s8_media_smoke", modality], modality_in=m_in, modality_out=m_out,
                                reference={"layout": seg.get("layout"), "checks": seg.get("checks")},
                                prov=fprov, batch_id=batch.id)
            pks.append(pk)
            seg_rows.append((modality, seg, pk, block))
            batch.wrote()
    order = ["text", "image", "video", "audio", "3d"]
    suite = db.ensure_suite(conn, "s8_media_smoke", tool.get("version") or "1", status="sealed",
                            modality_in=sorted(mods_in, key=order.index), modality_out=sorted(mods_out, key=order.index),
                            default_grader="s8_media_decode.beats_noisy_input",
                            description="S8 media smoke: one denoise step per segment through each charted media "
                                        "unit of OMNI1, decoded by the parent VAEs; diagnostics only",
                            prov=fprov, batch_id=batch.id)
    K.set_membership(conn, suite, pks)
    rid = K.eval_run(conn, batch, run_key=f"eval:s8-media:{f.get('started_utc')}:{fprov[1][:12]}",
                     name="S8 media smoke OMNI1 (ace/h3/qimg/hy3d)", project="media", model=model, suite=suite,
                     engine="chimera", gen={"mode": "single-step denoise", "dtype": f.get("dtype")},
                     config={**common_config, "decode_tool": decode.get("tool"), "decode_argv": decode.get("argv"),
                             "seconds": f.get("seconds"), "peak_rss_gib": f.get("peak_rss_gib")},
                     started=started, ended=ended, wall_seconds=f.get("seconds"), tool_version_id=tool_v,
                     env_id=env_id, receipt=fprov, run_dir=media_dir(), label="S8-MEDIA")
    place(rid, with_decode=True)
    grader = K.grader_id(conn, "s8_media_decode.beats_noisy_input")
    for modality, seg, pk, block in seg_rows:
        batch.saw()
        dec = (((decode.get("modalities") or {}).get(modality) or {}).get("segments") or {}).get(seg["segment"]) or {}
        finite = all(bool(x) for x in (seg.get("input_finite"), seg.get("final_hidden_finite"),
                                       seg.get("last_position_logits_finite"),
                                       (seg.get("emission_final") or {}).get("finite")))
        res = db.upsert_eval_result(conn, rid, pk, finite=finite,
                                    raw={"forward": {k: v for k, v in seg.items() if k not in ("layout", "checks")},
                                         "decode": {k: v for k, v in dec.items() if k != "files"}},
                                    prov=fprov, batch_id=batch.id)
        if block.get("latents"):
            db.add_eval_artifact(conn, res, "latent", block["latents"], sha256=block.get("latents_sha256"),
                                 metrics={"segment": seg["segment"]})
        for key, fdesc in (dec.get("files") or {}).items():
            if isinstance(fdesc, dict) and fdesc.get("path"):
                hw = fdesc.get("hw") or [None, None]
                db.add_eval_artifact(conn, res, artifact_kind(modality, key, fdesc["path"]), fdesc["path"],
                                     sha256=fdesc.get("sha256"), nbytes=fdesc.get("bytes"),
                                     height=hw[0] if len(hw) > 0 else None, width=hw[1] if len(hw) > 1 else None,
                                     metrics={"role": key, **{k: v for k, v in fdesc.items()
                                                             if k not in ("path", "sha256", "bytes", "hw")}})
        ok, detail = beats(dec, modality)
        if ok is not None:
            db.put_grade(conn, res, grader, "pass" if ok else "fail", score=1.0 if ok else 0.0,
                         why="x0_omni_final beats the noisy input" if ok else "x0_omni_final does not beat the noisy input",
                         detail=detail, prov=dprov, batch_id=batch.id)
        batch.wrote()

    # --- QVL describe ---------------------------------------------------------------------------
    qvl = f.get("qvl") or {}
    apath = K.at(qvl.get("answers")) or f"{media_dir()}/qvl/answers.json"
    if not os.path.isfile(apath):
        batch.skip(f"missing {apath}")
        return
    aprov = batch.prov(apath)
    a = C.load_json(apath) or {}
    q_pks, rows = [], []
    for ans in a.get("answers") or []:
        prompt = [{"role": "user", "content": a.get("question")}]
        media = None
        if ans.get("image"):
            media = [{"kind": "image", "path": ans["image"], "sha256": ans.get("image_sha256"),
                      "variant": ans.get("variant"), "side": a.get("side")}]
        pk = db.ensure_case(conn, ans["case"], prompt, media=media, category="vision-describe", lane=ans.get("variant"),
                            tags=["s8_qvl_describe", ans.get("variant") or ""],
                            modality_in=["text", "image"] if media else ["text"],
                            reference={"subject_prompt": ans.get("subject_prompt")}, prov=aprov, batch_id=batch.id)
        q_pks.append(pk)
        rows.append((ans, pk))
        batch.wrote()
    qsuite = db.ensure_suite(conn, "s8_qvl_describe", "1", status="sealed", modality_in=["text", "image"],
                             description="'Describe this image.' through OMNI1's charted QVL block: image vs "
                                         "zeroed-features vs no-image controls; reader-graded (none yet)",
                             prov=aprov, batch_id=batch.id)
    K.set_membership(conn, qsuite, q_pks)
    qrid = K.eval_run(conn, batch, run_key=f"eval:s8-qvl:{f.get('started_utc')}:{aprov[1][:12]}",
                      name="S8 QVL describe OMNI1", project="media", model=model, suite=qsuite, engine="chimera",
                      gen={"cap": a.get("max_new_tokens"), "decoding": a.get("decode"), "temperature": 0,
                           "extra_stop_token_ids": a.get("stops"), "chat": a.get("chat")},
                      config={**common_config, "tokenizer_directory": a.get("tokenizer_dir"),
                              "preflight": a.get("preflight"), "seconds": a.get("seconds"),
                              "prefill_seconds": a.get("prefill_seconds")},
                      started=started, wall_seconds=a.get("seconds"), tool_version_id=tool_v, env_id=env_id,
                      receipt=aprov, run_dir=str(Path(apath).parent), label="S8-QVL", thinking_requested="off")
    place(qrid, with_decode=False)
    for ans, pk in rows:
        batch.saw()
        term = "stop" if ans.get("stopped") else "cap"
        db.upsert_eval_result(conn, qrid, pk, output_text=ans.get("answer_verbatim"), output_tokens=ans.get("new_tokens"),
                              termination=term,
                              artifact_flags=K.machine_flags(ans.get("answer_verbatim"), ans.get("new_tokens"), None, term),
                              raw={k: ans.get(k) for k in ("variant", "media_rows", "grid_thw", "stopped")},
                              prov=aprov, batch_id=batch.id)
        batch.wrote()
    if isinstance(a.get("answers_differ_between_images"), bool):
        db.put_metric(conn, qrid, "qvl/answers_differ_between_images", 1.0 if a["answers_differ_between_images"] else 0.0)
    eq = a.get("image_answer_equals_zeroed_answer")
    if isinstance(eq, list):
        db.put_metric(conn, qrid, "qvl/image_answer_equals_zeroed", float(sum(1 for x in eq if x)), denominator=float(len(eq)))
