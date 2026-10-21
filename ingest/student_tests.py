"""ingest.student_tests -- student_test.py runs (results.json + grades.json) -> suite, eval run, reader grades.

Default source: $L/scratch/student-tests-*/results.json (schema chimera-omni-student-tests-*-v1) with the
grades.json a reader wrote beside it.  For 2026-10-21 that is 12 greedy cases on the full-KD run-1 student:
8 bank_v1 cases, 3 new prompts, 1 toolcall_v1 case.

Writes
  * suite student_tests_<date>@1 (the 12 cases run, bank order of the run) and student_new_<date>@1 (the new
    prompts alone, so any model can be run on them later); bank cases are the SAME test_case rows as
    bank_v1 / toolcall_v1 (matched by content hash).
  * run 'eval:student_test:<started>:<results sha12>' on the student (engine 'qwen35-student'), its placement
    (CUDA_VISIBLE_DEVICES from the run, layer plan, kernels, peak memory) and env.
  * machine grades the run recorded: toolcall_selfgen.grade_case (tool case), student_test.code_check (code case).
  * reader grades (grader 'human:claude') from grades.json for the student, and for the parent / OMNI1
    columns onto the dossier and toolcall-bank runs the student run compared against (joined by the sha256
    of the files named in the run's argv).  A 'not run' cell writes nothing.  Run `dossiers` and
    `toolcall_bankruns` first; a missing comparison run is skipped and named on the batch.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from . import common as C
from . import evalkit as K
from .common import db

SOURCE = "student_tests"


def result_files() -> list[str]:
    override = os.environ.get("OMNI_LAB_STUDENT_TESTS", "")
    dirs = [p for p in override.split(os.pathsep) if p] if override else sorted(glob.glob(f"{C.LX}/scratch/student-tests-*"))
    return [f"{d}/results.json" for d in dirs if os.path.isfile(f"{d}/results.json")]


def argv_value(argv: list[str], flag: str) -> str | None:
    """The value of `flag` in a recorded argv (re-rooted by evalkit.at when the tree moved)."""
    for i, a in enumerate(argv or []):
        if a == flag and i + 1 < len(argv):
            return K.at(argv[i + 1])
        if a.startswith(flag + "="):
            return K.at(a.split("=", 1)[1])
    return None


def layer_types(student_dir: str) -> list[str]:
    cfg = C.load_json(f"{student_dir}/config.json") or {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    lt = text.get("layer_types") if isinstance(text, dict) else None
    return lt if isinstance(lt, list) else []


def ingest_results(conn, batch: C.Batch, rpath: str) -> None:
    r = C.load_json(rpath)
    if not isinstance(r, dict) or not str(r.get("schema", "")).startswith("chimera-omni-student-tests-"):
        batch.skip(f"{rpath}: not a student-tests results file")
        return
    rprov = batch.prov(rpath)
    run_dir = str(Path(rpath).parent)
    date = re.sub(r"[^0-9]", "", Path(run_dir).name)[-8:] or "undated"
    argv = r.get("argv") or []
    gen = r.get("generation") or {}
    load = r.get("load") or {}
    place = load.get("placement") or {}

    bank_path = argv_value(argv, "--bank")
    tool_bank = argv_value(argv, "--tool-bank")
    bank_members: dict = {}
    tool_rows: dict = {}
    if bank_path and os.path.isfile(bank_path):
        s, _ = K.ensure_bank_v1(conn, batch, bank_path)
        bank_members = K.suite_members(conn, s)
    if tool_bank and os.path.isfile(tool_bank):
        s, _ = K.ensure_toolcall_v1(conn, batch, tool_bank)
        bank_members.update(K.suite_members(conn, s))
        tool_rows = {row["id"]: row for row in K.bank_rows(tool_bank)[1]}

    # cases: bank rows resolve to the bank's own case rows; new prompts become cases with their expected answer
    case_pk: dict[str, int] = {}
    new_pks: list[int] = []
    for rec in r.get("records") or []:
        cid = rec["id"]
        tools = rec.get("tools")
        sha = db.case_content_sha256(rec.get("prompt_messages"), tools, None)
        if (cid, sha) in bank_members:
            case_pk[cid] = bank_members[(cid, sha)]
            continue
        asserts = [a[0] for a in (rec.get("code_check") or {}).get("asserts") or [] if a]
        reference = {"expected": rec.get("expected")}
        if asserts:
            reference["asserts"] = asserts
        if cid in tool_rows:
            batch.skip(f"prompt-mismatch {cid}: recorded tool prompt differs from toolcall_v1")
        pk = db.ensure_case(conn, cid, rec.get("prompt_messages"), tools=tools, reference=reference,
                            category=rec.get("category"), tags=[f"student_new_{date}"], prov=rprov, batch_id=batch.id)
        case_pk[cid] = pk
        new_pks.append(pk)
        batch.wrote()
    suite = db.ensure_suite(conn, f"student_tests_{date}", "1", status="sealed",
                            description=f"the cases student_test {r.get('tool_version')} ran on {date}: "
                                        "bank_v1 subset + new prompts + a toolcall_v1 case; reader-graded",
                            bank_sha256=gen.get("bank_file_sha256"), prov=rprov, batch_id=batch.id)
    K.set_membership(conn, suite, [case_pk[rec["id"]] for rec in r.get("records") or []])
    if new_pks:
        new_suite = db.ensure_suite(conn, f"student_new_{date}", "1", status="sealed",
                                    description=f"the new checkable prompts first used by student_test on {date}",
                                    prov=rprov, batch_id=batch.id)
        K.set_membership(conn, new_suite, new_pks)

    # model under test
    student_recorded = (r.get("student") or "").rstrip("/")
    student = (K.at(student_recorded) or "").rstrip("/")
    man_path = f"{student}/student_manifest.json"
    man = C.load_json(man_path) if os.path.isfile(man_path) else None
    base_dir = (((man or {}).get("base_identity") or {}).get("chimera_dir") or "").rstrip("/")
    base_rec = K.ledger_by_directory().get(base_dir)
    lm = load.get("manifest") or {}
    model = K.ensure_model_at(conn, student_recorded, role="student", layout="qwen3_5", family="Qwen3.8-27B-U",
                              stage="fullkd-run1" if "/fullkd/run1/" in student_recorded else "student",
                              config={"layout_note": lm.get("layout"), "tensors": lm.get("tensors"),
                                      "step": lm.get("step"), "tokens_seen": lm.get("tokens_seen"),
                                      "trainer": f"{lm.get('tool')} {lm.get('tool_version')}",
                                      "parameters_counted_live": load.get("parameters_counted_live")},
                              notes="full-KD student; not registered in the ledger",
                              lineage=[(base_rec["checkpoint_id"], "distill",
                                        {"via": "student_manifest.base_identity.chimera_dir", "step": lm.get("step")})]
                              if base_rec else [], prov=rprov, batch_id=batch.id)

    # devices: CUDA_VISIBLE_DEVICES from the run, roles from the layer plan x layer types
    launcher = C.read_text(f"{run_dir}/run_student_tests.sh")
    order = "PCI_BUS_ID" if "CUDA_DEVICE_ORDER=PCI_BUS_ID" in launcher else None
    visible = place.get("visible")
    kernels = load.get("kernels") or {}
    tool_v = C.tool_version_for(conn, r.get("tool") or "student_test", r.get("tool_version"), r.get("tool_sha256"),
                                path=argv[0] if argv else None)
    env_id = K.env_for(conn, torch=gen.get("torch_version"), transformers=gen.get("transformers_version"),
                       env_vars={"CUDA_VISIBLE_DEVICES": visible, "CUDA_DEVICE_ORDER": order},
                       packages={"oc_calib_sha256": gen.get("oc_calib_sha256"),
                                 "toolcall_selfgen_sha256": gen.get("toolcall_selfgen_sha256")})
    thinking = gen.get("thinking") or {}
    started, ended = C.parse_ts(r.get("started_utc")), C.parse_ts(r.get("ended_utc"))
    rid = K.eval_run(conn, batch, run_key=f"eval:student_test:{r.get('started_utc')}:{rprov[1][:12]}",
                     name=f"student_test {date} {Path(student_recorded).parent.parent.name}/{Path(student_recorded).name}",
                     project="chimera-omni", model=model, suite=suite, engine="qwen35-student",
                     gen={"cap": gen.get("maximum_new_tokens"), "decoding": gen.get("decoding"), "temperature": 0,
                          "eos_token_id": gen.get("eos_token_id"), "extra_stop_token_ids": gen.get("extra_stop_token_ids"),
                          "ledger_band_logit_mask": gen.get("ledger_band_logit_mask"),
                          "ledger_band_begin": gen.get("ledger_band_begin"), "dtype": gen.get("dtype"),
                          "runner": gen.get("runner")},
                     config={"dtype": gen.get("dtype"), "argv": argv, "student": student_recorded,
                             "tokenizer_directory": gen.get("tokenizer_directory"), "kernels": kernels,
                             "placement_plan": place.get("plan"), "cards_at_start": r.get("cards_at_start"),
                             "finite_forward": r.get("finite_forward"), "decode_preflight": r.get("decode_preflight"),
                             "strict_key_check": load.get("strict_key_check"), "purpose": r.get("purpose")},
                     started=started, ended=ended, wall_seconds=C.seconds_between(started, ended),
                     tool_version_id=tool_v, env_id=env_id, receipt=rprov, run_dir=run_dir, owner="claude",
                     label=f"student-tests-{date}", tokenizer_sha256=gen.get("tokenizer_directory_sha256"),
                     bank_file_sha256=gen.get("bank_file_sha256"), thinking_requested=thinking.get("requested"),
                     thinking_took_effect=thinking.get("took_effect"))
    types = layer_types(student)
    plan = place.get("plan") or {}
    peaks = r.get("peak_allocated_gib") or []
    K.clear_devices(conn, rid)
    for (gid, detail), dev in zip(K.visible_gpus(conn, visible, order), place.get("devices") or []):
        n = detail["cuda"]
        layers = sorted(int(k.rsplit(".", 1)[1]) for k, v in plan.items() if k.startswith("model.layers.") and v == n)
        kinds = {types[i] for i in layers if i < len(types)}
        role = ("full-attn" if kinds == {"full_attention"} else "linear-attn" if kinds == {"linear_attention"}
                else "compute")
        kpath = {"full-attn": f"sdpa-{dev.get('capability')}",
                 "linear-attn": f"fla-triton-{dev.get('capability')}"}.get(role, "mixed")
        if gid is None:
            batch.skip(f"student run: no gpu row for visible device {detail['visible']}")
            continue
        peak = peaks[n] if n < len(peaks) and isinstance(peaks[n], (int, float)) else None
        db.add_run_device(conn, rid, gid, role=role, kernel_path=kpath, kernels=kernels,
                          layers=(layers[0], layers[-1] + 1) if layers and layers == list(range(layers[0], layers[-1] + 1)) else None,
                          mem_mib_peak=int(peak * 1024) if peak is not None else None,
                          detail={**detail, "layers": layers, "other_modules": sorted(k for k, v in plan.items()
                                                                                         if v == n and not k.startswith("model.layers.")),
                                  "card": dev})

    tool_grader = K.grader_id(conn, "toolcall_selfgen.grade_case")
    code_grader = K.grader_id(conn, "student_test.code_check")
    for rec in r.get("records") or []:
        batch.saw()
        form = rec.get("form") or {}
        term = K.termination_of(rec.get("termination"), rec.get("runtime_error"))
        wall = rec.get("wall_time_seconds")
        res = db.upsert_eval_result(
            conn, rid, case_pk[rec["id"]], output_text=rec.get("response_verbatim"),
            output_tokens=rec.get("generated_tokens"), termination=term, form=form.get("form_box"),
            artifact_flags=K.machine_flags(rec.get("response_verbatim"), rec.get("generated_tokens"),
                                           rec.get("loop_detected"), term, form.get("form_box")),
            latency_ms=int(wall * 1000) if isinstance(wall, (int, float)) else None, tok_s=rec.get("decode_tok_per_s"),
            raw={k: rec.get(k) for k in ("prompt_tokens", "response_sha256", "generated_token_ids_sha256",
                                         "generated_ids_match_runner_sha", "prefill_seconds", "decode_steps",
                                         "end_to_end_tok_per_s", "s8_prompt_tokens", "prompt_tokens_match_s8", "form",
                                         "code_check", "tool_grade") if rec.get(k) is not None},
            prov=rprov, batch_id=batch.id)
        batch.wrote()
        tg = rec.get("tool_grade")
        if isinstance(tg, dict) and isinstance(tg.get("passed"), bool):
            db.put_grade(conn, res, tool_grader, "pass" if tg["passed"] else "fail", score=1.0 if tg["passed"] else 0.0,
                         why="; ".join(tg.get("problems") or []) or None, detail={"calls": tg.get("calls")},
                         prov=rprov, batch_id=batch.id)
        cc = rec.get("code_check")
        if isinstance(cc, dict) and isinstance(cc.get("of"), int) and cc["of"] > 0:
            frac = (cc.get("passed") or 0) / cc["of"]
            db.put_grade(conn, res, code_grader, "pass" if frac == 1.0 else "fail", score=frac,
                         why=f"{cc.get('passed')}/{cc['of']} asserts pass", detail={"asserts": cc.get("asserts")},
                         prov=rprov, batch_id=batch.id)

    # reader grades
    gpath = f"{run_dir}/grades.json"
    g = C.load_json(gpath) if os.path.isfile(gpath) else None
    if not isinstance(g, dict):
        return
    gprov = batch.prov(gpath)
    if g.get("results_json_sha256") not in (None, rprov[1]):
        batch.skip(f"{gpath}: graded a different results.json ({str(g.get('results_json_sha256'))[:12]}); not attached")
        return
    others = {}
    for column, flag, prefix in (("parent", "--parent-dossier", "eval:dossier:"),
                                 ("omni1", "--omni1-dossier", "eval:dossier:")):
        p = argv_value(argv, flag)
        others[column] = K.run_id_by_receipt_sha(conn, db.file_sha256(p), prefix) if p and os.path.isfile(p) else None
    tp = argv_value(argv, "--parent-tool-responses")
    parent_tool_run = K.run_id_by_receipt_sha(conn, db.file_sha256(tp), "eval:toolcall-bank:") if tp and os.path.isfile(tp) else None
    for cid, cols in (g.get("cases") or {}).items():
        for column, entry in cols.items():
            batch.saw()
            if str(entry.get("grade", "")).strip().lower() == "not run":
                continue                      # a blank cell: no result exists to grade
            if column == "student":
                target = rid
            elif cid in tool_rows and column == "parent":
                target = parent_tool_run
            else:
                target = others.get(column)
            if target is None:
                batch.skip(f"reader grade {cid}/{column}: comparison run not ingested")
                continue
            res = K.result_id(conn, target, cid)
            if res is None:
                batch.skip(f"reader grade {cid}/{column}: no result for this case in run {target}")
                continue
            K.put_reader_grade(conn, batch, res, "human:claude", entry, gprov)


def run(conn, batch: C.Batch, args) -> None:
    K.ensure_gpus(conn)
    for rpath in result_files():
        batch.saw()
        ingest_results(conn, batch, rpath)
