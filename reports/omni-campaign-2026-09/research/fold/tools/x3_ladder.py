#!/usr/bin/env python3
"""x3_ladder.py -- pilot X3 on the Qwen3.5-0.8B ladder: how much of a C2 4x MLP fold's damage does a GLOBAL
correction recover, and is on-policy (student-prefix) distillation worth its cost over off-policy KD?

Critique of 2026-10-14 ("Fold 1"): arms KD, KD->OPD with the horizon controller (from X2d) and OPD-only, graded at
0.25 / 0.5 / 1 / 2M trained tokens so a slope, not a single point, is judged.  Kill rule: no positive content slope
across the rungs.  This is a RECOVERY result, reported apart from fold-only numbers (critique fix 9).

  prepare-data (CPU)  re-render D2 train documents (27B-tokenized packed rows -> text -> 0.8B chat template, thinking
                      off) into KD documents and OPD prompts; screen every one against the eval surface of fold_eval
                      prepare; write the training-group list the retention scorer checks.
  train        (GPU)  one arm.  Teacher = the parent (fp32, co-resident; no cache needed at 0.8B), student = the
                      folded child; ONLY the folded component (all 24 trunk gate/up/down projections) trains, full
                      rank (peft is absent on the Pascal venv; at 0.8B the whole MLP set is 66M parameters).
                        kd : forward KL on the teacher's top-64 + tail bucket + 0.1 CE floor, assistant positions only
                        opd: the student samples (T 0.8) from its own prefix; loss on every sampled position =
                             JSD(beta .5) on teacher top-100 + tail + 0.3 x forward KL (top-64 + tail); rollouts cut
                             at a detected loop (periods 1-10, 32-token anchor, >= 90%, >= 64 tokens); the rollout
                             budget follows the horizon controller (256 start, 128..1024).
                      Constant LR after warm-up, so rung-to-rung differences are dose, not schedule.
                      SIGTERM or <out>/STOP: finish the step, save a resume point, exit 75 (the queue re-waits).

Codex's train_onpolicy.py route (prompt screen, packed traces, rescoring) targets the 27B D2 bank and is not
re-implemented here: this is the ladder's on-line form with the same JSD .5 divergence and the same "only the
prompt before the first assistant message" prompt rule.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_common as fc  # noqa: E402

TOOL = "x3_ladder"
VERSION = "1.0.0"
SALT = "fold-research-x3-data-v1"
TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)
EMPTY_THINK = re.compile(r"\A\s*<think>\s*</think>\s*")
EXIT_PREEMPTED = 75


# -------------------------------------------------------------------------------------------------- prepare-data

def parse_doc(text: str) -> list[dict] | None:
    """Decoded D2 document -> [system?, user, assistant] or None (anything else is skipped, not repaired)."""
    turns = TURN_RE.findall(text)
    roles = [r for r, _ in turns]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        return None
    messages = [{"role": r, "content": c} for r, c in turns]
    messages[-1]["content"] = EMPTY_THINK.sub("", messages[-1]["content"])
    if "<think>" in messages[-1]["content"] or not messages[-1]["content"].strip():
        return None
    return messages


def cmd_prepare_data(args) -> None:
    import numpy as np
    from safetensors.numpy import load_file
    started = time.time()
    out = Path(args.out)
    if (out / "data_receipt.json").exists():
        raise SystemExit(f"{out}/data_receipt.json exists; frozen data is never rewritten")
    from transformers import AutoTokenizer
    tok27 = AutoTokenizer.from_pretrained(args.tok27)
    tok08, eos08 = fc.load_tokenizer(args.tok08)
    im_end = tok08.convert_tokens_to_ids("<|im_end|>")

    shards = sorted(Path(args.shards).glob("text-*.safetensors"))
    flat = np.concatenate([load_file(str(s))["input_ids"].reshape(-1) for s in shards])
    docs = fc.read_jsonl(args.documents)
    surface = fc.read_jsonl(args.surface)
    screen = fc.Screen()
    for item in surface:
        screen.add(item["source"], item["id"], item["text"])
    retention_ids = {item["id"][len("ret:"):] for item in surface if item["source"] == "retention64"}

    docs = [d for d in docs if d.get("render") == "chat" and d["kept"] == d["length"]]
    docs.sort(key=lambda d: fc.salted_rank(SALT, d["uid"]))
    kd, opd, skipped, hits = [], [], {"parse": 0, "prompt_long": 0, "retention_record": 0}, []
    kd_tokens = 0
    for d in docs:
        if kd_tokens >= args.kd_tokens and len(opd) >= args.opd_prompts:
            break
        if str(d.get("record_id")) in retention_ids:
            skipped["retention_record"] += 1
            continue
        text = tok27.decode(flat[d["start"]:d["end"]].tolist(), skip_special_tokens=False)
        messages = parse_doc(text)
        if messages is None:
            skipped["parse"] += 1
            continue
        found = screen.hits("\n".join(m["content"] for m in messages))
        if found:
            hits.append({"uid": d["uid"], "record_id": d.get("record_id"), "hits": found[:5]})
            continue
        prompt = fc.chat_ids(tok08, messages[:-1])
        if len(prompt) > args.max_prompt:
            skipped["prompt_long"] += 1
            continue
        row = {"uid": d["uid"], "record_id": d.get("record_id"), "route": d.get("route"), "dataset_id": d.get("dataset_id")}
        if kd_tokens < args.kd_tokens:
            content = tok08.encode(messages[-1]["content"], add_special_tokens=False)
            ids = (prompt + content + [im_end])[:args.max_len]
            n_loss = len(ids) - len(prompt)
            kd.append({**row, "ids": ids, "loss_start": len(prompt) - 1, "loss_tokens": n_loss,
                       "truncated": len(prompt) + len(content) + 1 > args.max_len})
            kd_tokens += n_loss
        elif len(opd) < args.opd_prompts and len(prompt) <= args.max_opd_prompt:
            opd.append({**row, "prompt_ids": prompt})
    if kd_tokens < args.kd_tokens or len(opd) < args.opd_prompts:
        raise SystemExit(f"pool exhausted: kd {kd_tokens} tokens, opd {len(opd)} prompts")
    groups = sorted({f"carve-eval:{r['record_id']}" for r in kd + opd if r.get("record_id") is not None})
    kd_sha = fc.write_jsonl(out / "kd.jsonl", kd)
    opd_sha = fc.write_jsonl(out / "opd.jsonl", opd)
    groups_sha = fc.write_json(out / "training_groups.json", groups)
    fc.write_jsonl(out / "screen_hits.jsonl", hits)
    receipt = {"schema": "fold-research.x3-data.v1", "tool": TOOL, "version": VERSION, "created_utc": _utc(),
               "inputs": {"shards": [{"path": str(s), "sha256": fc.file_sha256(s)} for s in shards],
                          "documents": {"path": args.documents, "sha256": fc.file_sha256(args.documents)},
                          "surface": {"path": args.surface, "sha256": fc.file_sha256(args.surface)},
                          "tok27_sha256": fc.file_sha256(Path(args.tok27) / "tokenizer.json"),
                          "tok08_sha256": fc.file_sha256(Path(args.tok08) / "tokenizer.json")},
               "render": "decode with the 27B tokenizer, parse [system?] user / assistant, drop the empty think block, "
                         "re-render the prompt with the 0.8B chat template (thinking off, generation prompt) and append "
                         "the assistant content + <|im_end|>; only uncut single-exchange chat documents",
               "kd": {"docs": len(kd), "loss_tokens": kd_tokens, "truncated": sum(r["truncated"] for r in kd),
                      "routes": _count(kd), "sha256": kd_sha},
               "opd": {"prompts": len(opd), "routes": _count(opd), "sha256": opd_sha},
               "training_groups": {"n": len(groups), "sha256": groups_sha},
               "screen": {**screen.receipt(), "docs_hit": len(hits)}, "skipped": skipped,
               "wall_s": round(time.time() - started, 1), "peak_rss_gib": fc.peak_rss_gib(), "tools": fc.own_shas()}
    fc.write_json(out / "data_receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("kd", "opd", "skipped")} | {"screen_docs_hit": len(hits)}))


def _count(rows):
    out = {}
    for r in rows:
        out[r.get("route")] = out.get(r.get("route"), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------------------------------------- train

def _cuda_gib() -> float:
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0.0


class Stop:
    requested = False


def _on_term(signum, frame):  # noqa: ARG001
    Stop.requested = True


def kd_doc_loss(teacher, student, row, device, k, ce_floor):
    import torch
    ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
    pos = torch.arange(row["loss_start"], len(row["ids"]) - 1, device=device)
    targets = ids[0, pos + 1]
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, logits_to_keep=pos).logits[0]
    s_logits = student(input_ids=ids, logits_to_keep=pos).logits[0]
    fkl = fc.bucket_forward_kl(s_logits, t_logits, k=k)
    ce = torch.nn.functional.cross_entropy(s_logits.float(), targets, reduction="none")
    return (fkl + ce_floor * ce).sum(), int(pos.numel()), float(fkl.mean()), float(ce.mean())


def opd_seq_loss(teacher, student, prompt, gen, device, k_jsd, beta, k_fkl, w_fkl):
    import torch
    seq = prompt + gen
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    pos = torch.arange(len(prompt) - 1, len(seq) - 1, device=device)
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, logits_to_keep=pos).logits[0]
    s_logits = student(input_ids=ids, logits_to_keep=pos).logits[0]
    jsd = fc.bucket_jsd(s_logits, t_logits, k=k_jsd, beta=beta)
    fkl = fc.bucket_forward_kl(s_logits, t_logits, k=k_fkl)
    return (jsd + w_fkl * fkl).sum(), int(pos.numel()), float(jsd.mean()), float(fkl.mean())


def cmd_train(args) -> None:
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rungs = sorted(int(x) for x in args.rungs.split(","))
    done_marker = out / "ARM_DONE.json"
    if done_marker.exists():
        print(f"SKIP {done_marker} exists")
        return
    signal.signal(signal.SIGTERM, _on_term)
    fc.verify_vendor_pins()
    torch.manual_seed(args.seed)
    device = args.device
    tok, eos = fc.load_tokenizer(args.child)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    teacher, t_load = fc.load_model(args.parent, device=device, dtype="float32")
    resume = out / "resume"
    init_overlay = args.init_overlay
    state = None
    if (resume / "state.json").exists():
        state = json.loads((resume / "state.json").read_text(encoding="utf-8"))
        init_overlay = str(resume / "mlp.safetensors")
    student, s_load = fc.load_model(args.child, device=device, dtype="float32", overlay=init_overlay)
    for p in student.parameters():
        p.requires_grad_(False)
    params = fc.mlp_parameters(student)
    for p in params.values():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in params.values())
    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    controller = fc.HorizonController(start=args.budget_start, low=args.budget_low, high=args.budget_high)
    data = fc.read_jsonl(Path(args.data) / ("kd.jsonl" if args.arm == "kd" else "opd.jsonl"))
    tokens, step, cursor, saved = 0, 0, 0, []
    if state is not None:
        opt.load_state_dict(torch.load(resume / "optim.pt", map_location=device))
        tokens, step, cursor, saved = state["tokens"], state["step"], state["cursor"], state["saved"]
        controller.load(state["controller"])
        torch.set_rng_state(torch.tensor(state["cpu_rng"], dtype=torch.uint8))
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.tensor(state["cuda_rng"], dtype=torch.uint8))
    identity = {"parent": fc.checkpoint_identity(args.parent, None),
                "child": fc.checkpoint_identity(args.child, args.init_overlay),
                "data": json.loads((Path(args.data) / "data_receipt.json").read_text(encoding="utf-8"))["kd" if args.arm == "kd" else "opd"]}
    config = {k: v for k, v in vars(args).items() if k != "func"}
    fc.write_json(out / "arm_config.json", {"tool": TOOL, "version": VERSION, "config": config, "identity": identity,
                                             "trainable": {"tensors": len(params), "parameters": n_train},
                                             "teacher_load": t_load, "student_load": s_load, "tools": fc.own_shas(),
                                             "vendor": fc.verify_vendor_pins(), "resumed": state is not None})
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    print(f"[x3] arm={args.arm} trainable={n_train:,} start tokens={tokens} cursor={cursor} rungs={rungs}", flush=True)
    t_arm = time.time()

    def lr_at(s):
        return args.lr * min(1.0, (s + 1) / max(1, args.warmup))

    def save_resume():
        resume.mkdir(exist_ok=True)
        fc.save_overlay(student, resume / "mlp.safetensors")
        torch.save(opt.state_dict(), resume / "optim.pt")
        fc.write_json(resume / "state.json", {"tokens": tokens, "step": step, "cursor": cursor, "saved": saved,
                                               "controller": controller.state(),
                                               "cpu_rng": torch.get_rng_state().tolist(),
                                               "cuda_rng": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None})

    while tokens < rungs[-1]:
        if Stop.requested or (out / "STOP").exists():
            save_resume()
            print(f"[x3] PREEMPTED at tokens={tokens} step={step}; resume point saved", flush=True)
            log.close()
            sys.exit(EXIT_PREEMPTED)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step, "lr": lr_at(step)}
        if args.arm == "kd":
            step_tokens, parts, fkls, ces = 0, 0, [], []
            while step_tokens < args.kd_step_tokens:
                row = data[cursor % len(data)]
                cursor += 1
                loss, n, fkl, ce = kd_doc_loss(teacher, student, row, device, args.kd_k, args.ce_floor)
                (loss / args.kd_step_tokens).backward()
                step_tokens += n
                parts += 1
                fkls.append(fkl * n)
                ces.append(ce * n)
            rec.update(tokens=step_tokens, docs=parts, fkl=sum(fkls) / step_tokens, ce=sum(ces) / step_tokens)
        else:
            batch = [data[(cursor + i) % len(data)] for i in range(args.opd_batch)]
            cursor += args.opd_batch
            budget = controller.budget
            tg = time.time()
            got = fc.generate(student, [b["prompt_ids"] for b in batch], max_new=budget, eos_ids=eos, pad_id=pad,
                              batch_size=args.opd_batch, do_sample=True, temperature=args.temperature)
            gen_s = time.time() - tg
            seqs, loops, caps, gen_tokens = [], 0, 0, 0
            for b, g in zip(batch, got):
                ids = g["generated_ids"]
                gen_tokens += len(ids)
                content = ids[:-1] if g["stop_reason"] == "eos" else ids
                cut = fc.detect_loop(content)
                if cut is not None:
                    loops += 1
                    ids = content[:cut]
                elif g["stop_reason"] == "max_tokens":
                    caps += 1
                if ids:
                    seqs.append((b["prompt_ids"], ids))
            step_tokens = sum(len(gen) for _, gen in seqs)
            jsds, fkls = [], []
            for prompt, gen in seqs:
                loss, n, jsd, fkl = opd_seq_loss(teacher, student, prompt, gen, device, args.jsd_k, args.jsd_beta,
                                                 args.kd_k, args.fkl_weight)
                (loss / max(1, step_tokens)).backward()
                jsds.append(jsd * n)
                fkls.append(fkl * n)
            new_budget = controller.update(len(batch), loops, caps)
            rec.update(tokens=step_tokens, rollouts=len(batch), loops=loops, caps=caps, budget=budget,
                       next_budget=new_budget, gen_tokens=gen_tokens, gen_tok_s=round(gen_tokens / max(gen_s, 1e-6), 1),
                       jsd=sum(jsds) / max(1, step_tokens), fkl=sum(fkls) / max(1, step_tokens))
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), args.clip)
        opt.step()
        tokens += rec["tokens"]
        step += 1
        rec.update(total_tokens=tokens, grad_norm=float(grad_norm), step_s=round(time.time() - t0, 2),
                   mem_gib=_cuda_gib())
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0:
            print(f"[x3] {json.dumps(rec)}", flush=True)
        for rung in rungs:
            if tokens >= rung and rung not in [s["rung"] for s in saved]:
                path = out / f"rung-{rung}.safetensors"
                sha = fc.save_overlay(student, path)
                saved.append({"rung": rung, "tokens": tokens, "step": step, "overlay": str(path), "sha256": sha,
                              "wall_s": round(time.time() - t_arm, 1), "controller": controller.state()})
                fc.write_json(out / f"rung-{rung}.json", saved[-1])
                print(f"[x3] RUNG {rung} saved at tokens={tokens} step={step} sha={sha[:16]}", flush=True)
    log.close()
    fc.write_json(done_marker, {"arm": args.arm, "tokens": tokens, "steps": step, "rungs": saved,
                                "controller_history_tail": controller.history[-50:], "peak_rss_gib": fc.peak_rss_gib(),
                                "max_cuda_gib": _cuda_gib(), "finished_utc": _utc()})
    print(f"[x3] ARM DONE {args.arm} tokens={tokens} steps={step}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("prepare-data")
    d.add_argument("--shards", required=True)
    d.add_argument("--documents", required=True)
    d.add_argument("--tok27", required=True)
    d.add_argument("--tok08", required=True)
    d.add_argument("--surface", required=True)
    d.add_argument("--kd-tokens", type=int, default=2_200_000)
    d.add_argument("--opd-prompts", type=int, default=8000)
    d.add_argument("--max-len", type=int, default=1024)
    d.add_argument("--max-prompt", type=int, default=768)
    d.add_argument("--max-opd-prompt", type=int, default=512)
    d.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=("kd", "opd"), required=True)
    t.add_argument("--parent", required=True)
    t.add_argument("--child", required=True)
    t.add_argument("--init-overlay", default=None)
    t.add_argument("--data", required=True)
    t.add_argument("--rungs", default="250000,500000,1000000,2000000")
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=20261015)
    t.add_argument("--kd-k", type=int, default=64)
    t.add_argument("--ce-floor", type=float, default=0.1)
    t.add_argument("--kd-step-tokens", type=int, default=8192)
    t.add_argument("--opd-batch", type=int, default=16)
    t.add_argument("--temperature", type=float, default=0.8)
    t.add_argument("--jsd-k", type=int, default=100)
    t.add_argument("--jsd-beta", type=float, default=0.5)
    t.add_argument("--fkl-weight", type=float, default=0.3)
    t.add_argument("--budget-start", type=int, default=256)
    t.add_argument("--budget-low", type=int, default=128)
    t.add_argument("--budget-high", type=int, default=1024)
    args = p.parse_args(argv)
    {"prepare-data": cmd_prepare_data, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
