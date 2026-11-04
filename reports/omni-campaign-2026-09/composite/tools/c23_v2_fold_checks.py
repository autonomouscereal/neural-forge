#!/usr/bin/env python3
"""C23 -- behaviour checks of a c21 fold child against a random-init and an embed-only control.

V2_FOLD_INIT_PLAN_2026-10-21 section 8, the proxy subset (the 2B's F0-F8 need the media folds this
build does not have).  The rules are read from a PASS_RULES file that is logged in omni_lab BEFORE
the fold run it judges; this tool applies them as written and never edits them.

    R0  structural      the child loads (sha-verified), validates, every tensor is finite, its parameter
                        count equals the profile's meta build, the c21 receipt names a fold method for
                        every parameter, carries no copy, enforced the 4x rule and every selection
                        component clears it from shapes
    R1  knowledge       next-token NLL on the held corpus rows (never fitted): the child's token-weighted
                        mean is below BOTH controls' and the child is below both on >= `min_rows` rows
    R2  decoded         greedy continuations of the smoke prompts (chat template) and of each held row's
                        first tokens; graded afterwards by a reader (collapse-only bar, Cyril's
                        artifact-tolerant read); the machine pre-screen (distinct-token ratio, loop flag,
                        letter fraction) is recorded, never a grade

Controls (plan F8): `random-init` is `ChimeraOmniModel(cfg)` under a recorded seed (the module's own
init); `embed-only` is the child with every block's `mixer.out` and `ffn.down` zeroed, i.e. the folded
vocabulary and norms with no folded layer writing anything.  It separates what the folded LAYERS
carry from what the charted vocabulary alone gives.  No placement control is built (the ruling).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c21_v2_fold as c21  # noqa: E402  (resolves CHIMERA_SRC the same way)
from chimera_omni import conformance as conf  # noqa: E402
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint  # noqa: E402
from chimera_omni.modeling import ChimeraOmniModel  # noqa: E402

TOOL_NAME = "c23_v2_fold_checks"
TOOL_VERSION = "1.0.1"   # 1.0.1 unigram reference column; tool sha hashed at import
TOOL_SHA256 = c21._file_sha256_at_import(__file__)
REPORT_SCHEMA = "chimera_omni.composite.v2_fold_checks.v1"
SUITE = {"name": "v2_fold_proxy_text", "version": "1",
         "description": "v2 fold-init proxy checks (V2_FOLD_INIT_PLAN 8, proxy subset): held omni-evidence rows "
                        "(next-token NLL vs a random-init and an embed-only control) and greedy continuations of "
                        "the 3 sealed smoke prompts and of each held row's opening tokens (reader-graded, "
                        "collapse-only)"}
READER = "reader.collapse_only"
GRADERS = {READER: {"kind": "human", "scale": "correct/partial/miss/none",
                    "description": "reader grade of a decoded continuation under Cyril's artifact-tolerant, "
                                   "collapse-only bar: on-topic language outside loops/termination artifacts"}}


def log(message: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


@torch.no_grad()
def held_nll(model: ChimeraOmniModel, calib: c21.Calibration, device: torch.device, chunk: int = 512) -> dict[str, Any]:
    """Token-weighted next-token NLL per held row, through the model's real trunk (AttnRes and all)."""

    base = int(model.cfg.ledger_base_vocab_size)
    per_row, total, count = {}, 0.0, 0
    for r in calib.held_rows:
        length = int(calib.valid[r].sum())
        ids = calib.input_ids[r:r + 1, :length].to(device)
        valid = torch.ones_like(ids, dtype=torch.bool)
        x, _, _ = model._forward_trunk(model.embed(ids), valid)
        hidden = model.norm(x)[0, :-1]
        labels = ids[0, 1:]
        nll = 0.0
        for s in range(0, hidden.shape[0], chunk):
            logits = model.lm_head(hidden[s:s + chunk]).float()[:, :base]   # the ledger band is masked, as decoding does
            nll += float(torch.nn.functional.cross_entropy(logits, labels[s:s + chunk], reduction="sum"))
        per_row[str(r)] = nll / (length - 1)
        total += nll
        count += length - 1
    return {"mean": total / max(count, 1), "tokens": count, "per_row": per_row}


@torch.no_grad()
def greedy(model: ChimeraOmniModel, prompt_ids: list[int], new_tokens: int, eos: set[int],
           device: torch.device) -> tuple[list[int], str]:
    ids = list(prompt_ids)
    out: list[int] = []
    for _ in range(new_tokens):
        t = torch.tensor([ids], device=device)
        logits = model(t, ledger_band_logit_mask=True)["logits"][0, -1].float()
        nxt = int(torch.argmax(logits))
        out.append(nxt)
        ids.append(nxt)
        if nxt in eos:
            return out, "eos"
    return out, "cap"


def unigram_reference(calib: c21.Calibration, vocab: int) -> dict[str, Any]:
    """Held-row NLL of the fit rows' add-one unigram: what token frequency alone buys (a diagnostic column)."""

    fit = calib.mask(calib.fit_rows)
    counts = torch.bincount(calib.input_ids[fit], minlength=vocab).double() + 1.0
    logp = torch.log(counts / counts.sum())
    total, n = 0.0, 0
    for r in calib.held_rows:
        length = int(calib.valid[r].sum())
        labels = calib.input_ids[r, 1:length]
        total += float(-logp[labels].sum())
        n += length - 1
    return {"mean": total / max(n, 1), "tokens": n, "what": "add-one unigram of the fit rows (diagnostic, not a rule)"}


def prescreen(tokens: list[int], text: str) -> dict[str, Any]:
    distinct = len(set(tokens)) / max(len(tokens), 1)
    grams: dict[tuple[int, ...], int] = {}
    for i in range(len(tokens) - 3):
        g = tuple(tokens[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    letters = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    return {"distinct_token_ratio": distinct, "loop_flag": any(v >= 3 for v in grams.values()),
            "letter_or_space_fraction": letters, "what": "machine pre-screen, recorded only (never a grade)"}


def structural(child_dir: Path, receipt: dict[str, Any], device: torch.device) -> tuple[ChimeraOmniModel, dict[str, Any]]:
    model, cfg, _ = load_chimera_checkpoint(child_dir, dtype=torch.float32, verify=True)
    cfg.validate()
    model = model.to(device).eval()
    meta = sum(conf._numel(shape) for _, shape, _ in conf._meta_named_shapes(cfg))
    params = sum(p.numel() for p in model.parameters())
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    by_method = receipt.get("parameters_by_method", {})
    selections = [c for c in receipt.get("components", []) if c.get("selection_component")]
    checks = {
        "loads_sha_verified": True,
        "config_validates": True,
        "all_finite": finite,
        "parameters": params,
        "meta_parameters": meta,
        "parameters_equal_meta": params == meta,
        "receipt_names_every_parameter": sum(by_method.values()) == params,
        "methods": by_method,
        "methods_are_folds": set(by_method) <= set(c21.FOLD_METHODS),
        "copy_methods": receipt.get("copy_methods"),
        "four_x_rule": receipt.get("four_x_rule"),
        "selection_components": len(selections),
        "selection_min_compression": min((c["compression"] for c in selections), default=None),
        "every_selection_clears_4x": bool(selections) and all(c["clears_4x"] for c in selections),
        "assembly_check": receipt.get("assembly_check"),
    }
    checks["pass"] = bool(finite and checks["parameters_equal_meta"] and checks["receipt_names_every_parameter"]
                          and checks["methods_are_folds"] and checks["copy_methods"] == 0
                          and checks["four_x_rule"] == "enforced" and checks["every_selection_clears_4x"])
    return model, checks


def embed_only(child: ChimeraOmniModel) -> ChimeraOmniModel:
    ctl = copy.deepcopy(child)
    with torch.no_grad():
        for block in ctl.blocks:
            block.mixer.out.weight.zero_()
            block.ffn.down.weight.zero_()
    return ctl.eval()


def proof_eval_doc(*, role: str, model_key: str, parent: str | None, cases: list[dict], results: list[dict],
                   metrics: list[dict], report: dict, report_path: str, run_dir: str, fold_run: int | None,
                   experiment_key: str, model_path: str | None, params: int) -> dict[str, Any]:
    return {
        "schema": "omni-lab-proof-eval-v1",
        "suite": SUITE, "graders": GRADERS, "cases": cases,
        "model": {"model_key": model_key, "project": "chimera-omni", "role": "other", "layout": "chimera.checkpoint.v1",
                  "family": "chimera-omni-v2", "stage": "fold-proxy", "path": model_path, "parameter_count": params,
                  "dtype": "float32", "config": {"profile": report["profile"], "role": role},
                  "notes": {"child": "c21 fold child (trunk fold of T; K3/media not in this build)",
                            "random-init": f"ChimeraOmniModel default init, seed {report['seed']}",
                            "embed-only": "the child with every mixer.out and ffn.down zeroed"}[role],
                  **({"parent": parent, "relation": "fold"} if parent else {})},
        "run": {"run_key": f"eval:v2-fold-proxy:{report['child_sha16']}:{role}",
                "name": f"v2 fold-proxy checks ({role})", "engine": "chimera-v2-fold-checks",
                "gen": {"decode": f"greedy, {report['decode_tokens']} new tokens, ledger band masked",
                        "nll": "teacher-forced held rows"},
                "label": f"V2FOLD-{role.upper()}", "status": "pass",
                "started_utc": report["started_utc"], "ended_utc": report["ended_utc"],
                "config": {"rules": report["rules"], "verdict": report["verdict"]},
                "parent_run_id": fold_run, "receipt_path": report_path, "run_dir": run_dir, "owner": "claude",
                "notes": "rules pre-registered before the fold run; R2 reader grades are added with omni_lab.py grade"},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__, "code_commit": report.get("code")},
        "results": results, "metrics": metrics,
        "experiment": {"key": experiment_key, "kind": "fold", "role": "child" if role == "child" else "baseline",
                       "technique": "v2 fold-init (c21 trunk fold)", "family": "fold",
                       **({"child": model_key, "run_id": fold_run} if role == "child" else {}),
                       "params": {"profile": report["profile"], "source": report["source"]},
                       "evidence_level": "receipt", "date": report["started_utc"][:10], "results_path": run_dir},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold-dir", required=True, help="c21 --out (child/ + receipt.json)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--rules", required=True, help="the pre-registered PASS_RULES json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True, help="sealed smoke prompt set (neural-forge-matrix-prompt-set-v1)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold-run", type=int, default=None, help="omni_lab run id of the fold")
    parser.add_argument("--experiment-key", default="V2-FOLD-INIT-PROXY-20261022")
    parser.add_argument("--code", default=None, help="code identity string for the env row")
    args = parser.parse_args(argv)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    device = torch.device(args.device)
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    rules_sha = hashlib.sha256(Path(args.rules).read_bytes()).hexdigest()
    seed = int(rules["controls"]["random_init_seed"])
    decode_tokens = int(rules["R2"]["new_tokens"])
    prefix_tokens = int(rules["R2"]["held_prefix_tokens"])
    min_rows = int(rules["R1"]["min_rows_below_both"])
    fold_dir = Path(args.fold_dir)
    receipt = json.loads((fold_dir / "receipt.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    child, r0 = structural(fold_dir / "child", receipt, device)
    c21.use_chunked_gdn_scan(child, 64)
    child_sha = c21.sha256_file(fold_dir / "receipt.json")
    log(f"R0 structural pass={r0['pass']} params={r0['parameters']} min selection compression={r0['selection_min_compression']}")
    torch.manual_seed(seed)
    with torch.device(device):
        rand = ChimeraOmniModel(child.cfg)
    rand.eval()
    c21.use_chunked_gdn_scan(rand, 64)
    ctl = embed_only(child)
    models = {"child": child, "random-init": rand, "embed-only": ctl}

    calib = c21.load_corpus(args.corpus)
    nll = {}
    for role, model in models.items():
        nll[role] = held_nll(model, calib, device)
        log(f"held NLL {role}: {nll[role]['mean']:.4f} over {nll[role]['tokens']} tokens")
    rows = list(nll["child"]["per_row"])
    below = sum(1 for r in rows if nll["child"]["per_row"][r] < nll["random-init"]["per_row"][r]
                and nll["child"]["per_row"][r] < nll["embed-only"]["per_row"][r])
    r1 = {"child_mean": nll["child"]["mean"], "random_init_mean": nll["random-init"]["mean"],
          "embed_only_mean": nll["embed-only"]["mean"], "rows": len(rows), "rows_below_both": below,
          "min_rows_below_both": min_rows,
          "source_reference_mean": receipt["source_reference"]["held_nll_per_token"],
          "unigram_reference": unigram_reference(calib, int(child.cfg.ledger_base_vocab_size))}
    r1["pass"] = bool(r1["child_mean"] < r1["random_init_mean"] and r1["child_mean"] < r1["embed_only_mean"]
                      and below >= min_rows)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_sha = c21.sha256_file(Path(args.tokenizer) / "tokenizer.json")
    eos = {int(t) for t in ([tok.eos_token_id] if tok.eos_token_id is not None else [])}
    eos |= {int(t) for t in tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) if t is not None and t >= 0}
    prompt_set = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    prompts: list[dict[str, Any]] = []
    for i, text in enumerate(prompt_set["prompts"]):
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          enable_thinking=False, tokenize=True)
            if hasattr(ids, "keys"):                     # transformers 5 returns a BatchEncoding
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(t) for t in ids]
            form = "chat template (enable_thinking=False)"
        except Exception as exc:  # noqa: BLE001 -- recorded, raw text is the fallback
            ids, form = tok(text)["input_ids"], f"raw (chat template failed: {exc})"
        prompts.append({"id": f"smoke-{i}", "category": "smoke", "prompt_text": text, "ids": list(ids), "form": form})
    for r in calib.held_rows:
        ids = calib.input_ids[r, :prefix_tokens].tolist()
        prompts.append({"id": f"held-row-{r}", "category": "held_prefix", "prompt_text": tok.decode(ids),
                        "ids": ids, "form": f"raw continuation of held row {r}'s first {prefix_tokens} tokens"})
    decoded: dict[str, list[dict[str, Any]]] = {role: [] for role in models}
    for role, model in models.items():
        for p in prompts:
            toks, term = greedy(model, p["ids"], decode_tokens, eos, device)
            text = tok.decode(toks)
            decoded[role].append({"id": p["id"], "tokens": toks, "text": text, "termination": term,
                                  "prescreen": prescreen(toks, text)})
        log(f"decoded {len(prompts)} prompts with {role}")
    ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdict = {"R0": r0["pass"], "R1": r1["pass"], "R2": "pending reader grade (collapse-only bar)",
               "machine_part_pass": bool(r0["pass"] and r1["pass"])}
    report = {"schema": REPORT_SCHEMA, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                                                 "sha256": TOOL_SHA256},
              "started_utc": started, "ended_utc": ended, "profile": receipt["target"]["profile"],
              "source": receipt["source"]["label"], "child": str(fold_dir / "child"), "child_sha16": child_sha[:16],
              "fold_receipt_sha256": child_sha, "rules": {"path": args.rules, "sha256": rules_sha, **rules},
              "seed": seed, "decode_tokens": decode_tokens, "tokenizer": {"path": args.tokenizer, "sha256": tok_sha},
              "R0": r0, "R1": r1, "nll": nll, "prompts": [{k: v for k, v in p.items()} for p in prompts],
              "decoded": decoded, "verdict": verdict, "code": args.code}
    report_path = out / "checks_report.json"
    report_path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    cases = []
    for p in prompts:
        cases.append({"id": f"v2fold-{p['id']}", "category": p["category"], "lane": "v2-fold",
                      "prompt": [{"role": "user", "content": p["prompt_text"]}],
                      "media": None, "expected": None, "modality_in": ["text"], "modality_out": ["text"],
                      "tags": ["v2-fold", "proxy", "decoded"]})
    for r in calib.held_rows:
        cases.append({"id": f"v2fold-nll-held-row-{r}", "category": "held_nll", "lane": "v2-fold",
                      "prompt": [{"role": "user", "content": f"<omni-evidence-27b-v1 held row {r}: teacher-forced NLL>"}],
                      "media": None, "expected": None, "modality_in": ["text"], "modality_out": ["text"],
                      "tags": ["v2-fold", "proxy", "nll"]})
    keys = {"child": f"v2fold:co2-fold-proxy-FT@{child_sha[:12]}",
            "random-init": f"v2fold:co2-fold-proxy-random-init-seed{seed}",
            "embed-only": f"v2fold:co2-fold-proxy-FT-embed-only@{child_sha[:12]}"}
    for role in models:
        results = []
        for p, d in zip(prompts, decoded[role]):
            results.append({"case_id": f"v2fold-{p['id']}", "output_text": "".join(
                c if c.isprintable() or c in "\n\t" else f"<0x{ord(c):02x}>" for c in d["text"]),
                "output_tokens": len(d["tokens"]), "termination": d["termination"], "finite": True,
                "raw": {"prescreen": d["prescreen"], "prompt_form": p["form"]}})
        for r in calib.held_rows:
            results.append({"case_id": f"v2fold-nll-held-row-{r}", "output_text": None, "termination": None,
                            "finite": math.isfinite(nll[role]["per_row"][str(r)]),
                            "raw": {"nll_per_token": nll[role]["per_row"][str(r)]}})
        metrics = [{"name": "held_nll_mean", "value": nll[role]["mean"], "denominator": nll[role]["tokens"]}]
        if role == "child":
            metrics += [{"name": "r0_structural_pass", "value": 1.0 if r0["pass"] else 0.0},
                        {"name": "r1_knowledge_pass", "value": 1.0 if r1["pass"] else 0.0},
                        {"name": "r1_rows_below_both", "value": float(below), "denominator": len(rows)},
                        {"name": "source_reference_held_nll", "value": r1["source_reference_mean"]},
                        {"name": "unigram_reference_held_nll", "value": r1["unigram_reference"]["mean"]}]
        doc = proof_eval_doc(role=role, model_key=keys[role], parent=None, cases=cases, results=results,
                             metrics=metrics, report=report, report_path=str(report_path), run_dir=str(out),
                             fold_run=args.fold_run, experiment_key=args.experiment_key,
                             model_path=str(fold_dir / "child") if role != "random-init" else None,
                             params=sum(p.numel() for p in models[role].parameters()))
        (out / f"omni_lab_eval_{role}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "verdict": verdict, "R1": r1}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
