#!/usr/bin/env python3
"""15-prompt parity check: a GGUF served by llama-server vs its source HF checkpoint.

Both sides get the BYTE-IDENTICAL prompt text and greedy decoding. The script records
every completion verbatim on both sides plus four diagnostics per prompt:

    exact_match          strings identical
    normalized_match     identical after whitespace collapse + strip
    char_prefix          length of the common character prefix
    token_prefix         length of the common token prefix (HF tokenizer, diagnostic only)

None of these is a gate. Token match in particular is a DIAGNOSTIC ONLY in this estate
(standing rule: never grade on token match). A GGUF that diverges in wording but answers
the same is a normal, expected quantization artifact and is reported as such: the
`READ-<tag>.txt` file holds both answers side by side for a human read.

Modes
-----
  --skip-hf      only run the GGUF side (no torch needed)
  --skip-gguf    only run the HF side (produces the reference file for a later compare)
  --reference    path to a previously written parity JSON to compare the GGUF side against

Examples
--------
  # both sides, small child, HF on GPU4
  python3 parity_check.py --tag Q35-0.8B-Q4KM \
      --server-url http://127.0.0.1:18090 \
      --hf-dir /home/cereal/models/Qwen3.5-0.8B \
      --hf-device cuda:0 --hf-dtype bfloat16 \
      --out-dir /home/cereal/omni-campaign/export/parity

  # GGUF side only, against an already-running server
  python3 parity_check.py --tag P40-Q36-27B --server-url http://127.0.0.1:18080 \
      --skip-hf --out-dir /home/cereal/omni-campaign/export/parity
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    LlamaServer,
    host_facts,
    safe_tag,
    sha256_file,
    utc_now,
    write_receipt,
)

DEFAULT_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_bank.json")


def load_bank(path):
    with open(path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    if not bank.get("prompts"):
        raise ValueError("bank %s has no prompts" % path)
    return bank


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()


def common_prefix_len(left, right):
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def compare_pair(gguf_text, hf_text, tokenizer=None):
    result = {
        "exact_match": gguf_text == hf_text,
        "normalized_match": normalize(gguf_text) == normalize(hf_text),
        "char_prefix": common_prefix_len(gguf_text or "", hf_text or ""),
        "char_len_gguf": len(gguf_text or ""),
        "char_len_hf": len(hf_text or ""),
        "token_prefix": None,
        "token_len_gguf": None,
        "token_len_hf": None,
    }
    if tokenizer is not None:
        left = tokenizer(gguf_text or "", add_special_tokens=False)["input_ids"]
        right = tokenizer(hf_text or "", add_special_tokens=False)["input_ids"]
        result["token_prefix"] = common_prefix_len(left, right)
        result["token_len_gguf"] = len(left)
        result["token_len_hf"] = len(right)
    return result


def load_hf(hf_dir, device, dtype_name, device_map, trust_remote_code):
    """Load an HF causal LM through whichever Auto class accepts this architecture."""
    import torch  # noqa: PLC0415 - only needed on the HF side
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415

    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=trust_remote_code)
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=trust_remote_code)

    candidates = []
    import transformers  # noqa: PLC0415
    for name in ("AutoModelForCausalLM", "AutoModelForConditionalGeneration",
                 "AutoModelForVision2Seq", "AutoModel"):
        klass = getattr(transformers, name, None)
        if klass is not None:
            candidates.append((name, klass))

    kwargs = {"trust_remote_code": trust_remote_code}
    if device_map:
        kwargs["device_map"] = device_map
    errors = []
    for name, klass in candidates:
        for dtype_kwarg in ("dtype", "torch_dtype"):
            try:
                model = klass.from_pretrained(hf_dir, **dict(kwargs, **{dtype_kwarg: dtype}))
                if not device_map:
                    model = model.to(device)
                model.eval()
                return model, tokenizer, config, "%s(%s)" % (name, dtype_kwarg)
            except TypeError as exc:
                errors.append("%s/%s: %r" % (name, dtype_kwarg, exc))
            except (ValueError, KeyError) as exc:
                errors.append("%s/%s: %r" % (name, dtype_kwarg, exc))
                break
    raise RuntimeError("no Auto class loaded %s:\n%s" % (hf_dir, "\n".join(errors)))


def hf_generate(model, tokenizer, prompt, n_predict, device):
    import torch  # noqa: PLC0415

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    target = device
    if hasattr(model, "device") and model.device is not None:
        target = model.device
    encoded = {key: value.to(target) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    started = time.time()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=n_predict,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    wall = time.time() - started
    new_tokens = generated[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "content": text,
        "prompt_tokens": int(prompt_len),
        "predicted_n": int(new_tokens.shape[0]),
        "wall_s": round(wall, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--out-dir", default="/home/cereal/omni-campaign/export/parity")
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--gguf-path", default=None, help="hashed into the receipt if given")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--reference", default=None,
                        help="previous parity JSON whose hf_side is used as the reference")
    parser.add_argument("--hf-dir", default=None)
    parser.add_argument("--hf-device", default="cuda:0")
    parser.add_argument("--hf-dtype", default="bfloat16")
    parser.add_argument("--hf-device-map", default=None, help="e.g. auto / sequential")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat", action="store_true",
                        help="render each prompt through the HF chat template first "
                             "(the rendered text is what BOTH sides receive)")
    args = parser.parse_args(argv)
    # --tag becomes the receipt and READ-<tag>.txt filenames; validated here too.
    args.tag = safe_tag(args.tag)

    bank = load_bank(args.bank)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = None
    model = None
    loader = None
    if args.chat or not args.skip_hf:
        if not args.hf_dir:
            parser.error("--hf-dir is required unless --skip-hf and no --chat")

    if not args.skip_hf:
        model, tokenizer, config, loader = load_hf(
            args.hf_dir, args.hf_device, args.hf_dtype, args.hf_device_map, args.trust_remote_code
        )
    elif args.chat:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=args.trust_remote_code)

    reference = None
    reference_meta = None
    if args.reference:
        with open(args.reference, "r", encoding="utf-8") as handle:
            reference_payload = json.load(handle)
        reference = {row["id"]: row for row in reference_payload["rows"]}
        # A borrowed hf column with no provenance is a duplicated contract: the receipt
        # reads exactly like one where the HF side was measured in this run, and if the
        # reference receipt is later re-run in place (write_receipt os.replace's it) the
        # borrowed numbers describe bytes that no longer exist. Everything needed to
        # re-resolve the reference is copied here, hash included.
        reference_meta = {
            "path": os.path.abspath(args.reference),
            "sha256": sha256_file(args.reference),
            "tag": reference_payload.get("tag"),
            "created_at": reference_payload.get("created_at"),
            "bank_id": reference_payload.get("bank_id"),
            "bank_sha256": reference_payload.get("bank_sha256"),
            "hf_dir": reference_payload.get("hf_dir"),
            "hf_dtype": reference_payload.get("hf_dtype"),
            "hf_loader": reference_payload.get("hf_loader"),
            "chat_template_applied": reference_payload.get("chat_template_applied"),
            "gguf_path": reference_payload.get("gguf_path"),
            "gguf_sha256": reference_payload.get("gguf_sha256"),
            "rows_available": len(reference),
            "rows_borrowed": 0,
            "bank_matches_this_run": reference_payload.get("bank_sha256") == sha256_file(args.bank),
            "chat_rendering_matches_this_run": (
                bool(reference_payload.get("chat_template_applied")) == bool(args.chat)),
        }
        if reference_meta["hf_dir"] is None:
            # A reference that itself borrowed its hf column cannot say where the answers
            # came from, and neither can this receipt.
            reference_meta["warning"] = (
                "the reference receipt records no hf_dir, so the HF answers it supplies "
                "have no resolvable source of their own"
            )
        if not reference_meta["bank_matches_this_run"]:
            print("WARNING: the reference receipt was measured against a DIFFERENT prompt "
                  "bank (%s) than this run (%s)" % (
                      str(reference_meta["bank_sha256"])[:16], sha256_file(args.bank)[:16]),
                  file=sys.stderr)
        if not reference_meta["chat_rendering_matches_this_run"]:
            print("WARNING: the reference receipt was measured with chat_template_applied="
                  "%s but this run uses %s; the two sides did not receive the same text."
                  % (reference_payload.get("chat_template_applied"), bool(args.chat)),
                  file=sys.stderr)

    server = LlamaServer(args.server_url, timeout=args.request_timeout)
    server_props = {}
    if not args.skip_gguf:
        health = server.health()
        if health.get("status") != "ok":
            print("llama-server not healthy at %s: %s" % (args.server_url, health), file=sys.stderr)
            return 3
        server_props = server.props()

    rows = []
    for item in bank["prompts"]:
        prompt = item["prompt"]
        if args.chat and tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        row = {
            "id": item["id"],
            "category": item["category"],
            "prompt": prompt,
            "n_predict": item["n_predict"],
            "gguf": None,
            "hf": None,
            # "measured" when this run generated the HF answer, "reference" when it was
            # copied out of --reference, None when there is no HF side at all. A borrowed
            # column must never read as a measured one.
            "hf_source": None,
            "compare": None,
        }
        if not args.skip_gguf:
            try:
                row["gguf"] = server.completion(
                    prompt, n_predict=item["n_predict"], cache_prompt=False,
                    timeout=args.request_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - a failed request is a measurement
                row["gguf"] = {"error": repr(exc)}
        if not args.skip_hf:
            row["hf"] = hf_generate(model, tokenizer, prompt, item["n_predict"], args.hf_device)
            row["hf_source"] = "measured"
        elif reference is not None and item["id"] in reference:
            row["hf"] = reference[item["id"]].get("hf")
            if row["hf"] is not None:
                row["hf_source"] = "reference"
                row["hf_reference_sha256"] = reference_meta["sha256"]
                row["hf_reference_tag"] = reference_meta["tag"]
                reference_meta["rows_borrowed"] += 1

        if row["gguf"] and row["hf"] and "error" not in row["gguf"]:
            row["compare"] = compare_pair(
                row["gguf"].get("content", ""), row["hf"].get("content", ""), tokenizer
            )
        rows.append(row)
        marker = ""
        if row["compare"]:
            marker = "exact" if row["compare"]["exact_match"] else (
                "normalized" if row["compare"]["normalized_match"] else
                "diverges@%d" % row["compare"]["char_prefix"]
            )
        print("%-8s %-22s %s" % (item["id"], item["category"], marker))

    compared = [row for row in rows if row["compare"]]
    summary = {
        "prompts": len(rows),
        "compared": len(compared),
        "exact_matches": sum(1 for row in compared if row["compare"]["exact_match"]),
        "normalized_matches": sum(1 for row in compared if row["compare"]["normalized_match"]),
        "mean_char_prefix": (
            round(sum(row["compare"]["char_prefix"] for row in compared) / len(compared), 2)
            if compared else None
        ),
        "gguf_errors": sum(1 for row in rows if row["gguf"] and "error" in row["gguf"]),
    }

    served_model = server_props.get("model_path") if isinstance(server_props, dict) else None
    served_matches = None
    if args.gguf_path and served_model:
        served_matches = os.path.basename(str(served_model)) == os.path.basename(args.gguf_path)
        if not served_matches:
            print(
                "WARNING: the server is serving %r but --gguf-path is %r; this receipt "
                "describes the SERVED model, not the file that was hashed."
                % (served_model, args.gguf_path),
                file=sys.stderr,
            )

    payload = {
        "kind": "gguf_hf_parity",
        "created_at": utc_now(),
        "tag": args.tag,
        "bank_id": bank["bank_id"],
        "bank_sha256": sha256_file(args.bank),
        "server_url": args.server_url if not args.skip_gguf else None,
        "server_props_model": (server_props.get("model_path") if isinstance(server_props, dict) else None),
        # WHICH server answered. Two runs of this bank against the same GGUF file gave
        # different continuations under greedy decode (temperature 0, top_k 1), and the
        # receipt could not say what differed because it recorded nothing about the server
        # beyond the model path. build_info and n_ctx are the parts /props will tell us;
        # the KV types are not exposed there, so a parity receipt still cannot fully
        # identify the serving configuration (see the gap in EXPORT_LANE_SPEC.md).
        "server_props": {
            key: (server_props.get(key)[:200]
                  if isinstance(server_props.get(key), str) else server_props.get(key))
            for key in ("build_info", "n_ctx", "total_slots", "chat_template")
            if isinstance(server_props, dict) and key in server_props
        } or None,
        # Pointing --gguf-path at one file while the server serves another produces a
        # clean-looking receipt about the wrong model, so the two are compared here.
        "served_model_matches_gguf_path": served_matches,
        "hf_dir": args.hf_dir if not args.skip_hf else None,
        "hf_loader": loader,
        "hf_dtype": args.hf_dtype if not args.skip_hf else None,
        # Where the HF side of every comparison came from. None means it was measured in
        # this run; a dict means it was borrowed, and names the exact bytes it came from.
        "hf_reference": reference_meta,
        "hf_side_source": (
            "measured" if not args.skip_hf
            else ("reference" if reference_meta else "absent")
        ),
        "chat_template_applied": bool(args.chat),
        "host": host_facts(),
        "summary": summary,
        "rows": rows,
        "grading_note": (
            "Diagnostics only. Token/char match is never a gate in this estate; wording "
            "divergence after quantization is expected. A human reads READ-<tag>.txt."
        ),
    }
    if args.gguf_path and os.path.exists(args.gguf_path):
        payload["gguf_sha256"] = sha256_file(args.gguf_path)
        payload["gguf_path"] = args.gguf_path

    json_path = os.path.join(args.out_dir, "parity-%s.json" % args.tag)
    digest = write_receipt(json_path, payload)

    read_path = os.path.join(args.out_dir, "READ-%s.txt" % args.tag)
    with open(read_path, "w", encoding="utf-8") as handle:
        handle.write("PARITY READ  tag=%s  bank=%s  %s\n" % (args.tag, bank["bank_id"], utc_now()))
        # The human reading this must be able to see, without opening the JSON, that the
        # HF column was copied from another receipt rather than generated here.
        if reference_meta:
            handle.write(
                "HF SIDE: BORROWED from %s (sha256=%s, tag=%s, hf_dir=%s, dtype=%s); "
                "%d of %d rows borrowed. Nothing in the HF column was measured in this run.\n"
                % (reference_meta["path"], reference_meta["sha256"], reference_meta["tag"],
                   reference_meta["hf_dir"], reference_meta["hf_dtype"],
                   reference_meta["rows_borrowed"], len(rows)))
        elif not args.skip_hf:
            handle.write("HF SIDE: measured in this run from %s (%s)\n"
                         % (args.hf_dir, args.hf_dtype))
        handle.write("=" * 100 + "\n")
        for row in rows:
            handle.write("\n[%s] %s  n_predict=%s\n" % (row["id"], row["category"], row["n_predict"]))
            handle.write("--- PROMPT ---\n%s\n" % row["prompt"])
            if row["gguf"]:
                handle.write("--- GGUF (llama-server) ---\n%s\n" % row["gguf"].get("content", row["gguf"].get("error", "")))
                handle.write("    [stop=%s predicted=%s pp=%s tg=%s]\n" % (
                    row["gguf"].get("stop_type"), row["gguf"].get("predicted_n"),
                    row["gguf"].get("prompt_per_second"), row["gguf"].get("predicted_per_second")))
            if row["hf"]:
                handle.write("--- HF (transformers, %s) ---\n%s\n" % (
                    row.get("hf_source") or "unknown source", row["hf"].get("content", "")))
            if row["compare"]:
                handle.write("    [exact=%s normalized=%s char_prefix=%s token_prefix=%s]\n" % (
                    row["compare"]["exact_match"], row["compare"]["normalized_match"],
                    row["compare"]["char_prefix"], row["compare"]["token_prefix"]))
        handle.write("\n" + "=" * 100 + "\nSUMMARY: %s\n" % json.dumps(summary, sort_keys=True))

    print("\nsummary: %s" % json.dumps(summary, sort_keys=True))
    print("parity receipt: %s sha256=%s" % (json_path, digest))
    print("human read:     %s" % read_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
