#!/usr/bin/env python3
"""D4.  A DIAGNOSTIC, and under no reading of anything a fold result.

WAVE 35'S HEADER, and it states one change and one reason.

**The change is a second knob and a grid.**  `mx34_penaltydecode.py` moved ONE
value -- repetition penalty 1.3 -- and found that it eliminated the artifact
class completely (37 of 37 artifact lanes became readable), recovered ZERO
answers, and destroyed all 13 lanes of intelligence signal the population held.
That is a real result and it is not the result it was read as.  **It priced one
setting, not "penalties in general",** and 1.3 is an aggressive setting: it is
the value a text-generation demo reaches for, not the mildest thing that could
plausibly break a loop.  A remedy clause is not refuted by the failure of its
most aggressive form, so wave 35 prices the MILD end of the same clause and the
other mechanism people actually mean when they say "token penalties":

* ``--penalty 1.1`` and ``--penalty 1.15`` -- below wave 34's 1.3;
* ``--no-repeat-ngram 3`` alone -- a HARD constraint rather than a soft one,
  which cannot touch a logit that is not completing a repeated trigram and so
  cannot flatten a lane that was answering;
* ``--penalty 1.1 --no-repeat-ngram 3`` -- the two together.

Four settings on wave 34's five children plus its frontier child, which makes
the grid comparable to wave 34's row rather than to a new population.

**The n-gram bank includes the PROMPT, and that choice is stated rather than
buried.**  ``transformers``' own ``NoRepeatNGramLogitsProcessor`` bans an n-gram
seen anywhere in the sequence, prompt included, and that is the thing a reader
who says "just use no-repeat-ngram" is asking for; it is also the form that can
stop a prompt ECHO, which is a third of the artifact class.  It has a cost this
diagnostic must not hide: the frozen fifteen include lanes whose correct answer
QUOTES the context -- *the latest checkout time is 4:15 p.m.* is a trigram of
the prompt -- so on this population a full-sequence ban can forbid the right
answer.  Both directions are counted; neither is assumed.

**Nothing else moved.**  The generation loop, the chat template, the cap
accounting, the termination taxonomy, the artifact collapse and the identity
refusal are wave 34's.  The identity setting is now BOTH knobs off
(``--penalty 1.0 --no-repeat-ngram 0``) and the refusal fires on the bytes
exactly as before.

--- wave 34's header ---

C3.  A DIAGNOSTIC, and under no reading of anything a fold result.

Read this paragraph before reading any number this file writes.  **Nothing here
is sealed, nothing here enters the ledger, nothing here moves a frontier rung,
and no child in this campaign may ever be described using a count computed by
this file.**  The sealed fifteen are decoded greedily with no penalty, and that
is the only decode any verdict in this estate rests on.  What this file does is
change ONE runtime knob -- a repetition penalty on the next-token logits -- and
re-decode the same frozen prompts through the same weights, so that a claim
about the *decode* can be priced instead of assumed.

**The claim being priced.**  Cyril's canonical standard strikes loops, cap-rides
and prompt echo from the intelligence question on the grounds that they are
compression artifacts, "fixable with token penalties and retraining".  The
canonical re-read found 846 such lanes -- 43.6 per cent of everything the
campaign scored as a behavioural failure.  That is a very large number to rest
on an untested clause.  So: on the lanes the sealed decode lost to repetition,
does a penalised decode of the SAME WEIGHTS produce readable text, and is that
text an answer?

**Three outcomes and they are not the same.**

* ``recovered`` -- the raw lane was an artifact (loop, cap-ride, empty, echo) and
  the penalised lane is readable AND carries a correct, specific element of what
  the lane asked.  This is the outcome that supports the clause.
* ``readable-but-empty`` -- the penalised lane is readable and says nothing.  The
  penalty converted an artifact into a CONTENT miss.  This is evidence AGAINST
  reading artifact lanes as suppressed intelligence: the intelligence was not
  under the loop.
* ``still-artifact`` -- the penalty did not even buy readability.

The three are counted separately and never summed into a score, because a
"penalised count" is exactly the number this file must not produce.

**What the penalty is.**  The standard multiplicative form: for every token id
already in the context or already generated, a logit is divided by ``penalty``
if positive and multiplied by it if negative.  ``penalty`` 1.0 is a bit-exact
no-op and is available as ``--penalty 1.0``, which is the control that proves
this loop reproduces the sealed loop.  **Run that control.**  A diagnostic whose
own decoder has not been shown to reproduce the sealed decoder at the identity
setting is a diagnostic about two things at once.

    mx35_penaltydecode.py --deployment D --pack P --receipt R --child-model C
                          --label V12CL896G03 --penalty 1.3 --out O

`--penalty 1.0 --compare-to <sealed dossier>` runs the identity control and
refuses if any lane's response bytes differ from the sealed dossier's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "neural-forge-matrix-wave35-penalty-decode-diagnostic-v1"
NOT_A_FOLD = (
    "DIAGNOSTIC ONLY.  This file changes a decode parameter and not one byte of "
    "any child.  It is not sealed, it produces no verdict and no count, no "
    "ledger row is generated from it, and no frontier claim in this wave or any "
    "later one may cite it."
)


def _load_runner(deployment: Path):
    for sub in ("src", "experiments"):
        path = str(deployment / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import run_lfm2_composition_frozen_review as runner  # noqa: E402

    return runner


def generate_penalised(
    runner: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    cap: int,
    device: str,
    eos_id: int | None,
    penalty: float,
    no_repeat_ngram: int,
    label: str,
) -> list[dict[str, Any]]:
    """The sealed runner's greedy loop with TWO knobs added, and both named.

    Everything else -- the chat template, the cap accounting, the termination
    taxonomy, the `skip_special_tokens=False` decode, the per-case digest -- is
    the sealed runner's, reached through its own helpers, so the only difference
    between this and the frozen decode is the two knobs.  At `penalty == 1.0`
    and `no_repeat_ngram == 0` both additions are arithmetically the identity
    and `--compare-to` asserts that on the bytes.
    """
    import torch

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        started = time.monotonic()
        messages = runner._messages_for(row)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - the LFM2 tokenizer carries a template
            text = "\n".join(message["content"] for message in messages)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        seen: set[int] = set(int(t) for t in ids[0].tolist())
        # THE FULL SEQUENCE, prompt included, exactly as `transformers`'
        # NoRepeatNGramLogitsProcessor sees it.  `banned` maps an (n-1)-gram
        # prefix to the token ids that have already followed it, so the ban is
        # maintained incrementally rather than by rescanning the sequence at
        # every one of up to 1,024 steps.
        sequence: list[int] = [int(t) for t in ids[0].tolist()]
        banned: dict[tuple[int, ...], set[int]] = {}
        if no_repeat_ngram > 1:
            for i in range(len(sequence) - no_repeat_ngram + 1):
                prefix = tuple(sequence[i:i + no_repeat_ngram - 1])
                banned.setdefault(prefix, set()).add(sequence[i + no_repeat_ngram - 1])
        generated: list[int] = []
        past = None
        current = ids
        termination = "censored-at-cap"
        try:
            for _step in range(cap):
                result = model(input_ids=current, past_key_values=past, use_cache=True)
                past = result.past_key_values
                logits = result.logits[0, -1]
                # ---- KNOB ONE: wave 34's repetition penalty, unchanged ----
                if penalty != 1.0 and seen:
                    idx = torch.tensor(sorted(seen), device=logits.device, dtype=torch.long)
                    picked = logits.index_select(0, idx)
                    logits = logits.index_copy(
                        0, idx, torch.where(picked > 0, picked / penalty, picked * penalty)
                    )
                # ---- KNOB TWO: wave 35's HARD n-gram ban ----
                # Hard rather than soft, and that is the point of running it.
                # A token that would repeat an n-gram is removed from the argmax
                # entirely; a token that would not is left at EXACTLY its sealed
                # logit.  The multiplicative penalty cannot do that -- it
                # rescales every token the context has seen, including the ones
                # a correct answer is made of, which is the mechanism wave 34
                # measured turning thirteen holding lanes into nothing.
                if no_repeat_ngram > 1 and len(sequence) >= no_repeat_ngram - 1:
                    prefix = tuple(sequence[len(sequence) - no_repeat_ngram + 1:])
                    forbid = banned.get(prefix)
                    if forbid:
                        idx = torch.tensor(sorted(forbid), device=logits.device,
                                           dtype=torch.long)
                        logits = logits.index_fill(0, idx, float("-inf"))
                # -----------------------------------------------------------
                next_id = int(logits.argmax().item())
                if eos_id is not None and next_id == eos_id:
                    termination = "complete"
                    break
                generated.append(next_id)
                seen.add(next_id)
                if no_repeat_ngram > 1 and len(sequence) >= no_repeat_ngram - 1:
                    prefix = tuple(sequence[len(sequence) - no_repeat_ngram + 1:])
                    banned.setdefault(prefix, set()).add(next_id)
                sequence.append(next_id)
                current = torch.tensor([[next_id]], device=device, dtype=ids.dtype)
        except RuntimeError as exc:  # noqa: BLE001 - recorded, never swallowed
            termination = "runtime-error"
            print(f"{label} {row['id']}: {type(exc).__name__}: {exc}", flush=True)
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        records.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "response_verbatim": decoded,
                "response_sha256": runner.canonical_sha256({"text": decoded}),
                "generated_tokens": len(generated),
                "termination": termination,
                "wall_time_seconds": round(time.monotonic() - started, 3),
            }
        )
        print(
            f"{label} {index}/{len(rows)} {row['id']}: {len(generated)} tokens "
            f"({termination}) in {records[-1]['wall_time_seconds']:.1f}s",
            flush=True,
        )
    return records


def collapse(text: str) -> str:
    """Unique-content view: the artifact classifier's own collapse, reused.

    Wave 34R bucketed a lane as an artifact on, among other things, fewer than
    ~80 characters of unique text after repeated n-grams are dropped.  The same
    collapse is applied here so the before/after comparison is on one definition.
    """
    words = text.split()
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    n = 6
    i = 0
    while i < len(words):
        gram = tuple(words[i:i + n])
        if len(gram) == n and gram in seen:
            i += 1
            continue
        seen.add(gram)
        out.append(words[i])
        i += 1
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True,
                    help="the sealed frozen prompt pack, unchanged")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--child-model", type=Path, required=True)
    ap.add_argument("--child-tokenizer", type=Path)
    ap.add_argument("--parent-model", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--penalty", type=float, default=1.3)
    ap.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=0,
        help=(
            "ban any n-gram of this length already present in the sequence, "
            "PROMPT INCLUDED, which is what transformers' own processor does "
            "and what makes the arm able to stop a prompt echo; 0 disables it "
            "and 0 is the identity"
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--maximum-new-tokens", type=int, default=1024)
    ap.add_argument("--compare-to", type=Path,
                    help="a sealed dossier; at --penalty 1.0 every response byte must match")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    runner = _load_runner(args.deployment)
    import torch  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    torch.set_grad_enabled(False)
    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    rows = pack["prompts"] if isinstance(pack, dict) and "prompts" in pack else pack
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.child_tokenizer or args.parent_model)
    )
    child = runner._load_standard_shape_child(
        receipt=receipt,
        child_model=args.child_model,
        device=str(args.device),
        dtype=torch.float32,
    )
    setting = f"p{args.penalty}n{args.no_repeat_ngram}"
    records = generate_penalised(
        runner, child, tokenizer, list(rows),
        cap=int(args.maximum_new_tokens),
        device=str(args.device),
        eos_id=getattr(tokenizer, "eos_token_id", None),
        penalty=float(args.penalty),
        no_repeat_ngram=int(args.no_repeat_ngram),
        label=f"{args.label}@{setting}",
    )

    identity_control: dict[str, Any] | None = None
    if args.compare_to is not None:
        sealed = json.loads(args.compare_to.read_text(encoding="utf-8"))
        by_id = {
            r["id"]: r.get("response_verbatim", "")
            for r in (sealed.get("child_responses") or sealed.get("responses") or [])
        }
        differing = [
            r["id"] for r in records
            if r["id"] in by_id and by_id[r["id"]] != r["response_verbatim"]
        ]
        # THE IDENTITY SETTING IS NOW BOTH KNOBS OFF.  Wave 34 had one knob and
        # `--penalty 1.0` was the whole of it; wave 35 has two, and an identity
        # claim that checked only the first would pass while the second was
        # silently on.
        is_identity = float(args.penalty) == 1.0 and int(args.no_repeat_ngram) == 0
        identity_control = {
            "penalty": float(args.penalty),
            "no_repeat_ngram": int(args.no_repeat_ngram),
            "is_the_identity_setting": is_identity,
            "lanes_compared": sum(1 for r in records if r["id"] in by_id),
            "lanes_differing": len(differing),
            "differing_ids": differing[:16],
            "reproduces_the_sealed_decode": is_identity and not differing,
        }
        if is_identity and differing:
            raise SystemExit(
                "REFUSED: at penalty 1.0 this loop must be byte-identical to the "
                f"sealed decode and {len(differing)} lanes differ: {differing[:8]}"
            )

    body = {
        "schema": SCHEMA,
        "NOT_A_FOLD": NOT_A_FOLD,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "cell_id": receipt.get("cell_id"),
        "cell_instance_id": receipt.get("cell_instance_id"),
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        "pack_sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
        "repetition_penalty": float(args.penalty),
        "no_repeat_ngram": int(args.no_repeat_ngram),
        "no_repeat_ngram_bank": (
            "the FULL sequence, prompt included, as transformers' own processor "
            "does; on this population that can forbid a correct answer which "
            "quotes the context, and both directions are counted"
        ),
        "decode": "greedy, cap 1024, sealed chat template, sealed termination taxonomy",
        "identity_control": identity_control,
        "responses": [
            dict(r, unique_content_chars=len(collapse(r["response_verbatim"])))
            for r in records
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")
    caps = sum(1 for r in records if r["termination"] == "censored-at-cap")
    print(f"\n{args.label} @ penalty {args.penalty} ngram {args.no_repeat_ngram}: "
          f"{len(records)} lanes, "
          f"{caps} cap-rides, "
          f"{sum(1 for r in body['responses'] if r['unique_content_chars'] < 80)} "
          f"under 80 unique chars")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
