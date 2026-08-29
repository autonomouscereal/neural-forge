#!/usr/bin/env python3
"""Is a MANY-TO-ONE `model.vocab` loadable on THIS stack?

Wave 40 measured that it is not, for C1's tie arm, with a NONDETERMINISTIC
refusal naming a different token on every load.  ALIAS's whole mechanism is the
same object -- dropped type strings addressing a surviving row's id -- so the
answer decides whether ALIAS is a stock child or a probe, and it is taken
BEFORE the builder is written, exactly as wave 40 took its own.

Three constructions are tried, smallest first, on the real child on the share:

  A  the sealed injective child, unmodified          -- the control
  B  A plus ONE dropped token aliased onto a kept id -- the minimal many-to-one
  C  A plus 256 dropped tokens aliased               -- the same at scale

Each is loaded FIVE times in this process and the id->token map hashed, so a
nondeterministic inverse shows up as a digest that moves.
"""
import hashlib, json, shutil, sys, tempfile, traceback
from pathlib import Path

CHILD = Path(sys.argv[1])
PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")

from transformers import AutoTokenizer

child_body = json.loads((CHILD / "tokenizer.json").read_text(encoding="utf-8"))
parent_body = json.loads((PARENT / "tokenizer.json").read_text(encoding="utf-8"))
kept_tokens = set(child_body["model"]["vocab"])
dropped = [t for t in parent_body["model"]["vocab"] if t not in kept_tokens]
print(f"kept {len(kept_tokens)} tokens, dropped {len(dropped)}")


def attempt(name: str, body: dict, loads: int = 5) -> None:
    digests, errors = set(), []
    for _ in range(loads):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for f in ("tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
                if (CHILD / f).exists():
                    shutil.copy(CHILD / f, d / f)
            (d / "tokenizer.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            try:
                tok = AutoTokenizer.from_pretrained(str(d))
                n = tok.vocab_size
                toks = tok.convert_ids_to_tokens(list(range(n)))
                digests.add(hashlib.sha256("\x00".join(map(str, toks)).encode()).hexdigest())
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}".split("\n")[0][:200])
    print(f"\n[{name}] {loads} loads -> {len(digests)} distinct digests, {len(errors)} errors")
    for dg in sorted(digests):
        print(f"   digest {dg}")
    for e in errors[:5]:
        print(f"   ERROR  {e}")


attempt("A control (injective)", child_body)

b = json.loads(json.dumps(child_body))
victim = dropped[0]
target_id = b["model"]["vocab"][sorted(kept_tokens)[len(kept_tokens) // 2]]
b["model"]["vocab"][victim] = target_id
print(f"\nB aliases {victim!r} -> id {target_id}")
attempt("B one alias", b)

c = json.loads(json.dumps(child_body))
kept_sorted = sorted(kept_tokens)
for i, t in enumerate(dropped[:256]):
    c["model"]["vocab"][t] = c["model"]["vocab"][kept_sorted[i % len(kept_sorted)]]
attempt("C 256 aliases", c)
