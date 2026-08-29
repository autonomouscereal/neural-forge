#!/usr/bin/env python3
"""Route D: can the alias live in `added_tokens` instead of `model.vocab`?

`model.vocab` stays INJECTIVE, so the merge-closure check that produced the
"Token ... out of vocabulary" refusal cannot fire.  The many-to-one map is
pushed into the AddedVocabulary, whose id->token inverse is a second HashMap.
If that inverse is also built nondeterministically the class is closed on three
constructions rather than two; if it loads deterministically, ALIAS has an
injective-vocab realisation and the semantics are then the question.
"""
import hashlib, json, shutil, sys, tempfile
from pathlib import Path

CHILD = Path(sys.argv[1])
PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
from transformers import AutoTokenizer

child_body = json.loads((CHILD / "tokenizer.json").read_text(encoding="utf-8"))
parent_body = json.loads((PARENT / "tokenizer.json").read_text(encoding="utf-8"))
kept = child_body["model"]["vocab"]
dropped = [t for t in parent_body["model"]["vocab"] if t not in kept]
kept_sorted = sorted(kept)


def attempt(name, body, loads=5):
    digests, errors, extra = set(), [], set()
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
                extra.add((n, len(tok)))
                probe = " understanding the document"
                ids = tok(probe, add_special_tokens=False).input_ids
                extra.add(("encode", tuple(ids)[:12], tok.decode(ids) == probe))
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}".split("\n")[0][:180])
    print(f"\n[{name}] {loads} loads -> {len(digests)} digests, {len(errors)} errors")
    for dg in sorted(digests):
        print(f"   digest {dg}")
    for e in sorted(map(str, extra))[:6]:
        print(f"   obs    {e}")
    for e in errors[:4]:
        print(f"   ERROR  {e}")


d1 = json.loads(json.dumps(child_body))
d1.setdefault("added_tokens", [])
d1["added_tokens"].append({
    "id": kept[kept_sorted[len(kept_sorted) // 2]], "content": dropped[0],
    "single_word": False, "lstrip": False, "rstrip": False,
    "normalized": False, "special": False})
attempt("D one aliased added_token (duplicate id)", d1)

d2 = json.loads(json.dumps(child_body))
d2.setdefault("added_tokens", [])
for i, t in enumerate(dropped[:256]):
    d2["added_tokens"].append({
        "id": kept[kept_sorted[i % len(kept_sorted)]], "content": t,
        "single_word": False, "lstrip": False, "rstrip": False,
        "normalized": False, "special": False})
attempt("D 256 aliased added_tokens", d2)
