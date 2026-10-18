"""C1 -- the vocabulary coreset.  Deletion, not factorisation.

At the 27B the embedding and the untied head are 2.54B parameters, 9% of the
model; at the 4B the tied table is ~16%.  The transaction is: keep ``k`` token
rows, delete the rest from BOTH tables, and rewrite the tokenizer so the child's
BPE is a real BPE over the kept tokens rather than a table with holes in it.

Three refusals, all of them things that produce a tokenizer which loads and is
broken:

* **byte alphabet.**  A ByteLevel BPE that cannot spell every byte cannot encode
  arbitrary text.  All 256 byte tokens are mandatory.
* **merge closure.**  A merge ``"a b" -> "ab"`` is only usable if ``a``, ``b``
  and ``ab`` are all in the table.  Keeping ``ab`` without its parents leaves a
  rule that never fires; keeping the rule without ``ab`` is a crash.  The keep
  set is closed transitively before it is counted.
* **hard ids.**  ``bos``/``eos``/``pad`` in the text config and
  ``image_token_id``/``video_token_id``/``vision_start``/``vision_end`` at the
  top level point into the table (configuration_qwen3_5.py: 169-172, 98-100).
  A remap that renumbers rows and leaves those alone points them at whatever
  token inherited the number.

  Three things about hard ids that are easy to get wrong and that this module
  now handles explicitly, because each of them ships a child that loads and
  cannot be generated from:

  * they are not always SCALARS.  A real Qwen text config carries
    ``eos_token_id`` as a LIST (e.g. ``[151645, 151643]``: ``<|im_end|>`` and
    ``<|endoftext|>``).  An ``isinstance(value, int)`` filter skips those --
    they are then neither protected from the coreset nor renumbered.
  * the config is not the only place they live.  ``model.generation_config`` is
    a SEPARATE object loaded from ``generation_config.json``, it is what
    ``generate`` actually reads, and ``save_pretrained`` writes it back
    verbatim.  A child whose ``generation_config.json`` still names the parent's
    ``eos`` can never emit ``eos``: every generation runs to
    ``max_new_tokens``.  Worse for this campaign specifically, the
    artifact-tolerant grading rule ("missing-EOS never counts against
    intelligence") makes a frozen review structurally BLIND to it while paying
    the full generation cost.
  * a remap is not verifiable by inspection.  :func:`assert_ids_addressable` is
    the post-condition: after the fold, every integer id named by the child's
    config, sub-configs or generation config must be ``< child vocab_size``.  It
    runs inside :func:`vocab_axis`, again over the files in
    :func:`write_tokenizer`, and once more on the re-loaded child in
    ``qf_export.verify_child``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import qf_arch


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2/ByteLevel byte -> printable-character table, verbatim.

    Every ByteLevel BPE (Qwen's included) writes raw bytes as these 256
    characters: the printable ASCII/Latin-1 ranges map to themselves and the
    remaining 68 bytes are pushed up into U+0100.. so that no token contains a
    control character or a bare space.  This is the definition of "the byte
    alphabet"; anything else is a guess.
    """

    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = list(printable)
    shifted = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shifted)
            shifted += 1
    return {byte: chr(code) for byte, code in zip(printable, codes)}


class Vocabulary:
    """A ``tokenizer.json`` BPE table, read for what a coreset needs to know."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        body = json.loads(self.path.read_text(encoding="utf-8"))
        model = body.get("model") or {}
        if model.get("type") != "BPE":
            raise SystemExit(f"{path}: tokenizer model is {model.get('type')!r}, not BPE")
        self.body = body
        self.vocab: dict[str, int] = dict(model["vocab"])
        self.by_id: dict[int, str] = {int(v): k for k, v in self.vocab.items()}
        if len(self.by_id) != len(self.vocab):
            raise SystemExit(f"{path}: the vocab maps two tokens to one id")
        self.merges: list[Any] = list(model.get("merges") or [])
        self.added: list[dict[str, Any]] = list(body.get("added_tokens") or [])
        self._merge_pairs = [self._pair(entry) for entry in self.merges]
        self._parents: dict[str, tuple[str, str]] = {}
        for left, right in self._merge_pairs:
            self._parents.setdefault(left + right, (left, right))

    @staticmethod
    def _pair(entry: Any) -> tuple[str, str]:
        """Merges are stored either as ``"a b"`` or as ``["a", "b"]``.

        Both forms are in the wild (the Qwen 4B ships the string form, 247,587
        of them).  Splitting on the FIRST space only -- a token may itself be a
        space.
        """

        if isinstance(entry, str):
            left, _, right = entry.partition(" ")
            return left, right
        return str(entry[0]), str(entry[1])

    def added_ids(self) -> set[int]:
        return {int(row["id"]) for row in self.added}

    def byte_alphabet(self) -> set[int]:
        """The rows holding the 256 tokens a ByteLevel BPE spells raw bytes with.

        Named from the CANONICAL ``bytes_to_unicode`` table, not by scanning for
        ``len(token) == 1``.  Those are not the same set on a real parent: Qwen's
        248,320-row table has thousands of single-character tokens that are CJK
        ideographs, accented Latin and symbols, none of which is a byte token.
        Reserving all of them would spend an unmeasured slice of the keep budget
        on tokens the evidence may never use and would report that slice as "the
        byte alphabet".
        """

        table = bytes_to_unicode()
        missing = sorted(
            byte for byte, character in table.items() if character not in self.vocab
        )
        if missing:
            raise SystemExit(
                f"{self.path}: {len(missing)} of the 256 ByteLevel byte tokens are "
                f"not in this vocabulary (first bytes: {missing[:8]}); it cannot "
                f"spell arbitrary text and a coreset of it cannot either"
            )
        return {self.vocab[character] for character in table.values()}

    def parents_of(self, token: str) -> tuple[str, str] | None:
        return self._parents.get(token)

    def dead_ids(self, vocab_size: int) -> list[int]:
        """Rows the model has and the tokenizer does not name."""

        return sorted(set(range(vocab_size)) - set(self.by_id))


def mandatory_ids(vocab: Vocabulary, hard_ids: set[int]) -> set[int]:
    """Added tokens, the byte alphabet, the config's hard ids, and their closure."""

    return mandatory_breakdown(vocab, hard_ids)["ids"]


def mandatory_breakdown(vocab: Vocabulary, hard_ids: set[int]) -> dict[str, Any]:
    """The same set, with each reason counted separately for the receipt.

    A receipt that says "260 mandatory rows (added + byte alphabet + hard ids +
    closure)" and cannot say how many came from where is describing a number
    nobody can check.
    """

    added = set(vocab.added_ids())
    alphabet = set(vocab.byte_alphabet())
    for index in sorted(hard_ids):
        if index not in vocab.by_id:
            raise SystemExit(
                f"config names id {index} but the tokenizer has no such row; a "
                f"coreset cannot preserve a token that is not there"
            )
    required = added | alphabet | set(hard_ids)
    closed = close_over_merges(vocab, required)
    return {
        "ids": closed,
        "added_rows": len(added),
        "byte_alphabet_rows": len(alphabet),
        "hard_id_rows": len(set(hard_ids) - added - alphabet),
        "closure_added_rows": len(closed - required),
        "mandatory_rows": len(closed),
    }


def close_over_merges(vocab: Vocabulary, keep: set[int]) -> set[int]:
    """Add every ancestor of every kept token, transitively."""

    frontier = list(keep)
    closed = set(keep)
    while frontier:
        token = vocab.by_id[frontier.pop()]
        parents = vocab.parents_of(token)
        if not parents:
            continue
        for parent in parents:
            index = vocab.vocab.get(parent)
            if index is None:
                continue
            if index not in closed:
                closed.add(index)
                frontier.append(index)
    return closed


def _ancestor_index(vocab: Vocabulary) -> dict[int, set[int]]:
    """``{id: the id and every id it is transitively built from}``, memoised once.

    The naive fill re-closes the whole keep set for every candidate, which is
    quadratic in a 248,320-row table.  A merge's parents always have SMALLER ids
    than the token they compose (BPE numbers tokens in merge order), so one
    ascending pass builds every closure.
    """

    ancestry: dict[int, set[int]] = {}
    for index in sorted(vocab.by_id):
        token = vocab.by_id[index]
        closure = {index}
        parents = vocab.parents_of(token)
        if parents:
            for parent in parents:
                parent_index = vocab.vocab.get(parent)
                if parent_index is None:
                    continue
                closure |= ancestry.get(parent_index, {parent_index})
        ancestry[index] = closure
    return ancestry


def choose_kept(vocab: Vocabulary, counts: dict[int, int], k: int, hard_ids: set[int]) -> list[int]:
    """The kept id set of size EXACTLY ``k``, closure-aware.

    Ranked by ``(-count, id)`` so ties are broken deterministically by the
    parent's own numbering, then filled one candidate at a time WITH its
    ancestors, because admitting a token admits everything it is built from and
    a fill that ignores that overshoots ``k``.
    """

    breakdown = mandatory_breakdown(vocab, hard_ids)
    mandatory = breakdown["ids"]
    if len(mandatory) > k:
        raise SystemExit(
            f"{len(mandatory)} tokens are mandatory and k is {k}; this width cannot "
            f"be spelled.  {breakdown['added_rows']} added tokens + "
            f"{breakdown['byte_alphabet_rows']} ByteLevel byte-alphabet rows + "
            f"{breakdown['hard_id_rows']} config hard ids + "
            f"{breakdown['closure_added_rows']} merge ancestors of those"
        )
    kept = set(mandatory)
    ancestry = _ancestor_index(vocab)
    ranked = sorted(vocab.by_id, key=lambda index: (-counts.get(index, 0), index))
    for index in ranked:
        if len(kept) >= k:
            break
        if index in kept:
            continue
        addition = ancestry[index] - kept
        if len(kept) + len(addition) <= k:
            kept |= addition
    if len(kept) != k:
        raise SystemExit(
            f"closure-aware fill reached {len(kept)} of {k}; every remaining "
            f"candidate drags in more ancestors than the budget allows"
        )
    return sorted(kept)


def token_counts(manifest: dict[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in manifest["rows"]:
        for index in row["ids"]:
            counts[int(index)] = counts.get(int(index), 0) + 1
    return counts


def merge_closure_intact(vocab: Vocabulary, kept: set[int]) -> None:
    for index in sorted(kept):
        parents = vocab.parents_of(vocab.by_id[index])
        if parents is None:
            continue
        for parent in parents:
            parent_index = vocab.vocab.get(parent)
            if parent_index is not None and parent_index not in kept:
                raise SystemExit(
                    f"kept token {vocab.by_id[index]!r} is built from {parent!r}, "
                    f"which was dropped"
                )


def remapped_tokenizer(vocab: Vocabulary, kept: list[int]) -> tuple[dict[str, Any], dict[int, int]]:
    """A real BPE over the kept tokens, densely renumbered, with the merges pruned."""

    order = {old: new for new, old in enumerate(kept)}
    tokens = {vocab.by_id[old]: new for old, new in order.items()}
    body = json.loads(json.dumps(vocab.body))
    body["model"]["vocab"] = tokens
    body["model"]["merges"] = [
        entry
        for entry, (left, right) in zip(vocab.merges, vocab._merge_pairs)
        if left in tokens and right in tokens and (left + right) in tokens
    ]
    body["added_tokens"] = [
        {**row, "id": order[int(row["id"])]}
        for row in vocab.added
        if int(row["id"]) in order
    ]
    return body, order


def gather_rows(
    tensor: torch.Tensor, kept: list[int]
) -> torch.Tensor:
    index = torch.tensor(kept, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index).contiguous()


# ------------------------------------------------------------------- hard ids
#: Fields on the TEXT config that name a token id.  Scalar or list.
TEXT_ID_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id",
                  "decoder_start_token_id")

#: Fields on the TOP-level (multimodal) config that name a token id.
TOP_ID_FIELDS = ("image_token_id", "video_token_id", "vision_start_token_id",
                 "vision_end_token_id", "audio_token_id")

#: Fields on the GENERATION config that name a token id.  ``generate`` reads
#: this object and not the model config, so a fold that renumbers rows without
#: touching it ships a child that cannot emit its own stop token.
GENERATION_ID_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id",
                        "decoder_start_token_id", "forced_bos_token_id",
                        "forced_eos_token_id")

#: Generation fields that name token ids in shapes this module does NOT
#: renumber (nested pairs, ranges, ``[[id, ...], ...]``).  Refused rather than
#: silently carried, because carrying them is the same defect the whole hard-id
#: block exists to prevent.
GENERATION_TOKEN_LISTS = ("suppress_tokens", "begin_suppress_tokens",
                          "bad_words_ids", "forced_decoder_ids")

#: Key suffixes/names that :func:`assert_ids_in_mapping` treats as token ids.
ID_KEY_SUFFIXES = ("_token_id", "_token_index")


def id_field_value(value: Any) -> list[int] | None:
    """``value`` as a list of token ids, or ``None`` if it names none.

    ``bool`` is excluded explicitly: it is an ``int`` subclass, and a config
    flag that happens to be named like an id must not be renumbered.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)) and len(value) > 0 and all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in value
    ):
        return [int(entry) for entry in value]
    return None


def remap_id_field(value: Any, order: dict[int, int], *, where: str) -> Any:
    """Renumber one hard-id field through ``order``, keeping its shape.

    A dropped id is a refusal and not a substitution: there is no other row that
    means what ``eos`` meant.
    """

    def one(entry: Any) -> int:
        index = int(entry)
        if index not in order:
            raise SystemExit(
                f"{where} names token id {index}, which the coreset dropped; a "
                f"vocabulary child cannot be built that cannot spell its own "
                f"special tokens"
            )
        return order[index]

    if isinstance(value, list):
        return [one(entry) for entry in value]
    if isinstance(value, tuple):
        return tuple(one(entry) for entry in value)
    return one(value)


def hard_id_fields(tc: Any, top: Any, generation_config: Any = None) -> dict[str, list[int]]:
    """``{where: [ids]}`` over the text config, the top config and generation.

    ``where`` is the dotted path the receipt prints, and it is also the key the
    remap writes back through, so the two can never drift apart.
    """

    fields: dict[str, list[int]] = {}
    for field in TEXT_ID_FIELDS:
        ids = id_field_value(getattr(tc, field, None))
        if ids is not None:
            fields[f"text_config.{field}"] = ids
    if top is not tc:
        for field in TOP_ID_FIELDS:
            ids = id_field_value(getattr(top, field, None))
            if ids is not None:
                fields[field] = ids
    if generation_config is not None:
        for field in GENERATION_ID_FIELDS:
            ids = id_field_value(getattr(generation_config, field, None))
            if ids is not None:
                fields[f"generation_config.{field}"] = ids
        for field in GENERATION_TOKEN_LISTS:
            value = getattr(generation_config, field, None)
            if value:
                raise SystemExit(
                    f"generation_config.{field} is set ({value!r}); this lane does "
                    f"not renumber that shape, and carrying it verbatim past a row "
                    f"remap points it at whatever token inherited the number.  "
                    f"Clear it on the parent, or extend remap_id_field first"
                )
    return fields


def assert_ids_in_mapping(mapping: dict[str, Any], vocab_size: int, *, where: str) -> int:
    """Every token id anywhere in ``mapping`` is a row the child actually has.

    Walks nested dicts (a multimodal config's ``text_config``/``vision_config``
    live there) and checks any key ending in :data:`ID_KEY_SUFFIXES`.  Returns
    the number of fields checked so a caller can prove the walk was not empty.
    """

    offenders: list[str] = []
    checked = 0

    def walk(prefix: str, node: dict[str, Any]) -> None:
        nonlocal checked
        for raw_key, value in node.items():
            # `str()` because a config dict is not string-keyed throughout:
            # `id2label` is ``{0: 'LABEL_0', ...}`` and walking it with a bare
            # `key.endswith` is an AttributeError on the first entry.
            key = str(raw_key)
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(f"{name}.", value)
                continue
            if not any(key.endswith(suffix) for suffix in ID_KEY_SUFFIXES):
                continue
            ids = id_field_value(value)
            if ids is None:
                continue
            checked += 1
            for index in ids:
                if not 0 <= index < vocab_size:
                    offenders.append(f"{name}={index}")

    walk("", mapping)
    if offenders:
        raise SystemExit(
            f"{where}: {len(offenders)} token id(s) fall outside the child's "
            f"vocabulary of {vocab_size} rows ({', '.join(sorted(offenders))}); "
            f"the id-space remap did not reach them"
        )
    return checked


def assert_ids_addressable(
    config: Any, generation_config: Any, vocab_size: int, *, where: str = "child"
) -> dict[str, int]:
    """The C1 post-condition, on the live objects.

    Cheap, total, and run three times on purpose: inside :func:`vocab_axis` (so
    a bad remap never reaches disk), over the files in :func:`write_tokenizer`,
    and on the RE-LOADED child in ``qf_export.verify_child`` (so the check is
    also made by code that never saw the fold).
    """

    checked = assert_ids_in_mapping(config.to_dict(), vocab_size, where=f"{where} config")
    generation_checked = 0
    if generation_config is not None:
        generation_checked = assert_ids_in_mapping(
            generation_config.to_dict(), vocab_size, where=f"{where} generation_config"
        )
    return {
        "config_id_fields_checked": checked,
        "generation_config_id_fields_checked": generation_checked,
        "vocab_size": int(vocab_size),
    }


def vocab_axis(
    harness,
    *,
    keep: int,
    tokenizer_path: str | Path,
    counts: dict[int, int] | None = None,
    log=print,
) -> dict[str, Any]:
    """Fold the embedding and the head to ``keep`` rows and rewrite the config ids.

    The tied case and the untied case are both handled and the difference is
    material: at 0.8B/4B the head IS the embedding (``_tied_weights_keys``,
    modeling 1614/1710), so the fold removes ``(V-k)*H`` parameters; at 27B the
    head is its own tensor and the same fold removes twice that.
    """

    model = harness.model
    tc = harness.text_config
    top = model.config
    counts = counts if counts is not None else token_counts(harness.manifest)

    vocab = Vocabulary(tokenizer_path)
    # Scalars AND lists, config AND generation config, in one table keyed by the
    # dotted path so the protection, the remap and the receipt cannot disagree.
    generation = getattr(model, "generation_config", None)
    hard = hard_id_fields(tc, top, generation)
    hard_values = {index for ids in hard.values() for index in ids}

    parent_vocab = int(tc.vocab_size)
    dead = vocab.dead_ids(parent_vocab)
    breakdown = mandatory_breakdown(vocab, hard_values)
    kept = choose_kept(vocab, counts, keep, hard_values)
    merge_closure_intact(vocab, set(kept))
    body, order = remapped_tokenizer(vocab, kept)

    embed = qf_arch.embed(model)
    head = qf_arch.lm_head(model)
    tied = embed.weight.data_ptr() == head.weight.data_ptr()
    hidden = int(embed.weight.shape[1])

    new_embed = torch.nn.Embedding(keep, hidden, dtype=embed.weight.dtype,
                                   device=embed.weight.device)
    new_embed.weight.data.copy_(gather_rows(embed.weight.data, kept))
    new_head = torch.nn.Linear(hidden, keep, bias=False, dtype=head.weight.dtype,
                               device=head.weight.device)
    new_head.weight.data.copy_(
        new_embed.weight.data if tied else gather_rows(head.weight.data, kept)
    )
    qf_arch.text_model(model).embed_tokens = new_embed
    model.lm_head = new_head
    if tied:
        model.lm_head.weight = new_embed.weight

    qf_arch.config_field(tc, "vocab_size", keep)
    remapped_hard: dict[str, Any] = {}
    generation_ids: dict[str, Any] = {}
    for where in hard:
        name = where.split(".")[-1]
        if where.startswith("generation_config."):
            # A GenerationConfig is a plain object, not the @strict dataclass
            # `config_field` validates against, so it is set directly.
            parent_value = getattr(generation, name)
            child_value = remap_id_field(parent_value, order, where=where)
            setattr(generation, name, child_value)
            generation_ids[name] = child_value
        else:
            target = tc if where.startswith("text_config.") else top
            parent_value = getattr(target, name)
            child_value = remap_id_field(parent_value, order, where=where)
            qf_arch.config_field(target, name, child_value)
        remapped_hard[where] = {"parent": parent_value, "child": child_value}

    # Post-condition, on the live objects, before anything reaches disk.
    addressable = assert_ids_addressable(model.config, generation, keep, where="live child")

    log(
        f"[qf] C1 vocab {parent_vocab} -> {keep} ({parent_vocab / keep:.2f}x), "
        f"{'tied' if tied else 'untied'}, {len(dead)} dead rows dropped free"
    )
    return {
        "axis": "vocab_size",
        "kind": "row coreset (deletion) + merge-closed tokenizer remap",
        "parent_vocab": parent_vocab,
        "child_vocab": keep,
        "component_ratio": keep / float(parent_vocab),
        "tied": bool(tied),
        "dead_parent_rows": len(dead),
        "mandatory_rows": breakdown["mandatory_rows"],
        "mandatory_breakdown": {
            "added_rows": breakdown["added_rows"],
            "byte_alphabet_rows": breakdown["byte_alphabet_rows"],
            "hard_id_rows": breakdown["hard_id_rows"],
            "closure_added_rows": breakdown["closure_added_rows"],
            "byte_alphabet_source": "canonical ByteLevel bytes_to_unicode table",
        },
        "merges_parent": len(vocab.merges),
        "merges_child": len(body["model"]["merges"]),
        "hard_ids": remapped_hard,
        "generation_config_ids": generation_ids,
        "ids_addressable": addressable,
        "parameters_removed": (parent_vocab - keep) * hidden * (1 if tied else 2),
        "kept_ids": kept,
        "tokenizer_body": body,
    }


def write_tokenizer(report: dict[str, Any], out_dir: str | Path, source_dir: str | Path) -> list[str]:
    """Write the child's ``tokenizer.json`` and regenerate the BPE sidecars.

    ``vocab.json``/``merges.txt`` are a SECOND copy of the same table that some
    loaders prefer; leaving the parent's beside a remapped ``tokenizer.json``
    gives a directory that tokenizes differently depending on which file wins.

    ``generation_config.json`` is the same failure one level up: it is a third
    copy of the special-token ids, it is the copy ``generate`` reads, and
    ``qf_export.SIDECARS`` will happily copy the parent's if the child has none.
    It is rewritten here from the remapped ids and then re-checked against the
    child's vocabulary, so the refusal happens on the FILE and not only on the
    live object that produced it.
    """

    out_dir, source_dir = Path(out_dir), Path(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = report["tokenizer_body"]
    written = []

    (out_dir / "tokenizer.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append("tokenizer.json")

    if (source_dir / "vocab.json").exists():
        (out_dir / "vocab.json").write_text(
            json.dumps(body["model"]["vocab"], ensure_ascii=False), encoding="utf-8"
        )
        written.append("vocab.json")
    if (source_dir / "merges.txt").exists():
        lines = []
        for entry in body["model"]["merges"]:
            left, right = Vocabulary._pair(entry)
            lines.append(f"{left} {right}")
        (out_dir / "merges.txt").write_text(
            "#version: 0.2\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )
        written.append("merges.txt")

    config_path = source_dir / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        added = config.get("added_tokens_decoder")
        if isinstance(added, dict):
            renumbered = {}
            order = {int(old): new for new, old in enumerate(report["kept_ids"])}
            for key, value in added.items():
                if int(key) in order:
                    renumbered[str(order[int(key)])] = value
            config["added_tokens_decoder"] = renumbered
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append("tokenizer_config.json")

    # `save_pretrained` has usually already written the child's own (remapped)
    # generation_config.json; the parent's is the fallback, and the ids are
    # overwritten from the report either way so neither route can leave a
    # parent-space id behind.
    child_generation = out_dir / "generation_config.json"
    parent_generation = source_dir / "generation_config.json"
    base: dict[str, Any] | None = None
    if child_generation.exists():
        base = json.loads(child_generation.read_text(encoding="utf-8"))
    elif parent_generation.exists():
        base = json.loads(parent_generation.read_text(encoding="utf-8"))
    if base is not None:
        for field, value in (report.get("generation_config_ids") or {}).items():
            base[field] = value
        child_generation.write_text(
            json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        assert_ids_in_mapping(
            base, int(report["child_vocab"]), where="child generation_config.json"
        )
        written.append("generation_config.json")
    return written


def remap_manifest(
    manifest: dict[str, Any],
    kept_ids: list[int],
    out: str | Path,
    *,
    child_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """Rename a sealed stream into a vocabulary child's id space.

    A vocabulary fold renames rows, so every capture taken after it is on the
    CHILD's table and the parent's id stream is not a stream of that model's
    tokens.  This is the id-space rename, which is exact when every id in the
    stream survived the coreset; when one did not, the honest operation is to
    re-tokenize the SOURCE TEXT with the child's tokenizer (mx45's
    ``retokenize``) and this function refuses rather than substituting a token
    the parent never emitted at that position.

    ``child_tokenizer_sha256`` is the sha256 of the tokenizer.json the C1 child
    just wrote.  The remapped stream belongs to THAT table, so it is sealed under
    that digest and the next build's consumption-time tokenizer check
    (:func:`qf_build.assert_evidence_tokenizer`) passes on a measurement instead
    of on an exception.  Left unset, the manifest carries the parent's digest
    with a ``|remapped`` marker, which no tokenizer.json on disk can match --
    deliberately, because an unlabelled remap would be indistinguishable from
    the parent's own stream.
    """

    from qf_corpus import write_manifest

    order = {old: new for new, old in enumerate(kept_ids)}
    missing = sorted({
        int(i) for row in manifest["rows"] for i in row["ids"] if int(i) not in order
    })
    if missing:
        raise SystemExit(
            f"{len(missing)} distinct ids in the sealed stream were dropped by the "
            f"coreset (first: {missing[:8]}); re-tokenize the source text with the "
            f"child's tokenizer instead of renaming this one"
        )
    return write_manifest(
        [
            {"batch_index": row["batch_index"], "split": row["split"],
             "ids": [order[int(i)] for i in row["ids"]]}
            for row in manifest["rows"]
        ],
        tokenizer_sha256=(
            child_tokenizer_sha256 or manifest["tokenizer_sha256"] + "|remapped"
        ),
        source=manifest["source"] + " | qf_vocab.remap_manifest",
        out=out,
    )


def synthetic_tokenizer(vocab_size: int, out: str | Path, *, merges: int = 200) -> Path:
    """A minimal but REAL ByteLevel BPE for the tiny-model self-tests.

    The CANONICAL 256 byte tokens (:func:`bytes_to_unicode`, so the fixture is a
    real ByteLevel alphabet rather than a block of 256 arbitrary characters that
    happen to be one character long), ``merges`` composed tokens each built from
    two tokens already in the table (so the closure is non-trivial and the
    coreset has something to close over), and four added specials.  It is small,
    but every property C1 asserts about a tokenizer is true of it.
    """

    alphabet = [bytes_to_unicode()[byte] for byte in range(256)]
    vocab = {token: index for index, token in enumerate(alphabet)}
    merge_list: list[str] = []
    cursor = 256
    tokens = list(alphabet)
    index = 0
    while len(merge_list) < merges and cursor < vocab_size - 4:
        left = tokens[index % len(tokens)]
        right = tokens[(index * 7 + 3) % len(tokens)]
        composed = left + right
        index += 1
        if composed in vocab:
            continue
        vocab[composed] = cursor
        merge_list.append(f"{left} {right}")
        tokens.append(composed)
        cursor += 1
    added = []
    for name in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_pad|>"):
        vocab[name] = cursor
        added.append(
            {"id": cursor, "content": name, "special": True, "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False}
        )
        cursor += 1
    body = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True,
                           "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab, "merges": merge_list,
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
