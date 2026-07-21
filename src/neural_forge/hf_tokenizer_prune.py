from __future__ import annotations

import json
import operator
from collections import Counter
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_forge.hf_token_align import StrictTokenTransportProfile
from neural_forge.hf_token_align import build_strict_token_transport_profile


class HFTokenizerPruneError(ValueError):
    """Raised when a byte-level BPE tokenizer cannot be pruned safely."""


@dataclass(frozen=True)
class HFTokenizerPruneResult:
    tokenizer: Any
    output_dir: Path
    target_vocab_size: int
    selected_donor_token_ids: tuple[int, ...]
    required_donor_token_ids: tuple[int, ...]
    required_donor_token_dependency_ids: tuple[int, ...]
    mandatory_donor_token_ids: tuple[int, ...]
    retained_merges: tuple[tuple[str, str], ...]
    transport_profile: StrictTokenTransportProfile

    @property
    def transport_fingerprint(self) -> str:
        return self.transport_profile.fingerprint()


@dataclass(frozen=True)
class _Merge:
    left: str
    right: str
    result: str
    serialized: Any


class _ExactTokenSurfaceView:
    """Tokenizer protocol view over raw BPE token strings.

    Byte-level token rows representing incomplete UTF-8 sequences cannot be
    audited through ordinary text decoding. This view lets the existing strict
    transport builder audit the actual BPE row identity instead.
    """

    def __init__(self, tokenizer: Any) -> None:
        vocabulary = tokenizer.get_vocab()
        self._tokens = tuple(
            tokenizer.convert_ids_to_tokens(token_id)
            for token_id in range(len(tokenizer))
        )
        self._vocabulary = dict(vocabulary)
        self.all_special_ids = list(tokenizer.all_special_ids)
        for attribute in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        ):
            setattr(self, attribute, getattr(tokenizer, attribute, None))

    def __len__(self) -> int:
        return len(self._tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)

    def decode(self, token_ids: Sequence[int], **_: Any) -> str:
        return "".join(self._tokens[int(token_id)] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise ValueError("The strict surface view does not add special tokens.")
        token_id = self._vocabulary.get(text)
        if token_id is None:
            raise ValueError(f"No exact raw token surface for {text!r}.")
        return [token_id]


def prune_hf_tokenizer(
    donor_tokenizer: Any,
    output_dir: str | Path,
    *,
    target_vocab_size: int,
    calibration_texts: Sequence[str] | None = None,
    required_donor_token_ids: Sequence[int] | None = None,
) -> HFTokenizerPruneResult:
    """Build and save a deterministic strict-transport BPE tokenizer subset.

    The target vocabulary size includes added special tokens. Selection keeps
    the complete ByteLevel alphabet, every semantic special token, and complete
    BPE merge dependency chains. Explicit required donor rows and their complete
    merge dependency closures are selected before calibration frequency and the
    donor's original token-ID order.
    """

    target_size = _positive_int(target_vocab_size, "target_vocab_size")
    texts = _calibration_texts(calibration_texts)
    backend = getattr(donor_tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(donor_tokenizer, "_tokenizer", None)
    if backend is None:
        raise HFTokenizerPruneError("donor_tokenizer must be a fast HF tokenizer")

    try:
        tokenizer_data = json.loads(backend.to_str())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HFTokenizerPruneError(
            "donor_tokenizer must expose a valid structured tokenizer JSON"
        ) from exc

    model = tokenizer_data.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise HFTokenizerPruneError("only Hugging Face BPE tokenizers are supported")
    if not _contains_component(tokenizer_data.get("pre_tokenizer"), "ByteLevel"):
        raise HFTokenizerPruneError(
            "the donor BPE tokenizer must use a ByteLevel pre-tokenizer"
        )
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise HFTokenizerPruneError(
            "continuing-subword and end-of-word BPE variants are not supported"
        )

    vocabulary = _model_vocabulary(model)
    donor_vocab = donor_tokenizer.get_vocab()
    _validate_donor_ids(donor_tokenizer, donor_vocab)
    if target_size > len(donor_tokenizer):
        raise HFTokenizerPruneError(
            f"target_vocab_size {target_size} exceeds donor size {len(donor_tokenizer)}"
        )
    forced_ids = _normalized_required_donor_token_ids(
        required_donor_token_ids,
        donor_vocab_size=len(donor_tokenizer),
    )

    merges = _parse_merges(model.get("merges", ()), vocabulary)
    producer = _producer_map(merges)
    required_ids = _required_token_ids(
        donor_tokenizer,
        tokenizer_data,
        vocabulary,
    )
    forced_dependency_ids = _merge_dependency_closure(
        forced_ids,
        vocabulary,
        producer,
    )
    mandatory_ids = _merge_dependency_closure(
        required_ids.union(forced_ids),
        vocabulary,
        producer,
    )
    if forced_ids:
        if len(mandatory_ids) > target_size:
            raise HFTokenizerPruneError(
                f"target_vocab_size {target_size} is too small; byte coverage, "
                "special tokens, required donor token IDs, and BPE merge "
                "dependencies require at least "
                f"{len(mandatory_ids)} rows"
            )
        required_ids.update(forced_ids)
    selected_ids = _select_token_ids(
        donor_tokenizer,
        target_size,
        texts,
        vocabulary,
        producer,
        required_ids,
    )
    id_map = {
        donor_id: target_id for target_id, donor_id in enumerate(sorted(selected_ids))
    }

    pruned_data, retained_merges = _pruned_tokenizer_data(
        tokenizer_data,
        vocabulary,
        merges,
        selected_ids,
        id_map,
    )
    target_tokenizer = _build_hf_tokenizer(donor_tokenizer, pruned_data)
    selected_tuple = tuple(sorted(selected_ids))
    _verify_target(
        donor_tokenizer,
        target_tokenizer,
        target_size,
        texts,
        selected_tuple,
    )
    profile = build_strict_token_transport_profile(
        _ExactTokenSurfaceView(donor_tokenizer),
        _ExactTokenSurfaceView(target_tokenizer),
        donor_row_capacity=len(donor_tokenizer),
        target_row_capacity=target_size,
    )
    expected_mapping = selected_tuple
    if profile.target_to_donor_row_indices != expected_mapping:
        raise HFTokenizerPruneError(
            "strict transport profile does not match the selected donor rows"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target_tokenizer.save_pretrained(destination)
    reloaded = _reload_hf_tokenizer(destination)
    _verify_target(
        donor_tokenizer,
        reloaded,
        target_size,
        texts,
        selected_tuple,
    )
    saved_profile = build_strict_token_transport_profile(
        _ExactTokenSurfaceView(donor_tokenizer),
        _ExactTokenSurfaceView(reloaded),
        donor_row_capacity=len(donor_tokenizer),
        target_row_capacity=target_size,
    )
    if saved_profile.fingerprint() != profile.fingerprint():
        raise HFTokenizerPruneError(
            "saved tokenizer changed the strict transport fingerprint"
        )

    return HFTokenizerPruneResult(
        tokenizer=reloaded,
        output_dir=destination,
        target_vocab_size=target_size,
        selected_donor_token_ids=selected_tuple,
        required_donor_token_ids=forced_ids,
        required_donor_token_dependency_ids=tuple(
            sorted(forced_dependency_ids)
        ),
        mandatory_donor_token_ids=tuple(sorted(mandatory_ids)),
        retained_merges=tuple((merge.left, merge.right) for merge in retained_merges),
        transport_profile=saved_profile,
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise HFTokenizerPruneError(f"{label} must be a positive integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFTokenizerPruneError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise HFTokenizerPruneError(f"{label} must be a positive integer")
    return result


def _calibration_texts(texts: Sequence[str] | None) -> tuple[str, ...]:
    if texts is None:
        return ()
    if isinstance(texts, (str, bytes)):
        raise HFTokenizerPruneError("calibration_texts must be a sequence of strings")
    result = tuple(texts)
    if any(not isinstance(text, str) for text in result):
        raise HFTokenizerPruneError("calibration_texts must contain only strings")
    return result


def _normalized_required_donor_token_ids(
    values: Sequence[int] | None,
    *,
    donor_vocab_size: int,
) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise HFTokenizerPruneError(
            "required_donor_token_ids must be a sequence of integers"
        )
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise HFTokenizerPruneError(
            "required_donor_token_ids must be a sequence of integers"
        ) from exc
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in supplied:
        if isinstance(raw_value, bool):
            raise HFTokenizerPruneError(
                "required_donor_token_ids must contain only integers"
            )
        try:
            token_id = operator.index(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFTokenizerPruneError(
                "required_donor_token_ids must contain only integers"
            ) from exc
        if token_id < 0 or token_id >= donor_vocab_size:
            raise HFTokenizerPruneError(
                f"required donor token ID {token_id} is outside donor vocabulary "
                f"[0, {donor_vocab_size})"
            )
        if token_id in seen:
            raise HFTokenizerPruneError(
                f"required donor token ID {token_id} is duplicated"
            )
        seen.add(token_id)
        normalized.append(token_id)
    return tuple(sorted(normalized))


def _contains_component(component: Any, component_type: str) -> bool:
    if isinstance(component, Mapping):
        if component.get("type") == component_type:
            return True
        return any(
            _contains_component(value, component_type) for value in component.values()
        )
    if isinstance(component, list):
        return any(_contains_component(value, component_type) for value in component)
    return False


def _model_vocabulary(model: Mapping[str, Any]) -> dict[str, int]:
    raw_vocab = model.get("vocab")
    if not isinstance(raw_vocab, Mapping) or not raw_vocab:
        raise HFTokenizerPruneError("BPE model vocabulary must be a non-empty mapping")
    result: dict[str, int] = {}
    for token, raw_id in raw_vocab.items():
        if not isinstance(token, str) or isinstance(raw_id, bool):
            raise HFTokenizerPruneError("BPE model vocabulary is malformed")
        try:
            token_id = operator.index(raw_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HFTokenizerPruneError("BPE model vocabulary is malformed") from exc
        result[token] = token_id
    return result


def _validate_donor_ids(tokenizer: Any, vocabulary: Mapping[str, int]) -> None:
    actual = set(vocabulary.values())
    expected = set(range(len(tokenizer)))
    if actual != expected:
        raise HFTokenizerPruneError(
            "donor tokenizer vocabulary IDs must be contiguous from zero"
        )


def _parse_merges(
    raw_merges: Any,
    vocabulary: Mapping[str, int],
) -> tuple[_Merge, ...]:
    if not isinstance(raw_merges, list):
        raise HFTokenizerPruneError("BPE merges must be a structured list")
    result: list[_Merge] = []
    for raw_merge in raw_merges:
        if isinstance(raw_merge, list) and len(raw_merge) == 2:
            left, right = raw_merge
        elif isinstance(raw_merge, str):
            pieces = raw_merge.split(" ")
            if len(pieces) != 2:
                raise HFTokenizerPruneError(f"malformed BPE merge {raw_merge!r}")
            left, right = pieces
        else:
            raise HFTokenizerPruneError(f"malformed BPE merge {raw_merge!r}")
        if not isinstance(left, str) or not isinstance(right, str):
            raise HFTokenizerPruneError(f"malformed BPE merge {raw_merge!r}")
        merged = left + right
        if (
            left not in vocabulary
            or right not in vocabulary
            or merged not in vocabulary
        ):
            raise HFTokenizerPruneError(
                f"BPE merge {left!r} + {right!r} references a missing token"
            )
        result.append(_Merge(left, right, merged, raw_merge))
    return tuple(result)


def _producer_map(merges: Sequence[_Merge]) -> dict[str, _Merge]:
    result: dict[str, _Merge] = {}
    for merge in merges:
        result.setdefault(merge.result, merge)
    return result


def _required_token_ids(
    tokenizer: Any,
    tokenizer_data: Mapping[str, Any],
    vocabulary: Mapping[str, int],
) -> set[int]:
    try:
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as exc:
        raise HFTokenizerPruneError("tokenizers is required") from exc

    alphabet = set(ByteLevel.alphabet())
    missing_bytes = sorted(alphabet - vocabulary.keys())
    if missing_bytes:
        raise HFTokenizerPruneError(
            "donor BPE vocabulary lacks required ByteLevel byte atoms: "
            + ", ".join(repr(token) for token in missing_bytes[:8])
        )
    required = {vocabulary[token] for token in alphabet}
    required.update(int(token_id) for token_id in tokenizer.all_special_ids)

    added_tokens = tokenizer_data.get("added_tokens", ())
    if not isinstance(added_tokens, list):
        raise HFTokenizerPruneError("added_tokens must be a structured list")
    for item in added_tokens:
        if not isinstance(item, Mapping):
            raise HFTokenizerPruneError("added_tokens contains a malformed entry")
        if item.get("special") is True:
            required.add(int(item["id"]))

    model = tokenizer_data["model"]
    unk_token = model.get("unk_token")
    if unk_token is not None:
        if unk_token not in vocabulary:
            raise HFTokenizerPruneError("BPE unk_token is missing from its vocabulary")
        required.add(vocabulary[unk_token])
    required.update(_post_processor_token_ids(tokenizer_data.get("post_processor")))
    return required


def _post_processor_token_ids(component: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(component, Mapping):
        special_tokens = component.get("special_tokens")
        if isinstance(special_tokens, Mapping):
            for value in special_tokens.values():
                if isinstance(value, Mapping) and isinstance(value.get("ids"), list):
                    result.update(int(item) for item in value["ids"])
        for value in component.values():
            result.update(_post_processor_token_ids(value))
    elif isinstance(component, list):
        for value in component:
            result.update(_post_processor_token_ids(value))
    return result


def _merge_dependency_closure(
    token_ids: Sequence[int] | set[int],
    vocabulary: Mapping[str, int],
    producer: Mapping[str, _Merge],
) -> set[int]:
    token_by_id = {token_id: token for token, token_id in vocabulary.items()}
    closure_cache: dict[int, frozenset[int]] = {}

    def closure(
        token_id: int,
        active: frozenset[int] = frozenset(),
    ) -> frozenset[int]:
        if token_id in closure_cache:
            return closure_cache[token_id]
        if token_id in active:
            raise HFTokenizerPruneError("BPE merge dependencies contain a cycle")
        token = token_by_id.get(token_id)
        merge = producer.get(token) if token is not None else None
        if merge is None:
            result = frozenset((token_id,))
        else:
            next_active = active | {token_id}
            result = frozenset((token_id,))
            result |= closure(vocabulary[merge.left], next_active)
            result |= closure(vocabulary[merge.right], next_active)
        closure_cache[token_id] = result
        return result

    result: set[int] = set()
    for token_id in sorted(token_ids):
        result.update(closure(token_id))
    return result


def _select_token_ids(
    tokenizer: Any,
    target_size: int,
    calibration_texts: Sequence[str],
    vocabulary: Mapping[str, int],
    producer: Mapping[str, _Merge],
    required_ids: set[int],
) -> set[int]:
    token_by_id = {token_id: token for token, token_id in vocabulary.items()}
    closure_cache: dict[int, frozenset[int]] = {}

    def closure(token_id: int, active: frozenset[int] = frozenset()) -> frozenset[int]:
        if token_id in closure_cache:
            return closure_cache[token_id]
        if token_id in active:
            raise HFTokenizerPruneError("BPE merge dependencies contain a cycle")
        token = token_by_id.get(token_id)
        merge = producer.get(token) if token is not None else None
        if merge is None:
            result = frozenset((token_id,))
        else:
            next_active = active | {token_id}
            result = frozenset((token_id,))
            result |= closure(vocabulary[merge.left], next_active)
            result |= closure(vocabulary[merge.right], next_active)
        closure_cache[token_id] = result
        return result

    selected: set[int] = set()
    for token_id in sorted(required_ids):
        selected.update(closure(token_id))
    if len(selected) > target_size:
        raise HFTokenizerPruneError(
            f"target_vocab_size {target_size} is too small; byte coverage, special "
            f"tokens, and merge dependencies require at least {len(selected)} rows"
        )

    frequencies: Counter[int] = Counter()
    for text in calibration_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        frequencies.update(int(token_id) for token_id in encoded)

    candidates = sorted(
        set(range(len(tokenizer))) - selected,
        key=lambda token_id: (-frequencies[token_id], token_id),
    )
    for token_id in candidates:
        additions = closure(token_id) - selected
        if additions and len(selected) + len(additions) <= target_size:
            selected.update(additions)
        if len(selected) == target_size:
            return selected
    raise HFTokenizerPruneError(
        f"target_vocab_size {target_size} is unreachable while preserving BPE "
        "merge dependency closure"
    )


def _pruned_tokenizer_data(
    tokenizer_data: Mapping[str, Any],
    vocabulary: Mapping[str, int],
    merges: Sequence[_Merge],
    selected_ids: set[int],
    id_map: Mapping[int, int],
) -> tuple[dict[str, Any], tuple[_Merge, ...]]:
    data = json.loads(json.dumps(tokenizer_data, ensure_ascii=False))
    model = data["model"]
    model["vocab"] = {
        token: id_map[token_id]
        for token, token_id in sorted(vocabulary.items(), key=lambda item: item[1])
        if token_id in selected_ids
    }
    retained = tuple(
        merge
        for merge in merges
        if vocabulary[merge.left] in selected_ids
        and vocabulary[merge.right] in selected_ids
        and vocabulary[merge.result] in selected_ids
    )
    model["merges"] = [merge.serialized for merge in retained]

    data["added_tokens"] = [
        {**item, "id": id_map[int(item["id"])]}
        for item in data.get("added_tokens", ())
        if int(item["id"]) in selected_ids
    ]
    _remap_post_processor(data.get("post_processor"), id_map)
    padding = data.get("padding")
    if isinstance(padding, dict) and "pad_id" in padding:
        padding["pad_id"] = id_map[int(padding["pad_id"])]
    return data, retained


def _remap_post_processor(component: Any, id_map: Mapping[int, int]) -> None:
    if isinstance(component, dict):
        special_tokens = component.get("special_tokens")
        if isinstance(special_tokens, dict):
            for value in special_tokens.values():
                if isinstance(value, dict) and isinstance(value.get("ids"), list):
                    value["ids"] = [id_map[int(item)] for item in value["ids"]]
        for value in component.values():
            _remap_post_processor(value, id_map)
    elif isinstance(component, list):
        for value in component:
            _remap_post_processor(value, id_map)


def _build_hf_tokenizer(donor: Any, tokenizer_data: Mapping[str, Any]) -> Any:
    try:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise HFTokenizerPruneError("tokenizers and transformers are required") from exc

    backend = Tokenizer.from_str(
        json.dumps(tokenizer_data, ensure_ascii=False, sort_keys=True)
    )
    kwargs = dict(donor.special_tokens_map)
    kwargs.update(
        {
            "model_max_length": donor.model_max_length,
            "padding_side": donor.padding_side,
            "truncation_side": donor.truncation_side,
            "clean_up_tokenization_spaces": getattr(
                donor, "clean_up_tokenization_spaces", False
            ),
        }
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **kwargs)
    chat_template = getattr(donor, "chat_template", None)
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    return tokenizer


def _reload_hf_tokenizer(output_dir: Path) -> Any:
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise HFTokenizerPruneError("transformers is required") from exc
    return PreTrainedTokenizerFast.from_pretrained(
        output_dir,
        local_files_only=True,
    )


def _verify_target(
    donor: Any,
    target: Any,
    target_size: int,
    calibration_texts: Sequence[str],
    selected_donor_ids: Sequence[int],
) -> None:
    if len(target) != target_size:
        raise HFTokenizerPruneError(
            f"built tokenizer has {len(target)} rows instead of {target_size}"
        )
    actual_ids = set(target.get_vocab().values())
    if actual_ids != set(range(target_size)):
        raise HFTokenizerPruneError("built tokenizer IDs are not contiguous")

    for target_id, donor_id in enumerate(selected_donor_ids):
        target_token = target.convert_ids_to_tokens(target_id)
        donor_token = donor.convert_ids_to_tokens(donor_id)
        if target_token != donor_token:
            raise HFTokenizerPruneError(
                f"target row {target_id} does not preserve donor token surface"
            )
        target_text = target.decode(
            [target_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        donor_text = donor.decode(
            [donor_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if target_text != donor_text:
            raise HFTokenizerPruneError(
                f"target row {target_id} does not preserve donor decoding"
            )

    probes = (
        "Byte-level BPE roundtrip: ASCII 0123.\nTabs\tand UTF-8: cafe\u0301 \u4e16\u754c",
        *calibration_texts,
    )
    for text in probes:
        donor_token_ids = list(donor.encode(text, add_special_tokens=False))
        donor_decoded = donor.decode(
            donor_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token_ids = list(target.encode(text, add_special_tokens=False))
        if any(token_id < 0 or token_id >= target_size for token_id in token_ids):
            raise HFTokenizerPruneError(
                "built tokenizer emitted an ID outside target row capacity"
            )
        decoded = target.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != donor_decoded:
            raise HFTokenizerPruneError(
                "built tokenizer does not preserve the donor encode/decode "
                f"roundtrip for {text!r}"
            )
