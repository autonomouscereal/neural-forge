from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.tokenizer_control_contract import (
    TOKENIZER_CONTROL_CONTRACT_SCHEMA,
    TOKENIZER_CONTROL_MANIFEST_SCHEMA,
    TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
    TokenizerControlContractError,
    compile_local_tokenizer_control_contract,
    inventory_local_tokenizer_control_sources,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _artifact(tmp_path: Path, *, with_manifest: bool = True) -> Path:
    root = tmp_path / "local-lfm2-tokenizer"
    root.mkdir()
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "<|chat_start|>": 4,
        "<|reserved_special_token_0|>": 5,
        "<|vision_start|>": 6,
        "[unused0]": 7,
        "<|config_only|>": 8,
        "<|modality_slot|>": 9,
        "ordinary": 10,
        "<|orphan_control|>": 11,
    }
    added_tokens = [
        {"id": 0, "content": "<unk>", "special": True},
        {"id": 1, "content": "<s>", "special": True},
        {"id": 2, "content": "</s>", "special": True},
        {"id": 3, "content": "<pad>", "special": True},
        {"id": 4, "content": "<|chat_start|>", "special": False},
        {
            "id": 5,
            "content": "<|reserved_special_token_0|>",
            "special": True,
        },
        {"id": 6, "content": "<|vision_start|>", "special": False},
        {"id": 8, "content": "<|config_only|>", "special": False},
    ]
    _write_json(
        root / "tokenizer.json",
        {"model": {"type": "BPE", "vocab": vocab}, "added_tokens": added_tokens},
    )
    _write_json(
        root / "tokenizer_config.json",
        {
            "bos_token": "<s>",
            "bos_token_id": 1,
            "eos_token": {
                "__type": "AddedToken",
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "eos_token_id": 2,
            "pad_token": "<pad>",
            "pad_token_id": 3,
            "additional_special_tokens": ["<|reserved_special_token_0|>"],
            "added_tokens_decoder": {
                str(record["id"]): {
                    "content": record["content"],
                    "special": record["special"],
                }
                for record in added_tokens
            },
            "chat_template": "{{ '<|config_only|>' + messages[0]['content'] }}",
            # Deliberately wrong and irrelevant.  The compiler must never use it.
            "all_special_ids": [10],
        },
    )
    (root / "chat_template.jinja").write_text(
        "{{ '<|chat_start|>' + messages[0]['role'] + '\\n' }}",
        encoding="utf-8",
    )
    if with_manifest:
        _write_json(
            root / "modality-controls.json",
            {
                "schema": TOKENIZER_CONTROL_MANIFEST_SCHEMA,
                "manifest_kind": "modality",
                "active_control_tokens": [{"id": 6, "content": "<|vision_start|>"}],
                "reserved_placeholder_tokens": [
                    {"id": 9, "content": "<|modality_slot|>"}
                ],
            },
        )
    return root


def _hashes(root: Path, *, with_manifest: bool = True) -> dict[str, str]:
    manifests = ("modality-controls.json",) if with_manifest else ()
    return inventory_local_tokenizer_control_sources(
        root,
        manifest_files=manifests,
    )


def _compile(root: Path):
    return compile_local_tokenizer_control_contract(
        root,
        expected_file_sha256=_hashes(root),
        manifest_files=("modality-controls.json",),
    )


def test_fail_closed_contract_discovers_all_local_control_classes(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)

    contract = _compile(root)
    receipt = contract.as_dict()
    records = {record["token_id"]: record for record in receipt["token_records"]}

    assert contract.protected_id_values == tuple(range(10))
    assert contract.mandatory_active_id_values == (0, 1, 2, 3, 4, 6, 8)
    assert contract.reserved_placeholder_id_values == (5, 7, 9)
    assert contract.reclaimed_id_values == ()
    assert records[4]["classifications"] == ["chat-template-literal-control"]
    assert records[4]["mandatory_active_structural_control"] is True
    assert records[5]["reserved_placeholder"] is True
    assert records[5]["reclaim_eligible"] is True
    assert records[6]["classifications"] == ["manifest-active-control"]
    assert records[8]["classifications"] == ["chat-template-literal-control"]
    assert records[9]["classifications"] == ["manifest-reserved-placeholder"]
    assert 10 not in records  # all_special_ids is intentionally ignored.
    assert 11 not in records  # Control-shaped but unused and not special.
    assert receipt["tokenizer_all_special_ids_consulted"] is False
    assert receipt["network_access_performed"] is False
    assert receipt["hugging_face_loader_used"] is False


def test_protected_ids_are_fresh_cpu_int64_tensors(tmp_path: Path) -> None:
    contract = _compile(_artifact(tmp_path))

    first = contract.protected_ids
    second = contract.protected_token_ids
    assert first.dtype is torch.int64
    assert first.device == torch.device("cpu")
    assert torch.equal(first, torch.arange(10, dtype=torch.int64))
    first[0] = 999
    assert second[0].item() == 0
    assert contract.protected_id_values[0] == 0


def test_receipt_seals_sources_ids_classifications_and_default_policy(
    tmp_path: Path,
) -> None:
    contract = _compile(_artifact(tmp_path))
    receipt = contract.as_dict()

    assert receipt["schema"] == TOKENIZER_CONTROL_CONTRACT_SCHEMA
    assert receipt["source_inventory"]["files"] == [
        {
            "path": name,
            "sha256": _hashes(tmp_path / "local-lfm2-tokenizer")[name],
            "byte_length": (tmp_path / "local-lfm2-tokenizer" / name).stat().st_size,
        }
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "modality-controls.json",
        )
    ]
    assert receipt["reclaim_policy"]["mode"] == "protect-all"
    assert receipt["reclaim_policy"]["release_reserved_token_ids"] == []
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert contract.receipt_sha256 == receipt["receipt_sha256"]
    assert contract.receipt["protected_token_ids"] == list(range(10))


def test_authenticated_policy_reclaims_only_explicit_inactive_reserved_ids(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    source_hashes = _hashes(root)
    default = compile_local_tokenizer_control_contract(
        root,
        expected_file_sha256=source_hashes,
        manifest_files=("modality-controls.json",),
    )
    policy = {
        "schema": TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
        "source_inventory_sha256": default.as_dict()["source_inventory_sha256"],
        "release_reserved_token_ids": [9, 5],
        "reason": "Rows 5 and 9 are intentionally assigned as semantic carriers.",
    }
    policy_path = root / "reclaim-policy.json"
    _write_json(policy_path, policy)
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    contract = compile_local_tokenizer_control_contract(
        root,
        expected_file_sha256=source_hashes,
        manifest_files=("modality-controls.json",),
        reclaim_policy_file="reclaim-policy.json",
        expected_reclaim_policy_sha256=policy_sha,
    )

    assert contract.default_protected_id_values == tuple(range(10))
    assert contract.reclaimed_id_values == (5, 9)
    assert contract.protected_id_values == (0, 1, 2, 3, 4, 6, 7, 8)
    receipt = contract.as_dict()
    assert receipt["reclaim_policy"]["policy_file_sha256"] == policy_sha
    assert receipt["reclaim_policy"]["policy_body_sha256"] == canonical_sha256(policy)
    assert receipt["reclaim_policy"]["release_reserved_token_ids"] == [5, 9]
    records = {record["token_id"]: record for record in receipt["token_records"]}
    assert records[5]["reclaimed"] is True
    assert records[7]["protected"] is True
    assert records[9]["reclaimed"] is True


@pytest.mark.parametrize("released_id", (0, 4, 6, 10))
def test_policy_cannot_release_active_nonreserved_or_unknown_ids(
    tmp_path: Path,
    released_id: int,
) -> None:
    root = _artifact(tmp_path)
    source_hashes = _hashes(root)
    default = _compile(root)
    policy = {
        "schema": TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
        "source_inventory_sha256": default.as_dict()["source_inventory_sha256"],
        "release_reserved_token_ids": [released_id],
        "reason": "Negative contract test.",
    }
    policy_path = root / "reclaim-policy.json"
    _write_json(policy_path, policy)

    with pytest.raises(TokenizerControlContractError, match="active|not protected"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=source_hashes,
            manifest_files=("modality-controls.json",),
            reclaim_policy_file="reclaim-policy.json",
            expected_reclaim_policy_sha256=hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest(),
        )


def test_reserved_name_becomes_nonreclaimable_when_template_uses_it(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    (root / "chat_template.jinja").write_text(
        "{{ '<|chat_start|><|reserved_special_token_0|>' }}",
        encoding="utf-8",
    )
    source_hashes = _hashes(root)
    default = compile_local_tokenizer_control_contract(
        root,
        expected_file_sha256=source_hashes,
        manifest_files=("modality-controls.json",),
    )
    assert 5 in default.mandatory_active_id_values
    assert 5 in default.reserved_placeholder_id_values
    record = next(
        item for item in default.as_dict()["token_records"] if item["token_id"] == 5
    )
    assert record["reclaim_eligible"] is False

    policy_path = root / "reclaim-policy.json"
    _write_json(
        policy_path,
        {
            "schema": TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
            "source_inventory_sha256": default.as_dict()["source_inventory_sha256"],
            "release_reserved_token_ids": [5],
            "reason": "An active reserved-looking row must remain protected.",
        },
    )
    with pytest.raises(TokenizerControlContractError, match="active"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=source_hashes,
            manifest_files=("modality-controls.json",),
            reclaim_policy_file="reclaim-policy.json",
            expected_reclaim_policy_sha256=hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest(),
        )


def test_policy_must_be_authenticated_and_bound_to_exact_sources(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    source_hashes = _hashes(root)
    default = _compile(root)
    policy_path = root / "reclaim-policy.json"
    _write_json(
        policy_path,
        {
            "schema": TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA,
            "source_inventory_sha256": "0" * 64,
            "release_reserved_token_ids": [5],
            "reason": "Stale source binding.",
        },
    )
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    with pytest.raises(TokenizerControlContractError, match="not bound"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=source_hashes,
            manifest_files=("modality-controls.json",),
            reclaim_policy_file="reclaim-policy.json",
            expected_reclaim_policy_sha256=policy_sha,
        )
    assert default.protected_id_values == tuple(range(10))

    with pytest.raises(TokenizerControlContractError, match="hash mismatch"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=source_hashes,
            manifest_files=("modality-controls.json",),
            reclaim_policy_file="reclaim-policy.json",
            expected_reclaim_policy_sha256="f" * 64,
        )


def test_source_hash_mismatch_fails_before_classification(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    expected = _hashes(root)
    (root / "chat_template.jinja").write_text(
        "{{ '<|orphan_control|>' }}",
        encoding="utf-8",
    )

    with pytest.raises(TokenizerControlContractError, match="hash mismatch"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=expected,
            manifest_files=("modality-controls.json",),
        )


def test_every_consumed_manifest_requires_an_exact_expected_hash(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    incomplete = _hashes(root)
    incomplete.pop("modality-controls.json")

    with pytest.raises(TokenizerControlContractError, match="exactly every"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=incomplete,
            manifest_files=("modality-controls.json",),
        )


def test_manifest_references_and_declared_role_ids_fail_closed(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path)
    manifest_path = root / "modality-controls.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_control_tokens"] = [{"id": 6, "content": "ordinary"}]
    _write_json(manifest_path, manifest)

    with pytest.raises(TokenizerControlContractError, match="disagree"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=_hashes(root),
            manifest_files=("modality-controls.json",),
        )


def test_duplicate_authenticated_json_keys_are_rejected(tmp_path: Path) -> None:
    root = _artifact(tmp_path, with_manifest=False)
    (root / "tokenizer_config.json").write_text(
        '{"bos_token":"<s>","bos_token":"</s>"}',
        encoding="utf-8",
    )

    with pytest.raises(TokenizerControlContractError, match="duplicate key"):
        compile_local_tokenizer_control_contract(
            root,
            expected_file_sha256=_hashes(root, with_manifest=False),
        )
