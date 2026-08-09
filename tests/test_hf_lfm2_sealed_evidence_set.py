from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import neural_forge.hf_lfm2_sealed_evidence_set as MODULE  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
    build_lfm2_circuit_trace_cache,
)


def _capture_config() -> Lfm2CircuitTraceCaptureConfig:
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=4,
        max_decode_steps=8,
        max_capture_bytes=64 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )


def _batch(prompt, decode):
    return {
        "input_ids": torch.tensor([prompt], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
        "forced_decode_ids": torch.tensor([decode], dtype=torch.long),
    }


def _tiny_parent(*, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    """One sealed base corpus plus two disjoint sealed extension corpora."""

    root = tmp_path_factory.mktemp("lfm2-sealed-evidence-set")
    parent = _tiny_parent(seed=909)
    capture_config = _capture_config()
    base = build_lfm2_circuit_trace_cache(
        root / "base",
        parent,
        (
            _batch([2, 4, 5], [6, 7]),
            _batch([2, 5, 6], [7, 8]),
            _batch([2, 6, 7], [8, 4]),
        ),
        config=capture_config,
        input_device="cpu",
    )
    first = build_lfm2_circuit_trace_cache(
        root / "extension-0",
        parent,
        (_batch([4, 5, 6], [7, 8]), _batch([5, 6, 7], [8, 4])),
        config=capture_config,
        input_device="cpu",
    )
    second = build_lfm2_circuit_trace_cache(
        root / "extension-1",
        parent,
        (_batch([6, 7, 8], [4, 5]),),
        config=capture_config,
        input_device="cpu",
    )
    other_parent_cache = build_lfm2_circuit_trace_cache(
        root / "foreign",
        _tiny_parent(seed=404),
        (_batch([7, 8, 4], [5, 6]),),
        config=capture_config,
        input_device="cpu",
    )
    # A distinct cache whose only batch is already sealed in the base corpus.
    replayed = build_lfm2_circuit_trace_cache(
        root / "replayed",
        parent,
        (_batch([2, 4, 5], [6, 7]),),
        config=capture_config,
        input_device="cpu",
    )
    return SimpleNamespace(
        root=root,
        parent=parent,
        base=base,
        first=first,
        second=second,
        foreign=other_parent_cache,
        replayed=replayed,
    )


def _request(cache, *, splits=None, extension=False, manifest=None):
    ordered = splits or (
        tuple("fit" for _ in range(cache.batch_count))
        if extension
        else ("selection", *("fit" for _ in range(cache.batch_count - 1)))
    )
    return MODULE.Lfm2SealedEvidenceRequest(
        cache=cache,
        splits=ordered,
        split_sha256=canonical_sha256(
            {"cache_key": cache.cache_key, "splits": list(ordered)}
        ),
        round_manifest_sha256=(None if not extension else (manifest or "a" * 64)),
        all_rows_are_fit=extension,
    )


def _open(lane, *extensions, **kwargs):
    return MODULE.Lfm2SealedEvidenceSet.open(
        _request(lane.base),
        tuple(extensions),
        **kwargs,
    )


def test_base_only_evidence_set_is_a_byte_identical_pass_through(lane) -> None:
    """With no extension the aggregate is the base cache's own identity."""

    evidence = _open(lane, open_sessions=False)

    assert evidence.has_extension_evidence is False
    assert evidence.member_count == 1
    assert evidence.cache_key == lane.base.cache_key
    assert evidence.corpus_fingerprint == lane.base.corpus_fingerprint
    assert evidence.batch_count == lane.base.batch_count
    assert evidence.evidence_identity_sha256 is None
    assert evidence.round_count == 0
    assert evidence.extension_cache_keys == ()
    assert evidence.fit_caches() == (lane.base,)
    assert evidence.sessions == (None,)
    # Nothing was opened, so nothing can leak.
    evidence.close()
    assert evidence.closed is True


def test_extension_members_carry_provenance_for_every_batch(lane) -> None:
    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
        _request(lane.second, extension=True, manifest="c" * 64),
    )
    try:
        batches = tuple(evidence.iter_batches())

        assert evidence.batch_count == 6
        assert len(batches) == 6
        assert [value.merged_index for value in batches] == list(range(6))
        assert [value.member_ordinal for value in batches] == [0, 0, 0, 1, 1, 2]
        assert [value.member_role for value in batches] == [
            MODULE.BASE_EVIDENCE_ROLE,
            MODULE.BASE_EVIDENCE_ROLE,
            MODULE.BASE_EVIDENCE_ROLE,
            MODULE.EXTENSION_EVIDENCE_ROLE,
            MODULE.EXTENSION_EVIDENCE_ROLE,
            MODULE.EXTENSION_EVIDENCE_ROLE,
        ]
        assert [value.member_batch_index for value in batches] == [
            0,
            1,
            2,
            0,
            1,
            0,
        ]
        # Every extension row is a fit row; none is selection or validation.
        assert evidence.splits == (
            "selection",
            "fit",
            "fit",
            "fit",
            "fit",
            "fit",
        )
        assert all(
            value.is_fit
            for value in batches
            if value.member_role == MODULE.EXTENSION_EVIDENCE_ROLE
        )
        # Provenance names the exact sealed batch content hash.
        sealed = [
            str(value)
            for cache in (lane.base, lane.first, lane.second)
            for value in cache.manifest["corpus"]["ordered_batch_input_sha256"]
        ]
        assert [value.input_sha256 for value in batches] == sealed
        assert evidence.member_for_batch(4).cache is lane.first
        assert evidence.fit_caches() == (lane.base, lane.first, lane.second)
        assert evidence.round_count == 2
        assert evidence.round_manifest_sha256 == ("b" * 64, "c" * 64)
        assert evidence.extension_batch_counts == (2, 1)
    finally:
        evidence.close()


def test_merged_corpus_and_cache_key_change_with_extension_evidence(
    lane,
) -> None:
    evidence = _open(lane, _request(lane.first, extension=True))
    try:
        assert evidence.cache_key != lane.base.cache_key
        assert evidence.corpus_fingerprint != lane.base.corpus_fingerprint
        assert evidence.corpus_fingerprint != lane.first.corpus_fingerprint
        # The merged corpus body is derived from the members' own sealed corpus
        # documents, so it reproduces the exact recipe the cache module seals.
        base_corpus = dict(lane.base.manifest["corpus"])
        base_corpus.pop("corpus_sha256")
        first_corpus = dict(lane.first.manifest["corpus"])
        first_corpus.pop("corpus_sha256")
        merged = {
            "schema": base_corpus["schema"],
            "batch_count": (base_corpus["batch_count"] + first_corpus["batch_count"]),
            **{
                name: [*base_corpus[name], *first_corpus[name]]
                for name in (
                    "forced_decode_token_counts",
                    "ordered_batch_input_sha256",
                    "prompt_token_counts",
                )
            },
        }
        assert evidence.corpus_fingerprint == canonical_sha256(merged)
    finally:
        evidence.close()


def test_evidence_identity_is_the_one_digest_the_runner_folds_into_keys(
    lane,
) -> None:
    """The combined digest has exactly one definition, shared with the runner."""

    runner = pytest.importorskip("experiments.run_lfm2_dynamic_catalog")
    # The runner keeps the schema name on its published surface, bound to this
    # module's single definition.
    assert runner.ON_POLICY_EVIDENCE_IDENTITY_SCHEMA == (
        MODULE.ON_POLICY_EVIDENCE_IDENTITY_SCHEMA
    )
    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
        _request(lane.second, extension=True, manifest="c" * 64),
    )
    try:
        resolved = runner._ResolvedOnPolicyEvidence(
            extension_cache_paths=(lane.first.path, lane.second.path),
            extension_cache_keys=(
                lane.first.cache_key,
                lane.second.cache_key,
            ),
            extension_batch_counts=(
                lane.first.batch_count,
                lane.second.batch_count,
            ),
            round_manifest_sha256=("b" * 64, "c" * 64),
            fit_prompts_disjoint=True,
        )

        assert evidence.evidence_identity_sha256 == resolved.identity_sha256(
            lane.base.cache_key
        )
        assert evidence.evidence_identity_sha256 == (
            MODULE.lfm2_on_policy_evidence_identity_sha256(
                base_cache_key=lane.base.cache_key,
                extension_cache_keys=evidence.extension_cache_keys,
                extension_batch_counts=evidence.extension_batch_counts,
                round_manifest_sha256=evidence.round_manifest_sha256,
            )
        )
        # It is exactly what the sealed split digest binds, so a base-only
        # solve can never be reused for this request.
        _splits, extended = runner._derive_sealed_cache_splits(
            lane.base,
            on_policy_evidence_identity_sha256=(evidence.evidence_identity_sha256),
        )
        _base_splits, base_digest = runner._derive_sealed_cache_splits(lane.base)
        assert extended != base_digest
    finally:
        evidence.close()


def test_an_extension_row_can_never_be_selection_or_validation(lane) -> None:
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="must be a fit row",
    ):
        _request(
            lane.first,
            splits=("selection", "fit"),
            extension=True,
        )
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="all-fit split assignment",
    ):
        MODULE.Lfm2SealedEvidenceSet.open(
            _request(lane.base),
            (
                MODULE.Lfm2SealedEvidenceRequest(
                    cache=lane.first,
                    splits=("fit", "fit"),
                    split_sha256="d" * 64,
                    round_manifest_sha256="b" * 64,
                    all_rows_are_fit=False,
                ),
            ),
        )


def test_members_must_share_one_parent_and_stay_distinct(lane) -> None:
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="extend the same parent",
    ):
        _open(lane, _request(lane.foreign, extension=True))
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="distinct paths",
    ):
        _open(
            lane,
            _request(lane.first, extension=True),
            _request(lane.first, extension=True),
        )
    # Regenerating evidence the base corpus already seals is a closed path.
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="repeat a batch input",
    ):
        _open(lane, _request(lane.replayed, extension=True))


def test_a_failure_mid_open_closes_every_session_it_already_opened(
    lane,
    monkeypatch,
) -> None:
    """A half-open evidence set never survives and never leaks a descriptor."""

    opened: list[object] = []
    real = Lfm2CircuitTraceCache.selective_authenticated_session

    def counting_session(self):
        if len(opened) >= 2:
            raise OSError("selective session refused")
        session = real(self)
        opened.append(session)
        return session

    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        counting_session,
    )
    with pytest.raises(OSError, match="selective session refused"):
        _open(
            lane,
            _request(lane.first, extension=True, manifest="b" * 64),
            _request(lane.second, extension=True, manifest="c" * 64),
        )

    assert len(opened) == 2
    assert all(session.closed for session in opened)


def test_close_is_idempotent_and_never_closes_the_callers_own_session(
    lane,
) -> None:
    base_session = lane.base.selective_authenticated_session()
    try:
        evidence = _open(
            lane,
            _request(lane.first, extension=True),
            base_session=base_session,
        )
        extension_session = evidence.sessions[1]

        assert evidence.sessions[0] is base_session
        assert evidence.members[0].session_owned is False
        assert evidence.members[1].session_owned is True
        evidence.verify()
        evidence.close()
        evidence.close()

        assert evidence.closed is True
        assert extension_session.closed is True
        # The caller's session keeps its own single close.
        assert base_session.closed is False
        with pytest.raises(
            MODULE.Lfm2SealedEvidenceSetError,
            match="is closed",
        ):
            evidence.fit_caches()
    finally:
        base_session.close()


def test_receipt_fields_report_the_consumed_evidence(lane) -> None:
    evidence = _open(lane, _request(lane.first, extension=True, manifest="b" * 64))
    try:
        report = evidence.receipt_fields()

        assert report["member_count"] == 2
        assert report["batch_count"] == 5
        assert report["base_batch_count"] == 3
        assert report["extension_cache_keys"] == [lane.first.cache_key]
        assert report["extension_batch_counts"] == [2]
        assert report["extension_fit_batch_count"] == 2
        assert report["extension_selection_or_validation_batch_count"] == 0
        assert report["fit_batch_count"] == 4
        assert report["parent_forward_calls"] == 0
        assert report["round_manifest_sha256"] == ["b" * 64]
        document = evidence.as_dict()
        assert document["schema"] == MODULE.SEALED_EVIDENCE_SET_SCHEMA
        assert document["parent_forward_calls"] == 0
        assert document["extension_rows_are_never_review_rows"] is True
        assert json.dumps(document, sort_keys=True)
    finally:
        evidence.close()


def test_split_assignment_must_cover_its_cache(lane) -> None:
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="does not cover its cache",
    ):
        MODULE.Lfm2SealedEvidenceRequest(
            cache=lane.base,
            splits=("fit",),
            split_sha256="f" * 64,
        )
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="Unsupported sealed evidence split labels",
    ):
        MODULE.Lfm2SealedEvidenceRequest(
            cache=lane.second,
            splits=("review",),
            split_sha256="f" * 64,
        )
    with pytest.raises(
        MODULE.Lfm2SealedEvidenceSetError,
        match="carries no round manifest",
    ):
        MODULE.Lfm2SealedEvidenceSet.open(
            _request(lane.first, extension=True),
            (),
        )


def test_deferred_base_session_still_opens_every_extension(lane) -> None:
    evidence = MODULE.Lfm2SealedEvidenceSet.open(
        _request(lane.base),
        (_request(lane.first, extension=True),),
        open_base_session=False,
    )
    try:
        assert evidence.sessions[0] is None
        assert evidence.sessions[1] is not None
        assert Path(evidence.extension_caches[0].path) == lane.first.path
    finally:
        evidence.close()
        assert evidence.sessions[1].closed is True


def test_merged_views_present_one_cache_and_one_session(lane) -> None:
    """A one-handle writer can bind base+extension without a new session."""

    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
        _request(lane.second, extension=True, manifest="c" * 64),
    )
    try:
        cache = evidence.merged_cache()
        session = evidence.merged_session()

        # Memoized: the writer's handle and the evidence provider's seal are
        # the same object, so an ``is`` comparison downstream still holds.
        assert cache is evidence.merged_cache()
        assert session is evidence.merged_session()
        assert cache.selective_authenticated_session() is session

        assert cache.cache_key == session.cache_key == evidence.cache_key
        assert cache.corpus_fingerprint == evidence.corpus_fingerprint
        assert cache.parent_fingerprint == lane.base.parent_fingerprint
        assert cache.implementation_fingerprint == (
            lane.base.implementation_fingerprint
        )
        assert cache.layer_topology_fingerprint == (
            lane.base.layer_topology_fingerprint
        )
        assert len(cache) == len(session) == cache.batch_count == 6
        assert cache.manifest == session.manifest
        # The merged manifest restates the base manifest over the merged
        # corpus and never claims a sealed file digest it does not have.
        assert cache.manifest["corpus"]["batch_count"] == 6
        assert cache.manifest["corpus"]["corpus_sha256"] == (
            evidence.corpus_fingerprint
        )
        assert cache.manifest["capture"] == lane.base.manifest["capture"]
        assert cache.manifest["parent"] == lane.base.manifest["parent"]
        assert "manifest_sha256" not in cache.manifest
        assert (
            cache.manifest["merged_sealed_evidence_set"][
                "on_policy_evidence_identity_sha256"
            ]
            == evidence.evidence_identity_sha256
        )

        # Extension batches are named once, in merged order, as fit rows.
        assert cache.extension_batch_indices == (3, 4, 5)
        assert session.extension_batch_indices == (3, 4, 5)
        assert all(
            evidence.batch(index).is_fit for index in cache.extension_batch_indices
        )

        # Batch references are renumbered into merged order but keep the exact
        # sealed content hashes their member authenticated.
        references = session.batch_references
        assert [value.index for value in references] == list(range(6))
        sealed = [
            str(value)
            for member in (lane.base, lane.first, lane.second)
            for value in member.manifest["corpus"]["ordered_batch_input_sha256"]
        ]
        assert [value.input_sha256 for value in references] == sealed
        assert [value.input_sha256 for value in cache.iter_batches()] == sealed
        assert [value.index for value in cache.iter_batches()] == list(range(6))
    finally:
        evidence.close()


def test_merged_session_selection_is_routed_to_its_sealed_member(lane) -> None:
    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
    )
    try:
        session = evidence.merged_session()
        # Merged batch 3 is extension batch 0; the payload has to come from the
        # extension cache, and it must equal what that member serves directly.
        names = tuple(session.batch_references[3].ordered_tensor_names[:2])
        payload = session.load_selected_tensors(3, names)
        try:
            assert payload.batch_index == 3
            assert payload.session_sha256 == session.evidence_sha256
            assert (
                payload.input_sha256
                == (lane.first.manifest["corpus"]["ordered_batch_input_sha256"][0])
            )
            reference = session.batch_references[3]
            inventory = reference.record["tensors"]
            assert payload.selection_sha256 == canonical_sha256(
                {
                    "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
                    "session_sha256": session.evidence_sha256,
                    "cache_key": session.cache_key,
                    "batch_index": 3,
                    "input_sha256": reference.input_sha256,
                    "record_sha256": reference.record_sha256,
                    "ordered_tensor_names": list(names),
                    "ordered_tensor_descriptors": [inventory[name] for name in names],
                    "logical_bytes": payload.logical_bytes,
                }
            )
            member_session = evidence.members[1].session
            direct = member_session.load_selected_tensors(0, names)
            try:
                assert direct.selection_sha256 != payload.selection_sha256
                for name in names:
                    assert torch.equal(
                        payload.tensors[name],
                        direct.tensors[name],
                    )
            finally:
                member_session.release_selected_tensors(direct)
        finally:
            session.release_selected_tensors(payload)
        assert payload.released is True
        # The merged view opened nothing, so closing it closes nothing.
        session.close()
        assert session.closed is False
        assert session.verify() is session
        assert evidence.merged_cache().verify() is evidence.merged_cache()
    finally:
        evidence.close()
    assert session.closed is True


def test_merged_selection_preserves_member_validation_and_emergency_release(
    lane,
    monkeypatch,
) -> None:
    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
    )
    try:
        session = evidence.merged_session()
        names = tuple(session.batch_references[3].ordered_tensor_names[:2])
        with pytest.raises(TypeError, match="sequence of strings"):
            session.load_selected_tensors(3, names[0])
        with pytest.raises(TypeError, match="sequence of strings"):
            session.load_selected_tensors(3, iter(names))

        member_session = evidence.members[1].session
        member_type = type(member_session)
        original_load = member_type.load_selected_tensors
        captured = {}

        def corrupt_load(owner, index, tensor_names):
            payload = original_load(owner, index, tensor_names)
            if owner is member_session:
                payload.session_sha256 = "0" * 64
                captured["payload"] = payload
            return payload

        monkeypatch.setattr(member_type, "load_selected_tensors", corrupt_load)
        with pytest.raises(
            MODULE.Lfm2SealedEvidenceSetError,
            match="inconsistent selected payload",
        ):
            session.load_selected_tensors(3, names)
        assert captured["payload"].released is True
    finally:
        evidence.close()


def test_merged_views_are_refused_without_extension_evidence(lane) -> None:
    """A base-only set is already one handle; a merged view would be a lie."""

    evidence = _open(lane, open_sessions=False)
    try:
        with pytest.raises(
            MODULE.Lfm2SealedEvidenceSetError,
            match="merged view exists only for base",
        ):
            evidence.merged_cache()
        with pytest.raises(
            MODULE.Lfm2SealedEvidenceSetError,
            match="already has its own session",
        ):
            evidence.merged_session()
        assert evidence.extension_batch_indices == ()
    finally:
        evidence.close()


def test_merged_session_row_reads_are_routed_to_their_sealed_member(lane) -> None:
    """Bounded row reads carry the member's authentication, merged numbering."""

    from neural_forge.hf_lfm2_circuit_trace_row_selection import (
        lfm2_circuit_trace_selected_rows_payload_sha256,
        load_lfm2_circuit_trace_selected_rows,
        release_lfm2_circuit_trace_selected_rows,
    )

    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
    )
    try:
        session = evidence.merged_session()
        reference = session.batch_references[3]
        name = next(
            value
            for value in reference.ordered_tensor_names
            if ".residual." in value and not value.startswith("cached_prefill.")
        )
        selection = {name: (0,)}
        payload = session.load_selected_tensor_rows(3, selection)
        try:
            assert payload.batch_index == 3
            assert payload.session_sha256 == session.evidence_sha256
            assert (
                payload.input_sha256
                == (lane.first.manifest["corpus"]["ordered_batch_input_sha256"][0])
            )
            assert dict(payload.row_indices) == {name: (0,)}
            assert payload.selection_sha256 == (
                lfm2_circuit_trace_selected_rows_payload_sha256(
                    reference=reference,
                    session_sha256=session.evidence_sha256,
                    cache_key=session.cache_key,
                    batch_index=3,
                    payload=payload,
                )
            )
            member_session = evidence.members[1].session
            direct = load_lfm2_circuit_trace_selected_rows(
                member_session,
                0,
                selection,
            )
            try:
                assert direct.selection_sha256 != payload.selection_sha256
                assert torch.equal(payload.tensors[name], direct.tensors[name])
            finally:
                release_lfm2_circuit_trace_selected_rows(member_session, direct)
        finally:
            session.release_selected_tensor_rows(payload)
        assert payload.released is True
        # A payload the merged view did not hand out is refused, and a released
        # one is never re-released through another member.
        foreign = load_lfm2_circuit_trace_selected_rows(
            evidence.members[1].session,
            0,
            selection,
        )
        try:
            with pytest.raises(
                MODULE.Lfm2SealedEvidenceSetError,
                match="belongs to another authenticated session",
            ):
                session.release_selected_tensor_rows(foreign)
            with pytest.raises(TypeError):
                session.release_selected_tensor_rows(object())
        finally:
            release_lfm2_circuit_trace_selected_rows(
                evidence.members[1].session,
                foreign,
            )
    finally:
        evidence.close()


def test_merged_selection_rejects_noninteger_batch_identity_and_releases_rows(
    lane,
    monkeypatch,
) -> None:
    evidence = _open(
        lane,
        _request(lane.first, extension=True, manifest="b" * 64),
    )
    try:
        session = evidence.merged_session()
        reference = session.batch_references[3]
        name = next(
            value
            for value in reference.ordered_tensor_names
            if ".residual." in value and not value.startswith("cached_prefill.")
        )
        selection = {name: (0,)}

        whole_names = tuple(reference.ordered_tensor_names[:2])
        whole = session.load_selected_tensors(3, whole_names)
        try:
            whole.batch_index = "3"
            with pytest.raises(
                MODULE.Lfm2SealedEvidenceSetError,
                match="batch identity changed",
            ):
                session.release_selected_tensors(whole)
        finally:
            whole.batch_index = 3
            session.release_selected_tensors(whole)

        member_session = evidence.members[1].session
        original_load = MODULE.load_lfm2_circuit_trace_selected_rows
        captured = {}

        def corrupt_row_load(owner, index, row_indices):
            payload = original_load(owner, index, row_indices)
            if owner is member_session:
                payload.session_sha256 = "0" * 64
                captured["payload"] = payload
            return payload

        monkeypatch.setattr(
            MODULE,
            "load_lfm2_circuit_trace_selected_rows",
            corrupt_row_load,
        )
        with pytest.raises(
            MODULE.Lfm2SealedEvidenceSetError,
            match="inconsistent selected rows",
        ):
            session.load_selected_tensor_rows(3, selection)
        assert captured["payload"].released is True

        monkeypatch.setattr(
            MODULE,
            "load_lfm2_circuit_trace_selected_rows",
            original_load,
        )
        rows = session.load_selected_tensor_rows(3, selection)
        try:
            rows.batch_index = 3.0
            with pytest.raises(
                MODULE.Lfm2SealedEvidenceSetError,
                match="batch identity changed",
            ):
                session.release_selected_tensor_rows(rows)
        finally:
            rows.batch_index = 3
            session.release_selected_tensor_rows(rows)
    finally:
        evidence.close()
