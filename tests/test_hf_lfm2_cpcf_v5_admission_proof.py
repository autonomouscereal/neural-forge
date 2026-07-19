from __future__ import annotations

import copy

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    Lfm2CpcfResumeArtifactStore,
)
from neural_forge.hf_lfm2_cpcf_v5_admission_proof import (
    Lfm2CpcfV5AdmissionProofError,
    authenticate_lfm2_cpcf_v5_admission_chain_proof,
    build_lfm2_cpcf_v5_admission_chain_proof,
)
from neural_forge.hf_lfm2_cpcf_v5_resume import (
    Lfm2CpcfV5ResumeCoordinator,
)
from tests.test_hf_lfm2_cpcf_resume_bridge import (
    _v5_progress,
    _v5_rss_expectation,
    _v5_rss_receipt,
)


def _proof(tmp_path):
    progress = _v5_progress(1)
    store = Lfm2CpcfResumeArtifactStore(tmp_path / "v5-resume")
    coordinator = Lfm2CpcfV5ResumeCoordinator(store)
    expectation = _v5_rss_expectation(1)
    pending = coordinator.publish_pending(
        progress["event"],
        rss_checkpoint_expectation=expectation,
    )
    admission = coordinator.admit(
        pending,
        rss_checkpoint_receipt=_v5_rss_receipt(
            expectation,
            passed=True,
        ),
    )
    proof = build_lfm2_cpcf_v5_admission_chain_proof(
        coordinator,
        admission["admission_sha256"],
    )
    return proof, pending, admission


def test_admission_chain_proof_reconstructs_exact_store_documents(tmp_path) -> None:
    proof, pending, admission = _proof(tmp_path)

    authenticated = authenticate_lfm2_cpcf_v5_admission_chain_proof(proof)

    assert authenticated["admission_document"]["admission_sha256"] == (
        admission["admission_sha256"]
    )
    assert authenticated["pending_document"]["pending_sha256"] == (
        pending.pending_sha256
    )
    assert authenticated["pending_document"]["frontier_reference"] == (
        pending.frontier_reference.as_dict()
    )
    assert authenticated["pending_document"]["checkpoint_reference"] == (
        pending.checkpoint.reference.as_dict()
    )
    assert authenticated["parent_or_cache_tensors_embedded"] == 0


def test_admission_chain_proof_rejects_consistently_resealed_blank_reference(
    tmp_path,
) -> None:
    proof, _pending, _admission = _proof(tmp_path)
    forged = copy.deepcopy(dict(proof))
    forged_admission = forged["admission_document"]
    forged_admission["frontier_reference"] = {}
    forged_admission.pop("admission_sha256")
    forged_admission["admission_sha256"] = canonical_sha256(forged_admission)
    forged_terminal = forged["terminal_document"]
    forged_terminal["outcome_sha256"] = forged_admission["admission_sha256"]
    forged_terminal["outcome_filename"] = (
        f"{forged_admission['admission_sha256']}.json"
    )
    forged_terminal.pop("terminal_sha256")
    forged_terminal["terminal_sha256"] = canonical_sha256(forged_terminal)
    forged.pop("proof_sha256")
    forged["proof_sha256"] = canonical_sha256(forged)

    with pytest.raises(
        Lfm2CpcfV5AdmissionProofError,
        match="lineage differs",
    ):
        authenticate_lfm2_cpcf_v5_admission_chain_proof(forged)
