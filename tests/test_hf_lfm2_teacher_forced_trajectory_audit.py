from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_teacher_forced_trajectory_audit import (
    LANE_CONTRACT,
    SAME_RUN_HIDDEN_CONTRAST_CONTRACT,
    Lfm2TeacherForcedTrajectoryAuditError,
    aggregate_teacher_forced_trajectory_audit,
    assert_text_span_roundtrip_only,
    build_teacher_forced_trajectory_audit_request,
    parent_winner_rank_margin,
    resolve_runtime_dtype_policy,
    run_teacher_forced_trajectory_batch,
    seal_parent_trajectory_batch,
)


_D = "a" * 64


class _ToyTokenizer:
    def __init__(self, rows: tuple[str, ...]) -> None:
        self.rows = rows
        self.inverse = {value: index for index, value in enumerate(rows)}

    def decode(self, ids, **_kwargs):
        return "".join(self.rows[int(index)] for index in ids)

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.inverse[character] for character in text]


class _ToyRoutingSession:
    pass


class _ToyRouting:
    has_separate_output_semantic_code = True

    def __init__(self, live_table: torch.Tensor) -> None:
        self.live_table = live_table
        self.c_in = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.c_out = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.75, -0.25, 0.5],
            ],
            dtype=torch.float32,
        )

    def as_dict(self):
        return {
            "runtime_sha256": "b" * 64,
            "source_artifact_sha256": "c" * 64,
            "input_output_coefficient_storage_shared": False,
            "donor_row_count": 4,
            "target_row_count": 3,
            "hidden_size": 2,
        }

    def reconstruct_selected_donor_embeddings(
        self, _session, donor_ids, *, maximum_rows_per_chunk
    ):
        assert maximum_rows_per_chunk > 0
        ids = torch.as_tensor(donor_ids, dtype=torch.int64)
        return self.c_in.index_select(0, ids) @ self.live_table

    def score_target_tokens(self, _session, hidden):
        return hidden @ self.live_table.transpose(0, 1)

    def compute_all_donor_logits_from_target_scores(
        self, _session, target_scores, *, maximum_donor_rows_per_chunk
    ):
        assert maximum_donor_rows_per_chunk > 0
        return target_scores @ self.c_out.transpose(0, 1)


class _ToyCausalBody:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        embeddings = kwargs["inputs_embeds"]
        return SimpleNamespace(last_hidden_state=embeddings.cumsum(dim=1))


def _batch(
    exact_table: torch.Tensor, *, round_cached_parent_hidden_to_bfloat16: bool = False
):
    prompt = torch.tensor([0, 3], dtype=torch.int64)
    forced = torch.tensor([2, 1], dtype=torch.int64)
    trajectory = exact_table.index_select(0, torch.cat((prompt, forced)))
    full_hidden = trajectory.cumsum(dim=0)
    states = full_hidden.index_select(0, torch.tensor([1, 2, 3]))
    if round_cached_parent_hidden_to_bfloat16:
        states = states.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    logits = states @ exact_table.transpose(0, 1)
    return seal_parent_trajectory_batch(
        batch_index=0,
        prompt_id="toy",
        trace_cache_key=_D,
        trace_parent_fingerprint=_D,
        trace_selective_session_sha256=_D,
        trace_input_sha256=_D,
        trace_record_sha256=_D,
        selected_payload_sha256=_D,
        prompt_token_ids=prompt,
        forced_token_ids=forced,
        parent_hidden_states=states,
        parent_logits=logits,
    )


def test_factorial_lanes_use_donor_clock_and_isolate_prompt_continuation() -> None:
    exact_table = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=torch.float32,
    )
    live_table = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
    )
    routing = _ToyRouting(live_table)
    body = _ToyCausalBody()
    progress = []
    roundtrip = assert_text_span_roundtrip_only(
        donor_tokenizer=_ToyTokenizer(("a", "b", "c", "d")),
        target_tokenizer=_ToyTokenizer(("a", "b", "c", "d")),
        prompt_token_ids=torch.tensor([0, 3]),
        forced_token_ids=torch.tensor([2, 1]),
    )
    result = run_teacher_forced_trajectory_batch(
        child_body=body,
        routing=routing,
        routing_session=_ToyRoutingSession(),
        live_target_table=live_table,
        exact_parent_table=exact_table,
        batch=_batch(exact_table),
        text_span_roundtrip_receipt=roundtrip,
        progress=progress.append,
    )

    assert tuple(result["lanes"]) == tuple(LANE_CONTRACT)
    assert len(body.calls) == 4
    assert all(call["use_cache"] is False for call in body.calls)
    assert all(tuple(call["inputs_embeds"].shape) == (1, 4, 2) for call in body.calls)
    assert result["body_step_domain"] == "authenticated-donor-token-trajectory-only"
    assert result["target_token_spans_used_as_body_steps"] is False

    lanes = result["lanes"]
    ee = lanes["exact_exact"]["tokens"]
    ef = lanes["exact_folded"]["tokens"]
    fe = lanes["folded_exact"]["tokens"]
    ff = lanes["folded_folded"]["tokens"]
    assert ee[0]["hidden_vs_cached_parent"]["relative_l2"] == pytest.approx(0.0)
    assert ef[0]["hidden_vs_cached_parent"] == ee[0]["hidden_vs_cached_parent"]
    assert ff[0]["hidden_vs_cached_parent"] == fe[0]["hidden_vs_cached_parent"]
    assert ef[1]["hidden_vs_cached_parent"]["relative_l2"] > 0.0
    assert all(
        row["hidden_vs_same_run_exact_exact"]["l2"] == pytest.approx(0.0)
        for row in ee
    )
    assert ef[0]["hidden_vs_same_run_exact_exact"]["l2"] == pytest.approx(0.0)
    assert ef[1]["hidden_vs_same_run_exact_exact"]["l2"] > 0.0
    assert fe[0]["hidden_vs_same_run_exact_exact"]["l2"] > 0.0
    assert ff[0]["hidden_vs_same_run_exact_exact"] == fe[0][
        "hidden_vs_same_run_exact_exact"
    ]
    assert (
        lanes["folded_folded"]["summary"][
            "mean_same_run_exact_exact_hidden_l2"
        ]
        > 0.0
    )
    token_events = [
        event for event in progress if event["event"] == "trajectory-token-complete"
    ]
    assert len(token_events) == 12
    assert all("hidden_vs_same_run_exact_exact" in event for event in token_events)
    assert lanes["exact_exact"]["summary"]["hidden_relative_l2_drift_slope"] == 0.0
    assert result["body_forward_count"] == 4

    request = build_teacher_forced_trajectory_audit_request(
        artifact_bindings={"trace": _D, "route": "b" * 64},
        runtime_dtype_policy=resolve_runtime_dtype_policy("float32", "cpu"),
        metric_dtype="float32",
        score_dtype="float32",
        donor_row_count=4,
        target_row_count=3,
        hidden_size=2,
    )
    audit = aggregate_teacher_forced_trajectory_audit(
        request=request,
        prompt_results=[result],
        trace_session_receipt={"session_sha256": _D},
    )
    aggregate = audit["lane_summaries"]["folded_folded"]
    assert aggregate[
        "state_weighted_mean_same_run_exact_exact_hidden_l2"
    ] == pytest.approx(
        lanes["folded_folded"]["summary"][
            "mean_same_run_exact_exact_hidden_l2"
        ]
    )


def test_same_run_reference_is_independent_of_cached_parent_dtype_rounding() -> None:
    exact_table = torch.tensor(
        [
            [1.0003, 0.0031],
            [0.0077, 1.0009],
            [1.0033, 0.9981],
            [2.0047, -1.0029],
        ],
        dtype=torch.float32,
    )
    live_table = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
    )
    result = run_teacher_forced_trajectory_batch(
        child_body=_ToyCausalBody(),
        routing=_ToyRouting(live_table),
        routing_session=_ToyRoutingSession(),
        live_target_table=live_table,
        exact_parent_table=exact_table,
        batch=_batch(exact_table, round_cached_parent_hidden_to_bfloat16=True),
        metric_dtype="float64",
    )

    exact_first = result["lanes"]["exact_exact"]["tokens"][0]
    folded_first = result["lanes"]["folded_exact"]["tokens"][0]
    assert exact_first["hidden_vs_cached_parent"]["l2"] > 0.0
    assert exact_first["hidden_vs_same_run_exact_exact"]["l2"] == pytest.approx(0.0)
    assert folded_first["hidden_vs_same_run_exact_exact"]["l2"] > 0.0


def test_parent_winner_rank_margin_is_signed_and_exact() -> None:
    top = parent_winner_rank_margin(torch.tensor([1.0, 4.0, 2.0]), 1)
    assert top["parent_winner_rank"] == 1
    assert top["parent_winner_margin"] == pytest.approx(2.0)
    missed = parent_winner_rank_margin(torch.tensor([5.0, 4.0, 2.0]), 1)
    assert missed["parent_winner_rank"] == 2
    assert missed["parent_winner_margin"] == pytest.approx(-1.0)
    assert missed["top1_matches_parent"] is False


def test_request_cache_binds_math_and_excludes_chunk_logging() -> None:
    policy = resolve_runtime_dtype_policy("float32", "cpu")
    base = dict(
        artifact_bindings={"trace": _D, "route": "b" * 64},
        runtime_dtype_policy=policy,
        metric_dtype="float64",
        score_dtype="float32",
        donor_row_count=4,
        target_row_count=3,
        hidden_size=2,
    )
    left = build_teacher_forced_trajectory_audit_request(
        **base, execution_contract={"rows": 1, "log": "a"}
    )
    right = build_teacher_forced_trajectory_audit_request(
        **base, execution_contract={"rows": 999, "log": "b"}
    )
    changed = build_teacher_forced_trajectory_audit_request(
        **{**base, "artifact_bindings": {"trace": _D, "route": "c" * 64}}
    )
    assert left["cache_key"] == right["cache_key"]
    assert left["cache_key"] != changed["cache_key"]
    assert left["same_run_hidden_contrast_contract"] == dict(
        SAME_RUN_HIDDEN_CONTRAST_CONTRACT
    )
    tampered_math = {
        key: value
        for key, value in left.items()
        if key
        not in {
            "cache_key",
            "execution_contract",
            "execution_contract_changes_cache_identity",
        }
    }
    tampered_math["same_run_hidden_contrast_contract"] = {
        **tampered_math["same_run_hidden_contrast_contract"],
        "reference_lane": "folded_exact",
    }
    assert canonical_sha256(tampered_math) != left["cache_key"]


def test_pre_ampere_bfloat16_policy_widens_explicitly() -> None:
    policy = resolve_runtime_dtype_policy(
        "bfloat16", "cuda:4", cuda_capability=(6, 1)
    )
    assert policy["requested_dtype"] == "bfloat16"
    assert policy["effective_dtype"] == "float32"
    assert "lacks-native-bfloat16" in policy["typed_fallback"]
    native = resolve_runtime_dtype_policy(
        "bfloat16", "cuda:0", cuda_capability=(8, 0)
    )
    assert native["effective_dtype"] == "bfloat16"
    assert native["typed_fallback"] is None
    cpu = resolve_runtime_dtype_policy("float16", "cpu")
    assert cpu["effective_dtype"] == "float32"
    assert "cpu-low-precision" in cpu["typed_fallback"]


def test_parent_batch_receipt_tampering_fails_closed() -> None:
    exact = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    batch = _batch(exact)
    receipt = dict(batch.receipt)
    receipt["hidden_size"] = 3
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(Lfm2TeacherForcedTrajectoryAuditError, match="binding"):
        type(batch)(
            batch.prompt_token_ids,
            batch.forced_token_ids,
            batch.parent_hidden_states,
            batch.parent_logits,
            receipt,
        )


def test_shared_route_is_rejected_before_any_body_forward() -> None:
    exact = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    live = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    route = _ToyRouting(live)
    route.has_separate_output_semantic_code = False
    body = _ToyCausalBody()
    with pytest.raises(Lfm2TeacherForcedTrajectoryAuditError, match="separate C_in/C_out"):
        run_teacher_forced_trajectory_batch(
            child_body=body,
            routing=route,
            routing_session=_ToyRoutingSession(),
            live_target_table=live,
            exact_parent_table=exact,
            batch=_batch(exact),
        )
    assert body.calls == []
