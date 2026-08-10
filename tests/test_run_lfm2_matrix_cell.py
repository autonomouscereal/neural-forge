"""Toy-scale contract tests for the matrix campaign cell runner.

The real runner opens an 85 GB sealed trace cache and a 230M-parameter parent,
neither of which belongs in a test.  What is testable, and what actually decides
whether a sealed receipt means anything, is everything around that:

* the behavioural measurements -- fixed-length free-run, token match, longest
  common prefix, loop-versus-cap termination -- because §5 rule 1 reads only
  those and a bug in them would silently move the campaign's only gate;
* the evidence view's row alignment, because a technique takes
  ``fit_rows(input)`` and ``fit_rows(write)`` as an (input, target) pair and a
  misalignment would look like a solver failure;
* the calibration comparison for cell 0b, which is what stops the campaign;
* the whole control run end to end against a stub model, which proves the
  receipt, the ledger and the renderer actually connect.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

torch = pytest.importorskip("torch")

_MODULE_PATH = _ROOT / "experiments" / "run_lfm2_matrix_cell.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_lfm2_matrix_cell", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load_module()

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    CONTROL_CELL_ID,
    read_ledger,
    verify_seal,
)

HIDDEN = 8
VOCAB = 32


# ------------------------------------------------------------ prompt sealing


def test_campaign_prompt_sets_are_sealed_and_disjoint_from_the_review_set():
    probe, probe_sha = runner._load_prompt_set(
        runner.PROBE_PROMPTS_FILE, "free-run-probe"
    )
    smoke, smoke_sha = runner._load_prompt_set(
        runner.SMOKE_PROMPTS_FILE, "decode-smoke"
    )
    assert len(probe) == 8 and len(smoke) == 3
    assert len(probe_sha) == 64 and len(smoke_sha) == 64
    assert not set(probe) & set(smoke)
    review_path = (
        _ROOT
        / "reports"
        / "89cd45b0-same-body-v4096-control"
        / "lfm2_semantic_whole_response_prompts_v1.json"
    )
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review_text = {
            message["content"]
            for entry in review["prompts"]
            for message in entry["messages"]
        }
        assert not review_text & set(probe)
        assert not review_text & set(smoke)


def test_prompt_set_seal_mismatch_is_fatal(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "kind": "free-run-probe",
                "prompts": ["a"],
                "prompt_set_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.MatrixCellRunError, match="seal mismatch"):
        runner._load_prompt_set(path, "free-run-probe")


# ------------------------------------------------------------------ evidence


def test_attention_endpoints_flatten_heads_into_the_projection_width():
    # The trace cache stores q/k/v/context as (batch, HEADS, tokens, head_dim) --
    # the axis order the attention op works in -- and the projection that
    # produced them wrote heads*head_dim contiguous channels.  So a cell scoring
    # attn_v against its own write has to see 8*4 as the width, one row per
    # token, not one row per head.
    value = torch.arange(2 * 8 * 3 * 4, dtype=torch.float32).reshape(2, 8, 3, 4)
    rows = runner._flatten_rows(value)
    assert tuple(rows.shape) == (6, 32)
    assert rows.dtype is torch.float64
    # row 0 is batch 0 / token 0, its 32 channels head-major.
    assert torch.equal(rows[0], value[0, :, 0, :].reshape(-1).to(torch.float64))


def test_attention_flatten_inverts_the_projection_that_wrote_the_endpoint():
    # The layout is not asserted from a convention, it is derived from the
    # operation: HF writes q_proj(x).view(B, T, H, D).transpose(1, 2), and the
    # calibration arm every attention cell must pass computes
    # rows(operator.input) @ q_proj.T and compares it *positionally* to
    # rows(attention.q_linear).  Those two have to agree exactly.
    torch.manual_seed(0)
    batch, tokens, heads, head_dim, hidden = 2, 5, 4, 3, 7
    inputs = torch.randn((batch, tokens, hidden), dtype=torch.float64)
    weight = torch.randn((heads * head_dim, hidden), dtype=torch.float64)
    projected = inputs @ weight.T
    sealed = (
        projected.view(batch, tokens, heads, head_dim).transpose(1, 2).contiguous()
    )
    assert tuple(sealed.shape) == (batch, heads, tokens, head_dim)
    # This is the calibration arm, verbatim: input rows times the parent weight,
    # against the sealed write endpoint's rows.
    reproduced = runner._flatten_rows(inputs) @ weight.T
    assert torch.allclose(runner._flatten_rows(sealed), reproduced, atol=1e-12)


def test_attention_phases_concatenate_at_the_real_capture_shapes():
    # The exact shapes the fresh matrix-campaign capture holds at an attention
    # site: one 256-token prefill and eight one-token forced decode steps.  The
    # previous fold made these (16, 16384) and (16, 64) and the per-endpoint
    # concatenation raised before any number was computed.
    prefill = torch.zeros((1, 16, 256, 64), dtype=torch.bfloat16)
    decode = torch.zeros((1, 16, 1, 64), dtype=torch.bfloat16)
    rows = [runner._flatten_rows(prefill)] + [
        runner._flatten_rows(decode) for _ in range(8)
    ]
    assert tuple(rows[0].shape) == (256, 1024)
    assert tuple(rows[1].shape) == (1, 1024)
    # 264 rows per batch is what the capture manifest declares (256 + 8).
    assert tuple(torch.cat(rows, dim=0).shape) == (264, 1024)


def test_grouped_query_endpoints_keep_their_own_narrower_width():
    # k_linear/v carry key_value_heads, not query_heads, so the flattened width
    # is 8*64 and a cell contracting attn_v's output axis contracts 512.
    keys = torch.zeros((1, 8, 256, 64), dtype=torch.bfloat16)
    assert tuple(runner._flatten_rows(keys).shape) == (256, 512)


def test_three_dimensional_endpoints_are_untouched_by_the_head_transpose():
    # Every conv/mlp/residual endpoint is 3-D, so the fix cannot have moved a
    # single number in any cell that does not read a per-head endpoint.
    value = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    rows = runner._flatten_rows(value)
    assert tuple(rows.shape) == (6, 5)
    assert torch.equal(rows, value.reshape(6, 5).to(torch.float64))


def test_site_evidence_keeps_endpoints_row_aligned():
    left = torch.randn((10, 4), dtype=torch.float64)
    right = left @ torch.randn((6, 4), dtype=torch.float64).T
    evidence = runner.TraceSiteEvidence(
        layer_index=5,
        phases=("prefill",),
        endpoints=("mlp.input", "mlp.gate"),
        fit={"mlp.input": left, "mlp.gate": right},
        heldout={"mlp.input": left[:3], "mlp.gate": right[:3]},
        selection_sha256="0" * 64,
    )
    assert evidence.fit_row_count == 10
    assert evidence.heldout_row_count == 3
    assert torch.equal(evidence.fit_rows("mlp.input"), left)
    with pytest.raises(runner.MatrixCellRunError, match="was not selected"):
        evidence.fit_rows("attention.v")


# ----------------------------------------------------------------- behaviour


def test_token_match_counts_every_position_including_length_disagreement():
    match, lcp = runner._token_match([[1, 2, 3, 4]], [[1, 2, 9, 4]])
    assert match == pytest.approx(3 / 4)
    assert lcp == pytest.approx(2 / 4)
    match, lcp = runner._token_match([[1, 2]], [[1, 2, 3, 4]])
    # A short run is not flattered: the denominator is the longer of the two.
    assert match == pytest.approx(2 / 4)
    assert lcp == pytest.approx(2 / 4)
    assert runner._token_match([[5, 5]], [[5, 5]]) == (1.0, 1.0)


def test_token_match_refuses_a_mismatched_control():
    with pytest.raises(runner.MatrixCellRunError):
        runner._token_match([[1]], [[1], [2]])


def test_loop_detection_separates_loop_from_cap():
    assert runner._detect_loop([7] * 40) is True
    assert runner._detect_loop([1, 2, 3] * 12) is True
    assert runner._detect_loop(list(range(40))) is False
    assert runner._detect_loop([1, 2]) is False


# --------------------------------------------------------- A9 calibration 0b


def _sealed_reference():
    return {
        "sparse_semantic_codebook_artifact_sha256": "4" * 64,
        "base_codebook_artifact_sha256": "5" * 64,
        "codebook_phase_key_sha256": "6" * 64,
        "codebook_phase_completion_sha256": "7" * 64,
        "candidate_support_math_abi_sha256": "8" * 64,
        "coefficient_solve_math_abi_sha256": "9" * 64,
        "cached_hidden_top1_match_count": 551,
        "cached_hidden_top5_match_count": 589,
        "cached_hidden_top10_match_count": 605,
        "cached_hidden_row_count": 648,
    }


def _route_result(**overrides):
    sealed = _sealed_reference()
    identities = {
        key: sealed[key] for key in sealed if key.endswith("sha256")
    }
    solve = {
        key: sealed[key] for key in sealed if key.startswith("cached_hidden")
    }
    identities.update(
        {k: v for k, v in overrides.items() if k.endswith("sha256")}
    )
    solve.update({k: v for k, v in overrides.items() if k.startswith("cached_hidden")})
    return runner.CellTechniqueResult(
        installed_weights={},
        stored_parameter_count=1,
        solve=solve,
        artifact_identities=identities,
        child_directory=Path("."),
    )


def test_a9_calibration_passes_on_bit_identity():
    report = runner._a9_calibration(_route_result(), _sealed_reference())
    assert report["passed"] is True
    assert all(entry["matched"] for entry in report["checks"])


def test_a9_calibration_fails_on_a_single_changed_digest():
    report = runner._a9_calibration(
        _route_result(sparse_semantic_codebook_artifact_sha256="0" * 64),
        _sealed_reference(),
    )
    assert report["passed"] is False
    failed = [e["field"] for e in report["checks"] if not e["matched"]]
    assert failed == ["sparse_semantic_codebook_artifact_sha256"]


def test_a9_calibration_fails_on_a_drifted_advisory_count():
    report = runner._a9_calibration(
        _route_result(cached_hidden_top1_match_count=550), _sealed_reference()
    )
    assert report["passed"] is False


def test_dossier_extraction_prefers_the_authoritative_target_text():
    # The sealed A9 dossier keeps a *diagnostic* donor decode next to the
    # authoritative retokenized target text; scoring the wrong one would compare
    # the calibration cell against text the schema says is not authoritative.
    body = {
        "comparisons": [
            {
                "id": "a",
                "authoritative_child_target_text": "right",
                "diagnostic_child_donor_text": "wrong",
                "child_response": "also wrong",
            }
        ]
    }
    assert runner._dossier_responses(body) == {"a": "right"}
    legacy = {
        "responses": [
            {"prompt_id": "a", "child_response": "one"},
            {"id": "b", "text": "two"},
        ]
    }
    assert runner._dossier_responses(legacy) == {"a": "one", "b": "two"}
    with pytest.raises(runner.MatrixCellRunError):
        runner._dossier_responses({"responses": [{"prompt_id": "a"}]})


def test_dossier_extraction_reads_the_real_sealed_a9_dossier():
    sealed = (
        _ROOT
        / "reports"
        / "sparse-semantic-a9-k32-residual-omp-20260730"
        / "whole-response-dossier.json"
    )
    if not sealed.is_file():
        pytest.skip("sealed A9 dossier is not present in this checkout")
    body = json.loads(sealed.read_text(encoding="utf-8"))
    responses = runner._dossier_responses(body)
    assert len(responses) == 16
    assert all(isinstance(text, str) for text in responses.values())


# ------------------------------------------------------ end-to-end (control)


class _StubTokenizerOutput:
    def __init__(self, ids):
        self.input_ids = ids


class _StubTokenizer:
    """Deterministic byte-fold tokenizer.  No chat template: raw prompts only."""

    eos_token_id = VOCAB - 1
    chat_template = None

    def __call__(self, text, return_tensors=None):
        ids = [((ord(ch) * 7 + 3) % (VOCAB - 2)) + 1 for ch in text[:24]] or [1]
        return _StubTokenizerOutput(torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


class _StubResult:
    def __init__(self, logits, past, hidden=None):
        self.logits = logits
        self.past_key_values = past
        self.hidden_states = hidden


class _StubModel:
    """A tiny stand-in for the LFM2 child: deterministic, cheap, and honest.

    It is not a language model; it is a fixed function of the last token id,
    which is exactly enough to prove the runner's plumbing -- greedy loop, cache
    threading, hidden-state collection, termination classification -- without
    pretending to test the parent.
    """

    def __init__(self, state):
        self._state = state
        self._layers = 3

    def state_dict(self):
        return self._state

    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        length = int(input_ids.shape[1])
        last = int(input_ids[0, -1].item())
        logits = torch.zeros((1, length, VOCAB), dtype=torch.float32)
        for position in range(length):
            token = int(input_ids[0, position].item())
            logits[0, position, (token * 5 + 1) % VOCAB] = 1.0
        hidden = None
        if output_hidden_states:
            base = torch.full((1, length, HIDDEN), float(last) / VOCAB)
            hidden = tuple(base * (index + 1) for index in range(self._layers + 1))
        return _StubResult(logits, (past_key_values or 0) + 1, hidden)


def _stub_state():
    return {
        "model.embed_tokens.weight": torch.randn(
            (VOCAB, HIDDEN), dtype=torch.float32
        ),
        "model.embedding_norm.weight": torch.ones(HIDDEN, dtype=torch.float32),
    }


def test_greedy_free_run_is_fixed_length_and_deterministic():
    model = _StubModel(_stub_state())
    tokenizer = _StubTokenizer()
    first = runner._greedy_fixed(
        model, tokenizer, ["alpha", "beta"], tokens=6, device="cpu"
    )
    second = runner._greedy_fixed(
        model, tokenizer, ["alpha", "beta"], tokens=6, device="cpu"
    )
    assert first == second
    assert [len(row) for row in first] == [6, 6]


def test_decode_smoke_records_termination_and_verbatim_text():
    model = _StubModel(_stub_state())
    tokenizer = _StubTokenizer()
    records = runner._decode_smoke(
        model, tokenizer, ["alpha"], cap=32, device="cpu", eos_id=None
    )
    assert len(records) == 1
    assert records[0]["terminated"] in runner.__dict__["_detect_loop"].__doc__ or True
    assert records[0]["terminated"] in ("cap", "loop", "eos", "error")
    assert isinstance(records[0]["output_verbatim"], str)
    assert len(records[0]["prompt_sha"]) == 64


def test_control_cell_runs_end_to_end_and_renders(tmp_path, monkeypatch):
    state = _stub_state()
    monkeypatch.setattr(runner, "_load_parent", lambda d, dev, dt: _StubModel(state))
    monkeypatch.setattr(runner, "_load_tokenizer", lambda d: _StubTokenizer())
    output = tmp_path / "CTRL"
    ledger = tmp_path / "ledger.jsonl"
    args = runner._parser().parse_args(
        [
            "--cell-id",
            CONTROL_CELL_ID,
            "--mode",
            "control",
            "--out",
            str(output),
            "--ledger",
            str(ledger),
            "--parent-model",
            str(tmp_path),
            "--device",
            "cpu",
            "--probe-tokens",
            "6",
            "--smoke-token-cap",
            "8",
        ]
    )
    assert runner.run(args) == 0

    receipt = json.loads((output / "cell-receipt.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["cell_id"] == CONTROL_CELL_ID
    assert receipt["cell_kind"] == "control"
    assert receipt["calibration"]["passed"] is True
    # The control child *is* the parent, so it agrees with itself perfectly and
    # its terminal drift is exactly zero.  Those two numbers are the campaign's
    # behavioural baseline; every later cell is scored against them.
    assert receipt["behavior"]["free_run"]["token_match_vs_control"] == 1.0
    assert receipt["behavior"]["free_run"]["mean_lcp"] == 1.0
    assert receipt["composed"]["vs_copy_exact_control_delta"] == 0.0
    assert receipt["composed"]["emit_zero_floor_passed"] is True
    assert receipt["outcome"]["status"] == "pending-review"
    assert receipt["outcome"]["first_divergent_metric"] is None
    assert len(receipt["behavior"]["decode_smoke"]) == 3

    control = json.loads((output / "control.json").read_text(encoding="utf-8"))
    assert len(control["probe_token_ids"]) == 8
    assert (output / "control-states.pt").is_file()

    rows = read_ledger(ledger)
    assert len(rows) == 1 and rows[0]["cell_id"] == CONTROL_CELL_ID

    render = importlib.util.spec_from_file_location(
        "render_matrix_ledger", _ROOT / "experiments" / "render_matrix_ledger.py"
    )
    module = importlib.util.module_from_spec(render)
    sys.modules[render.name] = module
    render.loader.exec_module(module)
    report = module.render(rows, ledger_path=ledger)
    assert CONTROL_CELL_ID in report
    assert "Calibration row" in report


def test_a_cell_without_a_control_artifact_seals_a_failure(tmp_path, monkeypatch):
    state = _stub_state()
    monkeypatch.setattr(runner, "_load_parent", lambda d, dev, dt: _StubModel(state))
    monkeypatch.setattr(runner, "_load_tokenizer", lambda d: _StubTokenizer())
    # A real (tiny) parent shard, so the refusal below is genuinely about the
    # missing calibration reference and not about a missing file.
    save_file = pytest.importorskip("safetensors.torch").save_file
    parent = tmp_path / "parent"
    parent.mkdir()
    save_file(
        {"model.embed_tokens.weight": torch.zeros((VOCAB, HIDDEN))},
        str(parent / "model.safetensors"),
    )
    output = tmp_path / "CAL"
    args = runner._parser().parse_args(
        [
            "--cell-id",
            "CAL.C01.T5.X4",
            "--out",
            str(output),
            "--parent-model",
            str(parent),
            "--device",
            "cpu",
        ]
    )
    # No sealed_reference knobs: the calibration cell has nothing to compare
    # against, so it must seal a void cell rather than invent a verdict -- and
    # it must refuse *before* spending a GPU-hour on the route.
    assert runner.run(args) == 1
    receipt = json.loads((output / "cell-receipt.json").read_text(encoding="utf-8"))
    verify_seal(receipt)
    assert receipt["outcome"]["status"] == "infra"
    assert receipt["outcome"]["first_divergent_metric"] == "infra"
    assert receipt["calibration"]["passed"] is False
    assert "sealed_reference" in receipt["outcome"]["notes"]
    assert not (output / "route").exists()
