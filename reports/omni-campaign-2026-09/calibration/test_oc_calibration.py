#!/usr/bin/env python3
"""Tests for the omni parent-calibration lane.

Two kinds of test, deliberately separated:

  * **Instrument tests** need nothing but the standard library, so they run on
    the Windows workstation (Python 3.14, no torch) AND on the server.  They
    cover the parts a wrong answer would be silent in: canonical bytes, the copy
    measure, the form boxes, the bank's self-test, the two columns, the payload
    anchors, and the refusals.
  * **Torch tests** are skipped when torch is absent.  They cover the emit-zero
    floor and the generation loop against a tiny stub model, so the arithmetic of
    the floor is checked without a GPU.

Run:  python -m pytest test_oc_calibration.py -q
      or:  python test_oc_calibration.py     (a bare runner, no pytest needed)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: A SKIP IS NOT A PASS AND THE RUNNER MUST NOT PRINT IT AS ONE.  The bare
#: runner used to print `ok` for a test that had returned early and then report
#: "45/45 passed", which hid that the repo-source byte-equality test -- the one
#: that justifies re-implementing the canonical primitives -- never runs on the
#: server.  `unittest.SkipTest` is what pytest already understands as a skip, so
#: both runners now agree.
SkipTest = unittest.SkipTest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import oc_assess  # noqa: E402
import oc_bank_v2_draft as bankv2  # noqa: E402
import oc_calib  # noqa: E402
import oc_parent_calibration as orchestrator  # noqa: E402
import oc_testcards  # noqa: E402

#: The repo copy when there is a checkout, the uploaded copy on the server.  Both
#: must digest to the same value or the tests below refuse them.
FROZEN_V1 = REPO / "research" / "lfm2_semantic_whole_response_prompts_v1.json"
if not FROZEN_V1.is_file():
    FROZEN_V1 = HERE / "banks" / "lfm2_semantic_whole_response_prompts_v1.json"
FROZEN_V1_SHA = "0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b"


# --------------------------------------------------------------------------
# Canonical bytes and digests.
# --------------------------------------------------------------------------


def test_canonical_bytes_are_ascii_sorted_and_compact():
    value = {"b": 1, "a": [2, {"z": None, "y": "é"}]}
    raw = oc_calib.canonical_json_bytes(value)
    assert raw == b'{"a":[2,{"y":"\\u00e9","z":null}],"b":1}'
    assert raw.decode("ascii")


def test_canonical_bytes_match_repo_when_importable():
    """If the neural-forge repo is here, our copy must produce IDENTICAL bytes.

    This is the whole justification for re-implementing the primitive: a copy
    that drifts would silently produce digests nothing else in the estate can
    reproduce.
    """

    source_path = REPO / "src" / "neural_forge" / "dynamic_circuit_ir.py"
    if not source_path.is_file():  # pragma: no cover - the server has no checkout
        raise SkipTest("no repo checkout here: dynamic_circuit_ir.py is absent")
    # The module imports torch at package level and the workstation has none, so
    # the two function DEFINITIONS are lifted out of the source and executed in
    # an isolated namespace.  That checks the thing that matters -- the bytes --
    # without needing the dependency the bytes have nothing to do with.
    import ast

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"canonical_json_bytes", "canonical_sha256"}
    picked = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in picked} == wanted, "the repo functions moved"
    namespace: dict = {
        "json": json,
        "hashlib": __import__("hashlib"),
        "Any": object,
        "DynamicCircuitIRError": ValueError,
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<repo>", "exec"), namespace)
    canonical_json_bytes = namespace["canonical_json_bytes"]
    canonical_sha256 = namespace["canonical_sha256"]
    for value in (
        {"a": 1},
        {"z": [1, 2, {"q": "x"}], "a": None},
        {"text": "line\nbreak é"},
        [],
    ):
        assert oc_calib.canonical_json_bytes(value) == canonical_json_bytes(value)
        assert oc_calib.canonical_sha256(value) == canonical_sha256(value)


def test_canonical_json_refuses_nan():
    try:
        oc_calib.canonical_json_bytes({"x": float("nan")})
    except ValueError:
        return
    raise AssertionError("NaN must not serialise: allow_nan=False is load-bearing")


# --------------------------------------------------------------------------
# The frozen pack still digests to what the seal names.
# --------------------------------------------------------------------------


def test_frozen_v1_pack_digest_and_shape():
    if not FROZEN_V1.is_file():
        raise SkipTest("the frozen v1 pack is not on this machine")
    rows, payload, digest = oc_calib.load_bank(
        FROZEN_V1, expected_file_sha256=FROZEN_V1_SHA
    )
    assert digest == FROZEN_V1_SHA
    assert payload["schema"] == "neural-forge-whole-response-review-prompts-v1"
    assert len(rows) == 16, "15 intelligence cases plus the safety lane"
    assert {row["id"] for row in rows} == set(oc_assess.LANE_ASK_V1)


def test_load_bank_refuses_a_wrong_digest():
    if not FROZEN_V1.is_file():
        raise SkipTest("the frozen v1 pack is not on this machine")
    try:
        oc_calib.load_bank(FROZEN_V1, expected_file_sha256="0" * 64)
    except oc_calib.CalibrationError:
        return
    raise AssertionError("a pack whose digest does not match must be refused")


def test_load_bank_refuses_duplicate_ids_and_double_prompt_forms():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {"id": "a", "category": "c", "prompt": "x"},
                        {"id": "a", "category": "c", "prompt": "y"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            oc_calib.load_bank(path)
        except oc_calib.CalibrationError:
            pass
        else:
            raise AssertionError("duplicate ids must be refused")
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {
                            "id": "a",
                            "category": "c",
                            "prompt": "x",
                            "messages": [{"role": "user", "content": "y"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            oc_calib.load_bank(path)
        except oc_calib.CalibrationError:
            return
        raise AssertionError("a row with both prompt forms must be refused")


# --------------------------------------------------------------------------
# The copy measure, ported unchanged.
# --------------------------------------------------------------------------


def test_union_coverage_separates_a_copy_from_a_summary():
    payload = (
        "The team compared three battery suppliers. Supplier A is cheapest but "
        "cannot guarantee delivery before launch. Supplier B costs 11 percent "
        "more and passed every reliability test. Supplier C can deliver fastest, "
        "but two samples overheated. The team selected Supplier B."
    )
    copy_with_substitutions = payload.replace("cheapest", "least costly").replace(
        "overheated", "ran hot"
    )
    summary = (
        "They picked Supplier B because it passed reliability testing, and the "
        "open risk is a possible one-week factory slip."
    )
    assert oc_calib.union_coverage(payload, copy_with_substitutions) > 0.826971
    assert oc_calib.union_coverage(payload, summary) < 0.483462


def test_longest_common_substring_is_exact():
    assert oc_calib.longest_common_substring_length("abcdefg", "zzcdefzz") == 4
    assert oc_calib.longest_common_substring_length("", "abc") == 0
    assert oc_calib.longest_common_substring_length("aaaa", "aa") == 2


def test_threshold_numbers_are_the_lfm2_ones():
    assert oc_calib.UNION_MIN_BLOCK == 8
    assert oc_calib.COPY_BACK_UNION_COVERAGE == 0.60
    assert oc_calib.COPY_THRESHOLD_INVARIANCE_INTERVAL == (0.483461, 0.826972)


class _StubTokenizer:
    """A tokenizer whose template branches on `enable_thinking`, like Qwen3.5's."""

    chat_template = "yes"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        body = "".join(m["content"] for m in messages)
        if kwargs.get("enable_thinking") is False:
            return body + "<think></think>"
        return body + "<think>"


class _TemplateWithoutThinking(_StubTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return "".join(m["content"] for m in messages) + "<end>"


def test_thinking_switch_changes_the_rendered_prompt_and_is_recorded():
    """`--thinking off` must reach the template, and the DEFAULT must not move.

    The frozen V5 cap of 1,024 tokens was set on a parent with no reasoning
    mode.  Qwen3.5 opens a think block unless told not to, which is why most
    lanes of the first Q35-4B pass terminated `censored-at-cap`.  The switch
    exists so that surface can be chosen deliberately; this pins that choosing
    it actually changes the prompt, and that not choosing it changes nothing.
    """

    tok = _StubTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    assert oc_calib.render_prompt(tok, messages) == "hello<think>"
    assert oc_calib.render_prompt(tok, messages, thinking="template-default") == "hello<think>"
    assert oc_calib.render_prompt(tok, messages, thinking="off") == "hello<think></think>"

    report = oc_calib.thinking_report(tok, messages, "off")
    assert report["requested"] == "off"
    assert report["took_effect"] is True
    assert report["template_opens_think_block"] is True

    default = oc_calib.thinking_report(tok, messages, "template-default")
    assert default["took_effect"] is False


def test_thinking_off_that_the_template_ignores_is_reported_as_no_effect():
    """A recorded setting the template ignored is worse than no setting.

    LFM2-class templates have no `enable_thinking` variable at all.  Passing
    `off` there must not be written into the dossier as though the surface
    changed, so the report carries the ANSWER and not the request.
    """

    report = oc_calib.thinking_report(
        _TemplateWithoutThinking(), [{"role": "user", "content": "hi"}], "off"
    )
    assert report["requested"] == "off"
    assert report["took_effect"] is False
    assert report["template_opens_think_block"] is False


def test_unknown_thinking_mode_is_refused():
    try:
        oc_calib.chat_template_kwargs_for("maybe")
    except oc_calib.CalibrationError:
        return
    raise AssertionError("an unknown thinking mode must be refused, not guessed")


def test_thinking_is_inside_the_lane_digest_validity_tuple():
    source = (HERE / "oc_run_frozen_bank.py").read_text(encoding="utf-8")
    start = source.index('body["lane_digest_validity"] = {')
    body = source[start : source.index('body["dossier_sha256"]', start)]
    assert '"chat_template_thinking"' in body
    assert '"chat_template_thinking_took_effect"' in body
    #: the smoke and the preflight must render on the SAME surface as the bank
    assert "thinking=str(args.thinking)," in source


def test_bank_expectation_reaches_the_sheet_and_is_labelled_an_annotation():
    """A bank-v2 case's stated expectation must be readable beside the answer.

    The frozen v1 pack's fifteen cases are described in the V5 protocol text; a
    reader has that in front of them.  Bank v2's cases are new, and its
    long-context lanes bury a pinned needle in ten thousand tokens of filler --
    a sheet that prints the haystack and the response but not the needle makes
    Cyril re-derive ground truth the bank already pinned.  This pins that it is
    carried AND that it is labelled as the bank's annotation rather than a key.
    """

    bank_case = {
        "id": "lane-x",
        "category": "long-context",
        "ask": "SELECT",
        "context": "day Thursday, time 6:40 a.m., entrance the north gate",
        "messages": [{"role": "user", "content": "DOCUMENT: filler Question: when?"}],
    }
    record = _record("lane-x", "Thursday at 6:40 a.m. via the north gate.")
    record["category"] = "long-context"
    record["prompt_messages"] = bank_case["messages"]
    dossier = _dossier([record])
    artifact = oc_assess.assess(
        dossier,
        bank={"lane-x": bank_case},
        per_case=None,
        sealed_useful_or_partial=None,
    )
    lane = artifact["lanes"][0]
    assert lane["bank_expectation"] == bank_case["context"]

    sheet = oc_assess.markdown_sheet(artifact, dossier, None)
    assert bank_case["context"] in sheet
    assert "not a key, not a grade" in sheet


def test_out_of_memory_is_an_artifact_box_not_a_content_miss():
    """A prompt that did not fit says nothing about the model.

    The bank-v2 64k rung raised `torch.OutOfMemoryError` on a 3090 before a
    review token existed.  The lane must land in an artifact box -- which counts
    NOTHING under the canonical lens -- and never in a content box, because the
    model was not asked the question.
    """

    box = oc_calib.classify_form(
        response="", payload="some payload", termination="out-of-memory"
    )
    assert box["form_box"] == "EMPTY"
    assert box["is_artifact_box"] is True
    assert "envelope" in box["form_reason"]


def test_the_runner_guards_the_preflight_and_the_decode_against_oom():
    """Neither the diagnostic nor one long lane may end a rung.

    Both of these were real: the preflight (two prefills of the longest prompt,
    ungated) killed the 64k rung with a traceback and no dossier, and a single
    over-long case would otherwise have condemned every case after it.
    """

    runner = (HERE / "oc_run_frozen_bank.py").read_text(encoding="utf-8")
    start = runner.index("preflight = decode_path_agreement(")
    guard = runner[start : start + 2000]
    assert "except RuntimeError" in guard
    assert '"agree": None' in guard
    assert "not_run" in guard

    instrument = (HERE / "oc_calib.py").read_text(encoding="utf-8")
    assert "except torch.OutOfMemoryError as exc:" in instrument
    assert 'termination = "out-of-memory"' in instrument
    assert "torch.cuda.empty_cache()" in instrument
    #: A GREP IS NOT A TEST AND THIS ONE ADMITS IT.  What the source substrings
    #: above cannot see is that `empty_cache()` frees nothing while the forward
    #: output still references the cache; that is checked for real, from inside
    #: a model, by `test_generate_greedy_holds_at_most_one_row_of_kv_cache` and
    #: `test_a_real_out_of_memory_does_not_condemn_the_next_row` below.
    body = instrument[
        instrument.index("def generate_greedy(") : instrument.index("def decode_smoke(")
    ]
    assert body.count("result = None") >= 3, (
        "the forward output must be dropped at end of row and in BOTH handlers"
    )


def test_the_orchestrator_passes_one_thinking_surface_to_every_step():
    """Bank v1 and bank v2 in one calibration must share a surface.

    The orchestrator builds one `common` argument list and reuses it for the
    v1 pass, the v2 pass and the smoke.  If `--thinking` were not in that list,
    every step would silently take the template default while the flag appeared
    to be honoured -- two surfaces filed as one parent's calibration.
    """

    source = (HERE / "oc_parent_calibration.py").read_text(encoding="utf-8")
    common = source[source.index("    common = [") : source.index("    steps = [")]
    assert '"--thinking", str(args.thinking),' in common
    assert '"--thinking",' in source[: source.index("    common = [")], (
        "the orchestrator must expose the switch it forwards"
    )


def _fake_gated_delta_model(*, chunk="fla", conv_prefill=None):
    """A stand-in for a loaded Qwen3.5, with the four primitives bound by hand.

    The real bindings can only be produced by importing `fla` on a machine that
    has it, so the RULE is pinned against a constructed layer instead: that way
    the half-install case -- fla for the recurrence, torch for the convolution --
    is checked on the workstation, where neither package exists, and on the
    server, where one does.
    """

    def stub(module, name):
        def fn(*args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("the probe must not call the kernel")

        fn.__module__ = module
        fn.__name__ = name
        fn.__qualname__ = name
        return fn

    transformers_module = "transformers.models.qwen3_5.modeling_qwen3_5"

    class Layer:
        def __init__(self):
            if chunk == "fla":
                self.chunk_gated_delta_rule = stub(
                    "fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule"
                )
                self.recurrent_gated_delta_rule = stub(
                    "fla.ops.gated_delta_rule.fused_recurrent",
                    "fused_recurrent_gated_delta_rule",
                )
            else:
                self.chunk_gated_delta_rule = stub(
                    transformers_module, "torch_chunk_gated_delta_rule"
                )
                self.recurrent_gated_delta_rule = stub(
                    transformers_module, "torch_recurrent_gated_delta_rule"
                )
            self.causal_conv1d_fn = (
                None
                if conv_prefill is None
                else stub("causal_conv1d.causal_conv1d_interface", "causal_conv1d_fn")
            )
            self.causal_conv1d_update = stub(
                transformers_module, "torch_causal_conv1d_update"
            )

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

        def modules(self):
            return [self, *self.layers]

    return Model()


def test_kernel_path_reports_the_binding_not_the_inventory():
    """THE REFUTED RULE, PINNED SO IT CANNOT COME BACK.

    The first version of this field asserted the pure-torch recurrence whenever
    `causal_conv1d` was missing, on the strength of the transformers log line
    "The fast path is not available ... Falling back to torch implementation".
    That line is printed off `is_fast_path_available`, which is `all()` of four
    callables; the four are bound independently, and
    out/r6-lc64k-Q35-4B.log:137-152 is a traceback through
    `fla/ops/gated_delta_rule/chunk.py:567` taken on a box with no
    `causal_conv1d`.  fla's Triton kernel ran while the field said
    `torch-fallback`.  The label now has to name the binding.
    """

    report = oc_calib.kernel_path_report(_fake_gated_delta_model(chunk="fla"))
    assert report["kernel_report_schema"] == "neural-forge-omni-kernel-path-v2"
    assert report["gated_delta_layers_probed"] == 2
    assert report["bindings_uniform_across_layers"] is True
    assert report["bindings"]["chunk"]["provider"] == "fla"
    assert report["bindings"]["recurrent"]["provider"] == "fla"
    # `causal_conv1d_fn is None` is not "missing", it is the torch convolution:
    # transformers branches `if self.causal_conv1d_fn is not None`.
    assert report["bindings"]["conv-prefill"]["provider"] == "torch"
    assert report["bindings"]["conv-decode"]["provider"] == "torch"
    assert report["kernel_path"].startswith("chunk=fla")
    assert "conv-prefill=torch" in report["kernel_path"]
    assert "torch-fallback" not in report["kernel_path"], (
        "a half-install runs Triton for the recurrence; calling the whole path "
        "torch-fallback is the refuted claim"
    )
    # the inventory is still published, and is a DIFFERENT thing
    assert report["packages_present"] == [
        f"{name}-{report[key]}"
        for name, key in (("causal_conv1d", "causal_conv1d_version"), ("fla", "fla_version"))
        if report[key] is not None
    ]
    assert "inventory_label_v1_refuted" in report


def test_a_pure_torch_binding_says_torch_on_every_slot():
    report = oc_calib.kernel_path_report(_fake_gated_delta_model(chunk="torch"))
    assert report["bindings"]["chunk"]["provider"] == "torch"
    assert report["kernel_path"] == (
        "chunk=torch; recurrent=torch; conv-prefill=torch; conv-decode=torch"
    )


def test_an_unprobed_report_refuses_to_name_a_path():
    """No model, no binding, no path -- and the inventory may not stand in.

    `kernel_path_report()` used to answer from the package list alone, which is
    how a label that was never measured got into a validity tuple.
    """

    report = oc_calib.kernel_path_report()
    assert report["kernel_path"] == "not-probed"
    assert report["gated_delta_layers_probed"] == 0
    assert report["bindings"] == {}
    assert report["bindings_uniform_across_layers"] is None


def test_a_model_without_linear_attention_is_not_applicable_not_torch():
    class Dense:
        def modules(self):
            return [self]

    report = oc_calib.kernel_path_report(Dense())
    assert report["kernel_path"].startswith("not-applicable")


def test_the_kernel_report_schema_is_carried_in_the_validity_tuple():
    """A v1 string and a v2 string are not comparable, so the tuple says which.

    The delivered r6 dossiers carry the v1 inventory semantics, in which
    "fla-0.5.2" meant *fla present, causal_conv1d absent*.  Under v2 the same box
    reads `chunk=fla-0.5.2; ... conv-prefill=torch; ...`.  Without the schema
    element beside it a reader comparing the two would see the kernel field move
    and conclude the surface moved.
    """

    source = (HERE / "oc_run_frozen_bank.py").read_text(encoding="utf-8")
    start = source.index('body["lane_digest_validity"] = {')
    body = source[start : source.index('body["dossier_sha256"]', start)]
    assert '"linear_attention_kernel_report_schema"' in body


def test_kernel_path_is_inside_the_lane_digest_validity_tuple():
    """A moved lane must be attributable, so the kernel path is in the tuple.

    The dtype is in `lane_digest_validity` because a review taken in one dtype
    is not comparable with one taken in another.  The Gated-DeltaNet kernel path
    is in it for the identical reason and this test is what stops it being
    quietly dropped: without it, installing `fla` for the K3 lane would move a
    greedy lane on a near-tie and the dossier would offer no explanation.
    """

    source = (HERE / "oc_run_frozen_bank.py").read_text(encoding="utf-8")
    start = source.index('body["lane_digest_validity"] = {')
    body = source[start : source.index('body["dossier_sha256"]', start)]
    assert '"linear_attention_kernel_path"' in body
    assert '"dtype"' in body
    assert '"linear_attention_kernels": kernel_path_report(model),' in source, (
        "the report must be probed off the LOADED MODEL; a no-argument call "
        "can only answer from the package inventory"
    )


# --------------------------------------------------------------------------
# Form boxes.
# --------------------------------------------------------------------------


def test_form_boxes():
    payload = "The note says the meeting is on Saturday at 4:15 p.m. at the north gate."
    empty = oc_calib.classify_form(
        response="", payload=payload, termination="complete"
    )
    assert empty["form_box"] == "EMPTY" and empty["is_artifact_box"]

    echo = oc_calib.classify_form(
        response=payload, payload=payload, termination="complete"
    )
    assert echo["form_box"] == "ECHO" and echo["is_artifact_box"]

    loop = oc_calib.classify_form(
        response="yes yes yes " * 40,
        payload=payload,
        termination="censored-at-cap",
        token_ids=[7, 7, 7] * 40,
    )
    assert loop["form_box"] == "LOOP"

    cap_ride = oc_calib.classify_form(
        response="the the the the the the the the the the the the",
        payload=payload,
        termination="censored-at-cap",
    )
    assert cap_ride["form_box"] == "CAP_RIDE"

    content = oc_calib.classify_form(
        response="Saturday at 4:15 in the afternoon, entering by the north side.",
        payload=payload,
        termination="complete",
    )
    assert content["form_box"] == "CONTENT" and not content["is_artifact_box"]

    long_answer = oc_calib.classify_form(
        response=" ".join(f"distinct-word-{i}" for i in range(200)),
        payload=payload,
        termination="censored-at-cap",
    )
    assert long_answer["form_box"] == "CONTENT", (
        "a cap hit with unique content is NOT an artifact; the protocol says a "
        "cap hit is never itself a failure"
    )


def test_form_box_never_claims_to_grade():
    box = oc_calib.classify_form(response="x", payload="y", termination="complete")
    assert box["decides_nothing_about_intelligence"] is True


def test_detect_loop_matches_the_sealed_definition():
    assert oc_calib.detect_loop([1, 2, 3] * 30)
    assert not oc_calib.detect_loop(list(range(64)))
    assert not oc_calib.detect_loop([1, 2, 3])


# --------------------------------------------------------------------------
# Bank v2 draft.
# --------------------------------------------------------------------------


def test_bank_v2_builds_and_passes_its_own_self_test():
    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
        failures = bankv2.self_test(body["prompts"], v1_pack=FROZEN_V1)
        assert not failures, failures
        assert body["counts"] == {
            "text_scored": 30,
            "vision_scored": 5,
            "av_spec_only": 5,
            "runnable_now": 35,
            "total_entries": 40,
        }
        assert "DRAFT" in body["status"]
        assert all("DRAFT" in case["status"] for case in body["prompts"])


def test_bank_v2_is_byte_reproducible():
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        a = bankv2.build_bank(Path(one) / "assets")
        b = bankv2.build_bank(Path(two) / "assets")
        assert a["bank_sha256"] == b["bank_sha256"]
        assert oc_calib.canonical_json_bytes(a) == oc_calib.canonical_json_bytes(b)


def test_haystacks_are_deterministic_and_pinned():
    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    for case in body["prompts"]:
        spec = case.get("haystack_spec")
        if not spec:
            continue
        text = bankv2.materialise_haystack(spec)
        assert oc_calib.canonical_sha256({"text": text}) == spec["haystack_sha256"]
        for needle in spec["needles"]:
            assert text.count(needle["text"]) == 1, (
                f"{case['id']}: needle must appear exactly once, not "
                f"{text.count(needle['text'])}"
            )
        assert "<<HAYSTACK>>" in case["messages"][0]["content"]


def test_long_context_targets_are_present_at_every_ordered_length():
    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    targets = {
        case["haystack_spec"]["target_tokens_nominal"]
        for case in body["prompts"]
        if case.get("haystack_spec")
    }
    assert {8192, 32768, 65536, 200000} <= targets


def test_self_test_catches_a_prompt_quotable_case():
    """The self-test must actually fire; a test that never fails proves nothing."""

    case = bankv2._case(
        "quotable-x",
        "lane",
        "cat",
        "A",
        "reachable",
        "PRODUCE",
        [{"role": "user", "content": "Say the word tungsten."}],
        "the model should say tungsten",
        {"required_all": [r"tungsten"], "max_words": 50, "case_sensitive": False},
    )
    failures = bankv2.self_test([case], v1_pack=None)
    assert any("PROMPT-QUOTABLE" in failure for failure in failures), failures


def test_rubric_scoring_publishes_every_check():
    rubric = {
        "required_all": [r"\b100\b", r"\b12\b"],
        "forbidden_any": [r"banana"],
        "max_words": 10,
        "case_sensitive": False,
    }
    good = bankv2.score_rubric("100 then 12", rubric)
    assert good["passed"] and len(good["checks"]) == 4
    bad = bankv2.score_rubric("100 and a banana", rubric)
    assert not bad["passed"]
    kinds = {check["kind"]: check["ok"] for check in bad["checks"]}
    assert kinds["forbidden_any"] is False


#: The shape of the answer that was published as a content failure and was not
#: one: four top-level options, each with indented sub-bullets under it.  Kept
#: verbatim from the 0.8B's `mt-dietary-correction-B` response so the test fails
#: if the counting regresses to `^\s*`.
NESTED_LIST_RESPONSE = """Here are four options:

1.  **Cheese and crackers**
    *   *Why:* familiar and quick to set out.
    *   *Allergy Note:* no shellfish contact.
2.  **Hummus with vegetables (vegan)**
    *   *Why:* naturally vegan.
    *   *Allergy Note:* check the label.
3.  **Fruit platter (vegan)**
    *   *Why:* no substitutions needed.
    *   *Allergy Note:* cut on a clean board.
4.  **Mini quiches**
    *   *Why:* the warm option.
    *   *Allergy Note:* dedicated tray.
"""


def test_list_checks_count_top_level_items_and_not_sub_bullets():
    """The published-as-a-miss formatting artifact, pinned.

    `^\\s*` before the marker made an indented sub-bullet a top-level item, so
    four options with two notes each counted as nine distinct items and
    `mt-dietary-correction-B` was reported as a model failure.  It was a
    formatting artifact and the artifact-tolerant lens forbids scoring one.
    """

    items = bankv2.top_level_list_items(NESTED_LIST_RESPONSE)
    assert len(items) == 4, items
    verdict = bankv2.score_rubric(
        NESTED_LIST_RESPONSE, {"distinct_numbered_items": 4, "case_sensitive": False}
    )
    assert verdict["passed"] is True
    assert verdict["checks"][0]["observed"] == 4

    # And the FALSE PASS in the other direction: one step with four sub-bullets
    # is one step, not four.
    one_step = "1. Do the thing\n    - a\n    - b\n    - c\n    - d\n"
    assert len(bankv2.top_level_list_items(one_step)) == 1
    assert (
        bankv2.score_rubric(one_step, {"numbered_steps_min": 4})["passed"] is False
    )

    # A uniformly indented list is still a list -- anchoring hard at column 0
    # would have scored this zero.
    indented = "  1. first\n  2. second\n     - detail\n  3. third\n"
    assert len(bankv2.top_level_list_items(indented)) == 3
    assert bankv2.top_level_list_items("no list here at all") == []


def test_every_rubric_budget_agrees_with_the_ask_it_is_beside():
    """A diagnostic that disagrees with its own prompt misleads the reader.

    `instr-word-budget-A` asks for 30 words or fewer and its check allowed 40;
    `code-rle-encode-A` says "do not import any module" and its check forbade
    only `re` and `itertools`, so `import collections` passed a case that forbids
    it.  Both are pinned here against the built bank, not against a copy.
    """

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    cases = {case["id"]: case for case in body["prompts"]}

    budget = cases["instr-word-budget-A"]
    asked = int(
        [
            word
            for word in bankv2.user_turns_text(budget["messages"]).split()
            if word.isdigit()
        ][0]
    )
    assert budget["rubric"]["max_words"] == asked == 30

    rle = cases["code-rle-encode-A"]
    assert "Do not import any module" in bankv2.user_turns_text(rle["messages"])
    for module in ("re", "itertools", "collections", "os"):
        offending = f"```python\nimport {module}\ndef run_length(s):\n    return []\n```"
        assert bankv2.score_rubric(offending, rle["rubric"])["passed"] is False, module
    clean = "```python\ndef run_length(s):\n    return []\n```"
    assert bankv2.score_rubric(clean, rle["rubric"])["passed"] is True
    from_form = (
        "```python\nfrom collections import Counter\n"
        "def run_length(s):\n    return []\n```"
    )
    assert bankv2.score_rubric(from_form, rle["rubric"])["passed"] is False


def test_rubric_ast_check_parses_and_never_executes():
    rubric = {"ast_function_name": "run_length", "case_sensitive": True}
    ok = bankv2.score_rubric(
        "```python\ndef run_length(s):\n    return []\n```", rubric
    )
    assert ok["passed"]
    dangerous = bankv2.score_rubric(
        "```python\nimport os\nos.system('echo owned')\ndef run_length(s):\n    return []\n```",
        rubric,
    )
    # It parses (so the check passes) and NOTHING RAN -- if it had, the test
    # process would have printed 'owned'.
    assert dangerous["passed"]
    assert dangerous["checks"][0]["note"] == "PARSED, NEVER EXECUTED"


def test_media_out_cases_are_not_machine_gradable():
    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    media = [c for c in body["prompts"] if c.get("media_out")]
    assert media
    for case in media:
        verdict = bankv2.score_rubric("anything at all", case["rubric"])
        assert verdict["applicable"] is False
        assert verdict["passed"] is None


# --------------------------------------------------------------------------
# Test cards.
# --------------------------------------------------------------------------


def test_test_cards_are_valid_pngs_with_pinned_truth():
    with tempfile.TemporaryDirectory() as tmp:
        assets = oc_testcards.build_cards(Path(tmp))
        assert len(assets) == 5
        for asset in assets:
            raw = (Path(tmp) / asset["path"]).read_bytes()
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"
            # length(0) + "IEND" + crc32("IEND") == 12 bytes
            assert raw[-12:] == b"\x00\x00\x00\x00IEND\xaeB`\x82"
            assert asset["sha256"] == oc_calib.file_sha256(Path(tmp) / asset["path"])
        truth = {asset["asset_id"]: asset["ground_truth"] for asset in assets}
        assert truth["vis-ocr-receipt-total-A"]["total"] == "8.37"
        assert truth["vis-count-shapes-B"]["blue_squares"] == 4
        assert truth["vis-chart-trend-B"]["highest_month"] == "MAY"
        assert truth["vis-grid-cell-select-A"]["green_letter"] == "E"


def test_test_cards_are_byte_reproducible():
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        a = oc_testcards.build_cards(Path(one))
        b = oc_testcards.build_cards(Path(two))
        assert [x["sha256"] for x in a] == [x["sha256"] for x in b]


def test_card_deflate_is_self_written_and_round_trips():
    """THE CARDS' BYTES MUST NOT DEPEND ON THE ZLIB BUILD THAT RAN THEM.

    `zlib.compress(raw, 9)` was measured producing different bytes for three of
    the five cards on CPython 3.14 (workstation) and CPython 3.12 (server), so
    the bank pinned a different asset digest on each machine.  The stream is now
    written here -- fixed Huffman codes out of RFC 1951, one matcher, no library
    heuristics -- and `zlib.decompress` is the independent reader that proves it
    is a real deflate stream rather than merely a stable one.
    """

    import zlib

    for blob in (
        b"",
        b"a",
        b"ab" * 64,
        bytes(range(256)) * 4,
        b"\x00" * 70_000,
        bytes((i * 31 + 7) % 251 for i in range(20_000)),
    ):
        stream = oc_testcards.deflate_deterministic(blob, row_length=97)
        assert zlib.decompress(stream) == blob, len(blob)
        assert stream[:2] == b"\x78\x01"
        assert (stream[0] << 8 | stream[1]) % 31 == 0

    canvas = oc_testcards.Canvas(24, 9)
    canvas.rectangle(2, 2, 8, 4, (200, 30, 30))
    png = canvas.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8 : start - 4], "big")
    raw = zlib.decompress(png[start : start + length])
    assert len(raw) == 9 * (24 * 3 + 1)
    assert raw[0] == 0, "filter type 0, so a decoder sees the pixels"


def test_card_font_refuses_an_unknown_glyph():
    canvas = oc_testcards.Canvas(40, 20)
    try:
        canvas.text(0, 0, "é")
    except SystemExit:
        return
    raise AssertionError("an unknown glyph must refuse, not silently blank")


# --------------------------------------------------------------------------
# The two columns.
# --------------------------------------------------------------------------


def _dossier(records):
    return {
        "schema": "neural-forge-omni-parent-calibration-dossier-v1",
        "label": "TEST",
        "parent_code": "TEST",
        "stage": "parent",
        "is_random_init": False,
        "dossier_sha256": "0" * 64,
        "lane_digest": "1" * 64,
        "lane_digest_validity": {},
        "bank": {"file_sha256": FROZEN_V1_SHA},
        "generation": {
            "dtype": "bfloat16",
            "compute_architecture": "sm86 test",
            "maximum_new_tokens": 1024,
        },
        "emit_zero_floor": {
            "passed": True,
            "min_terminal_l2": 1.0,
            "min_logit_std": 1.0,
            "probe_count": 8,
            "distinct_argmax_tokens": 5,
            "definition": "test",
        },
        "decode_path_preflight": {"agree": True},
        "decode_smoke": {"verdict": "pending", "records": []},
        "provenance": {"model_directory": "/tmp/x", "directory_sha256": "2" * 64},
        "export": {
            "loaded_class": "TestModel",
            "parameters_counted_live": 123,
            "transformers_version": "5.12.1",
        },
        "child_responses": records,
    }


def _record(lane, text, termination="complete", messages=None):
    return {
        "id": lane,
        "category": "test",
        "prompt_messages": messages or [{"role": "user", "content": "prompt text"}],
        "response_verbatim": text,
        "response_sha256": oc_calib.canonical_sha256({"text": text}),
        "generated_tokens": len(text.split()),
        "termination": termination,
        "loop_detected": False,
    }


def test_columns_are_upper_bounds_without_a_reading():
    records = [
        _record("conversation-natural-greeting", "Hello, how did it go today?"),
        _record("instruction-three-actions", ""),
        _record("explanation-seasons", "The tilt of the axis changes sunlight."),
    ]
    artifact = oc_assess.assess(
        _dossier(records), bank={}, per_case=None, sealed_useful_or_partial=None
    )
    columns = artifact["columns"]
    assert columns["is_upper_bound"] is True
    assert columns["column_a_copy_back_counts"] == 2, "the empty lane cannot hold"
    assert columns["column_b_copy_back_refused"] == 2
    assert artifact["canonical_three_box"]["read_required"] == 3


def test_column_b_subtracts_an_echo_on_a_held_lane():
    payload = (
        "The team compared three battery suppliers. Supplier A is cheapest but "
        "cannot guarantee delivery before launch. Supplier B costs 11 percent "
        "more and passed every reliability test. The team selected Supplier B."
    )
    prompt = (
        "Summarize this update in two sentences, preserving the decision and "
        "the unresolved risk:\n\n" + payload
    )
    records = [
        _record(
            "summarization-priorities",
            payload,
            messages=[{"role": "user", "content": prompt}],
        ),
        _record("explanation-seasons", "Axial tilt changes how much sun each pole gets."),
    ]
    per_case = {
        "summarization-priorities": "Useful",
        "explanation-seasons": "Useful",
    }
    artifact = oc_assess.assess(
        _dossier(records), bank={}, per_case=per_case, sealed_useful_or_partial=2
    )
    assert artifact["columns"]["column_a_copy_back_counts"] == 2
    assert artifact["columns"]["column_b_copy_back_refused"] == 1
    assert artifact["columns"]["voided_lanes"] == ["summarization-priorities"]
    assert artifact["columns"]["is_upper_bound"] is False


def test_assess_refuses_a_per_case_that_disagrees_with_the_sealed_count():
    records = [_record("explanation-seasons", "text")]
    try:
        oc_assess.assess(
            _dossier(records),
            bank={},
            per_case={"explanation-seasons": "Useful"},
            sealed_useful_or_partial=7,
        )
    except oc_calib.CalibrationError:
        return
    raise AssertionError("column A is the reading, not a recount; it must refuse")


def test_a_miss_is_never_re_read_into_a_hold():
    records = [_record("explanation-seasons", "Axial tilt changes the sunlight.")]
    artifact = oc_assess.assess(
        _dossier(records),
        bank={},
        per_case={"explanation-seasons": "Observed miss"},
        sealed_useful_or_partial=0,
    )
    assert artifact["columns"]["column_a_copy_back_counts"] == 0
    assert artifact["columns"]["column_b_copy_back_refused"] == 0
    assert artifact["lanes"][0]["canonical_box"] == "content-miss"


def test_artifact_miss_versus_content_miss():
    records = [
        _record("explanation-seasons", "", termination="complete"),
        _record("comparison-ram-rom", "RAM is a kind of disk."),
    ]
    artifact = oc_assess.assess(
        _dossier(records),
        bank={},
        per_case={
            "explanation-seasons": "Observed miss",
            "comparison-ram-rom": "Observed miss",
        },
        sealed_useful_or_partial=0,
    )
    boxes = {lane["lane"]: lane["canonical_box"] for lane in artifact["lanes"]}
    assert boxes["explanation-seasons"] == "artifact-miss"
    assert boxes["comparison-ram-rom"] == "content-miss"
    assert artifact["canonical_three_box"]["artifact_miss"] == 1
    assert artifact["canonical_three_box"]["content_miss"] == 1


def test_safety_lane_is_charged_not_scored():
    records = [
        _record("safety-household-chemical", "I will not help with that."),
        _record("explanation-seasons", "Axial tilt."),
    ]
    artifact = oc_assess.assess(
        _dossier(records), bank={}, per_case=None, sealed_useful_or_partial=None
    )
    assert artifact["columns"]["denominator_scored_lanes"] == 1
    safety = [lane for lane in artifact["lanes"] if lane["is_safety_lane"]][0]
    assert safety["scored"] is False


def test_echo_on_a_produce_lane_is_published_as_an_instrument_caution():
    """Measured on Qwen3.5-0.8B: four of fifteen PRODUCE/SELECT lanes tripped the
    LFM2 copy threshold because a capable model legitimately quotes the material
    it was asked to work on.  The caution must fire without a reading attached,
    because that is exactly when nobody is looking."""

    payload = "def last_three(values, start):\n    return values[start - 1:]"
    prompt = "Fix this off-by-one bug.\n\n" + payload
    records = [
        _record(
            "debugging-off-by-one",
            "The bug is the -1. " + payload.replace("start - 1", "start"),
            messages=[{"role": "user", "content": prompt}],
        )
    ]
    artifact = oc_assess.assess(
        _dossier(records), bank={}, per_case=None, sealed_useful_or_partial=None
    )
    assert artifact["lanes"][0]["form_box"] == "ECHO"
    assert artifact["instrument_caution_count"] == 1
    assert artifact["instrument_cautions"][0]["lane"] == "debugging-off-by-one"
    assert artifact["anomalies"] == [], "an anomaly needs a READING to fire"
    sheet = oc_assess.markdown_sheet(artifact, _dossier(records), None)
    assert "Instrument cautions" in sheet
    assert "The threshold was not moved" in sheet
    assert oc_calib.COPY_BACK_UNION_COVERAGE == 0.60, "and it really was not moved"


def test_assess_refuses_an_unknown_lane_in_the_per_case_file():
    try:
        oc_assess.assess(
            _dossier([_record("explanation-seasons", "x")]),
            bank={},
            per_case={"not-a-lane": "Useful"},
            sealed_useful_or_partial=None,
        )
    except oc_calib.CalibrationError:
        return
    raise AssertionError("a per-case file naming an absent lane must be refused")


def test_lane_ask_and_anchor_tables_are_the_sealed_ones():
    assert len(oc_assess.LANE_ASK_V1) == 16
    assert len(oc_assess.PAYLOAD_ANCHORS_V1) == 4
    assert set(oc_assess.PAYLOAD_ANCHORS_V1) <= set(oc_assess.LANE_ASK_V1)
    assert oc_assess.LANE_ASK_V1["summarization-priorities"] == "PRESERVE"
    assert oc_assess.LANE_ASK_V1["context-retrieval"] == "SELECT"


def test_payload_anchor_slices_the_supplied_material_only():
    prompt = (
        "Answer using only the note.\n\nNOTE: the meeting is Saturday at 4:15 "
        "p.m. at the north gate.\n\nQuestion: when and where?"
    )
    payload = oc_assess.lane_payload(
        "context-retrieval",
        [{"role": "user", "content": prompt}],
        oc_assess.PAYLOAD_ANCHORS_V1["context-retrieval"],
    )
    assert "Saturday" in payload
    assert "Answer using only" not in payload
    assert "Question:" not in payload


def test_markdown_sheet_carries_the_decision_disclaimer_and_every_case():
    records = [
        _record("explanation-seasons", "Axial tilt changes the sunlight."),
        _record("safety-household-chemical", "No."),
    ]
    dossier = _dossier(records)
    artifact = oc_assess.assess(
        dossier, bank={}, per_case=None, sealed_useful_or_partial=None
    )
    sheet = oc_assess.markdown_sheet(artifact, dossier, None)
    assert "THE DECISION IS CYRIL'S" in sheet
    assert "Axial tilt changes the sunlight." in sheet
    assert "SAFETY LANE" in sheet
    assert "artifact-miss or content-miss" in sheet
    assert "UPPER BOUND" in sheet


# --------------------------------------------------------------------------
# Landing detection.
# --------------------------------------------------------------------------


def _safetensors_bytes(payload: int = 2048, *, truncate_to: int | None = None) -> bytes:
    """A REAL safetensors file: 8-byte header length, JSON header, tensor bytes.

    The landing test reads the header and computes the size the file must have,
    so a fixture of `b"x" * 2048` is not a shard and must not be used to prove a
    directory landed.
    """

    header = json.dumps(
        {"t": {"dtype": "F32", "shape": [payload // 4], "data_offsets": [0, payload]}}
    ).encode("utf-8")
    blob = len(header).to_bytes(8, "little") + header + (b"\x00" * payload)
    return blob if truncate_to is None else blob[:truncate_to]


def test_landing_report_rejects_a_missing_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}),
            encoding="utf-8",
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}),
            encoding="utf-8",
        )
        (root / "s1.safetensors").write_bytes(_safetensors_bytes())
        report = orchestrator.landing_report(root)
        assert report["landed"] is False
        assert report["shards_present"] == 1 and report["shards_expected"] == 2
        (root / "s2.safetensors").write_bytes(_safetensors_bytes())
        assert orchestrator.landing_report(root)["landed"] is True


def test_landing_report_rejects_a_truncated_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(
            json.dumps({"architectures": ["X"]}), encoding="utf-8"
        )
        (root / "model.safetensors").write_bytes(b"x" * 10)
        report = orchestrator.landing_report(root)
        assert report["landed"] is False


def test_a_big_shard_truncated_to_a_stub_is_not_landed():
    """The bug this replaced a flat byte floor for.

    2 KiB of a shard whose header says it needs 5 GB used to pass the landing
    test, because 2,048 > 1,024, and the failure then surfaced as a loader
    exception -- which reads as a code fault, not as a bad download.
    """

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(
            json.dumps({"architectures": ["X"]}), encoding="utf-8"
        )
        whole = _safetensors_bytes(1 << 20)
        (root / "model.safetensors").write_bytes(whole[:2048])
        report = orchestrator.landing_report(root)
        assert report["landed"] is False
        detail = report["shard_completeness"][0]
        assert detail["ok"] is False
        assert detail["observed_bytes"] == 2048
        assert detail["required_bytes"] > 1 << 20
        assert "truncated" in detail["why"]
        (root / "model.safetensors").write_bytes(whole)
        again = orchestrator.landing_report(root)
        assert again["landed"] is True
        assert again["shard_completeness"][0]["tensors"] == 1


def test_shard_completeness_refuses_a_nonsense_header_length():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.safetensors"
        path.write_bytes((1 << 40).to_bytes(8, "little") + b"{}")
        verdict = orchestrator.shard_completeness(path)
        assert verdict["ok"] is False
        assert "not a safetensors header" in verdict["why"]


def test_directory_digest_is_stable_and_excludes_caches():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
        (root / "README.md").write_text("ignored", encoding="utf-8")
        (root / ".cache").mkdir()
        (root / ".cache" / "junk.json").write_text("{}", encoding="utf-8")
        first, files = oc_calib.directory_digest(root)
        assert {f["path"] for f in files} == {"config.json", "model.safetensors"}
        (root / "README.md").write_text("changed", encoding="utf-8")
        second, _ = oc_calib.directory_digest(root)
        assert first == second
        (root / "model.safetensors").write_bytes(b"different")
        third, _ = oc_calib.directory_digest(root)
        assert third != first


def test_ordinary_file_refuses_a_hard_link():
    import os

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = root / "pack.json"
        original.write_text("{}", encoding="utf-8")
        link = root / "link.json"
        try:
            os.link(original, link)
        except OSError:
            raise SkipTest("hard links are unavailable on this filesystem")
        try:
            oc_calib.ordinary_file(link, "pack")
        except oc_calib.CalibrationError:
            return
        raise AssertionError("a hard-linked pack must be refused")


# --------------------------------------------------------------------------
# Torch-dependent.
# --------------------------------------------------------------------------


def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except Exception:
        raise SkipTest("torch is not installed here; run this file on the server")
    return True


def test_emit_zero_floor_fails_a_model_that_emits_nothing():
    if not _torch_or_skip():
        return
    import torch

    class ZeroModel(torch.nn.Module):
        def forward(self, input_ids, output_hidden_states=False, use_cache=False):
            batch, length = input_ids.shape
            zeros = torch.zeros(batch, length, 8)
            return type(
                "Out",
                (),
                {"hidden_states": [zeros], "logits": torch.zeros(batch, length, 5)},
            )()

    class LiveModel(torch.nn.Module):
        def forward(self, input_ids, output_hidden_states=False, use_cache=False):
            batch, length = input_ids.shape
            hidden = torch.arange(batch * length * 8, dtype=torch.float32).reshape(
                batch, length, 8
            )
            return type(
                "Out",
                (),
                {
                    "hidden_states": [hidden],
                    "logits": torch.arange(
                        batch * length * 5, dtype=torch.float32
                    ).reshape(batch, length, 5),
                },
            )()

    class StubTokenizer:
        def __call__(self, text, return_tensors=None):
            return type(
                "Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])}
            )()

    prompts = ["a", "b"]
    dead = oc_calib.emit_zero_floor(
        ZeroModel(), StubTokenizer(), prompts, device="cpu"
    )
    assert dead["passed"] is False
    assert dead["min_terminal_l2"] == 0.0

    alive = oc_calib.emit_zero_floor(
        LiveModel(), StubTokenizer(), prompts, device="cpu"
    )
    assert alive["passed"] is True
    assert alive["min_terminal_l2"] > 0
    assert alive["reference_terminal_relative_l2"] is None


def test_generate_greedy_records_all_three_terminations():
    if not _torch_or_skip():
        return
    import torch

    class CountingModel(torch.nn.Module):
        """Emits ids 10, 11, 12, ... and then the stop id."""

        def __init__(self, stop_after):
            super().__init__()
            self.stop_after = stop_after
            self.step = 0

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            self.step += 1
            logits = torch.zeros(1, input_ids.shape[-1], 128)
            token = 99 if self.step > self.stop_after else 10 + self.step
            logits[0, -1, token] = 10.0
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    class StubTokenizer:
        chat_template = "yes"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return " ".join(m["content"] for m in messages)

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2]])})()

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

    rows = [{"id": "case", "category": "c", "prompt": "hello"}]
    complete = oc_calib.generate_greedy(
        CountingModel(3), StubTokenizer(), rows, cap=16, device="cpu",
        eos_id=99, label="t",
    )
    assert complete[0]["termination"] == "complete"
    assert complete[0]["generated_tokens"] == 3

    capped = oc_calib.generate_greedy(
        CountingModel(1000), StubTokenizer(), rows, cap=5, device="cpu",
        eos_id=99, label="t",
    )
    assert capped[0]["termination"] == "censored-at-cap"
    assert capped[0]["generated_tokens"] == 5

    class RaisingModel(torch.nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("CUDA out of memory (simulated)")

    errored = oc_calib.generate_greedy(
        RaisingModel(), StubTokenizer(), rows, cap=5, device="cpu",
        eos_id=99, label="t",
    )
    assert errored[0]["termination"] == "runtime-error"
    assert "CUDA out of memory" in errored[0]["runtime_error"]


def _cache_counting_pieces():
    """A model that reports how many of its own forward outputs are still alive.

    This is the only way to see the defect it was written for.  `result` in
    `generate_greedy` is a function-scoped local rebound by the NEXT forward, so
    the previous row's `result.past_key_values` -- a live Cache -- stayed
    referenced through the whole of the next row's prefill.  `past = None` frees
    nothing while `result` holds it, and `torch.cuda.empty_cache()` then frees
    nothing either.  On the 64k rung that is ~20 GiB carried into the next lane:
    one genuine out-of-memory becomes three, and the published envelope depends
    on the order the lanes happen to run in.
    """

    import torch
    import weakref

    class Out:
        __slots__ = ("logits", "past_key_values", "__weakref__")

    class Model(torch.nn.Module):
        def __init__(self, oom_on_prefill=()):
            super().__init__()
            self.live: list[Any] = []
            self.alive_at_prefill: list[int] = []
            self.prefills = 0
            self.oom_on_prefill = set(oom_on_prefill)

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            # a prompt is 4 tokens, a decode step is 1: a wide input is a new row
            if int(input_ids.shape[-1]) > 1:
                self.prefills += 1
                self.alive_at_prefill.append(
                    sum(1 for ref in self.live if ref() is not None)
                )
                if self.prefills in self.oom_on_prefill:
                    raise torch.OutOfMemoryError(
                        "CUDA out of memory. Tried to allocate 3.95 GiB (simulated)"
                    )
            out = Out()
            out.logits = torch.zeros(1, int(input_ids.shape[-1]), 64)
            out.logits[0, -1, 21] = 7.0
            out.past_key_values = object()
            self.live.append(weakref.ref(out))
            return out

    class StubTokenizer:
        chat_template = "yes"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3, 4]])})()

        def decode(self, ids, skip_special_tokens=False):
            return ",".join(str(i) for i in ids)

    return Model, StubTokenizer


def test_generate_greedy_holds_at_most_one_row_of_kv_cache():
    """No row may still be holding its cache when the next row prefills."""

    if not _torch_or_skip():
        return

    Model, StubTokenizer = _cache_counting_pieces()
    model = Model()
    rows = [{"id": f"case-{i}", "category": "c", "prompt": "p"} for i in range(3)]
    records = oc_calib.generate_greedy(
        model, StubTokenizer(), rows, cap=3, device="cpu", eos_id=None, label="t"
    )
    assert len(records) == 3
    assert model.alive_at_prefill == [0, 0, 0], (
        "a previous row's forward output was still alive at the next prefill: "
        f"{model.alive_at_prefill}"
    )


def test_a_real_out_of_memory_does_not_condemn_the_next_row():
    """The rung's remaining lanes must start on a clean card.

    Constructed as a real `torch.OutOfMemoryError` rather than the
    `RuntimeError("CUDA out of memory (simulated)")` the terminations test uses,
    because the classifier keying on the type is not the claim -- the claim is
    that lane 2 begins with nothing of lane 1 held.
    """

    if not _torch_or_skip():
        return

    Model, StubTokenizer = _cache_counting_pieces()
    model = Model(oom_on_prefill=(2,))
    rows = [{"id": f"case-{i}", "category": "c", "prompt": "p"} for i in range(3)]
    records = oc_calib.generate_greedy(
        model, StubTokenizer(), rows, cap=3, device="cpu", eos_id=None, label="t"
    )
    assert [r["termination"] for r in records] == [
        "censored-at-cap",
        "out-of-memory",
        "censored-at-cap",
    ]
    assert "3.95 GiB" in records[1]["runtime_error"]
    assert records[1]["generated_tokens"] == 0
    assert records[2]["generated_tokens"] == 3, (
        "the lane after an out-of-memory must still get its chance"
    )
    assert model.alive_at_prefill == [0, 0, 0], model.alive_at_prefill


def test_the_decode_path_preflight_drops_its_cache_before_generate():
    """Two prefills of the longest prompt, never both resident.

    The preflight is the most expensive thing in a long-context rung and it is a
    diagnostic that gates nothing, so it must not be what runs the card out of
    memory.  Holding the manual loop's cache across `model.generate` is exactly
    how it did.
    """

    runner = (HERE / "oc_run_frozen_bank.py").read_text(encoding="utf-8")
    body = runner[
        runner.index("def decode_path_agreement(") : runner.index("def stop_token_ids(")
    ]
    loop = body.index("for _ in range(steps):")
    generate = body.index("produced = model.generate(")
    between = body[loop:generate]
    assert "result = None" in between
    assert "torch.cuda.empty_cache()" in between
    assert between.index("result = None") < between.index("torch.cuda.empty_cache()")


def test_extra_stop_ids_end_a_turn():
    if not _torch_or_skip():
        return
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = 0

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            self.step += 1
            logits = torch.zeros(1, input_ids.shape[-1], 64)
            logits[0, -1, 55 if self.step > 2 else 20] = 5.0
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    class StubTokenizer:
        chat_template = "yes"

        def apply_chat_template(self, messages, **kwargs):
            return "x"

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1]])})()

        def decode(self, ids, skip_special_tokens=False):
            return ",".join(str(i) for i in ids)

    rows = [{"id": "c", "category": "c", "prompt": "p"}]
    out = oc_calib.generate_greedy(
        Model(), StubTokenizer(), rows, cap=10, device="cpu", eos_id=None,
        label="t", extra_eos_ids=[55],
    )
    assert out[0]["termination"] == "complete"
    assert out[0]["generated_tokens"] == 2


def test_logits_to_keep_is_detected_and_changes_no_token():
    """The prefill must not materialise 8k x 248k logits, and trimming them must
    not move a single generated id.  Both halves are checked here because the
    first is why the flag exists and the second is why it is safe."""

    if not _torch_or_skip():
        return
    import torch

    class Model(torch.nn.Module):
        """Emits a deterministic sequence; honours logits_to_keep like HF does."""

        def __init__(self):
            super().__init__()
            self.step = 0
            self.widths: list[int] = []

        def forward(self, input_ids, past_key_values=None, use_cache=True,
                    logits_to_keep=0):
            self.step += 1
            length = input_ids.shape[-1]
            width = length if not logits_to_keep else min(logits_to_keep, length)
            self.widths.append(width)
            logits = torch.zeros(1, width, 128)
            logits[0, -1, 30 + self.step] = 9.0
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    class NoKeepModel(Model):
        def forward(self, input_ids, past_key_values=None, use_cache=True):
            return super().forward(input_ids, past_key_values, use_cache, 0)

    class StubTokenizer:
        chat_template = "yes"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})()

        def decode(self, ids, skip_special_tokens=False):
            return ",".join(str(i) for i in ids)

    assert oc_calib.logits_to_keep_kwargs(Model()) == {"logits_to_keep": 1}
    assert oc_calib.logits_to_keep_kwargs(NoKeepModel()) == {}

    rows = [{"id": "c", "category": "c", "prompt": "p"}]
    trimmed_model = Model()
    trimmed = oc_calib.generate_greedy(
        trimmed_model, StubTokenizer(), rows, cap=4, device="cpu",
        eos_id=None, label="t",
    )
    full_model = NoKeepModel()
    full = oc_calib.generate_greedy(
        full_model, StubTokenizer(), rows, cap=4, device="cpu",
        eos_id=None, label="t",
    )
    assert trimmed_model.widths[0] == 1, "the prefill must keep ONE position"
    assert full_model.widths[0] == 5, "without the flag it keeps all five"
    assert trimmed[0]["response_verbatim"] == full[0]["response_verbatim"]
    assert trimmed[0]["logits_to_keep_used"] is True
    assert full[0]["logits_to_keep_used"] is False


def test_render_prompt_refuses_a_tokenizer_with_no_chat_template():
    class NoTemplate:
        chat_template = None

    try:
        oc_calib.render_prompt(NoTemplate(), [{"role": "user", "content": "x"}])
    except oc_calib.CalibrationError:
        return
    raise AssertionError(
        "a raw-concatenation fallback would silently grade a different surface"
    )


# --------------------------------------------------------------------------
# End to end: the bank emitter's CLI refuses to write a failing bank.
# --------------------------------------------------------------------------


def test_bank_cli_self_test_exits_zero_and_actually_writes_the_bank():
    """A REPRODUCIBILITY RECIPE THAT WRITES NOTHING CANNOT DETECT A MOVED DIGEST.

    `--self-test` used to print its failure count and return before the write.
    README.md documents that exact command as "re-author + self-test; the digest
    must not move", so anyone following it compared the on-disk bank against
    itself and read an unchanged digest as a successful reproduction.  Now it
    writes, and `--check-only` is the flag for the other intent.
    """

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bank.json"
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "oc_bank_v2_draft.py"),
                "--out", str(out),
                "--assets-dir", str(Path(tmp) / "assets"),
                "--v1-pack", str(FROZEN_V1),
                "--self-test",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "0 failure(s)" in result.stdout
        assert out.is_file(), "--self-test must leave the bank it claims to check"
        first = oc_calib.file_sha256(out)
        assert first in result.stdout
        assert (out.parent / (out.name + ".sha256")).is_file()

        # And it is deterministic: the second pass reproduces the bytes.
        again = subprocess.run(
            [
                sys.executable,
                str(HERE / "oc_bank_v2_draft.py"),
                "--out", str(out),
                "--assets-dir", str(Path(tmp) / "assets"),
                "--v1-pack", str(FROZEN_V1),
                "--self-test",
            ],
            capture_output=True,
            text=True,
        )
        assert again.returncode == 0, again.stdout + again.stderr
        assert oc_calib.file_sha256(out) == first


def test_bank_cli_check_only_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bank.json"
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "oc_bank_v2_draft.py"),
                "--out", str(out),
                "--assets-dir", str(Path(tmp) / "assets"),
                "--v1-pack", str(FROZEN_V1),
                "--check-only",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "nothing was written" in result.stdout
        assert not out.exists()


def test_emitted_bank_round_trips_through_load_bank():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bank.json"
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "oc_bank_v2_draft.py"),
                "--out", str(out),
                "--assets-dir", str(Path(tmp) / "assets"),
                "--v1-pack", str(FROZEN_V1),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rows, payload, digest = oc_calib.load_bank(out, expected_file_sha256=None)
        assert len(rows) == 40
        assert payload["bank_sha256"]
        assert digest == oc_calib.file_sha256(out)
        sidecar = (out.parent / (out.name + ".sha256")).read_text(encoding="utf-8")
        assert digest in sidecar


# --------------------------------------------------------------------------
# The three review blockers, each with a test that fails on the old behaviour.
# --------------------------------------------------------------------------


def test_an_echo_lane_stays_inside_the_column_a_upper_bound():
    """A BOUND A LEGAL READING CAN EXCEED IS NOT A BOUND.

    The no-reading branch used to drop ECHO lanes from column A while listing
    them as voided, so `A == B` was printed beside a list of subtractions, and a
    reading that held an ECHO lane -- which the per-case branch explicitly
    permits, publishing an anomaly rather than refusing -- came out ABOVE the
    published upper bound.
    """

    payload = "def last_three(values, start):\n    return values[start - 1:]"
    prompt = "Fix this off-by-one bug.\n\n" + payload
    records = [
        _record(
            "debugging-off-by-one",
            "The bug is the -1. " + payload.replace("start - 1", "start"),
            messages=[{"role": "user", "content": prompt}],
        ),
        _record("explanation-seasons", "Axial tilt changes how much sun each pole gets."),
        _record("instruction-three-actions", ""),
    ]
    bound = oc_assess.assess(
        _dossier(records), bank={}, per_case=None, sealed_useful_or_partial=None
    )
    columns = bound["columns"]
    assert columns["is_upper_bound"] is True
    assert columns["voided_lanes"] == ["debugging-off-by-one"]
    assert columns["column_a_copy_back_counts"] == 2, (
        "the ECHO lane and the content lane are both inside the bound; only the "
        "empty lane is refutable"
    )
    assert columns["column_b_copy_back_refused"] == 1
    assert (
        columns["column_b_copy_back_refused"]
        == columns["column_a_copy_back_counts"] - len(columns["voided_lanes"])
    ), "the same subtraction in both branches, or the sheet is incoherent"

    reading = oc_assess.assess(
        _dossier(records),
        bank={},
        per_case={
            "debugging-off-by-one": "Useful",
            "explanation-seasons": "Useful",
            "instruction-three-actions": "Observed miss",
        },
        sealed_useful_or_partial=2,
    )
    assert reading["columns"]["column_a_copy_back_counts"] == 2
    assert (
        reading["columns"]["column_a_copy_back_counts"]
        <= columns["column_a_copy_back_counts"]
    ), "a legal reading must never exceed the published upper bound"
    assert reading["anomalies"], "a held ECHO lane is still published as an exception"


def test_assess_refuses_a_half_finished_reading():
    records = [
        _record("explanation-seasons", "Axial tilt."),
        _record("comparison-ram-rom", "RAM is volatile working memory."),
    ]
    try:
        oc_assess.assess(
            _dossier(records),
            bank={},
            per_case={"explanation-seasons": "Useful"},
            sealed_useful_or_partial=None,
        )
    except oc_calib.CalibrationError as exc:
        assert "comparison-ram-rom" in str(exc)
        return
    raise AssertionError(
        "an unnamed lane would be counted as not-held, which invents a miss"
    )


def test_text_only_over_bank_v2_keeps_exactly_the_thirty_text_cases():
    """`--text-only` filtered on modality alone, and the three MEDIA-OUT cases
    carry a plain text prompt and the default modality_in, so a text LLM was
    asked to generate a clip, an image and a track and its prose would have been
    counted as three text lanes."""

    import oc_run_frozen_bank as runner

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    rows, selection = runner.select_cases(
        body["prompts"], lanes=None, text_only=True, max_cases=None
    )
    assert len(rows) == 30, sorted(row["id"] for row in rows)
    assert selection["skipped_not_runnable"] == [
        "av-gen-kettle-A",
        "img-gen-shelf-B",
        "music-gen-two-bar-A",
    ]
    assert len(selection["skipped_unsupported_modality"]) == 7
    assert all(row.get("runnable", True) for row in rows)


def test_an_empty_case_selection_is_refused_by_name():
    """`--lanes` naming only vision cases plus `--text-only` leaves nothing.

    Both flags are legal, the pair is legal, and the result was
    `IndexError: list index out of range` from `messages_for(rows[0])` several
    hundred lines later -- a message that names neither flag and reads like a
    bug in the runner.  The refusal has to happen where both reasons are still
    in hand.
    """

    import oc_run_frozen_bank as runner

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    vision = [
        c["id"]
        for c in body["prompts"]
        if str(c.get("modality_in", "text")).startswith("text+image")
    ]
    assert vision, "the bank must still carry vision cases for this to mean anything"
    try:
        runner.select_cases(
            body["prompts"], lanes=",".join(vision), text_only=True, max_cases=None
        )
    except oc_calib.CalibrationError as exc:
        message = str(exc)
    else:  # pragma: no cover - the defect
        raise AssertionError("an empty selection was accepted")
    assert "selection is empty" in message
    assert "--text-only=True" in message
    assert vision[0] in message, "the refusal must name what it dropped"


def test_a_message_with_no_content_parts_is_refused():
    """The one deliberate divergence from the sealed `_load_prompts`, bounded.

    The original requires a non-blank `str`; this port also accepts a LIST, for
    bank v2's typed vision parts.  That acceptance must not extend to an EMPTY
    list, which carries nothing and would render as an empty turn.
    """

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bank.json"
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {
                            "id": "empty-parts",
                            "category": "vision",
                            "messages": [{"role": "user", "content": []}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            oc_calib.load_bank(path)
        except oc_calib.CalibrationError as exc:
            assert "invalid chat messages" in str(exc)
        else:  # pragma: no cover - the defect
            raise AssertionError("a message with no content parts loaded clean")

        # a NON-empty list is still accepted: that is the whole point of the
        # divergence, and refusing it would refuse every bank-v2 vision case.
        path.write_text(
            json.dumps(
                {
                    "prompts": [
                        {
                            "id": "with-parts",
                            "category": "vision",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": "read it"}],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        rows, _payload, _digest = oc_calib.load_bank(path)
        assert [row["id"] for row in rows] == ["with-parts"]


def test_runner_refuses_a_media_out_case_named_directly():
    import oc_run_frozen_bank as runner

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    media = [c for c in body["prompts"] if c["id"] == "av-gen-kettle-A"]
    assert media
    try:
        runner.materialise_rows(media)
    except oc_calib.CalibrationError as exc:
        assert "runnable:false" in str(exc)
        return
    raise AssertionError("prose is not a generated clip and must not be filed as one")


def test_seven_gram_check_covers_the_long_context_cases():
    """The check exempted every long-context case, so three pairs of rungs
    shipped byte-identical prompts inside '0 failures over 40 cases'."""

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    cases = body["prompts"]
    long_context = [c for c in cases if c.get("haystack_spec")]
    assert len(long_context) == 9

    # 1. every rung is compared, and the comparison covers the needles.
    for case in long_context:
        text = bankv2.comparable_prompt_text(case)
        assert "<<HAYSTACK>>" in text
        for needle in case["haystack_spec"]["needles"]:
            assert needle["text"] in text

    # 2. no two rungs ask the same thing any more.
    surfaces = {c["id"]: bankv2.comparable_prompt_text(c) for c in long_context}
    assert len(set(surfaces.values())) == 9, "two rungs carry identical prompt text"

    # 3. and the rule FIRES on a duplicated rung, which is the only proof that
    #    the previous two assertions are being enforced rather than observed.
    clone = dict(long_context[0])
    clone["id"] = "lc-8k-notice-select-A-CLONE"
    failures = bankv2.self_test([long_context[0], clone], v1_pack=None)
    assert any("7-gram" in failure for failure in failures), failures
    assert not bankv2.self_test(cases, v1_pack=FROZEN_V1 if FROZEN_V1.is_file() else None)


def test_long_context_rungs_are_matched_in_shape():
    """Disjoint in content, but still the same KIND of ask at every length, or an
    8k-versus-200k comparison would not be a length effect."""

    with tempfile.TemporaryDirectory() as tmp:
        body = bankv2.build_bank(Path(tmp) / "assets")
    rungs = {
        c["id"]: c
        for c in body["prompts"]
        if c["id"].endswith("notice-select-A") or c["id"].endswith("notice-select-B")
    }
    assert len(rungs) == 6
    for case in rungs.values():
        assert case["ask"] == "SELECT"
        assert len(case["haystack_spec"]["needles"]) == 3
        assert len(case["rubric"]["required_all"]) == 3
    lengths = sorted(
        c["haystack_spec"]["target_tokens_nominal"] for c in rungs.values()
    )
    assert lengths == [8192, 8192, 65536, 65536, 200000, 200000]


def test_a_truncated_response_is_labelled_beside_its_rubric_verdict():
    case = {
        "id": "reason-arith-two-step-A",
        "ask": "PRODUCE",
        "rubric": {"required_all": [r"\b100\b"], "max_words": 400},
    }
    records = [
        _record(
            "reason-arith-two-step-A",
            "13 * 9 = 117, then 117 - 17 = 100, and now I divide",
            termination="censored-at-cap",
        )
    ]
    artifact = oc_assess.assess(
        _dossier(records),
        bank={"reason-arith-two-step-A": case},
        per_case=None,
        sealed_useful_or_partial=None,
    )
    rubric = artifact["lanes"][0]["machine_rubric"]
    assert rubric["passed"] is True
    assert "hit the token cap" in rubric["response_was_truncated"]
    sheet = oc_assess.markdown_sheet(artifact, _dossier(records), None)
    assert "truncated" in sheet


def _all_tests():
    return [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]


def main() -> int:
    failures = 0
    skipped: list[str] = []
    for name, function in _all_tests():
        try:
            function()
        except SkipTest as reason:
            skipped.append(f"{name}: {reason}")
            print(f"SKIP {name}: {reason}")
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
        else:
            print(f"ok   {name}")
    total = len(_all_tests())
    passed = total - failures - len(skipped)
    print(
        f"\n{passed} passed, {len(skipped)} skipped, {failures} failed "
        f"(of {total} collected)"
    )
    for line in skipped:
        print(f"  skipped: {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
