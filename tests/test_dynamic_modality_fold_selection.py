"""Pinned regression proof for the shared modality fold-selection numerics.

``src/neural_forge/dynamic_modality_fold_selection.py`` is now the single copy
of the heldout-blind selector that the NEO vision, native audio, and native
video lanes all delegate to.  Those functions feed sealed content-addressed
fold caches, so an edit that moves their output silently invalidates already
published artifacts in *three* lanes at once.  This file exists so that cannot
happen quietly:

* the fixtures are RNG-free -- every tensor comes from a fixed splitmix64
  stream -- so the pinned digests do not depend on a torch version, a BLAS
  backend, or a platform;
* floating point is compared through a 12-significant-digit canonical form,
  which leaves headroom for last-bit reduction-order noise while still
  catching any real change of clamp, ridge, tie-break, or scoring;
* the structural outputs (chosen neuron, anchor order, target indices) are
  compared exactly, because those are what the seal ultimately encodes;
* every lane's fail-closed message is pinned verbatim, including the one
  message the three lanes deliberately disagree on.

If a wave intends to change the selection policy, these digests must be
updated in the same commit that bumps the affected fold-cache schemas.
"""

from __future__ import annotations

from typing import Any

import pytest

from neural_forge.dynamic_audio_fold import (
    DynamicAudioFoldError,
    NativeAudioLayerFoldSelection,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_modality_fold_selection import (
    ModalityFoldSelectionVocabulary,
    selection_from_statistics,
    sparse_anchor_regression,
    sparse_compensation_terms,
    survivors_for,
)
from neural_forge.dynamic_neo_fold import DynamicNeoFoldError, NeoLayerFoldSelection
from neural_forge.dynamic_video_fold import (
    DynamicVideoFoldError,
    NativeVideoLayerFoldSelection,
)


torch = pytest.importorskip("torch")


_MASK = (1 << 64) - 1

# The lane-specific key each lane adds to its own ``as_dict()``; stripped when
# the three lanes are compared against one another.
_FRONT_END_KEYS = (
    "audio_front_end_copied_exactly",
    "video_front_end_copied_exactly",
)


def _lane_vocabulary(module_name: str) -> ModalityFoldSelectionVocabulary:
    import importlib

    return importlib.import_module(module_name)._SELECTION_VOCABULARY


LANES = (
    ("neo", "neural_forge.dynamic_neo_fold", NeoLayerFoldSelection),
    ("audio", "neural_forge.dynamic_audio_fold", NativeAudioLayerFoldSelection),
    ("video", "neural_forge.dynamic_video_fold", NativeVideoLayerFoldSelection),
)


# --------------------------------------------------------------------------
# Deterministic, RNG-free fixtures
# --------------------------------------------------------------------------


def _stream(seed: int):
    state = seed & _MASK

    def nxt() -> float:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & _MASK
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
        value ^= value >> 31
        # A 53-bit mantissa maps exactly onto a float64 in [-1, 1).
        return (value >> 11) / float(1 << 53) * 2.0 - 1.0

    return nxt


def _matrix(seed: int, rows: int, columns: int) -> list[list[float]]:
    nxt = _stream(seed)
    return [[nxt() for _ in range(columns)] for _ in range(rows)]


def _positive_vector(seed: int, size: int) -> list[float]:
    nxt = _stream(seed)
    return [abs(nxt()) + 0.125 for _ in range(size)]


def _statistics(
    *,
    seed: int,
    rows: int,
    width: int,
    dtype: Any = torch.float64,
    zero_columns: tuple[int, ...] = (),
    duplicate: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    sketch = _matrix(seed, rows, width)
    for source, target in duplicate:
        for row in sketch:
            row[target] = row[source]
    for column in zero_columns:
        for row in sketch:
            row[column] = 0.0
    return {
        "mean_abs": torch.tensor(_positive_vector(seed + 1, width), dtype=dtype),
        "mean_square": torch.tensor(_positive_vector(seed + 2, width), dtype=dtype),
        "sketch": torch.tensor(sketch, dtype=dtype),
        "down_norm": torch.tensor(_positive_vector(seed + 3, width), dtype=dtype),
        "residual_endpoint_mean_square_l2": torch.tensor([1.5], dtype=dtype),
    }


CASES: tuple[dict[str, Any], ...] = (
    dict(name="w6_r4_a1_c3", seed=11, rows=4, width=6, candidate_count=3,
         anchor_count=1),
    dict(name="w6_r4_a2_c3", seed=11, rows=4, width=6, candidate_count=3,
         anchor_count=2),
    dict(name="w6_r4_a3_c5", seed=11, rows=4, width=6, candidate_count=5,
         anchor_count=3),
    dict(name="w6_r4_a9_c5_clamped", seed=11, rows=4, width=6, candidate_count=5,
         anchor_count=9),
    dict(name="w12_r5_a2_c11", seed=23, rows=5, width=12, candidate_count=11,
         anchor_count=2),
    dict(name="w12_r5_a4_c1", seed=23, rows=5, width=12, candidate_count=1,
         anchor_count=4),
    dict(name="w33_r9_a3_c7", seed=41, rows=9, width=33, candidate_count=7,
         anchor_count=3),
    dict(name="w6_r4_a2_c3_float32_input", seed=11, rows=4, width=6,
         candidate_count=3, anchor_count=2, dtype=torch.float32),
    dict(name="w8_r4_a2_c7_zero_columns", seed=57, rows=4, width=8,
         candidate_count=7, anchor_count=2, zero_columns=(2, 5)),
    dict(name="w8_r4_a3_c7_duplicate_columns", seed=57, rows=4, width=8,
         candidate_count=7, anchor_count=3, duplicate=((0, 3), (1, 6))),
    dict(name="w2_r4_a1_c1_minimum_width", seed=71, rows=4, width=2,
         candidate_count=1, anchor_count=1),
)


def _quantize(value: float) -> str:
    """Twelve significant digits: tight enough to catch policy drift, loose
    enough to survive last-bit reduction-order noise in a different BLAS."""

    return f"{float(value):.12e}"


def _case_statistics(case: dict[str, Any]) -> dict[str, Any]:
    return _statistics(
        seed=case["seed"],
        rows=case["rows"],
        width=case["width"],
        dtype=case.get("dtype", torch.float64),
        zero_columns=case.get("zero_columns", ()),
        duplicate=case.get("duplicate", ()),
    )


# --------------------------------------------------------------------------
# Pinned digests
# --------------------------------------------------------------------------

# ``sparse_anchor_regression`` over every removable column of every case.
REGRESSION_DIGEST = "7647bd3f25d7376ffe2643bc871654ed483cbd6893ad1a8c3ac64f88397a7470"

# ``selection_from_statistics`` over every case, for every lane.
SELECTION_DIGEST = "df2e698d60b6c6a2916938a87226b32cee9cd260d5d067cfd3a31bd7644b585a"

# The exact removed neuron and anchor order per case -- the structural part of
# the seal, compared without any floating point tolerance at all.
SELECTION_STRUCTURE: dict[str, dict[str, Any]] = {
    "w6_r4_a1_c3": {
        "removed_neuron": 0,
        "anchor_neurons": [4],
        "anchor_target_indices": [3],
        "source_neurons": 6,
        "candidate_count": 3,
    },
    "w6_r4_a2_c3": {
        "removed_neuron": 0,
        "anchor_neurons": [4, 5],
        "anchor_target_indices": [3, 4],
        "source_neurons": 6,
        "candidate_count": 3,
    },
    "w6_r4_a3_c5": {
        "removed_neuron": 0,
        "anchor_neurons": [4, 5, 1],
        "anchor_target_indices": [3, 4, 0],
        "source_neurons": 6,
        "candidate_count": 5,
    },
    "w6_r4_a9_c5_clamped": {
        "removed_neuron": 0,
        "anchor_neurons": [4, 5, 1, 3],
        "anchor_target_indices": [3, 4, 0, 2],
        "source_neurons": 6,
        "candidate_count": 5,
    },
    "w12_r5_a2_c11": {
        "removed_neuron": 1,
        "anchor_neurons": [5, 0],
        "anchor_target_indices": [4, 0],
        "source_neurons": 12,
        "candidate_count": 11,
    },
    "w12_r5_a4_c1": {
        "removed_neuron": 1,
        "anchor_neurons": [5, 0, 6, 8],
        "anchor_target_indices": [4, 0, 5, 7],
        "source_neurons": 12,
        "candidate_count": 1,
    },
    "w33_r9_a3_c7": {
        "removed_neuron": 12,
        "anchor_neurons": [9, 25, 18],
        "anchor_target_indices": [9, 24, 17],
        "source_neurons": 33,
        "candidate_count": 7,
    },
    "w6_r4_a2_c3_float32_input": {
        "removed_neuron": 0,
        "anchor_neurons": [4, 5],
        "anchor_target_indices": [3, 4],
        "source_neurons": 6,
        "candidate_count": 3,
    },
    "w8_r4_a2_c7_zero_columns": {
        "removed_neuron": 3,
        "anchor_neurons": [0, 1],
        "anchor_target_indices": [0, 1],
        "source_neurons": 8,
        "candidate_count": 7,
    },
    "w8_r4_a3_c7_duplicate_columns": {
        "removed_neuron": 3,
        "anchor_neurons": [0, 2, 7],
        "anchor_target_indices": [0, 2, 6],
        "source_neurons": 8,
        "candidate_count": 7,
    },
    "w2_r4_a1_c1_minimum_width": {
        "removed_neuron": 0,
        "anchor_neurons": [1],
        "anchor_target_indices": [0],
        "source_neurons": 2,
        "candidate_count": 1,
    },
}


def _regression_payload() -> list[Any]:
    vocabulary = _lane_vocabulary("neural_forge.dynamic_neo_fold")
    payload: list[Any] = []
    for case in CASES:
        statistics = _case_statistics(case)
        sketch = statistics["sketch"].to(dtype=torch.float64)
        column_norms = torch.linalg.vector_norm(sketch, dim=0)
        epsilon = torch.finfo(torch.float64).eps
        rows = []
        for removed in range(int(sketch.shape[1])):
            anchors, coefficients, correlation, relative_error = (
                sparse_anchor_regression(
                    sketch,
                    removed=removed,
                    anchor_count=case["anchor_count"],
                    column_norms=column_norms,
                    epsilon=epsilon,
                    vocabulary=vocabulary,
                )
            )
            rows.append(
                {
                    "removed": removed,
                    "anchors": list(anchors),
                    "coefficients": [_quantize(value) for value in coefficients],
                    "correlation": _quantize(correlation),
                    "relative_error": _quantize(relative_error),
                }
            )
        payload.append({"case": case["name"], "rows": rows})
    return payload


def _lane_selection(case: dict[str, Any], module_name: str) -> Any:
    import importlib

    module = importlib.import_module(module_name)
    return module._selection_from_statistics(
        3,
        _case_statistics(case),
        trace_rows=17,
        candidate_count=case["candidate_count"],
        anchor_count=case["anchor_count"],
    )


def _selection_payload() -> list[Any]:
    payload: list[Any] = []
    for case in CASES:
        for lane, module_name, _cls in LANES:
            selection = _lane_selection(case, module_name)
            body = {
                key: (_quantize(value) if isinstance(value, float) else value)
                for key, value in selection.as_dict().items()
            }
            payload.append(
                {
                    "case": case["name"],
                    "lane": lane,
                    "as_dict": body,
                    "compensation_terms": [
                        [term[0], term[1], _quantize(term[2])]
                        for term in selection.compensation_terms
                    ],
                    "survivors": list(selection.survivors),
                }
            )
    return payload


def test_sparse_anchor_regression_output_digest_is_pinned() -> None:
    assert canonical_sha256(_regression_payload()) == REGRESSION_DIGEST


def test_selection_from_statistics_output_digest_is_pinned() -> None:
    assert canonical_sha256(_selection_payload()) == SELECTION_DIGEST


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_selection_structure_is_pinned_without_float_tolerance(
    case: dict[str, Any],
) -> None:
    expected = SELECTION_STRUCTURE[case["name"]]
    selection = _lane_selection(case, "neural_forge.dynamic_neo_fold")

    assert selection.removed_neuron == expected["removed_neuron"]
    assert list(selection.anchor_neurons) == expected["anchor_neurons"]
    assert list(selection.anchor_target_indices) == expected["anchor_target_indices"]
    assert selection.source_neurons == expected["source_neurons"]
    assert selection.target_neurons == expected["source_neurons"] - 1
    assert selection.candidate_count == expected["candidate_count"]


# --------------------------------------------------------------------------
# One module, three lanes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_every_lane_selects_identically_from_identical_statistics(
    case: dict[str, Any],
) -> None:
    """The point of the extraction: identical evidence, identical selection."""

    bodies = {}
    for lane, module_name, _cls in LANES:
        selection = _lane_selection(case, module_name)
        body = dict(selection.as_dict())
        for key in _FRONT_END_KEYS:
            body.pop(key, None)
        bodies[lane] = body

    assert bodies["audio"] == bodies["neo"]
    assert bodies["video"] == bodies["neo"]


def test_survivors_and_compensation_terms_are_shared_helpers() -> None:
    for _lane, module_name, _cls in LANES:
        selection = _lane_selection(CASES[2], module_name)
        assert selection.survivors == survivors_for(
            selection.source_neurons,
            selection.removed_neuron,
        )
        assert selection.compensation_terms == sparse_compensation_terms(
            selection.anchor_neurons,
            selection.anchor_target_indices,
            selection.compensation_coefficients,
        )
        assert selection.anchor_count == len(selection.anchor_neurons)


def test_each_lane_keeps_its_own_error_type_and_selection_dataclass() -> None:
    expected = {
        "neo": (DynamicNeoFoldError, NeoLayerFoldSelection),
        "audio": (DynamicAudioFoldError, NativeAudioLayerFoldSelection),
        "video": (DynamicVideoFoldError, NativeVideoLayerFoldSelection),
    }
    for lane, module_name, cls in LANES:
        vocabulary = _lane_vocabulary(module_name)
        assert vocabulary.error_type is expected[lane][0]
        assert cls is expected[lane][1]
        assert isinstance(_lane_selection(CASES[1], module_name), cls)


# --------------------------------------------------------------------------
# Fail-closed messages, pinned verbatim per lane
# --------------------------------------------------------------------------


MALFORMED_STATISTICS = {
    "neo": "NEO layer 7 trace statistics are malformed.",
    "audio": "Native audio layer 7 trace statistics are malformed.",
    "video": "Native video layer 7 trace statistics are malformed.",
}
NARROW_TRACE = {
    "neo": "NEO trace width must exceed one neuron.",
    "audio": "Native audio trace width must exceed one neuron.",
    "video": "Native video trace width must exceed one neuron.",
}
TORCH_REQUIRED = {
    "neo": "PyTorch is required for NEO selection.",
    "audio": "PyTorch is required for native audio selection.",
    "video": "PyTorch is required for native video selection.",
}


def test_selection_rejects_malformed_statistics_with_the_lane_message() -> None:
    malformed = {
        "mean_abs": torch.zeros(4, 2, dtype=torch.float64),
        "mean_square": torch.zeros(4, 2, dtype=torch.float64),
        "sketch": torch.zeros(4, 2, dtype=torch.float64),
        "down_norm": torch.zeros(4, 2, dtype=torch.float64),
        "residual_endpoint_mean_square_l2": torch.tensor([1.0], dtype=torch.float64),
    }
    for lane, module_name, _cls in LANES:
        import importlib

        module = importlib.import_module(module_name)
        with pytest.raises(module._SELECTION_VOCABULARY.error_type) as raised:
            module._selection_from_statistics(
                7,
                malformed,
                trace_rows=5,
                candidate_count=1,
                anchor_count=1,
            )
        assert str(raised.value) == MALFORMED_STATISTICS[lane]


def test_selection_rejects_a_single_neuron_trace_with_the_lane_message() -> None:
    narrow = {
        "mean_abs": torch.ones(1, dtype=torch.float64),
        "mean_square": torch.ones(1, dtype=torch.float64),
        "sketch": torch.ones(3, 1, dtype=torch.float64),
        "down_norm": torch.ones(1, dtype=torch.float64),
        "residual_endpoint_mean_square_l2": torch.tensor([1.0], dtype=torch.float64),
    }
    for lane, module_name, _cls in LANES:
        import importlib

        module = importlib.import_module(module_name)
        with pytest.raises(module._SELECTION_VOCABULARY.error_type) as raised:
            module._selection_from_statistics(
                7,
                narrow,
                trace_rows=5,
                candidate_count=1,
                anchor_count=1,
            )
        assert str(raised.value) == NARROW_TRACE[lane]


def test_missing_torch_message_is_reconstructed_per_lane() -> None:
    """The import guard is unreachable while torch is installed, so pin the
    text the vocabulary would produce instead of faking an ImportError."""

    for lane, module_name, _cls in LANES:
        vocabulary = _lane_vocabulary(module_name)
        assert (
            f"PyTorch is required for {vocabulary.selection_label} selection."
            == TORCH_REQUIRED[lane]
        )


VALIDATION_MESSAGES: dict[str, dict[str, str]] = {
    "neo": {
        "not_one_removed": "NEO selection must remove exactly one source neuron.",
        "lengths_differ": "NEO sparse anchor and compensation lengths differ.",
        "repeated_anchor": "NEO sparse anchor neurons repeat.",
        "anchor_out_of_range": "NEO sparse anchor neuron is out of range.",
        "inconsistent_target": "NEO sparse anchor target index is inconsistent.",
        "primary_mismatch": (
            "NEO primary anchor fields differ from sparse compensation."
        ),
        # The NEO lane's ``_finite`` deliberately disagrees with the other two.
        "non_real_beta": "compensation_beta must be finite.",
    },
    "audio": {
        "not_one_removed": (
            "Native audio selection must remove exactly one source neuron."
        ),
        "lengths_differ": (
            "Native audio sparse anchor and compensation lengths differ."
        ),
        "repeated_anchor": "Native audio sparse anchor neurons repeat.",
        "anchor_out_of_range": "Native audio sparse anchor neuron is out of range.",
        "inconsistent_target": (
            "Native audio sparse anchor target index is inconsistent."
        ),
        "primary_mismatch": (
            "Native audio primary anchor fields differ from sparse compensation."
        ),
        "non_real_beta": "compensation_beta must be a real number.",
    },
    "video": {
        "not_one_removed": (
            "Native video selection must remove exactly one source neuron."
        ),
        "lengths_differ": (
            "Native video sparse anchor and compensation lengths differ."
        ),
        "repeated_anchor": "Native video sparse anchor neurons repeat.",
        "anchor_out_of_range": "Native video sparse anchor neuron is out of range.",
        "inconsistent_target": (
            "Native video sparse anchor target index is inconsistent."
        ),
        "primary_mismatch": (
            "Native video primary anchor fields differ from sparse compensation."
        ),
        "non_real_beta": "compensation_beta must be a real number.",
    },
}

SHARED_VALIDATION_MESSAGES = {
    "anchor_neuron_out_of_range": "anchor_neuron is out of range.",
    "removed_equals_anchor": "Removed and anchor neurons must differ.",
    "target_index_out_of_range": "anchor_target_index is out of range.",
    "negative_layer_index": "layer_index must be a non-negative integer.",
    "zero_trace_rows": "trace_rows must be a positive integer.",
    "non_finite_beta": "compensation_beta must be finite.",
}

_BASE_SELECTION = dict(
    layer_index=0,
    source_neurons=4,
    target_neurons=3,
    removed_neuron=1,
    anchor_neuron=0,
    anchor_target_index=0,
    compensation_beta=0.5,
    activation_correlation=0.25,
    relative_reconstruction_error=0.125,
    residual_endpoint_relative_rmse=0.0625,
    removed_importance=0.03125,
    trace_rows=9,
    candidate_count=2,
)

_MUTATIONS = {
    "not_one_removed": dict(target_neurons=2),
    "lengths_differ": dict(
        anchor_neurons=(0, 2),
        anchor_target_indices=(0,),
        compensation_coefficients=(0.5, 0.25),
    ),
    "repeated_anchor": dict(
        anchor_neurons=(0, 0),
        anchor_target_indices=(0, 0),
        compensation_coefficients=(0.5, 0.25),
    ),
    "anchor_out_of_range": dict(
        anchor_neurons=(0, 99),
        anchor_target_indices=(0, 1),
        compensation_coefficients=(0.5, 0.25),
    ),
    "inconsistent_target": dict(
        anchor_neurons=(0, 2),
        anchor_target_indices=(0, 2),
        compensation_coefficients=(0.5, 0.25),
    ),
    "primary_mismatch": dict(
        anchor_neurons=(2, 0),
        anchor_target_indices=(1, 0),
        compensation_coefficients=(0.25, 0.5),
    ),
    "non_real_beta": dict(compensation_beta="0.5"),
}

_SHARED_MUTATIONS = {
    "anchor_neuron_out_of_range": dict(anchor_neuron=9),
    "removed_equals_anchor": dict(anchor_neuron=1),
    "target_index_out_of_range": dict(anchor_target_index=9),
    "negative_layer_index": dict(layer_index=-1),
    "zero_trace_rows": dict(trace_rows=0),
    "non_finite_beta": dict(compensation_beta=float("inf")),
}


@pytest.mark.parametrize("label", sorted(_MUTATIONS))
def test_sparse_anchor_normalization_messages_stay_lane_specific(label: str) -> None:
    for lane, _module_name, cls in LANES:
        payload = dict(_BASE_SELECTION)
        payload.update(_MUTATIONS[label])
        with pytest.raises(Exception) as raised:  # noqa: PT011 -- per-lane type
            cls(**payload)
        assert str(raised.value) == VALIDATION_MESSAGES[lane][label]


@pytest.mark.parametrize("label", sorted(_SHARED_MUTATIONS))
def test_sparse_anchor_normalization_shares_its_neutral_messages(label: str) -> None:
    for _lane, _module_name, cls in LANES:
        payload = dict(_BASE_SELECTION)
        payload.update(_SHARED_MUTATIONS[label])
        with pytest.raises(Exception) as raised:  # noqa: PT011 -- per-lane type
            cls(**payload)
        assert str(raised.value) == SHARED_VALIDATION_MESSAGES[label]


def test_shared_selection_factory_accepts_any_lane_dataclass() -> None:
    """``selection_from_statistics`` is genuinely modality-neutral: swapping
    the factory swaps the produced type and nothing else."""

    statistics = _case_statistics(CASES[2])
    neo_vocabulary = _lane_vocabulary("neural_forge.dynamic_neo_fold")
    produced = {
        cls: selection_from_statistics(
            3,
            _case_statistics(CASES[2]),
            trace_rows=17,
            candidate_count=CASES[2]["candidate_count"],
            anchor_count=CASES[2]["anchor_count"],
            vocabulary=neo_vocabulary,
            selection_factory=cls,
        )
        for _lane, _module_name, cls in LANES
    }
    assert statistics is not None
    reference = produced[NeoLayerFoldSelection]
    for cls, selection in produced.items():
        assert isinstance(selection, cls)
        assert selection.removed_neuron == reference.removed_neuron
        assert selection.anchor_neurons == reference.anchor_neurons
        assert selection.compensation_coefficients == (
            reference.compensation_coefficients
        )
