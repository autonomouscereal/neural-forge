"""Regression checks on private future recipe candidates; no folding/weights."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("folding_recipe_candidates", ROOT / "experiments/chimera_recovery/folding_recipe_candidates.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
SOURCE_ROOT = Path(os.environ.get("CHIMERA_H3_CAMPAIGN_SOURCE", "C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09")) / "composite/tools"


def source(kind):
    path = SOURCE_ROOT / ("receipt_q27_streamed.py" if kind == "receipt" else "s7_unit_build.py")
    if not path.exists():
        raise unittest.SkipTest("authentic 2026 campaign mirror not mounted")
    return candidate.patch_source(kind, path)[0]


def functions(text, names, namespace):
    parsed = ast.parse(text)
    selected = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(selected) != len(names):
        raise AssertionError("candidate function inventory differs")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "private candidate functions", "exec"), namespace)
    return namespace


class CandidateTests(unittest.TestCase):
    def test_pinned_originals_and_candidate_source_are_distinct(self):
        for kind in ("receipt", "unit"):
            path = SOURCE_ROOT / ("receipt_q27_streamed.py" if kind == "receipt" else "s7_unit_build.py")
            if not path.exists():
                self.skipTest("authentic campaign mirror not mounted")
            after, receipt = candidate.patch_source(kind, path)
            self.assertNotEqual(receipt["candidate_sha256"], receipt["baseline_sha256"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt["baseline_sha256"])
            self.assertFalse(receipt["fold_launched"])
            self.assertFalse(receipt["weights_changed"])
            ast.parse(after)

    def test_hy3d_parent_option_reaches_only_time_reference_contract(self):
        text = source("unit")
        calls = []
        ns = {"argparse": argparse, "Path": Path, "os": os, "json": json,
              "__doc__": "private CLI regression fixture", "Any": Any,
              "C16": types.SimpleNamespace(tree_identity=lambda _: None, Refusal=ValueError),
              "build_hy3d": lambda *a, **kw: calls.append((a, kw)) or {"checked": True}}
        functions(text, {"main"}, ns)
        with patch.dict(os.environ, {"CHIMERA_OMNI_SRC": "/test/campaign-source"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ns["main"](["hy3d", "--src", "/test/block-parent", "--dense-child", "/test/child",
                                        "--out", "/test/new-output", "--parent", "/test/time-parent"]), 0)
            self.assertEqual(ns["main"](["hy3d", "--src", "/test/block-parent", "--dense-child", "/test/child",
                                        "--out", "/test/new-output-2"]), 0)
        self.assertEqual(calls[0][0][0], Path("/test/block-parent"))
        self.assertEqual(calls[0][1]["time_reference_parent"], Path("/test/time-parent"))
        self.assertIsNone(calls[1][1]["time_reference_parent"])
        self.assertIn('"reference_source_weights_sha256": sha256_file(time_parent)', text)
        self.assertIn('hy3d_reference_vs_capture(time_parent, block0_capture)', text)


class TorchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("supported CPU Torch runtime required")
        cls.torch = torch
        cls.ns = functions(source("receipt"), {"compare_prompts", "validate_prompt_measurement"},
                           {"Any": Any, "Refusal": candidate.Refusal, "hashlib": hashlib,
                            "json": json, "log": lambda _: None})

    def fixture(self):
        t = self.torch
        reference = t.tensor([[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]])
        produced = t.cat([reference.clone(), t.full((1, 2, 2), float("-inf"))], dim=-1)
        taps = {0: t.tensor([[[0.1, 0.2], [0.2, 0.3]]]), 1: t.ones(1, 2, 2)}
        return {"packs": [[7, 9]], "reference_logits": [reference], "produced_logits": [produced],
                "reference_hidden": [taps], "produced_hidden": [{k: v.clone() for k, v in taps.items()}],
                "base_vocab": 3, "block_kinds": ["full_attention", "linear_attention"], "layers_reported": 2}

    def test_valid_complete_parity_and_exact_mask(self):
        prompts, summary = self.ns["compare_prompts"](**self.fixture())
        self.assertEqual(summary["max_abs_logit_diff"], 0.0)
        self.assertEqual(summary["max_abs_hidden_diff_first_layers"], 0.0)
        self.assertEqual(len(prompts[0]["per_layer_hidden_max_abs_diff"]), 2)

    def test_missing_requested_hidden_taps_are_refused(self):
        f = self.fixture()
        f["produced_hidden"][0].pop(1)
        with self.assertRaisesRegex(candidate.Refusal, "coverage"):
            self.ns["compare_prompts"](**f)

    def test_nan_inf_base_logits_and_hidden_taps_refused(self):
        for field in ("reference_logits", "produced_logits"):
            for value in (float("nan"), float("inf"), float("-inf")):
                f = self.fixture()
                f[field][0][0, 0, 0] = value
                with self.assertRaisesRegex(candidate.Refusal, "non-finite"):
                    self.ns["compare_prompts"](**f)
        for field in ("reference_hidden", "produced_hidden"):
            f = self.fixture()
            f[field][0][0][0, 0, 0] = float("nan")
            with self.assertRaisesRegex(candidate.Refusal, "non-finite"):
                self.ns["compare_prompts"](**f)

    def test_interface_band_nan_positive_inf_and_finite_are_refused(self):
        for value in (float("nan"), float("inf"), 0.0):
            f = self.fixture()
            f["produced_logits"][0][0, 0, 3] = value
            with self.assertRaisesRegex(candidate.Refusal, "mask"):
                self.ns["compare_prompts"](**f)

    def test_missing_prompt_or_broadcast_hidden_geometry_refused(self):
        f = self.fixture()
        f["produced_logits"] = []
        with self.assertRaisesRegex(candidate.Refusal, "count"):
            self.ns["compare_prompts"](**f)
        f = self.fixture()
        f["produced_hidden"][0][0] = self.torch.ones(1, 1, 2)
        with self.assertRaisesRegex(candidate.Refusal, "geometry"):
            self.ns["compare_prompts"](**f)
        f = self.fixture()
        f["reference_hidden"][0][0] = self.torch.ones(1, 2)
        f["produced_hidden"][0][0] = self.torch.ones(1, 2)
        with self.assertRaisesRegex(candidate.Refusal, "geometry"):
            self.ns["compare_prompts"](**f)


if __name__ == "__main__":
    unittest.main()
