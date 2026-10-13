"""Exercise actual D3a admission before any checkpoint/model can be opened."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_job_identity_test_helpers", ROOT / "experiments/chimera_recovery/stage_jobs.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


class AdmissionIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "experiments/chimera_recovery/train_stages.py").read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "stage_data")
        cls.code = compile(ast.Module(body=[function], type_ignores=[]), "actual train_stages.stage_data", "exec")

    def setUp(self):
        test_root = Path(os.environ.get("CHIMERA_STAGE_TEST_TMPROOT", str(ROOT / "proof_outputs")))
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="refit-admission-test-", dir=test_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files_checked = []
        def pinned(entry, root):
            self.files_checked.append(entry["path"])
            return J.check_pin(entry, root)
        ns = {"T": types.SimpleNamespace(read_json=J.read_json, pinned_file=pinned),
              "R": types.SimpleNamespace(require_sha=lambda value, label: J.require_sha(value)),
              "Error": J.Refusal, "STAGES": J.STAGES, "Path": Path}
        exec(self.code, ns)
        self.admit = ns["stage_data"]
        # Explicit contract-test bytes, not a captured teacher or real tensors.
        tensor = self.root / "synthetic-not-tensors.fixture"
        tensor.write_bytes(b"contract test only; never passed to a model")
        self.receipt_path = self.root / "teacher.fixture.json"
        self.receipt = {"schema": "chimera_recovery.refit_cache.v1", "sealed": True, "all_finite": True,
                        "split": "train", "held": False, "tensors_sha256": J.sha(tensor), "corpus_sha256": "a" * 64,
                        "amendment_sha256": "b" * 64, "recovery_group_id": "contract-fixture-group",
                        "D2_child_sha256": "c" * 64, "recaptured_host_cache_sha256": "d" * 64,
                        "teacher_cache_sha256": "e" * 64, "host_write_target_sha256": "f" * 64,
                        "chart_out_objective": "teacher_delta_host_write", "host_write_target_independent": True}
        self.receipt_path.write_bytes(J.canonical(self.receipt))
        self.manifest_path = self.root / "manifest.fixture.json"
        self.data = {"schema": "chimera_recovery.refit_batches.v1", "stage": "D3a", "sealed": True,
                     "complete_source_traversal": True, "split": "train", "heldout_groups_excluded": True,
                     "amendment_sha256": "b" * 64, "corpus_sha256": "a" * 64, "source_manifest_sha256": "e" * 64,
                     "D2_child_sha256": "c" * 64, "recaptured_host_cache_sha256": "d" * 64,
                     "host_write_target_sha256": "f" * 64,
                     "batches": [{"id": "contract-fixture-batch", "recovery_group_id": "contract-fixture-group",
                                  "tensors": J.pin(tensor), "teacher_receipt": J.pin(self.receipt_path)}]}

    def admit_data(self, value):
        self.manifest_path.write_bytes(J.canonical(value))
        return self.admit("D3a", self.manifest_path, "b" * 64)

    def test_matching_manifest_and_sidecar_engine_identity_is_admitted(self):
        data, summary = self.admit_data(self.data)
        self.assertEqual(data["D2_child_sha256"], self.receipt["D2_child_sha256"])
        self.assertEqual(summary, {"admitted_groups": 1, "accepted_batches": 1})

    def test_missing_manifest_identity_refused_before_any_tensor_file_check(self):
        for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256"):
            with self.subTest(key=key):
                value = copy.deepcopy(self.data)
                value.pop(key)
                self.files_checked.clear()
                with self.assertRaises(J.Refusal):
                    self.admit_data(value)
                self.assertEqual(self.files_checked, [])

    def test_malformed_manifest_identity_refused_before_any_tensor_file_check(self):
        value = {**self.data, "host_write_target_sha256": "not-a-hash"}
        with self.assertRaises(J.Refusal):
            self.admit_data(value)
        self.assertEqual(self.files_checked, [])

    def test_each_sidecar_identity_must_match_the_manifest(self):
        for key in ("D2_child_sha256", "recaptured_host_cache_sha256", "host_write_target_sha256", "teacher_cache_sha256"):
            with self.subTest(key=key):
                receipt = {**self.receipt, key: "0" * 64}
                self.receipt_path.write_bytes(J.canonical(receipt))
                value = copy.deepcopy(self.data)
                value["batches"][0]["teacher_receipt"] = J.pin(self.receipt_path)
                with self.assertRaisesRegex(J.Refusal, "manifest engine identity"):
                    self.admit_data(value)

    def test_non_independent_outward_targets_are_refused(self):
        receipt = {**self.receipt, "host_write_target_independent": False}
        self.receipt_path.write_bytes(J.canonical(receipt))
        value = copy.deepcopy(self.data)
        value["batches"][0]["teacher_receipt"] = J.pin(self.receipt_path)
        with self.assertRaisesRegex(J.Refusal, "independent sealed write"):
            self.admit_data(value)


if __name__ == "__main__":
    unittest.main()
