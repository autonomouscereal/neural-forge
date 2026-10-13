"""Stage preparation proves real source pins; fixtures do not claim recovery."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chimera_stage_jobs", ROOT / "experiments/chimera_recovery/stage_jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
WORKSPACE = Path("C:/Users/cereal/Documents/ai training")
ENGINE_DIR = Path(os.environ.get("CHIMERA_STAGE_ENGINE_DIR", str(WORKSPACE / ".agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f/experiments/chimera_recovery")))
PREREG = Path(os.environ.get("CHIMERA_STAGE_PREREGISTRATION", str(WORKSPACE / "reviews/2026-10-13-folding/recovery-preparation/preregistration-v1/PREREGISTRATION_OMNI_STAGE_D_v1.md")))


class ContractTests(unittest.TestCase):
    def test_explicit_new_stage_settings_match_distinct_engine_contracts(self):
        self.assertEqual(set(jobs.OPTIMIZERS["D3a"]), {"lr", "betas", "eps", "weight_decay", "clip"})
        self.assertNotIn("warmup_steps", jobs.OPTIMIZERS["D3a"])
        self.assertIn("band_lr", jobs.OPTIMIZERS["D3c"])
        self.assertIn("bridge_lr", jobs.OPTIMIZERS["D3d"])
        self.assertNotIn("emission_lr", jobs.OPTIMIZERS["D3d"])
        self.assertIn(b"independently sealed host-write", jobs.amendment("D3a"))
        self.assertIn(b"not approved historical defaults", jobs.amendment("D3c"))
        self.assertIn(b"weight0.1", jobs.amendment("D3d"))

    def test_templates_never_invent_bindings_or_budget(self):
        value = jobs.template("D3a", jobs.UNITS[0], {"path": "test-doc", "sha256": "a" * 64}, "float32")
        for key in ("steps", "batches_manifest", "immutable_base_receipt", "runtime_assignment", "omni"):
            self.assertIsNone(value[key])
        self.assertEqual(value["objective"], "teacher_delta_host_write")
        self.assertEqual(set(value["refit_weights"]), {"chart_in", "block", "chart_out"})
        value["optimizer"]["betas"][0] = .8
        self.assertEqual(jobs.OPTIMIZERS["D3a"]["betas"][0], .9)

    def test_preregistration_refuses_a_made_up_frozen_body(self):
        with self.assertRaisesRegex(jobs.Refusal, "body differs"):
            jobs.preregistration_parts(b"## 1. Scope\nfiction\n## Amendments\nguided_velocity 2026-10-13\n")


class AuthenticatedSourcePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PREREG.is_file() or not (ENGINE_DIR / "train_stages.py").is_file():
            raise unittest.SkipTest("authentic current engines and preregistration are not mounted; use environment paths")
        jobs.engine_inventory(ENGINE_DIR)
        jobs.preregistration_parts(PREREG.read_bytes())

    def setUp(self):
        test_root = Path(os.environ.get("CHIMERA_STAGE_TEST_TMPROOT", str(ROOT / "proof_outputs")))
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="stage-jobs-test-", dir=test_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "prepared"
        self.prereg_sha = jobs.sha(PREREG)
        self.result = jobs.prepare(ENGINE_DIR, PREREG, self.out, dtype="float32", python="fixture-python")
        self.plan_path = self.out / "manifest.json"
        self.plan_sha = jobs.sha(self.plan_path)
        self.plan = jobs.read_json(self.plan_path)

    def test_all_seven_jobs_have_same_frozen_body_and_preserved_d3b_tail(self):
        self.assertEqual(self.result["jobs"], 7)
        _, source_body, source_tail = jobs.preregistration_parts(PREREG.read_bytes())
        self.assertEqual(jobs.sha(PREREG), self.prereg_sha)
        amendment_shas = set()
        for job in self.plan["jobs"]:
            path = Path(job["preregistration"]["path"])
            raw, body, tail = jobs.preregistration_parts(path.read_bytes())
            self.assertNotIn(b"\r", raw)
            self.assertEqual(body, source_body)
            self.assertTrue(tail.startswith(source_tail))
            self.assertEqual(hashlib.sha256(tail).hexdigest(), job["preregistration"]["amendment_sha256"])
            amendment_shas.add(hashlib.sha256(tail).hexdigest())
            self.assertFalse(job["native_train_ready"])
        self.assertEqual(len(amendment_shas), 3)
        self.assertFalse(self.plan["existing_d3b_seals_changed"])

    def test_preparation_preserves_existing_outputs_and_plan_changes_are_refused(self):
        with self.assertRaisesRegex(jobs.Refusal, "new preparation"):
            jobs.prepare(ENGINE_DIR, PREREG, self.out, dtype="float32")
        self.plan_path.write_bytes(self.plan_path.read_bytes() + b" ")
        with self.assertRaisesRegex(jobs.Refusal, "manifest changed"):
            jobs.verify_plan(self.plan_path, self.plan_sha)

    def test_current_engine_pin_is_rechecked_after_preparation(self):
        fixture_source = self.root / "engine.py"
        fixture_source.write_bytes(b"# fixture source version1\n")
        altered = copy.deepcopy(self.plan)
        altered["engine_sources"]["fixture"] = jobs.pin(fixture_source)
        path = self.root / "fixture-plan.json"
        path.write_bytes(jobs.canonical(altered))
        actual = jobs.sha(path)
        fixture_source.write_bytes(b"# fixture source version2\n")
        with self.assertRaisesRegex(jobs.Refusal, "changed"):
            jobs.verify_plan(path, actual)

    def fixture_bindings(self, stage="D3a"):
        # Explicit synthetic contract fixture, never a production tensor seal.
        job = next(job for job in self.plan["jobs"] if job["stage"] == stage)
        config = jobs.read_json(job["template"]["path"])
        batch_dir = self.root / "synthetic-contract"
        batch_dir.mkdir()
        clone = batch_dir / "fixture-clone"
        clone.mkdir()
        def write(name, value):
            path = batch_dir / name
            path.write_bytes(jobs.canonical(value) + b"\n")
            return path
        tensor = batch_dir / "not-real-tensors.fixture"
        tensor.write_bytes(b"synthetic test bytes, not safetensors")
        base = write("base.json", {"immutable_clone": True, "registered": True, "verified_equal_to_registered_parent": True,
                                    "base_sha256": "b" * 64, "clone_sha256": "b" * 64, "clone_path": str(clone)})
        parity = write("parity.json", {"verdict": "PASS", "base_sha256": "b" * 64})
        runtime = write("runtime.json", {"active": True, "device": "cpu"})
        parent = write("parent.json", {"synthetic_contract_test": True})
        module = batch_dir / "hybrid.fixture.py"
        module.write_bytes(b"# not imported, no model execution\n")
        lane = jobs.STAGES[stage][0]
        receipt = {"schema": f"chimera_recovery.{lane}_cache.v1", "sealed": True, "all_finite": True,
                   "split": "train", "held": False, "tensors_sha256": jobs.sha(tensor), "corpus_sha256": "c" * 64,
                   "amendment_sha256": config["preregistration"]["amendment_sha256"], "recovery_group_id": "fixture-family-train",
                   "D2_child_sha256": "b" * 64, "recaptured_host_cache_sha256": "d" * 64,
                   "teacher_cache_sha256": "e" * 64, "host_write_target_sha256": "f" * 64,
                   "chart_out_objective": "teacher_delta_host_write", "host_write_target_independent": True}
        receipt_path = write("teacher.json", receipt)
        data = {"schema": f"chimera_recovery.{lane}_batches.v1", "stage": stage, "sealed": True,
                "complete_source_traversal": True, "split": "train", "heldout_groups_excluded": True,
                "amendment_sha256": config["preregistration"]["amendment_sha256"], "corpus_sha256": "c" * 64,
                "source_manifest_sha256": "e" * 64, "D2_child_sha256": "b" * 64,
                "recaptured_host_cache_sha256": "d" * 64, "host_write_target_sha256": "f" * 64,
                "batches": [{"id": "fixture-batch1", "recovery_group_id": "fixture-family-train",
                             "tensors": jobs.pin(tensor), "teacher_receipt": jobs.pin(receipt_path)}]}
        data_path = write("batches.json", data)
        bindings = {"hybrid_module": str(module), "immutable_base_receipt": str(base), "parity_receipt": str(parity),
                    "runtime_assignment": str(runtime), "batches_manifest": str(data_path), "omni": str(clone),
                    "parent_config": str(parent), "device": "cpu", "kernels": "fixture-none",
                    "trunk_ffn_width": 16, "python": "fixture-python"}
        return job["id"], write("bindings.json", bindings), data_path, data, receipt_path, receipt

    def test_bound_actual_file_references_still_require_native_preflight(self):
        ident, bindings, _, _, _, _ = self.fixture_bindings()
        out = self.root / "bound.json"
        result = jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, out)
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["training_launched"])
        self.assertFalse(result["tensor_targets_loaded"])
        config = jobs.read_json(out)
        self.assertEqual(config["steps"], 1)
        self.assertEqual(result["preflight_argv"][2], "preflight")
        self.assertEqual(config["sampling_policy"], "one_pass")
        with self.assertRaisesRegex(jobs.Refusal, "new materialized"):
            jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, out)

    def test_partial_old_amendment_holdout_and_duplicate_batches_fail_closed(self):
        ident, bindings, path, original, _, _ = self.fixture_bindings()
        variants = []
        for key, value in (("complete_source_traversal", False), ("heldout_groups_excluded", False),
                           ("split", "test"), ("amendment_sha256", "a" * 64)):
            data = copy.deepcopy(original)
            data[key] = value
            variants.append(data)
        data = copy.deepcopy(original)
        data["batches"].append(copy.deepcopy(data["batches"][0]))
        variants.append(data)
        for index, data in enumerate(variants):
            with self.subTest(index=index):
                path.write_bytes(jobs.canonical(data))
                with self.assertRaises(jobs.Refusal):
                    jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, self.root / f"refused-{index}.json")

    def test_d3a_rejects_self_targets_and_other_child(self):
        ident, bindings, data_path, data, receipt_path, original = self.fixture_bindings()
        for field, value in (("host_write_target_independent", False), ("D2_child_sha256", "a" * 64)):
            with self.subTest(field=field):
                receipt = {**original, field: value}
                receipt_path.write_bytes(jobs.canonical(receipt))
                data["batches"][0]["teacher_receipt"] = jobs.pin(receipt_path)
                data_path.write_bytes(jobs.canonical(data))
                with self.assertRaisesRegex(jobs.Refusal, "independent outward"):
                    jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, self.root / (field + ".json"))

    def test_d3a_manifest_target_identity_is_required_before_model_loading(self):
        ident, bindings, data_path, data, _, _ = self.fixture_bindings()
        data.pop("host_write_target_sha256")
        data_path.write_bytes(jobs.canonical(data))
        with self.assertRaisesRegex(jobs.Refusal, "actual lowercase SHA256"):
            jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, self.root / "missing-refit-identity.json")

    def test_d3d_refuses_feature_only_cache(self):
        ident, bindings, _, _, _, _ = self.fixture_bindings("D3d")
        with self.assertRaisesRegex(jobs.Refusal, "feature-only"):
            jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, self.root / "not-vision-ready.json")

    def test_d3c_refuses_unshifted_or_unconditional_targets(self):
        ident, bindings, _, _, _, _ = self.fixture_bindings("D3c")
        with self.assertRaisesRegex(jobs.Refusal, "shifted conditional"):
            jobs.materialize(self.plan_path, self.plan_sha, ident, bindings, self.root / "not-music-ready.json")


if __name__ == "__main__":
    unittest.main()
