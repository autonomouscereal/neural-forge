"""Synthetic metadata/file-IO and extracted real loader tests; no tensor bodies."""
from __future__ import annotations
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import c10_native_candidate_binding as binding
import c17_source_binding as source_binding

ROOT = Path(__file__).resolve().parent


class FakeTensor:
    def __init__(self, shape, *, dtype="F32", finite=True, nonzero=False):
        self.shape, self.dtype, self.finite, self.nonzero = tuple(shape), dtype, finite, nonzero
    def float(self):
        return self
    @property
    def T(self):
        return FakeTensor(self.shape[::-1], dtype=self.dtype, finite=self.finite, nonzero=self.nonzero)
    def contiguous(self):
        return self


class FakeTorch:
    float32 = "F32"
    @staticmethod
    def zeros(n):
        return FakeTensor([n])
    @staticmethod
    def isfinite(tensor):
        return SimpleNamespace(all=lambda: tensor.finite)
    @staticmethod
    def count_nonzero(tensor):
        return int(tensor.nonzero)


@dataclass
class FakeChart:
    trunk_layer: int
    path: str
    sha256: str
    arm: str
    trunk_to_k3: object
    bias_to_k3: object
    k3_to_trunk: object
    bias_to_trunk: object
    layer_json_sha256: str | None = None
    fit_trunk_rows_key: str | None = None
    native_boundary_provenance: object = None
    native_candidate_binding: object = None


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.charts = self.folder / "charts"
        self.layer = self.charts / "L03"
        self.layer.mkdir(parents=True)
        self.chart = self.layer / "ridge.safetensors"
        self.chart.write_bytes(b"synthetic sentinel, deliberately not a tensor payload")
        self.expected = {"host_layer": 3, "host_index_sha256": "1" * 64,
                         "host_config_sha256": "2" * 64, "evidence_sha256": "3" * 64,
                         "source_layers": [5, 6], "installed_ffn_sha256": {"5": "5" * 64, "6": "6" * 64}}
        imported = {"schema": "chimera_omni.import.qwen35.v1", "output_gate_waiver": None,
                    "checkpoint": {"config_sha256": "2" * 64, "index_sha256": "1" * 64,
                                   "directory_shard_files_match_index": True}}
        self.imported = self.save("host_import.json", imported)
        host_rows = self.folder / "host_rows.safetensors"
        host_rows.write_bytes(b"synthetic host row sentinel, no tensor body")
        self.host = self.save("host_capture.json", {"synthetic_test_metadata": True,
            "schema": "neural-forge.c17-host-boundaries.v1", "gradients_or_training": False,
            "fixture": False, "host_receipt": self.imported, "config_sha256": "2" * 64,
            "index_sha256": "1" * 64, "evidence_sha256": "3" * 64,
            "layers": [{"layer": 3, "rows": 12, "path": host_rows.name, "sha256": binding.file_digest(host_rows)}]})
        self.sources = []
        for n in (5, 6):
            source_rows = self.folder / f"source_rows{n}.safetensors"
            source_rows.write_bytes(b"synthetic source row sentinel, no tensor body")
            source_ref = self.save(f"source{n}.json", {"synthetic_test_metadata": True,
                "resume_context": {"synthetic": True}, "layers": {str(n): {"ffn": {"sha256": str(n) * 64}}}})
            capture_ref = self.save(f"capture{n}.json", {"synthetic_test_metadata": True,
                "schema": "neural-forge.c17-k3-boundaries.v1", "gradients_or_training": False, "fixture": False,
                "stream": "folded", "accepted_prefix_replay_exact": True, "accepted_receipt": source_ref,
                "accepted_resume_context": {"synthetic": True}, "installed_ffn_sha256": {str(n): str(n) * 64},
                "layers": [{"layer": n, "rows": 12, "path": source_rows.name, "sha256": binding.file_digest(source_rows)}]})
            self.sources.append({"layer": n, "capture": capture_ref, "accepted_receipt": source_ref, "installed_ffn_sha256": str(n) * 64})
        maps = {"C_out": {"domain": [], "codomain": []}, "C_in": {"domain": [], "codomain": []}}
        for item in self.sources:
            for name, boundary in (("C_out", "ffn_input"), ("C_in", "ffn_output")):
                h = {"family": "host", "boundary": boundary, "layer": 3, "capture_receipt": self.host}
                k = {"family": "k3", "boundary": boundary, "layer": item["layer"], "capture_receipt": item["capture"]}
                maps[name]["domain"].append(h if name == "C_out" else k)
                maps[name]["codomain"].append(k if name == "C_out" else h)
        self.doc = {"schema": binding.SCHEMA, "fixture": False, "gradients_or_training": False,
                    "status": "candidate_only", "production_install_eligible": False,
                    "source_binding_policy": binding.OWNER_POLICY, "host_layer": 3,
                    "host_index_sha256": self.expected["host_index_sha256"],
                    "host_config_sha256": self.expected["host_config_sha256"],
                    "evidence_sha256": self.expected["evidence_sha256"], "boundary_sha256": "4" * 64,
                    "k3_layers": [5, 6], "k3_sources": self.sources, "host_capture": self.host,
                    "host_import_receipt": self.imported, "code_sha256": {"synthetic/producer.py": "a" * 64},
                    "host_checkpoint_binding": source_binding.host_checkpoint_binding(imported, source_binding_policy=binding.OWNER_POLICY),
                    "arms": {"ridge": {"path": str(self.chart), "sha256": binding.file_digest(self.chart),
                        "production_install_eligible": False,
                        "constraints": {"C_out_bias": "fixed zero during fit", "C_in_bias": "fixed zero during fit"},
                        "native_boundary_provenance": {"schema": "neural-forge.c17-native-chart-boundaries.v1",
                                                       "fixture": False, "maps": maps}}}}
        plan = {"schema": binding.PLAN_SCHEMA, "fixture": False, "source_binding_policy": binding.OWNER_POLICY,
                "host_layer": 3, "host_import_receipt": self.imported, "host_capture": self.host,
                "evidence": {"sha256": "3" * 64}, "boundary_map": {"sha256": "4" * 64},
                "k3_captures": [{k: item[k] for k in ("layer", "capture", "accepted_receipt")} for item in self.sources]}
        self.doc["plan"] = self.save("plan.json", plan)
        self.ref = self.save("charts/L03/candidate_layer.json", self.doc)
        self.tensors = {"C_in": FakeTensor([7, 4]), "C_out": FakeTensor([4, 7]),
                        "C_in_bias": FakeTensor([4]), "C_out_bias": FakeTensor([7])}
        self.native_calls = []

    def save(self, name, value):
        path = self.folder / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(path), "sha256": binding.file_digest(path)}

    def loader(self):
        tree = ast.parse((ROOT / "c10_union_install.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("load_chart", "chart_from_c17_tensors")]
        def observed(doc, path, k, arm, **geometry):
            self.native_calls.append((doc, path, k, arm, geometry))
            return {**self.expected, "fixture": False}
        env = {"__name__": "isolated_actual_loader_functions", "Path": Path, "torch": FakeTorch,
               "Any": object, "Chart": FakeChart, "native_candidate_binding": binding,
               "fixture_root_checked": lambda *a, **kw: None, "native_chart_boundaries": observed}
        extracted = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
                                                         type_ignores=[]))
        exec(compile(extracted, "isolated_actual_loader_functions", "exec"), env)
        package = ModuleType("safetensors")
        module = ModuleType("safetensors.torch")
        module.load_file = lambda path: self.tensors
        return env["load_chart"], {"safetensors": package, "safetensors.torch": module}

    def use_loader(self):
        fn, modules = self.loader()
        with patch.dict(sys.modules, modules):
            return fn(str(self.charts), 3, 7, 4, False, candidate_ref=self.ref, candidate_expected=self.expected)

    def test_explicit_candidate_preserves_classification_and_canonical_file(self):
        result = binding.read_binding(self.ref, self.expected, chart_path=self.chart)
        self.assertEqual(result["status"], "candidate_only")
        self.assertFalse(result["production_install_eligible"])
        self.assertFalse((self.layer / "layer.json").exists())
        self.assertEqual(binding.file_digest(Path(self.ref["path"])), self.ref["sha256"])
        binding.check_observed(result, self.expected)

    def test_loader_executes_existing_shape_finite_and_observed_checks(self):
        chart = self.use_loader()
        self.assertEqual(chart.trunk_to_k3.shape, (7, 4))
        self.assertEqual(chart.k3_to_trunk.shape, (4, 7))
        self.assertEqual(len(self.native_calls), 1)
        self.assertEqual(self.native_calls[0][2:4], (3, "ridge"))
        self.assertEqual(chart.layer_json_sha256, self.ref["sha256"])
        self.assertFalse(chart.native_candidate_binding["production_install_eligible"])

    def test_default_real_route_remains_unchanged(self):
        fn, modules = self.loader()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(SystemExit, "production native chart admission is unavailable"):
            fn(str(self.charts), 3, 7, 4, False)
        self.assertEqual(self.native_calls, [])

    def test_rejects_host_evidence_and_source_identity_mismatch(self):
        for key in ("host_index_sha256", "host_config_sha256", "evidence_sha256"):
            with self.subTest(key=key):
                wrong = deepcopy(self.expected)
                wrong[key] = "b" * 64
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    binding.read_binding(self.ref, wrong, chart_path=self.chart)
        wrong = deepcopy(self.expected)
        wrong["installed_ffn_sha256"]["5"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "source FFN mismatch"):
            binding.read_binding(self.ref, wrong, chart_path=self.chart)

    def test_rejects_policy_promotion_fixture_gradient_or_missing_direction(self):
        for key, value in (("source_binding_policy", None), ("production_install_eligible", True),
                           ("status", "accepted"), ("fixture", True), ("gradients_or_training", True)):
            with self.subTest(key=key):
                wrong = deepcopy(self.doc)
                wrong[key] = value
                with self.assertRaises(ValueError):
                    binding.validate_metadata(wrong, self.expected, chart_path=self.chart)
        wrong = deepcopy(self.doc)
        del wrong["arms"]["ridge"]["native_boundary_provenance"]["maps"]["C_in"]
        with self.assertRaisesRegex(ValueError, "both directed maps"):
            binding.validate_metadata(wrong, self.expected, chart_path=self.chart)

    def test_rejects_wrong_boundaries_paired_source_order_or_coverage(self):
        for mutate in (
            lambda d: d["arms"]["ridge"]["native_boundary_provenance"]["maps"]["C_in"]["domain"][0].update(boundary="ffn_input"),
            lambda d: d["arms"]["ridge"]["native_boundary_provenance"]["maps"]["C_out"]["codomain"].reverse(),
            lambda d: d.update(k3_layers=[5]),
            lambda d: d["k3_sources"].append(deepcopy(d["k3_sources"][0])),
        ):
            wrong = deepcopy(self.doc)
            mutate(wrong)
            with self.assertRaises(ValueError):
                binding.validate_metadata(wrong, self.expected, chart_path=self.chart)

    def test_rejects_chart_plan_candidate_or_source_metadata_tampering(self):
        for ref in (self.ref, self.doc["plan"], self.imported, self.host, self.sources[0]["accepted_receipt"]):
            with self.subTest(path=ref["path"]):
                path = Path(ref["path"])
                old = path.read_bytes()
                path.write_bytes(old + b" ")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    binding.read_binding(self.ref, self.expected, chart_path=self.chart)
                path.write_bytes(old)
        self.chart.write_bytes(b"changed synthetic sentinel")
        with self.assertRaisesRegex(ValueError, "chart SHA256 mismatch"):
            binding.read_binding(self.ref, self.expected, chart_path=self.chart)

    def test_rejects_host_receipt_join_or_recorded_binding_mismatch(self):
        wrong = deepcopy(self.doc)
        wrong["host_checkpoint_binding"]["checkpoint"]["config_sha256"] = "b" * 64
        changed_ref = self.save("charts/L03/candidate_layer.json", wrong)
        with self.assertRaisesRegex(ValueError, "host checkpoint binding mismatch"):
            binding.read_binding(changed_ref, self.expected, chart_path=self.chart)

    def test_rejects_missing_empty_or_different_pooled_source_contexts(self):
        original_capture, _ = binding.read_json(self.sources[0]["capture"])
        original_receipt, _ = binding.read_json(self.sources[0]["accepted_receipt"])
        for case, context in enumerate((None, {}, "not a context", {"different": True})):
            with self.subTest(context=context):
                capture, receipt = deepcopy(original_capture), deepcopy(original_receipt)
                capture["accepted_resume_context"] = context
                receipt["resume_context"] = context
                receipt_ref = self.save(f"wrong_source_{case}.json", receipt)
                capture["accepted_receipt"] = receipt_ref
                capture_ref = self.save(f"wrong_capture_{case}.json", capture)
                wrong = deepcopy(self.doc)
                wrong["k3_sources"][0]["capture"] = capture_ref
                wrong["k3_sources"][0]["accepted_receipt"] = receipt_ref
                for arm in wrong["arms"]["ridge"]["native_boundary_provenance"]["maps"].values():
                    for endpoint in arm["domain"] + arm["codomain"]:
                        if endpoint["family"] == "k3" and endpoint["layer"] == 5:
                            endpoint["capture_receipt"] = capture_ref
                plan, _ = binding.read_json(wrong["plan"])
                plan["k3_captures"] = [{k: item[k] for k in ("layer", "capture", "accepted_receipt")} for item in wrong["k3_sources"]]
                wrong["plan"] = self.save("wrong_plan.json", plan)
                wrong_ref = self.save("charts/L03/candidate_layer.json", wrong)
                message = "different native contexts" if isinstance(context, dict) and context else "resume context must be a nonempty object"
                with self.assertRaisesRegex(ValueError, message):
                    binding.read_binding(wrong_ref, self.expected, chart_path=self.chart)

    def test_rejects_loader_shape_nonfinite_dtype_and_nonzero_bias(self):
        for name, tensor in (("C_out", FakeTensor([4, 6])), ("C_in", FakeTensor([7, 4], finite=False)),
                             ("C_in", FakeTensor([7, 4], dtype="BF16")), ("C_out_bias", FakeTensor([7], nonzero=True))):
            with self.subTest(name=name, shape=tensor.shape, dtype=tensor.dtype):
                old = self.tensors[name]
                self.tensors[name] = tensor
                with self.assertRaises(SystemExit):
                    self.use_loader()
                self.tensors[name] = old

    def test_stability_check_rejects_post_read_mutation(self):
        result = binding.read_binding(self.ref, self.expected, chart_path=self.chart)
        Path(self.host["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed during consumption"):
            binding.verify_pins(result["pins"])

    def test_selected_capture_row_bodies_are_pinned_and_rechecked(self):
        result = binding.read_binding(self.ref, self.expected, chart_path=self.chart)
        rows = [item for item in result["pins"] if "_rows" in Path(item["path"]).name]
        self.assertEqual(len(rows), 3)
        Path(rows[0]["path"]).write_bytes(b"mutated synthetic row sentinel")
        with self.assertRaisesRegex(ValueError, "changed during consumption"):
            binding.verify_pins(result["pins"])

    def test_captured_row_subtree_and_exact_layer_are_required(self):
        capture, path = binding.read_json(self.host)
        capture["layers"][0]["path"] = "../outside.safetensors"
        with self.assertRaises((ValueError, FileNotFoundError)):
            binding.captured_rows_pin(capture, path, 3)
        capture, path = binding.read_json(self.host)
        capture["layers"].append(deepcopy(capture["layers"][0]))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            binding.captured_rows_pin(capture, path, 3)

    def test_cli_bindings_are_absolute_exact_and_unique(self):
        value = f"3={self.ref['path']}@{self.ref['sha256']}"
        self.assertEqual(binding.parse_bindings([value]), {3: self.ref})
        for values in ([value, value], ["3=relative.json@" + "a" * 64], ["3=" + self.ref["path"] + "@bad"]):
            with self.assertRaises(ValueError):
                binding.parse_bindings(values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
