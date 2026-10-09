"""Tiny synthetic mechanics only; no source model weights or capability claims."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parent / "tools"))
import torch
from safetensors.torch import save_file, load_file
import hy3d_dense_child_contract as contract
from hy3d_dense_child_fit import fit_dense
from hy3d_dense_child_native import DenseChild, native_route_masses
from hy3d_full21_contract import source_groups
import hy3d_dense_child_bundle as bundle

torch.set_num_threads(2)


def tiny_branch(name, seed, inner=3):
    generator = torch.Generator().manual_seed(seed)
    return {"id": name, "first.weight": torch.randn(inner, 2, generator=generator).half() / 4,
            "first.bias": torch.randn(inner, generator=generator).half() / 4,
            "second.weight": torch.randn(2, inner, generator=generator).half() / 4,
            "second.bias": torch.randn(2, generator=generator).half() / 4}


def fixture():
    x = [torch.tensor([[1, 0], [0, 1], [1, 1], [-1, .5]], dtype=torch.float16),
         torch.tensor([[.5, .25], [.1, -.2], [2, 1], [-.5, -1]], dtype=torch.float16)]
    branches = [tiny_branch("expert0", 1), tiny_branch("expert1", 2), tiny_branch("shared", 3)]
    masses = [torch.tensor([[.45, .25, 1], [.35, .3, 1], [.4, .2, 1], [.5, .15, 1]], dtype=torch.float16)] * 2
    targets = []
    for data, mass in zip(x, masses):
        outputs = [torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(data, b["first.weight"], b["first.bias"]), approximate="none"), b["second.weight"], b["second.bias"]) for b in branches]
        targets.append(sum((o * mass[:, i:i + 1] for i, o in enumerate(outputs)), torch.zeros_like(data)))
    policy = {"row_chunk": 2, "max_working_bytes": 1 << 20, "ridge_ladder": [1e-4, .01, 1], "selection": "fit_only_source_unit_output_energy_stable_global_index"}
    return x, targets, branches, masses, policy


class TinyTests(unittest.TestCase):
    def test_full_output_dense_fit_and_reload(self):
        x, y, b, m, p = fixture(); state, selected, notes = fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        self.assertFalse(notes["union_gram_allocated"]); self.assertEqual(len(notes["coverage"]), 3)
        self.assertEqual(notes["fit_rows"], 8); self.assertEqual(len(selected), 3)
        child = DenseChild(state, 2, 3); before = child(x[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"; save_file(state, str(path)); saved = load_file(str(path))
            self.assertTrue(torch.equal(before, DenseChild(saved, 2, 3)(x[0])))

    def test_held_context_rejected_before_fit(self):
        x, y, b, m, p = fixture()
        with self.assertRaises(ValueError): fit_dense(x, y, b, 3, p, m, ["fit", "held"])

    def test_deterministic_source_units(self):
        x, y, b, m, p = fixture()
        first = fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        second = fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        self.assertTrue(torch.equal(first[1], second[1]))
        for name in first[0]: self.assertTrue(torch.equal(first[0][name], second[0][name]))

    def test_route_shared_bias_and_no_renormalization(self):
        x, y, b, m, p = fixture(); normalized = [v.clone() for v in m]
        for mass in normalized: mass[:, :2] /= mass[:, :2].sum(-1, keepdim=True)
        dense = lambda data, source: torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(data, source["first.weight"], source["first.bias"])), source["second.weight"], source["second.bias"])
        wrong = sum(dense(x[0], source) * normalized[0][:, i:i + 1] for i, source in enumerate(b))
        self.assertFalse(torch.equal(y[0], wrong))
        no_shared = sum(dense(x[0], source) * m[0][:, i:i + 1] for i, source in enumerate(b[:2]))
        self.assertFalse(torch.equal(y[0], no_shared))
        state, _, notes = fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        self.assertEqual(notes["coverage"][-1]["fit_mean_native_mass"], 1)

    def test_memory_guard_before_gram(self):
        x, y, b, m, p = fixture(); p["max_working_bytes"] = 1
        with self.assertRaises(ValueError): fit_dense(x, y, b, 3, p, m, ["fit", "fit"])

    def test_nonfinite_source_and_target_rejected(self):
        x, y, b, m, p = fixture(); y[0][0, 0] = float("nan")
        with self.assertRaises(ValueError): fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        x, y, b, m, p = fixture(); b[0]["first.weight"][0, 0] = float("inf")
        with self.assertRaises(ValueError): fit_dense(x, y, b, 3, p, m, ["fit", "fit"])

    def test_shape_corruption_and_meta_reload(self):
        x, y, b, m, p = fixture(); state, _, _ = fit_dense(x, y, b, 3, p, m, ["fit", "fit"])
        invalid = dict(state); invalid["fc1.bias"] = torch.zeros(2, dtype=torch.float16)
        with self.assertRaises(ValueError): DenseChild(invalid, 2, 3)
        child = DenseChild.meta(2, 3); self.assertTrue(child.fc1.weight.is_meta)
        child.load_state_dict(state, strict=True, assign=True); self.assertFalse(child.fc1.weight.is_meta)
        self.assertTrue(torch.equal(child(x[0]), DenseChild(state, 2, 3)(x[0])))

    def test_complete_source_groups_reject_missing_decoder(self):
        names = ["model.x", "vae.encoder.x", "vae.post_kl.x", "vae.transformer.x", "vae.geo_decoder.x", "conditioner.x"]
        self.assertEqual(set(source_groups(names)), {"dit", "shape_vae_encoder", "shape_vae_decoder", "conditioner"})
        with self.assertRaises(ValueError): source_groups([n for n in names if not n.startswith("vae.geo_decoder")])
        with self.assertRaises(ValueError): source_groups(names + ["unknown.x"])

    def test_actual_route_masses_are_not_rewritten(self):
        logits = torch.tensor([[[4, 2, 1, 0, -1, -2, -3, -4]]], dtype=torch.float16)
        ids = logits.topk(2, dim=-1).indices; weights = logits.softmax(-1).gather(-1, ids)
        saved = weights.clone(); masses, notes = native_route_masses({"router_logits": logits, "router_topk_ids": ids, "router_topk_weights": weights}, [0])
        self.assertTrue(torch.equal(weights, saved)); self.assertLess(float(masses[0][:, :8].sum()), 1)
        self.assertEqual(float(masses[0][:, 8].sum()), 1); self.assertFalse(notes[0]["numeric_veto"])
        with self.assertRaises(ValueError): native_route_masses({"router_logits": logits, "router_topk_ids": torch.zeros_like(ids), "router_topk_weights": weights}, [0])

    def test_bundle_missing21_and_bad_metadata_rejected(self):
        with self.assertRaises(ValueError): contract.coverage21([])
        good = contract.child_contract(15, 2048, 1, [0], [8192] * 9)
        for key, bad in (("width", 1), ("source_branch_widths", [1] * 9), ("activation", "relu")):
            changed = copy.deepcopy(good); changed[key] = bad
            with self.assertRaises(ValueError): contract.validate_child(changed)

    def test_native_factory_meta_fourier_reconstruction(self):
        class FourierEmbedder(torch.nn.Module):
            def __init__(self, num_freqs=8, logspace=True, input_dim=3, include_input=True, include_pi=False):
                super().__init__(); self.num_freqs=num_freqs; self.include_input=include_input
                self.out_dim=input_dim*(num_freqs*2+int(include_input))
                values=2.0**torch.arange(num_freqs,dtype=torch.float32)
                if include_pi: values*=torch.pi
                self.register_buffer('frequencies',values,persistent=False)
        class Holder(torch.nn.Module): pass
        with torch.device('meta'):
            vae=Holder(); vae.fourier_embedder=FourierEmbedder(); vae.encoder=Holder(); vae.geo_decoder=Holder()
            vae.encoder.fourier_embedder=vae.fourier_embedder; vae.geo_decoder.fourier_embedder=vae.fourier_embedder
        self.assertTrue(vae.fourier_embedder.frequencies.is_meta)
        receipt=bundle.materialize_fourier_buffer(vae, {'vae':{'params':{'num_freqs':8,'include_pi':False}}})
        self.assertTrue(torch.equal(vae.fourier_embedder.frequencies, torch.tensor([1,2,4,8,16,32,64,128],dtype=torch.float32)))
        self.assertIs(vae.encoder.fourier_embedder, vae.geo_decoder.fourier_embedder)
        self.assertNotIn('fourier_embedder.frequencies',vae.state_dict()); self.assertFalse(receipt['trained_weight'])
        vae.to(dtype=torch.float16); self.assertEqual(vae.fourier_embedder.frequencies.dtype,torch.float16)
        with self.assertRaises(ValueError): bundle.materialize_fourier_buffer(vae, {'vae':{'params':{'num_freqs':7,'include_pi':False}}})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TinyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"schema": "hy3d-dense-child-tiny-functional.v1", "tests_run": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
                      "torch_version": torch.__version__, "threads": torch.get_num_threads(),
                      "model_weights": False, "gpu": False, "scope": "synthetic tiny mechanics only; no source-native or capability claims"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
