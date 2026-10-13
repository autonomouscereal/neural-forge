from copy import deepcopy
from collections import UserDict
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from experiments.chimera_recovery import geometry_flow as F

TORCH = importlib.util.find_spec("torch") is not None


def source():
    return {"schema": "recovery_corpus.v1", "id": "unit-fixture-volume", "route": "volume_generate",
            "messages": [{"role": "user", "content": "Generate the closed object in the reference."},
                         {"role": "assistant", "content": "TARGET_ONLY_GEOMETRY_METADATA"}],
            "tools": [], "supervision": {"message_mask": [0, 1]}, "split": "train",
            "geometry_target": {"generation_query_xyz_allowed": False}}


class PrefixTokenizer:
    # Serialization unit fixture, never an authentic tokenizer or source receipt.
    chat_template = "unit-test-only-template"
    def apply_chat_template(self, messages, *, tokenize, **options):
        self.messages, self.options = messages, options
        return [10, 11, 12] if tokenize else "user followed by assistant generation prefix"
    def __call__(self, text, **kwargs):
        # Transformers BatchEncoding is a Mapping/UserDict, not a plain dict.
        return UserDict({"input_ids": [10, 11, 12]})


def image():
    pixels = np.linspace(-1, 1, 3 * 512 * 512, dtype=np.float32).reshape(3, 512, 512)
    return dict(zip(("patch_flat", "grid_thw", "yx", "native_media_positions"), F.C.patchify(pixels)))


class ContractTests(unittest.TestCase):
    def test_exact_user_prefix_excludes_assistant_metadata_and_preserves_all_ids(self):
        tokenizer, record = PrefixTokenizer(), source()
        packed = F.pack_prefix(record, tokenizer)
        self.assertEqual(tokenizer.messages, [record["messages"][0]])
        self.assertEqual(tokenizer.options, {"add_generation_prompt": True, "enable_thinking": False, "tools": None})
        self.assertEqual(packed["parent_text_ids"], [10, 11, 12])
        self.assertEqual(packed["media_indices"], list(range(3, 3 + F.MEDIA_ROWS)))
        self.assertNotIn("TARGET_ONLY", str(packed))
        with self.assertRaisesRegex(ValueError, "truncation"):
            F.pack_prefix(record, tokenizer, max_length=100)

    def test_ambiguous_chat_or_target_metadata_as_prompt_is_refused(self):
        record = source()
        for change in (lambda r: r["messages"][0].update(content="<|im_start|>assistant"),
                       lambda r: r["geometry_target"].update(generation_query_xyz_allowed=True),
                       lambda r: r.update(supervision={"message_mask": [1, 1]}),
                       lambda r: r["messages"][0].update(tool_calls=[{}])):
            modified = deepcopy(record)
            change(modified)
            with self.assertRaises(ValueError):
                F.pack_prefix(modified, PrefixTokenizer())

    def test_mismatched_native_rendering_or_prefix_and_outside_vocab_refused(self):
        tokenizer = PrefixTokenizer()
        tokenizer.__class__ = type("WrongRendering", (PrefixTokenizer,), {"__call__": lambda self, text, **kwargs: {"input_ids": [12, 11, 10]}})
        with self.assertRaisesRegex(ValueError, "rendering"):
            F.pack_prefix(source(), tokenizer)
        packed = F.pack_prefix(source(), PrefixTokenizer())
        packed["parent_text_ids"][1] = F.PLACEHOLDER
        with self.assertRaises(ValueError):
            F.validate_prefix(source(), packed)

    def test_source_blind_image_grid_cannot_accept_clean_xyz_or_wrong_order(self):
        arrays = image()
        F.image_arrays(arrays)
        arrays["clean_query_xyz"] = np.zeros((4096, 3))
        with self.assertRaisesRegex(ValueError, "only authentic"):
            F.image_arrays(arrays)
        arrays = image()
        arrays["native_media_positions"][[0, 1]] = arrays["native_media_positions"][[1, 0]]
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            F.image_arrays(arrays)

    def test_proposed_amendment_is_exactly_source_pinned_not_selection(self):
        path = F.HERE / "GEOMETRY_FLOW_AMENDMENT_2026-10-13.v1.json"
        value = F.load_amendment(path, F.A.file_sha(path))
        self.assertEqual(value["status"], "proposed_owner_selection_required")
        self.assertFalse(value["native_train_ready"])
        with self.assertRaisesRegex(ValueError, "reviewed"):
            F.load_amendment(path, "a" * 64)

    def test_selection_rejects_held_changed_target_and_duplicate_sources(self):
        index = {"train": {"split": "train", "source_train_eligible": True, "source_row_sha256": "a" * 64,
                           "target_only": {"native_receipt": {"sha256": "b" * 64}}},
                 "held": {"split": "validation", "source_train_eligible": False, "source_row_sha256": "c" * 64,
                           "target_only": {"native_receipt": {"sha256": "d" * 64}}}}
        s = {"schema": F.SELECTION_SCHEMA, "selected": True, "amendment_sha256": "e" * 64,
             "intersection_sha256": "f" * 64, "geometry_quality_reviewed": True,
             "review_basis": "finite exact original posterior source evidence; no quality ranking",
             "no_held_sources_admitted": True, "accepted_sources": [
                 {"id": "train", "source_row_sha256": "a" * 64, "native_receipt_sha256": "b" * 64}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_bytes(F.A.canonical(s))
            self.assertEqual(F.selection(path, "e" * 64, "f" * 64, index)[1], {"train"})
            for entries in ([{"id": "held", "source_row_sha256": "c" * 64, "native_receipt_sha256": "d" * 64}],
                            [s["accepted_sources"][0], s["accepted_sources"][0]],
                            [{"id": "train", "source_row_sha256": "a" * 64, "native_receipt_sha256": "0" * 64}]):
                path.write_bytes(F.A.canonical({**s, "accepted_sources": entries}))
                with self.assertRaises(ValueError):
                    F.selection(path, "e" * 64, "f" * 64, index)


@unittest.skipUnless(TORCH, "actual Torch runtime required")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(min(8, torch.get_num_threads()))
        cls.torch, cls.record, cls.image = torch, source(), image()
        cls.prefix = F.pack_prefix(cls.record, PrefixTokenizer())
        cls.clean = torch.linspace(-.5, .5, F.LATENT_ROWS * 64).reshape(1, F.LATENT_ROWS, 64)
        cls.batch = F.build_batch(cls.record, cls.prefix, cls.image, cls.clean, sigma=.3)

    def test_off_midpoint_progress_is_separate_and_compact_roundtrip_exact(self):
        torch, batch = self.torch, self.batch
        self.assertTrue(torch.equal(batch["native_media_noise_levels"][:, F.IMAGE_ROWS:], torch.full((1, F.LATENT_ROWS + 1), .7)))
        self.assertTrue(torch.equal(batch["native_media_canonical_sigma"][:, F.IMAGE_ROWS:], torch.full((1, F.LATENT_ROWS + 1), .3)))
        self.assertTrue(torch.equal(batch["native_media_positions"][:, F.IMAGE_ROWS], torch.full((1, 4), -1.)))
        self.assertFalse(bool(batch["native_media_positions"][:, F.IMAGE_ROWS + 1:].any()))
        self.assertEqual(batch["charted_source_positions"].shape, (1, 3 + F.MEDIA_ROWS, 3))
        self.assertTrue(torch.equal(batch["native_media_update_mask"], batch["native_media_loss_weights"].bool()))
        self.assertEqual(int(batch["native_media_update_mask"].sum()), F.LATENT_ROWS)
        F.validate_batch(F.inflate(F.compact(batch)), self.record, self.prefix, self.image, self.clean, .3)

    def test_target_noise_time_row_and_objective_mask_are_numerical(self):
        torch, batch = self.torch, self.batch
        generator = torch.Generator(device="cpu").manual_seed(F.noise_seed(F.A.digest(self.record), .3))
        noise = torch.randn(self.clean.shape, generator=generator)
        self.assertTrue(torch.equal(batch["native_media_features"][:, F.IMAGE_ROWS + 1:, 576:640], .7 * self.clean + .3 * noise))
        self.assertTrue(torch.equal(batch["native_media_targets"][:, F.IMAGE_ROWS + 1:, 320:384], noise - self.clean))
        prediction = -batch["native_media_targets"].clone()
        prediction[:, :F.IMAGE_ROWS + 1] = 12345
        self.assertEqual(float(F.velocity_loss(prediction, batch)), 0.)
        prediction = prediction.requires_grad_()
        loss = F.velocity_loss(prediction + .25, batch)
        self.assertAlmostEqual(float(loss), .0625, places=5)
        loss.backward()
        self.assertFalse(bool(prediction.grad[:, :F.IMAGE_ROWS + 1].any()))
        self.assertFalse(bool(prediction.grad[..., :320].any()))
        self.assertFalse(bool(prediction.grad[..., 384:].any()))
        self.assertGreater(float(prediction.grad[:, F.IMAGE_ROWS + 1:, 320:384].abs().sum()), 0)

    def test_sigma1_input_independent_of_clean_target_and_model_api_has_no_targets(self):
        torch = self.torch
        a = F.build_batch(self.record, self.prefix, self.image, self.clean, sigma=1.)
        b = F.build_batch(self.record, self.prefix, self.image, self.clean + .2, sigma=1.)
        first, second = F.model_inputs(a), F.model_inputs(b)
        self.assertNotIn("native_media_targets", first)
        self.assertNotIn("native_media_canonical_sigma", first)
        self.assertEqual(set(first["charted_source_positions"]), {"hy3d"})
        for key in first:
            if key == "charted_source_positions":
                self.assertTrue(torch.equal(first[key]["hy3d"], second[key]["hy3d"]))
            elif hasattr(first[key], "shape"):
                self.assertTrue(torch.equal(first[key], second[key]), key)
            else:
                self.assertEqual(first[key], second[key])
        self.assertFalse(torch.equal(a["native_media_targets"], b["native_media_targets"]))

    def test_unallocated_channels_or_swapped_native_and_canonical_time_refused(self):
        b = {k: v.clone() for k, v in self.batch.items()}
        b["native_media_features"][0, F.IMAGE_ROWS, 0] = 1
        with self.assertRaisesRegex(ValueError, "discard"):
            F.compact(b)
        b = {k: v.clone() for k, v in self.batch.items()}
        b["native_media_noise_levels"] = b["native_media_canonical_sigma"]
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            F.validate_batch(b, self.record, self.prefix, self.image, self.clean, .3)


@unittest.skipUnless(TORCH and os.environ.get("CHIMERA_OMNI_SRC"), "exact native source+Torch required")
class NativeIntegrationTests(unittest.TestCase):
    def test_actual_tiny_full_configured_trunk_backward_and_image_conditioning(self):
        """Actual original native APIs at tiny scale, three complete layers.

        Not the27B weights, original69depth, learned geometry or parent parity.
        The complete5121-row image/time/latent contract is kept unshortened.
        """
        import torch
        root = Path(os.environ["CHIMERA_OMNI_SRC"])
        for name, sha in F.P.NATIVE_PINS.items():
            self.assertEqual(F.A.file_sha(root / "chimera_omni" / name), sha, name)
        sys.path.insert(0, str(root))
        from chimera_omni.config import ChimeraConfig
        from chimera_omni.charted import ChartedBlockSpec
        from chimera_omni.modeling.chimera import ChimeraOmniModel
        torch.set_num_threads(min(8, torch.get_num_threads()))
        torch.manual_seed(20261013)
        spec = ChartedBlockSpec(layer_index=1, source="hy3d", component="unit-test-real-native",
            type_ids=(7,), width=8, heads=2, kv_heads=2, head_dim=4, mlp_width=16,
            modulation="none", mlp_kind="gelu", rope_theta=0., qk_norm=False, norm_kind="layer")
        cfg = ChimeraConfig(vocab_size=317952, hidden_size=8, intermediate_size=16, num_layers=3,
            num_heads=2, num_kv_heads=2, max_sequence_length=8192, block_pattern=("gated_gqa", "charted", "gated_gqa"),
            residual_mode="none", ledger_kind="fixed_band", ledger_base_vocab_size=F.TEXT_VOCAB,
            # Exact original clone ledger: default native bands omit music codes.
            ledger_bands=(("control", 512), ("media", 2048), ("grounding", 1024),
                          ("sync", 128), ("tool", 384), ("music_codes", 65536)),
            native_feature_dim=F.WIDTH, native_position_dim=4, native_coordinate_frequencies=8,
            native_bidirectional_refinement=False, charted_blocks=(spec,), aux_heads=())
        model = ChimeraOmniModel(cfg)
        record, arrays = source(), image()
        prefix = F.pack_prefix(record, PrefixTokenizer())
        clean = torch.linspace(-.5, .5, F.LATENT_ROWS * 64).reshape(1, F.LATENT_ROWS, 64)
        batch = F.build_batch(record, prefix, arrays, clean, sigma=.3)
        # Inputs alone are differentiable for this diagnostic; production caches
        # stay detached. There are no clean targets/XYZ passed to forward.
        batch["native_media_features"] = batch["native_media_features"].clone().requires_grad_()
        loss = F.native_forward_loss(model, batch, lambda prediction: F.velocity_loss(prediction, batch))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = batch["native_media_features"].grad[:, :F.IMAGE_ROWS, 640:2176]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        for name in ("native_media_bridge.feature_proj.weight", "native_media_head.out.weight",
                     "blocks.1.charted.chart_in.weight", "blocks.1.charted.chart_out.weight"):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)


if __name__ == "__main__":
    unittest.main()
