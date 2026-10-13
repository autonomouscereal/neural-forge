import inspect
import unittest

import numpy as np

from experiments.chimera_recovery import geometry_bridge_probe as P

try:
    import torch
except ImportError:
    torch = None


class BoundaryContractTests(unittest.TestCase):
    def test_generation_inputs_have_no_target_xyz_parameter(self):
        self.assertEqual(tuple(inspect.signature(P.media).parameters), ("torch", "noisy", "image", "sigma", "policy"))

    def test_incomplete_noisy_layout_or_unselected_policy_rejected_before_tensor_allocation(self):
        for noisy, policy in ((np.zeros((1, 4095, 64)), "ordinal_slot"),
                              (np.zeros((1, 4096, 64)), "physical_xyz")):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                P.media(None, noisy, {"patch_flat": np.zeros((1024, 1536))}, .2, policy)


@unittest.skipIf(torch is None, "actual Torch runtime required")
class ActualTensorContractTests(unittest.TestCase):
    def test_progress_input_and_real_time_sentinel_are_separate_from_sigma(self):
        noisy = torch.full((1, 4096, 64), 3.)
        image = {"patch_flat": np.full((1024, 1536), .25, dtype=np.float32),
                 "native_media_positions": np.zeros((1024, 4), dtype=np.float32)}
        features, types, position, valid, progress = P.media(torch, noisy, image, .2, "position_free_slot")
        self.assertTrue(torch.equal(features[0, 1025:, 576:640], noisy[0]))
        self.assertTrue(torch.equal(features[0, :1024, 640:2176], torch.from_numpy(image["patch_flat"])))
        self.assertFalse(features[0, 1024].any())
        self.assertEqual(position[0, 1024].tolist(), [-1.] * 4)
        self.assertFalse(position[0, 1025:].any())
        self.assertTrue(torch.equal(progress[0, 1024:], torch.full((4097,), .8)))
        self.assertTrue(valid.all())
        self.assertEqual(types[0, 1024:].unique().tolist(), [7])

    def test_storage_seal_supports_actual_bfloat16_without_numpy_recast(self):
        value = torch.tensor([1., 2.], dtype=torch.bfloat16)
        self.assertEqual(P.storage_hash(torch, value), P.storage_hash(torch, value.clone()))
        self.assertNotEqual(P.storage_hash(torch, value), P.storage_hash(torch, value.float()))


if __name__ == "__main__":
    unittest.main()
