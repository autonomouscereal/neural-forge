from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from experiments.chimera_recovery import geometry_targets as V


class TargetFilesTests(unittest.TestCase):
    def write_target(self, path, change=None, tail=b""):
        header, blobs, cursor = {}, [], 0
        for name, (dtype, shape) in V.TENSORS.items():
            a = np.zeros(shape, dtype="<f4" if dtype == "F32" else np.bool_)
            blob = a.tobytes()
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + len(blob)]}
            blobs.append(blob)
            cursor += len(blob)
        if change:
            change(header)
        encoded = V.A.canonical(header)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs) + tail)

    def test_exact_target_contract_and_all_bytes_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.safetensors"
            self.write_target(path)
            arrays = V.targets(path)
            self.assertEqual(set(arrays), set(V.TENSORS))
            self.assertEqual(arrays["clean_raw_latent"].shape, (1, 4096, 64))
            self.assertEqual(arrays["physical_oracle_inside"].dtype, np.bool_)

    def test_wrong_layout_leak_key_and_overlapping_bytes_fail_closed(self):
        changes = (
            lambda h: h["clean_raw_latent"].update(shape=[1, 64, 4096]),
            lambda h: h["clean_raw_latent"].update(dtype="F16"),
            lambda h: h.update(generation_query_xyz=h["target_only_encoder_query_xyz"]),
            lambda h: h["clean_model_space_latent"].update(data_offsets=h["clean_raw_latent"]["data_offsets"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.safetensors"
            for change in changes:
                with self.subTest(change=change):
                    self.write_target(path, change)
                    with self.assertRaises(ValueError):
                        V.targets(path)
            self.write_target(path, tail=b"untracked")
            with self.assertRaisesRegex(ValueError, "unaccounted"):
                V.targets(path)

    def test_truncated_or_nonfinite_target_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.safetensors"
            self.write_target(path)
            data = path.read_bytes()
            path.write_bytes(data[:-1])
            with self.assertRaises(ValueError):
                V.targets(path)
            length = struct.unpack("<Q", data[:8])[0]
            modified = bytearray(data)
            modified[8 + length:12 + length] = struct.pack("<f", float("nan"))
            path.write_bytes(modified)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                V.targets(path)

    def test_collected_container_paths_require_identical_file_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipt.json"
            path.write_text("actual file", encoding="utf-8")
            self.assertEqual(V.collected_path(root, {"path": "/work/output/receipt.json", "sha256": V.A.file_sha(path)}), path)
            with self.assertRaisesRegex(ValueError, "file seal"):
                V.collected_path(root, {"path": "/work/output/receipt.json", "sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
