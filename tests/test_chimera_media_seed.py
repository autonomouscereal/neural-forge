"""Actual procedural asset, decoded synchronization, and geometry checks."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.chimera_recovery import media_seed


ROOT = Path(__file__).resolve().parents[1]


class MediaSeedTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(prefix="media-seed-test-", dir=ROOT)

    def test_edit_has_distinct_real_before_after_and_unchanged_other_object(self):
        with self.temporary() as root:
            row = media_seed.build_fixture(root, 0, 2, "", "")
            before, after = row["media"]
            self.assertEqual([before["role"], after["role"]], ["input", "target"])
            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertEqual(Image.open(before["path"]).getpixel((72,72)), Image.open(after["path"]).getpixel((72,72)))
            self.assertFalse(row["quality"]["train_ready"])
            self.assertNotIn("teacher", after)

    def test_mesh_is_genuine_closed_geometry_with_verified_volume_and_render(self):
        with self.temporary() as root:
            for family_number, vertices, triangles in ((6,8,12),(7,6,8),(8,4,4)):
                row = media_seed.build_fixture(root, family_number, 0, "", "")
                truth = json.loads(Path(row["provenance"]["groundtruth_path"]).read_text())["groundtruth"]
                self.assertEqual((truth["vertex_count"],truth["triangle_count"]),(vertices,triangles))
                self.assertTrue(truth["closed_manifold"])
                self.assertGreater(truth["volume"],0)
                self.assertEqual([item["type"] for item in row["media"]],["image","mesh"])

    def test_non_speech_waveform_events_have_actual_energy_and_no_invented_asr(self):
        with self.temporary() as root:
            for family_number, count in ((9,1),(10,2),(11,3)):
                row = media_seed.build_fixture(root, family_number, 0, "", "")
                truth = json.loads(Path(row["provenance"]["groundtruth_path"]).read_text())["groundtruth"]
                self.assertEqual(truth["count"],count)
                self.assertFalse(truth["contains_speech"])
                self.assertEqual(len(truth["measured_events"]),count)
                self.assertNotIn("measured_events",row["messages"][1]["content"])
                self.assertTrue(row["quality"]["train_ready"])

    def test_visual_grounding_centers_and_hashes_match_actual_pixels(self):
        with self.temporary() as root:
            row = media_seed.build_fixture(root, 13, 2, "", "")
            truth = json.loads(Path(row["provenance"]["groundtruth_path"]).read_text())["groundtruth"]
            image = Image.open(row["media"][0]["path"])
            for item in truth["objects"]:
                self.assertEqual(image.getpixel((item["x"],item["y"])),media_seed.COLORS[item["color"]])
            self.assertEqual(media_seed.file_hash(row["media"][0]["path"]),row["media"][0]["sha256"])

    def test_sound_verifier_rejects_fabricated_event_labels(self):
        events = [{"kind":"tone","start_seconds":0.25,"duration_seconds":0.125,"frequency_hz":440}]
        with self.assertRaisesRegex(ValueError,"missing"):
            media_seed.verify_sound(np.zeros(media_seed.SAMPLE_RATE),events)
        samples = media_seed.sound_samples(events,0)
        incorrect = [{**events[0],"frequency_hz":880}]
        with self.assertRaisesRegex(ValueError,"frequency"):
            media_seed.verify_sound(samples,incorrect)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"requires existing ffmpeg/ffprobe")
    def test_actual_mux_and_decoded_audio_visual_events_share_one_origin(self):
        with self.temporary() as root:
            row = media_seed.build_fixture(root,3,0,shutil.which("ffmpeg"),shutil.which("ffprobe"))
            video,audio = row["media"]
            self.assertEqual(video["source_media_sha256"],audio["source_media_sha256"])
            self.assertEqual(video["start_seconds"],audio["start_seconds"])
            self.assertTrue(row["av_alignment"]["visual_events_verified"])
            self.assertTrue(row["av_alignment"]["audio_events_verified"])
            self.assertFalse(row["quality"]["train_ready"])

    def test_family_partitions_and_generation_bounds_are_explicit(self):
        ownership = {}
        for family,route,split in media_seed.FAMILIES:
            self.assertNotIn(family,ownership)
            ownership[family]=split
        self.assertEqual(set(ownership.values()),{"train","validation","test"})
        self.assertEqual(len(ownership),18)
        with self.assertRaises(ValueError):
            media_seed.generate("unused",127)


if __name__ == "__main__":
    unittest.main()
