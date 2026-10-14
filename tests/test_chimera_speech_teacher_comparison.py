import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

from experiments.chimera_recovery import speech_teacher_comparison as C


def model(kind):
    common = ["README.md", "config.json", "generation_config.json", "preprocessor_config.json",
              "tokenizer_config.json", "merges.txt", "vocab.json"]
    names = common + (["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                       "model.safetensors.index.json", "chat_template.json"] if kind == "asr" else
                      ["model.safetensors", "speech_tokenizer/model.safetensors",
                       "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json"])
    return {"id": C.PARENTS[kind][0], "sha": C.PARENTS[kind][1], "cardData": {"license": "apache-2.0"},
            "siblings": [{"rfilename": name, "size": 1, "blobId": hashlib.sha1(b"blob 1\0x").hexdigest(),
                          **({"lfs": {"sha256": hashlib.sha256(b"x").hexdigest()}} if name.endswith("safetensors") else {})}
                         for name in names]}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def waveform(self, name, count):
        path = self.root / name
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(b"\1\0" * 1000 + b"\0\0" * (count - 1000))
        return path

    def source(self, identity="one", split="train", speaker="Ryan", group="group-one"):
        original = self.waveform(identity + "-original.wav", 1000)
        aligned = self.waveform(identity + "-aligned.wav", 1200)
        row = {"id": identity, "route": "speech_understand", "split": split,
               "messages": [{"role": "user", "content": "Transcribe the speech in this audio."},
                            {"role": "assistant", "content": "The boat is blue."}],
               "media": [{"path": str(aligned), "sha256": C.file_sha(aligned)}],
               "provenance": {"speaker": speaker, "independent_asr_exact_match": True,
                              "synthetic_official_voice": True, "user_reference_voice_used": False,
                              "original_audio_path": str(original), "original_audio_sha256": C.file_sha(original)}}
        admission = {"id": identity, "source_row_sha256": C.digest(row), "split": split,
                     "train_admitted": split == "train", "recovery_group_id": group,
                     "asset_sha256": row["media"][0]["sha256"],
                     "authored_answer_sha256": C.digest(row["messages"][-1]["content"])}
        return row, admission

    def write(self, name, value, lines=False):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in value) if lines else json.dumps(value), encoding="utf-8")
        return {"path": str(path), "sha256": C.file_sha(path), **({"rows": len(value)} if lines else {})}

    def test_complete_comparison_preserves_partition_waveform_and_false_optimizer(self):
        source, admission = self.source()
        held, held_admission = self.source("held", "test", group="held-group")
        ready = self.write("ready.jsonl", [source, held], True)
        inventory = self.write("inventory.jsonl", [source, held], True)
        spec = {"speech_preparation": self.write("preparation.json", {"outputs": {"source_ready": ready, "source_inventory": inventory}}),
                "global_admission": self.write("admission.json", {"sealed": True, "complete_source_traversal": True,
                     "connected_groups_verified": True, "frozen_exclusions_verified": True,
                     "outputs": {"admission": self.write("admission.jsonl", [admission, held_admission], True)}}),
                "baseline_completion": self.write("baseline.json", {"complete": True, "models": [{"model_id": m} for m in C.BASELINE_IDS]}),
                "parents": {kind: self.write(kind + ".json", model(kind)) for kind in C.PARENTS}}
        original_bytes = Path(source["media"][0]["path"]).read_bytes()
        plan = C.prepare(spec, self.root / "out")
        self.assertEqual(plan["split_counts"], {"train": 1, "validation": 0, "test": 1})
        self.assertEqual(plan["source_speech_rows"], 2)
        self.assertFalse(plan["native_train_ready"])
        self.assertTrue(all(r["optimizer_admitted"] is False for r in plan["rows"]))
        self.assertEqual(Path(source["media"][0]["path"]).read_bytes(), original_bytes)
        self.assertFalse(plan["duration_policy"]["tts_same_duration_guaranteed"])
        self.assertEqual(plan["rows"][1]["reference_h3_frames"], 2)

    def test_changed_source_or_answer_join_refused(self):
        source, admission = self.source()
        source["messages"][-1]["content"] = "A different answer."
        with self.assertRaisesRegex(ValueError, "join mismatch"):
            C.compare_rows([source], [admission])

    def test_quarantine_is_not_readmitted_by_better_parent(self):
        source, admission = self.source()
        source["provenance"]["independent_asr_exact_match"] = False
        admission["source_row_sha256"] = C.digest(source)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            C.compare_rows([source], [admission])

    def test_connected_group_cross_split_refused(self):
        a, aa = self.source()
        b, bb = self.source("two", "test")
        bb["recovery_group_id"] = aa["recovery_group_id"]
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            C.compare_rows([a, b], [aa, bb])

    def test_nonadmitted_training_or_foreign_voice_refused(self):
        source, admission = self.source()
        admission["train_admitted"] = False
        with self.assertRaisesRegex(ValueError, "non-admitted"):
            C.compare_rows([source], [admission])
        source["provenance"]["speaker"] = "Unverified clone"
        admission["source_row_sha256"] = C.digest(source)
        with self.assertRaisesRegex(ValueError, "speaker policy"):
            C.compare_rows([source], [admission])

    def test_baseline_tail_cannot_change_speech(self):
        source, admission = self.source()
        path = Path(source["media"][0]["path"])
        data = bytearray(path.read_bytes())
        data[-1] = 1
        path.write_bytes(data)
        source["media"][0]["sha256"] = C.file_sha(path)
        admission.update(source_row_sha256=C.digest(source), asset_sha256=source["media"][0]["sha256"])
        with self.assertRaisesRegex(ValueError, "changed original speech"):
            C.compare_rows([source], [admission])

    def test_pinned_parent_variant_license_and_index_required(self):
        metadata = {kind: model(kind) for kind in C.PARENTS}
        self.assertEqual(len(C.download_plan(metadata)), 2)
        wrong = copy.deepcopy(metadata)
        wrong["asr"]["id"] = "Qwen/Qwen3-ASR-0.6B"
        with self.assertRaisesRegex(ValueError, "identity/revision"):
            C.download_plan(wrong)
        wrong = copy.deepcopy(metadata)
        wrong["tts"]["cardData"]["license"] = "other"
        with self.assertRaisesRegex(ValueError, "license"):
            C.download_plan(wrong)
        wrong = copy.deepcopy(metadata)
        wrong["asr"]["siblings"] = [r for r in wrong["asr"]["siblings"] if r["rfilename"] != "model.safetensors.index.json"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            C.download_plan(wrong)

    def test_upstream_lfs_and_git_blob_validation_and_tamper(self):
        plan = C.download_plan({kind: model(kind) for kind in C.PARENTS})["asr"]
        for item in plan["files"]:
            path = self.root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        receipt = C.verify_download(plan, self.root)
        self.assertTrue(receipt["all_source_bytes_verified"])
        self.assertFalse(receipt["model_loaded"])
        (self.root / "model-00002-of-00002.safetensors").write_bytes(b"y")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            C.verify_download(plan, self.root)

    def test_source_pin_change_refused(self):
        spec = self.write("pin.json", {"a": 1})
        Path(spec["path"]).write_text('{"a":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            C.load_pin(spec)

    def test_download_is_separate_verified_and_performs_no_inference(self):
        plan = {"comparison_parents": C.download_plan({kind: model(kind) for kind in C.PARENTS}), "manifest_sha256": "test-fixture"}
        out = self.root / "private-models"
        result = C.download_candidates(plan, out, reserve_bytes=0, opener=lambda *args, **kw: io.BytesIO(b"x"))
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_source_bytes_verified"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["training_started"])
        with self.assertRaises(FileExistsError):
            C.download_candidates(plan, out, reserve_bytes=0, opener=lambda *args, **kw: io.BytesIO(b"x"))

    def test_failed_download_keeps_receipt_and_partial_without_false_seal(self):
        plan = {"comparison_parents": C.download_plan({kind: model(kind) for kind in C.PARENTS}), "manifest_sha256": "test-fixture"}
        out = self.root / "failed-models"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            C.download_candidates(plan, out, reserve_bytes=0, opener=lambda *args, **kw: io.BytesIO(b"y"))
        receipt = json.loads((out / "download.json").read_text())
        self.assertFalse(receipt["complete"])
        self.assertNotIn("all_source_bytes_verified", receipt)
        self.assertTrue(list(out.rglob("*.partial")))

    def test_changed_download_source_url_refused_before_output_creation(self):
        plan = {"comparison_parents": C.download_plan({kind: model(kind) for kind in C.PARENTS}), "manifest_sha256": "test-fixture"}
        plan["comparison_parents"]["asr"]["files"][0]["url"] = "https://example.invalid/model"
        with self.assertRaisesRegex(ValueError, "selected source"):
            C.download_candidates(plan, self.root / "bad", reserve_bytes=0)
        self.assertFalse((self.root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
