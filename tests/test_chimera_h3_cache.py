"""Bounded H3 contracts; synthetic numerical fixtures do not claim teacher capture."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_cache", ROOT / "experiments/chimera_recovery/h3_cache.py")
h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3)
SOURCE_ROOT = Path("C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09")


def item():
    return {"id": "bounded-2026-fixture", "video": "/remote/same.mp4", "audio": "/remote/other.wav",
            "sha256": "a" * 64, "audio_sha256": "b" * 64, "source_row_sha256": "c" * 64,
            "source_group_id": "synthetic-2026-family-train", "split": "train",
            "source_window": {"start_seconds": 0.0, "end_seconds": 22 / 24, "native_frames": 22,
                              "native_fps": 24, "source_fps": 8, "source_duration_seconds": 1.0,
                              "spatial_size": [96, 96],
                              "source_frame_indices": [int((i / 24) * 8 + 1e-10) for i in range(22)]}}


def source_record():
    c = h3.strict_item(item())["h3_source_contract"]
    c["measured"] = {"video_frame_indices": c["window"]["source_frame_indices"],
                     "video_target_times": [i / 24 for i in range(22)], "video_source_times": [i / 8 for i in range(8)],
                     "audio_source_frame_pts": [0.0, 0.064, 0.128], "audio_source_rate": 16000,
                     "audio_rate": 32000, "audio_samples": round(22 / 24 * 32000), "audio": "decoded",
                     "waveform_padded_samples": 0, "audio_discontinuities": 0,
                     "video_first_pts_seconds": 0.0, "audio_first_pts_seconds": 0.0,
                     "minimax_model_sha256": h3.H3_MODEL_SHA256}
    return c


def write_tensor_fixture(path: Path):
    # Header-only structural fixture with actual allocated zero payload bytes.
    # It exercises seals/layout, not numerical teacher correctness.
    shapes = {k: [1] for k in h3.REQUIRED_TENSORS}
    for key in ("x_t_video", "v_video", "x0hat_video", "item.x0_video"):
        shapes[key] = [1, 24, 7, 12, 12]
    for key in ("x_t_audio", "v_audio", "v_audio_native", "x0hat_audio", "item.x0_audio"):
        shapes[key] = [1, 32, 2, 37]
    header, data, cursor = {}, bytearray(), 0
    for name, shape in sorted(shapes.items()):
        dtype = "I64" if name in {"seed", "target_item", "item.index"} else "F32"
        size = (8 if dtype == "I64" else 4)
        for n in shape:
            size *= n
        header[name] = {"shape": shape, "dtype": dtype, "data_offsets": [cursor, cursor + size]}
        data.extend(bytes(size))
        cursor += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return header


def sealed_fixture(directory: Path):
    p = directory / "shard-000000.safetensors"
    write_tensor_fixture(p)
    side = {"sha256": h3.sha(p), "finite": True,
            "targets": [{"item_index": 4, "sigma": 0.7, "seed": 11, "noise_sha256": "d" * 64,
                         "h3_source_contract": source_record()}]}
    p.with_suffix(".json").write_bytes(h3.canonical(side))
    weights = {"/models/actual-h3.safetensors": {"sha256": "e" * 64, "bytes": 32, "mtime_ns": 1, "inode": 2}}
    manifest = {"kind": "h3-velocity", "h3_joint_contract": h3.CONTRACT, "all_finite": True,
                "tool": {"sha256": "f" * 64}, "teacher_integrity": {"mode": "full", "unchanged": True,
                    "before": weights, "after": copy.deepcopy(weights)},
                "parity_arm": [{"pass": True}], "plan": {"cfg": 1.0},
                "encode_cache": {"manifest_sha256": "1" * 64}, "te_cache": {"manifest_sha256": "2" * 64},
                "read_order": [p.name], "shards": [{"file": p.name, "sha256": h3.sha(p),
                    "bytes": p.stat().st_size, "sidecar_sha256": h3.sha(p.with_suffix('.json'))}]}
    reseal(directory, manifest)
    return manifest, side


def reseal(directory: Path, manifest):
    p = directory / "manifest.json"
    p.write_bytes(h3.canonical(manifest))
    p.with_suffix(".json.sha256").write_text(h3.sha(p))


class ContractTests(unittest.TestCase):
    def test_same_mux_adaptation_removes_unrelated_assignment(self):
        row = h3.strict_item(item())
        self.assertEqual(row["audio"], row["video"])
        self.assertEqual(row["audio_sha256"], row["sha256"])
        self.assertNotEqual(row["audio_sha256"], item()["audio_sha256"])

    def test_wrong_window_or_holdout_refused(self):
        for key, value in (("end_seconds", 1), ("source_frame_indices", [0] * 22), ("start_seconds", 0.125)):
            row = item()
            row["source_window"][key] = value
            with self.assertRaises(h3.Refusal):
                h3.strict_item(row)
        row = item()
        row["split"] = "test"
        with self.assertRaises(h3.Refusal):
            h3.strict_item(row)

    def test_clock_padding_source_and_parent_refusals(self):
        h3.check_source_record(source_record())
        for key, value in (("waveform_padded_samples", 1), ("audio_first_pts_seconds", 0.01),
                           ("audio_discontinuities", 1), ("minimax_model_sha256", "0" * 64)):
            c = source_record()
            c["measured"][key] = value
            with self.assertRaises(h3.Refusal):
                h3.check_source_record(c)
        c = source_record()
        c["audio_sha256"] = "0" * 64
        with self.assertRaises(h3.Refusal):
            h3.check_source_record(c)

    def test_structural_preflight_never_claims_native_training_ready(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            sealed_fixture(directory)
            result = h3.preflight(directory, "f" * 64)
            self.assertTrue(result["capture_contract_ok"], result["problems"])
            self.assertFalse(result["native_train_ready"])
            self.assertFalse(h3.preflight(directory)["capture_contract_ok"])

    def test_unsealed_sidecar_and_legacy_targets_refused(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            m, side = sealed_fixture(directory)
            (directory / "shard-000000.json").write_bytes(h3.canonical({**side, "targets": []}))
            self.assertFalse(h3.preflight(directory, "f" * 64)["capture_contract_ok"])
            m.pop("h3_joint_contract")
            m["shards"][0].pop("sidecar_sha256")
            reseal(directory, m)
            result = h3.preflight(directory, "f" * 64)
            self.assertFalse(result["capture_contract_ok"])
            self.assertTrue(any("legacy" in p for p in result["problems"]))

    def test_pinned_writer_candidate_ast_and_actual_capture_anchors(self):
        p = SOURCE_ROOT / "distill/tools/stage_c_media_a.py"
        if not p.exists():
            self.skipTest("authentic 2026 source not mounted")
        candidate, receipt = h3.writer_candidate(p)
        self.assertEqual(receipt["baseline_sha256"], h3.sha(p))
        for anchor in ('put("x_t_video", xt[0][0])', 'put("x_t_audio", xt[1][0])',
                       'put("v_audio_native", tgt["v_audio_native"][0])', '"sidecar_sha256"',
                       '"source_frame_indices": picks', '"source_frame_pts": source_pts'):
            self.assertIn(anchor, candidate)
        self.assertFalse(receipt["teacher_job_launched"])

    def test_dual_clock_schedule_differs_from_video_clock(self):
        sa, slope = h3.schedule(0.6)
        self.assertAlmostEqual(sa, 0.2727272727272727)
        self.assertNotAlmostEqual(slope, 1.0)
        for value in (0, 1, float("nan"), True):
            with self.assertRaises(h3.Refusal):
                h3.schedule(value)


class TorchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("run in the supported Torch runtime")
        cls.torch = torch

    def tiny_pair(self):
        torch = self.torch
        sv, seed, index = 0.6, 47, 19
        sa, slope = h3.schedule(sv)
        g = torch.Generator().manual_seed(991)
        x0 = [torch.randn((1, 24, 2, 4, 4), generator=g), torch.randn((1, 32, 2, 3), generator=g)]
        g = torch.Generator().manual_seed(seed)
        noise = [torch.randn(x.shape, generator=g) for x in x0]
        xt = [(1 - s) * x + s * n for s, x, n in zip((sv, sa), x0, noise)]
        den = [x * 0.75 for x in xt]
        vv, va = [((x.double() - d.double()) / sv).float() for x, d in zip(xt, den)]
        va_native = (va.double() / slope).float()
        tensors = {"x_t_video": xt[0], "x_t_audio": xt[1], "v_video": vv, "v_audio": va,
                   "v_audio_native": va_native, "x0hat_video": den[0],
                   "x0hat_audio": (xt[1].double() - sa * va_native.double()).float(),
                   "sigma": torch.tensor([sv]), "sigma_audio": torch.tensor([sa]), "slope_audio": torch.tensor([slope]),
                   "seed": torch.tensor([seed]), "target_item": torch.tensor([index]),
                   "item.index": torch.tensor([index]), "item.x0_video": x0[0], "item.x0_audio": x0[1]}
        target = {"item_index": index, "seed": seed, "sigma": sv,
                  "noise_sha256": hashlib.sha256("".join(h3.tensor_digest(n) for n in noise).encode()).hexdigest()}
        return tensors, target

    def test_actual_noisy_pair_and_audio_slope_conversion(self):
        tensors, target = self.tiny_pair()
        pair = h3.validate_pair(tensors, target, 0, 1.0)
        self.assertTrue(self.torch.equal(pair["audio_velocity"], tensors["v_audio_native"]))
        self.assertFalse(self.torch.equal(pair["audio_velocity"], tensors["v_audio"]))
        for key in ("x_t_audio", "v_audio_native"):
            bad = {k: v.clone() for k, v in tensors.items()}
            bad[key].flatten()[0] += 0.1
            with self.assertRaises(h3.Refusal):
                h3.validate_pair(bad, target, 0, 1.0)

    def test_nonordinal_item_join_and_seed_digest_refusal(self):
        tensors, target = self.tiny_pair()
        self.assertEqual(target["item_index"], 19)
        h3.validate_pair(tensors, target, 0, 1.0)
        for key, value in (("item_index", 0), ("seed", 48), ("noise_sha256", "0" * 64)):
            with self.assertRaises(h3.Refusal):
                h3.validate_pair(tensors, {**target, key: value}, 0, 1.0)

    def test_packer_full_chart_and_separate_noisy_emission_allocations(self):
        torch = self.torch
        tensors, target = self.tiny_pair()
        pair = h3.validate_pair(tensors, target, 0, 1.0)
        prefix = torch.tensor([[9, 7, 5]], dtype=torch.int64)
        # Synthetic bridge geometry explicitly for an API fixture; no native
        # clock provenance is asserted by this test or by pack_pair.
        m = 2 * 3 + 2 * 2 * 2
        positions = torch.ones(1, m, 4)
        batch = h3.pack_pair(pair, prefix, positions, media_token_id=11)
        self.assertEqual(tuple(batch["charted_source_positions"].shape), (1, 3 + m, 3))
        self.assertTrue(torch.equal(batch["charted_source_positions"][0, :3, 0], torch.arange(3).float()))
        self.assertEqual(batch["native_media_type_ids"].tolist(), [[3] * 6 + [2] * 8])
        self.assertTrue(torch.equal(batch["native_media_loss_weights"], batch["native_media_loss_mask"]))
        self.assertTrue(torch.equal(batch["native_media_noise_levels"][0, :6], torch.full((6,), pair["sigma_audio"])))
        self.assertTrue(bool((batch["native_media_features"][..., :384] == 0).all()))
        self.assertTrue(bool((batch["native_media_targets"][..., 256:] == 0).all()))
        self.assertTrue(bool((batch["native_media_features"][0, :6, 384:480] == 0).all()))
        self.assertTrue(bool((batch["native_media_targets"][0, :6, 128:224] == 0).all()))
        self.assertNotEqual(batch["native_media_features"].data_ptr(), batch["native_media_targets"].data_ptr())
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            typed = h3.pack_pair(pair, prefix, positions, media_token_id=11)
            for field in ("native_media_noise_levels", "native_media_loss_mask", "native_media_loss_weights"):
                self.assertEqual(typed[field].dtype, torch.float32)
        finally:
            torch.set_default_dtype(original_dtype)
        with self.assertRaises(h3.Refusal):
            h3.pack_pair(pair, prefix, positions[:, :-1], media_token_id=11)

    def test_chart_matches_authentic_2026_packed_layout_transcription(self):
        p = SOURCE_ROOT / "composite/tools/s8_media_smoke.py"
        if not p.exists():
            self.skipTest("authentic 2026 source not mounted")
        spec = importlib.util.spec_from_file_location("h3_chart_reference_2026", p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        got = h3.chart_positions(7, 7, 12, 12, 37)
        expected = module.h3_source_positions(7, 7, 12, 12, 37).float().unsqueeze(0)
        self.assertTrue(self.torch.equal(got, expected))
        self.assertEqual(tuple(got.shape), (1, 7 + 74 + 252, 3))


if __name__ == "__main__":
    unittest.main()
