"""Integration tests for stage_c_media_a.py 1.0.2 (lane media-a, Claude, 2026-10-12/13).

Runs INSIDE comfyui:codex-minimax-h3-runtime-env-r5-20260808 (ComfyUI 0.30.0, torch 2.6.0+cu124) on ONE
P40 with the REAL parent loaders and REDUCED shapes (64x64 canvas, 5 frames, 2 sigmas per item). Nothing
here is a capability claim; it proves the code paths the stage-C jobs run: real VAE encodes of decoded
media, real text encoders (H3 Qwen3-VL-32B NVFP4-AWQ, Qwen2.5-VL-7B FP8-scaled), the real H3 int8 ConvRot
and Qwen-Image FP8 DiTs through ComfyUI's CFGGuider/KSAMPLER path, the direct-call parity arm, comfy's own
noise_scaling against our x_t, CFG recombination, block-25 row pairs, resume, sealing and verify.

    docker run --rm --name media_a_itest --user $(id -u):$(id -g) --cpus 16 \\
      --gpus '"device=<P40 UUID>"' -e NVIDIA_VISIBLE_DEVICES=<P40 UUID> -e CUDA_VISIBLE_DEVICES=0 \\
      -e HOME=/tmp -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \\
      -e OC=$OC -e ITEST_DIR=<scratch dir> -w $OC/distill/tools \\
      -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro -v $OC:$OC \\
      --entrypoint nice comfyui:codex-minimax-h3-runtime-env-r5-20260808 -n 15 \\
      /opt/environments/python/comfyui/bin/python -m unittest -v itest_stage_c_media_a
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEST = Path(os.environ.get("ITEST_DIR", str(HERE / "_itest_media_a")))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load(HERE / "stage_c_media_a.py", "stage_c_media_a")
STATE: dict = {}
WAIT = ["--guard-wait", os.environ.get("ITEST_GUARD_WAIT", "1200")]  # shared P40s: other lanes run short smokes


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def free_gpu() -> None:
    gc.collect()
    M.release_gpu()
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(f"[free_gpu] device used after release: {(total - free) >> 20} MiB", flush=True)
    except Exception:  # noqa: BLE001
        pass


def sealed(directory: Path) -> dict:
    result = M.verify_cache(directory)
    assert result["ok"], result
    return json.loads((directory / "manifest.json").read_text())


def only_dir(root: Path, prefix: str) -> Path:
    found = [p for p in root.glob(f"{prefix}*/*") if p.is_dir()]
    assert len(found) == 1, (prefix, found)
    return found[0]


class StageCMediaA(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ITEST.mkdir(parents=True, exist_ok=True)
        media = ITEST / "media"
        media.mkdir(exist_ok=True)
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=1.5", "-f", "lavfi", "-i",
               "sine=frequency=440:sample_rate=44100:duration=1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-shortest", str(media / "clip_av.mp4"))
        ffmpeg("-f", "lavfi", "-i", "mandelbrot=size=64x96:rate=24", "-t", "1.0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               str(media / "clip_silent.mp4"))
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=96x64:rate=1", "-frames:v", "1", str(media / "img0.png"))
        ffmpeg("-f", "lavfi", "-i", "smptebars=size=64x80:rate=1", "-frames:v", "1", str(media / "img1.png"))
        cls.h3_items = ITEST / "h3_items.jsonl"
        cls.h3_items.write_text(
            json.dumps({"id": "av0", "prompt": "A colourful test pattern scrolls while a clear tone plays.", "video": str(media / "clip_av.mp4")}) + "\n"
            + json.dumps({"id": "sil1", "prompt": "A slow fractal zoom in silence.", "video": str(media / "clip_silent.mp4")}) + "\n")
        cls.qimg_items = ITEST / "qimg_items.jsonl"
        cls.qimg_items.write_text(
            json.dumps({"id": "i0", "prompt": "A test card with coloured bars.", "negative": "blurry", "image": str(media / "img0.png")}) + "\n"
            + json.dumps({"id": "i1", "prompt": "Television colour bars.", "image": str(media / "img1.png")}) + "\n")
        cls.root = ITEST / "cache"

    # The card is HELD with a 2.2 GB placeholder from the first step to the end of the run, so other lanes'
    # <= 2000 MiB guards never see it idle (2026-10-13: three foreign jobs landed mid-run and one OOM'd the H3 TE
    # load). Models load around it; ComfyUI partial-loads a model that no longer fits (recorded in the manifests).
    _hold = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls._hold = None
        free_gpu()

    def tearDown(self) -> None:
        free_gpu()
        if StageCMediaA._hold is None:
            try:
                import torch

                if torch.cuda.is_available():
                    StageCMediaA._hold = torch.empty(int(2.2 * 2**30), dtype=torch.uint8, device="cuda:0")
            except Exception as exc:  # noqa: BLE001 - holding is a courtesy, never a failure
                print(f"[hold] not held: {exc}", flush=True)

    def test_01_runtime_p40(self):
        info = M.comfy_boot("cuda:0", vram_guard_mib=2000, expect_device_name=None, guard_wait=float(WAIT[1]))
        print("RUNTIME", json.dumps(info, default=str))
        self.assertTrue(info["arch_compatible"])
        self.assertEqual(info["comfy_version"], "0.30.0")
        STATE["runtime"] = info

    def test_02_h3_encode(self):
        rc = M.main(["h3-encode", "--items", str(self.h3_items), "--cache-root", str(self.root), "--width", "64", "--height", "64",
                     "--length", "5", "--items-per-shard", "1", "--teacher-digest", "full", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(self.root, "h3-vae")
        man = sealed(d)
        self.assertEqual(man["dropped"], [])
        self.assertTrue(man["teacher_integrity"]["unchanged"])
        self.assertIn("sha256", next(iter(man["teacher_integrity"]["before"].values())))
        from safetensors import safe_open

        with safe_open(str(d / "shard-00000.safetensors"), "pt") as h:
            self.assertEqual(list(h.get_tensor("x0_raw_video").shape), [1, 24, 2, 4, 4])
            self.assertEqual(list(h.get_tensor("x0_raw_audio").shape), [1, 32, 2, 8])
        side = json.loads((d / "shard-00001.json").read_text())
        self.assertEqual(side["items"][0]["audio_meta"]["audio"], "silence")
        STATE["h3_vae"] = d

    def test_03_h3_text(self):
        # comfy's text_encoder_device() picks the CPU on a P40 (should_use_fp16 False): 126.5 s/text in the
        # 2026-10-13 00:44Z run. The stage-C job pins the TE to the card.
        rc = M.main(["h3-text", "--items", str(self.h3_items), "--cache-root", str(self.root), "--te", str(M.H3_TE_NVFP4),
                     "--with-negatives", "--te-load-device", "gpu", "--teacher-digest", "stat", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(self.root, "h3-te-")
        man = sealed(d)
        self.assertTrue(man["te"]["load_device"].startswith("cuda"), man["te"])
        index = json.loads((d / "index.json").read_text())
        self.assertEqual(len(index), 3)  # two prompts + the empty negative
        cache = M.TextCache(d)
        cond, extra = cache.get("A slow fractal zoom in silence.")
        self.assertEqual(cond.shape[-1], 5120)
        self.assertEqual(int(extra["minimax_token_tags"].numel()), int(cond.shape[1]))
        print("H3 TE", man["te"], man["timing"])
        STATE["h3_te"] = d

    def _h3_velocity(self, *extra):
        return M.main(["h3-velocity", "--items", str(self.h3_items), "--cache-root", str(self.root), "--dit", str(M.H3_DIT_INT8),
                       "--encode-cache", str(STATE["h3_vae"]), "--te-cache", str(STATE["h3_te"]), "--sigmas-per-item", "2",
                       "--block-rows", "8", "--items-per-shard", "1", "--parity-every", "1", "--teacher-digest", "stat", *WAIT, *extra])

    def test_04_h3_velocity_int8(self):
        self.assertEqual(self._h3_velocity(), 0)
        d = only_dir(self.root, "h3-velocity-minimax_h3_fl2va_pruned_int8_convrot")
        man = sealed(d)
        self.assertEqual(man["targets"], 4)
        sig = man["runtime_signature"]  # 1.0.2: from the shards, identical on every one
        self.assertEqual((sig["device_name"], sig["tool_version"]), ("Tesla P40", M.TOOL_VERSION))
        self.assertEqual(sig["component"]["inference_dtype"], man["dit"]["inference_dtype"])
        self.assertEqual(len(man["parity_arm"]), 4)
        self.assertTrue(all(p["pass"] for p in man["parity_arm"]))
        for reading in man["comfy_noise_scaling_parity"]:
            self.assertLess(reading["rel_l2_stream0"], 1e-5, reading)
        from safetensors import safe_open

        with safe_open(str(d / "shard-00000.safetensors"), "pt") as h:
            self.assertEqual(list(h.get_tensor("v_video").shape), [2, 24, 2, 4, 4])
            self.assertEqual(list(h.get_tensor("x0hat_audio").shape), [2, 32, 2, 8])
            self.assertEqual(list(h.get_tensor("blk25.x_in").shape), [2, 8, 5376])
            self.assertEqual(list(h.get_tensor("blk25.x_out").shape), [2, 8, 5376])
            kinds = h.get_tensor("blk25.kind")
            self.assertTrue(bool(((kinds == 1) | (kinds == 2)).all()))
            self.assertEqual(list(h.get_tensor("item.x0_video").shape), [1, 24, 2, 4, 4])
            # the stored x0hat and v are one consistent pair: x0hat_video = x_t - sigma v
            import torch

            x0 = h.get_tensor("item.x0_video")[0]
            sig = h.get_tensor("sigma")
            seeds = h.get_tensor("seed")
            for j in range(2):
                noise = M.make_noise([(1, 24, 2, 4, 4), (1, 32, 2, 8)], int(seeds[j]))
                xt = M.flow_interp(x0.unsqueeze(0), noise[0], float(sig[j]))[0]
                recon = xt - float(sig[j]) * h.get_tensor("v_video")[j]
                self.assertLess(M.rel_l2(recon, h.get_tensor("x0hat_video")[j]), 1e-5)
        side = json.loads((d / "shard-00000.json").read_text())
        self.assertEqual(side["targets"][0]["noise_sha256"], M.noise_sha(M.make_noise([(1, 24, 2, 4, 4), (1, 32, 2, 8)], side["targets"][0]["seed"])))
        print("H3 VELOCITY", json.dumps({"dit": man["dit"], "timing": man["timing"], "vram_peak": man["vram_peak"],
                                         "parity": man["parity_arm"], "noise": man["comfy_noise_scaling_parity"]}, default=str))
        STATE["h3_vel"] = d
        STATE["h3_vel_sha"] = [s["sha256"] for s in man["shards"]]

    def test_05_h3_velocity_resume_is_byte_identical(self):
        self.assertEqual(self._h3_velocity(), 0)
        man = sealed(STATE["h3_vel"])
        self.assertEqual([s["sha256"] for s in man["shards"]], STATE["h3_vel_sha"])

    def test_06_h3_velocity_cfg2(self):
        self.assertEqual(self._h3_velocity("--cfg", "2.0", "--cache-tag", "cfg2", "--max-items", "1", "--block-pairs", ""), 0)
        d = only_dir(self.root, "h3-velocity-minimax_h3_fl2va_pruned_int8_convrot-cfg2")
        man = sealed(d)
        from safetensors import safe_open

        with safe_open(str(d / "shard-00000.safetensors"), "pt") as h:
            v, c, u = h.get_tensor("v_video"), h.get_tensor("v_cond_video"), h.get_tensor("v_uncond_video")
            self.assertLess(M.rel_l2(u + 2.0 * (c - u), v), 1e-5)
            self.assertGreater(M.rel_l2(c, u), 1e-4)  # the negative changes the prediction
            self.assertNotIn("blk25.x_in", h.keys())
        self.assertEqual(man["timing"]["forwards"], 4)  # 2 targets x (cond + uncond)

    def test_07_qimg_encode(self):
        rc = M.main(["qimg-encode", "--items", str(self.qimg_items), "--cache-root", str(self.root), "--width", "64",
                     "--height", "64", "--teacher-digest", "full", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(self.root, "qimg-vae")
        sealed(d)
        from safetensors import safe_open

        with safe_open(str(d / "shard-00000.safetensors"), "pt") as h:
            self.assertEqual(list(h.get_tensor("x0_raw").shape), [2, 16, 1, 8, 8])
        STATE["qimg_vae"] = d

    def test_08_qimg_text(self):
        rc = M.main(["qimg-text", "--items", str(self.qimg_items), "--cache-root", str(self.root), "--te-load-device", "gpu",
                     "--teacher-digest", "stat", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(self.root, "qimg-te-")
        man = sealed(d)
        self.assertEqual(len(json.loads((d / "index.json").read_text())), 4)  # 2 prompts + "blurry" + ""
        cond, extra = M.TextCache(d).get("Television colour bars.")
        self.assertEqual(cond.shape[-1], 3584)
        print("QIMG TE", man["te"], man["timing"], sorted(extra))
        STATE["qimg_te"] = d

    def test_09_qimg_velocity_fp8(self):
        rc = M.main(["qimg-velocity", "--items", str(self.qimg_items), "--cache-root", str(self.root), "--encode-cache",
                     str(STATE["qimg_vae"]), "--te-cache", str(STATE["qimg_te"]), "--sigmas-per-item", "2", "--items-per-shard", "2",
                     "--parity-every", "1", "--teacher-digest", "stat", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(self.root, "qimg-velocity-")
        man = sealed(d)
        self.assertEqual(man["targets"], 4)
        self.assertTrue(all(p["pass"] for p in man["parity_arm"]))
        for reading in man["comfy_noise_scaling_parity"]:
            self.assertLess(reading["rel_l2_stream0"], 1e-5, reading)  # Wan21 process_latent_in + noise_scaling agree
        self.assertEqual(man["deviations"][0]["name"], "fp8_teacher")
        from safetensors import safe_open

        with safe_open(str(d / "shard-00000.safetensors"), "pt") as h:
            v, c, u = h.get_tensor("v"), h.get_tensor("v_cond"), h.get_tensor("v_uncond")
            self.assertEqual(list(v.shape), [4, 16, 1, 8, 8])
            self.assertLess(M.rel_l2(u + 4.0 * (c - u), v), 1e-5)
        print("QIMG VELOCITY", json.dumps({"dit": man["dit"], "timing": man["timing"], "vram_peak": man["vram_peak"],
                                           "parity": man["parity_arm"], "noise": man["comfy_noise_scaling_parity"]}, default=str))

    def test_10_probe_writes_p4_receipt(self):
        out = ITEST / "p4_h3_int8_tiny.json"
        rc = M.main(["probe", "--teacher", "h3", "--component", "dit", "--path", str(M.H3_DIT_INT8), "--label", "itest-tiny",
                     "--out", str(out), "--width", "64", "--height", "64", "--length", "5", "--text-len", "16", "--forwards", "2", *WAIT])
        self.assertEqual(rc, 0)
        receipt = json.loads(out.read_text())
        self.assertEqual(len(receipt["forwards"]), 2)
        self.assertTrue(all(f["finite"] for f in receipt["forwards"]))
        self.assertGreater(receipt["vram_peak"]["max_allocated_bytes"], 0)

    def test_11_verify_cli(self):
        dirs = [str(p) for p in self.root.glob("*/*") if (p / "manifest.json").is_file()]
        made = [k for k in ("h3_vae", "h3_te", "h3_vel", "qimg_vae", "qimg_te") if k in STATE]
        self.assertGreaterEqual(len(dirs), len(made))
        self.assertTrue(dirs)
        self.assertEqual(M.main(["verify", *dirs]), 0)


    def test_12_real_decode_verdicts_and_drop_accounting(self):
        """1.0.2: PyAV / PIL failures on corrupt media are drop verdicts, a missing file is fatal, and a sealed
        cache's drop list comes from the shard records (real Qwen-Image VAE, one corrupt image)."""

        media = ITEST / "media"
        bad_mp4, bad_png = media / "corrupt.mp4", media / "corrupt.png"
        bad_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(2048))
        bad_png.write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(512))
        dropped: list = []
        self.assertIsNone(M.decode_or_drop(lambda: M.read_video(str(bad_mp4), frames=5, fps=24, width=64, height=64), {"id": "v", "index": 0}, dropped))
        self.assertIsNone(M.decode_or_drop(lambda: M.read_image(str(bad_png), width=64, height=64), {"id": "i", "index": 1}, dropped))
        print("DECODE VERDICTS", [d["dropped"] for d in dropped])
        self.assertEqual(len(dropped), 2)
        with self.assertRaises(OSError):
            M.decode_or_drop(lambda: M.read_video(str(media / "absent.mp4"), frames=5, fps=24, width=64, height=64), {"id": "x"}, [])
        items = ITEST / "qimg_items_drop.jsonl"
        items.write_text("".join(json.dumps(row) + "\n" for row in ({"id": "ok", "prompt": "bars", "image": str(media / "img1.png")},
                                                                   {"id": "bad", "prompt": "noise", "image": str(bad_png)})))
        root = ITEST / "cache_drops"
        rc = M.main(["qimg-encode", "--items", str(items), "--cache-root", str(root), "--width", "64", "--height", "64",
                     "--items-per-shard", "1", "--teacher-digest", "stat", *WAIT])
        self.assertEqual(rc, 0)
        d = only_dir(root, "qimg-vae")
        man = sealed(d)
        self.assertEqual([m["id"] for m in man["dropped"]], ["bad"])
        self.assertEqual((man["kept"], man["drop_check"]["total"]), (1, 2))
        self.assertEqual(man["empty_shard_markers"], ["shard-00001.empty.json"])
        self.assertEqual(man["runtime_signature"]["device_name"], "Tesla P40")
        # a resume re-derives the same drop list from the records
        self.assertEqual(M.main(["qimg-encode", "--items", str(items), "--cache-root", str(root), "--width", "64", "--height", "64",
                                 "--items-per-shard", "1", "--teacher-digest", "stat", *WAIT]), 0)
        self.assertEqual([m["id"] for m in sealed(d)["dropped"]], ["bad"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
