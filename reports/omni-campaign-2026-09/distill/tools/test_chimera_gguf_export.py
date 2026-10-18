#!/usr/bin/env python3
"""Tests for `chimera_gguf_export.py` (E3 / S0-P9, tool 1.0.0).  Server only: torch,
safetensors, transformers 5.12.1, the chimera_omni tree, llama.cpp b10752 under
$OC/export/llamacpp and the parent's tokenizer files.

Run::

    OC=/home/cereal/omni-campaign
    EF=$OC/import/rq27b/tree_ef1ffb9
    CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$OC/distill/tools \\
        /home/cereal/model-surgery-lab/venv/bin/python test_chimera_gguf_export.py -v

The central test: a tiny Qwen3.5-shaped HF checkpoint (P1's fixture, with the
parent's real 248,320-row text vocabulary) is imported into Chimera with the REAL
importer and ledger bands, converted by this tool through the P1 reader, and
converted again by the STOCK `convert_hf_to_gguf.py` straight from the HF
directory.  The two GGUFs must be byte-identical tensor for tensor, with equal
non-`general.` metadata: the reader path is the stock path without the copy.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = os.environ.get("CHIMERA_OMNI_SRC", "")
if _SRC:
    sys.path.insert(0, str(Path(_SRC) / "src"))
PARENT = Path(os.environ.get("OC_PARENT_DIR", "") or "/home/cereal/models/Qwen3.8-27B-Uncensored")
LLAMACPP = Path(os.environ.get("OC_LLAMACPP_ROOT", "") or "/home/cereal/omni-campaign/export/llamacpp")
DISTILL_DIR = Path(os.environ.get("OC_DISTILL_DIR", "") or HERE.parent)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import chimera_gguf_export as e3  # noqa: E402
import test_p1_qwen35_reader as p1t  # noqa: E402  (its fixture builders, not its tests)
from chimera_omni.importing.qwen35 import import_qwen35_state  # noqa: E402

TEXT_VOCAB = 248320   # the parent tokenizer's rows; the bands sit above them
p1t.VOCAB = TEXT_VOCAB
LAYERS = p1t.LAYERS


def run_main(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = e3.main(argv)
    return code, out.getvalue()


def insert_charted(chimera: dict[str, torch.Tensor], cfg, directory: Path, at: tuple[int, ...]) -> Path:
    """The tiny child with stand-in charted blocks inserted, as c16 does for OMNI1."""

    total = LAYERS + len(at)
    text_of = [b for b in range(total) if b not in at]
    gen = torch.Generator().manual_seed(5)
    params = {}
    for name, tensor in chimera.items():
        if name.startswith("blocks."):
            _, index, rest = name.split(".", 2)
            params[f"blocks.{text_of[int(index)]}.{rest}"] = tensor.clone()
        else:
            params[name] = tensor.clone()
    for block in at:
        params[f"blocks.{block}.charted.chart_in.weight"] = (torch.randn(16, p1t.HIDDEN, generator=gen)).to(torch.bfloat16)
        params[f"blocks.{block}.charted.chart_out.weight"] = (torch.randn(p1t.HIDDEN, 16, generator=gen)).to(torch.bfloat16)
    p1t.write_chimera(directory, params, cfg)
    path = directory / "config.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    pattern = list(document["config"]["block_pattern"])
    kinds = iter(pattern[t % len(pattern)] for t in range(LAYERS))
    document["config"]["block_pattern"] = ["charted" if b in at else next(kinds) for b in range(total)]
    document["config"]["num_layers"] = total
    document["config"]["charted_blocks"] = [{"layer_index": b, "component": "tiny", "source": "tiny"} for b in at]
    path.write_text(json.dumps(document), encoding="utf-8")
    return directory


_SHARED: dict = {}


def tearDownModule() -> None:
    if "tmp" in _SHARED:
        _SHARED["tmp"].cleanup()


class Fixture(unittest.TestCase):
    """Built once per module (two conversions + the stock one)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _SHARED:
            _SHARED.update(cls._build())
        for key, value in _SHARED.items():
            setattr(cls, key, value)

    @staticmethod
    def _build() -> dict:
        cls = type("Shared", (), {})
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("E3_TEST_TMP") or None)
        cls.root = Path(cls.tmp.name)
        cls.conv = e3.Converter(LLAMACPP)
        cls.cfg = p1t.chimera_cfg()
        cls.qwen = p1t.qwen_state()
        model, _receipt = import_qwen35_state(cls.qwen, cls.cfg, dtype=torch.bfloat16)
        cls.chimera = {n: p.detach().clone() for n, p in model.named_parameters()}
        cls.chimera_dir = cls.root / "chimera"
        p1t.write_chimera(cls.chimera_dir, cls.chimera, cls.cfg)
        # the equivalent HF checkpoint, for the stock converter, and the "parent"
        # (config + tokenizer) the reader path takes its vocabulary from
        cls.hf_dir = cls.root / "hf"
        cls.hf_dir.mkdir()
        save_file({k: v.contiguous() for k, v in cls.qwen.items()}, str(cls.hf_dir / "model.safetensors"))
        (cls.hf_dir / "config.json").write_text(json.dumps(p1t.hf_config()), encoding="utf-8")
        for name in e3.TOKENIZER_FILES:
            if (PARENT / name).is_file():
                shutil.copyfile(PARENT / name, cls.hf_dir / name)
        cls.out = cls.root / "out"
        code, text = run_main(["convert", "--chimera", str(cls.chimera_dir), "--parent-dir", str(cls.hf_dir),
                               "--out-dir", str(cls.out), "--tag", "tiny", "--min-free-gib", "0",
                               "--parent-map", "", "--work-dir", str(cls.root)])
        if code != 0:
            raise AssertionError(f"fixture convert failed ({code}):\n{text}")
        cls.gguf = cls.out / "tiny.bf16.gguf"
        cls.stock = cls.root / "stock.bf16.gguf"
        env = dict(os.environ, PYTHONPATH=f"{LLAMACPP / 'llama.cpp' / 'gguf-py'}:{os.environ.get('PYTHONPATH', '')}")
        done = subprocess.run([sys.executable, str(LLAMACPP / "llama.cpp" / "convert_hf_to_gguf.py"),
                               "--outtype", "bf16", "--no-mtp", "--outfile", str(cls.stock), str(cls.hf_dir)],
                              capture_output=True, text=True, env=env, timeout=900)
        if done.returncode != 0:
            raise AssertionError(f"stock converter failed:\n{done.stderr[-3000:]}")
        return {k: v for k, v in vars(cls).items() if not k.startswith("__")}

    def tensors(self, path: Path) -> dict:
        return e3._gguf_tensors(self.conv, path)[1]


class TestAgainstTheStockConverter(Fixture):
    def test_tool_version(self) -> None:
        self.assertTrue(e3.TOOL_VERSION.startswith("chimera_gguf_export 1."))

    def test_reader_path_is_byte_identical_to_the_stock_hf_path(self) -> None:
        """Every tensor and every non-general key is byte-identical, except the
        `1 + w` norms: the importer stored them as bf16(1 + w) (that IS the
        child), the stock converter on the HF weights writes fp32(w) + 1.  So for
        exactly those, the reader path must equal the stock value rounded to bf16."""

        import numpy as np

        result = e3.compare_ggufs(self.conv, self.gguf, self.stock)
        self.assertEqual([], result["only_left"])
        self.assertEqual([], result["only_right"])
        self.assertEqual([], result["kv_differ"])
        self.assertEqual(3 + 3 * 14 + 11, result["common"])
        plus_one = sorted(n for n in self.tensors(self.gguf)
                          if n.endswith(("attn_norm.weight", "post_attention_norm.weight",
                                         "attn_q_norm.weight", "attn_k_norm.weight"))
                          or n == "output_norm.weight")
        self.assertEqual(4 * 2 + 2 + 1, len(plus_one))
        self.assertEqual(set(result["differ"]) - set(plus_one), set())
        mine, stock = self.tensors(self.gguf), self.tensors(self.stock)
        rounded = 0
        for name in plus_one:
            got = np.asarray(mine[name].data, dtype=np.float32)
            want = torch.from_numpy(np.array(stock[name].data, dtype=np.float32)).to(torch.bfloat16).float().numpy()
            self.assertTrue(np.array_equal(got, want), name)
            rounded += int(not np.array_equal(got, np.asarray(stock[name].data, dtype=np.float32)))
        # the rounding is real on random norms, so this test would see a wrong +1 path
        self.assertGreater(rounded, 0)

    def test_band_rows_never_reach_the_gguf(self) -> None:
        tensors = self.tensors(self.gguf)
        self.assertGreater(self.chimera["embed.weight"].shape[0], TEXT_VOCAB)
        for name in ("token_embd.weight", "output.weight"):
            self.assertEqual([p1t.HIDDEN, TEXT_VOCAB], [int(n) for n in tensors[name].shape])
        self.assertFalse([n for n in tensors if "nextn" in n])

    def test_inserted_charted_blocks_change_no_byte(self) -> None:
        directory = insert_charted(self.chimera, self.cfg, self.root / "charted", at=(2, 3))
        out = self.root / "out-charted"
        code, text = run_main(["convert", "--chimera", str(directory), "--parent-dir", str(self.hf_dir),
                               "--out-dir", str(out), "--tag", "tiny", "--min-free-gib", "0",
                               "--parent-map", "", "--work-dir", str(self.root)])
        self.assertEqual(0, code, text)
        self.assertTrue(e3.compare_ggufs(self.conv, out / "tiny.bf16.gguf", self.gguf)["equal"])

    def test_receipt_records_the_source_and_the_converter(self) -> None:
        body = json.loads((self.out / "e3-export-receipt-tiny.json").read_text(encoding="utf-8"))
        self.assertTrue(body["pass"])
        self.assertEqual(e3.TOOL_VERSION, body["tool"]["version"])
        self.assertIn("conversion/qwen.py", body["converter"]["converter_files"])
        self.assertEqual(LAYERS, body["layers"])
        self.assertFalse(body["truncated_test_artifact"])
        self.assertEqual(TEXT_VOCAB, body["source"]["reader_summary"]["base_vocab"])
        hfmeta = Path(body["hfmeta"]["directory"])
        self.assertFalse([p for p in hfmeta.iterdir() if p.name.endswith(e3.WEIGHT_SUFFIXES)])


class TestMapAndTruncation(Fixture):
    def test_map_is_headers_only_and_accounts_for_every_tensor(self) -> None:
        before = sorted(p.name for p in self.out.iterdir())
        receipt = self.root / "map.json"
        code, text = run_main(["map", "--chimera", str(self.chimera_dir), "--parent-dir", str(self.hf_dir),
                               "--parent-map", "", "--work-dir", str(self.root), "--receipt", str(receipt)])
        self.assertEqual(0, code, text)
        self.assertIn("P9 MAP PASS", text)
        body = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(3 + 3 * 14 + 11, body["gguf_tensor_count"])
        self.assertEqual({"0": 14, "1": 14, "2": 14, "3": 11}, body["facts"]["per_block_counts"])
        self.assertEqual(len(self.qwen), body["served_tensors"])
        self.assertFalse(body["wrote_files"])
        self.assertEqual(before, sorted(p.name for p in self.out.iterdir()))
        traced = [row for row in body["tensors"] if row["gguf"] == "blk.3.attn_q.weight"][0]
        self.assertEqual("interleave", traced["p1_kind"])
        self.assertEqual(["blocks.3.mixer.q_proj.weight", "blocks.3.mixer.gate_proj.weight"],
                         [part[0] for part in traced["chimera_parts"]])

    def test_parent_map_comparison_tolerates_only_the_ffn_width(self) -> None:
        tensors = self.tensors(self.stock)
        mapped = []
        for name, t in tensors.items():
            shape = [int(n) for n in t.shape]
            if name.endswith(("ffn_gate.weight", "ffn_up.weight")):
                shape[1] = 3 * p1t.WIDTH          # a wider parent FFN, as 17,408 vs 16,640
            if name.endswith("ffn_down.weight"):
                shape[0] = 3 * p1t.WIDTH
            mapped.append({"gguf": name, "shape": shape, "to": self.conv.gguf.GGMLQuantizationType(int(t.tensor_type)).name})
        mapped.append({"gguf": "blk.4.nextn.eh_proj.weight", "shape": [1, 1], "to": "BF16"})
        p1 = e3.load_p1()
        reader = p1.open_reader(self.chimera_dir)
        self.addCleanup(reader.close)
        table = [{"gguf": n, "shape": [int(x) for x in t.shape],
                  "type": self.conv.gguf.GGMLQuantizationType(int(t.tensor_type)).name}
                 for n, t in self.tensors(self.gguf).items()]
        good = e3.compare_with_parent_map(table, {"mapped": mapped}, reader, LAYERS)
        self.assertEqual([], good["problems"])
        self.assertEqual(3 * p1t.WIDTH, good["parent_ffn_width"])
        broken = [dict(row) for row in mapped if row["gguf"] != "blk.0.ssm_out.weight"]
        broken[0]["to"] = "F16"
        bad = e3.compare_with_parent_map(table, {"mapped": broken}, reader, LAYERS)
        self.assertEqual(["blk.0.ssm_out.weight"], bad["extra"])
        self.assertEqual(1, len(bad["differ"]))

    def test_truncation_keeps_the_first_layers_byte_for_byte(self) -> None:
        out = self.root / "out-l2"
        code, text = run_main(["convert", "--chimera", str(self.chimera_dir), "--parent-dir", str(self.hf_dir),
                               "--out-dir", str(out), "--tag", "tiny-l2", "--layers", "2", "--min-free-gib", "0",
                               "--parent-map", "", "--work-dir", str(self.root)])
        self.assertEqual(0, code, text)
        short = self.tensors(out / "tiny-l2.bf16.gguf")
        full = self.tensors(self.gguf)
        self.assertEqual({"0", "1"}, {n.split(".")[1] for n in short if n.startswith("blk.")})
        for name, tensor in short.items():
            self.assertEqual(e3._raw(full[name]), e3._raw(tensor), name)
        body = json.loads((out / "e3-export-receipt-tiny-l2.json").read_text(encoding="utf-8"))
        self.assertTrue(body["truncated_test_artifact"])
        self.assertEqual(2, body["kv"]["qwen35.block_count"])


class TestVerifyAndRefusals(Fixture):
    def test_verify_passes_and_catches_one_flipped_byte(self) -> None:
        receipt = self.root / "verify.json"
        code, text = run_main(["verify", "--gguf", str(self.gguf), "--chimera", str(self.chimera_dir),
                               "--parent-dir", str(self.hf_dir), "--work-dir", str(self.root),
                               "--receipt", str(receipt)])
        self.assertEqual(0, code, text)
        body = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertTrue(body["pass"])
        independent = body["checked_independently_against_chimera_bytes"]
        # 3 x (gate, up, down) + attn k/v/o + embed + head; 3 x 2 + q/k + final norms; 3 ssm norms
        self.assertEqual({"identity": 4 * 3 + 3 + 2, "norm": 4 * 2 + 2 + 1, "ssm_norm": 3}, independent)
        broken = self.root / "flipped.gguf"
        shutil.copyfile(self.gguf, broken)
        target = self.tensors(broken)["blk.1.ffn_down.weight"]
        with open(broken, "r+b") as handle:
            handle.seek(int(target.data_offset) + 7)
            byte = handle.read(1)
            handle.seek(int(target.data_offset) + 7)
            handle.write(bytes([byte[0] ^ 0x01]))
        code, text = run_main(["verify", "--gguf", str(broken), "--chimera", str(self.chimera_dir),
                               "--parent-dir", str(self.hf_dir), "--work-dir", str(self.root)])
        self.assertEqual(1, code, text)
        self.assertIn("blk.1.ffn_down.weight", text)

    def test_the_free_space_floor_and_an_existing_file_refuse_the_write(self) -> None:
        out = self.root / "out-floor"
        code, text = run_main(["convert", "--chimera", str(self.chimera_dir), "--parent-dir", str(self.hf_dir),
                               "--out-dir", str(out), "--tag", "floor", "--min-free-gib", "1e9",
                               "--parent-map", "", "--work-dir", str(self.root)])
        self.assertEqual(3, code, text)
        self.assertIn("floor", text)
        self.assertFalse([p for p in out.iterdir() if p.name.endswith((".gguf", ".partial"))])
        code, text = run_main(["convert", "--chimera", str(self.chimera_dir), "--parent-dir", str(self.hf_dir),
                               "--out-dir", str(self.out), "--tag", "tiny", "--min-free-gib", "0",
                               "--parent-map", "", "--work-dir", str(self.root)])
        self.assertEqual(3, code, text)
        self.assertIn("exists", text)

    def test_a_weight_file_in_the_staging_dir_is_refused(self) -> None:
        p1 = e3.load_p1()
        reader = p1.open_reader(self.chimera_dir)
        self.addCleanup(reader.close)
        staging = self.root / "stray-hfmeta"
        staging.mkdir()
        (staging / "model.safetensors").write_bytes(b"x")
        with self.assertRaises(e3.E3Error):
            e3.write_hfmeta(reader, self.hf_dir, staging, 0)


_Usage = collections.namedtuple("_Usage", "total used free")


def fake_disk(free_of) -> unittest.mock._patch:
    """`shutil.disk_usage` as the tool sees it, with `free_of(call_index)` free bytes."""

    calls = []

    def usage(path):
        calls.append(str(path))
        return _Usage(1 << 50, 0, int(free_of(len(calls) - 1)))

    patch = unittest.mock.patch.object(e3.shutil, "disk_usage", usage)
    patch.calls = calls
    return patch


class TestDiskFloor(Fixture):
    """1.0.1: the floor counts the llama-quantize outputs, and is re-checked during
    the bf16 write and before each llama-quantize call."""

    def convert(self, out: Path, tag: str, *extra: str) -> tuple[int, str]:
        return run_main(["convert", "--chimera", str(self.chimera_dir), "--parent-dir", str(self.hf_dir),
                         "--out-dir", str(out), "--tag", tag, "--min-free-gib", "0",
                         "--parent-map", "", "--work-dir", str(self.root), *extra])

    def bf16_estimate(self) -> int:
        body = json.loads((self.out / "e3-export-receipt-tiny.json").read_text(encoding="utf-8"))
        return int(body["estimated_bytes"])

    def test_quant_size_bound_follows_llama_quantize_rules(self) -> None:
        info = lambda shape, nbytes: types.SimpleNamespace(shape=shape, nbytes=nbytes)  # noqa: E731
        slack, pad = e3.GGUF_METADATA_SLACK, e3.GGUF_ALIGNMENT
        # a quantizable weight whose rows divide 256: the mix ceiling (Q6_K for Q4_K_M)
        self.assertEqual(slack + 1024 * 512 * 6.5625 / 8 + pad,
                         e3.quant_size_bound([("blk.0.ffn_up.weight", info([1024, 512], 1024 * 512 * 2))], "Q4_K_M"))
        # rows that divide 32 only -> Q8_0 at most; rows that do not divide 32 -> F16
        self.assertEqual(slack + 64 * 96 * 8.5 / 8 + pad,
                         e3.quant_size_bound([("blk.0.ffn_up.weight", info([64, 96], 64 * 96 * 2))], "Q4_K_M"))
        self.assertEqual(slack + 64 * 100 * 2 + pad,
                         e3.quant_size_bound([("blk.0.ffn_up.weight", info([64, 100], 64 * 100 * 2))], "Q4_K_M"))
        # left alone by llama-quantize: copied at the written size
        for name, shape in (("blk.0.attn_norm.weight", [512]), ("blk.0.ssm_conv1d.weight", [512, 4]),
                            ("blk.0.ssm_a", [512, 512]), ("blk.0.ffn_up.weight", [1, 512])):
            self.assertEqual(slack + 777 + pad, e3.quant_size_bound([(name, info(shape, 777))], "Q4_K_M"), name)
        # an unknown mix is costed as F16
        self.assertEqual(slack + 1024 * 512 * 2 + pad,
                         e3.quant_size_bound([("blk.0.ffn_up.weight", info([1024, 512], 0))], "NEW_MIX"))

    def test_a_real_q4_k_m_fits_its_bound(self) -> None:
        out = self.root / "out-q4"
        code, text = self.convert(out, "q4", "--quants", "Q4_K_M")
        self.assertEqual(0, code, text)
        body = json.loads((out / "e3-export-receipt-q4.json").read_text(encoding="utf-8"))
        written = body["artifacts"]["Q4_K_M"]["bytes"]
        bound = body["estimated_quant_bytes"]["Q4_K_M"]
        print(f"\n[q4 bound] tiny Q4_K_M {written} B <= bound {bound} B; bf16 {body['artifacts']['bf16']['bytes']} B"
              f" <= estimate {body['estimated_bytes']} B", file=sys.stderr)
        self.assertLessEqual(written, bound)
        self.assertLessEqual(body["artifacts"]["bf16"]["bytes"], body["estimated_bytes"])
        self.assertEqual(body["estimated_bytes"] + bound, body["estimated_total_bytes"])

    def test_the_floor_counts_the_requested_quants(self) -> None:
        """Room for the bf16 file but not for bf16 + Q4: 1.0.0 wrote it, 1.0.1 refuses."""

        free = self.bf16_estimate() + 1
        out = self.root / "out-floor-q"
        with fake_disk(lambda i: free):
            code, text = self.convert(out, "floorq", "--quants", "Q4_K_M")
        self.assertEqual(3, code, text)
        self.assertIn("quant bound", text)
        self.assertFalse([p for p in out.iterdir() if p.name.endswith((".gguf", ".partial"))])
        # the control: the same free space without --quants is enough
        with fake_disk(lambda i: free):
            code, text = self.convert(out, "floorq")
        self.assertEqual(0, code, text)

    def test_free_space_is_rechecked_before_each_quantize(self) -> None:
        out = self.root / "out-requant"
        final = out / "requant.bf16.gguf"
        with fake_disk(lambda i: 1 if final.exists() else 1 << 50):
            code, text = self.convert(out, "requant", "--quants", "Q4_K_M")
        self.assertEqual(1, code, text)
        self.assertIn("Q4_K_M not quantized", text)
        self.assertTrue(final.is_file())
        self.assertFalse((out / "requant.Q4_K_M.gguf").exists())
        body = json.loads((out / "e3-export-receipt-requant.json").read_text(encoding="utf-8"))
        stage = [s for s in body["stages"] if s["stage"] == "quantize:Q4_K_M"][0]
        self.assertEqual("free-space floor", stage["refused"])
        self.assertNotIn("cmd", stage)    # llama-quantize never ran

    def test_the_write_stops_when_the_disk_fills_mid_write(self) -> None:
        out = self.root / "out-midwrite"
        disk = fake_disk(lambda i: 1 << 50 if i == 0 else 0)
        with disk:
            code, text = self.convert(out, "midwrite")
        self.assertEqual(3, code, text)
        self.assertIn("while writing", text)
        self.assertGreater(len(disk.calls), 1)
        self.assertFalse([p for p in out.iterdir() if p.name.endswith((".gguf", ".partial"))])

    def test_an_unknown_quant_is_refused_before_anything_is_written(self) -> None:
        out = self.root / "out-badquant"
        with self.assertRaises(ValueError):
            self.convert(out, "badquant", "--quants", "Q9_ZZ")
        self.assertFalse([p for p in out.iterdir() if p.name.endswith((".gguf", ".partial"))])


class TestLlamaServer(Fixture):
    def test_llama_server_on_cpu_loads_the_gguf_and_its_logits_are_finite(self) -> None:
        receipt = self.root / "logits.json"
        code, text = run_main(["logits", "--gguf", str(self.gguf), "--chimera", str(self.chimera_dir),
                               "--parent-dir", str(self.hf_dir), "--layers", str(LAYERS),
                               "--tps-probe-dir", str(DISTILL_DIR), "--tokens", "12", "--threads", "4",
                               "--work-dir", str(self.root), "--receipt", str(receipt)])
        self.assertEqual(0, code, text)
        body = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertTrue(body["finite"])
        self.assertEqual(12, body["top1_agreement"]["positions"])
        print(f"\n  tiny: top-1 agreement {body['top1_agreement']}, max |dlogprob| "
              f"{body['max_abs_logprob_diff_on_llama_topk']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
