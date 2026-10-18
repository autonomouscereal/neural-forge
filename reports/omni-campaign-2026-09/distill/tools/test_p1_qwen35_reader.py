#!/usr/bin/env python3
"""Tests for `p1_qwen35_reader.py` (S0-P1, tool 1.0.5).  Server only: torch, safetensors,
transformers 5.12.1 and the chimera_omni tree the child was written with.

Run (the tree is named explicitly, never picked up by accident)::

    OC=/home/cereal/omni-campaign
    EF=$OC/import/rq27b/tree_ef1ffb9
    CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$OC/distill/tools \\
        /home/cereal/model-surgery-lab/venv/bin/python test_p1_qwen35_reader.py -v

The central test builds a tiny Qwen3.5-shaped HF checkpoint, imports it with the
REAL `chimera_omni.importing.qwen35.import_qwen35_state`, writes it as a sharded
Chimera checkpoint, and then requires the reader to be that importer's exact
inverse: re-importing what the reader serves gives back every Chimera byte.  The
streamed engine's own `StreamedLayerList` and HF's `Qwen3_5TextModel` then load
the reader with `strict=True`, and the delta overlay is driven end to end through
the real ledger (`oc_checkpoint.py register --delta-base` -> `delta-compose`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = os.environ.get("CHIMERA_OMNI_SRC", "")
if _SRC:
    sys.path.insert(0, str(Path(_SRC) / "src"))
LEDGER_DIR = Path(os.environ.get("OC_LEDGER_DIR", "") or HERE.parents[1] / "ledger")
DISTILL_DIR = Path(os.environ.get("OC_DISTILL_DIR", "") or HERE.parent)
sys.path.insert(0, str(LEDGER_DIR))
sys.path.insert(0, str(DISTILL_DIR))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import p1_qwen35_reader as p1  # noqa: E402
import chimera_omni  # noqa: E402
from chimera_omni.importing.profiles import qwen35_chimera_config  # noqa: E402
from chimera_omni.importing.qwen35 import import_qwen35_state  # noqa: E402

LAYERS = 4
HIDDEN = 64
WIDTH = 24      # uniform FFN width: 8 "trunk" + 16 "K3" channels, like 4,352 + 12,288
TRUNK, K3 = 8, 16
VOCAB = 40
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 32
V_HEADS, K_HEADS, K_DIM, V_DIM = 4, 2, 16, 16
BANDS = [("control", 8), ("music_codes", 16)]


def hf_config() -> dict:
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": HIDDEN, "intermediate_size": WIDTH, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "linear_num_value_heads": V_HEADS, "linear_num_key_heads": K_HEADS,
            "linear_key_head_dim": K_DIM, "linear_value_head_dim": V_DIM,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "layer_types": layer_types, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128, "hidden_act": "silu", "attn_output_gate": True,
            "attention_bias": False, "attention_dropout": 0.0, "tie_word_embeddings": False,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1],
                                "partial_rotary_factor": 0.25, "rope_theta": 10000.0,
                                "rope_type": "default"},
            "dtype": "bfloat16",
        },
    }


def qwen_state(seed: int = 7) -> dict[str, torch.Tensor]:
    """Random bf16 tensors under every qwen3_5 text name, with plausible norms."""

    gen = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    key_dim, value_dim = K_HEADS * K_DIM, V_HEADS * V_DIM
    t = "model.language_model."
    state = {
        f"{t}embed_tokens.weight": rand(VOCAB, HIDDEN),
        "lm_head.weight": rand(VOCAB, HIDDEN),
        f"{t}norm.weight": rand(HIDDEN, scale=0.3),
    }
    for i in range(LAYERS):
        q = f"{t}layers.{i}."
        state[f"{q}input_layernorm.weight"] = rand(HIDDEN, scale=0.3)
        state[f"{q}post_attention_layernorm.weight"] = rand(HIDDEN, scale=0.3)
        state[f"{q}mlp.gate_proj.weight"] = rand(WIDTH, HIDDEN)
        state[f"{q}mlp.up_proj.weight"] = rand(WIDTH, HIDDEN)
        state[f"{q}mlp.down_proj.weight"] = rand(HIDDEN, WIDTH)
        if i % 4 != 3:
            a = f"{q}linear_attn."
            state[f"{a}in_proj_qkv.weight"] = rand(2 * key_dim + value_dim, HIDDEN)
            state[f"{a}conv1d.weight"] = rand(2 * key_dim + value_dim, 1, 4)
            state[f"{a}in_proj_z.weight"] = rand(value_dim, HIDDEN)
            state[f"{a}in_proj_b.weight"] = rand(V_HEADS, HIDDEN)
            state[f"{a}in_proj_a.weight"] = rand(V_HEADS, HIDDEN)
            state[f"{a}dt_bias"] = rand(V_HEADS, scale=1.0)
            state[f"{a}A_log"] = rand(V_HEADS, scale=1.0)
            state[f"{a}norm.weight"] = (1.0 + rand(V_DIM, scale=0.1).float()).to(torch.bfloat16)
            state[f"{a}out_proj.weight"] = rand(HIDDEN, value_dim)
        else:
            s = f"{q}self_attn."
            state[f"{s}q_proj.weight"] = rand(HEADS * 2 * HEAD_DIM, HIDDEN)
            state[f"{s}k_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{s}v_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{s}o_proj.weight"] = rand(HIDDEN, HEADS * HEAD_DIM)
            state[f"{s}q_norm.weight"] = rand(HEAD_DIM, scale=0.3)
            state[f"{s}k_norm.weight"] = rand(HEAD_DIM, scale=0.3)
    return state


def chimera_cfg():
    return qwen35_chimera_config(
        hf_config(), ledger_bands=BANDS, max_sequence_length=128,
        metadata={"c10_trunk_intermediate_size": str(TRUNK), "c10_k3_channels": str(K3)},
    )


def write_chimera(directory: Path, params: dict[str, torch.Tensor], cfg) -> None:
    """A sharded `chimera_omni.checkpoint.v1` directory, two shards + index."""

    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(params)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for filename, chunk in halves.items():
        save_file({n: params[n].contiguous() for n in chunk}, str(directory / filename),
                  metadata={"format": "pt", "schema": "chimera_omni.checkpoint.v1"})
        weight_map.update({n: filename for n in chunk})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({
        "schema": "chimera_omni.checkpoint.v1", "chimera_commit": "tiny-test",
        "config": cfg.to_manifest(),
    }), encoding="utf-8")


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.cfg = chimera_cfg()
        cls.qwen = qwen_state()
        model, cls.receipt = import_qwen35_state(cls.qwen, cls.cfg, dtype=torch.bfloat16)
        cls.chimera = {n: p.detach().clone() for n, p in model.named_parameters()}
        cls.chimera_dir = cls.root / "chimera"
        write_chimera(cls.chimera_dir, cls.chimera, cls.cfg)
        cls.parent_dir = cls.root / "parent"
        cls.parent_dir.mkdir()
        save_file({k: v.contiguous() for k, v in cls.qwen.items()},
                  str(cls.parent_dir / "model.safetensors"))
        (cls.parent_dir / "config.json").write_text(json.dumps(hf_config()), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def reader(self, directory: Path | None = None, **kwargs) -> p1.Qwen35LayoutReader:
        reader = p1.Qwen35LayoutReader(directory or self.chimera_dir, **kwargs)
        self.addCleanup(reader.close)
        return reader

    def variant(self, name: str, edit) -> Path:
        """A copy of the tiny Chimera checkpoint with `edit(params)` applied."""

        params = {k: v.clone() for k, v in self.chimera.items()}
        cfg = edit(params)
        directory = self.root / name
        if directory.exists():
            shutil.rmtree(directory)
        write_chimera(directory, params, cfg or self.cfg)
        return directory


class TestInverse(Fixture):
    def test_tool_version(self) -> None:
        self.assertTrue(p1.TOOL_VERSION.startswith("p1_qwen35_reader 1."))
        print(f"\n  chimera_omni from {Path(chimera_omni.__file__).resolve().parents[1]}")

    def test_reader_is_the_exact_inverse_of_the_real_importer(self) -> None:
        reader = self.reader()
        served = {name: reader.get(name) for name in reader.names()}
        again, receipt = import_qwen35_state(served, self.cfg, dtype=torch.bfloat16)
        params = dict(again.named_parameters())
        written = receipt["written_parameters"]
        self.assertEqual(set(), set(self.receipt["written_parameters"]) ^ set(written))
        self.assertEqual([], receipt["skipped"]["other"])
        self.assertEqual(len(served), receipt["consumed_tensors"])
        checked = 0
        for key in written:
            name, _, rows = key.partition("[")
            mine = params[name].detach()
            want = self.chimera[name]
            if rows:  # "embed.weight[:base]" -- the band rows are zero-initialised
                mine, want = mine[: self.cfg.ledger_base_vocab_size], want[: self.cfg.ledger_base_vocab_size]
            self.assertTrue(torch.equal(mine, want), key)
            checked += 1
        self.assertEqual(len(written), checked)

    def test_every_non_norm_tensor_equals_the_original_qwen_tensor(self) -> None:
        reader = self.reader()
        for name in reader.names():
            got = reader.get(name)
            if reader.kind(name) == "one_plus":
                # the importer rounds 1 + w into bf16 once, so w itself is not
                # recoverable; what IS exact is (served + 1) == the stored c.
                cname = reader.specs[name]["parts"][0][0]
                self.assertTrue(torch.equal(got.float() + 1.0, self.chimera[cname].float()), name)
            else:
                self.assertEqual(self.qwen[name].dtype, got.dtype, name)
                self.assertTrue(torch.equal(got, self.qwen[name]), name)
        self.assertEqual(sorted(self.qwen), reader.names())

    def test_ffn_is_one_uniform_width_sliced_from_the_fused_tensor_without_a_copy(self) -> None:
        reader = self.reader()
        self.assertEqual(WIDTH, reader.ffn_width)
        self.assertEqual({"trunk": [0, TRUNK], "k3": [TRUNK, WIDTH]}, reader.channel_partition)
        fused = reader.shards.view("blocks.0.ffn.up.weight")
        gate = reader.get("model.language_model.layers.0.mlp.gate_proj.weight")
        up = reader.get("model.language_model.layers.0.mlp.up_proj.weight")
        for tensor in (gate, up):
            self.assertEqual(fused.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
        self.assertEqual(fused[0].data_ptr(), gate.data_ptr())
        self.assertEqual(fused[WIDTH].data_ptr(), up.data_ptr())
        views = [n for n in reader.names() if reader.kind(n) == "view"]
        for name in views:
            got = reader.get(name)
            base = reader.shards.view(reader.specs[name]["parts"][0][0])
            self.assertEqual(base.untyped_storage().data_ptr(), got.untyped_storage().data_ptr(), name)
        self.assertNotIn("gain", " ".join(reader.shards.names()))

    def test_dropping_pages_keeps_every_view_valid(self) -> None:
        reader = self.reader(page_budget_bytes=1)  # drop before every read
        first = {name: reader.get(name).clone() for name in reader.names()}
        held = reader.get("model.language_model.layers.0.mlp.gate_proj.weight")
        reader.drop_pages()
        self.assertGreater(reader.page_drops, len(first) // 2)
        self.assertTrue(torch.equal(held, first["model.language_model.layers.0.mlp.gate_proj.weight"]))
        for name, value in first.items():
            self.assertTrue(torch.equal(reader.get(name), value), name)

    def test_extras_and_coverage_account_for_every_chimera_element(self) -> None:
        reader = self.reader()
        coverage = reader.coverage()
        self.assertEqual({}, coverage["miscovered"])
        total = sum(t.numel() for t in self.chimera.values())
        self.assertEqual(total, coverage["chimera_elements"])
        self.assertEqual(total, coverage["mapped_elements"] + coverage["extra_elements"])
        self.assertIn("embed.weight[base:]", reader.extras())
        band = reader.extra("lm_head.weight[base:]")
        self.assertEqual([sum(w for _, w in BANDS), HIDDEN], list(band.shape))
        for name in reader.extras():
            self.assertFalse(name.startswith("model.language_model."), name)

    def test_qwen35_config_is_the_parent_with_the_child_width(self) -> None:
        config = self.reader().qwen35_config(hf_config())
        self.assertEqual(WIDTH, config["text_config"]["intermediate_size"])
        bad = hf_config()
        bad["text_config"]["vocab_size"] = VOCAB + 1
        with self.assertRaises(p1.P1Error):
            self.reader().qwen35_config(bad)


class TestEngines(Fixture):
    def _text_config(self, reader):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

        config = Qwen3_5Config.from_dict(reader.qwen35_config(hf_config()))
        return config.text_config

    def test_hf_qwen3_5_text_model_loads_the_reader_strict(self) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        reader = self.reader()
        model = Qwen3_5TextModel(self._text_config(reader))
        prefix = p1.QWEN_TEXT_PREFIX
        state = {n[len(prefix):]: reader.get(n) for n in reader.names() if n.startswith(prefix)}
        result = model.load_state_dict(state, strict=True)
        self.assertEqual(([], []), (list(result.missing_keys), list(result.unexpected_keys)))
        loaded = dict(model.named_parameters())
        for key, value in state.items():
            self.assertTrue(torch.equal(loaded[key].detach().to(value.dtype), value), key)

    def test_the_streamed_engines_layer_list_drives_the_reader_unchanged(self) -> None:
        import tps_probe
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        reader = self.reader()
        text_config = self._text_config(reader)
        layers_class = tps_probe.make_streamed_layer_list(Qwen3_5DecoderLayer)
        layers = layers_class(text_config, reader, tps_probe.TEXT_TRUNK_PREFIX + ".layers",
                              "cpu", torch.bfloat16)
        for index, layer in enumerate(layers[0:LAYERS]):  # the generator the forward uses
            prefix = f"{p1.QWEN_LAYER_PREFIX}{index}."
            params = dict(layer.named_parameters())
            for name in reader.names_under(prefix):
                want = reader.get(name)
                got = params[name[len(prefix):]].detach()
                # a bf16 engine holds what the reader served, cast once to bf16
                # (exact for every bf16-served tensor; fp32-served norms round)
                self.assertTrue(torch.equal(got, want.to(got.dtype)), name)
        self.assertEqual(LAYERS, layers.layer_loads)
        self.assertGreater(reader.bytes_read, 0)


    def test_tps_probe_builder_runs_unchanged_through_the_seam(self) -> None:
        """tps_probe's own `build_streamed_text_model` + forward, on the reader.

        Logits from the streamed engine (one layer resident, loaded through the
        reader) against a fully loaded HF model holding the same tensors.
        """

        import tps_probe
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        reader = self.reader()
        config_dir = self.root / "qwen35-config-only"
        if config_dir.exists():
            shutil.rmtree(config_dir)
        p1.write_qwen35_config(reader, self.parent_dir, config_dir)
        self.assertEqual(["config.json"], sorted(p.name for p in config_dir.iterdir()))
        with p1.tps_probe_seam(reader, tps_probe):
            model, head, info = tps_probe.build_streamed_text_model(
                config_dir, "cpu", torch.float32, "sdpa")
        self.assertTrue(isinstance(tps_probe.ShardReader, type))  # seam restored
        self.assertEqual("ShardReader", tps_probe.ShardReader.__name__)
        self.assertIs(reader, info["reader"])
        ids = torch.randint(0, VOCAB, (1, 12), generator=torch.Generator().manual_seed(3))
        # use_cache=False exactly as tps_probe calls it (lines 1203-1229, 1705): the
        # engine reuses ONE decoder buffer per layer type, built with that type's
        # first layer_idx, so under HF's default use_cache=True every linear layer
        # after the first reads layer 0's conv/recurrent cache slot.
        with torch.no_grad():
            streamed = head(model(input_ids=ids, use_cache=False).last_hidden_state)
        full = Qwen3_5TextModel(self._text_config(reader)).to(torch.float32)
        prefix = p1.QWEN_TEXT_PREFIX
        full.load_state_dict(
            {n[len(prefix):]: reader.get(n) for n in reader.names() if n.startswith(prefix)},
            strict=True)
        full.eval()
        with torch.no_grad():
            reference = full(input_ids=ids, use_cache=False).last_hidden_state @ reader.get(
                p1.QWEN_LM_HEAD).float().T
        self.assertTrue(bool(torch.isfinite(streamed).all()))
        self.assertTrue(torch.allclose(streamed, reference, atol=1e-5, rtol=1e-5),
                        float((streamed - reference).abs().max()))
        self.assertEqual(LAYERS, model.layers.layer_loads)
        # and the caveat above is real, not folklore: with the cache on, it diverges
        with torch.no_grad():
            cached = head(model(input_ids=ids).last_hidden_state)
        self.assertFalse(torch.allclose(cached, reference, atol=1e-3))


    def test_seam_smoke_cli_on_the_tiny_child(self) -> None:
        receipt = self.root / "smoke.json"
        config_dir = self.root / "smoke-config"
        if config_dir.exists():
            shutil.rmtree(config_dir)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = p1.main(["seam-smoke", "--chimera", str(self.chimera_dir),
                            "--parent-dir", str(self.parent_dir), "--config-dir", str(config_dir),
                            "--tps-probe-dir", str(DISTILL_DIR), "--forward-tokens", "6",
                            "--receipt", str(receipt), "--threads", "2"])
        self.assertEqual(0, code, out.getvalue())
        body = json.loads(receipt.read_text(encoding="utf-8"))
        p1.verify_sealed(body, "smoke receipt")
        self.assertTrue(body["pass"])
        self.assertEqual(len(self.qwen), body["tensors_loaded_and_compared"])
        self.assertTrue(body["forward"]["finite"])


class TestRefusals(Fixture):
    def test_unequal_output_norm_rows_are_refused_on_read(self) -> None:
        def edit(params):
            params["blocks.0.mixer.output_norm.weight"][1, 0] += 1.0

        reader = self.reader(self.variant("rows", edit))
        with self.assertRaises(p1.P1Error):
            reader.get("model.language_model.layers.0.linear_attn.norm.weight")

    def test_a_gain_tensor_in_the_ffn_is_refused(self) -> None:
        def edit(params):
            params["blocks.2.ffn.gain"] = torch.ones(WIDTH, dtype=torch.bfloat16)

        with self.assertRaises(p1.P1Error) as caught:
            self.reader(self.variant("gain", edit))
        self.assertIn("gain", str(caught.exception))

    def test_non_uniform_ffn_is_refused_unless_asked(self) -> None:
        def edit(params):
            params["blocks.1.ffn.up.weight"] = params["blocks.1.ffn.up.weight"][: 2 * 20].clone()
            params["blocks.1.ffn.down.weight"] = params["blocks.1.ffn.down.weight"][:, :20].clone()
            cfg = chimera_cfg()
            metadata = {k: v for k, v in dict(cfg.metadata).items() if not k.startswith("c10_")}
            return dataclasses.replace(cfg, metadata=metadata)

        directory = self.variant("ragged", edit)
        with self.assertRaises(p1.P1Error):
            self.reader(directory)
        reader = self.reader(directory, require_uniform_ffn=False)
        self.assertEqual([WIDTH, 20, WIDTH, WIDTH], reader.ffn_widths)
        self.assertIsNone(reader.ffn_width)

    def test_partition_metadata_that_does_not_sum_to_the_width_is_refused(self) -> None:
        def edit(params):
            cfg = chimera_cfg()
            metadata = dict(cfg.metadata, c10_k3_channels=str(K3 + 1))
            return dataclasses.replace(cfg, metadata=metadata)

        with self.assertRaises(p1.P1Error):
            self.reader(self.variant("partition", edit))

    def test_a_norm_whose_inverse_is_not_exact_even_in_fp32_is_refused(self) -> None:
        def edit(params):
            params["norm.weight"][3] = 1e-30  # c - 1 == -1 in fp32; -1 + 1 != c

        reader = self.reader(self.variant("norm", edit))
        with self.assertRaises(p1.P1Error):
            reader.get("model.language_model.norm.weight")

    def test_a_norm_served_in_fp32_is_recorded(self) -> None:
        def edit(params):
            params["norm.weight"][5] = 0.2  # bf16(0.2) - 1 needs more than 8 bits

        reader = self.reader(self.variant("norm32", edit))
        got = reader.get("model.language_model.norm.weight")
        self.assertEqual(torch.float32, got.dtype)
        self.assertEqual("float32", reader.norm_dtypes["model.language_model.norm.weight"])
        self.assertTrue(torch.equal(got + 1.0, reader.shards.view("norm.weight").float()))

    def test_block_kind_disagreeing_with_the_pattern_is_refused(self) -> None:
        def edit(params):
            return dataclasses.replace(chimera_cfg(), block_pattern=("kda", "kda", "gated_gqa", "kda"))

        with self.assertRaises(p1.P1Error):
            self.reader(self.variant("pattern", edit))

    def test_parity_cli_passes_on_the_tiny_child_and_fails_on_a_broken_one(self) -> None:
        receipt = self.root / "parity.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = p1.main(["parity", "--chimera", str(self.chimera_dir),
                            "--parent-dir", str(self.parent_dir), "--receipt", str(receipt),
                            "--threads", "2"])
        self.assertEqual(0, code, out.getvalue())
        body = json.loads(receipt.read_text(encoding="utf-8"))
        p1.verify_sealed(body, "receipt")
        self.assertTrue(body["pass"])
        self.assertEqual(len(self.qwen), body["tensors_checked"])
        self.assertTrue(all(t["equal_direct"] and t["equal_importer_round_trip"]
                            for t in body["per_tensor"]))

        broken = self.variant("broken-rows", lambda params: params[
            "blocks.0.mixer.output_norm.weight"].__setitem__((2, 1), 3.0))
        with contextlib.redirect_stdout(io.StringIO()):
            code = p1.main(["parity", "--chimera", str(broken),
                            "--parent-dir", str(self.parent_dir), "--threads", "2"])
        self.assertEqual(1, code)


class TestDeltaOverlay(Fixture):
    """Through the real ledger: register -> clone -> delta -> delta-compose -> read."""

    def setUp(self) -> None:
        import oc_checkpoint
        import oc_seal

        self.oc = oc_checkpoint
        self.oc_seal = oc_seal
        self.work = Path(tempfile.mkdtemp(dir=self.root))
        self.registry = self.work / "registry"
        self.ledger = self.work / "ledger.jsonl"
        base_dir = self.work / "omni1"
        shutil.copytree(self.chimera_dir, base_dir)

        def run(*argv: str) -> int:
            with contextlib.redirect_stdout(io.StringIO()):
                return oc_checkpoint.main(
                    [*argv, "--registry", str(self.registry), "--ledger", str(self.ledger)])

        self.run_oc = run
        self.assertEqual(0, run(
            "register", "--checkpoint-id", "PARENT-TINYOMNI", "--stage", "parent",
            "--directory", str(base_dir), "--parent-directory", str(base_dir),
            "--parent-provenance-sha256", "a" * 64, "--recipe-command", "tiny", "--confirm"))
        self.assertEqual(0, run(
            "clone", "--source", "PARENT-TINYOMNI", "--stage-about-to-run", "D",
            "--allow-unreviewed-source", "--confirm"))
        self.base_id = "PARENT-TINYOMNI.clone-pre-D1"
        self.base = oc_checkpoint.current(self.registry, self.base_id)["checkpoint"]
        gen = torch.Generator().manual_seed(11)
        self.rows = (torch.randn(TRUNK, HIDDEN, generator=gen) * 0.05).to(torch.bfloat16)
        self.lora_a = (torch.randn(2, K3, generator=gen) * 0.1).to(torch.bfloat16)
        self.lora_b = (torch.randn(HIDDEN, 2, generator=gen) * 0.1).to(torch.bfloat16)
        self.gain = 1.0 + torch.randn(K3, generator=gen) * 0.1
        self.delta_dir = self.work / "d1"
        self.delta_dir.mkdir()
        save_file({"trunk_gate_rows": self.rows, "k3_down.lora_a": self.lora_a,
                   "k3_down.lora_b": self.lora_b, "k3_down.gain": self.gain},
                  str(self.delta_dir / "delta.safetensors"))
        (self.delta_dir / "DELTA.json").write_text(json.dumps({
            "schema": oc_checkpoint.DELTA_RECIPE_SCHEMA,
            "base_checkpoint_id": self.base_id,
            "base_directory_sha256": self.base["directory_sha256"],
            "entries": [
                {"op": "replace", "target": "blocks.0.ffn.up.weight",
                 "tensor": "trunk_gate_rows", "slice": [[0, 0, TRUNK]]},
                {"op": "lora", "target": "blocks.0.ffn.down.weight", "a": "k3_down.lora_a",
                 "b": "k3_down.lora_b", "alpha": 4.0, "slice": [[1, TRUNK, WIDTH]]},
                {"op": "scale", "target": "blocks.0.ffn.down.weight", "tensor": "k3_down.gain",
                 "dim": 1, "slice": [[1, TRUNK, WIDTH]]},
            ],
        }), encoding="utf-8")
        self.assertEqual(0, run(
            "register", "--checkpoint-id", "OC02-D-Q38-27BU-TINYD1", "--stage", "distill",
            "--directory", str(self.delta_dir), "--delta-base", self.base_id,
            "--parent-directory", str(base_dir), "--parent-code", "Q38-27BU",
            "--recipe-command", "tiny-d1", "--confirm"))
        self.plan_path = self.work / "plan.json"
        self.assertEqual(0, run("delta-compose", "--checkpoint-id", "OC02-D-Q38-27BU-TINYD1",
                                "--deep", "--plan-out", str(self.plan_path)))
        self.plan = json.loads(self.plan_path.read_text(encoding="utf-8"))

    def test_composed_tensors_are_exactly_base_plus_delta(self) -> None:
        reader = self.reader(Path(self.base["directory"]), plan=self.plan)  # digest check by default
        check = reader.summary()["root_check"]
        self.assertEqual("directory_sha256", check["mode"])
        self.assertEqual(self.base["directory_sha256"], check["live_directory_sha256"])
        self.assertTrue(check["same_path"])
        gate = reader.get("model.language_model.layers.0.mlp.gate_proj.weight")
        base_up = self.chimera["blocks.0.ffn.up.weight"]
        self.assertTrue(torch.equal(gate[:TRUNK], self.rows))
        self.assertTrue(torch.equal(gate[TRUNK:], base_up[TRUNK:WIDTH]))
        up = reader.get("model.language_model.layers.0.mlp.up_proj.weight")
        self.assertTrue(torch.equal(up, base_up[WIDTH:]))
        want = self.chimera["blocks.0.ffn.down.weight"].float()
        want[:, TRUNK:WIDTH] += (4.0 / 2) * (self.lora_b.float() @ self.lora_a.float())
        want[:, TRUNK:WIDTH] *= self.gain.view(1, -1)
        down = reader.get("model.language_model.layers.0.mlp.down_proj.weight")
        self.assertTrue(torch.equal(down, want.to(torch.bfloat16)))
        self.assertTrue(torch.equal(down[:, :TRUNK], self.chimera["blocks.0.ffn.down.weight"][:, :TRUNK]))
        # untouched tensors are still zero-copy views of the base
        other = reader.get("model.language_model.layers.1.mlp.down_proj.weight")
        self.assertEqual(reader.shards.view("blocks.1.ffn.down.weight").untyped_storage().data_ptr(),
                         other.untyped_storage().data_ptr())
        self.assertEqual(self.plan["receipt_sha"], reader.summary()["delta_plan"])
        # the base itself was not touched
        self.assertTrue(torch.equal(reader.shards.view("blocks.0.ffn.up.weight"), base_up))

    def test_a_plan_for_other_bytes_is_refused(self) -> None:
        # one extra NON-FFN tensor: a checkpoint the reader opens fine on its own,
        # whose shards are simply not the ones the plan is about
        other = self.variant("other-root", lambda params: params.__setitem__(
            "native_extra.weight", torch.zeros(3, dtype=torch.bfloat16)) or None)
        self.reader(other)
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(other, plan=self.plan)
        self.assertIn("not the base the plan is about", str(caught.exception))

    def test_a_delta_file_changed_after_the_plan_is_refused(self) -> None:
        target = self.delta_dir / "delta.safetensors"
        os.chmod(target, 0o644)
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 0x01
        target.write_bytes(bytes(raw))
        with self.assertRaises(p1.P1Error):
            self.reader(Path(self.base["directory"]), plan=self.plan)

    def test_an_edited_plan_is_refused(self) -> None:
        edited = json.loads(json.dumps(self.plan))
        edited["deltas"][0]["entries"][1]["alpha"] = 8.0
        with self.assertRaises(p1.P1Error):
            self.reader(Path(self.base["directory"]), plan=edited)

    def test_a_base_whose_bytes_moved_in_place_is_refused_by_default(self) -> None:
        # the clone-rewritten-in-place case: same path, same names, same sizes
        shard = sorted(Path(self.base["directory"]).glob("*.safetensors"))[0]
        os.chmod(shard, 0o644)
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 0x01
        shard.write_bytes(bytes(raw))
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(Path(self.base["directory"]), plan=self.plan)
        self.assertIn("BASE DIGEST MISMATCH", str(caught.exception))
        # and the ledger agrees: the pin is broken at the bytes
        self.assertEqual(3, self.run_oc("delta-compose", "--checkpoint-id",
                                        "OC02-D-Q38-27BU-TINYD1", "--deep"))

    def _same_shape_other_root(self) -> Path:
        """The review's DEMO1: a sibling whose names, shapes and sizes all match."""

        def edit(params):
            params["blocks.0.ffn.up.weight"] = -params["blocks.0.ffn.up.weight"]

        other = self.variant("same-shape-other", edit)
        sizes = lambda d: {f.name: f.stat().st_size for f in d.glob("*.safetensors")}  # noqa: E731
        self.assertEqual(sizes(Path(self.base["directory"])), sizes(other))
        return other

    def test_a_same_shape_root_with_other_bytes_is_refused(self) -> None:
        other = self._same_shape_other_root()
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(other, plan=self.plan)
        self.assertIn("BASE DIGEST MISMATCH", str(caught.exception))
        # and the opt-out does not reopen the hole: without a re-hash, the opened
        # directory must BE the plan's root, and this one is not
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(other, plan=self.plan, verify_root_sha=False)
        self.assertIn("is not the plan's root directory", str(caught.exception))

    def test_the_opt_out_on_the_true_root_is_accepted_and_recorded(self) -> None:
        reader = self.reader(Path(self.base["directory"]), plan=self.plan, verify_root_sha=False)
        check = reader.summary()["root_check"]
        self.assertEqual("path_identity_opt_out", check["mode"])
        self.assertIsNone(check["live_directory_sha256"])
        self.assertTrue(check["same_path"])

    def test_a_byte_identical_copy_elsewhere_passes_the_digest_and_records_both_paths(self) -> None:
        copy = self.work / "copy-of-base"
        shutil.copytree(self.base["directory"], copy)
        reader = self.reader(copy, plan=self.plan)
        check = reader.summary()["root_check"]
        self.assertFalse(check["same_path"])
        self.assertEqual(str(copy.resolve()), check["opened_directory"])
        self.assertEqual(self.base["directory_sha256"], check["live_directory_sha256"])

    def test_the_readers_digest_walk_is_oc_seals(self) -> None:
        # the restated contract, asserted against its source
        self.assertEqual(self.oc_seal.DIGEST_EXCLUDE_SUFFIXES, p1.DIGEST_EXCLUDE_SUFFIXES)
        self.assertEqual(self.oc_seal.DIGEST_EXCLUDE_DIRS, p1.DIGEST_EXCLUDE_DIRS)
        odd = self.work / "odd-tree"
        shutil.copytree(self.chimera_dir, odd)
        (odd / "sub" / "deeper").mkdir(parents=True)
        (odd / "sub" / "deeper" / "note.txt").write_text("x", encoding="utf-8")
        (odd / "half.safetensors.partial").write_bytes(b"skip me")
        (odd / "__pycache__").mkdir()
        (odd / "__pycache__" / "m.pyc").write_bytes(b"skip me too")
        for directory in (Path(self.base["directory"]), self.chimera_dir, odd):
            self.assertEqual(self.oc_seal.directory_digest(directory, allow_hardlinks=True)[0],
                             p1.directory_digest(directory)[0], directory)

    def _parity(self, chimera: Path, *extra: str) -> tuple[int, dict | None, str]:
        receipt = self.work / f"parity-{len(list(self.work.glob('parity-*')))}.json"
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = p1.main(["parity", "--chimera", str(chimera), "--parent-dir", str(self.parent_dir),
                            "--plan", str(self.plan_path), "--receipt", str(receipt),
                            "--threads", "2", *extra])
        body = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else None
        return code, body, out.getvalue() + err.getvalue()

    def test_parity_with_a_plan_passes_on_the_true_base(self) -> None:
        code, body, text = self._parity(Path(self.base["directory"]))
        self.assertEqual(0, code, text)
        p1.verify_sealed(body, "receipt")
        self.assertTrue(body["pass"])
        self.assertEqual(self.plan["receipt_sha"], body["plan"])
        self.assertEqual("directory_sha256", body["summary"]["root_check"]["mode"])
        touched = {t["name"] for t in body["per_tensor"] if t["delta_composed"]}
        self.assertEqual({f"model.language_model.layers.0.mlp.{n}.weight"
                          for n in ("gate_proj", "up_proj", "down_proj")}, touched)
        self.assertEqual(3, body["delta_composed_tensors"])
        self.assertTrue(all(t["equal_direct"] and t["equal_importer_round_trip"]
                            for t in body["per_tensor"]))
        # every untouched view is still zero-copy under the plan
        self.assertTrue(all(t["zero_copy"] for t in body["per_tensor"]
                            if t["kind"] == "view" and not t["delta_composed"]))
        code, body, text = self._parity(Path(self.base["directory"]), "--no-verify-root-sha")
        self.assertEqual(0, code, text)
        self.assertEqual("path_identity_opt_out", body["summary"]["root_check"]["mode"])

    def test_parity_with_a_plan_refuses_a_wrong_root(self) -> None:
        other = self._same_shape_other_root()
        for extra in ((), ("--no-verify-root-sha",)):
            code, body, text = self._parity(other, *extra)
            self.assertEqual(3, code, text)
            self.assertIsNone(body)
            self.assertIn("REFUSED", text)

    def test_parity_with_a_plan_catches_a_wrong_composition(self) -> None:
        # the reference is independent of DeltaOverlay: a composition that is off
        # by one element on one touched tensor fails parity loudly
        original = p1.DeltaOverlay.compose

        def off_by_one(overlay, name, base):
            out = original(overlay, name, base).clone()
            if name == "blocks.0.ffn.down.weight":
                out[0, TRUNK] += 1.0
            return out

        with mock.patch.object(p1.DeltaOverlay, "compose", off_by_one):
            code, body, text = self._parity(Path(self.base["directory"]))
        self.assertEqual(1, code, text)
        failed = [t["name"] for t in body["per_tensor"] if not t["ok"]]
        self.assertEqual(["model.language_model.layers.0.mlp.down_proj.weight"], failed)


#: a stand-in media unit: `charted.*` tensors only, like c16_install_media writes
CHARTED_TENSORS = {
    "charted.chart_in.weight": (16, HIDDEN), "charted.chart_in.bias": (16,),
    "charted.chart_out.weight": (HIDDEN, 16), "charted.mlp.fc1.weight": (32, 16),
}


class TestChartedBlocks(Fixture):
    """1.0.5: OMNI1 has media blocks INSERTED between text blocks (32-36 of 69)."""

    def inserted(self, name: str, at: tuple[int, ...] = (2, 3), stray: str | None = None,
                 listed: list[int] | None = None) -> Path:
        """The tiny child with charted blocks inserted at Chimera indices `at`."""

        total = LAYERS + len(at)
        text_of = [b for b in range(total) if b not in at]
        gen = torch.Generator().manual_seed(11)
        params: dict[str, torch.Tensor] = {}
        for tname, tensor in self.chimera.items():
            if tname.startswith("blocks."):
                _, index, rest = tname.split(".", 2)
                params[f"blocks.{text_of[int(index)]}.{rest}"] = tensor.clone()
            else:
                params[tname] = tensor.clone()
        for block in at:
            for tname, shape in CHARTED_TENSORS.items():
                params[f"blocks.{block}.{tname}"] = (torch.randn(*shape, generator=gen) * 0.05).to(torch.bfloat16)
        if stray:
            params[f"blocks.{at[0]}.{stray}"] = self.chimera[f"blocks.0.{stray}"].clone()
        directory = self.root / name
        if directory.exists():
            shutil.rmtree(directory)
        write_chimera(directory, params, self.cfg)
        path = directory / "config.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        cfg = document["config"]
        pattern = list(cfg["block_pattern"])
        text_kinds = iter(pattern[t % len(pattern)] for t in range(LAYERS))
        cfg["block_pattern"] = ["charted" if b in at else next(text_kinds) for b in range(total)]
        cfg["num_layers"] = total
        cfg["charted_blocks"] = [{"layer_index": b, "component": f"tiny.unit{b}", "source": "tiny"}
                                 for b in (at if listed is None else listed)]
        path.write_text(json.dumps(document), encoding="utf-8")
        return directory

    def test_text_layers_skip_the_inserted_blocks_and_stay_tensor_exact(self) -> None:
        reader = self.reader(self.inserted("charted"))
        base = self.reader()
        self.assertEqual(LAYERS, reader.num_layers)
        self.assertEqual(LAYERS + 2, reader.num_blocks)
        self.assertEqual([0, 1, 4, 5], reader.text_blocks)
        self.assertEqual([2, 3], reader.charted_blocks)
        self.assertEqual(base.names(), reader.names())
        for name in reader.names():
            got, want = reader.get(name), base.get(name)
            self.assertEqual(want.dtype, got.dtype, name)
            self.assertTrue(torch.equal(got, want), name)

    def test_charted_tensors_are_extras_and_every_element_is_accounted_for(self) -> None:
        reader = self.reader(self.inserted("charted-extras"))
        for block in (2, 3):
            for tname in CHARTED_TENSORS:
                self.assertIn(f"blocks.{block}.{tname}", reader.extras())
        summary = reader.summary()
        self.assertEqual({}, summary["coverage"]["miscovered"])
        self.assertEqual([2, 3], [item["block"] for item in summary["charted_blocks"]])
        self.assertEqual([0, 1, 4, 5], summary["text_blocks"])

    def test_parity_cli_passes_on_an_inserted_child(self) -> None:
        receipt = self.root / "parity-charted.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = p1.main(["parity", "--chimera", str(self.inserted("charted-parity")),
                            "--parent-dir", str(self.parent_dir), "--receipt", str(receipt),
                            "--threads", "2"])
        self.assertEqual(0, code, out.getvalue())
        body = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertTrue(body["pass"])
        self.assertEqual(len(self.qwen), body["tensors_checked"])

    def test_a_text_tensor_inside_a_charted_block_is_refused(self) -> None:
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(self.inserted("charted-stray", stray="ffn.up.weight"))
        self.assertIn("charted.* only", str(caught.exception))

    def test_a_charted_list_disagreeing_with_the_pattern_is_refused(self) -> None:
        with self.assertRaises(p1.P1Error) as caught:
            self.reader(self.inserted("charted-listed", listed=[2, 4]))
        self.assertIn("ambiguous", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
