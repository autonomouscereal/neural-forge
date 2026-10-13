"""Standalone convention tests; genuine encoder inference belongs to parent proof."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_audio_chart", ROOT / "experiments/chimera_recovery/h3_audio_chart.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
PINS = {name: str(i) * 64 for i, name in enumerate(("stage_c", "comfy_audio_vae", "comfy_sd", "charted_attention"), 1)}


def chart(answer_count=2, **kwargs):
    # Four prompt text slots, eighty channel-major media slots, then answers.
    return C.chart(4, 40, 84 + answer_count, list(range(4, 84)), [0, 1, 2, 3] + list(range(84, 84 + answer_count)),
                   clean_encoder_receipt_sha256="a" * 64, amendment_sha256="b" * 64,
                   source_pins=PINS, **kwargs)


def proof():
    return {"schema": "chimera_recovery.h3_standalone_audio_probe.v1", "probe_passed": True,
            "teacher_unchanged": True, "model_space_verified": True,
            "input_contract": {"target_rate": 32000, "target_samples": 32000, "duration_seconds": 1.0,
                               "no_input_padding_or_trim": True},
            "measurement": {"raw_latent_shape": [1,32,2,40], "packed_shape": [80,32],
                            "decoded_shape": [1,2,32000], "measured_samples_per_latent_frame": 800,
                            "nominal_hop_seconds": .025, "encoder_repeat_bitwise_equal": True,
                            "channel_major_pack_bitwise_invertible": True, "latent_padded_frames": 0,
                            "latent_trimmed_frames": 0,
                            "nominal_frame_interval_seconds": [[i/40,(i+1)/40] for i in range(40)]},
            "decode": {"waveform_padded_samples": 0, "waveform_trimmed_samples": 0,
                       "first_pts_seconds": 0., "last_end_seconds": 1., "target_samples": 32000,
                       "resampler_flushed": True}}


class ChartTests(unittest.TestCase):
    def test_measured_time_and_stereo_row_identity(self):
        result = chart()
        positions = result["native_media_positions"][0]
        self.assertEqual(len(positions), 80)
        self.assertEqual(positions[0], [.0125,0.,.025,0.])
        self.assertEqual(positions[39], [.9875,0.,.025,0.])
        self.assertEqual(positions[40], [.0125,0.,.025,1.])
        self.assertEqual(positions[79], [.9875,0.,.025,1.])
        self.assertFalse(result["contract"]["native_parent_xyz_parity"])
        self.assertFalse(result["contract"]["native_admission_granted"])
        self.assertEqual(result["contract"]["spec_type_ids"], [2,3])

    def test_real_slot_scatter_and_prefix_invariance(self):
        a, b = chart(2), chart(9)
        self.assertEqual(a["charted_source_positions"][0][:84], b["charted_source_positions"][0][:84])
        self.assertEqual(a["h3_prompt_read_mask"][0], [True]*84+[False]*2)
        self.assertEqual(a["charted_source_positions"][0][4], [4.,0.,0.])
        self.assertEqual(a["charted_source_positions"][0][44], [4.,0.,1.])
        # Actual interleaved prompt slots retain text ordinals and channel order.
        media = [i for i in range(84) if i not in [0,10,50,83]]
        result = C.chart(4,40,85,media,[0,10,50,83,84],clean_encoder_receipt_sha256="a"*64,
                         amendment_sha256="b"*64,source_pins=PINS)
        self.assertEqual(result["charted_source_positions"][0][50], [2.,0.,0.])
        self.assertEqual(result["charted_source_positions"][0][media[40]], [4.,0.,1.])
        self.assertFalse(result["h3_prompt_read_mask"][0][84])

    def test_reject_incomplete_duplicate_future_or_unsealed_mapping(self):
        args = (4,40,86,list(range(4,84)),[0,1,2,3,84,85])
        cases = [(4,39,*args[2:]), (4,40,86,[4]*80,args[4]),
                 (4,40,86,list(range(5,85)),[0,1,2,3,4,85]),
                 (4,40,87,args[3],args[4])]
        for case in cases:
            with self.subTest(case=str(case)[:70]), self.assertRaises(ValueError):
                C.chart(*case,clean_encoder_receipt_sha256="a"*64,amendment_sha256="b"*64,source_pins=PINS)
        with self.assertRaises(ValueError):
            C.chart(*args,clean_encoder_receipt_sha256="",amendment_sha256="b"*64,source_pins=PINS)
        with self.assertRaises(ValueError):
            C.chart(*args,clean_encoder_receipt_sha256="a"*64,amendment_sha256="b"*64,source_pins={})

    def test_actual_proof_scope_and_raw_space_refusal(self):
        self.assertTrue(C.validate_encoder_proof(proof()))
        mutations = [("model_space_verified",False), ("probe_passed",False)]
        for key,value in mutations:
            doc = proof(); doc[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_encoder_proof(doc)
        for section,key,value in [("decode","waveform_trimmed_samples",1),
                                  ("measurement","latent_padded_frames",1),
                                  ("measurement","raw_latent_shape",[1,32,2,37]),
                                  ("decode","last_end_seconds",22/24)]:
            doc = proof(); doc[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_encoder_proof(doc)
        doc=proof();doc["measurement"]["nominal_frame_interval_seconds"][20]=[.51,.525]
        with self.assertRaises(ValueError):
            C.validate_encoder_proof(doc)


try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch CPU check runs on parent's supported runtime")
class ActualAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("CHIMERA_AUDIO_CHARTED_SOURCE", "C:/Users/cereal/Documents/ai training/chimera-omni/src/chimera_omni/modeling/charted.py"))
        if not path.is_file():
            raise unittest.SkipTest("source-pinned genuine charted attention not mounted")
        text=path.read_text(encoding="utf-8")
        # Exact reviewed source; CRLF/LF normalization is recorded independently.
        if hashlib.sha256(path.read_bytes()).hexdigest() not in {
                "0ee1875895a192cc4ce0dcd469a7cc2e9014b29b68ed3d8c4be978f24828ddc3",  # Reviewed Windows CRLF.
                "eaf590abf619e5dafab38b0d940d27dc79ef1d8a3eceb4db72843162fb2391b6"}:  # Exact same source with LF.
            raise AssertionError("genuine charted attention source pin mismatch")
        parsed=ast.parse(text)
        nodes=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)]
        nodes += [n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name in {"HeadRMSNorm","ChartedAttention"}]
        space={"torch":torch,"nn":torch.nn,"F":F,"ChartedBlockSpec":object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),space)
        cls.Attention=space["ChartedAttention"]

    def make_attention(self):
        cfg=types.SimpleNamespace(width=8,heads=1,kv_heads=1,head_dim=8,rope_theta=0.,sliding_window=0,
                                  modulation="h3_curve6",h3_rope_freqs=1,attn_bias=False,qk_norm=True,norm_eps=1e-6)
        torch.manual_seed(21)
        layer=self.Attention(cfg)
        with torch.no_grad():
            layer.rope_inv_freq.fill_(.7)
        return layer

    def test_genuine_attention_angles_match_independent_rotation(self):
        layer=self.make_attention(); layout=chart(2)
        hidden=torch.randn(1,86,8); xyz=torch.tensor(layout["charted_source_positions"])
        valid=torch.tensor(layout["h3_prompt_read_mask"])
        seen={}; original=F.scaled_dot_product_attention
        def observe(q,k,v,**kwargs):
            seen["q"]=q.detach().clone()
            return original(q,k,v,**kwargs)
        with patch.object(F,"scaled_dot_product_attention",observe):
            layer(hidden,valid,xyz)
        values=layer.q_proj(hidden).view(1,86,1,8).float()
        values=values*torch.rsqrt(values.square().mean(-1,keepdim=True)+1e-6)*layer.q_norm.weight.float()
        # Source times and categorical channels independently define three angles.
        angle=torch.stack((xyz[...,0]*.7,xyz[...,1]*.7,xyz[...,2]*.7),dim=-1).unsqueeze(2)
        left,right=values[...,:3],values[...,3:6]
        expected=torch.cat((left*angle.cos()-right*angle.sin(),right*angle.cos()+left*angle.sin(),values[...,6:]),dim=-1)
        torch.testing.assert_close(seen["q"],expected.transpose(1,2),rtol=1e-6,atol=1e-6)
        self.assertEqual(float(angle[0,44,0,2]),float(torch.tensor(.7)))

    def test_excluded_answers_cannot_change_prompt_attention_or_gradients(self):
        layer=self.make_attention(); layout=chart(2)
        xyz=torch.tensor(layout["charted_source_positions"]); valid=torch.tensor(layout["h3_prompt_read_mask"])
        hidden=torch.randn(1,86,8,requires_grad=True)
        before=layer(hidden,valid,xyz)
        changed=hidden.detach().clone();changed[:,84:]+=10000
        changed_xyz=xyz.clone();changed_xyz[:,84:]=100000
        after=layer(changed,valid,changed_xyz)
        torch.testing.assert_close(before[:,:84],after[:,:84],rtol=0,atol=0)
        before[:,:84].square().sum().backward()
        self.assertEqual(float(hidden.grad[:,84:].abs().sum()),0.)
        self.assertGreater(float(layer.q_proj.weight.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(layer.q_proj.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()
