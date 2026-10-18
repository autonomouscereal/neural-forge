"""Tests for the fold research lane tools.  CPU only:  CUDA_VISIBLE_DEVICES= nice -n 15 $PY -m unittest test_fold_tools -v

Pure-logic tests need no torch.  The integration tests build a TINY random qwen3_5 text model (2 layers: one GDN,
one gated attention) with the real Qwen3.5-0.8B tokenizer (env FOLD_TEST_TOKENIZER, default the server path) and
drive generate / overlay / both losses / fold_eval run+grade+pair / x3 train with a STOP-file preemption + resume.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fold_common as fc  # noqa: E402
import fold_eval  # noqa: E402

TOKENIZER_DIR = Path(os.environ.get("FOLD_TEST_TOKENIZER", "/home/cereal/models/Qwen3.5-0.8B"))
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class PureLogic(unittest.TestCase):
    def test_vendor_pins(self):
        pins = fc.verify_vendor_pins()
        self.assertEqual(pins["chimera_recovery/evaluation.py"][:8], "6a84626d")

    def test_screen_rules(self):
        s = fc.Screen()
        long_item = "compute the integer product of minus seventeen and six then report the final numeric answer on the last line please"
        s.add("codex", "a", long_item)
        s.add("bank", "b", "Hi! I had a long day today.")
        self.assertTrue(any(h["rule"] == "13gram>50%" for h in s.hits("Preamble. " + long_item + " trailing words")))
        self.assertTrue(any(h["rule"] == "exact" for h in s.hits(long_item.upper())))
        self.assertTrue(any(h["rule"] == "short-contained" for h in s.hits("so: hi i had a long day today, ok")))
        half = " ".join(long_item.split()[:10])
        self.assertEqual(s.hits(half + " completely different continuation about weather in paris"), [])

    def test_tool_parser_shapes(self):
        cases = {
            '[{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]': ("whole_json", 2),
            '{"name": "f", "parameters": {"x": 1}}': ("whole_json", 1),
            'Sure:\n```json\n{"tool_calls": [{"function": {"name": "f", "arguments": "{\\"x\\": 3}"}}]}\n```': ("whole_json", 1),
            '[{"f": {"x": 1}}]': ("whole_json", 1),
            '<tool_call>\n<function=f>\n<parameter=x>\n4\n</parameter>\n</function>\n</tool_call>': ("qwen_xml", 1),
            'The call is {"name": "f", "arguments": {"x": 5}} as requested.': ("first_json", 1),
        }
        for text, (route, n) in cases.items():
            got = fc.parse_tool_calls(text)
            self.assertEqual((got["route"], len(got["calls"])), (route, n), text)
        self.assertIsNone(fc.parse_tool_calls("I cannot help with that.")["calls"])
        self.assertEqual(fc.parse_tool_calls('<tool_call>\n<function=f>\n<parameter=x>\n4\n</parameter>\n</function>\n</tool_call>')["calls"][0]["arguments"], {"x": 4})

    def test_tool_grading_parallel_any_order(self):
        task = {"id": "t", "route": "tool_parallel", "input": {"messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "function", "function": {"name": "tax", "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}, "amount": {"type": "number"}}, "required": ["city", "amount"]}}}]},
                "oracle": {"kind": "tool", "steps": [{"parallel": True, "calls": [
                    {"name": "tax", "argument_choices": {"city": ["Chicago"], "amount": [30.45]}, "optional_arguments": []},
                    {"name": "tax", "argument_choices": {"city": ["Portland"], "amount": [11.23]}, "optional_arguments": []}]}]}}
        good = '[{"name":"tax","arguments":{"city":"Portland","amount":11.23}},{"name":"tax","arguments":{"city":"Chicago","amount":30.45}}]'
        bad = '[{"name":"tax","arguments":{"city":"Chicago","amount":30.45}}]'
        row = lambda t: {"text": t, "stop_reason": "eos", "generated_ids": [1, 2, 3]}
        self.assertEqual(fold_eval.grade_semantic_task(task, row(good))["class"], "pass")
        self.assertEqual(fold_eval.grade_semantic_task(task, row(bad))["class"], "content_miss")

    def test_arithmetic_artifact_tolerance(self):
        task = {"id": "a", "route": "text_arithmetic", "oracle": {"kind": "arithmetic", "expected": "-102", "absolute_tolerance": "0"}}
        row = lambda t, s="eos": {"text": t, "stop_reason": s, "generated_ids": list(range(5 if s == "eos" else 512))}
        self.assertEqual(fold_eval.grade_semantic_task(task, row("-17*6\n-102"))["class"], "pass")
        self.assertEqual(fold_eval.grade_semantic_task(task, row("Work.\n**Final answer: -102.**"))["class"], "artifact_pass")
        self.assertEqual(fold_eval.grade_semantic_task(task, row("the answer is -102 but\nthen 7"))["class"], "content_miss")
        self.assertEqual(fold_eval.grade_semantic_task(task, row("thinking forever", "max_tokens"))["class"], "artifact_miss")
        self.assertEqual(fold_eval.grade_semantic_task(task, row("<think>\n\n</think>\n\n-102"))["class"], "pass")

    def test_loop_detector(self):
        natural = list(range(200))
        self.assertIsNone(fc.detect_loop(natural))
        looped = list(range(70)) + [5, 6, 7] * 30
        end = fc.detect_loop(looped)
        self.assertIsNotNone(end)
        self.assertTrue(70 < end <= 70 + 32 + 4)
        self.assertIsNone(fc.detect_loop([1, 2, 3] * 15))          # shorter than 64 tokens
        self.assertIsNotNone(fc.detect_loop(list(range(64)) + [9] * 40))

    def test_horizon_controller(self):
        c = fc.HorizonController(start=256, low=128, high=1024)
        for _ in range(6):
            c.update(16, 0, 16)
        self.assertEqual(c.budget, 1024)
        for _ in range(12):
            c.update(16, 16, 0)
        self.assertEqual(c.budget, 128)
        c2 = fc.HorizonController()
        c2.load(c.state())
        self.assertEqual(c2.state(), c.state())

    def test_mcnemar(self):
        self.assertEqual(fc.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(fc.mcnemar_exact(0, 2), 0.5)
        self.assertAlmostEqual(fc.mcnemar_exact(0, 6), 0.03125)
        self.assertAlmostEqual(fc.mcnemar_exact(6, 0), 0.03125)


@unittest.skipUnless(HAVE_TORCH, "torch missing")
class Losses(unittest.TestCase):
    def test_bucket_kl_matches_full_kl_when_tail_is_one_token(self):
        import torch
        torch.manual_seed(0)
        t, s = torch.randn(7, 50), torch.randn(7, 50)
        full = (t.softmax(-1) * (t.log_softmax(-1) - s.log_softmax(-1))).sum(-1)
        self.assertTrue(torch.allclose(fc.bucket_forward_kl(s, t, k=49), full, atol=1e-5))
        self.assertTrue(torch.allclose(fc.bucket_forward_kl(t, t, k=8), torch.zeros(7), atol=1e-6))
        self.assertTrue(bool((fc.bucket_forward_kl(s, t, k=8) >= -1e-6).all()))

    def test_bucket_jsd(self):
        import torch
        torch.manual_seed(1)
        t, s = torch.randn(5, 40), torch.randn(5, 40, requires_grad=True)
        p, q = t.softmax(-1), s.softmax(-1)
        m = 0.5 * p + 0.5 * q
        full = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
        got = fc.bucket_jsd(s, t, k=39, beta=0.5)
        self.assertTrue(torch.allclose(got, full, atol=1e-5))
        self.assertTrue(bool((got <= math.log(2) + 1e-6).all()))
        got.sum().backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.allclose(fc.bucket_jsd(t, t, k=10), torch.zeros(5), atol=1e-6))


def build_tiny(root: Path, seed: int) -> Path:
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    from transformers import Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                            num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                            tie_word_embeddings=True, max_position_embeddings=4096, eos_token_id=248044)
    fc.force_torch_kernels()
    model = mod.Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():                       # sharper logits so greedy paths are not argmax ties
        model.lm_head.weight.mul_(8.0)
    model.save_pretrained(str(root))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if (TOKENIZER_DIR / name).exists():
            shutil.copy(TOKENIZER_DIR / name, root / name)
    return root


@unittest.skipUnless(HAVE_TORCH and (TOKENIZER_DIR / "tokenizer.json").exists(), "needs torch + the Qwen3.5 tokenizer")
class TinyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="foldtest-"))
        cls.parent = build_tiny(cls.tmp / "parent", 1)
        cls.child = build_tiny(cls.tmp / "child", 2)
        cls.tok, cls.eos = fc.load_tokenizer(cls.parent)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_thinking_off_template(self):
        ids = fc.chat_ids(self.tok, [{"role": "user", "content": "hi"}])
        self.assertTrue(self.tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_generate_padding_and_decode_path(self):
        import torch
        model, _ = fc.load_model(self.parent, device="cpu")
        prompts = [fc.chat_ids(self.tok, [{"role": "user", "content": t}]) for t in
                   ("hi", "Write a long poem about rivers and mountains please", "2+2?")]
        pad = self.tok.pad_token_id
        single = [fc.generate(model, [p], max_new=12, eos_ids=self.eos, pad_id=pad, batch_size=1)[0] for p in prompts]
        batched = fc.generate(model, prompts, max_new=12, eos_ids=self.eos, pad_id=pad, batch_size=3)
        self.assertEqual([s["generated_ids"] for s in single], [b["generated_ids"] for b in batched])
        ids = torch.tensor([prompts[1]])                 # manual no-cache argmax loop == generate
        manual = []
        with torch.no_grad():
            for _ in range(8):
                nxt = int(model(input_ids=ids).logits[0, -1].argmax())
                manual.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        self.assertEqual(manual, single[1]["generated_ids"][:8])
        pos = torch.tensor([3, 5, 9])
        with torch.no_grad():
            full = model(input_ids=torch.tensor([prompts[1]])).logits[0]
            kept = model(input_ids=torch.tensor([prompts[1]]), logits_to_keep=pos).logits[0]
        self.assertTrue(torch.allclose(full[pos], kept, atol=1e-5))

    def test_overlay_roundtrip(self):
        import torch
        child, _ = fc.load_model(self.child, device="cpu")
        path = self.tmp / "ov.safetensors"
        fc.save_overlay(child, path)
        parent, rep = fc.load_model(self.parent, device="cpu", overlay=path)
        for (n, a), (_, b) in zip(sorted(fc.mlp_parameters(child).items()), sorted(fc.mlp_parameters(parent).items())):
            self.assertTrue(torch.equal(a, b), n)
        self.assertEqual(rep["overlay"]["tensors"], 6)
        ident = fc.checkpoint_identity(self.parent, path)
        self.assertNotEqual(ident["checkpoint_sha256"], ident["dir_sha256"])

    def _pack(self, root: Path) -> Path:
        retention = []
        for i, t in enumerate(("Explain photosynthesis briefly to a child please.", "List three prime numbers and say why.")):
            msgs = [{"role": "user", "content": t}]
            retention.append({"id": f"ret:{i}", "source_group_id": f"carve-eval:{i}", "split": "validation", "route": "text",
                              "messages": msgs, "prompt_ids": fc.chat_ids(self.tok, msgs)})
        semantic = [
            {"id": "ar", "route": "text_arithmetic", "group_id": "g", "split": "test",
             "input": {"messages": [{"role": "user", "content": "Compute 2+2. Final line numeric."}], "tools": [], "media": []},
             "oracle": {"kind": "arithmetic", "expected": "4", "absolute_tolerance": "0"}},
            {"id": "js", "route": "text_instruction", "group_id": "g", "split": "test",
             "input": {"messages": [{"role": "user", "content": "Return {\"a\":1} only."}], "tools": [], "media": []},
             "oracle": {"kind": "instruction_json", "expected": {"a": 1}, "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}}}]
        pack = {"schema": fold_eval.PACK_SCHEMA, "tokenizer_sha256": fc.file_sha256(self.parent / "tokenizer.json"),
                "eos_ids": self.eos, "max_new": {"retention": 32, "semantic": 16, "bank": 16},
                "semantic": semantic, "retention": retention,
                "bank_v1": [{"id": "b0", "category": "c", "messages": [{"role": "user", "content": "Hello there"}], "context": "x"}],
                "retention_prompts_sha256": fc.digest([r["prompt_ids"] for r in retention])}
        fc.write_json(root / "pack.json", pack)
        return root / "pack.json"

    def test_eval_run_grade_pair(self):
        root = self.tmp / "eval"
        pack = self._pack(root)
        for label, model in (("parent", self.parent), ("child", self.child)):
            fold_eval.main(["run", "--pack", str(pack), "--model-dir", str(model), "--label", label, "--device", "cpu",
                            "--batch", "2", "--out", str(root / label)])
        fc.write_json(root / "groups.json", ["carve-eval:train-only"])
        for label in ("parent", "child"):
            fold_eval.main(["grade", "--pack", str(pack), "--run", str(root / label), "--parent-run", str(root / "parent"),
                            "--training-groups", str(root / "groups.json"), "--out", str(root / label / "grade.json")])
        child = json.loads((root / "child" / "grade.json").read_text())
        self.assertEqual(child["semantic"]["n"], 2)
        self.assertEqual(child["retention"]["summary"]["pairs"], 2)
        self.assertTrue((root / "child" / "bank_v1_answers.md").exists())
        self.assertTrue(json.loads((root / "parent" / "grade.json").read_text())["retention"]["self"])
        fold_eval.main(["pair", "--base", str(root / "parent" / "grade.json"), "--cand", str(root / "child" / "grade.json"),
                        "--out", str(root / "pair.json")])
        pair = json.loads((root / "pair.json").read_text())
        self.assertEqual(pair["content_pooled"]["n"], 2)
        fc.write_json(root / "groups_bad.json", ["carve-eval:0"])       # retention group listed as training -> refused
        with self.assertRaises(ValueError):
            fold_eval.main(["grade", "--pack", str(pack), "--run", str(root / "child"), "--parent-run", str(root / "parent"),
                            "--training-groups", str(root / "groups_bad.json"), "--out", str(root / "child" / "grade2.json")])

    def test_summary_layout(self):
        import x3_summary
        root = self.tmp / "sumroot"
        pack = self._pack(root / "evalpack")
        fc.write_json(root / "groups.json", [])
        for label, model in (("parent", self.parent), ("c2k896", self.child), ("kd-250000", self.child)):
            d = root / "eval" / label
            fold_eval.main(["run", "--pack", str(pack), "--model-dir", str(model), "--label", label, "--device", "cpu",
                            "--batch", "2", "--out", str(d)])
            fold_eval.main(["grade", "--pack", str(pack), "--run", str(d), "--parent-run", str(root / "eval" / "parent"),
                            "--training-groups", str(root / "groups.json"), "--out", str(d / "grade.json")])
        fold_eval.main(["pair", "--base", str(root / "eval" / "c2k896" / "grade.json"),
                        "--cand", str(root / "eval" / "kd-250000" / "grade.json"),
                        "--out", str(root / "eval" / "kd-250000" / "pair_vs_c2k896.json")])
        x3_summary.main(["--root", str(root)])
        summary = json.loads((root / "SUMMARY.json").read_text())
        self.assertEqual([r["label"] for r in summary["rows"]], ["parent", "c2k896", "kd-250000"])
        self.assertEqual(summary["arms"]["kd"]["rungs_graded"], [250000])
        self.assertTrue((root / "SUMMARY.md").read_text().startswith("# X3 0.8B ladder"))

    def test_losses_descend_along_their_gradient(self):
        """A small normalised step against the gradient of each training loss must lower that loss on the same
        sequence: the loss, its masking and the MLP-only parameter set are wired the same way the trainer uses them."""
        import torch
        import x3_ladder
        teacher, _ = fc.load_model(self.parent, device="cpu")
        student, _ = fc.load_model(self.child, device="cpu")
        for p in student.parameters():
            p.requires_grad_(False)
        params = fc.mlp_parameters(student)
        for p in params.values():
            p.requires_grad_(True)
        prompt = fc.chat_ids(self.tok, [{"role": "user", "content": "What is water?"}])
        answer = self.tok.encode("Water is H2O, a liquid.", add_special_tokens=False) + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        row = {"ids": prompt + answer, "loss_start": len(prompt) - 1}
        fns = {"kd": lambda: x3_ladder.kd_doc_loss(teacher, student, row, "cpu", 64, 0.1)[0],
               "opd": lambda: x3_ladder.opd_seq_loss(teacher, student, prompt, answer, "cpu", 100, 0.5, 64, 0.3)[0]}
        for name, fn in fns.items():
            for p in params.values():
                p.grad = None
            before = fn()
            before.backward()
            norm = torch.sqrt(sum((p.grad ** 2).sum() for p in params.values()))
            self.assertGreater(float(norm), 0.0, name)
            with torch.no_grad():
                saved = {n: p.clone() for n, p in params.items()}
                for p in params.values():
                    p -= 1e-2 * p.grad / norm
                after = fn()
                for n, p in params.items():
                    p.copy_(saved[n])
            self.assertLess(float(after), float(before), name)

    def test_x3_train_kd_and_opd_with_preemption(self):
        import x3_ladder
        data = self.tmp / "data"
        kd, opd = [], []
        for i, (q, a) in enumerate([("What is water?", "Water is H2O, a liquid at room temperature."),
                                    ("Name a colour.", "Blue is a colour often seen in the sky.")] * 4):
            prompt = fc.chat_ids(self.tok, [{"role": "user", "content": q}])
            content = self.tok.encode(a, add_special_tokens=False)
            ids = prompt + content + [self.tok.convert_tokens_to_ids("<|im_end|>")]
            kd.append({"uid": str(i), "record_id": f"r{i}", "ids": ids, "loss_start": len(prompt) - 1, "loss_tokens": len(ids) - len(prompt)})
            opd.append({"uid": str(i), "record_id": f"r{i}", "prompt_ids": prompt})
        fc.write_jsonl(data / "kd.jsonl", kd)
        fc.write_jsonl(data / "opd.jsonl", opd)
        fc.write_json(data / "data_receipt.json", {"kd": {"docs": len(kd)}, "opd": {"prompts": len(opd)}})
        common = ["--parent", str(self.parent), "--child", str(self.child), "--data", str(data), "--device", "cpu",
                  "--lr", "1e-3", "--warmup", "1"]
        fc.write_jsonl(data / "kd.jsonl", [kd[0]] * 8)        # one document: the loss on it must fall
        x3_ladder.main(["train", "--arm", "kd", *common, "--rungs", "40,120", "--kd-step-tokens", "20", "--out", str(self.tmp / "kd")])
        done = json.loads((self.tmp / "kd" / "ARM_DONE.json").read_text())
        self.assertEqual([r["rung"] for r in done["rungs"]], [40, 120])
        log = [json.loads(l) for l in (self.tmp / "kd" / "train_log.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), done["steps"])
        self.assertTrue(all(math.isfinite(r["fkl"]) and r["fkl"] >= 0 for r in log))
        out = self.tmp / "opd"
        out.mkdir()
        (out / "STOP").write_text("x")
        with self.assertRaises(SystemExit) as stop:
            x3_ladder.main(["train", "--arm", "opd", *common, "--init-overlay", str(self.tmp / "kd" / "rung-40.safetensors"),
                            "--rungs", "30,60", "--opd-batch", "4", "--budget-start", "16", "--budget-low", "16",
                            "--budget-high", "32", "--out", str(out)])
        self.assertEqual(stop.exception.code, x3_ladder.EXIT_PREEMPTED)
        self.assertTrue((out / "resume" / "state.json").exists())
        (out / "STOP").unlink()
        x3_ladder.Stop.requested = False
        x3_ladder.main(["train", "--arm", "opd", *common, "--init-overlay", str(self.tmp / "kd" / "rung-40.safetensors"),
                        "--rungs", "30,60", "--opd-batch", "4", "--budget-start", "16", "--budget-low", "16",
                        "--budget-high", "32", "--out", str(out)])
        done = json.loads((out / "ARM_DONE.json").read_text())
        self.assertGreaterEqual(done["tokens"], 60)
        self.assertTrue(json.loads((out / "arm_config.json").read_text())["resumed"])


if __name__ == "__main__":
    unittest.main()
