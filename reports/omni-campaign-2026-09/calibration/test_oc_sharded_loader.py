"""CPU tests for explicit stock-HF placement; no model downloads or GPU work."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oc_run_frozen_bank as runner


def fake_torch(monkeypatch, free=(24 << 30, 24 << 30)):
    cuda = SimpleNamespace(device_count=lambda: len(free),
                           mem_get_info=lambda i: (free[i], free[i]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runner, "available_host_memory_bytes", lambda: 16 << 30,
                        raising=False)


def test_default_plan_needs_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert runner.make_load_plan(None, None, device="cuda:0") is None


def test_explicit_all_visible_budgets_and_zero_cpu(monkeypatch):
    fake_torch(monkeypatch)
    plan = runner.make_load_plan("balanced", ["0=20GiB", "1=20GiB", "cpu=0GiB"],
                                 device="cuda:0")
    assert plan["max_memory"] == {0: 20 << 30, 1: 20 << 30, "cpu": 0}
    assert plan["device_map"] == "balanced"


@pytest.mark.parametrize("strategy,limits,device", [
    (None, ["cpu=1GiB"], "cpu"),
    ("/tmp/map.json", ["cpu=1GiB"], "cpu"),
    ("auto", None, "cuda:0"),
    ("auto", ["0=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB", "cpu=0GiB", "cpu=1GiB"], "cuda:0"),
    ("auto", ["00=20GiB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20.5GiB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=-1GiB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB", "disk=1GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB", "2=1GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=25GiB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB", "cpu=17GiB"], "cuda:0"),
    ("auto", ["0=0GiB", "1=20GiB", "cpu=0GiB"], "cuda:0"),
    ("auto", ["0=20GiB", "1=20GiB", "cpu=0GiB"], "cpu"),
    ("auto", ["0=20GiB", "1=20GiB", "cpu=0GiB"], "cuda"),
    ("auto", ["0=20GiB", "1=20GiB", "cpu=0GiB"], "mps"),
])
def test_invalid_limits_are_refused(monkeypatch, strategy, limits, device):
    fake_torch(monkeypatch)
    with pytest.raises(runner.CalibrationError):
        runner.make_load_plan(strategy, limits, device=device)


def test_zero_gpu_budgets_do_not_probe_busy_devices(monkeypatch):
    fake_torch(monkeypatch)
    sys.modules["torch"].cuda.mem_get_info = lambda i: (_ for _ in ()).throw(AssertionError())
    plan = runner.make_load_plan("sequential", ["0=0GiB", "1=0GiB", "cpu=4MiB"],
                                 device="cpu")
    assert plan["max_memory"]["cpu"] == 4 << 20


def test_tiny_random_refuses_placement(monkeypatch):
    fake_torch(monkeypatch)
    with pytest.raises(runner.CalibrationError):
        runner.make_load_plan("auto", ["0=20GiB", "1=20GiB", "cpu=0GiB"],
                              device="cuda:0", tiny_random=True)


class DummyModel:
    def __init__(self, placement=None, embedding="cuda:1"):
        if placement is not None:
            self.hf_device_map = placement
        self.embedding = SimpleNamespace(weight=SimpleNamespace(device=embedding))
        self.moves = []

    def get_input_embeddings(self):
        return self.embedding

    def to(self, device):
        self.moves.append(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return []


def patch_loader(monkeypatch, model, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DummyModel"]}))
    calls = []
    class Loader:
        __name__ = "DummyModel"
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(kwargs)
            return model
    monkeypatch.setattr(runner, "resolve_loader_class", lambda *_: Loader)
    monkeypatch.setattr(runner, "kernel_path_report", lambda *_: {})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test"))
    return calls


def test_default_loader_keeps_old_call_and_move(monkeypatch, tmp_path):
    model = DummyModel()
    calls = patch_loader(monkeypatch, model, tmp_path)
    _, export = runner.load_model(tmp_path, dtype="bf16", device="cuda:0", requested_class=None)
    assert calls == [{"dtype": "bf16", "trust_remote_code": False}]
    assert model.moves == ["cuda:0"]
    assert "device_map" not in export


def test_mapped_loader_streams_and_uses_embedding_device(monkeypatch, tmp_path):
    model = DummyModel({"embed": 1, "layers": 0})
    calls = patch_loader(monkeypatch, model, tmp_path)
    budget = {0: 20 << 30, 1: 20 << 30, "cpu": 0}
    _, export = runner.load_model(tmp_path, dtype="bf16", device="cuda:0", requested_class=None,
                                  load_plan={"device_map": "balanced", "max_memory": budget})
    assert calls == [{"dtype": "bf16", "trust_remote_code": False,
                     "device_map": "balanced", "max_memory": budget}]
    assert model.moves == []
    assert export["input_device"] == "cuda:1"
    assert export["device_map"] == {"embed": "cuda:1", "layers": "cuda:0"}


def test_stock_cpu_map_absent_is_measured_from_resident_parameters(monkeypatch, tmp_path):
    model = DummyModel(embedding="cpu")
    parameter = SimpleNamespace(device="cpu", dtype="float32", numel=lambda: 10,
                                is_floating_point=lambda: True)
    model.named_parameters = lambda: [("embed.weight", parameter)]
    model.parameters = lambda: [parameter]
    patch_loader(monkeypatch, model, tmp_path)
    _, export = runner.load_model(tmp_path, dtype="float32", device="cpu", requested_class=None,
                                  load_plan={"device_map": "sequential", "max_memory": {"cpu": 4 << 20}})
    assert export["device_map"] == {"embed.weight": "cpu"}
    assert export["device_map_source"] == "resident parameter devices"
    assert export["floating_parameter_dtypes"] == {"float32": 10}


@pytest.mark.parametrize("actual_device", ["meta", "cuda:2", "cpu"])
def test_resident_parameter_outside_budget_is_refused(monkeypatch, tmp_path, actual_device):
    model = DummyModel({"": 0}, "cuda:0")
    model.parameters = lambda: [SimpleNamespace(device=actual_device)]
    patch_loader(monkeypatch, model, tmp_path)
    with pytest.raises(runner.CalibrationError):
        runner.load_model(tmp_path, dtype="bf16", device="cuda:0", requested_class=None,
                          load_plan={"device_map": "auto", "max_memory": {0: 20 << 30, "cpu": 0}})


def test_fp32_norm_parameters_are_recorded_without_casting(monkeypatch, tmp_path):
    model = DummyModel({"": 0}, "cuda:0")
    parameters = [SimpleNamespace(device="cuda:0", dtype=dtype, numel=lambda: 10,
                                  is_floating_point=lambda: True)
                  for dtype in ["bfloat16", "float32"]]
    model.parameters = lambda: parameters
    calls = patch_loader(monkeypatch, model, tmp_path)
    _, export = runner.load_model(tmp_path, dtype="bfloat16", device="cuda:0", requested_class=None,
                                  load_plan={"device_map": "auto", "max_memory": {0: 20 << 30, "cpu": 0}})
    assert export["floating_parameter_dtypes"] == {"bfloat16": 10, "float32": 10}
    assert calls[0]["dtype"] == "bfloat16"
    assert model.moves == []


@pytest.mark.parametrize("placement,embedding", [
    ({"": "disk"}, "cuda:0"), ({"": "meta"}, "cuda:0"),
    ({"": 2}, "cuda:0"), ({"": "cpu"}, "cpu"),
    ({"": True}, "cuda:0"), ({"": 0}, "meta"),
    ({"": 0}, "cuda:2"), (None, "cuda:0"),
])
def test_bad_actual_placement_refused(monkeypatch, tmp_path, placement, embedding):
    model = DummyModel(placement, embedding)
    patch_loader(monkeypatch, model, tmp_path)
    with pytest.raises(runner.CalibrationError):
        runner.load_model(tmp_path, dtype="bf16", device="cuda:0", requested_class=None,
                          load_plan={"device_map": "auto", "max_memory": {0: 20 << 30, "cpu": 0}})


def test_cli_defaults_unchanged_and_choices_are_bounded():
    args = runner.parser().parse_args(["--model-dir", "unused", "--label", "test", "--out", "unused.json"])
    assert args.device_map is None and args.max_memory is None
    assert args.maximum_new_tokens == 1024 and args.smoke_maximum_new_tokens == 128
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["--model-dir", "unused", "--label", "test", "--out", "unused.json",
                                    "--device-map", "map.json"])


@pytest.mark.parametrize("mapped", [False, True])
def test_real_tiny_sharded_hf_roundtrip(tmp_path, mapped):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if mapped:
        pytest.importorskip("accelerate")
    torch.set_num_threads(1)
    torch.manual_seed(20260901)
    config = transformers.GPT2Config(vocab_size=16, n_positions=16, n_embd=8,
                                     n_layer=2, n_head=2, resid_pdrop=0,
                                     embd_pdrop=0, attn_pdrop=0)
    original = transformers.GPT2LMHeadModel(config).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = original(ids).logits
    original.save_pretrained(tmp_path, max_shard_size="1KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()
    limits = [f"{i}=0MiB" for i in range(torch.cuda.device_count())] + ["cpu=4MiB"]
    plan = runner.make_load_plan("sequential", limits, device="cpu") if mapped else None
    loaded, export = runner.load_model(tmp_path, dtype=torch.float32, device="cpu",
                                       requested_class=None, load_plan=plan)
    with torch.no_grad():
        actual = loaded(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    assert export["parameters_counted_live"] == sum(p.numel() for p in original.parameters())
    if mapped:
        assert export["input_device"] == "cpu"
        assert set(export["device_map"].values()) == {"cpu"}
