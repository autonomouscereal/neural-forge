from __future__ import annotations

import pytest

from neural_forge.cli import build_parser
from neural_forge.cli import _parse_layer_groups
from neural_forge.hf_recovery import RECOVERY_TEXTS
from neural_forge.hf_recovery import _assert_same_vocab
from neural_forge.hf_recovery import _set_train_scope
from neural_forge.hf_tools import TEXT_EVALS
from neural_forge.hf_tools import _layer_groups_for_report
from neural_forge.hf_tools import _layer_groups_tag
from neural_forge.hf_tools import _normalize_layer_groups
from neural_forge.hf_xfamily import _canonical_token
from neural_forge.hf_xfamily import _canonical_token_map
from neural_forge.hf_xfamily import _mlp_spec
from neural_forge.hf_xfamily import _proportional_index
from neural_forge.hf_xfamily import sweep_cross_family_vocab_merge
from neural_forge.hf_xfamily import sweep_cross_family_mlp_merge


class _Tokenizer:
    all_special_ids = [0]

    def __init__(self, decoded: list[str]) -> None:
        self._decoded = decoded

    def __len__(self) -> int:
        return len(self._decoded)

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        assert clean_up_tokenization_spaces is False
        return self._decoded[ids[0]]

    def get_vocab(self) -> dict[str, int]:
        return {text: idx for idx, text in enumerate(self._decoded)}


class _Obj:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def test_canonical_token_normalizes_common_tokenizer_markers() -> None:
    assert _canonical_token("Ġhello") == "hello"
    assert _canonical_token("\u2581world") == "world"
    assert _canonical_token("  text ") == "text"
    assert _canonical_token("") == ""
    assert _canonical_token("\n") == ""


def test_canonical_token_map_skips_special_and_empty_tokens() -> None:
    tokenizer = _Tokenizer(["<eos>", "Ġhello", "", "\u2581world", "hello"])
    mapping = _canonical_token_map(tokenizer)
    assert mapping == {"hello": 1, "world": 3}


def test_recovery_same_vocab_guard_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="matching tokenizers"):
        _assert_same_vocab(_Tokenizer(["a", "b"]), _Tokenizer(["a", "c"]))


def test_recovery_defaults_do_not_reuse_evaluation_prompts() -> None:
    assert set(RECOVERY_TEXTS).isdisjoint(TEXT_EVALS)


def test_cross_family_sweep_rejects_duplicate_alpha_tags(tmp_path) -> None:
    with pytest.raises(ValueError, match="Duplicate alpha output tag"):
        sweep_cross_family_vocab_merge(
            host_model="host",
            donor_model="donor",
            output_dir=tmp_path,
            alphas=[0.1, 0.10001],
        )


def test_cross_family_mlp_sweep_rejects_duplicate_alpha_tags(tmp_path) -> None:
    with pytest.raises(ValueError, match="Duplicate alpha output tag"):
        sweep_cross_family_mlp_merge(
            host_model="host",
            donor_model="donor",
            output_dir=tmp_path,
            alphas=[0.1, 0.10001],
        )


def test_cli_accepts_cross_family_sweep_and_topk_resize() -> None:
    parser = build_parser()
    sweep_args = parser.parse_args(
        [
            "hf-sweep-xfamily",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.005,0.01",
        ]
    )
    resize_args = parser.parse_args(["hf-resize-ffn", "in", "out", "--ratio", "0.95", "--assignment", "topk"])
    merge_layer_args = parser.parse_args(["hf-merge-layers", "in", "out", "--groups", "3+4,7+8"])
    drop_args = parser.parse_args(["hf-drop-layers", "in", "out", "--layers", "3,7"])
    search_args = parser.parse_args(
        [
            "hf-search-compress",
            "model",
            "out",
            "--width-ratios",
            "0.9975,0.995",
            "--layer-groups",
            "13+14",
            "--target-shrink",
            "0.02",
        ]
    )
    xfamily_mlp_args = parser.parse_args(
        [
            "hf-sweep-xfamily-mlp",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.001,0.005",
            "--max-host-channels",
            "128",
            "--max-donor-channels",
            "256",
            "--recover",
            "--recover-steps",
            "3",
            "--recover-train-scope",
            "ffn",
        ]
    )
    assert sweep_args.output_dir == "out"
    assert resize_args.assignment == "topk"
    assert merge_layer_args.groups == "3+4,7+8"
    assert drop_args.layers == "3,7"
    assert search_args.width_ratios == "0.9975,0.995"
    assert search_args.layer_groups == ["13+14"]
    assert search_args.target_shrink == 0.02
    assert xfamily_mlp_args.max_host_channels == 128
    assert xfamily_mlp_args.max_donor_channels == 256
    assert xfamily_mlp_args.recover is True
    assert xfamily_mlp_args.recover_steps == 3
    assert xfamily_mlp_args.recover_train_scope == "ffn"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "hf-sweep-task",
                "out",
                "--base-model",
                "base",
                "--tuned-model",
                "tuned",
                "--alphas",
                "0.1",
                "--recover",
            ]
        )


def test_parse_layer_groups() -> None:
    assert _parse_layer_groups("3+4, 7+8+9") == [[3, 4], [7, 8, 9]]


def test_layer_group_normalization_rejects_pruning_groups() -> None:
    with pytest.raises(ValueError, match="at least two layers"):
        _normalize_layer_groups([[3]])
    with pytest.raises(ValueError, match="overlap"):
        _normalize_layer_groups([[3, 4], [4, 5]])


def test_layer_group_tags_are_stable() -> None:
    groups = _normalize_layer_groups([[14, 13], [20, 19]])
    assert _layer_groups_tag(groups) == "13-14_19-20"
    assert _layer_groups_for_report(groups) == [[13, 14], [19, 20]]


def test_proportional_layer_index_maps_depths() -> None:
    assert [_proportional_index(idx, 4, 2) for idx in range(4)] == [0, 0, 1, 1]
    assert [_proportional_index(idx, 3, 5) for idx in range(3)] == [0, 2, 4]


def test_mlp_spec_detects_lfm_feed_forward_names() -> None:
    layer = _Obj(feed_forward=_Obj(w1=object(), w3=object(), w2=object()))
    spec = _mlp_spec(layer)
    assert spec is not None
    assert spec["kind"] == "gated"
    assert spec["input_names"] == ["w1", "w3"]
    assert spec["output_name"] == "w2"


def test_recovery_train_scope_supports_feed_forward_layers() -> None:
    class _Param:
        def __init__(self) -> None:
            self.requires_grad = False

        def requires_grad_(self, value: bool):
            self.requires_grad = value
            return self

        def numel(self) -> int:
            return 1

    class _Module:
        def __init__(self) -> None:
            self.param = _Param()

        def parameters(self):
            return [self.param]

    class _Layer:
        def __init__(self) -> None:
            self.feed_forward = _Module()

        def parameters(self):
            return self.feed_forward.parameters()

    class _Model:
        def __init__(self) -> None:
            self.model = _Obj(layers=[_Layer(), _Layer()])
            self.output = _Module()

        def parameters(self):
            params = []
            for layer in self.model.layers:
                params.extend(layer.parameters())
            params.extend(self.output.parameters())
            return params

        def get_output_embeddings(self):
            return self.output

    model = _Model()
    trainable = _set_train_scope(model, "ffn")
    assert trainable == 2
    assert all(layer.feed_forward.param.requires_grad for layer in model.model.layers)


def test_xfamily_mlp_handler_passes_recovery_flags(monkeypatch) -> None:
    import neural_forge.hf_xfamily as hf_xfamily

    captured = {}

    def fake_sweep(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(hf_xfamily, "sweep_cross_family_mlp_merge", fake_sweep)
    args = build_parser().parse_args(
        [
            "hf-sweep-xfamily-mlp",
            "out",
            "--host-model",
            "host",
            "--donor-model",
            "donor",
            "--alphas",
            "0.005",
            "--recover",
            "--recover-only-failed",
            "--recover-steps",
            "7",
            "--recover-train-scope",
            "last2",
        ]
    )
    result = args.func(args)
    assert result == {"ok": True}
    assert captured["recover_candidates"] is True
    assert captured["recover_only_failed"] is True
    assert captured["recovery_steps"] == 7
    assert captured["recovery_train_scope"] == "last2"
