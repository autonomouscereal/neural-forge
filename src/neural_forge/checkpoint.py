"""Dense checkpoint format used by Neural Forge.

The first implementation targets feed-forward dense networks because their
neuron axes are explicit: a hidden unit is a row in one layer and a column in
the next layer. Transformer MLP blocks and attention heads can be mapped onto
the same primitives by adapters later.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_ACTIVATIONS = {"relu", "linear", "tanh"}


@dataclass
class DenseModel:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activation: str = "relu"
    name: str = ""

    def __post_init__(self) -> None:
        self.weights = [np.asarray(w, dtype=np.float64).copy() for w in self.weights]
        self.biases = [np.asarray(b, dtype=np.float64).copy() for b in self.biases]
        self.activation = self.activation.lower()
        self.validate()

    def validate(self) -> None:
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if not self.weights:
            raise ValueError("A model must contain at least one layer.")
        if len(self.weights) != len(self.biases):
            raise ValueError("weights and biases must have the same length.")
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if weight.ndim != 2:
                raise ValueError(f"W{idx} must be 2-D, got {weight.ndim}-D.")
            if bias.ndim != 1:
                raise ValueError(f"b{idx} must be 1-D, got {bias.ndim}-D.")
            if weight.shape[0] != bias.shape[0]:
                raise ValueError(
                    f"W{idx}.shape[0] ({weight.shape[0]}) must match b{idx}.shape[0] ({bias.shape[0]})."
                )
            if idx > 0 and weight.shape[1] != self.weights[idx - 1].shape[0]:
                raise ValueError(
                    f"W{idx}.shape[1] ({weight.shape[1]}) must match W{idx - 1}.shape[0] "
                    f"({self.weights[idx - 1].shape[0]})."
                )

    @property
    def depth(self) -> int:
        return len(self.weights)

    @property
    def hidden_depth(self) -> int:
        return max(0, len(self.weights) - 1)

    @property
    def widths(self) -> list[int]:
        return [int(self.weights[0].shape[1])] + [int(weight.shape[0]) for weight in self.weights]

    @property
    def hidden_widths(self) -> list[int]:
        return self.widths[1:-1]

    @property
    def input_dim(self) -> int:
        return self.widths[0]

    @property
    def output_dim(self) -> int:
        return self.widths[-1]

    @property
    def param_count(self) -> int:
        return int(sum(weight.size + bias.size for weight, bias in zip(self.weights, self.biases)))

    def copy(self, name: str | None = None) -> "DenseModel":
        return DenseModel(
            weights=[weight.copy() for weight in self.weights],
            biases=[bias.copy() for bias in self.biases],
            activation=self.activation,
            name=self.name if name is None else name,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if idx < len(self.weights) - 1:
                values = apply_activation(values, self.activation)
        return values

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "activation": self.activation,
            "depth": self.depth,
            "widths": self.widths,
            "hidden_widths": self.hidden_widths,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "param_count": self.param_count,
        }


def apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "linear":
        return values
    if activation == "tanh":
        return np.tanh(values)
    raise ValueError(f"Unsupported activation: {activation}")


def param_count_for_widths(widths: Iterable[int]) -> int:
    width_list = [int(width) for width in widths]
    if len(width_list) < 2:
        raise ValueError("Need at least input and output widths.")
    return int(sum((out_width * in_width) + out_width for in_width, out_width in zip(width_list, width_list[1:])))


def random_toy_model(widths: Iterable[int], seed: int = 0, activation: str = "relu", name: str = "toy") -> DenseModel:
    width_list = [int(width) for width in widths]
    if len(width_list) < 2:
        raise ValueError("A toy model needs at least input and output widths.")
    if any(width <= 0 for width in width_list):
        raise ValueError("All widths must be positive.")
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for in_width, out_width in zip(width_list, width_list[1:]):
        scale = np.sqrt(2.0 / max(1, in_width))
        weights.append(rng.normal(0.0, scale, size=(out_width, in_width)))
        biases.append(rng.normal(0.0, scale * 0.05, size=(out_width,)))
    return DenseModel(weights=weights, biases=biases, activation=activation, name=name)


def save_model(model: DenseModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "neural-forge-dense-v1",
        "activation": model.activation,
        "name": model.name,
        "depth": model.depth,
    }
    arrays: dict[str, np.ndarray] = {"__meta__": np.array(json.dumps(meta), dtype=np.str_)}
    for idx, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        arrays[f"W{idx}"] = weight
        arrays[f"b{idx}"] = bias
    np.savez_compressed(target, **arrays)


def load_model(path: str | Path) -> DenseModel:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if "__meta__" not in data.files:
            raise ValueError(f"{source} is missing __meta__.")
        meta = json.loads(str(data["__meta__"].item()))
        if meta.get("format") != "neural-forge-dense-v1":
            raise ValueError(f"Unsupported checkpoint format: {meta.get('format')}")
        depth = int(meta["depth"])
        weights = [data[f"W{idx}"] for idx in range(depth)]
        biases = [data[f"b{idx}"] for idx in range(depth)]
        return DenseModel(
            weights=weights,
            biases=biases,
            activation=str(meta.get("activation", "relu")),
            name=str(meta.get("name", source.stem)),
        )
