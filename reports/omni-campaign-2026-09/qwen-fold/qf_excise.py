"""C3 full-attention layer excision and C4 GDN layer excision.

One transaction, no solve.  Re-solving after an excision was a MEASURED no-op on
LFM2 across five waves, so the default is the honest one -- delete the layer,
re-derive the config, and let the review say what it cost.

The whole risk of this class on qwen3_5 is bookkeeping, and it is a real risk.
``Qwen3_5TextModel.forward`` chooses each layer's mask by
``self.config.layer_types[i]`` (modeling_qwen3_5.py:1201) and iterates
``self.layers[: self.config.num_hidden_layers]`` (1200).  Delete a layer and
leave either of those alone and the model still runs:

* an attention module that lands in a slot the config calls linear gets
  ``linear_attn_mask``, which is ``None`` for an unpadded batch -- **so it
  attends to the future** and every number measured after that is measured on a
  model that cheats;
* a GDN module that lands in a full slot gets a 4D causal mask, which
  ``apply_mask_to_padding_states`` (210) throws away at batch 1 -- so that
  direction is invisible.

The trap is therefore ASYMMETRIC, and only the attention direction is
observable.  :func:`assert_causal_masks` observes it, on every excising child,
before any capture.
"""

from __future__ import annotations

from typing import Any

import torch

import qf_arch
from qf_arch import FULL, LINEAR


def excise_live(model: torch.nn.Module, excised: list[int]) -> dict[str, Any]:
    """Delete layers from a live model and re-derive everything that indexes them.

    Descending order, because deleting position 3 renames position 7.
    """

    layers = qf_arch.layers(model)
    tc = qf_arch.text_config(model.config)
    parent_types = list(tc.layer_types)
    parent_count = len(layers)
    excised = sorted({int(i) for i in excised}, reverse=True)
    for index in excised:
        if index < 0 or index >= parent_count:
            raise SystemExit(f"cannot excise layer {index} of {parent_count}")
        del layers[index]

    surviving_positions = [i for i in range(parent_count) if i not in set(excised)]
    surviving_types = [parent_types[i] for i in surviving_positions]
    qf_arch.derive_layer_types(tc, surviving_types)
    if tc is not model.config:
        # `export` writes the TOP-level config, and a top-level COPY of either
        # field that disagrees with the text sub-config produces a child that
        # loads with the parent's depth.  The previous form of this repair read
        # `text_config(model.config)`, which IS `tc` -- it re-wrote the value
        # that had just been written and never touched the top level.  Only
        # fields the top-level config already carries in its own __dict__ are
        # repaired; inventing one on a composite config is a different bug.
        for field, value in (
            ("layer_types", list(surviving_types)),
            ("num_hidden_layers", len(surviving_types)),
        ):
            if field in model.config.__dict__:
                setattr(model.config, field, value)
    qf_arch.reindex_layers(model)
    qf_arch.assert_layer_types_agree(model)

    return {
        "axis": "layer_excision",
        "excised": sorted(int(i) for i in excised),
        "excised_kinds": [parent_types[i] for i in sorted(excised)],
        "parent_layers": parent_count,
        "child_layers": len(surviving_positions),
        "surviving_parent_positions": surviving_positions,
        "child_layer_types": surviving_types,
        "full_layers_removed": sum(1 for i in excised if parent_types[i] == FULL),
        "linear_layers_removed": sum(1 for i in excised if parent_types[i] == LINEAR),
    }


def parameters_removed(tc_parent: Any, excised_kinds: list[str]) -> int:
    """Closed form, from the parent's own geometry -- never a measured difference."""

    counts = qf_arch.text_parameter_count(tc_parent)
    return sum(
        counts["per_full_layer"] if kind == FULL else counts["per_linear_layer"]
        for kind in excised_kinds
    )


@torch.no_grad()
def assert_causal_masks(
    harness, *, tolerance: float = 0.0, batches: int = 1, strict_linear: bool = True
) -> dict[str, Any]:
    """Refuse a model whose attention can see the future, or whose GDN gets a mask.

    Eager attention returns ``attn_weights`` as the second element of
    ``Qwen3_5Attention.forward``'s output (modeling 717); a forward hook on the
    module gets the whole tuple.  Above-diagonal mass is
    ``triu(A, 1).sum() / A.sum()`` per layer and it is EXACTLY zero under a
    correct ``-inf`` causal mask, so the default tolerance is exactly zero.

    The GDN side is checked from the other direction: the pre-hook asserts the
    mask it was handed is ``None``, which is what an unpadded batch must produce
    -- a 4D causal mask arriving here would be silently discarded, so seeing it
    at all means the dispatch is wrong.
    """

    layers = harness.layers()
    seen: dict[int, dict[str, Any]] = {}
    handles = []

    def attention_hook(position: int):
        def hook(module, args, kwargs, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise SystemExit(
                    f"layer {position}: attention returned no weights -- "
                    f"attn_implementation is {harness.model.config._attn_implementation!r}, "
                    f"and only 'eager' can be checked for causality"
                )
            A = weights.detach().to(torch.float64)
            above = float(torch.triu(A, diagonal=1).sum())
            total = float(A.sum())
            row = seen.setdefault(position, {"kind": FULL, "above_diagonal_mass": 0.0, "total": 0.0})
            row["above_diagonal_mass"] += above
            row["total"] += total
            return None

        return hook

    def linear_hook(position: int):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
            row = seen.setdefault(position, {"kind": LINEAR, "masks_seen": []})
            row["masks_seen"].append(None if mask is None else list(mask.shape))
            return None

        return hook

    for position, layer in enumerate(layers):
        if qf_arch.layer_kind(layer) == FULL:
            handles.append(
                layer.self_attn.register_forward_hook(attention_hook(position), with_kwargs=True)
            )
        else:
            handles.append(
                layer.linear_attn.register_forward_pre_hook(linear_hook(position), with_kwargs=True)
            )
    try:
        for index, ids in harness.batches[:batches]:
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    report: dict[str, Any] = {}
    offenders: list[str] = []
    tc = qf_arch.text_config(harness.model.config)
    for position in sorted(seen):
        row = seen[position]
        entry = {
            "parent_layer": position,
            "kind": row["kind"],
            "config_says": tc.layer_types[position],
        }
        if row["kind"] == FULL:
            fraction = row["above_diagonal_mass"] / max(row["total"], 1e-300)
            entry["above_diagonal_mass"] = fraction
            if fraction > tolerance:
                offenders.append(
                    f"layer {position} attends to the future: above-diagonal mass "
                    f"{fraction:.6f} > {tolerance}"
                )
        else:
            entry["masks_seen"] = row["masks_seen"]
            if strict_linear and any(shape is not None for shape in row["masks_seen"]):
                offenders.append(
                    f"layer {position} is a GDN block and was handed a mask "
                    f"{row['masks_seen']}; the config calls it "
                    f"{tc.layer_types[position]!r}"
                )
        report[str(position)] = entry
    if offenders:
        raise SystemExit(
            "causal-mask assertion failed:\n  " + "\n  ".join(offenders)
        )
    return report


@torch.no_grad()
def above_diagonal_mass(harness, *, batches: int = 1) -> dict[int, float]:
    """The same measurement without the refusal -- for the negative control."""

    report = assert_causal_masks(
        harness, tolerance=float("inf"), batches=batches, strict_linear=False
    )
    return {
        int(position): float(row.get("above_diagonal_mass", 0.0))
        for position, row in report.items()
        if row["kind"] == FULL
    }


def unrepaired_excision(model: torch.nn.Module, index: int) -> None:
    """Delete one layer and DELIBERATELY leave the config alone.

    The negative control for :func:`assert_causal_masks`.  A test that only ever
    sees zero above-diagonal mass has not shown that the instrument can see
    anything; this makes the trap happen on purpose so the assertion can be
    caught firing.  Never call it in a build.
    """

    del qf_arch.layers(model)[index]
    qf_arch.reindex_layers(model)


def excision_census(harness, *, batches: int = 1) -> dict[str, Any]:
    """Per-layer contribution to the residual stream, as an ORDERING for the ladder.

    ``||block output|| / ||block input||`` at each layer, on the fit split.  It
    ranks which single-layer children to build first and it closes nothing: a
    layer is dead only when a BUILT child says so on the review.
    """

    layers = harness.layers()
    targets = {f"L{i:03d}": layer for i, layer in enumerate(layers)}
    captured = harness.capture(targets, outputs=targets)
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        name = f"L{index:03d}"
        X = captured[name]["fit"]
        Y = captured[name]["out_fit"]
        delta = Y - X
        rows.append(
            {
                "layer": index,
                "kind": qf_arch.layer_kind(layer),
                "input_norm": float(X.norm()),
                "delta_norm": float(delta.norm()),
                "relative_contribution": float(delta.norm() / X.norm().clamp_min(1e-30)),
                "cos_input_output": float(
                    torch.nn.functional.cosine_similarity(X, Y, dim=1).mean()
                ),
            }
        )
    ordered = sorted(rows, key=lambda r: r["relative_contribution"])
    return {
        "schema": "neural-forge-omni-qwen3_5-excision-census-v1",
        "layers": rows,
        "cheapest_first": [row["layer"] for row in ordered],
        "note": (
            "an ordering for the excision ladder, not a verdict; every layer stays "
            "open until a built child is reviewed"
        ),
    }
