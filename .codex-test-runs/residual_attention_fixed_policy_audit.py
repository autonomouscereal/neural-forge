"""Read-only audit of singular/four-policy sequence-effective V/O arms."""

from __future__ import annotations

import json

import residual_attention_sequence_operator_experiment as experiment


def main() -> None:
    policies = (
        (1e-10, 0.25),
        (1e-10, 0.5),
        (1e-4, 0.25),
        (1e-4, 0.5),
    )
    results = {}
    for seed in (1101, 2202, 3303):
        parent = experiment._make_parent(seed)
        fit = experiment._make_dataset(
            parent,
            seed=seed + 10_000,
            batch_size=28,
        )
        holdout = experiment._make_dataset(
            parent,
            seed=seed + 30_000,
            batch_size=28,
        )
        (
            _direct,
            producer,
            _consumer,
            direct_state,
            split_state,
        ) = experiment._compile_current_frontier(
            parent,
            fit,
            seed=seed + 40_000,
        )
        baselines = {
            "direct_monolithic": {
                "endpoint": experiment._teacher_forced_metrics(
                    direct_state,
                    holdout,
                )["endpoint_rrmse"],
                "drift": experiment._closed_loop_metrics(
                    direct_state,
                    holdout,
                    parent,
                )["mean_decode_endpoint_rrmse"],
            },
            "residual_attention_split": {
                "endpoint": experiment._teacher_forced_metrics(
                    split_state,
                    holdout,
                )["endpoint_rrmse"],
                "drift": experiment._closed_loop_metrics(
                    split_state,
                    holdout,
                    parent,
                )["mean_decode_endpoint_rrmse"],
            },
        }
        qk = {
            "query": producer.query_weight.to(experiment.DTYPE),
            "key": producer.key_weight.to(experiment.DTYPE),
        }
        candidates = {}
        for ridge, anchor in policies:
            state, _receipt = experiment._fit_sequence_effective_operator(
                qk,
                fit.cases,
                ridge_scale=ridge,
                horizon_power=0.0,
                value_subspace_anchor=producer.value_weight,
                value_anchor_fraction=anchor,
            )
            endpoint = experiment._teacher_forced_metrics(
                state,
                holdout,
            )["endpoint_rrmse"]
            drift = experiment._closed_loop_metrics(
                state,
                holdout,
                parent,
            )["mean_decode_endpoint_rrmse"]
            candidates[f"ridge={ridge:g},anchor={anchor:g}"] = {
                "endpoint": endpoint,
                "drift": drift,
                "beats_both": all(
                    endpoint < value["endpoint"] and drift < value["drift"]
                    for value in baselines.values()
                ),
            }
        results[str(seed)] = {
            "baselines": baselines,
            "candidates": candidates,
        }
    singular = {}
    for ridge, anchor in policies:
        name = f"ridge={ridge:g},anchor={anchor:g}"
        singular[name] = all(
            results[str(seed)]["candidates"][name]["beats_both"]
            for seed in (1101, 2202, 3303)
        )
    print(
        json.dumps(
            {
                "per_seed": results,
                "singular_policy_beats_both_every_seed": singular,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
