"""Small exact oracle for a sufficient-stat sequence-effective attention solve.

The script proves that the proposed structured equations reproduce the current
explicit value-ALS coefficients and output-ridge normal equations.
"""

from __future__ import annotations

import json

import torch


def relative(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(left - right)
        / torch.linalg.vector_norm(right).clamp_min(torch.finfo(right.dtype).eps)
    )


def pcg(
    normal,
    right: torch.Tensor,
    diagonal: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, list[float]]:
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual / diagonal
    rz = torch.dot(residual, direction)
    initial = torch.linalg.vector_norm(residual)
    history = [1.0]
    for _ in range(iterations):
        applied = normal(direction)
        alpha = rz / torch.dot(direction, applied)
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        next_preconditioned = residual / diagonal
        next_rz = torch.dot(residual, next_preconditioned)
        direction = next_preconditioned + (next_rz / rz) * direction
        rz = next_rz
        history.append(float(torch.linalg.vector_norm(residual) / initial))
    return solution, history


def main() -> None:
    torch.manual_seed(20260719)
    dtype = torch.float64
    rows, heads, groups, hidden, head_dim, output = 37, 4, 2, 9, 3, 7
    heads_per_group = heads // groups
    attended = torch.randn(rows, heads, hidden, dtype=dtype)
    target = torch.randn(rows, output, dtype=dtype)
    values = torch.randn(groups, hidden, head_dim, dtype=dtype)
    outputs = torch.randn(heads, head_dim, output, dtype=dtype)
    cross = [
        [attended[:, h].T @ attended[:, k] for k in range(heads)]
        for h in range(heads)
    ]
    target_cross = [
        attended[:, h].T @ target for h in range(heads)
    ]

    group = 0
    start = group * heads_per_group
    stop = start + heads_per_group
    prediction = sum(
        attended[:, head] @ values[head // heads_per_group] @ outputs[head]
        for head in range(heads)
    )
    group_prediction = sum(
        attended[:, head] @ values[group] @ outputs[head]
        for head in range(start, stop)
    )
    residual_excluding = target - (prediction - group_prediction)

    coefficient = torch.einsum(
        "nhi,hdo->noid",
        attended[:, start:stop],
        outputs[start:stop],
    ).reshape(rows * output, hidden * head_dim)
    explicit_right = coefficient.T @ residual_excluding.reshape(-1)
    explicit_gram = coefficient.T @ coefficient

    structured_right_matrix = sum(
        target_cross[h] @ outputs[h].T for h in range(start, stop)
    )
    for h in range(start, stop):
        for k in range(heads):
            if start <= k < stop:
                continue
            structured_right_matrix -= (
                cross[h][k]
                @ values[k // heads_per_group]
                @ (outputs[k] @ outputs[h].T)
            )
    structured_right = structured_right_matrix.reshape(-1)

    def structured_normal_matrix(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(value)
        for h in range(start, stop):
            for k in range(start, stop):
                result += (
                    cross[h][k]
                    @ value
                    @ (outputs[k] @ outputs[h].T)
                )
        return result

    basis = torch.eye(hidden * head_dim, dtype=dtype)
    structured_gram = torch.stack(
        [
            structured_normal_matrix(column.reshape(hidden, head_dim)).reshape(-1)
            for column in basis.T
        ],
        dim=1,
    )
    explicit_trace_scale = float(coefficient.square().sum() / (hidden * head_dim))
    structured_trace_scale = sum(
        float(torch.trace(cross[h][k]))
        * float((outputs[h] * outputs[k]).sum())
        for h in range(start, stop)
        for k in range(start, stop)
    ) / (hidden * head_dim)
    ridge = 1e-4 * explicit_trace_scale
    regularized = explicit_gram + ridge * torch.eye(
        hidden * head_dim, dtype=dtype
    )
    exact_solution = torch.linalg.solve(regularized, explicit_right)
    diagonal = torch.diagonal(structured_gram) + ridge

    def flattened_normal(vector: torch.Tensor) -> torch.Tensor:
        return (
            structured_normal_matrix(vector.reshape(hidden, head_dim)).reshape(-1)
            + ridge * vector
        )

    pcg_solution, pcg_history = pcg(
        flattened_normal,
        structured_right,
        diagonal,
        hidden * head_dim,
    )

    contexts = torch.stack(
        [
            attended[:, head] @ values[head // heads_per_group]
            for head in range(heads)
        ],
        dim=1,
    )
    design = contexts.reshape(rows, heads * head_dim)
    explicit_output_gram = design.T @ design
    explicit_output_right = design.T @ target
    block_rows = []
    right_rows = []
    for h in range(heads):
        value_h = values[h // heads_per_group]
        right_rows.append(value_h.T @ target_cross[h])
        block_rows.append(
            torch.cat(
                [
                    value_h.T
                    @ cross[h][k]
                    @ values[k // heads_per_group]
                    for k in range(heads)
                ],
                dim=1,
            )
        )
    structured_output_gram = torch.cat(block_rows, dim=0)
    structured_output_right = torch.cat(right_rows, dim=0)

    output_flat = outputs.reshape(heads * head_dim, output)
    explicit_prediction = design @ output_flat
    explicit_sse = float((target - explicit_prediction).square().sum())
    stats_sse = float(
        target.square().sum()
        - 2.0 * (output_flat * structured_output_right).sum()
        + (output_flat * (structured_output_gram @ output_flat)).sum()
    )

    report = {
        "geometry": {
            "rows": rows,
            "heads": heads,
            "groups": groups,
            "hidden": hidden,
            "head_dim": head_dim,
            "output": output,
        },
        "value_right_relative_error": relative(
            structured_right, explicit_right
        ),
        "value_gram_relative_error": relative(
            structured_gram, explicit_gram
        ),
        "trace_scale_absolute_error": abs(
            structured_trace_scale - explicit_trace_scale
        ),
        "pcg_solution_relative_error_after_dimension_iterations": relative(
            pcg_solution, exact_solution
        ),
        "pcg_final_relative_normal_residual": pcg_history[-1],
        "output_gram_relative_error": relative(
            structured_output_gram, explicit_output_gram
        ),
        "output_right_relative_error": relative(
            structured_output_right, explicit_output_right
        ),
        "sse_relative_error": abs(stats_sse - explicit_sse) / explicit_sse,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
