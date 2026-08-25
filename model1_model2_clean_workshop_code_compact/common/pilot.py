"""Data-only one-dimensional pilot helpers shared by both models."""

from __future__ import annotations

import numpy as np


def cumulative_trapezoid_rows(score: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Integrate each score-field row, fixing the left-edge potential to zero."""
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    if score.ndim != 2 or score.shape[1] != u_grid.size:
        raise ValueError("score must have shape (n, len(u_grid))")
    increments = 0.5 * (score[:, 1:] + score[:, :-1]) * np.diff(u_grid)[None, :]
    return np.concatenate(
        [np.zeros((score.shape[0], 1), dtype=np.float64), np.cumsum(increments, axis=1)],
        axis=1,
    )


def constrained_modes_from_score_grid(
    score: np.ndarray,
    u_grid: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Select the global constrained mode implied by each 1-D score field."""
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    if score.ndim != 2 or score.shape[1] != u_grid.size:
        raise ValueError("score must have shape (n, len(u_grid))")
    potential = cumulative_trapezoid_rows(score, u_grid)
    roots = np.empty(score.shape[0], dtype=np.float64)
    statuses: list[str] = []
    n_stationary = np.zeros(score.shape[0], dtype=np.int64)
    residual = np.empty(score.shape[0], dtype=np.float64)

    for i, (row, row_potential) in enumerate(zip(score, potential, strict=True)):
        candidates: list[tuple[float, float, float, str]] = [
            (float(u_grid[0]), float(row_potential[0]), abs(float(row[0])), "left_boundary"),
            (float(u_grid[-1]), float(row_potential[-1]), abs(float(row[-1])), "right_boundary"),
        ]
        zero_mask = np.isclose(row, 0.0, atol=1e-12)
        crossings = np.flatnonzero(
            (row[:-1] * row[1:] < 0.0) & ~zero_mask[:-1] & ~zero_mask[1:]
        )
        exact_zero = np.flatnonzero(zero_mask)
        n_stationary[i] = int(crossings.size + exact_zero.size)
        for j in exact_zero:
            candidates.append((float(u_grid[j]), float(row_potential[j]), 0.0, "interior"))
        for j in crossings:
            left_u, right_u = float(u_grid[j]), float(u_grid[j + 1])
            left_s, right_s = float(row[j]), float(row[j + 1])
            weight = float(np.clip(-left_s / (right_s - left_s), 0.0, 1.0))
            root_u = left_u + weight * (right_u - left_u)
            root_potential = float(row_potential[j] + 0.5 * left_s * (root_u - left_u))
            candidates.append((root_u, root_potential, 0.0, "interior"))

        best = max(candidates, key=lambda item: (item[1], -item[2], item[3] == "interior"))
        roots[i] = best[0]
        residual[i] = best[2]
        if best[3] == "interior" and n_stationary[i] > 1:
            statuses.append("multiple_selected_global")
        else:
            statuses.append(best[3])
    return roots, statuses, n_stationary, residual

