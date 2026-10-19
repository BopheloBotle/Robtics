"""
task_allocation.py
==================
Task (target) allocation for Component B.

Two methods are implemented:
  1. PSO-assisted allocation  — uses PSO to minimise total travel distance.
  2. Greedy allocation        — assigns each robot to its nearest unassigned target.

Both methods return a list of (robot_index → target_index) assignments.
The greedy method serves as the baseline for comparison.

run_allocation_experiment() runs both methods on a set of test scenarios
and returns a structured results dict used by evaluation.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from particle_swarm import run_pso, PSOResult

if TYPE_CHECKING:
    from configuration_b import ConfigB


def greedy_allocation(
    robot_positions: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
) -> list[int]:
    """
    Greedy nearest-neighbour allocation.

    Each robot is assigned to the closest unassigned target in order of
    robot index.  Simple and fast but not globally optimal.

    Returns assignment[i] = target_index for robot i.
    """
    available = set(range(len(target_positions)))
    assignment: list[int] = []

    for rx, ry in robot_positions:
        best_t = -1
        best_d = float("inf")
        for t_idx in available:
            tx, ty = target_positions[t_idx]
            d = math.hypot(tx - rx, ty - ry)
            if d < best_d:
                best_d = d
                best_t = t_idx
        if best_t >= 0:
            assignment.append(best_t)
            available.discard(best_t)
        else:
            # No unassigned target left — assign to nearest regardless
            t_idx = min(
                range(len(target_positions)),
                key=lambda i: math.hypot(
                    target_positions[i][0] - rx,
                    target_positions[i][1] - ry,
                ),
            )
            assignment.append(t_idx)

    return assignment


def allocation_cost(
    assignment: list[int],
    robot_positions: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
) -> float:
    """Total travel distance for a given assignment."""
    total = 0.0
    for i, t_idx in enumerate(assignment):
        rx, ry = robot_positions[i]
        tx, ty = target_positions[t_idx]
        total += math.hypot(tx - rx, ty - ry)
    return total


def duplicate_count(assignment: list[int], n_targets: int) -> int:
    """Number of targets assigned to more than one robot."""
    counts = [0] * n_targets
    for t in assignment:
        if 0 <= t < n_targets:
            counts[t] += 1
    return sum(max(0, c - 1) for c in counts)


def run_allocation_experiment(cfg: "ConfigB") -> dict:
    """
    Compare PSO vs greedy allocation on the configured scenario.

    Returns a dict with PSO and greedy results including cost, assignment,
    duplicate count, and PSO fitness history.
    """
    robot_pos  = list(cfg.ROBOT_STARTS)
    target_pos = list(cfg.TARGETS)

    # --- PSO ---
    pso_result: PSOResult = run_pso(robot_pos, target_pos, cfg)

    # --- Greedy ---
    greedy_assign = greedy_allocation(robot_pos, target_pos)
    greedy_cost   = allocation_cost(greedy_assign, robot_pos, target_pos)
    greedy_dups   = duplicate_count(greedy_assign, len(target_pos))

    # --- Metrics ---
    pso_cost_raw = allocation_cost(pso_result.assignment, robot_pos, target_pos)
    # Note: pso_result.cost includes duplicate penalty, pso_cost_raw is travel only

    return {
        "pso": {
            "assignment": pso_result.assignment,
            "cost": pso_cost_raw,       # travel distance only for fair comparison
            "cost_with_penalty": pso_result.cost,
            "duplicate_count": pso_result.duplicate_count,
            "fitness_history": pso_result.fitness_history,
            "iterations": pso_result.iterations,
        },
        "greedy": {
            "assignment": greedy_assign,
            "cost": greedy_cost,
            "duplicate_count": greedy_dups,
        },
        "robot_positions": robot_pos,
        "target_positions": target_pos,
    }
