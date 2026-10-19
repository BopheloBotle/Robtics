"""
particle_swarm.py
=================
Particle Swarm Optimisation (PSO) for task allocation.

Purpose
-------
The swarm has NUM_ROBOTS robots and NUM_TARGETS targets.  PSO finds an
assignment of robots to targets that minimises total travel distance
(the allocation cost).  The result is used by task_allocation.py to
send each robot to its assigned target.

Particle representation
-----------------------
Each particle encodes one complete assignment as a vector of NUM_ROBOTS
real-valued scalars x_i ∈ [0, NUM_TARGETS).  The integer floor of x_i
gives the target index assigned to robot i.  Duplicate assignments are
penalised in the fitness function.

Fitness function
----------------
  cost = total travel distance + PENALTY × duplicate_count

where total travel distance = Σ_i dist(robot_i_start, assigned_target_i)
and duplicate_count = number of targets assigned to more than one robot.

The fitness is minimised (lower is better).

PSO update equations
--------------------
  v_i(t+1) = w × v_i(t)
            + c1 × r1 × (pbest_i − x_i(t))
            + c2 × r2 × (gbest − x_i(t))
  x_i(t+1) = x_i(t) + v_i(t+1)

where w, c1, c2 are from ConfigB and r1, r2 ∈ Uniform[0,1].
Velocity is clipped to ±PSO_V_MAX per dimension.
Position is clipped to [0, NUM_TARGETS − ε] per dimension.

Limitations documented
----------------------
- With only 3 robots, PSO is overkill.  It is included to satisfy the
  assignment requirement and to demonstrate the method.
- The continuous-to-discrete encoding introduces aliasing.  A
  dedicated combinatorial solver (e.g. Hungarian algorithm) would be
  more efficient for small instances.
- PSO results depend on the random seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configuration_b import ConfigB


@dataclass
class PSOResult:
    """Output of a PSO run."""
    assignment: list[int]          # assignment[i] = target index for robot i
    cost: float                    # final fitness (total distance + penalties)
    fitness_history: list[float]   # best fitness per iteration
    iterations: int
    duplicate_count: int


def _decode_assignment(x: np.ndarray, num_targets: int) -> list[int]:
    """Map continuous particle position to integer target indices."""
    return [int(min(math.floor(xi), num_targets - 1)) for xi in x]


def _fitness(
    x: np.ndarray,
    robot_positions: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
    penalty: float = 5.0,
) -> float:
    """
    Compute the fitness (cost) for particle x.

    Lower is better.
    """
    n_robots  = len(robot_positions)
    n_targets = len(target_positions)
    assignment = _decode_assignment(x, n_targets)

    total_dist = 0.0
    for i, t_idx in enumerate(assignment):
        rx, ry = robot_positions[i]
        tx, ty = target_positions[t_idx]
        total_dist += math.hypot(tx - rx, ty - ry)

    # Count duplicate assignments
    counts = [0] * n_targets
    for t_idx in assignment:
        counts[t_idx] += 1
    duplicates = sum(max(0, c - 1) for c in counts)

    return total_dist + penalty * duplicates


def run_pso(
    robot_positions: list[tuple[float, float]],
    target_positions: list[tuple[float, float]],
    cfg: "ConfigB",
    seed: int | None = None,
) -> PSOResult:
    """
    Run PSO to find a near-optimal robot-to-target assignment.

    Parameters
    ----------
    robot_positions  : list of (x, y) for each robot
    target_positions : list of (x, y) for each target
    cfg              : ConfigB
    seed             : random seed (defaults to cfg.RANDOM_SEED)

    Returns
    -------
    PSOResult
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.RANDOM_SEED)
    n_robots  = len(robot_positions)
    n_targets = len(target_positions)
    n_dim     = n_robots   # one dimension per robot

    # Position bounds: [0, n_targets)
    pos_min = np.zeros(n_dim)
    pos_max = np.full(n_dim, n_targets - 1e-6)

    n_particles = cfg.PSO_NUM_PARTICLES
    max_iter    = cfg.PSO_MAX_ITER
    w  = cfg.PSO_W
    c1 = cfg.PSO_C1
    c2 = cfg.PSO_C2
    v_max = cfg.PSO_V_MAX

    # Initialise particles
    positions  = rng.uniform(0, n_targets - 1e-6, size=(n_particles, n_dim))
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, n_dim))

    pbest_pos = positions.copy()
    pbest_fit = np.array([
        _fitness(positions[p], robot_positions, target_positions)
        for p in range(n_particles)
    ])

    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    fitness_history: list[float] = [gbest_fit]

    for _ in range(max_iter):
        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, pos_min, pos_max)

        for p in range(n_particles):
            fit = _fitness(positions[p], robot_positions, target_positions)
            if fit < pbest_fit[p]:
                pbest_fit[p] = fit
                pbest_pos[p] = positions[p].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = positions[p].copy()

        fitness_history.append(gbest_fit)

    best_assignment = _decode_assignment(gbest_pos, n_targets)
    counts = [0] * n_targets
    for t in best_assignment:
        counts[t] += 1
    dup = sum(max(0, c - 1) for c in counts)

    return PSOResult(
        assignment=best_assignment,
        cost=gbest_fit,
        fitness_history=fitness_history,
        iterations=max_iter,
        duplicate_count=dup,
    )
