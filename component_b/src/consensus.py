"""
consensus.py
============
Average consensus algorithm for Component B.

Average consensus
-----------------
Each agent i maintains a scalar state x_i.  At each iteration, agent i
updates its state using the states of its ring neighbours j ∈ N(i):

  x_i(k+1) = x_i(k) + α × Σ_{j ∈ N(i)} (x_j(k) − x_i(k))

where α is the update gain.

Convergence condition
---------------------
For a connected, undirected graph and constant gain α, the algorithm
converges to the average of the initial values if:

  0 < α < 1 / max_degree

For a 3-agent ring, max_degree = 2, so α < 0.5.
The default α = 0.3 satisfies this condition.

The consensus state here represents a shared estimate of the number of
remaining targets (a meaningful distributed quantity in the swarm task).

Measurement
-----------
Convergence error at iteration k:
  e(k) = max_i |x_i(k) − x_avg| / |x_avg|   (relative max deviation)

The algorithm is declared converged when e(k) < CONSENSUS_EPSILON.

run_consensus_experiment() runs the full offline experiment and returns
all agent trajectories and the convergence profile for plotting.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configuration_b import ConfigB


def average_consensus_step(
    states: list[float],
    alpha: float,
    adjacency: list[list[int]],
) -> list[float]:
    """
    Perform one synchronous average-consensus update.

    Parameters
    ----------
    states    : current state of each agent
    alpha     : update gain
    adjacency : adjacency list — adjacency[i] = list of neighbour indices

    Returns
    -------
    new_states : updated state list (same length as states)
    """
    n = len(states)
    new_states = states.copy()
    for i in range(n):
        update = sum(states[j] - states[i] for j in adjacency[i])
        new_states[i] = states[i] + alpha * update
    return new_states


def _ring_adjacency(n: int) -> list[list[int]]:
    """Return adjacency list for an n-node ring graph."""
    return [[(i - 1) % n, (i + 1) % n] for i in range(n)]


def run_consensus_experiment(cfg: "ConfigB") -> dict:
    """
    Run the average consensus experiment offline.

    Returns a dict with:
      agent_histories   : list of lists — agent_histories[i] = x_i values per iter
      errors            : relative max deviation per iteration
      iterations        : number of iterations until convergence (or max_iter)
      final_error       : convergence error at termination
      converged         : bool
      true_average      : theoretical consensus value
    """
    n = cfg.CONSENSUS_NUM_AGENTS
    alpha = cfg.CONSENSUS_ALPHA
    epsilon = cfg.CONSENSUS_EPSILON
    max_iter = cfg.CONSENSUS_MAX_ITER
    init_vals = list(cfg.CONSENSUS_INIT_VALUES[:n])

    adjacency = _ring_adjacency(n)
    true_avg = sum(init_vals) / n

    states = init_vals.copy()
    histories: list[list[float]] = [[v] for v in states]
    errors: list[float] = []

    converged_at = max_iter
    converged = False

    for k in range(max_iter):
        states = average_consensus_step(states, alpha, adjacency)
        for i in range(n):
            histories[i].append(states[i])

        max_dev = max(abs(s - true_avg) for s in states)
        rel_err = max_dev / abs(true_avg) if abs(true_avg) > 1e-12 else max_dev
        errors.append(rel_err)

        if rel_err < epsilon and not converged:
            converged = True
            converged_at = k + 1

        if converged and k > converged_at + 5:
            break

    return {
        "agent_histories": histories,
        "errors": errors,
        "iterations": converged_at,
        "final_error": errors[-1],
        "converged": converged,
        "true_average": true_avg,
        "init_values": init_vals,
        "alpha": alpha,
    }


def online_consensus_update(
    current_val: float,
    neighbour_vals: list[float],
    alpha: float,
) -> float:
    """
    Single-step consensus update for use inside the Webots controller.

    Parameters
    ----------
    current_val    : this agent's current state
    neighbour_vals : list of received neighbour states
    alpha          : update gain

    Returns
    -------
    Updated state value.
    """
    update = sum(v - current_val for v in neighbour_vals)
    return current_val + alpha * update
