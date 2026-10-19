"""
test_task_allocation.py
=======================
Offline unit tests for task allocation (greedy and PSO).

Tests cover:
  - Greedy assignment produces one target per robot (no duplicates for N≤targets).
  - Greedy assignment returns valid target indices.
  - PSO assignment returns valid target indices.
  - Allocation cost is non-negative.
  - PSO cost ≤ greedy cost * some tolerance factor (PSO should not be much worse).
  - duplicate_count utility function counts correctly.
  - run_allocation_experiment returns expected structure.

No Webots connection required.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from task_allocation import (
    greedy_allocation,
    allocation_cost,
    duplicate_count,
    run_allocation_experiment,
)
from particle_swarm import run_pso
from configuration_b import ConfigB


@pytest.fixture
def cfg() -> ConfigB:
    return ConfigB()


# ------------------------------------------------------------------ #
#  Greedy allocation                                                   #
# ------------------------------------------------------------------ #

class TestGreedyAllocation:
    def test_returns_correct_length(self, cfg):
        assign = greedy_allocation(cfg.ROBOT_STARTS, cfg.TARGETS)
        assert len(assign) == cfg.NUM_ROBOTS

    def test_valid_target_indices(self, cfg):
        assign = greedy_allocation(cfg.ROBOT_STARTS, cfg.TARGETS)
        for t in assign:
            assert 0 <= t < cfg.NUM_TARGETS, f"Target index {t} out of range"

    def test_no_duplicate_assignments_when_enough_targets(self, cfg):
        """When num_targets >= num_robots, greedy should assign distinct targets."""
        assign = greedy_allocation(cfg.ROBOT_STARTS, cfg.TARGETS)
        dups = duplicate_count(assign, cfg.NUM_TARGETS)
        assert dups == 0, f"Greedy produced {dups} duplicate(s)"

    def test_nearest_target_selected(self):
        """Robot at origin should be assigned to target at (0,1), not (10,0)."""
        robots  = [(0.0, 0.0)]
        targets = [(10.0, 0.0), (0.0, 1.0)]
        assign = greedy_allocation(robots, targets)
        assert assign[0] == 1, "Should assign the closer target"

    def test_three_robots_three_targets(self):
        """Aligned robots and targets: each robot gets its nearest target."""
        robots  = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        targets = [(0.0, 1.0), (5.0, 1.0), (10.0, 1.0)]
        assign = greedy_allocation(robots, targets)
        assert assign == [0, 1, 2] or len(set(assign)) == 3


# ------------------------------------------------------------------ #
#  Allocation cost                                                     #
# ------------------------------------------------------------------ #

class TestAllocationCost:
    def test_cost_non_negative(self, cfg):
        assign = greedy_allocation(cfg.ROBOT_STARTS, cfg.TARGETS)
        cost = allocation_cost(assign, cfg.ROBOT_STARTS, cfg.TARGETS)
        assert cost >= 0.0

    def test_cost_zero_when_at_target(self):
        robots  = [(1.0, 2.0)]
        targets = [(1.0, 2.0)]
        assign = [0]
        cost = allocation_cost(assign, robots, targets)
        assert abs(cost) < 1e-9

    def test_cost_equals_euclidean(self):
        robots  = [(0.0, 0.0)]
        targets = [(3.0, 4.0)]
        assign = [0]
        cost = allocation_cost(assign, robots, targets)
        assert abs(cost - 5.0) < 1e-9


# ------------------------------------------------------------------ #
#  Duplicate count                                                     #
# ------------------------------------------------------------------ #

class TestDuplicateCount:
    def test_no_duplicates(self):
        assert duplicate_count([0, 1, 2], 3) == 0

    def test_one_duplicate(self):
        # Both robots 0 and 1 assigned to target 0
        assert duplicate_count([0, 0, 2], 3) == 1

    def test_all_same(self):
        assert duplicate_count([1, 1, 1], 3) == 2

    def test_empty(self):
        assert duplicate_count([], 3) == 0


# ------------------------------------------------------------------ #
#  PSO allocation                                                      #
# ------------------------------------------------------------------ #

class TestPSOAllocation:
    def test_returns_correct_length(self, cfg):
        result = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg)
        assert len(result.assignment) == cfg.NUM_ROBOTS

    def test_valid_target_indices(self, cfg):
        result = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg)
        for t in result.assignment:
            assert 0 <= t < cfg.NUM_TARGETS

    def test_fitness_history_non_empty(self, cfg):
        result = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg)
        assert len(result.fitness_history) > 0

    def test_fitness_non_increasing(self, cfg):
        """Global best fitness should never increase."""
        result = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg)
        for i in range(1, len(result.fitness_history)):
            assert result.fitness_history[i] <= result.fitness_history[i - 1] + 1e-9, \
                f"Fitness increased at iteration {i}"

    def test_reproducible_with_same_seed(self, cfg):
        r1 = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg, seed=42)
        r2 = run_pso(cfg.ROBOT_STARTS, cfg.TARGETS, cfg, seed=42)
        assert r1.assignment == r2.assignment
        assert abs(r1.cost - r2.cost) < 1e-9


# ------------------------------------------------------------------ #
#  run_allocation_experiment                                           #
# ------------------------------------------------------------------ #

class TestRunAllocationExperiment:
    def test_structure(self, cfg):
        result = run_allocation_experiment(cfg)
        assert "pso" in result
        assert "greedy" in result
        assert "robot_positions" in result
        assert "target_positions" in result

    def test_costs_positive(self, cfg):
        result = run_allocation_experiment(cfg)
        assert result["pso"]["cost"] >= 0
        assert result["greedy"]["cost"] >= 0

    def test_pso_not_catastrophically_worse(self, cfg):
        """PSO should not produce a cost more than 3× greedy."""
        result = run_allocation_experiment(cfg)
        if result["greedy"]["cost"] > 0:
            ratio = result["pso"]["cost"] / result["greedy"]["cost"]
            assert ratio < 3.0, f"PSO cost ({result['pso']['cost']:.2f}) is {ratio:.1f}× greedy"
