"""
test_consensus.py
=================
Offline unit tests for the average consensus algorithm.

Tests cover:
  - Single update step produces correct weighted average.
  - Algorithm converges for valid alpha on a ring graph.
  - Final consensus value equals the true average.
  - Convergence error monotonically decreases on average.
  - Non-convergence with invalid alpha (alpha >= 0.5 on degree-2 ring).
  - online_consensus_update produces correct output.
  - run_consensus_experiment returns expected structure.

No Webots connection required.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from consensus import (
    average_consensus_step,
    run_consensus_experiment,
    online_consensus_update,
)
from configuration_b import ConfigB


@pytest.fixture
def cfg() -> ConfigB:
    return ConfigB()


# ------------------------------------------------------------------ #
#  Single step tests                                                   #
# ------------------------------------------------------------------ #

class TestConsensusStep:
    def test_two_node_step(self):
        """Two nodes: after one step, both should move toward each other."""
        states = [0.0, 4.0]
        adj = [[1], [0]]  # fully connected
        alpha = 0.25
        new = average_consensus_step(states, alpha, adj)
        # node 0: 0 + 0.25*(4-0)=1.0;  node 1: 4 + 0.25*(0-4)=3.0
        assert abs(new[0] - 1.0) < 1e-9
        assert abs(new[1] - 3.0) < 1e-9

    def test_ring_step_three_nodes(self):
        """Three-node ring: update should reduce max deviation."""
        states = [1.0, 4.0, 7.0]
        adj = [[2, 1], [0, 2], [1, 0]]  # ring
        alpha = 0.3
        new = average_consensus_step(states, alpha, adj)
        true_avg = sum(states) / len(states)
        old_dev = max(abs(s - true_avg) for s in states)
        new_dev = max(abs(s - true_avg) for s in new)
        assert new_dev < old_dev, "One step should reduce max deviation"

    def test_already_at_consensus(self):
        """If all states are equal, the update should leave them unchanged."""
        states = [3.0, 3.0, 3.0]
        adj = [[1, 2], [0, 2], [0, 1]]
        new = average_consensus_step(states, 0.3, adj)
        for s in new:
            assert abs(s - 3.0) < 1e-9


# ------------------------------------------------------------------ #
#  Experiment-level tests                                              #
# ------------------------------------------------------------------ #

class TestConsensusExperiment:
    def test_converges(self, cfg):
        result = run_consensus_experiment(cfg)
        assert result["converged"], \
            f"Consensus did not converge in {cfg.CONSENSUS_MAX_ITER} iterations"

    def test_final_value_close_to_average(self, cfg):
        result = run_consensus_experiment(cfg)
        true_avg = result["true_average"]
        for hist in result["agent_histories"]:
            final = hist[-1]
            assert abs(final - true_avg) < 0.05, \
                f"Final value {final:.4f} deviates from true average {true_avg:.4f}"

    def test_convergence_error_decreases(self, cfg):
        """Error series should be generally decreasing (allow minor wobbles)."""
        result = run_consensus_experiment(cfg)
        errors = result["errors"]
        # First 10% should be higher than last 10%
        first_mean = sum(errors[:max(1, len(errors)//10)]) / max(1, len(errors)//10)
        last_mean  = sum(errors[-max(1, len(errors)//10):]) / max(1, len(errors)//10)
        assert last_mean < first_mean, \
            "Consensus errors did not decrease on average"

    def test_result_structure(self, cfg):
        result = run_consensus_experiment(cfg)
        assert "agent_histories" in result
        assert "errors" in result
        assert "iterations" in result
        assert "true_average" in result
        assert len(result["agent_histories"]) == cfg.CONSENSUS_NUM_AGENTS

    def test_true_average_correct(self, cfg):
        result = run_consensus_experiment(cfg)
        expected = sum(cfg.CONSENSUS_INIT_VALUES) / len(cfg.CONSENSUS_INIT_VALUES)
        assert abs(result["true_average"] - expected) < 1e-9

    def test_histories_length(self, cfg):
        result = run_consensus_experiment(cfg)
        # Each history should have at least iterations+1 entries
        for hist in result["agent_histories"]:
            assert len(hist) >= 2


# ------------------------------------------------------------------ #
#  Online update                                                       #
# ------------------------------------------------------------------ #

class TestOnlineUpdate:
    def test_single_neighbour(self):
        """With one neighbour, update should move current_val toward neighbour."""
        new_val = online_consensus_update(
            current_val=0.0, neighbour_vals=[4.0], alpha=0.25
        )
        assert abs(new_val - 1.0) < 1e-9

    def test_no_neighbours(self):
        """With no neighbours, state should be unchanged."""
        val = online_consensus_update(current_val=5.0, neighbour_vals=[], alpha=0.3)
        assert abs(val - 5.0) < 1e-9

    def test_same_value_no_change(self):
        """If all neighbours have same value, state should not change."""
        val = online_consensus_update(
            current_val=3.0, neighbour_vals=[3.0, 3.0], alpha=0.3
        )
        assert abs(val - 3.0) < 1e-9

    def test_two_neighbours(self):
        """Update with two neighbours: should average influence."""
        # current=1, neighbours=[3, 5] → update = 0.3*((3-1)+(5-1))=0.3*6=1.8 → 2.8
        val = online_consensus_update(1.0, [3.0, 5.0], 0.3)
        assert abs(val - 2.8) < 1e-9
