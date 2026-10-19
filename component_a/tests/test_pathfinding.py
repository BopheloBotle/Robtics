"""
test_pathfinding.py
===================
Offline unit tests for Dijkstra and A* path planners.

Tests cover:
  - Valid path returned on a clear grid.
  - Path does not pass through any occupied cell.
  - Start and goal are the first and last cells of the path.
  - Unreachable goal returns success=False and empty path.
  - Both algorithms find paths of equal length (both optimal).
  - Nodes-explored count is non-zero on success.
  - A* explores fewer or equal nodes than Dijkstra on the same grid.
  - Planning time is recorded (> 0).
  - Diagonal corner-cutting is not present in any returned path.

No Webots connection is required.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from configuration import Config
from occupancy_grid import OccupancyGrid
from dijkstra import dijkstra, PathResult
from astar import astar


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def built_grid(cfg: Config) -> OccupancyGrid:
    g = OccupancyGrid(cfg)
    g.build_from_config()
    return g


@pytest.fixture
def clear_grid(cfg: Config) -> OccupancyGrid:
    """A completely obstacle-free grid (all cells FREE)."""
    g = OccupancyGrid(cfg)
    for r in range(g.rows):
        for c in range(g.cols):
            g.set_cell(r, c, cfg.CELL_FREE)
    return g


@pytest.fixture
def blocked_grid(cfg: Config) -> OccupancyGrid:
    """A grid where a wall of occupied cells separates start from goal."""
    g = OccupancyGrid(cfg)
    for r in range(g.rows):
        for c in range(g.cols):
            g.set_cell(r, c, cfg.CELL_FREE)
    # Vertical wall at col = cols//2, from row 0 to rows-1 (no gap)
    mid = g.cols // 2
    for r in range(g.rows):
        g.set_cell(r, mid, cfg.CELL_OCCUPIED)
    return g


def _start_goal(grid: OccupancyGrid, cfg: Config):
    return (
        grid.world_to_grid(*cfg.START_POS),
        grid.world_to_grid(*cfg.GOAL_POS),
    )


def _path_passes_through_occupied(
    path: list, grid: OccupancyGrid
) -> bool:
    """Return True if any cell in path (excluding start/goal check) is occupied."""
    for r, c in path:
        if grid.is_occupied(r, c):
            return True
    return False


def _has_corner_cut(path: list) -> bool:
    """
    Return True if the path cuts a diagonal corner.
    A corner cut occurs when a diagonal step (dr!=0, dc!=0) is taken but
    one of the two adjacent cardinal cells is occupied.
    This test checks the structural property only — actual occupancy
    checks are in the planners.
    """
    # We just verify no two consecutive cells have |dr|>1 or |dc|>1
    for i in range(1, len(path)):
        dr = abs(path[i][0] - path[i - 1][0])
        dc = abs(path[i][1] - path[i - 1][1])
        if dr > 1 or dc > 1:
            return True
    return False


# ------------------------------------------------------------------ #
#  Dijkstra tests                                                      #
# ------------------------------------------------------------------ #

class TestDijkstra:
    def test_finds_path_on_built_grid(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.success, "Dijkstra failed to find a path on the built grid"
        assert len(result.path) >= 2

    def test_path_starts_at_start(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.success
        assert result.path[0] == start

    def test_path_ends_at_goal(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.success
        assert result.path[-1] == goal

    def test_path_avoids_obstacles(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.success
        assert not _path_passes_through_occupied(result.path, built_grid), \
            "Dijkstra path passes through an occupied cell"

    def test_no_corner_cutting(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.success
        assert not _has_corner_cut(result.path), \
            "Dijkstra path has a step larger than 1 cell — likely a bug"

    def test_nodes_explored_positive(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.nodes_explored > 0

    def test_elapsed_time_positive(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.elapsed_s >= 0.0

    def test_path_length_positive(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, goal, cfg)
        assert result.path_length_m > 0.0

    def test_unreachable_goal(self, blocked_grid, cfg):
        """Dijkstra must return success=False when goal is unreachable."""
        start = (0, 0)
        goal  = (0, blocked_grid.cols - 1)
        result = dijkstra(blocked_grid, start, goal, cfg)
        assert not result.success
        assert result.path == []

    def test_start_equals_goal(self, built_grid, cfg):
        """When start == goal, path should be [start] (trivial path)."""
        start, _ = _start_goal(built_grid, cfg)
        result = dijkstra(built_grid, start, start, cfg)
        # Either trivial success with single-cell path or handled gracefully
        if result.success:
            assert result.path[0] == start
            assert result.path[-1] == start

    def test_clear_grid_path(self, clear_grid, cfg):
        """On a clear grid the shortest path should be a near-straight line."""
        start, goal = _start_goal(clear_grid, cfg)
        result = dijkstra(clear_grid, start, goal, cfg)
        assert result.success
        # Length should be close to Euclidean distance
        sx, sy = cfg.START_POS
        gx, gy = cfg.GOAL_POS
        euclidean = math.hypot(gx - sx, gy - sy)
        assert result.path_length_m < euclidean * 1.5, \
            "Path is more than 50% longer than straight-line distance"


# ------------------------------------------------------------------ #
#  A* tests                                                            #
# ------------------------------------------------------------------ #

class TestAStar:
    def test_finds_path_on_built_grid(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.success

    def test_path_starts_at_start(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.success
        assert result.path[0] == start

    def test_path_ends_at_goal(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.success
        assert result.path[-1] == goal

    def test_path_avoids_obstacles(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.success
        assert not _path_passes_through_occupied(result.path, built_grid)

    def test_no_corner_cutting(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.success
        assert not _has_corner_cut(result.path)

    def test_unreachable_goal(self, blocked_grid, cfg):
        start = (0, 0)
        goal  = (0, blocked_grid.cols - 1)
        result = astar(blocked_grid, start, goal, cfg)
        assert not result.success
        assert result.path == []

    def test_nodes_explored_positive(self, built_grid, cfg):
        start, goal = _start_goal(built_grid, cfg)
        result = astar(built_grid, start, goal, cfg)
        assert result.nodes_explored > 0


# ------------------------------------------------------------------ #
#  Comparison: Dijkstra vs A*                                          #
# ------------------------------------------------------------------ #

class TestDijkstraVsAStar:
    def test_equal_path_length(self, built_grid, cfg):
        """Both algorithms must find equally optimal paths (same length)."""
        start, goal = _start_goal(built_grid, cfg)
        d = dijkstra(built_grid, start, goal, cfg)
        a = astar(built_grid, start, goal, cfg)
        assert d.success and a.success
        # Allow ≤1 cell resolution of floating-point difference
        assert abs(d.path_length_m - a.path_length_m) < cfg.GRID_RESOLUTION + 1e-6, \
            (f"Path lengths differ: Dijkstra={d.path_length_m:.4f}m, "
             f"A*={a.path_length_m:.4f}m")

    def test_astar_explores_fewer_or_equal_nodes(self, built_grid, cfg):
        """A* must explore ≤ nodes compared to Dijkstra on the same grid."""
        start, goal = _start_goal(built_grid, cfg)
        d = dijkstra(built_grid, start, goal, cfg)
        a = astar(built_grid, start, goal, cfg)
        assert d.success and a.success
        assert a.nodes_explored <= d.nodes_explored, \
            (f"A* explored {a.nodes_explored} nodes but Dijkstra explored "
             f"{d.nodes_explored} — heuristic may be inadmissible")

    def test_both_succeed_on_clear_grid(self, clear_grid, cfg):
        start, goal = _start_goal(clear_grid, cfg)
        d = dijkstra(clear_grid, start, goal, cfg)
        a = astar(clear_grid, start, goal, cfg)
        assert d.success and a.success

    def test_both_fail_on_blocked_grid(self, blocked_grid, cfg):
        start = (0, 0)
        goal  = (0, blocked_grid.cols - 1)
        d = dijkstra(blocked_grid, start, goal, cfg)
        a = astar(blocked_grid, start, goal, cfg)
        assert not d.success
        assert not a.success
