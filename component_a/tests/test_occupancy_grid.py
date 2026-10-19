"""
test_occupancy_grid.py
======================
Offline unit tests for the OccupancyGrid class.

Tests cover:
  - World-to-grid coordinate conversion accuracy.
  - Grid-to-world coordinate conversion (round-trip).
  - Boundary checking (in-bounds and out-of-bounds).
  - Cell state setting and retrieval.
  - Obstacle marking from Config.
  - Obstacle inflation.
  - ValueError raised for out-of-range world coordinates.

No Webots connection is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math
import pytest
from configuration import Config
from occupancy_grid import OccupancyGrid


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def empty_grid(cfg: Config) -> OccupancyGrid:
    """A grid with all cells left as UNKNOWN (not built from config)."""
    return OccupancyGrid(cfg)


@pytest.fixture
def built_grid(cfg: Config) -> OccupancyGrid:
    """A grid built from Config obstacles (free + occupied cells)."""
    grid = OccupancyGrid(cfg)
    grid.build_from_config()
    return grid


# ------------------------------------------------------------------ #
#  Coordinate conversion                                               #
# ------------------------------------------------------------------ #

class TestCoordinateConversion:
    def test_world_to_grid_origin(self, empty_grid, cfg):
        """World origin → should map to bottom-left area of grid."""
        # The world origin (WORLD_ORIGIN_X, WORLD_ORIGIN_Y) should map to
        # the bottom-left cell (last row, first col) in image convention.
        row, col = empty_grid.world_to_grid(cfg.WORLD_ORIGIN_X + 0.05,
                                             cfg.WORLD_ORIGIN_Y + 0.05)
        assert row == empty_grid.rows - 1
        assert col == 0

    def test_world_to_grid_top_right(self, empty_grid, cfg):
        """Top-right world corner maps to top-right grid cell."""
        max_x = cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH  - 0.05
        max_y = cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT - 0.05
        row, col = empty_grid.world_to_grid(max_x, max_y)
        assert row == 0
        assert col == empty_grid.cols - 1

    def test_round_trip_centre(self, empty_grid):
        """grid_to_world → world_to_grid round-trip at grid centre."""
        r0, c0 = empty_grid.rows // 2, empty_grid.cols // 2
        wx, wy = empty_grid.grid_to_world(r0, c0)
        r1, c1 = empty_grid.world_to_grid(wx, wy)
        assert r0 == r1
        assert c0 == c1

    def test_round_trip_all_cells(self, empty_grid):
        """Round-trip conversion must be lossless for every cell centre."""
        for r in range(0, empty_grid.rows, 10):
            for c in range(0, empty_grid.cols, 10):
                wx, wy = empty_grid.grid_to_world(r, c)
                r2, c2 = empty_grid.world_to_grid(wx, wy)
                assert r == r2, f"row mismatch at ({r},{c}): got {r2}"
                assert c == c2, f"col mismatch at ({r},{c}): got {c2}"

    def test_out_of_bounds_raises(self, empty_grid, cfg):
        """world_to_grid raises ValueError for positions outside the arena."""
        with pytest.raises(ValueError):
            empty_grid.world_to_grid(cfg.WORLD_ORIGIN_X - 1.0, 0.0)

    def test_out_of_bounds_grid_raises(self, empty_grid):
        """grid_to_world raises ValueError for cells outside the grid."""
        with pytest.raises(ValueError):
            empty_grid.grid_to_world(-1, 0)
        with pytest.raises(ValueError):
            empty_grid.grid_to_world(0, empty_grid.cols + 1)


# ------------------------------------------------------------------ #
#  Grid dimensions                                                     #
# ------------------------------------------------------------------ #

class TestGridDimensions:
    def test_dimensions(self, empty_grid, cfg):
        expected_rows = math.ceil(cfg.WORLD_HEIGHT / cfg.GRID_RESOLUTION)
        expected_cols = math.ceil(cfg.WORLD_WIDTH  / cfg.GRID_RESOLUTION)
        assert empty_grid.rows == expected_rows
        assert empty_grid.cols == expected_cols

    def test_is_in_bounds(self, empty_grid):
        assert empty_grid.is_in_bounds(0, 0)
        assert empty_grid.is_in_bounds(empty_grid.rows - 1, empty_grid.cols - 1)
        assert not empty_grid.is_in_bounds(-1, 0)
        assert not empty_grid.is_in_bounds(0, empty_grid.cols)


# ------------------------------------------------------------------ #
#  Cell state                                                          #
# ------------------------------------------------------------------ #

class TestCellState:
    def test_initial_state_unknown(self, empty_grid, cfg):
        """Freshly created grid has all cells UNKNOWN."""
        assert empty_grid.get_cell(0, 0) == cfg.CELL_UNKNOWN
        assert empty_grid.get_cell(
            empty_grid.rows // 2, empty_grid.cols // 2
        ) == cfg.CELL_UNKNOWN

    def test_set_and_get_cell(self, empty_grid, cfg):
        empty_grid.set_cell(5, 5, cfg.CELL_OCCUPIED)
        assert empty_grid.get_cell(5, 5) == cfg.CELL_OCCUPIED
        assert empty_grid.is_occupied(5, 5)
        assert not empty_grid.is_free(5, 5)

    def test_out_of_bounds_get_returns_unknown(self, empty_grid, cfg):
        """get_cell on OOB coordinates returns UNKNOWN without raising."""
        val = empty_grid.get_cell(-1, -1)
        assert val == cfg.CELL_UNKNOWN


# ------------------------------------------------------------------ #
#  Obstacle marking                                                    #
# ------------------------------------------------------------------ #

class TestObstacleMarking:
    def test_obstacles_marked_occupied(self, built_grid, cfg):
        """Each obstacle centre should map to an OCCUPIED cell."""
        for obs in cfg.OBSTACLES:
            try:
                row, col = built_grid.world_to_grid(obs["x"], obs["y"])
                assert built_grid.is_occupied(row, col), (
                    f"Obstacle at ({obs['x']}, {obs['y']}) not marked occupied"
                )
            except ValueError:
                pytest.fail(f"Obstacle at ({obs['x']}, {obs['y']}) is outside the grid")

    def test_free_cells_exist(self, built_grid, cfg):
        """The start position should be in a free (or inflated) area."""
        try:
            row, col = built_grid.world_to_grid(*cfg.START_POS)
            # After inflation the start cell should be free
            assert built_grid.is_free(row, col), \
                "Start position is occupied — check obstacle placement or inflation"
        except ValueError:
            pytest.fail("START_POS is outside the grid")

    def test_goal_cell_free(self, built_grid, cfg):
        """Goal position should be reachable (free cell)."""
        row, col = built_grid.world_to_grid(*cfg.GOAL_POS)
        assert built_grid.is_free(row, col), \
            "Goal position is occupied — pathfinding will fail"

    def test_majority_cells_free(self, built_grid, cfg):
        """More than 50% of cells should be free (arena is not overcrowded)."""
        arr = built_grid.get_array()
        free_count = (arr == cfg.CELL_FREE).sum()
        total = arr.size
        assert free_count / total > 0.5, \
            f"Only {free_count/total:.1%} of cells are free — grid may be over-inflated"
