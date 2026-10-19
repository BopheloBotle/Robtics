"""
occupancy_grid.py
=================
Binary occupancy grid supporting free / occupied / unknown cells.

Coordinate convention
---------------------
World frame : origin at (WORLD_ORIGIN_X, WORLD_ORIGIN_Y), X right, Y up.
Grid frame  : row 0 → maximum world-Y (top of map),
              col 0 → minimum world-X (left of map).
This matches the standard image convention so that Matplotlib imshow
displays the map in the expected orientation.

  grid_row = (WORLD_HEIGHT / resolution - 1) - floor((wy - origin_y) / res)
  grid_col = floor((wx - origin_x) / res)

All public methods perform boundary checking and raise ValueError for
out-of-range coordinates.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configuration import Config


class OccupancyGrid:
    """2-D occupancy grid with world ↔ grid coordinate conversion."""

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._res: float = cfg.GRID_RESOLUTION
        self._ox: float = cfg.WORLD_ORIGIN_X
        self._oy: float = cfg.WORLD_ORIGIN_Y
        self._ww: float = cfg.WORLD_WIDTH
        self._wh: float = cfg.WORLD_HEIGHT

        self.cols: int = math.ceil(self._ww / self._res)
        self.rows: int = math.ceil(self._wh / self._res)

        # Initialise all cells as unknown
        self._grid: np.ndarray = np.full(
            (self.rows, self.cols), cfg.CELL_UNKNOWN, dtype=np.int8
        )

    # ------------------------------------------------------------------ #
    #  Coordinate conversion                                               #
    # ------------------------------------------------------------------ #

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world (x, y) to (row, col). Raises ValueError if OOB."""
        col = int(math.floor((wx - self._ox) / self._res))
        # row 0 is at the top (max y), so we invert the y-axis
        row = int(math.floor((self._oy + self._wh - wy) / self._res))
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"World position ({wx:.2f}, {wy:.2f}) maps to ({row},{col}) "
                f"which is outside grid ({self.rows}×{self.cols})"
            )
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert (row, col) to world (x, y) at cell centre. Raises ValueError if OOB."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"Grid cell ({row},{col}) is outside grid ({self.rows}×{self.cols})"
            )
        wx = self._ox + (col + 0.5) * self._res
        # row=0 → top of map (max world-Y), so invert:
        wy = self._oy + self._wh - (row + 0.5) * self._res
        return wx, wy

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is inside the grid."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------ #
    #  Cell access                                                         #
    # ------------------------------------------------------------------ #

    def set_cell(self, row: int, col: int, value: int) -> None:
        if self.is_in_bounds(row, col):
            self._grid[row, col] = value

    def get_cell(self, row: int, col: int) -> int:
        if not self.is_in_bounds(row, col):
            return self._cfg.CELL_UNKNOWN
        return int(self._grid[row, col])

    def is_free(self, row: int, col: int) -> bool:
        return self.get_cell(row, col) == self._cfg.CELL_FREE

    def is_occupied(self, row: int, col: int) -> bool:
        return self.get_cell(row, col) == self._cfg.CELL_OCCUPIED

    def get_array(self) -> np.ndarray:
        """Return a copy of the internal grid array."""
        return self._grid.copy()

    # ------------------------------------------------------------------ #
    #  Map building                                                        #
    # ------------------------------------------------------------------ #

    def build_from_config(self) -> None:
        """
        Build the grid from the obstacle list in Config.

        All non-obstacle cells within the arena boundary are set to FREE.
        Obstacle cells are set to OCCUPIED.
        Obstacle inflation is then applied.
        """
        cfg = self._cfg

        # Mark everything inside the arena as free first
        for r in range(self.rows):
            for c in range(self.cols):
                self._grid[r, c] = cfg.CELL_FREE

        # Mark obstacle footprints
        for obs in cfg.OBSTACLES:
            cx, cy = obs["x"], obs["y"]
            hw = obs["w"] / 2.0
            hh = obs["h"] / 2.0
            x_min, x_max = cx - hw, cx + hw
            y_min, y_max = cy - hh, cy + hh
            self._mark_rectangle_occupied(x_min, x_max, y_min, y_max)

        # Inflate obstacles for collision clearance
        self._inflate_obstacles(cfg.OBSTACLE_INFLATION)

    def _mark_rectangle_occupied(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        """Mark all cells whose centres fall inside the rectangle as OCCUPIED."""
        cfg = self._cfg
        for r in range(self.rows):
            for c in range(self.cols):
                wx, wy = self.grid_to_world(r, c)
                if x_min <= wx <= x_max and y_min <= wy <= y_max:
                    self._grid[r, c] = cfg.CELL_OCCUPIED

    def _inflate_obstacles(self, n_cells: int) -> None:
        """
        Morphological dilation: any FREE cell within n_cells of an OCCUPIED
        cell is also marked OCCUPIED.  Prevents the planned path from grazing
        obstacle edges.
        """
        cfg = self._cfg
        original = self._grid.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                if original[r, c] == cfg.CELL_OCCUPIED:
                    for dr in range(-n_cells, n_cells + 1):
                        for dc in range(-n_cells, n_cells + 1):
                            nr, nc = r + dr, c + dc
                            if self.is_in_bounds(nr, nc):
                                self._grid[nr, nc] = cfg.CELL_OCCUPIED

    # ------------------------------------------------------------------ #
    #  LiDAR update                                                        #
    # ------------------------------------------------------------------ #

    def update_from_lidar(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        ranges: list[float],
    ) -> None:
        """
        Update the grid using a LiDAR scan.

        Parameters
        ----------
        robot_x, robot_y : robot position in world frame
        robot_heading     : robot heading in radians (measured from +X axis)
        ranges            : list of range readings, one per angular step.
                            The angular step is 2π / len(ranges).
                            Readings ≤ LIDAR_MIN_RANGE or > LIDAR_RANGE
                            are treated as invalid and ignored.
        """
        cfg = self._cfg
        n = len(ranges)
        if n == 0:
            return
        angular_step = 2.0 * math.pi / n

        for i, r in enumerate(ranges):
            # Skip invalid readings
            if r < cfg.LIDAR_MIN_RANGE or r > cfg.LIDAR_RANGE:
                continue
            angle = robot_heading + i * angular_step
            hit_x = robot_x + r * math.cos(angle)
            hit_y = robot_y + r * math.sin(angle)
            try:
                row, col = self.world_to_grid(hit_x, hit_y)
                if r < cfg.LIDAR_OBSTACLE_THRESHOLD + cfg.LIDAR_MIN_RANGE:
                    self._grid[row, col] = cfg.CELL_OCCUPIED
                else:
                    # Mark the hit cell as occupied only if within obstacle threshold
                    if r <= cfg.LIDAR_RANGE:
                        self._grid[row, col] = cfg.CELL_OCCUPIED
                # Cells along the beam between robot and hit are free
                self._ray_trace_free(robot_x, robot_y, hit_x, hit_y)
            except ValueError:
                pass

    def _ray_trace_free(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        """Mark cells along the ray from (x0,y0) to (x1,y1) as FREE."""
        cfg = self._cfg
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (self._res * 0.5)))
        for k in range(steps):
            t = k / steps
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            try:
                row, col = self.world_to_grid(wx, wy)
                if self._grid[row, col] != cfg.CELL_OCCUPIED:
                    self._grid[row, col] = cfg.CELL_FREE
            except ValueError:
                pass

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        """Save the grid array as a numpy .npy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self._grid)

    def load(self, path: Path) -> None:
        """Load a previously saved grid array."""
        self._grid = np.load(str(path))
        self.rows, self.cols = self._grid.shape
