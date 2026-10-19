"""
sensor_fusion.py
================
Sensor fusion: combines LiDAR and camera observations into occupancy-grid updates.

Fusion strategy
---------------
This implementation uses a simple evidence-accumulation strategy rather than
a fully probabilistic Bayesian filter.  Each sensor contributes independently:

  - LiDAR hits are high-confidence (confidence weight 0.9) and mark obstacle
    cells directly.  Ray-traced free-space cells are marked free with
    confidence weight 0.8.
  - Camera blob detections are lower-confidence (0.6 default) because the
    2-D camera cannot determine obstacle range without depth information.
    Camera detections are used to reinforce grid cells that are also hit by
    LiDAR (agreement → confidence boost) or to flag potential obstacles
    beyond the LiDAR scan in the camera's field of view.

Limitations documented
----------------------
- No true probabilistic log-odds update (Bayes filter) is implemented.
- Camera range estimation is approximate (angular size heuristic only).
- Sensor extrinsic calibration is assumed perfect (camera bore-sight aligned
  with robot heading, LiDAR co-located with GPS).
- False positives from specular LiDAR reflections are not modelled.
- Camera observations degrade in low-contrast conditions.
- Partial occlusion by one sensor not modelled by the other.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configuration import Config
    from occupancy_grid import OccupancyGrid
    from perception import LidarObservation, CameraObservation


# Confidence weights
_LIDAR_HIT_WEIGHT = 0.9
_LIDAR_FREE_WEIGHT = 0.8
_CAMERA_WEIGHT = 0.6
_CAMERA_RANGE_ESTIMATE = 2.0   # metres — assumed range for camera-only detections


class SensorFusion:
    """Fuses LiDAR and camera observations into the occupancy grid."""

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        # Optional confidence accumulator (not used by occupancy grid directly
        # but kept for analysis / future Bayesian upgrade)
        # confidence_grid[r][c] in [0, 1]
        import numpy as np
        import math as _math
        rows = _math.ceil(cfg.WORLD_HEIGHT / cfg.GRID_RESOLUTION)
        cols = _math.ceil(cfg.WORLD_WIDTH  / cfg.GRID_RESOLUTION)
        self.confidence: "np.ndarray" = __import__("numpy").zeros((rows, cols), dtype=float)

    def update(
        self,
        grid: "OccupancyGrid",
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        lidar_obs: "LidarObservation",
        camera_obs: "CameraObservation | None" = None,
    ) -> None:
        """
        Update the occupancy grid and confidence accumulator with one set of
        sensor observations.

        Parameters
        ----------
        grid         : OccupancyGrid to update in-place
        robot_x/y    : robot position in world frame
        robot_heading: robot heading in radians
        lidar_obs    : processed LiDAR observation (world-frame hit points)
        camera_obs   : processed camera observation (optional)
        """
        cfg = self._cfg

        # --- LiDAR hits ---
        for hx, hy in lidar_obs.hits_world:
            try:
                row, col = grid.world_to_grid(hx, hy)
                grid.set_cell(row, col, cfg.CELL_OCCUPIED)
                if grid.is_in_bounds(row, col):
                    self.confidence[row, col] = min(
                        1.0, self.confidence[row, col] + _LIDAR_HIT_WEIGHT
                    )
                # Ray-trace free space between robot and hit
                self._mark_ray_free(
                    grid, robot_x, robot_y, hx, hy, _LIDAR_FREE_WEIGHT
                )
            except ValueError:
                pass

        # --- Camera observations ---
        if camera_obs is not None:
            for blob in camera_obs.blobs:
                bearing = blob["bearing_rad"]
                confidence = blob.get("confidence", _CAMERA_WEIGHT)
                # Estimate a candidate obstacle position
                world_angle = robot_heading + bearing
                est_range = _CAMERA_RANGE_ESTIMATE
                ex = robot_x + est_range * math.cos(world_angle)
                ey = robot_y + est_range * math.sin(world_angle)
                try:
                    row, col = grid.world_to_grid(ex, ey)
                    # Only reinforce if already OCCUPIED by LiDAR (agreement)
                    # or mark UNKNOWN region as tentatively occupied
                    current = grid.get_cell(row, col)
                    if current == cfg.CELL_OCCUPIED:
                        if grid.is_in_bounds(row, col):
                            self.confidence[row, col] = min(
                                1.0,
                                self.confidence[row, col]
                                + confidence * _CAMERA_WEIGHT,
                            )
                    elif current == cfg.CELL_UNKNOWN:
                        # Tentative mark — camera only, lower confidence
                        if grid.is_in_bounds(row, col):
                            self.confidence[row, col] += confidence * _CAMERA_WEIGHT * 0.4
                except ValueError:
                    pass

    def _mark_ray_free(
        self,
        grid: "OccupancyGrid",
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        weight: float,
    ) -> None:
        """Mark cells along the ray as FREE with confidence accumulation."""
        cfg = self._cfg
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / (cfg.GRID_RESOLUTION * 0.5)))
        for k in range(steps):
            t = k / steps
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            try:
                row, col = grid.world_to_grid(wx, wy)
                if grid.get_cell(row, col) != cfg.CELL_OCCUPIED:
                    grid.set_cell(row, col, cfg.CELL_FREE)
                    if grid.is_in_bounds(row, col):
                        self.confidence[row, col] = min(
                            1.0, self.confidence[row, col] + weight * 0.2
                        )
            except ValueError:
                pass
