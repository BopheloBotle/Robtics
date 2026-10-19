"""
sensors.py
==========
Sensor data structures and synthetic sensor generators for offline testing.

The classes here mirror the data produced by real Webots sensor devices.
In the Webots controller, sensor readings are obtained from the Webots API;
these synthetic generators are used only in offline algorithm testing.

LiDAR model
-----------
Simulates a 360° LiDAR with Gaussian noise.  The LiDAR is placed at the
robot's position and fires rays in equal angular steps.  A ray that hits an
obstacle within LIDAR_RANGE returns the distance to the hit; otherwise it
returns LIDAR_RANGE (no-obstacle reading).

Camera model
------------
Simulates a monochrome forward-looking camera.  Returns a NumPy uint8 array
of shape (height, width).  Obstacles in front of the robot are drawn as dark
rectangular regions.

Both generators accept a random seed for reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from configuration import Config


@dataclass
class LidarScan:
    """Container for a single LiDAR scan."""
    ranges: list[float]          # metres, one per angular step
    robot_x: float               # robot position when scan was taken
    robot_y: float
    robot_heading: float         # radians
    timestamp: float = 0.0       # simulation time (seconds)


@dataclass
class CameraFrame:
    """Container for a single camera frame."""
    image: np.ndarray            # shape (height, width) uint8
    robot_x: float
    robot_y: float
    robot_heading: float
    timestamp: float = 0.0


class SyntheticLidar:
    """
    Synthetic LiDAR generator for offline testing.

    Obstacles are modelled as axis-aligned rectangles (same format as
    Config.OBSTACLES).  Each ray is traced against all obstacle rectangles
    and returns the nearest hit distance.
    """

    def __init__(self, cfg: "Config", seed: int | None = None) -> None:
        self._cfg = cfg
        self._rng = random.Random(seed if seed is not None else cfg.RANDOM_SEED)
        self._noise_std = 0.02  # metres standard deviation

    def scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        timestamp: float = 0.0,
    ) -> LidarScan:
        """Generate a synthetic LiDAR scan from the given robot pose."""
        cfg = self._cfg
        n = cfg.LIDAR_SAMPLES
        angular_step = 2.0 * math.pi / n
        ranges: list[float] = []

        for i in range(n):
            angle = robot_heading + i * angular_step
            dist = self._ray_cast(robot_x, robot_y, angle)
            # Add Gaussian noise
            noise = self._rng.gauss(0.0, self._noise_std)
            dist = max(cfg.LIDAR_MIN_RANGE, min(cfg.LIDAR_RANGE, dist + noise))
            ranges.append(dist)

        return LidarScan(
            ranges=ranges,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_heading=robot_heading,
            timestamp=timestamp,
        )

    def _ray_cast(
        self, ox: float, oy: float, angle: float
    ) -> float:
        """Cast a ray from (ox, oy) at angle and return distance to nearest obstacle."""
        cfg = self._cfg
        best = cfg.LIDAR_RANGE
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Check each obstacle rectangle
        for obs in cfg.OBSTACLES:
            cx, cy = obs["x"], obs["y"]
            hw, hh = obs["w"] / 2.0, obs["h"] / 2.0
            x_min, x_max = cx - hw, cx + hw
            y_min, y_max = cy - hh, cy + hh
            t = _ray_aabb_intersect(ox, oy, dx, dy, x_min, x_max, y_min, y_max)
            if t is not None and cfg.LIDAR_MIN_RANGE < t < best:
                best = t

        # Check arena walls
        wall = cfg.WORLD_WIDTH / 2.0
        for t in [
            _ray_plane_intersect(ox, oy, dx, dy, "x", -wall),
            _ray_plane_intersect(ox, oy, dx, dy, "x",  wall),
            _ray_plane_intersect(ox, oy, dx, dy, "y", -wall),
            _ray_plane_intersect(ox, oy, dx, dy, "y",  wall),
        ]:
            if t is not None and cfg.LIDAR_MIN_RANGE < t < best:
                best = t

        return best


class SyntheticCamera:
    """
    Synthetic grayscale camera generator for offline testing.

    Obstacles in the camera's forward field of view are drawn as dark
    rectangular regions.  The camera image is used only for basic obstacle
    detection (dark-region segmentation); no advanced computer vision is
    claimed.
    """

    def __init__(self, cfg: "Config", seed: int | None = None) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(
            seed if seed is not None else cfg.RANDOM_SEED
        )

    def capture(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        timestamp: float = 0.0,
    ) -> CameraFrame:
        """Generate a synthetic camera frame."""
        cfg = self._cfg
        h, w = cfg.CAMERA_HEIGHT, cfg.CAMERA_WIDTH
        # Start with a bright background (floor)
        image = np.full((h, w), 200, dtype=np.uint8)
        # Add per-pixel Gaussian noise
        noise = self._rng.integers(-15, 15, size=(h, w), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        fov = cfg.CAMERA_FOV
        half_fov = fov / 2.0
        max_range = cfg.LIDAR_RANGE

        for obs in cfg.OBSTACLES:
            cx, cy = obs["x"], obs["y"]
            # Vector from robot to obstacle centre
            rel_x = cx - robot_x
            rel_y = cy - robot_y
            dist = math.hypot(rel_x, rel_y)
            if dist > max_range or dist < 0.1:
                continue
            # Angle to obstacle in robot frame
            angle_to_obs = math.atan2(rel_y, rel_x) - robot_heading
            # Normalise to (-π, π)
            while angle_to_obs > math.pi:
                angle_to_obs -= 2.0 * math.pi
            while angle_to_obs < -math.pi:
                angle_to_obs += 2.0 * math.pi
            if abs(angle_to_obs) > half_fov:
                continue
            # Map angle to pixel column
            col_centre = int((angle_to_obs + half_fov) / fov * w)
            # Apparent size (closer → larger)
            apparent_w = max(2, int(w * obs["w"] / (dist * fov * 2)))
            apparent_h = max(2, int(h * obs["h"] / (dist * 3.0)))
            row_centre = h // 2
            c1 = max(0, col_centre - apparent_w // 2)
            c2 = min(w, col_centre + apparent_w // 2)
            r1 = max(0, row_centre - apparent_h // 2)
            r2 = min(h, row_centre + apparent_h // 2)
            # Draw obstacle as dark region (intensity proportional to distance)
            intensity = max(20, int(80 - dist * 10))
            image[r1:r2, c1:c2] = intensity

        return CameraFrame(
            image=image,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_heading=robot_heading,
            timestamp=timestamp,
        )


# ------------------------------------------------------------------ #
#  Ray-geometry helpers                                                #
# ------------------------------------------------------------------ #

def _ray_aabb_intersect(
    ox: float, oy: float, dx: float, dy: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
) -> float | None:
    """
    Slab method: intersect ray (o + t·d) with axis-aligned bounding box.
    Returns the entry t if the ray hits, else None.
    """
    t_min, t_max = 0.0, float("inf")

    if abs(dx) < 1e-12:
        if not (x_min <= ox <= x_max):
            return None
    else:
        tx1 = (x_min - ox) / dx
        tx2 = (x_max - ox) / dx
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))

    if abs(dy) < 1e-12:
        if not (y_min <= oy <= y_max):
            return None
    else:
        ty1 = (y_min - oy) / dy
        ty2 = (y_max - oy) / dy
        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))

    if t_max < t_min or t_max < 0:
        return None
    return t_min if t_min >= 0 else t_max


def _ray_plane_intersect(
    ox: float, oy: float, dx: float, dy: float,
    axis: str, value: float,
) -> float | None:
    """Intersect ray with an axis-aligned infinite plane."""
    if axis == "x":
        if abs(dx) < 1e-12:
            return None
        t = (value - ox) / dx
    else:
        if abs(dy) < 1e-12:
            return None
        t = (value - oy) / dy
    return t if t > 0 else None
