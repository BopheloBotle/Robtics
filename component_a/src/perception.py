"""
perception.py
=============
Perception pipeline for Component A.

Processes raw sensor data (LiDAR scans and camera frames) into
obstacle observations that can be fed to the occupancy grid.

LiDAR pipeline
--------------
1. Filter invalid readings (< LIDAR_MIN_RANGE or > LIDAR_RANGE).
2. Convert polar range readings to Cartesian obstacle coordinates in the
   robot frame.
3. Transform to the world frame using the robot pose.
4. Return a list of (world_x, world_y) obstacle hit points.

Camera pipeline
---------------
1. Convert to grayscale if needed.
2. Apply Gaussian blur to reduce noise.
3. Threshold to segment dark (obstacle) regions.
4. Find contours (OpenCV) or use a fallback connected-component approach.
5. Estimate the bearing angle to each detected blob.
6. Return a list of (bearing_rad, pixel_width) observations.

A combined obstacle map is obtained by fusing LiDAR hits with camera
observations in sensor_fusion.py.

Fallback strategy
-----------------
If OpenCV is unavailable, the camera pipeline uses NumPy thresholding only
and does not perform contour detection.  This fallback is clearly labelled in
the output.

Coordinate convention
---------------------
Robot frame: X forward (heading direction), Y left.
Bearing angle: measured from the robot's forward axis, positive counter-clockwise.
World frame: X right (East), Y up (North).  See occupancy_grid.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from sensors import LidarScan, CameraFrame

if TYPE_CHECKING:
    from configuration import Config


@dataclass
class LidarObservation:
    """World-frame obstacle hit points from one LiDAR scan."""
    hits_world: list[tuple[float, float]] = field(default_factory=list)
    valid_count: int = 0
    invalid_count: int = 0


@dataclass
class CameraObservation:
    """Bearing estimates to detected obstacle blobs from one camera frame."""
    blobs: list[dict] = field(default_factory=list)  # {bearing_rad, pixel_width, confidence}
    method: str = "opencv"   # "opencv" or "numpy_fallback"


class PerceptionPipeline:
    """
    Processes LiDAR scans and camera frames into structured observations.
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._cv2_available = _CV2_AVAILABLE
        if not self._cv2_available:
            print("[Perception] OpenCV unavailable — using NumPy fallback for camera.")

    # ------------------------------------------------------------------ #
    #  LiDAR processing                                                   #
    # ------------------------------------------------------------------ #

    def process_lidar(self, scan: LidarScan) -> LidarObservation:
        """
        Convert a LiDAR scan into world-frame obstacle hit points.

        Invalid readings (< LIDAR_MIN_RANGE or > LIDAR_RANGE) are discarded.
        Valid readings at exactly LIDAR_RANGE are treated as free-space
        endpoints (no obstacle at that range) and not marked as hits.
        Readings inside the obstacle threshold produce hit points.
        """
        cfg = self._cfg
        n = len(scan.ranges)
        if n == 0:
            return LidarObservation()

        angular_step = 2.0 * math.pi / n
        hits: list[tuple[float, float]] = []
        valid = 0
        invalid = 0

        cos_h = math.cos(scan.robot_heading)
        sin_h = math.sin(scan.robot_heading)

        for i, r in enumerate(scan.ranges):
            if r < cfg.LIDAR_MIN_RANGE or r >= cfg.LIDAR_RANGE - 0.01:
                invalid += 1
                continue
            valid += 1
            # Angle in world frame
            angle_world = scan.robot_heading + i * angular_step
            hx = scan.robot_x + r * math.cos(angle_world)
            hy = scan.robot_y + r * math.sin(angle_world)
            hits.append((hx, hy))

        return LidarObservation(
            hits_world=hits,
            valid_count=valid,
            invalid_count=invalid,
        )

    # ------------------------------------------------------------------ #
    #  Camera processing                                                  #
    # ------------------------------------------------------------------ #

    def process_camera(self, frame: CameraFrame) -> CameraObservation:
        """
        Detect obstacle blobs in a camera frame and return bearing estimates.

        Uses OpenCV if available, otherwise NumPy thresholding fallback.
        """
        if self._cv2_available:
            return self._process_camera_opencv(frame)
        return self._process_camera_numpy(frame)

    def _process_camera_opencv(self, frame: CameraFrame) -> CameraObservation:
        """OpenCV-based blob detection pipeline."""
        cfg = self._cfg
        img = frame.image

        # Ensure grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Gaussian blur to suppress noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Threshold: dark regions (obstacles) → white mask
        _, mask = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        w = cfg.CAMERA_WIDTH
        fov = cfg.CAMERA_FOV

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            col_centre = x + bw / 2.0
            # Convert pixel column to bearing angle
            bearing = ((col_centre / w) - 0.5) * fov
            confidence = min(1.0, area / (w * cfg.CAMERA_HEIGHT * 0.1))
            blobs.append({
                "bearing_rad": bearing,
                "pixel_width": bw,
                "confidence": confidence,
            })

        return CameraObservation(blobs=blobs, method="opencv")

    def _process_camera_numpy(self, frame: CameraFrame) -> CameraObservation:
        """NumPy fallback: simple column-sum thresholding."""
        cfg = self._cfg
        img = frame.image
        if img.ndim == 3:
            gray = img.mean(axis=2).astype(np.uint8)
        else:
            gray = img

        # Threshold
        dark_mask = gray < 100  # True where obstacle pixel

        # Column projection
        col_sum = dark_mask.sum(axis=0)
        threshold = cfg.CAMERA_HEIGHT * 0.2

        blobs = []
        w = cfg.CAMERA_WIDTH
        fov = cfg.CAMERA_FOV
        in_blob = False
        blob_start = 0

        for c in range(w):
            if col_sum[c] >= threshold and not in_blob:
                in_blob = True
                blob_start = c
            elif col_sum[c] < threshold and in_blob:
                in_blob = False
                col_centre = (blob_start + c) / 2.0
                blob_width = c - blob_start
                bearing = ((col_centre / w) - 0.5) * fov
                blobs.append({
                    "bearing_rad": bearing,
                    "pixel_width": blob_width,
                    "confidence": 0.5,
                })
        if in_blob:
            col_centre = (blob_start + w) / 2.0
            blob_width = w - blob_start
            bearing = ((col_centre / w) - 0.5) * fov
            blobs.append({
                "bearing_rad": bearing,
                "pixel_width": blob_width,
                "confidence": 0.5,
            })

        return CameraObservation(blobs=blobs, method="numpy_fallback")
