"""
navigation.py
=============
Waypoint-following navigation controller for Component A.

Integrates the kinematic model and PID controller into a closed-loop
navigation system that follows a sequence of waypoints produced by the
path planner.

Controller structure
--------------------
Two PID loops are used:
  1. Heading PID — regulates heading error (desired_heading − current_heading).
     Output is added to the angular-velocity command ω.
  2. Speed PID — regulates distance error to the current waypoint.
     Output is the base linear-velocity command v.

Heading error is always normalised to (−π, π] before being passed to the
heading PID.

Wheel-motor commands are derived from (v, ω) using the differential-drive
inverse kinematics in robot_model.py.

Waypoint switching
------------------
The robot switches to the next waypoint when it is within WAYPOINT_TOLERANCE
metres of the current waypoint.  The final waypoint uses the tighter
GOAL_TOLERANCE.

Safety checks
-------------
- Output wheel speeds are clipped to ±MAX_WHEEL_SPEED.
- A timeout aborts navigation if the goal is not reached within NAV_TIMEOUT.
- A proximity check stops the robot if an imminent collision is detected
  (closest LiDAR reading < LIDAR_MIN_RANGE + 0.05 m).

Logging
-------
At each step the controller appends a dict to self.log containing:
  t, x, y, theta, waypoint_idx, dist_to_wp, heading_error, v, omega,
  omega_R, omega_L.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from robot_model import DifferentialDriveRobot, normalise_angle
from pid_controller import PIDController

if TYPE_CHECKING:
    from configuration import Config


class NavigationController:
    """
    Closed-loop waypoint-following controller.

    Usage
    -----
    nav = NavigationController(robot, cfg)
    nav.set_path(world_waypoints)          # list of (wx, wy)
    omega_R, omega_L = nav.step(lidar_min_range)
    """

    def __init__(self, robot: DifferentialDriveRobot, cfg: "Config") -> None:
        self._robot = robot
        self._cfg = cfg

        # Heading PID (output = angular velocity correction, rad/s)
        self._heading_pid = PIDController(
            Kp=cfg.PID_KP,
            Ki=cfg.PID_KI,
            Kd=cfg.PID_KD,
            output_limit=cfg.MAX_WHEEL_SPEED,
            integral_limit=cfg.PID_INTEGRAL_LIMIT,
            dt=cfg.PID_DT,
        )

        # Speed PID (output = linear velocity, m/s)
        self._speed_pid = PIDController(
            Kp=0.8,
            Ki=0.0,
            Kd=0.05,
            output_limit=cfg.WHEEL_RADIUS * cfg.MAX_WHEEL_SPEED,
            integral_limit=1.0,
            dt=cfg.PID_DT,
        )

        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx: int = 0
        self._start_time: float | None = None
        self.log: list[dict] = []
        self.goal_reached: bool = False
        self.timed_out: bool = False
        self.collision_count: int = 0

    def set_path(self, waypoints: list[tuple[float, float]]) -> None:
        """Set a new sequence of world-frame waypoints and reset state."""
        self._waypoints = list(waypoints)
        self._wp_idx = 0
        self._heading_pid.reset()
        self._speed_pid.reset()
        self._start_time = time.time()
        self.goal_reached = False
        self.timed_out = False
        self.log.clear()

    def step(self, min_lidar_range: float = 999.0) -> tuple[float, float]:
        """
        Compute one control step and return (omega_R, omega_L) in rad/s.

        Parameters
        ----------
        min_lidar_range : minimum current LiDAR reading (metres).  If below
                          the safety threshold the robot stops immediately.
        """
        cfg = self._cfg
        robot = self._robot

        # --- Timeout check ---
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed > cfg.NAV_TIMEOUT:
                self.timed_out = True
                return 0.0, 0.0

        # --- No waypoints ---
        if not self._waypoints or self._wp_idx >= len(self._waypoints):
            self.goal_reached = True
            return 0.0, 0.0

        # --- Collision safety check ---
        if min_lidar_range < cfg.LIDAR_MIN_RANGE + 0.05:
            self.collision_count += 1
            return 0.0, 0.0

        # --- Current waypoint ---
        wx, wy = self._waypoints[self._wp_idx]
        dist = robot.distance_to(wx, wy)
        heading_err = robot.heading_error_to(wx, wy)

        # Determine tolerance for this waypoint
        is_final = (self._wp_idx == len(self._waypoints) - 1)
        tolerance = cfg.GOAL_TOLERANCE if is_final else cfg.WAYPOINT_TOLERANCE

        # --- Waypoint reached? ---
        if dist < tolerance:
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                self.goal_reached = True
                return 0.0, 0.0
            self._heading_pid.reset()
            wx, wy = self._waypoints[self._wp_idx]
            dist = robot.distance_to(wx, wy)
            heading_err = robot.heading_error_to(wx, wy)

        # --- PID computation ---
        omega_correction = self._heading_pid.compute(heading_err)
        v_cmd = self._speed_pid.compute(dist)
        # Reduce speed when heading error is large
        v_cmd *= max(0.1, 1.0 - abs(heading_err) / math.pi)

        omega_cmd = omega_correction * cfg.HEADING_SCALE

        # --- Convert to wheel speeds ---
        omega_R, omega_L = robot.wheel_speeds_from_v_omega(v_cmd, omega_cmd)

        # --- Log ---
        t = time.time() - (self._start_time or time.time())
        self.log.append({
            "t": t,
            "x": robot.x,
            "y": robot.y,
            "theta": robot.theta,
            "waypoint_idx": self._wp_idx,
            "dist_to_wp": dist,
            "heading_error": heading_err,
            "v": v_cmd,
            "omega": omega_cmd,
            "omega_R": omega_R,
            "omega_L": omega_L,
        })

        return omega_R, omega_L

    def is_done(self) -> bool:
        """Return True when the goal is reached or a timeout/failure occurred."""
        return self.goal_reached or self.timed_out

    def path_length_from_log(self) -> float:
        """Compute the actual path length (metres) from the log."""
        if len(self.log) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(self.log)):
            dx = self.log[i]["x"] - self.log[i - 1]["x"]
            dy = self.log[i]["y"] - self.log[i - 1]["y"]
            length += math.hypot(dx, dy)
        return length
