"""
robot_model.py
==============
Differential-drive kinematic model.

For a robot with:
  - wheel radius r (metres)
  - wheel separation L (metres)
  - right-wheel angular velocity ω_R (rad/s)
  - left-wheel angular velocity  ω_L (rad/s)

Linear velocity  : v = r(ω_R + ω_L) / 2
Angular velocity : ω = r(ω_R - ω_L) / L

Pose update (forward Euler, time step dt seconds):
  x(t+dt) = x(t) + v·cos(θ) · dt
  y(t+dt) = y(t) + v·sin(θ) · dt
  θ(t+dt) = θ(t) + ω · dt

Heading θ is normalised to (−π, π] after every update.

Note on Webots integration
--------------------------
When this model runs inside the Webots controller, (x, y, θ) are updated
using the actual wheel velocities set on the motor devices.  The model
provides a dead-reckoning estimate; ground-truth position is available via
the Webots GPS device and is used for evaluation only.
"""

from __future__ import annotations

import math


def normalise_angle(angle: float) -> float:
    """Wrap angle to the interval (−π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


class DifferentialDriveRobot:
    """
    Kinematic state and update for a differential-drive robot.

    Attributes
    ----------
    x, y  : position in world frame (metres)
    theta : heading in world frame (radians, 0 = +X axis)
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float,
        wheel_separation: float,
        max_wheel_speed: float,
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.theta: float = normalise_angle(theta)

        self._r: float = wheel_radius
        self._L: float = wheel_separation
        self._max_speed: float = max_wheel_speed

        # Track commanded speeds for logging
        self.v: float = 0.0
        self.omega: float = 0.0

    # ------------------------------------------------------------------ #
    #  Kinematics                                                          #
    # ------------------------------------------------------------------ #

    def set_wheel_speeds(self, omega_right: float, omega_left: float) -> None:
        """
        Record the commanded wheel angular velocities (rad/s).

        Clips to ±max_wheel_speed before storing.
        """
        omega_right = max(-self._max_speed, min(self._max_speed, omega_right))
        omega_left  = max(-self._max_speed, min(self._max_speed, omega_left))
        self.v     = self._r * (omega_right + omega_left) / 2.0
        self.omega = self._r * (omega_right - omega_left) / self._L

    def update(self, dt: float) -> None:
        """
        Advance the pose by one time step dt (seconds) using Forward Euler.

        Call after set_wheel_speeds().
        """
        self.x     += self.v * math.cos(self.theta) * dt
        self.y     += self.v * math.sin(self.theta) * dt
        self.theta  = normalise_angle(self.theta + self.omega * dt)

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def wheel_speeds_from_v_omega(
        self, v: float, omega: float
    ) -> tuple[float, float]:
        """
        Convert desired (v, ω) to (ω_R, ω_L) in rad/s.

        Clips the resulting speeds to ±max_wheel_speed.
        """
        omega_right = (v + omega * self._L / 2.0) / self._r
        omega_left  = (v - omega * self._L / 2.0) / self._r
        omega_right = max(-self._max_speed, min(self._max_speed, omega_right))
        omega_left  = max(-self._max_speed, min(self._max_speed, omega_left))
        return omega_right, omega_left

    @property
    def pose(self) -> tuple[float, float, float]:
        """Return (x, y, theta)."""
        return self.x, self.y, self.theta

    def distance_to(self, wx: float, wy: float) -> float:
        """Euclidean distance from current position to world point (wx, wy)."""
        return math.hypot(wx - self.x, wy - self.y)

    def heading_to(self, wx: float, wy: float) -> float:
        """
        Desired heading angle (radians) from current position toward (wx, wy).
        """
        return math.atan2(wy - self.y, wx - self.x)

    def heading_error_to(self, wx: float, wy: float) -> float:
        """
        Signed heading error (radians) to reach (wx, wy).
        Positive → turn left (counter-clockwise), negative → turn right.
        """
        desired = self.heading_to(wx, wy)
        return normalise_angle(desired - self.theta)
