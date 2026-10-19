"""
pid_controller.py
=================
Discrete-time PID controller.

Implements:
  - Proportional, integral, and derivative terms.
  - Anti-windup via integral clamping (integral term is clamped to
    ±integral_limit to prevent the integrator from accumulating large
    errors during periods of saturation).
  - Derivative filtering using a first-order low-pass filter to reduce
    sensitivity to high-frequency noise.
  - Output clamping to ±output_limit.
  - Reset method to clear integrator and derivative state between episodes.
  - Sampling interval dt (seconds) must be consistent with the simulation
    time step or the tuning trial period.

Design note
-----------
For heading control, the error supplied to the controller MUST be normalised
to (−π, π] before each call.  The caller (navigation.py) is responsible for
this normalisation.

The derivative filter time constant τ_f defaults to 0 (no filter, equivalent
to a pure derivative).  Set τ_f > 0 to smooth the derivative signal.  A
reasonable starting value is τ_f = Td / 10 where Td = Kd / Kp.
"""

from __future__ import annotations


class PIDController:
    """Discrete-time PID with anti-windup and derivative filtering."""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        output_limit: float = 10.0,
        integral_limit: float = 2.0,
        dt: float = 0.032,
        tau_f: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        Kp             : Proportional gain
        Ki             : Integral gain
        Kd             : Derivative gain
        output_limit   : Maximum absolute value of controller output
        integral_limit : Maximum absolute integral accumulation (anti-windup)
        dt             : Sampling interval in seconds
        tau_f          : Derivative filter time constant (0 = no filter)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.dt = dt
        self.tau_f = tau_f

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._filtered_deriv: float = 0.0  # derivative filter state
        self._first_call: bool = True

    # ------------------------------------------------------------------ #
    #  Core update                                                         #
    # ------------------------------------------------------------------ #

    def compute(self, error: float) -> float:
        """
        Compute the control output for the given signed error.

        Parameters
        ----------
        error : signed error (e.g. heading error in radians, or position
                error in metres).  For angle errors the caller must
                normalise to (−π, π] before calling.

        Returns
        -------
        output : clipped to ±output_limit.
        """
        # Proportional term
        p_term = self.Kp * error

        # Integral term with anti-windup clamping
        self._integral += error * self.dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        i_term = self.Ki * self._integral

        # Derivative term with optional first-order low-pass filter
        if self._first_call:
            raw_deriv = 0.0
            self._first_call = False
        else:
            raw_deriv = (error - self._prev_error) / self.dt

        if self.tau_f > 0.0:
            alpha = self.dt / (self.tau_f + self.dt)
            self._filtered_deriv = (
                alpha * raw_deriv + (1.0 - alpha) * self._filtered_deriv
            )
        else:
            self._filtered_deriv = raw_deriv

        d_term = self.Kd * self._filtered_deriv
        self._prev_error = error

        output = p_term + i_term + d_term
        return max(-self.output_limit, min(self.output_limit, output))

    # ------------------------------------------------------------------ #
    #  Housekeeping                                                        #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset internal state (call between episodes or after goal reached)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_deriv = 0.0
        self._first_call = True

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def last_error(self) -> float:
        return self._prev_error
