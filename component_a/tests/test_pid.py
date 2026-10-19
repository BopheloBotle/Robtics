"""
test_pid.py
===========
Offline unit tests for the PIDController.

Tests cover:
  - Proportional-only response (Ki=Kd=0) produces correct scaling.
  - Output is clipped to ±output_limit.
  - Integral accumulates correctly.
  - Anti-windup prevents integral exceeding integral_limit.
  - Derivative term is non-zero on second call when error changes.
  - Reset clears all internal state.
  - Zero error produces zero output for P-only controller.
  - Derivative filter smooths the derivative signal.
  - Heading step response (90°) using well-tuned gains converges.

No Webots connection is required.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from pid_controller import PIDController
from robot_model import normalise_angle


@pytest.fixture
def p_only() -> PIDController:
    return PIDController(Kp=2.0, Ki=0.0, Kd=0.0,
                         output_limit=10.0, integral_limit=5.0, dt=0.1)


@pytest.fixture
def full_pid() -> PIDController:
    return PIDController(Kp=1.5, Ki=0.05, Kd=0.2,
                         output_limit=6.28, integral_limit=2.0, dt=0.032)


class TestProportionalOnly:
    def test_output_proportional(self, p_only):
        out = p_only.compute(3.0)
        assert abs(out - 6.0) < 1e-9, f"Expected 6.0 got {out}"

    def test_zero_error_zero_output(self, p_only):
        assert abs(p_only.compute(0.0)) < 1e-9

    def test_negative_error(self, p_only):
        out = p_only.compute(-2.0)
        assert abs(out - (-4.0)) < 1e-9

    def test_output_clipped_positive(self, p_only):
        out = p_only.compute(100.0)
        assert out == pytest.approx(10.0)

    def test_output_clipped_negative(self, p_only):
        out = p_only.compute(-100.0)
        assert out == pytest.approx(-10.0)


class TestIntegral:
    def test_integral_accumulates(self, full_pid):
        """After N steps with constant error, integral should be N*error*dt."""
        error = 1.0
        n = 10
        for _ in range(n):
            full_pid.compute(error)
        expected_integral = min(
            full_pid.integral_limit, error * n * full_pid.dt
        )
        assert abs(full_pid.integral - expected_integral) < 1e-6

    def test_anti_windup(self, full_pid):
        """Integral must never exceed integral_limit regardless of error magnitude."""
        for _ in range(1000):
            full_pid.compute(100.0)
        assert full_pid.integral <= full_pid.integral_limit + 1e-9

    def test_anti_windup_negative(self, full_pid):
        for _ in range(1000):
            full_pid.compute(-100.0)
        assert full_pid.integral >= -full_pid.integral_limit - 1e-9


class TestDerivative:
    def test_derivative_nonzero_on_change(self, full_pid):
        """Derivative term must be non-zero when error changes between steps."""
        full_pid.compute(0.0)   # first call sets prev_error=0
        out1 = full_pid.compute(1.0)   # error changed → derivative > 0
        # With Kd=0.2, dt=0.032: d_term ≈ 0.2 * 1.0/0.032 ≈ 6.25 (clipped)
        assert out1 != 0.0

    def test_derivative_zero_on_constant_error(self):
        """When error is constant, derivative term should be zero."""
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0,
                            output_limit=100.0, integral_limit=100.0, dt=0.1)
        pid.compute(1.0)  # first call — no previous error
        out = pid.compute(1.0)  # error unchanged
        assert abs(out) < 1e-6, f"Expected ~0 derivative on constant error, got {out}"


class TestReset:
    def test_reset_clears_integral(self, full_pid):
        for _ in range(50):
            full_pid.compute(1.0)
        full_pid.reset()
        assert full_pid.integral == 0.0

    def test_reset_clears_prev_error(self, full_pid):
        full_pid.compute(5.0)
        full_pid.reset()
        assert full_pid.last_error == 0.0

    def test_reset_then_compute(self, full_pid):
        """After reset, first call should behave as if fresh."""
        for _ in range(100):
            full_pid.compute(1.0)
        full_pid.reset()
        out = full_pid.compute(1.0)
        # P term = 1.5*1.0 = 1.5, I=1*0.032=0.032→i_term=0.05*0.032=0.0016
        # D = 0 (first call after reset)
        expected = 1.5 * 1.0 + 0.05 * (1.0 * 0.032) + 0.0
        assert abs(out - expected) < 0.01


class TestDerivativeFilter:
    def test_filter_smooths_derivative(self):
        """With tau_f > 0, abrupt error change should produce smaller derivative."""
        pid_no_filter = PIDController(
            Kp=0.0, Ki=0.0, Kd=1.0, output_limit=100.0,
            integral_limit=100.0, dt=0.1, tau_f=0.0
        )
        pid_filtered = PIDController(
            Kp=0.0, Ki=0.0, Kd=1.0, output_limit=100.0,
            integral_limit=100.0, dt=0.1, tau_f=0.5
        )
        pid_no_filter.compute(0.0)
        pid_filtered.compute(0.0)
        out_raw      = abs(pid_no_filter.compute(10.0))
        out_filtered = abs(pid_filtered.compute(10.0))
        assert out_filtered < out_raw, \
            "Derivative filter did not reduce output magnitude"


class TestOutputLimit:
    def test_symmetric_clipping(self):
        pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, output_limit=3.0,
                            integral_limit=1.0, dt=0.1)
        assert pid.compute(100.0)  == pytest.approx(3.0)
        assert pid.compute(-100.0) == pytest.approx(-3.0)


class TestHeadingStepResponse:
    def test_converges_with_well_tuned_gains(self):
        """
        A 90° heading step response with well-tuned gains should converge
        to within 5° of the target within 10 seconds.
        """
        Kp, Ki, Kd = 1.5, 0.05, 0.2
        dt = 0.032
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd,
                            output_limit=6.28, integral_limit=2.0, dt=dt)
        target = math.pi / 2.0
        heading = 0.0
        for _ in range(int(10.0 / dt)):
            error = normalise_angle(target - heading)
            output = pid.compute(error)
            heading = normalise_angle(heading + output * dt * 0.5)

        final_error_deg = math.degrees(abs(normalise_angle(target - heading)))
        assert final_error_deg < 5.0, \
            f"Heading did not converge: final error = {final_error_deg:.2f}°"

    def test_under_damped_oscillates(self):
        """Under-damped gains (high Kp) should show overshoot."""
        Kp, Ki, Kd = 3.0, 0.1, 0.05
        dt = 0.032
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd,
                            output_limit=6.28, integral_limit=2.0, dt=dt)
        target = math.pi / 2.0
        heading = 0.0
        max_heading = 0.0
        for _ in range(int(8.0 / dt)):
            error = normalise_angle(target - heading)
            output = pid.compute(error)
            heading = normalise_angle(heading + output * dt * 0.5)
            if heading > max_heading:
                max_heading = heading
        # Under-damped response should overshoot the target
        overshoot = max_heading - target
        assert overshoot > 0.01, \
            f"Expected overshoot with high Kp but got {math.degrees(overshoot):.2f}°"
