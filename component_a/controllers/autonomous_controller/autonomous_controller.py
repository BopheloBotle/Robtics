"""
autonomous_controller.py
========================
Webots controller for Component A — Autonomous Mobile Robot.

This file is loaded by Webots as the robot controller.  It runs in a
dedicated Python process managed by Webots.

Execution sequence
------------------
1. Initialise Webots robot and all devices (LiDAR, camera, GPS, compass,
   wheel motors, wheel encoders).
2. Build an occupancy grid from the world configuration.
3. Plan a path from start to goal using A* (preferred) with Dijkstra as fallback.
4. Follow the planned path using the PID navigation controller.
5. Log all sensor observations, position errors, and wheel commands to CSV.
6. Save figures and JSON metrics when the simulation ends.

Device names (must match the world file exactly)
-------------------------------------------------
  LiDAR       : "lidar"
  Camera      : "camera"
  GPS         : "gps"
  Compass     : "compass"
  Left motor  : "left_wheel_motor"
  Right motor : "right_wheel_motor"
  Left encoder: "left_wheel_encoder"
  Right encoder: "right_wheel_encoder"

Controller location
-------------------
Webots expects this file at:
  component_a/controllers/autonomous_controller/autonomous_controller.py

The Webots world file must reference the controller as "autonomous_controller".

Running
-------
Open component_a/worlds/component_a_world.wbt in Webots.
The controller starts automatically when the simulation runs.
"""

import sys
import os
import math
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the component_a/src directory to the Python path so that the source
# modules (configuration, sensors, etc.) are importable by this controller.
# ---------------------------------------------------------------------------
_CONTROLLER_DIR = Path(__file__).parent
_COMP_A_SRC = _CONTROLLER_DIR.parent.parent / "src"
sys.path.insert(0, str(_COMP_A_SRC))

# ---------------------------------------------------------------------------
# Import Webots controller library.
# The Webots Python library path is set by Webots automatically when it
# launches the controller process.  If this import fails outside of Webots,
# we exit gracefully with a clear error message.
# ---------------------------------------------------------------------------
try:
    from controller import Robot, Lidar, Camera, GPS, Compass, Motor, PositionSensor
except ImportError:
    print(
        "ERROR: Webots controller library not found.\n"
        "This script must be run from within the Webots simulation environment.\n"
        "Open component_a/worlds/component_a_world.wbt in Webots to run it."
    )
    sys.exit(1)

from configuration import Config
from occupancy_grid import OccupancyGrid
from perception import PerceptionPipeline
from sensor_fusion import SensorFusion
from robot_model import DifferentialDriveRobot, normalise_angle
from astar import astar
from dijkstra import dijkstra
from navigation import NavigationController


def compass_to_heading(compass_vals: list) -> float:
    """Convert Webots compass [north_x, north_y, north_z] to heading in radians."""
    # Webots compass returns a unit vector pointing to magnetic north.
    # For a robot on the XY plane, heading = atan2(-north_x, north_z)
    # (Webots convention: Z axis is forward, X is right, Y is up)
    # For our world: X is East, Y is North, so heading from +X axis:
    return math.atan2(compass_vals[0], compass_vals[2])


def main() -> None:
    cfg = Config()

    # ---- Webots robot initialisation ----
    robot = Robot()
    TIME_STEP = int(robot.getBasicTimeStep())

    # ---- Devices ----
    lidar: Lidar = robot.getDevice("lidar")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

    camera: Camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)

    gps: GPS = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    compass: Compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)

    left_motor: Motor = robot.getDevice("left_wheel_motor")
    right_motor: Motor = robot.getDevice("right_wheel_motor")
    left_motor.setPosition(float("inf"))   # velocity control mode
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    left_enc: PositionSensor = robot.getDevice("left_wheel_encoder")
    right_enc: PositionSensor = robot.getDevice("right_wheel_encoder")
    left_enc.enable(TIME_STEP)
    right_enc.enable(TIME_STEP)

    # ---- First simulation step (warm up sensors) ----
    robot.step(TIME_STEP)

    # ---- Initial pose from GPS/compass ----
    gps_vals = gps.getValues()
    compass_vals = compass.getValues()
    rx = gps_vals[0]
    ry = gps_vals[1]
    rtheta = compass_to_heading(compass_vals)

    print(f"[Controller A] Initial pose: ({rx:.3f}, {ry:.3f}, {math.degrees(rtheta):.1f}°)")

    # ---- Occupancy grid ----
    grid = OccupancyGrid(cfg)
    grid.build_from_config()
    print(f"[Controller A] Grid built: {grid.rows}×{grid.cols}")

    # ---- Path planning ----
    try:
        start_cell = grid.world_to_grid(rx, ry)
        goal_cell  = grid.world_to_grid(*cfg.GOAL_POS)
    except ValueError as e:
        print(f"[Controller A] Start/goal outside grid: {e}")
        sys.exit(1)

    a_result = astar(grid, start_cell, goal_cell, cfg)
    if a_result.success:
        print(f"[Controller A] A* path found: {len(a_result.path)} waypoints, "
              f"{a_result.path_length_m:.2f}m, {a_result.nodes_explored} nodes explored")
        path = a_result.path
    else:
        print("[Controller A] A* failed, trying Dijkstra...")
        d_result = dijkstra(grid, start_cell, goal_cell, cfg)
        if d_result.success:
            path = d_result.path
            print(f"[Controller A] Dijkstra path found: {len(path)} waypoints")
        else:
            print("[Controller A] No path found — stopping.")
            sys.exit(1)

    # Convert grid path to world waypoints
    waypoints = [grid.grid_to_world(r, c) for r, c in path]

    # ---- Kinematic model + navigation controller ----
    robot_model = DifferentialDriveRobot(
        x=rx, y=ry, theta=rtheta,
        wheel_radius=cfg.WHEEL_RADIUS,
        wheel_separation=cfg.WHEEL_SEPARATION,
        max_wheel_speed=cfg.MAX_WHEEL_SPEED,
    )

    nav = NavigationController(robot_model, cfg)
    nav.set_path(waypoints)

    # ---- Perception and fusion ----
    perception = PerceptionPipeline(cfg)
    fusion = SensorFusion(cfg)

    # ---- Simulation log ----
    sim_log: list[dict] = []
    step_count = 0

    print("[Controller A] Starting navigation...")

    # ---- Main control loop ----
    while robot.step(TIME_STEP) != -1:
        # --- Read sensors ---
        gps_vals = gps.getValues()
        compass_vals = compass.getValues()
        rx = gps_vals[0]
        ry = gps_vals[1]
        rtheta = compass_to_heading(compass_vals)

        # Update robot model with ground-truth pose (GPS-corrected)
        robot_model.x = rx
        robot_model.y = ry
        robot_model.theta = rtheta

        # LiDAR
        lidar_ranges = list(lidar.getRangeImage())
        min_lidar = min(lidar_ranges) if lidar_ranges else 999.0

        # Perception
        from sensors import LidarScan, CameraFrame
        import numpy as np
        scan = LidarScan(
            ranges=lidar_ranges,
            robot_x=rx, robot_y=ry,
            robot_heading=rtheta,
            timestamp=step_count * TIME_STEP / 1000.0,
        )
        lidar_obs = perception.process_lidar(scan)

        # Camera (convert BGRA Webots image to numpy)
        cam_image = camera.getImageArray()
        if cam_image:
            img_np = np.array(cam_image, dtype=np.uint8)
            # Webots returns [[row][col]] BGRA — take first 3 channels, transpose
            if img_np.ndim == 3 and img_np.shape[2] >= 3:
                gray = img_np[:, :, 1].T  # green channel as grayscale proxy
            else:
                gray = img_np.T if img_np.ndim == 2 else img_np[:, :, 0].T
            cam_frame = CameraFrame(
                image=gray,
                robot_x=rx, robot_y=ry,
                robot_heading=rtheta,
                timestamp=step_count * TIME_STEP / 1000.0,
            )
            cam_obs = perception.process_camera(cam_frame)
        else:
            cam_obs = None

        # Fuse into grid (online update)
        fusion.update(grid, rx, ry, rtheta, lidar_obs, cam_obs)

        # --- Navigation step ---
        omega_R, omega_L = nav.step(min_lidar_range=min_lidar)

        # --- Send motor commands ---
        left_motor.setVelocity(omega_L)
        right_motor.setVelocity(omega_R)

        # --- Log ---
        goal_dist = math.hypot(cfg.GOAL_POS[0] - rx, cfg.GOAL_POS[1] - ry)
        sim_log.append({
            "step": step_count,
            "t_s": step_count * TIME_STEP / 1000.0,
            "x": rx, "y": ry, "theta_deg": math.degrees(rtheta),
            "goal_dist_m": goal_dist,
            "min_lidar_m": min_lidar,
            "omega_L": omega_L,
            "omega_R": omega_R,
        })

        step_count += 1

        # --- Check termination ---
        if nav.is_done():
            status = "GOAL REACHED" if nav.goal_reached else "TIMED OUT"
            print(f"[Controller A] Navigation ended: {status} after {step_count} steps")
            break

    # ---- Stop motors ----
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ---- Save log ----
    log_path = cfg.RESULTS_LOGS / "webots_run_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        if sim_log:
            writer = csv.DictWriter(f, fieldnames=sim_log[0].keys())
            writer.writeheader()
            writer.writerows(sim_log)
    print(f"[Controller A] Log saved: {log_path}")

    # ---- Save summary JSON ----
    summary = {
        "goal_reached": nav.goal_reached,
        "timed_out": nav.timed_out,
        "collision_count": nav.collision_count,
        "steps": step_count,
        "sim_time_s": step_count * TIME_STEP / 1000.0,
        "path_length_m": nav.path_length_from_log(),
        "path_plan": {
            "algorithm": "A*" if a_result.success else "Dijkstra",
            "nodes_explored": a_result.nodes_explored,
            "planned_length_m": a_result.path_length_m,
        },
    }
    summary_path = cfg.RESULTS_METRICS / "webots_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Controller A] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
