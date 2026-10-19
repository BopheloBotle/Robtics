"""
evaluation.py
=============
Offline evaluation and visualisation for Component A.

Functions
---------
run_pathfinding_comparison  — run Dijkstra vs A* benchmark and return metrics
run_pid_tuning_experiment   — simulate PID step responses for multiple gain sets
generate_all_component_a_figures — produce and save all required Matplotlib figures
save_metrics_csv            — save a metrics dict to CSV
save_metrics_json           — save a metrics dict to JSON

All figures are saved to component_a/results/figures/.
All metrics are saved to component_a/results/metrics/.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from occupancy_grid import OccupancyGrid
from dijkstra import dijkstra
from astar import astar
from pid_controller import PIDController
from robot_model import DifferentialDriveRobot, normalise_angle
from navigation import NavigationController

if TYPE_CHECKING:
    from configuration import Config


# ------------------------------------------------------------------ #
#  Pathfinding comparison                                              #
# ------------------------------------------------------------------ #

def run_pathfinding_comparison(
    grid: OccupancyGrid, cfg: "Config"
) -> dict:
    """
    Run Dijkstra and A* on the configured grid and return a metrics dict.

    Runs BENCHMARK_RUNS timing trials and averages the planning time.
    Returns actual path and node counts (not averaged).
    """
    start_cell = grid.world_to_grid(*cfg.START_POS)
    goal_cell  = grid.world_to_grid(*cfg.GOAL_POS)

    # Single-run results (for path and node counts)
    d_result = dijkstra(grid, start_cell, goal_cell, cfg)
    a_result = astar(grid, start_cell, goal_cell, cfg)

    # Timing benchmark
    d_times: list[float] = []
    a_times: list[float] = []
    for _ in range(cfg.BENCHMARK_RUNS):
        t0 = time.perf_counter()
        dijkstra(grid, start_cell, goal_cell, cfg)
        d_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        astar(grid, start_cell, goal_cell, cfg)
        a_times.append(time.perf_counter() - t0)

    return {
        "dijkstra": {
            "path": d_result.path,
            "nodes_explored": d_result.nodes_explored,
            "path_length": d_result.path_length_m,
            "elapsed_s": d_result.elapsed_s,
            "mean_elapsed_s": sum(d_times) / len(d_times),
            "success": d_result.success,
        },
        "astar": {
            "path": a_result.path,
            "nodes_explored": a_result.nodes_explored,
            "path_length": a_result.path_length_m,
            "elapsed_s": a_result.elapsed_s,
            "mean_elapsed_s": sum(a_times) / len(a_times),
            "success": a_result.success,
        },
        "start_cell": start_cell,
        "goal_cell": goal_cell,
    }


# ------------------------------------------------------------------ #
#  PID tuning experiment                                               #
# ------------------------------------------------------------------ #

def _simulate_heading_pid(
    Kp: float, Ki: float, Kd: float, cfg: "Config",
    duration_s: float = 8.0
) -> dict:
    """
    Simulate a heading step-response test.

    The robot is commanded from heading 0 to heading π/2 (90°).
    The PID output drives a simple first-order heading integrator.
    Records error, output, and heading over time.
    """
    dt = cfg.PID_DT
    steps = int(duration_s / dt)
    target = math.pi / 2.0  # 90° step

    pid = PIDController(
        Kp=Kp, Ki=Ki, Kd=Kd,
        output_limit=cfg.MAX_WHEEL_SPEED,
        integral_limit=cfg.PID_INTEGRAL_LIMIT,
        dt=dt,
    )

    heading = 0.0
    times, errors, outputs, headings = [], [], [], []

    for k in range(steps):
        t = k * dt
        error = normalise_angle(target - heading)
        output = pid.compute(error)
        # Simple integrator model: heading changes proportional to output
        heading = normalise_angle(heading + output * dt * 0.5)

        times.append(t)
        errors.append(error)
        outputs.append(output)
        headings.append(heading)

    return {
        "times": times,
        "errors": errors,
        "outputs": outputs,
        "headings": headings,
        "target": target,
        "steady_state_error": abs(normalise_angle(target - headings[-1])),
        "overshoot_rad": max(
            0.0,
            max(headings) - target
            if max(headings) > target
            else 0.0
        ),
    }


def run_pid_tuning_experiment(cfg: "Config") -> list[dict]:
    """Run step-response simulations for each gain set in Config.PID_GAINS."""
    results = []
    for gains in cfg.PID_GAINS:
        res = _simulate_heading_pid(
            gains["Kp"], gains["Ki"], gains["Kd"], cfg
        )
        res["name"] = gains["name"]
        res["Kp"] = gains["Kp"]
        res["Ki"] = gains["Ki"]
        res["Kd"] = gains["Kd"]
        results.append(res)
    return results


# ------------------------------------------------------------------ #
#  Simulate closed-loop navigation (offline)                           #
# ------------------------------------------------------------------ #

def simulate_navigation(grid: OccupancyGrid, path: list[tuple[int, int]],
                        cfg: "Config") -> dict:
    """
    Simulate the robot following a path offline (no Webots).

    Uses a pure PID + kinematic-model loop that does not depend on
    wall-clock time, so it runs reliably without a Webots connection.
    Returns planned vs actual trajectory data and a full timestep log.
    """
    if not path:
        return {"success": False, "log": [], "planned_world": [], "actual_world": []}

    waypoints = [grid.grid_to_world(r, c) for r, c in path]

    sx, sy = cfg.START_POS
    robot = DifferentialDriveRobot(
        x=sx, y=sy, theta=math.pi / 4.0,
        wheel_radius=cfg.WHEEL_RADIUS,
        wheel_separation=cfg.WHEEL_SEPARATION,
        max_wheel_speed=cfg.MAX_WHEEL_SPEED,
    )

    heading_pid = PIDController(
        Kp=cfg.PID_KP, Ki=cfg.PID_KI, Kd=cfg.PID_KD,
        output_limit=cfg.MAX_WHEEL_SPEED,
        integral_limit=cfg.PID_INTEGRAL_LIMIT,
        dt=cfg.PID_DT,
    )
    speed_pid = PIDController(
        Kp=0.8, Ki=0.0, Kd=0.05,
        output_limit=cfg.WHEEL_RADIUS * cfg.MAX_WHEEL_SPEED,
        integral_limit=1.0,
        dt=cfg.PID_DT,
    )

    dt = cfg.PID_DT
    max_steps = int(cfg.NAV_TIMEOUT / dt)
    actual: list[tuple[float, float]] = [(robot.x, robot.y)]
    log: list[dict] = []
    wp_idx = 0
    goal_reached = False
    sim_time = 0.0

    for _ in range(max_steps):
        if wp_idx >= len(waypoints):
            goal_reached = True
            break

        wx, wy = waypoints[wp_idx]
        dist = robot.distance_to(wx, wy)
        heading_err = robot.heading_error_to(wx, wy)
        is_final = (wp_idx == len(waypoints) - 1)
        tolerance = cfg.GOAL_TOLERANCE if is_final else cfg.WAYPOINT_TOLERANCE

        if dist < tolerance:
            wp_idx += 1
            heading_pid.reset()
            speed_pid.reset()
            if wp_idx >= len(waypoints):
                goal_reached = True
                break
            wx, wy = waypoints[wp_idx]
            dist = robot.distance_to(wx, wy)
            heading_err = robot.heading_error_to(wx, wy)

        omega_corr = heading_pid.compute(heading_err)
        v_cmd = speed_pid.compute(dist) * max(0.1, 1.0 - abs(heading_err) / math.pi)
        omega_cmd = omega_corr * cfg.HEADING_SCALE
        omega_R, omega_L = robot.wheel_speeds_from_v_omega(v_cmd, omega_cmd)
        robot.set_wheel_speeds(omega_R, omega_L)
        robot.update(dt)
        sim_time += dt
        actual.append((robot.x, robot.y))
        log.append({
            "t": sim_time,
            "x": robot.x, "y": robot.y,
            "theta": robot.theta,
            "waypoint_idx": wp_idx,
            "dist_to_wp": dist,
            "heading_error": heading_err,
            "v": v_cmd, "omega": omega_cmd,
            "omega_R": omega_R, "omega_L": omega_L,
        })

    return {
        "success": goal_reached,
        "log": log,
        "planned_world": waypoints,
        "actual_world": actual,
        "sim_time": sim_time,
    }


# ------------------------------------------------------------------ #
#  Figure generation                                                   #
# ------------------------------------------------------------------ #

def _ensure_dirs(cfg: "Config") -> None:
    cfg.RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_METRICS.mkdir(parents=True, exist_ok=True)


def generate_all_component_a_figures(
    grid: OccupancyGrid,
    path_results: dict,
    pid_results: list[dict],
    cfg: "Config",
) -> None:
    """Generate and save all required Component A Matplotlib figures."""
    _ensure_dirs(cfg)

    # Simulate navigation offline
    nav_sim = simulate_navigation(grid, path_results["dijkstra"]["path"], cfg)

    fig1_occupancy_grid(grid, path_results, cfg)
    fig2_dijkstra_vs_astar(grid, path_results, cfg)
    fig3_planned_vs_actual(nav_sim, cfg)
    fig4_position_error(nav_sim, cfg)
    fig5_heading_error(nav_sim, cfg)
    fig6_pid_tuning(pid_results, cfg)
    fig7_performance_comparison(path_results, pid_results, cfg)

    save_metrics_csv(path_results, pid_results, nav_sim, cfg)
    save_metrics_json(path_results, pid_results, nav_sim, cfg)


def fig1_occupancy_grid(
    grid: OccupancyGrid, path_results: dict, cfg: "Config"
) -> None:
    """Figure 1: Occupancy grid with start/goal markers."""
    fig, ax = plt.subplots(figsize=(8, 8))
    data = grid.get_array().astype(float)
    cmap = plt.cm.gray_r
    im = ax.imshow(
        data, cmap=cmap, vmin=0, vmax=2,
        extent=[
            cfg.WORLD_ORIGIN_X,
            cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH,
            cfg.WORLD_ORIGIN_Y,
            cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT,
        ],
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="0=Free  1=Occupied  2=Unknown")

    sx, sy = cfg.START_POS
    gx, gy = cfg.GOAL_POS
    ax.plot(sx, sy, "go", markersize=12, label="Start", zorder=5)
    ax.plot(gx, gy, "r*", markersize=14, label="Goal", zorder=5)

    ax.set_title("Figure 1 — Occupancy Grid (start/goal)", fontsize=13)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig1_occupancy_grid.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig1_occupancy_grid.png saved")


def fig2_dijkstra_vs_astar(
    grid: OccupancyGrid, path_results: dict, cfg: "Config"
) -> None:
    """Figure 2: Dijkstra and A* paths overlaid on occupancy grid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    extent = [
        cfg.WORLD_ORIGIN_X, cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH,
        cfg.WORLD_ORIGIN_Y, cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT,
    ]
    data = grid.get_array().astype(float)

    for ax, algo in zip(axes, ["dijkstra", "astar"]):
        ax.imshow(data, cmap="gray_r", vmin=0, vmax=2,
                  extent=extent, origin="lower", alpha=0.7)
        res = path_results[algo]
        if res["path"]:
            wx = [grid.grid_to_world(r, c)[0] for r, c in res["path"]]
            wy = [grid.grid_to_world(r, c)[1] for r, c in res["path"]]
            color = "blue" if algo == "dijkstra" else "darkorange"
            ax.plot(wx, wy, color=color, linewidth=2, label=algo.capitalize())
        sx, sy = cfg.START_POS
        gx, gy = cfg.GOAL_POS
        ax.plot(sx, sy, "go", markersize=10, zorder=5)
        ax.plot(gx, gy, "r*", markersize=12, zorder=5)
        name = "Dijkstra" if algo == "dijkstra" else "A*"
        ax.set_title(
            f"{name}  |  nodes={res['nodes_explored']}  |  "
            f"length={res['path_length']:.2f}m  |  "
            f"time={res['mean_elapsed_s']*1000:.2f}ms",
            fontsize=10,
        )
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 2 — Dijkstra vs A* Path Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig2_dijkstra_vs_astar.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig2_dijkstra_vs_astar.png saved")


def fig3_planned_vs_actual(nav_sim: dict, cfg: "Config") -> None:
    """Figure 3: Planned trajectory vs actual (simulated) trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if nav_sim["planned_world"]:
        px = [w[0] for w in nav_sim["planned_world"]]
        py = [w[1] for w in nav_sim["planned_world"]]
        ax.plot(px, py, "b--", linewidth=2, label="Planned (waypoints)")
    if nav_sim["actual_world"]:
        ax_vals = [w[0] for w in nav_sim["actual_world"]]
        ay_vals = [w[1] for w in nav_sim["actual_world"]]
        ax.plot(ax_vals, ay_vals, "r-", linewidth=1.5, alpha=0.8, label="Actual")

    sx, sy = cfg.START_POS
    gx, gy = cfg.GOAL_POS
    ax.plot(sx, sy, "go", markersize=12, label="Start", zorder=5)
    ax.plot(gx, gy, "r*", markersize=14, label="Goal", zorder=5)

    status = "Goal reached" if nav_sim["success"] else "Goal NOT reached"
    ax.set_title(f"Figure 3 — Planned vs Actual Trajectory  [{status}]", fontsize=12)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_xlim(cfg.WORLD_ORIGIN_X, cfg.WORLD_ORIGIN_X + cfg.WORLD_WIDTH)
    ax.set_ylim(cfg.WORLD_ORIGIN_Y, cfg.WORLD_ORIGIN_Y + cfg.WORLD_HEIGHT)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig3_planned_vs_actual.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig3_planned_vs_actual.png saved")


def fig4_position_error(nav_sim: dict, cfg: "Config") -> None:
    """Figure 4: Position error over time."""
    log = nav_sim.get("log", [])
    if not log:
        print("  [fig] fig4 — no log data, skipping")
        return

    times = [e["t"] for e in log]
    errors = [e["dist_to_wp"] for e in log]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, errors, color="steelblue", linewidth=1.5)
    ax.axhline(cfg.GOAL_TOLERANCE, color="red", linestyle="--", label=f"Goal tolerance ({cfg.GOAL_TOLERANCE}m)")
    ax.set_title("Figure 4 — Position Error Over Time", fontsize=12)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Distance to current waypoint (m)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig4_position_error.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig4_position_error.png saved")


def fig5_heading_error(nav_sim: dict, cfg: "Config") -> None:
    """Figure 5: Heading error over time."""
    log = nav_sim.get("log", [])
    if not log:
        print("  [fig] fig5 — no log data, skipping")
        return

    times = [e["t"] for e in log]
    errors = [math.degrees(e["heading_error"]) for e in log]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, errors, color="darkorange", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Figure 5 — Heading Error Over Time", fontsize=12)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Heading error (degrees)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig5_heading_error.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig5_heading_error.png saved")


def fig6_pid_tuning(pid_results: list[dict], cfg: "Config") -> None:
    """Figure 6: PID tuning comparison — heading error over time for each gain set."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = ["crimson", "seagreen", "royalblue"]

    for ax, res, color in zip(axes, pid_results, colors):
        times = res["times"]
        errors_deg = [math.degrees(e) for e in res["errors"]]
        ax.plot(times, errors_deg, color=color, linewidth=1.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"{res['name']}\n"
            f"Kp={res['Kp']}  Ki={res['Ki']}  Kd={res['Kd']}\n"
            f"SS error={math.degrees(res['steady_state_error']):.2f}°",
            fontsize=9,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heading error (°)")
        ax.grid(True, alpha=0.4)

    fig.suptitle("Figure 6 — PID Tuning Comparison (90° Step Response)", fontsize=13)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig6_pid_tuning.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig6_pid_tuning.png saved")


def fig7_performance_comparison(
    path_results: dict, pid_results: list[dict], cfg: "Config"
) -> None:
    """Figure 7: Bar chart comparing Dijkstra vs A* performance metrics."""
    labels = ["Nodes explored", "Path length (×10 m)", "Planning time (ms)"]
    d = path_results["dijkstra"]
    a = path_results["astar"]
    d_vals = [
        d["nodes_explored"],
        d["path_length"] * 10,
        d["mean_elapsed_s"] * 1000,
    ]
    a_vals = [
        a["nodes_explored"],
        a["path_length"] * 10,
        a["mean_elapsed_s"] * 1000,
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_d = ax.bar(x - width / 2, d_vals, width, label="Dijkstra", color="steelblue")
    bars_a = ax.bar(x + width / 2, a_vals, width, label="A*", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Figure 7 — Dijkstra vs A* Performance Comparison", fontsize=12)
    ax.set_ylabel("Value (see x-axis labels for units)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for bar in list(bars_d) + list(bars_a):
        ax.annotate(
            f"{bar.get_height():.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_FIGURES / "fig7_performance_comparison.png", dpi=150)
    plt.close(fig)
    print("  [fig] fig7_performance_comparison.png saved")


# ------------------------------------------------------------------ #
#  Metrics saving                                                      #
# ------------------------------------------------------------------ #

def save_metrics_csv(
    path_results: dict,
    pid_results: list[dict],
    nav_sim: dict,
    cfg: "Config",
) -> None:
    """Save key metrics to CSV."""
    csv_path = cfg.RESULTS_METRICS / "component_a_metrics.csv"
    rows = [
        ["metric", "value", "unit"],
        ["dijkstra_nodes_explored", path_results["dijkstra"]["nodes_explored"], "cells"],
        ["dijkstra_path_length",    path_results["dijkstra"]["path_length"],    "m"],
        ["dijkstra_mean_time_ms",   path_results["dijkstra"]["mean_elapsed_s"] * 1000, "ms"],
        ["astar_nodes_explored",    path_results["astar"]["nodes_explored"],    "cells"],
        ["astar_path_length",       path_results["astar"]["path_length"],       "m"],
        ["astar_mean_time_ms",      path_results["astar"]["mean_elapsed_s"] * 1000, "ms"],
        ["nav_success",             int(nav_sim.get("success", False)),         "bool"],
        ["nav_sim_time_s",          nav_sim.get("sim_time", 0.0),               "s"],
    ]
    for res in pid_results:
        rows.append([
            f"pid_{res['name']}_ss_error_deg",
            math.degrees(res["steady_state_error"]),
            "deg",
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  [metrics] component_a_metrics.csv saved")


def save_metrics_json(
    path_results: dict,
    pid_results: list[dict],
    nav_sim: dict,
    cfg: "Config",
) -> None:
    """Save key metrics to JSON."""
    json_path = cfg.RESULTS_METRICS / "component_a_metrics.json"
    data = {
        "pathfinding": {
            "dijkstra": {
                "nodes_explored": path_results["dijkstra"]["nodes_explored"],
                "path_length_m": path_results["dijkstra"]["path_length"],
                "mean_planning_time_ms": path_results["dijkstra"]["mean_elapsed_s"] * 1000,
                "success": path_results["dijkstra"]["success"],
            },
            "astar": {
                "nodes_explored": path_results["astar"]["nodes_explored"],
                "path_length_m": path_results["astar"]["path_length"],
                "mean_planning_time_ms": path_results["astar"]["mean_elapsed_s"] * 1000,
                "success": path_results["astar"]["success"],
            },
        },
        "navigation": {
            "success": nav_sim.get("success", False),
            "sim_time_s": nav_sim.get("sim_time", 0.0),
        },
        "pid_tuning": [
            {
                "name": r["name"],
                "Kp": r["Kp"], "Ki": r["Ki"], "Kd": r["Kd"],
                "steady_state_error_deg": math.degrees(r["steady_state_error"]),
                "overshoot_rad": r["overshoot_rad"],
            }
            for r in pid_results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [metrics] component_a_metrics.json saved")
