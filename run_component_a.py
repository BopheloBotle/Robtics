"""
run_component_a.py
==================
Offline experiment runner for Component A.

Executes the full offline pipeline without starting Webots:
  - Builds an occupancy grid from a synthetic map matching the Webots world.
  - Runs Dijkstra and A* on the grid and records metrics.
  - Runs PID tuning trials on a simulated heading/position error signal.
  - Generates all required Matplotlib figures.
  - Writes CSV and JSON metric summaries.

To run the actual Webots simulation, open Webots and load:
  component_a/worlds/component_a_world.wbt
Then the controller runs automatically inside Webots.
"""

import sys
from pathlib import Path

# Ensure the component_a src directory is on the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "component_a" / "src"))

from configuration import Config
from occupancy_grid import OccupancyGrid
from dijkstra import dijkstra
from astar import astar
from evaluation import (
    run_pathfinding_comparison,
    run_pid_tuning_experiment,
    generate_all_component_a_figures,
)


def main() -> None:
    cfg = Config()
    print("[Component A] Starting offline experiment pipeline...")

    # --- Occupancy grid ---
    grid = OccupancyGrid(cfg)
    grid.build_from_config()
    print(f"[Component A] Grid built: {grid.rows}x{grid.cols} cells, "
          f"resolution={cfg.GRID_RESOLUTION}m")

    # --- Pathfinding comparison ---
    print("[Component A] Running Dijkstra vs A* comparison...")
    results = run_pathfinding_comparison(grid, cfg)
    print(f"[Component A] Dijkstra nodes explored : {results['dijkstra']['nodes_explored']}")
    print(f"[Component A] A*       nodes explored : {results['astar']['nodes_explored']}")
    print(f"[Component A] Dijkstra path length    : {results['dijkstra']['path_length']:.3f}m")
    print(f"[Component A] A*       path length    : {results['astar']['path_length']:.3f}m")

    # --- PID tuning ---
    print("[Component A] Running PID tuning trials...")
    pid_results = run_pid_tuning_experiment(cfg)
    print(f"[Component A] PID trials completed: {len(pid_results)} gain sets")

    # --- Figures and metrics ---
    print("[Component A] Generating figures and saving metrics...")
    generate_all_component_a_figures(grid, results, pid_results, cfg)

    print("[Component A] Offline pipeline complete.")
    print(f"[Component A] Figures saved to : {ROOT / 'component_a' / 'results' / 'figures'}")
    print(f"[Component A] Metrics saved to : {ROOT / 'component_a' / 'results' / 'metrics'}")


if __name__ == "__main__":
    main()
