# Component A — Autonomous Mobile Robot

## Overview

A single differential-drive robot navigates from start (−3.5, −3.5) to goal (3.5, 3.5)
in a 10 m × 10 m arena with seven static obstacles.

## Key modules

| Module | Purpose |
|--------|---------|
| `src/configuration.py` | All parameters (grid, PID, sensor, paths) |
| `src/occupancy_grid.py` | 100×100 binary occupancy grid with world↔grid conversion |
| `src/dijkstra.py` | Dijkstra's shortest-path algorithm |
| `src/astar.py` | A\* with octile heuristic |
| `src/robot_model.py` | Differential-drive kinematic model |
| `src/pid_controller.py` | PID with anti-windup + derivative filter |
| `src/sensors.py` | Synthetic LiDAR and camera for offline testing |
| `src/perception.py` | LiDAR and camera processing pipeline |
| `src/sensor_fusion.py` | Evidence-accumulation sensor fusion |
| `src/navigation.py` | Waypoint-following navigation controller |
| `src/evaluation.py` | Offline experiments, simulation, all 7 figures |

## Offline pipeline

```
python run_component_a.py
```

## Webots simulation

Open `worlds/component_a_world.wbt` in Webots R2025a and press Play.

## Tests (54 total, all passing)

```
python -m pytest component_a/tests/ -v
```
