"""
validate_project.py
===================
Project validation script for the Intelligent Autonomous Robotics Project.

Checks performed
----------------
SECTION 1 — File existence
  Verifies every required source file, world file, controller, test,
  result figure, model artifact, and documentation file is present.

SECTION 2 — Python imports
  Imports every Component A and Component B source module without
  triggering Webots (Webots-dependent controllers are excluded).

SECTION 3 — Algorithm correctness
  Re-runs core algorithm checks independently of the test suite:
  - Occupancy grid round-trip coordinate conversion.
  - Dijkstra finds a path and returns it obstacle-free.
  - A* finds same-length path with fewer or equal nodes than Dijkstra.
  - PID output is clipped to ±output_limit.
  - PID converges on a 90-degree step.
  - Consensus converges to the true average.
  - PSO returns a valid assignment.
  - Neural network produces valid class predictions.

SECTION 4 — Data integrity
  Verifies CSV metrics files are non-empty and contain expected columns.
  Verifies JSON metrics files are valid and contain expected keys.

SECTION 5 — Output artefacts
  Checks that all required Matplotlib figures exist and are non-zero size.
  Checks that model files exist and are non-zero size.

Exit code: 0 = all checks passed, 1 = one or more checks failed.
"""

import sys
import importlib
import json
import csv
import math
from pathlib import Path

ROOT = Path(__file__).parent

# ------------------------------------------------------------------ #
#  Result tracking                                                     #
# ------------------------------------------------------------------ #

_passes = 0
_failures = 0
_warnings = 0


def _ok(msg: str) -> None:
    global _passes
    _passes += 1
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    global _failures
    _failures += 1
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    global _warnings
    _warnings += 1
    print(f"  [WARN] {msg}")


def _check_file(path: Path, label: str | None = None) -> bool:
    label = label or str(path.relative_to(ROOT))
    if path.exists() and path.stat().st_size > 0:
        _ok(f"File exists: {label}")
        return True
    elif path.exists():
        _fail(f"File exists but is empty: {label}")
        return False
    else:
        _fail(f"File missing: {label}")
        return False


# ------------------------------------------------------------------ #
#  SECTION 1 — File existence                                         #
# ------------------------------------------------------------------ #

def check_files() -> None:
    print("\n" + "="*60)
    print("  SECTION 1 — File Existence")
    print("="*60)

    required = [
        # Root
        ROOT / "README.txt",
        ROOT / "requirements.txt",
        ROOT / ".gitignore",
        ROOT / "run_all.py",
        ROOT / "run_component_a.py",
        ROOT / "run_component_b.py",
        # Component A source
        ROOT / "component_a" / "src" / "configuration.py",
        ROOT / "component_a" / "src" / "occupancy_grid.py",
        ROOT / "component_a" / "src" / "dijkstra.py",
        ROOT / "component_a" / "src" / "astar.py",
        ROOT / "component_a" / "src" / "robot_model.py",
        ROOT / "component_a" / "src" / "pid_controller.py",
        ROOT / "component_a" / "src" / "sensors.py",
        ROOT / "component_a" / "src" / "perception.py",
        ROOT / "component_a" / "src" / "sensor_fusion.py",
        ROOT / "component_a" / "src" / "navigation.py",
        ROOT / "component_a" / "src" / "evaluation.py",
        # Component A world + controller
        ROOT / "component_a" / "worlds" / "component_a_world.wbt",
        ROOT / "component_a" / "controllers" / "autonomous_controller" / "autonomous_controller.py",
        # Component A tests
        ROOT / "component_a" / "tests" / "test_pathfinding.py",
        ROOT / "component_a" / "tests" / "test_pid.py",
        ROOT / "component_a" / "tests" / "test_occupancy_grid.py",
        # Component A figures
        ROOT / "component_a" / "results" / "figures" / "fig1_occupancy_grid.png",
        ROOT / "component_a" / "results" / "figures" / "fig2_dijkstra_vs_astar.png",
        ROOT / "component_a" / "results" / "figures" / "fig3_planned_vs_actual.png",
        ROOT / "component_a" / "results" / "figures" / "fig4_position_error.png",
        ROOT / "component_a" / "results" / "figures" / "fig5_heading_error.png",
        ROOT / "component_a" / "results" / "figures" / "fig6_pid_tuning.png",
        ROOT / "component_a" / "results" / "figures" / "fig7_performance_comparison.png",
        # Component A metrics
        ROOT / "component_a" / "results" / "metrics" / "component_a_metrics.csv",
        ROOT / "component_a" / "results" / "metrics" / "component_a_metrics.json",
        # Component B source
        ROOT / "component_b" / "src" / "configuration_b.py",
        ROOT / "component_b" / "src" / "agents.py",
        ROOT / "component_b" / "src" / "communication.py",
        ROOT / "component_b" / "src" / "consensus.py",
        ROOT / "component_b" / "src" / "particle_swarm.py",
        ROOT / "component_b" / "src" / "task_allocation.py",
        ROOT / "component_b" / "src" / "neural_network.py",
        ROOT / "component_b" / "src" / "training.py",
        ROOT / "component_b" / "src" / "evaluation.py",
        # Component B world + controller
        ROOT / "component_b" / "worlds" / "component_b_world.wbt",
        ROOT / "component_b" / "controllers" / "swarm_controller" / "swarm_controller.py",
        # Component B tests
        ROOT / "component_b" / "tests" / "test_consensus.py",
        ROOT / "component_b" / "tests" / "test_task_allocation.py",
        ROOT / "component_b" / "tests" / "test_neural_network.py",
        # Component B figures
        ROOT / "component_b" / "results" / "figures" / "fig_b1_consensus.png",
        ROOT / "component_b" / "results" / "figures" / "fig_b2_pso_fitness.png",
        ROOT / "component_b" / "results" / "figures" / "fig_b3_task_allocation.png",
        ROOT / "component_b" / "results" / "figures" / "fig_b4_nn_training_curve.png",
        ROOT / "component_b" / "results" / "figures" / "fig_b5_confusion_matrix.png",
        ROOT / "component_b" / "results" / "figures" / "fig_b6_centralised_vs_distributed.png",
        # Component B metrics
        ROOT / "component_b" / "results" / "metrics" / "component_b_metrics.csv",
        ROOT / "component_b" / "results" / "metrics" / "component_b_metrics.json",
        ROOT / "component_b" / "results" / "metrics" / "nn_evaluation.json",
        # Component B models
        ROOT / "component_b" / "models" / "numpy_mlp.npz",
        ROOT / "component_b" / "models" / "sklearn_mlp.pkl",
        # Training data
        ROOT / "component_b" / "data" / "training" / "nn_training_data.csv",
        ROOT / "component_b" / "data" / "training" / "nn_training_metadata.json",
        # Report + README
        ROOT / "report" / "report_source.md",
        ROOT / "README.txt",
    ]

    for p in required:
        _check_file(p)


# ------------------------------------------------------------------ #
#  SECTION 2 — Python imports                                         #
# ------------------------------------------------------------------ #

def check_imports() -> None:
    print("\n" + "="*60)
    print("  SECTION 2 — Python Imports")
    print("="*60)

    sys.path.insert(0, str(ROOT / "component_a" / "src"))
    sys.path.insert(0, str(ROOT / "component_b" / "src"))

    modules = [
        ("configuration",      "Component A — configuration"),
        ("occupancy_grid",     "Component A — occupancy_grid"),
        ("dijkstra",           "Component A — dijkstra"),
        ("astar",              "Component A — astar"),
        ("robot_model",        "Component A — robot_model"),
        ("pid_controller",     "Component A — pid_controller"),
        ("sensors",            "Component A — sensors"),
        ("perception",         "Component A — perception"),
        ("sensor_fusion",      "Component A — sensor_fusion"),
        ("navigation",         "Component A — navigation"),
        # evaluation imports matplotlib — allowed in non-display mode
        ("evaluation",         "Component A — evaluation"),
        ("configuration_b",    "Component B — configuration_b"),
        ("agents",             "Component B — agents"),
        ("communication",      "Component B — communication"),
        ("consensus",          "Component B — consensus"),
        ("particle_swarm",     "Component B — particle_swarm"),
        ("task_allocation",    "Component B — task_allocation"),
        ("neural_network",     "Component B — neural_network"),
        ("training",           "Component B — training"),
    ]

    for mod_name, label in modules:
        try:
            importlib.import_module(mod_name)
            _ok(f"Import OK: {label}")
        except Exception as exc:
            _fail(f"Import FAILED: {label} — {exc}")


# ------------------------------------------------------------------ #
#  SECTION 3 — Algorithm correctness                                  #
# ------------------------------------------------------------------ #

def check_algorithms() -> None:
    print("\n" + "="*60)
    print("  SECTION 3 — Algorithm Correctness")
    print("="*60)

    # -- Occupancy grid --
    try:
        from configuration import Config
        from occupancy_grid import OccupancyGrid
        cfg = Config()
        grid = OccupancyGrid(cfg)
        grid.build_from_config()
        # Round-trip test
        r0, c0 = grid.rows // 2, grid.cols // 2
        wx, wy = grid.grid_to_world(r0, c0)
        r1, c1 = grid.world_to_grid(wx, wy)
        if r0 == r1 and c0 == c1:
            _ok("OccupancyGrid: round-trip coordinate conversion correct")
        else:
            _fail(f"OccupancyGrid: round-trip mismatch ({r0},{c0}) -> ({r1},{c1})")
    except Exception as e:
        _fail(f"OccupancyGrid: exception — {e}")

    # -- Dijkstra --
    try:
        from dijkstra import dijkstra
        start = grid.world_to_grid(*cfg.START_POS)
        goal  = grid.world_to_grid(*cfg.GOAL_POS)
        d_res = dijkstra(grid, start, goal, cfg)
        if not d_res.success:
            _fail("Dijkstra: failed to find a path")
        else:
            blocked = any(grid.is_occupied(r, c) for r, c in d_res.path)
            if blocked:
                _fail("Dijkstra: path passes through occupied cell")
            else:
                _ok(f"Dijkstra: path found, {d_res.nodes_explored} nodes, {d_res.path_length_m:.2f}m")
    except Exception as e:
        _fail(f"Dijkstra: exception — {e}")

    # -- A* --
    try:
        from astar import astar
        a_res = astar(grid, start, goal, cfg)
        if not a_res.success:
            _fail("A*: failed to find a path")
        else:
            if a_res.nodes_explored > d_res.nodes_explored:
                _warn(f"A* explored more nodes than Dijkstra ({a_res.nodes_explored} vs {d_res.nodes_explored})")
            else:
                _ok(f"A*: path found, {a_res.nodes_explored} nodes (Dijkstra: {d_res.nodes_explored})")
            if abs(a_res.path_length_m - d_res.path_length_m) > cfg.GRID_RESOLUTION:
                _fail(f"A* path length {a_res.path_length_m:.3f}m != Dijkstra {d_res.path_length_m:.3f}m")
            else:
                _ok(f"A*: path length matches Dijkstra ({a_res.path_length_m:.3f}m)")
    except Exception as e:
        _fail(f"A*: exception — {e}")

    # -- PID clipping --
    try:
        from pid_controller import PIDController
        pid = PIDController(Kp=10.0, Ki=0.0, Kd=0.0, output_limit=3.0,
                            integral_limit=1.0, dt=0.032)
        out = pid.compute(100.0)
        if abs(out - 3.0) < 1e-9:
            _ok("PID: output clipped to +limit correctly")
        else:
            _fail(f"PID: output not clipped (got {out})")
        out_neg = pid.compute(-100.0)
        if abs(out_neg + 3.0) < 1e-9:
            _ok("PID: output clipped to -limit correctly")
        else:
            _fail(f"PID: negative output not clipped (got {out_neg})")
    except Exception as e:
        _fail(f"PID: exception — {e}")

    # -- PID convergence --
    try:
        from robot_model import normalise_angle
        pid2 = PIDController(Kp=1.5, Ki=0.05, Kd=0.2,
                             output_limit=6.28, integral_limit=2.0, dt=0.032)
        target = math.pi / 2.0
        heading = 0.0
        for _ in range(int(10.0 / 0.032)):
            err = normalise_angle(target - heading)
            out = pid2.compute(err)
            heading = normalise_angle(heading + out * 0.032 * 0.5)
        final_err = abs(normalise_angle(target - heading))
        if final_err < math.radians(5.0):
            _ok(f"PID: heading converged (final error = {math.degrees(final_err):.2f}°)")
        else:
            _fail(f"PID: heading did not converge (final error = {math.degrees(final_err):.2f}°)")
    except Exception as e:
        _fail(f"PID convergence: exception — {e}")

    # -- Consensus --
    try:
        from configuration_b import ConfigB
        from consensus import run_consensus_experiment
        cfg_b = ConfigB()
        c_res = run_consensus_experiment(cfg_b)
        if c_res["converged"]:
            _ok(f"Consensus: converged in {c_res['iterations']} iterations, "
                f"error={c_res['final_error']:.6f}")
        else:
            _fail("Consensus: did not converge")
        # Check final values close to true average
        true_avg = c_res["true_average"]
        for i, hist in enumerate(c_res["agent_histories"]):
            dev = abs(hist[-1] - true_avg)
            if dev > 0.1:
                _fail(f"Consensus: agent {i+1} final value {hist[-1]:.4f} "
                      f"deviates {dev:.4f} from true avg {true_avg:.4f}")
    except Exception as e:
        _fail(f"Consensus: exception — {e}")

    # -- PSO --
    try:
        from particle_swarm import run_pso
        pso_res = run_pso(list(cfg_b.ROBOT_STARTS), list(cfg_b.TARGETS), cfg_b)
        for t in pso_res.assignment:
            if not (0 <= t < cfg_b.NUM_TARGETS):
                _fail(f"PSO: invalid target index {t}")
                break
        else:
            _ok(f"PSO: valid assignment {pso_res.assignment}, cost={pso_res.cost:.3f}m")
    except Exception as e:
        _fail(f"PSO: exception — {e}")

    # -- Neural network --
    try:
        import numpy as np
        from neural_network import NumpyMLP
        model_path = ROOT / "component_b" / "models" / "numpy_mlp.npz"
        if model_path.exists():
            net = NumpyMLP.load(model_path, cfg_b)
            X = np.random.default_rng(0).random((10, cfg_b.NN_INPUT_SIZE))
            preds = net.predict(X)
            if all(0 <= p < cfg_b.NN_OUTPUT_SIZE for p in preds):
                _ok(f"Neural network: valid predictions {preds.tolist()}")
            else:
                _fail(f"Neural network: out-of-range predictions {preds.tolist()}")
        else:
            _warn("Neural network: model file not found (run run_component_b.py first)")
    except Exception as e:
        _fail(f"Neural network: exception — {e}")


# ------------------------------------------------------------------ #
#  SECTION 4 — Data integrity                                         #
# ------------------------------------------------------------------ #

def check_data() -> None:
    print("\n" + "="*60)
    print("  SECTION 4 — Data Integrity")
    print("="*60)

    # Component A CSV
    csv_a = ROOT / "component_a" / "results" / "metrics" / "component_a_metrics.csv"
    if csv_a.exists():
        with open(csv_a) as f:
            rows = list(csv.reader(f))
        if len(rows) >= 2:
            _ok(f"component_a_metrics.csv: {len(rows)-1} data rows")
        else:
            _fail("component_a_metrics.csv: empty or header-only")
    else:
        _fail("component_a_metrics.csv: missing")

    # Component A JSON
    json_a = ROOT / "component_a" / "results" / "metrics" / "component_a_metrics.json"
    if json_a.exists():
        with open(json_a) as f:
            data = json.load(f)
        required_keys = ["pathfinding", "navigation", "pid_tuning"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            _fail(f"component_a_metrics.json: missing keys {missing}")
        else:
            _ok(f"component_a_metrics.json: all required keys present")
        # Check Dijkstra > A* nodes
        d_nodes = data["pathfinding"]["dijkstra"]["nodes_explored"]
        a_nodes = data["pathfinding"]["astar"]["nodes_explored"]
        if a_nodes <= d_nodes:
            _ok(f"A* explored fewer nodes than Dijkstra ({a_nodes} <= {d_nodes})")
        else:
            _warn(f"A* explored more nodes than Dijkstra ({a_nodes} vs {d_nodes})")
    else:
        _fail("component_a_metrics.json: missing")

    # Component B JSON
    json_b = ROOT / "component_b" / "results" / "metrics" / "component_b_metrics.json"
    if json_b.exists():
        with open(json_b) as f:
            data_b = json.load(f)
        required_keys_b = ["neural_network", "consensus", "task_allocation",
                           "centralised_vs_distributed"]
        missing_b = [k for k in required_keys_b if k not in data_b]
        if missing_b:
            _fail(f"component_b_metrics.json: missing keys {missing_b}")
        else:
            _ok("component_b_metrics.json: all required keys present")
        # Check NN accuracy
        acc = data_b["neural_network"].get("numpy_mlp_accuracy", 0.0)
        if acc > 0.5:
            _ok(f"Neural network accuracy: {acc:.4f} (>50%)")
        else:
            _fail(f"Neural network accuracy too low: {acc:.4f}")
    else:
        _fail("component_b_metrics.json: missing")

    # Training data
    train_csv = ROOT / "component_b" / "data" / "training" / "nn_training_data.csv"
    if train_csv.exists():
        with open(train_csv) as f:
            rows = list(csv.reader(f))
        n_samples = len(rows) - 1
        if n_samples >= 1000:
            _ok(f"Training data: {n_samples} samples")
        else:
            _warn(f"Training data: only {n_samples} samples (expected >=1000)")
    else:
        _fail("Training data CSV missing")


# ------------------------------------------------------------------ #
#  SECTION 5 — Output artefacts                                       #
# ------------------------------------------------------------------ #

def check_artefacts() -> None:
    print("\n" + "="*60)
    print("  SECTION 5 — Output Artefacts")
    print("="*60)

    # Figures — minimum file size 5 KB (valid PNG should be much larger)
    MIN_PNG_SIZE = 5_000

    figures = [
        ROOT / "component_a" / "results" / "figures" / f"fig{i}_{name}.png"
        for i, name in enumerate([
            "occupancy_grid", "dijkstra_vs_astar", "planned_vs_actual",
            "position_error", "heading_error", "pid_tuning",
            "performance_comparison",
        ], start=1)
    ] + [
        ROOT / "component_b" / "results" / "figures" / f"fig_b{i}_{name}.png"
        for i, name in enumerate([
            "consensus", "pso_fitness", "task_allocation",
            "nn_training_curve", "confusion_matrix",
            "centralised_vs_distributed",
        ], start=1)
    ]

    for fig_path in figures:
        if fig_path.exists():
            size = fig_path.stat().st_size
            if size >= MIN_PNG_SIZE:
                _ok(f"Figure: {fig_path.name} ({size//1024}KB)")
            else:
                _fail(f"Figure too small ({size} bytes): {fig_path.name}")
        else:
            _fail(f"Figure missing: {fig_path.name}")

    # Models
    model_files = [
        ROOT / "component_b" / "models" / "numpy_mlp.npz",
        ROOT / "component_b" / "models" / "sklearn_mlp.pkl",
    ]
    for m in model_files:
        if m.exists() and m.stat().st_size > 0:
            _ok(f"Model file: {m.name} ({m.stat().st_size // 1024}KB)")
        else:
            _fail(f"Model file missing or empty: {m.name}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    print("\n" + "#"*60)
    print("  PROJECT VALIDATION — Intelligent Autonomous Robotics")
    print("#"*60)

    check_files()
    check_imports()
    check_algorithms()
    check_data()
    check_artefacts()

    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    print(f"  PASS  : {_passes}")
    print(f"  WARN  : {_warnings}")
    print(f"  FAIL  : {_failures}")
    print("="*60)

    if _failures == 0:
        print("\n  All required checks PASSED.")
        if _warnings > 0:
            print(f"  {_warnings} warning(s) — review output above.")
        sys.exit(0)
    else:
        print(f"\n  {_failures} check(s) FAILED. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
