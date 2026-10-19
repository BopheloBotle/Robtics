"""
configuration.py
================
Central configuration for Component A.

All physical constants, grid parameters, sensor parameters, PID gains,
navigation parameters, and file paths are defined here. Changing a value
in this file propagates throughout the entire Component A pipeline.

Coordinate convention
---------------------
World frame : X points right (East), Y points up (North) in the horizontal plane.
Grid frame  : row 0 is at the top of the map (maximum world-Y), column 0 is at
              the left (minimum world-X). This matches the standard image convention
              used by Matplotlib imshow so that displayed maps look natural.
"""

from pathlib import Path


class Config:
    """All Component A configuration values in one place."""

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    ROOT: Path = Path(__file__).parent.parent.parent   # project root
    COMP_A: Path = ROOT / "component_a"
    RESULTS_FIGURES: Path = COMP_A / "results" / "figures"
    RESULTS_METRICS: Path = COMP_A / "results" / "metrics"
    RESULTS_LOGS: Path = COMP_A / "results" / "logs"
    DATA_RAW: Path = COMP_A / "data" / "raw"
    DATA_PROCESSED: Path = COMP_A / "data" / "processed"

    # ------------------------------------------------------------------ #
    #  Simulation / world                                                  #
    # ------------------------------------------------------------------ #
    WORLD_WIDTH: float = 10.0    # metres, X dimension of arena
    WORLD_HEIGHT: float = 10.0   # metres, Y dimension of arena
    # Arena walls are at ±WORLD_WIDTH/2, ±WORLD_HEIGHT/2 (centred at origin)
    WORLD_ORIGIN_X: float = -5.0  # minimum world X
    WORLD_ORIGIN_Y: float = -5.0  # minimum world Y

    SIM_TIME_STEP: int = 32       # milliseconds per control step in Webots

    # ------------------------------------------------------------------ #
    #  Robot physical parameters                                           #
    # ------------------------------------------------------------------ #
    WHEEL_RADIUS: float = 0.033    # metres (e-puck sized)
    WHEEL_SEPARATION: float = 0.104  # metres (distance between wheel centres)
    MAX_WHEEL_SPEED: float = 6.28   # rad/s (≈ 1 m/s linear)

    # ------------------------------------------------------------------ #
    #  Occupancy grid                                                      #
    # ------------------------------------------------------------------ #
    GRID_RESOLUTION: float = 0.10  # metres per cell
    OBSTACLE_INFLATION: int = 2    # cells to inflate obstacles by
    # Cell state constants
    CELL_FREE: int = 0
    CELL_OCCUPIED: int = 1
    CELL_UNKNOWN: int = 2
    # Sensor hit threshold: LiDAR reading ≤ this → obstacle detected
    LIDAR_OBSTACLE_THRESHOLD: float = 0.25  # metres

    # ------------------------------------------------------------------ #
    #  Sensor parameters                                                   #
    # ------------------------------------------------------------------ #
    LIDAR_RANGE: float = 3.5       # maximum usable LiDAR range (metres)
    LIDAR_MIN_RANGE: float = 0.05  # readings below this are invalid
    LIDAR_SAMPLES: int = 360       # angular samples per scan
    CAMERA_WIDTH: int = 64         # camera image width (pixels)
    CAMERA_HEIGHT: int = 48        # camera image height (pixels)
    CAMERA_FOV: float = 1.0472     # horizontal field of view (radians, ≈60°)

    # ------------------------------------------------------------------ #
    #  Start / goal positions (world frame, metres)                        #
    # ------------------------------------------------------------------ #
    START_POS: tuple[float, float] = (-3.5, -3.5)
    GOAL_POS: tuple[float, float] = (3.5, 3.5)
    GOAL_TOLERANCE: float = 0.15   # metres — distance to consider goal reached

    # ------------------------------------------------------------------ #
    #  Obstacles in the Webots world (centre x, centre y, half-size)      #
    #  These must match the Webots world file exactly.                     #
    # ------------------------------------------------------------------ #
    OBSTACLES: list[dict] = [
        {"x": 0.0,  "y": 0.0,  "w": 0.8, "h": 0.8},
        {"x": 2.0,  "y": -1.5, "w": 0.6, "h": 1.2},
        {"x": -2.0, "y": 1.5,  "w": 1.0, "h": 0.5},
        {"x": 1.0,  "y": 2.5,  "w": 0.5, "h": 0.5},
        {"x": -1.5, "y": -2.0, "w": 0.7, "h": 0.7},
        {"x": 3.0,  "y": 0.5,  "w": 0.4, "h": 0.8},
        {"x": -3.0, "y": 0.0,  "w": 0.5, "h": 0.5},
    ]

    # ------------------------------------------------------------------ #
    #  PID gains — three gain sets for the tuning experiment               #
    # ------------------------------------------------------------------ #
    PID_GAINS: list[dict] = [
        {"name": "Under-damped",  "Kp": 3.0, "Ki": 0.1, "Kd": 0.05},
        {"name": "Well-tuned",    "Kp": 1.5, "Ki": 0.05, "Kd": 0.2},
        {"name": "Over-damped",   "Kp": 0.5, "Ki": 0.01, "Kd": 0.5},
    ]
    # Default gains used by the navigation controller
    PID_KP: float = 1.5
    PID_KI: float = 0.05
    PID_KD: float = 0.2
    PID_OUTPUT_LIMIT: float = MAX_WHEEL_SPEED
    PID_INTEGRAL_LIMIT: float = 2.0
    PID_DT: float = SIM_TIME_STEP / 1000.0  # seconds

    # ------------------------------------------------------------------ #
    #  Navigation                                                          #
    # ------------------------------------------------------------------ #
    WAYPOINT_TOLERANCE: float = 0.20   # metres — switch to next waypoint
    NAV_TIMEOUT: float = 120.0         # seconds — abort if not converged
    HEADING_SCALE: float = 1.0         # wheel-speed differential scale factor

    # ------------------------------------------------------------------ #
    #  Pathfinding benchmark                                               #
    # ------------------------------------------------------------------ #
    BENCHMARK_RUNS: int = 5   # repeated timing trials for averaging

    # ------------------------------------------------------------------ #
    #  Random seed for reproducibility                                     #
    # ------------------------------------------------------------------ #
    RANDOM_SEED: int = 42

    def __post_init__(self) -> None:
        """Create output directories if they do not exist."""
        for path in [
            self.RESULTS_FIGURES, self.RESULTS_METRICS,
            self.RESULTS_LOGS, self.DATA_RAW, self.DATA_PROCESSED,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def __init__(self) -> None:
        self.__post_init__()
