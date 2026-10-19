"""
configuration_b.py
==================
Central configuration for Component B (swarm robotics).

All physical, simulation, algorithm, and neural-network parameters are
defined here.  Changing a value propagates to all Component B modules.
"""

from pathlib import Path


class ConfigB:
    """All Component B configuration values."""

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    ROOT: Path = Path(__file__).parent.parent.parent
    COMP_B: Path = ROOT / "component_b"
    RESULTS_FIGURES: Path  = COMP_B / "results" / "figures"
    RESULTS_METRICS: Path  = COMP_B / "results" / "metrics"
    RESULTS_LOGS: Path     = COMP_B / "results" / "logs"
    DATA_TRAINING: Path    = COMP_B / "data" / "training"
    DATA_EVALUATION: Path  = COMP_B / "data" / "evaluation"
    MODELS_DIR: Path       = COMP_B / "models"

    # ------------------------------------------------------------------ #
    #  World / arena                                                       #
    # ------------------------------------------------------------------ #
    WORLD_WIDTH: float  = 12.0   # metres
    WORLD_HEIGHT: float = 12.0
    WORLD_ORIGIN_X: float = -6.0
    WORLD_ORIGIN_Y: float = -6.0
    SIM_TIME_STEP: int = 32      # milliseconds

    # ------------------------------------------------------------------ #
    #  Robot                                                               #
    # ------------------------------------------------------------------ #
    WHEEL_RADIUS: float     = 0.033
    WHEEL_SEPARATION: float = 0.104
    MAX_WHEEL_SPEED: float  = 6.28
    ROBOT_IDS: list[str]    = ["robot_1", "robot_2", "robot_3"]
    NUM_ROBOTS: int         = 3

    # Starting positions (world frame, metres) — must match world file
    ROBOT_STARTS: list[tuple[float, float]] = [
        (-3.0, -3.0),
        ( 0.0, -3.0),
        ( 3.0, -3.0),
    ]

    # ------------------------------------------------------------------ #
    #  Targets                                                             #
    # ------------------------------------------------------------------ #
    TARGETS: list[tuple[float, float]] = [
        (-3.0,  3.0),
        ( 0.0,  3.5),
        ( 3.0,  3.0),
        (-3.5,  0.0),
        ( 3.5,  0.0),
    ]
    NUM_TARGETS: int = len(TARGETS)

    # ------------------------------------------------------------------ #
    #  Obstacles (for PSO fitness and task-allocation cost)               #
    # ------------------------------------------------------------------ #
    OBSTACLES: list[dict] = [
        {"x": -2.0, "y":  2.0, "w": 0.8, "h": 0.8},
        {"x":  2.0, "y":  1.5, "w": 0.6, "h": 1.0},
        {"x":  0.0, "y":  0.0, "w": 1.0, "h": 0.5},
    ]

    # ------------------------------------------------------------------ #
    #  Communication                                                       #
    # ------------------------------------------------------------------ #
    COMM_RANGE: float = 10.0        # Webots emitter/receiver range (metres)
    MSG_TIMEOUT_S: float = 2.0      # seconds before a message is considered stale
    # Topology: ring (each robot communicates with its two neighbours)
    COMM_TOPOLOGY: str = "ring"

    # ------------------------------------------------------------------ #
    #  Consensus algorithm                                                 #
    # ------------------------------------------------------------------ #
    # Average consensus on shared scalar state (e.g. target count estimate)
    CONSENSUS_EPSILON: float = 0.005   # convergence threshold
    CONSENSUS_ALPHA: float   = 0.3     # update gain (must be < 1/max_degree)
    CONSENSUS_MAX_ITER: int  = 200     # maximum iterations
    CONSENSUS_NUM_AGENTS: int = 3
    # Initial state values differ so convergence is non-trivial
    CONSENSUS_INIT_VALUES: list[float] = [1.0, 4.0, 7.0]

    # ------------------------------------------------------------------ #
    #  Particle Swarm Optimisation                                         #
    # ------------------------------------------------------------------ #
    PSO_NUM_PARTICLES: int = 30
    PSO_MAX_ITER: int      = 100
    PSO_W: float           = 0.7    # inertia weight
    PSO_C1: float          = 1.5    # cognitive coefficient
    PSO_C2: float          = 1.5    # social coefficient
    PSO_V_MAX: float       = 2.0    # maximum velocity per dimension

    # ------------------------------------------------------------------ #
    #  Neural network                                                      #
    # ------------------------------------------------------------------ #
    # Input features: front_dist, left_dist, right_dist, goal_angle_norm, heading_err_norm
    NN_INPUT_SIZE: int  = 5
    # Output classes: 0=forward, 1=turn_left, 2=turn_right, 3=stop
    NN_OUTPUT_SIZE: int = 4
    NN_HIDDEN_SIZES: list[int] = [16, 8]    # two hidden layers
    NN_LEARNING_RATE: float    = 0.01
    NN_EPOCHS: int             = 200
    NN_BATCH_SIZE: int         = 32
    NN_TRAIN_RATIO: float      = 0.8    # fraction of data for training
    NN_RANDOM_SEED: int        = 42

    # Number of synthetic training samples to generate
    NN_NUM_SAMPLES: int = 2000

    # Obstacle-distance threshold below which the teacher labels a stop/turn
    TEACHER_OBSTACLE_THRESHOLD: float = 0.3   # metres
    TEACHER_TURN_THRESHOLD: float     = 0.6   # metres

    # ------------------------------------------------------------------ #
    #  Navigation (per-robot)                                              #
    # ------------------------------------------------------------------ #
    PID_KP: float          = 1.5
    PID_KI: float          = 0.05
    PID_KD: float          = 0.2
    PID_OUTPUT_LIMIT: float = 6.28
    PID_INTEGRAL_LIMIT: float = 2.0
    PID_DT: float          = SIM_TIME_STEP / 1000.0
    GOAL_TOLERANCE: float  = 0.20    # metres
    WAYPOINT_TOLERANCE: float = 0.25

    # ------------------------------------------------------------------ #
    #  Experiment / random seed                                            #
    # ------------------------------------------------------------------ #
    RANDOM_SEED: int = 42
    SIM_DURATION_S: float = 120.0   # offline experiment duration

    def __init__(self) -> None:
        for p in [
            self.RESULTS_FIGURES, self.RESULTS_METRICS, self.RESULTS_LOGS,
            self.DATA_TRAINING, self.DATA_EVALUATION, self.MODELS_DIR,
        ]:
            p.mkdir(parents=True, exist_ok=True)
