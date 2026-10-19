"""
swarm_controller.py
===================
Webots controller for Component B — Cooperative Swarm Robotics.

This single controller file is used by ALL three robots in the swarm.
Each robot instance identifies itself via its controllerArgs[0] parameter
(e.g. "robot_1", "robot_2", "robot_3").

Execution sequence per robot
-----------------------------
1. Read robot ID from controllerArgs.
2. Initialise Webots devices (LiDAR, GPS, Compass, Emitter, Receiver, motors).
3. Load PSO-computed target assignment from models/ (or compute on the fly).
4. Load trained neural network for navigation action classification.
5. Main loop:
   a. Read GPS + Compass → update agent pose.
   b. Read LiDAR → extract front/left/right distances.
   c. Receive neighbour messages → update consensus state.
   d. Broadcast own state message.
   e. Query neural network for navigation action.
   f. Override with safety fallback if obstacle is too close.
   g. Convert action to wheel velocities.
   h. Log step.
6. Stop when target reached or simulation ends.

Communication topology
-----------------------
Webots Emitter broadcasts on channel 1 to all robots.
Receiver filters messages to ring neighbours in software
(see communication.py get_ring_neighbours).

Device names (must match world file)
--------------------------------------
  LiDAR    : "lidar"
  GPS      : "gps"
  Compass  : "compass"
  Emitter  : "emitter"
  Receiver : "receiver"
  Left motor : "left_wheel_motor"
  Right motor: "right_wheel_motor"

Running
-------
Open component_b/worlds/component_b_world.wbt in Webots.
All three controllers start automatically.
"""

import sys
import os
import math
import csv
import json
import struct
from pathlib import Path

_CTRL_DIR = Path(__file__).parent
_COMP_B_SRC = _CTRL_DIR.parent.parent / "src"
sys.path.insert(0, str(_COMP_B_SRC))

try:
    from controller import Robot, Lidar, GPS, Compass, Motor, Emitter, Receiver
except ImportError:
    print(
        "ERROR: Webots controller library not found.\n"
        "Open component_b/worlds/component_b_world.wbt in Webots to run this controller."
    )
    sys.exit(1)

from configuration_b import ConfigB
from agents import AgentState
from communication import encode_message, decode_message, get_ring_neighbours
from consensus import online_consensus_update
from neural_network import NumpyMLP


def compass_to_heading(cv: list) -> float:
    return math.atan2(cv[0], cv[2])


def _lidar_sector(ranges: list[float], centre_angle_deg: float,
                  width_deg: float, lidar_samples: int) -> float:
    """Return the minimum LiDAR range in a sector around centre_angle_deg."""
    n = lidar_samples
    centre_idx = int((centre_angle_deg % 360) / 360 * n)
    half = int(width_deg / 360 * n / 2)
    indices = [(centre_idx + i) % n for i in range(-half, half + 1)]
    vals = [ranges[i] for i in indices if 0 <= i < n]
    return min(vals) if vals else 2.5


def main() -> None:
    cfg = ConfigB()

    robot = Robot()
    TIME_STEP = int(robot.getBasicTimeStep())

    # Identify this robot
    args = robot.getControllerArguments()
    robot_id = args.split()[0] if args.strip() else "robot_1"
    print(f"[SwarmCtrl] Controller started for: {robot_id}")

    # ---- Devices ----
    lidar: Lidar = robot.getDevice("lidar")
    lidar.enable(TIME_STEP)

    gps: GPS = robot.getDevice("gps")
    gps.enable(TIME_STEP)

    compass: Compass = robot.getDevice("compass")
    compass.enable(TIME_STEP)

    emitter: Emitter = robot.getDevice("emitter")

    receiver: Receiver = robot.getDevice("receiver")
    receiver.enable(TIME_STEP)

    left_motor:  Motor = robot.getDevice("left_wheel_motor")
    right_motor: Motor = robot.getDevice("right_wheel_motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ---- Warm up ----
    robot.step(TIME_STEP)

    # ---- Initial pose ----
    gps_v = gps.getValues()
    com_v = compass.getValues()
    rx, ry = gps_v[0], gps_v[1]
    rtheta = compass_to_heading(com_v)

    # Determine consensus init value from robot index
    try:
        r_idx = cfg.ROBOT_IDS.index(robot_id)
    except ValueError:
        r_idx = 0
    consensus_init = cfg.CONSENSUS_INIT_VALUES[r_idx % len(cfg.CONSENSUS_INIT_VALUES)]

    agent = AgentState(
        agent_id=robot_id,
        x=rx, y=ry, heading=rtheta,
        consensus_init=consensus_init,
    )

    # ---- Load neural network ----
    model_path = cfg.MODELS_DIR / "numpy_mlp.npz"
    nn: NumpyMLP | None = None
    if model_path.exists():
        try:
            nn = NumpyMLP.load(model_path, cfg)
            print(f"[SwarmCtrl {robot_id}] Neural network loaded: {model_path}")
        except Exception as e:
            print(f"[SwarmCtrl {robot_id}] NN load failed: {e}, using rule-based fallback")
    else:
        print(f"[SwarmCtrl {robot_id}] NN model not found, using rule-based fallback")

    # ---- Assign target (PSO result or nearest unassigned) ----
    # Each robot reads a shared allocation JSON written by offline pipeline.
    # If unavailable, each robot takes its own index as target.
    alloc_path = cfg.RESULTS_METRICS / "component_b_metrics.json"
    target_idx = r_idx % len(cfg.TARGETS)  # default
    if alloc_path.exists():
        try:
            with open(alloc_path) as f:
                alloc_data = json.load(f)
            pso_assign = alloc_data["task_allocation"]["pso"]["assignment"]
            if r_idx < len(pso_assign):
                target_idx = pso_assign[r_idx]
        except Exception:
            pass

    agent.target_idx = target_idx
    tx, ty = cfg.TARGETS[target_idx]
    print(f"[SwarmCtrl {robot_id}] Assigned target {target_idx}: ({tx}, {ty})")

    # ---- Simulation log ----
    sim_log: list[dict] = []
    step_count = 0
    MAX_SPEED = cfg.MAX_WHEEL_SPEED
    LIDAR_N   = 180  # samples in the lidar (world file: horizontalResolution 180)

    # ---- Action → wheel speed map ----
    ACTIONS = {
        0: (MAX_SPEED * 0.6, MAX_SPEED * 0.6),   # forward
        1: (-MAX_SPEED * 0.3, MAX_SPEED * 0.5),  # turn_left
        2: (MAX_SPEED * 0.5, -MAX_SPEED * 0.3),  # turn_right
        3: (0.0, 0.0),                             # stop
    }

    while robot.step(TIME_STEP) != -1:
        t_s = step_count * TIME_STEP / 1000.0

        # ---- Read sensors ----
        gps_v = gps.getValues()
        com_v = compass.getValues()
        rx, ry  = gps_v[0], gps_v[1]
        rtheta  = compass_to_heading(com_v)
        agent.update_pose(rx, ry, rtheta)

        lidar_ranges = list(lidar.getRangeImage())
        front = _lidar_sector(lidar_ranges, 0,   20, LIDAR_N)
        left  = _lidar_sector(lidar_ranges, 90,  20, LIDAR_N)
        right = _lidar_sector(lidar_ranges, 270, 20, LIDAR_N)
        max_r = 2.5

        # ---- Check if target reached ----
        dist_to_target = agent.distance_to(tx, ty)
        if dist_to_target < cfg.GOAL_TOLERANCE:
            agent.target_reached = True
            print(f"[SwarmCtrl {robot_id}] Target {target_idx} reached!")
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            break

        # ---- Receive neighbour messages ----
        neighbours = get_ring_neighbours(robot_id, cfg.ROBOT_IDS)
        neighbour_consensus_vals: list[float] = []
        while receiver.getQueueLength() > 0:
            data = receiver.getBytes()
            msg = decode_message(data)
            if msg is not None and msg.sender_id in neighbours:
                agent.neighbour_msgs[msg.sender_id] = msg
                neighbour_consensus_vals.append(msg.consensus_val)
            receiver.nextPacket()

        # Remove stale messages
        stale = [k for k, v in agent.neighbour_msgs.items()
                 if t_s - v.timestamp > cfg.MSG_TIMEOUT_S]
        for k in stale:
            del agent.neighbour_msgs[k]

        # ---- Consensus update ----
        if neighbour_consensus_vals:
            agent.consensus_val = online_consensus_update(
                agent.consensus_val, neighbour_consensus_vals, cfg.CONSENSUS_ALPHA
            )

        # ---- Broadcast message ----
        msg_out = agent.build_message(t_s)
        emitter.send(encode_message(msg_out))

        # ---- Neural network action ----
        heading_err = agent.heading_error_to(tx, ty)
        goal_angle_norm  = heading_err / math.pi
        heading_err_norm = heading_err / math.pi

        features = [
            front  / max_r,
            left   / max_r,
            right  / max_r,
            goal_angle_norm,
            heading_err_norm,
        ]

        if nn is not None:
            action = nn.safe_predict_action(features, cfg.TEACHER_OBSTACLE_THRESHOLD / max_r)
        else:
            # Rule-based fallback
            ot = cfg.TEACHER_OBSTACLE_THRESHOLD
            tt = cfg.TEACHER_TURN_THRESHOLD
            if front < ot:
                action = 3
            elif front < tt:
                action = 1 if left >= right else 2
            else:
                action = 0

        agent.nn_action = action

        # ---- Motor commands ----
        omega_L, omega_R = ACTIONS[action]
        left_motor.setVelocity(omega_L)
        right_motor.setVelocity(omega_R)

        # ---- Log ----
        sim_log.append({
            "step": step_count,
            "t_s": t_s,
            "robot_id": robot_id,
            "x": rx, "y": ry, "theta": math.degrees(rtheta),
            "dist_to_target": dist_to_target,
            "front": front, "left": left, "right": right,
            "action": action,
            "consensus_val": agent.consensus_val,
        })
        step_count += 1

    # ---- Save log ----
    log_path = cfg.RESULTS_LOGS / f"webots_swarm_{robot_id}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        if sim_log:
            writer = csv.DictWriter(f, fieldnames=sim_log[0].keys())
            writer.writeheader()
            writer.writerows(sim_log)
    print(f"[SwarmCtrl {robot_id}] Log saved: {log_path}")


if __name__ == "__main__":
    main()
