"""
agents.py
=========
Agent state representation for Component B swarm robots.

Each agent has:
  - A unique string identifier (e.g. "robot_1").
  - A 2-D world-frame position and heading.
  - A local target assignment (index into the global targets list, or -1).
  - A local consensus state value.
  - A shared-state dict for information broadcast to neighbours.
  - A history log for evaluation.

The AgentState class is used both by the Webots controller (real sensor
values) and by the offline experiments (simulated values).  It contains
no Webots API calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """
    Message broadcast by one agent to its neighbours.

    Fields
    ------
    sender_id    : identifier of the sending agent
    x, y         : sender's world position
    heading      : sender's heading (radians)
    target_idx   : currently assigned target index (-1 = unassigned)
    consensus_val: agent's current consensus state value
    timestamp    : simulation time when the message was created (seconds)
    """
    sender_id: str
    x: float
    y: float
    heading: float
    target_idx: int
    consensus_val: float
    timestamp: float


class AgentState:
    """
    Mutable state for one swarm agent.

    All navigation, communication, and decision-making modules update
    or read this object.
    """

    def __init__(
        self,
        agent_id: str,
        x: float,
        y: float,
        heading: float,
        consensus_init: float,
    ) -> None:
        self.agent_id: str = agent_id
        # Pose
        self.x: float = x
        self.y: float = y
        self.heading: float = heading
        # Navigation
        self.target_idx: int = -1          # assigned target index
        self.target_reached: bool = False
        # Consensus
        self.consensus_val: float = consensus_init
        # Neighbour information: dict[sender_id → AgentMessage]
        self.neighbour_msgs: dict[str, AgentMessage] = {}
        # Neural network action: 0=forward, 1=left, 2=right, 3=stop
        self.nn_action: int = 0
        # History log (one entry per step)
        self.log: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Pose update                                                         #
    # ------------------------------------------------------------------ #

    def update_pose(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading

    # ------------------------------------------------------------------ #
    #  Outgoing message                                                    #
    # ------------------------------------------------------------------ #

    def build_message(self, timestamp: float) -> AgentMessage:
        """Build the broadcast message for this time step."""
        return AgentMessage(
            sender_id=self.agent_id,
            x=self.x,
            y=self.y,
            heading=self.heading,
            target_idx=self.target_idx,
            consensus_val=self.consensus_val,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    def distance_to(self, wx: float, wy: float) -> float:
        return math.hypot(wx - self.x, wy - self.y)

    def heading_error_to(self, wx: float, wy: float) -> float:
        desired = math.atan2(wy - self.y, wx - self.x)
        err = desired - self.heading
        while err > math.pi:
            err -= 2.0 * math.pi
        while err < -math.pi:
            err += 2.0 * math.pi
        return err

    def log_step(self, t: float, extra: dict | None = None) -> None:
        entry: dict[str, Any] = {
            "t": t,
            "agent_id": self.agent_id,
            "x": self.x, "y": self.y,
            "heading": self.heading,
            "target_idx": self.target_idx,
            "consensus_val": self.consensus_val,
            "nn_action": self.nn_action,
        }
        if extra:
            entry.update(extra)
        self.log.append(entry)
