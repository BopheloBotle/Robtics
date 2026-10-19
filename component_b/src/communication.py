"""
communication.py
================
Inter-agent communication for Component B.

Webots path
-----------
The Webots controller uses Emitter and Receiver devices.  Each robot
broadcasts a JSON-encoded AgentMessage on channel 1, and receives
messages from all other robots on the same channel.

Offline path
------------
For offline experiments, a LocalBus simulates the broadcast medium:
all agents share the same LocalBus instance.  Each agent calls
``send()`` then all agents call ``receive()`` to collect pending messages.

Message encoding
----------------
AgentMessage is serialised to/from a compact JSON byte string so that
the same format works with both Webots Emitter/Receiver (which accept
bytes) and the offline LocalBus (which stores Python strings).

Stale message handling
----------------------
Messages older than MSG_TIMEOUT_S are discarded during receive().

Topology: ring
--------------
In a ring topology each agent communicates with the two agents whose
IDs are immediately adjacent in the sorted ID list.  The topology is
enforced at the receive stage: messages from non-neighbours are ignored.
This is consistent with the assignment requirement of a neighbour-based
communication graph.

Limitations documented
----------------------
- Webots Emitter/Receiver broadcasts to ALL robots on the channel.
  Topology filtering is implemented in software at the receiver, not
  in hardware.  This is standard practice for simulated swarms.
- No packet loss or latency is modelled.
- No encryption or authentication is implemented.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from agents import AgentMessage

if TYPE_CHECKING:
    from configuration_b import ConfigB


def encode_message(msg: AgentMessage) -> bytes:
    """Serialise an AgentMessage to UTF-8 JSON bytes."""
    d = {
        "sid": msg.sender_id,
        "x": round(msg.x, 4),
        "y": round(msg.y, 4),
        "h": round(msg.heading, 4),
        "ti": msg.target_idx,
        "cv": round(msg.consensus_val, 6),
        "ts": round(msg.timestamp, 3),
    }
    return json.dumps(d).encode("utf-8")


def decode_message(data: bytes | str) -> AgentMessage | None:
    """Deserialise bytes or str to an AgentMessage.  Returns None on error."""
    try:
        if isinstance(data, (bytes, bytearray)):
            d = json.loads(data.decode("utf-8"))
        else:
            d = json.loads(data)
        return AgentMessage(
            sender_id=d["sid"],
            x=d["x"], y=d["y"], heading=d["h"],
            target_idx=d["ti"],
            consensus_val=d["cv"],
            timestamp=d["ts"],
        )
    except (KeyError, json.JSONDecodeError, ValueError):
        return None


def get_ring_neighbours(agent_id: str, all_ids: list[str]) -> list[str]:
    """
    Return the two ring neighbours of agent_id in a sorted ID list.

    For a 3-agent ring: robot_1 ↔ robot_2 ↔ robot_3 ↔ robot_1
    """
    sorted_ids = sorted(all_ids)
    n = len(sorted_ids)
    if n < 2:
        return []
    idx = sorted_ids.index(agent_id)
    left  = sorted_ids[(idx - 1) % n]
    right = sorted_ids[(idx + 1) % n]
    return [left, right]


class LocalBus:
    """
    Simulated broadcast medium for offline experiments.

    Usage
    -----
    bus = LocalBus()
    bus.send(msg)                # any agent posts a message
    msgs = bus.receive(agent_id, current_time, cfg)   # agent collects messages
    bus.flush()                  # clear all pending messages (end of step)
    """

    def __init__(self) -> None:
        self._pending: list[AgentMessage] = []

    def send(self, msg: AgentMessage) -> None:
        self._pending.append(msg)

    def receive(
        self,
        agent_id: str,
        current_time: float,
        cfg: "ConfigB",
    ) -> list[AgentMessage]:
        """
        Collect messages for agent_id.

        Filters:
        - Excludes own messages.
        - Enforces ring-topology (only ring neighbours pass through).
        - Discards stale messages (timestamp > MSG_TIMEOUT_S ago).
        """
        neighbours = get_ring_neighbours(agent_id, cfg.ROBOT_IDS)
        result: list[AgentMessage] = []
        for msg in self._pending:
            if msg.sender_id == agent_id:
                continue
            if msg.sender_id not in neighbours:
                continue
            if current_time - msg.timestamp > cfg.MSG_TIMEOUT_S:
                continue
            result.append(msg)
        return result

    def flush(self) -> None:
        """Clear all pending messages (call once per simulation step)."""
        self._pending.clear()
