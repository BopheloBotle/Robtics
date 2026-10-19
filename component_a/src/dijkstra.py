"""
dijkstra.py
===========
Dijkstra's shortest-path algorithm on a 2-D occupancy grid.

Movement model
--------------
8-connected (allows diagonal moves).  Diagonal moves have cost √2 × cell_size;
cardinal moves have cost 1 × cell_size.  Corner-cutting is prevented: a
diagonal step is blocked if either of the two adjacent cardinal cells is
occupied.

Returns
-------
dijkstra(grid, start, goal, cfg) → PathResult

PathResult fields:
  path          : list of (row, col) from start to goal (inclusive), or []
  nodes_explored: number of cells popped from the priority queue
  elapsed_s     : wall-clock planning time in seconds
  path_length_m : Euclidean path length in metres (0 if no path)
  success       : bool
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from occupancy_grid import OccupancyGrid
    from configuration import Config


@dataclass
class PathResult:
    """Container for pathfinding output."""
    path: list[tuple[int, int]] = field(default_factory=list)
    nodes_explored: int = 0
    elapsed_s: float = 0.0
    path_length_m: float = 0.0
    success: bool = False
    algorithm: str = "unknown"


# Cardinal and diagonal neighbours with their base costs
_MOVES = [
    (-1,  0, 1.0), ( 1,  0, 1.0),  # N, S
    ( 0, -1, 1.0), ( 0,  1, 1.0),  # W, E
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),  # NW, NE
    ( 1, -1, math.sqrt(2)), ( 1, 1, math.sqrt(2)),  # SW, SE
]


def dijkstra(
    grid: "OccupancyGrid",
    start: tuple[int, int],
    goal: tuple[int, int],
    cfg: "Config",
) -> PathResult:
    """
    Run Dijkstra's algorithm from *start* to *goal* on *grid*.

    Parameters
    ----------
    grid  : OccupancyGrid instance (already built)
    start : (row, col) start cell
    goal  : (row, col) goal cell
    cfg   : Config instance (used for cell state constants and resolution)

    Returns
    -------
    PathResult with path, nodes_explored, elapsed_s, path_length_m, success.
    """
    t0 = time.perf_counter()
    res = cfg.GRID_RESOLUTION

    # Validate start and goal
    if grid.is_occupied(*start):
        return PathResult(algorithm="Dijkstra",
                          elapsed_s=time.perf_counter() - t0)
    if grid.is_occupied(*goal):
        return PathResult(algorithm="Dijkstra",
                          elapsed_s=time.perf_counter() - t0)

    # dist[r][c] = best known cost from start
    INF = float("inf")
    dist: dict[tuple[int, int], float] = {}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}

    dist[start] = 0.0
    prev[start] = None

    # Priority queue: (cost, row, col)
    pq: list[tuple[float, int, int]] = []
    heapq.heappush(pq, (0.0, start[0], start[1]))
    nodes_explored = 0

    while pq:
        cost, r, c = heapq.heappop(pq)
        nodes_explored += 1

        if (r, c) == goal:
            break

        if cost > dist.get((r, c), INF):
            continue  # stale entry

        for dr, dc, step_cost in _MOVES:
            nr, nc = r + dr, c + dc
            if not grid.is_in_bounds(nr, nc):
                continue
            if grid.is_occupied(nr, nc):
                continue
            # Prevent diagonal corner-cutting
            if abs(dr) == 1 and abs(dc) == 1:
                if grid.is_occupied(r + dr, c) or grid.is_occupied(r, c + dc):
                    continue

            new_cost = cost + step_cost * res
            if new_cost < dist.get((nr, nc), INF):
                dist[(nr, nc)] = new_cost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_cost, nr, nc))

    elapsed = time.perf_counter() - t0

    # Reconstruct path
    if goal not in dist or dist[goal] == INF:
        return PathResult(
            algorithm="Dijkstra",
            nodes_explored=nodes_explored,
            elapsed_s=elapsed,
        )

    path = _reconstruct_path(prev, start, goal)
    length = _path_length(path, res)
    return PathResult(
        path=path,
        nodes_explored=nodes_explored,
        elapsed_s=elapsed,
        path_length_m=length,
        success=True,
        algorithm="Dijkstra",
    )


def _reconstruct_path(
    prev: dict[tuple[int, int], tuple[int, int] | None],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path


def _path_length(path: list[tuple[int, int]], resolution: float) -> float:
    """Compute the total Euclidean length of a grid path in metres."""
    length = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        length += math.hypot(dr, dc) * resolution
    return length
