"""
astar.py
========
A* shortest-path algorithm on a 2-D occupancy grid.

Uses the same 8-connected movement model and corner-cutting prevention as
dijkstra.py.  The heuristic is octile distance, which is admissible and
consistent for 8-connected grids with diagonal cost √2:

  h(n, goal) = resolution × (|Δr| + |Δc| + (√2 - 2) × min(|Δr|, |Δc|))

Because A* with an admissible heuristic is optimally efficient over the
explored node set, it will find the same optimal path as Dijkstra while
typically expanding fewer nodes.

Returns a PathResult (same dataclass as dijkstra.py).
"""

from __future__ import annotations

import heapq
import math
import time
from typing import TYPE_CHECKING

from dijkstra import PathResult, _reconstruct_path, _path_length, _MOVES

if TYPE_CHECKING:
    from occupancy_grid import OccupancyGrid
    from configuration import Config


def _octile_heuristic(
    r: int, c: int, gr: int, gc: int, resolution: float
) -> float:
    """Octile distance heuristic — admissible for 8-connected movement."""
    dr = abs(r - gr)
    dc = abs(c - gc)
    return resolution * (dr + dc + (math.sqrt(2) - 2.0) * min(dr, dc))


def astar(
    grid: "OccupancyGrid",
    start: tuple[int, int],
    goal: tuple[int, int],
    cfg: "Config",
) -> PathResult:
    """
    Run A* from *start* to *goal* on *grid*.

    Parameters
    ----------
    grid  : OccupancyGrid instance (already built)
    start : (row, col) start cell
    goal  : (row, col) goal cell
    cfg   : Config instance

    Returns
    -------
    PathResult with path, nodes_explored, elapsed_s, path_length_m, success.
    """
    t0 = time.perf_counter()
    res = cfg.GRID_RESOLUTION
    gr, gc = goal

    if grid.is_occupied(*start):
        return PathResult(algorithm="A*", elapsed_s=time.perf_counter() - t0)
    if grid.is_occupied(*goal):
        return PathResult(algorithm="A*", elapsed_s=time.perf_counter() - t0)

    INF = float("inf")
    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    # Priority queue: (f = g + h, g, row, col)
    pq: list[tuple[float, float, int, int]] = []
    h0 = _octile_heuristic(start[0], start[1], gr, gc, res)
    heapq.heappush(pq, (h0, 0.0, start[0], start[1]))
    closed: set[tuple[int, int]] = set()
    nodes_explored = 0

    while pq:
        f, g, r, c = heapq.heappop(pq)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        nodes_explored += 1

        if (r, c) == goal:
            break

        for dr, dc, step_cost in _MOVES:
            nr, nc = r + dr, c + dc
            if not grid.is_in_bounds(nr, nc):
                continue
            if grid.is_occupied(nr, nc):
                continue
            if (nr, nc) in closed:
                continue
            # Prevent diagonal corner-cutting
            if abs(dr) == 1 and abs(dc) == 1:
                if grid.is_occupied(r + dr, c) or grid.is_occupied(r, c + dc):
                    continue

            new_g = g + step_cost * res
            if new_g < g_cost.get((nr, nc), INF):
                g_cost[(nr, nc)] = new_g
                prev[(nr, nc)] = (r, c)
                h = _octile_heuristic(nr, nc, gr, gc, res)
                heapq.heappush(pq, (new_g + h, new_g, nr, nc))

    elapsed = time.perf_counter() - t0

    if goal not in g_cost:
        return PathResult(
            algorithm="A*",
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
        algorithm="A*",
    )
