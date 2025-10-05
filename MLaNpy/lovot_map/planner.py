import math
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
from attr import attrs

from lovot_map.occupancy_grid import OccupancyGrid

logger = getLogger(__name__)


@attrs(auto_attribs=True)
class _Node:
    x: int
    y: int
    cost: float  # cost of this node
    prev_idx: int  # previous node index


@attrs(auto_attribs=True, frozen=True)
class _MotionCost:
    x: int
    y: int
    cost: float


class FrontierFinder:
    """
    Dijkstra based frontier finder.
    based on 'A Frontier-Based Approach for Autonoumous Exploration'
    http://www.robotfrontier.com/papers/cira97.pdf

    frontier is the border between free cells (floor) and unknown cells (unobserved area).
    """
    _MOTION_MODEL: Tuple[_MotionCost, ...] = (
        _MotionCost(1, 0, 1),
        _MotionCost(0, 1, 1),
        _MotionCost(-1, 0, 1),
        _MotionCost(0, -1, 1),
        _MotionCost(-1, -1, math.sqrt(2)),
        _MotionCost(-1, 1, math.sqrt(2)),
        _MotionCost(1, -1, math.sqrt(2)),
        _MotionCost(1, 1, math.sqrt(2)))

    def __init__(self, grid_map: OccupancyGrid) -> None:
        self._grid_map = grid_map

        self._openset: Optional[Dict[int, _Node]] = None
        self._closedset: Optional[Dict[int, _Node]] = None

    def _verify_node(self, node: _Node, prohibited_values: List[int] = [OccupancyGrid.OCCUPIED_CODE]) -> bool:
        # check the coordinate is within the map
        if node.x < 0 or node.y < 0 \
                or node.x >= self._grid_map.img.shape[1] or node.y >= self._grid_map.img.shape[0]:
            return False

        # check whether the node has prohibited value
        if self._grid_map.img[node.y, node.x] in prohibited_values:
            return False

        return True

    def _get_index(self, node: _Node) -> int:
        return node.y * self._grid_map.img.shape[1] + node.x

    def _search_neighbors(self, current_node: _Node, current_idx: int) -> None:
        assert self._openset is not None
        assert self._closedset is not None

        for motion in self._MOTION_MODEL:
            node = _Node(current_node.x + motion.x,
                         current_node.y + motion.y,
                         current_node.cost + motion.cost,
                         current_idx)

            if not self._verify_node(node):
                continue

            next_idx = self._get_index(node)
            if next_idx in self._closedset:
                continue

            if next_idx not in self._openset:
                self._openset[next_idx] = node  # Discover a new node
            else:
                if self._openset[next_idx].cost >= node.cost:
                    # This path is the best until now. record it!
                    self._openset[next_idx] = node

    def _reset(self, node: _Node) -> None:
        self._openset = {}
        self._closedset = {}
        self._openset[self._get_index(node)] = node

    def _is_frontier_node(self, node: _Node) -> bool:
        """Check whether the given node is a part of a frontier or not.
        A frontier node is an unknown node which faces another free node.
        """
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]
        if self._grid_map.img[node.y, node.x] != OccupancyGrid.UNKNOWN_CODE:
            return False
        for m in motion:
            neighbor = _Node(node.x + m[0],
                             node.y + m[1],
                             node.cost + m[2], 0)
            if not self._verify_node(neighbor):
                continue
            if self._grid_map.img[neighbor.y, neighbor.x] == OccupancyGrid.FREE_CODE:
                return True
        return False

    def _build_frontier(self, node: _Node, current_idx: int) -> List[_Node]:
        """Build a frontier from the given node which is a part of the frontier.
        Recursively searching the neighbor nodes and collect them to build a frontier.
        """
        node_list: List[_Node] = []
        self._closedset[current_idx] = node
        node_list.append(node)

        for motion in self._MOTION_MODEL:
            neighbor = _Node(node.x + motion.x,
                             node.y + motion.y,
                             node.cost + motion.cost,
                             current_idx)

            next_idx = self._get_index(neighbor)

            if (not self._verify_node(neighbor)) or (next_idx in self._closedset):
                continue

            if self._is_frontier_node(neighbor):
                if next_idx not in self._openset:
                    self._openset[next_idx] = neighbor
                node_list = node_list + self._build_frontier(neighbor, next_idx)
        return node_list

    def _find_frontiers(self, start_node: _Node) -> List[List[_Node]]:
        self._reset(start_node)

        frontier_nodes_list: List[List[_Node]] = []
        while 1:
            if len(self._openset) == 0:
                logger.debug('find_frontiers: no openset nodes.')
                break
            # Dijkstra
            current_idx = min(self._openset, key=lambda o: self._openset[o].cost)
            current = self._openset[current_idx]

            # check frontier
            if self._is_frontier_node(current) and current_idx not in self._closedset:
                # build frontier area around current node
                frontier_nodes = self._build_frontier(current, current_idx)
                frontier_nodes_list.append(frontier_nodes)

            # Remove the item from the open set
            del self._openset[current_idx]
            # Add it to the closed set
            self._closedset[current_idx] = current

            if self._grid_map.img[current.y, current.x] == OccupancyGrid.UNKNOWN_CODE:
                continue

            # expand search grid based on motion model
            self._search_neighbors(current, current_idx)

        return frontier_nodes_list

    def find(self, start: np.ndarray) -> List[List[np.ndarray]]:
        """Find frontiers.
        :param start: start position of ndarray (x [m], y [m])
        :return: list of real world coordinates (int meter) of frontiers
        """
        logger.debug('find_frontiers: given cell {}'.format(start))
        start = self._grid_map.get_nearest_free_cell(start)
        logger.debug('find_frontiers: start cell map coords {}'.format(start))
        if start is None:
            logger.warning('find_frontiers: failed to find frontier (start position is None)')
            return []

        start_pos = self._grid_map.realcoords_to_npcoords(start)
        start_node = _Node(start_pos[1], start_pos[0], 0.0, -1)

        if not self._verify_node(start_node):
            logger.warning(
                'find_frontiers: failed to find frontier (start position is invalid)')
            return []

        frontier_nodes_list = self._find_frontiers(start_node)
        frontier_list = [[self._grid_map.npcoords_to_realcoords(np.array([node.y, node.x])) for node in nodes]
                         for nodes in frontier_nodes_list]

        return frontier_list
