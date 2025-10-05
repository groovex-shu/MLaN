import numpy as np
import pytest

from grid_map_util.occupancy_grid import OccupancyGrid
from grid_map_util.planner import FrontierFinder


def dec(d):
    if d == 2:  # Obstacle
        return 0
    elif d == 1:  # Unknown
        return 205
    else:  # No obstacles
        return 254


def _decode_text_occupancy_grid(text_map: str, width: int, height: int,
                                origin: np.ndarray, resolution: float) -> OccupancyGrid:
    img = [dec(int(ch)) for ch in list(text_map)]
    img = np.array(img).astype(np.uint8).reshape((height, width))
    return OccupancyGrid(img, resolution, origin)


@pytest.mark.parametrize('grid_map,correct_frontier_list', [
    (_decode_text_occupancy_grid(
        "11111"  # no frontiers
        "22222"
        "20002"
        "20002"
        "22222",
        5, 5, np.array((-0.10, -0.10)), 0.05),
     []),
    (_decode_text_occupancy_grid(
        "11111"  # <- frontier
        "20002"
        "20002"
        "20002"
        "22222",
        5, 5, np.array((-0.10, -0.10)), 0.05),
     [[np.array([0.05, 0.10]), np.array([0.0, 0.10]), np.array([-0.05, 0.10])]]),
    (_decode_text_occupancy_grid(
        "11111"  # <- frontier
        "20002"
        "20002"
        "20002"
        "11111",  # <- frontier
        5, 5, np.array((-0.10, -0.10)), 0.05),
     [[np.array([0.05, 0.10]), np.array([0.0, 0.10]), np.array([-0.05, 0.10])],
      [np.array([0.05, -0.10]), np.array([0.0, -0.10]), np.array([-0.05, -0.10])]]),
])
def test_frontier_finder(grid_map, correct_frontier_list):
    frontier_finder = FrontierFinder(grid_map)
    start = np.array([0., 0.])
    frontier_list = frontier_finder.find(start)

    if not correct_frontier_list:
        assert correct_frontier_list == frontier_list
        return

    frontier_list = np.sort(np.array(frontier_list), axis=1)
    frontier_list = np.sort(np.array(frontier_list), axis=0)

    correct_frontier_list = np.sort(np.array(correct_frontier_list), axis=1)
    correct_frontier_list = np.sort(np.array(correct_frontier_list), axis=0)

    assert np.array_equal(frontier_list, correct_frontier_list)
