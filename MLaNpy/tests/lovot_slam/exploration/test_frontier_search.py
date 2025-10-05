import pathlib

import numpy as np
import pytest

from grid_map_util.accuracy_map import AccuracyMap, logarithm_probability
from grid_map_util.occupancy_grid import OccupancyGrid

from lovot_slam.exploration.frontier_search import (LTM_FRONTIER_HISTORY_KEY, FrontierSearch,
                                                    mask_obstacle_with_accuracy_map)
from lovot_slam.redis import create_ltm_client
from MLaNpy.lovot_map.rosmap import RosMap

TEST_MAP_PATH = pathlib.Path('/tmp/test_map')
TEST_MAP_YAML_PATH = TEST_MAP_PATH / 'map.yaml'
DOOR_OPEN_ROSMAP = RosMap.from_hashed_values(
    (  # width, height, resolution, data, origin, name
        # "width":
        "31",
        # "height":
        "31",
        # "resolution":
        "0.05",
        # "data": upside down
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111122222222222222222222211111"
        "1111122222222222222222222211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"  # <- origin
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111111111111111111111111111111"  # <- frontier (y: -0.55)
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111",
        # "origin": center of the image
        "-0.75, -0.75, 0, 0, 0, 0, 1",
        # "name":
        "test_map",
    )
)
DOOR_CLOSE_ROSMAP = RosMap.from_hashed_values(
    (  # width, height, resolution, data, origin, name
        # "width":
        "31",
        # "height":
        "31",
        # "resolution":
        "0.05",
        # "data": upside down
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111122222222222222222222211111"
        "1111122222222222222222222211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122222222222222222222211111"
        "1111122222222222222222222211111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111",
        # "origin": center of the image
        "-0.75, -0.75, 0, 0, 0, 0, 1",
        # "name":
        "test_map",
    )
)
THREE_FRONTIERS_ROSMAP = RosMap.from_hashed_values(
    (  # width, height, resolution, data, origin, name
        # "width":
        "31",
        # "height":
        "31",
        # "resolution":
        "0.05",
        # "data": upside down
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111122111111122222222222211111"
        "1111122111111122222222222211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000001111111"
        "1111122000000000000000001111111"
        "1111122000000000000000001111111"
        "1111122000000000000000001111111"  # <- origin
        "1111122000000000000000001111111"
        "1111122000000000000000001111111"
        "1111122000000000000000001111111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111122000000000000000002211111"
        "1111111111111111111111111111111"  # <- frontier (y: -0.55)
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111"
        "1111111111111111111111111111111",
        # "origin": center of the image
        "-0.75, -0.75, 0, 0, 0, 0, 1",
        # "name":
        "test_map",
    )
)


def _decode_text_accuracy_map(text_map: str, width: int, height: int, origin: np.ndarray, resolution: float) -> AccuracyMap:
    data = [int(ch) / 10. for ch in list(text_map)]
    data = np.array(data).reshape((height, width))
    data = logarithm_probability(data)
    return AccuracyMap(data, origin, resolution)


# represents the first decimal place of the probability (0.0 ~ 0.9)
ACCURACY_MAP_DATA = _decode_text_accuracy_map(
    "5555555555555555555555555555555"
    "5555555555555555555555555555555"
    "5555555555555555555555555555555"
    "5555555555555555555555555555555"
    "5555555555555555555555555555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555777777777777777775555555"
    "5555555555555555555555555555555",
    31,
    31,
    np.array((-0.75, -0.75)),
    0.05
)


@pytest.fixture
def setup_ltm(monkeypatch):
    ltm_client = create_ltm_client()
    ltm_client.delete(LTM_FRONTIER_HISTORY_KEY)
    yield ltm_client
    ltm_client.delete(LTM_FRONTIER_HISTORY_KEY)


# update_historyを有効にすると、同じfrontierは3回目以降は無視される
# update_historyを無効にすると、3回目以降も同じfrontierが出続ける
@pytest.mark.parametrize("map,update_history,frontier_for_each_time", [
    (DOOR_OPEN_ROSMAP, True, [np.array((0.0, 0.55)), np.array((0.0, 0.55)), None, None]),
    (DOOR_OPEN_ROSMAP, False, [np.array((0.0, 0.55)), np.array((0.0, 0.55)),
                               np.array((0.0, 0.55)), np.array((0.0, 0.55))]),
    (DOOR_CLOSE_ROSMAP, True, [None, None, None, None]),
])
def test_get_frontier(setup_ltm, map, update_history, frontier_for_each_time):
    map.as_occupancy_grid().save(TEST_MAP_YAML_PATH)

    for frontier in frontier_for_each_time:
        frontier_search = FrontierSearch()
        grid_map = OccupancyGrid.from_yaml_file(TEST_MAP_YAML_PATH)
        start = np.array((0., 0.))
        obtained_frontier = frontier_search.find(grid_map, start, update_history=update_history)

        assert (np.all(np.isclose(frontier, obtained_frontier)) if frontier is not None
                else frontier == obtained_frontier)


@pytest.mark.parametrize("map,frontier", [
    (DOOR_OPEN_ROSMAP, np.array((0.0, 0.55))),
    (DOOR_CLOSE_ROSMAP, np.array((0.0, 0.55))),
])
def test_mask_obstacle_with_accuracy_map(setup_ltm, map, frontier):
    map.as_occupancy_grid().save(TEST_MAP_YAML_PATH)
    grid_map = OccupancyGrid.from_yaml_file(TEST_MAP_YAML_PATH)
    grid_map = mask_obstacle_with_accuracy_map(grid_map, ACCURACY_MAP_DATA)

    start = np.array((0., 0.))
    frontier_search = FrontierSearch()
    obtained_frontier = frontier_search.find(grid_map, start)

    assert (np.all(np.isclose(frontier, obtained_frontier)) if frontier is not None
            else frontier == obtained_frontier)


@pytest.mark.parametrize("map,start,frontier", [
    (THREE_FRONTIERS_ROSMAP, np.array((0.0, 0.0)), np.array((0.45, 0.0))),
    (THREE_FRONTIERS_ROSMAP, np.array((-0.25, -0.25)), np.array((-0.25, -0.55))),
    (THREE_FRONTIERS_ROSMAP, np.array((0.0, 0.35)), np.array((0.0, 0.55))),
])
def test_get_frontier_sort_by_distance(setup_ltm, map, start, frontier):
    map.as_occupancy_grid().save(TEST_MAP_YAML_PATH)

    frontier_search = FrontierSearch()
    grid_map = OccupancyGrid.from_yaml_file(TEST_MAP_YAML_PATH)
    obtained_frontier = frontier_search.find(grid_map, start, update_history=False)
    assert np.all(np.isclose(obtained_frontier, frontier))
