import numpy as np
import pytest

from lovot_slam.feature_map.feature_map_vertices import FeatureMapVertices, create_cost_map_from_vertices
from lovot_slam.feature_map.missions_filter import (
    _MissionsAdjacency, MissionsFilter, create_coverage_map, _check_connectivity_degradation_on_missions
)
from .feature_map_vertices_data import setup_dataset, teardown_dataset, MAPS_ROOT_PATH


CIRCLE_9_9 = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
])


def setup_module():
    setup_dataset()


def teardown_module():
    teardown_dataset()


def test_create_coverage_map():
    vertices_list = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / 'map_a')
    cost_map = create_cost_map_from_vertices([vertices_list], margin=0.5, resolution=0.05)
    coverage_map = create_coverage_map(vertices_list, cost_map)

    # something like this picture
    # ┌───────┐
    # │ * * * │
    # │     * │
    # └───────┘
    correct_map = np.zeros((30, 40))
    correct_map[6:15, 6:15] = CIRCLE_9_9
    correct_map[6:15, 16:25] = CIRCLE_9_9
    correct_map[6:15, 26:35] = CIRCLE_9_9
    correct_map[16:25, 26:35] = CIRCLE_9_9

    assert np.allclose(coverage_map, correct_map)


@pytest.mark.parametrize("mission_ids,adjacency_matrix,fully_connected", [
    (['0', '1', '2'],
     np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]),
     False),
    (['0', '1', '2'],
     np.array([[1, 1, 0],
               [1, 1, 0],
               [0, 0, 1]]),
     False),
    (['0', '1', '2'],
     np.array([[1, 1, 0],
               [1, 1, 1],
               [0, 1, 1]]),
     True),
])
def test_is_fully_connected(mission_ids, adjacency_matrix, fully_connected):
    adjacency = _MissionsAdjacency(mission_ids, adjacency_matrix)

    assert fully_connected == adjacency.is_fully_connected()


@pytest.mark.parametrize("mission_ids,adjacency_matrix,groups", [
    (['0', '1', '2'],
     np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]),
     {frozenset(('0')), frozenset(('1')), frozenset(('2'))}),
    (['0', '1', '2'],
     np.array([[1, 1, 0],
               [1, 1, 0],
               [0, 0, 1]]),
     {frozenset(('0', '1')), frozenset(('2'))}),
    (['0', '1', '2'],
     np.array([[1, 1, 0],
               [1, 1, 1],
               [0, 1, 1]]),
     {frozenset(('0', '1', '2'))}),
])
def test_obtain_connected_groups(mission_ids, adjacency_matrix, groups):
    adjacency = _MissionsAdjacency(mission_ids, adjacency_matrix)
    obtained_groups = adjacency.obtain_connected_groups()

    assert obtained_groups == groups


@pytest.mark.parametrize("mission_ids,adjacency_matrix,missions_to_remove,connectivity", [
    (['0', '1', '2', '3'],
     np.array([[1, 1, 0, 0],  # full connected
               [1, 1, 1, 0],
               [0, 1, 1, 1],
               [0, 0, 1, 1]]),
     ['0'],  # groups after filtering: [1, 2, 3]
     True),
    (['0', '1', '2', '3'],
     np.array([[1, 1, 0, 0],  # full connected
               [1, 1, 1, 0],
               [0, 1, 1, 1],
               [0, 0, 1, 1]]),
     ['1'],  # groups after filtering: [0], [2, 3]
     False),
    (['0', '1', '2', '3'],
     np.array([[1, 1, 0, 0],  # [0, 1, 2], [3]
               [1, 1, 1, 0],
               [0, 1, 1, 0],
               [0, 0, 0, 1]]),
     ['0'],  # groups after filtering: [1, 2], [3]
     True),
    (['0', '1', '2', '3'],
     np.array([[1, 1, 0, 0],  # [0, 1, 2], [3]
               [1, 1, 1, 0],
               [0, 1, 1, 0],
               [0, 0, 0, 1]]),
     ['1'],  # groups after filtering: [0], [2], [3]
     False),
    (['0', '1', '2', '3'],
     np.array([[1, 1, 0, 0],  # [0, 1, 2], [3]
               [1, 1, 1, 0],
               [0, 1, 1, 0],
               [0, 0, 0, 1]]),
     ['3'],  # groups after filtering: [0, 1, 2]
     True),
])
def test_check_connectivity_degradation_on_missions(mission_ids, adjacency_matrix, missions_to_remove, connectivity):
    adjacency = _MissionsAdjacency(mission_ids, adjacency_matrix)
    obtained_groups = adjacency.obtain_connected_groups()
    obtained_connectivity = _check_connectivity_degradation_on_missions(obtained_groups, adjacency, missions_to_remove)

    assert obtained_connectivity == connectivity


# map_d has three missions and all missions have the same trajectories,
# so the old mission (01) could be removed depending on the connectivity.
# map_e has three missions and all missions have the difference trajectories.
@pytest.mark.parametrize("map_name,adjacency_str,missions_to_remove", [
    ('map_d',  # full connected case
     "01, 0., 1., 1.\n"
     "02, 1., 0., 1.\n"
     "03, 1., 1., 0.\n",
     ["01"]),
    ('map_d',  # 02 is only connected to 01, so that 01 cannot be removed
     "01, 0., 1., 1.\n"
     "02, 1., 0., 0.\n"
     "03, 1., 0., 0.\n",
     []),
    ('map_e',  # full connected case
     "01, 0., 1., 1.\n"
     "02, 1., 0., 1.\n"
     "03, 1., 1., 0.\n",
     []),
])
def test_missions_filter_by_overlapping(map_name, adjacency_str, missions_to_remove):
    map_path = MAPS_ROOT_PATH / map_name
    with open(map_path / 'feature_map' / 'adjacency.csv', 'w') as f:
        f.write(adjacency_str)

    missions_filter = MissionsFilter.create_from_map_path(map_path)
    obtained_missions_to_remove = missions_filter.filter_by_overlapping()

    assert obtained_missions_to_remove == missions_to_remove


@pytest.mark.parametrize("map_name,max_merge_missions_count,missions_to_remove_in,missions_to_remove", [
    ('map_d', # 3 missions included
    1, # must remove 2 missions
    [],
    ['02', '03']),
    ('map_f',   # 3 missions included
    1, # must remove 2 missions
    [],
    ['01', '03']),
    ('map_d',   # 3 missions included
    2, # must remove 1 mission
    [],
     ['03']),
    ('map_f',   # 3 missions included
    2, # must remove 1 mission
    [],
     ['01']),
    ('map_d',   # 3 missions included
    3, # no missions to remove
    [],
     []),
    ('map_f',   # 3 missions included
    3, # no missions to remove
    [],
     []),
    ('map_f',   # 3 missions included
    1, # must remove additional 1 mission
    ['02'],
    ['02', '01']),
    ('map_f',   # 3 missions included
    None, # limited by default value (30 missions)
    ['02'],
    ['02']),
])
def test_missions_filter_by_count(map_name, max_merge_missions_count, missions_to_remove_in, missions_to_remove):
    map_path = MAPS_ROOT_PATH / map_name

    adjacency_str = "01, 0., 1., 1.\n" + "02, 1., 0., 1.\n" + "03, 1., 1., 0.\n"
    with open(map_path / 'feature_map' / 'adjacency.csv', 'w') as f:
        f.write(adjacency_str)

    missions_filter = MissionsFilter.create_from_map_path(map_path)
    obtained_missions_to_remove = missions_filter.filter_by_count(
        missions_to_remove=missions_to_remove_in,
        max_merge_missions_count=max_merge_missions_count)

    assert obtained_missions_to_remove == missions_to_remove
