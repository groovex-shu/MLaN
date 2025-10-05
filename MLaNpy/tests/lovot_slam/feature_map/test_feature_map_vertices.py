import numpy as np
import pytest

from lovot_slam.feature_map.feature_map_vertices import (
    FeatureMapVertices,
    create_cost_map_from_vertices,
    find_common_mission_ids,
    rotation_matrix_from_pose,
    look_up_transform,
    transform,
    transform_pose_between_maps,
    transform_points_between_maps,
)

from .feature_map_vertices_data import setup_dataset, teardown_dataset, MAPS_ROOT_PATH


def setup_module():
    setup_dataset()


def teardown_module():
    teardown_dataset()


@pytest.mark.parametrize("map_name,shape", [
    ('map_a', (5, 7)),
    ('map_b', (10, 7)),
    ('map_c', (15, 7)),
])
def test_from_map_path(map_name, shape):
    # this test is bad (not checking contents)
    vertices = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    assert vertices.vertices.shape == shape


@pytest.mark.parametrize("missions_to_remain,", [
    (['28e237ce950ed6150e00000000000000', '35db4906360cd6150e00000000000000', '5097404d4d0dd6150e00000000000000']),
    (['35db4906360cd6150e00000000000000', '5097404d4d0dd6150e00000000000000']),
    (['35db4906360cd6150e00000000000000']),
    ([]),
])
def test_feature_map_vertices_filter_missions(missions_to_remain):
    map_name = 'map_c'
    original = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    filtered = FeatureMapVertices.filter_missions(original, missions_to_remain)

    assert filtered.missions == missions_to_remain

    for mission_id in missions_to_remain:
        assert np.all(np.isclose(original.vertices_of(mission_id),
                                 filtered.vertices_of(mission_id)))


@pytest.mark.parametrize("map_name,shape", [
    ('map_a', (5, 7)),
    ('map_b', (10, 7)),
    ('map_c', (15, 7)),
])
def test_get_number_of_vertices(map_name, shape):
    vertices = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    assert vertices.get_number_of_vertices() == shape[0]


@pytest.mark.parametrize("map_name,average,stdev", [
    ('map_a', 0.0, 0.0),
    ('map_b', 0.0, 0.0),
    ('map_c', 0.4, 0.879393730551528),
])
def test_get_height_statistics(map_name, average, stdev):
    vertices = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    assert np.isclose(vertices.get_height_statistics()[0], average)
    assert np.isclose(vertices.get_height_statistics()[1], stdev)


@pytest.mark.parametrize("map_name,boundary", [
    ('map_a', np.array([0.0, 0.0, 1.0, 0.5])),
    ('map_b', np.array([0.0, 2.0, 11.0, 3.0])),
    ('map_c', np.array([0.0, 0.0, 11.0, 10.5])),
])
def test_get_boundary(map_name, boundary):
    vertices = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    assert np.allclose(vertices.get_boundary(), boundary)


@pytest.mark.parametrize("pose,rotation_matrix", [
    (np.array([0, 0, 0, 0, 0, 0, 1]),
     np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])),
    (np.array([0, 0, 0, 0.4469983, 0.4469983, 0.7240368, 0.2759632]),
     np.array([[-0.4480736, -0.0000000, 0.8939967],
               [0.7992300, -0.4480736, 0.4005763],
               [0.4005763, 0.8939967, 0.2007700]])),
])
def test_rotation_matrix_from_pose(pose, rotation_matrix):
    _rotation_matrix = rotation_matrix_from_pose(pose)
    assert np.allclose(rotation_matrix, _rotation_matrix, atol=1e-05)


@pytest.mark.parametrize("pose,base,rot,trans", [
    (np.array([0, 0, 0, 0, 0, 0, 1]),
     np.array([0, 0, 0, 0, 0, 0, 1]),
     np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]),
     np.array([0, 0, 0])),
    (np.array([1, 2, 0, 0, 0, -0.3826834, 0.9238795]),
     np.array([1, 1, 0, 0, 0, 0.3826834, 0.9238795]),
     np.array([[0, 1, 0],
               [-1, 0, 0],
               [0, 0, 1]]),
     np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])),
])
def test_look_up_transform(pose, base, rot, trans):
    _rot, _trans = look_up_transform(pose, base)
    assert np.allclose(rot, _rot, atol=1e-05)
    assert np.allclose(trans, _trans, atol=1e-05)


@pytest.mark.parametrize("pose,rot,trans,transformed", [
    (np.array([0, 0, 0, 0, 0, 0, 1]),
     np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]),
     np.array([0, 0, 0]),
     np.array([0, 0, 0, 0, 0, 0, 1])),
    (np.array([1, 1, 0, 0, 0, 0.3826834, 0.9238795]),
     np.array([[0, 1, 0],
               [-1, 0, 0],
               [0, 0, 1]]),
     np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
     np.array([1, 2, 0, 0, 0, -0.3826834, 0.9238795])),
])
def test_transform(pose, rot, trans, transformed):
    _transformed = transform(pose, rot, trans)

    assert np.allclose(_transformed, transformed, atol=1e-05)


@pytest.mark.parametrize("missions_to_remain,", [
    (['28e237ce950ed6150e00000000000000', '35db4906360cd6150e00000000000000', '5097404d4d0dd6150e00000000000000']),
    (['35db4906360cd6150e00000000000000', '5097404d4d0dd6150e00000000000000']),
    (['35db4906360cd6150e00000000000000']),
])
def test_find_common_mission_ids(missions_to_remain):
    map_name = 'map_c'
    original = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    filtered = FeatureMapVertices.filter_missions(original, missions_to_remain)

    common_missions = find_common_mission_ids(original, filtered)

    assert set(common_missions) == set(missions_to_remain)


@pytest.mark.parametrize("orig_map_name,orig_pose,dest_map_name,dest_pose", [
    ('map_a', np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
     'map_a', np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
    ('map_a', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795]),
     'map_b', np.array([0.9, 3.1, 0.0, 0.0, 0.0, 0.9238795, 0.3826834])),
    ('map_b', np.array([0.9, 3.1, 0.0, 0.0, 0.0, 0.9238795, 0.3826834]),
     'map_c', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795])),
    ('map_a', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795]),
     'map_c', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795])),
    # the closes key frame in map_c is not in map_b
    ('map_c', np.array([11.0, 10.0, 0.0, 0.0, 0.0, 0, 1.0]),
     'map_b', np.array([11.0, 13.0, 0.0, 0.0, 0.0, 0.70710678, 0.70710678])),
])
def test_transform_pose_between_maps(orig_map_name, orig_pose, dest_map_name, dest_pose):
    orig_map_path = MAPS_ROOT_PATH / orig_map_name
    dest_map_path = MAPS_ROOT_PATH / dest_map_name
    transformed_pose = transform_pose_between_maps(orig_map_path,
                                                   dest_map_path, orig_pose)

    assert np.allclose(transformed_pose[:3], dest_pose[:3], atol=1e-05)
    # quaternion = -quaternion
    assert np.allclose(transformed_pose[3:], dest_pose[3:], atol=1e-05) \
        or np.allclose(transformed_pose[3:], -dest_pose[3:], atol=1e-05)


@pytest.mark.parametrize("orig_map_name,orig_pose,dest_map_name,dest_pose", [
    ('map_a', np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
     'map_a', np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
    ('map_a', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795]),
     'map_b', np.array([0.9, 3.1, 0.0, 0.0, 0.0, 0.9238795, 0.3826834])),
    ('map_b', np.array([0.9, 3.1, 0.0, 0.0, 0.0, 0.9238795, 0.3826834]),
     'map_c', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795])),
    ('map_a', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795]),
     'map_c', np.array([1.1, 0.1, 0.0, 0.0, 0.0, 0.3826834, 0.9238795])),
    # the closes key frame in map_c is not in map_b
    ('map_c', np.array([11.0, 10.0, 0.0, 0.0, 0.0, 0, 1.0]),
     'map_b', np.array([11.0, 13.0, 0.0, 0.0, 0.0, 0.70710678, 0.70710678])),
])
def test_transform_points_between_maps(orig_map_name, orig_pose: np.ndarray, dest_map_name, dest_pose):
    orig_map_path = MAPS_ROOT_PATH / orig_map_name
    dest_map_path = MAPS_ROOT_PATH / dest_map_name
    transformed_points = transform_points_between_maps(orig_map_path,
                                                       dest_map_path,
                                                       orig_pose[np.newaxis, :2])

    assert np.allclose(transformed_points[0], dest_pose[:2], atol=1e-05)


@pytest.mark.parametrize("map_name,shape", [
    ('map_a', (30, 40)),  # h: 1.5 m, w: 2.0 m
    ('map_b', (40, 240)),  # h: 6.0 m, w: 12.0 m
    ('map_c', (230, 240)),  # h: 11.5 m, w: 12.0 m
])
def test_create_cost_map_from_vertices(map_name, shape):
    vertices = FeatureMapVertices.from_map_path(MAPS_ROOT_PATH / map_name)
    cost_map = create_cost_map_from_vertices([vertices], margin=0.5, resolution=0.05)
    assert np.all(cost_map.data == np.zeros(shape))
