import importlib
import os
import pathlib
import re
import shutil
import sys
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple

import numpy as np
import pytest
import responses
import yaml

from lovot_apis.lovot.navigation.domain_event_pb2 import HomeMapEvent, SpotEvent, UnwelcomedAreaEvent
from lovot_apis.lovot.navigation.navigation_pb2 import Coordinate

import lovot_slam.env
import lovot_slam.exploration
import lovot_slam.utils.map_utils
from lovot_slam.env import DataDirectories, data_directories
from lovot_slam.exploration.frontier_search import LTM_FRONTIER_HISTORY_KEY
from lovot_slam.redis import create_ltm_client, redis_keys
from lovot_slam.redis.keys import COLONY_ID_KEY
from lovot_slam.utils.map_utils import MapSetUtils
from MLaNpy.lovot_map.rosmap import RosMap
from lovot_slam.utils.unwelcomed_area import Polygon

# TODO: make dataset package
BASE_DIR = pathlib.Path(__file__).parent
DATASET_ROOT = BASE_DIR.parents[3] / 'dataset'
MAP_YAML = BASE_DIR.parent / '2d_map' / 'map.yaml'
sys.path.append(DATASET_ROOT / 'maps')


def setup_module():
    lovot_slam.env.SPIKE_LOCALHOST = 'localhost'
    lovot_slam.env.DataDirectories.DATA_ROOT = pathlib.Path('/tmp/localization')
    if lovot_slam.env.data_directories.data_root.exists():
        shutil.rmtree(lovot_slam.env.data_directories.data_root)
    maps_root = lovot_slam.env.data_directories.maps
    bags_root = lovot_slam.env.data_directories.bags
    bag_utils = lovot_slam.utils.map_utils.BagUtils(bags_root)
    bag_utils.create_directory()
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    map_utils.create_directory()
    importlib.reload(lovot_slam.utils.map_utils)
    importlib.reload(lovot_slam.exploration)
    ltm_client = create_ltm_client()
    ltm_client.set(COLONY_ID_KEY, 'testcolonyid')
    ltm_client.delete(redis_keys.map)


def teardown_module():
    lovot_slam.env.SPIKE_LOCALHOST = 'spike'
    importlib.reload(lovot_slam.utils)
    ltm_client = create_ltm_client()
    ltm_client.delete(COLONY_ID_KEY)


@pytest.fixture
def mock_server():
    with responses.RequestsMock() as mock:
        yield mock


@pytest.fixture
def dummy_data_directory() -> Tuple[pathlib.Path, pathlib.Path]:
    maps_root = lovot_slam.env.data_directories.maps
    bags_root = lovot_slam.env.data_directories.bags
    shutil.rmtree(maps_root)
    shutil.rmtree(bags_root)
    shutil.copytree(DATASET_ROOT / 'dummy' / 'maps', maps_root)
    shutil.copytree(DATASET_ROOT / 'dummy' / 'rosbag', bags_root)
    return (maps_root, bags_root)


@pytest.fixture
def mock_frontier_history(monkeypatch):
    ltm_client = create_ltm_client()
    ltm_client.delete(LTM_FRONTIER_HISTORY_KEY)
    yield ltm_client
    ltm_client.delete(LTM_FRONTIER_HISTORY_KEY)


@pytest.mark.parametrize("base,add,target", [
    ({},
     {"b": {"b_b": {"added": {"nested": "dict added"}}}},
     {"b": {"b_b": {"added": {"nested": "dict added"}}}}),
    ({"a": "hoge", "b": {"b_a": "hoge", "b_b": {"b_b_a": "fuga"}}},
     {"a": "replace"},
     {"a": "replace", "b": {"b_a": "hoge", "b_b": {"b_b_a": "fuga"}}}),
    ({"a": "hoge", "b": {"b_a": "hoge", "b_b": {"b_b_a": "fuga"}}},
     {"b": {"b_a": "nested_replace"}},
     {"a": "hoge", "b": {"b_a": "nested_replace", "b_b": {"b_b_a": "fuga"}}}),
    ({"a": "hoge", "b": {"b_a": "hoge", "b_b": {"b_b_a": "fuga"}}},
     {"b": {"b_b": {"added": {"nested": "dict added"}}}},
     {"a": "hoge", "b": {"b_a": "hoge", "b_b": {"b_b_a": "fuga", "added": {"nested": "dict added"}}}}),
])
def test_update_dict(base, add, target):
    lovot_slam.utils.map_utils.update_dict(base, add)
    assert target == base


def test_get_merged_map_list(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    merged_map_list = map_utils.get_merged_map_list()
    assert merged_map_list == ['20190714_051701_8d2552ddd47de903deb2b21ef533ea11',
                               '20190714_063835_461d5acc6f8fa2adb28fd23f671477ed',
                               '20190714_090357_6d426e27972055e9b2a79514d707d2b0']


def test_get_latest_merged_map(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    latest_map = map_utils.get_latest_merged_map()
    assert latest_map == '20190714_090357_6d426e27972055e9b2a79514d707d2b0'


@pytest.mark.parametrize("ready", [
    (True),
    (False)
])
async def test_upload_map_to_cloud(ready, mock_httpx):
    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)
    map_utils = lovot_slam.utils.map_utils.MapUtils(None, None)

    assert await map_utils.upload_map_to_cloud(rosmap, ready)

    assert mock_httpx.call_count == 1
    assert mock_httpx.call_args[0][0] == 'http://localhost:48480/navigation/home-map-event'

    event = HomeMapEvent()
    event.ParseFromString(mock_httpx.call_args_list[0][1]["data"])

    assert event.colony_id == "testcolonyid"
    assert event.map_id == "1"
    assert event.home_map.width == 147
    assert event.home_map.height == 205
    assert event.home_map.resolution == pytest.approx(0.05)
    assert event.home_map.origin == Coordinate(px=-4.25, py=-3.55, pz=0.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    assert len(event.home_map.data) > 0
    # the flag named 'completed' in the cloud is acutally 'ready'
    assert event.home_map.completed == ready


def test_spot_to_cloud(mock_server):
    mock_server.add(
        mock_server.POST, 'http://localhost:48480/navigation/spot-event',
        body='{}', status=201,
        content_type='application/json')

    name = 'entrance'
    coordinate = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1]

    spot_utils = lovot_slam.utils.map_utils.SpotUtils(None)
    assert spot_utils.upload_spot_to_cloud(name, coordinate)

    req = mock_server.calls[0].request
    assert req.method == 'POST'
    assert req.url == 'http://localhost:48480/navigation/spot-event'

    event = SpotEvent()
    event.ParseFromString(req.body)

    assert event.colony_id == "testcolonyid"
    assert event.map_id == "1"
    assert event.spot_name == name
    assert event.event == SpotEvent.spot_updated
    assert event.spot.colony_id == "testcolonyid"
    assert event.spot.map_id == "1"
    assert event.spot.name == name
    assert event.spot.coordinate == Coordinate(
        px=coordinate[0],
        py=coordinate[1],
        pz=coordinate[2],
        ox=coordinate[3],
        oy=coordinate[4],
        oz=coordinate[5],
        ow=coordinate[6]
    )


def test_unwelcomed_area_to_cloud(mock_server):
    mock_server.add(
        mock_server.POST, 'http://localhost:48480/navigation/unwelcomed-area-event',
        body='{}', status=201,
        content_type='application/json')

    data = '{UnwelcomedArea.}data is not "validated"' \
        ' and thus any string[, such as this,] will be accepted.'

    spot_utils = lovot_slam.utils.map_utils.SpotUtils(None)
    assert spot_utils.upload_unwelcomed_area_to_cloud(data)

    req = mock_server.calls[0].request
    assert req.method == 'POST'
    assert req.url == 'http://localhost:48480/navigation/unwelcomed-area-event'

    event = UnwelcomedAreaEvent()
    event.ParseFromString(req.body)

    assert event.colony_id == "testcolonyid"
    assert event.map_id == "1"
    assert event.area_id == "1"
    assert event.event == UnwelcomedAreaEvent.unwelcomed_area_updated
    assert event.area.area_id == "1"
    assert event.area.colony_id == "testcolonyid"
    assert event.area.map_id == "1"
    assert event.area.data == data


def test_remove_all_maps(dummy_data_directory):
    maps_root: pathlib.Path = dummy_data_directory[0]
    bags_root: pathlib.Path = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    map_utils.remove_all_maps()
    assert all([file == 'frontier_history.yaml' or file.suffix == '.zip' for file in maps_root.iterdir()])
    assert list(bags_root.iterdir()) == []


def test_remove_unused_resources(dummy_data_directory):
    maps_root: pathlib.Path = dummy_data_directory[0]
    bags_root: pathlib.Path = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    map_utils.remove_unused_resources()
    assert {'20190714_044330_converted.bag',
            '20190714_055219_converted.bag',
            '20190714_083020_converted.bag'} == set(item.name for item in bags_root.iterdir())
    assert {'20190714_044330',
            '20190714_055219',
            '20190714_083020',
            '20190714_063835_461d5acc6f8fa2adb28fd23f671477ed',
            '20190714_090357_6d426e27972055e9b2a79514d707d2b0'} == set(item.name for item in maps_root.iterdir())


def test_remove_unused_resources_with_exclusions(dummy_data_directory):
    maps_root: pathlib.Path = dummy_data_directory[0]
    bags_root: pathlib.Path = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    # assume that 20190714_092510 is not processed yet
    # but is on the queue, so it should not be removed
    (bags_root / '20190714_092510_converted.bag').unlink()
    shutil.rmtree(maps_root / '20190714_092510')

    # print([file.stem for file in bags_root.iterdir()])
    # print([file.stem for file in maps_root.iterdir()])
    exclusions = ['20190714_092510']
    map_utils.remove_unused_resources(exclusions=exclusions)
    assert {'20190714_044330_converted.bag',
            '20190714_055219_converted.bag',
            '20190714_083020_converted.bag',
            '20190714_092510.bag'} == set(item.name for item in bags_root.iterdir())
    assert {'20190714_044330',
            '20190714_055219',
            '20190714_083020',
            '20190714_063835_461d5acc6f8fa2adb28fd23f671477ed',
            '20190714_090357_6d426e27972055e9b2a79514d707d2b0'} == set(item.name for item in maps_root.iterdir())


@pytest.mark.parametrize("map_name,mission_ids", [
    ('20190714_044330', ['fb815673000000001000000000000000']),
    ('20190714_090357_6d426e27972055e9b2a79514d707d2b0',
     ['e7cbed54000000001000000000000000', 'bb74a2bcffffffff1000000000000000', 'fb815673000000001000000000000000'])
])
def test_get_mission_id_list_from_feature_map(dummy_data_directory, map_name, mission_ids):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    _mission_ids = map_utils.get_mission_id_list_from_feature_map(map_name)

    assert set(_mission_ids) == set(mission_ids)


def test_update_metadata(dummy_data_directory):
    maps_root: pathlib.Path = dummy_data_directory[0]
    bags_root: pathlib.Path = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_name = '20190714_044330'
    source_list = ['dummy_1', 'dummy_2']
    map_utils.update_metadata(map_name, source_list=source_list, write_statistics=True)

    metadata_yaml = maps_root / map_name / 'lovot_slam.yaml'
    with open(metadata_yaml, 'r') as f:
        metadata = yaml.safe_load(f)

    assert re.match('[0-9]{8}', metadata['lovot_slam']['date']) is not None
    assert metadata['lovot_slam']['source'] == source_list
    assert metadata['lovot_slam']['version'] == lovot_slam.utils.map_utils.MAP_VERSION
    assert np.isclose(metadata['lovot_slam']['feature_map']['height_mean'], 0.016144933017849666)
    assert np.isclose(metadata['lovot_slam']['feature_map']['height_std'], 0.021896628069977025)


def test_get_map_version(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_name = '20190714_044330'
    source_list = ['dummy_1', 'dummy_2']
    map_utils.update_metadata(map_name, source_list=source_list)
    _version = map_utils.get_map_version(map_name)

    assert _version == lovot_slam.utils.map_utils.MAP_VERSION


def test_get_map_stamp(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_name = '20190714_044330'
    source_list = ['dummy_1', 'dummy_2']
    map_utils.update_metadata(map_name, source_list=source_list)

    slam_yaml = map_utils.get_full_path(map_name) / 'lovot_slam.yaml'
    with open(slam_yaml, 'r') as f:
        metadata = yaml.safe_load(f)
    metadata['lovot_slam']['date'] = '20190714044330'
    with open(slam_yaml, 'w') as f:
        yaml.safe_dump(metadata, f)

    stamp = map_utils.get_map_stamp(map_name)

    # offset of the timezone to the UTC
    utc_offset = datetime.fromtimestamp(0, tz=timezone.utc).astimezone().utcoffset()
    assert stamp == 1563079410.0 - utc_offset.total_seconds()


@pytest.mark.parametrize("version,compatibility", [
    (1, 1 in lovot_slam.utils.map_utils.SUPPROTED_MAP_VERSIONS),
    (0, 0 in lovot_slam.utils.map_utils.SUPPROTED_MAP_VERSIONS)
])
def test_check_map(dummy_data_directory, version, compatibility):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_name = '20190714_044330'
    source_list = ['dummy_1', 'dummy_2']
    map_utils.update_metadata(map_name, source_list=source_list)

    metadata = map_utils.get_metadata(map_name)
    metadata['lovot_slam']['version'] = version
    map_utils._write_metadata(map_name, metadata)

    assert map_utils.check_map(map_name) == compatibility


def test_get_metadata_with_broken_yaml(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_name = '20250228_012345'
    (maps_root / map_name).mkdir()

    # binary 0000
    with open(maps_root / map_name / 'lovot_slam.yaml', 'wb') as f:
        f.write(b'\x00\x00\x00\x00')
    metadata = map_utils.get_metadata(map_name)
    assert metadata == {}


@pytest.mark.parametrize("map_name", [
    ('20190714_051701_8d2552ddd47de903deb2b21ef533ea11'),
    ('20190714_063835_461d5acc6f8fa2adb28fd23f671477ed'),
    ('20190714_090357_6d426e27972055e9b2a79514d707d2b0'),
])
def test_get_merged_single_mission_map_names(dummy_data_directory, map_name):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    map_names = map_utils.get_merged_single_mission_map_names(map_name)

    metadata = map_utils.get_metadata(map_name)
    source_list = metadata['lovot_slam']['source']

    assert map_names == source_list


@pytest.mark.parametrize("number,map_names", [
    (1, ['20190714_083020']),
    (2, ['20190714_083020', '20190714_055219']),
    (20, ['20190714_083020', '20190714_055219', '20190714_044330']),
    (-1, ['20190714_083020', '20190714_055219', '20190714_044330']),
])
def test_get_recent_single_mission_maps(dummy_data_directory, number, map_names):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    _map_names = map_utils.get_recent_single_mission_maps(number)

    assert set(_map_names) == set(map_names)


def test_get_maps_number_in_latest_merged_map(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)

    assert map_utils.get_maps_number_in_latest_merged_map() == 3


@pytest.mark.parametrize("pose_str,pose", [
    ('0,0,0,0,0,0,1', np.array([0, 0, 0, 0, 0, 0, 1])),
])
def test_pose_from_string(pose_str, pose):
    _pose = lovot_slam.utils.map_utils.SpotUtils.pose_from_string(pose_str)

    assert np.all(np.isclose(pose, _pose))


@pytest.mark.parametrize("pose_str,pose", [
    ('0,0,0,0,0,0,1', np.array([0, 0, 0, 0, 0, 0, 1])),
])
def test_pose_to_string(pose_str, pose):
    _pose_str = lovot_slam.utils.map_utils.SpotUtils.pose_to_string(pose)

    assert np.all(np.isclose(np.array(list(map(float, pose_str.split(",")))),
                             np.array(list(map(float, _pose_str.split(","))))))


MAPS_VERTICES = {
    'map_a': {
        #     ↑ 4
        #     ↑ 3
        # → → →
        # 0 1 2
        'mission_0': np.array([
            # [index, timestamp] + [position] + [quaternion]
            [0, 1] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0],
            [1, 2] + [0.5, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0],
            [2, 3] + [1.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0],
            [3, 4] + [1.0, 0.5, 0.0] + [0.0, 0.0, 0.7071068, 0.7071068],
            [4, 5] + [1.0, 1.0, 0.0] + [0.0, 0.0, 0.7071068, 0.7071068],
        ])
    },
    # map origin is shifted by (1.0, 2.0) and then rotated by +90 deg.
    'map_b': {
        # 4 3
        # ← ← ↑ 2
        #     ↑ 1
        #     ↑ 0
        'mission_0': np.array([
            [0, 1] + [1.0, 2.0, 0.0] + [0.0, 0.0, 0.7071068, 0.7071068],
            [1, 2] + [1.0, 2.5, 0.0] + [0.0, 0.0, 0.7071068, 0.7071068],
            [2, 3] + [1.0, 3.0, 0.0] + [0.0, 0.0, 0.7071068, 0.7071068],
            [3, 4] + [0.5, 3.0, 0.0] + [0.0, 0.0, 1.0, 0.0],
            [4, 5] + [0.0, 3.0, 0.0] + [0.0, 0.0, 1.0, 0.0],
        ])
    },
}


@pytest.mark.parametrize("maps_vertices,orig_pose,dest_pose", 
                         [  # NOTE: map origin is shifted by (1.0, 2.0) and then rotated by +90 deg.
                            # exactly the same as the 1st vertex in the map_a/map_b
                            (MAPS_VERTICES,
                             np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                             np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.7071068, 0.7071068])),
                            # exactly the same as the 3rd vertex in the map_a/map_b
                            (MAPS_VERTICES,
                             np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]),
                             np.array([0.5, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
                            # 0.1 m forward from the 1st vertex in the map_a/map_b
                            (MAPS_VERTICES,
                             np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                             np.array([1.0, 2.1, 0.0, 0.0, 0.0, 0.7071068, 0.7071068]))])
def test_transform_pose_to_the_latest_map(orig_pose, dest_pose, mock_maps_from_maps_vertices):
    maps_root, bags_root = mock_maps_from_maps_vertices
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    spot_utils = lovot_slam.utils.map_utils.SpotUtils(map_utils)

    pose = spot_utils._transform_pose_to_the_latest_map(orig_pose)

    # compare positions
    assert np.all(np.isclose(pose[:3], dest_pose[:3]))
    # compare quaternions (allowing the sign difference)
    assert np.all(np.isclose(pose[3:], dest_pose[3:])) or \
        np.all(np.isclose(pose[3:], -dest_pose[3:]))


NOGO_AREA_RECT = Polygon(np.array([[0.0, 0.0], 
                                   [0.0, 2.0], 
                                   [1.0, 2.0], 
                                   [1.0, 0.0]]))

AFTER_TRANSFORMED_RECT = Polygon(np.array([[1., 2.],
                                           [-1., 2.],
                                           [-1., 3.],
                                           [1., 3.]]))


@pytest.mark.parametrize("maps_vertices,orig_rect,dest_rect", 
                         [(MAPS_VERTICES, NOGO_AREA_RECT, AFTER_TRANSFORMED_RECT) ])
def test_transform_unwelcomed_area_to_the_latest_map(orig_rect, dest_rect, mock_maps_from_maps_vertices):
    maps_root, bags_root = mock_maps_from_maps_vertices
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    spot_utils = lovot_slam.utils.map_utils.SpotUtils(map_utils)

    rect = spot_utils.transform_unwelcomed_area_to_the_latest_map(orig_rect)

    assert isinstance(rect, Polygon)
    assert np.all(np.isclose(rect.vertices, dest_rect.vertices))


@pytest.mark.parametrize("maximum_area_size_pixel_square,latest_map", [
    (25 * 25 / (0.05 * 0.05), '20191111_075253_fe5c04d5e8ad71d013cdb478784da980'),
    (30 * 30 / (0.05 * 0.05), '20191111_081536_45ed1034fddaeae54069c1b799fb81fa'),
])
def test_maximum_area_size(maximum_area_size_pixel_square, latest_map):
    maps_root = DATASET_ROOT / 'maps' / 'sequence_f'
    lovot_slam.utils.map_utils.MAXIMUM_MAP_AREA_PIXEL_SQUARE = maximum_area_size_pixel_square
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, None)

    assert map_utils.get_latest_merged_map() == latest_map


def test_update_redis(dummy_data_directory):
    maps_root = dummy_data_directory[0]
    bags_root = dummy_data_directory[1]
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    ltm_client = create_ltm_client()

    latest_map_name = map_utils.get_latest_merged_map()
    rosmap = map_utils.get_ros_map(latest_map_name)
    map_utils.update_redis(rosmap)
    map_from_hash = ltm_client.hgetall(redis_keys.map)

    assert map_from_hash['name'] == '20190714_090357_6d426e27972055e9b2a79514d707d2b0'
    assert map_from_hash['width'] == '100'
    assert map_from_hash['height'] == '100'
    assert map_from_hash['resolution'] == '0.05'


def test_reset_map_accuracy():
    map_utils = lovot_slam.utils.map_utils.MapUtils(None, None)
    map_utils.reset_map_accuracy()

    monitor_dir = lovot_slam.env.data_directories.monitor
    assert not monitor_dir.exists()


@pytest.fixture
def mock_mapset_dir(monkeypatch) -> Iterable[pathlib.Path]:
    with TemporaryDirectory() as tmpdir:
        mapset_root = pathlib.Path(tmpdir)
        # need a mapset directory in the mapset root
        dummy_mapset_dir = mapset_root / 'mapset_dummy'
        dummy_mapset_dir.mkdir()
        # need a symlink, which is the current data root
        data_root = mapset_root / 'current'
        data_root.symlink_to(dummy_mapset_dir)
        # set the data root
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', data_root)
        yield mapset_root


def test_mapset_utils_create_mapset(mock_mapset_dir):
    mapset_utils = MapSetUtils(mock_mapset_dir)
    mapset_utils.create_mapset('test_mapset')

    assert (mock_mapset_dir / 'test_mapset').exists()


def test_mapset_utils_does_mapset_exist(mock_mapset_dir):
    mapset_utils = MapSetUtils(mock_mapset_dir)

    assert not mapset_utils.does_mapset_exist('mapset_1')

    mapset_utils.create_mapset('mapset_1')
    assert mapset_utils.does_mapset_exist('mapset_1')


def test_mapset_utils_change_mapset(mock_mapset_dir):
    mapset_utils = MapSetUtils(mock_mapset_dir)

    mapset_utils.change_mapset('mapset_1')
    assert pathlib.Path(os.readlink(data_directories.data_root)) == mock_mapset_dir / 'mapset_1'

    mapset_utils.change_mapset('mapset_2')
    assert pathlib.Path(os.readlink(data_directories.data_root)) == mock_mapset_dir / 'mapset_2'


def test_mapset_utils_generate_new_mapset_name(mock_mapset_dir):
    mapset_utils = MapSetUtils(mock_mapset_dir)

    for _ in range(10):
        new_mapset_name = mapset_utils.generate_new_mapset_name()
        mapset_utils.change_mapset(new_mapset_name)

    print(list(mock_mapset_dir.iterdir()))
    # additional 2 is for the current and the dummy mapset
    assert len(list(mock_mapset_dir.iterdir())) == 10 + 2
