import pathlib

import numpy as np
import pytest

from grid_map_util.occupancy_grid import OccupancyGrid
from lovot_apis.lovot.navigation.navigation_pb2 import Coordinate

from lovot_slam.redis import create_ltm_client, redis_keys
from MLaNpy.lovot_map.rosmap import RosMap

# TODO: make dataset package
BASE_DIR = pathlib.Path(__file__).parent
MAP_YAML = BASE_DIR.parent / '2d_map' / 'map.yaml'


def test_open_and_write_map_to_redis():
    r = create_ltm_client()
    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)
    r.hset(redis_keys.map, mapping=rosmap.to_dict_for_redis())
    name = r.hget('slam:map', 'name')
    origin = r.hget('slam:map', 'origin')
    width = r.hget('slam:map', 'width')
    height = r.hget('slam:map', 'height')
    resolution = r.hget('slam:map', 'resolution')
    assert name == 'test_map'
    assert origin == '-4.25, -3.55, 0.0, 0.0, 0.0, 0.0, 1.0'
    assert width == '147'
    assert height == '205'
    assert resolution == '0.05'


def test_read_map_from_redis():
    r = create_ltm_client()
    map_name = 'test_map'
    rosmap = RosMap.from_map_yaml(map_name, MAP_YAML)
    r.hset(redis_keys.map, mapping=rosmap.to_dict_for_redis())
    keys = ['width', 'height', 'resolution', 'data', 'origin', 'name']
    hashed_values = r.hmget(redis_keys.map, keys)
    rosmap_loaded = RosMap.from_hashed_values(hashed_values)

    assert rosmap.name == rosmap_loaded.name
    assert rosmap.position == rosmap_loaded.position
    assert rosmap.orientation == rosmap_loaded.orientation
    assert rosmap.width == rosmap_loaded.width
    assert rosmap.height == rosmap_loaded.height
    assert rosmap.resolution == rosmap_loaded.resolution
    assert rosmap.data == rosmap_loaded.data


def test_as_occupancy_grid():
    oc_grid = OccupancyGrid.from_yaml_file(MAP_YAML)

    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)
    oc_grid_from_rosmap = rosmap.as_occupancy_grid()

    assert np.all(oc_grid.origin == oc_grid_from_rosmap.origin)
    assert oc_grid.origin_yaw == oc_grid_from_rosmap.origin_yaw
    assert oc_grid.resolution == oc_grid_from_rosmap.resolution
    assert np.all(oc_grid.img == oc_grid_from_rosmap.img)


def test_from_occupancy_grid():
    oc_grid = OccupancyGrid.from_yaml_file(MAP_YAML)
    rosmap_from_oc_grid = RosMap.from_occupancy_grid(oc_grid, name='test_map')

    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)

    assert rosmap.name == rosmap_from_oc_grid.name
    assert np.all(rosmap.origin_pos_2d == rosmap_from_oc_grid.origin_pos_2d)
    assert rosmap.origin_yaw == rosmap_from_oc_grid.origin_yaw
    assert np.all(rosmap.data == rosmap_from_oc_grid.data)
    assert rosmap.resolution == rosmap_from_oc_grid.resolution


def test_to_proto():
    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)
    proto = rosmap.to_proto("dummy_colony_id")
    assert proto.colony_id == "dummy_colony_id"
    assert proto.map_id == "1"
    assert proto.width == 147
    assert proto.height == 205
    assert proto.resolution == pytest.approx(0.05)
    assert proto.origin == Coordinate(px=-4.25, py=-3.55, pz=0.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    assert len(proto.data) > 0


def test_to_event_proto():
    rosmap = RosMap.from_map_yaml('test_map', MAP_YAML)
    event = rosmap.to_event_proto("dummy_colony_id")
    assert event.colony_id == "dummy_colony_id"
    assert event.map_id == "1"
    home_map = event.home_map
    assert home_map.colony_id == "dummy_colony_id"
    assert home_map.map_id == "1"
    assert home_map.width == 147
    assert home_map.height == 205
    assert home_map.resolution == pytest.approx(0.05)
    assert home_map.origin == Coordinate(px=-4.25, py=-3.55, pz=0.0, ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    assert len(home_map.data) > 0
