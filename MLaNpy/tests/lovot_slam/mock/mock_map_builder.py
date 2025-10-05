import datetime
import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import yaml

from grid_map_util.occupancy_grid import OccupancyGrid

from lovot_slam.env import (MAP_2DMAP, MAP_FEATUREMAP, MAP_FEATUREMAP_MISSIONS_YAML, MAP_FEATUREMAP_SENSORS_YAML,
                            MAP_MD5SUM_YAML, MAP_SUMMARYMAP, MAP_YAML, data_directories)
from lovot_slam.utils.file_util import get_file_md5sum
from lovot_slam.utils.map_utils import MAP_SUMMARYMAP_FILES, MapUtils

from ..feature_map.feature_map_vertices_data import create_vertices_csv_files

# mock maplab result directory structure
MAP_FEATUREMAP_ALL_FILES = [
    'distance_edge.csv',
    'metadata',
    'resource_info',
    'resource/',
    'vi_map/edges0',
    'vi_map/landmark_index',
    'vi_map/missions',
    'vi_map/optional_sensor_data',
    'vi_map/sensors.yaml',
    'vi_map/vertices0',
]


def _create_uuid() -> str:
    return str(uuid.uuid4()).replace('-', '')


class MockMap:
    def __init__(self, maps_dir: Path, map_name: str, mission_ids: List[str], source_maps: List['MockMap'],
                 timestamp: float = 0.0) -> None:
        self._maps_dir = maps_dir
        self.map_name = map_name
        self.map_dir = self._maps_dir / self.map_name
        self.mission_ids = mission_ids
        self.source_maps = source_maps
        self.timestamp = timestamp
        self.version = 4

    @classmethod
    def from_map_dir(cls, map_dir: Path) -> 'MockMap':
        maps_dir = map_dir.parent
        # read source list from lovot_slam.yaml
        with open(map_dir / MAP_YAML) as f:
            yaml_dict = yaml.safe_load(f)
        source_maps = [cls.from_map_dir(maps_dir / map_name) for map_name in yaml_dict['lovot_slam']['source']]
        # read missions list from vi_map/missions.yaml
        with open(map_dir / MAP_FEATUREMAP / MAP_FEATUREMAP_MISSIONS_YAML) as f:
            yaml_dict = yaml.safe_load(f)
        mission_ids = yaml_dict['mission_ids']
        return cls(maps_dir, map_dir.name, mission_ids, source_maps)

    def _create_mock_files(self, dir_name: str, files: List[str]) -> None:
        dir_path = self.map_dir / dir_name

        for file in files:
            file_path = dir_path / file
            if file.endswith('/'):
                file_path.mkdir(parents=True, exist_ok=True)
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()

    def _create_2d_map(self) -> None:
        img = np.full((100, 100), OccupancyGrid.UNKNOWN_CODE, dtype=np.uint8)
        img[40:60, 40:60] = OccupancyGrid.OCCUPIED_CODE
        img[42:58, 42:58] = OccupancyGrid.FREE_CODE
        grid_map = OccupancyGrid(img, 0.05, np.zeros(2))
        grid_map.save(self.map_dir / MAP_2DMAP / 'map.yaml')

    def _create_lovot_slam_yaml(self) -> None:
        # from timestamp [sec]
        timestamp_dt = datetime.datetime.fromtimestamp(self.timestamp)
        config = {
            "date": timestamp_dt.strftime('%Y%m%d%H%M%S'),
            "feature_map": {
                    "height_mean": 0.0,
                    "height_std": 0.0,
            },
            "scale": {mission_ids: 1.0 for mission_ids in self.mission_ids},
            "scale_odom_loc": 1.0045394174291271,
            "source": [map_.map_name for map_ in self.source_maps],
            "version": self.version,
        }

        yaml_path = self.map_dir / MAP_YAML
        with open(yaml_path, 'w') as f:
            f.write(yaml.dump({"lovot_slam": config}))

    def _get_map_content_path_list(self) -> List[Path]:
        path_list = []
        path_list += self.map_dir.glob(f'{MAP_FEATUREMAP}/**/*')
        path_list += self.map_dir.glob(f'{MAP_SUMMARYMAP}/**/*')
        path_list += self.map_dir.glob(f'{MAP_2DMAP}/**/*')
        path_list += self.map_dir.glob(f'{MAP_YAML}')
        return path_list

    def _create_md5sum_list(self) -> None:
        path_list = self._get_map_content_path_list()
        md5sum_dict = {str(path.relative_to(self.map_dir)): get_file_md5sum(path)
                       for path in path_list if path.is_file()}
        with open(self.map_dir / MAP_MD5SUM_YAML, 'w') as f:
            yaml.safe_dump(md5sum_dict, f)

    def _create_vi_map_sensors_yaml(self) -> None:
        with open(self.map_dir / MAP_FEATUREMAP / MAP_FEATUREMAP_SENSORS_YAML, 'w') as f:
            yaml.safe_dump({'mission_sensors_associations': [
                           {'mission_id': mission_id} for mission_id in self.mission_ids]}, f)

    def _create_vi_map_missions_yaml(self) -> None:
        with open(self.map_dir / MAP_FEATUREMAP / MAP_FEATUREMAP_MISSIONS_YAML, 'w') as f:
            yaml.safe_dump({'mission_ids': self.mission_ids}, f)

    def _create_vertices_csv(self) -> None:
        for mission_id in self.mission_ids:
            csv_path = self.map_dir / MAP_FEATUREMAP / 'vertices' / mission_id / 'vertices.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.touch()

    def create(self, exist_ok: bool = False) -> None:
        self.map_dir.mkdir(parents=True, exist_ok=exist_ok)
        self._create_lovot_slam_yaml()
        self._create_mock_files(MAP_FEATUREMAP, MAP_FEATUREMAP_ALL_FILES)
        self._create_vertices_csv()
        self._create_vi_map_sensors_yaml()
        self._create_vi_map_missions_yaml()
        self._create_mock_files(MAP_SUMMARYMAP, MAP_SUMMARYMAP_FILES)
        self._create_2d_map()
        self._create_md5sum_list()

    def update_md5sum(self) -> None:
        self._create_md5sum_list()

    def create_corresponding_bag(self, bags_dir: Path, original: bool, converted: bool) -> None:
        bags_dir.mkdir(parents=True, exist_ok=True)
        if original:
            (bags_dir / f'{self.map_name}.bag').touch()
        if converted:
            (bags_dir / f'{self.map_name}_converted.bag').touch()


class MockSingleMissionMap(MockMap):
    """single mission mapのダミーディレクトリを作成する
    現状、map_nameには現在時刻を使っているので、同じタイミングで複数のmapを作成することはできない
    """
    def __init__(self, maps_dir: Path, map_name: Optional[str] = None,
                 mission_id: Optional[str] = None, timestamp: float = 0.0) -> None:
        if map_name is None:
            now = datetime.datetime.now()
            map_name = now.strftime("%Y%m%d_%H%M%S")
        if mission_id is None:
            mission_id = _create_uuid()

        super().__init__(maps_dir, map_name, [mission_id], [], timestamp=timestamp)


class MockMergedMap(MockMap):
    """merged mapのダミーディレクトリを作成する
    """
    def __init__(self, maps_dir: Path, source_maps: List[MockSingleMissionMap],
                 map_name: str = None, timestamp: float = 0.0) -> None:
        assert len(source_maps) > 0

        now = datetime.datetime.now()
        prefix = now.strftime("%Y%m%d_%H%M%S")
        source_map_names = [map_.map_name for map_ in source_maps]
        key = hashlib.md5(''.join(source_map_names).encode()).hexdigest()
        map_name = map_name or f'{prefix}_{key}'
        mission_ids = [map_.mission_ids[0] for map_ in source_maps]

        super().__init__(maps_dir, map_name, mission_ids, source_maps, timestamp=timestamp)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        maps_root = Path(tmpdir) / 'maps'
        bags_root = Path(tmpdir) / 'bags'

        map_utils = MapUtils(maps_root, bags_root)

        single_map = MockSingleMissionMap(maps_root)
        single_map.create()
        single_map.create_corresponding_bag(bags_root, original=True, converted=True)

        merged_map = MockMergedMap(maps_root, source_maps=[single_map])
        merged_map.create()

        print(map_utils.get_map_list())


def prepare_dummy_single_merged_maps_pair() -> Tuple[str, str]:
    data_directories.maps.mkdir(parents=True, exist_ok=True)
    data_directories.bags.mkdir(parents=True, exist_ok=True)

    maps_root = data_directories.maps
    single_map = MockSingleMissionMap(maps_root)
    single_map.create()

    merged_map = MockMergedMap(maps_root, source_maps=[single_map])
    merged_map.create()
    return single_map.map_name, merged_map.map_name


@pytest.fixture
def mock_maps_from_maps_vertices(maps_vertices: dict):
    """Create mock maps from maps_vertices which is as follows:
    {
        'map_a': {'01': np.ndarray},
        'map_b': {'01': np.ndarray, '02': np.ndarray},
        'map_c': {'01': np.ndarray, '02': np.ndarray, ...},
    }
    where np.ndarray is an array of shape (n, 18) representing vertices in a mission.
    The columns are as follows:
      index, timestamp, position(3), quaternion(4), velocity(3), acc_bias(3), gyro_bias(3)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        maps_root = Path(tmpdir) / 'maps'
        bags_root = Path(tmpdir) / 'bags'

        # List up all unique mission_ids
        unique_mission_ids = set()
        for missions in maps_vertices.values():
            unique_mission_ids.update(missions.keys())

        # Create single mission maps which correspond to the missions
        single_mission_maps = {}
        for mission_id in unique_mission_ids:
            single_mission_map = MockSingleMissionMap(maps_root, mission_id=mission_id)
            single_mission_map.create()
            single_mission_map.create_corresponding_bag(bags_root, original=True, converted=True)
            single_mission_maps[mission_id] = single_mission_map

        # Create merged maps
        timestamp = 0.0
        for map_name, missions in maps_vertices.items():
            source_maps = [single_mission_maps[mission_id] for mission_id in missions.keys()]
            merged_map = MockMergedMap(maps_root, source_maps=source_maps, map_name=map_name,
                                       timestamp=timestamp)
            merged_map.create()
            create_vertices_csv_files(merged_map.map_dir, missions)
            merged_map.update_md5sum()
            timestamp += 3 * 24 * 3600

        yield maps_root, bags_root
