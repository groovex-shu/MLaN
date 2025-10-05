import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import time
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import anyio
import yaml
from attr import attrib, attrs
from scipy.spatial.transform import Rotation as R

from lovot_map.occupancy_grid import OccupancyGrid
from lovot_nav.protobufs.domain_event_pb2 import SpotEvent, UnwelcomedAreaEvent, HomeMapEvent
from lovot_nav.protobufs.navigation_pb2 import Coordinate, Spot, UnwelcomedArea

# from lovot_slam.client.agent_client import upload_data_to_cloud as upload_data_to_cloud_async
from lovot_slam.env import (AREA_ID, CLOUD_HOME_MAP_EVENT_ENDPOINT, CLOUD_SPOT_EVENT_ENDPOINT,
                            CLOUD_UNWELCOMED_AREA_EVENT_ENDPOINT, CLOUD_UPLOAD_PORT, LOCALHOST, MAP_2DMAP,
                            MAP_FEATUREMAP, MAP_FEATUREMAP_MISSIONS_YAML, MAP_FEATUREMAP_SENSORS_YAML, MAP_ID,
                            MAP_MD5SUM_YAML, MAP_SUMMARYMAP, SLAM_YAML, data_directories)
from lovot_slam.feature_map.feature_map_vertices import (FeatureMapVertices, transform_points_between_maps,
                                                         transform_pose_between_maps)
from lovot_slam.redis.clients import create_ltm_client, create_stm_client
from lovot_slam.utils.file_util import (get_file_md5sum, get_last_modified_time, remove_directory_if_exists,
                                        remove_file_if_exists)
from lovot_map.rosmap import RosMap
from lovot_slam.utils.unwelcomed_area import Polygon, decode_unwelcomed_area

logger = getLogger(__name__)

MAP_FEATUREMAP_FILES = [
    'metadata',
    'resource_info',
    'vi_map/edges0',
    'vi_map/landmark_index',
    'vi_map/missions',
    'vi_map/optional_sensor_data',
    'vi_map/sensors.yaml',
    'vi_map/vertices0']


MAP_SUMMARYMAP_FILES = [
    'localization_summary_map',
]

MAP_2DMAP_FILES = [
    'map.pgm',
    'map.yaml']

MAP_BUILDING_FLAG_FILE = 'building'

# MAP VERSION 変更履歴
# 2 (from 0.6.0)
#     - maplabのupdateに伴い、feature_mapの形式が変わったため、ver.1との互換性なし
# 2 (from 0.7.1)
#     - merged mapのfeature_mapフォルダ内に verticesフォルダが生成され、keyframes情報を保持するようにした
#     - verticesはspot位置修正のために使われる
# 3 (from 0.9.1)
#     - single mission mapのfeature_mapフォルダ内に verticesフォルダが生成され、keyframes情報を保持するようにした
#     - lovot_slam.yamlの`feature_map`keyにkeyframesの高さ情報を持たせるようにした
#     - 高さ情報を元にマップをcheckするようにしたため、ver.3以降のマップでこの情報がないマップは不正なマップと判定される
# 4 (from 0.10.3)
#     - マップ構造そのものの変更は無し
#     - 2Dマップの品質が向上した
#     - 旧マップを自動的に再生成するために便宜上バージョンを上げた
MAP_VERSION = 4
SUPPROTED_MAP_VERSIONS = [2, 3, 4]

MERGE_NUMBER_TO_NOTIFICATION = 10
MAXIMUM_MERGE_NUMBER = 20

DEFAULT_MAP_RESOLUTION = 0.05
MAXIMUM_MAP_AREA_METER_SQUARE = 25 * 25
MAXIMUM_MAP_AREA_PIXEL_SQUARE = MAXIMUM_MAP_AREA_METER_SQUARE / \
    (DEFAULT_MAP_RESOLUTION * DEFAULT_MAP_RESOLUTION)
# TODO: these limits should be more strict (mean:0.2, std:0.05)
# will be changed after checking metrics
MAP_STATS_HEIGHT_MEAN_LIMIT = 0.5
MAP_STATS_HEIGHT_STD_LIMIT = 0.1
# TODO: these limits should be more strict (0.7~1.3)
# will be changed after checking metrics
FEATURE_MAP_SCALE_LOWER_LIMIT = 0.5
FEATURE_MAP_SCALE_UPPER_LIMIT = 2.0

# TODO: tune this value, when the map updating function is introduced.
MIN_GOOD_MAP_ACCURACY = 0.4

MAX_DATA_DIR_SIZE = 35 * 1024 * 1024 * 1024  # 35GB


def upload_data_to_cloud(end_point, data):
    host = f"http://{LOCALHOST}:{CLOUD_UPLOAD_PORT}/{end_point}"
    logger.info(f'uploading to {host}')
    try:
        r = requests.post(host, data=data, timeout=5.0)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logger.error(e)
        return False
    if r.status_code != requests.codes.created:
        logger.error(f"failed to upload data to {end_point}. status_code: {r.status_code}")
        return False
    logger.info(f"succeeded to upload to {end_point}")
    return True


def update_dict(base, add):
    for k, v in add.items():
        if isinstance(v, dict) and k in base:
            update_dict(base[k], v)
        else:
            base[k] = v


@attrs
class _MapListCache:
    list: List[str] = attrib(default=[])
    timestamp: float = attrib(default=0)

    def is_valid(self, timestamp: float):
        return self.list and self.timestamp == timestamp


class BagUtils:
    def __init__(self, bags_root: pathlib.Path):
        self._root = bags_root

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def create_directory(self):
        self._root.mkdir(parents=True, exist_ok=True)

    def get_map_name_from_filename(self, filename: str) -> Optional[str]:
        m = re.match(r'^([0-9]{8}_[0-9]{6})(_converted)?\.bag', filename)
        if not m:
            return None
        return m.groups()[0]

    def is_converted(self, filename: str) -> bool:
        return bool(re.match(r'^([0-9]{8}_[0-9]{6})_converted\.bag', filename))

    def remove_all_bags(self):
        for file in self._root.iterdir():
            remove_file_if_exists(file)

    def remove_all_bags_except(self, exclusions: List[str]):
        for file in self._root.iterdir():
            map_name = self.get_map_name_from_filename(file.name)
            if not map_name:
                # invalid file name that doesn't match the pattern
                remove_file_if_exists(file)
                continue
            converted_bag = self.get_full_path(map_name, is_converted=True)
            if not self.is_converted(file.name) and converted_bag.exists():
                # remove an original (non-converted) bag file
                # if a corresponding converted bag file exists,
                # because the original bag file will not be used.
                remove_file_if_exists(file)
                continue
            if map_name in exclusions:
                continue
            remove_file_if_exists(file)

    def remove(self, bag_name: str, original: bool = True, converted: bool = False):
        if original:
            remove_file_if_exists(self.get_full_path(bag_name))
        if converted:
            remove_file_if_exists(self.get_full_path(bag_name + '_converted'))

    def get_full_path(self, bag_name: str, is_converted: bool = False) -> pathlib.Path:
        if is_converted:
            return self._root / f'{bag_name}_converted.bag'
        return self._root / f'{bag_name}.bag'

    async def filter_topics(self, bag_name: str, expression: str) -> bool:
        """Filter rosbag to reduce the size of the bag
        :param bag_name: name of the bag file
        :param expression: expression to filter the topic (e.g. 'topic != "/image_raw"')
        :return: True if succeeded, False otherwise
        """
        rosbag_path = self.get_full_path(bag_name)
        tmp_filtered_rosbag = self._root / "tmp.bag"
        cmd = ['rosbag', 'filter', rosbag_path, tmp_filtered_rosbag, expression]

        try:
            await anyio.run_process(cmd)
        except subprocess.CalledProcessError:
            return False

        # replace the original bag file with the filtered one
        shutil.move(tmp_filtered_rosbag, rosbag_path)
        return True

    @staticmethod
    async def get_duration(file_path: pathlib.Path) -> float:
        """ Get duration of bag file in seconds. """
        cmd = ['rosbag', 'info', '-y', '-k', 'duration', file_path]
        with anyio.fail_after(60):
            result = await anyio.run_process(cmd, capture_stdout=True)
            assert result.stdout, f"Failed to get duration for {file_path.name}."
            return float(result.stdout.decode('utf-8'))
        raise TimeoutError(f"Failed to get duration for {file_path.name} due to timeout.")


class MapUtils:
    def __init__(self, maps_root: pathlib.Path, bags_root: pathlib.Path):
        self._root = maps_root
        self._bags = bags_root
        self.redis_stm = create_stm_client()
        self.redis_ltm = create_ltm_client()

        self.bag_utils = BagUtils(bags_root)

        self._map_list_cache = _MapListCache()

        # remove unused key
        # TODO: remove this code (perhaps from OS 22.xx)
        self.redis_ltm.delete('slam:map_completed')

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def create_directory(self):
        self._root.mkdir(parents=True, exist_ok=True)

    def clean_map_directory(self):
        self.remove_all_maps()
        for path in self._root.iterdir():
            if not path.is_file():
                continue
            remove_file_if_exists(path)

    def remove_all_maps(self):
        for path in self._root.iterdir():
            if not path.is_dir():
                continue
            self.remove(path.name)

    def remove_all_maps_except(self, except_maps):
        if not isinstance(except_maps, list):
            except_maps = [except_maps]
        for path in self._root.iterdir():
            if not path.is_dir():
                continue
            if path.name in except_maps:
                continue
            self.remove(path.name)

    def remove_unused_resources(self, exclusions: List[str] = []) -> None:
        """
        remove all maps except the latest two merged maps and their source single mission maps.
        Pass the list of resource names (map names) to remain as exclusions,
        if you want to keep some rosbags which are not used by any maps.

        :param exclusions: list of map names (resource names) to remain
        """
        latest_two_merged_maps = self.get_recent_merged_maps(num=2)
        if latest_two_merged_maps:
            logger.info(f'remove all maps except {latest_two_merged_maps + exclusions} and their sources.')
            source_list = []
            for merged_map in latest_two_merged_maps:
                source_list += self.get_source_map_list(merged_map)
            maps_to_remain = source_list + latest_two_merged_maps + exclusions
            logger.debug(f'remove all maps except {maps_to_remain}')
            self.remove_all_maps_except(maps_to_remain)
            self.bag_utils.remove_all_bags_except(maps_to_remain)
        else:
            self.remove_all_maps_except(exclusions)
            self.bag_utils.remove_all_bags_except(exclusions)
        # remove remaining *.zip files
        # TODO: remove this code
        for file in self._root.glob('*.zip'):
            remove_file_if_exists(file)

    def remove(self, map_name: str):
        if not map_name:
            return
        if self.is_single_mission_map(map_name):
            self.bag_utils.remove(map_name, original=True, converted=True)
        remove_directory_if_exists(self._root / map_name)

    def get_metadata(self, map_name) -> dict:
        metadata_yaml = self._root / map_name / SLAM_YAML
        if not metadata_yaml.exists():
            return {}

        try:
            with open(metadata_yaml, 'r') as f:
                metadata = yaml.safe_load(f)
                return metadata
        except (OSError, yaml.YAMLError) as e:
            logger.error(f'failed to read {metadata_yaml}: {e}')
            return {}

    def _create_map_dir(self, map_name: str):
        map_dir = self._root / map_name
        if not map_dir.exists():
            map_dir.mkdir(parents=True)
            self.create_building_flag(map_name)

    def _write_metadata(self, map_name, metadata):
        self._create_map_dir(map_name)
        metadata_yaml = self._root / map_name / SLAM_YAML
        with open(metadata_yaml, 'w') as f:
            yaml.dump(metadata, f)

    # TODO: make metadata class
    def _update_basic_information_to_metadata(self, metadata):
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        basic_information_dict = {"lovot_slam": {
            "date": now,
            "version": MAP_VERSION,
        }}
        update_dict(metadata, basic_information_dict)

    def _append_source_list_to_metadata(self, metadata, source_list):
        update_dict(metadata, {"lovot_slam": {"source": source_list}})

    def _append_statistics_to_metadata(self, metadata, height_mean, height_std):
        statistics_dict = {"lovot_slam": {"feature_map": {
            "height_mean": height_mean,
            "height_std": height_std,
        }}}
        update_dict(metadata, statistics_dict)

    def update_metadata(self, map_name, source_list=None, write_statistics=False):
        metadata = self.get_metadata(map_name)
        if not metadata or 'lovot_slam' not in metadata:
            metadata['lovot_slam'] = {}
        self._update_basic_information_to_metadata(metadata)

        if source_list is not None:
            self._append_source_list_to_metadata(metadata, source_list)

        if write_statistics:
            # calculate statistics of the feature map and write it
            vertices = FeatureMapVertices.from_map_path(self._root / map_name)
            if vertices:
                statistics = vertices.get_height_statistics()
                self._append_statistics_to_metadata(metadata, *statistics)

        self._write_metadata(map_name, metadata)

    def create_building_flag(self, map_name: str):
        """Create flag file to know the map is currently building.
        """
        building_flag = self._root / map_name / MAP_BUILDING_FLAG_FILE
        if not building_flag.exists():
            building_flag.touch()

    def remove_bulding_flag(self, map_name: str):
        """Remove flag file to know the map is currently building.
        """
        building_flag = self._root / map_name / MAP_BUILDING_FLAG_FILE
        if building_flag.exists():
            building_flag.unlink()

    def is_map_building(self, map_name: str) -> bool:
        """Check the map is currentli building or not by the flag file.
        The map is under building if the file exists.
        """
        building_flag = self._root / map_name / MAP_BUILDING_FLAG_FILE
        return building_flag.exists()

    def get_source_map_list(self, map_name):
        metadata = self.get_metadata(map_name)
        if metadata:
            return metadata.get('lovot_slam', {}).get('source', [])
        return []

    def is_single_mission_map(self, map_name):
        source_list = self.get_source_map_list(map_name)
        return len(source_list) == 0

    def _get_mission_id_list_from_sensors_yaml(self, map_name):
        sensors_yaml = self._root / map_name / MAP_FEATUREMAP / MAP_FEATUREMAP_SENSORS_YAML
        try:
            with open(sensors_yaml, 'r') as f:
                metadata = yaml.safe_load(f)
                mission_sensors_associates = metadata.get('mission_sensors_associations', {})
                mission_ids = [item.get('mission_id') for item in mission_sensors_associates]
                return mission_ids
        except (OSError, yaml.YAMLError) as e:
            logger.error(f'failed to read {sensors_yaml}: {e}')
        return []

    def _get_mission_id_list_from_missions_yaml(self, map_name):
        missions_yaml = self._root / map_name / MAP_FEATUREMAP / MAP_FEATUREMAP_MISSIONS_YAML
        try:
            with open(missions_yaml, 'r') as f:
                metadata = yaml.safe_load(f)
                mission_ids = metadata.get('mission_ids', [])
                return mission_ids
        except (OSError, yaml.YAMLError) as e:
            logger.error(f'failed to read {missions_yaml}: {e}')
        return []

    def get_mission_id_list_from_feature_map(self, map_name):
        mission_ids = self._get_mission_id_list_from_missions_yaml(map_name)
        if mission_ids:
            return mission_ids
        logger.debug(f'No missions.yaml found in {map_name}, read mission ids from sensors.yaml')
        mission_ids = self._get_mission_id_list_from_sensors_yaml(map_name)
        return mission_ids

    def get_merged_single_mission_map_names(self, merged_map_name: str) -> List[str]:
        """Get the list of single mission maps that are merged into the merged map.

        This is similar to get_source_map_list,
        but this function returns the list of single mission maps that are actually merged into the feature map,
        while get_source_map_list just returns the list written in lovot_slam.yaml.
        Those would be different depending on the merge results.

        :param merged_map_name: name of the merged map
        :return: list of single mission map names
        """
        mission_ids = self.get_mission_id_list_from_feature_map(merged_map_name)

        all_single_mission_maps = self.get_recent_single_mission_maps(-1)
        merged_single_mission_map_names = []
        for map_name in all_single_mission_maps:
            mission_id = self.get_mission_id_list_from_feature_map(map_name)[0]
            if mission_id in mission_ids:
                merged_single_mission_map_names.append(map_name)
        return merged_single_mission_map_names

    def get_mapname_missionid_pairs(self, map_name: str) -> Dict[str, str]:
        """Get map_name and mission_id pairs from a map.
        :param map_name: name of the map
        :return: dictionary of map_name and mission_id pairs
        """
        single_mission_map_names = self.get_merged_single_mission_map_names(map_name)
        return {_map_name: self.get_mission_id_list_from_feature_map(_map_name)[0]
                for _map_name in single_mission_map_names}

    def check_feature_map(self, map_name: str, verbose: bool = False) -> bool:
        dir = self._root / map_name / MAP_FEATUREMAP
        missing_files = [file for file in MAP_FEATUREMAP_FILES if not (dir / file).exists()]
        if not missing_files:
            return True
        if verbose:
            logger.warning(f'Feature map {map_name} has missing files: {missing_files}')
        return False

    def check_dense_map(self, map_name: str, verbose: bool = False) -> bool:
        dir = self._root / map_name / MAP_2DMAP
        missing_files = [file for file in MAP_2DMAP_FILES if not (dir / file).exists()]
        if not missing_files:
            return True
        if verbose:
            logger.warning(f'Dense map {map_name} has missing files: {missing_files}')
        return False

    def get_map_scale(self, map_name: str) -> Optional[float]:
        data = self.get_metadata(map_name)
        return data['lovot_slam']['scale_odom_loc']

    def check_map_scale(self, map_name: str, verbose: bool = False) -> bool:
        try:
            scale = self.get_map_scale(map_name)
            if FEATURE_MAP_SCALE_LOWER_LIMIT < scale < FEATURE_MAP_SCALE_UPPER_LIMIT:
                return True
            if verbose:
                logger.warning(f'Invalid scale {scale} in {map_name}')
        except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
            if verbose:
                logger.warning(f'Failed to get scale from {map_name}: {e}')
        return False

    def get_occupancy_grid(self, map_name: str) -> OccupancyGrid:
        # TODO: memoize grid_map
        map_yaml_path = self._root / map_name / MAP_2DMAP / 'map.yaml'
        if not map_yaml_path.exists():
            raise RuntimeError(f'invalid map_yaml_path: {map_yaml_path}')

        grid_map = OccupancyGrid.from_yaml_file(map_yaml_path)
        return grid_map

    def check_map_size(self, map_name: str, verbose: bool = False) -> bool:
        try:
            grid_map = self.get_occupancy_grid(map_name)
            area_pixel_square = grid_map.get_area_pixel_square()
            # logger.debug(f"Area size of {map_name} is {area_pixel_square} [px2].")
            if area_pixel_square <= MAXIMUM_MAP_AREA_PIXEL_SQUARE:
                return True
        except RuntimeError as e:
            logger.warning(e)
            return False

        if verbose:
            logger.warning(f'Area size of {map_name} is too large: {area_pixel_square} [px2]')
        return False

    def get_map_statistics(self, map_name: str) -> Tuple[float, float]:
        data = self.get_metadata(map_name)
        height_mean = data['lovot_slam']['feature_map']['height_mean']
        height_std = data['lovot_slam']['feature_map']['height_std']
        return height_mean, height_std

    def check_map_statistics(self, map_name: str, verbose: bool = False) -> bool:
        if self.get_map_version(map_name) < 3:
            # version 1 and 2 maps don't have statistics
            return True
        try:
            height_mean, height_std = self.get_map_statistics(map_name)
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            return False

        if height_mean <= -MAP_STATS_HEIGHT_MEAN_LIMIT:
            if verbose:
                logger.warning(f'Height mean of {map_name} is too low: {height_mean}')
            return False
        if height_mean >= MAP_STATS_HEIGHT_MEAN_LIMIT:
            if verbose:
                logger.warning(f'Height mean of {map_name} is too high: {height_mean}')
            return False
        if height_std >= MAP_STATS_HEIGHT_STD_LIMIT:
            if verbose:
                logger.warning(f'Height std of {map_name} is too high: {height_std}')
            return False
        return True

    def get_map_version(self, map_name):
        try:
            data = self.get_metadata(map_name)
            version = data['lovot_slam']['version']
        except KeyError:
            return 1
        return version

    def get_map_stamp(self, map_name: str) -> Optional[float]:
        try:
            data = self.get_metadata(map_name)
            date = data['lovot_slam']['date']
            date = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
            stamp = datetime.datetime.timestamp(date)
        except (KeyError, ValueError):
            return None
        return stamp

    def is_newest_map_version(self, map_name):
        """Check if the existing map version is the newest."""
        return self.get_map_version(map_name) == MAP_VERSION

    def _get_map_content_path_list(self, map_name: str) -> List[pathlib.Path]:
        map_path = self._root / map_name
        path_list = []
        path_list += map_path.glob(f'{MAP_FEATUREMAP}/**/*')
        path_list += map_path.glob(f'{MAP_SUMMARYMAP}/**/*')
        path_list += map_path.glob(f'{MAP_2DMAP}/**/*')
        path_list += map_path.glob(f'{SLAM_YAML}')
        return path_list

    def create_md5sum_list(self, map_name: str) -> bool:
        path_list = self._get_map_content_path_list(map_name)
        map_path = self._root / map_name

        try:
            md5sum_dict = {str(path.relative_to(map_path)): get_file_md5sum(path)
                           for path in path_list if path.is_file()}
            with open(map_path / MAP_MD5SUM_YAML, 'w') as f:
                yaml.safe_dump(md5sum_dict, f)
        except EnvironmentError:
            logger.error('File access error while creating md5sum list.')
            return False
        return True

    def check_md5sum(self, map_name: str) -> bool:
        map_path = self._root / map_name
        try:
            with open(map_path / MAP_MD5SUM_YAML, 'r') as f:
                md5sum_dict = yaml.safe_load(f)
            unmatched_files = [rel_path
                               for rel_path, md5sum in md5sum_dict.items()
                               if md5sum != get_file_md5sum(map_path / rel_path)]
        except (OSError, yaml.YAMLError, AttributeError, TypeError) as e:
            logger.error(f'failed to read {map_path / MAP_MD5SUM_YAML}, possibly broken: {e}')
            return False
        if unmatched_files:
            logger.error(f'md5sum unmatch: {unmatched_files}')
            return False
        return True

    def _has_md5sum_list(self, map_name: str) -> bool:
        return (self._root / map_name / MAP_MD5SUM_YAML).is_file()

    def _verify_map_contents(self, map_name):
        """Verify map contents.
        Basically, check whether the required files exist.
        Some values (scale, size, stat) are also verified.
        """
        if not self.check_feature_map(map_name):
            logger.warning(f"{map_name} feature map is invalid.")
            return False
        if not self.check_dense_map(map_name):
            logger.warning(f"{map_name} dense map is invalid.")
            return False
        if not self.check_map_scale(map_name):
            logger.warning(f"{map_name} map scale is invalid.")
            return False
        if not self.check_map_size(map_name):
            logger.warning(f"{map_name} area size is invalid.")
            return False
        if not self.check_map_statistics(map_name):
            logger.warning(f"{map_name} statistics is invalid.")
            return False
        return True

    def check_map(self, map_name, ignore_md5sum: bool = False):
        """Check map by md5sum or contents verification.
        When md5sum list file exists, only the md5sum of each file is checked.
        This assumes that the contents of the map are already verified before calculating the md5sums.
        When md5sum list file doesn't exist, verify all contents of the map.
        """
        if self.is_map_building(map_name):
            return False
        if not ignore_md5sum and self._has_md5sum_list(map_name):
            if not self.check_md5sum(map_name):
                return False
        elif not self._verify_map_contents(map_name):
            return False
        map_version = self.get_map_version(map_name)
        if map_version not in SUPPROTED_MAP_VERSIONS:
            logger.warning(f"{map_name} map version {map_version} is not supported.")
            logger.warning(f"Supported versions: {SUPPROTED_MAP_VERSIONS}.")
            return False
        return True

    def create_map_name(self, source_list):
        now = datetime.datetime.now()
        map_name = now.strftime("%Y%m%d_%H%M%S")
        if len(source_list) > 0:
            key = hashlib.md5(''.join(source_list).encode()).hexdigest()
            map_name += '_' + key
        logger.debug(f'map_name: {map_name}')
        return map_name

    def get_full_path(self, map_name: str) -> pathlib.Path:
        return self._root / map_name

    def get_map_list(self, use_cache=True):
        """Get map list from the maps root directory (usually /data/localization/maps).
        Since this requires some computational resources to check map contents,
        cache the list and reuse it until timestamp of the contents in the maps root direcotry is unchanged.
        """
        # If the contents in the maps root direcotry have not been modified
        # since the last time the cache was updated,
        # reuse the chached list.
        timestamp_of_map_list = get_last_modified_time(self._root)
        if use_cache and self._map_list_cache.is_valid(timestamp_of_map_list):
            return self._map_list_cache.list

        # Scan the maps root directory and list maps.
        map_list = [str(path.name) for path in self._root.glob('*')
                    if path.is_dir() and self.check_map(path.name)]
        map_list = self._sort_map_list_by_date(map_list)

        # Update cache.
        self._map_list_cache = _MapListCache(map_list, timestamp_of_map_list)
        # logger.info(f'map_list is updated: {map_list}')
        return map_list

    def _sort_map_list_by_date(self, map_list):
        all_metadata = {}
        for map_name in map_list:
            meta_data = self.get_metadata(map_name)
            if meta_data:
                all_metadata[map_name] = meta_data
        sorted_maps = []
        for k, v in sorted(all_metadata.items(),
                           key=lambda x: x[1].get('lovot_slam', {}).get('date', '')):
            lovot_slam = v.get('lovot_slam', {})
            if lovot_slam.get('date', '') == '':
                continue
            sorted_maps.append(str(k))
        return sorted_maps

    def _extract_single_mission_maps(self, map_list):
        return [map_name for map_name in map_list if self.is_single_mission_map(map_name)]

    def _extract_merged_maps(self, map_list):
        single_mission_maps = self._extract_single_mission_maps(map_list)
        merged_maps = [item for item in map_list if item not in single_mission_maps]
        return merged_maps

    def get_recent_single_mission_maps(self, num):
        map_list = self.get_map_list()
        single_mission_maps = self._extract_single_mission_maps(map_list)
        sorted_single_mission_maps = self._sort_map_list_by_date(single_mission_maps)

        if num == -1 or len(sorted_single_mission_maps) <= num:
            return sorted_single_mission_maps
        selected_single_mission_maps = sorted_single_mission_maps[-num:]
        return selected_single_mission_maps

    def get_merged_map_list(self):
        """
        get merged map list in chronological order
        """
        map_list = self.get_map_list()
        merged_maps = self._extract_merged_maps(map_list)
        sorted_merged_maps = self._sort_map_list_by_date(merged_maps)
        return sorted_merged_maps

    def get_latest_merged_map(self):
        merged_map_list = self.get_merged_map_list()
        if len(merged_map_list) == 0:
            return ''
        return merged_map_list[-1]

    def get_recent_merged_maps(self, num: int, get_all: bool = False):
        merged_map_list = self.get_merged_map_list()

        if get_all or len(merged_map_list) <= num:
            return merged_map_list
        return merged_map_list[-num:]

    def get_maps_number_in_latest_merged_map(self):
        latest_merged_map = self.get_latest_merged_map()
        if not latest_merged_map:
            return 0
        merged_single_mission_map_names = self.get_merged_single_mission_map_names(latest_merged_map)
        return len(merged_single_mission_map_names)

    def set_map_to_redis_stm(self, map_name):
        map2d_path = self.get_full_path(map_name) / MAP_2DMAP / 'map.yaml'
        rosmap = RosMap.from_map_yaml(map_name, map2d_path)
        if rosmap.image is not None:
            self.redis_stm.hset(redis_keys.map, mapping=rosmap.to_dict_for_redis())

    def delete_map_from_redis_stm(self):
        self.redis_stm.delete(redis_keys.map)

    def delete_map_from_redis_ltm(self):
        self.redis_ltm.delete(redis_keys.map)

    def get_ros_map(self, map_name: str) -> RosMap:
        map2d_path = self.get_full_path(map_name) / MAP_2DMAP / 'map.yaml'
        return RosMap.from_map_yaml(map_name, map2d_path)

    def update_redis(self, rosmap: RosMap):
        if rosmap.image is not None:
            self.redis_ltm.hset(redis_keys.map, mapping=rosmap.to_dict_for_redis())

    # async def upload_map_to_cloud(self, rosmap: RosMap, ready: bool) -> bool:
    #     """ lovot-agent 経由で地図をクラウドにアップロードする """
    #     colony_id = get_colony_id()
    #     if colony_id is None:
    #         await anyio.sleep(0)
    #         logger.error("colony_id not found")
    #         return False
    #     event = rosmap.to_event_proto(colony_id)
    #     # the flag named 'completed' in the cloud is acutally 'ready' for use of the map,
    #     # and is used as a flag to allow users to set unwelcomed area.
    #     event.home_map.completed = ready
    #     data = event.SerializeToString()
    #     return await upload_data_to_cloud_async(CLOUD_HOME_MAP_EVENT_ENDPOINT, data)

    # async def reset_cloud_map(self) -> bool:
    #     colony_id = get_colony_id()
    #     if colony_id is None:
    #         await anyio.sleep(0)
    #         logger.error("colony_id not found")
    #         return False

    #     evt = HomeMapEvent(
    #         colony_id=colony_id,
    #         map_id=MAP_ID,
    #         event=HomeMapEvent.Event.home_map_reset,
    #         home_map=None
    #     )
    #     return await upload_data_to_cloud_async("navigation/home-map-event", evt.SerializePartialToString())

    def reset_map_accuracy(self):
        """Remove the accuracy_map directory.
        """
        logger.info("reset map accuracy")
        remove_directory_if_exists(data_directories.monitor)

    def has_summary_map(self, map_name: str) -> bool:
        summary_map_path = self._root / map_name / MAP_SUMMARYMAP
        for item in MAP_SUMMARYMAP_FILES:
            if not (summary_map_path / item).exists():
                return False
        return True


class SpotUtils:
    def __init__(self, map_utils):
        self.redis_ltm = create_ltm_client()
        self.map_utils = map_utils

    @staticmethod
    def pose_from_string(pose_str: str) -> Optional[np.ndarray]:
        """
        parse string "px,py,pz,ox,oy,oz,ow" to numpy ndarray
        """
        pose = list(map(float, pose_str.split(",")))
        if len(pose) != 7:
            logger.warning(f"invalid pose length {len(pose)} != 7")
            return None
        return np.array(pose)

    @staticmethod
    def pose_to_string(pose: np.ndarray) -> Optional[str]:
        return f"{pose[0]:.4f},{pose[1]:.4f},{pose[2]:.4f}," \
               f"{pose[3]:.4f},{pose[4]:.4f},{pose[5]:.4f},{pose[6]:.4f}"

    def get_spot_coordinate_from_redis(self, spot_name):
        pose_str = self.redis_ltm.hget(redis_keys.spot(spot_name), 'coordinate')
        return self.pose_from_string(pose_str) if pose_str else None

    def set_spot_coordinate_to_redis(self, spot_name, coordinate):
        spot_dict = {
            "map_id": MAP_ID,
            "name": spot_name,
            "coordinate": self.pose_to_string(coordinate)
        }
        self.redis_ltm.hset(redis_keys.spot(spot_name), mapping=spot_dict)

    def remove_spot_from_redis(self, spot_name):
        self.redis_ltm.delete(redis_keys.spot(spot_name))

    def get_unwelcomed_area_from_redis(self) -> List[Polygon]:
        unwelcomed_area_str = self.redis_ltm.get(redis_keys.unwelcomed_area)
        return decode_unwelcomed_area(unwelcomed_area_str) if unwelcomed_area_str else []

    def set_unwelcomed_area_to_redis(self, unwelcomed_area) -> Optional[str]:
        """ Set unwelcomed area to redis.
        Args:
            unwelcomed_area: List[Polygon] or None. If None, delete the key from redis.
        Returns:
            encoded: str or None. If unwelcomed_area is None, return None.
        """
        if unwelcomed_area is None:
            self.redis_ltm.delete(redis_keys.unwelcomed_area)
            return None
        encoded = json.dumps([shape.json_serialize() for shape in unwelcomed_area])
        self.redis_ltm.set(redis_keys.unwelcomed_area, encoded)
        return encoded

    def _transform_pose_to_the_latest_map(self, pose):
        """
        Args:
            pose: np.ndarray, [x, y, z, qx, qy, qz, qw]
        Returns:
            transformed_pose: np.ndarray, [x, y, z, qx, qy, qz, qw]
        """
        merged_map_list = self.map_utils.get_merged_map_list()
        if len(merged_map_list) < 2:
            logger.warning("transform pose: last map is deleted or broken; no transform")
            return pose
        orig_map_name = merged_map_list[-2]
        dest_map_name = merged_map_list[-1]
        transformed_pose = transform_pose_between_maps(self.map_utils.get_full_path(orig_map_name),
                                                       self.map_utils.get_full_path(dest_map_name),
                                                       pose)
        return transformed_pose

    def _transform_points_to_the_latest_map(self, points):
        merged_map_list = self.map_utils.get_merged_map_list()
        if len(merged_map_list) < 2:
            logger.warning("transform points: last map is deleted or broken; no transfrom")
            return points
        orig_map_path, dest_map_path = (
            self.map_utils.get_full_path(map_name) for map_name in merged_map_list[-2:])
        transformed_points = transform_points_between_maps(orig_map_path, dest_map_path, points)
        return transformed_points

    def transform_spot_and_register(self, spot_name):
        """
        Transform spot pose with the latest map, set it to redis and upload it to cloud.
        """
        pose = self.get_spot_coordinate_from_redis(spot_name)
        if pose is None:
            return
        pose = self._transform_pose_to_the_latest_map(pose)
        if pose is None:
            return
        self.set_spot_coordinate_to_redis(spot_name, pose)
        self.upload_spot_to_cloud(spot_name, pose)

    def transform_unwelcomed_area_to_the_latest_map(self, poly:Polygon) -> Polygon:
        """
        Move unwelcomed area according to the latest map.
        """
        logger.info("Transform unwelcomed area to the latest map:")
        logger.info(f"original: {poly.vertices}")
        
        center = poly.center
        relative_vertices = poly.vertices - center
            
        # padding 0.0 for the height(z) + (0, 0, 0, 1) for the quaternion
        composed = np.hstack([center, np.array([0.0, 0.0, 0.0, 0.0, 1.0])])
        transformed_pose = self._transform_pose_to_the_latest_map(composed)
            
        transformed_center = transformed_pose[:2]
        transformed_quat = transformed_pose[3:]
        euler = R.from_quat(transformed_quat).as_euler('zyx')
        yaw = euler[0]
        yaw_rotation_matrix = R.from_euler('z', yaw).as_matrix()
        rotated_pts = (yaw_rotation_matrix[:2, :2] @ relative_vertices.T).T
            
        transformed_pts = rotated_pts + transformed_center
            
        logger.info(f"transformed: {transformed_pts}")
        return Polygon(transformed_pts)

    def transform_unwelcomed_area_and_register(self):
        """
        Transform unwelcomed area with the latest map, set it to redis, and uplaod it to cloud.
        """
        unwelcomed_area = self.get_unwelcomed_area_from_redis()
        transformed_unwelcomed_area = []
        if unwelcomed_area:
            for poly in unwelcomed_area:
                transformed_poly = self.transform_unwelcomed_area_to_the_latest_map(poly)
                transformed_unwelcomed_area.append(transformed_poly)

            logger.info(f"Transformed {len(transformed_unwelcomed_area)} unwelcomed area")
            data = self.set_unwelcomed_area_to_redis(transformed_unwelcomed_area)
            self.upload_unwelcomed_area_to_cloud(data)

    def upload_spot_to_cloud(self, name, coordinate):
        """ lovot-agent 経由でSpot情報をクラウドにアップロードする """
        colony_id = get_colony_id()
        if colony_id is None:
            logger.error("colony_id not found")
            return False
        logger.debug(f"Uploading spot {name}:{coordinate} to cloud.")
        event = SpotEvent(
            colony_id=colony_id,
            map_id=MAP_ID,
            spot_name=name,
            event=SpotEvent.spot_updated,
            spot=Spot(
                colony_id=colony_id,
                map_id=MAP_ID,
                name=name,
                coordinate=Coordinate(
                    px=coordinate[0],
                    py=coordinate[1],
                    pz=coordinate[2],
                    ox=coordinate[3],
                    oy=coordinate[4],
                    oz=coordinate[5],
                    ow=coordinate[6],
                ),
            ),
        )
        data = event.SerializeToString()
        return upload_data_to_cloud(CLOUD_SPOT_EVENT_ENDPOINT, data)

    def upload_unwelcomed_area_to_cloud(self, unwelcomed_area_data):
        """ lovot-agent 経由で来ないでエリア情報をクラウドにアップロードする """
        colony_id = get_colony_id()
        if colony_id is None:
            logger.error("colony_id not found")
            return False
        logger.debug("Uploading UnwelcomedArea to cloud.")
        event = UnwelcomedAreaEvent(
            colony_id=colony_id,
            map_id=MAP_ID,
            area_id=AREA_ID,
            event=UnwelcomedAreaEvent.unwelcomed_area_updated,
            area=UnwelcomedArea(
                area_id=AREA_ID,
                colony_id=colony_id,
                map_id=MAP_ID,
                data=unwelcomed_area_data,
                ),
            )
        data = event.SerializeToString()
        return upload_data_to_cloud(CLOUD_UNWELCOMED_AREA_EVENT_ENDPOINT, data)


class MapSetUtils:
    def __init__(self, mapset_root: pathlib.Path) -> None:
        self._root = mapset_root

    def create_mapset(self, mapset_name: str) -> None:
        mapset_dir = self._root / mapset_name
        mapset_dir.mkdir(parents=True, exist_ok=True)
        # TODO: copy necessary files to the new mapset directory

    def does_mapset_exist(self, mapset_name: str) -> bool:
        return (self._root / mapset_name).exists()

    def change_mapset(self, mapset_name: str) -> None:
        """Change the current mapset to the specified mapset.
        if the mapset doesn't exist, create it.
        """
        assert data_directories.data_root.is_symlink(), \
            f"{data_directories.data_root} should be a symlink"
        if not self.does_mapset_exist(mapset_name):
            self.create_mapset(mapset_name)

        # change the symlink to the new mapset directory
        data_directories.data_root.unlink()
        data_directories.data_root.symlink_to(self._root / mapset_name)

    def generate_new_mapset_name(self) -> str:
        """Generate a unique mapset name based on the current time.
        """
        current_time = int(time.time())
        while True:
            mapset_name = f"mapset_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(current_time))}"
            if not self.does_mapset_exist(mapset_name):
                return mapset_name
            current_time += 1
