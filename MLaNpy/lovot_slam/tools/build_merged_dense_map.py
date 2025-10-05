"""This script is used to build the merged dense map.

1. build the dense map for each mission (run octomap)
   2D dense map and path map are generated
2. merge the 2D maps
3. filter the merged 2D map
"""
import logging
from typing import List

import trio

from lovot_slam.env import data_directories
from lovot_slam.flags.debug_params import PARAM_BUILD_DENSE_MAP_RATE
from lovot_slam.map_build.rosbag_info import is_camera_info_recorded
from lovot_slam.tools.merge_2d_maps import (FILTER_DIAMETER_PIXELS, UNKNOWN_EROSION_RADIUS, CostMapsMerger,
                                            convert_to_occupancy_grid)
from lovot_slam.tools.utils import SubprocessRunner

_logger = logging.getLogger(__name__)


class MergedDenseMapBuilder:
    def __init__(self, map_name: str, source_maps: List[str], mission_ids: List[str],
                 machine_type: str = "coro1") -> None:
        self._map_name = map_name
        self._source_maps = source_maps
        self._mission_ids = mission_ids
        self._machine_type = machine_type

    async def build(self) -> bool:
        # run octomap for each mission
        for source_map, mission_id in zip(self._source_maps, self._mission_ids):
            if not await self._run_single_octomap(source_map, mission_id):
                return False

        # merge the 2d maps (dense map and path map)
        self._merge_2d_maps()
        return True

    def _merge_2d_maps(self) -> None:
        map_dirs = [data_directories.maps / self._map_name / source_map
                    for source_map in self._source_maps]

        _logger.info(f'merging {len(map_dirs)} 2D dense and path maps')
        merger = CostMapsMerger.from_paths(map_dirs)
        merged_cost_map = merger.merge()

        grid_map = convert_to_occupancy_grid(merged_cost_map)
        grid_map.filter(radius=(FILTER_DIAMETER_PIXELS/2)*grid_map.resolution)
        # 床に囲まれた未探索領域を床で埋める
        grid_map.erode_unknown(UNKNOWN_EROSION_RADIUS)
        grid_map.save(data_directories.maps / self._map_name / '2d_map' / 'map.yaml')

    async def _run_single_octomap(self, source_map: str, mission_id: str) -> bool:
        rosbag = data_directories.bags / f'{source_map}_converted.bag'
        output_dense_map = data_directories.maps / self._map_name / source_map / '2d_map' / 'map'
        output_path_map = data_directories.maps / self._map_name / source_map / 'path_map' / 'map'
        publish_camera_info = not await is_camera_info_recorded(rosbag)
        cmd = ['roslaunch',
               'lovot_mapping',
               'octomap.launch',
               f'maps_root:={data_directories.maps}',
               f'map_name:={self._map_name}',
               f'mission_id:={mission_id}',
               f'output_dense_map:={output_dense_map}',
               f'output_path_map:={output_path_map}',
               f'rosbag:={rosbag}',
               f'rosbag_playback_rate:={PARAM_BUILD_DENSE_MAP_RATE}',
               f'machine_type:={self._machine_type}',
               f'publish_camera_info:={"true" if publish_camera_info else "false"}']
        return SubprocessRunner(cmd).run() == 0


def main(args):
    map_name = args.map_name
    source_maps = args.source_maps
    mission_ids = args.mission_ids

    builder = MergedDenseMapBuilder(map_name, source_maps, mission_ids,
                                    machine_type=args.machine_type)
    trio.run(builder.build)
