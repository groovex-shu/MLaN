"""
This script is used to calculate and optimize the scale of the map.
"""
import logging
import time
from typing import List

from lovot_slam.tools.optimize_scale import optimize_scale
from lovot_slam.tools.utils import SubprocessRunner

_logger = logging.getLogger(__name__)


def calculate_map_scale(map_name: str, source_maps: List[str], mission_ids: List[str]) -> bool:
    cmd = ['roslaunch',
           'lovot_mapping',
           'scale_map.launch',
           f'map_name:={map_name}',
           f'source_maps:={" ".join(source_maps)}',
           f'mission_ids:={" ".join(mission_ids)}']
    return SubprocessRunner(cmd).run() == 0


def main(args):
    map_name = args.map_name
    source_maps = args.source_maps
    mission_ids = args.mission_ids

    # calculate scale before the optimization
    calculate_map_scale(map_name, source_maps, mission_ids)

    # optimize the map with the distance edges
    start = time.time()
    optimize_scale(map_name)
    _logger.info(f"scale optimization duration: {int(time.time() - start)} sec")

    # recalculate scale after the optimization
    calculate_map_scale(map_name, source_maps, mission_ids)
