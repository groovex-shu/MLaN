import time
from logging import getLogger
from typing import Iterable, Optional

from lovot_map.accuracy_map import AccuracyMap, CostMap, logarithm_probability
from lovot_map.occupancy_grid import OccupancyGrid

from lovot_slam.env import data_directories
from lovot_slam.utils.file_util import remove_directory_if_exists

COSTMAP_OBSTACLE = logarithm_probability(OccupancyGrid.OCCUPIED_CODE / 255.)
COSTMAP_UNKNOWN = logarithm_probability(OccupancyGrid.UNKNOWN_CODE / 255.)
COSTMAP_NO_OBSTACLE = logarithm_probability(OccupancyGrid.FREE_CODE / 255.)

logger = getLogger(__name__)

ELAPSED_TIME_TO_IGNORE_SEC = 24 * 60 * 60


def match_map_size(cost_map_a: CostMap, cost_map_b: CostMap,
                   fill_a: float = 0., fill_b: float = 0.):
    """match size and origin of the maps, so that they can be merged into one map
    """
    boundary_a = cost_map_a.get_boundary()
    boundary_b = cost_map_b.get_boundary()

    cost_map_a.extend_including(*boundary_b, fill=fill_a)
    cost_map_b.extend_including(*boundary_a, fill=fill_b)


def load_accuracy_map(ghost_id: Optional[str] = None) -> Optional[AccuracyMap]:
    """Load an accuracy map from monitor directory.
    There is an accuracy map in monitor/<ghost_id>/accuracy_map for each lovot.
    :ghost_id: target ghost_id. if unspecified, load from monitor/accuracy_map
    :return: loaded accuracy map if successfully loaded else None
    """
    accuracy_map_dir = data_directories.monitor / 'accuracy_map'
    map_yaml = accuracy_map_dir / 'map.yaml'
    if ghost_id:
        map_yaml = data_directories.monitor / ghost_id / 'accuracy_map' / 'map.yaml'
    logger.debug(f'loading accuracy map from: {map_yaml}')

    # Ignore old accuracy maps
    if not map_yaml.exists() or \
            time.time() - map_yaml.stat().st_mtime > ELAPSED_TIME_TO_IGNORE_SEC:
        return None

    try:
        grid_map = OccupancyGrid.from_yaml_file(map_yaml)
        accuracy_map = AccuracyMap.from_occupancy_grid(grid_map)
        return accuracy_map
    except RuntimeError as e:
        logger.warning(f'Failed to load accuracy map from {ghost_id}: {e}')
        remove_directory_if_exists(accuracy_map_dir)
        return None


def load_accuracy_maps_and_merge(ghost_ids: Iterable[str]) -> Optional[AccuracyMap]:
    """Load accuracy maps and merge them.
    :return: the merged accuracy map if some of them can be loaded else None
    """
    merged_accuracy_map: Optional[AccuracyMap] = None
    for ghost_id in ghost_ids:
        accuracy_map = load_accuracy_map(ghost_id)
        if not accuracy_map:
            continue

        if not merged_accuracy_map:
            merged_accuracy_map = accuracy_map
        else:
            match_map_size(merged_accuracy_map, accuracy_map)
            merged_accuracy_map = merged_accuracy_map + accuracy_map

    return merged_accuracy_map
