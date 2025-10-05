import logging
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from scipy import ndimage

from lovot_map.accuracy_map import (LOG_PROBABILITY_ABS_LIMIT, CostMap, logarithm_probability,
                                        unlogarithm_probability)
from lovot_map.occupancy_grid import OccupancyGrid

from lovot_slam.exploration.accuracy_map_util import match_map_size

logger = logging.getLogger(__name__)

UNKNOWN_EROSION_RADIUS = 0.2
# diameter should be an odd number, not to shift image
FILTER_DIAMETER_PIXELS = 3.0


class MapType(Enum):
    DENSE_MAP = '2d_map'
    PATH_MAP = 'path_map'


# occupied probability of 0.5 (50%) means unobserved state of the cell (unknown)
# occupied cell (obstacle) has a lager value(>50%), and free cell (floor) has a smaller value (<50%)
FREE_PROBABILITIES = {
    MapType.DENSE_MAP: 0.2,
    # in order not to erode the wall, reduce weighting of the path_map
    MapType.PATH_MAP: 0.3
}
OCCUPIED_PROBABILITY = 0.65


def convert_to_cost_map(grid_map: OccupancyGrid,
                        occupied_probability: float = OCCUPIED_PROBABILITY,
                        free_probability: float = 0.4) -> CostMap:
    img = np.where(grid_map.img == OccupancyGrid.OCCUPIED_CODE,
                   logarithm_probability(occupied_probability), 0.0)
    img = np.where(grid_map.img == OccupancyGrid.FREE_CODE,
                   logarithm_probability(free_probability), img)
    img = np.where(grid_map.img == OccupancyGrid.UNKNOWN_CODE,
                   logarithm_probability(0.5), img)
    img = np.flipud(img)
    return CostMap(img, grid_map.origin, grid_map.resolution)


def convert_to_occupancy_grid(cost_map: CostMap,
                              free_thresh: float = 0.5,
                              occupied_thresh: float = 0.5) -> OccupancyGrid:
    probability_map = unlogarithm_probability(cost_map.data)
    img = np.zeros(probability_map.shape, dtype=np.uint8)
    img = np.where(probability_map < free_thresh, OccupancyGrid.FREE_CODE,
                   np.where(probability_map > occupied_thresh, OccupancyGrid.OCCUPIED_CODE, OccupancyGrid.UNKNOWN_CODE))
    img = np.flipud(img.astype(np.uint8))

    return OccupancyGrid(img, cost_map.resolution, cost_map.origin[:2])


def load_cost_map(yaml_path: Path, free_probability: float) -> CostMap:
    grid_map = OccupancyGrid.from_yaml_file(yaml_path)
    return convert_to_cost_map(grid_map, free_probability=free_probability)


class CostMapsMerger:
    def __init__(self, dense_maps: List[CostMap], path_maps: List[CostMap]) -> None:
        self._dense_maps = dense_maps
        self._path_maps = path_maps

    @classmethod
    def from_paths(cls, paths: List[Path]) -> 'CostMapsMerger':
        """Create CostMapsMerger from paths
        each directory should contain both "2d_map" and "path_map" directories
        """
        dense_maps: List[CostMap] = []
        path_maps: List[CostMap] = []
        for map_path in paths:
            assert map_path.is_dir()

            dense_maps.append(load_cost_map(map_path / MapType.DENSE_MAP.value / 'map.yaml',
                                            FREE_PROBABILITIES[MapType.DENSE_MAP]))
            path_maps.append(load_cost_map(map_path / MapType.PATH_MAP.value / 'map.yaml',
                                           FREE_PROBABILITIES[MapType.PATH_MAP]))
        return cls(dense_maps, path_maps)

    def _filter_path_maps(self) -> List[CostMap]:
        """Filter only the path maps
        Apply gaussian filter in order to reduce the effect of the peripheral areas,
        in order not to erode the walls.
        """
        new_path_maps: List[CostMap] = []

        for path_map in self._path_maps:
            sigma_meter = 0.05
            filtered = ndimage.gaussian_filter(path_map.data, sigma_meter / path_map.resolution)
            filtered = np.where(path_map.data < 0, filtered, 0.0)  # mask with the free area
            new_path_maps.append(CostMap(filtered, path_map.origin, path_map.resolution))
        return new_path_maps

    def merge(self) -> CostMap:
        cost_maps = self._dense_maps + self._filter_path_maps()

        # make the target map size cover all maps' area
        target_map = CostMap(np.empty((0, 0), dtype=np.float64),
                             np.zeros(2), self._dense_maps[0].resolution)
        for cost_map in cost_maps:
            match_map_size(target_map, cost_map)

        # create the container of the merged map
        merged_data = np.zeros(target_map.data.shape, dtype=np.float64)
        merged_origin = target_map.origin
        resolution = target_map.resolution
        logger.info(f'merged map size is {merged_data.shape} and origin is {merged_origin}')

        # sum of logarithmized probability of each cell
        for cost_map in cost_maps:
            match_map_size(target_map, cost_map)
            merged_data += cost_map.data
        merged_data = np.clip(merged_data,
                              -LOG_PROBABILITY_ABS_LIMIT, LOG_PROBABILITY_ABS_LIMIT)
        return CostMap(merged_data, merged_origin, resolution)


def main(args):
    map_dirs = [Path(map_dir) for map_dir in args.maps]
    logger.info(f'merging {len(map_dirs)} 2D dense and path maps')
    merger = CostMapsMerger.from_paths(map_dirs)
    merged_cost_map = merger.merge()

    grid_map = convert_to_occupancy_grid(merged_cost_map)
    grid_map.filter(radius=(FILTER_DIAMETER_PIXELS/2)*grid_map.resolution)
    # 床に囲まれた未探索領域を床で埋める
    grid_map.erode_unknown(UNKNOWN_EROSION_RADIUS)
    grid_map.save(Path(args.output) / 'map.yaml')
