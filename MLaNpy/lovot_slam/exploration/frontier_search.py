from logging import getLogger
from typing import Optional

import numpy as np
from scipy import ndimage

from lovot_map.accuracy_map import CostMap
from lovot_map.occupancy_grid import OccupancyGrid
from lovot_map.planner import FrontierFinder
from lovot_map.rosmap import create_kernel

from lovot_slam.exploration.accuracy_map_util import COSTMAP_NO_OBSTACLE, COSTMAP_UNKNOWN, match_map_size
from lovot_slam.exploration.area_search import AreaSearchBase


logger = getLogger(__name__)

LTM_FRONTIER_HISTORY_KEY = 'slam:exploration:frontier:history'

# using value smaller than 0.2 m could cause false positives of the new frontiers.
# it requires that the good accuracy area wider than 0.7 m traverse the wall area
# to get the new frontier.
# 0.7 = (0.2 + 0.15) * 2, while 0.2 is for accuracy map erosion, 0.15 is for wall dilation
ACCURACY_MAP_DILATION_KERNEL_SIZE = 0.2


def _get_centroid(points: np.ndarray) -> Optional[np.ndarray]:
    if len(points) == 0:
        return None
    ret = sum(points) / len(points)
    return np.array(ret)


def mask_obstacle_with_accuracy_map(grid_map: OccupancyGrid, accuracy_map: CostMap) -> OccupancyGrid:
    """Mask obstacles (walls) with the area whose accuracy is good.
    Remove walls where the lovot seems to be passing through there with high probability.
    The area seems to be a door which was closed while making the current map, but was opened after that.
    :accuracy_map: the accuracy map with which to mask the obstacles
    """
    cost_map = CostMap.from_occupancy_grid(grid_map)
    match_map_size(cost_map, accuracy_map, fill_a=COSTMAP_UNKNOWN, fill_b=0.)

    # filter accuracy map
    # use only good accuracy area (more than 50 % probability inside 0.2 m radius circle)
    # and erode the area in order to prevent erode acutal walls.
    # NOTE: argument robot_radius is jsut a radius for kernel, does not mean robot's radius
    struct = create_kernel(robot_radius=ACCURACY_MAP_DILATION_KERNEL_SIZE,
                           map_resolution=cost_map.resolution)
    mask = ndimage.binary_erosion(accuracy_map._data > 0.,  # use only good accuracy
                                  structure=struct)

    # mask obstacles with the filtered accuracy map
    data = np.where(np.logical_and(mask, cost_map._data < 0.0), COSTMAP_NO_OBSTACLE, cost_map._data)
    return CostMap(data, cost_map.origin, cost_map.resolution).to_occupancy_grid()


class FrontierSearch(AreaSearchBase):
    ROBOT_RADIUS = 0.15
    # the following value had been relaxed from 1.5 m,
    # because the positions in the history is transformed following the map keyframes on map update
    # and it improves the accuracy of the comparison.
    SAME_POSITION_DISTANCE_THRESHOLD = 0.75
    # kernel size of the unknown area erosion (to ignore small hole of unknown in the floor)
    UNKNOWN_EROSION_KERNEL_SIZE = 0.4

    def __init__(self) -> None:
        super().__init__(LTM_FRONTIER_HISTORY_KEY)

    def find(self, grid_map: OccupancyGrid, start: np.ndarray, update_history: bool = True) -> Optional[np.ndarray]:
        # NOTE: unknown erosion may cause inconsistency with navigation
        # (fail to plan a path to the frontier in a narrow path)
        # But it must be rare, because unknown erosion is already done in the map creation process
        # and the problematic unknown pixels in narrow paths should be already removed.

        # apply unknown closing, to ignore small hole of unknown in the floor
        grid_map.apply_closing_unknown(start, self.UNKNOWN_EROSION_KERNEL_SIZE)
        # dilate obstacles to avoid finding frontiers passing through narrow paths
        grid_map.dilate_obstacle(self.ROBOT_RADIUS)

        # find frontier candidates
        frontier_finder = FrontierFinder(grid_map)
        frontier_list = frontier_finder.find(start)

        # sort by distance from the start position
        centroid_list = [centroid for frontier in frontier_list
                         if (centroid := _get_centroid(frontier)) is not None]
        centroid_list = sorted(centroid_list, key=lambda centroid: np.linalg.norm(centroid - start))

        return self._choose_target(centroid_list, update_history=update_history)
