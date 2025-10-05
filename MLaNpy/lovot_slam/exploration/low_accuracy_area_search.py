from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage

from lovot_map.accuracy_map import AccuracyMap, CostMap, logarithm_probability

from lovot_slam.exploration.accuracy_map_util import COSTMAP_UNKNOWN, match_map_size
from lovot_slam.exploration.area_search import AreaSearchBase

LTM_LOW_ACCURACY_HISTORY_KEY = 'slam:exploration:low_accuracy:history'


def _accuracy_closing_filter(accuracy_map: AccuracyMap) -> AccuracyMap:
    """closing filter with dilation then erosion
    :return: filtered accuracy map
    """
    data = ndimage.grey_dilation(accuracy_map.data, size=[10, 10])
    data = ndimage.grey_erosion(data, size=[10, 10])
    return AccuracyMap(data, accuracy_map.origin, accuracy_map.resolution)


def _mask_accuracy_map_with_obstacle(accuracy_map: AccuracyMap, cost_map: CostMap) -> AccuracyMap:
    """Mask accuracy map with obstacle.
    fill region of obstacles with 0 in accuracy map.
    :return: masked accuracy map
    """
    data = np.where(cost_map.data < 0.0, 0.0, accuracy_map.data)
    return AccuracyMap(data, accuracy_map.origin, accuracy_map.resolution)


def _get_low_accuracy_areas_centroid(accuracy_map: AccuracyMap, threshold: float) -> List[np.ndarray]:
    """Get centroid position list of low accuracy areas.
    :accuracy_map: map to search
    :threshold: logarithm threshold of accuracy which should be lower than 0.0
    :return: centroid list
    """
    map_of_low_accuracy = (accuracy_map.data < threshold).astype(np.uint8)
    n_labels, labeled_image = cv2.connectedComponents(map_of_low_accuracy)
    target_centroid = []
    # skip the 1st label which is the base area
    for i in range(1, n_labels):
        centroid = np.mean(np.where(labeled_image == i), axis=1)
        centroid = accuracy_map.map_to_world(centroid)
        target_centroid.append(centroid)

    return target_centroid


class LowAccuracyAreaSearch(AreaSearchBase):
    ACCURACY_THRESHOLD = logarithm_probability(0.49)  # should be lower than 0.5

    def __init__(self) -> None:
        super().__init__(LTM_LOW_ACCURACY_HISTORY_KEY)

    def find(self, accuracy_map: AccuracyMap, cost_map: CostMap, update_history: bool = True) -> Optional[np.ndarray]:
        """Find a low accuracy area to explore.
        :accuracy_map: map describing localization accuracy
        :cost_map: map describing obstacles and floor
        :return: 2D position (2d array) if area found, else None
        """
        match_map_size(cost_map, accuracy_map, fill_a=COSTMAP_UNKNOWN, fill_b=0.)
        filtered_accuracy_map = _accuracy_closing_filter(accuracy_map)
        masked_accuracy_map = _mask_accuracy_map_with_obstacle(filtered_accuracy_map, cost_map)
        centroid_list = _get_low_accuracy_areas_centroid(masked_accuracy_map, self.ACCURACY_THRESHOLD)
        return self._choose_target(centroid_list, update_history=update_history)
