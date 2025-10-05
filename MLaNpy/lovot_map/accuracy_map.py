from logging import getLogger
from typing import Tuple

import numpy as np

from lovot_map.occupancy_grid import OccupancyGrid

_logger = getLogger(__name__)

MAX_MAP_SIZE_HALF_M = 20
PROBABILITY_LIMIT_MARGIN = 1.e-6  # margin to 0 or 1 (within 0~1)
LOG_PROBABILITY_ABS_LIMIT = np.log((1. - PROBABILITY_LIMIT_MARGIN) / PROBABILITY_LIMIT_MARGIN)


def logarithm_probability(prob):
    # clip to avoid zero division (within 0 ~ 1 with some margin)
    clipped = np.clip(prob, PROBABILITY_LIMIT_MARGIN, 1. - PROBABILITY_LIMIT_MARGIN)
    return np.log(clipped / (1 - clipped))


def unlogarithm_probability(log_prob):
    # clip to avoid nan (output should be within 0 ~ 1 with some margin)
    clipped = np.clip(log_prob, -LOG_PROBABILITY_ABS_LIMIT, LOG_PROBABILITY_ABS_LIMIT)
    return np.exp(clipped) / (1 + np.exp(clipped))


class CostMap(object):
    def __init__(self, data, origin, resolution):
        """
        :data: 2d array with any types (float, int)
        :origin: 2d position of image origin (left, bottom) relative to the map origin
            this should be an integer multiple of the resolution.
        :resolution: resolution of a pixel [m/px] normally 0.05 m/px
        """
        self._data = data
        self._origin = origin
        self._resolution = resolution

    @property
    def data(self):
        return self._data

    @property
    def origin(self):
        return self._origin

    @property
    def resolution(self):
        return self._resolution

    def world_to_map(self, position):
        return ((position - self._origin) / self._resolution + 0.5).astype(np.int32)[..., ::-1]

    def map_to_world(self, points):
        return np.round(points[..., ::-1] + self._origin / self._resolution) * self._resolution

    def get_boundary(self):
        leftbottom = self.map_to_world(np.array([0, 0]))
        righttop = self.map_to_world(np.array(self._data.shape))
        return leftbottom[0], leftbottom[1], righttop[0], righttop[1]

    def _floor_with_grid(self, metric):
        return np.floor(metric / self._resolution) * self._resolution

    def _ceil_with_grid(self, metric):
        return np.ceil(metric / self._resolution) * self._resolution

    def _round_with_grid(self, metric):
        return np.round(metric / self._resolution) * self._resolution

    def extend_including(self, left, bottom, right, top,
                         fill=0.0):
        """extend the map including the given boundary
        left, bottom, right, top in metric
        """
        new_l = min(self._origin[0],
                    self._round_with_grid(left))
        new_b = min(self._origin[1],
                    self._round_with_grid(bottom))
        new_r = max(self._origin[0] + self._data.shape[1] * self._resolution,
                    self._round_with_grid(right))
        new_t = max(self._origin[1] + self._data.shape[0] * self._resolution,
                    self._round_with_grid(top))
        width = int(round((new_r - new_l) / self._resolution))
        height = int(round((new_t - new_b) / self._resolution))

        old_origin = self._origin
        self._origin = np.array([new_l, new_b])
        # copy the old data to the new container
        index = self.world_to_map(old_origin)
        data = np.full([height, width], fill, dtype=self._data.dtype)
        data[index[0]:index[0]+self._data.shape[0],
             index[1]:index[1]+self._data.shape[1]] = np.copy(self._data)
        self._data = data

    def to_occupancy_grid(self):
        img = np.floor(unlogarithm_probability(self._data) * 255).astype(np.uint8)
        img = np.flipud(img)
        # TODO: Crop to nonzero area
        return OccupancyGrid(img, self._resolution, self._origin)

    @classmethod
    def from_occupancy_grid(cls, grid_map):
        data = logarithm_probability(grid_map.img.astype(np.float64) / 255)
        data = np.flipud(data)
        return cls(data=data,
                   origin=grid_map.origin,
                   resolution=grid_map.resolution)


class AccuracyMap(CostMap):
    _PROBABILITY_LOWER_LIMIT = 0.1
    _PROBABILITY_UPPER_LIMIT = 0.99

    def __init__(self, data: np.ndarray, origin: np.ndarray, resolution: float,
                 decay_half_life: float = 12*60*60):
        super(AccuracyMap, self).__init__(data, origin, resolution)

        self._decay_half_life = decay_half_life
        decay_rate = 0.999
        self._decay_period = decay_half_life * np.log(decay_rate) / np.log(0.5)
        self._last_time_decayed = None
        _logger.debug(f'AccuracyMap decay period: {self._decay_period} sec.')

    def decay(self, timestamp: float) -> None:
        """Decay map data by specified half-life.
        The actual decay doesn't occur everytime but with a certain period.
        The decay_period is determined by given decay_half_life and decay_rate, in __init__.
        """
        if not self._last_time_decayed:
            self._last_time_decayed = timestamp
            return

        period = timestamp - self._last_time_decayed
        if period > self._decay_period:
            decay_rate = 0.5 ** (period / self._decay_half_life)
            _logger.info(f'AccuracyMap decay by: {decay_rate}.')
            self._data *= decay_rate
            self._last_time_decayed = timestamp

    def update(self, position: np.ndarray, probability: float, radius: float,
               apply_ratio: float = 1.0):
        """Update the accuracy map by
        :param position: specified position [m]
        :param probability: probability of localization inside the circle of the radius
        :param radius: of the circle [m]
        :param apply_ratio: ratio (0 ~ 1.0) to apply it on the cost map
        """
        if probability < self._PROBABILITY_LOWER_LIMIT:
            return

        # TODO: compare with the localization map to check position
        if MAX_MAP_SIZE_HALF_M < np.max(np.abs(position)):
            return

        # extend map size to plot the new area
        self.extend_including(position[0] - 2 * radius,
                              position[1] - 2 * radius,
                              position[0] + 2 * radius,
                              position[1] + 2 * radius)

        # position and radius in pixels
        index = self.world_to_map(position)
        radius_px = radius / self._resolution

        # calc square of distance from (i, j)
        dist_i_sqr = (np.arange(self._data.shape[0]) - index[0]) ** 2
        dist_j_sqr = (np.arange(self._data.shape[1]) - index[1]) ** 2
        dist_sqr = dist_i_sqr[:, None] + dist_j_sqr[None, :]

        cost = logarithm_probability(probability)
        self._data[dist_sqr < radius_px**2 + 0.5] += apply_ratio * cost
        self._data = np.clip(self._data,
                             logarithm_probability(self._PROBABILITY_LOWER_LIMIT),
                             logarithm_probability(self._PROBABILITY_UPPER_LIMIT))

    def __add__(self, other):
        """Add probabilities.
        shape and origin should be the same.
        """
        assert self._data.shape == other.data.shape
        assert np.all(np.isclose(self._origin, other.origin))
        data = self._data + other.data
        data = np.clip(data,
                       logarithm_probability(AccuracyMap._PROBABILITY_LOWER_LIMIT),
                       logarithm_probability(AccuracyMap._PROBABILITY_UPPER_LIMIT))
        return self.__class__(data, self._origin, self._resolution,
                              decay_half_life=self._decay_half_life)

    def average(self):
        """Get average accuracy over the map area.
        """
        average_log_prob = 0.
        if self._data.shape[0] and self._data.shape[1]:
            average_log_prob = np.mean(self._data)
        return unlogarithm_probability(average_log_prob)

    def histogram(self, bucket_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get histogram with the given bucket count (0.0 ~ 1.0).
        The initial value is 0.5, so ignore 0.5 to calculate the histogram.
        :param bucket_count: count of bucket. step = 1.0 / (bucket_count - 1) (so, if you need step of 0.1, set 11)
        :return: Tuple of the bucket (length: bucket_count) and the histogram (length: bucket_count - 1)
        """
        bucket = np.linspace(0.0, 1.0, bucket_count)
        hist, _ = np.histogram(unlogarithm_probability(self.data[self.data != 0.0]), bucket)
        return bucket, hist

    def to_occupancy_grid(self):
        # discretize 0~1.0 into 0~65534 with 0.5 as base so that 0.5 = 32767
        # NOTE: default value of the accuracy map is 0.5, so we want 0.5 to be restored exactly as 0.5
        img = np.round(unlogarithm_probability(self._data) * 65534).astype(np.uint16)
        img = np.flipud(img)
        # TODO: Crop to nonzero area
        return OccupancyGrid(img, self._resolution, self._origin)

    @classmethod
    def from_occupancy_grid(cls, grid_map: OccupancyGrid) -> 'AccuracyMap':
        """Convert from OccupancyGrid
        accepts 8 bit or 16 bit
        """
        _logger.debug(f'AccuracyMap load from {grid_map.img.dtype}.')

        def convert(grid_map: OccupancyGrid, fullscale: int) -> 'AccuracyMap':
            data = logarithm_probability(grid_map.img.astype(np.float64) / fullscale)
            data = np.flipud(data)
            return cls(data=data,
                       origin=grid_map.origin,
                       resolution=grid_map.resolution)

        if grid_map.img.dtype == np.uint8:
            # restore 0~1.0 from 0~254 with 127 as base so that 0.5 = 127
            # NOTE: default value of the accuracy map is 0.5, so we want 0.5 to be restored exactly as 0.5
            return convert(grid_map, 254)
        elif grid_map.img.dtype == np.uint16:
            # restore 0~1.0 from 0~65534 with 32767 as base so that 0.5 = 32767
            # NOTE: default value of the accuracy map is 0.5, so we want 0.5 to be restored exactly as 0.5
            return convert(grid_map, 65534)
