import datetime
import json
import time
from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, List, NamedTuple, Optional

import numpy as np
import prometheus_client
import anyio
from attr import attrib, attrs

from lovot_map.accuracy_map import AccuracyMap, CostMap
from lovot_map.occupancy_grid import OccupancyGrid

from lovot_slam.env import data_directories
from lovot_slam.exploration.accuracy_map_util import load_accuracy_maps_and_merge
from lovot_slam.exploration.frontier_search import FrontierSearch, mask_obstacle_with_accuracy_map
from lovot_slam.exploration.low_accuracy_area_search import LowAccuracyAreaSearch
from lovot_slam.flags.debug_params import (PARAM_EXPLORATION_STATUS_INTERVAL_UNTIL_ACCURACY_CHECK,
                                           PARAM_EXPLORATION_STATUS_RECHECK_PERIOD_AFTER_SUSPENSION,
                                           PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN,
                                           PARAM_MISSIONS_COUNT_TO_GIVE_UP_EXPLORATION)
from lovot_slam.redis.clients import create_ltm_client
from lovot_slam.redis.keys import COLONY_LOVOTS_KEY
from lovot_map.utils.map_utils import MapUtils

logger = getLogger(__name__)

EXPLORATION_STATUS_KEY = 'slam:exploration:status'
MAP_METRICS_HISTORY_KEY = 'slam:map:metrics:history'

_missions_count_metric = prometheus_client.Gauge(
    'localization_exploration_status_missions_count', 'missions count')
_frontier_remained_metric = prometheus_client.Gauge(
    'localization_exploration_status_frontier_remained', 'frontier remained')
_low_accuracy_area_remained_metric = prometheus_client.Gauge(
    'localization_exploration_status_low_accuracy_area_remained', 'low accuracy area remained')
_can_explore_metric = prometheus_client.Gauge(
    'localization_exploration_status_can_explore', 'can explore')
_initially_triggered_metric = prometheus_client.Gauge(
    'localization_exploration_status_initially_triggered', 'triggered for the first time')

_map_area_metric = prometheus_client.Gauge(
    'localization_exploration_status_map_area', 'map area',
    labelnames=['gen_from_latest', 'code'])
_map_accuracy_metric = prometheus_client.Gauge(
    'localization_exploration_status_last_map_accuracy', 'map accuracy of the last map',
    labelnames=['gen_from_latest'])

_accuracy_map_histogram_metric = prometheus_client.Gauge(
    'localization_accuracy_map_bucket', 'accuracy map histogram',
    labelnames=['le'])


@attrs(frozen=True)
class _MapArea:
    free: float = attrib(default=0.0)
    occupied: float = attrib(default=0.0)
    unknown: float = attrib(default=0.0)

    @classmethod
    def from_occupancy_grid(cls, grid_map: OccupancyGrid) -> '_MapArea':
        return cls(grid_map.get_free_area(),
                   grid_map.get_occupied_area(),
                   grid_map.get_unknown_area())

    def set_metric(self, gen_from_latest: int) -> None:
        """
        :param gen_from_latest: latest is 0, previous is 1, ...
        """
        _map_area_metric.labels(gen_from_latest=gen_from_latest, code='free').set(self.free)
        _map_area_metric.labels(gen_from_latest=gen_from_latest, code='occupied').set(self.occupied)
        _map_area_metric.labels(gen_from_latest=gen_from_latest, code='unknown').set(self.unknown)


@attrs
class _MapMetrics:
    """Metrics associated with a map
    """
    map_name: str = attrib()  # fixed value
    map_area: _MapArea = attrib()  # fixed value
    # TODO: 平均, 移動平均も取れるようにする?
    last_map_accuracy: float = attrib()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'map_name': self.map_name,
            'map_area': [self.map_area.free, self.map_area.occupied, self.map_area.unknown],
            'last_map_accuracy': self.last_map_accuracy,
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> Optional['_MapMetrics']:
        try:
            return cls(dict['map_name'],
                       _MapArea(*dict['map_area']),
                       dict['last_map_accuracy'])
        except KeyError as e:
            logger.warning(f'{e} not found in json string')
        except ValueError as e:
            logger.warning(f'{e}')
        return None

    def set_metric(self, gen_from_latest: int) -> None:
        self.map_area.set_metric(gen_from_latest)
        _map_accuracy_metric.labels(gen_from_latest=gen_from_latest).set(self.last_map_accuracy)


def _load_map_metrics_list(json_str: str) -> List[_MapMetrics]:
    try:
        dict = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        logger.warning('failed to decode json string')
        return None

    map_metrics_list: List[_MapMetrics] = []
    for dict_single in dict:
        metrics = _MapMetrics.from_dict(dict_single)
        if metrics:
            map_metrics_list.append(metrics)
    return map_metrics_list


class _MapMetricsHistory:
    """History of metrics for each generation of maps.
    This class stores metrics of the recent maps (number of maps is defined in _HISTORY_COUNT),
    It saves the history to LTM when updated, and load it from LTM on start.
    Only metrics of the latest two maps are set to prometheus client.
    """
    _HISTORY_COUNT = 4
    # 1 means uploading only the latest (current) map's metrics,
    # and 2 means the latest and the previous
    _GENERATION_TO_UPLOAD = 2

    def __init__(self) -> None:
        self._map_utils = MapUtils(data_directories.maps, data_directories.bags)
        self._redis_ltm = create_ltm_client()

        self._history: Deque[_MapMetrics] = deque(maxlen=_MapMetricsHistory._HISTORY_COUNT)
        self._load_history()
        self._set_metrics()

    def _store_history(self) -> None:
        """Store the history to redis.
        """
        json_str = json.dumps([map_metrics.to_dict() for map_metrics in self._history])
        self._redis_ltm.set(MAP_METRICS_HISTORY_KEY, json_str)

    def _load_history(self) -> None:
        """Load the history from redis.
        """
        json_str = self._redis_ltm.get(MAP_METRICS_HISTORY_KEY)
        if json_str:
            self._history = deque(_load_map_metrics_list(json_str),
                                  maxlen=_MapMetricsHistory._HISTORY_COUNT)
        if self._history:
            logger.info(f'map metrics history has been loaded as {self._history}')

    def _get_map_area(self, map_name: str) -> Optional[_MapArea]:
        try:
            grid_map = self._map_utils.get_occupancy_grid(map_name)
        except RuntimeError as e:
            logger.warning(e)
            return None
        return _MapArea.from_occupancy_grid(grid_map)

    def _set_metrics(self) -> None:
        newest_order_history = list(reversed(self._history))
        for i in range(_MapMetricsHistory._GENERATION_TO_UPLOAD):
            if len(newest_order_history) <= i:
                break
            last_map_metrics = newest_order_history[i]
            last_map_metrics.set_metric(i)

    def update(self, accuracy_map: Optional[AccuracyMap]) -> None:
        latest_map_name = self._map_utils.get_latest_merged_map()
        if not latest_map_name:
            return

        accuracy = 0.0
        if accuracy_map:
            accuracy = accuracy_map.average()

        if len(self._history) > 0 and self._history[-1].map_name == latest_map_name:
            # if the map is already in the queue
            self._history[-1].last_map_accuracy = accuracy
        else:
            # insert the latest map to the history queue
            map_area = self._get_map_area(latest_map_name)
            if map_area:
                self._history.append(_MapMetrics(latest_map_name, map_area, accuracy))

        self._set_metrics()
        self._store_history()


class ExplorationStatus(NamedTuple):
    """
    status of the exploration in rooms (stored with timestamp).
    there are three status:
    - ready
      ready to use the map (e.g. for navigation, set unwelcomed area, etc).
      this flag is used in smartphone application as a flag whether users can set unwelcomed area,
      which is named 'completed' flag.
    - well explored
      the rooms are well explored (seems to cover all area, without any frontiers).
      minimum missions count in the latest merged map is also satisfied.
    - gave up
      the rooms seem to be too wide, and we should give up exploration.
      some frontiers remains, but the missions count in the latest merged map exceeds limits.

    'ready' and the other two may coexist.
    'well explored' and 'gave up' are exclusive.

    exploration will be suspended, if the status is either 'well explored' or 'gave up' condition.
    """
    missions_count: int  # missions count in the latest merged map
    frontier_remained: bool  # if frontier remains in the latest merged map (masked with the accuracy map)
    low_accuracy_area_remained: bool  # if low accuracy area remains in the latest map which is older than a month
    initially_triggered: bool  # True when can_explore is continuously True from the last map reset
    timestamp: float  # timestamp of the status

    # "ready" means that the map is ready to use (e.g. for App map opertations)
    def is_ready(self) -> bool:
        return (self.missions_count >= PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN) or \
            (not self.frontier_remained and self.missions_count > 1)

    def is_well_explored(self) -> bool:
        return (not self.frontier_remained
                and self.missions_count >= PARAM_MERGE_MAPS_MINIMUM_MISSIONS_TO_MAINTAIN)

    def is_gave_up(self) -> bool:
        return (self.frontier_remained
                and self.missions_count > PARAM_MISSIONS_COUNT_TO_GIVE_UP_EXPLORATION)

    def can_explore(self) -> bool:
        return (not self.is_well_explored() and not self.is_gave_up()) or self.low_accuracy_area_remained

    def to_json(self) -> str:
        return json.dumps({
            'missions_count': self.missions_count,
            'frontier_remained': self.frontier_remained,
            'low_accuracy_area_remained': self.low_accuracy_area_remained,
            'initially_triggered': self.initially_triggered,
            'timestamp': self.timestamp,
        })

    @classmethod
    def from_json(cls, json_str: str) -> Optional['ExplorationStatus']:
        try:
            dict = json.loads(json_str)
        except json.decoder.JSONDecodeError:
            logger.warning('failed to decode json string')
            return None

        try:
            return cls(dict['missions_count'],
                       dict['frontier_remained'],
                       dict['low_accuracy_area_remained'],
                       dict.get('initially_triggered', False),
                       dict['timestamp'])
        except KeyError as e:
            logger.warning(f'{e} not found in json string')
        except ValueError as e:
            logger.warning(f'{e}')
        return None

    def is_equal(self, target: 'ExplorationStatus') -> bool:
        """Compare values except timestamp with target status.
        """
        return (self.missions_count == target.missions_count
                and self.frontier_remained == target.frontier_remained
                and self.low_accuracy_area_remained == target.low_accuracy_area_remained)

    def __str__(self) -> str:
        dt = datetime.datetime.fromtimestamp(self.timestamp)
        return (f'ExplorationStatus(missions_count={self.missions_count}, '
                f'frontier_remained={self.frontier_remained}, '
                f'low_accuracy_area_remained={self.low_accuracy_area_remained}, '
                f'timestamp={dt.strftime("%Y/%m/%d %H:%M:%S")})'
                f'{", ready" if self.is_ready() else ""}'
                f'{", well explored" if self.is_well_explored() else ""}'
                f'{", gave up" if self.is_gave_up() else ""}'
                f'{", can explore" if self.can_explore() else ""}'
                f'{", initially triggered" if self.initially_triggered else ""}')

    def set_metric(self) -> None:
        _missions_count_metric.set(self.missions_count)
        _frontier_remained_metric.set(1 if self.frontier_remained else 0)
        _low_accuracy_area_remained_metric.set(1 if self.low_accuracy_area_remained else 0)
        _can_explore_metric.set(1 if self.can_explore() else 0)
        _initially_triggered_metric.set(1 if self.initially_triggered else 0)


class ExplorationStatusMonitor:
    _MONITOR_PERIOD_SEC = 30 * 60  # 30 minutes

    def __init__(self) -> None:
        self._map_utils = MapUtils(data_directories.maps, data_directories.bags)
        self._redis_ltm = create_ltm_client()
        self._exploration_status = None
        self._map_metrics_history = _MapMetricsHistory()

        # Exploration area search
        self._frontier_search = FrontierSearch()
        self._low_accuracy_area_search = LowAccuracyAreaSearch()

        self._load_exploration_status_from_redis()
        # set force_update=False,
        # because we want to preserve the timestamp if the status has not been changed.
        # on the first boot or the first time fater software update, it would be updated anyway.
        self.update(force_update=False)

    @property
    def status(self) -> ExplorationStatus:
        return self._exploration_status

    def _load_accuracy_map(self) -> Optional[AccuracyMap]:
        ghost_ids = self._redis_ltm.smembers(COLONY_LOVOTS_KEY)
        return load_accuracy_maps_and_merge(ghost_ids)

    def find_new_frontier(self, map_name: str, update_history: bool = False,
                          start: np.ndarray = np.array((0., 0.))) -> Optional[np.ndarray]:
        """Find new frontiers by masking the map with the accuracy map.
        :param map_name: target map name
        :param update_history: update history by this seesion if True
        :param start: start position of finding the frontiers
        :return: frontier centroid if found, else None
        """
        # TODO: cache start position and reuse it
        map_path = self._map_utils.get_full_path(map_name)
        map_yaml = map_path / '2d_map' / 'map.yaml'
        if not map_yaml.exists():
            return None

        grid_map = OccupancyGrid.from_yaml_file(map_yaml)
        if not grid_map:
            return None

        accuracy_map = self._load_accuracy_map()
        if accuracy_map:
            grid_map = mask_obstacle_with_accuracy_map(grid_map, accuracy_map)
        return self._frontier_search.find(grid_map, start, update_history=update_history)

    def find_low_accuracy_area(self, map_name: str, update_history: bool = False) -> Optional[np.ndarray]:
        """Find a low accuracy area with the accuracy map.
        :param map_name: target map name
        :param update_history: update history by this seesion if True
        :return: a centroid representing a low accuracy area
        """
        map_path = self._map_utils.get_full_path(map_name)
        # load accuracy map
        accuracy_map = self._load_accuracy_map()
        if not accuracy_map:
            logger.info('failed to find low accuracy area: accuracy map does not exist')
            return None
        # load 2d map as a cost map
        try:
            grid_map = OccupancyGrid.from_yaml_file(map_path / '2d_map' / 'map.yaml')
            cost_map = CostMap.from_occupancy_grid(grid_map)
        except RuntimeError as e:
            logger.warning(e)
            return None

        return self._low_accuracy_area_search.find(accuracy_map, cost_map, update_history=update_history)

    def _check_low_accuracy_area_remained(self) -> bool:
        map_name = self._map_utils.get_latest_merged_map()
        map_stamp = self._map_utils.get_map_stamp(map_name)

        # tested map should be older than a certain period
        if (not map_stamp or
                time.time() - map_stamp < PARAM_EXPLORATION_STATUS_INTERVAL_UNTIL_ACCURACY_CHECK):
            return False

        # chck if low accuracy area remains
        low_accuracy_area = self.find_low_accuracy_area(map_name, update_history=False)
        if low_accuracy_area is None:
            return False

        logger.info(f'low accuracy area remained in {map_name}')
        return True

    def _check_exploration_status(self) -> ExplorationStatus:
        """Check exploration status which is obtained by the latest map and the accuracy map.
        """
        logger.info('checking exploration status')
        missions_count = 0
        frontier_remained = True
        low_accuracy_area_remained = False
        initially_triggered = True

        map_name = self._map_utils.get_latest_merged_map()
        if map_name:
            # missions count in the merged map
            missions_count = self._map_utils.get_maps_number_in_latest_merged_map()
            logger.info(f'missions count in {map_name} is {missions_count}')

            # find new frontier by masking it with the accuracy map
            frontier = self.find_new_frontier(map_name)
            if frontier is not None:
                frontier_remained = True
                logger.info(f'one of the remained frontier in {map_name} is {frontier}')
            else:
                frontier_remained = False
                logger.info(f'no frontier remained in {map_name}')

            # check if low accuracy area remained
            low_accuracy_area_remained = self._check_low_accuracy_area_remained()

            # check exploration trigger state (initial or continous-update)
            if self._exploration_status and \
                    (not self._exploration_status.initially_triggered
                     or not self._exploration_status.can_explore()):
                # once can_explore flag falls,
                # initially_triggered does not return to True until the map is reset
                initially_triggered = False

        # update status
        timestamp = time.time()
        return ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained,
                                 initially_triggered, timestamp)

    def _store_exploration_status(self, exploration_status: ExplorationStatus):
        self._exploration_status = exploration_status
        self._redis_ltm.set(EXPLORATION_STATUS_KEY, exploration_status.to_json())
        logger.info(f'exploration status has been stored as {self._exploration_status}')
        self._exploration_status.set_metric()

    def _load_exploration_status_from_redis(self):
        """Load exploration status from redis to self._exploration_status.
        """
        self._exploration_status = None
        json_str = self._redis_ltm.get(EXPLORATION_STATUS_KEY)
        if json_str:
            self._exploration_status = ExplorationStatus.from_json(json_str)
        if self._exploration_status:
            logger.info(f'exploration status has been loaded as {self._exploration_status}')
            self._exploration_status.set_metric()

    def update(self, force_update: bool = True):
        """Update (check and store) the exploration status.
        this is supposed to be called when the map is updated or is reset.
        :param force_update: store the status even when it is not changed if True,
            else store the status only when it is changed from the current one.
            when you want to preserve the timestamp if the status has not been changed,
            set froce_update == False.
        :return: True if updated, else False
        """
        exploration_status = self._check_exploration_status()
        if (force_update
                or not self._exploration_status
                or not exploration_status.is_equal(self._exploration_status)):
            self._store_exploration_status(exploration_status)

        accuracy_map = self._load_accuracy_map()
        self._map_metrics_history.update(accuracy_map)

    def transform_area_histories(self):
        """Transform the search area histories.
        This is supposed to be called after the map update.
        """
        self._frontier_search.update_history_points_on_map_changes()
        self._low_accuracy_area_search.update_history_points_on_map_changes()

    def reset(self):
        """Reset status.
        This is supposed to be called AFTER all maps are cleared.
        """
        self._frontier_search.reset()
        self._low_accuracy_area_search.reset()

        self.update()

    def _update_accuracy_map_metrics(self, accuracy_map: AccuracyMap) -> None:
        bucket, histogram = accuracy_map.histogram(11)
        sum = 0.0
        for le, value in zip(bucket[1:], histogram):
            sum += value
            _accuracy_map_histogram_metric.labels(le=f'{le:.2f}').set(sum)

    async def _monitor(self):
        """Periodically monitor the exploration status.
        This monitor has responsible to monitor the exploration status,
        only after the exploration status has became 'well_explored' or 'gave_up'
        (in which status exploration is suspended).
        The status will be rechecked when the specified period (e.g. 1 week) has been passed
        after the last update of the status.
        And the status will be updated only when it has changed.
        """
        while True:
            accuracy_map = self._load_accuracy_map()
            self._map_metrics_history.update(accuracy_map)
            if accuracy_map:
                self._update_accuracy_map_metrics(accuracy_map)

            current_time = time.time()
            elapsed_time = current_time - self._exploration_status.timestamp
            # TODO: randomize the period (1 week +/- 24 hours)
            if elapsed_time < PARAM_EXPLORATION_STATUS_RECHECK_PERIOD_AFTER_SUSPENSION:
                # if it is within the specified period (e.g. 1 week) since the last update of the status,
                # the status will not be rechecked.
                await anyio.sleep(self._MONITOR_PERIOD_SEC)
                continue

            exploration_status = self._check_exploration_status()
            if not exploration_status.is_equal(self._exploration_status):
                self._store_exploration_status(exploration_status)
            else:
                logger.info('exploration status has not been updated')

            await anyio.sleep(self._MONITOR_PERIOD_SEC)

    async def run(self):
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._monitor)
