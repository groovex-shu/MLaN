import json
import time
from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import List, Optional

import attr
import numpy as np
from attr import attrib, attrs

from lovot_slam.env import data_directories
from lovot_slam.feature_map.feature_map_vertices import transform_points_between_maps
from lovot_slam.redis.clients import create_ltm_client
from lovot_map.utils.map_utils import MapUtils

_logger = getLogger()


@attrs(auto_attribs=True, frozen=True)
class _PositionalHistoryItem:
    """2D positional information with timestamp.
    """
    timestamp: float
    position: np.ndarray = attrib(eq=attr.cmp_using(eq=np.array_equal))

    def __attrs_post_init__(self):
        assert self.position.shape[0] == 2

    def as_list(self) -> List[float]:
        return [self.timestamp, self.position[0], self.position[1]]

    @classmethod
    def from_list(cls, listed_items: List[float]) -> '_PositionalHistoryItem':
        return cls(listed_items[0], np.array([listed_items[1], listed_items[2]]))

    @classmethod
    def create_form_position(cls, position: np.ndarray) -> '_PositionalHistoryItem':
        return cls(time.time(), position)

    def get_distance(self, target: np.ndarray) -> float:
        assert target.shape == self.position.shape
        return np.linalg.norm(target - self.position)


class _PositionalHistory:
    """Container to store positional history items,
    with some helper apis to get near items, or to filter items by positional condition.
    """

    def __init__(self, items: List[_PositionalHistoryItem]) -> None:
        self._items = items

    @property
    def items(self) -> List[_PositionalHistoryItem]:
        return self._items

    def json_serialize(self) -> str:
        return json.dumps([item.as_list() for item in self._items])

    @classmethod
    def json_deserialize(cls, json_str: str) -> '_PositionalHistory':
        try:
            dict = json.loads(json_str)
        except json.JSONDecodeError:
            _logger.warning('JSON decode error')
            return None

        try:
            return _PositionalHistory([_PositionalHistoryItem.from_list(items_list) for items_list in dict])
        except TypeError as e:
            _logger.warning(e)
        except IndexError as e:
            _logger.warning(e)
        return None

    def get_near_items(self, target: np.ndarray, radius_th=0.5) -> List[_PositionalHistoryItem]:
        """Get items in the history within a specific region: circular area centered from the target position and radius.
        :param target: target position, center of the cicular region
        :param radius_th: distance threshold [m], radius of the circle
        :return: list of history items
        """
        list_near = [history_item for history_item in self.items
                     if history_item.get_distance(target) < radius_th]
        return list_near

    def filter(self, positions_to_keep: List[np.ndarray], radius_th: float = 1.0) -> '_PositionalHistory':
        """Filter the history by keeping items which are close to the positions in positions_to_keep.
        :param positions_to_keep: keep original positions which are close to positions_to_keep
        :param radius_th: threshold for the filter [m], radius of the circle
        :return: filtered history
        """
        new_items = [history_item for history_item in self._items
                     if any(history_item.get_distance(position) < radius_th
                            for position in positions_to_keep)]
        return self.__class__(new_items)

    def append(self, item: _PositionalHistoryItem) -> None:
        """Append the given item to the list.
        This just appends, without any filtering.
        """
        self._items.append(item)


class AreaSearchBase(metaclass=ABCMeta):
    """
    探索対象となる場所を見つけるためのclass

    最新の情報をもとにして探索場所(frontierや自己位置精度が低い場所など)を見つけ、 
    過去に複数回探索したことがある場所は無視した上で、探索すべき場所を返す。
    探索した場所は履歴に残され、複数回探索したのにも関わらず残っている場所は諦める。

    find_target_areaで返した時点で履歴を更新するため、実際には探索できなかった場合も対象となることに注意。
    実際にはExploreが中断されたり、Explore側のナビゲーションが失敗するなどして、対象のエリアを探索できないこともあるが、
    現状はその場合でも履歴には「探索した」として残る。
    TODO: Exploreの結果を見て履歴を更新する

    また、最新の情報では対象として見つからなかったエリアは履歴から削除する。
    (例えばfrontierのケースだと、前回の探索でfrontierが無くなった場合など)

    履歴はLTMに保存される
    """

    def __init__(self, ltm_key: str) -> None:
        self._ltm_client = create_ltm_client()
        self._ltm_key = ltm_key

        self._map_utils = MapUtils(data_directories.maps, data_directories.bags)

        self._history: Optional[_PositionalHistory] = None
        self._load_history_from_ltm()

    def _load_history_from_ltm(self):
        """Reads JSON string-represented history from redis and decode as dict.
        ex. "[[100.1, 0.0, 1.0], [200.0, 1.0, 0.0], ...]"
        """
        self._history = None
        json_str = self._ltm_client.get(self._ltm_key)
        if not json_str:
            return

        self._history = _PositionalHistory.json_deserialize(json_str)

    def _store_history_to_ltm(self):
        if self._history:
            json_str = self._history.json_serialize()
            self._ltm_client.set(self._ltm_key, json_str)

    def update_history_points_on_map_changes(self):
        """Transform every positions in the history
        from the coords of the previous map to the coords of the latest map.
        This is supposed to be called after the map update.
        """
        if not self._history:
            return

        # Get the previous map and the latest map.
        merged_map_list = self._map_utils.get_merged_map_list()
        if len(merged_map_list) < 2:
            return
        _logger.info(f'transform area history from {merged_map_list[-2]} to {merged_map_list[-1]}')
        orig_map_path, dest_map_path = (
            self._map_utils.get_full_path(map_name) for map_name in merged_map_list[-2:])

        # Transform all positions in the history.
        transformed_history = _PositionalHistory([])
        for item in self._history.items:
            position = transform_points_between_maps(orig_map_path, dest_map_path, item.position[np.newaxis, :2])
            transformed_history.append(_PositionalHistoryItem(item.timestamp, position.reshape([2])))

        # Update and store
        self._history = transformed_history
        self._store_history_to_ltm()

    def reset(self):
        """Reset history and delete the corresponding redis key.
        """
        self._history = None
        self._ltm_client.delete(self._ltm_key)

    def _choose_target(self, area_candidates: List[np.ndarray], update_history: bool) -> Optional[np.ndarray]:
        """Choose one target area from given area candiates, considering history.
        At first, filter the history, keeping only the areas that are close to one of the candidates.
        Then, select one target area from the candidates.
        If there are more than two areas in the history that are close to the candidate area, ignore the candidate.
        Finally update the history by appending the selected target area.
        """
        # Filtering the history
        new_history = _PositionalHistory([])
        if self._history:
            new_history = self._history.filter(area_candidates)

        # Select a target
        target = None
        for area in area_candidates:
            if len(new_history.get_near_items(area)) >= 2:
                continue
            target = area
            break

        # Update history for next exploration
        if update_history:
            # Append the selected target area
            if target is not None:
                new_history.append(_PositionalHistoryItem.create_form_position(target))
            self._history = new_history
            self._store_history_to_ltm()

        return target

    @abstractmethod
    def find(self, update_history: bool = True) -> Optional[np.ndarray]:
        """Find one target area, considering history.
        Implement like follows:
        area_candidates = self._find_area_candidates()
        return self._filter_target(area_candidates, update_history=update_history)
        """
        raise NotImplementedError
