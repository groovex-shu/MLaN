from typing import List, Optional

import numpy as np
import pytest

from lovot_slam.env import DataDirectories
from lovot_slam.exploration.area_search import AreaSearchBase, _PositionalHistory, _PositionalHistoryItem
from lovot_slam.redis import create_ltm_client
from lovot_slam.utils.map_utils import MapUtils

from ..feature_map.feature_map_vertices_data import TEST_DATASET_PATH, setup_dataset, teardown_dataset

LTM_TEST_AREA_SEARCH_HISTORY_KEY = "slam:test:area_search:history"


class MockAreaSearch(AreaSearchBase):
    def __init__(self) -> None:
        super().__init__(LTM_TEST_AREA_SEARCH_HISTORY_KEY)

    def find(self, update_history: bool = True) -> Optional[np.ndarray]:
        centroid_list = [np.array([0., 0.])]
        return self._choose_target(centroid_list, update_history=update_history)


@pytest.fixture
def setup_for_area_search():
    setup_dataset()
    ltm_client = create_ltm_client()
    ltm_client.delete(LTM_TEST_AREA_SEARCH_HISTORY_KEY)
    yield
    ltm_client.delete(LTM_TEST_AREA_SEARCH_HISTORY_KEY)
    teardown_dataset()


@pytest.mark.parametrize('history,target,expected_items', [
    (_PositionalHistory([]), np.array([0, 0]), []),
    (_PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                         _PositionalHistoryItem(200, np.array([5., 5.]))]),
     np.array([0, 0]),
     [_PositionalHistoryItem(100, np.array([0., 0.]))]),
    (_PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                         _PositionalHistoryItem(200, np.array([5., 5.]))]),
     np.array([0.2, 0.2]),
     [_PositionalHistoryItem(100, np.array([0., 0.]))]),
    (_PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                         _PositionalHistoryItem(200, np.array([5., 5.]))]),
     np.array([4.8, 4.8]),
     [_PositionalHistoryItem(200, np.array([5., 5.]))]),
    (_PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                         _PositionalHistoryItem(150, np.array([0.5, 0.])),
                         _PositionalHistoryItem(200, np.array([5., 5.]))]),
     np.array([0.25, 0.]),
     [_PositionalHistoryItem(100, np.array([0., 0.])), _PositionalHistoryItem(150, np.array([0.5, 0.]))]),
])
def test_history_get_near_positions(
        history: _PositionalHistory, target: np.ndarray, expected_items: List[_PositionalHistory]):
    items = history.get_near_items(target, radius_th=0.5)
    assert items == expected_items


@pytest.mark.parametrize('history_items,potisions_to_keep,radius_th,expected_result', [
    ([_PositionalHistoryItem(100, np.array([0.0, 0.0])), _PositionalHistoryItem(100, np.array([1.0, 1.0]))],
     [np.array([0.0, 0.0])], 1.0, _PositionalHistory([_PositionalHistoryItem(100, np.array([0.0, 0.0]))])),
    ([_PositionalHistoryItem(100, np.array([0.0, 0.0])), _PositionalHistoryItem(100, np.array([1.0, 1.0]))],
     [np.array([1.0, 1.0])], 1.0, _PositionalHistory([_PositionalHistoryItem(100, np.array([1.0, 1.0]))])),
    ([_PositionalHistoryItem(100, np.array([0.0, 0.0])), _PositionalHistoryItem(100, np.array([1.0, 1.0]))],
     [np.array([2.0, 2.0])], 1.0, _PositionalHistory([])),
])
def test_history_filter(history_items, potisions_to_keep, radius_th, expected_result):
    history = _PositionalHistory(history_items)
    filtered_history = history.filter(potisions_to_keep, radius_th)

    assert len(filtered_history.items) == len(expected_result.items)
    for i in range(len(filtered_history.items)):
        assert filtered_history.items[i] == expected_result.items[i]


def test_json_serialize():
    history = _PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                                  _PositionalHistoryItem(200, np.array([5., 5.]))])
    assert history.json_serialize() == '[[100, 0.0, 0.0], [200, 5.0, 5.0]]'


def test_json_deserialize():
    expected = _PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                                   _PositionalHistoryItem(200, np.array([5., 5.]))])
    history = _PositionalHistory.json_deserialize(
        '[[100, 0.0, 0.0], [200, 5.0, 5.0]]')

    assert len(history._items) == len(expected._items)
    assert history._items[0] == expected._items[0]
    assert history._items[1] == expected._items[1]


def test_append():
    history = _PositionalHistory([_PositionalHistoryItem(100, np.array([0., 0.])),
                                  _PositionalHistoryItem(200, np.array([5., 5.]))])

    new_item = _PositionalHistoryItem(300, np.array([10., 10.]))
    history.append(new_item)

    assert len(history._items) == 3
    assert history._items[2] == new_item


@pytest.mark.parametrize("merged_maps,orig_points,dest_points", [
    (['map_a', 'map_b'],  # map origin is shifted by (1.0, 2.0) and then rotated by +90 deg
     [np.array([1.1, 0.1]), np.array([1.0, 0.0])],
     [np.array([0.9, 3.1]), np.array([1.0, 3.0])]),
    (['map_b', 'map_c'],  # reverse of the above
     [np.array([0.9, 3.1]), np.array([1.0, 3.0])],
     [np.array([1.1, 0.1]), np.array([1.0, 0.0])]),
    (['map_a', 'map_c'],  # origins are the same (no transformation)
     [np.array([1.1, 0.1]), np.array([1.0, 0.0])],
     [np.array([1.1, 0.1]), np.array([1.0, 0.0])]),
    (['map_c', 'map_b'],  # the closest key frame in map_c is not in map_b
     [np.array([11.0, 10.0])],
     [np.array([11.0, 13.0])]),
    (['map_a', 'map_b'], [], []),
    (['map_a'],  # no transformation occurs, with only a map
     [np.array([1.0, 0.0]), np.array([1.0, 3.0])],
     [np.array([1.0, 0.0]), np.array([1.0, 3.0])]),
    ([],  # no transformation occurs, without maps
     [np.array([1.0, 0.0]), np.array([1.0, 3.0])],
     [np.array([1.0, 0.0]), np.array([1.0, 3.0])]),
])
def test_transform_points_between_maps(monkeypatch, setup_for_area_search, merged_maps, orig_points, dest_points):
    monkeypatch.setattr(DataDirectories, "DATA_ROOT", TEST_DATASET_PATH)

    area_search = MockAreaSearch()
    area_search._history = _PositionalHistory([_PositionalHistoryItem(0., point) for point in orig_points])

    # mock the original map and the destination map to transform
    monkeypatch.setattr(MapUtils, "get_merged_map_list", lambda self: merged_maps)

    area_search.update_history_points_on_map_changes()

    assert all(np.allclose(area_search._history.items[i].position, point, atol=1e-05)
               for i, point in enumerate(dest_points))
