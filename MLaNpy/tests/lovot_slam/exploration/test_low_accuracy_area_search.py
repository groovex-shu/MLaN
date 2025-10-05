import json
from typing import List

import numpy as np
import pytest

from lovot_slam.exploration.area_search import _PositionalHistory, _PositionalHistoryItem
from lovot_slam.exploration.low_accuracy_area_search import LTM_LOW_ACCURACY_HISTORY_KEY, LowAccuracyAreaSearch
from lovot_slam.redis import create_ltm_client


@pytest.fixture
def setup_ltm():
    ltm_client = create_ltm_client()
    ltm_client.delete(LTM_LOW_ACCURACY_HISTORY_KEY)
    yield ltm_client
    ltm_client.delete(LTM_LOW_ACCURACY_HISTORY_KEY)


@pytest.mark.parametrize('history,expected_list', [
    (_PositionalHistory([]), []),
    (_PositionalHistory([_PositionalHistoryItem(
        100, np.array([0.0, 1.0]))]), [[100, 0.0, 1.0]]),
    (_PositionalHistory([_PositionalHistoryItem(100, np.array([0.0, 1.0])), _PositionalHistoryItem(200, np.array([1.0, 2.0]))]),
     [[100, 0.0, 1.0], [200, 1.0, 2.0]]),
])
def test_low_accuracy_area_search_store(monkeypatch, setup_ltm, history, expected_list):
    def dummy_match_map_size(a, b, fill_a=0, fill_b=0):
        return

    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search.match_map_size',
                        dummy_match_map_size)
    ltm_client = setup_ltm

    low_accuracy_area_search = LowAccuracyAreaSearch()
    low_accuracy_area_search._history = history
    low_accuracy_area_search._store_history_to_ltm()

    json_str = ltm_client.get(LTM_LOW_ACCURACY_HISTORY_KEY)
    assert json.loads(json_str) == expected_list


@pytest.mark.parametrize('stored_list,expected_history', [
    ([], _PositionalHistory([])),
    ([[100, 0.0, 1.0]], _PositionalHistory(
        [_PositionalHistoryItem(100, np.array([0.0, 1.0]))])),
    ([[100, 0.0, 1.0], [200, 1.0, 2.0]],
     _PositionalHistory([_PositionalHistoryItem(100, np.array([0.0, 1.0])), _PositionalHistoryItem(200, np.array([1.0, 2.0]))])),
])
def test_low_accuracy_area_search_load(monkeypatch, setup_ltm, stored_list, expected_history):
    def dummy_match_map_size(a, b, fill_a=0, fill_b=0):
        return

    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search.match_map_size',
                        dummy_match_map_size)
    ltm_client = setup_ltm

    ltm_client.set(LTM_LOW_ACCURACY_HISTORY_KEY, json.dumps(stored_list))

    low_accuracy_area_search = LowAccuracyAreaSearch()
    low_accuracy_area_search._load_history_from_ltm()
    history = low_accuracy_area_search._history

    assert len(history.items) == len(expected_history.items)
    for i in range(len(history.items)):
        assert history.items[i] == expected_history.items[i]


@pytest.mark.parametrize('history,area_candidates,target,new_history', [
    ([], [np.array([0., 0.])], np.array([0., 0.]), [[100, 0., 0.]]),
    ([[10, 0., 0.]], [np.array([0., 0.])], np.array([0., 0.]), [[10, 0., 0.], [100, 0., 0.]]),
    ([[10, 0., 0.], [20, 0., 0.]],
     [np.array([0., 0.])], None, [[10, 0., 0.], [20, 0., 0.]]),
    ([[10, 0., 0.], [20, 0., 0.]], [np.array([1., 1.])], np.array([1., 1.]), [[100, 1., 1.]]),
    ([[10, 0., 0.], [20, 0., 0.], [30, 1., 1.]],
     [np.array([0., 0.]), np.array([1., 1.])], np.array([1., 1.]),
     [[10, 0., 0.], [20, 0., 0.], [30, 1., 1.], [100, 1., 1.]]),
])
def test_low_accuracy_area_search_get_low_accuracy_area(
        monkeypatch, setup_ltm, history, area_candidates, target, new_history):
    ltm_client = setup_ltm
    if history:
        ltm_client.set(LTM_LOW_ACCURACY_HISTORY_KEY, json.dumps(history))

    def dummy_match_map_size(a, b, fill_a=0, fill_b=0):
        return

    def get_low_accuracy_areas_centroid(accuracy_map, threshold) -> List[np.ndarray]:
        return area_candidates

    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search.match_map_size',
                        dummy_match_map_size)
    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search._accuracy_closing_filter',
                        lambda x: None)
    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search._mask_accuracy_map_with_obstacle',
                        lambda x, y: None)
    monkeypatch.setattr('lovot_slam.exploration.low_accuracy_area_search._get_low_accuracy_areas_centroid',
                        get_low_accuracy_areas_centroid)
    monkeypatch.setattr('time.time', lambda: 100)

    low_accuracy_area_search = LowAccuracyAreaSearch()
    obtained_target = low_accuracy_area_search.find(None, None, update_history=True)

    assert (np.all(np.isclose(obtained_target, target)) if target is not None
            else obtained_target == target)

    obtained_new_history = json.loads(ltm_client.get(LTM_LOW_ACCURACY_HISTORY_KEY))
    assert obtained_new_history == new_history
