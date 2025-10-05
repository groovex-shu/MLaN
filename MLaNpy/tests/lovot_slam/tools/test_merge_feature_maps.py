from typing import List

import pytest

import lovot_slam.tools.merge_feature_maps
from lovot_slam.tools.merge_feature_maps import _find_missions_to_remove


class MockMissionsFilter:
    def __init__(self, sorted_mission_ids: List[str]) -> None:
        self.sorted_mission_ids = sorted_mission_ids

    @classmethod
    def create_from_map_path(cls, map_path: str) -> 'MockMissionsFilter':
        return MockMissionsFilter([])

    def filter_by_overlapping(self) -> List[str]:
        return self.filterred_by_overlapping

    def filter_by_count(self, missions_to_remove: List[str],
                        max_merge_missions_count: int) -> List[str]:
        missions_to_remain = list(set(self.sorted_mission_ids) - set(missions_to_remove))
        if len(missions_to_remain) > max_merge_missions_count:
            return missions_to_remain[max_merge_missions_count:] + missions_to_remove
        return missions_to_remove


@pytest.mark.parametrize("original_missions,filterred_by_overlapping,max_count,count_to_be_removed", [
    (list(map(str, range(1))), [], 30, 0),
    (list(map(str, range(2))), [], 30, 0),
    (list(map(str, range(3))), list(map(str, range(1))), 30, 0),  # 8個以下の場合はremoveしない
    (list(map(str, range(8))), list(map(str, range(7))), 30, 0),  # 8個以下の場合はremoveしない
    (list(map(str, range(10))), list(map(str, range(3))), 9, 2),  # 8個以下の場合はremoveしない
    (list(map(str, range(6))), list(map(str, range(3))), 30, 0),  # 8個以下の場合はremoveしない
    (list(map(str, range(7))), list(map(str, range(6))), 30, 0),  # 8個以下の場合はremoveしない
    (list(map(str, range(36))), [], 30, 6),
])
def test_find_missions_to_remove(monkeypatch,
                                 original_missions, filterred_by_overlapping, max_count, count_to_be_removed):
    """_find_missions_to_remove に直で書かれたロジックのみをテストしている
    MissionsFilterは期待通りに動作しているものとし、Mockに置き換えている (MissionsFilterのテストは別で実施されているため)
    """
    lovot_slam.tools.merge_feature_maps.MissionsFilter = MockMissionsFilter

    monkeypatch.setattr(MockMissionsFilter, 'create_from_map_path', lambda x: MockMissionsFilter(original_missions))
    monkeypatch.setattr(MockMissionsFilter, 'filter_by_overlapping', lambda x: filterred_by_overlapping)

    missions_to_remove = _find_missions_to_remove('', max_count)

    assert len(missions_to_remove) == count_to_be_removed
