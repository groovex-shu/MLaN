import re
import tempfile
import time
from pathlib import Path

import numpy as np
import prometheus_client
import pytest

from lovot_slam.env import DataDirectories, NestSlamState, data_directories
from lovot_slam.exploration.exploration_status import ExplorationStatus, ExplorationStatusMonitor
from lovot_slam.exploration.exploration_token import MAX_DAILY_TOKEN, ExplorationTokenManager
from lovot_slam.map_build.map_build_metrics import MapBuildAttemptResultsMetric
from lovot_slam.map_build.request_queue import BuildSingleMapOption, RequestQueue, RequestTypes
from lovot_slam.redis import create_ltm_client
from lovot_slam.utils.map_utils import MapUtils

_LTM_TOKEN_TTL_KEY = 'localiation_test:slam:exploration_token_ttl'
_LTM_CONTINUOUS_FAIL_COUNTS_KEY = 'localiation_test:slam:map_build:counter_metric'


def _unregister_prometheus_collectors():
    collectors = list(prometheus_client.REGISTRY._collector_to_names.keys())
    for collector in collectors:
        prometheus_client.REGISTRY.unregister(collector)


@pytest.fixture
def setup(monkeypatch):
    ltm_client = create_ltm_client()
    ltm_client.delete(_LTM_TOKEN_TTL_KEY)
    monkeypatch.setattr(ExplorationTokenManager, '_LTM_TOKEN_TTL_KEY', _LTM_TOKEN_TTL_KEY)
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)
    monkeypatch.setattr(MapBuildAttemptResultsMetric, '_LTM_CONTINUOUS_FAIL_COUNTS_KEY',
                        _LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    _unregister_prometheus_collectors()

    yield
    ltm_client.delete(_LTM_TOKEN_TTL_KEY)
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)

    _unregister_prometheus_collectors()


@pytest.fixture
def map_utils(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', Path(tmpdir))
        data_directories.maps.mkdir(parents=True, exist_ok=True)
        data_directories.bags.mkdir(parents=True, exist_ok=True)

        map_utils = MapUtils(data_directories.maps, data_directories.bags)

        yield map_utils


@pytest.fixture
def mock_sleep_time(sleep_time_end, sleep_time_start):
    ltm_client = create_ltm_client()
    if sleep_time_end:
        ltm_client.set('colony:sleep_time:end', sleep_time_end)
    if sleep_time_start:
        ltm_client.set('colony:sleep_time:start', sleep_time_start)
    yield
    ltm_client.delete('colony:sleep_time:end')
    ltm_client.delete('colony:sleep_time:start')


@pytest.mark.parametrize("is_building_map,elapsed,can_explore,should_issue,expected_token", [
    (True, ExplorationTokenManager._FALLBACK_TOKEN_TTL - 100, False, False, r""),
    (True, ExplorationTokenManager._FALLBACK_TOKEN_TTL - 100, True, False, r""),
    (True, ExplorationTokenManager._FALLBACK_TOKEN_TTL + 100, False, False, r""),
    (False, ExplorationTokenManager._FALLBACK_TOKEN_TTL - 100, False, False, r""),
    (False, ExplorationTokenManager._FALLBACK_TOKEN_TTL - 100, True, False, r""),
    (False, ExplorationTokenManager._FALLBACK_TOKEN_TTL + 100, False, False, r""),
    (True, ExplorationTokenManager._FALLBACK_TOKEN_TTL + 100, True, False, r""),
    (False, ExplorationTokenManager._FALLBACK_TOKEN_TTL + 100, True, True, r"[0-9]+\.[0-9]+"),
])
def test_inquire_token(monkeypatch, setup,
                       is_building_map, elapsed, can_explore, should_issue, expected_token, map_utils):
    ltm_client = create_ltm_client()

    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: can_explore)
    build_metric = MapBuildAttemptResultsMetric()
    exploration_status_monitor = ExplorationStatusMonitor()

    token_manager = ExplorationTokenManager(
        ltm_client, exploration_status_monitor, build_metric, lambda: is_building_map, False,
        RequestQueue(ltm_client), map_utils)
    token_manager._token_timestamp = time.time() - elapsed

    success, token = token_manager.inquire_token()
    assert success == should_issue
    assert re.match(expected_token, token)


@pytest.mark.parametrize("sleep_time_start,sleep_time_end,expected_token_ttl", [
    ('20:00', '08:00', 12 * 3600 / (MAX_DAILY_TOKEN - 1)),
    ('10:00', '20:00', 14 * 3600 / (MAX_DAILY_TOKEN - 1)),
    (None, None, ExplorationTokenManager._FALLBACK_TOKEN_TTL),
])
def test_calculate_ttl_from_sleep_time(monkeypatch, setup, map_utils,
                                       mock_sleep_time, sleep_time_start, sleep_time_end, expected_token_ttl):
    ltm_client = create_ltm_client()

    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: True)
    build_metric = MapBuildAttemptResultsMetric()
    exploration_status_monitor = ExplorationStatusMonitor()

    token_manager = ExplorationTokenManager(
        ltm_client, exploration_status_monitor, build_metric, lambda: False, True,
        RequestQueue(ltm_client), map_utils)

    assert token_manager._default_token_ttl == expected_token_ttl


@pytest.mark.parametrize("sleep_time_start,sleep_time_end,queue_num, expected_success, expected_token", [
    ('20:00', '08:00', 0, True, r"[0-9]+\.[0-9]+"),
    ('20:00', '08:00', 1, False, r""),
    ('20:00', '08:00', 2, False, r""),
])
def test_inquire_token_with_diff_queue_size(monkeypatch, setup, queue_num, expected_success, expected_token, map_utils,
                                            mock_sleep_time):
    ltm_client = create_ltm_client()

    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: True)
    build_metric = MapBuildAttemptResultsMetric()
    exploration_status_monitor = ExplorationStatusMonitor()

    queue = RequestQueue(ltm_client)
    while queue_num:
        queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name='test_map'))
        queue_num -= 1

    token_manager = ExplorationTokenManager(
        ltm_client, exploration_status_monitor, build_metric, lambda: False, True, queue, map_utils)

    while queue_num:
        token_manager._request_queue.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name='test_map'))
        queue_num -= 1

    success, token = token_manager.inquire_token()
    assert success == expected_success
    assert re.match(expected_token, token)


def test_inquire_token_with_full_disk(monkeypatch, setup, map_utils):
    ltm_client = create_ltm_client()

    dummy_file = map_utils._root / 'test_map'
    limit_size = 1024 * 1024

    monkeypatch.setattr('lovot_slam.exploration.exploration_token.MAX_DATA_DIR_SIZE', limit_size)

    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: True)
    build_metric = MapBuildAttemptResultsMetric()
    exploration_status_monitor = ExplorationStatusMonitor()

    token_manager = ExplorationTokenManager(
        ltm_client, exploration_status_monitor, build_metric, lambda: False, False,
        RequestQueue(ltm_client), map_utils)

    # 1. create dummy file that exceeds the limit size
    with open(dummy_file, 'w') as f:
        # 100 as margin
        f.write('a' * (limit_size + 100))
    # token is not published
    success, _ = token_manager.inquire_token()
    assert not success

    # 2. remove the dummy file
    dummy_file.unlink()
    # token is published
    success, _ = token_manager.inquire_token()
    assert success


@pytest.mark.parametrize("clear_by", [
    ("success"),
    ("reset"),
])
def test_token_extension(monkeypatch, setup, clear_by, map_utils):
    ltm_client = create_ltm_client()

    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: True)
    build_metric = MapBuildAttemptResultsMetric()
    exploration_status_monitor = ExplorationStatusMonitor()

    token_manager = ExplorationTokenManager(
        ltm_client, exploration_status_monitor, build_metric, lambda: False, False,
        RequestQueue(ltm_client), map_utils)

    current_time = 0

    def mock_time():
        return current_time

    monkeypatch.setattr(time, 'time', mock_time)

    def check_token_ttl(expected_ttl):
        nonlocal current_time
        # TTL経過前: token発行されない
        current_time += expected_ttl - 10
        success, token = token_manager.inquire_token()
        assert not success
        # TTL経過直後: token発行される
        current_time += 20
        success, token = token_manager.inquire_token()
        assert success

    # 1. マップ生成に失敗していない時、TTLはデフォルト値のまま
    expected_ttl = ExplorationTokenManager._FALLBACK_TOKEN_TTL
    for _ in range(100):
        check_token_ttl(expected_ttl)
        build_metric.success()

    # 2. マップ生成に失敗し続けてる時
    # _MIN_BUILD_FAIL_COUNT_TO_EXTEND_TTL 回までは、デフォルトTTLのまま
    for _ in range(ExplorationTokenManager._MIN_BUILD_FAIL_COUNT_TO_EXTEND_TTL):
        check_token_ttl(expected_ttl)
        build_metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)

    # 3. それ以降は、_MAX_TOKEN_TTLになるまで、_TTL_EXTENSION_RATE倍されていく
    count_to_max = int(np.log(ExplorationTokenManager._MAX_TOKEN_TTL / ExplorationTokenManager._FALLBACK_TOKEN_TTL)
                       / np.log(ExplorationTokenManager._TTL_EXTENSION_RATE)) + 1
    for i in range(count_to_max):
        check_token_ttl(expected_ttl)
        build_metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)
        expected_ttl *= ExplorationTokenManager._TTL_EXTENSION_RATE

    # 4. _MAX_TOKEN_TTLを維持し続ける
    expected_ttl = ExplorationTokenManager._MAX_TOKEN_TTL
    for _ in range(100):
        check_token_ttl(expected_ttl)
        build_metric.fail(True, NestSlamState.BUILD_FEATURE_MAP)

    # 5. マップ生成に成功するか、リセットされるとTTLもリセットされる
    if clear_by == "reset":
        build_metric.reset()
    elif clear_by == "success":
        build_metric.success()
    expected_ttl = ExplorationTokenManager._FALLBACK_TOKEN_TTL
    for _ in range(10):
        check_token_ttl(expected_ttl)
