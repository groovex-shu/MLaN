import json

import numpy as np
import pytest
import trio

from grid_map_util.accuracy_map import AccuracyMap, logarithm_probability
from grid_map_util.occupancy_grid import OccupancyGrid

from lovot_slam.exploration import ExplorationStatus, ExplorationStatusMonitor
from lovot_slam.exploration.exploration_status import (EXPLORATION_STATUS_KEY, MAP_METRICS_HISTORY_KEY,
                                                       _map_area_metric, _MapMetricsHistory)
from lovot_slam.exploration.frontier_search import LTM_FRONTIER_HISTORY_KEY
from lovot_slam.exploration.low_accuracy_area_search import LTM_LOW_ACCURACY_HISTORY_KEY
from lovot_slam.redis import create_ltm_client
from lovot_slam.utils.map_utils import MapUtils


@pytest.mark.parametrize(
    "missions_count,frontier_remained,low_accuracy_area_remained,is_ready,is_well_explored,is_gave_up,can_explore", [
        (1, True, False, False, False, False, True),
        (1, False, False, False, False, False, True),
        (7, True, False, False, False, False, True),
        (8, True, False, True, False, False, True),
        (7, False, False, True, False, False, True),
        (20, True, False, True, False, False, True),
        (21, True, False, True, False, True, False),
        (20, False, False, True, True, False, False),
        (8, False, False, True, True, False, False),
        (21, False, False, True, True, False, False),
        # TODO: add patterns with low_accuracy_area_remained=True
    ])
def test_exploration_status(missions_count, frontier_remained, low_accuracy_area_remained,
                            is_ready, is_well_explored, is_gave_up, can_explore):
    timestamp = 100.0
    exploration_status = ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained,
                                           True, timestamp)

    assert is_ready == exploration_status.is_ready()
    assert is_well_explored == exploration_status.is_well_explored()
    assert is_gave_up == exploration_status.is_gave_up()
    assert can_explore == exploration_status.can_explore()


@pytest.mark.parametrize("status_a,status_b,is_equal", [
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, True, 100), True),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, True, 200), True),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, False, 200), True),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(1, True, False, True, 100), False),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, False, False, True, 100), False),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(1, False, False, True, 100), False),
    (ExplorationStatus(0, True, True, True, 100), ExplorationStatus(0, True, False, True, 100), False),
])
def test_exploration_status_is_equal(status_a: ExplorationStatus, status_b: ExplorationStatus, is_equal: bool):
    assert status_a.is_equal(status_b) == is_equal
    assert status_b.is_equal(status_a) == is_equal


@pytest.mark.parametrize("status", [
    (ExplorationStatus(0, True, False, True, 1622619068.833377)),
])
def test_exploration_status_str(status: ExplorationStatus):
    # just to check no exceptions occur
    print(status)


@pytest.fixture
def setup_redis():
    ltm_client = create_ltm_client()
    ltm_client.delete(EXPLORATION_STATUS_KEY)
    ltm_client.delete(MAP_METRICS_HISTORY_KEY)
    yield
    ltm_client.delete(EXPLORATION_STATUS_KEY)
    ltm_client.delete(MAP_METRICS_HISTORY_KEY)


def test_map_metrics_history_update(monkeypatch, setup_redis):
    ltm_client = create_ltm_client()

    map_name = None
    map_size = None
    map_accuracy = None

    def get_occupancy_grid(_self, map_name: str) -> OccupancyGrid:
        nonlocal map_size
        return OccupancyGrid(np.zeros(map_size, dtype=np.uint8), 1, np.zeros(2))

    monkeypatch.setattr(MapUtils, 'get_latest_merged_map', lambda _self: map_name)
    monkeypatch.setattr(MapUtils, 'get_occupancy_grid', get_occupancy_grid)

    history = _MapMetricsHistory()

    map_name = 'map_0'
    map_size = (11, 10)
    map_accuracy = 0.5
    accuracy_map = AccuracyMap(np.full((100, 100), logarithm_probability(map_accuracy)),
                               np.zeros(2),
                               0.05)
    history.update(accuracy_map)
    assert [metrics.to_dict() for metrics in history._history] ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.5}]
    assert json.loads(ltm_client.get(MAP_METRICS_HISTORY_KEY)) ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.5}]

    assert _map_area_metric.labels(gen_from_latest="0", code="occupied")._value._value == 110

    map_accuracy = 0.6
    accuracy_map = AccuracyMap(np.full((100, 100), logarithm_probability(map_accuracy)),
                               np.zeros(2),
                               0.05)
    history.update(accuracy_map)
    # accuracy以外の情報は変わらない
    assert [metrics.to_dict() for metrics in history._history] ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6}]
    assert json.loads(ltm_client.get(MAP_METRICS_HISTORY_KEY)) ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6}]

    map_name = 'map_1'
    map_size = (12, 10)
    map_accuracy = 0.7
    accuracy_map = AccuracyMap(np.full((100, 100), logarithm_probability(map_accuracy)),
                               np.zeros(2),
                               0.05)
    history.update(accuracy_map)
    # 前のmapの情報は保持されたまま
    assert [metrics.to_dict() for metrics in history._history] ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6},
         {"map_name": "map_1", "map_area": [0, 120, 0], "last_map_accuracy": 0.7}]
    assert json.loads(ltm_client.get(MAP_METRICS_HISTORY_KEY)) ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6},
         {"map_name": "map_1", "map_area": [0, 120, 0], "last_map_accuracy": 0.7}]

    assert _map_area_metric.labels(gen_from_latest="0", code="occupied")._value._value == 120
    assert _map_area_metric.labels(gen_from_latest="1", code="occupied")._value._value == 110


def test_map_metrics_history_load(monkeypatch, setup_redis):
    ltm_client = create_ltm_client()
    ltm_client.set(MAP_METRICS_HISTORY_KEY,
        '[{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6},'
        ' {"map_name": "map_1", "map_area": [0, 120, 0], "last_map_accuracy": 0.7}]')

    history = _MapMetricsHistory()
    assert [metrics.to_dict() for metrics in history._history] ==\
        [{"map_name": "map_0", "map_area": [0, 110, 0], "last_map_accuracy": 0.6},
         {"map_name": "map_1", "map_area": [0, 120, 0], "last_map_accuracy": 0.7}]


async def wait_for_monitoring_period():
    # 10 seconds for margin,
    # because `trioe_util.period` may not be treggered if the sleep period is exactly the same
    await trio.sleep(ExplorationStatusMonitor._MONITOR_PERIOD_SEC + 10)


@pytest.mark.parametrize("initial_state,initial_can_explore,duration,after_state,after_can_explore", [
    ((0, True, False), True, 600, (0, True, False), True),
    ((8, False, False), False, 600, (8, False, False), False),
    ((8, False, False), False, 600, (8, True, False), False),
    ((8, False, False), False, 7*24*60*60, (8, True, False), True),
    ((8, False, False), False, 7*24*60*60, (8, False, False), False),
])
async def test_exploration_status_monitor_transition(initial_state, initial_can_explore,
                                                     duration, after_state, after_can_explore,
                                                     monkeypatch, nursery, setup_redis, autojump_clock):
    missions_count, frontier_remained, low_accuracy_area_remained = initial_state

    def check_exploration_status(arg):
        nonlocal missions_count, frontier_remained, low_accuracy_area_remained
        return ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained, True, trio.current_time())

    monkeypatch.setattr(ExplorationStatusMonitor, '_check_exploration_status', check_exploration_status)
    monkeypatch.setattr('time.time', trio.current_time)

    monitor = ExplorationStatusMonitor()
    nursery.start_soon(monitor.run)
    await wait_for_monitoring_period()
    assert initial_can_explore == monitor.status.can_explore()

    await trio.sleep(duration)

    missions_count, frontier_remained, low_accuracy_area_remained = after_state
    await wait_for_monitoring_period()
    assert after_can_explore == monitor.status.can_explore()


async def test_exploration_status_monitor_scenario(monkeypatch, nursery, setup_redis, autojump_clock):
    missions_count, frontier_remained, low_accuracy_area_remained = (0, True, False)

    def check_exploration_status(arg):
        nonlocal missions_count, frontier_remained, low_accuracy_area_remained
        return ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained, True,
                                 trio.current_time())

    monkeypatch.setattr(ExplorationStatusMonitor, '_check_exploration_status', check_exploration_status)
    monkeypatch.setattr('time.time', trio.current_time)

    # status starts with not ready, not well_explored and not gave_up
    missions_count, frontier_remained, low_accuracy_area_remained = (0, False, False)
    monitor = ExplorationStatusMonitor()
    nursery.start_soon(monitor.run)
    await trio.sleep(10)
    assert monitor.status.can_explore()

    # status becomes ready (but still can explore)
    missions_count, frontier_remained, low_accuracy_area_remained = (0, False, False)
    monitor.update()
    await trio.sleep(10)
    assert monitor.status.can_explore()

    # status becomes well_explored (cannot explore)
    missions_count, frontier_remained, low_accuracy_area_remained = (8, False, False)
    monitor.update()
    await trio.sleep(10)
    assert not monitor.status.can_explore()

    # exploration suspends for 1 week (cannot explore)
    await wait_for_monitoring_period()
    assert not monitor.status.can_explore()

    # after 1 week, no status changed (cannot explore)
    await trio.sleep(7 * 24 * 60 * 60)
    await wait_for_monitoring_period()
    assert not monitor.status.can_explore()

    # status becomes not well_explored (can explore)
    missions_count, frontier_remained, low_accuracy_area_remained = (8, True, False)
    await wait_for_monitoring_period()
    assert monitor.status.can_explore()

    # after map build, status becomes well_explored (cannot explore)
    missions_count, frontier_remained, low_accuracy_area_remained = (9, False, False)
    monitor.update()
    await trio.sleep(10)
    assert not monitor.status.can_explore()

    # status becomes well_explored but low_accuracy_area_remained (can explore)
    missions_count, frontier_remained, low_accuracy_area_remained = (9, False, True)
    await trio.sleep(7 * 24 * 60 * 60)
    await wait_for_monitoring_period()
    assert monitor.status.can_explore()


@pytest.mark.parametrize("status_a,status_b,force_update,expected_status", [
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, True, 200), True,
     ExplorationStatus(0, True, False, True, 200)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, True, 200), False,
     ExplorationStatus(0, True, False, True, 100)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, False, False, True, 200), True,
     ExplorationStatus(0, False, False, True, 200)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, False, False, True, 200), False,
     ExplorationStatus(0, False, False, True, 200)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(1, True, False, True, 200), True,
     ExplorationStatus(1, True, False, True, 200)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(1, True, False, True, 200), False,
     ExplorationStatus(1, True, False, True, 200)),
])
def test_exploration_status_monitor_update(monkeypatch, setup_redis, status_a, status_b, force_update, expected_status):
    current_status = status_a

    def check_exploration_status(arg):
        nonlocal current_status
        return current_status

    monkeypatch.setattr(ExplorationStatusMonitor, '_check_exploration_status', check_exploration_status)

    monitor = ExplorationStatusMonitor()
    assert monitor._exploration_status == status_a

    current_status = status_b
    monitor.update(force_update=force_update)
    assert monitor._exploration_status == expected_status


@pytest.mark.parametrize("status_a,status_b", [
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(0, True, False, True, 100)),
    (ExplorationStatus(0, True, False, True, 100), ExplorationStatus(5, False, False, True, 200)),
    (ExplorationStatus(0, False, False, True, 100), ExplorationStatus(5, False, False, True, 200)),
    (ExplorationStatus(0, False, False, True, 100), ExplorationStatus(5, True, False, True, 200)),
    (ExplorationStatus(0, False, False, False, 100), ExplorationStatus(5, True, False, False, 200)),
])
def test_exploration_status_monitor_redis(monkeypatch, setup_redis, status_a, status_b):
    current_status = status_a

    def check_exploration_status(arg):
        nonlocal current_status
        return current_status

    monkeypatch.setattr(ExplorationStatusMonitor, '_check_exploration_status', check_exploration_status)

    monitor = ExplorationStatusMonitor()  # store status_a to redis
    monitor._load_exploration_status_from_redis()
    assert monitor._exploration_status == status_a

    current_status = status_b
    monitor.update()  # store status_b to redis
    monitor._load_exploration_status_from_redis()
    assert monitor._exploration_status == status_b


@pytest.mark.parametrize("interval_from_map_build,low_accuracy_area,remained", [
    (20*24*60*60, None, False),
    (20*24*60*60, np.array([0., 0.]), False),
    (40*24*60*60, None, False),
    (40*24*60*60, np.array([0., 0.]), True),
])
def test_exploration_status_check_low_accuracy_area_remained(
        monkeypatch, interval_from_map_build, low_accuracy_area, remained):
    now = 1621906157.6038687
    monkeypatch.setattr('time.time', lambda: now)
    monkeypatch.setattr('lovot_slam.utils.map_utils.MapUtils.get_map_stamp',
                        lambda self, map_name: now - interval_from_map_build)
    monkeypatch.setattr(ExplorationStatusMonitor, 'find_low_accuracy_area',
                        lambda self, map_name, update_history=False: low_accuracy_area)

    monitor = ExplorationStatusMonitor()
    obtained = monitor._check_low_accuracy_area_remained()

    assert obtained == remained


async def test_exploration_status_monitor_reset(monkeypatch, nursery, setup_redis, autojump_clock):
    missions_count, frontier_remained, low_accuracy_area_remained, initially_triggered = (0, True, False, True)

    def check_exploration_status(arg):
        return ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained, initially_triggered,
                                 trio.current_time())

    monkeypatch.setattr(ExplorationStatusMonitor, '_check_exploration_status', check_exploration_status)
    monkeypatch.setattr('time.time', trio.current_time)

    # setup
    ltm_client = create_ltm_client()
    ltm_client.set(LTM_FRONTIER_HISTORY_KEY, '')
    ltm_client.set(LTM_LOW_ACCURACY_HISTORY_KEY, '')

    monitor = ExplorationStatusMonitor()
    nursery.start_soon(monitor.run)
    monitor.reset()

    # initially_triggered and timestamp are ignored in comparison
    assert monitor.status == ExplorationStatus(missions_count, frontier_remained, low_accuracy_area_remained,
                                               initially_triggered, 0)

    assert ltm_client.get(LTM_FRONTIER_HISTORY_KEY) is None
    assert ltm_client.get(LTM_LOW_ACCURACY_HISTORY_KEY) is None

# TODO: write tests for ExplorationStatusMonitor._check_exploration_status


async def test_exploration_monitor_initally_triggered(monkeypatch, nursery, setup_redis, autojump_clock):
    latest_map_name = ''
    missions_count = 0
    frontier = np.array([0, 0])
    low_accuracy_area_remained = False

    monkeypatch.setattr(MapUtils, "get_latest_merged_map", lambda self_: latest_map_name)
    monkeypatch.setattr(MapUtils, "get_maps_number_in_latest_merged_map", lambda self_: missions_count)
    monkeypatch.setattr(ExplorationStatusMonitor, "find_new_frontier", lambda self_, map_name: frontier)
    monkeypatch.setattr(ExplorationStatusMonitor, "_check_low_accuracy_area_remained",
                        lambda self_: low_accuracy_area_remained)
    monkeypatch.setattr('time.time', trio.current_time)

    monitor = ExplorationStatusMonitor()
    nursery.start_soon(monitor.run)

    def assert_status(can_explore: bool, initially_triggered: bool):
        assert monitor.status.missions_count == missions_count
        assert monitor.status.frontier_remained is (True if frontier is not None else False)
        assert monitor.status.low_accuracy_area_remained is low_accuracy_area_remained
        assert monitor.status.initially_triggered is initially_triggered
        assert monitor.status.can_explore() is can_explore

    # first map build done -> still can explore
    latest_map_name, missions_count = ('20220101_000000', 1)
    monitor.update()
    assert_status(can_explore=True, initially_triggered=True)

    # another map build done and frontier disappear -> cannot explore
    # NOTE: initially_triggered is still True
    latest_map_name, missions_count, frontier = ('20220102_000000', 9, None)
    monitor.update()
    assert_status(can_explore=False, initially_triggered=True)

    # frontier found after a week
    # -> can explore again, and not initially_triggered
    await trio.sleep(7 * 24 * 60 * 60)
    frontier = np.array([0, 0])
    await wait_for_monitoring_period()
    assert_status(can_explore=True, initially_triggered=False)

    # another map build done and frontier disappear
    # -> cannot explore, and not initially_triggered
    latest_map_name, missions_count, frontier = ('20220102_000000', 9, None)
    monitor.update()
    assert_status(can_explore=False, initially_triggered=False)

    # frontier found after a week
    # -> can explore again, and not initially_triggered
    await trio.sleep(7 * 24 * 60 * 60)
    frontier = np.array([0, 0])
    await wait_for_monitoring_period()
    assert_status(can_explore=True, initially_triggered=False)

    latest_map_name, missions_count = ('', 0)
    monitor.reset()
    assert_status(can_explore=True, initially_triggered=True)
