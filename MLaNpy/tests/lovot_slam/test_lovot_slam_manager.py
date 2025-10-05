import tempfile
import time
from pathlib import Path
from typing import List

import pytest
import trio
from redis import StrictRedis
import prometheus_client
import numpy as np
import json

from lovot_slam import Context, context
from lovot_slam.env import MAP_MD5SUM_YAML, PUSHBAG_RETRY_INTERVAL, DataDirectories, data_directories
from lovot_slam.lovot_slam_manager import LovotSlamManager
from lovot_slam.redis import create_device_client, create_ltm_client, create_stm_client
from lovot_slam.redis.keys import ROBOT_MODEL_KEY
from lovot_slam.utils.exceptions import SlamProcedureCallError, SlamSensorError
from lovot_slam.utils.omni_camera_mode import CONVERTER_MODE_KEY
from lovot_slam.lovot_slam_manager import MARKERS_POSITION_KEY

from .mock.mock_map_builder import MockMergedMap, MockSingleMissionMap, prepare_dummy_single_merged_maps_pair
from .mock.mock_subprocess import BaseSubprocess, LocalizationSubprocess, RecordSubprocess, mock_rosmaster


class MockSlamServicerClient:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.communication_error = False
        self.retried_bag = None
        self.build_triggerd_map = None
        self.spot_from_spike = None

    async def upload_rosbag(self, map_name, file_path):
        self.retried_bag = map_name
        assert file_path.exists()

    async def build_single_mission_map(self, map_name):
        self.build_triggerd_map = map_name

    async def get_spots(self, spot_names):
        if self.communication_error:
            print('raise error')
            raise SlamProcedureCallError
        return self.spot_from_spike


def setup_module():
    stm_client = create_stm_client()
    device_client = create_device_client()
    stm_client.set(CONVERTER_MODE_KEY, '2')
    device_client.set(ROBOT_MODEL_KEY, 'lv101')


def teardown_module():
    stm_client = create_stm_client()
    device_client = create_device_client()
    stm_client.delete(CONVERTER_MODE_KEY)
    device_client.delete(ROBOT_MODEL_KEY)


@pytest.fixture
def setup_mock_context() -> None:
    context.set(Context(
        slam_servicer_client=MockSlamServicerClient(),
        wifi_client=None,
        lovot_tf_client=None,
        localization_client=None,
        fingerprint_sync=None,
        wifi_scan=None,
        radio_map=None,
    ))


@pytest.fixture
def dummy_data_directory(monkeypatch) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', tmpdir)
        monkeypatch.setattr('lovot_slam.wifi.mapping.mapping.RADIO_MAP_FILE',
                            trio.Path(data_directories.maps / 'wifi_fingerprints'))
        yield tmpdir


@pytest.fixture
def prepare_slam_spots() -> List[StrictRedis]:
    ltm_client = create_ltm_client()
    for key in ltm_client.keys('slam:spot:*'):
        ltm_client.delete(key)

    yield ltm_client

    for key in ltm_client.keys('slam:spot:*'):
        ltm_client.delete(key)


@pytest.fixture
def mock_roslaunch_subprocesses(monkeypatch):
    monkeypatch.setattr('lovot_slam.lovot_slam_manager.BaseSubprocess', BaseSubprocess)
    monkeypatch.setattr('lovot_slam.lovot_slam_manager.LocalizationSubprocess', LocalizationSubprocess)
    monkeypatch.setattr('lovot_slam.lovot_slam_manager.RecordSubprocess', RecordSubprocess)
    monkeypatch.setattr('lovot_slam.lovot_slam_manager.rosgraph', mock_rosmaster)


@pytest.fixture
def prepare_prometheus_client():
    def unregister_prometheus_collectors():
        collectors = list(prometheus_client.REGISTRY._collector_to_names.keys())
        for collector in collectors:
            prometheus_client.REGISTRY.unregister(collector)

    unregister_prometheus_collectors()
    yield
    unregister_prometheus_collectors()


@pytest.mark.parametrize('crashed_processes,expected_exception,should_undeploy_map', [
    ([None] * 10, RuntimeError, False),
    (['twist_publisher'] * 10, RuntimeError, False),
    (['omni_streamer'] * 10, SlamSensorError, False),
    (['shm_to_depth'] * 10, SlamSensorError, False),
    (['localizer'] * 10, RuntimeError, True),
    (['map_server'] * 10, RuntimeError, True),
    (['accuracy_monitor'] * 10, RuntimeError, True),
    (['omni_streamer'] * 6 + ['localizer'] * 4, SlamSensorError, False),
    (['omni_streamer'] * 4 + ['localizer'] * 6, RuntimeError, True),
    # depends on the order of occurrence, if the number of occurrences is the same
    (['localizer'] * 5 + ['omni_streamer'] * 5, SlamSensorError, False),
    (['omni_streamer'] * 5 + ['localizer'] * 5, RuntimeError, True),
])
async def test_handle_ros_crash_and_raise(monkeypatch, dummy_data_directory, prepare_prometheus_client,
                                          crashed_processes, expected_exception, should_undeploy_map):
    map_undeployed = False

    async def undeploy_map(self_):
        nonlocal map_undeployed
        map_undeployed = True

    monkeypatch.setattr(LovotSlamManager, '_undeploy_map', undeploy_map)

    slam_manager = LovotSlamManager()
    with pytest.raises(expected_exception) as e_info:
        await slam_manager._handle_ros_crash_and_raise(crashed_processes)
    print(e_info)

    assert map_undeployed == should_undeploy_map


async def test_reload_slam_spots_from_nest(monkeypatch, dummy_data_directory, prepare_slam_spots, setup_mock_context, prepare_prometheus_client,
                                           nursery, autojump_clock):
    stm_client = create_stm_client()
    ltm_client = create_ltm_client()
    assert {} == ltm_client.hgetall('slam:spot:entrance')

    mock_client = context.get().slam_servicer_client
    mock_client.communication_error = False
    mock_client.spot_from_spike = None

    async def publish_command_and_wait():
        request_spots = ['entrance']
        command = f'reload_slam_spots_from_nest 0 {",".join(request_spots)}'
        stm_client.publish('slam:command', command)
        time.sleep(0.1)  # wait until the message is processed in redis_listner
        await trio.sleep(1)

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager._monitor_command)

    try:
        slam_manager._redis_listener.start()

        # LTM key should be reloaded with a value from spike
        mock_client.spot_from_spike = \
            {'entrance': {'id': 'entrance', 'name': 'entrance', 'coordinate': '0,0,0,1,0,0,0'}}
        await publish_command_and_wait()
        assert mock_client.spot_from_spike['entrance'] == ltm_client.hgetall('slam:spot:entrance')

        # LTM key should be reloaded with the updated value
        mock_client.spot_from_spike['entrance']['coordinate'] = '1,0,0,1,0,0,0'
        await publish_command_and_wait()
        assert mock_client.spot_from_spike['entrance'] == ltm_client.hgetall('slam:spot:entrance')

        # LTM key should be preserved (not deleted with communication errors)
        mock_client.communication_error = True
        await publish_command_and_wait()
        assert mock_client.spot_from_spike['entrance'] == ltm_client.hgetall('slam:spot:entrance')

        # LTM key should be deleted
        mock_client.communication_error = False
        mock_client.spot_from_spike = {}
        await publish_command_and_wait()
        assert {} == ltm_client.hgetall('slam:spot:entrance')
    finally:
        slam_manager._redis_listener.stop()


async def test_load_map_on_start(
        dummy_data_directory, mock_roslaunch_subprocesses, setup_mock_context, prepare_prometheus_client,
        nursery, autojump_clock):
    _, latest_merge_map = prepare_dummy_single_merged_maps_pair()

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(30)

    assert latest_merge_map == slam_manager.map_name
    assert slam_manager.localization_process.is_running()


@pytest.mark.parametrize('del_tail_chars', [
    (15),  # invalid md5sum length of the last key
    (33),  # empty md5sum of the last key
    (34),  # invalid yaml format
])
async def test_handle_broken_md5sum_map(
        del_tail_chars,
        dummy_data_directory, mock_roslaunch_subprocesses, setup_mock_context, prepare_prometheus_client,
        nursery, autojump_clock):
    def break_md5sum_yaml(map_name):
        md5sum_yaml = data_directories.maps / map_name / MAP_MD5SUM_YAML
        contents = md5sum_yaml.read_text().rstrip()
        md5sum_yaml.write_text(contents[:-del_tail_chars])
        print(md5sum_yaml.read_text().split('\n')[-1])

    _, latest_merge_map = prepare_dummy_single_merged_maps_pair()
    break_md5sum_yaml(latest_merge_map)

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(30)

    assert slam_manager.map_name is None
    assert not slam_manager.localization_process.is_running()


async def test_retry_push_bag(nursery, dummy_data_directory, prepare_prometheus_client, setup_mock_context, autojump_clock):
    """_failed_push_bag_fileがセットされた場合
    60秒後にリトライする
    60秒以内にキャンセルされた場合はリトライしない
    """
    mock_spike_client = context.get().slam_servicer_client

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager._monitor_retry_push_bag)

    bag_name = 'test'
    bag_file = data_directories.bags / f'{bag_name}.bag'
    bag_file.touch()

    # retry push-bag after the interval
    slam_manager._failed_push_bag_file.value = bag_name
    await trio.sleep(10)
    assert mock_spike_client.retried_bag is None \
        and mock_spike_client.build_triggerd_map is None
    await trio.sleep(PUSHBAG_RETRY_INTERVAL)
    # bag should be uploaded, map build should be triggered and the bag should be removed
    assert mock_spike_client.retried_bag == bag_name
    assert mock_spike_client.build_triggerd_map == bag_name
    assert not bag_file.exists()
    mock_spike_client.reset()

    # cancel retry push-bag within the interval
    slam_manager._failed_push_bag_file.value = bag_name
    await trio.sleep(10)
    assert mock_spike_client.retried_bag is None \
        and mock_spike_client.build_triggerd_map is None
    slam_manager.cancel_retry_push_bag()
    await trio.sleep(PUSHBAG_RETRY_INTERVAL)
    assert mock_spike_client.retried_bag is None \
        and mock_spike_client.build_triggerd_map is None


async def test_retry_push_bag_when_bag_remained_on_start(nursery,
                                                         dummy_data_directory, setup_mock_context, prepare_prometheus_client, autojump_clock):
    """起動時点でbagファイルが残っているケース
    起動後60秒後にリトライする
    """
    mock_spike_client = context.get().slam_servicer_client

    # bag exists prior the manager starts
    bag_name = 'test'
    bag_file = data_directories.bags / f'{bag_name}.bag'
    data_directories.bags.mkdir(parents=True)
    bag_file.touch()

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager._monitor_retry_push_bag)

    # retry push-bag after the interval
    await trio.sleep(10)
    assert mock_spike_client.retried_bag is None \
        and mock_spike_client.build_triggerd_map is None
    await trio.sleep(PUSHBAG_RETRY_INTERVAL)
    # bag should be uploaded, map build should be triggered and the bag should be removed
    assert mock_spike_client.retried_bag == bag_name
    assert mock_spike_client.build_triggerd_map == bag_name
    assert not bag_file.exists()


async def test_restarting_base_after_base_is_killed(
        nursery, mock_roslaunch_subprocesses, dummy_data_directory, setup_mock_context, prepare_prometheus_client, autojump_clock):
    stm_client = create_stm_client()
    stm_client.set('omni:converter_mode', '2')

    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager.run)

    # wait for base to be running
    with trio.fail_after(15):
        while not slam_manager.base_process.is_running():
            await trio.sleep(0.1)
    assert mock_rosmaster.is_master_online()
    print('base and rosmaster is running')

    # force stopping rosmaster
    print('force stopping rosmaster...')
    mock_rosmaster.stop(None, force=True)
    assert not mock_rosmaster.is_master_online()

    # wait for base and rosmaster to be running (rosmaster will be running with base)
    with trio.fail_after(25):
        while not mock_rosmaster.is_master_online():
            await trio.sleep(0.1)
    assert slam_manager.base_process.is_running()
    print('base and rosmaster is running')


async def test_expecting_inconsistency_error(
        nursery, mock_roslaunch_subprocesses, dummy_data_directory, setup_mock_context, prepare_prometheus_client, autojump_clock):
    stm_client = create_stm_client()
    stm_client.set('omni:converter_mode', '2')

    slam_manager = LovotSlamManager()

    with pytest.raises(RuntimeError):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(slam_manager.run)

            # wait for base to be running
            with trio.fail_after(15):
                while not slam_manager.base_process.is_running():
                    await trio.sleep(0.1)
            assert mock_rosmaster.is_master_online()
            print('base and rosmaster is running')

            # kill base
            print('killing base...')
            slam_manager.base_process._running = False

            # NOTE: RuntimeError is raised in the main loop
            await trio.sleep(20)
            nursery.cancel_scope.cancel()


async def test_undeploy_map(mock_roslaunch_subprocesses, dummy_data_directory, prepare_prometheus_client, nursery, autojump_clock):
    def prepare_dummy_map():
        data_directories.segmentation.mkdir(parents=True)
        maps_root = data_directories.maps
        single_map = MockSingleMissionMap(maps_root)
        merged_map = MockMergedMap(maps_root, source_maps=[single_map])
        merged_map.create()
        return merged_map.map_name

    def prepare_marker_info():
        ltm_client = create_ltm_client()
        marker_pose = np.identity(4)
        ltm_client.hset(MARKERS_POSITION_KEY, str(11),
                        json.dumps(marker_pose.tolist()))

    def is_marker_removed():
        ltm_client = create_ltm_client()
        result = ltm_client.hgetall(MARKERS_POSITION_KEY)
        return not result

    # preparation of test
    latest_merge_map = prepare_dummy_map()
    prepare_marker_info()

    # run main loop
    slam_manager = LovotSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(30)

    # check the initial state
    assert latest_merge_map == slam_manager.map_name
    assert len(list(data_directories.maps.glob('*'))) == 1
    assert slam_manager.localization_process.is_running()
    assert data_directories.segmentation.exists()

    # undeploy_map
    slam_manager._redis_listener._queue.put({'data': 'undeploy_map 0'})
    await trio.sleep(30)

    # check the reset state
    assert slam_manager.map_name is None
    assert len(list(data_directories.maps.glob('*'))) == 0
    assert not slam_manager.localization_process.is_running()
    assert not data_directories.segmentation.exists()
    assert is_marker_removed()
