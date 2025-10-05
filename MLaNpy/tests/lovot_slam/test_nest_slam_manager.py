import tempfile
import time
from pathlib import Path
from typing import Iterator, List

import prometheus_client
import pytest
import trio

from lovot_slam.env import MAP_MD5SUM_YAML, DataDirectories, data_directories
from lovot_slam.map_build.request_queue import BuildSingleMapOption, RequestQueue, RequestTypes
from lovot_slam.nest_slam_manager import NestSlamManager
from lovot_slam.redis import create_device_client, create_ltm_client, create_stm_client, redis_keys
from lovot_slam.redis.keys import INTENTION_KEY, PHYSICAL_STATE_KEY, ROBOT_MODEL_KEY
from lovot_slam.subprocess.subprocess import SubprocessBase
from lovot_slam.utils.map_utils import MapUtils
from .client.util import _get_free_port
from .mock.mock_map_builder import MockMap, MockMergedMap, MockSingleMissionMap, prepare_dummy_single_merged_maps_pair


class MockBuildMapSubprocess(SubprocessBase):
    def __init__(self, model, output_to_console: bool = False, journal: bool = False) -> None:
        super().__init__(output_to_console)
        self._name = ""
        self._journal = journal

        self._map_utils = MapUtils(data_directories.maps, data_directories.bags)

    def start_bag_conversion(self, original_bag: str, converted_bag: str):
        self._name = "bag_conversion"
        cmd = ['sleep', '0.01']
        if Path(original_bag).exists():
            Path(converted_bag).touch()
        self._start_process(cmd)

    def start_bag_diminish(self, original_bag: str, topics: str, vertices_csv: str, converted_bag: str):
        self._name = "bag_diminish"
        cmd = ['sleep', '0.01']
        if Path(original_bag).exists():
            Path(converted_bag).touch()
        self._start_process(cmd)

    def start_bag_prune(self, original_bag: str, topics: str, converted_bag: str):
        self._name = "bag_prune"
        cmd = ['sleep', '0.01']
        if Path(original_bag).exists():
            Path(converted_bag).touch()
        self._start_process(cmd)

    def start_build_feature_map(self, converted_bag: str, map_dir: str, config_dir: str):
        self._name = "build_feature_map"
        cmd = ['sleep', '0.01']
        map_name = Path(map_dir).stem
        # create mock single mission map
        if Path(converted_bag).exists():
            builder = MockSingleMissionMap(data_directories.maps, map_name)
        builder.create(exist_ok=True)
        self._start_process(cmd)

    def start_scale_map(self, map_name: str, source_maps: List[str], mission_ids: List[str]):
        self._name = "scale_map"
        cmd = ['sleep', '0.01']
        self._start_process(cmd)

    def start_build_dense_map(self, maps_root: str, bags_root: str, config_yaml: str, map_name: str, mission_id: str):
        self.name = "build_dense_map"
        cmd = ['sleep', '0.01']
        self._start_process(cmd)

    def start_merge_feature_maps(self, input_map: str, output_map: str, maps_to_append: List[str]) -> None:
        self.name = "merge_feature_maps"
        cmd = ['sleep', '0.01']
        # create map list
        map_list = self._map_utils.get_source_map_list(Path(input_map).stem)
        map_list = map_list or [Path(input_map).stem]
        map_list += maps_to_append
        source_maps = [MockMap.from_map_dir(data_directories.maps / map_name) for map_name in map_list]
        # create mock merged map
        builder = MockMergedMap(data_directories.maps, source_maps, map_name=Path(output_map).stem)
        builder.create(exist_ok=True)
        self._start_process(cmd)

    def start_merge_dense_maps(self, map_name: str, source_maps: List[str], mission_ids: List[str]):
        self._name = "merge_dense_map"
        cmd = ['sleep', '0.01']
        self._start_process(cmd)


async def _mock_filter_topics(self_, bag_name: str, expression: str) -> bool:
    await trio.sleep(0)


def teardown_module():
    stm_client = create_stm_client()
    stm_client.delete(PHYSICAL_STATE_KEY)
    stm_client.delete(INTENTION_KEY)


@pytest.fixture(name='mock_servicer_port')
async def fixture_servicer_port(monkeypatch):
    ports = set()
    while len(ports) < 2:
        port = await _get_free_port()
        if port in ports:
            continue
        ports.add(port)

    monkeypatch.setattr('lovot_slam.nest_slam_manager.GRPC_PORT', ports.pop())
    monkeypatch.setattr('lovot_slam.service.navigation_service._NAVIGATION_SERVICE_PORT', ports.pop())


@pytest.fixture(name='spike_model')
def fixture_spike_model():
    device_client = create_device_client()
    device_client.set(ROBOT_MODEL_KEY, 'ln101')
    yield
    device_client.delete(ROBOT_MODEL_KEY)


@pytest.fixture(name='shaun_model')
def fixture_shaun_model():
    device_client = create_device_client()
    # FIXME: use shaun model when it is available (lv110)
    device_client.set(ROBOT_MODEL_KEY, 'lv101')
    yield
    device_client.delete(ROBOT_MODEL_KEY)


@pytest.fixture(name='redis_sleep_time')
def fixture_redis_sleep_time():
    ltm_client = create_ltm_client()
    key_values = {
        'colony:sleep_time:start': '22:00',
        'colony:sleep_time:end': '7:00',
    }
    for key, value in key_values.items():
        ltm_client.set(key, value)
    yield
    for key in key_values.keys():
        ltm_client.delete(key)


@pytest.fixture
def prepare_prometheus_client():
    def unregister_prometheus_collectors():
        collectors = list(prometheus_client.REGISTRY._collector_to_names.keys())
        for collector in collectors:
            prometheus_client.REGISTRY.unregister(collector)

    unregister_prometheus_collectors()
    yield
    unregister_prometheus_collectors()


@pytest.fixture
def dummy_data_directory(monkeypatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', tmpdir)
        yield tmpdir


@pytest.fixture(name='ltm_client')
def ltm_client_with_cleanup():
    ltm_client = create_ltm_client()

    ltm_client.delete(RequestQueue._PERSIST_REDIS_KEY)
    yield ltm_client
    ltm_client.delete(RequestQueue._PERSIST_REDIS_KEY)


async def _wait_for_map_build_finished(slam_manager: NestSlamManager):
    # wait until is_processing_map continues failed in two seconds
    with trio.fail_after(60 * 60):
        while True:
            # actual sleep time is needed, because of subprocess requires actual time
            time.sleep(0.01)
            await trio.sleep(2)
            if slam_manager._map_builder.is_processing_map():
                continue
            await trio.sleep(2)
            if slam_manager._map_builder.is_processing_map():
                continue
            return


async def test_build_map(dummy_data_directory, prepare_prometheus_client,
                         autojump_clock, mock_servicer_port, spike_model, nursery, monkeypatch):
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)
    monkeypatch.setattr(
        'lovot_slam.utils.map_utils.BagUtils.filter_topics', _mock_filter_topics)

    # reset repair flag to allow map building
    stm_client = create_stm_client()
    stm_client.set("under_repair", "0")
    stm_client.set("cloud:ghost:status", "identified")

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # emulate bag transfer
    map_name = '20000101_000000'
    (data_directories.bags / f'{map_name}.bag').touch()

    # push build map request to the queue
    slam_manager.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name))

    await _wait_for_map_build_finished(slam_manager)

    # expect a merged maps and two single mission maps
    assert len(slam_manager.map_utils.get_map_list()) == 2
    assert len(slam_manager.map_utils.get_merged_map_list()) == 1


async def test_preserve_map_on_initialization(dummy_data_directory, prepare_prometheus_client,
                                              spike_model, mock_servicer_port,
                                              autojump_clock, nursery):
    _, latest_merge_map = prepare_dummy_single_merged_maps_pair()

    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(10)

    assert latest_merge_map == slam_manager.map_utils.get_latest_merged_map()


@pytest.mark.parametrize('del_tail_chars', [
    (15),  # invalid md5sum length of the last key
    (33),  # empty md5sum of the last key
    (34),  # invalid yaml format
])
async def test_handle_broken_md5sum_map(dummy_data_directory, prepare_prometheus_client,
                                        autojump_clock, spike_model, mock_servicer_port, nursery,
                                        del_tail_chars):
    def break_md5sum_yaml(map_name):
        md5sum_yaml = data_directories.maps / map_name / MAP_MD5SUM_YAML
        contents = md5sum_yaml.read_text().rstrip()
        md5sum_yaml.write_text(contents[:-del_tail_chars])
        print(md5sum_yaml.read_text().split('\n')[-1])

    _, latest_merge_map = prepare_dummy_single_merged_maps_pair()

    break_md5sum_yaml(latest_merge_map)

    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(10)

    # expect the broken map is deleted
    assert slam_manager.map_utils.get_latest_merged_map() == ''
    broken_map_dir = data_directories.maps / latest_merge_map
    assert not broken_map_dir.exists()


async def test_reset(dummy_data_directory, prepare_prometheus_client,
                     autojump_clock, spike_model, mock_servicer_port,
                     ltm_client, nursery):
    def prepare_dummy_map():
        maps_root = data_directories.maps
        single_map = MockSingleMissionMap(maps_root)
        single_map.create()
        data_directories.bags.mkdir(parents=True, exist_ok=True)
        (data_directories.bags / f'{single_map.map_name}_converted.bag').touch()

        merged_map = MockMergedMap(maps_root, source_maps=[single_map])
        merged_map.create()
        return merged_map.map_name

    # preparation of test
    latest_merge_map = prepare_dummy_map()
    ltm_client = create_ltm_client()
    ltm_client.set(redis_keys.unwelcomed_area,
                   '[{"shape":"polygon","vertices":[[-1,-1],[-1,1],[1,1],[1,-1]]}]')

    # run main loop
    slam_manager = NestSlamManager()

    slam_manager.redis_stm.set("cloud:ghost:status", "identified")
    slam_manager.redis_stm.set("under_repair", "0")

    nursery.start_soon(slam_manager.run)
    await trio.sleep(10)

    # check the initial state
    assert latest_merge_map == slam_manager.map_utils.get_latest_merged_map()
    assert data_directories.segmentation.exists()
    assert len(list(data_directories.maps.glob('*'))) == 2
    assert len(list(data_directories.bags.glob('*'))) == 1
    assert ltm_client.exists(redis_keys.map)

    # reset
    slam_manager._redis_listener._queue.put({'data': 'reset 0'})
    await trio.sleep(10)

    # check the reset state
    assert '' == slam_manager.map_utils.get_latest_merged_map()
    assert not data_directories.segmentation.exists()
    assert len(list(data_directories.maps.glob('*'))) == 0
    assert len(list(data_directories.bags.glob('*'))) == 0
    assert not ltm_client.exists(redis_keys.map)


async def test_can_build_map_on_nest(dummy_data_directory, prepare_prometheus_client,
                                     autojump_clock, spike_model, mock_servicer_port, nursery):
    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)

    await trio.sleep(10)
    assert slam_manager._can_build_map

    await trio.sleep(10)
    assert slam_manager._can_build_map


async def test_can_build_map_on_shaun(dummy_data_directory, prepare_prometheus_client,
                                      autojump_clock, shaun_model, mock_servicer_port,
                                      redis_sleep_time, nursery):
    stm_client = create_stm_client()

    def _set_ps_and_intention(physical_state, intention):
        stm_client.set(PHYSICAL_STATE_KEY, physical_state)
        stm_client.set(INTENTION_KEY, intention)

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)

    assert not slam_manager._can_build_map

    _set_ps_and_intention('STANDING', 'DEFAULT')
    await trio.sleep(15 * 60)
    assert not slam_manager._can_build_map

    _set_ps_and_intention('ON_NEST', 'CHARGE')
    await trio.sleep(15 * 60)
    assert not slam_manager._can_build_map

    _set_ps_and_intention('STANDING', 'SLEEPY')
    await trio.sleep(15 * 60)
    assert not slam_manager._can_build_map

    _set_ps_and_intention('ON_NEST', 'SLEEPY')
    await trio.sleep(15 * 60)
    # only at this point, build map is allowed
    assert slam_manager._can_build_map

    _set_ps_and_intention('ON_NEST', 'DEFAULT')
    await trio.sleep(5)
    assert not slam_manager._can_build_map


async def test_cancel_build_map(dummy_data_directory, prepare_prometheus_client, shaun_model,
                                autojump_clock, mock_servicer_port, redis_sleep_time, ltm_client,
                                nursery, monkeypatch):
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)
    monkeypatch.setattr(
        'lovot_slam.utils.map_utils.BagUtils.filter_topics', _mock_filter_topics)

    stm_client = create_stm_client()

    def _set_ps_and_intention(physical_state, intention):
        stm_client.set(PHYSICAL_STATE_KEY, physical_state)
        stm_client.set(INTENTION_KEY, intention)

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # push build map request to the queue
    map_name = '20000101_000000'
    (data_directories.bags / f'{map_name}.bag').touch()
    slam_manager.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name))

    _set_ps_and_intention('STANDING', 'DEFAULT')
    with trio.move_on_after(15 * 60) as cancel_scope:
        while not slam_manager._map_builder.is_processing_map():
            await trio.sleep(0.1)
    assert cancel_scope.cancelled_caught, 'build map should NOT be started'

    _set_ps_and_intention('ON_NEST', 'SLEEPY')
    with trio.move_on_after(15 * 60) as cancel_scope:
        while not slam_manager._map_builder.is_processing_map():
            await trio.sleep(0.1)
    assert not cancel_scope.cancelled_caught, 'build map should be started'

    _set_ps_and_intention('STANDING', 'DEFAULT')
    with trio.move_on_after(15) as cancel_scope:
        while slam_manager._map_builder.is_processing_map():
            await trio.sleep(0.1)
    assert not cancel_scope.cancelled_caught, 'build map should be cancelled'


async def test_multiple_build_map_requests_on_spike(
        dummy_data_directory, prepare_prometheus_client,
        autojump_clock, mock_servicer_port, spike_model, ltm_client, nursery, monkeypatch):
    """Test if unsued bags are removed after map build ON SPIKE only"""
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)
    monkeypatch.setattr(
        'lovot_slam.utils.map_utils.BagUtils.filter_topics', _mock_filter_topics)

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # emulate bag transfer and push build map request to the queue
    map_names = ['20000101_000000', '20000101_000001']
    for map_name in map_names:
        (data_directories.bags / f'{map_name}.bag').touch()
        slam_manager.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name))

    await _wait_for_map_build_finished(slam_manager)

    # expect only one merged maps and a single mission maps
    assert len(slam_manager.map_utils.get_map_list()) == 2
    assert len(slam_manager.map_utils.get_merged_map_list()) == 1


async def test_multiple_build_map_requests_on_shaun(
        dummy_data_directory, prepare_prometheus_client,
        autojump_clock, mock_servicer_port, shaun_model, redis_sleep_time,
        nursery, monkeypatch):
    """Test if the map builder can process multiple requests continuously ON SHAUN only"""
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)
    # patch BagUtils.filter_topics
    monkeypatch.setattr(
        'lovot_slam.utils.map_utils.BagUtils.filter_topics', _mock_filter_topics)
    stm_client = create_stm_client()

    def _set_ps_and_intention(physical_state, intention):
        stm_client.set(PHYSICAL_STATE_KEY, physical_state)
        stm_client.set(INTENTION_KEY, intention)

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # wait for can_build_map triggered
    _set_ps_and_intention('ON_NEST', 'SLEEPY')
    await trio.sleep(15 * 60)

    # emulate bag transfer and push build map request to the queue
    map_names = ['20000101_000000', '20000101_000001']
    for map_name in map_names:
        (data_directories.bags / f'{map_name}.bag').touch()
        slam_manager.requests.push(RequestTypes.BuildMap, BuildSingleMapOption(map_name))

    await _wait_for_map_build_finished(slam_manager)

    # expect two merged maps and two single mission maps
    assert len(slam_manager.map_utils.get_map_list()) == 4
    assert len(slam_manager.map_utils.get_merged_map_list()) == 2


async def test_remove_unused_resources(
        dummy_data_directory, prepare_prometheus_client, shaun_model,
        autojump_clock, mock_servicer_port, redis_sleep_time,
        ltm_client, nursery, monkeypatch):
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)

    # prepare dummy rosbag
    bag_names = ['20240731_000000', '20240731_000001']
    data_directories.bags.mkdir(parents=True, exist_ok=True)
    bags = [data_directories.bags / f'{bag_name}.bag' for bag_name in bag_names]
    for bag in bags:
        bag.touch()

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # expect the bags are removed
    for bag in bags:
        assert not bag.exists()


async def test_remove_unused_resources_excluding_bags_in_requests(
        dummy_data_directory, prepare_prometheus_client, shaun_model,
        autojump_clock, mock_servicer_port, redis_sleep_time,
        ltm_client, nursery, monkeypatch):
    monkeypatch.setattr(
        'lovot_slam.map_build.map_build.BuildMapSubprocess', MockBuildMapSubprocess)

    # prepare dummy rosbag
    bag_names_wo_request = ['20240731_000000', '20240731_000001']
    bag_names_w_request = ['20240731_000002', '20240731_000003']
    bags_wo_request = [data_directories.bags / f'{bag_name}.bag'
                       for bag_name in bag_names_wo_request]
    bags_w_request = [data_directories.bags / f'{bag_name}.bag'
                      for bag_name in bag_names_w_request]
    data_directories.bags.mkdir(parents=True, exist_ok=True)
    for bag in bags_wo_request + bags_w_request:
        bag.touch()

    # prepare the persisted request
    serialized_requests = [f'[0, {{"map_name": "{bag_name}"}}]' for bag_name in bag_names_w_request]
    serialized_requests = '[' + ', '.join(serialized_requests) + ']'
    ltm_client.set(RequestQueue._PERSIST_REDIS_KEY, serialized_requests)

    # run main loop
    slam_manager = NestSlamManager()
    nursery.start_soon(slam_manager.run)
    await trio.sleep(20)

    # expect some bags are removed and some are not
    for bag in bags_wo_request:
        assert not bag.exists()
    for bag in bags_w_request:
        assert bag.exists()


async def test_register_latest_map_no_reset_cloud_map(dummy_data_directory, prepare_prometheus_client, autojump_clock,
                                                      shaun_model, mock_servicer_port, nursery, monkeypatch,
                                                      ltm_client):
    """Test map registration with different sync decisions"""

    # Mock map_utils methods
    mock_reset_called = False

    async def mock_reset_cloud_map():
        nonlocal mock_reset_called
        mock_reset_called = True
        return True

    monkeypatch.setattr(
    "lovot_slam.nest_slam_manager.MAP_UPLOAD_RETRY_INTERVAL", 1)

    slam_manager = NestSlamManager()
    stm_client = create_stm_client()
    monkeypatch.setattr(slam_manager.map_utils, 'reset_cloud_map', mock_reset_cloud_map)
    monkeypatch.setattr(slam_manager.map_utils, 'get_latest_merged_map', lambda: None)  # No map

    # Set unwelcomed area and entrance
    ltm_client.set(redis_keys.unwelcomed_area,
                   '[{"shape":"polygon","vertices":[[-1,-1],[-1,1],[1,1],[1,-1]]}]')
    ltm_client.hset(redis_keys.spot("entrance"), "id", "entrance")
    ltm_client.hset(redis_keys.spot("entrance"), "name", "entrance")
    ltm_client.hset(redis_keys.spot("entrance"), "coordinate", "-3.9056,1.1606,0.0000,0.0000,0.0000,0.3312,-0.9436")

    # Test 1: Sync undetermined - should set retry flag and return
    stm_client.set("cloud:ghost:status", "identifying")

    # run main loop and wait for retrying map upload for some cycles
    nursery.start_soon(slam_manager.run)
    await trio.sleep(15)
    assert not mock_reset_called

    stm_client.set("cloud:ghost:status", "unidentified")
    await trio.sleep(5)
    assert not mock_reset_called

    # set repair flag -> should not reset cloud map
    stm_client.set("cloud:ghost:status", "identified")
    stm_client.set("under_repair", "1")
    await trio.sleep(5)
    assert not mock_reset_called

    # Reset repair flag -> reset cloud map
    stm_client.set("under_repair", "0")
    await trio.sleep(5)
    assert not mock_reset_called

    # unwelcomed area and entrance is not reset
    assert ltm_client.exists(redis_keys.unwelcomed_area)
    assert ltm_client.exists(redis_keys.spot("entrance"))


async def test_register_latest_map_reset_cloud_map_at_startup(dummy_data_directory, prepare_prometheus_client, autojump_clock,
                                                              shaun_model, mock_servicer_port, nursery, monkeypatch,
                                                              ltm_client):
    """Test map registration with different sync decisions"""

    # Mock map_utils methods
    mock_reset_called = False

    async def mock_reset_cloud_map():
        nonlocal mock_reset_called
        mock_reset_called = True
        return True

    monkeypatch.setattr(
    "lovot_slam.nest_slam_manager.MAP_UPLOAD_RETRY_INTERVAL", 1)

    slam_manager = NestSlamManager()
    stm_client = create_stm_client()
    monkeypatch.setattr(slam_manager.map_utils, 'reset_cloud_map', mock_reset_cloud_map)
    monkeypatch.setattr(slam_manager.map_utils, 'get_latest_merged_map', lambda: None)  # No map

    # Set unwelcomed area and entrance
    ltm_client.set(redis_keys.unwelcomed_area,
                   '[{"shape":"polygon","vertices":[[-1,-1],[-1,1],[1,1],[1,-1]]}]')
    ltm_client.hset(redis_keys.spot("entrance"), "id", "entrance")
    ltm_client.hset(redis_keys.spot("entrance"), "name", "entrance")
    ltm_client.hset(redis_keys.spot("entrance"), "coordinate", "-3.9056,1.1606,0.0000,0.0000,0.0000,0.3312,-0.9436")

    # not under repair: sync map with cloud
    stm_client.set("cloud:ghost:status", "identified")
    stm_client.set("under_repair", "0")

    # run main loop and wait for retrying map upload for some cycles
    nursery.start_soon(slam_manager.run)
    await trio.sleep(2)
    assert mock_reset_called

    # unwelcomed area and entrance is reset
    assert not ltm_client.exists(redis_keys.unwelcomed_area)
    assert not ltm_client.exists(redis_keys.spot("entrance"))

