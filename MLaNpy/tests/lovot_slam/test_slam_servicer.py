import hashlib
import importlib
import json
import pathlib
import re
import shutil
import tempfile
import time
from concurrent import futures
from typing import IO, Iterable, Optional

import grpc
import numpy as np
import pytest
import trio
import yaml
from google.protobuf import empty_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from lovot_apis.lovot_spike_svcs.slam.slam_pb2 import (BuildSingleMissionMapRequest, CheckExploreRightsRequest,
                                                       Coordinate, DownloadFileRequest, DownloadFileResponse,
                                                       FileInformation, GetFrontierRequest, GetSpotsRequest,
                                                       PullMapRequest, SetSpotsRequest, Spot, UploadFileRequest)
from lovot_apis.lovot_spike_svcs.slam.slam_pb2_grpc import SlamStub, add_SlamServicer_to_server

import lovot_slam.env
import lovot_slam.exploration.exploration_status
import lovot_slam.slam_servicer
import lovot_slam.utils
from lovot_slam.exploration.exploration_status import ExplorationStatus, ExplorationStatusMonitor
from lovot_slam.exploration.exploration_token import MAX_DAILY_TOKEN, ExplorationTokenManager
from lovot_slam.map_build.map_build_metrics import MapBuildAttemptResultsMetric
from lovot_slam.map_build.request_queue import RequestQueue
from lovot_slam.redis import create_ltm_client, redis_keys
from lovot_slam.redis.keys import COLONY_ID_KEY
from lovot_slam.slam_servicer import SlamServicer
from lovot_slam.utils.file_util import get_file_md5sum, unzip_archive
from lovot_slam.utils.segmentation import SEGMENTATION_VERSION

from .client.util import _get_free_port
from .exploration.test_frontier_search import DOOR_CLOSE_ROSMAP, DOOR_OPEN_ROSMAP
from .stub import PseudoSlamManager

_LTM_TOKEN_TTL_KEY = 'localiation_test:slam:exploration_token_ttl'
_LTM_CONTINUOUS_FAIL_COUNTS_KEY = 'localiation_test:slam:map_build:continuous_fail_counts'

# TODO: make dataset package
BASE_DIR = pathlib.Path(__file__).parent
DATASET_ROOT = BASE_DIR.parents[2] / 'dataset'

DUMMY_DATA_ROOT = pathlib.Path('/tmp/localization')

MAP_YAML = BASE_DIR / '2d_map' / 'map.yaml'
grpc_server = None
ltm_client = create_ltm_client()
requests = RequestQueue(ltm_client)
slam_manager = PseudoSlamManager()
exploration_status_monitor = None
exploration_token_manager = None
slam_servicer = None
ltm_client = None

testing_port = None


def setup_module():
    global grpc_server, slam_servicer, ltm_client, exploration_status_monitor, exploration_token_manager
    lovot_slam.env.SPIKE_LOCALHOST = 'localhost'
    lovot_slam.env.DataDirectories.DATA_ROOT = DUMMY_DATA_ROOT
    if lovot_slam.env.DataDirectories.DATA_ROOT.exists():
        shutil.rmtree(lovot_slam.env.DataDirectories.DATA_ROOT)
    maps_root = lovot_slam.env.data_directories.maps
    bags_root = lovot_slam.env.data_directories.bags
    bag_utils = lovot_slam.utils.map_utils.BagUtils(bags_root)
    bag_utils.create_directory()
    map_utils = lovot_slam.utils.map_utils.MapUtils(maps_root, bags_root)
    map_utils.create_directory()
    importlib.reload(lovot_slam.utils)

    ltm_client = create_ltm_client()
    ltm_client.delete(_LTM_TOKEN_TTL_KEY)
    ExplorationTokenManager._LTM_TOKEN_TTL_KEY = _LTM_TOKEN_TTL_KEY
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)
    MapBuildAttemptResultsMetric._LTM_CONTINUOUS_FAIL_COUNTS_KEY = _LTM_CONTINUOUS_FAIL_COUNTS_KEY

    exploration_status_monitor = ExplorationStatusMonitor()
    build_metric = MapBuildAttemptResultsMetric()
    exploration_token_manager = ExplorationTokenManager(ltm_client,
                                                        exploration_status_monitor,
                                                        build_metric,
                                                        slam_manager.is_processing_map,
                                                        False,
                                                        RequestQueue(ltm_client),
                                                        map_utils)
    slam_servicer = lovot_slam.slam_servicer.SlamServicer(requests,
                                                          exploration_status_monitor,
                                                          exploration_token_manager)
    # the worker thread is limited to 1 to prohibit concurrent process,
    # because some of the calls treat files and are not thread-safe.
    # process each request one by one.
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
                              options=[('grpc.max_receive_message_length',
                                        lovot_slam.env.GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    serve()
    ltm_client.set(COLONY_ID_KEY, 'testcolonyid')


def teardown_module():
    global ltm_client
    lovot_slam.env.SPIKE_LOCALHOST = 'spike'
    importlib.reload(lovot_slam.utils)
    ltm_client = create_ltm_client()
    ltm_client.delete(COLONY_ID_KEY)
    ltm_client.delete(_LTM_TOKEN_TTL_KEY)
    ltm_client.delete(_LTM_CONTINUOUS_FAIL_COUNTS_KEY)
    if lovot_slam.env.DataDirectories.DATA_ROOT.exists():
        shutil.rmtree(lovot_slam.env.DataDirectories.DATA_ROOT)
    grpc_server.stop(None)


def serve():
    global requests, grpc_server, slam_servicer, testing_port
    # TODO: rewrite with async
    testing_port = trio.run(_get_free_port)
    add_SlamServicer_to_server(
        slam_servicer, grpc_server
    )
    grpc_server.add_insecure_port(f'[::]:{testing_port}')
    grpc_server.start()
    # wait for the server starts
    with open_channel() as channel:
        grpc.channel_ready_future(channel).result(timeout=20)


def open_channel():
    return grpc.insecure_channel(
        target=f'localhost:{testing_port}',
        options=[('grpc.enable_retries', 0), ('grpc.keepalive_timeout_ms', 5000)])


@pytest.fixture
def setup_slam_spots():
    r = create_ltm_client()

    def _write_spots(spot_list):
        for spot in spot_list:
            spot_name = spot["name"]
            pose_str = spot["coordinate"]
            key = redis_keys.spot(spot_name)
            spot = {
                "id": spot_name,
                "name": spot_name,
                "coordinate": pose_str
            }
            r.hset(key, mapping=spot)

    yield _write_spots

    r.delete(redis_keys.spot('*'))


@pytest.fixture
def setup_unwelcomed_area():
    r = create_ltm_client()
    key = redis_keys.unwelcomed_area

    def _write_unwelcomed_area(unwelcomed_area):
        if unwelcomed_area is not None:
            r.set(key, unwelcomed_area)
        else:
            r.delete(key)

    yield _write_unwelcomed_area

    r.delete(key)


@pytest.fixture
def dummy_data_directory():
    maps_root = lovot_slam.env.data_directories.maps
    bags_root = lovot_slam.env.data_directories.bags
    shutil.rmtree(maps_root)
    shutil.rmtree(bags_root)
    shutil.copytree(DATASET_ROOT / 'dummy' / 'maps', maps_root)
    shutil.copytree(DATASET_ROOT / 'dummy' / 'rosbag', bags_root)
    return (maps_root, bags_root)


@pytest.fixture
def temp_data_directory(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        monkeypatch.setattr(lovot_slam.env.DataDirectories, 'DATA_ROOT', tmpdir)
        yield tmpdir


@pytest.fixture
def slam_servicer_stub():
    with open_channel() as channel:
        yield SlamStub(channel)


def test_build_single_mission_map(temp_data_directory, slam_servicer_stub):
    bags_root = lovot_slam.env.data_directories.bags
    bags_root.mkdir(parents=True, exist_ok=True)

    def create_bag_and_request(map_name):
        bag_file = bags_root / f"{map_name}.bag"
        bag_file.touch()
        req = BuildSingleMissionMapRequest(map_name=map_name)
        slam_servicer_stub.BuildSingleMissionMap(req)
        return bag_file

    # Request queue grows up to MAX_DAILY_TOKEN
    for i in range(MAX_DAILY_TOKEN):
        map_name = f"dummy_{i}"
        bag_file = create_bag_and_request(map_name)

        last_request = slam_servicer.requests._queue.queue[-1]
        assert last_request[1].map_name == map_name
        assert bag_file.exists()

    # The queue size should be MAX_DAILY_TOKEN
    assert len(slam_servicer.requests._queue.queue) == MAX_DAILY_TOKEN

    # Request queue is full, so the last request is not added
    map_name = "dummy_x"
    bag_file = create_bag_and_request(map_name)

    last_request = slam_servicer.requests._queue.queue[-1]
    assert last_request[1].map_name != map_name
    assert not bag_file.exists()


@pytest.mark.parametrize("spot_list", [
    ([{"name": "nest", "coordinate": "0,0,0,0,0,0,1"}]),
    ([{"name": "entrance", "coordinate": "1,2,3,0.0,0,0,1"}]),
    ([{"name": "nest", "coordinate": "1,2,3,4,5,6,7"},
      {"name": "entrance", "coordinate": "2,3,4,5,6,7,8"}]),
])
def test_get_spots(setup_slam_spots, spot_list):
    setup_slam_spots(spot_list)

    with open_channel() as channel:
        stub = SlamStub(channel)
        req = GetSpotsRequest()
        req.spot_names.extend([spot["name"] for spot in spot_list])
        res = stub.GetSpots(req)

    for i, spot in enumerate(spot_list):
        assert res.spots[i].name == spot["name"]
        assert res.spots[i].coordinate == SlamServicer.coordinate_from_string(spot["coordinate"])


@pytest.mark.parametrize("spot_list", [
    ([{"name": "nest", "coordinate": "0,0,0,0,0,0,1"}]),
    ([{"name": "entrance", "coordinate": "1,2,3,0.0,0,0,1"}]),
    ([{"name": "nest", "coordinate": "1,2,3,4,5,6,7"},
      {"name": "entrance", "coordinate": "2,3,4,5,6,7,8"}]),
])
def test_set_spots(spot_list):
    req = SetSpotsRequest()
    spots = [Spot(name=spot["name"],
                  coordinate=SlamServicer.coordinate_from_string(spot["coordinate"]))
             for spot in spot_list]

    with open_channel() as channel:
        stub = SlamStub(channel)
        req = SetSpotsRequest()
        req.spots.extend(spots)
        _ = stub.SetSpots(req)

    r = create_ltm_client()
    for i, spot in enumerate(spot_list):
        key = redis_keys.spot(spot["name"])
        spot_read = r.hgetall(key)
        assert spot_read["id"] == spot["name"]
        assert spot_read["name"] == spot["name"]
        assert list(map(float, spot_read["coordinate"].split(","))) \
            == list(map(float, spot["coordinate"].split(",")))


@pytest.mark.parametrize("teststr", [
    '[{"shape": "rectangle", "vertices": [[1., 2.], [3., 4.], [5., 6.], [7., 8.]}]',
    'tom-spike sync. of unwelcomed area does not validate the datum. it can be any string',
    '',
    None,
])
def test_get_unwelcomed_area(setup_unwelcomed_area, teststr):
    setup_unwelcomed_area(teststr)

    with open_channel() as channel:
        stub = SlamStub(channel)
        res = stub.GetUnwelcomedArea(empty_pb2.Empty())
    assert res.unwelcomed_area == (teststr or '')


@pytest.mark.parametrize("state,elapsed,can_explore,should_issue,expected_token", [
    (PseudoSlamManager.STATE_BAG_CONVERSION, 3500, False, False, r""),
    (PseudoSlamManager.STATE_BAG_CONVERSION, 3500, True, False, r""),
    (PseudoSlamManager.STATE_BAG_CONVERSION, 3700, False, False, r""),
    (PseudoSlamManager.STATE_IDLE, 3500, False, False, r""),
    (PseudoSlamManager.STATE_IDLE, 3500, True, False, r""),
    (PseudoSlamManager.STATE_IDLE, 3700, False, False, r""),
    (PseudoSlamManager.STATE_BAG_CONVERSION, 3700, True, False, r""),
    (PseudoSlamManager.STATE_IDLE, 3700, True, True, r"[0-9]+\.[0-9]+"),
])
def test_check_explore_rights(monkeypatch, dummy_data_directory,
                              state, elapsed, can_explore, should_issue, expected_token):
    global exploration_token_manager
    slam_manager.change_state(state)
    exploration_token_manager._token_timestamp = time.time() - elapsed
    monkeypatch.setattr(ExplorationStatus, 'can_explore', lambda x: can_explore)

    with open_channel() as channel:
        stub = SlamStub(channel)
        req = CheckExploreRightsRequest(ghost_id="ghost_id")
        res = stub.CheckExploreRights(req)

    assert res.success == should_issue
    assert re.match(expected_token, res.token)


def pull_map(map_name, filepath):
    with open_channel() as channel:
        stub = SlamStub(channel)
        req = PullMapRequest(map_name=map_name)
        with open(filepath, 'wb') as f:
            it = stub.PullMap(req)
            md5_ret = ''
            for r in it:
                f.write(r.data)
                md5_ret = r.md5
    return md5_ret


def test_pull_map(dummy_data_directory):
    global slam_servicer
    maps_root: pathlib.Path = dummy_data_directory[0]
    map_name = "20190714_090357_6d426e27972055e9b2a79514d707d2b0"
    filepath = "/tmp/test.zip"

    with open_channel() as channel:
        stub = SlamStub(channel)
        req = PullMapRequest(map_name=map_name)
        with open(filepath, 'wb') as f:
            it = stub.PullMap(req)
            md5_ret = ''
            for r in it:
                f.write(r.data)
                md5_ret = r.md5

    with open(filepath, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    assert md5 == md5_ret
    # check file removed from spike
    assert not (maps_root / f'{map_name}.zip').is_file()


def test_pull_map_concurrently(dummy_data_directory):
    """Test to try PullMap from multiple sessions concurrently.
    the worker thread of gRPC server is single, so it processes one by one.
    """
    global slam_servicer
    maps_root: pathlib.Path = dummy_data_directory[0]
    map_name = "20190714_090357_6d426e27972055e9b2a79514d707d2b0"
    sessions = 10
    filepaths = [f"/tmp/test{i}.zip" for i in range(sessions)]

    executor = futures.ThreadPoolExecutor(max_workers=sessions)
    future_list = []

    for filepath in filepaths:
        future = executor.submit(pull_map, map_name, filepath)
        future_list.append(future)

    for i, future in enumerate(future_list):
        md5_ret = future.result()
        with open(filepaths[i], 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        assert md5 == md5_ret

    # check file removed from spike
    assert not (maps_root / f'{map_name}.zip').is_file()


@pytest.fixture
def mock_map_data(monkeypatch):
    map_name = 'dummy_map'
    map_dir = lovot_slam.env.data_directories.maps / map_name

    # mock dummy map directory
    map_dir.mkdir(parents=True, exist_ok=True)
    with open(map_dir / 'lovot_slam.yaml', 'w') as f:
        yaml.safe_dump({'lovot_slam': {'version': 4}}, f)
    with open(map_dir / 'md5sum_list.yaml', 'w') as f:
        yaml.safe_dump({'lovot_slam.yaml': get_file_md5sum(map_dir / 'lovot_slam.yaml')}, f)

    yield lovot_slam.env.data_directories.maps, map_name

    shutil.rmtree(lovot_slam.env.data_directories.maps)


@pytest.fixture
def temporary_directory():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


@pytest.mark.parametrize("map,frontier", [
    (DOOR_OPEN_ROSMAP, np.array((0.0, 0.55))),  # a map with some frontiers
    (DOOR_CLOSE_ROSMAP, None),  # a map without frontier
    (None, None),  # query with map name that doesn't exist on the nest
])
def test_get_frontier(mock_map_data, map, frontier):
    map_name = 'unknown_map'
    if map:
        maps_root, map_name = mock_map_data
        map.as_occupancy_grid().save(maps_root / map_name / '2d_map' / 'map.yaml')

    # req.map_name = map_name
    coordinate = Coordinate(px=0.0, py=0.0, pz=0.0,
                            ox=0.0, oy=0.0, oz=0.0, ow=1.0)
    with open_channel() as channel:
        try:
            stub = SlamStub(channel)
            req = GetFrontierRequest(map_name=map_name, pose=coordinate)
            res = stub.GetFrontier(req)
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
                print('planning failed.')
            else:
                raise

    obtained_frontier = np.array([res.pose[0].px, res.pose[0].py]) if res.pose else None

    assert np.all(np.isclose(obtained_frontier, frontier)) if frontier is not None else \
        obtained_frontier == frontier


def test_upload_file():
    lovot_slam.env.data_directories.tmp.mkdir(exist_ok=True, parents=True)

    tmp_filepath = pathlib.Path('/tmp/test_file')
    with open(tmp_filepath, 'wb') as f:
        for i in range(1000 * 1000):  # 10 MBytes
            f.write(b'0123456789')

    def gen(f: IO, information: FileInformation) -> UploadFileRequest:
        chunksize = 2000000
        while True:
            d = f.read(chunksize)
            yield UploadFileRequest(data=d, information=information)
            information = None
            if len(d) == 0:
                # send 0-byte datum at last, for confirmation of end
                break

    with open_channel() as channel:
        stub = SlamStub(channel)
        md5 = get_file_md5sum(tmp_filepath)
        timestamp = tmp_filepath.stat().st_ctime
        timestamp = Timestamp(seconds=int(timestamp),
                              nanos=int((timestamp % 1) * 1e9))
        metadata = {'foo': 'bar'}
        jsonized_metadata = json.dumps(metadata)
        information = FileInformation(type='',
                                      name=tmp_filepath.name,
                                      timestamp=timestamp,
                                      md5=md5,
                                      metadata=jsonized_metadata)
        with open(tmp_filepath, 'rb') as f:
            ret = stub.UploadFile(gen(f, information))

    assert ret.status == 'ok'
    assert ret.message == 'no operation has been done'


def test_download_file(temporary_directory: pathlib.Path):
    original_dir = lovot_slam.env.data_directories.segmentation
    original_dir.mkdir()

    # create dummy data
    dummy_contents = {
        'segmentation.json': json.dumps({'version': SEGMENTATION_VERSION}),
        'segmentation.png': 'dummy image content',
    }
    for filename, content in dummy_contents.items():
        with open(original_dir / filename, 'w') as f:
            f.write(content)
    metadata = json.dumps({'version': SEGMENTATION_VERSION, 'include_png': True})

    # download file
    information: Optional[FileInformation] = None
    # create temporary file
    archive_path = temporary_directory / 'archive'
    with open_channel() as channel:
        stub = SlamStub(channel)
        req = DownloadFileRequest(type='segmentation', metadata=metadata)
        with open(archive_path, 'wb') as f:
            ite: Iterable[DownloadFileResponse] = stub.DownloadFile(req)
            for r in ite:
                f.write(r.data)
                if r.WhichOneof('information_optional') == 'information':
                    information = r.information

    assert get_file_md5sum(archive_path) == information.md5

    # check contents
    destination_dir = temporary_directory / 'destination'
    destination_dir.mkdir()
    unzip_archive(archive_path, destination_dir)

    for filename, content in dummy_contents.items():
        with open(destination_dir / filename, 'r') as f:
            assert f.read() == content

    shutil.rmtree(original_dir)


def test_download_file_with_invalid_type():
    # download file
    with open_channel() as channel:
        stub = SlamStub(channel)
        req = DownloadFileRequest(type='invalid')
        with pytest.raises(grpc.RpcError) as exc:
            it: Iterable[DownloadFileResponse] = stub.DownloadFile(req)
            for r in it:
                pass
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_download_file_without_file():
    original_dir = lovot_slam.env.data_directories.segmentation
    if original_dir.exists():
        shutil.rmtree(original_dir)

    # download file
    with open_channel() as channel:
        stub = SlamStub(channel)
        req = DownloadFileRequest(type='segmentation')
        with pytest.raises(grpc.RpcError) as exc:
            it: Iterable[DownloadFileResponse] = stub.DownloadFile(req)
            for r in it:
                pass
        print(exc.value)
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND
