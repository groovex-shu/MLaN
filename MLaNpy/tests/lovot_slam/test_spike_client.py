import base64
import hashlib
import io
import json
import pathlib
import pickle
import tempfile
import zipfile
from typing import Tuple
from unittest.mock import patch
import os

import cv2
import numpy as np
import purerpc
import pytest
import trio
import yaml
from google.protobuf import empty_pb2
from h2.connection import ConnectionState

from lovot_apis.lovot_spike_svcs.slam.slam_grpc import SlamServicer
from lovot_apis.lovot_spike_svcs.slam.slam_pb2 import (BuildSingleMissionMapRequest, CheckExploreRightsRequest,
                                                       CheckExploreRightsResponse, Coordinate, DownloadFileResponse,
                                                       FileInformation, GetFrontierRequest, GetFrontierResponse,
                                                       GetLatestMapResponse, GetSpotsRequest, GetSpotsResponse,
                                                       GetUnwelcomedAreaResponse, ListMapResponse, Spot,
                                                       UploadFileResponse)

from lovot_slam.client.spike_host_monitor import SpikeHostMonitor
from lovot_slam.slam_servicer import DownloadFileRequest, DownloadFileType, UploadFileType
from lovot_slam.spike_client import SlamServicerClient, open_slam_servicer_client
from lovot_slam.utils.exceptions import SlamProcedureCallError, SlamTransferError
from lovot_slam.utils.segmentation import SEGMENTATION_VERSION

from .client.test_spike_host_monitor import mock_socket_getaddreinfo, redis_fixture  # noqa: F401
from .client.util import _get_free_port, open_servicer_and_client


class MockSlamServicer(SlamServicer):
    def __init__(self) -> None:
        super().__init__()

        self.response = None
        self.request = None

    async def BuildSingleMissionMap(self, request):
        self.request = request
        return empty_pb2.Empty()

    async def ListMap(self, request):
        self.request = request
        return self.response

    async def GetFrontier(self, request):
        self.request = request
        return self.response

    async def GetLatestMap(self, request):
        self.request = request
        return self.response

    async def CheckExploreRights(self, request):
        self.request = request
        return self.response

    async def GetSpots(self, request):
        self.request = request
        return self.response

    async def GetUnwelcomedArea(self, request):
        self.request = request
        return self.response

    async def UploadFile(self, input_messages):
        self.request = {}
        self.request['data'] = bytes()
        async for message in input_messages:
            if message.WhichOneof('information_optional') == 'information':
                self.request['information'] = message.information
            self.request['data'] += message.data
            self.request['empty_end'] = len(message.data) == 0
        return self.response

    async def DownloadFile(self, message):
        self.request = message
        data = self.response['data']
        information = self.response['information']
        while len(data):
            yield DownloadFileResponse(data=data[:10], information=information)
            data = data[10:]


class MockSlamServicerFailure(SlamServicer):
    def __init__(self) -> None:
        super().__init__()

        self.response = None
        self.request = None

    async def GetFrontier(self, request):
        self.request = request
        raise purerpc.NotFoundError

    async def UploadFile(self, input_messages):
        async for message in input_messages:
            print('UploadFile')
            await trio.sleep(100)
        return self.response

    async def DownloadFile(self, message):
        try:
            metadata = json.loads(message.metadata)
            status_code = pickle.loads(base64.b64decode(metadata['exception']))
            raise status_code
        except KeyError:
            raise purerpc.NotFoundError
        yield DownloadFileResponse()


class MockSlamServicerTimeout(SlamServicer):
    _WAIT_INTERVAL = 100

    def __init__(self) -> None:
        super().__init__()

    async def DownloadFile(self, message):
        for _ in range(10):
            yield DownloadFileResponse(data='0'.encode('utf-8') * 100,
                                       information=FileInformation())
            await trio.sleep(self._WAIT_INTERVAL)


@pytest.fixture(name='mock_servicer_and_client')
async def open_mock_servicer_and_client(monkeypatch, nursery):
    async with open_servicer_and_client(MockSlamServicer, open_slam_servicer_client) as (mock_service, client):
        yield mock_service, client


@pytest.fixture(name='failure_servicer_and_client')
async def open_mock_failure_servicer_and_client(monkeypatch, nursery):
    async with open_servicer_and_client(MockSlamServicerFailure, open_slam_servicer_client) as (mock_service, client):
        yield mock_service, client


@pytest.fixture(name='timeout_servicer_and_client')
async def open_mock_timeout_servicer_and_client(monkeypatch, nursery):
    async with open_servicer_and_client(MockSlamServicerTimeout, open_slam_servicer_client) as (mock_service, client):
        yield mock_service, client


async def test_no_server(autojump_clock):
    port = await _get_free_port()
    async with open_slam_servicer_client(host='localhost', port=port) as client:
        with pytest.raises(SlamProcedureCallError):
            await client.build_single_mission_map('20230122_123456')


async def test_build_single_mission_map(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    await client.build_single_mission_map('20230122_123456')

    assert mock_servicer.request == BuildSingleMissionMapRequest(map_name='20230122_123456')


async def test_map_list(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = ListMapResponse(map_name=['testmap1', 'testmap2'])
    map_list = await client.map_list()

    assert mock_servicer.request == empty_pb2.Empty()
    assert map_list == ['testmap1', 'testmap2']


async def test_get_frontier(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = GetFrontierResponse(
        map_name='testmap1',
        pose=[Coordinate(px=4, py=5, pz=6, ox=0, oy=0, oz=0, ow=1),
              Coordinate(px=7, py=8, pz=9, ox=0, oy=0, oz=0, ow=1)])
    frontiers = await client.get_frontier('testmap1', [1, 2, 3, 1, 0, 0, 0])

    assert mock_servicer.request == \
        GetFrontierRequest(map_name='testmap1', pose=Coordinate(px=1, py=2, pz=3, ox=1, oy=0, oz=0, ow=0))
    assert np.isclose(np.array(frontiers),
                      np.array([(4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0),
                                (7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0)])).all()


async def test_get_latest_map(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = GetLatestMapResponse(map_name='testmap1')
    map_name = await client.get_latest_map()

    assert mock_servicer.request == empty_pb2.Empty()
    assert map_name == 'testmap1'


async def test_check_explore_rights(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client

    mock_servicer.response = CheckExploreRightsResponse(success=True, token='testtoken')
    success, token = await client.check_explore_rights('ghost_id')
    assert mock_servicer.request == CheckExploreRightsRequest(ghost_id='ghost_id')
    assert success
    assert token == 'testtoken'

    mock_servicer.response = CheckExploreRightsResponse(success=False, token='')
    success, token = await client.check_explore_rights('ghost_id')
    assert mock_servicer.request == CheckExploreRightsRequest(ghost_id='ghost_id')
    assert not success
    assert token == ''


async def test_get_spots(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = GetSpotsResponse(
        spots=[Spot(name='spot1',
                    coordinate=Coordinate(px=1, py=2, pz=3, ox=0, oy=0, oz=0, ow=1)),
               Spot(name='spot2',
                    coordinate=Coordinate(px=4, py=5, pz=6, ox=0, oy=0, oz=0, ow=1))])

    spots = await client.get_spots(spot_names=['spot1', 'spot2'])

    assert mock_servicer.request == GetSpotsRequest(spot_names=['spot1', 'spot2'])
    assert spots.keys() == {'spot1', 'spot2'}
    assert spots['spot1']['id'] == 'spot1'
    assert spots['spot1']['name'] == 'spot1'
    assert np.isclose(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
                      np.array(list(map(float, spots['spot1']['coordinate'].split(','))))).all()
    assert spots['spot2']['id'] == 'spot2'
    assert spots['spot2']['name'] == 'spot2'
    assert np.isclose(np.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]),
                      np.array(list(map(float, spots['spot2']['coordinate'].split(','))))).all()


@pytest.mark.parametrize("teststr", [
    '[{"shape": "rectangle", "vertices": [[1., 2.], [3., 4.], [5., 6.], [7., 8.]}]',
    'tom-spike sync. of unwelcomed area does not validate the datum. it can be any string',
    '',
    None,
])
async def test_get_unwelcomed_area(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient], teststr):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = GetUnwelcomedAreaResponse(unwelcomed_area=teststr)

    unwelcomed_area = await client.get_unwelcomed_area()
    assert unwelcomed_area == (teststr or '')


async def test_upload_file(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = UploadFileResponse(status='ok', message='stored')

    data = 'abcdefg'.encode('utf-8') * 1000

    with tempfile.TemporaryDirectory() as dir:
        filename = pathlib.Path(dir) / 'temp.file'
        with open(filename, 'wb') as f:
            f.write(data)
        await client._upload_file(filename, UploadFileType.ACCURACY_MAP)

    assert mock_servicer.request['data'] == data
    assert mock_servicer.request['information'].type == 'accuracy_map'
    assert mock_servicer.request['information'].name == 'temp.file'
    assert mock_servicer.request['information'].md5 == '6f90c9bf2b3b59b99441d8ae38799ec5'
    assert mock_servicer.request['information'].metadata == '{}'


async def test_upload_file_with_timeout(
        failure_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient], monkeypatch):
    monkeypatch.setattr('lovot_slam.spike_client.GRPC_TIMEOUT', 1.0)

    mock_servicer, client = failure_servicer_and_client
    mock_servicer.response = UploadFileResponse(status='ok', message='stored')

    data = 'abcdefg'.encode('utf-8') * 1000

    with tempfile.TemporaryDirectory() as dir:
        filename = pathlib.Path(dir) / 'temp.file'
        with open(filename, 'wb') as f:
            f.write(data)
        with pytest.raises(SlamProcedureCallError):
            await client._upload_file(filename, UploadFileType.ACCURACY_MAP)


async def test_upload_file_without_connection(autojump_clock):
    port = await _get_free_port()
    async with open_slam_servicer_client(host='localhost', port=port) as client:
        with tempfile.TemporaryDirectory() as dir:
            filename = pathlib.Path(dir) / 'temp.file'
            with open(filename, 'wb') as f:
                f.write('abcdefg'.encode('utf-8') * 1000)

            with pytest.raises(SlamProcedureCallError):
                await client._upload_file(filename, UploadFileType.ACCURACY_MAP)


@pytest.fixture
def mock_accuracy_map(monkeypatch):
    with tempfile.TemporaryDirectory() as dir:
        temporary_root = pathlib.Path(dir)
        from lovot_slam.env import DataDirectories, data_directories
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', temporary_root)

        accuracy_map_dir = data_directories.monitor / 'accuracy_map'
        accuracy_map_dir.mkdir(parents=True, exist_ok=True)

        # create dummy map.yaml
        config = {
            'free_thresh': 0.196,
            'image': 'map.pgm',
            'negate': 0,
            'occupied_thresh': 0.65,
            'origin': [-4.25, -3.55, 0],
            'resolution': 0.05
        }
        with open(accuracy_map_dir / 'map.yaml', 'w') as f:
            yaml.safe_dump(config, f)

        # create dummy map.pgm
        image = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
        cv2.imwrite(str(accuracy_map_dir / 'map.pgm'), image)

        print([f for f in temporary_root.glob('**/*')])

        yield config, image


async def test_upload_accuracy_map(mock_accuracy_map,
                                   mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    config, image = mock_accuracy_map
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = UploadFileResponse(status='ok', message='stored')

    ghost_id = "dummy"
    await client.upload_accuracy_map(ghost_id)

    assert mock_servicer.request['information'].type == 'accuracy_map'
    assert mock_servicer.request['information'].name == 'accuracy_map.zip'

    # check yaml file from zip file
    z = zipfile.ZipFile(io.BytesIO(mock_servicer.request['data']))
    assert yaml.safe_load(z.read('map.yaml')) == config
    # check image file from zip file
    nparr = np.frombuffer(z.read('map.pgm'), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    assert np.all(img_np == image)


@pytest.mark.parametrize("bag_file_size,raise_error",
                         [(100, False), (20000000000, True)])
@patch("lovot_slam.utils.map_utils.BagUtils.get_duration")
@patch("lovot_slam.spike_client.get_file_size")
async def test_upload_rosbag(mock_get_file_size, mock_get_duration,
                             mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient],
                             bag_file_size, raise_error):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = UploadFileResponse(status='ok', message='stored')

    data = 'abcdefg'.encode('utf-8') * 1000

    with tempfile.TemporaryDirectory() as dir:
        filename = pathlib.Path(dir) / '20230124_123456.bag'
        with open(filename, 'wb') as f:
            f.write(data)

        mock_get_duration.return_value = 10.0
        mock_get_file_size.return_value = bag_file_size
        if raise_error:
            with pytest.raises(RuntimeError):
                await client.upload_rosbag('20230124_123456', filename)
            assert not os.path.exists(filename)
            return
        else:
            await client.upload_rosbag('20230124_123456', filename)

    assert mock_servicer.request['data'] == data
    assert mock_servicer.request['information'].type == 'rosbag'
    assert mock_servicer.request['information'].name == '20230124_123456.bag'
    assert mock_servicer.request['information'].md5 == '6f90c9bf2b3b59b99441d8ae38799ec5'
    assert mock_servicer.request['information'].metadata == '{"map_name": "20230124_123456"}'


async def test_download_file(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = {}
    mock_servicer.response['information'] = FileInformation(
        name='test.txt',
        type='text/plain',
        metadata='{"meta": "data"}',
        md5='6f90c9bf2b3b59b99441d8ae38799ec5',
    )
    mock_servicer.response['data'] = 'abcdefg'.encode('utf-8') * 1000

    with tempfile.TemporaryDirectory() as dir:
        archive_path = pathlib.Path(dir) / 'temp.file'
        ret = await client._download_file(archive_path, DownloadFileType.MAP, request_metadata={'name': 'map'})

        with open(archive_path, 'rb') as f:
            data = f.read()

    assert mock_servicer.request == DownloadFileRequest(type='map', metadata='{"name": "map"}')

    assert data == mock_servicer.response['data']
    assert ret == {"meta": "data"}


async def test_download_file_timeout(timeout_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient],
                                     monkeypatch):
    _, client = timeout_servicer_and_client

    # NOTE: it requires actual time of 1.0 sec
    # autojump_clock can't be used, because it's not applied to the server side
    monkeypatch.setattr('lovot_slam.spike_client.GRPC_TIMEOUT', 1.0)

    with tempfile.TemporaryDirectory() as dir:
        archive_path = pathlib.Path(dir) / 'temp.file'
        with pytest.raises(SlamProcedureCallError, match=r'gRPC timeout error'):
            await client._download_file(archive_path, DownloadFileType.MAP, request_metadata={'name': 'map'})


async def test_download_file_invalid_md5(mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client
    mock_servicer.response = {}
    mock_servicer.response['information'] = FileInformation(
        name='test.txt',
        type='text/plain',
        metadata='{"meta": "data"}',
        md5='6f90c9bf2b3b59b99441d8ae38799ec5',
    )
    mock_servicer.response['data'] = 'abcdefg'.encode('utf-8') * 900

    with tempfile.TemporaryDirectory() as dir:
        archive_path = pathlib.Path(dir) / 'temp.file'
        with pytest.raises(SlamTransferError):
            await client._download_file(archive_path, DownloadFileType.MAP, request_metadata={'name': 'map'})


@pytest.mark.parametrize("exception", [
    purerpc.NotFoundError,
    purerpc.InvalidArgumentError,
])
async def test_download_file_exception(exception,
                                       failure_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = failure_servicer_and_client

    metadata = {'exception': base64.b64encode(pickle.dumps(exception)).decode('ascii')}

    with tempfile.TemporaryDirectory() as dir:
        archive_path = pathlib.Path(dir) / 'temp.file'
        with pytest.raises(SlamTransferError):
            await client._download_file(archive_path, DownloadFileType.MAP, request_metadata=metadata)


async def test_download_segmentation(monkeypatch,
                                     mock_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = mock_servicer_and_client

    with tempfile.TemporaryDirectory() as dir:
        from lovot_slam.env import DataDirectories, data_directories
        temporary_root = pathlib.Path(dir)
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', temporary_root)
        data_directories.tmp.mkdir()

        # create dummy archive file in memory
        dummy_contents = {
            'segmentation.json': json.dumps({'version': SEGMENTATION_VERSION}),
            'segmentation.png': 'dummy image content',
        }
        file = io.BytesIO()
        with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in dummy_contents.items():
                zf.writestr(filename, content)

        # set response
        mock_servicer.response = {}
        mock_servicer.response['information'] = FileInformation(
            metadata='{"meta": "data"}',
            md5=hashlib.md5(file.getvalue()).hexdigest(),
        )
        mock_servicer.response['data'] = file.getvalue()

        # download
        await client.download_segmentation()

        # check contents
        print([f for f in data_directories.segmentation.iterdir()])
        for filename, content in dummy_contents.items():
            with open(data_directories.segmentation / filename, 'r') as f:
                assert f.read() == content


async def test_download_map_with_failure(monkeypatch,
                                         failure_servicer_and_client: Tuple[MockSlamServicer, SlamServicerClient]):
    mock_servicer, client = failure_servicer_and_client

    with tempfile.TemporaryDirectory() as dir:
        from lovot_slam.env import DataDirectories, data_directories
        temporary_root = pathlib.Path(dir)
        monkeypatch.setattr(DataDirectories, 'DATA_ROOT', temporary_root)
        data_directories.tmp.mkdir()

        with pytest.raises(SlamTransferError) as e_info:
            # this call always fails, because the specified map does NOT exist
            await client.download_map('not_exist_map_name', data_directories.maps)
        print(e_info)

        # check the temporary file is removed
        assert len(list(data_directories.tmp.iterdir())) == 0


@pytest.fixture(name='mock_servicer_and_client_with_host_monitor')
async def open_mock_servicer_and_client_with_host_monitor(nursery):
    port = await _get_free_port()
    server = purerpc.Server(port)
    mock_servicer = MockSlamServicer()
    server.add_service(mock_servicer.service)
    await nursery.start(server.serve_async)

    async with open_slam_servicer_client(host=None, port=port) as client:
        yield mock_servicer, client


async def test_spike_host_monitor(
        mock_servicer_and_client_with_host_monitor: Tuple[MockSlamServicer, SlamServicerClient],
        redis_fixture, mock_socket_getaddreinfo, autojump_clock):  # noqa: F811
    mock_servicer, client = mock_servicer_and_client_with_host_monitor
    mock_servicer.response = ListMapResponse(map_name=['testmap1', 'testmap2'])

    # device_id is not set, and the host is not specified: SHOULD FAIL
    with pytest.raises(SlamProcedureCallError):
        map_list = await client.map_list()

    # device_id is set
    redis_fixture.set(SpikeHostMonitor._COLONY_NEST_DEVICE_ID_KEY, 'DN00000ZZZZZZZZZZZZZ')
    with trio.fail_after(SpikeHostMonitor._RETRY_INTERVAL + 10):
        while client.state != ConnectionState.IDLE:
            await trio.sleep(0.1)

    # SHOULD SUCCEED
    map_list = await client.map_list()
    assert mock_servicer.request == empty_pb2.Empty()
    assert map_list == ['testmap1', 'testmap2']
