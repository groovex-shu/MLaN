import json
import math
import pathlib
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Tuple

import purerpc
import trio
from async_generator import aclosing
from google.protobuf import empty_pb2
from h2.connection import ConnectionState
from h2.exceptions import ProtocolError
from trio_util import trio_async_generator

from lovot_apis.lovot_spike_svcs.slam.slam_grpc import SlamStub
from lovot_apis.lovot_spike_svcs.slam.slam_pb2 import (BuildSingleMissionMapRequest, CheckExploreRightsRequest,
                                                       Coordinate, DownloadFileRequest, FileInformation,
                                                       GetFrontierRequest, GetSpotsRequest, Spot, UploadFileRequest)

from lovot_slam.client.purerpc_context import GrpcContextManager
from lovot_slam.client.spike_host_monitor import SpikeHostMonitor
from lovot_slam.env import GRPC_PORT, GRPC_STREAM_CHUNK_SIZE, GRPC_TIMEOUT, data_directories
from lovot_slam.slam_servicer import DownloadFileType, SlamServicer, UploadFileType
from lovot_slam.utils.exceptions import SlamProcedureCallError, SlamTransferError
from lovot_slam.utils.file_util import (get_file_md5sum, sync_to_disk, unzip_archive,
                                        verify_md5sum, zip_archive, get_file_size,
                                        remove_file_if_exists)
from lovot_slam.utils.protobuf_util import unix_time_to_pb_timestamp
from lovot_slam.utils.segmentation import SEGMENTATION_VERSION
from lovot_slam.utils.map_utils import BagUtils


_logger = getLogger(__name__)

# NOTE: This timeout is only for downloading files.
# Because the first chunk is transferred after archiving files
# and it takes 20 seconds with the map size of 500 MB,
# the general timeout of 10 seconds is not enough.
# 40 seconds allows 1 GB of map size,
# while the maximum map size is assumed to be about 300~400 MB.
_GRPC_DOWNLOAD_TIMEOUT = 40


def _as_async_generator(wrapped):
    """Provide the contextmanager as an async generator, when using trio_util.trio_async_generator.
    This is a workaround for the fact that trio_util.trio_async_generator wraps as an async context manager
    while purerpc stream requires an async generator (not an async context manager).

    example usage:
    @_as_async_generator
    @trio_async_generator
    async def async_generator(*args, **kwargs):
        ...
    """
    async def wrapper_as_async_generator(*args, **kwargs):
        async with wrapped(*args, **kwargs) as agen:
            async for item in agen:
                yield item

    return wrapper_as_async_generator


@_as_async_generator
@trio_async_generator
async def _timed_async_generator(async_iterable, timeout):
    """
    Wraps an asynchronous iterable and applies a timeout for each item.

    This function creates an asynchronous generator that iterates over the `async_iterable`,
    applying a timeout for each `await` operation when fetching the next item.
    If the timeout is exceeded, the generator raises trio.TooSlowError.

    Example:
        try:
            async for message in _timed_async_generator(async_iterable, timeout=20):
                await f.write(message.data)
        except trio.TooSlowError:
            print("timeout")

    :param async_iterable: An asynchronous iterable to iterate over.
    :param timeout (float): The maximum time (in seconds) to wait for each item.
    :return: The next item from the `async_iterable`.
    :raises: trio.TooSlowError if the timeout is exceeded.
    """
    async with aclosing(async_iterable.__aiter__()) as iterator:
        while True:
            with trio.fail_after(timeout):
                try:
                    message = await iterator.__anext__()
                except StopAsyncIteration:
                    return  # End of iteration
                yield message


def _pose_list_to_list(pose_list) -> List[List[float]]:
    return [list((pose.px, pose.py, pose.pz, pose.ox, pose.oy, pose.oz, pose.ow)) for pose in pose_list]


def _create_fileinformation(filepath: pathlib.Path, file_type: UploadFileType, metadata: Dict = {}) -> FileInformation:
    # prepare infomation
    md5 = get_file_md5sum(filepath)
    timestamp = unix_time_to_pb_timestamp(filepath.stat().st_ctime)
    jsonized_metadata = json.dumps(metadata)
    return FileInformation(type=file_type.value,
                           name=filepath.name,
                           timestamp=timestamp,
                           md5=md5,
                           metadata=jsonized_metadata)


def _convert_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KiB", "MiB", "GiB")
    i = min(int(math.floor(math.log(size_bytes, 1024))), len(size_name) - 1)
    return f"{size_bytes / (1024 ** i):.2f} {size_name[i]}"


class SlamServicerClient:
    MAX_BAG_SIZE = 10000000000  # 10GB

    def __init__(self, grpc_context_manager: GrpcContextManager,
                 spike_host_monitor: Optional[SpikeHostMonitor] = None) -> None:
        self._grpc_context_manager = grpc_context_manager
        self._spike_host_monitor = spike_host_monitor

    @property
    def state(self) -> ConnectionState:
        return self._grpc_context_manager.state

    async def run(self) -> None:
        """Monitor spike host and change gRPC host when spike host is changed.
        When spike host monitor is not provided, this method does nothing.
        """
        if self._spike_host_monitor is None:
            await trio.sleep_forever()
        while True:
            addr = self._spike_host_monitor.spike_host_event.value
            self._grpc_context_manager.change_host(addr)
            await self._spike_host_monitor.spike_host_event.wait_transition()
            _logger.info(f"Spike servicer gRPC host has been changed to {addr}")

    def grpcclient(func):
        async def wrapper(self, *args, **kwargs):
            try:
                async with self._grpc_context_manager.open() as stub:
                    return await func(self, *args, **kwargs, stub=stub)
            except trio.TooSlowError:
                _logger.warning("gRPC timeout error")
                self._grpc_context_manager.trigger_reconnect()
                raise SlamProcedureCallError("gRPC timeout error") from None
            except ProtocolError as e:
                # NOTE: this is raised when the server closes the connection during the stream
                if str(e) == "Invalid input ConnectionInputs.SEND_DATA in state ConnectionState.CLOSED":
                    _logger.warning(f"gRPC protocol error: {e}")
                    self._grpc_context_manager.trigger_reconnect()
                    raise SlamProcedureCallError("h2 protocol error") from None
                # otherwise, it's unexpected
                raise
            except purerpc.RpcFailedError as e:
                _logger.warning(f"gRPC error: {e}")
                raise SlamProcedureCallError from None
        return wrapper

    @grpcclient
    async def build_single_mission_map(self, map_name: str, *, stub: SlamStub) -> None:
        """ request to build single mission map """
        _logger.debug(f"start to build single mission map: {map_name}")
        req = BuildSingleMissionMapRequest(map_name=map_name)
        with trio.fail_after(GRPC_TIMEOUT):
            await stub.BuildSingleMissionMap(req)
        _logger.info('Single mission map build request sent')

    @grpcclient
    async def map_list(self, *, stub: SlamStub) -> List[str]:
        """ request map list """
        _logger.debug("request map list")
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.ListMap(empty_pb2.Empty())
        _logger.debug(f"map list: {res.map_name}")
        return list(res.map_name)

    @grpcclient
    async def get_frontier(self, map_name: str, pose: Iterable[float], *, stub: SlamStub) -> List[List[float]]:
        coordinate = Coordinate(
            px=pose[0], py=pose[1], pz=pose[2],
            ox=pose[3], oy=pose[4], oz=pose[5], ow=pose[6]
        )
        req = GetFrontierRequest(map_name=map_name, pose=coordinate)
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.GetFrontier(req)
        poses = _pose_list_to_list(res.pose)
        _logger.debug(f"frontier: {poses}")
        return poses

    @grpcclient
    async def get_latest_map(self, *, stub: SlamStub) -> str:
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.GetLatestMap(empty_pb2.Empty())
        _logger.debug(f"latest map: {res.map_name}")
        return res.map_name

    @grpcclient
    async def check_explore_rights(self, ghost_id: str, *, stub: SlamStub) -> Tuple[bool, str]:
        req = CheckExploreRightsRequest(ghost_id=ghost_id)
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.CheckExploreRights(req)
        _logger.debug(f'check explore rights response success: {res.success}, token: {res.token}')
        return res.success, res.token

    @grpcclient
    async def get_spots(self, spot_names: str, *, stub: SlamStub) -> Dict[str, Dict[str, Any]]:
        def spot_to_dict(spot: Spot):
            return {
                "id": spot.name,
                "name": spot.name,
                "coordinate": SlamServicer.coordinate_to_string(spot.coordinate)
            }

        req = GetSpotsRequest(spot_names=spot_names)
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.GetSpots(req)
        return {spot.name: spot_to_dict(spot) for spot in res.spots}

    @grpcclient
    async def get_unwelcomed_area(self, *, stub: SlamStub) -> str:
        with trio.fail_after(GRPC_TIMEOUT):
            res = await stub.GetUnwelcomedArea(empty_pb2.Empty())
        return res.unwelcomed_area

    @grpcclient
    async def _upload_file(self, filepath: pathlib.Path, file_type: UploadFileType,
                           *,
                           metadata: Dict = {}, stub: SlamStub) -> None:
        """upload a file to the server, with metadata.
        this method just uploads a file.
        you need to implement specific file preparers to client and handlers to servicer.
        """
        @_as_async_generator
        @trio_async_generator
        async def gen(f) -> UploadFileRequest:
            information = _create_fileinformation(filepath, file_type, metadata)
            while True:
                d = await f.read(GRPC_STREAM_CHUNK_SIZE)
                with trio.fail_after(GRPC_TIMEOUT):
                    yield UploadFileRequest(data=d, information=information)

                information = None
                if len(d) == 0:
                    # send 0-byte datum at last, for confirmation of end
                    break

        _logger.info(f'Uploading {filepath.name} to nest')
        filesize = filepath.stat().st_size
        timeout = int(filesize / GRPC_STREAM_CHUNK_SIZE + 1) * GRPC_TIMEOUT

        try:
            async with await trio.open_file(filepath, 'rb') as f:
                start = trio.current_time()
                with trio.fail_after(timeout):
                    ret = await stub.UploadFile(gen(f))
                size = await f.tell()
                duration = trio.current_time() - start
                _logger.info(f'upload_file: uploaded {_convert_size(size)} '
                             f'in {duration:.2f} sec, '
                             f'{_convert_size(size / (duration))}/s')
        except (EnvironmentError, IndexError) as e:
            _logger.error(f'upload_file: file error {e}')
            raise SlamTransferError from None

        if ret.status != 'ok':
            _logger.error(f'upload_file: {ret.status} {ret.message}')
            raise SlamTransferError
        if ret.message:
            _logger.info(f'upload_file: {ret.message}')

    async def upload_accuracy_map(self, ghost_id: str) -> None:
        """Upload accuracy map to the nest.
        """
        # set metadata
        metadata = {}
        metadata['ghost_id'] = ghost_id

        # archive data
        # TODO: asynchronize
        tmp_file = data_directories.monitor / 'accuracy_map.zip'
        zip_archive(tmp_file, data_directories.monitor / 'accuracy_map')

        await self._upload_file(tmp_file, UploadFileType.ACCURACY_MAP, metadata=metadata)
        tmp_file.unlink()

    async def upload_rosbag(self, map_name: str, file_path: pathlib.Path) -> None:
        """Upload accuracy map to the nest.
        """
        file_size = get_file_size(file_path)
        if file_size > self.MAX_BAG_SIZE:
            try:
                bag_duration = await BagUtils.get_duration(file_path)
            finally:
                remove_file_if_exists(file_path)  # Anyway remove the file so that the error is not reproduced.
            raise RuntimeError(f"Too large bag file is tried to upload: {file_path.name}"
                               f", file size = {file_size} bytes, duration = {bag_duration} seconds.")
        await self._upload_file(file_path, UploadFileType.ROSBAG, metadata={'map_name': map_name})

    @grpcclient
    async def _download_file(self, archive_path: pathlib.Path,
                             request_type: DownloadFileType, request_metadata: Optional[Dict] = None,
                             *,
                             stub: SlamStub) -> Dict:
        """Download an archive file from spike.
        :archive_path: path to which the archive file is saved
        :request_type: request file type
        :request_metadata: request metadata (optional)
        :return: response metadata
        raise
            SlamTransferError
            grpc.RpcError
        """
        _logger.info(f'_download_file: downloading {request_type}')
        information: Optional[FileInformation] = None
        req = DownloadFileRequest(type=request_type.value)
        if request_metadata:
            req.metadata = json.dumps(request_metadata)
        try:
            async with await trio.open_file(archive_path, 'wb') as f:
                start = trio.current_time()
                async with aclosing(stub.DownloadFile(req)) as async_iterable:
                    async for message in _timed_async_generator(async_iterable, _GRPC_DOWNLOAD_TIMEOUT):
                        await f.write(message.data)
                        if message.WhichOneof('information_optional') == 'information':
                            information = message.information
                size = await f.tell()
                duration = trio.current_time() - start
                _logger.info(f'_download_file: downloaded {_convert_size(size)} '
                             f'in {duration:.2f} sec, '
                             f'{_convert_size(size / (duration))}/s')
        except (purerpc.NotFoundError, purerpc.InvalidArgumentError) as e:
            _logger.warning(f'_download_file: {e.status}')
            raise SlamTransferError from None

        if not verify_md5sum(archive_path, information.md5):
            _logger.error("_download_file: md5 does not match")
            raise SlamTransferError

        return json.loads(information.metadata)

    async def download_segmentation(self):
        try:
            archive_path = data_directories.tmp / 'temporary_segmentation'
            request_metadata = {
                'version': SEGMENTATION_VERSION,
                'include_png': True,
            }
            metadata = await self._download_file(archive_path, DownloadFileType.SEGMENTATION,
                                                 request_metadata=request_metadata)
            _logger.debug(f'download_segmentation metadata: {metadata}')

            unzip_archive(archive_path, data_directories.segmentation)
        except RuntimeError as e:
            _logger.error(f'download_segmentation error: {e}')
        finally:
            archive_path.unlink(missing_ok=True)
            # sync to be secure writing to the disk
            sync_to_disk()

    async def download_map(self, map_name: str, target_dir: pathlib.Path):
        try:
            archive_path = data_directories.tmp / 'temporary_map'
            request_metadata = {
                'map_name': map_name
            }
            response_metadata = await self._download_file(archive_path, DownloadFileType.MAP,
                                                          request_metadata=request_metadata)
            _logger.debug(f'download_map metadata: {response_metadata}')

            unzip_archive(archive_path, target_dir)
        except RuntimeError as e:
            _logger.error(f'download_map error: {e}')
        finally:
            archive_path.unlink(missing_ok=True)
            # sync to be secure writing to the disk
            sync_to_disk()


@asynccontextmanager
async def open_slam_servicer_client(host: Optional[str] = None, port: int = GRPC_PORT):
    """Open a connection to the slam service.
    When host is not specified, it will launch a periodical monitor to retrieve the host name.
    This takes a few seconds to yield the context.

    :param host: host name or IP address of the spike
    :param port: port number of the gRPC server
    e.g.
    async with open_slam_servicer_client() as client:
        async for _ in periodic(5):
            ret = await client.get_latest_map()
            print(ret)
    """
    async with trio.open_nursery() as nursery:
        spike_host_monitor = None
        if not host:
            spike_host_monitor = SpikeHostMonitor()
            nursery.start_soon(spike_host_monitor.run)
            host = await spike_host_monitor.get_addr()

        connection_manager = GrpcContextManager(host, port, SlamStub, concurrent_connections=1)
        nursery.start_soon(connection_manager.run)

        client = SlamServicerClient(connection_manager, spike_host_monitor)
        nursery.start_soon(client.run)
        yield client
        nursery.cancel_scope.cancel()
